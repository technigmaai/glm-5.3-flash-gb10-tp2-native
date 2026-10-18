// arx: one-shot all-reduce over RoCE for GB10, no NCCL.
//
// GB10's CPU and GPU share one coherent memory, so pinned host buffers are
// GPU buffers at full speed and the ConnectX can DMA into them without
// GPUDirect. Per all-reduce:
//   1. the GPU writes its partial into send[seq & 1] and publishes seq;
//   2. a CPU proxy sees seq, RDMA-writes the partial into every peer's
//      recv[seq & 1][rank], then writes seq into the peer's flag[rank]
//      (same QP, so the data is placed first);
//   3. the GPU waits for every peer's flag to reach seq and sums.
// Two buffers by parity are enough: a rank reaches seq + 2 only after every
// peer consumed seq + 1, which needed this rank's seq + 1 flag, which the
// proxy posts after this rank's seq data was read.
//
// The proxy posts every seq in order. The GPU can publish seq + 1 before the
// proxy has seen seq (finishing seq needs only the peers' data), and skipping
// seq would let peers accept seq + 1's flag over stale seq data.
//
// Every partial is split in half, one half per ConnectX root, each followed by
// its own flag on that root's QP: flag[src * 2 + root]. Rank r sends to
// r+1, r+2, ... in turn, so no receiver takes every sender at once.
//
// Ring mode is for boxes cabled in a ring with no switch. The ConnectX cannot
// forward RoCE for other boxes, so QPs go only to prev = r-1 and next = r+1,
// each over the two functions of the port facing that neighbour. A rank sends
// its partial to next, and at world 3 and 4 to prev as well. At world 4 the
// proxy forwards the partial from prev on to next. The relayed data lands in
// the same recv[seq & 1][src] and flag[src * 2 + root] as a direct write, so
// the GPU side is unchanged. At world 3 prev and next are the two other ranks,
// so every partial goes direct and nothing is relayed. World 2 sends to next
// only.
//
// The relay adds one reuse hazard. Rank m reads recv[par][m-1] to forward it
// to m+1, and m-1 writes seq + 2 there next. m-1 publishes seq + 2 after
// finishing seq + 1. Finishing needs m+1's seq + 1 partial. m+1 publishes that
// after finishing seq, and finishing seq needed the relayed data. The relay
// takes the partial's size from this rank's own publish of the same seq. The
// same chain keeps this rank from publishing seq + 2 before the relay.
//
// Two-shot (mesh only), for partials too big to send whole to every peer:
// allreduce2() runs two collectives in a row, each its own seq. Reduce-scatter:
// the proxy writes chunk j of this rank's partial into rank j's recv slot, and
// the GPU sums its own chunk from every rank (fp32, rank order, one rounding,
// so the bits match the one-shot sum). All-gather: the proxy writes the summed
// chunk into every peer's slot, and the GPU assembles the chunks in rank order.
// Each rank sends 2 (world - 1) / world partials' worth instead of world - 1.
// The scatter's blocks sum parts of send[par] that other blocks copied, so they
// wait for this rank's own publish as well as the peers' flags.
// Ctl::mode tells the proxy which pattern a seq uses.
//
// Every seq counter is 64-bit: a 32-bit one wraps after 2^32 collectives, and
// the wrapped value compares below the 64-bit flags.
// ARX_SEQ_START (debug, default 0, the same on every rank) starts the count
// elsewhere, so a test can cross 2^32 without running that long.
//
// Failing loudly: a proxy that stops (failed post, completion error) would
// leave every GPU spinning on flags that never come. The proxy writes a code
// into Ctl::err before it exits, and every GPU spin-wait polls that word and
// a %globaltimer deadline, ARX_SPIN_TIMEOUT_S (default 300 s, far above any
// legitimate lag between ranks such as a JIT compile on one; 0 is off), then
// traps with a one-line reason. Device printf can be lost with the trapped
// context, so the GPU also writes why into Ctl::gpu_err for the proxy to log.
// ARX_FAIL_AFTER=N (test hook, default off) makes the proxy fail at its Nth
// collective.
//
// vLLM form: prepare() returns this rank's connection details, the caller
// all-gathers them over its own group, connect() wires the QPs and starts the
// proxy, then allreduce(in, out) runs on the current stream and is
// CUDA-graph capturable. in/out are bf16 device tensors.
#include <cmath>
#include <arpa/inet.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define IBCK(x) do { if (!(x)) { fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__, #x, strerror(errno)); exit(1); } } while (0)

constexpr int kMaxWorld = 8;
constexpr int kMaxDev = 4;  // ring mode: 2 ports x 2 roots
constexpr size_t kMaxBytes = 512 << 10;  // one-shot partial: bf16 [32, 4096] is 256 KB
constexpr size_t kSendBytes = 4 << 20;   // per parity: a two-shot partial
constexpr size_t kSlotBytes = 1 << 20;   // per peer per parity: a one-shot partial or a two-shot chunk
enum : uint64_t { kFull = 0, kScatter = 1, kGather = 2 };
// Ctl::err, set by the proxy before it exits.
enum : uint64_t { kErrPost = 1, kErrCompletion = 2, kErrInjected = 3 };
// Ctl::gpu_err, set by a GPU spin-wait before it traps.
enum : uint64_t { kGpuSawErr = 1, kGpuTimeout = 2 };
// QP retry timeout, 4.096 us * 2^x per try. 14 (67 ms) gave up after ~0.5 s
// of link trouble; 18 (1.07 s, ~8.6 s over the 7 retries) rides out longer
// blips and stays well inside the GPU spin deadline.
constexpr int kQpTimeout = 18;

struct Ctl {                   // pinned; written by the GPU, read by the proxy
  volatile uint64_t seq;       // last published partial
  volatile uint64_t bytes[2];  // size of the partial in send[parity]
  volatile uint64_t mode[2];   // kFull, kScatter or kGather for send[parity]
  volatile uint64_t stop;
  volatile uint64_t done;      // last seq the GPU finished summing (for the watchdog)
  volatile uint64_t err;       // written by the proxy: why it stopped (kErr*), 0 while it runs
  volatile uint64_t gpu_err;   // written by the GPU: why it trapped (kGpu*)
};

struct PeerInfo {  // what rank r tells everyone about its QPs
  uint32_t qpn[kMaxWorld][2];  // per channel (see State), per root
  uint8_t gid[kMaxDev][16];    // per RDMA device
  uint32_t mtu[kMaxDev];       // per device: the port's active MTU (enum ibv_mtu)
  uint64_t recv_addr, flag_addr;
  uint32_t recv_rkey[kMaxDev], flag_rkey[kMaxDev];
  uint64_t seq_start;          // ARX_SEQ_START: every rank must start at the same seq
};

// ---- GPU side ---------------------------------------------------------------

__device__ __forceinline__ uint64_t ld_acquire_sys(const volatile uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint4 ld_cv(const void* p) {  // bypass caches: the NIC wrote it
  uint4 v;
  asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ uint64_t globaltimer() {  // ns, the same on every SM
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

struct Dev {
  __nv_bfloat16* send;           // [2][kSendBytes / 2]
  const __nv_bfloat16* recv;     // [2][world][kSlotBytes / 2]
  const volatile uint64_t* flag; // [2 * world]
  Ctl* ctl;
  unsigned long long* seq_dev;   // device: last completed seq
  unsigned* blocks;         // device [2]: blocks past the copy, blocks finished
  unsigned long long* go;        // device: seq whose peer data has all arrived
  int rank, world;
  uint64_t spin_ns;              // spin-wait deadline, 0 = wait forever
#ifdef ARX_TEST_HOOKS
  long long slow_copy_ns;        // two-shot copy block 1 sleeps this long first (test_ring.py --slow-copy-us)
#endif
};

// Out of line so the spin loops stay small.
__device__ __noinline__ void spin_fail(const Dev& d, uint64_t code, uint64_t seq, const char* what, int idx,
                                       uint64_t have) {
  d.ctl->gpu_err = code;
  __threadfence_system();
  printf("arx rank %d seq %llu: %s waiting for %s %d (have %llu, proxy err %llu), trapping\n", d.rank,
         (unsigned long long)seq, code == kGpuTimeout ? "timed out" : "relay failed", what, idx,
         (unsigned long long)have, (unsigned long long)d.ctl->err);
  __trap();
}

// Called on every turn of a spin-wait. Every 1024 turns it checks the proxy's
// error word and the deadline, so the hot loop stays a single load.
__device__ __forceinline__ void spin_tick(const Dev& d, unsigned& n, uint64_t& t0, uint64_t seq, const char* what,
                                          int idx, uint64_t have) {
  if (++n & 1023) return;
  if (ld_acquire_sys(&d.ctl->err)) spin_fail(d, kGpuSawErr, seq, what, idx, have);
  const uint64_t now = globaltimer();
  if (t0 == 0) t0 = now;  // the clock starts at the first check: short waits never read it
  else if (d.spin_ns && now - t0 > d.spin_ns) spin_fail(d, kGpuTimeout, seq, what, idx, have);
}

// Weight ranges the next kernels read; the kernel asks L2 for them before it waits.
constexpr int kMaxPrefetch = 4;
struct Prefetch {
  const uint8_t* ptr[kMaxPrefetch];
  long long bytes[kMaxPrefetch];
  int count;
};

// out = sum over ranks of in. in and out may alias.
__global__ void arx_kernel(Dev d, const __nv_bfloat16* __restrict__ in, __nv_bfloat16* out, int n, Prefetch pf) {
  const uint64_t seq = *d.seq_dev + 1;
  const int par = seq & 1;
  __nv_bfloat16* mine = d.send + (size_t)par * (kSendBytes / 2);
  const int stride = gridDim.x * blockDim.x * 8;
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8; i < n; i += stride)
    *reinterpret_cast<uint4*>(mine + i) = *reinterpret_cast<const uint4*>(in + i);
  __threadfence_system();
  __syncthreads();
  // Separate counters per phase: blocks that start late would otherwise add
  // their first count after early blocks added their second, and the
  // publish would be skipped or would go out before every block copied.
  if (threadIdx.x == 0 && atomicAdd(&d.blocks[0], 1) == gridDim.x - 1) {  // last block publishes
    d.blocks[0] = 0;
    d.ctl->mode[par] = kFull;
    d.ctl->bytes[par] = (uint64_t)n * 2;
    __threadfence_system();
    d.ctl->seq = seq;
  }
  // DRAM is idle while the peers' data is in flight: spread one bulk L2
  // prefetch per thread over each range.
  if (pf.count > 0) {
    const long long nthr = (long long)gridDim.x * blockDim.x, t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (int r = 0; r < pf.count; ++r) {
      const long long chunk = ((pf.bytes[r] + nthr - 1) / nthr + 15) & ~15LL, off = t * chunk;
      const long long len = min(chunk, pf.bytes[r] - off) & ~15LL;
      if (len > 0)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(pf.ptr[r] + off), "r"((unsigned)len));
    }
  }
  // Block 0 polls the pinned flags the NIC writes; the others wait on a word in
  // device memory, so only a few threads touch lines the NIC is filling.
  unsigned spins = 0;
  uint64_t t0 = 0;
  if (blockIdx.x == 0) {
    if (threadIdx.x < 2 * d.world && threadIdx.x / 2 != d.rank)
      for (uint64_t f; (f = ld_acquire_sys(d.flag + threadIdx.x)) < seq;)
        spin_tick(d, spins, t0, seq, "peer flag", threadIdx.x, f);
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicExch(d.go, (unsigned long long)seq); }
  } else {
    if (threadIdx.x == 0)
      while (true) {
        unsigned long long g;
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(g) : "l"(d.go) : "memory");
        if (g >= seq) break;
        spin_tick(d, spins, t0, seq, "go", blockIdx.x, g);
      }
    __syncthreads();
  }
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8; i < n; i += stride) {
    uint4 v[kMaxWorld];
#pragma unroll
    for (int src = 0; src < kMaxWorld; ++src)
      if (src < d.world)
        v[src] = src == d.rank ? *reinterpret_cast<const uint4*>(mine + i)
                               : ld_cv(d.recv + ((size_t)par * d.world + src) * (kSlotBytes / 2) + i);
    float acc[8] = {};
#pragma unroll
    for (int src = 0; src < kMaxWorld; ++src)  // rank order: every rank gets the same bits
      if (src < d.world) {
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v[src]);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(b[k]);
      }
    __align__(16) __nv_bfloat16 o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(o);
  }
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(&d.blocks[1], 1) == gridDim.x - 1) {  // last block out
    d.blocks[1] = 0;
    *d.seq_dev = seq;
    d.ctl->done = seq;
  }
}

// The two-shot kernels: copy n bf16 into send[par], publish them as `mode`,
// wait for every peer's flag, then leave the rest to the caller.
__device__ __forceinline__ uint64_t publish_and_wait(Dev d, const __nv_bfloat16* __restrict__ in, int n,
                                                     uint64_t mode) {
  const uint64_t seq = *d.seq_dev + 1;
  const int par = seq & 1;
  __nv_bfloat16* mine = d.send + (size_t)par * (kSendBytes / 2);
  const int stride = gridDim.x * blockDim.x * 8;
#ifdef ARX_TEST_HOOKS
  if (blockIdx.x == 1 && d.slow_copy_ns > 0) {
    if (threadIdx.x == 0) {
      unsigned long long t0, t;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      do {
        __nanosleep(1000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
      } while (t - t0 < (unsigned long long)d.slow_copy_ns);
    }
    __syncthreads();
  }
#endif
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8; i < n; i += stride)
    *reinterpret_cast<uint4*>(mine + i) = *reinterpret_cast<const uint4*>(in + i);
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(&d.blocks[0], 1) == gridDim.x - 1) {
    d.blocks[0] = 0;
    d.ctl->mode[par] = mode;
    d.ctl->bytes[par] = (uint64_t)n * 2;
    __threadfence_system();
    d.ctl->seq = seq;
  }
  // Block 0 also waits for this rank's own publish before it releases the
  // others. ctl->seq reaches seq only after the last block's count, i.e. after
  // every block copied its share of `in` into send[par]. The scatter then sums
  // this rank's chunk of send[par] with a stride that does not line up with
  // the copy's unless the chunk is a multiple of it, so its blocks read
  // elements other blocks copied. Peers' flags do not depend on that copy, so
  // without this wait a block could read last parity's data.
  unsigned spins = 0;
  uint64_t t0 = 0;
  if (blockIdx.x == 0) {
    if (threadIdx.x < 2 * d.world && threadIdx.x / 2 != d.rank)
      for (uint64_t f; (f = ld_acquire_sys(d.flag + threadIdx.x)) < seq;)
        spin_tick(d, spins, t0, seq, "peer flag", threadIdx.x, f);
    if (threadIdx.x == 0)
      for (uint64_t p; (p = ld_acquire_sys(&d.ctl->seq)) < seq;)
        spin_tick(d, spins, t0, seq, "own publish", 0, p);
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicExch(d.go, (unsigned long long)seq); }
  } else {
    if (threadIdx.x == 0)
      while (true) {
        unsigned long long g;
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(g) : "l"(d.go) : "memory");
        if (g >= seq) break;
        spin_tick(d, spins, t0, seq, "go", blockIdx.x, g);
      }
    __syncthreads();
  }
  return seq;
}

__device__ __forceinline__ void finish(Dev d, uint64_t seq) {
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(&d.blocks[1], 1) == gridDim.x - 1) {
    d.blocks[1] = 0;
    *d.seq_dev = seq;
    d.ctl->done = seq;
  }
}

// out[chunk] = this rank's chunk of the sum of every rank's in[n].
__global__ void arx_scatter_kernel(Dev d, const __nv_bfloat16* __restrict__ in, __nv_bfloat16* out, int n) {
  const uint64_t seq = publish_and_wait(d, in, n, kScatter);
  const int par = seq & 1, chunk = n / d.world;
  const __nv_bfloat16* mine = d.send + (size_t)par * (kSendBytes / 2) + (size_t)d.rank * chunk;
  const int stride = gridDim.x * blockDim.x * 8;
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8; i < chunk; i += stride) {
    uint4 v[kMaxWorld];
#pragma unroll
    for (int src = 0; src < kMaxWorld; ++src)
      if (src < d.world)
        v[src] = src == d.rank ? *reinterpret_cast<const uint4*>(mine + i)
                               : ld_cv(d.recv + ((size_t)par * d.world + src) * (kSlotBytes / 2) + i);
    float acc[8] = {};
#pragma unroll
    for (int src = 0; src < kMaxWorld; ++src)  // rank order, as the one-shot sum
      if (src < d.world) {
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v[src]);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(b[k]);
      }
    __align__(16) __nv_bfloat16 o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(o);
  }
  finish(d, seq);
}

// out[world * chunk] = every rank's chunk, in rank order.
__global__ void arx_gather_kernel(Dev d, const __nv_bfloat16* __restrict__ chunk_in, __nv_bfloat16* out, int chunk) {
  const uint64_t seq = publish_and_wait(d, chunk_in, chunk, kGather);
  const int par = seq & 1, n = chunk * d.world;
  const int stride = gridDim.x * blockDim.x * 8;
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8; i < n; i += stride) {
    const int src = i / chunk, j = i - src * chunk;
    *reinterpret_cast<uint4*>(out + i) =
        src == d.rank ? *reinterpret_cast<const uint4*>(chunk_in + j)
                      : ld_cv(d.recv + ((size_t)par * d.world + src) * (kSlotBytes / 2) + j);
  }
  finish(d, seq);
}

// ---- host side ---------------------------------------------------------------

namespace {
// A channel is one QP per root to one peer. Mesh: channel j goes to rank j
// over device r. Ring: channel 0 goes to prev over devices 0 and 1, channel 1
// to next over devices 2 and 3, and meets the peer's channel on the other side.
struct State {
  int rank = -1, world = 0, ndev = 0;
  bool ring = false;
  int prev = 0, next = 0;
  int gid_idx[kMaxDev] = {};
  uint8_t *send_h = nullptr, *recv_h = nullptr;
  uint64_t* flag_h = nullptr;
  Ctl* ctl = nullptr;
  ibv_context* ctx[kMaxDev] = {};
  ibv_pd* pd[kMaxDev] = {};
  ibv_mr *send_mr[kMaxDev] = {}, *recv_mr[kMaxDev] = {}, *flag_mr[kMaxDev] = {};
  ibv_cq* cq[kMaxDev] = {};
  ibv_qp* qp[kMaxWorld][2] = {};
  PeerInfo mine{};
  std::vector<PeerInfo> all;
  Dev dev{};
  bool connected = false;
  uint64_t seq_start = 0;   // ARX_SEQ_START
  double spin_s = 300;      // ARX_SPIN_TIMEOUT_S
  uint64_t fail_after = 0;  // ARX_FAIL_AFTER
};
State S;
std::atomic<bool> g_proxy_err{false};

void read_env() {
  const char* sp = std::getenv("ARX_SPIN_TIMEOUT_S");
  S.spin_s = sp && *sp ? std::atof(sp) : 300;
  // Finite and at most a day, so spin_s * 1e9 fits the uint64_t deadline.
  TORCH_CHECK(std::isfinite(S.spin_s) && S.spin_s >= 0 && S.spin_s <= 86400,
              "arx: ARX_SPIN_TIMEOUT_S must be between 0 and 86400, got ", S.spin_s);
  const char* fa = std::getenv("ARX_FAIL_AFTER");
  S.fail_after = fa && *fa ? std::strtoull(fa, nullptr, 0) : 0;
  // A deadline inside the QP's retry budget would trap while the NIC may still
  // land the write: warn, do not refuse.
  const double rdma_s = 4.096e-6 * (double)(1ull << kQpTimeout) * 8;  // first try + 7 retries
  if (S.spin_s > 0 && S.spin_s <= rdma_s)
    fprintf(stderr, "arx: ARX_SPIN_TIMEOUT_S %.1f s is within the RDMA retry budget of %.1f s\n", S.spin_s, rdma_s);
  if (S.fail_after) fprintf(stderr, "arx: ARX_FAIL_AFTER=%lu, the proxy will fail on purpose\n",
                            (unsigned long)S.fail_after);
}

// Records why the proxy stops, for the GPU spin-waits, then logs it.
void proxy_fail(uint64_t code, const char* why) {
  S.ctl->err = code;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  g_proxy_err = true;
  fprintf(stderr, "arx rank %d: proxy stopped (err %lu): %s; waiting GPUs will trap\n", S.rank, (unsigned long)code,
          why);
}

// Idle backoff (#55): keep the exact busy spin while there is work and for ARX_IDLE_SPIN_MS after it (default
// 200 ms), then nap ARX_IDLE_NAP_US (default 200 us) per loop turn. ARX_IDLE_NAP_US=0 keeps the old spin.
struct IdleNap {
  std::chrono::steady_clock::duration spin;
  std::chrono::microseconds nap;
  std::chrono::steady_clock::time_point last;
  IdleNap() {
    const char* s = std::getenv("ARX_IDLE_SPIN_MS");
    const char* n = std::getenv("ARX_IDLE_NAP_US");
    spin = std::chrono::milliseconds(s ? std::atol(s) : 200);
    nap = std::chrono::microseconds(n ? std::atol(n) : 200);
    last = std::chrono::steady_clock::now();
    fprintf(stderr, "arx idle nap: spin %ld ms, nap %ld us%s\n", (long)(s ? std::atol(s) : 200), (long)nap.count(),
            nap.count() <= 0 ? " (disabled = old spin)" : "");
  }
  void busy() { last = std::chrono::steady_clock::now(); }
  void idle() {
    if (nap.count() <= 0) return;
    if (std::chrono::steady_clock::now() - last > spin) std::this_thread::sleep_for(nap);
  }
};

int chan_peer(int c) { return S.ring ? (c ? S.next : S.prev) : c; }
int chan_dev(int c, int r) { return S.ring ? c * 2 + r : r; }         // device here
int chan_rdev(int c, int r) { return S.ring ? (1 - c) * 2 + r : r; }  // device at the peer

// One write of [laddr, laddr + len) to the peer's recv at roff, then seq into
// its flag slot, on channel c's root r QP.
bool post(int c, int r, uint64_t laddr, uint32_t lkey, uint32_t len, uint64_t roff, int slot, uint64_t seq,
          uint64_t& posted) {
  const PeerInfo& p = S.all[chan_peer(c)];
  const int rd = chan_rdev(c, r);
  ibv_sge sg{laddr, len, lkey};
  ibv_send_wr w{}, f{}, *bad;
  w.opcode = IBV_WR_RDMA_WRITE; w.sg_list = &sg; w.num_sge = 1;
  w.wr.rdma.remote_addr = p.recv_addr + roff;
  w.wr.rdma.rkey = p.recv_rkey[rd];
  ibv_sge fs{(uint64_t)&seq, 8, 0};
  f.opcode = IBV_WR_RDMA_WRITE; f.sg_list = &fs; f.num_sge = 1; f.send_flags = IBV_SEND_INLINE;
  f.wr.rdma.remote_addr = p.flag_addr + slot * 8; f.wr.rdma.rkey = p.flag_rkey[rd];
  // Unsignalled WRs are reclaimed only by a later signalled one on the
  // same QP, so the count is per QP.
  if ((++posted & 31) == 0) f.send_flags |= IBV_SEND_SIGNALED;
  w.next = &f;
  if (int e = ibv_post_send(S.qp[c][r], &w, &bad)) {
    fprintf(stderr, "arx rank %d: post to %d root %d failed: %s\n", S.rank, chan_peer(c), r, strerror(e));
    return false;
  }
  return true;
}

// Error completions arrive even for unsignalled WRs; any one means a QP is in
// the error state and no later write on it will land.
bool poll_cqs() {
  for (int r = 0; r < S.ndev; ++r) {
    ibv_wc wc[16];
    const int n = ibv_poll_cq(S.cq[r], 16, wc);
    for (int i = 0; i < n; ++i)
      if (wc[i].status != IBV_WC_SUCCESS) {
        char why[128];
        snprintf(why, sizeof why, "completion error %d (%s) on device %d", wc[i].status,
                 ibv_wc_status_str(wc[i].status), r);
        proxy_fail(kErrCompletion, why);
        return false;
      }
  }
  return true;
}

void proxy_loop() {
  // Every counter starts at ARX_SEQ_START, as seq_dev and Ctl::seq do: the
  // proxy posts ctl->seq's successors in order.
  const uint64_t start = S.seq_start;
  uint64_t seq = start, posted[kMaxWorld][2] = {};
  const int rank = S.rank, world = S.world;
  auto last_move = std::chrono::steady_clock::now();
  uint64_t last_done = start;
  bool reported = false, gpu_reported = false;
  // Ring at world 4: the last seq relayed from prev to next, per root, and
  // the size of this rank's own partial for each parity.
  uint64_t relayed[2] = {start, start}, own_bytes[2] = {};
  const bool relay = S.ring && world == 4;
  const volatile uint64_t* flags = S.flag_h;
  IdleNap nap;
  while (!S.ctl->stop) {
    for (int r = 0; relay && r < 2; ++r)
      while (relayed[r] < seq && flags[S.prev * 2 + r] > relayed[r]) {
        std::atomic_thread_fence(std::memory_order_acquire);
        const uint64_t s = ++relayed[r];
        const int par = s & 1;
        const uint32_t bytes = (uint32_t)own_bytes[par], half = (bytes / 2 + 15) & ~15u;
        const uint32_t off = r ? half : 0, len = r ? bytes - half : half;
        const uint64_t at = ((uint64_t)par * world + S.prev) * kSlotBytes + off;
        if (!post(1, r, (uint64_t)(S.recv_h + at), S.recv_mr[chan_dev(1, r)]->lkey, len, at, S.prev * 2 + r, s,
                  posted[1][r])) {
          proxy_fail(kErrPost, "relay post failed");
          return;
        }
      }
    if (S.ctl->seq <= seq) {
      // Idle: still drain the CQs, or an error on the last posted seq would
      // go unseen until the next publish, which never comes.
      if (!poll_cqs()) return;
      if (S.ctl->gpu_err && !gpu_reported) {
        gpu_reported = true;
        fprintf(stderr, "arx rank %d: GPU trapped in a spin-wait (gpu_err %lu: %s), published %lu posted %lu\n", rank,
                (unsigned long)S.ctl->gpu_err, S.ctl->gpu_err == kGpuTimeout ? "timeout" : "saw proxy error",
                (unsigned long)S.ctl->seq, (unsigned long)seq);
      }
      // Watchdog: published work that has not completed for 2 s is a stall,
      // and so is a peer whose flags are ahead of what this rank posted.
      // Print what this rank sent and what it has heard from each peer.
      const uint64_t done = S.ctl->done;
      const auto now = std::chrono::steady_clock::now();
      bool behind = false;
      for (int j = 0; j < world; ++j)
        if (j != rank) behind |= S.flag_h[j * 2] > seq;
      if (done != last_done || (done >= seq && !behind)) {
        last_done = done; last_move = now; reported = false;
      }
      else if (!reported && now - last_move > std::chrono::seconds(2)) {
        reported = true;
        char buf[512];
        int o = snprintf(buf, sizeof buf, "arx rank %d STALL: published %lu posted %lu done %lu; flags from peers:",
                         rank, (unsigned long)S.ctl->seq, (unsigned long)seq, (unsigned long)done);
        for (int j = 0; j < world; ++j)
          if (j != rank)
            o += snprintf(buf + o, sizeof buf - o, " r%d=%lu/%lu", j, (unsigned long)S.flag_h[j * 2],
                          (unsigned long)S.flag_h[j * 2 + 1]);
        if (relay)
          o += snprintf(buf + o, sizeof buf - o, "; relayed %lu/%lu", (unsigned long)relayed[0],
                        (unsigned long)relayed[1]);
        fprintf(stderr, "%s\n", buf);
      }
      if (relay && (relayed[0] < seq || relayed[1] < seq)) nap.busy(); else nap.idle();
      continue;
    }
    nap.busy();
    ++seq;  // every seq, in order
    if (S.fail_after && seq == S.fail_after) {  // ARX_FAIL_AFTER: test hook
      char why[96];
      snprintf(why, sizeof why, "injected failure at seq %lu (ARX_FAIL_AFTER)", (unsigned long)seq);
      proxy_fail(kErrInjected, why);
      return;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const int par = seq & 1;
    const uint64_t mode = S.ctl->mode[par];
    const uint32_t total = (uint32_t)S.ctl->bytes[par];
    // A scatter sends each peer its own chunk; full and gather send everything.
    const uint32_t bytes = mode == kScatter ? total / world : total;
    const uint32_t half = (bytes / 2 + 15) & ~15u;  // root 0 takes the first half
    own_bytes[par] = bytes;
    // Mesh: channel j is rank j. Ring: next, then prev unless prev is next.
    const int nchan = S.ring ? (world == 2 ? 1 : 2) : world - 1;
    for (int step = 1; step <= nchan; ++step) {
      const int c = S.ring ? 2 - step : (rank + step) % world;
      const uint64_t base = mode == kScatter ? (uint64_t)chan_peer(c) * bytes : 0;
      for (int r = 0; r < 2; ++r) {
        const uint32_t off = r ? half : 0, len = r ? bytes - half : half;
        if (!post(c, r, (uint64_t)(S.send_h + par * kSendBytes + base + off), S.send_mr[chan_dev(c, r)]->lkey, len,
                  ((uint64_t)par * world + rank) * kSlotBytes + off, rank * 2 + r, seq, posted[c][r])) {
          proxy_fail(kErrPost, "post failed");
          return;
        }
      }
    }
    if (!poll_cqs()) return;
  }
}
}  // namespace

// Allocates buffers, opens the RDMA devices and creates the QPs. Returns this
// rank's PeerInfo for the caller to all-gather. Mesh: devs are root 0 and root
// 1. Ring: the port facing prev (root 0, root 1), then the port facing next.
// gids: each device's GID index.
py::bytes arx_prepare(int64_t rank, int64_t world, std::vector<std::string> devs, std::vector<int64_t> gids,
                      bool ring) {
  TORCH_CHECK(S.rank < 0, "arx is already prepared in this process");
  TORCH_CHECK(world >= 2 && world <= kMaxWorld);
  TORCH_CHECK(!ring || (world >= 2 && world <= 4), "arx ring mode supports world 2 to 4, not ", world);
  S.ndev = ring ? 4 : 2;
  TORCH_CHECK((int)devs.size() == S.ndev && (int)gids.size() == S.ndev, "arx: expected ", S.ndev, " devices");
  read_env();
  S.rank = rank; S.world = world; S.ring = ring;
  S.prev = (rank + world - 1) % world; S.next = (rank + 1) % world;
  const char* start = std::getenv("ARX_SEQ_START");
  S.seq_start = start && *start ? std::strtoull(start, nullptr, 0) : 0;
  // Leave room above the start: a value near 2^64 would wrap seq_dev + 1 to 0 while the proxy stays put.
  TORCH_CHECK(S.seq_start < (1ull << 63), "arx: ARX_SEQ_START must be below 2^63, got ", S.seq_start);
  CK(cudaHostAlloc(&S.send_h, 2 * kSendBytes, cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.recv_h, 2 * world * kSlotBytes, cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.flag_h, 2 * kMaxWorld * sizeof(uint64_t), cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.ctl, sizeof(Ctl), cudaHostAllocMapped));
  // Peer flags start at 0, below every seq from seq_start + 1 on.
  memset(S.flag_h, 0, 2 * kMaxWorld * sizeof(uint64_t));
  memset((void*)S.ctl, 0, sizeof(Ctl));
  S.ctl->seq = S.seq_start;
  S.ctl->done = S.seq_start;
  S.mine.seq_start = S.seq_start;
  int ndev;
  ibv_device** list = ibv_get_device_list(&ndev);
  const int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  for (int d = 0; d < S.ndev; ++d) {
    S.gid_idx[d] = gids[d];
    for (int i = 0; i < ndev; ++i)
      if (devs[d] == ibv_get_device_name(list[i])) S.ctx[d] = ibv_open_device(list[i]);
    TORCH_CHECK(S.ctx[d], "arx: no RDMA device ", devs[d]);
    S.pd[d] = ibv_alloc_pd(S.ctx[d]); IBCK(S.pd[d]);
    S.send_mr[d] = ibv_reg_mr(S.pd[d], S.send_h, 2 * kSendBytes, acc); IBCK(S.send_mr[d]);
    S.recv_mr[d] = ibv_reg_mr(S.pd[d], S.recv_h, 2 * world * kSlotBytes, acc); IBCK(S.recv_mr[d]);
    S.flag_mr[d] = ibv_reg_mr(S.pd[d], S.flag_h, 2 * kMaxWorld * sizeof(uint64_t), acc); IBCK(S.flag_mr[d]);
    S.cq[d] = ibv_create_cq(S.ctx[d], 4096, nullptr, nullptr, 0); IBCK(S.cq[d]);
    ibv_gid gid;
    IBCK(ibv_query_gid(S.ctx[d], 1, S.gid_idx[d], &gid) == 0);
    memcpy(S.mine.gid[d], gid.raw, 16);
    ibv_port_attr port{};
    IBCK(ibv_query_port(S.ctx[d], 1, &port) == 0);
    S.mine.mtu[d] = port.active_mtu;
    S.mine.recv_rkey[d] = S.recv_mr[d]->rkey;
    S.mine.flag_rkey[d] = S.flag_mr[d]->rkey;
  }
  ibv_free_device_list(list);
  for (int c = 0; c < (ring ? 2 : world); ++c)
    for (int r = 0; r < 2; ++r) {
      if (!ring && c == rank) continue;
      const int d = chan_dev(c, r);
      ibv_qp_init_attr ia{};
      ia.send_cq = S.cq[d]; ia.recv_cq = S.cq[d]; ia.qp_type = IBV_QPT_RC;
      ia.cap.max_send_wr = 1024; ia.cap.max_recv_wr = 1; ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1;
      ia.cap.max_inline_data = 64;
      S.qp[c][r] = ibv_create_qp(S.pd[d], &ia); IBCK(S.qp[c][r]);
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = 1; a.qp_access_flags = acc;
      IBCK(ibv_modify_qp(S.qp[c][r], &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0);
      S.mine.qpn[c][r] = S.qp[c][r]->qp_num;
    }
  S.mine.recv_addr = (uint64_t)S.recv_h;
  S.mine.flag_addr = (uint64_t)S.flag_h;
  return py::bytes(reinterpret_cast<const char*>(&S.mine), sizeof(PeerInfo));
}

// Takes every rank's PeerInfo in rank order, brings the QPs up, starts the
// proxy. The caller must barrier its group before the first allreduce.
void arx_connect(std::vector<std::string> infos) {
  TORCH_CHECK(S.rank >= 0 && !S.connected && (int)infos.size() == S.world);
  S.all.resize(S.world);
  for (int j = 0; j < S.world; ++j) {
    TORCH_CHECK(infos[j].size() == sizeof(PeerInfo), "arx: PeerInfo size mismatch");
    memcpy(&S.all[j], infos[j].data(), sizeof(PeerInfo));
    TORCH_CHECK(S.all[j].seq_start == S.seq_start, "arx: rank ", j, " has ARX_SEQ_START ", S.all[j].seq_start,
                ", this rank ", S.seq_start);
  }
  for (int c = 0; c < (S.ring ? 2 : S.world); ++c)
    for (int r = 0; r < 2; ++r) {
      if (!S.ring && c == S.rank) continue;
      const PeerInfo& p = S.all[chan_peer(c)];
      const int d = chan_dev(c, r), rd = chan_rdev(c, r);
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_RTR; a.path_mtu = (ibv_mtu)std::min(S.mine.mtu[d], p.mtu[rd]);
      a.dest_qp_num = S.ring ? p.qpn[1 - c][r] : p.qpn[S.rank][r]; a.rq_psn = 0;
      a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
      a.ah_attr.is_global = 1; a.ah_attr.port_num = 1; a.ah_attr.grh.hop_limit = 1;
      a.ah_attr.grh.sgid_index = S.gid_idx[d];
      memcpy(a.ah_attr.grh.dgid.raw, p.gid[rd], 16);
      IBCK(ibv_modify_qp(S.qp[c][r], &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                             IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0);
      if (S.ring || c == (S.rank + 1) % S.world)
        fprintf(stderr, "arx rank %d%s root %d: path MTU %d bytes (ours %d, peer %d)\n", S.rank,
                S.ring ? (c ? " next" : " prev") : "", r, 128 << a.path_mtu, 128 << S.mine.mtu[d], 128 << p.mtu[rd]);
      a = {};
      a.qp_state = IBV_QPS_RTS; a.timeout = kQpTimeout; a.retry_cnt = 7; a.rnr_retry = 7; a.sq_psn = 0;
      a.max_rd_atomic = 1;
      IBCK(ibv_modify_qp(S.qp[c][r], &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                             IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0);
    }
  Dev& d = S.dev;
  d.send = (__nv_bfloat16*)S.send_h; d.recv = (const __nv_bfloat16*)S.recv_h; d.flag = S.flag_h; d.ctl = S.ctl;
  d.rank = S.rank; d.world = S.world;
#ifdef ARX_TEST_HOOKS
  const char* slow = std::getenv("ARX_TEST_SLOW_COPY_US");
  d.slow_copy_ns = slow ? std::atoll(slow) * 1000 : 0;
  fprintf(stderr, "arx rank %d: test build, two-shot copy block 1 sleeps %lld us\n", S.rank, d.slow_copy_ns / 1000);
#endif
  d.spin_ns = (uint64_t)(S.spin_s * 1e9);
  // seq_dev and go start at seq_start, so the first kernel's seq is the
  // successor the proxy waits for.
  const unsigned long long start = S.seq_start;
  CK(cudaMalloc(&d.seq_dev, 8)); CK(cudaMalloc(&d.blocks, 8)); CK(cudaMalloc(&d.go, 8));
  CK(cudaMemcpy(d.seq_dev, &start, 8, cudaMemcpyHostToDevice)); CK(cudaMemset(d.blocks, 0, 8));
  CK(cudaMemcpy(d.go, &start, 8, cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());
  std::thread(proxy_loop).detach();
  S.connected = true;
}

void arx_allreduce(torch::Tensor in, torch::Tensor out, std::vector<int64_t> pf_ptrs, std::vector<int64_t> pf_bytes) {
  TORCH_CHECK(S.connected, "arx is not connected");
  TORCH_CHECK(!g_proxy_err, "arx proxy failed");
  TORCH_CHECK(in.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 && in.is_contiguous() &&
              out.is_contiguous() && in.numel() == out.numel());
  const int n = in.numel();
  TORCH_CHECK(n % 8 == 0 && (size_t)n * 2 <= kMaxBytes);
  const int threads = 256, blocks = std::max(1, std::min(8, n / (threads * 8)));
  Prefetch pf{};
  pf.count = (int)std::min<size_t>(pf_ptrs.size(), kMaxPrefetch);
  for (int r = 0; r < pf.count; ++r) {
    pf.ptr[r] = reinterpret_cast<const uint8_t*>(pf_ptrs[r]);
    pf.bytes[r] = pf_bytes[r];
  }
  arx_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      S.dev, (const __nv_bfloat16*)in.data_ptr(), (__nv_bfloat16*)out.data_ptr(), n, pf);
}

int64_t arx_max_bytes() { return kMaxBytes; }

// Two-shot all-reduce (mesh only): a reduce-scatter then an all-gather. n must
// split into world chunks of whole 16-byte vectors.
void arx_allreduce2(torch::Tensor in, torch::Tensor out) {
  TORCH_CHECK(S.connected, "arx is not connected");
  TORCH_CHECK(!g_proxy_err, "arx proxy failed");
  TORCH_CHECK(!S.ring, "arx two-shot is mesh only");
  TORCH_CHECK(in.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 && in.is_contiguous() &&
              out.is_contiguous() && in.numel() == out.numel());
  const int n = in.numel(), chunk = n / S.world;
  TORCH_CHECK(n % (8 * S.world) == 0 && (size_t)n * 2 <= kSendBytes && (size_t)chunk * 2 <= kSlotBytes);
  auto mid = torch::empty({chunk}, in.options());
  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  arx_scatter_kernel<<<std::max(1, std::min(8, chunk / (threads * 8))), threads, 0, stream>>>(
      S.dev, (const __nv_bfloat16*)in.data_ptr(), (__nv_bfloat16*)mid.data_ptr(), n);
  arx_gather_kernel<<<std::max(1, std::min(8, n / (threads * 8))), threads, 0, stream>>>(
      S.dev, (const __nv_bfloat16*)mid.data_ptr(), (__nv_bfloat16*)out.data_ptr(), chunk);
}

int64_t arx_max_bytes2() { return S.ring ? 0 : (int64_t)std::min(kSendBytes, kSlotBytes * S.world); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &arx_prepare);
  m.def("connect", &arx_connect);
  m.def("allreduce", &arx_allreduce);
  m.def("max_bytes", &arx_max_bytes);
  m.def("allreduce2", &arx_allreduce2);
  m.def("max_bytes2", &arx_max_bytes2);
}
