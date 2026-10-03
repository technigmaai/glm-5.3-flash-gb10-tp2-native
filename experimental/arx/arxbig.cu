// arxbig: prefill-sized all-gather and reduce-scatter over RoCE for GB10, no NCCL.
//
// Same transport as arx_vllm.cu (pinned host buffers are GPU buffers at full
// speed on GB10, and the ConnectX DMAs them without GPUDirect; a CPU proxy
// posts RDMA writes when the GPU publishes a sequence number), sized for
// sequence-parallel prefill: tens of MB per rank per call.
//
// all_gather: the GPU copies this rank's slice into its slot of out[seq % 3];
// the proxy writes that slot into the same place in every peer's out[seq % 3],
// then a flag; the GPU waits for every peer's flags. The result is a view of
// out[seq % 3], valid until three more all-gathers have run.
// reduce_scatter: the GPU copies its full input into send[seq & 1]; the proxy
// writes chunk j into peer j's recv[seq & 1][rank], then a flag; the GPU waits
// and sums the world chunks for this rank in rank order (fp32).
//
// Every write is split in half, one half per ConnectX root, each followed by
// its own flag on that root's QP. Rank r sends to r+1, r+2, ... in turn.
// Reuse is safe for the same reason as in arx: a peer can only reach seq + 2
// (or + 3) after this rank posted seq + 1, which it does after its stream
// finished with seq.
//
// Ring mode (see arx_vllm.cu): QPs go only to prev and next, one channel
// each. Data for the rank two away goes half each way round the ring. Of each
// root's share of a piece, the first half travels next-ward and the second
// prev-ward. The rank in between forwards each piece as soon as its flag
// arrives.
// all_gather: a rank sends its whole slice to prev and to next. The rank
// forwards the next-ward half of prev's slice to next and the prev-ward half
// of next's to prev, from and to the same place in out[seq % 3]. Each port
// carries 1.5 slices out per call. Relaying whole slices one way would put 2
// on one port.
// reduce_scatter: chunk r+1 goes to next, chunk r-1 to prev, and chunk r+2
// half to each. Each neighbour stages its half in its own recv[seq & 1][self],
// a slot a rank never otherwise uses, and forwards it into recv[seq & 1][r]
// at r+2. Each port carries 2 slices out, the least possible when relays carry
// raw data. The two relays through a rank carry opposite halves of each piece,
// so they share the staging slot.
// World 2 sends half of everything each way and relays nothing.
// Ring flags are per (source, arrival channel, root), because data from rank
// r+2 arrives on both channels. Dev::mask names the flags the GPU waits on.
// Staged chunks have their own flags for the proxy. A relay takes the piece's
// bounds from this rank's own publish of the same seq.
// Reuse with relays: rank m reads out[seq % 3] or its staging slot to forward
// seq. The sender writes there again at seq + 3 or seq + 2, after finishing
// seq + 1. Finishing seq + 1 needs data from m's other neighbour. That
// neighbour published it after finishing seq, and finishing seq needed m's
// relay. The same chain holds m's own seq + 2 publish, and its new piece
// bounds, behind the relay. At the final destination the mesh argument holds
// as is. Relayed data for seq + 2 left its source after the source finished
// seq + 1, and that needed the destination's seq + 1 data.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define IBCK(x) do { if (!(x)) { fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__, #x, strerror(errno)); exit(1); } } while (0)

namespace {

constexpr int kMaxWorld = 8, kAgSlots = 3, kChunks = 8, kRing = 256;
constexpr int kMaxDev = 4;  // ring mode: 2 ports x 2 roots
// Mesh: [op][src][root]. Ring: [op][src][channel][root], then [channel][root]
// for staged reduce-scatter chunks.
constexpr int kFlagWords = 2 * kMaxWorld * 4 + 4;
// kRsPiece: one row range of one destination's chunk, published by a producer
// kernel as soon as it has written those rows (see finalize_rs_kernel).
enum Op : uint64_t { kAllGather = 1, kReduceScatter = 2, kRsPiece = 3 };

// Each op goes out as kChunks pieces so the network starts on piece 0 while
// the GPU still copies piece 1. A peer's flag for op seq, piece c, is
// seq * 16 + c + 1: monotonic, and it covers every earlier piece on that QP.
// Producers publish in any order, so each ring entry carries its own index + 1
// (tag), written last; the proxy consumes entries strictly in index order.
struct Work {
  uint64_t op, seq, chunk, slice, lo, hi;
  volatile uint64_t tag;
};
struct Ctl {                    // pinned; written by the GPU, read by the proxy
  Work ring[kRing];
  volatile uint64_t stop;
};

struct PeerInfo {
  uint32_t qpn[kMaxWorld][2];  // per channel (see State), per root
  uint8_t gid[kMaxDev][16];
  uint32_t mtu[kMaxDev];  // per device: the port's active MTU (enum ibv_mtu)
  uint64_t ag_addr, rs_addr, flag_addr;
  uint32_t ag_rkey[kMaxDev], rs_rkey[kMaxDev], flag_rkey[kMaxDev];
};

struct Dev {
  uint8_t* ag;                    // [kAgSlots][slot_bytes]
  uint8_t* rs_send;               // [2][slot_bytes]
  uint8_t* rs_recv;               // [2][world][slot_bytes / world]
  const volatile uint64_t* flag;  // [2 ops][nslot], see kFlagWords
  Ctl* ctl;
  unsigned* blocks;               // [kChunks * kMaxWorld] blocks done with each piece
  unsigned long long* work_next;  // next ring index to hand out
  unsigned long long* ready;      // highest flag_of(seq, piece) whose data has fully arrived
  int rank, world;
  int nslot;      // flag words per op
  uint32_t mask;  // the words of an op that carry data for this rank
  size_t slot_bytes;
};

__device__ __forceinline__ uint64_t ld_acquire_sys(const volatile uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint4 ld_cv(const void* p) {
  uint4 v;
  asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

#ifndef RS_CACHED
#define RS_CACHED 1
#endif
__device__ __forceinline__ uint4 ld_plain(const void* p) { return *reinterpret_cast<const uint4*>(p); }
#if RS_CACHED
#define RS_LOAD ld_plain
#else
#define RS_LOAD ld_cv
#endif

__device__ __forceinline__ uint64_t flag_of(uint64_t seq, int c) { return seq * 16 + c + 1; }

// Piece c of one rank's slice: [lo, hi) bytes, 16-aligned.
__device__ __forceinline__ void piece(size_t slice, int c, size_t& lo, size_t& hi) {
  const size_t step = (slice / kChunks + 15) & ~size_t(15);
  lo = min(slice, step * c);
  hi = c == kChunks - 1 ? slice : min(slice, step * (c + 1));
}

__device__ void publish(const Dev& d, uint64_t op, uint64_t seq, uint64_t chunk, uint64_t slice, uint64_t lo,
                        uint64_t hi) {
  const unsigned long long idx = atomicAdd(d.work_next, 1ull);
  Work& it = d.ctl->ring[idx % kRing];
  it.op = op; it.seq = seq; it.chunk = chunk; it.slice = slice; it.lo = lo; it.hi = hi;
  __threadfence_system();
  it.tag = idx + 1;
}

// Copy piece c of every slice in `slices` (count of them, stride `slice`) from
// src to dst; the last block to finish a piece publishes it.
__device__ void copy_publish(const Dev& d, const uint8_t* src, uint8_t* dst, size_t slice, int slices, uint64_t op,
                             uint64_t seq) {
  for (int c = 0; c < kChunks; ++c) {
    size_t lo, hi;
    piece(slice, c, lo, hi);
    const size_t len = hi - lo, total = len * slices;
    const size_t stride = (size_t)gridDim.x * blockDim.x * 16;
    if (src != dst)
      for (size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 16; i < total; i += stride) {
        const size_t s_ = i / len, o = s_ * slice + lo + i % len;
        *reinterpret_cast<uint4*>(dst + o) = *reinterpret_cast<const uint4*>(src + o);
      }
    __threadfence_system();
    __syncthreads();
    // One counter per piece: blocks do not wait for each other between pieces.
    if (threadIdx.x == 0 && atomicAdd(&d.blocks[c], 1) == gridDim.x - 1) {
      d.blocks[c] = 0;
      publish(d, op, seq, c, slice, lo, hi);
    }
  }
}

// Wait until every flag in d.mask for op reaches seq, piece c. Only
// block 0 reads the pinned flags the NIC writes; it relays progress through a
// device word, since many blocks polling those lines slows the NIC's writes.
__device__ void wait_piece(const Dev& d, uint64_t op, uint64_t seq, int c) {
  const uint64_t want = flag_of(seq, c);
  if (blockIdx.x == 0) {
    const volatile uint64_t* f = d.flag + (op == kAllGather ? 0 : d.nslot);
    if (threadIdx.x < d.nslot && (d.mask >> threadIdx.x & 1))
      while (ld_acquire_sys(f + threadIdx.x) < want) {}
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicMax(d.ready, (unsigned long long)want); }
  } else {
    if (threadIdx.x == 0)
      while (true) {
        unsigned long long g;
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(g) : "l"(d.ready) : "memory");
        if (g >= want) break;
      }
    __syncthreads();
  }
}

__global__ void ag_kernel(Dev d, const uint8_t* __restrict__ in, size_t slice, uint64_t seq) {
  uint8_t* out = d.ag + (seq % kAgSlots) * d.slot_bytes;
  copy_publish(d, in, out + d.rank * slice, slice, 1, kAllGather, seq);
  wait_piece(d, kAllGather, seq, kChunks - 1);
}

// out [slice / 2] bf16 = sum over ranks of their chunk `rank` of in.
__global__ void rs_kernel(Dev d, const uint8_t* __restrict__ in, size_t slice, __nv_bfloat16* __restrict__ out,
                          uint64_t seq) {
  const int par = seq & 1;
  uint8_t* send = d.rs_send + par * d.slot_bytes;
  copy_publish(d, in, send, slice, d.world, kReduceScatter, seq);
  const uint8_t* recv = d.rs_recv + (size_t)par * d.slot_bytes;
  const size_t stride = (size_t)gridDim.x * blockDim.x * 16;
  for (int c = 0; c < kChunks; ++c) {
    size_t lo, hi;
    piece(slice, c, lo, hi);
    wait_piece(d, kReduceScatter, seq, c);
    for (size_t i = lo + ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 16; i < hi; i += stride) {
      float acc[8] = {};
      for (int src = 0; src < d.world; ++src) {  // rank order: every rank rounds the same way
        const uint4 v = src == d.rank ? *reinterpret_cast<const uint4*>(send + (size_t)d.rank * slice + i)
                                      : RS_LOAD(recv + (size_t)src * slice + i);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(b[k]);
      }
      __align__(16) __nv_bfloat16 o[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16(acc[k]);
      *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(out) + i) = *reinterpret_cast<const uint4*>(o);
    }
  }
}


// Rows [lo, hi) of piece k of a destination's Rr-row chunk.
__device__ __host__ __forceinline__ void row_piece(int64_t Rr, int k, int64_t& lo, int64_t& hi) {
  lo = Rr * k / kChunks;
  hi = Rr * (k + 1) / kChunks;
}

// MoE output straight into the reduce-scatter send buffer:
//   row t = shared[t] + sum_k w[t, k] * y[pos[t, k]]  (fp32, rounded once),
// zero for padding rows t >= T. Blocks take rows piece-major (piece 0 of
// every destination first) and the last block of a piece publishes it, so
// the network starts while later rows are still being computed.
// y8s != nullptr: y holds e4m3 bytes with one fp32 scale per (row, 128 columns).
__global__ void finalize_rs_kernel(Dev d, const void* __restrict__ yv, const float* __restrict__ y8s,
                                   const int* __restrict__ pos, const float* __restrict__ w,
                                   const __nv_bfloat16* __restrict__ shared, int64_t T, int64_t Tpad, int H, int topk,
                                   uint64_t seq) {
  const __nv_bfloat16* y = reinterpret_cast<const __nv_bfloat16*>(yv);
  const uint8_t* y8 = reinterpret_cast<const uint8_t*>(yv);
  const int64_t Rr = Tpad / d.world;
  int64_t b = blockIdx.x, lo = 0, hi = 0;
  int k = 0;
  for (; k < kChunks; ++k) {
    row_piece(Rr, k, lo, hi);
    const int64_t n = (hi - lo) * d.world;
    if (b < n) break;
    b -= n;
  }
  const int j = (int)(b / (hi - lo));
  const int64_t t = j * Rr + lo + b % (hi - lo);
  __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(d.rs_send + (seq & 1) * d.slot_bytes) + t * H;
  __shared__ int ps[16];
  __shared__ float ws[16];
  if (t < T && threadIdx.x < topk) {
    ps[threadIdx.x] = pos[t * topk + threadIdx.x];
    ws[threadIdx.x] = w[t * topk + threadIdx.x];
  }
  __syncthreads();
  for (int c = threadIdx.x * 8; c < H; c += blockDim.x * 8) {
    float acc[8] = {};
    if (t < T) {
      const uint4 sv = *reinterpret_cast<const uint4*>(shared + t * H + c);
      const __nv_bfloat162* sb = reinterpret_cast<const __nv_bfloat162*>(&sv);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(sb[i]);
        acc[2 * i] = f.x; acc[2 * i + 1] = f.y;
      }
      for (int q = 0; q < topk; ++q) {
        if (y8s != nullptr) {
          const uint2 v = *reinterpret_cast<const uint2*>(y8 + (size_t)ps[q] * H + c);
          const float f = ws[q] * y8s[(size_t)ps[q] * (H / 128) + c / 128];
          const __nv_fp8x4_e4m3* q4 = reinterpret_cast<const __nv_fp8x4_e4m3*>(&v);
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const float4 x4 = static_cast<float4>(q4[i]);
            acc[4 * i] += f * x4.x; acc[4 * i + 1] += f * x4.y; acc[4 * i + 2] += f * x4.z; acc[4 * i + 3] += f * x4.w;
          }
          continue;
        }
        const uint4 v = *reinterpret_cast<const uint4*>(y + (size_t)ps[q] * H + c);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 f = __bfloat1622float2(b2[i]);
          acc[2 * i] += ws[q] * f.x;
          acc[2 * i + 1] += ws[q] * f.y;
        }
      }
    }
    uint4 o;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) ob[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    *reinterpret_cast<uint4*>(dst + c) = o;
  }
  __threadfence_system();
  __syncthreads();
  const int idx = k * d.world + j;
  if (threadIdx.x == 0 && atomicAdd(&d.blocks[idx], 1) == (unsigned)(hi - lo) - 1) {
    d.blocks[idx] = 0;
    const uint64_t row = (uint64_t)H * 2;
    publish(d, kRsPiece, seq, (uint64_t)j * kChunks + k, Rr * row, lo * row, hi * row);
  }
}

// out [Rr, H] = sum over ranks, in rank order, of their rows for this rank,
// piece by piece as the pieces arrive.
__global__ void rs_finish_kernel(Dev d, __nv_bfloat16* __restrict__ out, int64_t Rr, int H, uint64_t seq) {
  const int par = seq & 1;
  const size_t row = (size_t)H * 2, slice = Rr * row;
  const uint8_t* send = d.rs_send + par * d.slot_bytes + d.rank * slice;
  const uint8_t* recv = d.rs_recv + (size_t)par * d.slot_bytes;
  const size_t stride = (size_t)gridDim.x * blockDim.x * 16;
  for (int k = 0; k < kChunks; ++k) {
    int64_t lo, hi;
    row_piece(Rr, k, lo, hi);
    wait_piece(d, kReduceScatter, seq, k);
    for (size_t i = lo * row + ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 16; i < hi * row; i += stride) {
      float acc[8] = {};
      for (int src = 0; src < d.world; ++src) {
        const uint4 v = src == d.rank ? *reinterpret_cast<const uint4*>(send + i)
                                      : RS_LOAD(recv + (size_t)src * slice + i);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int q = 0; q < 8; ++q) acc[q] += __bfloat162float(b[q]);
      }
      __align__(16) __nv_bfloat16 o[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) o[q] = __float2bfloat16(acc[q]);
      *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(out) + i) = *reinterpret_cast<const uint4*>(o);
    }
  }
}

// A channel is one QP per root to one peer. Mesh: channel j goes to rank j
// over device r. Ring: channel 0 goes to prev over devices 0 and 1, channel 1
// to next over devices 2 and 3, and meets the peer's channel on the other side.
struct State {
  int rank = -1, world = 0, ndev = 0;
  bool ring = false;
  int prev = 0, next = 0, nslot = 0;
  int gid_idx[kMaxDev] = {};
  size_t slot_bytes = 0, rs_slot_bytes = 0;
  uint8_t *ag = nullptr, *rs_send = nullptr, *rs_recv = nullptr;
  uint64_t* flag = nullptr;
  Ctl* ctl = nullptr;
  ibv_context* ctx[kMaxDev] = {};
  ibv_pd* pd[kMaxDev] = {};
  ibv_mr *ag_mr[kMaxDev] = {}, *rs_send_mr[kMaxDev] = {}, *rs_recv_mr[kMaxDev] = {}, *flag_mr[kMaxDev] = {};
  ibv_cq* cq[kMaxDev] = {};
  ibv_qp* qp[kMaxWorld][2] = {};
  PeerInfo mine{};
  std::vector<PeerInfo> all;
  Dev dev{};
  bool connected = false;
};
State S;
std::atomic<bool> g_err{false};

int chan_peer(int c) { return S.ring ? (c ? S.next : S.prev) : c; }
int chan_dev(int c, int r) { return S.ring ? c * 2 + r : r; }         // device here
int chan_rdev(int c, int r) { return S.ring ? (1 - c) * 2 + r : r; }  // device at the peer

bool post(int c, int r, uint64_t laddr, uint32_t lkey, uint64_t len, uint64_t raddr, uint32_t rkey, uint64_t flag_off,
          uint64_t* seq_word) {
  const int j = chan_peer(c);
  ibv_sge sg{laddr, (uint32_t)len, lkey};
  ibv_sge fs{(uint64_t)seq_word, 8, 0};
  ibv_send_wr w{}, f{}, *bad;
  w.opcode = IBV_WR_RDMA_WRITE; w.sg_list = &sg; w.num_sge = 1;
  w.wr.rdma.remote_addr = raddr; w.wr.rdma.rkey = rkey;
  f.opcode = IBV_WR_RDMA_WRITE; f.sg_list = &fs; f.num_sge = 1;
  f.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
  f.wr_id = (uint64_t)j;
  f.wr.rdma.remote_addr = S.all[j].flag_addr + flag_off; f.wr.rdma.rkey = S.all[j].flag_rkey[chan_rdev(c, r)];
  w.next = len ? &f : nullptr;
  ibv_send_wr* head = len ? &w : &f;
  if (int e = ibv_post_send(S.qp[c][r], head, &bad)) {
    fprintf(stderr, "arxbig rank %d: post to %d root %d failed: %s\n", S.rank, j, r, strerror(e));
    return false;
  }
  return true;
}

// Sends one piece [lo, hi) of op `it` to peer j, split over both roots, each
// half followed by a flag write of `flag` on that root.
bool send_piece(const Work& it, int j, uint64_t* flag) {
  const int rank = S.rank, world = S.world;
  const uint64_t len = it.hi - it.lo, half = (len / 2 + 15) & ~15ull;
  for (int r = 0; r < 2; ++r) {
    const uint64_t off = it.lo + (r ? half : 0), n = r ? len - half : half;
    bool ok;
    if (it.op == kAllGather) {
      const uint64_t at = (it.seq % kAgSlots) * S.slot_bytes + rank * it.slice + off;
      ok = post(j, r, (uint64_t)S.ag + at, S.ag_mr[r]->lkey, n, S.all[j].ag_addr + at, S.all[j].ag_rkey[r],
                (rank * 2 + r) * 8, flag);
    } else {
      const uint64_t par = it.seq & 1;
      ok = post(j, r, (uint64_t)S.rs_send + par * S.rs_slot_bytes + j * it.slice + off, S.rs_send_mr[r]->lkey, n,
                S.all[j].rs_addr + par * S.rs_slot_bytes + rank * it.slice + off, S.all[j].rs_rkey[r],
                (world * 2 + rank * 2 + r) * 8, flag);
    }
    if (!ok) return false;
  }
  return true;
}

// ---- ring mode ----------------------------------------------------------------

// Byte offsets of a ring flag: data from src that arrived on channel ch, and
// a staged reduce-scatter chunk that arrived on channel ch.
uint64_t ring_slot(int op, int src, int ch, int r) { return ((uint64_t)op * S.nslot + (src * 2 + ch) * 2 + r) * 8; }
uint64_t stage_slot(int ch, int r) { return ((uint64_t)2 * S.nslot + ch * 2 + r) * 8; }

// Narrows [lo, hi) to half q. send_piece splits roots at the same point.
void halve(uint64_t& lo, uint64_t& hi, int q) {
  const uint64_t h = ((hi - lo) / 2 + 15) & ~15ull;
  if (q) lo += h; else hi = lo + h;
}

// Piece [lo, hi) of this rank's all-gather slice, to both neighbours: whole at
// world 4, half each way at world 2.
bool ring_send_ag(const Work& it, uint64_t* flag) {
  const uint64_t at = (it.seq % kAgSlots) * S.slot_bytes + S.rank * it.slice;
  for (int c = 1; c >= 0; --c)
    for (int r = 0; r < 2; ++r) {
      uint64_t lo = it.lo, hi = it.hi;
      halve(lo, hi, r);
      if (S.world == 2) halve(lo, hi, 1 - c);
      const PeerInfo& p = S.all[chan_peer(c)];
      if (!post(c, r, (uint64_t)S.ag + at + lo, S.ag_mr[chan_dev(c, r)]->lkey, hi - lo, p.ag_addr + at + lo,
                p.ag_rkey[chan_rdev(c, r)], ring_slot(0, S.rank, 1 - c, r), flag))
        return false;
    }
  return true;
}

// Piece [lo, hi) of chunk j of this rank's reduce-scatter input. The chunk for
// the rank two away goes half each way into the neighbours' staging slots.
bool ring_send_chunk(const Work& it, int j, uint64_t* flag) {
  const int world = S.world, f = (j - S.rank + world) % world;
  const bool split = world == 2 || f == 2, stage = world == 4 && f == 2;
  const uint64_t base = (it.seq & 1) * S.rs_slot_bytes;
  for (int c = 1; c >= 0; --c) {
    if (!split && (c == 1) != (f == 1)) continue;
    const int peer = chan_peer(c);
    const PeerInfo& p = S.all[peer];
    for (int r = 0; r < 2; ++r) {
      uint64_t lo = it.lo, hi = it.hi;
      halve(lo, hi, r);
      if (split) halve(lo, hi, 1 - c);
      if (!post(c, r, (uint64_t)S.rs_send + base + j * it.slice + lo, S.rs_send_mr[chan_dev(c, r)]->lkey, hi - lo,
                p.rs_addr + base + (stage ? peer : S.rank) * it.slice + lo, p.rs_rkey[chan_rdev(c, r)],
                stage ? stage_slot(1 - c, r) : ring_slot(1, S.rank, 1 - c, r), flag))
        return false;
    }
  }
  return true;
}

// The proxy's relay state at world 4. Every rank runs the same
// collectives in the same order, so this rank's own publishes say which seqs
// are all-gathers or reduce-scatters and where each piece starts and ends.
constexpr int kQ = 8;
struct Bound { uint64_t seq, slice, lo, hi; };
struct Relays {
  Bound bnd[2][kChunks] = {};              // by seq parity, piece
  uint64_t seqs[2][kQ] = {}, nseq[2] = {};  // per kind (0 all-gather, 1 reduce-scatter), in order
  struct { uint64_t i; int c; } at[2][2][2] = {};  // per kind, out channel, root: next seq index and piece
};

bool relay_note(Relays& R, const Work& it) {
  const int kind = it.op == kAllGather ? 0 : 1;
  const int c = (int)(it.op == kRsPiece ? it.chunk % kChunks : it.chunk);
  if (R.nseq[kind] == 0 || R.seqs[kind][(R.nseq[kind] - 1) % kQ] != it.seq) {
    for (int o = 0; o < 2; ++o)
      for (int r = 0; r < 2; ++r)
        if (R.nseq[kind] - R.at[kind][o][r].i >= kQ) {
          fprintf(stderr, "arxbig rank %d: relays fell %d collectives behind\n", S.rank, kQ);
          return false;
        }
    R.seqs[kind][R.nseq[kind]++ % kQ] = it.seq;
  }
  R.bnd[it.seq & 1][c] = {it.seq, it.slice, it.lo, it.hi};
  return true;
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

bool relay_pending(const Relays& R) {  // a relay this rank noted but has not forwarded yet
  for (int kind = 0; kind < 2; ++kind)
    for (int o = 0; o < 2; ++o)
      for (int r = 0; r < 2; ++r)
        if (R.at[kind][o][r].i < R.nseq[kind]) return true;
  return false;
}

// Forwards every piece whose flag has arrived, in order per stream. Out
// channel 1 carries prev's data on to next (first halves), channel 0 next's
// on to prev (second halves).
bool relay_step(Relays& R) {
  const volatile uint64_t* flags = S.flag;
  for (int kind = 0; kind < 2; ++kind)
    for (int o = 0; o < 2; ++o)
      for (int r = 0; r < 2; ++r) {
        auto& st = R.at[kind][o][r];
        const int src = o ? S.prev : S.next, peer = chan_peer(o);
        const uint64_t in = (kind ? stage_slot(1 - o, r) : ring_slot(0, src, 1 - o, r)) / 8;
        while (st.i < R.nseq[kind]) {
          const uint64_t s = R.seqs[kind][st.i % kQ];
          const Bound& B = R.bnd[s & 1][st.c];
          uint64_t flag = s * 16 + st.c + 1;
          if (B.seq != s || flags[in] < flag) break;
          std::atomic_thread_fence(std::memory_order_acquire);
          uint64_t lo = B.lo, hi = B.hi;
          halve(lo, hi, r);
          halve(lo, hi, 1 - o);
          const PeerInfo& p = S.all[peer];
          const int d = chan_dev(o, r), rd = chan_rdev(o, r);
          bool ok;
          if (kind == 0) {
            const uint64_t at = (s % kAgSlots) * S.slot_bytes + src * B.slice;
            ok = post(o, r, (uint64_t)S.ag + at + lo, S.ag_mr[d]->lkey, hi - lo, p.ag_addr + at + lo, p.ag_rkey[rd],
                      ring_slot(0, src, 1 - o, r), &flag);
          } else {
            const uint64_t base = (s & 1) * S.rs_slot_bytes;
            ok = post(o, r, (uint64_t)S.rs_recv + base + S.rank * B.slice + lo, S.rs_recv_mr[d]->lkey, hi - lo,
                      p.rs_addr + base + src * B.slice + lo, p.rs_rkey[rd], ring_slot(1, src, 1 - o, r), &flag);
          }
          if (!ok) return false;
          if (++st.c == kChunks) { st.c = 0; ++st.i; }
        }
      }
  return true;
}

void proxy_loop() {
  uint64_t done = 0;
  static Work pend[kMaxWorld][kChunks];  // kRsPiece items that arrived early
  uint32_t pend_mask[kMaxWorld] = {}, next_k[kMaxWorld] = {};
  uint64_t cur_seq[kMaxWorld] = {};
  const int rank = S.rank, world = S.world;
  const bool relay = S.ring && world == 4;
  static Relays R;
  auto send = S.ring ? ring_send_chunk : send_piece;
  IdleNap nap;
  while (!S.ctl->stop) {
    bool moved = false;
    for (int d = 0; d < S.ndev; ++d) {
      ibv_wc wc[32];
      const int n = ibv_poll_cq(S.cq[d], 32, wc);
      if (n > 0) moved = true;
      for (int i = 0; i < n; ++i)
        if (wc[i].status != IBV_WC_SUCCESS) {
          fprintf(stderr, "arxbig rank %d: completion error %d\n", rank, wc[i].status);
          g_err = true;
          return;
        }
    }
    if (relay && !relay_step(R)) { g_err = true; return; }
    Work& slot = S.ctl->ring[done % kRing];
    if (slot.tag != done + 1) {
      if (moved || (relay && relay_pending(R))) nap.busy(); else nap.idle();
      continue;
    }
    nap.busy();
    std::atomic_thread_fence(std::memory_order_acquire);
    Work it;
    it.op = slot.op; it.seq = slot.seq; it.chunk = slot.chunk; it.slice = slot.slice; it.lo = slot.lo; it.hi = slot.hi;
    ++done;
    if (relay && !relay_note(R, it)) { g_err = true; return; }
    if (it.op == kAllGather && S.ring) {
      uint64_t flag = it.seq * 16 + it.chunk + 1;
      if (!ring_send_ag(it, &flag)) { g_err = true; return; }
      continue;
    }
    if (it.op != kRsPiece) {  // one piece for every peer
      uint64_t flag = it.seq * 16 + it.chunk + 1;  // inline: copied at post time
      for (int k = 1; k < world; ++k)
        if (!send(it, (rank + k) % world, &flag)) { g_err = true; return; }
      continue;
    }
    // Row piece `sub` of destination j's chunk. Peers read the flag as "all
    // pieces up to sub arrived", so pieces go out in order per destination.
    const int j = (int)(it.chunk / kChunks), sub = (int)(it.chunk % kChunks);
    if (j == rank) continue;  // this rank's own rows never leave
    if (cur_seq[j] != it.seq) { cur_seq[j] = it.seq; next_k[j] = 0; pend_mask[j] = 0; }
    pend[j][sub] = it;
    pend_mask[j] |= 1u << sub;
    while (next_k[j] < kChunks && (pend_mask[j] >> next_k[j] & 1)) {
      const int k = next_k[j]++;
      uint64_t flag = it.seq * 16 + k + 1;
      if (!send(pend[j][k], j, &flag)) { g_err = true; return; }
    }
  }
}

Dev make_dev() {
  Dev d{};
  d.ag = S.ag; d.rs_send = S.rs_send; d.rs_recv = S.rs_recv; d.flag = S.flag; d.ctl = S.ctl;
  CK(cudaMalloc(&d.blocks, kChunks * kMaxWorld * 4)); CK(cudaMalloc(&d.ready, 8)); CK(cudaMalloc(&d.work_next, 8));
  CK(cudaMemset(d.blocks, 0, kChunks * kMaxWorld * 4)); CK(cudaMemset(d.ready, 0, 8)); CK(cudaMemset(d.work_next, 0, 8));
  d.rank = S.rank; d.world = S.world; d.slot_bytes = S.slot_bytes;
  d.nslot = S.nslot;
  for (int src = 0; src < S.world; ++src) {
    if (src == S.rank) continue;
    if (!S.ring) { d.mask |= 3u << (src * 2); continue; }
    // f hops from src to this rank going next-ward, which arrives on channel 0.
    const int f = (S.rank - src + S.world) % S.world;
    if (2 * f <= S.world) d.mask |= 3u << ((src * 2 + 0) * 2);
    if (2 * f >= S.world) d.mask |= 3u << ((src * 2 + 1) * 2);
  }
  return d;
}

}  // namespace

// Mesh: devs are root 0 and root 1. Ring: the port facing prev (root 0, root
// 1), then the port facing next. gids: each device's GID index.
// rs_slot_bytes: 0 leaves reduce_scatter unavailable and saves its buffers.
py::bytes prepare(int64_t rank, int64_t world, std::vector<std::string> devs, std::vector<int64_t> gids, bool ring,
                  int64_t slot_bytes, int64_t rs_slot_bytes) {
  TORCH_CHECK(S.rank < 0 && world >= 2 && world <= kMaxWorld && slot_bytes % (world * 16) == 0);
  TORCH_CHECK(rs_slot_bytes == 0 || rs_slot_bytes == slot_bytes, "arxbig: reduce_scatter uses the same slot size");
  TORCH_CHECK(!ring || world == 2 || world == 4, "arxbig ring mode supports world 2 or 4, not ", world,
              ": the ring reduce-scatter sends the rank two away half each way. Set VLLM_ARXBIG=0 to run arx alone");
  S.ndev = ring ? 4 : 2;
  TORCH_CHECK((int)devs.size() == S.ndev && (int)gids.size() == S.ndev, "arxbig: expected ", S.ndev, " devices");
  S.rank = rank; S.world = world; S.ring = ring; S.slot_bytes = slot_bytes; S.rs_slot_bytes = rs_slot_bytes;
  S.prev = (rank + world - 1) % world; S.next = (rank + 1) % world;
  S.nslot = (ring ? 4 : 2) * world;
  const size_t rs_alloc = rs_slot_bytes ? 2 * rs_slot_bytes : 4096;
  CK(cudaHostAlloc(&S.ag, kAgSlots * slot_bytes, cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.rs_send, rs_alloc, cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.rs_recv, rs_alloc, cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.flag, kFlagWords * sizeof(uint64_t), cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.ctl, sizeof(Ctl), cudaHostAllocMapped));
  memset(S.flag, 0, kFlagWords * sizeof(uint64_t));
  memset((void*)S.ctl, 0, sizeof(Ctl));
  int ndev;
  ibv_device** list = ibv_get_device_list(&ndev);
  const int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  for (int d = 0; d < S.ndev; ++d) {
    S.gid_idx[d] = gids[d];
    for (int i = 0; i < ndev; ++i)
      if (devs[d] == ibv_get_device_name(list[i])) S.ctx[d] = ibv_open_device(list[i]);
    TORCH_CHECK(S.ctx[d], "arxbig: no RDMA device ", devs[d]);
    S.pd[d] = ibv_alloc_pd(S.ctx[d]); IBCK(S.pd[d]);
    S.ag_mr[d] = ibv_reg_mr(S.pd[d], S.ag, kAgSlots * slot_bytes, acc); IBCK(S.ag_mr[d]);
    S.rs_send_mr[d] = ibv_reg_mr(S.pd[d], S.rs_send, rs_alloc, acc); IBCK(S.rs_send_mr[d]);
    S.rs_recv_mr[d] = ibv_reg_mr(S.pd[d], S.rs_recv, rs_alloc, acc); IBCK(S.rs_recv_mr[d]);
    S.flag_mr[d] = ibv_reg_mr(S.pd[d], S.flag, kFlagWords * sizeof(uint64_t), acc); IBCK(S.flag_mr[d]);
    S.cq[d] = ibv_create_cq(S.ctx[d], 4096, nullptr, nullptr, 0); IBCK(S.cq[d]);
    ibv_gid gid;
    IBCK(ibv_query_gid(S.ctx[d], 1, S.gid_idx[d], &gid) == 0);
    memcpy(S.mine.gid[d], gid.raw, 16);
    ibv_port_attr port{};
    IBCK(ibv_query_port(S.ctx[d], 1, &port) == 0);
    S.mine.mtu[d] = port.active_mtu;
    S.mine.ag_rkey[d] = S.ag_mr[d]->rkey;
    S.mine.rs_rkey[d] = S.rs_recv_mr[d]->rkey;
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
  S.mine.ag_addr = (uint64_t)S.ag;
  S.mine.rs_addr = (uint64_t)S.rs_recv;
  S.mine.flag_addr = (uint64_t)S.flag;
  return py::bytes(reinterpret_cast<const char*>(&S.mine), sizeof(PeerInfo));
}

void connect(std::vector<std::string> infos) {
  TORCH_CHECK(S.rank >= 0 && !S.connected && (int)infos.size() == S.world);
  S.all.resize(S.world);
  for (int j = 0; j < S.world; ++j) {
    TORCH_CHECK(infos[j].size() == sizeof(PeerInfo), "arxbig: PeerInfo size mismatch");
    memcpy(&S.all[j], infos[j].data(), sizeof(PeerInfo));
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
        fprintf(stderr, "arxbig rank %d%s root %d: path MTU %d bytes (ours %d, peer %d)\n", S.rank,
                S.ring ? (c ? " next" : " prev") : "", r, 128 << a.path_mtu, 128 << S.mine.mtu[d], 128 << p.mtu[rd]);
      a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.sq_psn = 0; a.max_rd_atomic = 1;
      IBCK(ibv_modify_qp(S.qp[c][r], &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                              IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0);
    }
  S.dev = make_dev();
  CK(cudaDeviceSynchronize());
  std::thread(proxy_loop).detach();
  S.connected = true;
}

static int blocks_for(size_t bytes) { return (int)std::max<size_t>(1, std::min<size_t>(48, bytes / (256 * 16 * 8))); }
static uint64_t g_seq = 0;  // collectives run in stream order, so the host numbers them

// in: this rank's slice (any dtype, contiguous). Returns the gathered data as a
// view of the pinned output slot, shaped [world * in.size(0), ...]; it stays
// valid until three more all-gathers have run.
torch::Tensor all_gather(torch::Tensor in) {
  TORCH_CHECK(S.connected && !g_err, "arxbig is not connected or its proxy failed");
  TORCH_CHECK(in.is_contiguous() && in.is_cuda() && in.dim() >= 1);
  const size_t slice = in.numel() * in.element_size();
  TORCH_CHECK(slice % 16 == 0 && slice * S.world <= S.slot_bytes, "arxbig: all_gather slice size");
  const uint64_t seq = ++g_seq;
  ag_kernel<<<blocks_for(slice), 256, 0, at::cuda::getCurrentCUDAStream()>>>(S.dev, (const uint8_t*)in.data_ptr(),
                                                                              slice, seq);
  std::vector<int64_t> shape(in.sizes().begin(), in.sizes().end());
  shape[0] *= S.world;
  return torch::from_blob(S.ag + (seq % kAgSlots) * S.slot_bytes, shape, in.options());
}

// in: bf16 [world * m, ...] contiguous. Returns bf16 [m, ...]: the sum over
// ranks of each rank's chunk `rank`.
torch::Tensor reduce_scatter(torch::Tensor in) {
  TORCH_CHECK(S.connected && !g_err, "arxbig is not connected or its proxy failed");
  TORCH_CHECK(S.rs_slot_bytes, "arxbig: prepared without reduce_scatter buffers");
  TORCH_CHECK(in.is_contiguous() && in.is_cuda() && in.scalar_type() == at::kBFloat16 && in.size(0) % S.world == 0);
  const size_t slice = in.numel() * 2 / S.world;
  TORCH_CHECK(slice % 16 == 0 && slice * S.world <= S.slot_bytes, "arxbig: reduce_scatter size");
  std::vector<int64_t> shape(in.sizes().begin(), in.sizes().end());
  shape[0] /= S.world;
  auto out = torch::empty(shape, in.options());
  const uint64_t seq = ++g_seq;
  rs_kernel<<<blocks_for(slice * S.world), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      S.dev, (const uint8_t*)in.data_ptr(), slice, (__nv_bfloat16*)out.data_ptr(), seq);
  return out;
}

// The pinned buffer the next reduce_scatter sends from, as a bf16 tensor of
// `shape`. A producer that writes its output here saves reduce_scatter's copy;
// it stays valid until that reduce_scatter.
torch::Tensor rs_input(std::vector<int64_t> shape) {
  TORCH_CHECK(S.connected && S.rs_slot_bytes);
  int64_t n = 1;
  for (auto v : shape) n *= v;
  TORCH_CHECK(n * 2 <= (int64_t)S.slot_bytes, "arxbig: rs_input too large");
  const uint64_t par = (g_seq + 1) & 1;
  return torch::from_blob(S.rs_send + par * S.slot_bytes, shape,
                          torch::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA));
}

// The MoE's final combine, reduce-scattered: finalize_rs_kernel writes and
// publishes rows; rs_finish (called next on the same stream) returns this
// rank's [Tpad / world, H] sum. y bf16 [R, H], pos int32 [T * topk],
// w fp32 [T, topk] (routing weights, scale folded in), shared bf16 [T, H].
int64_t moe_finalize_rs(torch::Tensor y, torch::Tensor pos, torch::Tensor w, torch::Tensor shared, int64_t T,
                        int64_t Tpad, c10::optional<torch::Tensor> y8s) {
  TORCH_CHECK(S.connected && !g_err && S.rs_slot_bytes, "arxbig: reduce_scatter unavailable");
  const int H = y.size(1), topk = w.size(1);
  TORCH_CHECK(Tpad % S.world == 0 && Tpad / S.world >= kChunks && T <= Tpad && topk <= 16 && H % 8 == 0);
  TORCH_CHECK((size_t)Tpad * H * 2 <= S.slot_bytes, "arxbig: MoE output larger than a slot");
  TORCH_CHECK(y.is_contiguous() && pos.is_contiguous() && w.is_contiguous() && shared.is_contiguous());
  TORCH_CHECK(!y8s.has_value() || (y.scalar_type() == at::kByte && H % 128 == 0), "arxbig: y8s needs e4m3 y");
  const uint64_t seq = ++g_seq;
  finalize_rs_kernel<<<Tpad, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      S.dev, y.data_ptr(), y8s.has_value() ? y8s->data_ptr<float>() : nullptr, pos.data_ptr<int>(),
      w.data_ptr<float>(), (const __nv_bfloat16*)shared.data_ptr(), T, Tpad, H, topk, seq);
  return (int64_t)seq;
}

torch::Tensor rs_finish(int64_t seq, int64_t rows, int64_t H) {
  TORCH_CHECK((uint64_t)seq == g_seq, "arxbig: rs_finish must follow its producer directly");
  auto out = torch::empty({rows, H}, torch::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA));
  rs_finish_kernel<<<blocks_for(rows * H * 2 * S.world), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      S.dev, (__nv_bfloat16*)out.data_ptr(), rows, (int)H, (uint64_t)seq);
  return out;
}

bool failed() { return g_err; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &prepare);
  m.def("connect", &connect);
  m.def("all_gather", &all_gather);
  m.def("reduce_scatter", &reduce_scatter);
  m.def("rs_input", &rs_input);
  m.def("moe_finalize_rs", &moe_finalize_rs, py::arg("y"), py::arg("pos"), py::arg("w"), py::arg("shared"), py::arg("T"), py::arg("Tpad"), py::arg("y8s") = py::none());
  m.def("rs_finish", &rs_finish);
  m.def("failed", &failed);
}
