// SPDX-License-Identifier: Apache-2.0
// megamoe on CUTLASS's own NVFP4 MoE layout (the tensors vLLM's FlashInfer
// CUTLASS backend keeps after processing), so decode and prefill share one copy
// of the weights: this kernel for decode-sized batches, CUTLASS for prefill.
//
// That layout, per expert: w13 [2I][K/2] bytes, rows 0..I-1 up and I..2I-1
// gate; w2 [H][I/2]; block scales e4m3 in FlashInfer's 128x4 swizzle; weight
// global scale = weight_scale_2 / input_scale (processing folded the input
// scale in). W4A16 as in megamoe.cu: activations stay fp16, exact in weights.
//
// A 16-row weight tile is contiguous in this layout. FC1 streams tiles into
// shared memory with cp.async, 128 bytes per row per copy, so DRAM sees long
// runs; loading mma fragments straight from global memory spans 16 rows per
// instruction and reaches only ~205 GB/s. Inside a 64-byte chunk the bytes are
// already in the order chunk_mma wants (lane t owns k 32t..32t+31).
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kSlots = 8;  // tokens per expert entry: the mma's N

__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ uint32_t e2m1x2(uint32_t byte) {
  uint32_t r;
  asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(r) : "r"(byte));
  return r;
}

__device__ __forceinline__ uint32_t e4m3x2(uint32_t two) {
  uint32_t r;
  asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(two));
  return r;
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// Broadcast the low or high half of a half2 to both lanes.
__device__ __forceinline__ uint32_t bcast(uint32_t h2, int hi) {
  uint32_t v = hi ? (h2 >> 16) : (h2 & 0xffff);
  return v | (v << 16);
}

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// One 128-wide k chunk of a 16x8 tile: rows g and g+8 from w0/w1, their
// packed scales, and this lane's 32 activation values x[0..3].
__device__ __forceinline__ void chunk_mma(float c[4], const uint4& w0, const uint4& w1,
                                          uint32_t sc, const uint4 x[4]) {
  const uint32_t s_lo = e4m3x2(sc & 0xffff);  // row g:   block 2t, 2t+1
  const uint32_t s_hi = e4m3x2(sc >> 16);     // row g+8: block 2t, 2t+1
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int blk = j >> 2;
    const uint32_t sg = bcast(s_lo, blk), sg8 = bcast(s_hi, blk);
    const uint32_t p0 = (word(w0, j >> 1) >> ((j & 1) * 16)) & 0xffff;
    const uint32_t p1 = (word(w1, j >> 1) >> ((j & 1) * 16)) & 0xffff;
    uint32_t a[4];
    a[0] = hmul2(e2m1x2(p0 & 0xff), sg);   // row g,   k+0..1
    a[1] = hmul2(e2m1x2(p1 & 0xff), sg8);  // row g+8, k+0..1
    a[2] = hmul2(e2m1x2(p0 >> 8), sg);     // row g,   k+2..3
    a[3] = hmul2(e2m1x2(p1 >> 8), sg8);    // row g+8, k+2..3
    const uint4& xv = x[j >> 1];
    const int wi = (j & 1) * 2;
    mma16816(c, a, word(xv, wi), word(xv, wi + 1));
  }
}

// Groups the M*topk (token, k) pairs by expert, eight tokens per entry, in
// ascending expert order, and converts x to fp16. One block.
__global__ void route_prep(const int* __restrict__ topk_ids, int M, int topk, int E,
                           const __nv_bfloat16* __restrict__ x, int K, __half* __restrict__ xh,
                           int* __restrict__ n_entries, int* __restrict__ entry_expert,
                           int* __restrict__ entry_tk) {
  extern __shared__ int sm[];
  int* cnt = sm;            // [E]
  int* base = sm + E;       // [E]
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int i = tid; i < M * K; i += nt) xh[i] = __float2half_rn(__bfloat162float(x[i]));
  for (int e = tid; e < E; e += nt) cnt[e] = 0;
  __syncthreads();
  const int pairs = M * topk;
  int my_pos[4];
  // A pair's slot is its rank among earlier pairs with the same expert, so
  // every block that rebuilds this table agrees on it.
  // A pair with expert id -1 is skipped: no entry, so no weight reads, and
  // combine leaves it out.
  for (int i = tid, n = 0; i < pairs; i += nt, ++n) {
    const int e = topk_ids[i];
    if (e < 0) continue;
    int r = 0;
    for (int j = 0; j < i; ++j) r += topk_ids[j] == e;
    my_pos[n] = r;
    atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  // Exclusive scan of entries-per-expert (ceil(cnt/8)) across the block,
  // one expert per thread: E <= blockDim.
  int* wsum = sm + 2 * E;  // [32]
  const int lane = tid & 31, wid = tid >> 5;
  const int mine = tid < E ? (cnt[tid] + kSlots - 1) / kSlots : 0;
  int incl = mine;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) wsum[wid] = incl;
  for (int i = tid; i < pairs * kSlots; i += nt) entry_tk[i] = -1;
  for (int i = tid; i < pairs; i += nt) entry_expert[i] = -1;
  __syncthreads();
  if (wid == 0) {
    int s = lane < nt / 32 ? wsum[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, s, o);
      if (lane >= o) s += v;
    }
    wsum[lane] = s;  // inclusive per-warp totals
  }
  __syncthreads();
  const int excl = incl - mine + (wid ? wsum[wid - 1] : 0);
  if (tid < E) {
    base[tid] = excl;
    for (int j = 0; j < mine; ++j) entry_expert[excl + j] = tid;
  }
  if (tid == nt - 1) *n_entries = wsum[nt / 32 - 1];
  __syncthreads();
  for (int i = tid, n = 0; i < pairs; i += nt, ++n) {
    const int e = topk_ids[i], p = my_pos[n];
    if (e < 0) continue;
    entry_tk[(base[e] + p / kSlots) * kSlots + p % kSlots] = i;  // i = token*topk + k
  }
}


__device__ __forceinline__ size_t scale_off(int r, int c, int C) {  // 128x4 swizzle within one expert
  return ((size_t)(r >> 7) * (C >> 2) + (c >> 2)) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4 + (c & 3);
}

// Scales for chunk_mma: {row a: blocks c, c+1; row b: blocks c, c+1}.
__device__ __forceinline__ uint32_t scale_pair(const uint8_t* s, int ra, int rb, int c, int C) {
  const uint32_t lo = *reinterpret_cast<const uint16_t*>(s + scale_off(ra, c, C));
  const uint32_t hi = *reinterpret_cast<const uint16_t*>(s + scale_off(rb, c, C));
  return lo | (hi << 16);
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

constexpr int kRowBytes = 128;    // bytes per row per stage (256 k)
constexpr int kRowStride = 192;   // padded: rows g and g+1 land on disjoint banks
constexpr int kStageBytes = 32 * kRowStride;  // 16 up rows + 16 gate rows

// FC1 + GLM SwiGLU for one expert entry and 16 output rows. KSPLIT warps
// split K; each double-buffers its slice through shared memory.
template <int KSPLIT>
__global__ void __launch_bounds__(KSPLIT * 32) fc1_rm(
    const uint8_t* __restrict__ w13, const uint8_t* __restrict__ s13,
    const float* __restrict__ alpha1, const __half* __restrict__ xh, int topk,
    const int* __restrict__ n_entries, const int* __restrict__ entry_expert,
    const int* __restrict__ entry_tk, __half* __restrict__ hbuf, int K, int I, float limit) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int u = blockIdx.x;
  if (u >= *n_entries) return;
  const int e = entry_expert[u], tile = blockIdx.y;
  const int ks = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int tk = entry_tk[u * kSlots + g];
  const __half* xrow = tk >= 0 ? xh + (size_t)(tk / topk) * K + 32 * t : nullptr;
  const int C = K / 16, rowb = K / 2;
  const uint8_t* wexp = w13 + (size_t)e * 2 * I * rowb;
  const uint8_t* sexp = s13 + (size_t)e * 2 * I * C;
  uint8_t* buf = smem + ks * 2 * kStageBytes;
  const int kb0 = rowb / KSPLIT * ks, nst = rowb / KSPLIT / kRowBytes;

  // This lane's copies: rows 4i + lane/8, 16-byte segment lane%8.
  auto issue = [&](int st) {
    uint8_t* dst = buf + (st & 1) * kStageBytes;
    const int kb = kb0 + st * kRowBytes;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = 4 * i + (lane >> 3), seg = lane & 7;
      const int wrow = row < 16 ? tile * 16 + row : I + tile * 16 + row - 16;
      cp_async16(dst + row * kRowStride + seg * 16, wexp + (size_t)wrow * rowb + kb + seg * 16);
    }
    cp_commit();
  };
  float cg[4] = {0.f, 0.f, 0.f, 0.f}, cu[4] = {0.f, 0.f, 0.f, 0.f};
  issue(0);
  for (int st = 0; st < nst; ++st) {
    if (st + 1 < nst) { issue(st + 1); cp_wait<1>(); } else { cp_wait<0>(); }
    __syncwarp();
    const uint8_t* b = buf + (st & 1) * kStageBytes;
    const int kbyte = kb0 + st * kRowBytes;
#pragma unroll
    for (int j = 0; j < 2; ++j) {  // two 128-k chunks per stage
      const int k = kbyte * 2 + j * 128, c = k / 16 + 2 * t;
      uint4 X[4] = {};
      if (xrow) {
#pragma unroll
        for (int i = 0; i < 4; ++i) X[i] = __ldg(reinterpret_cast<const uint4*>(xrow + k) + i);
      }
      const int off = j * 64 + t * 16;
      const uint4 U0 = *reinterpret_cast<const uint4*>(b + g * kRowStride + off);
      const uint4 U8 = *reinterpret_cast<const uint4*>(b + (g + 8) * kRowStride + off);
      const uint4 G0 = *reinterpret_cast<const uint4*>(b + (16 + g) * kRowStride + off);
      const uint4 G8 = *reinterpret_cast<const uint4*>(b + (24 + g) * kRowStride + off);
      const int ru = tile * 16 + g, rg = I + tile * 16 + g;
      chunk_mma(cu, U0, U8, scale_pair(sexp, ru, ru + 8, c, C), X);
      chunk_mma(cg, G0, G8, scale_pair(sexp, rg, rg + 8, c, C), X);
    }
    __syncwarp();  // the next issue overwrites this buffer
  }
  if constexpr (KSPLIT > 1) {
    float* red = reinterpret_cast<float*>(smem);  // reuses the stage buffers
    __syncthreads();
    if (ks > 0)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        red[((ks - 1) * 32 + lane) * 8 + i] = cg[i];
        red[((ks - 1) * 32 + lane) * 8 + 4 + i] = cu[i];
      }
    __syncthreads();
    if (ks > 0) return;
#pragma unroll
    for (int s = 0; s < KSPLIT - 1; ++s)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        cg[i] += red[(s * 32 + lane) * 8 + i];
        cu[i] += red[(s * 32 + lane) * 8 + 4 + i];
      }
  }
  const float a = alpha1[e];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float gv = fminf(a * cg[i], limit);
    const float uv = fminf(fmaxf(a * cu[i], -limit), limit);
    const float h = gv / (1.f + expf(-gv)) * uv;
    const int row = tile * 16 + g + (i >> 1) * 8, slot = 2 * t + (i & 1);
    hbuf[((size_t)u * kSlots + slot) * I + row] = __float2half_rn(h);
  }
}

// FC2 for one expert entry and WARPS 16-row tiles (each 16 * I/2 contiguous
// bytes), scaled by alpha2 and the routing weight, into the (token, k) slot.
template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32) fc2_rm(
    const uint8_t* __restrict__ w2, const uint8_t* __restrict__ s2,
    const float* __restrict__ alpha2, const __half* __restrict__ hbuf,
    const float* __restrict__ topk_w, const int* __restrict__ n_entries,
    const int* __restrict__ entry_expert, const int* __restrict__ entry_tk,
    float* __restrict__ ypart, int Hout, int I) {
  const int u = blockIdx.x;
  if (u >= *n_entries) return;
  const int e = entry_expert[u];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int tile = blockIdx.y * WARPS + warp;
  if (tile * 16 >= Hout) return;
  const int nchunk = I / 128, rowb = I / 2, C = I / 16;
  const uint8_t* r0 = w2 + ((size_t)e * Hout + tile * 16 + g) * rowb + 16 * t;
  const uint8_t* r8 = r0 + (size_t)8 * rowb;
  const uint8_t* sexp = s2 + (size_t)e * Hout * C;
  const __half* hrow = hbuf + ((size_t)u * kSlots + g) * I + 32 * t;
  constexpr int kMaxChunks = 8;  // I <= 1024
  uint4 W0[kMaxChunks], W8[kMaxChunks];
  uint32_t S[kMaxChunks];
#pragma unroll
  for (int k = 0; k < kMaxChunks; ++k)
    if (k < nchunk) {
      W0[k] = ld_stream(r0 + k * 64); W8[k] = ld_stream(r8 + k * 64);
      S[k] = scale_pair(sexp, tile * 16 + g, tile * 16 + g + 8, k * 8 + 2 * t, C);
    }
  float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < kMaxChunks; ++k)
    if (k < nchunk) {
      uint4 X[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) X[i] = *(reinterpret_cast<const uint4*>(hrow + k * 128) + i);
      chunk_mma(c, W0[k], W8[k], S[k], X);
    }
  const float a = alpha2[e];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int slot = 2 * t + (i & 1);
    const int tk = entry_tk[u * kSlots + slot];
    if (tk < 0) continue;
    const int row = tile * 16 + g + (i >> 1) * 8;
    ypart[(size_t)tk * Hout + row] = a * topk_w[tk] * c[i];
  }
}

// out[m] = sum over k of ypart[m, k], in k order, over the pairs with an expert.
__global__ void combine(const float* __restrict__ ypart, const int* __restrict__ topk_ids, int topk,
                        int Hout, __nv_bfloat16* __restrict__ out) {
  const int m = blockIdx.y;
  const int f = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (f >= Hout) return;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int k = 0; k < topk; ++k) {
    if (topk_ids[m * topk + k] < 0) continue;
    const float4 v = *reinterpret_cast<const float4*>(ypart + ((size_t)m * topk + k) * Hout + f);
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out + (size_t)m * Hout + f);
  o[0] = __floats2bfloat162_rn(acc.x, acc.y);
  o[1] = __floats2bfloat162_rn(acc.z, acc.w);
}



}  // namespace

// w13 [E, 2I, K/2] u8, s13 [E, 2I, K/16] e4m3 (swizzled), alpha1 [E] f32 (the
// weight's global scale), likewise w2/s2/alpha2 with [E, H, I/2]. Scratch is
// passed in (sized for M tokens) so a CUDA graph can hold it.
void megamoe_rm_forward(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w,
                        torch::Tensor w13, torch::Tensor s13, torch::Tensor alpha1,
                        torch::Tensor w2, torch::Tensor s2, torch::Tensor alpha2,
                        torch::Tensor xh, torch::Tensor ints, torch::Tensor hbuf,
                        torch::Tensor ypart, torch::Tensor out, double limit, int64_t variant) {
  const int M = x.size(0), K = x.size(1), topk = topk_ids.size(1);
  const int E = w13.size(0), I = w13.size(1) / 2, Hout = w2.size(0) ? w2.size(1) : 0;
  const int pairs = M * topk;
  TORCH_CHECK(I <= 1024 && I % 128 == 0 && K % (128 * 8) == 0 && Hout % 128 == 0 && pairs <= 4096 && E <= 1024);
  TORCH_CHECK(w13.size(2) == K / 2 && w2.size(2) == I / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  int* n_entries = ints.data_ptr<int>();
  int* entry_expert = n_entries + 1;
  int* entry_tk = entry_expert + pairs;
  route_prep<<<1, 1024, (2 * E + 32) * sizeof(int), stream>>>(
      topk_ids.data_ptr<int>(), M, topk, E, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), K,
      reinterpret_cast<__half*>(xh.data_ptr()), n_entries, entry_expert, entry_tk);
  const uint8_t* s13p = reinterpret_cast<const uint8_t*>(s13.data_ptr());
  const uint8_t* s2p = reinterpret_cast<const uint8_t*>(s2.data_ptr());
  auto fc1 = [&](auto ksplit) {
    constexpr int KS = decltype(ksplit)::value;
    const size_t smem = (size_t)KS * 2 * kStageBytes;
    cudaFuncSetAttribute(fc1_rm<KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    fc1_rm<KS><<<dim3(pairs, I / 16), KS * 32, smem, stream>>>(
        w13.data_ptr<uint8_t>(), s13p, alpha1.data_ptr<float>(),
        reinterpret_cast<const __half*>(xh.data_ptr()), topk, n_entries, entry_expert, entry_tk,
        reinterpret_cast<__half*>(hbuf.data_ptr()), K, I, (float)limit);
  };
  auto fc2 = [&](auto warps) {
    constexpr int W = decltype(warps)::value;
    fc2_rm<W><<<dim3(pairs, (Hout / 16 + W - 1) / W), W * 32, 0, stream>>>(
        w2.data_ptr<uint8_t>(), s2p, alpha2.data_ptr<float>(),
        reinterpret_cast<const __half*>(hbuf.data_ptr()), topk_w.data_ptr<float>(), n_entries,
        entry_expert, entry_tk, ypart.data_ptr<float>(), Hout, I);
  };
  using std::integral_constant;
  switch (variant / 10) {  // tens digit: FC1 K split
    case 0: fc1(integral_constant<int, 4>{}); break;
    case 1: fc1(integral_constant<int, 2>{}); break;
    default: fc1(integral_constant<int, 8>{}); break;
  }
  switch (variant % 10) {  // units digit: FC2 warps per block
    case 0: fc2(integral_constant<int, 4>{}); break;
    case 1: fc2(integral_constant<int, 2>{}); break;
    case 2: fc2(integral_constant<int, 8>{}); break;
    default: fc2(integral_constant<int, 1>{}); break;
  }
  combine<<<dim3((Hout / 4 + 127) / 128, M), 128, 0, stream>>>(
      ypart.data_ptr<float>(), topk_ids.data_ptr<int>(), topk, Hout,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &megamoe_rm_forward); }
