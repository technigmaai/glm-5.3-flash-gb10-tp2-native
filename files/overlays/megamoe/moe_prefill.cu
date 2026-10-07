// SPDX-License-Identifier: Apache-2.0
// Prefill MoE for GB10 (sm_121a): NVFP4 x NVFP4 grouped GEMMs on the tensors
// vLLM hands FlashInfer's CUTLASS MoE.
//   fc1: gathers token rows by index, SwiGLU and NVFP4 requantization in the
//        epilogue. xq [Mtok, K/2] u8 (e2m1 pairs, low nibble first) with
//        128x4-swizzled e4m3 scales xs; w13 [E, 2I, K/2] (rows 0..I-1 up,
//        I..2I-1 gate). Writes hq [R, I/2] in expert-sorted row order, hs
//        swizzled over R.
//   fc2: hq x w2 [E, H, I/2], scaled by g2, to bf16 rows y [R, H].
//   finalize: out[t] = sum_k w[t, k] * y[pos[t, k]].
// Rows are grouped by expert (expert_off) and cut into 128-row tiles; tiles
// [T, 2] lists (expert, m tile) and may end in padding entries with expert >= E.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

#ifndef MOE_BKB
#define MOE_BKB 64
#endif
#ifndef MOE_STAGES
#define MOE_STAGES 4
#endif
#ifndef MOE_PREFETCH
#define MOE_PREFETCH 0  // bit 0: bulk-prefetch the A rows to L2 at tile start; bit 1: the B rows
#endif
#ifndef MOE_MINB
#define MOE_MINB 1
#endif
constexpr int BM = 128, BN = 128, BKB = MOE_BKB;  // rows, weight rows, bytes of K per stage
constexpr int STAGES = MOE_STAGES, THREADS = 256;
constexpr int CPR = BKB / 16;                   // 16-byte chunks per smem row
constexpr int NCP = BM * CPR / THREADS;         // chunks per thread per operand
constexpr int WPR = BKB / 32;                   // 4-byte scale words (64 elements) per row per stage
constexpr int NSW = BM * WPR / THREADS;         // scale words per thread per operand
constexpr int A_BYTES = BM * BKB, B_BYTES = BN * BKB, SA_BYTES = BM * WPR * 4, SB_BYTES = BN * WPR * 4;
constexpr int STAGE_BYTES = A_BYTES + B_BYTES + SA_BYTES + SB_BYTES;

__device__ __forceinline__ size_t scale_off(int r, int c, int C) {  // flashinfer's 128x4 swizzle
  return ((size_t)(r >> 7) * (C >> 2) + (c >> 2)) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4 + (c & 3);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp4(void* dst, const void* src, bool ok) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                        uint32_t sb) {
  const uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z),
        "h"(z));
}

// 16-byte chunk c of smem row r, XOR-swizzled so ldmatrix is conflict-free.
__device__ __forceinline__ int swz(int r, int c) {
  return r * BKB + ((CPR == 4 ? (c ^ ((r >> 1) & 3)) : (c ^ (r & 7))) << 4);
}

__device__ __forceinline__ uint8_t e2m1x2(float lo, float hi) {
  uint16_t r;  // hi goes to the upper nibble
  asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t, t}; }\n" : "=h"(r) : "f"(lo), "f"(hi));
  return (uint8_t)(r & 0xff);
}

struct Args {
  const uint8_t* a;        // fc1: xq, fc2: hq
  const uint8_t* as;       // their swizzled scales
  const int* rows;         // fc1: token of each expanded row
  const int* expert_off;   // [E+1]
  const int2* tiles;       // [T] (expert, m tile)
  const uint8_t* w;        // w13 or w2
  const uint8_t* ws;       // their per-expert swizzled scales
  const float* alpha;      // g13 or g2 [E]
  const float* a2;         // fc1: fc2's input global scale [E]
  uint8_t* hq;             // fc1 outputs
  uint8_t* hs;
  __nv_bfloat16* y;        // fc2 output (bf16), or
  uint8_t* y8;             // e4m3 rows with
  float* y8s;              // one scale per (row, 128 columns)
  int K, N, I, E;          // K of this GEMM, weight rows per expert, intermediate size
  float limit;
};

// FC1: B tile = 64 up rows + the 64 matching gate rows; each warp holds 32 up
// and the same 32 gate columns. FC2: B tile = 128 contiguous rows of w2.
template <bool FC1>
__global__ void __launch_bounds__(THREADS, MOE_MINB) moe_gemm(const Args p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int2 tile = p.tiles[blockIdx.y];
  if (tile.x >= p.E) return;
  const int e = tile.x;
  const int n0 = blockIdx.x * (FC1 ? BN / 2 : BN);
  const int r_begin = p.expert_off[e] + tile.y * BM, r_end = p.expert_off[e + 1];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp & 3, wn = warp >> 2;
  const int KB = p.K / 2, KS = p.K / 16;
  const uint8_t* wexp = p.w + (size_t)e * p.N * KB;
  const uint8_t* sexp = p.ws + (size_t)e * p.N * KS;
  auto wrow_of = [&](int r) { return FC1 ? (r < BN / 2 ? n0 + r : p.I + n0 + r - BN / 2) : n0 + r; };

  const uint8_t* a_src[NCP]; bool a_ok[NCP]; int a_dst[NCP];
  const uint8_t* b_src[NCP];
#pragma unroll
  for (int j = 0; j < NCP; ++j) {
    const int idx = tid + j * THREADS, r = idx / CPR, c = idx % CPR;
    const int gr = r_begin + r;
    a_ok[j] = gr < r_end;
    const int arow = a_ok[j] ? (FC1 ? p.rows[gr] : gr) : 0;
    a_src[j] = p.a + (size_t)arow * KB + c * 16;
    a_dst[j] = swz(r, c);
    b_src[j] = wexp + (size_t)wrow_of(r) * KB + c * 16;
  }
  if constexpr ((MOE_PREFETCH & 1) != 0) {
    // One whole-row request per gathered row: DRAM serves 2 KB runs instead of
    // 64-byte pieces spread over the K loop. Bit 2: only the first N tile asks.
    // Bits 3+: ask for the rows of the tile that many tiles ahead instead.
    constexpr int AHEAD = MOE_PREFETCH >> 3;
    const bool asker = (MOE_PREFETCH & 4) == 0 || blockIdx.x == 0;
    const int ty = blockIdx.y + AHEAD;
    if (asker && tid < BM && ty < gridDim.y) {
      const int2 pt = p.tiles[ty];
      if (pt.x < p.E) {
        const int pr = p.expert_off[pt.x] + pt.y * BM + tid;
        if (pr < p.expert_off[pt.x + 1]) {
          const uint8_t* src = p.a + (size_t)(FC1 ? p.rows[pr] : pr) * KB;
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(src), "r"(KB));
        }
      }
    }
  }
  if constexpr ((MOE_PREFETCH & 2) != 0) {
    if (tid >= BM && tid < BM + BN) {
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(wexp + (size_t)wrow_of(tid - BM) * KB), "r"(KB));
    }
  }
  int s_r[NSW], s_w[NSW], a_srow[NSW], b_srow[NSW]; bool sa_ok[NSW];
#pragma unroll
  for (int j = 0; j < NSW; ++j) {
    const int idx = tid + j * THREADS;
    s_r[j] = idx / WPR; s_w[j] = idx % WPR;
    sa_ok[j] = r_begin + s_r[j] < r_end;
    a_srow[j] = sa_ok[j] ? (FC1 ? p.rows[r_begin + s_r[j]] : r_begin + s_r[j]) : 0;
    b_srow[j] = wrow_of(s_r[j]);
  }

  auto load_stage = [&](int s, int kt) {
    uint8_t* st = smem + s * STAGE_BYTES;
    const int kb = kt * BKB;
#pragma unroll
    for (int j = 0; j < NCP; ++j) {
      cp16(st + a_dst[j], a_src[j] + kb, a_ok[j]);
      cp16(st + A_BYTES + a_dst[j], b_src[j] + kb, true);
    }
#pragma unroll
    for (int j = 0; j < NSW; ++j) {
      const int c = kt * (BKB / 8) + s_w[j] * 4;
      cp4(st + A_BYTES + B_BYTES + (s_r[j] * WPR + s_w[j]) * 4, p.as + scale_off(a_srow[j], c, KS), sa_ok[j]);
      cp4(st + A_BYTES + B_BYTES + SA_BYTES + (s_r[j] * WPR + s_w[j]) * 4, sexp + scale_off(b_srow[j], c, KS), true);
    }
  };

  float acc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.f;

  const int KT = KB / BKB;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_stage(s, s);
    cp_commit();
  }
  const int g = lane >> 2, q = lane >> 3, rr = lane & 7;
  for (int kt = 0; kt < KT; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (kt + STAGES - 1 < KT) load_stage((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
    cp_commit();
    const uint8_t* st = smem + (kt % STAGES) * STAGE_BYTES;
    const uint8_t* sA = st;
    const uint8_t* sB = st + A_BYTES;
    const uint32_t* sSA = reinterpret_cast<const uint32_t*>(st + A_BYTES + B_BYTES);
    const uint32_t* sSB = reinterpret_cast<const uint32_t*>(st + A_BYTES + B_BYTES + SA_BYTES);
#pragma unroll
    for (int kk = 0; kk < BKB / 32; ++kk) {
      uint32_t af[2][4], sa[2];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int r = wm * 32 + mi * 16 + rr + (q & 1) * 8;
        ldsm4(af[mi][0], af[mi][1], af[mi][2], af[mi][3], sA + swz(r, kk * 2 + (q >> 1)));
        sa[mi] = sSA[(wm * 32 + mi * 16 + g + 8 * (lane & 1)) * WPR + kk];
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {  // pairs of n8 blocks, j = 2np and 2np+1
        const int nb = FC1 ? (np < 2 ? 0 : BN / 2) + wn * 32 + (np & 1) * 16 : wn * 64 + np * 16;
        uint32_t b[4];
        ldsm4(b[0], b[1], b[2], b[3], sB + swz(nb + (q >> 1) * 8 + rr, kk * 2 + (q & 1)));
        const uint32_t sb0 = sSB[(nb + g) * WPR + kk], sb1 = sSB[(nb + 8 + g) * WPR + kk];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          mma_fp4(acc[mi][np * 2], af[mi], b[0], b[1], sa[mi], sb0);
          mma_fp4(acc[mi][np * 2 + 1], af[mi], b[2], b[3], sa[mi], sb1);
        }
      }
    }
  }
  cp_wait<0>();

  const float ga = p.alpha[e];
  const int t4 = lane & 3;
  if constexpr (FC1) {
    // acc[mi][j] (j<4) is up, acc[mi][j+4] the matching gate. Intermediate column
    // = n0 + wn*32 + j*8 + 2t + {0,1}; rows g and g+8 of each m16 tile.
    const float qs = p.a2[e];
    const int IB = p.I / 2, IS = p.I / 16;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = r_begin + wm * 32 + mi * 16 + g + half * 8;
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {  // two 16-column quant blocks per warp
          float h[4];
#pragma unroll
          for (int jj = 0; jj < 2; ++jj)
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              const int j = blk * 2 + jj;
              const float gv = fminf(ga * acc[mi][j + 4][half * 2 + v], p.limit);
              const float uv = fminf(fmaxf(ga * acc[mi][j][half * 2 + v], -p.limit), p.limit);
              h[jj * 2 + v] = gv / (1.f + __expf(-gv)) * uv;
            }
          float amax = fmaxf(fmaxf(fabsf(h[0]), fabsf(h[1])), fmaxf(fabsf(h[2]), fabsf(h[3])));
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
          const __nv_fp8_e4m3 sf8(amax * qs * (1.f / 6.f));
          const float sf = float(sf8);
          const float inv = sf > 0.f ? qs / sf : 0.f;
          const uint32_t b0 = e2m1x2(h[0] * inv, h[1] * inv), b1 = e2m1x2(h[2] * inv, h[3] * inv);
          // Thread t holds byte t of each 4-byte word (columns j*8 .. j*8+7).
          uint32_t w0 = b0 << (8 * t4), w1 = b1 << (8 * t4);
          w0 |= __shfl_xor_sync(0xffffffff, w0, 1); w0 |= __shfl_xor_sync(0xffffffff, w0, 2);
          w1 |= __shfl_xor_sync(0xffffffff, w1, 1); w1 |= __shfl_xor_sync(0xffffffff, w1, 2);
          const int col = n0 + wn * 32 + blk * 16;
          if (row < r_end) {
            if (t4 == 0) *reinterpret_cast<uint32_t*>(p.hq + (size_t)row * IB + col / 2) = w0;
            if (t4 == 1) *reinterpret_cast<uint32_t*>(p.hq + (size_t)row * IB + col / 2 + 4) = w1;
            if (t4 == 2) p.hs[scale_off(row, col / 16, IS)] = sf8.__x;
          }
        }
      }
  } else {
    // Stage the 128x128 bf16 tile in smem, then write whole 16-byte chunks.
    constexpr int YS = BN + 8;  // padded row stride (bf16 elements)
    __nv_bfloat16* sy = reinterpret_cast<__nv_bfloat16*>(smem);
    __syncthreads();
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int r = wm * 32 + mi * 16 + g + half * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j)
          *reinterpret_cast<__nv_bfloat162*>(sy + r * YS + wn * 64 + j * 8 + 2 * t4) =
              __floats2bfloat162_rn(ga * acc[mi][j][half * 2], ga * acc[mi][j][half * 2 + 1]);
      }
    __syncthreads();
    const int H = p.N;
    if (p.y8 != nullptr) {
      // 8 warps x 16 rows: each warp reduces a row's amax, then writes e4m3 in 16-byte chunks.
      for (int r = warp; r < BM; r += THREADS / 32) {
        if (r_begin + r >= r_end) break;
        const __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(sy + r * YS + lane * 4);
        const __nv_bfloat162 w2v = *reinterpret_cast<const __nv_bfloat162*>(sy + r * YS + lane * 4 + 2);
        const float2 f0 = __bfloat1622float2(v2), f1 = __bfloat1622float2(w2v);
        float amax = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        const float sc = amax > 0.f ? amax / 448.f : 1.f, inv = 1.f / sc;
        const __nv_fp8x4_e4m3 q4(make_float4(f0.x * inv, f0.y * inv, f1.x * inv, f1.y * inv));
        *reinterpret_cast<uint32_t*>(p.y8 + (size_t)(r_begin + r) * H + n0 + lane * 4) =
            *reinterpret_cast<const uint32_t*>(&q4);
        if (lane == 0) p.y8s[(size_t)(r_begin + r) * (H / BN) + n0 / BN] = sc;
      }
    } else {
#pragma unroll
      for (int i = 0; i < BM * BN / 8 / THREADS; ++i) {
        const int idx = tid + i * THREADS, r = idx / (BN / 8), c = idx % (BN / 8);
        if (r_begin + r < r_end)
          *reinterpret_cast<uint4*>(p.y + (size_t)(r_begin + r) * H + n0 + c * 8) =
              *reinterpret_cast<const uint4*>(sy + r * YS + c * 8);
      }
    }
  }
}

// out[t] = sum_k w[t, k] * y8[pos[t, k]] * y8s[pos[t, k], column / 128].
__global__ void finalize8_kernel(const uint8_t* __restrict__ y8, const float* __restrict__ y8s,
                                 const int* __restrict__ pos, const float* __restrict__ w,
                                 __nv_bfloat16* __restrict__ out, int H, int topk) {
  const int t = blockIdx.x;
  __shared__ int ps[16];
  __shared__ float ws[16];
  if (threadIdx.x < topk) {
    ps[threadIdx.x] = pos[t * topk + threadIdx.x];
    ws[threadIdx.x] = w[t * topk + threadIdx.x];
  }
  __syncthreads();
  for (int c = threadIdx.x * 16; c < H; c += blockDim.x * 16) {
    float s[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) s[i] = 0.f;
    for (int k = 0; k < topk; ++k) {
      const uint4 v = *reinterpret_cast<const uint4*>(y8 + (size_t)ps[k] * H + c);
      const float f = ws[k] * y8s[(size_t)ps[k] * (H / BN) + c / BN];
      const __nv_fp8x4_e4m3* q = reinterpret_cast<const __nv_fp8x4_e4m3*>(&v);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float4 d = static_cast<float4>(q[i]);
        s[4 * i] += f * d.x; s[4 * i + 1] += f * d.y; s[4 * i + 2] += f * d.z; s[4 * i + 3] += f * d.w;
      }
    }
    uint4 o[2];
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(o);
#pragma unroll
    for (int i = 0; i < 8; ++i) ob[i] = __floats2bfloat162_rn(s[2 * i], s[2 * i + 1]);
    *reinterpret_cast<uint4*>(out + (size_t)t * H + c) = o[0];
    *reinterpret_cast<uint4*>(out + (size_t)t * H + c + 8) = o[1];
  }
}

// One block per token: out[t] = sum_k w[t, k] * y[pos[t*topk + k]], fp32 accumulation.
__global__ void finalize_kernel(const __nv_bfloat16* __restrict__ y, const int* __restrict__ pos,
                                const float* __restrict__ w, __nv_bfloat16* __restrict__ out, int H, int topk) {
  const int t = blockIdx.x;
  __shared__ int ps[16];
  __shared__ float ws[16];
  if (threadIdx.x < topk) {
    ps[threadIdx.x] = pos[t * topk + threadIdx.x];
    ws[threadIdx.x] = w[t * topk + threadIdx.x];
  }
  __syncthreads();
  for (int c = threadIdx.x * 8; c < H; c += blockDim.x * 8) {
    float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < topk; ++k) {
      const uint4 v = *reinterpret_cast<const uint4*>(y + (size_t)ps[k] * H + c);
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(b[i]);
        s[2 * i] += ws[k] * f.x;
        s[2 * i + 1] += ws[k] * f.y;
      }
    }
    uint4 o;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) ob[i] = __floats2bfloat162_rn(s[2 * i], s[2 * i + 1]);
    *reinterpret_cast<uint4*>(out + (size_t)t * H + c) = o;
  }
}

template <bool FC1>
void launch(const Args& a, int n_tiles_n, int n_tiles) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(moe_gemm<FC1>, cudaFuncAttributeMaxDynamicSharedMemorySize, STAGES * STAGE_BYTES);
    attr = true;
  }
  moe_gemm<FC1><<<dim3(n_tiles_n, n_tiles), THREADS, STAGES * STAGE_BYTES, at::cuda::getCurrentCUDAStream()>>>(a);
}

}  // namespace

void fc1(torch::Tensor xq, torch::Tensor xs, torch::Tensor rows, torch::Tensor expert_off, torch::Tensor tiles,
         torch::Tensor w13, torch::Tensor s13, torch::Tensor g13, torch::Tensor a2, torch::Tensor hq, torch::Tensor hs,
         double limit) {
  Args a{};
  a.a = xq.data_ptr<uint8_t>(); a.as = xs.data_ptr<uint8_t>(); a.rows = rows.data_ptr<int>();
  a.expert_off = expert_off.data_ptr<int>(); a.tiles = reinterpret_cast<const int2*>(tiles.data_ptr<int>());
  a.w = w13.data_ptr<uint8_t>(); a.ws = s13.data_ptr<uint8_t>(); a.alpha = g13.data_ptr<float>(); a.a2 = a2.data_ptr<float>();
  a.hq = hq.data_ptr<uint8_t>(); a.hs = hs.data_ptr<uint8_t>();
  a.K = xq.size(1) * 2; a.N = w13.size(1); a.I = w13.size(1) / 2; a.E = w13.size(0); a.limit = (float)limit;
  TORCH_CHECK(a.I % (BN / 2) == 0 && a.K % (BKB * 2) == 0);
  launch<true>(a, a.I / (BN / 2), tiles.size(0));
}

void fc2(torch::Tensor hq, torch::Tensor hs, torch::Tensor expert_off, torch::Tensor tiles, torch::Tensor w2,
         torch::Tensor s2, torch::Tensor g2, torch::Tensor y, c10::optional<torch::Tensor> y8s) {
  Args a{};
  a.a = hq.data_ptr<uint8_t>(); a.as = hs.data_ptr<uint8_t>();
  a.expert_off = expert_off.data_ptr<int>(); a.tiles = reinterpret_cast<const int2*>(tiles.data_ptr<int>());
  a.w = w2.data_ptr<uint8_t>(); a.ws = s2.data_ptr<uint8_t>(); a.alpha = g2.data_ptr<float>();
  if (y8s.has_value()) {  // y is e4m3 [R, H] as u8
    a.y8 = y.data_ptr<uint8_t>(); a.y8s = y8s->data_ptr<float>();
  } else {
    a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  }
  a.K = hq.size(1) * 2; a.N = w2.size(1); a.I = a.K; a.E = w2.size(0);
  TORCH_CHECK(a.N % BN == 0 && a.K % (BKB * 2) == 0);
  launch<false>(a, a.N / BN, tiles.size(0));
}

void finalize(torch::Tensor y, torch::Tensor pos, torch::Tensor w, torch::Tensor out) {
  const int T = out.size(0), H = out.size(1), topk = pos.numel() / T;
  TORCH_CHECK(H % 8 == 0 && topk <= 16);
  finalize_kernel<<<T, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), pos.data_ptr<int>(), w.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, topk);
}

void finalize8(torch::Tensor y8, torch::Tensor y8s, torch::Tensor pos, torch::Tensor w, torch::Tensor out) {
  const int T = out.size(0), H = out.size(1), topk = pos.numel() / T;
  TORCH_CHECK(H % 16 == 0 && topk <= 16);
  finalize8_kernel<<<T, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      y8.data_ptr<uint8_t>(), y8s.data_ptr<float>(), pos.data_ptr<int>(), w.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, topk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("finalize8", &finalize8);
  m.def("fc1", &fc1);
  m.def("fc2", &fc2, py::arg("hq"), py::arg("hs"), py::arg("expert_off"), py::arg("tiles"), py::arg("w2"), py::arg("s2"), py::arg("g2"), py::arg("y"), py::arg("y8s") = py::none());
  m.def("finalize", &finalize);
}
