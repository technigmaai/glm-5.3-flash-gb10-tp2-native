// SPDX-License-Identifier: Apache-2.0
// megadense4: W4A16 dense GEMM for decode-sized batches on GB10 (sm_121a),
// y[m][n] = gscale * sum_k x[m][k] w[n][k], M <= 64, and the dequantization
// that larger batches use to hand the weight to cuBLAS.
//
// Weights are NVFP4 (e2m1 values, e4m3 scale per 16 k, one fp32 global scale)
// that dense_fp8.py quantizes at load and stores in megamoe.cu's tiled layout:
//   w [N/16][K/128][half][lane][16 B], half 0 row g and half 1 row g+8 of the
//     tile (lane = 4g + t), lane t holding bytes 64c + 16t.. of the row, i.e.
//     k 32t..32t+31 of chunk c, low nibble first;
//   s [N/16][K/128][lane] u32 = {row g: blocks 2t, 2t+1; row g+8: same}.
// A warp's load is then 512 contiguous bytes, which is what reaches 230 GB/s.
// The math is megamoe.cu's chunk_mma: fp4 x e4m3 is exact in fp16, and x
// stays 16-bit.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

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

__device__ __forceinline__ uint4 bf16x8_to_f16x8(uint4 v) {
  uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&w[i]));
    __half2 h = __float22half2_rn(f);
    w[i] = *reinterpret_cast<uint32_t*>(&h);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

// The dequantized A fragments of mma step j for rows g and g+8.
__device__ __forceinline__ void a_frag(uint32_t a[4], const uint4& w0, const uint4& w1, uint32_t sc, int j) {
  const uint32_t s_lo = e4m3x2(sc & 0xffff), s_hi = e4m3x2(sc >> 16);
  const int blk = j >> 2;
  const uint32_t sg = bcast(s_lo, blk), sg8 = bcast(s_hi, blk);
  const uint32_t p0 = (word(w0, j >> 1) >> ((j & 1) * 16)) & 0xffff;
  const uint32_t p1 = (word(w1, j >> 1) >> ((j & 1) * 16)) & 0xffff;
  a[0] = hmul2(e2m1x2(p0 & 0xff), sg);
  a[1] = hmul2(e2m1x2(p1 & 0xff), sg8);
  a[2] = hmul2(e2m1x2(p0 >> 8), sg);
  a[3] = hmul2(e2m1x2(p1 >> 8), sg8);
}

__device__ __forceinline__ void load_x(uint4 X[4], const __nv_bfloat16* x, int tok, int M, int K, int k) {
  if (tok >= M) {
#pragma unroll
    for (int i = 0; i < 4; ++i) X[i] = make_uint4(0, 0, 0, 0);
    return;
  }
  const uint4* p = reinterpret_cast<const uint4*>(x + (size_t)tok * K + k);
#pragma unroll
  for (int i = 0; i < 4; ++i) X[i] = bf16x8_to_f16x8(__ldg(p + i));
}

// One block: TILES 16-row tiles, KSPLIT warps per tile splitting K. Each warp
// keeps the next chunk's weights in flight during this chunk's math.
template <int TILES, int KSPLIT, int MT>
__global__ void __launch_bounds__(TILES * KSPLIT * 32) dense_w4(
    const uint8_t* __restrict__ w, const uint32_t* __restrict__ s, const float* __restrict__ gs,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, int M, int K, int N) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int tl = warp % TILES, ks = warp / TILES;
  const int tile = blockIdx.x * TILES + tl;
  const bool live = tile * 16 < N;
  const int nchunk = K / 128;
  const uint8_t* w0 = w + (size_t)tile * nchunk * 1024 + 16 * lane;
  const uint32_t* sp = s + (size_t)tile * nchunk * 32 + lane;
  float c[MT][4] = {};
  const int c0 = nchunk * ks / KSPLIT, c1 = live ? nchunk * (ks + 1) / KSPLIT : c0;
  uint4 A0{}, A8{};
  uint32_t S = 0;
  if (c0 < c1) { A0 = ld_stream(w0 + (size_t)c0 * 1024); A8 = ld_stream(w0 + (size_t)c0 * 1024 + 512); S = __ldg(sp + c0 * 32); }
  for (int cc = c0; cc < c1; ++cc) {
    if constexpr (MT <= 4) {
      uint4 X[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) load_x(X[mt], x, mt * 8 + g, M, K, cc * 128 + 32 * t);
      const uint4 cA0 = A0, cA8 = A8;
      const uint32_t cS = S;
      if (cc + 1 < c1) {
        A0 = ld_stream(w0 + (size_t)(cc + 1) * 1024); A8 = ld_stream(w0 + (size_t)(cc + 1) * 1024 + 512);
        S = __ldg(sp + (cc + 1) * 32);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint32_t a[4];
        a_frag(a, cA0, cA8, cS, j);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint4& xv = X[mt][j >> 1];
          const int wi = (j & 1) * 2;
          mma16816(c[mt], a, word(xv, wi), word(xv, wi + 1));
        }
      }
    } else {
      // Past 4 tiles, all of them holding x for the whole chunk runs out of
      // registers. Decode the chunk's weights once, then run the rows in two
      // halves of up to 4 tiles each, loading x one half at a time.
      const uint4 cA0 = A0, cA8 = A8;
      const uint32_t cS = S;
      if (cc + 1 < c1) {
        A0 = ld_stream(w0 + (size_t)(cc + 1) * 1024); A8 = ld_stream(w0 + (size_t)(cc + 1) * 1024 + 512);
        S = __ldg(sp + (cc + 1) * 32);
      }
      uint32_t a[8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j) a_frag(a[j], cA0, cA8, cS, j);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint4 X[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          if (h * 4 + i < MT) load_x(X[i], x, (h * 4 + i) * 8 + g, M, K, cc * 128 + 32 * t);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            if (h * 4 + i < MT)
              mma16816(c[h * 4 + i], a[j], word(X[i][j >> 1], (j & 1) * 2), word(X[i][j >> 1], (j & 1) * 2 + 1));
      }
    }
  }
  if constexpr (KSPLIT > 1) {
    __shared__ float red[KSPLIT - 1][TILES][32][MT * 4];
    if (ks > 0)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[ks - 1][tl][lane][mt * 4 + i] = c[mt][i];
    __syncthreads();
    if (ks > 0 || !live) return;
#pragma unroll
    for (int r = 0; r < KSPLIT - 1; ++r)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[mt][i] += red[r][tl][lane][mt * 4 + i];
  } else if (!live) {
    return;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int n = tile * 16 + g + (i >> 1) * 8;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int m = mt * 8 + 2 * t + (i & 1);
      if (m < M) y[(size_t)m * N + n] = __float2bfloat16(c[mt][i] * *gs);
    }
  }
}

// Tiled NVFP4 -> row-major bf16 [N][K], for batches too large for dense_w4.
__global__ void dequant_w4(const uint8_t* __restrict__ w, const uint32_t* __restrict__ s, const float* __restrict__ gs,
                           __nv_bfloat16* __restrict__ out, int K, int N) {
  const int nchunk = K / 128;
  const int tile = blockIdx.x, cc = blockIdx.y, lane = threadIdx.x & 31, half = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3, row = tile * 16 + half * 8 + g;
  const float gscale = *gs;
  const uint4 v = *reinterpret_cast<const uint4*>(w + ((size_t)tile * nchunk + cc) * 1024 + half * 512 + 16 * lane);
  const uint32_t sc = s[((size_t)tile * nchunk + cc) * 32 + lane];
  const uint32_t two = half ? (sc >> 16) : (sc & 0xffff);
  const float f0 = __half2float(__ushort_as_half((unsigned short)(e4m3x2(two) & 0xffff))) * gscale;
  const float f1 = __half2float(__ushort_as_half((unsigned short)(e4m3x2(two) >> 16))) * gscale;
  __nv_bfloat16* o = out + (size_t)row * K + cc * 128 + 32 * t;
#pragma unroll
  for (int b = 0; b < 16; ++b) {  // 16 bytes = 32 k; blocks of 16 k: first 8 bytes block 2t, rest 2t+1
    const uint32_t byte = (word(v, b >> 2) >> ((b & 3) * 8)) & 0xff;
    const uint32_t hv = e2m1x2(byte);
    const __half2 h = *reinterpret_cast<const __half2*>(&hv);
    const float f = b < 8 ? f0 : f1;
    o[2 * b] = __float2bfloat16(__low2float(h) * f);
    o[2 * b + 1] = __float2bfloat16(__high2float(h) * f);
  }
}

}  // namespace

// x [M, K] bf16, w/s tiled (see above), gscale [1] fp32 on the device, y [M, N] bf16. variant: tens digit
// KSPLIT choice, units digit TILES choice; 0 picks from K.
void megadense4(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor gscale, torch::Tensor y,
                int64_t variant) {
  const int M = x.size(0), K = x.size(1), N = y.size(1);
  TORCH_CHECK(M >= 1 && M <= 64 && K % 128 == 0 && N % 16 == 0 && x.is_contiguous() && y.is_contiguous());
  TORCH_CHECK(w.numel() == (int64_t)N * K / 2 && s.numel() * 4 == (int64_t)N * K / 16);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto tiles, auto ksplit, auto mt) {
    constexpr int T = decltype(tiles)::value, KS = decltype(ksplit)::value, MT = decltype(mt)::value;
    dense_w4<T, KS, MT><<<(N / 16 + T - 1) / T, T * KS * 32, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(w.data_ptr()), reinterpret_cast<const uint32_t*>(s.data_ptr()),
        gscale.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, K, N);
  };
  using std::integral_constant;
  auto go = [&](auto mt) {
    int v = variant ? variant : (K >= 4096 ? 81 : K >= 2048 ? 41 : 21);
    switch (v) {
      case 81: launch(integral_constant<int, 1>{}, integral_constant<int, 8>{}, mt); break;
      case 82: launch(integral_constant<int, 2>{}, integral_constant<int, 8>{}, mt); break;
      case 41: launch(integral_constant<int, 1>{}, integral_constant<int, 4>{}, mt); break;
      case 42: launch(integral_constant<int, 2>{}, integral_constant<int, 4>{}, mt); break;
      case 44: launch(integral_constant<int, 4>{}, integral_constant<int, 4>{}, mt); break;
      case 21: launch(integral_constant<int, 1>{}, integral_constant<int, 2>{}, mt); break;
      case 24: launch(integral_constant<int, 4>{}, integral_constant<int, 2>{}, mt); break;
      default: launch(integral_constant<int, 4>{}, integral_constant<int, 1>{}, mt); break;
    }
  };
  if (M <= 8) go(integral_constant<int, 1>{});
  else if (M <= 16) go(integral_constant<int, 2>{});
  else if (M <= 24) go(integral_constant<int, 3>{});
  else if (M <= 32) go(integral_constant<int, 4>{});
  else if (M <= 40) go(integral_constant<int, 5>{});
  else if (M <= 48) go(integral_constant<int, 6>{});
  else if (M <= 56) go(integral_constant<int, 7>{});
  else go(integral_constant<int, 8>{});
}

void dequant4(torch::Tensor w, torch::Tensor s, torch::Tensor gscale, torch::Tensor out) {
  const int N = out.size(0), K = out.size(1);
  dequant_w4<<<dim3(N / 16, K / 128), 64, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(w.data_ptr()), reinterpret_cast<const uint32_t*>(s.data_ptr()),
      gscale.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), K, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &megadense4);
  m.def("dequant", &dequant4);
}
