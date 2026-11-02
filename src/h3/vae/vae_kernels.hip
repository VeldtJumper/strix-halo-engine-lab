// HIP kernels of the H3 ViT3D video decoder (gfx1151, wave32, v_wmma_f32_16x16x16_f16).
// All kernels are deterministic and batch invariant: every output element's reduction order is fixed by its
// own index (row / column / head), never by the row count, the grid or the rank that runs it.
//
// FP16 rounding points follow the reference module (see vae_ref.hpp): one FP32 computation and one FP16
// rounding per tensor op; linears accumulate K16 tiles in ascending order into one FP32 accumulator.

#include "h3/vae/vae_kernels.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::h3::vae::gpu {

namespace {
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("h3 vae hip: ") + what + ": " + hipGetErrorString(e));
}
void launched(const char* what) { check(hipGetLastError(), what); }

// 16 halves at a 16-byte aligned address (LDS rows are padded to 16-byte multiples, not 32).
__device__ __forceinline__ h16 ld16(const _Float16* p) {
    const u32x4 lo = *reinterpret_cast<const u32x4*>(p), hi = *reinterpret_cast<const u32x4*>(p + 8);
    typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
    return __builtin_bit_cast(h16, u32x8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w});
}
__device__ __forceinline__ void st16(_Float16* p, h16 v) {
    typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
    const u32x8 u = __builtin_bit_cast(u32x8, v);
    *reinterpret_cast<u32x4*>(p) = u32x4{u[0], u[1], u[2], u[3]};
    *reinterpret_cast<u32x4*>(p + 8) = u32x4{u[4], u[5], u[6], u[7]};
}

__device__ __forceinline__ float h(const _Float16 x) { return static_cast<float>(x); }
__device__ __forceinline__ _Float16 r16(const float x) { return static_cast<_Float16>(x); }
// The reference sanitizes the FP16 attention output: NaN -> 0, +-Inf -> +-65504 (largest finite FP16).
__device__ __forceinline__ _Float16 nan_to_num16(const _Float16 v) {
    const unsigned short b = __builtin_bit_cast(unsigned short, v);
    if ((b & 0x7c00U) != 0x7c00U) return v;
    if (b & 0x03ffU) return r16(0.0F);
    return __builtin_bit_cast(_Float16, static_cast<unsigned short>((b & 0x8000U) | 0x7bffU));
}

// Block-wide FP32 sum, fixed tree: lane-strided partials, wave shuffles (xor 16..1), then the waves in order.
template <unsigned Threads> __device__ float block_sum(float v, float* red) {
    for (unsigned d = 16; d; d >>= 1U) v += __shfl_xor(v, int(d), 32);
    const unsigned w = threadIdx.x / 32U;
    if (threadIdx.x % 32U == 0U) red[w] = v;
    __syncthreads();
    float s = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < Threads / 32U; ++i) s += red[i];
    __syncthreads();
    return s;
}

// ---------------------------------------------------------------------------------------------------------
// RMSNorm over 2048 (weighted) and LayerNorm over 2048 (weight + bias): one 256-thread block per row,
// each thread owns 8 consecutive features.
__global__ __launch_bounds__(256) void rms_norm_kernel(const _Float16* __restrict__ x, const _Float16* __restrict__ w,
                                                       _Float16* __restrict__ y, unsigned width, float eps) {
    __shared__ float red[8];
    const std::size_t row = blockIdx.x;
    const unsigned c0 = threadIdx.x * 8U;
    float v[8];
    float ss = 0.0F;
    const h8 xv = *reinterpret_cast<const h8*>(x + row * width + c0);
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) { v[k] = h(xv[k]); ss += v[k] * v[k]; }
    const float total = block_sum<256>(ss, red);
    const float inv = 1.0F / sqrtf(total / float(width) + eps);
    const h8 wv = *reinterpret_cast<const h8*>(w + c0);
    h8 out;
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) out[k] = r16((v[k] * inv) * h(wv[k]));
    *reinterpret_cast<h8*>(y + row * width + c0) = out;
}

__global__ __launch_bounds__(256) void layer_norm_kernel(const _Float16* __restrict__ x, const _Float16* __restrict__ w,
                                                         const _Float16* __restrict__ b, _Float16* __restrict__ y,
                                                         unsigned width, float eps) {
    __shared__ float red[8];
    const std::size_t row = blockIdx.x;
    const unsigned c0 = threadIdx.x * 8U;
    float v[8];
    float s = 0.0F;
    const h8 xv = *reinterpret_cast<const h8*>(x + row * width + c0);
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) { v[k] = h(xv[k]); s += v[k]; }
    const float mu = block_sum<256>(s, red) / float(width);
    float q = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) { const float d = v[k] - mu; q += d * d; }
    const float var = block_sum<256>(q, red) / float(width);
    const float inv = 1.0F / sqrtf(var + eps);
    const h8 wv = *reinterpret_cast<const h8*>(w + c0), bv = *reinterpret_cast<const h8*>(b + c0);
    h8 out;
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) out[k] = r16(((v[k] - mu) * inv) * h(wv[k]) + h(bv[k]));
    *reinterpret_cast<h8*>(y + row * width + c0) = out;
}

// ---------------------------------------------------------------------------------------------------------
// GEMM: C[m][n] = sum_k A[m][k] W[n][k] (+ epilogue), A FP16 [M][K] (row pitch lda), W FP16 [N][K].
// Block 128 x 128, 8 waves (4 x 2) of 32 x 64 (2 x 4 accumulators), K stages of 32 halves, LDS rows padded
// to 40 halves; global loads of stage s + 1 are in registers while stage s is multiplied.
// N % 128 == 0, K % 32 == 0; any M (A rows past the end read the last row, their outputs are not stored).
constexpr unsigned kGBM = 128U, kGBN = 128U, kGBK = 32U, kGPitch = 40U;

template <unsigned Mode>  // 0: C = fp16(acc + b); 1: X = fp16(X + fp16(acc + b) * scale[n]); 2: C = fp16(fp16(acc + b) + R[m][n]) (R = scale, pitch ldc)
__global__ __launch_bounds__(256) void gemm_kernel(const _Float16* __restrict__ A, unsigned lda, const _Float16* __restrict__ W,
                                                   const _Float16* __restrict__ bias, const _Float16* __restrict__ scale,
                                                   _Float16* __restrict__ C, unsigned ldc, unsigned M, unsigned N, unsigned K) {
    __shared__ __align__(16) _Float16 as[kGBM * kGPitch];
    __shared__ __align__(16) _Float16 bs[kGBN * kGPitch];
    const unsigned t = threadIdx.x, lane = t % 32U, wave = t / 32U;
    const unsigned wm = wave / 2U, wn = wave % 2U;
    const unsigned m0 = blockIdx.y * kGBM, n0 = blockIdx.x * kGBN;
    // staging: thread t loads rows t / 4 and t / 4 + 64, halves 8 (t % 4) .. + 7 of A and of W
    const unsigned srow = t / 4U, scol = 8U * (t % 4U);
    const _Float16* a0 = A + std::size_t(min(m0 + srow, M - 1U)) * lda + scol;
    const _Float16* a1 = A + std::size_t(min(m0 + srow + 64U, M - 1U)) * lda + scol;
    const _Float16* b0 = W + std::size_t(n0 + srow) * K + scol;
    const _Float16* b1 = W + std::size_t(n0 + srow + 64U) * K + scol;
    u32x4 ra0 = *reinterpret_cast<const u32x4*>(a0), ra1 = *reinterpret_cast<const u32x4*>(a1);
    u32x4 rb0 = *reinterpret_cast<const u32x4*>(b0), rb1 = *reinterpret_cast<const u32x4*>(b1);
    f8 acc[2][4];
#pragma unroll
    for (unsigned i = 0; i < 2U; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) acc[i][j] = f8{};
    const unsigned stages = K / kGBK;
    for (unsigned s = 0; s < stages; ++s) {
        *reinterpret_cast<u32x4*>(as + srow * kGPitch + scol) = ra0;
        *reinterpret_cast<u32x4*>(as + (srow + 64U) * kGPitch + scol) = ra1;
        *reinterpret_cast<u32x4*>(bs + srow * kGPitch + scol) = rb0;
        *reinterpret_cast<u32x4*>(bs + (srow + 64U) * kGPitch + scol) = rb1;
        __syncthreads();
        if (s + 1U < stages) {
            const std::size_t o = std::size_t(s + 1U) * kGBK;
            ra0 = *reinterpret_cast<const u32x4*>(a0 + o); ra1 = *reinterpret_cast<const u32x4*>(a1 + o);
            rb0 = *reinterpret_cast<const u32x4*>(b0 + o); rb1 = *reinterpret_cast<const u32x4*>(b1 + o);
        }
#pragma unroll
        for (unsigned kt = 0; kt < 2U; ++kt) {
            h16 bf[4];
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) bf[j] = ld16(bs + (wn * 64U + 16U * j + lane % 16U) * kGPitch + 16U * kt);
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) {
                const h16 af = ld16(as + (wm * 32U + 16U * i + lane % 16U) * kGPitch + 16U * kt);
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf[j], acc[i][j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < 2U; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const unsigned n = n0 + wn * 64U + 16U * j + lane % 16U;
            const float bn = bias ? h(bias[n]) : 0.0F;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned m = m0 + wm * 32U + 16U * i + 2U * e + lane / 16U;
                if (m >= M) continue;
                const float y = acc[i][j][e] + bn;
                _Float16* c = C + std::size_t(m) * ldc + n;
                if constexpr (Mode == 0U) *c = r16(y);
                else if constexpr (Mode == 1U) *c = r16(h(*c) + h(r16(y)) * h(scale[n]));
                else *c = r16(h(r16(y)) + h(scale[std::size_t(m) * ldc + n]));  // Mode 2: + residual matrix
            }
        }
}

// ---------------------------------------------------------------------------------------------------------
// Tiled GEMM (default): the structure of the engine's tuned WMMA tilings - block BM x BN, waves of 64 x 64
// (4 x 4 accumulators, every A and B fragment reused 4x), 32 K halves per stage (64-byte LDS rows, 16-byte chunks
// XOR-swizzled by (row / 2) % 4), two LDS stages with the next stage's global loads in registers while the
// current one is multiplied, one LDS-only barrier per stage. Same operands, same ascending K16 order into one FP32
// accumulator per output as gemm_kernel: the two kernels give the same bits (checked by the GPU test).
// N % BN == 0, K % 64 == 0; any M (rows past the end read the last row, not stored).
__device__ __forceinline__ unsigned swz(unsigned row, unsigned chunk) { return row * 32U + ((chunk ^ ((row / 2U) % 4U)) * 8U); }
__device__ __forceinline__ void lds_barrier() { __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
__device__ __forceinline__ u32x4 g16(const _Float16* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) u32x4*>(reinterpret_cast<std::uintptr_t>(p));
}

template <unsigned BM, unsigned BN, unsigned Mode, bool Conv = false>
__global__ __launch_bounds__((BM / 64U) * (BN / 64U) * 32U) void gemm_tiled_kernel(const _Float16* __restrict__ A, unsigned lda,
    const _Float16* __restrict__ W, const _Float16* __restrict__ bias, const _Float16* __restrict__ scale, _Float16* __restrict__ C,
    unsigned ldc, unsigned M, unsigned N, unsigned K, gpu::ConvGeom geo) {
    constexpr unsigned WavesN = BN / 64U, Threads = (BM / 64U) * WavesN * 32U, StageH = (BM + BN) * 32U;
    constexpr unsigned AChunks = BM * 4U / Threads, BChunks = BN * 4U / Threads;
    __shared__ __align__(16) _Float16 lds[2U * StageH];
    const unsigned thread = threadIdx.x, lane = thread % 32U;
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(thread / 32U)));
    const unsigned wm = wave / WavesN, wn = wave % WavesN;
    const unsigned tiles_m = (M + BM - 1U) / BM;
    // consecutive blocks take all row tiles of one weight tile (the weight tile stays in L2)
    const unsigned m0 = (blockIdx.x % tiles_m) * BM, n0 = (blockIdx.x / tiles_m) * BN;
    unsigned stages = K / 32U;
    asm volatile("" : "+s"(stages));
    std::uint64_t aoff[AChunks], boff[BChunks];
#pragma unroll
    for (unsigned i = 0; i < AChunks; ++i) {
        const unsigned c = thread + Threads * i;
        aoff[i] = std::uint64_t(min(m0 + c / 4U, M - 1U)) * lda + 8U * (c % 4U);
    }
#pragma unroll
    for (unsigned i = 0; i < BChunks; ++i) {
        const unsigned c = thread + Threads * i;
        boff[i] = std::uint64_t(n0 + c / 4U) * K + 8U * (c % 4U);
    }
    u32x4 areg[AChunks], breg[BChunks];
    // implicit-GEMM conv (Conv): row m = (t, oy, ox) of the output frames, column k = tap * C + ci with
    // tap = (kt * k + ky) * k + kx, exactly the im2col column order; 8 consecutive k stay inside one tap (C % 8 == 0)
    int ct[Conv ? AChunks : 1], cy[Conv ? AChunks : 1], cx[Conv ? AChunks : 1];
    if constexpr (Conv) {
#pragma unroll
        for (unsigned i = 0; i < AChunks; ++i) {
            const unsigned c = thread + Threads * i;
            const int m = int(min(m0 + c / 4U, M - 1U)), P = geo.Ho * geo.Wo;
            ct[i] = m / P; cy[i] = (m % P) / geo.Wo; cx[i] = m % geo.Wo;
        }
    }
    const auto fetch = [&](unsigned s) {
#pragma unroll
        for (unsigned i = 0; i < AChunks; ++i) {
            if constexpr (Conv) {
                const int kk = int(32U * s + 8U * ((thread + Threads * i) % 4U));
                const int tap = kk / geo.C, ci = kk % geo.C, kk2 = geo.k * geo.k;
                const int tt = tap / kk2, ky = (tap / geo.k) % geo.k, kx = tap % geo.k;
                const int it = ct[i] * geo.ts + tt - geo.front;
                if (kk >= geo.Kreal || it < 0 || it >= geo.T) { areg[i] = u32x4{0U, 0U, 0U, 0U}; continue; }
                const int pad = geo.ss == 1 ? geo.k / 2 : 0;
                int iy = cy[i] * geo.ss + ky - pad, ix = cx[i] * geo.ss + kx - pad;
                iy = iy < 0 ? -iy : (iy >= geo.H ? 2 * geo.H - 2 - iy : iy);
                ix = ix < 0 ? -ix : (ix >= geo.W ? 2 * geo.W - 2 - ix : ix);
                areg[i] = g16(A + ((std::size_t(it) * geo.H + iy) * geo.W + ix) * geo.C + ci);
            } else {
                areg[i] = g16(A + 32U * s + aoff[i]);
            }
        }
#pragma unroll
        for (unsigned i = 0; i < BChunks; ++i) breg[i] = g16(W + 32U * s + boff[i]);
    };
    const auto store = [&](unsigned buffer) {
        _Float16* as = lds + buffer * StageH;
        _Float16* bs = as + BM * 32U;
#pragma unroll
        for (unsigned i = 0; i < AChunks; ++i) { const unsigned c = thread + Threads * i; *reinterpret_cast<u32x4*>(as + swz(c / 4U, c % 4U)) = areg[i]; }
#pragma unroll
        for (unsigned i = 0; i < BChunks; ++i) { const unsigned c = thread + Threads * i; *reinterpret_cast<u32x4*>(bs + swz(c / 4U, c % 4U)) = breg[i]; }
    };
    const auto frag = [](const _Float16* base, unsigned row, unsigned kt) {
        typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
        const u32x4 lo = *reinterpret_cast<const u32x4*>(base + swz(row, 2U * kt));
        const u32x4 hi = *reinterpret_cast<const u32x4*>(base + swz(row, 2U * kt + 1U));
        return __builtin_bit_cast(h16, u32x8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w});
    };
    f8 acc[4][4];
#pragma unroll
    for (unsigned t = 0; t < 4U; ++t)
#pragma unroll
        for (unsigned u = 0; u < 4U; ++u) acc[t][u] = f8{};
    const auto compute = [&](unsigned buffer) {
        const _Float16* as = lds + buffer * StageH;
        const _Float16* bs = as + BM * 32U;
#pragma unroll
        for (unsigned kt = 0; kt < 2U; ++kt) {
            h16 b[4];
#pragma unroll
            for (unsigned u = 0; u < 4U; ++u) b[u] = frag(bs, wn * 64U + 16U * u + lane % 16U, kt);
#pragma unroll
            for (unsigned t = 0; t < 4U; ++t) {
                const h16 a = frag(as, wm * 64U + 16U * t + lane % 16U, kt);
#pragma unroll
                for (unsigned u = 0; u < 4U; ++u) acc[t][u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[u], acc[t][u]);
            }
        }
    };
    fetch(0U);
    store(0U);
    lds_barrier();
#pragma unroll 1
    for (unsigned s = 0; s < stages; s += 2U) {  // stages even (K % 64 == 0): even stages buffer 0, odd buffer 1
        fetch(s + 1U);
        compute(0U);
        store(1U);
        lds_barrier();
        const bool more = s + 2U < stages;
        if (more) fetch(s + 2U);
        compute(1U);
        if (more) store(0U);
        lds_barrier();
    }
    if constexpr (Mode == 3U) {
        // fused gated SiLU (weights in swiglu_source_row order): tiles u, u + 1 of a wave = gate, up of 16 columns;
        // g = fp16(fp16(silu(fp16(gate + bg))) * fp16(up + bu)), the same roundings as gemm + swiglu_kernel
#pragma unroll
        for (unsigned t = 0; t < 4U; ++t)
#pragma unroll
            for (unsigned u = 0; u < 4U; u += 2U) {
                const unsigned pg = n0 + wn * 64U + 16U * u + lane % 16U;
                const float bg = h(bias[pg]), bu = h(bias[pg + 16U]);
                const unsigned col = n0 / 2U + wn * 32U + 8U * u + lane % 16U;
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    const unsigned m = m0 + wm * 64U + 16U * t + 2U * e + lane / 16U;
                    if (m >= M) continue;
                    const float gate = h(r16(acc[t][u][e] + bg)), up = h(r16(acc[t][u + 1U][e] + bu));
                    const float si = h(r16(gate / (1.0F + expf(-gate))));
                    C[std::size_t(m) * ldc + col] = r16(si * up);
                }
            }
        return;
    }
#pragma unroll
    for (unsigned t = 0; t < 4U; ++t)
#pragma unroll
        for (unsigned u = 0; u < 4U; ++u) {
            const unsigned n = n0 + wn * 64U + 16U * u + lane % 16U;
            const float bn = bias ? h(bias[n]) : 0.0F;
            const float sc = Mode == 1U ? h(scale[n]) : 0.0F;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned m = m0 + wm * 64U + 16U * t + 2U * e + lane / 16U;
                if (m >= M) continue;
                const float y = acc[t][u][e] + bn;
                _Float16* c = C + std::size_t(m) * ldc + n;
                if constexpr (Mode == 0U) *c = r16(y);
                else if constexpr (Mode == 1U) *c = r16(h(*c) + h(r16(y)) * sc);
                else *c = r16(h(r16(y)) + h(scale[std::size_t(m) * ldc + n]));  // Mode 2: + residual matrix
            }
        }
}

// STRIX_H3_VAE_GEMM=0: the simple 128 x 128 kernel (same bits, slower).
int tiled_override = -1;
bool use_tiled() {
    static const bool on = [] { const char* v = std::getenv("STRIX_H3_VAE_GEMM"); return !v || v[0] != '0'; }();
    return tiled_override < 0 ? on : tiled_override != 0;
}
// Block tile choice (routing only: every tile gives the same bits, each output's K16 order is fixed).
// 0: 128 x 256 (8 waves), 1: 256 x 256 (16 waves), 2: 256 x 128 (8 waves), 3: 128 x 128 (4 waves).
// STRIX_H3_VAE_TILE=<0..3> forces one for every GEMM; set_gemm_tile() for the bench.
int tile_override = -1;
// Defaults measured at M = 1797 (gfx1151): qkv (N 6144) 256 x 128 26.4 TF; w1 + swiglu (N 16384)
// 128 x 128 30.1 TF; w2 (K 8192) 256 x 256 21.2 TF; out (2048 x 2048) and the rest 128 x 256.
unsigned tile_for(unsigned M, unsigned N, unsigned K, unsigned mode) {
    if (tile_override >= 0) return unsigned(tile_override);
    static const int env = [] { const char* v = std::getenv("STRIX_H3_VAE_TILE"); return v ? std::atoi(v) : -1; }();
    if (env >= 0 && env <= 3) return unsigned(env);
    (void)M;
    if (mode == 3U) return 3U;
    if (N == 6144U) return 2U;
    if (K >= 8192U) return 1U;
    return 0U;
}
template <unsigned BM, unsigned BN, unsigned Mode, bool Conv = false> void launch_bmbn(const _Float16* A, unsigned lda, const _Float16* W,
        const _Float16* bias, const _Float16* scale, _Float16* C, unsigned ldc, unsigned M, unsigned N, unsigned K, hipStream_t st,
        gpu::ConvGeom geo = {}) {
    const unsigned blocks = (M + BM - 1U) / BM * (N / BN);
    gemm_tiled_kernel<BM, BN, Mode, Conv><<<blocks, (BM / 64U) * (BN / 64U) * 32U, 0, st>>>(A, lda, W, bias, scale, C, ldc, M, N, K, geo);
}
template <unsigned Mode> void launch_tiled(const _Float16* A, unsigned lda, const _Float16* W, const _Float16* bias, const _Float16* scale,
                                           _Float16* C, unsigned ldc, unsigned M, unsigned N, unsigned K, hipStream_t st) {
    switch (tile_for(M, N, K, Mode)) {
        case 1U: launch_bmbn<256U, 256U, Mode>(A, lda, W, bias, scale, C, ldc, M, N, K, st); break;
        case 2U: launch_bmbn<256U, 128U, Mode>(A, lda, W, bias, scale, C, ldc, M, N, K, st); break;
        case 3U: launch_bmbn<128U, 128U, Mode>(A, lda, W, bias, scale, C, ldc, M, N, K, st); break;
        default: launch_bmbn<128U, 256U, Mode>(A, lda, W, bias, scale, C, ldc, M, N, K, st); break;
    }
}

// ---------------------------------------------------------------------------------------------------------
// gated SiLU: g[r][n] = fp16(fp16(silu(h[r][n])) * h[r][8192 + n])
__global__ __launch_bounds__(256) void swiglu_kernel(const _Float16* __restrict__ hbuf, _Float16* __restrict__ g, unsigned rows) {
    const std::size_t i = std::size_t(blockIdx.x) * 256U + threadIdx.x;
    if (i >= std::size_t(rows) * 8192U) return;
    const std::size_t r = i / 8192U, n = i % 8192U;
    const float gate = h(hbuf[r * 16384U + n]), up = h(hbuf[r * 16384U + 8192U + n]);
    const float si = h(r16(gate / (1.0F + expf(-gate))));
    g[i] = r16(si * up);
}

// ---------------------------------------------------------------------------------------------------------
// Per-head RMSNorm (unit weight, eps 1e-5) + split-half rope on dims [0, 48) of q and k; v copied.
// qkv [L][6144] (head h: q at 192 h, k at 192 h + 64, v at 192 h + 128) -> Q, K, V [32][Lp][64].
// rope: [L][24] FP16 cos, [L][24] FP16 sin. One thread per (token, head, q|k|v).
__global__ __launch_bounds__(256) void rms_rope_kernel(const _Float16* __restrict__ qkv, const _Float16* __restrict__ rc,
                                                       const _Float16* __restrict__ rs, _Float16* __restrict__ Q,
                                                       _Float16* __restrict__ Kh, _Float16* __restrict__ V, unsigned L, unsigned Lp) {
    const unsigned item = blockIdx.x * 256U + threadIdx.x;
    if (item >= Lp * 32U * 3U) return;
    const unsigned which = item % 3U, hd = (item / 3U) % 32U, tok = item / 96U;
    _Float16* dst = (which == 0U ? Q : which == 1U ? Kh : V) + (std::size_t(hd) * Lp + tok) * 64U;
    if (tok >= L) {  // padding rows [L, Lp): zeroed on every call (no stale values from an earlier, larger call)
#pragma unroll
        for (unsigned d = 0; d < 64U; d += 8U) *reinterpret_cast<h8*>(dst + d) = h8{};
        return;
    }
    const _Float16* src = qkv + std::size_t(tok) * 6144U + hd * 192U + which * 64U;
    if (which == 2U) {
#pragma unroll
        for (unsigned d = 0; d < 64U; d += 8U) *reinterpret_cast<h8*>(dst + d) = *reinterpret_cast<const h8*>(src + d);
        return;
    }
    float n[64];
    float ss = 0.0F;
#pragma unroll
    for (unsigned d = 0; d < 64U; ++d) { n[d] = h(src[d]); ss += n[d] * n[d]; }
    const float inv = 1.0F / sqrtf(ss / 64.0F + 1e-5F);
#pragma unroll
    for (unsigned d = 0; d < 64U; ++d) n[d] = h(r16(n[d] * inv));
#pragma unroll
    for (unsigned i = 0; i < 24U; ++i) {
        const float c = h(rc[tok * 24U + i]), s = h(rs[tok * 24U + i]);
        const float x0 = n[i], x1 = n[i + 24U];
        dst[i] = r16(h(r16(x0 * c)) + x1 * -s);
        dst[i + 24U] = r16(h(r16(x1 * c)) + x0 * s);
    }
#pragma unroll
    for (unsigned d = 48U; d < 64U; ++d) dst[d] = r16(n[d]);
}

// ---------------------------------------------------------------------------------------------------------
// Non-causal attention, head dim 64, one call's L tokens (keys masked past L). Block = 8 waves = 128 queries
// of one head; each wave owns 16 queries. Per 64-key step: S^T = K Q^T (keys on the WMMA rows, queries on
// the columns: each lane holds 8 scores of its own query, the other parity in lane ^ 16), online softmax in
// FP32 (exp2 of scaled scores), P rounded to FP16 for O^T += V^T P^T. Key order is ascending and fixed.
// Q, K, V: [32][Lp][64] (rows L..Lp zero); O: [L][2048], column 64 h + d.
constexpr unsigned kAQ = 128U, kAK = 64U, kAPitch = 72U;
__global__ __launch_bounds__(256) void attention_kernel(const _Float16* __restrict__ Q, const _Float16* __restrict__ Kh,
                                                        const _Float16* __restrict__ V, _Float16* __restrict__ O,
                                                        unsigned L, unsigned Lp) {
    __shared__ __align__(16) _Float16 ks[kAK * kAPitch];   // [key][d]
    __shared__ __align__(16) _Float16 vt[64U * kAPitch];   // [d][key]
    const unsigned t = threadIdx.x, lane = t % 32U, wave = t / 32U, hd = blockIdx.y;
    const unsigned q0 = blockIdx.x * kAQ + wave * 16U;
    const _Float16* qh = Q + std::size_t(hd) * Lp * 64U;
    const _Float16* kh = Kh + std::size_t(hd) * Lp * 64U;
    const _Float16* vh = V + std::size_t(hd) * Lp * 64U;
    const unsigned myq = min(q0 + lane % 16U, Lp - 1U);
    h16 qf[4];
#pragma unroll
    for (unsigned dt = 0; dt < 4U; ++dt) qf[dt] = ld16(qh + std::size_t(myq) * 64U + 16U * dt);
    f8 o[4];
#pragma unroll
    for (unsigned dt = 0; dt < 4U; ++dt) o[dt] = f8{};
    float mrow = -INFINITY, lrow = 0.0F;
    const float sl2 = 0.125F * 1.4426950408889634F;  // 1/sqrt(64) * log2(e)
    const bool upper = lane >= 16U;
    for (unsigned k0 = 0; k0 < L; k0 += kAK) {
        __syncthreads();
        {   // stage 64 keys: thread t loads key t / 4, halves 16 (t % 4) .. + 15 of K and V
            const unsigned kr = t / 4U, kc = 16U * (t % 4U);
            const std::size_t g = std::size_t(k0 + kr) * 64U + kc;  // k0 + kr < Lp
            // keys >= L are staged as zeros (not loaded): the WMMA result must not depend on padding contents, even
            // where P is exactly 0 (a V entry under P = 0 can still move the f32 result by an ulp on this target)
            const bool live = k0 + kr < L;
            const h16 kv = live ? ld16(kh + g) : h16{};
            const h16 vv = live ? ld16(vh + g) : h16{};
            st16(ks + kr * kAPitch + kc, kv);
#pragma unroll
            for (unsigned d = 0; d < 16U; ++d) vt[(kc + d) * kAPitch + kr] = vv[d];
        }
        __syncthreads();
        f8 s[4];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            s[j] = f8{};
#pragma unroll
            for (unsigned dt = 0; dt < 4U; ++dt) {
                const h16 kf = ld16(ks + (16U * j + lane % 16U) * kAPitch + 16U * dt);
                s[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf, qf[dt], s[j]);
            }
        }
        // s[j][e] = score(query lane % 16, key k0 + 16 j + 2 e + lane / 16)
        float mx = mrow;
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned key = k0 + 16U * j + 2U * e + (upper ? 1U : 0U);
                s[j][e] = key < L ? s[j][e] * sl2 : -INFINITY;
                mx = fmaxf(mx, s[j][e]);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float alpha = exp2f(mrow - mx);  // mrow = -inf on the first step: alpha = 0
        float ps = 0.0F;
        h16 pf[4];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const float p = exp2f(s[j][e] - mx);
                s[j][e] = p;
                ps += p;
            }
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const float other = __shfl_xor(s[j][e], 16, 32);
                pf[j][2U * e] = r16(upper ? other : s[j][e]);
                pf[j][2U * e + 1U] = r16(upper ? s[j][e] : other);
            }
        }
        ps += __shfl_xor(ps, 16, 32);
        lrow = lrow * alpha + ps;
        mrow = mx;
#pragma unroll
        for (unsigned dt = 0; dt < 4U; ++dt) {
            o[dt] *= alpha;
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                const h16 vf = ld16(vt + (16U * dt + lane % 16U) * kAPitch + 16U * j);
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf, pf[j], o[dt]);
            }
        }
    }
    // o[dt][e] = O^T[d = 16 dt + 2 e + lane / 16][query lane % 16]
    const unsigned q = q0 + lane % 16U;
    if (q >= L) return;
    const float inv = 1.0F / lrow;
#pragma unroll
    for (unsigned dt = 0; dt < 4U; ++dt)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned d = 16U * dt + 2U * e + (upper ? 1U : 0U);
            O[std::size_t(q) * 2048U + hd * 64U + d] = nan_to_num16(r16(o[dt][e] * inv));
        }
}

// ---------------------------------------------------------------------------------------------------------
// x_embedder (24 -> 2048) on the decoder input [24][P] (channel-major), the reference's two roundings
// (matmul to FP16, then + bias); rows P..P+3 = register tokens, row P+4 = zeros.
__global__ __launch_bounds__(256) void embed_kernel(const _Float16* __restrict__ zin, const _Float16* __restrict__ w,
                                                    const _Float16* __restrict__ b, const _Float16* __restrict__ reg,
                                                    _Float16* __restrict__ x, unsigned P) {
    const std::size_t i = std::size_t(blockIdx.x) * 256U + threadIdx.x;
    if (i >= std::size_t(P + 5U) * 2048U) return;
    const unsigned r = unsigned(i / 2048U), n = unsigned(i % 2048U);
    if (r >= P) {
        x[i] = r < P + 4U ? reg[(r - P) * 2048U + n] : r16(0.0F);
        return;
    }
    float acc[8] = {};
#pragma unroll
    for (unsigned c = 0; c < 24U; ++c) acc[c % 8U] += h(zin[std::size_t(c) * P + r]) * h(w[n * 24U + c]);
    const float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    x[i] = r16(h(r16(s)) + h(b[n]));
}
}  // namespace

void set_gemm_tiled(int on) { tiled_override = on; }
void set_gemm_tile(int tile) { tile_override = tile; }
void rms_norm(const _Float16* x, const _Float16* w, _Float16* y, unsigned rows, hipStream_t st) {
    rms_norm_kernel<<<rows, 256, 0, st>>>(x, w, y, 2048U, 1e-5F);
    launched("rms_norm");
}
void layer_norm(const _Float16* x, const _Float16* w, const _Float16* b, _Float16* y, unsigned rows, hipStream_t st) {
    layer_norm_kernel<<<rows, 256, 0, st>>>(x, w, b, y, 2048U, 1e-5F);
    launched("layer_norm");
}
void gemm(const _Float16* A, unsigned lda, const _Float16* W, const _Float16* bias, _Float16* C, unsigned ldc,
          unsigned M, unsigned N, unsigned K, hipStream_t st) {
    if (N % kGBN || K % kGBK || M == 0) throw std::invalid_argument("h3 vae gemm: N % 128, K % 32");
    if (use_tiled() && N % 256U == 0 && K % 64U == 0 && M > 0) { launch_tiled<0>(A, lda, W, bias, nullptr, C, ldc, M, N, K, st); launched("gemm"); return; }
    gemm_kernel<0><<<dim3(N / kGBN, (M + kGBM - 1U) / kGBM), 256, 0, st>>>(A, lda, W, bias, nullptr, C, ldc, M, N, K);
    launched("gemm");
}
void gemm_residual(const _Float16* A, unsigned lda, const _Float16* W, const _Float16* bias, const _Float16* scale,
                   _Float16* X, unsigned ldx, unsigned M, unsigned N, unsigned K, hipStream_t st) {
    if (N % kGBN || K % kGBK || M == 0) throw std::invalid_argument("h3 vae gemm: N % 128, K % 32");
    if (use_tiled() && N % 256U == 0 && K % 64U == 0 && M > 0) { launch_tiled<1>(A, lda, W, bias, scale, X, ldx, M, N, K, st); launched("gemm_residual"); return; }
    gemm_kernel<1><<<dim3(N / kGBN, (M + kGBM - 1U) / kGBM), 256, 0, st>>>(A, lda, W, bias, scale, X, ldx, M, N, K);
    launched("gemm_residual");
}
void gemm_add(const _Float16* A, unsigned lda, const _Float16* W, const _Float16* bias, const _Float16* R, _Float16* C, unsigned ldc,
              unsigned M, unsigned N, unsigned K, hipStream_t st) {
    if (use_tiled() && N % 256U == 0 && K % 64U == 0 && M > 0) { launch_tiled<2>(A, lda, W, bias, R, C, ldc, M, N, K, st); launched("gemm_add"); return; }
    if (N % kGBN || K % kGBK || M == 0) throw std::invalid_argument("h3 vae gemm: N % 128, K % 32");
    gemm_kernel<2><<<dim3(N / kGBN, (M + kGBM - 1U) / kGBM), 256, 0, st>>>(A, lda, W, bias, R, C, ldc, M, N, K);
    launched("gemm_add");
}
void gemm_swiglu(const _Float16* A, unsigned lda, const _Float16* Wp, const _Float16* bp, _Float16* G, unsigned ldg,
                 unsigned M, unsigned N, unsigned K, hipStream_t st) {
    if (N % 256U || K % 64U || M == 0) throw std::invalid_argument("h3 vae gemm_swiglu: N % 256, K % 64");
    launch_tiled<3>(A, lda, Wp, bp, nullptr, G, ldg, M, N, K, st);
    launched("gemm_swiglu");
}
void conv_gemm(const _Float16* x, const ConvGeom& geo, const _Float16* W, const _Float16* bias, const _Float16* R, _Float16* C,
               unsigned ldc, unsigned M, unsigned N, unsigned Kp, hipStream_t st) {
    if (geo.C % 32 || Kp % 64U || N % 128U || M == 0) throw std::invalid_argument("h3 vae conv_gemm: C % 32, Kp % 64, N % 128");
    // same tile family as the explicit form; routing by N only (any tile gives the same bits)
    if (R) {
        if (N % 256U == 0U) launch_bmbn<128U, 256U, 2U, true>(x, 0U, W, bias, R, C, ldc, M, N, Kp, st, geo);
        else launch_bmbn<128U, 128U, 2U, true>(x, 0U, W, bias, R, C, ldc, M, N, Kp, st, geo);
    } else {
        if (N % 256U == 0U) launch_bmbn<128U, 256U, 0U, true>(x, 0U, W, bias, nullptr, C, ldc, M, N, Kp, st, geo);
        else launch_bmbn<128U, 128U, 0U, true>(x, 0U, W, bias, nullptr, C, ldc, M, N, Kp, st, geo);
    }
    launched("conv_gemm");
}
void swiglu(const _Float16* hbuf, _Float16* g, unsigned rows, hipStream_t st) {
    swiglu_kernel<<<unsigned((std::size_t(rows) * 8192U + 255U) / 256U), 256, 0, st>>>(hbuf, g, rows);
    launched("swiglu");
}
void rms_rope(const _Float16* qkv, const _Float16* rc, const _Float16* rs, _Float16* Q, _Float16* K, _Float16* V,
              unsigned L, unsigned Lp, hipStream_t st) {
    rms_rope_kernel<<<(Lp * 96U + 255U) / 256U, 256, 0, st>>>(qkv, rc, rs, Q, K, V, L, Lp);
    launched("rms_rope");
}
void attention(const _Float16* Q, const _Float16* K, const _Float16* V, _Float16* O, unsigned L, unsigned Lp, hipStream_t st) {
    if (Lp % kAK) throw std::invalid_argument("h3 vae attention: Lp % 64");
    attention_kernel<<<dim3((L + kAQ - 1U) / kAQ, 32U), 256, 0, st>>>(Q, K, V, O, L, Lp);
    launched("attention");
}
void embed(const _Float16* zin, const _Float16* w, const _Float16* b, const _Float16* reg, _Float16* x, unsigned P, hipStream_t st) {
    embed_kernel<<<unsigned((std::size_t(P + 5U) * 2048U + 255U) / 256U), 256, 0, st>>>(zin, w, b, reg, x, P);
    launched("embed");
}

}  // namespace strix::h3::vae::gpu
