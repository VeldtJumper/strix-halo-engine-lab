// MiniMax H3 block glue kernels (see glue.hpp for the contract; arithmetic = ops_ref.hpp / host_ref.cpp).
#include "h3/block/glue.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strix::h3::block::gpu {

namespace {

constexpr std::uint32_t kHidden = 5376, kHeads = 56, kHeadDim = 128, kInner = 7168;

__device__ __forceinline__ float bf(std::uint16_t b) { return __uint_as_float(static_cast<std::uint32_t>(b) << 16U); }
__device__ __forceinline__ std::uint16_t tob(float f) {
    std::uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
    u += 0x7fffU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}

// Lane t of a wave owns elements 32 i + t (i < size = n / 32) of a row; the reference sum of the squares: per lane a
// 4-level cascade (16-step levels for size < 2^16), then lanes l, l + 8, l + 16, l + 24 added in that order, then the
// 8 results in lane order. Returns the total on every lane. (n % 32 == 0.)
template <class Load>
__device__ float wave_ref_sumsq(std::uint32_t n, Load&& load) {
    const unsigned t = threadIdx.x & 31U;
    const std::uint32_t size = n / 32U;
    unsigned cl = 0;
    while ((1U << cl) < size) ++cl;
    unsigned lp = cl / 4U;
    if (lp < 4U) lp = 4U;
    const std::uint32_t step = 1U << lp, mask = step - 1U;
    float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    std::uint32_t i = 0;
    while (i + step <= size) {
        for (std::uint32_t j = 0; j < step; ++j, ++i) {
            const float v = load(32U * i + t);
            acc[0] += v * v;
        }
        for (unsigned l = 1; l < 4U; ++l) {
            acc[l] += acc[l - 1];
            acc[l - 1] = 0.0F;
            if ((i & (mask << (l * lp))) != 0U) break;
        }
    }
    for (; i < size; ++i) {
        const float v = load(32U * i + t);
        acc[0] += v * v;
    }
    float a = acc[0];
    a += acc[1];
    a += acc[2];
    a += acc[3];
    // lanes l (< 8): a_l + a_{l+8} + a_{l+16} + a_{l+24}
    const float a8 = __shfl(a, (t + 8U) & 31U, 32), a16 = __shfl(a, (t + 16U) & 31U, 32), a24 = __shfl(a, (t + 24U) & 31U, 32);
    float p = a;
    p += a8;
    p += a16;
    p += a24;
    float fin = 0.0F;
    for (unsigned l = 0; l < 8U; ++l) fin += __shfl(p, l, 32);
    return fin;
}

__device__ __forceinline__ float rms_inv(float ss, std::uint32_t n, float eps) {
    const float ms = ss / static_cast<float>(n);
    return 1.0F / __builtin_sqrtf(ms + eps);  // llvm.sqrt: correctly rounded with -fhip-fp32-correctly-rounded-divide-sqrt
}

// one wave per row, 8 rows per block
__global__ void __launch_bounds__(256) k_rmsnorm_mod(const std::uint16_t* __restrict__ x, const std::uint16_t* __restrict__ w,
    const std::uint32_t* __restrict__ mod_row, const std::uint16_t* __restrict__ s1, const std::uint16_t* __restrict__ shift,
    std::uint32_t rows, std::uint32_t n, float eps, std::uint16_t* __restrict__ h) {
    const std::uint32_t r = blockIdx.x * 8U + (threadIdx.x >> 5U);
    if (r >= rows) return;
    const unsigned t = threadIdx.x & 31U;
    const std::uint16_t* xr = x + static_cast<std::size_t>(r) * n;
    const float ss = wave_ref_sumsq(n, [&](std::uint32_t e) { return bf(xr[e]); });
    const float inv = rms_inv(ss, n, eps);
    std::uint16_t* hr = h + static_cast<std::size_t>(r) * n;
    const std::size_t mo = s1 ? static_cast<std::size_t>(mod_row[r]) * n : 0;
    for (std::uint32_t e = t; e < n; e += 32U) {
        std::uint16_t y = tob((bf(xr[e]) * inv) * bf(w[e]));
        if (s1) y = tob(bf(tob(bf(y) * bf(s1[mo + e]))) + bf(shift[mo + e]));
        hr[e] = y;
    }
}

__global__ void __launch_bounds__(256) k_gate_residual(std::uint16_t* __restrict__ x, const std::uint16_t* __restrict__ o,
    const std::uint32_t* __restrict__ mod_row, const std::uint16_t* __restrict__ g, std::uint32_t rows, std::uint32_t n) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::uint64_t>(rows) * n) return;
    const std::uint32_t r = static_cast<std::uint32_t>(i / n), c = static_cast<std::uint32_t>(i % n);
    x[i] = tob(bf(x[i]) + bf(o[i]) * bf(g[static_cast<std::size_t>(mod_row[r]) * n + c]));
}

// one wave per (row, head, q|k): 8 waves per block, grid (S, 2 * 56 / 8)
__global__ void __launch_bounds__(256) k_qk_norm_rope(const std::uint16_t* __restrict__ qkv, std::uint32_t S,
    const std::uint16_t* __restrict__ qw, const std::uint16_t* __restrict__ kw, const std::uint16_t* __restrict__ cs,
    const std::uint16_t* __restrict__ sn, float eps, std::uint16_t* __restrict__ q, std::uint16_t* __restrict__ k,
    std::uint32_t* __restrict__ head_sumsq_max) {
    __shared__ std::uint16_t lds[8][kHeadDim];
    const std::uint32_t r = blockIdx.x;
    const unsigned wv = threadIdx.x >> 5U, t = threadIdx.x & 31U;
    const std::uint32_t hk = blockIdx.y * 8U + wv;  // 0 .. 111: q heads then k heads
    const bool is_k = hk >= kHeads;
    const std::uint32_t head = is_k ? hk - kHeads : hk;
    const std::uint16_t* src = qkv + static_cast<std::size_t>(r) * 3U * kInner + (is_k ? kInner : 0U) + head * kHeadDim;
    const std::uint16_t* w = is_k ? kw : qw;
    const float ss = wave_ref_sumsq(kHeadDim, [&](std::uint32_t e) { return bf(src[e]); });
    const float inv = rms_inv(ss, kHeadDim, eps);
    for (std::uint32_t e = t; e < kHeadDim; e += 32U) lds[wv][e] = tob((bf(src[e]) * inv) * bf(w[e]));
    __syncthreads();
    std::uint16_t* dst = (is_k ? k : q) + (static_cast<std::size_t>(r) * kHeads + head) * kHeadDim;
    float osq = 0.0F;
    for (std::uint32_t e = t; e < kHeadDim; e += 32U) {
        std::uint16_t o = lds[wv][e];
        if (cs && e < 96U) {
            const std::uint32_t i = e < 48U ? e : e - 48U;
            const float a = bf(lds[wv][i]), b = bf(lds[wv][48U + i]);
            const float c = bf(cs[static_cast<std::size_t>(r) * 48U + i]), s = bf(sn[static_cast<std::size_t>(r) * 48U + i]);
            o = e < 48U ? tob(bf(tob(a * c)) + b * (-s)) : tob(bf(tob(b * c)) + a * s);
        }
        dst[e] = o;
        osq += bf(o) * bf(o);
    }
    if (head_sumsq_max) {
        for (unsigned d = 16U; d > 0U; d >>= 1U) osq += __shfl_xor(osq, d, 32);
        // non-negative floats order as their bits; read first so that only increases take the atomic
        const std::uint32_t bits = __float_as_uint(osq);
        if (t == 0U && bits > *static_cast<volatile const std::uint32_t*>(head_sumsq_max + hk)) atomicMax(head_sumsq_max + hk, bits);
    }
}

// V^T: block = 128 rows x one head (128 dims) through LDS; 256 threads; 32-bit loads, 256-byte stores per V^T row.
__global__ void __launch_bounds__(256) k_transpose_v(const std::uint16_t* __restrict__ qkv, std::uint32_t S, std::uint16_t* __restrict__ vt,
    std::uint64_t pitch) {
    __shared__ std::uint16_t tile[128][kHeadDim + 2];
    const std::uint32_t r0 = blockIdx.x * 128U, head = blockIdx.y;
    for (std::uint32_t i = threadIdx.x; i < 128U * (kHeadDim / 2U); i += 256U) {
        const std::uint32_t rr = i / (kHeadDim / 2U), d2 = i % (kHeadDim / 2U);
        if (r0 + rr < S) {
            const std::uint32_t v = *reinterpret_cast<const std::uint32_t*>(
                qkv + static_cast<std::size_t>(r0 + rr) * 3U * kInner + 2U * kInner + head * kHeadDim + 2U * d2);
            tile[rr][2U * d2] = static_cast<std::uint16_t>(v & 0xffffU);
            tile[rr][2U * d2 + 1U] = static_cast<std::uint16_t>(v >> 16U);
        }
    }
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < 128U * kHeadDim; i += 256U) {
        const std::uint32_t d = i / 128U, rr = i % 128U;
        if (r0 + rr < S) vt[(static_cast<std::size_t>(head) * kHeadDim + d) * pitch + r0 + rr] = tile[rr][d];
    }
}

__global__ void __launch_bounds__(256) k_add_residual(std::uint16_t* __restrict__ x, const std::uint16_t* __restrict__ o, std::uint64_t n) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tob(bf(o[i]) + bf(x[i]));
}

// bf16 linear, fp32 k-ascending: tile 64 rows x 64 cols, 256 threads (thread: one column, 16 rows), 32-wide k chunks.
__global__ void __launch_bounds__(256) k_bf16_linear(const std::uint16_t* __restrict__ x, std::uint32_t m, const std::uint16_t* __restrict__ w,
    const std::uint16_t* __restrict__ b, std::uint32_t n, std::uint32_t k, std::uint16_t* __restrict__ y) {
    __shared__ float xs[32][65];  // [kk][row]
    __shared__ float ws[32][65];  // [kk][col]
    const std::uint32_t r0 = blockIdx.y * 64U, c0 = blockIdx.x * 64U;
    const unsigned t = threadIdx.x, col = t & 63U, rg = (t >> 6U) * 16U;
    float acc[16];
    for (unsigned j = 0; j < 16U; ++j) acc[j] = 0.0F;
    for (std::uint32_t k0 = 0; k0 < k; k0 += 32U) {
        for (unsigned i = t; i < 64U * 32U; i += 256U) {
            const unsigned rr = i / 32U, kk = i % 32U;
            xs[kk][rr] = (r0 + rr < m && k0 + kk < k) ? bf(x[static_cast<std::size_t>(r0 + rr) * k + k0 + kk]) : 0.0F;
            ws[kk][rr] = (c0 + rr < n && k0 + kk < k) ? bf(w[static_cast<std::size_t>(c0 + rr) * k + k0 + kk]) : 0.0F;
        }
        __syncthreads();
        const unsigned kn = k - k0 < 32U ? k - k0 : 32U;
        for (unsigned kk = 0; kk < kn; ++kk) {
            const float wv = ws[kk][col];
            for (unsigned j = 0; j < 16U; ++j) acc[j] = acc[j] + xs[kk][rg + j] * wv;  // exact products: mul + add == fma
        }
        __syncthreads();
    }
    if (c0 + col >= n) return;
    const float bias = b ? bf(b[c0 + col]) : 0.0F;
    for (unsigned j = 0; j < 16U; ++j)
        if (r0 + rg + j < m) y[static_cast<std::size_t>(r0 + rg + j) * n + c0 + col] = tob(acc[j] + bias);
}

__global__ void __launch_bounds__(256) k_swiglu(const std::uint16_t* __restrict__ f1, std::uint32_t rows, std::uint32_t ffn,
    const std::uint16_t* __restrict__ silu, std::uint16_t* __restrict__ out) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::uint64_t>(rows) * ffn) return;
    const std::uint64_t r = i / ffn, c = i % ffn;
    const std::uint16_t g = f1[r * 2U * ffn + c], u = f1[r * 2U * ffn + ffn + c];
    out[i] = tob(bf(silu[g]) * bf(u));
}

__global__ void __launch_bounds__(256) k_linear_f64(const float* __restrict__ x, std::uint32_t x_rs, std::uint32_t m,
    const float* __restrict__ w, const float* __restrict__ b, std::uint32_t n, std::uint32_t k, float* __restrict__ y) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::uint64_t>(m) * n) return;
    const std::uint32_t r = static_cast<std::uint32_t>(i / n), c = static_cast<std::uint32_t>(i % n);
    const float* xr = x + static_cast<std::size_t>(r) * x_rs;
    const float* wr = w + static_cast<std::size_t>(c) * k;
    double s = 0.0;
    for (std::uint32_t j = 0; j < k; ++j) s += static_cast<double>(xr[j]) * static_cast<double>(wr[j]);
    y[i] = static_cast<float>(s + static_cast<double>(b[c]));
}

struct ModOut {
    std::uint16_t* p[6];
};
__global__ void __launch_bounds__(256) k_mod_vectors(const float* __restrict__ ada, std::uint32_t rows, ModOut o) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::uint64_t>(rows) * kHidden) return;
    const std::uint32_t r = static_cast<std::uint32_t>(i / kHidden), c = static_cast<std::uint32_t>(i % kHidden);
    const float* row = ada + static_cast<std::size_t>(r) * 6U * kHidden;
    for (unsigned j = 0; j < 6U; ++j) {
        const float v = row[j * kHidden + c];
        o.p[j][i] = (j == 1U || j == 4U) ? tob(1.0F + bf(tob(v))) : tob(v);
    }
}

// patch projection, fp32 k-ascending (mul then add, -ffp-contract=off): tile 64 rows x 64 cols, 256 threads (thread: one
// column, 16 rows), 32-wide k chunks through LDS; rows scattered to dst_row.
__global__ void __launch_bounds__(256) k_patch_proj(const float* __restrict__ rows, std::uint32_t m, std::uint32_t k,
    const float* __restrict__ w, const float* __restrict__ b, const std::int32_t* __restrict__ dst_row, std::uint16_t* __restrict__ h) {
    __shared__ float xs[32][65];  // [kk][row]
    __shared__ float ws[32][65];  // [kk][col]
    const std::uint32_t r0 = blockIdx.y * 64U, c0 = blockIdx.x * 64U;
    const unsigned t = threadIdx.x, col = t & 63U, rg = (t >> 6U) * 16U;
    float acc[16];
    for (unsigned j = 0; j < 16U; ++j) acc[j] = 0.0F;
    for (std::uint32_t k0 = 0; k0 < k; k0 += 32U) {
        for (unsigned i = t; i < 64U * 32U; i += 256U) {
            const unsigned rr = i / 32U, kk = i % 32U;
            xs[kk][rr] = (r0 + rr < m && k0 + kk < k) ? rows[static_cast<std::size_t>(r0 + rr) * k + k0 + kk] : 0.0F;
            ws[kk][rr] = (c0 + rr < kHidden && k0 + kk < k) ? w[static_cast<std::size_t>(c0 + rr) * k + k0 + kk] : 0.0F;
        }
        __syncthreads();
        const unsigned kn = k - k0 < 32U ? k - k0 : 32U;
        for (unsigned kk = 0; kk < kn; ++kk) {
            const float wv = ws[kk][col];
            for (unsigned j = 0; j < 16U; ++j) acc[j] = acc[j] + xs[kk][rg + j] * wv;
        }
        __syncthreads();
    }
    const std::uint32_t c = c0 + col;
    if (c >= kHidden) return;
    for (unsigned j = 0; j < 16U; ++j)
        if (r0 + rg + j < m) h[static_cast<std::size_t>(dst_row[r0 + rg + j]) * kHidden + c] = tob(acc[j] + b[c]);
}

__global__ void __launch_bounds__(256) k_final_mod(const std::uint16_t* __restrict__ x, std::uint32_t row0, std::uint32_t m,
    const std::uint16_t* __restrict__ w, const float* __restrict__ shift, const float* __restrict__ scale, float eps, float* __restrict__ f) {
    const std::uint32_t r = blockIdx.x * 8U + (threadIdx.x >> 5U);
    if (r >= m) return;
    const unsigned t = threadIdx.x & 31U;
    const std::uint16_t* xr = x + static_cast<std::size_t>(row0 + r) * kHidden;
    const float ss = wave_ref_sumsq(kHidden, [&](std::uint32_t e) { return bf(xr[e]); });
    const float inv = rms_inv(ss, kHidden, eps);
    float* fr = f + static_cast<std::size_t>(r) * kHidden;
    for (std::uint32_t e = t; e < kHidden; e += 32U) {
        const std::uint16_t nv = tob((bf(xr[e]) * inv) * bf(w[e]));
        fr[e] = (bf(nv) * (1.0F + scale[e])) + shift[e];
    }
}

__global__ void __launch_bounds__(256) k_row_rms_debug(const std::uint16_t* __restrict__ x, std::uint32_t rows, std::uint32_t n, float eps,
    float* __restrict__ ss_out, float* __restrict__ inv_out) {
    const std::uint32_t r = blockIdx.x * 8U + (threadIdx.x >> 5U);
    if (r >= rows) return;
    const std::uint16_t* xr = x + static_cast<std::size_t>(r) * n;
    const float ss = wave_ref_sumsq(n, [&](std::uint32_t e) { return bf(xr[e]); });
    if ((threadIdx.x & 31U) == 0U) {
        ss_out[r] = ss;
        inv_out[r] = rms_inv(ss, n, eps);
    }
}

void check_launch(const char* what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        std::fprintf(stderr, "h3 glue: %s launch failed: %s\n", what, hipGetErrorString(e));
        std::abort();
    }
}
unsigned blocks_for(std::uint64_t n, unsigned per) { return static_cast<unsigned>((n + per - 1) / per); }

}  // namespace

void launch_rmsnorm_mod(const std::uint16_t* x, const std::uint16_t* w, const std::uint32_t* mod_row, const std::uint16_t* s1,
    const std::uint16_t* shift, std::uint32_t rows, std::uint32_t n, float eps, std::uint16_t* h, void* stream) {
    if (n % 32U || rows == 0) return;
    hipLaunchKernelGGL(k_rmsnorm_mod, dim3(blocks_for(rows, 8)), dim3(256), 0, static_cast<hipStream_t>(stream), x, w, mod_row, s1,
        shift, rows, n, eps, h);
    check_launch("rmsnorm_mod");
}
void launch_gate_residual(std::uint16_t* x, const std::uint16_t* o, const std::uint32_t* mod_row, const std::uint16_t* g,
    std::uint32_t rows, std::uint32_t n, void* stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(k_gate_residual, dim3(blocks_for(static_cast<std::uint64_t>(rows) * n, 256)), dim3(256), 0,
        static_cast<hipStream_t>(stream), x, o, mod_row, g, rows, n);
    check_launch("gate_residual");
}
void launch_qk_norm_rope(const std::uint16_t* qkv, std::uint32_t S, const std::uint16_t* qw, const std::uint16_t* kw,
    const std::uint16_t* c, const std::uint16_t* s, float eps, std::uint16_t* q, std::uint16_t* k, void* stream,
    std::uint32_t* head_sumsq_max) {
    if (S == 0) return;
    hipLaunchKernelGGL(k_qk_norm_rope, dim3(S, 2U * kHeads / 8U), dim3(256), 0, static_cast<hipStream_t>(stream), qkv, S, qw, kw, c, s,
        eps, q, k, head_sumsq_max);
    check_launch("qk_norm_rope");
}
void launch_transpose_v(const std::uint16_t* qkv, std::uint32_t S, std::uint16_t* vt, std::uint64_t vt_pitch, void* stream) {
    if (S == 0) return;
    hipLaunchKernelGGL(k_transpose_v, dim3(blocks_for(S, 128), kHeads), dim3(256), 0, static_cast<hipStream_t>(stream), qkv, S, vt, vt_pitch);
    check_launch("transpose_v");
}
void launch_add_residual(std::uint16_t* x, const std::uint16_t* o, std::uint64_t n, void* stream) {
    if (n == 0) return;
    hipLaunchKernelGGL(k_add_residual, dim3(blocks_for(n, 256)), dim3(256), 0, static_cast<hipStream_t>(stream), x, o, n);
    check_launch("add_residual");
}
void launch_bf16_linear(const std::uint16_t* x, std::uint32_t m, const std::uint16_t* w, const std::uint16_t* b, std::uint32_t n,
    std::uint32_t k, std::uint16_t* y, void* stream) {
    if (m == 0) return;
    hipLaunchKernelGGL(k_bf16_linear, dim3(blocks_for(n, 64), blocks_for(m, 64)), dim3(256), 0, static_cast<hipStream_t>(stream), x, m, w, b,
        n, k, y);
    check_launch("bf16_linear");
}
void launch_swiglu(const std::uint16_t* f1, std::uint32_t rows, std::uint32_t ffn, const std::uint16_t* silu_table,
    std::uint16_t* out, void* stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(k_swiglu, dim3(blocks_for(static_cast<std::uint64_t>(rows) * ffn, 256)), dim3(256), 0,
        static_cast<hipStream_t>(stream), f1, rows, ffn, silu_table, out);
    check_launch("swiglu");
}
void launch_linear_f64(const float* x, std::uint32_t x_rs, std::uint32_t m, const float* w, const float* b, std::uint32_t n,
    std::uint32_t k, float* y, void* stream) {
    if (m == 0) return;
    hipLaunchKernelGGL(k_linear_f64, dim3(blocks_for(static_cast<std::uint64_t>(m) * n, 256)), dim3(256), 0,
        static_cast<hipStream_t>(stream), x, x_rs, m, w, b, n, k, y);
    check_launch("linear_f64");
}
void launch_mod_vectors(const float* ada, std::uint32_t rows, std::uint16_t* const out[6], void* stream) {
    ModOut o;
    for (int j = 0; j < 6; ++j) o.p[j] = out[j];
    hipLaunchKernelGGL(k_mod_vectors, dim3(blocks_for(static_cast<std::uint64_t>(rows) * kHidden, 256)), dim3(256), 0,
        static_cast<hipStream_t>(stream), ada, rows, o);
    check_launch("mod_vectors");
}
void launch_patch_proj(const float* rows, std::uint32_t m, std::uint32_t k, const float* w, const float* b,
    const std::int32_t* dst_row, std::uint16_t* h, void* stream) {
    if (m == 0) return;
    hipLaunchKernelGGL(k_patch_proj, dim3(blocks_for(kHidden, 64), blocks_for(m, 64)), dim3(256), 0, static_cast<hipStream_t>(stream),
        rows, m, k, w, b, dst_row, h);
    check_launch("patch_proj");
}
void launch_final_mod(const std::uint16_t* x, std::uint32_t row0, std::uint32_t m, const std::uint16_t* w, const float* shift,
    const float* scale, float eps, float* f, void* stream) {
    if (m == 0) return;
    hipLaunchKernelGGL(k_final_mod, dim3(blocks_for(m, 8)), dim3(256), 0, static_cast<hipStream_t>(stream), x, row0, m, w, shift, scale,
        eps, f);
    check_launch("final_mod");
}

void launch_row_rms_debug(const std::uint16_t* x, std::uint32_t rows, std::uint32_t n, float eps, float* ss, float* inv, void* stream) {
    hipLaunchKernelGGL(k_row_rms_debug, dim3(blocks_for(rows, 8)), dim3(256), 0, static_cast<hipStream_t>(stream), x, rows, n, eps, ss, inv);
    check_launch("row_rms_debug");
}

}  // namespace strix::h3::block::gpu
