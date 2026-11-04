#pragma once
// LoRA bypass side branch (FAST-class, OFF by default; identities h3-turbo-v4-bypass, h3-lora-*): the run-time branch
// of the reference's default (bypass) mode, y = base(x); t = x A^T (rank r, bf16 out); l = t B^T (bf16 out);
// y = bf16(y + s * l). Arithmetic (deterministic, batch invariant, mirrored bit for bit by bypass_host in bypass_ref.hpp):
//   t[m][j] = bf16( fma chain over k = 0, 1, .. of x[m][k] * A[j][k] from +0 )                 (fp32)
//   l[m][o] = bf16( fma chain over j = 0 .. r - 1 of t[m][j] * B[o][j] from +0 )               (fp32)
//   y[m][o] = bf16( fmaf(s, l[m][o], y[m][o]) )     (the reference's in-place add with alpha: fused multiply-add)
// The reference's two bf16 linears accumulate in the library's order, so the branch matches it within the usual bf16
// GEMM tolerance (G-class), not bitwise. For fc1 the branch is added to the [gate | up] output before SwiGLU (the
// fused SwiGLU epilogue cannot be used with it: run fc1 plain, then this, then the driver's SwiGLU).
// Shapes: rank r % 32 == 0 and 32 <= r <= 512, k % 32 == 0, n % 64 == 0, rows >= 1; x / y bf16 with row pitches
// x_ld / y_ld (elements); t scratch [rows][r].
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/gemm/ref.hpp"
#include "h3/levers/lora/bypass_ref.hpp"

namespace strix::h3::levers::lora {

constexpr std::uint32_t kMaxRank = 512U;

namespace dev {
// Register-tiled VALU kernels. Each output keeps the spec's single fp32 fma chain in ascending reduction order from
// +0; the tiling (and the rank chunking) only decides which thread runs which chains.
constexpr std::uint32_t kTile = 64U;  // rows x (rank | columns) per work-group, 4 x 4 outputs per thread
constexpr std::uint32_t kKc = 32U;    // reduction chunk staged in LDS (down)
constexpr std::uint32_t kPad = 4U;    // LDS row pad (floats)

// t[m][j] = bf16(fma chain over k of x[m][k] * A[j][k]); work-group = 64 rows x 64 rank columns (blockIdx.y = rank
// chunk); rank rows >= r are staged as zero and never stored (they are separate outputs, not a reduction term).
template <int = 0>
__global__ __launch_bounds__(256) void down_kernel(const std::uint16_t* __restrict__ x, std::uint64_t x_ld, std::uint32_t rows, std::uint32_t k,
    const std::uint16_t* __restrict__ a, std::uint32_t rank, std::uint16_t* __restrict__ t) {
#pragma clang fp contract(off)
    __shared__ float xs[kKc][kTile + kPad];
    __shared__ float as[kKc][kTile + kPad];
    const std::uint32_t tid = threadIdx.x, tr = tid / 16U, tc = tid % 16U, r0 = blockIdx.x * kTile, jb = blockIdx.y * kTile;
    const std::uint32_t lr = tid / 4U, lk = (tid % 4U) * 8U;  // staging: one 16-byte load of x and of A per thread
    const bool row_ok = r0 + lr < rows, a_ok = jb + lr < rank;
    const std::uint16_t* xr = x + static_cast<std::uint64_t>(r0 + lr) * x_ld + lk;
    const std::uint16_t* ar = a + static_cast<std::uint64_t>(jb + lr) * k + lk;
    float acc[4][4] = {};
    for (std::uint32_t k0 = 0; k0 < k; k0 += kKc) {
        uint4 xv = make_uint4(0U, 0U, 0U, 0U), av = make_uint4(0U, 0U, 0U, 0U);
        if (row_ok) xv = *reinterpret_cast<const uint4*>(xr + k0);
        if (a_ok) av = *reinterpret_cast<const uint4*>(ar + k0);
        const std::uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w}, aw[4] = {av.x, av.y, av.z, av.w};
        __syncthreads();
#pragma unroll
        for (std::uint32_t i = 0; i < 4U; ++i) {
            xs[lk + 2U * i][lr] = gemm::bf16_to_f32(static_cast<std::uint16_t>(xw[i] & 0xffffU));
            xs[lk + 2U * i + 1U][lr] = gemm::bf16_to_f32(static_cast<std::uint16_t>(xw[i] >> 16U));
            as[lk + 2U * i][lr] = gemm::bf16_to_f32(static_cast<std::uint16_t>(aw[i] & 0xffffU));
            as[lk + 2U * i + 1U][lr] = gemm::bf16_to_f32(static_cast<std::uint16_t>(aw[i] >> 16U));
        }
        __syncthreads();
#pragma unroll 8
        for (std::uint32_t kk = 0; kk < kKc; ++kk) {
            const float4 xv4 = *reinterpret_cast<const float4*>(&xs[kk][tr * 4U]);
            const float4 av4 = *reinterpret_cast<const float4*>(&as[kk][tc * 4U]);
            const float xq[4] = {xv4.x, xv4.y, xv4.z, xv4.w}, aq[4] = {av4.x, av4.y, av4.z, av4.w};
#pragma unroll
            for (std::uint32_t i = 0; i < 4U; ++i)
#pragma unroll
                for (std::uint32_t j = 0; j < 4U; ++j) acc[i][j] = __builtin_fmaf(xq[i], aq[j], acc[i][j]);
        }
    }
    if (jb + tc * 4U >= rank) return;
#pragma unroll
    for (std::uint32_t i = 0; i < 4U; ++i) {
        const std::uint32_t r = r0 + tr * 4U + i;
        if (r >= rows) break;
        std::uint16_t* tp = t + static_cast<std::uint64_t>(r) * rank + jb + tc * 4U;
        const std::uint32_t lo = gemm::f32_to_bf16(acc[i][0]) | (static_cast<std::uint32_t>(gemm::f32_to_bf16(acc[i][1])) << 16U);
        const std::uint32_t hi = gemm::f32_to_bf16(acc[i][2]) | (static_cast<std::uint32_t>(gemm::f32_to_bf16(acc[i][3])) << 16U);
        *reinterpret_cast<uint2*>(tp) = make_uint2(lo, hi);
    }
}

// y[m][o] = bf16(fmaf(s, bf16(fma chain over j of t[m][j] * B[o][j]), y[m][o])); work-group = 64 rows x 64 columns;
// the rank is walked in chunks of 64 staged in LDS, the accumulators carried across chunks (one chain per output),
// and the inner loop runs only over the chunk's valid j (no zero padding enters the chain).
template <int = 0>
__global__ __launch_bounds__(256) void up_add_kernel(const std::uint16_t* __restrict__ t, std::uint32_t rows, const std::uint16_t* __restrict__ b,
    std::uint32_t rank, float s, std::uint16_t* __restrict__ y, std::uint64_t y_ld) {
#pragma clang fp contract(off)
    __shared__ float ts[kTile][kTile + kPad];
    __shared__ float bs[kTile][kTile + kPad];
    const std::uint32_t tid = threadIdx.x, tr = tid / 16U, tc = tid % 16U, r0 = blockIdx.y * kTile, c0 = blockIdx.x * kTile;
    float acc[4][4] = {};
    for (std::uint32_t jb = 0; jb < rank; jb += kTile) {
        const std::uint32_t jn = rank - jb < kTile ? rank - jb : kTile;
        if (jb) __syncthreads();  // previous chunk consumed
#pragma unroll
        for (std::uint32_t h = 0; h < 2U; ++h) {  // 64 x 64 bf16 of t and of B: two 16-byte loads of each per thread
            const std::uint32_t idx = tid + h * 256U, lr = idx / 8U, lj = (idx % 8U) * 8U;
            uint4 tv = make_uint4(0U, 0U, 0U, 0U), bv = make_uint4(0U, 0U, 0U, 0U);
            if (lj < jn) {
                if (r0 + lr < rows) tv = *reinterpret_cast<const uint4*>(t + static_cast<std::uint64_t>(r0 + lr) * rank + jb + lj);
                bv = *reinterpret_cast<const uint4*>(b + static_cast<std::uint64_t>(c0 + lr) * rank + jb + lj);
            }
            const std::uint32_t tw[4] = {tv.x, tv.y, tv.z, tv.w}, bw[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (std::uint32_t i = 0; i < 4U; ++i) {
                ts[lj + 2U * i][lr] = gemm::bf16_to_f32(static_cast<std::uint16_t>(tw[i] & 0xffffU));
                ts[lj + 2U * i + 1U][lr] = gemm::bf16_to_f32(static_cast<std::uint16_t>(tw[i] >> 16U));
                bs[lj + 2U * i][lr] = gemm::bf16_to_f32(static_cast<std::uint16_t>(bw[i] & 0xffffU));
                bs[lj + 2U * i + 1U][lr] = gemm::bf16_to_f32(static_cast<std::uint16_t>(bw[i] >> 16U));
            }
        }
        __syncthreads();
#pragma unroll 8
        for (std::uint32_t j = 0; j < jn; ++j) {
            const float4 tv4 = *reinterpret_cast<const float4*>(&ts[j][tr * 4U]);
            const float4 bv4 = *reinterpret_cast<const float4*>(&bs[j][tc * 4U]);
            const float tq[4] = {tv4.x, tv4.y, tv4.z, tv4.w}, bq[4] = {bv4.x, bv4.y, bv4.z, bv4.w};
#pragma unroll
            for (std::uint32_t i = 0; i < 4U; ++i)
#pragma unroll
                for (std::uint32_t o = 0; o < 4U; ++o) acc[i][o] = __builtin_fmaf(tq[i], bq[o], acc[i][o]);
        }
    }
#pragma unroll
    for (std::uint32_t i = 0; i < 4U; ++i) {
        const std::uint32_t r = r0 + tr * 4U + i;
        if (r >= rows) break;
        uint2* yp = reinterpret_cast<uint2*>(y + static_cast<std::uint64_t>(r) * y_ld + c0 + tc * 4U);
        const uint2 yv = *yp;
        const std::uint32_t yw[2] = {yv.x, yv.y};
        std::uint32_t ow[2];
#pragma unroll
        for (std::uint32_t h = 0; h < 2U; ++h) {
            std::uint32_t o2[2];
#pragma unroll
            for (std::uint32_t e = 0; e < 2U; ++e) {
                const float l = gemm::bf16_to_f32(gemm::f32_to_bf16(acc[i][2U * h + e]));
                const float yo = gemm::bf16_to_f32(static_cast<std::uint16_t>((yw[h] >> (16U * e)) & 0xffffU));
                o2[e] = gemm::f32_to_bf16(__builtin_fmaf(s, l, yo));
            }
            ow[h] = o2[0] | (o2[1] << 16U);
        }
        *yp = make_uint2(ow[0], ow[1]);
    }
}
}  // namespace dev

// Argument contract: rank % 32 == 0 in [32, 512], k % 32 == 0, n % 64 == 0 (every H3 linear: k, n multiples of 256),
// k <= 16384, x_ld % 8 == 0, y_ld % 4 == 0, x / a / b / t 16-byte aligned, y 8-byte aligned.
inline void check_bypass_args(const std::uint16_t* x, std::uint64_t x_ld, std::uint32_t rows, std::uint32_t k, const std::uint16_t* a,
    const std::uint16_t* b, std::uint32_t n, const std::uint16_t* t, const std::uint16_t* y, std::uint64_t y_ld, std::uint32_t rank = kRank) {
    const auto al = [](const void* p, std::uintptr_t m) { return (reinterpret_cast<std::uintptr_t>(p) % m) == 0U; };
    if (!rows || !k || k % dev::kKc || !n || n % dev::kTile || !x || !a || !b || !t || !y || x_ld < k || y_ld < n || k > 16384U || x_ld % 8U ||
        y_ld % 4U || !al(x, 16U) || !al(a, 16U) || !al(b, 16U) || !al(t, 16U) || !al(y, 8U) || rows > (1U << 26U) || !rank || rank % 32U ||
        rank > kMaxRank)
        throw std::invalid_argument("h3 lora: invalid bypass arguments");
}
// The two halves separately (benchmarks; a caller may run the down projection before the base linear overwrites x).
inline void launch_down(const std::uint16_t* x, std::uint64_t x_ld, std::uint32_t rows, std::uint32_t k, const std::uint16_t* a, std::uint16_t* t,
    hipStream_t stream, std::uint32_t rank = kRank) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::down_kernel<0>), dim3((rows + dev::kTile - 1U) / dev::kTile, (rank + dev::kTile - 1U) / dev::kTile), dim3(256), 0,
        stream, x, x_ld, rows, k, a, rank, t);
}
inline void launch_up_add(const std::uint16_t* t, std::uint32_t rows, const std::uint16_t* b, std::uint32_t n, float s, std::uint16_t* y, std::uint64_t y_ld,
    hipStream_t stream, std::uint32_t rank = kRank) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::up_add_kernel<0>), dim3(n / dev::kTile, (rows + dev::kTile - 1U) / dev::kTile), dim3(256), 0, stream, t, rows, b,
        rank, s, y, y_ld);
}

// t: device scratch [rows][rank] bf16. a: [rank][k] bf16 (lora_A), b: [n][rank] bf16 (lora_B; for fc1 in checkpoint
// row order). rank defaults to 64 (Turbo v4).
inline void add_bypass(const std::uint16_t* x, std::uint64_t x_ld, std::uint32_t rows, std::uint32_t k, const std::uint16_t* a, const std::uint16_t* b,
    std::uint32_t n, float s, std::uint16_t* t, std::uint16_t* y, std::uint64_t y_ld, hipStream_t stream, std::uint32_t rank = kRank) {
    check_bypass_args(x, x_ld, rows, k, a, b, n, t, y, y_ld, rank);
    static_cast<void>(hipGetLastError());
    launch_down(x, x_ld, rows, k, a, t, stream, rank);
    launch_up_add(t, rows, b, n, s, y, y_ld, stream, rank);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("h3 lora: launch failed: ") + hipGetErrorString(e));
}

}  // namespace strix::h3::levers::lora
