#pragma once
#include <hip/hip_runtime.h>
#include "ops.hpp"
#include "norm_bounds.hpp"
#include <stdexcept>
#include <string>

namespace strix::gemma4::gpu {
inline void launch_checked() {
    const auto e = hipGetLastError();
    if (e != hipSuccess)
        throw std::runtime_error(std::string("gemma4 kernel launch: ") + hipGetErrorString(e));
}
inline void valid_rows(std::uint32_t rows) {
    if (rows == 0 || rows > 8)
        throw std::runtime_error("g4-v1 admits one to eight device rows");
}
__device__ inline float wave_sum(float a) {
    for (unsigned d = 16; d; d >>= 1U)
        a = a + __shfl_down(a, d, 32);
    return __shfl(a, 0, 32);
}
// The same integer comparisons and midpoint rule as pow32_rsqrt, searched by
// one wave in 32 ordered intervals. No floating approximation seeds the search.
__device__ inline float wave_rsqrt_wide(float x) {
    const auto bits = bit_cast<std::uint32_t>(x);
    if (bits == 0 || bits >= 0x7f800000U)
        return pow32_rsqrt(x);
    std::uint32_t mx;
    int ex;
    strix::gemma4::detail::dyadic(bits, mx, ex);
    unsigned lo = 0x00800000U, hi = 0x7f7fffffU;
    const unsigned lane = threadIdx.x % 32U;
    while (lo < hi) {
        const auto step = (hi - lo + 31U) / 32U;
        const auto candidate = min(hi, lo + step * (lane + 1U));
        std::uint32_t my;
        int ey;
        strix::gemma4::detail::dyadic(candidate, my, ey);
        const auto valid = strix::gemma4::detail::square_compare(mx, my, ex + 2 * ey) <= 0;
        const unsigned count = unsigned(__popc(unsigned(__ballot(valid))));
        lo = min(hi, lo + step * count);
        hi = min(hi, lo + step - 1U);
    }
    std::uint32_t my;
    int ey;
    strix::gemma4::detail::dyadic(lo, my, ey);
    const int c = strix::gemma4::detail::square_compare(mx, 2U * my + 1U, ex + 2 * (ey - 1));
    return bit_cast<float>(lo + (c < 0 || (c == 0 && (lo & 1U)) ? 1U : 0U));
}
// Seeded: one ballot over the 32 bracket candidates around the hardware
// estimate. Validity is monotone in the candidate, so when the window shows the
// boundary (or touches the bracket end that the full search would stop at), the
// window gives the search's exact result; otherwise the full search runs.
__device__ inline float wave_rsqrt(float x, bool bounded = true, bool intervals = false, bool seeded = false) {
    if (!bounded)
        return wave_rsqrt_wide(x);
    const auto raw = bit_cast<std::uint32_t>(x);
    if (raw == 0 || raw >= 0x7f800000U)
        return pow32_rsqrt(x);
    // The input is uniform within the wave. The result always has a normal
    // exponent, even for subnormal x. An even power of two hits the upper
    // exponent boundary; all other inputs lie strictly below that boundary.
    const auto bits = unsigned(__builtin_amdgcn_readfirstlane(int(raw)));
    std::uint32_t mx;
    int ex;
    strix::gemma4::detail::dyadic(bits, mx, ex);
    const int result_exponent = rsqrt_exponent(mx, ex);
    const int ey = result_exponent - 23;
    unsigned lo = unsigned(result_exponent + 127) << 23U, hi = lo | 0x7fffffU;
    if (intervals)
        rsqrt_interval_bounds(mx, ex, result_exponent, lo, hi);
    const unsigned lane = threadIdx.x % 32U;
    if (seeded && lo < hi) {
        const auto estimate = bit_cast<std::uint32_t>(__builtin_amdgcn_rsqf(bit_cast<float>(bits)));
        auto first = estimate > lo + 15U ? estimate - 15U : lo;
        if (first > hi)
            first = hi;
        const auto last = hi - first > 31U ? first + 31U : hi;
        if (last - first < 31U && last == hi)
            first = hi - lo > 31U ? hi - 31U : lo;
        const auto candidate = first + lane;
        bool valid = false;
        if (candidate <= hi) {
            const auto my = (candidate & 0x7fffffU) | 0x800000U;
            valid = strix::gemma4::detail::square_compare(mx, my, ex + 2 * ey) <= 0;
        }
        const unsigned count = unsigned(__popc(unsigned(__ballot(valid))));
        const auto size = (hi - first > 31U ? 32U : hi - first + 1U);
        // Lower end settled: a valid first candidate, or the bracket start.
        // Upper end settled: an invalid candidate inside the window, or the bracket end.
        if ((count > 0U || first == lo) && (count < size || first + size - 1U == hi)) {
            lo = count > 0U ? first + count - 1U : lo;
            hi = lo;
        }
    }
    while (lo < hi) {
        const auto step = (hi - lo + 31U) / 32U;
        const auto candidate = min(hi, lo + step * (lane + 1U));
        const auto my = (candidate & 0x7fffffU) | 0x800000U;
        const auto valid = strix::gemma4::detail::square_compare(mx, my, ex + 2 * ey) <= 0;
        const unsigned count = unsigned(__popc(unsigned(__ballot(valid))));
        lo = min(hi, lo + step * count);
        hi = min(hi, lo + step - 1U);
    }
    const auto my = (lo & 0x7fffffU) | 0x800000U;
    const int c = strix::gemma4::detail::square_compare(mx, 2U * my + 1U, ex + 2 * (ey - 1));
    return bit_cast<float>(lo + (c < 0 || (c == 0 && (lo & 1U)) ? 1U : 0U));
}
// One wave normalizes (and optionally rotates) one Dim-wide head held in
// registers; lane l owns coordinates l+32j. The operation order is the register
// head path: ordered lane squares, lane tree, rsqrt, x*r, then *weight, and
// each rotary multiply/add rounded separately on the declared pairs.
template <unsigned Dim>
__device__ inline void register_head_values(float (&values)[Dim / 32U], LinearView w, const float* rotation,
                                            bool parallel, bool bounded, bool intervals,
                                            bool seeded = false) {
    constexpr unsigned Values = Dim / 32U, Active = Dim == 512U ? 64U : 128U, Rotations = Active / 32U;
    const unsigned lane = threadIdx.x % 32U;
    float weights[Values], cs[Rotations], sn[Rotations];
#pragma unroll
    for (unsigned j = 0; j < Values; ++j)
        if (w.data)
            weights[j] = coefficient(w, 0, lane + 32U * j);
    if (rotation) {
#pragma unroll
        for (unsigned j = 0; j < Rotations; ++j) {
            cs[j] = rotation[2U * (lane + 32U * j)];
            sn[j] = rotation[2U * (lane + 32U * j) + 1U];
        }
    }
    float sum = 0.0F;
#pragma unroll
    for (unsigned j = 0; j < Values; ++j)
        sum = sum + values[j] * values[j];
    sum = wave_sum(sum);
    const float u = sum / float(Dim) + norm_eps;
    const float r = (bit_cast<std::uint32_t>(sum) & 0x7f800000U) == 0x7f800000U
                        ? bit_cast<float>(0x7fc00000U)
                        : (parallel ? wave_rsqrt(u, bounded, intervals, seeded) : pow32_rsqrt(u));
#pragma unroll
    for (unsigned j = 0; j < Values; ++j)
        values[j] = w.data ? (values[j] * r) * weights[j] : values[j] * r;
    if (rotation) {
#pragma unroll
        for (unsigned j = 0; j < Rotations; ++j) {
            const float a = values[j], b = values[j + Values / 2U];
            values[j] = a * cs[j] - b * sn[j];
            values[j + Values / 2U] = b * cs[j] + a * sn[j];
        }
    }
}
__device__ inline void finite_flag(float x, std::uint32_t* failure) {
    if ((bit_cast<std::uint32_t>(x) & 0x7f800000U) == 0x7f800000U)
        atomicOr(failure, 1U);
}
// Ordered norm reduction shared by the separate epilogue and A8 consumer prologue.
template <bool Scaled = false> __device__ inline float square_sum(const float* x, unsigned width, bool prefetch) {
    float sum = 0.0F;
    unsigned i = threadIdx.x;
    if (prefetch)
        for (; i < width && width - i > 224U; i += 256U) {
            float values[8];
#pragma unroll
            for (unsigned j = 0; j < 8; ++j)
                values[j] = x[i + 32U * j];
#pragma unroll
            for (unsigned j = 0; j < 8; ++j) {
                const float a = Scaled ? values[j] * model_ple_scale : values[j];
                sum = sum + a * a;
            }
        }
    for (; i < width; i += 32U) {
        const float a = Scaled ? x[i] * model_ple_scale : x[i];
        sum = sum + a * a;
    }
    return wave_sum(sum);
}
template <bool Parallel>
__device__ inline float epilogue_factor(const float* x, unsigned width, bool prefetch, bool bounded, bool intervals,
                                       bool seeded = false) {
    const float sum = square_sum(x, width, prefetch);
    if ((bit_cast<std::uint32_t>(sum) & 0x7f800000U) == 0x7f800000U)
        return bit_cast<float>(0x7fc00000U);
    const float u = sum / float(width) + norm_eps;
    return Parallel ? wave_rsqrt(u, bounded, intervals, seeded) : pow32_rsqrt(u);
}
} // namespace strix::gemma4::gpu
