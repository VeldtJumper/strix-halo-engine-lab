#pragma once
// Device-side KV8 group conversion shared by the mirror sync and the FAST decode attention (g4-dattn-f2).
// One group = kv8_group consecutive features of one K or V row: amax = max |x| (order independent),
// d = half(amax / 127) (RNE, correctly rounded division), code = clamp(rint(x / float(d)), -127, 127), 0 when d == 0;
// the dequantized value float(d) * code is exact in FP32. A nonfinite source element sets bit 1, an overflowing
// scale bit 4.
#include <hip/hip_runtime.h>
#include "fp16_mirror.hpp"

namespace strix::gemma4::gpu {
// A nonfinite source element sets failure bit 1 before the group maximum (which ignores NaN) could hide it.
__device__ inline void kv8_finite(float x, unsigned* failure) {
    if ((__builtin_bit_cast(unsigned, x) & 0x7f800000U) == 0x7f800000U)
        atomicOr(failure, 1U);
}
__device__ inline float kv8_abs(float x) { return __builtin_bit_cast(float, __builtin_bit_cast(unsigned, x) & 0x7fffffffU); }
__device__ inline unsigned short kv8_scale_bits(float amax, unsigned* failure) {
    const auto d = static_cast<_Float16>(amax / 127.0F);
    const auto bits = __builtin_bit_cast(unsigned short, d);
    if ((bits & 0x7c00U) == 0x7c00U)
        atomicOr(failure, 4U);
    return bits;
}
__device__ inline float kv8_scale_value(unsigned short bits) { return float(__builtin_bit_cast(_Float16, bits)); }
__device__ inline int kv8_code(float x, float d) {
    if (d == 0.0F)
        return 0;
    float c = __builtin_rintf(x / d);
    c = c > 127.0F ? 127.0F : (c < -127.0F ? -127.0F : c);
    return int(c);
}
} // namespace strix::gemma4::gpu
