#pragma once
// IEEE binary16 <-> binary32 on the host (round to nearest even, subnormals, inf/nan), and the rounding
// points the reference decoder uses: every elementwise op on FP16 tensors computes in FP32 and rounds once.

#include <cstdint>
#include <cstring>

namespace strix::h3::vae {

using half_bits = std::uint16_t;

inline float bits_f32(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline std::uint32_t f32_bits(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }

inline float h2f(half_bits h) {
    const std::uint32_t sign = (std::uint32_t(h) & 0x8000U) << 16U;
    std::uint32_t exp = (h >> 10U) & 0x1fU, man = h & 0x3ffU;
    if (exp == 0x1fU) return bits_f32(sign | 0x7f800000U | (man << 13U));
    if (exp == 0U) {
        if (man == 0U) return bits_f32(sign);
        // subnormal: value = man * 2^-24 (exact in FP32)
        float v = float(man) * bits_f32(0x33800000U);
        return sign ? -v : v;
    }
    return bits_f32(sign | ((exp + 112U) << 23U) | (man << 13U));
}

inline half_bits f2h(float f) {
    const std::uint32_t u = f32_bits(f);
    const std::uint32_t sign = (u >> 16U) & 0x8000U;
    const std::uint32_t a = u & 0x7fffffffU;
    if (a >= 0x7f800000U) return half_bits(sign | 0x7c00U | (a > 0x7f800000U ? 0x200U | ((a >> 13U) & 0x3ffU) : 0U));
    if (a >= 0x477ff000U) return half_bits(sign | 0x7c00U);  // rounds to >= 65520: inf
    if (a < 0x38800000U) {                                    // below the smallest normal half 2^-14
        if (a < 0x33000000U) return half_bits(sign);          // < 2^-25: rounds to 0 (2^-25 exactly ties to even 0)
        const std::uint32_t e = a >> 23U, m = (a & 0x7fffffU) | 0x800000U;
        const std::uint32_t shift = 126U - e;                 // e in [102, 112]: shift 14..24
        const std::uint32_t q = m >> shift, rem = m & ((1U << shift) - 1U), halfway = 1U << (shift - 1U);
        const std::uint32_t r = q + ((rem > halfway || (rem == halfway && (q & 1U))) ? 1U : 0U);
        return half_bits(sign | r);
    }
    std::uint32_t r = a - 0x38000000U;  // rebias exponent 127 -> 15
    const std::uint32_t q = r >> 13U, rem = r & 0x1fffU;
    std::uint32_t out = q + ((rem > 0x1000U || (rem == 0x1000U && (q & 1U))) ? 1U : 0U);
    return half_bits(sign | out);
}

// FP32 value rounded to FP16 and back (the value an FP16 tensor op stores).
inline float rh(float f) { return h2f(f2h(f)); }

}  // namespace strix::h3::vae
