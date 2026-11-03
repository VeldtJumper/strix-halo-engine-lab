#pragma once
// Reproduction of the reference's CPU normal generator for fp32 tensors of >= 16 elements (host, scalar code):
// MT19937 (standard seeding of a 32-bit seed), uniforms u = (r & 0xffffff) * 2^-24 for every element first, then
// Box-Muller over blocks of 16 (u1 = 1 - d[j], u2 = d[j + 8], radius = sqrt(-2 log u1), theta = 2 pi u2,
// d[j] = radius cos theta, d[j + 8] = radius sin theta) with the 8-lane polynomial log / sincos of the library's AVX2
// path (Cephes coefficients, separate fp32 multiplies and adds), and the tail block recomputed from 16 fresh uniforms
// on the last 16 elements. Successive calls continue the same generator (video noise, then audio noise).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace strix::h3::block {

namespace rng_detail {
inline float as_f(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline std::uint32_t as_u(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
// log of the 8-lane polynomial path (x > 0 finite).
inline float log_ps(float x) {
    const float one = 1.0F;
    const bool invalid = !(x > 0.0F);
    if (x < as_f(0x00800000U)) x = as_f(0x00800000U);
    std::int32_t imm0 = static_cast<std::int32_t>(as_u(x) >> 23);
    x = as_f((as_u(x) & ~0x7f800000U) | as_u(0.5F));
    imm0 -= 0x7f;
    float e = static_cast<float>(imm0);
    e = e + one;
    const bool mask = x < 0.707106781186547524F;
    const float tmp0 = mask ? x : 0.0F;
    x = x - one;
    e = e - (mask ? one : 0.0F);
    x = x + tmp0;
    const float z = x * x;
    float y = 7.0376836292E-2F;
    y = y * x; y = y + -1.1514610310E-1F;
    y = y * x; y = y + 1.1676998740E-1F;
    y = y * x; y = y + -1.2420140846E-1F;
    y = y * x; y = y + 1.4249322787E-1F;
    y = y * x; y = y + -1.6668057665E-1F;
    y = y * x; y = y + 2.0000714765E-1F;
    y = y * x; y = y + -2.4999993993E-1F;
    y = y * x; y = y + 3.3333331174E-1F;
    y = y * x;
    y = y * z;
    float tmp = e * -2.12194440e-4F;
    y = y + tmp;
    tmp = z * 0.5F;
    y = y - tmp;
    tmp = e * 0.693359375F;
    x = x + y;
    x = x + tmp;
    if (invalid) return as_f(as_u(x) | 0xffffffffU);
    return x;
}
inline void sincos_ps(float xin, float* s, float* c) {
    float x = as_f(as_u(xin) & 0x7fffffffU);
    std::uint32_t sign_bit_sin = as_u(xin) & 0x80000000U;
    float y = x * 1.27323954473516F;
    std::int32_t imm2 = static_cast<std::int32_t>(y);  // truncation
    imm2 = imm2 + 1;
    imm2 = imm2 & ~1;
    y = static_cast<float>(imm2);
    std::int32_t imm4 = imm2;
    const std::uint32_t swap_sign_bit_sin = static_cast<std::uint32_t>(imm2 & 4) << 29;
    const bool poly_mask = (imm2 & 2) == 0;
    const float xmm1 = y * -0.78515625F, xmm2 = y * -2.4187564849853515625e-4F, xmm3 = y * -3.77489497744594108e-8F;
    x = x + xmm1;
    x = x + xmm2;
    x = x + xmm3;
    imm4 = imm4 - 2;
    imm4 = ~imm4 & 4;
    const std::uint32_t sign_bit_cos = static_cast<std::uint32_t>(imm4) << 29;
    sign_bit_sin ^= swap_sign_bit_sin;
    const float z = x * x;
    float yc = 2.443315711809948E-005F;
    yc = yc * z; yc = yc + -1.388731625493765E-003F;
    yc = yc * z; yc = yc + 4.166664568298827E-002F;
    yc = yc * z;
    yc = yc * z;
    const float tmp = z * 0.5F;
    yc = yc - tmp;
    yc = yc + 1.0F;
    float y2 = -1.9515295891E-4F;
    y2 = y2 * z; y2 = y2 + 8.3321608736E-3F;
    y2 = y2 * z; y2 = y2 + -1.6666654611E-1F;
    y2 = y2 * z;
    y2 = y2 * x;
    y2 = y2 + x;
    const float ysin2 = poly_mask ? y2 : 0.0F;
    const float ysin1 = poly_mask ? 0.0F : yc;
    y2 = y2 - ysin2;
    yc = yc - ysin1;
    const float r1 = ysin1 + ysin2;
    const float r2 = yc + y2;
    *s = as_f(as_u(r1) ^ sign_bit_sin);
    *c = as_f(as_u(r2) ^ sign_bit_cos);
}
inline void normal_fill_16(float* d) {
    const float two_pi = static_cast<float>(2.0 * 3.14159265358979323846);
    for (int j = 0; j < 8; ++j) {
        const float u1 = 1.0F - d[j];
        const float u2 = d[j + 8];
        const float radius = std::sqrt(-2.0F * log_ps(u1));
        const float theta = two_pi * u2;
        float sn, cs;
        sincos_ps(theta, &sn, &cs);
        d[j] = radius * cs;  // fma(n, std = 1, mean = 0) is exact
        d[j + 8] = radius * sn;
    }
}
}  // namespace rng_detail

class TorchCpuGenerator {
public:
    explicit TorchCpuGenerator(std::uint64_t seed) : mt_(static_cast<std::uint32_t>(seed & 0xffffffffULL)) {}
    float uniform() { return static_cast<float>(mt_() & 0xffffffU) * (1.0F / 16777216.0F); }
    // randn(n) fp32 (n >= 16, the vectorised path).
    std::vector<float> randn(std::size_t n) {
        if (n < 16) throw std::runtime_error("randn: the scalar path (n < 16) is not reproduced");
        std::vector<float> d(n);
        for (auto& v : d) v = uniform();
        for (std::size_t i = 0; i + 16 <= n; i += 16) rng_detail::normal_fill_16(d.data() + i);
        if (n % 16 != 0) {
            float* t = d.data() + n - 16;
            for (int i = 0; i < 16; ++i) t[i] = uniform();
            rng_detail::normal_fill_16(t);
        }
        return d;
    }

private:
    std::mt19937 mt_;
};

}  // namespace strix::h3::block
