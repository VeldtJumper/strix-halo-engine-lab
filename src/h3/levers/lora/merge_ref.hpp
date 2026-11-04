#pragma once
// Turbo LoRA (identity h3-turbo-v4-bypass): host reference pieces.
//   TorchCpuGenerator: the reference's CPU generator (MT19937, init_genrand(seed)); rand_f32() = (u32 & 0xffffff) * 2^-24,
//     one draw per element in row-major order (the reference's rand on a contiguous CPU tensor).
//   string_to_seed: CRC-32 (reflected, poly 0xEDB88320, init/final 0xFFFFFFFF) of the weight key.
//   merge_fc2_reference: the fc2 merge in exact arithmetic with the reference's fp32 rounding points (see the CPU test);
//     the shipped merge is the merge tool's output (tools/h3/levers/lora/merge_fc2.py).
//   bypass_ref: the run-time side branch of the reference's default mode for one output element row:
//     t = bf16(sum_k x A[r][k]) (r < rank), l = bf16(sum_r t B[o][r]), y = bf16(fma(s, l, y_base)).
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "h3/gemm/ref.hpp"

namespace strix::h3::levers::lora {

class TorchCpuGenerator {
public:
    explicit TorchCpuGenerator(std::uint32_t seed) {
        mt_[0] = seed;
        for (unsigned i = 1; i < 624U; ++i) mt_[i] = 1812433253U * (mt_[i - 1] ^ (mt_[i - 1] >> 30U)) + i;
        idx_ = 624U;
    }
    std::uint32_t next_u32() {
        if (idx_ >= 624U) twist();
        std::uint32_t y = mt_[idx_++];
        y ^= y >> 11U;
        y ^= (y << 7U) & 0x9d2c5680U;
        y ^= (y << 15U) & 0xefc60000U;
        y ^= y >> 18U;
        return y;
    }
    float rand_f32() { return static_cast<float>(next_u32() & 0xffffffU) * 0x1p-24F; }

private:
    void twist() {
        for (unsigned i = 0; i < 624U; ++i) {
            const std::uint32_t y = (mt_[i] & 0x80000000U) | (mt_[(i + 1U) % 624U] & 0x7fffffffU);
            mt_[i] = mt_[(i + 397U) % 624U] ^ (y >> 1U) ^ ((y & 1U) ? 0x9908b0dfU : 0U);
        }
        idx_ = 0;
    }
    std::array<std::uint32_t, 624> mt_{};
    unsigned idx_ = 624U;
};

inline std::uint32_t string_to_seed(const std::string& s) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (unsigned char c : s) {
        crc ^= c;
        for (int i = 0; i < 8; ++i) crc = (crc & 1U) ? (crc >> 1U) ^ 0xEDB88320U : crc >> 1U;
    }
    return crc ^ 0xFFFFFFFFU;
}

// In-place exact rotation of one 256-group by H = kron(H4, H4, H4, H4) / 16 (symmetric: W @ H^T == W @ H).
inline void rotate256_f64(double* v) {
    for (unsigned stride = 1U; stride < 256U; stride *= 4U)
        for (unsigned base = 0; base < 256U; ++base) {
            if ((base / stride) % 4U) continue;
            const double a0 = v[base], a1 = v[base + stride], a2 = v[base + 2U * stride], a3 = v[base + 3U * stride];
            const double s = (a0 + a1) + (a2 + a3);
            v[base] = s - 2.0 * a3;
            v[base + stride] = s - 2.0 * a2;
            v[base + 2U * stride] = s - 2.0 * a1;
            v[base + 3U * stride] = s - 2.0 * a0;
        }
    for (unsigned i = 0; i < 256U; ++i) v[i] *= 0.0625;
}

// q8 [n][k] int8 + s8 [n] (checkpoint), A [rank][k] bf16, B [n][rank] bf16 -> q [n][k] int8 + s [n] (merged).
inline void merge_fc2_reference(const std::int8_t* q8, const float* s8, const std::uint16_t* a, const std::uint16_t* b, std::uint32_t n,
    std::uint32_t k, std::uint32_t rank, float strength, std::uint32_t seed, std::int8_t* q, float* s) {
    TorchCpuGenerator gen(seed);
    std::vector<double> w(k), af(static_cast<std::size_t>(rank) * k);
    for (std::size_t i = 0; i < af.size(); ++i) af[i] = gemm::bf16_to_f32(a[i]);
    std::vector<float> wr(k);
    const double st = static_cast<double>(strength);  // the scalar multiplies the fp32 delta (fp32 opmath)
    for (std::uint32_t r = 0; r < n; ++r) {
        // dequantize: fp32(q * s), inverse rotation
        for (std::uint32_t c = 0; c < k; ++c) w[c] = static_cast<double>(static_cast<float>(q8[static_cast<std::size_t>(r) * k + c]) * s8[r]);
        for (std::uint32_t g = 0; g < k / 256U; ++g) rotate256_f64(w.data() + 256U * g);
        // + strength * (B @ A) (fp32 delta: B @ A rounded to fp32, times fp32(strength), rounded, added in fp32)
        for (std::uint32_t c = 0; c < k; ++c) {
            double d = 0.0;
            for (std::uint32_t j = 0; j < rank; ++j) d += static_cast<double>(gemm::bf16_to_f32(b[static_cast<std::size_t>(r) * rank + j])) * af[static_cast<std::size_t>(j) * k + c];
            const float df = static_cast<float>(static_cast<float>(d) * static_cast<float>(st));
            w[c] = static_cast<double>(static_cast<float>(static_cast<float>(w[c]) + df));
        }
        // requantize: rotation, per-row absmax / 127, fp32 x / scale + rand, floor, clamp
        for (std::uint32_t g = 0; g < k / 256U; ++g) rotate256_f64(w.data() + 256U * g);
        float amax = 0.0F;
        for (std::uint32_t c = 0; c < k; ++c) { wr[c] = static_cast<float>(w[c]); amax = std::fmax(amax, std::fabs(wr[c])); }
        float sc = amax / 127.0F;
        if (!(sc >= 1e-30F)) sc = 1e-30F;
        s[r] = sc;
        for (std::uint32_t c = 0; c < k; ++c) {
            const float t = wr[c] / sc + gen.rand_f32();
            float f = std::floor(t);
            f = f < -128.0F ? -128.0F : f > 127.0F ? 127.0F : f;
            q[static_cast<std::size_t>(r) * k + c] = static_cast<std::int8_t>(f);
        }
    }
}

}  // namespace strix::h3::levers::lora
