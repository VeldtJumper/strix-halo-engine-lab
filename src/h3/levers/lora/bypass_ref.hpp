#pragma once
// Host mirror of the LoRA bypass branch (no HIP; usable by CPU forwards and tests). Arithmetic in bypass.hpp.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "h3/gemm/ref.hpp"

namespace strix::h3::levers::lora {

constexpr std::uint32_t kRank = 64U;  // Turbo v4 rank (default of the rank parameters)

// y[rows][n] (dense) += branch, bit for bit with add_bypass on the device (a [rank][k], b [n][rank]).
inline void bypass_host(const std::uint16_t* x, std::uint32_t rows, std::uint32_t k, const std::uint16_t* a, const std::uint16_t* b, std::uint32_t n,
    float s, std::uint16_t* y, std::uint32_t rank = kRank) {
    std::vector<float> t(rank);
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t j = 0; j < rank; ++j) {
            float acc = 0.0F;
            for (std::uint32_t i = 0; i < k; ++i)
                acc = std::fma(gemm::bf16_to_f32(x[static_cast<std::size_t>(r) * k + i]), gemm::bf16_to_f32(a[static_cast<std::size_t>(j) * k + i]), acc);
            t[j] = gemm::bf16_to_f32(gemm::f32_to_bf16(acc));
        }
        for (std::uint32_t o = 0; o < n; ++o) {
            float acc = 0.0F;
            for (std::uint32_t j = 0; j < rank; ++j) acc = std::fma(t[j], gemm::bf16_to_f32(b[static_cast<std::size_t>(o) * rank + j]), acc);
            const float l = gemm::bf16_to_f32(gemm::f32_to_bf16(acc));
            std::uint16_t& out = y[static_cast<std::size_t>(r) * n + o];
            out = gemm::f32_to_bf16(std::fma(s, l, gemm::bf16_to_f32(out)));
        }
    }
}

}  // namespace strix::h3::levers::lora
