#pragma once
#include "gemma4/ref/target.hpp"
#include "test.hpp"
#include <random>

namespace fixture {
using namespace strix::gemma4;
inline void put16(std::uint8_t* p, std::uint16_t x) {
    p[0] = std::uint8_t(x);
    p[1] = std::uint8_t(x >> 8U);
}
inline std::vector<std::uint8_t> packed(WeightFormat f, unsigned outputs, unsigned width, unsigned seed = 471) {
    const auto stride = packed_row_bytes(f, width);
    std::vector<std::uint8_t> b(std::size_t(outputs) * stride);
    std::mt19937 rng(seed);
    if (f == WeightFormat::bf16 || f == WeightFormat::f32) {
        for (std::size_t i = 0; i < std::size_t(outputs) * width; ++i) {
            const float x = float(int(rng() % 127) - 63) * 0.0009765625F;
            if (f == WeightFormat::bf16)
                put16(b.data() + i * 2, bf16_bits(x));
            else
                std::memcpy(b.data() + i * 4, &x, 4);
        }
    } else {
        for (auto& x : b)
            x = std::uint8_t(rng());
        const unsigned block = f == WeightFormat::q8_0 ? 32U : 256U,
                       size = f == WeightFormat::q8_0 ? 34U : (f == WeightFormat::q4_k ? 144U : 210U);
        for (std::size_t i = 0; i < b.size(); i += size) {
            // Nontrivial FP16 mantissas exercise coefficient and embedding rounding boundaries.
            put16(b.data() + i + (f == WeightFormat::q6_k ? 208 : 0), std::uint16_t(0x1401U + rng() % 1023U));
            if (f == WeightFormat::q4_k)
                put16(b.data() + i + 2, std::uint16_t(0x1001U + rng() % 1023U));
        }
        (void)block;
    }
    return b;
}
inline ModelManifest tiny(unsigned count = 42, bool quant = false, unsigned hidden_width = 0) {
    ModelManifest m;
    auto& c = m.config;
    c.hidden = hidden_width ? hidden_width : (quant ? 256U : 32U);
    c.layers = count;
    c.inter = quant ? 256U : 48U;
    c.vocab = 37;
    c.heads = 4;
    c.kv_heads = 1;
    c.ple_dim = quant ? 256U : 16U;
    c.window = 512;
    c.max_positions = 1024;
    for (unsigned l = 0; l < count; ++l) {
        const bool g = l % 6 == 5;
        c.layer.push_back({g ? AttentionKind::global : AttentionKind::sliding, quant ? (g ? 512U : 256U) : 8U,
                           l < 24 ? l : (g ? 23U : 22U)});
    }
    std::mt19937 rng(55);
    for (const auto& [name, shape] : required_tensors(c)) {
        const auto outputs = shape.size() == 2 ? unsigned(shape[0]) : 1U, inputs = unsigned(shape.back());
        const bool matrix = shape.size() == 2;
        const auto f =
            quant && matrix
                ? (name.find("embed_tokens") != std::string::npos
                       ? WeightFormat::q8_0
                       : (name.find("per_layer_model_projection") != std::string::npos
                              ? WeightFormat::bf16
                              : (name.find("down_proj") != std::string::npos || name.find("v_proj") != std::string::npos
                                     ? WeightFormat::q6_k
                                     : WeightFormat::q4_k)))
                : (matrix ? WeightFormat::bf16 : WeightFormat::f32);
        auto b = std::make_shared<std::vector<std::uint8_t>>(packed(f, outputs, inputs, unsigned(rng())));
        if (!matrix)
            for (unsigned i = 0; i < inputs; ++i) {
                float x = name.ends_with("layer_scalar") ? 0.875F : 0.5F + float(int(rng() % 17) - 8) * 0.00390625F;
                std::memcpy(b->data() + std::size_t(i) * 4, &x, 4);
            }
        m.tensors.push_back({name, shape, {b->data(), outputs, inputs, packed_row_bytes(f, inputs), f}, {}});
        m.storage.push_back(b);
    }
    return m;
}
struct Tables {
    std::vector<float> sliding, global;
    explicit Tables(unsigned count = 1024) : sliding(std::size_t(count) * 256), global(std::size_t(count) * 512) {
        for (unsigned p = 0; p < count; ++p) {
            for (unsigned i = 0; i < 128; ++i) {
                sliding[std::size_t(p) * 256 + 2 * i] = (p % 4 == 0 ? 1.0F : 0.5F);
                sliding[std::size_t(p) * 256 + 2 * i + 1] = (p % 4 == 0 ? 0.0F : 0.25F);
            }
            for (unsigned i = 0; i < 256; ++i) {
                global[std::size_t(p) * 512 + 2 * i] = (i >= 64 || p % 4 == 0 ? 1.0F : 0.5F);
                global[std::size_t(p) * 512 + 2 * i + 1] = (i >= 64 || p % 4 == 0 ? 0.0F : 0.25F);
            }
        }
    }
    RopeTables view() const { return {sliding, global, unsigned(sliding.size() / 256), "synthetic-dyadic-tables"}; }
};
} // namespace fixture
