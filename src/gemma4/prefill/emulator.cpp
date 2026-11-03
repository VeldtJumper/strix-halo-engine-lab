#include "emulator.hpp"
#include "gemma4/exl3/emulator.hpp"
#include "gemma4/model/model.hpp"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <stdexcept>

namespace strix::gemma4::prefill {
namespace {
struct Product {
    std::uint32_t magnitude = 0;
    int exponent = 0;
    bool negative = false;
};
Product product(std::uint16_t a, std::uint16_t b) {
    const unsigned ae = (a >> 10U) & 31U, be = (b >> 10U) & 31U;
    if (ae == 31 || be == 31) throw std::invalid_argument("nonfinite WMMA half operand");
    const unsigned am = (a & 1023U) | (ae ? 1024U : 0U), bm = (b & 1023U) | (be ? 1024U : 0U);
    return {am * bm, (ae ? int(ae) - 25 : -24) + (be ? int(be) - 25 : -24), ((a ^ b) & 0x8000U) != 0};
}
std::uint64_t aligned(std::uint32_t magnitude, int shift) {
    if (!magnitude) return 0;
    if (shift >= 0) return std::uint64_t(magnitude) << unsigned(shift);
    return shift <= -32 ? 0 : magnitude >> unsigned(-shift);
}
std::uint64_t round_shift(std::uint64_t m, int shift) {
    if (shift <= 0) return m << unsigned(-shift);
    if (shift >= 64) return 0;
    const auto q = m >> unsigned(shift), remainder = m & ((std::uint64_t(1) << unsigned(shift)) - 1);
    const auto midpoint = std::uint64_t(1) << unsigned(shift - 1);
    return q + unsigned(remainder > midpoint || (remainder == midpoint && (q & 1U)));
}
// Round the exact integer dyadic once to FP32 RNE, without host FP arithmetic.
float rounded(std::int64_t value, int exponent) {
    if (!value) return 0.0F;
    const auto sign = value < 0 ? 0x80000000U : 0U;
    const auto m = std::uint64_t(value < 0 ? -value : value);
    const int high = 63 - int(std::countl_zero(m));
    int e = high + exponent;
    if (e < -126) return bit_cast<float>(sign | std::uint32_t(round_shift(m, -149 - exponent)));
    auto q = round_shift(m, high - 23);
    if (q == 0x1000000U) { q >>= 1U; ++e; }
    if (e > 127) throw std::runtime_error("nonfinite WMMA accumulator");
    return bit_cast<float>(sign | (unsigned(e + 127) << 23U) | (std::uint32_t(q) & 0x7fffffU));
}
float dot2(const Product& a, const Product& b, float accumulator) {
    const auto bits = bit_cast<std::uint32_t>(accumulator), field = (bits >> 23U) & 255U;
    if (field == 255U) throw std::invalid_argument("nonfinite WMMA accumulator input");
    // The matrix unit flushes a subnormal FP32 C input, retaining its sign for
    // operand alignment. FP16 subnormal products keep their unnormalized exponents.
    const auto cm = field ? (bits & 0x7fffffU) | 0x800000U : 0U;
    const int ce = int(field) - 150;
    const bool negative = (bits & 0x80000000U) != 0;
    if (!cm && !a.magnitude && !b.magnitude) return 0.0F;
    int unit = cm ? ce - 3 : -1000;
    if (a.magnitude) unit = std::max(unit, a.exponent - 4);
    if (b.magnitude) unit = std::max(unit, b.exponent - 4);
    std::int64_t sum = std::int64_t(aligned(cm, ce - unit));
    for (const auto p : {a, b}) {
        const auto q = std::int64_t(aligned(p.magnitude, p.exponent - unit));
        // Opposite-sign terms use one's complement after truncating magnitude;
        // the extra -1 also applies to signed zero. There is no sticky-bit repair.
        sum += p.negative == negative ? q : -q - 1;
    }
    return rounded(negative ? -sum : sum, unit);
}
} // namespace
float instruction(const std::array<std::uint16_t, 16>& a, const std::array<std::uint16_t, 16>& b, float accumulator) {
    for (unsigned k = 0; k < 16; k += 2) accumulator = dot2(product(a[k], b[k]), product(a[k + 1], b[k + 1]), accumulator);
    return accumulator;
}
std::vector<float> emulate_linear(LinearView w, std::span<const float> input, unsigned rows) {
    if (w.format == WeightFormat::exl3_mul1) return exl3::emulate_prefill(w, input, rows);
    if (!rows || rows > chunk_rows || !w.data || !w.inputs || w.inputs > 10752 || !w.outputs || w.outputs > vocab ||
        w.row_bytes < packed_row_bytes(w.format, w.inputs) || input.size() != std::size_t(rows) * w.inputs || std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("WMMA emulator geometry or rounding environment");
    const unsigned stride = ((w.inputs + 15U) / 16U) * 16U;
    std::vector<std::uint16_t> x(std::size_t(rows) * stride), weight(stride);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned i = 0; i < w.inputs; ++i)
            if (!half_bits(input[std::size_t(r) * w.inputs + i], x[std::size_t(r) * stride + i]))
                throw std::runtime_error("WMMA emulator activation overflow");
    std::vector<float> out(std::size_t(rows) * w.outputs);
    for (unsigned n = 0; n < w.outputs; ++n) {
        for (unsigned i = 0; i < w.inputs; ++i)
            if (!half_bits(weight_at(w, n, i), weight[i])) throw std::runtime_error("WMMA emulator weight overflow");
        for (unsigned r = 0; r < rows; ++r) {
            float sum = 0.0F;
            for (unsigned k = 0; k < stride; k += 16) {
                std::array<std::uint16_t, 16> a{}, b{};
                std::copy_n(x.data() + std::size_t(r) * stride + k, 16, a.data());
                std::copy_n(weight.data() + k, 16, b.data());
                sum = instruction(a, b, sum);
            }
            out[std::size_t(r) * w.outputs + n] = sum;
        }
    }
    return out;
}
std::vector<float> emulate_linear_v2(LinearView w, std::span<const float> input, unsigned rows, bool blocked) {
    if (!rows || rows > chunk_rows || !w.data || !w.inputs || w.inputs > 10752 || !w.outputs || w.outputs > vocab ||
        w.format == WeightFormat::exl3_mul1 || w.row_bytes < packed_row_bytes(w.format, w.inputs) ||
        input.size() != std::size_t(rows) * w.inputs || std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("WMMA v2 emulator geometry or rounding environment");
    const unsigned stride = ((w.inputs + 15U) / 16U) * 16U;
    std::vector<std::uint16_t> high(std::size_t(rows) * stride), low(high.size()), weight(stride);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned i = 0; i < w.inputs; ++i) {
            const float x = input[std::size_t(r) * w.inputs + i];
            auto& h = high[std::size_t(r) * stride + i];
            if (!half_bits(x, h)) throw std::runtime_error("WMMA v2 emulator activation overflow");
            const float rest = (x - fp16_value(h)) * v2_low_scale;
            if (!half_bits(rest, low[std::size_t(r) * stride + i])) throw std::runtime_error("WMMA v2 emulator low overflow");
        }
    std::vector<float> out(std::size_t(rows) * w.outputs);
    const unsigned block = blocked ? v2_block : stride;
    for (unsigned n = 0; n < w.outputs; ++n) {
        for (unsigned i = 0; i < w.inputs; ++i)
            if (!half_bits(weight_at(w, n, i), weight[i])) throw std::runtime_error("WMMA v2 emulator weight overflow");
        for (unsigned r = 0; r < rows; ++r) {
            float s = 0.0F;
            for (unsigned k0 = 0; k0 < stride; k0 += block) {
                float h = 0.0F, l = 0.0F;
                for (unsigned k = k0; k < std::min(stride, k0 + block); k += 16) {
                    std::array<std::uint16_t, 16> ah{}, al{}, b{};
                    std::copy_n(high.data() + std::size_t(r) * stride + k, 16, ah.data());
                    std::copy_n(low.data() + std::size_t(r) * stride + k, 16, al.data());
                    std::copy_n(weight.data() + k, 16, b.data());
                    h = instruction(ah, b, h); l = instruction(al, b, l);
                }
                const float scaled = l * v2_low_unscale;
                const float part = h + scaled;
                s = blocked ? s + part : part;
            }
            out[std::size_t(r) * w.outputs + n] = s;
        }
    }
    return out;
}
} // namespace strix::gemma4::prefill
