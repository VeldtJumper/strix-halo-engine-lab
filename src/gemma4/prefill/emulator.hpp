#pragma once
#include "prefill.hpp"
#include <array>

namespace strix::gemma4::prefill {
// One gfx1151 wave32 instruction for a single output element, including its C
// input. Half operands must be finite. The characterization fixtures bind the
// intra-instruction accumulator recipe; a host IEEE FMA chain is insufficient.
[[nodiscard]] float instruction(const std::array<std::uint16_t, 16>& a,
                                const std::array<std::uint16_t, 16>& b, float accumulator);
// Host-only bounded operator reference. It retains FP32 output and rounds both
// operands to FP16 before fixed, increasing K tiles, including masked tail tiles.
[[nodiscard]] std::vector<float> emulate_linear(LinearView weights, std::span<const float> input, unsigned rows);
// g4-wmma-pf-v2 corrected linear (blocked = hilo_k128, else hilo). Activations
// split into high = RNE16(x) and low = RNE16((x - high) * 2^12) (both exact FP32
// steps before the rounding); weights are the same FP16 panels as v1. Two
// independent accumulators per output run the v1 increasing-K16 instruction
// chain on high and low. part = high_acc + (low_acc * 2^-12), each FP32 RNE.
// Unblocked: y = part over all of K. Blocked: both accumulators restart at +0
// every 128 inputs (the last block may be shorter) and the block parts are added
// in increasing order: s = s + part (FP32 RNE, s starts at +0); y = s.
[[nodiscard]] std::vector<float> emulate_linear_v2(LinearView weights, std::span<const float> input, unsigned rows, bool blocked);

} // namespace strix::gemma4::prefill
