#pragma once

// g4-a8-v1 device primitives (declaration and host oracle: a8.hpp). All pointers are device-resident.
#include "a8.hpp"
#include "ops.hpp"

namespace strix::gemma4::gpu {
// A8G32 quantization of rows x [rows, width]; width must be a multiple of 256.
void a8_quantize(const float* x, std::uint32_t rows, std::uint32_t width, A8Rows out, Queue queue = nullptr);
// Shared-pass linear for a quantized weight format. The epilogue fields (product, cap, extra matrices of
// the same format) apply as in linear_fused. Returns false, enqueueing nothing, for BF16/F32 views or a
// mismatched shared grid.
// With quant.codes set (outputs a multiple of 32, no extra matrices), the stored values are also
// quantized into quant (width = outputs), from blocks of 32 outputs.
bool a8_linear(LinearView w, A8Rows x, std::uint32_t rows, float* out, const LinearEpilogue& epilogue, Queue queue,
               std::uint32_t* failure, A8Rows quant = {});
// out = gelu_tanh(gate . x) * (up . x); both views must share one quantized format.
bool a8_gate_up(LinearView gate, LinearView up, A8Rows x, std::uint32_t rows, float* out, Queue queue,
                std::uint32_t* failure, A8Rows quant = {});
// FFN norm/residual + A8 PLE gate/activation. False enqueues nothing. C1..C4, hidden 2560,
// aligned F32/BF16 norm, pair A8 schedule only. Publishes residual and its A8 bytes in input;
// A8 slots must be disjoint. quant optionally receives the activation's A8 bytes, as in a8_linear.
// ticket starts at zero and is reset by the last consumer, including graph replays.
bool a8_ffn_ple_gate(const float* branch, LinearView norm, float* residual, LinearView gate,
                      const float* ple, std::uint32_t product_stride, std::uint32_t rows, float* out,
                      A8Rows input, std::uint32_t* failure, std::uint32_t* ticket, Queue queue,
                      A8Rows quant = {});
} // namespace strix::gemma4::gpu
