#pragma once

// g4-a8-v1 prefill linears (contract: gemma4/kernels/a8.hpp). Same bytes as the decode a8 linears for
// every row: the per-32-group integer products come from IU8 WMMA (exact int32, so equal to sudot4),
// and each group term and the 16-slot / 8-4-2-1 reduction are the declared FP32 operations. Two
// exact rewrites keep the per-element work small:
//   d * float(sc * I)    == RN((d * sc) * float(I))      (FP16 value x 6-bit integer is exact in FP32)
//   dmin * float(m * S)  == RN((dmin * m) * float(S))
// so d*sc and dmin*m are tabulated once per (output, group) at prepare time.
#include "gemma4/kernels/device.hpp"
#include "gemma4/contracts/g4_types.hpp"
#include <cstddef>
#include <cstdint>

namespace strix::gemma4::prefill {

// One-time weight preparation: int8 panels [outputs16][inputs] (Q4_K q, Q6_K q-32, Q8_0 c) and per
// (output, group) FP32 pairs [outputs16][inputs/32]: Q4_K {d*sc, dmin*m}, Q6_K {d, s0 | s1 << 8 bits},
// Q8_0 {d, 0}.
class A8Prepared {
  public:
    explicit A8Prepared(LinearView weights, gpu::Queue queue = nullptr);
    [[nodiscard]] LinearView weights() const { return weights_; }
    [[nodiscard]] const gpu::Buffer& panels() const { return panels_; }
    [[nodiscard]] const gpu::Buffer& scales() const { return scales_; }
  private:
    LinearView weights_;
    gpu::Buffer panels_, scales_;
};
[[nodiscard]] bool a8_prefill_supported(LinearView weights);
// Kernel form for tests and benchmarks: 0 unstaged control, 1 staged 32x32, 2 staged 32x64, 3 staged 64x64,
// 4 staged 32x64 with two load stages in flight, 5-9 wide 8/16-wave forms (same bytes; see a8_linear.hip).
void set_a8_kernel(unsigned form);
// Scratch for a8_linear: activation codes and per-group {delta, float(S)} of rows rounded up to 128.
[[nodiscard]] std::size_t a8_scratch_bytes(unsigned rows, unsigned inputs);
// out[rows][outputs] = the g4-a8-v1 dot of every row (FP32, no epilogue). failure bit 1 = nonfinite input.
// quantize = false reuses the codes a previous call left in scratch for the same rows and width.
void a8_linear(const A8Prepared& prepared, const float* x, unsigned rows, float* out, void* scratch,
               std::uint32_t* failure, gpu::Queue queue = nullptr, bool quantize = true);
// FFN down: the decode target's product gelu_tanh(gate[i]) * up[i] quantized in the same launch (the product
// itself is not stored), then the a8 linear of the product rows.
void a8_product_linear(const A8Prepared& prepared, float* gate, const float* up, unsigned rows, float* out, void* scratch,
                       std::uint32_t* failure, gpu::Queue queue = nullptr);

} // namespace strix::gemma4::prefill
