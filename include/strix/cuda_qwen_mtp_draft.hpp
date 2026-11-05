// CUDA kernels of the RTX MTP draft chain that have no exact family: Q8_0 routed experts of the MTP layer for one row,
// the draft head over a vocabulary subset, and its greedy selection with the top probability. Drafts only steer which
// tokens the exact verify window checks, so these kernels are free arithmetic (plain FP32 dot products in any order);
// they can never change committed tokens or logits. Host launch API over device pointers.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace strix::cuda::qwen_mtp_draft {

inline constexpr std::uint32_t kHidden = 2560, kFfn = 640, kTopK = 10;

// out[i] = scale * code of Q8_0 blocks (34 B: FP16 scale, 32 int8), elements a multiple of 32.
void q8_to_f32(const void* q8, std::uint64_t elements, float* out, cudaStream_t stream);
// Copies whole Q8_0 rows ids[i] (row_bytes each) of src into dst row i.
void gather_rows(const void* src, std::uint64_t row_bytes, const std::uint32_t* ids, std::uint32_t count, void* dst,
    cudaStream_t stream);
// One row through top-k routed Q8_0 experts: h[s] = silu(gate_e . x) * (up_e . x), out = bf16(sum_s w_s * down_e . h[s]).
// gate/up: [experts][640][2560] Q8_0, down: [experts][2560][640] Q8_0. scratch: kTopK * kFfn floats.
void routed_q8_one_row(const std::uint16_t* hidden, const std::uint32_t* routes, const float* route_weights, const void* gate,
    const void* up, const void* down, float* scratch, std::uint16_t* out, cudaStream_t stream);
// Greedy draft selection over subset logits [count]: token = ids[first max], probability = 1 / sum exp(l - max).
// Writes out[0] = token, reinterpret(out[1]) = probability, and also next_token (device word, may be null) = token.
void select(const float* logits, std::uint32_t count, const std::uint32_t* ids, std::uint32_t* out, std::uint32_t* next_token,
    cudaStream_t stream);

}  // namespace strix::cuda::qwen_mtp_draft
