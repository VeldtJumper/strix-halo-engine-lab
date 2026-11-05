// Load-time transcoding of GGUF tensors (any cuda_qwen_quant.cuh type) into the storage the RTX kernels take: Q8_0
// (ggml's quantize_row_q8_0_ref rule on the exactly decoded values) or FP32 (exact for F16/BF16/F32). Used for files whose
// dense tensors are not Q8_0/F32 (the GSQ-RCO IQ build); a FAST path, not a node profile.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace strix::cuda::transcode {

[[nodiscard]] bool supported(std::uint32_t ggml_type);
[[nodiscard]] std::uint64_t q8_0_bytes(std::uint64_t elements);
// src/dst device pointers; elements a multiple of 32 (rows are whole blocks, so the tensor is a flat run of groups).
void to_q8_0(const void* src, std::uint32_t ggml_type, std::uint64_t elements, void* dst, cudaStream_t stream);
void to_f32(const void* src, std::uint32_t ggml_type, std::uint64_t elements, float* dst, cudaStream_t stream);

}  // namespace strix::cuda::transcode
