// CUDA port of the Qwen routed-MoE prefill (A8 profile, --known-prefix-expert-union), bit-identical to the HIP path
// moe_experts_known_prefill_rows_launch -> known_moe_prefill_enqueue (src/hip_moe_experts_launcher.hip:2907) ->
// moe_q4k_a8::known_rows::enqueue_prefill(form -1 = iu8 WMMA) (src/hip_moe_q4k_a8_known_rows.hip:1492).
//
// Kernels, in stream order (one launch each per layer):
//   prologue  = prefill_prologue<Bf16=true>        (src/hip_moe_q4k_a8_probe.hip:159, quantize_row :41)
//   csr       = build_prefill_csr_waves, cap 16     (src/hip_moe_q4k_a8_known_rows.hip:460) + this port's 64-assignment tiles
//   project   = project_prefill_wmma<Q4K|Q5K,16>    (:860, gate_up_chains :791, SwiGLU + h quantizer :885-909)
//   down      = down_prefill_wmma<Q51|Q80,16>       (:1005, down_chains :957)
//   merge     = known_moe_prefill_merge_kernel      (src/hip_moe_experts_launcher.hip:2660)
//   commit    = known_moe_prefill_commit_kernel     (:2684)
// Every (row, slot, output) value depends only on that (row, slot): the CUDA kernels tile the CSR in runs of up to 64
// assignments of one expert (HIP: 16) and compute each value with the HIP operation order (see qwen_moe.cu).
// Fixed model geometry: hidden 2560, expert FFN 640, 512 experts, top-k 10. Host API only: device pointers and sizes.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "qwen_amd_math.cuh"

namespace strix::cuda::qwen_moe {

inline constexpr std::uint32_t hidden = 2560, ffn = 640, experts = 512, top_k = 10;
inline constexpr std::uint32_t hip_batch = 16, tile_rows = 64;
// Row status / summary bits (include/strix/hip_moe_q4k_a8_probe.hpp:6, known_rows.hip:75).
inline constexpr std::uint32_t invalid_route = 1, nonfinite_activation = 2, invalid_delta = 4, projection_nonfinite = 8;
// ggml type ids.
inline constexpr std::uint32_t type_q5_1 = 7, type_q8_0 = 8, type_q4_k = 12, type_q5_k = 13;

// Row r of a view lives at base + r * stride (bytes). Every base and stride must be 16-byte aligned.
template <class T> struct Strided { T* base = nullptr; std::uint64_t stride = 0; };

// CSR workspace (q51_known_rows::scratch_layout equivalent) plus this port's tiles.
struct Csr {
    std::uint32_t* summary = nullptr;   // [32] (summary words of build_prefill_csr_waves)
    std::uint32_t* offsets = nullptr;   // [experts + 1]
    std::uint32_t* touched = nullptr;   // [experts]
    std::uint32_t* rows = nullptr;      // [A]  A = rows * top_k
    std::uint32_t* slots = nullptr;     // [A]
    std::uint32_t* batch_experts = nullptr;  // [max_batches]     HIP batches of <= 16 (not consumed by this port)
    std::uint32_t* batch_begins = nullptr;   // [max_batches + 1]
    std::uint32_t* tile_count = nullptr;     // [1]
    std::uint32_t* tile_experts = nullptr;   // [max_tiles]       tiles of <= 64 consumed by project/down
    std::uint32_t* tile_begins = nullptr;    // [max_tiles + 1]
};
struct CsrSizes { std::uint32_t assignments, max_batches, max_tiles; };
[[nodiscard]] CsrSizes csr_sizes(std::uint32_t rows);

// The per-row views of the HIP KnownMoePrefillPrepared (prepare_known_moe_prefill, hip_moe_experts_launcher.hip:2884-2904).
struct Rows {
    Strided<const std::uint16_t> input;        // BF16 [hidden]
    Strided<const std::uint32_t> routes;       // [top_k]
    Strided<const float> route_weights;        // [top_k]
    Strided<std::int8_t> codes;                // [hidden]
    Strided<float> deltas;                     // [hidden / 32]
    Strided<std::int32_t> sums;                // [hidden / 32]
    Strided<std::uint32_t> status;             // [1]
    Strided<float> intermediates;              // [top_k * 2 * ffn]: hv (or g) | u per slot; base may be null (not written)
    Strided<float> accumulation;               // [clear_words >= hidden]: zeroed by the prologue, merge result
    Strided<std::int8_t> down_codes;           // [top_k * ffn]
    Strided<float> down_deltas;                // [top_k * ffn / 32]
    Strided<std::int32_t> down_sums;           // [top_k * ffn / 32]
    Strided<float> rawdots;                    // [top_k * hidden]
    Strided<std::uint16_t> output;             // BF16 [hidden]
    Csr csr;
};

struct Weights {
    std::uint32_t gate_type = type_q4_k;       // 12 or 13 (gate and up share it)
    const std::uint8_t* gate = nullptr;        // [experts][ffn][hidden/256 blocks]
    const std::uint8_t* up = nullptr;
    std::uint32_t down_type = type_q5_1;       // 7 or 8
    const std::uint8_t* down = nullptr;        // [experts][hidden][ffn/32 blocks]
};

// The single steps (tests) and the whole routed MoE of one layer. Any rows >= 1 is accepted (the HIP entry requires
// 16 <= rows <= prefill_max_rows, rows % 8 == 0; the bits of a row never depend on rows). record: optional device
// DeferredMoeFirstError {domain, bits} (include/strix/deferred_moe_status.hpp).
void enqueue_prologue(const Rows&, std::uint32_t rows, std::uint32_t clear_words, cudaStream_t);
void enqueue_csr(const Rows&, std::uint32_t rows, cudaStream_t);
void enqueue_project(const Rows&, std::uint32_t rows, const Weights&, const amdmath::Tables&, cudaStream_t);
void enqueue_down(const Rows&, std::uint32_t rows, const Weights&, cudaStream_t);
void enqueue_merge(const Rows&, std::uint32_t rows, cudaStream_t);
void enqueue_commit(const Rows&, std::uint32_t rows, std::uint32_t* record, cudaStream_t);
void enqueue(const Rows&, std::uint32_t rows, std::uint32_t clear_words, const Weights&, const amdmath::Tables&,
             std::uint32_t* record, cudaStream_t);
// True when the expert types are not the A8 profile's (gate Q4_K/Q5_K with down Q5_1/Q8_0): enqueue() then runs the native
// FAST path (any cuda_qwen_quant.cuh type; needs Rows::intermediates for the FP32 h values).
[[nodiscard]] bool native_types(const Weights&);

}  // namespace strix::cuda::qwen_moe
