// CUDA port, bit-identical to the HIP A8-profile tiled prefill (gfx1151), of the Qwen3.8-Flash-Next hyper-connection
// mixes, the elementwise steps (token embedding, hyper expand, inject, MoE combine), the MoE router (logits + top-10
// selection) and the one-row Q8_0 LM head. Host launch API over device pointers; no engine types.
//
// Layouts (all row-major, rows contiguous):
//   expanded BF16 rows    [rows][10240]          (4 streams x 2560; stream s at s*2560)
//   hidden BF16 rows      [rows][2560]
//   normalized workspace  [rows][kNormalizedPitch] FP32: 10240 normalized values, then the 320 low values (HIP layout)
//   injection             [rows][4] FP32
//   Q8_0 tensors exactly as in the GGUF (34-byte blocks: FP16 scale, 32 int8 codes).
// Every kernel is stream ordered, has no atomics, and computes each (row, output) value with the HIP operation order.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "qwen_amd_math.cuh"

namespace strix::cuda::qwen_hc {

constexpr unsigned kHidden = 2560U, kStreams = 4U, kWide = 10240U, kRank = 320U;
constexpr unsigned kNormalizedPitch = kWide + kRank;  // floats per normalized-workspace row
constexpr unsigned kExperts = 512U, kTopK = 10U, kVocab = 248320U;

// Per-call (or per-load) repack of one mix's Q8_0 down [320][10240] and up [10240][320] weights into coalesced
// layouts. Pure data movement: the values are the same FP16 scales (widened exactly) and int8 codes.
struct HcPacked {
    std::int8_t* down_codes;   // [320 outputs][10240 features]
    float* down_scales;        // [320 outputs][320 blocks]
    std::uint32_t* up_codes;   // [80 k-quads][4 streams][2560 features]: codes k = 4q .. 4q+3 of up row s*2560+f
    float* up_scales;          // [10 blocks][4 streams][2560 features]
};
[[nodiscard]] std::size_t hc_packed_bytes() noexcept;
[[nodiscard]] HcPacked hc_packed_view(void* base) noexcept;  // base: cudaMalloc'ed, hc_packed_bytes(), 16-byte aligned
void hc_pack(const void* down_q8, const void* up_q8, const HcPacked& packed, cudaStream_t stream);

struct HcMixArgs {
    const std::uint16_t* input = nullptr;  // expanded BF16 [rows][10240]
    const float* norm = nullptr;           // hc_*_norm / output_hc_norm F32 [10240]
    const float* inject = nullptr;         // hc_*_inject F32 [4][10240]; nullptr for final_hyper_mix
    HcPacked packed{};                     // from hc_pack of hc_*_down / hc_*_up
    float epsilon = 1e-6F;                 // model rms_norm_epsilon
    std::uint32_t rows = 0;
    float* normalized = nullptr;           // workspace [rows][kNormalizedPitch]
    float* injection = nullptr;            // [rows][4]; ignored when inject == nullptr
    std::uint16_t* output = nullptr;       // BF16 [rows][2560]
    float* gates = nullptr;                // optional scratch [rows][4][2560]: split the sigmoid/mix epilogue into its own
                                           // full-occupancy kernel (same values; nullptr = fused output kernel)
};
// rows_prepare_hyper_norm + wide2_splitk_hyper_projection<8,Vopd> + wide2_hyper_output<u16,32,RowsFirst> (3 launches).
// stages (test/timing seam): bit 0 norm, bit 1 projection, bit 2 output (gates), bit 3 split epilogue alone.
void hc_mix(const HcMixArgs& args, const amdmath::Tables& tables, cudaStream_t stream, unsigned stages = 15U);

// embedding_kernel<u16, q8_0>: out[r][f] = bf16(scale * code) of row ids[r] of token_embd (Q8_0 [vocab][2560]).
void token_embedding(const void* token_embd_q8, const std::uint32_t* ids, std::uint32_t rows, std::uint16_t* out,
    cudaStream_t stream);
// hyper_expand_kernel<u16>: out[r][s*2560+f] = in[r][f].
void hyper_expand(const std::uint16_t* in, std::uint32_t rows, std::uint16_t* out, cudaStream_t stream);
// inject_rows_kernel / inject_kernel<u16>: out[r][s*2560+f] = bf16(fma(hidden[r][f], injection[r][s], hyper[r][s*2560+f])).
void inject(const std::uint16_t* hyper, const std::uint16_t* hidden, const float* injection, std::uint32_t rows,
    std::uint16_t* out, cudaStream_t stream);
// combine_rows_kernel<u16>: out[r][f] = bf16(shared + routed).
void combine(const std::uint16_t* shared, const std::uint16_t* routed, std::uint32_t rows, std::uint16_t* out,
    cudaStream_t stream);

// lanes_tiles_moe_route_logits_kernel<u16,2,64>: logits[r][e] = fma chain over f = 0..2559 ascending from 0.
void route_logits(const std::uint16_t* hidden, const float* gate_inp, std::uint32_t rows, float* logits, cudaStream_t stream);
// rows_moe_route_select_kernel<wave>: softmax, top-k (ties: lower index), optional renormalisation.
void route_select(const float* logits, std::uint32_t rows, std::uint32_t top_k, bool normalize, const amdmath::Tables& tables,
    std::uint32_t* ids, float* weights, cudaStream_t stream);

// prefetch8_wave_q8_lm_head_kernel<u16,8> for one row: logits[v] over output.weight (Q8_0 [248320][2560]).
void lm_head(const std::uint16_t* hidden, const void* output_q8, std::uint32_t vocab, float* logits, cudaStream_t stream);
// The same per-row values for 1..8 rows reading output.weight once (hidden row r at hidden + r * hidden_stride, logits row
// r at logits + r * logits_stride).
void lm_head_rows(const std::uint16_t* hidden, std::uint64_t hidden_stride, std::uint32_t rows, const void* output_q8,
    std::uint32_t vocab, float* logits, std::uint64_t logits_stride, cudaStream_t stream);
// Device first_max_token per row: out[r] = token, or 0xffffffff when a logit is non-finite.
void first_max_rows(const float* logits, std::uint64_t stride, std::uint32_t rows, std::uint32_t count, std::uint32_t* out,
    cudaStream_t stream);
// Host first maximum (strict >, lowest index wins); throws std::invalid_argument on a non-finite logit or empty input.
[[nodiscard]] std::uint32_t first_max_token(const float* logits, std::size_t count);

}  // namespace strix::cuda::qwen_hc
