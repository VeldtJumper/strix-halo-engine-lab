// CUDA port of the Qwen3.8-Flash-Next prompt-prefill PLE step (one-based layer 2, before layer 1's attention mix) and
// of the MTP predictor cache rows over the prompt, bit-identical to the HIP gfx1151 engine (family
// "plemtp"). Host launch API: device pointers and sizes only, no engine types. Every launch is stream ordered.
//
// PLE (HIP: src/ple_lookup.cpp, src/iq4_nl.cpp, src/hip_ple_launcher.hip, src/hip_q8_rows_gemv.hpp):
//   host   ple_hash_rows            3-gram hashing, EOS cut rule, n-gram history       (ple_lookup.cpp:71-256)
//   host   ple_gather_iq4nl_host     IQ4_NL head rows -> BF16 RNE, up to N threads      (ple_lookup.cpp:600-656)
//   device launch_ple_gather_iq4nl   the same from a mapped (pinned host) or device table
//   device launch_ple_project        Q8_0 key (10240) / value (2560) projections, FP32 lane chain with the fixed
//                                    256-leaf tree (project_wave8_q8_bf16_to_fp32_rows_kernel, :496-538, and its
//                                    lane-kernel equivalent project_wide_rows_lane_kernel<Site,16,8,2>)
//   device launch_ple_tail           ple_rows_gate_norm_kernel, ple_rows_conv_residual_kernel,
//                                    ple_rows_conv_state_kernel (hip_ple_launcher.hip:882-1054)
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "qwen_amd_math.cuh"

namespace strix::cuda::qwen_plemtp {

// ---------------------------------------------------------------------------------------------------------------
// PLE geometry (pinned by the HIP launcher, hip_ple_launcher.hip validate_config)
constexpr std::uint32_t kPleHeads = 16, kPleHeadDim = 160, kPleWidth = 2560;   // gathered row = 16 x 160 BF16
constexpr std::uint32_t kHidden = 2560, kHc = 4, kExpanded = kHidden * kHc;     // 10240 channels
constexpr std::uint32_t kConvKernel = 4, kNgram = 3, kConvHistory = (kConvKernel - 1U) * kNgram;  // 9
constexpr std::uint32_t kPleEos = 248044;
constexpr std::uint64_t kPleTableRows = 320001536ULL;   // per_layer_token_embd rows (IQ4_NL, 160 elements, 90 B)
constexpr std::uint32_t kIq4RowBytes = 90U;

struct PleHashContract {
    std::uint64_t multipliers[kNgram]{};
    std::uint64_t head_offsets[kPleHeads]{};
    std::uint64_t head_vocab_sizes[kPleHeads]{};
    std::uint64_t minimum_table_rows = 0;
};
// The production contract (vocab 248320, seed 1234, base 20,000,000, 8 heads per n-gram, n = 3).
[[nodiscard]] PleHashContract ple_hash_contract();
// Hashes `rows` consecutive prompt tokens starting at absolute position `position0`. history[2] (oldest first) is
// the session's n-gram word: ignored at position 0 (then {EOS, EOS}), updated to the last row's next context.
// row_ids: [rows][16].
void ple_hash_rows(const PleHashContract& contract, const std::uint32_t* tokens, std::uint32_t rows,
    std::uint64_t position0, std::uint32_t history[2], std::uint64_t* row_ids);
// Gathers and decodes [rows][16] head rows of the IQ4_NL table (host memory, e.g. pinned) into BF16 [rows][2560]
// (row stride out_stride elements). Throws on a non-finite block scale or an out-of-range row (as the engine).
void ple_gather_iq4nl_host(const std::uint8_t* table, std::uint64_t table_rows, const std::uint64_t* row_ids,
    std::uint32_t rows, std::uint16_t* out, std::uint64_t out_stride, unsigned threads);
// Device form: `table` is a device pointer (device memory or the device view of mapped pinned host memory).
// status (device word, may be null) gets bit 0 for a non-finite scale, bit 1 for an out-of-range row id.
void launch_ple_gather_iq4nl(const std::uint8_t* table, std::uint64_t table_rows, const std::uint64_t* row_ids,
    std::uint32_t rows, std::uint16_t* out, std::uint64_t out_stride, std::uint32_t* status, cudaStream_t stream);

// Q8_0 [outputs][width/32 blocks of 34 B] x BF16 rows [rows][in_stride] -> FP32 out [rows][out_stride].
// width must be a multiple of 256.
void launch_ple_project(const std::uint16_t* input, std::uint32_t in_stride, std::uint32_t rows, const void* weights,
    std::uint32_t width, std::uint32_t outputs, float* out, std::uint32_t out_stride, cudaStream_t stream);

struct PleTailArgs {
    float* keys; std::uint32_t key_stride;                 // FP32 [rows][10240]; overwritten by the normalized values
    const float* values; std::uint32_t value_stride;       // FP32 [rows][2560]
    const std::uint16_t* input; std::uint32_t input_stride;    // BF16 wide residual [rows][10240] (the query)
    std::uint16_t* output; std::uint32_t output_stride;        // BF16 [rows][10240]; may equal input
    std::uint16_t* conv_state;                             // BF16 [10240][9], read then rewritten
    std::uint16_t* const* conv_state_slot;                 // non-null: conv_state is read from this device word instead
    const float* norm_key; const float* norm_query; const float* norm_conv;   // F32 [10240]
    const float* conv;                                     // F32 [10240][4]
    float* gates;                                          // FP32 [rows][4] scratch
    std::uint32_t rows;
    float epsilon;                                         // rms_norm_epsilon (1e-6)
};
void launch_ple_tail(const PleTailArgs& args, const amdmath::Tables& tables, cudaStream_t stream);
// PLE conv history after the first `keep` rows of a tail call: `before` = the history before it, keys = that call's
// normalized keys (the tail rewrites PleTailArgs::keys in place).
void launch_ple_conv_rollback(const std::uint16_t* before, const float* keys, std::uint32_t key_stride, std::uint32_t keep,
    std::uint16_t* conv_state, cudaStream_t stream);

// ---------------------------------------------------------------------------------------------------------------
// MTP predictor cache rows over the prompt (HIP: HipMtpRuntime::prime_hidden_rows, src/hip_mtp_runtime.hip:584-613, with
// the production draft chain: cache_row_front = launch_hip_mtp_input_fusion_draft + draft attention HC mix, then
// launch_hip_mtp_cache_rows with fast_cache = false). For every pair (h_i, t_{i+1}) at source position p_i = position0 + i:
//   1 embedding_kernel<bf16, q8_0>            target token_embd row of t_{i+1} -> BF16 [2560]
//   2 mtp_input_norms<1024>                   enorm / hnorm RMS norms (gain w, effective_scale), optional BF16 RNE
//   3 mtp_input_projections_wave_rows<Q4,2>    Q4_0 copy of eh_proj (mtp_q8_to_q4), bf16(rnd(e) + rnd(s_k)) -> [4][2560]
//   4 prepare_hyper_norm_kernel<bf16>         hc_attn_norm per stream (rsqrtf)
//   5 rows_splitk_hyper_projection_kernel<2>   hc_attn_down split-K (16 tiles of 640), low = s * sigmoid(s), s = total / 4
//   6 lane_chain_hyper_output_kernel<bf16>     hc_attn_up lane chain, sigmoid, stream mix / 4 -> hidden0 BF16 [2560]
//   7 linear_to_float_kernel<bf16 | q8_0>      index key 128 (BF16 weights), K 512, V 512 (Q8), each round_bf16
//   8 norm_rope_kernel<f32> on K (2 heads x 256, rotary 64 from cos/sin[p][32]), store_raw_index_key_kernel, store_kv_kernel
// Writes kv[p_i * 1024 + 0..511] = K, [512..1023] = V (BF16) and raw_index[p_i * 128 + 0..127] (BF16). The held wide row
// h_{N-1} (the "MTP draft state", BF16 4 x 2560) is the caller's last tap row and needs no kernel.
// CUDA batches all rows of a call; each row keeps the HIP per-row arithmetic.
constexpr std::uint32_t kMtpKvWidth = 512, kMtpHeadDim = 256, kMtpKvHeads = 2, kMtpIndexDim = 128, kMtpRank = 320;
constexpr std::uint32_t kMtpRotary = 64;
constexpr std::uint32_t kMtpEhRowQ8 = 2U * kHidden / 32U * 34U, kMtpEhRowQ4 = 2U * kHidden / 32U * 18U;

// Q8_0 -> Q4_0 per 32-block (mtp_q8_to_q4, src/hip_mtp_fast_draft.hip:145-161). blocks = elements / 32.
void launch_mtp_q8_to_q4(const void* q8, std::uint64_t blocks, void* q4, cudaStream_t stream);

struct MtpWeights {
    const void* token_embd;          // target token_embd.weight, Q8_0 [vocab][2560]
    std::uint32_t vocab;
    const float* enorm;              // blk.48.nextn.enorm.weight F32 [2560]
    const float* hnorm;              // blk.48.nextn.hnorm.weight F32 [10240]
    const void* eh_q4;               // Q4_0 copy of blk.48.nextn.eh_proj.weight [2560][5120]
    const float* hc_norm;            // blk.48.hc_attn_norm.weight F32 [10240]
    const void* hc_down;             // blk.48.hc_attn_down.weight Q8_0 [320][10240]
    const void* hc_up;               // blk.48.hc_attn_up.weight Q8_0 [10240][320]
    const std::uint16_t* index_k;    // blk.48.indexer.k_proj.weight BF16 [128][2560]
    const void* k;                   // blk.48.attn_k.weight Q8_0 [512][2560]
    const void* v;                   // blk.48.attn_v.weight Q8_0 [512][2560]
    const float* k_norm;             // blk.48.attn_k_norm.weight F32 [256]
    const float* rope_cs;            // [position][32][cos, sin] FP32, the HIP angle expression's cosf/sinf
    std::uint64_t rope_positions;
};
struct MtpRounding {
    bool normalized_bf16 = true;     // --mtp-normalized-rounding bf16-rne (production)
    bool projection_bf16 = true;     // --mtp-projection-rounding bf16-rne (production)
    bool gemma_offset = false;       // effective-norm conversion: gain = w (production); gemma: 1 + w
    float epsilon = 1e-6F;
};
[[nodiscard]] std::uint64_t mtp_workspace_bytes(std::uint32_t rows);
// hidden: BF16 wide rows h_i [rows][10240]; next_tokens: device [rows]; kv / raw_index: cache bases (row p at p * 1024 /
// p * 128 elements). status (device, may be null): bit 0 token out of range, bit 1 position beyond the RoPE table,
// bit 3 a non-finite or BF16-overflowing cache value (validate_mtp_cache_casts; the row is then not stored).
void launch_mtp_cache_rows(const MtpWeights& w, const MtpRounding& rounding, const std::uint16_t* hidden,
    const std::uint32_t* next_tokens, std::uint32_t rows, std::uint64_t position0, std::uint16_t* kv, std::uint16_t* raw_index,
    void* workspace, std::uint32_t* status, const amdmath::Tables& tables, cudaStream_t stream);

// Intermediate views of the workspace (tests): offsets in bytes for `rows` rows.
struct MtpWorkspaceViews {
    std::uint16_t* embedding; float* normalized; std::uint16_t* fused; float* hc_normalized; float* low; std::uint16_t* hidden0;
    float* index_key; float* key; float* value;
};
[[nodiscard]] MtpWorkspaceViews mtp_workspace_views(void* workspace, std::uint32_t rows);

}  // namespace strix::cuda::qwen_plemtp
