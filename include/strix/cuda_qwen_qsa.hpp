// CUDA port of the Qwen QSA (full-attention) layer core of the HIP tiled prefill, bit-identical to the single A8 profile
// (notes/qwen-prefill-map-a8.md H3.2; HIP src/hip_indexed_attention_launcher.hip, enqueue_qsa_known_rows part 1 indexer
// + part 2). The dense A8 q/k/v and output projections are not here (dense family): this code takes their outputs
// (q_gate, k, v as BF16-valued FP32 rows) and produces the FP32 (BF16-valued) attention_output rows that feed the A8 o
// projection, plus the session state (KV and raw index keys) of the layer.
//
// Host launch API over device pointers. Every function enqueues on `stream` in stream order; no host synchronisation.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "qwen_amd_math.cuh"

namespace strix::cuda::qwen_qsa {

// Model geometry (include/strix/model_contract.hpp defaults; the engine uses these, not GGUF metadata).
inline constexpr std::uint32_t kHidden = 2560;
inline constexpr std::uint32_t kHeads = 24, kKvHeads = 2, kHeadDim = 256;
inline constexpr std::uint32_t kQGateWidth = kHeads * kHeadDim * 2U;   // 12288: per head [256 q | 256 gate]
inline constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;         // 512
inline constexpr std::uint32_t kAttnWidth = kHeads * kHeadDim;         // 6144
inline constexpr std::uint32_t kIndexHeads = 4, kIndexDim = 128;
inline constexpr std::uint32_t kIndexQWidth = kIndexHeads * kIndexDim; // 512
inline constexpr std::uint32_t kRotaryDim = 64;                        // 256 * partial_rotary_factor 0.25, all 4 sites
inline constexpr std::uint32_t kRotaryPairs = kRotaryDim / 2U;         // 32
inline constexpr std::uint32_t kRatio = 4;                             // indexer_compress_ratio
inline constexpr std::uint32_t kTopK = 512;                            // indexer_budget 2048 / ratio
inline constexpr std::uint32_t kMaxSelected = 2048U + kRatio - 1U;     // 2051 = maximum_selected_tokens (score row stride)
inline constexpr float kEpsilon = 1e-6F;                               // rms_norm_epsilon
// theta = 1e7 (rope_theta) only enters through the cos/sin table.

// cs[position * 32 + pair] = {cos, sin} of angle = (float)position * powf(1e7f, -(2.0f*pair/64.0f)), computed on the node
// with the HIP expression (gfx1151 OCML powf/cosf/sinf). Must cover every position < first_position + rows.
using RopeTable = float2;

struct LayerWeights {
    const std::uint16_t* index_q_proj;  // BF16 [512][2560] (blk.L.indexer.q_proj.weight, type 30)
    const std::uint16_t* index_k_proj;  // BF16 [128][2560] (blk.L.indexer.k_proj.weight)
    const float* q_norm;                // F32 [256] (attn_q_norm)
    const float* k_norm;                // F32 [256] (attn_k_norm)
    const float* index_q_norm;          // F32 [128] (indexer.q_norm)
    const float* index_k_norm;          // F32 [128] (indexer.k_norm)
};

// Session state of ONE QSA layer: kv = BF16 [context][1024] ([K h0 256][K h1 256][V h0 256][V h1 256], 2048 B/token),
// raw index keys = BF16 [context][128] (pre-norm, pre-RoPE, 256 B/token).
struct LayerState {
    std::uint16_t* kv;
    std::uint16_t* raw_index;
    // Optional decode cache of the pooled index keys of this layer ([max_blocks][128] FP32) and the host count of its
    // valid blocks: layer_core then pools only the blocks it does not hold (a pooled block depends only on its 4 raw
    // rows, so the values are the same); blocks touching rows >= first_position are recomputed.
    float* pooled = nullptr;
    std::uint32_t* pooled_valid = nullptr;
};

// Row buffers of `rows` consecutive positions first_position .. first_position + rows - 1 (row-major, dense).
struct RowBuffers {
    const std::uint16_t* hidden;   // BF16 [rows][2560]: the attention input (same rows the dense q/k/v projection reads)
    float* q_gate;                 // [rows][12288] BF16-valued; q halves normalised + rotated in place
    float* k;                      // [rows][512]   BF16-valued; normalised + rotated in place
    const float* v;                // [rows][512]   BF16-valued
    float* index_query;            // [rows][512]   written by the indexer projection, then normalised + rotated
    float* index_key;              // [rows][128]   written by the indexer projection (raw key)
    float* attention_output;       // [rows][6144]  BF16-valued result (input of the A8 o projection)
};

// Workspace for chunks of up to chunk_rows rows over a context of up to max_positions positions.
struct Workspace {
    std::uint32_t chunk_rows = 0, max_blocks = 0;
    float* pooled_keys = nullptr;      // [max_blocks][128]
    float* block_scores = nullptr;     // [chunk_rows][max_blocks]
    std::uint32_t* selected = nullptr; // [chunk_rows][2051]
    float* scores = nullptr;           // [chunk_rows][24][2051]
};
[[nodiscard]] std::size_t workspace_bytes(std::uint32_t chunk_rows, std::uint64_t max_positions);
Workspace allocate_workspace(std::uint32_t chunk_rows, std::uint64_t max_positions);  // cudaMalloc; throws on failure
void free_workspace(Workspace& w);

// ---- individual steps (HIP kernel named on each) ----
// linear_to_float_rows_kernel<bf16,16>: index_query[r][0..512) and index_key[r][0..128).
void indexer_projection(const std::uint16_t* hidden, std::uint32_t rows, const LayerWeights& w, float* index_query,
    float* index_key, cudaStream_t stream);
// norm_rope_rows_kernel: per (row, head) RMS norm + partial NeoX RoPE (first 64 features) of
// values[row * row_stride + head * head_stride + f], f < head_dim (head_dim 128 or 256).
void norm_rope(float* values, std::uint32_t rows, std::uint64_t row_stride, std::uint32_t heads, std::uint32_t head_dim,
    std::uint32_t head_stride, const float* norm, std::uint64_t first_position, const RopeTable* cs,
    const amdmath::Tables& tables, cudaStream_t stream);
// store_rows_kernel: raw index key, K and V of every row into the layer state.
void store_rows(const float* index_key, const float* k, const float* v, std::uint32_t rows, std::uint64_t first_position,
    const LayerState& state, cudaStream_t stream);
// block_pool_kernel: pooled/normalised/rotated keys of blocks [first_block, first_block + count).
void block_pool(const std::uint16_t* raw_index, std::uint32_t first_block, std::uint32_t count, const float* index_k_norm,
    const RopeTable* cs, const amdmath::Tables& tables, float* pooled_keys, cudaStream_t stream);
// block_score_pooled_rows_kernel<Rows,4>: block_scores[r][b] for b < complete(r) of every row with complete(r) > 512.
void block_scores(const float* index_query, std::uint32_t rows, std::uint64_t first_position, const float* pooled_keys,
    float* block_scores, std::uint32_t max_blocks, cudaStream_t stream);
// select_rows_kernel: selected[r][0..count(r)) (also sets the selected block scores to -inf, as HIP).
void select_rows(float* block_scores, std::uint32_t max_blocks, std::uint32_t rows, std::uint64_t first_position,
    std::uint32_t* selected, cudaStream_t stream);
// attention_score_rows_group256_kernel<64,12,16> + attention_softmax_rows_kernel<64> +
// attention_output_rows_group_prefetch_kernel<64,12>. scores: [rows][24][2051] workspace.
void attention(const float* q_gate, std::uint32_t rows, std::uint64_t first_position, const std::uint32_t* selected,
    const std::uint16_t* kv, float* scores, float* attention_output, const amdmath::Tables& tables, cudaStream_t stream);

// The three kernels of attention() separately (same arguments).
void attention_scores(const float* q_gate, std::uint32_t rows, std::uint64_t first_position, const std::uint32_t* selected,
    const std::uint16_t* kv, float* scores, cudaStream_t stream);
void attention_softmax(std::uint32_t rows, std::uint64_t first_position, float* scores, const amdmath::Tables& tables, cudaStream_t stream);
void attention_values(const float* q_gate, std::uint32_t rows, std::uint64_t first_position, const std::uint32_t* selected,
    const std::uint16_t* kv, const float* scores, float* attention_output, const amdmath::Tables& tables, cudaStream_t stream);

// Selected-token count of the row at `position` (min(complete, 512) * 4 + tail).
[[nodiscard]] __host__ __device__ constexpr std::uint32_t selected_count(std::uint64_t position) {
    const auto visible = static_cast<std::uint32_t>(position + 1U);
    const auto complete = visible / kRatio;
    return (complete < kTopK ? complete : kTopK) * kRatio + visible % kRatio;
}

// ---- several sessions' rows in one window (decode cohorts): segment i = rows [first_row[i], first_row[i+1]) of the
// row buffers at positions position[i].. of its own state. Device-side map of a multi-segment launch.
inline constexpr std::uint32_t kMaxSegments = 8;
struct SegmentMap {
    std::uint32_t n = 0;
    std::uint32_t first_row[kMaxSegments + 1] = {};
    std::uint64_t position[kMaxSegments] = {};
    std::uint16_t* kv[kMaxSegments] = {};
    std::uint16_t* raw_index[kMaxSegments] = {};
    float* pooled[kMaxSegments] = {};
    std::uint32_t pool_first[kMaxSegments] = {}, pool_begin[kMaxSegments + 1] = {};   // pooled blocks to (re)compute
    std::uint32_t scored[kMaxSegments] = {}, score_begin[kMaxSegments + 1] = {};      // block-score waves
};
struct SegmentState {
    std::uint32_t rows = 0;
    std::uint64_t first_position = 0;
    LayerState state;
};
// layer_core for every segment of a window with one launch per step (each block runs one segment's single-call body):
// the same values as layer_core per segment (rows = the segment's rows, b offset to its first row). Every session needs its
// own pooled cache (LayerState::pooled). Falls back to one layer_core per segment beyond the multi-launch shapes.
void layer_core_multi(const RowBuffers& rows_buf, const SegmentState* segments, std::uint32_t count, const LayerWeights& w,
    const RopeTable* cs, const amdmath::Tables& tables, Workspace& ws, cudaStream_t stream);

// ---- the whole QSA core of one layer for rows [first_position, first_position + rows) ----
// indexer projection, 3 norms, store, pool (blocks needed by the last row), then per chunk of workspace.chunk_rows rows:
// block scores, selection, attention. Positions < first_position must already be in `state`.
void layer_core(const RowBuffers& rows_buf, std::uint32_t rows, std::uint64_t first_position, const LayerWeights& w,
    const LayerState& state, const RopeTable* cs, const amdmath::Tables& tables, Workspace& ws, cudaStream_t stream);

}  // namespace strix::cuda::qwen_qsa
