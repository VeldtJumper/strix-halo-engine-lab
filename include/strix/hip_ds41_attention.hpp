#pragma once
// gfx1151 kernels of the deepseek41 decode attention; arithmetic = the host references in ds41_attention.hpp.
// All pointers are device memory; nothing allocates or synchronizes. Rows of one call are consecutive positions.
#include "strix/ds41_attention.hpp"

#include <hip/hip_runtime.h>

#include <span>

namespace strix::ds41::attn::hip {

// RoPE tables are per row: the rows' own positions (and, for compress, the group's first position).
struct WindowView {
    const float* kv;              // R x 512
    const float* norm;            // 512
    const RopeRow* rope;          // R, at pos[r]
    const std::uint64_t* pos;     // R
    std::uint32_t rows, capacity;
    std::uint8_t* ring;           // capacity x 528
};
void enqueue_window_write(const WindowView& v, hipStream_t stream);

struct CompressView {
    std::uint32_t ratio, rows;
    const float* kv;              // R x 512
    const float* score;           // R x 512 (ratio 2)
    const float* norm;            // 512
    const std::uint16_t* wk;      // 128 x 512 F16
    const float* knorm;           // 128
    const RopeRow* rope;          // R, at the group's first position (pos / ratio * ratio)
    const std::uint64_t* pos;     // R
    const float* state_in;        // ratio 2: 1024 floats (kv, score) of the pending group's first token, valid flag below
    const std::uint32_t* state_valid_in;
    float* state_out;             // ratio 2: the pending group after the last row
    std::uint32_t* state_valid_out;
    std::uint8_t* kv_cache;       // entries of 288 B, index pos / ratio
    std::uint8_t* idx_cache;      // entries of 68 B
};
void enqueue_compress(const CompressView& v, hipStream_t stream);
// STRIX_DS41_COMPRESS_AHEAD=1 (default off): the index key weights of each thread are read three 64-weight steps ahead,
// the first ones before the latent barrier; the same chains, the same bytes. Tests: 0 off, 1 on, -1 the environment.
void set_compress_ahead(int mode);
// STRIX_DS41_COMPRESS_FAST (default on, =0 off; =0 with STRIX_DS41_COMPRESS_AHEAD=1 selects the ahead form): the same kernel in fewer instructions (hardware F16 conversion of finite
// weights, 16-byte latent reads, the next weight step requested ahead, branch-free exp range tests); the same chains,
// the same bytes. Tests: 0 off, 1 on, -1 the environment.
void set_compress_fast(int mode);

struct IndexView {
    std::uint32_t rows, ratio, max_count;   // max_count >= every row's (pos + 1) / ratio
    const float* q;               // R x 32 x 128
    const float* w;               // R x 32
    const RopeRow* rope;          // R
    const std::uint64_t* pos;     // R
    const std::uint8_t* idx_cache;
    float* scores;                // R x max_count
    void* qprep = nullptr;        // scratch R x kIdxPrepBytes (required): quantised queries, scales and head weights
    // candidate-user call with STRIX_DS41_IDX_CAND_FILTER: the rows' candidate blocks (SelectView::candidates, row stride
    // ceil(max_count / block_size)); only the keys of a block some row takes are read and scored, and a row's score is
    // stored only where its block is a candidate (select maps every other position to -inf without reading it). Null: all.
    // Callers set it only when the pool is a strict subset (count > cand_blocks x block_size): otherwise every block is one.
    const std::uint8_t* candidates = nullptr;
    std::uint32_t block_size = 0;
};
inline constexpr unsigned kIdxPrepBytes = (kIdxHeads * 32 + kIdxHeads * 4 + kIdxHeads) * 4;
void enqueue_index_scores(const IndexView& v, hipStream_t stream);

struct SelectView {
    std::uint32_t rows, ratio, max_count, topk, cand_blocks, block_size;
    Pool pool;
    const std::uint64_t* pos;     // R (count = (pos + 1) / ratio)
    const float* scores;          // R x max_count
    std::uint8_t* candidates;     // R x ceil(max_count / block_size): written by the source, read by users
    std::uint32_t* keys;          // scratch R x max_count
    std::uint32_t* idx;           // R x topk, ascending
    std::uint32_t* idx_count;     // R
    // candidate user with STRIX_DS41_IDX_CAND_FILTER: the two-level selection (max_count > 4096) keeps only candidate
    // positions in each slice (same result; today's one-block selection for a row where that cannot be shown); the caller
    // sets it with IndexView::candidates. Needs a block_size >= 8 dividing 4096.
    bool compact = false;
};
void enqueue_select(const SelectView& v, hipStream_t stream);
// Every row's count (pos + 1) / ratio <= topk: the selection is every position in order whatever the scores, so with
// STRIX_DS41_IDX_DIRECT the caller skips the indexer projections and scores and writes it directly: idx[t] = t,
// idx_count = count, and a candidate source publishes its pool as enqueue_select does (every block of the row; only
// v.pos, v.ratio, v.rows, v.max_count, v.block_size, v.pool, v.candidates, v.idx, v.idx_count are used).
void enqueue_select_direct(const SelectView& v, hipStream_t stream);
void enqueue_select_direct_segments(std::span<const SelectView> views, hipStream_t stream);
// STRIX_DS41_IDX_CAND_FILTER (default on, =0 off): candidate-user index scores read only candidate blocks (IndexView::candidates;
// the caller sets it when this is on). STRIX_DS41_IDX_DIRECT (default on, =0 off): see enqueue_select_direct.
// Tests / A-B: 0 off, 1 on, -1 the environment.
void set_index_candidate_filter(int mode);
[[nodiscard]] bool index_candidate_filter();
void set_index_direct(int mode);
[[nodiscard]] bool index_direct();

// heads: the query heads of this call (all 64, or a tensor-parallel rank's 32): q, sink, o and partial hold them
// contiguously (head index local to the call); the shared KV head is the same for every head.
struct AttentionView {
    std::uint32_t rows, capacity, topk;
    const float* q;               // R x heads x 512
    const float* sink;            // heads
    const RopeRow* rope;          // R
    const std::uint64_t* pos;     // R
    const std::uint8_t* ring;
    const std::uint8_t* kv_cache; // may be null when idx_count is null (window-only layer)
    const std::uint32_t* idx;     // R x topk
    const std::uint32_t* idx_count; // R, or null
    float* partial;               // scratch R x heads x kMaxChunks x (2 + 512)
    float* o;                     // R x heads x 512
    std::uint32_t* tickets = nullptr; // R x kAttnGroupCount counters, zero before the first call and left zero by every
                                  // call: the last chunk block of each (row, head group) combines; null: combine kernel
    std::uint32_t heads = kHeads; // multiple of 8
    // optional A8 of o (int8 per 32, enqueue_quantize_a8's bits) written by the combine: codes R x heads x 512,
    // deltas / sums R x heads x 16
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    std::int32_t* sums = nullptr;
    // STRIX_DS41_ATTN_WINDOW_FOLD: the call's window insert done by the attention kernel instead of enqueue_window_write:
    // kv = the rows' K/V (R x 512, as WindowView::kv), kv_norm its norm weight (512), ring_out = ring (written: each row's
    // own slot, the same bytes as window_write); rope must be the table window_write would use (it is, per layer).
    const float* kv = nullptr;
    const float* kv_norm = nullptr;
    std::uint8_t* ring_out = nullptr;
    // STRIX_DS41_ATTN_LIVE_CHUNKS: the caller's upper bound on any row's visible keys (window + selected), or 0 (unknown):
    // the grid and the heads-per-wave choice then count only the chunks it can hold. A row above it traps.
    std::uint32_t max_visible = 0;
};
inline constexpr unsigned kAttnGroupCount = 8;   // at most 8 head groups per row (8 heads per block)
// variant 0: STRIX_DS41_BF16_ATTN_VARIANT (default 1, the first kernel); 2: QK split over the waves, 16-byte key stores
void enqueue_attention_bf16_prefill(const AttentionView&, hipStream_t stream, int variant = 0);
// FAST lever ds41-attn-f16wmma-v1 (prefill only, NOT bit-exact): QK^T and PV on f16 WMMA with FP32 accumulation, FP32
// softmax statistics with the ordered rescale, today's FP32 chunk combine. Same view contract as enqueue_attention (no
// window fold; heads a multiple of 16). Selected by the prefill with STRIX_DS41_FAST_ATTN=1 (default off).
void enqueue_attention_f16wmma_prefill(const AttentionView& v, hipStream_t stream);
// Benchmarks only: chunks per block (1, 2 = default, 5, 10), an ablation mask (0 = none; see the kernel; non-zero = wrong
// results) and heads per block (16 = default, 32 where the call's heads allow it: measured slower, kept for benchmarks).
void set_fast_attn_tuning(unsigned chunks_per_block, unsigned ablate, unsigned heads_per_block = 16U);
void enqueue_attention(const AttentionView& v, hipStream_t stream);
// STRIX_DS41_ATTN_LDS_SWIZZLE (default on, =0 off): permute the f32 tile's 16-byte LDS vectors only (the tile choice is unchanged).
// Test override: 0 off, 1 on, -1 environment. Arithmetic and reduction order are unchanged.
void set_attention_lds_swizzle(int mode);
// STRIX_DS41_ATTN_TILE_F16 (default on, =0 off): the attention tile in LDS as f16 (every dequantised key value is exact in
// f16 unless above 65504 or below its 2^-24 subnormal step; such a tile stays f32): half the LDS bytes, the same floats in every product.
// Calls at 1 or 2 heads per wave only (4 keeps today's tile). Not combined with the swizzle. Test override: 0 off, 1 on,
// -1 environment.
void set_attention_tile_f16(int mode);
// STRIX_DS41_ATTN_LIVE_CHUNKS (default on, =0 off): callers pass AttentionView::max_visible (same bits: every heads-per-wave
// form computes each head identically). Test override: 0 off, 1 on, -1 environment.
void set_attention_live_chunks(int mode);
[[nodiscard]] bool attention_live_chunks();
// STRIX_DS41_ATTN_WINDOW_FOLD (default on, =0 off): callers pass AttentionView::kv / kv_norm / ring_out and skip the window
// write. Test override: 0 off, 1 on, -1 environment.
void set_attention_window_fold(int mode);
[[nodiscard]] bool attention_window_fold();
// STRIX_DS41_ATTN_LEAN (default 3, =0 off): the chunk kernels in fewer instructions, the same values in the same order
// (same bits): 1 = the softmax and PV steps, 2 = also the key dequantisation into the tile, 3 = also the QK tile with
// lane-permuted keys (no butterfly selects; at 4 heads per wave one fixed-register asm block, every FMA pair dual-issued),
// one address per PV key, the rescale skipped when the running max holds (factor exactly 1), and the probabilities
// evaluated once per (head, key). Test override: 0..3, or -1 environment.
void set_attention_lean(int mode);
// STRIX_DS41_ATTN_WAVE_CAP (default on, =0 off): lean 3 calls at 4 heads per wave with compressed keys or 64 heads use a
// kernel compiled for 6 waves per SIMD (240 VGPRs on gfx1151 with this toolchain; 243 and 5 waves without); same bits.
// Test override: 0 / 1, or -1 environment.
void set_attention_wave_cap(int mode);
[[nodiscard]] int attention_lean();

// Segment launches (several requests of a batch step per launch, STRIX_DS41_BATCH_SEGMENTS): one view per request
// segment, each exactly the call above (its checks, its grid, every block's work), at most kMaxSegments per launch.
// The segments must not share state (ring, caches, carry, scratch rows). Index: every qprep, then every scores grid;
// select: the one-block rows, then the slices, then the merges; attention: grouped by the call's heads per wave.
inline constexpr unsigned kMaxSegments = 8;
void enqueue_window_write_segments(std::span<const WindowView> views, hipStream_t stream);
void enqueue_compress_segments(std::span<const CompressView> views, hipStream_t stream);
void enqueue_index_scores_segments(std::span<const IndexView> views, hipStream_t stream);
void enqueue_select_segments(std::span<const SelectView> views, hipStream_t stream);
void enqueue_attention_segments(std::span<const AttentionView> views, hipStream_t stream);
// STRIX_DS41_ATTN_SEG_HPW (default on, =0 off): the segments of one launch share one heads-per-wave form chosen from their
// blocks together (the per-call rule over the sum), instead of each segment's own (an N8 step of one row per request
// takes 4 heads per wave instead of 1); every form computes each head identically. Tests: 0 off, 1 on, -1 environment.
void set_attention_segment_hpw(int mode);
// STRIX_DS41_ATTN_MIN_BLOCKS=N (default 40): a call takes the most heads per wave (4, 2, 1) whose grid still has >= N
// blocks; more heads per wave dequantise each key tile once for more heads, fewer give more blocks. Same bits for any N.
// Tests: 0 = the environment.
void set_attention_min_blocks(unsigned blocks);
#if defined(STRIX_DS41_ATTN_TIMING)
// timing build: slot k (0 window .. 6 combine) x 65536 blocks x {wall0, wall1, cycles0, cycles1}
void set_timing_buffer(void* p);
#endif

}  // namespace strix::ds41::attn::hip
