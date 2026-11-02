#pragma once
// minimax-m3 MSA decode, fused and batched (gfx1151, wave32). Same declared arithmetic as strix/minimax_msa.hpp and
// the K1-K4 kernels of strix/hip_minimax_msa.hpp, so every output equals theirs bit for bit
// (src/minimax/attn/msa_fused_tests.hip); only the launch structure and the memory schedule change:
//
//   index launch  = K1 (K/V/index-key rows to the caches, index q norm + RoPE) + K2 (block scores)
//                   persistent grid over (sequence, key block) items, coalesced key tiles staged in LDS, the next
//                   tile prefetched into registers while the current one is scored.
//   attend launch = K3 (selection, recomputed by every chunk block of a (row, KV head)) + q norm + RoPE + K4 chunk
//                   (K tile in LDS, V tile loaded up front) + the chunk combine done by the last chunk block (ticket).
//
// Two launches per layer instead of five, and N independent sequences (users) in the same two launches. All
// per-step values (positions, RoPE rows, cache tables) are read from device memory and the grids depend only on the
// view's shape, so a captured graph can be replayed step after step.
#include <cstdint>

#include <hip/hip_runtime.h>

#include "strix/minimax_msa.hpp"

namespace strix::minimax::msa::fused {

inline constexpr unsigned kMaxRows = 16;      // rows per call (seqs * rows_per_seq)
inline constexpr unsigned kMaxIdxHeads = 4;
inline constexpr unsigned kMaxBlocks = 8192;  // 1M positions (selection scratch in LDS)

// Rows are grouped by sequence: sequence s owns rows [s * rows_per_seq, (s + 1) * rows_per_seq), at consecutive
// ascending positions (a decode step of N users: rows_per_seq = 1; a verify window of one user: seqs = 1).
// Every row's visible positions except its own window must already be in its sequence's caches.
struct DecodeView {
    std::uint32_t seqs, rows_per_seq;
    std::uint32_t q_heads, kv_heads, idx_heads;  // per rank: 32 / 2 / 2 (the attend launch needs idx_heads == kv_heads)
    std::uint32_t max_blocks;                    // capacity in 128-position blocks: >= every row's block_count(pos + 1)
    const float* q;                              // R x q_heads x 128, raw projection (not modified)
    const float* k;                              // R x kv_heads x 128 raw
    const float* v;                              // R x kv_heads x 128
    const float* iq;                             // R x idx_heads x 128 raw (not modified)
    const float* ik;                             // R x 128 raw
    const float* q_w;                            // 128 stored Gemma weights each (scale 1 + w)
    const float* k_w;
    const float* iq_w;
    const float* ik_w;
    const RopeRow* rope;                         // R
    const std::uint64_t* pos;                    // R
    std::uint16_t* const* k_cache;               // seqs device pointers, each capacity x kv_heads x 128 BF16
    std::uint16_t* const* v_cache;
    std::uint16_t* const* ik_cache;              // seqs device pointers, each capacity x 128 BF16
    float* scores;                               // R x idx_heads x max_blocks (local block index when sharded)
    std::uint32_t* idx;                          // R x kv_heads x kTopK: written by the attend launch with select,
    std::uint32_t* idx_count;                    // R x kv_heads            read by it without
    float* partial;                              // R x q_heads x kMaxChunks x (2 + 128)
    std::uint32_t* ticket;                       // R x kv_heads x 4, zero before the first launch; left zero after each
    float* o;                                    // R x q_heads x 128
    // Context-parallel indexer (strix/hip_minimax_msa.hpp ScoreView): score only global blocks
    // b = j * shard_count + shard_rank, written at local index j. The selection then comes from
    // enqueue_candidates/enqueue_merge and the attend launch runs with select = false.
    std::uint32_t shard_rank = 0, shard_count = 1;
    // Attend blocks run heads_per_block query heads each (16, 8 or 4; 0 = the measured default policy).
    std::uint32_t heads_per_block = 0;
    // Attend kernel: 1 = thread-per-key scores; 2 = lane-owns-dims scores with the transposed butterfly, unrolled
    // softmax, parallel combine; 3 = version 2 with rolled outer loops (small code); 4 = version 3 with several
    // heads per wave (heads_per_wave). 0 = default (3). Same bits.
    std::uint32_t attend_version = 0;
    // Version 4: every block runs all 16 heads of (row, KV head, chunk), heads_per_wave (1, 2, 4; 0 = 2) per wave
    // sharing each K/V LDS read. Same bits.
    std::uint32_t heads_per_wave = 0;
    // Versions 3/4: full 16-key tiles take a predicate-free PV path (16 V-row LDS reads issued together); 0 = every
    // tile on the predicated path (diagnostic / disable flag). Same bits.
    std::uint32_t pv_full_tiles = 1;
    // Versions 2/3 (sparse): launch with the actual dynamic LDS (the 32 KB tile when the selection scratch fits it)
    // instead of the 43008-byte floor, four 16-wave blocks per WGP instead of three. Per call; the engine sets it only on
    // the measured winning grids (route::attend_n2_sparse, STRIX_MM_ATTEND_DIET). Same bits.
    bool lds_diet = false;
    // Dense: the caller guarantees every row's pos < kTopK * kBlock (2048): every block is attended, no scores are
    // needed, and enqueue_decode is ONE launch (the attend blocks write the new K/V/index-key rows themselves).
    // scores / idx / idx_count are not written in this mode.
    bool dense = false;
    // Profiling (nullptr = off): wall_clock64 stamps (100 MHz) per attend block, [chunks x grid.x][16] (8 wall, 8 shader-cycle stamps), and per index
    // block [grid][4].
    std::uint64_t* phase = nullptr;
    std::uint64_t* phase_index = nullptr;
};

// Single sequence, consecutive positions. Scratch has scratch_windows independent 16-row regions.
// window carries chunk input/output bases and scratch bases; dense is derived per window.
// Only v3/default and v4/hpw2 with all 16 heads per block are supported; other shapes fail closed.
struct PrefillView {
    DecodeView window;
    std::uint32_t rows;
    std::uint64_t pos0;
    std::uint32_t scratch_windows;
    std::uint32_t first_window = 0;  // internal launch range, must be zero on entry
    bool index_tile = false;
    std::uint32_t index_rows = 32;  // 16/32/64; 64 reuses 32 KiB LDS across four 16-query subtiles
    std::uint32_t msa_rework = 0;  // 0 = original; 1 = shared selection/HPW2/separate combine; 2 = 16-key K/V tiles; 3 = direct sparse mapping + visible tiles; 4 = R16 shared K/V tiles; 5 = mode 3 with packed probabilities; 6 = mode 5 with the tile corrections in the packed exp lanes
};
using PrefillStageMarker = void (*)(void* user, const char* category, bool before);
void enqueue_prefill(const PrefillView& v, hipStream_t stream,
                     PrefillStageMarker marker = nullptr, void* user = nullptr);
// Score the prepared cache for [first_window, ceil(rows/16)); scratch is not rebased.
// Exposed for index-only timing and byte comparisons; single sequence, two index heads.
void enqueue_prefill_index(const PrefillView& v, hipStream_t stream);
// Diagnostic producer timing/byte checks: one scratch batch, caches/IDs already prepared.
// Mode 0 retains the rectangular grid; all modes omit selection, tickets and combine.
void enqueue_prefill_partials(const PrefillView& v, hipStream_t stream);
// Diagnostic shared-selection timing: one prepared-score batch, same dense split.
// R16 row membership stays inside the producer and is included in its timing.
void enqueue_prefill_selection(const PrefillView& v, hipStream_t stream);
// Scratch bytes for one full window (scores, idx, counts, partials, tickets).
std::size_t prefill_window_bytes(std::uint32_t max_blocks);

// K1 + K2. Grid = min(items, index_grid_cap) persistent blocks; items = seqs x local blocks of capacity.
void enqueue_index(const DecodeView& v, hipStream_t stream);
// K3 (select = true: from v.scores; false: v.idx / v.idx_count are inputs) + K4 + combine.
void enqueue_attend(const DecodeView& v, bool select, hipStream_t stream);
// index + attend (select), or the single dense launch when v.dense.
void enqueue_decode(const DecodeView& v, hipStream_t stream);
unsigned attend_heads(const DecodeView& v);    // heads per attend block the launch will use
unsigned attend_version(const DecodeView& v);  // attend kernel version the launch will use

// Persistent index grid cap (default 160 blocks = 4 per WGP); STRIX_MMATTN_INDEX_GRID overrides it (bench only).
unsigned index_grid_cap();
void set_index_grid_cap(unsigned cap);  // tests and bench only (changes the grid: recapture graphs)
void set_attend_lds_min(std::size_t bytes);  // bench only: minimum dynamic LDS of the v2/v3 attend (occupancy A/B);
                                             // a view with lds_diet whose selection fits the tile ignores it

}  // namespace strix::minimax::msa::fused
