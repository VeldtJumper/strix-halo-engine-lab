#pragma once
// minimax-m3 prefill MoE (gfx1151, one TP2 rank): every routed expert and the shared expert of a T-row chunk in a
// fixed number of grouped launches, instead of 3 strix_gemm (x 3 launches) + gather/scatter per expert.
//
// Identity (bitwise): per expert, the bring-up chain of tests/hip_minimax_g4.hip for prefill chunks, i.e. strix_gemm
// (strix-trellis 0.3, WMMA core at every row count) on the expert's gathered rows (ascending token order), swiglu,
// strix_gemm down, and per token acc = 0; acc = fmaf(w_e, y_e, acc) over its experts in ascending id; acc + shared.
// Each grouped CTA runs the library's wmma_core for one (expert, 4 tile columns, row tile, split) with the library's
// split and tile policy for that expert's row count (split_count / wmma_tile_rows, ported verbatim), and the
// partials are reduced in ascending split order before the output Hadamard, as output_transform does.
//
// Launches: gather+transform, gate/up (grouped WMMA), gate/up finish + SwiGLU + down transform, down (grouped WMMA,
// one launch per row-tile class), down finish + combine. Routes come from the host (as in G4; one D2H per chunk).
#include "minimax/moe/mmx_moe.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <hip/hip_runtime.h>

namespace strix::minimax::moe::opt {

struct PrefillMoeView {
    std::uint32_t rows = 0;                     // T
    const float* y = nullptr;                   // T x 6144 normed hidden (device)
    const std::uint32_t* ids = nullptr;         // T x 4 routes (host), route_row order
    const float* weights = nullptr;             // T x 4 (host)
    const ExpertSlices* experts = nullptr;      // host array [128] of device slices
    ExpertSlices shared{};
    float* out = nullptr;                       // T x 6144 (device)
};

// Dense prefill linears on the v4 core: jobs share one input x [rows][input] (input % 128 == 0, output % 128 == 0)
// and each writes out [rows][output] fp32 with the library's gemm bits (input transform, gemm_splits split-K WMMA
// chain, ordered split sum + output transform). Supported bit widths: 3, 4, 5, 6, 8 (prefill_dense_supported). Item
// lists are staged through a small pinned ring and copied on the stream into the workspace; cleanup_prefill_dense()
// frees the ring.
inline constexpr unsigned kMaxDenseJobs = 4;
struct PrefillDenseJob {
    Exl3 w{};
    float* out = nullptr;
};
bool prefill_dense_supported(const Exl3& w);
std::size_t prefill_dense_workspace_bytes(const PrefillDenseJob* jobs, unsigned count, unsigned rows);
void enqueue_prefill_dense(const PrefillDenseJob* jobs, unsigned count, const float* x, unsigned rows, void* workspace,
                           std::size_t workspace_bytes, hipStream_t stream);
void cleanup_prefill_dense() noexcept;

// Device workspace for one call (depends on the routes: split counts follow each expert's row count).
std::size_t prefill_moe_workspace_bytes(const PrefillMoeView& v);
// Upper bound for any routes of T rows.
std::size_t prefill_moe_workspace_bound(std::uint32_t rows);
void enqueue_prefill_moe(const PrefillMoeView& v, void* workspace, std::size_t workspace_bytes, hipStream_t stream);
// WMMA core of the gate/up and down stages (same bits): 0 = library port, 1 = v2, 2 = v3, 3 = v4 (v3 + tile-major
// transformed rows; mmx_prefill_core.hpp), -1 = the process default from STRIX_MMX_PREFILL_CORE (v4 = default | v3 |
// v2 | lib). Any other value throws.
void set_prefill_core(int core);
// v4 exact-fit row tiles; -1 = STRIX_MMX_PREFILL_MTFIT (default off), 0 = off, 1 = on.
void set_prefill_mtfit(int on);
// Pinned H2D table ring; -1 = STRIX_MM_PF_TABLE_RING (default off), 0 = off, 1 = on.
// Single host thread per process; order reuse of each device workspace on the same stream.
void set_prefill_table_ring(int on);
// Wait for table copies and release pinned slots/events before HIP runtime teardown.
void cleanup_prefill_table_ring() noexcept;
// v4 single-split gate/up epilogue fusion; -1 = environment (default off), 0 = off, 1 = on.
void set_prefill_gufuse(int on);
// v4 token-major gather (one input read per token); -1 = environment (default on), 0 = off, 1 = on. Same bits.
void set_prefill_gather_tok(int on);
// Tests only: materialize fused gate/up partials. Optional device buffer [4][pairs][1536]
// captures finished gate, up, SwiGLU and FP16-rounded down input in both v4 arms.
// The caller keeps the buffer alive until the stream completes; nullptr disables capture.
// producer_only writes partials and uses the original finish kernel for GEMM timing.
void set_prefill_gufuse_debug(bool on, float* intermediates = nullptr, bool producer_only = false);
// PIPE is opt-in; -1 restores the environment. Getter preserves explicit overrides across A/B runs.
void set_prefill_pipe(int value);
int prefill_pipe_override();
// v4 + mtfit tail classes beside the MT8 launch on a side stream: -1 = STRIX_MMX_PREFILL_TAIL_STREAM (default on),
// 0 = off, 1 = on. cleanup_ releases the side stream and its events (Engine teardown).
void set_prefill_tail_stream(int on);
void cleanup_prefill_tail_stream() noexcept;
void set_prefill_core8(int value);  // -1 = environment, 0 = off (default), 1 = A LDS, 2 = shared B
// Bench only (tests/hip_mmx_moe_bench.hip stage split): 8 / 9 = v3 without A loads / without decode, WRONG outputs,
// timing only; never selected by the environment or by set_prefill_core.
void set_prefill_diagnostic_core(int core);
// Profiling only: `marker(label, stream, user)` after "start" and each launch ("gather", "gate_up[.mtN]", "finish",
// "down[.mtN]", "combine"); nullptr = off.
using PrefillMarker = void (*)(const char* label, hipStream_t stream, void* user);
void set_prefill_marker(PrefillMarker marker, void* user);
// Tests: workspace regions of one call (transformed gate/up rows, down input, all split partials), as byte ranges.
// xt / at are row-major [2][pairs][6144] / [pairs][1536] for the lib, v2 and v3 cores; v4 stores them tile-major
// [2][tpairs/16][384][16][16] / [tpairs/16][96][16][16] with group g's rows at tbegin.. (the bench un-tiles them;
// xt_bytes / at_bytes cover the row-major extent).
struct PrefillDebugRegions {
    std::size_t xt_offset = 0, xt_bytes = 0, at_offset = 0, at_bytes = 0, part_offset = 0, part_bytes = 0;
    std::size_t pairs = 0, tpairs = 0;
    std::vector<std::array<unsigned, 3>> groups;  // pair_begin, tbegin, rows
};
PrefillDebugRegions prefill_moe_debug_regions(const PrefillMoeView& v);

// strix-trellis 0.3 policy for a mul1 matrix at `rows` rows through strix_gemm (prefer_wmma): split count and WMMA
// row tile (backend.hip split_count, launch_policy.hpp wmma_tile_rows, gemv_policy.hpp measured_vector_splits).
unsigned gemm_splits(unsigned input, unsigned output, unsigned bits, unsigned rows);
unsigned gemm_tile_rows(unsigned input, unsigned output, unsigned bits, unsigned rows);

}  // namespace strix::minimax::moe::opt
