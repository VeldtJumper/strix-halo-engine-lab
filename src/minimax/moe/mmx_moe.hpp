#pragma once
// minimax-m3 decode MoE and dense EXL3 linears for one TP2 rank (gfx1151), N = 1..8 rows.
//
// Identity (bitwise): per row, the same bits as the bring-up chain of tests/hip_minimax_g4.hip at <= 8 rows:
//   router      strix/minimax_moe.hpp route_row (FP32 logits, lane l chains j = l, l + 32, ...; butterfly; sigmoid +
//               bias choice; top-4 by (choice desc, id asc); w = s / (sum + 1e-20) x 2);
//   experts     strix_gemv (strix-trellis 0.3, the decode GEMV; not strix_gemm, which takes WMMA) for gate (w1) and up (w3) [6144 -> 1536 rank slice, K3], swiglu
//               (min(g, 7) sigmoid(1.702 min(g, 7)) (clamp(u, +-7) + 1)), strix_gemv down (w2) [1536 -> 6144];
//   combine     acc = 0; acc = fmaf(w_e, y_e, acc) over the row's 4 experts in ascending expert id; acc + shared;
//   shared      the same MLP with the shared expert (K5), added unscaled;
//   linears     strix_gemv (input transform, split GEMV core, output transform) for the dense sites.
// Every row's bits depend only on that row (batch invariance N = 1..8): the per-row chains, split boundaries and
// reduction orders never depend on the other rows of the call.
//
// Launch structure of one MoE layer (enqueue_moe, defaults; no host synchronisation):
//   1. router + shared expert gate/up in one launch (128 router CTAs, the last one does top-4 + dispatch table;
//      the other CTAs stream the shared expert's gate/up meanwhile)          STRIX_MMX_MOE_OVERLAP=0 disables;
//   2. routed gate/up: the library GEMV grid (CTA = 4 tile columns x 1 split) over all active slots, split input
//      transform in LDS, raw split partials; the last CTA per block does H, svh, SwiGLU and the down transform;
//   3. down, same grid; the last CTA per output block does H, svh and the fixed-order combine (+ shared).
// Eight-row v4 gate/up tiles interleave gate and up with or without overlap; STRIX_MMX_MOE_GU_INTERLEAVE=0 restores the former z planes
// for diagnosis (same CTA count, partials and counters). It is read per call, as is STRIX_MMX_MOE_SINGLE.
// The engine resolver supplies 1 for both switches; raw-library unset is 0 for both.
// STRIX_MMX_MOE_SINGLE=1 enables the coalesced singleton reader: device dispatch selects it
// only at N=8, U>=25, S>=18, nrows=1, K3, narrow W=4, Depth=4. Other slots use the existing reader in the same launch.
// STRIX_MMX_MOE_DOWN_SPLIT (read per call; resolver default 1, raw-library unset 0) makes v4 down partial-only and runs
// one stream-ordered finish/combine launch (48 CTAs), removing (U+1)*8 narrow / (U+1)*4 wide arrivals per output tile.
// 1 = only the measured winning regime (route::moe_down_split: N=4, wide, W=4, Depth=4), 2 = every v4 down (diagnostic,
// the B8 remeasure), 0 = off. It reuses the same scratch and ordered finish; only the combine launch publishes.
// Without the split, both reader settings keep three overlap launches; N<8, Depth=8/12 and wide/W=8 use the same
// kernels at SINGLE=0/1.
// Contract: 16 gate/up splits, 4 down splits, ascending k tiles, both fdot2 chains, reduce4
// offsets 2 then 1, ascending split sum, H128/scales, SwiGLU rounding, ascending-expert fmaf, then shared add.
// The fused residual remains h+(p0+p1). Native/model/complete-workload qualification is still required.
// Flags (all paths bitwise): STRIX_MMX_MOE_WIDE unset = wide at N2-4 only (measurement), 1 = wide at every N (expert
// gate/up/down: CTA = 16 columns, wave = 4 columns
// streaming 16-byte k-row runs; 0 = 4-column CTAs), STRIX_MMX_MOE_V 4 (1/3 older forms), STRIX_MMX_MOE_DEPTH3 4 (8/12),
// STRIX_MMX_MOE_W 4 (8), STRIX_MMX_MOE_DEPTH8R 0 (2), STRIX_MMX_ROUTER 2 (1 = 16-CTA form; used when the
// post-exchange is fused, h != nullptr), STRIX_MMX_LINEAR_V unset = a measured per-site / row-count schedule of
// v4 / v8 / v9 (mmx_moe.hip scheduled_linear_version; 1, 3..9 force one form). Not for the head: the library's strix_gemv
// is faster there (it already runs at ~230 of the ~247 GB/s read ceiling).
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace strix::minimax::moe::opt {

inline constexpr unsigned kHidden = 6144, kInter = 1536 /* per rank */, kExperts = 128, kTopE = 4, kMaxRows = 8;
inline constexpr unsigned kMaxSlots = kMaxRows * kTopE;  // routed slots (distinct experts of a call)
inline constexpr unsigned kSharedSlot = kMaxSlots;        // slot index of the shared expert
inline constexpr unsigned kMaxPairs = kMaxSlots + kMaxRows;  // (slot, row) pairs: routed <= 32, shared <= 8

// One EXL3 matrix (rank slice) resident on the device, in the strix-trellis device layout (= file layout):
// trellis [input / 16][output / 16][bits * 8] uint32 words, suh [input] FP16, svh [output] FP16.
struct Exl3 {
    const std::uint32_t* trellis = nullptr;
    const std::uint16_t* suh = nullptr;
    const std::uint16_t* svh = nullptr;
    std::uint32_t input = 0, output = 0, bits = 0;
};
struct ExpertSlices {  // w1 (gate), w3 (up): 6144 -> 1536; w2 (down): 1536 -> 6144
    Exl3 gate, up, down;
};

// Written by the router kernel, read by the later MoE kernels (device memory).
struct Dispatch {
    std::uint32_t rows;                     // N of the call
    std::uint32_t count;                    // distinct routed experts (slots 0..count-1, ascending expert id)
    std::uint32_t expert[kMaxSlots];
    std::uint32_t nrows[kMaxSlots + 1];     // rows per slot (slot kSharedSlot = all rows)
    std::uint32_t first[kMaxSlots + 1];     // first (slot, row) pair index of the slot
    std::uint32_t row[kMaxSlots][kMaxRows]; // call row of each local row, ascending
    std::uint32_t row_pair[kMaxRows][kTopE];  // per call row: its 4 pairs in ascending expert id
    float row_weight[kMaxRows][kTopE];        // and their routing weights
    std::uint32_t row_expert[kMaxRows][kTopE];  // and their expert ids
    std::uint32_t ids[kMaxRows][kTopE];       // route_row output (choice order)
    float weights[kMaxRows][kTopE];
    // Device-built execution buckets; entries are original ascending-id slots, preserving pair numbering/combine.
    std::uint32_t single_count, reuse_count;
    std::uint32_t single_slot[kMaxSlots], reuse_slot[kMaxSlots];
};

struct MoeWorkspace {  // one device allocation of moe_workspace_bytes(), zeroed once before first use
    void* base = nullptr;
};
std::size_t moe_workspace_bytes();
// Start of the routed pipeline's sequencing words (route epoch, ticket, ready epochs) inside the workspace: they change
// on every routed call by design, so workspace reuse checks compare the bytes before it and check the ticket separately.
std::size_t moe_pipe_offset();
// Bench diagnostic: print HIP VGPR/scratch/LDS for all Single=true instances; throw on scratch or query failure.
void check_moe_single_resources();

// decode (STRIX_MM_XB_PUBLISH): the producer kernel publishes its own exchange message (lean form): every lane releases
// its stores at agent scope, each CTA counts on `counter` (agent scope), and only the last CTA of the grid resets the
// counter, releases at system scope and stores `sequence` into the host-visible `flag`. flag == nullptr: no publish
// (today's code). Only the CTA that finishes each 128-output tile counts (tiles arrivals per message). Accepted by the decode forms the engine pins (producer_publish_supported()), else enqueue throws.
struct PublishTarget {
    std::uint64_t* flag = nullptr;
    unsigned* counter = nullptr;
    std::uint64_t sequence = 0;
    unsigned tiles = 0;  // set by the enqueue functions: output tiles whose finishers count (callers leave 0)
};

struct MoeView {
    std::uint32_t rows = 0;               // 1..8
    const float* y = nullptr;             // rows x 6144, normed hidden (post-exchange output; written when h is set)
    // Optional fused post-exchange (residual + Gemma norm of strix/minimax_moe.hpp post_exchange_row, same bits): when
    // h is set, the router computes h += part0 + part1 and y itself (one launch instead of post-exchange + router).
    float* h = nullptr;                   // rows x 6144 residual, updated in place
    const float* part0 = nullptr;         // rank 0's partial rows
    const float* part1 = nullptr;         // rank 1's partial rows
    const float* norm_w = nullptr;        // 6144 Gemma norm weights
    const float* router_w = nullptr;      // 128 x 6144 FP32 (replicated)
    const float* bias = nullptr;          // 128 score-correction bias
    const ExpertSlices* experts = nullptr;  // device array [128] (routed experts, K3)
    ExpertSlices shared{};                // shared expert slices (K5)
    float* out = nullptr;                 // rows x 6144: routed sum + shared (this rank's partial, to exchange)
    std::uint32_t* ids = nullptr;         // optional rows x 4 (route_row order)
    float* weights = nullptr;             // optional rows x 4
    MoeWorkspace ws{};
    PublishTarget pub{};                  // optional: the down/combine launch publishes `out` (STRIX_MM_XB_PUBLISH)
    // Measurement only: device [rows][4], distinct ids <128 per row and finite weights. Overrides selection at
    // route_tail after the real router executes; leaves the overlap/expert path intact. Both null in production.
    const std::uint32_t* controlled_ids = nullptr;
    const float* controlled_weights = nullptr;
    // Optional diagnostic device counters [gate/up, down], caller-zeroed; count actual reader invocations.
    std::uint32_t* single_reader_counts = nullptr;
};
// Throws std::invalid_argument on unsupported shapes (bits other than 3/5 for experts/shared, rows outside 1..8).
void enqueue_moe(const MoeView& v, hipStream_t stream);
// Profiling only: marks[0..1] after router/shared (or prepare), after routed gate/up. Optional down_done after
// partial production (before the separate combine when DOWN_SPLIT=1; after fused down+combine otherwise).
// Same production launches; a fused path has no independently measurable combine stage.
void enqueue_moe_profiled(const MoeView& v, hipStream_t stream, hipEvent_t* marks, hipEvent_t down_done = nullptr);
// Pieces of enqueue_moe (for the microbenchmark and for a caller that overlaps them with other work).
void enqueue_router(const MoeView& v, hipStream_t stream);
// prepare + gate_up + down (+ combine when split); marks (optional, 2 events): after prepare and gate_up.
// Optional down_done has the same boundary as enqueue_moe_profiled.
void enqueue_experts(const MoeView& v, hipStream_t stream, hipEvent_t* marks = nullptr, hipEvent_t down_done = nullptr);
const Dispatch* moe_dispatch(const MoeWorkspace& ws);        // device pointer

// ---------------------------------------------------------------- router for many rows (prefill chunks)
// Same bits as route_row / the fused post-exchange router per row (lane chains, butterfly, sigmoid + bias, top-4,
// weights), computed as a tiled pass: one wave per expert over 32 rows per CTA (router weights read once per 32 rows
// instead of once per row), then one thread per row for the top-4. scratch: rows x 128 x 2 floats.
void enqueue_router_rows(unsigned rows, const float* y, const float* router_w, const float* bias, float* scratch,
                         std::uint32_t* ids, float* weights, hipStream_t stream);

// ---------------------------------------------------------------- dense linears (attention projections, shared /
// dense MLP halves, head slice): y[r][0..output) = strix_gemm(x[r][0..input)) for r < rows <= 8, several matrices
// with their own inputs and outputs in one launch pair (prepare + core with the output transform in the last CTA of
// each 128-output block).
struct LinearJob {
    Exl3 w{};
    const float* x = nullptr;
    std::uint32_t x_stride = 0;  // floats between rows
    float* y = nullptr;
    std::uint32_t y_stride = 0;
};
inline constexpr unsigned kMaxJobs = 4;
std::size_t linear_workspace_bytes(const LinearJob* jobs, unsigned count);  // for 8 rows
// marks (optional, 1 event, profiling): recorded after the prepare launch of the two-launch forms (v1/v4).
void enqueue_linears(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream,
                     hipEvent_t* marks = nullptr);
// The same with the measured per-shape / N schedule regardless of STRIX_MMX_LINEAR_V (which is read once per process
// and pinned by the engine for its own sites): for callers such as the DSpark drafter. Same output bits.
// draft_block (v11): the caller is the DSpark draft block (DraftLinears); at 7 rows its q+k+v / o / gate+up groups
// then take linear core v11 (same bits, measured faster there; STRIX_MMX_LINEAR_V11=0 disables). Other callers: false.
void enqueue_linears_scheduled(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream,
                               bool draft_block = false);
// One form for this call (1, 3..10, as STRIX_MMX_LINEAR_V=version would select it), for callers whose shape the schedule
// does not cover (the DSpark draft-only head: v9 at R=7). Same output bits.
void enqueue_linears_version(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream, int version);
// monodec (STRIX_MM_XB_PREP): the input transform of the two-launch forms written by the producer of x instead of the
// prepare launch. linear_xt_offsets gives, per job, the offset in halves (_Float16) of its transformed input inside the
// workspace (row r of job i at workspace + offs[i] + r * input); the producer writes it with dev::transform_block (the
// prepare kernel's function, same bits) and the call then passes inputs_prepared = true (no prepare launch). Only the
// two-launch forms (STRIX_MMX_LINEAR_V 1, 4, 7, 8, 9, 10 / scheduled: all of them run the same prepare transform and
// accept prepared inputs); the one-launch forms 3, 5, 6 transform inside their core and throw.
void linear_xt_offsets(const LinearJob* jobs, unsigned count, std::size_t* offs);
void enqueue_linears_prepared(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream);
// The same call whose core launch publishes its outputs (lean producer publish, PublishTarget above); prepared as above.
void enqueue_linears_publish(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream,
                             const PublishTarget& pub, bool prepared);
// True when the process's linear form (v4) and MoE decode form (overlap, v4, 4 waves) carry the publish epilogue.
bool producer_publish_supported();
// Profiling only: when set (device buffer, 5 x 64-bit per core CTA), the v4/v7 core writes per CTA the realtime counter
// at start, after its inputs are in LDS, after its split, at exit (or finish start), and its HW_ID1. nullptr = off.
void set_linear_trace(std::uint64_t* trace);

}  // namespace strix::minimax::moe::opt
