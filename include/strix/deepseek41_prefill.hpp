#pragma once

#include "strix/deepseek41_attention_state.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <array>
#include <vector>

// Device prefill of a deepseek41 prompt: T rows of the decode plan at once, layer by layer, on the fronts' many-row
// kernels (layer sites, the prefill experts' router / routed / shared WMMA kernels, the layer piece's dense prefill
// projections) and the attention piece's kernels in windows of consecutive rows. Every row gets the bits the decode
// step of its position would give, and after the prompt every layer's attention state (window ring, compressed KV,
// index keys, ratio-2 carry) is the state T decode steps leave, so decode continues from it.
//
// world 1: one process holds every expert of the range (a trimmed model; the full model does not fit one node) and
// emulates the two expert halves (canonical (shared + low) + high). world 2: rank `rank` of two keeps only its expert
// half and its vocabulary slice; the half sums of each MoE chunk and the argmax candidate are exchanged with the peer.
namespace strix::dist {
class Communicator;
}

namespace strix::deepseek41 {

struct PrefillOptions {
    bool fp16_dense = false;          // non-exact, optional hipBLASLt backend
    bool bf16_attention = false;      // non-exact, prefill only, independent of dense
    // FAST lever ds41-attn-f16wmma-v1 (STRIX_DS41_FAST_ATTN=1): non-exact f16 WMMA attention, prefill only, layers
    // [approx_layer_first, approx_layer_end); bf16_attention wins where both are set
    bool fast_attention = false;
    // FAST lever ds41-moe-relaxed-v1 (STRIX_DS41_FAST_MOE=1): routed experts in relaxed order. Read from the environment by
    // the MoE dispatch; set_precision_for_new_sequence applies an explicit value (quality arms).
    bool fast_moe = false;
    // Diagnostics/selection for the non-exact paths (no effect while both switches above are off, except null_ulp_layer):
    // projection bits 1 q_a, 2 kv, 4 q_b, 8 o_a, 16 o_b; layers [approx_layer_first, approx_layer_end); FP16 only at
    // calls of at least fp16_min_rows rows; fp16_keep_a8: FP16 operands from the A8 codes (the exact path's activations)
    std::uint32_t fp16_projections = 31, approx_layer_first = 0, approx_layer_end = 1000, fp16_min_rows = 0;
    bool fp16_keep_a8 = false;
    // FP16 output form of the FP16 backend (STRIX_DS41_PREFILL_FP16_FAST=1; needs fp16_dense): hipBLASLt writes FP16,
    // the attention norms emit the FP16 activations (no conversion pass), q_a || kv run as one GEMM, o_b as one K
    // (not two halves) outside TP
    bool fp16_fast = false;
    bool fp16_wsplit = false;  // FP16 output form with the weights as hi + lo FP16 halves (STRIX_DS41_PREFILL_FP16_WSPLIT=1)
    int null_ulp_layer = -1;          // >=0: flip the last mantissa bit of every stream value after that layer (sensitivity probe)
    std::uint32_t null_seed = 0;      // with null_ulp_layer, != 0: each value moves one ulp up or down by a hash of (seed, index)
    bool trace = false;               // per-layer router ids, indexer selections and stream row hashes (PrefillResult::trace)
    std::uint32_t quality_head_rows = 1; // >1: last N teacher-forced positions, no cone/row split
    std::uint32_t layers = 40;           // target layers [0, layers)
    std::uint32_t max_positions = 16384; // prompt rows the state and the buffers are sized for
    // Rows of one run (0: max_positions). With extend() a long prompt runs in pieces of at most max_rows rows: the row
    // buffers hold max_rows rows, the attention caches max_positions positions.
    std::uint32_t max_rows = 0;
    std::uint32_t rank = 0, world = 1;
    dist::Communicator* communicator = nullptr;  // world 2 (not owned)
    std::uint32_t chunk_rows = 1024;     // rows per MoE / attention-projection / Engram chunk (half sums of a chunk <= 21 MB)
    std::uint32_t window_rows = 8;       // rows per attention window (1..attn::kMaxCallRows; also the unit of the row split)
    // rows per routed-MoE chunk (a multiple of chunk_rows; the expert weights are read once per chunk); world 2 sends
    // a chunk's half sums in pieces of chunk_rows rows on a communication stream while the next chunk runs
    std::uint32_t moe_chunk_rows = 4096;
    std::uint32_t attention_lanes = 4;   // streams the windows of a chunk are spread over (index, select, attention)
    // world 2: tensor-parallel attention as the two-rank decode forward: rank r runs q_b of heads [32r, 32r + 32), their
    // attention, wo_a of o-groups [4r, 4r + 4) and wo_b's K-half r; the halves of every chunk are exchanged and
    // y = h0 + h1 (the declared K-split, so the same bits as one process). Window, compressor and indexer replicated.
    bool tp_attention = true;
    // world 2: the rows of the prompt split between the ranks (8-row windows alternate, window k -> rank k % 2). Each
    // rank runs the row-local work (sites, norms, Engram, projections, attention with all heads, router, shared expert)
    // on its own rows only; the window ring and the compressor run on both ranks over every row from the owners' K/V
    // rows (all-gathered), the routed experts stay split by expert (the MoE input all-gathered, each half sum sent to
    // the row's owner). Same bits as one process; replaces tp_attention.
    bool row_split = true;  // the world-2 default; false: tensor-parallel attention, every row on both ranks
    // The routed experts are not resident: two layer slots, the next layer's experts read from the file (O_DIRECT) on a
    // thread while the current layer runs (one node: the whole model does not fit). Same bits.
    bool stream_experts = false;
    // Routed experts in the expert-TP identity ds41-expert-tp2-i1152-v1 (world 2, resident experts, I = 2304, the
    // dup layout: expert_tp::packed_from_env() false): every rank all experts' K-half parts, its partial sums in the
    // existing half-sum exchange; the same bits as the expert-TP decode forward. Default off (the expert-parallel
    // identity); STRIX_DS41_PREFILL_EXPERT_TP=1 turns it on. Reads BatchForward's TP tensors in place via shared_weights.
    bool expert_tp = false;
    bool engram_direct_io = true;        // Engram rows from the NVMe with O_DIRECT (else the file mapping)
    bool stream_hashes = false;          // FNV of the last row's residual stream after each layer (gate against decode)
    bool gather_logits = true;           // world 2: exchange the vocabulary slices so both ranks hold every logit
    bool timing = false;                 // per-layer wall times (synchronises at every layer end)
    // Exact suffix replay (the full 40 layers only): decoder layers compute only the cone of rows the final state and
    // the last row's logits depend on (ds41_prefill.hpp plan_suffix_replay); every result and the live attention state
    // equal those of a full prefill.
    bool cone = true;
    // rows of the drafter's seed (0: none): the last draft_rows rows of a run; the cone also covers them, and the run
    // keeps their inputs of the last three layers (both ranks hold every seed row; draft_seed())
    std::uint32_t draft_rows = 0;
    bool routing_loads = false;          // record the routed experts' entry counts per layer (load-balance study)
    // world 2 with row_split: a run of fewer rows takes the tensor-parallel attention form instead (same bits; short
    // prompts: fewer exchanges per layer). Default 9: runs of one window (<= 8 rows) take the TP form (pair: 8 rows 115
    // vs 124 ms; at 32 rows and above row_split + joint is faster). 0: always row_split. STRIX_DS41_PREFILL_SPLIT_MIN_ROWS
    // overrides.
    std::uint32_t split_min_rows = 9;
    // world 2 row_split MoE: a run of at most this many rows takes the joint expert pass over [peer rows; own rows] (one
    // read of the expert half per layer instead of two: the short prompts are bound by those reads; same bits). 0: only
    // STRIX_DS41_PREFILL_JOINT=1 (every run). Default 2048 (pair: -10 % at 32, -12 % at 128, -6 % at 512, -8 % at 2K
    // against two passes; P11: neutral at 16K). STRIX_DS41_PREFILL_JOINT_MAX_ROWS overrides.
    std::uint32_t joint_max_rows = 512;  // P14: the joint pass loses 3 % at 2K, wins at 128-512 rows
    // Rows of a dummy prompt run once at construction (first kernel launches, scratch first touch), so the first real
    // run does not pay them; the next run starts a fresh sequence as usual (world 2: both ranks must agree). 0: none.
    std::uint32_t warmup_rows = 0;
    // Requantization overlay (ds41::WeightOverlay: tensors of the same type and size replacing the model's by name),
    // loaded in its stored Q8_0 form: the overlay's bits, the base kernels. Empty: STRIX_DS41_OVERLAY, else none.
    std::filesystem::path weights_overlay;
    // Device tensors of another forward in this process (BatchForward::file_form_weights), by name: a tensor found here
    // with the bytes this prefill would load (its expert range, its vocabulary slice) is read in place, not loaded nor
    // owned; the others are loaded as usual. Serving keeps one copy of the weights per node this way. The owner must
    // outlive the prefill.
    std::map<std::string, std::pair<const void*, std::uint64_t>> shared_weights;
};

// timing (diagnostic: the stream is synchronized around each part, which removes overlap across parts). attention
// and moe are wall intervals that INCLUDE their own exchanges; exchange is the communicator's wall time inside the
// layer (start to end of every exchange, overlapped work included) and exchange_blocked the part of it the CPU spent
// in finish() waiting for the transfer. Do not add exchange to attention + moe.
struct PrefillLayerTiming {
    double seconds = 0.0, attention = 0.0, moe = 0.0, exchange = 0.0, exchange_blocked = 0.0;
};

struct PrefillLayerTrace {
    std::vector<std::uint32_t> router_ids;   // rows x top_k, this rank's output rows
    std::vector<std::uint64_t> index_sets;   // per row: order-independent hash of the selected compressed entries (owners only)
    std::vector<std::uint64_t> row_hashes;   // per row: hash of the stream row after the layer
    std::vector<float> tail;                 // the last quality_head_rows stream rows after the layer
};

struct PrefillResult {
    std::vector<float> teacher_logits; // quality_head_rows x vocabulary; empty in normal operation
    std::vector<PrefillLayerTrace> trace; // options.trace
    std::vector<float> logits;        // last row (world 2 without gather_logits: this rank's slice, other entries zero)
    std::uint32_t next_token = 0;
    bool decoder_deferred = false;     // set_decoder_deferral admitted: no logits / next token / seed, decode state completed by the next run
    std::vector<std::uint64_t> stream_hashes;  // last row, after each layer (stream_hashes)
    double seconds = 0.0, engram_seconds = 0.0, head_seconds = 0.0;
    double engram_wait_seconds = 0.0;  // overlapped Engram read: the part the run waited for (before the Engram layers)
    std::vector<PrefillLayerTiming> layers;
    std::uint64_t row_layers = 0;        // rows x layers computed (the cone's saving against rows x 40)
    std::vector<std::vector<std::uint32_t>> routing_loads;  // routing_loads: [layer][expert] entries of the output rows
    double expert_read_seconds = 0.0, expert_wait_seconds = 0.0;  // stream_experts: file reads (thread), waits for them
    // row_split: whether the last row is this rank's (stream_hashes and the last row's top-k only there), and the rows
    // this rank computed in the encoder layers
    bool last_row_local = true;
    std::uint32_t local_rows = 0;
    // timing: attention of every layer in parts (projections for compressor / indexer, window K/V + compressor,
    // q_b, windows (index, select, attention; TP: the previous chunk's halves), o_a + wo_b)
    std::array<double, 5> attention_parts{};
};

// ---- resumable prefill (gap #2: a prefill that other requests' decode steps can interleave with) ----
//
// A PrefillTask is one run()/extend() of DevicePrefill cut into units. Work happens only inside step(k), which runs the
// next k units in order and returns when their device work is complete (the prefill's own streams synchronized, no
// exchange of the prefill pending), so the caller can run other work on the GPU and the communicator (e.g. a
// BatchForward step of other slots) before the next step(). The units are the same enqueues, in the same order, as
// the uninterrupted run (cut points only; the cut adds a stream synchronisation at the end of a step() and nothing
// else), so every row, the logits, the next token and the attention state equal those of run()/extend() for any unit
// size. The prefill's device scratch, streams and events are its own (it reads the shared weights only); the work runs
// on a worker thread of the task (its thread-local kernel choices do not leak into the caller's thread) while the
// caller waits in step(): the caller must not use the DevicePrefill (or call its forwards concurrently) during step().
//
// World 2: the list of units, and the place of every exchange in it, is the same on both ranks (loops whose count
// differs between the ranks are padded with empty units), so both ranks must call step() with the same k sequence
// (rank 0 decides, e.g. from units_within(), and sends k to rank 1 with its schedule).
//
// Unit levels (cut points used): layer = one unit per layer (and the head); phase = + the attention part and the MoE part
// of a layer; finest = + every attention chunk of chunk_rows rows (q_b, windows, o_a, wo_b), the window K/V + compressor
// part, every Engram piece of the task's engram_rows rows and every MoE chunk pass of the task's moe_chunk_rows rows.
enum class PrefillUnitLevel : std::uint32_t { layer = 0, phase = 1, finest = 2 };

struct PrefillTaskOptions {
    PrefillUnitLevel level = PrefillUnitLevel::finest;
    // rows per routed-MoE chunk inside the task (0: PrefillOptions::moe_chunk_rows; capped at that). The rows are
    // independent in the MoE (PrefillMoe: same bits for any chunk), so smaller chunks only bound the unit (each chunk
    // reads the expert half once). The prefill's own PrefillMoe runs them on row views: no extra scratch. Must be the
    // same on both ranks. Default 2048 (pair, 8K: largest unit 75 ms vs 123 ms with 4096, +1 % device time).
    std::uint32_t moe_chunk_rows = 2048;
    // rows per Engram piece inside the task (0: PrefillOptions::chunk_rows; at most that): the Engram layer's
    // gather + engram_kv projection + inject is per row, so pieces only bound the unit (finest level: one unit each)
    std::uint32_t engram_rows = 256;
    bool unit_timing = true;  // device time of every unit (hip events on the prefill's main stream)
};

struct PrefillUnitTiming {
    std::uint32_t index = 0;     // unit number in the task
    std::int32_t layer = -1;     // layer the unit starts in (-1: setup / head)
    const char* kind = "";       // what the unit runs (static string)
    double device_ms = 0.0;      // device time on the main stream from the unit's first to its last enqueue (incl. waits)
};

class DevicePrefill;
class PrefillTask {
public:
    ~PrefillTask();  // a task not done is cancelled (see cancel())
    PrefillTask(const PrefillTask&) = delete;
    PrefillTask& operator=(const PrefillTask&) = delete;

    // Runs the next min(max_units, units left) units (max_units >= 1) and returns how many ran. On return their device
    // work is complete and no exchange is in flight. The last unit ends with the result (next token, logits).
    std::uint32_t step(std::uint32_t max_units = 1U);
    [[nodiscard]] bool done() const noexcept;
    [[nodiscard]] std::uint32_t units_done() const noexcept;
    // Rank 0's scheduler: how many of the next units are predicted to fit in budget_ms (>= 1), from the device times of
    // the same units of the previous layer (the layers' units repeat), else the largest unit seen so far.
    [[nodiscard]] std::uint32_t units_within(double budget_ms) const;
    // The run's result (valid once done()); throws if the task failed or was cancelled.
    [[nodiscard]] const PrefillResult& result() const;
    [[nodiscard]] const std::vector<PrefillUnitTiming>& unit_timings() const noexcept;
    // wall time of every step() call so far (ms), and their maximum
    [[nodiscard]] const std::vector<double>& step_walls() const noexcept;
    // Abandons the run at the current unit boundary (no further units run). World 2: both ranks must cancel at the
    // same unit. The DevicePrefill's sequence is then invalid: its next call must be run() (a fresh sequence).
    void cancel();

    struct State;

private:
    friend class DevicePrefill;
    explicit PrefillTask(std::unique_ptr<State> s);
    std::unique_ptr<State> s_;
};

class DevicePrefill {
public:
    DevicePrefill(const std::filesystem::path& gguf, const PrefillOptions& options);
    ~DevicePrefill();
    // The overlay's path, identity, tensor count, bytes and content FNV (empty without an overlay).
    [[nodiscard]] std::string overlay_description() const;
    DevicePrefill(const DevicePrefill&) = delete;
    DevicePrefill& operator=(const DevicePrefill&) = delete;

    // Discards the live sequence: the next operation must be run(). External prefix/state caches
    // must use separate precision identities. Intended for controlled quality comparisons.
    void set_precision_for_new_sequence(bool fp16_dense, bool bf16_attention);
    // The same, with every diagnostic/selection field of PrefillOptions (fp16_*, approx_*, null_ulp_layer, trace).
    void set_precision_for_new_sequence(const PrefillOptions& precision);
    // The exact dense selector for the next sequences (same bits either way; in-process A/B of its speed): on = every
    // exact dense call with the per-site form of STRIX_DS41_DENSE_TALL=auto and q_b / o_a on the dense kernel; off =
    // the loaded configuration (environment).
    void set_dense_select(bool on);
    // The whole prompt from position 0 (a fresh sequence).
    PrefillResult run(std::span<const std::uint32_t> tokens);
    // The next piece of the same sequence: tokens at positions [rows so far, + tokens.size()), reading the attention
    // state (rings, compressed KV and index keys, ratio-2 carries) and the Engram history the earlier runs left. The rows
    // so far must be a multiple of window_rows. Every row and the final state equal those of one run of the whole prompt.
    PrefillResult extend(std::span<const std::uint32_t> tokens);
    // The same runs as resumable tasks (see PrefillTask; the tokens are copied). Nothing runs until the task's first
    // step(). While a task is not done, this prefill must not be used otherwise (run / extend / begin / export_state /
    // import_state / state_hashes throw or are undefined); destroy or finish the task first.
    std::unique_ptr<PrefillTask> begin_run(std::span<const std::uint32_t> tokens, const PrefillTaskOptions& task = {});
    std::unique_ptr<PrefillTask> begin_extend(std::span<const std::uint32_t> tokens, const PrefillTaskOptions& task = {});
    [[nodiscard]] std::uint32_t positions() const noexcept;  // rows of the sequence so far
    // draft_rows: the last run's seed -- the input streams of layers layers-3, layers-2, layers-1 of its last `rows`
    // rows ([rows][hc][hidden] each, device memory, in position order from `first`), valid until the next run.
    struct DraftSeed {
        std::array<const float*, 3> streams{};
        std::uint32_t rows = 0, first = 0;
    };
    [[nodiscard]] DraftSeed draft_seed() const noexcept;
    // FNV of every layer's attention state after the last run (the ring's live window, the used compressed KV and
    // index-key entries, carry, the last row's top-k), labelled; the same labels and order as DeviceForward::state_hashes.
    [[nodiscard]] std::vector<std::pair<std::string, std::uint64_t>> state_hashes() const;
    // The attention state after the last run (deepseek41_attention_state.hpp), for a decode forward to continue from:
    // every layer's ring, the KV sources' caches and carries, and the last row's top-k on the rank that computed it
    // (row_split: topk_known false on the other). position = the prompt length; tokens and next_token stay empty / 0
    // (the caller holds the prompt and PrefillResult::next_token).
    [[nodiscard]] AttentionState export_state() const;
    // Continue a sequence from a state (another run's export, read from a file): rings, caches, carries, the Engram
    // history of state.tokens and the rows so far = state.position (a multiple of window_rows); then extend().
    void import_state(const AttentionState& state);
    // In-run snapshots (the serve's --prefix-inrun-snapshots): positions p of the next run / extend / begin_* (P0 < p <
    // P0 + rows, multiples of window_rows) where that run also keeps the attention state, so a prefix cache snapshots p
    // without splitting the run there (a split reads every expert once more and replays the cone again). The decoder
    // layers' cone then also covers the rows the state at p depends on (the extra rows are exact), and each layer's ring
    // at p is copied out of the run's ring before the next layer reuses it. After the run, export_state_at(p) gives the
    // state a run ending at p exports, bitwise (rings, the caches' first p / ratio entries, a ratio-2 carry: empty at an
    // even p), with topk_known false; tokens and next_token are the caller's. The points apply to one run only.
    void set_snapshot_points(std::vector<std::uint32_t> positions);
    // The tokens of the sequence's next run (the piece after the next run / extend / begin; copied, one run only). Once
    // that run's own Engram rows are read, a thread reads this rank's Engram rows of `next` into the reader's payload
    // cache, so the next run's read is served from host memory (the same bytes; nothing else changes). A next run with
    // other tokens or boundaries reads what the cache misses as before. STRIX_DS41_ENGRAM_LOOKAHEAD=0: ignored.
    void set_engram_lookahead(std::span<const std::uint32_t> next);
    // The next run (run / extend / begin) is followed by `next_rows` more rows of the same sequence and nobody reads its
    // decode state, logits or seed: its decoder layers then skip their cone (only the first decoder layer's compressed-KV /
    // index-key entries of every row, which the later runs read, are computed). Admitted only for the full layer set with
    // the cone, no snapshot points in the run, and a following run of `next_rows` whose own cone (rounded to the window
    // rows, with its draft rows) starts at least one attention window after its first row, so it reads no decoder ring
    // entry of the deferred run; the caller must then run that continuation (an extend of at least `next_rows` rows).
    // Both ranks must ask the same (the run checks it and throws on a mismatch). Until the next run completes,
    // export_state / export_state_at / state_hashes refuse and draft_seed is empty. STRIX_DS41_DEFER_DECODER=0: never admitted.
    void set_decoder_deferral(std::uint32_t next_rows);
    [[nodiscard]] AttentionState export_state_at(std::uint32_t position) const;
    [[nodiscard]] std::uint64_t weight_bytes() const noexcept;         // loaded (owned) by this prefill
    [[nodiscard]] std::uint64_t shared_weight_bytes() const noexcept;  // read in place from shared_weights
    // The bytes of weights a prefill with these options would load itself (not found in options.shared_weights; streamed
    // experts: both layer slots), from the model's tensor inventory, before constructing it (memory admission).
    [[nodiscard]] static std::uint64_t owned_weight_bytes(const std::filesystem::path& gguf, const PrefillOptions& options);
    [[nodiscard]] double load_seconds() const noexcept;

    struct Impl;

private:
    friend class PrefillTask;
    friend struct PrefillTask::State;
    PrefillResult run_from(std::span<const std::uint32_t> tokens, std::uint32_t start);
    std::unique_ptr<PrefillTask> begin_from(std::span<const std::uint32_t> tokens, bool extend, const PrefillTaskOptions& task);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::deepseek41
