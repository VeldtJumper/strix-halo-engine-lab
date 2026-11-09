#pragma once

#include "strix/deepseek41_attention_state.hpp"
#include "strix/deepseek41_model.hpp"
#include "strix/ds41_pipeline_wire.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Several requests in one forward step of the deepseek41 decode plan, the weights read once per step.
//
// A step takes the rows of up to `slots` requests (sequences). The rows of one request form a segment: consecutive
// positions of that request (one decode row, or several rows of a prompt chunk; later a verify window). Every
// stateless operation runs once over all the rows of the step: the hyper-connection sites, the norms, the Q8_0
// projections (q_a, q_b, kv, wo_a, wo_b), the F16 projections (Engram, compressor, indexer), the router, the shared
// expert, the routed experts (the union of the rows' selections), the combine, the head and the argmax. Every
// operation with per-request state runs per segment on that request's own state: the window ring, the compressed
// KV and index caches with the ratio-2 carry (window_write, compress, index scores and selection, attention), the
// Engram n-gram history and the positions (RoPE rows).
//
// Exactness: the arithmetic of a row never depends on the other rows of the step nor on their number. Every kernel
// computes each row with the same declared order whatever R is (the verify-window contract), the per-request
// kernels see only their own segment, and the experts are grouped by expert but written per (row, slot) and summed
// in ascending expert id per row. So each request's tokens and logits are bitwise those of running it alone (gated
// by strix-deepseek41-batch --mode both, and against strix-deepseek41-rank).
//
// World 2 is the two-node layout of DeviceForward: rank r holds experts [r n/2, (r+1) n/2) and its vocabulary slice
// of the head; the half sums, the tensor-parallel attention partials and the argmax candidates of all the rows of a
// step travel in one exchange each.
namespace strix::dist {
class Communicator;
}

namespace strix::deepseek41 {

struct BatchForwardOptions {
    std::uint32_t layers = 40;
    std::uint32_t max_positions = 4096;  // per request
    std::uint32_t slots = 8;             // requests that can hold state at once
    // Shared KV pool: the compressed KV and index-key caches of every slot live in one arena per KV-source layer sized
    // for pool_positions positions in total (plus a slack of 16 per slot); a slot holds room only after reserve(). 0: no
    // pool, every slot owns caches for max_positions (the old layout). Rings and carries stay per slot either way.
    std::uint32_t pool_positions = 0;
    std::uint32_t rank = 0, world = 1;   // world 1 (every expert, emulated halves) or 2
    dist::Communicator* communicator = nullptr;
    bool tp_attention = false;        // world 2: the split of DeviceForwardOptions::tp_attention
    bool engram_direct_io = false;    // Engram rows from the NVMe with O_DIRECT (else from the file mapping)
    bool slice_logits = false;        // keep each row's logits of this rank's vocabulary slice on the host
    bool gather_logits = false;       // world 2: exchange the slices so every row's full logits are on the host
    bool layer_timing = false;        // per-layer wall times (synchronises at every layer end)
    bool capture_main_hidden = false; // keep the inputs of layers [layers - 3, layers) of every row (DSpark drafter input)
    // Requantization overlay (ds41::WeightOverlay: tensors of the same type and size replacing the model's by name),
    // loaded in its stored Q8_0 form: the overlay's bits, the base kernels. Empty: STRIX_DS41_OVERLAY, else none.
    std::filesystem::path weights_overlay;
};

// One row of a step: a token of request `slot` (rows of the same slot are consecutive in the step and at
// consecutive positions starting at position(slot)).
struct BatchRow {
    std::uint32_t slot = 0;
    std::uint32_t token = 0;
};

struct BatchStepStats {
    std::uint32_t rows = 0, segments = 0;
    double seconds = 0.0;           // wall time of the step, inputs to tokens on the host
    double engram_seconds = 0.0;    // Engram row reads
    double exchange_seconds = 0.0;  // inside the communicator (world 2)
    // per layer: the routed experts of this rank used by the step's rows (the union), and the most rows one of them
    // served; filled every step (copied asynchronously, read after the step)
    std::vector<std::uint32_t> union_experts, max_rows_per_expert;
    std::uint64_t expert_bytes = 0;  // routed-expert weight bytes of this rank read by the step (union x expert size)
    std::vector<double> layer_seconds;  // layer_timing
};

class BatchForward {
public:
    static constexpr std::uint32_t max_rows = 8;  // legacy step limit and maximum segment of one request
    static constexpr std::uint32_t extended_max_rows = 16;
    // Total independent rows, not the length of one request's verify window. The latter stays <=8
    // so existing rings, snapshots and rollback semantics keep their layout.
    [[nodiscard]] static std::uint32_t configured_row_limit() noexcept {
        const char* v = std::getenv("STRIX_DS41_BATCH_ROWS16");
        return v && v[0] == '1' && v[1] == '\0' ? extended_max_rows : max_rows;
    }
    [[nodiscard]] std::uint32_t row_limit() const noexcept;

    BatchForward(const std::filesystem::path& gguf, const BatchForwardOptions& options);
    ~BatchForward();
    // The overlay's path, identity, tensor count, bytes and content FNV (empty without an overlay).
    [[nodiscard]] std::string overlay_description() const;
    // trellis sites (STRIX_DS41_TRELLIS / _MIX): the identity of the prepared sites ("" without trellis; see
    // ds41_trellis.hpp trellis_config_identity), their device bytes (weights + staging + HEADOUT rows + o_a group staging,
    // == footprint().trellis) and the library workspaces queried at load (not in footprint()).
    [[nodiscard]] std::string trellis_identity() const;
    [[nodiscard]] std::uint64_t trellis_bytes() const noexcept;
    [[nodiscard]] std::uint64_t trellis_library_bytes() const noexcept;
    BatchForward(const BatchForward&) = delete;
    BatchForward& operator=(const BatchForward&) = delete;

    // An empty request in `slot` (position 0, zero caches and carries, empty Engram history): a new request.
    void reset(std::uint32_t slot);
    // KV pool: give `slot` room for `positions` positions (<= max_positions) from the arenas, first fit, and reset it;
    // false when no free range is large enough. A slot already holding room gives it back first (also when false).
    // Without a pool: true when positions <= max_positions (the slot's own caches), and the slot is reset.
    bool reserve(std::uint32_t slot, std::uint32_t positions);
    void release(std::uint32_t slot);                                 // KV pool: the slot's room back (no-op without one)
    [[nodiscard]] std::uint32_t capacity(std::uint32_t slot) const;  // positions the slot may hold now
    [[nodiscard]] std::uint64_t pool_free_positions() const;          // KV pool: the largest reservation that fits now
    // `to` continues exactly as `from` would (its state copied): measurement runs reuse one filled context.
    void copy_slot(std::uint32_t from, std::uint32_t to);
    // `slot` continues the sequence of a prefill's (or a forward's) attention state, as DeviceForward::import_state:
    // rings, caches, carries, Engram history and position. The slot must have room (capacity > state.position).
    void import_state(std::uint32_t slot, const AttentionState& state);
    [[nodiscard]] std::uint32_t position(std::uint32_t slot) const;
    // The attention state of `slot` (as DeviceForward::export_state; tokens empty, next_token 0: the caller knows them;
    // the last row's top-k left out). kv_from (a multiple of every ratio): the KV and index-key entries of positions
    // [kv_from, position) only (a prefix cache that holds the earlier ones takes a session snapshot this way).
    [[nodiscard]] AttentionState export_state(std::uint32_t slot, std::uint32_t kv_from = 0) const;

    // One forward step over `rows` (1..row_limit(), at most max_rows per request's segment);
    // returns each row's greedy token, in the order of `rows`.
    const std::vector<std::uint32_t>& step(std::span<const BatchRow> rows);
    // Pipeline TAIL of several requests (strix/ds41-pipeline-v1, ds41_pipeline_wire.hpp): a step whose layers
    // [0, first_layer) ran elsewhere (a FRONT stage). boundaries[i] is the cut after layer first_layer - 1 of the step's
    // i-th segment (the runs of equal slots of `rows`, in order): its position, rows, tokens, the stream after that
    // layer, its FFN mix and the sections the cut carries (the KV entries the FRONT completed go into the slot's own
    // caches of those source layers, the top-k and candidate rows into the step's scratch). The step then runs layers
    // [first_layer, layers) and the head over every row exactly as step() does from those inputs, so every row's bits
    // are those of the whole step (and of running its request alone). The slots' states of layers < first_layer are
    // not advanced (their sequences live in the FRONT): export_state() of such a slot is valid only for layers >=
    // first_layer and the KV sources the cut carries. Engram rows are read only for layers >= first_layer.
    const std::vector<std::uint32_t>& step_boundary(std::span<const BatchRow> rows, std::uint32_t first_layer,
                                                    std::span<const ds41::pipe::Boundary> boundaries);
    // The FRONT half of a step on this forward (tests and single-node gates of the split): layers [0, cut) of `rows`, then
    // the cut after layer cut - 1 of every segment as step_boundary reads it (stream, FFN mix, the slot's KV entries the
    // rows completed on the cut's KV sources, the rows' top-k and candidate blocks). The slots' positions and Engram
    // histories do not move; step_boundary(rows, cut, boundaries) right after completes the step, bitwise the step() of
    // the same rows (a commit() after it rolls back every layer, as after step()).
    [[nodiscard]] std::vector<ds41::pipe::Boundary> step_front(std::span<const BatchRow> rows, std::uint32_t cut);
    [[nodiscard]] const std::vector<std::uint32_t>& last_tokens() const noexcept { return tokens_; }
    // Verify windows: after a step, keep rows [0, kept) of `slot`'s segment and roll the others back (its position,
    // the ratio-2 compressor carry, the Engram history; ring and compressed entries of dropped rows are rewritten
    // before any later read), as DeviceForward::commit_window does for one sequence.
    void commit(std::uint32_t slot, std::uint32_t kept);
    // slice_logits: row r's logits of this rank's vocabulary slice; gather_logits: row r's full logits.
    [[nodiscard]] std::span<const float> row_logits(std::uint32_t row) const;
    // Between steps, identically on both ranks: gather full scores only for a sampled batch.
    // Allocates host scratch lazily; leaves all target arithmetic and greedy decisions unchanged.
    void set_gather_logits(bool enabled);
    [[nodiscard]] const BatchStepStats& last_stats() const noexcept { return stats_; }
    // capture_main_hidden: the input streams of layers layers-3, layers-2, layers-1 of the last step's rows, device
    // memory [rows][hc][hidden] each in the order of the step's rows (as DeviceForward::captured_streams); valid until
    // the next step, ordered on stream().
    [[nodiscard]] std::array<const float*, 3> captured_streams() const;
    [[nodiscard]] void* stream() const noexcept;  // the hipStream_t of the steps

    // Weight bytes read by every step whatever its rows (all but the routed experts; with TP only this rank's
    // parts of q_b, wo_a and wo_b) and the bytes of one routed expert (gate + up + down) per layer.
    [[nodiscard]] std::uint64_t fixed_bytes_per_step() const noexcept;
    [[nodiscard]] std::uint64_t expert_bytes() const noexcept;
    [[nodiscard]] std::uint64_t weight_bytes() const noexcept;        // resident on this rank
    [[nodiscard]] std::uint64_t state_bytes_per_slot() const noexcept;  // device state of one request (without the pool)
    [[nodiscard]] std::uint64_t pool_bytes() const noexcept;            // the KV pool's arenas (0 without one)
    [[nodiscard]] std::uint64_t workspace_bytes() const noexcept;       // activations and scratch
    [[nodiscard]] double load_seconds() const noexcept { return load_seconds_; }
    // The device tensors held in the model file's form (this rank's expert range and vocabulary slice), by name: a
    // device prefill in the same process reads them instead of loading a second copy (PrefillOptions::shared_weights).
    // Tensors held in another form (the F8 Engram KV, the BF16 routers, packed overlay sites) are left out.
    [[nodiscard]] std::map<std::string, std::pair<const void*, std::uint64_t>> file_form_weights() const;
    [[nodiscard]] const TargetModel& model() const noexcept;
    // The decode identity: true when the routed experts run expert TP (ds41-expert-tp2-i1152-v1), false expert parallel.
    [[nodiscard]] bool expert_tp() const noexcept;
    // Device memory now: free and total bytes (hipMemGetInfo).
    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> device_memory() const;
    // World 2: `out` to the peer and the peer's message into `in` (same size on both ranks, <= 64 KiB) through the
    // communicator, on this forward's stream (the serving control channel between steps).
    void exchange_host(std::span<const std::byte> out, std::span<std::byte> in);
    // GPU-direct: no receive left posted early for a guessed next exchange (Communicator::quiesce on this forward's
    // stream, waited). Both ranks at the same point between exchanges, e.g. a server going idle; nothing on the host path.
    void quiesce();

    // What a BatchForward with `options` would hold on this rank, from the model header alone (no device work):
    // resident weights, device state per request (slot) and activations/scratch.
    // state_per_slot: rings and carries, plus the caches for max_positions without a pool; pool: the shared arenas.
    struct Footprint {
        std::uint64_t weights = 0, state_per_slot = 0, workspace = 0, pool = 0;
        std::uint64_t trellis = 0;  // trellis sites on top of the dense weights (they stay loaded), as trellis_bytes()
    };
    [[nodiscard]] static Footprint footprint(const std::filesystem::path& gguf, const BatchForwardOptions& options);

    struct Impl;

private:
    const std::vector<std::uint32_t>& run_step(std::span<const BatchRow> rows, std::uint32_t first_layer,
                                               std::span<const ds41::pipe::Boundary> boundaries, std::uint32_t front_cut,
                                               std::vector<ds41::pipe::Boundary>* front_out);
    std::unique_ptr<Impl> impl_;
    std::vector<std::uint32_t> tokens_;
    BatchStepStats stats_;
    double load_seconds_ = 0.0;
};

}  // namespace strix::deepseek41
