#pragma once

#include "strix/deepseek41_batch_forward.hpp"
#include "strix/deepseek41_batch_schedule.hpp"
#include "strix/ds41_dspark_driver.hpp"
#include "strix/hip_ds41_dspark.hpp"
#include "strix/lookup/index.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

// DSpark windows inside BatchForward, several requests per step (N > 1).
//
// Every decoding request of a step verifies its own window: its last token, then the drafts its own drafting context
// proposed, up to 6 rows (1 row = a plain decode row, no draft). The windows of all requests, and the prompt chunks
// of requests still reading their prompt, go in one BatchForward step (at most BatchForward::max_rows rows). Each
// request then keeps the rows whose drafts equalled the target's greedy token after the previous row, plus that
// correction: commit(slot, kept). So every request emits exactly its greedy tokens alone, whatever the windows were.
//
// The drafter runs on one rank (the planner, rank 0): one DeviceDrafter, one drafting context per slot (its rings,
// position and draft vocabulary; DeviceDrafter::select), fed by BatchForward's captured inputs of the last three
// layers for every kept row (prompt rows included). The plan (rows) and the commits are what the other rank needs:
// it steps the same rows and commits the same rows (the serving control channel already carries the rows).
//
// Which requests draft and how many rows each verifies: BatchWindowController (two stages, measured costs) or a fixed
// window (`fixed_rows`). Prompt rows come before drafts: decoding requests always get their own row, the prompt
// chunks the rows left (oldest first, as plan_step), the windows what remains.
namespace strix::deepseek41 {

struct BatchSpecOptions {
    std::string vocabulary = "full";  // ds41::dspark::choose_vocabulary spec, chosen per request from its prompt
    std::uint32_t fixed_rows = 0;     // 0: BatchWindowController; 2..6: every decoding request with room verifies that many
    std::uint32_t prompt_rows = 8;    // prompt rows per request per step (plan_step)
    std::uint32_t max_rows = BatchForward::max_rows;
    // a window never crosses a multiple of `align` positions (0: no limit): the prefix cache's session snapshots are
    // taken when a request's position lands on a block boundary
    std::uint32_t align = 0;
    // Drafts on both ranks (world 2): a request's drafting context lives on rank slot % 2, and every rank drafts its
    // own requests in parallel. Rank 0 plans; a draft round per step (512-byte exchanges through `exchange`, the same
    // size on both ranks) sends rank 1 the commits of the last step and the requests to draft, and brings their drafts
    // back. Rank 1 runs follow_round() before every step. world 1: rank 0 drafts every request.
    std::uint32_t rank = 0, world = 1;
    std::function<void(std::span<const std::byte>, std::span<std::byte>)> exchange;
    // world 2: a request start()ed from its prompt drafts on rank slot % 2 (both ranks call start with the prompt);
    // false: on rank 0 (the server's rank 1 does not see a prompt read in rows before it is read). imported()
    // requests always go to rank slot % 2.
    bool split_started = true;
    // world 2: a request drafting alone in a step is drafted by both ranks (DeviceDrafter::set_rank); both ranks must
    // agree (the server compares it in its hello; STRIX_DS41_DSPARK_COOP=0 turns it off)
    bool coop = true;
    // Prompt lookup (rank 0, the planner): a request may draft its window from an earlier occurrence of its context
    // (prompt + committed tokens, strix/lookup/index.hpp) instead of DSpark, or when DSpark does not draft; chosen per
    // request from measured survival (no draft time). Exact: verification decides every token. STRIX_DS41_LOOKUP=0
    // turns it off; rank 1 needs no setting (the rows carry the tokens).
    bool lookup = true;
    // The wall time of the last target step for the cost model (empty: target.last_stats().seconds). A split decode
    // (ds41_hetero_stage.hpp) passes the step it waited for: the FRONT's boundaries, the forward and the TAIL step.
    std::function<double()> step_seconds;
    // The cost calibration's default (STRIX_DS41_DSPARK_CALIB=0/1 overrides it): the server turns it on with its
    // warm-up (the qualification measured the two together); off: the old estimator.
    bool calib = false;
};

struct BatchSpecStats {
    std::uint64_t steps = 0, drafted_steps = 0, windows = 0, plain_rows = 0, drafts_verified = 0, drafts_accepted = 0, rows = 0;
    std::uint64_t decode_steps = 0, decode_tokens = 0;  // steps without prompt rows and the tokens they emitted
    std::uint64_t probe_rows = 0;  // calib: extra rows verified as price-independent probes (every 32nd drafted plan, at most one)
    std::uint64_t coop_drafts = 0;  // drafts both ranks ran together (one head half each)
    std::uint64_t lookup_windows = 0, lookup_verified = 0, lookup_accepted = 0;  // windows drafted by prompt lookup
    double lookup_seconds = 0.0;  // host time of the lookup indexes, queries and scoring
    double draft_seconds = 0.0, step_seconds = 0.0, finish_seconds = 0.0, plan_seconds = 0.0;
    double decode_seconds = 0.0;  // wall of the decode-only steps (plan + drafts + step + finish)
    std::array<std::uint64_t, ds41::dspark::block_size + 2U> window_rows{};  // windows by rows (1 = plain row)
    std::array<std::uint64_t, ds41::dspark::block_size> reached{}, hit{};    // per depth: verified after j correct, correct
};

class BatchSpeculator {
public:
    // `target` must capture the main hidden rows (BatchForwardOptions::capture_main_hidden); `drafter` gets one
    // context per slot (set_sequences(slots)).
    BatchSpeculator(BatchForward& target, ds41::dspark::DeviceDrafter& drafter, std::uint32_t slots, BatchSpecOptions options);
    ~BatchSpeculator();
    BatchSpeculator(const BatchSpeculator&) = delete;
    BatchSpeculator& operator=(const BatchSpeculator&) = delete;

    // A new request in `slot` reading `prompt` from position 0: its drafting context and its controller history are
    // reset, its draft vocabulary chosen from the prompt.
    void start(std::uint32_t slot, std::span<const std::uint32_t> prompt);
    // A request whose prompt another path read (the device prefill, an imported state): its drafting context is reset
    // and starts at `position` (DeviceDrafter::skip_to; the drafter sees the rows from there on), the prompt and the
    // first token `first` in its draft vocabulary.
    void imported(std::uint32_t slot, std::span<const std::uint32_t> prompt, std::uint32_t position, std::uint32_t first);
    // The same with the seed rows the prefill kept (the input streams of the last three layers of the prompt's last
    // `rows` rows, [rows][hc][hidden] each, device memory ordered on `stream`): the context starts at position - rows
    // with them, as if the drafter had seen those rows (DevicePrefill::draft_seed).
    void imported(std::uint32_t slot, std::span<const std::uint32_t> prompt, std::uint32_t position, std::uint32_t first,
                  std::array<const float*, 3> seed, std::uint32_t rows, void* stream);

    // The next step of `seqs` (each with `remaining[i]` tokens it may still emit; a window never emits more): the
    // drafter runs here for the requests that draft. Returns the rows to step (owner: index into seqs).
    const PlannedStep& plan(std::span<BatchSequence* const> seqs, std::span<const std::uint32_t> remaining);
    // After target.step(plan().rows) returned `tokens`: verifies every window, commits, appends every kept row to its
    // drafting context and advances the sequences (position, fed, generated). Returns the commits that dropped rows
    // (slot, kept) -- what the other rank must apply before its next step.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> finish(std::span<BatchSequence* const> seqs, std::span<const std::uint32_t> tokens);

    // world 2. A draft round goes out only in steps where rank 1 drafts; it carries the commits made before it. The other
    // commits travel in the caller's own per-step message (take_commits on rank 0, follow_commit on rank 1).
    // rank 0: the commits not sent in a round yet (slot << 16 | kept); the caller sends them before the next step.
    [[nodiscard]] std::vector<std::uint32_t> take_commits();
    // rank 1: a commit of rank 0's last step, then follow_stage before the next step (or a round does it): the kept rows
    // of the last step (its rows and tokens, `step` its count; once per step) go to this rank's drafting contexts.
    void follow_commit(std::uint32_t slot, std::uint32_t kept);
    void follow_stage(std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens, std::uint64_t step);
    // rank 1: a round (receives it, applies its commits, stages, drafts the requests, sends the drafts back); false when
    // rank 0 closed the rounds. follow_packet: the same for a round already received (round_bytes, magic round_magic)
    // by a loop that also receives other 512-byte messages; the commits it applied are appended to `commits`.
    bool follow_round(std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens, std::uint64_t step);
    bool follow_packet(std::span<const std::byte> packet, std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens,
                       std::uint64_t step, std::vector<std::pair<std::uint32_t, std::uint32_t>>* commits = nullptr);
    // Several plans in flight (micro-batches of a split decode): park() takes the plan of the last plan() out of the
    // speculator, resume() puts one back; finish() then verifies that plan (its step must be the target's last step).
    // plan() A, park A, ..., resume A, step A, finish A: groups of requests alternate without mixing their windows.
    struct ParkedPlan;
    [[nodiscard]] std::shared_ptr<ParkedPlan> park();
    void resume(const std::shared_ptr<ParkedPlan>& plan);
    // world 2, rank 1: `slot` was reset (a new request comes): its context is no longer this rank's.
    void release(std::uint32_t slot) { owners_.at(slot) = 0U, in_step_.at(slot) = false; }
    // both ranks: `slot`'s target state was replaced from outside (an imported state): its drafting contexts no longer
    // follow the same rows on both ranks, no cooperative draft until a new imported() request
    // world 2, rank 0: the slots whose window in the current plan came from prompt lookup (bit s); the server sends it
    // with the step. Rank 1: that mask for the step about to run, so follow_stage keeps those requests' cooperative
    // state (a lookup window flushes no drafter on either rank: the contexts stay identical).
    [[nodiscard]] std::uint32_t lookup_window_slots() const noexcept { return lookup_mask_; }
    void follow_lookup_windows(std::uint32_t mask) noexcept { lookup_mask_follow_ = mask; }
    void forget(std::uint32_t slot) {
        in_step_.at(slot) = false, coop_pending_.at(slot) = false;
        if (slot < lookup_slots_.size()) lookup_slots_[slot] = LookupSlot{};  // no lookup until a new request
    }
    static constexpr std::uint32_t round_magic = 0x44524631U;
    static constexpr std::size_t round_bytes = 512;
    [[nodiscard]] std::uint32_t owner(std::uint32_t slot) const noexcept { return owners_[slot]; }

    // Device bytes of the kept-row staging the constructor allocates for `slots` (the drafter's own: DeviceDrafter::footprint).
    [[nodiscard]] static std::uint64_t staging_bytes(std::uint32_t slots) noexcept;
    // false: the next plan()s draft nothing (every decoding request steps one plain row; the drafting contexts still
    // get every kept row, so drafting can resume). The tokens are those of plain steps either way.
    void allow_drafts(bool on) noexcept { allow_drafts_ = on; }

    [[nodiscard]] const BatchSpecStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const ds41::dspark::BatchWindowController& controller() const noexcept { return controller_; }

private:
    struct Slots;
    float* staging(std::uint32_t slot) const;  // the slot's next free staging row
    void stage(std::uint32_t slot, std::uint32_t first, std::uint32_t count);
    void stage_from(std::uint32_t slot, std::array<const float*, 3> streams, std::uint32_t first, std::uint32_t count, void* stream);
    void flush(std::uint32_t slot);           // staged rows and pending context tokens -> the slot's drafting context
    // the owner stages a request's kept rows; with coop_ the other rank too while in_step_ (the same rows flushed at the
    // same steps). Both ranks end in_step_ on a step's group of rows longer than one that was not a cooperative draft
    // (the same rule on the same rows), rank 1 also when it drafts the request alone, rank 0 on any draft of one rank.
    [[nodiscard]] bool keeps(std::uint32_t slot) const noexcept { return owners_[slot] == options_.rank || in_step_[slot]; }
    // world 2: one draft on both ranks, each scanning half the head (DeviceDrafter::set_rank): the same tokens
    ds41::dspark::Draft coop_draft(std::uint32_t slot, std::uint32_t token);

    BatchForward& target_;
    ds41::dspark::DeviceDrafter& drafter_;
    BatchSpecOptions options_;
    ds41::dspark::BatchWindowController controller_;
    std::unique_ptr<Slots> slots_;
    PlannedStep plan_;
    std::vector<ds41::dspark::Draft> drafts_;  // per owner (index into seqs): its draft
    std::vector<std::uint32_t> window_rows_;   // per owner: rows of its window (0: prompt rows or none)
    std::vector<bool> drafted_;                // per owner: the drafter ran
    std::uint32_t planned_drafting_ = 0, planned_decoders_ = 0, planned_total_ = 0;
    bool planned_prompt_rows_ = false;
    bool allow_drafts_ = true;
    double plan_seconds_ = 0.0;
    // measured costs (seconds, rank 0): a decode step of n requests without drafts (step + finish), a window row,
    // one request's draft (flush of its staged rows + the drafter call)
    std::array<double, ds41::dspark::BatchWindowController::max_requests + 1U> plain_mean_{};
    std::array<std::uint32_t, ds41::dspark::BatchWindowController::max_requests + 1U> plain_n_{};
    double row_mean_ = 0.0, draft_mean_ = 0.0, round_mean_ = 0.0;
    std::uint32_t row_n_ = 0, draft_n_ = 0, round_n_ = 0;
    // STRIX_DS41_DSPARK_CALIB (default BatchSpecOptions::calib; =0: the estimates above alone, =1: on): the step costs from EVERY decode step
    // (design notes). The row cost is a least-squares slope over pairs of steps of the same requests at most 2 steps
    // apart (their contexts differ by a few positions, so the drift cancels), with the default as a prior and capped at
    // half a one-request step. plain[n] is a fast mean of wall - row * x per request set (an LRU table), plus the
    // measured per-step loop overhead. First use (the first step of each row count, the drafter's first call, the first
    // split round) is not a sample; a step with prompt rows is not a sample and breaks the pairs.
    bool calib_ = false, cost_log_ = false, seen_draft_ = false, seen_round_ = false;
    std::array<bool, 64> seen_rows_{};  // a decode step of this many rows ran
    std::array<std::uint8_t, ds41::dspark::BatchWindowController::max_requests> epoch_{};  // per slot: requests started
    std::uint64_t planned_key_ = 0;  // the step's decoding requests (slot, epoch)
    // per request set (key; 8 entries, least recently used replaced): its plain wall estimate and its last sample (a
    // pair needs the same requests in steps at most 2 apart: two interleaved groups still pair, a stale sample does not)
    struct CostGroup {
        std::uint64_t key = 0, used = 0, prev_step = 0;
        std::uint32_t n = 0, base_n = 0, prev_x = 0;
        double base = 0.0, prev_wall = 0.0;
        bool prev_ok = false;
    };
    std::array<CostGroup, 8> groups_{};
    std::uint64_t steps_ = 0;  // finish() calls (every step)
    // the most recently used group with n requests (nullptr: none); plain[n] and stage 1 read it
    [[nodiscard]] const CostGroup* cost_group(std::uint32_t n) const noexcept;
    // the row slope: decayed sums of dx * dy and dx^2 over the pairs (a least-squares slope through the origin, so a pair
    // whose rows differ more weighs more), seeded with the controller's default as 4 pseudo-pairs of dx = 1
    double pair_xy_ = 0.0, pair_xx_ = 0.0;
    std::uint32_t pairs_ = 0;
    // the per-step cost outside the step wall that every plan pays alike: the loop's time between a decode step's
    // finish and the next plan (control exchange, publishing, admission) plus plan() without its drafts. A sample is
    // dropped (not clipped) when larger than 3x the estimate (before 4 samples: larger than the step's own wall): a
    // prefill or an idle period between steps is not a per-step cost; after 8 drops in a row samples count again. Added
    // to every plain[n] so the controller prices extra rows against the whole step cycle.
    double overhead_ = 0.0;
    std::uint32_t overhead_n_ = 0, overhead_drops_ = 0;  // drops in a row: after 8 the regime changed, samples count again
    // the price-independent probe: every 32 decode plans one is due; the next drafted plan whose windows are all one row
    // verifies one extra row of a drafted request (exact verify; a pair with dx != 0 whatever the current price)
    std::uint64_t probe_plans_ = 0;
    bool probe_due_ = false, planned_probe_ = false;
    double base1_last_ = 0.0;  // the last 1-request base (the prior's seed survives the LRU)
    std::array<bool, ds41::dspark::BatchWindowController::max_requests + 1U> measured_n_{};  // stage 1: a sampled step of n ran
    double planned_gap_ = -1.0, planned_overhead_ = 0.0;  // the current plan's gap before it (-1: none) and plan time without drafts
    bool last_finish_ok_ = false;  // the last finish() was a decode-step sample and no request started since: the gap counts
    std::chrono::steady_clock::time_point last_finish_end_{};
    std::vector<std::uint32_t> pending_commits_;  // world 2: slot << 16 | kept, for the next draft round
    // world 2, rank 1: what it knows of each slot (prompt, position) to stage the kept rows of its own requests
    std::vector<std::vector<std::uint32_t>> prompts_;
    std::vector<std::uint32_t> positions_;
    std::vector<std::uint32_t> owners_;  // the rank that drafts each slot's request
    // world 2, options_.coop: when one request drafts in a step, both ranks draft it together (half the output head
    // each). Exact: the tokens equal the one-rank draft. in_step_[slot]: the slot's context got the same rows in the
    // same flushes on both ranks since imported() (a draft on one rank only ends it: appends of other row counts may
    // round differently); rank 0 drafts cooperatively only then.
    bool coop_ = false;
    std::vector<bool> in_step_;
    std::vector<bool> coop_pending_;  // drafted by both ranks, its rows not staged yet (plans may queue)
    double coop_mean_ = 0.0;         // a cooperative draft with its entry exchange (rank 0)
    std::uint32_t coop_n_ = 0;
    bool seen_coop_ = false;
    std::array<std::uint32_t, ds41::dspark::BatchWindowController::max_requests> kept_{};  // rank 1: last step's commits (all ones: none)
    std::uint64_t staged_step_ = ~std::uint64_t{0};
    BatchSpecStats stats_;

    // prompt lookup (rank 0, options_.lookup), per request (slot) as the Nemotron selector: an index over its prompt
    // (indexed a chunk per plan) and committed tokens; every plan's candidate is shadow-scored token by token against
    // what the request commits (per match-length bucket and depth, decayed); a window drafts from lookup only where its
    // estimated rate (posterior-mean survival; tokens per priced step time, rows only) beats the mean rate of the request's alternative
    // (its DSpark windows, once measured, or a plain row) by STRIX_DS41_LOOKUP_MARGIN
    static constexpr std::uint32_t lookup_buckets = 5;
    struct Moments {  // decayed tokens and rows per window (priced at the current costs when compared)
        double n = 0.0, sum = 0.0, rows = 0.0;
        void add(double tokens, double window_rows);
        [[nodiscard]] double mean_tokens() const { return n > 0.0 ? sum / n : 0.0; }
    };
    struct LookupSlot {
        std::unique_ptr<strix::lookup::Index> index;  // null: no lookup for this request
        std::vector<std::uint32_t> backlog;           // tokens to index before it is current (the prompt, then commits)
        std::size_t backlog_at = 0;
        std::uint64_t next_position = 0;              // stream position of the next committed token
        struct Pending {
            std::uint64_t first = 0;
            std::array<std::uint32_t, ds41::dspark::block_size> token{};
            std::uint32_t count = 0, done = 0, bucket = 0;
        };
        std::vector<Pending> pending;
        std::array<std::array<double, ds41::dspark::block_size>, lookup_buckets> eligible{}, accepted{};
        std::array<double, lookup_buckets> evidence{};
        Moments dspark;  // the request's DSpark windows
    };
    bool lookup_on_ = false;
    double lookup_margin_ = 1.0;
    std::uint32_t lookup_capacity_ = 1U << 16;
    std::vector<LookupSlot> lookup_slots_;
    std::vector<bool> lookup_;  // per owner (index into seqs): its window came from prompt lookup
    std::vector<std::uint32_t> lookup_cap_;  // per owner: the drafts of its lookup window (executed as chosen)
    std::uint32_t lookup_mask_ = 0, lookup_mask_follow_ = 0;  // see lookup_window_slots()
    // STRIX_DS41_LOOKUP_LOG=1 (diagnostic, rank 0): per decoding request and step, the lookup candidate, its predicted
    // rate against the alternative and what ran (stderr "dspark lookup:" lines), for calibration and agreement studies
    struct LookupTrace {
        bool candidate = false;
        std::uint32_t length = 0, bucket = 0, available = 0, cap = 0;
        double rate = 0.0, alternative = 0.0;
        std::array<std::uint32_t, ds41::dspark::block_size> token{};
    };
    bool lookup_log_ = false;
    std::vector<LookupTrace> lookup_trace_;  // per owner
    void lookup_reset(std::uint32_t slot, std::span<const std::uint32_t> prompt);
    void lookup_commit(std::uint32_t slot, std::uint32_t token, std::uint64_t position);
    void lookup_catch_up(std::uint32_t slot);
};

}  // namespace strix::deepseek41
