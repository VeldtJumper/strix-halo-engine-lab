#pragma once

// Shared speculative-drafting layer: context proposals + one arbiter + one stats line, for every engine.
//
// ContextProposer: an incremental index over one stream's emitted tokens (prompt + every emitted token, the target's
// correction token included as soon as it is emitted) proposing variable-length continuations (up to 64 tokens) from
// an earlier occurrence, with a copy cursor that follows a source run in O(1) while the stream keeps matching it.
//
// Arbiter: per step picks serial, lookup(k) or the engine's learned drafter(k) by expected committed tokens per ms:
//     E(k) = 1 + sum_{j<=k} prod_{i<=j} p_i,   score = E(k) / (V(R) + D(k) + overhead),   R = physical window >= k+1
// with p_i learned per (source, bucket, depth class) from verified outcomes (lookup: every step's proposal scored against the emitted tokens) and
// V(R) learned per window size from the engine's measured walls (seeded by the engine).
//
// Exactness: proposals are arbitrary ids; the engine verifies them with its serial arithmetic, commits the greedy-equal
// prefix + correction and rolls back the rejected rows' state. The layer only reads emitted tokens, so it needs no
// rollback, and no decision of it can change the emitted ids.
#include "strix/lookup/index.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix::spec {

inline constexpr std::uint32_t max_proposal = 64;  // longest lookup proposal (verify windows up to 65 rows)

enum class Source : std::uint8_t { serial = 0, lookup = 1, model = 2 };
inline constexpr std::size_t source_count = 3;
[[nodiscard]] const char* source_name(Source source);

struct Proposal {
    std::vector<std::uint32_t> tokens;  // empty: no proposal
    std::uint32_t match_length = 0;     // matched suffix length (resumed: run length followed by the cursor)
    std::uint64_t distance = 0;         // recency of the source occurrence
    bool resumed = false;               // continued from the copy cursor
    std::uint64_t run = 0;              // copy run id (a fresh source starts a new run; resumed proposals keep it)
    [[nodiscard]] std::uint32_t bucket() const;  // survival bucket (fresh match class, or resumed run class)
};
inline constexpr std::uint32_t lookup_buckets = 6;  // fresh: match 2-3, 4-7, 8-15, 16+; resumed: run < 16, >= 16

struct ProposerConfig {
    strix::lookup::IndexConfig index{1U << 17, 64};  // 128K tokens: prompt + output of every suite task
    std::uint32_t min_match = 2;                     // shortest fresh match proposed
};

class ContextProposer {
public:
    explicit ContextProposer(const ProposerConfig& config = {});
    // Emitted tokens, once each, in stream order (prompt first). Advances or drops the copy cursor.
    void commit(std::span<const std::uint32_t> tokens);
    // Continuation of stream + extra (extra = a hypothetical prefix, e.g. a model draft to extend), at most max_len.
    [[nodiscard]] Proposal propose(std::span<const std::uint32_t> extra, std::uint32_t max_len);
    [[nodiscard]] std::uint64_t tokens() const { return index_->tokens(); }
    [[nodiscard]] std::size_t bytes() const { return index_->bytes(); }

private:
    ProposerConfig config_;
    std::unique_ptr<strix::lookup::Index> index_;
    bool cursor_valid_ = false;
    std::uint64_t cursor_ = 0;  // source position whose successor is predicted to be the next emitted token
    std::uint32_t run_ = 0;     // tokens the cursor has followed (anchor match included)
    std::uint64_t run_id_ = 0;
};

// Survival per depth class: depths 1..8 each, then 9-12, 13-16, 17-24, 25-32, 33-48, 49-64 (one hazard per class).
inline constexpr std::size_t depth_classes = 14;
[[nodiscard]] std::size_t depth_class(std::uint32_t depth);
// Verify costs are kept per context bin: < 2K, 2K-8K, 8K-32K, 32K-128K, >= 128K positions.
inline constexpr std::size_t context_bins = 5;
[[nodiscard]] std::size_t context_bin(std::uint64_t positions);

struct ArbiterConfig {
    std::vector<std::uint32_t> windows{1, 2, 3, 4, 5, 6, 7, 8};  // physical verify windows (rows), ascending, 1 first
    double switch_in = 1.05;            // another drafting arm must beat the current one by this factor
    double half_life_tokens = 256.0;    // evidence decay per emitted token
    std::uint32_t probe_every = 32;     // passes: re-probe a barred / unused learned drafter (at its deepest k)
    std::uint32_t retreat_misses = 2;   // consecutive first-row misses that bar a source (lookup: until new evidence;
                                        // never a black-box drafter: its measured rate already prices its misses)
    double min_lookup_evidence = 4.0;   // observations a lookup bucket needs before lookup may run (fresh buckets:
                                        // depth 1; resumed copy buckets: one hazard per emitted position)
    std::uint32_t min_model_passes = 3; // the learned drafter runs this often before it is compared
    double cost_weight_cap = 8.0;       // verify-cost EWMA: a new wall weighs >= 1/(cap+1)
    double cost_aging = 0.995;          // per pass, every window's weight decays (stale windows re-learn fast)
    double lookup_overhead_ms = 0.0;    // extra cost of a lookup pass (the proposal itself is computed every step: sunk)
    std::array<double, 2> lookup_prior{1.0, 2.0};  // Beta prior (accept, reject) per depth for lookup
    std::array<double, 2> model_prior{2.0, 2.0};
    double lookup_sd = 1.0;  // lookup: cumulative survival bounded once per depth at mean - lookup_sd * SD
    double model_sd = 0.0;
};

struct StepInput {
    const Proposal* lookup = nullptr;             // null/empty: no lookup candidate this step
    std::uint32_t model_max_k = 0;                // 0: no learned drafter; every depth 1..model_max_k must be runnable
    std::span<const double> model_draft_ms;       // [k-1] = draft wall for k model tokens (required with a drafter)
    // A drafter that picks its own window (confidence stop): its measured committed tokens and complete ms per pass
    // (engine's upper-bound estimate); when ms > 0 the model arm is scored on this rate instead of per-depth survival.
    double model_pass_tokens = 0.0, model_pass_ms = 0.0;
    std::uint32_t remaining = ~0U;                // tokens left in the budget (k <= remaining - 1; 0: nothing to do)
    std::uint32_t max_rows = ~0U;                 // legal physical rows this step (context room, alignment, slots)
    std::uint64_t context = 0;                    // positions already in the context (cost bin)
};

struct Decision {
    Source source = Source::serial;
    std::uint32_t k = 0;      // draft tokens to verify (lookup: a prefix of the proposal)
    std::uint32_t rows = 1;   // physical window (>= k + 1, from ArbiterConfig::windows); 0: nothing to do
    bool padded = false;      // rows > k + 1: the engine masks the padding rows
    double expected_tokens = 1.0;
    double expected_ms = 0.0;
    bool probe = false;       // exploration (forced model measurement or re-probe), not the best estimate
};

struct Outcome {
    Source source = Source::serial;
    std::uint32_t bucket = 0;     // lookup: Proposal::bucket()
    std::uint32_t proposed = 0;   // k
    std::uint32_t accepted = 0;   // draft tokens equal to the target's greedy choice (prefix)
    std::uint32_t rows = 1;       // physical window verified
    double verify_ms = 0.0;       // complete host wall of the pass minus draft: verify + commit + exchange
    double draft_ms = 0.0;        // model draft wall (0 for lookup/serial)
    std::uint64_t context = 0;    // positions in the context when the pass started
};

class Arbiter {
public:
    explicit Arbiter(ArbiterConfig config = {});
    // Engine's tabled verify wall at `rows`: the shape of the cost curve; measured walls set its scale and replace
    // the seed where measured. context: the bin to seed; the default seeds every bin.
    void seed_cost(std::uint32_t rows, double ms, std::uint64_t context = ~0ULL);
    void seed_model_survival(std::span<const double> conditional_by_depth, double weight = 8.0);
    // The step's lookup proposal (if any) is kept and scored against the tokens emitted from now on, whatever runs,
    // one pending proposal per copy run (overlapping cursor proposals are not counted twice).
    [[nodiscard]] Decision decide(const StepInput& input);
    // After the pass: its outcome (costs, model survival) and the tokens it emitted (accepted drafts + correction).
    void observe(const Outcome& outcome, std::span<const std::uint32_t> emitted);
    // The stream restarted elsewhere (new request / non-contiguous position): drop unresolved proposals.
    void reset_pending() { pending_.clear(); }
    // Current estimate: interpolated between measured windows, never below a smaller measured window.
    [[nodiscard]] double verify_ms(std::uint32_t rows, std::uint64_t context = 0) const;
    // Posterior mean of the conditional survival at `depth` (decisions bound the cumulative survival, see .cpp).
    [[nodiscard]] double survival(Source source, std::uint32_t bucket, std::uint32_t depth) const;
    [[nodiscard]] const ArbiterConfig& config() const { return config_; }

private:
    struct Beta { double a = 0, b = 0; };
    using Table = std::array<Beta, depth_classes>;
    struct Pending { std::vector<std::uint32_t> tokens; std::uint32_t bucket = 0, matched = 0; std::uint64_t run = 0; };
    struct Costs { std::vector<double> ms, weight, seed; };  // [rows]
    [[nodiscard]] double known_ms(const Costs& c, std::uint32_t rows) const;  // measured, scaled seed, or -1
    void decay(double tokens);
    void learn(Table& table, std::uint32_t proposed, std::uint32_t accepted, double weight);
    [[nodiscard]] double expected(Source source, std::uint32_t bucket, std::uint32_t k) const;
    [[nodiscard]] std::uint32_t physical(std::uint32_t rows) const;
    ArbiterConfig config_;
    std::array<Table, lookup_buckets> lookup_{};
    std::array<double, lookup_buckets> lookup_evidence_{};
    Table model_{};
    std::vector<Pending> pending_;  // unresolved lookup proposals, oldest first (at most max_proposal)
    std::array<Costs, context_bins> cost_{};
    Source last_ = Source::serial;
    std::uint64_t passes_ = 0, last_model_ = 0, model_passes_ = 0, model_barred_until_ = 0;
    std::uint32_t model_misses_ = 0, lookup_misses_ = 0;
    bool last_black_box_ = false;  // the last model decision used the drafter's measured rate (no retreat for it)
    bool lookup_barred_ = false;
};

// One counters object per stream (or summed over a run) and THE line every engine prints at the end of a run:
//   strix-spec-stats {"passes":..,"emitted":..,"tokens_per_pass":..,"sources":{..},"depth_offered":[..],
//                     "depth_accepted":[..],"rows":{..},"draft_ms":..,"lookup_us":{..},"index_bytes":..}
struct SpecStats {
    std::uint64_t passes = 0, emitted = 0;
    std::array<std::uint64_t, source_count> source_passes{}, source_proposed{}, source_accepted{};
    std::array<std::uint64_t, max_proposal> depth_offered{}, depth_accepted{};  // [d-1]
    std::vector<std::uint64_t> rows_passes;   // [rows]
    std::vector<double> rows_ms;              // [rows] summed verify wall
    double draft_ms = 0;
    std::array<std::uint64_t, 32> lookup_us_hist{};  // [i]: proposal host time in [2^(i-3), 2^(i-2)) us
    double lookup_us_max = 0;
    std::size_t index_bytes = 0;
    void record(const Outcome& outcome, std::uint32_t emitted_this_pass);
    void lookup_time(double us);
    [[nodiscard]] double lookup_us_quantile(double q) const;  // upper bucket edge
    [[nodiscard]] std::string json() const;
    [[nodiscard]] std::string line() const { return "strix-spec-stats " + json(); }
};

} // namespace strix::spec
