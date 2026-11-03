#pragma once

#include "gemma4/dflash/dflash.hpp"
#include "strix/serve/json.hpp"

#include <array>
#include <span>

namespace strix::gemma4::engine {

// Exponentially decayed first/second moments; bound() = mean +- one standard error.
struct DecayedMoments {
    double n = 0, sum = 0, square = 0;
    void decay();
    void add(double x);
    [[nodiscard]] double bound(bool upper) const;
};

struct WindowDecision {
    std::uint32_t k = 0;
    bool probe = false;
    double expected = 1, cost_ms = 0, gain = 0;
};

// Stream-local, bounded evidence. No language, model or context-length timing
// is substituted for this stream's measurements. Probes are explicitly counted
// calibration work, distinct from windows admitted by the >=5% gain rule.
class AutoPolicy {
public:
    [[nodiscard]] WindowDecision choose(std::uint32_t max_k, std::uint64_t position) const;
    void observe(std::uint64_t position, std::uint32_t k, double total_ms,
        double draft_ms, double verify_ms, double context_ms,
        const dflash::Acceptance& delta, bool probe);
    [[nodiscard]] strix::serve::Json receipt() const;
    // Read-only views used by MultiStreamPolicy (>1 live stream), same estimators as choose(): survival lower
    // bound -> expected emitted tokens of a k-window; fitted draft cost of a k-window (-1: no evidence).
    [[nodiscard]] double expected_tokens(std::uint64_t position, std::uint32_t k) const;
    // Decayed per-depth survival evidence of one length bin, and the expected emitted tokens of a k-window from any
    // such evidence (Beta(1,2) minus one posterior SD per depth, exactly the estimator choose() uses).
    struct Survival { std::array<double,7> eligible{}, accepted{}; };
    [[nodiscard]] Survival survival(std::uint64_t position) const;
    [[nodiscard]] static double expected_tokens(const Survival& evidence, std::uint32_t k);
    [[nodiscard]] double draft_cost(std::uint64_t position, std::uint32_t k) const;
    // A drafted window verified inside a multi-stream batch: acceptance (a token property, batch-independent) and
    // this stream's own draft time feed survival and the draft-cost fit. Batch timing is NOT stream evidence (no
    // C1 cycle/verify/plain moment changes; MultiStreamPolicy owns it). Never called for C1 cycles.
    void observe_multi(std::uint64_t position, std::uint32_t k, double draft_ms, const dflash::Acceptance& delta);
private:
    using Moments = DecayedMoments;
    struct Bin {
        std::array<Moments,8> cycle, verify;
        Moments plain, draft, context;
        // Draft cost as a function of depth: decayed least-squares fit draft_ms ~ fixed + per_depth*k over drafting
        // cycles (DFlash: fixed per cycle; MTP: ~0.56 ms per depth measured).
        double fit_n = 0, fit_k = 0, fit_kk = 0, fit_d = 0, fit_kd = 0;
        std::array<double,7> eligible{}, accepted{};
        std::uint64_t cycles = 0, probes = 0;
        std::uint32_t pending_probe = 0;
        std::array<std::uint64_t,8> probe_windows{};
    };
    std::array<Bin,5> bins_{};
    static std::size_t bucket(std::uint64_t position);
    static double expected(const Bin& bin, std::uint32_t k);
    // Fitted draft cost of a k-window; -1 without evidence. Used only to raise (never lower) a window's cost.
    static double draft_estimate(const Bin& bin, std::uint32_t k);
    static void learn(Bin& bin, std::uint32_t k, double draft_ms, const dflash::Acceptance& delta);
};

// multi-stream drafting. All tuning knobs live here; env overrides are read once per
// Coordinator. Default cap k=1: one-node C4 EN forced-k aggregate OFF 131.0 / k1 144.0 / k2 127.4 / k3 113.3 tok/s.
struct MultiStreamConfig {
    bool enabled = true;           // STRIX_G4_DRAFT_MULTI=0 -> the previous behaviour: auto stays k=0 whenever >1 stream is live
    std::uint32_t max_k = 1;       // STRIX_G4_DRAFT_MULTI_K (integer 1..7): window cap while >1 stream is live
    double max_draft_share = 0.25; // STRIX_G4_DRAFT_MULTI_COST (0..1): a stream may draft only if its fitted k-window
                                   // draft cost <= share x (plain batch step / live streams): cheap per depth (MTP)
    double min_gain = 1.05;        // STRIX_G4_DRAFT_MULTI_GAIN (1..4): measured batch rate must beat plain by this
    [[nodiscard]] static MultiStreamConfig from_environment();
    [[nodiscard]] strix::serve::Json json() const;
};
// Drafter-confidence-gated C1 window (MTP; drafters that report a per-depth top-1 probability). Replaces the C1
// AutoPolicy window choice (no probes, no plain refresh) for confidence-capable sessions in auto mode. Exactness: the
// window only decides how many drafted tokens are verified; greedy verification decides every emitted token, so ids
// equal drafting off for any window. Env read once per Coordinator. DEFAULT ON: with STRIX_G4_MTP_CONF unset the
// session resolves tau (and max_k/first unless their env is set) from its drafter: drafter-mtp/config.json keys
// confidence_tau / confidence_max_k / confidence_first, else the builtin per-drafter default. "0"/"off" = OFF switch.
struct ConfidenceConfig {
    enum class Source : std::uint8_t { off, env, drafter_config, builtin };
    static constexpr double builtin_tau = 0.6; // unknown drafter (no config key, no builtin entry)
    double tau = 0;                // STRIX_G4_MTP_CONF (0..1]: cumulative top-1 probability threshold; 0 = disabled
    std::uint32_t max_k = 7;       // STRIX_G4_MTP_CONF_K (1..7): window cap (also capped by the request limit)
    std::uint32_t first = 2;       // STRIX_G4_MTP_CONF_FIRST (1..7): depths drafted before the first check
    std::uint32_t step = 1;        // STRIX_G4_MTP_CONF_STEP (1..7): depths drafted per extension
    double min1 = 0;               // STRIX_G4_MTP_CONF_MIN1 [0..1]: depth-1 probability below this verifies no draft
    double verify_tau = 0;         // > 0: verify the leading depths with P_d >= verify_tau instead of tau
    // Cost-aware long-context rule (STRIX_G4_MTP_CONF_LONG, default 0.04; 0/off = OFF switch): when this stream's
    // MEASURED draft cost per depth is at least long_ratio x its measured one-row pass, drafting depths is expensive
    // relative to verifying them, so the window starts at one depth, extends while P >= long_extend and verifies the
    // drafted prefix with P >= long_verify (at least one). Below the ratio the rule above is unchanged.
    double long_ratio = 0.04, long_extend = 0.6, long_verify = 0.15;
    [[nodiscard]] ConfidenceConfig long_context() const {
        auto c = *this;
        c.first = 1; c.step = 1; c.tau = long_extend; c.verify_tau = long_verify;
        return c;
    }
    Source source = Source::off;   // where tau came from (receipts)
    bool per_drafter = false;      // env unset: resolved() takes the drafter's values
    bool max_k_env = false, first_env = false;
    [[nodiscard]] bool enabled() const { return tau > 0; }
    // Session-side resolution when the env leaves the choice to the drafter: config-file values (> 0) win, else the
    // drafter's builtin tau; max_k / first only when their env is unset. Explicit env configs are returned unchanged.
    [[nodiscard]] ConfidenceConfig resolved(double config_tau, std::uint32_t config_max_k, std::uint32_t config_first,
                                            double drafter_builtin_tau) const {
        if (!per_drafter) return *this;
        auto c = *this;
        c.per_drafter = false;
        if (config_tau > 0) { c.tau = config_tau; c.source = Source::drafter_config; }
        else { c.tau = drafter_builtin_tau > 0 ? drafter_builtin_tau : builtin_tau; c.source = Source::builtin; }
        if (!max_k_env && config_max_k) c.max_k = config_max_k;
        if (!first_env && config_first) c.first = config_first;
        return c;
    }
    // P_d = prod_{i<=d} prob_i over the drafted depths (prob.size() = drafted). Next total depth to draft: the first
    // chunk when nothing is drafted, then +step while P at the last drafted depth >= tau and depth < kmax; returns
    // prob.size() when drafting stops. (Inline: drafter sessions use it without linking the engine.)
    [[nodiscard]] std::uint32_t next_depth(std::span<const float> prob, std::uint32_t kmax) const {
        const auto drafted = std::uint32_t(prob.size());
        kmax = kmax < 7 ? kmax : 7;
        if (!drafted) return first < kmax ? first : kmax;
        if (drafted >= kmax) return drafted;
        double cumulative = 1;
        for (const auto p : prob) cumulative *= double(p);
        return cumulative >= tau ? (drafted+step < kmax ? drafted+step : kmax) : drafted;
    }
    // Verified window: leading depths with P_d >= tau, at least 1 (0 when nothing is drafted or prob_1 < min1).
    [[nodiscard]] std::uint32_t window(std::span<const float> prob) const {
        if (prob.empty() || double(prob[0]) < min1) return 0;
        std::uint32_t k = 0;
        double cumulative = 1;
        const double threshold = verify_tau > 0 ? verify_tau : tau;
        for (const auto p : prob) {
            cumulative *= double(p);
            if (!(cumulative >= threshold)) break;
            ++k;
        }
        return k ? k : 1;
    }
    [[nodiscard]] static ConfidenceConfig from_environment();
    [[nodiscard]] static const char* source_name(Source source);
    [[nodiscard]] strix::serve::Json json() const;
};
// Measured draft-vs-pass cost of one stream's confidence cycles (decayed), the input of the long-context rule. The
// one-row pass is estimated from each cycle's (k+1)-row verify time with the release-bench row-cost shape. The ratio
// needs n >= 3, i.e. four observed cycles at decay 0.95.
struct DraftCostRatio {
    static constexpr double decay = 0.95;
    double draft_ms = 0, depths = 0, row_ms = 0, n = 0;
    void observe(double cycle_draft_ms, std::uint32_t drafted, double verify_ms, std::uint32_t k) {
        static constexpr std::array<double,8> rows{0.0,1.08,3.41,5.74,8.24,10.74,13.24,15.74};
        draft_ms = draft_ms*decay+cycle_draft_ms; depths = depths*decay+drafted;
        row_ms = row_ms*decay+verify_ms/(1+rows[k < 7 ? k : 7]/17.89);
        n = n*decay+1;
    }
    [[nodiscard]] double ratio() const { return n >= 3 && depths > 0 && row_ms > 0 ? (draft_ms/depths)/(row_ms/n) : 0; }
    [[nodiscard]] bool long_rule(const ConfidenceConfig& c) const { return c.long_ratio > 0 && ratio() >= c.long_ratio; }
};
// One live decode stream as seen by the batch planner, in slot order.
struct StreamCandidate {
    const AutoPolicy* policy = nullptr; // nullptr: never drafts at >1 stream (DFlash, no drafter, sampling, off)
    std::uint64_t position = 0;
    std::uint32_t limit = 0;            // min(draft_k, 7, budget-1); the 8-row capacity is applied by the planner
};
struct BatchPlan {
    std::array<std::uint32_t,8> k{};    // per candidate, never above MultiStreamConfig::max_k
    std::uint32_t extra = 0;            // total draft rows of the plan
    bool probe = false;                 // bounded calibration, not a gain admission
    double tokens = 0, cost_ms = 0, gain = 0;
};
// Whole-batch policy for steps where every live stream decodes (C = 2..8). Evidence is the measured step time
// (drafting + target + context + commit of all streams) per (C, extra draft rows); never extrapolated from C1 or
// across C. Admission: (sum of per-stream expected tokens) / step upper bound > min_gain x C / plain lower bound,
// so it declines k>0 whenever its measured estimate loses. Probes: at most six bootstrap probes per C, then
// two-sample refreshes every 16 (32 when admitted) steps, plus a plain refresh every 16th step - the C1 bounds.
class MultiStreamPolicy {
public:
    explicit MultiStreamPolicy(MultiStreamConfig config = MultiStreamConfig::from_environment());
    [[nodiscard]] const MultiStreamConfig& config() const { return config_; }
    [[nodiscard]] BatchPlan choose(std::span<const StreamCandidate> streams) const;
    void observe(std::size_t streams, std::uint32_t extra_rows, double step_ms, bool probe);
    [[nodiscard]] strix::serve::Json receipt() const;
private:
    struct Row {
        std::array<DecayedMoments,8> step;
        std::uint64_t cycles = 0, probes = 0;
        std::uint32_t pending_probe = 0;
    };
    [[nodiscard]] BatchPlan plan(std::span<const StreamCandidate> streams, double plain_ms, bool include_unmeasured) const;
    MultiStreamConfig config_;
    std::array<Row,8> rows_{};
};

} // namespace strix::gemma4::engine
