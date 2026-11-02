// Prompt-lookup hybrid draft session (see lookup.hpp). Exact: proposals only; greedy verification decides every token.
#include "lookup.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::gemma4::engine::lookup {
namespace {
double env_value(const char* name, double fallback, double low, double high) {
    const auto* v = std::getenv(name);
    if (!v || !*v) return fallback;
    std::size_t used = 0;
    double x = 0;
    try { x = std::stod(v, &used); } catch (...) { used = 0; }
    if (used != std::string(v).size() || !(x >= low && x <= high))
        throw std::invalid_argument(std::string(name) + " outside [" + std::to_string(low) + "," + std::to_string(high) + "]");
    return x;
}
// Verify cost of R = 1..8 rows relative to R1 (release bench, 64-token context: 17.89 ms + 0 / 1.08 / 3.41 / 5.74 /
// 8.24 / 10.74 / 13.24 / 15.74 ms). Scaled by the stream's measured base cycle (cycle minus rows minus drafting), so
// the row term grows with the context-dependent pass cost.
constexpr std::array<double, 8> extra_rows{0.0, 1.08 / 17.89, 3.41 / 17.89, 5.74 / 17.89, 8.24 / 17.89, 10.74 / 17.89,
                                           13.24 / 17.89, 15.74 / 17.89};
constexpr double decay = 0.98;
constexpr std::size_t buckets = 5;
std::size_t bucket_of(std::uint32_t L) { return L < 2 ? 0 : L < 4 ? 1 : L < 8 ? 2 : L < 16 ? 3 : 4; }

struct Moments {
    double n = 0, sum = 0, square = 0, ms = 0;
    void add(double tokens, double cost) {
        n = n * decay + 1; sum = sum * decay + tokens; square = square * decay + tokens * tokens; ms = ms * decay + cost;
    }
    [[nodiscard]] double upper_rate() const {
        if (n <= 0) return 0;
        const double mean = sum / n, var = std::max(0.0, square / n - mean * mean);
        return (mean + std::sqrt(var / n)) / (ms / n);
    }
    [[nodiscard]] double mean_ms() const { return n > 0 ? ms / n : 0; }
};

class HybridSession final : public DraftSession {
public:
    HybridSession(std::unique_ptr<DraftSession> inner, const LookupConfig& config)
        : inner_(std::move(inner)), config_(config), index_(make_ngram_index(config.index)) {}

    std::array<std::uint32_t, 7> propose(std::uint32_t anchor, std::uint64_t position) override {
        return propose_window(anchor, position, max_draft);
    }
    // Forced windows / multi-stream / AutoPolicy path: the inner drafter unchanged; lookup only without one.
    std::array<std::uint32_t, 7> propose_window(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) override {
        close_cycle();
        if (inner_) return inner_->propose_window(anchor, position, k);
        std::array<std::uint32_t, 7> out{};
        if (synced(anchor, position)) {
            const auto m = index_->longest({}, config_.min_length, std::min(k, max_draft));
            for (std::uint32_t i = 0; i < m.available; ++i) out[i] = m.next[i];
            remember(m, position);
        }
        return out;
    }
    void attach(StreamState& state) override { if (inner_) inner_->attach(state); }
    [[nodiscard]] bool wants_hidden() const override { return inner_ && inner_->wants_hidden(); }
    // Lookup reads committed tokens only, never target taps: forward the inner drafter's need (MTP: none).
    [[nodiscard]] bool wants_taps() const override { return inner_ && inner_->wants_taps(); }
    [[nodiscard]] bool multi_stream_candidate() const override { return inner_ && inner_->multi_stream_candidate(); }
    [[nodiscard]] bool confidence_capable() const override { return !inner_ || inner_->confidence_capable(); }
    [[nodiscard]] ConfidenceConfig resolve_confidence(const ConfidenceConfig& config) const override {
        return inner_ ? inner_->resolve_confidence(config) : config;
    }
    Confident propose_confident(std::uint32_t anchor, std::uint64_t position, std::uint32_t kmax,
                                const ConfidenceConfig& config) override {
        close_cycle();
        cycle_start_ = Clock::now();
        Match m;
        if (synced(anchor, position)) m = index_->longest({}, config_.min_length, std::min(kmax, max_draft));
        const std::uint32_t cap = choose(m, kmax);
        remember(m, position); // after choose(): the candidate is scored against what this cycle commits
        Confident out;
        if (cap) {
            for (std::uint32_t i = 0; i < cap; ++i) out.tokens[i] = m.next[i];
            out.window = out.drafted = cap;
            kind_ = Kind::lookup; window_ = cap;
            ++lookup_cycles_; lookup_proposed_ += cap;
        } else if (inner_) {
            const auto t = Clock::now();
            out = inner_->propose_confident(anchor, position, kmax, config);
            draft_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - t).count();
            kind_ = Kind::inner; window_ = out.window;
        } else {
            kind_ = Kind::plain; window_ = 0;
        }
        return out;
    }
    void stage_rows(std::span<const float> taps, std::span<const float> hidden, std::span<const std::uint32_t> next_tokens,
                    std::span<const std::uint64_t> positions) override {
        staged_tokens_.assign(next_tokens.begin(), next_tokens.end());
        staged_positions_.assign(positions.begin(), positions.end());
        if (inner_) inner_->stage_rows(taps, hidden, next_tokens, positions);
    }
    void stage(std::span<const float> taps, std::span<const std::uint64_t> positions) override {
        if (inner_) inner_->stage(taps, positions);
    }
    // The index then holds the tokens at positions 1..F, as publishing the prefix rows would have fed them.
    bool seed(std::span<const std::uint32_t> tokens, std::span<const float> hidden) override {
        if (inner_ && !inner_->seed(tokens, hidden)) return false;
        for (std::size_t p = 1; p < tokens.size(); ++p) feed(tokens[p], p);
        return true;
    }
    void append(std::span<const float> taps, std::span<const std::uint64_t> positions) override {
        if (inner_) inner_->append(taps, positions);
    }
    void publish() override {
        for (std::size_t i = 0; i < staged_tokens_.size(); ++i) feed(staged_tokens_[i], staged_positions_[i] + 1);
        if (kind_ != Kind::none) emitted_ = std::uint32_t(staged_tokens_.size());
        staged_tokens_.clear(); staged_positions_.clear();
        if (inner_) inner_->publish();
    }
    void discard() noexcept override {
        staged_tokens_.clear(); staged_positions_.clear();
        if (inner_) inner_->discard();
    }
    void report(strix::serve::Json& receipt) const override {
        if (inner_) inner_->report(receipt);
        auto j = config_.json();
        j.set("cycles", lookup_cycles_); j.set("proposed", lookup_proposed_); j.set("accepted", lookup_accepted_);
        j.set("index_tokens", index_->tokens()); j.set("index_bytes", std::uint64_t(index_->bytes()));
        j.set("shadow_scored", scored_);
        auto per = strix::serve::Json::array();
        for (std::size_t b = 0; b < buckets; ++b) {
            auto row = strix::serve::Json::array();
            for (std::size_t d = 0; d < max_draft; ++d) row.push(eligible_[b][d] > 0 ? accepted_[b][d] / eligible_[b][d] : 0.0);
            per.push(row);
        }
        j.set("bucket_survival", per); // buckets: match length 1, 2-3, 4-7, 8-15, 16+
        receipt.set("prompt_lookup", j);
    }

private:
    using Clock = std::chrono::steady_clock;
    enum class Kind { none, inner, lookup, plain };
    struct Pending {
        std::uint64_t first = 0;
        std::array<std::uint32_t, max_draft> tokens{};
        std::uint32_t n = 0, done = 0;
        std::size_t bucket = 0;
    };

    // The index holds committed tokens up to the anchor (position p): its newest token is the anchor itself.
    [[nodiscard]] bool synced(std::uint32_t anchor, std::uint64_t position) const {
        return have_last_ && last_position_ == position && last_token_ == anchor;
    }
    // Best lookup cap by lower-bound tokens/ms against the alternative's upper bound; 0 = do not use lookup.
    [[nodiscard]] std::uint32_t choose(const Match& m, std::uint32_t kmax) const {
        if (m.length < config_.min_length || !m.available) return 0;
        const auto b = bucket_of(m.length);
        if (evidence_[b] < double(config_.evidence)) return 0;
        const double base = base_ms();
        const double alternative = alternative_rate(base);
        if (!(base > 0) || !(alternative > 0)) return 0;
        double best = alternative * config_.margin, survival = 1, yield = 1;
        std::uint32_t cap = 0;
        for (std::uint32_t k = 1; k <= m.available && k <= kmax; ++k) {
            // Beta(1,2) minus one posterior SD per depth (the AutoPolicy estimator)
            const double n = eligible_[b][k - 1] + 3, p = (accepted_[b][k - 1] + 1) / n;
            survival *= std::max(0.0, p - std::sqrt(p * (1 - p) / (n + 1)));
            yield += survival;
            const double rate = yield / (base * (1 + extra_rows[k]));
            if (rate > best) { best = rate; cap = k; }
        }
        return cap;
    }
    void feed(std::uint32_t token, std::uint64_t position) {
        if (have_last_ && position != last_position_ + 1) { index_ = make_ngram_index(config_.index); pending_.clear(); }
        index_->append(token);
        last_token_ = token; last_position_ = position; have_last_ = true;
        // Shadow scoring: each pending candidate whose next unscored position is this one (censor-aware).
        for (auto it = pending_.begin(); it != pending_.end();) {
            auto& c = *it;
            const auto d = c.done;
            if (c.first + d != position) { ++it; continue; }
            eligible_[c.bucket][d] += 1;
            const bool hit = c.tokens[d] == token;
            if (hit) accepted_[c.bucket][d] += 1;
            c.done = d + 1;
            if (!hit || c.done == c.n) { ++scored_; it = pending_.erase(it); } else ++it;
        }
    }
    void remember(const Match& m, std::uint64_t position) {
        if (m.length < config_.min_length || !m.available) return;
        const auto b = bucket_of(m.length);
        for (auto& x : eligible_[b]) x *= decay;
        for (auto& x : accepted_[b]) x *= decay;
        evidence_[b] = evidence_[b] * decay + 1;
        Pending c;
        c.first = position + 1; c.n = m.available; c.bucket = b;
        for (std::uint32_t i = 0; i < m.available; ++i) c.tokens[i] = m.next[i];
        if (pending_.size() < 64) pending_.push_back(c);
    }
    // At the next cycle start: the previous cycle's measured wall time and emitted tokens.
    void close_cycle() {
        if (kind_ == Kind::none) return;
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - cycle_start_).count();
        if (emitted_ && ms > 0) {
            if (kind_ == Kind::inner) {
                inner_stats_.add(emitted_, ms);
                inner_rows_ = inner_rows_ * decay + window_; inner_draft_ = inner_draft_ * decay + draft_ms_; inner_n_ = inner_n_ * decay + 1;
            } else if (kind_ == Kind::plain) {
                plain_stats_.add(emitted_, ms);
            } else {
                lookup_accepted_ += emitted_ - 1;
                base_lookup_ = base_lookup_ * decay + ms / (1 + extra_rows[window_]); base_lookup_n_ = base_lookup_n_ * decay + 1;
            }
        }
        kind_ = Kind::none; emitted_ = 0; draft_ms_ = 0;
    }
    // Cycle time without draft rows and drafting: plain cycles, else inner cycles minus rows/drafting, else lookup.
    [[nodiscard]] double base_ms() const {
        if (plain_stats_.n > 0) return plain_stats_.mean_ms();
        if (inner_n_ > 0) {
            const double rows = std::min(7.0, inner_rows_ / inner_n_);
            const auto lo = std::size_t(std::floor(rows)), hi = std::min<std::size_t>(lo + 1, 7);
            const double extra = extra_rows[lo] + (rows - double(lo)) * (extra_rows[hi] - extra_rows[lo]);
            const double undrafted = inner_stats_.mean_ms() - inner_draft_ / inner_n_;
            return undrafted > 0 ? undrafted / (1 + extra) : 0;
        }
        if (base_lookup_n_ > 0) return base_lookup_ / base_lookup_n_;
        return 0;
    }
    // Upper-bound rate of the alternative to lookup (inner window, or a plain step); 0 = no evidence yet.
    [[nodiscard]] double alternative_rate(double base) const {
        if (inner_) return inner_stats_.n >= 3 ? inner_stats_.upper_rate() : 0;
        if (plain_stats_.n > 0) return plain_stats_.upper_rate();
        return base > 0 ? 1.0 / base : 0;
    }

    std::unique_ptr<DraftSession> inner_;
    LookupConfig config_;
    std::unique_ptr<Index> index_;
    std::vector<std::uint32_t> staged_tokens_;
    std::vector<std::uint64_t> staged_positions_;
    std::vector<Pending> pending_;
    std::array<std::array<double, max_draft>, buckets> eligible_{}, accepted_{};
    std::array<double, buckets> evidence_{};
    Moments inner_stats_, plain_stats_;
    double inner_rows_ = 0, inner_draft_ = 0, inner_n_ = 0, base_lookup_ = 0, base_lookup_n_ = 0, draft_ms_ = 0;
    Clock::time_point cycle_start_{};
    Kind kind_ = Kind::none;
    std::uint32_t window_ = 0, emitted_ = 0, last_token_ = 0;
    std::uint64_t last_position_ = 0;
    bool have_last_ = false;
    std::uint64_t lookup_cycles_ = 0, lookup_proposed_ = 0, lookup_accepted_ = 0, scored_ = 0;
};
} // namespace

LookupConfig LookupConfig::from_environment() {
    LookupConfig c;
    const auto* v = std::getenv("STRIX_G4_LOOKUP");
    const std::string mode = v ? v : "";
    if (mode == "0" || mode == "off") {
        c.enabled = false;
        return c;
    }
    if (!mode.empty() && mode != "auto" && mode != "1" && mode != "on")
        throw std::invalid_argument("STRIX_G4_LOOKUP must be auto, 1/on or 0/off");
    c.min_length = std::uint32_t(env_value("STRIX_G4_LOOKUP_MIN", c.min_length, 1, 16));
    c.margin = env_value("STRIX_G4_LOOKUP_MARGIN", c.margin, 1, 2);
    c.index.capacity = std::uint32_t(env_value("STRIX_G4_LOOKUP_TOKENS", c.index.capacity, 64, double(1U << 22)));
    return c;
}
strix::serve::Json LookupConfig::json() const {
    return strix::serve::Json(strix::serve::Json::Object{{"enabled", enabled}, {"min_length", std::uint64_t(min_length)},
        {"margin", margin}, {"evidence", std::uint64_t(evidence)}, {"capacity", std::uint64_t(index.capacity)}});
}
DraftFactory hybrid_factory(DraftFactory inner, LookupConfig config) {
    return [inner = std::move(inner), config]() -> std::unique_ptr<DraftSession> {
        return std::make_unique<HybridSession>(inner ? inner() : nullptr, config);
    };
}

} // namespace strix::gemma4::engine::lookup
