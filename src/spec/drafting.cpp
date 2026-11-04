#include "strix/spec/drafting.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace strix::spec {

const char* source_name(Source source) {
    switch (source) {
    case Source::serial: return "serial";
    case Source::lookup: return "lookup";
    case Source::model: return "model";
    }
    return "?";
}

std::uint32_t Proposal::bucket() const {
    // Resumed copy runs: two classes only (run < 16, >= 16), so a growing run keeps its evidence.
    if (resumed) return match_length < 16 ? 4U : 5U;
    return match_length < 4 ? 0 : match_length < 8 ? 1 : match_length < 16 ? 2 : 3;
}

std::size_t depth_class(std::uint32_t depth) {
    if (depth == 0) throw std::invalid_argument("draft depth starts at 1");
    if (depth <= 8) return depth - 1;
    return depth <= 12 ? 8 : depth <= 16 ? 9 : depth <= 24 ? 10 : depth <= 32 ? 11 : depth <= 48 ? 12 : 13;
}

namespace {
// A resumed copy run follows one source token by token: its hazard is per emitted position, not per draft depth,
// so resumed buckets pool every depth into one class and count each emitted position once.
bool pooled(std::uint32_t bucket) { return bucket >= 4; }
std::size_t lookup_class(std::uint32_t bucket, std::uint32_t depth) { return pooled(bucket) ? 0 : depth_class(depth); }
} // namespace

std::size_t context_bin(std::uint64_t positions) {
    return positions < 2048 ? 0 : positions < 8192 ? 1 : positions < 32768 ? 2 : positions < 131072 ? 3 : 4;
}

// ------------------------------------------------------------------------------------------------ ContextProposer
ContextProposer::ContextProposer(const ProposerConfig& config)
    : config_(config), index_(strix::lookup::make_ngram_index(config.index)) {
    if (config_.min_match == 0) throw std::invalid_argument("lookup min_match must be >= 1");
}

void ContextProposer::commit(std::span<const std::uint32_t> tokens) {
    for (const auto token : tokens) {
        if (cursor_valid_) {
            const auto next = cursor_ + 1;
            // The successor must be an older committed token (not the one being appended) still in the ring.
            if (next < index_->tokens() && next >= index_->oldest() && index_->at(next) == token) {
                cursor_ = next;
                ++run_;
            } else {
                cursor_valid_ = false;
                run_ = 0;
            }
        }
        index_->append(token);
    }
}

Proposal ContextProposer::propose(std::span<const std::uint32_t> extra, std::uint32_t max_len) {
    Proposal p;
    max_len = std::min(max_len, max_proposal);
    if (max_len == 0 || index_->tokens() < 2) return p;
    std::array<std::uint32_t, max_proposal> buffer{};
    const std::span<std::uint32_t> out(buffer.data(), max_len);
    if (cursor_valid_) {
        // The cursor predicts stream positions after the last committed token; extra must follow the same source.
        bool follows = true;
        for (std::size_t i = 0; i < extra.size() && follows; ++i) {
            const auto source = cursor_ + 1 + i;
            follows = source < index_->tokens() && index_->at(source) == extra[i];
        }
        if (follows) {
            const auto n = index_->continuation(cursor_ + extra.size(), extra, out);
            if (n > 0) {
                p.tokens.assign(buffer.begin(), buffer.begin() + std::ptrdiff_t(n));
                p.match_length = run_ + std::uint32_t(extra.size());
                p.distance = index_->tokens() - 1 - cursor_;
                p.resumed = true;
                p.run = run_id_;
                return p;
            }
        }
    }
    const auto m = index_->longest(extra, config_.min_match, strix::lookup::max_draft);
    if (m.length == 0) return p;
    const auto n = index_->continuation(m.source_end, extra, out);
    if (n == 0) return p;
    p.tokens.assign(buffer.begin(), buffer.begin() + std::ptrdiff_t(n));
    p.match_length = m.length;
    p.distance = m.distance;
    if (extra.empty()) {  // a fresh source for the committed stream: follow it from now on
        cursor_valid_ = true;
        cursor_ = m.source_end;
        run_ = m.length;
        p.run = ++run_id_;
    }
    return p;
}

// ------------------------------------------------------------------------------------------------ Arbiter
Arbiter::Arbiter(ArbiterConfig config) : config_(std::move(config)) {
    const auto& w = config_.windows;
    if (w.empty() || w.front() != 1 || !std::is_sorted(w.begin(), w.end()) ||
        std::adjacent_find(w.begin(), w.end()) != w.end() || w.back() > max_proposal + 1)
        throw std::invalid_argument("arbiter windows must be ascending, unique, start at 1 and be <= 65 rows");
    if (!(config_.switch_in >= 1.0) || !(config_.half_life_tokens > 0.0) || !(config_.cost_weight_cap >= 1.0) ||
        !(config_.cost_aging > 0.0 && config_.cost_aging <= 1.0))
        throw std::invalid_argument("arbiter: switch_in >= 1, half_life_tokens > 0, cost_weight_cap >= 1, aging in (0, 1]");
    for (auto& c : cost_) { c.ms.assign(w.back() + 1, 0.0); c.weight.assign(w.back() + 1, 0.0); c.seed.assign(w.back() + 1, 0.0); }
}

std::uint32_t Arbiter::physical(std::uint32_t rows) const {
    for (const auto w : config_.windows)
        if (w >= rows) return w;
    return 0;
}

void Arbiter::seed_cost(std::uint32_t rows, double ms, std::uint64_t context) {
    if (rows == 0 || rows >= cost_[0].ms.size() || !(ms > 0.0)) return;
    for (std::size_t b = 0; b < context_bins; ++b) {
        if (context != ~0ULL && b != context_bin(context)) continue;
        cost_[b].seed[rows] = ms;
    }
}

// Seeds give the shape of the cost curve, measurements its scale: an unmeasured seeded window costs its seed times
// measured/seed at the nearest window that has both (1 when none), so a seed table in another unit or from another
// context still prices rows correctly once one window is measured.
double Arbiter::known_ms(const Costs& c, std::uint32_t rows) const {
    if (c.weight[rows] > 0.0) return c.ms[rows];
    if (!(c.seed[rows] > 0.0)) return -1.0;
    double scale = 1.0;
    for (std::uint32_t d = 1; d < c.ms.size(); ++d) {
        const std::uint32_t lo = rows > d ? rows - d : 0, hi = rows + d;
        if (lo > 0 && c.weight[lo] > 0.0 && c.seed[lo] > 0.0) { scale = c.ms[lo] / c.seed[lo]; break; }
        if (hi < c.ms.size() && c.weight[hi] > 0.0 && c.seed[hi] > 0.0) { scale = c.ms[hi] / c.seed[hi]; break; }
        if (lo == 0 && hi >= c.ms.size()) break;
    }
    return c.seed[rows] * scale;
}

void Arbiter::seed_model_survival(std::span<const double> conditional, double weight) {
    for (std::size_t d = 1; d <= conditional.size() && d <= max_proposal; ++d) {
        const double p = std::clamp(conditional[d - 1], 0.0, 1.0);
        auto& beta = model_[depth_class(std::uint32_t(d))];
        beta.a += p * weight;
        beta.b += (1.0 - p) * weight;
    }
}

double Arbiter::verify_ms(std::uint32_t rows, std::uint64_t context) const {
    const auto& c = cost_[context_bin(context)];
    if (rows == 0 || rows >= c.ms.size()) return 0.0;
    double floor = 0.0;  // monotone: never below a known smaller window
    std::uint32_t lo = 0, lo2 = 0, hi = 0;
    double vlo = 0, vlo2 = 0, vhi = 0;
    for (std::uint32_t r = 1; r < rows; ++r)
        if (const double v = known_ms(c, r); v > 0.0) { floor = std::max(floor, v); lo2 = lo; vlo2 = vlo; lo = r; vlo = v; }
    if (const double v = known_ms(c, rows); v > 0.0) return std::max(v, floor);
    for (std::uint32_t r = rows + 1; r < c.ms.size(); ++r)
        if (const double v = known_ms(c, r); v > 0.0) { hi = r; vhi = v; break; }
    double est;
    if (lo && hi) est = vlo + (vhi - vlo) * double(rows - lo) / double(hi - lo);
    else if (lo && lo2) est = vlo + std::max(0.0, (vlo - vlo2) / double(lo - lo2)) * double(rows - lo);
    else if (lo) est = vlo * (1.0 + 0.1 * double(rows - lo));  // one point: +10 %/row until measured
    else if (hi) est = vhi;                                      // optimistic below the first known window
    else est = 1.0 + 0.1 * double(rows - 1);                     // nothing known: relative units
    return std::max(est, floor);
}

double Arbiter::survival(Source source, std::uint32_t bucket, std::uint32_t depth) const {
    const auto& table = source == Source::lookup ? lookup_[std::min(bucket, lookup_buckets - 1)] : model_;
    const auto& prior = source == Source::lookup ? config_.lookup_prior : config_.model_prior;
    const auto& beta = table[source == Source::lookup ? lookup_class(bucket, depth) : depth_class(depth)];
    return (beta.a + prior[0]) / (beta.a + beta.b + prior[0] + prior[1]);
}

// Committed tokens of k drafts: 1 + sum_j B_j, where B_j bounds the cumulative survival S_j = prod_{i<=j} r_i (r_i =
// empirical conditional rate at depth i) ONCE: the prior and the SD are applied to S_j with the evidence reaching
// depth j, and B_j is non-increasing. Bounding every factor separately compounds prior and SD per depth (a perfect
// 15-candidate bucket would predict ~4.4 of 8 tokens); a depth no proposal has reached adds nothing.
double Arbiter::expected(Source source, std::uint32_t bucket, std::uint32_t k) const {
    const auto& table = source == Source::lookup ? lookup_[std::min(bucket, lookup_buckets - 1)] : model_;
    const auto& prior = source == Source::lookup ? config_.lookup_prior : config_.model_prior;
    const double z = source == Source::lookup ? config_.lookup_sd : config_.model_sd;
    double e = 1.0, alive = 1.0, bound = 1.0;
    for (std::uint32_t j = 1; j <= k; ++j) {
        const auto& beta = table[source == Source::lookup ? lookup_class(bucket, j) : depth_class(j)];
        const double n = beta.a + beta.b;
        if (!(n > 0.0)) break;
        alive *= beta.a / n;
        const double q = (alive * n + prior[0]) / (n + prior[0] + prior[1]);
        bound = std::min(bound, std::max(0.0, q - z * std::sqrt(q * (1.0 - q) / (n + prior[0] + prior[1] + 1.0))));
        e += bound;
    }
    return e;
}

Decision Arbiter::decide(const StepInput& input) {
    if (input.remaining == 0 || input.max_rows == 0) return Decision{Source::serial, 0, 0, false, 0.0, 0.0, false};
    const bool black_box = input.model_max_k > 0 && input.model_pass_ms > 0.0;
    if (input.model_max_k > 0 && !black_box && input.model_draft_ms.empty())
        throw std::invalid_argument("arbiter: a learned drafter needs its measured draft cost per depth");
    struct Arm { Decision d; double score = -1.0; };
    std::array<Arm, source_count> best{};
    const std::uint32_t limit = input.remaining - 1;  // drafts that still fit the budget
    const auto ctx = input.context;
    best[0].d = Decision{Source::serial, 0, 1, false, 1.0, verify_ms(1, ctx), false};
    best[0].score = 1.0 / std::max(verify_ms(1, ctx), 1e-9);
    const auto consider = [&](Source source, std::uint32_t k, std::uint32_t bucket, double extra_ms) {
        const auto rows = physical(k + 1);
        if (rows == 0 || rows > input.max_rows) return;
        const double tokens = expected(source, bucket, k), ms = verify_ms(rows, ctx) + extra_ms;
        const double score = tokens / std::max(ms, 1e-9);
        auto& arm = best[std::size_t(source)];
        if (score > arm.score) arm = Arm{Decision{source, k, rows, rows > k + 1, tokens, ms, false}, score};
    };
    const auto* lookup = input.lookup && !input.lookup->tokens.empty() ? input.lookup : nullptr;
    if (lookup) {
        // A resumed proposal predicts the same positions as its run's unresolved pooled entry (same source, cursor
        // moving on): it replaces that entry's tokens, aligned at the next emitted position, so every position of a
        // copy run is scored once. Fresh proposals get their own per-depth entry.
        const auto bucket = std::min(lookup->bucket(), lookup_buckets - 1);
        const auto same = lookup->resumed && lookup->run != 0
            ? std::find_if(pending_.begin(), pending_.end(),
                           [&](const Pending& p) { return p.run == lookup->run && pooled(p.bucket); })
            : pending_.end();
        if (same != pending_.end()) {
            *same = Pending{lookup->tokens, bucket, 0, lookup->run};
        } else {
            if (pending_.size() == max_proposal) pending_.erase(pending_.begin());  // bound; its depths are learned
            pending_.push_back({lookup->tokens, bucket, 0, lookup->run});
        }
        if (!lookup_barred_ && lookup_evidence_[bucket] >= config_.min_lookup_evidence) {
            const auto n = std::min<std::uint32_t>(std::uint32_t(lookup->tokens.size()), limit);
            for (const auto w : config_.windows)
                if (w >= 2 && w <= input.max_rows && n > 0)
                    consider(Source::lookup, std::min(w - 1, n), bucket, config_.lookup_overhead_ms);
        }
    }
    const auto draft_ms = [&](std::uint32_t k) {
        const auto i = std::min<std::size_t>(k, input.model_draft_ms.size()) - 1;
        return input.model_draft_ms[i] * double(k) / double(i + 1);  // past the table: linear in k
    };
    const std::uint32_t model_n = std::min(input.model_max_k, limit);
    if (model_n > 0 && black_box) {  // the drafter decides its own k <= model_n; scored on its measured rate
        const auto rows = physical(model_n + 1);
        if (rows != 0 && rows <= input.max_rows)
            best[2] = Arm{Decision{Source::model, model_n, rows, rows > model_n + 1, input.model_pass_tokens,
                                   input.model_pass_ms, false}, input.model_pass_tokens / input.model_pass_ms};
    } else if (model_n > 0) {
        for (const auto w : config_.windows)
            if (w >= 2 && w <= input.max_rows) {
                const auto k = std::min(w - 1, model_n);
                consider(Source::model, k, 0, draft_ms(k));
            }
    }
    std::size_t pick = 0;
    for (std::size_t s = 1; s < source_count; ++s)
        if (best[s].score > best[pick].score) pick = s;
    // Hysteresis between arms, but never staying on an arm that is below serial.
    const auto current = std::size_t(last_);
    if (pick != current && best[current].score >= best[0].score &&
        best[pick].score < best[current].score * config_.switch_in)
        pick = current;
    if (pick == std::size_t(Source::model) && passes_ < model_barred_until_) pick = 0;
    Decision d = best[pick].d;
    // Exploration of the learned drafter at its deepest legal k: first measurements, then re-probes.
    const bool explore = model_passes_ < config_.min_model_passes || passes_ >= last_model_ + config_.probe_every;
    if (model_n > 0 && d.source != Source::model && explore && black_box) {
        d = best[2].d;
        d.probe = true;
        model_barred_until_ = 0;
    } else if (model_n > 0 && d.source != Source::model && explore) {
        std::uint32_t k = 0, rows = 0;
        for (const auto w : config_.windows)
            if (w >= 2 && w <= input.max_rows) { k = std::min(w - 1, model_n); rows = physical(k + 1); }
        if (k > 0) {
            d = Decision{Source::model, k, rows, rows > k + 1, expected(Source::model, 0, k),
                         verify_ms(rows, ctx) + draft_ms(k), true};
            model_barred_until_ = 0;
        }
    }
    last_ = d.source;
    last_black_box_ = d.source == Source::model && black_box;
    return d;
}

void Arbiter::decay(double tokens) {
    const double f = std::exp2(-tokens / config_.half_life_tokens);
    for (auto& table : lookup_)
        for (auto& beta : table) { beta.a *= f; beta.b *= f; }
    for (auto& e : lookup_evidence_) e *= f;
    for (auto& beta : model_) { beta.a *= f; beta.b *= f; }
}

void Arbiter::learn(Table& table, std::uint32_t proposed, std::uint32_t accepted, double weight) {
    proposed = std::min(proposed, max_proposal);
    accepted = std::min(accepted, proposed);
    for (std::uint32_t d = 1; d <= accepted; ++d) table[depth_class(d)].a += weight;
    if (accepted < proposed) table[depth_class(accepted + 1)].b += weight;
}

void Arbiter::observe(const Outcome& o, std::span<const std::uint32_t> emitted) {
    ++passes_;
    decay(double(std::max<std::size_t>(emitted.size(), 1)));
    if (o.source == Source::model) {
        learn(model_, o.proposed, o.accepted, 1.0);
        last_model_ = passes_;
        ++model_passes_;
        // Retreat only for a per-depth drafter: a black-box drafter's measured rate already prices its misses
        // (on prose, ~50 % first-row acceptance barred a winning MTP drafter for 32 passes again and again).
        model_misses_ = !last_black_box_ && o.proposed > 0 && o.accepted == 0 ? model_misses_ + 1 : 0;
        if (model_misses_ >= config_.retreat_misses) { model_barred_until_ = passes_ + config_.probe_every; model_misses_ = 0; }
    }
    if (o.source == Source::lookup) {
        lookup_misses_ = o.proposed > 0 && o.accepted == 0 ? lookup_misses_ + 1 : 0;
        if (lookup_misses_ >= config_.retreat_misses) { lookup_barred_ = true; lookup_misses_ = 0; }
    }
    auto& c = cost_[context_bin(o.context)];
    for (auto& w : c.weight) w *= config_.cost_aging;
    if (o.rows > 0 && o.rows < c.ms.size() && o.verify_ms > 0.0) {
        auto& w = c.weight[o.rows];
        c.ms[o.rows] = (c.ms[o.rows] * w + o.verify_ms) / (w + 1.0);
        w = std::min(w + 1.0, config_.cost_weight_cap);
    }
    // Resolve pending lookup proposals: each one predicted the tokens emitted after the step it was made at.
    std::size_t keep = 0;
    for (std::size_t i = 0; i < pending_.size(); ++i) {
        auto& p = pending_[i];
        bool resolved = false;
        auto& table = lookup_[p.bucket];
        for (const auto t : emitted) {  // depth d is known as soon as token d is emitted
            auto& beta = table[lookup_class(p.bucket, p.matched + 1)];
            const bool hit = p.tokens[p.matched] == t;
            if (p.matched == 0 || pooled(p.bucket)) {
                lookup_evidence_[p.bucket] += 1.0;
                if (hit) lookup_barred_ = false;  // new first-row evidence lifts a lookup retreat
            }
            if (!hit) { beta.b += 1.0; resolved = true; break; }
            beta.a += 1.0;
            if (++p.matched == p.tokens.size()) { resolved = true; break; }
        }
        if (resolved) continue;
        if (keep != i) pending_[keep] = std::move(p);
        ++keep;
    }
    pending_.resize(keep);
}

// ------------------------------------------------------------------------------------------------ SpecStats
void SpecStats::record(const Outcome& o, std::uint32_t emitted_this_pass) {
    ++passes;
    emitted += emitted_this_pass;
    const auto s = std::size_t(o.source);
    ++source_passes[s];
    source_proposed[s] += o.proposed;
    source_accepted[s] += std::min(o.accepted, o.proposed);
    const auto proposed = std::min(o.proposed, max_proposal), accepted = std::min(o.accepted, proposed);
    for (std::uint32_t d = 1; d <= proposed && d <= accepted + 1; ++d) {
        ++depth_offered[d - 1];
        if (d <= accepted) ++depth_accepted[d - 1];
    }
    if (rows_passes.size() <= o.rows) { rows_passes.resize(o.rows + 1, 0); rows_ms.resize(o.rows + 1, 0.0); }
    ++rows_passes[o.rows];
    rows_ms[o.rows] += o.verify_ms;
    draft_ms += o.draft_ms;
}

void SpecStats::lookup_time(double us) {
    lookup_us_max = std::max(lookup_us_max, us);
    std::size_t i = 0;
    while (i + 1 < lookup_us_hist.size() && us >= std::ldexp(1.0, int(i) - 2)) ++i;
    ++lookup_us_hist[i];
}

double SpecStats::lookup_us_quantile(double q) const {
    std::uint64_t total = 0;
    for (const auto n : lookup_us_hist) total += n;
    if (total == 0) return 0.0;
    const auto rank = std::uint64_t(std::ceil(q * double(total)));
    std::uint64_t seen = 0;
    for (std::size_t i = 0; i < lookup_us_hist.size(); ++i) {
        seen += lookup_us_hist[i];
        if (seen >= rank) return std::min(std::ldexp(1.0, int(i) - 2), lookup_us_max);
    }
    return lookup_us_max;
}

std::string SpecStats::json() const {
    std::string j;
    char b[96];
    const auto num = [&](double v) { std::snprintf(b, sizeof b, "%.6g", v); return std::string(b); };
    j += "{\"passes\":" + std::to_string(passes) + ",\"emitted\":" + std::to_string(emitted);
    j += ",\"tokens_per_pass\":" + num(passes ? double(emitted) / double(passes) : 0.0);
    j += ",\"sources\":{";
    for (std::size_t s = 0; s < source_count; ++s) {
        j += std::string(s ? "," : "") + "\"" + source_name(Source(s)) + "\":{\"passes\":" +
             std::to_string(source_passes[s]) + ",\"proposed\":" + std::to_string(source_proposed[s]) +
             ",\"accepted\":" + std::to_string(source_accepted[s]) + "}";
    }
    std::size_t depth = max_proposal;
    while (depth > 0 && depth_offered[depth - 1] == 0) --depth;
    j += "},\"depth_offered\":[";
    for (std::size_t d = 0; d < depth; ++d) j += (d ? "," : "") + std::to_string(depth_offered[d]);
    j += "],\"depth_accepted\":[";
    for (std::size_t d = 0; d < depth; ++d) j += (d ? "," : "") + std::to_string(depth_accepted[d]);
    j += "],\"rows\":{";
    bool first = true;
    for (std::size_t r = 0; r < rows_passes.size(); ++r) {
        if (!rows_passes[r]) continue;
        j += std::string(first ? "" : ",") + "\"" + std::to_string(r) + "\":{\"passes\":" + std::to_string(rows_passes[r]) +
             ",\"verify_ms\":" + num(rows_ms[r] / double(rows_passes[r])) + "}";
        first = false;
    }
    std::uint64_t lookups = 0;
    for (const auto n : lookup_us_hist) lookups += n;
    j += "},\"draft_ms\":" + num(draft_ms) + ",\"lookup_us\":{\"count\":" + std::to_string(lookups) +
         ",\"p50\":" + num(lookup_us_quantile(0.5)) + ",\"p99\":" + num(lookup_us_quantile(0.99)) +
         ",\"max\":" + num(lookup_us_max) + "},\"index_bytes\":" + std::to_string(index_bytes) + "}";
    return j;
}

} // namespace strix::spec
