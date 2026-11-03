#include "auto_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::gemma4::engine {
namespace {
constexpr double evidence_decay = 0.98;
}
void DecayedMoments::decay() { n *= evidence_decay; sum *= evidence_decay; square *= evidence_decay; }
void DecayedMoments::add(double x) { n += 1; sum += x; square += x*x; }
double DecayedMoments::bound(bool upper) const {
    if (n <= 0) return 0;
    const double mean = sum/n;
    const double error = std::sqrt(std::max(0.0,square/n-mean*mean)/n);
    return upper ? mean+error : std::max(0.0,mean-error);
}
std::size_t AutoPolicy::bucket(std::uint64_t p) { return p < 512 ? 0 : p < 2048 ? 1 : p < 8192 ? 2 : p < 32768 ? 3 : 4; }
double AutoPolicy::expected(const Bin& b, std::uint32_t k) {
    return expected_tokens(Survival{b.eligible,b.accepted},k);
}
double AutoPolicy::expected_tokens(const Survival& b, std::uint32_t k) {
    if (k > 7) throw std::invalid_argument("invalid auto window");
    double emitted = 1, survival = 1;
    for (std::uint32_t d = 0; d < k; ++d) {
        // Beta(1,2): one prior success and two failures. Subtract one posterior
        // standard deviation; unobserved/rejected suffixes are never failures.
        const double n = b.eligible[d]+3, p = (b.accepted[d]+1)/n;
        const double lower = std::max(0.0,p-std::sqrt(p*(1-p)/(n+1)));
        survival *= lower;
        emitted += survival;
    }
    return emitted;
}
double AutoPolicy::draft_estimate(const Bin& b, std::uint32_t k) {
    if (b.fit_n <= 0) return -1;
    const double mean = b.fit_d/b.fit_n;
    const double var = b.fit_n*b.fit_kk-b.fit_k*b.fit_k;
    if (b.fit_n < 2 || var <= 1e-9) return mean;
    const double slope = std::max(0.0,(b.fit_n*b.fit_kd-b.fit_k*b.fit_d)/var);
    const double fixed = std::max(0.0,(b.fit_d-slope*b.fit_k)/b.fit_n);
    return fixed+slope*k;
}
WindowDecision AutoPolicy::choose(std::uint32_t max_k, std::uint64_t position) const {
    if (max_k > 7) throw std::invalid_argument("invalid auto window cap");
    const auto& b = bins_[bucket(position)];
    WindowDecision result;
    if (!max_k || b.plain.n < 3) return result;
    const double plain = b.plain.bound(false);
    if (!(plain > 0)) return result;
    // Refresh the bare target baseline at least every sixteen cycles.
    if (b.cycles%16 == 15) return result;
    if (b.pending_probe && b.pending_probe <= max_k)
        return {b.pending_probe,true,expected(b,b.pending_probe),b.cycle[b.pending_probe].bound(true),0};
    double best_rate = 1.05/plain;
    for (std::uint32_t k = 1; k <= max_k; ++k) {
        if (b.cycle[k].n < 1.5) continue;
        // Conservative cost: the measured cycle upper bound, or the measured verify upper bound plus the fitted
        // per-depth draft cost and context, whichever is larger. The fit can only make a window look more expensive.
        const double fitted = draft_estimate(b,k);
        const double composed = b.verify[k].n >= 1.5 && fitted >= 0 ? b.verify[k].bound(true)+fitted+b.context.bound(true) : 0;
        const double cost = std::max(b.cycle[k].bound(true),composed), tokens = expected(b,k);
        const double rate = tokens/cost;
        if (rate > best_rate) {
            best_rate = rate;
            result = {k,false,tokens,cost,rate*plain};
        }
    }
    // At most six bootstrap probes, every other cycle after four plain rows.
    // Later calibration uses two adjacent samples per refresh, every sixteen
    // serial cycles or thirty-two admitted cycles (plus baseline refreshes).
    // These bounded exceptions gather evidence; they are not gain admissions.
    const bool bootstrap = b.cycles >= 4 && b.probes < 6 && b.cycles%2 == 0;
    const bool refresh = b.cycles >= 16 && b.cycles%(result.k ? 32 : 16) == 0;
    if (bootstrap || refresh) {
        std::uint32_t probe = std::min(2U,max_k);
        if (!bootstrap) {
            // Measure every R directly. No independent-row or interpolated
            // verify time is used to admit an unmeasured causal window.
            for (std::uint32_t k = 1; k <= max_k; ++k)
                if (b.cycle[k].n < b.cycle[probe].n) probe = k;
        }
        result = {probe,true,expected(b,probe),b.cycle[probe].bound(true),0};
    }
    return result;
}
void AutoPolicy::observe(std::uint64_t position, std::uint32_t k, double total_ms,
    double draft_ms, double verify_ms, double context_ms, const dflash::Acceptance& delta, bool probe) {
    if (k > 7 || !std::isfinite(total_ms) || total_ms <= 0 || !std::isfinite(draft_ms) || draft_ms < 0 ||
        !std::isfinite(verify_ms) || verify_ms <= 0 || !std::isfinite(context_ms) || context_ms < 0 ||
        !delta.emitted) throw std::invalid_argument("invalid auto cycle observation");
    auto& b = bins_[bucket(position)];
    for (auto& m : b.cycle) m.decay();
    for (auto& m : b.verify) m.decay();
    b.plain.decay(); b.draft.decay(); b.context.decay();
    learn(b,k,draft_ms,delta);
    if (probe && k) {
        ++b.probes; ++b.probe_windows[k];
        if (b.cycles >= 16) b.pending_probe = b.pending_probe == k ? 0 : k;
    }
    ++b.cycles;
    b.cycle[k].add(total_ms); b.verify[k].add(verify_ms); b.context.add(context_ms);
    if (k) b.draft.add(draft_ms);
    // A serial cycle with a live drafter still appends context. Compare against
    // the bare OFF route by removing that measured context maintenance cost.
    else b.plain.add(std::max(0.001,total_ms-context_ms));
}
void AutoPolicy::learn(Bin& b, std::uint32_t k, double draft_ms, const dflash::Acceptance& delta) {
    for (std::size_t d = 0; d < 7; ++d) {
        if (delta.depth_accepted[d] > delta.depth_eligible[d]) throw std::invalid_argument("invalid conditional survival");
        // Depth-local decay: only depths that received evidence this cycle age. A short admitted window no longer
        // erases the deeper depths' (probe) evidence; their survival estimate stays as last measured.
        if (!delta.depth_eligible[d]) continue;
        b.eligible[d] = b.eligible[d]*evidence_decay+double(delta.depth_eligible[d]);
        b.accepted[d] = b.accepted[d]*evidence_decay+double(delta.depth_accepted[d]);
    }
    if (k) {
        for (auto* v : {&b.fit_n,&b.fit_k,&b.fit_kk,&b.fit_d,&b.fit_kd}) *v *= evidence_decay;
        b.fit_n += 1; b.fit_k += k; b.fit_kk += double(k)*k; b.fit_d += draft_ms; b.fit_kd += double(k)*draft_ms;
    }
}
double AutoPolicy::expected_tokens(std::uint64_t position, std::uint32_t k) const {
    if (k > 7) throw std::invalid_argument("invalid auto window");
    return expected(bins_[bucket(position)],k);
}
AutoPolicy::Survival AutoPolicy::survival(std::uint64_t position) const {
    const auto& b = bins_[bucket(position)];
    return {b.eligible,b.accepted};
}
double AutoPolicy::draft_cost(std::uint64_t position, std::uint32_t k) const {
    if (k > 7) throw std::invalid_argument("invalid auto window");
    return draft_estimate(bins_[bucket(position)],k);
}
void AutoPolicy::observe_multi(std::uint64_t position, std::uint32_t k, double draft_ms, const dflash::Acceptance& delta) {
    if (!k || k > 7 || !std::isfinite(draft_ms) || draft_ms < 0 || !delta.emitted)
        throw std::invalid_argument("invalid multi-stream draft observation");
    learn(bins_[bucket(position)],k,draft_ms,delta);
}
strix::serve::Json AutoPolicy::receipt() const {
    using Json = strix::serve::Json;
    auto result = Json::array();
    for (std::size_t i = 0; i < bins_.size(); ++i) {
        const auto& b = bins_[i]; if (!b.cycles) continue;
        auto eligible = Json::array(), accepted = Json::array(), windows = Json::array(), costs = Json::array(), verify = Json::array();
        for (const auto n : b.eligible) eligible.push(n);
        for (const auto n : b.accepted) accepted.push(n);
        for (const auto n : b.probe_windows) windows.push(n);
        for (const auto& m : b.cycle) costs.push(m.bound(true));
        for (const auto& m : b.verify) verify.push(m.bound(true));
        result.push(Json(Json::Object{{"length_bin",std::uint64_t(i)},{"cycles",b.cycles},{"probes",b.probes},
            {"probe_windows",windows},{"decayed_eligible",eligible},{"decayed_accepted",accepted},
            {"cycle_upper_ms",costs},{"verify_upper_ms",verify},{"plain_lower_ms",b.plain.bound(false)},
            {"draft_upper_ms",b.draft.bound(true)},{"context_upper_ms",b.context.bound(true)},
            {"draft_fit_k1_ms",draft_estimate(b,1)},{"draft_fit_k7_ms",draft_estimate(b,7)}}));
    }
    return result;
}
namespace {
double env_number(const char* name, double fallback, double low, double high) {
    const auto* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const double x = std::strtod(v,&end);
    if (end == v || *end || !std::isfinite(x) || x < low || x > high)
        throw std::invalid_argument(std::string("invalid ")+name+"="+v);
    return x;
}
}
MultiStreamConfig MultiStreamConfig::from_environment() {
    MultiStreamConfig c;
    if (const auto* v = std::getenv("STRIX_G4_DRAFT_MULTI")) c.enabled = std::string(v) != "0";
    const double k = env_number("STRIX_G4_DRAFT_MULTI_K",c.max_k,1,7);
    if (k != std::floor(k)) throw std::invalid_argument("STRIX_G4_DRAFT_MULTI_K must be an integer 1..7");
    c.max_k = std::uint32_t(k);
    c.max_draft_share = env_number("STRIX_G4_DRAFT_MULTI_COST",c.max_draft_share,0,1);
    c.min_gain = env_number("STRIX_G4_DRAFT_MULTI_GAIN",c.min_gain,1,4);
    return c;
}
strix::serve::Json MultiStreamConfig::json() const {
    return strix::serve::Json(strix::serve::Json::Object{{"enabled",enabled},{"max_k",std::uint64_t(max_k)},
        {"max_draft_share",max_draft_share},{"min_gain",min_gain}});
}
ConfidenceConfig ConfidenceConfig::from_environment() {
    ConfidenceConfig c;
    const auto* v = std::getenv("STRIX_G4_MTP_CONF");
    if (v && (std::string(v) == "0" || std::string(v) == "off")) return c; // the OFF switch
    if (!v || !*v || std::string(v) == "auto") { c.tau = builtin_tau; c.source = Source::builtin; c.per_drafter = true; } // default ON, per drafter
    else {
        c.tau = env_number("STRIX_G4_MTP_CONF",0,0,1);
        if (!(c.tau > 0)) return ConfidenceConfig{}; // "0.0": disabled, as "0"
        c.source = Source::env;
    }
    const auto integer = [](const char* name, std::uint32_t fallback) {
        const double x = env_number(name,fallback,1,7);
        if (x != std::floor(x)) throw std::invalid_argument(std::string(name)+" must be an integer 1..7");
        return std::uint32_t(x);
    };
    const auto set = [](const char* name) { const auto* e = std::getenv(name); return e && *e; };
    c.max_k_env = set("STRIX_G4_MTP_CONF_K"); c.first_env = set("STRIX_G4_MTP_CONF_FIRST");
    c.max_k = integer("STRIX_G4_MTP_CONF_K",c.max_k);
    c.first = integer("STRIX_G4_MTP_CONF_FIRST",c.first);
    c.step = integer("STRIX_G4_MTP_CONF_STEP",c.step);
    c.min1 = env_number("STRIX_G4_MTP_CONF_MIN1",c.min1,0,1);
    if (const auto* l = std::getenv("STRIX_G4_MTP_CONF_LONG"); l && *l && std::string(l) != "auto")
        c.long_ratio = std::string(l) == "off" ? 0 : env_number("STRIX_G4_MTP_CONF_LONG",c.long_ratio,0,1);
    return c;
}
const char* ConfidenceConfig::source_name(Source s) {
    switch (s) {
    case Source::env: return "env";
    case Source::drafter_config: return "drafter-config";
    case Source::builtin: return "builtin";
    default: return "off";
    }
}
strix::serve::Json ConfidenceConfig::json() const {
    return strix::serve::Json(strix::serve::Json::Object{{"enabled",enabled()},{"tau",tau},{"max_k",std::uint64_t(max_k)},
        {"first",std::uint64_t(first)},{"step",std::uint64_t(step)},{"min1",min1},{"source",source_name(source)},
        {"per_drafter",per_drafter},{"long_ratio",long_ratio},{"long_extend",long_extend},{"long_verify",long_verify}});
}
MultiStreamPolicy::MultiStreamPolicy(MultiStreamConfig config) : config_(config) {
    if (config_.max_k < 1 || config_.max_k > 7 || !(config_.max_draft_share >= 0) || !(config_.min_gain >= 1))
        throw std::invalid_argument("invalid multi-stream policy config");
}
BatchPlan MultiStreamPolicy::plan(std::span<const StreamCandidate> streams, double plain_ms, bool include_unmeasured) const {
    // Mirrors Coordinator::step's row allocation: slot order, every later stream keeps at least one row.
    BatchPlan p;
    const double share_ms = config_.max_draft_share*plain_ms/double(streams.size());
    std::size_t remaining = 8, pending = streams.size();
    for (std::size_t i = 0; i < streams.size(); ++i) {
        const auto& s = streams[i];
        const auto capacity = remaining-(--pending);
        std::uint32_t k = 0;
        if (s.policy) {
            k = std::min({config_.max_k,s.limit,std::uint32_t(capacity-1)});
            if (k) {
                // Per-depth cost gate: the stream's own fitted draft cost of this window (C1 and C>1 draft
                // samples). Unmeasured drafters are only ever probed, never admitted.
                const double cost = s.policy->draft_cost(s.position,k);
                const bool cheap = cost >= 0 && cost <= share_ms;
                if (!cheap && !(cost < 0 && include_unmeasured)) k = 0;
            }
        }
        p.k[i] = k; p.extra += k; remaining -= k+1;
    }
    // Batch tokens: the decision is about the batch aggregate, so survival is pooled over the drafting streams
    // (sum of their decayed per-depth evidence, then the same Beta(1,2)-minus-SD estimator as C1). Per-stream
    // evidence at >1 stream is one sample per drafted step, too thin alone to ever clear the bound.
    AutoPolicy::Survival pooled;
    std::uint32_t drafting = 0;
    for (std::size_t i = 0; i < streams.size(); ++i) {
        if (!p.k[i]) continue;
        const auto e = streams[i].policy->survival(streams[i].position);
        for (std::size_t d = 0; d < 7; ++d) { pooled.eligible[d] += e.eligible[d]; pooled.accepted[d] += e.accepted[d]; }
        ++drafting;
    }
    for (std::size_t i = 0; i < streams.size(); ++i)
        if (p.k[i]) p.tokens += AutoPolicy::expected_tokens(pooled,p.k[i]);
    p.tokens += double(streams.size()-drafting);
    return p;
}
BatchPlan MultiStreamPolicy::choose(std::span<const StreamCandidate> streams) const {
    const auto c = streams.size();
    if (!config_.enabled || c < 2 || c > 8) return {};
    const auto& row = rows_[c-1];
    if (row.step[0].n < 3) return {};
    const double plain = row.step[0].bound(false);
    if (!(plain > 0)) return {};
    // Refresh the plain batch baseline at least every sixteen steps.
    if (row.cycles%16 == 15) return {};
    BatchPlan result;
    const auto admitted = plan(streams,plain,false);
    const double plain_rate = double(c)/plain;
    if (admitted.extra && row.step[admitted.extra].n >= 1.5) {
        const double cost = row.step[admitted.extra].bound(true), rate = admitted.tokens/cost;
        if (rate > config_.min_gain*plain_rate) {
            result = admitted; result.cost_ms = cost; result.gain = rate/plain_rate;
        }
    }
    const bool bootstrap = row.cycles >= 4 && row.probes < 6 && row.cycles%2 == 0;
    const bool refresh = row.cycles >= 16 && (row.pending_probe || row.cycles%(result.extra ? 32 : 16) == 0);
    if (bootstrap || refresh) {
        auto probe = plan(streams,plain,true);
        if (probe.extra) {
            probe.probe = true; probe.cost_ms = row.step[probe.extra].bound(true);
            return probe;
        }
    }
    return result;
}
void MultiStreamPolicy::observe(std::size_t streams, std::uint32_t extra_rows, double step_ms, bool probe) {
    if (streams < 2 || streams > 8 || extra_rows > 8-streams || !std::isfinite(step_ms) || step_ms <= 0)
        throw std::invalid_argument("invalid multi-stream step observation");
    if (!config_.enabled) return;
    auto& row = rows_[streams-1];
    for (auto& m : row.step) m.decay();
    row.step[extra_rows].add(step_ms);
    if (probe && extra_rows) {
        ++row.probes;
        if (row.cycles >= 16) row.pending_probe = row.pending_probe ? 0 : 1;
    }
    ++row.cycles;
}
strix::serve::Json MultiStreamPolicy::receipt() const {
    using Json = strix::serve::Json;
    auto result = Json::array();
    for (std::size_t i = 1; i < rows_.size(); ++i) {
        const auto& r = rows_[i]; if (!r.cycles) continue;
        auto upper = Json::array(), samples = Json::array();
        for (const auto& m : r.step) { upper.push(m.bound(true)); samples.push(m.n); }
        result.push(Json(Json::Object{{"streams",std::uint64_t(i+1)},{"cycles",r.cycles},{"probes",r.probes},
            {"plain_lower_ms",r.step[0].bound(false)},{"step_upper_ms_by_extra_rows",upper},{"decayed_samples",samples}}));
    }
    return result;
}
} // namespace strix::gemma4::engine
