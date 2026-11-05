#include "coordinator.hpp"
#include "strix/model_content_hash.hpp"
#include "gemma4/prefill/prefill.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#ifdef __linux__
#include <fstream>
#include <sys/utsname.h>
#endif

namespace strix::gemma4::engine {
namespace {
using Clock = std::chrono::steady_clock;
using Json = strix::serve::Json;
double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now()-start).count(); }
bool env_on(const char* name) { const auto* v = std::getenv(name); return !v || std::string(v) != "0"; }
// PREFILL_BATCH (STRIX_G4_PREFILL_BATCH=0 off): streams with prompt chunks pending share one prefill call per step.
bool prefill_batch_enabled() { static const bool on = env_on("STRIX_G4_PREFILL_BATCH"); return on; }
void require(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }
void add(dflash::Acceptance& a, const dflash::Acceptance& b) {
    a.cycles += b.cycles; a.attempted += b.attempted; a.eligible += b.eligible;
    a.accepted += b.accepted; a.emitted += b.emitted; a.all_accepted += b.all_accepted;
    for (std::size_t i = 0; i < a.depth_eligible.size(); ++i) {
        a.depth_eligible[i] += b.depth_eligible[i]; a.depth_accepted[i] += b.depth_accepted[i];
    }
    for (std::size_t i = 0; i < a.rejection_depth.size(); ++i) a.rejection_depth[i] += b.rejection_depth[i];
}
std::string cost_bucket(const std::string& workload, std::uint64_t position) {
    const auto length = position < 512 ? 0 : position < 2048 ? 1 : position < 8192 ? 2 : position < 32768 ? 3 : 4;
    return workload+":"+std::to_string(length);
}
}
std::string measurement_key(const Backend& backend, const Request& request) {
    strix::Sha256Fast hash;
    const auto assets = backend.identity().dump();
    hash.update(std::as_bytes(std::span(assets.data(),assets.size())));
    hash.update(std::as_bytes(std::span(request.prompt)));
    return strix::model_content_identity_hex(hash.finalize());
}
void Sampling::validate() const {
    require(std::isfinite(temperature) && temperature >= 0 && std::isfinite(top_p) && top_p > 0 && top_p <= 1,
            "invalid temperature/top_p");
}
std::vector<std::uint32_t> candidate_order(std::span<const float> logits, std::uint32_t top_k) {
    // One 64-bit key per id, ascending in better() order: the logit's order-preserving bits inverted (descending
    // logit, -0 == +0), then the id (low id first on ties). Keys are distinct, so any correct sort gives this order.
    const auto n = logits.size();
    std::vector<std::uint64_t> keys(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto bits = bit_cast<std::uint32_t>(logits[i]);
        if ((bits & 0x7fffffffU) == 0) bits = 0;
        const auto ascending = bits & 0x80000000U ? ~bits : bits | 0x80000000U;
        keys[i] = std::uint64_t(~ascending) << 32U | i;
    }
    if (top_k && top_k < n) {
        std::nth_element(keys.begin(),keys.begin()+top_k,keys.end());
        keys.resize(top_k);
        std::sort(keys.begin(),keys.end());
    } else {
        // LSD radix sort, four 16-bit digits.
        std::vector<std::uint64_t> other(n);
        std::vector<std::size_t> count(65537);
        for (unsigned shift = 0; shift < 64; shift += 16) {
            std::fill(count.begin(),count.end(),0);
            for (const auto k : keys) ++count[(k >> shift & 0xffffU)+1];
            for (std::size_t d = 1; d < count.size(); ++d) count[d] += count[d-1];
            for (const auto k : keys) other[count[k >> shift & 0xffffU]++] = k;
            keys.swap(other);
        }
    }
    std::vector<std::uint32_t> order(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) order[i] = std::uint32_t(keys[i]);
    return order;
}
std::uint32_t select_token(std::span<const float> logits, const Sampling& sampling, std::uint64_t& draws) {
    sampling.validate();
    require(!logits.empty() && logits.size() <= UINT32_MAX, "empty/oversized logits");
    for (const auto x : logits) require(std::isfinite(x), "nonfinite target logits");
    std::uint32_t best = 0;
    for (std::uint32_t i = 1; i < logits.size(); ++i) if (better(logits[i],i,logits[best],best)) best = i;
    if (sampling.temperature == 0) return best;
    // STRIX_G4_SAMPLE_ORDER=legacy restores the comparator sort; both give the same order.
    const auto* order_env = std::getenv("STRIX_G4_SAMPLE_ORDER");
    std::vector<std::uint32_t> order;
    if (order_env && std::string(order_env) == "legacy") {
        order.resize(logits.size());
        std::iota(order.begin(),order.end(),0U);
        std::sort(order.begin(),order.end(),[&](auto a, auto b) { return better(logits[a],a,logits[b],b); });
        if (sampling.top_k && order.size() > sampling.top_k) order.resize(sampling.top_k);
    } else {
        order = candidate_order(logits,sampling.top_k);
    }
    std::vector<double> weights;
    double sum = 0;
    for (const auto id : order) {
        const float delta = (logits[id]-logits[best])/sampling.temperature;
        // Negative overflow is the zero-mass tail, never a nonfinite model value.
        const double w = declared_exp(delta);
        sum += w; weights.push_back(w);
    }
    std::size_t count = 0;
    double retained = 0;
    do { retained += weights[count++]; } while (count < weights.size() && retained < double(sampling.top_p)*sum);
    std::uint64_t z = sampling.seed + (++draws)*UINT64_C(0x9e3779b97f4a7c15);
    z = (z^(z>>30U))*UINT64_C(0xbf58476d1ce4e5b9);
    z = (z^(z>>27U))*UINT64_C(0x94d049bb133111eb);
    z ^= z>>31U;
    const double threshold = double(z>>11U)*0x1p-53*retained;
    double cdf = 0;
    for (std::size_t i = 0; i < count; ++i) { cdf += weights[i]; if (threshold < cdf) return order[i]; }
    return order[count-1];
}
const char* status_name(Status s) {
    switch (s) {
    case Status::running: return "running";
    case Status::stopped: return "stop";
    case Status::length: return "length";
    case Status::cancelled: return "cancelled";
    case Status::failed: return "error";
    }
    return "error";
}
Json effective_config(const Backend& backend, const Request& request, bool drafter_available, std::size_t streams) {
    auto j = backend.identity();
    const auto* identity = j.find("identity");
    j.set("decode_identity",identity ? *identity : Json(arithmetic_id)); j.set("prefill_identity","prefill-exact"); j.set("row_cap",8);
    j.set("sampling_identity","splitmix64-cdf-v1"); j.set("temperature",double(request.sampling.temperature));
    j.set("top_p",double(request.sampling.top_p)); j.set("top_k",std::uint64_t(request.sampling.top_k)); j.set("seed",request.sampling.seed);
    j.set("max_tokens",std::uint64_t(request.max_tokens)); j.set("draft_k_max",std::uint64_t(request.draft_k)); j.set("streams",std::uint64_t(streams));
    auto stops = Json::array(); for (const auto id : request.stop_ids) stops.push(std::uint64_t(id));
    j.set("stop_ids",stops);
    j.set("dflash_requested",request.dflash == DraftMode::on ? "on" : request.dflash == DraftMode::off ? "off" : "auto");
    const bool enabled = dflash::enabled_from_environment();
    const auto* environment = std::getenv("STRIX_G4_DFLASH");
    const bool forced = environment && std::string(environment) == "on" && request.dflash == DraftMode::automatic;
    j.set("dflash_admission",!enabled ? "environment-off" : request.sampling.temperature != 0 ? "serial-sampling" :
        request.dflash == DraftMode::off ? "request-off" : !drafter_available ? "no-drafter" :
        request.dflash == DraftMode::on || forced ? "explicit-bounded-window" : "measured-conservative-auto");
    auto switches = j.find("off_switches") ? Json(j.find("off_switches")->as_object()) : Json::object();
    switches.set("STRIX_G4_DFLASH",enabled);
    const auto multi = MultiStreamConfig::from_environment();
    switches.set("STRIX_G4_DRAFT_MULTI",multi.enabled);
    if (!switches.find("STRIX_G4_TOKENIZER_HEAP")) switches.set("STRIX_G4_TOKENIZER_HEAP",env_on("STRIX_G4_TOKENIZER_HEAP"));
    j.set("off_switches",switches);
    j.set("auto_policy",multi.enabled ?
        "stream-survival-cost-v1; C1 length bins; Beta(1,2) minus one posterior SD; decay 0.98; measured cycle upper/plain lower; >=5% gain; bounded calibration probes; "
        "C2..C8 multi-stream-batch-v1: window <= max_k for per-depth-cost drafters (MTP; DFlash C1-only) under max_draft_share, measured batch step upper/plain lower per (C, draft rows), >= min_gain" :
        "stream-survival-cost-v1; C1 length bins; Beta(1,2) minus one posterior SD; decay 0.98; measured cycle upper/plain lower; >=5% gain; bounded calibration probes; C2..C8 plain");
    if (multi.enabled) j.set("multi_stream_policy",multi.json());
    // Only when enabled: OFF receipts stay byte-identical.
    const auto confidence = ConfidenceConfig::from_environment();
    if (confidence.enabled()) {
        j.set("auto_policy",j.find("auto_policy")->as_string()+"; C1 MTP drafter-confidence window: cumulative top-1 prob >= tau, chunked drafting, no probes");
        j.set("mtp_confidence",confidence.json());
    }
    j.set("dflash_environment",environment ? environment : "unset");
    j.set("qualification","per profile: see profile.levers (ready rows applied, pending rows listed)");
#ifdef __linux__
    utsname system{};
    j.set("kernel",uname(&system) == 0 ? system.release : "unavailable");
    std::string perfopt;
    std::ifstream parameter("/sys/module/amdgpu/parameters/iommu_perfopt");
    j.set("amdgpu.iommu_perfopt",parameter >> perfopt ? perfopt : "not-present");
#endif
    return j;
}
void CostModel::observe(std::size_t streams, std::uint32_t k, double ms, std::size_t emitted, std::uint64_t position, const std::string& workload) {
    require(streams >= 1 && streams <= 8 && k <= 7 && std::isfinite(ms) && ms > 0 && emitted > 0, "invalid cycle observation");
    // Bound retained evidence in a long-lived service. Eviction only forces
    // conservative remeasurement; it cannot change the verified output.
    const auto key = cost_bucket(workload,position);
    if (!bins_.contains(key) && bins_.size() >= 128) bins_.erase(bins_.begin());
    auto& b = bins_[key][streams-1][k];
    b.n += 1; b.time += ms; b.time2 += ms*ms;
    const double t = double(emitted); b.tokens += t; b.tokens2 += t*t;
}
std::uint32_t CostModel::choose(std::size_t streams, std::uint32_t max_k, std::uint64_t position, const std::string& workload) const {
    require(streams >= 1 && streams <= 8 && max_k <= 7, "invalid cost-model bucket");
    const auto found = bins_.find(cost_bucket(workload,position));
    if (found == bins_.end()) return 0;
    const auto& bins = found->second[streams-1];
    const auto bound = [](const Bin& b, bool tokens, bool upper) {
        const double sum = tokens ? b.tokens : b.time, square = tokens ? b.tokens2 : b.time2;
        const double mean = sum/b.n, error = 2*std::sqrt(std::max(0.0,square/b.n-mean*mean)/b.n);
        return upper ? mean+error : std::max(0.0,mean-error);
    };
    if (bins[0].n < 4) return 0;
    const double plain_time = bound(bins[0],false,false);
    if (plain_time <= 0) return 0;
    double best = 1.05*bound(bins[0],true,true)/plain_time;
    std::uint32_t chosen = 0;
    for (std::uint32_t k = 1; k <= max_k; ++k) if (bins[k].n >= 4) {
        const double rate = bound(bins[k],true,false)/bound(bins[k],false,true);
        if (rate > best) { best = rate; chosen = k; }
    }
    return chosen;
}
struct Coordinator::Slot {
    Request request;
    Result out;
    std::unique_ptr<StreamState> state;
    std::unique_ptr<DraftSession> draft;
    std::size_t prompt_processed = 0;
    std::uint64_t generation = 0;
    bool dflash_enabled = false;
    std::string workload;
    AutoPolicy policy;
    std::array<std::uint64_t,8> admitted{};
    double minimum_gain = 0;
    ConfidenceConfig confidence; // resolved per session (env, drafter config or builtin); disabled when not capable
    // Measured confidence-window costs of this stream (decayed): drafting ms, drafted depths, one-row pass ms.
    // Only cycles whose window the confidence rule chose feed it (not prompt-lookup proposals).
    DraftCostRatio conf_cost;
    std::vector<float> last_hidden; // final hidden row of the last committed position (drafters that read it)
};
Coordinator::Coordinator(Backend& backend, DraftFactory drafts, std::shared_ptr<CostModel> costs)
    : backend_(backend), drafts_(std::move(drafts)), costs_(costs ? std::move(costs) : std::make_shared<CostModel>()) {}
Coordinator::~Coordinator() = default;
std::size_t Coordinator::submit(Request r, std::shared_ptr<const PrefixSnapshot> prefix) {
    const auto setup_start = Clock::now();
    const auto& c = backend_.config();
    r.sampling.validate();
    const auto* environment = std::getenv("STRIX_G4_DFLASH");
    if (environment && std::string(environment) == "on" && r.dflash == DraftMode::automatic) r.dflash = DraftMode::on;
    require(!r.prompt.empty() && r.max_tokens > 0 && r.draft_k <= 7, "invalid prompt/max_tokens/draft_k");
    require(r.prompt.size() <= c.max_positions && r.max_tokens <= c.max_positions-r.prompt.size(), "request exceeds context capacity");
    for (const auto id : r.prompt) require(id < c.vocab, "prompt token outside vocabulary");
    std::size_t slot = 0;
    while (slot < slots_.size() && slots_[slot]->out.status == Status::running) ++slot;
    require(slot < 8, "at most eight live streams");
    auto s = std::make_unique<Slot>();
    s->request = std::move(r);
    // Return a completed/cancelled lease before acquiring the replacement.
    // Receipts retain their values; no live transaction crosses this boundary.
    if (slot < slots_.size()) slots_[slot]->draft.reset();
    const auto admitted = std::uint32_t(s->request.prompt.size()+s->request.max_tokens);
    const auto& prompt = s->request.prompt;
    if (prefix && (prefix->tokens.empty() || prefix->tokens.size()+1 > prompt.size() ||
            !std::equal(prefix->tokens.begin(),prefix->tokens.end(),prompt.begin()) || !prefix->state ||
            prefix->state->frontier() != prefix->tokens.size()))
        prefix.reset();
    if (prefix) s->state = prefix->state->fork(admitted);
    if (!s->state) { prefix.reset(); s->state = backend_.create_stream(admitted); }
    s->generation = ++generation_;
    require(bool(s->state) && s->state->frontier() == (prefix ? prefix->tokens.size() : 0), "backend must provide a fresh stream");
    if (!prefix) s->out.digest = s->state->digest();
    s->workload = measurement_key(backend_,s->request);
    s->out.rng_draws = s->request.sampling.draws;
    s->dflash_enabled = dflash::enabled_from_environment();
    if (s->dflash_enabled && s->request.dflash != DraftMode::off && drafts_ && s->request.sampling.temperature == 0)
        s->draft = drafts_();
    if (s->draft) s->draft->attach(*s->state);
    if (s->draft && confidence_.enabled()) s->confidence = s->draft->resolve_confidence(confidence_);
    if (s->request.dflash == DraftMode::on && s->dflash_enabled && s->request.sampling.temperature == 0)
        require(bool(s->draft), "DFlash on requires a drafter");
    if (prefix) {
        const auto F = prefix->tokens.size();
        // The drafter must hold exactly the context of the restored rows; one that cannot be seeded gets a fresh stream.
        if (s->draft && (s->draft->wants_hidden() != !prefix->hidden.empty() ||
                !s->draft->seed({prompt.data(),F+1},prefix->hidden))) {
            s->draft.reset();
            s->state = backend_.create_stream(admitted);
            if (s->dflash_enabled && s->request.dflash != DraftMode::off && drafts_ && s->request.sampling.temperature == 0)
                s->draft = drafts_();
            if (s->draft) s->draft->attach(*s->state);
            if (s->draft && confidence_.enabled()) s->confidence = s->draft->resolve_confidence(confidence_);
            s->out.digest = s->state->digest();
        } else {
            s->prompt_processed = F;
            s->out.cached_tokens = F;
            s->out.prefill_frontier = prefix->prefill_written ? F : 0;
            s->out.frontier = F;
            s->last_hidden = prefix->hidden;
        }
    }
    s->out.setup_ms = elapsed(setup_start);
    if (slot == slots_.size()) slots_.push_back(std::move(s)); else slots_[slot] = std::move(s);
    return slot;
}
std::shared_ptr<PrefixSnapshot> Coordinator::snapshot(std::size_t slot) const {
    const auto& s = *slots_.at(slot);
    const auto F = s.state->frontier();
    if (!F || s.out.status == Status::failed) return nullptr;
    auto state = s.state->fork(std::uint32_t(F));
    if (!state) return nullptr;
    auto out = std::make_shared<PrefixSnapshot>();
    const auto& prompt = s.request.prompt;
    out->tokens.assign(prompt.begin(),prompt.begin()+std::ptrdiff_t(std::min<std::uint64_t>(F,prompt.size())));
    for (std::size_t i = 0; out->tokens.size() < F; ++i) out->tokens.push_back(s.out.ids.at(i));
    out->state = std::move(state);
    out->prefill_written = s.out.prefill_frontier == F;
    if (s.draft && s.draft->wants_hidden()) out->hidden = s.last_hidden;
    return out;
}
void Coordinator::cancel(std::size_t slot) {
    auto& s = *slots_.at(slot);
    if (s.out.status == Status::running) {
        if (s.draft) s.draft->discard();
        s.out.status = Status::cancelled; s.out.digest = s.state->digest();
    }
}
const Result& Coordinator::result(std::size_t slot) const { return slots_.at(slot)->out; }
bool Coordinator::step() {
    struct Work {
        std::size_t slot = 0, start = 0, rows = 1;
        bool prefill = false;
        std::uint32_t k = 0;
        WindowDecision decision;
        bool confident = false, confidence_rule = false; // confidence_rule: the session's rule chose the window
        std::uint32_t drafted = 0;
        dflash::Acceptance delta;
        double draft_ms = 0, context_ms = 0;
        std::array<std::uint32_t,7> proposals{};
        std::unique_ptr<KvTxn> tx;
    };
    std::vector<Work> work;
    std::vector<RowPlan> plans;
    std::size_t live = 0;
    for (const auto& s : slots_) if (s->out.status == Status::running) ++live;
    if (!live) return false;
    const auto start = Clock::now();
    ++steps_;
    // Multi-stream: a whole-batch plan when >1 stream is live, every live stream decodes and none is forced on. Mixed
    // prefill/forced steps stay plain for auto (as before) and are not batch evidence.
    bool batch = false;
    BatchPlan plan;
    if (live > 1 && multi_.config().enabled) {
        batch = true;
        std::vector<StreamCandidate> candidates;
        for (const auto& p : slots_) {
            const auto& s = *p;
            if (s.out.status != Status::running) continue;
            if (s.prompt_processed+1 < s.request.prompt.size() || s.request.dflash == DraftMode::on) { batch = false; break; }
            StreamCandidate c;
            c.position = s.state->frontier();
            const auto budget = s.request.max_tokens-s.out.ids.size();
            if (s.draft && s.draft->multi_stream_candidate() && s.request.dflash == DraftMode::automatic && s.dflash_enabled &&
                s.request.sampling.temperature == 0 && c.position+7 < backend_.config().max_positions) {
                c.policy = &s.policy;
                c.limit = std::min({s.request.draft_k,std::uint32_t(7),std::uint32_t(std::min<std::size_t>(7,budget-1))});
            }
            candidates.push_back(c);
        }
        if (batch) plan = multi_.choose(candidates);
    }
    try {
        // Exact many-row prefill: a stream with more than one pass of prompt
        // rows left prefills one chunk (ending on a chunk multiple) on its own.
        // Committed KV and taps equal the eight-row passes; the drafter receives
        // the same rows in groups of at most eight.
        std::vector<bool> prefilled(slots_.size(), false);
        std::size_t rowed = live; // streams left for this step's forward_rows pass
        if (const auto chunk = backend_.prefill_chunk_rows()) {
            // With a prefill limit, at most that many prefill calls run this step, round-robin from the cursor; the
            // other prompt streams sit this step out (scheduling only: rows and their order per stream are unchanged).
            // PREFILL_BATCH: streams whose chunks fit join one call (one weight pass for all of them); without it each
            // stream's chunk is its own call.
            struct Chunk { std::size_t slot = 0; std::unique_ptr<KvTxn> tx; std::uint64_t position = 0; std::size_t rows = 0; };
            std::vector<std::vector<Chunk>> calls;
            std::size_t cursor = prefill_cursor_, decoding = 0;
            for (const auto& p : slots_)
                if (p->out.status == Status::running && p->prompt_processed+1 >= p->request.prompt.size()) ++decoding;
            const bool batching = prefill_batch_enabled() && backend_.prefill_batch_supported();
            // A pass's rows share the linears; where prefill and decode arithmetic differ, a segment of <= 8 rows
            // would take another kernel than alone, so only longer segments share a call there.
            const auto joinable = [&](std::size_t n) { return backend_.prefill_matches_decode() || n > 8; };
            for (std::size_t j = 0; j < slots_.size(); ++j) {
                const auto i = (prefill_cursor_+j)%slots_.size();
                auto& s = *slots_[i];
                if (s.out.status != Status::running || s.prompt_processed+1+8 >= s.request.prompt.size()) continue;
                s.out.peak_streams = std::max(s.out.peak_streams,live);
                const std::uint64_t position = s.state->frontier();
                auto n = std::min<std::size_t>(s.request.prompt.size()-1-s.prompt_processed, chunk-position%chunk);
                // While other streams decode, a shorter chunk bounds their wait. Only when prefill rows equal decode
                // rows: otherwise where a chunk ends decides which prompt rows a decode pass writes (a short tail goes
                // through forward_rows), so chunk ends stay those of the stream alone.
                if (prefill_budget_ && decoding && backend_.prefill_matches_decode()) n = std::min(n,prefill_budget_);
                std::size_t used = 0;
                if (!calls.empty()) for (const auto& c : calls.back()) used += c.rows;
                const bool join = batching && !calls.empty() && joinable(n) && joinable(calls.back().front().rows) && used+n <= chunk;
                if (!join && prefill_limit_ && calls.size() == prefill_limit_) { prefilled[i] = true; --rowed; continue; }
                if (!join) calls.emplace_back();
                auto tx = s.state->begin();
                require(tx->frontier() == position, "prefill transaction frontier");
                calls.back().push_back({i,std::move(tx),position,n});
                cursor = i+1;
            }
            for (auto& call : calls) {
                const auto begin = Clock::now();
                std::vector<PrefillRows> results;
                bool any_hidden = false, any_taps = false;
                for (const auto& c : call) {
                    const auto& s = *slots_[c.slot];
                    any_hidden |= s.draft && s.draft->wants_hidden(); any_taps |= s.draft && s.draft->wants_taps();
                }
                const auto call_start = Clock::now();
                if (call.size() > 1) {
                    std::vector<PrefillItem> items;
                    for (const auto& c : call) {
                        const auto& s = *slots_[c.slot];
                        items.push_back({c.tx.get(),{s.request.prompt.data()+s.prompt_processed,c.rows}});
                    }
                    results = backend_.prefill_rows_batch(items,any_hidden,any_taps);
                    if (results.empty())  // unsupported: the same chunks, one call each
                        for (const auto& c : call) {
                            const auto& s = *slots_[c.slot];
                            results.push_back(backend_.prefill_rows(*c.tx,{s.request.prompt.data()+s.prompt_processed,c.rows},
                                s.draft && s.draft->wants_hidden(),s.draft && s.draft->wants_taps()));
                        }
                    require(results.size() == call.size(), "backend batched prefill returned another segment count");
                } else {
                    const auto& s = *slots_[call.front().slot];
                    results.push_back(backend_.prefill_rows(*call.front().tx,{s.request.prompt.data()+s.prompt_processed,call.front().rows},
                        s.draft && s.draft->wants_hidden(),s.draft && s.draft->wants_taps()));
                }
                const auto call_ms = elapsed(call_start);
                for (std::size_t k = 0; k < call.size(); ++k) {
                auto& c = call[k];
                const auto i = c.slot;
                auto& s = *slots_[i];
                auto tx = std::move(c.tx);
                const auto position = c.position;
                const auto n = c.rows;
                const bool hidden = s.draft && s.draft->wants_hidden(), taps = s.draft && s.draft->wants_taps();
                const auto& rows = results[k];
                const auto width = std::size_t(backend_.config().hidden);
                const auto tap_rows = rows.tap_rows(), hidden_rows = rows.hidden_rows();
                // Taps nobody consumes may be omitted. Returned taps are scanned
                // here unless the backend checked them on the device before its failure fence.
                require(rows.tap_width == 5*width && (tap_rows.size() == n*rows.tap_width || (!taps && tap_rows.empty())) &&
                    (!hidden || hidden_rows.size() == n*width), "invalid backend prefill layout");
                if (!(rows.levers & prefill::lever_device_finite))
                    for (const auto x : tap_rows) require(std::isfinite(x), "nonfinite backend taps");
                for (const auto& lever : prefill::lever_names) if (rows.levers & lever.bit) ++s.out.prefill_levers[lever.name];
                if (rows.levers & prefill::lever_pinned_taps) s.out.prefill_taps = tap_rows.empty() ? "skipped (no consumer)" : "pinned-async";
                if (call.size() > 1) { ++s.out.prefill_batched_calls; s.out.prefill_batch_streams = std::max<std::uint64_t>(s.out.prefill_batch_streams,call.size()); }
                std::size_t staged = 0;
                const auto stage = [&] {
                    const auto first = staged, count = std::min<std::size_t>(8,n-first);
                    std::vector<std::uint64_t> positions;
                    std::vector<std::uint32_t> next;
                    for (std::size_t r = first; r < first+count; ++r) {
                        positions.push_back(position+r);
                        next.push_back(s.request.prompt[s.prompt_processed+r+1]);
                    }
                    s.draft->stage_rows(tap_rows.empty() ? std::span<const float>{} : tap_rows.subspan(first*rows.tap_width,count*rows.tap_width),
                        hidden ? hidden_rows.subspan(first*width,count*width) : std::span<const float>{},
                        next,positions);
                    staged += count;
                };
                double context_ms = 0;
                if (s.draft) { const auto t = Clock::now(); stage(); context_ms += elapsed(t); }
                const auto commit_start = Clock::now();
                tx->commit(n);
                tx.reset();
                s.out.commit_ms += elapsed(commit_start);
                if (s.draft) {
                    const auto t = Clock::now();
                    s.draft->publish();
                    while (staged < n) { stage(); s.draft->publish(); }
                    if (hidden) s.last_hidden.assign(hidden_rows.end()-std::ptrdiff_t(width),hidden_rows.end());
                    context_ms += elapsed(t);
                }
                s.out.context_ms += context_ms;
                ++s.out.target_calls; s.out.target_rows += n;
                if (s.out.prefill_frontier == position) s.out.prefill_frontier = s.state->frontier();
                s.out.frontier = s.state->frontier();
                s.prompt_processed += n;
                s.out.prefill_ms += elapsed(begin);
                ++s.out.prefill_calls; s.out.prefill_host_ms += elapsed(begin)-call_ms+rows.host_ms;
                prefilled[i] = true;
                --rowed;
                }
            }
            if (prefill_limit_) prefill_cursor_ = cursor%slots_.size();
            if (!rowed) return true;
        }
        std::size_t remaining = 8, pending = rowed, index = 0;
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            auto& s = *slots_[i];
            if (s.out.status != Status::running || prefilled[i]) continue;
            s.out.peak_streams = std::max(s.out.peak_streams,live);
            Work w; w.slot = i; w.start = plans.size(); w.tx = s.state->begin();
            const auto position = w.tx->frontier();
            const auto capacity = remaining-(--pending);
            w.prefill = s.prompt_processed+1 < s.request.prompt.size();
            if (w.prefill) {
                w.rows = std::min(capacity,s.request.prompt.size()-1-s.prompt_processed);
            } else {
                const auto budget = s.request.max_tokens-s.out.ids.size();
                std::uint32_t k = 0;
                if (s.draft) {
                    const auto limit = std::min({s.request.draft_k,std::uint32_t(capacity-1),
                        std::uint32_t(std::min<std::size_t>(7,budget-1))});
                    // Multi-stream auto remains plain until a whole-batch estimator is
                    // available; independent per-stream rates cannot qualify batch gains.
                    if (s.request.dflash == DraftMode::on) k = limit;
                    // Drafter-confidence-gated C1 window: k is only the cap; the session picks the window below.
                    else if (live == 1 && s.confidence.enabled() && s.request.dflash == DraftMode::automatic &&
                             s.draft->confidence_capable()) { w.confident = true; k = std::min(limit,s.confidence.max_k); }
                    else if (live == 1) { w.decision = s.policy.choose(limit,position); k = w.decision.k; }
                    // Multi-stream: >1 live stream -> only the batch plan (k <= max_k; 0 for DFlash or when it loses).
                    else if (batch && plan.k[index]) {
                        k = std::min(plan.k[index],limit);
                        w.decision = {k,plan.probe,k ? s.policy.expected_tokens(position,k) : 1,plan.cost_ms,plan.gain};
                    }
                }
                k = dflash::admit_window(k,position,budget,s.dflash_enabled,
                    s.request.sampling.temperature != 0);
                if (position+7 >= backend_.config().max_positions) k = 0;
                w.k = k; w.rows = k+1;
                if (k) {
                    const auto begin = Clock::now();
                    const auto anchor = s.out.ids.empty() ? s.request.prompt.back() : s.out.ids.back();
                    try {
                        if (w.confident) {
                            // Exactness: the window only selects how many drafted tokens are verified; the greedy
                            // verdict decides every emitted token. No probes, no plain refresh on this path.
                            // Cost-aware long-context rule: this stream's measured draft cost per depth against its
                            // measured one-row pass (both from its own confidence cycles); the rule above otherwise.
                            const bool long_rule = s.conf_cost.long_rule(s.confidence);
                            const auto c = s.draft->propose_confident(anchor,position,k,
                                long_rule ? s.confidence.long_context() : s.confidence);
                            require(c.window <= c.drafted && c.drafted <= k, "invalid confidence window");
                            w.drafted = c.drafted; w.confidence_rule = c.rule;
                            s.out.confidence_long_cycles += long_rule && c.rule ? 1 : 0;
                            w.k = c.window; w.rows = w.k+1;
                            for (std::uint32_t d = 0; d < w.k; ++d) w.proposals[d] = c.tokens[d];
                            w.decision = {w.k,false,w.k ? s.policy.expected_tokens(position,w.k) : 1,0,0};
                            ++s.out.confidence_cycles; s.out.confidence_drafted += c.drafted;
                            s.out.confidence_unverified += c.drafted-c.window;
                        }
                        else w.proposals = s.draft->propose_window(anchor,position,k);
                    }
                    catch (...) { s.out.draft_ms += elapsed(begin); throw; }
                    w.draft_ms = elapsed(begin); s.out.draft_ms += w.draft_ms;
                    for (const auto id : w.proposals) require(id < backend_.config().vocab, "draft token outside vocabulary");
                }
            }
            const bool hidden = s.draft && s.draft->wants_hidden();
            // Logits leave the device only for sampled decode rows (greedy rows use
            // the backend's device winners), taps only for a drafter that reads them.
            const bool logits = !w.prefill && s.request.sampling.temperature != 0, taps = s.draft && s.draft->wants_taps();
            for (std::size_t r = 0; r < w.rows; ++r) {
                const auto token = w.prefill ? s.request.prompt[s.prompt_processed+r] :
                    (r ? w.proposals[r-1] : (s.out.ids.empty() ? s.request.prompt.back() : s.out.ids.back()));
                plans.push_back({i,s.generation,position+r,token,w.tx.get(),plans.size(),true,hidden,logits,taps});
            }
            remaining -= w.rows;
            ++index;
            work.push_back(std::move(w));
        }
        const auto target_start = Clock::now();
        const auto out = backend_.forward_rows(plans);
        const auto verify_ms = elapsed(target_start);
        // A backend that checked its outputs on the device may omit logits/taps no
        // row exported, and returns its greedy winners (the host tie rule).
        const bool wanted_logits = std::any_of(plans.begin(),plans.end(),[](const RowPlan& p) { return p.export_logits; });
        const bool wanted_taps = std::any_of(plans.begin(),plans.end(),[](const RowPlan& p) { return p.export_taps; });
        require(out.rows == plans.size() && out.vocab == backend_.config().vocab && out.tap_width == 5*backend_.config().hidden &&
            (out.logits.size() == out.rows*out.vocab || (out.device_checked && !wanted_logits && out.logits.empty())) &&
            (out.taps.size() == out.rows*out.tap_width || (out.device_checked && !wanted_taps && out.taps.empty())) &&
            (out.selected.empty() || out.selected.size() == out.rows) && (!out.device_checked || !out.selected.empty()),
            "invalid backend result layout");
        if (!out.device_checked) {
            for (const auto x : out.logits) require(std::isfinite(x), "nonfinite backend logits");
            for (const auto x : out.taps) require(std::isfinite(x), "nonfinite backend taps");
        }
        const bool any_hidden = std::any_of(plans.begin(),plans.end(),[](const RowPlan& p) { return p.export_hidden; });
        require(!any_hidden || out.hidden.size() == out.rows*backend_.config().hidden, "missing exported hidden rows");
        for (auto& w : work) {
            auto& s = *slots_[w.slot];
            std::size_t processed = w.rows;
            dflash::Verdict verdict;
            if (!w.prefill) {
                std::vector<std::uint32_t> selected;
                for (std::size_t r = 0; r < w.rows; ++r) {
                    const auto row = w.start+r;
                    if (s.request.sampling.temperature == 0 && out.device_checked) {
                        require(out.selected[row] < out.vocab, "device winner outside vocabulary");
                        selected.push_back(out.selected[row]);
                    }
                    else selected.push_back(select_token({out.logits.data()+row*out.vocab,out.vocab},s.request.sampling,s.out.rng_draws));
                }
                verdict = dflash::verify_greedy({w.proposals.data(),w.k},selected,
                    s.request.max_tokens-s.out.ids.size(),s.request.stop_ids,&w.delta);
                processed = verdict.processed;
            }
            if (s.draft) {
                const auto begin = Clock::now();
                std::vector<std::uint64_t> positions;
                for (std::size_t r = 0; r < processed; ++r) positions.push_back(plans[w.start+r].position);
                // The token that follows each committed row. Prefill windows cover prompt[0..n-2] only (the
                // last prompt token is the first decode anchor, never a prefill row), so prompt_processed+r+1 <= n-1.
                // The first EMITTED token follows the anchor row and is staged by that decode cycle (verdict.emitted).
                std::vector<std::uint32_t> next;
                if (w.prefill) for (std::size_t r = 0; r < processed; ++r) {
                    require(s.prompt_processed+r+1 < s.request.prompt.size(), "prefill row has no following prompt token");
                    next.push_back(s.request.prompt[s.prompt_processed+r+1]);
                }
                else next.assign(verdict.emitted.begin(),verdict.emitted.begin()+std::ptrdiff_t(processed));
                const auto width = std::size_t(backend_.config().hidden);
                const std::span<const float> hidden = s.draft->wants_hidden() ?
                    std::span<const float>(out.hidden.data()+w.start*width,processed*width) : std::span<const float>{};
                s.draft->stage_rows(out.taps.empty() ? std::span<const float>{} :
                    std::span<const float>(out.taps.data()+w.start*out.tap_width,processed*out.tap_width),hidden,next,positions);
                if (!hidden.empty()) s.last_hidden.assign(hidden.end()-std::ptrdiff_t(width),hidden.end());
                w.context_ms += elapsed(begin);
            }
            const auto commit_start = Clock::now();
            w.tx->commit(processed);
            w.tx.reset();
            s.out.commit_ms += elapsed(commit_start);
            if (s.draft) {
                const auto begin = Clock::now();
                s.draft->publish();
                w.context_ms += elapsed(begin); s.out.context_ms += w.context_ms;
            }
            // As in the device cycle helper, statistics/output become visible
            // only after both target and draft publications have succeeded.
            if (!w.prefill) add(s.out.acceptance,w.delta);
            ++s.out.target_calls; s.out.target_rows += w.rows;
            s.out.frontier = s.state->frontier();
            if (w.prefill) s.prompt_processed += processed;
            else {
                s.out.ids.insert(s.out.ids.end(),verdict.emitted.begin(),verdict.emitted.end());
                ++s.out.windows[w.k];
                if (live > 1 && s.request.dflash == DraftMode::automatic) {
                    ++s.out.multistream_auto_cycles;
                    s.out.multistream_auto_proposals += w.k;
                    s.out.multistream_auto_max_window = std::max(s.out.multistream_auto_max_window,w.k);
                }
                s.out.probes += w.k && w.decision.probe ? 1 : 0;
                if (w.k && s.request.dflash == DraftMode::automatic && !w.decision.probe) {
                    ++s.admitted[w.k];
                    s.minimum_gain = s.minimum_gain > 0 ? std::min(s.minimum_gain,w.decision.gain) : w.decision.gain;
                }
                s.out.verify_ms += verify_ms;
                if (verdict.stopped) s.out.status = Status::stopped;
                else if (s.out.ids.size() == s.request.max_tokens) s.out.status = Status::length;
            }
        }
        const double ms = elapsed(start);
        for (const auto& w : work) {
            auto& s = *slots_[w.slot];
            if (w.prefill) s.out.prefill_ms += ms; else s.out.decode_ms += ms;
        }
        // Only complete C1 cycles qualify auto. Diagnostic state hashing is
        // outside decode timing and runs once at completion/cancellation.
        if (work.size() == 1 && !work[0].prefill) {
            const auto& w = work[0]; auto& s = *slots_[w.slot];
            costs_->observe(1,w.k,ms,std::size_t(s.out.frontier-plans[w.start].position),plans[w.start].position,s.workload);
            if (s.draft) s.policy.observe(plans[w.start].position,w.k,ms,w.draft_ms,verify_ms,w.context_ms,w.delta,w.k && w.decision.probe);
            if (w.confident && w.confidence_rule && w.drafted && w.draft_ms > 0) {
                s.conf_cost.observe(w.draft_ms,w.drafted,verify_ms,w.k);
                s.out.confidence_draft_ratio = s.conf_cost.ratio();
            }
        }
        // Multi-stream: complete all-decode C>1 steps are whole-batch evidence; drafted streams add acceptance/draft cost.
        if (batch && work.size() == live) {
            std::uint32_t extra = 0;
            for (const auto& w : work) {
                extra += w.k;
                if (w.k) slots_[w.slot]->policy.observe_multi(plans[w.start].position,w.k,w.draft_ms,w.delta);
            }
            multi_.observe(live,extra,ms,plan.probe);
        }
        for (const auto& w : work) {
            auto& s = *slots_[w.slot];
            if (s.out.status != Status::running) s.out.digest = s.state->digest();
        }
    } catch (const std::exception& e) {
        // Publication/append failure invalidates affected requests. Never resume
        // a target whose committed frontier and drafter context might disagree.
        for (auto& s : slots_) {
            if (s->draft) s->draft->discard();
            if (s->out.status != Status::running) continue;
            s->out.status = Status::failed; s->out.error = e.what();
            s->out.frontier = s->state->frontier();
            // A stream poisoned by the failure has no canonical state to hash; the request still fails cleanly.
            try { s->out.digest = s->state->digest(); } catch (const std::exception&) { s->out.digest.clear(); }
        }
    }
    return true;
}
void Coordinator::run() { while (step()) {} }
Json Coordinator::receipt(std::size_t slot) const {
    const auto& s = *slots_.at(slot); const auto& r = s.out;
    Json j = effective_config(backend_,s.request,bool(drafts_),r.peak_streams), acceptance = Json::object(), timings = Json::object();
    j.set("rng_draws",r.rng_draws);
    auto switches = Json(j.find("off_switches")->as_object());
    switches.set("STRIX_G4_DFLASH",s.dflash_enabled);
    j.set("off_switches",switches); j.set("finish_reason",status_name(r.status)); j.set("frontier",r.frontier);
    j.set("state_digest",r.status == Status::running ? s.state->digest() : r.digest);
    j.set("target_calls",r.target_calls); j.set("target_rows",r.target_rows);
    const auto& a = r.acceptance;
    acceptance.set("cycles",a.cycles); acceptance.set("attempted",a.attempted); acceptance.set("eligible",a.eligible);
    acceptance.set("accepted",a.accepted); acceptance.set("emitted",a.emitted); acceptance.set("all_accepted",a.all_accepted);
    const auto array = [](const auto& values) { Json v = Json::array(); for (const auto n : values) v.push(n); return v; };
    acceptance.set("depth_eligible",array(a.depth_eligible)); acceptance.set("depth_accepted",array(a.depth_accepted));
    acceptance.set("rejection_depth",array(a.rejection_depth)); acceptance.set("windows",array(r.windows));
    acceptance.set("calibration_probes",r.probes); acceptance.set("auto_admitted_windows",array(s.admitted));
    acceptance.set("multistream_auto_cycles",r.multistream_auto_cycles);
    acceptance.set("multistream_auto_proposals",r.multistream_auto_proposals);
    acceptance.set("multistream_auto_max_window",std::uint64_t(r.multistream_auto_max_window));
    if (s.confidence.enabled()) {
        acceptance.set("confidence_cycles",r.confidence_cycles); acceptance.set("confidence_drafted",r.confidence_drafted);
        acceptance.set("confidence_unverified",r.confidence_unverified);
        acceptance.set("confidence_long_cycles",r.confidence_long_cycles);
        acceptance.set("confidence_draft_ratio",r.confidence_draft_ratio);
        j.set("mtp_confidence_resolved",s.confidence.json()); // tau + source (env, drafter-config or builtin)
    }
    j.set("minimum_admitted_gain_estimate",s.minimum_gain);
    j.set("auto_evidence",s.policy.receipt());
    if (multi_.config().enabled) j.set("multistream_evidence",multi_.receipt());
    j.set("acceptance",acceptance);
    if (s.draft) s.draft->report(j);
    if (!r.prefill_levers.empty()) {
        auto levers = Json::object();
        for (const auto& [name,chunks] : r.prefill_levers) levers.set(name,chunks);
        j.set("prefill_levers",levers);
    }
    if (!r.prefill_taps.empty()) j.set("prefill_taps",r.prefill_taps);
    if (r.cached_tokens) j.set("cached_tokens",r.cached_tokens);
    timings.set("prefill_ms",r.prefill_ms); timings.set("decode_ms",r.decode_ms); timings.set("draft_ms",r.draft_ms); timings.set("context_ms",r.context_ms);
    timings.set("verify_ms",r.verify_ms); timings.set("commit_ms",r.commit_ms);
    if (r.prefill_calls) timings.set("prefill_host_ms",r.prefill_host_ms);
    if (r.prefill_batched_calls) {
        j.set("prefill_batched_calls",r.prefill_batched_calls); j.set("prefill_batch_streams",r.prefill_batch_streams);
    }
    timings.set("session_setup_ms",r.setup_ms);
    timings.set("state_hash_scope","completion/cancellation or explicit receipt; outside decode_ms");
    j.set("timings",timings);
    if (!r.error.empty()) j.set("error",r.error);
    return j;
}
} // namespace strix::gemma4::engine
