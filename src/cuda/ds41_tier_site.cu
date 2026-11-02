#include "strix/cuda_ds41_tier_site.hpp"

#include "strix/cuda_ds41_hot_residency.hpp"
#include "strix/cuda_ds41_moe_v2.hpp"
#include "strix/cuda_ds41_parts_host.hpp"
#include "strix/ds41_rtx_schedule.hpp"

#include "tierstream/ts.h"

#include <algorithm>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include "strix_stream/ss_test.h"  // the pressure phases' knob (opened with dg_test_hooks only for them)

namespace strix::ds41::cuda {

namespace {

constexpr std::uint32_t kMaxPred = 64;  // rows * predict

void ts_check(ss_status s, const char* what) {
    if (s != TS_OK) throw std::runtime_error(std::string("ds41 tier site: ") + what + ": " + ts_last_error());
}

// desc-v1 of the routed experts of layers [0, layers): one moe layer per decoder layer, parts gate / up / down at
// their GGUF tensors (stride = bytes, so the library's arena stride is the engine's part size).
std::string descriptor(const GgufExpertReader& r, const std::string& model, std::uint32_t layers) {
    const MoeShape& s = r.shape();
    const std::uint64_t bytes[3] = {s.gate_expert_bytes(), s.gate_expert_bytes(), s.down_expert_bytes()};
    const char* names[3] = {"gate", "up", "down"};
    const char* formats[3] = {"IQ2_XXS", "IQ2_XXS", "Q2_K"};
    std::string path = model;
    std::replace(path.begin(), path.end(), '\\', '/');
    std::ostringstream j;
    j << R"({"schema":"tierstream/desc-v1","engine":"strix-ds41-replica","unit_align":256,"sources":[{"path":")" << path << R"(","align":4096}],"layers":[)";
    for (std::uint32_t l = 0; l < layers; ++l) {
        if (!r.has_layer(l)) throw std::runtime_error("ds41 tier site: the model has no routed experts at layer " + std::to_string(l));
        j << (l ? "," : "") << R"({"index":)" << l << R"(,"kind":"moe","n_units":)" << s.experts << R"(,"top_k":)" << s.top_k << R"(,"parts":[)";
        for (int p = 0; p < 3; ++p)
            j << (p ? "," : "") << R"({"name":")" << names[p] << R"(","format":")" << formats[p] << R"(","offset":)" << r.tensor_offset(l, p) << R"(,"stride":)"
              << bytes[p] << R"(,"bytes":)" << bytes[p] << "}";
        j << "]}";
    }
    j << "]}";
    return j.str();
}

}  // namespace

struct TierSite::Impl {
    TierSiteOptions o;
    MoeShape s;
    std::uint32_t L = 0, K = 0, predict = 0, residents = 0;
    ss_desc* desc = nullptr;
    ss_plan* plan = nullptr;
    ss_ctx* c = nullptr;
    ss_dg_device_view v{};
    std::vector<std::uint32_t> unit_base;  // host copy of the view's unit_base
    // the predictor runs on a forked branch of the step (captured with it)
    cudaStream_t side = nullptr;
    std::vector<cudaEvent_t> ev_fork, ev_join;
    DeviceBuffer<std::uint32_t> d_pred;
    DeviceBuffer<float> d_scores, d_pred_w;
    DeviceBuffer<std::uint8_t> workspace;
    std::vector<ss_dg_unit> seed;  // the seed keys as (gated, unit), read by ts_open
    ss_dg_stats before{};
    TierTokenStats last{};
    bool in_token = false;
    bool winpf = false;
    bool three_tier = false, inclusive = false;  // the file tier / the inclusive share are on

    Impl(const GgufExpertReader& r, const TierSiteOptions& opt) : o(opt), s(r.shape()), L(opt.layers), K(r.shape().top_k ? r.shape().top_k : 6) {
        s.top_k = K;
        if (o.max_rows == 0 || o.max_rows * K > SS_DG_MAX_IDS || o.max_rows * o.max_predict > kMaxPred) throw std::invalid_argument("ds41 tier site: rows / top_k / predict");
        const std::uint32_t E = s.experts;
        const std::uint64_t keys = std::uint64_t{L} * E;
        const std::uint32_t D = o.spare_slots;  // global, as the engine cache (dg_spare_slots)
        if (o.slots <= D || o.slots - D > keys || o.spare_target > D) throw std::invalid_argument("ds41 tier site: slots / spare target");
        residents = o.slots - D;
        // the library places the residents by this priority (then the plan's residents, then key order), and it is the
        // initial recency order, as the engine cache's seed (the plan's tiers below only give the counts)
        for (const std::uint32_t k : o.seed)
            if (k < keys) seed.push_back(ss_dg_unit{static_cast<std::int32_t>(k / E), static_cast<std::int32_t>(k % E)});

        const std::string json = descriptor(r, o.model, L);
        ts_check(ts_desc_from_json(json.data(), json.size(), &desc), "descriptor");
        // tiers (counts per layer): the first slots - spares distinct keys of the seed resident (then key order)
        std::vector<std::uint8_t> tiers(static_cast<std::size_t>(keys), static_cast<std::uint8_t>(TS_TIER_PINNED));
        std::uint32_t placed = 0;
        for (const std::uint32_t k : o.seed) {
            if (placed == residents) break;
            if (k < keys && tiers[k] != TS_TIER_RESIDENT) {
                tiers[k] = static_cast<std::uint8_t>(TS_TIER_RESIDENT);
                ++placed;
            }
        }
        for (std::uint64_t k = 0; k < keys && placed < residents; ++k)
            if (tiers[static_cast<std::size_t>(k)] != TS_TIER_RESIDENT) {
                tiers[static_cast<std::size_t>(k)] = static_cast<std::uint8_t>(TS_TIER_RESIDENT);
                ++placed;
            }
        // three-tier: the next pinned_units keys by the same priority stay pinned, every other non-resident key is
        // file-tier (rtx::tier_plan sized the counts); pinned_units >= the non-resident keys = the two-tier tiers
        three_tier = std::uint64_t{o.pinned_units} < keys - residents;
        if (three_tier) {
#ifndef SS_DG_DISK_TIER
            throw std::runtime_error("ds41 tier site: the file tier needs a tierstream with SS_DG_DISK_TIER");
#endif
            std::vector<std::uint8_t> pin(static_cast<std::size_t>(keys), 0);
            std::uint32_t pinned = 0;
            for (const std::uint32_t k : o.seed)
                if (pinned < o.pinned_units && k < keys && tiers[k] != TS_TIER_RESIDENT && !pin[k]) {
                    pin[k] = 1;
                    ++pinned;
                }
            for (std::uint64_t k = 0; k < keys && pinned < o.pinned_units; ++k)
                if (tiers[static_cast<std::size_t>(k)] != TS_TIER_RESIDENT && !pin[static_cast<std::size_t>(k)]) {
                    pin[static_cast<std::size_t>(k)] = 1;
                    ++pinned;
                }
            for (std::uint64_t k = 0; k < keys; ++k)
                if (tiers[static_cast<std::size_t>(k)] != TS_TIER_RESIDENT && !pin[static_cast<std::size_t>(k)])
                    tiers[static_cast<std::size_t>(k)] = static_cast<std::uint8_t>(TS_TIER_DISK);
        }
        inclusive = o.inclusive_units != 0;
#ifndef SS_DG_DISK_TIER
        if (inclusive) throw std::runtime_error("ds41 tier site: the inclusive share needs a tierstream with SS_DG_DISK_TIER");
#endif
        ss_budget b;
        ts_budget_defaults(&b);
        b.device_bytes = o.device_bytes;
        b.host_pinned_bytes = o.host_bytes;
        b.margin_gib = 0;  // the caller's budgets are already net of its margins
        b.margin_fraction = 0;
        b.tier_mode = TS_TIER_MODE_EXCLUSIVE;
        b.dynamic_slots_per_layer = 2;  // the plan's billing only: dg_spare_slots sets the arena's D
        b.ring_slots = 2U * o.max_rows * K + 1U;
        b.host_spare_units = o.host_spares;  // (+ the inclusive share: the replica adds it)
        ts_check(ts_plan_from_tiers(desc, &b, tiers.data(), tiers.size(), &plan), "plan");
        ss_options so;
        ts_options_defaults(&so);
#ifdef STRIX_HAVE_TIERSTREAM_MFILE
        if (o.weight_read.has_secondary(o.model)) {
            so.weight_dirs = o.weight_read.weight_dirs.c_str();
            so.drive_readers = o.weight_read.drive_readers;
            so.weight_verify = o.weight_read.weight_verify == "always" ? SS_MFILE_VERIFY_ALWAYS : SS_MFILE_VERIFY_CACHED;
        }
#else
        o.weight_read.require_supported(o.model);
#endif
        so.backend = SS_BACKEND_CUDA;
#if defined(SS_DG_SET_POLICY) && defined(SS_DG_AUTO_EVICT)
        so.policy = SS_POLICY_SCORED_AUTO;  // provision public score tables once; requests reset / select the policy
#elif defined(SS_DG_SET_POLICY) && defined(SS_DG_SCORED)
        so.policy = SS_POLICY_SCORED;
#endif
        so.forward_mode = TS_FORWARD_DEVICE_GATED;
        so.pin_planned = 0;
        so.verify_hash = 0;
        so.dg_max_ids = o.max_rows * K;
        so.dg_prefetch_per_layer = o.prefetch_per_layer;
        so.dg_prefetch_per_token = o.prefetch_per_token;
        so.dg_diag = o.diag ? 1U : 0U;
        so.dg_test_hooks = o.test_hooks ? 1U : 0U;
        so.dg_spare_target = o.spare_target;
        so.dg_spare_slots = D;
        so.dg_flush_ready = o.flush_ready ? 1U : 0U;
        so.dg_seed = seed.empty() ? nullptr : seed.data();
        so.dg_seed_n = static_cast<std::uint32_t>(seed.size());
        so.dg_prior = o.prior.empty() ? nullptr : o.prior.data();  // read during ts_open only
        so.dg_prior_len = o.prior.size();
        so.dg_wb_defer = o.wb_defer ? 1U : 0U;
        so.dg_wb_stream = o.wb_stream ? 1U : 0U;  // explicit both ways: an older tierstream's default was 0
#ifdef SS_DG_DISK_TIER
        if (three_tier) {  // only then: a full pinned tier keeps the two-tier library path byte for byte
            so.dg_disk_tier = 1U;
            so.dg_disk_prefetch = o.disk_prefetch ? 1U : 0U;
            so.dg_disk_staging = o.disk_staging;
#ifdef SS_DG_DISK_QUEUE
#ifdef SS_DG_DISK_QUEUE_AUTO
            so.dg_disk_queue = o.disk_queue == kDiskQueueAuto ? SS_DG_DISK_QUEUE_AUTO : o.disk_queue;
#else
            so.dg_disk_queue = o.disk_queue == kDiskQueueAuto ? 4U : o.disk_queue;
#endif
#endif
        }
        so.dg_inclusive_blocks = o.inclusive_units;
#endif
        if (o.ce_pacing) {
            const auto p = rtx::ce_pacing(static_cast<std::uint32_t>(s.gate_expert_bytes()), static_cast<std::uint32_t>(s.down_expert_bytes()));
            so.dg_wb_layer_bytes = p.layer_bytes;
            so.dg_wb_max_outstanding = p.outstanding_threshold;
        }
        ts_check(ts_open(desc, &b, plan, &so, &c), "open");
        ts_check(ts_dg_get_device_view(c, &v), "device view");
        if (v.struct_size < sizeof(ss_dg_device_view) || !v.absent_err) throw std::runtime_error("ds41 tier site: the library's view has no absent_err (needs rev 3.4)");
        // the routed GEMVs derive the slot stride from the shape: the library's arena must match it exactly
        const std::uint64_t want[3] = {s.gate_expert_bytes(), s.gate_expert_bytes(), s.down_expert_bytes()};
        if (v.protocol != SS_DG_PROTOCOL || v.n_parts != 3 || v.n_gated != L) throw std::runtime_error("ds41 tier site: unexpected device view");
        for (int p = 0; p < 3; ++p)
            if (v.parts[p].stride != want[p] || v.parts[p].bytes != want[p]) throw std::runtime_error("ds41 tier site: part " + std::to_string(p) + " stride != the engine's part bytes");
        for (std::uint32_t l = 0; l < L; ++l)
            if (ts_dg_gated_index(c, static_cast<int>(l)) != static_cast<int>(l)) throw std::runtime_error("ds41 tier site: layer " + std::to_string(l) + " is not gated in order");
        unit_base.resize(L + 1U);
        STRIX_CUDA_CHECK(cudaMemcpy(unit_base.data(), v.unit_base, (L + 1U) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
        for (std::uint32_t l = 0; l <= L; ++l)
            if (unit_base[l] != l * E) throw std::runtime_error("ds41 tier site: unit_base");

        STRIX_CUDA_CHECK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
        ev_fork.resize(L);
        ev_join.resize(L);
        for (std::uint32_t l = 0; l < L; ++l) {
            STRIX_CUDA_CHECK(cudaEventCreateWithFlags(&ev_fork[l], cudaEventDisableTiming));
            STRIX_CUDA_CHECK(cudaEventCreateWithFlags(&ev_join[l], cudaEventDisableTiming));
        }
        d_pred.resize(std::size_t{L} * kMaxPred);
        d_pred.zero();
        d_scores.resize(std::size_t{o.max_rows} * s.experts);
        d_pred_w.resize(kMaxPred);
        workspace.resize(moe_layer_workspace_bytes(o.max_rows, s));
        ts_check(ts_stats_dg(c, &before), "stats");
    }
    ~Impl() {
        for (auto e : ev_fork)
            if (e) (void)cudaEventDestroy(e);
        for (auto e : ev_join)
            if (e) (void)cudaEventDestroy(e);
        if (side) (void)cudaStreamDestroy(side);
        ts_close(c);
        ts_plan_free(plan);
        ts_desc_free(desc);
    }
};

TierSite::TierSite(const GgufExpertReader& reader, const TierSiteOptions& options) : impl_(std::make_unique<Impl>(reader, options)) {}

TierSite::~TierSite() = default;

void TierSite::enqueue(std::uint32_t l, std::uint32_t rows, const A8Rows& x, const std::uint32_t* ids, const float* weights, float* shared, float* raw, float* y,
                       const std::function<void(cudaStream_t)>& shared_expert,
                       const std::function<void(std::uint32_t, std::uint32_t, float*, std::uint32_t*, float*, cudaStream_t)>& route_other, cudaStream_t st,
                       cudaEvent_t shared_join, const float* hc_residual, const float* hc_mixes, float* hc_out, float* hc_pre_carry) {
    auto& m = *impl_;
    if (rows == 0 || rows > m.o.max_rows || l >= m.L) throw std::invalid_argument("ds41 tier site: rows / layer");
    const auto& s = m.s;
    const int g = static_cast<int>(l);
    if (l == 0U) ts_check(ts_dg_enqueue_begin(m.c, st), "enqueue begin");
    int n_pred = 0;  // > 0: the export promises a prediction for layer l+1 from the side branch
    const bool fork = m.predict && l + 1U < m.L && route_other;
    if (fork) {  // layer l+1's router on this layer's input, in parallel with the export / shared expert / routed experts
        std::uint32_t* pred = m.d_pred.get() + std::size_t{l + 1U} * kMaxPred;
        STRIX_CUDA_CHECK(cudaEventRecord(m.ev_fork[l], st));
        STRIX_CUDA_CHECK(cudaStreamWaitEvent(m.side, m.ev_fork[l], 0));
        route_other(l + 1U, m.predict, m.d_scores.get(), pred, m.d_pred_w.get(), m.side);
        n_pred = static_cast<int>(rows * m.predict);
        ts_check(ts_dg_enqueue_predict(m.c, g, reinterpret_cast<const std::int32_t*>(pred), n_pred, m.side), "enqueue predict");
        STRIX_CUDA_CHECK(cudaEventRecord(m.ev_join[l], m.side));
    }
    ts_check(ts_dg_enqueue_export(m.c, g, reinterpret_cast<const std::int32_t*>(ids), static_cast<int>(rows * m.K), n_pred, st), "enqueue export");
    // the shared expert (its own scratch) runs while the service reacts to a miss (or on the engine's side branch)
    if (shared_expert && !shared_join) shared_expert(st);
    ts_check(ts_dg_enqueue_wait(m.c, g, st), "enqueue wait");
    // every routed slot is in the library's arena now: the all-resident MoE with id -> slot routes, guarded like
    // ts_dg_slot (abort -> the zeroed guard slot; an entry < 0 -> the view's absent_err + guard), so a refused or
    // unauthorized step never reads a live slot; ts_end then fails the token.
    const MoeLayerScratch scratch = moe_layer_workspace_views(m.workspace.get(), m.o.max_rows, s);
    ExpertWeights w;
    w.gate = static_cast<const std::uint8_t*>(m.v.parts[0].base);
    w.up = static_cast<const std::uint8_t*>(m.v.parts[1].base);
    w.down = static_cast<const std::uint8_t*>(m.v.parts[2].base);
    ResidentRoutesView rv;
    rv.ids = ids;
    rv.slot_of = m.v.slot_of + m.unit_base[l];
    rv.rows = rows;
    rv.top_k = m.K;
    rv.experts = s.experts;
    rv.split = m.o.split;
    rv.out = scratch.routes;
    rv.abort = m.v.abort;
    rv.guard_slot = static_cast<std::int32_t>(m.v.guard_slot);
    rv.absent_err = m.v.absent_err;
    enqueue_resident_routes(rv, st);
    const std::uint32_t max_union = std::min(rows * m.K, s.experts);
    RoutedGateUpView gu;
    gu.shape = s;
    gu.weights = w;
    gu.routes = scratch.routes;
    gu.max_union = max_union;
    gu.max_rows = rows;
    gu.x = x;
    gu.act = scratch.act;
    // v1b gate/up from 2 rows (at 1 row v1 is 2 us faster on the PRO), v3 down always; same bits
    const bool v1b3 = m.o.moe_v1b3 && routed_v3_supported(s, rows);
    if (v1b3 && rows > 1U) enqueue_routed_gate_up_v1b(gu, st);
    else enqueue_routed_gate_up(gu, st);
    RoutedDownView d;
    d.shape = s;
    d.weights = w;
    d.routes = scratch.routes;
    d.max_union = max_union;
    d.max_rows = rows;
    d.act = scratch.act;
    d.raw = raw;
    if (v1b3) enqueue_routed_down_v3(d, st);
    else enqueue_routed_down(d, st);
    MoeCombineView c;
    c.raw = raw;
    c.shared = shared;
    c.weights = weights;
    c.order = scratch.routes.order;
    c.low_counts = scratch.routes.low_counts;
    c.rows = rows;
    c.top_k = m.K;
    c.hidden = s.hidden;
    c.out = y;
    c.hc_residual = hc_residual;
    c.hc_mixes = hc_mixes;
    c.hc_out = hc_out;
    c.hc_pre_carry = hc_pre_carry;
    if (shared_join) STRIX_CUDA_CHECK(cudaStreamWaitEvent(st, shared_join, 0));  // the shared expert's side branch
    enqueue_moe_combine(c, st);
    if (fork) STRIX_CUDA_CHECK(cudaStreamWaitEvent(st, m.ev_join[l], 0));  // join before the next layer's export
    if (l + 1U == m.L) ts_check(ts_dg_enqueue_end(m.c, st), "enqueue end");
}

void TierSite::set_predict(std::uint32_t k) {
    if (impl_->o.max_rows * k > kMaxPred || k > impl_->o.max_predict) throw std::invalid_argument("ds41 tier site: predict");
    impl_->predict = k;
}

void TierSite::set_winpf(const TierWinpf& w) {
    auto& m = *impl_;
    if (m.in_token) throw std::logic_error("ds41 tier site: set_winpf inside a token");
    ss_dg_winpf_params p;
    ts_dg_winpf_defaults(&p);
    p.enable = w.enable ? 1U : 0U;
    p.top_k = m.K;  // an export holds rows x top_k ids, the prediction rows x predict (row-major)
    p.l1_cap = w.l1_cap;
    p.l1_pmin = w.l1_pmin;
    p.prior_cap = w.prior_cap;
    p.prior_pmin = w.prior_pmin;
    p.revert_unused = w.revert_unused ? 1U : 0U;
    ts_check(ts_dg_winpf_set(m.c, &p), "winpf");
    m.winpf = w.enable;
}

void TierSite::hint_window(const std::uint32_t* tokens, std::uint32_t rows) {
    auto& m = *impl_;
    if (m.in_token) throw std::logic_error("ds41 tier site: hint_window inside a token");
    if (!m.winpf || rows == 0) return;
    std::int32_t sym[SS_DG_MAX_IDS];
    rows = std::min<std::uint32_t>(rows, SS_DG_MAX_IDS);
    for (std::uint32_t r = 0; r < rows; ++r) sym[r] = static_cast<std::int32_t>(tokens[r]);
    ts_check(ts_dg_window_hint(m.c, sym, nullptr, rows), "window hint");
}

bool TierSite::winpf() const noexcept { return impl_->winpf; }

void TierSite::set_pressure(std::uint32_t free_per_token, std::uint32_t evict_recent) {
    ts_check(ss_test_dg_pressure(impl_->c, free_per_token, evict_recent), "pressure");
}

void TierSite::set_budgets(std::uint32_t spare_target, std::uint32_t prefetch_per_layer, std::uint32_t prefetch_per_token) {
    ts_check(ts_dg_set_budgets(impl_->c, spare_target, prefetch_per_layer, prefetch_per_token), "budgets");
}

void TierSite::begin_token() {
    auto& m = *impl_;
    if (m.in_token) throw std::logic_error("ds41 tier site: token already open");
    ts_check(ts_begin(m.c, nullptr, 0), "begin");
    m.in_token = true;
}

void TierSite::end_token(std::uint32_t rows) {
    auto& m = *impl_;
    if (!m.in_token) throw std::logic_error("ds41 tier site: no token open");
    m.in_token = false;
#ifdef SS_DG_AUTO_EVICT
    ts_check(ts_dg_end_rows(m.c, rows), "token not accepted");
#else
    (void)rows;
    ts_check(ts_end(m.c), "token not accepted");
#endif
}

std::vector<TierLayer> TierSite::collect() {
    auto& m = *impl_;
    if (m.in_token) throw std::logic_error("ds41 tier site: collect inside a token");
    ss_dg_stats now{};
    ts_check(ts_stats_dg(m.c, &now), "stats");
    const auto d = [](std::uint64_t a, std::uint64_t b) { return static_cast<std::uint32_t>(a - b); };
    TierTokenStats t;
    t.lookups = d(now.lookups, m.before.lookups);
    t.misses = d(now.misses, m.before.misses);
    t.covered = d(now.covered, m.before.covered);
    t.fetched = d(now.fetched, m.before.fetched);
    t.prefetched = d(now.prefetched, m.before.prefetched);
    t.resurrected = d(now.resurrected, m.before.resurrected);
    t.ring = d(now.bank_serves, m.before.bank_serves);
    t.stalled = d(now.stalled, m.before.stalled);
    t.collected_mid = d(now.collected_mid, m.before.collected_mid);
    t.evicted = d(now.evicted, m.before.evicted);
    t.writebacks_done = d(now.writebacks_done, m.before.writebacks_done);
    t.free_slots = static_cast<std::uint32_t>(now.free_slots);
    t.miss_layers = d(now.miss_layers, m.before.miss_layers);
    t.predicted = d(now.predicted, m.before.predicted);
    t.predicted_true = d(now.predicted_true, m.before.predicted_true);
    t.prior_issued = d(now.prior_issued, m.before.prior_issued);
    t.prior_used = d(now.prior_used, m.before.prior_used);
    t.prior_late = d(now.prior_late, m.before.prior_late);
    t.prior_reverted = d(now.prior_reverted, m.before.prior_reverted);
    t.l1_issued = d(now.l1_issued, m.before.l1_issued);
    t.l1_used = d(now.l1_used, m.before.l1_used);
    t.l1_late = d(now.l1_late, m.before.l1_late);
    t.l1_reverted = d(now.l1_reverted, m.before.l1_reverted);
    t.l1_priced_out = d(now.l1_priced_out, m.before.l1_priced_out);
    t.l1_skipped_ahead = d(now.l1_skipped_ahead, m.before.l1_skipped_ahead);
    t.prior_budget_stops = d(now.prior_budget_stops, m.before.prior_budget_stops);
    t.wb_issued_bytes = now.wb_issued_bytes - m.before.wb_issued_bytes;
    t.wb_outstanding_start_bytes = m.before.wb_outstanding_bytes;
    t.wb_outstanding_bytes = now.wb_outstanding_bytes;
    t.wb_completed_bytes = now.wb_completed_bytes - m.before.wb_completed_bytes;
    t.wb_cancelled_bytes = now.wb_cancelled_bytes - m.before.wb_cancelled_bytes;
    t.wb_boundary_flushed_bytes = now.wb_boundary_flushed_bytes - m.before.wb_boundary_flushed_bytes;
    t.wb_forced_wait_ns = now.wb_forced_wait_ns - m.before.wb_forced_wait_ns;
    t.wb_deferred = now.wb_deferred - m.before.wb_deferred;
    t.wb_cancelled = now.wb_cancelled - m.before.wb_cancelled;
    t.wb_forced = now.wb_forced - m.before.wb_forced;
    t.wb_boundary_flushed = now.wb_boundary_flushed - m.before.wb_boundary_flushed;
    t.device_timeouts = now.device_timeouts - m.before.device_timeouts;
#ifdef SS_DG_DISK_TIER
    t.disk_reads = now.disk_reads - m.before.disk_reads;
    t.disk_read_bytes = now.disk_read_bytes - m.before.disk_read_bytes;
    t.disk_stage_waits = now.disk_stage_waits - m.before.disk_stage_waits;
    t.disk_drops = now.disk_drops - m.before.disk_drops;
#endif
#ifdef SS_DG_DISK_QUEUE
    t.disk_wait_ns = now.disk_wait_ns - m.before.disk_wait_ns;
#endif
#ifdef SS_DG_DISK_RECEIPTS
    t.disk_carried = now.disk_carried - m.before.disk_carried;
#endif
    ss_dg_boundary_timeline bt{};
    ts_check(ts_dg_boundary_timeline_sized(m.c, &bt, sizeof bt), "boundary timeline");
    t.boundary_sync_us = bt.sync_us;
    t.boundary_select_us = bt.select_us;
    t.boundary_upload_us = bt.upload_us;
    t.boundary_issue_us = bt.issue_us;
    if (now.device_timeouts != m.before.device_timeouts) throw std::runtime_error("ds41 tier site: device wait timeout (token not accepted)");
    m.before = now;
    std::vector<TierLayer> layers(m.L);
    if (m.o.diag)
        for (std::uint32_t l = 0; l < m.L; ++l) {
            ss_dg_layer_timeline tl{};
            ts_check(ts_dg_timeline_sized(m.c, static_cast<int>(l), &tl, sizeof tl), "timeline");
            auto& o = layers[l];
            o.ids = tl.ids;
            o.misses = tl.misses;
            o.covered = tl.covered;
            o.fetched = tl.fetched;
            o.prefetched = tl.prefetched;
            o.seen_us = tl.seen_us;
            o.serve_us = tl.serve_us;
            o.demand_us = tl.demand_us;
            o.pred_wait_us = tl.pred_wait_us;
            o.prefetch_us = tl.prefetch_us;
            o.predicted_true = tl.predicted_true;
            o.dma_ms = tl.dma_ms;
            o.covered_waits = tl.covered_waits;
            o.gpu_demand_wait_us = tl.gpu_wait_us;
            o.export_to_ready_us = tl.gpu_export_ready_us;
            o.dma_first_ms = tl.dma_first_ms;
            o.wb_left_ms = tl.wb_left_ms;
        }
    m.last = t;
    return layers;
}

const TierTokenStats& TierSite::stats() const noexcept { return impl_->last; }

bool TierSite::can_disk_tier() noexcept {
#ifdef SS_DG_DISK_TIER
    return true;
#else
    return false;
#endif
}

std::string TierSite::tier3_json() const {
    const auto& m = *impl_;
    if (!m.three_tier && !m.inclusive) return {};
#ifdef SS_DG_DISK_TIER
    ss_dg_stats s{};
    ts_check(ts_stats_dg(m.c, &s), "stats");
    std::ostringstream j;
    j << "\"file_tier\":" << (m.three_tier ? "true" : "false") << ",\"disk_reads\":" << s.disk_reads << ",\"disk_read_bytes\":" << s.disk_read_bytes
      << ",\"disk_read_ms\":" << static_cast<double>(s.disk_read_ns) / 1e6 << ",\"disk_stage_waits\":" << s.disk_stage_waits
      << ",\"disk_drops\":" << s.disk_drops << ",\"disk_discards\":" << s.disk_discards << ",\"inclusive_kept\":" << s.inclusive_kept
      << ",\"inclusive_drops\":" << s.inclusive_drops << ",\"disk_units\":" << s.disk_units << ",\"backed_units\":" << s.backed_units
      << ",\"tokens\":" << s.tokens << ",\"misses\":" << s.misses << ",\"wb_issued_bytes\":" << s.wb_issued_bytes;
    return j.str();
#else
    return {};
#endif
}

std::vector<std::uint32_t> TierSite::residency() const {
    std::vector<std::uint32_t> out;
#ifdef SS_DG_RESIDENCY
    auto& m = *impl_;
    if (m.in_token) throw std::logic_error("ds41 tier site: residency inside a token");
    std::uint32_t total = 0;
    ts_check(ss_dg_residency(m.c, nullptr, 0, &total), "residency");
    std::vector<ss_dg_unit> u(total);
    ts_check(ss_dg_residency(m.c, u.data(), total, &total), "residency");
    if (m.unit_base.size() < 2U) throw std::runtime_error("ds41 tier site: no unit base");
    const std::uint32_t E = m.unit_base[1] - m.unit_base[0];
    for (std::uint32_t i = 0; i < total && i < u.size(); ++i) out.push_back(static_cast<std::uint32_t>(u[i].gated) * E + static_cast<std::uint32_t>(u[i].unit));
#endif
    return out;
}

bool TierSite::can_set_sched() noexcept {
#ifdef SS_DG_SET_SCHED
    return true;
#else
    return false;
#endif
}

bool TierSite::can_set_policy() noexcept {
#ifdef SS_DG_SET_POLICY
    return true;
#else
    return false;
#endif
}

bool TierSite::can_auto_evict() noexcept {
#if defined(SS_DG_AUTO_EVICT) && defined(SS_DG_SET_POLICY)
    return true;
#else
    return false;
#endif
}

void TierSite::set_policy(std::uint32_t policy) {
    auto& m = *impl_;
    if (m.in_token) throw std::logic_error("ds41 tier site: set_policy inside a token");
#ifdef SS_DG_SET_POLICY
    if (policy == 2 && !can_auto_evict())
        throw std::runtime_error("--evict auto: this tierstream has no ts_dg_end_rows; rebuild against a supporting library");
    ts_check(ts_dg_set_policy(m.c, static_cast<ss_policy>(policy)), "set_policy");
#else
    if (policy) throw std::runtime_error("--evict scored: this tierstream has no ts_dg_set_policy; rebuild against a supporting library");
#endif
}

void TierSite::set_sched(bool wb_defer, bool flush_ready) {
    auto& m = *impl_;
    if (m.in_token) throw std::logic_error("ds41 tier site: set_sched inside a token");
#ifdef SS_DG_SET_SCHED
    ts_check(ts_dg_set_sched(m.c, wb_defer ? 1U : 0U, flush_ready ? 1U : 0U), "set_sched");
#else
    if (wb_defer != m.o.wb_defer || flush_ready != m.o.flush_ready) throw std::runtime_error("ds41 tier site: this tierstream has no ts_dg_set_sched");
#endif
}

bool TierSite::can_reset_cache() noexcept {
#ifdef SS_DG_RESET_OPEN
    return true;
#else
    return false;
#endif
}

void TierSite::reset_cache() {
    auto& m = *impl_;
    if (m.in_token) throw std::logic_error("ds41 tier site: reset_cache inside a token");
#ifdef SS_DG_RESET_OPEN
    ts_check(ts_dg_reset_open(m.c), "reset_open");
    ts_check(ts_stats_dg(m.c, &m.before), "stats");  // the per-token deltas start from the rebased counters
    m.last = TierTokenStats{};
#else
    throw std::runtime_error("ds41 tier site: this tierstream has no ts_dg_reset_open");
#endif
}

std::uint32_t TierSite::verify(std::uint32_t count, std::uint64_t seed) {
    auto& m = *impl_;
    std::mt19937_64 rng(seed);
    std::uint32_t bad = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        const int layer = static_cast<int>(rng() % m.L), unit = static_cast<int>(rng() % m.s.experts);
        const ss_status st = ts_check_unit(m.c, layer, unit);
        if (st == SS_E_HASH) ++bad;
        else ts_check(st, "check unit");
    }
    return bad;
}

std::uint32_t TierSite::device_experts() const noexcept { return impl_->residents; }

std::uint64_t TierSite::device_bytes() const noexcept { return std::uint64_t{impl_->v.slots} * impl_->s.expert_bytes(); }

std::uint64_t TierSite::host_bytes() const noexcept { return impl_->before.blocks * impl_->s.expert_bytes(); }

}  // namespace strix::ds41::cuda
