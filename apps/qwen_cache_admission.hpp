#pragma once
#include "strix/qpf2k_defaults.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>

namespace strix::decode_cli {
enum class CacheRunner { pending, cli, target, single_mtp, batch_mtp };
enum class CacheUse { prefix, batch_l0, batch_l1, decline_compatibility, session };
struct CacheSchedulerFamily {
    std::uint64_t schedule=12, pipe=255, lookahead=2, short_rows=2048;
    double service_ms=250, prompt_share=0;
    std::uint64_t nest_rows=0, interleave=1, group_rows=0;
    bool groups=false, cohort=false, small=false, group_owner=false, windows=true, cancel=true;
    bool priming=false, worker=false, tiled_start=false, merge=false, balance=false;
    std::string shared="none";
    bool operator==(const CacheSchedulerFamily&) const = default;
};
// Full asset digests, never paths or an executable hash embedded in its own source.
// Empty identities are unknown, not wildcards. Evidence binds the final executable.
struct CacheFamilyIdentity {
    std::string model, layout, tokenizer, sidecar, golden, replay, build_family, gfx, numerical_contract;
    std::string kv_precision, recurrent_precision, prefill_profile, decode_profile, output_policy, profile_pair;
    bool mtp_contract_qualified=true;
    std::vector<std::uint32_t> tails;
    CacheSchedulerFamily scheduler;
    bool operator==(const CacheFamilyIdentity&) const = default;
    bool complete() const {
        for (const auto* id : {&model,&layout,&tokenizer,&sidecar,&golden,&replay,&build_family,&gfx,
                &kv_precision,&recurrent_precision,&prefill_profile,&decode_profile,&output_policy,&profile_pair,&numerical_contract})
            if (id->empty()) return false;
        return true;
    }
};
// The schedule a runner actually executes: STRIX_QPF2K_SCHED bit 2 runs only on the CLI known-prompt cursor, and
// there not with a defaulted schedule under a prefix-state cache (the only CLI case that resolves a family). Other
// runners never defer, so bit 2 is masked before any qualification comparison; nothing is re-qualified.
inline bool cache_runner_defers(CacheRunner runner) {
    return runner==CacheRunner::cli && qpf2k::cli_sched_defers(true);
}
inline CacheFamilyIdentity effective_cache_family(CacheFamilyIdentity id, CacheRunner runner) {
    id.scheduler.schedule=qpf2k::effective_sched(id.scheduler.schedule,cache_runner_defers(runner));
    return id;
}
struct CacheQualification {
    std::string_view family, route, evidence_id;
    CacheUse use;
    CacheRunner runner;
    std::uint32_t n_min=1, n_max=1, context_min=1, context_max=8192, tile=2048;
    bool gpu_qualified=false, cost_qualified=false, resource_qualified=false;
    unsigned version=1;
    CacheFamilyIdentity identity{};
    bool matches(CacheRunner r, std::uint32_t n, std::uint64_t context, std::uint32_t t,
                 const CacheFamilyIdentity& actual) const {
        return runner==r && n>=n_min && n<=n_max && context>=context_min && context<=context_max &&
            tile==t && identity.complete() && actual.complete() && identity==actual;
    }
    bool qualified() const { return gpu_qualified && cost_qualified && resource_qualified && !evidence_id.empty(); }
};
// Candidate identity resolved from full asset bytes and the effective production
// configuration. Qualification is a separate row switch; native equality, capacity
// and cost gates must still establish release evidence for the final executable.
// Layout/signature use the canonical 16384 capacity; matches checks actual context.
inline const CacheFamilyIdentity single_t2048_default_identity = [] {
    CacheFamilyIdentity id;
    id.model="8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2";
    id.layout="8317644047050514147:1602021066712137840";
    id.tokenizer="8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2";
    id.sidecar="5516d3ce407783a424315b6a2a0d8b06341665bb3541b98e5db0fe944add4c3c";
    id.golden="5db345508f79bda1af87a8c929ea463ca914f17dec3dd3b02273daf6ed7ce413";
    id.replay="5db345508f79bda1af87a8c929ea463ca914f17dec3dd3b02273daf6ed7ce413";
    id.build_family="15.2.0:71626385:71626385";
    id.gfx="gfx1151:110501";
    id.numerical_contract="2bb43351b6e3204ec79fae1718f4ae8e9c78d86e986c1f17d510dfd21c1fe454";
    id.kv_precision="0";
    id.recurrent_precision="0";
    id.prefill_profile="hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1";
    id.decode_profile="hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1";
    id.output_policy="1";
    id.tails={1024,256,64,16};
    id.scheduler.schedule=12;
    id.scheduler.pipe=0;
    id.scheduler.lookahead=2;
    id.scheduler.short_rows=2048;
    id.scheduler.service_ms=250;
    id.scheduler.prompt_share=0;
    id.scheduler.nest_rows=2048;
    id.scheduler.interleave=1;
    id.scheduler.group_rows=4294967295;
    id.scheduler.groups=true;
    id.scheduler.cohort=true;
    id.scheduler.small=false;
    id.scheduler.group_owner=false;
    id.scheduler.windows=true;
    id.scheduler.cancel=true;
    id.scheduler.priming=false;
    id.scheduler.worker=false;
    id.scheduler.tiled_start=true;
    id.scheduler.merge=false;
    id.scheduler.balance=true;
    id.scheduler.shared="hc,router,moe,elementwise";
    id.profile_pair="6222990063351448574";
    return id;
}();
inline const CacheQualification automatic_prefix_auto_route_qualifications[] = {
    {"single-t2048-default","http-mtp-fixed","QPC-SINGLE-v1",CacheUse::prefix,CacheRunner::single_mtp,1,1,1,16384,2048,true,true,true,1,single_t2048_default_identity},
    {"strict-l0-n1","http-mtp-batch-l0","QPC-B1-v1",CacheUse::prefix,CacheRunner::batch_mtp,1,1},
    {"strict-l0-n2","http-mtp-batch-l0","QPC-B2-v1",CacheUse::prefix,CacheRunner::batch_mtp,2,2},
    {"strict-l0-n4","http-mtp-batch-l0","QPC-B4-v1",CacheUse::prefix,CacheRunner::batch_mtp,4,4},
    {"target","http-target","QPC-TARGET-v1",CacheUse::prefix,CacheRunner::target},
    {"cli-reuse","cli-target-tiled","QPC-CLI-v1",CacheUse::prefix,CacheRunner::cli}
};
inline const CacheQualification automatic_prefix_batch_cert_route_qualifications[] = {
    {"strict-l0-n1","http-mtp-batch-certified-l0","QBC-B1-v1",CacheUse::batch_l0,CacheRunner::batch_mtp,1,1},
    {"strict-l0-n2","http-mtp-batch-certified-l0","QBC-B2-v1",CacheUse::batch_l0,CacheRunner::batch_mtp,2,2},
    {"strict-l0-n4","http-mtp-batch-certified-l0","QBC-B4-v1",CacheUse::batch_l0,CacheRunner::batch_mtp,4,4},
    {"ordinary-scheduler-decline","http-mtp-batch","QBC-DECLINE-v1",CacheUse::decline_compatibility,CacheRunner::batch_mtp,1,4},
    {"broader-l1","http-mtp-batch-certified-l1","QBC-L1-v1",CacheUse::batch_l1,CacheRunner::batch_mtp,1,4}
};
inline const CacheQualification automatic_session_route_qualifications[] = {
    {"explicit-fixed-tiled","http-mtp-fixed","QSC-FIXED-v1",CacheUse::session,CacheRunner::single_mtp},
    {"adaptive","http-mtp-fixed","QSC-ADAPTIVE-v1",CacheUse::session,CacheRunner::single_mtp},
    {"target","http-target","QSC-TARGET-v1",CacheUse::session,CacheRunner::target},
    {"batch","http-mtp-batch","QSC-BATCH-v1",CacheUse::session,CacheRunner::batch_mtp,1,4}
};
struct CachePolicyActivation {
    bool automatic_path=false;
    bool prefix_requested=true, session_requested=true;
    unsigned cert_requested=1, cert_effective=0;
    bool cert_policy_parsed=false;
    bool prefix=false, session=false, automatic_qualified=false, finalized=false;
    std::string decision_phase; // Automatic-only; legacy serialization is unchanged.
    std::function<void(bool)> certification_transition;
    CacheRunner runner=CacheRunner::pending;
    std::string suppression;
    std::string source="off", reason="pending-post-residency", family, evidence_id;
    std::string cert_reason="pending-post-residency", session_reason="pending-post-residency";
    std::optional<std::uint64_t> U,L,A,reserve,observation_age_ns;
    double pool_setup_ms=0;
    bool limits_consistent=false;
    std::optional<std::uint64_t> B,S,H,F,E, context_off, slots_off, context_admitted, slots_admitted;
};
// One process-wide decline also reaches runtime Options copies. Numerical
// protocol flags and the protected OFF plan are deliberately outside this object.
inline void decline_cache_activation(CachePolicyActivation& a, std::string_view reason) {
    if(a.certification_transition) a.certification_transition(false);
    a.prefix=false; a.session=false; a.cert_effective=0; a.B=0; a.reason=reason;
    if(a.cert_requested && a.suppression.empty()) a.cert_reason=reason;
}
// OFF's full lazy growth is a commitment even when its slot has no owners yet.
inline std::uint64_t cache_future_slot_bytes(std::uint64_t retained, std::uint64_t native,
    std::uint64_t slots, std::uint64_t states_built, std::uint64_t native_built) {
    if(states_built>slots || native_built>slots) throw std::overflow_error("invalid cache OFF owner counts");
    const auto states=slots-states_built, owners=slots-native_built;
    if((states && retained>UINT64_MAX/states) || (owners && native>UINT64_MAX/owners))
        throw std::overflow_error("cache OFF commitment multiplication overflow");
    const auto a=states*retained, b=owners*native;
    if(b>UINT64_MAX-a) throw std::overflow_error("cache OFF commitment addition overflow");
    return a+b;
}
// Both CLI and HTTP ready samples include completed native/shared backing in L.
// Only the checked remaining transient/lazy commitment is still unpaid in F.
inline std::optional<std::uint64_t> cache_unpaid_commitment(bool ready,
    std::optional<std::uint64_t> initial, std::optional<std::uint64_t> remaining) {
    return ready?remaining:initial;
}
struct CacheLateFit {
    std::optional<std::uint64_t> U,L,A,F,E,S,device_free,device_unpaid,checkpoint;
    std::uint64_t headroom=0, observation_id=0, age_ns=0;
    std::uint64_t context_off=0, slots_off=0, context=0, slots=0;
    std::optional<std::uint64_t> explicit_budget;
    std::uint64_t resident_payload=0, resident_staging=0; // live ledger backing, not reserved credits
    bool limits_consistent=false, aliasing_known=false;
    bool cgroup_ancestry_unproven=false;
};
inline void intersect_cache_capacity(CacheLateFit& fit, std::uint64_t configured) {
    if(!fit.U || !fit.L || !fit.A) return;
    if(configured) fit.U=std::min(*fit.U,configured);
    fit.A=std::min(*fit.A,*fit.L>=*fit.U?0:*fit.U-*fit.L);
}
// Capture includes the last full block before the prompt boundary, off cadence.
inline std::uint64_t prefix_capture_cadence(std::uint64_t tile, std::uint64_t requested) {
    if(requested || !tile) return requested;
    return std::max(tile,((4096U+tile-1U)/tile)*tile);
}
inline bool prefix_capture_boundary_selected(std::uint64_t position, std::uint64_t covered,
    std::uint64_t tile, std::uint64_t cadence) {
    return tile && cadence && (position%cadence==0 || position>covered || covered-position<tile);
}
// Observer ownership never depends on storage admission, budget or family.
inline bool cache_observer_required(bool tiled, bool import_export, bool diagnostic) {
    return tiled && (import_export || diagnostic);
}
inline std::uint64_t minimum_prefix_capture_boundary(std::uint64_t tile, std::uint64_t context,
    std::uint64_t cadence) {
    return tile<context && prefix_capture_boundary_selected(tile,tile,tile,cadence)?tile:0;
}
inline std::uint64_t cache_unpaid_slot_bytes(std::uint64_t retained, std::uint64_t native,
    std::uint64_t transient, std::uint64_t slots, std::uint64_t states_built, std::uint64_t native_built) {
    auto future=cache_future_slot_bytes(retained,native,slots,states_built,native_built);
    if(slots && transient>(UINT64_MAX-future)/slots) throw std::overflow_error("cache transient overflow");
    return future+transient*slots;
}
// Reporting uses the same checked reserve as admission, including live staging.
inline std::optional<std::uint64_t> cache_fit_reserve(const CacheLateFit& x) {
    if(!x.F || !x.E || !x.S || x.resident_staging>*x.S) return {};
    auto reserve=std::max(UINT64_C(4)<<30,x.headroom);
    for(auto bytes:{*x.F,*x.E,UINT64_C(256)<<20,*x.S-x.resident_staging}) {
        if(bytes>UINT64_MAX-reserve) return {};
        reserve+=bytes;
    }
    return reserve;
}
// Only facts that native residency cannot repair may terminate feasibility.
// Pressure, stale observations and unpaid owner sizes must be decided afresh.
inline std::string_view cache_feasibility_decline(const CacheLateFit& x) {
    if (x.cgroup_ancestry_unproven) return "cgroup-ancestry-unproven";
    if (!x.observation_id || !x.limits_consistent || !x.U || !x.L || !x.A ||
        *x.A>*x.U || *x.L>*x.U) return "post-residency-memory-unavailable";
    if (!x.aliasing_known) return "owner-size-unknown";
    if (x.context!=x.context_off || x.slots!=x.slots_off) return "would-reduce-context-or-slots";
    return {};
}
// Both pre-existing and newly attached registries receive the same binding.
// Automatic setup is transactional; no request/copy has been submitted yet.
template<class Attach, class Bind, class Rollback, class Publish = std::function<void()>>
inline void finalize_cache_attachment(Attach attach, Bind bind, Rollback rollback, Publish publish=[] {}) {
    try { attach(); bind(); publish(); }
    catch (const std::exception&) { rollback(); }
}
struct CacheFitResult { std::uint64_t budget=0; std::string_view reason; };
inline CacheFitResult fit_cache_late(const CacheLateFit& x) {
    constexpr auto gib=UINT64_C(1)<<30;
    if (!x.observation_id || !x.limits_consistent || !x.U || !x.L || !x.A)
        return {0,"post-residency-memory-unavailable"};
    if (x.age_ns>UINT64_C(1000000000)) return {0,"memory-observation-stale"};
    if (!x.aliasing_known || !x.F || !x.E || !x.S || !x.checkpoint || !x.device_unpaid)
        return {0,"owner-size-unknown"};
    if (x.context!=x.context_off || x.slots!=x.slots_off) return {0,"would-reduce-context-or-slots"};
    const auto add=[](std::uint64_t& a,std::uint64_t b) { if (b>UINT64_MAX-a) return false; a+=b; return true; };
    if (x.resident_staging>*x.S || (x.resident_payload && !x.explicit_budget) ||
        (x.explicit_budget && x.resident_payload>*x.explicit_budget)) return {0,"post-residency-memory-unavailable"};
    std::uint64_t reserve=std::max(4*gib,x.headroom);
    if (!add(reserve,*x.F) || !add(reserve,*x.E) || !add(reserve,UINT64_C(256)<<20) || !add(reserve,*x.S-x.resident_staging))
        return {0,"memory-arithmetic-overflow"};
    if (*x.A>*x.U || *x.L>*x.U) return {0,"post-residency-memory-unavailable"};
    if (*x.A<reserve) return {0,"insufficient-host-headroom"};
    auto budget=x.explicit_budget.value_or(std::min({4*gib,*x.A/16,*x.A-reserve}));
    if (!x.explicit_budget) budget-=budget%4096;
    auto total=*x.L;
    if (!add(total,reserve) || !add(total,budget-x.resident_payload)) return {0,"memory-arithmetic-overflow"};
    if (budget-x.resident_payload>*x.A-reserve || total>*x.U) return {0,"insufficient-host-headroom"};
    auto device=*x.S-x.resident_staging;
    if (!add(device,*x.device_unpaid)) return {0,"memory-arithmetic-overflow"};
    if (!x.device_free || device>*x.device_free) return {0,"staging-does-not-fit"};
    if (!*x.checkpoint || !budget || *x.checkpoint>budget) return {0,"checkpoint-does-not-fit"};
    return {budget,"fitted-host-budget"};
}
} // namespace strix::decode_cli
