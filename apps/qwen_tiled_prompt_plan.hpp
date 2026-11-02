#pragma once
#include "strix/known_prefill_limits.hpp"
#include "strix/qpf2k_defaults.hpp"
#include <array>
#include <algorithm>
#include <bit>
#include <string>
#include <optional>
#include <stdexcept>
#include <vector>
namespace strix::decode_cli {
enum class QwenPlanOutput { state_only, terminal, all_rows };
enum class QwenPlanKind { t_owner, tail_owner, serial, window };
enum class QwenPlanPhase { blocks, tails, serial, done };
struct QwenPlanOwner {
    std::uint32_t rows=0;
    QwenPlanOutput policy=QwenPlanOutput::terminal;
    std::uint64_t execution=0,kernel=0;
    bool operator==(const QwenPlanOwner&) const = default;
};
// Resolved numerical inputs only. Identity binds profiles, kernels, handoffs and layout
// outside this enumerator; none of quota, cadence, retention or store presence is an input.
struct QwenTiledPlanConfig {
    std::uint32_t rows=0, schedule=0, priming=0;
    bool tiled_start=true, suppressed=false, force_sync_priming=false;
    // The existing numerical identity still binds the detailed execution/profile pair.
    // These explicit generation receipts additionally reject inconsistent metadata.
    std::array<std::uint64_t,6> initial_generations{},generations_per_row{};
    std::uint32_t valid_state=0;
    std::uint64_t decode_execution=0,decode_kernel=0,prefill_execution=0,prefill_kernel=0;
    std::uint64_t prefill_profile=0,decode_profile=0,profile_pair=0;
    QwenPlanOwner block;
    std::vector<QwenPlanOwner> tails;
    std::vector<std::uint32_t> ladder;
};
// The schedule the tiled MTP bridge (CLI --mtp, HTTP serving, cohorts) executes for a STRIX_QPF2K_SCHED value:
// the bridge never defers a block, so bit 2 is inert there and its plan, batch certification and plan identity
// see the schedule-12 equivalent (the default 14 and an explicit 14 alike).
inline std::uint32_t qwen_bridge_plan_schedule(const char* sched) {
    return static_cast<std::uint32_t>(qpf2k::effective_sched(sched?std::strtoul(sched,nullptr,0):0UL,false));
}
struct QwenPlanCursor {
    QwenPlanPhase phase=QwenPlanPhase::blocks;
    std::size_t tail=0;
    int balanced=-1;
    bool operator==(const QwenPlanCursor&) const = default;
};
struct QwenPlanUnit {
    std::uint64_t begin=0,end=0;
    std::uint32_t owner_rows=0,priming=0;
    std::size_t tail=0;
    bool t_owner=false,capture=false,final=false,balanced=false,merged=false,heads=false;
    QwenPlanCursor cursor;
    QwenPlanKind kind=QwenPlanKind::serial;
    bool async_priming=false;
    QwenPlanOwner owner;
    bool operator==(const QwenPlanUnit&) const = default;
};
inline std::optional<QwenPlanUnit> qwen_tiled_next(const QwenTiledPlanConfig& config,
    QwenPlanCursor& cursor,std::uint64_t pos_,std::uint64_t n_,bool small_=false,bool adopted_=false) {
    using Phase=QwenPlanPhase;
    auto& phase_=cursor.phase; auto& tail_=cursor.tail; auto& balanced_=cursor.balanced;
    const auto unit=[&](std::uint32_t rows,bool t,std::size_t tail,bool capture,bool final,bool balanced,bool merged,bool heads) {
        return QwenPlanUnit{pos_,pos_+rows,t?config.rows:config.tails[tail].rows,config.priming,
            tail,t,capture,final,balanced,merged,final && heads,cursor,
            t?QwenPlanKind::t_owner:QwenPlanKind::tail_owner,
            (config.priming&1U)!=0U && !final && !config.force_sync_priming,
            t?config.block:config.tails[tail]};
    };
    while(phase_==Phase::blocks || phase_==Phase::tails) {
        const bool blocks=phase_==Phase::blocks;
        if(!blocks && !config.ladder.empty()) {
            // STRIX_QPF2K_SCHED bit 8 fail-closed: rows the ladder runs in one unit must not fall to a smaller tail
            // owner or to the serial suffix (row classes sit on a 128-row grid per block; a pruned owner would change
            // their arithmetic silently). Owners larger than the unit are skipped below and re-checked next pass.
            const bool start=config.tiled_start;
            // rows the current owner may take (the step below: pos + rows <= n - 1, or == n with heads)
            const bool theads=start && (tail_>=config.tails.size() ||
                config.tails[tail_].policy!=QwenPlanOutput::state_only);
            const std::uint64_t m=n_-pos_-(theads?0U:1U);
            std::uint32_t want=0;
            for(const auto r:config.ladder) if(r && m>=r) {want=r;break;}
            if(want && (tail_>=config.tails.size() || config.tails[tail_].rows<want))
                throw std::logic_error("qpf2k sched: the ladder's "+std::to_string(want)+"-row unit has no tail owner (pruned)");
        }
        if(!blocks && tail_>=config.tails.size()){phase_=Phase::serial;break;}
        auto& backend=blocks?config.block:config.tails[tail_];

        const auto rows=blocks?config.rows:config.tails[tail_].rows;
        // STRIX_MTP_TILED_START=0: the last prompt row always runs serially (the previous handoff).
        // Otherwise a block that ends exactly at the last prompt row also computes that row's head, and
        // the first draft cycle starts from the block's last row (its tap and FP32 logits).
        const bool tiled_start=config.tiled_start;
        const bool heads=tiled_start && backend.policy!=QwenPlanOutput::state_only;
        // STRIX_QPF2K_SCHED bit 16: balanced blocks on the T owner (the prompt's rows down to a multiple of 128 as
        // ceil(rest / T) blocks of a multiple of 128 rows <= T, each the first rows of the T arena; the last rest mod 128
        // rows then run as the ladder runs them). Bit 4: the ladder's leading multiple-of-128 tail units merged into one
        // partial block of the T owner. Small units retain the incumbent dynamic scheduling policy.
        const bool balanced_raw=(config.schedule&16U)!=0;
        const bool merged_raw=(config.schedule&4U)!=0;
        const bool defaults_off=config.suppressed;
        const bool balanced_env=balanced_raw&&!defaults_off,merged_env=merged_raw&&!defaults_off;
        // Per prompt, decided at its first unit: only where the balanced plan is bitwise equal to the ladder.
        if(blocks && (balanced_env || merged_env) && balanced_<0) {
            std::vector<std::uint32_t> ladder=config.ladder;
            if(ladder.empty()) for(const auto& t:config.tails) ladder.push_back(t.rows);
            // the tiled start: the ladder may end exactly at n (heads); without it the last row is a suffix row
            const std::uint64_t m=heads?n_-pos_:n_-pos_-1U;
            // the balanced plan needs the tiled start (the ladder then covers exactly n - pos rows)
            balanced_=balanced_env && heads && known_prefill_balanced_eligible(m,rows,ladder)?1:0;
        }
        if(blocks && balanced_env && balanced_==1 && !adopted_ && !small_ &&
           backend.policy!=QwenPlanOutput::all_rows && heads) {
            const auto size=known_prefill_balanced_rows(n_-pos_,rows);
            if(size) {
                const auto r=static_cast<std::uint32_t>(size);
                return unit(r,blocks,tail_,r==rows,pos_+r==n_,true,false,heads);
            }
        }
        // Bit 4 (not balanced): the ladder's leading multiple-of-128 tail units merged into one partial block of the T owner.
        if(blocks && merged_env && balanced_==0 && !adopted_ && !small_ &&
           backend.policy!=QwenPlanOutput::all_rows && heads) {
            std::vector<std::uint32_t> ladder=config.ladder;
            if(ladder.empty()) for(const auto& t:config.tails) ladder.push_back(t.rows);
            if(const auto merged=known_prefill_merged_tail_rows(n_-pos_,rows,ladder)) {
                const auto r=static_cast<std::uint32_t>(merged);
                return unit(r,blocks,tail_,false,pos_+r==n_,false,true,heads);
            }
        }
        if(!adopted_ && (pos_+rows<=n_-1U || (heads && pos_+rows==n_))) {
            // small_blocks: the T block as units of the largest tail size dividing T (the same rows and bits: a tiled row's
            // arithmetic does not depend on its block's size), e.g. while other requests decode
            if(blocks && small_ && !config.tails.empty() && config.tails[0].rows<rows && rows%config.tails[0].rows==0) {
                auto& t=config.tails[0];
                const bool theads=tiled_start && t.policy!=QwenPlanOutput::state_only;
                if(pos_+t.rows<=n_-1U || (theads && pos_+t.rows==n_)) return unit(t.rows,false,0,true,pos_+t.rows==n_,false,false,theads);
            }
            return unit(rows,blocks,tail_,blocks,pos_+rows==n_,false,false,heads);
        }
        if(blocks)phase_=Phase::tails;else ++tail_;
    }
    return std::nullopt;
}
// The incumbent suffix policy: one odd row, then largest 8/4/2 window.
// Refusal/availability is a runtime capability, never a storage decision.
inline QwenPlanUnit qwen_suffix_unit(std::uint64_t pos,std::uint64_t n,bool windows) {
    QwenPlanUnit u;u.begin=pos;u.end=pos;u.cursor.phase=QwenPlanPhase::serial;
    if(pos>=n) {u.cursor.phase=QwenPlanPhase::done;return u;}
    std::uint32_t rows=1;
    if(windows && (n-pos)%2U==0) {rows=8;while(rows>n-pos) rows/=2U;}
    u.end=pos+rows;u.owner_rows=rows;u.final=u.end==n;
    u.kind=rows>1?QwenPlanKind::window:QwenPlanKind::serial;
    return u;
}
inline std::optional<QwenPlanUnit> qwen_prompt_next(const QwenTiledPlanConfig& config,
    QwenPlanCursor& cursor,std::uint64_t pos,std::uint64_t n,bool windows=false,bool small=false,bool adopted=false) {
    if(auto u=qwen_tiled_next(config,cursor,pos,n,small,adopted)) return u;
    if(pos>=n) {cursor.phase=QwenPlanPhase::done;return std::nullopt;}
    return qwen_suffix_unit(pos,n,windows);
}
// Eligibility check without constructing a certificate, also used for no-store suppression.
inline std::optional<QwenPlanCursor> qwen_full_t_continuation(const QwenTiledPlanConfig& config,
    std::uint64_t n,std::uint64_t p) {
    if(!p || p>=n || !config.rows || p%config.rows) return std::nullopt;
    QwenPlanCursor cursor;std::uint64_t pos=0;
    while(pos<p) {
        const auto u=qwen_tiled_next(config,cursor,pos,n);
        if(!u || !u->t_owner || u->end-u->begin!=config.rows || u->balanced || u->merged || u->final || u->end>p) return std::nullopt;
        pos=u->end;
    }
    return cursor;
}
// A certificate is deliberately limited to invariant full-T producer prefixes.
// Encode the actual trace (including priming), not the remaining length n-P.
inline std::vector<std::uint64_t> qwen_full_t_certificate(const QwenTiledPlanConfig& config,
    std::uint64_t n,std::uint64_t p,QwenPlanCursor* continuation=nullptr) {
    if(!p || p>=n || !config.rows || p%config.rows) return {};
    QwenPlanCursor cursor; std::uint64_t pos=0;
    std::vector<std::uint64_t> proof{0x514650554c4c5431ULL,config.rows,config.priming};
    while(pos<p) {
        const auto u=qwen_tiled_next(config,cursor,pos,n);
        if(!u || !u->t_owner || u->end-u->begin!=config.rows || u->balanced || u->merged || u->final || u->end>p) return {};
        proof.insert(proof.end(),{u->begin,u->end,static_cast<std::uint64_t>(config.block.policy),
            u->heads?1U:0U,u->balanced?1U:0U,u->owner.execution,u->owner.kernel});
        pos=u->end;
    }
    proof.insert(proof.end(),{config.decode_execution,config.decode_kernel,config.prefill_execution,
        config.prefill_kernel,config.prefill_profile,config.decode_profile,config.profile_pair});
    proof.push_back(config.valid_state);
    for(std::size_t f=0;f<config.initial_generations.size();++f) {
        if(config.generations_per_row[f] && p>(UINT64_MAX-config.initial_generations[f])/config.generations_per_row[f]) return {};
        proof.push_back(config.initial_generations[f]+p*config.generations_per_row[f]);
    }
    if(continuation) *continuation=cursor;
    return proof;
}
inline bool qwen_capture_boundary(std::uint64_t p,std::uint64_t n,std::uint32_t grid,
    bool complete,bool pending,bool certified) {
    return p && p<n && grid && p%grid==0 && complete && !pending && certified;
}

// Batch certificates describe executed units; a replay alone never authorizes capture.
inline bool qwen_batch_nest_possible(bool worker,std::uint64_t limit,std::uint32_t t,std::uint64_t remaining) {
    return worker && limit && t==8192 && remaining && remaining<t && remaining<=limit;
}
inline std::uint64_t qwen_batch_staging_cap(std::uint64_t explicit_cap,std::uint64_t target_row,
    std::uint64_t target_fixed,std::uint64_t predictor_row,std::uint64_t held,std::uint64_t t) {
    if(explicit_cap) return explicit_cap;
    const auto add=[](std::uint64_t a,std::uint64_t b) {
        if(b>UINT64_MAX-a) throw std::overflow_error("batch staging reservation overflow");
        return a+b;
    };
    const auto row=add(target_row,predictor_row);
    if(t && row>UINT64_MAX/t) throw std::overflow_error("batch staging reservation overflow");
    return add(row*t,add(target_fixed,add(held,255U)&~std::uint64_t{255}));
}
struct QwenUnitExec {
    std::uint64_t begin=0,end=0;
    std::uint32_t rows=0;
    bool t_owner=false,own_owner=false,grouped=false,small=false,forced=false,adopted=false;
    std::uint64_t exec_class() const {return forced?5U:small?3U:grouped?2U:!t_owner?0U:!own_owner?4U:1U;}
};
struct QwenBatchConfig {
    bool groups=true,group_owner=false,small=false,windows=true,cohort=true,service=false,cancel=false;
    std::uint64_t group_rows=UINT32_MAX,nest_rows=0;
    double service_ms=250.0,prompt_share=0.0;
    std::uint64_t lookahead=2,short_rows=2048;
    bool interleave=true;
    std::string shared="hc,router,moe,elementwise";
    const char* decline(std::uint32_t schedule) const {
        return small?"small":group_owner?"group-owner:prompt-batch-rows":(schedule&2U)?"sched-bit-2":"";
    }
    bool allowed(std::uint32_t schedule) const {return !decline(schedule)[0];}
    std::vector<std::uint64_t> frame() const {
        std::vector<std::uint64_t> result{groups,group_rows,group_owner,small,windows,cohort,service,cancel,nest_rows,
            std::bit_cast<std::uint64_t>(service_ms),lookahead,std::bit_cast<std::uint64_t>(prompt_share),short_rows,interleave,shared.size()};
        for(const unsigned char c:shared) result.push_back(c);
        return result;
    }
};
inline bool qwen_batch_capture_eligible(const QwenBatchConfig& config,std::uint32_t schedule,
    const QwenUnitExec& unit,std::uint32_t t,bool trace_ok,bool next_t,std::uint32_t next_rows,bool nest_possible) {
    return config.allowed(schedule) && trace_ok && unit.t_owner && unit.own_owner && unit.rows==t &&
        !unit.grouped && !unit.small && !unit.forced && !unit.adopted && next_t &&
        next_rows>0 && next_rows<=t && (next_rows==t || !nest_possible);
}
inline bool qwen_producer_trace_ok(const QwenTiledPlanConfig& config,std::uint64_t n,std::uint64_t p,
    const std::vector<QwenUnitExec>& trace) {
    QwenPlanCursor cursor;std::uint64_t pos=0;
    for(const auto& actual:trace) {
        if(actual.begin>=p) break;
        const auto expected=qwen_tiled_next(config,cursor,pos,n);
        if(!expected || actual.begin!=pos || actual.end!=expected->end || actual.end>p ||
            !actual.t_owner || !actual.own_owner || actual.rows!=config.rows || actual.exec_class()!=1 ||
            actual.adopted || expected->balanced || expected->merged || expected->final) return false;
        pos=actual.end;
    }
    return pos==p && qwen_full_t_continuation(config,n,p).has_value();
}
inline std::vector<std::uint64_t> qwen_full_t_certificate_v2(const QwenTiledPlanConfig& config,
    std::uint64_t n,std::uint64_t p,std::uint64_t route=2,std::uint64_t exec_class=1) {
    auto proof=qwen_full_t_certificate(config,n,p);
    if(proof.empty() || exec_class!=1 || (route!=1 && route!=2)) return {};
    std::vector<std::uint64_t> result{0x5146554c4c5432ULL,route,config.rows,config.priming}; // QFULLT2
    std::size_t at=3;
    for(std::uint64_t pos=0;pos<p;pos+=config.rows,at+=7) {
        result.insert(result.end(),proof.begin()+static_cast<std::ptrdiff_t>(at),proof.begin()+static_cast<std::ptrdiff_t>(at+7));
        result.push_back(exec_class);
    }
    result.insert(result.end(),proof.begin()+static_cast<std::ptrdiff_t>(at),proof.end());
    return result;
}
inline bool qwen_suffix_solo_class(const QwenTiledPlanConfig& config,std::uint64_t n,std::uint64_t p,
    const QwenBatchConfig& batch,bool nesting_possible) {
    if(!batch.allowed(config.schedule) || nesting_possible) return false;
    auto cursor=qwen_full_t_continuation(config,n,p);
    if(!cursor) return false;
    while(const auto unit=qwen_prompt_next(config,*cursor,p,n,batch.windows)) {
        if((unit->kind==QwenPlanKind::tail_owner && batch.groups) ||
            (unit->kind==QwenPlanKind::window && batch.cohort)) return false;
        p=unit->end;
    }
    return p==n;
}

// Join boundaries are full-T certified boundaries. Any cache grid that divides T
// (including the HTTP 1024-row checkpoint default) contains every such boundary,
// so the producer's capture and the joiner's ordinary lookup meet there.
inline bool qwen_prefix_join_grid(std::uint64_t cache_rows,std::uint64_t t_rows) {
    return cache_rows && t_rows && t_rows%cache_rows==0;
}
// A wait is tied to an admission generation, never just a reusable slot address.
struct QwenPrefixJoinWait {
    std::uint64_t generation=0,boundary=0;
    bool pending(std::uint64_t current,bool active,bool prefilling,std::uint64_t position) const {
        return boundary && current==generation && active && prefilling && position<=boundary;
    }
};
inline std::uint64_t qwen_prefix_join_boundary(const QwenTiledPlanConfig& producer,std::uint64_t pn,
    const QwenTiledPlanConfig& consumer,std::uint64_t cn,std::uint64_t common,std::uint64_t from,
    const QwenBatchConfig& batch,bool nesting_possible) {
    if(!producer.rows || producer.rows!=consumer.rows || pn<2 || cn<2 ||
        !batch.allowed(producer.schedule) || !batch.allowed(consumer.schedule)) return 0;
    auto p=std::min({common,pn-1U,cn-1U})/producer.rows*producer.rows;
    for(;p>from;p-=producer.rows) {
        const auto proof=qwen_full_t_certificate_v2(producer,pn,p);
        if(proof.empty() || proof!=qwen_full_t_certificate_v2(consumer,cn,p) ||
            !qwen_suffix_solo_class(consumer,cn,p,batch,nesting_possible)) continue;
        auto cursor=qwen_full_t_continuation(producer,pn,p);
        const auto next=qwen_prompt_next(producer,*cursor,p,pn);
        // Existing asynchronous capture needs the next writer on the same T owner.
        if(next && next->t_owner && next->owner==producer.block) return p;
    }
    return 0;
}
} // namespace strix::decode_cli
