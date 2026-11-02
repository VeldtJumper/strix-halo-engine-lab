#pragma once
// Shared-execution witness of batched MTP serving (STRIX_MTP_DRAFT_BATCH=1, STRIX_MTP_BATCH_JOINT=1, the
// cohort paths): counters that tell "grouped but executed separately" from "shared". Host-side counters only:
// no syncs, no arithmetic change. A session's cycle records the shared verification window it ran in (sessions,
// its own rows) and the batched draft launches it ran in; a session that drafted or verified alone records why.
// HipMtpGreedyCohortStats carries the server totals (the 'batch cycle' line under STRIX_MTP_BATCH_LOG prints
// their per-cycle delta), MtpServeResult the per-request receipt (strix.shared_execution).
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>

namespace strix {

// Why a drafting session of a cohort cycle ran its own chain instead of a batched draft launch.
enum class MtpDraftBatchDecline : std::uint8_t {
    none=0,
    ab_off,             // STRIX_MTP_DRAFT_BATCH_AB=1 parity cycle without the batch
    inactive,           // no open chain on the predictor
    sampled,            // sampled drafts (a sampler is attached)
    exact,              // exact (teacher) draft arithmetic
    device_draft,       // drafts sampled on the device
    keep,               // keep callback (cost-model chain cut)
    stopped,            // the chain already stopped
    depth,              // the chain has no step left
    busy,               // the chain is already in a batch
    broken,             // the chain failed
    unsupported_branch, // the batched launchers do not run this member's kernel branch
    incompatible_owner, // model, plan, profile, weights or draft head differ from the batch's
    alone,              // the only batchable chain of its cycle
    capacity,           // beyond the batched launchers' rows
    count
};
// Why a session of a cohort cycle verified alone instead of in a shared window.
enum class MtpCohortVerifyDecline : std::uint8_t {
    none=0,
    no_rows,  // no window rows (serial target step)
    wide,     // eight rows or more: a window of its own (8-row path)
    padding,  // its cohort could not be rounded up to 4 or 8 rows (8-row path)
    alone,    // the only session of its group or chunk
    no_owner, // its chunk found no idle primary owner of the chunk's rows
    count
};
inline constexpr std::size_t mtp_draft_batch_decline_count=static_cast<std::size_t>(MtpDraftBatchDecline::count);
inline constexpr std::size_t mtp_cohort_verify_decline_count=static_cast<std::size_t>(MtpCohortVerifyDecline::count);

inline const char* mtp_decline_name(MtpDraftBatchDecline d) noexcept {
    static constexpr const char* names[]={"none","ab_off","inactive","sampled","exact","device_draft","keep","stopped",
        "depth","busy","broken","unsupported_branch","incompatible_owner","alone","capacity"};
    static_assert(std::size(names)==mtp_draft_batch_decline_count);
    const auto i=static_cast<std::size_t>(d);
    return i<mtp_draft_batch_decline_count?names[i]:"unknown";
}
inline const char* mtp_decline_name(MtpCohortVerifyDecline d) noexcept {
    static constexpr const char* names[]={"none","no_rows","wide","padding","alone","no_owner"};
    static_assert(std::size(names)==mtp_cohort_verify_decline_count);
    const auto i=static_cast<std::size_t>(d);
    return i<mtp_cohort_verify_decline_count?names[i]:"unknown";
}

// One session's share of one cohort cycle.
struct MtpSharedExecutionCycle {
    std::uint32_t verify_members=0;   // sessions of the shared window it verified in (0: verified alone)
    std::uint32_t verify_rows=0;      // its own rows in that window
    std::uint32_t draft_steps=0;      // draft steps launched for it (batched or its own chain)
    std::uint32_t draft_shared_steps=0; // of them, in a batched launch of two or more rows
    std::uint32_t draft_max_rows=0;   // widest batched launch it ran in
    MtpDraftBatchDecline draft_decline=MtpDraftBatchDecline::none;
    MtpCohortVerifyDecline verify_decline=MtpCohortVerifyDecline::none;
};

// Server totals (HipMtpGreedyCohortStats::shared).
struct MtpSharedExecutionStats {
    // Verification windows (8-row cohorts, budget/joint chunks) that ran two or more sessions on one primary
    // owner: windows, their sessions and rows.
    std::uint64_t shared_windows=0,shared_sessions=0,shared_rows=0;
    // Batched drafting: cycles whose chains entered one batched launch chain, the sessions entered, the batched
    // launches, the launches of two or more rows and their rows.
    std::uint64_t draft_batches=0,draft_batch_sessions=0,draft_launches=0,draft_shared_launches=0,draft_shared_rows=0;
    // Joint cycles that fell back to the ordinary cohort cycle (a member's context or budget tail).
    std::uint64_t joint_fallbacks=0;
    std::array<std::uint64_t,mtp_draft_batch_decline_count> draft_declines{};
    std::array<std::uint64_t,mtp_cohort_verify_decline_count> verify_declines{};

    void shared_window(std::size_t sessions,std::uint64_t rows) noexcept {
        if(sessions<2U) return;
        ++shared_windows;shared_sessions+=sessions;shared_rows+=rows;
    }
    void draft_batch(std::size_t sessions) noexcept {++draft_batches;draft_batch_sessions+=sessions;}
    void draft_launch(std::size_t rows) noexcept {
        ++draft_launches;
        if(rows>=2U) {++draft_shared_launches;draft_shared_rows+=rows;}
    }
    void decline(MtpDraftBatchDecline d) noexcept {if(d!=MtpDraftBatchDecline::none) ++draft_declines[static_cast<std::size_t>(d)];}
    void decline(MtpCohortVerifyDecline d) noexcept {if(d!=MtpCohortVerifyDecline::none) ++verify_declines[static_cast<std::size_t>(d)];}
};

namespace shared_execution_detail {
template<class Enum,std::size_t N>
std::string declines(const std::array<std::uint64_t,N>& counts,const std::type_identity_t<std::array<std::uint64_t,N>>* before,
    bool json) {
    std::string out;
    for(std::size_t i=1;i<N;++i) {
        const auto n=counts[i]-(before?(*before)[i]:0U);
        if(!n) continue;
        if(!out.empty()) out+=',';
        out+=json?std::string("\"")+mtp_decline_name(static_cast<Enum>(i))+"\":":std::string(mtp_decline_name(static_cast<Enum>(i)))+":";
        out+=std::to_string(n);
    }
    return json?"{"+out+"}":out.empty()?std::string("none"):out;
}
}

// The 'batch cycle' log suffix: the cycle's delta of the totals, plus the widest shared window and batched draft
// launch of the cycle. Format (tests/serve_slots_gate.py parses it; keep the key order):
//   " shared_windows=W shared_sessions=S shared_rows=R max_members=M draft_batches=B draft_batch_sessions=D
//     draft_launches=L draft_shared_launches=X draft_shared_rows=Y draft_max_rows=Z joint_fallbacks=F
//     draft_declines=reason:n,...|none verify_declines=reason:n,...|none"
inline std::string mtp_shared_execution_log(const MtpSharedExecutionStats& before,const MtpSharedExecutionStats& after,
    std::uint32_t max_members,std::uint32_t draft_max_rows) {
    const auto d=[](std::uint64_t a,std::uint64_t b){return std::to_string(b-a);};
    return " shared_windows="+d(before.shared_windows,after.shared_windows)+" shared_sessions="+d(before.shared_sessions,after.shared_sessions)+
        " shared_rows="+d(before.shared_rows,after.shared_rows)+" max_members="+std::to_string(max_members)+
        " draft_batches="+d(before.draft_batches,after.draft_batches)+" draft_batch_sessions="+d(before.draft_batch_sessions,after.draft_batch_sessions)+
        " draft_launches="+d(before.draft_launches,after.draft_launches)+
        " draft_shared_launches="+d(before.draft_shared_launches,after.draft_shared_launches)+
        " draft_shared_rows="+d(before.draft_shared_rows,after.draft_shared_rows)+" draft_max_rows="+std::to_string(draft_max_rows)+
        " joint_fallbacks="+d(before.joint_fallbacks,after.joint_fallbacks)+
        " draft_declines="+shared_execution_detail::declines<MtpDraftBatchDecline>(after.draft_declines,&before.draft_declines,false)+
        " verify_declines="+shared_execution_detail::declines<MtpCohortVerifyDecline>(after.verify_declines,&before.verify_declines,false);
}

// One request's witness (the per-request receipt object strix.shared_execution).
struct MtpSharedExecutionReceipt {
    std::uint64_t cohort_cycles=0;        // cycles it ran in a cohort cycle (two or more live sessions)
    std::uint64_t verify_shared_cycles=0; // of them, verified in a shared window
    std::uint64_t verify_shared_rows=0;   // its rows verified in shared windows
    std::uint64_t draft_steps=0,draft_shared_steps=0;
    std::uint32_t verify_max_members=0,draft_max_rows=0;
    std::array<std::uint64_t,mtp_draft_batch_decline_count> draft_declines{};
    std::array<std::uint64_t,mtp_cohort_verify_decline_count> verify_declines{};

    void add(const MtpSharedExecutionCycle& c) noexcept {
        ++cohort_cycles;
        if(c.verify_members>=2U) {++verify_shared_cycles;verify_shared_rows+=c.verify_rows;}
        verify_max_members=std::max(verify_max_members,c.verify_members);
        draft_steps+=c.draft_steps;draft_shared_steps+=c.draft_shared_steps;
        draft_max_rows=std::max(draft_max_rows,c.draft_max_rows);
        if(c.draft_decline!=MtpDraftBatchDecline::none) ++draft_declines[static_cast<std::size_t>(c.draft_decline)];
        if(c.verify_decline!=MtpCohortVerifyDecline::none) ++verify_declines[static_cast<std::size_t>(c.verify_decline)];
    }
    [[nodiscard]] std::string json() const {
        return "{\"cohort_cycles\":"+std::to_string(cohort_cycles)+",\"verify_shared_cycles\":"+std::to_string(verify_shared_cycles)+
            ",\"verify_shared_rows\":"+std::to_string(verify_shared_rows)+",\"verify_max_members\":"+std::to_string(verify_max_members)+
            ",\"draft_steps\":"+std::to_string(draft_steps)+",\"draft_shared_steps\":"+std::to_string(draft_shared_steps)+
            ",\"draft_max_rows\":"+std::to_string(draft_max_rows)+
            ",\"draft_declines\":"+shared_execution_detail::declines<MtpDraftBatchDecline>(draft_declines,nullptr,true)+
            ",\"verify_declines\":"+shared_execution_detail::declines<MtpCohortVerifyDecline>(verify_declines,nullptr,true)+"}";
    }
};

} // namespace strix
