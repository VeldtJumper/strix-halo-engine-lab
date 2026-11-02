#include "strix/mtp_cohort_partition.hpp"
#include "strix/known_window_cohort.hpp"
#include "strix/host_phase_timing.hpp"
#include "strix/prompt_terminal_diagnostics.hpp"
#include "strix/hip_mtp_greedy_runtime.hpp"
#include "strix/mtp_sidecar_binding.hpp"
#include "strix/hip_mtp_window_verifier.hpp"
#include "strix/mtp_batch_window_controller.hpp"
#include "strix/hip_mtp_fast_draft.hpp"
#include <hip/hip_runtime.h>
#include <optional>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
namespace strix {
namespace {
void require(bool ok,const char* msg){if(!ok)throw std::invalid_argument(msg);}
bool session_matches(const SerialSessionContext& a,const SerialSessionContext& b) {
    return a.execution_identity==b.execution_identity && a.state_layout_identity==b.state_layout_identity &&
        a.runtime_compatibility_identity==b.runtime_compatibility_identity &&
        a.next_position==b.next_position && a.committed_tokens==b.committed_tokens &&
        a.health==b.health && a.state.valid==b.state.valid && a.state.generations==b.state.generations &&
        a.arithmetic_lineage.history_identity==b.arithmetic_lineage.history_identity &&
        a.arithmetic_lineage.tainted==b.arithmetic_lineage.tainted &&
        a.arithmetic_lineage.intervals==b.arithmetic_lineage.intervals;
}
}
struct HipMtpGreedyRuntime::Impl {
    std::shared_ptr<HipMtpRuntime> mtp;
    QualifiedMtpDraftPlan plan;
    ActivatedMtpDraftPlan activation;
    std::shared_ptr<HipMtpTargetHidden> target;
    std::shared_ptr<HipMtpWindowVerifier> window;
    SerialSessionContext* session;
    ExecutionRuntimeResources resources;
    DirectLaunchDispatcher* dispatcher;
    std::shared_ptr<const void> owner;
    DraftProposalStateContext draft;
    std::unique_ptr<QualifiedMtpForwardRuntimeAdapter> adapter;
    std::unique_ptr<QualifiedMtpDraftBackend> backend;
    bool busy=false,failed=false,cycled=false;
    float min_probability=0.0F;
    // The drafts of the cycle in progress (between begin_cycle_draft and finish_cycle).
    std::vector<std::uint32_t> candidates;
    std::vector<float> probabilities,runner_ups;
    // A draft chain between draft_begin and draft_end (begin_cycle_draft runs all phases in order).
    struct DraftState {
        std::uint32_t depth=0,pending=0;bool cost_model=false,survival=false,full_chain=false,open=false,copy_first=false;
        // Candidate rows this cycle could hold whatever the drafted depth (window, budget, context and retained
        // rows): the room of a cohort cycle's context extension (extend_cohort_drafts).
        std::uint32_t room=0;
        std::chrono::steady_clock::time_point start{};
    } ds;
    std::unique_ptr<MtpNgramDraft> ngram;
    std::unique_ptr<MtpFrameDraft> frame; // strict tool-call frames (batch runner default on; before the n-gram extension)
    std::unique_ptr<MtpContextDraft> context; // opt-in replacement of `ngram`, or beside it when context_observe
    bool context_observe=false; // the context drafter only learns and proposes wide cycles
    double mtp_accepted_ema=-1.0; // accepted candidates per drafted (not copy-first) cycle, for copy-first
    // Wide copy verify: the open proposal's frontier and pre-proposal state (abandon_wide_cycle restores it), and
    // the cooldown after an abandoned cycle (no proposal while the committed frontier is at or before wide_resume).
    bool wide_open=false,wide_cycled=false;
    DraftProposalStateContext wide_draft;
    std::uint64_t wide_frontier=0,wide_resume=0;
    std::vector<double> ngram_acceptance;
    // Candidates from this index on came from a context source (n-gram/context drafter); SIZE_MAX: none.
    std::size_t external_first=SIZE_MAX;
    bool cohort_external=false; // a cohort extension set the verifier's external rows (cleared at finish_cycle)
    // STRIX_MTP_DRAFT_DEPTH_MAX=N (opt-in, drafts only) or set_draft_depth_max: predictor drafts per cycle at most.
    std::uint32_t depth_max=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_DEPTH_MAX");
        const auto n=v?std::atoi(v):0;return n>0?static_cast<std::uint32_t>(n):0U;}();
    Impl(std::shared_ptr<HipMtpRuntime> m,QualifiedMtpDraftPlan q,ActivatedMtpDraftPlan a,
         std::shared_ptr<HipMtpTargetHidden> t,SerialSessionContext& s,const ExecutionRuntimeResources& r,
         DirectLaunchDispatcher& d,std::shared_ptr<const void> o)
        :mtp(std::move(m)),plan(std::move(q)),activation(std::move(a)),target(std::move(t)),
         session(&s),resources(r),dispatcher(&d),owner(std::move(o)) {
        require(mtp && target && owner,"MTP greedy consumer requires retained actual owners");
        require_qualified_mtp_draft_plan(plan);
        const auto& p=mtp->resources();
        require_rotary_producer_identity(target->model().rotary_identity(),p.model().rotary_identity());
        require(plan.native_arithmetic_profile_identity==mtp->numerical_profile_identity() &&
            activation.native_arithmetic_profile_identity==mtp->numerical_profile_identity(),
            "MTP greedy qualification/activation does not bind actual native arithmetic profile");
        require(plan.base.plan_identity==p.plan().plan_identity &&
            plan.base.proposal_identity==p.binding().compatibility_identity,
            "MTP greedy qualified plan differs from actual native owner");
        draft=make_mtp_draft_proposal_state(p.binding(),s);
        adapter=std::make_unique<QualifiedMtpForwardRuntimeAdapter>(p.model(),activation,
            MtpForwardPhysicalResources{&p.manifest(),&p.runtime_binding()},*mtp,*mtp,*mtp);
        backend=std::make_unique<QualifiedMtpDraftBackend>(plan,p.binding(),p.state_memory(),*adapter);
    }
};
HipMtpGreedyRuntime::HipMtpGreedyRuntime(std::shared_ptr<HipMtpRuntime> m,
    QualifiedMtpDraftPlan q,ActivatedMtpDraftPlan a,std::shared_ptr<HipMtpTargetHidden> t,
    SerialSessionContext& s,const ExecutionRuntimeResources& r,DirectLaunchDispatcher& d,
    std::shared_ptr<const void> owner)
    :impl_(std::make_unique<Impl>(std::move(m),std::move(q),std::move(a),std::move(t),s,r,d,std::move(owner))){}
HipMtpGreedyRuntime::~HipMtpGreedyRuntime()=default;
bool HipMtpGreedyRuntime::failed() const noexcept {
    return impl_->failed || impl_->mtp->failed() || (impl_->window && impl_->window->failed());
}
const HipMtpRuntime& HipMtpGreedyRuntime::native_runtime() const noexcept {return *impl_->mtp;}
void HipMtpGreedyRuntime::use_window_verifier(std::shared_ptr<HipMtpWindowVerifier> window) {
    auto& i=*impl_;require(!i.busy && !failed() && !i.cycled && window && window->target_owner()==i.target,
        "MTP window verifier must retain the same actual idle target before the first cycle");
    i.window=std::move(window);
    i.window->prepare_admission(*i.session);
}
void HipMtpGreedyRuntime::enable_ngram_draft(std::span<const std::uint32_t> prompt,MtpNgramDraftConfig config) {
    auto& i=*impl_;
    require(!i.busy && !i.cycled && i.window && !i.window->survival_controller() && !i.target->token_sampler(),
        "MTP n-gram draft extension needs the window verifier with the min-p controller, greedy selection and no cycle yet");
    i.ngram=std::make_unique<MtpNgramDraft>(config);
    i.ngram->reset(prompt);
}
void HipMtpGreedyRuntime::enable_context_draft(std::span<const std::uint32_t> prompt,MtpContextDraftConfig config,bool observe_only) {
    auto& i=*impl_;
    require(!i.busy && !i.cycled && i.window && !i.window->survival_controller() && !i.target->token_sampler() &&
        (observe_only || !i.ngram),
        "MTP context draft needs the window verifier with the min-p controller, greedy selection, no n-gram extension and no cycle yet");
    i.context=std::make_unique<MtpContextDraft>(config);i.context_observe=observe_only;
    i.context->reset(prompt);
}
void HipMtpGreedyRuntime::enable_frame_draft(std::span<const std::uint32_t> prompt,MtpFrameDraftConfig config) {
    auto& i=*impl_;
    require(!i.busy && !i.cycled && i.window && !i.window->survival_controller() && !i.target->token_sampler(),
        "MTP frame draft needs the window verifier with the min-p controller, greedy selection and no cycle yet");
    i.frame=std::make_unique<MtpFrameDraft>(config);
    i.frame->reset(prompt);
}
const MtpFrameDraftStats* HipMtpGreedyRuntime::frame_stats() const noexcept {
    return impl_->frame?&impl_->frame->stats():nullptr;
}
const MtpContextDraftStats* HipMtpGreedyRuntime::context_stats() const noexcept {
    return impl_->context?&impl_->context->stats():nullptr;
}
const MtpNgramDraftStats* HipMtpGreedyRuntime::ngram_stats() const noexcept {
    return impl_->ngram?&impl_->ngram->stats():nullptr;
}
void HipMtpGreedyRuntime::set_draft_min_probability(float p) {
    auto& i=*impl_;require(!i.busy && !i.cycled && p>=0.0F && p<=1.0F,"MTP draft minimum probability is set once in [0, 1] before cycles");
    i.min_probability=p;
}
void HipMtpGreedyRuntime::set_draft_depth_max(std::uint32_t n) {
    auto& i=*impl_;require(!i.busy,"MTP draft depth cap is set between cycles");
    i.depth_max=n;
}
std::uint32_t HipMtpGreedyRuntime::draft_depth_max() const noexcept {return impl_->depth_max;}
void HipMtpGreedyRuntime::prime_prompt(std::span<const HipMtpReplayRow> rows) {
    auto& i=*impl_;require(!i.busy && !failed(),"MTP greedy consumer is busy or failed");
    i.busy=true;
    try {i.mtp->prime_committed_rows(i.plan,i.activation,rows);i.busy=false;}
    catch(...){i.mtp->cancel();i.failed=true;i.busy=false;poison_draft_proposal_state(i.draft);throw;}
}
HipMtpTargetHiddenTicket HipMtpGreedyRuntime::prefill_prompt(std::span<const std::uint32_t> tokens) {
    auto& i=*impl_;
    require(!i.busy && !failed() && !i.cycled && !tokens.empty() &&
        i.session->next_position==0 && i.mtp->next_source_position()==0 &&
        i.resources.session_state_layout && i.target->retained_rows()>=3,
        "MTP prompt ingestion requires fresh actual target/predictor owners");
    const auto capacity=std::min(i.resources.session_state_layout->context_tokens,
        i.mtp->resources().model().effective_max_positions());
    require(tokens.size()<=capacity && tokens.size()<=i.mtp->resources().context_capacity(),
        "MTP prompt exceeds retained target/predictor capacity");
    for(auto token:tokens) require(token<i.mtp->resources().model().vocab_size,
        "MTP prompt token is outside target vocabulary");
    i.busy=true;
    try {
        HipMtpTargetHiddenTicket previous;
        for(auto token:tokens) {
            auto current=i.target->execute_serial(*i.session,i.resources,*i.dispatcher,token);
            if(previous.valid()) {
                const HipMtpReplayRow row{previous,token};
                i.mtp->prime_committed_rows(i.plan,i.activation,std::span<const HipMtpReplayRow>(&row,1));
            }
            previous=std::move(current);
        }
        i.draft=make_mtp_draft_proposal_state(i.mtp->resources().binding(),*i.session);
        i.busy=false;return previous;
    } catch(...) {i.mtp->cancel();i.failed=true;i.busy=false;poison_draft_proposal_state(i.draft);throw;}
}
std::uint32_t HipMtpGreedyRuntime::begin_cycle_draft(const HipMtpTargetHiddenTicket& initial,
    std::uint32_t max_candidates,std::uint32_t budget,std::span<const std::uint32_t> eos,bool cost_model) {
    const auto depth=draft_begin(initial,max_candidates,budget,eos,cost_model);
    if(!drafting()) return depth;
    while(draft_launch_step() && draft_after_step()) {}
    draft_end();
    return depth;
}
std::uint32_t HipMtpGreedyRuntime::draft_begin(const HipMtpTargetHiddenTicket& initial,
    std::uint32_t max_candidates,std::uint32_t budget,std::span<const std::uint32_t> eos,bool cost_model) {
    auto& i=*impl_;require(!i.busy && !failed() && budget,"MTP greedy cycle requires idle healthy owner and nonzero budget");
    require(initial.valid() && session_matches(initial.committed_session(),*i.session),
        "MTP greedy cycle requires exact current committed target ticket");
    initial.admit_rotary_reader(i.target->model().rotary_identity());
    require(initial.target_content_identity()==i.mtp->resources().shared_target().target_model_content_identity,
        "MTP greedy cycle target content differs from bound target");
    require(i.resources.session_state_layout && i.target->retained_rows()>=3,
        "MTP greedy cycle requires bounded target context and at least three retained target rows");
    const auto capacity=std::min(i.resources.session_state_layout->context_tokens,
        i.mtp->resources().model().effective_max_positions());
    require(i.session->next_position<capacity &&
        initial.source_position()<i.mtp->resources().context_capacity(),
        "MTP greedy cycle has no remaining target/predictor capacity");
    const auto remaining=std::min(capacity-i.session->next_position,
        i.mtp->resources().context_capacity()-initial.source_position());
    auto limit=std::min(max_candidates,i.mtp->resources().budget().maximum_candidate_depth);
    if(i.window)limit=std::min(limit,i.window->logical_rows()-1);
    const auto room=static_cast<std::uint32_t>(std::min<std::uint64_t>(std::min(std::min(
        i.mtp->resources().budget().maximum_candidate_depth,i.window?i.window->logical_rows()-1:max_candidates),budget-1),
        std::min<std::uint64_t>(remaining-1,i.target->retained_rows()-3)));
    // At most depth_max predictor drafts per cycle (the "MTP depth N" policy); the window controller still sizes
    // the window and context rows may still fill it.
    if(i.depth_max)limit=std::min(limit,i.depth_max);
    const auto depth=static_cast<std::uint32_t>(std::min<std::uint64_t>(
        std::min(limit,budget-1),std::min<std::uint64_t>(remaining-1,i.target->retained_rows()-3)));
    i.candidates.clear();i.probabilities.clear();i.external_first=SIZE_MAX;
    const auto pending=initial.predicted_token();
    if(std::find(eos.begin(),eos.end(),pending)!=eos.end())return depth;
    i.busy=true;
    try {
        host_phase_mark("0 cycle start",true);
        validate_draft_proposal_state(i.draft,*i.session);
        i.cycled=true;
        i.mtp->begin_cycle(pending,initial);
        begin_draft_proposal(i.draft,*i.session);
        host_phase_mark("1 begin cycle");
        // STRIX_MTP_DRAFT_WORTH=0: draft until min_probability only, without asking the window cost model.
        static const bool worth=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_WORTH");return !v || v[0]!='0';}();
        // The survival controller owns the chain cut (no min-p stop) and sees runner-up probabilities.
        // STRIX_MTP_DIAG_FULL_CHAIN=1 drafts the full depth without any cut.
        const bool full_chain=hip_mtp_full_chain_diagnostic();
        const bool survival=cost_model && i.window && i.window->survival_controller() && !full_chain;
        // Without a min-p cut (and without the survival controller) the chain runs sync-free: no keep test.
        const std::function<bool(std::span<const float>)> keep=((worth && i.min_probability>0.0F) || survival) && cost_model && i.window && !full_chain?
            std::function<bool(std::span<const float>)>([&](std::span<const float> p){
                return i.window->worth_drafting(p,i.mtp->draft_runner_up_probabilities());}):
            std::function<bool(std::span<const float>)>{};
        const auto sampler=i.target->token_sampler();
        require(!sampler || !depth || i.mtp->draft_head(),"MTP speculative sampling requires the draft chain");
        i.ds=Impl::DraftState{};
        i.ds.depth=depth;i.ds.room=room;i.ds.pending=pending;i.ds.cost_model=cost_model;i.ds.survival=survival;i.ds.full_chain=full_chain;
        i.ds.start=std::chrono::steady_clock::now();
        if(i.context && !i.context_observe && cost_model && !survival && !full_chain && i.window && i.mtp->draft_head()) {
            i.candidates=i.context->begin_copy_cycle(pending,depth,i.ngram_acceptance,i.min_probability,i.mtp_accepted_ema);
            i.ds.copy_first=!i.candidates.empty();
        }
        // Chain copy: an established copy continues without a predictor forward.
        if(i.context && i.context_observe && i.context->chain_enabled() && cost_model && !survival && !full_chain && i.window &&
           i.mtp->draft_head()) {
            i.candidates=i.context->chain_first(pending,depth,i.ngram_acceptance);
            i.ds.copy_first=!i.candidates.empty();
        }
        if(i.ds.copy_first) {
            // Retain the native cycle/checkpoint and normal teacher reconciliation,
            // but perform no predictor forward for an already gated context span (depth 0 opens no chain).
            i.ds.open=i.mtp->propose_chain_begin(i.plan,i.activation,*i.session,i.draft,0U,i.min_probability,{});
            for(const auto p:i.ngram_acceptance)i.probabilities.push_back(static_cast<float>(p));
        } else if(i.mtp->draft_head())
            i.ds.open=i.mtp->propose_chain_begin(i.plan,i.activation,*i.session,i.draft,depth,
                full_chain || survival?0.0F:i.min_probability,keep);
        else i.candidates=i.backend->propose_candidates(*i.session,i.draft,depth);
    } catch(...) {fail_cycle();throw;}
    return depth;
}
bool HipMtpGreedyRuntime::draft_launch_step() {
    auto& i=*impl_;
    if(!i.busy || !i.ds.open) return false;
    try {return i.mtp->propose_chain_launch_step();} catch(...) {fail_cycle();throw;}
}
HipMtpRuntime* HipMtpGreedyRuntime::draft_batch_runtime() noexcept {
    auto& i=*impl_;
    return i.busy && !i.failed && i.ds.open && i.mtp->draft_batch_open()?i.mtp.get():nullptr;
}
MtpDraftBatchDecline HipMtpGreedyRuntime::draft_batch_decline() const noexcept {
    const auto& i=*impl_;
    return i.busy && !i.failed && i.ds.open?i.mtp->draft_batch_decline():MtpDraftBatchDecline::inactive;
}
bool HipMtpGreedyRuntime::draft_after_step() {
    auto& i=*impl_;
    if(!i.busy || !i.ds.open) return false;
    try {return i.mtp->propose_chain_after_step();} catch(...) {fail_cycle();throw;}
}
void HipMtpGreedyRuntime::draft_end() {
    auto& i=*impl_;require(i.busy && !i.failed,"MTP draft end requires a begun draft");
    const auto depth=i.ds.depth;const auto pending=i.ds.pending;const bool cost_model=i.ds.cost_model;
    const bool survival=i.ds.survival,full_chain=i.ds.full_chain;
    try {
        if(i.ds.open) {i.candidates=i.mtp->propose_chain_finish();i.ds.open=false;}
        else if(i.mtp->draft_head() && !i.ds.copy_first) i.candidates.clear();
        if(survival || (i.window && i.window->depth_controller())) i.window->note_draft(i.ds.copy_first?0U:static_cast<std::uint32_t>(i.candidates.size()),
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-i.ds.start).count());
        i.runner_ups.clear();
        if(i.mtp->draft_head() && !i.ds.copy_first){
            const auto p=i.mtp->draft_probabilities();i.probabilities.assign(p.begin(),p.end());
            const auto q=i.mtp->draft_runner_up_probabilities();i.runner_ups.assign(q.begin(),q.end());
        }
        // Chain copy-first cycles keep the observing shadow (t36: without it the wide cells learn less, wide cycles
        // 18 -> 9 on rewrite-py-es); STRIX_MTP_CHAIN_SHADOW=0 drops it (diagnostic).
        static const bool chain_shadow=[]{const char* v=std::getenv("STRIX_MTP_CHAIN_SHADOW");return !v || v[0]!='0';}();
        if(i.context && i.context_observe && (!i.ds.copy_first || chain_shadow) && cost_model && !survival && !full_chain && i.window &&
           i.probabilities.size()==i.candidates.size()) {
            // Observe only: the context drafter builds its shadows over the MTP drafts (learning at finish_cycle);
            // nothing it proposes is delivered, so the extension below sees the cycle unchanged. A chain copy-first
            // cycle has no drafts to observe (its shadow would learn from rows past the copy and inflate the cells
            // that gate wide cycles: json-transform t34, wide abandons 6 -> 15).
            std::size_t confident=0;
            double kept_expected=0.0,run=1.0;
            while(confident<i.candidates.size() && i.probabilities[confident]>=i.min_probability) {
                run*=i.probabilities[confident];kept_expected+=run;++confident;
            }
            std::vector<double> unused;
            (void)i.context->begin_cycle(pending,i.candidates,confident,depth,unused,i.min_probability,kept_expected);
            i.context->discard_selection();
        }
        // Strict tool-call frames take precedence over the n-gram extension in a cycle they propose for;
        // both are computed on the predictor's drafts so that each one's shadow learns as usual.
        std::vector<std::uint32_t> frame_extension;std::vector<double> frame_acceptance;std::size_t frame_confident=0;
        if(i.frame && cost_model && !survival && !full_chain && i.window && i.probabilities.size()==i.candidates.size() &&
           !i.ds.copy_first) {
            while(frame_confident<i.candidates.size() && i.probabilities[frame_confident]>=i.min_probability) ++frame_confident;
            frame_extension=i.frame->begin_cycle(pending,i.candidates,frame_confident,i.ds.room,frame_acceptance);
            // Only a proposal that goes beyond the drafts it would replace.
            if(frame_confident+frame_extension.size()<=i.candidates.size() && std::equal(frame_extension.begin(),frame_extension.end(),
               i.candidates.begin()+static_cast<std::ptrdiff_t>(frame_confident))) frame_extension.clear();
        }
        const bool framed=!frame_extension.empty();
        bool chained=false;
        if(!framed && i.context && i.context_observe && i.context->chain_enabled() && !i.ds.copy_first && cost_model && !survival && !full_chain &&
           i.window && i.probabilities.size()==i.candidates.size()) {
            // Chain copy after the confident drafts (it replaces the extension's rows for this cycle); a strict
            // tool-call frame takes precedence in a cycle it proposes for.
            std::size_t confident=0;
            while(confident<i.candidates.size() && i.probabilities[confident]>=i.min_probability) ++confident;
            if(confident<depth) {
                auto copy=i.context->chain_extend(pending,std::span<const std::uint32_t>(i.candidates).first(confident),depth-confident,
                    i.ngram_acceptance);
                if(!copy.empty()) {
                    chained=true;
                    i.candidates.resize(confident);i.probabilities.resize(confident);
                    if(i.runner_ups.size()>confident) i.runner_ups.resize(confident);
                    for(std::size_t r=0;r<copy.size();++r) {
                        i.candidates.push_back(copy[r]);
                        i.probabilities.push_back(static_cast<float>(i.ngram_acceptance[r]));
                        if(!i.runner_ups.empty()) i.runner_ups.push_back(0.0F);
                    }
                    i.window->set_external_rows(confident,i.ngram_acceptance);
                }
            }
        }
        if(!chained && i.ngram && cost_model && !survival && !full_chain && i.window && i.probabilities.size()==i.candidates.size()) {
            // Keep the drafts up to the first one below the minimum probability; a context n-gram may
            // continue them up to the draft depth. The target verifies every row either way.
            std::size_t confident=0;
            while(confident<i.candidates.size() && i.probabilities[confident]>=i.min_probability) ++confident;
            auto extension=i.ngram->begin_cycle(pending,i.candidates,confident,depth,i.ngram_acceptance);
            if(framed) extension.clear(); // its shadow still learns
            if(!extension.empty()) {
                i.candidates.resize(confident);i.probabilities.resize(confident);
                if(i.runner_ups.size()>confident) i.runner_ups.resize(confident);
                for(std::size_t r=0;r<extension.size();++r) {
                    i.candidates.push_back(extension[r]);
                    i.probabilities.push_back(static_cast<float>(i.ngram_acceptance[r]));
                    if(!i.runner_ups.empty()) i.runner_ups.push_back(0.0F);
                }
                i.window->set_external_rows(confident,i.ngram_acceptance);
                i.ngram->used(extension.size());i.external_first=confident;
            }
        }
        if(framed) {
            const auto confident=frame_confident;
            i.candidates.resize(confident);i.probabilities.resize(confident);
            if(i.runner_ups.size()>confident) i.runner_ups.resize(confident);
            for(std::size_t r=0;r<frame_extension.size();++r) {
                i.candidates.push_back(frame_extension[r]);
                i.probabilities.push_back(static_cast<float>(frame_acceptance[r]));
                if(!i.runner_ups.empty()) i.runner_ups.push_back(0.0F);
            }
            i.ngram_acceptance=frame_acceptance;
            i.window->set_external_rows(confident,i.ngram_acceptance);
            i.frame->used(frame_extension.size());i.external_first=confident;
        }
        if(i.ds.copy_first){i.window->set_external_rows(0U,i.ngram_acceptance);i.external_first=0U;}
        if(i.context && !i.context_observe && cost_model && !survival && !full_chain && !i.ds.copy_first && i.window && !framed &&
           i.probabilities.size()==i.candidates.size()) {
            // Context proposals may follow any confident prefix (also none) and may replace a confident
            // chain with a longer context span; kept_drafts() is the MTP prefix they continue.
            std::size_t confident=0;
            double kept_expected=0.0,run=1.0;
            while(confident<i.candidates.size() && i.probabilities[confident]>=i.min_probability) {
                run*=i.probabilities[confident];kept_expected+=run;++confident;
            }
            auto extension=i.context->begin_cycle(pending,i.candidates,confident,depth,i.ngram_acceptance,i.min_probability,kept_expected);
            if(!extension.empty()) {
                confident=i.context->kept_drafts();
                i.candidates.resize(confident);i.probabilities.resize(confident);
                if(i.runner_ups.size()>confident) i.runner_ups.resize(confident);
                for(std::size_t r=0;r<extension.size();++r) {
                    i.candidates.push_back(extension[r]);
                    i.probabilities.push_back(static_cast<float>(i.ngram_acceptance[r]));
                    if(!i.runner_ups.empty()) i.runner_ups.push_back(0.0F);
                }
                i.window->set_external_rows(confident,i.ngram_acceptance);i.external_first=confident;
            }
        }
        host_phase_mark("2 draft chain");
        if(full_chain && std::getenv("STRIX_CYCLE_LOG"))
            std::fprintf(stderr,"draft_cost at=%llu depth=%zu ms=%.6f\n",
                static_cast<unsigned long long>(i.session->next_position-1U),i.candidates.size(),
                std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-i.ds.start).count());
        finish_draft_proposal(i.draft);
        host_phase_mark("3 finish proposal");
    } catch(...) {fail_cycle();throw;}
}
std::span<const std::uint32_t> HipMtpGreedyRuntime::drafted_candidates() const noexcept {return impl_->candidates;}
std::span<const TokenDraft> HipMtpGreedyRuntime::drafted_distributions() const noexcept {
    const auto& i=*impl_;return i.target->token_sampler()?i.mtp->sampled_drafts():std::span<const TokenDraft>{};
}
std::span<const float> HipMtpGreedyRuntime::drafted_probabilities() const noexcept {return impl_->probabilities;}
std::size_t HipMtpGreedyRuntime::drafted_external_first() const noexcept {return impl_->external_first;}
std::size_t HipMtpGreedyRuntime::extend_cohort_drafts(std::uint32_t rows,double gate) {
    auto& i=*impl_;
    require(i.busy && !i.failed && !i.ds.open,"MTP cohort extension requires an ended draft chain");
    if((!i.ngram && !i.context && !i.frame) || !i.window || i.ds.copy_first || i.ds.survival || i.ds.full_chain || i.context_observe ||
       !i.mtp->draft_head() || i.target->token_sampler() || i.probabilities.size()!=i.candidates.size() ||
       i.external_first!=SIZE_MAX || rows<2U) return 0U;
    const std::size_t room=std::min<std::size_t>(rows-1U,i.ds.room);
    std::size_t confident=0;double kept_expected=0.0,run=1.0;
    while(confident<i.candidates.size() && i.probabilities[confident]>=i.min_probability) {
        run*=i.probabilities[confident];kept_expected+=run;++confident;
    }
    std::vector<std::uint32_t> extension;
    bool framed=false;
    if(i.frame) {
        extension=i.frame->begin_cycle(i.ds.pending,i.candidates,confident,room,i.ngram_acceptance);
        framed=!extension.empty() && !i.ngram_acceptance.empty() && i.ngram_acceptance.front()>=gate;
        if(!framed) extension.clear();
    }
    // Called once per drafted cohort cycle: the drafter's shadow learns at finish_cycle whether or not its
    // proposal clears the gate.
    if(framed) {
        if(i.ngram){std::vector<double> unused;(void)i.ngram->begin_cycle(i.ds.pending,i.candidates,confident,room,unused);}
    } else if(!i.ngram && !i.context) {
        return 0U;
    } else if(i.context) {
        extension=i.context->begin_cycle(i.ds.pending,i.candidates,confident,room,i.ngram_acceptance,i.min_probability,kept_expected);
        if(!extension.empty()) confident=i.context->kept_drafts();
    } else extension=i.ngram->begin_cycle(i.ds.pending,i.candidates,confident,room,i.ngram_acceptance);
    // The chain gate: only the leading context rows whose learned acceptance reaches `gate` (the rows of a cohort
    // window are shared by every session in it: a weak row costs the others).
    std::size_t keep=0;
    while(keep<extension.size() && keep<i.ngram_acceptance.size() && i.ngram_acceptance[keep]>=gate) ++keep;
    if(!keep) {
        if(i.context)i.context->discard_selection();
        return 0U;
    }
    extension.resize(keep);i.ngram_acceptance.resize(keep);
    i.candidates.resize(confident);i.probabilities.resize(confident);
    if(i.runner_ups.size()>confident) i.runner_ups.resize(confident);
    for(std::size_t r=0;r<keep;++r) {
        i.candidates.push_back(extension[r]);
        i.probabilities.push_back(static_cast<float>(i.ngram_acceptance[r]));
        if(!i.runner_ups.empty()) i.runner_ups.push_back(0.0F);
    }
    i.window->set_external_rows(confident,i.ngram_acceptance);
    i.cohort_external=true;i.external_first=confident;
    if(framed) i.frame->used(keep);
    else if(!i.context)i.ngram->used(keep);
    return keep;
}
void HipMtpGreedyRuntime::set_ngram_peers(std::span<const HipMtpGreedyRuntime* const> peers) {
    auto& i=*impl_;
    if(!i.ngram) return;
    std::vector<const MtpNgramDraft*> out;
    for(const auto* p:peers) if(p && p!=this && p->impl_->ngram) out.push_back(p->impl_->ngram.get());
    i.ngram->set_peers(std::move(out));
}
bool HipMtpGreedyRuntime::drafting() const noexcept {return impl_->busy;}
HipMtpWindowVerifier* HipMtpGreedyRuntime::window_verifier() const noexcept {return impl_->window.get();}
SerialSessionContext& HipMtpGreedyRuntime::target_session() const noexcept {return *impl_->session;}
DirectLaunchDispatcher& HipMtpGreedyRuntime::target_dispatcher() const noexcept {return *impl_->dispatcher;}
void HipMtpGreedyRuntime::fail_cycle() noexcept {
    auto& i=*impl_;
    if(i.cohort_external && i.window){i.window->clear_external_rows();i.cohort_external=false;}
    i.mtp->cancel();i.failed=true;i.busy=false;poison_draft_proposal_state(i.draft);
}
namespace {
// STRIX_MTP_DRAFT_LOG=1: the cycle's outcome after its draft lines (source position, window rows, drafts, accepted, emitted ids).
void log_cycle(const HipMtpTargetHiddenTicket& initial,const HipMtpGreedyCycleResult& out) {
    static const bool log_cycles=std::getenv("STRIX_MTP_DRAFT_LOG")!=nullptr;
    if(!log_cycles) return;
    std::fprintf(stderr,"cycle at=%llu rows=%u proposed=%u accepted=%u emitted=",static_cast<unsigned long long>(initial.source_position()),
        out.target_window_rows,out.proposed_candidates,out.accepted_candidates);
    for(std::size_t e=0;e<out.emitted_tokens.size();++e) std::fprintf(stderr,e?",%u":"%u",out.emitted_tokens[e]);
    std::fprintf(stderr,"\n");
}
}
HipMtpGreedyCycleResult HipMtpGreedyRuntime::finish_cycle(const HipMtpTargetHiddenTicket& initial,
    HipMtpVerifiedTargetRows verified,std::uint32_t budget,std::span<const std::uint32_t> eos) {
    auto& i=*impl_;require(i.busy && !i.failed,"MTP greedy cycle finish requires a drafted cycle");
    auto is_eos=[&](std::uint32_t t){return std::find(eos.begin(),eos.end(),t)!=eos.end();};
    HipMtpGreedyCycleResult out;
    out.final_target=initial;
    out.proposed_candidates=static_cast<std::uint32_t>(i.candidates.size());
    try {
        host_phase_mark("8 verify tail");
        // The request ends here. Its model owner committed a safe prefix or
        // restored its checkpoint; admission resets both target and predictor.
        // There is no next pending output to reconcile on a selection failure.
        if(verified.selection_failure.has_value()) {
            out.selection_failure=std::move(verified.selection_failure);
            out.emitted_tokens=std::move(verified.emitted_tokens);
            out.selection_receipt=std::move(verified.selection_receipt);
            require(out.selection_receipt && out.selection_receipt->size()==out.emitted_tokens.size(),"MTP failure receipt length differs");
            if(i.cohort_external){i.window->clear_external_rows();i.cohort_external=false;}
            i.busy=false;return out;
        }
        require(!verified.committed_rows.empty() && verified.committed_rows.size()==verified.emitted_tokens.size(),
            "MTP target verifier returned no actual aligned committed output rows");
        std::vector<HipMtpReplayRow> replay;replay.reserve(verified.committed_rows.size());
        for(const auto& ticket:verified.committed_rows) {
            if(prefix_batch_certification_enabled() && prompt_terminal_digest_enabled()) out.output_logits_digests.push_back(ticket.terminal_logits_digest());
            replay.push_back({out.final_target,ticket.input_token()});out.final_target=ticket;
        }
        out.emitted_tokens=std::move(verified.emitted_tokens);
        out.consumed_target_rows=verified.committed_rows.size();
        out.accepted_candidates=verified.accepted_candidates;
        out.attempted_target_window=verified.attempted_window;out.accepted_target_window=verified.accepted_window;
        out.target_checkpoint_restored=verified.restored_checkpoint;
        out.target_r2_dispatches=verified.r2_dispatches;
        out.target_projection_launches_saved=verified.projection_launches_saved;
        out.target_window_rows=verified.chosen_rows;
        out.eos=is_eos(out.emitted_tokens.back());out.terminal_budget=out.emitted_tokens.size()>=budget;
        log_cycle(initial,out);
        if(i.cohort_external){i.window->clear_external_rows();i.cohort_external=false;}
        if(i.frame)i.frame->finish_cycle(out.emitted_tokens);
        if(i.ngram)i.ngram->finish_cycle(out.emitted_tokens);
        if(i.context) {
            i.context->finish_cycle(out.emitted_tokens,verified.verified_candidates,verified.accepted_candidates);
            if(!i.ds.copy_first) i.mtp_accepted_ema=i.mtp_accepted_ema<0.0?verified.accepted_candidates:
                0.75*i.mtp_accepted_ema+0.25*verified.accepted_candidates;
        }
        host_phase_mark("9 cycle bookkeeping");
        i.mtp->set_reconciliation_rows(replay,out.final_target);
        i.backend->reconcile_after_target(i.draft,*i.session,out.emitted_tokens);
        reconcile_draft_proposal_state(i.draft,*i.session);
        host_phase_mark("a reconcile");
        if(verified.selection_receipt) {
            require(verified.selection_receipt->size()==out.emitted_tokens.size(),"MTP reconciled receipt length differs");
            out.selection_receipt=std::move(verified.selection_receipt);
        }
        i.busy=false;return out;
    } catch(...) {fail_cycle();throw;}
}
std::vector<std::uint32_t> HipMtpGreedyRuntime::propose_wide(const HipMtpTargetHiddenTicket& initial,std::uint32_t rows,
    std::uint32_t budget,std::span<const std::uint32_t> eos,std::span<const std::uint32_t> forced) {
    auto& i=*impl_;
    require(!i.busy && !failed() && i.window && !i.target->token_sampler() && rows>=2U,
        "MTP wide cycle requires an idle healthy greedy owner with the window verifier");
    require(initial.valid() && session_matches(initial.committed_session(),*i.session),
        "MTP wide cycle requires exact current committed target ticket");
    initial.admit_rotary_reader(i.target->model().rotary_identity());
    require(forced.empty() || forced.size()+1U==rows,"MTP forced wide proposal must fill the wide rows");
    const auto pending=initial.predicted_token();
    if(budget<rows || std::find(eos.begin(),eos.end(),pending)!=eos.end() || !i.resources.session_state_layout ||
       i.session->next_position<=i.wide_resume) return {};
    const auto capacity=std::min(i.resources.session_state_layout->context_tokens,i.mtp->resources().model().effective_max_positions());
    if(i.session->next_position+rows>capacity || i.mtp->next_source_position()!=initial.source_position() ||
       initial.source_position()+rows>=i.mtp->resources().context_capacity()) return {};
    for(const auto t:forced) require(t<i.mtp->resources().model().vocab_size,"MTP forced wide proposal is outside the vocabulary");
    validate_draft_proposal_state(i.draft,*i.session);
    std::vector<std::uint32_t> tokens;
    if(!forced.empty()) tokens.assign(forced.begin(),forced.end());
    else if(i.context) tokens=i.context->propose_wide(pending,rows-1U);
    if(tokens.empty()) return tokens;
    i.busy=true;
    try {
        i.wide_open=true;i.wide_cycled=i.cycled;i.wide_draft=i.draft;i.wide_frontier=i.session->next_position;
        i.cycled=true;
        // A wide cycle drafts nothing: the proposal state passes through its phases around the external forward.
        begin_draft_proposal(i.draft,*i.session);finish_draft_proposal(i.draft);
        i.candidates=tokens;i.probabilities.clear();i.runner_ups.clear();
    } catch(...) {fail_cycle();throw;}
    return tokens;
}
HipMtpGreedyCycleResult HipMtpGreedyRuntime::finish_wide_cycle(const HipMtpTargetHiddenTicket& initial,
    HipMtpTargetHiddenTicket final_target,std::span<const MtpDeviceView> hidden,std::span<const std::uint32_t> inputs,
    std::vector<std::uint32_t> emitted,std::uint32_t verified,std::uint32_t rows,std::uint32_t budget,
    std::span<const std::uint32_t> eos) {
    auto& i=*impl_;require(i.busy && !i.failed && i.wide_open,"MTP wide cycle finish requires a proposed wide cycle");
    initial.admit_rotary_reader(i.target->model().rotary_identity());
    final_target.admit_rotary_reader(i.target->model().rotary_identity());
    HipMtpGreedyCycleResult out;
    try {
        const auto n=emitted.size();
        require(n && hidden.size()==n && inputs.size()==n && n<=rows && verified<=i.candidates.size() && n<=verified+1U &&
            inputs[0]==initial.predicted_token() && final_target.valid() &&
            final_target.source_position()==initial.source_position()+n && final_target.input_token()==inputs.back() &&
            final_target.predicted_token()==emitted.back() && session_matches(final_target.committed_session(),*i.session) &&
            hidden[0].address==initial.view().address,"MTP wide cycle rows do not continue the actual committed ticket");
        for(std::size_t r=1;r<n;++r) require(inputs[r]==i.candidates[r-1U] && inputs[r]==emitted[r-1U],
            "MTP wide cycle committed a row whose input is not the preceding target selection");
        // Teacher rows (h_{p-1}, pending), (h_p, x_1), ..., (h_{p+k-1}, x_k): the predictor cache reaches the new ticket.
        require(i.mtp->next_source_position()==initial.source_position(),"MTP wide cycle predictor frontier differs");
        i.mtp->prime_hidden_rows(i.plan,i.activation,hidden,inputs,i.target->model().rotary_identity());
        require(i.mtp->next_source_position()==final_target.source_position(),"MTP wide cycle did not reach the new frontier");
        // The draft vocabulary observes the pending token and the outputs, as begin_cycle and reconciliation do.
        if(i.mtp->draft_head()) {i.mtp->observe_draft_context(inputs.first(1));i.mtp->observe_draft_context(emitted);}
        reconcile_draft_proposal_state(i.draft,*i.session);
        const auto accepted=static_cast<std::uint32_t>(n-1U);
        out.final_target=std::move(final_target);
        out.proposed_candidates=static_cast<std::uint32_t>(i.candidates.size());
        out.accepted_candidates=accepted;out.consumed_target_rows=n;
        out.attempted_target_window=true;out.accepted_target_window=n==rows;out.target_checkpoint_restored=false;
        out.target_window_rows=rows;
        out.emitted_tokens=std::move(emitted);
        out.eos=std::find(eos.begin(),eos.end(),out.emitted_tokens.back())!=eos.end();
        out.terminal_budget=out.emitted_tokens.size()>=budget;
        log_cycle(initial,out);
        // A wide cycle is not an MTP-drafted cycle: mtp_accepted_ema (copy-first's comparison) is unchanged.
        if(i.context) i.context->finish_cycle(out.emitted_tokens,verified,accepted);
        if(i.ngram) i.ngram->finish_cycle(out.emitted_tokens);
        if(i.frame) i.frame->finish_cycle(out.emitted_tokens);
        i.candidates.clear();
        i.wide_open=false;i.busy=false;return out;
    } catch(...) {fail_cycle();throw;}
}
void HipMtpGreedyRuntime::abandon_wide_cycle(std::span<const std::uint32_t> truth) {
    auto& i=*impl_;require(i.busy && !i.failed && i.wide_open,"MTP wide cycle abandon requires a proposed wide cycle");
    try {
        require(!truth.empty() && truth.size()<=i.candidates.size()+1U && i.session->next_position==i.wide_frontier,
            "MTP abandoned wide cycle requires the restored proposal frontier");
        // Exactly the state before propose_wide: the predictor never saw the cycle (no begin_cycle, no rows).
        i.draft=i.wide_draft;i.cycled=i.wide_cycled;
        validate_draft_proposal_state(i.draft,*i.session);
        if(i.context) i.context->abandon_wide(truth);
        // Rows [frontier, frontier + truth.size()] hold the accepted prefix and the rejected row.
        i.wide_resume=i.wide_frontier+truth.size();
        i.candidates.clear();
        i.wide_open=false;i.busy=false;
    } catch(...) {fail_cycle();throw;}
}
HipMtpGreedyCycleResult HipMtpGreedyRuntime::cycle(const HipMtpTargetHiddenTicket& initial,
    std::uint32_t max_candidates,std::uint32_t budget,std::span<const std::uint32_t> eos,MtpSelectionTransaction* transaction) {
    auto& i=*impl_;
    auto is_eos=[&](std::uint32_t t){return std::find(eos.begin(),eos.end(),t)!=eos.end();};
    require(!transaction || (!i.target->token_sampler() && (!i.window || i.window->policy_selection_admitted())),
        "MTP transaction requires serial-equivalent greedy target execution");
    const auto depth=begin_cycle_draft(initial,max_candidates,budget,eos,true);
    if(!i.busy){HipMtpGreedyCycleResult out;out.final_target=initial;out.eos=true;return out;}
    if(i.window) {
        HipMtpVerifiedTargetRows verified;
        try {
            verified=i.window->verify(*i.session,*i.dispatcher,initial,i.candidates,budget,eos,
                i.mtp->draft_head()?std::span<const float>(i.probabilities):std::span<const float>{},
                i.target->token_sampler()?i.mtp->sampled_drafts():std::span<const TokenDraft>{},
                i.mtp->draft_head()?std::span<const float>(i.runner_ups):std::span<const float>{},transaction);
        } catch(...) {fail_cycle();throw;}
        return finish_cycle(initial,std::move(verified),budget,eos);
    }
    static_cast<void>(depth);
    HipMtpGreedyCycleResult out;
    out.emitted_tokens.reserve(i.candidates.size()+2);
    std::vector<HipMtpReplayRow> replay;replay.reserve(i.candidates.size()+1);
    out.final_target=initial;
    out.proposed_candidates=static_cast<std::uint32_t>(i.candidates.size());
    auto pending=initial.predicted_token();
    try {
        if(transaction)transaction->reset();
        const auto& candidates=i.candidates;
        for(std::size_t index=0;;++index) {
            replay.push_back({out.final_target,pending});
            const auto drafts=i.target->token_sampler()?i.mtp->sampled_drafts():std::span<const TokenDraft>{};
            auto completed=i.target->execute_serial(*i.session,i.resources,*i.dispatcher,pending,
                index<drafts.size()?&drafts[index]:nullptr,
                transaction?HipMtpHostRowSelector{[&](auto row){
                    try{return transaction->select(row);}catch(const MtpSelectionError& e){out.selection_failure=e.what();return 0U;}
                }}:HipMtpHostRowSelector{});
            if(out.selection_failure.has_value()) {
                out.selection_receipt=transaction->seal();i.busy=false;return out;
            }
            ++out.consumed_target_rows;
            if(prefix_batch_certification_enabled() && prompt_terminal_digest_enabled()) out.output_logits_digests.push_back(completed.terminal_logits_digest());
            pending=completed.predicted_token();out.final_target=std::move(completed);
            out.emitted_tokens.push_back(pending);
            const bool accepted=index<candidates.size() && candidates[index]==pending;
            if(accepted)++out.accepted_candidates;
            out.eos=is_eos(pending);out.terminal_budget=out.emitted_tokens.size()>=budget;
            if(out.eos || out.terminal_budget || !accepted)break;
        }
        log_cycle(initial,out);
        if(i.ngram)i.ngram->finish_cycle(out.emitted_tokens);
        if(i.frame)i.frame->finish_cycle(out.emitted_tokens);
        if(i.context)i.context->finish_cycle(out.emitted_tokens,std::min(i.candidates.size(),out.emitted_tokens.size()),out.accepted_candidates);
        host_phase_mark("9 cycle bookkeeping");
        i.mtp->set_reconciliation_rows(replay,out.final_target);
        i.backend->reconcile_after_target(i.draft,*i.session,out.emitted_tokens);
        reconcile_draft_proposal_state(i.draft,*i.session);
        host_phase_mark("a reconcile");
        if(transaction) {out.accepted_candidates=static_cast<std::uint32_t>(out.emitted_tokens.size()-1U);
            out.selection_receipt=transaction->seal();
            require(out.selection_receipt->size()==out.emitted_tokens.size(),"MTP serial receipt length differs");}
        i.busy=false;return out;
    } catch(...) {
        if(transaction)transaction->discard();
        // The target producer already poisons native execution failures. A
        // reconciliation/proposal failure disables this consumer and publishes
        // no speculative outputs; completed target history is never rolled back.
        fail_cycle();
        throw;
    }
}
namespace {
struct CohortItem {std::size_t member;std::uint32_t rows;};
// Global row budget (STRIX_COHORT_ROWS): items (largest first) fill groups of at most `budget` rows; a group's
// items fill chunks of at most eight rows; a chunk of two or more items runs on an idle owner of its total rows
// (4 or 8) that no member of the group verifies with, else its items run alone in the group's stream of steps.
// A group of one item verifies alone.
std::vector<HipMtpGreedyCycleResult> budget_cohorts(std::span<const HipMtpGreedyCohortMember> members,
    std::vector<CohortItem>& items,std::vector<HipMtpGreedyCycleResult>& out,std::vector<bool>& open,
    HipMtpGreedyCohortStats* stats,std::uint32_t budget) {
    const auto sum_of=[](const std::vector<CohortItem>& v){std::uint32_t s=0;for(const auto& x:v)s+=x.rows;return s;};
    std::stable_sort(items.begin(),items.end(),[](const CohortItem& a,const CohortItem& b){return a.rows>b.rows;});
    std::vector<CohortItem> solo;
    std::vector<MtpCohortVerifyDecline> why(members.size(),MtpCohortVerifyDecline::none); // a solo session's reason
    std::vector<std::vector<CohortItem>> groups;
    for(const auto& item:items) {
        if(!item.rows){solo.push_back(item);why[item.member]=MtpCohortVerifyDecline::no_rows;continue;}
        bool placed=false;
        for(auto& g:groups) if(sum_of(g)+item.rows<=budget){g.push_back(item);placed=true;break;}
        if(!placed)groups.push_back({item});
    }
    for(auto& g:groups) {
        if(g.size()<2U){for(const auto& x:g) why[x.member]=MtpCohortVerifyDecline::alone;solo.insert(solo.end(),g.begin(),g.end());continue;}
        // Chunks: first fit of the (descending) items into bins of eight rows; a bin of six rows gives its
        // last item a chunk of its own, so every chunk of two or more items has 4 or 8 rows.
        std::vector<std::vector<CohortItem>> bins;
        for(const auto& item:g) {
            bool placed=false;
            for(auto& b:bins) if(sum_of(b)+item.rows<=8U){b.push_back(item);placed=true;break;}
            if(!placed)bins.push_back({item});
        }
        std::vector<std::vector<CohortItem>> chunks;
        for(auto& b:bins) {
            const auto sum=sum_of(b);
            if(sum!=4U && sum!=8U && b.size()>=2U){const auto last=b.back();b.pop_back();chunks.push_back(b);chunks.push_back({last});}
            else chunks.push_back(b);
        }
        // Primaries: owners no member verifies with in this cycle, one per chunk.
        std::vector<const HipKnownPrefixWindow*> used;
        for(const auto& x:g) used.push_back(members[x.member].runtime->window_verifier()->owner_with_rows(x.rows).get());
        std::vector<std::shared_ptr<HipKnownPrefixWindow>> keep;
        std::vector<HipKnownPrefixWindow*> primaries;
        std::vector<std::size_t> sizes;
        std::vector<CohortItem> order;
        // Witness per order entry: the shared chunk's sessions (0: alone) and why a chunk member ran alone.
        std::vector<std::pair<std::uint32_t,MtpCohortVerifyDecline>> shared;
        for(const auto& c:chunks) {
            if(c.size()>=2U) {
                const auto total=sum_of(c);
                std::shared_ptr<HipKnownPrefixWindow> primary;
                for(const auto& x:g) {
                    auto o=members[x.member].runtime->window_verifier()->owner_with_rows(total);
                    if(o && std::find(used.begin(),used.end(),o.get())==used.end()){primary=std::move(o);break;}
                }
                if(primary) {
                    used.push_back(primary.get());primaries.push_back(primary.get());keep.push_back(std::move(primary));
                    sizes.push_back(c.size());order.insert(order.end(),c.begin(),c.end());
                    shared.insert(shared.end(),c.size(),std::pair{static_cast<std::uint32_t>(c.size()),MtpCohortVerifyDecline::none});
                    if(stats) stats->shared.shared_window(c.size(),total);
                    continue;
                }
            }
            const auto alone=c.size()>=2U?MtpCohortVerifyDecline::no_owner:MtpCohortVerifyDecline::alone;
            for(const auto& x:c){primaries.push_back(nullptr);sizes.push_back(1U);order.push_back(x);shared.push_back({0U,alone});}
        }
        std::vector<HipMtpCohortVerifyRequest> requests;
        std::uint32_t total=0;
        for(const auto& x:order) {
            const auto& c=members[x.member];auto& r=*c.runtime;total+=x.rows;
            requests.push_back({r.window_verifier(),&r.target_session(),c.initial,r.drafted_candidates(),
                r.drafted_probabilities(),c.budget,c.eos,x.rows,r.drafted_distributions(),c.transaction});
        }
        auto verified=HipMtpWindowVerifier::verify_cohort(requests,primaries,sizes);
        for(std::size_t k=0;k<order.size();++k) {
            const auto& c=members[order[k].member];
            auto& o=out[order[k].member];
            o=c.runtime->finish_cycle(*c.initial,std::move(verified[k]),c.budget,c.eos);
            open[order[k].member]=false;
            o.shared.verify_members=shared[k].first;o.shared.verify_rows=shared[k].first?order[k].rows:0U;
            o.shared.verify_decline=shared[k].second;
            if(stats) stats->shared.decline(shared[k].second);
        }
        if(stats){++stats->cohort_windows;stats->cohort_sessions+=order.size();stats->cohort_rows+=total;stats->cohort_chunks+=sizes.size();}
    }
    for(const auto& x:solo) {
        const auto& c=members[x.member];auto& r=*c.runtime;
        auto verified=r.window_verifier()->verify(r.target_session(),r.target_dispatcher(),*c.initial,r.drafted_candidates(),
            c.budget,c.eos,r.drafted_probabilities(),r.drafted_distributions(),{},c.transaction);
        out[x.member]=r.finish_cycle(*c.initial,std::move(verified),c.budget,c.eos);
        open[x.member]=false;
        out[x.member].shared.verify_decline=why[x.member];
        if(stats){++stats->solo_windows;stats->shared.decline(why[x.member]);}
    }
    return out;
}
// The drafting half of the witness (cohort_drafts), copied into the cycle results (the verification half is set
// where each session verifies).
void stamp_drafts(std::vector<HipMtpGreedyCycleResult>& out,const std::vector<MtpSharedExecutionCycle>& drafts) {
    for(std::size_t k=0;k<drafts.size() && k<out.size();++k) {
        auto& s=out[k].shared;const auto& d=drafts[k];
        s.draft_steps=d.draft_steps;s.draft_shared_steps=d.draft_shared_steps;s.draft_max_rows=d.draft_max_rows;s.draft_decline=d.draft_decline;
    }
}
}
namespace {
// Cohort context rows (opt-in through the request's context source, see extend_cohort_drafts): the chain gate,
// STRIX_BATCH_NGRAM_GATE (default 0.9): a context row joins a cohort window only at this learned acceptance.
double cohort_context_gate() {
    static const double gate=[]{const char* v=std::getenv("STRIX_BATCH_NGRAM_GATE");
        const double g=v && *v?std::atof(v):0.9;return g>=0.0 && g<=1.0?g:0.9;}();
    return gate;
}
std::vector<HipMtpGreedyCycleResult> cohort_cycle(std::span<const HipMtpGreedyCohortMember> members,
    HipMtpGreedyCohortStats* stats);
// STRIX_MTP_DRAFT_BATCH=1 (opt-in): the open fast greedy chains of a cycle run draft step k as one multi-token MTP
// forward on a batch stream (HipMtpRuntime::draft_batch_*); every row's drafts, probabilities and caches equal
// its own chain's. Sessions that cannot join (sampled, exact draft, other kernel branches, copy-first, depth 0)
// keep their own interleaved chains. STRIX_MTP_DRAFT_BATCH_LOG=1 prints the batch counters at exit.
bool draft_batch_enabled() {
    static const bool on=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_BATCH");return v && v[0]=='1';}();
    return on;
}
bool draft_batch_verify() {
    static const bool on=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_BATCH_VERIFY");return v && (v[0]=='1' || v[0]=='2');}();
    return on;
}
bool draft_batch_verify_strict() {
    static const bool on=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_BATCH_VERIFY");return v && v[0]=='2';}();
    return on;
}
struct DraftBatchStream {
    hipStream_t stream=nullptr;
    ~DraftBatchStream() {if(stream) {(void)hipStreamSynchronize(stream);(void)hipStreamDestroy(stream);}}
    void* get() {
        if(!stream && hipStreamCreateWithFlags(&stream,hipStreamNonBlocking)!=hipSuccess) {
            stream=nullptr;throw std::runtime_error("MTP draft batch stream creation failed");
        }
        return stream;
    }
};
struct DraftBatchCounters {
    std::uint64_t cycles=0,batched_cycles=0,declined=0,steps=0,rows=0,solo_rows=0,verified=0,mismatched=0;
    std::uint64_t live[hip_mtp_draft_batch_rows+1U]{};
    static bool log() {
        static const bool on=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_BATCH_LOG");return v && v[0]=='1';}();
        return on;
    }
    ~DraftBatchCounters() {print();}
    void print() const {
        if((!log() && !draft_batch_verify()) || !cycles) return;
        std::fprintf(stderr,"draft batch: cycles=%llu batched=%llu declined=%llu steps=%llu rows=%llu solo_rows=%llu live=",
            static_cast<unsigned long long>(cycles),static_cast<unsigned long long>(batched_cycles),static_cast<unsigned long long>(declined),
            static_cast<unsigned long long>(steps),static_cast<unsigned long long>(rows),static_cast<unsigned long long>(solo_rows));
        for(std::uint32_t t=1;t<=hip_mtp_draft_batch_rows;++t) std::fprintf(stderr,t>1U?",%llu":"%llu",static_cast<unsigned long long>(live[t]));
        std::fprintf(stderr," verified=%llu mismatched=%llu\n",static_cast<unsigned long long>(verified),static_cast<unsigned long long>(mismatched));
    }
};
// STRIX_MTP_DRAFT_BATCH_AB=1 (measurement, with STRIX_MTP_DRAFT_BATCH=1): the batch runs on even cohort cycles only; joint
// cycles add their draft/target wall time and emitted tokens to the parity's totals per live-session count, printed every
// 64 cycles ("draft batch ab N=<sessions> on/off: cycles, draft ms, cycle ms, tokens per cycle").
bool draft_batch_ab() {
    static const bool on=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_BATCH_AB");return v && v[0]=='1';}();
    return on;
}
struct DraftBatchAb {
    std::uint64_t cycle=0;bool on=false;
    // Host time of the current cycle's draft loop: enqueue (launch calls) and per-step waits (after-step calls), and steps.
    double pending_launch=0.0,pending_wait=0.0;std::uint64_t pending_steps=0;
    double draft[2][hip_mtp_draft_batch_rows+1U]{},total[2][hip_mtp_draft_batch_rows+1U]{};
    double launch[2][hip_mtp_draft_batch_rows+1U]{},wait[2][hip_mtp_draft_batch_rows+1U]{};
    std::uint64_t cycles[2][hip_mtp_draft_batch_rows+1U]{},tokens[2][hip_mtp_draft_batch_rows+1U]{},steps[2][hip_mtp_draft_batch_rows+1U]{};
    void add(std::size_t sessions,double draft_ms,double total_ms,std::uint32_t emitted) {
        const auto n=std::min<std::size_t>(sessions,hip_mtp_draft_batch_rows);const auto p=on?1U:0U;
        draft[p][n]+=draft_ms;total[p][n]+=total_ms;++cycles[p][n];tokens[p][n]+=emitted;
        launch[p][n]+=pending_launch;wait[p][n]+=pending_wait;steps[p][n]+=pending_steps;
        pending_launch=pending_wait=0.0;pending_steps=0;
        if((++cycle&31U)!=0U) return;
        for(std::size_t k=2;k<=hip_mtp_draft_batch_rows;++k) for(unsigned q=0;q<2U;++q) if(cycles[q][k]) {
            const auto c=static_cast<double>(cycles[q][k]);
            std::fprintf(stderr,"draft batch ab N=%zu %s: cycles=%llu draft_ms=%.3f cycle_ms=%.3f tokens=%.3f launch_ms=%.3f wait_ms=%.3f steps=%.2f\n",
                k,q?"on":"off",static_cast<unsigned long long>(cycles[q][k]),draft[q][k]/c,total[q][k]/c,static_cast<double>(tokens[q][k])/c,
                launch[q][k]/c,wait[q][k]/c,static_cast<double>(steps[q][k])/c);
        }
    }
};
DraftBatchAb& draft_batch_ab_timing() {thread_local DraftBatchAb t;return t;}
// The drafts of a cycle's open chains, step k of every session enqueued before any wait (each session's launches
// and waits are those of its own chain); with the draft batch, the batchable sessions' step k is one launch.
// executed (optional): + the predictor steps launched per member (STRIX_MTP_BATCH_LOG=2). Returns each member's
// drafting witness (steps, steps in a batched launch of two or more rows, why it drafted alone); stats (optional)
// gets the totals. Host counters only.
std::vector<MtpSharedExecutionCycle> cohort_drafts(std::span<const HipMtpGreedyCohortMember> members,std::vector<bool> stepping,
    HipMtpGreedyCohortStats* stats,std::vector<std::uint32_t>* executed=nullptr) {
    thread_local DraftBatchStream batch_stream;
    thread_local DraftBatchCounters counters;
    std::vector<std::size_t> batch;std::vector<HipMtpRuntime*> runtimes;
    std::optional<std::uint32_t> shared;
    void* stream=nullptr;
    bool batch_this_cycle=draft_batch_enabled();
    if(batch_this_cycle && draft_batch_ab()) {
        auto& ab=draft_batch_ab_timing();ab.on=(ab.cycle&1U)==0U;batch_this_cycle=ab.on;
    }
    std::vector<MtpSharedExecutionCycle> witness(members.size());
    if(batch_this_cycle) {
        ++counters.cycles;
        thread_local std::size_t start=0;
        const auto first=members.empty()?0U:start++%members.size();
        for(std::size_t i=0;i<members.size();++i) {
            const auto m=(first+i)%members.size();
            if(!stepping[m]) continue;
            if(batch.size()>=hip_mtp_draft_batch_rows) {witness[m].draft_decline=MtpDraftBatchDecline::capacity;continue;}
            if(auto* r=members[m].runtime->draft_batch_runtime()) {batch.push_back(m);runtimes.push_back(r);}
            else witness[m].draft_decline=members[m].runtime->draft_batch_decline();
        }
        auto why=MtpDraftBatchDecline::alone;
        if(batch.size()>=2U) shared=HipMtpRuntime::draft_batch_compatible(runtimes,&why);
        if(!shared) {
            if(batch.size()>=2U) ++counters.declined;
            for(const auto m:batch) witness[m].draft_decline=why;
            batch.clear();runtimes.clear();
        }
        else {
            stream=batch_stream.get();
            if(draft_batch_verify()) {
                // STRIX_MTP_DRAFT_BATCH_VERIFY_STAGGER=1: member i requests at most 1 + i % 8 drafts (divergent requested
                // depths and every live count as members stop), diagnostic only.
                static const bool stagger=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_BATCH_VERIFY_STAGGER");return v && v[0]=='1';}();
                if(stagger) for(std::size_t i=0;i<runtimes.size();++i) runtimes[i]->draft_batch_verify_cap(1U+static_cast<std::uint32_t>(i%8U));
                HipMtpRuntime::draft_batch_verify_reference(runtimes);
            }
            HipMtpRuntime::draft_batch_enter(runtimes,stream);++counters.batched_cycles;
            if(stats) stats->shared.draft_batch(batch.size());
        }
    } else if(draft_batch_enabled()) {
        for(std::size_t m=0;m<members.size();++m) if(stepping[m]) witness[m].draft_decline=MtpDraftBatchDecline::ab_off;
    }
    std::vector<bool> in_batch(members.size(),false);
    for(const auto m:batch) in_batch[m]=true;
    using clock=std::chrono::steady_clock;
    const bool timing=draft_batch_enabled() && draft_batch_ab();
    for(bool any=true;any;) {
        any=false;
        const auto t0=timing?clock::now():clock::time_point{};
        std::vector<bool> launched(members.size(),false);
        std::vector<HipMtpRuntime*> rows;
        for(std::size_t b=0;b<batch.size();++b) if(stepping[batch[b]]) {rows.push_back(runtimes[b]);launched[batch[b]]=true;}
        if(rows.size()>=1U) {
            HipMtpRuntime::draft_batch_launch(rows,stream,*shared);
            ++counters.steps;counters.rows+=rows.size();++counters.live[rows.size()];
            if(stats) stats->shared.draft_launch(rows.size());
            for(const auto m:batch) if(launched[m]) {
                auto& w=witness[m];
                w.draft_shared_steps+=rows.size()>=2U?1U:0U;
                w.draft_max_rows=std::max(w.draft_max_rows,static_cast<std::uint32_t>(rows.size()));
            }
            if(draft_batch_verify()) HipMtpRuntime::draft_batch_verify_step(runtimes,rows,stream);
        }
        for(std::size_t m=0;m<members.size();++m) if(stepping[m] && !in_batch[m]) {
            launched[m]=members[m].runtime->draft_launch_step();
            if(draft_batch_enabled() && launched[m]) ++counters.solo_rows;
        }
        const auto t1=timing?clock::now():clock::time_point{};
        if(executed) for(std::size_t m=0;m<members.size();++m) (*executed)[m]+=launched[m]?1U:0U;
        for(std::size_t m=0;m<members.size();++m) witness[m].draft_steps+=launched[m]?1U:0U;
        for(std::size_t m=0;m<members.size();++m) {
            if(!stepping[m]) continue;
            stepping[m]=launched[m] && members[m].runtime->draft_after_step();
            any=any || stepping[m];
        }
        if(timing) {
            auto& ab=draft_batch_ab_timing();
            ab.pending_launch+=std::chrono::duration<double,std::milli>(t1-t0).count();
            ab.pending_wait+=std::chrono::duration<double,std::milli>(clock::now()-t1).count();++ab.pending_steps;
        }
    }
    // The running totals every 8 cycles, every cycle when verifying (the exit summary needs an orderly shutdown).
    if(draft_batch_enabled() && DraftBatchCounters::log() && ((counters.cycles&7U)==0U || draft_batch_verify())) counters.print();
    if(!runtimes.empty() && draft_batch_verify()) {
        ++counters.verified;
        const auto differing=HipMtpRuntime::draft_batch_verify_check(runtimes,stream);
        counters.mismatched+=differing;
        // A running count after every verified cycle (the exit summary needs an orderly shutdown).
        std::fprintf(stderr,"draft batch verify: %llu cycles checked, %llu members differ, coverage %s\n",
                static_cast<unsigned long long>(counters.verified),static_cast<unsigned long long>(counters.mismatched),
                draft_batch_verify_coverage().c_str());
        // STRIX_MTP_DRAFT_BATCH_VERIFY=2: stop at the first difference.
        if(differing && draft_batch_verify_strict()) throw std::runtime_error("MTP draft batch differs from the one-token chains");
    }
    if(stats) for(const auto& w:witness) stats->shared.decline(w.draft_decline);
    return witness;
}
std::vector<HipMtpGreedyCycleResult> joint_cycle(std::span<const HipMtpGreedyCohortMember> members,
    HipMtpGreedyCohortStats* stats,MtpBatchWindowController& controller) {
    using clock=std::chrono::steady_clock;
    const auto start=clock::now();
    std::vector<MtpBatchWindowController::Request> requests;
    for(const auto& c:members) {
        require(c.runtime && c.initial && c.runtime->window_verifier(),"joint MTP cycle requires a window verifier");
        std::uint32_t limit=0;
        for(auto r:{2U,4U,8U}) if(c.runtime->window_verifier()->cohort_fits(c.runtime->target_session(),r,1U,c.budget)) limit=r;
        // Context/budget tails use the established serial fallback. In particular
        // never admit physical padding outside the request's remaining capacity.
        if(!limit) {if(stats) ++stats->shared.joint_fallbacks;return cohort_cycle(members,stats);}
        const auto cap=c.runtime->draft_depth_max()?std::min(c.maximum_candidates,c.runtime->draft_depth_max()):c.maximum_candidates;
        while(limit>2U && limit>cap+1U) limit/=2U;
        requests.push_back({c.slot,limit,{}});
    }
    static const std::uint32_t configured=[] {const char* v=std::getenv("STRIX_COHORT_ROWS");
        const auto n=v?std::atoi(v):32;return n==16?16U:32U;}();
    const auto budget=std::max(configured,2U*static_cast<std::uint32_t>(members.size()));
    const auto plan=controller.plan(requests,budget);
    const auto planned=clock::now();
    std::vector<bool> open(members.size(),false);
    std::vector<std::uint32_t> executed(members.size(),0U); // predictor steps launched (paid), before extension
    std::vector<HipMtpGreedyCycleResult> out(members.size());
    try {
        for(std::size_t k=0;k<members.size();++k) {
            const auto& c=members[k];
            static_cast<void>(c.runtime->draft_begin(*c.initial,plan[k]?plan[k]-1U:0U,c.budget,c.eos,false));
            open[k]=c.runtime->drafting();
            if(!open[k]) {out[k].final_target=*c.initial;out[k].eos=true;}
        }
        const auto drafts=cohort_drafts(members,open,stats,&executed);
        const auto chained=clock::now();
        std::uint32_t draft_steps=0;
        std::vector<std::size_t> retained(members.size(),0U);
        // The controller's acceptance history learns from predictor drafts only (context rows have their own).
        auto observed=requests;
        for(std::size_t k=0;k<members.size();++k) if(open[k]) {
            members[k].runtime->draft_end();
            retained[k]=members[k].runtime->drafted_probabilities().size();
            static_cast<void>(members[k].runtime->extend_cohort_drafts(requests[k].limit,cohort_context_gate()));
            requests[k].probabilities=members[k].runtime->drafted_probabilities();
            requests[k].external_first=members[k].runtime->drafted_external_first();
            const auto own=std::min(requests[k].probabilities.size(),members[k].runtime->drafted_external_first());
            observed[k].probabilities=requests[k].probabilities.first(own);
            draft_steps+=static_cast<std::uint32_t>(own);
        }
        const auto drafted=clock::now();
        const auto rows=controller.choose(requests,budget);
        const auto chosen=clock::now();
        std::vector<CohortItem> items;
        for(std::size_t k=0;k<members.size();++k) if(open[k]) items.push_back({k,rows[k]});
        budget_cohorts(members,items,out,open,stats,budget);
        stamp_drafts(out,drafts);
        const auto end=clock::now();
        const auto draft_ms=std::chrono::duration<double,std::milli>(drafted-start).count();
        const auto target_ms=std::chrono::duration<double,std::milli>(end-drafted).count();
        std::vector<std::uint32_t> accepted;
        std::uint32_t emitted=0;
        for(const auto& r:out) {accepted.push_back(r.accepted_candidates);emitted+=static_cast<std::uint32_t>(r.emitted_tokens.size());}
        controller.observe(observed,rows,accepted,draft_ms,target_ms,draft_steps);
        if(draft_batch_enabled() && draft_batch_ab()) draft_batch_ab_timing().add(members.size(),draft_ms,draft_ms+target_ms,emitted);
        // STRIX_MTP_BATCH_LOG=2 adds the plan's estimate (of the executed plan, probes included) and per member
        // slot:position:limit:remaining budget:planned rows/predictor steps launched/predictor drafts retained
        // before extension/predictor drafts kept after it/all candidates/verified rows/accepted/emitted, the
        // phase times (plan, chain, extension, choose, verification) and each member's draft probabilities.
        static const int log=[] {const char* v=std::getenv("STRIX_MTP_BATCH_LOG");return v && (v[0]=='1' || v[0]=='2')?v[0]-'0':0;}();
        if(log==2) {
            const auto& info=controller.last_plan();
            const auto ms=[](clock::time_point a,clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();};
            std::fprintf(stderr,"joint plan wall_ms=%lld sessions=%zu probe=%d expected=%.4f est_ms=%.4f members=",
                static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()),members.size(),
                info.probe?1:0,info.expected,info.ms);
            for(std::size_t k=0;k<members.size();++k)
                std::fprintf(stderr,"%s%u:%llu:%u:%u:%u/%u/%zu/%zu/%zu/%u/%u/%zu",k?",":"",members[k].slot,
                    static_cast<unsigned long long>(members[k].runtime->target_session().next_position),requests[k].limit,
                    members[k].budget,plan[k],executed[k],retained[k],observed[k].probabilities.size(),
                    requests[k].probabilities.size(),rows[k],out[k].accepted_candidates,out[k].emitted_tokens.size());
            std::fprintf(stderr," draft_ms=%.6f target_ms=%.6f phases=%.3f/%.3f/%.3f/%.3f/%.3f p=",draft_ms,target_ms,
                ms(start,planned),ms(planned,chained),ms(chained,drafted),ms(drafted,chosen),ms(chosen,end));
            for(std::size_t k=0;k<members.size();++k) {
                std::fputs(k?";":"",stderr);
                for(std::size_t j=0;j<requests[k].probabilities.size();++j)
                    std::fprintf(stderr,j?",%.3f":"%.3f",static_cast<double>(requests[k].probabilities[j]));
            }
            std::fputs("\n",stderr);
        }
        if(log) {
            std::fprintf(stderr,"joint cycle sessions=%zu rows=",members.size());
            for(std::size_t k=0;k<rows.size();++k) std::fprintf(stderr,k?",%u":"%u",rows[k]);
            std::fprintf(stderr," drafts=%u emitted=%u draft_ms=%.6f target_ms=%.6f\n",draft_steps,emitted,draft_ms,target_ms);
        }
        return out;
    } catch(...) {
        for(std::size_t k=0;k<members.size();++k) if(open[k]) members[k].runtime->fail_cycle();
        throw;
    }
}
// STRIX_COHORT_AB=1 (measurement): per cycle parity, the mean wall time of a cohort cycle (drafts, windows,
// acceptance and reconciliation) and the tokens it emitted, printed at exit; odd cycles run without the
// features of STRIX_COHORT_AB_OFF.
struct CohortAbTiming {
    double ms[2]{}; std::uint64_t cycles[2]{}, tokens[2]{};
    ~CohortAbTiming() {
        for (unsigned p = 0; p < 2U; ++p) if (cycles[p])
            std::fprintf(stderr, "cohort ab %s: %llu cycles x %.3f ms, %.3f tokens/cycle, %.2f tok/s\n", p ? "odd (features off)" : "even",
                static_cast<unsigned long long>(cycles[p]), ms[p] / static_cast<double>(cycles[p]),
                static_cast<double>(tokens[p]) / static_cast<double>(cycles[p]), 1000.0 * static_cast<double>(tokens[p]) / ms[p]);
    }
};
}
std::vector<HipMtpGreedyCycleResult> hip_mtp_greedy_cohort_cycle(std::span<const HipMtpGreedyCohortMember> members,
    HipMtpGreedyCohortStats* stats,MtpBatchWindowController* joint,MtpBatchWindowController* policy_joint) {
    for(const auto& c:members) if(c.transaction) require(c.runtime && c.runtime->window_verifier() &&
        c.runtime->window_verifier()->policy_selection_admitted(),"MTP cohort transaction owner is not serial equivalent");
    if(joint && std::any_of(members.begin(),members.end(),[](const auto& c){return c.transaction!=nullptr;})) {
        // Policy history never trains the ordinary shared controller. The ordinary
        // and policy groups still use their per-member transactions.
        std::vector<HipMtpGreedyCycleResult> out(members.size());
        for(bool policy:{false,true}) {
            std::vector<HipMtpGreedyCohortMember> group;std::vector<std::size_t> indices;
            for(std::size_t k=0;k<members.size();++k) if((members[k].transaction!=nullptr)==policy){group.push_back(members[k]);indices.push_back(k);}
            if(group.empty())continue;
            auto results=policy && policy_joint?joint_cycle(group,stats,*policy_joint):hip_mtp_greedy_cohort_cycle(group,stats,policy?nullptr:joint);
            for(std::size_t k=0;k<indices.size();++k)out[indices[k]]=std::move(results[k]);
        }
        return out;
    }
    ++known_window_cohort_cycle();
    static const bool log=[] {const char* v=std::getenv("STRIX_MTP_BATCH_LOG");return v && (v[0]=='1' || v[0]=='2');}();
    if(joint || (!known_window_cohort_ab() && log)) {
        const auto start=std::chrono::steady_clock::now();
        const auto before=stats?*stats:HipMtpGreedyCohortStats{};
        std::vector<HipMtpGreedyCycleResult> out;
        if(joint) {
            thread_local std::size_t group_start=0;
            const auto groups=mtp_cohort_partition(members.size(),group_start++);
            out.resize(members.size());
            for(const auto& group:groups) {
                std::vector<HipMtpGreedyCohortMember> part;
                for(const auto index:group) part.push_back(members[index]);
                auto results=joint_cycle(part,stats,*joint);
                for(std::size_t k=0;k<group.size();++k) out[group[k]]=std::move(results[k]);
            }
        } else out=cohort_cycle(members,stats);
        if(log) {
            const auto end=std::chrono::steady_clock::now();
            std::uint32_t emitted=0,max_members=0,draft_max_rows=0;
            for(const auto& r:out) {
                emitted+=static_cast<std::uint32_t>(r.emitted_tokens.size());
                max_members=std::max(max_members,r.shared.verify_members);draft_max_rows=std::max(draft_max_rows,r.shared.draft_max_rows);
            }
            // Shared-execution witness (tests/serve_slots_gate.py): chunks = budget/joint chunks, then
            // mtp_shared_execution_log (shared windows of two or more sessions, batched draft launches, declines).
            const auto witness=stats?" chunks="+std::to_string(stats->cohort_chunks-before.cohort_chunks)+
                mtp_shared_execution_log(before.shared,stats->shared,max_members,draft_max_rows):std::string();
            std::fprintf(stderr,"batch cycle sessions=%zu groups=%llu rows=%llu solo=%llu emitted=%u ms=%.6f begin_ns=%lld end_ns=%lld%s\n",
                members.size(),static_cast<unsigned long long>(stats?stats->cohort_windows-before.cohort_windows:0U),
                static_cast<unsigned long long>(stats?stats->cohort_rows-before.cohort_rows:0U),
                static_cast<unsigned long long>(stats?stats->solo_windows-before.solo_windows:0U),emitted,
                std::chrono::duration<double,std::milli>(end-start).count(),
                static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count()),
                static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count()),
                witness.c_str());
        }
        return out;
    }
    if (!known_window_cohort_ab()) return cohort_cycle(members, stats);
    static CohortAbTiming timing;
    const auto parity = known_window_cohort_cycle() & 1U;
    const auto start = std::chrono::steady_clock::now();
    auto out = cohort_cycle(members, stats);
    timing.ms[parity] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    ++timing.cycles[parity];
    for (const auto& r : out) timing.tokens[parity] += r.emitted_tokens.size();
    return out;
}
namespace {
std::vector<HipMtpGreedyCycleResult> cohort_cycle(std::span<const HipMtpGreedyCohortMember> members,
    HipMtpGreedyCohortStats* stats) {
    std::vector<HipMtpGreedyCycleResult> out(members.size());
    std::vector<bool> open(members.size(),false);
    using Item=CohortItem;
    std::vector<Item> items;
    // Rows per session: at most 8 / sessions (4 for two, 2 for three or four), so that the sessions of a
    // cycle fill one 8-row cohort; drafts beyond that window are not drafted.
    // STRIX_COHORT_MAX_ROWS=2|4|8 overrides it.
    static const std::uint32_t max_rows_override=[]{const char* v=std::getenv("STRIX_COHORT_MAX_ROWS");
        const auto n=v?std::atoi(v):0;return n==2||n==4||n==8?static_cast<std::uint32_t>(n):0U;}();
    // STRIX_COHORT_ROWS=16|32: a global row budget per cycle instead of one 8-row cohort. Each session keeps
    // the rows it gets in an 8-row cohort (8 / sessions, at least 2), so up to four sessions run as before and
    // more fill one group of up to the budget instead of several cohorts; the sessions of a group run in one
    // stream of steps as chunks of at most eight rows (verify_cohort over chunks). Measured: more rows per
    // session (4 at four sessions, 8 at two) cost more than the drafts they accept.
    static const std::uint32_t budget_rows=[]{const char* v=std::getenv("STRIX_COHORT_ROWS");
        const auto n=v?std::atoi(v):0;return n==16||n==32?static_cast<std::uint32_t>(n):0U;}();
    const std::uint32_t cap=max_rows_override?max_rows_override:members.size()>=3U?2U:members.size()==2U?4U:8U;
    try {
        // Drafts: every session's chain in phases, step k of all sessions (each on its own stream) enqueued
        // before any wait (STRIX_COHORT_DRAFT_INTERLEAVE=0: one session's chain after another). Each session's
        // launches and waits are those of its own chain.
        static const bool interleave=[]{const char* v=std::getenv("STRIX_COHORT_DRAFT_INTERLEAVE");return !v || v[0]!='0';}();
        const bool interleaved=interleave && !known_window_cohort_feature_off("draft");
        for(std::size_t m=0;m<members.size();++m) {
            const auto& c=members[m];
            require(c.runtime && c.initial && c.runtime->window_verifier(),"MTP cohort cycle requires window-verified greedy runtimes");
            for(std::size_t o=0;o<m;++o)require(members[o].runtime!=c.runtime,"MTP cohort cycle requires distinct sessions");
            if(interleaved) {
                static_cast<void>(c.runtime->draft_begin(*c.initial,std::min(c.maximum_candidates,cap-1U),c.budget,c.eos,false));
                if(c.runtime->drafting()) open[m]=true;
            } else {
                static_cast<void>(c.runtime->begin_cycle_draft(*c.initial,std::min(c.maximum_candidates,cap-1U),c.budget,c.eos,false));
                if(c.runtime->drafting()) open[m]=true;
            }
        }
        std::vector<MtpSharedExecutionCycle> drafts;
        if(interleaved) {
            drafts=cohort_drafts(members,open,stats);
            for(std::size_t m=0;m<members.size();++m) if(open[m]) members[m].runtime->draft_end();
        }
        for(std::size_t m=0;m<members.size();++m) if(open[m])
            static_cast<void>(members[m].runtime->extend_cohort_drafts(cap,cohort_context_gate()));
        for(std::size_t m=0;m<members.size();++m) {
            const auto& c=members[m];
            if(!open[m]){out[m].final_target=*c.initial;out[m].eos=true;continue;}
            auto rows=c.runtime->window_verifier()->cohort_rows(c.runtime->target_session(),
                c.runtime->drafted_candidates(),c.runtime->drafted_probabilities(),c.budget);
            if(rows>cap && c.runtime->window_verifier()->cohort_fits(c.runtime->target_session(),cap,
                c.runtime->drafted_candidates().size(),c.budget)) rows=cap;
            items.push_back({m,rows});
        }
        if(budget_rows) {
            budget_cohorts(members,items,out,open,stats,budget_rows);
            stamp_drafts(out,drafts);
            return out;
        }
        // Pack sessions into cohorts of 4 or 8 rows: larger windows first, each into the first cohort
        // with room; a lone session, or one whose owner does not fit, verifies alone.
        std::vector<Item> solo;
        std::vector<MtpCohortVerifyDecline> why(members.size(),MtpCohortVerifyDecline::none); // a solo session's reason
        std::vector<std::vector<Item>> groups;
        std::stable_sort(items.begin(),items.end(),[](const Item& a,const Item& b){return a.rows>b.rows;});
        for(const auto& item:items) {
            if(!item.rows || item.rows>=8U){
                solo.push_back(item);why[item.member]=item.rows?MtpCohortVerifyDecline::wide:MtpCohortVerifyDecline::no_rows;continue;
            }
            bool placed=false;
            for(auto& g:groups) {
                std::uint32_t sum=0;for(const auto& x:g)sum+=x.rows;
                if(sum+item.rows<=8U){g.push_back(item);placed=true;break;}
            }
            if(!placed)groups.push_back({item});
        }
        const auto fits=[&](const Item& x,std::uint32_t rows) {
            const auto& r=*members[x.member].runtime;
            return r.window_verifier()->cohort_fits(r.target_session(),rows,r.drafted_candidates().size(),members[x.member].budget);
        };
        for(auto& g:groups) {
            for(;;) {
                if(g.size()<2U) break;
                std::uint32_t sum=0;for(const auto& x:g)sum+=x.rows;
                if(sum==4U || sum==8U) break;
                // Round up to the next cohort size with the padding rows of a smaller window.
                const auto target=sum<4U?4U:8U;
                bool grown=false;
                for(auto x=g.rbegin();x!=g.rend() && !grown;++x)
                    for(const auto rows:{x->rows*2U,x->rows*4U})
                        if(rows<=8U && sum-x->rows+rows==target && fits(*x,rows)){x->rows=rows;grown=true;break;}
                if(!grown){why[g.back().member]=MtpCohortVerifyDecline::padding;solo.push_back(g.back());g.pop_back();}
            }
            if(g.size()==1U){why[g.front().member]=MtpCohortVerifyDecline::alone;solo.push_back(g.front());g.clear();}
        }
        for(const auto& g:groups) {
            if(g.empty()) continue;
            std::uint32_t total=0;for(const auto& x:g)total+=x.rows;
            // Primary: the idle owner of the cohort's rows of any member that verifies with fewer rows.
            std::shared_ptr<HipKnownPrefixWindow> primary;
            for(const auto& x:g) if(x.rows!=total && (primary=members[x.member].runtime->window_verifier()->owner_with_rows(total))) break;
            require(!!primary,"MTP cohort has no idle owner of its total rows");
            std::vector<HipMtpCohortVerifyRequest> requests;
            for(const auto& x:g) {
                const auto& c=members[x.member];auto& r=*c.runtime;
                requests.push_back({r.window_verifier(),&r.target_session(),c.initial,r.drafted_candidates(),
                    r.drafted_probabilities(),c.budget,c.eos,x.rows,r.drafted_distributions(),c.transaction});
            }
            auto verified=HipMtpWindowVerifier::verify_cohort(requests,*primary);
            for(std::size_t k=0;k<g.size();++k) {
                const auto& c=members[g[k].member];
                auto& o=out[g[k].member];
                o=c.runtime->finish_cycle(*c.initial,std::move(verified[k]),c.budget,c.eos);
                open[g[k].member]=false;
                o.shared.verify_members=static_cast<std::uint32_t>(g.size());o.shared.verify_rows=g[k].rows;
            }
            if(stats){++stats->cohort_windows;stats->cohort_sessions+=g.size();stats->cohort_rows+=total;stats->shared.shared_window(g.size(),total);}
        }
        for(const auto& x:solo) {
            const auto& c=members[x.member];auto& r=*c.runtime;
            auto verified=r.window_verifier()->verify(r.target_session(),r.target_dispatcher(),*c.initial,r.drafted_candidates(),
                c.budget,c.eos,r.drafted_probabilities(),r.drafted_distributions(),{},c.transaction);
            out[x.member]=r.finish_cycle(*c.initial,std::move(verified),c.budget,c.eos);
            open[x.member]=false;
            out[x.member].shared.verify_decline=why[x.member];
            if(stats){++stats->solo_windows;stats->shared.decline(why[x.member]);}
        }
        stamp_drafts(out,drafts);
        return out;
    } catch(...) {
        for(std::size_t m=0;m<members.size();++m) if(open[m]) members[m].runtime->fail_cycle();
        throw;
    }
}
}
} // namespace strix
