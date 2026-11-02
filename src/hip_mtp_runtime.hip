#include "strix/host_phase_timing.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/hip_mtp_runtime.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/hip_token_sampling.hpp"
#include "strix/hip_token_sampling_device.hpp"
#include <chrono>
#include "strix/hip_mtp_attention.hpp"
#include "strix/hip_mtp_projection.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include "strix/hip_mtp_shared_expert.hpp"
#include "strix/hip_mtp_fast_draft.hpp"
#include "strix/hip_embedding_launcher.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_ple_launcher.hpp"
#include "strix/mtp_sidecar_binding.hpp"
#include <cstring>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <array>
#include <functional>
#include <bit>
#include <string>
#include <limits>
#include <optional>
#include <stdexcept>

namespace strix {
namespace {
constexpr std::array<MtpDraftPlanStep,11> native_steps{MtpDraftPlanStep::input_fusion,MtpDraftPlanStep::attention_hyper_mix,
    MtpDraftPlanStep::qsa_indexer,MtpDraftPlanStep::sparse_attention,MtpDraftPlanStep::attention_combine,
    MtpDraftPlanStep::mlp_hyper_mix,MtpDraftPlanStep::routed_and_shared_moe,MtpDraftPlanStep::mlp_combine,
    MtpDraftPlanStep::publish_next_state,MtpDraftPlanStep::output_head_mix,MtpDraftPlanStep::lm_head};
void require(bool ok,const char* message) {if(!ok) throw std::invalid_argument(message);}
void check(hipError_t e,const char* message) {if(e!=hipSuccess) throw std::runtime_error(message);}
bool same_session(const SerialSessionContext& a,const SerialSessionContext& b) {
    return a.execution_identity==b.execution_identity && a.runtime_compatibility_identity==b.runtime_compatibility_identity &&
        a.state_layout_identity==b.state_layout_identity && a.next_position==b.next_position && a.committed_tokens==b.committed_tokens &&
        a.health==SerialSessionHealth::ready && b.health==SerialSessionHealth::ready && a.state.valid==b.state.valid &&
        a.state.generations==b.state.generations && arithmetic_lineage_matches(a.arithmetic_lineage,b.arithmetic_lineage,a.next_position);
}
bool same_input(const MtpDraftCycleInput& a,const MtpDraftCycleInput& b) {
    return a.proposal_identity==b.proposal_identity && a.activation_identity==b.activation_identity &&
        a.target_runtime_compatibility_identity==b.target_runtime_compatibility_identity && a.seed_token_id==b.seed_token_id &&
        a.seed_position==b.seed_position && a.hidden_source_position==b.hidden_source_position && a.hidden_address==b.hidden_address &&
        a.hidden_elements==b.hidden_elements && a.hidden_element_bytes==b.hidden_element_bytes &&
        a.hidden_allocation_generation==b.hidden_allocation_generation;
}
void require_chain(const HipMtpTargetHiddenTicket& a,const HipMtpTargetHiddenTicket& b) {
    require(a.valid() && b.valid() && a.same_producer(b) && a.source_position()!=UINT64_MAX &&
        b.source_position()==a.source_position()+1U,"MTP target tickets are not one actual consecutive producer");
    const auto& previous=a.committed_session();const auto& next=b.committed_session();
    if(a.same_commit(b)) {
        // Both snapshots belong to the same real whole-window commit. There is
        // deliberately no invented SessionContext for an intermediate row.
        require(a.is_window() && b.is_window() && a.chunk_begin()==b.chunk_begin() &&
            a.chunk_end()==b.chunk_end() && same_session(previous,next),
            "MTP adjacent rows do not retain the same actual whole-window commit");
        return;
    }
    // Consecutive rows keep one executor, except across the producer's admitted prompt handoff
    // (split prefill profile), where the lineage below still proves the exact continuation.
    const bool same_executor=previous.execution_identity==next.execution_identity &&
        previous.runtime_compatibility_identity==next.runtime_compatibility_identity;
    require(a.source_position()+1U==a.chunk_end() && b.source_position()==b.chunk_begin() &&
        (same_executor || a.admits_executor_change(b)) &&
        previous.state_layout_identity==next.state_layout_identity && previous.next_position==b.chunk_begin() &&
        b.chunk_end()>b.chunk_begin() && next.next_position==b.chunk_end() &&
        b.chunk_end()-b.chunk_begin()<=UINT64_MAX-previous.committed_tokens &&
        next.committed_tokens==previous.committed_tokens+(b.chunk_end()-b.chunk_begin()) &&
        !next.arithmetic_lineage.intervals.empty(),"MTP target ticket chunk frontier changed");
    const auto& actual=next.arithmetic_lineage.intervals.back();
    ArithmeticLineage history;
    if(b.is_window()) {
        require(actual.producer==ArithmeticProducer::known_window_chunk && actual.begin==b.chunk_begin() &&
            actual.end==b.chunk_end(),"MTP target window lacks its actual whole-chunk producer receipt");
        history=prepare_known_window_chunk_lineage(previous.arithmetic_lineage,b.chunk_begin(),b.chunk_end(),
            actual.numerical_plan_identity,actual.reduction_plan_identity,actual.state_update_identity);
    } else {
        require(b.chunk_end()==b.chunk_begin()+1U && actual.producer==ArithmeticProducer::serial_row,
            "MTP target serial ticket is not one actual serial row");
        const auto producer=b.producer_execution_identity();
        require(producer==next.execution_identity || (!same_executor && a.admits_executor_change(b)),
            "MTP target serial ticket producer differs from its session executor");
        history=prepare_serial_row_lineage(previous.arithmetic_lineage,b.source_position(),producer,
            actual.reduction_plan_identity);
    }
    require(arithmetic_lineage_matches(history,next.arithmetic_lineage,next.next_position),
        "MTP target ticket arithmetic history is not the actual producer continuation");
}
}
namespace {
enum class DraftCacheProfile {mixed,fast,teacher};
DraftCacheProfile configured_draft_cache_profile() {
    const char* v=std::getenv("STRIX_MTP_DRAFT_CACHE");
    if(!v || !*v || std::strcmp(v,"mixed")==0)return DraftCacheProfile::mixed;
    if(std::strcmp(v,"fast")==0)return DraftCacheProfile::fast;
    if(std::strcmp(v,"teacher")==0)return DraftCacheProfile::teacher;
    throw std::invalid_argument("MTP draft cache must be mixed, fast or teacher");
}
}
struct HipMtpRuntime::Impl {
    std::shared_ptr<HipMtpResources> owner;
    std::unique_ptr<HipGreedyOutput> selector;
    std::unique_ptr<HipMtpDraftHead> draft;
    std::unique_ptr<float,void(*)(float*)> fast_workspace{nullptr,[](float* p){(void)hipFree(p);}};
    // F32 copies of the attention/MLP HC injection matrices for the split-K draft mix.
    std::unique_ptr<float,void(*)(float*)> hc_inject{nullptr,[](float* p){(void)hipFree(p);}};
    bool draft_exact=false;
    DraftCacheProfile cache_profile=configured_draft_cache_profile();
    std::optional<HipMtpResources::NativeValidationScope> validation;
    std::optional<HipMtpTargetHiddenTicket> last_target,cycle_base,base_previous_target,reconcile_final;
    std::optional<std::uint32_t> last_teacher_next_input,base_previous_input;
    std::vector<HipMtpReplayRow> replay;
    std::vector<std::uint32_t> candidates;
    std::vector<float> probabilities,runner_ups; // runner_ups: draft head runner-up probabilities (0 unless enabled)
    // Sampled drafts (null: greedy drafts): one distribution per proposed draft of the last chain.
    std::shared_ptr<TokenSampler> sampler;
    std::vector<TokenDraft> sampled;
    std::vector<float> draft_logits;
    // Drafts sampled on the device (device_draft_sampling_enabled): no per-step wait or readback.
    std::unique_ptr<HipTokenDraftSamplerDevice> draft_device;
    MtpDraftCycleInput current;
    HipMtpExecutionStats stats;
    std::uint64_t next_source=0,base_source=0,epoch=0,profile=0;
    // Rows [0, ticketless_frontier) were primed without target tickets (tiled prompt or restore):
    // the first ticketed teacher row may start there instead of at source zero.
    std::uint64_t ticketless_frontier=0;
    std::uint32_t step=0,pending_token=0;
    bool active=false,cycle=false,have_replay=false,broken=false,cache_only_active=false,first_cache_reusable=false;
    // Rows of a submitted, not yet completed prime_hidden_rows_submit (STRIX_PREFILL_PIPE).
    std::size_t primed_rows=0;
    // Multi-session draft batch: the batch stream carrying this chain's steps (null outside a batch) and the
    // entry/exit events ordering it with the owner stream.
    hipStream_t batch_stream=nullptr;
    hipEvent_t batch_entry=nullptr,batch_exit=nullptr;
    struct BatchReference {
        std::uint32_t depth=0,steps=0;std::uint64_t first=0;
        std::vector<std::uint32_t> tokens;std::vector<float> probabilities,runner_ups;
        // Per step and stage: every step buffer, attention workspace, status words, fast-MoE workspace, head logits, slots.
        std::vector<std::vector<std::vector<std::vector<std::uint8_t>>>> stages;
        std::vector<std::uint8_t> kv,index;std::uint64_t prefix_hash=0,guard_rows=0;
        std::string mismatch;
    };
    std::optional<BatchReference> batch_reference;
    // STRIX_MTP_DRAFT_BATCH_VERIFY only: called after each of the 12 stages of a one-token draft step (null otherwise).
    std::function<void(unsigned)> stage_hook;
    void stage_mark(unsigned stage) {if(stage_hook) stage_hook(stage);}
    void batch_verify_poison(std::uint64_t first,std::uint64_t rows,std::uint64_t guard);
    std::uint64_t verify_previous_first=0;std::uint32_t verify_previous_steps=0;
    explicit Impl(std::shared_ptr<HipMtpResources> actual):owner(std::move(actual)) {
        require(owner && !owner.owner_before(owner->weak_from_this().lock()) &&
            !owner->weak_from_this().lock().owner_before(owner),"MTP runtime requires its actual retained native owner");
        owner->require_idle();
        require(owner->plan().steps.size()==native_steps.size() &&
            std::equal(native_steps.begin(),native_steps.end(),owner->plan().steps.begin()),"MTP native step order is unsupported");
        candidates.reserve(owner->budget().maximum_candidate_depth);replay.reserve(owner->budget().maximum_candidate_depth+1U);
        const auto pool=owner->view(HipMtpBuffer::wide_current);
        selector=std::make_unique<HipGreedyOutput>(static_cast<std::uint32_t>(owner->model().vocab_size),
            ExecutionMemoryBinding{owner->physical_identity(),pool.address,owner->reservation().device_pool_bytes,pool.allocation_generation},owner,owner->guard());
        const auto input=fusion();profile=hip_mtp_input_fusion_profile_identity(input);
        for(const auto word:{owner->plan().forward_semantics.contract_identity,owner->plan().plan_identity,
            hip_mtp_lm_head_profile_identity(owner->model(),owner->tensors().lm_head),
            std::uint64_t{0x4d54504341535431ULL}, // durable cache BF16-cast contract v1
            std::uint64_t{0x4d545052415731ULL}})
            for(unsigned shift=0;shift<64;shift+=8) profile=(profile^((word>>shift)&255U))*1099511628211ULL;
        if(!profile) profile=1U;
    }
    HipMtpInputFusionRequest fusion() const {
        const auto device=[](MtpDeviceView v) {return HipMtpInputFusionView{v.address,v.bytes,v.allocation_generation};};
        const auto tensor=[](MtpTensorView v) {return HipMtpInputFusionView{v.runtime->address,v.runtime->bytes,v.runtime->allocation_generation};};
        HipMtpInputFusionRequest r;const auto& m=owner->model();const auto& t=owner->tensors();
        r.hidden_size=static_cast<std::uint32_t>(m.hidden_size);r.hc_count=static_cast<std::uint32_t>(m.hc_count);r.epsilon=m.rms_norm_epsilon;
        r.embedding_storage=r.hidden_storage=r.output_storage=HipMtpInputStorage::bf16;
        r.normalized_rounding=owner->budget().normalized_rounding;r.projection_rounding=owner->budget().projection_rounding;
        r.norm_weight_encoding=owner->shared_target().source.norm_weight_encoding;r.norm_encoding_admitted=true;
        r.arithmetic=owner->plan().forward_semantics.input_arithmetic;
        r.embedding=device(owner->view(HipMtpBuffer::embedding));r.wide_hidden=device(owner->view(HipMtpBuffer::wide_current));
        r.embedding_norm=tensor(t.enorm);r.hidden_norm=tensor(t.hnorm);r.packed_projection=tensor(t.eh_projection);
        r.projection_row_stride_bytes=(m.hidden_size*2U/32U)*34U;
        r.normalized_scratch=device(owner->view(HipMtpBuffer::normalized_input));r.projection_scratch=device(owner->view(HipMtpBuffer::projected_input));
        r.output=device(owner->view(HipMtpBuffer::fused_wide));return r;
    }
    MtpNativeContext context(std::uint64_t position) const {return {&owner->model(),&owner->plan(),&owner->tensors(),owner.get(),position};}
    void preflight(std::uint64_t position,std::uint32_t token,bool already_admitted=false,bool cache_only=false) const {
        require(!broken && !active && !current_hip_launch_context(),"MTP native input requires one healthy idle host protocol");
        std::optional<HipMtpResources::NativeValidationScope> local_validation;
        if(!already_admitted && !validation) local_validation.emplace(owner->admit_native_entry());
        const auto c=context(position);mtp_native::context(c);
        hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
        check(hipStreamIsCapturing(static_cast<hipStream_t>(owner->stream()),&capture),"MTP capture query failed");
        require(capture==hipStreamCaptureStatusNone,"MTP native runtime is eager only");
        preflight_hip_mtp_embedding(c,token);preflight_hip_mtp_input_fusion(fusion());
        preflight_hip_mtp_hyper_mix(c,MtpHyperRole::attention);
        if(cache_only){preflight_hip_mtp_cache_rows(c);return;}
        preflight_hip_mtp_attention(c);
        preflight_hip_mtp_inject(c,MtpInjectRole::attention);preflight_hip_mtp_hyper_mix(c,MtpHyperRole::mlp);
        preflight_hip_mtp_route(c);preflight_hip_mtp_shared_expert(c);preflight_hip_mtp_routed_experts(c);
        preflight_hip_mtp_moe_combine(c);preflight_hip_mtp_inject(c,MtpInjectRole::mlp);
        preflight_hip_mtp_hyper_mix(c,MtpHyperRole::head);preflight_hip_mtp_lm_head(c);
    }
    void validate_ticket(const HipMtpTargetHiddenTicket& ticket) const {
        require(ticket.valid() && ticket.target_content_identity()==owner->shared_target().target_model_content_identity,
            "MTP target hidden is not an actual compatible committed ticket");
        ticket.admit_rotary_reader(owner->model().rotary_identity());
        const auto v=ticket.view();const auto& session=ticket.committed_session();
        require(v.bytes==owner->view(HipMtpBuffer::wide_current).bytes && v.address && v.allocation_generation &&
            ticket.source_position()<owner->context_capacity() && ticket.source_position()!=UINT64_MAX &&
            ticket.chunk_begin()<=ticket.source_position() && ticket.source_position()<ticket.chunk_end() &&
            session.next_position==ticket.chunk_end() && session.health==SerialSessionHealth::ready &&
            arithmetic_lineage_valid(session.arithmetic_lineage,session.next_position),"MTP target hidden width/frontier/history is invalid");
        const auto& actual=session.arithmetic_lineage.intervals.back();
        if(ticket.is_window()) {
            require(actual.producer==ArithmeticProducer::known_window_chunk && actual.begin==ticket.chunk_begin() &&
                actual.end==ticket.chunk_end(),"MTP target window ticket lacks its actual chunk history");
        } else if(ticket.is_prefill_block()) {
            require((actual.producer==ArithmeticProducer::prefill_chunk || actual.producer==ArithmeticProducer::known_window_chunk) &&
                actual.begin==ticket.chunk_begin() && actual.end==ticket.chunk_end() && ticket.source_position()+1U==ticket.chunk_end(),
                "MTP target prefill-block ticket lacks its actual block history");
        } else {
            require(ticket.chunk_begin()==ticket.source_position() && ticket.chunk_end()==ticket.source_position()+1U &&
                actual.producer==ArithmeticProducer::serial_row,"MTP target serial ticket frontier differs from its actual row");
        }
    }
    void copy(MtpDeviceView destination,MtpDeviceView source) {
        require(destination.bytes==source.bytes,"MTP copy extents differ");
        check(hipMemcpyAsync(reinterpret_cast<void*>(destination.address),reinterpret_cast<const void*>(source.address),
            static_cast<std::size_t>(source.bytes),hipMemcpyDeviceToDevice,static_cast<hipStream_t>(owner->stream())),"MTP owned copy enqueue failed");
    }
    void failure() noexcept {
        if(owner->qsa_pool()) owner->qsa_pool()->cancel();
        if(batch_stream) {(void)hipStreamSynchronize(batch_stream);batch_stream=nullptr;}
        broken=true;active=false;cache_only_active=false;first_cache_reusable=false;primed_rows=0;owner->fail();validation.reset();
    }
    // The owner stream waits for every batch step enqueued for this chain (before completion or reuse).
    void batch_leave() {
        if(!batch_stream) return;
        if(!batch_exit) check(hipEventCreateWithFlags(&batch_exit,hipEventDisableTiming),"MTP draft batch exit event");
        check(hipEventRecord(batch_exit,batch_stream),"MTP draft batch exit record");
        check(hipStreamWaitEvent(static_cast<hipStream_t>(owner->stream()),batch_exit,0),"MTP draft batch exit wait");
        batch_stream=nullptr;
    }
    ~Impl() {
        if(batch_stream) (void)hipStreamSynchronize(batch_stream);
        if(batch_entry) (void)hipEventDestroy(batch_entry);
        if(batch_exit) (void)hipEventDestroy(batch_exit);
    }
    void stage(const MtpForwardStepContext& call) {
        const auto& input=*call.cycle_input;
        require(active && !cache_only_active && step<native_steps.size() && call.step_index==step && call.step==native_steps[step] &&
            call.input_token==current.seed_token_id && call.target_position==current.seed_position &&
            call.predictor_position==current.hidden_source_position && call.plan_identity==owner->plan().plan_identity &&
            call.proposal_identity==owner->plan().proposal_identity && same_input(input,current),"MTP native dispatch order/input changed");
        require(current.hidden_source_position!=UINT64_MAX &&
            current.seed_position==current.hidden_source_position+1U &&
            call.predictor_position+1U==call.target_position,
            "MTP hidden source/cache position i requires logical seed i+1");
        const auto c=context(current.hidden_source_position);void* stream=owner->stream();
        switch(call.step) {
        case MtpDraftPlanStep::input_fusion:launch_hip_mtp_embedding(c,current.seed_token_id,stream);launch_hip_mtp_input_fusion(fusion(),stream);break;
        case MtpDraftPlanStep::attention_hyper_mix:launch_hip_mtp_hyper_mix(c,MtpHyperRole::attention,stream);break;
        case MtpDraftPlanStep::qsa_indexer:launch_hip_mtp_indexer(c,stream);break;
        case MtpDraftPlanStep::sparse_attention:launch_hip_mtp_sparse_attention(c,stream);break;
        case MtpDraftPlanStep::attention_combine:launch_hip_mtp_inject(c,MtpInjectRole::attention,stream);break;
        case MtpDraftPlanStep::mlp_hyper_mix:launch_hip_mtp_hyper_mix(c,MtpHyperRole::mlp,stream);break;
        case MtpDraftPlanStep::routed_and_shared_moe:
            launch_hip_mtp_route(c,stream);launch_hip_mtp_shared_expert(c,stream);launch_hip_mtp_routed_experts(c,stream);
            launch_hip_mtp_moe_combine(c,stream);break;
        case MtpDraftPlanStep::mlp_combine:launch_hip_mtp_inject(c,MtpInjectRole::mlp,stream);break;
        case MtpDraftPlanStep::publish_next_state:copy(owner->view(HipMtpBuffer::wide_current),owner->view(HipMtpBuffer::wide_next));break;
        case MtpDraftPlanStep::output_head_mix:launch_hip_mtp_hyper_mix(c,MtpHyperRole::head,stream);break;
        case MtpDraftPlanStep::lm_head:launch_hip_mtp_lm_head(c,stream);break;
        }
        ++step;
    }
    void start(MtpDraftCycleInput input,const HipMtpTargetHiddenTicket* source,bool cache_only=false) {
        require(input.hidden_source_position==next_source && input.seed_position==next_source+1U &&
            input.seed_token_id<owner->model().vocab_size,"MTP source position/seed is not the next native row");
        require(!active && !validation,"MTP input cannot replace an active native admission");
        if(source) validate_ticket(*source);
        validation.emplace(owner->admit_native_entry());
        try {
            preflight(next_source,input.seed_token_id,false,cache_only);
            require(cache_only?stats.cache_only_rows!=UINT64_MAX:stats.full_forward_rows!=UINT64_MAX,
                "MTP completed row counter exhausted");
            if(source) validate_ticket(*source);
            require(epoch!=UINT64_MAX,"MTP native epoch exhausted");
            owner->mark_submitted();
        } catch(...) {validation.reset();throw;}
        ++epoch;
        active=true;cache_only_active=cache_only;step=0;current=input;
        try {if(source) copy(owner->view(HipMtpBuffer::wide_current),source->view());} catch(...) {failure();throw;}
    }
    std::uint32_t finish() {
        require(active && !cache_only_active && step==native_steps.size(),"MTP native output requested before all eleven steps");
        try {
            owner->complete();const auto logits=owner->view(HipMtpBuffer::logits);
            const auto token=selector->select(ExecutionBufferAddress{{},logits.address,logits.bytes,logits.allocation_generation},
                static_cast<std::uint32_t>(owner->model().vocab_size),current.hidden_source_position,epoch);
            active=false;++next_source;++stats.full_forward_rows;validation.reset();return token;
        } catch(...) {failure();throw;}
    }
    std::uint32_t run_current() {
        for(std::size_t i=0;i<native_steps.size();++i) {
            MtpForwardStepContext call;call.step=native_steps[i];call.step_index=static_cast<std::uint32_t>(i);
            call.input_token=current.seed_token_id;call.target_position=current.seed_position;call.predictor_position=current.hidden_source_position;
            call.proposal_identity=current.proposal_identity;call.plan_identity=owner->plan().plan_identity;call.cycle_input=&current;stage(call);
        }
        return finish();
    }
    HipMtpNativeRowResult teacher(const HipMtpTargetHiddenTicket& ticket,std::uint32_t token) {
        validate_ticket(ticket);require(ticket.source_position()==next_source,"MTP teacher row is not contiguous");
        if(last_target) {
            require_chain(*last_target,ticket);
            require(last_teacher_next_input && *last_teacher_next_input==ticket.input_token(),"MTP prior teacher pair differs from the actual next target input");
        } else require(next_source==ticketless_frontier,"MTP initial history must start at source zero");
        const auto v=owner->view(HipMtpBuffer::wide_current);
        MtpDraftCycleInput input{owner->plan().proposal_identity,0U,ticket.committed_session().runtime_compatibility_identity,
            token,next_source+1U,next_source,v.address,owner->plan().state.state_width,2U,v.allocation_generation};
        start(input,&ticket);
        try {
            const auto selected=run_current();last_target=ticket;last_teacher_next_input=token;
            if(owner->qsa_pool()) owner->qsa_pool()->commit(true,next_source);
            return {input.hidden_source_position,input.seed_position,profile,selected,static_cast<std::uint32_t>(native_steps.size()),
                owner->view(HipMtpBuffer::wide_current),owner->view(HipMtpBuffer::logits),true};
        } catch(...) {failure();throw;}
    }
    void teacher_cache(const HipMtpTargetHiddenTicket& ticket,std::uint32_t token) {
        validate_ticket(ticket);require(ticket.source_position()==next_source,"MTP teacher cache row is not contiguous");
        if(last_target) {
            require_chain(*last_target,ticket);
            require(last_teacher_next_input && *last_teacher_next_input==ticket.input_token(),
                "MTP prior cache teacher pair differs from the actual next target input");
        } else require(next_source==ticketless_frontier,"MTP initial cache history must start at source zero");
        const auto v=owner->view(HipMtpBuffer::wide_current);
        MtpDraftCycleInput input{owner->plan().proposal_identity,0U,ticket.committed_session().runtime_compatibility_identity,
            token,next_source+1U,next_source,v.address,owner->plan().state.state_width,2U,v.allocation_generation};
        // Copy the authoritative wide source on every teacher row. No previous
        // predictor-wide result, logits or scratch payload is a cache input.
        start(input,&ticket,true);
        try {
            require(current.hidden_source_position!=UINT64_MAX &&
                current.seed_position==current.hidden_source_position+1U,
                "MTP cache-only source i requires logical seed i+1");
            const auto c=context(current.hidden_source_position);void* stream=owner->stream();
            launch_hip_mtp_embedding(c,current.seed_token_id,stream);
            cache_row_front(c,stream);
            launch_hip_mtp_cache_rows(c,stream);
            owner->complete();
            active=false;cache_only_active=false;++next_source;++stats.cache_only_rows;validation.reset();
            if(owner->qsa_pool()) owner->qsa_pool()->commit(true,next_source);
            last_target=ticket;last_teacher_next_input=token;
        } catch(...) {failure();throw;}
    }
    void reuse_first_teacher_cache(const HipMtpReplayRow& row) {
        // The only caller is reconciliation after its full metadata admission.
        // The proof is private and minted after the first full proposal row's
        // native completion and finite selector, never by predicted-token match.
        require(first_cache_reusable && cycle_base && !broken && !active && !validation &&
            row.hidden.same_commit(*cycle_base) && row.hidden.source_position()==base_source &&
            row.hidden.view()==cycle_base->view() && row.next_input_token==pending_token && next_source==base_source,
            "MTP first cache reuse lacks its actual completed proposal source proof");
        validate_ticket(row.hidden);
        if(last_target) {
            require_chain(*last_target,row.hidden);
            require(last_teacher_next_input && *last_teacher_next_input==row.hidden.input_token(),
                "MTP reused cache teacher pair differs from the actual preceding input");
        } else require(next_source==ticketless_frontier,"MTP reused initial cache history must start at source zero");
        owner->require_idle();
        require(stats.cache_reused_rows!=UINT64_MAX,"MTP reused cache row counter exhausted");
        ++next_source;++stats.cache_reused_rows;
        if(owner->qsa_pool()) owner->qsa_pool()->commit(true,next_source);
        last_target=row.hidden;last_teacher_next_input=row.next_input_token;
    }
    // Batched teacher cache rows: same per-row launches as teacher_cache, one
    // admission and one completion for the whole contiguous run.
    void teacher_cache_rows(std::span<const HipMtpReplayRow> rows,bool fast_cache=false) {
        if(rows.empty()) return;
        require(!active && !validation && stats.cache_only_rows<=UINT64_MAX-rows.size(),"MTP batched cache rows require an idle owner");
        for(const auto& row:rows) row.hidden.admit_rotary_reader(owner->model().rotary_identity());
        validation.emplace(owner->admit_native_entry());
        try {
            for(std::size_t i=0;i<rows.size();++i) {
                const auto& row=rows[i];validate_ticket(row.hidden);
                require(row.hidden.source_position()==next_source+i,"MTP teacher cache row is not contiguous");
                if(i) {
                    require_chain(rows[i-1U].hidden,row.hidden);
                    require(rows[i-1U].next_input_token==row.hidden.input_token(),"MTP prior cache teacher pair differs from the actual next target input");
                } else if(last_target) {
                    require_chain(*last_target,row.hidden);
                    require(last_teacher_next_input && *last_teacher_next_input==row.hidden.input_token(),
                        "MTP prior cache teacher pair differs from the actual next target input");
                } else require(next_source==ticketless_frontier,"MTP initial cache history must start at source zero");
                preflight(row.hidden.source_position(),row.next_input_token,true,true);
            }
            owner->mark_submitted();
        } catch(...) {validation.reset();throw;}
        active=true;cache_only_active=true;
        try {
            void* stream=owner->stream();
            for(const auto& row:rows) {
                const auto c=context(row.hidden.source_position());
                copy(owner->view(HipMtpBuffer::wide_current),row.hidden.view());
                launch_hip_mtp_embedding(c,row.next_input_token,stream);cache_row_front(c,stream);
                launch_hip_mtp_cache_rows(c,stream,fast_cache);
            }
            owner->complete();
            active=false;cache_only_active=false;next_source+=rows.size();stats.cache_only_rows+=rows.size();validation.reset();
            if(owner->qsa_pool()) owner->qsa_pool()->commit(true,next_source);
            last_target=rows.back().hidden;last_teacher_next_input=rows.back().next_input_token;
        } catch(...) {failure();throw;}
    }
    // Sync-free proposal chain: identical stage launches to run_current per row,
    // except the head/selector and the device-fed next input token.
    // Each draft completes before the next when min_probability > 0 or keep_drafting is set.
    // The draft chain in phases (chain() runs them in order; a cohort interleaves the phases of several
    // sessions, each on its own stream): begin, then per step launch and decide, then finish.
    struct ChainState {
        std::uint32_t requested=0,depth=0,k=0;float min_probability=0.0F;
        std::function<bool(std::span<const float>)> keep;
        bool device_draft=false,stop=false;
        const std::uint16_t* hidden=nullptr;
    } cs;
    std::vector<std::uint32_t> chain(std::uint32_t depth,float min_probability,
        const std::function<bool(std::span<const float>)>& keep_drafting) {
        if(!chain_begin(depth,min_probability,keep_drafting)) return {};
        while(chain_launch_step() && chain_after_step()) {}
        return chain_finish();
    }
    // False: nothing to draft (depth 0).
    bool chain_begin(std::uint32_t depth,float min_probability,const std::function<bool(std::span<const float>)>& keep_drafting) {
        require(draft && cycle && cycle_base && !active && !validation && candidates.empty() &&
            depth<=owner->budget().maximum_candidate_depth,"MTP draft chain requires a fresh bounded proposal");
        sampled.clear();
        if(!depth) return false;
        require(stats.full_forward_rows<=UINT64_MAX-depth && epoch!=UINT64_MAX,"MTP draft chain counters exhausted");
        cycle_base->admit_rotary_reader(owner->model().rotary_identity());
        validation.emplace(owner->admit_native_entry());
        try {
            for(std::uint32_t k=0;k<depth;++k) preflight(next_source+k,k?0U:pending_token,true);
            validate_ticket(*cycle_base);owner->mark_submitted();
        } catch(...) {validation.reset();throw;}
        ++epoch;active=true;
        try {
            void* stream=owner->stream();
            cs=ChainState{};cs.requested=cs.depth=depth;cs.min_probability=min_probability;cs.keep=keep_drafting;
            cs.hidden=reinterpret_cast<const std::uint16_t*>(owner->view(HipMtpBuffer::hidden0).address);
            copy(owner->view(HipMtpBuffer::wide_current),cycle_base->view());
            draft->begin_chain();
            cs.device_draft=sampler && device_draft_sampling_enabled(sampler->policy());
            if(cs.device_draft) {
                if(!draft_device) draft_device=std::make_unique<HipTokenDraftSamplerDevice>(draft->capacity(),
                    std::max<std::uint32_t>(owner->budget().maximum_candidate_depth,1U));
                draft_device->set_vocabulary(draft->host_ids(),draft->stats().rows,stream);
            }
            ple_window_warm::begin(pending_token); // STRIX_PLE_WINDOW_WARM: verify row 0
            return true;
        } catch(...) {failure();throw;}
    }
    // Enqueues step k; false when the chain has no step left.
    bool chain_launch_step() {
        if(cs.stop || cs.k>=cs.requested) return false;
        try {
            void* stream=owner->stream();
            const auto* hidden=cs.hidden;
            const bool device_draft=cs.device_draft;
            const auto k=cs.k;
            {
                const auto c=context(next_source+k);
                if(k) launch_hip_mtp_embedding_device(c,draft->device_token(k-1U),stream);
                else launch_hip_mtp_embedding(c,pending_token,stream);
                if(draft_exact) launch_hip_mtp_input_fusion(fusion(),stream);
                else launch_hip_mtp_input_fusion_draft(fusion(),stream);
                stage_mark(0);
                hyper(c,MtpHyperRole::attention,stream);stage_mark(1);
                launch_hip_mtp_indexer(c,stream);stage_mark(2);
                if(draft_exact) launch_hip_mtp_sparse_attention(c,stream); else launch_hip_mtp_sparse_attention_draft(c,stream);
                stage_mark(3);
                launch_hip_mtp_inject(c,MtpInjectRole::attention,stream);stage_mark(4);
                hyper(c,MtpHyperRole::mlp,stream);stage_mark(5);
                launch_hip_mtp_route(c,stream);stage_mark(6);
                if(draft_exact) {
                    launch_hip_mtp_shared_expert(c,stream);launch_hip_mtp_routed_experts(c,stream);launch_hip_mtp_moe_combine(c,stream);
                } else launch_hip_mtp_fast_moe(fast_moe(),stream);
                stage_mark(7);
                launch_hip_mtp_inject(c,MtpInjectRole::mlp,stream);stage_mark(8);
                copy(owner->view(HipMtpBuffer::wide_current),owner->view(HipMtpBuffer::wide_next));stage_mark(9);
                hyper(c,MtpHyperRole::head,stream);stage_mark(10);
                draft->launch(hidden,k,stream);stage_mark(11);
                if(device_draft) {
                    // The sampled token overwrites the argmax in the step's slot before the next step reads it.
                    const auto t0=std::chrono::steady_clock::now();
                    draft_device->step(*sampler,k,draft->logits(),next_source+k+2U,draft->device_token_slot(k),stream);
                    if(sample_timing_enabled()) record_sample_timing(SampleTimingKind::draft,1U,0.0,
                        std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(),1U);
                } else if(sampler) {
                    // Draft k proposes the token at next_source + k + 2 (row k's seed sits at next_source + k + 1).
                    const auto t0=std::chrono::steady_clock::now();
                    draft->read_logits(draft_logits);
                    const auto t1=std::chrono::steady_clock::now();
                    sampled.push_back(sampler->draft(draft_logits,draft->host_ids(),next_source+k+2U));
                    if(sample_timing_enabled()) record_sample_timing(SampleTimingKind::draft,1U,
                        std::chrono::duration<double>(t1-t0).count(),
                        std::chrono::duration<double>(std::chrono::steady_clock::now()-t1).count(),1U);
                    draft->set_token(k,sampled.back().token,stream);
                }
            }
            host_phase_mark("2l draft step launch");
            return true;
        } catch(...) {failure();throw;}
    }
    // After step k: the minimum-probability / keep test (waits for the step when one applies); false stops.
    bool chain_after_step() {
        try {
            const auto k=cs.k++;
            if((cs.min_probability>0.0F || cs.keep) && k+1U<cs.requested) {
                if(!sampler || cs.device_draft) draft->wait_step(k,batch_stream?static_cast<void*>(batch_stream):owner->stream());
                host_phase_mark("2w draft step wait");
                if(ple_window_warm::enabled()) ple_window_warm::draft(k,draft->step_token(k)); // verify row k + 1
                if(cs.min_probability>0.0F && draft->probability(k)<cs.min_probability) {cs.depth=k+1U;cs.stop=true;return false;}
                if(cs.keep) {
                    probabilities.resize(k+1U);runner_ups.resize(k+1U);
                    for(std::uint32_t j=0;j<=k;++j) {probabilities[j]=draft->probability(j);runner_ups[j]=draft->runner_up_probability(j);}
                    if(!cs.keep(probabilities)) {cs.depth=k+1U;cs.stop=true;return false;}
                }
            }
            return cs.k<cs.requested;
        } catch(...) {failure();throw;}
    }
    std::vector<std::uint32_t> chain_finish() {
        try {
            batch_leave();
            auto depth=cs.depth;
            const bool device_draft=cs.device_draft;
            owner->complete();
            if(device_draft) {
                draft_device->finish(*sampler,depth,sampled);
                for(std::uint32_t k=0;k<depth;++k) draft->record_token(k,sampled[k].token);
            }
            auto tokens=draft->read(depth);
            for(std::uint32_t k=0;k<depth;++k) ple_window_warm::draft(k,tokens[k]); // rows not warmed per step
            probabilities.resize(depth);runner_ups.resize(depth);
            for(std::uint32_t k=0;k<depth;++k) {probabilities[k]=draft->probability(k);runner_ups[k]=draft->runner_up_probability(k);}
            // STRIX_MTP_DRAFT_LOG=1: every proposal (chain start position, step, token, probability bits and
            // runner-up probability bits, zero unless the head reduces runner-ups) on stderr.
            static const bool log_drafts=std::getenv("STRIX_MTP_DRAFT_LOG")!=nullptr;
            // With STRIX_MTP_DRAFT_TOP2=1 each line also carries the step's runner-up token and probability.
            if(log_drafts) for(std::uint32_t k=0;k<depth;++k) {
                std::uint32_t bits=0,bits2=0;std::memcpy(&bits,&probabilities[k],sizeof(bits));std::memcpy(&bits2,&runner_ups[k],sizeof(bits2));
                std::uint32_t second=0;float q=0.0F;
                if(draft->runner_up(k,second,q)) std::fprintf(stderr,"draft at=%llu k=%u token=%u p=%08x p2=%08x t2=%u q2=%.6f p1=%.6f\n",
                    static_cast<unsigned long long>(next_source),k,tokens[k],bits,bits2,second,q,probabilities[k]);
                else std::fprintf(stderr,"draft at=%llu k=%u token=%u p=%08x p2=%08x\n",static_cast<unsigned long long>(next_source),k,tokens[k],bits,bits2);
            }
            active=false;next_source+=depth;stats.full_forward_rows+=depth;validation.reset();
            // Row zero is the pending cycle input at base_source, as in read_greedy_token.
            first_cache_reusable=true;candidates=tokens;return tokens;
        } catch(...) {failure();throw;}
    }
    // Teacher cache rows only feed the predictor's own attention cache, so a
    // draft-only chain builds them with the same fast kernels as its proposals.
    void cache_row_front(const MtpNativeContext& c,void* stream) const {
        if(draft && !draft_exact) {launch_hip_mtp_input_fusion_draft(fusion(),stream);hyper(c,MtpHyperRole::attention,stream);return;}
        launch_hip_mtp_input_fusion(fusion(),stream);launch_hip_mtp_hyper_mix(c,MtpHyperRole::attention,stream);
    }
    void hyper(const MtpNativeContext& c,MtpHyperRole role,void* stream) const {
        if(draft_exact) {launch_hip_mtp_hyper_mix(c,role,stream);return;}
        constexpr std::uint64_t inject=4U*10240U;
        launch_hip_mtp_hyper_mix_draft(c,role,role==MtpHyperRole::attention?hc_inject.get():
            role==MtpHyperRole::mlp?hc_inject.get()+inject:nullptr,stream);
    }
    HipMtpFastMoeViews fast_moe() const {
        const auto& t=owner->tensors();const auto address=[](MtpTensorView v){return reinterpret_cast<const void*>(v.runtime->address);};
        HipMtpFastMoeViews v;
        v.input=reinterpret_cast<const std::uint16_t*>(owner->view(HipMtpBuffer::hidden0).address);
        v.routes=reinterpret_cast<const std::uint32_t*>(owner->view(HipMtpBuffer::routes).address);
        v.route_weights=reinterpret_cast<const float*>(owner->view(HipMtpBuffer::route_weights).address);
        v.gate=address(t.expert_gate);v.up=address(t.expert_up);v.down=address(t.expert_down);
        v.shared_gate=address(t.shared_gate);v.shared_up=address(t.shared_up);v.shared_down=address(t.shared_down);
        v.shared_scalar=static_cast<const float*>(address(t.shared_scalar_gate));v.workspace=fast_workspace.get();
        v.output=reinterpret_cast<std::uint16_t*>(owner->view(HipMtpBuffer::hidden0).address);
        v.status=reinterpret_cast<std::uint32_t*>(owner->view(HipMtpBuffer::status).address);
        return v;
    }
    void qualification(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated) const {
        require_qualified_mtp_draft_plan(plan);
        require(plan.base.plan_identity==owner->plan().plan_identity && plan.base.proposal_identity==owner->plan().proposal_identity &&
            plan.native_arithmetic_profile_identity==profile && activated.native_arithmetic_profile_identity==profile &&
            activated.semantics_qualified && activated.structural_plan_identity==plan.base.plan_identity &&
            activated.proposal_identity==plan.base.proposal_identity && activated.activation_identity,
            "MTP native qualified entry does not match its actual source/profile plan");
    }
};
HipMtpRuntime::HipMtpRuntime(std::shared_ptr<HipMtpResources> owner):impl_(std::make_unique<Impl>(std::move(owner))) {}
HipMtpRuntime::~HipMtpRuntime()=default;
const HipMtpResources& HipMtpRuntime::resources() const noexcept {return *impl_->owner;}
std::uint64_t HipMtpRuntime::next_source_position() const noexcept {return impl_->next_source;}
bool HipMtpRuntime::failed() const noexcept {return impl_->broken || impl_->owner->failed();}
void HipMtpRuntime::cancel() noexcept {impl_->failure();}
std::uint64_t HipMtpRuntime::numerical_profile_identity() const noexcept {return impl_->profile;}
HipMtpExecutionStats HipMtpRuntime::execution_stats() const noexcept {return impl_->stats;}
const char* HipMtpRuntime::draft_cache_profile() const noexcept {
    const auto& p=*impl_;
    if(!p.draft || p.draft_exact)return "native-teacher-q8-v1";
    if(p.cache_profile==DraftCacheProfile::teacher)return "generation-teacher-q8-v1";
    const bool q4=hip_mtp_draft_q4(reinterpret_cast<const void*>(p.owner->tensors().key.runtime->address))!=nullptr;
    if(p.cache_profile==DraftCacheProfile::fast)
        return q4?"generation-fast-q4-v1":"generation-fast-q8-v1";
    return q4?"mixed-fast-q4-teacher-q8-v1":"mixed-fast-q8-teacher-q8-v1";
}
HipMtpCacheStateView HipMtpRuntime::cache_state() const {
    const auto& p=*impl_;
    require(!p.broken && !p.active && !p.cycle && !p.validation,
        "MTP durable cache view requires an idle completed producer outside any proposal");
    p.owner->require_idle();
    const auto kv_stride=mtp_native::product(mtp_native::product(p.owner->model().kv_heads,p.owner->model().head_dim),4U);
    const auto index_stride=mtp_native::product(p.owner->model().indexer_head_dim,2U);
    auto kv=p.owner->view(HipMtpBuffer::kv),index=p.owner->view(HipMtpBuffer::raw_index);
    const auto kv_bytes=mtp_native::product(p.next_source,kv_stride),index_bytes=mtp_native::product(p.next_source,index_stride);
    require(p.next_source<=p.owner->context_capacity() && kv_bytes<=kv.bytes && index_bytes<=index.bytes,
        "MTP durable cache prefix exceeds actual retained backing");
    kv.bytes=kv_bytes;index.bytes=index_bytes;
    return {kv,index,p.next_source,kv_stride,index_stride};
}
HipMtpNativeRowResult HipMtpRuntime::execute_diagnostic(const HipMtpTargetHiddenTicket& ticket,std::uint32_t token) {
    require(!impl_->cycle,"MTP diagnostic cannot interleave a live production proposal");return impl_->teacher(ticket,token);
}
void HipMtpRuntime::prime_committed_rows(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,std::span<const HipMtpReplayRow> rows) {
    auto& p=*impl_;p.qualification(plan,activated);require(!p.cycle && !p.active,"MTP priming cannot interleave a proposal");
    for(const auto& row:rows) row.hidden.admit_rotary_reader(p.owner->model().rotary_identity());
    {
    const auto admission=p.owner->admit_native_entry();
    for(std::size_t i=0;i<rows.size();++i) {
        p.validate_ticket(rows[i].hidden);require(rows[i].hidden.source_position()==p.next_source+i &&
            rows[i].next_input_token<p.owner->model().vocab_size,"MTP priming rows are invalid");
        if(i) {require_chain(rows[i-1U].hidden,rows[i].hidden);require(rows[i-1U].next_input_token==rows[i].hidden.input_token(),"MTP teacher priming token pair changed");}
        p.preflight(rows[i].hidden.source_position(),rows[i].next_input_token,true,true);
    }
    }
    if(p.draft) p.teacher_cache_rows(rows);
    else for(const auto& row:rows) p.teacher_cache(row.hidden,row.next_input_token);
}
void HipMtpRuntime::prime_hidden_rows(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,
    std::span<const MtpDeviceView> hidden,std::span<const std::uint32_t> next_tokens,
    const RotaryIdentity& producer_rotary) {
    if(prime_hidden_rows_submit(plan,activated,hidden,next_tokens,producer_rotary)) prime_hidden_rows_complete();
}
bool HipMtpRuntime::prime_hidden_rows_submit(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,
    std::span<const MtpDeviceView> hidden,std::span<const std::uint32_t> next_tokens,
    const RotaryIdentity& producer_rotary) {
    auto& p=*impl_;
    require_rotary_producer_identity(producer_rotary,p.owner->model().rotary_identity());
    p.qualification(plan,activated);
    require(!p.cycle && !p.active && !p.validation && !p.broken && hidden.size()==next_tokens.size() &&
        p.stats.cache_only_rows<=UINT64_MAX-hidden.size(),"MTP hidden-row priming requires an idle owner and one token per row");
    if(hidden.empty()) return false;
    const auto wide=p.owner->view(HipMtpBuffer::wide_current);
    p.validation.emplace(p.owner->admit_native_entry());
    try {
        for(std::size_t i=0;i<hidden.size();++i) {
            require(hidden[i].address && hidden[i].bytes==wide.bytes && next_tokens[i]<p.owner->model().vocab_size &&
                p.next_source+i+1U<p.owner->context_capacity(),"MTP hidden priming row is invalid");
            p.preflight(p.next_source+i,next_tokens[i],true,true);
        }
        p.owner->mark_submitted();
    } catch(...) {p.validation.reset();throw;}
    p.active=true;p.cache_only_active=true;
    try {
        void* stream=p.owner->stream();
        for(std::size_t i=0;i<hidden.size();++i) {
            const auto c=p.context(p.next_source+i);
            p.copy(wide,hidden[i]);
            launch_hip_mtp_embedding(c,next_tokens[i],stream);p.cache_row_front(c,stream);
            launch_hip_mtp_cache_rows(c,stream);
        }
    } catch(...) {p.failure();throw;}
    p.primed_rows=hidden.size();
    return true;
}
bool HipMtpRuntime::prime_hidden_rows_pending() const noexcept {return impl_->primed_rows!=0U;}
void HipMtpRuntime::prime_hidden_rows_complete() {
    auto& p=*impl_;
    require(p.primed_rows && p.active && p.cache_only_active && !p.cycle,"MTP hidden-row completion requires submitted priming rows");
    const auto rows=p.primed_rows;p.primed_rows=0;
    try {
        p.owner->complete();
        p.active=false;p.cache_only_active=false;p.next_source+=rows;p.stats.cache_only_rows+=rows;p.validation.reset();
        if(p.owner->qsa_pool()) p.owner->qsa_pool()->commit(true,p.next_source);
        p.last_target.reset();p.last_teacher_next_input.reset();p.first_cache_reusable=false;p.ticketless_frontier=p.next_source;
    } catch(...) {p.failure();throw;}
}
void HipMtpRuntime::restore_cache_frontier(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,std::uint64_t source) {
    auto& p=*impl_;p.qualification(plan,activated);
    require(!p.cycle && !p.active && !p.validation && !p.broken && p.next_source==0U && !p.last_target &&
        source<p.owner->context_capacity(),"MTP cache frontier restore requires a fresh idle owner within capacity");
    p.owner->require_idle();
    if(p.owner->qsa_pool()) {
        p.owner->qsa_pool()->invalidate(false,0);p.owner->qsa_pool()->invalidate(true,0);
        p.owner->qsa_pool()->commit(false,source+1);p.owner->qsa_pool()->commit(true,source);
    }
    p.next_source=source;p.ticketless_frontier=source;p.first_cache_reusable=false;
}
void HipMtpRuntime::begin_cycle(std::uint32_t token,const HipMtpTargetHiddenTicket& ticket) {
    auto& p=*impl_;p.validate_ticket(ticket);require(!p.cycle && !p.active && ticket.source_position()==p.next_source &&
        ticket.source_position()+1U==ticket.committed_session().next_position &&
        token==ticket.predicted_token(),"MTP cycle requires the actual pending target selection/frontier");
    if(p.last_target) {
        require_chain(*p.last_target,ticket);
        require(p.last_teacher_next_input && *p.last_teacher_next_input==ticket.input_token(),"MTP cycle does not continue its actual primed target input");
    }
    const auto admission=p.owner->admit_native_entry();
    p.preflight(p.next_source,token,true);p.owner->require_idle();
    // Metadata-only checkpoint: proposals overwrite rows starting at
    // base_source. All earlier rows are immutable, and every later reader is
    // bounded by its current source position after that row was overwritten.
    // The first proposal input always copies the authoritative cycle ticket.
    p.base_previous_target=p.last_target;p.base_previous_input=p.last_teacher_next_input;
    p.cycle_base=ticket;p.base_source=p.next_source;p.pending_token=token;p.candidates.clear();p.probabilities.clear();p.runner_ups.clear();p.cycle=true;p.first_cache_reusable=false;
    p.replay.clear();p.reconcile_final.reset();p.have_replay=false;
    if(p.draft) p.draft->observe(std::span<const std::uint32_t>(&token,1));
}
void HipMtpRuntime::enable_draft_chain(HipMtpDraftHeadConfig config,bool exact) {
    auto& p=*impl_;require(!p.draft && !p.cycle && !p.active && !p.broken,"MTP draft chain must be enabled once on an idle owner");
    const auto& m=p.owner->model();const auto& head=p.owner->tensors().lm_head;
    require(select_hip_mtp_lm_head_recipe(m,head)==HipMtpLmHeadRecipe::prefetch8_wave8_q8,
        "MTP draft chain requires the Q8_0 prefetch8 head recipe");
    config.hidden_size=static_cast<std::uint32_t>(m.hidden_size);config.vocab_size=static_cast<std::uint32_t>(m.vocab_size);
    config.max_depth=p.owner->budget().maximum_candidate_depth;
    p.draft_exact=exact;
    if(!exact) {
        // The fast MoE is specialized to the canonical sidecar; preflight keeps the exact tensor checks.
        require(m.hidden_size==2560U && m.moe_intermediate_size==640U && m.shared_expert_intermediate_size==640U &&
            m.num_experts==512U && m.num_experts_per_token==10U,"MTP fast draft MoE requires the canonical sidecar geometry");
        float* workspace=nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&workspace),hip_mtp_fast_moe_workspace_floats*sizeof(float)),"MTP fast draft workspace");
        p.fast_workspace.reset(workspace);
        require(m.hc_count==4U && m.hc_lowrank==320U,"MTP fast draft HC mix requires the canonical sidecar geometry");
        constexpr std::uint64_t inject=4U*10240U;float* hc=nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&hc),2U*inject*sizeof(float)),"MTP draft HC injection copies");
        p.hc_inject.reset(hc);
        launch_hip_mtp_dequantize_q8(reinterpret_cast<const void*>(p.owner->tensors().attention_hc_inject.runtime->address),inject,hc,p.owner->stream());
        launch_hip_mtp_dequantize_q8(reinterpret_cast<const void*>(p.owner->tensors().mlp_hc_inject.runtime->address),inject,hc+inject,p.owner->stream());
        // The draft's dense and expert projections read Q4_0 copies (proposals only; verification
        // is unchanged). With {2,4,8} windows the cheaper deep drafts win: 46.8 -> 47.9 tok/s.
        // STRIX_MTP_DRAFT_Q4_ALL=0 keeps the Q8 sidecar.
        const char* q4_all=std::getenv("STRIX_MTP_DRAFT_Q4_ALL");
        if(!q4_all || q4_all[0]!='0') {
            const auto& t=p.owner->tensors();
            for(const auto* v:{&t.eh_projection,&t.query,&t.key,&t.value,&t.attention_output,&t.expert_gate,&t.expert_up,
                    &t.expert_down,&t.shared_gate,&t.shared_up,&t.shared_down}) {
                require(v->manifest && v->runtime && v->manifest->ggml_type==8U && v->runtime->bytes%34U==0U,
                    "MTP draft Q4 copies require Q8_0 sidecar projections");
                if(m.rotary_identity().descriptor().mode()==RotaryMode::yarn)
                    register_hip_mtp_draft_q4(reinterpret_cast<const void*>(v->runtime->address),v->runtime->bytes/34U*32U,p.owner->stream(),
                        p.owner->shared_target().actual_sidecar_owner,v->runtime->allocation_generation,
                        p.owner->shared_target().sidecar_model_content_identity);
                else register_hip_mtp_draft_q4(reinterpret_cast<const void*>(v->runtime->address),v->runtime->bytes/34U*32U,p.owner->stream());
            }
        }
        check(hipStreamSynchronize(static_cast<hipStream_t>(p.owner->stream())),"MTP draft HC injection copies");
    }
    p.draft=std::make_unique<HipMtpDraftHead>(reinterpret_cast<const void*>(head.runtime->address),std::move(config),p.owner->stream());
}
const HipMtpDraftHead* HipMtpRuntime::draft_head() const noexcept {return impl_->draft.get();}
void HipMtpRuntime::observe_draft_context(std::span<const std::uint32_t> tokens) {
    auto& p=*impl_;require(p.draft && !p.active && !p.broken,"MTP draft context requires an enabled idle chain");p.draft->observe(tokens);
}
std::vector<std::uint32_t> HipMtpRuntime::propose_chain(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,
    const SerialSessionContext& target,const DraftProposalStateContext& proposal,std::uint32_t depth,float min_probability,
    const std::function<bool(std::span<const float>)>& keep_drafting) {
    auto& p=*impl_;p.qualification(plan,activated);
    validate_mtp_draft_proposal_phase(p.owner->binding(),proposal,target,DraftProposalStatePhase::proposal_active);
    require(p.cycle && p.cycle_base && same_session(target,p.cycle_base->committed_session()),"MTP draft chain does not match its native cycle");
    return p.chain(depth,min_probability,keep_drafting);
}
bool HipMtpRuntime::propose_chain_begin(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,
    const SerialSessionContext& target,const DraftProposalStateContext& proposal,std::uint32_t depth,float min_probability,
    const std::function<bool(std::span<const float>)>& keep_drafting) {
    auto& p=*impl_;p.qualification(plan,activated);
    validate_mtp_draft_proposal_phase(p.owner->binding(),proposal,target,DraftProposalStatePhase::proposal_active);
    require(p.cycle && p.cycle_base && same_session(target,p.cycle_base->committed_session()),"MTP draft chain does not match its native cycle");
    return p.chain_begin(depth,min_probability,keep_drafting);
}
namespace {
bool draft_batch_env_branches() {
    return hip_mtp_hyper_mix_draft_rows_supported() && hip_mtp_attention_draft_rows_supported() && hip_mtp_route_rows_supported();
}
// Every owner holds its own copy of the model contract: the fields the draft step depends on must be equal.
bool same_draft_model(const ModelContract& a,const ModelContract& b) {
    return same_rotary_model(a,b) && a.vocab_size==b.vocab_size && a.hidden_size==b.hidden_size && a.max_position_embeddings==b.max_position_embeddings &&
        a.hc_count==b.hc_count && a.hc_lowrank==b.hc_lowrank && a.attention_heads==b.attention_heads && a.kv_heads==b.kv_heads &&
        a.head_dim==b.head_dim && a.indexer_n_heads==b.indexer_n_heads && a.indexer_kv_heads==b.indexer_kv_heads &&
        a.indexer_head_dim==b.indexer_head_dim && a.indexer_budget==b.indexer_budget && a.indexer_compress_ratio==b.indexer_compress_ratio &&
        a.partial_rotary_factor==b.partial_rotary_factor && a.rope_theta==b.rope_theta && a.moe_intermediate_size==b.moe_intermediate_size &&
        a.shared_expert_intermediate_size==b.shared_expert_intermediate_size && a.num_experts==b.num_experts &&
        a.num_experts_per_token==b.num_experts_per_token && a.rms_norm_epsilon==b.rms_norm_epsilon &&
        a.norm_topk_probability==b.norm_topk_probability && a.attention_kv_storage==b.attention_kv_storage;
}
}
bool HipMtpRuntime::draft_batch_open() const noexcept {
    return draft_batch_decline()==MtpDraftBatchDecline::none;
}
MtpDraftBatchDecline HipMtpRuntime::draft_batch_decline() const noexcept {
    const auto& p=*impl_;
    using D=MtpDraftBatchDecline;
    if(!p.active || !p.draft) return D::inactive;
    if(p.draft_exact) return D::exact;
    if(p.sampler) return D::sampled;
    if(p.cs.device_draft) return D::device_draft;
    // A keep callback (cost-model chain cut, ordinary cycles only) is never batched.
    if(p.cs.keep) return D::keep;
    if(p.cs.stop) return D::stopped;
    if(p.cs.k>=p.cs.requested) return D::depth;
    if(p.batch_stream) return D::busy;
    if(p.broken) return D::broken;
    return D::none;
}
std::optional<std::uint32_t> HipMtpRuntime::draft_batch_compatible(std::span<HipMtpRuntime* const> members,MtpDraftBatchDecline* why) {
    // The reason every member of a declined set records (shared-execution witness).
    const auto decline=[why](MtpDraftBatchDecline d)->std::optional<std::uint32_t> {if(why) *why=d;return std::nullopt;};
    if(members.size()<2U) return decline(MtpDraftBatchDecline::alone);
    if(members.size()>hip_mtp_draft_batch_rows) return decline(MtpDraftBatchDecline::capacity);
    if(!draft_batch_env_branches()) return decline(MtpDraftBatchDecline::unsupported_branch);
    const auto& a=*members.front()->impl_;
    const auto& ta=a.owner->tensors();
    std::vector<HipMtpDraftHead*> heads;
    for(auto* m:members) {
        if(!m) return decline(MtpDraftBatchDecline::inactive);
        if(const auto d=m->draft_batch_decline();d!=MtpDraftBatchDecline::none) return decline(d);
        const auto& p=*m->impl_;const auto& t=p.owner->tensors();
        for(std::size_t o=0;o<heads.size();++o) if(heads[o]==p.draft.get()) return decline(MtpDraftBatchDecline::incompatible_owner);
        // One model, plan, profile and weights (the shared sidecar residency); every buffer stays per owner.
        if(!same_draft_model(p.owner->model(),a.owner->model()) || p.owner->plan().plan_identity!=a.owner->plan().plan_identity ||
           p.profile!=a.profile || p.cs.k!=a.cs.k ||
           t.embedding.runtime->address!=ta.embedding.runtime->address || t.eh_projection.runtime->address!=ta.eh_projection.runtime->address ||
           t.attention_hc_inject.runtime->address!=ta.attention_hc_inject.runtime->address ||
           t.mlp_hc_inject.runtime->address!=ta.mlp_hc_inject.runtime->address || t.router.runtime->address!=ta.router.runtime->address ||
           t.expert_gate.runtime->address!=ta.expert_gate.runtime->address || t.shared_gate.runtime->address!=ta.shared_gate.runtime->address ||
           !p.hc_inject || !p.fast_workspace)
            return decline(MtpDraftBatchDecline::incompatible_owner);
        // The batched launchers' alignment requirements (staged router, HC rows workspaces, 16-byte state copy):
        // a member that the one-token chain would run on another branch stays out of the batch.
        if(p.owner->view(HipMtpBuffer::hidden0).address%16U || t.router.runtime->address%16U ||
           p.owner->view(HipMtpBuffer::hyper_workspace).address%16U || p.owner->view(HipMtpBuffer::wide_current).address%16U ||
           p.owner->view(HipMtpBuffer::wide_next).address%16U || reinterpret_cast<std::uintptr_t>(p.hc_inject.get())%16U)
            return decline(MtpDraftBatchDecline::unsupported_branch);
        // The batched launchers run only the Q4-copy branch: every draft weight they read must have its copy, and
        // every member reads the same tensors.
        for(const auto* v:{&t.eh_projection,&t.query,&t.key,&t.value,&t.attention_output,&t.expert_gate,&t.expert_up,&t.expert_down,
                           &t.shared_gate,&t.shared_up,&t.shared_down})
            if(!hip_mtp_draft_q4(reinterpret_cast<const void*>(v->runtime->address))) return decline(MtpDraftBatchDecline::unsupported_branch);
        if(t.query.runtime->address!=ta.query.runtime->address || t.key.runtime->address!=ta.key.runtime->address ||
           t.value.runtime->address!=ta.value.runtime->address || t.attention_output.runtime->address!=ta.attention_output.runtime->address ||
           t.index_query.runtime->address!=ta.index_query.runtime->address || t.index_key.runtime->address!=ta.index_key.runtime->address ||
           t.expert_up.runtime->address!=ta.expert_up.runtime->address || t.expert_down.runtime->address!=ta.expert_down.runtime->address ||
           t.shared_up.runtime->address!=ta.shared_up.runtime->address || t.shared_down.runtime->address!=ta.shared_down.runtime->address ||
           t.shared_scalar_gate.runtime->address!=ta.shared_scalar_gate.runtime->address ||
           t.attention_hc_down.runtime->address!=ta.attention_hc_down.runtime->address ||
           t.attention_hc_up.runtime->address!=ta.attention_hc_up.runtime->address ||
           t.mlp_hc_down.runtime->address!=ta.mlp_hc_down.runtime->address || t.mlp_hc_up.runtime->address!=ta.mlp_hc_up.runtime->address ||
           t.head_down.runtime->address!=ta.head_down.runtime->address || t.head_up.runtime->address!=ta.head_up.runtime->address ||
           t.enorm.runtime->address!=ta.enorm.runtime->address || t.hnorm.runtime->address!=ta.hnorm.runtime->address ||
           t.attention_hc_norm.runtime->address!=ta.attention_hc_norm.runtime->address ||
           t.mlp_hc_norm.runtime->address!=ta.mlp_hc_norm.runtime->address || t.head_norm.runtime->address!=ta.head_norm.runtime->address ||
           t.query_norm.runtime->address!=ta.query_norm.runtime->address || t.key_norm.runtime->address!=ta.key_norm.runtime->address ||
           t.index_query_norm.runtime->address!=ta.index_query_norm.runtime->address ||
           t.index_key_norm.runtime->address!=ta.index_key_norm.runtime->address)
            return decline(MtpDraftBatchDecline::incompatible_owner);
        heads.push_back(p.draft.get());
    }
    const auto shared=HipMtpDraftHead::shared_rows(heads);
    if(shared==UINT32_MAX) return decline(MtpDraftBatchDecline::incompatible_owner);
    return shared;
}
void HipMtpRuntime::draft_batch_enter(std::span<HipMtpRuntime* const> members,void* stream) {
    require(stream && !members.empty(),"MTP draft batch needs a stream");
    for(auto* m:members) {
        auto& p=*m->impl_;
        require(m->draft_batch_open(),"MTP draft batch member has no open fast greedy chain");
        try {
            if(!p.batch_entry) check(hipEventCreateWithFlags(&p.batch_entry,hipEventDisableTiming),"MTP draft batch entry event");
            // The cycle-base copy, the head's row gathers and any earlier owner work precede the batch steps.
            check(hipEventRecord(p.batch_entry,static_cast<hipStream_t>(p.owner->stream())),"MTP draft batch entry record");
            check(hipStreamWaitEvent(static_cast<hipStream_t>(stream),p.batch_entry,0),"MTP draft batch entry wait");
            p.batch_stream=static_cast<hipStream_t>(stream);
        } catch(...) {p.failure();throw;}
    }
}
void HipMtpRuntime::draft_batch_leave() {
    auto& p=*impl_;
    try {p.batch_leave();} catch(...) {p.failure();throw;}
}
namespace {
// Per-owner buffers a draft step writes before it reads them (poisoned between the reference and the batch, and
// captured after every step together with the attention workspace and the status words).
constexpr std::array<HipMtpBuffer,14> batch_step_buffers{HipMtpBuffer::embedding,HipMtpBuffer::normalized_input,
    HipMtpBuffer::fused_wide,HipMtpBuffer::wide_work,HipMtpBuffer::hidden0,HipMtpBuffer::hidden1,HipMtpBuffer::injection,
    HipMtpBuffer::routes,HipMtpBuffer::route_weights,HipMtpBuffer::hyper_workspace,HipMtpBuffer::router_workspace,
    HipMtpBuffer::attention_workspace,HipMtpBuffer::wide_next,HipMtpBuffer::wide_current};
std::vector<std::uint8_t> read_device(std::uintptr_t address,std::uint64_t bytes) {
    std::vector<std::uint8_t> v(static_cast<std::size_t>(bytes));
    if(bytes) check(hipMemcpy(v.data(),reinterpret_cast<const void*>(address),static_cast<std::size_t>(bytes),hipMemcpyDeviceToHost),
        "MTP draft batch verify readback");
    return v;
}
// One step's state of this owner (the caller completed the stream that carried it).
std::vector<std::vector<std::uint8_t>> draft_batch_step_state(const HipMtpResources& owner,const HipMtpDraftHead& head,const float* fast) {
    std::vector<std::vector<std::uint8_t>> out;
    for(const auto b:batch_step_buffers) {const auto v=owner.view(b);out.push_back(read_device(v.address,v.bytes));}
    {const auto v=owner.view(HipMtpBuffer::status);out.push_back(read_device(v.address,v.bytes));}
    out.push_back(read_device(reinterpret_cast<std::uintptr_t>(fast),hip_mtp_fast_moe_workspace_floats*sizeof(float)));
    out.push_back(read_device(reinterpret_cast<std::uintptr_t>(head.logits()),static_cast<std::uint64_t>(head.stats().rows)*sizeof(float)));
    out.push_back(read_device(reinterpret_cast<std::uintptr_t>(head.device_token(0)),static_cast<std::uint64_t>(owner.budget().maximum_candidate_depth)*4U));
    return out;
}
std::string draft_batch_state_difference(const std::vector<std::vector<std::uint8_t>>& a,const std::vector<std::vector<std::uint8_t>>& b) {
    static constexpr const char* names[]={"embedding","normalized","fused_wide","wide_work","hidden0","hidden1","injection","routes",
        "route_weights","hyper_ws","router_ws","attention_ws","wide_next","wide_current","status","moe_ws","head_logits","head_slots"};
    std::string what;
    for(std::size_t i=0;i<a.size() && i<b.size();++i) if(a[i]!=b[i]) {what+=" ";what+=i<std::size(names)?names[i]:"?";}
    if(a.size()!=b.size()) what+=" shape";
    return what;
}
}
namespace {
std::uint64_t fnv(const std::vector<std::uint8_t>& v) {
    std::uint64_t h=1469598103934665603ULL;for(const auto b:v) {h^=b;h*=1099511628211ULL;}return h;
}
// Verify coverage counters (printed with the verify summary).
struct VerifyCoverage {std::uint64_t members=0,steps=0,partial_groups=0,selection_rows=0,full_rows=0,selection_transitions=0,
    shrinking_steps=0,divergent_cycles=0,min_p_below=0,min_p_equal=0,min_p_above=0,after_rejection=0;};
VerifyCoverage& verify_coverage() {static VerifyCoverage c;return c;}
}
std::string draft_batch_verify_coverage() {
    const auto& c=verify_coverage();
    return "members="+std::to_string(c.members)+" steps="+std::to_string(c.steps)+" partial_groups="+std::to_string(c.partial_groups)+
        " selection_rows="+std::to_string(c.selection_rows)+" full_rows="+std::to_string(c.full_rows)+
        " selection_transitions="+std::to_string(c.selection_transitions)+" shrinking_steps="+std::to_string(c.shrinking_steps)+
        " divergent_requested="+std::to_string(c.divergent_cycles)+" min_p_below="+std::to_string(c.min_p_below)+
        " min_p_equal="+std::to_string(c.min_p_equal)+" min_p_above="+std::to_string(c.min_p_above)+
        " after_rejection="+std::to_string(c.after_rejection);
}
// Scratch both executions start from: the step buffers (except the wide input), the fast-MoE workspace, the head
// logits and device slots, the proposal cache rows the chain may write and a guard window past them, all 0xFF.
void HipMtpRuntime::Impl::batch_verify_poison(std::uint64_t first,std::uint64_t rows,std::uint64_t guard) {
    if(owner->qsa_pool()) owner->qsa_pool()->invalidate(true,first);
    const auto stream=static_cast<hipStream_t>(owner->stream());
    const auto capacity=owner->context_capacity();
    const auto kv=owner->view(HipMtpBuffer::kv);const auto ix=owner->view(HipMtpBuffer::raw_index);
    const auto kv_row=kv.bytes/capacity,ix_row=ix.bytes/capacity;
    const auto poison=[&](std::uintptr_t address,std::uint64_t bytes) {
        if(bytes) check(hipMemsetAsync(reinterpret_cast<void*>(address),0xff,static_cast<std::size_t>(bytes),stream),"MTP draft batch verify poison");
    };
    for(const auto b:batch_step_buffers) if(b!=HipMtpBuffer::wide_current) {const auto v=owner->view(b);poison(v.address,v.bytes);}
    poison(reinterpret_cast<std::uintptr_t>(fast_workspace.get()),hip_mtp_fast_moe_workspace_floats*sizeof(float));
    poison(reinterpret_cast<std::uintptr_t>(draft->logits()),static_cast<std::uint64_t>(draft->stats().rows)*sizeof(float));
    poison(reinterpret_cast<std::uintptr_t>(draft->device_token(0)),static_cast<std::uint64_t>(owner->budget().maximum_candidate_depth)*4U);
    poison(kv.address+first*kv_row,(rows+guard)*kv_row);poison(ix.address+first*ix_row,(rows+guard)*ix_row);
    copy(owner->view(HipMtpBuffer::wide_current),cycle_base->view());
    draft->begin_chain();
    check(hipStreamSynchronize(stream),"MTP draft batch verify poison");
}
void HipMtpRuntime::draft_batch_verify_cap(std::uint32_t requested) {
    auto& p=*impl_;
    require(draft_batch_open() && p.cs.k==0U && requested,"MTP draft batch depth cap needs a freshly begun chain");
    if(requested<p.cs.requested) p.cs.requested=p.cs.depth=requested;
}
void HipMtpRuntime::draft_batch_verify_reference(std::span<HipMtpRuntime* const> members) {
    auto& coverage=verify_coverage();
    for(auto* m:members) {
        auto& p=*m->impl_;
        require(m->draft_batch_open() && p.cs.k==0U && p.cycle_base,"MTP draft batch verify needs a freshly begun chain");
        try {
            const auto saved=p.cs;
            const auto stream=static_cast<hipStream_t>(p.owner->stream());
            const auto capacity=p.owner->context_capacity();
            const auto kv=p.owner->view(HipMtpBuffer::kv);const auto ix=p.owner->view(HipMtpBuffer::raw_index);
            require(kv.bytes%capacity==0U && ix.bytes%capacity==0U,"MTP draft batch verify cache rows are not uniform");
            const auto kv_row=kv.bytes/capacity,ix_row=ix.bytes/capacity;
            Impl::BatchReference ref;ref.first=p.next_source;
            const auto rows=std::min<std::uint64_t>(saved.requested,capacity-ref.first);
            ref.guard_rows=std::min<std::uint64_t>(8U,capacity-ref.first-rows);
            // The committed prefix must be identical before the reference and after the batch.
            ref.prefix_hash=fnv(read_device(kv.address,ref.first*kv_row))^(fnv(read_device(ix.address,ref.first*ix_row))*31U);
            p.batch_verify_poison(ref.first,rows,ref.guard_rows);
            p.stage_hook=[&](unsigned stage) {
                check(hipStreamSynchronize(stream),"MTP draft batch verify reference stage");
                if(stage==0U) ref.stages.emplace_back();
                ref.stages.back().push_back(draft_batch_step_state(*p.owner,*p.draft,p.fast_workspace.get()));
            };
            try {while(p.chain_launch_step() && p.chain_after_step()) {}} catch(...) {p.stage_hook=nullptr;throw;}
            p.stage_hook=nullptr;
            check(hipStreamSynchronize(stream),"MTP draft batch verify reference");
            ref.depth=p.cs.depth;ref.steps=p.cs.k;
            ref.tokens=p.draft->read(ref.steps);
            for(std::uint32_t k=0;k<ref.steps;++k) {
                ref.probabilities.push_back(p.draft->probability(k));ref.runner_ups.push_back(p.draft->runner_up_probability(k));
                // Only steps followed by a stop decision (not the last requested one).
                if(saved.min_probability>0.0F && k+1U<saved.requested) {
                    if(ref.probabilities[k]<saved.min_probability) ++coverage.min_p_below;
                    else if(ref.probabilities[k]==saved.min_probability) ++coverage.min_p_equal;
                    else ++coverage.min_p_above;
                }
                const auto visible=ref.first+k+1U;const auto& mm=p.owner->model();
                const auto blocks=visible/mm.indexer_compress_ratio,topk=mm.indexer_budget/mm.indexer_compress_ratio;
                const auto selected=std::min(blocks,topk)*mm.indexer_compress_ratio+visible%mm.indexer_compress_ratio;
                if(selected%4U) ++coverage.partial_groups;
                if(blocks>topk) ++coverage.selection_rows; else ++coverage.full_rows;
                if(k && (blocks>topk)!=((visible-1U)/mm.indexer_compress_ratio>topk)) ++coverage.selection_transitions;
            }
            // The previous verified cycle of this owner proposed more rows than the target committed (a rejection),
            // and this cycle drafts after the teacher overwrite of the rejected rows.
            if(p.verify_previous_first && ref.first-p.verify_previous_first<p.verify_previous_steps+1U) ++coverage.after_rejection;
            ref.kv=read_device(kv.address+ref.first*kv_row,ref.steps*kv_row);
            ref.index=read_device(ix.address+ref.first*ix_row,ref.steps*ix_row);
            p.cs=saved;
            p.batch_verify_poison(ref.first,rows,ref.guard_rows);
            p.verify_previous_first=ref.first;p.verify_previous_steps=ref.steps;
            p.batch_reference=std::move(ref);
            ++coverage.members;
        } catch(...) {p.failure();throw;}
    }
}
void HipMtpRuntime::draft_batch_verify_stage(std::span<HipMtpRuntime* const> rows,void* stream,unsigned stage) {
    check(hipStreamSynchronize(static_cast<hipStream_t>(stream)),"MTP draft batch verify stage");
    for(auto* m:rows) {
        auto& p=*m->impl_;
        if(!p.batch_reference) continue;
        auto& ref=*p.batch_reference;
        const auto k=p.cs.k;
        if(k>=ref.stages.size() || stage>=ref.stages[k].size()) {ref.mismatch+=" extra-step"+std::to_string(k);continue;}
        const auto what=draft_batch_state_difference(draft_batch_step_state(*p.owner,*p.draft,p.fast_workspace.get()),ref.stages[k][stage]);
        if(!what.empty()) ref.mismatch+=" step"+std::to_string(k)+"/stage"+std::to_string(stage)+":"+what;
    }
}
void HipMtpRuntime::draft_batch_verify_step(std::span<HipMtpRuntime* const> members,std::span<HipMtpRuntime* const> rows,void* stream) {
    // Members that already stopped must keep their final state while the others step on.
    check(hipStreamSynchronize(static_cast<hipStream_t>(stream)),"MTP draft batch verify step");
    auto& coverage=verify_coverage();++coverage.steps;
    if(rows.size()<members.size()) ++coverage.shrinking_steps;
    for(auto* m:members) {
        if(std::find(rows.begin(),rows.end(),m)!=rows.end()) continue;
        auto& p=*m->impl_;
        if(!p.batch_reference || p.batch_reference->stages.empty()) continue;
        auto& ref=*p.batch_reference;
        const auto what=draft_batch_state_difference(draft_batch_step_state(*p.owner,*p.draft,p.fast_workspace.get()),ref.stages.back().back());
        if(!what.empty()) ref.mismatch+=" stopped-row-changed:"+what;
    }
}
std::uint32_t HipMtpRuntime::draft_batch_verify_check(std::span<HipMtpRuntime* const> members,void* stream) {
    check(hipStreamSynchronize(static_cast<hipStream_t>(stream)),"MTP draft batch verify completion");
    std::uint32_t differing=0;std::uint32_t lo=UINT32_MAX,hi=0;
    for(auto* m:members) {
        auto& p=*m->impl_;
        std::string what;
        if(!p.batch_reference) what=" no-reference";
        else {
            const auto ref=std::move(*p.batch_reference);p.batch_reference.reset();
            what=ref.mismatch;lo=std::min(lo,ref.steps);hi=std::max(hi,ref.steps);
            const auto capacity=p.owner->context_capacity();
            const auto kv=p.owner->view(HipMtpBuffer::kv);const auto ix=p.owner->view(HipMtpBuffer::raw_index);
            const auto kv_row=kv.bytes/capacity,ix_row=ix.bytes/capacity;
            if(p.cs.depth!=ref.depth || p.cs.k!=ref.steps) what+=" depth/steps";
            else {
                if(p.draft->read(ref.steps)!=ref.tokens) what+=" tokens";
                for(std::uint32_t k=0;k<ref.steps;++k) {
                    if(std::bit_cast<std::uint32_t>(p.draft->probability(k))!=std::bit_cast<std::uint32_t>(ref.probabilities[k])) {what+=" probability";break;}
                    if(std::bit_cast<std::uint32_t>(p.draft->runner_up_probability(k))!=std::bit_cast<std::uint32_t>(ref.runner_ups[k])) {what+=" runner-up";break;}
                }
                if(read_device(kv.address+ref.first*kv_row,ref.steps*kv_row)!=ref.kv) what+=" kv-rows";
                if(read_device(ix.address+ref.first*ix_row,ref.steps*ix_row)!=ref.index) what+=" index-rows";
                if((fnv(read_device(kv.address,ref.first*kv_row))^(fnv(read_device(ix.address,ref.first*ix_row))*31U))!=ref.prefix_hash)
                    what+=" committed-prefix";
                // Rows past the chain's last step, and the guard window past the requested ones, must hold the poison.
                const auto requested=std::min<std::uint64_t>(p.cs.requested,capacity-ref.first);
                const auto tail=requested-ref.steps+ref.guard_rows;
                if(tail) {
                    const auto tail_kv=read_device(kv.address+(ref.first+ref.steps)*kv_row,tail*kv_row);
                    const auto tail_ix=read_device(ix.address+(ref.first+ref.steps)*ix_row,tail*ix_row);
                    if(std::any_of(tail_kv.begin(),tail_kv.end(),[](std::uint8_t x){return x!=0xffU;}) ||
                       std::any_of(tail_ix.begin(),tail_ix.end(),[](std::uint8_t x){return x!=0xffU;})) what+=" write-outside-proposal";
                }
            }
        }
        if(!what.empty()) {
            ++differing;
            std::fprintf(stderr,"draft batch verify MISMATCH at=%llu steps=%u rows=%zu:%s\n",static_cast<unsigned long long>(p.next_source),
                p.cs.k,members.size(),what.c_str());
        }
    }
    {
        std::uint32_t rlo=UINT32_MAX,rhi=0;
        for(auto* m:members) {rlo=std::min(rlo,m->impl_->cs.requested);rhi=std::max(rhi,m->impl_->cs.requested);}
        if(rlo!=rhi) ++verify_coverage().divergent_cycles;
    }
    (void)lo;(void)hi;
    return differing;
}
void HipMtpRuntime::draft_batch_launch(std::span<HipMtpRuntime* const> rows,void* stream,std::uint32_t shared_head_rows) {
    require(!rows.empty() && rows.size()<=hip_mtp_draft_batch_rows && stream,"MTP draft batch launch is invalid");
    try {
        const auto n=rows.size();
        std::vector<MtpNativeContext> contexts;contexts.reserve(n);
        std::vector<HipMtpInputFusionRequest> fusion;fusion.reserve(n);
        std::vector<HipMtpFastMoeViews> moe;moe.reserve(n);
        std::vector<const std::uint32_t*> tokens(n,nullptr);std::vector<std::uint32_t> values(n,0U);
        std::vector<HipMtpDraftHead*> heads(n,nullptr);std::vector<const std::uint16_t*> hidden(n,nullptr);
        const auto k=rows.front()->impl_->cs.k;
        for(std::size_t r=0;r<n;++r) {
            auto& p=*rows[r]->impl_;
            require(p.active && p.batch_stream==static_cast<hipStream_t>(stream) && p.draft && !p.draft_exact && !p.sampler &&
                !p.cs.device_draft && !p.cs.stop && p.cs.k<p.cs.requested && p.cs.k==k,"MTP draft batch row is not an entered open chain at this step");
            contexts.push_back(p.context(p.next_source+k));
            fusion.push_back(p.fusion());moe.push_back(p.fast_moe());
            if(k) tokens[r]=p.draft->device_token(k-1U); else values[r]=p.pending_token;
            heads[r]=p.draft.get();hidden[r]=p.cs.hidden;
        }
        const auto& lead=*rows.front()->impl_;
        constexpr std::uint64_t inject=4U*10240U;
        const auto* hc=lead.hc_inject.get();
        // Step k of every row, in the one-token chain's order (chain_launch_step, fast greedy branch).
        // STRIX_MTP_DRAFT_BATCH_VERIFY: each stage's per-row state against the member's own chain at that stage.
        const bool verify=rows.front()->impl_->batch_reference.has_value();
        const auto stage=[&](unsigned s) {if(verify) draft_batch_verify_stage(rows,stream,s);};
        launch_hip_mtp_embedding_rows(contexts,tokens,values,stream);
        launch_hip_mtp_input_fusion_draft_rows(fusion,stream);stage(0);
        launch_hip_mtp_hyper_mix_draft_rows(contexts,MtpHyperRole::attention,hc,stream);stage(1);
        launch_hip_mtp_indexer_rows(contexts,stream);stage(2);
        launch_hip_mtp_sparse_attention_draft_rows(contexts,stream);stage(3);
        launch_hip_mtp_inject_rows(contexts,MtpInjectRole::attention,stream);stage(4);
        launch_hip_mtp_hyper_mix_draft_rows(contexts,MtpHyperRole::mlp,hc+inject,stream);stage(5);
        launch_hip_mtp_route_rows(contexts,stream);stage(6);
        launch_hip_mtp_fast_moe_rows(moe,stream);stage(7);
        launch_hip_mtp_inject_rows(contexts,MtpInjectRole::mlp,stream);stage(8);
        launch_hip_mtp_publish_next_rows(contexts,stream);stage(9);
        launch_hip_mtp_hyper_mix_draft_rows(contexts,MtpHyperRole::head,nullptr,stream);stage(10);
        HipMtpDraftHead::launch_rows(heads,hidden,k,shared_head_rows,stream);stage(11);
        host_phase_mark("2l draft batch step launch");
    } catch(...) {
        for(auto* m:rows) m->impl_->failure();
        throw;
    }
}
bool HipMtpRuntime::propose_chain_launch_step() {return impl_->chain_launch_step();}
bool HipMtpRuntime::propose_chain_after_step() {return impl_->chain_after_step();}
std::vector<std::uint32_t> HipMtpRuntime::propose_chain_finish() {return impl_->chain_finish();}
std::span<const float> HipMtpRuntime::draft_probabilities() const noexcept {return impl_->probabilities;}
void HipMtpRuntime::set_draft_sampler(std::shared_ptr<TokenSampler> sampler) {
    auto& p=*impl_;require(p.draft && sampler && !p.sampler && !p.cycle && !p.active && !p.broken,
        "MTP sampled drafts need an enabled idle draft chain, once before cycles");
    require(p.draft_exact || p.cache_profile==DraftCacheProfile::mixed,
        "uniform generation cache profiles currently require greedy drafts");
    p.sampler=std::move(sampler);
}
std::span<const TokenDraft> HipMtpRuntime::sampled_drafts() const noexcept {return impl_->sampled;}
std::span<const float> HipMtpRuntime::draft_runner_up_probabilities() const noexcept {return impl_->runner_ups;}
void HipMtpRuntime::dispatch_mtp_step(const MtpForwardStepContext& call) {
    auto& p=*impl_;
    try {
        require(call.cycle_input && call.binding && call.state_memory && call.manifest && call.runtime,
            "MTP dispatch lacks its typed physical source bindings");
        validate_mtp_draft_runtime_binding(*call.manifest,*call.runtime,*call.binding);
        validate_mtp_draft_state_memory_binding(plan_mtp_draft_state_memory(p.owner->plan().proposal_identity,p.owner->plan().state),*call.state_memory);
        require(call.binding->physical_binding_identity==p.owner->binding().physical_binding_identity &&
            call.binding->compatibility_identity==p.owner->binding().compatibility_identity &&
            call.state_memory->device_base==p.owner->state_memory().device_base &&
            call.state_memory->allocation_generation==p.owner->state_memory().allocation_generation &&
            call.state_memory->plan_identity==p.owner->state_memory().plan_identity,
            "MTP dispatch physical source differs from its native owner");
        p.stage(call);
    } catch(...) {if(p.active) p.failure();throw;}
}
MtpDraftCycleInput HipMtpRuntime::make_cycle_input(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,
    const MtpDraftRuntimeBinding& binding,const MtpDraftStateMemoryBinding& state,const SerialSessionContext& target,
    const DraftProposalStateContext& proposal,std::span<const std::uint32_t> prior) {
    auto& p=*impl_;p.qualification(plan,activated);
    validate_mtp_draft_proposal_phase(binding,proposal,target,DraftProposalStatePhase::proposal_active);
    validate_mtp_draft_runtime_binding(p.owner->manifest(),p.owner->runtime_binding(),binding);
    validate_mtp_draft_state_memory_binding(plan_mtp_draft_state_memory(plan.base.proposal_identity,plan.base.state),state);
    require(p.cycle && p.cycle_base && same_session(target,p.cycle_base->committed_session()) &&
        prior.size()==p.candidates.size() && prior.size()<p.owner->budget().maximum_candidate_depth &&
        std::equal(prior.begin(),prior.end(),p.candidates.begin()) && state.device_base==p.owner->state_memory().device_base &&
        state.allocation_generation==p.owner->state_memory().allocation_generation,"MTP cycle history/state does not match its native owner");
    const auto v=p.owner->view(HipMtpBuffer::wide_current);
    MtpDraftCycleInput input{plan.base.proposal_identity,activated.activation_identity,target.runtime_compatibility_identity,
        prior.empty()?p.pending_token:prior.back(),p.next_source+1U,p.next_source,v.address,plan.base.state.state_width,2U,v.allocation_generation};
    p.start(input,prior.empty()?&*p.cycle_base:nullptr);return input;
}
std::uint32_t HipMtpRuntime::read_greedy_token(const QualifiedMtpDraftPlan& plan,const MtpDraftCycleInput& input,
    const MtpForwardExecutionResult& result,std::uint32_t index) {
    auto& p=*impl_;
    try {
        require_qualified_mtp_draft_plan(plan);
        require(plan.base.plan_identity==p.owner->plan().plan_identity && p.cycle && index==p.candidates.size() && same_input(input,p.current) &&
            plan.native_arithmetic_profile_identity==p.profile &&
            result.completed_steps==native_steps.size() && result.next_state_published && result.logits_published,
            "MTP token read does not match a complete pending native forward");
        const auto token=p.finish();
        if(index==0U) p.first_cache_reusable=p.cycle_base && p.current.hidden_source_position==p.base_source &&
            p.current.seed_position==p.base_source+1U && p.current.seed_token_id==p.pending_token;
        p.candidates.push_back(token);return token;
    } catch(...) {if(p.active) p.failure();throw;}
}
void HipMtpRuntime::set_reconciliation_rows(std::span<const HipMtpReplayRow> rows,const HipMtpTargetHiddenTicket& final_target) {
    auto& p=*impl_;require(p.cycle && !p.active && p.cycle_base && rows.size()<=p.owner->budget().maximum_candidate_depth+1U,
        "MTP reconciliation requires its bounded completed proposal");p.validate_ticket(final_target);
    require(final_target.same_producer(*p.cycle_base) &&
        final_target.source_position()+1U==final_target.committed_session().next_position &&
        final_target.source_position()>=p.base_source &&
        final_target.source_position()-p.base_source==rows.size(),"MTP reconciliation target rows do not cover the checkpoint frontier");
    for(std::size_t i=0;i<rows.size();++i) {
        p.validate_ticket(rows[i].hidden);require(rows[i].hidden.source_position()==p.base_source+i &&
            rows[i].hidden.same_producer(*p.cycle_base),"MTP reconciliation source row changed");
        const auto& next=i+1U<rows.size()?rows[i+1U].hidden:final_target;
        require_chain(rows[i].hidden,next);require(rows[i].next_input_token==next.input_token(),"MTP reconciliation is not actual teacher-forced target input");
        if(!i) require(same_session(rows[i].hidden.committed_session(),p.cycle_base->committed_session()),"MTP reconciliation checkpoint history changed");
    }
    if(rows.empty()) require(same_session(final_target.committed_session(),p.cycle_base->committed_session()),"MTP empty reconciliation changed target history");
    p.replay.assign(rows.begin(),rows.end());p.reconcile_final=final_target;p.have_replay=true;
}
void HipMtpRuntime::reconcile_after_target(const QualifiedMtpDraftPlan& plan,const ActivatedMtpDraftPlan& activated,
    const MtpDraftRuntimeBinding& binding,const MtpDraftStateMemoryBinding& state,const DraftProposalStateContext& proposal,
    const SerialSessionContext& target,std::span<const std::uint32_t> emitted) {
    auto& p=*impl_;p.qualification(plan,activated);
    validate_mtp_draft_proposal_phase(binding,proposal,target,DraftProposalStatePhase::reconciliation_required);
    validate_mtp_draft_runtime_binding(p.owner->manifest(),p.owner->runtime_binding(),binding);
    validate_mtp_draft_state_memory_binding(plan_mtp_draft_state_memory(plan.base.proposal_identity,plan.base.state),state);
    require(p.cycle && !p.active && p.have_replay && p.reconcile_final && same_session(target,p.reconcile_final->committed_session()) &&
        state.device_base==p.owner->state_memory().device_base && state.allocation_generation==p.owner->state_memory().allocation_generation &&
        emitted.size()<=p.owner->budget().maximum_candidate_depth+1U,"MTP reconciliation requires actual committed target rows and owner state");
    // Output tokens do not attest draft-cache correctness; real committed
    // tickets and complete teacher-pair coverage above are the authority.
    host_phase_mark("9a reconcile validation");
    {
        const auto admission=p.owner->admit_native_entry();
        for(const auto& row:p.replay) p.preflight(row.hidden.source_position(),row.next_input_token,true,true);
    }
    host_phase_mark("9b reconcile preflight");
    try {
        if(p.owner->qsa_pool()) p.owner->qsa_pool()->invalidate(true,p.base_source);
        p.next_source=p.base_source;
        // Earlier occupied rows were never overwritten. The whole proposal
        // suffix is now logically invalid until actual teacher pairs overwrite
        // its committed prefix in order. Bytes beyond next_source stay unused;
        // they are not restored, compared as committed data or read by a later
        // attention row before that row has produced its own cache values.
        p.last_target=p.base_previous_target;p.last_teacher_next_input=p.base_previous_input;
        std::size_t first=0;
        if(p.draft) {
            if((p.draft_exact || p.cache_profile!=DraftCacheProfile::teacher) &&
               !p.replay.empty() && p.first_cache_reusable && p.replay[0].hidden.same_commit(*p.cycle_base) &&
               p.replay[0].hidden.source_position()==p.base_source && p.replay[0].hidden.view()==p.cycle_base->view() &&
               p.replay[0].next_input_token==p.pending_token) {p.reuse_first_teacher_cache(p.replay[0]);first=1;}
            host_phase_mark("9c reuse first cache");
            p.teacher_cache_rows(std::span<const HipMtpReplayRow>(p.replay).subspan(first),
                !p.draft_exact && p.cache_profile==DraftCacheProfile::fast);
            host_phase_mark("9d teacher cache rows (gpu+sync)");
            p.draft->observe(emitted);first=p.replay.size();
            host_phase_mark("9e draft observe");
        }
        for(std::size_t i=first;i<p.replay.size();++i) {
            const auto& row=p.replay[i];
            const bool reuse=i==0U && p.first_cache_reusable && p.cycle_base &&
                row.hidden.same_commit(*p.cycle_base) && row.hidden.source_position()==p.base_source &&
                row.hidden.view()==p.cycle_base->view() && row.next_input_token==p.pending_token;
            if(reuse)p.reuse_first_teacher_cache(row);
            else p.teacher_cache(row.hidden,row.next_input_token);
        }
        require(p.next_source==p.reconcile_final->source_position(),"MTP replay did not reach the actual committed target frontier");
        p.cycle=false;p.cycle_base.reset();p.base_previous_target.reset();p.base_previous_input.reset();
        p.replay.clear();p.reconcile_final.reset();p.have_replay=false;p.candidates.clear();p.first_cache_reusable=false;
        // Diagnostic only: compare the occupied proposer prefix across schedules.
        static const bool audit=[] {const char* v=std::getenv("STRIX_MTP_CACHE_AUDIT");return v && v[0]=='1';}();
        if(audit) {
            const auto cache=cache_state();
            const auto kv_bytes=cache.logical_rows*cache.kv_row_bytes;
            const auto raw_bytes=cache.logical_rows*cache.raw_index_row_bytes;
            require(kv_bytes<=cache.kv.bytes && raw_bytes<=cache.raw_index.bytes &&
                kv_bytes+raw_bytes<=(1ULL<<30U),"MTP cache audit exceeds occupied views or 1 GiB");
            std::vector<std::byte> kv(static_cast<std::size_t>(kv_bytes)),raw(static_cast<std::size_t>(raw_bytes));
            const auto proof=p.owner->admit_native_entry();p.owner->mark_submitted();
            if(kv_bytes)check(hipMemcpyAsync(kv.data(),reinterpret_cast<const void*>(cache.kv.address),kv.size(),
                hipMemcpyDeviceToHost,static_cast<hipStream_t>(p.owner->stream())),"MTP cache audit KV read");
            if(raw_bytes)check(hipMemcpyAsync(raw.data(),reinterpret_cast<const void*>(cache.raw_index.address),raw.size(),
                hipMemcpyDeviceToHost,static_cast<hipStream_t>(p.owner->stream())),"MTP cache audit raw-index read");
            p.owner->complete();
            const auto kh=model_content_identity_hex(sha256_bytes(std::span<const std::byte>(kv)));
            const auto rh=model_content_identity_hex(sha256_bytes(std::span<const std::byte>(raw)));
            std::fprintf(stderr,"mtp_cache at=%llu profile=%s kv=%s raw=%s\n",
                static_cast<unsigned long long>(p.next_source),draft_cache_profile(),kh.c_str(),rh.c_str());
        }
    } catch(...) {p.failure();throw;}
}
} // namespace strix
