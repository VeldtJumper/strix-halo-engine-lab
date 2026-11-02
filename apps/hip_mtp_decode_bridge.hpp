#include "strix/startup_markers.hpp"
#pragma once
#include "qwen_yarn_memory.hpp"
#include "mtp_asset_verification.hpp"
#include "strix/prompt_terminal_diagnostics.hpp"
#include "qwen_tiled_prompt_plan.hpp"
#include "strix/serve/engine.hpp"
#include "strix/qpf2k_defaults.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/serve/health.hpp"
#include "strix/batch_service_handoff.hpp"
#include "strix/batch_nested_prefill.hpp"
#include "mtp_prompt_phase.hpp"
#include "strix/execution_telemetry.hpp"
#include "strix/known_prefill_limits.hpp"
#include "hip_decode_experiment.hpp"
#include "hip_decode_options.hpp"
#include "http_slot_capacity.hpp"
#include "hip_decode_receipt_json.hpp"
#include "mtp_window_memory_reservation.hpp"
#include "strix/mtp_payload_geometry.hpp"
#include "hip_prefix_state_cache.hpp"
#include "hip_mtp_wide_copy.hpp"
#include "strix/moe_q4k_a8_runtime.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/hip_known_prefix_window.hpp"
#include "strix/hip_known_prompt_backend.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_mtp_asset_loader.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_mtp_forward_replay.hpp"
#include "strix/mtp_forward_golden_file.hpp"
#include "strix/hip_mtp_greedy_runtime.hpp"
#include "strix/mtp_batch_window_controller.hpp"
#include "hip_session_store.hpp"
#include <mutex>
#include "strix/hip_mtp_window_verifier.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/mtp_draft_activation.hpp"
#include "strix/mtp_draft_qualification.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/linux_uma_observation.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/session_executor_handoff.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include "strix/serve/constraint_selection.hpp"
#include <span>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <utility>
#include <cstdlib>

namespace strix::decode_cli {
// Serving (--http-serve): after the calibration replay the bridge hands a request runner to the
// serve hook, which calls it for one request at a time on the bridge's thread until it returns.
// Each request starts from a fresh target state (restored from the prefix state cache when a
// cached block boundary matches) and decodes greedily with MTP verification.
using MtpConstraintRow = serve::ConstraintRowSelection;
struct MtpServeRequest {
    std::function<void()> on_admitted;
    serve::BackendProbeSlot* backend_probe = nullptr;
    std::shared_ptr<MtpConstraintRow> constraint;
    serve::CacheRetention cache_retention = serve::CacheRetention::server_policy;
    std::string session_id, session_operation, session_durability;
    std::span<const std::uint32_t> prompt;
    std::uint64_t max_tokens = 0;
    std::span<const std::uint32_t> stop_tokens;
    // Context n-gram draft extension for this request: forced on/off, or unset = the engine's default.
    std::optional<bool> ngram_draft;
    // Newly verified tokens after each cycle (the first call carries the prompt's own prediction);
    // false stops the request.
    std::function<bool(std::span<const std::uint32_t>)> on_tokens;
    std::function<void(serve::GenerationProgress)> on_progress;
};
struct MtpServeResult {
    std::optional<serve::ConstraintVerdict> constraint;
    std::string error;
    BatchServiceStats service;
    std::string prompt_state_digest,final_state_digest,final_state_components;
    std::uint64_t final_state_position=0;
    std::uint64_t restored_tokens = 0, generated_tokens = 0, cycles = 0, proposed = 0, accepted = 0;
    std::uint64_t paid_drafts = 0;
    bool qsa_mtp_pool_active = false; // actual request owner, not the requested default
    std::array<std::uint64_t,9> chosen_window_rows{};
    std::uint64_t ngram_rows = 0, ngram_accepted = 0;
    // Context-source diagnostics (batched, opt-in sources): shadow proposals made / scored / leading rows right
    // (n-gram or context drafter), and the frame drafter's cycles inside a frame / its shadow proposals.
    std::uint64_t source_shadow = 0, source_scored = 0, source_right = 0, frame_cycles = 0, frame_shadow = 0;
    std::uint64_t tool_rows = 0, tool_accepted = 0; // strict tool-call frame rows proposed / accepted (batched, default on)
    double prompt_ms = 0.0, decode_ms = 0.0;
    std::string finish = "length"; // stop|length|context|cancelled
    // --rtx-state-import: the receipt of the imported state this request started from (JSON object), else empty.
    std::string rtx_import_json;
    std::string session_json, cache_refusal;
};
inline std::string batch_service_receipt(const BatchServiceStats& s) {
    std::ostringstream out;
    out<<"\"service_rounds\":"<<s.service_rounds<<",\"service_points\":"<<s.service_points
       <<",\"service_ms_suspended\":"<<s.service_ms_suspended<<",\"max_bundle_gpu_ms\":"<<s.max_bundle_gpu_ms
       <<",\"service_overshoot_ms_max\":"<<s.service_overshoot_ms_max<<",\"cancel_points_cancelled\":"<<s.cancel_points_cancelled;
    out<<",\"nested_admits\":"<<s.nested_admits<<",\"nested_rows\":"<<s.nested_rows<<",\"nested_ms\":"<<s.nested_ms
       <<",\"nest_rows_limit\":"<<s.nest_rows_limit<<",\"nest_skips\":"<<s.nest_skips<<",\"nest_skip_reasons\":{";
    bool first=true;
    for(const auto& [reason,count]:s.nest_skip_reasons) {if(!first) out<<',';first=false;out<<'"'<<reason<<"\":"<<count;}
    out<<'}';
    return out.str();
}
using MtpServeRunner = std::function<MtpServeResult(const MtpServeRequest&)>;
using MtpDecodeServe = std::function<void(const MtpServeRunner&,bool selection_mtp_available)>;
class MtpBatchRunner;
struct MtpBatchSlotSetup;
// Batched decoding (--http-batch, cohort requests on the tiled prompt): after the calibration replay the
// bridge builds one slot per request it can hold and hands the runner to this hook.
using MtpDecodeBatchServe = std::function<void(MtpBatchRunner&)>;
// These are the actual target parents. The PLE backend pins its source mapping;
// the private hash/prefetch metadata remains stable until all native owners die.
// No pointer to a caller's temporary PLE transaction is retained.
struct MtpDecodeTargetParents final {
    std::shared_ptr<HipDecodeExperimentResources> experiment;
    std::shared_ptr<HipLauncherCatalog> catalog;
    std::shared_ptr<PleLookupBackend> ple;
    std::shared_ptr<const HipMtpAssets> assets; // final sidecar registrations precede credited table drop
    PleHashContract hash;
    PleLookupPrefetchState prefetch;
    ExecutionRuntimeResources resources;
    // Split prefill numerical profile: the sibling serial execution that runs the prompt rows,
    // its launch dispatcher and `resources` with the sibling view of the target scratch. All are
    // retained by the caller through the bridge call. Null compiled: one profile for every row.
    struct PromptExecution {
        const CompiledSerialExecution* compiled = nullptr;
        DirectLaunchDispatcher* dispatcher = nullptr;
        ExecutionRuntimeResources resources;
    } prompt;
    // Tiled MTP prompt: the prompt runs through the tiled known-prefix prefill (complete T-row
    // blocks, optional smaller tail blocks, the rest serially through the MTP target producer) and
    // the predictor cache is built from every block row's wide hidden state. Null blocks: the
    // ordinary serial MTP prompt.
    struct TiledPrompt {
        HipKnownPromptBackend* blocks = nullptr;
        std::shared_ptr<HipKnownPrefixWindow> block_owner;
        struct Tail {
            HipKnownPromptBackend* backend = nullptr;
            std::shared_ptr<HipKnownPrefixWindow> owner;
            std::uint32_t rows = 0;
        };
        std::vector<Tail> tails; // descending rows
        // Split prefill profile with --mtp-copy-wide-rows: the wide copy verify owners on the decode execution (no
        // backend; the tails keep the prefill profile). Empty with one profile: the tails verify the wide cycles.
        std::vector<Tail> wide;
        std::uint32_t rows = 0;
        QwenTiledPlanConfig numerical_plan; // actual owner/profile identities, independent of storage
        // Split prefill numerical profile: the sibling prefill execution every block and tail backend hands
        // the session to around its block (0: one profile). Calibration, serial and decode rows keep `compiled`.
        std::uint64_t prefill_execution_identity = 0;
        HipPrefixStateCacheSession* cache = nullptr;
        bool cache_tails = false;
        std::vector<std::uint32_t> ladder_tail_rows; // STRIX_QPF2K_SCHED: configured tail ladder (empty: the tails')  // --prefix-cache-block-rows < T: tail block boundaries are checkpoints too
        // --rtx-state-import: the imports the HTTP routes fill; `cache` restores a parked one (docs/rtx-state-import.md).
        std::shared_ptr<RtxStateImportRegistry> rtx_import;
        // Fresh target state and session, exactly as after setup (after the calibration replay).
        std::function<void()> reset_state;
        std::function<void()> reset_prefill_memory;
        SerialSessionContext fresh;
        // Prompt groups (STRIX_PROMPT_BATCH): the owner the blocks of several requests run on, when it is not the
        // T-row block owner (STRIX_PROMPT_BATCH_ROWS above T). Null: the blocks backend and its owner.
        HipKnownPromptBackend* group = nullptr;
        std::shared_ptr<HipKnownPrefixWindow> group_owner;
        // STRIX_PROMPT_WINDOW=1: the request's window verifier, which runs the serial suffix as known-row windows.
        HipMtpWindowVerifier* window = nullptr;
    } tiled;
    // Set: serve requests after the calibration replay instead of the command-line request.
    MtpDecodeServe serve;
    // Set: batched decoding after the calibration replay. This request's state is slot 0; batch_slots are
    // the further slots (their own parents, session and dispatcher, window owners on a shared stream).
    MtpDecodeBatchServe serve_batch;
    std::vector<MtpBatchSlotSetup> batch_slots;
    std::size_t batch_capacity = 0;
    std::function<MtpBatchSlotSetup()> grow_batch_slot;
    std::function<bool()> batch_memory_ok;
    // Closed CLI only: M1 remains live while the prompt primes predictor rows.
    std::function<void()> prepare_prompt_pool;
    std::function<void()> retire_prompt;
    struct WindowSetupReceipt {
        std::uint64_t union_source_steps = 0;
        std::uint64_t union_reserved_bytes = 0;
        std::uint64_t union_generation = 0;
        std::uint64_t retained_window_bytes = 0;
    } window_setup;
    MtpDecodeTargetParents(std::shared_ptr<HipDecodeExperimentResources> actual,
        std::shared_ptr<HipLauncherCatalog> launchers,std::shared_ptr<PleLookupBackend> lookup,const PleHashContract& contract)
        :experiment(std::move(actual)),catalog(std::move(launchers)),ple(std::move(lookup)),hash(contract) {
        if(!experiment || !catalog || !ple)throw std::invalid_argument("MTP CLI requires actual retained target/launcher/PLE parents");
        resources=experiment->resources;
        resources.ple_lookup=ple.get();resources.ple_hash_contract=&hash;resources.ple_prefetch=&prefetch;
    }
    MtpDecodeTargetParents(const MtpDecodeTargetParents&)=delete;
    MtpDecodeTargetParents& operator=(const MtpDecodeTargetParents&)=delete;
};
// Private diagnostic evidence only. Every copied view is an actual completed,
// retained producer output. This does not run a reference forward or alter state.
inline void mtp_pool_finish_diagnostics(const MtpDecodeTargetParents& parents,const HipMtpRuntime& runtime,
    const SerialSessionContext& session,const HipMtpTargetHiddenTicket& pending) {
    if(pending.valid())pending.print_step_logits_summary(session.next_position);
    if(parents.resources.qsa_committed_pool) parents.resources.qsa_committed_pool->print_stats();
    const char* digest=std::getenv("STRIX_MTP_LOGITS_DIGEST");
    if(!digest || digest[0]!='1' || !pending.valid())return;
    if(const auto sha=pending.serial_logits_digest())
        std::fprintf(stderr,"mtp-logits terminal position=%llu sha256=%s\n",static_cast<unsigned long long>(pending.source_position()),
            model_content_identity_hex(*sha).c_str());
    else std::fprintf(stderr,"mtp-logits terminal NO-REFERENCE position=%llu\n",static_cast<unsigned long long>(pending.source_position()));
    const auto view=pending.view();std::vector<std::byte> held(static_cast<std::size_t>(view.bytes));
    const auto stream=static_cast<hipStream_t>(runtime.resources().stream());
    if(hipMemcpyAsync(held.data(),reinterpret_cast<const void*>(view.address),held.size(),hipMemcpyDeviceToHost,stream)!=hipSuccess ||
        hipStreamSynchronize(stream)!=hipSuccess)throw std::runtime_error("MTP final actual hidden diagnostic copy");
    std::fprintf(stderr,"mtp-final-hidden position=%llu predictor_end=%llu lineage=%llu sha256=%s\n",
        static_cast<unsigned long long>(session.next_position),static_cast<unsigned long long>(runtime.next_source_position()),
        static_cast<unsigned long long>(session.arithmetic_lineage.history_identity),model_content_identity_hex(sha256_bytes(held)).c_str());
}
inline void write_mtp_union_observation(std::ostream& output,
    const std::optional<HipKnownPrefixUnionObservation>& observed) {
    if (!observed) { output << "null"; return; }
    const auto& group = observed->group;
    output << "{\"scope\":\"last-completed-window-last-eligible-layer-including-restored-windows-not-aggregate-reuse-or-traffic\","
        << "\"first_position\":" << observed->first_position << ",\"layer\":" << observed->layer
        << ",\"logical_rows\":" << group.logical_rows << ",\"status\":" << group.status
        << ",\"version\":" << group.version << ",\"assignments\":" << group.assignments
        << ",\"touched\":" << group.touched << ",\"batches\":" << group.batches
        << ",\"duplicate_assignments\":" << group.duplicate_assignments
        << ",\"maximum_assignments\":" << group.maximum_assignments << ",\"rows_per_expert_histogram\":[";
    for (unsigned i = 0; i < 8U; ++i) output << (i ? "," : "") << group.rows_per_expert_histogram[i];
    output << "],\"batch_size_histogram\":[";
    for (unsigned i = 0; i < 8U; ++i) output << (i ? "," : "") << group.batch_size_histogram[i];
    output << "]}";
}
inline std::uint64_t mtp_cli_add(std::uint64_t a,std::uint64_t b) {
    if(b>UINT64_MAX-a)throw std::overflow_error("MTP CLI payload budget overflow");
    return a+b;
}
inline std::uint64_t mtp_cli_mul(std::uint64_t a,std::uint64_t b) {
    if(b && a>UINT64_MAX/b)throw std::overflow_error("MTP CLI payload budget overflow");
    return a*b;
}
// Sampling receipt: the policy, its identity and the decision counts of this request.
inline std::string mtp_sampling_receipt(const MtpCliOptions& m,const TokenSampler* sampler) {
    std::ostringstream out;out<<std::setprecision(9);
    if(!sampler){out<<"{\"identity\":\"greedy-first-max\"}";return out.str();}
    const auto& p=sampler->policy();const auto& s=sampler->stats();
    out<<"{\"identity\":\"sampling-v1\",\"temperature\":"<<p.temperature<<",\"top_k\":"<<p.top_k<<",\"top_p\":"<<p.top_p
       <<",\"min_p\":"<<p.min_p<<",\"seed\":"<<p.seed<<",\"policy_identity\":"<<token_sampling_policy_identity(p)
       <<",\"speculation\":"<<(m.sample_speculation?"true":"false")<<",\"schedule\":\""<<(m.sample_timing_schedule?"adaptive-timed":"fixed")
       <<"\",\"drafts\":"<<s.drafts<<",\"plain_decisions\":"<<s.plain<<",\"accepted_drafts\":"<<s.accepted
       <<",\"rejected_drafts\":"<<s.rejected<<"}";
    return out.str();
}
using MtpDecodeWindowFactory=std::function<std::shared_ptr<HipMtpWindowVerifier>(std::shared_ptr<HipMtpTargetHidden>)>;
struct MtpDecodeQualificationContext {
    const Options& options;
    std::shared_ptr<MtpDecodeTargetParents> parents;
    std::shared_ptr<HipMtpResources> predictor;
    std::shared_ptr<HipMtpTargetHidden> target;
    std::shared_ptr<HipMtpWindowVerifier> window;
    SerialSessionContext& session;
    DirectLaunchDispatcher& dispatcher;
    AllocationGenerationSource& generations;
    HipMtpGreedyRuntime& greedy;
    HipMtpRuntime& runtime;
    HipMtpTargetHiddenTicket initial;
};
using MtpDecodeQualification=std::function<void(MtpDecodeQualificationContext&)>;
using MtpDecodeComponentDiagnostic=std::function<void(const std::shared_ptr<HipMtpResources>&,std::uint64_t)>;

// The receipt of an imported state (--rtx-state-import).
inline std::string rtx_import_receipt(const PrefixStateCacheRequestStats& q) {
    std::ostringstream out;
    out<<"{\"id\":\""<<q.rtx_id<<"\",\"position\":"<<q.rtx_position<<",\"bytes\":"<<q.rtx_bytes<<",\"chunks\":"<<q.rtx_chunks
       <<",\"restore_ms\":"<<q.rtx_restore_ms<<",\"verify_ms\":"<<q.rtx_verify_ms<<",\"sha_ok\":"<<(q.rtx_sha_ok?"true":"false")
       <<",\"manifest_sha256\":\""<<q.rtx_manifest_sha256<<"\"";
    if(q.rtx_local_rows) out<<",\"local_rows\":"<<q.rtx_local_rows<<",\"local_bytes\":"<<q.rtx_local_bytes;
    out<<"}";
    return out.str();
}
// Tiled MTP prompt after the calibration replay. Complete T-row blocks (then complete tail blocks)
// run through the tiled prefill; a step observer copies every row's wide hidden state (the MTP
// target tap) out of the block arena, and the predictor's cache rows are built from those copies
// (pair (h_i, t_{i+1})). The remaining rows, at least the last prompt row, run serially through
// the MTP target producer, whose last ticket is the pending row of the first cycle. With a prefix
// state cache the longest cached block boundary P is restored first (target state, predictor
// rows [0, P-1) and h_{P-1}), and selected block boundaries are captured.
// The tiled MTP prompt as resumable steps (one T-row block, one tail block or up to `serial_rows_per_step`
// serial rows per step), so that a batched run can interleave a joining request's prompt with the decode cycles
// of the others. The sequence of operations is run_tiled_mtp_prompt's, whatever the interleaving.
struct BatchCertContext {
    bool enabled=false,allow_restore=false,nesting_possible=false;
    QwenBatchConfig config;
};
class TiledMtpPrompt {
public:
    TiledMtpPrompt(std::span<const std::uint32_t> tokens,const ModelContract& model,const CompiledSerialExecution& compiled,
        MtpDecodeTargetParents::TiledPrompt& tiled,SerialSessionContext& session,HipMtpTargetHidden& target,HipMtpRuntime& runtime,
        const QualifiedMtpDraftPlan& qualified,const ActivatedMtpDraftPlan& activated,const ExecutionRuntimeResources& resources,
        DirectLaunchDispatcher& dispatcher,std::byte* held,std::shared_ptr<TokenSampler> final_sampler={},
        HipMtpHostRowSelector final_selector={},std::function<void(serve::GenerationProgress)> progress={}, bool session_resume=false, bool tiled_suffix=true, bool batch=false, BatchCertContext batch_cert={}, bool defer_cache_lookup=false)
        :tokens_(tokens.begin(),tokens.end()),tiled_(tiled),session_(session),target_(target),runtime_(runtime),qualified_(qualified),
         activated_(activated),resources_(resources),dispatcher_(dispatcher),held_(held),final_sampler_(std::move(final_sampler)),final_selector_(std::move(final_selector)),progress_(std::move(progress)) {
        producer_rotary_=model.rotary_identity();
        require_rotary_producer_identity(producer_rotary_,target_.model().rotary_identity());
        require_rotary_producer_identity(producer_rotary_,runtime_.resources().model().rotary_identity());
        const auto admit_owner=[&](const std::shared_ptr<HipKnownPrefixWindow>& owner) {
            if(owner) require_rotary_producer_identity(owner->rotary_identity(),producer_rotary_);
        };
        admit_owner(tiled_.block_owner);admit_owner(tiled_.group_owner);
        for(const auto& tail:tiled_.tails) admit_owner(tail.owner);
        for(const auto& wide:tiled_.wide) admit_owner(wide.owner);
        n_=static_cast<std::uint64_t>(tokens_.size());
        if(!n_)throw std::invalid_argument("tiled MTP prompt is empty");
        // Sampled decoding: only the final prompt row selects through the sampler (earlier rows' predictions
        // are never emitted), so the target holds it from that row on.
        if(final_sampler_ && target_.token_sampler()) throw std::invalid_argument("tiled MTP prompt sampler is set on the final row only");
        width_=mtp_cli_mul(mtp_cli_mul(model.hidden_size,model.hc_count),2U);
        // The target tap: the last MLP injection before the final hyper mix (same step in every sibling).
        for(std::size_t i=1;i<compiled.execution.steps.size();++i)
            if(compiled.execution.steps[i].op==SerialOp::final_hyper_mix && compiled.execution.steps[i-1U].op==SerialOp::mlp_inject) tap_=i-1U;
        if(tap_==SIZE_MAX)throw std::logic_error("tiled MTP prompt has no wide hidden tap step");
        // STRIX_PREFILL_PIPE=1: two tap buffers, so block k's predictor rows build on the MTP stream while block k+1 runs.
        const auto buffer=static_cast<std::size_t>(mtp_cli_mul(tiled_.rows,width_));
        check(hipMalloc(reinterpret_cast<void**>(&block_hidden_),pipe_mtp()?2U*buffer:buffer),"tiled MTP hidden rows");
        hidden_allocation_=block_hidden_;
        if(pipe_mtp()) spare_hidden_=block_hidden_+buffer;
        plan_config_=tiled_.numerical_plan;
        plan_config_.rows=tiled_.rows;
        // Bit 2 is inert on this route (no block is deferred): plan, batch certification and identity use 12 for 14.
        plan_config_.schedule=qwen_bridge_plan_schedule(std::getenv("STRIX_QPF2K_SCHED"));
        const char* ts=std::getenv("STRIX_MTP_TILED_START");
        plan_config_.tiled_start=!ts || ts[0]!='0';
        plan_config_.suppressed=qpf2k::suppressed(qpf2k::Item::sched,tiled_.blocks->gsq_model());
        plan_config_.initial_generations=session_.state.generations;
        plan_config_.valid_state=static_cast<std::uint32_t>(session_.state.valid);
        for(const auto& step:compiled.execution.steps) {
            const auto flags=static_cast<std::uint32_t>(step.writes_state);
            plan_config_.valid_state|=flags;
            for(std::size_t f=0;f<plan_config_.generations_per_row.size();++f)
                if(flags&(1U<<f)) ++plan_config_.generations_per_row[f];
        }
        plan_config_.priming=pipe_flags();
        plan_config_.force_sync_priming=bool(rtx_state_export_request());
        const auto policy=[](KnownWindowOutputPolicy p) {return p==KnownWindowOutputPolicy::state_only?
            QwenPlanOutput::state_only:p==KnownWindowOutputPolicy::all_rows?QwenPlanOutput::all_rows:QwenPlanOutput::terminal;};
        plan_config_.block.rows=tiled_.rows;plan_config_.block.policy=policy(tiled_.blocks->output_policy());
        plan_config_.tails.resize(tiled_.tails.size());
        for(std::size_t k=0;k<tiled_.tails.size();++k) {
            plan_config_.tails[k].rows=tiled_.tails[k].rows;
            plan_config_.tails[k].policy=policy(tiled_.tails[k].backend->output_policy());
        }
        plan_config_.ladder=tiled_.ladder_tail_rows;
        batch_=batch;batch_cert_=batch_cert;
        certified_request_=!final_selector_ && (!batch || (batch_cert.enabled && batch_cert.config.allowed(plan_config_.schedule))) &&
            !session_resume && !tiled_.rtx_import && !rtx_state_export_request();
        if(tiled_.cache && !legacy_cache()) {
            if(batch && batch_cert.enabled && !certified_request_ && tiled_.cache->has_store())
                tiled_.cache->store().record_batch_config_skip();
            tiled_.cache->set_plan(plan_config_,n_,certified_request_,batch && batch_cert.enabled?2:1,
                batch_cert.config,!batch || (certified_request_ && batch_cert.allow_restore),batch_cert.nesting_possible);
            if(prefix_batch_certification_enabled()) tiled_.cache->bind_restore_runtime([&runtime,&qualified,&activated](std::uint64_t p) {
                runtime.restore_cache_frontier(qualified,activated,p-1U);
            });
        }
        if(tiled_.cache) tiled_.cache->set_policy_selection(!!final_selector_);
        deferred_lookup_=defer_cache_lookup && tiled_.cache && tiled_.cache->joining_allowed() && !session_resume;
        start_=std::chrono::steady_clock::now();
        if(session_resume) {
            restored_=session_.next_position;
            if(!restored_ || restored_>=n_ || runtime_.next_source_position()+1U!=restored_)
                throw std::invalid_argument("MTP session suffix frontier differs");
        } else if(tiled_.cache && !deferred_lookup_) restored_=tiled_.cache->begin_request(session_,tokens_);
        pos_=restored_;
        if(restored_ && !session_resume && !tiled_.rtx_import && !legacy_cache()) {
            const auto& cursor=tiled_.cache->continuation();
            if(!cursor) throw std::logic_error("prefix restore lacks a preflighted cold continuation");
            phase_=cursor->phase; tail_=cursor->tail; balanced_=cursor->balanced;
        }
        if(restored_) report_progress();
        if(session_resume) {
            const auto plan=plan_session_suffix(pos_,n_-pos_,tiled_.rows,tiled_suffix);
            if(!plan.tiled) phase_=Phase::serial;
            else align_end_=pos_+plan.serial_rows;
        }
    }
    ~TiledMtpPrompt(){
        // Pending priming still reads the tap rows and the held row.
        if(capture_) {
            try {
                if(submitter_.valid()) (void)submitter_.get();
                if(runtime_.prime_hidden_rows_pending())
                    check(hipStreamSynchronize(static_cast<hipStream_t>(runtime_.resources().stream())),"cancelled prefix producer drain");
            } catch(...) {capture_->fatal();std::terminate();}
        } else {
            if(submitter_.valid()) submitter_.wait();
            if(runtime_.prime_hidden_rows_pending()) (void)hipStreamSynchronize(static_cast<hipStream_t>(runtime_.resources().stream()));
        }
        if(capture_ && !capture_->committed) capture_->cancel_target();
        if(hidden_allocation_)(void)hipFree(hidden_allocation_);
    }
    TiledMtpPrompt(const TiledMtpPrompt&)=delete;
    TiledMtpPrompt& operator=(const TiledMtpPrompt&)=delete;
    [[nodiscard]] bool done() const noexcept {return phase_==Phase::done;}
    [[nodiscard]] std::uint64_t position() const noexcept {return pos_;}
    [[nodiscard]] std::uint64_t remaining() const noexcept {return n_-pos_;}
    [[nodiscard]] std::uint64_t restored() const noexcept {return restored_;}
    [[nodiscard]] bool lookup_deferred() const noexcept {return deferred_lookup_;}
    // Only an untouched cold slot can retry lookup after its producer publishes.
    void retry_join_lookup() {
        if(!deferred_lookup_ || pos_ || restored_ || !certified_request_ || !tiled_.cache || !tiled_.cache->joining_allowed())
            throw std::logic_error("prefix join retry requires an untouched certified prompt");
        deferred_lookup_=false;
        restored_=tiled_.cache->begin_request(session_,tokens_);
        pos_=restored_;
        if(restored_) {
            const auto& cursor=tiled_.cache->continuation();
            if(!cursor) throw std::logic_error("prefix join lacks cold continuation");
            phase_=cursor->phase;tail_=cursor->tail;balanced_=cursor->balanced;
            report_progress();
        }
    }
    // The prompt's next unit when it is a tiled block (a T-row block or a tail block): its backend, owner, rows,
    // whether it is a T block (prefix cache capture) and whether it ends the prompt. Null when the next unit is
    // serial rows or the prompt is done. Only advances past exhausted block phases, as step() would.
    struct Block {
        HipKnownPromptBackend* backend=nullptr;
        HipKnownPrefixWindow* owner=nullptr;
        std::uint32_t rows=0;
        bool capture=false,final=false;
    };
    // Units of the largest tail size instead of whole T blocks from now on (on) or again T blocks (off).
    void set_small_blocks(bool on) noexcept {small_=on;}
    void force_ladder() noexcept {forced_ladder_=true;}
    [[nodiscard]] BatchNestedRecipe nested_recipe(std::uint64_t limit,bool qualified) const {
        if(tiled_.blocks->output_policy()==KnownWindowOutputPolicy::all_rows) return BatchNestedRecipe{{},0,"route"};
        std::vector<std::uint32_t> owners;
        for(const auto& tail:tiled_.tails) {
            if(!tail.backend || !tail.owner) return BatchNestedRecipe{{},0,"missing_owner"};
            if(tail.backend->logical_rows()!=tail.rows || tail.owner->logical_rows()!=tail.rows ||
               tail.backend->output_policy()==KnownWindowOutputPolicy::all_rows) return BatchNestedRecipe{{},0,"row_class"};
            owners.push_back(tail.rows);
        }
        static const bool start=[]{const char* v=std::getenv("STRIX_MTP_TILED_START");return !v || v[0]!='0';}();
        const bool heads=start && tiled_.blocks->output_policy()!=KnownWindowOutputPolicy::state_only;
        for(const auto& tail:tiled_.tails) if(tail.backend &&
            (start && tail.backend->output_policy()!=KnownWindowOutputPolicy::state_only)!=heads)
            return BatchNestedRecipe{{},0,"row_class"};
        // No explicit ladder: the tail owners are the ladder (as next_block runs them).
        const auto& ladder=tiled_.ladder_tail_rows.empty()?owners:tiled_.ladder_tail_rows;
        return batch_nested_recipe(remaining(),limit,tiled_.rows,ladder,owners,qualified,heads);
    }
    [[nodiscard]] std::optional<Block> next_block() {
        if(forced_ladder_ && phase_==Phase::blocks) phase_=Phase::tails;
        if(final_selector_ || legacy_cache()) return legacy_next_block();
        QwenPlanCursor cursor{phase_,tail_,balanced_};
        const auto u=qwen_prompt_next(plan_config_,cursor,pos_,n_,false,small_,adopted_);
        phase_=cursor.phase; tail_=cursor.tail; balanced_=cursor.balanced;
        if(!u || u->kind==QwenPlanKind::serial || u->kind==QwenPlanKind::window) return std::nullopt;
        if(u->t_owner) return Block{tiled_.blocks,tiled_.block_owner.get(),
            static_cast<std::uint32_t>(u->end-u->begin),u->capture,u->final};
        auto& tail=tiled_.tails[u->tail];
        return Block{tail.backend,tail.owner.get(),static_cast<std::uint32_t>(u->end-u->begin),u->capture,u->final};
    }
    [[nodiscard]] std::optional<Block> legacy_next_block() {
        while(phase_==Phase::blocks || phase_==Phase::tails) {
            const bool blocks=phase_==Phase::blocks;
            if(!blocks && !tiled_.ladder_tail_rows.empty()) {
                // STRIX_QPF2K_SCHED bit 8 fail-closed: rows the ladder runs in one unit must not fall to a smaller tail
                // owner or to the serial suffix (row classes sit on a 128-row grid per block; a pruned owner would change
                // their arithmetic silently). Owners larger than the unit are skipped below and re-checked next pass.
                static const bool start=[]{const char* v=std::getenv("STRIX_MTP_TILED_START");return !v || v[0]!='0';}();
                // rows the current owner may take (the step below: pos + rows <= n - 1, or == n with heads)
                const bool theads=start && (tail_>=tiled_.tails.size() ||
                    tiled_.tails[tail_].backend->output_policy()!=KnownWindowOutputPolicy::state_only);
                const std::uint64_t m=n_-pos_-(theads?0U:1U);
                std::uint32_t want=0;
                for(const auto r:tiled_.ladder_tail_rows) if(r && m>=r) {want=r;break;}
                if(want && (tail_>=tiled_.tails.size() || tiled_.tails[tail_].rows<want))
                    throw std::logic_error("qpf2k sched: the ladder's "+std::to_string(want)+"-row unit has no tail owner (pruned)");
            }
            if(!blocks && tail_>=tiled_.tails.size()){phase_=Phase::serial;break;}
            auto& backend=blocks?*tiled_.blocks:*tiled_.tails[tail_].backend;
            auto& owner=blocks?*tiled_.block_owner:*tiled_.tails[tail_].owner;
            const auto rows=blocks?tiled_.rows:tiled_.tails[tail_].rows;
            // STRIX_MTP_TILED_START=0: the last prompt row always runs serially (the previous handoff).
            // Otherwise a block that ends exactly at the last prompt row also computes that row's head, and
            // the first draft cycle starts from the block's last row (its tap and FP32 logits).
            static const bool tiled_start=[]{const char* v=std::getenv("STRIX_MTP_TILED_START");return !v || v[0]!='0';}();
            const bool heads=tiled_start && backend.output_policy()!=KnownWindowOutputPolicy::state_only;
            // STRIX_QPF2K_SCHED bit 16: balanced blocks on the T owner (the prompt's rows down to a multiple of 128 as
            // ceil(rest / T) blocks of a multiple of 128 rows <= T, each the first rows of the T arena; the last rest mod 128
            // rows then run as the ladder runs them). Bit 4: the ladder's leading multiple-of-128 tail units merged into one
            // partial block of the T owner. Not with a prefix store (its captures are at T boundaries) nor small units.
            static const bool balanced_raw=[]{const char* v=std::getenv("STRIX_QPF2K_SCHED");return v&&(std::strtoul(v,nullptr,0)&16UL)!=0UL;}();
            static const bool merged_raw=[]{const char* v=std::getenv("STRIX_QPF2K_SCHED");return v&&(std::strtoul(v,nullptr,0)&4UL)!=0UL;}();
            const bool defaults_off=qpf2k::suppressed(qpf2k::Item::sched,backend.gsq_model());
            const bool balanced_env=balanced_raw&&!defaults_off,merged_env=merged_raw&&!defaults_off;
            // Per prompt, decided at its first unit: only where the balanced plan is bitwise equal to the ladder.
            if(blocks && (balanced_env || merged_env) && balanced_<0) {
                std::vector<std::uint32_t> ladder=tiled_.ladder_tail_rows;
                if(ladder.empty()) for(const auto& t:tiled_.tails) ladder.push_back(t.rows);
                // the tiled start: the ladder may end exactly at n (heads); without it the last row is a suffix row
                const std::uint64_t m=heads?n_-pos_:n_-pos_-1U;
                // the balanced plan needs the tiled start (the ladder then covers exactly n - pos rows)
                balanced_=balanced_env && heads && known_prefill_balanced_eligible(m,rows,ladder)?1:0;
                if(balanced_env) std::fprintf(stderr,"qpf2k sched balanced %s for %llu rows\n",balanced_?"on":"off (ladder differs)",static_cast<unsigned long long>(m));
            }
            if(blocks && balanced_env && balanced_==1 && !adopted_ && !small_ && !(tiled_.cache && tiled_.cache->has_store()) &&
               backend.output_policy()!=KnownWindowOutputPolicy::all_rows && heads) {
                const auto size=known_prefill_balanced_rows(n_-pos_,rows);
                if(size) {
                    const auto r=static_cast<std::uint32_t>(size);
                    return Block{&backend,&owner,r,r==rows,pos_+r==n_};
                }
            }
            // Bit 4 (not balanced): the ladder's leading multiple-of-128 tail units merged into one partial block of the T owner.
            if(blocks && merged_env && balanced_==0 && !adopted_ && !small_ && !(tiled_.cache && tiled_.cache->has_store()) &&
               backend.output_policy()!=KnownWindowOutputPolicy::all_rows && heads) {
                std::vector<std::uint32_t> ladder=tiled_.ladder_tail_rows;
                if(ladder.empty()) for(const auto& t:tiled_.tails) ladder.push_back(t.rows);
                if(const auto merged=known_prefill_merged_tail_rows(n_-pos_,rows,ladder)) {
                    const auto r=static_cast<std::uint32_t>(merged);
                    return Block{&backend,&owner,r,false,pos_+r==n_};
                }
            }
            if(!adopted_ && (pos_+rows<=n_-1U || (heads && pos_+rows==n_))) {
                // small_blocks: the T block as units of the largest tail size dividing T (the same rows and bits: a tiled row's
                // arithmetic does not depend on its block's size), e.g. while other requests decode
                if(blocks && small_ && !tiled_.tails.empty() && tiled_.tails[0].rows<rows && rows%tiled_.tails[0].rows==0) {
                    auto& t=tiled_.tails[0];
                    const bool theads=tiled_start && t.backend->output_policy()!=KnownWindowOutputPolicy::state_only;
                    if(pos_+t.rows<=n_-1U || (theads && pos_+t.rows==n_)) return Block{&*t.backend,&*t.owner,t.rows,true,pos_+t.rows==n_};
                }
                return Block{&backend,&owner,rows,blocks,pos_+rows==n_};
            }
            if(blocks)phase_=Phase::tails;else ++tail_;
        }
        return std::nullopt;
    }
    // The block's tokens (valid until the prompt is destroyed).
    [[nodiscard]] std::span<const std::uint32_t> block_tokens(const Block& block) const {
        return std::span<const std::uint32_t>(tokens_).subspan(static_cast<std::size_t>(pos_),block.rows);
    }
    // The step observer of a block: copies every row's wide hidden state (the MTP target tap) out of the arena.
    [[nodiscard]] HipKnownPrefixWindow::PrefillStepObserver block_observer(std::uint32_t rows) {
        return [this,rows](std::size_t step,const std::function<ExecutionDispatchContext(std::uint32_t)>& row) {
            if(step!=tap_) return;
            const auto stream=hip_execution_stream();
            if(pipe_tap_copy() && rows>1U) {
                // STRIX_PREFILL_PIPE bit 2: the same bytes in one 2-D copy when the arena spaces the tap rows evenly.
                std::vector<std::uintptr_t> address(rows);std::uint64_t bytes=UINT64_MAX;
                for(std::uint32_t r=0;r<rows;++r){const auto v=resolve_dispatch_write(row(r),0);address[r]=v.address;bytes=std::min(bytes,v.bytes);}
                const auto pitch=address[1]-address[0];
                bool even=address[1]>address[0] && pitch>=width_ && bytes>=width_;
                for(std::uint32_t r=2;even && r<rows;++r) even=address[r]-address[r-1U]==pitch;
                if(even) {
                    check(hipMemcpy2DAsync(block_hidden_,static_cast<std::size_t>(width_),reinterpret_cast<const void*>(address[0]),
                        static_cast<std::size_t>(pitch),static_cast<std::size_t>(width_),rows,hipMemcpyDeviceToDevice,stream),"tiled MTP tap rows copy");
                    return;
                }
            }
            for(std::uint32_t r=0;r<rows;++r) {
                const auto c=row(r);
                const auto source=resolve_dispatch_write(c,0);
                if(source.bytes<width_)throw std::logic_error("tiled MTP tap row is undersized");
                check(hipMemcpyAsync(block_hidden_+static_cast<std::size_t>(r)*width_,reinterpret_cast<const void*>(source.address),
                    static_cast<std::size_t>(width_),hipMemcpyDeviceToDevice,stream),"tiled MTP tap copy");
            }
        };
    }
    // After the block ran (alone or in a group): the predictor rows, the held row, the prefix cache capture and,
    // for the final block, the adoption of its last row (logits: its terminal logits).
    void finish_block(const Block& block,const std::vector<float>& logits,bool grouped=false,HipKnownPrefixWindow* executed_owner=nullptr) {
        const auto rows=block.rows;
        if(batch_ && batch_cert_.enabled)
            trace_.push_back({pos_,pos_+rows,rows,block.backend==tiled_.blocks,
                (executed_owner?executed_owner:block.owner)==tiled_.block_owner.get(),grouped,small_,forced_ladder_,adopted_});
        drain_priming();
        check_yarn_memory();
        if(block.capture && rows!=tiled_.rows)++small_units_;      // STRIX_BATCH_PROMPT_SMALL: a T block run as tail-size units
        if(grouped && !block.capture)++grouped_tail_blocks_;      // a tail-backend block run in a prompt group
        // STRIX_RTX_EXPORT_STATE keeps every block synchronous: its export (after_unit) must see this block's predictor
        // rows and held row, which the async path below completes only at the next unit's drain.
        // A registry without a store (STRIX_PROMPT_STATE_DIGEST / --rtx-state-import) captures nothing: those blocks stay async.
        if(pipe_mtp() && !block.final && !(legacy_cache() && (block.capture || tiled_.cache_tails)) &&
            !rtx_state_export_request()) {
            // Same rows, kernels and order as below; the wait moves to the next unit of the prompt.
            std::vector<MtpDeviceView> views;std::vector<std::uint32_t> next;
            if(pos_){views.push_back(view(held_));next.push_back(tokens_[static_cast<std::size_t>(pos_)]);}
            for(std::uint32_t r=0;r+1U<rows;++r){views.push_back(view(block_hidden_+static_cast<std::size_t>(r)*width_));
                next.push_back(tokens_[static_cast<std::size_t>(pos_+r+1U)]);}
            const auto stream=static_cast<hipStream_t>(runtime_.resources().stream());
            const auto* last=block_hidden_+static_cast<std::size_t>(rows-1U)*width_;
            if((block.capture || tiled_.cache_tails) && tiled_.cache && !legacy_cache())
                capture_=tiled_.cache->try_prepare_capture(session_,tokens_,block.owner->producer_stream(),
                    !block.owner->completion_pending(),capture_order(block,grouped,stream),n_-1U,block.owner->producer_lifetime(),runtime_.resources().shared_from_this(),batch_?this:nullptr);
            int device=0;check(hipGetDevice(&device),"get priming device");
            const auto submit=[this,device,stream,last,capture=capture_,views=std::move(views),next=std::move(next)] {
                try {
                    check(hipSetDevice(device),"bind priming device");
                    const bool pending=runtime_.prime_hidden_rows_submit(qualified_,activated_,views,next,producer_rotary_);
                    check(hipMemcpyAsync(held_,last,static_cast<std::size_t>(width_),hipMemcpyDeviceToDevice,stream),"tiled MTP held copy");
                    if(capture) capture->submit_predictor(stream);
                    if(!pending) check(hipStreamSynchronize(stream),"tiled MTP held copy");
                    return pending;
                } catch(...) {if(capture && capture->quarantine) capture->fatal();throw;}
            };
            // STRIX_PREFILL_PIPE bit 32: the rows are enqueued from a worker thread while this thread runs the next
            // block (the predictor, its stream, the held row and this block's tap buffer are the worker's until the drain).
            if(pipe_worker()) {submitter_=std::async(std::launch::async,submit);priming_=true;}
            else priming_=submit();
            std::swap(block_hidden_,spare_hidden_);
            pos_+=rows;
            if(!priming_) report_progress();
            if(block.capture)++blocks_;else ++tail_blocks_;
            if(grouped)++grouped_blocks_;
            ++pipe_async_blocks_;
            return;
        }
        std::vector<MtpDeviceView> views;std::vector<std::uint32_t> next;
        if(pos_){views.push_back(view(held_));next.push_back(tokens_[static_cast<std::size_t>(pos_)]);}
        for(std::uint32_t r=0;r+1U<rows;++r){views.push_back(view(block_hidden_+static_cast<std::size_t>(r)*width_));
            next.push_back(tokens_[static_cast<std::size_t>(pos_+r+1U)]);}
        const bool split_capture=(block.capture || tiled_.cache_tails) && tiled_.cache && tiled_.cache->pooled_capture();
        if(split_capture) {
            const auto stream=static_cast<hipStream_t>(runtime_.resources().stream());
            capture_=tiled_.cache->try_prepare_target_capture(session_,tokens_,block.owner->producer_stream(),
                !block.owner->completion_pending(),capture_order(block,grouped,stream),n_-1U,
                block.owner->producer_lifetime(),runtime_.resources().shared_from_this(),batch_?this:nullptr);
        }
        runtime_.prime_hidden_rows(qualified_,activated_,views,next,producer_rotary_);
        check(hipMemcpy(held_,block_hidden_+static_cast<std::size_t>(rows-1U)*width_,static_cast<std::size_t>(width_),hipMemcpyDeviceToDevice),
            "tiled MTP held copy");
        pos_+=rows;
        const bool complete=!block.owner->completion_pending() && !runtime_.prime_hidden_rows_pending() &&
            runtime_.next_source_position()+1U==pos_;
        if((block.capture || tiled_.cache_tails) && tiled_.cache) {
            if(legacy_cache()) tiled_.cache->after_block(session_,tokens_,n_-1U);
            else if(split_capture) {
                if(capture_) {
                    const auto stream=static_cast<hipStream_t>(runtime_.resources().stream());
                    const bool valid=tiled_.cache->complete_target_capture(capture_,session_,complete,stream);
                    capture_->commit(valid);
                    PrefixAsyncCapture::reap();
                    capture_.reset();
                }
            } else if(complete) {
                const auto stream=static_cast<hipStream_t>(runtime_.resources().stream());
                capture_=tiled_.cache->try_prepare_capture(session_,tokens_,block.owner->producer_stream(),true,capture_order(block,grouped,stream),n_-1U,
                    block.owner->producer_lifetime(),runtime_.resources().shared_from_this(),batch_?this:nullptr);
                if(capture_) {
                    capture_->submit_predictor(stream);
                    capture_->commit(true);
                    PrefixAsyncCapture::reap();
                    capture_.reset();
                }
            }
        }
        if(tiled_.cache) tiled_.cache->after_unit(session_,tokens_,pos_);
        if(block.capture)++blocks_;else ++tail_blocks_;
        if(grouped)++grouped_blocks_;
        if(block.final) {
            // STRIX_PREFIX_CACHE_DIGEST=1: also the SHA-256 of the final row's FP32 logits in the receipt.
            static const bool digest=[]{const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST");return v && v[0]=='1';}();
            if(digest) {
                const auto bytes=std::as_bytes(std::span<const float>(logits));
                final_logits_digest_=sha256_bytes(std::vector<std::byte>(bytes.begin(),bytes.end()));
                terminal_logits_digest_=raw_fp32_logits_digest(logits);
            }
            if(final_sampler_) target_.replace_token_sampler(final_sampler_);
            pending_=target_.adopt_prefill_row(session_,tokens_[static_cast<std::size_t>(n_-1U)],view(held_),logits,producer_rotary_,final_selector_);
            adopted_=true;
            phase_=Phase::done;
        }
        report_progress();
    }
    // Runs the next unit of the prompt.
    void step(std::uint32_t serial_rows_per_step=UINT32_MAX) {
        for(;;) {
            if(pos_<align_end_) {run_serial_row();return;}
            if(phase_==Phase::blocks || phase_==Phase::tails) {
                if(const auto block=next_block()) {run_block(*block);return;}
                continue;
            }
            if(phase_==Phase::serial) {
                if(pos_>=n_){phase_=Phase::done;return;}
                if(run_window_rows()) {if(pos_>=n_)phase_=Phase::done;return;}
                for(std::uint32_t k=0;k<serial_rows_per_step && pos_<n_;++k) run_serial_row();
                report_progress();
                if(pos_>=n_)phase_=Phase::done;
                return;
            }
            return;
        }
    }
    HipMtpTargetHiddenTicket finish(std::string& receipt) {
        drain_priming();
        check_yarn_memory();
        if(!done())throw std::logic_error("tiled MTP prompt finished before its last row");
        if(runtime_.next_source_position()+1U!=n_ || session_.next_position!=n_)
            throw std::logic_error("tiled MTP prompt did not reach the final prompt frontier");
        const auto prompt_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start_).count();
        std::ostringstream out;
        // The prompt this receipt belongs to (server gates match receipts to requests): its length and FNV-1a 64 of its tokens.
        std::uint64_t prompt_fnv=14695981039346656037ULL;
        for(const auto t:tokens_) for(unsigned k=0;k<4U;++k){prompt_fnv^=(t>>(8U*k))&0xffU;prompt_fnv*=1099511628211ULL;}
        out<<"\"prompt_tokens\":"<<n_<<",\"prompt_fnv\":\""<<std::hex<<std::setw(16)<<std::setfill('0')<<prompt_fnv<<std::dec<<std::setfill(' ')<<"\",";
        out<<"\"rows\":"<<tiled_.rows<<",\"tail_rows\":[";
        for(std::size_t t=0;t<tiled_.tails.size();++t)out<<(t?",":"")<<tiled_.tails[t].rows;
        out<<"],\"restored_position\":"<<restored_
           <<",\"blocks\":"<<blocks_<<",\"tail_blocks\":"<<tail_blocks_<<",\"serial_rows\":"<<serial_rows_<<",\"tiled_start\":"<<(adopted_?"true":"false")
           <<",\"prompt_ms\":"<<prompt_ms<<",\"grouped_blocks\":"<<grouped_blocks_
           <<",\"block_ms\":"<<block_ms_<<",\"tail_ms\":"<<tail_ms_<<",\"serial_ms\":"<<serial_ms_;
        if(window_rows_) out<<",\"window_rows\":"<<window_rows_;
        // Execution witnesses for gates: pipe-async blocks, SMALL units, grouped tail blocks, window rows run in a cohort.
        out<<",\"pipe_async_blocks\":"<<pipe_async_blocks_<<",\"small_units\":"<<small_units_<<",\"grouped_tail_blocks\":"<<grouped_tail_blocks_
           <<",\"cohort_window_rows\":"<<cohort_window_rows_;
        if(final_logits_digest_) out<<",\"final_logits_digest\":\""<<model_content_identity_hex(*final_logits_digest_)<<'"';
        // STRIX_SERIAL_LOGITS_DIGEST=1: the terminal row ran serially -> its logits digest (gates; block-terminal prompts use the above).
        if(!adopted_ && pending_.valid()) if(const auto d=pending_.serial_logits_digest()) out<<",\"serial_logits_digest\":\""<<model_content_identity_hex(*d)<<'"';
        if(tiled_.cache && tiled_.cache->request().rtx_import) out<<",\"rtx_import\":"<<rtx_import_receipt(tiled_.cache->request());
        if(tiled_.cache && tiled_.cache->has_store()) {
            const auto& q=tiled_.cache->request();
            if(!q.refusal.empty()) out<<",\"cache_refusal\":\""<<q.refusal<<'"';
            out<<",\"prefix_cache\":{\"lookup_ms\":"<<q.lookup_ms<<",\"restore_ms\":"<<q.restore_ms<<",\"restored_bytes\":"<<q.restored_bytes
               <<",\"captured_checkpoints\":"<<q.captured_checkpoints<<",\"captured_segment_bytes\":"<<q.captured_segment_bytes
               <<",\"captured_fixed_bytes\":"<<q.captured_fixed_bytes<<",\"capture_ms\":"<<q.capture_ms
               <<",\"row_bytes_per_position\":"<<tiled_.cache->row_bytes()<<",\"fixed_bytes_per_checkpoint\":"<<tiled_.cache->fixed_bytes()<<'}';
        }
        // STRIX_PREFIX_CACHE_DIGEST=1: the end-of-prompt state digest wherever the component registry exists (a prefix
        // store, --rtx-state-import, STRIX_RTX_EXPORT_STATE or STRIX_PROMPT_STATE_DIGEST).
        static const bool digest=[]{const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST");return v && v[0]=='1';}();
        if(digest) materialize_window();
        if(digest) {
            const auto terminal=adopted_?terminal_logits_digest_:pending_.terminal_logits_digest();
            out<<",\"terminal_logits_digest\":";
            if(terminal) out<<'"'<<model_content_identity_hex(*terminal)<<'"';else out<<"null";
            out<<",\"terminal_metadata\":{\"committed_tokens\":"<<session_.committed_tokens
               <<",\"next_position\":"<<session_.next_position
               <<",\"generations_lineage_digest\":\""<<model_content_identity_hex(terminal_generations_lineage_digest(session_))
               <<"\",\"predictor_frontier\":"<<runtime_.next_source_position()<<'}';
        }
        if(digest && tiled_.cache) out<<",\"state_digest\":\""<<model_content_identity_hex(tiled_.cache->state_digest(n_))<<'"';
        if(digest && batch_ && batch_cert_.enabled) {
            out<<",\"request_order\":"<<request_order_<<",\"omitted_full_t_units\":"<<(tiled_.rows?restored_/tiled_.rows:0);
            out<<",\"batch_cert_decline\":\""<<(!batch_cert_.enabled?"disabled":
                !batch_cert_.config.allowed(plan_config_.schedule)?batch_cert_.config.decline(plan_config_.schedule):
                !certified_request_?(final_selector_?"policy-selection":"state-transfer"):"")<<'"';
            if(tiled_.cache && tiled_.cache->has_store()) {
                const auto counters=tiled_.cache->store().counters();
                out<<",\"capture_counters_global\":{\"tripwire\":"<<counters.capture_skips_tripwire
                   <<",\"batch_config\":"<<counters.capture_skips_batch_config<<",\"cross_stream\":"<<counters.capture_skips_cross_stream
                   <<",\"staging_busy\":"<<counters.capture_busy_skips<<",\"staging_budget\":"<<counters.staging_cap_skips
                   <<",\"snapshot_preparations\":"<<counters.snapshot_preparations
                   <<",\"staging_bytes\":"<<counters.staging_bytes
                   <<",\"no_store\":"<<counters.no_store_skips<<'}';
            }
            if(tiled_.cache && tiled_.cache->continuation()) {
                const auto& cursor=*tiled_.cache->continuation();
                out<<",\"continuation\":{\"phase\":"<<static_cast<unsigned>(cursor.phase)
                   <<",\"tail\":"<<cursor.tail<<",\"balanced\":"<<cursor.balanced<<'}';
            }
            out<<",\"batch_cert_enabled\":"<<(batch_cert_.enabled?"true":"false")
               <<",\"batch_cert_config_allowed\":"<<(batch_cert_.config.allowed(plan_config_.schedule)?"true":"false")
               <<",\"batch_restore_allowed\":"<<(batch_cert_.allow_restore?"true":"false")
               <<",\"execution_trace\":[";
            for(std::size_t k=0;k<trace_.size();++k) {
                const auto& u=trace_[k];
                out<<(k?",":"")<<"{\"begin\":"<<u.begin<<",\"end\":"<<u.end<<",\"rows\":"<<u.rows
                   <<",\"t_owner\":"<<u.t_owner<<",\"own_owner\":"<<u.own_owner<<",\"grouped\":"<<u.grouped
                   <<",\"small\":"<<u.small<<",\"forced_ladder\":"<<u.forced<<",\"adopted\":"<<u.adopted
                   <<",\"exec_class\":"<<u.exec_class()<<'}';
            }
            out<<']';
        }
        receipt=out.str();
        // Final-only diagnostics log the execution witness without reading prompt state.
        static const bool log=[]{const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST");
            const char* f=std::getenv("STRIX_FINAL_STATE_DIGEST");return (v && v[0]=='1') || (f && f[0]=='1');}();
        if(log) std::cerr<<"tiled prompt: {"<<receipt<<"}"<<std::endl;
        return std::move(pending_);
    }
private:
    using Phase=QwenPlanPhase;
    PrefixCaptureOrder capture_order(const Block& block,bool grouped,hipStream_t snapshot_predictor) const {
        if(grouped || small_ || (batch_ && forced_ladder_) || adopted_ || !certified_request_) return {};
        QwenPlanCursor cursor{phase_,tail_,balanced_};
        const auto next=qwen_prompt_next(plan_config_,cursor,session_.next_position,n_,false,small_,adopted_);
        if(!next || !next->t_owner ||
            block.owner!=tiled_.block_owner.get() || next->owner!=plan_config_.block) return {};
        if(batch_) {
            const auto p=session_.next_position;
            const bool nest=qwen_batch_nest_possible(batch_cert_.nesting_possible,batch_cert_.config.nest_rows,tiled_.rows,n_-p);
            if(trace_.empty() || !qwen_batch_capture_eligible(batch_cert_.config,plan_config_.schedule,
                trace_.back(),tiled_.rows,qwen_producer_trace_ok(plan_config_,n_,p,trace_),
                next->t_owner,static_cast<std::uint32_t>(next->end-next->begin),nest)) return {};
        }
        const auto mtp=static_cast<hipStream_t>(runtime_.resources().stream());
        // The next target writes on this actual owner stream. Predictor rows and held_ are
        // next written on mtp. The next target uses the spare tap buffer; drain_priming
        // joins the worker and completes mtp (including its snapshot) before tap reuse or
        // any synchronous/default-stream held copy. No dependency is added for capture.
        return {tiled_.block_owner->producer_stream(),snapshot_predictor,mtp};
    }
    QwenTiledPlanConfig plan_config_;
    bool certified_request_=false,batch_=false;
    bool deferred_lookup_=false;
    BatchCertContext batch_cert_;
    std::vector<QwenUnitExec> trace_;
    std::uint64_t request_order_=0;
    bool legacy_cache() const {return tiled_.cache && tiled_.cache->has_store() && tiled_.cache->store().policy().legacy;}
    static void check(hipError_t e,const char* what){if(e!=hipSuccess)throw std::runtime_error(std::string(what)+": "+hipGetErrorString(e));}
    [[nodiscard]] MtpDeviceView view(const std::byte* p) const {return MtpDeviceView{reinterpret_cast<std::uintptr_t>(p),width_,1U};}
    // STRIX_PREFILL_PIPE (bit mask, default 0): 1 = a T block's predictor rows build on the MTP stream while the next
    // block runs (completed before any other use of the predictor, the held row or the prefix cache);
    // 2 = the tap rows of a block in one strided copy.
    unsigned pipe_flags() const {static const unsigned v=[]{const char* e=std::getenv("STRIX_PREFILL_PIPE");return e?static_cast<unsigned>(std::strtoul(e,nullptr,0)):0U;}();return strix::qpf2k::suppressed(strix::qpf2k::Item::pipe, resources_.model_manifest && strix::qpf2k::has_gsq_gate(*resources_.model_manifest))?0U:v;}
    bool pipe_mtp() const {return (pipe_flags()&1U)!=0U;}
    bool pipe_tap_copy() const {return (pipe_flags()&2U)!=0U;}
    bool pipe_worker() const {return (pipe_flags()&32U)!=0U;}
    void drain_priming() {
        if(!priming_) {PrefixAsyncCapture::reap();return;}
        priming_=false;
        try {
            const bool pending=!submitter_.valid() || submitter_.get();
            if(pending) runtime_.prime_hidden_rows_complete(); // incumbent completion location
            if(capture_) {
                capture_->commit(!runtime_.prime_hidden_rows_pending() &&
                    runtime_.next_source_position()+1U==capture_->checkpoint->position &&
                    session_.health==SerialSessionHealth::ready);
                capture_.reset();
            }
            PrefixAsyncCapture::reap();
        } catch(...) {
            if(capture_ && capture_->quarantine) capture_->fatal();
            throw;
        }
        report_progress();
    }
public:
    std::function<void(HipKnownPrefixWindow&,const std::function<void()>&)> execute_native;
    void drain_for_handoff(){drain_priming();}
    void set_request_order(std::uint64_t value) {request_order_=value;}
    void before_writer(hipStream_t stream) const {if(batch_cert_.enabled) PrefixAsyncCapture::before_writer(this,stream);}
private:
    void run_block(const Block& block) {
        check_yarn_memory();
        auto& owner=*block.owner;
        before_writer(owner.producer_stream());
        // The tap observer reads the tap step only (STRIX_QPF2K_HC prefill folds stay on for the other steps).
        owner.set_prefill_step_observer(block_observer(block.rows),[tap=tap_](std::size_t step){return step==tap;});
        const auto t0=std::chrono::steady_clock::now();
        try {
            std::vector<float> logits;
            const auto launch=[&] {
            // STRIX_QPF2K_SCHED bit 4: a balanced block shorter than its backend's T runs as a partial block.
            if(block.rows!=block.backend->logical_rows()) {
                if(block.final) logits=block.backend->execute_rows_partial(block_tokens(block),pos_);
                else block.backend->execute_state_rows_partial(block_tokens(block),pos_,false);
            } else if(block.final) logits=block.backend->execute_rows(block_tokens(block),pos_);
            else block.backend->execute_state_rows(block_tokens(block),pos_);
            };
            if(execute_native) execute_native(owner,launch);else launch();
            owner.set_prefill_step_observer({});
            check_yarn_memory();
            finish_block(block,logits);
        } catch(...) {owner.set_prefill_step_observer({});throw;}
        (block.capture?block_ms_:tail_ms_)+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    }
    // STRIX_PROMPT_WINDOW (default on; =0 runs the suffix serially as before): the serial suffix (fewer than 16 rows after the tiled blocks) runs as
    // known-row windows of 8, 4 and 2 rows on the request's window owners instead of one row at a time: an odd row
    // first runs serially, then the windows, each committing all its rows (a window row's arithmetic is the serial
    // row's, so the state, the predictor rows and the first token are the serial suffix's). Greedy requests only;
    // false (nothing ran) when the suffix has fewer than two rows or no window admits it.
    static bool prompt_window_enabled() {
        static const bool on=[]{const char* v=std::getenv("STRIX_PROMPT_WINDOW");return !v || v[0]!='0';}();
        return on;
    }
    bool run_window_rows() {
        check_yarn_memory();
        bool ran=false;
        for(std::uint32_t rows;(rows=window_unit())!=0U;) {
            before_writer(nullptr); // unknown foreign stream is conservatively foreign
            check_yarn_memory();
            auto tickets=tiled_.window->prompt_rows(session_,window_tokens(rows));
            check_yarn_memory();
            if(tickets.empty()){window_refused_=true;break;}
            finish_window(std::move(tickets));ran=true;
        }
        return ran;
    }
public:
    // The rows of the serial suffix's next known-row window (8, 4 or 2), after running an odd remaining row
    // serially; 0 when the prompt's next unit is no such window (STRIX_PROMPT_WINDOW off, a block, a sampled
    // request, fewer than two rows, or a window was refused). Only advances past exhausted block phases.
    [[nodiscard]] std::uint32_t window_unit() {
        if(!prompt_window_enabled() || !tiled_.window || window_refused_ || final_sampler_ || target_.token_sampler()) return 0U;
        if((phase_==Phase::blocks || phase_==Phase::tails) && next_block()) return 0U;
        // Reserve the terminal row for host selection, including prompt cohorts.
        const auto end=n_-(final_selector_?1U:0U);
        if(phase_!=Phase::serial || end-pos_<2U) return 0U;
        // A window extends the session's arithmetic lineage: the first prompt row always runs serially.
        if(!pos_) {run_serial_row();report_progress();if(end-pos_<2U) return 0U;}
        if((end-pos_)%2U) {run_serial_row();report_progress();}
        std::uint32_t rows=8U;
        while(rows>end-pos_) rows/=2U;
        return rows;
    }
    [[nodiscard]] std::span<const std::uint32_t> window_tokens(std::uint32_t rows) const {
        return std::span<const std::uint32_t>(tokens_).subspan(static_cast<std::size_t>(pos_),rows);
    }
    [[nodiscard]] HipMtpWindowVerifier* window_verifier() const noexcept {return tiled_.window;}
    [[nodiscard]] SerialSessionContext& session() const noexcept {return session_;}
    // After the window ran (alone or in a cohort): the predictor rows (h_{p-1}, t_p) .. (h_{p+r-2}, t_{p+r-1}), the
    // held row and the pending ticket, as run_serial_row leaves them after the same rows; empty: the window was
    // refused (nothing ran) and the suffix continues serially.
    void finish_window(std::vector<HipMtpTargetHiddenTicket> tickets,bool cohort=false) {
        if(tickets.empty()){window_refused_=true;return;}
        if(batch_ && batch_cert_.enabled)
            trace_.push_back({pos_,pos_+tickets.size(),static_cast<std::uint32_t>(tickets.size()),false,false,cohort,small_,forced_ladder_,adopted_});
        if(cohort)cohort_window_rows_+=tickets.size();
        drain_priming();   // STRIX_PREFILL_PIPE: the last block's predictor rows and held row precede the window's
        const auto rows=static_cast<std::uint32_t>(tickets.size());
        std::vector<MtpDeviceView> views;std::vector<std::uint32_t> next;
        if(pos_){views.push_back(view(held_));next.push_back(tokens_[static_cast<std::size_t>(pos_)]);}
        for(std::uint32_t r=0;r+1U<rows;++r){views.push_back(tickets[r].view());next.push_back(tokens_[static_cast<std::size_t>(pos_+r+1U)]);}
        runtime_.prime_hidden_rows(qualified_,activated_,views,next,producer_rotary_);
        check(hipMemcpy(held_,reinterpret_cast<const void*>(tickets.back().view().address),static_cast<std::size_t>(width_),
            hipMemcpyDeviceToDevice),"tiled MTP held window copy");
        pending_=std::move(tickets.back());pos_+=rows;window_rows_+=rows;
        window_pending_=true;
        // after_unit copies physical state only at the selected export position.
        const auto& export_request=rtx_state_export_request();
        if(tiled_.cache && export_request && pos_==(export_request->position?export_request->position:n_))
            materialize_window();
        if(tiled_.cache) tiled_.cache->after_unit(session_,tokens_,pos_);
        if(pos_>=n_)phase_=Phase::done;
        report_progress();
    }
private:
    void materialize_window() {
        if(!window_pending_) return;
        // Standalone commit replays the same ordered float recurrence and BF16
        // history shifts as the next window, then clears its shared pending rows.
        // Certification off: the incumbent unconditional flush (owner sync + persistent-pool invalidation).
        if(!batch_cert_.enabled) tiled_.window->flush_pending_window();
        else if(tiled_.window->pending_window()) {before_writer(nullptr);tiled_.window->flush_pending_window();}
        window_pending_=false;
    }
    void report_progress() const {
        if(progress_) progress_({serve::GenerationPhase::prefill,pos_-restored_,restored_,n_});
    }
    void run_serial_row() {
        before_writer(nullptr);
        if(batch_ && batch_cert_.enabled)
            trace_.push_back({pos_,pos_+1U,1,false,false,false,small_,forced_ladder_,adopted_});
        check_yarn_memory();
        drain_priming();
        // Every serial row consumes recurrent/conv state, including fallback
        // after a refused window. Fold earlier rows before advancing it.
        materialize_window();
        const auto t0=std::chrono::steady_clock::now();
        if(final_sampler_ && pos_+1U==n_) target_.replace_token_sampler(final_sampler_);
        auto ticket=target_.execute_serial(session_,resources_,dispatcher_,tokens_[static_cast<std::size_t>(pos_)],nullptr,
            pos_+1U==n_?final_selector_:HipMtpHostRowSelector{});
        if(pos_) {
            const std::array<MtpDeviceView,1> one{view(held_)};const std::array<std::uint32_t,1> token{tokens_[static_cast<std::size_t>(pos_)]};
            runtime_.prime_hidden_rows(qualified_,activated_,one,token,producer_rotary_);
        }
        check(hipMemcpy(held_,reinterpret_cast<const void*>(ticket.view().address),static_cast<std::size_t>(width_),hipMemcpyDeviceToDevice),
            "tiled MTP held ticket copy");
        check_yarn_memory();
        pending_=std::move(ticket);++pos_;++serial_rows_;
        if(tiled_.cache) tiled_.cache->after_unit(session_,tokens_,pos_);
        serial_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    }
    std::uint64_t align_end_=0;
    std::vector<std::uint32_t> tokens_;
    MtpDecodeTargetParents::TiledPrompt& tiled_;
    SerialSessionContext& session_;
    RotaryIdentity producer_rotary_;
    HipMtpTargetHidden& target_;
    HipMtpRuntime& runtime_;
    const QualifiedMtpDraftPlan& qualified_;
    const ActivatedMtpDraftPlan& activated_;
    const ExecutionRuntimeResources& resources_;
    DirectLaunchDispatcher& dispatcher_;
    std::byte* held_=nullptr;
    std::shared_ptr<TokenSampler> final_sampler_;
    HipMtpHostRowSelector final_selector_;
    std::function<void(serve::GenerationProgress)> progress_;
    bool window_pending_=false;
    std::uint64_t n_=0,width_=0,restored_=0,pos_=0,blocks_=0,tail_blocks_=0,serial_rows_=0,grouped_blocks_=0,window_rows_=0,
        pipe_async_blocks_=0,small_units_=0,grouped_tail_blocks_=0,cohort_window_rows_=0;
    double block_ms_=0.0,tail_ms_=0.0,serial_ms_=0.0;  // wall time of the T blocks, the tail blocks, the serial rows (not the windows)
    bool forced_ladder_=false;
    bool small_=false;
    std::optional<ModelContentIdentity> final_logits_digest_;
    std::optional<ModelContentIdentity> terminal_logits_digest_;
    std::size_t tap_=SIZE_MAX,tail_=0;
    std::byte* block_hidden_=nullptr;
    std::byte* spare_hidden_=nullptr;
    std::byte* hidden_allocation_=nullptr;
    Phase phase_=Phase::blocks;
    bool adopted_=false,window_refused_=false,priming_=false;
    int balanced_=-1;
    HipPrefixStateCacheSession::Capture capture_;
    std::future<bool> submitter_;
    HipMtpTargetHiddenTicket pending_;
    std::chrono::steady_clock::time_point start_;
};

// --mtp-copy-wide-rows N[,M]: the wide copy verify over the tiled tail owner of exactly each listed size (descending;
// null when none). Each needs the all-rows output policy and the decode catalog: no split prefill profile hands the
// session to another execution.
inline std::unique_ptr<MtpWideCopyCycle> make_mtp_wide_copy(const MtpDecodeTargetParents& parents,
    std::span<const std::uint32_t> sizes,HipMtpTargetHidden& target) {
    if(sizes.empty()) return {};
    const auto& tiled=parents.tiled;
    if(!tiled.blocks || parents.prompt.compiled || (tiled.prefill_execution_identity && tiled.wide.size()!=sizes.size()))
        throw std::invalid_argument("--mtp-copy-wide-rows requires the tiled MTP prompt (with a split prefill numerical "
            "profile: its decode-profile wide owners)");
    // Split profile: the decode-profile wide owners; one profile: the tails of the same sizes.
    const auto& candidates=tiled.prefill_execution_identity?tiled.wide:tiled.tails;
    std::vector<std::shared_ptr<HipKnownPrefixWindow>> owners;
    for(const auto rows:sizes) {
        const auto tail=std::find_if(candidates.begin(),candidates.end(),[&](const auto& t){return t.rows==rows && t.owner;});
        if(tail==candidates.end())
            throw std::invalid_argument("--mtp-copy-wide-rows requires a tiled tail block of each listed size (--known-prefix-tail-rows)");
        if(tail->owner->output_policy()!=KnownWindowOutputPolicy::all_rows || !tail->owner->supports_tiled_prefill() ||
           tail->owner->logical_rows()!=rows)
            throw std::invalid_argument("--mtp-copy-wide-rows requires its tiled tail owners with the all-rows output policy");
        owners.push_back(tail->owner);
    }
    return std::make_unique<MtpWideCopyCycle>(std::move(owners),target,parents.resources);
}

// Tiled MTP prompt after the calibration replay, in one call (see TiledMtpPrompt).
inline HipMtpTargetHiddenTicket run_tiled_mtp_prompt(std::span<const std::uint32_t> tokens,const ModelContract& model,
    const CompiledSerialExecution& compiled,MtpDecodeTargetParents::TiledPrompt& tiled,SerialSessionContext& session,
    HipMtpTargetHidden& target,HipMtpRuntime& runtime,const QualifiedMtpDraftPlan& qualified,const ActivatedMtpDraftPlan& activated,
    const ExecutionRuntimeResources& resources,DirectLaunchDispatcher& dispatcher,std::byte* held,std::string& receipt,
    std::shared_ptr<TokenSampler> final_sampler={},HipMtpHostRowSelector final_selector={},
    std::function<void(serve::GenerationProgress)> progress={}) {
    TiledMtpPrompt prompt(tokens,model,compiled,tiled,session,target,runtime,qualified,activated,resources,dispatcher,held,
        std::move(final_sampler),std::move(final_selector),std::move(progress));
    while(!prompt.done()) prompt.step();
    return prompt.finish(receipt);
}

// Batched decoding of independent requests on the tiled MTP route (--http-batch, cohort requests with the
// tiled prompt). Every slot owns what a single request owns: target state and session, launch plan, PLE
// backend, tiled prefill backends (the prefill arena is shared and used by one prompt at a time), prefix
// cache view, predictor, target producer and 2/4/8 window owners on one shared window stream. A request is
// admitted into a free slot between cycles: its prompt runs exactly as a single request's (fresh state,
// tiled blocks, the first token), then it joins the cycles of the others; it leaves when it finishes,
// without stopping them. With two or more active requests a cycle is a cohort cycle
// (hip_mtp_greedy_cohort_cycle); with one, the ordinary cycle.
struct MtpBatchRequest {
    std::shared_ptr<MtpConstraintRow> constraint;
    serve::CacheRetention cache_retention = serve::CacheRetention::server_policy;
    std::vector<std::uint32_t> prompt;
    std::uint64_t max_tokens = 0;
    std::vector<std::uint32_t> stop_tokens;
    std::shared_ptr<TokenSampler> sampler; // null: first-max selection
    // Request field "ngram_draft": true/false forces this request's context draft source on/off (greedy only);
    // absent: the engine's batch setting (STRIX_BATCH_NGRAM, off by default).
    std::optional<bool> ngram_draft;
    std::uint32_t draft_depth_max = 0; // request field "draft_depth_max": predictor drafts per cycle at most (0: engine)
    std::optional<bool> tool_draft;    // request field "tool_draft": strict tool-call frames (unset: STRIX_BATCH_TOOL_DRAFT)
    std::optional<bool> ngram_peers;   // request field "ngram_peers": cross-request lookup (unset: STRIX_BATCH_NGRAM_PEERS)
    // Newly verified tokens (the first call carries the prompt's own prediction); false cancels.
    std::function<bool(std::span<const std::uint32_t>)> on_tokens;
    std::function<void(serve::GenerationProgress)> on_progress;
    std::function<void(const MtpServeResult&)> on_finish;
    std::function<bool()> cancelled;
};
struct MtpBatchSlotSetup {
    std::shared_ptr<MtpDecodeTargetParents> parents; // parents->tiled bound to this slot's state and session
    SerialSessionContext* session = nullptr;
    DirectLaunchDispatcher* dispatcher = nullptr;
    MtpDecodeWindowFactory make_window;
};
class MtpBatchRunner {
    struct EngineCall {
        bool& active;
        explicit EngineCall(bool& a):active(a) {if(active) throw std::logic_error("batch callback re-entry");active=true;}
        ~EngineCall(){active=false;}
    };
public:
    std::mutex* session_mutex = nullptr;
    MtpServeRunner session_run; // Only invoked between ordinary batch jobs, on the bridge thread.
    std::function<void()> publish_session_stats; // Bridge thread, under session_mutex.
    struct Slot {
        std::shared_ptr<MtpDecodeTargetParents> parents;
        SerialSessionContext* session = nullptr;
        DirectLaunchDispatcher* dispatcher = nullptr;
        std::shared_ptr<HipMtpResources> native;
        std::shared_ptr<HipMtpRuntime> runtime;
        std::shared_ptr<HipMtpTargetHidden> target;
        std::shared_ptr<HipMtpWindowVerifier> window;
        std::byte* held = nullptr; // h_{P-1} of the tiled prompt
        std::unique_ptr<HipMtpGreedyRuntime> greedy;
        HipMtpTargetHiddenTicket pending;
        OutputLogitsChain output_chain;
        bool active = false;     // the slot holds a request (prefilling or decoding)
        bool prefilling = false; // its prompt is still running (interleaved admission)
        std::unique_ptr<TiledMtpPrompt> prefill;
        Slot* waits_for=nullptr;
        QwenPrefixJoinWait join_wait;
        std::uint64_t order = 0; // admission order: prompts advance oldest first
        std::chrono::steady_clock::time_point prompt_start;
        MtpBatchRequest request;
        MtpServeResult result;
        std::string prompt_receipt;
        std::chrono::steady_clock::time_point decode_start;
        std::array<std::uint64_t, 9> chosen_rows{};
        std::uint64_t draft_rows_begin=0;
    };
    struct Stats : BatchServiceStats {
        std::uint64_t admitted = 0, finished = 0, steps = 0, cohort_steps = 0, solo_steps = 0, max_active = 0, prompt_steps = 0;
        // Prompt groups (STRIX_PROMPT_BATCH=1): segmented runs of several prompts' blocks, their blocks and rows.
        std::uint64_t prompt_groups = 0, grouped_blocks = 0, grouped_rows = 0;
        double prompt_group_ms = 0.0;
        std::uint64_t lazy_slot_builds = 0;
        std::uint64_t inflight_waits=0,inflight_wait_tokens=0,inflight_restored_tokens=0;
        double lazy_slot_ms = 0.0, lazy_slot_max_ms = 0.0;
        bool lazy_slot_memory_declined = false;
        // Prompt window cohorts (STRIX_PROMPT_WINDOW_COHORT=1): steps that ran known-row windows, and their windows.
        std::uint64_t prompt_window_cohorts = 0, prompt_window_members = 0;
        HipMtpGreedyCohortStats cohort;
    };
    // The calibration replay that qualified the predictor for every slot (set by the bridge).
    struct Calibration {
        std::uint64_t rows = 0;
        double ms = 0.0, maximum_hidden_abs_error = 0.0, maximum_logit_abs_error = 0.0;
    } calibration;
    using EnableDraft = std::function<void(HipMtpRuntime&, std::span<const std::uint32_t>, const std::shared_ptr<TokenSampler>&)>;
    // A decoding request's context draft source (prompt lookup), set before its first cycle: the request's
    // greedy runtime, prompt, sampler and per-request choice. Null: never.
    using EnableContext = std::function<void(HipMtpGreedyRuntime&, std::span<const std::uint32_t>, const std::shared_ptr<TokenSampler>&,
        std::optional<bool>)>;
    EnableContext enable_context;
    // STRIX_BATCH_NGRAM_PEERS=1 (opt-in): each decoding request's n-gram extension also looks up the committed
    // contexts of the other decoding requests (joint speculation across concurrent conversations).
    bool context_peers = [] { const char* v = std::getenv("STRIX_BATCH_NGRAM_PEERS"); return v && v[0] == '1'; }();
    // Strict tool-call frame proposals (strix/mtp_frame_draft.hpp): the template's <tool_call> / </tool_call> token
    // ids (set by the server from its tokenizer; {0, 0}: unavailable) and the engine default: on (exact, the target
    // verifies every row; not slower on the tool-call traces with the committed-prefix gate), STRIX_BATCH_TOOL_DRAFT=0
    // turns it off.
    std::array<std::uint32_t, 2> frame_tokens{};
    bool tool_draft_default = [] { const char* v = std::getenv("STRIX_BATCH_TOOL_DRAFT"); return !(v && v[0] == '0'); }();
    // interleave: a joining request's prompt advances one unit (a tiled block, a tail block or
    // `serial_rows_per_step` serial rows) per step, between the decode cycles of the others; otherwise
    // admit() runs the whole prompt.
    MtpBatchRunner(const Options& options, const ModelContract& model, const CompiledSerialExecution& compiled,
        const QualifiedMtpDraftPlan& qualified, const ActivatedMtpDraftPlan& activated, EnableDraft enable_draft,
        std::vector<std::unique_ptr<Slot>> slots, bool interleave = false, std::uint32_t serial_rows_per_step = 16)
        : options_(options), model_(model), compiled_(compiled), qualified_(qualified), activated_(activated),
          enable_draft_(std::move(enable_draft)), slots_(std::move(slots)), interleave_(interleave),
          serial_rows_per_step_(serial_rows_per_step ? serial_rows_per_step : 1U),joint_(slots_.size(),MtpBatchWindowController::rule_from_env(options.baseline)),policy_joint_(slots_.size(),MtpBatchWindowController::rule_from_env(options.baseline)) {
        stats_.nest_rows_limit=nest_rows_;
        if (slots_.empty()) throw std::invalid_argument("batched MTP decoding needs a slot");
        if(service_ms_>0.0 || cancel_points_ || nest_rows_) {
            check_service(hipGetDevice(&device_),"capture prefill worker device");
            worker_=std::make_unique<BatchServiceHandoff>();
            std::fprintf(stderr,"batch service ms %.3f lookahead %u cancel_points %u nest_rows %llu (native blocks only)\n",service_ms_,lookahead_,cancel_points_?1U:0U,static_cast<unsigned long long>(nest_rows_));
        }

    }
    MtpBatchRunner(const MtpBatchRunner&) = delete;
    MtpBatchRunner& operator=(const MtpBatchRunner&) = delete;
    [[nodiscard]] bool supports_thinking_budget() const noexcept {return options_.http_thinking_budget;}
    [[nodiscard]] bool supports_constraints() const noexcept {return options_.http_constraints;}
    [[nodiscard]] std::vector<std::string> constraint_execution_routes() const {return options_.http_selection_mtp &&
        std::any_of(slots_.begin(),slots_.end(),[](const auto& s){return s->window && s->window->policy_selection_admitted();})?
        std::vector<std::string>{"mtp-host","serial-host"}:std::vector<std::string>{"serial-host"};}
    std::function<std::unique_ptr<Slot>()> grow_factory;
    std::function<bool()> growth_memory_ok;
    // STRIX_HTTP_SPARE_SLOTS (default 1; 0 = grow only for queued work): keep one built slot free while capacity remains.
    bool spare_slot_=[]{const char* v=std::getenv("STRIX_HTTP_SPARE_SLOTS");return !v || v[0]!='0';}();
    std::size_t slots_capacity = 0;
    // Called only by the engine loop, between steps and outside service callbacks.
    bool add_slot(bool queued) {
        EngineCall call(engine_call_);
        if (stats_.lazy_slot_memory_declined || !grow_factory ||
            !http_slot_should_grow(queued, active(), slots(), slots_capacity, true, spare_slot_)) return false;
        if (paused_ || paused_owner_) throw std::logic_error("batch growth during service park");
        for (auto& slot : slots_) if (slot->prefilling) slot->prefill->drain_for_handoff();
        if (!growth_memory_ok || !growth_memory_ok()) {
            stats_.lazy_slot_memory_declined = true;
            std::fprintf(stderr,"http slot growth stopped: insufficient MemAvailable/GTT with headroom\n");
            return false;
        }
        const auto start = std::chrono::steady_clock::now();
        auto slot = grow_factory();
        if (!slot) throw std::logic_error("batch growth factory returned no slot");
        joint_.resize_slots(slots_.size() + 1U);policy_joint_.resize_slots(slots_.size()+1U);
        slots_.push_back(std::move(slot));
        const auto ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        ++stats_.lazy_slot_builds;
        stats_.lazy_slot_ms += ms;
        stats_.lazy_slot_max_ms = std::max(stats_.lazy_slot_max_ms, ms);
        return true;
    }
    [[nodiscard]] std::size_t slots() const noexcept { return slots_.size(); }
    [[nodiscard]] std::size_t active() const noexcept {
        return static_cast<std::size_t>(std::count_if(slots_.begin(), slots_.end(), [](const auto& s) { return s->active; }));
    }
    [[nodiscard]] bool has_free_slot() const noexcept { return active() < slots_.size(); }
    [[nodiscard]] bool interleaved() const noexcept { return interleave_; }
    [[nodiscard]] static bool prompt_batching() noexcept { return prompt_batch_enabled(); }
    [[nodiscard]] static std::uint32_t prompt_group_rows() { return prompt_batch_rows(); }
    [[nodiscard]] static bool prompt_window_cohort() { return prompt_window_cohort_enabled(); }
    [[nodiscard]] std::size_t prefilling() const noexcept {
        return static_cast<std::size_t>(std::count_if(slots_.begin(), slots_.end(), [](const auto& s) { return s->prefilling; }));
    }
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    [[nodiscard]] const Slot& slot(std::size_t index) const { return *slots_.at(index); }

    // Engine-thread hook lifetime belongs to the HTTP loop; queued is a thread-safe read-only hint.
    using AdmissionSink=std::function<void(MtpBatchRequest)>;
    using AdmissionPump=std::function<void(const AdmissionSink&,const std::function<bool()>&)>;
    void set_service_admission(std::function<bool()> queued,AdmissionPump pump) {
        has_queued_work_=std::move(queued);admission_pump_=std::move(pump);
    }
    // Runs or prepares the prompt in a free slot, retaining the public callback re-entry guard.
    std::size_t admit(MtpBatchRequest request) {
        EngineCall call(engine_call_);
        const auto index=prepare_admission(std::move(request));
        auto& s=*slots_[index];
        if (!interleave_) {
            try {
                while (!s.prefill->done() && !cancel_requested(s)) s.prefill->step();
                if(cancel_requested(s)) cancel_prefill(s,false);else complete_prompt(s);
            } catch(const KnownPrefillCancelled&) {cancel_prefill(s,true);}
            catch(const serve::ConstraintError& e) {constraint_failed(s,e);}
        }
        return index;
    }
private:
    BatchCertContext batch_cert_context(const Slot& s) const {
        BatchCertContext result;
        result.enabled=prefix_batch_cert_execution_mode(options_) && !options_.baseline;
        if(!result.enabled) return result;
        result.allow_restore=prefix_batch_cert_execution_mode(options_)==1 && !service_mode_;
        auto& c=result.config;
        c.groups=prompt_batch_enabled();c.group_rows=prompt_batch_rows();
        c.group_owner=bool(slots_.front()->parents->tiled.group);
        c.small=batch_prompt_small(options_);
        const char* windows=std::getenv("STRIX_PROMPT_WINDOW");c.windows=(!windows || windows[0]!='0') && bool(s.window);
        c.cohort=prompt_window_cohort_enabled();c.service=service_ms_>0;c.cancel=cancel_points_;c.nest_rows=nest_rows_;
        c.service_ms=service_ms_;c.lookahead=lookahead_;c.prompt_share=prompt_share_;c.interleave=interleave_;
        const char* short_rows=std::getenv("STRIX_BATCH_SHORT_FIRST_ROWS");
        c.short_rows=short_rows?std::strtoull(short_rows,nullptr,10):2048ULL;
        const char* shared=std::getenv("STRIX_PROMPT_BATCH_SHARED");
        c.shared=shared?shared:"hc,router,moe,elementwise";
        // Conservative row-class bound: only the 8192-row family can be nested.
        result.nesting_possible=qwen_batch_nest_possible(bool(worker_),nest_rows_,s.parents->tiled.rows,1);
        return result;
    }
    std::size_t prepare_admission(MtpBatchRequest request) {
        if(!engine_call_ || (paused_owner_ && !service_mode_)) throw std::logic_error("invalid admission mode");
        const auto free = std::find_if(slots_.begin(), slots_.end(), [](const auto& s) { return !s->active; });
        if (free == slots_.end()) throw std::logic_error("batched MTP admission without a free slot");
        const auto vocabulary = model_.vocab_size;
        if (request.prompt.empty() || request.prompt.size() >= options_.context_tokens || !request.max_tokens || !request.on_tokens)
            throw std::invalid_argument("batched request needs a prompt shorter than the context, a token budget and a sink");
        for (const auto t : request.prompt) if (t >= vocabulary) throw std::invalid_argument("batched prompt token outside the vocabulary");
        for (const auto t : request.stop_tokens) if (t >= vocabulary) throw std::invalid_argument("batched stop token outside the vocabulary");
        auto& s = **free;
        const auto& m = options_.mtp;
        const auto start = std::chrono::steady_clock::now();
        s.request = std::move(request);
        if(s.request.constraint) {const auto route=resolve_mtp_selection_route(options_.http_selection_mtp,!!s.window,s.window && s.window->policy_selection_admitted());
            s.request.constraint->set_execution(route.mtp?"mtp-host":"serial-host",route.fallback?route.fallback:"");}
        if (s.request.ngram_peers) any_peers_ = true;
        MtpServeResult fresh_result;
        s.result=std::move(fresh_result);s.result.service.nest_rows_limit=nest_rows_;
        s.chosen_rows = {};
        joint_.reset(static_cast<std::uint32_t>(free-slots_.begin()));policy_joint_.reset(static_cast<std::uint32_t>(free-slots_.begin()));
        // STRIX_MTP_JOINT_RULE_FILE (measurement, in-process A/B): the joint plan rule is re-read from this file at
        // every admission (its text as STRIX_MTP_JOINT_PLAN; missing/empty file = the startup rule). The file's
        // "wave=N" token names the wave; on the first admission of a new wave, "snapshot" saves the controller's
        // shared state and "restore" restores the saved state (applied once per wave, never mid-wave).
        std::string_view restored = "-";
        if (const char* path = std::getenv("STRIX_MTP_JOINT_RULE_FILE")) {
            std::string text;
            if (std::FILE* f = std::fopen(path, "r")) {
                char buffer[128]{};
                const auto n = std::fread(buffer, 1, sizeof(buffer) - 1U, f);
                std::fclose(f);
                text.assign(buffer, n);
            }
            joint_.set_rule(text.empty() ? MtpBatchWindowController::rule_from_env(options_.baseline) : MtpBatchWindowController::parse_rule(text));
            policy_joint_.set_rule(joint_.rule());
            const auto at = text.find("wave=");
            const std::string wave = at == std::string::npos ? std::string{} : text.substr(at, text.find_first_of(" \n", at) - at);
            if (!wave.empty() && wave != rule_wave_) {
                rule_wave_ = wave;
                if (text.find("snapshot") != std::string::npos) { joint_.save_shared(); restored = "saved"; }
                if (text.find("restore") != std::string::npos) restored = joint_.restore_shared() ? "restored" : "nosnapshot";
            }
        }
        // STRIX_MTP_BATCH_LOG=2 (or =a: admissions only, no per-cycle lines): the admission that resets this
        // slot's planner history (request identity, applied rule, controller state event).
        static const bool admit_log = [] { const char* v = std::getenv("STRIX_MTP_BATCH_LOG"); return v && (v[0] == '2' || v[0] == 'a'); }();
        if (admit_log)
            std::fprintf(stderr, "joint admit rule=%d%d%u state=%.*s slot=%td request=%llu prompt_tokens=%zu wall_ms=%lld cal_min_members=%zu\n", joint_.rule().pooled ? 1 : 0,
                joint_.rule().calibrated ? 1 : 0, joint_.rule().cap, static_cast<int>(restored.size()), restored.data(),
                free - slots_.begin(),
                static_cast<unsigned long long>(++admitted_), s.request.prompt.size(),
                static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()), joint_.rule().calibrated_min_members);
        // Exactly the single request's start: fresh target state and session, fresh predictor, no
        // deferred window rows, the request's sampler on the final prompt row and every later row.
        s.greedy.reset();
        s.output_chain={};
        s.pending = {};
        s.runtime = std::make_shared<HipMtpRuntime>(s.native);
        if(s.parents->resources.qsa_committed_pool) s.parents->resources.qsa_committed_pool->begin_epoch();
        s.parents->tiled.reset_state();
        *s.session = s.parents->tiled.fresh;
        if (s.window) s.window->begin_session();
        s.target->replace_token_sampler({});
        // Predictor priming and cache checkpoints use the configured draft identity for every prompt.
        enable_draft_(*s.runtime, s.request.prompt, s.request.sampler);
        s.prompt_start = start;
        s.waits_for=nullptr;s.join_wait={};
        if (s.parents->tiled.cache) s.parents->tiled.cache->set_retention(s.request.cache_retention != serve::CacheRetention::no_store);
        s.prefill = std::make_unique<TiledMtpPrompt>(s.request.prompt, model_, compiled_, s.parents->tiled, *s.session, *s.target,
            *s.runtime, qualified_, activated_, s.parents->resources, *s.dispatcher, s.held, s.request.sampler,
            s.request.constraint?HipMtpHostRowSelector{[state=s.request.constraint](auto row){return state->select(row);}}:HipMtpHostRowSelector{},
            s.request.on_progress,false,true,true,batch_cert_context(s),
            options_.prefix_cache_inflight && interleave_ && !service_mode_);
        static_cast<void>(m);
        ++stats_.admitted;
        s.order = stats_.admitted;
        s.prefill->set_request_order(s.order);
        s.active = true;
        s.prefilling = true;
        if(options_.prefix_cache_inflight && interleave_ && !service_mode_ && !s.request.constraint &&
            s.prefill->lookup_deferred()) wait_for_shared_prefix(s);
        if(!s.waits_for && s.prefill->lookup_deferred()) s.prefill->retry_join_lookup();
        stats_.max_active = std::max<std::uint64_t>(stats_.max_active, active());
        if(worker_) s.prefill->execute_native=[this,&s](HipKnownPrefixWindow& owner,const std::function<void()>& job){
            if(service_mode_) {
                if(!leaf_ || &s!=leaf_ || !paused_owner_ || paused_owner_->aliases_prefill_storage(owner) || current_hip_launch_context())
                    throw std::logic_error("nested prefill lease violation");
                owner.finish_pending_completion();
                job();
                if(current_hip_launch_context()) throw std::logic_error("nested prefill retained launch scope");
            } else execute_prefill(owner,job,&s);
        };
        return static_cast<std::size_t>(free-slots_.begin());
    }
public:

    // Interleaved admission: one unit of the oldest running prompt, when the decode cycles have had at least
    // `prompt_share` times the time of the previous units (STRIX_BATCH_PROMPT_SHARE, default 0: one unit per
    // step; always when nothing decodes). Then one decode cycle
    // of every request past its prompt: a cohort cycle with two or more, the ordinary cycle with one.
    // Only a pending probe uses completion tracking. The default specialization
    // has no additional tracking or health calls in its timing scopes.
    template<bool track_completion = false>
    auto step() {
        EngineCall call(engine_call_);
        [[maybe_unused]] bool serial_completed = false;
        PrefixAsyncCapture::reap();
        using clock = std::chrono::steady_clock;
        Slot* prompt = nullptr;
        bool decoding = false;
        for(auto& owner:slots_) {
            auto& s=*owner;
            if(!s.prefilling || !s.waits_for) continue;
            if(cancel_requested(s)) {cancel_prefill(s,false);continue;}
            const auto& producer=*s.waits_for;
            if(s.join_wait.pending(producer.order,producer.active,producer.prefilling,
                producer.prefill?producer.prefill->position():0)) continue;
            s.waits_for=nullptr;s.join_wait={};
            PrefixAsyncCapture::reap();
            s.prefill->retry_join_lookup();
            stats_.inflight_restored_tokens+=s.prefill->restored();
        }
        // STRIX_BATCH_SHORT_FIRST_ROWS (default 2048; 0 = oldest prompt first): a prompt with at most that many rows left runs
        // its units before longer prompts (oldest such prompt first), so a short request waits for the current unit of a
        // long prompt, not for the rest of it. Exact: every prompt's units, rows and bits are its own whatever the order.
        static const std::uint64_t short_rows=[]{const char* v=std::getenv("STRIX_BATCH_SHORT_FIRST_ROWS");return v?std::strtoull(v,nullptr,10):2048ULL;}();
        const auto ahead=[&](const Slot& a,const Slot& b) {
            const bool sa=short_rows && a.prefill->remaining()<=short_rows,sb=short_rows && b.prefill->remaining()<=short_rows;
            return sa!=sb?sa:a.order<b.order;
        };
        for (auto& s : slots_) {
            if (s->prefilling && !s->waits_for && (!prompt || ahead(*s, *prompt))) prompt = s.get();
            if (s->active && !s->prefilling) decoding = true;
        }
        if (!prompt) prompt_debt_ms_ = 0.0;
        if (prompt && (!decoding || prompt_debt_ms_ <= 0.0)) {
            ++stats_.prompt_steps;
            // STRIX_BATCH_PROMPT_SMALL=1: T blocks only while nobody decodes; tail-size units in between decode cycles
            // (exact either way: the prompt's rows and bits do not depend on the block sizes)
            const bool small=batch_prompt_small(options_);
            if (small) for (auto& s : slots_) if (s->prefilling) s->prefill->set_small_blocks(decoding);
            const auto t0 = clock::now();const auto suspended_before=stats_.service_ms_suspended;
            try {
                if(cancel_requested(*prompt)) cancel_prefill(*prompt,false);
                else if (!run_prompt_group(*prompt) && !run_prompt_windows(*prompt)) {
                    prompt->prefill->step(serial_rows_per_step_);
                    if(cancel_requested(*prompt)) cancel_prefill(*prompt,false);
                    else if (prompt->prefill->done()) complete_prompt(*prompt);
                }
            } catch(const KnownPrefillCancelled&) {cancel_prefill(*prompt,true);}
            catch(const serve::ConstraintError& e) {constraint_failed(*prompt,e);}

            prompt_debt_ms_ += prompt_share_ * std::max(0.0,std::chrono::duration<double, std::milli>(clock::now() - t0).count()-(stats_.service_ms_suspended-suspended_before));
        }
        for(auto& owner:slots_) {
            auto& s=*owner;
            if(!s.active || s.prefilling || !s.request.constraint || s.greedy) continue;
            [[maybe_unused]] std::uint64_t previous_cycles = 0;
            if constexpr (track_completion) previous_cycles = s.result.cycles;
            const auto serial_start=clock::now();
            ++stats_.steps;++stats_.solo_steps;
            try {
                s.request.constraint->prepare();
                auto next=s.target->execute_serial(*s.session,s.parents->resources,*s.dispatcher,s.pending.predicted_token(),nullptr,
                    [state=s.request.constraint](auto row){return state->select(row);});
                s.request.constraint->commit();
                s.pending=std::move(next);++s.result.cycles;++s.result.generated_tokens;++s.chosen_rows[1];
                const std::array<std::uint32_t,1> token{s.pending.predicted_token()};
                emitted(s,token);
            } catch(const serve::ConstraintError& e) {constraint_failed(s,e);}
            // Serial work pays prompt debt too, including when no ordinary slot
            // remains to run the cohort timing scope below.
            prompt_debt_ms_-=std::chrono::duration<double,std::milli>(clock::now()-serial_start).count();
            if constexpr (track_completion) serial_completed |= s.result.cycles != previous_cycles;
        }
        if constexpr (track_completion) return run_decode_round() || serial_completed;
        else (void)run_decode_round();
    }
private:
    bool run_decode_round() {
        using clock=std::chrono::steady_clock;
        last_decode_round_=clock::now();
        std::vector<Slot*> live;
        for (auto& s : slots_) if (s->active && !s->prefilling && s->greedy) {
            if(s.get()==paused_ || (paused_ && (s->session==paused_->session || s->target==paused_->target || s->native==paused_->native)))
                throw std::logic_error("paused prefill selected for decode");
            // STRIX_BATCH_SERVICE_MS: cohort primaries come only from these live window owners.
            if(paused_owner_) {
                if(s->parents->resources.scratch && paused_owner_->aliases_prefill_storage(*s->parents->resources.scratch))
                    throw std::logic_error("decode scratch aliases paused prefill lease");
                if(s->window) for(const auto rows:{2U,4U,8U,16U,32U})
                    if(const auto owner=s->window->owner_with_rows(rows);owner && paused_owner_->aliases_prefill_storage(*owner))
                        throw std::logic_error("decode window primary aliases paused prefill lease");
            }
            live.push_back(s.get());
        }
        if (live.empty()) return false;
        if (context_peers || any_peers_) {
            std::vector<const HipMtpGreedyRuntime*> peers;
            for (auto* s : live) if(!s->request.constraint) peers.push_back(s->greedy.get());
            for (auto* s : live)
                s->greedy->set_ngram_peers(!s->request.constraint && (s->request.ngram_peers ? *s->request.ngram_peers : context_peers)
                    ? std::span<const HipMtpGreedyRuntime* const>(peers) : std::span<const HipMtpGreedyRuntime* const>{});
        }
        const auto cycle_start = clock::now();
        struct Debt {double& debt; clock::time_point t0;
            ~Debt(){debt -= std::chrono::duration<double, std::milli>(clock::now() - t0).count();}} debt{prompt_debt_ms_, cycle_start};
        const auto& m = options_.mtp;
        const auto budget = [](const Slot& s) {
            return static_cast<std::uint32_t>(std::min<std::uint64_t>(s.request.max_tokens - s.result.generated_tokens, UINT32_MAX));
        };
        ++stats_.steps;
        if (live.size() == 1U) {
            auto& s = *live.front();
            ++stats_.solo_steps;
            try {
                auto transaction=s.request.constraint?s.request.constraint->window_transaction():nullptr;
                apply(s, s.greedy->cycle(s.pending, m.candidate_depth, budget(s), s.request.stop_tokens,transaction.get()));
            } catch(const serve::ConstraintError& e) {
                if(s.session->health==SerialSessionHealth::poisoned || (s.window && s.window->failed()))throw;
                constraint_failed(s,e);
            }
            return true;
        }
        ++stats_.cohort_steps;
        std::vector<HipMtpGreedyCohortMember> members;
        members.reserve(live.size());
        std::vector<std::unique_ptr<MtpSelectionTransaction>> transactions;transactions.reserve(live.size());
        for (auto* s : live) {
            const auto slot=std::find_if(slots_.begin(),slots_.end(),[&](const auto& p){return p.get()==s;})-slots_.begin();
            transactions.push_back(s->request.constraint?s->request.constraint->window_transaction():nullptr);
            members.push_back({s->greedy.get(), &s->pending, m.candidate_depth, budget(*s), s->request.stop_tokens,
                static_cast<std::uint32_t>(slot),transactions.back().get()});
        }
        static const bool joint=[] {const char* v=std::getenv("STRIX_MTP_BATCH_JOINT");return v && v[0]=='1';}();
        auto results = hip_mtp_greedy_cohort_cycle(members, &stats_.cohort,joint?&joint_:nullptr,joint?&policy_joint_:nullptr);
        for (std::size_t k = 0; k < live.size(); ++k) apply(*live[k], std::move(results[k]));
        return true;
    }

public:
    // Snapshot committed frontiers on the engine thread before dropping any slot state.
    void fail_requests(const std::string& error) {
        for (auto& s : slots_) {
            // Also covers an admission that failed before setting active.
            if (!s->request.on_finish) continue;
            s->result.finish = "execution_failure";
            s->result.error = error;
            if (s->request.constraint) s->result.constraint = s->request.constraint->verdict(s->result.finish);
            s->request.on_finish(s->result);
            s->request.on_finish = {};
        }
    }

    // After an engine failure: every active request ends (the caller reports the failure).
    void abandon() noexcept {
        for (auto& s : slots_) {
            const bool was_active=s->active;
            s->active = false;
            s->prefilling = false;
            s->waits_for=nullptr;s->join_wait={};
            s->prefill.reset();
            s->greedy.reset();
            s->pending = {};
            if (was_active && s->runtime) s->runtime->cancel();
            s->request={};
        }
    }

private:
    void wait_for_shared_prefix(Slot& s) {
        auto* cache=s.parents->tiled.cache;
        if(!cache || !cache->joining_allowed()) return;
        Slot* best=nullptr;std::uint64_t boundary=0;
        for(auto& owner:slots_) {
            auto& other=*owner;
            auto* source=other.parents->tiled.cache;
            if(&other==&s || !other.prefilling || other.waits_for || !other.prefill || !source ||
                other.prefill->restored() || other.request.constraint) continue;
            std::uint64_t common=0;
            const auto count=std::min(s.request.prompt.size(),other.request.prompt.size());
            while(common<count && s.request.prompt[common]==other.request.prompt[common]) ++common;
            const auto p=cache->join_boundary(*source,common,other.prefill->position());
            if(p>boundary) {best=&other;boundary=p;}
        }
        if(!best || boundary<=cache->cached_boundary(s.request.prompt)) return;
        best->parents->tiled.cache->request_capture(boundary);
        s.waits_for=best;s.join_wait={best->order,boundary};
        ++stats_.inflight_waits;stats_.inflight_wait_tokens+=boundary;
    }
    static void check_service(hipError_t e,const char* what) {
        if(e!=hipSuccess) throw std::runtime_error(std::string(what)+": "+hipGetErrorString(e));
    }
    static bool cancel_requested(const Slot& s) {return s.request.cancelled && s.request.cancelled();}
    void cancel_prefill(Slot& s,bool boundary) {
        if(s.parents->resources.qsa_committed_pool) s.parents->resources.qsa_committed_pool->cancel();
        s.prefill->drain_for_handoff();
        if(!service_mode_) s.parents->tiled.blocks->finish_deferred();
        for(const auto& tail:s.parents->tiled.tails) {
            if(service_mode_ && paused_owner_->aliases_prefill_storage(*tail.owner)) continue;
            tail.backend->finish_deferred();
        }
        if(service_mode_ && boundary) throw std::logic_error("nested cancellation inside native unit");
        if(boundary) {
            s.prefill->before_writer(nullptr);
            s.parents->tiled.block_owner->reset_cancelled_prefill();
            for(const auto& tail:s.parents->tiled.tails) tail.owner->reset_cancelled_prefill();
            if(s.parents->tiled.reset_prefill_memory) s.parents->tiled.reset_prefill_memory();
            ++stats_.cancel_points_cancelled;++s.result.service.cancel_points_cancelled;
        }
        s.prefill.reset();s.pending={};s.greedy.reset();
        if(s.parents->resources.qsa_committed_pool) s.parents->resources.qsa_committed_pool->begin_epoch();
        s.parents->tiled.reset_state();*s.session=s.parents->tiled.fresh;
        if(s.window) s.window->begin_session();
        s.runtime=std::make_shared<HipMtpRuntime>(s.native);
        s.prefilling=false;s.active=false;s.result.finish="cancelled";
        s.result.prompt_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s.prompt_start).count();
        ++stats_.finished;
        if(s.request.constraint) s.result.constraint=s.request.constraint->verdict(s.result.finish);
        if(s.request.on_finish) s.request.on_finish(s.result);
        s.request={};
    }
    bool slot_disjoint(const Slot& s) const {
        if(!paused_owner_ || &s==paused_ || (paused_ && (s.session==paused_->session || s.target==paused_->target || s.native==paused_->native))) return false;
        if(s.parents->resources.scratch && paused_owner_->aliases_prefill_storage(*s.parents->resources.scratch)) return false;
        if(s.window) for(const auto rows:{2U,4U,8U,16U,32U})
            if(const auto owner=s.window->owner_with_rows(rows);owner && paused_owner_->aliases_prefill_storage(*owner)) return false;
        return true;
    }
    const char* nested_decline(const Slot& s) const {
        const auto pair=numerical_profile_pair(options_);
        // Fast HC output has no nested scheduling qualification yet.
        if ((pair.fast_components & kFastPrefillHc) && prefill_option_enabled(PrefillOption::hc_output_wmma))
            return "hc-output-wmma-unqualified";
        if (index_wmma_unqualified_route(prefill_options_active(), pair.fast_components))
            return "index-proj-wmma-unqualified";
        const bool qualified=numerical_profile_dense_a8(pair.prefill) && !numerical_profile_dense_trellis(pair.prefill) &&
            !numerical_profile_dense_trellis(pair.decode) && pair.fast_components==0U;
        const auto recipe=s.prefill->nested_recipe(nest_rows_,qualified && !s.parents->tiled.blocks->gsq_model());
        if(!recipe) return recipe.decline;
        if(!slot_disjoint(s)) return "alias";
        for(const auto rows:recipe.units) {
            const auto tail=std::find_if(s.parents->tiled.tails.begin(),s.parents->tiled.tails.end(),[&](const auto& t){return t.rows==rows;});
            if(paused_owner_->aliases_prefill_storage(*tail->owner)) return "alias";
        }
        return nullptr;
    }
    void run_nested_prompt(BatchServiceStats& metrics) {
        if(!nest_rows_ || !paused_) return;
        const auto skip=[&](Slot* s,const char* reason) {
            ++metrics.nest_skips;++metrics.nest_skip_reasons[reason];
            if(s) {++s->result.service.nest_skips;++s->result.service.nest_skip_reasons[reason];}
        };
        // Retire earlier idle users before borrowing shared tail D1/union storage.
        // The parked owner (and every physical alias of its lease) remains untouched.
        for(auto& s:slots_) {
            for(const auto& tail:s->parents->tiled.tails) if(!paused_owner_->aliases_prefill_storage(*tail.owner))
                tail.backend->finish_deferred();
            if(s->window) for(const auto rows:{2U,4U,8U,16U,32U})
                if(const auto owner=s->window->owner_with_rows(rows);owner && !paused_owner_->aliases_prefill_storage(*owner))
                    owner->finish_pending_completion();
        }
        // The pump supplies requests only; preparation never executes prompt work.
        const auto can_prepare=[&] {
            const auto free=std::find_if(slots_.begin(),slots_.end(),[](const auto& s){return !s->active;});
            if(free==slots_.end()) return false;
            if(!slot_disjoint(**free)) {skip(nullptr,"alias");return false;}
            return true;
        };
        if(admission_pump_) admission_pump_([&](MtpBatchRequest request) {
            if(!service_mode_ || !has_free_slot()) throw std::logic_error("invalid service admission");
            const auto free=std::find_if(slots_.begin(),slots_.end(),[](const auto& s){return !s->active;});
            if(!slot_disjoint(**free)) throw std::logic_error("service admission aliases parked slot");
            prepare_admission(std::move(request));
        },can_prepare);
        Slot* candidate=nullptr;
        for(auto& slot:slots_) {
            auto& s=*slot;
            if(!s.prefilling || s.waits_for || &s==paused_) continue;
            if(const auto reason=nested_decline(s)) {skip(&s,reason);continue;}
            if(!candidate || s.order<candidate->order) candidate=&s;
        }
        if(!candidate) return;
        auto& s=*candidate;
        if(cancel_requested(s)) {cancel_prefill(s,false);return;}
        const auto position=s.prefill->position();
        const auto start=std::chrono::steady_clock::now();
        ++metrics.nested_admits;++s.result.service.nested_admits;
        s.prefill->force_ladder();leaf_=&s;
        try {
            // Decoding requests keep their service budget during a nested prompt: a decode round between its completed
            // units (tail blocks / windows) once STRIX_BATCH_SERVICE_MS of nested wall time passed (units are whole, no
            // native call is open on this thread; the round completes on the device before the next unit).
            auto last_round=start;
            while(!s.prefill->done() && !cancel_requested(s)) {
                if(current_hip_launch_context()) throw std::logic_error("nested step has an active launch scope");
                s.prefill->step(serial_rows_per_step_);
                if(current_hip_launch_context()) throw std::logic_error("nested step retained launch scope");
                if(service_ms_>0.0 && !s.prefill->done() &&
                   std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-last_round).count()>=service_ms_) {
                    s.prefill->drain_for_handoff();
                    if(run_decode_round()) {++metrics.service_rounds;check_service(hipDeviceSynchronize(),"complete nested decode round");}
                    last_round=std::chrono::steady_clock::now();
                }
            }
            s.prefill->drain_for_handoff();
            const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            const auto rows=s.prefill->position()-position;
            metrics.nested_rows+=rows;s.result.service.nested_rows+=rows;
            metrics.nested_ms+=ms;s.result.service.nested_ms+=ms;
            if(cancel_requested(s)) cancel_prefill(s,false);else complete_prompt(s);
        } catch(const serve::ConstraintError& e) {constraint_failed(s,e);}
        catch(...) {leaf_=nullptr;throw;}
        leaf_=nullptr;
    }
    void execute_prefill(HipKnownPrefixWindow& owner,const std::function<void()>& job,Slot* slot,std::span<Slot* const> group={}) {
        BatchServiceStats metrics;
        const auto account=[&]{stats_.add(metrics);if(slot) slot->result.service.add(metrics);else for(auto* member:group) member->result.service.add(metrics);};
        BatchServiceLookahead lookahead(lookahead_);
        auto* telemetry=current_execution_telemetry();const auto correlation=current_execution_correlation();
        // STRIX_PREFILL_PIPE: retire every earlier submitter before another host owns native scratch.
        for(auto& s:slots_) if(s->prefilling) s->prefill->drain_for_handoff();
        struct Events {
            std::vector<hipEvent_t> ring;
            explicit Events(unsigned count):ring(count,nullptr) {
                try {for(auto& e:ring) check_service(hipEventCreate(&e),"create prefill service event");}
                catch(...) {for(auto e:ring) if(e) (void)hipEventDestroy(e);throw;}
            }
            ~Events(){for(auto e:ring) (void)hipEventDestroy(e);}
        } events(lookahead_+3U);
        std::uint64_t boundary=0,retired=0;double& completed_ms=service_completed_ms_;double last_bundle_ms=0.0;
        std::exception_ptr service_error;
        const auto event=[&](std::uint64_t n){return events.ring[static_cast<std::size_t>(n%events.ring.size())];};
        const auto retire=[&](std::uint64_t end) {
            if(end<=retired) return;
            check_service(hipEventSynchronize(event(end)),"wait prefill lookahead");
            while(retired<end) {
                float ms=0.0F;check_service(hipEventElapsedTime(&ms,event(retired),event(retired+1U)),"measure completed prefill bundle");
                completed_ms+=ms;last_bundle_ms=ms;metrics.max_bundle_gpu_ms=std::max(metrics.max_bundle_gpu_ms,static_cast<double>(ms));++retired;
            }
        };
        paused_=slot;paused_owner_=&owner;
        owner.set_prefill_service_point([&](hipStream_t stream,const std::function<void()>& prepare){
            ++metrics.service_points;
            static const bool log=[]{const char* v=std::getenv("STRIX_MTP_BATCH_LOG");return v && (v[0]=='2' || v[0]=='a');}();
            if(log && !boundary && slot) std::fprintf(stderr,"batch prefill boundary rows=%u\n",owner.logical_rows());
            const auto wait=lookahead.advance();boundary=lookahead.submitted();check_service(hipEventRecord(event(boundary),stream),"record prefill boundary");
            retire(wait);
            if(cancel_points_ && slot && cancel_requested(*slot)) {prepare();throw KnownPrefillCancelled{};}
            const bool decoding=std::any_of(slots_.begin(),slots_.end(),[](const auto& s){return s->active&&!s->prefilling;});
            // The bundles still in flight (lookahead) complete before the decode round can start: count them at the last
            // completed bundle's time, so the round starts near the budget instead of `lookahead` bundles after it.
            // Due on completed GPU prefill time (bundles in flight counted at the last one's time), or on wall time since the
            // last decode round: the first bundles of a block have no completed time yet, and the gap is what decoders see.
            const bool decode_due=service_ms_>0.0 && decoding && (completed_ms+static_cast<double>(boundary-retired)*last_bundle_ms>=service_ms_ ||
                std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-last_decode_round_).count()>=service_ms_);
            const bool nest_due=nest_rows_ && slot &&
                (std::any_of(slots_.begin(),slots_.end(),[&](const auto& s){return s.get()!=slot && s->prefilling && !nested_decline(*s);}) ||
                 (has_free_slot() && has_queued_work_ && has_queued_work_()));
            if(!decode_due && !nest_due) return;
            prepare();retire(boundary);
            metrics.service_overshoot_ms_max=std::max(metrics.service_overshoot_ms_max,completed_ms-service_ms_);
            const auto start=std::chrono::steady_clock::now();
            worker_->park();
            metrics.service_ms_suspended+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(service_error) std::rethrow_exception(service_error);
            if(cancel_points_ && slot && cancel_requested(*slot)) {prepare();throw KnownPrefillCancelled{};}
            lookahead.reset();boundary=0;retired=0;completed_ms=0.0;
            check_service(hipEventRecord(event(0),stream),"reanchor prefill service clock");
        });
        // STRIX_BATCH_SERVICE_MS: GPU timestamps exclude each parked interval by re-anchoring after resume.
        try {
            worker_->run([&]{
                check_service(hipSetDevice(device_),"bind prefill worker device");
                ExecutionTelemetryBinding binding(telemetry,correlation);
                check_service(hipEventRecord(event(0),owner.prefill_stream()),"anchor prefill service clock");
                job();
                ++boundary;check_service(hipEventRecord(event(boundary),owner.prefill_stream()),"record completed prefill unit");
                retire(boundary);
            },[&]{
                service_mode_=true;
                try {
                    run_nested_prompt(metrics);
                    if(run_decode_round()) ++metrics.service_rounds;
                } catch(...) {service_error=std::current_exception();}
                service_mode_=false;
                try {check_service(hipDeviceSynchronize(),"complete decode service streams");} catch(...) {service_error=std::current_exception();}
            });
        } catch(...) {
            owner.set_prefill_service_point({});paused_=nullptr;paused_owner_=nullptr;account();
            throw;
        }
        owner.set_prefill_service_point({});paused_=nullptr;paused_owner_=nullptr;account();
    }
    // STRIX_PROMPT_BATCH (default on; =0 runs every prompt unit alone): the prompt unit of a step is a group. When the oldest running prompt's next unit is a
    // tiled block (a T-row or tail block), the next blocks of the other running prompts join it, oldest first,
    // while the rows fit the budget (the T-row block owner's arena, or STRIX_PROMPT_BATCH_ROWS if smaller). The
    // group runs as one segmented prefill on slot 0's block owner (HipKnownPromptBackend::execute_group): each
    // prompt's block is the one it would run alone, with the same per-row arithmetic. A prompt whose next unit
    // is serial rows, or a lone block, runs alone as before. Returns false when no group of two or more formed.
    static bool prompt_batch_enabled() {
        static const bool on = [] { const char* v = std::getenv("STRIX_PROMPT_BATCH"); return !v || v[0] != '0'; }();
        return on;
    }
    static std::uint32_t prompt_batch_rows() {
        static const std::uint32_t rows = [] {
            const char* v = std::getenv("STRIX_PROMPT_BATCH_ROWS");
            const long n = v ? std::atol(v) : 0L;
            return n > 0L ? static_cast<std::uint32_t>(n) : UINT32_MAX;
        }();
        return rows;
    }
    bool run_prompt_group(Slot& oldest) {
        if (!prompt_batch_enabled() || prefilling() < 2U) return false;
        const auto& lead = slots_.front()->parents->tiled;
        auto& group = lead.group ? *lead.group : *lead.blocks;
        const auto budget = std::min(lead.group ? lead.group_owner->logical_rows() : lead.rows, prompt_batch_rows());
        struct Member { Slot* slot; TiledMtpPrompt::Block block; };
        std::vector<Member> members;
        std::uint32_t rows = 0;
        const auto join = [&](Slot& s) {
            if(s.waits_for) return;
            const auto block = s.prefill->next_block();
            // A balanced partial block (STRIX_QPF2K_SCHED bit 4) runs alone.
            if (!block || block->rows > budget - rows || block->rows != block->backend->logical_rows()) return;
            members.push_back({&s, *block});
            rows += block->rows;
        };
        join(oldest);
        if (members.empty()) return false;
        std::vector<Slot*> others;
        for (auto& s : slots_) if (s.get() != &oldest && s->prefilling && !s->waits_for) others.push_back(s.get());
        std::sort(others.begin(), others.end(), [](const Slot* a, const Slot* b) { return a->order < b->order; });
        for (auto* s : others) join(*s);
        if (members.size() < 2U) return false;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<HipKnownPromptBackend::GroupMember> group_members;
        group_members.reserve(members.size());
        for (const auto& m : members)
            group_members.push_back({m.block.backend, m.slot->prefill->block_tokens(m.block), m.slot->prefill->position(), m.block.final,
                m.slot->prefill->block_observer(m.block.rows)});
        std::vector<std::vector<float>> logits;
        for(const auto& member:members)
            member.slot->prefill->before_writer((lead.group?lead.group_owner:lead.block_owner)->producer_stream());
        const auto launch=[&]{logits=HipKnownPromptBackend::execute_group(group,group_members);};
        std::vector<Slot*> group_slots;for(const auto& member:members) group_slots.push_back(member.slot);
        if(worker_) execute_prefill(*(lead.group?lead.group_owner:lead.block_owner),launch,nullptr,group_slots);else launch();
        for (std::size_t k = 0; k < members.size(); ++k) {
            auto& s = *members[k].slot;
            try {
                s.prefill->finish_block(members[k].block, logits[k], true,(lead.group?lead.group_owner:lead.block_owner).get());
                if(cancel_requested(s)) cancel_prefill(s,false);
                else if (s.prefill->done()) complete_prompt(s);
            } catch(const serve::ConstraintError& e) {constraint_failed(s,e);}
        }
        ++stats_.prompt_groups;
        stats_.grouped_blocks += members.size();
        stats_.grouped_rows += rows;
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        stats_.prompt_group_ms += ms;
        // STRIX_PROMPT_BATCH_LOG=1: one stderr line per group (blocks, rows, wall).
        static const bool log = [] { const char* v = std::getenv("STRIX_PROMPT_BATCH_LOG"); return v && v[0] == '1'; }();
        if (log) {
            std::cerr << "prompt group " << stats_.prompt_groups << ": " << members.size() << " blocks, " << rows << " rows (";
            for (std::size_t k = 0; k < members.size(); ++k) std::cerr << (k ? "+" : "") << members[k].block.rows;
            std::cerr << "), " << ms << " ms" << std::endl;
        }
        return true;
    }
    // STRIX_PROMPT_WINDOW_COHORT (default on, with STRIX_PROMPT_WINDOW; =0 runs each prompt's windows alone): when the oldest running prompt's next unit is a
    // known-row window of its serial suffix, the next windows of the other running prompts join it (oldest first)
    // and all run in one stream of steps (HipMtpWindowVerifier::prompt_rows_cohort); each prompt's window is the one
    // it runs alone. Returns false when the oldest prompt's next unit is no such window.
    static bool prompt_window_cohort_enabled() {
        static const bool on = [] { const char* v = std::getenv("STRIX_PROMPT_WINDOW_COHORT"); return !v || v[0] != '0'; }();
        return on;
    }
    bool run_prompt_windows(Slot& oldest) {
        if (!prompt_window_cohort_enabled()) return false;
        const auto first = oldest.prefill->window_unit();
        if (!first) return false;
        std::vector<Slot*> members{&oldest};
        std::vector<HipMtpWindowVerifier::PromptRowsRequest> requests{{oldest.prefill->window_verifier(), &oldest.prefill->session(),
            oldest.prefill->window_tokens(first)}};
        std::vector<Slot*> others;
        for (auto& s : slots_) if (s.get() != &oldest && s->prefilling && !s->waits_for) others.push_back(s.get());
        std::sort(others.begin(), others.end(), [](const Slot* a, const Slot* b) { return a->order < b->order; });
        for (auto* s : others) {
            const auto rows = s->prefill->window_unit();
            if (!rows) continue;
            members.push_back(s);
            requests.push_back({s->prefill->window_verifier(), &s->prefill->session(), s->prefill->window_tokens(rows)});
        }
        for(auto* member:members) member->prefill->before_writer(nullptr);
        auto tickets=HipMtpWindowVerifier::prompt_rows_cohort(requests);
        for (std::size_t k = 0; k < members.size(); ++k) {
            auto& s = *members[k];
            s.prefill->finish_window(std::move(tickets[k]),members.size()>1U);
            if(cancel_requested(s)) cancel_prefill(s,false);
            else if (s.prefill->done()) complete_prompt(s);
        }
        ++stats_.prompt_window_cohorts;
        stats_.prompt_window_members += members.size();
        return true;
    }
    // The end of a request's prompt: its pending row, its greedy runtime and its first token.
    void complete_prompt(Slot& s) {
        const auto& m = options_.mtp;
        s.pending = s.prefill->finish(s.prompt_receipt);
        static const bool state_digest=[] {const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST");return v&&v[0]=='1';}();
        if(state_digest && s.parents->tiled.cache)
            s.result.prompt_state_digest=model_content_identity_hex(s.parents->tiled.cache->state_digest(s.session->next_position));
        s.result.restored_tokens = s.prefill->restored();
        if(s.parents->tiled.cache) s.result.cache_refusal=s.parents->tiled.cache->request().refusal;
        s.prefill.reset();
        s.prefilling = false;
        s.decode_start = std::chrono::steady_clock::now();
        s.result.prompt_ms = std::chrono::duration<double, std::milli>(s.decode_start - s.prompt_start).count();
        const auto route=resolve_mtp_selection_route(options_.http_selection_mtp,!!s.window,s.window && s.window->policy_selection_admitted());
        if(s.request.constraint && !route.mtp && s.window && s.window->pending_window()) s.window->flush_pending_window();
        if(s.request.constraint) {s.request.constraint->commit();
            s.request.constraint->set_execution(route.mtp?"mtp-host":"serial-host",route.fallback?route.fallback:"");}
        if(!s.request.constraint || route.mtp) {
            s.greedy = std::make_unique<HipMtpGreedyRuntime>(s.runtime, qualified_, activated_, s.target, *s.session,
                s.parents->resources, *s.dispatcher, s.parents);
            if (s.window) s.greedy->use_window_verifier(s.window);
            if (m.draft_chain) s.greedy->set_draft_min_probability(mtp_draft_min_probability(m));
            if (!s.request.constraint && enable_context) enable_context(*s.greedy, s.request.prompt, s.request.sampler, s.request.ngram_draft);
            if (s.request.draft_depth_max) s.greedy->set_draft_depth_max(s.request.draft_depth_max);
            if (!s.request.constraint && (s.request.tool_draft ? *s.request.tool_draft : tool_draft_default) && frame_tokens[0] != frame_tokens[1] &&
                !s.request.sampler && s.window && !s.window->survival_controller() && m.draft_chain) {
                MtpFrameDraftConfig frame;
                frame.open = frame_tokens[0];
                frame.close = frame_tokens[1];
                if (const char* v = std::getenv("STRIX_TOOL_DRAFT_SINGLE_PRIOR")) {
                    const double p = std::atof(v);
                    if (p > 0.0 && p < 1.0) frame.single_prior = p;
                }
                // Committed-prefix gate (default 16 tokens; STRIX_TOOL_DRAFT_PREFIX_GATE=0 turns it off, invalid values keep 16).
                frame.prefix_gate = parse_frame_draft_prefix_gate(std::getenv("STRIX_TOOL_DRAFT_PREFIX_GATE"));
                s.greedy->enable_frame_draft(s.request.prompt, frame);
            }
        }
        const std::array<std::uint32_t, 1> first{s.pending.predicted_token()};
        s.draft_rows_begin=s.runtime->execution_stats().full_forward_rows;
        s.pending.begin_step_logits_digest();
        s.result.generated_tokens = 1;
        if(prefix_batch_cert_execution_mode(options_) && !s.request.constraint && prompt_terminal_digest_enabled()) {
            s.output_chain.append(s.pending.terminal_logits_digest());
            if(diagnostic_failed(s)) return;
        }
        emitted(s, first);
    }
    void apply(Slot& s, HipMtpGreedyCycleResult result) {
        if (result.emitted_tokens.empty() && !result.selection_failure) throw std::runtime_error("batched MTP generation made no completed target progress");
        if(result.selection_receipt){result.selection_receipt->publish();result.selection_receipt.reset();}
        if(result.selection_failure.has_value()) {
            s.result.generated_tokens+=result.emitted_tokens.size();
            if(!result.emitted_tokens.empty())s.request.on_tokens(result.emitted_tokens);
            constraint_failed(s,serve::ConstraintError(*result.selection_failure));return;
        }
        if(prefix_batch_cert_execution_mode(options_) && !s.request.constraint && prompt_terminal_digest_enabled()) {
            if(result.output_logits_digests.size()!=result.emitted_tokens.size())
                s.output_chain.fail("request final incomplete output logits coverage");
            for(const auto& row:result.output_logits_digests) s.output_chain.append(row);
            if(diagnostic_failed(s)) return;
        }
        s.pending = result.final_target;
        ++s.result.cycles;
        s.result.proposed += result.proposed_candidates;
        s.result.accepted += result.accepted_candidates;
        s.result.generated_tokens += result.emitted_tokens.size();
        ++s.chosen_rows[std::clamp<std::uint32_t>(result.target_window_rows, 1U, 8U)];
        emitted(s, result.emitted_tokens);
    }
    // Publishes tokens and ends the request at a stop token, its budget, the context, or a cancel.
    void emitted(Slot& s, std::span<const std::uint32_t> tokens) {
        const auto& stops = s.request.stop_tokens;
        if (!s.request.on_tokens(tokens)) s.result.finish = "cancelled";
        else if (std::find(stops.begin(), stops.end(), tokens.back()) != stops.end()) s.result.finish = "stop";
        else if (s.result.generated_tokens >= s.request.max_tokens) s.result.finish = "length";
        else if (s.session->next_position >= options_.context_tokens) s.result.finish = "context";
        else return;
        s.result.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s.decode_start).count();
        if (s.greedy) {
            if (const auto* ng = s.greedy->ngram_stats()) {
                s.result.ngram_rows = ng->extension_rows; s.result.ngram_accepted = ng->accepted_rows;
                s.result.source_shadow = ng->cycles; s.result.source_scored = ng->scored_cycles; s.result.source_right = ng->shadow_right_rows;
            }
            if (const auto* cd = s.greedy->context_stats()) {
                s.result.ngram_rows = cd->extension_rows; s.result.ngram_accepted = cd->accepted_rows;
                s.result.source_shadow = cd->cycles; s.result.source_scored = cd->scored_cycles; s.result.source_right = cd->shadow_right_rows;
            }
            if (const auto* fd = s.greedy->frame_stats()) {
                s.result.tool_rows = fd->extension_rows; s.result.tool_accepted = fd->accepted_rows;
                s.result.frame_cycles = fd->frame_cycles; s.result.frame_shadow = fd->cycles;
            }
        }
        static const bool state_digest=[] {const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST");return v&&v[0]=='1';}();
        static const bool final_digest=[] {const char* v=std::getenv("STRIX_FINAL_STATE_DIGEST");return v&&v[0]=='1';}();
        if((state_digest || final_digest) && s.parents->tiled.cache) {
            if(s.window && s.window->pending_window()) s.window->flush_pending_window();
            check_service(hipDeviceSynchronize(),"complete final committed state digest");
            s.result.final_state_position=s.session->next_position;
            s.result.final_state_digest=model_content_identity_hex((prefix_batch_cert_execution_mode(options_)?s.parents->tiled.cache->state_digest(s.session->next_position,s.runtime->next_source_position()):
                s.parents->tiled.cache->state_digest(s.session->next_position)));
            static const bool components=[] {const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST_COMPONENTS");return v&&v[0]=='1';}();
            if(components) s.result.final_state_components=s.parents->tiled.cache->component_digests(s.session->next_position);
        }
        if(state_digest && prefix_batch_cert_execution_mode(options_) && !s.request.constraint) {
            s.output_chain.complete(s.result.generated_tokens);
            if(!s.parents->tiled.cache) s.output_chain.fail("request final missing diagnostic state");
            if(diagnostic_failed(s)) return;
            const auto predictor=s.runtime->next_source_position();
            const auto state=s.parents->tiled.cache->state_digest(s.session->next_position,predictor);
            std::uint64_t fnv=14695981039346656037ULL;
            for(const auto token:s.request.prompt) for(unsigned k=0;k<4;++k) {fnv^=(token>>(8U*k))&255U;fnv*=1099511628211ULL;}
            std::ostringstream out;
            out<<"request final: {\"request_order\":"<<s.order<<",\"prompt_tokens\":"<<s.request.prompt.size()
               <<",\"prompt_fnv\":\""<<std::hex<<std::setw(16)<<std::setfill('0')<<fnv<<std::dec<<"\",\"generated_ids_count\":"<<s.result.generated_tokens
               <<",\"output_logits_rows\":"<<s.output_chain.rows<<",\"output_logits_chain\":\""<<model_content_identity_hex(s.output_chain.digest)
               <<"\",\"committed_tokens\":"<<s.session->committed_tokens<<",\"next_position\":"<<s.session->next_position
               <<",\"predictor_frontier\":"<<predictor<<",\"pending_token\":"<<s.pending.predicted_token()
               <<",\"generations_lineage_digest\":\""<<model_content_identity_hex(terminal_generations_lineage_digest(*s.session))
               <<"\",\"state_digest\":\""<<model_content_identity_hex(state)<<"\"}";
            std::cerr<<out.str()<<'\n';
        }
        s.result.paid_drafts=s.runtime->execution_stats().full_forward_rows-s.draft_rows_begin;
        s.result.qsa_mtp_pool_active=static_cast<bool>(s.parents->resources.qsa_committed_pool);
        s.result.chosen_window_rows=s.chosen_rows;
        mtp_pool_finish_diagnostics(*s.parents,*s.runtime,*s.session,s.pending);
        finish(s);
    }
    bool diagnostic_failed(Slot& s) {
        if(!s.output_chain.error) return false;
        // Diagnostic coverage is a request error, not an unhealthy model/engine.
        // Never publish a partial chain as a successful qualification receipt.
        s.result.finish="diagnostic_failure";
        s.result.error=s.output_chain.error;
        std::cerr<<"request final: {\"request_order\":"<<s.order
                 <<",\"diagnostic_error\":\""<<s.output_chain.error
                 <<"\",\"output_logits_rows\":"<<s.output_chain.rows
                 <<",\"output_logits_chain\":null}\n";
        finish(s);
        return true;
    }
    void constraint_failed(Slot& s,const serve::ConstraintError& e) {
        s.result.finish="constraint_failure";s.result.error=e.what();
        finish(s);
    }
    void finish(Slot& s) {
        s.waits_for=nullptr;s.join_wait={};
        if(s.request.constraint) s.result.constraint=s.request.constraint->verdict(s.result.finish);
        s.prefill.reset();s.prefilling=false;
        s.active = false;
        s.greedy.reset();
        s.output_chain={};
        s.pending = {};
        ++stats_.finished;
        if(prefix_batch_cert_execution_mode(options_) && !options_.baseline) PrefixAsyncCapture::reap();
        if (publish_session_stats) publish_session_stats();
        else if (s.parents->tiled.cache && s.parents->tiled.cache->has_store()) options_.state_cache_stats->publish(prefix_snapshot(s.parents->tiled.cache->store()));
        if (s.request.on_finish) s.request.on_finish(s.result);
        s.request={};
    }

    const Options& options_;
    const ModelContract& model_;
    const CompiledSerialExecution& compiled_;
    const QualifiedMtpDraftPlan& qualified_;
    const ActivatedMtpDraftPlan& activated_;
    EnableDraft enable_draft_;
    std::vector<std::unique_ptr<Slot>> slots_;
    bool interleave_ = false;
    std::uint32_t serial_rows_per_step_ = 16;
    // Interleaved admission: decode time owed to the requests past their prompt for the prompt units run.
    double prompt_debt_ms_ = 0.0;
    double prompt_share_ = [] {
        const char* v = std::getenv("STRIX_BATCH_PROMPT_SHARE");
        const double x = v ? std::atof(v) : 0.0;
        return x >= 0.0 ? x : 0.0;
    }();
    // STRIX_BATCH_SERVICE_MS=0 and STRIX_BATCH_CANCEL_POINTS=0 retain the synchronous native path.
    double service_ms_=[] {const char* v=std::getenv("STRIX_BATCH_SERVICE_MS");return v?std::max(0.0,std::atof(v)):250.0;}();
    unsigned lookahead_=[] {const char* v=std::getenv("STRIX_BATCH_SERVICE_LOOKAHEAD");return v?static_cast<unsigned>(std::clamp(std::atol(v),0L,64L)):2U;}();
    bool cancel_points_=[] {const char* v=std::getenv("STRIX_BATCH_CANCEL_POINTS");return !v || v[0]!='0';}();
    bool engine_call_=false;
    int device_=0;
    std::uint64_t nest_rows_=[] {const char* v=std::getenv("STRIX_BATCH_NEST_ROWS");return v?std::strtoull(v,nullptr,10):2048ULL;}();
    bool service_mode_=false;
    Slot* leaf_=nullptr;
    std::function<bool()> has_queued_work_;
    AdmissionPump admission_pump_;
    Slot* paused_=nullptr;
    HipKnownPrefixWindow* paused_owner_=nullptr;
    double service_completed_ms_=0.0;
    std::chrono::steady_clock::time_point last_decode_round_=std::chrono::steady_clock::now();
    Stats stats_;
    // Destroy/join before the slot owners and all captures.
    std::unique_ptr<BatchServiceHandoff> worker_;
    bool any_peers_ = false; // a request asked for cross-request lookup (the peers are refreshed every step from then on)
    MtpBatchWindowController joint_,policy_joint_;
    std::uint64_t admitted_ = 0;
    std::string rule_wave_;
};

// One synchronous CLI request with externally pinned evidence. Setup/loading,
// independent replay, activation and generation have separate receipt scopes.
// There is no path from command-line booleans to a qualified/activated plan.
inline void run_mtp_decode_bridge(const Options& options,const ModelContract& model,
    const CompiledSerialExecution& compiled,const std::shared_ptr<MtpDecodeTargetParents>& parents,
    SerialSessionContext& session,DirectLaunchDispatcher& dispatcher,AllocationGenerationSource& generations,
    bool deferred_status,std::string_view resolved_profile,std::ostream& output,
    MtpDecodeWindowFactory make_window={},std::function<std::string()> completion_receipt={},
    MtpDecodeQualification qualify={},MtpDecodeComponentDiagnostic component_diagnostic={},
    bool qualification_after_tiled_prompt=false,std::function<void()> before_target_mutation={}) {

    struct CaptureShutdown {
        ~CaptureShutdown() noexcept {
            try {PrefixAsyncCapture::shutdown();} catch(...) {std::terminate();}
        }
    } capture_shutdown;
    if(options.cache_activation->automatic_path)
        options.cache_activation->certification_transition=[](bool enabled) { set_prefix_batch_certification_enabled(enabled); };
    set_prefix_batch_certification_enabled(prefix_batch_cert_execution_mode(options) && !options.baseline);
    validate_resolved_mtp_cli(options);
    if(!options.mtp.enabled || !parents || session.next_position || session.committed_tokens ||
       session.health!=SerialSessionHealth::ready || session.execution_identity!=compiled.execution_identity)
        throw std::invalid_argument("MTP CLI requires a fresh actual target session");
    const auto& m=options.mtp;const auto& r=parents->resources;const auto& actual=*parents->experiment;
    const auto width=mtp_cli_mul(model.hidden_size,model.hc_count);
    HipMtpForwardGolden golden;
    std::vector<std::uint32_t> session_calibration;
    const bool dump=!m.golden_dump.empty();
    if(((m.metadata_only || dump) && (qualify || component_diagnostic)) || (qualify && component_diagnostic))
        throw std::invalid_argument("MTP metadata, golden dump, full fixture and component diagnostic routes are mutually exclusive");
    const auto pair=numerical_profile_pair(options);
    // The recorded target pair also binds the target KV identity (kv-int8-tok-v1 prompt rows differ
    // from the BF16 cache's); the canonical BF16 KV keeps the historical pair identities.
    const auto kv_bound=[&](std::uint64_t identity) {
        return model.attention_kv_storage==AttentionKvStorage::bf16 ? identity :
            (identity^0x4b564938544f4b31ULL)*0x100000001b3ULL+static_cast<std::uint32_t>(model.attention_kv_storage);
    };
    const auto prefill_profile_identity=kv_bound(numerical_profile_identity(pair.prefill));
    const auto decode_profile_identity=kv_bound(numerical_profile_identity(pair.decode));
    const bool split=pair.split();
    const bool tiled_prompt=parents && parents->tiled.blocks;
    // Under a split pair the serial MTP prompt runs every prompt row on the sibling prompt execution and
    // hands the session to decode after the replay. The tiled MTP prompt runs only its tiled blocks on the
    // prefill profile (each block backend hands the session to the prefill execution and back); its
    // calibration rows, serial suffix and decode run the decode profile, and the target adopts the last
    // block's row on the decode execution (it never holds a prompt execution).
    const bool serial_split=split && !tiled_prompt;
    const std::size_t calibration_rows=tiled_prompt ? mtp_calibration_rows(options) : options.tokens.size()-1U;
    // A tiled prompt's blocks hand off to the prefill execution exactly under a split pair.
    const auto tiled_profile_bound=[&](const MtpDecodeTargetParents::TiledPrompt& t) {
        const auto expected=split ? t.prefill_execution_identity : 0U;
        if(split && (!expected || expected==compiled.execution_identity)) return false;
        if(t.blocks->handoff_prefill_execution_identity()!=expected) return false;
        return std::all_of(t.tails.begin(),t.tails.end(),[&](const auto& tail){
            return tail.backend && tail.backend->handoff_prefill_execution_identity()==expected;});
    };
    if(qualification_after_tiled_prompt && (!qualify || component_diagnostic))
        throw std::invalid_argument("tiled prompt qualification admission requires only its full qualification callback");
    if(tiled_prompt && ((qualify && !qualification_after_tiled_prompt) || component_diagnostic || !parents->tiled.block_owner || !parents->tiled.reset_state ||
       !parents->tiled.rows || std::any_of(parents->tiled.tails.begin(),parents->tiled.tails.end(),
           [&](const auto& t){return !t.backend || !t.owner || t.rows>=parents->tiled.rows;}) || !tiled_profile_bound(parents->tiled)))
        throw std::invalid_argument("tiled MTP prompt requires its block owners, a state reset and, exactly under a split prefill "
            "profile, block handoffs to the prefill execution");
    if(serial_split!=(parents && parents->prompt.compiled) ||
       (serial_split && (!parents->prompt.dispatcher || parents->prompt.resources.session_state_memory!=parents->resources.session_state_memory)))
        throw std::invalid_argument("MTP split prefill profile requires its actual prompt execution on the serial MTP prompt and only there");
    // The calibration rows of a tiled prompt are serial decode rows: its golden records the decode profile
    // as the prefill of those rows.
    const auto golden_prefill_identity=tiled_prompt ? decode_profile_identity : prefill_profile_identity;
    const auto golden_prefill_profile=tiled_prompt ? pair.decode : pair.prefill;
    if(!m.metadata_only && !dump){
    const auto golden_limit=options.rotary.mode()==RotaryMode::yarn?
        std::min(m.golden_max_bytes,mtp_payload::golden_file(calibration_rows,width,model.vocab_size,288U)):m.golden_max_bytes;
    golden=read_mtp_forward_golden_file(m.golden,mtp_cli_digest(m.golden_sha256),golden_limit);
    validate_mtp_forward_golden_rotary(golden,model.rotary_identity(),options.context_tokens,
        prefill_profile_identity,golden_prefill_identity,decode_profile_identity);
    // A golden records the target prefill/decode pair whose prefill produced its prompt rows. A
    // version-1 golden records none and is admitted only for a single-profile run, as before.
    if(golden.target_decode_profile_identity ?
           golden.target_prefill_profile_identity!=golden_prefill_identity || golden.target_decode_profile_identity!=decode_profile_identity :
           serial_split || model.attention_kv_storage!=AttentionKvStorage::bf16)
        throw std::invalid_argument("MTP golden was dumped under another target prefill/decode numerical profile pair "
            "or KV identity (a version-1 golden records none and requires one profile for both and the BF16 KV); dump a golden with this pair");
    if(tiled_prompt ? golden.rows.size()!=calibration_rows :
       golden.rows.size()+1U!=options.tokens.size())
        throw std::invalid_argument(tiled_prompt ? "MTP golden of a tiled prompt must cover exactly its calibration rows" :
            "MTP golden must cover every prompt predictor pair, leaving only its final pending target row");
    if(options.session_mtp_fixed && !options.http_serve) {
        // The pinned replay payload, not a new request's suffix, supplies the independent
        // calibration chain. All native replay comparisons and activation gates remain intact.
        if(golden.rows.empty())throw std::invalid_argument("session MTP calibration is empty");
        session_calibration.push_back(golden.rows.front().source_input_token);
        for(const auto& row:golden.rows) {
            if(row.source_input_token!=session_calibration.back() || row.source_input_token>=model.vocab_size ||
               row.next_input_token>=model.vocab_size)throw std::invalid_argument("session MTP golden chain differs");
            session_calibration.push_back(row.next_input_token);
        }
    }
    const auto& calibration_tokens=session_calibration.empty()?options.tokens:session_calibration;
    for(std::size_t i=0;i<golden.rows.size();++i) {
        const auto& row=golden.rows[i];
        if(row.source_position!=i || row.source_input_token!=calibration_tokens[i] || row.next_input_token!=calibration_tokens[i+1U] ||
           row.target_hidden_bf16.size()!=width || row.next_hidden_bf16.size()!=width || row.full_logits.size()!=model.vocab_size)
            throw std::invalid_argument("MTP independent golden prompt chain or full-row geometry differs before target execution");
    }
    }
    const auto assets=parents->assets ? parents->assets : load_mtp_cli_assets(m,model,actual.mapped,actual.manifest,actual.model,generations);
    HipMtpBudget native_budget;native_budget.retained_bytes=m.runtime_bytes;native_budget.context_capacity=options.context_tokens;
    native_budget.maximum_candidate_depth=m.candidate_depth;
    native_budget.normalized_rounding=m.normalized_rounding=="fp32"?HipMtpInputRounding::fp32:HipMtpInputRounding::bf16_rne;
    native_budget.projection_rounding=m.projection_rounding=="fp32"?HipMtpInputRounding::fp32:HipMtpInputRounding::bf16_rne;
    auto native=HipMtpResources::create(model,assets->compiled(),assets->sidecar_manifest(),assets->sidecar_residency(),
        assets->target_manifest(),assets->target_residency(),assets->shared_target(),native_budget,actual.allocator,model.rotary_identity());
    auto runtime=std::make_shared<HipMtpRuntime>(native);
    if(m.metadata_only){
        assets->validate_assets_unchanged();
        output<<'{'<<prefill_identity_fields(prefill_options_active())<<",\"mode\":\"mtp-metadata-only\",\"mtp_executed\":false,\"mtp_activated\":false,\"mtp_qualified\":false,"
            <<"\"proposal_identity\":"<<assets->compiled().proposal_identity<<",\"plan_identity\":"<<assets->compiled().plan_identity
            <<",\"semantic_identity\":"<<assets->compiled().forward_semantics.contract_identity<<",\"native_profile_identity\":"<<runtime->numerical_profile_identity()
            <<",\"head_profile_identity\":"<<hip_mtp_lm_head_profile_identity(model,native->tensors().lm_head)
            <<",\"head_recipe\":"<<static_cast<std::uint32_t>(select_hip_mtp_lm_head_recipe(model,native->tensors().lm_head))
            <<",\"target_content\":\""<<model_content_identity_hex(assets->shared_target().target_model_content_identity)
            <<"\",\"sidecar_content\":\""<<model_content_identity_hex(assets->shared_target().sidecar_model_content_identity)
            <<"\",\"target_file_sha256\":\""<<model_content_identity_hex(actual.mapped->file_sha256())
            <<"\",\"sidecar_file_sha256\":\""<<model_content_identity_hex(assets->sidecar_mapping()->file_sha256())
            <<"\",\"normalized_rounding\":\""<<m.normalized_rounding<<"\",\"projection_rounding\":\""<<m.projection_rounding<<"\"}\n";
        return;
    }
    MtpForwardReplayPolicy policy{m.policy_identity,m.max_hidden_error,m.max_logit_error,2U,true};
    (void)assess_mtp_forward_replay(policy,{});
    if(!dump && (golden.target_content!=assets->shared_target().target_model_content_identity ||
       golden.sidecar_content!=assets->shared_target().sidecar_model_content_identity ||
       golden.structural_plan_identity!=assets->compiled().plan_identity ||
       golden.semantic_identity!=assets->compiled().forward_semantics.contract_identity ||
       golden.numerical_profile_identity!=runtime->numerical_profile_identity()))
        throw std::invalid_argument("MTP golden does not bind actual verified assets, semantics and native cast profile");
    if(component_diagnostic){
        // Independent component inputs are admitted against these actual assets.
        // This diagnostic neither runs full replay nor creates qualified/active
        // greedy state, and cannot enter the ordinary generation path.
        try {assets->validate_assets_unchanged();component_diagnostic(native,runtime->numerical_profile_identity());assets->validate_assets_unchanged();}
        catch(...){runtime->cancel();throw;}
        return;
    }
    // Same complete known payload bound as the replay consumer; admit it before
    // the target prompt is mutated, rather than waiting until replay entry.
    const auto replay_payload=mtp_payload::replay_host(golden.rows.size(),width,model.vocab_size);
    if(!dump && replay_payload>m.replay_host_bytes)throw std::invalid_argument("MTP complete independent replay exceeds host payload budget before target mutation");
    HipMtpForwardReplayRequest replay_request{policy,hip_mtp_forward_golden_payload_identity(golden),golden.rows.size(),m.replay_host_bytes,
        golden_prefill_identity,decode_profile_identity,prefill_profile_identity};
    // Tiled prompt: only the calibration rows and the serial remainder take capture slots.
    const auto retained_rows=static_cast<std::uint32_t>(tiled_prompt
        ? std::max<std::size_t>({calibration_rows+1U,static_cast<std::size_t>(options.known_prefix_tail_capture_rows?options.known_prefix_tail_capture_rows:
            parents->tiled.tails.empty()?parents->tiled.rows:parents->tiled.tails.back().rows)+1U,
            static_cast<std::size_t>(m.candidate_depth)+3U})
        : std::max<std::size_t>(options.tokens.size(),static_cast<std::size_t>(m.candidate_depth)+3U));
    HipMtpTargetDeferredConfig d1;
    if(deferred_status)d1={r.session_state_layout,&generations};
    auto target=HipMtpTargetHidden::create(model,compiled,actual.memory,actual.model,actual.allocator,
        retained_rows,m.target_capture_bytes,d1);
    if(serial_split)target->set_prompt_execution(*parents->prompt.compiled);
    // Sampled decoding: the final prompt row and every generated row select by the sampler;
    // earlier prompt rows keep first-max selection (their predictions are never emitted).
    const auto sampler=mtp_sampling(m) && !dump && !qualify ? std::make_shared<TokenSampler>(mtp_sampling_policy(m))
                                                           : std::shared_ptr<TokenSampler>{};
    const bool separated_prompt=mtp_prompt_phase_separation(options);
    if(separated_prompt && (qualify || component_diagnostic || !parents->retire_prompt || !parents->prepare_prompt_pool))
        throw std::invalid_argument("MTP phased prompt lacks its closed CLI retirement contract");
    if(separated_prompt) parents->prepare_prompt_pool();
    std::shared_ptr<HipMtpWindowVerifier> window;
    const auto allocate_window=[&] {
    if(m.window_rows && !dump && (!options.session_mtp_fixed || options.http_serve)) {
        if(!make_window)throw std::invalid_argument("MTP requested target window has no actual prepared native factory");
        window=make_window(target);
        if(!window || window->target_owner()!=target || window->logical_rows()!=m.window_rows)
            throw std::invalid_argument("MTP target window factory returned incompatible actual owner");
    }
    };
    if(!separated_prompt) allocate_window();
    if(parents->resources.qsa_committed_pool) native->attach_qsa_pool(parents->resources.qsa_committed_pool);
    if(!options.http_serve && options.cache_readiness_recheck) options.cache_readiness_recheck();
    assets->validate_assets_unchanged();
    std::vector<HipMtpReplayRow> replay_rows;replay_rows.reserve(golden.rows.size());
    std::vector<std::uint32_t> generated;generated.reserve(static_cast<std::size_t>(options.generate_tokens));
    std::uint64_t cycles=0,proposed=0,accepted=0,target_rows=0,window_attempts=0,window_accepts=0,window_restores=0,r2_dispatches=0;
    std::array<std::uint64_t,9> chosen_rows{};
    HipMtpTargetHiddenTicket pending;
    HipMtpNativeForwardReplayResult replay;
    QualifiedMtpDraftPlan qualified;ActivatedMtpDraftPlan activated;
    const std::array<std::uint32_t,1> eos{options.eos_token.value_or(static_cast<std::uint32_t>(model.ple_eos_token_id))};
    std::string stop="token-budget";
    double calibration_ms=0.0;
    std::string tiled_receipt;
    // h_{P-1} of the tiled prompt (one wide row), retained through the request.
    std::byte* tiled_held=nullptr;
    if(tiled_prompt && hipMalloc(reinterpret_cast<void**>(&tiled_held),static_cast<std::size_t>(width*2U))!=hipSuccess)
        throw std::runtime_error("tiled MTP held hidden allocation failed");
    struct HeldFree{std::byte* p;~HeldFree(){if(p)(void)hipFree(p);}} held_free{tiled_held};
    try {
        // The prompt rows are the prefill: under a split pair they run the sibling prompt
        // execution, and the session moves to the decode execution only after the replay.
        if(serial_split)hand_off_serial_session(session,{&compiled,&r},{parents->prompt.compiled,&parents->prompt.resources});
        const auto prompt_rows=tiled_prompt ? calibration_rows+1U : options.tokens.size();
        const auto calibration_start=std::chrono::steady_clock::now();
        for(std::size_t i=0;i<prompt_rows;++i) {
            check_yarn_memory();
            if(sampler && !tiled_prompt && i+1U==options.tokens.size())target->set_token_sampler(sampler);
            const auto input=session_calibration.empty()?options.tokens[i]:session_calibration[i];
            if(before_target_mutation) before_target_mutation();
            auto current=serial_split ? target->execute_serial(session,parents->prompt.resources,*parents->prompt.dispatcher,input)
                               : target->execute_serial(session,r,dispatcher,input);
            check_yarn_memory();
            if(pending.valid())replay_rows.push_back({pending,input});
            pending=std::move(current);
        }
        if(dump){
            // Same native forward as replay, recorded instead of compared. The
            // explicit self-generated pins keep it distinguishable from an
            // independent reference; nothing is qualified or generated here.
            golden=capture_hip_mtp_forward_golden(*runtime,replay_rows,mtp_self_generated_golden_source(),mtp_self_generated_golden_reference,
                options.rotary.mode()==RotaryMode::yarn ? std::function<void()>{check_yarn_memory} : std::function<void()>{});
            golden.target_prefill_profile_identity=golden_prefill_identity;
            golden.target_decode_profile_identity=decode_profile_identity;
            if(golden.rotary_factor) golden.prompt_prefill_profile_identity=prefill_profile_identity;
            ModelContentIdentity sha;
            publish_yarn_output([&] { sha=write_mtp_forward_golden_file(m.golden_dump,golden,m.golden_max_bytes); });
            assets->validate_assets_unchanged();
            const auto memory_receipt=yarn_memory_receipt();
            publish_yarn_output([&] {
            output<<"{"<<prefill_identity_fields(prefill_options_active())<<","<<memory_receipt<<"\"mode\":\"mtp-self-generated-golden-dump\",\"mtp_reference_independence\":\"self-generated-native-forward-not-independent\","
                <<"\"mtp_golden_path\":\""<<decode_receipt::escape_json_string(m.golden_dump)<<"\",\"mtp_golden_sha256\":\""<<model_content_identity_hex(sha)
                <<"\",\"mtp_golden_rows\":"<<golden.rows.size()<<",\"mtp_golden_payload_sha256\":\""<<model_content_identity_hex(hip_mtp_forward_golden_payload_identity(golden))
                <<"\",\"mtp_independent_source_content\":\""<<model_content_identity_hex(golden.independent_source_content)
                <<"\",\"mtp_independent_reference_identity\":"<<golden.independent_reference_identity
                <<",\"mtp_structural_plan_identity\":"<<golden.structural_plan_identity<<",\"mtp_semantic_identity\":"<<golden.semantic_identity
                <<",\"mtp_native_arithmetic_profile_identity\":"<<golden.numerical_profile_identity
                <<",\"mtp_golden_target_prefill_numerical_profile\":\""<<numerical_profile_name(golden_prefill_profile)
                <<"\",\"mtp_golden_target_decode_numerical_profile\":\""<<numerical_profile_name(pair.decode)
                <<"\",\"mtp_golden_target_prefill_profile_identity\":"<<golden.target_prefill_profile_identity
                <<",\"mtp_golden_target_decode_profile_identity\":"<<golden.target_decode_profile_identity;
            if(golden.rotary_factor) output<<",\"mtp_golden_version\":3,\"mtp_golden_capacity\":"<<golden.context_capacity
                <<",\"mtp_golden_calibration_begin\":"<<golden.calibration_begin<<",\"mtp_golden_calibration_end\":"<<golden.calibration_end
                <<",\"mtp_golden_prompt_prefill_profile_identity\":"<<golden.prompt_prefill_profile_identity
                <<",\"mtp_golden_rotary\":"<<model.rotary_identity().json();
            output<<"}\n";
            });
            return;
        }
        // This call executes all eleven native stages for every independent
        // reference pair. The resulting real observation is the sole evidence.
        if(options.rotary.mode()==RotaryMode::yarn) replay_request.safe_boundary=check_yarn_memory;
        replay=replay_hip_mtp_forward(*runtime,replay_rows,golden,replay_request);
        check_yarn_memory();
        if(!replay.assessment.passes_declared_policy){
            std::ostringstream error;error<<std::setprecision(17)
                <<"MTP actual full-hidden/full-vocabulary replay failed its declared policy; generation is not admitted"
                <<"; hidden_max_abs_error="<<replay.observation.maximum_hidden_abs_error
                <<"; logit_max_abs_error="<<replay.observation.maximum_logit_abs_error
                <<"; hidden_values="<<replay.observation.hidden_values_compared
                <<"; logit_values="<<replay.observation.logit_values_compared
                <<"; tokens_match="<<replay.assessment.tokens_match
                <<"; hidden_within_policy="<<replay.assessment.hidden_within_policy
                <<"; logits_within_policy="<<replay.assessment.logits_within_policy
                <<"; token_pairs=";
            for(std::size_t i=0;i<replay.observation.engine_tokens.size();++i){
                if(i)error<<',';
                error<<replay.observation.reference_tokens.at(i)<<':'<<replay.observation.engine_tokens[i];
            }
            throw std::runtime_error(error.str());
        }
        qualified=qualify_mtp_draft_plan(assets->compiled(),policy,replay.observation);
        require_qualified_mtp_draft_plan(qualified);
        activated=activate_mtp_draft_plan(assets->compiled(),policy,replay.observation);
        if(runtime->next_source_position()+1U!=session.next_position)
            throw std::logic_error("MTP replay did not prime the actual final prompt frontier");
        replay_rows.clear(); // release prompt snapshots before bounded cycles
        if(options.rotary.mode()==RotaryMode::yarn) {
            // Qualification retains its receipts/pins; calibration values are no
            // longer read. Release their storage before prompt/chain owners grow.
            std::vector<HipMtpForwardGoldenRow>().swap(golden.rows);
        }
        // Context n-gram draft extension (D-047, default auto since D-055): `forced` = explicitly required (the
        // runtime rejects it where it cannot run); otherwise it runs where admissible, and not under the survival
        // controller. STRIX_MTP_NGRAM_FLOOR / STRIX_MTP_NGRAM_PRIOR tune its gate (defaults 0.5 / 0.2).
        const auto ngram_wanted=[&](bool forced){
            return forced || (mtp_ngram_draft_admissible(m) && window && !window->survival_controller());
        };
        const auto ngram_config=[]{
            MtpNgramDraftConfig ngram;
            if(const char* v=std::getenv("STRIX_MTP_NGRAM_FLOOR"))ngram.floor=std::strtod(v,nullptr);
            if(const char* v=std::getenv("STRIX_MTP_NGRAM_PRIOR"))ngram.prior=std::strtod(v,nullptr);
            return ngram;
        };
        // --mtp-ngram-draft=ctx|copy: the context drafter instead of the extension (same admission).
        const auto enable_ngram=[&](HipMtpGreedyRuntime& greedy,std::span<const std::uint32_t> prompt){
            if(!m.ngram_context) {greedy.enable_ngram_draft(prompt,ngram_config());return;}
            MtpContextDraftConfig context;context.copy_edit=m.ngram_copy_edit;
            // =wide proposes only from a 16-token unanimous anchor (ABA t28: json +1.7 % -> +0.2 %, rewrite -36.8 % ->
            // -40.6 %, agent unchanged). Diagnostic override: minimum wide anchor (drafts only, never target output).
            if(m.ngram_wide_only) context.wide_anchor=16U;
            // Chain copy (=wide only, opt-in): STRIX_MTP_CHAIN_COPY=FRESH[,RESUME] (e.g. 12,5).
            if(const char* v=std::getenv("STRIX_MTP_CHAIN_COPY"); v && m.ngram_wide_only) {
                char* end=nullptr;context.chain_fresh=static_cast<std::uint32_t>(std::strtoul(v,&end,10));
                if(end && *end==',') context.chain_resume=static_cast<std::uint32_t>(std::strtoul(end+1,nullptr,10));
            }
            if(const char* v=std::getenv("STRIX_MTP_WIDE_ANCHOR")) context.wide_anchor=static_cast<std::uint32_t>(std::strtoul(v,nullptr,10));
            // =wide: the extension drafts every ordinary cycle; the context drafter only observes and proposes wide.
            if(m.ngram_wide_only) greedy.enable_ngram_draft(prompt,ngram_config());
            greedy.enable_context_draft(prompt,context,m.ngram_wide_only);
        };
        const auto enable_draft_on=[&](HipMtpRuntime& draft_runtime,std::span<const std::uint32_t> prompt,
            const std::shared_ptr<TokenSampler>& draft_sampler) {
            if(!m.draft_chain) return;
            // Drafts only: the subset (or full) head changes acceptance, never target output.
            HipMtpDraftHeadConfig draft;draft.full=m.draft_vocab.empty() && !m.draft_context_vocab;
            if(!m.draft_vocab.empty()) {
                std::string chosen;
                const auto spec=HipMtpDraftHead::resolve_vocabulary_spec(m.draft_vocab,prompt,
                    static_cast<std::uint32_t>(model.vocab_size),chosen);
                if(spec!=m.draft_vocab) std::cerr<<"MTP draft vocabulary: "<<chosen<<'\n';
                draft.base_tokens=HipMtpDraftHead::read_token_file(spec,static_cast<std::uint32_t>(model.vocab_size));
            }
            draft.context_capacity=m.draft_context_vocab;draft.q4=m.draft_q4;
            // Runner-up probabilities feed the survival controller; STRIX_MTP_DRAFT_TOP2=1 logs them (diagnostic).
            draft.runner_up=hip_mtp_survival_controller() || std::getenv("STRIX_MTP_DRAFT_TOP2")!=nullptr;
            draft_runtime.enable_draft_chain(std::move(draft),m.draft_exact);draft_runtime.observe_draft_context(prompt);
            if(draft_sampler && m.sample_speculation)draft_runtime.set_draft_sampler(draft_sampler);
        };
        const auto enable_draft=[&](std::span<const std::uint32_t> prompt) {enable_draft_on(*runtime,prompt,sampler);};
        // --mtp-copy-wide-rows (single requests; the batched route rejects it): null when off.
        const auto wide_copy=qualify?std::unique_ptr<MtpWideCopyCycle>{}:make_mtp_wide_copy(*parents,m.copy_wide_rows,*target);
        if(tiled_prompt) {
            calibration_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-calibration_start).count();
            // The calibration rows only qualified the predictor: start the real prompt from a fresh
            // target and a fresh predictor (the qualified/activated plans bind assets, not an owner).
            pending={};
            runtime.reset();
            runtime=std::make_shared<HipMtpRuntime>(native);
            if (auto* lookup = dynamic_cast<FileMappedPleLookupBackend*>(parents->ple.get()))
                if (auto turn = lookup->residency()) {
                    cancel_ple_lookup_prefetch(*parents->ple, parents->prefetch);
                    turn->phase(true);
                }
            if(parents->resources.qsa_committed_pool) parents->resources.qsa_committed_pool->begin_epoch();
            parents->tiled.reset_state();
            session=parents->tiled.fresh;
            enable_draft(options.tokens);
            // The predictor's durable rows and the held h_{P-1} travel with each checkpoint; the identity
            // gains everything that changes them (cache rows use the draft kernels when the draft chain
            // is not exact). Every slot of a batched run extends its own view identically.
            const auto predictor_profile_identity=runtime->numerical_profile_identity();
            const auto extend_cache=[&](HipPrefixStateCacheSession& cache,const HipMtpResources& predictor,std::byte* held,
                std::function<void(std::uint64_t)> restore={}) {
                const auto pool=predictor.qsa_pool();
                if(pool) cache.set_before_restore_hook([pool] {pool->invalidate(false,0);pool->invalidate(true,0);});
                const auto kv=predictor.view(HipMtpBuffer::kv),index=predictor.view(HipMtpBuffer::raw_index);
                bind_mtp_prefix_cache(cache,kv.address,mtp_cli_mul(mtp_cli_mul(model.kv_heads,model.head_dim),4U),
                    index.address,mtp_cli_mul(model.indexer_head_dim,2U),reinterpret_cast<std::uintptr_t>(held),width*2U,
                    assets->shared_target().sidecar_model_content_identity,assets->compiled().plan_identity,
                    predictor_profile_identity,m.draft_chain,m.draft_exact,m.draft_q4,std::move(restore));
                if(options.cache_activation->automatic_path && options.cache_activation->source=="auto" &&
                    options.cache_activation->prefix && options.cache_activation->decision_phase=="post-residency-ready") {
                    const auto setup_start=std::chrono::steady_clock::now();
                    try {
                        options.cache_activation->pool_setup_ms=cache.attach_capture_pool(true);
                    } catch(...) {
                        options.cache_activation->pool_setup_ms=std::chrono::duration<double,std::milli>(
                            std::chrono::steady_clock::now()-setup_start).count();
                        options.cache_activation->reason="capture-pool-allocation-failed";
                        throw;
                    }
                }
            };
            const auto bind_cache=[&] {
            if(parents->tiled.cache) {
                extend_cache(*parents->tiled.cache,*native,tiled_held,
                    [&](std::uint64_t position){runtime->restore_cache_frontier(qualified,activated,position-1U);});
                // --rtx-state-import / STRIX_RTX_EXPORT_STATE: what a state at position p must carry besides its bytes.
                // Every prompt row (tiled or serial) runs each step of the serial plan once, and each step's writes
                // advance the generation of every flag it writes: generation f = base + p * (steps writing f).
                if(parents->tiled.rtx_import || rtx_state_export_request()) {
                    RtxStateImportFacts facts;
                    for(const auto& step:compiled.execution.steps) {
                        const auto written=static_cast<std::uint32_t>(step.writes_state);
                        facts.valid_union=facts.valid_union|step.writes_state;
                        for(std::size_t f=0;f<facts.generations_per_row.size();++f) if(written&(1U<<f)) ++facts.generations_per_row[f];
                    }
                    facts.model_content=assets->shared_target().target_model_content_identity;
                    parents->tiled.cache->set_rtx_facts(facts);
                    if(parents->tiled.rtx_import) parents->tiled.cache->enable_rtx_import(parents->tiled.rtx_import,parents->tiled.fresh);
                }
            } else if(parents->tiled.rtx_import)
                throw std::invalid_argument("--rtx-state-import needs the state component registry of the tiled MTP prompt");
            };
            // Required diagnostic/import observers must reach exactly the OFF binding
            // before the optional transaction. Required setup failures propagate.
            bind_cache();
            const auto* off_cache=parents->tiled.cache;
            if(options.cache_attach_ready)
                finalize_cache_attachment(options.cache_attach_ready,[&] {
                    if(parents->tiled.cache!=off_cache) bind_cache();
                },options.cache_attach_rollback,[&] {
                    // Binding can attach retained payload/staging resources. Publish
                    // their finalized commitment before exposing the serving runner.
                    if(parents->tiled.cache && parents->tiled.cache->has_store())
                        options.state_cache_stats->publish(prefix_snapshot(parents->tiled.cache->store()));
                });

            std::uint64_t served=0;
                const MtpServeRunner ordinary_request=[&](const MtpServeRequest& q) {
                    const auto stats_guard=std::shared_ptr<void>(nullptr,[&](void*) noexcept {
                        if(parents->tiled.cache && parents->tiled.cache->has_store()) try {
                            options.state_cache_stats->publish(prefix_snapshot(parents->tiled.cache->store()));
                        } catch (...) {}
                    });
                    const auto vocabulary=model.vocab_size;
                    if(q.prompt.empty() || q.prompt.size()>=options.context_tokens || !q.max_tokens || !q.on_tokens)
                        throw std::invalid_argument("served request needs a prompt shorter than the context, a token budget and a sink");
                    for(const auto t:q.prompt)if(t>=vocabulary)throw std::invalid_argument("served prompt token outside the vocabulary");
                    for(const auto t:q.stop_tokens)if(t>=vocabulary)throw std::invalid_argument("served stop token outside the vocabulary");
                    MtpServeResult out;
                    out.qsa_mtp_pool_active=static_cast<bool>(parents->resources.qsa_committed_pool);
                    if(q.constraint) {const auto route=resolve_mtp_selection_route(options.http_selection_mtp,!!window,window && window->policy_selection_admitted());
                        q.constraint->set_execution(route.mtp?"mtp-host":"serial-host",route.fallback?route.fallback:"");}
                    const auto request_start=std::chrono::steady_clock::now();
                    try {
                        // Every request starts exactly like the command-line request after calibration:
                        // fresh target state and session, fresh predictor, no deferred window rows.
                        pending={};
                        ++served;
                        runtime.reset();
                        runtime=std::make_shared<HipMtpRuntime>(native);
                        if(parents->resources.qsa_committed_pool) parents->resources.qsa_committed_pool->begin_epoch();
                        parents->tiled.reset_state();
                        session=parents->tiled.fresh;
                        if(window)window->begin_session();
                        // Keep predictor priming/checkpoints identical under the shared cache identity.
                        if(parents->tiled.cache) parents->tiled.cache->set_retention(q.cache_retention != serve::CacheRetention::no_store, q.session_id);
                        enable_draft(q.prompt);
                        std::string request_receipt;
                        pending=run_tiled_mtp_prompt(q.prompt,model,compiled,parents->tiled,session,*target,*runtime,qualified,activated,
                            r,dispatcher,tiled_held,request_receipt,{},
                            q.constraint?HipMtpHostRowSelector{[state=q.constraint](auto row){return state->select(row);}}:HipMtpHostRowSelector{},q.on_progress);
                        static const bool state_digest=[] {const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST");return v&&v[0]=='1';}();
                        if(state_digest && parents->tiled.cache)
                            out.prompt_state_digest=model_content_identity_hex(parents->tiled.cache->state_digest(session.next_position));
                        if(q.constraint) q.constraint->commit();
                        if(parents->tiled.cache) {
                            out.restored_tokens=parents->tiled.cache->request().restored_position;
                            out.cache_refusal=parents->tiled.cache->request().refusal;
                        }
                        if(parents->tiled.cache && parents->tiled.cache->request().rtx_import)
                            out.rtx_import_json=rtx_import_receipt(parents->tiled.cache->request());
                        const auto decode_start=std::chrono::steady_clock::now();
                        out.prompt_ms=std::chrono::duration<double,std::milli>(decode_start-request_start).count();
                        if(parents->tiled.cache && prompt_state_digest_request()) {
                            // The served request's own prompt is the committed history (never the startup tokens).
                            if(window)window->flush_pending_window();
                            print_target_state_digest(stderr,"prompt",session.next_position,[&] {
                                return parents->tiled.cache->target_state_digest(session.next_position,q.prompt,pending.predicted_token());});
                        }
        HipMtpGreedyRuntime greedy(runtime,qualified,activated,target,session,r,dispatcher,parents);
                        const auto route=resolve_mtp_selection_route(options.http_selection_mtp,!!window,window && window->policy_selection_admitted());
                        if(q.constraint && !route.mtp && window && window->pending_window())window->flush_pending_window();
                        if(q.constraint)q.constraint->set_execution(route.mtp?"mtp-host":"serial-host",route.fallback?route.fallback:"");
                        if((!q.constraint || route.mtp) && window)greedy.use_window_verifier(window);
                        if(m.draft_chain)greedy.set_draft_min_probability(mtp_draft_min_probability(m));
                        // n-gram extension per request (D-055): the request's "ngram_draft" (true = on where admissible,
                        // false = off), else the engine's mode.
                        if(!q.constraint && (q.ngram_draft ? *q.ngram_draft && ngram_wanted(false) : !m.ngram_off && ngram_wanted(m.ngram_draft)))
                            enable_ngram(greedy,q.prompt);
                        const auto is_stop=[&](std::uint32_t t){return std::find(q.stop_tokens.begin(),q.stop_tokens.end(),t)!=q.stop_tokens.end();};
                        const std::array<std::uint32_t,1> first{pending.predicted_token()};
                        const auto draft_rows_begin=runtime->execution_stats().full_forward_rows;
                        pending.begin_step_logits_digest();
                        out.generated_tokens=1;
                        if(q.backend_probe) q.backend_probe->boundary(serve::BackendProof::work_completion);
                        if(!q.on_tokens(first))out.finish="cancelled";
                        else if(is_stop(first[0]))out.finish="stop";
                        while(out.finish=="length" && out.generated_tokens<q.max_tokens) {
                            PrefixAsyncCapture::reap();
                            if(session.next_position>=options.context_tokens){out.finish="context";break;}
                            if(q.constraint && !route.mtp) {
                                q.constraint->prepare();
                                auto next=target->execute_serial(session,r,dispatcher,pending.predicted_token(),nullptr,
                                    [state=q.constraint](auto row){return state->select(row);});
                                q.constraint->commit();pending=std::move(next);++out.cycles;++out.generated_tokens;++out.chosen_window_rows[1];
                                const std::array<std::uint32_t,1> token{pending.predicted_token()};
                                if(q.backend_probe) q.backend_probe->boundary(serve::BackendProof::work_completion);
                                if(!q.on_tokens(token))out.finish="cancelled";
                                else if(is_stop(token[0]))out.finish="stop";
                                continue;
                            }
                            const auto budget=std::min<std::uint64_t>(q.max_tokens-out.generated_tokens,UINT32_MAX);
                            auto transaction=q.constraint?q.constraint->window_transaction():nullptr;
                            auto wide=!q.constraint && wide_copy?wide_copy->cycle(greedy,pending,static_cast<std::uint32_t>(budget),q.stop_tokens):std::nullopt;
                            auto result=wide?std::move(*wide):greedy.cycle(pending,m.candidate_depth,static_cast<std::uint32_t>(budget),q.stop_tokens,transaction.get());
                            if(result.emitted_tokens.empty() && !result.selection_failure)throw std::runtime_error("MTP generation made no completed target progress");
                            if(result.selection_receipt){result.selection_receipt->publish();result.selection_receipt.reset();}
                            if(result.selection_failure.has_value()) {
                                out.generated_tokens+=result.emitted_tokens.size();
                                if(!result.emitted_tokens.empty())q.on_tokens(result.emitted_tokens);
                                out.finish="constraint_failure";out.error=std::move(*result.selection_failure);break;
                            }
                            pending=result.final_target;++out.cycles;
                            out.proposed+=result.proposed_candidates;out.accepted+=result.accepted_candidates;
                            if(!wide)++out.chosen_window_rows[std::clamp<std::uint32_t>(result.target_window_rows,1U,8U)];
                            out.generated_tokens+=result.emitted_tokens.size();
                            if(q.backend_probe) q.backend_probe->boundary(serve::BackendProof::work_completion);
                            if(!q.on_tokens(result.emitted_tokens)){out.finish="cancelled";break;}
                            if(is_stop(result.emitted_tokens.back()))out.finish="stop";
                        }
                        if(const auto* ng=greedy.ngram_stats()){out.ngram_rows=ng->extension_rows;out.ngram_accepted=ng->accepted_rows;}
                        if(const auto* cd=greedy.context_stats()){out.ngram_rows=cd->extension_rows;out.ngram_accepted=cd->accepted_rows;}
                        static const bool final_digest=[] {const char* v=std::getenv("STRIX_FINAL_STATE_DIGEST");return v&&v[0]=='1';}();
                        if((state_digest || final_digest) && parents->tiled.cache) {
                            if(window && window->pending_window())window->flush_pending_window();
                            if(hipDeviceSynchronize()!=hipSuccess) throw std::runtime_error("complete final committed state digest");
                            out.final_state_position=session.next_position;
                            out.final_state_digest=model_content_identity_hex(parents->tiled.cache->state_digest(session.next_position));
                            static const bool components=[] {const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST_COMPONENTS");return v&&v[0]=='1';}();
                            if(components) out.final_state_components=parents->tiled.cache->component_digests(session.next_position);
                        }
                        out.paid_drafts=runtime->execution_stats().full_forward_rows-draft_rows_begin;
                        mtp_pool_finish_diagnostics(*parents,*runtime,session,pending);
                        pending={};
                        out.decode_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-decode_start).count();
                    } catch(const serve::ConstraintError& e) {
                        if(session.health==SerialSessionHealth::poisoned || (window && window->failed()))throw;
                        out.finish="constraint_failure";out.error=e.what();pending={};
                    }
                    if(q.constraint) out.constraint=q.constraint->verdict(out.finish);
                    return out;
                };

            const auto run_batch = [&](const MtpServeRunner& named_run, std::mutex* shared_mutex = nullptr, std::function<void()> publish_stats = {}) {
                // Slot 0 is this request's state; every further slot gets the same objects over its own state.
                std::vector<std::unique_ptr<MtpBatchRunner::Slot>> slots;
                std::vector<std::byte*> held_rows;
                struct HeldRowsFree{std::vector<std::byte*>& p;~HeldRowsFree(){for(auto* x:p)(void)hipFree(x);}} held_rows_free{held_rows};
                auto first=std::make_unique<MtpBatchRunner::Slot>();
                first->parents=parents;first->session=&session;first->dispatcher=&dispatcher;
                first->native=native;first->target=target;first->window=window;first->held=tiled_held;
                parents->tiled.window=window.get();
                slots.push_back(std::move(first));
                const auto install_restore = [&](MtpBatchRunner::Slot& slot) {
                    if(slot.parents->tiled.cache) {
                        auto* s=&slot;
                        slot.parents->tiled.cache->set_restore_hook([s,&qualified,&activated](std::uint64_t position){
                            s->runtime->restore_cache_frontier(qualified,activated,position-1U);});
                    }
                };
                const auto prepare_slot = [&](const MtpBatchSlotSetup& setup, std::size_t index) {
                    if(!setup.parents || !setup.session || !setup.dispatcher || !setup.parents->tiled.blocks || !setup.parents->tiled.reset_state ||
                       !tiled_profile_bound(setup.parents->tiled) || setup.parents->prompt.compiled ||
                       setup.parents->tiled.prefill_execution_identity!=parents->tiled.prefill_execution_identity)
                        throw std::invalid_argument("batched MTP slot lacks its parents, session, dispatcher or tiled prompt, "
                            "or its tiled blocks do not run the request's prefill profile");
                    const auto slot_begin=std::chrono::steady_clock::now();
                    auto slot=std::make_unique<MtpBatchRunner::Slot>();
                    slot->parents=setup.parents;slot->session=setup.session;slot->dispatcher=setup.dispatcher;
                    const auto& own=*setup.parents->experiment;
                    slot->native=HipMtpResources::create(model,assets->compiled(),assets->sidecar_manifest(),assets->sidecar_residency(),
                        assets->target_manifest(),assets->target_residency(),assets->shared_target(),native_budget,own.allocator,model.rotary_identity());
                    HipMtpTargetDeferredConfig own_d1;
                    if(deferred_status)own_d1={setup.parents->resources.session_state_layout,&generations};
                    slot->target=HipMtpTargetHidden::create(model,compiled,own.memory,own.model,own.allocator,
                        retained_rows,m.target_capture_bytes,own_d1);
                    if(m.window_rows) {
                        if(!setup.make_window)throw std::invalid_argument("batched MTP slot has no window factory");
                        slot->window=setup.make_window(slot->target);
                        if(!slot->window || slot->window->target_owner()!=slot->target || slot->window->logical_rows()!=m.window_rows)
                            throw std::invalid_argument("batched MTP slot window factory returned an incompatible owner");
                    }
                    if(setup.parents->resources.qsa_committed_pool) slot->native->attach_qsa_pool(setup.parents->resources.qsa_committed_pool);
                    setup.parents->tiled.window=slot->window.get();
                    std::byte* held=nullptr;
                    if(hipMalloc(reinterpret_cast<void**>(&held),static_cast<std::size_t>(width*2U))!=hipSuccess)
                        throw std::runtime_error("batched MTP held hidden allocation failed");
                    held_rows.push_back(held);slot->held=held;
                    if(setup.parents->tiled.cache) extend_cache(*setup.parents->tiled.cache,*slot->native,held);
                    install_restore(*slot);
                    // STRIX_HTTP_SLOT_TIMING=1: MTP resources, target capture and window owners of every slot.
                    static const bool slot_timing=[]{const char* v=std::getenv("STRIX_HTTP_SLOT_TIMING");return v && v[0]=='1';}();
                    if(slot_timing) std::fprintf(stderr,"batched MTP slot %zu prepared in %.0f ms\n",index,
                        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-slot_begin).count());
                    return slot;
                };
                if (!named_run || prefix_batch_cert_execution_mode(options)) install_restore(*slots.front());
                for(const auto& setup:parents->batch_slots) slots.push_back(prepare_slot(setup, slots.size()));
                pending={};
                if (!named_run) runtime.reset();
                // STRIX_BATCH_INTERLEAVE: 1/0 forces block-interleaved admission on/off (default: on when serving).
                const char* interleave_env=std::getenv("STRIX_BATCH_INTERLEAVE");
                const bool interleave=interleave_env?interleave_env[0]=='1':options.http_serve;
                MtpBatchRunner runner(options,model,compiled,qualified,activated,enable_draft_on,std::move(slots),interleave);
                runner.slots_capacity=parents->batch_capacity?parents->batch_capacity:runner.slots();
                runner.growth_memory_ok=parents->batch_memory_ok;
                if(parents->grow_batch_slot) runner.grow_factory=[&] {return prepare_slot(parents->grow_batch_slot(), runner.slots());};
                // Prompt lookup in batched decoding (opt-in, off by default): STRIX_BATCH_NGRAM=1 the n-gram extension
                // (the single request's default source), =ctx the context drafter (--mtp-ngram-draft=ctx); a request's
                // "ngram_draft" field forces it on/off. Solo cycles use it as a single request does; cohort cycles keep
                // only context rows at the chain gate (STRIX_BATCH_NGRAM_GATE, default 0.9). Greedy requests only.
                {
                    const char* mode_env=std::getenv("STRIX_BATCH_NGRAM");
                    const std::string mode=mode_env?mode_env:"";
                    const bool engine_on=mode=="1" || mode=="on" || mode=="ctx";
                    const bool use_context=mode=="ctx" || m.ngram_context;
                    runner.enable_context=[engine_on,use_context,ngram_config,&m](HipMtpGreedyRuntime& greedy,std::span<const std::uint32_t> prompt,
                        const std::shared_ptr<TokenSampler>& request_sampler,std::optional<bool> request) {
                        if(!(request?*request:engine_on) || request_sampler || !greedy.window_verifier() ||
                           greedy.window_verifier()->survival_controller() || !m.draft_chain) return;
                        if(!use_context) {greedy.enable_ngram_draft(prompt,ngram_config());return;}
                        MtpContextDraftConfig context;context.copy_edit=m.ngram_copy_edit;
                        greedy.enable_context_draft(prompt,context,false);
                    };
                }
                runner.calibration={calibration_rows,calibration_ms,replay.observation.maximum_hidden_abs_error,
                    replay.observation.maximum_logit_abs_error};
                runner.session_run = named_run; runner.session_mutex = shared_mutex;
                runner.publish_session_stats=std::move(publish_stats);
                try {parents->serve_batch(runner);}
                catch(...) {runner.abandon();throw;}
                assets->validate_assets_unchanged();
                if(!options.http_serve) return; // the cohort run wrote its own receipt
                const auto& st=runner.stats();
                output<<'{'<<prefill_identity_fields(prefill_options_active())<<",\"mode\":\"native-mtp-batch\",\"slots\":"<<runner.slots()
                    <<",\"slots_built\":"<<runner.slots()<<",\"slots_capacity\":"<<runner.slots_capacity
                    <<",\"lazy_slot_builds\":"<<st.lazy_slot_builds<<",\"lazy_slot_ms\":{\"sum\":"<<st.lazy_slot_ms
                    <<",\"max\":"<<st.lazy_slot_max_ms<<"},\"lazy_slot_memory_declined\":"<<(st.lazy_slot_memory_declined?"true":"false")
                    <<",\"admitted\":"<<st.admitted
                    <<",\"finished\":"<<st.finished<<",\"steps\":"<<st.steps<<",\"cohort_steps\":"<<st.cohort_steps
                    <<",\"solo_steps\":"<<st.solo_steps<<",\"max_active\":"<<st.max_active<<",\"prompt_steps\":"<<st.prompt_steps
                    <<",\"interleaved_admission\":"<<(runner.interleaved()?"true":"false")
                    <<",\"cohort_windows\":"<<st.cohort.cohort_windows<<",\"cohort_window_sessions\":"<<st.cohort.cohort_sessions
                    <<",\"cohort_window_rows\":"<<st.cohort.cohort_rows<<",\"cohort_solo_windows\":"<<st.cohort.solo_windows<<",\"cohort_chunks\":"<<st.cohort.cohort_chunks
                    <<",\"prompt_groups\":"<<st.prompt_groups<<",\"prompt_window_cohorts\":"<<st.prompt_window_cohorts
                    <<",\"inflight_waits\":"<<st.inflight_waits<<",\"inflight_wait_tokens\":"<<st.inflight_wait_tokens
                    <<",\"inflight_restored_tokens\":"<<st.inflight_restored_tokens
                    <<",\"prompt_window_members\":"<<st.prompt_window_members<<","<<batch_service_receipt(st)
                    <<",\"calibration_rows\":"<<calibration_rows<<",\"calibration_ms\":"<<calibration_ms;
                if(prefix_batch_cert_execution_mode(options) && !options.baseline && parents->tiled.cache && parents->tiled.cache->has_store()) {
                    PrefixAsyncCapture::reap();
                    const auto counters=parents->tiled.cache->store().counters();
                    output<<",\"snapshot_preparations\":"<<counters.snapshot_preparations
                          <<",\"staging_bytes\":"<<counters.staging_bytes;
                }
                output<<"}\n";
                return;
            };

            const bool serving = static_cast<bool>(parents->serve) || static_cast<bool>(parents->serve_batch);
            if(options.session_mtp_fixed && (!serving || options.session_management_requested || options.session_live_control)) {
                const auto session_check=[](hipError_t e,const char* what) {
                    if(e!=hipSuccess)throw std::runtime_error(std::string(what)+": "+hipGetErrorString(e));
                };
                if(!parents->tiled.cache || parents->tiled.cache->has_store() || qualify || sampler ||
                   !m.cohort.empty() || !m.copy_wide_rows.empty() || m.ngram_draft ||
                   m.draft_context_vocab || !m.draft_vocab.empty())
                    throw std::invalid_argument("fixed MTP sessions need a private complete registry, full draft vocabulary and no context/copy drafter");
                auto& registry=*parents->tiled.cache;
                registry.enable_fixed_mtp_sessions();
                // The only session controller is fixed-depth, serial target verification. No timing or
                // learned acceptance controller is read; each committed turn starts the same idle protocol.

                SessionCompatibility compat; compat.mtp=true;
                std::string roots;
                std::vector<std::pair<std::string,ModelContentIdentity>> files;
                for(const auto& path:{options.model_path,m.sidecar,m.golden,options.dense_q6p_pack_path,
                        options.dense_trellis_pack_path,options.offline_tuning_cache_path,options.offline_compile_provenance_path}) {
                    if(path.empty())continue;
                    const auto hash=session_cached_file_hash(path);files.emplace_back(path,hash);
                    roots+=std::to_string(path.size())+":"+path+model_content_identity_hex(hash);
                }
                compat.model=sha256_bytes(std::as_bytes(std::span(roots.data(),roots.size())));
                compat.executable=session_cached_file_hash("/proc/self/exe");
                hipDeviceProp_t props{};int rt=0,driver=0,device=0;
                session_check(hipGetDevice(&device),"session current device");
                session_check(hipGetDeviceProperties(&props,device),"session device");
                session_check(hipRuntimeGetVersion(&rt),"session runtime");session_check(hipDriverGetVersion(&driver),"session driver");
                compat.numerical_config=session_numerical_config(options.session_effective_cli,model.rotary_identity())+
                    ";mtp-session-fixed-serial-v1;registry="+model_content_identity_hex(registry.identity())+
                    ";arch="+props.gcnArchName+";name="+props.name+";runtime="+std::to_string(rt)+
                    ";driver="+std::to_string(driver)+";rank=0;world=1";
                std::unique_ptr<HttpSessionStore> store;
                if(options.session_management_requested) store=std::make_unique<HttpSessionStore>(options.session_disk ? options.session_dir : std::string{},
                    options.session_ram_bytes,options.session_max_parked,hip_pinned_prefix_buffer,compat,options.session_disk_bytes);
                std::mutex session_mutex;
                std::optional<SessionCheckpoint> cli_restore;
                std::optional<std::chrono::steady_clock::time_point> disk_restore_started;
                const auto max_envelope=options.context_tokens*(registry.row_bytes()+8U)+registry.fixed_bytes()+(16U<<20);
                if(!options.session_load.empty()) {
                    const auto t=std::chrono::steady_clock::now();disk_restore_started=t;
                    SessionCheckpoint decoded;
                    const auto available=observe_linux_uma_memory();
                    const auto host_limit=session_checkpoint_envelope_limit(options.session_capture_host_bytes,
                        available.memory.observation_identity ? available.memory.mem_available_bytes.value_or(0U) : 0U);
                    (void)read_session_checkpoint(options.session_load,compat,hip_pinned_prefix_buffer,
                        std::min(max_envelope,host_limit),nullptr,&decoded);
                    cli_restore=std::move(decoded);
                    std::cerr<<"session restore_read_verify_ms="<<std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-t).count()<<'\n';
                }
                std::ofstream logit_file;
                std::vector<std::uint32_t> last_history,last_generated;
                SessionFrontier last_frontier;
                bool record_cli=false, http_live_ready=false;
                const auto admit_capture=[&](std::uint64_t rows) {
                    const auto available=observe_linux_uma_memory();
                    const auto bound=session_capture_peak_bound(registry.fixed_bytes()+(16U<<20),
                        registry.row_bytes()+8U,rows,options.session_capture_host_bytes,
                        available.memory.observation_identity ? available.memory.mem_available_bytes.value_or(0U) : 0U);
                    if(store)store->admit(bound);
                };
                const auto capture=[&] {
                    for(const auto& [path,hash]:files) if(session_cached_file_hash(path)!=hash)
                        throw std::runtime_error("session asset changed during execution");
                    if(session_cached_file_hash("/proc/self/exe")!=compat.executable)
                        throw std::runtime_error("session executable changed during execution");
                    admit_capture(session.next_position);
                    std::cerr<<"session checkpoint_capture consumed="<<session.next_position<<'\n';
                    return encode_session_checkpoint(registry.capture_session(session,compat,last_frontier),hip_pinned_prefix_buffer);
                };
                const MtpServeRunner session_run=[&](const MtpServeRequest& q) {
                    std::lock_guard lock(session_mutex);
                    // Publish on every exit, including unnamed turns and refused management calls.
                    const auto stats_guard=std::shared_ptr<void>(nullptr,[&](void*) noexcept {
                        if (store) try {
                            options.state_cache_stats->publish(store->snapshot(),serve::parse_json(store->command("stats","","")));
                        } catch (...) {} // Diagnostics must not mask the request's failure.
                    });
                    if (serving && q.session_id.empty() && q.session_operation.empty() && !options.session_live_control) return ordinary_request(q);
                    if (q.constraint) throw std::invalid_argument("fixed sessions do not support constrained output");
                    if(parents->tiled.cache) parents->tiled.cache->set_retention(q.cache_retention != serve::CacheRetention::no_store, q.session_id);
                    target->replace_token_sampler({});
                    const auto saved_window=parents->tiled.window;
                    // An ordinary request may have left a deferred window on these owners.
                    // Complete it before parking/management, then clear the pending guard exactly
                    // as on request reset, before any checkpoint restore installs complete state.
                    if(saved_window) {saved_window->flush_pending_window();saved_window->begin_session();}
                    // Session registry restores bypass HipPrefixStateCache's restore hook.
                    // Retire both key domains before importing or continuing a fixed session.
                    if(parents->resources.qsa_committed_pool) parents->resources.qsa_committed_pool->begin_epoch();
                    // Fixed sessions never execute windows: captures, digests and live resumes
                    // below therefore observe fully materialized recurrent/conv state.
                    parents->tiled.window=nullptr;
                    const auto window_guard=std::shared_ptr<void>(nullptr,[&](void*){parents->tiled.window=saved_window;});
                    if(session.health==SerialSessionHealth::poisoned)
                        throw std::runtime_error("session lane is poisoned; restart required");
                    MtpServeResult out;
                    out.qsa_mtp_pool_active=static_cast<bool>(parents->resources.qsa_committed_pool);
                    if(!q.session_operation.empty()) {
                        http_live_ready=false;
                        if(!store)throw std::invalid_argument("session management is disabled");
                        if(q.session_operation=="resume") {
                            auto envelope=store->get(q.session_id,true);
                            if(!envelope.data)throw SessionApiError(404,"session_not_found","unknown or evicted session");
                            auto restored=decode_session_checkpoint(envelope,compat);
                            for(auto token:restored.frontier.tokens)if(token>=model.vocab_size)
                                throw std::invalid_argument("session token outside vocabulary");
                            if(restored.frontier.pending && *restored.frontier.pending>=model.vocab_size)
                                throw std::invalid_argument("session pending token outside vocabulary");
                            auto fresh=parents->tiled.fresh;registry.restore_session(fresh,restored,true);
                            pending={};runtime.reset();runtime=std::make_shared<HipMtpRuntime>(native);
                            parents->tiled.reset_state();session=parents->tiled.fresh;
                            registry.restore_session(session,restored);enable_draft(restored.frontier.tokens);
                            runtime->restore_cache_frontier(qualified,activated,restored.frontier.predictor);
                        }
                        out.session_json=store->command(q.session_operation,q.session_id,q.session_durability);
                        return out;
                    }
                    if(!q.session_id.empty()) {
                        if(!store)throw std::invalid_argument("session management is disabled");
                        validate_http_session_id(q.session_id);
                    }
                    if(q.prompt.empty() || !q.max_tokens || !q.on_tokens || q.ngram_draft.value_or(false))
                        throw std::invalid_argument("fixed session needs a nonempty greedy prompt/budget and no context drafter");
                    for(const auto token:q.prompt)if(token>=model.vocab_size)throw std::invalid_argument("session token outside vocabulary");
                    std::optional<SessionCheckpoint> checkpoint;
                    std::vector<std::uint32_t> full(q.prompt.begin(),q.prompt.end());
                    if(cli_restore) {
                        checkpoint=std::move(cli_restore);cli_restore.reset();
                        full=checkpoint->frontier.tokens;
                        if(checkpoint->frontier.pending)full.push_back(*checkpoint->frontier.pending);
                        full.insert(full.end(),q.prompt.begin(),q.prompt.end());
                    } else if(store && !q.session_id.empty()) checkpoint=store->matching(q.session_id,full,q.cache_retention != serve::CacheRetention::no_store);
                    else if(serving && options.session_live_control && http_live_ready &&
                        session_prompt_extends(last_frontier,full)) {
                        // Qualification control: continue the occupied lane directly. No capture,
                        // envelope encoding/decoding or device restore has occurred for this turn.
                        checkpoint=SessionCheckpoint{compat,last_frontier,{}};
                    }
                    http_live_ready=false;
                    if(full.size()>=options.context_tokens || q.max_tokens>options.context_tokens-full.size())
                        throw std::invalid_argument("session prompt plus output exceeds context");
                    for(const auto token:full)if(token>=model.vocab_size)throw std::invalid_argument("restored token outside vocabulary");
                    // Upper bound includes every possible consumed input of this turn. Check
                    // before reset, prefill or generation; repeat at capture against fresh UMA.
                    if(store || !options.session_save.empty()) admit_capture(full.size()+q.max_tokens);
                    if(q.on_admitted) q.on_admitted();
                    const auto started=std::chrono::steady_clock::now();
                    const bool live_boundary=checkpoint && checkpoint->sections.empty();
                    if(live_boundary && serving)
                        std::cerr<<"session boundary_source=never-captured-live consumed="<<session.next_position<<'\n';
                    if(checkpoint && !live_boundary) {
                        auto fresh=parents->tiled.fresh;registry.restore_session(fresh,*checkpoint,true);
                    }
                    pending={};runtime.reset();runtime=std::make_shared<HipMtpRuntime>(native);
                    if(!live_boundary){parents->tiled.reset_state();session=parents->tiled.fresh;}
                    enable_draft(full);
                    std::uint64_t emitted=0;
                    if(checkpoint) {
                        if(!live_boundary)registry.restore_session(session,*checkpoint);
                        emitted=checkpoint->frontier.emitted;out.restored_tokens=session.next_position;
                        runtime->restore_cache_frontier(qualified,activated,checkpoint->frontier.predictor);
                        checkpoint.reset();
                        if(disk_restore_started) {
                            std::cerr<<"session restore_ms="<<std::chrono::duration<double,std::milli>(
                                std::chrono::steady_clock::now()-*disk_restore_started).count()<<'\n';
                            disk_restore_started.reset();
                        }
                    }
                    // HTTP qualification exports one complete file per serialized request.
                    if(serving && !options.session_logits.empty()) {
                        logit_file.open(options.session_logits,std::ios::binary|std::ios::trunc);
                        if(!logit_file)throw std::runtime_error("cannot open HTTP session logits");
                    }
                    std::vector<ModelContentIdentity> row_hashes;
                    target->set_session_logits_observer([&](std::uint64_t position,std::span<const float> row) {
                        if(position+1U<full.size())return;
                        const auto bytes=std::as_bytes(row);row_hashes.push_back(sha256_bytes(bytes));
                        if((record_cli || serving) && logit_file.is_open()) {
                            logit_file.write(reinterpret_cast<const char*>(row.data()),static_cast<std::streamsize>(bytes.size()));
                            if(!logit_file)throw std::runtime_error("session logits write failed");
                        }
                    });
                    auto observer_guard=std::shared_ptr<void>(nullptr,[&](void*){target->set_session_logits_observer({});});
                    std::string receipt;
                    if(out.restored_tokens) {
                        const auto plan=plan_session_suffix(session.next_position,full.size()-session.next_position,parents->tiled.rows,options.session_tiled_suffix);
                        std::cerr<<"session suffix_route="<<(plan.tiled?"aligned-tiled":"serial")<<" alignment_steps="<<plan.serial_rows<<'\n';
                    }
                    {
                    TiledMtpPrompt prompt(full,model,compiled,parents->tiled,session,*target,*runtime,qualified,activated,
                        r,dispatcher,tiled_held,{},{},{},out.restored_tokens!=0,options.session_tiled_suffix);
                    while(!prompt.done())prompt.step();
                    pending=prompt.finish(receipt);
                    }
                    const auto decode_start=std::chrono::steady_clock::now();
                    out.prompt_ms=std::chrono::duration<double,std::milli>(decode_start-started).count();
                    std::vector<std::uint32_t> tokens{pending.predicted_token()};
                    const auto is_stop=[&](std::uint32_t t){return std::find(q.stop_tokens.begin(),q.stop_tokens.end(),t)!=q.stop_tokens.end();};
                    if(q.backend_probe) q.backend_probe->boundary(serve::BackendProof::work_completion);
                    if(!q.on_tokens(tokens))out.finish="cancelled";else if(is_stop(tokens.back()))out.finish="stop";
                    {
                        HipMtpGreedyRuntime fixed(runtime,qualified,activated,target,session,r,dispatcher,parents);
                        // No verifier/controller, ngram, subset vocabulary or adaptive min-p cut.
                        while(out.finish=="length" && tokens.size()<q.max_tokens) {
                            auto result=fixed.cycle(pending,m.candidate_depth,
                                static_cast<std::uint32_t>(std::min<std::uint64_t>(q.max_tokens-tokens.size(),UINT32_MAX)),q.stop_tokens);
                            if(result.emitted_tokens.empty())throw std::runtime_error("MTP session made no progress");
                            pending=result.final_target;++out.cycles;
                            out.proposed+=result.proposed_candidates;out.accepted+=result.accepted_candidates;
                            tokens.insert(tokens.end(),result.emitted_tokens.begin(),result.emitted_tokens.end());
                            if(q.backend_probe) q.backend_probe->boundary(serve::BackendProof::work_completion);
                            if(!q.on_tokens(result.emitted_tokens))out.finish="cancelled";
                            else if(is_stop(tokens.back()))out.finish="stop";
                        }
                    }
                    if(serving && logit_file.is_open()) {
                        logit_file.flush();
                        if(!logit_file)throw std::runtime_error("HTTP session logits flush failed");
                        logit_file.close();
                    }
                    out.generated_tokens=tokens.size();
                    out.decode_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-decode_start).count();
                    last_history=full;last_history.insert(last_history.end(),tokens.begin(),tokens.end()-1);
                    if(last_history.size()!=session.next_position || runtime->next_source_position()+1U!=session.next_position)
                        throw std::runtime_error("MTP session committed frontier differs");
                    session_check(hipMemcpy(tiled_held,reinterpret_cast<const void*>(pending.view().address),
                        static_cast<std::size_t>(width*2U),hipMemcpyDeviceToDevice),"session final target hidden");
                    // Canonical turn boundary: drop ephemeral tickets/proposal counters in ALL arms. No
                    // device row is recomputed. Next turn starts with this held target row and p-1 caches.
                    pending={};runtime.reset();runtime=std::make_shared<HipMtpRuntime>(native);enable_draft(last_history);
                    runtime->restore_cache_frontier(qualified,activated,session.next_position-1U);
                    last_frontier={};last_frontier.consumed=session.next_position;last_frontier.accepted=session.next_position;
                    last_frontier.predictor=session.next_position-1U;last_frontier.tokens=last_history;
                    last_frontier.pending=tokens.back();last_frontier.emitted=emitted+tokens.size();last_generated=tokens;
                    if(store)admit_capture(session.next_position); // existing diagnostic digest scratch admission, independent of retention
                    assets->validate_assets_unchanged();
                    const auto digest=model_content_identity_hex(registry.complete_session_digest(session,last_frontier));
                    serve::Json diagnostics=serve::Json::object(),ids=serve::Json::array(),hashes=serve::Json::array();
                    for(const auto token:tokens)ids.push(serve::Json(static_cast<std::uint64_t>(token)));
                    for(const auto& hash:row_hashes)hashes.push(serve::Json(model_content_identity_hex(hash)));
                    if(row_hashes.size()!=tokens.size())throw std::runtime_error("MTP session did not observe every output head");
                    diagnostics.set("tokens",std::move(ids));diagnostics.set("fp32_logits_sha256",std::move(hashes));
                    diagnostics.set("final_digest",serve::Json(digest));diagnostics.set("accepted",serve::Json(out.accepted));
                    diagnostics.set("proposed",serve::Json(out.proposed));out.session_json=diagnostics.dump();
                    if(store && !q.session_id.empty() && q.cache_retention != serve::CacheRetention::no_store) {
                        if(out.finish=="cancelled")store->invalidate_turn(q.session_id);
                        else {auto envelope=capture();store->park(q.session_id,std::move(envelope));}
                    }
                    if(store) {
                        if(!q.session_id.empty() && q.cache_retention == serve::CacheRetention::no_store) store->no_store_skip();
                    }
                    http_live_ready=serving && options.session_live_control && q.cache_retention != serve::CacheRetention::no_store && q.session_id.empty() && out.finish!="cancelled";
                    std::cerr<<"session "<<(out.restored_tokens?"suffix_ms=":"initial_prefill_ms=")<<out.prompt_ms
                        <<"\nsession "<<(out.restored_tokens?"generation_ms=":"initial_generation_ms=")<<out.decode_ms
                        <<"\nsession final_digest="<<digest<<'\n';
                    return out;
                };
                if(parents->serve_batch) {run_batch(session_run, &session_mutex, [&] {
                    if(store) options.state_cache_stats->publish(store->snapshot(),serve::parse_json(store->command("stats","","")));
                }); return;}
                if(serving) {parents->serve(session_run,options.http_selection_mtp && window && window->policy_selection_admitted());assets->validate_assets_unchanged();return;}
                if(!options.session_logits.empty()) {
                    logit_file.open(options.session_logits,std::ios::binary|std::ios::trunc);
                    if(!logit_file)throw std::runtime_error("cannot open session logits");
                }
                const auto sink=[](std::span<const std::uint32_t>){return true;};
                MtpServeRequest first;first.prompt=options.tokens;first.max_tokens=options.session_first_generate?
                    options.session_first_generate:options.generate_tokens;first.stop_tokens=eos;first.on_tokens=sink;
                record_cli=!options.session_load.empty();
                auto result=session_run(first);
                if(!options.session_save.empty()) {
                    const auto t=std::chrono::steady_clock::now();auto envelope=capture();SessionIoStats io;
                    publish_session_checkpoint(options.session_save,envelope,&io);
                    std::cerr<<"session save_stall_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()
                        <<" write_fsync_ms="<<io.write_fsync_ms<<" bytes="<<io.bytes<<" consumed="<<session.next_position
                        <<" pending="<<*last_frontier.pending<<'\n';
                }
                if(!options.session_then_prompt.empty()) {
                    const auto suffix=read_prefix_cache_prompt(options.session_then_prompt);
                    // Live continuation carries the exact same checkpoint semantics without capture in L.
                    // Re-enter directly on occupied rows, rather than encode/decode a live state.
                    cli_restore=SessionCheckpoint{compat,last_frontier,{}};
                    // An empty section list marks an already-live boundary, handled below.
                    MtpServeRequest next;next.prompt=suffix;next.max_tokens=options.generate_tokens;next.stop_tokens=eos;next.on_tokens=sink;
                    record_cli=true;result=session_run(next);
                }
                if(logit_file.is_open()){
                    logit_file.flush();if(!logit_file)throw std::runtime_error("session logits flush failed");
                    std::cerr<<"session logits_rows="<<last_generated.size()<<" vocabulary="<<model.vocab_size<<'\n';
                }
                output<<"{\"mode\":\"session-mtp-fixed\",\"generated_tokens\":";
                serve::Json ids=serve::Json::array();for(auto t:last_generated)ids.push(serve::Json(static_cast<std::uint64_t>(t)));
                output<<ids.dump()<<",\"accepted\":"<<result.accepted<<",\"proposed\":"<<result.proposed
                    <<",\"session\":"<<result.session_json<<"}\n";
                return;
            }
            if(parents->serve_batch) { run_batch({}); return; }
            if(parents->serve) {
                parents->serve(ordinary_request,options.http_selection_mtp && window && window->policy_selection_admitted());
                assets->validate_assets_unchanged();
                output<<'{'<<prefill_identity_fields(prefill_options_active())<<",\"mode\":\"native-mtp-http-serve\",\"served_requests\":"<<served
                    <<",\"calibration_rows\":"<<calibration_rows<<",\"calibration_ms\":"<<calibration_ms;
                if(prefix_batch_cert_execution_mode(options) && !options.baseline && parents->tiled.cache && parents->tiled.cache->has_store()) {
                    PrefixAsyncCapture::reap();
                    const auto counters=parents->tiled.cache->store().counters();
                    output<<",\"snapshot_preparations\":"<<counters.snapshot_preparations
                          <<",\"staging_bytes\":"<<counters.staging_bytes;
                }
                output<<"}\n";
                return;
            }
            pending=run_tiled_mtp_prompt(options.tokens,model,compiled,parents->tiled,session,*target,*runtime,qualified,activated,
                r,dispatcher,tiled_held,tiled_receipt);
        } else
        // Prefill handoff: every later target row, window and checkpoint runs the decode execution
        // from the state this prefill produced. The pending prompt row is re-minted with the
        // handed-off session; the lineage keeps the prefill identity of every prompt row.
        {
            if(serial_split)pending=target->hand_off_prompt(session,pending,parents->prompt.resources,r);
            enable_draft(options.tokens);
        }
        if (auto* lookup = dynamic_cast<FileMappedPleLookupBackend*>(parents->ple.get()))
            if (auto turn = lookup->residency()) {
                cancel_ple_lookup_prefetch(*parents->ple, parents->prefetch);
                turn->phase(false);
            }
        if(parents->tiled.cache && prompt_state_digest_request()) {
            if(window)window->flush_pending_window();
            std::fprintf(stderr,"qwen-target-state phase=prompt position=%llu sha256=%s\n",
                static_cast<unsigned long long>(session.next_position),
                model_content_identity_hex(parents->tiled.cache->target_state_digest(session.next_position,options.tokens,pending.predicted_token())).c_str());
        }
        if(separated_prompt) complete_mtp_prompt_phase(
            [&] {
                // Includes the target owner's detached work, ring consumers, terminal
                // head adoption and the predictor stream's final tap/held-row reads.
                if(hipDeviceSynchronize()!=hipSuccess) throw std::runtime_error("MTP prompt retirement drain failed");
                if(runtime->prime_hidden_rows_pending()) throw std::logic_error("MTP prompt priming still pending");
            },parents->retire_prompt,allocate_window);
        HipMtpGreedyRuntime greedy(runtime,qualified,activated,target,session,r,dispatcher,parents);
        if(window && !qualify)greedy.use_window_verifier(window);
        if(m.draft_chain)greedy.set_draft_min_probability(mtp_draft_min_probability(m));
        if(!qualify && !m.ngram_off && ngram_wanted(m.ngram_draft))enable_ngram(greedy,options.tokens);
        if(qualify){
            MtpDecodeQualificationContext context{options,parents,native,target,std::move(window),session,dispatcher,generations,greedy,*runtime,std::move(pending)};
            qualify(context);
            assets->validate_assets_unchanged();
            return;
        }
        // The final prompt target already selected this unconsumed pending
        // token. Emit it once, then each cycle returns only subsequent outputs.
        const auto draft_rows_begin=runtime->execution_stats().full_forward_rows;
        pending.begin_step_logits_digest();
        publish_yarn_output([&] { strix::startup::first_token(); generated.push_back(pending.predicted_token()); });
        const auto start=std::chrono::steady_clock::now();
        while(generated.size()<options.generate_tokens && generated.back()!=eos.front()) {
            check_yarn_memory();
            PrefixAsyncCapture::reap();
            if(session.next_position>=options.context_tokens){stop="context-capacity";break;}
            const auto budget=static_cast<std::uint32_t>(options.generate_tokens-generated.size());
            auto wide=wide_copy?wide_copy->cycle(greedy,pending,budget,eos):std::nullopt;
            const auto result=wide?std::move(*wide):greedy.cycle(pending,sampler && !m.sample_speculation?0U:m.candidate_depth,budget,eos);
            if(result.emitted_tokens.empty())throw std::runtime_error("MTP generation made no completed target progress");
            publish_yarn_output([&] { generated.insert(generated.end(),result.emitted_tokens.begin(),result.emitted_tokens.end()); });
            pending=result.final_target;++cycles;proposed+=result.proposed_candidates;accepted+=result.accepted_candidates;
            target_rows+=result.consumed_target_rows;window_attempts+=result.attempted_target_window;window_accepts+=result.accepted_target_window;
            window_restores+=result.target_checkpoint_restored;r2_dispatches+=result.target_r2_dispatches;
            if(!wide)++chosen_rows[std::clamp<std::uint32_t>(result.target_window_rows,1U,8U)]; // wide cycles: mtp_ngram_draft
        }
        const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(generated.back()==eos.front())stop="eos";
        if (auto* lookup = dynamic_cast<FileMappedPleLookupBackend*>(parents->ple.get()))
            if (auto turn = lookup->residency()) {
                cancel_ple_lookup_prefetch(*parents->ple, parents->prefetch);
                turn->finish();
            }
        mtp_pool_finish_diagnostics(*parents,*runtime,session,pending);
        const char* final_digest=std::getenv("STRIX_FINAL_STATE_DIGEST");
        const char* prefix_digest=std::getenv("STRIX_PREFIX_CACHE_DIGEST");
        if(((final_digest && final_digest[0]=='1') || (prefix_digest && prefix_digest[0]=='1')) && parents->tiled.cache) {
            if(window)window->flush_pending_window();
            if(hipDeviceSynchronize()!=hipSuccess)throw std::runtime_error("MTP CLI final state diagnostic completion");
            std::fprintf(stderr,"mtp-final-state position=%llu predictor_end=%llu lineage=%llu sha256=%s\n",
                static_cast<unsigned long long>(session.next_position),static_cast<unsigned long long>(runtime->next_source_position()),
                static_cast<unsigned long long>(session.arithmetic_lineage.history_identity),
                model_content_identity_hex(parents->tiled.cache->state_digest(session.next_position)).c_str());
        }
        if(parents->tiled.cache && prompt_state_digest_request()) {
            if(window)window->flush_pending_window();
            auto consumed=options.tokens;
            if(!generated.empty()) consumed.insert(consumed.end(),generated.begin(),generated.end()-1);
            std::fprintf(stderr,"qwen-target-state phase=final position=%llu sha256=%s\n",
                static_cast<unsigned long long>(session.next_position),
                model_content_identity_hex(parents->tiled.cache->target_state_digest(session.next_position,consumed,
                    generated.empty()?std::optional<std::uint32_t>{}:std::optional{generated.back()})).c_str());
        }
        assets->validate_assets_unchanged();
        const auto native_stats=runtime->execution_stats();
        const auto draft_stats=runtime->draft_head()?runtime->draft_head()->stats():HipMtpDraftHeadStats{};
        std::optional<HipKnownPrefixUnionObservation> union_observation;
        if(window)for(const auto& owner:window->window_owners())
            if(auto observed=owner->latest_routed_union_observation())union_observation=observed;
        if(m.window_expert_union && window_attempts &&
            (!parents->window_setup.union_source_steps || !union_observation))
            throw std::logic_error("MTP expert union completed no actual native group for an attempted window");
        std::ostringstream union_receipt;
        union_receipt << "  \"mtp_window_expert_union_requested\":" << (m.window_expert_union ? "true" : "false")
            << ",\n  \"mtp_window_expert_union_source_steps\":" << parents->window_setup.union_source_steps
            << ",\n  \"mtp_window_expert_union_workspace_budget_bytes\":" << m.window_expert_union_bytes
            << ",\n  \"mtp_window_expert_union_workspace_reserved_bytes\":" << parents->window_setup.union_reserved_bytes
            << ",\n  \"mtp_window_expert_union_workspace_generation\":" << parents->window_setup.union_generation
            << ",\n  \"mtp_window_arena_completion_union_retained_bytes\":" << parents->window_setup.retained_window_bytes
            << ",\n  \"mtp_window_expert_union_latest_observation\":";
        write_mtp_union_observation(union_receipt, union_observation);
        union_receipt << ",\n  \"mtp_window_arms\":[";
        if(window) {
            bool comma=false;
            for(const auto& owner:window->window_owners()) {
                union_receipt << (comma?",":"") << owner->arithmetic_witness(); comma=true;
            }
        }
        union_receipt << "],\n";
        const auto mtp_d1_inventory=options.execution_inventory
            ? std::string("  \"mtp_target_d1_inventory\":{\"boundary\":\"after-mtp-target-setup-and-request-completion\",\"present\":")+
                (deferred_status?"true":"false")+",\"storage\":\""+(deferred_status?"unaccounted":"absent")+
                "\",\"included_in_base_runtime_backing_inventory\":false,\"physical_fit\":null,"
                "\"scope\":\"actual-target-private-D1-record-and-host-shadow-presence-not-allocation-census\"},\n"
            : std::string{};
        const auto completed_receipt=yarn_memory_receipt()+(completion_receipt?completion_receipt():std::string{});
        publish_yarn_output([&] {
        output<<"{\n"<<prefill_identity_fields(prefill_options_active())<<",\n"<<completed_receipt<<mtp_d1_inventory<<union_receipt.str()<<"  \"mode\":\"native-mtp-independent-replay-then-target-verified-greedy\",\n"
            <<"  \"mtp_native_hardware_qualified\":false,\n  \"mtp_performance_qualified\":false,\n"
            <<"  \"optimization_profile_selected\":\""<<decode_receipt::escape_json_string(resolved_profile)<<"\",\n"
            <<"  \"mtp_target_verification\":\""<<mtp_target_verification_name(window_attempts)<<"\",\n"
            <<"  \"mtp_target_window_configured\":"<<(window?"true":"false")<<",\n"
            <<"  \"mtp_requested_target_window_rows\":"<<m.window_rows<<",\n"
            <<"  \"mtp_reference_independence\":\""<<(golden.independent_source_content==mtp_self_generated_golden_source()?
                "self-generated-native-forward-not-independent":"externally-supplied-provenance-pinned-not-inferred-from-digest")<<"\",\n"
            <<"  \"mtp_golden_file_sha256\":\""<<model_content_identity_hex(mtp_cli_digest(m.golden_sha256))<<"\",\n"
            <<"  \"mtp_golden_payload_sha256\":\""<<model_content_identity_hex(replay.golden_payload)<<"\",\n"
            <<"  \"mtp_independent_source_content\":\""<<model_content_identity_hex(golden.independent_source_content)<<"\",\n"
            <<"  \"mtp_independent_reference_identity\":"<<golden.independent_reference_identity<<",\n"
            <<"  \"mtp_sidecar_file_sha256\":\""<<model_content_identity_hex(assets->sidecar_mapping()->file_sha256())<<"\",\n"
            <<"  \"mtp_sidecar_content_identity\":\""<<model_content_identity_hex(assets->shared_target().sidecar_model_content_identity)<<"\",\n"
            <<"  \"mtp_target_file_sha256\":\""<<model_content_identity_hex(actual.mapped->file_sha256())<<"\",\n"
            <<"  \"mtp_target_content_identity\":\""<<model_content_identity_hex(assets->shared_target().target_model_content_identity)<<"\",\n"
            <<"  \"mtp_source_identity\":"<<assets->source().source_identity<<",\n"
            <<"  \"mtp_source_repository\":\""<<decode_receipt::escape_json_string(m.repository)<<"\",\n"
            <<"  \"mtp_source_revision\":\""<<decode_receipt::escape_json_string(m.revision)<<"\",\n"
            <<"  \"mtp_source_path\":\""<<decode_receipt::escape_json_string(m.source_path)<<"\",\n"
            <<"  \"mtp_converter_reference_sha256\":\""<<model_content_identity_hex(mtp_cli_digest(m.converter_sha256))<<"\",\n"
            <<"  \"mtp_norm_converter_reference_sha256\":\""<<model_content_identity_hex(mtp_cli_digest(m.norm_converter_sha256))<<"\",\n"
            <<"  \"target_prefill_numerical_profile\":\""<<numerical_profile_name(pair.prefill)<<"\",\n"
            <<"  \"target_decode_numerical_profile\":\""<<numerical_profile_name(pair.decode)<<"\",\n"
            <<"  \"target_prefill_execution_identity\":"<<(serial_split?parents->prompt.compiled->execution_identity:
                tiled_prompt && split?parents->tiled.prefill_execution_identity:compiled.execution_identity)<<",\n"
            <<"  \"mtp_golden_target_profile_pair_recorded\":"<<(golden.target_decode_profile_identity?"true":"false")<<",\n"
            <<"  \"target_execution_identity\":"<<compiled.execution_identity<<",\n"
            <<"  \"target_kernel_plan_identity\":"<<compiled.kernel_plan_identity<<",\n"
            <<"  \"mtp_structural_plan_identity\":"<<assets->compiled().plan_identity<<",\n"
            <<"  \"mtp_semantic_identity\":"<<assets->compiled().forward_semantics.contract_identity<<",\n"
            <<"  \"mtp_native_arithmetic_profile_identity\":"<<runtime->numerical_profile_identity()<<",\n"
            <<"  \"mtp_qualification_identity\":"<<qualified.qualification_identity<<",\n  \"mtp_activation_identity\":"<<activated.activation_identity<<",\n"
            <<"  \"mtp_replay_policy_identity\":"<<policy.policy_identity<<",\n  \"mtp_replay_requires_exact_tokens\":true,\n"
            <<"  \"mtp_replay_hidden_error_limit\":"<<policy.max_hidden_abs_error<<",\n  \"mtp_replay_logit_error_limit\":"<<policy.max_logit_abs_error<<",\n"
            <<"  \"mtp_replay_maximum_hidden_abs_error\":"<<replay.observation.maximum_hidden_abs_error<<",\n"
            <<"  \"mtp_replay_maximum_logit_abs_error\":"<<replay.observation.maximum_logit_abs_error<<",\n"
            <<"  \"mtp_replay_rows\":"<<replay.rows.size()<<",\n  \"mtp_replay_passed_declared_policy\":true,\n"
            <<"  \"mtp_replay_native_steps\":"<<mtp_cli_mul(replay.rows.size(),assets->compiled().steps.size())<<",\n"
            <<"  \"mtp_requested_candidate_depth\":"<<m.candidate_depth<<",\n  \"mtp_target_capture_rows\":"<<retained_rows<<",\n"
            <<"  \"mtp_normalized_rounding\":\""<<m.normalized_rounding<<"\",\n  \"mtp_projection_rounding\":\""<<m.projection_rounding<<"\",\n"
            <<"  \"mtp_runtime_retained_payload_bytes\":"<<native->retained_payload_bytes()<<",\n"
            <<"  \"mtp_head_recipe\":"<<static_cast<std::uint32_t>(select_hip_mtp_lm_head_recipe(model,native->tensors().lm_head))<<",\n"
            <<"  \"mtp_head_profile_identity\":"<<hip_mtp_lm_head_profile_identity(model,native->tensors().lm_head)<<",\n"
            <<"  \"mtp_native_full_forward_rows\":"<<native_stats.full_forward_rows<<",\n"
            <<"  \"mtp_cache_only_rows\":"<<native_stats.cache_only_rows<<",\n"
            <<"  \"mtp_reused_cache_rows\":"<<native_stats.cache_reused_rows<<",\n"
            <<"  \"mtp_native_row_counter_scope\":\"completed-rows-including-prompt-independent-replay-and-generation\",\n"
            <<"  \"mtp_window_retained_payload_bytes\":"<<(window?window->retained_payload_bytes():0U)<<",\n"
            <<"  \"mtp_target_capture_budget_bytes\":"<<m.target_capture_bytes<<",\n  \"mtp_target_d1\":"<<(deferred_status?"true":"false")<<",\n"
            <<[]{  // GSQ-RCO routed experts (STRIX_GSQ_Q20_A8=1): prompt rows x layers per prefill kernel form, when any ran
                const auto& g = strix::moe_q4k_a8::gsq_prefill_rows();
                const auto f = g.f16wmma.load(), b = g.f16wmma_batch128.load(), i = g.iu8wmma.load(), d = g.dot4.load();
                return f || i || d ? "  \"prefill_moe_gsq_rows\": {\"f16wmma\": " + std::to_string(f) + ", \"f16wmma_batch128\": " +
                    std::to_string(b) + ", \"iu8wmma\": " + std::to_string(i) + ", \"dot4\": " + std::to_string(d) + "},\n" : std::string{};
            }()
            <<"  \"mtp_cycles\":"<<cycles<<",\n  \"mtp_proposed_candidates\":"<<proposed<<",\n  \"mtp_accepted_candidates\":"<<accepted<<",\n"
            <<"  \"mtp_consumed_generation_input_rows\":"<<target_rows<<",\n  \"mtp_attempted_target_windows\":"<<window_attempts<<",\n"
            <<"  \"mtp_accepted_target_windows\":"<<window_accepts<<",\n  \"mtp_restored_target_windows\":"<<window_restores<<",\n"
            <<"  \"mtp_target_r2_dispatches\":"<<r2_dispatches<<",\n"
            <<"  \"mtp_target_r2_dispatches_scope\":\"actual-pair-entry-dispatches-only-excludes-wide-R4-R8\",\n"
            <<"  \"mtp_target_wide_dispatches\":null,\n"
            <<"  \"mtp_target_wide_dispatches_scope\":\"not-exposed-by-current-verifier-result-not-inferred-from-geometry\",\n"
            <<"  \"mtp_target_window_rows_set\":[";
        if(window){const auto stats=window->owner_stats();for(std::size_t s=1;s<stats.size();++s)output<<(s>1?",":"")<<stats[s].rows;}
        output<<"],\n  \"mtp_spec_min_p\":"<<mtp_draft_min_probability(m)<<",\n"
            <<"  \"mtp_window_controller\":\""<<(window && window->depth_controller()?"depth-v1":window && window->survival_controller()?"survival-v1":"min-p")<<"\",\n"
            <<"  \"mtp_window_option_ms_ema\":[";
        if(window){const auto stats=window->option_stats();
            for(std::size_t s=0;s<stats.size();++s)output<<(s?",":"")<<"{\"rows\":"<<stats[s].rows<<",\"active\":"<<stats[s].active
                <<",\"samples\":"<<stats[s].samples<<",\"ema_ms\":"<<stats[s].ms_ema<<"}";}
        output<<"],\n  \"mtp_depth_domain\":\""<<(window && window->depth_controller()?mtp_depth_domain():"inactive")<<"\",\n"
            <<"  \"mtp_depth_non_draft_ms\":[";
        if(window){const auto stats=window->depth_cost_stats();
            for(std::size_t s=0;s<stats.size();++s)output<<(s?",":"")<<"{\"rows\":"<<stats[s].rows<<",\"filled_drafts\":"<<stats[s].filled
                <<",\"samples\":"<<stats[s].samples<<",\"ms\":"<<stats[s].ms<<",\"probes\":"<<stats[s].probes<<"}";}
        output<<"],\n  \"mtp_depth_draft_ms\":[";
        if(window){const auto stats=window->draft_depth_stats();
            for(std::size_t s=0;s<stats.size();++s)output<<(s?",":"")<<"{\"depth\":"<<stats[s].filled
                <<",\"samples\":"<<stats[s].samples<<",\"ms\":"<<stats[s].ms<<"}";}
        output<<"],\n"
            <<"  \"mtp_chosen_window_rows_histogram\":{";
        for(std::size_t t=0,first=1;t<chosen_rows.size();++t)if(chosen_rows[t]){output<<(first?"":",")<<"\""<<t<<"\":"<<chosen_rows[t];first=0;}
        output<<"},\n  \"mtp_chosen_window_rows_scope\":\"verifier-choice-per-cycle-1-is-serial-target-steps\",\n"
            <<"  \"mtp_window_owner_cycle_ms_ema\":{";
        if(window){const auto stats=window->owner_stats();
            for(std::size_t s=0;s<stats.size();++s)output<<(s?",":"")<<"\""<<stats[s].rows<<"\":{\"cycles\":"<<stats[s].cycles<<",\"ema_ms\":"<<stats[s].cycle_ms_ema
                <<",\"probes\":"<<stats[s].probes<<"}";}
        output<<"},\n"
            <<"  \"mtp_window_cost_model\":\""<<(window && window->depth_controller()?"depth-robust-v1":window && !window->robust_cost_model()?"legacy-ema":"robust-median-reprobe-v1")<<"\",\n"
            <<"  \"mtp_draft_chain\":"<<(m.draft_chain?"true":"false")<<",\n"
            <<"  \"mtp_draft_cache_profile\":\""<<runtime->draft_cache_profile()<<"\",\n"
            <<"  \"mtp_draft_exact_kernels\":"<<(m.draft_exact?"true":"false")<<",\n"
            <<"  \"mtp_draft_q4_head\":"<<(m.draft_q4?"true":"false")<<",\n"
            <<"  \"mtp_ngram_draft\":"<<[&]{
                if(const auto* c=greedy.context_stats())
                    return "{\"mode\":\""+std::string(m.ngram_wide_only?"wide":(m.ngram_copy_edit?"copy":"ctx"))+"\",\"proposal_cycles\":"+std::to_string(c->cycles)+
                        ",\"selected_cycles\":"+std::to_string(c->selected_cycles)+",\"selected_rows\":"+std::to_string(c->selected_rows)+
                        ",\"copy_first_cycles\":"+std::to_string(c->copy_first_cycles)+",\"replaced_cycles\":"+std::to_string(c->replaced_cycles)+
                        ",\"copy_first_declined\":"+std::to_string(c->copy_first_declined)+
                        ",\"extended_cycles\":"+std::to_string(c->extended_cycles)+",\"extension_rows\":"+std::to_string(c->extension_rows)+
                        ",\"accepted_rows\":"+std::to_string(c->accepted_rows)+",\"shadow_scored_cycles\":"+std::to_string(c->scored_cycles)+
                        ",\"shadow_right_rows\":"+std::to_string(c->shadow_right_rows)+",\"learning_rows\":"+std::to_string(c->learning_rows)+
                        ",\"wide_rows\":"+std::to_string(m.copy_wide_rows.empty()?0U:m.copy_wide_rows.front())+",\"wide_cycles\":"+std::to_string(c->wide_cycles)+",\"wide_blocked\":"+std::to_string(c->wide_blocked)+
                        ",\"edit_cuts\":"+std::to_string(c->edit_cuts)+",\"chain_cycles\":"+std::to_string(c->chain_cycles)+
                        ",\"chain_first_cycles\":"+std::to_string(c->chain_first_cycles)+",\"chain_rows\":"+std::to_string(c->chain_rows)+
                        ",\"chain_accepted_rows\":"+std::to_string(c->chain_accepted_rows)+",\"chain_established\":"+std::to_string(c->chain_established)+
                        ",\"wide_rows_accepted\":"+std::to_string(c->wide_rows_accepted)+",\"wide_abandoned\":"+std::to_string(c->wide_abandoned)+
                        ",\"wide_abandoned_rows_wasted\":"+std::to_string(wide_copy?wide_copy->stats().abandoned_rows_wasted:0U)+
                        ",\"wide_ms_committed\":"+std::to_string(wide_copy?wide_copy->stats().ms_committed:0.0)+
                        ",\"wide_ms_abandoned\":"+std::to_string(wide_copy?wide_copy->stats().ms_abandoned:0.0)+
                        ",\"wide_sizes\":"+[&]{
                            // One object per configured wide size (largest first); the keys above are their totals.
                            std::string sizes="[";
                            if(wide_copy) for(const auto& size:wide_copy->sizes())
                                sizes+=std::string(sizes.size()>1U?",":"")+"{\"rows\":"+std::to_string(size.rows)+
                                    ",\"cycles\":"+std::to_string(size.stats.cycles)+",\"rows_accepted\":"+std::to_string(size.stats.rows_accepted)+
                                    ",\"abandoned\":"+std::to_string(size.stats.abandoned)+
                                    ",\"abandoned_rows_wasted\":"+std::to_string(size.stats.abandoned_rows_wasted)+
                                    ",\"ms_committed\":"+std::to_string(size.stats.ms_committed)+
                                    ",\"ms_abandoned\":"+std::to_string(size.stats.ms_abandoned)+",\"abandon_rows\":["+[&]{
                                        std::string rows;
                                        for(const auto k:size.stats.abandon_rows) rows+=(rows.empty()?"":",")+std::to_string(k);
                                        return rows;
                                    }()+"]}";
                            return sizes+"]";
                        }()+"}";
                const auto* s=greedy.ngram_stats();
                if(!s) return std::string("null");
                return "{\"proposal_cycles\":"+std::to_string(s->cycles)+",\"extended_cycles\":"+std::to_string(s->extended_cycles)+
                    ",\"extension_rows\":"+std::to_string(s->extension_rows)+",\"accepted_rows\":"+std::to_string(s->accepted_rows)+
                    ",\"shadow_scored_cycles\":"+std::to_string(s->scored_cycles)+",\"shadow_right_rows\":"+std::to_string(s->shadow_right_rows)+"}";
            }()<<",\n"
            <<"  \"mtp_draft_head_rows\":"<<draft_stats.rows<<",\n  \"mtp_draft_base_rows\":"<<draft_stats.base_rows<<",\n"
            <<"  \"mtp_draft_context_rows\":"<<draft_stats.context_rows<<",\n  \"mtp_draft_context_dropped\":"<<draft_stats.context_dropped<<",\n"
            <<"  \"sampling\":"<<mtp_sampling_receipt(m,sampler.get())<<",\n"
            <<"  \"mtp_paid_drafts\":"<<(native_stats.full_forward_rows-draft_rows_begin)<<",\n"
            <<"  \"mtp_paid_drafts_scope\":\"completed-predictor-full-forward-rows-during-generation-excludes-prompt-and-calibration\",\n"
            <<"  \"emitted_tokens\":"<<generated.size()<<",\n"
            <<"  \"generation_ms\":"<<elapsed<<",\n"
            <<"  \"mtp_generation_cycle_elapsed_ms\":"<<elapsed<<",\n"
            <<"  \"mtp_timing_scope\":\"post-prompt-and-independent-replay-cycles-only-host-completed-wall-not-GPU-kernel-time\",\n"
            <<(tiled_prompt ? "  \"mtp_tiled_prompt\":{"+tiled_receipt+",\"block_numerical_profile\":\""+
                std::string(numerical_profile_name(pair.prefill))+"\",\"serial_numerical_profile\":\""+
                std::string(numerical_profile_name(pair.decode))+"\",\"calibration_rows\":"+std::to_string(calibration_rows)+
                ",\"calibration_ms\":"+std::to_string(calibration_ms)+"},\n" : std::string{})
            <<"  \"prefix_tokens_consumed\":"<<options.tokens.size()<<",\n  \"final_position\":"<<session.next_position<<",\n"
            <<"  \"stop_reason\":\""<<stop<<"\",\n  \"generated_tokens\":[";
        for(std::size_t i=0;i<generated.size();++i){if(i)output<<',';output<<generated[i];}
        output<<"]\n}\n";
        });
    } catch(...) {
        if(runtime)runtime->cancel();
        if (auto* lookup = dynamic_cast<FileMappedPleLookupBackend*>(parents->ple.get()))
            if (auto turn = lookup->residency()) {
                cancel_ple_lookup_prefetch(*parents->ple, parents->prefetch);
                turn->cancel();
            }
        throw;
    }
}
} // namespace strix::decode_cli
