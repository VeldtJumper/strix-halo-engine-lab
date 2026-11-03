#include "strix/host_phase_timing.hpp"
#include "strix/hip_mtp_window_verifier.hpp"
#include "strix/mtp_payload_geometry.hpp"
#include "strix/mtp_target_window_policy.hpp"
#include "strix/mtp_window_controller.hpp"
#include "strix/mtp_cost_table_session.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include "strix/arithmetic_lineage.hpp"
#include "strix/session_state_memory.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_token_sampling.hpp"
#include "strix/dispatch_resources.hpp"
#include <functional>
#include <vector>
#include "strix/hip_token_sampling_device.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <optional>
#include <stdexcept>
namespace strix {
namespace {
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
void write_position_word(const ExecutionRuntimeResources& resources,std::uint64_t value) {
    for(const auto& allocation:resources.session_state_layout->allocations) {
        if(allocation.component!=SessionStateComponent::position_history) continue;
        require(allocation.storage==SessionStateStorage::host && allocation.bytes>=sizeof(value) &&
            resources.session_state_memory->host_base,"MTP prefix commit requires the host position word");
        std::memcpy(reinterpret_cast<void*>(resources.session_state_memory->host_base+allocation.offset),&value,sizeof(value));
        return;
    }
    throw std::invalid_argument("MTP prefix commit found no position history");
}
bool session_equal(const SerialSessionContext& a,const SerialSessionContext& b) {
    return a.execution_identity==b.execution_identity && a.state_layout_identity==b.state_layout_identity &&
        a.runtime_compatibility_identity==b.runtime_compatibility_identity && a.next_position==b.next_position &&
        a.committed_tokens==b.committed_tokens && a.health==b.health && a.state.valid==b.state.valid &&
        a.state.generations==b.state.generations && a.arithmetic_lineage.history_identity==b.arithmetic_lineage.history_identity &&
        a.arithmetic_lineage.tainted==b.arithmetic_lineage.tainted && a.arithmetic_lineage.intervals==b.arithmetic_lineage.intervals;
}
// STRIX_MTP_REQUEST_RESET=0 keeps the controllers' estimates across requests.
bool mtp_request_reset() noexcept {
    static const bool reset=[]{const char* v=std::getenv("STRIX_MTP_REQUEST_RESET");return !v || v[0]!='0';}();
    return reset;
}
// Cost-table device key: architecture, marketing name, compute units, memory size, clock, HIP runtime and driver.
std::string hip_cost_device_identity() {
    int device=0,runtime=0,driver=0;hipDeviceProp_t p{};
    if(hipGetDevice(&device)!=hipSuccess || hipGetDeviceProperties(&p,device)!=hipSuccess ||
        hipRuntimeGetVersion(&runtime)!=hipSuccess || hipDriverGetVersion(&driver)!=hipSuccess) return {};
    return std::string(p.gcnArchName)+'|'+p.name+"|cu"+std::to_string(p.multiProcessorCount)+"|mem"+
        std::to_string(p.totalGlobalMem)+"|clk"+std::to_string(p.clockRate)+"|hip"+std::to_string(runtime)+"|drv"+std::to_string(driver);
}
}
bool hip_mtp_survival_controller() noexcept {
    static const bool enabled=[]{const char* v=std::getenv("STRIX_MTP_CONTROLLER");return v && std::string_view(v)=="survival";}();
    return enabled;
}
namespace {
double env_number(const char* name,double fallback) {const char* v=std::getenv(name);return v && *v?std::atof(v):fallback;}
double clamped_logit(double p) {p=std::clamp(p,1e-6,1.0-1e-6);return std::log(p/(1.0-p));}
double sigmoid(double z) {return 1.0/(1.0+std::exp(-std::clamp(z,-30.0,30.0)));}
// P(draft k equals the target's greedy token | drafts 0..k-1 did): a logistic over the draft's own
// probability, its margin over the runner-up and its depth. The draft probability is a softmax over the draft
// vocabulary, not an acceptance rate. Prior weights are fitted offline on full-chain logs (every start
// position drafted to depth 7 on EN/ES/FR/code/agent prompts, Q4 draft, auto vocabulary); the history of
// earlier probabilities added nothing there once depth is known. Online, only the bias adapts to the
// recent workload (SGD on verified drafts, pulled back to the prior).
struct SurvivalCalibrator {
    static constexpr std::size_t width=5;
    // bias, logit p1, log(p1/p2) capped at 12, depth 1, depth >= 2
    static constexpr std::array<double,width> prior{0.1956,1.0131,-0.4434,-0.3505,-0.7112};
    // Acceptance of a draft not made yet, from its depth and the previous draft's probability:
    // bias, logit p(k-1), depth 1, depth >= 2.
    static constexpr std::array<double,4> lookahead{0.3911,0.1133,0.1019,-0.1015};
    double bias=prior[0],rate=0.0,pull=0.0;
    static std::array<double,width> features(std::size_t k,float p1,float p2) {
        const double margin=std::min(12.0,std::log(std::max(static_cast<double>(p1),1e-6)/std::max(static_cast<double>(p2),1e-6)));
        return {1.0,clamped_logit(p1),margin,k==1?1.0:0.0,k>=2?1.0:0.0};
    }
    double probability(std::size_t k,float p1,float p2) const {
        const auto x=features(k,p1,p2);double z=bias;
        for(std::size_t f=1;f<width;++f) z+=prior[f]*x[f];
        return sigmoid(z);
    }
    void update(std::size_t k,float p1,float p2,bool right) {
        if(rate>0.0) bias-=rate*(probability(k,p1,p2)-(right?1.0:0.0)+pull*(bias-prior[0]));
    }
    static double next(std::size_t k,float previous) {
        return sigmoid(lookahead[0]+lookahead[1]*clamped_logit(previous)+(k==1?lookahead[2]:0.0)+(k>=2?lookahead[3]:0.0));
    }
};
}
bool hip_mtp_full_chain_diagnostic() noexcept {
    static const bool enabled=[]{const char* v=std::getenv("STRIX_MTP_DIAG_FULL_CHAIN");return v && v[0]=='1';}();
    return enabled;
}
struct HipMtpWindowVerifier::Impl {
    struct Owner {
        std::shared_ptr<HipKnownPrefixWindow> window;
        std::array<std::shared_ptr<HipMtpTargetWindowCapture>,4> captures;
        std::uint32_t rows=0;
        std::shared_ptr<void> greedy_partials,greedy_replies;
        bool warmed=false; // first window of an owner pays one-time admission and first-touch costs
    };
    std::shared_ptr<HipMtpTargetHidden> target;
    std::vector<Owner> owners; // ascending rows
    std::shared_ptr<HipKnownPrefixWindow> window; // largest owner; the checkpoint path has exactly one
    ExecutionRuntimeResources resources;
    std::shared_ptr<const void> resource_owner;
    std::unique_ptr<HipMtpTargetCheckpoint> checkpoint;
    std::uint64_t retained=0,maximum_history_intervals=0;
    bool busy=false,failed=false,bounded=false,prefix=false;
    // Min-p controller (D-022): owner cost per cycle (draft included), serial cost per emitted token,
    // per-bin acceptance; host logic in MtpMinPWindowController.
    float min_probability=0.0F;
    // Candidate rows from `external_first` on were proposed by another source (context n-grams) with
    // their own learned acceptance: used to size the next window, never fed to the draft calibration.
    std::size_t external_first=SIZE_MAX;
    std::vector<double> external;
    bool timing_independent=false;
    std::vector<float> sampled_logits;
    HipMtpTargetWindowCapture::TransactionSelection transaction_selection(MtpSelectionTransaction& transaction,
        void* stream,std::span<const std::uint32_t> candidates,std::uint32_t rows,std::uint32_t budget,
        std::span<const std::uint32_t> eos) {
        return [&transaction,stream,candidates,rows,budget,eos,host=std::vector<float>{},vocab=target->model().vocab_size](
            std::span<const ExecutionBufferAddress> logits,std::span<const HipGreedyRowsReply> replies) mutable {
            return walk_mtp_selection(candidates,rows,budget,eos,[&](std::size_t r) {
                const auto t0=std::chrono::steady_clock::now();
                struct Timed {MtpSelectionTransaction& t;std::chrono::steady_clock::time_point t0;
                    ~Timed(){t.note_row_ns(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-t0).count()));}}
                    timed{transaction,t0};
                if(r<replies.size()) if(const auto known=transaction.select_known(replies[r].token,replies[r].status)) return *known;
                read_hip_logits_row(logits[r],static_cast<std::uint32_t>(vocab),stream,host);
                return transaction.select(host);
            });
        };
    }
    std::unique_ptr<MtpMinPWindowController> minp;
    std::unique_ptr<MtpDepthWindowController> depth;
    std::unique_ptr<HipTokenSamplerDevice> device_sampler; // sampling-v1 on the GPU (STRIX_SAMPLE_PATH=host: host)
    // Sampled rows: decided in order from row 0 (the token at `first`) until the first row without a
    // draft, a rejection, EOS or `n` rows; later rows are never accepted. Spans must outlive the window.
    HipMtpTargetWindowCapture::RowSelection sampled_selection(std::shared_ptr<TokenSampler> sampler,std::uint64_t first,void* stream,
        std::uint32_t rows,std::uint32_t n,std::span<const TokenDraft> drafts,std::span<const std::uint32_t> eos) {
        return [this,sampler=std::move(sampler),first,stream,rows,n,drafts,eos](
            std::span<const ExecutionBufferAddress> logits,std::span<std::uint32_t> predictions) {
            const auto vocabulary=target->model().vocab_size;
            const bool timed=sample_timing_enabled();
            double readback=0.0;
            const auto start=std::chrono::steady_clock::now();
            std::vector<TokenDecision> decisions;
            if(device_token_sampling_enabled(sampler->policy())) {
                // Every row on the device at once; the stopping rule of TokenSampler::decide_rows on the results.
                if(!device_sampler) device_sampler=std::make_unique<HipTokenSamplerDevice>(static_cast<std::uint32_t>(vocabulary),8U);
                std::vector<DeviceSampleRow> device_rows(n);
                for(std::uint32_t r=0;r<n;++r) device_rows[r]={logits[r],first+r,r<drafts.size()?&drafts[r]:nullptr};
                const auto all=device_sampler->decide(*sampler,device_rows,stream);
                for(std::uint32_t r=0;r<n;++r) {
                    decisions.push_back(all[r]);sampler->count(all[r]);
                    if(!device_rows[r].draft || !all[r].accepted || std::find(eos.begin(),eos.end(),all[r].token)!=eos.end()) break;
                }
            } else decisions=sampler->decide_rows(n,first,drafts,eos,[&](std::uint32_t r) {
                const auto t0=std::chrono::steady_clock::now();
                read_hip_logits_row(logits[r],static_cast<std::uint32_t>(vocabulary),stream,sampled_logits);
                if(timed) readback+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
                return std::span<const float>(sampled_logits);});
            if(timed) record_sample_timing(SampleTimingKind::window,rows,readback,
                std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()-readback,
                static_cast<std::uint32_t>(decisions.size()));
            for(std::size_t r=0;r<decisions.size();++r) predictions[r]=decisions[r].token;
        };
    }
    Owner* last=nullptr; // owner of the last prefix window; its stream carries the PLE commit
    // Survival controller (STRIX_MTP_CONTROLLER=survival). An option is an owner and its active rows: with
    // STRIX_MTP_ACTIVE_ROWS=1 the R4 owner also verifies 3 active rows and the R8 owner 6 (the rest are
    // padding outside the expert union and acceptance). Costs are separated: per option the cycle time
    // minus the draft chain (verify, commit, reconcile and host work), per draft step, per serial row.
    struct Option {Owner* owner=nullptr;std::uint32_t active=0;std::uint64_t cycles=0,last_used=0,reprobe_after=0;double cost_ms_ema=0.0;bool probing=false;};
    bool survival=false;
    std::vector<Option> options; // ascending active rows
    SurvivalCalibrator calibrator;
    double explore=0.0; // one more draft after a first draft whose calibrated acceptance is below this
    double draft_ms_ema=0.0,serial_row_ms_ema=0.0;std::uint64_t draft_samples=0,serial_rows=0;
    double rho=0.0; // long-run tokens per ms (EMA over cycles): the price of a millisecond in survival_best
    // An option's cost is only sampled when it is chosen, so one slow sample (page faults, a stalled host) could
    // price it out for the rest of the run. An option whose estimate is implausible (above a larger option's, or
    // for the largest one, more than 1.25x the next smaller one scaled by rows) and unused for reprobe_cycles
    // cycles is chosen once more; that sample replaces a higher estimate. A probe that confirms the estimate
    // doubles the option's interval (STRIX_MTP_REPROBE=N cycles, 0 = never).
    std::uint64_t cycle_index=0,reprobe_cycles=64;
    double chain_ms=0.0;std::uint32_t chain_steps=0; // the current cycle's draft chain (note_draft)
    std::optional<std::chrono::steady_clock::time_point> last_end;
    // Persisted min-p controller costs (cost_table_directory_from_env plus the app's resolved cost_table_configuration; the
    // D-022 controller only; MtpCostTableSession). Controller input only: the costs choose which owner verifies;
    // the table is used only when every owner passes the policy-selection (serial-equivalence) guard.
    // The table is opened at the first decode cycle after the configuration is resolved (cold until then).
    bool cost_tables=false,cost_baseline=false;
    std::shared_ptr<const std::string> cost_configuration;
    std::vector<std::uint32_t> cost_rows; // 1 (serial) and every owner (all admitted), else table off
    std::unique_ptr<MtpCostTableSession> costs;
    // Before every draft decision and verify: resolve the session (greedy, context bucket) for the controller.
    void prepare_costs(const SerialSessionContext& session) {
        if(!cost_tables) return;
        if(!costs) {
            if(!cost_configuration || cost_configuration->empty()) return;
            cost_tables=false; // one attempt
            const auto directory=cost_table_directory_resolve(cost_baseline).path;
            if(!directory) return;
            CostTableKey key;
            key.engine="qwen-mtp-minp";
            key.binary=cost_table_executable_identity();
            key.device=hip_cost_device_identity();
            key.model=model_content_identity_hex(resources.model_weights->model_content_identity);
            std::string configuration=*cost_configuration+";owners=";
            for(const auto r:cost_rows) configuration+=std::to_string(r)+',';
            configuration+=";min-p="+std::to_string(min_probability)+";cost="+
                (minp->model()==MtpWindowCostEstimate::Model::legacy?"legacy":"robust");
            key.policy=cost_table_policy_identity(configuration);
            if(key.binary.empty() || key.device.empty()) return;
            costs=std::make_unique<MtpCostTableSession>(*directory,std::move(key),cost_table_options_from_env(),cost_rows);
            cost_tables=true;
        }
        costs->prepare(*minp,session.next_position,target->token_sampler()!=nullptr);
    }
    void flush_costs() noexcept {if(costs) costs->flush(*minp);}
    ~Impl() {flush_costs();}
    Impl(std::shared_ptr<HipMtpTargetHidden> t,std::vector<std::shared_ptr<HipKnownPrefixWindow>> w,
         const ExecutionRuntimeResources& r,std::shared_ptr<const void> owner,
         HipMtpWindowVerificationPolicy policy,AllocationGenerationSource& generations)
        :target(std::move(t)),resources(r),resource_owner(std::move(owner)),
         maximum_history_intervals(policy.capture.maximum_history_intervals),bounded(policy.bounded_window_rollback),
         min_probability(policy.min_probability),timing_independent(policy.timing_independent_schedule) {
        require(target && resource_owner && resources.session_state_layout && !w.empty() &&
            resources.session_state_memory==&target->memory_owner()->state() &&
            resources.scratch==&target->memory_owner()->scratch() &&
            resources.model_weights==&target->weights_owner()->binding() &&
            min_probability>=0.0F && min_probability<=1.0F,"MTP window verifier requires actual target owners");
        for(const auto& candidate:w) require(!!candidate,"MTP window verifier requires actual closed all-row target owners");
        std::sort(w.begin(),w.end(),[](const auto& a,const auto& b){return a->logical_rows()<b->logical_rows();});
        for(const auto& candidate:w) {
            require(candidate->output_policy()==KnownWindowOutputPolicy::all_rows &&
                (candidate->logical_rows()==2 || candidate->logical_rows()==4 || candidate->logical_rows()==8),
                "MTP window verifier requires actual closed all-row target owners");
            require(owners.empty() || (owners.back().rows<candidate->logical_rows() && candidate->defers_state() &&
                owners.front().window->defers_state()),"MTP window owners must have distinct rows and prefix acceptance");
            if(target->model().rotary_identity().descriptor().mode()==RotaryMode::yarn)
                require(candidate->policy_selection_admitted(),
                    "YaRN MTP verifier arithmetic is not serial equivalent; refused before mutation");
            if(target->model().rotary_identity().descriptor().mode()==RotaryMode::yarn)
                std::fprintf(stderr,"mtp-window-arms %s\n",candidate->arithmetic_witness().c_str());
            owners.push_back({candidate,{},candidate->logical_rows(),{},{},false});
        }
        window=owners.back().window;
        prefix=window->defers_state();
        {
            // STRIX_MTP_WINDOW_COST=legacy: EMA cost estimates without re-probes (the D-022 behaviour before D-042).
            const char* cost=std::getenv("STRIX_MTP_WINDOW_COST");
            const auto model=cost && std::string_view(cost)=="legacy"?MtpWindowCostEstimate::Model::legacy:MtpWindowCostEstimate::Model::robust;
            std::vector<std::uint32_t> rows;for(const auto& o:owners) rows.push_back(o.rows);
            minp=std::make_unique<MtpMinPWindowController>(std::move(rows),min_probability,model,64,timing_independent);
        }
        if(prefix && mtp_depth_controller_enabled() && !timing_independent) {
            require(!target->token_sampler(),"depth MTP controller requires greedy selection");
            std::vector<std::uint32_t> rows;for(const auto& o:owners)rows.push_back(o.rows);
            depth=std::make_unique<MtpDepthWindowController>(std::move(rows),min_probability,mtp_depth_domain_prior());
        }
        // The survival controller prices options by measured time: never with a timing-independent schedule.
        survival=prefix && hip_mtp_survival_controller() && !timing_independent;
        if(survival) {
            const bool active_rows=env_number("STRIX_MTP_ACTIVE_ROWS",0.0)!=0.0;
            for(auto& o:owners) {
                if(active_rows && (o.rows==4U || o.rows==8U)) options.push_back({&o,o.rows==4U?3U:6U});
                options.push_back({&o,o.rows});
            }
            explore=env_number("STRIX_MTP_EXPLORE",0.0);
            reprobe_cycles=static_cast<std::uint64_t>(std::max(0.0,env_number("STRIX_MTP_REPROBE",64.0)));
            if(env_number("STRIX_MTP_SURVIVAL_ADAPT",1.0)!=0.0) {calibrator.rate=0.01;calibrator.pull=0.01;}
        }
        // Persisted costs: the default min-p controller with measured costs only (no depth/survival controller,
        // no timing-independent schedule, no diagnostics), per-request resets (the default), and every owner admitted
        // as serial-equivalent by the policy-selection guard.
        cost_configuration=policy.cost_table_configuration;
        cost_baseline=policy.cost_table_baseline;
        cost_tables=cost_configuration && prefix && min_probability>0.0F && !timing_independent && !depth && !survival &&
            !hip_mtp_full_chain_diagnostic() && mtp_request_reset();
        if(cost_tables) {
            const auto resolved=cost_table_directory_resolve(cost_baseline);
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true))
                std::fprintf(stderr,"cost-table: directory source=%s%s%s\n",cost_table_source_name(resolved.source),
                    resolved.path?" dir=":"",resolved.path?resolved.path->string().c_str():"");
            cost_tables=resolved.path.has_value();
        }
        if(cost_tables) {
            // Every selectable owner must pass the policy-selection guard, or the verifier declines the table.
            std::vector<std::uint32_t> rows;std::vector<bool> admitted;
            for(const auto& o:owners) {rows.push_back(o.rows);admitted.push_back(o.window->policy_selection_admitted());}
            cost_rows=MtpCostTableSession::seedable_rows(rows,admitted);
            cost_tables=!cost_rows.empty();
        }
        for(std::size_t o=1;o<owners.size();++o) owners[o].window->share_deferred(*owners.front().window);
        // STRIX_WINDOW_ROUTE_LOG=1 (diagnostics, never timed): after each router step of a verify window, the
        // routed expert ids of every row ("route pos=P rows=R row=r layer=L ids=..." on stderr). Synchronizes.
        if(env_number("STRIX_WINDOW_ROUTE_LOG",0.0)!=0.0) for(auto& o:owners) {
            const auto top_k=static_cast<std::size_t>(target->model().num_experts_per_token);
            o.window->set_window_step_observer([rows=o.rows,top_k](std::size_t,const std::function<ExecutionDispatchContext(std::uint32_t)>& row) {
                const auto first=row(0);
                if(first.op!=SerialOp::moe_route) return;
                require(hipDeviceSynchronize()==hipSuccess,"route log fence");
                for(std::uint32_t r=0;r<rows;++r) {
                    const auto c=row(r);
                    const auto out=resolve_dispatch_write(c,0);
                    std::vector<std::uint32_t> ids(std::min<std::size_t>(top_k,static_cast<std::size_t>(out.bytes/4U)));
                    require(hipMemcpy(ids.data(),reinterpret_cast<const void*>(out.address),ids.size()*4U,hipMemcpyDeviceToHost)==hipSuccess,"route log copy");
                    std::fprintf(stderr,"route pos=%llu rows=%u row=%u layer=%u ids=",static_cast<unsigned long long>(c.position),
                        rows,r,c.layer.value_or(UINT32_MAX));
                    for(std::size_t k=0;k<ids.size();++k) std::fprintf(stderr,k?",%u":"%u",ids[k]);
                    std::fputs(" end\n",stderr);
                }
            });
        }
        const auto reservation=bounded?plan_hip_mtp_target_window_rollback(target->model(),*resources.session_state_layout,0,window->logical_rows(),true).reservation:
            plan_hip_mtp_target_checkpoint(target->model(),*resources.session_state_layout,policy.snapshot_policy);
        const auto checkpoint_payload=prefix?0U:reservation.retained_bytes;
        auto upper=checkpoint_payload;
        for(const auto& o:owners) upper=mtp_native::sum(upper,mtp_native::product(4U,
            mtp_payload::window_capture(target->model(),o.rows,policy.capture.maximum_history_intervals,resources.model_weights->tensors.size())));
        require(policy.total_retained_byte_budget && checkpoint_payload<=policy.checkpoint_retained_byte_budget &&
            upper<=policy.total_retained_byte_budget,"MTP window verification exceeds upfront combined retained reservation");
        // Every row's argmax runs on the window stream right after its head rows and lands in
        // host-mapped memory: one completion per window instead of a selector round trip per
        // row. STRIX_MTP_GREEDY_ROWS=0 keeps the per-row selectors.
        const char* greedy_rows=std::getenv("STRIX_MTP_GREEDY_ROWS");
        if(!greedy_rows || std::string_view(greedy_rows)!="0") for(auto& o:owners) {
            const auto vocab=static_cast<std::uint32_t>(target->model().vocab_size);
            const auto bytes=hip_greedy_rows_scratch_bytes(o.rows,vocab);
            void* partials=nullptr;void* replies=nullptr;
            require(hipMalloc(&partials,bytes)==hipSuccess,"MTP greedy rows scratch");
            o.greedy_partials=std::shared_ptr<void>(partials,[](void* p){(void)hipFree(p);});
            require(hipHostMalloc(&replies,sizeof(HipGreedyRowsReply)*o.rows,hipHostMallocMapped|hipHostMallocCoherent)==hipSuccess,
                "MTP greedy rows replies");
            o.greedy_replies=std::shared_ptr<void>(replies,[](void* p){(void)hipHostFree(p);});
            ExecutionBufferAddress view{};view.address=reinterpret_cast<std::uintptr_t>(partials);view.bytes=bytes;
            o.window->attach_greedy_rows(view,static_cast<HipGreedyRowsReply*>(replies));
        }
        for(auto& o:owners) for(auto& capture:o.captures) {
            capture=o.window->prepare_mtp_target_capture(policy.capture);
            target->validate_window_capture(*capture);
            retained=mtp_native::sum(retained,capture->retained_payload_bytes());
        }
        // Completed serial metadata admits capture, but bounded rollback belongs
        // to the actual known-window producer queue and its retained owner.
        // Full snapshots retain the original serial-target queue contract.
        if(prefix) return; // deferred GDN + PLE snapshots replace the state checkpoint
        const auto& captures=owners.back().captures;
        const auto checkpoint_stream=bounded?captures.front()->source_stream():target->stream();
        const std::shared_ptr<const void> checkpoint_stream_owner=bounded?
            std::shared_ptr<const void>(window):std::shared_ptr<const void>(target);
        checkpoint=std::make_unique<HipMtpTargetCheckpoint>(target->model(),target->compiled(),*resources.session_state_layout,
            target->memory_owner(),target->weights_owner(),target->allocator(),static_cast<hipStream_t>(checkpoint_stream),
            checkpoint_stream_owner,target->guard(),policy.snapshot_policy,policy.checkpoint_retained_byte_budget,generations,
            bounded?captures.front():std::shared_ptr<const HipMtpTargetWindowCapture>{},resources.qsa_committed_pool);
        retained=mtp_native::sum(retained,checkpoint->reserved_bytes());
        require(retained<=policy.total_retained_byte_budget,"MTP actual verification pools exceed admitted reservation");
    }
    // Survival controller. Calibrated acceptance of each draft so far.
    std::vector<double> survival_acceptance(std::span<const float> p,std::span<const float> p2) const {
        std::vector<double> out(p.size());
        for(std::size_t k=0;k<p.size();++k) out[k]=calibrator.probability(k,p[k],k<p2.size()?p2[k]:0.0F);
        return out;
    }
    // Unmeasured options are priced just under the cheapest measured one, so each is tried.
    double option_cost(const Option& o) const {
        if(o.cycles) return o.cost_ms_ema;
        const Option* cheapest=nullptr;
        for(const auto& m:options) if(m.cycles && (!cheapest || m.cost_ms_ema<cheapest->cost_ms_ema)) cheapest=&m;
        return cheapest?0.9*cheapest->cost_ms_ema:1.0;
    }
    // Best option for drafts with acceptance P (n=P.size() drafts paid): E[N] = 1 + sum_j prod_{i<j} P_i over
    // the option's active rows against C = option cost + n draft steps. The value is E[N] - rho*C, rho the
    // long-run rate (a cycle is worth its tokens minus the tokens its time would earn at that rate), or
    // E[N]/C before the first cycle. Serial target steps earn one token per serial row cost.
    // Returns the option (nullptr = serial) and its value.
    template<class Fits> std::pair<Option*,double> survival_best(std::span<const double> P,Fits fits) {
        std::vector<double> expected(P.size()+1U,1.0);double run=1.0;
        for(std::size_t j=0;j<P.size();++j) {run*=P[j];expected[j+1U]=expected[j]+run;}
        const double drafts=draft_samples?draft_ms_ema*static_cast<double>(P.size()):0.0;
        const auto value=[&](double tokens,double ms){return rho>0.0?tokens-rho*ms:tokens/ms;};
        Option* best=nullptr;double best_value=serial_rows?value(expected.back(),serial_row_ms_ema*expected.back()+drafts):0.0;
        for(auto& o:options) {
            if(!fits(o)) continue;
            const auto v=value(expected[std::min<std::size_t>(o.active-1U,P.size())],option_cost(o)+drafts);
            if(v>best_value || (!best && !serial_rows)) {best=&o;best_value=v;}
        }
        return {best,best_value};
    }
    // Keep drafting while some longer chain (future drafts at the lookahead acceptance) has a better best
    // option than the drafts so far; after a weak first draft, one exploration step regardless.
    bool survival_keep(std::span<const float> p,std::span<const float> p2) {
        if(p.empty()) return true;
        auto P=survival_acceptance(p,p2);
        const auto any=[](const Option&){return true;};
        const auto now=survival_best(P,any).second;
        const auto next=SurvivalCalibrator::next(p.size(),p.back());
        for(std::size_t m=p.size();m<owners.back().rows-1U;++m) {
            P.push_back(next);
            if(survival_best(P,any).second>now) return true;
        }
        return p.size()==1U && P.front()<explore;
    }
    bool implausible(const Option& o) const {
        const Option* smaller=nullptr;
        for(const auto& m:options) {
            if(!m.cycles || &m==&o) continue;
            if(m.active>o.active && m.cost_ms_ema<o.cost_ms_ema) return true;
            if(m.active<o.active && (!smaller || m.active>smaller->active)) smaller=&m;
        }
        const bool largest=std::none_of(options.begin(),options.end(),[&](const Option& m){return m.cycles && m.active>o.active;});
        return largest && smaller && o.cost_ms_ema>1.25*smaller->cost_ms_ema*o.active/smaller->active;
    }
    template<class Fits> Option* survival_choose(std::span<const float> p,std::span<const float> p2,std::size_t candidates,Fits fits) {
        ++cycle_index;
        const auto use=[&](Option* o){if(o) o->last_used=cycle_index;return o;};
        // Options never measured and holding the drafts are tried first, largest first.
        for(auto o=options.rbegin();o!=options.rend();++o) if(!o->cycles && o->active<=candidates+1U && fits(*o)) return use(&*o);
        if(reprobe_cycles) for(auto o=options.rbegin();o!=options.rend();++o)
            if(o->cycles && cycle_index-o->last_used>std::max(o->reprobe_after,reprobe_cycles) && implausible(*o) && fits(*o)) {
                o->probing=true;return use(&*o);}
        return use(survival_best(survival_acceptance(p,p2),fits).first);
    }
    void survival_calibrate(std::span<const float> p,std::span<const float> p2,std::size_t accepted_drafts,std::size_t verified_rows) {
        for(std::size_t j=0;j<p.size() && j<verified_rows && j<=accepted_drafts;++j)
            calibrator.update(j,p[j],j<p2.size()?p2[j]:0.0F,j<accepted_drafts);
    }
    void survival_observe(Option* option,std::size_t emitted) {
        const auto now=std::chrono::steady_clock::now();
        if(last_end && emitted) {
            const auto ms=std::chrono::duration<double,std::milli>(now-*last_end).count();
            static const bool log_cycles=std::getenv("STRIX_CYCLE_LOG")!=nullptr;
            if(log_cycles) std::fprintf(stderr,"cycle rows=%u emitted=%zu ms=%.2f draft_ms=%.2f drafts=%u\n",
                option?option->active:1U,emitted,ms,chain_ms,chain_steps);
            constexpr double alpha=0.25;
            auto update=[&](double& ema,std::uint64_t& count,double value){
                ema=count?ema+alpha*(std::min(value,2.0*ema)-ema):value;++count;};
            if(chain_steps) update(draft_ms_ema,draft_samples,chain_ms/chain_steps);
            const auto rate=static_cast<double>(emitted)/std::max(ms,1e-3);
            rho=rho>0.0?rho+0.1*(rate-rho):rate;
            const auto rest=std::max(ms-chain_ms,0.0);
            if(option) {
                // An owner's first window pays one-time admission and first touch: not a cost sample.
                if(option->owner->warmed) {
                    if(option->probing) {
                        option->reprobe_after=rest<0.9*option->cost_ms_ema?reprobe_cycles:2U*std::max(option->reprobe_after,reprobe_cycles);
                        option->cost_ms_ema=std::min(option->cost_ms_ema,rest);++option->cycles;
                    }
                    else update(option->cost_ms_ema,option->cycles,rest);
                }
                option->owner->warmed=true;option->probing=false;
            } else update(serial_row_ms_ema,serial_rows,rest/static_cast<double>(emitted));
        }
        last_end=now;chain_ms=0.0;chain_steps=0;
    }
    // nullptr = serial target steps.
    template<class Fits> Owner* choose(std::span<const float> p,std::size_t candidates,Fits fits) {
        std::array<std::uint8_t,8> mask{};
        for(std::size_t o=0;o<owners.size();++o) mask[o]=fits(owners[o])?1U:0U;
        const auto admitted=std::span<const std::uint8_t>(mask.data(),owners.size());
        const auto chosen=depth?depth->choose(p,candidates,admitted):minp->choose(p,candidates,admitted);
        return chosen==MtpMinPWindowController::serial?nullptr:&owners[chosen];
    }
    // A cohort cycle is not a sample of any owner's cost; the next cycle measures from here.
    void skip_sample() {last_end=std::chrono::steady_clock::now();}
    Owner* owner_of(std::uint32_t rows) {for(auto& o:owners) if(o.rows==rows) return &o;return nullptr;}
    bool fits(const SerialSessionContext& session,const Owner& o,std::size_t candidates,std::uint32_t budget) const {
        const auto capacity=std::min(resources.session_state_layout->context_tokens,target->model().effective_max_positions());
        if(session.next_position>=capacity) return false;
        const bool pad=prefix && min_probability>0.0F;
        // Capture::begin needs a valid lineage: a window never starts an empty session.
        const bool history=!session.arithmetic_lineage.intervals.empty() && session.arithmetic_lineage.intervals.size()<maximum_history_intervals;
        return budget>=o.rows && capacity-session.next_position>=o.rows && history &&
            (pad?candidates!=0U:candidates+1U>=o.rows);
    }
    // Wall time between consecutive verified cycles, attributed to this cycle's choice.
    void observe(Owner* owner,std::size_t emitted) {
        const auto now=std::chrono::steady_clock::now();
        static const bool log_cycles=std::getenv("STRIX_CYCLE_LOG")!=nullptr;
        if(log_cycles && emitted) {
            // Per window (diagnostics): rows, emitted tokens, wall ms (-1 = first cycle, untimed), draft steps, the
            // controller's cost in use for this option before this sample, its own samples, live table priors.
            const auto stats=minp->owner_stats();
            const auto& st=stats.at(owner?static_cast<std::size_t>(owner-owners.data())+1U:0U);
            std::fprintf(stderr,"cycle rows=%u emitted=%zu ms=%.2f draft=%u est=%.2f local=%llu priors=%zu\n",owner?owner->rows:1U,
                emitted,last_end?std::chrono::duration<double,std::milli>(now-*last_end).count():-1.0,chain_steps,st.cycle_ms,
                static_cast<unsigned long long>(st.cycles),minp->active_priors());
        }
        if(last_end && emitted) {
            const auto ms=std::chrono::duration<double,std::milli>(now-*last_end).count();
            minp->observe(owner?static_cast<std::size_t>(owner-owners.data()):MtpMinPWindowController::serial,emitted,ms);
            if(depth)depth->observe(owner?static_cast<std::size_t>(owner-owners.data()):MtpDepthWindowController::serial,emitted,ms);
        }
        last_end=now;
    }
};
HipMtpWindowVerifier::HipMtpWindowVerifier(std::shared_ptr<HipMtpTargetHidden> target,
    std::shared_ptr<HipKnownPrefixWindow> window,const ExecutionRuntimeResources& resources,
    std::shared_ptr<const void> owner,HipMtpWindowVerificationPolicy policy,AllocationGenerationSource& generations)
    :HipMtpWindowVerifier(std::move(target),std::vector<std::shared_ptr<HipKnownPrefixWindow>>{std::move(window)},
        resources,std::move(owner),policy,generations){}
HipMtpWindowVerifier::HipMtpWindowVerifier(std::shared_ptr<HipMtpTargetHidden> target,
    std::vector<std::shared_ptr<HipKnownPrefixWindow>> windows,const ExecutionRuntimeResources& resources,
    std::shared_ptr<const void> owner,HipMtpWindowVerificationPolicy policy,AllocationGenerationSource& generations)
    :impl_(std::make_unique<Impl>(std::move(target),std::move(windows),resources,std::move(owner),policy,generations)){}
HipMtpWindowVerifier::~HipMtpWindowVerifier()=default;
std::shared_ptr<HipMtpTargetHidden> HipMtpWindowVerifier::target_owner() const noexcept {return impl_->target;}
std::shared_ptr<HipKnownPrefixWindow> HipMtpWindowVerifier::window_owner() const noexcept {return impl_->window;}
std::vector<std::shared_ptr<HipKnownPrefixWindow>> HipMtpWindowVerifier::window_owners() const {
    std::vector<std::shared_ptr<HipKnownPrefixWindow>> out;for(const auto& o:impl_->owners)out.push_back(o.window);return out;
}
std::vector<HipMtpWindowOwnerStats> HipMtpWindowVerifier::owner_stats() const {
    std::vector<HipMtpWindowOwnerStats> out;
    for(const auto& o:impl_->minp->owner_stats()) out.push_back({o.rows,o.cycles,o.cycle_ms,o.probes});
    return out;
}
bool HipMtpWindowVerifier::robust_cost_model() const noexcept {return impl_->depth || impl_->minp->model()==MtpWindowCostEstimate::Model::robust;}
std::uint32_t HipMtpWindowVerifier::logical_rows() const noexcept {return impl_->window->logical_rows();}
void HipMtpWindowVerifier::prepare_admission(const SerialSessionContext& session) {
    // STRIX_WINDOW_SETUP_PREFLIGHT=0 leaves each owner's admission to its first window.
    static const bool enabled=[]{const char* v=std::getenv("STRIX_WINDOW_SETUP_PREFLIGHT");return !v || v[0]!='0';}();
    auto& i=*impl_;
    if(!enabled || i.busy || failed()) return;
    const auto capacity=std::min(i.resources.session_state_layout->context_tokens,i.target->model().effective_max_positions());
    for(auto& o:i.owners) {
        if(session.next_position+o.rows>capacity) continue;
        const std::vector<std::uint32_t> tokens(o.rows,0U);
        o.window->preflight_admission(session,tokens);
    }
}
std::uint32_t HipMtpWindowVerifier::cohort_rows(const SerialSessionContext& session,std::span<const std::uint32_t> candidates,
    std::span<const float> probabilities,std::uint32_t budget) const {
    const auto& i=*impl_;
    if(!i.prefix || i.busy || failed() || candidates.empty()) return 0U;
    std::uint32_t want=2U;
    if(i.min_probability>0.0F && probabilities.size()==candidates.size()) {
        std::uint32_t lead=1U;
        while(lead-1U<probabilities.size() && probabilities[lead-1U]>=i.min_probability) ++lead;
        want=std::max(want,lead);
    } else want=std::max<std::uint32_t>(want,static_cast<std::uint32_t>(candidates.size())+1U);
    for(const auto& o:i.owners) if(o.rows>=want && i.fits(session,o,candidates.size(),budget)) return o.rows;
    for(auto o=i.owners.rbegin();o!=i.owners.rend();++o) if(i.fits(session,*o,candidates.size(),budget)) return o->rows;
    return 0U;
}
bool HipMtpWindowVerifier::cohort_fits(const SerialSessionContext& session,std::uint32_t rows,std::size_t candidates,
    std::uint32_t budget) const {
    auto& i=*impl_;
    const auto* owner=i.owner_of(rows);
    return i.prefix && !i.busy && !failed() && owner && i.fits(session,*owner,candidates,budget);
}
std::shared_ptr<HipKnownPrefixWindow> HipMtpWindowVerifier::owner_with_rows(std::uint32_t rows) const {
    auto* owner=impl_->owner_of(rows);
    return owner?owner->window:nullptr;
}
std::vector<HipMtpVerifiedTargetRows> HipMtpWindowVerifier::verify_cohort(std::span<const HipMtpCohortVerifyRequest> requests,
    HipKnownPrefixWindow& primary) {
    HipKnownPrefixWindow* const primaries[]={&primary};
    const std::size_t chunks[]={requests.size()};
    return verify_cohort(requests,primaries,chunks);
}
std::vector<HipMtpVerifiedTargetRows> HipMtpWindowVerifier::verify_cohort(std::span<const HipMtpCohortVerifyRequest> requests,
    std::span<HipKnownPrefixWindow* const> primaries,std::span<const std::size_t> chunks) {
    require(requests.size()>=2U,"MTP cohort verification requires two or more sessions");
    struct Prepared {
        Impl* impl=nullptr;Impl::Owner* owner=nullptr;
        std::vector<std::uint32_t> inputs;
        SerialSessionContext previous;
        std::shared_ptr<HipMtpTargetWindowCapture> capture;
    };
    struct DiscardAttempts {std::span<const HipMtpCohortVerifyRequest> requests;
        ~DiscardAttempts(){for(const auto& q:requests)if(q.transaction)q.transaction->discard();}} discard_attempts{requests};
    std::vector<Prepared> prepared(requests.size());
    std::vector<HipKnownPrefixCohortMember> members(requests.size());
    if(std::any_of(requests.begin(),requests.end(),[](const auto& q){return q.transaction!=nullptr;}))
        for(const auto* primary:primaries)require(!primary || primary->policy_selection_admitted(),"MTP cohort primary arithmetic is not serial equivalent");
    // The selections borrow the requests' spans: never leave them attached past the call.
    struct ClearSelections {
        std::vector<Prepared>& p;
        ~ClearSelections(){for(auto& x:p) if(x.capture) {x.capture->set_row_selection({});x.capture->set_transaction_selection({});}}
    } clear_selections{prepared};
    for(std::size_t a=0;a<requests.size();++a) {
        const auto& q=requests[a];
        require(q.verifier && q.session && q.initial,"MTP cohort verification request is incomplete");
        auto& i=*q.verifier->impl_;
        for(std::size_t b=0;b<a;++b)require(requests[b].verifier!=q.verifier && requests[b].session!=q.session,
            "MTP cohort verification requires distinct sessions");
        const auto& session=*q.session;const auto& initial=*q.initial;
        require(i.prefix && !i.busy && !q.verifier->failed() && q.budget && initial.valid() &&
            initial.source_position()+1==session.next_position && session_equal(initial.committed_session(),session),
            "MTP cohort verifier requires exact idle committed prefix-owner frontiers");
        const auto vocabulary=i.target->model().vocab_size;
        for(auto token:q.candidates)require(token<vocabulary,"MTP target candidate outside vocabulary");
        for(auto token:q.eos)require(token<vocabulary,"MTP target EOS outside vocabulary");
        require(q.candidates.size()<i.window->logical_rows(),"MTP verifier admits only one closed window per cycle");
        const auto sampler=i.target->token_sampler();
        require(sampler?q.drafts.size()==q.candidates.size():q.drafts.empty(),"MTP sampled verification needs one draft distribution per candidate");
        for(std::size_t k=0;k<q.drafts.size();++k)
            require(q.drafts[k].token==q.candidates[k] && q.drafts[k].position==session.next_position+1U+k,
                "MTP sampled draft does not match its candidate");
        const auto pending=initial.predicted_token();
        require(std::find(q.eos.begin(),q.eos.end(),pending)==q.eos.end(),"MTP cohort session has an EOS pending token");
        auto* owner=i.owner_of(q.rows);
        // An empty joint draft uses R2 as one real target row plus padding.
        // Nothing after row zero is accepted; the normal prefix commit restores
        // the complete request state exactly as for any rejected draft.
        require(owner && (q.candidates.empty()?i.fits(session,*owner,1U,q.budget):
            i.fits(session,*owner,q.candidates.size(),q.budget)),"MTP cohort session owner does not fit");
        auto& p=prepared[a];p.impl=&i;p.owner=owner;
        for(const auto& pool:owner->captures)if(pool->ready_for_reuse()){p.capture=pool;break;}
        require(!!p.capture,"MTP window capture pools are retained by previous batch readers");
        i.target->validate_window_capture(*p.capture);
        p.inputs.reserve(q.rows);p.inputs.push_back(pending);
        for(std::uint32_t r=1;r<q.rows;++r) p.inputs.push_back(q.candidates.empty()?pending:
            q.candidates[std::min<std::size_t>(r-1U,q.candidates.size()-1U)]);
        members[a]={owner->window.get(),q.session,p.inputs,p.capture};
        if(q.transaction) {
            require(!sampler && q.verifier->policy_selection_admitted(),"MTP cohort policy route is not serial equivalent");
            q.transaction->reset();
            p.capture->set_transaction_selection(i.transaction_selection(*q.transaction,p.capture->source_stream(),q.candidates,q.rows,q.budget,q.eos));
        }
        if(sampler) p.capture->set_row_selection(i.sampled_selection(sampler,session.next_position+1U,p.capture->source_stream(),
            q.rows,std::min(q.rows,q.budget),q.drafts,q.eos));
    }
    for(auto& p:prepared) {
        // The previous owner's PLE commit runs on its own stream.
        if(p.impl->last && p.impl->last!=p.owner &&
           hipStreamSynchronize(static_cast<hipStream_t>(p.impl->last->captures.front()->source_stream()))!=hipSuccess)
            throw std::runtime_error("MTP window owner switch could not complete the previous commit");
    }
    for(std::size_t a=0;a<requests.size();++a){prepared[a].previous=*requests[a].session;prepared[a].impl->busy=true;}
    bool attempted=false;
    try {
        host_phase_mark("4 cohort verify prologue");
        auto natives=HipKnownPrefixWindow::execute_mtp_cohort_chunks(primaries,chunks,members);
        host_phase_mark("7 cohort window launch+wait");
        attempted=true;
        std::vector<HipMtpVerifiedTargetRows> out(requests.size());
        for(std::size_t a=0;a<requests.size();++a) {
            const auto& q=requests[a];auto& p=prepared[a];auto& i=*p.impl;auto& session=*q.session;
            const auto rows=q.rows;
            auto is_eos=[&](std::uint32_t token){return std::find(q.eos.begin(),q.eos.end(),token)!=q.eos.end();};
            auto& result=out[a];result.chosen_rows=rows;result.attempted_window=true;
            result.verified_candidates=static_cast<std::uint32_t>(std::min<std::size_t>(q.candidates.size(),rows-1U));
            result.r2_dispatches=natives[a].execution.r2_dispatches;
            result.projection_launches_saved=natives[a].execution.projection_launches_saved;
            auto provisional=std::make_shared<const HipMtpTargetWindowBatch>(std::move(natives[a].batch));
            std::vector<std::uint32_t> selected;selected.reserve(rows);
            for(std::uint32_t row=0;row<(provisional->selection_prefix()?provisional->selection_prefix()->decided_rows:rows);++row)selected.push_back(provisional->predicted_token(row));
            // Longest accepted prefix: row a+1's input equals row a's prediction.
            std::uint32_t k=0;
            while(!q.transaction && k+1U<rows && k<q.candidates.size() && k+1U<q.budget && p.inputs[k+1U]==selected[k] && !is_eos(selected[k])) ++k;
            const auto n_keep=q.transaction?static_cast<std::uint32_t>(provisional->selection_prefix()->decided_rows):k+1U;
            if(q.transaction) k=static_cast<std::uint32_t>(provisional->selection_prefix()->accepted_draft_inputs);
            // Prefix owners require one consumed input at row-zero failure.
            // Never promote a ticket or emit an ID for that undecided row.
            const auto committed=std::max(1U,n_keep);
            if(q.transaction)result.selection_failure=provisional->selection_prefix()->selection_failure;
            p.owner->window->commit_prefix(committed);i.last=p.owner;
            if(i.resources.qsa_committed_pool) i.resources.qsa_committed_pool->commit(false,p.previous.next_position+committed);
            if(committed<rows) {
                const auto dropped=rows-committed;
                session.next_position-=dropped;session.committed_tokens-=dropped;
                const auto& last=session.arithmetic_lineage.intervals.back();
                session.arithmetic_lineage=prepare_known_window_chunk_lineage(p.previous.arithmetic_lineage,last.begin,
                    last.begin+committed,last.numerical_plan_identity,last.reduction_plan_identity,last.state_update_identity);
                write_position_word(i.resources,session.next_position-1U);
            }
            if(n_keep)result.committed_rows=i.target->accept_verified_prefix(provisional,*q.initial,session,n_keep);
            result.emitted_tokens.assign(selected.begin(),selected.begin()+n_keep);
            result.accepted_candidates=k;result.accepted_window=n_keep==rows;
            if(q.transaction) {result.selection_receipt=q.transaction->seal();
                require(result.selection_receipt->size()==n_keep && result.committed_rows.size()==n_keep,"MTP cohort receipt lengths differ");}
            if(!q.transaction && q.probabilities.size()==q.candidates.size())i.minp->calibrate(q.probabilities,k,rows-1U);
            i.skip_sample();
            // A cohort session's cycle times include the other members: never persisted.
            if(i.costs) i.costs->exclude();
            provisional.reset();
            natives[a]={};
            p.capture->set_row_selection({});p.capture->set_transaction_selection({});
            p.capture.reset();
            i.busy=false;
        }
        host_phase_mark("7e cohort accept");
        return out;
    } catch(...) {
        for(std::size_t a=0;a<requests.size();++a) {
            auto& i=*prepared[a].impl;i.failed=true;i.busy=false;
        if(i.resources.qsa_committed_pool) i.resources.qsa_committed_pool->cancel();
            auto& session=*requests[a].session;
            if(attempted || session.health==SerialSessionHealth::poisoned){
                session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
                i.target->guard()->recover_or_fail_stop();
            }
        }
        throw;
    }
}
std::vector<HipMtpTargetHiddenTicket> HipMtpWindowVerifier::prompt_rows(SerialSessionContext& session,
    std::span<const std::uint32_t> tokens) {
    auto& i=*impl_;
    require(!i.busy && !failed(),"MTP prompt rows require an idle healthy window verifier");
    const auto rows=static_cast<std::uint32_t>(tokens.size());
    auto* owner=i.prefix && !i.target->token_sampler()?i.owner_of(rows):nullptr;
    if(!owner || !i.fits(session,*owner,rows-1U,rows)) return {};
    const auto vocabulary=i.target->model().vocab_size;
    for(auto token:tokens)require(token<vocabulary,"MTP prompt row token outside vocabulary");
    std::shared_ptr<HipMtpTargetWindowCapture> capture;
    for(const auto& pool:owner->captures)if(pool->ready_for_reuse()){capture=pool;break;}
    if(!capture) return {};
    i.target->validate_window_capture(*capture);
    capture->set_row_selection({});
    i.busy=true;bool attempted=false;
    try {
        // The previous owner's PLE commit runs on its own stream.
        if(i.last && i.last!=owner && hipStreamSynchronize(static_cast<hipStream_t>(i.last->captures.front()->source_stream()))!=hipSuccess)
            throw std::runtime_error("MTP window owner switch could not complete the previous commit");
        const std::vector<std::uint32_t> inputs(tokens.begin(),tokens.end());
        auto native=owner->window->execute_mtp_captured(session,inputs,capture,{},rows);
        attempted=true;
        auto provisional=std::make_shared<const HipMtpTargetWindowBatch>(std::move(native.batch));
        owner->window->commit_prefix(rows);i.last=owner;
        if(i.resources.qsa_committed_pool) i.resources.qsa_committed_pool->commit(false,session.next_position);
        auto tickets=i.target->accept_known_rows(provisional,session);
        // Prompt rows are no decode cycle: keep them out of the next cycle's measured time.
        i.skip_sample();
        provisional.reset();
        {auto done=std::move(native);}
        capture.reset();
        i.busy=false;return tickets;
    } catch(...) {
        i.failed=true;i.busy=false;
        if(i.resources.qsa_committed_pool) i.resources.qsa_committed_pool->cancel();
        if(attempted || session.health==SerialSessionHealth::poisoned){
            session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
            i.target->guard()->recover_or_fail_stop();
        }
        throw;
    }
}
std::vector<std::vector<HipMtpTargetHiddenTicket>> HipMtpWindowVerifier::prompt_rows_cohort(std::span<const PromptRowsRequest> requests) {
    std::vector<std::vector<HipMtpTargetHiddenTicket>> out(requests.size());
    struct Item {std::size_t request;Impl* impl;Impl::Owner* owner;std::uint32_t rows;std::shared_ptr<HipMtpTargetWindowCapture> capture;};
    std::vector<Item> items;
    for(std::size_t a=0;a<requests.size();++a) {
        const auto& q=requests[a];
        require(q.verifier && q.session,"MTP prompt rows cohort request is incomplete");
        auto& i=*q.verifier->impl_;
        for(std::size_t b=0;b<a;++b)require(requests[b].verifier!=q.verifier && requests[b].session!=q.session,
            "MTP prompt rows cohort requires distinct sessions");
        require(!i.busy && !q.verifier->failed(),"MTP prompt rows require an idle healthy window verifier");
        const auto rows=static_cast<std::uint32_t>(q.tokens.size());
        auto* owner=i.prefix && !i.target->token_sampler()?i.owner_of(rows):nullptr;
        if(!owner || !i.fits(*q.session,*owner,rows-1U,rows)) continue;
        const auto vocabulary=i.target->model().vocab_size;
        for(auto token:q.tokens)require(token<vocabulary,"MTP prompt row token outside vocabulary");
        std::shared_ptr<HipMtpTargetWindowCapture> capture;
        for(const auto& pool:owner->captures)if(pool->ready_for_reuse()){capture=pool;break;}
        if(!capture) continue;
        i.target->validate_window_capture(*capture);
        capture->set_row_selection({});
        items.push_back({a,&i,owner,rows,std::move(capture)});
    }
    if(items.empty()) return out;
    if(items.size()==1U) {
        const auto a=items.front().request;items.clear();
        out[a]=requests[a].verifier->prompt_rows(*requests[a].session,requests[a].tokens);
        return out;
    }
    // Chunks: first fit of the (descending) windows into bins of eight rows; a bin of two or more windows whose
    // rows are 4 or 8 runs on an idle owner of that many rows no window of this call uses, else each of its
    // windows runs alone in the stream of steps.
    std::stable_sort(items.begin(),items.end(),[](const Item& x,const Item& y){return x.rows>y.rows;});
    std::vector<std::vector<std::size_t>> bins;std::vector<std::uint32_t> sums;
    for(std::size_t k=0;k<items.size();++k) {
        bool placed=false;
        for(std::size_t b=0;b<bins.size() && !placed;++b) if(sums[b]+items[k].rows<=8U){bins[b].push_back(k);sums[b]+=items[k].rows;placed=true;}
        if(!placed){bins.push_back({k});sums.push_back(items[k].rows);}
    }
    std::vector<const HipKnownPrefixWindow*> used;
    for(const auto& x:items) used.push_back(x.owner->window.get());
    std::vector<HipKnownPrefixWindow*> primaries;std::vector<std::size_t> chunks;std::vector<std::size_t> order;
    std::vector<std::shared_ptr<HipKnownPrefixWindow>> keep;
    for(std::size_t b=0;b<bins.size();++b) {
        if(bins[b].size()>=2U && (sums[b]==4U || sums[b]==8U)) {
            std::shared_ptr<HipKnownPrefixWindow> primary;
            for(const auto& x:items) {
                auto* o=x.impl->owner_of(sums[b]);
                if(o && std::find(used.begin(),used.end(),o->window.get())==used.end()){primary=o->window;break;}
            }
            if(primary) {
                used.push_back(primary.get());primaries.push_back(primary.get());keep.push_back(std::move(primary));
                chunks.push_back(bins[b].size());order.insert(order.end(),bins[b].begin(),bins[b].end());
                continue;
            }
        }
        for(const auto k:bins[b]){primaries.push_back(nullptr);chunks.push_back(1U);order.push_back(k);}
    }
    std::vector<HipKnownPrefixCohortMember> members;std::vector<std::vector<std::uint32_t>> inputs(order.size());
    for(std::size_t m=0;m<order.size();++m) {
        auto& x=items[order[m]];const auto& q=requests[x.request];
        inputs[m].assign(q.tokens.begin(),q.tokens.end());
        members.push_back({x.owner->window.get(),q.session,inputs[m],x.capture});
    }
    for(const auto& x:items) {
        // The previous owner's PLE commit runs on its own stream.
        if(x.impl->last && x.impl->last!=x.owner &&
           hipStreamSynchronize(static_cast<hipStream_t>(x.impl->last->captures.front()->source_stream()))!=hipSuccess)
            throw std::runtime_error("MTP window owner switch could not complete the previous commit");
    }
    for(auto& x:items) x.impl->busy=true;
    bool attempted=false;
    try {
        auto natives=HipKnownPrefixWindow::execute_mtp_cohort_chunks(primaries,chunks,members);
        attempted=true;
        for(std::size_t m=0;m<order.size();++m) {
            auto& x=items[order[m]];auto& i=*x.impl;const auto& q=requests[x.request];
            auto provisional=std::make_shared<const HipMtpTargetWindowBatch>(std::move(natives[m].batch));
            x.owner->window->commit_prefix(x.rows);i.last=x.owner;
            if(i.resources.qsa_committed_pool) i.resources.qsa_committed_pool->commit(false,q.session->next_position);
            out[x.request]=i.target->accept_known_rows(provisional,*q.session);
            i.skip_sample();
            provisional.reset();natives[m]={};
            x.capture->set_row_selection({});x.capture.reset();
            i.busy=false;
        }
        return out;
    } catch(...) {
        for(auto& x:items) {
            auto& i=*x.impl;i.failed=true;i.busy=false;
        if(i.resources.qsa_committed_pool) i.resources.qsa_committed_pool->cancel();
            auto& session=*requests[x.request].session;
            if(attempted || session.health==SerialSessionHealth::poisoned) {
                session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
                i.target->guard()->recover_or_fail_stop();
            }
        }
        throw;
    }
}
void HipMtpWindowVerifier::begin_session() {
    auto& i=*impl_;
    require(!i.busy && !failed(),"MTP window verifier session restarts only while idle and healthy");
    for(auto& o:i.owners) o.window->discard_deferred();
    i.last=nullptr;i.last_end.reset();
    // The owners and their device pools are reusable; their workload estimates
    // are not. In particular, multi-stream cycle wall times include scheduler
    // work that must not price deeper windows out of a later single-stream run.
    i.flush_costs();
    if(mtp_request_reset()) {
        i.minp->begin_session();
        if(i.depth) {
            std::vector<std::uint32_t> rows;for(const auto& o:i.owners)rows.push_back(o.rows);
            i.depth=std::make_unique<MtpDepthWindowController>(std::move(rows),i.min_probability,mtp_depth_domain_prior());
        }
        for(auto& o:i.owners)o.warmed=false;
        for(auto& option:i.options)option=Impl::Option{option.owner,option.active};
        i.calibrator.bias=SurvivalCalibrator::prior[0];
        i.draft_ms_ema=i.serial_row_ms_ema=i.rho=i.chain_ms=0.0;
        i.draft_samples=i.serial_rows=i.cycle_index=0;
        i.chain_steps=0;i.external_first=SIZE_MAX;i.external.clear();
    }
}
bool HipMtpWindowVerifier::pending_window() const noexcept {
    const auto& i=*impl_;
    return i.prefix && i.last && i.last->window->deferred_pending();
}
void HipMtpWindowVerifier::flush_pending_window() {
    auto& i=*impl_;
    require(!i.busy && !failed(),"MTP window verifier flushes only while idle and healthy");
    if(i.last) {
        if(i.prefix) i.last->window->flush_deferred();
        if(hipStreamSynchronize(static_cast<hipStream_t>(i.last->captures.front()->source_stream()))!=hipSuccess)
            throw std::runtime_error("MTP window owner could not complete its pending commit");
    }
    // Nothing is pending now: this only invalidates the persistent QSA pools of every owner.
    for(auto& o:i.owners) o.window->discard_deferred();
}
void HipMtpWindowVerifier::skip_cycle_sample() {
    auto& i=*impl_;require(!i.busy,"MTP window verifier sample skip requires an idle verifier");
    i.last_end=std::chrono::steady_clock::now();
}
bool HipMtpWindowVerifier::failed() const noexcept {return impl_->failed || (impl_->checkpoint && impl_->checkpoint->failed());}
std::uint64_t HipMtpWindowVerifier::retained_payload_bytes() const noexcept {return impl_->retained;}
void HipMtpWindowVerifier::set_external_rows(std::size_t first,std::span<const double> acceptance) {
    auto& i=*impl_;
    require(!i.busy && !i.survival && i.minp,"MTP window verifier external rows are set between cycles of the min-p controller");
    i.minp->set_external_rows(first,acceptance);
    if(i.depth)i.depth->set_external_rows(first,acceptance);
}
void HipMtpWindowVerifier::clear_external_rows() noexcept {
    auto& i=*impl_;
    if(i.minp)i.minp->clear_external_rows();
    if(i.depth)i.depth->clear_external_rows();
}
bool HipMtpWindowVerifier::worth_drafting(std::span<const float> probabilities,std::span<const float> runner_ups) const {
    if(impl_->depth)return impl_->depth->worth_drafting(probabilities);
    return impl_->survival?impl_->survival_keep(probabilities,runner_ups):impl_->minp->worth_drafting(probabilities);
}
bool HipMtpWindowVerifier::survival_controller() const noexcept {return impl_->survival;}
bool HipMtpWindowVerifier::depth_controller() const noexcept {return static_cast<bool>(impl_->depth);}
std::vector<MtpDepthWindowController::CostStats> HipMtpWindowVerifier::depth_cost_stats() const {
    return impl_->depth?impl_->depth->cost_stats():std::vector<MtpDepthWindowController::CostStats>{};
}
std::vector<MtpDepthWindowController::CostStats> HipMtpWindowVerifier::draft_depth_stats() const {
    return impl_->depth?impl_->depth->draft_stats():std::vector<MtpDepthWindowController::CostStats>{};
}
void HipMtpWindowVerifier::prepare_cycle(const SerialSessionContext& session) {impl_->prepare_costs(session);}
void HipMtpWindowVerifier::note_draft(std::uint32_t steps,double ms) noexcept {
    impl_->chain_steps=steps;impl_->chain_ms=ms;
    if(impl_->depth) {
        if(steps<=7U && std::isfinite(ms) && ms>=0)impl_->depth->note_draft(steps,ms);
        else impl_->depth->note_draft(0,0);
    }
}
std::vector<HipMtpWindowOptionStats> HipMtpWindowVerifier::option_stats() const {
    std::vector<HipMtpWindowOptionStats> out;
    if(!impl_->survival) return out;
    out.push_back({1U,1U,impl_->serial_rows,impl_->serial_row_ms_ema});
    for(const auto& o:impl_->options) out.push_back({o.owner->rows,o.active,o.cycles,o.cost_ms_ema});
    out.push_back({0U,0U,impl_->draft_samples,impl_->draft_ms_ema});
    return out;
}
bool HipMtpWindowVerifier::policy_selection_admitted() const noexcept {
    return std::all_of(impl_->owners.begin(),impl_->owners.end(),[](const auto& o){return o.window->policy_selection_admitted();});
}
HipMtpVerifiedTargetRows HipMtpWindowVerifier::verify(SerialSessionContext& session,DirectLaunchDispatcher& dispatcher,
    const HipMtpTargetHiddenTicket& initial,std::span<const std::uint32_t> candidates,
    std::uint32_t budget,std::span<const std::uint32_t> eos,std::span<const float> probabilities,
    std::span<const TokenDraft> drafts,std::span<const float> runner_ups,MtpSelectionTransaction* transaction) {
    auto& i=*impl_;
    struct DiscardAttempt {MtpSelectionTransaction* t;~DiscardAttempt(){if(t)t->discard();}} discard_attempt{transaction};
    // Matched-position diagnostics: execute a physical owner, but commit only one
    // exact target row. Every arm therefore visits the same target frontiers.
    const bool diagnostic=hip_mtp_full_chain_diagnostic();
    const auto diagnostic_rows=diagnostic?static_cast<std::uint32_t>(env_number("STRIX_MTP_DIAG_ROWS",1.0)):0U;
    require(!diagnostic || diagnostic_rows==1U || diagnostic_rows==2U || diagnostic_rows==4U || diagnostic_rows==8U,
        "MTP diagnostic rows must be 1, 2, 4 or 8");
    if(diagnostic && diagnostic_rows==1U) budget=std::min(budget,1U);
    struct ClearExternal{Impl& impl;~ClearExternal(){
        if(impl.minp)impl.minp->clear_external_rows();
        if(impl.depth)impl.depth->clear_external_rows();
    }} clear_external{i};
    const auto sampler=i.target->token_sampler();
    require(sampler?drafts.size()==candidates.size():drafts.empty(),"MTP sampled verification needs one draft distribution per candidate");
    for(std::size_t k=0;k<drafts.size();++k)
        require(drafts[k].token==candidates[k] && drafts[k].position==session.next_position+1U+k,"MTP sampled draft does not match its candidate");
    require(!i.busy && !failed() && budget && initial.valid() &&
        initial.source_position()+1==session.next_position && session_equal(initial.committed_session(),session),
        "MTP window verifier requires exact idle committed target frontier");
    const auto vocabulary=i.target->model().vocab_size;
    for(auto token:candidates)require(token<vocabulary,"MTP target candidate outside vocabulary");
    for(auto token:eos)require(token<vocabulary,"MTP target EOS outside vocabulary");
    auto is_eos=[&](std::uint32_t token){return std::find(eos.begin(),eos.end(),token)!=eos.end();};
    require(candidates.size()<i.window->logical_rows(),"MTP verifier admits only one closed window per cycle");
    auto pending=initial.predicted_token();
    HipMtpVerifiedTargetRows out;out.committed_rows.reserve(candidates.size()+1);out.emitted_tokens.reserve(candidates.size()+1);
    if(is_eos(pending))return out;
    const auto capacity=std::min(i.resources.session_state_layout->context_tokens,i.target->model().effective_max_positions());
    require(session.next_position<capacity,"MTP verifier target context is full");
    // Capture::begin needs room for one actual known-window chunk. Capacity
    // exhaustion is an ordinary serial fallback, before pool/checkpoint work.
    // Adaptive prefix windows may pad unconfident rows: they are never accepted.
    const bool pad=i.prefix && (i.min_probability>0.0F || i.survival);
    const auto room=capacity-session.next_position;
    const bool history=session.arithmetic_lineage.intervals.size()<i.maximum_history_intervals;
    const bool survival=i.survival && probabilities.size()==candidates.size();
    Impl::Option* option=nullptr;Impl::Owner* owner=nullptr;
    if(survival) {
        option=i.survival_choose(probabilities,runner_ups,candidates.size(),[&](const Impl::Option& o){
            return budget>=o.active && room>=o.owner->rows && history && !candidates.empty();});
        owner=option?option->owner:nullptr;
    } else {
        i.prepare_costs(session);
        owner=i.choose(probabilities,candidates.size(),[&](const Impl::Owner& o){
            return budget>=o.rows && room>=o.rows && history && (pad?!candidates.empty():candidates.size()+1>=o.rows);});
    }
    if(diagnostic) {
        owner=diagnostic_rows>1U?i.owner_of(diagnostic_rows):nullptr;
        require(diagnostic_rows==1U || owner,"MTP diagnostic owner is not admitted");
        if(owner && (budget<owner->rows || room<owner->rows || !history || candidates.empty())) owner=nullptr;
        option=nullptr;
    }
    const bool use_window=owner!=nullptr;
    const auto rows=use_window?owner->rows:1U;
    // Rows whose drafts can be accepted; the owner's later rows are padding (outside the expert union
    // when fewer than its physical rows).
    const auto active=option?option->active:rows;
    out.chosen_rows=active;
    const auto calibrate=[&](std::size_t accepted,std::size_t verified) {
        if(transaction || probabilities.size()!=candidates.size()) return;
        if(survival) i.survival_calibrate(probabilities,runner_ups,accepted,verified);
        else {
            i.minp->calibrate(probabilities,accepted,verified);
            if(i.depth)i.depth->calibrate(probabilities,accepted,verified);
        }
    };
    const auto observe=[&](bool window,std::size_t emitted) {
        if(survival) i.survival_observe(window?option:nullptr,emitted);
        else i.observe(window?owner:nullptr,emitted);
    };
    std::shared_ptr<HipMtpTargetWindowCapture> capture;
    struct ClearSelection {std::shared_ptr<HipMtpTargetWindowCapture>& c;~ClearSelection(){if(c){c->set_row_selection({});c->set_transaction_selection({});}}} clear_selection{capture};
    if(transaction) {require(!sampler && policy_selection_admitted(),"MTP policy route is not serial equivalent");transaction->reset();}
    if(use_window) {
        for(const auto& pool:owner->captures)if(pool->ready_for_reuse()){capture=pool;break;}
        require(!!capture,"MTP window capture pools are retained by previous batch readers");
        i.target->validate_window_capture(*capture);
        // Sampled rows: decided in order from row 0 (the token at next_position + 1) until the first
        // row without a draft, a rejection, EOS or the budget; later rows are never accepted.
        if(sampler) capture->set_row_selection(i.sampled_selection(sampler,session.next_position+1U,capture->source_stream(),
            active,std::min(active,budget),drafts,eos)); // padding rows are never decided
        else capture->set_row_selection({});
        if(transaction) capture->set_transaction_selection(i.transaction_selection(*transaction,capture->source_stream(),
            candidates,diagnostic?1U:active,budget,eos));
    }
    // The selection borrows this call's locals: never leave it attached past the call.

    const auto seal=[&] {if(transaction) {out.accepted_candidates=static_cast<std::uint32_t>(out.emitted_tokens.empty()?0:out.emitted_tokens.size()-1U);
        out.selection_receipt=transaction->seal();
        require(out.selection_receipt->size()==out.emitted_tokens.size() && out.committed_rows.size()==out.emitted_tokens.size(),
            "MTP transaction retained rows, IDs and records differ");}};
    i.busy=true;bool attempted_target=false;
    try {
        if(use_window && i.prefix) {
            // The previous owner's PLE commit runs on its own stream.
            if(i.last && i.last!=owner && hipStreamSynchronize(static_cast<hipStream_t>(i.last->captures.front()->source_stream()))!=hipSuccess)
                throw std::runtime_error("MTP window owner switch could not complete the previous commit");
            std::vector<std::uint32_t> inputs;inputs.reserve(rows);inputs.push_back(pending);
            for(std::uint32_t r=1;r<rows;++r) inputs.push_back(candidates[std::min<std::size_t>(r-1U,candidates.size()-1U)]);
            const auto previous=session;
            host_phase_mark("4 verify prologue");
            auto native=owner->window->execute_mtp_captured(session,inputs,capture,{},active);
            host_phase_mark("7 window launch+wait");
            attempted_target=true;out.attempted_window=true;
            out.verified_candidates=static_cast<std::uint32_t>(std::min<std::size_t>(candidates.size(),active-1U));
            out.r2_dispatches=native.execution.r2_dispatches;
            out.projection_launches_saved=native.execution.projection_launches_saved;
            auto provisional=std::make_shared<const HipMtpTargetWindowBatch>(std::move(native.batch));
            std::vector<std::uint32_t> selected;selected.reserve(rows);
            for(std::uint32_t row=0;row<(provisional->selection_prefix()?provisional->selection_prefix()->decided_rows:rows);++row)selected.push_back(provisional->predicted_token(row));
            // Longest accepted prefix: row a+1's input equals row a's prediction.
            std::uint32_t a=0;
            while(!transaction && !diagnostic && a+1U<active && a<candidates.size() && a+1U<budget && inputs[a+1U]==selected[a] && !is_eos(selected[a])) ++a;
            const auto n_keep=transaction?static_cast<std::uint32_t>(provisional->selection_prefix()->decided_rows):a+1U;
            if(transaction) a=static_cast<std::uint32_t>(provisional->selection_prefix()->accepted_draft_inputs);
            host_phase_mark("7a select/accept");
            // Prefix owners require one consumed input at row-zero failure.
            // Never promote a ticket or emit an ID for that undecided row.
            const auto committed=std::max(1U,n_keep);
            if(transaction)out.selection_failure=provisional->selection_prefix()->selection_failure;
            owner->window->commit_prefix(committed);i.last=owner;
            if(i.resources.qsa_committed_pool) i.resources.qsa_committed_pool->commit(false,previous.next_position+committed);
            host_phase_mark("7b commit prefix");
            if(committed<rows) {
                const auto dropped=rows-committed;
                session.next_position-=dropped;session.committed_tokens-=dropped;
                const auto& last=session.arithmetic_lineage.intervals.back();
                session.arithmetic_lineage=prepare_known_window_chunk_lineage(previous.arithmetic_lineage,last.begin,
                    last.begin+committed,last.numerical_plan_identity,last.reduction_plan_identity,last.state_update_identity);
                write_position_word(i.resources,session.next_position-1U);
            }
            host_phase_mark("7c lineage");
            if(n_keep)out.committed_rows=i.target->accept_verified_prefix(provisional,initial,session,n_keep);
            host_phase_mark("7d accept prefix");
            out.emitted_tokens.assign(selected.begin(),selected.begin()+n_keep);
            out.accepted_candidates=a;out.accepted_window=n_keep==active;
            calibrate(a,active-1U);
            observe(true,n_keep);
            host_phase_mark("7e calibrate/observe");
            provisional.reset();
            host_phase_mark("7f release provisional");
            {auto done=std::move(native);}
            host_phase_mark("7g release native");
            capture->set_transaction_selection({});capture->set_row_selection({});capture.reset();
            host_phase_mark("7h release capture");
            seal();i.busy=false;return out;
        }
        if(i.prefix && i.last) i.last->window->flush_deferred();
        if(use_window) {
            std::vector<std::uint32_t> inputs;inputs.reserve(rows);inputs.push_back(pending);
            inputs.insert(inputs.end(),candidates.begin(),candidates.begin()+static_cast<std::ptrdiff_t>(rows-1));
            if(i.bounded)i.checkpoint->capture_window(session);else i.checkpoint->capture(session);
            auto native=owner->window->execute_mtp_captured(session,inputs,capture);
            // Successful return proves actual target consumption. Rejected
            // admission leaves the original ready session untouched; dispatch
            // failures are already poisoned/drained by the native producer.
            attempted_target=true;out.attempted_window=true;
            out.verified_candidates=rows-1U;
            out.r2_dispatches=native.execution.r2_dispatches;
            out.projection_launches_saved=native.execution.projection_launches_saved;
            auto provisional=std::make_shared<const HipMtpTargetWindowBatch>(std::move(native.batch));
            std::vector<std::uint32_t> selected;selected.reserve(rows);
            for(std::uint32_t row=0;row<(provisional->selection_prefix()?provisional->selection_prefix()->decided_rows:rows);++row)selected.push_back(provisional->predicted_token(row));
            const bool complete=transaction?(selected.size()==rows && provisional->selection_prefix()->accepted_draft_inputs==rows-1U &&
                std::none_of(inputs.begin(),inputs.end(),is_eos)):
                assess_mtp_target_window(inputs,selected,pending,budget,eos,vocabulary).accepts_complete_chunk();
            if(complete) {
                if(i.resources.qsa_committed_pool) i.resources.qsa_committed_pool->commit(false,session.next_position);
                out.committed_rows=i.target->accept_verified_window(provisional,initial,session,eos,budget);
                out.emitted_tokens=std::move(selected);out.accepted_candidates=rows-1;out.accepted_window=true;
                i.checkpoint->discard();i.observe(owner,rows);seal();i.busy=false;return out;
            }
            // No provisional row has been promoted. Restore the exact actual
            // checkpoint, then select/reverify the serial greedy continuation.
            if(i.bounded)i.checkpoint->restore_window(session,*provisional);else i.checkpoint->restore(session);
            if(transaction)out.selection_failure=provisional->selection_prefix()->selection_failure;
            if(out.selection_failure.has_value())out.emitted_tokens=std::move(selected);
            provisional.reset();out.restored_checkpoint=true;
            require(session_equal(initial.committed_session(),session),"MTP rejected window did not restore exact target frontier");
            if(out.selection_failure.has_value()) {
                // The request ends: retain its decided selection records while
                // leaving this checkpoint owner at the restored model frontier.
                out.selection_receipt=transaction->seal();i.busy=false;return out;
            }
            if(transaction){transaction->discard();transaction->reset();}
        }
        for(std::size_t index=0;out.emitted_tokens.size()<budget && session.next_position<capacity;++index) {
            auto ticket=i.target->execute_serial(session,i.resources,dispatcher,pending,
                index<drafts.size()?&drafts[index]:nullptr,
                transaction?HipMtpHostRowSelector{[&](auto row){
                    try{return transaction->select(row);}catch(const MtpSelectionError& e){out.selection_failure=e.what();return 0U;}
                }}:HipMtpHostRowSelector{});
            attempted_target=true;
            if(out.selection_failure.has_value())break;
            pending=ticket.predicted_token();out.committed_rows.push_back(std::move(ticket));out.emitted_tokens.push_back(pending);
            if(index<candidates.size()) out.verified_candidates=std::max(out.verified_candidates,static_cast<std::uint32_t>(index+1U));
            const bool accepted=index<candidates.size() && candidates[index]==pending;
            if(accepted)++out.accepted_candidates;
            if(diagnostic || is_eos(pending) || !accepted)break;
        }
        calibrate(out.accepted_candidates,out.emitted_tokens.size());
        observe(use_window,out.emitted_tokens.size());
        seal();i.busy=false;return out;
    } catch(...) {
        if(transaction)transaction->discard();
        i.failed=true;i.busy=false;
        if(i.resources.qsa_committed_pool) i.resources.qsa_committed_pool->cancel();
        // Producers/checkpoint own cleanup for their submitted operations.
        // Do not synchronize or poison a purely rejected predispatch scope.
        // A failure after any successful target consumption still cannot
        // publish a reusable partially completed verification transaction.
        if(attempted_target || session.health==SerialSessionHealth::poisoned){
            session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
            i.target->guard()->recover_or_fail_stop();
        }
        throw;
    }
}
} // namespace strix
