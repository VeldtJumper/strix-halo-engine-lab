#include "../apps/qwen_prompt_organization.hpp"
#include "../apps/qwen_http_memory.hpp"
#include "../apps/mtp_asset_verification.hpp"
#include "../apps/qwen_cache_family_contract.hpp"
#include "strix/mtp_selection_transaction.hpp"
#include "../apps/qwen_prefill_plan.hpp"
#include "../apps/prefix_state_cache_identity.hpp"
#include "qwen_prefill_contract.hpp"
#include "rotary_mock.hpp"
#include "strix/memory_plan.hpp"
#include "strix/session_state_cache.hpp"
#include "../apps/qwen_yarn_owner_inventory.hpp"
#include "../apps/qwen_yarn_registered_admission.hpp"
#include "../apps/qwen_mtp_owner_plan.hpp"
#include "../apps/known_prefill_reservation.hpp"
#include "../apps/ple_resident_fallback.hpp"
#include "qwen_serial_contract.hpp"
#include "qwen_historical_resolver.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include "strix/prompt_terminal_diagnostics.hpp"
#include "qwen_cache_compatibility_cases.hpp"
#include "../apps/http_slot_capacity.hpp"
#include "../apps/qwen_tiled_prompt_plan.hpp"
#include "strix/moe_route_bf16.hpp"
#include <limits>
#include <set>
#include "strix/q8_projection_organization.hpp"
#include <filesystem>
#include <chrono>

using namespace strix;
using namespace strix::decode_cli;
namespace {
// Synthetic CPU kernel fixture: compares compiler identities without HIP launchers.
strix::kernels::KernelVariant config_test_variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    const strix::kernels::ShapeClass& shape,
    double measured_seconds) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        strix::kernels::ExecutionPhase::decode;
    variant.shape = shape;
    variant.wave =
        strix::kernels::WaveSize::wave32;
    variant.resources.workgroup_threads = 256;
    variant.numerics.input_precision = "bf16";
    variant.numerics.accumulator_precision = "fp32";
    variant.numerics.output_precision = "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        strix::kernels::QualificationStatus::oracle_matched;
    variant.graph_safe = true;
    variant.input_layout = "test";
    variant.output_layout = "test";
    variant.compiler_identity = "test-compiler";
    variant.measured_seconds = measured_seconds;
    return variant;
}

strix::kernels::KernelRegistry config_test_registry(
    const strix::ModelContract& model,
    const strix::ExecutionBindingRequest& request,
    double measured_seconds = 0.0,
    std::string suffix = "") {
    strix::kernels::KernelRegistry registry;
    const auto execution =
        strix::build_serial_decode_execution_plan(model);

    std::set<std::string> added;
    for (const auto& step : execution.steps) {
        const auto family =
            strix::operator_family_for(step.op);
        if (!family.has_value()) {
            continue;
        }

        const auto shape =
            strix::execution_shape_for(
                model, step.op, request);

        const auto key =
            std::to_string(
                static_cast<unsigned>(*family)) +
            "." +
            std::to_string(shape.logical_rows) +
            "." +
            std::to_string(shape.request_concurrency) +
            "." +
            std::to_string(shape.hidden_size) +
            "." +
            std::to_string(shape.auxiliary_size);

        if (!added.insert(key).second) {
            continue;
        }

        auto variant = config_test_variant(
                "test." + key + suffix +
                    (step.op == strix::SerialOp::moe_route && request.staged_moe_router_logits ? ".staged_logits" : "") +
                    (step.op == strix::SerialOp::moe_route && request.wave_moe_router_topk ? ".wave.parallel" :
                     step.op == strix::SerialOp::moe_route && request.ranked_moe_router_topk ? ".ranked.parallel" :
                     step.op == strix::SerialOp::moe_route && request.parallel_moe_router ? ".parallel" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.batched_moe_activation ? ".batched-slot-activation-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.deferred_moe_status ? ".deferred-status-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.q5k_a8g32_gate_up ? ".q5k-a8g32-gateup-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.a8_down_q51 ? ".q51-a8g32-down-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.a8_down_q80 ? ".q80-a8g32-down-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.a8_down_q51 ? ".mixed-a8-down-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.grouped_exact_q51_down && !request.a8_down_q51 ? ".grouped-exact-q51-down-v1" : "") +
                    (step.op == strix::SerialOp::moe_experts && request.q4k_a8g32_gate_up ? ".q4k-a8g32-gateup-v2" : ""),
                *family,
                shape,
                measured_seconds);
        using namespace strix;
        using namespace q8_projection_organization;
        if (step.op==SerialOp::gated_delta_net) {
            if(request.wave8_q8_gdn_input) variant.id+=gdn_wave8_marker;
            if(request.wave8_q8_gdn_output) variant.id+=gdn_output_wave8_marker;
        }
        if(step.op==SerialOp::indexed_attention && request.wave8_q8_qsa_output) variant.id+=qsa_output_wave8_marker;
        if(request.dense_a8 && (step.op==SerialOp::gated_delta_net || step.op==SerialOp::indexed_attention || step.op==SerialOp::shared_expert))
            variant.id+=dense_a8_marker;
        if(step.op==SerialOp::lm_head && request.prefetch_q8_lm_head)
            variant.id+=request.wave8_q8_lm_head ? lm_wave8_marker : lm_prefetch_marker;
        if(*family==kernels::OperatorFamily::hyper_mix && request.prefetch_hc_projection_next_tile) variant.id+=".prefetch-next-tile-v1";
        registry.add(variant);
        // Layer-scoped selection can require either terminal suffix for one family.
        if(step.op==SerialOp::indexed_attention) {variant.id+=qsa_input_prefetch::kernel_suffix; registry.add(variant);}
        if(step.op==SerialOp::shared_expert) {variant.id+=shared_expert_prefetch::kernel_suffix; registry.add(variant);}
        if(*family==kernels::OperatorFamily::hyper_mix) {variant.id+=lane_chain_hyper_output_suffix; registry.add(variant);}
    }

    return registry;
}

int failures = 0;
void expect(bool ok, const char* message) {
    if (!ok) { ++failures; std::cerr << message << '\n'; }
}
Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return parse_options(static_cast<int>(argv.size()), argv.data());
}
std::vector<std::string> mtp_fixture(){
    const std::string sha(64,'1');
    return {"decoder","target.gguf","1","2","3","--generate","512","--mtp",
        "--mtp-sidecar","sidecar.gguf","--mtp-sidecar-logical-name","mtp-sidecar",
        "--mtp-sidecar-sha256",sha,"--mtp-target-sha256",sha,"--mtp-target-content",sha,
        "--mtp-golden","independent.bin","--mtp-golden-sha256",sha,
        "--mtp-source-repository","publisher/model","--mtp-source-revision","pinned-revision",
        "--mtp-source-path","sidecar.gguf","--mtp-converter-revision","converter-revision",
        "--mtp-converter-sha256",sha,"--mtp-norm-converter-sha256",sha,
        "--mtp-conversion","effective-norm-embedding-first-v1","--mtp-normalized-rounding","fp32",
        "--mtp-projection-rounding","bf16-rne","--mtp-sidecar-residency","register",
        "--mtp-sidecar-bytes","1024","--mtp-header-bytes","512",
        "--mtp-sidecar-retained-bytes","2048","--mtp-sidecar-setup-bytes","2048",
        "--mtp-golden-max-bytes","4096","--mtp-replay-host-bytes","8192",
        "--mtp-runtime-bytes","16384","--mtp-target-capture-bytes","8192",
        "--mtp-policy-id","7","--mtp-candidate-depth","2"};
}
// The explicit MTP command of docs/getting-started/engine-reference.md ("Command line", groups 1-4; placeholders
// filled as in mtp_fixture) with the known-prefix window options of the measured MTP campaign argv.
std::vector<std::string> documented_mtp_argv(){
    const std::string sha(64,'1');
    return {"decoder","target.gguf","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","--generate","256",
        // 1. Target model and generation.
        "--context","8192","--execution-mode","eager","--residency","register","--optimization-profile","auto",
        "--numerical-profile","hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1",
        // 2. Kernel organization required by the windows.
        "--q51-slot-parallel-down","--batched-moe-activation","--q4k-a8g32-gate-up","--grouped-exact-q51-down",
        "--deferred-moe-status","--parallel-moe-router","--wave-moe-router-topk",
        "--virtual-q8-gdn-input","--prefetch-q8-gdn-input","--virtual-q8-gdn-output",
        "--wave8-q8-gdn-input","--wave8-q8-gdn-output","--wave8-q8-qsa-output",
        "--prefetch-q8-lm-head","--wave8-q8-lm-head","--virtual-q8-lm-head",
        "--prefetch-q8-qsa-input","--prefetch-q8-shared-expert",
        "--parallel-hyper-mix","--staged-hyper-projections","--fused-staged-hyper-mix",
        "--typed-fused-hyper-projections","--prefetch-typed-hyper-rows",
        "--prefetch-q8-hyper-output","--lane-chain-hyper-output",
        // 3. MTP sidecar, provenance and budgets.
        "--mtp","--mtp-sidecar","sidecar.gguf","--mtp-sidecar-logical-name","mtp-sidecar","--mtp-sidecar-sha256",sha,
        "--mtp-sidecar-bytes","1024","--mtp-target-sha256",sha,"--mtp-target-content",sha,
        "--mtp-source-repository","publisher/model","--mtp-source-revision","pinned-revision","--mtp-source-path","sidecar.gguf",
        "--mtp-converter-revision","converter-revision","--mtp-converter-sha256",sha,"--mtp-norm-converter-sha256",sha,
        "--mtp-conversion","effective-norm-embedding-first-v1",
        "--mtp-normalized-rounding","bf16-rne","--mtp-projection-rounding","bf16-rne","--mtp-sidecar-residency","register",
        "--mtp-header-bytes","33554432","--mtp-sidecar-retained-bytes","3221225472","--mtp-sidecar-setup-bytes","4294967296",
        "--mtp-golden","independent.bin","--mtp-golden-sha256",sha,"--mtp-golden-max-bytes","67108864",
        "--mtp-replay-host-bytes","67108864","--mtp-runtime-bytes","134217728","--mtp-target-capture-bytes","16777216",
        "--mtp-policy-id","1","--mtp-candidate-depth","7","--mtp-max-hidden-error","0","--mtp-max-logit-error","0",
        // 4. Speculative decode.
        "--mtp-target-window-rows","8","--mtp-target-window-rows-set","2,4,8",
        "--mtp-window-arena-bytes","2147483648","--mtp-window-capture-bytes","67108864",
        "--mtp-window-checkpoint-bytes","8589934592","--mtp-window-total-bytes","17179869184",
        "--mtp-window-history-intervals","8192",
        "--mtp-window-expert-union","--mtp-window-expert-union-bytes","65536",
        "--mtp-window-prefix","--mtp-window-hyper-mix","--mtp-window-qsa-rows",
        "--mtp-draft-chain","--mtp-spec-min-p","0.85","--mtp-draft-q4",
        "--mtp-draft-vocab","auto:data/draft-vocab","--mtp-draft-context-vocab","4096",
        // Known-prefix window options of the campaign argv.
        "--known-prefix-window","--known-prefix-full-rows","--known-prefix-workspace-budget-bytes","8589934592",
        "--known-prefix-host-metadata-budget-bytes","1073741824","--known-prefix-uma-budget-bytes","128849018880",
        "--known-prefix-host-headroom-bytes","4294967296","--known-prefix-output-policy","terminal-row",
        "--known-prefix-expert-union","--known-window-router-rows","--known-prefix-prefill-rows","8192"};
}
constexpr std::uint64_t gib_bytes = UINT64_C(1024) * 1024U * 1024U;
void resident_fallback_tests() {
    Options options; options.mtp.enabled=true;
    options.known_prefill_uma_budget=120U*gib_bytes;
    options.known_prefill_workspace_budget=8U*gib_bytes;
    options.known_prefill_metadata_budget=gib_bytes;
    PrefillRequirements required;
    required.model_backing=90U*gib_bytes; required.workspace=2U*gib_bytes; required.metadata=gib_bytes/4U;
    required.device_base=4U*gib_bytes;
    required.owners={{"model_backing",required.model_backing},{"state",4U*gib_bytes},
        {"workspace",required.workspace},{"metadata",required.metadata}};
    PrefillMemory memory; memory.total=128U*gib_bytes; memory.available=20U*gib_bytes;
    memory.gtt_limit=120U*gib_bytes; memory.gtt_used=90U*gib_bytes;
    const auto admitted=resident_prefill_admission(options,required,memory);
    expect(admitted.at("verified_ple_credit_bytes")==0 && admitted.at("required_bytes")==admitted.at("gross_required_bytes"),
        "fresh resident admission never retains rolling credit");
    for(const bool yarn:{false,true}) for(const bool reserve:{false,true}) {
        auto route=options;if(yarn) route.rotary=RotaryDescriptor::yarn(2);
        route.known_prefill_host_headroom=reserve?yarn_legacy_host_reserve:0;
        auto edge=memory;
        const auto baseline=resident_prefill_admission(route,required,edge);
        edge.available=baseline.at("incremental_bytes")+prefill_physical_margin;
        expect(resident_prefill_admission(route,required,edge).at("resident_fallback")==1,"resident exact physical margin admits");
        --*edge.available;bool refused=false;
        try {(void)resident_prefill_admission(route,required,edge);}catch(const std::runtime_error&){refused=true;}
        expect(refused,"native/YaRN resident MemAvailable minus 4 GiB short by one byte refuses in both reserve modes");
    }
    {
        // A refused physical check reports its inputs and the shortfall on each side.
        const auto m=prefill_physical_margin;
        const auto detail=prefill_physical_margin_detail(100U*gib_bytes,m+5U,10U*gib_bytes,8U);
        expect(detail.find(" incremental_shortfall=3")!=std::string::npos && detail.find(" retained_shortfall=0 ")!=std::string::npos &&
            detail.find("mem_available="+std::to_string(m+5U)+" ")!=std::string::npos,"physical margin detail names the short side");
        expect(prefill_physical_margin_detail(m-1U,0U,1U,1U).find(" retained_shortfall=2 incremental_shortfall="+std::to_string(m+1U))!=std::string::npos,
            "below-margin observations include the missing margin in the shortfall");
    }
    for(unsigned fault=0;fault<4;++fault) {
        auto observed=memory; auto owners=required;
        if(fault==0) owners.owners["model_backing"]=121U*gib_bytes;
        if(fault==1) observed.available=gib_bytes;
        if(fault==2) observed.gtt_used=119U*gib_bytes;
        if(fault==3) observed.total.reset();
        bool refused=false;
        try { (void)resident_prefill_admission(options,owners,observed); }
        catch(const std::runtime_error& error) { refused=std::string(error.what()).find("before prompt mutation")!=std::string::npos; }
        expect(refused,"resident fallback refuses unavailable/insufficient capacity before warming");
    }
}
// Retain the pre-existing T8192 ladder cases independently of automatic T16384.
EffectiveConfig resolve_w7_case(const RequestedOptions& request, const ModelContract& model,
    const gguf::FileInfo& info, const ModelManifest& manifest, std::string_view content,
    const PrefillCapabilities& capabilities, const PrefillMemory& memory) {
    auto historical = request;
    if (!historical.environment.contains("STRIX_QPF2K_T16K"))
        historical.environment["STRIX_QPF2K_T16K"] = "0";
    return resolve_effective_config(historical,model,info,manifest,content,capabilities,memory);
}
// The source command did not enable MTP or cohorts. Their inactive controls
// and disabled experiments are excluded. The logits digest is diagnostic output,
// not part of the prefill arithmetic/organization contract.
constexpr std::array excluded_inactive_env{
    "STRIX_QKERN_QSA_PERSISTENT", "STRIX_QKERN_GDN_SCATTER", "STRIX_QKERN_ROUTER8", "STRIX_QKERN_HEAD_DIRECT",
    "STRIX_QSA_WINDOW_POOLED", "STRIX_QSA_WINDOW_GROUPED", "STRIX_MTP_CONTROLLER", "STRIX_MTP_DRAFT_CACHE",
    "STRIX_MTP_CACHE_AUDIT", "STRIX_QWEN_SHARED_DOWN_NARROW", "STRIX_QPERSIST_DRAFT_SCORE",
    "STRIX_QPERSIST_QSA_TAIL", "STRIX_QPERSIST_MTP_CACHE", "STRIX_QPERSIST_GDN_TAIL", "STRIX_MTP_HC_PERSISTENT",
    "STRIX_MTP_BATCH_JOINT", "STRIX_COHORT_ROWS", "STRIX_COHORT_WIDE",
    "STRIX_PREFILL_LOGITS_DIGEST"};
PrefillMemory ample_memory() {
    return {128U*gib_bytes,128U*gib_bytes,0U,[](const Options&) {
        return PrefillRequirements{70U*gib_bytes,8U*gib_bytes,gib_bytes,2U*gib_bytes,512U*1024U*1024U,128U*1024U*1024U,0U,9U*gib_bytes};
    },[](const Options&) { return HttpServingMemory{}; }};
}
// CPU regression: production organizations/window ceilings with synthetic tokens,
// asset pins and backing requirements. No model, golden, HIP or external files.
void serve_prefill_admission_tests(const ModelContract& model, const ModelManifest& manifest) {
    auto args=mtp_fixture();
    args.insert(args.begin()+5,13,"1");
    const auto set = [&](const char* flag, const char* value) {
        const auto it=std::find(args.begin(),args.end(),flag);
        if(it==args.end()) args.insert(args.end(),{flag,value}); else *(it+1)=value;
    };
    for(std::size_t i=0;i<qwen_contract::argv.size();++i) {
        const auto flag=qwen_contract::argv[i];
        // Reuse the production serial organizations, without tiled opt-ins or
        // split prefill arithmetic. The serving resolver preserves dense A8.
        if(flag.starts_with("--known-") || flag=="--prefill-numerical-profile") {
            if(i+1<qwen_contract::argv.size() && !qwen_contract::argv[i+1].starts_with("--")) ++i;
            continue;
        }
        if(flag=="--context" || flag=="--generate") { ++i; continue; }
        args.emplace_back(flag);
    }
    args.insert(args.end(),{"--context","24576","--http-serve","--http-batch","2",
        "--prefix-cache-block-rows","0","--mtp-window-expert-union","--mtp-window-prefix",
        "--mtp-window-hyper-mix","--mtp-window-qsa-rows","--mtp-draft-chain","--mtp-draft-q4"});
    for(const auto& [flag,value] : std::vector<std::pair<const char*,const char*>>{
        {"--mtp-normalized-rounding","bf16-rne"},{"--mtp-sidecar-retained-bytes","3221225472"},
        {"--mtp-sidecar-setup-bytes","4294967296"},{"--mtp-golden-max-bytes","6442450944"},
        {"--mtp-replay-host-bytes","6442450944"},{"--mtp-runtime-bytes","134217728"},
        {"--mtp-target-capture-bytes","268435456"},{"--mtp-candidate-depth","7"},
        {"--mtp-target-window-rows","8"},{"--mtp-window-arena-bytes","2147483648"},
        {"--mtp-window-capture-bytes","67108864"},{"--mtp-window-checkpoint-bytes","8589934592"},
        {"--mtp-window-total-bytes","17179869184"},{"--mtp-window-history-intervals","8192"},
        {"--mtp-window-expert-union-bytes","65536"},{"--mtp-target-window-rows-set","2,4,8"},
        {"--mtp-draft-vocab","auto:assets/draft-vocab"},{"--mtp-draft-context-vocab","4096"},
        {"--mtp-spec-min-p","0.7"}}) set(flag,value);
    RequestedOptions request{parse(args),{{"STRIX_BATCH_PROMPT_SMALL","0"}}};
    PrefillMemory memory;
    memory.available=131766833152ULL; memory.gtt_limit=125829120000ULL; memory.gtt_used=23539712U;
    memory.estimate=[](const Options& o) {
        return PrefillRequirements{100*gib_bytes,gib_bytes,gib_bytes,
            o.known_prefix_prefill_rows*(gib_bytes/1024),gib_bytes/2,gib_bytes/8,gib_bytes,2*gib_bytes};
    };
    const auto resolve=[&](const RequestedOptions& r,const PrefillMemory& m) {
        return resolve_effective_config(r,model,{},manifest,validated_model_content_identity,{true,true,true},m);
    };
    {
        // Native joining gate shape: pinned MTP server args (C24576, T8192 serving prefill),
        // explicit 4 GiB host budget, --http-batch 3 --prefix-batch-cert l0, gate env.
        // The HTTP 1024-row checkpoint default applies; joining must still be armed.
        auto gate_args=args;
        const auto block_rows_flag=std::find(gate_args.begin(),gate_args.end(),"--prefix-cache-block-rows");
        if(block_rows_flag!=gate_args.end()) gate_args.erase(block_rows_flag,block_rows_flag+2);
        *(std::find(gate_args.begin(),gate_args.end(),"--http-batch")+1)="3";
        // The gate's pinned known-prefix window (explicit T8192 and tails), as srvargs passes it.
        gate_args.insert(gate_args.end(),{"--known-prefix-window","--known-prefix-full-rows",
            "--known-prefix-workspace-budget-bytes","8589934592","--known-prefix-host-metadata-budget-bytes","1073741824",
            "--known-prefix-uma-budget-bytes","128849018880","--known-prefix-host-headroom-bytes","4294967296",
            "--known-prefix-output-policy","terminal-row","--known-prefix-expert-union","--known-window-router-rows",
            "--known-prefix-prefill-rows","8192","--known-prefix-tail-rows","1024,256,64,16",
            "--prefix-cache-host-bytes","4294967296","--prefix-batch-cert","l0"});
        RequestedOptions gate{parse(gate_args),{{"STRIX_BATCH_PROMPT_SMALL","0"},
            {"STRIX_PREFIX_CACHE_DIGEST","1"},{"STRIX_PREFIX_CACHE_DIGEST_COMPONENTS","1"},
            {"STRIX_FINAL_STATE_DIGEST","1"},{"STRIX_PROMPT_STATE_DIGEST","1"},{"STRIX_PREFIX_CACHE_LOOKUP","1"},
            {"STRIX_PREFIX_CACHE_INFLIGHT","1"},{"STRIX_PROMPT_BATCH","0"},{"STRIX_PROMPT_WINDOW_COHORT","0"},
            {"STRIX_BATCH_NEST_ROWS","0"},{"STRIX_MTP_BATCH_LOG","a"}}};
        // The resolver reads memory only through PrefillMemory: check the fixture and a
        // small CI-runner-like observation (16 GiB available/GTT); an explicit T and host
        // budget keep the gate route identical on any host.
        auto runner_memory=memory;
        runner_memory.available=16*gib_bytes; runner_memory.gtt_limit=16*gib_bytes; runner_memory.gtt_used=0;
        runner_memory.total=16*gib_bytes;
        for(const auto* observed:{&memory,&runner_memory}) {
        const auto resolved_gate=resolve(gate,*observed);
        const auto& o=resolved_gate.options;
        expect(o.prefix_cache && o.prefix_cache_host_bytes==4*gib_bytes && o.prefix_cache_source=="explicit" &&
            !o.cache_activation->automatic_path,"gate route: explicit 4 GiB cache stays ON");
        expect(prefix_batch_cert_execution_mode(o)==1 && ordinary_http_batch_slots(o)!=0,"gate route: strict L0 batch cert");
        if(o.known_prefix_prefill_rows!=8192 || o.prefix_cache_block_rows!=1024)
            std::cerr<<"gate route T="<<o.known_prefix_prefill_rows<<" block_rows="<<o.prefix_cache_block_rows
                <<" http_serve="<<o.http_serve<<" admitted="<<resolved_gate.admitted<<" reason="<<resolved_gate.reason<<'\n';
        expect(o.known_prefix_prefill_rows==8192 && o.prefix_cache_block_rows==1024,
            "gate route: T8192 with the 1024-row checkpoint default");
        expect(qwen_prefix_join_grid(o.prefix_cache_block_rows,o.known_prefix_prefill_rows),
            "gate route: joining accepts the 1024-row grid");
        const auto state=prefix_cache_inflight_state(o);
        if(!state.effective) std::cerr<<"gate inflight reason "<<state.reason<<'\n';
        expect(state.effective && state.reason=="armed" && state.source=="env","gate route: joining armed");
        auto off=gate;off.environment["STRIX_PREFIX_CACHE_INFLIGHT"]="0";
        expect(!prefix_cache_inflight_state(resolve(off,*observed).options).effective,"gate route: joining OFF arm");
        }
    }
    // Native dense-A8 serving geometry at C24576. Reserve-OFF preserves the old
    // envelope; registration is charged to both UMA and GTT. Represent
    // the aggregate extents with synthetic tensors, without model assets.
    {
        constexpr std::uint64_t file_bytes=111334654400ULL, registered=82523500544ULL;
        ModelManifest layout;
        ManifestTensor pinned,cache;
        pinned.byte_size=registered;
        cache.absolute_offset=registered; cache.byte_size=file_bytes-registered;
        cache.residency=ResidencyPolicy::file_backed_candidate;
        layout.tensors={pinned,cache};
        {
            auto overlap=layout;
            overlap.tensors[0].byte_size=4097;
            overlap.tensors[1].absolute_offset=4097;overlap.tensors[1].byte_size=8191;
            PrefillRequirements r;
            auto o=request.options;o.coalesce_gap_bytes=0;
            account_http_registered_target(r,o,overlap,16384,4096);
            expect(r.model_backing==8192 && r.organization.at("target_file_backed_unregistered_bytes")==4096,
                "PLE reservation excludes the page already charged to registration");
            overlap.tensors[0].byte_size=12288;
            account_http_registered_target(r,o,overlap,16384,4096);
            expect(r.organization.at("target_file_backed_unregistered_bytes")==0,
                "fully registered PLE extents receive no second reservation");
        }
        auto production=request;
        production.options.http_batch_cap=0;
        production.options.generate_tokens=1;
        auto observed=memory;
        observed.total=134161678336ULL;
        observed.estimate=[&](const Options& o) {
            const bool large=o.known_prefix_prefill_rows==8192;
            PrefillRequirements r{file_bytes,795857215U,998400U,
                large?8178958352ULL:4089511952ULL,large?666918440U:381115944U,
                large?16338988U:9162796U,1358086208U,796855296U};
            account_http_registered_target(r,o,layout,file_bytes,4096);
            const auto external=qwen_prefill_external_memory(o,model);
            r.external_host=external.host; r.external_device=external.device;
            return r;
        };
        observed.serving=[](const Options& o) {
            const auto slot=o.known_prefix_prefill_rows==8192?919596612U:751824452U;
            return HttpServingMemory{2819704288ULL,2775625728ULL,slot,slot};
        };
        for (bool inspect:{false,true}) for (auto available:{130374098944ULL,131955068928ULL,132000509952ULL}) {
            production.options.print_effective_config=inspect;
            observed.available=available;
            const auto admitted=resolve(production,observed);
            validate_effective_startup(admitted);
            expect(admitted.options.known_prefix_prefill_rows==8192 &&
                admitted.host_compilation.at("target_registered_backing_bytes")==registered &&
                admitted.host_compilation.at("serve_committed_host_peak_bytes")==103224760751ULL,
                "native serving committed target backing admits T8192 in startup and inspection");
            const auto& o=admitted.options;
            const auto r=observed.estimate(o); const auto m=observed.serving(o);
            const auto table=r.organization.at("target_file_backed_unregistered_bytes");
            expect(table==(file_bytes+4095U)/4096U*4096U-registered,
                "HTTP backing inventory includes page-rounded unregistered PLE bytes once");
            auto reserved=production;reserved.environment["STRIX_PLE_SERVE_RESERVE"]="1";
            expect(resolve(reserved,observed).options.known_prefix_prefill_rows==4096,
                "warmed PLE reservation prevents T8192 consuming the file cache under pressure");
            auto roomy=observed;roomy.available=200000000000ULL;
            reserved.options.known_prefill_uma_explicit=true;reserved.options.known_prefill_uma_budget=200000000000ULL;
            const auto warm=resolve(reserved,roomy);
            expect(warm.options.known_prefix_prefill_rows==8192 &&
                warm.host_compilation.at("serve_committed_host_peak_bytes")==103224760751ULL+table,
                "HTTP automatic prefill reserves warmed PLE backing exactly once");
            reserved.environment["STRIX_PLE_WARM"]="0";
            expect(resolve(reserved,roomy).host_compilation.at("serve_committed_host_peak_bytes")==103224760751ULL,
                "PLE warm OFF restores automatic prefill backing total");
            const auto primary=r.model_backing+r.state_d1+r.serial_scratch+r.workspace+r.tails+
                r.metadata+r.runtime_payload+r.external_host;
            const auto capacity=resolve_http_bridge_capacity(gib_bytes,m,available-primary,
                *observed.gtt_limit-*observed.gtt_used-r.device_base-r.workspace-r.tails-r.external_device,
                o.known_prefill_host_headroom,o.known_prefill_host_headroom);
            expect(capacity.slots>=1,"T8192 reserves at least one runnable slot after the primary");
            auto tighter=observed; tighter.available=100000000000ULL;
            expect(resolve(production,tighter).options.known_prefix_prefill_rows==4096,
                "native serving still falls back under host pressure");
            tighter=observed; tighter.gtt_limit=100000000000ULL;
            expect(resolve(production,tighter).options.known_prefix_prefill_rows==4096,
                "registered target also consumes GTT and still falls back under GTT pressure");
        }
    }
    // Synthetic verified-asset/owner evidence, deliberately far below ceilings.
    memory.serving=[](const Options&) { return HttpServingMemory{gib_bytes,gib_bytes,gib_bytes,gib_bytes}; };
    const auto startup=resolve(request,memory);
    expect(startup.options.known_prefix_prefill_rows==8192 && startup.options.automatic_prefill,
        "serve_prefill_production_mtp_ceilings: budget ceilings cannot reject the shared tiled owner");
    validate_effective_startup(startup);
    expect(startup.options.known_prefix_window && startup.options.known_prefix_full_rows &&
        startup.options.known_prefix_expert_union && startup.options.known_window_router_rows &&
        startup.options.known_prefix_tail_rows==std::vector<std::uint32_t>({1024,256,64,16}) &&
        startup.options.prefix_cache_block_rows==0 &&
        numerical_profile_pair(startup.options).prefill==dense_a8_profile,
        "serve_prefill_production_mtp_ceilings: preserve explicit route organizations, tails, cache and arithmetic");
    auto inspect_args=args; inspect_args.emplace_back("--print-effective-config");
    RequestedOptions inspection{parse(inspect_args),request.environment};
    const auto dry=resolve(inspection,memory);
    validate_effective_startup(dry);
    auto execution_view=dry;
    execution_view.sources.erase("--print-effective-config");
    expect(dry.options.http_serve && dry.options.http_batch==startup.options.http_batch &&
        effective_config_json(startup)==effective_config_json(execution_view) && startup.serve_prefill_notice==dry.serve_prefill_notice,
        "serve_prefill_inspection_parity: identical startup and dry-run admission");
    auto rejected=memory;
    rejected.estimate=[](const Options&) -> PrefillRequirements {
        throw std::invalid_argument("plan failure: \"test\"\nreason");
    };
    const auto failure=resolve(inspection,rejected);
    for(const auto rows : {8192,4096,2048,1024,256,64,16})
        expect(failure.serve_prefill_notice.find("T"+std::to_string(rows)+":plan failure:")!=std::string::npos,
            "serve_prefill_rejection_details: retain exception text for every candidate");
    const auto receipt=serve::parse_json(effective_config_json(failure));
    expect(receipt.find("serve_prefill_notice")->as_string()==failure.serve_prefill_notice,
        "serve_prefill_rejection_details: dry-run JSON escapes complete exception text");
    bool refused=false;
    try { validate_effective_startup(failure); } catch(const std::invalid_argument&) {refused=true;}
    expect(refused,"serve_prefill_inspection_parity: failed admission also fails HTTP preflight");
    rejected=memory; rejected.available=1;
    expect(resolve(request,rejected).serve_prefill_notice.find("host-available needs")!=std::string::npos,
        "serve_prefill_rejection_details: name failed memory bound and byte counts");
    for (const char* name : {"STRIX_SERVE_PREFILL","STRIX_KNOWN_PREFIX_WINDOW","STRIX_KNOWN_PREFIX_FULL_ROWS",
            "STRIX_KNOWN_PREFIX_EXPERT_UNION","STRIX_KNOWN_WINDOW_ROUTER_ROWS","STRIX_KNOWN_PREFIX_PREFILL_ROWS"}) {
        auto off=request; off.environment[name]="0";
        expect(!resolve(off,memory).options.known_prefix_prefill_rows,
            "serve_prefill_production_guards: every off switch wins");
    }
    auto baseline=request; baseline.options.baseline=true;
    expect(!resolve(baseline,memory).options.known_prefix_prefill_rows,
        "serve_prefill_production_guards: baseline suppresses automatic admission");
    auto explicit_args=args;
    explicit_args.insert(explicit_args.end(),{"--known-prefix-window","--known-prefix-full-rows",
        "--known-prefix-expert-union","--known-window-router-rows","--known-prefix-prefill-rows","2048",
        "--known-prefix-workspace-budget-bytes","8589934592","--known-prefix-host-metadata-budget-bytes","1073741824",
        "--known-prefix-uma-budget-bytes","128849018880","--known-prefix-host-headroom-bytes","4294967296",
        "--known-prefix-output-policy","terminal-row","--known-prefix-tail-rows","1024,256,64,16"});
    const auto explicit_config=resolve({parse(explicit_args),request.environment},memory);
    validate_effective_startup(explicit_config);
    expect(explicit_config.options.known_prefix_prefill_rows==2048 &&
        explicit_config.options.known_prefill_workspace_budget==8*gib_bytes &&
        explicit_config.options.known_prefill_metadata_budget==startup.options.known_prefill_metadata_budget &&
        explicit_config.options.known_prefill_uma_budget==startup.options.known_prefill_uma_budget &&
        explicit_config.options.known_prefill_host_headroom==startup.options.known_prefill_host_headroom &&
        explicit_config.options.known_prefix_tail_rows==startup.options.known_prefix_tail_rows &&
        numerical_profile_pair(explicit_config.options)==numerical_profile_pair(startup.options),
        "serve_prefill_production_guards: explicit T/budgets win and automatic organizations preserve arithmetic");
    // Prefill alone fits at every tile; adding the verified first slot forces
    // a retry under either host-only or GTT-only pressure, including batch 0.
    for (const bool inspect : {false,true}) for (const bool single : {false,true}) {
        auto r=request; r.options.print_effective_config=inspect;
        if (single) { r.options.http_batch_explicit=true; r.options.http_batch_cap=0; r.options.http_batch=0; }
        auto pressure=memory;
        pressure.estimate=[](const Options& o) {
            PrefillRequirements need;
            need.model_backing=2*gib_bytes; need.device_base=gib_bytes;
            need.workspace=o.known_prefix_prefill_rows*(gib_bytes/1024);
            need.metadata=1;
            return need;
        };
        pressure.serving=[](const Options&) { return HttpServingMemory{gib_bytes,gib_bytes,2*gib_bytes,2*gib_bytes}; };
        pressure.available=128*gib_bytes; pressure.gtt_used=0; pressure.gtt_limit=13*gib_bytes;
        auto prefill_only=pressure;
        prefill_only.serving=[](const Options&) { return HttpServingMemory{}; };
        expect(resolve(r,prefill_only).options.known_prefix_prefill_rows==8192,
            "GTT pressure fixture admits prefill alone");
        const auto gtt=resolve(r,pressure);
        expect(gtt.options.known_prefix_prefill_rows==4096 && gtt.serve_prefill_notice.find("GTT-free")!=std::string::npos,
            "first slot GTT pressure retries T even on explicit single/inspection");
        pressure.gtt_limit=128*gib_bytes; pressure.available=15*gib_bytes;
        const auto host=resolve(r,pressure);
        expect(host.options.known_prefix_prefill_rows==4096 && host.serve_prefill_notice.find("host-available")!=std::string::npos,
            "first slot host pressure retries T even on explicit single/inspection");
        r.options.known_prefill_headroom_explicit=true; r.options.known_prefill_host_headroom=0;
        pressure.available=11*gib_bytes;
        prefill_only=pressure; prefill_only.serving=[](const Options&) { return HttpServingMemory{}; };
        expect(resolve(r,prefill_only).options.known_prefix_prefill_rows==8192 &&
            resolve(r,pressure).options.known_prefix_prefill_rows==4096,
            "zero headroom still reserves one actual runnable slot under host pressure");
        pressure.serving=[](const Options&) { return HttpServingMemory{128*gib_bytes,0,0,0}; };
        const auto no_slot=resolve(r,pressure);
        expect(!no_slot.options.known_prefix_prefill_rows,"no first slot declines every automatic T");
        bool invalid=false; try { validate_effective_startup(no_slot); } catch(const std::invalid_argument&) { invalid=true; }
        expect(invalid,"no first slot refuses startup and inspection before allocation");
        pressure.serving={};
        expect(!resolve(r,pressure).options.known_prefix_prefill_rows,"missing verified assets declines");
        for (const auto mode : {0,1,2}) {
            auto copied=r;
            if (mode==0) copied.options.residency=HipModelResidencyMode::resident_copy;
            if (mode==1) copied.options.model_copy_tier=HipModelCopyTier::normalization;
            if (mode==2) copied.environment["STRIX_MODEL_COPY_ALL"]="1";
            copied.options.numerical_profile=Options{}.numerical_profile;
            copied.options.prefill_numerical_profile.reset();
            copied.options.model_copy_payload_bytes=mode==1?std::optional<std::uint64_t>(1):std::nullopt;
            copied.options.model_retained_budget=128*gib_bytes;
            copied.options.model_setup_peak_budget=128*gib_bytes;
            const auto declined=resolve(copied,memory);
            expect(!declined.options.known_prefix_prefill_rows && declined.serve_prefill_notice.find("copy-all")!=std::string::npos,
                "copy/selective/copy-all declines automatic serving prefill in startup and inspection");
            expect(declined.options.residency==copied.options.residency && declined.options.model_copy_tier==copied.options.model_copy_tier,
                "declining automatic T preserves explicit residency");
        }
    }
    // Grouped automatic candidates decline before inspection or execution can
    // commit an unmodeled owner. Explicit T and workspace budgets keep precedence.
    for (bool inspect : {false,true}) for (bool context_fallback : {false,true})
        for (bool explicit_workspace : {false,true}) {
            auto grouped=request; grouped.options.print_effective_config=inspect;
            grouped.environment["STRIX_PROMPT_BATCH_ROWS"]="8192";
            if(context_fallback) grouped.options.context_tokens=4096;
            grouped.options.known_prefill_workspace_explicit=explicit_workspace;
            grouped.options.known_prefill_workspace_budget=128*gib_bytes;
            auto limited=memory;
            if(!context_fallback) limited.estimate=[](const Options& o) {
                if(o.known_prefix_prefill_rows==8192) throw std::runtime_error("memory:fixture");
                PrefillRequirements need; need.workspace=4096; need.metadata=4096; return need;
            };
            const auto declined=resolve(grouped,limited);
            expect(!declined.options.known_prefix_prefill_rows &&
                declined.serve_prefill_notice.find("STRIX_PROMPT_BATCH_ROWS")!=std::string::npos &&
                declined.serve_prefill_notice.find("explicit tiled configuration")!=std::string::npos,
                "context/memory fallback with extra group declines in inspection and execution");
            bool group_refused=false; try { validate_effective_startup(declined); }
            catch(const std::invalid_argument&) { group_refused=true; }
            expect(group_refused,"unsupported automatic group refuses startup before allocations");
            expect(declined.options.known_prefill_workspace_budget==grouped.options.known_prefill_workspace_budget,
                "decline never rewrites workspace limits");
            grouped.options=startup.options; // a valid, fully organized explicit route
            grouped.options.print_effective_config=inspect;
            grouped.options.known_prefix_prefill_explicit=true;
            grouped.options.known_prefix_prefill_rows=2048;
            grouped.options.known_prefill_workspace_explicit=explicit_workspace;
            grouped.options.known_prefill_workspace_budget=128*gib_bytes;
            const auto explicit_group=resolve(grouped,limited);
            expect(explicit_group.options.known_prefix_prefill_rows==2048 &&
                explicit_group.options.known_prefill_workspace_budget==grouped.options.known_prefill_workspace_budget,
                "explicit T/group route and budgets preserved");
        }
    for (const auto value : {"8192","4096","0"}) {
        auto no_extra=request; no_extra.environment["STRIX_PROMPT_BATCH_ROWS"]=value;
        const auto admitted=resolve(no_extra,memory);
        expect(admitted.options.known_prefix_prefill_rows==8192,
            "group value equal to or below selected T creates no extra owner");
    }
    auto geometric=startup.options; geometric.mtp.window_rows=0;
    geometric.mtp.runtime_bytes=geometric.mtp.target_capture_bytes=128*gib_bytes;
    const auto mapped=http_mtp_memory(geometric,model,{},manifest,{},gib_bytes,gib_bytes,4096);
    const auto copied_sidecar=http_mtp_memory(geometric,model,{},manifest,{},2*gib_bytes,gib_bytes,4096);
    expect(copied_sidecar.shared_host==mapped.shared_host+gib_bytes && copied_sidecar.slot_host==mapped.slot_host,
        "sidecar residency setup peak is shared once, independent of per-slot geometry");
    const auto calibration=http_mtp_memory(geometric,model,{},manifest,{},gib_bytes,gib_bytes,8192);
    expect(calibration.shared_host==mapped.shared_host+4U*4096U && calibration.shared_device==mapped.shared_device,
        "verified calibration size changes host peak, not GTT sidecar backing");
    geometric.context_tokens+=1024;
    expect(http_mtp_memory(geometric,model,{},manifest,{},gib_bytes,gib_bytes,4096).slot_host>mapped.slot_host,
        "native predictor pool follows context geometry");
    // Real native owner geometry, independent of the synthetic arena sizes.
    auto owners=startup.options;
    owners.known_prefix_tail_rows.clear();
    const auto primary=qwen_prefill_external_memory(owners,model);
    expect(primary.staging==268435456U && primary.projections==419430400U && primary.gates==131072U,
        "T8192 PLE payloads match native constructor geometry");
    expect(primary.device==primary.staging+primary.projections+primary.gates+primary.snapshots+primary.gdn+primary.csr_fixed+primary.csr_variants &&
        primary.host==primary.device+primary.hashes && primary.gdn>8U,
        "mapped staging, device payloads, hashes and deferred banks counted once");
    owners.known_prefix_tail_rows={1024,256,64,16};
    const auto with_tails=qwen_prefill_external_memory(owners,model);
    expect(with_tails.host>primary.host && with_tails.gdn==primary.gdn*5U,
        "all five primary/tail owners have independent banks and buffers");
    expect(primary.csr_fixed==1315584U && with_tails.csr_fixed==1552896U,
        "native T8192 and primary/default-tail fixed CSR payloads use production layout");
    std::uint64_t all_variants=0;
    for (const auto t : {8192U,1024U,256U,64U,16U}) {
        for (std::uint32_t n=16;n<t;n+=8U) {
            const auto layout=q51_known_rows::scratch_layout(n,512,10,true);
            expect(layout.bytes>=layout.batch_begins+4U*(n*10U+1U),
                "partial native CSR layout covers final batch-begin payload");
            all_variants+=layout.bytes;
        }
    }
    expect(with_tails.csr_variants==all_variants,
        "inventory bounds all retained partial merged/grouped row-count branches");
    for (bool moe : {false,true}) for (bool expert_union : {false,true})
        for (bool q51 : {false,true}) for (bool q80 : {false,true}) {
            expect(q51_known_rows::prefill_csr_enabled(moe,expert_union,q51,q80,8192)==
                (moe && expert_union && (q51 || q80)),"native CSR branch guards shared with inventory");
        }
    auto no_union=owners; no_union.known_prefix_expert_union=false;
    const auto no_csr=qwen_prefill_external_memory(no_union,model);
    expect(!no_csr.csr_fixed && !no_csr.csr_variants,"non-union owner has no CSR allocation");
    auto small_owners=owners; small_owners.known_prefix_prefill_rows=4096;
    expect(qwen_prefill_external_memory(small_owners,model).host<with_tails.host,
        "smaller T reduces actual owner storage");
    SerialSessionStateLayoutRequest layout_request; layout_request.context_tokens=owners.context_tokens;
    const auto layout=build_serial_session_state_layout(model,layout_request);
    auto cert=owners; cert.prefix_batch_cert=2; cert.prefix_cache=true; cert.prefix_cache_host_bytes=gib_bytes;
    cert.prefix_cache_staging_bytes=0;
    std::uint64_t registry_row=0,registry_fixed=0;
    for(const auto& a:layout.allocations) {
        if(a.component==SessionStateComponent::attention_kv || a.component==SessionStateComponent::sparse_index)
            registry_row+=a.bytes/layout.context_tokens;
        else registry_fixed+=(a.bytes+255U)&~UINT64_C(255);
    }
    const auto staging=qwen_http_batch_staging_reserve(cert,model,layout);
    expect(staging==qwen_batch_staging_cap(0,registry_row,registry_fixed,
        model.kv_heads*model.head_dim*4U+model.indexer_head_dim*2U,
        model.hidden_size*model.hc_count*2U,8192),"pre-device target registry matches runtime capture geometry");
    cert.known_prefix_prefill_rows=4096;
    expect(qwen_http_batch_staging_reserve(cert,model,layout)<staging,"default staging follows T");
    cert.prefix_cache_staging_bytes=3*gib_bytes;
    expect(qwen_http_batch_staging_reserve(cert,model,layout)==3*gib_bytes,"explicit staging cap wins");
    for(unsigned guard=0;guard<7;++guard) {
        auto off=cert;
        if(guard==0) off.baseline=true;
        if(guard==1) off.prefix_batch_cert=0;
        if(guard==2) off.prefix_cache=false;
        if(guard==3) off.prefix_cache_host_bytes=0;
        if(guard==4) off.http_serve=false;
        if(guard==5) off.session_management_requested=true;
        if(guard==6) off.http_batch=0;
        expect(!qwen_http_batch_staging_reserve(off,model,layout),"staging mode/route guard");
    }
    auto real_memory=memory;
    real_memory.serving=[](const Options&) { return HttpServingMemory{}; };
    real_memory.estimate=[&](const Options& o) {
        PrefillRequirements r{gib_bytes,gib_bytes,gib_bytes,gib_bytes,4096,4096,0,2*gib_bytes};
        const auto external=qwen_prefill_external_memory(o,model);
        r.external_host=external.host; r.external_device=external.device;
        return r;
    };
    for(bool inspect:{false,true}) for(unsigned stage_mode:{0U,1U,2U}) for(bool gtt:{false,true}) {
        auto r=inspect?inspection:request;
        r.options.prefix_batch_cert=stage_mode?2:0;
        r.options.prefix_cache_host_bytes=stage_mode?gib_bytes:0;
        r.options.prefix_cache_staging_bytes=stage_mode==2?3*gib_bytes:0;
        const auto need=[&](std::uint32_t t,bool device) {
            auto o=owners; o.known_prefix_prefill_rows=t;
            o.prefix_batch_cert=r.options.prefix_batch_cert;
            o.prefix_cache_host_bytes=r.options.prefix_cache_host_bytes;
            o.prefix_cache_staging_bytes=r.options.prefix_cache_staging_bytes;
            const auto e=qwen_prefill_external_memory(o,model);
            return (device?7*gib_bytes:8*gib_bytes+8192U)+(device?e.device:e.host)+
                o.prefix_cache_host_bytes+qwen_http_batch_staging_reserve(o,model,layout);
        };
        const auto boundary=need(4096,gtt)+(need(8192,gtt)-need(4096,gtt))/2U;
        real_memory.available=gtt?128*gib_bytes:boundary;
        real_memory.gtt_limit=gtt?boundary:128*gib_bytes; real_memory.gtt_used=0;
        const auto fit=resolve(r,real_memory);
        expect(fit.options.known_prefix_prefill_rows==4096,"real owner/staging pressure falls back before allocation in inspection and execution");
        validate_effective_startup(fit);
        expect(fit.host_compilation.at("serve_batch_staging_bytes")==
            qwen_http_batch_staging_reserve(fit.options,model,layout),"inspection records the runtime shared staging reserve");
        real_memory.available=1;
        const auto fail=resolve(r,real_memory);
        bool refused_all=false;
        try { validate_effective_startup(fail); } catch(const std::exception&) { refused_all=true; }
        expect(!fail.options.known_prefix_prefill_rows && refused_all,"complete minimum failure refuses all candidates");
    }
    // Hash real bytes; exercise the production verifier through the same startup
    // hook for automatic/explicit T and inspection/execution, in both pin fields.
    const std::array<std::byte,3> asset{std::byte{1},std::byte{2},std::byte{3}};
    const auto digest=sha256_bytes(asset);
    const auto lower=model_content_identity_hex(digest);
    for(bool inspect:{false,true}) for(bool explicit_t:{false,true}) for(unsigned casing:{0U,1U,2U}) {
        auto r=inspect?inspection:request;
        if(explicit_t) {
            r.options=startup.options;
            r.options.known_prefix_prefill_explicit=true;
            r.options.print_effective_config=inspect;
        }
        auto pin=lower;
        for(std::size_t i=0;i<pin.size();++i)
            if(casing==1 || (casing==2 && i%2))
                if(pin[i]>='a' && pin[i]<='f') pin[i]=static_cast<char>(pin[i]-'a'+'A');
        r.options.mtp.sidecar_sha256=pin; r.options.mtp.golden_sha256=pin;
        r.options.mtp.sidecar_bytes=3; r.options.mtp.golden_max_bytes=3;
        auto pinned=memory;
        pinned.serving=[&](const Options& o) {
            verify_mtp_serving_assets(o.mtp,digest,3,digest,3);
            return HttpServingMemory{};
        };
        const auto valid=resolve(r,pinned);
        validate_effective_startup(valid);
        validate_http_serving_memory(valid.options,pinned);
        expect(valid.options.known_prefix_prefill_rows==(explicit_t?startup.options.known_prefix_prefill_rows:8192),
            "valid case-insensitive pins preserve automatic and explicit T");
        for(bool golden:{false,true}) {
            auto bad=valid.options;
            auto& wrong=golden?bad.mtp.golden_sha256:bad.mtp.sidecar_sha256;
            wrong[0]=wrong[0]=='0'?'1':'0';
            bool rejected_pin=false;
            try { validate_http_serving_memory(bad,pinned); } catch(const std::invalid_argument&) { rejected_pin=true; }
            expect(rejected_pin,"real digest mismatch rejected for either asset before inspection exit");
            if(!explicit_t) {
                auto bad_request=r; bad_request.options.mtp=bad.mtp;
                expect(!resolve(bad_request,pinned).options.known_prefix_prefill_rows,"mismatch refuses every automatic candidate");
            }
        }
    }
    std::cout << "serve prefill admission: " << failures << " failures\n";
}

void yarn_owner_inventory_regression(RequestedOptions request, const gguf::FileInfo& info,
    const ModelManifest& manifest) {
    auto model = ModelContract::qwen38_flash_next();
    model.qwen_rotary_table = mock_rotary(4U);
    auto& o = request.options;
    o.rotary = RotaryDescriptor::yarn(4U); o.context_tokens = 1048576U;
    o.tokens.assign(1048320U,1U); o.generate_tokens = 128U;
    o.known_prefix_prefill_rows = 2048U;
    o.known_prefix_tail_rows.clear(); o.known_prefix_ladder_tail_rows = {1024U,256U,64U,16U};
    o.yarn_mtp_pipeline = false;
    SerialSessionStateLayoutRequest layout_request; layout_request.context_tokens = o.context_tokens;
    const auto state = build_serial_session_state_layout(model,layout_request);
    const auto inventory = [&](const Options& candidate) {
        PrefillRequirements r;
        r.state_d1 = state.device_total_bytes + state.host_total_bytes * 2U + state.host_alignment - 1U;
        r.model_backing = 111334654400ULL;
        r.workspace = 639172624ULL + 167772160U + 993280U;
        append_yarn_owner_requirements(candidate,model,state,r.model_backing,r,
            {true,167772160U,993280U,0U,168765440U});
        return r;
    };
    // Project-owned regression operands, with unrelated owners aggregated.
    // Both historical columns are estimates at T2048, not native allocations.
    struct Row { const char* name; std::uint64_t before, inflated; };
    const std::array rows{
        Row{"dense_tiled_f16",33816576U,337641472U},
        Row{"gdn_core16",25427968U,101711872U},
        Row{"hc_fold_workspaces",173015040U,692060160U},
        Row{"merged_routed_csr",291584U,1274624U},
        Row{"ple_projections_and_gates",104890368U,419561472U},
        Row{"ple_snapshot_and_hashes",790528U,2609152U},
        Row{"ple_staging",67108864U,268435456U},
        Row{"prefill_workspace_and_status",664338448U,639172624U},
        Row{"routed_csr",332544U,1315584U},
        Row{"native_setup_growth_peak",0U,168820736U},
        Row{"ple_packed_growth_peak",0U,41943040U},
        Row{"shared_pre_f16",0U,11534336U}};
    std::uint64_t before = 0, inflated = 0, corrected = 0;
    const auto sized = inventory(o);
    for (const auto& row : rows) {
        before += row.before; inflated += row.inflated;
        corrected += sized.owners.at(row.name);
    }
    constexpr auto unchanged = 150639567943ULL - 1070011920ULL;
    expect(unchanged + before == 150639567943ULL && unchanged + inflated == 152255636551ULL &&
        inflated - before == 1616068608ULL,"1M f4 T2048 historical prompt totals and complete delta");
    expect(sized.owners.at("ple_staging") == 67108864U &&
        sized.owners.at("ple_projections_and_gates") == 104890368U &&
        sized.owners.at("gdn_core16") == 25427968U &&
        sized.owners.at("hc_fold_workspaces") == 173015040U &&
        sized.owners.at("routed_csr") == 332544U && sized.owners.at("merged_routed_csr") == 291584U,
        "call-sized allocations use chosen T, including the merged tail remainder");
    expect(sized.owners.at("dense_tiled_f16") == 33816576U &&
        sized.owners.at("native_setup_growth_peak") == 0U &&
        sized.owners.at("shared_pre_f16") == 2883584U &&
        sized.owners.at("ple_packed_growth_peak") == 20971520U && // one T2048 slab: no replacement on a fresh MTP CLI
        sized.owners.at("hc_f16_weights_low_norm_inverse") == 187302144U &&
        sized.owners.at("moe_f16_input_and_da16") == 146800640U,
        "stream scratch, retained packed slab and fixed setup pools are counted separately");
    for (const bool native : {false,true}) {
        auto untouched = o;
        if (native) untouched.rotary = RotaryDescriptor::native();
        else untouched.mtp.enabled = false;
        const auto original = inventory(untouched);
        for (const auto& row : rows)
            expect(original.owners.at(row.name) == row.inflated,"serial and non-YaRN owner accounting unchanged");
    }
    // Preserve the actual inventory's T dependence; hold the other phase
    // owners fixed to isolate admission from compiler/workspace variation.
    for (const auto context : {262144U,1048576U}) for(const bool mtp : {false,true})
        for (const auto tile : {8192U,4096U,2048U}) {
            auto candidate=o;candidate.context_tokens=context;
            candidate.rotary=RotaryDescriptor::yarn(context==262144U?2U:4U);
            candidate.known_prefix_prefill_rows=tile;candidate.known_prefix_tail_rows={1024U,256U,16U};
            candidate.mtp.enabled=mtp;
            candidate.resolved_prefill_option_mask=prefill_option_bit(PrefillOption::gdn_fp16_chunk);
            const auto charged=inventory(candidate);
            candidate.resolved_prefill_option_mask=0U;
            const auto off=inventory(candidate);
            std::uint64_t delta=0;
            const auto check=[&](const std::string& prefix,std::uint64_t count) {
                const auto n=(count+31U)/32U,padded=n*32U;
                // Allocation expressions in hip_gdn_fp16_chunk.inc, including
                // a partial chunk tail, with tiled_in's projection width.
                const auto projection=2U*count*((2U*16U+48U)*128U+48U*128U);
                const auto scratch=2U*(2U*16U*padded*128U+48U*n*2U*32U*32U)+
                    8U*48U*n*32U+2U*48U*padded*128U;
                expect(charged.owners.at(prefix+"gdn_fp16_chunk_projection_bf16")==projection &&
                    charged.owners.at(prefix+"gdn_fp16_chunk_scratch")==scratch &&
                    !off.owners.contains(prefix+"gdn_fp16_chunk_scratch"),
                    "candidate ON charges exact primary/tail allocation formulas; OFF adds no owner");
                for(const auto* suffix:{"gdn_fp16_chunk_projection_bf16","gdn_fp16_chunk_scratch"})
                    expect(charged.owner_phases.at(prefix+suffix)==(mtp_prompt|mtp_decode|mtp_drain),
                        "candidate slots survive prompt retirement and are absent during setup/calibration");
                delta+=projection+scratch;
            };
            check("",tile);
            expect(delta==static_cast<std::uint64_t>(tile)*59776U,
                "candidate primary scales from 467 MiB at T8192 to 116.75 MiB at T2048");
            for(const auto tail:candidate.known_prefix_tail_rows) check("tail_"+std::to_string(tail)+"_",tail);
            expect(charged.device_base==off.device_base+delta,"all persistent chunk stream owners enter GTT once");
            for(const auto& [name,bytes]:off.owners)
                expect(charged.owners.at(name)==bytes,"candidate charge leaves other owners unchanged");
        }
    request.environment = {{"STRIX_PLE_ROLLING","0"}};
    PrefillMemory memory;
    memory.total = 127473287168ULL; memory.available = 110231265280ULL;
    memory.gtt_limit = 125829120000ULL; memory.gtt_used = 18231255040ULL;
    memory.verify_ple_credit = [](const Options&,const Environment&) { return 28000000000ULL; };
    memory.estimate = [&](const Options& candidate) {
        auto r = inventory(candidate);
        std::uint64_t variable = 0;
        for (const auto& row : rows) variable += r.owners.at(row.name);
        r.owners = {{"model_backing",r.model_backing},
            {"other_phase_owners",unchanged-r.model_backing-candidate.rotary.table_bytes()},{"prefill_owners",variable}};
        r.device_base = 40ULL << 30U; r.metadata = 1U;
        return r;
    };
    const auto resolve = [&](const PrefillMemory& mem) {
        return resolve_effective_config(request,model,info,manifest,validated_model_content_identity,{true,true,true},mem);
    };
    const auto admitted = resolve(memory);
    if (!admitted.admitted) std::cerr << "inventory admission: " << admitted.error << '\n';
    expect(admitted.admitted && admitted.options.known_prefix_prefill_rows == 4096U,
        "1M f4 full census admits a smaller exact tile instead of serial");
    expect(admitted.options.context_tokens == o.context_tokens &&
        admitted.options.prefill_numerical_profile == DecodeNumericalProfile::fast_prefill_f16wmma_v1,
        "tile storage fallback preserves context and numerical profile");
    auto ample = memory; ample.total = ample.available = ample.gtt_limit = 1ULL << 40U; ample.gtt_used = 0U;
    for(const auto context:{262144U,1048576U}) for(const bool mtp:{false,true}) {
        auto candidate=request;
        if(!mtp) {
            candidate.options=Options{};
            candidate.options.context_explicit=true;
            candidate.options.generate_tokens=128U;
        }
        auto candidate_model=model;candidate_model.qwen_rotary_table=mock_rotary(context==262144U?2U:4U);
        candidate.options.rotary=RotaryDescriptor::yarn(context==262144U?2U:4U);
        candidate.options.context_tokens=context;candidate.options.tokens.resize(context-256U);
        if(mtp) candidate.options.prefill_numerical_profile=DecodeNumericalProfile::fast_prefill_f16wmma_v1;
        candidate.options.prefill_disabled=prefill_option_bit(PrefillOption::hc_output_wmma);
        candidate.environment["STRIX_QPF2K_GDN_FP16_CHUNK"]="1";
        auto chunk_memory=ample;
        const auto original_estimate=memory.estimate;
        chunk_memory.estimate=[&](const Options& trial) {
            auto required=original_estimate(trial);
            const auto real=inventory(trial);
            for(const auto& [name,bytes]:real.owners) if(name.find("gdn_fp16_chunk_")!=std::string::npos) {
                required.owners[name]=bytes;
                required.owner_phases[name]=mtp_prompt|mtp_decode|mtp_drain;
                required.device_base=prefill_checked_add(required.device_base,bytes);
            }
            return required;
        };
        const auto run=[&](const RequestedOptions& arm) {
            return resolve_effective_config(arm,candidate_model,info,manifest,
                validated_model_content_identity,{true,true,true},chunk_memory);
        };
        const auto roomy=run(candidate);
        if(!roomy.admitted) std::cerr << "chunk admission C=" << context << " mtp=" << mtp << ": " << roomy.error << '\n';
        expect(roomy.admitted && roomy.prefill_option_mask & prefill_option_bit(PrefillOption::gdn_fp16_chunk),
            "candidate ON is admitted on serial and explicit-profile MTP");
        // Give the smaller tile exactly the physical incremental room it needs.
        auto small=roomy.options;small.known_prefix_prefill_rows=2048U;
        apply_default_known_prefill_tails(small,"12");
        auto needed=chunk_memory.estimate(small);needed.owners["rotary_table"]=small.rotary.table_bytes();
        const auto peak=prefill_owner_phase_peaks(needed.owners,needed.owner_phases);
        chunk_memory.available=*std::max_element(peak.begin(),peak.end())-needed.model_backing+prefill_physical_margin;
        const auto fit=run(candidate);
        expect(fit.admitted && fit.options.known_prefix_prefill_rows==2048U &&
            fit.admission.at("owner_gdn_fp16_chunk_projection_bf16_bytes")+
                fit.admission.at("owner_gdn_fp16_chunk_scratch_bytes")==122421248U,
            "262K f2 / 1M f4 candidate-on ladder charges the allocation formula and fits T2048");
        if(mtp) expect(fit.admission.at("storage_ladder_step")==4U,"candidate-on MTP reaches the final fitting storage step");
        for(const bool master:{false,true}) {
            auto disabled=candidate;
            if(master) disabled.options.prefill_exact_fp16=true;
            else disabled.environment["STRIX_QPF2K_GDN_FP16_CHUNK"]="0";
            const auto off=run(disabled);
            expect(off.admitted && !off.admission.contains("owner_gdn_fp16_chunk_scratch_bytes"),
                "individual and master OFF preserve candidate-free admission");
        }
    }
    const auto initial = resolve(ample);
    if (!initial.admitted) { std::cerr << "inventory initial: " << initial.error << '\n'; return; }
    expect(initial.admitted && initial.options.known_prefix_prefill_rows == 8192U,"late retry starts at T8192");
    auto post = memory;
    auto pinned_tile = initial.options; pinned_tile.known_prefix_prefill_rows = 4096U;
    apply_default_known_prefill_tails(pinned_tile,"12");
    const auto pinned_required = memory.estimate(pinned_tile);
    std::uint64_t pinned_payload = 0;
    for (const auto& [name,bytes] : pinned_required.owners) { (void)name; pinned_payload += bytes; }
    post.available = pinned_payload - pinned_required.model_backing + prefill_physical_margin;
    post.verify_ple_credit = [](const Options&,const Environment&) -> std::uint64_t {
        throw std::runtime_error("late retry must reuse the verified live lease");
    };
    const auto late = readmit_registered_yarn_mtp(request,initial,model,info,manifest,
        validated_model_content_identity,{true,true,true},post);
    if (!late.admitted || late.options.known_prefix_prefill_rows != 4096U)
        std::cerr << "late inventory retry T=" << late.options.known_prefix_prefill_rows << " " << late.error
                  << " late incremental=" << late.admission.at("incremental_bytes")
                  << " expected=" << admitted.admission.at("incremental_bytes")
                  << " fallback=" << late.tile_fallback << '\n';
    expect(late.admitted && late.options.known_prefix_prefill_rows == 4096U &&
        late.admission.at("storage_ladder_step") == 3U && late.admission.at("post_registration_readmission") == 1U &&
        late.ple_reclaim_credit == initial.ple_reclaim_credit,"post-registration retry shrinks before state allocation");
    expect(yarn_mtp_storage_receipt(late).find("\"window_capacity_rows\":4096") != std::string::npos &&
        effective_config_json(late).find("\"storage_ladder_step\":3") != std::string::npos,
        "final receipt and inspection carry chosen T and step");
    expect(late.environment.at("STRIX_QSA_SCORE_RING_ROWS") == std::to_string(initial.admission.at("qsa_score_ring_rows")) &&
        late.environment.at("STRIX_PREFILL_PIPE") == std::to_string(initial.admission.at("prefill_pipe")),
        "late retry pins implicit step-zero storage and keeps the live pipeline");
    post.available = 1U;
    const auto late_refusal = readmit_registered_yarn_mtp(request,initial,model,info,manifest,
        validated_model_content_identity,{true,true,true},post);
    expect(!late_refusal.admitted,"late retry cannot admit absent physical margin");
    expect(late_refusal.admission.at("available_bytes")==1U &&
        late_refusal.admission.at("observed_mem_available_bytes")==1U &&
        late_refusal.admission.at("gtt_used_bytes")==*post.gtt_used &&
        late_refusal.admission.at("resident_rotary_table_bytes")==o.rotary.table_bytes() &&
        late_refusal.admission.at("remaining_incremental_bytes")==
            late_refusal.admission.at("incremental_bytes")-o.rotary.table_bytes() &&
        late_refusal.admission.at("remaining_device_growth_bytes")==
            late_refusal.admission.at("device_growth_bytes")-o.rotary.table_bytes(),
        "late refusal reports raw observations and remaining allocation");
    // Registration can pass the older estimate yet miss the runtime growth
    // margin by only a few MiB. Use the same host envelope as the runtime.
    auto physical_post=memory;
    physical_post.total=physical_post.gtt_limit=1ULL<<40U;
    physical_post.gtt_used=o.rotary.table_bytes();
    physical_post.available=pinned_payload-pinned_required.model_backing+o.rotary.table_bytes()+
        prefill_physical_margin-(8U<<20U);
    expect(prefill_physical_margin_fits(*physical_post.total,*physical_post.available,
        pinned_payload+o.rotary.table_bytes()-late.ple_reclaim_credit,
        pinned_payload-pinned_required.model_backing),"step 3 passes the older table-adjusted estimate");
    std::vector<std::uint32_t> tried;
    const auto physical_estimate=[&](const Options& trial) {
        tried.push_back(trial.known_prefix_prefill_rows);
        auto required=memory.estimate(trial);
        std::uint64_t payload=0;
        for (const auto& [name,bytes]:required.owners) { (void)name; payload+=bytes; }
        required.registered_plan=RegisteredPrefillPlan{0,0,payload-required.model_backing+(8U<<20U),0};
        return required;
    };
    physical_post.estimate=physical_estimate;
    const auto physical_fit=readmit_registered_yarn_mtp(request,late,model,info,manifest,
        validated_model_content_identity,{true,true,true},physical_post);
    expect(physical_fit.admitted && physical_fit.admission.at("storage_ladder_step")==4U &&
        physical_fit.options.known_prefix_prefill_rows==2048U && tried==std::vector<std::uint32_t>{4096U,2048U},
        "step 3 estimate fits but physical margin refusal advances to step 4 without a larger T");
    expect(physical_fit.admission.at("refused_step_3_T4096_incremental_shortfall")==8U<<20U &&
        physical_fit.tile_fallback.find("incremental_shortfall=8388608")!=std::string::npos,
        "every refused step retains its exact physical shortfall in admission and fallback diagnostics");
    for (const auto* name:{"STRIX_PLE_ROLLING_POSITIONS","STRIX_PLE_ROLLING_CAP_BYTES","STRIX_QSA_SCORE_RING_ROWS","STRIX_PREFILL_PIPE"})
        expect(physical_fit.environment.at(name)==late.environment.at(name),"allocated storage pins survive physical-margin retry");
    expect(physical_fit.options.context_tokens==late.options.context_tokens &&
        physical_fit.options.prefill_numerical_profile==late.options.prefill_numerical_profile &&
        physical_fit.ple_reclaim_credit==late.ple_reclaim_credit,"physical retry preserves identity, capacity and verified credit");
    const auto physical_2048=physical_estimate(physical_fit.options);
    physical_post.available=physical_fit.admission.at("registered_physical_incremental_bytes")+
        prefill_physical_margin-(4U<<20U);
    tried.clear();
    const auto physical_refusal=readmit_registered_yarn_mtp(request,late,model,info,manifest,
        validated_model_content_identity,{true,true,true},physical_post);
    expect(!physical_refusal.admitted && tried==std::vector<std::uint32_t>{4096U,2048U} &&
        physical_refusal.admission.at("refused_step_4_T2048_incremental_shortfall")==4U<<20U &&
        physical_refusal.error.find("refused_step_3_T4096_incremental_shortfall")!=std::string::npos &&
        physical_refusal.error.find("refused_step_4_T2048_incremental_shortfall")!=std::string::npos,
        "final physical refusal is bounded and lists both shortfalls for serial fallback");
    bool serial_called=false;
    MtpBundleSetupRetry bundle; bundle.selected=true;
    const auto fallback=retry_mtp_bundle_setup(bundle,std::runtime_error(physical_refusal.error),[&] {
        serial_called=yarn_bundle_serial_retry; return 23;
    });
    expect(fallback==23 && serial_called && !yarn_bundle_serial_retry,"exhausted ladder uses clean existing serial fallback");
    // A later observation can invalidate an already admitted step. The next
    // retry must skip it, preserve its refusal and terminate at T2048.
    auto raced=late;
    record_registered_yarn_refusal(raced,"prefill physical admission",physical_post,
        physical_fit.admission.at("registered_physical_retained_bytes"),physical_2048.registered_plan->growth,
        physical_2048.registered_plan->growth);
    physical_post.available=1ULL<<40U;
    tried.clear();
    const auto continued=readmit_registered_yarn_mtp(request,raced,model,info,manifest,
        validated_model_content_identity,{true,true,true},physical_post,true);
    expect(continued.admitted && tried==std::vector<std::uint32_t>{2048U} &&
        continued.admission.contains("refused_step_3_T4096_incremental_shortfall"),
        "runtime refusal re-reads memory and resumes strictly at the next step");
    tried.clear();
    const auto exhausted=readmit_registered_yarn_mtp(request,continued,model,info,manifest,
        validated_model_content_identity,{true,true,true},physical_post,true);
    expect(!exhausted.admitted && tried.empty(),"no retry remains after T2048");
    unsigned samples=0;
    physical_post.refresh_registered=[&](PrefillMemory& sample) {
        ++samples;
        sample.available=samples==1U?pinned_payload-pinned_required.model_backing+o.rotary.table_bytes()+
            prefill_physical_margin-(8U<<20U):1ULL<<40U;
    };
    tried.clear();
    const auto refreshed=readmit_registered_yarn_mtp(request,late,model,info,manifest,
        validated_model_content_identity,{true,true,true},physical_post);
    expect(refreshed.admitted && samples==2U && tried==std::vector<std::uint32_t>{4096U,2048U} &&
        refreshed.admission.at("post_registration_available_bytes")==1ULL<<40U,
        "each registered trial takes a new observation and reports the admitted sample");
    physical_post.refresh_registered={};
    physical_post.yarn_registered_base_complete=true;
    physical_post.yarn_registered_base=pinned_required.model_backing;
    physical_post.gtt_limit=*physical_post.gtt_used+
        physical_fit.admission.at("refused_step_3_T4096_registered_physical_incremental_bytes")-(1U<<20U);
    const auto gtt_retry=readmit_registered_yarn_mtp(request,late,model,info,manifest,
        validated_model_content_identity,{true,true,true},physical_post);
    expect(gtt_retry.admitted && gtt_retry.admission.at("storage_ladder_step")==4U &&
        gtt_retry.admission.at("refused_step_3_T4096_gtt_shortfall")==1U<<20U,
        "runtime GTT refusal advances without charging the resident base state twice");
    {
        // Runtime diagnostics match the trial when GTT usage is already over its limit.
        auto over=late; auto observed=physical_post;
        observed.gtt_limit=1U<<30U; observed.gtt_used=(1U<<30U)+(3U<<20U);
        record_registered_yarn_refusal(over,"prefill physical admission",observed,0U,0U,5U<<20U);
        expect(over.admission.at("refused_step_3_T4096_gtt_shortfall")==8U<<20U,
            "runtime GTT shortfall adds usage already above the limit to the requested growth");
    }
    {
        // Non-separated runtime: retained includes setup temporary storage,
        // while physical growth never receives the verified file credit.
        UmaOwnerInventory live; live.reservation_upper_bound_bytes=1000; live.known_file_mapping_bytes=800;
        const RegisteredPrefillPlan plan{0,0,100,50};
        const auto registration=registered_prefill_envelope(live,800,plan,{},20,{30,40},200,{});
        const auto runtime=registered_prefill_envelope(live,1000,plan,{},20,{30,40},200,{});
        expect(registration.reservation.retained_reserved_bytes==950 &&
            registration.reservation.setup_peak_reserved_bytes==1040 && registration.incremental==440 &&
            runtime.reservation.setup_peak_reserved_bytes==1040 && runtime.incremental==240,
            "non-separated registration/runtime share retained, temporary and uncredited incremental arithmetic");
    }
    std::cout << "YaRN owner inventory corrected prompt=" << unchanged+corrected
              << " modeled T=" << admitted.options.known_prefix_prefill_rows
              << " spare=" << *memory.total-admitted.admission.at("required_bytes") << '\n';
}

void yarn_admission_tests(const gguf::FileInfo& info, const ModelManifest& manifest) {
    for (const auto factor : {0U,2U,4U}) {
        auto model = ModelContract::qwen38_flash_next();
        if (factor) model.qwen_rotary_table = mock_rotary(factor);
        for (const std::uint64_t capacity : {0U,262143U,262144U,262145U,524288U,524289U,1048576U,1048577U}) {
            const bool fits = capacity && capacity <= model.effective_max_positions();
            bool layout = true, memory = true;
            try { SerialSessionStateLayoutRequest request; request.context_tokens = capacity;
                const auto state = build_serial_session_state_layout(model,request);
                expect(state.context_tokens == capacity,"YaRN layout preserves requested capacity");
            } catch (const std::invalid_argument&) { layout = false; }
            try { MemoryPlanRequest request; request.context_tokens = capacity;
                (void)build_memory_plan(model,request);
            } catch (const std::invalid_argument&) { memory = false; }
            expect(layout == fits && memory == (!factor || fits),"native planning and YaRN execution cap matrix");
        }
    }
    {
        RequestedOptions off;
        off.options = parse({"decoder","MODEL","1","--rope-yarn","off"});
        off.environment["STRIX_QWEN_ROPE_YARN"] = "2";
        const auto native = resolve_effective_config(off,ModelContract::qwen38_flash_next(),info,manifest,
            validated_model_content_identity,{true,true,true},ample_memory());
        expect(native.options.rotary.mode() == RotaryMode::native &&
            native.environment.at("STRIX_QWEN_ROPE_YARN") == "off", "CLI off receipt reports effective native environment");
    }
    auto model = ModelContract::qwen38_flash_next(); model.qwen_rotary_table = mock_rotary(2);
    RequestedOptions request;
    request.options = parse({"decoder","MODEL","1","--rope-yarn","2","--context","300000","--generate","32"});
    request.options.tokens.resize(8192U,1U);
    auto memory = ample_memory(); memory.total = 124U*gib_bytes;
    const auto resolve = [&](const RequestedOptions& r, const PrefillMemory& m) {
        return resolve_effective_config(r,model,info,manifest,validated_model_content_identity,{true,true,true},m);
    };
    {
        constexpr auto key = "STRIX_QPF2K_MOE_ROUTE_BF16";
        auto plain = request; plain.options.tokens.resize(262144U,1U);
        const auto off = resolve(plain,memory);
        expect(off.admitted && !off.options.prefill_moe_route_bf16,"YaRN long prompt candidate defaults off");
        auto fast = plain;
        fast.options.product_profile = qwen::profile::resolve(qwen::profile::Name::fast, {});
        const auto fast_config = resolve(fast, memory);
        Environment exported;
        qwen::profile::apply(fast_config.options.product_profile, [&](const char* name, const char* value) {
            exported[name] = value; return true;
        });
        expect(fast_config.admitted && fast_config.options.prefill_moe_route_bf16 &&
            fast_config.options.product_profile.find("moe_route_bf16")->status == qwen::profile::Status::provisional &&
            exported.at(key) == "1" && exported.at("STRIX_QWEN_PROFILE_OWNED_moe_route_bf16") == "fast:provisional:1",
            "fast YaRN long CLI still admits and exports provisional storage");
        expect(effective_config_json(fast_config).find("fp16-v1+moeroute16") != std::string::npos,
            "fast YaRN serial retains composed storage identity");
        const auto off_pair = numerical_profile_pair(off.options);
        // Exercise the host compiler with synthetic kernels, without HIP. The
        // production prefix key includes this execution identity and pair hash.
        const auto compile_identity = [&](const Options& o) {
            auto req=prompt_compiler_requests(o,model,manifest);
            req.binding.prefill_arithmetic_identity=numerical_profile_pair_identity(numerical_profile_pair(o));
            const auto registry=config_test_registry(model,req.binding);
            return compile_serial_execution(model,registry,req.binding,req.buffers,256).execution_identity;
        };
        auto compiled_memory=memory;
        compiled_memory.estimate = [&](const Options& o) {
            auto required=memory.estimate(o);
            required.organization["prefill_execution_identity"]=compile_identity(o);
            return required;
        };
        const auto compiled_off=resolve(plain,compiled_memory);
        for (const bool cli : {false,true}) {
            auto on = plain;
            if (cli) {
                on.options = parse({"decoder","MODEL","1","--rope-yarn","2","--context","300000",
                    "--generate","32","--prefill-moe-route-bf16","on"});
                on.options.tokens = plain.options.tokens;
                expect(!on.options.any_low_level_flag && !on.options.route_owned && yarn_route_refusal(on.options).empty(),
                    "candidate CLI does not own the YaRN execution route");
            } else on.environment[key] = "1";
            const auto got = resolve(on,memory);
            const auto pair = numerical_profile_pair(got.options);
            const auto compiled_on=resolve(on,compiled_memory);
            expect(compiled_on.admitted && compiled_on.host_compilation.at("prefill_execution_identity")==compile_identity(got.options) &&
                compiled_on.host_compilation!=compiled_off.host_compilation,
                "YaRN candidate reaches admission compiler and separates prefix execution identity");
            expect(got.admitted && got.error.empty() && got.options.prefill_moe_route_bf16 &&
                (pair.fast_components & kFastPrefillMoeRouteBF16) && numerical_profile_pair_admitted(pair),
                "YaRN env and CLI candidate admitted with composed prefill identity");
            expect(pair.decode==off_pair.decode && pair.prefill==off_pair.prefill &&
                numerical_profile_pair_identity(pair)!=numerical_profile_pair_identity(off_pair) &&
                got.rotary_identity.json()==off.rotary_identity.json() && got.admission==off.admission,
                "YaRN candidate separates arithmetic identity without changing rotary or capacity");
            const auto json = effective_config_json(got);
            expect(json.find("\"prefill_moe_route_bf16\":true")!=std::string::npos &&
                json.find("fp16-v1+moeroute16")!=std::string::npos &&
                json.find(std::to_string(moe_prefill_storage_identity(true)))!=std::string::npos,
                "YaRN inspection carries candidate storage descriptor and hash");
            on.options.prefill_moe_route_bf16_requested = false;
            const auto cli_off = resolve(on,memory);
            const auto recompiled_off=resolve(on,compiled_memory);
            expect(recompiled_off.admitted && recompiled_off.host_compilation==compiled_off.host_compilation,
                "YaRN candidate off restores compiled prefix execution identity");
            expect(cli_off.admitted && !cli_off.options.prefill_moe_route_bf16 &&
                numerical_profile_pair(cli_off.options)==off_pair && cli_off.host_compilation==off.host_compilation &&
                cli_off.admission==off.admission && cli_off.rotary_identity.json()==off.rotary_identity.json(),
                "YaRN CLI off restores plain identities and admission including env precedence");
        }
        // The environment candidate parser intentionally accepts every string.
        for (const char* value : {"","0","0suffix","1","on","other"}) {
            auto r=plain; r.environment[key]=value; const auto got=resolve(r,memory);
            const bool enabled=value[0] && value[0]!='0';
            expect(yarn_environment_refusal({{key,value}},2).empty() && got.admitted &&
                got.options.prefill_moe_route_bf16==enabled,"YaRN matches native candidate environment value semantics");
            if (!enabled) expect(numerical_profile_pair(got.options)==off_pair &&
                effective_config_json(got).find("fp16-v1+moeroute16")==std::string::npos,
                "YaRN environment off retains plain storage identity");
        }
        for (const char* value : {"on","off","1","0"}) {
            const auto o=parse({"decoder","MODEL","1","--rope-yarn","2","--context","300000",
                "--prefill-moe-route-bf16",value});
            expect(!o.any_low_level_flag && !o.route_owned,"all candidate CLI values are product switches");
        }
        for (const char* value : {"","perhaps","01","true"}) {
            bool rejected=false;
            try { (void)parse({"decoder","MODEL","1","--rope-yarn","2","--context","300000",
                "--prefill-moe-route-bf16",value}); }
            catch (const std::invalid_argument&) { rejected=true; }
            expect(rejected,"YaRN invalid candidate CLI values refused");
        }
        for (const auto& [name,value] : Environment{{"STRIX_QPF2K_MOE","513"},{"STRIX_QPF2K_MOE_SMALL","16"},
            {"STRIX_QPF2K_MOE_MC","2"},{"STRIX_QPF2K_MOE_DA16","0"},{"STRIX_QPF2K_MOE_ROUTE","6"},
            {"STRIX_INTPREFILL_MOE_TUNE","0"}}) {
            auto env=off.environment; env[key]="1"; env[name]=value;
            auto yarn=off.options, native=yarn; native.rotary=RotaryDescriptor::native();
            std::string yarn_error,native_error;
            resolve_moe_route_bf16(yarn,env,false,true,yarn_error);
            resolve_moe_route_bf16(native,env,false,true,native_error);
            expect(!yarn.prefill_moe_route_bf16 && yarn_error==native_error &&
                yarn_error.find("other writers/readers unsupported")!=std::string::npos,
                "YaRN uncovered writers and readers retain native candidate decline");
        }
        for (unsigned route=0;route<10U;++route) {
            auto yarn=off.options; auto env=off.environment; env[key]="1";
            if(route==0U)yarn.mtp.enabled=true;
            if(route==1U)yarn.http_serve=true;
            if(route==2U)yarn.http_batch=true;
            if(route==3U)yarn.serving_concurrency=2;
            if(route==4U)yarn.prefix_cache_host_bytes=4096;
            if(route==5U)yarn.session_save="checkpoint";
            if(route==6U)yarn.known_prefix_tail_rows={8};
            if(route==7U)yarn.known_prefix_prefill_rows=0;
            auto native=yarn; native.rotary=RotaryDescriptor::native();
            std::string yarn_error,native_error;
            resolve_moe_route_bf16(yarn,env,route==8U,route!=9U,yarn_error);
            resolve_moe_route_bf16(native,env,route==8U,route!=9U,native_error);
            expect(!yarn.prefill_moe_route_bf16 && !yarn_error.empty() && yarn_error==native_error,
                "YaRN unsupported controllers, tails, GSQ and backend keep native decline reasons");
        }
    }
    {
        SerialSessionStateLayoutRequest state_request; state_request.context_tokens = 300000U;
        const auto state = build_serial_session_state_layout(model,state_request);
        auto sized = request.options; sized.known_prefix_prefill_rows = 8192U;
        PrefillRequirements required;
        required.state_d1 = prefill_checked_add(state.device_total_bytes,
            prefill_checked_add(state.host_total_bytes,prefill_checked_add(state.host_total_bytes,state.host_alignment-1U)));
        required.model_backing = 111334654400ULL;
        append_yarn_owner_requirements(sized,model,state,required.model_backing,required);
        const auto native = prefill_native_pools(sized,model);
        expect(required.owners.at("native_setup_growth_peak") == native.bytes.at("native_setup_growth_peak") &&
            required.owners.at("dense_tiled_f16") == native.bytes.at("native_dense_operands") &&
            required.owners.at("ple_packed_growth_peak") == native.bytes.at("native_ple_packed_peak"),
            "YaRN inventory prices bundle setup, dense and retained packed PLE owners");
        expect(required.owners.at("target_kv") == 7372800000ULL && required.owners.at("target_raw_index") == 921600000ULL,
            "YaRN exact target KV and raw index owners at C300000");
        expect(required.owners.at("qsa_shared_prefill_pool") == 38400000U &&
            required.owners.at("qsa_window_pool_and_slab_bound") == 0U,"YaRN one pool across twelve layers and absent window owners");
        std::uint64_t state_sum = 0;
        for (const auto* name : {"target_kv","target_raw_index","gdn_matrix","gdn_convolution","ple_convolution",
                "ngram_history","position_history","state_alignment_and_d1_shadow"})
            state_sum = prefill_checked_add(state_sum,required.owners.at(name));
        expect(state_sum == required.state_d1,"YaRN state owner inventory reconciles aligned payload and shadow");
        expect(required.owners.at("routed_csr") == 1315584U &&
            required.owners.at("merged_routed_csr") == 0U &&
            !required.owners.contains("tail_64_ple_staging"),
            "YaRN primary CSR present, tail/merged owners absent for exact T prompt");
        const auto primary = required;
        {
            // Compact geometry: the ring and terminal stripes are split out of workspace/tails once, the embedded maxima
            // owners drop to zero and the decode pool is named apart from the state shadow.
            auto compact = PrefillRequirements{};
            compact.model_backing = required.model_backing;
            const auto decode = qsa_decode_pool_bytes(sized,model);
            compact.state_d1 = required.state_d1 + decode;
            compact.workspace = 1000000000U;
            compact.tails = 200000000U;
            const YarnPrefillOwnerGeometry geometry{true,671088640U,993280U,decode,672081920U};
            append_yarn_owner_requirements(sized,model,state,compact.model_backing,compact,geometry);
            expect(compact.owners.at("qsa_selection_maxima")==0 &&
                compact.owners.at("qsa_batch_maxima_growth_peak")==0,
                "compact YaRN removes embedded maxima owners");
            expect(compact.owners.at("qsa_score_ring")+compact.owners.at("terminal_logits_stripes")+
                compact.owners.at("prefill_workspace_and_status")+compact.owners.at("tail_workspaces_and_status") ==
                compact.workspace+compact.tails,"compact ring and logits split workspace exactly once");
            expect(compact.owners.at("qsa_decode_pool")==decode &&
                compact.owners.at("state_alignment_and_d1_shadow")==primary.owners.at("state_alignment_and_d1_shadow"),
                "decode pool is separate from state shadow and charged once");
            auto over = compact; over.workspace = 1U; over.tails = 0U;
            bool refused = false;
            try { append_yarn_owner_requirements(sized,model,state,over.model_backing,over,geometry); }
            catch (const std::invalid_argument&) { refused = true; }
            expect(refused,"compact owners larger than the workspace reservation refuse");
        }
        sized.tokens.resize(299000U,1U);
        sized.known_prefix_tail_rows = {64U,16U};
        sized.known_prefix_ladder_tail_rows = {1024U,256U,64U,16U};
        PrefillRequirements with_tails;
        with_tails.state_d1 = primary.state_d1;
        with_tails.model_backing = primary.model_backing;
        with_tails.workspace = 1U;
        append_yarn_owner_requirements(sized,model,state,with_tails.model_backing,with_tails);
        std::uint64_t tail_ple = 0;
        for (const auto rows : {64U,16U}) {
            const auto prefix = "tail_" + std::to_string(rows) + "_";
            for (const auto* owner : {"ple_staging","ple_projections_and_gates","ple_snapshot_and_hashes"})
                tail_ple = prefill_checked_add(tail_ple,with_tails.owners.at(prefix+owner));
            expect(with_tails.owners.at(prefix+"routed_csr") ==
                q51_known_rows::prefill_scratch_bytes(rows,512U,10U),"YaRN each retained tail has independent CSR");
        }
        expect(tail_ple == 7111040U && with_tails.owners.at("merged_routed_csr") ==
            q51_known_rows::prefill_scratch_bytes(3840U,512U,10U),
            "YaRN tail PLE regression and once-retained merged primary CSR");
        // Mock the same compiled owners present/absent. Every T trial uses the
        // same fixture so fallback cannot conceal an envelope or GTT edge.
        auto absent = with_tails;
        std::uint64_t omitted = 0;
        for (const auto& [name,bytes] : with_tails.owners) {
            if ((name.starts_with("tail_") && name != "tail_workspaces_and_status") ||
                name == "routed_csr" || name == "merged_routed_csr") {
                omitted = prefill_checked_add(omitted,bytes);
                absent.owners.erase(name);
            }
        }
        absent.device_base -= omitted;
        std::uint64_t gross = request.options.rotary.table_bytes();
        for (const auto& [name,bytes] : with_tails.owners) { (void)name; gross = prefill_checked_add(gross,bytes); }
        auto edge = memory;
        edge.verify_ple_credit = [](const Options&, const Environment&) { return 16U*gib_bytes; };
        edge.total = gross-16U*gib_bytes+prefill_physical_margin-1U; // physical total edge after verified credit
        edge.estimate = [=](const Options&) { return absent; };
        expect(resolve(request,edge).admitted,"YaRN missing external owners would admit envelope edge");
        edge.estimate = [=](const Options&) { return with_tails; };
        auto refused = resolve(request,edge);
        expect(!refused.admitted && refused.error.find("owner_tail_64_ple_staging_bytes=") != std::string::npos &&
            refused.error.find("owner_merged_routed_csr_bytes=") != std::string::npos &&
            effective_config_json(refused).find("owner_tail_16_routed_csr_bytes") != std::string::npos,
            "YaRN complete owner refusal and effective-config list each external owner");
        edge.total = 124U*gib_bytes;
        edge.gtt_limit = with_tails.device_base + with_tails.workspace + request.options.rotary.table_bytes()-1U;
        edge.estimate = [=](const Options&) { return absent; };
        expect(resolve(request,edge).admitted,"YaRN missing external owners would admit device edge");
        edge.estimate = [=](const Options&) { return with_tails; };
        expect(!resolve(request,edge).admitted,"YaRN external owners count toward device growth independently");
        auto invalid = model; invalid.num_experts = UINT64_MAX;
        bool csr_guarded = false;
        try { auto value = primary; append_yarn_owner_requirements(sized,invalid,state,value.model_backing,value); }
        catch (const std::invalid_argument&) { csr_guarded = true; }
        expect(csr_guarded,"YaRN CSR dimensions checked before narrowing or layout arithmetic");
        auto bad = model; bad.gdn_key_heads = UINT64_MAX;
        bool guarded = false;
        try { append_yarn_owner_requirements(sized,bad,state,required.model_backing,required); }
        catch (const std::overflow_error&) { guarded = true; }
        expect(guarded,"YaRN owner width products checked before context multiplication");
    }
    {
        // Exercise the process collector, including ordinary shell storage paths.
        const auto saved = [](const char* name) -> std::optional<std::string> {
            const auto* value = std::getenv(name);
            return value ? std::optional<std::string>{value} : std::nullopt;
        };
        const auto home = saved("HOME"), xdg = saved("XDG_CACHE_HOME");
        const auto batch_cert = saved("STRIX_PREFIX_BATCH_CERT");
        ::unsetenv("STRIX_PREFIX_BATCH_CERT");
        const auto signature = session_numerical_config("same execution");
        ::setenv("HOME", "storage/home", 1);
        ::setenv("XDG_CACHE_HOME", "storage/cache", 1);
        auto collected = request;
        collected.options.context_tokens = 524288U;
        collected.environment = requested_prefill_environment();
        expect(collected.environment.at("HOME") == "storage/home" &&
            collected.environment.at("XDG_CACHE_HOME") == "storage/cache", "real collector retains storage inputs");
        expect(resolve(collected,memory).admitted, "YaRN admits real collector with shell storage locations");
        expect(session_numerical_config("same execution") == signature, "storage paths do not change execution signature");
        for (const auto* value : {"0", "off", "1"}) {
            ::setenv("STRIX_PREFIX_BATCH_CERT",value,1);
            collected.environment = requested_prefill_environment();
            expect(collected.environment.at("STRIX_PREFIX_BATCH_CERT") == value,"real collector retains certification control");
            expect(resolve(collected,memory).admitted,"YaRN admits operational certification control without cache route");
            auto cli = collected;
            cli.options = parse({"decoder","MODEL","1","--rope-yarn","2","--context","524288",
                "--generate","32","--prefix-batch-cert",std::string_view(value)=="1"?"off":"l0"});
            apply_prefix_cache_environment(cli.options,[&](const char* name)->std::optional<std::string_view> {
                const auto it=cli.environment.find(name);
                return it==cli.environment.end()?std::nullopt:std::optional<std::string_view>{it->second};
            });
            expect(cli.options.prefix_batch_cert==(std::string_view(value)=="1"?0U:1U),"CLI certification mode wins real collected environment");
            expect(resolve(cli,memory).admitted,"YaRN CLI precedence remains admissible");
            auto forbidden=cli; forbidden.options.mtp.enabled=true;forbidden.options.mtp.sample_options=true;
            expect(resolve(forbidden,memory).error=="YaRN MTP cohort/wide/sampling/metadata routes are not supported",
                "cert control preserves the bounded explicit MTP route refusal");
            forbidden=cli; forbidden.options.http_serve=true;
            expect(resolve(forbidden,memory).error=="YaRN on HTTP serving not supported","cert control preserves named HTTP refusal");
            forbidden=cli; forbidden.options.prefix_cache_roundtrip=true;
            expect(resolve(forbidden,memory).error=="YaRN requires plain serial CLI without cache, replay or recording","cert control preserves named cache refusal");
        }
        // A real collected malformed value is ignored only when CLI overrides it.
        ::unsetenv("STRIX_PREFIX_BATCH_CERT");
        auto native_options = parse({"decoder","MODEL","1","--generate","32"});
        ::setenv("STRIX_PREFIX_BATCH_CERT","ON",1);
        const auto malformed_environment = requested_prefill_environment();
        expect(malformed_environment.at("STRIX_PREFIX_BATCH_CERT")=="ON","real collector retains malformed certification control");
        for (const bool yarn : {false,true}) {
            auto malformed = collected;
            malformed.environment = malformed_environment;
            malformed.options = yarn ? collected.options : native_options;
            const auto run = [&](const RequestedOptions& r) {
                return resolve_effective_config(r,yarn?model:ModelContract::qwen38_flash_next(),info,manifest,
                    validated_model_content_identity,{true,true,true},memory);
            };
            bool rejected = false;
            try { (void)run(malformed); }
            catch (const std::invalid_argument& error) {
                rejected = std::string_view(error.what())=="prefix batch certification requires 0|1|capture|l0|l1";
            }
            expect(rejected,"native and YaRN reject malformed env-only certification with shared parser error");
            for (const auto* mode : {"off","l0","capture","l1"}) {
                malformed.options = yarn
                    ? parse({"decoder","MODEL","1","--rope-yarn","2","--context","524288",
                        "--generate","32","--prefix-batch-cert",mode})
                    : parse({"decoder","MODEL","1","--generate","32","--prefix-batch-cert",mode});
                const auto resolved = run(malformed);
                const auto expected = std::string_view(mode)=="off"?0U:std::string_view(mode)=="l0"?1U:
                    std::string_view(mode)=="capture"?2U:3U;
                expect(resolved.error.empty() && (!yarn || resolved.admitted) && resolved.options.prefix_batch_cert==expected,
                    "native and YaRN CLI modes override malformed real collected certification environment");
            }
        }
        if(batch_cert) ::setenv("STRIX_PREFIX_BATCH_CERT",batch_cert->c_str(),1); else ::unsetenv("STRIX_PREFIX_BATCH_CERT");
        collected.environment["STRIX_MOCK_UNCOVERED_ROTARY"] = "1";
        expect(resolve(collected,memory).error ==
            "YaRN execution environment not supported: STRIX_MOCK_UNCOVERED_ROTARY",
            "storage classification still refuses unknown execution controls");
        if (home) ::setenv("HOME",home->c_str(),1); else ::unsetenv("HOME");
        if (xdg) ::setenv("XDG_CACHE_HOME",xdg->c_str(),1); else ::unsetenv("XDG_CACHE_HOME");
    }
    // Reproduce the complete startup environment from the 8192-ID plain
    // command, including the two-bit MoE marker that previously refused.
    auto startup = request;
    qpf2k::initialize([&](const char* name) -> const char* {
        const auto it = startup.environment.find(name);
        return it == startup.environment.end() ? nullptr : it->second.c_str();
    }, [&](const char* name, const char* value) {
        startup.environment.emplace(name,value); return true;
    }, qpf2k::startup_items);
    const Environment recorded_startup{
        {"STRIX_PLE_RECLAIMABLE","auto"}, {"STRIX_QPF2K_DENSE","1"},
        {"STRIX_QPF2K_GDN","1"}, {"STRIX_QPF2K_GDN_CORE16","1"},
        {"STRIX_QPF2K_GDN_GATES_BATCH","1"}, {"STRIX_QPF2K_HC","1"},
        {"STRIX_QPF2K_HC_MASK","14"}, {"STRIX_QPF2K_HC_OUTPUT3","1"},
        {"STRIX_QPF2K_HC_OUTPUT3_BAND","16"}, {"STRIX_QPF2K_MOE_DA16","1"},
        {"STRIX_QPF2K_MOE_DEFAULTED","3"}, {"STRIX_QPF2K_MOE_ROUTE","14"},
        {"STRIX_QPF2K_QSA","1"}, {"STRIX_QPF2K_QSA_MASK","0x2257"},
        {"STRIX_QPF2K_SCHED","14"}};
    expect(startup.environment == recorded_startup, "YaRN exact recorded startup environment");
    // The binary never passes pre-injected startup values as requested ones (requested_prefill_environment):
    // YaRN re-injects the defaults and keeps a defaulted schedule at its audited 12.
    auto startup_request = startup;
    startup_request.environment.erase("STRIX_QPF2K_SCHED");
    {
        const auto resolved = resolve(startup_request,memory);
        expect(resolved.admitted && !resolved.admission.empty() &&
            resolved.environment.at("STRIX_QPF2K_SCHED") == "12" &&
            resolved.sources.at("STRIX_QPF2K_SCHED") == OptionSource::automatic,
            "YaRN exact resolver startup environment reaches full admission with the defaulted schedule at 12");
        const auto plain = resolve(request,memory);
        expect(plain.admitted && plain.environment.at("STRIX_QPF2K_SCHED") == "12",
            "YaRN with the default schedule resolves to 12, not the native default 14");
    }
    // The recorded startup environment as a restarted image sees it: eager code loading re-executes the process
    // before undo_pre_injected, so the new image inherits every injected value. The startup marker hands their
    // provenance over; the binary's capture still excludes them and YaRN resolves the defaulted schedule to 12.
    {
        const auto saved_injected = qpf2k::pre_injected, saved_defaulted = qpf2k::defaulted_items;
        const auto saved_marker = qpf2k::pre_injected_moe_marker;
        const auto get_in = [](const Environment& env) {
            return [&env](const char* name) -> const char* {
                const auto it = env.find(name);
                return it == env.end() ? nullptr : it->second.c_str();
            };
        };
        const auto put_in = [](Environment& env) {
            return [&env](const char* name, const char* value) { return env.emplace(name,value).second; };
        };
        auto parent = request.environment;
        qpf2k::pre_injected = qpf2k::defaulted_items = 0U; qpf2k::pre_injected_moe_marker = false;
        qpf2k::initialize_before_main(get_in(parent),put_in(parent));
        const auto parent_injected = qpf2k::pre_injected;
        auto child = parent; // execv hands the environment over unchanged
        qpf2k::pre_injected = qpf2k::defaulted_items = 0U; qpf2k::pre_injected_moe_marker = false;
        qpf2k::initialize_before_main(get_in(child),put_in(child));
        expect(child == parent && qpf2k::pre_injected == parent_injected && qpf2k::pre_injected_moe_marker &&
            (qpf2k::pre_injected & qpf2k::bit(qpf2k::Item::sched)) && qpf2k::defaulted_items == parent_injected,
            "restarted image keeps the startup injections as defaulted");
        auto restarted = startup;
        restarted.environment = requested_prefill_environment(get_in(child));
        expect(!restarted.environment.contains("STRIX_QPF2K_SCHED") &&
            !restarted.environment.contains("STRIX_QPF2K_MOE_DEFAULTED") &&
            !restarted.environment.contains(qpf2k::pre_injected_env), "restarted capture excludes startup injections");
        const auto resolved = resolve(restarted,memory);
        expect(resolved.admitted && !resolved.admission.empty() &&
            resolved.environment.at("STRIX_QPF2K_SCHED") == "12" &&
            resolved.sources.at("STRIX_QPF2K_SCHED") == OptionSource::automatic &&
            resolved.sources.at("STRIX_QPF2K_MOE_DEFAULTED") == OptionSource::automatic,
            "YaRN after an image restart resolves the defaulted schedule to 12, not a refusal");
        qpf2k::pre_injected = saved_injected; qpf2k::defaulted_items = saved_defaulted;
        qpf2k::pre_injected_moe_marker = saved_marker;
    }
    // An explicit 14 (the native default's value, requested by name) is still refused, as before.
    {
        auto explicit_14 = request;
        explicit_14.environment["STRIX_QPF2K_SCHED"] = "14";
        const auto refused = resolve(explicit_14,memory);
        expect(!refused.admitted && refused.error == "YaRN requires STRIX_QPF2K_SCHED=12; got 14",
            "YaRN refuses an explicit STRIX_QPF2K_SCHED=14");
    }
    for (const auto& item : qpf2k::defaults) {
        expect(yarn_environment_path(item.item) != YarnEnvironmentPath::uncovered_rotary,
            "every resolver item has an audited rotary classification");
        for (const auto* value : {"0", "1", "0xffffffff"}) {
            auto explicit_override = request;
            explicit_override.environment[item.name] = value;
            expect(yarn_environment_refusal(explicit_override.environment,2).empty() ==
                (item.item != qpf2k::Item::sched),
                "explicit overrides use the same rotary classification");
        }
    }
    expect(yarn_environment_path(static_cast<qpf2k::Item>(999U)) == YarnEnvironmentPath::uncovered_rotary,
        "new rotary items fail closed until audited");
    for (const auto* off : {"STRIX_PLE_ROLLING","STRIX_CTX_EXACT_MEMORY"}) {
        auto disabled=request.environment;
        disabled[off]="0";
        disabled["STRIX_PLE_ROLLING_POSITIONS"]="disabled-tuning";
        disabled["STRIX_PLE_ROLLING_CAP_BYTES"]="disabled-tuning";
        expect(yarn_environment_refusal(disabled,2).empty(),"YaRN rolling off ignores disabled tuning");
    }
    for (const auto* knob : {"STRIX_PLE_ROLLING_POSITIONS","STRIX_PLE_ROLLING_CAP_BYTES",
            "STRIX_PLE_RECLAIM_DECODE_BYTES","STRIX_PLE_RECLAIM_AHEAD_ROWS",
            "STRIX_PLE_RECLAIM_STRICT","STRIX_PLE_STAGE_PREFETCH","STRIX_PLE_ROLLING_STATS",
            "STRIX_PLE_DIRECT_IO","STRIX_PLE_DIRECT_IO_STATS"}) {
        auto controls=request.environment; controls[knob]="1";
        expect(yarn_environment_refusal(controls,2).empty(),"YaRN classifies documented PLE controls as no rotary");
        controls[knob]="bad";
        expect(!yarn_environment_refusal(controls,2).empty(),"YaRN active PLE controls reject malformed values");
    }
    auto uncovered = request;
    uncovered.environment["STRIX_MOCK_UNCOVERED_ROTARY"] = "1";
    unsigned callbacks = 0;
    auto observed = memory;
    observed.verify_ple_credit = [&](const Options&, const Environment&) { ++callbacks; return 0U; };
    observed.estimate = [&](const Options&) { ++callbacks; return PrefillRequirements{}; };
    for (const auto* schedule : {"0","8","13","16","12"}) {
        auto scheduled = request;
        scheduled.options.tokens.resize(299000U,1U);
        scheduled.environment["STRIX_QPF2K_SCHED"] = schedule;
        callbacks = 0;
        const auto planned = resolve(scheduled,observed);
        if (std::string_view(schedule) != "12")
            expect(planned.error.find("STRIX_QPF2K_SCHED") != std::string::npos &&
                callbacks == 0U && planned.admission.empty(), "YaRN schedule refuses before reclaim and estimation");
        else {
            const auto admitted = resolve(scheduled,memory);
            expect(admitted.admitted && admitted.options.known_prefix_tail_rows ==
                std::vector<std::uint32_t>{64U,16U}, "YaRN schedule 12 admits the cursor owner ladder");
        }
        auto native = scheduled;
        native.options.rotary = RotaryDescriptor::native();
        native.options.context_tokens = 8192U;
        native.options.tokens.resize(4096U);
        const auto native_plan = resolve_effective_config(native,ModelContract::qwen38_flash_next(),
            info,manifest,validated_model_content_identity,{true,true,true},memory);
        expect(native_plan.environment.at("STRIX_QPF2K_SCHED") == schedule &&
            native_plan.error.find("YaRN") == std::string::npos, "native schedule remains unchanged");
    }
    callbacks = 0;
    auto defaults_off = request;
    defaults_off.environment["STRIX_QPF2K_DEFAULTS"] = "0";
    expect(resolve(defaults_off,observed).error.find("STRIX_QPF2K_SCHED") != std::string::npos && callbacks == 0U,
        "YaRN unset schedule with injection disabled refuses before callbacks");
    const auto rejected = resolve(uncovered,observed);
    expect(rejected.error == "YaRN execution environment not supported: STRIX_MOCK_UNCOVERED_ROTARY" &&
        callbacks == 0U && rejected.admission.empty(), "uncovered rotary refuses by name before admission");
    auto config = resolve(request,memory);
    expect(config.admitted && config.options.context_tokens == 300000U && config.options.known_prefix_prefill_rows == 8192U,
        "YaRN fits without shrinking context");
    expect(config.admission.at("host_reserve_bytes") == 0U && config.ple_reclaim_credit == 0U,
        "YaRN reserve and zero unverified credit");
    expect(effective_config_json(config).find("admission") != std::string::npos &&
        effective_config_json(config).find("table_sha256") != std::string::npos,"YaRN inspection resolved identity and owners");
    auto large = memory;
    const auto estimate = memory.estimate;
    large.estimate = [=](const Options& o) { auto value = estimate(o); value.model_backing = 110U*gib_bytes; return value; };
    config = resolve(request,large);
    expect(!config.admitted && config.error.find("owner_model_backing_bytes") != std::string::npos &&
        config.error.find("verified_ple_credit_bytes=0") != std::string::npos &&
        config.error.find("smallest_T_tried=2048") != std::string::npos,"YaRN refusal lists owners credit limits and smallest T");
    expect(config.tile_fallback == "T8192:insufficient-memory,T4096:insufficient-memory,T2048:insufficient-memory",
        "YaRN fallback order");
    bool execution = false;
    try { require_effective_prefill_execution(config); execution = true; } catch (const std::invalid_argument&) {}
    expect(!execution,"YaRN refuses before any execution/allocation callback");
    large.verify_ple_credit = [](const Options&, const Environment&) { return 16U*gib_bytes; };
    expect(resolve(request,large).admitted,"YaRN verified PLE credit admits");
    auto failed_drop = large;
    unsigned estimates_after_drop_failure = 0;
    failed_drop.verify_ple_credit = [](const Options&, const Environment&) -> std::uint64_t { throw std::runtime_error("required drop not verified"); };
    failed_drop.estimate = [&](const Options&) { ++estimates_after_drop_failure; return PrefillRequirements{}; };
    expect(resolve(request,failed_drop).error.find("required drop not verified") != std::string::npos && !estimates_after_drop_failure,
        "YaRN required reclaim failure refuses before candidate growth or device setup");
    large.available = gib_bytes;
    expect(!resolve(request,large).admitted,"YaRN credit never inflates available headroom");
    auto fallback = memory;
    std::vector<std::uint32_t> trials;
    fallback.estimate = [&](const Options& o) { trials.push_back(o.known_prefix_prefill_rows); auto value = estimate(o);
        if (o.known_prefix_prefill_rows > 2048U) value.workspace = 9U*gib_bytes;
        return value; };
    config = resolve(request,fallback);
    expect(config.admitted && trials == std::vector<std::uint32_t>{8192U,4096U,2048U} &&
        config.options.prefill_numerical_profile == DecodeNumericalProfile::fast_prefill_f16wmma_v1,
        "YaRN T fallback keeps numerical lineage");
    auto reserve = memory;
    reserve.estimate = [=](const Options& o) { auto value = estimate(o); value.model_backing = 103U*gib_bytes; return value; };
    auto legacy_reserve = request;
    legacy_reserve.environment["STRIX_QWEN_YARN_HOST_RESERVE"] = "1";
    expect(!resolve(legacy_reserve,reserve).admitted,"YaRN eight GiB reserve honored when restored");
    const auto legacy = resolve(legacy_reserve,memory);
    const auto current = resolve(request,memory);
    expect(legacy.admitted && current.admitted && legacy.options.known_prefill_host_headroom == yarn_legacy_host_reserve &&
        current.options.known_prefill_host_headroom == 0U && legacy.host_compilation == current.host_compilation,
        "serial YaRN reserve changes admission without changing kernel plans");
    for (const auto* phase : {"setup","calibration","prompt","decode","drain"}) {
        const auto key = std::string("phase_")+phase+"_gross_bytes";
        expect(legacy.admission.at(key) == current.admission.at(key)+yarn_legacy_host_reserve,
            "legacy reserve restored byte-for-byte in every serial phase");
    }
    for(const bool reserve_on:{false,true}) {
        auto route=request;route.environment["STRIX_QWEN_YARN_HOST_RESERVE"]=reserve_on?"1":"0";
        auto edge=memory;const auto admitted=resolve(route,edge);
        expect(admitted.admitted,"serial margin fixture starts admitted");
        // Freeze the owner geometry so a smaller T cannot hide a one-byte deficit.
        const auto fixed=estimate(admitted.options);edge.estimate=[fixed](const Options&){return fixed;};
        edge.available=admitted.admission.at("incremental_bytes")+prefill_physical_margin;
        expect(resolve(route,edge).admitted,"serial YaRN exact 4 GiB margin admits");
        --*edge.available;
        const auto short_by_one=resolve(route,edge);
        expect(!short_by_one.admitted && short_by_one.error.find("4 GiB margin")!=std::string::npos,
            "serial YaRN MemAvailable minus 4 GiB short by one byte refuses in both reserve modes");
        edge=memory;edge.estimate=[fixed](const Options&){return fixed;};
        edge.total=admitted.admission.at("required_bytes")+prefill_physical_margin;
        expect(resolve(route,edge).admitted,"serial YaRN exact MemTotal margin admits");
        --*edge.total;expect(!resolve(route,edge).admitted,"serial YaRN total margin short by one byte refuses");
    }
    auto invalid_reserve = request;
    invalid_reserve.environment["STRIX_QWEN_YARN_HOST_RESERVE"] = "8";
    expect(resolve(invalid_reserve,memory).error.find("host reserve switch must be 0 or 1") != std::string::npos,
        "reserve switch values fail closed");
    auto native_reserve = request;
    native_reserve.options.rotary = RotaryDescriptor::native();
    native_reserve.options.context_tokens = 32768U;
    native_reserve.environment["STRIX_QWEN_YARN_HOST_RESERVE"] = "0";
    const auto native_zero = resolve_effective_config(native_reserve,ModelContract::qwen38_flash_next(),info,manifest,
        validated_model_content_identity,{true,true,true},memory);
    native_reserve.environment["STRIX_QWEN_YARN_HOST_RESERVE"] = "1";
    const auto native_one = resolve_effective_config(native_reserve,ModelContract::qwen38_flash_next(),info,manifest,
        validated_model_content_identity,{true,true,true},memory);
    expect(native_zero.admitted && native_one.admitted && native_zero.admission == native_one.admission &&
        native_zero.host_compilation == native_one.host_compilation &&
        native_zero.options.known_prefill_host_headroom == native_one.options.known_prefill_host_headroom,
        "native admission and plans ignore YaRN reserve switch");
    for (unsigned route = 0; route < 6; ++route) {
        auto bad = request;
        const char* reason = nullptr;
        if (route == 0) { bad.options.mtp.enabled = true; bad.options.mtp.sample_options=true; reason = "YaRN MTP cohort/wide/sampling/metadata routes are not supported"; }
        if (route == 1) { bad.options.http_serve = true; reason = "YaRN on HTTP serving not supported"; }
        if (route == 2) { bad.options.teacher_forced_score = true; reason = "YaRN with teacher scoring not yet qualified"; }
        if (route == 3) { bad.options.gdn_islands = true; reason = "YaRN graph capture path not yet qualified"; }
        if (route == 4) { bad.options.session_save = "checkpoint"; reason = "YaRN with session save/restore not yet qualified"; }
        if (route == 5) { bad.options.prefix_cache_host_bytes = 4096; reason = "YaRN with prefix state cache not yet qualified"; }
        expect(resolve(bad,memory).error == reason,"YaRN named route refusal");
    }
    {
        auto unknown = request; unknown.environment["STRIX_QPF2K_SHADOW_MAX_TOKENS"] = "300000";
        expect(resolve(unknown,memory).error.find("YaRN execution environment not supported") != std::string::npos,
            "YaRN unbounded alternate launcher controls refuse");
        for (const char* name : {"STRIX_QPF2K_T16K", "STRIX_QSA_DECODE_POOL", "STRIX_QSA_DECODE_POOL_CHECK", "STRIX_QSA_DECODE_POOL_BF16", "STRIX_QSA_DECODE_SELECT",
                "STRIX_QSA_DECODE_FUSED_MAXIMA", "STRIX_QSA_DECODE_ATTN_V2", "STRIX_QSA_DECODE_VALUE_V2", "STRIX_MOE_R1_GLUE_FUSE"})
            expect(yarn_environment_refusal({{name, "0"}}, 2).empty(), "YaRN accepts the audited serial decode pool controls");
        for (const char* name : {"STRIX_QPF2K_COMPACT_IMAGES", "STRIX_CTX_EXACT_MEMORY", "STRIX_QSA_SCORE_RING", "STRIX_QSA_SCORE_RING_ROWS", "STRIX_TERMINAL_LOGITS_ARENA"})
            expect(yarn_environment_refusal({{name, "0"}}, 2).empty(), "YaRN accepts the compact-memory controls");
        for (const char* value : {"0", "1"})
            expect(yarn_environment_refusal({{"STRIX_QSA_CACHE_ONLY", value}}, 2).empty(), "YaRN accepts the cache-only QSA switch");
        expect(effective_prefill_argv(config.options)[2] == "--rope-yarn","YaRN receipt argv retains explicit opt-in");
    }
    {
        // Classification table: exact OFF switches and host diagnostics admit, arithmetic switches and unknowns refuse by name.
        for (const auto& rule : yarn_environment_rules) {
            expect(std::count_if(yarn_environment_rules.begin(), yarn_environment_rules.end(),
                [&](const auto& other) { return other.name == rule.name; }) == 1 &&
                std::none_of(qpf2k::defaults.begin(), qpf2k::defaults.end(),
                    [&](const auto& item) { return rule.name == item.name; }),
                "YaRN classification rows are unique and disjoint from the qpf2k items");
            expect((rule.kind == YarnEnvironmentClass::refused) == !rule.why.empty() &&
                ((rule.kind != YarnEnvironmentClass::off_switch && rule.kind != YarnEnvironmentClass::diagnostic) ||
                    !rule.values.empty()), "YaRN refused rows carry a reason; OFF and diagnostic rows bound their values");
        }
        for (const char* name : {"STRIX_QSA_UNION_MERGE", "STRIX_QSA_SCORE_ROWS64", "STRIX_PREFILL_MOE_MERGE_VEC",
                "STRIX_QPF2K_WMMA_PIN", "STRIX_QPF2K_HC_UPCOPY", "STRIX_PREFILL_ROUTER_SELECT", "STRIX_ROUTER_STAGE_ACT",
                "STRIX_QSA_WINDOW_ATTN_V2", "STRIX_PREFIX_CACHE", "STRIX_QPF2K_GDN_CHUNK16"}) {
            auto off = request; off.environment[name] = "0";
            const auto admitted = resolve(off, memory);
            expect(yarn_environment_refusal({{name, "0"}}, 2).empty() && admitted.admitted,
                "YaRN admits the exact OFF switch arm");
            expect(admitted.environment.at(name) == "0", "YaRN keeps the explicit OFF value");
            expect(yarn_environment_refusal({{name, "2"}}, 2) ==
                std::string("YaRN execution environment not supported: ") + name + " (accepts " +
                std::string(yarn_environment_rule(name)->values) + ')', "YaRN OFF switch values stay bounded");
        }
        for (const char* name : {"STRIX_QSA_UNION_MERGE", "STRIX_QSA_SCORE_ROWS64", "STRIX_PREFILL_MOE_MERGE_VEC"})
            expect(yarn_environment_refusal({{name, "1"}}, 2).empty(), "YaRN admits the explicit default arm of code-default items");
        for (const char* value : {"0", "1"}) {
            auto launch = request; launch.environment["STRIX_QPF2K_GDN_COOP_LAUNCH"] = value;
            const auto admitted = resolve(launch, memory);
            expect(yarn_environment_refusal({{"STRIX_QPF2K_GDN_COOP_LAUNCH", value}}, 2).empty() && admitted.admitted &&
                admitted.environment.at("STRIX_QPF2K_GDN_COOP_LAUNCH") == value,
                "YaRN admits 0 = plain default and 1 = cooperative OFF arm");
        }
        expect(yarn_environment_refusal({{"STRIX_QPF2K_GDN_COOP_LAUNCH", "2"}}, 2) ==
            "YaRN execution environment not supported: STRIX_QPF2K_GDN_COOP_LAUNCH (accepts 0|1)",
            "YaRN launch policy values stay bounded");
        // Chunked GDN prefill keeps main's admission: no rotary site, any value (its option identity governs arithmetic).
        expect(yarn_environment_rule("STRIX_QPF2K_GDN_FP16_CHUNK") &&
            yarn_environment_rule("STRIX_QPF2K_GDN_FP16_CHUNK")->kind == YarnEnvironmentClass::nonrotary &&
            yarn_environment_rule("STRIX_QPF2K_GDN_FP16_CHUNK")->values.empty(), "YaRN classifies the chunked GDN route as nonrotary");
        for (const char* value : {"", "0", "1", "0x1"})
            expect(yarn_environment_refusal({{"STRIX_QPF2K_GDN_FP16_CHUNK", value}}, 2).empty(),
                "YaRN admits every chunked GDN route value as before");
        for (const char* name : {"STRIX_PREFILL_TIMING", "STRIX_QSA_FAST_STATS", "STRIX_QPF2K_DENSE_TRACE",
                "STRIX_QPF2K_HC_VERBOSE", "STRIX_PREFLIGHT"})
            for (const char* value : {"0", "1"}) {
                auto diagnostic = request; diagnostic.environment[name] = value;
                expect(yarn_environment_refusal({{name, value}}, 2).empty() && resolve(diagnostic, memory).admitted,
                    "YaRN admits host-only diagnostics");
            }
        expect(!yarn_environment_refusal({{"STRIX_PREFILL_TIMING", "verbose"}}, 2).empty(),
            "YaRN diagnostic values stay bounded");
        auto arithmetic = request; arithmetic.environment["STRIX_HCW_OUTPUT"] = "2";
        expect(resolve(arithmetic, memory).error ==
            "YaRN execution environment not supported: STRIX_HCW_OUTPUT (changes HC output arithmetic)",
            "YaRN refuses arithmetic switches with their reason");
        for (const char* value : {"0", "1"})
            expect(!yarn_environment_refusal({{"STRIX_HCW_NORM16", value}}, 2).empty(),
                "YaRN refuses arithmetic switches at every value");
        auto unknown = request; unknown.environment["STRIX_MOCK_UNCLASSIFIED"] = "0";
        expect(resolve(unknown, memory).error == "YaRN execution environment not supported: STRIX_MOCK_UNCLASSIFIED",
            "YaRN keeps refusing unclassified variables by name, including OFF-looking values");
        auto mixed = request;
        mixed.environment["STRIX_PREFILL_TIMING"] = "1";
        mixed.environment["STRIX_QSA_UNION_MERGE"] = "0";
        mixed.environment["STRIX_HCW_OUTPUT"] = "2";
        mixed.environment["STRIX_QSA_FAST_SPLITS"] = "4";
        mixed.environment["STRIX_MOCK_UNCLASSIFIED"] = "1";
        expect(resolve(mixed, memory).error == "YaRN execution environment not supported: "
            "STRIX_HCW_OUTPUT (changes HC output arithmetic), STRIX_MOCK_UNCLASSIFIED, "
            "STRIX_QSA_FAST_SPLITS (changes the QSA attend split combine)",
            "YaRN refusal lists every offending variable and no admitted one");
    }
    // Explicit YMTP shares target organization, with independent serial and pool controls.
    auto mtp=request;
    auto& mo=mtp.options;auto& mm=mo.mtp;
    mm.enabled=true;mm.sidecar="sidecar.gguf";mm.sidecar_logical_name="fixture";
    mm.sidecar_sha256=mm.target_sha256=mm.target_content=mm.golden_sha256=mm.converter_sha256=mm.norm_converter_sha256=std::string(64,'1');
    mm.golden="golden.bin";mm.repository="fixture";mm.revision="revision";mm.source_path="sidecar.gguf";mm.converter_revision="revision";
    mm.conversion="effective-norm-embedding-first-v1";mm.normalized_rounding="fp32";mm.projection_rounding="bf16-rne";mm.sidecar_residency="register";
    mm.sidecar_bytes=mm.header_bytes=mm.sidecar_retained_bytes=mm.sidecar_setup_bytes=4096;
    mm.golden_max_bytes=mm.replay_host_bytes=mm.target_capture_bytes=4096;mm.runtime_bytes=UINT64_C(2)<<30U;
    mm.policy_identity=7;mm.candidate_depth=7;mm.window_rows=8;mm.window_rows_set={2,4,8};mm.window_prefix=true;mm.window_qsa_rows=true;
    mm.window_arena_bytes=mm.window_capture_bytes=mm.window_checkpoint_bytes=4096;mm.window_total_bytes=UINT64_C(2)<<30U;mm.window_history_intervals=4096;
    mm.draft_chain=true;mo.known_prefix_window=mo.known_prefix_full_rows=true;mo.known_prefix_prefill_rows=8192;
    const auto ym=resolve(mtp,memory);
    yarn_owner_inventory_regression(mtp,info,manifest);
    expect(ym.admitted && ym.admission.at("mtp_enabled")==1,"explicit YaRN MTP admitted after resolved target organization");
    {
        constexpr auto key = "STRIX_QPF2K_MOE_ROUTE_BF16";
        auto fast = mtp;
        fast.options.product_profile = parse({"decoder","MODEL","1","--profile","fast"}).product_profile;
        const auto got = resolve(fast,memory);
        const auto* row = got.options.product_profile.find("moe_route_bf16");
        expect(got.admitted && got.error.empty() && !got.options.prefill_moe_route_bf16 &&
            row && row->status == qwen::profile::Status::pending &&
            row->reason == "MTP, nested, serving, cache and session routes are unsupported" &&
            row->reason == got.options.prefill_moe_route_bf16_reason,
            "fast YaRN MTP softly declines profile storage with a pending reason");
        Environment exported;
        qwen::profile::apply(got.options.product_profile, [&](const char* name, const char* value) {
            exported[name] = value; return true;
        });
        expect(!got.environment.contains(key) && !exported.contains(key) &&
            !exported.contains("STRIX_QWEN_PROFILE_OWNED_moe_route_bf16"),
            "declined YaRN MTP profile storage has no BF16 export");
        expect(got.admission == ym.admission && got.host_compilation == ym.host_compilation &&
            numerical_profile_pair(got.options) == numerical_profile_pair(ym.options),
            "fast YaRN MTP decline preserves FP32 storage admission and identity");
        fast.options.prefill_moe_route_bf16_requested = true;
        const auto explicit_on = resolve(fast,memory);
        expect(!explicit_on.admitted && !explicit_on.options.prefill_moe_route_bf16 &&
            explicit_on.error == "MoE route BF16 candidate declined: MTP, nested, serving, cache and session routes are unsupported",
            "explicit YaRN MTP BF16 request keeps hard refusal under fast profile");
        fast.options.supplied_options.insert("--prefill-moe-route-bf16");
        expect(!resolve(fast,memory).error.empty(), "explicit YaRN MTP BF16 CLI keeps route refusal under fast profile");
        fast.options.supplied_options.erase("--prefill-moe-route-bf16");
        fast.options.prefill_moe_route_bf16_requested = false;
        fast.options.product_profile = parse({"decoder","MODEL","1","--profile","fast",
            "--prefill-moe-route-bf16","off"}).product_profile;
        const auto explicit_off = resolve(fast,memory);
        expect(explicit_off.admitted && explicit_off.error.empty() && !explicit_off.options.prefill_moe_route_bf16 &&
            !explicit_off.environment.contains(key) && explicit_off.admission == ym.admission &&
            numerical_profile_pair(explicit_off.options) == numerical_profile_pair(ym.options),
            "explicit YaRN MTP BF16 off preserves FP32 storage under fast profile");
    }
    {
        auto candidate=mtp; candidate.environment["STRIX_QPF2K_MOE_ROUTE_BF16"]="1";
        const auto declined=resolve(candidate,memory);
        expect(!declined.admitted && !declined.options.prefill_moe_route_bf16 &&
            declined.error=="MoE route BF16 candidate declined: MTP, nested, serving, cache and session routes are unsupported",
            "YaRN MTP environment candidate refuses without mixing storage");
        candidate.environment["STRIX_QPF2K_MOE_ROUTE_BF16"]="0";
        const auto unchanged=resolve(candidate,memory);
        expect(unchanged.admitted && unchanged.admission==ym.admission && unchanged.host_compilation==ym.host_compilation &&
            numerical_profile_pair(unchanged.options)==numerical_profile_pair(ym.options),"YaRN MTP candidate off unchanged");
        for (const bool on : {false,true}) {
            candidate.options.prefill_moe_route_bf16_requested=on;
            candidate.options.supplied_options.insert("--prefill-moe-route-bf16");
            expect(yarn_route_refusal(candidate.options)=="YaRN MTP explicit target override not supported: --prefill-moe-route-bf16",
                "YaRN MTP closed explicit organization unchanged for candidate flag");
        }
    }
    for(const auto c:{32768U,131072U,262144U,393216U,524288U}) {
        auto f2=mtp;f2.options.context_tokens=c;
        const auto unchanged=resolve(f2,memory);
        expect(unchanged.admitted && unchanged.admission.at("storage_ladder_step")==0U &&
            !unchanged.environment.contains("STRIX_PLE_ROLLING_POSITIONS") &&
            !unchanged.environment.contains("STRIX_PLE_ROLLING_CAP_BYTES") &&
            !unchanged.environment.contains("STRIX_QSA_SCORE_RING_ROWS"),
            "f2 through 524288 keeps step-zero storage defaults");
    }
    expect(!config.admission.contains("storage_ladder_step") &&
        yarn_mtp_storage_receipt(config).empty(),
        "serial YaRN retains original plan and receipt contract");
    for(const bool cli:{false,true}) {
        auto cert=mtp;cert.environment["STRIX_PREFIX_BATCH_CERT"]="1";
        if(cli) {cert.options.supplied_options.insert("--prefix-batch-cert");cert.options.prefix_batch_cert=2;}
        const auto admitted=resolve(cert,memory);
        expect(admitted.admitted && admitted.options.prefix_batch_cert==(cli?2U:1U) &&
            admitted.host_compilation==ym.host_compilation && numerical_profile_pair(admitted.options)==numerical_profile_pair(ym.options),
            "operational certification control preserves bounded YaRN MTP admission and numerical organization");
    }
    const auto serial=resolve(request,memory);
    expect(numerical_profile_pair(ym.options)==numerical_profile_pair(serial.options) &&
        ym.options.known_prefix_prefill_rows==serial.options.known_prefix_prefill_rows &&
        ym.options.known_prefix_tail_rows==serial.options.known_prefix_tail_rows,"serial comparator uses same target profile pair and schedule");
    auto disabled=mtp;disabled.environment["STRIX_QWEN_YARN_MTP"]="0";
    expect(resolve(disabled,memory).error.find("conflicts")!=std::string::npos,"explicit MTP/off conflict refuses before allocation");
    auto m1off=mtp;m1off.environment["STRIX_QSA_MTP_POOL"]="0";
    const auto historical=resolve(m1off,memory);
    expect(historical.admitted && historical.options.mtp.enabled && !qsa_mtp_pool_route(historical.options),"M1 OFF remains genuine admitted MTP");
    YarnMtpGeometry fixture;fixture.sidecar.file_size=mm.sidecar_bytes;fixture.sidecar.data_offset=1024;
    gguf::TensorInfo qt;qt.name="blk.48.attn_q.weight";qt.ggml_type=8;qt.byte_size=2745937920ULL;fixture.sidecar.tensors.push_back(qt);
    fixture.window_arenas={{2,1024},{4,2048},{8,4096}};fixture.target_tensor_count=1024;
    const auto offowners=yarn_mtp_owners(historical.options,model,fixture);
    expect(offowners.bytes.at("mtp_historical_window_pools")==3U*16U*(historical.options.context_tokens/4U)*128U*4U,
        "historical OFF counts every window slab");
    auto comparator=request;comparator.environment["STRIX_QWEN_YARN_MTP"]="0";
    expect(resolve(comparator,memory).admitted && !resolve(comparator,memory).options.mtp.enabled,"off is plain serial YaRN");
    expect(yarn_environment_refusal({{"STRIX_MTP_CONTROLLER","survival"}},2).size()>0,"unbounded lease controller refused");
    expect(yarn_environment_refusal({{"STRIX_QWEN_YARN_MTP","bad"}},2).size()>0,"unknown switch value refused");
    for(const auto c:{32768U,131072U,262144U,393216U,524288U,1048576U,32771U,3U}) {
        auto sized=ym.options;sized.context_tokens=c;sized.rotary=RotaryDescriptor::yarn(c<=524288?2:4);
        sized.qsa_mtp_pool_explicit_opt_in=true; // explicit capacity census, including qualified f4
        sized.mtp.window_total_bytes=UINT64_C(4)<<30U;
        auto sized_model=model;sized_model.qwen_rotary_table=mock_rotary(c<=524288?2:4);
        sized.mtp.runtime_bytes=mtp_payload::runtime(sized_model,c).retained;
        const auto extras=yarn_mtp_owners(sized,sized_model,fixture);
        expect(extras.bytes.at("mtp_predictor_kv_raw")==static_cast<std::uint64_t>(c)*2304U,"actual C predictor cache exactly once");
        expect(extras.bytes.at("mtp_predictor_workspace_alignment")==mtp_payload::runtime(sized_model,c).retained-static_cast<std::uint64_t>(c)*2304U,"actual runtime workspace without cap padding");
        expect(extras.bytes.at("mtp_window_arenas_captures_checkpoints")+qsa_mtp_pool_bytes(sized,model)<sized.mtp.window_total_bytes,
            "window payload excludes unused cap and charges shared M1 once");
        expect(extras.bytes.at("mtp_q4_shared_weights")==1453731840U,"manifest Q4 retained once");
        expect(extras.bytes.at("mtp_historical_window_pools")==0U && qsa_mtp_pool_route(sized),"table-backed M1 uses shared owner");
        auto insufficient=sized;insufficient.mtp.runtime_bytes--;
        expect(!yarn_mtp_owners(insufficient,sized_model,fixture).refusal.empty(),"runtime cap below actual payload refused");
    }
    {
        gguf::FileInfo q4;gguf::TensorInfo t;t.name="blk.48.attn_q.weight";t.ggml_type=8;t.byte_size=34U*128U;q4.tensors.push_back(t);
        expect(mtp_q4_manifest_bytes(q4)==18U*128U,"Q4 payload derives from actual Q8 manifest blocks");
        auto source=std::make_shared<int>(1);ModelContentIdentity content;content.bytes[0]=1;
        const MtpQ4SourceIdentity old{source,4,128,content};
        require_same_mtp_q4_source(old,old);
        for(unsigned fault=0;fault<3;++fault) {
            auto changed=old;if(!fault)++changed.generation;if(fault==1)changed.content.bytes[1]=1;
            if(fault==2)changed.owner=std::make_shared<int>(1);
            bool refused=false;try{require_same_mtp_q4_source(old,changed);}catch(const std::invalid_argument&){refused=true;}
            expect(refused,"stale Q4 address/new generation/content/owner refused");
        }
    }
    auto overflow = memory;
    overflow.estimate = [=](const Options& o) { auto value = estimate(o); value.model_backing = UINT64_MAX; return value; };
    expect(resolve(request,overflow).error.find("overflow") != std::string::npos,"YaRN checked owner sum");
    bool multiplied = false;
    try { (void)prefill_checked_mul(UINT64_MAX,2U); } catch (const std::overflow_error&) { multiplied = true; }
    expect(multiplied,"YaRN checked width product");
    auto boundary = request; boundary.options.tokens.resize(300000U,1U);
    expect(resolve(boundary,memory).error.find("prompt + generation") != std::string::npos,"YaRN positions are below capacity");
}
void mtp_reservation_arithmetic_tests() {
    constexpr std::uint64_t budget = 128849018880ULL, headroom = 4294967296ULL;
    constexpr std::uint64_t credit = 25193033728ULL, bound = 3607104512ULL;
    constexpr std::uint64_t metadata = 768ULL << 20U; // conservative synthetic live metadata
    constexpr std::uint64_t temporary = 128ULL << 20U;
    const auto limits = UmaReservationLimits{budget,budget};
    // The supplied setup credit already reserves C + D + margin and table edges.
    expect(28800138240ULL - credit == bound, "MTP source bound/decode quota counted once in net credit");
    MtpPrefillReservation m{3221225472ULL,134217728ULL,16ULL<<20U,64ULL<<20U,64ULL<<20U,
        2ULL*gib_bytes,64ULL<<20U,8ULL*gib_bytes,16ULL*gib_bytes,3,true};
    const auto extra = mtp_prefill_reservation(m);
    expect(extra.retained == 3U*gib_bytes + (128U<<20U) + (16U<<20U) + 2U*gib_bytes + (768U<<20U) &&
        extra.temporary == (128U<<20U), "prefix windows charge shared arena and all twelve capture caps, no checkpoint");
    // Remainder of the inventory is synthetic: no model or native receipt was supplied.
    // Keep the reported context, UMA, headroom, sidecar and runtime quantities exact.
    UmaOwnerInventory live;
    live.known_file_mapping_bytes = 96U*gib_bytes;
    live.reservation_upper_bound_bytes = 96U*gib_bytes + 33536ULL*(120U<<10U) + gib_bytes;
    const auto growth = 8U*gib_bytes + headroom;
    const auto normal = plan_known_prefill_reservation(live,growth,temporary,limits,credit,metadata,extra);
    expect(normal.retained_reserved_bytes == live.reservation_upper_bound_bytes + growth + metadata + extra.retained - credit &&
        normal.setup_peak_reserved_bytes == normal.retained_reserved_bytes + temporary + extra.temporary &&
        normal.reclaimable_file_mapping_credit_bytes == credit, "shared MTP retained/setup equations use verified credit exactly once");
    const auto plain = plan_known_prefill_reservation(live,growth,temporary,limits,credit,metadata);
    const auto original = plan_uma_reservation_growth(live,growth+metadata,temporary,limits,credit);
    expect(plain.retained_reserved_bytes == original.retained_reserved_bytes &&
        plain.setup_peak_reserved_bytes == original.setup_peak_reserved_bytes, "plain admission remains byte-for-byte arithmetic equivalent");
    // A loose aggregate window limit cannot become a phantom retained owner.
    // This reproduces the former MTP-only cap-summing refusal while OFF fits.
    m.window_total = 64U*gib_bytes;
    const auto capped = mtp_prefill_reservation(m);
    expect(capped.retained == extra.retained, "unused total-window cap does not enlarge retained ownership");
    bool old_refused = false;
    try { (void)plan_uma_reservation_growth(live,growth+metadata+4U*gib_bytes+m.runtime+m.target_capture+m.replay+m.window_total,
        temporary,limits,credit); } catch (const std::invalid_argument&) { old_refused = true; }
    expect(old_refused, "old rolling MTP cap sum rejects the regression envelope");
    (void)plan_known_prefill_reservation(live,growth,temporary,limits,credit,metadata,capped);
    (void)plan_known_prefill_reservation(live,growth,temporary,limits,0,0);
    // At 262144, identical synthetic state/profile/capacity and MTP owners:
    // OFF admits T2048 but refuses T8192; verified rolling admits T8192.
    live.reservation_upper_bound_bytes = 76U*gib_bytes + 262144ULL*(120U<<10U) + gib_bytes;
    const auto admit = [&](std::uint64_t rows, bool rolling) {
        try {
            (void)plan_known_prefill_reservation(live,rows*(gib_bytes/2048U)+headroom+gib_bytes/2U,
                temporary,limits,rolling?credit:0,rolling?metadata:0,extra);
            return true;
        } catch (const std::invalid_argument&) { return false; }
    };
    expect(admit(2048,false) && !admit(8192,false) && admit(8192,true), "long-context MTP rolling admits larger T with fixed owners");
    m.prefix = false;
    expect(mtp_prefill_reservation(m).retained == extra.retained + 8U*gib_bytes, "nonprefix rollback checkpoint remains charged");
    m.window_total = gib_bytes;
    expect(mtp_prefill_reservation(m).retained == 3U*gib_bytes + (128U<<20U) + (16U<<20U) + gib_bytes,
        "combined native verifier cap remains an independent upper bound");
    bool invalid_credit = false;
    try { (void)plan_known_prefill_reservation(live,0,0,limits,live.known_file_mapping_bytes+1U,0); }
    catch (const std::invalid_argument&) { invalid_credit = true; }
    expect(invalid_credit, "credit cannot exceed known mapped backing");
    bool overflow = false;
    try { (void)plan_known_prefill_reservation(live,UINT64_MAX,0,{},credit,1U); }
    catch (const std::overflow_error&) { overflow = true; }
    expect(overflow, "shared reservation additions reject overflow");
}
void t16k_admission_tests(const RequestedOptions& original, const ModelContract& model,
    const gguf::FileInfo& info, const ModelManifest& manifest, const PrefillCapabilities& capabilities) {
    const auto resolve = [&](const RequestedOptions& request, const PrefillMemory& memory) {
        return resolve_effective_config(request,model,info,manifest,validated_model_content_identity,capabilities,memory);
    };
    auto memory = ample_memory();
    memory.estimate = [](const Options& options) {
        PrefillRequirements r{70U*gib_bytes,8U*gib_bytes,gib_bytes,0U,0U,gib_bytes/8U,0U,9U*gib_bytes};
        r.workspace = static_cast<std::uint64_t>(options.known_prefix_prefill_rows) * 998400U + 65552U;
        r.metadata = 2U * (static_cast<std::uint64_t>(options.known_prefix_prefill_rows) * 34888U + 41808U + 4U);
        for (const auto rows : options.known_prefix_tail_rows) {
            r.tails += static_cast<std::uint64_t>(rows) * 998400U + 65552U;
            r.metadata += 2U * (static_cast<std::uint64_t>(rows) * 34888U + 41808U + 4U);
        }
        return r;
    };
    const auto estimate = memory.estimate;
    for (const auto count : {32767U,32768U,65535U,65536U,131072U,262143U}) {
        auto request = original;
        request.options.tokens.resize(count,1U); request.options.generate_tokens = 1U;
        request.environment["STRIX_QPF2K_T16K"] = "0";
        const auto legacy = resolve(request,memory);
        expect(legacy.admitted && legacy.options.known_prefix_prefill_rows == 8192U,
            "T16K off retains the historical T8192 ladder at every policy boundary");
        const auto small_pools = prefill_native_pools(legacy.options, model);
        expect(legacy.host_compilation.at("native_device_bytes") == small_pools.device &&
            legacy.options.known_prefill_host_headroom >= small_pools.host + small_pools.device &&
            small_pools.bytes.at("native_setup_growth_peak") == small_pools.bytes.at("native_hc_norm16"),
            "T8192 admission prices every native pool and the largest setup replacement overlap");
        if (count == 32767U) {
            auto explicit_group = request;
            explicit_group.options = legacy.options;
            explicit_group.options.known_prefill_host_headroom = 1U;
            explicit_group.environment["STRIX_PROMPT_BATCH_ROWS"] = "16384";
            const auto declined = resolve(explicit_group,memory);
            expect(declined.error.find("native pools exceed") != std::string::npos &&
                declined.host_compilation.at("native_moe_a16_da16") == UINT64_C(16384) * 2U * (2560U + 10U * 640U) &&
                declined.host_compilation.at("native_device_bytes") > small_pools.device &&
                declined.host_compilation.at("native_ple_staging") > small_pools.bytes.at("native_ple_staging"),
                "explicit MTP group capacity and retained main owner are priced before native setup");
        }
        request.environment.erase("STRIX_QPF2K_T16K");
        const auto automatic = resolve(request,memory);
        const bool wide = count >= 32768U;
        expect(automatic.admitted && automatic.options.known_prefix_prefill_rows == (wide ? 16384U : 8192U) &&
            automatic.t16k_policy_source == (wide ? std::optional{OptionSource::automatic} : std::nullopt) &&
            automatic.reason == (wide ? "automatic-t16k" : "qualified-prompt"),
            "unset T16K tries T16384 only at and above 32768 prompt tokens");
        expect(!automatic.environment.contains("STRIX_QPF2K_T16K"), "automatic policy does not invent an environment override");
        if (!wide)
            expect(effective_config_json(automatic) == effective_config_json(legacy) &&
                effective_config_notice(automatic) == effective_config_notice(legacy),
                "below-threshold default preserves every legacy receipt byte");
        else
            expect(effective_config_json(automatic).find("\"t16k_policy\":{\"source\":\"automatic\",\"threshold_tokens\":32768}") != std::string::npos &&
                effective_config_notice(automatic).find("automatic T16384 ") != std::string::npos &&
                effective_config_notice(automatic).find("T16384 policy=automatic threshold=32768") != std::string::npos,
                "automatic T16384 notice and JSON report provenance and threshold");
        request.environment["STRIX_QPF2K_T16K"] = "1";
        const auto config = resolve(request,memory);
        expect(config.admitted && config.options.known_prefix_prefill_rows == 16384U && config.reason == "opt-in-t16k" &&
            config.ple_reclaim_credit == 0U && config.options.known_prefill_uma_budget == 120U*gib_bytes,
            "T16K override admits every large planner class with capacity and no invented reclaim credit");
        expect(config.t16k_policy_source == OptionSource::explicit_value &&
            effective_config_json(config).find("\"t16k_policy\":{\"source\":\"explicit\",\"threshold_tokens\":32768}") != std::string::npos &&
            effective_config_notice(config).find("automatic T16384 ") != std::string::npos &&
            effective_config_notice(config).find("T16384 policy=explicit threshold=32768") != std::string::npos,
            "override T16384 notice and JSON report explicit provenance and the automatic threshold");
        if (wide) {
            auto expected = config;
            expected.environment.erase("STRIX_QPF2K_T16K");
            expected.reason = "automatic-t16k";
            expected.t16k_policy_source = OptionSource::automatic;
            expect(effective_config_json(automatic) == effective_config_json(expected),
                "automatic and explicit T16384 share all execution, budget and compiler decisions");
        }
        const auto required = estimate(config.options);
        const auto pools = prefill_native_pools(config.options,model);
        expect(config.options.known_prefill_workspace_budget == required.workspace + required.tails &&
            config.options.known_prefill_metadata_budget == required.metadata && required.metadata > gib_bytes &&
            config.options.known_prefill_host_headroom == pools.host + pools.device &&
            config.options.known_prefix_tail_rows == legacy.options.known_prefix_tail_rows,
            "T16K budgets cover both compiled metadata copies, retained tails and explicit native pools");
        expect(pools.bytes.at("native_qsa_maxima_peak") == 3U * UINT64_C(16384) *
            ((config.options.context_tokens / 4U + 3U) / 4U) * 4U &&
            config.host_compilation.at("native_device_bytes") == pools.device,
            "T16K receipt prices two QSA ring slots plus conservative maxima headroom");
        // One byte below device admission: model backing credit cannot pay GTT.
        auto tight = memory;
        tight.gtt_limit = required.device_base + required.workspace + required.tails + pools.device + pools.mapped - 1U;
        const auto fallback = resolve(request,tight);
        auto expected_fallback = legacy;
        expected_fallback.environment["STRIX_QPF2K_T16K"] = "1";
        expected_fallback.t16k_policy_source = OptionSource::explicit_value;
        expected_fallback.tile_fallback = "T16384:insufficient-memory";
        expect(fallback.admitted && fallback.options.known_prefix_prefill_rows == 8192U &&
            fallback.tile_fallback == "T16384:insufficient-memory" &&
            effective_config_json(fallback) == effective_config_json(expected_fallback) &&
            effective_prefill_argv(fallback.options) == effective_prefill_argv(legacy.options),
            "T16K GTT fallback preserves every T8192 argv/budget/tail/environment decision and reports reason");
        if (wide) {
            request.environment.erase("STRIX_QPF2K_T16K");
            const auto auto_fallback = resolve(request,tight);
            expected_fallback.environment.erase("STRIX_QPF2K_T16K");
            expected_fallback.t16k_policy_source = OptionSource::automatic;
            expect(effective_config_json(auto_fallback) == effective_config_json(expected_fallback) &&
                effective_prefill_argv(auto_fallback.options) == effective_prefill_argv(legacy.options),
                "automatic memory decline restores the exact historical T8192 configuration with policy and reason");
            request.environment["STRIX_QPF2K_T16K"] = "1";
        }
        auto uma_tight = memory;
        uma_tight.estimate = [&](const Options& options) {
            auto r = estimate(options);
            // Charge exactly one byte beyond 120 GiB at T16384, without L1 support.
            r.model_backing = 120U*gib_bytes + 1U - required.state_d1 - required.serial_scratch -
                required.workspace - required.tails - required.metadata - required.runtime_payload - pools.host - pools.device;
            return r;
        };
        const auto uma_fallback = resolve(request,uma_tight);
        expect(uma_fallback.admitted && uma_fallback.options.known_prefix_prefill_rows == 8192U &&
            uma_fallback.tile_fallback == "T16384:insufficient-memory",
            "T16K native pools participate in UMA admission exactly once; no reference credit is assumed");
        if (wide) {
            request.environment.erase("STRIX_QPF2K_T16K");
            auto expected = uma_fallback;
            expected.environment.erase("STRIX_QPF2K_T16K");
            expected.t16k_policy_source = OptionSource::automatic;
            expect(effective_config_json(resolve(request,uma_tight)) == effective_config_json(expected),
                "automatic and explicit T16384 share UMA fallback and restore T8192 budgets");
        }
        for (const auto* value : {"", "0", "00", "0disabled", "true", "01", "2", "1 ", "-1"}) {
            request.environment["STRIX_QPF2K_T16K"] = value;
            auto off = resolve(request,memory);
            expect(!off.environment.contains("STRIX_QPF2K_T16K"), "ineffective T16K values are absent from the receipt");
            expect(effective_config_json(off) == effective_config_json(legacy), "only the exact T16K value 1 opts in");
            expect(effective_config_notice(off) == effective_config_notice(legacy), "ineffective T16K controls do not change notices");
        }
    }
    auto request = original; request.environment["STRIX_QPF2K_T16K"] = "1";
    auto tight = memory;
    std::vector<std::uint32_t> attempts;
    tight.estimate = [&](const Options& options) {
        attempts.push_back(options.known_prefix_prefill_rows);
        if (options.known_prefix_prefill_rows > 2048U) throw KnownPrefillBudgetExceeded("fixture budget");
        return estimate(options);
    };
    const auto fallback = resolve(request,tight);
    expect(fallback.admitted && fallback.options.known_prefix_prefill_rows == 2048U &&
        attempts == std::vector<std::uint32_t>{16384U,16384U,8192U,8192U,4096U,4096U,2048U} &&
        fallback.tile_fallback == "T16384:insufficient-memory,T8192:insufficient-memory,T4096:insufficient-memory" &&
        fallback.options.known_prefix_tail_rows == std::vector<std::uint32_t>{256U,16U},
        "T16K retry chain preserves FP16/dense retries and historical T2048 tails");
    // Both policy sources retain arithmetic retries, the entire ladder and all decline reasons.
    for (const bool explicit_t16k : {false,true}) for (const auto chosen : {8192U,4096U,2048U,0U}) {
        auto chain = original;
        chain.options.tokens.resize(65536U,1U);
        chain.options.generate_tokens = 1U;
        if (explicit_t16k) chain.environment["STRIX_QPF2K_T16K"] = "1";
        attempts.clear();
        tight.estimate = [&](const Options& options) {
            attempts.push_back(options.known_prefix_prefill_rows);
            if (options.known_prefix_prefill_rows > chosen) throw KnownPrefillBudgetExceeded("fixture budget");
            return estimate(options);
        };
        const auto config = resolve(chain,tight);
        std::vector<std::uint32_t> expected_attempts;
        std::string expected_reasons;
        for (const auto rows : {16384U,8192U,4096U,2048U}) {
            expected_attempts.push_back(rows);
            if (rows == chosen) break;
            expected_attempts.push_back(rows); // FP16 then dense-A8 compiler retry.
            if (!expected_reasons.empty()) expected_reasons += ',';
            expected_reasons += "T" + std::to_string(rows) + ":insufficient-memory";
        }
        expect(config.admitted == (chosen != 0U) && attempts == expected_attempts && config.tile_fallback == expected_reasons &&
            (!chosen || config.options.known_prefix_prefill_rows == chosen) &&
            config.t16k_policy_source == (explicit_t16k ? OptionSource::explicit_value : OptionSource::automatic),
            "automatic/explicit T16384 keep the complete fallback ladder and its memory reasons");
        if (chosen == 2048U)
            expect(config.options.known_prefix_tail_rows == std::vector<std::uint32_t>{256U,16U},
                "automatic/explicit T16384 retain historical T2048 tails");
    }
    const auto captured = requested_prefill_environment([](const char* name) {
        return std::string_view(name) == "STRIX_QPF2K_T16K" ? "1" : nullptr;
    });
    expect(captured.at("STRIX_QPF2K_T16K") == "1", "runtime environment capture sees the opt-in without injecting it");
    auto explicit_args = effective_prefill_argv(resolve(request,memory).options);
    explicit_args.insert(explicit_args.begin(), {"decoder","MODEL","1"});
    const auto explicit_route = parse(explicit_args);
    expect(explicit_route.known_prefix_prefill_explicit && explicit_route.known_prefix_prefill_rows == 16384U &&
        explicit_route.known_prefix_window && explicit_route.known_prefix_full_rows,
        "explicit T16384 window/full-rows argv round-trips through CLI validation");
    for (const auto count : {16U,64U,256U,1024U,2048U,4096U,8192U,16383U,16384U}) {
        request.options.tokens.resize(count,1U);
        auto absent = request; absent.environment.clear();
        auto expected = resolve(absent,memory);
        const auto on = resolve(request,memory);
        if (count == 16384U)
            expect(on.admitted && on.options.known_prefix_prefill_rows == 16384U &&
                on.t16k_policy_source == OptionSource::explicit_value, "override admits T16384 at its minimum prompt length");
        else {
            expected.environment["STRIX_QPF2K_T16K"] = "1";
            expect(effective_config_json(on) == effective_config_json(expected), "T16K opt-in preserves smaller routes and records the request");
        }
    }
    // Global/owner exclusions are evaluated before any T16384 setup or memory estimate.
    tight.estimate = [&](const Options& options) {
        attempts.push_back(options.known_prefix_prefill_rows);
        return estimate(options);
    };
    for (const bool explicit_t16k : {false,true}) for (unsigned excluded = 0; excluded < 6U; ++excluded) {
        auto blocked = original;
        blocked.options.tokens.resize(131072U,1U); blocked.options.generate_tokens = 1U;
        if (explicit_t16k) blocked.environment["STRIX_QPF2K_T16K"] = "1";
        auto layout = manifest;
        if (excluded == 0U) blocked.options.baseline = true;
        if (excluded == 1U) blocked.environment["STRIX_BASELINE"] = "1";
        if (excluded == 2U) blocked.environment["STRIX_QPF2K_DEFAULTS"] = "0";
        if (excluded == 3U) blocked.environment["STRIX_QPF2K_DEFAULTS"] = "00";
        if (excluded == 4U) blocked.environment["STRIX_QPF2K_DEFAULTS"] = "0disabled";
        if (excluded == 5U) {
            ManifestTensor gate;
            gate.name = "blk.0.ffn_gate_exps.weight"; gate.ggml_type = gsq_a8::type_q2_0;
            layout.tensors.push_back(gate);
            layout.layout_identity = model_manifest_layout_identity(layout);
        }
        attempts.clear();
        const auto config = resolve_effective_config(blocked,model,info,layout,validated_model_content_identity,capabilities,tight);
        expect(!config.admitted && !config.t16k_policy_source && attempts.empty() && config.tile_fallback.empty() &&
            config.reason == (excluded < 2U ? "baseline" : excluded < 5U ? "feature-disabled" : "checkpoint-content-not-qualified"),
            "baseline, defaults-off and GSQ exclude automatic and override T16384 before setup");
    }
}
void long_context_admission_tests(const RequestedOptions& original, const ModelContract& model,
    const gguf::FileInfo& info, ModelManifest manifest, const PrefillCapabilities& capabilities) {
    const auto contract = build_ple_hash_contract(model);
    ManifestTensor table;
    table.name = "ple.ngram_table.weight";
    table.role = TensorRole::ngram_table;
    table.residency = ResidencyPolicy::file_backed_candidate;
    table.ggml_type = 20U;
    table.dimensions = {160U,contract.minimum_table_rows};
    table.absolute_offset = 4U*gib_bytes + 32U;
    table.byte_size = table.dimensions[1] * 90U;
    manifest.tensors.push_back(table);
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    auto observed = ample_memory();
    observed.available = UINT64_C(127800000000);
    observed.gtt_limit = 128U*gib_bytes;
    observed.gtt_used = 2U*gib_bytes;
    observed.ple_reclaim_page = 4096U;
    observed.estimate = [](const Options& options) {
        auto required = PrefillRequirements{};
        required.model_backing = 96U*gib_bytes;
        required.state_d1 = options.context_tokens * 120U*1024U;
        required.serial_scratch = gib_bytes;
        required.workspace = options.known_prefix_prefill_rows * (gib_bytes / 2048U);
        required.metadata = gib_bytes / 2U;
        required.runtime_payload = gib_bytes / 8U;
        required.device_base = 40U*gib_bytes;
        required.tails = static_cast<std::uint64_t>(options.known_prefix_tail_rows.size()) * (gib_bytes / 16U);
        return required;
    };
    const auto resolve = [&](const RequestedOptions& request, const PrefillMemory& memory) {
        return resolve_w7_case(request,model,info,manifest,validated_model_content_identity,capabilities,memory);
    };
    {
        ple_reclaim::Config config;
        config.page = 4096U;
        config.table_offset = table.absolute_offset;
        config.table_rows = table.dimensions[1];
        config.stored_row_bytes = 90U;
        config.mapping_bytes = table.absolute_offset + table.byte_size;
        config.decode_allowance_bytes = ple_reclaim::decode_allowance(1U,1U,contract.head_count,config.page);
        config.reclaim = ple_reclaim::reclaim_intervals(config.table_offset,table.byte_size,config.page,
            {{config.table_offset + 4096U,config.table_offset + 8192U}});
        const auto plan = ple_reclaim::plan_prompt(config,contract,original.options.tokens);
        std::set<std::uint64_t> touched;
        std::array<std::uint32_t,PleHashContract::maximum_ngram - 1U> previous{};
        const auto count = contract.ngram_size - 1U;
        std::fill_n(previous.begin(),count,contract.eos_token_id);
        std::size_t row_index = 0;
        for (const auto token : original.options.tokens) {
            const auto hash = hash_ple_token(contract,token,{previous.data(),count});
            for (std::uint32_t h = 0; h < hash.row_count; ++h) {
                const auto offset = config.table_offset + hash.row_ids[h] * config.stored_row_bytes;
                expect(plan.rows[row_index++] == hash.row_ids[h], "ple_reclaim_prompt_rows: shared planner retains runtime hash rows");
                for (auto page = offset / config.page; page <= (offset + config.stored_row_bytes - 1U) / config.page; ++page)
                    for (const auto& interval : config.reclaim)
                        if (page * config.page >= interval.begin && page * config.page < interval.end) touched.insert(page);
            }
            std::copy_n(hash.next_context.begin(),count,previous.begin());
        }
        std::uint64_t reclaimable = 0;
        for (const auto& interval : config.reclaim) reclaimable += interval.end - interval.begin;
        const auto touched_bytes = touched.size() * config.page;
        const auto admitted_bytes = touched_bytes + (256U << 20U) + (64U << 20U) + touched_bytes / 64U;
        expect(plan.stats.reclaimable_bytes == reclaimable && plan.stats.touched_bytes == touched_bytes &&
            plan.stats.admitted_bytes == admitted_bytes && plan.stats.credit_bytes == reclaimable - admitted_bytes,
            "ple_reclaim_exact_credit: unregistered interior minus unique prompt pages, decode allowance and margin");
        expect(ple_reclaim::decode_allowance(100000U,1U,contract.head_count,config.page) ==
            UINT64_C(100001) * contract.head_count * 2U * config.page,
            "ple_reclaim_decode_bound: generation allowance grows beyond its 256 MiB floor");
        bool overflow = false;
        try { (void)ple_reclaim::decode_allowance(UINT64_MAX,1U,contract.head_count,config.page); }
        catch (const std::overflow_error&) { overflow = true; }
        expect(overflow, "ple_reclaim_decode_overflow: allowance arithmetic is checked");
    }
    auto request = original;
    request.options.generate_tokens = 1U;
    for (const auto n : {131072U,196608U,262143U}) {
        request.options.tokens.resize(n,1U);
        // Repeated, changing IDs exercise the full hash chain without loading a model or prompt file.
        for (std::size_t i = 0; i < request.options.tokens.size(); ++i)
            request.options.tokens[i] = static_cast<std::uint32_t>(i % 1024U);
        const auto config = resolve(request,observed);
        const auto expected_context = std::min(model.max_position_embeddings, static_cast<std::uint64_t>(n) + 512U);
        expect(config.admitted && config.options.known_prefix_prefill_rows == 8192U &&
            config.options.context_tokens == expected_context && config.ple_reclaim_credit > 23U*gib_bytes && config.ple_rolling_cap == 3U*gib_bytes,
            "long_context_l1_T8192: 128K/192K/262143 admit with observed memory and capped L1 credit");
        const auto text = effective_config_json(config);
        expect(text.find("\"ple_reclaim_credit_bytes\":" + std::to_string(config.ple_reclaim_credit)) != std::string::npos,
            "long_context_l1_receipt: JSON reports the predicted reclaim credit");
        auto no_l1 = observed; no_l1.ple_reclaim_page.reset();
        const auto retained = resolve(request,no_l1);
        expect(!retained.admitted || retained.options.known_prefix_prefill_rows < 8192U,
            "long_context_l1_required_for_T8192: unsupported reclaim earns no credit");
    }
    // Tight GTT exercises the independent device cap: L1 cannot credit registered/device aliases.
    const auto estimate = observed.estimate;
    for (const auto chosen : {4096U,2048U,0U}) {
        auto tight = observed;
        auto priced = resolve(request,observed).options;
        priced.known_prefix_prefill_rows = chosen ? chosen : 2048U;
        priced.known_prefix_tail_rows.clear();
        const auto pools = prefill_native_pools(priced,model);
        tight.gtt_limit = 2U*gib_bytes + 40U*gib_bytes + pools.device + pools.mapped +
            (chosen == 4096U ? 3U*gib_bytes : chosen == 2048U ? 2U*gib_bytes : gib_bytes / 2U);
        std::vector<std::uint32_t> attempts;
        tight.estimate = [&](const Options& candidate) {
            attempts.push_back(candidate.known_prefix_prefill_rows);
            return estimate(candidate);
        };
        const auto config = resolve(request,tight);
        const std::vector<std::uint32_t> expected = chosen == 4096U ? std::vector<std::uint32_t>{8192U,4096U} :
            std::vector<std::uint32_t>{8192U,4096U,2048U};
        expect(attempts == expected && config.admitted == (chosen != 0U) &&
            (!chosen || config.options.known_prefix_prefill_rows == chosen),
            "long_context_tile_fallback_order: T8192 then T4096 then T2048 before decline");
        if (chosen == 2048U) {
            const auto args = effective_prefill_argv(config.options);
            const auto tail = std::find(args.begin(),args.end(),"--known-prefix-tail-rows");
            expect(config.options.known_prefix_tail_rows == std::vector<std::uint32_t>{256U,16U} &&
                config.options.known_prefix_ladder_tail_rows == config.options.known_prefix_tail_rows &&
                config.sources.at("--known-prefix-tail-rows") == OptionSource::automatic &&
                tail != args.end() && std::next(tail) != args.end() && *std::next(tail) == "256,16",
                "long_context_T2048_tails: active/configured tails and replay argv use 256,16");
        }
        const auto notice = effective_config_notice(config);
        expect(chosen ? notice.find("T" + std::to_string(chosen)) != std::string::npos &&
            notice.find("T8192:insufficient-memory") != std::string::npos &&
            effective_config_json(config).find("T8192:insufficient-memory") != std::string::npos :
            notice.find("using serial prefill") == std::string::npos && !config.error.empty(),
            "long_context_fallback_receipt: selection and memory reason reported; oversized serial is refused");
    }
    for (const auto chosen : {4096U,2048U}) {
        auto tight = observed;
        tight.available = 100U*gib_bytes;
        const auto total = *tight.available;
        const auto credit = prefill_reclaim_credit(resolve(request,observed).options,model,manifest,{},tight.ple_reclaim_page);
        tight.estimate = [&](const Options& candidate) {
            auto required = estimate(candidate);
            const auto other = required.state_d1 + required.serial_scratch + required.metadata + required.runtime_payload +
                required.tails + candidate.known_prefill_host_headroom;
            const auto allowed_workspace = chosen == 4096U ? 3U*gib_bytes : gib_bytes + gib_bytes / 2U;
            required.model_backing = total + credit - other - allowed_workspace;
            return required;
        };
        const auto config = resolve(request,tight);
        expect(config.admitted && config.options.known_prefix_prefill_rows == chosen,
            "long_context_uma_fallback: retained host backing includes headroom and runtime payload, credit applied once");
    }
    auto compiler_failure = observed;
    {
        auto edge=observed;
        edge.available=100U*gib_bytes;
        const auto selected=resolve(request,observed).options;
        const auto owner=prefill_rolling_owner_bytes(selected,model,manifest,{},edge.ple_reclaim_page);
        const auto credit=prefill_reclaim_credit(selected,model,manifest,{},edge.ple_reclaim_page);
        expect(owner>400U*(1U<<20U),"rolling owner charges reader metadata beyond the cap/decode credit withholding");
        edge.estimate=[&](const Options& candidate) {
            auto required=estimate(candidate);
            const auto other=required.state_d1+required.serial_scratch+required.workspace+required.metadata+
                required.runtime_payload+required.tails+candidate.known_prefill_host_headroom;
            // T8192 fits by one byte without owner metadata; the full equation
            // must reject it and select T4096, which has ample workspace slack.
            const auto primary_workspace=estimate(selected).workspace;
            required.model_backing=*edge.available+credit-other-1U-
                (primary_workspace-required.workspace);
            return required;
        };
        const auto config=resolve(request,edge);
        expect(config.admitted && config.options.known_prefix_prefill_rows==4096U &&
            config.tile_fallback=="T8192:insufficient-memory","rolling owner boundary selects the runtime-admissible smaller tile");
    }
    for (const auto* message : {"known prefill workspace exceeds its explicit arena budget",
            "known prefill host metadata payload exceeds its budget: T = 8192 needs 1073741825 bytes"}) {
        compiler_failure.estimate = [&](const Options& candidate) {
            if (candidate.known_prefix_prefill_rows == 8192U) throw KnownPrefillBudgetExceeded(message);
            return estimate(candidate);
        };
        const auto compiled = resolve(request,compiler_failure);
        expect(compiled.admitted && compiled.options.known_prefix_prefill_rows == 4096U &&
            compiled.tile_fallback == "T8192:insufficient-memory" && compiled.arithmetic_fallback.empty(),
            "long_context_budget_fallback: workspace and metadata budget rejections report insufficient memory");
    }
    compiler_failure.estimate = [&](const Options& candidate) {
        if (candidate.known_prefix_prefill_rows == 8192U) throw std::invalid_argument("plan budget exceeded");
        return estimate(candidate);
    };
    const auto compiled = resolve(request,compiler_failure);
    expect(compiled.admitted && compiled.options.known_prefix_prefill_rows == 4096U &&
        compiled.tile_fallback == "T8192:candidate-not-admitted" && compiled.arithmetic_fallback.empty(),
        "long_context_compiler_fallback: other validation failures retain candidate-not-admitted");
    for (const auto* value : {"0","00"}) {
        auto disabled = request;
        disabled.environment["STRIX_QPF2K_DEFAULTS"] = value;
        const auto config = resolve(disabled,observed);
        expect(!config.admitted && config.reason == "feature-disabled" && config.ple_reclaim_credit == 0U,
            "long_context_defaults_off: leading zero disables automatic admission and reclaim credit");
    }
    // These pure route checks share the runtime gate, including graph/copy/MTP/IO exclusions.
    const auto admitted = resolve(request,observed);
    expect(admitted.admitted, "long_context_reclaim_route_fixture");
    {
        auto wide = request; wide.environment["STRIX_QPF2K_T16K"] = "1";
        const auto resolve_wide = [&](const PrefillMemory& memory) {
            return resolve_effective_config(wide,model,info,manifest,validated_model_content_identity,capabilities,memory);
        };
        auto wide_observed = observed;
        wide_observed.estimate = [&](const Options& candidate) {
            auto required = observed.estimate(candidate);
            // Isolate the wide-pool boundary while retaining upstream rolling metadata.
            required.model_backing -= 8U*gib_bytes;
            return required;
        };
        const auto config = resolve_wide(wide_observed);
        expect(config.admitted && config.options.known_prefix_prefill_rows == 16384U &&
            config.ple_reclaim_credit == admitted.ple_reclaim_credit,
            "T16K uses the same prompt-specific reclaim proof as T8192");
        auto tight = wide_observed; tight.available = 120U*gib_bytes;
        const auto r = wide_observed.estimate(config.options);
        const auto pools = prefill_native_pools(config.options,model);
        const auto other = r.state_d1 + r.serial_scratch + r.workspace + r.tails + r.metadata +
            r.runtime_payload + pools.host + pools.device +
            prefill_rolling_owner_bytes(config.options,model,manifest,wide.environment,tight.ple_reclaim_page);
        tight.estimate = [&](const Options& options) {
            auto required = observed.estimate(options);
            required.model_backing = 120U*gib_bytes + config.ple_reclaim_credit + 1U - other;
            return required;
        };
        const auto fallback = resolve_wide(tight);
        expect(fallback.admitted && fallback.options.known_prefix_prefill_rows == 8192U &&
            fallback.tile_fallback == "T16384:insufficient-memory",
            "T16K charges native pools once and subtracts verified-eligible model backing credit only once");
    }
    if (admitted.admitted) {
        const auto credit = [&](const Options& options, const Environment& environment, const ModelManifest& layout) {
            return prefill_reclaim_credit(options,model,layout,environment,observed.ple_reclaim_page);
        };
        expect(admitted.ple_rolling_h == 1024U && admitted.ple_rolling_bound > 600000000U &&
            effective_config_json(admitted).find("\"ple_verified_credit_bytes\":0") != std::string::npos,
            "rolling inspection exposes H/bound and never claims setup verification");
        {
            const auto json = effective_config_json(admitted);
            const auto first = json.find("\"ple_rolling_cap_bytes\":");
            expect(first != std::string::npos && json.find("\"ple_rolling_cap_bytes\":", first + 1U) == std::string::npos,
                "rolling inspection emits ple_rolling_cap_bytes once");
        }
        auto off_request = request;
        off_request.environment["STRIX_PLE_ROLLING"] = "0";
        const auto off = resolve(off_request,observed);
        expect(off.admitted && !off.ple_rolling_h && credit(off.options,{{"STRIX_PLE_ROLLING","0"}},manifest) ==
            credit(off.options,{{"STRIX_CTX_EXACT_MEMORY","0"}},manifest), "both memory switches restore prompt-wide credit");
        auto historical_print = off;
        historical_print.environment.erase("STRIX_PLE_ROLLING");
        expect(effective_config_json(off) == effective_config_json(historical_print),
            "rolling off preserves historical printed config byte for byte");
        for (unsigned route = 0; route < 4; ++route) {
            auto reused = admitted.options;
            if (route == 0) reused.prefix_cache_host_bytes = gib_bytes;
            if (route == 1) reused.prefix_cache_dir = "cache";
            if (route == 2) reused.prefix_cache_then_prompts.push_back("suffix.ids");
            if (route == 3) reused.rtx_state_import = true;
            std::uint64_t reuse_h = 999;
            const auto reused_credit = prefill_reclaim_credit(reused,model,manifest,{},observed.ple_reclaim_page,&reuse_h);
            expect(!reuse_h && reused_credit == credit(reused,{{"STRIX_PLE_ROLLING","0"}},manifest),
                "prefix restore and follow-up prompts retain public prompt-wide PLE credit");
        }
        auto fixed = admitted.options;
        fixed.mtp.enabled = true; fixed.session_mtp_fixed = true;
        expect(!credit(fixed,{},manifest), "fixed MTP sessions never inherit a rolling turn lease");
        std::uint64_t h = 0, bound = 0;
        (void)prefill_reclaim_credit(admitted.options,model,manifest,{{"STRIX_PLE_ROLLING_POSITIONS","256"}},
            observed.ple_reclaim_page,&h,&bound);
        expect(h == 256U && bound == admitted.ple_rolling_bound, "H tuning leaves the hard cap admission unchanged");

        std::uint64_t cap = 0;
        (void)prefill_reclaim_credit(admitted.options,model,manifest,{{"STRIX_PLE_ROLLING_CAP_BYTES","1073741824"}},
            observed.ple_reclaim_page,&h,&bound,&cap);
        expect(cap == (1ULL<<30U) && bound < admitted.ple_rolling_bound &&
            effective_config_json(admitted).find("\"ple_rolling_cap_bytes\":3221225472") != std::string::npos,
            "cap tuning and inspection use the shared budget");
        off_request.environment["STRIX_PLE_ROLLING_CAP_BYTES"] = "invalid-but-disabled";
        expect(effective_config_json(resolve(off_request,observed)) == effective_config_json(historical_print),
            "off route ignores cap tuning and preserves printed bytes");
        auto master_request = request;
        master_request.environment["STRIX_CTX_EXACT_MEMORY"] = "0";
        master_request.environment["STRIX_PLE_ROLLING_CAP_BYTES"] = "invalid-but-disabled";
        master_request.environment["STRIX_PLE_ROLLING_POSITIONS"] = "invalid-but-disabled";
        const auto master_off = resolve(master_request,observed);
        auto master_historical = master_off;
        master_historical.environment.erase("STRIX_PLE_ROLLING_CAP_BYTES");
        master_historical.environment.erase("STRIX_PLE_ROLLING_POSITIONS");
        expect(!master_off.ple_rolling_h && effective_config_json(master_off)==effective_config_json(master_historical),
            "master off preserves printed bytes and ignores rolling tuning");
        auto mtp = admitted.options;
        mtp.mtp.enabled = true;
        std::uint64_t mtp_h=0, mtp_bound=0;
        const auto mtp_credit=prefill_reclaim_credit(mtp,model,manifest,{},observed.ple_reclaim_page,&mtp_h,&mtp_bound);
        expect(mtp_credit>0 && mtp_h==1024 && mtp_bound>600000000U, "fixed-policy tiled MTP has predicted credit only");
        expect(prefill_reclaim_credit(mtp,model,manifest,{{"STRIX_PLE_ROLLING_CAP_BYTES","ignored"}},
            observed.ple_reclaim_page)==0, "invalid explicit owned cap refuses MTP credit");
        expect(prefill_reclaim_credit(mtp,model,manifest,{{"STRIX_PLE_ROLLING_CAP_BYTES","1073741824"}},
            observed.ple_reclaim_page)>mtp_credit, "MTP allocation and prediction honor the same owned cap");
        mtp.generate_tokens=mtp.context_tokens-mtp.tokens.size();
        expect(credit(mtp,{},manifest)==mtp_credit,"MTP bounded decode allowance independent of G");
        for (const auto& off_env : {Environment{{"STRIX_PLE_ROLLING","0"}}, Environment{{"STRIX_CTX_EXACT_MEMORY","0"}}}) {
            h=123; bound=123;
            expect(prefill_reclaim_credit(mtp,model,manifest,off_env,observed.ple_reclaim_page,&h,&bound)==0 && !h && !bound,
                "MTP off switches preserve whole-table warm plan and zero credit");
        }
        auto over_capacity=mtp; ++over_capacity.generate_tokens;
        expect(credit(over_capacity,{},manifest)==0,"MTP prompt plus G capacity overflow gets no credit");
        auto adaptive=mtp;
        adaptive.mtp.spec_min_p=0.85; adaptive.mtp.draft_chain=true;
        adaptive.mtp.window_prefix=true; adaptive.mtp.window_rows=8; adaptive.mtp.window_rows_set={2,4,8};
        const Environment campaign{{"STRIX_MTP_CONTROLLER","minp"},{"STRIX_COHORT_ROWS","32"},
            {"STRIX_COHORT_WIDE","all"},{"STRIX_LM_HEAD_COHORT_LEAN","1"}};
        expect(credit(adaptive,campaign,manifest)==mtp_credit,"adaptive min-p shares fixed-window source allowance and bound");
        for (const auto* key : {"STRIX_PLE_ROLLING","STRIX_CTX_EXACT_MEMORY"}) {
            auto disabled=campaign; disabled[key]="0";
            disabled["STRIX_PLE_ROLLING_POSITIONS"]="invalid-but-disabled";
            disabled["STRIX_PLE_ROLLING_CAP_BYTES"]="invalid-but-disabled";
            expect(credit(adaptive,disabled,manifest)==0,"default MTP switches restore historical warm zero-credit plan");
        }
        for (const auto& environment : {Environment{{"STRIX_MTP_CONTROLLER","depth"}},Environment{{"STRIX_MTP_CONTROLLER","survival"}}})
            expect(credit(mtp,environment,manifest)==0,"timing adaptive controllers remain gated");
        auto legacy_mtp=off;
        legacy_mtp.options.mtp=mtp.mtp;
        legacy_mtp.admitted=false; legacy_mtp.reason="explicit-route"; legacy_mtp.ple_reclaim_credit=0;
        auto off_print=legacy_mtp;
        off_print.environment["STRIX_PLE_ROLLING"]="0";
        auto historical=legacy_mtp;
        historical.environment.erase("STRIX_PLE_ROLLING");
        expect(effective_config_json(off_print)==effective_config_json(historical),"MTP rolling off printed config byte identity");
        for (unsigned excluded = 0; excluded < 13U; ++excluded) {
            auto options = admitted.options;
            if (excluded == 0U) options.residency = HipModelResidencyMode::resident_copy;
            if (excluded == 1U) options.model_copy_tier = HipModelCopyTier::all;
            if (excluded == 2U) options.execution.mode = DecodeExecutionMode::graph;
            if (excluded == 3U) options.known_prefix_prefill_rows = 0U;
            if (excluded == 4U) options.http_serve = true;
            if (excluded == 5U) { options.mtp.enabled = true; options.mtp.sample_timing_schedule = true; }
            if (excluded == 6U) options.planned_ple_io = true;
            if (excluded == 7U) options.serving_prefix_files.push_back("unused");
            if (excluded == 8U) options.tokens.clear();
            if (excluded == 9U) options.mtp.cohort.push_back({});
            if (excluded == 10U) options.session_load = "checkpoint";
            if (excluded == 11U) options.session_live_control = true;
            if (excluded == 12U) options.session_then_prompt = "suffix.ids";
            expect(credit(options,{},manifest) == 0U, "long_context_reclaim_exclusions: unsupported runtime route earns no credit");
        }
        for (const auto& environment : {Environment{{"STRIX_MODEL_COPY_ALL","1"}},
                Environment{{"STRIX_MODEL_COPY_ALL","fine"}}, Environment{{"STRIX_MODEL_COPY_ALL","uc"}},
                Environment{{"STRIX_MODEL_COPY_ALL","yes"}},
                Environment{{"STRIX_PLE_RECLAIMABLE","off"}}, Environment{{"STRIX_PLE_RECLAIMABLE","0"}}})
            expect(credit(admitted.options,environment,manifest) == 0U, "long_context_reclaim_environment: copy-all/off earns no credit");
        expect(credit(admitted.options,{{"STRIX_PLE_RECLAIMABLE","1"}},manifest) == admitted.ple_reclaim_credit &&
            credit(admitted.options,{{"STRIX_PLE_RECLAIMABLE","auto"}},manifest) == admitted.ple_reclaim_credit,
            "long_context_reclaim_auto_required: auto and required use identical accounting");
        const auto raised = credit(admitted.options,{{"STRIX_PLE_RECLAIM_DECODE_BYTES",std::to_string(gib_bytes)}},manifest);
        expect(admitted.ple_reclaim_credit - raised == gib_bytes - (256U << 20U),
            "long_context_reclaim_decode_allowance: explicit allowance only reduces credit");
        auto pinned = manifest;
        auto overlap = table;
        overlap.residency = ResidencyPolicy::resident;
        pinned.tensors.push_back(overlap);
        expect(credit(admitted.options,{},pinned) == 0U, "long_context_reclaim_registered: registered table pages earn no credit");
    }
    for (const auto n : {261632U,261633U,262112U,262143U,262144U,262145U}) {
        request.options.tokens.resize(n,1U);
        for (const auto generate : {1U,32U,512U}) {
            request.options.generate_tokens = generate;
            const auto config = resolve(request,observed);
            if (static_cast<std::uint64_t>(n) + generate <= model.max_position_embeddings) {
                expect(config.admitted && config.options.context_tokens == model.max_position_embeddings,
                    "long_context_clamp: padding is capped at the model contract limit");
            } else {
                bool refused = false;
                try { require_effective_prefill_execution(config); } catch (const std::invalid_argument&) { refused = true; }
                expect(!config.admitted && config.reason == "prompt-exceeds-model-context" && refused &&
                    effective_config_notice(config).find("using serial prefill") == std::string::npos,
                    "long_context_decline: prompt plus generation over model maximum is an explicit execution error");
            }
        }
    }
}

}
void prefill_identity_plumbing_tests() {
    const auto saved = detail::prefill_identity_override();
    detail::prefill_identity_override() = {};
    const auto options = parse({"decoder", "model.gguf", "1", "--prefill-exact-fp16",
        "--prefill-disable", "hc_output_wmma", "--prefill-disable=hc_output_wmma"});
    expect(options.prefill_exact_fp16 && options.prefill_disabled == 1U && !options.route_owned,
        "prefill switches parse without owning the automatic route");
    expect(parse_prefill_option_list("hc_output_wmma") == 1U, "all declared option names are recognized");
    for (const auto* value : {"unknown", "", "hc_output_wmma,", ",hc_output_wmma", "hc_output_wmma,,hc_output_wmma"}) {
        bool rejected = false;
        try { (void)parse({"decoder", "model.gguf", "1", "--prefill-disable", value}); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "malformed CLI prefill lists are rejected");
    }
    for (const auto* flag : {"--config", "--prefill-disable"}) {
        bool rejected = false;
        try { (void)parse({"decoder", "model.gguf", "1", flag}); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "prefill startup flags require values");
    }
    bool exact = false; unsigned disabled = 0U;
    std::istringstream config("# startup identity\nprefill_exact_fp16 = true\nprefill_disable = [\"hc_output_wmma\", \"hc_output_wmma\"] # off\n");
    read_prefill_config(config, exact, disabled);
    expect(exact && disabled == 1U, "startup config parses bool and quoted option array");
    const auto config_root = [] {
        for (unsigned attempt = 0;; ++attempt) {
            const auto path = std::filesystem::temp_directory_path() / ("qwen-prefill-startup-test-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(path)) return path;
        }
    }();
    const auto config_path = config_root / "startup.conf";
    struct RemoveConfig {
        std::filesystem::path path;
        ~RemoveConfig() {
            std::error_code error;
            std::filesystem::remove(path, error);
            std::filesystem::remove(path.parent_path(), error);
        }
    } cleanup{config_path};
    {
        std::ofstream file(config_path);
        file << "prefill_exact_fp16 = false\nprefill_disable = [\"hc_output_wmma\"]\n";
    }
    for (const bool config_first : {true, false}) {
        auto argv = std::vector<std::string>{"decoder", "model.gguf", "1"};
        const auto config_args = std::vector<std::string>{"--config", config_path.string()};
        const auto cli_args = std::vector<std::string>{"--prefill-exact-fp16", "--prefill-disable=hc_output_wmma"};
        const auto& first = config_first ? config_args : cli_args;
        const auto& second = config_first ? cli_args : config_args;
        argv.insert(argv.end(), first.begin(), first.end());
        argv.insert(argv.end(), second.begin(), second.end());
        const auto loaded = parse(argv);
        expect(loaded.prefill_exact_fp16 && loaded.prefill_disabled == 1U && !loaded.route_owned,
            "config file and CLI restrictions combine regardless of argument order");
    }
    expect(parse({"decoder", "model.gguf", "1", "--config=" + config_path.string()}).prefill_disabled == 1U,
        "config supports equals syntax");
    const auto captured = requested_prefill_environment([](const char* name) -> const char* {
        const std::string_view key(name);
        if (key == "STRIX_QPF2K_EARLY_SETUP") return "0";
        if (key == "STRIX_PREFILL_EXACT_FP16") return "1";
        for (const auto& option : kPrefillOptions) if (key == option.env) return "0";
        return nullptr;
    });
    expect(captured.size() == kPrefillOptions.size() + 2U && captured.at("STRIX_PREFILL_EXACT_FP16") == "1" &&
        captured.at("STRIX_QPF2K_EARLY_SETUP") == "0",
        "effective config captures setup order, master, diagnostic and every option environment control");
    for (const auto* text : {"prefill_exact_fp16 = 1", "unknown = true", "prefill_disable = [\"unknown\"]",
        "prefill_disable = [\"hc_output_wmma\",]", "prefill_disable = [hc_output_wmma]",
        "prefill_disable = [\"\"]", "prefill_exact_fp16 = false\nprefill_exact_fp16 = true"}) {
        bool rejected = false; bool test_exact = false; unsigned test_disabled = 0U;
        std::istringstream input(text);
        try { read_prefill_config(input, test_exact, test_disabled); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "malformed or unknown startup config is rejected");
    }
    const auto resolved = [&](Options parsed, Environment env) {
        return resolve_effective_config({std::move(parsed), std::move(env)},
            ModelContract::qwen38_flash_next(), {}, {}, "", {}, {});
    };
    const char* old_env = std::getenv("STRIX_PREFILL_HC_OUTPUT_WMMA");
    const std::optional<std::string> saved_env = old_env ? std::optional<std::string>(old_env) : std::nullopt;
    const auto set_option_env = [](const char* value) {
#ifdef _WIN32
        _putenv_s("STRIX_PREFILL_HC_OUTPUT_WMMA", value ? value : "");
#else
        if (value) ::setenv("STRIX_PREFILL_HC_OUTPUT_WMMA", value, 1);
        else ::unsetenv("STRIX_PREFILL_HC_OUTPUT_WMMA");
#endif
    };
    set_option_env("0");
    const auto plain_options = parse({"decoder", "model.gguf", "1"});
    const auto enabled = resolved(plain_options, {});
    expect(kPrefillOptionsReady == 1U && enabled.prefill_option_mask == kPrefillOptionsReady &&
        prefill_identity_hash(1U) == UINT64_C(17162327878354112617) &&
        effective_config_json(enabled).find(prefill_identity_fields(1U)) != std::string::npos,
        "clean supplied environment reports real READY default ON name/hash/options");
    for (const auto rows : {4096U, 32768U}) {
        for (unsigned policy = 0; policy < 3U; ++policy) {
            auto baseline_options = plain_options;
            baseline_options.tokens.resize(rows, 1U);
            baseline_options.baseline = policy == 0U;
            Environment env{{"STRIX_QPF2K_HC_LOWBAND", "3"}};
            if (policy == 1U) env["STRIX_BASELINE"] = "1";
            if (policy == 2U) env["STRIX_QPF2K_DEFAULTS"] = "0";
            const auto initialized = initialize_prefill_config({baseline_options, env});
            expect(initialized.options.prefill_disabled == baseline_options.prefill_disabled &&
                initialized.environment.at("STRIX_QPF2K_HC_LOWBAND") == "3",
                "bundle masters add no governed restrictions or initializer low-band override");
            const auto on = resolved(baseline_options, env);
            const auto repeated = resolved(baseline_options, env);
            expect(on.prefill_option_mask == kPrefillOptionsReady &&
                on.options.resolved_prefill_option_mask == kPrefillOptionsReady &&
                repeated.prefill_option_mask == on.prefill_option_mask &&
                effective_config_json(on).find(prefill_identity_fields(kPrefillOptionsReady)) != std::string::npos,
                "4K/32K baseline and defaults-off ON runs repeat the public fp16-v1+hcout16 identity");
            if (policy == 2U)
                expect(on.environment.at("STRIX_QPF2K_HC_LOWBAND") == "3",
                    "defaults-off preserves an explicit low-band policy");
        }
    }
    set_option_env("1");
    for (const auto& env : {Environment{{"STRIX_PREFILL_HC_OUTPUT_WMMA", "0"}},
             Environment{{"STRIX_PREFILL_EXACT_FP16", "1"}}}) {
        const auto off = resolved(plain_options, env);
        expect(off.prefill_option_mask == 0U &&
            effective_config_json(off).find(prefill_identity_fields(0U)) != std::string::npos,
            "captured individual/master OFF determines serialized identity");
    }
    for (const char* flag : {"--prefill-exact-fp16", "--prefill-disable=hc_output_wmma"})
        expect(resolved(parse({"decoder", "model.gguf", "1", flag}), {}).prefill_option_mask == 0U,
            "public CLI controls resolve OFF with real READY mask");
    for (const char* body : {"prefill_disable = [\"hc_output_wmma\"]\n", "prefill_exact_fp16 = true\n"}) {
        { std::ofstream file(config_path); file << body; }
        expect(resolved(parse({"decoder", "model.gguf", "1", "--config", config_path.string()}), {}).prefill_option_mask == 0U,
            "config-file individual/master controls resolve OFF");
    }
    expect(governed_hc_output_kernel(2U, 0U) == 1U && governed_hc_output_kernel(2U, 1U) == 2U,
        "legacy output selector obeys public OFF controls");
    set_prefill_identity_override(false, 1U);
    set_prefill_identity_override(true, 8U);
    expect(detail::prefill_identity_override().exact && detail::prefill_identity_override().disabled == 9U &&
        prefill_options_active() == 0U, "startup overrides accumulate and master restores FP16");
    expect(resolved(plain_options, {}).prefill_option_mask == 1U &&
        effective_config_json(enabled).find(prefill_identity_fields(1U)) != std::string::npos,
        "global master override and later process state cannot suppress captured READY configuration");
    expect(prefill_identity_name() == "fp16-v1" && prefill_identity_hash() == prefill_identity_hash(0U) && prefill_identity_hash(0U) == UINT64_C(11531825512229770990),
        "plain FP16 name and hash agree with the active set");
    expect(prefill_identity_name(1U) == "fp16-v1+hcout16",
        "identity components follow enum order");
    std::set<std::uint64_t> hashes;
    for (unsigned mask = 0; mask < 2U; ++mask) hashes.insert(prefill_identity_hash(mask));
    expect(hashes.size() == 2U, "every declared combination has a distinct identity hash");
    const auto fields = prefill_identity_fields(0U);
    const auto hash_begin = fields.find("\"prefill_identity_hash\":\"") + std::string("\"prefill_identity_hash\":\"").size();
    expect(fields.substr(hash_begin, 16U).find_first_not_of("0123456789abcdef") == std::string::npos &&
        fields[hash_begin + 16U] == '"' && fields.find("\"prefill_options_active\":[]") != std::string::npos,
        "receipt hash is 16 lowercase hex digits and inactive options form an empty array");
    expect(prefill_identity_fields(1U).find("\"hc_output_wmma\"") != std::string::npos,
        "receipt active option names follow enum order");
    EffectiveConfig effective;
    effective.options = options;
    expect(effective_config_json(effective).find(fields) != std::string::npos,
        "effective config reports the same identity fields as receipts");
    const auto replay = effective_prefill_argv(options);
    expect(std::find(replay.begin(), replay.end(), "--prefill-exact-fp16") != replay.end(),
        "effective argv preserves the master switch");
    for (unsigned mask : {0U, 1U}) {
        auto requested = plain_options;
        requested.rotary = RotaryDescriptor::yarn(2U);
        requested.prefill_disabled = mask ? 0U : 1U;
        const RequestedOptions request{requested, {}};
        const auto direct = resolve_yarn_config(request, ModelContract::qwen38_flash_next(), {}, {}, "", {}, {});
        expect(direct.prefill_option_mask == mask && direct.options.resolved_prefill_option_mask == mask,
            "direct YaRN resolver initializes options even before a refusal");
        const auto native = initialize_prefill_config(request);
        expect(native.prefill_option_mask == direct.prefill_option_mask, "both resolvers use one initializer");
    }
    auto bundled_mtp = plain_options;
    bundled_mtp.mtp.enabled = true;
    bundled_mtp.mtp_bundle_path = "model.mtp-bundle.json";
    const auto bundled = initialize_prefill_config({bundled_mtp, {}});
    expect(bundled.prefill_option_mask == 0U &&
        (bundled.options.prefill_disabled & prefill_option_bit(PrefillOption::hc_output_wmma)),
        "automatic YaRN MTP bundle retains its qualified prefill arithmetic");
    auto split = plain_options;
    split.numerical_profile = dense_a8_profile;
    split.numerical_profile_explicit = true;
    split.prefill_numerical_profile = DecodeNumericalProfile::fast_prefill_f16wmma_v1;
    split.known_prefix_window = split.known_prefix_full_rows = true;
    split.known_prefix_prefill_rows = 2048U;
    split.validation_final_state_path = "private-state";
    validate_prefill_numerical_profile_cli(split);
    const auto pair = numerical_profile_pair(split);
    const auto on_identity = prefill_arithmetic_identity(pair, 1U, false);
    expect(on_identity != prefill_arithmetic_identity(pair, 0U, false), "host/runtime arithmetic binds option mask");
    expect(on_identity != prefill_arithmetic_identity(pair, 1U, true), "host/runtime arithmetic binds GSQ");
    set_option_env(saved_env ? saved_env->c_str() : nullptr);
    detail::prefill_identity_override() = saved;
    expect((prefill_options_active() & ~kPrefillOptionsReady) == 0U,
        "unready options never appear in the active identity");
}

// Receipt contract: hcout16 declares the governed option on every route, but HC output WMMA executes only
// with a fast prefill pair. Equivalence must therefore never key on the declared mask alone.
void prefill_identity_equivalence_tests() {
    const auto route = [](std::vector<std::string> extra) {
        std::vector<std::string> args{"decoder", "model.gguf", "1", "--numerical-profile",
            "hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1", "--known-prefix-window",
            "--known-prefix-full-rows", "--known-prefix-workspace-budget-bytes", "8589934592",
            "--known-prefix-host-metadata-budget-bytes", "1073741824", "--known-prefix-uma-budget-bytes",
            "128849018880", "--known-prefix-host-headroom-bytes", "4294967296", "--known-prefix-prefill-rows", "4096"};
        args.insert(args.end(), extra.begin(), extra.end());
        return parse(args);
    };
    const auto fast = route({"--prefill-numerical-profile", "fast-prefill-f16wmma-v1"});
    const auto plain = route({});
    const auto baseline = route({"--baseline"});
    expect(numerical_profile_pair(fast).fast_components != 0U && numerical_profile_pair(plain).fast_components == 0U &&
        numerical_profile_pair(baseline).fast_components == 0U, "routes resolve to one fast and two non-fast prefill pairs");
    const auto key = [](const Options& o) {
        std::string cli;
        for (const auto& arg : effective_prefill_argv(o)) cli += std::to_string(arg.size()) + ":" + arg;
        return prefix_effective_execution_signature(cli, o.baseline, false, {}, kPrefillOptionsReady);
    };
    expect(key(fast) != key(plain) && key(fast) != key(baseline) && key(plain) != key(baseline),
        "prefix-cache keys of non-fast and baseline routes never equal the fast-pair key under the same declared mask");
    const auto arithmetic = [](const Options& o, unsigned mask) {
        return prefill_arithmetic_identity(numerical_profile_pair(o), mask, false);
    };
    expect(arithmetic(plain, kPrefillOptionsReady) == arithmetic(plain, 0U) &&
        arithmetic(baseline, kPrefillOptionsReady) == arithmetic(baseline, 0U),
        "on non-fast pairs the declared mask is inert in the compiled prefill identity");
    expect(arithmetic(fast, kPrefillOptionsReady) != arithmetic(fast, 0U) &&
        arithmetic(fast, kPrefillOptionsReady) != arithmetic(plain, kPrefillOptionsReady),
        "the session identity follows the executed arithmetic, not the declared name");
}

int main(int argc, char** arguments) {
    resident_fallback_tests();
    mtp_reservation_arithmetic_tests();
    prefill_identity_plumbing_tests();
    prefill_identity_equivalence_tests();
    const auto model = ModelContract::qwen38_flash_next();
    gguf::FileInfo info;
    ModelManifest manifest;
    manifest.architecture = "qwen3next";
    const auto tensor = [&](std::uint32_t layer, const char* name, std::uint64_t width, std::uint64_t outputs) {
        ManifestTensor item;
        item.name = "blk." + std::to_string(layer) + "." + name + ".weight";
        item.layer = layer; item.role = TensorRole::dense_projection;
        item.dimensions = {width,outputs}; item.ggml_type = 8U;
        item.byte_size = width / 32U * outputs * 34U;
        manifest.tensors.push_back(std::move(item));
    };
    for (std::uint32_t layer = 0; layer < model.hidden_layers; ++layer) {
        tensor(layer,"hc_attn_up",model.hc_lowrank,model.hidden_size * model.hc_count);
        tensor(layer,"hc_ffn_up",model.hc_lowrank,model.hidden_size * model.hc_count);
        tensor(layer,"ffn_gate_shexp",model.hidden_size,model.shared_expert_intermediate_size);
        tensor(layer,"ffn_up_shexp",model.hidden_size,model.shared_expert_intermediate_size);
        tensor(layer,"ffn_down_shexp",model.shared_expert_intermediate_size,model.hidden_size);
        if ((layer + 1U) % model.full_attention_interval != 0U) continue;
        tensor(layer,"attn_q",model.hidden_size,model.attention_heads * model.head_dim * 2U);
        tensor(layer,"attn_k",model.hidden_size,model.kv_heads * model.head_dim);
        tensor(layer,"attn_v",model.hidden_size,model.kv_heads * model.head_dim);
        tensor(layer,"indexer_q_proj",model.hidden_size,model.indexer_n_heads * model.indexer_head_dim);
        tensor(layer,"indexer_k_proj",model.hidden_size,model.indexer_head_dim);
    }
    tensor(0U,"final_hc_up",model.hc_lowrank,model.hidden_size * model.hc_count);
    manifest.tensors.back().name = "output_hc_up.weight";
    manifest.tensors.back().layer.reset();
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    if (argc==2 && std::string_view(arguments[1])=="--serve-prefill-admission") {
        serve_prefill_admission_tests(model,manifest);
        return failures ? 1 : 0;
    }
    if (argc==2 && std::string_view(arguments[1])=="--yarn-owner-inventory") {
        RequestedOptions request; request.options = parse(mtp_fixture());
        request.options.context_explicit = true;
        auto& mtp = request.options.mtp;
        mtp.window_rows = 8U; mtp.window_rows_set = {2U,4U,8U};
        mtp.candidate_depth = 7U;
        mtp.window_prefix = mtp.window_qsa_rows = mtp.draft_chain = true;
        mtp.window_arena_bytes = mtp.window_capture_bytes = mtp.window_checkpoint_bytes = 4096U;
        mtp.window_total_bytes = UINT64_C(2)<<30U; mtp.window_history_intervals = 4096U;
        request.options.known_prefix_window = request.options.known_prefix_full_rows = true;
        yarn_owner_inventory_regression(request,info,manifest);
        return failures ? 1 : 0;
    }
    yarn_admission_tests(info,manifest);
    const PrefillCapabilities capabilities{true,true,true};
    auto memory = ample_memory();
    const auto fixture_requirements = memory.estimate;
    memory.estimate = [&](const Options& options) {
        auto requirements = fixture_requirements(options);
        const auto compiler = prompt_compiler_requests(options,model,manifest);
        const auto buffers = build_serial_execution_buffer_plan(model,compiler.buffers);
        requirements.organization = {{"buffer_peak_internal_bytes",buffers.peak_internal_bytes},
            {"gdn_workspace_bytes",buffers.gdn_workspace_bytes}, {"hyper_workspace_bytes",buffers.hyper_workspace_bytes},
            {"wave8_gdn_input",compiler.binding.wave8_q8_gdn_input},
            {"wave8_gdn_output",compiler.binding.wave8_q8_gdn_output},
            {"wave8_qsa_output",compiler.binding.wave8_q8_qsa_output},
            {"wave8_lm_head",compiler.binding.wave8_q8_lm_head}, {"dense_a8",compiler.binding.dense_a8}};
        return requirements;
    };
    std::vector<std::string> plain_argv{"decoder","MODEL"};
    plain_argv.insert(plain_argv.end(),32768U,"1");
    plain_argv.insert(plain_argv.end(),{"--generate","32"});
    auto request = RequestedOptions{parse(std::move(plain_argv)),{}};
    const auto resolve = [&](const RequestedOptions& value, const PrefillMemory& observation) {
        return resolve_w7_case(value,model,info,manifest,validated_model_content_identity,capabilities,observation);
    };
    {
        constexpr auto candidate_key = "STRIX_QPF2K_MOE_ROUTE_BF16";
        const auto base = resolve(request,memory);
        expect(!base.options.prefill_moe_route_bf16 && base.error.empty(),"route BF16 default off");
        for (const auto profile : {qwen::profile::Name::standard, qwen::profile::Name::fast, qwen::profile::Name::quality}) {
            for (const bool allow : {false, true}) {
                auto selected = request;
                qwen::profile::Inputs inputs;
                inputs.env = [](const char*) -> const char* { return nullptr; };
                inputs.allow_pending = allow;
                selected.options.product_profile = qwen::profile::resolve(profile, inputs);
                const auto got = resolve(selected, memory);
                const bool enabled = profile == qwen::profile::Name::fast;
                Environment exported;
                qwen::profile::apply(got.options.product_profile, [&](const char* key, const char* value) {
                    exported[key] = value; return true;
                });
                expect(enabled ? exported.at(candidate_key) == "1" &&
                    exported.at("STRIX_QWEN_PROFILE_OWNED_moe_route_bf16") == "fast:provisional:1" : exported.empty(),
                    "native CLI exports storage only after route admission");
                expect(got.error.empty() && got.options.prefill_moe_route_bf16 == enabled,
                    "fast profile applies provisional storage through native resolver without permission");
                expect(moe_prefill_storage_name(got.options.prefill_moe_route_bf16) ==
                    std::string_view(enabled ? "fp16-v1+moeroute16" : "fp16-v1") &&
                    got.options.product_profile.qualification() == (enabled ? (allow ?
                        "qualification in progress: moe_route_bf16; unqualified: gdn_fp16_chunk" : "qualification in progress: moe_route_bf16")
                        : "qualified"),
                    "product receipt and actual storage identity agree");
                const auto pair = numerical_profile_pair(got.options);
                // fast with --allow-pending also runs the runnable GDN chunk candidate.
                const unsigned gdn = enabled && allow ? prefill_option_bit(PrefillOption::gdn_fp16_chunk) : 0U;
                expect(got.prefill_option_mask == (kPrefillOptionsReady | gdn),
                    "every product profile retains the governed HC default");
                const auto composed = prefill_arithmetic_identity(pair, got.prefill_option_mask, false);
                for (const bool master : {false, true}) {
                    auto restricted = selected;
                    if (master) restricted.options.prefill_exact_fp16 = true;
                    else restricted.options.prefill_disabled = prefill_option_bit(PrefillOption::hc_output_wmma);
                    const auto off = resolve(restricted, memory);
                    expect(off.error.empty() && off.prefill_option_mask == (master ? 0U : gdn) &&
                        off.options.product_profile.json() == got.options.product_profile.json() &&
                        numerical_profile_pair(off.options) == pair,
                        "HC restriction preserves product profile and BF16 route admission");
                    expect(prefill_arithmetic_identity(pair, off.prefill_option_mask, false) != composed,
                        "HC restriction changes composed arithmetic identity independently of BF16 storage");
                    const auto report = serve::parse_json(effective_config_json(off));
                    expect(report.find("prefill_identity")->as_string() == (master || !gdn ? "fp16-v1" : "fp16-v1+gdnchunk16") &&
                        report.find("prefill_moe_route_bf16")->as_bool() == enabled,
                        "inspection reports independent HC and BF16 controls");
                }
            }
        }
        expect(moe_prefill_storage_identity(false) == 0xa0094634843e16eeULL &&
            numerical_profile_pair_identity(numerical_profile_pair(base.options)) == 0xf51680d10954bfe0ULL,
            "plain FP16 pair identity unchanged");
        for (const unsigned length : {4096U,32768U}) for (const char* vec : {"0","1"}) {
            auto on = request; on.options.tokens.resize(length); on.environment[candidate_key] = "1";
            on.environment["STRIX_PREFILL_MOE_MERGE_VEC"] = vec;
            const auto got = resolve(on,memory);
            expect(got.error.empty() && got.options.prefill_moe_route_bf16,"4K/32K route BF16 admitted as candidate");
            const auto pair = numerical_profile_pair(got.options);
            expect((pair.fast_components & kFastPrefillMoeRouteBF16) && numerical_profile_pair_admitted(pair) &&
                numerical_profile_pair_identity(pair) != numerical_profile_pair_identity(numerical_profile_pair(base.options)) &&
                moe_prefill_storage_identity(true) != moe_prefill_storage_identity(false),"route BF16 distinct composed identity");
            expect(effective_config_json(got).find("\"prefill_moe_route_bf16\":true") != std::string::npos &&
                effective_config_json(got).find("needs qualification") != std::string::npos,"candidate inspection disclosure");
        }
        for (const char* off : {"0","0anything"}) {
            auto r=request; r.environment[candidate_key]=off; const auto got=resolve(r,memory);
            expect(!got.options.prefill_moe_route_bf16 && numerical_profile_pair(got.options)==numerical_profile_pair(base.options),
                "zero-prefixed env values preserve arithmetic");
        }
        for (const char* suppress : {"STRIX_BASELINE","STRIX_QPF2K_DEFAULTS"}) {
            auto r=request; r.environment[candidate_key]="1";
            r.environment[suppress]=std::string_view(suppress)=="STRIX_BASELINE"?"1":"0suffix";
            expect(!resolve(r,memory).options.prefill_moe_route_bf16,"master env suppresses candidate");
        }
        auto baseline=request; baseline.options.baseline=true; baseline.options.prefill_moe_route_bf16_requested=true;
        expect(!resolve(baseline,memory).options.prefill_moe_route_bf16,"CLI baseline suppresses candidate CLI");
        const auto parsed=parse({"decoder","MODEL","1","--prefill-moe-route-bf16","on"});
        expect(parsed.prefill_moe_route_bf16_requested==true && !parsed.any_low_level_flag && !parsed.route_owned,
            "native candidate CLI parses without owning execution organization");
        auto native_cli=request; native_cli.options=parsed; native_cli.options.tokens=request.options.tokens;
        native_cli.options.generate_tokens=request.options.generate_tokens;
        const auto native_on=resolve(native_cli,memory);
        auto native_env=request; native_env.environment[candidate_key]="1";
        const auto native_env_on=resolve(native_env,memory);
        expect(native_on.admitted && native_on.options.prefill_moe_route_bf16 &&
            numerical_profile_pair(native_on.options)==numerical_profile_pair(native_env_on.options) &&
            native_on.options.rotary.mode()==RotaryMode::native,"native CLI and environment retain candidate semantics");
        native_cli.options.prefill_moe_route_bf16_requested=false;
        const auto native_off=resolve(native_cli,memory);
        expect(native_off.admitted && !native_off.options.prefill_moe_route_bf16 &&
            numerical_profile_pair(native_off.options)==numerical_profile_pair(base.options) &&
            native_off.host_compilation==base.host_compilation,"native CLI off retains plain identities");
        auto precedence=request; precedence.environment[candidate_key]="1";
        precedence.options.prefill_moe_route_bf16_requested=false;
        expect(!resolve(precedence,memory).options.prefill_moe_route_bf16,"CLI off overrides env on");
        bool bad=false; try { (void)parse({"decoder","MODEL","1","--prefill-moe-route-bf16","perhaps"}); }
        catch(const std::invalid_argument&) {bad=true;} expect(bad,"candidate invalid CLI rejected");
        for (const char* vec : {"0","1"}) for (const auto& [key,value] : Environment{{"STRIX_QPF2K_MOE","513"},{"STRIX_QPF2K_MOE_SMALL","16"},
            {"STRIX_QPF2K_MOE_MC","2"},{"STRIX_QPF2K_MOE_DA16","0"},{"STRIX_QPF2K_MOE_ROUTE","6"},
            {"STRIX_INTPREFILL_MOE_TUNE","0"}}) {
            auto o=base.options; Environment env=base.environment; env[candidate_key]="1";env[key]=value;env["STRIX_PREFILL_MOE_MERGE_VEC"]=vec;std::string error;
            resolve_moe_route_bf16(o,env,false,true,error);
            expect(!o.prefill_moe_route_bf16 && error.find("declined")!=std::string::npos,"unsupported MoE organization declines");
        }
        {
            auto o=base.options; auto env=base.environment;env[candidate_key]="1";env["STRIX_QPF2K_MOE_DA16"]="";std::string error;
            resolve_moe_route_bf16(o,env,false,true,error);
            expect(!o.prefill_moe_route_bf16 && !error.empty(),"explicit empty organization declines");
        }
        for (unsigned route=0;route<8U;++route) {
            auto o=base.options; Environment env=base.environment;env[candidate_key]="1";std::string error;
            if(route==0U)o.mtp.enabled=true;
            if(route==1U)o.http_serve=true;
            if(route==2U)o.http_batch=true;
            if(route==3U)o.known_prefix_prefill_rows=0;
            if(route==4U)o.prefill_fast_components=kFastPrefillDense;
            if(route==5U)o.serving_concurrency=2;
            resolve_moe_route_bf16(o,env,route==6U,route!=7U,error);
            expect(!o.prefill_moe_route_bf16 && !error.empty(),"excluded controller/writer/backend declines");
        }
        auto caps=serve::Json::object();moe_route_bf16_capabilities(caps);
        expect(!caps.find("prefill_moe_route_bf16")->find("supported")->as_bool(),"HTTP capability refuses candidate");
        {
            const char* key = "STRIX_QPF2K_MOE_ROUTE_BF16";
            const char* old = std::getenv(key);
            const auto saved = old ? std::optional<std::string>{old} : std::nullopt;
            ::setenv(key,"0",1);
            const auto off = session_numerical_config("same execution");
            const auto yarn_rotary=mock_rotary(2)->identity();
            const auto yarn_off=session_numerical_config("same execution",yarn_rotary);
            ::setenv(key,"1",1);
            expect(session_numerical_config("same execution") != off,"route BF16 changes session namespace");
            expect(session_numerical_config("same execution",yarn_rotary)!=yarn_off,
                "YaRN candidate separates persisted state namespace with fixed rotary identity");
            if(saved) ::setenv(key,saved->c_str(),1); else ::unsetenv(key);
        }
        const auto captured=requested_prefill_environment([](const char* name)->const char* {
            return std::string_view(name)=="STRIX_QPF2K_MOE_ROUTE_BF16"?"1":nullptr;
        });
        expect(captured.contains(candidate_key),"candidate env captured for inspection and execution");
        expect(moe_route_bf16_environment_requested("1",nullptr,nullptr) &&
            !moe_route_bf16_environment_requested("0suffix",nullptr,nullptr) &&
            !moe_route_bf16_environment_requested("1","1",nullptr) &&
            !moe_route_bf16_environment_requested("1",nullptr,"0suffix"),"CUDA environment suppression contract");
        for (const auto& [bits,expected] : std::initializer_list<std::pair<std::uint32_t,std::uint16_t>>{
            {0x3f808000U,0x3f80U},{0x3f818000U,0x3f82U},{0xbf808000U,0xbf80U},{0xbf818000U,0xbf82U},
            {0x00008000U,0U},{0x00018000U,2U},{0x80000000U,0x8000U},{0x7f800000U,0x7f80U},
            {0xff800000U,0xff80U},{0x7f800001U,0x7fc0U},{0x7fffffffU,0x7fffU},{0xffffffffU,0xffffU}})
            expect(moe_route_bf16_bits(bits)==expected,"route BF16 RNE / nonfinite / denormal bits");
    }
    {
        auto argv=mtp_fixture(); argv.insert(argv.begin()+5,13,"1");
        // Exercise the standard serving argv and MTP CLI through the same
        // inspection resolution used before any profile process exports.
        for (const auto batch : {"cli", "auto", "0", "2"}) {
            auto args = argv;
            const bool http = std::string_view(batch) != "cli";
            if (http) *std::next(std::find(args.begin(), args.end(), "--mtp-candidate-depth")) = "7";
            if (http) args.insert(args.end(), {"--http-serve", "--mtp-window-prefix",
                "--mtp-draft-chain", "--mtp-target-window-rows-set", "2,4,8",
                "--mtp-window-arena-bytes", "2147483648", "--mtp-window-capture-bytes", "67108864",
                "--mtp-window-checkpoint-bytes", "8589934592", "--mtp-window-total-bytes", "17179869184",
                "--mtp-window-history-intervals", "8192"});
            if (http && std::string_view(batch) != "auto")
                args.insert(args.end(), {"--http-batch", batch});
            args.insert(args.end(), {"--profile", "fast", "--print-effective-config"});
            auto selected = RequestedOptions{parse(args), {}};
            const auto got = resolve(selected, memory);
            expect(got.error.empty() && !got.options.prefill_moe_route_bf16,
                "fast HTTP and MTP inspection softly decline candidate storage");
            const auto* row = got.options.product_profile.find("moe_route_bf16");
            const auto reason = got.options.prefill_moe_route_bf16_reason;
            expect(row && row->status == qwen::profile::Status::pending && row->value == "0" &&
                row->reason == reason && !reason.empty() && reason != "off" &&
                got.options.product_profile.qualification() == "qualified",
                "declined row reports builtin, pending and the storage decline reason");
            Environment exported;
            qwen::profile::apply(got.options.product_profile, [&](const char* key, const char* value) {
                exported[key] = value; return true;
            });
            expect(!exported.contains("STRIX_QPF2K_MOE_ROUTE_BF16") &&
                !exported.contains("STRIX_QWEN_PROFILE_OWNED_moe_route_bf16") &&
                !got.environment.contains("STRIX_QPF2K_MOE_ROUTE_BF16"),
                "declined candidate is never exported");
            const auto json = serve::parse_json_unique(effective_config_json(got));
            const auto* receipt = json.find("profile")->find("levers")->find("moe_route_bf16");
            expect(receipt->find("status")->as_string() == "pending" &&
                receipt->find("reason")->as_string() == reason,
                "inspection profile reports the final route admission");
            unsigned notices = 0;
            for (const auto& notice : got.options.product_profile.notices())
                if (notice.find("moe_route_bf16") != std::string::npos) {
                    ++notices;
                    expect(notice.find(" pending ") != std::string::npos && notice.find(reason) != std::string::npos &&
                        notice.find("is bit-changing") == std::string::npos,
                        "startup notice reports pending rather than applied storage");
                }
            expect(notices == 1, "one final candidate notice");
            if (http) validate_effective_startup(got);
            for (const bool cli : {false, true}) {
                auto explicit_args = args;
                if (cli) explicit_args.insert(explicit_args.end(), {"--prefill-moe-route-bf16", "on"});
                auto explicit_request = RequestedOptions{parse(explicit_args), {}};
                if (!cli) {
                    explicit_request.environment["STRIX_QPF2K_MOE_ROUTE_BF16"] = "1";
                    qwen::profile::Inputs inputs;
                    inputs.env = [](const char* key) -> const char* {
                        return std::string_view(key) == "STRIX_QPF2K_MOE_ROUTE_BF16" ? "1" : nullptr;
                    };
                    explicit_request.options.product_profile = qwen::profile::resolve(qwen::profile::Name::fast, inputs);
                }
                const auto refused = resolve(explicit_request, memory);
                expect(!refused.options.prefill_moe_route_bf16 &&
                    refused.error == "MoE route BF16 candidate declined: " + reason,
                    "explicit CLI and unowned environment keep the hard refusal");
                explicit_request.options.prefill_moe_route_bf16_requested = false;
                const auto off = resolve(explicit_request, memory);
                expect(off.error.empty() && !off.options.prefill_moe_route_bf16 &&
                    off.options.prefill_moe_route_bf16_reason == "off", "explicit OFF preserves FP32 storage");
            }
        }
        argv.insert(argv.end(),{"--http-serve","--http-batch","0"});
        auto served=RequestedOptions{parse(argv),{}};
        const auto resolved=resolve(served,memory);
        expect(mtp_tiled_prompt(resolved.options) && resolved.options.known_prefix_prefill_rows==8192,
            "real MTP HTTP argv defers tiled prefill until capacity resolution");
        validate_http_serve(resolved.options);
        argv.insert(argv.end(),{"--prefix-cache-block-rows","0","--prefix-cache-block-rows","0"});
        bool duplicate=false;
        try {(void)parse(argv);} catch(const std::invalid_argument&) {duplicate=true;}
        expect(duplicate,"duplicate zero checkpoint options are rejected");
        auto budget_argv=mtp_fixture(); budget_argv.insert(budget_argv.begin()+5,13,"1");
        budget_argv.insert(budget_argv.end(),{"--http-serve","--http-batch","0",
            "--known-prefix-window","--known-prefix-workspace-budget-bytes","4294967296",
            "--known-prefix-host-headroom-bytes","1073741824","--mtp-calibration-rows","4"});
        const auto budgeted=resolve(RequestedOptions{parse(budget_argv),{}},memory);
        expect(budgeted.options.known_prefix_prefill_rows==8192 &&
            budgeted.options.known_prefill_workspace_budget==4*gib_bytes &&
            budgeted.options.known_prefill_host_headroom==gib_bytes && budgeted.options.mtp.calibration_rows==4,
            "HTTP preserves partial explicit budgets, organization and calibration before admission");
        auto http=request; http.options.http_serve=true; http.options.route_owned=true;
        http.options.context_tokens=32768; http.options.context_explicit=true; http.options.unqualified_context=true;
        http.options.tokens.clear();
        const auto fast=resolve(http,memory);
        expect(fast.options.known_prefix_prefill_rows==8192 && fast.options.automatic_prefill,
            "HTTP allocates automatic T8192 before prompts arrive");
        expect(fast.options.context_tokens==32768 && !fast.options.prefill_numerical_profile &&
            fast.options.numerical_profile==resolve_prefill_effective_config(http,model,info,manifest,
                validated_model_content_identity,capabilities,memory).options.numerical_profile,
            "HTTP preserves context and numerical identity");
        expect(fast.options.known_prefix_tail_rows==std::vector<std::uint32_t>({1024,256,64,16}),
            "HTTP keeps all tail owners for future requests");
        auto context=http; context.options.context_tokens=4096;
        const auto smaller=resolve(context,memory);
        expect(smaller.options.known_prefix_prefill_rows==4096 && smaller.serve_prefill_notice.find("T8192:context")!=std::string::npos,
            "HTTP context fallback is logged");
        context.options.context_tokens=8;
        expect(!resolve(context,memory).options.known_prefix_prefill_rows,"HTTP context below T16 declines");
        auto pressure=memory;
        pressure.estimate=[&](const Options& o) {
            auto r=memory.estimate(o); r.workspace=12*gib_bytes; return r;
        };
        const auto uniform=resolve(http,pressure);
        expect(uniform.options.known_prefix_prefill_rows==8192 && uniform.options.known_prefill_workspace_budget==12*gib_bytes,
            "HTTP uniform logits workspace above the compact CLI ceiling still admits T8192");
        pressure.estimate=[&](const Options& o) {
            auto r=memory.estimate(o); if(o.known_prefix_prefill_rows==8192) r.workspace=128*gib_bytes; return r;
        };
        const auto fitted=resolve(http,pressure);
        expect(fitted.options.known_prefix_prefill_rows==4096 && fitted.serve_prefill_notice.find("T8192:memory")!=std::string::npos,
            "HTTP retries a smaller tile on memory pressure");
        pressure.available=1;
        const auto declined=resolve(http,pressure);
        expect(!declined.options.known_prefix_prefill_rows && declined.serve_prefill_notice.find("explicit tiled configuration required")!=std::string::npos,
            "HTTP insufficient memory requires an explicit tiled route");
        pressure=memory; pressure.gtt_limit.reset();
        expect(!resolve(http,pressure).options.known_prefix_prefill_rows,"HTTP unknown GTT declines");
        auto off=http; off.environment["STRIX_SERVE_PREFILL"]="0";
        expect(!resolve(off,memory).options.known_prefix_prefill_rows,"HTTP T automatic off switch");
        off=http; off.options.baseline=true;
        expect(!resolve(off,memory).options.known_prefix_prefill_rows,"HTTP baseline leaves T alone");
        auto explicit_t=http; explicit_t.options=fast.options;
        explicit_t.options.known_prefix_prefill_explicit=true; explicit_t.options.known_prefix_prefill_rows=2048;
        explicit_t.options.known_prefix_tail_rows.clear();
        explicit_t.options.prefix_cache_host_bytes=gib_bytes;
        const auto cached=resolve(explicit_t,memory);
        expect(cached.options.known_prefix_prefill_rows==2048 && cached.options.prefix_cache_block_rows==1024,
            "explicit T wins and configured cache defaults to 1024 checkpoints");
        explicit_t.options.prefix_cache_block_rows_explicit=true;
        explicit_t.options.prefix_cache_block_rows=0;
        expect(resolve(explicit_t,memory).options.prefix_cache_block_rows==0,"explicit zero restores T checkpoints");
        explicit_t.options.prefix_cache_block_rows=256;
        expect(resolve(explicit_t,memory).options.prefix_cache_block_rows==256,"explicit cache checkpoint size wins");
        auto no_cache=http; no_cache.environment["STRIX_PREFIX_CACHE"]="0";
        expect(resolve(no_cache,memory).options.prefix_cache_block_rows==0,"disabled cache gets no checkpoint default");
        auto cli=fast.options; cli.http_serve=false;
        const char* prior_small=std::getenv("STRIX_BATCH_PROMPT_SMALL");
        const auto saved=prior_small ? std::optional<std::string>(prior_small) : std::nullopt;
        unsetenv("STRIX_BATCH_PROMPT_SMALL");
        expect(!batch_prompt_small(fast.options) && !batch_prompt_small(cli),"SMALL is opt-in for HTTP and CLI");
        setenv("STRIX_BATCH_PROMPT_SMALL","0",1);
        expect(!batch_prompt_small(fast.options),"SMALL explicit off");
        setenv("STRIX_BATCH_PROMPT_SMALL","1",1);
        expect(batch_prompt_small(fast.options) && batch_prompt_small(cli),"SMALL opt-in for HTTP and CLI");
        if(saved) setenv("STRIX_BATCH_PROMPT_SMALL",saved->c_str(),1); else unsetenv("STRIX_BATCH_PROMPT_SMALL");
    }
    {
        auto configured=request;
        configured.environment=requested_prefill_environment([](const char* name) -> const char* {
            return std::string_view(name)=="STRIX_QSA_SCORE_RING_ROWS"?"256":nullptr;
        });
        const auto resolved=resolve(configured,memory);
        expect(resolved.environment.at("STRIX_QSA_SCORE_RING_ROWS")=="256" &&
            effective_config_json(resolved).find("STRIX_QSA_SCORE_RING_ROWS")!=std::string::npos,
            "ring row override is captured in effective configuration");
        configured.environment["STRIX_QSA_SCORE_RING_ROWS"]="192";
        bool rejected=false;
        try { (void)resolve(configured,memory); } catch(const std::invalid_argument&) { rejected=true; }
        expect(rejected,"invalid ring override cannot silently decline the automatic route");
        for(const char* off : {"STRIX_CTX_EXACT_MEMORY","STRIX_QSA_SCORE_RING"}) {
            auto disabled=configured;disabled.environment[off]="0";
            expect(resolve(disabled,memory).admitted,"ring off switches ignore an inactive rows override");
        }
        const auto base=resolve(request,memory);
        const auto plan_ids=[&](const Options& options) {
            std::vector<std::uint64_t> ids;
            for (const auto profile : {options.numerical_profile,numerical_profile_pair(options).prefill}) {
                auto req=prompt_compiler_requests(options,model,manifest);
                if(profile!=options.numerical_profile) req.binding.prefill_arithmetic_identity=numerical_profile_pair_identity(numerical_profile_pair(options));
                const auto registry=config_test_registry(model,req.binding);
                const auto compiled=compile_serial_execution(model,registry,req.binding,req.buffers,256);
                ids.push_back(compiled.execution_identity); ids.push_back(compiled.kernel_plan_identity);
            }
            return ids;
        };
        const auto base_ids=plan_ids(base.options);
        const auto numerical=[](EffectiveConfig c) {
            c.environment.erase("STRIX_PREFIX_CACHE"); c.environment.erase("STRIX_PREFIX_CACHE_RETENTION");
            c.sources.erase("--prefix-cache"); c.sources.erase("--prefix-cache-retention");
            c.options.prefix_cache=true; c.options.prefix_cache_retention="fair";
            c.options.prefix_cache_reason="not-serving";
            // Joining follows the cache switch; it is operational, not numerical.
            c.options.prefix_cache_inflight=true; c.options.prefix_cache_inflight_source="default";
            return without_cache_policy(effective_config_json(c));
        };
        for (const auto& suffix : std::vector<std::vector<std::string>>{{},{"--prefix-cache","off"},{"--prefix-cache-retention","lru"}}) {
            std::vector<std::string> args{"decoder","MODEL"}; args.insert(args.end(),32768,"1");
            args.insert(args.end(),{"--generate","32"}); args.insert(args.end(),suffix.begin(),suffix.end());
            auto r=request; r.options=parse(args);
            const auto got=resolve(r,memory);
            expect(got.options.route_owned==base.options.route_owned && numerical(got)==numerical(base) && plan_ids(got.options)==base_ids,
                "operational prefix options preserve full numerical effective config and plan identities");
        }
        auto r=request; r.environment["STRIX_PREFIX_CACHE"]="0";
        const auto got=resolve(r,memory);
        expect(got.options.route_owned==base.options.route_owned && numerical(got)==numerical(base) && plan_ids(got.options)==base_ids,"environment off numerical identity");
        const auto set_measurement=[](const char* name,const char* value) {
#if defined(_WIN32)
            _putenv_s(name,value?value:"");
#else
            if(value) ::setenv(name,value,1); else ::unsetenv(name);
#endif
        };
        const auto saved=[](const char* name)->std::optional<std::string> {
            if(const char* value=std::getenv(name)) return std::string(value);
            return {};
        };
        const auto saved_out=saved("STRIX_BITCALC_LOGITS_OUT");
        const auto saved_force=saved("STRIX_BITCALC_FORCE_TOKENS");
        const auto saved_coop=saved("STRIX_QPF2K_GDN_COOP_LAUNCH");
        for (const char* value : {static_cast<const char*>(nullptr), "", "0", "1", "10", "2"}) {
            set_measurement("STRIX_QPF2K_GDN_COOP_LAUNCH",value);
            auto launch=request;
            launch.environment=requested_prefill_environment();
            const auto report=serve::parse_json(effective_config_json(resolve(launch,memory)));
            const auto* policy=report.find("gdn_core_launch");
            const bool present=std::getenv("STRIX_QPF2K_GDN_COOP_LAUNCH")!=nullptr;
            expect(policy && (present ? policy->find("requested")->as_string()==value : policy->find("requested")->is_null()) &&
                policy->find("effective")->as_string()==(value && value[0]=='1'?"cooperative":"plain") &&
                policy->find("read_once_per_process")->as_bool(),
                "real environment collector reports requested GDN launch and the cached reader's plain/cooperative policy");
            expect(present ? launch.environment.at("STRIX_QPF2K_GDN_COOP_LAUNCH")==value :
                !launch.environment.contains("STRIX_QPF2K_GDN_COOP_LAUNCH"), "GDN launch override is captured without normalization");
        }
        set_measurement("STRIX_QPF2K_GDN_COOP_LAUNCH",saved_coop?saved_coop->c_str():nullptr);
        set_measurement("STRIX_BITCALC_LOGITS_OUT",nullptr);
        set_measurement("STRIX_BITCALC_FORCE_TOKENS",nullptr);
        auto measured=request;
        measured.environment=requested_prefill_environment();
        const auto plain=resolve(measured,memory);
        set_measurement("STRIX_BITCALC_LOGITS_OUT","capture.bin");
        set_measurement("STRIX_BITCALC_FORCE_TOKENS","tokens.txt");
        measured.environment=requested_prefill_environment();
        const auto hooked=resolve(measured,memory);
        const auto plain_json=serve::parse_json(effective_config_json(plain));
        const auto hooked_json=serve::parse_json(effective_config_json(hooked));
        expect(!plain_json.find("measurement")->find("logits_capture")->as_bool() &&
            !plain_json.find("measurement")->find("forced_tokens")->as_bool() &&
            hooked_json.find("measurement")->find("logits_capture")->as_bool() &&
            hooked_json.find("measurement")->find("forced_tokens")->as_bool(),
            "real environment collector discloses both measurement controls");
        expect(effective_config_json(hooked).find("capture.bin")==std::string::npos &&
            effective_config_json(hooked).find("tokens.txt")==std::string::npos,
            "measurement inspection omits paths and file contents");
        auto normalized=hooked;
        normalized.environment.erase("STRIX_BITCALC_LOGITS_OUT");
        normalized.environment.erase("STRIX_BITCALC_FORCE_TOKENS");
        expect(effective_config_json(plain)==effective_config_json(normalized) &&
            plain.options.route_owned==hooked.options.route_owned &&
            plan_ids(plain.options)==plan_ids(hooked.options),
            "real collector measurement controls preserve every numerical route and arithmetic field and plan identity");
        set_measurement("STRIX_BITCALC_LOGITS_OUT",saved_out?saved_out->c_str():nullptr);
        set_measurement("STRIX_BITCALC_FORCE_TOKENS",saved_force?saved_force->c_str():nullptr);
    }
    // Model the runtime TU's pre-main environment before exercising the same
    // capture/resolver and baseline cleanup used by decode-token.
    {
        Environment startup{{"STRIX_QPF2K_DENSE", "0"}, {"STRIX_QPF2K_GDN", "1"}};
        const auto get = [&](const char* name) -> const char* {
            const auto it = startup.find(name);
            return it == startup.end() ? nullptr : it->second.c_str();
        };
        const auto put = [&](const char* name, const char* value) { return startup.emplace(name,value).second; };
        qpf2k::initialize_before_main(get,put);
        expect((qpf2k::pre_injected & qpf2k::bit(qpf2k::Item::sched)) != 0U &&
            qpf2k::defaulted_items == qpf2k::pre_injected, "runtime startup restores shared default provenance");
        for (const auto& entry : qpf2k::defaults)
            if (static_cast<unsigned>(entry.item) > static_cast<unsigned>(qpf2k::Item::ple_reclaim))
                expect(!startup.contains(entry.name), "non-decoder startup injects none of the new wrapper settings");
        auto captured = request;
        captured.environment = requested_prefill_environment(get);
        expect(!captured.environment.contains("STRIX_QPF2K_SCHED") &&
            !captured.environment.contains("STRIX_QPF2K_MOE_DEFAULTED"), "requested environment excludes startup injections");
        const auto resolved = resolve(captured,memory);
        expect(resolved.sources.at("STRIX_QPF2K_SCHED") == OptionSource::automatic &&
            resolved.sources.at("STRIX_QPF2K_MOE_DEFAULTED") == OptionSource::automatic,
            "pre-injected defaults resolve as automatic");
        expect(resolved.environment.at("STRIX_QPF2K_DENSE") == "0" &&
            resolved.sources.at("STRIX_QPF2K_DENSE") == OptionSource::explicit_value, "explicit startup environment stays explicit");
        expect(resolved.sources.at("STRIX_QPF2K_GDN") == OptionSource::explicit_value,
            "explicit value equal to the default stays explicit");
        captured.options.baseline = true;
        expect(resolve(captured,memory).reason == "baseline", "CLI baseline dominates startup defaults");
        qpf2k::undo_pre_injected([&](const char* name) { startup.erase(name); });
        expect(startup == Environment{{"STRIX_QPF2K_DENSE", "0"}, {"STRIX_QPF2K_GDN", "1"}} && qpf2k::baseline &&
            qpf2k::defaulted_items == 0U && qpf2k::pre_injected == 0U, "baseline unsets exactly its startup injections and clears provenance");
        qpf2k::baseline = false;
        for (const auto* disable : {"STRIX_QPF2K_DEFAULTS", "STRIX_BASELINE"}) {
            startup = {{disable, std::string_view(disable) == "STRIX_BASELINE" ? "1" : "0"}};
            qpf2k::initialize_before_main(get,put);
            expect(startup.size() == 1U && qpf2k::pre_injected == 0U && qpf2k::defaulted_items == 0U,
                "startup honors defaults-off and baseline environment");
            if (std::string_view(disable) == "STRIX_QPF2K_DEFAULTS") {
                captured = request; captured.environment = requested_prefill_environment(get);
                const auto disabled = resolve(captured,memory);
                expect(!disabled.environment.contains("STRIX_QPF2K_SCHED"), "resolver also honors defaults-off after startup");
            }
        }
    }
    long_context_admission_tests(request,model,info,manifest,capabilities);
    const auto automatic = resolve(request,memory);
    expect(automatic.admitted && automatic.reason == "qualified-prompt", "plain default admits the measured prompt recipe");
    const auto argv = effective_prefill_argv(automatic.options);
    expect(std::vector<std::string_view>(argv.begin(),argv.end()) ==
        std::vector<std::string_view>(qwen_contract::argv.begin(),qwen_contract::argv.end()), "plain default matches normalized argv contract");
    for (const auto entry : qwen_contract::env) {
        const auto equal = entry.find('=');
        expect(automatic.environment.at(std::string(entry.substr(0,equal))) == entry.substr(equal+1U), "active environment matches contract");
    }
    for (const auto& entry : qpf2k::defaults) {
        expect(automatic.environment.at(entry.name) == entry.value && automatic.sources.at(entry.name) == OptionSource::automatic,
            "existing and new defaults retain automatic provenance");
    }
    for (const auto* name : excluded_inactive_env)
        expect(!automatic.environment.contains(name), "inactive environment is excluded from actual resolution");
    expect(!automatic.options.mtp.enabled && !automatic.options.prepared_eager && !automatic.options.moe_shared_overlap &&
        !automatic.options.gdn_islands, "prefill organization does not enable speculation, prepared overlap or islands");
    expect(automatic.options.known_prefix_tail_rows.empty() && automatic.options.known_prefix_ladder_tail_rows ==
        std::vector<std::uint32_t>({1024U,256U,64U,16U}), "default SCHED (bit 8) retains configured ladder and prunes unused owners");
    const auto json = effective_config_json(automatic);
    expect(json.starts_with("{\"rotary\":{\"rope_mode\":\"native\",\"factor\":0,"),
        "effective-config inspection golden deliberately includes native rotary status; numerical identity unchanged");
    if (argc == 2 && std::string_view(arguments[1]) == "--dump-fixture-config") { std::cout << json; return 0; }
    expect(json == effective_config_json(resolve(request,memory)) && json.find("MODEL") == std::string::npos &&
        json.find("\"source\":\"automatic\"") != std::string::npos && !automatic.host_compilation.empty(), "dry-run JSON is deterministic, path-free and reports provenance");
    // Compile organization/buffer requests from each route, not only booleans.
    auto manual_args = std::vector<std::string>{"decoder","MODEL","1"};
    for (const auto arg : qwen_contract::argv) manual_args.emplace_back(arg);
    auto manual = parse(std::move(manual_args));
    manual.tokens = request.options.tokens;
    const auto expected_compiler = prompt_compiler_requests(manual,model,manifest);
    const auto actual_compiler = prompt_compiler_requests(automatic.options,model,manifest);
    expect(actual_compiler.binding.wave8_q8_gdn_input && actual_compiler.binding.wave8_q8_gdn_output &&
        actual_compiler.binding.wave8_q8_qsa_output && actual_compiler.binding.wave8_q8_lm_head &&
        actual_compiler.binding.prefetch_q8_lm_head && actual_compiler.binding.q51_slot_parallel_down &&
        actual_compiler.binding.batched_moe_activation && actual_compiler.binding.a8_down_q51 &&
        actual_compiler.binding.a8_down_q80 && actual_compiler.binding.q5k_a8g32_gate_up &&
        actual_compiler.binding.dense_a8 && actual_compiler.binding.lane_chain_hyper_output_eligible_steps.size() == 97U &&
        actual_compiler.binding.qsa_input_prefetch_layers.size() == 12U &&
        actual_compiler.binding.shared_expert_prefetch_layers.size() == 48U,
        "host compiler receives the complete measured projection, arithmetic and prefetch organization");
    const auto expected_buffers = build_serial_execution_buffer_plan(model,expected_compiler.buffers);
    const auto actual_buffers = build_serial_execution_buffer_plan(model,actual_compiler.buffers);
    expect(expected_buffers.peak_internal_bytes == actual_buffers.peak_internal_bytes &&
        expected_buffers.mlp_intermediate_bytes == actual_buffers.mlp_intermediate_bytes &&
        expected_buffers.gdn_workspace_bytes == actual_buffers.gdn_workspace_bytes &&
        expected_buffers.hyper_workspace_bytes == actual_buffers.hyper_workspace_bytes,
        "normalized recipe and automatic organization compile identical buffer requirements");
    expect(actual_compiler.binding.wave8_q8_gdn_input == expected_compiler.binding.wave8_q8_gdn_input &&
        actual_compiler.binding.wave8_q8_gdn_output == expected_compiler.binding.wave8_q8_gdn_output &&
        actual_compiler.binding.wave8_q8_qsa_output == expected_compiler.binding.wave8_q8_qsa_output &&
        actual_compiler.binding.q51_slot_parallel_down == expected_compiler.binding.q51_slot_parallel_down &&
        actual_compiler.binding.dense_a8 == expected_compiler.binding.dense_a8 &&
        actual_compiler.catalog.numerical_profile == expected_compiler.catalog.numerical_profile,
        "compiler binding and arithmetic organization match the explicit recipe");
    for (const auto rows : {1024U,4096U}) {
        auto short_request = request; short_request.options.tokens.resize(rows);
        const auto short_config = resolve(short_request,memory);
        expect(short_config.admitted && short_config.options.known_prefix_prefill_rows == rows,
            "short prompt chooses its measured T1024/T4096 class");
    }
    auto four_k = request; four_k.options.tokens.resize(4096U);
    auto four_k_receipt = argv;
    four_k_receipt[1] = "8192";
    four_k_receipt.erase(std::find(four_k_receipt.begin(),four_k_receipt.end(),"--unqualified-context"));
    const auto tile_arg = std::find(four_k_receipt.begin(),four_k_receipt.end(),"--known-prefix-prefill-rows");
    *std::next(tile_arg) = "4096";
    expect(effective_prefill_argv(resolve(four_k,memory).options) == four_k_receipt,
        "short_prompt_receipt_argv: 4K preserves the exact historical receipt argv order");
    // Compare all emitted organization/arithmetic/context/budget/tail decisions
    // with the frozen pre-policy resolver, rather than a second call to live logic.
    const auto historical_equal = [&](const RequestedOptions& input, const EffectiveConfig& actual) {
        auto old = input.options;
        const auto expected = resolve_historical_9b1d1c3e(old,model,info,manifest,
            validated_model_content_identity,true,input.permit_generation_defaults);
        auto normalized = actual.options;
        normalized.baseline = old.baseline; // The policy marker itself is new.
        const bool master_off = input.options.baseline || env_is(input.environment,"STRIX_BASELINE","1") ||
            (input.environment.contains("STRIX_QPF2K_DEFAULTS") && input.environment.at("STRIX_QPF2K_DEFAULTS").starts_with('0'));
        if (master_off) {
            expect(actual.prefill_option_mask == resolve_prefill_options([&](const char* name) -> const char* {
                const auto it = input.environment.find(name);
                return it == input.environment.end() ? nullptr : it->second.c_str();
            }, old.prefill_exact_fp16, old.prefill_disabled) && actual.options.prefill_disabled == old.prefill_disabled,
                "bundle masters preserve governed arithmetic controls and the public ON identity");
        }
        expect(!actual.admitted && effective_prefill_argv(normalized) == effective_prefill_argv(old) &&
            normalized.known_prefix_tail_rows == old.known_prefix_tail_rows &&
            normalized.known_prefix_ladder_tail_rows == old.known_prefix_ladder_tail_rows &&
            actual.organization.selected == expected.selected && actual.organization.reason == expected.reason &&
            actual.organization.automatic_applied == expected.automatic_applied,
            "complete fallback resolution equals the frozen historical resolver");
        Environment expected_env = input.environment;
        if (master_off) expected_env["STRIX_QPF2K_COMPACT_IMAGES"] = "0";
        if (!master_off && input.options.eager_code_load && !env_is(input.environment,"STRIX_EAGER_CODE_LOAD","0") &&
            !expected_env.contains("HIP_ENABLE_DEFERRED_LOADING")) expected_env["HIP_ENABLE_DEFERRED_LOADING"] = "0";
        const bool baseline = input.options.baseline || env_is(input.environment,"STRIX_BASELINE","1");
        if (baseline) {
            for (const auto& entry : qpf2k::defaults) expected_env[entry.name] = "0";
            expected_env["STRIX_QPF2K_DEFAULTS"] = "0";
        } else {
            qpf2k::initialize([&](const char* name) -> const char* {
                const auto it = expected_env.find(name);
                return it == expected_env.end() ? nullptr : it->second.c_str();
            }, [&](const char* name, const char* value) { return expected_env.emplace(name,value).second; },
                qpf2k::historical_items | (single_session_mtp_route(input.options, input.environment, manifest,
                    validated_model_content_identity, capabilities) ? qpf2k::bit(qpf2k::Item::dispatch_fuse) : 0U));
        }
        expect(actual.environment == expected_env, "fallback environment equals historical environment plus explicit controls");
    };
    for (const auto& entry : qpf2k::defaults) {
        auto off = request; off.environment[entry.name] = "0";
        const auto config = resolve(off,memory);
        if (qpf2k::exact_prefill_items & qpf2k::bit(entry.item)) {
            expect(config.admitted && effective_prefill_argv(config.options) == effective_prefill_argv(automatic.options),
                "an exact prefill off switch keeps the admitted route");
            for (const auto& other : qpf2k::defaults)
                if (other.item != entry.item)
                    expect(config.environment.at(other.name) == other.value && config.sources.at(other.name) == OptionSource::automatic,
                        "an exact prefill off switch disables only that item");
            expect(effective_config_json(config).find(std::string("\"") + entry.name +
                "\":{\"value\":\"0\",\"source\":\"explicit\"}") != std::string::npos,
                "effective config reports the explicit per-item off switch");
        } else historical_equal(off,config);
        expect(config.environment.at(entry.name) == "0" && config.sources.at(entry.name) == OptionSource::explicit_value,
            "every feature off switch preserves explicit zero");
    }
    for (const auto* flag : {"--known-prefix-window", "--known-prefix-full-rows", "--known-prefix-expert-union",
            "--known-window-router-rows", "--known-prefix-prefill-rows"}) {
        auto off = request;
        off.options = parse({"decoder","MODEL","1","--generate","32",std::string(flag)+"=0"});
        off.options.tokens = request.options.tokens;
        historical_equal(off,resolve(off,memory));
    }
    for (const auto* name : {"STRIX_KNOWN_PREFIX_WINDOW", "STRIX_KNOWN_PREFIX_FULL_ROWS",
            "STRIX_KNOWN_PREFIX_EXPERT_UNION", "STRIX_KNOWN_WINDOW_ROUTER_ROWS", "STRIX_KNOWN_PREFIX_PREFILL_ROWS"}) {
        auto off = request; off.environment[name] = "0";
        historical_equal(off,resolve(off,memory));
    }
    for (const auto* name : {"STRIX_QPF2K_DEFAULTS", "STRIX_PREFILL_F16"}) {
        auto off = request; off.environment[name] = "0";
        historical_equal(off,resolve(off,memory));
    }
    auto dense_cli = request; dense_cli.options.prefill_numerical_profile = dense_a8_profile;
    expect(!numerical_profile_pair(resolve(dense_cli,memory).options).split(), "explicit dense-A8 prefill wins");
    auto baseline = request; baseline.options.baseline = true;
    baseline.environment["STRIX_PREFILL_PIPE"] = "255";
    const auto base = resolve(baseline,memory);
    historical_equal(baseline,base);
    expect(numerical_profile_name(base.options.numerical_profile) == qwen_contract::baseline_decode &&
        base.options.prepared_eager == qwen_contract::baseline_prepared &&
        base.options.gdn_islands && base.options.moe_shared_overlap && base.options.context_tokens == 8192U &&
        !base.options.unqualified_context && !base.options.prefill_numerical_profile,
        "baseline retains historical exact-v5 prepared overlap islands and implicit 8192 context");
    baseline.options.baseline = false; baseline.environment["STRIX_BASELINE"] = "1";
    historical_equal(baseline,resolve(baseline,memory));
    expect(effective_prefill_argv(resolve(baseline,memory).options) == effective_prefill_argv(base.options), "baseline CLI and env are equivalent");
    for (const auto context : {8192U,65536U}) {
        auto explicit_context = request;
        explicit_context.options.context_explicit = true;
        explicit_context.options.context_tokens = context;
        explicit_context.options.unqualified_context = context > 8192U;
        explicit_context.options.supplied_options.insert("--context");
        if (context > 8192U) explicit_context.options.supplied_options.insert("--unqualified-context");
        const auto config = resolve(explicit_context,memory);
        expect(config.admitted && config.options.context_tokens == context && config.sources.at("--context") == OptionSource::explicit_value &&
            (context == 8192U || config.sources.at("--unqualified-context") == OptionSource::explicit_value), "explicit context is preserved without automatic provenance");
        const auto explicit_notice = effective_config_notice(config);
        expect(context == 8192U ? explicit_notice.find("unqualified-context") == std::string::npos :
            explicit_notice.find("beyond qualified 8192; explicit") != std::string::npos,
            "notice labels explicit long context without inventing automatic acknowledgement");
        explicit_context.options.baseline = true;
        historical_equal(explicit_context,resolve(explicit_context,memory));
    }
    auto unacknowledged = request;
    unacknowledged.options.context_tokens = 65536U; unacknowledged.options.context_explicit = true;
    expect(!resolve(unacknowledged,memory).admitted, "explicit unqualified context still needs acknowledgement");
    const auto auto_notice = effective_config_notice(automatic);
    expect(auto_notice.find("automatic T8192 context=33280") != std::string::npos &&
        auto_notice.find("unqualified-context (beyond qualified 8192; automatic)") != std::string::npos &&
        std::count(auto_notice.begin(),auto_notice.end(),'\n') == 1 &&
        automatic.sources.at("--unqualified-context") == OptionSource::automatic,
        "automatic long context emits one qualified-limit notice and records automatic provenance");
    auto exported = request; exported.options.tokens.resize(1024U); exported.options.route_owned = true; exported.permit_generation_defaults = false;
    const auto export_config = resolve(exported,memory);
    historical_equal(exported,export_config);
    expect(!export_config.options.prepared_eager && !export_config.options.moe_shared_overlap && !export_config.options.gdn_islands,
        "target export excludes ordinary-generation submission defaults independently of route ownership");
    for (unsigned route = 0; route != 4; ++route) {
        auto owned = request;
        if (route == 0 || route == 1) owned.options = parse(mtp_fixture());
        if (route == 1) {
            auto& m = owned.options.mtp;
            m.window_rows = 8U; m.window_rows_set = {2U,4U,8U}; m.candidate_depth = 7U;
            m.window_prefix = m.draft_chain = true;
            m.window_arena_bytes = m.window_capture_bytes = m.window_checkpoint_bytes = m.window_total_bytes = 1024U;
            m.window_history_intervals = 1U;
            m.cohort.push_back({{1U,2U,3U},"cohort.golden",std::string(64,'1')});
            owned.options.wave8_q8_gdn_input = owned.options.wave8_q8_gdn_output = true;
        }
        if (route == 2) owned.options = parse({"decoder","MODEL","--generate","32","--serve-prefix-u32le","prefix.u32",
            "--serve-memory-budget-bytes","8192","--serve-request-budget-bytes","1024","--serve-headroom-bytes","1024"});
        if (route == 3) {
            owned.options = manual;
            owned.options.prefill_disabled = 1U; // fast HC MTP requires explicit OFF until qualified
            owned.options.mtp = parse(mtp_fixture()).mtp;
            auto& m = owned.options.mtp;
            m.window_rows = 8U; m.window_rows_set = {2U,4U,8U}; m.candidate_depth = 7U;
            m.window_prefix = m.draft_chain = true;
            m.window_arena_bytes = m.window_capture_bytes = m.window_checkpoint_bytes = m.window_total_bytes = 1024U;
            m.window_history_intervals = 1U;
            owned.options.http_serve = true;
            validate_http_serve(owned.options);
        }
        owned.environment["STRIX_PREFILL_PIPE"] = "7";
        owned.environment["STRIX_LM_HEAD_COHORT_LEAN"] = "0";
        const auto config = resolve(owned,memory);
        historical_equal(owned,config);
        expect(config.environment.at("STRIX_PREFILL_PIPE") == "7" && config.environment.at("STRIX_LM_HEAD_COHORT_LEAN") == "0",
            "declined serving/MTP/cohort preserves explicit wrapper overrides");
        owned.environment.clear();
        const auto absent = resolve(owned,memory);
        historical_equal(owned,absent);
        expect(!absent.environment.contains("STRIX_PREFILL_PIPE") && !absent.environment.contains("STRIX_LM_HEAD_COHORT_LEAN"),
            "declined serving/MTP/cohort injects no wrapper settings");
        // Only the single-session MTP route (route 0) takes the window dispatch fusion; cohort, serving and HTTP do not.
        expect(route == 0 ? absent.environment.contains("STRIX_DISPATCH_FUSE") &&
                absent.environment.at("STRIX_DISPATCH_FUSE") == "1" &&
                absent.sources.at("STRIX_DISPATCH_FUSE") == OptionSource::automatic
            : !absent.environment.contains("STRIX_DISPATCH_FUSE"), "dispatch fusion is injected on the single-session MTP route only");
        if (route == 0) {
            auto off = owned; off.environment["STRIX_DISPATCH_FUSE"] = "0";
            const auto kept = resolve(off,memory);
            expect(kept.environment.at("STRIX_DISPATCH_FUSE") == "0" &&
                kept.sources.at("STRIX_DISPATCH_FUSE") == OptionSource::explicit_value,
                "STRIX_DISPATCH_FUSE=0 keeps the single-session MTP route unfused");
            expect(effective_config_notice(absent).find("Qwen MTP route: STRIX_DISPATCH_FUSE=1 (automatic") == 0U &&
                effective_config_notice(kept).find("STRIX_DISPATCH_FUSE") == std::string::npos,
                "the startup notice reports the automatic MTP-route dispatch fusion only when injected");
            // Every other owner of the MTP request keeps the historical environment: route options beyond --mtp*,
            // low-level or known-prefix options, recording, final-state validation and diagnostic controls.
            const auto unfused = [&](const char* why, auto&& change) {
                auto other = owned;
                change(other);
                const auto resolved = resolve(other, memory);
                historical_equal(other, resolved);
                expect(!resolved.environment.contains("STRIX_DISPATCH_FUSE"), why);
            };
            unfused("an option beyond --mtp keeps the MTP route unfused", [](auto& r) { r.options.route_owned_beyond_mtp = true; });
            unfused("STRIX_QRTX_STEP_DUMP keeps the MTP route unfused", [](auto& r) { r.environment["STRIX_QRTX_STEP_DUMP"] = "1"; });
            unfused("STRIX_RTX_EXPORT_STATE keeps the MTP route unfused", [](auto& r) { r.environment["STRIX_RTX_EXPORT_STATE"] = "1"; });
            // Options that need their own companions to resolve at all: the route predicate alone.
            const auto route_with = [&](const char* why, auto&& change) {
                auto other = owned.options;
                change(other);
                expect(!single_session_mtp_route(other, owned.environment, manifest, validated_model_content_identity, capabilities), why);
            };
            // The documented MTP command takes the item: its window organization owns only the MTP route.
            auto documented = owned;
            documented.options = parse(documented_mtp_argv());
            expect(documented.options.any_low_level_flag && documented.options.known_prefix_window &&
                documented.options.route_owned && !documented.options.route_owned_beyond_mtp,
                "the documented MTP window organization owns only the MTP route");
            const auto fused = resolve(documented, memory);
            expect(fused.environment.contains("STRIX_DISPATCH_FUSE") && fused.environment.at("STRIX_DISPATCH_FUSE") == "1" &&
                fused.sources.at("STRIX_DISPATCH_FUSE") == OptionSource::automatic,
                "the documented MTP command takes the automatic dispatch fusion");
            auto documented_off = documented; documented_off.environment["STRIX_DISPATCH_FUSE"] = "0";
            const auto documented_kept = resolve(documented_off, memory);
            expect(documented_kept.environment.at("STRIX_DISPATCH_FUSE") == "0" &&
                documented_kept.sources.at("STRIX_DISPATCH_FUSE") == OptionSource::explicit_value,
                "STRIX_DISPATCH_FUSE=0 keeps the documented MTP command unfused");
            // One more low-level option outside the documented organization owns its own route.
            auto extra_argv = documented_mtp_argv();
            extra_argv.insert(extra_argv.end(), {"--coalesce-gap", "0"});
            auto extra = documented;
            extra.options = parse(extra_argv);
            expect(extra.options.route_owned_beyond_mtp, "a low-level option outside the MTP organization owns its own route");
            expect(!resolve(extra, memory).environment.contains("STRIX_DISPATCH_FUSE"),
                "a low-level option outside the MTP organization keeps the MTP route unfused");
            // One more --mtp* option outside the documented groups 3-4 owns its own route: the allow-list is explicit,
            // not a --mtp prefix. The golden dump (the documented benchmarking step, in place of --mtp-golden and
            // --mtp-golden-sha256) must write its oracle from an unfused run.
            const auto without_golden = [](std::vector<std::string> args) {
                for (const char* key : {"--mtp-golden", "--mtp-golden-sha256"}) {
                    const auto at = std::find(args.begin(), args.end(), std::string(key));
                    if (at != args.end()) args.erase(at, at + 2);
                }
                return args;
            };
            const std::vector<std::pair<const char*, std::vector<std::string>>> extra_mtp = {
                {"--mtp-golden-dump keeps the documented MTP command unfused", {"--mtp-golden-dump", "prompt.golden"}},
                {"--mtp-ngram-draft=off keeps the documented MTP command unfused", {"--mtp-ngram-draft=off"}},
                {"--mtp-calibration-rows keeps the documented MTP command unfused", {"--mtp-calibration-rows", "8"}},
            };
            for (const auto& [why, flag] : extra_mtp) {
                auto args = flag.front() == std::string("--mtp-golden-dump") ? without_golden(documented_mtp_argv()) :
                    documented_mtp_argv();
                args.insert(args.end(), flag.begin(), flag.end());
                auto one_more = documented;
                try {
                    one_more.options = parse(args);
                } catch (const std::exception& error) {
                    std::cerr << why << ": " << error.what() << '\n';
                    expect(false, why);
                    continue;
                }
                expect(one_more.options.route_owned_beyond_mtp, why);
                expect(!single_session_mtp_route(one_more.options, one_more.environment, manifest,
                    validated_model_content_identity, capabilities), why);
                expect(!resolve(one_more, memory).environment.contains("STRIX_DISPATCH_FUSE"), why);
            }
            // The exact documented argv still takes the injection (above); the predicate also refuses the oracle- and
            // state-producing, sampling and n-gram modes when they are set without their own flag.
            const auto documented_route_with = [&](const char* why, auto&& change) {
                auto other = documented.options;
                expect(single_session_mtp_route(other, documented.environment, manifest, validated_model_content_identity,
                    capabilities), "the documented MTP command is the single-session MTP route");
                change(other);
                expect(!single_session_mtp_route(other, documented.environment, manifest, validated_model_content_identity,
                    capabilities), why);
            };
            documented_route_with("a golden dump keeps the MTP route unfused", [](auto& o) { o.mtp.golden_dump = "prompt.golden"; });
            documented_route_with("the metadata-only diagnostic keeps the MTP route unfused", [](auto& o) { o.mtp.metadata_only = true; });
            documented_route_with("RTX state import keeps the MTP route unfused", [](auto& o) { o.rtx_state_import = true; });
            documented_route_with("sampling keeps the MTP route unfused", [](auto& o) { o.mtp.sample_options = true; o.mtp.sample_temperature = 0.7; });
            documented_route_with("the adaptive sample schedule keeps the MTP route unfused", [](auto& o) { o.mtp.sample_timing_schedule = true; });
            documented_route_with("an explicit n-gram draft keeps the MTP route unfused", [](auto& o) { o.mtp.ngram_draft = true; });
            documented_route_with("n-gram off keeps the MTP route unfused", [](auto& o) { o.mtp.ngram_off = true; });
            documented_route_with("the n-gram copy drafter keeps the MTP route unfused", [](auto& o) { o.mtp.ngram_copy_edit = true; });
            documented_route_with("copy-wide rows keep the MTP route unfused", [](auto& o) { o.mtp.copy_wide_rows = {16U}; });
            documented_route_with("a session save keeps the MTP route unfused", [](auto& o) { o.session_save = "s.bin"; });
            documented_route_with("a session load keeps the MTP route unfused", [](auto& o) { o.session_load = "s.bin"; });
            documented_route_with("a prefix-cache restore keeps the MTP route unfused", [](auto& o) { o.prefix_cache_dir = "cache"; });
            for (const char* key : {"--mtp-golden-dump", "--mtp-metadata-only", "--mtp-sample-temperature", "--mtp-sample-seed",
                     "--mtp-ngram-draft", "--mtp-copy-wide-rows", "--mtp-calibration-rows", "--mtp-cohort-request",
                     "--mtp-draft-profile", "--mtp-draft-exact", "--mtp-bundle", "--mtp-prefill-gate", "--mtp-k"})
                expect(!mtp_window_organization_option(key) && !mtp_window_organization_option(std::string(key) + "=1"),
                    "only the documented group 3-4 --mtp* options are in the MTP window organization");
            route_with("raw recording keeps the MTP route unfused", [](auto& o) { o.raw_recording_path = "raw.bin"; });
            route_with("final-state validation keeps the MTP route unfused", [](auto& o) { o.validation_final_state_path = "final.bin"; });
            expect(!owned.options.route_owned_beyond_mtp && owned.options.route_owned,
                "the --mtp* options own the route without owning one beyond MTP");
            auto qualified = capabilities;
            expect(single_session_mtp_route(owned.options, owned.environment, manifest, validated_model_content_identity, qualified),
                "the fixture MTP request is the single-session MTP route");
            qualified.target_qualified = false;
            expect(!single_session_mtp_route(owned.options, owned.environment, manifest, validated_model_content_identity, qualified),
                "an unqualified target keeps the MTP route unfused");
            expect(!single_session_mtp_route(owned.options, owned.environment, manifest, "other-content", capabilities),
                "an unqualified checkpoint keeps the MTP route unfused");
        }
        if(route==3) {
            expect(!absent.environment.contains("STRIX_DISPATCH_FUSE") &&
                !absent.environment.contains("STRIX_WINDOW_INJECT_NORM") &&
                !absent.environment.contains("STRIX_WINDOW_COMMIT_COMBINE"),
                "default HTTP route does not inject window-only folds");
            const auto policy=resolve_mtp_selection_route(absent.options.http_selection_mtp,true,true);
            expect(policy.mtp && !policy.fallback,"default HTTP policy uses an admitted serial-equivalent owner");
        }
        for (const bool explicit_t16k : {false,true}) {
            auto blocked = owned;
            if (explicit_t16k) blocked.environment["STRIX_QPF2K_T16K"] = "1";
            auto no_setup = memory;
            unsigned estimates = 0;
            no_setup.estimate = [&](const Options&) { ++estimates; return PrefillRequirements{}; };
            const auto excluded = resolve_effective_config(blocked,model,info,manifest,
                validated_model_content_identity,capabilities,no_setup);
            expect(!excluded.admitted && excluded.reason == "explicit-route" && !excluded.t16k_policy_source &&
                excluded.tile_fallback.empty() && estimates == 0U,
                "MTP, HTTP, serving prefixes and cohort never attempt automatic or override T16384 setup");
        }
    }
    auto dense_baseline = baseline;
    dense_baseline.options.numerical_profile = dense_a8_profile;
    dense_baseline.options.numerical_profile_explicit = true;
    const auto explicit_dense_base = resolve(dense_baseline,memory);
    historical_equal(dense_baseline,explicit_dense_base);
    auto invalid_baseline = explicit_dense_base.options;
    invalid_baseline.prepared_eager = false;
    bool rejected_baseline = false;
    try {
        auto org = explicit_dense_base.organization;
        apply_numerical_profile(invalid_baseline,model,manifest,validated_model_content_identity,true,org);
    } catch (const std::invalid_argument&) { rejected_baseline = true; }
    expect(rejected_baseline, "baseline has no exemption from historical dense-A8 submission prerequisites");
    auto missing = memory; missing.available.reset();
    expect(resolve(request,missing).reason == "memory-observation-unavailable", "missing memory observation cannot assume capacity");
    auto small = memory; small.available = 16U*gib_bytes;
    const auto declined = resolve(request,small);
    historical_equal(request,declined);
    const auto notice = effective_config_notice(declined);
    expect(!declined.admitted && !declined.options.known_prefix_window &&
        effective_config_notice(declined).starts_with("Qwen auto-prefill declined: insufficient-memory;") &&
        notice.find("using serial prefill") == std::string::npos && !declined.error.empty() &&
        std::count(notice.begin(),notice.end(),'\n') == 1,
        "shortage names the admission reason and refuses a prompt exceeding serial context");
    auto overflow = request; overflow.options.generate_tokens = UINT64_MAX;
    expect(resolve(overflow,memory).reason == "prompt-exceeds-model-context", "implicit context rejects overflow without adding");
    auto short_serial = request; short_serial.options.tokens.resize(4096U);
    const auto serial = resolve(short_serial,small);
    expect(!serial.admitted && serial.error.empty() &&
        effective_config_notice(serial).find("using serial prefill") != std::string::npos,
        "serial_fallback_fits: short prompts retain the historical serial fallback");
    require_effective_prefill_execution(serial);
    auto diagnostic = request; diagnostic.environment["STRIX_RTX_EXPORT_STATE"] = "1";
    expect(!resolve(diagnostic,memory).admitted, "explicit state-export control owns its existing execution route");
    auto no_target = capabilities; no_target.target_qualified = false;
    expect(!resolve_w7_case(request,model,info,manifest,validated_model_content_identity,no_target,memory).admitted,
        "unqualified target declines before estimating");
    auto f16_failure = memory;
    f16_failure.estimate = [&](const Options& candidate) {
        if (candidate.prefill_numerical_profile) throw std::invalid_argument("unavailable fast operands");
        return memory.estimate(candidate);
    };
    const auto dense_fallback = resolve(request,f16_failure);
    expect(dense_fallback.admitted && !numerical_profile_pair(dense_fallback.options).split() &&
        effective_config_notice(dense_fallback).find("fp16-plan-not-admitted") != std::string::npos,
        "automatic FP16 admission failure retries the complete qualified dense-A8 recipe");
    auto saturated = memory; saturated.gtt_used = *saturated.gtt_limit;
    expect(resolve(request,saturated).reason == "insufficient-memory", "device GTT remaining capacity is capped separately");
    for (const bool gsq : {true,false,true}) {
        qpf2k::defaulted_items = (1U << qpf2k::defaults.size()) - 1U;
        const qpf2k::ModelScope scope(gsq);
        for (const auto& entry : qpf2k::defaults)
            expect(qpf2k::suppressed(entry.item) == gsq, "owner alternation filters every newly injected feature after parsing");
    }
    qpf2k::defaulted_items = 0U;
    auto explicit_route = request; explicit_route.options = manual;
    const auto untouched = resolve(explicit_route,memory);
    expect(!untouched.admitted && effective_prefill_argv(untouched.options) == effective_prefill_argv(manual),
        "explicit route flags are untouched");
    const auto absent_build = resolve_w7_case(request,model,info,manifest,validated_model_content_identity,
        {false,true,true},memory);
    expect(!absent_build.admitted, "missing A8 capability declines");
    {
        Options route;
        route.http_serve=true;
        route.mtp.enabled=true; route.mtp.draft_chain=true;
        route.known_prefix_window=true; route.known_prefix_full_rows=true;
        route.known_prefix_prefill_rows=1024; route.generate_tokens=32;
        Environment env{{"HOME",std::filesystem::temp_directory_path().string()},{"STRIX_SESSION_DISK","0"}};
        for (const bool mtp : {false,true}) for (const unsigned slots : {0U,1U,4U}) {
            auto production=route; production.mtp.enabled=mtp; production.http_batch=slots;
            resolve_session_defaults(production,model,env,128*gib_bytes);
            EffectiveConfig c; c.options=production;
            expect(!production.session_management_requested && !production.session_cache &&
                effective_config_notice(c)=="session cache: off (route not yet GPU-qualified: " +
                    std::string(session_serving_route(production)) + ")\nprefix cache: off (not-serving)\n", "unqualified serving routes fail closed");
        }
        expect(!session_route_gpu_qualified("cli-target") && !session_route_gpu_qualified("unknown"), "serving-only qualification table");
        for (const auto& entry : session_route_qualifications)
            expect(!entry.gpu_qualified, "no serving route qualified today");
        auto explicit_cli=parse({"decoder","MODEL","1","--session-cache","on","--session-load","checkpoint"});
        resolve_session_defaults(explicit_cli,model,env,128*gib_bytes);
        EffectiveConfig cli_config; cli_config.options=explicit_cli;
        const auto cli_json=serve::parse_json(effective_config_json(cli_config));
        expect(!explicit_cli.session_cache && explicit_cli.session_load=="checkpoint" &&
            explicit_cli.session_off_reason=="cli-save-load-explicit" &&
            effective_config_notice(cli_config)=="session cache: off (cli-save-load-explicit)\n" &&
            cli_json.find("session_cache")->find("policy")->as_string()=="qualified-serving-routes-only" &&
            cli_json.find("session_cache")->find("cli_operations")->as_string()=="explicit-save-load",
            "CLI cache switch does not enable automatic save/load");
        route.supplied_options.insert("--session-mtp-fixed");
        route.supplied_options.insert("--session-tiled-suffix");
        auto live=route; live.session_live_control=true;
        resolve_session_defaults(live,model,env,128*gib_bytes);
        expect(live.session_management_requested,"explicit live control qualification remains accessible");
        for (const auto* flag : {"--session-cache", "--session-disk", "--session-dir", "--session-ram-bytes",
                "--session-disk-bytes", "--session-max-parked", "--session-capture-host-bytes", "--prefix-batch-cert"}) {
            auto parsed=parse({"decoder","MODEL","1",flag,
                std::string_view(flag)=="--prefix-batch-cert"?"1":std::string_view(flag)=="--session-cache" || std::string_view(flag)=="--session-disk" ? "off" : "42"});
            expect(!parsed.route_owned && !parsed.session_options_requested,"storage policy cannot change numerical route");
            auto incumbent=parse({"decoder","MODEL","1"});
            for (auto* candidate : {&parsed,&incumbent}) {
                candidate->tokens.resize(32768,1); candidate->known_prefix_window=true;
                candidate->known_prefix_full_rows=true; candidate->known_prefix_prefill_explicit=true;
                candidate->known_prefix_prefill_rows=8192;
                apply_default_known_prefill_tails(*candidate,"8");
            }
            expect(parsed.known_prefix_tail_rows==incumbent.known_prefix_tail_rows &&
                parsed.known_prefix_ladder_tail_rows==incumbent.known_prefix_ladder_tail_rows,
                "storage policy cannot affect tail pruning");
        }
        for (const unsigned slots : {1U,4U}) for (const bool enabled : {false,true}) {
            auto ordinary=route; ordinary.http_batch=slots; ordinary.session_management_requested=enabled;
            ordinary.http_sampling="sample";
            expect(ordinary_http_batch_slots(ordinary)==slots && ordinary.http_sampling=="sample",
                "unnamed scheduling and sampling survive session admission including batch 1");
        }
        auto on=route; resolve_session_defaults(on,model,env,128*gib_bytes);
        expect(on.session_cache && on.session_management_requested && !on.session_disk &&
            on.session_ram_bytes==8*gib_bytes && on.session_max_parked==8 &&
            on.session_dir==(std::filesystem::temp_directory_path()/".cache/strix-engine/sessions").string(),
            "serving session defaults and RAM-only switch");
        auto cli=parse({"decoder","MODEL","1","--session-cache","off","--session-disk","off","--session-disk-bytes","42"});
        expect(!cli.session_cache && !cli.session_disk && cli.session_disk_bytes==42,"session CLI switches parse");
        auto disabled=route; auto env_off=env; env_off["STRIX_SESSION_CACHE"]="0";
        resolve_session_defaults(disabled,model,env_off,128*gib_bytes);
        expect(!disabled.session_cache && !disabled.session_management_requested && disabled.session_off_reason=="disabled","environment cache off");
        for (const bool baseline_cli : {false,true}) {
            auto base_route=route; auto base_env=env;
            if (baseline_cli) base_route.baseline=true; else base_env["STRIX_BASELINE"]="1";
            resolve_session_defaults(base_route,model,base_env,128*gib_bytes);
            expect(!base_route.session_cache && base_route.session_off_reason=="baseline","baseline disables sessions");
        }
        auto disk_cli=parse({"decoder","MODEL","1","--session-disk","off"});
        auto disk_cli_route=route; disk_cli_route.session_disk=disk_cli.session_disk;
        disk_cli_route.supplied_options.insert(disk_cli.supplied_options.begin(),disk_cli.supplied_options.end());
        auto disk_cli_env=env; disk_cli_env["STRIX_SESSION_DISK"]="1";
        resolve_session_defaults(disk_cli_route,model,disk_cli_env,128*gib_bytes);
        expect(disk_cli_route.session_cache && !disk_cli_route.session_disk,"CLI disk off overrides env on");
        auto cli_off_route=route; cli_off_route.session_cache=cli.session_cache;
        cli_off_route.supplied_options=cli.supplied_options;
        resolve_session_defaults(cli_off_route,model,env,128*gib_bytes);
        expect(!cli_off_route.session_management_requested,"CLI cache off disables serving");
        auto resolver_off=request; resolver_off.options.http_serve=true;
        resolver_off.environment["STRIX_SESSION_CACHE"]="0";
        const auto resolved_off=resolve(resolver_off,memory);
        const auto off_json=serve::parse_json(effective_config_json(resolved_off));
        expect(!off_json.find("session_cache")->find("enabled")->as_bool() &&
            off_json.find("session_cache")->find("reason")->as_string()=="disabled", "full resolver JSON environment off");
        auto legacy_prefix=route; legacy_prefix.prefix_cache_host_bytes=1U<<20;
        resolve_session_defaults(legacy_prefix,model,env,128*gib_bytes);
        expect(!legacy_prefix.session_cache,"existing prefix cache retains unnamed request behavior");
        auto small_route=route; resolve_session_defaults(small_route,model,env,1);
        expect(!small_route.session_cache && small_route.session_off_reason=="insufficient-memory","session memory guard");
        auto unknown=route; resolve_session_defaults(unknown,model,env,{});
        expect(!unknown.session_cache && unknown.session_off_reason=="memory-observation-unavailable","unknown session memory declines");
        auto unqualified=route; unqualified.mtp.draft_chain=false;
        resolve_session_defaults(unqualified,model,env,128*gib_bytes);
        expect(!unqualified.session_cache && unqualified.session_off_reason=="route-not-qualified","unsupported HTTP route declines");
        auto batch=route; batch.http_batch=4; resolve_session_defaults(batch,model,env,128*gib_bytes);
        expect(!batch.session_management_requested && batch.http_batch==4 && batch.session_off_reason=="http-batch-requires-ordinary-scheduler","batch concurrency retained");
        auto disk=route; auto disk_env=env; disk_env.erase("STRIX_SESSION_DISK");
        disk.session_dir=(std::filesystem::temp_directory_path()/"qwen-session-defaults-test").string();
        resolve_session_defaults(disk,model,disk_env,128*gib_bytes);
        expect(disk.session_cache && disk.session_disk && disk.session_disk_bytes>0,"disk default cap and writable directory");
        // An existing regular file is not a usable session directory. __FILE__ can be build-dir relative (Ninja), so
        // prefer the reference table's absolute path, which the build passes.
#ifdef STRIX_QWEN_CACHE_REFERENCE
        auto bad_disk=route; bad_disk.session_dir=STRIX_QWEN_CACHE_REFERENCE;
#else
        auto bad_disk=route; bad_disk.session_dir=__FILE__;
#endif
        resolve_session_defaults(bad_disk,model,disk_env,128*gib_bytes);
        expect(bad_disk.session_cache && !bad_disk.session_disk && !bad_disk.session_disk_off_reason.empty(),"disk failure preserves RAM");
        EffectiveConfig config; config.options=on;
        const auto session_json=serve::parse_json(effective_config_json(config));
        const auto* fields=session_json.find("session_cache");
        expect(fields && fields->find("enabled")->as_bool() && fields->find("ram_bytes") &&
            fields->find("disk_bytes") && fields->find("dir") && fields->find("reason"),"effective JSON session fields");
        config.options=batch;
        expect(effective_config_notice(config)=="session cache: off (http-batch-requires-ordinary-scheduler)\nprefix cache: off (not-serving)\n","one decline notice per store");
    }
    {
        Options route; route.http_serve=true; route.known_prefix_window=true; route.known_prefix_full_rows=true;
        route.known_prefix_prefill_rows=1024; route.generate_tokens=10; route.context_tokens=8192;
        const auto resolved=[&](Options o, Environment e={}, std::optional<std::uint64_t> a=128*gib_bytes) {
            constexpr SessionRouteQualification test_only[]={{"http-target",true}};
            resolve_prefix_defaults(o,model,e,a,test_only); return o;
        };
        for (const auto& entry : prefix_auto_route_qualifications) {
            expect(!entry.gpu_qualified,"automatic prefix routes remain unqualified");
            auto production=route; production.mtp.enabled=entry.route!="http-target";
            production.http_batch=entry.route=="http-mtp-batch" ? 2 : 0;
            resolve_prefix_defaults(production,model,{},128*gib_bytes);
            expect(production.prefix_cache_host_bytes==0 && production.prefix_cache_source=="off" &&
                production.prefix_cache_reason=="automatic budget not yet GPU-qualified: " + std::string(entry.route),"production auto budget gated");
            EffectiveConfig prefix_notice; prefix_notice.options=production;
            expect(effective_config_notice(prefix_notice).find("prefix cache: off ("+production.prefix_cache_reason+")\n")!=std::string::npos,"qualified route notice");
        }
        const auto auto_prefix=resolved(route);
        expect(auto_prefix.prefix_cache_host_bytes==4*gib_bytes && auto_prefix.prefix_cache_source=="auto","prefix fitted default on");
        expect(resolved(route,{},16*gib_bytes).prefix_cache_host_bytes==gib_bytes,"prefix MemAvailable fraction");
        expect(resolved(route,{},{}).prefix_cache_reason=="memory-observation-unavailable","prefix unknown observation");
        expect(resolved(route,{},1).prefix_cache_reason=="checkpoint-does-not-fit","prefix insufficient checkpoint fit");
        auto wrong=route; wrong.known_prefix_full_rows=false;
        expect(resolved(wrong).prefix_cache_reason=="route-not-accepted","prefix existing acceptance predicate");
        auto nonhttp=route; nonhttp.http_serve=false;
        expect(resolved(nonhttp).prefix_cache_host_bytes==0,"prefix non-HTTP unchanged");
        auto sessions=route; sessions.session_management_requested=true; sessions.session_cache=true;
        auto kept=resolved(sessions);
        expect(kept.session_management_requested && kept.session_cache && kept.prefix_cache_host_bytes==0 &&
            kept.prefix_cache_reason=="sessions-admitted","automatic prefix never disables sessions");
        auto explicit_budget=route; explicit_budget.prefix_cache_host_bytes=12345;
        explicit_budget.supplied_options.insert("--prefix-cache-host-bytes");
        expect(resolved(explicit_budget,{},1).prefix_cache_host_bytes==12345,"explicit budget wins fitting");
        expect(resolved(explicit_budget,{{"STRIX_PREFIX_CACHE","0"}}).prefix_cache_host_bytes==0,"prefix env off");
        auto cli=route; cli.supplied_options.insert("--prefix-cache"); cli.prefix_cache=true;
        cli.supplied_options.insert("--prefix-cache-retention"); cli.prefix_cache_retention="lru";
        auto precedence=resolved(cli,{{"STRIX_PREFIX_CACHE","0"},{"STRIX_PREFIX_CACHE_RETENTION","fair"}});
        expect(precedence.prefix_cache_host_bytes==4*gib_bytes && precedence.prefix_cache_retention=="lru","prefix CLI wins env");
        auto prefix_baseline=resolved(cli,{{"STRIX_BASELINE","1"}});
        EffectiveConfig baseline_notice; baseline_notice.options=prefix_baseline;
        expect(effective_config_json(baseline_notice).find("\"prefix_cache\"")==std::string::npos,"environment baseline receipt omits prefix object");
        expect(effective_config_notice(baseline_notice).find("prefix cache:")==std::string::npos,"baseline has no new prefix stderr line");
        expect(!prefix_baseline.prefix_cache_host_bytes && prefix_baseline.prefix_cache_retention=="lru" && prefix_baseline.prefix_cache_reason=="baseline","baseline wins prefix policy");
        auto staging_cli=parse({"decoder","MODEL","1","--prefix-cache-staging-bytes","123456"});
        expect(staging_cli.prefix_cache_staging_bytes==123456 && operational_session_option("--prefix-cache-staging-bytes"),
            "staging cap is an operational CLI option");
        auto staging_env=resolved(route,{{"STRIX_PREFIX_CACHE_STAGING_BYTES","234567"}});
        expect(staging_env.prefix_cache_staging_bytes==234567,"staging environment cap");
        auto staging_input=route;staging_input.prefix_cache_staging_bytes=345678;
        staging_input.supplied_options.insert("--prefix-cache-staging-bytes");
        expect(resolved(staging_input,{{"STRIX_PREFIX_CACHE_STAGING_BYTES","234567"}}).prefix_cache_staging_bytes==345678,
            "staging CLI wins environment");
        auto parsed=parse({"decoder","MODEL","1","--prefix-cache","off","--prefix-cache-retention","lru"});
        expect(!parsed.prefix_cache && parsed.prefix_cache_retention=="lru","prefix switches parse");
        auto baseline_explicit=explicit_budget; baseline_explicit.baseline=true;
        baseline_explicit.prefix_cache_dir="explicit-dir"; baseline_explicit.prefix_cache_every=4096;
        baseline_explicit.prefix_cache_every_explicit=true;
        baseline_explicit.prefix_cache_then_prompts={"prompt-file"};
        for (const auto& e : {Environment{}, Environment{{"STRIX_BASELINE","1"}}}) {
            auto input=baseline_explicit; input.baseline=e.empty();
            const auto got=resolved(input,e);
            expect(got.prefix_cache_host_bytes==12345 && got.prefix_cache_dir=="explicit-dir" &&
                got.prefix_cache_every==4096 && got.prefix_cache_every_explicit &&
                got.prefix_cache_then_prompts==input.prefix_cache_then_prompts && got.prefix_cache_retention=="lru",
                "baseline preserves every explicit prefix option");
        }
        auto decline=route; decline.http_serve=false; decline.prefix_cache_dir="kept";
        decline.prefix_cache_every_explicit=true; decline.prefix_cache_every=4096;
        decline.prefix_cache_then_prompts={"kept-prompt"};
        auto declined_prefix=resolved(decline);
        expect(declined_prefix.prefix_cache_dir=="kept" && declined_prefix.prefix_cache_every_explicit &&
            declined_prefix.prefix_cache_every==4096 && declined_prefix.prefix_cache_then_prompts==decline.prefix_cache_then_prompts,
            "automatic decline preserves explicit options");
        for (const auto& e : {Environment{{"STRIX_PREFIX_CACHE","invalid"}},
                Environment{{"STRIX_PREFIX_CACHE_RETENTION","invalid"}}}) {
            bool threw=false;
            try { (void)resolved(route,e); } catch (const std::invalid_argument&) { threw=true; }
            expect(threw,"strict shared environment parser rejects invalid policy");
        }
        Options standalone;
        apply_prefix_cache_environment(standalone, [](const char* name) -> std::optional<std::string_view> {
            if (std::string_view(name)=="STRIX_BASELINE") return "1";
            if (std::string_view(name)=="STRIX_PREFIX_CACHE") return "on";
            return std::nullopt;
        });
        expect(standalone.baseline && standalone.prefix_cache_retention=="lru", "non-resolver policy honours baseline env");
        const auto baseline_cli=parse({"decoder","MODEL","1","--baseline","--generate","4",
            "--known-prefix-window","--known-prefix-full-rows","--known-prefix-prefill-rows","1024",
            "--known-prefix-workspace-budget-bytes","1048576","--known-prefix-host-metadata-budget-bytes","1048576",
            "--known-prefix-uma-budget-bytes","1048576","--known-prefix-host-headroom-bytes","0",
            "--deferred-moe-status","--virtual-q8-gdn-input","--prefetch-q8-gdn-input",
            "--virtual-q8-gdn-output","--wave8-q8-gdn-input","--wave8-q8-gdn-output",
            "--prefix-cache-host-bytes","12345","--prefix-cache-dir","explicit-dir",
            "--prefix-cache-every","4096","--prefix-cache-then-prompt","next-prompt"});
        expect(baseline_cli.prefix_cache_host_bytes==12345 && baseline_cli.prefix_cache_dir=="explicit-dir" &&
            baseline_cli.prefix_cache_every_explicit && baseline_cli.prefix_cache_every==4096 &&
            baseline_cli.prefix_cache_then_prompts==std::vector<std::string>{"next-prompt"} &&
            baseline_cli.prefix_cache_retention=="lru", "CLI baseline preserves explicit prefix options before resolver");
        auto off_cli=parse({"decoder","MODEL","1","--prefix-cache","off",
            "--prefix-cache-host-bytes","12345","--prefix-cache-dir","explicit-dir",
            "--prefix-cache-every","4096","--prefix-cache-then-prompt","next-prompt"});
        expect(!prefix_state_cache_requested(off_cli),"explicit off clears explicit prefix settings before validation");
        EffectiveConfig disabled; disabled.options=off_cli;
        expect(effective_config_notice(disabled).find("prefix cache: off (disabled)\n")!=std::string::npos,
            "explicit off prints the decline line outside HTTP too");
        EffectiveConfig c; c.options=auto_prefix;
        const auto j=serve::parse_json(effective_config_json(c));
        expect(j.find("prefix_cache")->find("enabled")->as_bool() &&
            j.find("prefix_cache")->find("source")->as_string()=="auto","effective prefix config JSON");
    }
    auto gsq_manifest = manifest;
    ManifestTensor gsq_gate;
    gsq_gate.name = "blk.0.ffn_gate_exps.weight"; gsq_gate.ggml_type = gsq_a8::type_q2_0;
    gsq_manifest.tensors.push_back(gsq_gate);
    gsq_manifest.layout_identity = model_manifest_layout_identity(gsq_manifest);
    const auto gsq_decline = resolve_w7_case(request,model,info,gsq_manifest,
        validated_model_content_identity,capabilities,memory);
    const auto nonqwen = resolve_w7_case(request,model,info,manifest,"unqualified-content",capabilities,memory);
    auto other_model = model; other_model.hidden_size += 32U;
    const auto noncanonical = resolve_w7_case(request,other_model,info,manifest,
        validated_model_content_identity,capabilities,memory);
    for (const auto& config : {absent_build, gsq_decline, nonqwen, noncanonical, declined, resolve(request,missing)}) {
        expect(!config.admitted && config.environment == export_config.environment,
            "declined build/GSQ/non-Qwen/memory routes have exactly the historical environment");
        expect(effective_config_notice(config).find("unqualified-context") == std::string::npos,
            "automatic long-context qualification notice stays confined to admission");
    }
    const auto baseline_absent = resolve_w7_case(baseline,model,info,manifest,
        "unqualified-content",{false,false,false},missing);
    expect(!baseline_absent.admitted && baseline_absent.options.numerical_profile == DecodeNumericalProfile::exact_v5 &&
        !baseline_absent.options.prefill_numerical_profile && baseline_absent.options.context_tokens == request.options.context_tokens,
        "baseline never forces dense-A8 or grows context when model/build admission fails");
    {
        Options mtp_pool;mtp_pool.mtp.enabled=true;mtp_pool.mtp.window_rows=8;
        mtp_pool.mtp.window_rows_set={2,4,8};mtp_pool.mtp.window_qsa_rows=true;mtp_pool.mtp.candidate_depth=8;
        mtp_pool.context_tokens=262144;
        expect(qsa_mtp_pool_route(mtp_pool),"MTP shared pool defaults ON on converted route");
        expect(qsa_mtp_pool_bytes(mtp_pool,ModelContract::qwen38_flash_next())==218644480,"one thirteen-role pool plus shared K1 scratch");
        expect(!qsa_mtp_pool_route(mtp_pool,true),"GSQ suppresses MTP shared pool");
        for(const char* name:{"STRIX_QSA_MTP_POOL","STRIX_QSA_DECODE_POOL_BF16","STRIX_QPF2K_DEFAULTS"}) {
            ::setenv(name,"0",1);expect(!qsa_mtp_pool_route(mtp_pool),"off switches keep historical MTP route");::unsetenv(name);
        }
        const auto before_digest_argv=effective_prefill_argv(mtp_pool);
        ::setenv("STRIX_MTP_STEP_LOGITS_DIGEST","1",1);
        expect(qsa_mtp_pool_route(mtp_pool) && effective_prefill_argv(mtp_pool)==before_digest_argv,
            "step logits diagnostics preserve resolved MTP route and argv");
        expect(requested_prefill_environment().at("STRIX_MTP_STEP_LOGITS_DIGEST")=="1",
            "step logits diagnostic request is visible in inspection");
        ::unsetenv("STRIX_MTP_STEP_LOGITS_DIGEST");
        const auto prior_image=requested_prefill_environment();
        ::setenv("STRIX_QSA_MTP_POOL","0",1);
        expect(requested_prefill_environment()==prior_image,"MTP OFF preserves historical printed environment bytes");
        expect(qsa_mtp_pool_bytes(mtp_pool,ModelContract::qwen38_flash_next())==0,"MTP OFF has no extra admission bytes");
        ::unsetenv("STRIX_QSA_MTP_POOL");
        for(unsigned route=0;route<15U;++route) {
            auto draft=mtp_pool;
            if(route==0) draft.http_serve=true;
            if(route==1) { draft.http_serve=true;draft.http_batch=8; }
            if(route==2) draft.session_load="STATE";
            if(route==3) draft.session_live_control=true;
            if(route==4) draft.session_then_prompt={1};
            if(route==5) draft.prefix_cache_host_bytes=1;
            if(route==6) draft.rtx_state_import=true;
            if(route==7) { draft.rotary=RotaryDescriptor::yarn(4);draft.context_tokens=1048576; }
            if(route==8) draft.serving_prefix_files={"PREFIX"};
            if(route==9) { draft.http_serve=true;draft.prefix_cache_host_bytes=1; }
            if(route==10) { draft.http_serve=true;draft.prefix_cache_dir="DIR"; }
            if(route==11) { draft.http_serve=true;draft.session_mtp_fixed=true; }
            if(route==12) { draft.http_serve=true;draft.session_management_requested=true; }
            if(route==13) { draft.http_serve=true;draft.rtx_state_import=true; }
            if(route==14) { draft.http_serve=true;draft.qsa_mtp_pool_fallback_unfit=true; }
            expect(qsa_mtp_pool_route(draft)==(route==0 || route==7 || route==9),
                "single HTTP (with RAM prefix cache) and YaRN f4 CLI default on; slots, sessions, disk restore and import stay opt-in");
            expect(qsa_mtp_pool_single_fallback_pending(draft)==(route==1),"only AUTO/capped batch has a pending single fallback");
            ::setenv("STRIX_QSA_MTP_POOL","1",1);
            expect(qsa_mtp_pool_route(draft),"explicit env enables unqualified M1 route when supported");
            ::setenv("STRIX_QPF2K_DEFAULTS","0",1);
            expect(!qsa_mtp_pool_route(draft),"opt-in cannot bypass defaults suppression");
            ::unsetenv("STRIX_QPF2K_DEFAULTS");::unsetenv("STRIX_QSA_MTP_POOL");
        }
        for(const auto c:{32768U,131072U,262144U,524288U}) {
            auto qualified=mtp_pool;qualified.rotary=RotaryDescriptor::yarn(2);qualified.context_tokens=c;
            expect(qsa_mtp_pool_route(qualified) && ple_rolling_prompt_supported(qualified),"YaRN f2 qualified CLI defaults on through 524K");
        }
        auto f4=mtp_pool;f4.rotary=RotaryDescriptor::yarn(4);f4.context_tokens=1048576;
        expect(ple_rolling_prompt_supported(f4) && qsa_mtp_pool_route(f4),"qualified f4 MTP CLI rolling and pool default on");
        ::setenv("STRIX_PLE_ROLLING","1",1);
        expect(ple_rolling_prompt_supported(f4),"explicit env enables f4 rolling qualification");
        ::unsetenv("STRIX_PLE_ROLLING");
        {
            auto single=mtp_pool;single.http_serve=true;
            auto fallback=single;fallback.http_batch=1;
            auto unfit=single;unfit.qsa_mtp_pool_fallback_unfit=true;
            expect(qsa_mtp_pool_bytes(single,ModelContract::qwen38_flash_next())==218644480 &&
                qsa_mtp_pool_bytes(fallback,ModelContract::qwen38_flash_next())==0 &&
                qsa_mtp_pool_bytes(unfit,ModelContract::qwen38_flash_next())==0,
                "batch sizing excludes the pool; an unfit one-slot fallback keeps it out of admission");
            expect(std::string_view(qsa_mtp_pool_decline(unfit))=="single-fallback-does-not-fit","unfit fallback names its decline");
            auto cap_one=fallback;cap_one.http_batch_explicit=true;cap_one.http_batch_cap=1;
            expect(!qsa_mtp_pool_single_fallback_pending(cap_one),"explicit cap 1 keeps the batch runner");
            auto yarn_http=single;yarn_http.rotary=RotaryDescriptor::yarn(2);yarn_http.context_tokens=262144;
            expect(!qsa_mtp_pool_route(yarn_http),"YaRN HTTP stays closed by default");
            for(const auto name:{qwen::profile::Name::standard,qwen::profile::Name::fast,qwen::profile::Name::quality}) {
                auto profiled=single;profiled.product_profile=qwen::profile::resolve(name,{});
                auto profiled_fallback=fallback;profiled_fallback.product_profile=profiled.product_profile;
                expect(qsa_mtp_pool_route(profiled) && qsa_mtp_pool_single_fallback_pending(profiled_fallback),
                    "product profiles carry no pool lever: the single HTTP decision is profile independent");
            }
            const auto pool_json=[](const Options& o) { EffectiveConfig c;c.options=o;return effective_config_json(c); };
            expect(pool_json(single).find("\"qsa_mtp_pool\":{\"requested\":true,\"effective\":\"runtime-admission-required\",\"route\":\"http-single\",\"source\":\"default\"")!=std::string::npos,
                "inspection reports the default single HTTP pool");
            expect(pool_json(fallback).find("\"effective\":\"single-fallback-only\",\"route\":\"http-batch\"")!=std::string::npos &&
                pool_json(fallback).find("\"reserved_bytes\":218644480")!=std::string::npos,
                "inspection reports the pool the one-slot fallback may admit");
            for(const char* name:{"STRIX_QSA_MTP_POOL","STRIX_QPF2K_DEFAULTS"}) {
                ::setenv(name,"0",1);
                expect(!qsa_mtp_pool_route(single) && !qsa_mtp_pool_single_fallback_pending(fallback) &&
                    pool_json(single).find("qsa_mtp_pool")==std::string::npos && pool_json(fallback).find("qsa_mtp_pool")==std::string::npos,
                    "off switches remove the single HTTP pool, its fallback and its inspection object");
                ::unsetenv(name);
            }
        }
        mtp_pool.http_serve=true;mtp_pool.qsa_mtp_pool_explicit_opt_in=true;
        mtp_pool.mtp.copy_wide_rows={32};expect(!qsa_mtp_pool_route(mtp_pool),"wide writers decline before allocation");
        mtp_pool.mtp.copy_wide_rows.clear();mtp_pool.baseline=true;expect(!qsa_mtp_pool_route(mtp_pool),"baseline retains historical MTP");
        // Serial decode pool route: plain append-only CLI only; every multi-request or rewriting route is excluded.
        const auto route = [&](std::vector<std::string> extra) {
            std::vector<std::string> args{"decoder","target.gguf","1","2","3","--generate","4"};
            args.insert(args.end(), extra.begin(), extra.end());
            return qsa_decode_pool_route(parse(args), false);
        };
        expect(route({}), "decode pool on for the plain CLI route");
        for(const auto& flags : std::vector<std::vector<std::string>>{
                {"--session-save","checkpoint"},{"--session-load","checkpoint"},
                {"--session-live-control"},{"--session-save","checkpoint","--session-then-prompt","4,5"}})
            expect(route(flags),"decode pool stays on for save/restore/park/continuation");
        expect(!qsa_decode_pool_route(parse({"decoder","target.gguf","1","2","3","--generate","4"}), true), "decode pool off for GSQ models");
        auto cached = parse({"decoder","target.gguf","1","2","3","--generate","4"});
        cached.prefix_cache_host_bytes = UINT64_C(1) << 30U;
        expect(!qsa_decode_pool_route(cached, false), "decode pool off with the prefix state cache");
        cached.prefix_cache_host_bytes = 0U;
        cached.prefix_cache_then_prompts = {"4,5,6"};
        expect(!qsa_decode_pool_route(cached, false), "decode pool off for follow-up prefix-cache prompts");
        expect(!route({"--baseline"}), "decode pool off with --baseline");
        ::setenv("STRIX_QSA_DECODE_POOL", "0", 1);
        expect(!route({}), "STRIX_QSA_DECODE_POOL=0 disables the decode pool");
        ::unsetenv("STRIX_QSA_DECODE_POOL");
        ::setenv("STRIX_QPF2K_DEFAULTS", "0", 1);
        expect(!route({}), "STRIX_QPF2K_DEFAULTS=0 disables the decode pool");
        ::unsetenv("STRIX_QPF2K_DEFAULTS");
        auto sized = parse({"decoder","target.gguf","1","2","3","--generate","4"});
        const auto check_bytes = [&](const Options& options, bool gsq, std::uint64_t key_bytes) {
            for (const auto context : {262144U, 524288U, 1048576U}) {
                auto o = options; o.context_tokens = context;
                const auto blocks = context / 4U;
                const auto expected = UINT64_C(12) * blocks * 128U * key_bytes + (blocks / 4U) * 4U;
                expect(qsa_decode_pool_key_bytes(o, gsq) == key_bytes, "decode pool key width follows the shared policy");
                expect(qsa_decode_pool_bytes(o, model, gsq) == expected, "decode pool exact bytes at 262144/524288/1048576");
            }
        };
        expect(qsa_decode_pool_bf16(sized, false), "BF16 decode pool default on");
        check_bytes(sized, false, 2U);
        // Literal totals also guard the model geometry and maxima row independently.
        for (const auto& [context, bytes] : std::array<std::pair<std::uint64_t,std::uint64_t>,3>{{
                {262144U,201392128U},{524288U,402784256U},{1048576U,805568512U}}}) {
            sized.context_tokens = context;
            expect(qsa_decode_pool_bytes(sized, model, false) == bytes, "BF16 decode pool literal total bytes");
        }
        auto baseline_pool = sized; baseline_pool.baseline = true;
        check_bytes(baseline_pool, false, 4U);
        check_bytes(sized, true, 4U);
        qpf2k::baseline = true;
        check_bytes(sized, false, 4U);
        qpf2k::baseline = false;
        for (const char* name : {"STRIX_QPF2K_DEFAULTS", "STRIX_QSA_DECODE_POOL_BF16", "STRIX_QSA_DECODE_POOL"}) {
            ::setenv(name, "0", 1);
            check_bytes(sized, false, 4U);
            ::unsetenv(name);
        }
        ::setenv("STRIX_QSA_DECODE_POOL_BF16", "1", 1);
        expect(requested_prefill_environment().at("STRIX_QSA_DECODE_POOL_BF16") == "1",
            "requested environment includes the decode pool format switch");
        check_bytes(sized, false, 2U);
        check_bytes(baseline_pool, false, 4U);
        check_bytes(sized, true, 4U);
        ::unsetenv("STRIX_QSA_DECODE_POOL_BF16");
        // --print-effective-config resolves each long-context item and names why it is off.
        const auto items = [&](const Options& options, bool gsq) {
            EffectiveConfig printed; printed.options = options; printed.gsq_owner = gsq;
            return effective_config_json(printed);
        };
        const auto has = [](const std::string& text, const char* field) { return text.find(field) != std::string::npos; };
        auto plain = items(sized, false);
        expect(has(plain, "\"STRIX_QSA_UNION_MERGE\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"STRIX_QSA_SCORE_ROWS64\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"STRIX_QSA_DECODE_POOL\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"STRIX_QSA_DECODE_SELECT\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"STRIX_QSA_DECODE_POOL_BF16\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"STRIX_PREFILL_MOE_MERGE_VEC\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"STRIX_QSA_CACHE_ONLY\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"STRIX_QPF2K_GDN_CHUNK16\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"qsa_decode_pool_key_bytes\":2,"), "effective config resolves the long-context items as on");
        ::setenv("STRIX_QSA_DECODE_SELECT", "0", 1);
        ::setenv("STRIX_QSA_DECODE_POOL_BF16", "0", 1);
        plain = items(sized, false);
        expect(has(plain, "\"STRIX_QSA_DECODE_SELECT\":{\"enabled\":false,\"suppression\":\"explicit-off\"}") &&
            has(plain, "\"STRIX_QSA_DECODE_POOL_BF16\":{\"enabled\":false,\"suppression\":\"explicit-off\"}") &&
            has(plain, "\"STRIX_QSA_DECODE_POOL\":{\"enabled\":true,\"suppression\":null}") &&
            has(plain, "\"qsa_decode_pool_key_bytes\":4,") &&
            has(requested_prefill_environment().at("STRIX_QSA_DECODE_SELECT"), "0"),
            "explicit item switches resolve off while the request stays in the environment");
        ::unsetenv("STRIX_QSA_DECODE_SELECT");
        ::unsetenv("STRIX_QSA_DECODE_POOL_BF16");
        ::setenv("STRIX_QPF2K_DEFAULTS", "0", 1);
        expect(has(items(sized, false), "\"STRIX_QSA_UNION_MERGE\":{\"enabled\":false,\"suppression\":\"STRIX_QPF2K_DEFAULTS=0\"}"),
            "STRIX_QPF2K_DEFAULTS=0 is the reported suppression");
        ::unsetenv("STRIX_QPF2K_DEFAULTS");
        const auto gsq_text = items(sized, true);
        expect(has(gsq_text, "\"STRIX_QSA_SCORE_ROWS64\":{\"enabled\":false,\"suppression\":\"gsq\"}") &&
            has(gsq_text, "\"STRIX_QSA_DECODE_SELECT\":{\"enabled\":false,\"suppression\":\"gsq\"}") &&
            has(gsq_text, "\"qsa_decode_pool_key_bytes\":0,"), "GSQ owners report the gsq suppression");
        qpf2k::baseline = true;
        expect(has(items(sized, false), "\"STRIX_QSA_DECODE_POOL\":{\"enabled\":false,\"suppression\":\"baseline\"}"),
            "--baseline is the reported suppression");
        qpf2k::baseline = false;
        auto mtp = sized; mtp.mtp.enabled = true;
        const auto mtp_text = items(mtp, false);
        expect(has(mtp_text, "\"STRIX_QSA_DECODE_POOL\":{\"enabled\":false,\"suppression\":\"mtp\"}") &&
            has(mtp_text, "\"STRIX_QSA_DECODE_SELECT\":{\"enabled\":false,\"suppression\":\"mtp\"}") &&
            has(mtp_text, "\"STRIX_QSA_UNION_MERGE\":{\"enabled\":true,\"suppression\":null}"),
            "an excluded decode route names the route; prefill items stay on");
        expect(std::string(qsa_decode_pool_route_suppression(cached, false)) == "prefix-state-cache",
            "the prefix state cache is the reported route exclusion");
    }
    for(int master=0;master<4;++master) {
        Options modes; modes.session_tiled_suffix=true; modes.session_mtp_fixed=true;
        modes.supplied_options.insert("--session-tiled-suffix"); modes.supplied_options.insert("--session-mtp-fixed");
        Environment env;
        if(master==1) modes.baseline=true;
        if(master==2) env["STRIX_BASELINE"]="1";
        if(master==3) env["STRIX_QPF2K_DEFAULTS"]="0";
        resolve_session_defaults(modes,model,env,UINT64_C(8)<<30);
        expect(modes.session_tiled_suffix_requested && modes.session_mtp_fixed_requested,"session mode requests retained");
        expect(modes.session_tiled_suffix==(master==0) && modes.session_mtp_fixed==(master==0),"masters suppress both numerical modes");
        auto modes_json=session_modes_json(modes);
        for(const char* name:{"tiled_suffix","mtp_fixed"}) {
            expect(modes_json.find(name)->find("requested")->as_bool(),"mode requested disclosed");
            expect(modes_json.find(name)->find("effective")->as_bool()==(master==0),"mode effective disclosed");
            expect(modes_json.find(name)->find("bit_changing")->as_bool(),"mode bit-changing disclosed");
            expect(modes_json.find(name)->find("suppression")->as_string().empty()==(master==0),"suppression disclosed");
        }
    }
    for (unsigned protocol=0;protocol<11U;++protocol) {
        Options session;
        const char* name="";
        switch(protocol) {
        case 0: session.session_save="state"; name="--session-save"; break;
        case 1: session.session_load="state"; name="--session-load"; break;
        case 2: session.session_live_control=true; name="--session-live-control"; break;
        case 3: session.session_management_requested=true; name="session management"; break;
        case 4: session.session_then_prompt="prompt"; name="--session-then-prompt"; break;
        case 5: session.session_logits="logits"; name="--session-logits"; break;
        case 6: session.session_first_generate=1; name="--session-first-generate"; break;
        case 7: session.session_validate_only=true; name="--session-validate-only"; break;
        case 8: session.session_tiled_suffix=true; name="--session-tiled-suffix"; break;
        case 9: session.session_mtp_fixed=true; name="--session-mtp-fixed"; break;
        default: session.supplied_options.insert("--session-first-generate"); name="--session-first-generate"; break;
        }
        bool rejected=false;
        try { require_bitcalc_session_route(session,true); }
        catch(const std::invalid_argument& error) {
            const std::string message=error.what();
            rejected=message.find("STRIX_BITCALC")!=std::string::npos && message.find(name)!=std::string::npos;
        }
        expect(rejected,"measurement session refusal names the hook and protocol");
        require_bitcalc_session_route(session,false);
    }
    EffectiveConfig receipt; receipt.options.baseline=true;
    expect(effective_config_json(receipt).find("\"prefix_cache\"")==std::string::npos,"baseline receipt omits additive prefix object");
    receipt.options.baseline=false;
    expect(effective_config_json(receipt).find("\"prefix_cache\"")!=std::string::npos,"nonbaseline receipt includes additive prefix object");
    for(const auto mode:{"0","1","capture","l0","l1"}) {
        Options batch;
        apply_prefix_cache_environment(batch,[&](const char* name)->std::optional<std::string_view> {
            if(std::string_view(name)=="STRIX_PREFIX_BATCH_CERT") return mode;
            return {};
        });
        expect(batch.prefix_batch_cert==(std::string_view(mode)=="0"?0U:std::string_view(mode)=="capture"?2U:std::string_view(mode)=="l1"?3U:1U),"batch certification env parser");
        batch.baseline=true;
        apply_prefix_cache_environment(batch,[](const char*)->std::optional<std::string_view>{return {};});
        expect(!batch.prefix_batch_cert,"baseline suppresses batch certification");
    }
    for(const auto* mode:{"off","l0","capture","l1"}) {
        const auto selected=parse({"decoder","MODEL","1","--prefix-batch-cert",mode});
        for(const auto* route:{"batch","single","cli","yarn","cache-off","no-budget","live-control"}) {
            auto caps_options=selected;
            caps_options.http_serve=true; caps_options.http_batch=2;
            caps_options.prefix_cache_host_bytes=4096;
            const std::string_view r=route;
            if(r=="single") caps_options.http_batch=0;
            if(r=="cli" || r=="yarn") { caps_options.http_serve=false; caps_options.http_batch=0; }
            if(r=="yarn") caps_options.rotary=validate_qwen_rotary_request(2,true,524288,false);
            if(r=="cache-off") caps_options.prefix_cache=false;
            if(r=="no-budget") caps_options.prefix_cache_host_bytes=0;
            if(r=="live-control") caps_options.session_live_control=true;
            auto caps=serve::Json::object(); caps.set("existing",serve::Json("unchanged"));
            const auto original=caps.dump();
            prefix_batch_cert_capabilities(caps,caps_options);
            EffectiveConfig cert_receipt; cert_receipt.options=caps_options;
            const auto effective=serve::parse_json(effective_config_json(cert_receipt));
            const auto* prefix=effective.find("prefix_cache");
            const auto* cert=caps.find("prefix_batch_certification");
            const bool active=std::string_view(mode)!="off" && r=="batch";
            if(!active) {
                expect(caps.dump()==original,"inactive certification capability bytes unchanged");
                expect(prefix && !prefix->find("batch_cert_mode") && !prefix->find("batch_l1_qualified"),
                    "inactive routes omit effective certification fields");
                auto disabled=cert_receipt; disabled.options.prefix_batch_cert=0;
                expect(effective_config_json(cert_receipt)==effective_config_json(disabled),
                    "inactive certification effective config bytes unchanged");
            } else {
                expect(cert && cert->find("mode")->as_string()==mode,"capabilities disclose canonical batch mode");
                expect(prefix && prefix->find("batch_cert_mode") && prefix->find("batch_cert_mode")->as_string()==mode,
                    "effective config and capabilities use identical canonical modes");
                expect(cert->find("experimental")->as_bool() && !cert->find("l1_qualified")->as_bool() &&
                    !cert->find("automatic_qualified")->as_bool(),"capabilities retain experimental qualification boundary");
            }
            caps_options.baseline=true; caps=serve::Json::object();caps.set("existing",serve::Json("unchanged"));
            prefix_batch_cert_capabilities(caps,caps_options);
            expect(caps.dump()==original,"baseline suppresses certification capabilities byte for byte");
        }
    }
    expect(operational_session_option("--prefix-batch-cert"),"batch certification is operational");
    auto batch_cli=parse({"decoder","MODEL","1","--prefix-batch-cert","l0"});
    expect(batch_cli.prefix_batch_cert==1,"batch certification CLI parses");
    const auto captured_env=requested_prefill_environment([](const char* name)->const char* {
        return std::string_view(name)=="STRIX_PREFIX_BATCH_CERT"?"1":nullptr;
    });
    expect(captured_env.contains("STRIX_PREFIX_BATCH_CERT"),"resolver retains batch opt-in environment");
    apply_prefix_cache_environment(batch_cli,[](const char* name)->std::optional<std::string_view> {
        if(std::string_view(name)=="STRIX_PREFIX_BATCH_CERT") return "0";
        return {};
    });
    expect(batch_cli.prefix_batch_cert==1,"batch certification CLI wins env");
    expect(parse({"decoder","MODEL","1","--prefix-batch-cert","1","--baseline"}).prefix_batch_cert==0,"baseline wins batch CLI");
    for(const auto& row:prefix_batch_cert_route_qualifications) expect(!row.gpu_qualified,"batch L1 remains unqualified");
    {
        const auto records=cache_compatibility_records(model);
        if(argc==2 && std::string_view(arguments[1])=="--dump-cache-reference") {
            for(const auto& [name,digest]:records) std::cout<<name<<' '<<digest<<'\n';
            return 0;
        }
        // The build passes the table's absolute path: __FILE__ can be relative to the build directory (Ninja), so
        // running the binary from another directory would otherwise miss the table and report every case.
#ifdef STRIX_QWEN_CACHE_REFERENCE
        std::ifstream reference(STRIX_QWEN_CACHE_REFERENCE);
#else
        std::ifstream reference(std::filesystem::path(__FILE__).parent_path()/"qwen_cache_public_main_reference.txt");
#endif
        expect(bool(reference),"public-main reference table exists");
        if(reference) for(const auto& [name,digest]:records) {
            std::string expected_name,expected_digest;reference>>expected_name>>expected_digest;
            if(expected_name!=name || expected_digest!=digest) std::cerr<<"reference mismatch "<<name<<'\n';
            expect(expected_name==name && expected_digest==digest,"explicit resolver output matches public main");
        }
        std::string extra;expect(!(reference>>extra),"reference table has no missing cases");
    }
    {
        for(const auto runner:{CacheRunner::cli,CacheRunner::target,CacheRunner::single_mtp,CacheRunner::batch_mtp}) {
            Options o;o.context_tokens=8192;o.http_serve=runner!=CacheRunner::cli;
            o.mtp.enabled=runner==CacheRunner::single_mtp || runner==CacheRunner::batch_mtp;
            o.known_prefix_prefill_rows=2048;
            resolve_session_defaults(o,model,{},128*gib_bytes);resolve_prefix_defaults(o,model,{},128*gib_bytes);
            const auto before=effective_prefill_argv(o);const auto session_reason=o.session_off_reason;
            expect(cache_activation_json(o).find("effective")->is_null(),"inspection activation is pending");
            expect(cache_activation_json(o).find("batch_certification")->find("effective")->is_null(),"inspection cert is pending");
            finalize_cache_family(o,runner,runner==CacheRunner::batch_mtp?4:1);
            expect(!automatic_cache_work(o) && !o.cache_activation->prefix && !o.cache_activation->session &&
                !prefix_batch_cert_execution_mode(o) && !o.prefix_cache_host_bytes,"unknown identity performs zero optional work");
            expect(effective_prefill_argv(o)==before && o.context_tokens==8192 && o.session_off_reason==session_reason,
                "decline preserves inference and incumbent reporting");
            if(runner==CacheRunner::cli) expect(o.cache_activation->session_reason=="cli-save-load-explicit","CLI reason is not overwritten");
        }
        CacheLateFit fit;fit.U=256*gib_bytes;fit.A=136*gib_bytes;fit.L=120*gib_bytes;
        intersect_cache_capacity(fit,128*gib_bytes);
        expect(*fit.L==120*gib_bytes && *fit.A==8*gib_bytes && *fit.U==128*gib_bytes,"configured ceiling cannot erase residency");
        fit.observation_id=1;fit.limits_consistent=fit.aliasing_known=true;fit.F=fit.E=fit.S=fit.device_unpaid=0;
        fit.device_free=32*gib_bytes;fit.checkpoint=gib_bytes;fit.explicit_budget=8*gib_bytes;
        expect(!fit_cache_late(fit).budget,"clamped residency reproducer refuses unsafe fit");
        fit.U=112*gib_bytes;fit.A=3*gib_bytes;fit.L=120*gib_bytes;intersect_cache_capacity(fit,128*gib_bytes);
        expect(*fit.A==0 && *fit.L==120*gib_bytes,"cgroup intersection keeps independent residency");
        for(const auto tile:{2048U,3072U,8192U}) for(const auto every:{0U,4096U,65536U}) {
            const auto cadence=prefix_capture_cadence(tile,every);
            expect(minimum_prefix_capture_boundary(tile,tile*2,cadence)==tile,"last-boundary checkpoint fits below periodic cadence");
            expect(prefix_capture_boundary_selected(tile,tile+1,tile,cadence),"off-cadence last boundary retained");
            expect(!minimum_prefix_capture_boundary(tile,tile,cadence),"checkpoint must precede context boundary");
        }
        expect(prefix_capture_cadence(3072,0)==6144,"non-divisor default cadence agrees with capture");
        expect(cache_unpaid_slot_bytes(100,200,30,4,4,4)==120,"eager native slots still owe simultaneous prompt taps");
        expect(cache_unpaid_slot_bytes(100,200,30,4,1,1)==1020,"lazy native/state commitments plus all taps");
        expect(cache_unpaid_slot_bytes(100,200,30,1,1,1)==30,"single route unpaid prompt taps");
#ifndef _WIN32
        {
            const char* prompt_name="STRIX_PROMPT_STATE_DIGEST";
            const char* prompt_old=std::getenv(prompt_name);
            const std::optional<std::string> prompt_saved=prompt_old?std::optional<std::string>(prompt_old):std::nullopt;
            const char* name="STRIX_FINAL_STATE_DIGEST";
            const char* old=std::getenv(name);
            const std::optional<std::string> saved=old?std::optional<std::string>(old):std::nullopt;
            for(const bool mtp:{false,true}) {
                Options o;o.tokens={1,2,3};o.mtp.enabled=mtp;
                o.known_prefix_prefill_rows=2048;o.known_prefix_window=true;
                resolve_session_defaults(o,model,{},128*gib_bytes);
                resolve_prefix_defaults(o,model,{},128*gib_bytes);
                expect(o.cache_activation->automatic_path && !o.prefix_cache_host_bytes,
                    "CLI observer startup has no storage");
                // Plain CLI requests prompt state; MTP CLI requests only final state.
                if(mtp) {::setenv(name,"1",1);::unsetenv(prompt_name);}
                else {::unsetenv(name);::setenv(prompt_name,"1",1);}
                const auto required=[&] {
                    const char* prompt=std::getenv(prompt_name);
                    return cache_startup_observer_required(o,false,prompt && prompt[0]=='1');
                };
                expect(required(),"upstream CLI diagnostic creates independent registry without storage");
                finalize_cache_family(o,CacheRunner::cli,1,{});
                expect(!o.cache_activation->prefix && required(),"registry survives automatic family refusal");
                decline_cache_activation(*o.cache_activation,"cache-setup-allocation-failed");
                expect(required(),"registry survives storage rollback");
                o.known_prefix_window=false;
                expect(!required(),"diagnostic registry requires upstream tiled predicate");
            }
            ::unsetenv(name);
            if(prompt_saved) ::setenv(prompt_name,prompt_saved->c_str(),1);else ::unsetenv(prompt_name);
            Options o;o.known_prefix_prefill_rows=2048;o.known_prefix_window=true;
            expect(cache_startup_observer_required(o,true,false),"import registry survives storage refusal");
            expect(!cache_startup_observer_required(o,false,false),"plain run needs no optional observer");
            if(saved) ::setenv(name,saved->c_str(),1);else ::unsetenv(name);
        }
#endif
        {
            CacheLateFit pressure;pressure.U=128*gib_bytes;pressure.L=120*gib_bytes;pressure.A=8*gib_bytes;
            pressure.E=pressure.S=pressure.device_unpaid=0;pressure.F=0;pressure.device_free=8*gib_bytes;
            pressure.checkpoint=256U<<20;pressure.observation_id=1;pressure.limits_consistent=pressure.aliasing_known=true;
            expect(fit_cache_late(pressure).budget!=0,"readiness pressure without unpaid owners would fit");
            pressure.F=cache_unpaid_slot_bytes(100,200,gib_bytes,4,4,4);
            expect(!fit_cache_late(pressure).budget,"readiness pressure reserves every simultaneous request owner");
        }
        CachePolicyActivation activation;activation.automatic_path=true;activation.prefix=true;activation.cert_effective=1;
        set_prefix_batch_certification_enabled(true);
        activation.certification_transition=[](bool enabled){set_prefix_batch_certification_enabled(enabled);};
        auto shared=std::make_shared<CachePolicyActivation>(activation);Options original;original.cache_activation=shared;auto copy=original;
        decline_cache_activation(*copy.cache_activation,"pressure-at-readiness");
        expect(!prefix_batch_certification_enabled() && !original.cache_activation->cert_effective && !original.cache_activation->prefix,"shared decline clears certification latch");
        Options protocol;protocol.http_serve=true;protocol.session_mtp_fixed=protocol.session_tiled_suffix=true;
        protocol.session_mtp_fixed_requested=protocol.session_tiled_suffix_requested=true;
        expect(!cache_activation_json(protocol).find("executed_session_protocol")->find("fixed_effective")->as_bool(),"declined HTTP protocol reports ineffective");
        protocol.session_management_requested=true;
        expect(cache_activation_json(protocol).find("executed_session_protocol")->find("fixed_effective")->as_bool(),"managed explicit protocol reports effective");
    }
    {
        for(const auto n:{1U,2U,4U}) for(unsigned built=1;built<=n;++built) {
            expect(cache_future_slot_bytes(100,20,n,built,0)==100U*(n-built)+20U*n,
                "initial cache fit reserves all lazy states and every unpaid native owner");
            expect(cache_future_slot_bytes(100,20,n,built,built)==120U*(n-built),
                "readiness fit retains the full remaining lazy growth commitment");
        }
        for(const auto mode:{0U,1U,2U}) {
            bool declined_overflow=false;
            try {
                if(mode==0) (void)cache_future_slot_bytes(UINT64_MAX,1,4,1,0);
                if(mode==1) (void)cache_future_slot_bytes(UINT64_MAX/2,UINT64_MAX/2,2,0,0);
                if(mode==2) (void)cache_future_slot_bytes(100,20,1,2,0);
            } catch(const std::overflow_error&) {declined_overflow=true;}
            expect(declined_overflow,"OFF owner commitment overflow/count mismatch refuses instead of wrapping");
        }
    }
    {
        CacheLateFit fit;
        fit.U=128*gib_bytes; fit.L=64*gib_bytes; fit.A=64*gib_bytes; fit.F=8*gib_bytes;
        fit.E=16U<<20; fit.S=128U<<20; fit.checkpoint=gib_bytes; fit.device_unpaid=gib_bytes;
        fit.device_free=64*gib_bytes; fit.observation_id=1; fit.limits_consistent=fit.aliasing_known=true;
        fit.context_off=fit.context=8192; fit.slots_off=fit.slots=4;
        expect(fit_cache_late(fit).budget==4*gib_bytes,"late min cap and A/16");
        const auto activation=[] {
            Options o; auto& a=*o.cache_activation;
            a.automatic_path=a.prefix_requested=a.automatic_qualified=a.finalized=true;
            a.runner=CacheRunner::single_mtp; a.source="auto"; a.reason="pending-post-residency";
            return o;
        };
        for(unsigned variant=0;variant<8;++variant) {
            auto early=fit;
            if(variant==0) early.A=1;
            if(variant==1) early.F=UINT64_MAX;
            if(variant==2) early.F.reset();
            if(variant==3) early.device_free=0;
            if(variant==4) early.checkpoint=5*gib_bytes;
            if(variant==5) early.age_ns=UINT64_MAX;
            if(variant==6) early.E.reset();
            if(variant==7) early.device_unpaid.reset();
            auto o=activation();
            expect(!prepare_automatic_cache_fit(o,early,false) && !o.cache_activation->B &&
                !o.cache_activation->prefix && o.prefix_cache_host_bytes==0 &&
                o.cache_activation->reason=="pending-post-residency", "pressure/age/unpaid size deferred without B");
            expect(o.cache_activation->decision_phase=="pre-bridge-feasibility","pending phase");
            expect(prepare_automatic_cache_fit(o,fit,true),"fresh readiness recovers early pressure");
            expect(o.cache_activation->B==4*gib_bytes && o.cache_activation->F==fit.F &&
                o.cache_activation->U==fit.U && o.cache_activation->L==fit.L && o.cache_activation->A==fit.A &&
                o.cache_activation->reserve==cache_fit_reserve(fit) &&
                o.cache_activation->decision_phase=="post-residency-ready","ready deciding observation published");
        }
        for(unsigned variant=0;variant<8;++variant) {
            auto early=fit;
            const char* refusal="post-residency-memory-unavailable";
            if(variant==0) early.U.reset();
            if(variant==1) early.limits_consistent=false;
            if(variant==2) early.L.reset();
            if(variant==3) early.A.reset();
            if(variant==4) early.L=*early.U+1;
            if(variant==5) {early.aliasing_known=false;refusal="owner-size-unknown";}
            if(variant==6) {early.context--;refusal="would-reduce-context-or-slots";}
            if(variant==7) {early.slots--;refusal="would-reduce-context-or-slots";}
            auto o=activation();
            expect(!prepare_automatic_cache_fit(o,early,false) && o.cache_activation->reason==refusal,
                "terminal feasibility reason retained");
            expect(!prepare_automatic_cache_fit(o,fit,true) &&
                o.cache_activation->decision_phase=="pre-bridge-feasibility","terminal early decline is not retried");
        }
        for(unsigned variant=0;variant<3;++variant) {
            auto o=activation(); auto late=fit;
            prepare_automatic_cache_fit(o,fit,false);
            if(variant==0) late.A=1;
            if(variant==1) late.device_free=0;
            if(variant==2) late.checkpoint=5*gib_bytes;
            expect(!prepare_automatic_cache_fit(o,late,true) && !o.cache_activation->prefix &&
                o.cache_activation->B==0 && o.prefix_cache_host_bytes==0 &&
                o.cache_activation->decision_phase=="post-residency-ready","late pressure decides decline");
        }
        auto fraction=fit; fraction.A=16*gib_bytes;
        expect(fit_cache_late(fraction).budget==gib_bytes,"fraction uses late availability");
        auto residual=fit; residual.A=13*gib_bytes; residual.F=8*gib_bytes; residual.checkpoint=4096;
        const auto expected=std::min(*residual.A/16,*residual.A-4*gib_bytes-*residual.F-*residual.E-(256U<<20)-*residual.S);
        expect(fit_cache_late(residual).budget==expected/4096*4096,"residual subtraction and 4096 alignment");
        const auto reason=[&](CacheLateFit x,const char* r) { const auto result=fit_cache_late(x);
            expect(result.budget==0 && result.reason==r,r); };
        auto x=fit; x.observation_id=0; reason(x,"post-residency-memory-unavailable");
        x=fit; x.limits_consistent=false; reason(x,"post-residency-memory-unavailable");
        x=fit; x.A.reset(); reason(x,"post-residency-memory-unavailable");
        x=fit; x.age_ns=1000000001; reason(x,"memory-observation-stale");
        x=fit; x.age_ns=1000000000; expect(fit_cache_late(x).budget!=0,"one second observation boundary");
        x=fit; x.aliasing_known=false; reason(x,"owner-size-unknown");
        x=fit; x.F.reset(); reason(x,"owner-size-unknown");
        x=fit; x.E.reset(); reason(x,"owner-size-unknown");
        x=fit; x.F=UINT64_MAX; reason(x,"memory-arithmetic-overflow");
        x=fit; x.L=UINT64_MAX; x.U=UINT64_MAX; reason(x,"memory-arithmetic-overflow");
        x=fit; x.A=1; reason(x,"insufficient-host-headroom");
        x=fit; x.U=70*gib_bytes; reason(x,"insufficient-host-headroom");
        x=fit; x.checkpoint=5*gib_bytes; reason(x,"checkpoint-does-not-fit");
        x=fit; x.context--; reason(x,"would-reduce-context-or-slots");
        x=fit; x.slots--; reason(x,"would-reduce-context-or-slots");
        x=fit; x.device_free=0; reason(x,"staging-does-not-fit");
        x=fit; x.device_free.reset(); reason(x,"staging-does-not-fit");
        x=fit; x.device_unpaid=UINT64_MAX; reason(x,"memory-arithmetic-overflow");
        x=fit; x.explicit_budget=6*gib_bytes;
        expect(fit_cache_late(x).budget==6*gib_bytes,"explicit budget bypasses fraction, retains safety");
        x.explicit_budget=64*gib_bytes; reason(x,"insufficient-host-headroom");
        x=fit; x.U=128*gib_bytes; x.L=123*gib_bytes; x.A=5*gib_bytes; x.F=0; x.E=0; x.device_unpaid=0;
        x.explicit_budget=4*gib_bytes;
        reason(x,"insufficient-host-headroom");
        x.resident_payload=4*gib_bytes;
        expect(fit_cache_late(x).budget==4*gib_bytes,"readiness recheck does not reserve live payload backing twice");
        x.resident_payload=5*gib_bytes; reason(x,"post-residency-memory-unavailable");
        x=fit; x.headroom=64*gib_bytes; reason(x,"insufficient-host-headroom");
    }
    {
        // Legacy explicit reservations can reduce capacity; no OFF plan is computed.
        Options o; o.http_serve=true; o.prefix_cache_host_bytes=2*gib_bytes;
        o.supplied_options.insert("--prefix-cache-host-bytes");
        resolve_session_defaults(o,model,{},128*gib_bytes);
        resolve_prefix_defaults(o,model,{},128*gib_bytes);
        const auto off=resolve_http_slot_capacity(gib_bytes,0,3*gib_bytes,3*gib_bytes,0);
        const auto incumbent=resolve_http_slot_capacity(gib_bytes,0,3*gib_bytes,3*gib_bytes,o.prefix_cache_host_bytes);
        expect(off.slots==4 && incumbent.slots==2,"explicit reservation reduces incumbent slots");
        finalize_cache_family(o,CacheRunner::batch_mtp,incumbent.slots);
        const auto report=cache_activation_json(o);
        expect(report.find("slots_off")->is_null() && report.find("context_off")->is_null(),
            "legacy report has no fabricated OFF capacity");
        expect(report.find("slots_admitted")->as_int()==2 && o.prefix_cache_host_bytes==2*gib_bytes,
            "legacy current capacity and budget survive reporting");
    }
    {
        // Parse first, then resolve; CLI wins env, and either baseline form suppresses raw execution only.
        for(const bool env_baseline:{false,true}) for(const bool env_cert:{false,true})
            for(const bool disk:{false,true}) {
            std::vector<std::string> args={"decoder","MODEL","1"};
            if(!env_baseline) args.push_back("--baseline");
            if(!env_cert) { args.push_back("--prefix-batch-cert"); args.push_back("l0"); }
            args.insert(args.end(),{"--generate","4",
                "--known-prefix-window","--known-prefix-full-rows","--known-prefix-prefill-rows","1024",
                "--known-prefix-workspace-budget-bytes","1048576","--known-prefix-host-metadata-budget-bytes","1048576",
                "--known-prefix-uma-budget-bytes","1048576","--known-prefix-host-headroom-bytes","0",
                "--deferred-moe-status","--virtual-q8-gdn-input","--prefetch-q8-gdn-input",
                "--virtual-q8-gdn-output","--wave8-q8-gdn-input","--wave8-q8-gdn-output",
                "--prefix-cache-host-bytes","1073741824"});
            if(disk) args.insert(args.end(),{"--prefix-cache-dir","cache-placeholder"});
            auto o=parse(args);
            Environment env;
            if(env_baseline) env["STRIX_BASELINE"]="1";
            env["STRIX_PREFIX_BATCH_CERT"]=env_cert?"l0":"l1";
            resolve_session_defaults(o,model,env,128*gib_bytes);
            resolve_prefix_defaults(o,model,env,128*gib_bytes);
            auto report=cache_activation_json(o);
            expect(report.find("batch_certification")->find("policy")->as_string()=="l0-where-eligible",
                "inspection preserves CLI/env L0 request before either baseline form");
            expect(report.find("batch_certification")->find("effective")->is_null(),"inspection cert stays pending");
            const auto legacy_source=o.prefix_cache_source;
            finalize_cache_family(o,CacheRunner::cli,1);
            report=cache_activation_json(o);
            expect(!o.prefix_batch_cert && !prefix_batch_cert_execution_mode(o),"baseline execution still suppressed");
            expect(report.find("batch_certification")->find("effective")->as_string()=="declined" &&
                report.find("batch_certification")->find("reason")->as_string()=="baseline","baseline cert decline disclosed");
            expect(report.find("source")->as_string()=="explicit" && report.find("suppression")->as_string()=="baseline" &&
                o.prefix_cache_source==legacy_source && o.prefix_cache_host_bytes==gib_bytes,
                "surviving baseline storage is explicit without changing legacy fields");
            expect(!disk || o.prefix_cache_dir=="cache-placeholder","baseline disk path survives");
        }
        auto off=parse({"decoder","MODEL","1","--prefix-batch-cert","off"});
        Environment env={{"STRIX_PREFIX_BATCH_CERT","l0"}};
        resolve_session_defaults(off,model,env,128*gib_bytes);resolve_prefix_defaults(off,model,env,128*gib_bytes);
        finalize_cache_family(off,CacheRunner::batch_mtp,2);
        expect(cache_activation_json(off).find("batch_certification")->find("policy")->as_string()=="off",
            "explicit CLI cert OFF wins env ON");
    }
    {
        for(const auto cli:{"omitted","on","off"}) for(const auto env_value:{"0","1"})
            for(const bool suppress_session:{false,true}) {
            std::vector<std::string> args={"decoder","MODEL","1"};
            if(std::string_view(cli)!="omitted") args.insert(args.end(),{"--session-cache",cli});
            if(suppress_session) args.push_back("--baseline");
            auto o=parse(args); Environment env={{"STRIX_SESSION_CACHE",env_value}};
            const bool requested=std::string_view(cli)=="omitted"?std::string_view(env_value)!="0":std::string_view(cli)=="on";
            auto legacy=o;resolve_legacy_session_defaults(legacy,model,env,128*gib_bytes);
            resolve_session_defaults(o,model,env,128*gib_bytes);resolve_prefix_defaults(o,model,env,128*gib_bytes);
            expect(cache_activation_json(o).find("session_requested")->as_bool()==requested,"session request uses CLI over env before admission");
            expect(o.session_cache==legacy.session_cache && o.session_off_reason==legacy.session_off_reason &&
                o.session_management_requested==legacy.session_management_requested,"session reporting preserves legacy admission");
            finalize_cache_family(o,CacheRunner::cli,1);
            expect(cache_activation_json(o).find("session_requested")->as_bool()==requested,"final session request survives suppression");
        }
    }
    {
        for(const auto runner:{CacheRunner::cli,CacheRunner::target,CacheRunner::single_mtp,CacheRunner::batch_mtp}) {
            Options o;o.http_serve=runner!=CacheRunner::cli;o.known_prefix_prefill_rows=2048;
            resolve_session_defaults(o,model,{},128*gib_bytes);resolve_prefix_defaults(o,model,{},128*gib_bytes);
            expect(cache_activation_json(o).find("batch_certification")->find("reason")->as_string()=="pending-post-residency",
                "inspection retains pending automatic certification");
            finalize_cache_family(o,runner,runner==CacheRunner::batch_mtp?4:1);
            const auto cert=cache_activation_json(o).find("batch_certification")->dump();
            expect(cert.find("pending")==std::string::npos,"finalized runner has no pending certification reason");
            expect(o.cache_activation->cert_reason==(runner==CacheRunner::cli?"no-reuse-route":
                runner==CacheRunner::batch_mtp?"batch-scheduler-family-unqualified":
                runner==CacheRunner::single_mtp?"mtp-contract-unqualified":"not-applicable-non-batch"),
                "finalized certification explains runner decline");
        }
    }
    {
        const auto root=std::filesystem::temp_directory_path()/"qwen-cache-cgroup-fixture";
        std::filesystem::create_directories(root/"parent"/"leaf");
        const auto write=[&](const char* path,const char* value) { std::ofstream(root/path)<<value; };
        write("parent/memory.max","max\n"); write("parent/memory.current","40\n");
        write("parent/leaf/memory.max","max\n"); write("parent/leaf/memory.current","20\n");
        const auto observe=[&](std::string groups="0::/parent/leaf\n") {
            PrefillMemory m;m.usable_ceiling=1000;m.cache_available=800;
            intersect_cache_cgroup_limits(m,groups,root,"1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw",
                "cgroup:[4026531835]");return m;
        };
        expect(observe().cache_limits_known,"v2 root without controller files is unlimited");
        std::filesystem::remove(root/"parent/memory.current");
        expect(!observe().cache_limits_known,"missing non-root current is unknown");
        write("parent/memory.current","40\n");
        std::filesystem::remove(root/"parent/leaf/memory.max");
        expect(!observe().cache_limits_known,"missing non-root maximum is unknown");
        write("parent/leaf/memory.max","500\n");write("parent/memory.max","600\n");
        auto m=observe();
        expect(m.cache_limits_known && m.usable_ceiling==500 && m.cache_available==480,
            "ancestor and leaf capacity/free limits intersect");
        write("parent/memory.current","590\n");m=observe();
        expect(m.cache_limits_known && m.usable_ceiling==500 && m.cache_available==10,
            "ancestor free headroom intersects independently");
        write("parent/memory.current","601\n");
        expect(!observe().cache_limits_known,"used above maximum declines");
        expect(!observe("1:memory:/parent/leaf\n").cache_limits_known,"v1 stays unknown");
        expect(!observe("0::/../escape\n").cache_limits_known,"unknown hierarchy cannot escape root");
        expect(!observe("1:cpu,memory:/x\n0::/\n").cache_limits_known,"hybrid memory ownership unknown");
        expect(observe("0::/\n").cache_limits_known,"initial namespace global root needs no controller files");
        PrefillMemory global; global.usable_ceiling=1000; global.cache_available=800;
        intersect_cache_cgroup_limits(global,"0::/\n",root,
            "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw","cgroup:[4026531835]");
        expect(global.cache_limits_known,"proven global root without files known");
        write("memory.max","120\n"); write("memory.current","119\n");
        auto limited=observe("0::/\n");
        expect(!limited.cache_limits_known,"controller files at purported global root are inconsistent");
        write("parent/memory.current","40\n");
        limited=observe();
        expect(!limited.cache_limits_known,"unexpected root controller files decline non-root membership too");
        PrefillMemory delegated;delegated.usable_ceiling=1000;delegated.cache_available=800;
        intersect_cache_cgroup_limits(delegated,"0::/delegated/parent/leaf\n",root,
            "1 0 0:1 /delegated /sys/fs/cgroup rw - cgroup2 cgroup rw");
        expect(!delegated.cache_limits_known,"subtree mount conceals limiting ancestors");
        write("memory.max","max\n");
        intersect_cache_cgroup_limits(delegated,"0::/delegated\n",root,
            "1 0 0:1 /delegated /sys/fs/cgroup rw - cgroup2 cgroup rw","cgroup:[4026531835]");
        expect(!delegated.cache_limits_known,"unlimited visible subtree cannot prove hidden parent headroom");
        intersect_cache_cgroup_limits(delegated,"0::/parent/leaf\n",root,
            "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw","cgroup:[1234]");
        expect(!delegated.cache_limits_known,"private namespace remains unknown even with shared container PID 1");
        intersect_cache_cgroup_limits(delegated,"0::/parent/leaf\n",root,
            "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw");
        expect(!delegated.cache_limits_known,"missing namespace evidence declines non-root membership too");
        PrefillMemory unmapped;
        intersect_cache_cgroup_limits(unmapped,"0::/\n",root);
        expect(!unmapped.cache_limits_known,"missing mount proof declines even with readable root files");
        std::filesystem::remove_all(root);
    }
    {
        Options o; o.cache_activation->automatic_path=true;
        o.cache_activation->automatic_qualified=true; o.cache_activation->source="auto";
        o.cache_activation->finalized=true;
        CacheLateFit x;x.U=128849018880ULL;x.L=100000000000ULL;x.A=28849018880ULL;
        x.F=35181467972ULL;x.E=2001312;x.S=178790912;
        x.observation_id=1;x.age_ns=12345;x.aliasing_known=true;
        x.checkpoint=1;x.device_unpaid=x.F;x.device_free=100000000000ULL;
        expect(!apply_automatic_cache_fit(o,x),"unknown limits decline with inputs");
        auto ancestry=x; ancestry.cgroup_ancestry_unproven=true;
        expect(cache_feasibility_decline(ancestry)=="cgroup-ancestry-unproven","named ancestry feasibility refusal");
        expect(!apply_automatic_cache_fit(o,ancestry) && o.cache_activation->reason=="cgroup-ancestry-unproven",
            "named ancestry final refusal");
        apply_automatic_cache_fit(o,x);
        auto report=cache_activation_json(o);
        expect(report.find("U")->as_int()==128849018880LL && report.find("L")->as_int()==100000000000LL &&
            report.find("A")->as_int()==28849018880LL && report.find("reserve")->as_int()==39925662948LL &&
            !report.find("limits_consistent")->as_bool() && report.find("observation_age_ns")->as_int()==12345,
            "late fit reports intersected inputs and checked reserve on decline");
        x.limits_consistent=true;
        expect(fit_cache_late(x).reason=="insufficient-host-headroom","real commitment needs enough available capacity");
        x.L=80000000000ULL;x.A=48849018880ULL;
        expect(fit_cache_late(x).budget!=0,"real commitment fits when post-residency capacity suffices");
        x.F=UINT64_MAX;
        expect(!cache_fit_reserve(x),"report reserve overflow stays unknown");
        Options explicit_path;
        expect(!cache_activation_json(explicit_path).find("U"),"explicit report has no new observation fields");
    }
    {
        // Caller accounting seam: native/shared setup moves from F to observed L.
        const std::uint64_t native=16*gib_bytes, transient=gib_bytes;
        CacheLateFit fit;fit.U=128*gib_bytes;fit.L=96*gib_bytes;fit.A=32*gib_bytes;
        fit.F=cache_unpaid_commitment(false,native+transient,transient);
        fit.E=fit.S=0;fit.device_unpaid=0;fit.device_free=32*gib_bytes;
        fit.checkpoint=gib_bytes;fit.observation_id=1;fit.limits_consistent=fit.aliasing_known=true;
        const auto fitted=fit_cache_late(fit).budget;
        expect(fitted!=0,"CLI initial accounting fits");
        fit.explicit_budget=fitted;fit.L=*fit.L+native;fit.A=*fit.A-native;
        fit.F=cache_unpaid_commitment(true,native+transient,transient);
        expect(fit.F==transient && fit_cache_late(fit).budget==fitted,
            "CLI ready sample preserves fitted cap after native setup becomes resident");
        fit.F=native+transient;
        expect(!fit_cache_late(fit).budget,"duplicated native commitment reproduces false readiness decline");
        expect(!cache_unpaid_commitment(true,native+transient,{}),"unknown ready inventory stays unknown");
    }
    {
        const auto& row=automatic_prefix_auto_route_qualifications[0];
        expect(row.identity.complete(),"single candidate binds every identity field");
        const auto match=[&](const CacheFamilyIdentity& id) {
            return row.matches(CacheRunner::single_mtp,1,16384,2048,id);
        };
        expect(match(row.identity),"single family exact identity matches");
        {
            // STRIX_QPF2K_SCHED bit 2 is inert on this route (the MTP bridge never defers a block): the default 14 and
            // an explicit 14 compare as 12; an explicit 13 still differs. Nothing is re-qualified.
            const auto saved=qpf2k::defaulted_items;
            for(const unsigned defaulted:{qpf2k::bit(qpf2k::Item::sched),0U}) {
                qpf2k::defaulted_items=defaulted;
                auto id=row.identity; id.scheduler.schedule=14;
                expect(!match(id),"raw schedule 14 differs from the qualified identity");
                for(const auto runner:{CacheRunner::single_mtp,CacheRunner::batch_mtp,CacheRunner::target})
                    expect(effective_cache_family(id,runner).scheduler.schedule==12,"non-deferring runners mask bit 2");
                expect(match(effective_cache_family(id,CacheRunner::single_mtp)),
                    defaulted?"default schedule 14 is admitted as 12 on the HTTP prefix-cache route":
                        "explicit schedule 14 is admitted on the HTTP prefix-cache route (bit 2 inert)");
                id.scheduler.schedule=13;
                expect(!match(effective_cache_family(id,CacheRunner::single_mtp)),"explicit schedule 13 still declines");
                // CLI: a defaulted schedule drops bit 2 with a prefix cache; an explicit 14 keeps it.
                id.scheduler.schedule=14;
                expect(effective_cache_family(id,CacheRunner::cli).scheduler.schedule==(defaulted?12U:14U),
                    "CLI family records the schedule the cached run executes");
            }
            qpf2k::defaulted_items=saved;
        }
        const auto mismatch=[&](const CacheFamilyIdentity& id) { expect(!match(id),"changed identity cannot match single family"); };
        { auto id=row.identity; id.model+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.layout+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.tokenizer+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.sidecar+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.golden+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.replay+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.build_family+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.gfx+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.numerical_contract+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.kv_precision+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.recurrent_precision+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.prefill_profile+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.decode_profile+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.output_policy+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.profile_pair+="-mismatch"; mismatch(id); }
        { auto id=row.identity; id.tails.push_back(1); mismatch(id); }
        { auto id=row.identity; ++id.scheduler.schedule; mismatch(id); }
        { auto id=row.identity; ++id.scheduler.pipe; mismatch(id); }
        { auto id=row.identity; ++id.scheduler.lookahead; mismatch(id); }
        { auto id=row.identity; ++id.scheduler.short_rows; mismatch(id); }
        { auto id=row.identity; ++id.scheduler.service_ms; mismatch(id); }
        { auto id=row.identity; ++id.scheduler.prompt_share; mismatch(id); }
        { auto id=row.identity; ++id.scheduler.nest_rows; mismatch(id); }
        { auto id=row.identity; ++id.scheduler.interleave; mismatch(id); }
        { auto id=row.identity; ++id.scheduler.group_rows; mismatch(id); }
        { auto id=row.identity; id.scheduler.groups=!id.scheduler.groups; mismatch(id); }
        { auto id=row.identity; id.scheduler.cohort=!id.scheduler.cohort; mismatch(id); }
        { auto id=row.identity; id.scheduler.small=!id.scheduler.small; mismatch(id); }
        { auto id=row.identity; id.scheduler.group_owner=!id.scheduler.group_owner; mismatch(id); }
        { auto id=row.identity; id.scheduler.windows=!id.scheduler.windows; mismatch(id); }
        { auto id=row.identity; id.scheduler.cancel=!id.scheduler.cancel; mismatch(id); }
        { auto id=row.identity; id.scheduler.priming=!id.scheduler.priming; mismatch(id); }
        { auto id=row.identity; id.scheduler.worker=!id.scheduler.worker; mismatch(id); }
        { auto id=row.identity; id.scheduler.tiled_start=!id.scheduler.tiled_start; mismatch(id); }
        { auto id=row.identity; id.scheduler.merge=!id.scheduler.merge; mismatch(id); }
        { auto id=row.identity; id.scheduler.balance=!id.scheduler.balance; mismatch(id); }
        { auto id=row.identity; id.scheduler.shared+="-mismatch"; mismatch(id); }
        for(auto context:{1U,2048U,8192U,10400U,16384U})
            expect(row.matches(CacheRunner::single_mtp,1,context,2048,row.identity),"candidate context interval");
        for(auto context:{0U,16385U}) expect(!row.matches(CacheRunner::single_mtp,1,context,2048,row.identity),"context outside interval");
        expect(!row.matches(CacheRunner::single_mtp,2,16384,2048,row.identity),"single runner has exactly N=1");
        expect(!row.matches(CacheRunner::single_mtp,0,16384,2048,row.identity),"zero slots cannot match");
        expect(!row.matches(CacheRunner::single_mtp,1,16384,8192,row.identity),"different tile declines");
        for(auto runner:{CacheRunner::cli,CacheRunner::target,CacheRunner::batch_mtp})
            expect(!row.matches(runner,1,16384,2048,row.identity),"batch N=1 and other runners cannot match single");
        const auto candidate=[&](int http_batch=0) {
            Options o; o.http_serve=true; o.http_batch=http_batch; o.mtp.enabled=true;
            o.context_tokens=16384; o.known_prefix_prefill_rows=2048;
            o.mtp.enabled=true;
            o.mtp.metadata_only=false;
            o.mtp.draft_chain=true;
            o.mtp.draft_exact=false;
            o.mtp.draft_q4=true;
            o.mtp.draft_context_vocab=4096;
            o.mtp.candidate_depth=7;
            o.mtp.window_rows=8;
            o.mtp.spec_min_p=0.7;
            o.mtp.window_prefix=true;
            o.mtp.window_expert_union=true;
            o.mtp.window_hyper_mix=true;
            o.mtp.window_qsa_rows=true;
            o.mtp.window_arena_bytes=2147483648ULL;
            o.mtp.window_capture_bytes=67108864;
            o.mtp.window_checkpoint_bytes=8589934592ULL;
            o.mtp.window_total_bytes=17179869184ULL;
            o.mtp.window_history_intervals=8192;
            o.mtp.window_expert_union_bytes=65536;
            o.mtp.calibration_rows=0;
            o.mtp.policy_identity=1;
            o.mtp.ngram_draft=false;
            o.mtp.ngram_off=false;
            o.mtp.ngram_context=false;
            o.mtp.ngram_copy_edit=false;
            o.mtp.ngram_wide_only=false;
            o.mtp.sample_options=false;
            o.mtp.sample_temperature=0;
            o.mtp.sample_top_p=1;
            o.mtp.sample_min_p=0;
            o.mtp.sample_top_k=0;
            o.mtp.sample_seed=0;
            o.mtp.sample_speculation=true;
            o.mtp.sample_timing_schedule=false;
            o.mtp.sample_repeats=1;
            o.mtp.max_hidden_error=0;
            o.mtp.max_logit_error=0;
            o.mtp.sidecar_bytes=2786568256ULL;
            o.mtp.header_bytes=33554432;
            o.mtp.sidecar_retained_bytes=3221225472ULL;
            o.mtp.sidecar_setup_bytes=4294967296ULL;
            o.mtp.golden_max_bytes=6442450944ULL;
            o.mtp.replay_host_bytes=6442450944ULL;
            o.mtp.runtime_bytes=134217728;
            o.mtp.target_capture_bytes=268435456;
            o.mtp.window_expert_union_bytes_explicit=true;
            o.mtp.window_rows_set={2,4,8};o.mtp.draft_vocab="auto:ASSETS";o.mtp.sidecar_residency="register";

            resolve_session_defaults(o,model,{},128*gib_bytes);resolve_prefix_defaults(o,model,{},128*gib_bytes);
            finalize_cache_family(o,CacheRunner::single_mtp,1,row.identity);
            return o;
        };
#ifndef _WIN32
        {
            const char* name="STRIX_PROMPT_STATE_DIGEST";
            const char* old=std::getenv(name);
            const std::optional<std::string> saved=old?std::optional<std::string>(old):std::nullopt;
            ::unsetenv(name);
            auto o=candidate();
            const auto cli=cache_family_execution_cli(o);
            const auto off=cache_family_execution_signature(cli,false,false);
            const auto storage=session_numerical_config(cli);
            expect(o.cache_activation->automatic_qualified==row.qualified(),"unset observer retains production admission");
            ::setenv(name,"1",1);
            const auto on=cache_family_execution_signature(cli,false,false);
            expect(on!=off && session_numerical_config(cli)!=storage,"prompt observer changes family and storage identity");
            auto identity=row.identity;
            identity.numerical_contract=model_content_identity_hex(on);
            finalize_cache_family(o,CacheRunner::single_mtp,1,identity);
            expect(!o.cache_activation->automatic_qualified && o.cache_activation->reason=="family-unqualified",
                "prompt observer declines production family by name");
            ::unsetenv(name);
            expect(cache_family_execution_signature(cli,false,false)==off,"unset production identity is stable");
            if(saved) ::setenv(name,saved->c_str(),1);else ::unsetenv(name);
        }
        {
            // STRIX_QPF2K_SCHED bit 2 on a runner that never defers: the numerical contract binds the executed
            // schedule, so 14 there equals 12 (the qualified contract); 13 and a deferring runner still differ.
            const char* name="STRIX_QPF2K_SCHED";
            const char* old=std::getenv(name);
            const std::optional<std::string> saved=old?std::optional<std::string>(old):std::nullopt;
            const auto cli=cache_family_execution_cli(candidate());
            ::setenv(name,"12",1);
            const auto twelve=cache_family_execution_signature(cli,false,false,{},false);
            expect(twelve==cache_family_execution_signature(cli,false,false),"schedule 12 has no bit 2 to mask");
            ::setenv(name,"14",1);
            expect(cache_family_execution_signature(cli,false,false,{},false)==twelve,
                "schedule 14 on a non-deferring runner has the schedule-12 numerical contract");
            expect(cache_family_execution_signature(cli,false,false,{},true)!=twelve,
                "a deferring runner binds schedule 14");
            ::setenv(name,"13",1);
            expect(cache_family_execution_signature(cli,false,false,{},false)!=twelve,"schedule 13 still differs");
            expect(!cache_runner_defers(CacheRunner::single_mtp) && !cache_runner_defers(CacheRunner::batch_mtp) &&
                !cache_runner_defers(CacheRunner::target),"HTTP/MTP runners never defer");
            if(saved) ::setenv(name,saved->c_str(),1);else ::unsetenv(name);
        }
#endif
        {auto o=candidate();o.mtp.window_expert_union_bytes_explicit=false;
            finalize_cache_family(o,CacheRunner::single_mtp,1,row.identity);
            expect(o.cache_activation->reason=="mtp-contract-unqualified","resolved explicit union budget mode bound");}
        for(const auto& [changed_name,unused_digest]:cache_mtp_vocabulary_digests) {
            (void)unused_digest;
            const auto hash=[&](const std::filesystem::path& file) {
                for(const auto& [name,digest]:cache_mtp_vocabulary_digests)
                    if(file.filename()==name) return std::string(name==changed_name?"changed":digest);
                return std::string{};
            };
            expect(!cache_mtp_vocabulary_qualified(candidate(),hash),"every auto vocabulary content mutation declines");
        }
        expect(cache_mtp_vocabulary_qualified(candidate(),[](const std::filesystem::path& file) {
            for(const auto& [name,digest]:cache_mtp_vocabulary_digests) if(file.filename()==name) return std::string(digest);
            return std::string{};
        }),"bound vocabulary content admitted independently of path");
        expect(!cache_mtp_vocabulary_qualified(candidate()),"missing vocabulary assets decline");
        expect(cache_mtp_contract_qualified(candidate()),"bound production MTP contract admitted");
        if(!row.cost_qualified) expect(candidate().cache_activation->reason=="cost-unqualified",
            "matched held row names the independent cost gate");
        const auto reject_mtp=[&](Options o) {
            finalize_cache_family(o,CacheRunner::single_mtp,1,row.identity);
            expect(!o.cache_activation->automatic_qualified && o.cache_activation->reason=="mtp-contract-unqualified",
                "resolved MTP mutation declines by name");
        };
        {auto o=candidate();o.mtp.enabled=!o.mtp.enabled;reject_mtp(o);}
        {auto o=candidate();o.mtp.metadata_only=!o.mtp.metadata_only;reject_mtp(o);}
        {auto o=candidate();o.mtp.draft_chain=!o.mtp.draft_chain;reject_mtp(o);}
        {auto o=candidate();o.mtp.draft_exact=!o.mtp.draft_exact;reject_mtp(o);}
        {auto o=candidate();o.mtp.draft_q4=!o.mtp.draft_q4;reject_mtp(o);}
        {auto o=candidate();o.mtp.draft_context_vocab=o.mtp.draft_context_vocab+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.candidate_depth=o.mtp.candidate_depth+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_rows=o.mtp.window_rows+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.spec_min_p=o.mtp.spec_min_p+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_prefix=!o.mtp.window_prefix;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_expert_union=!o.mtp.window_expert_union;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_hyper_mix=!o.mtp.window_hyper_mix;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_qsa_rows=!o.mtp.window_qsa_rows;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_arena_bytes=o.mtp.window_arena_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_capture_bytes=o.mtp.window_capture_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_checkpoint_bytes=o.mtp.window_checkpoint_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_total_bytes=o.mtp.window_total_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_history_intervals=o.mtp.window_history_intervals+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_expert_union_bytes=o.mtp.window_expert_union_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.calibration_rows=o.mtp.calibration_rows+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.policy_identity=o.mtp.policy_identity+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.ngram_draft=!o.mtp.ngram_draft;reject_mtp(o);}
        {auto o=candidate();o.mtp.ngram_off=!o.mtp.ngram_off;reject_mtp(o);}
        {auto o=candidate();o.mtp.ngram_context=!o.mtp.ngram_context;reject_mtp(o);}
        {auto o=candidate();o.mtp.ngram_copy_edit=!o.mtp.ngram_copy_edit;reject_mtp(o);}
        {auto o=candidate();o.mtp.ngram_wide_only=!o.mtp.ngram_wide_only;reject_mtp(o);}
        {auto o=candidate();o.mtp.sample_options=!o.mtp.sample_options;reject_mtp(o);}
        {auto o=candidate();o.mtp.sample_temperature=o.mtp.sample_temperature+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.sample_top_p=o.mtp.sample_top_p+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.sample_min_p=o.mtp.sample_min_p+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.sample_top_k=o.mtp.sample_top_k+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.sample_seed=o.mtp.sample_seed+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.sample_speculation=!o.mtp.sample_speculation;reject_mtp(o);}
        {auto o=candidate();o.mtp.sample_timing_schedule=!o.mtp.sample_timing_schedule;reject_mtp(o);}
        {auto o=candidate();o.mtp.sample_repeats=o.mtp.sample_repeats+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.max_hidden_error=o.mtp.max_hidden_error+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.max_logit_error=o.mtp.max_logit_error+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.sidecar_bytes=o.mtp.sidecar_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.header_bytes=o.mtp.header_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.sidecar_retained_bytes=o.mtp.sidecar_retained_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.sidecar_setup_bytes=o.mtp.sidecar_setup_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.golden_max_bytes=o.mtp.golden_max_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.replay_host_bytes=o.mtp.replay_host_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.runtime_bytes=o.mtp.runtime_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.target_capture_bytes=o.mtp.target_capture_bytes+1;reject_mtp(o);}
        {auto o=candidate();o.mtp.window_rows_set={2,4};reject_mtp(o);}
        {auto o=candidate();o.mtp.copy_wide_rows={16};reject_mtp(o);}
        {auto o=candidate();o.mtp.draft_vocab="other.bin";reject_mtp(o);}
        {auto o=candidate();o.mtp.draft_profile="base";reject_mtp(o);}
        {auto o=candidate();o.mtp.cohort_mode="sequential";reject_mtp(o);}
        {auto o=candidate();o.mtp.cohort.push_back({});reject_mtp(o);}
        {auto o=candidate();o.mtp.golden_dump="dump";reject_mtp(o);}
        {auto o=candidate();o.mtp.sidecar_residency="copy";reject_mtp(o);}
        {auto o=candidate();o.rtx_state_import=true;reject_mtp(o);}
        {auto o=candidate();o.rtx_state_import_decode_threads=0;reject_mtp(o);}
        CacheLateFit fit;fit.U=128*gib_bytes;fit.L=64*gib_bytes;fit.A=64*gib_bytes;
        fit.F=gib_bytes;fit.E=16U<<20;fit.S=128U<<20;fit.checkpoint=gib_bytes;
        fit.device_unpaid=gib_bytes;fit.device_free=64*gib_bytes;fit.observation_id=1;
        fit.limits_consistent=fit.aliasing_known=true;fit.context_off=fit.context=16384;fit.slots_off=fit.slots=1;
        for(int requested_batch:{0,-1}) {
            auto o=candidate(requested_batch);
            expect(!o.cache_activation->prefix && !o.prefix_cache_host_bytes,"identity match alone does not activate storage");
            expect(apply_automatic_cache_fit(o,fit)==row.qualified(),"explicit single and auto fallback activate only after fit");
            expect(o.cache_activation->prefix==row.qualified() && !o.cache_activation->session && !o.cache_activation->cert_effective,
                "single prefix admission never activates session or batch cert");
        }
        if(row.qualified()) {
            const auto decline=[&](CacheLateFit x,const char* reason) {
                auto o=candidate();expect(!apply_automatic_cache_fit(o,x) && !o.cache_activation->prefix && !o.prefix_cache_host_bytes &&
                    o.cache_activation->reason==reason,reason);
            };
            auto x=fit;x.observation_id=0;decline(x,"post-residency-memory-unavailable");
            x=fit;x.age_ns=1000000001;decline(x,"memory-observation-stale");
            x=fit;x.aliasing_known=false;decline(x,"owner-size-unknown");
            x=fit;x.context--;decline(x,"would-reduce-context-or-slots");
            x=fit;x.F=UINT64_MAX;decline(x,"memory-arithmetic-overflow");
            x=fit;x.A=0;decline(x,"insufficient-host-headroom");
            x=fit;x.device_free=0;decline(x,"staging-does-not-fit");
            x=fit;x.checkpoint=5*gib_bytes;decline(x,"checkpoint-does-not-fit");
        }
        for(auto runner:{CacheRunner::cli,CacheRunner::target,CacheRunner::batch_mtp}) {
            auto o=candidate();finalize_cache_family(o,runner,1,row.identity);
            expect(!apply_automatic_cache_fit(o,fit),"no other runner admits even the exact single identity");
        }
        {
            auto o=candidate();o.tokens={1};
            auto batch=candidate(8);batch.tokens={1};
            expect(qsa_mtp_pool_decline(o)==nullptr &&
                std::string_view(qsa_mtp_pool_decline(batch))=="unqualified-route-requires-STRIX_QSA_MTP_POOL=1" &&
                ple_reclaim_inactive_reason(o,HipModelResidencyPolicy{},false)=="http-serving" &&
                !mtp_prompt_phase_separation(o),
                "QSA pool defaults on for single HTTP only; rolling PLE and phase retirement exclude default HTTP");
        }
        for(const bool bundle_selected:{false,true}) {
            auto o=candidate();o.rotary=validate_qwen_rotary_request(2,true,524288,false);
            if(bundle_selected) {o.mtp_bundle_path="MODEL.gguf.mtp-bundle.json";o.yarn_decode_source=o.mtp_bundle_path;}
            finalize_cache_family(o,CacheRunner::single_mtp,1,row.identity);
            expect(!apply_automatic_cache_fit(o,fit) && o.cache_activation->reason=="rotary-cache-unqualified",
                "explicit and bundle-selected YaRN decline even with a supplied native identity");
        }
        for(const auto& env: {Environment{{"STRIX_BASELINE","1"}},Environment{{"STRIX_QPF2K_DEFAULTS","0"}}}) {
            Options o;o.http_serve=true;o.http_batch=0;o.mtp.enabled=true;
            o.context_tokens=16384;o.known_prefix_prefill_rows=2048;
            resolve_session_defaults(o,model,env,128*gib_bytes);resolve_prefix_defaults(o,model,env,128*gib_bytes);
            finalize_cache_family(o,CacheRunner::single_mtp,1,row.identity);
            expect(!apply_automatic_cache_fit(o,fit) && !o.cache_activation->prefix,"suppression defeats the bound family");
        }
        auto report=parse({"decoder","MODEL","1","--print-cache-family"});
        expect(report.print_cache_family && !report.route_owned,"identity reporting does not own numerical routing");
    }
    for(const auto& registry:{std::span<const CacheQualification>(automatic_prefix_auto_route_qualifications),
            std::span<const CacheQualification>(automatic_prefix_batch_cert_route_qualifications),
            std::span<const CacheQualification>(automatic_session_route_qualifications)})
        for(const auto& row:registry) {
            if(&row==&automatic_prefix_auto_route_qualifications[0]) {
                expect(row.gpu_qualified && row.resource_qualified,"single row retains equality and capacity qualification");
                expect(row.qualified()==(row.gpu_qualified && row.cost_qualified && row.resource_qualified && !row.evidence_id.empty()),
                    "all three independent row bits required");
                auto held=row;held.cost_qualified=false;
                expect(!held.qualified(),"equality and capacity cannot substitute for cost evidence");
            } else expect(!row.gpu_qualified && !row.cost_qualified && !row.resource_qualified && !row.identity.complete(),"every other automatic family stays unqualified");
        }
    expect(parse({"decoder","MODEL","1","--prefix-cache-host-bytes","42","--prefix-cache","off"}).route_owned,
        "explicit budget plus OFF retains public-main route ownership");
    {
        Options o;o.http_serve=true;o.http_batch=0;o.mtp.enabled=true;
        o.known_prefix_window=o.known_prefix_full_rows=true;o.known_prefix_prefill_rows=2048;
        const auto cli=cache_family_execution_cli(o);
        auto operational=o;operational.http_port=32123;operational.http_bind="0.0.0.0";operational.http_model_id="another-label";
        expect(cli==cache_family_execution_cli(operational),"family omits only operational listener fields");
        const auto signature=[&](const Options& x) {return cache_family_execution_signature(cache_family_execution_cli(x),false,false);};
        const auto original=signature(o);
        auto numerical=o;numerical.prefill_numerical_profile=DecodeNumericalProfile::fast_prefill_f16wmma_v1;
        expect(signature(numerical)!=original,"fast prefill remains a different family");
#ifndef _WIN32
        for(const char* name:{"STRIX_PROMPT_STATE_DIGEST","STRIX_PREFILL_LOGITS_DIGEST",
                "STRIX_PREFIX_CACHE_DIGEST","STRIX_SERIAL_LOGITS_DIGEST","STRIX_FINAL_STATE_DIGEST",
                "STRIX_QPF2K_SCHED","STRIX_UNKNOWN_NUMERICAL_CONTROL"}) {
            const char* old=std::getenv(name);const std::optional<std::string> saved=old?std::optional<std::string>(old):std::nullopt;
            ::unsetenv(name);
            const auto family=session_family_numerical_config(cli);
            const auto storage=session_numerical_config(cli);
            ::setenv(name,"1",1);
            const bool excluded=std::string_view(name)=="STRIX_PREFILL_LOGITS_DIGEST";
            expect((family==session_family_numerical_config(cli))==excluded,"family filters only audited observers");
            expect(storage!=session_numerical_config(cli),"storage namespace retains every diagnostic");
            if(saved) ::setenv(name,saved->c_str(),1);else ::unsetenv(name);
        }
#endif
    }
    for(const auto* flag:{"--prefix-cache-inflight","--prefix-cache-disk","--prefix-cache-durable"}) {
        expect(operational_session_option(flag),"prefix reuse switch is operational");
        const auto disabled=parse({"decoder","MODEL","1",flag,"off"});
        expect(std::string_view(flag)=="--prefix-cache-inflight"?!disabled.prefix_cache_inflight:
            std::string_view(flag)=="--prefix-cache-disk"?!disabled.prefix_cache_disk:!disabled.prefix_cache_durable,
            "prefix reuse CLI off switch");
        auto selected=parse({"decoder","MODEL","1",flag,"on"});
        apply_prefix_cache_environment(selected,[](const char* name)->std::optional<std::string_view> {
            if(std::string_view(name)=="STRIX_PREFIX_CACHE_INFLIGHT" ||
                std::string_view(name)=="STRIX_PREFIX_CACHE_DISK" || std::string_view(name)=="STRIX_PREFIX_CACHE_DURABLE") return "0";
            return {};
        });
        expect(std::string_view(flag)=="--prefix-cache-inflight"?selected.prefix_cache_inflight:
            std::string_view(flag)=="--prefix-cache-disk"?selected.prefix_cache_disk:selected.prefix_cache_durable,
            "prefix reuse CLI wins environment");
        bool invalid_cli=false;
        try {(void)parse({"decoder","MODEL","1",flag,"maybe"});} catch(const std::invalid_argument&) {invalid_cli=true;}
        expect(invalid_cli,"invalid prefix reuse CLI refused");
    }
    {
        const auto env=requested_prefill_environment([](const char* name)->const char* {
            return std::string_view(name)=="STRIX_PREFIX_FAIR_RECLAIM_IDLE"?"0":nullptr;
        });
        for (const auto value : {"0", "1"}) {
            Options options;
            apply_prefix_cache_environment(options,[&](const char* name)->std::optional<std::string_view> {
                return std::string_view(name)=="STRIX_PREFIX_FAIR_RECLAIM_IDLE" ?
                    std::optional<std::string_view>{value} : std::nullopt;
            });
            expect(options.prefix_fair_reclaim_idle==(std::string_view(value)!="0"),
                "app resolves fair retention policy before store construction");
        }
        expect(env.contains("STRIX_PREFIX_FAIR_RECLAIM_IDLE"),"inspection collects fair retention rollback");
        expect(!explicit_cache_configuration(Options{},env),"fair rollback preserves automatic cache routing");
        expect(yarn_environment_refusal(env,2).empty(),"fair retention rollback performs no rotary work");
    }
    for(const auto* key:{"STRIX_PREFIX_CACHE_INFLIGHT","STRIX_PREFIX_CACHE_DISK","STRIX_PREFIX_CACHE_DURABLE"}) {
        expect(yarn_environment_refusal(Environment{{key,"0"}},2).empty(),"reuse switch performs no rotary work");
        Options invalid;
        bool refused=false;
        try {
            apply_prefix_cache_environment(invalid,[&](const char* name)->std::optional<std::string_view> {
                if(std::string_view(name)==key) return "maybe";
                return {};
            });
        } catch(const std::invalid_argument&) {refused=true;}
        expect(refused,"invalid prefix reuse environment refused");
    }
    {
        Options defaults;
        expect(defaults.prefix_cache_inflight && defaults.prefix_cache_disk && defaults.prefix_cache_durable,
            "reuse policies default on within admitted routes");
        const auto env=requested_prefill_environment([](const char* name)->const char* {
            return std::string_view(name).starts_with("STRIX_PREFIX_CACHE_")?"0":nullptr;
        });
        expect(env.contains("STRIX_PREFIX_CACHE_INFLIGHT") && env.contains("STRIX_PREFIX_CACHE_DISK") &&
            env.contains("STRIX_PREFIX_CACHE_DURABLE"),"resolver collects reuse environment");
        apply_prefix_cache_environment(defaults,[](const char* name)->std::optional<std::string_view> {
            if(std::string_view(name)=="STRIX_PREFIX_CACHE_INFLIGHT" ||
                std::string_view(name)=="STRIX_PREFIX_CACHE_DISK" || std::string_view(name)=="STRIX_PREFIX_CACHE_DURABLE") return "off";
            return {};
        });
        expect(!defaults.prefix_cache_inflight && !defaults.prefix_cache_disk && !defaults.prefix_cache_durable,
            "environment off switches");
        auto reuse_baseline=parse({"decoder","MODEL","1","--baseline"});
        expect(!reuse_baseline.prefix_cache_inflight && !reuse_baseline.prefix_cache_durable,"baseline suppresses new reuse policy");
        Options admitted;admitted.http_serve=true;admitted.http_batch=2;admitted.prefix_cache_host_bytes=4096;
        admitted.prefix_cache_dir="prefix-cache";admitted.prefix_batch_cert=1;
        auto caps=serve::Json::object();prefix_cache_capabilities(caps,admitted);
        const auto* cache=caps.find("prefix_cache");
        expect(cache && cache->find("certified_disk")->as_bool() && cache->find("inflight")->as_bool() &&
            cache->find("durable")->as_bool(),"capabilities report guarded reuse policy");
        EffectiveConfig c;c.options=admitted;
        const auto effective=serve::parse_json(effective_config_json(c));
        expect(effective.find("prefix_cache")->find("inflight")->as_bool(),"effective config reports joining");
        admitted.prefix_cache_disk=false;admitted.prefix_cache_inflight=false;
        caps=serve::Json::object();prefix_cache_capabilities(caps,admitted);
        expect(!caps.find("prefix_cache")->find("certified_disk")->as_bool() &&
            !caps.find("prefix_cache")->find("inflight")->as_bool(),"capabilities report switches off");
    }
    {
        // In-flight joining follows the cache's own enablement: default ON with
        // the cache ON, OFF with STRIX_PREFIX_CACHE_INFLIGHT=0 or the cache off.
        const auto resolved=[&](std::vector<std::string> args,Environment env) {
            auto o=parse(args);
            resolve_session_defaults(o,model,env,128*gib_bytes);
            resolve_prefix_defaults(o,model,env,128*gib_bytes);
            return o;
        };
        const std::vector<std::string> serve_args={"decoder","MODEL","1"};
        const auto on=resolved(serve_args,{});
        expect(on.prefix_cache && on.prefix_cache_inflight && on.prefix_cache_inflight_source=="default",
            "joining resolves ON by default with the cache ON");
        expect(on.cache_activation->automatic_path,"joining default keeps the automatic cache path");
        const auto env_off=resolved(serve_args,{{"STRIX_PREFIX_CACHE_INFLIGHT","0"}});
        expect(!env_off.prefix_cache_inflight && env_off.prefix_cache_inflight_source=="env" &&
            prefix_cache_inflight_state(env_off).reason=="off-switch","STRIX_PREFIX_CACHE_INFLIGHT=0 turns joining OFF");
        expect(env_off.cache_activation->automatic_path,"joining OFF switch leaves the automatic cache path");
        const auto env_on=resolved(serve_args,{{"STRIX_PREFIX_CACHE_INFLIGHT","1"}});
        expect(env_on.prefix_cache_inflight && env_on.prefix_cache_inflight_source=="env","STRIX_PREFIX_CACHE_INFLIGHT=1 keeps joining ON");
        auto cli_args=serve_args;cli_args.insert(cli_args.end(),{"--prefix-cache-inflight","off"});
        const auto cli_off=resolved(cli_args,{{"STRIX_PREFIX_CACHE_INFLIGHT","1"}});
        expect(!cli_off.prefix_cache_inflight && cli_off.prefix_cache_inflight_source=="cli","joining CLI OFF wins env");
        expect(cli_off.cache_activation->automatic_path,"joining CLI switch is not a cache configuration");
        auto cache_off_args=serve_args;cache_off_args.insert(cache_off_args.end(),{"--prefix-cache","off"});
        const auto cache_off=resolved(cache_off_args,{});
        expect(!cache_off.prefix_cache_inflight && cache_off.prefix_cache_inflight_source=="prefix-cache-off" &&
            prefix_cache_inflight_state(cache_off).reason=="prefix-cache-off","cache OFF turns joining OFF");
        const auto cache_env_off=resolved(serve_args,{{"STRIX_PREFIX_CACHE","0"},{"STRIX_PREFIX_CACHE_INFLIGHT","1"}});
        expect(!cache_env_off.prefix_cache_inflight && !prefix_cache_inflight_state(cache_env_off).effective,
            "STRIX_PREFIX_CACHE=0 turns joining OFF");
        auto baseline_args=serve_args;baseline_args.push_back("--baseline");
        const auto baseline_run=resolved(baseline_args,{});
        expect(!baseline_run.prefix_cache_inflight && prefix_cache_inflight_state(baseline_run).reason=="baseline","baseline turns joining OFF");
        // The qualified automatic route is single-slot: joining is ON but inert.
        auto single=on;single.http_serve=true;single.http_batch=0;single.prefix_cache_host_bytes=4*gib_bytes;
        const auto single_state=prefix_cache_inflight_state(single);
        expect(single_state.requested && !single_state.effective && single_state.reason=="single-slot-route",
            "single-slot cache route reports joining inert");
        // A concurrent strict-L0 certified route arms joining with the cache on.
        auto batch=on;batch.http_serve=true;batch.http_batch=3;batch.prefix_cache_host_bytes=4*gib_bytes;batch.prefix_batch_cert=1;
        batch.cache_activation=std::make_shared<CachePolicyActivation>();
        expect(prefix_cache_inflight_state(batch).effective,"strict-L0 batch route arms joining by default");
        EffectiveConfig c;c.options=batch;
        auto effective=serve::parse_json(effective_config_json(c));
        const auto* prefix=effective.find("prefix_cache");
        expect(prefix->find("inflight")->as_bool() && prefix->find("inflight_requested")->as_bool() &&
            prefix->find("inflight_source")->as_string()=="default" && prefix->find("inflight_reason")->as_string()=="armed",
            "effective config reports joining state and source");
        auto batch_off=batch;batch_off.prefix_cache_inflight=false;batch_off.prefix_cache_inflight_source="env";
        c.options=batch_off;effective=serve::parse_json(effective_config_json(c));
        prefix=effective.find("prefix_cache");
        expect(!prefix->find("inflight")->as_bool() && prefix->find("inflight_source")->as_string()=="env" &&
            prefix->find("inflight_reason")->as_string()=="off-switch","effective config reports joining OFF source");
        auto capture=batch;capture.prefix_batch_cert=2;
        expect(prefix_cache_inflight_state(capture).reason=="batch-cert-not-strict-l0","capture-only cert does not arm joining");
    }

    t16k_admission_tests(request,model,info,manifest,capabilities);
    std::cout << "qwen effective config: " << failures << " failures\n";
    return failures ? 1 : 0;
}
