#include "../apps/qwen_prefill_plan.hpp"
#include "../apps/ple_resident_fallback.hpp"
#include "strix/mtp_forward_golden_file.hpp"
#include "strix/ctx_exact_memory.hpp"
#include "../apps/mtp_prompt_phase.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/qwen_yarn_table.hpp"

#include <fstream>
#include <cstdio>
#include <iostream>
#include <stdexcept>

using namespace strix;
using namespace strix::decode_cli;
namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

// Metadata only: no weights, device allocations, model file or external assets.
ModelManifest canonical_manifest(const ModelContract& model) {
    ModelManifest manifest;
    manifest.architecture = "qwen3next";
    std::uint64_t offset = 256;
    const auto add = [&](std::optional<std::uint32_t> layer, std::string name,
                         TensorRole role, std::vector<std::uint64_t> shape, unsigned type = 8U) {
        ManifestTensor t;
        t.name = (layer ? "blk." + std::to_string(*layer) + "." : "") + name + ".weight";
        t.layer = layer; t.role = role; t.dimensions = std::move(shape); t.ggml_type = type;
        std::uint64_t elements = 1;
        for (auto d : t.dimensions) elements *= d;
        t.byte_size = type == 8U ? elements / 32U * 34U : type == 2U ? elements / 32U * 18U : type == 12U ? elements / 256U * 144U :
            type == 13U ? elements / 256U * 176U : type == 7U ? elements / 32U * 24U : elements * 4U;
        t.absolute_offset = offset;
        offset += (t.byte_size + 255U) / 256U * 256U;
        manifest.total_tensor_bytes += t.byte_size;
        manifest.tensors.push_back(std::move(t));
    };
    add({}, "token_embd", TensorRole::embedding, {2560,model.vocab_size});
    const auto hash = build_ple_hash_contract(model);
    add({}, "per_layer_token_embd", TensorRole::ngram_table, {hash.head_dim,hash.minimum_table_rows},2U);
    for (unsigned l = 0; l < 48U; ++l) {
        if (l+1U==model.ple_layer_1based) {
            add(l,"ple_key",TensorRole::ple_parameter,{2560,10240});
            add(l,"ple_value",TensorRole::ple_parameter,{2560,2560});
            add(l,"ple_norm_key",TensorRole::ple_parameter,{10240},0);
            add(l,"ple_norm_query",TensorRole::ple_parameter,{10240},0);
            add(l,"ple_norm_conv",TensorRole::ple_parameter,{10240},0);
            add(l,"ple_conv1d",TensorRole::ple_parameter,{4,10240},0);
        }
        for (const auto* prefix : {"hc_attn_","hc_ffn_"}) {
            add(l,std::string(prefix)+"norm",TensorRole::normalization,{10240},0);
            add(l,std::string(prefix)+"down",TensorRole::dense_projection,{10240,320});
            add(l,std::string(prefix)+"up",TensorRole::dense_projection,{320,10240});
            add(l,std::string(prefix)+"inject",TensorRole::dense_projection,{10240,4},0);
        }
        add(l,"ffn_gate_inp",TensorRole::moe_router,{2560,512},0);
        const bool q80 = l==2U || l==4U || l==30U || l==46U || l==47U;
        for (const auto* name : {"ffn_gate_exps","ffn_up_exps"})
            add(l,name,TensorRole::moe_expert,{2560,640,512},l==2U ? 13U : 12U);
        add(l,"ffn_down_exps",TensorRole::moe_expert,{640,2560,512},q80 ? 8U : 7U);
        add(l,"ffn_gate_shexp",TensorRole::dense_projection,{2560,640});
        add(l,"ffn_up_shexp",TensorRole::dense_projection,{2560,640});
        add(l,"ffn_down_shexp",TensorRole::dense_projection,{640,2560});
        add(l,"ffn_gate_inp_shexp",TensorRole::dense_projection,{2560},0);
        if ((l+1U)%4U) {
            add(l,"attn_qkv",TensorRole::attention_projection,{2560,10240});
            add(l,"attn_gate",TensorRole::attention_projection,{2560,6144});
            add(l,"ssm_beta",TensorRole::recurrent_state,{2560,48},0);
            add(l,"ssm_alpha",TensorRole::recurrent_state,{2560,48},0);
            add(l,"ssm_out",TensorRole::recurrent_state,{6144,2560});
        } else {
            add(l,"attn_q",TensorRole::attention_projection,{2560,12288});
            add(l,"attn_k",TensorRole::attention_projection,{2560,512});
            add(l,"attn_v",TensorRole::attention_projection,{2560,512});
            add(l,"attn_output",TensorRole::attention_projection,{6144,2560});
            add(l,"attn_q_norm",TensorRole::normalization,{256},0);
            add(l,"attn_k_norm",TensorRole::normalization,{256},0);
            add(l,"indexer.q_proj",TensorRole::sparse_index,{2560,512});
            add(l,"indexer.k_proj",TensorRole::sparse_index,{2560,128});
        }
    }
    add({},"output_hc_norm",TensorRole::normalization,{10240},0);
    add({},"output_hc_down",TensorRole::dense_projection,{10240,320});
    add({},"output_hc_up",TensorRole::dense_projection,{320,10240});
    add({},"output",TensorRole::output_head,{2560,model.vocab_size});
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    return manifest;
}

void descriptors(const ModelContract& model, const HipLauncherCatalogConfig& config,
                 const CompiledSerialExecution& serial, const ModelManifest& manifest,
                 const ExecutionTensorAccessPlan& access, unsigned capacity) {
    // Non-dereferenceable address ranges exercise the real metadata/range
    // preflights. They are never allocated, read, uploaded or dispatched.
    ExecutionMemoryBinding scratch{serial.execution_identity,UINT64_C(1)<<44U,serial.arena.total_bytes,1};
    ModelRuntimeBinding weights;
    weights.manifest_layout_identity = manifest.layout_identity;
    weights.model_content_identity.bytes[0] = 1;
    for (std::size_t i=0;i<manifest.tensors.size();++i) {
        const auto& t = manifest.tensors[i];
        weights.tensors.push_back({static_cast<std::uint32_t>(i),RuntimeTensorStorage::resident,
            (UINT64_C(2)<<44U)+t.absolute_offset,t.byte_size,1,0});
    }
    SerialSessionStateLayoutRequest request;
    request.context_tokens = capacity;
    const auto state = build_serial_session_state_layout(model,request);
    SessionStateMemoryBinding state_memory{state.layout_identity,UINT64_C(3)<<44U,state.device_total_bytes,1,
        UINT64_C(4)<<44U,state.host_total_bytes,1};
    ExecutionRuntimeResources resources;
    resources.scratch = &scratch;
    resources.model_manifest = &manifest;
    resources.model_weights = &weights;
    resources.tensor_access = &access;
    resources.session_state_layout = &state;
    resources.session_state_memory = &state_memory;
    HipLauncherCatalog catalog(model,config);
    ExecutionLauncherRegistry launchers;
    catalog.register_for_compiled_execution(serial,launchers,&resources,{},false);
    require(!catalog.audit(serial,launchers).missing_steps,"runtime catalog coverage");
    const auto plan = compile_launch_plan(serial,launchers);
    require(plan.steps.size()==serial.execution.steps.size(),"runtime descriptor count");
    for (const auto& step : plan.steps) require(step.preflight != nullptr,"runtime deep preflight descriptor");
}

void run(unsigned factor, unsigned capacity, unsigned count, const ModelManifest& manifest) {
    auto model = ModelContract::qwen38_flash_next();
    // The CLI resolves this identity before upload. The opaque CPU owner below
    // is used only when constructing launch descriptors, never dispatched.
    const auto rotary = RotaryDescriptor::yarn(factor);
    Sha256Fast hash;
    generate_qwen_yarn_rows(rotary,0,rotary.effective_max(),[&](auto,auto rows) { hash.update(std::as_bytes(rows)); });
    model.resolved_rotary = RotaryIdentity(rotary,hash.finalize());
    const std::string path = "qwen-yarn-host-" + std::to_string(count) + ".ids";
    struct Cleanup { std::string path; ~Cleanup() { std::remove(path.c_str()); } } cleanup{path};
    { std::ofstream prompt(path); for (unsigned i=0;i<count;++i) prompt << (i%model.vocab_size) << '\n'; }
    std::vector<std::string> args{"decoder","MODEL","--prompt-ids",path,"--rope-yarn",std::to_string(factor),
        "--context",std::to_string(capacity),"--generate","32"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    RequestedOptions request;
    request.options = parse_options(static_cast<int>(argv.size()),argv.data());
    require(request.options.tokens.size()==count && request.environment.empty(),"parsed prompt and clean environment");
    PrefillMemory memory;
    memory.total = memory.available = memory.gtt_limit = UINT64_C(124)<<30U;
    memory.gtt_used = 0;
    // Simulated verified credit isolates host compilation from OS reclaim.
    memory.verify_ple_credit = [](const Options&, const Environment&) { return UINT64_C(28)<<30U; };
    memory.estimate = [&](const Options& o) { return compile_prompt_requirements(o,model,manifest,111334654400ULL); };
    const auto effective = resolve_effective_config(request,model,{},manifest,
        validated_model_content_identity,{true,true,true},memory);
    require(effective.admitted,effective.error);
    validate_effective_options(effective.options,model);
    require(yarn_environment_refusal(effective.environment,factor).empty(),"effective environment classification");
    for (const auto* value : {"0","1"}) {
        auto diagnostic=effective.environment;diagnostic["STRIX_STARTUP_MARKERS"]=value;
        require(yarn_environment_refusal(diagnostic,factor).empty(),"startup markers are a no-rotary diagnostic control");
    }
    { auto invalid=effective.environment;invalid["STRIX_STARTUP_MARKERS"]="invalid";
      require(!yarn_environment_refusal(invalid,factor).empty(),"invalid startup marker control refuses by name"); }

    require(!effective.options.mtp.enabled && effective.admission.at("owner_target_kv_bytes")>0,
        "complete admission inventory without MTP");
    validate_wave_router_model_inputs({});
    const auto layers = moe_q4k_a8::classify_runtime_manifest(model,manifest);
    require(!layers.q4k_with_q51_down_layers.empty(),"grouped Q5_1 down admission");
    moe_q4k_a8::validate_slot_parallel_manifest(model,layers);
    ModelContentIdentity content;
    for (std::size_t i=0;i<content.bytes.size();++i)
        content.bytes[i] = static_cast<unsigned char>(std::stoul(std::string(validated_model_content_identity.substr(i*2U,2U)),nullptr,16));
    moe_q4k_a8::validate_a8_down_content(content);
    moe_q4k_a8::validate_a8_down_manifest(model,layers,true,true);
    moe_q4k_a8::validate_q5k_a8_manifest(model,layers);
    model.qwen_rotary_table = std::make_shared<HipQwenRotaryTable>(model.resolved_rotary,
        std::make_shared<int>(0),rotary.table_bytes());
    auto requests = prompt_compiler_requests(effective.options,model,manifest);
    kernels::KernelRegistry registry;
    register_complete_hip_kernel_catalog(model,requests.binding,requests.catalog,registry);
    auto serial = compile_serial_execution(model,registry,requests.binding,requests.buffers,256);
    const auto access = compile_execution_tensor_access_plan(serial.execution,manifest,build_model_tensor_index(manifest),
        build_serial_execution_tensor_requirements(serial.execution));
    qsa_input_prefetch::validate_tensor_access(qsa_input_prefetch::build_plan(model,manifest),manifest,serial.execution,access);
    shared_expert_prefetch::validate_tensor_access(shared_expert_prefetch::build_plan(model,manifest),manifest,serial.execution,access);
    validate_prompt_tensor_contract(effective.options,manifest,serial,access);
    descriptors(model,requests.catalog,serial,manifest,access,capacity);
    const auto pair = numerical_profile_pair(effective.options);
    require(pair.split(),"production split prefill lineage");
    // Admission and the runtime bind the governed prefill identity (option mask, GSQ arithmetic).
    if (numerical_profile_fast_prefill(pair.prefill))
        requests.binding.prefill_arithmetic_identity = prefill_arithmetic_identity(pair,
            effective.options.resolved_prefill_option_mask.value_or(effective.prefill_option_mask),
            qpf2k::has_gsq_gate(manifest));
    requests.catalog.numerical_profile = pair.prefill;
    requests.catalog.fast_prefill_components = pair.fast_components;
    kernels::KernelRegistry prefill_registry;
    register_complete_hip_kernel_catalog(model,requests.binding,requests.catalog,prefill_registry);
    auto prefill = compile_serial_execution(model,prefill_registry,requests.binding,requests.buffers,256);
    validate_sibling_serial_executions(prefill,serial);
    require(effective.host_compilation.at("serial_execution_identity")==prefill.execution_identity &&
        effective.host_compilation.at("serial_kernel_plan_identity")==prefill.kernel_plan_identity,
        "admission and runtime serial identities match across table adoption");
    descriptors(model,requests.catalog,prefill,manifest,access,capacity);
    require(serial.max_position_embeddings==model.effective_max_positions() && model.max_position_embeddings==262144,
        "effective capacity preserves checkpoint geometry");
    for (const auto wrong_capacity : {model.max_position_embeddings,static_cast<std::uint64_t>(capacity)}) {
        if(wrong_capacity==model.effective_max_positions())continue;
        auto wrong = prefill;
        wrong.max_position_embeddings = wrong_capacity;
        bool rejected = false;
        try { (void)compile_known_prefix_window(model,wrong,manifest,access,{}); }
        catch (const std::invalid_argument& e) {
            rejected = std::string_view(e.what()).find("complete eager C1 serial source") != std::string_view::npos;
        }
        require(rejected,"native/context capacity cannot replace the resolved rotary maximum");
    }
    SerialSessionStateLayoutRequest state_request;
    state_request.context_tokens = capacity;
    require(build_serial_session_state_layout(model,state_request).context_tokens==capacity,"session capacity");
    auto rows = effective.options.known_prefix_tail_rows;
    rows.insert(rows.begin(),effective.options.known_prefix_prefill_rows);
    for (auto n : rows) {
        const auto plan = compile_known_prefix_prefill(model,prefill,manifest,access,
            {n,true,true,true,true,KnownWindowOutputPolicy::terminal_row,
             effective.options.known_prefill_workspace_budget,effective.options.known_prefill_metadata_budget/2U,
             true,true,true,true,true,false,true,true});
        validate_compiled_known_prefix_prefill(plan,prefill);
        const auto legacy=compile_known_prefix_prefill(model,prefill,manifest,access,
            {n,true,true,true,true,KnownWindowOutputPolicy::terminal_row,UINT64_MAX,UINT64_MAX,
             true,true,true,true,true,false,false,true});
        require(plan.prefill_kernel_plan_identity==legacy.prefill_kernel_plan_identity &&
            plan.tile8.window_kernel_plan_identity==legacy.tile8.window_kernel_plan_identity &&
            plan.tile8.r2_variant==legacy.tile8.r2_variant,"compact storage preserves every selected kernel identity");
        require(plan.prefill_execution_identity!=legacy.prefill_execution_identity,"storage identity reflects compact offsets");

        // The planner's compact layout: the score ring follows the YaRN effective positions.
        const QsaScoreRingGeometry ring(model.effective_max_positions(),model.indexer_compress_ratio,plan.qsa_score_ring_rows);
        require(plan.qsa_score_ring_workspace_bytes && plan.qsa_score_ring_workspace_bytes==ring.total_bytes() &&
            model.effective_max_positions()>model.max_position_embeddings,"YaRN score ring follows the effective positions");
        validate_known_prefix_window_formats(plan.tile8,prefill,manifest,access);
        if (n==effective.options.known_prefix_prefill_rows)
            require(effective.host_compilation.at("prefill_execution_identity")==plan.prefill_execution_identity &&
                effective.host_compilation.at("prefill_kernel_plan_identity")==plan.prefill_kernel_plan_identity,
                "admission and runtime prefill identities match");
        auto source = prefill;
        source.execution_identity = plan.prefill_execution_identity;
        source.arena = plan.terminal_logits_arena ? plan.arena.row_view.back() : plan.arena.row_view[0];
        source.whole_plan_graph_safe = false;
        descriptors(model,requests.catalog,source,manifest,access,capacity);
        require(plan.tile8.gdn_deferred_rows_steps+plan.tile8.gdn_rows_steps==36U && plan.tile8.qsa_rows_steps==12U &&
            plan.tile8.ple_rows_steps==1U,"all canonical layers admitted");
    }
    std::cout << "PASS host compile f=" << factor << " C=" << capacity << " prompt=" << count
              << " T=" << effective.options.known_prefix_prefill_rows << " tails=" << rows.size()-1U
              << " before device allocation\n";
}
// Exercise the setup transaction with actual legacy golden bytes and the real
// compiler/owner census. No admission callback substitutes a canned result.
void native_resident_golden_contract(const ModelManifest& manifest) {
    const auto model=ModelContract::qwen38_flash_next();
    const auto root=std::filesystem::temp_directory_path()/"strix-native-resident-golden";
    std::filesystem::create_directories(root);
    struct Cleanup { std::filesystem::path p;~Cleanup(){std::filesystem::remove_all(p);} } cleanup{root};
    const auto sidecar=root/"sidecar.gguf";
    { // Minimal real GGUF; no tensor payload is needed by this non-chain census.
        std::ofstream f(sidecar,std::ios::binary);
        const auto put=[&](std::uint64_t value,unsigned n) {
            for(unsigned i=0;i<n;++i) f.put(static_cast<char>((value>>(8U*i))&255U));
        };
        f.write("GGUF",4);put(3,4);put(0,8);put(0,8);put(0,8);
    }
    Options o;o.context_tokens=32768;o.tokens.resize(2048,1);o.generate_tokens=16;
    o.known_prefix_window=o.known_prefix_full_rows=true;o.known_prefix_prefill_rows=2048;
    o.known_prefix_output_policy=KnownWindowOutputPolicy::terminal_row;
    o.known_prefill_uma_budget=UINT64_C(512)<<30U;
    o.known_prefill_workspace_budget=UINT64_C(64)<<30U;o.known_prefill_metadata_budget=UINT64_C(4)<<30U;
    RequestedOptions request;request.options.tokens=o.tokens;request.options.context_tokens=o.context_tokens;
    request.options.generate_tokens=o.generate_tokens;request.options.context_explicit=true;request.options.unqualified_context=true;
    PrefillMemory ample;ample.total=ample.available=ample.gtt_limit=UINT64_C(512)<<30U;ample.gtt_used=0;
    ample.estimate=[&](const Options& candidate){return compile_prompt_requirements(candidate,model,manifest,UINT64_C(1)<<30U);};
    const auto effective=resolve_effective_config(request,model,{},manifest,validated_model_content_identity,{true,true,true},ample);
    require(effective.admitted,effective.error);o=effective.options;
    auto& m=o.mtp;m.enabled=true;m.sidecar=sidecar.string();m.sidecar_bytes=32;m.sidecar_residency="register";
    m.calibration_rows=1;m.window_rows=2;m.candidate_depth=1;m.window_history_intervals=8192;
    m.runtime_bytes=m.sidecar_retained_bytes=m.sidecar_setup_bytes=m.header_bytes=
        m.target_capture_bytes=m.golden_max_bytes=m.replay_host_bytes=m.window_arena_bytes=
        m.window_capture_bytes=m.window_checkpoint_bytes=m.window_total_bytes=UINT64_C(64)<<30U;
    HipMtpForwardGolden golden;
    golden.independent_source_content=mtp_self_generated_golden_source();
    golden.target_content.bytes[0]=1;golden.sidecar_content.bytes[0]=2;
    golden.independent_reference_identity=mtp_self_generated_golden_reference;
    golden.structural_plan_identity=golden.semantic_identity=golden.numerical_profile_identity=1;
    HipMtpForwardGoldenRow row;row.target_hidden_bf16.resize(model.hidden_size*model.hc_count);
    row.next_hidden_bf16=row.target_hidden_bf16;row.full_logits.resize(model.vocab_size);
    golden.rows.push_back(std::move(row));
    for(const bool v2:{false,true}) {
        if(v2) golden.target_prefill_profile_identity=golden.target_decode_profile_identity=1;
        const auto path=root/(v2?"native-v2.bin":"native-v1.bin");
        const auto pin=write_mtp_forward_golden_file(path,golden,m.golden_max_bytes);
        require(read_mtp_forward_golden_file(path,pin,m.golden_max_bytes).rows.size()==1,"real legacy golden round trip");
        m.golden=path.string();
        bool torn_down=false,warmed=false;
        const auto source=setup_ple_or_resident(ple_reclaim::Mode::automatic,
            []()->std::shared_ptr<ple_reclaim::Prefetcher>{throw std::runtime_error("fixture setup failure");},
            [&]{torn_down=true;},[&]{
                require(torn_down && !warmed,"teardown before actual resident census");
                const auto required=compile_prompt_requirements(o,model,manifest,UINT64_C(1)<<30U,nullptr,true);
                require(required.owners.contains("mtp_predictor_kv_raw") && required.owner_error.empty(),"native MTP full census");
                PrefillMemory fresh;fresh.total=fresh.available=fresh.gtt_limit=UINT64_C(512)<<30U;fresh.gtt_used=0;
                require(resident_prefill_admission(o,required,fresh).at("verified_ple_credit_bytes")==0,"real zero-credit native admission");
            },[&]{warmed=true;});
        require(!source && warmed && m.enabled,"native legacy fallback completes and retains MTP");
    }
    auto yarn_model=model;ModelContentIdentity table;table.bytes[0]=1;
    o.rotary=RotaryDescriptor::yarn(2);yarn_model.resolved_rotary=RotaryIdentity(o.rotary,table);
    o.mtp_bundle_path=(root/"bundle.json").string();m.golden.clear();
    const auto pending=compile_prompt_requirements(o,yarn_model,manifest,UINT64_C(1)<<30U);
    require(pending.owners.at("mtp_golden_host")>0,"pending self-generation still charges golden ownership");
    // A hit must retain the geometry check: legacy geometry is refused, not skipped.
    m.golden=(root/"native-v2.bin").string();
    bool refused=false;
    try {(void)compile_prompt_requirements(o,yarn_model,manifest,UINT64_C(1)<<30U);}
    catch(const std::exception& e) {refused=std::string(e.what()).find("YaRN requires golden v3 geometry")!=std::string::npos;}
    require(refused,"bundle cache hit validates geometry");

}
// Storage eligibility is a shared admission/runtime contract, independent of arithmetic.
void compact_mtp_route_contract() {
    Options o;o.mtp.enabled=true;o.known_prefix_window=o.known_prefix_full_rows=true;
    o.known_prefix_prefill_rows=2048;o.known_prefix_output_policy=KnownWindowOutputPolicy::terminal_row;
    require(compact_prompt_memory(o),"closed MTP CLI admits compact prefill");
    const auto excluded=[&](auto mutate) {auto other=o;mutate(other);require(!compact_prompt_memory(other),"reusable or replayed MTP keeps legacy layout");};
    excluded([](auto& x){x.http_serve=true;});
    excluded([](auto& x){x.mtp.cohort.resize(1);});
    excluded([](auto& x){x.mtp.copy_wide_rows={16};});
    excluded([](auto& x){x.mtp.sample_repeats=2;});
    excluded([](auto& x){x.session_management_requested=true;});
    excluded([](auto& x){x.prefix_cache_host_bytes=1;});
    excluded([](auto& x){x.rtx_state_import=true;});
    excluded([](auto& x){x.known_prefix_output_policy=KnownWindowOutputPolicy::all_rows;});
    excluded([](auto& x){x.execution.mode=DecodeExecutionMode::graph;});
    o.mtp={};require(compact_prompt_memory(o),"serial compact route unchanged");
}
void mtp_phase_handoff_contract() {
    bool pending=true,retained=true,allocated=false;
    complete_mtp_prompt_phase([&]{pending=false;},[&]{require(!pending,"last consumers drain before free");retained=false;},
        [&]{require(!retained,"prompt ownership retires before windows allocate");allocated=true;});
    require(allocated,"handoff allocated decode owners");
    for(bool fail_drain:{true,false}) {
        bool retired=false,created=false;
        try {complete_mtp_prompt_phase([&]{if(fail_drain)throw std::runtime_error("drain");},
            [&]{retired=true;throw std::runtime_error("retire");},[&]{created=true;});}
        catch(const std::runtime_error&) {}
        require(!created && retired!=fail_drain,"failed drain/retirement never admits decode allocation");
    }
    const auto peaks=prefill_owner_phase_peaks({{"shared",100},{"prompt",60},{"decode",80}},
        {{"prompt",7},{"decode",24}},8);
    require(peaks==std::array<std::uint64_t,5>{168,168,168,188,188},"phase peak charges shared backing in both phases");
    Options o;o.mtp.enabled=true;o.known_prefix_window=o.known_prefix_full_rows=true;
    o.known_prefix_prefill_rows=2048;o.known_prefix_output_policy=KnownWindowOutputPolicy::terminal_row;
    o.rotary=RotaryDescriptor::yarn(4);o.tokens.resize(1048320);
    require(mtp_prompt_phase_separation(o),"whole tiled YaRN prompt has no suffix verifier consumer");
    o.tokens.push_back(1);require(!mtp_prompt_phase_separation(o),"suffix prompts retain verifier during prompt");
    o.tokens.pop_back();o.rotary=RotaryDescriptor::native();
    require(!mtp_prompt_phase_separation(o),"native lifetime unchanged");
}
// Frozen campaign limits plus host-only manifest geometry. The OS verification
// callback is simulated; no predicted credit is used by production admission.
void mtp_receipt_admission(const ModelManifest& original) {
    auto manifest=original;
    for(auto& t:manifest.tensors) if(t.role==TensorRole::ngram_table) {
        t.ggml_type=20;t.residency=ResidencyPolicy::file_backed_candidate;
    }
    manifest.layout_identity=model_manifest_layout_identity(manifest);
    gguf::FileInfo sidecar;sidecar.file_size=2786568256ULL;sidecar.data_offset=65536;
    gguf::TensorInfo q4;q4.name="blk.48.attn_q.weight";q4.ggml_type=8;q4.byte_size=2745937920ULL;sidecar.tensors.push_back(q4);
    const std::string vocab_path="ymtp-host-vocab.bin";
    struct Cleanup { std::string path;~Cleanup(){std::remove(path.c_str());} } cleanup{vocab_path};
    { std::ofstream f(vocab_path,std::ios::binary);for(std::uint32_t id=0;id<32768U;++id)
        for(unsigned j=0;j<4;++j) f.put(static_cast<char>((id>>(8U*j))&255U)); }
    for(const auto c:{33024U,131328U,262400U,393472U,524544U,1048576U}) {
        const unsigned factor=c<=524544U?2U:4U;
        // f2's effective maximum is 524288: the requested 524544/f2 table
        // intentionally records the descriptor refusal rather than relabelling it.
        auto model=ModelContract::qwen38_flash_next();
        ModelContentIdentity hash;hash.bytes[0]=1;model.resolved_rotary=RotaryIdentity(RotaryDescriptor::yarn(factor),hash);
        RequestedOptions request;auto& o=request.options;
        o.rotary=RotaryDescriptor::yarn(factor);o.context_tokens=c;o.context_explicit=true;
        o.tokens.resize(c==1048576U?1048320U:c-256U,1U);o.generate_tokens=256;
        auto& mtp=o.mtp;mtp.enabled=true;mtp.sidecar="SIDECAR.gguf";mtp.sidecar_logical_name="fixture";
        mtp.sidecar_sha256=mtp.target_sha256=mtp.target_content=mtp.golden_sha256=mtp.converter_sha256=mtp.norm_converter_sha256=std::string(64,'1');
        mtp.repository="fixture";mtp.revision="revision";mtp.source_path="SIDECAR.gguf";mtp.converter_revision="revision";
        mtp.golden="GOLDEN.bin";mtp.conversion="effective-norm-embedding-first-v1";
        mtp.normalized_rounding=mtp.projection_rounding="bf16-rne";mtp.sidecar_residency="register";
        mtp.sidecar_bytes=sidecar.file_size;mtp.header_bytes=33554432;mtp.sidecar_retained_bytes=3221225472ULL;mtp.sidecar_setup_bytes=4294967296ULL;
        mtp.golden_max_bytes=mtp.replay_host_bytes=6442450944ULL;mtp.runtime_bytes=134217728;mtp.target_capture_bytes=16777216;
        mtp.policy_identity=1;mtp.candidate_depth=7;mtp.calibration_rows=16;
        mtp.window_rows=8;mtp.window_rows_set={2,4,8};mtp.window_arena_bytes=2147483648ULL;mtp.window_capture_bytes=67108864;
        mtp.window_checkpoint_bytes=8589934592ULL;mtp.window_total_bytes=17179869184ULL;mtp.window_history_intervals=8192;
        mtp.window_expert_union=true;mtp.window_expert_union_bytes=65536;mtp.window_expert_union_bytes_explicit=true;
        mtp.window_prefix=mtp.window_hyper_mix=mtp.window_qsa_rows=mtp.draft_chain=mtp.draft_q4=true;
        mtp.draft_vocab=vocab_path;mtp.draft_context_vocab=4096;
        o.known_prefix_window=o.known_prefix_full_rows=true;o.known_prefix_prefill_rows=8192;
        request.environment={{"STRIX_PLE_ROLLING","1"},{"STRIX_PLE_RECLAIMABLE","auto"},
            {"STRIX_QSA_MTP_POOL","1"}}; // explicit census includes factor-4 MTP
        PrefillMemory memory;memory.total=127470000000ULL;memory.available=127470000000ULL;
        memory.gtt_limit=127470000000ULL;memory.gtt_used=0;memory.ple_reclaim_page=4096;
        unsigned verified=0;
        memory.verify_ple_credit=[&](const Options& candidate, const Environment& environment) {
            ++verified;
            require(environment.at("STRIX_QPF2K_SCHED")=="12", "verification consumes resolved environment before runtime installation");
            auto config=prefill_reclaim_config(candidate,model,manifest,environment,memory.ple_reclaim_page);
            require(config.has_value(),"bounded YaRN MTP reclaim route includes dump");
            return prefill_reclaim_credit(candidate,model,manifest,environment,memory.ple_reclaim_page);
        };
        memory.estimate=[&](const Options& candidate){return compile_prompt_requirements(candidate,model,manifest,111334654400ULL,&sidecar);};
        memory.estimate_with_score_ring=[&](const Options& candidate,std::uint32_t ring_rows) {
            return compile_prompt_requirements(candidate,model,manifest,111334654400ULL,&sidecar,false,ring_rows);
        };
        const auto resolve=[&](const RequestedOptions& r,const PrefillMemory& mem) {
            return resolve_effective_config(r,model,{},manifest,validated_model_content_identity,{true,true,true},mem);
        };
        for(const bool dump:{true,false}) {
            auto arm=request;if(dump) { arm.options.mtp.golden_dump="DUMP.bin";arm.options.mtp.golden.clear();arm.options.mtp.golden_sha256.clear(); }
            const auto result=resolve(arm,memory);
            std::cout << "YMTP_RECEIPT C=" << c << " factor=" << factor << " arm=" << (dump?"dump":"generation") << '\n'
                      << effective_config_json(result);
            if(c==1048576U && !dump) {
                auto fit=arm;fit.options.mtp.runtime_bytes=2418528231ULL;
                fit.environment["STRIX_PLE_ROLLING_POSITIONS"]="256";
                fit.environment["STRIX_PLE_ROLLING_CAP_BYTES"]="33554432";
                fit.environment["STRIX_PREFILL_PIPE"]="0";
                fit.environment["STRIX_QSA_SCORE_RING_ROWS"]="64";
                const auto* prior_ring=std::getenv("STRIX_QSA_SCORE_RING_ROWS");
                const auto saved_ring=prior_ring?std::optional<std::string>{prior_ring}:std::nullopt;
                ::setenv("STRIX_QSA_SCORE_RING_ROWS","64",1);
                const auto residual=resolve(fit,memory);
                const auto needed=memory.estimate(residual.options);
                require(needed.owner_error.empty(),"1M payload caps pass independently of physical fit");
                require(residual.admitted,residual.error);
                const auto rolling_owner=prefill_rolling_owner_bytes(residual.options,model,manifest,fit.environment,memory.ple_reclaim_page);
                for (const bool separated:{false,true}) {
                    auto candidate=residual.options; candidate.mtp_cli_phase_retirement=separated;
                    const auto required=memory.estimate(candidate);
                    require(required.registered_plan.has_value(),"host compiler emits the runtime admission geometry");
                    auto owners=required.owners; owners["rotary_table"]=candidate.rotary.table_bytes();
                    owners["ple_rolling_owner"]=rolling_owner;
                    const auto peaks=prefill_owner_phase_peaks(owners,required.owner_phases,candidate.known_prefill_host_headroom);
                    UmaOwnerInventory live;
                    live.known_file_mapping_bytes=required.model_backing;
                    live.reservation_upper_bound_bytes=required.model_backing+required.registered_plan->base_runtime+
                        required.registered_plan->caller_live;
                    const auto early=yarn_registered_prefill_envelope(candidate,required,peaks,live,
                        required.model_backing,rolling_owner,residual.ple_reclaim_credit,false);
                    const auto runtime=yarn_registered_prefill_envelope(candidate,required,peaks,live,
                        live.reservation_upper_bound_bytes,rolling_owner,residual.ple_reclaim_credit,false);
                    require(early.reservation.setup_peak_reserved_bytes==runtime.reservation.setup_peak_reserved_bytes &&
                        early.incremental==runtime.incremental+required.registered_plan->base_runtime+required.registered_plan->caller_live,
                        "registration and runtime use identical retained/growth definitions and charge unpaid live owners once");
                    auto growth=required.registered_plan->growth+rolling_owner+required.mtp_post_setup_peak;
                    for (const auto* name:{"mtp_m1_keys","mtp_m1_scratch_metadata"})
                        if(required.owners.contains(name)) growth+=required.owners.at(name);
                    auto temporary=required.registered_plan->temporary;
                    if (mtp_prompt_phase_separation(candidate)) {
                        growth=*std::max_element(peaks.begin()+1,peaks.end())-live.reservation_upper_bound_bytes;
                        temporary=0;
                    }
                    const auto bridge=plan_known_prefill_reservation(live,growth,temporary,{},residual.ple_reclaim_credit,0);
                    require(bridge.setup_peak_reserved_bytes==runtime.reservation.setup_peak_reserved_bytes &&
                        growth+temporary==runtime.incremental,"shared envelope preserves the runtime UMA ledger and physical growth");
                    const auto total=early.reservation.setup_peak_reserved_bytes+prefill_physical_margin;
                    const auto available=early.incremental+prefill_physical_margin;
                    require(prefill_physical_margin_fits(total,available,early.reservation.setup_peak_reserved_bytes,early.incremental) &&
                        !prefill_physical_margin_fits(total,available-1U,early.reservation.setup_peak_reserved_bytes,early.incremental),
                        "same physical helper admits exact margin and refuses one byte short");
                }
                require(residual.admission.at("owner_ple_rolling_owner_bytes")==rolling_owner,
                    "complete shared reader/ledger owner included in every phase");
                for(const auto* phase:{"setup","calibration","prompt","decode","drain"}) {
                    const auto gross=residual.admission.at(std::string("phase_")+phase+"_gross_bytes");
                    require(gross-residual.ple_reclaim_credit<127473287168ULL,
                        "1M each phase fits with the shared reader/turn owner already charged");
                }
                // Matched 1M f4 candidate: production PIPE, Q4 retained, larger source waves/ring.
                auto candidate=fit;
                candidate.environment["STRIX_PLE_RECLAIMABLE"]="1";
                candidate.environment["STRIX_PLE_ROLLING_POSITIONS"]="1024";
                candidate.environment["STRIX_PLE_ROLLING_CAP_BYTES"]="268435456";
                candidate.environment["STRIX_QSA_SCORE_RING_ROWS"]="256";
                candidate.environment["STRIX_MTP_DRAFT_Q4_ALL"]="1";
                candidate.environment.erase("STRIX_PREFILL_PIPE");
                const auto* pipe=std::getenv("STRIX_PREFILL_PIPE");
                const auto saved_pipe=pipe?std::optional<std::string>{pipe}:std::nullopt;
                ::unsetenv("STRIX_PREFILL_PIPE");::setenv("STRIX_QSA_SCORE_RING_ROWS","256",1);
                auto candidate_memory=memory;
                candidate_memory.total=candidate_memory.gtt_limit=1ULL<<40U;
                candidate_memory.available=127473287168ULL;
                const auto admitted=resolve(candidate,candidate_memory);
                require(admitted.admitted,admitted.error);
                require(admitted.options.known_prefix_prefill_rows==8192U && admitted.options.yarn_mtp_pipeline &&
                    admitted.options.yarn_mtp_q4_all && admitted.environment.at("STRIX_PREFILL_PIPE")=="255",
                    "candidate retains T8192, production PIPE 255, Q4 and f4 identity");
                // Relative to the original pins: +11 MiB shared-pre image (all phases), +80 MiB packed PLE slab
                // (prompt phases only) and -96 MiB compact QSA output image (prompt arena).
                const std::array<std::uint64_t,5> expected_spare{5695117605ULL,2367600601ULL,558193949ULL,5534911325ULL,5534911325ULL};
                const std::array candidate_phases{"setup","calibration","prompt","decode","drain"};
                // Every phase is reported before the check, so one failure shows the whole spare profile.
                bool spare_ok=true;std::string spare_report;
                for(unsigned i=0;i<candidate_phases.size();++i) {
                    const auto gross=admitted.admission.at(std::string("phase_")+candidate_phases[i]+"_gross_bytes");
                    const auto net=gross-admitted.ple_reclaim_credit;
                    spare_ok=spare_ok && net<=127473287168ULL && 127473287168ULL-net==expected_spare[i] && expected_spare[i]>=500000000ULL;
                    spare_report+=" "+std::string(candidate_phases[i])+" actual="+
                        (net<=127473287168ULL?std::to_string(127473287168ULL-net):"-"+std::to_string(net-127473287168ULL))+
                        " expected="+std::to_string(expected_spare[i])+";";
                }
                require(spare_ok,"production candidate admission leaves at least 0.5 decimal GB spare in every phase:"+spare_report);
                // The plain 1M command chooses storage, without inheriting the process ring.
                auto automatic=candidate;automatic.environment.clear();
                ::unsetenv("STRIX_QSA_SCORE_RING_ROWS");
                const auto ladder=resolve(automatic,candidate_memory);
                require(ladder.admitted,ladder.error);
                std::cout << "YMTP_STORAGE auto-candidate\n" << effective_config_json(ladder);
                require(ladder.admission.at("storage_ladder_step")==1U &&
                    ladder.ple_rolling_h==1024U && ladder.ple_rolling_cap==268435456U &&
                    ladder.host_compilation.at("qsa_score_ring_rows")==256U && ladder.options.yarn_mtp_pipeline,
                    "empty storage environment chooses first fitting 1M candidate");
                for(const auto* phase:candidate_phases) {
                    const auto net=ladder.admission.at(std::string("phase_")+phase+"_gross_bytes")-ladder.ple_reclaim_credit;
                    require(net<=*candidate_memory.available && *candidate_memory.available-net>=500000000ULL,
                        "automatic 1M storage leaves at least 0.5 GB spare in every phase");
                }
                require(effective_config_json(ladder).find("\"storage_ladder_step\":1")!=std::string::npos &&
                    yarn_mtp_storage_receipt(ladder).find("\"qsa_score_ring_rows\":256")!=std::string::npos,
                    "inspection and run receipt expose chosen sizes and ladder step");
                auto lost_credit=candidate_memory;
                unsigned drops=0;
                lost_credit.verify_ple_credit=[&](const Options& plan,const Environment& env) {
                    if(++drops==2U) throw std::runtime_error("fixture fresh verification failure");
                    return memory.verify_ple_credit(plan,env);
                };
                const auto lost=resolve(automatic,lost_credit);
                require(!lost.admitted && drops==2U && !lost.ple_reclaim_credit && lost.admission.empty(),
                    "failed next-step verification cannot retain preceding credit or owners");
                auto tight_memory=memory;
                tight_memory.total=tight_memory.available=*memory.total-(UINT64_C(1)<<28U);
                const auto tight=resolve(automatic,tight_memory);
                std::cout << "YMTP_STORAGE auto-tight\n" << effective_config_json(tight);
                // The 4 GiB physical margin refuses the tight set at T8192 and T4096 (prompt phase binds); the
                // ladder numbers each (T, storage set) trial, so the tight set at T2048 is step 4.
                require(tight.admitted && tight.admission.at("storage_ladder_step")==4U &&
                    tight.options.known_prefix_prefill_rows==2048U &&
                    tight.ple_rolling_h==256U && tight.ple_rolling_cap==33554432U &&
                    tight.host_compilation.at("qsa_score_ring_rows")==64U && !tight.options.yarn_mtp_pipeline,
                    "physical-memory refusal advances to tight set");
                for(const auto* phase:candidate_phases) {
                    const auto net=tight.admission.at(std::string("phase_")+phase+"_gross_bytes")-tight.ple_reclaim_credit;
                    require(net<=*tight_memory.total && *tight_memory.total-net>=500000000ULL,
                        "tight 1M storage leaves at least 0.5 GB spare in every phase");
                }
                auto explicit_sizes=automatic;
                explicit_sizes.environment={{"STRIX_PLE_ROLLING_POSITIONS","256"},
                    {"STRIX_PLE_ROLLING_CAP_BYTES","33554432"},{"STRIX_QSA_SCORE_RING_ROWS","64"},
                    {"STRIX_PREFILL_PIPE","0"}};
                const auto pinned=resolve(explicit_sizes,memory);
                require(pinned.admitted && pinned.admission.at("storage_ladder_step")==0U,
                    "explicit fitting storage is step zero");
                for(const auto& [name,value]:explicit_sizes.environment)
                    require(pinned.environment.at(name)==value,"explicit storage preserved");
                for(const auto& [name,value]:explicit_sizes.environment) {
                    auto mixed=automatic;mixed.environment[name]=value;
                    const auto chosen=resolve(mixed,memory);
                    require(chosen.environment.at(name)==value,
                        "each explicit knob independently survives the ladder");
                }
                auto no_fit=automatic;
                no_fit.environment={{"STRIX_PLE_ROLLING_POSITIONS","8192"},
                    {"STRIX_PLE_ROLLING_CAP_BYTES","3221225472"},{"STRIX_QSA_SCORE_RING_ROWS","512"},
                    {"STRIX_PREFILL_PIPE","255"}};
                const auto refused=resolve(no_fit,memory);
                require(!refused.admitted && refused.admission.at("storage_ladder_step")==0U,
                    "fully pinned oversized storage refuses without a ladder");
                for(const auto& [name,value]:no_fit.environment)
                    require(refused.environment.at(name)==value,"ladder never overrides explicit storage on refusal");
                for(const char* off:{"STRIX_QSA_MTP_POOL","STRIX_PLE_ROLLING"}) {
                    auto disabled=automatic;disabled.environment[off]="0";
                    const auto resolved=resolve(disabled,candidate_memory);
                    require(resolved.environment.at(off)=="0","ladder preserves off switch");
                    if(std::string_view(off)=="STRIX_QSA_MTP_POOL")
                        require(!qsa_mtp_pool_route(resolved.options),"pool off keeps historical consumers");
                    else require(!resolved.ple_rolling_h,"rolling off keeps prompt-wide storage");
                }
                if(saved_pipe) ::setenv("STRIX_PREFILL_PIPE",saved_pipe->c_str(),1);else ::unsetenv("STRIX_PREFILL_PIPE");
                ::setenv("STRIX_QSA_SCORE_RING_ROWS","64",1);
                auto old=fit;old.environment["STRIX_QWEN_YARN_HOST_RESERVE"]="1";
                const auto legacy=resolve(old,memory);
                require(!legacy.admitted && legacy.options.known_prefix_prefill_rows==2048U &&
                    // The single fresh-CLI PLE slab lowers the old peak phase by 1310720 bytes.
                    legacy.admission.at("required_bytes")==130647357859ULL+rolling_owner,
                    "reserve off switch restores old 1M T2048 refusal plus separately charged rolling owner; actual="+
                    std::to_string(legacy.admission.at("required_bytes"))+" rolling="+std::to_string(rolling_owner));
                auto legacy_t=legacy.options;legacy_t.known_prefill_host_headroom=0;
                const auto old_required=memory.estimate(legacy_t);
                auto old_owners=old_required.owners;old_owners["rotary_table"]=legacy_t.rotary.table_bytes();
                old_owners["ple_rolling_owner"]=rolling_owner;
                const auto new_peaks=prefill_owner_phase_peaks(old_owners,old_required.owner_phases,0);
                const std::array phases{"setup","calibration","prompt","decode","drain"};
                for(unsigned i=0;i<phases.size();++i)
                    require(new_peaks[i]+yarn_legacy_host_reserve==legacy.admission.at(std::string("phase_")+phases[i]+"_gross_bytes"),
                        "admission and runtime shared phase helper restore exactly 8 GiB");
                require(old_required.organization==memory.estimate(legacy.options).organization,
                    "MTP reserve has no effect on kernel plans");
                auto edge=memory;
                edge.estimate_with_score_ring={};
                edge.estimate=[needed](const Options&) {return needed;};
                edge.total=residual.admission.at("required_bytes")+prefill_physical_margin;
                require(resolve(fit,edge).admitted,"1M exact physical total margin admits");
                --*edge.total;
                require(!resolve(fit,edge).admitted,"1M one-byte-short physical total margin refuses without fallback hiding the edge");
                for(const auto& [name,actual]:needed.owner_actuals) if(actual && needed.owner_caps.at(name)) {
                    auto low=residual.options;auto& m=low.mtp;
                    if(name=="mtp_predictor_runtime") m.runtime_bytes=actual-1U;
                    else if(name=="mtp_sidecar_retained") m.sidecar_retained_bytes=actual-1U;
                    else if(name=="mtp_sidecar_setup_peak") m.sidecar_setup_bytes=actual-1U;
                    else if(name=="mtp_sidecar_header") m.header_bytes=actual-1U;
                    else if(name=="mtp_target_hidden_captures") m.target_capture_bytes=actual-1U;
                    else if(name=="mtp_golden_file") m.golden_max_bytes=actual-1U;
                    else if(name=="mtp_replay_host") m.replay_host_bytes=actual-1U;
                    else if(name=="mtp_window_arena") m.window_arena_bytes=actual-1U;
                    else if(name=="mtp_window_union") m.window_expert_union_bytes=actual-1U;
                    else if(name=="mtp_window_arenas_captures_checkpoints") m.window_total_bytes=actual-1U;
                    else if(name.starts_with("mtp_window_capture_r")) m.window_capture_bytes=actual-1U;
                    else continue;
                    const auto rejected=compile_prompt_requirements(low,model,manifest,111334654400ULL,&sidecar);
                    require(rejected.owner_error.find("exceeds cap=")!=std::string::npos,"1M one-byte-short owner cap refuses by payload, not UMA: "+name);
                }
                if(saved_ring) ::setenv("STRIX_QSA_SCORE_RING_ROWS",saved_ring->c_str(),1);
                else ::unsetenv("STRIX_QSA_SCORE_RING_ROWS");
            }
            if(c!=33024U) { require(!result.admitted,"fixed 128 MiB predictor cap or descriptor must refuse larger C");continue; }
            require(result.admitted,result.error);
            const auto owners=compile_prompt_requirements(result.options,model,manifest,111334654400ULL,&sidecar);
            if(!dump) {
                require(owners.owner_phases.at("mtp_window_arenas_captures_checkpoints")==24U &&
                    owners.owner_phases.at("prefill_workspace_and_status")==7U,"real prompt/window handoff phases");
                for(const auto* name:{"mtp_predictor_kv_raw","mtp_q4_shared_weights","mtp_m1_keys"})
                    require((owners.owner_phases.at(name)&12U)==12U,"priming/cache identity owners overlap both phases");
            }
            if(factor==2U && result.admitted) {
                auto clean=arm;clean.environment.clear();
                const auto unchanged=resolve(clean,memory);
                require(unchanged.admitted && unchanged.admission.at("storage_ladder_step")==0U &&
                    unchanged.host_compilation==result.host_compilation,
                    "f2 empty environment retains step-zero compiler plan");
            }
            require(verified && result.ple_reclaim_credit>0,"mock verified PLE credit applied exactly once");
            require(effective_config_json(result).find("\"ple_verified_credit_bytes\":"+std::to_string(result.ple_reclaim_credit))!=std::string::npos,
                "inspection exposes verified credit instead of hardcoded zero");
            require(result.admission.at("gross_required_bytes")-result.ple_reclaim_credit==result.admission.at("required_bytes"),"single file credit");
            require(result.admission.at("owner_mtp_golden_host_bytes")<32U*1024U*1024U,"golden charges calibration payload, not six GiB cap");
            require(result.admission.at("owner_mtp_sidecar_retained_bytes")==mtp_payload::align(sidecar.file_size,4096U),"mapping charged once, without CLI spare capacity");
            if(dump) {
                require(result.admission.at("owner_mtp_replay_host_bytes")==0 && result.admission.at("owner_mtp_window_arenas_captures_checkpoints_bytes")==0,"dump does not create replay or windows");
                auto receipt_memory=memory;receipt_memory.total=UINT64_C(128)<<30U;receipt_memory.available=132064133120ULL;
                require(resolve(arm,receipt_memory).admitted,"exact supplied g-receipt available/UMA operands admit");
            } else {
                require(result.admission.at("owner_mtp_window_arenas_captures_checkpoints_bytes")<mtp.window_total_bytes/2U,"actual compiled R2/R4/R8 windows replace cap charge");
                // Captured arithmetic operands, with a synthetic aggregate owner
                // for the remaining peak. No device or model residency is inferred.
                for (const bool pool_off : {false,true}) {
                    auto captured=arm;
                    // Freeze this captured gross ledger; ladder sizing has separate integration tests.
                    captured.environment["STRIX_PLE_ROLLING_POSITIONS"]="1024";
                    captured.environment["STRIX_PLE_ROLLING_CAP_BYTES"]="3221225472";
                    captured.environment["STRIX_QSA_SCORE_RING_ROWS"]="512";
                    captured.environment["STRIX_PREFILL_PIPE"]="255";
                    captured.environment["STRIX_QWEN_YARN_HOST_RESERVE"]="1";
                    if(pool_off) captured.environment["STRIX_QSA_MTP_POOL"]="0";
                    const auto gross=pool_off?129170982139ULL:128991872251ULL;
                    constexpr auto limit=128849018880ULL, credit=25193033728ULL;
                    auto receipt=memory;
                    receipt.total=UINT64_C(128)<<30U; receipt.available=131757326336ULL;
                    receipt.gtt_limit=131757326336ULL;
                    unsigned drops=0,estimates=0;
                    receipt.estimate_with_score_ring={};
                    receipt.estimate=[&](const Options& candidate) {
                        ++estimates;
                        PrefillRequirements r; r.model_backing=111334654400ULL;
                        r.workspace=UINT64_C(1)<<30U; r.metadata=668821824;
                        r.device_base=8918706216ULL-r.workspace-candidate.rotary.table_bytes();
                        r.owners={{"model_backing",r.model_backing},
                            {"captured_other_owners",gross-(UINT64_C(8)<<30U)-r.model_backing-candidate.rotary.table_bytes()}};
                        return r;
                    };
                    receipt.verify_ple_credit=[&](const Options& candidate,const Environment& env) {
                        ++drops;
                        require(!estimates && candidate.mtp.enabled && candidate.known_prefix_prefill_rows==8192 &&
                            env.at("STRIX_QPF2K_SCHED")=="12", "verified drop precedes binding admission with resolved MTP organization");
                        return credit;
                    };
                    const auto admitted=resolve(captured,receipt);
                    const auto reader_owner=prefill_rolling_owner_bytes(admitted.options,model,manifest,
                        captured.environment,memory.ple_reclaim_page);
                    const auto complete_gross=gross+reader_owner;
                    require(admitted.admitted && drops==1 && estimates==1 && admitted.ple_reclaim_credit==credit &&
                        admitted.admission.at("gross_required_bytes")==complete_gross &&
                        admitted.admission.at("required_bytes")==complete_gross-credit &&
                        admitted.admission.at("device_growth_bytes")==8918706216ULL,
                        "captured zero-credit over-budget MTP operands admit only after verified credit");
                    require(effective_config_notice(admitted).find("verified PLE credit=25193033728 bytes")!=std::string::npos,
                        "YaRN notice prints the credit actually applied");
                    auto live=UmaOwnerInventory{}; live.reservation_upper_bound_bytes=111334654400ULL;
                    live.known_file_mapping_bytes=live.reservation_upper_bound_bytes;
                    const auto bridge=plan_known_prefill_reservation(live,gross-live.reservation_upper_bound_bytes,0,
                        {limit,limit},credit,reader_owner);
                    require(bridge.retained_reserved_bytes==admitted.admission.at("required_bytes") &&
                        bridge.reclaimable_file_mapping_credit_bytes==credit,"YaRN and bridge share one gross-to-net credit ledger");
                    receipt.verify_ple_credit={};
                    const auto zero=resolve(captured,receipt);
                    require(!zero.admitted && zero.admission.at("gross_required_bytes")==complete_gross &&
                        zero.admission.at("required_bytes")==complete_gross && complete_gross>limit,"unverified credit never admits either M1 arm");
                    auto inspection=captured; inspection.options.print_effective_config=true;
                    const auto provisional=resolve(inspection,receipt);
                    require(!provisional.admitted && provisional.admission.at("zero_credit_refusal_provisional")==1 &&
                        provisional.error.find("provisional zero-credit inspection refusal (runtime-setup-required)")!=std::string::npos &&
                        effective_config_json(provisional).find("runtime-setup-required")!=std::string::npos,
                        "inspection reports its zero-credit refusal as provisional");
                    receipt.verify_ple_credit=[&](const Options&,const Environment&) { return complete_gross-limit-1U; };
                    require(!resolve(captured,receipt).admitted,"insufficient verified credit still refuses");
                    receipt.verify_ple_credit=[&](const Options&,const Environment&) { return complete_gross-limit; };
                    require(resolve(captured,receipt).admitted,"verified boundary credit admits without double subtraction");
                    receipt.verify_ple_credit=[](const Options&,const Environment&) { return 0U; };
                    const auto failed=resolve(captured,receipt);
                    require(!failed.admitted && !failed.admission.at("zero_credit_refusal_provisional"),"failed automatic verification stays a binding refusal");
                    estimates=0;
                    receipt.verify_ple_credit=[](const Options&,const Environment&) -> std::uint64_t {
                        throw std::runtime_error("mincore verification failed");
                    };
                    require(!resolve(captured,receipt).admitted && !estimates,"verification error refuses before owner or device work");
                }

            }
            auto huge=arm;huge.options.mtp.golden_max_bytes*=2U;huge.options.mtp.replay_host_bytes*=2U;huge.options.mtp.window_total_bytes*=2U;
            const auto larger=resolve(huge,memory);
            require(larger.admitted && larger.admission.at("gross_required_bytes")==result.admission.at("gross_required_bytes"),"unused CLI cap never increases charge");
            const auto required=memory.estimate(result.options);
            for(const auto& [name,actual]:required.owner_actuals) if(actual && required.owner_caps.at(name)) {
                auto low=arm;
                auto& m=low.options.mtp;
                if(name=="mtp_predictor_runtime") m.runtime_bytes=actual-1U;
                else if(name=="mtp_sidecar_retained") m.sidecar_retained_bytes=actual-1U;
                else if(name=="mtp_sidecar_setup_peak") m.sidecar_setup_bytes=actual-1U;
                else if(name=="mtp_sidecar_header") m.header_bytes=actual-1U;
                else if(name=="mtp_target_hidden_captures") m.target_capture_bytes=actual-1U;
                else if(name=="mtp_golden_file") m.golden_max_bytes=actual-1U;
                else if(name=="mtp_replay_host") m.replay_host_bytes=actual-1U;
                else if(name=="mtp_window_arena") m.window_arena_bytes=actual-1U;
                else if(name=="mtp_window_union") m.window_expert_union_bytes=actual-1U;
                else if(name=="mtp_window_arenas_captures_checkpoints") m.window_total_bytes=actual-1U;
                else if(name.starts_with("mtp_window_capture_r")) m.window_capture_bytes=actual-1U;
                else continue;
                require(!resolve(low,memory).admitted,"cap smaller than actual payload refuses: "+name);
            }
            // Same C and shared descriptor give identical native and YaRN payload geometry.
            auto native=model;native.resolved_rotary={};
            const auto nr=mtp_payload::runtime(native,c),yr=mtp_payload::runtime(model,c);
            require(nr.sizes[19]==yr.sizes[19] && nr.sizes[20]==yr.sizes[20] && nr.reader==yr.reader,"native and YaRN C-sized caches and readers agree; descriptor-sized score scratch stays explicit");
        }
    }
}

}
int main() {
    try {
        const auto model = ModelContract::qwen38_flash_next();
        require(model.hidden_layers==48U && model.hidden_size==2560U && model.full_attention_layers==12U &&
            model.gdn_layers==36U && model.vocab_size==248320U,"canonical checkpoint geometry");
        const auto manifest = canonical_manifest(model);
        native_resident_golden_contract(manifest);
        compact_mtp_route_contract();
        mtp_phase_handoff_contract();
        mtp_receipt_admission(manifest);
        run(2,32768,32512,manifest);
        run(2,131072,130816,manifest);
        run(2,262144,261888,manifest);
        run(2,524288,8192,manifest);
        run(4,1048576,8192,manifest);
        run(2,300000,8192,manifest);
        run(2,300000,299000,manifest);
        run(4,600000,8192,manifest);
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
