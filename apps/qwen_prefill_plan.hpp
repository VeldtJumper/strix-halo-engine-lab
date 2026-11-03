#pragma once
#include "qwen_prompt_organization.hpp"
#include "qwen_yarn_owner_inventory.hpp"
#include "qwen_mtp_owner_plan.hpp"
#include "strix/compiled_speculative_verify.hpp"
#include "strix/ple_lookup.hpp"
namespace strix::decode_cli {
// Compile and size the admitted prompt before HIP setup or state mutation.
inline PrefillRequirements compile_prompt_requirements(const Options& options,
    const ModelContract& model, const ModelManifest& manifest, std::uint64_t model_bytes,
    const gguf::FileInfo* supplied_sidecar_info=nullptr, bool resident_fallback_inventory=false,
    std::uint32_t score_ring_rows=0) {
    auto requests = prompt_compiler_requests(options, model, manifest);
    auto& binding_request = requests.binding;
    auto& buffer_request = requests.buffers;
    auto& catalog_config = requests.catalog;
    kernels::KernelRegistry kernel_registry;
    register_complete_hip_kernel_catalog(model, binding_request, catalog_config, kernel_registry);
    auto compiled =
        strix::compile_serial_execution(
            model,
            kernel_registry,
            binding_request,
            buffer_request,
            256);

    YarnMtpGeometry mtp_geometry;
    mtp_geometry.target_tensor_count=manifest.tensors.size();
    const bool full_owners=options.rotary.mode()==RotaryMode::yarn || resident_fallback_inventory;
    if(full_owners && options.mtp.enabled && options.mtp.window_rows && options.mtp.golden_dump.empty()) {
        const auto access=compile_execution_tensor_access_plan(compiled.execution,manifest,build_model_tensor_index(manifest),
            build_serial_execution_tensor_requirements(compiled.execution));
        const auto rows=options.mtp.window_rows_set.empty()?std::vector<std::uint32_t>{options.mtp.window_rows}:options.mtp.window_rows_set;
        for(const auto r:rows) {
            const auto plan=compile_known_prefix_window(model,compiled,manifest,access,
                {true,true,true,r,options.mtp.window_expert_union,KnownWindowOutputPolicy::all_rows,
                 options.mtp.window_hyper_mix,options.mtp.window_qsa_rows,options.mtp.window_prefix,options.mtp.window_prefix,
                 options.mtp.window_prefix,options.mtp.window_prefix || options.known_window_router_rows,
                 options.mtp.window_prefix || options.known_window_router_rows,
                 options.rotary.mode()==strix::RotaryMode::yarn});
            mtp_geometry.window_arenas.emplace_back(r,plan.arena.physical.total_bytes);
            if(options.rotary.mode()==RotaryMode::yarn && !known_window_policy_serial_rows(plan))
                throw std::invalid_argument("YaRN MTP verifier recipe is not serial equivalent");
        }
    }
    const auto decode_scratch=compiled.arena.total_bytes;
    const auto decode_alignment=compiled.arena.alignment;
    const auto pair = numerical_profile_pair(options);
    if (pair.split()) {
        if (numerical_profile_fast_prefill(pair.prefill))
            binding_request.prefill_arithmetic_identity = prefill_arithmetic_identity(pair,
                options.resolved_prefill_option_mask.value_or(resolve_prefill_options(
                    [](const char* name) { return std::getenv(name); }, options.prefill_exact_fp16, options.prefill_disabled)),
                qpf2k::has_gsq_gate(manifest));
        catalog_config.numerical_profile = pair.prefill;
        catalog_config.fast_prefill_components = pair.fast_components;
        kernels::KernelRegistry prefill_registry;
        register_complete_hip_kernel_catalog(model, binding_request, catalog_config, prefill_registry);
        auto prefill = compile_serial_execution(model, prefill_registry, binding_request, buffer_request, 256);
        validate_sibling_serial_executions(prefill, compiled);
        compiled = std::move(prefill);
    }
    const auto index = build_model_tensor_index(manifest);
    const auto access = compile_execution_tensor_access_plan(compiled.execution, manifest, index,
        build_serial_execution_tensor_requirements(compiled.execution));
    validate_prompt_tensor_contract(options,manifest,compiled,access);
    const auto make_plan = [&](std::uint32_t rows) {
        return compile_known_prefix_prefill(model, compiled, manifest, access,
            {rows,true,true,true,true,KnownWindowOutputPolicy::terminal_row,
             options.rotary.mode() == RotaryMode::yarn ? UINT64_MAX : options.known_prefill_workspace_budget,
             options.rotary.mode() == RotaryMode::yarn ? UINT64_MAX : options.known_prefill_metadata_budget / 2U,
             true,true,true,true,true,false,compact_prompt_memory(options),true,score_ring_rows});
    };
    const auto plan = make_plan(options.known_prefix_prefill_rows);
    const auto add = prefill_checked_add;
    const auto mul = prefill_checked_mul;
    SerialSessionStateLayoutRequest state_request;
    state_request.context_tokens = options.context_tokens;
    state_request.recurrent_storage = options.gdn_recurrent_storage;
    const auto state = build_serial_session_state_layout(model, state_request);
    PrefillRequirements result;
    result.model_backing = model_bytes;
    result.state_d1 = add(state.device_total_bytes, add(state.host_total_bytes, add(state.host_total_bytes, state.host_alignment - 1U)));
    result.serial_scratch = compiled.arena.total_bytes;
    result.device_base = add(state.device_total_bytes, result.serial_scratch);
    // Automatic native HTTP separates registered backing from warmed file cache.
    // Registration aliases count once in UMA and also consume GTT; the uncovered
    // PLE pages are reserved separately on the host. Use registration geometry.
    if (options.automatic_prefill && options.http_serve && options.rotary.mode()==RotaryMode::native) {
#ifdef __linux__
        const auto page=::sysconf(_SC_PAGESIZE);
        if (page<=0) throw std::runtime_error("target registration page size unavailable");
#else
        throw std::runtime_error("target registration page size unavailable");
        const std::uint64_t page=0;
#endif
        account_http_registered_target(result,options,manifest,model_bytes,static_cast<std::uint64_t>(page));
    }
    // The serial decode pool (qsa_decode_pool_route) is retained device state from the first decode step on.
    const auto gsq = qpf2k::has_gsq_gate(manifest);
    const auto decode_pool = qsa_decode_pool_route(options, gsq) ? qsa_decode_pool_bytes(options, model, gsq) : 0U;
    const auto mtp_pool=qsa_mtp_pool_bytes(options,model,gsq);
    result.state_d1 = add(result.state_d1, mtp_pool);
    result.device_base = add(result.device_base, mtp_pool?mtp_pool-mtp_qsa_pool_plan(options.context_tokens).metadata_bytes:0U);
    result.state_d1 = add(result.state_d1, decode_pool);
    result.device_base = add(result.device_base, decode_pool);
    result.workspace = add(add(plan.arena.physical.total_bytes, 65552U),plan.qsa_score_ring_workspace_bytes);
    const auto map_bytes = mul(plan.logit_row_indices.size(), sizeof(std::uint32_t));
    RegisteredPrefillPlan registered;
    const bool registered_mtp=options.rotary.mode()==RotaryMode::yarn && options.mtp.enabled;
    std::uint64_t union_bytes=0;
    if (registered_mtp) {
        registered.base_runtime=add(add(decode_scratch,decode_alignment-1U),
            add(state.host_total_bytes,state.device_total_bytes?add(state.device_total_bytes,state.device_alignment-1U):0U));
        const auto ple_cache=PlannedPleLookupBackend::cache_reservation_bytes({options.ple_max_rows_per_io,
            options.ple_max_inflight_ios,options.ple_cpu_helper_threads,options.ple_cache_rows,options.ple_cache_budget_bytes,{}});
        registered.caller_live=add(add(state.host_total_bytes,state.host_alignment-1U),
            add(8U,add(ple_cache,add(plan.host_metadata_payload_bytes,map_bytes))));
        union_bytes=options.known_prefix_expert_union && plan.tile8.routed_expert_union_steps
            ? q51_known_rows::required_scratch_bytes(8U,static_cast<std::uint32_t>(model.num_experts),
                static_cast<std::uint32_t>(model.num_experts_per_token)) : 0U;
        registered.growth=add(add(plan.arena.physical.total_bytes,add(8U,union_bytes)),
            add(add(plan.host_metadata_payload_bytes,map_bytes),add(plan.qsa_score_ring_workspace_bytes,options.known_prefill_host_headroom)));
    }
    result.metadata = add(add(plan.host_metadata_payload_bytes, map_bytes), add(plan.host_metadata_payload_bytes, map_bytes));
    result.runtime_payload = add(mul(mul(plan.logical_rows, compiled.execution.steps.size()), sizeof(std::uint32_t)),
        add(map_bytes, add(mul(plan.logit_row_indices.size(), sizeof(ExecutionBufferAddress)), mul(model.vocab_size, 2U * sizeof(float)))));
    registered.temporary=result.runtime_payload;
    YarnPrefillOwnerGeometry geometry;
    geometry.compact_scores = plan.qsa_score_ring_workspace_bytes != 0;
    geometry.decode_pool_bytes = decode_pool;
    geometry.mtp_pool_bytes = mtp_pool;
    const auto collect_geometry = [&](const CompiledKnownPrefixPrefill& item) {
        geometry.ring_bytes = add(geometry.ring_bytes,item.qsa_score_ring_workspace_bytes);
        if (item.terminal_logits_arena)
            for (const auto& allocation : item.arena.physical.allocations)
                if (execution_use_is_logits(allocation.use))
                    geometry.terminal_logits_bytes = add(geometry.terminal_logits_bytes,allocation.bytes);
        geometry.compact_scores = geometry.compact_scores && item.qsa_score_ring_workspace_bytes != 0;
    };
    collect_geometry(plan);
    geometry.primary_compact_bytes = add(geometry.ring_bytes,geometry.terminal_logits_bytes);
    for (const auto rows : options.known_prefix_tail_rows) {
        const auto tail = make_plan(rows);
        if (registered_mtp)
            registered.growth=add(registered.growth,add(add(tail.arena.physical.total_bytes,add(8U,union_bytes)),
                add(tail.host_metadata_payload_bytes,tail.qsa_score_ring_workspace_bytes)));
        collect_geometry(tail);
        result.tails = add(result.tails, add(add(tail.arena.physical.total_bytes, 65552U),tail.qsa_score_ring_workspace_bytes));
        result.metadata = add(result.metadata, add(tail.host_metadata_payload_bytes, tail.host_metadata_payload_bytes));
        if (options.known_prefix_prefill_rows == 16384U)
            result.metadata = add(result.metadata, static_cast<std::uint64_t>(tail.logit_row_indices.size()) * 2U * sizeof(std::uint32_t));
    }
    if (full_owners)
        append_yarn_owner_requirements(options,model,state,model_bytes,result,geometry);
    if(full_owners && options.mtp.enabled) {
        // Header census, not a budget-derived guess of Q4 payload. Asset loader
        // independently verifies the pinned file/content before any execution.
        gguf::Limits limits;limits.max_metadata_entries=256;limits.max_tensors=64;
        const auto info=supplied_sidecar_info?*supplied_sidecar_info:gguf::read_file(options.mtp.sidecar,limits);
        if(info.file_size!=options.mtp.sidecar_bytes) throw std::invalid_argument("YaRN MTP sidecar size changed before admission");
        mtp_geometry.sidecar=info;
#ifdef __linux__
        const auto page=::sysconf(_SC_PAGESIZE);
        if(page<=0) throw std::invalid_argument("YaRN MTP page size unavailable");
        mtp_geometry.page_bytes=static_cast<std::uint64_t>(page);
#endif
        if(options.rotary.mode()==RotaryMode::yarn && !supplied_sidecar_info && options.mtp.golden_dump.empty() &&
            (options.mtp_bundle_path.empty() || !options.mtp.golden.empty()))
            require_mtp_forward_golden_geometry(options.mtp.golden,mtp_calibration_rows(options),
                mul(model.hidden_size,model.hc_count),model.vocab_size,options.context_tokens);
        const auto mtp=yarn_mtp_owners(options,model,mtp_geometry);
        result.owner_phases["mtp_m1_keys"]=result.owner_phases["mtp_m1_scratch_metadata"]=mtp_all^mtp_setup;
        result.owner_caps=mtp.caps;result.owner_actuals=mtp.actuals;result.owner_phases.insert(mtp.phases.begin(),mtp.phases.end());
        result.mtp_post_setup_peak=mtp.post_setup_peak;result.owner_error=mtp.refusal;
        for(const auto& [name,bytes]:mtp.bytes) result.owners.emplace(name,bytes);
        result.device_base=add(result.device_base,mtp.device);
    }
    if(mtp_prompt_phase_separation(options)) {
        for(const auto& [name,bytes]:result.owners) {
            (void)bytes;
            if(name=="prefill_workspace_and_status" || name=="tail_workspaces_and_status" ||
               name=="qsa_score_ring" || name=="terminal_logits_stripes" || name=="plan_metadata" ||
               name=="runtime_payload" || name=="qsa_shared_prefill_pool" || name=="qsa_batch_tables" ||
               name=="gdn_retained_rows_and_controls" || name=="merged_routed_csr" ||
               name=="routed_csr" || name=="ple_staging" || name=="ple_projections_and_gates" ||
               name=="ple_snapshot_and_hashes" || name=="ple_packed_growth_peak" ||
               (name.starts_with("tail_") && name.find("gdn_fp16_chunk_")==std::string::npos))
                result.owner_phases[name]=mtp_setup|mtp_calibration|mtp_prompt;
        }
    }
    if (options.http_serve && options.rotary.mode() == RotaryMode::native) {
        const auto external=qwen_prefill_external_memory(options,model);
        result.external_host=external.host;
        result.external_device=external.device;
    }
    result.organization.merge(std::map<std::string,std::uint64_t>{{"serial_execution_identity",compiled.execution_identity},
        {"serial_kernel_plan_identity",compiled.kernel_plan_identity},
        {"prefill_execution_identity",plan.prefill_execution_identity},
        {"prefill_kernel_plan_identity",plan.prefill_kernel_plan_identity},
        {"serial_scratch_bytes",compiled.arena.total_bytes},
        {"prefill_arena_bytes",plan.arena.physical.total_bytes},
        {"prefill_metadata_bytes",plan.host_metadata_payload_bytes},
        {"qsa_decode_pool_bytes",decode_pool},
        {"qsa_decode_pool_key_bytes",qsa_decode_pool_key_bytes(options, gsq)}});
    if(plan.compact_qsa_output_savings) result.organization.emplace("compact_qsa_output_savings_per_row",plan.compact_qsa_output_savings);
    if(plan.terminal_logits_arena) result.organization.emplace("terminal_logits_arena",1U);
    if(plan.qsa_score_ring_workspace_bytes) {
        result.organization.emplace("qsa_score_ring_workspace_bytes",plan.qsa_score_ring_workspace_bytes);
        result.organization.emplace("qsa_score_ring_rows",plan.qsa_score_ring_rows);
    }
    if(mtp_prompt_phase_separation(options)) result.organization.emplace("mtp_prompt_phase_separation",1U);
    if (registered_mtp) result.registered_plan=registered;
    if(mtp_pool) result.organization.emplace("qsa_mtp_pool_bytes",mtp_pool);
    return result;
}
} // namespace strix::decode_cli
