#pragma once
#include "qwen_yarn_memory.hpp"

#include "strix/compact_stage_images.hpp"
#include "hip_decode_options.hpp"
#include "known_prefill_reservation.hpp"
#include "qwen_moe_route_bf16.hpp"
#include "http_slot_capacity.hpp"
#include "qwen_prefill_external_memory.hpp"
#include "strix/known_prefix_prefill.hpp"
#include "strix/ctx_exact_memory.hpp"
#include "strix/q51_known_rows.hpp"
#include "strix/linux_uma_observation.hpp"
#include "strix/uma_owner_inventory.hpp"
#include <map>
#include <tuple>
#ifdef __linux__
#include <unistd.h>
#endif
#include "strix/session_state_layout.hpp"
#include "strix/serve/json.hpp"
#include <fstream>

namespace strix::decode_cli {
using Environment = std::map<std::string, std::string>;
enum class OptionSource { explicit_value, automatic, baseline };
inline constexpr std::size_t t16k_prompt_threshold = 32768U;
struct RequestedOptions {
    Options options;
    Environment environment;
    bool permit_generation_defaults = true;
};
struct PrefillCapabilities {
    bool a8_runtime = false;
    bool target_qualified = false;
    bool fp16_operands = false;
};
// These are simultaneous backing reservations, never RSS plus GTT. Registered
// aliases are already part of model_backing and must not be added a second time.
struct PrefillRequirements {
    std::uint64_t model_backing = 0, state_d1 = 0, serial_scratch = 0;
    std::uint64_t workspace = 0, metadata = 0, runtime_payload = 0, tails = 0;
    std::uint64_t device_base = 0;
    std::map<std::string, std::uint64_t> owners{},owner_caps{},owner_actuals{};
    std::map<std::string,unsigned> owner_phases{};
    std::uint64_t mtp_post_setup_peak=0;
    std::string owner_error{};
    std::map<std::string, std::uint64_t> organization{};
    std::uint64_t external_host = 0, external_device = 0;
};
// Future bridge commitments, excluding the already charged primary stack/prefill.
struct PrefillMemory {
    std::optional<std::uint64_t> available, gtt_limit, gtt_used;
    std::function<PrefillRequirements(const Options&)> estimate;
    std::function<HttpServingMemory(const Options&)> serving;
    // Present only for Linux reclaim support; host tests can supply the observation.
    std::optional<std::uint64_t> ple_reclaim_page{};
    std::optional<std::uint64_t> total{};
    // Must drop and verify with mincore. A predicted page union is never credit.
    std::function<std::uint64_t(const Options&, const Environment&)> verify_ple_credit{};
    // Optional explicit storage sizing for YaRN MTP, without mutating Options or process env.
    std::function<PrefillRequirements(const Options&, std::uint32_t)> estimate_with_score_ring{};
    std::uint64_t observation_id=0, end_steady_ns=0;
    std::optional<std::uint64_t> usable_ceiling{}, cache_available{}, gfx_target{}, resident{};
    bool cache_limits_known=false;
};
// Both inspection and execution validate explicit-T assets before the inspection exit.
inline void validate_http_serving_memory(const Options& o,const PrefillMemory& memory) {
    if(o.http_serve && o.mtp.enabled && o.known_prefix_prefill_rows) {
        if(!memory.serving) throw std::invalid_argument("one-slot:asset verification unavailable");
        (void)memory.serving(o);
    }
}
struct EffectiveConfig {
    Options options;
    unsigned prefill_option_mask = 0U;
    OptimizationProfileResolution organization;
    Environment environment;
    std::map<std::string, OptionSource> sources;
    std::string reason = "not-requested";
    std::string arithmetic_fallback;
    std::string tile_fallback;
    std::string serve_prefill_notice;
    // Present only when T16384 is tried; older/smaller-route receipts stay frozen.
    std::optional<OptionSource> t16k_policy_source;
    std::string error;
    std::uint64_t ple_reclaim_credit = 0;
    std::uint64_t ple_rolling_h = 0, ple_rolling_bound = 0, ple_rolling_cap = 0;
    std::vector<std::string> suppressed;
    bool admitted = false;
    bool gsq_owner = false;
    std::map<std::string, std::uint64_t> host_compilation;
    std::map<std::string, std::uint64_t> admission;
    RotaryIdentity rotary_identity;
};
inline void require_bitcalc_session_route(const Options& options, bool enabled) {
    if (enabled) {
        const auto refuse = [](bool active, const char* protocol) {
            if (active) throw std::invalid_argument(std::string("STRIX_BITCALC hooks refuse ") + protocol);
        };
        refuse(!options.session_save.empty(), "--session-save");
        refuse(!options.session_load.empty(), "--session-load");
        refuse(options.session_live_control, "--session-live-control");
        refuse(options.session_management_requested, "session management (--session-cache / STRIX_SESSION_CACHE / STRIX_SESSION_DISK)");
        refuse(!options.session_then_prompt.empty(), "--session-then-prompt");
        refuse(!options.session_logits.empty(), "--session-logits");
        refuse(options.session_first_generate != 0 || options.supplied_options.contains("--session-first-generate"), "--session-first-generate");
        refuse(options.session_validate_only, "--session-validate-only");
        refuse(options.session_tiled_suffix, "--session-tiled-suffix");
        refuse(options.session_mtp_fixed, "--session-mtp-fixed");
    }
}
inline bool env_is(const Environment& env, const char* name, std::string_view value) {
    const auto it = env.find(name);
    return it != env.end() && it->second == value;
}
// Storage locations are collected for policy resolution, never rotary execution.
inline bool storage_location_environment(std::string_view name) {
    return name == "HOME" || name == "XDG_CACHE_HOME";
}
template<class Get>
inline Environment requested_prefill_environment(Get get) {
    Environment env;
    for (const auto& item : qpf2k::defaults)
        if (!(qpf2k::pre_injected & qpf2k::bit(item.item)))
            if (const char* value = get(item.name)) env.emplace(item.name, value);
    for (const char* name : {"STRIX_QPF2K_DEFAULTS", "STRIX_QPF2K_MOE_DEFAULTED", "STRIX_BASELINE", "STRIX_PREFILL_F16",
            "STRIX_KNOWN_PREFIX_WINDOW", "STRIX_KNOWN_PREFIX_FULL_ROWS", "STRIX_KNOWN_PREFIX_EXPERT_UNION",
            "STRIX_KNOWN_WINDOW_ROUTER_ROWS", "STRIX_KNOWN_PREFIX_PREFILL_ROWS",
            "STRIX_RTX_EXPORT_STATE", "STRIX_QRTX_STEP_DUMP", "STRIX_PROMPT_BATCH_ROWS", "STRIX_PREFILL_LOGITS_DIGEST",
            "STRIX_PREFILL_EXACT_FP16", "STRIX_QPF2K_T16K", "STRIX_MODEL_COPY_ALL", "STRIX_PLE_RECLAIM_DECODE_BYTES",
            "STRIX_QSA_UNION_MERGE", "STRIX_QSA_SCORE_ROWS64", "STRIX_QSA_POOL_PACKED", "STRIX_QSA_DECODE_POOL",
            "STRIX_QSA_DECODE_SELECT", "STRIX_QSA_DECODE_POOL_BF16", "STRIX_QSA_DECODE_POOL_CHECK", "STRIX_PREFILL_MOE_MERGE_VEC", "STRIX_QSA_CACHE_ONLY",
            "STRIX_QSA_DECODE_FUSED_MAXIMA", "STRIX_QSA_DECODE_ATTN_V2", "STRIX_QSA_DECODE_VALUE_V2", "STRIX_MOE_R1_GLUE_FUSE",
            "STRIX_QPF2K_COMPACT_IMAGES", "STRIX_QPF2K_INDEX_BATCH", "STRIX_EAGER_CODE_LOAD", "HIP_ENABLE_DEFERRED_LOADING", "STRIX_QPF2K_EARLY_SETUP", "STRIX_CTX_EXACT_MEMORY", "STRIX_QSA_SCORE_RING", "STRIX_QSA_SCORE_RING_ROWS", "STRIX_TERMINAL_LOGITS_ARENA",
            "STRIX_SESSION_CACHE", "STRIX_SESSION_DISK", "XDG_CACHE_HOME", "HOME",
            "STRIX_PREFIX_BATCH_CERT", "STRIX_QWEN_YARN_MTP", "STRIX_QWEN_YARN_HOST_RESERVE", "STRIX_PLE_ROLLING_CAP_BYTES", "STRIX_PLE_ROLLING", "STRIX_PLE_ROLLING_POSITIONS", "STRIX_PREFIX_CACHE", "STRIX_PREFIX_CACHE_RETENTION",
            "STRIX_PREFIX_CACHE_INFLIGHT", "STRIX_PREFIX_CACHE_DISK", "STRIX_PREFIX_CACHE_DURABLE",
            "STRIX_SERVE_PREFILL", "STRIX_BATCH_PROMPT_SMALL", "STRIX_QPF2K_MOE_ROUTE_BF16",
            "STRIX_INTPREFILL_MOE_TUNE", "STRIX_QPF2K_MOE", "STRIX_QPF2K_MOE_SMALL"})
        if (!(qpf2k::pre_injected_moe_marker && std::string_view(name) == "STRIX_QPF2K_MOE_DEFAULTED"))
            if (const char* value = get(name)) env.emplace(name, value);
    for (const char* name : {"STRIX_BITCALC_LOGITS_OUT", "STRIX_BITCALC_FORCE_TOKENS"})
        if (get(name)) env.emplace(name, "1");
    // OFF keeps the complete historical inspection image, including its environment map.
    for(const char* name:{"STRIX_QSA_MTP_POOL","STRIX_QSA_MTP_POOL_CHECK","STRIX_MTP_LOGITS_DIGEST","STRIX_MTP_STEP_LOGITS_DIGEST"})
        if(const char* value=get(name); value && value[0]!='0') env.emplace(name,value);
    for (const auto& option : kPrefillOptions)
        if (const char* value = get(option.env.data())) env.emplace(option.env, value);
    return env;
}
inline Environment requested_prefill_environment() {
    return requested_prefill_environment([](const char* name) { return std::getenv(name); });
}
inline void apply_measured_prompt_organization(Options& o) {
    o.q51_slot_parallel_down = true;
    o.batched_moe_activation = true;
    o.wave8_q8_gdn_input = true;
    o.wave8_q8_gdn_output = true;
    o.prefetch_q8_lm_head = true;
    o.wave8_q8_lm_head = true;
    o.wave8_q8_qsa_output = true;
}
inline constexpr DecodeNumericalProfile dense_a8_profile =
    DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1;
inline std::uint64_t prefill_checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("prefill reservation overflow");
    return a + b;
}
// Automatic native HTTP: use the same page/coalescing geometry as registration.
// The source mmap's unregistered file cache remains reclaimable. Registered
// aliases consume UMA once and GTT once; no RSS-plus-GTT sum is meaningful.
inline void account_http_registered_target(PrefillRequirements& result,const Options& o,
    const ModelManifest& manifest,std::uint64_t file_bytes,std::uint64_t page) {
    if (o.residency!=HipModelResidencyMode::registered_file_mapping ||
        o.model_copy_tier!=HipModelCopyTier::none || hip_model_copy_all_mode())
        throw std::invalid_argument("automatic HTTP committed backing requires registered target");
    HipModelResidencyPolicy policy;
    policy.coalesce_gap_bytes=o.coalesce_gap_bytes;
    std::uint64_t committed=0;
    for (const auto range:build_registered_residency_ranges(manifest,policy,file_bytes,page))
        committed=prefill_checked_add(committed,range.end-range.begin);
    result.model_backing=committed;
    result.device_base=prefill_checked_add(result.device_base,committed);
    result.organization["target_mapped_file_bytes"]=file_bytes;
    result.organization["target_registered_backing_bytes"]=committed;
}
inline std::uint64_t prefill_checked_multiply(std::uint64_t a, std::uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error("prefill reservation overflow");
    return a * b;
}
struct PrefillNativePools {
    std::uint64_t host = 0, device = 0, mapped = 0;
    std::map<std::string, std::uint64_t> bytes;
};
// Canonical single-stream prompt route. Price cached high-water capacities and
// allocation replacement peaks, independently of reclaimable model backing.
inline PrefillNativePools prefill_native_pools(const Options& o, const ModelContract& m) {
    PrefillNativePools result;
    const auto add = prefill_checked_add, mul = prefill_checked_multiply;
    const std::uint64_t t = known_prefill_pool_rows(o.known_prefix_prefill_rows);
    const auto charge = [&](const char* name, std::uint64_t bytes, bool device = true, bool mapped = false) {
        result.bytes.emplace(name, bytes);
        auto& total = device ? result.device : result.host;
        total = add(total, bytes);
        if (mapped) result.mapped = add(result.mapped, bytes); // alias: GTT only, never a second UMA charge
    };
    charge("native_moe_a16_da16", mul(t, 2U * (2560U + 10U * 640U)));
    const auto counters = mul(2U * (t * 10U / 64U + 1024U) + t * 16U + 64U +
        514U + (t * 10U + 2047U) / 2048U * 512U, 4U);
    charge("native_moe_counters_csr", counters);
    std::uint64_t csr = q51_known_rows::prefill_scratch_bytes(o.known_prefix_prefill_rows, 512U, 10U);
    for (const auto rows : o.known_prefix_tail_rows)
        csr = add(csr, q51_known_rows::prefill_scratch_bytes(rows, 512U, 10U));
    charge("native_moe_owner_csr", mul(csr, 2U)); // full owners plus a merged partial union
    charge("native_hc_fold", mul(t, 42240U));
    charge("native_hc_lows_packs_inverse", add(mul(t, 320U * 6U + 4U * 4U), 2U * 2U * 320U * 10240U + 256U));
    charge("native_hc_norm16", mul(t, (10240U + 64U) * 2U)); // optional, conservatively reserved
    // Setup replaces one scratch allocation at a time after draining readers.
    // The largest old pool (normalized16) can coexist with its replacement.
    charge("native_setup_growth_peak", mul(t, (10240U + 64U) * 2U));
    charge("native_gdn_core16", mul(t, (6144U + 64U) * 2U));
    charge("native_shared_pre", mul(t, (640U + 64U) * 2U));
    // Two maximum-width tiled operand images; preserve each launch's <4 GiB guards.
    charge("native_dense_operands", mul(t, 2U * (10240U + 64U) * 2U));
    std::uint64_t ple_rows = o.known_prefix_prefill_rows;
    for (const auto rows : o.known_prefix_tail_rows) ple_rows = add(ple_rows, rows);
    charge("native_ple_staging", mul(ple_rows, 2U * PleHashContract::maximum_heads * 512U), false, true);
    charge("native_ple_hashes", mul(ple_rows, sizeof(PleHashResult)), false);
    charge("native_ple_projections_gates", mul(ple_rows, add(mul(m.hidden_size, m.hc_count + 1U), m.hc_count) * 4U));
    const auto conv = mul(mul(mul(m.hidden_size, m.hc_count), m.ple_conv_kernel_size - 1U), m.ngram_size * m.bytes_bf16);
    charge("native_ple_snapshots", mul(conv, o.known_prefix_tail_rows.size() + 1U));
    // Each PLE owner retains its packed slab; one replacing allocation can coexist.
    charge("native_ple_packed_peak", mul(add(ple_rows, o.known_prefix_prefill_rows), mul(m.ple_embed_dim, 4U)));
    const auto groups = known_prefill_stage_groups(o.known_prefix_prefill_rows);
    if (!m.indexer_compress_ratio) throw std::invalid_argument("prefill indexer ratio is zero");
    const auto blocks = o.context_tokens / m.indexer_compress_ratio;
    // Two fixed ring slots; retain the historical third-slab admission margin.
    charge("native_qsa_maxima_peak", mul(mul(3U * t, (blocks + 3U) / 4U), 4U));
    charge("native_qsa_tables", mul(2U * groups, 64U * 64U * 8U));
    // Includes two pinned slots plus the temporary preparation vectors. HIP asserts
    // the descriptor sum fits this per-group upper bound.
    charge("native_qsa_descriptors", mul(3U * groups, 20480U), false, true);
    charge("native_qsa_keys_ranges", mul(blocks, m.indexer_head_dim * 4U + 8U));
    // Residual packs, driver and small host vectors; replaces the old 4 GiB
    // catch-all. None of the explicitly priced pools above are paid from it.
    charge("native_residual", UINT64_C(1) << 30U);
    return result;
}
// Predict only the route's reclaim credit. Runtime still verifies the drop and
// admits allocations against the actual credit, including zero on setup failure.
inline std::optional<ple_reclaim::Config> prefill_reclaim_config(const Options& options, const ModelContract& model,
    const ModelManifest& manifest, const Environment& environment, std::optional<std::uint64_t> page) {
    if (!page || !*page) return {};
    const auto mode = environment.find("STRIX_PLE_RECLAIMABLE");
    if (mode != environment.end() && mode->second != "auto" && mode->second != "1") return {};
    const auto copy = environment.find("STRIX_MODEL_COPY_ALL");
    if (copy != environment.end() && !copy->second.empty() && copy->second[0] != '0') return {};
    HipModelResidencyPolicy policy;
    policy.copy_tier = options.model_copy_tier;
    const bool graph = options.execution.mode != DecodeExecutionMode::eager;
    policy.promote_file_backed_candidates = graph;
    const auto get = [&](const char* name) -> const char* {
        const auto it = environment.find(name);
        return it == environment.end() ? nullptr : it->second.c_str();
    };
    const char* controller = get("STRIX_MTP_CONTROLLER");
    const bool leased_mtp = ctx_exact_memory_enabled(get,"STRIX_PLE_ROLLING") &&
        (!controller || (std::string_view(controller) != "depth" && std::string_view(controller) != "survival"));
    auto route_options=options;
    route_options.ple_rolling_explicit_opt_in=env_is(environment,"STRIX_PLE_ROLLING","1");
    if (!ple_reclaim_route_supported(route_options, policy, graph, leased_mtp)) return {};
    if (options.mtp.enabled && (options.tokens.size() > options.context_tokens ||
        options.generate_tokens > options.context_tokens - options.tokens.size())) return {};
    try {
        const ManifestTensor* table = nullptr;
        std::vector<ple_reclaim::Interval> registered;
        for (const auto& tensor : manifest.tensors) {
            if (tensor.residency == ResidencyPolicy::file_backed_candidate) {
                if (table || tensor.role != TensorRole::ngram_table || tensor.dimensions.size() != 2U ||
                    (tensor.ggml_type != 20U && tensor.ggml_type != 30U) ||
                    (tensor.ggml_type == 20U && tensor.dimensions[0] % 32U)) return {};
                table = &tensor;
            } else if (tensor.placement.kind == PlacementKind::local) {
                registered.push_back({tensor.absolute_offset / *page * *page,
                    prefill_checked_add(prefill_checked_add(tensor.absolute_offset, tensor.byte_size), *page - 1U) / *page * *page});
            }
        }
        if (!table) return {};
        std::sort(registered.begin(), registered.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
        std::vector<ple_reclaim::Interval> merged;
        for (const auto& range : registered) {
            if (!merged.empty() && range.begin <= prefill_checked_add(merged.back().end, options.coalesce_gap_bytes))
                merged.back().end = std::max(merged.back().end, range.end);
            else merged.push_back(range);
        }
        ple_reclaim::Config config;
        config.page = *page;
        config.table_offset = table->absolute_offset;
        config.table_rows = table->dimensions[1];
        config.stored_row_bytes = table->ggml_type == 20U ? table->dimensions[0] / 32U * 18U : table->dimensions[0] * 2U;
        if (!config.stored_row_bytes || config.table_rows > UINT64_MAX / config.stored_row_bytes ||
            config.stored_row_bytes * config.table_rows != table->byte_size) return {};
        config.mapping_bytes = prefill_checked_add(config.table_offset, table->byte_size);
        config.reclaim = ple_reclaim::reclaim_intervals(config.table_offset, table->byte_size, *page, std::move(merged));
        config.rolling_positions = ple_rolling_prompt_supported(route_options) && ctx_exact_memory_enabled(get,"STRIX_PLE_ROLLING") ? ple_reclaim::rolling_positions(get("STRIX_PLE_ROLLING"), get("STRIX_CTX_EXACT_MEMORY"),
            get("STRIX_PLE_ROLLING_POSITIONS")) : 0U;
        std::uint64_t reclaim_bytes = 0;
        for (const auto& r : config.reclaim) reclaim_bytes += r.end - r.begin;
        const auto interior = (config.table_offset + table->byte_size) / *page * *page -
            (config.table_offset + *page - 1U) / *page * *page;
        if (reclaim_bytes != interior) config.rolling_positions = 0;
        if (options.mtp.enabled && !config.rolling_positions) return {};
        const auto contract = build_ple_hash_contract(model);
        const auto allowance = environment.find("STRIX_PLE_RECLAIM_DECODE_BYTES");
        config.decode_allowance_bytes = ple_reclaim::decode_allowance(config.rolling_positions ? 0U : options.generate_tokens, 1U, contract.head_count, *page,
            allowance == environment.end() ? 0U : std::strtoull(allowance->second.c_str(), nullptr, 10));
        config.bounded_turns = options.mtp.enabled;
        if (config.rolling_positions)
            config.rolling_cap_bytes = ple_reclaim::rolling_cap_bytes(get("STRIX_PLE_ROLLING_CAP_BYTES"));
        return config;
    } catch (const std::exception&) { return {}; }
}
inline std::uint64_t prefill_reclaim_credit(const Options& options, const ModelContract& model,
    const ModelManifest& manifest, const Environment& environment, std::optional<std::uint64_t> page,
    std::uint64_t* rolling_h = nullptr, std::uint64_t* rolling_bound = nullptr, std::uint64_t* rolling_cap = nullptr) {
    if (rolling_cap) *rolling_cap = 0;
    if (rolling_h) *rolling_h = 0;
    if (rolling_bound) *rolling_bound = 0;
    try {
        const auto config = prefill_reclaim_config(options,model,manifest,environment,page);
        if (!config) return 0;
        const auto stats = ple_reclaim::plan_prompt(*config,build_ple_hash_contract(model),options.tokens).stats;
        if (rolling_cap && config->rolling_positions)
            *rolling_cap = ple_reclaim::rolling_budget(*config, build_ple_hash_contract(model).head_count).cap;
        if (rolling_h) *rolling_h = config->rolling_positions;
        if (rolling_bound && config->rolling_positions)
            *rolling_bound = prefill_checked_add(stats.admitted_bytes, config->stored_row_bytes * config->table_rows - stats.reclaimable_bytes);
        return stats.credit_bytes;
    } catch (const std::exception&) { return 0; }
}
inline void require_effective_prefill_execution(const EffectiveConfig& config) {
    if (!config.error.empty()) throw std::invalid_argument(config.error);
}
inline std::uint64_t prefill_rolling_owner_bytes(const Options& options, const ModelContract& model,
    const ModelManifest& manifest, const Environment& environment, std::optional<std::uint64_t> page) {
    const auto config = prefill_reclaim_config(options,model,manifest,environment,page);
    return config && config->rolling_positions ? ple_reclaim::planned_owner_bytes(*config,
        build_ple_hash_contract(model).head_count,options.tokens.size()) : 0U;
}
// Shared CPU preflight for execution and HTTP --print-effective-config. Capacity
// requiring actual device allocations is resolved later and remains provisional.
inline void validate_effective_startup(const EffectiveConfig& config) {
    require_effective_prefill_execution(config);
    validate_http_serve(config.options);
    validate_prefix_state_cache(config.options);
}
inline void validate_effective_options(const Options& o, const ModelContract& model) {
    validate_resolved_mtp_cli(o);
    validate_resolved_prefix_roundtrip(o);
    validate_resolved_q8_projection_organization(o);
    validate_resolved_hc_projection_next_tile(o);
    validate_resolved_q51_slot_parallel_down(o);
    validate_resolved_gdn_islands(o);
    validate_resolved_gdn_recurrent_layout(o);
    validate_resolved_gdn_recurrent_storage(o);
    validate_resolved_attention_kv_storage(o);
    validate_resolved_qsa_hierarchical_selection(o);
    validate_resolved_qsa_gqa(o);
    validate_resolved_known_prefix_window(o);
    validate_moe_token_cost_options(o);
    validate_resolved_prompt_lookup(o);
    validate_resolved_serving(o);
    validate_resolved_ple_cpu_helpers(o);
    validate_resolved_ple_cache(o);
    validate_resolved_device_queue_trace(o);
    validate_resolved_requested_logprobs(o, model.vocab_size);
    validate_resolved_teacher_forced_score(o, model.vocab_size);
}
// Cache identity includes the resolved geometry, profiles and calibration chain.
inline std::string mtp_bundle_golden_identity(const Options& options, const ModelContract& model) {
    std::ifstream bundle_input(options.mtp_bundle_path,std::ios::binary);
    if (!bundle_input) throw std::runtime_error("cannot reread MTP bundle");
    std::string identity_text((std::istreambuf_iterator<char>(bundle_input)),{});
    identity_text += model.rotary_identity().json()+std::to_string(options.context_tokens)+":"+
        std::to_string(numerical_profile_identity(options.numerical_profile))+":"+
        std::to_string(numerical_profile_identity(numerical_profile_pair(options).prefill));
    const auto calibration_tokens=mtp_tiled_prompt(options)
        ? mtp_calibration_rows(options)+1U : options.tokens.size();
    for (std::size_t i=0;i<calibration_tokens;++i) identity_text += ":"+std::to_string(options.tokens[i]);
    return model_content_identity_hex(sha256_bytes(
        std::as_bytes(std::span<const char>(identity_text.data(),identity_text.size()))));
}
inline void select_mtp_bundle_golden(Options& options, const ModelContract& model) {
    if (options.mtp_bundle_path.empty() || !options.mtp.enabled) return;
    const auto identity=mtp_bundle_golden_identity(options,model);
    const auto shipped=std::filesystem::path(options.mtp_bundle_path).parent_path()/"goldens"/(identity+".golden");
    const auto cached=std::filesystem::is_regular_file(shipped)?shipped:mtp_golden_cache_dir()/(identity+".golden");
    options.mtp.golden=std::filesystem::is_regular_file(cached)?cached.string():std::string{};
}
inline EffectiveConfig initialize_prefill_config(const RequestedOptions& request) {
    EffectiveConfig result;
    result.options = request.options;
    result.environment = request.environment;
    const bool exact_defaults = !request.options.baseline && !env_is(request.environment, "STRIX_BASELINE", "1") &&
        !(request.environment.contains("STRIX_QPF2K_DEFAULTS") && request.environment.at("STRIX_QPF2K_DEFAULTS").starts_with('0'));
    if (!exact_defaults) result.environment["STRIX_QPF2K_COMPACT_IMAGES"] = "0";
    const auto enabled = [&](const char* name) {
        const auto it = request.environment.find(name);
        return it == request.environment.end() || !it->second.starts_with('0');
    };
    result.options.early_setup = exact_defaults && request.options.early_setup && enabled("STRIX_QPF2K_EARLY_SETUP");
    result.options.eager_code_load = exact_defaults && request.options.eager_code_load && enabled("STRIX_EAGER_CODE_LOAD");
    if (result.options.eager_code_load && !result.environment.contains("HIP_ENABLE_DEFERRED_LOADING"))
        result.environment["HIP_ENABLE_DEFERRED_LOADING"] = "0";
    result.prefill_option_mask = resolve_prefill_options([&](const char* name) -> const char* {
        const auto it = result.environment.find(name);
        return it == result.environment.end() ? nullptr : it->second.c_str();
    }, result.options.prefill_exact_fp16, result.options.prefill_disabled);
    // Automatic YaRN MTP profiles and verified bundles retain upstream arithmetic.
    // Governed candidates remain excluded until their MTP gates pass.
    if (result.options.mtp.enabled && (!result.options.mtp_bundle_path.empty() ||
        (result.options.rotary.mode() == RotaryMode::yarn && !result.options.prefill_numerical_profile))) {
        const auto excluded = prefill_option_bit(PrefillOption::hc_output_wmma) |
            prefill_option_bit(PrefillOption::index_proj_wmma) | prefill_option_bit(PrefillOption::gdn_fp16_chunk);
        result.options.prefill_disabled |= excluded;
        result.prefill_option_mask &= ~excluded;
    }
    result.options.resolved_prefill_option_mask = result.prefill_option_mask;
    return result;
}
#include "qwen_yarn_admission.inc"

// Test proposed exports against the actual YaRN environment admission before
// touching process state. Explicit requests keep their existing refusal contract.
inline void admit_product_profile(Options& o) {
    if (o.rotary.mode() != RotaryMode::yarn) return;
    for (auto& entry : o.product_profile.entries) {
        if (entry.status != qwen::profile::Status::set && entry.status != qwen::profile::Status::unqualified &&
            entry.status != qwen::profile::Status::provisional) continue;
        Environment candidate;
        if (!entry.lever->env.empty()) candidate.emplace(entry.lever->env, entry.value);
        const auto reason = yarn_environment_refusal(candidate, o.rotary.factor());
        if (reason.empty()) continue;
        entry.status = qwen::profile::Status::pending;
        entry.target = std::string(entry.lever->target);
        entry.value = std::string(entry.lever->builtin);
        entry.reason = reason;
    }
}

// Diagnostic or execution controls that keep a request on its explicit route.
inline bool explicit_execution_control(const Environment& environment) {
    for (const char* name : {"STRIX_RTX_EXPORT_STATE", "STRIX_QRTX_STEP_DUMP", "STRIX_PROMPT_BATCH_ROWS", "STRIX_PREFILL_LOGITS_DIGEST"}) {
        const auto control = environment.find(name);
        if (control != environment.end() && !control->second.empty() && control->second != "0") return true;
    }
    return false;
}
// The checkpoint content the automatic organization is qualified for (not GSQ, the validated content identity).
inline bool qualified_checkpoint_content(const ModelManifest& manifest, std::string_view verified_content) {
    return !qpf2k::has_gsq_gate(manifest) && verified_content == validated_model_content_identity;
}
// The plain documented MTP command: no oracle or state it writes or reads besides the independent golden (no
// golden dump, metadata-only diagnostic, RTX state import, session save/load/continuation or prefix-cache restore),
// greedy selection (no sampling option) and the default n-gram policy (no explicit --mtp-ngram-draft mode, no
// copy-wide rows). The default auto n-gram extension is part of the plain command and stays admitted.
inline bool plain_mtp_command(const Options& o) noexcept {
    const auto& m = o.mtp;
    return m.golden_dump.empty() && !m.metadata_only && !o.rtx_state_import &&
        !m.sample_options && !mtp_sampling(m) && !m.sample_timing_schedule &&
        !m.ngram_draft && !m.ngram_off && !m.ngram_context && !m.ngram_copy_edit && !m.ngram_wide_only &&
        m.copy_wide_rows.empty() && m.cohort_mode.empty() &&
        o.session_save.empty() && o.session_load.empty() && o.session_then_prompt.empty() && o.session_logits.empty() &&
        !o.session_live_control && !o.session_mtp_fixed &&
        !o.prefix_cache_host_bytes && o.prefix_cache_dir.empty() && o.prefix_cache_then_prompts.empty();
}
// The single-session MTP route: MTP owned only by the documented MTP window organization
// (mtp_window_organization_option: the documented group 3-4 --mtp* options, the window kernel organization and the
// known-prefix window options; any other route-owning or low-level option excludes it), the plain MTP command
// (plain_mtp_command), no cohort, HTTP, serving prefix, prefix roundtrip, prompt lookup, scoring, recording or
// final-state validation, no diagnostic/execution control, a qualified target and checkpoint. Its declined
// resolution also takes STRIX_DISPATCH_FUSE. Every fused form is
// operand-pinned to, and fixture-checked byte for byte against, the separate launches, so the window organization
// flags do not change the result; the exclusions only keep routes whose configuration identity, diagnostics or
// qualification must not change. YaRN resolves separately; STRIX_DISPATCH_FUSE=0 (or any explicit value) wins.
inline bool single_session_mtp_route(const Options& o, const Environment& environment, const ModelManifest& manifest,
    std::string_view verified_content, const PrefillCapabilities& capabilities) {
    return o.mtp.enabled && o.mtp.cohort.empty() && !o.route_owned_beyond_mtp && plain_mtp_command(o) &&
        !o.http_serve && o.serving_prefix_files.empty() && !o.prefix_cache_roundtrip &&
        !o.prompt_lookup && !o.teacher_forced_score && o.raw_recording_path.empty() && o.validation_final_state_path.empty() &&
        o.rotary.mode() != RotaryMode::yarn && !explicit_execution_control(environment) &&
        qualified_checkpoint_content(manifest, verified_content) && capabilities.target_qualified;
}
inline EffectiveConfig resolve_prefill_effective_config(const RequestedOptions& request,
    const ModelContract& model, const gguf::FileInfo& info, const ModelManifest& manifest,
    std::string_view verified_content, const PrefillCapabilities& capabilities, const PrefillMemory& memory) {
    if (request.options.rotary.mode() == RotaryMode::yarn && !request.options.baseline &&
        !env_is(request.environment, "STRIX_BASELINE", "1"))
        return resolve_yarn_config(request, model, info, manifest, verified_content, capabilities, memory);
    auto result = initialize_prefill_config(request);
    result.options.ple_rolling_explicit_opt_in=env_is(request.environment,"STRIX_PLE_ROLLING","1");
    result.options.qsa_mtp_pool_explicit_opt_in=env_is(request.environment,"STRIX_QSA_MTP_POOL","1");
    result.gsq_owner = qpf2k::has_gsq_gate(manifest);
    if (!env_is(result.environment, "STRIX_QPF2K_T16K", "1")) result.environment.erase("STRIX_QPF2K_T16K");
    for (const auto& name : request.options.supplied_options) result.sources[name] = OptionSource::explicit_value;
    auto& o = result.options;
    const bool baseline = o.baseline || env_is(request.environment, "STRIX_BASELINE", "1");
    if (o.rotary.mode() == RotaryMode::native && o.supplied_options.contains("--rope-yarn") &&
        result.environment.contains("STRIX_QWEN_ROPE_YARN")) {
        result.suppressed.emplace_back("STRIX_QWEN_ROPE_YARN");
        result.environment["STRIX_QWEN_ROPE_YARN"] = "off";
        result.sources["STRIX_QWEN_ROPE_YARN"] = OptionSource::explicit_value;
    }
    const auto inject = [&](unsigned allowed) {
        const unsigned injected = qpf2k::initialize([&](const char* name) -> const char* {
            const auto it = result.environment.find(name);
            return it == result.environment.end() ? nullptr : it->second.c_str();
        }, [&](const char* name, const char* value) { result.environment.emplace(name, value); return true; }, allowed);
        for (const auto& entry : qpf2k::defaults)
            if (result.environment.contains(entry.name)) result.sources[entry.name] =
                injected & qpf2k::bit(entry.item) ? OptionSource::automatic : OptionSource::explicit_value;
        if (result.environment.contains("STRIX_QPF2K_MOE_DEFAULTED"))
            result.sources["STRIX_QPF2K_MOE_DEFAULTED"] = request.environment.contains("STRIX_QPF2K_MOE_DEFAULTED") ?
                OptionSource::explicit_value : OptionSource::automatic;
    };
    // Baseline disables the new policy and all controlled environment items;
    // argv organization/arithmetic still goes through the historical resolver.
    o.baseline = baseline;
    if (baseline) {
        if (o.rotary.mode() == RotaryMode::yarn) {
            o.suppressed_yarn_request = "--rope-yarn=" + std::to_string(o.rotary.factor());
            o.rotary = validate_qwen_rotary_request(0,o.context_explicit,o.context_tokens,true);
        }
        if (result.environment.contains("STRIX_QWEN_ROPE_YARN")) {
            result.suppressed.emplace_back("STRIX_QWEN_ROPE_YARN");
            result.environment["STRIX_QWEN_ROPE_YARN"] = "off";
            result.sources["STRIX_QWEN_ROPE_YARN"] = OptionSource::baseline;
        }
        for (const auto& entry : qpf2k::defaults) {
            if (result.environment.contains(entry.name)) result.suppressed.emplace_back(entry.name);
            result.environment[entry.name] = "0";
            result.sources[entry.name] = OptionSource::baseline;
        }
        result.environment["STRIX_QPF2K_DEFAULTS"] = "0";
        result.sources["STRIX_QPF2K_DEFAULTS"] = OptionSource::baseline;
    }
    const bool single_session_mtp = single_session_mtp_route(o, request.environment, manifest, verified_content, capabilities);
    const auto decline = [&](std::string reason) {
        result.reason = std::move(reason);
        if (!baseline) inject(qpf2k::historical_items |
            (single_session_mtp ? qpf2k::bit(qpf2k::Item::dispatch_fuse) : 0U));
        if (o.prefill_profile_deferred && !o.numerical_profile_explicit) o.numerical_profile = dense_a8_profile;
        const auto schedule = result.environment.find("STRIX_QPF2K_SCHED");
        const bool injected_gsq = qpf2k::has_gsq_gate(manifest) && schedule != result.environment.end() &&
            !request.environment.contains("STRIX_QPF2K_SCHED");
        apply_default_known_prefill_tails(o, injected_gsq || schedule == result.environment.end() ? "0" : schedule->second.c_str());
        result.organization = resolve_optimization_profile(o, model, info, manifest, verified_content,
            capabilities.a8_runtime, request.permit_generation_defaults);
        auto validation_organization = result.organization;
        apply_numerical_profile(o, model, manifest, verified_content, capabilities.a8_runtime, validation_organization);
        if (!http_prefill_pending(o)) validate_effective_options(o, model);
        if (o.known_prefix_prefill_rows) {
            auto pool_options = o;
            if (const auto group = result.environment.find("STRIX_PROMPT_BATCH_ROWS"); group != result.environment.end()) {
                const auto rows = std::strtoull(group->second.c_str(), nullptr, 10);
                if (rows > known_prefill_max_rows || (rows && rows % 8U))
                    throw std::invalid_argument("STRIX_PROMPT_BATCH_ROWS must be a multiple of 8 up to 16384");
                if (rows > o.known_prefix_prefill_rows) {
                    // The group owner coexists with the original main owner.
                    pool_options.known_prefix_tail_rows.push_back(o.known_prefix_prefill_rows);
                    pool_options.known_prefix_prefill_rows = static_cast<std::uint32_t>(rows);
                }
            }
            const auto pools = prefill_native_pools(pool_options, model);
            result.host_compilation.insert(pools.bytes.begin(), pools.bytes.end());
            result.host_compilation["native_host_bytes"] = pools.host;
            result.host_compilation["native_device_bytes"] = pools.device;
            result.host_compilation["native_gtt_bytes"] = prefill_checked_add(pools.device, pools.mapped);
            if (o.known_prefill_host_headroom < prefill_checked_add(pools.host, pools.device))
                result.error = "known prefill native pools exceed --known-prefix-host-headroom-bytes; required " +
                    std::to_string(prefill_checked_add(pools.host, pools.device));
        }
        if (result.reason == "prompt-exceeds-model-context")
            result.error = "Qwen auto-prefill declined: prompt-exceeds-model-context (prompt + generation exceeds " +
                std::to_string(model.max_position_embeddings) + ")";
        else if (!o.known_prefix_window && o.tokens.size() > o.context_tokens)
            result.error = "Qwen auto-prefill declined: " + result.reason + "; prompt has " + std::to_string(o.tokens.size()) +
                " tokens, exceeding serial prefill context " + std::to_string(o.context_tokens) + "; cannot use serial prefill";
        if (o.mtp.enabled)
            result.ple_reclaim_credit = prefill_reclaim_credit(o, model, manifest, result.environment,
                memory.ple_reclaim_page, &result.ple_rolling_h, &result.ple_rolling_bound, &result.ple_rolling_cap);
        return result;
    };
    if (baseline) return decline("baseline");
    const auto exact_environment = [&](const char* name) -> const char* {
        const auto it = request.environment.find(name);
        return it == request.environment.end() ? nullptr : it->second.c_str();
    };
    if (ctx_exact_memory_enabled(exact_environment,"STRIX_QSA_SCORE_RING"))
        (void)qsa_score_ring_rows_override(exact_environment("STRIX_QSA_SCORE_RING_ROWS"));
    if (o.route_owned || o.any_low_level_flag || o.known_prefix_window || o.prompt_lookup || o.http_serve ||
        o.prefix_cache_roundtrip || !o.serving_prefix_files.empty() || o.mtp.supplied || o.mtp.enabled || !o.mtp.cohort.empty() || o.teacher_forced_score ||
        !o.raw_recording_path.empty() || !o.validation_final_state_path.empty()) return decline("explicit-route");
    if (explicit_execution_control(request.environment)) return decline("explicit-diagnostic-or-execution-control");
    for (const auto& entry : qpf2k::defaults)
        if (!(qpf2k::exact_prefill_items & qpf2k::bit(entry.item)) &&
            env_is(request.environment, entry.name, "0")) return decline("feature-disabled");
    const auto defaults = request.environment.find("STRIX_QPF2K_DEFAULTS");
    if (defaults != request.environment.end() && !defaults->second.empty() && defaults->second[0] == '0')
        return decline("feature-disabled");
    if (env_is(request.environment, "STRIX_PREFILL_F16", "0")) return decline("feature-disabled");
    if (!o.disabled_prefill_features.empty()) return decline("feature-disabled");
    for (const char* name : {"STRIX_KNOWN_PREFIX_WINDOW", "STRIX_KNOWN_PREFIX_FULL_ROWS",
            "STRIX_KNOWN_PREFIX_EXPERT_UNION", "STRIX_KNOWN_WINDOW_ROUTER_ROWS", "STRIX_KNOWN_PREFIX_PREFILL_ROWS"})
        if (env_is(request.environment, name, "0")) return decline("feature-disabled");
    if (o.tokens.size() < 16U || !o.generate_tokens) return decline("prompt-not-eligible");
    if (!qualified_checkpoint_content(manifest, verified_content)) return decline("checkpoint-content-not-qualified");
    if (!capabilities.target_qualified) return decline("target-not-qualified");
    if (!capabilities.a8_runtime) return decline("a8-runtime-not-built");
    if (!o.context_explicit && (o.tokens.size() > model.max_position_embeddings ||
        o.generate_tokens > model.max_position_embeddings - static_cast<std::uint64_t>(o.tokens.size())))
        return decline("prompt-exceeds-model-context");
    if (!memory.available || !memory.gtt_limit || !memory.gtt_used || !memory.estimate)
        return decline("memory-observation-unavailable");
    auto candidate = o;
    auto organization = resolve_optimization_profile(candidate, model, info, manifest, verified_content,
        capabilities.a8_runtime, false);
    if (!organization.automatic_applied) return decline(organization.reason);
    try {
        candidate.automatic_prefill = true;
        candidate.known_prefix_window = true;
        candidate.known_prefix_full_rows = true;
        candidate.known_prefix_expert_union = true;
        candidate.known_window_router_rows = true;
        candidate.known_prefix_output_policy = KnownWindowOutputPolicy::terminal_row;
        for (const auto rows : {8192U,4096U,2048U,1024U,256U,64U,16U})
            if (rows <= candidate.tokens.size()) { candidate.known_prefix_prefill_rows = rows; break; }
        const bool explicit_t16k = env_is(request.environment, "STRIX_QPF2K_T16K", "1");
        if (candidate.tokens.size() >= 16384U && (explicit_t16k ||
            (!request.environment.contains("STRIX_QPF2K_T16K") && candidate.tokens.size() >= t16k_prompt_threshold))) {
            candidate.known_prefix_prefill_rows = 16384U;
            result.t16k_policy_source = explicit_t16k ? OptionSource::explicit_value : OptionSource::automatic;
        }
        if (!o.context_explicit) {
            const auto needed = prefill_checked_add(static_cast<std::uint64_t>(o.tokens.size()), std::max(o.generate_tokens, UINT64_C(512)));
            candidate.context_tokens = std::min(model.max_position_embeddings,
                std::max(UINT64_C(8192), prefill_checked_add(needed, 511U) / 512U * 512U));
            candidate.unqualified_context = candidate.context_tokens != 8192U;
        }
        apply_measured_prompt_organization(candidate);
        if (!candidate.numerical_profile_explicit) candidate.numerical_profile = dense_a8_profile;
        if (!candidate.prefill_numerical_profile && !capabilities.fp16_operands &&
            candidate.numerical_profile == dense_a8_profile && !env_is(request.environment, "STRIX_PREFILL_F16", "0"))
            result.arithmetic_fallback = "fp16-operands-not-qualified";
        if (!candidate.prefill_numerical_profile && capabilities.fp16_operands &&
            candidate.numerical_profile == dense_a8_profile && !env_is(request.environment, "STRIX_PREFILL_F16", "0")) {
            candidate.prefill_numerical_profile = DecodeNumericalProfile::fast_prefill_f16wmma_v1;
            candidate.prefill_profile_automatic = true;
        }
        constexpr std::uint64_t mib = 1024U * 1024U, gib = 1024U * mib;
        const auto uma = std::min(120U * gib, *memory.available) / mib * mib;
        const auto scale = [&](std::uint64_t ceiling) { return (uma / mib * ceiling / 120U) * mib; };
        candidate.known_prefill_workspace_budget = scale(8U);
        candidate.known_prefill_metadata_budget = scale(1U);
        candidate.known_prefill_uma_budget = uma;
        candidate.known_prefill_host_headroom = std::max(gib, scale(4U));
        const auto primary_rows = candidate.known_prefix_prefill_rows;
        const auto primary = candidate;
        const auto arithmetic_fallback = result.arithmetic_fallback;
        std::uint64_t rolling_h = 0, rolling_bound = 0, rolling_cap = 0;
        const auto credit = prefill_reclaim_credit(candidate, model, manifest, request.environment, memory.ple_reclaim_page,
            &rolling_h, &rolling_bound, &rolling_cap);
        std::vector<std::uint32_t> trials{primary_rows};
        if (primary_rows == 16384U) trials.push_back(8192U);
        for (const auto rows : {4096U,2048U}) if (rows < primary_rows) trials.push_back(rows);
        std::string failed_reason = "insufficient-memory";
        for (const auto rows : trials) {
            candidate = primary;
            result.arithmetic_fallback = arithmetic_fallback;
            candidate.known_prefix_prefill_rows = rows;
            if (rows == 2048U && rows != primary_rows) {
                candidate.known_prefix_tail_rows = {256U,16U};
                candidate.known_prefix_ladder_tail_rows = candidate.known_prefix_tail_rows;
            }
            try {
                apply_default_known_prefill_tails(candidate, result.environment.contains("STRIX_QPF2K_SCHED") ?
                    result.environment.at("STRIX_QPF2K_SCHED").c_str() : "14");
                if (candidate.known_prefix_ladder_tail_rows.empty())
                    candidate.known_prefix_ladder_tail_rows = candidate.known_prefix_tail_rows;
                if (rows == 16384U) {
                    // Compile under the UMA ceiling, then replace these sizing
                    // bounds with the actual main+tail requirements below.
                    candidate.known_prefill_workspace_budget = uma;
                    candidate.known_prefill_metadata_budget = uma;
                    const auto pools = prefill_native_pools(candidate, model);
                    candidate.known_prefill_host_headroom = prefill_checked_add(pools.host, pools.device);
                }
                if (rows != 16384U) {
                    const auto pools = prefill_native_pools(candidate, model);
                    candidate.known_prefill_host_headroom = std::max(candidate.known_prefill_host_headroom,
                        prefill_checked_add(pools.host, pools.device));
                }
                validate_known_prefill_options(candidate);
                validate_numerical_profile_cli(candidate);
                validate_prefill_numerical_profile_cli(candidate);
                validate_effective_options(candidate, model);
                auto validation_organization = organization;
                apply_numerical_profile(candidate, model, manifest, verified_content, capabilities.a8_runtime, validation_organization);
                PrefillRequirements required;
                try { required = memory.estimate(candidate); }
                catch (const std::exception&) {
                    if (!candidate.prefill_profile_automatic) throw;
                    candidate.prefill_numerical_profile.reset();
                    candidate.prefill_profile_automatic = false;
                    result.arithmetic_fallback = "fp16-plan-not-admitted";
                    validate_prefill_numerical_profile_cli(candidate);
                    required = memory.estimate(candidate);
                }
                const auto add = prefill_checked_add;
                std::uint64_t native_gtt = 0;
                if (rows == 16384U) {
                    candidate.known_prefill_workspace_budget = add(required.workspace, required.tails);
                    candidate.known_prefill_metadata_budget = required.metadata;
                    const auto pools = prefill_native_pools(candidate, model);
                    native_gtt = add(pools.device, pools.mapped);
                    required.organization.insert(pools.bytes.begin(), pools.bytes.end());
                    required.organization["native_host_bytes"] = pools.host;
                    required.organization["native_device_bytes"] = pools.device;
                    required.organization["native_gtt_bytes"] = native_gtt;
                    required.organization["prefill_workspace_with_tails_bytes"] = candidate.known_prefill_workspace_budget;
                    required.organization["prefill_metadata_copies_bytes"] = required.metadata;
                }
                if (rows != 16384U) {
                    const auto pools = prefill_native_pools(candidate, model);
                    native_gtt = add(pools.device, pools.mapped);
                    candidate.known_prefill_host_headroom = std::max(candidate.known_prefill_host_headroom, add(pools.host, pools.device));
                    required.organization.insert(pools.bytes.begin(), pools.bytes.end());
                    required.organization["native_host_bytes"] = pools.host;
                    required.organization["native_device_bytes"] = pools.device;
                    required.organization["native_gtt_bytes"] = native_gtt;
                }
                auto live = UmaOwnerInventory{};
                live.reservation_upper_bound_bytes = add(required.model_backing - std::min(required.model_backing, credit),
                    add(required.state_d1, required.serial_scratch));
                const auto rolling_owner = prefill_rolling_owner_bytes(candidate,model,manifest,
                    request.environment,memory.ple_reclaim_page);
                const auto growth = add(rolling_owner, add(required.workspace,
                    add(required.metadata, add(required.tails, candidate.known_prefill_host_headroom))));
                if (!required.workspace || !required.metadata || add(required.workspace, required.tails) > candidate.known_prefill_workspace_budget ||
                    required.metadata > candidate.known_prefill_metadata_budget || *memory.gtt_used > *memory.gtt_limit ||
                    add(required.device_base, add(add(required.workspace, required.tails), native_gtt)) > *memory.gtt_limit - *memory.gtt_used)
                    throw std::runtime_error("insufficient-memory");
                if (add(add(live.reservation_upper_bound_bytes, growth), required.runtime_payload) > uma)
                    throw std::runtime_error("insufficient-memory");
                (void)plan_uma_reservation_growth(live, growth, required.runtime_payload, {uma, uma});
                result.ple_reclaim_credit = credit;
                result.ple_rolling_h = rolling_h;
                result.ple_rolling_bound = rolling_bound;
                result.ple_rolling_cap = rolling_cap;
                result.host_compilation = required.organization;
                for (const char* name : {"--known-prefix-window", "--known-prefix-full-rows", "--known-prefix-expert-union",
                        "--known-window-router-rows", "--known-prefix-prefill-rows", "--known-prefix-output-policy",
                        "--known-prefix-workspace-budget-bytes", "--known-prefix-host-metadata-budget-bytes",
                        "--known-prefix-uma-budget-bytes", "--known-prefix-host-headroom-bytes"})
                    result.sources[name] = OptionSource::automatic;
                for (const char* name : {"--parallel-hyper-mix", "--staged-hyper-projections",
                        "--fused-staged-hyper-mix", "--typed-fused-hyper-projections", "--prefetch-typed-hyper-rows",
                        "--prefetch-q8-hyper-output", "--lane-chain-hyper-output", "--prefetch-q8-qsa-input",
                        "--parallel-moe-router", "--wave-moe-router-topk", "--virtual-q8-gdn-input",
                        "--prefetch-q8-gdn-input", "--virtual-q8-gdn-output", "--deferred-moe-status",
                        "--prefetch-q8-shared-expert", "--q4k-a8g32-gate-up", "--grouped-exact-q51-down",
                        "--virtual-q8-lm-head", "--q51-slot-parallel-down", "--batched-moe-activation",
                        "--wave8-q8-gdn-input", "--wave8-q8-gdn-output", "--prefetch-q8-lm-head", "--wave8-q8-lm-head",
                        "--wave8-q8-qsa-output"})
                    result.sources[name] = OptionSource::automatic;
                if (!o.context_explicit) {
                    result.sources["--context"] = OptionSource::automatic;
                    if (candidate.unqualified_context) result.sources["--unqualified-context"] = OptionSource::automatic;
                }
                if (!o.numerical_profile_explicit) result.sources["--numerical-profile"] = OptionSource::automatic;
                if (candidate.prefill_profile_automatic) result.sources["--prefill-numerical-profile"] = OptionSource::automatic;
                if (rows == 2048U && rows != primary_rows) result.sources["--known-prefix-tail-rows"] = OptionSource::automatic;
                result.options = std::move(candidate);
                result.organization = std::move(organization);
                result.reason = rows == 16384U ? (explicit_t16k ? "opt-in-t16k" : "automatic-t16k") : "qualified-prompt";
                result.admitted = true;
                inject(~0U);
                return result;
            } catch (const std::overflow_error&) { throw; }
              catch (const KnownPrefillBudgetExceeded&) { failed_reason = "insufficient-memory"; }
              catch (const std::runtime_error&) { failed_reason = "insufficient-memory"; }
              catch (const std::exception&) { failed_reason = "candidate-not-admitted"; }
            if (!result.tile_fallback.empty()) result.tile_fallback += ',';
            result.tile_fallback += "T" + std::to_string(rows) + ':' + failed_reason;
        }
        return decline(failed_reason);
    } catch (const std::overflow_error&) { return decline("memory-overflow"); }
      catch (const std::runtime_error&) { return decline("insufficient-memory"); }
      catch (const std::exception&) { return decline("candidate-not-admitted"); }
    return result;
}
// Single admission registry: flip a serving entry only after its native exactness gate passes.
struct SessionRouteQualification { std::string_view route; bool gpu_qualified; };
inline constexpr SessionRouteQualification session_route_qualifications[] = {
    {"http-target", false}, {"http-mtp-fixed", false},
    {"http-mtp-batch", false}
};
inline std::string_view session_serving_route(const Options& o) {
    return !o.mtp.enabled ? "http-target" : o.http_batch > 1 ? "http-mtp-batch" : "http-mtp-fixed";
}
inline bool session_route_gpu_qualified(std::string_view route) {
    for (const auto& entry : session_route_qualifications)
        if (entry.route == route) return entry.gpu_qualified;
    return false;
}
// Session policy is independent of prompt auto-prefill admission. Never changes a route.
inline void resolve_legacy_session_defaults(Options& o, const ModelContract& model,
    const Environment& env, std::optional<std::uint64_t> available) {
    o.session_tiled_suffix_requested = o.session_tiled_suffix;
    o.session_mtp_fixed_requested = o.session_mtp_fixed;
    o.session_modes_suppression.clear();
    if (o.baseline || env_is(env,"STRIX_BASELINE","1")) o.session_modes_suppression="baseline";
    else if (const auto it=env.find("STRIX_QPF2K_DEFAULTS"); it!=env.end() && !it->second.empty() && it->second[0]=='0')
        o.session_modes_suppression="STRIX_QPF2K_DEFAULTS=0";
    if (!o.session_modes_suppression.empty()) {
        o.session_tiled_suffix=false;
        o.session_mtp_fixed=false;
    }
    constexpr std::uint64_t gib = UINT64_C(1) << 30;
    const auto supplied = [&](const char* flag) { return o.supplied_options.contains(flag); };
    if (!supplied("--session-cache") && env.contains("STRIX_SESSION_CACHE"))
        o.session_cache = !env_is(env, "STRIX_SESSION_CACHE", "0");
    if (!supplied("--session-disk") && env.contains("STRIX_SESSION_DISK"))
        o.session_disk = !env_is(env, "STRIX_SESSION_DISK", "0");
    if (!supplied("--session-ram-bytes")) o.session_ram_bytes = available ? std::min(8*gib, *available/10) : 0;
    if (!supplied("--session-max-parked")) o.session_max_parked = 8;
    if (o.session_dir.empty()) {
        if (env.contains("XDG_CACHE_HOME") && !env.at("XDG_CACHE_HOME").empty())
            o.session_dir = (std::filesystem::path(env.at("XDG_CACHE_HOME"))/"strix-engine/sessions").string();
        else if (env.contains("HOME") && !env.at("HOME").empty())
            o.session_dir = (std::filesystem::path(env.at("HOME"))/".cache/strix-engine/sessions").string();
    }
    o.session_management_requested = false;
    const auto off = [&](std::string reason) { o.session_cache = false; o.session_disk = false; o.session_off_reason = reason; };
    if (o.baseline || env_is(env,"STRIX_BASELINE","1")) return off("baseline");
    if (!o.session_modes_suppression.empty()) return off(o.session_modes_suppression);
    if (!o.session_cache) return off("disabled");
    if (!o.http_serve) return off("cli-save-load-explicit");
    // Explicit qualification instrumentation may exercise a draft without admitting production.
    const bool qualification = supplied("--session-mtp-fixed") && supplied("--session-tiled-suffix");
    if (!qualification && !session_route_gpu_qualified(session_serving_route(o)))
        return off("route not yet GPU-qualified: " + std::string(session_serving_route(o)));
    if (o.http_batch > 1) return off("http-batch-requires-ordinary-scheduler");
    // HTTP checkpoint ownership currently exists only on the fixed-MTP bridge.
    if (!o.mtp.enabled || !o.mtp.draft_chain ||
        !o.mtp.draft_vocab.empty() || o.mtp.draft_context_vocab || o.mtp.draft_q4 || o.mtp.sample_temperature > 0 ||
        !o.mtp.cohort.empty() || !o.mtp.copy_wide_rows.empty() || o.mtp.ngram_draft ||
        o.mtp.sample_repeats != 1 || o.mtp.metadata_only || !o.mtp.golden_dump.empty() ||
        !o.known_prefix_window || !o.known_prefix_full_rows || o.known_prefix_prefill_rows < 1024 ||
        (o.prefix_cache_block_rows && o.prefix_cache_block_rows < 1024) || o.prompt_lookup ||
        o.prefix_cache_roundtrip || o.gpu_greedy_token || o.teacher_forced_score || o.gdn_islands ||
        !o.raw_recording_path.empty() || !o.validation_final_state_path.empty() || !o.prefix_cache_then_prompts.empty() ||
        !o.generate_tokens || o.execution.mode != DecodeExecutionMode::eager ||
        o.execution.integer_head != DecodeIntegerHead::none || prefix_state_cache_requested(o) || o.rtx_state_import ||
        !o.session_save.empty() || !o.session_load.empty() || (o.session_live_control && !qualification)) return off("route-not-qualified");
    if (!available) return off("memory-observation-unavailable");
    std::uint64_t checkpoint = 0;
    try {
        SerialSessionStateLayoutRequest req; req.context_tokens=o.context_tokens; req.recurrent_storage=o.gdn_recurrent_storage;
        const auto layout=build_serial_session_state_layout(model,req);
        checkpoint=prefill_checked_add(layout.device_total_bytes,layout.host_total_bytes);
        // Predictor KV, held hidden row, frontier and envelope/lineage allowance.
        checkpoint=prefill_checked_add(checkpoint,prefill_checked_add(16U<<20,
            o.context_tokens*(model.mtp_full_attention_layers*model.kv_heads*model.head_dim*4U+16U)));
        if (checkpoint > UINT64_MAX/5 || checkpoint*5 > o.session_ram_bytes || checkpoint*5 > *available ||
            !o.session_max_parked) return off("insufficient-memory");
    } catch (const std::exception&) { return off("memory-estimate-failed"); }
    o.session_management_requested = true;
    o.session_mtp_fixed = true;
    if (!o.session_disk) { o.session_disk_off_reason="disabled"; return; }
    // The probe is operational only; it does not read or publish a checkpoint.
    try {
        if (o.session_dir.empty()) throw std::runtime_error("no cache home");
        std::filesystem::create_directories(o.session_dir);
        const auto free=std::filesystem::space(o.session_dir).available;
        if (!supplied("--session-disk-bytes")) o.session_disk_bytes=std::min(64*gib,free/4);
        if (free < checkpoint*2 || o.session_disk_bytes < checkpoint) throw std::runtime_error("space");
        const auto probe=std::filesystem::path(o.session_dir)/(".write-probe-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        { std::ofstream file(probe,std::ios::binary); if (!file) throw std::runtime_error("writable"); }
        std::filesystem::remove(probe);
    } catch (const std::exception&) { o.session_disk=false; o.session_disk_off_reason="directory-unavailable-or-insufficient-space"; }
}
inline constexpr SessionRouteQualification prefix_auto_route_qualifications[] = {
    {"http-target", false}, {"http-mtp-fixed", false}, {"http-mtp-batch", false}
};
inline const char* prefix_batch_cert_mode_name(const Options& o) {
    return o.prefix_batch_cert==1?"l0":o.prefix_batch_cert==2?"capture":"l1";
}
inline bool prefix_batch_cert_active(const Options& o) {
    return o.prefix_batch_cert && !o.baseline && ordinary_http_batch_slots(o) &&
        o.prefix_cache && o.prefix_cache_host_bytes;
}
// In-flight joining follows the prefix cache's own enablement: it is on by
// default wherever the cache is on, and armed only where concurrent producers
// and joiners share a strict-L0 certified batch route. A single-slot route has
// no concurrent producer, so the policy is inert there. It never widens the
// cache's qualified route set.
struct PrefixInflightState {
    bool requested=false, effective=false;
    std::string source, reason;
};
inline PrefixInflightState prefix_cache_inflight_state(const Options& o) {
    PrefixInflightState s{o.prefix_cache_inflight,false,o.prefix_cache_inflight_source,"armed"};
    if(o.baseline) s.reason="baseline";
    else if(!o.prefix_cache) s.reason="prefix-cache-off";
    else if(!o.prefix_cache_inflight) s.reason="off-switch";
    else if(!ordinary_http_batch_slots(o)) s.reason="single-slot-route";
    else if(!o.prefix_cache_host_bytes) s.reason="prefix-cache-inactive";
    else if(prefix_batch_cert_execution_mode(o)!=1) s.reason="batch-cert-not-strict-l0";
    else s.effective=true;
    return s;
}
inline void prefix_inflight_json(serve::Json& out,const Options& o) {
    const auto state=prefix_cache_inflight_state(o);
    out.set("inflight",serve::Json(state.effective));
    out.set("inflight_requested",serve::Json(state.requested));
    out.set("inflight_source",serve::Json(state.source));
    out.set("inflight_reason",serve::Json(state.reason));
    out.set("inflight_policy",serve::Json("strict-l0; matching-cold-full-t; ordinary-admission-only"));
}
inline void prefix_cache_capabilities(serve::Json& out,const Options& o) {
    if(o.baseline || !o.prefix_cache || !o.prefix_cache_host_bytes) return;
    serve::Json cache=serve::Json::object();
    const bool disk=o.prefix_cache_disk && !o.prefix_cache_dir.empty();
    cache.set("certified_disk",serve::Json(disk));
    cache.set("durable",serve::Json(disk && o.prefix_cache_durable));
    prefix_inflight_json(cache,o);
    out.set("prefix_cache",std::move(cache));
}
inline void prefix_batch_cert_capabilities(serve::Json& out, const Options& o) {
    if (!prefix_batch_cert_active(o)) return;
    serve::Json cert=serve::Json::object();
    cert.set("mode",serve::Json(prefix_batch_cert_mode_name(o)));
    cert.set("experimental",serve::Json(true));
    cert.set("l1_qualified",serve::Json(false));
    cert.set("automatic_qualified",serve::Json(false));
    cert.set("inflight",serve::Json(prefix_cache_inflight_state(o).effective));
    out.set("prefix_batch_certification",std::move(cert));
}
// Separate from cache-budget admission: Round 1 permits strict L0 restores only.
inline constexpr SessionRouteQualification prefix_batch_cert_route_qualifications[] = {
    {"http-mtp-batch-certified-l1", false}
};
inline void resolve_legacy_prefix_defaults(Options& o, const ModelContract& model,
    const Environment& env, std::optional<std::uint64_t> available,
    std::span<const SessionRouteQualification> qualifications = prefix_auto_route_qualifications) {
    const auto supplied = [&](const char* flag) { return o.supplied_options.contains(flag); };
    apply_prefix_cache_environment(o, [&](const char* name) -> std::optional<std::string_view> {
        const auto it = env.find(name);
        if (it != env.end()) return it->second;
        return std::nullopt;
    });
    const auto off = [&](const char* reason) {
        o.prefix_cache_source = "off"; o.prefix_cache_reason = reason;
    };
    if (!o.prefix_cache) return off("disabled");
    if (o.baseline) return off("baseline");
    if (supplied("--prefix-cache-host-bytes") || o.prefix_cache_host_bytes) {
        o.prefix_cache_source="explicit"; o.prefix_cache_reason="explicit-budget"; return;
    }
    if (!o.http_serve) return off("not-serving");
    if (o.session_management_requested) return off("sessions-admitted");
    // Reuse exactly the existing acceptance predicate; do not add a numerical route.
    auto probe = o; probe.prefix_cache_host_bytes=1;
    try { validate_prefix_state_cache(probe); } catch (const std::exception&) { return off("route-not-accepted"); }
    const auto route = session_serving_route(o);
    bool qualified = false;
    for (const auto& entry : qualifications) if (entry.route == route) qualified = entry.gpu_qualified;
    if (!qualified) {
        o.prefix_cache_source="off";
        o.prefix_cache_reason="automatic budget not yet GPU-qualified: " + std::string(route);
        return;
    }
    // TODO(R3): before qualifying a production route, re-observe after weight residency
    // and lane plans; fit prefix retained bytes AND capture peaks in the shared UMA
    // reservation, reserving host headroom. This pre-residency estimate is test-only.
    if (!available) return off("memory-observation-unavailable");
    auto budget = std::min(UINT64_C(4)<<30, *available/16);
    budget -= budget % 4096;
    try {
        SerialSessionStateLayoutRequest req; req.context_tokens=o.context_tokens; req.recurrent_storage=o.gdn_recurrent_storage;
        const auto layout=build_serial_session_state_layout(model,req);
        auto checkpoint=prefill_checked_add(layout.device_total_bytes,layout.host_total_bytes);
        if (o.mtp.enabled) checkpoint=prefill_checked_add(checkpoint,
            prefill_checked_add(16U<<20,o.context_tokens*(model.mtp_full_attention_layers*model.kv_heads*model.head_dim*4U+model.indexer_head_dim*2U+16U)));
        if (!budget || checkpoint>budget) return off("checkpoint-does-not-fit");
    } catch (const std::exception&) { return off("memory-estimate-failed"); }
    o.prefix_cache_host_bytes=budget; o.prefix_cache_source="auto"; o.prefix_cache_reason="fitted-host-budget";
}
// HTTP prompts arrive after startup. Size a shared tiled owner against context,
// preserving the already resolved arithmetic and every explicit organization.
inline void resolve_serve_prefill(EffectiveConfig& result, const ModelContract& model,
    std::string_view content, const PrefillCapabilities& capabilities, const PrefillMemory& memory) {
    auto& o = result.options;
    if (!o.http_serve) return;
    const auto notice = [&](std::string reason) {
        result.serve_prefill_notice = "Qwen serve-prefill: " + std::move(reason) + "\n";
    };
    if (o.known_prefix_prefill_explicit || o.known_prefix_prefill_rows) return notice("explicit T" + std::to_string(o.known_prefix_prefill_rows));
    if (o.baseline || env_is(result.environment,"STRIX_SERVE_PREFILL","0") ||
        !o.disabled_prefill_features.empty()) return notice("off; explicit tiled configuration required");
    for (const char* name : {"STRIX_KNOWN_PREFIX_WINDOW", "STRIX_KNOWN_PREFIX_FULL_ROWS",
            "STRIX_KNOWN_PREFIX_EXPERT_UNION", "STRIX_KNOWN_WINDOW_ROUTER_ROWS", "STRIX_KNOWN_PREFIX_PREFILL_ROWS"})
        if (env_is(result.environment,name,"0")) return notice("off; explicit tiled configuration required");
    if (content != validated_model_content_identity || !capabilities.target_qualified || !capabilities.a8_runtime ||
        (numerical_profile_pair(o).fast_components & kFastPrefillDense8) ||
        o.rotary.mode() != RotaryMode::native || o.prompt_lookup || o.known_prefix_rows_explicit ||
        o.teacher_forced_score || o.execution.mode != DecodeExecutionMode::eager ||
        (o.known_prefix_output_policy_explicit && o.known_prefix_output_policy != KnownWindowOutputPolicy::terminal_row))
        return notice("route not admitted; explicit tiled configuration required");
    const auto copy_all = result.environment.find("STRIX_MODEL_COPY_ALL");
    if (o.residency != HipModelResidencyMode::registered_file_mapping ||
        o.model_copy_tier != HipModelCopyTier::none ||
        (copy_all != result.environment.end() && !copy_all->second.empty() && copy_all->second.front() != '0'))
        return notice("automatic T declined for target residency copy, selective copy or copy-all; explicit tiled configuration required");
    if (!memory.available || !memory.gtt_limit || !memory.gtt_used || !memory.estimate)
        return notice("memory observation unavailable; explicit tiled configuration required");
    constexpr std::uint64_t gib = UINT64_C(1) << 30U;
    std::string attempts;
    for (const auto rows : {8192U,4096U,2048U,1024U,256U,64U,16U}) {
        if (rows > o.context_tokens || rows > model.max_position_embeddings) {
            attempts += "T" + std::to_string(rows) + ":context; "; continue;
        }
        auto candidate = o;
        candidate.automatic_prefill = true;
        candidate.known_prefix_window = true;
        candidate.known_prefix_full_rows = true;
        candidate.known_prefix_expert_union = true;
        candidate.known_window_router_rows = true;
        candidate.known_prefix_prefill_rows = rows;
        candidate.known_prefix_output_policy = KnownWindowOutputPolicy::terminal_row;
        // Required row-invariant organizations of the tiled known-prefix route.
        candidate.q51_slot_parallel_down = true;
        candidate.wave8_q8_gdn_input = true;
        candidate.wave8_q8_gdn_output = true;
        if (!candidate.known_prefill_metadata_explicit) candidate.known_prefill_metadata_budget = gib;
        if (!candidate.known_prefill_uma_explicit) candidate.known_prefill_uma_budget = std::min(120*gib,*memory.available);
        // Serving retains uniform logits storage and all tail owners. The CLI's
        // compact/pruned 8 GiB ceiling must not force a spurious T8192 fallback.
        if (!candidate.known_prefill_workspace_explicit)
            candidate.known_prefill_workspace_budget = candidate.known_prefill_uma_budget;
        if (!candidate.known_prefill_headroom_explicit) candidate.known_prefill_host_headroom = 4*gib;
        try {
            apply_default_known_prefill_tails(candidate);
            validate_known_prefill_options(candidate);
            validate_effective_options(candidate,model);
            // A larger group creates an independent arena/native owner at startup,
            // even when SMALL would not schedule it. Automatic sizing does not
            // support that organization; retry only candidates without this owner.
            if (const auto group=result.environment.find("STRIX_PROMPT_BATCH_ROWS");
                group!=result.environment.end() && !group->second.empty()) {
                const auto group_rows=parse_u64(group->second,"STRIX_PROMPT_BATCH_ROWS");
                if (group_rows > rows)
                    throw std::runtime_error("STRIX_PROMPT_BATCH_ROWS creates an unsupported extra group owner; "
                        "unset it or use an explicit tiled configuration with group workspace/metadata budgets");
            }
            const auto required = memory.estimate(candidate);
            const auto add = prefill_checked_add;
            const auto scratch = add(required.workspace,required.tails);
            if (!candidate.known_prefill_workspace_explicit) candidate.known_prefill_workspace_budget = scratch;
            // The same owner calculation admits the future bridge at runtime.
            // Missing asset evidence cannot establish even one runnable slot.
            if (candidate.mtp.enabled && !memory.serving)
                throw std::runtime_error("one-slot:verified MTP owner estimate unavailable");
            const auto serving = memory.serving ? memory.serving(candidate) : HttpServingMemory{};
            SerialSessionStateLayoutRequest state_request;
            state_request.context_tokens=candidate.context_tokens;
            state_request.recurrent_storage=candidate.gdn_recurrent_storage;
            const auto staging=qwen_http_batch_staging_reserve(candidate,model,
                build_serial_session_state_layout(model,state_request));
            const auto future_host = add(staging,add(serving.shared_host,serving.slot_host));
            const auto future_device = add(staging,add(serving.shared_device,serving.slot_device));
            const auto growth = add(scratch,add(required.metadata,add(required.runtime_payload,
                add(future_host,add(candidate.prefix_cache_host_bytes,candidate.known_prefill_host_headroom)))));
            const auto total = add(required.model_backing,add(required.state_d1,add(required.serial_scratch,add(required.external_host,growth))));
            const auto fit = [](const char* resource, std::uint64_t need, std::uint64_t limit) {
                if (need > limit) throw std::runtime_error(std::string("memory:") + resource +
                    " needs " + std::to_string(need) + " bytes, limit " + std::to_string(limit));
            };
            if (!required.workspace || !required.metadata)
                throw std::runtime_error("plan:empty workspace or metadata estimate");
            fit("workspace",scratch,candidate.known_prefill_workspace_budget);
            fit("metadata",required.metadata,candidate.known_prefill_metadata_budget);
            fit("host-available",total,*memory.available);
            fit("UMA",total,candidate.known_prefill_uma_budget);
            fit("GTT-used",*memory.gtt_used,*memory.gtt_limit);
            fit("GTT-free",add(required.device_base,add(required.external_device,add(scratch,add(future_device,add(candidate.prefix_cache_host_bytes,candidate.known_prefill_host_headroom))))),*memory.gtt_limit - *memory.gtt_used);
            o = std::move(candidate);
            result.host_compilation = required.organization;
            result.host_compilation["serve_committed_host_peak_bytes"]=total;
            result.host_compilation["serve_prefill_external_host_bytes"]=required.external_host;
            result.host_compilation["serve_prefill_external_device_bytes"]=required.external_device;
            result.host_compilation["serve_batch_staging_bytes"]=staging;
            result.host_compilation["serve_shared_host_peak_bytes"]=serving.shared_host;
            result.host_compilation["serve_shared_device_peak_bytes"]=serving.shared_device;
            result.host_compilation["serve_one_slot_host_peak_bytes"]=serving.slot_host;
            result.host_compilation["serve_one_slot_device_peak_bytes"]=serving.slot_device;
            for (const char* flag : {"--known-prefix-prefill-rows", "--known-prefix-window", "--known-prefix-full-rows",
                    "--known-prefix-expert-union", "--known-window-router-rows", "--known-prefix-output-policy",
                    "--known-prefix-workspace-budget-bytes", "--known-prefix-host-metadata-budget-bytes",
                    "--known-prefix-uma-budget-bytes", "--known-prefix-host-headroom-bytes",
                    "--q51-slot-parallel-down", "--wave8-q8-gdn-input", "--wave8-q8-gdn-output"})
                if (!o.supplied_options.contains(flag)) result.sources[flag] = OptionSource::automatic;
            return notice("automatic T" + std::to_string(rows) + "; " + attempts + "arithmetic preserved");
        } catch (const std::exception& error) {
            attempts += "T" + std::to_string(rows) + ":" + error.what() + "; ";
        }
    }
    notice(attempts + "explicit tiled configuration required");
}
// Stage A is pure: no host budget estimate, device work or directory probes.
inline std::string cache_defaults_suppression(const Options& o, const Environment& env) {
    if (o.baseline || env_is(env,"STRIX_BASELINE","1")) return "baseline";
    if (const auto i=env.find("STRIX_QPF2K_DEFAULTS"); i!=env.end() && !i->second.empty() && i->second[0]=='0')
        return "defaults-suppressed";
    return {};
}
inline bool explicit_cache_configuration(const Options& o, const Environment& env) {
    // Joining and disk-publication switches are policies within the cache, not
    // a cache configuration: they never move a run off the automatic path.
    for(const auto& flag:o.supplied_options)
        if((flag.starts_with("--prefix-cache") && flag!="--prefix-cache-inflight" && flag!="--prefix-cache-disk" &&
            flag!="--prefix-cache-durable") || flag=="--prefix-batch-cert" || flag.starts_with("--session-")) return true;
    for(const auto* name:{"STRIX_PREFIX_CACHE","STRIX_PREFIX_CACHE_RETENTION","STRIX_PREFIX_CACHE_STAGING_BYTES",
            "STRIX_PREFIX_BATCH_CERT","STRIX_SESSION_CACHE","STRIX_SESSION_DISK"})
        if(env.contains(name)) return true;
    return o.prefix_cache_host_bytes || !o.prefix_cache_dir.empty() || o.prefix_cache_disk_bytes ||
        o.prefix_cache_every || o.prefix_cache_block_rows || o.prefix_batch_cert || o.session_mtp_fixed ||
        o.session_tiled_suffix || !o.session_save.empty() || !o.session_load.empty() || !o.session_dir.empty();
}
inline void resolve_session_defaults(Options& o, const ModelContract& model,
    const Environment& env, std::optional<std::uint64_t> available) {
    o.cache_activation=std::make_shared<CachePolicyActivation>(*o.cache_activation);
    auto& a=*o.cache_activation;
    a.automatic_path=!explicit_cache_configuration(o,env) && o.prefix_cache && cache_defaults_suppression(o,env).empty();
    a.prefix_requested=o.prefix_cache;
    a.session_requested=!o.supplied_options.contains("--session-cache") && env.contains("STRIX_SESSION_CACHE")
        ? !env_is(env,"STRIX_SESSION_CACHE","0") : o.session_cache;
    a.suppression=cache_defaults_suppression(o,env);
    resolve_legacy_session_defaults(o,model,env,available);
    a.session=o.session_management_requested; a.session_reason=o.session_off_reason;
}
inline void resolve_prefix_defaults(Options& o, const ModelContract& model,
    const Environment& env, std::optional<std::uint64_t> available,
    std::span<const SessionRouteQualification> qualifications=prefix_auto_route_qualifications) {
    o.cache_activation=std::make_shared<CachePolicyActivation>(*o.cache_activation);
    // Preserve all public fields. The new automatic decision is reported separately.
    resolve_legacy_prefix_defaults(o,model,env,available,qualifications);
    auto& a=*o.cache_activation;
    a.prefix_requested=o.prefix_cache;
    a.prefix=o.prefix_cache_host_bytes!=0;
    a.source=a.prefix?"explicit":o.prefix_cache_source; a.reason=o.prefix_cache_reason;
    a.cert_requested=a.automatic_path?1U:a.cert_policy_parsed?a.cert_requested:o.prefix_batch_cert;
    a.cert_effective=a.automatic_path?0U:o.prefix_batch_cert;
    a.cert_reason=!a.cert_requested?"disabled":!a.cert_effective && a.suppression=="baseline"?"baseline":"explicit";
    if(a.automatic_path) {
        a.source="auto";
        a.reason=o.http_serve?"pending-post-residency":o.prefix_cache_then_prompts.empty()?"no-reuse-route":"serial-capture-cost-unqualified";
        a.cert_reason="pending-post-residency";
    }
}
inline serve::Json cache_activation_json(const Options& o) {
    const auto& a=*o.cache_activation;
    auto out=serve::Json::object();
    out.set("requested",serve::Json(a.prefix_requested));
    out.set("effective",a.finalized?serve::Json(a.prefix):serve::Json());
    out.set("session_requested",serve::Json(a.session_requested));
    out.set("session_effective",a.finalized?serve::Json(a.session):serve::Json());
    out.set("session_reason",serve::Json(a.session_reason));
    out.set("source",serve::Json(a.source)); out.set("reason",serve::Json(a.reason));
    out.set("suppression",serve::Json(a.suppression));
    out.set("runner",serve::Json(std::string(a.runner==CacheRunner::pending?"pending-runner":a.runner==CacheRunner::cli?"cli-target-tiled":a.runner==CacheRunner::batch_mtp?"http-mtp-batch":a.runner==CacheRunner::single_mtp?"http-mtp-fixed":"http-target")));
    out.set("family",serve::Json(a.family)); out.set("evidence_id",serve::Json(a.evidence_id));
    out.set("automatic_qualified",serve::Json(a.automatic_qualified));
    out.set("phase",serve::Json(a.automatic_path && !a.decision_phase.empty()?a.decision_phase:(a.finalized?"post-residency":"inspection")));
    for (const auto& [name,value]: {std::pair{"B",a.B},{"S",a.S},{"H",a.H},{"F",a.F},{"E",a.E},
            {"context_off",a.context_off},{"slots_off",a.slots_off},
            {"context_admitted",a.context_admitted},{"slots_admitted",a.slots_admitted}})
        out.set(name,value?serve::Json(*value):serve::Json());
    if(a.automatic_path && a.pool_setup_ms>0) out.set("pool_setup_ms",serve::Json(a.pool_setup_ms));
    if(a.observation_age_ns) {
        for(const auto& [name,value]: {std::pair{"U",a.U},{"L",a.L},{"A",a.A},{"reserve",a.reserve}})
            out.set(name,value?serve::Json(*value):serve::Json());
        out.set("limits_consistent",serve::Json(a.limits_consistent));
        out.set("observation_age_ns",serve::Json(*a.observation_age_ns));
    }
    auto protocol=serve::Json::object();
    const bool executed=o.session_mtp_fixed && (!o.http_serve || o.session_management_requested || o.session_live_control);
    protocol.set("fixed_requested",serve::Json(o.session_mtp_fixed_requested));
    protocol.set("fixed_effective",serve::Json(executed));
    protocol.set("tiled_suffix_requested",serve::Json(o.session_tiled_suffix_requested));
    protocol.set("tiled_suffix_effective",serve::Json(o.session_tiled_suffix && (!o.http_serve || executed)));
    protocol.set("reason",serve::Json(executed?"explicit-protocol":a.session_reason));
    out.set("executed_session_protocol",std::move(protocol));
    auto cert=serve::Json::object();
    cert.set("policy",serve::Json((a.cert_requested==0?"off":a.cert_requested==1?"l0-where-eligible":a.cert_requested==2?"capture":"l1")));
    cert.set("effective",!a.finalized?serve::Json():serve::Json(!a.cert_requested?"off":a.cert_effective?
        (a.cert_effective==1?"l0":a.cert_effective==2?"capture":"l1"):"declined"));
    cert.set("reason",serve::Json(a.cert_reason));
    cert.set("l1_qualified",serve::Json(false));
    cert.set("automatic_qualified",serve::Json(a.automatic_qualified && a.cert_effective==1));
    out.set("batch_certification",std::move(cert));
    return out;
}
inline serve::Json cache_family_identity_json(const CacheFamilyIdentity& id) {
    auto out=serve::Json::object();
    out.set("mtp_contract_qualified",serve::Json(id.mtp_contract_qualified));
    out.set("model",serve::Json(id.model));
    out.set("layout",serve::Json(id.layout));
    out.set("tokenizer",serve::Json(id.tokenizer));
    out.set("sidecar",serve::Json(id.sidecar));
    out.set("golden",serve::Json(id.golden));
    out.set("replay",serve::Json(id.replay));
    out.set("build_family",serve::Json(id.build_family));
    out.set("gfx",serve::Json(id.gfx));
    out.set("numerical_contract",serve::Json(id.numerical_contract));
    out.set("kv_precision",serve::Json(id.kv_precision));
    out.set("recurrent_precision",serve::Json(id.recurrent_precision));
    out.set("prefill_profile",serve::Json(id.prefill_profile));
    out.set("decode_profile",serve::Json(id.decode_profile));
    out.set("output_policy",serve::Json(id.output_policy));
    out.set("profile_pair",serve::Json(id.profile_pair));
    auto tails=serve::Json::array();
    for(auto value:id.tails) tails.as_array().emplace_back(static_cast<double>(value));
    out.set("tails",std::move(tails));
    auto scheduler=serve::Json::object();
    scheduler.set("schedule",serve::Json(static_cast<double>(id.scheduler.schedule)));
    scheduler.set("pipe",serve::Json(static_cast<double>(id.scheduler.pipe)));
    scheduler.set("lookahead",serve::Json(static_cast<double>(id.scheduler.lookahead)));
    scheduler.set("short_rows",serve::Json(static_cast<double>(id.scheduler.short_rows)));
    scheduler.set("service_ms",serve::Json(static_cast<double>(id.scheduler.service_ms)));
    scheduler.set("prompt_share",serve::Json(static_cast<double>(id.scheduler.prompt_share)));
    scheduler.set("nest_rows",serve::Json(static_cast<double>(id.scheduler.nest_rows)));
    scheduler.set("interleave",serve::Json(static_cast<double>(id.scheduler.interleave)));
    scheduler.set("group_rows",serve::Json(static_cast<double>(id.scheduler.group_rows)));
    scheduler.set("groups",serve::Json(id.scheduler.groups));
    scheduler.set("cohort",serve::Json(id.scheduler.cohort));
    scheduler.set("small",serve::Json(id.scheduler.small));
    scheduler.set("group_owner",serve::Json(id.scheduler.group_owner));
    scheduler.set("windows",serve::Json(id.scheduler.windows));
    scheduler.set("cancel",serve::Json(id.scheduler.cancel));
    scheduler.set("priming",serve::Json(id.scheduler.priming));
    scheduler.set("worker",serve::Json(id.scheduler.worker));
    scheduler.set("tiled_start",serve::Json(id.scheduler.tiled_start));
    scheduler.set("merge",serve::Json(id.scheduler.merge));
    scheduler.set("balance",serve::Json(id.scheduler.balance));
    scheduler.set("shared",serve::Json(id.scheduler.shared));
    out.set("scheduler",std::move(scheduler));
    return out;
}
// Called only on the automatic path after the actual runner/identity is bound.
inline bool apply_automatic_cache_fit(Options& o, const CacheLateFit& fit) {
    auto& a=*o.cache_activation;
    if(!automatic_cache_work(o)) return false;
    const auto result=fit.cgroup_ancestry_unproven ? CacheFitResult{0,"cgroup-ancestry-unproven"} : fit_cache_late(fit);
    a.S=fit.S; a.H=std::max(UINT64_C(4)<<30,fit.headroom); a.F=fit.F;
    if(fit.E && *fit.E<=UINT64_MAX-(UINT64_C(256)<<20)) a.E=*fit.E+(UINT64_C(256)<<20);
    a.U=fit.U; a.L=fit.L; a.A=fit.A; a.observation_age_ns=fit.age_ns;
    a.limits_consistent=fit.limits_consistent; a.reserve=cache_fit_reserve(fit);
    a.B=result.budget; a.prefix=result.budget!=0;
    a.reason=o.prefix_cache_reason=std::string(result.reason);
    o.prefix_cache_host_bytes=result.budget;
    if(!result.budget) decline_cache_activation(a,result.reason);
    if(a.runner==CacheRunner::batch_mtp && a.automatic_qualified) {
        a.cert_effective=result.budget?1U:0U; a.cert_reason=std::string(result.reason);
    }
    return result.budget!=0;
}
// Feasibility never assigns B. Only the ready sample may fit automatic single storage.
inline bool prepare_automatic_cache_fit(Options& o, const CacheLateFit& fit, bool ready) {
    auto& a=*o.cache_activation;
    if(a.automatic_path && a.source=="auto" && a.runner==CacheRunner::single_mtp) {
        if(a.decision_phase=="post-residency-ready") return a.prefix;
        if(ready && a.reason!="pending-post-residency") return false;
        a.decision_phase=ready?"post-residency-ready":"pre-bridge-feasibility";
        if(!ready) {
            // Record this observation without fitting or assigning a budget.
            a.U=fit.U; a.L=fit.L; a.A=fit.A; a.F=fit.F; a.S=fit.S;
            a.H=std::max(UINT64_C(4)<<30,fit.headroom);
            if(fit.E && *fit.E<=UINT64_MAX-(UINT64_C(256)<<20)) a.E=*fit.E+(UINT64_C(256)<<20);
            a.reserve=cache_fit_reserve(fit); a.observation_age_ns=fit.age_ns;
            a.limits_consistent=fit.limits_consistent;
            const auto reason=cache_feasibility_decline(fit);
            if(!reason.empty()) {
                decline_cache_activation(a,reason); o.prefix_cache_reason=a.reason;
            } else { a.B.reset(); a.prefix=false; }
            return false;
        }
    }
    return apply_automatic_cache_fit(o,fit);
}
// Bind the actual runner only after OFF sizing; batch N=1 remains a batch family.
inline void finalize_cache_family(Options& o, CacheRunner runner, std::uint32_t slots,
    const CacheFamilyIdentity& resolved_identity={}) {
    // Compare the schedule this runner executes (STRIX_QPF2K_SCHED bit 2 is inert off the CLI cursor).
    const auto identity=effective_cache_family(resolved_identity,runner);
    auto& a=*o.cache_activation;
    a.runner=runner; a.finalized=true;
    a.context_admitted=o.context_tokens; a.slots_admitted=slots;
    // Legacy sizing may reserve explicit cache backing. It computed no OFF plan.
    a.context_off.reset(); a.slots_off.reset();
    if(!a.automatic_path) {
        a.prefix=o.prefix_cache_host_bytes!=0; a.session=o.session_management_requested;
        a.cert_effective=o.prefix_batch_cert; a.B=o.prefix_cache_host_bytes;
        return;
    }
    a.decision_phase="pre-bridge-feasibility";
    a.context_off=o.context_tokens; a.slots_off=slots;
    a.B=0; a.S=0; a.H=std::max(UINT64_C(4)<<30,o.known_prefill_host_headroom);
    a.prefix=false; a.cert_effective=0; a.session=false;
    if(o.rotary.mode()==RotaryMode::yarn) {
        a.automatic_qualified=false; a.source="off";
        decline_cache_activation(a,"rotary-cache-unqualified");
        return;
    }
    if(runner==CacheRunner::single_mtp && (!cache_mtp_contract_qualified(o) || !identity.mtp_contract_qualified)) {
        a.automatic_qualified=false; a.source="off";
        decline_cache_activation(a,"mtp-contract-unqualified");
        return;
    }
    const CacheQualification* matched=nullptr;
    for (const auto& row:automatic_prefix_auto_route_qualifications)
        if (row.matches(runner,slots,o.context_tokens,o.known_prefix_prefill_rows,identity)) { matched=&row; break; }
    if (matched) { a.family=matched->family; a.evidence_id=matched->evidence_id; }
    else for(const auto& row:automatic_prefix_auto_route_qualifications)
        if(row.runner==runner && slots>=row.n_min && slots<=row.n_max && o.known_prefix_prefill_rows==row.tile) {
            a.family=row.family; a.evidence_id=row.evidence_id; break; // candidate, never a permission
        }
    a.automatic_qualified=matched && matched->qualified();
    if (runner!=CacheRunner::cli && a.session_requested && o.session_modes_suppression.empty()) {
        a.session_reason=runner==CacheRunner::batch_mtp?"session-batch-owner-unqualified":
            runner==CacheRunner::target?"session-owner-unavailable":
            !o.session_mtp_fixed || !o.session_tiled_suffix?"session-numerical-contract-unqualified":"family-unqualified";
    }
    if (!a.prefix_requested || a.reason=="baseline" || a.reason=="defaults-suppressed") return;
    if (a.source=="explicit" && !a.suppression.empty()) return;
    if (runner==CacheRunner::batch_mtp) {
        const CacheQualification* cert=nullptr;
        for(const auto& row:automatic_prefix_batch_cert_route_qualifications)
            if(row.use==CacheUse::batch_l0 && row.matches(runner,slots,o.context_tokens,o.known_prefix_prefill_rows,identity)) {
                cert=&row; break;
            }
        const bool admitted=a.automatic_qualified && cert && cert->qualified() && a.cert_requested==1;
        a.cert_reason=!a.cert_requested?"disabled":admitted?"pending-post-residency":"batch-scheduler-family-unqualified";
        if(!admitted) {
            a.reason=!a.cert_requested?"batch-cert-disabled":"batch-scheduler-family-unqualified";
            a.source="off"; o.prefix_cache_host_bytes=0;
            return;
        }
        a.reason="pending-post-residency";
        return; // effective L0 is assigned only after the final shared fit
    }
    a.cert_reason=!a.cert_requested?"disabled":!a.suppression.empty()?a.suppression:
        runner==CacheRunner::cli?a.reason:"not-applicable-non-batch";
    if (a.source=="explicit") return; // late physical fit still required
    if (a.reason=="pending-post-residency" && !a.automatic_qualified) {
        a.reason=matched && matched->gpu_qualified && matched->resource_qualified && !matched->cost_qualified?
            "cost-unqualified":runner==CacheRunner::target?"route-unqualified":"family-unqualified";
        a.source="off";
    }
}
inline EffectiveConfig resolve_effective_config(const RequestedOptions& request,
    const ModelContract& model, const gguf::FileInfo& info, const ModelManifest& manifest,
    std::string_view content, const PrefillCapabilities& capabilities, const PrefillMemory& memory) {
    // Stage proposals locally. Route admission needs resolved prefill/serving
    // options and model writers; no candidate reaches process switch readers yet.
    auto proposed = request;
    admit_product_profile(proposed.options);
    qwen::profile::apply(proposed.options.product_profile, [&](const char* key, const char* value) {
        if (!qwen::profile::metadata_environment(key)) proposed.environment[key] = value;
        return true;
    });
    auto result=resolve_prefill_effective_config(proposed,model,info,manifest,content,capabilities,memory);
    resolve_serve_prefill(result,model,content,capabilities,memory);
    resolve_session_defaults(result.options,model,request.environment,memory.available);
    resolve_prefix_defaults(result.options,model,request.environment,memory.available);
    auto& o = result.options;
    // A profile proposal may fall back to builtin storage. Explicit requests
    // retain the hard-refusal contract, including native CLI precedence.
    const bool profile_route = moe_route_bf16_profile_owned(o);
    std::string route_error;
    resolve_moe_route_bf16(o, result.environment, result.gsq_owner, capabilities.fp16_operands, route_error);
    if (o.prefill_moe_route_bf16) {
        const bool geometry = model.hidden_size == 2560U && model.moe_intermediate_size % 128U == 0U &&
            model.moe_intermediate_size >= 320U && model.moe_intermediate_size <= 640U &&
            model.num_experts > 0U && model.num_experts <= 512U &&
            model.num_experts_per_token > 0U && model.num_experts_per_token <= 10U;
        bool formats = true;
        for (const auto& t : manifest.tensors) {
            if (t.name.ends_with(".ffn_gate_exps.weight") || t.name.ends_with(".ffn_up_exps.weight"))
                formats = formats && (t.ggml_type == 12U || t.ggml_type == 13U);
            if (t.name.ends_with(".ffn_down_exps.weight")) formats = formats && (t.ggml_type == 7U || t.ggml_type == 8U);
        }
        if (!geometry || !formats) {
            o.prefill_moe_route_bf16 = false;
            o.prefill_moe_route_bf16_reason = "unsupported DA16 geometry or writer format";
            route_error = "MoE route BF16 candidate declined: " + o.prefill_moe_route_bf16_reason;
        }
    }
    if (profile_route && !o.prefill_moe_route_bf16) {
        for (auto& entry : o.product_profile.entries) if (entry.lever->name == "moe_route_bf16") {
            entry.status = qwen::profile::Status::pending;
            entry.target = std::string(entry.lever->target);
            entry.value = std::string(entry.lever->builtin);
            entry.reason = o.prefill_moe_route_bf16_reason;
            result.environment.erase(std::string(entry.lever->env));
        }
    } else if (!route_error.empty()) result.error = std::move(route_error);
    if (!result.error.empty()) result.admitted = false;
    if (o.http_serve && !o.baseline && o.prefix_cache && o.prefix_cache_host_bytes &&
        !o.prefix_cache_block_rows_explicit && o.known_prefix_prefill_rows >= 1024 &&
        o.known_prefix_prefill_rows % 1024 == 0) {
        o.prefix_cache_block_rows = 1024;
        result.sources["--prefix-cache-block-rows"] = OptionSource::automatic;
    }
    return result;
}
// Only the initial cgroup namespace and a hierarchy-root mount expose all ancestors.
// Linux include/linux/proc_ns.h: PROC_CGROUP_INIT_INO = 0xEFFFFFFB (4026531835).
// PID 1 namespace equality is insufficient: PID 1 can itself be in a container.
inline void intersect_cache_cgroup_limits(PrefillMemory& result, const std::string& groups,
    const std::filesystem::path& root, std::string_view mountinfo={},
    std::string_view self_namespace={}) {
    result.cache_limits_known=false;
    if(self_namespace!="cgroup:[4026531835]") return;
    std::istringstream group_lines(groups); std::string line, membership;
    while(std::getline(group_lines,line)) {
        const auto first=line.find(':');
        const auto second=first==std::string::npos?first:line.find(':',first+1);
        if(second==std::string::npos) return;
        const auto controllers=","+line.substr(first+1,second-first-1)+",";
        if(controllers.find(",memory,")!=std::string::npos) return; // v1 owns memory, including hybrid
        if(line.starts_with("0::/")) {
            if(!membership.empty()) return;
            membership=line.substr(3);
        }
    }
    if(membership.empty()) return;
    std::string mount_root;
    std::istringstream mounts{std::string(mountinfo)};
    while(std::getline(mounts,line)) {
        const auto separator=line.find(" - cgroup2 ");
        if(separator==std::string::npos) continue;
        std::istringstream fields(line.substr(0,separator));
        std::string id,parent,device,mapped,point;
        if(!(fields>>id>>parent>>device>>mapped>>point)) return;
        if(point!="/sys/fs/cgroup") continue;
        if(!mount_root.empty()) return;
        mount_root=mapped;
    }
    if(mount_root!="/") return; // subtree mounts conceal possibly limiting parents
    const auto safe=[](const std::filesystem::path& path) {
        for(const auto& part:path) if(part=="." || part=="..") return false;
        return true;
    };
    if(!safe(membership)) return;
    const auto relative=membership.substr(1);
    auto path=root/relative;
    if(relative.empty()) path=root;
    for(;;) {
        if(path==root) {
            std::error_code error1,error2;
            const bool has_limit=std::filesystem::exists(path/"memory.max",error1);
            const bool has_current=std::filesystem::exists(path/"memory.current",error2);
            if(has_limit || has_current || error1 || error2) return;
            break; // the global root has no memory controller files
        }
        const auto [limit,status]=read_linux_uma_text(path/"memory.max");
        const auto [current,current_status]=read_linux_uma_text(path/"memory.current");
        if(status!=LinuxUmaReadStatus::read || current_status!=LinuxUmaReadStatus::read) return;
        try {
            const auto token=[](const std::string& value) { return value.substr(0,value.find_first_of("\r\n \t")); };
            const auto used=parse_u64(token(current),"cgroup memory.current");
            if(token(limit)!="max") {
                const auto maximum=parse_u64(token(limit),"cgroup memory.max");
                if(used>maximum || !result.usable_ceiling) return;
                result.usable_ceiling=std::min(*result.usable_ceiling,maximum);
                if(result.cache_available) result.cache_available=std::min(*result.cache_available,maximum-used);
            }
        } catch(const std::exception&) { return; }
        if(path==root) break;
        path=path.parent_path();
    }
    result.cache_limits_known=true;
}
// Read sysfs only when one unambiguously selected KFD GPU is visible. Ambiguous
// visibility is unknown capacity and declines; HIP is never initialized here.
inline PrefillMemory observe_prefill_memory(PrefillCapabilities& capabilities, bool cache_limits=false) {
    PrefillMemory result;
    (void)cache_limits;
    const auto observed = observe_linux_uma_memory();
    if (observed.memory.observation_identity) {
        result.available = observed.memory.mem_available_bytes;
        result.total = observed.memory.mem_total_bytes;
        result.observation_id=observed.memory.observation_identity;
        result.end_steady_ns=observed.begin_steady_ns;
        result.usable_ceiling=result.total; result.cache_available=result.available;
        if(result.total && result.available && *result.available<=*result.total) result.resident=*result.total-*result.available;
    }
#ifdef __linux__
    if(cache_limits) {
    // Cache admission must intersect cgroup limits. Unknown v1/delegated layouts
    // decline cache admission; they do not change incumbent context/slot sizing.
    const auto [groups,group_status]=read_linux_uma_text("/proc/self/cgroup");
    const auto [mounts,mount_status]=read_linux_uma_text("/proc/self/mountinfo");
    std::error_code self_error;
    const auto self_ns=std::filesystem::read_symlink("/proc/self/ns/cgroup",self_error);
    if(group_status==LinuxUmaReadStatus::read && mount_status==LinuxUmaReadStatus::read)
        intersect_cache_cgroup_limits(result,groups,"/sys/fs/cgroup",mounts,
            self_error?"":self_ns.string());
    }
    const auto page = ::sysconf(_SC_PAGESIZE);
    if (page > 0) result.ple_reclaim_page = static_cast<std::uint64_t>(page);
    for (const char* name : {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"})
        if (const char* value = std::getenv(name); value && *value) return result;
    std::error_code error;
    const std::filesystem::path nodes("/sys/class/kfd/kfd/topology/nodes");
    std::optional<std::uint64_t> minor;
    std::uint64_t gfx = 0;
    unsigned node_count = 0;
    for (std::filesystem::directory_iterator node(nodes, error), end; !error && node != end; node.increment(error)) {
        if (++node_count > 256U) return result;
        std::ifstream properties_file(node->path() / "properties", std::ios::binary);
        std::array<char, 65537U> payload{};
        properties_file.read(payload.data(), static_cast<std::streamsize>(payload.size()));
        const auto bytes = properties_file.gcount();
        if (bytes < 0 || bytes > 65536) return result;
        std::istringstream properties(std::string(payload.data(), static_cast<std::size_t>(bytes)));
        std::string key;
        std::uint64_t value = 0, render = 0, target = 0;
        while (properties >> key >> value) {
            if (key == "drm_render_minor") render = value;
            if (key == "gfx_target_version") target = value;
        }
        if (!render) continue;
        if (minor) return result;
        minor = render; gfx = target;
    }
    if (!minor || error) return result;
    const auto device = std::filesystem::path("/sys/class/drm") / ("renderD" + std::to_string(*minor)) / "device";
    const auto read = [&](const char* name) -> std::optional<std::uint64_t> {
        std::ifstream file(device / name);
        std::string text;
        if (!std::getline(file, text) || text.size() > 32U) return {};
        try { return parse_u64(text, "GTT observation"); } catch (const std::exception&) { return {}; }
    };
    result.gtt_limit = read("mem_info_gtt_total");
    result.gtt_used = read("mem_info_gtt_used");
    result.gfx_target=gfx;
    capabilities.target_qualified = gfx == 110501U;
    capabilities.fp16_operands = capabilities.target_qualified;
#endif
    return result;
}
inline std::string effective_config_notice(const EffectiveConfig& config) {
    if (!config.options.suppressed_yarn_request.empty())
        return "Qwen baseline: suppressed YaRN request (" + config.options.suppressed_yarn_request + "); rotary=native\n";
    if (config.options.rotary.mode() == RotaryMode::yarn && config.admitted)
        return "Qwen experimental YaRN: factor=" + std::to_string(config.options.rotary.factor()) +
            " context=" + std::to_string(config.options.context_tokens) + " T=" +
            std::to_string(config.options.known_prefix_prefill_rows) + "; qualification=experimental; verified PLE credit=" +
            std::to_string(config.ple_reclaim_credit) + " bytes\n";
    const auto prefix = !config.options.baseline && (config.options.http_serve || config.options.prefix_cache_reason == "disabled") && config.options.prefix_cache_source == "off"
        ? "prefix cache: off (" + config.options.prefix_cache_reason + ")\n" : std::string{};
    const auto session = (config.options.http_serve || config.options.supplied_options.contains("--http-serve") ||
        config.options.supplied_options.contains("--session-cache")) && !config.options.session_cache
        ? "session cache: off (" + config.options.session_off_reason + ")\n" : std::string{};
    if (!session.empty() || !prefix.empty()) return session + prefix;
    if (config.admitted) {
        std::string line = "Qwen auto-prefill: automatic T" + std::to_string(config.options.known_prefix_prefill_rows) +
            " context=" + std::to_string(config.options.context_tokens);
        line += "; reason=" + config.reason;
        if (config.t16k_policy_source)
            line += "; T16384 policy=" + std::string(*config.t16k_policy_source == OptionSource::automatic ? "automatic" : "explicit") +
                " threshold=" + std::to_string(t16k_prompt_threshold);
        if (!config.tile_fallback.empty()) line += "; " + config.tile_fallback;
        if (config.options.context_tokens > 8192U)
            line += "; unqualified-context (beyond qualified 8192; " +
                std::string(config.sources.contains("--unqualified-context") &&
                    config.sources.at("--unqualified-context") == OptionSource::automatic ? "automatic" : "explicit") + ")";
        if (!config.arithmetic_fallback.empty()) line += "; " + config.arithmetic_fallback + "; using dense-A8 tiled prefill";
        return line + "\n";
    }
    if (!config.error.empty()) return config.error + '\n';
    // The single-session MTP route keeps its explicit prefill and takes the automatic window dispatch fusion.
    if (config.reason == "explicit-route" && config.sources.contains("STRIX_DISPATCH_FUSE") &&
        config.sources.at("STRIX_DISPATCH_FUSE") == OptionSource::automatic)
        return "Qwen MTP route: STRIX_DISPATCH_FUSE=1 (automatic; STRIX_DISPATCH_FUSE=0 keeps the separate window launches)\n";
    if (config.reason == "baseline" || config.reason == "explicit-route") return {};
    return "Qwen auto-prefill declined: " + config.reason + "; using serial prefill\n";
}
inline std::vector<std::string> effective_prefill_argv(const Options& o) {
    std::vector<std::string> args{"--context", std::to_string(o.context_tokens)};
    if (o.rotary.mode() == RotaryMode::yarn) args.insert(args.end(), {"--rope-yarn",std::to_string(o.rotary.factor())});
    if (o.prefill_exact_fp16) args.emplace_back("--prefill-exact-fp16");
    for (unsigned i = 0; i < kPrefillOptions.size(); ++i)
        if (o.prefill_disabled & (1U << i)) args.insert(args.end(), {"--prefill-disable", std::string(kPrefillOptions[i].name)});
    if (o.unqualified_context) args.emplace_back("--unqualified-context");
    args.insert(args.end(), {"--execution-mode", o.execution.mode == DecodeExecutionMode::eager ? "eager" : "graph",
        "--residency", o.residency == HipModelResidencyMode::registered_file_mapping ? "register" : "copy",
        "--generate", std::to_string(o.generate_tokens)});
    if (o.parallel_hyper_mix) args.emplace_back("--parallel-hyper-mix");
    if (o.staged_hyper_projections) args.emplace_back("--staged-hyper-projections");
    if (o.fused_staged_hyper_mix) args.emplace_back("--fused-staged-hyper-mix");
    if (o.typed_hyper_projection) args.emplace_back("--typed-fused-hyper-projections");
    if (o.prefetch_typed_hyper_rows) args.emplace_back("--prefetch-typed-hyper-rows");
    if (o.prefetch_q8_hyper_output) args.emplace_back("--prefetch-q8-hyper-output");
    if (o.lane_chain_hyper_output) args.emplace_back("--lane-chain-hyper-output");
    if (o.prefetch_q8_qsa_input) args.emplace_back("--prefetch-q8-qsa-input");
    if (o.parallel_moe_router) args.emplace_back("--parallel-moe-router");
    if (o.wave_moe_router_topk) args.emplace_back("--wave-moe-router-topk");
    if (o.virtual_q8_gdn_input) args.emplace_back("--virtual-q8-gdn-input");
    if (o.prefetch_q8_gdn_input) args.emplace_back("--prefetch-q8-gdn-input");
    if (o.virtual_q8_gdn_output) args.emplace_back("--virtual-q8-gdn-output");
    if (o.deferred_moe_status) args.emplace_back("--deferred-moe-status");
    if (o.prefetch_q8_shared_expert) args.emplace_back("--prefetch-q8-shared-expert");
    if (o.q4k_a8g32_gate_up) args.emplace_back("--q4k-a8g32-gate-up");
    if (o.grouped_exact_q51_down) args.emplace_back("--grouped-exact-q51-down");
    if (o.virtual_q8_lm_head) args.emplace_back("--virtual-q8-lm-head");
    if (o.q51_slot_parallel_down) args.emplace_back("--q51-slot-parallel-down");
    if (o.batched_moe_activation) args.emplace_back("--batched-moe-activation");
    if (o.wave8_q8_gdn_input) args.emplace_back("--wave8-q8-gdn-input");
    if (o.wave8_q8_gdn_output) args.emplace_back("--wave8-q8-gdn-output");
    if (o.prefetch_q8_lm_head) args.emplace_back("--prefetch-q8-lm-head");
    if (o.wave8_q8_lm_head) args.emplace_back("--wave8-q8-lm-head");
    if (o.wave8_q8_qsa_output) args.emplace_back("--wave8-q8-qsa-output");
    if (o.prepared_eager) args.emplace_back("--prepared-eager");
    if (o.moe_shared_overlap) args.emplace_back("--moe-shared-overlap");
    if (o.gdn_islands) args.emplace_back("--gdn-islands");
    if (o.baseline) args.emplace_back("--baseline");
    args.insert(args.end(), {"--optimization-profile", o.optimization_profile == OptimizationProfile::automatic ? "auto" : "manual",
        "--numerical-profile", std::string(numerical_profile_name(o.numerical_profile))});
    if (o.known_prefix_window) args.emplace_back("--known-prefix-window");
    if (o.known_prefix_full_rows) args.emplace_back("--known-prefix-full-rows");
    if (o.known_prefix_rows_explicit) args.insert(args.end(), {"--known-prefix-rows", std::to_string(o.known_prefix_rows)});
    if (o.known_prefix_prefill_rows) args.insert(args.end(), {
        "--known-prefix-workspace-budget-bytes", std::to_string(o.known_prefill_workspace_budget),
        "--known-prefix-host-metadata-budget-bytes", std::to_string(o.known_prefill_metadata_budget),
        "--known-prefix-uma-budget-bytes", std::to_string(o.known_prefill_uma_budget),
        "--known-prefix-host-headroom-bytes", std::to_string(o.known_prefill_host_headroom),
        "--known-prefix-output-policy", o.known_prefix_output_policy == KnownWindowOutputPolicy::terminal_row ? "terminal-row" :
            o.known_prefix_output_policy == KnownWindowOutputPolicy::all_rows ? "all-rows" : "state-only"});
    if (o.known_prefix_window && !o.known_prefix_prefill_rows) args.insert(args.end(), {"--known-prefix-output-policy",
        o.known_prefix_output_policy == KnownWindowOutputPolicy::terminal_row ? "terminal-row" :
        o.known_prefix_output_policy == KnownWindowOutputPolicy::state_only ? "state-only" : "all-rows"});
    if (o.supplied_options.contains("--known-prefix-tail-rows") ||
        (o.automatic_prefill && o.known_prefix_ladder_tail_rows == std::vector<std::uint32_t>{256U,16U})) {
        std::string tails = o.known_prefix_tail_rows_off ? "none" : "";
        for (const auto rows : o.known_prefix_tail_rows) { if (!tails.empty()) tails += ','; tails += std::to_string(rows); }
        args.insert(args.end(), {"--known-prefix-tail-rows", std::move(tails)});
    }
    if (o.known_prefix_expert_union) args.emplace_back("--known-prefix-expert-union");
    if (o.known_window_router_rows) args.emplace_back("--known-window-router-rows");
    if (o.known_prefix_prefill_rows) args.insert(args.end(), {"--known-prefix-prefill-rows", std::to_string(o.known_prefix_prefill_rows)});
    if (o.prefill_numerical_profile) args.insert(args.end(), {"--prefill-numerical-profile", std::string(numerical_profile_name(*o.prefill_numerical_profile))});
    return args;
}
inline std::string yarn_json_error(std::string_view value) {
    std::ostringstream out;
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') out << '\\' << static_cast<char>(ch);
        else if (ch < 32U) out << "\\u00" << hex[ch >> 4U] << hex[ch & 15U];
        else out << static_cast<char>(ch);
    }
    return out.str();
}
inline serve::Json session_modes_json(const Options& o) {
    auto modes=serve::Json::object();
    for (const auto& [name,requested,effective] : {
        std::tuple{"tiled_suffix",o.session_tiled_suffix_requested,o.session_tiled_suffix},
        std::tuple{"mtp_fixed",o.session_mtp_fixed_requested,o.session_mtp_fixed}}) {
        auto mode=serve::Json::object();
        mode.set("requested",serve::Json(requested)); mode.set("effective",serve::Json(effective));
        mode.set("suppression",serve::Json(o.session_modes_suppression));
        mode.set("bit_changing",serve::Json(true));
        mode.set("qualification",serve::Json("draft-native-equality-unqualified"));
        modes.set(name,std::move(mode));
    }
    return modes;
}
inline std::string yarn_mtp_storage_receipt(const EffectiveConfig& config) {
    if (config.options.rotary.mode() != RotaryMode::yarn || !config.options.mtp.enabled ||
        !config.admission.contains("storage_ladder_step")) return {};
    std::string out = "\"yarn_mtp_storage\":{\n";
    bool first = true;
    for (const char* name : {"storage_ladder_step", "ple_rolling_positions", "ple_rolling_cap_bytes",
            "qsa_score_ring_rows", "prefill_pipe"})
    {
        out += (first ? "" : ",") + std::string("\"") + name + "\":" + std::to_string(config.admission.at(name));
        first = false;
    }
    return out + "},\n\"storage_ladder_step\":" + std::to_string(config.admission.at("storage_ladder_step")) + ",\n";
}
inline std::string effective_config_json(const EffectiveConfig& config) {
    const auto& o = config.options;
    std::ostringstream out;
    // Inspection status is separate from native identity serialization (zero bytes).
    if (o.rotary.mode() == strix::RotaryMode::yarn)
        out << "{\"rotary\":" << (config.rotary_identity.descriptor().mode() == RotaryMode::yarn ?
            config.rotary_identity.json() : o.rotary.json()) << ',';
    else out << "{\"rotary\":" << o.rotary.json() << ',';
    out << "\"profile\":" << o.product_profile.json() << ',';
    if (!o.suppressed_yarn_request.empty())
        out << "\"suppressed_yarn_request\":\"" << yarn_json_error(o.suppressed_yarn_request) << "\",";
    // YaRN preflight can verify a drop before device setup. Native inspection
    // retains its historical predicted-only serialization.
    out << yarn_mtp_storage_receipt(config);
    if (config.ple_rolling_h) {
        out << "\"ple_mode\":\"rolling\",\"ple_rolling_positions\":" << config.ple_rolling_h
            << ",\"ple_rolling_bound_bytes\":" << config.ple_rolling_bound
            << ",\"ple_rolling_cap_bytes\":" << config.ple_rolling_cap
            << ",\"ple_verified_credit_bytes\":" << (o.rotary.mode()==RotaryMode::yarn?config.ple_reclaim_credit:0U)
            << ",\"ple_credit_verification\":\"" << (o.rotary.mode()==RotaryMode::yarn && config.ple_reclaim_credit?"verified-pre-device-drop":"runtime-setup-required") << "\",";
    }
    if (o.rotary.mode() == RotaryMode::yarn) {
        out << "\"yarn_decode_route\":{\"requested\":\"" << (o.mtp.enabled?"mtp":"serial")
            << "\",\"effective\":\"" << (config.admitted?(o.mtp.enabled?"mtp":"serial"):"refused")
            << "\",\"source\":" << serve::Json(o.yarn_decode_source).dump()
            << ",\"gpu_qualified\":false,\"contract_version\":1,\"capacity\":" << o.context_tokens
            << ",\"verifier_requires_serial_rows\":true,\"predictor_position\":\"hidden-source\"},";
        if(o.mtp.enabled) out << "\"yarn_mtp_assets\":{\"target_content\":\"" << o.mtp.target_content
            << "\",\"sidecar_sha256\":\"" << o.mtp.sidecar_sha256 << "\",\"golden_sha256\":\"" << o.mtp.golden_sha256
            << "\",\"golden\":" << serve::Json(!o.mtp_bundle_path.empty() && o.mtp.golden.empty()?"pending-self-generation":o.mtp.golden).dump()
            << ",\"draft_chain\":" << (o.mtp.draft_chain?"true":"false") << ",\"draft_q4\":" << (o.mtp.draft_q4?"true":"false") << "},";
        out << "\"admission\":{";
        bool first = true;
        for (const auto& [name, bytes] : config.admission) {
            out << (first ? "" : ",") << '"' << name << "\":" << bytes;
            first = false;
        }
        out << "},";
    }
    out << "\"measurement\":{\"logits_capture\":"
        << (config.environment.contains("STRIX_BITCALC_LOGITS_OUT") ? "true" : "false")
        << ",\"forced_tokens\":"
        << (config.environment.contains("STRIX_BITCALC_FORCE_TOKENS") ? "true" : "false") << "},";
    if(qsa_mtp_pool_route(o,config.gsq_owner)) {
        const auto plan=mtp_qsa_pool_plan(o.context_tokens);
        out << "\"qsa_mtp_pool\":{\"requested\":true,\"effective\":\"runtime-admission-required\",\"format\":\"bf16\",\"entries\":13,\"key_bytes\":"
            << plan.key_bytes << ",\"scratch_bytes\":" << plan.scratch_bytes << ",\"metadata_bytes\":" << plan.metadata_bytes << ",\"reserved_bytes\":" << plan.reserved_bytes << "},";
    }
    serve::Json session=serve::Json::object();
    session.set("numerical_modes",session_modes_json(o));
    session.set("policy",serve::Json("qualified-serving-routes-only"));
    session.set("cli_operations",serve::Json("explicit-save-load"));
    session.set("enabled",serve::Json(o.session_cache));
    session.set("disk",serve::Json(o.session_disk));
    session.set("ram_bytes",serve::Json(o.session_ram_bytes));
    session.set("max_parked",serve::Json(o.session_max_parked));
    session.set("disk_bytes",serve::Json(o.session_disk_bytes));
    session.set("dir",serve::Json(o.session_dir));
    session.set("reason",serve::Json(o.session_off_reason));
    session.set("disk_reason",serve::Json(o.session_disk_off_reason));
    serve::Json prefix=serve::Json::object();
    prefix.set("enabled",serve::Json(o.prefix_cache_host_bytes != 0));
    prefix.set("source",serve::Json(o.prefix_cache_source));
    prefix.set("budget",serve::Json(o.prefix_cache_host_bytes));
    prefix.set("retention",serve::Json(o.prefix_cache_retention));
    prefix.set("reason",serve::Json(o.prefix_cache_reason));
    const bool disk=o.prefix_cache && o.prefix_cache_host_bytes && o.prefix_cache_disk && !o.prefix_cache_dir.empty();
    prefix.set("disk",serve::Json(disk));
    prefix.set("certified_disk",serve::Json(disk && !o.baseline));
    prefix.set("durable",serve::Json(disk && o.prefix_cache_durable && !o.baseline));
    prefix_inflight_json(prefix,o);
    prefix.set("serial_certificate",serve::Json("full-t-v1; whole-prompt-cursor; synchronous-snapshot"));
    prefix.set("serial_declines",serve::Json("serial-async-capture-unsupported,serial-lookup-disabled-or-short,serial-certificate-missing-or-incompatible,serial-plan-unsupported,serial-no-certified-checkpoint,serial-producer-plan-mismatch,serial-cursor-continuation-unsupported,serial-checkpoint-preflight,serial-capture-unit-unsupported,serial-capture-boundary-unsupported,serial-capture-metadata-mismatch"));
    if(prefix_batch_cert_active(o)) {
        prefix.set("batch_cert_mode",serve::Json(prefix_batch_cert_mode_name(o)));
        prefix.set("batch_l1_qualified",serve::Json(false));
    }
    if (!o.baseline) out << "\"prefix_cache\":" << prefix.dump() << ",";
    out << "\"cache_policy\":" << cache_activation_json(o).dump() << ",";
    out << prefill_identity_fields(config.prefill_option_mask) << ",";
    out << "\"session_cache\":" << session.dump() << ",\"http_slots\":{";
    out << "\"mode\":\"" << (o.http_batch_explicit ? (o.http_batch_cap ? "cap" : "single") : "auto")
        << "\",\"cap\":" << o.http_batch_cap << ",\"slots\":";
    if (o.http_serve && o.http_batch) out << "null"; else out << 1;
    out << ",\"route\":\"" << (o.http_batch ? "batch" : "single")
        << "\",\"reason\":\"" << o.http_slot_reason << "\"},";
    // The rolling block above already carries the cap while rolling is active.
    if (config.ple_rolling_cap && !config.ple_rolling_h) out << "\"ple_rolling_cap_bytes\":" << config.ple_rolling_cap << ',';
    if (o.http_serve) out << "\"serve_prefill_notice\":" << serve::Json(config.serve_prefill_notice).dump() << ',';
    out << "\"admitted\":" << (config.admitted ? "true" : "false") << ",\"reason\":\"" << config.reason
        << "\",\"tile_fallback\":\"" << config.tile_fallback
        << "\",\"error\":\"" << (o.rotary.mode() == RotaryMode::yarn ? yarn_json_error(config.error) : config.error)
        << "\",\"ple_reclaim_credit_bytes\":" << config.ple_reclaim_credit
        << ",\"arithmetic_fallback\":\"" << config.arithmetic_fallback
        << "\",\"longer_context_qualified\":" << (o.context_tokens > 8192U ? "false" : "null")
        << ",\"context\":" << o.context_tokens << ",\"T\":" << o.known_prefix_prefill_rows
        << ",\"serve_automatic_prefill\":" << (o.http_serve && o.automatic_prefill ? "true" : "false")
        << ",\"prefix_cache_block_rows\":" << o.prefix_cache_block_rows
        << ",\"batch_prompt_small\":" << ((config.environment.contains("STRIX_BATCH_PROMPT_SMALL") &&
            config.environment.at("STRIX_BATCH_PROMPT_SMALL").starts_with('1')) ? "true" : "false")
        << ",\"decode_profile\":\"" << numerical_profile_name(o.numerical_profile)
        << "\",\"prefill_profile\":\"" << numerical_profile_name(numerical_profile_pair(o).prefill) << "\",\"numerical_profile_pair_identity\":" << numerical_profile_pair_identity(numerical_profile_pair(o))
        << ",\"prefill_moe_route_bf16\":" << (o.prefill_moe_route_bf16 ? "true" : "false")
        << ",\"prefill_storage_identity\":" << ((numerical_profile_pair(o).fast_components & kFastPrefillMoe) ?
            serve::Json(moe_prefill_storage_name(o.prefill_moe_route_bf16)).dump() : "null")
        << ",\"prefill_storage_identity_hash\":" << ((numerical_profile_pair(o).fast_components & kFastPrefillMoe) ?
            std::to_string(moe_prefill_storage_identity(o.prefill_moe_route_bf16)) : "null")
        << ",\"prefill_moe_route_bf16_reason\":" << serve::Json(o.prefill_moe_route_bf16_reason).dump()
        << ",\"prefill_fast_components\":" << numerical_profile_pair(o).fast_components
        << ",\"tails\":[";
    for (std::size_t i = 0; i < o.known_prefix_tail_rows.size(); ++i) out << (i ? "," : "") << o.known_prefix_tail_rows[i];
    out << "],\"budgets\":[" << o.known_prefill_workspace_budget << ',' << o.known_prefill_metadata_budget << ','
        << o.known_prefill_uma_budget << ',' << o.known_prefill_host_headroom << "],\"environment\":{";
    bool first = true;
    for (const auto& [name,value] : config.environment) {
        if (name == "STRIX_BITCALC_LOGITS_OUT" || name == "STRIX_BITCALC_FORCE_TOKENS") continue;
        if (!config.ple_rolling_cap && name == "STRIX_PLE_ROLLING_CAP_BYTES") continue;
        if (!config.ple_rolling_h && (name == "STRIX_PLE_ROLLING" ||
            name == "STRIX_PLE_ROLLING_POSITIONS" || name == "STRIX_PLE_ROLLING_CAP_BYTES")) continue;
        // Values are externally supplied strings, so quote every control byte.
        const auto quote = [](std::string_view text) {
            std::ostringstream escaped;
            constexpr char hex[] = "0123456789abcdef";
            for (const char raw : text) {
                const auto ch = static_cast<unsigned char>(raw);
                if (ch == '"' || ch == '\\') escaped << '\\' << static_cast<char>(ch);
                else if (ch < 32U) escaped << "\\u00" << hex[ch >> 4U] << hex[ch & 15U];
                else escaped << static_cast<char>(ch);
            }
            return escaped.str();
        };
        out << (first ? "" : ",") << '"' << quote(name) << "\":{\"value\":\"" << quote(value) << "\",\"source\":\"";
        const auto it = config.sources.find(name);
        out << (it == config.sources.end() || it->second == OptionSource::explicit_value ? "explicit" :
            it->second == OptionSource::baseline ? "baseline" : "automatic") << "\"}";
        first = false;
    }
    out << "},\"gsq_filtered_defaults\":[";
    bool first_filtered = true;
    if (config.gsq_owner) for (const auto& entry : qpf2k::defaults) {
        const auto source = config.sources.find(entry.name);
        if (source == config.sources.end() || source->second != OptionSource::automatic) continue;
        out << (first_filtered ? "" : ",") << '"' << entry.name << '"'; first_filtered = false;
    }
    out << "],\"argv\":[";
    bool first_arg = true;
    for (const auto& arg : effective_prefill_argv(o)) { out << (first_arg ? "" : ",") << '"' << arg << '"'; first_arg = false; }
    out << "],\"organization\":\"" << config.organization.selected << "\",\"configured_tails\":[";
    for (std::size_t i = 0; i < o.known_prefix_ladder_tail_rows.size(); ++i) out << (i ? "," : "") << o.known_prefix_ladder_tail_rows[i];
    out << "],\"suppressed\":[";
    for (std::size_t i = 0; i < config.suppressed.size(); ++i) out << (i ? "," : "") << '"' << config.suppressed[i] << '"';
    out << "],\"option_sources\":{";
    bool first_source = true;
    for (const auto& [name,source] : config.sources) {
        if (!name.starts_with("--") || qwen::profile::policy_option(name)) continue;
        out << (first_source ? "" : ",") << '"' << name << "\":\"" <<
            (source == OptionSource::automatic ? "automatic" : source == OptionSource::baseline ? "baseline" : "explicit") << '"';
        first_source = false;
    }
    out << "},\"host_compilation\":{";
    bool first_field = true;
    for (const auto& [name,value] : config.host_compilation) {
        out << (first_field ? "" : ",") << '"' << name << "\":" << value; first_field = false;
    }
    // Resolved code-default long-context items (the requested values stay in "environment"): enabled, and when off the
    // suppression (baseline, gsq, STRIX_QPF2K_DEFAULTS=0, explicit-off, or the excluded decode route). The union merge
    // and the 64-row scorer apply from 49,152 and 65,536 blocks.
    const auto item = [&](const char* name, const char* why, bool comma) {
        out << '"' << name << "\":{\"enabled\":" << (why ? "false" : "true") << ",\"suppression\":";
        if (why) out << '"' << why << '"'; else out << "null";
        out << '}' << (comma ? "," : "");
    };
    const auto pool_off = qsa_decode_pool_route_suppression(o, config.gsq_owner);
    const auto select_off = pool_off ? pool_off : qpf2k::long_context_item_suppression("STRIX_QSA_DECODE_SELECT", config.gsq_owner);
    const auto bf16_off = pool_off ? pool_off : qpf2k::long_context_item_suppression("STRIX_QSA_DECODE_POOL_BF16", config.gsq_owner);
    out << "},\"long_context_items\":{";
    item("STRIX_QSA_UNION_MERGE", qpf2k::long_context_item_suppression("STRIX_QSA_UNION_MERGE", config.gsq_owner), true);
    item("STRIX_QSA_SCORE_ROWS64", qpf2k::long_context_item_suppression("STRIX_QSA_SCORE_ROWS64", config.gsq_owner), true);
    item("STRIX_QSA_DECODE_POOL", pool_off, true);
    item("STRIX_QSA_DECODE_SELECT", select_off, true);
    item("STRIX_QSA_DECODE_POOL_BF16", bf16_off, true);
    item("STRIX_QSA_CACHE_ONLY", qpf2k::long_context_item_suppression("STRIX_QSA_CACHE_ONLY", config.gsq_owner), true);
    item("STRIX_PREFILL_MOE_MERGE_VEC", qpf2k::long_context_item_suppression("STRIX_PREFILL_MOE_MERGE_VEC", config.gsq_owner), false);
    out << "},\"qsa_decode_pool_key_bytes\":" << (pool_off ? 0U : qsa_decode_pool_key_bytes(o, config.gsq_owner));
    if (config.t16k_policy_source)
        out << ",\"t16k_policy\":{\"source\":\"" << (*config.t16k_policy_source == OptionSource::automatic ? "automatic" : "explicit")
            << "\",\"threshold_tokens\":" << t16k_prompt_threshold << '}';
    const auto index_batch = qpf2k::index_batch_enabled([&](const char* name) -> const char* {
        const auto it = config.environment.find(name);
        return it == config.environment.end() ? nullptr : it->second.c_str();
    }, o.baseline || env_is(config.environment, "STRIX_BASELINE", "1"), config.gsq_owner);
    out << ",\"compact_stage_images_requested\":" << (compact_stage_images_requested(config.environment.contains("STRIX_QPF2K_COMPACT_IMAGES") ?
        config.environment.at("STRIX_QPF2K_COMPACT_IMAGES").c_str() : nullptr) ? "true" : "false");
    out << ",\"hc_lowband_policy\":" << (config.environment.contains("STRIX_QPF2K_HC_LOWBAND") ?
        std::strtoul(config.environment.at("STRIX_QPF2K_HC_LOWBAND").c_str(), nullptr, 0) : 0U);
    out << ",\"hc_lowband_route\":\"exact-fp16-output-only\"";
    out << ",\"index_batch\":" << (index_batch ? "true" : "false");
    out << ",\"early_setup\":" << (o.early_setup ? "true" : "false")
        << ",\"eager_code_load\":" << (o.eager_code_load ? "true" : "false")
        << ",\"hip_enable_deferred_loading\":";
    if (const auto it = config.environment.find("HIP_ENABLE_DEFERRED_LOADING"); it != config.environment.end())
        out << serve::Json(it->second).dump();
    else out << "null";
    out << "}\n";
    return out.str();
}
} // namespace strix::decode_cli
