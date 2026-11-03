// Included inside strix::decode_cli after the native resolver helpers.
// Classify code paths, not default values or environment provenance. New items
// fail closed until their rotary sites have been audited against the B3 inventory.
enum class YarnEnvironmentPath { no_rotary, b3_rotary, uncovered_rotary };
inline YarnEnvironmentPath yarn_environment_path(qpf2k::Item item) {
    using enum qpf2k::Item;
    switch (item) {
    case qsa: case qsa_mask: case norm_group:
        // hip_indexed_attention_launcher.hip:5768-5825,5885-5890,5993-6020;
        // hip_qsa_prefill_batch.inc:193-220 (batched table/norm/pool).
        return YarnEnvironmentPath::b3_rotary;
    case sched: case dense: case gdn: case gdn_gates_batch: case gdn_core16:
    case hc: case hc_mask: case hc_output3: case hc_output3_band: case hc_lowband:
    case moe_da16: case moe_route: case ple_reclaim: case dense_rows: case pipe:
    case attend_batch: case score_park: case index_rows: case cohort_lean:
    case rows_lean: case dispatch_fuse:
    // Audited at the 86e169e1 merge: score_rows re-tiles the stage-batch scorer over
    // already-rotated pooled keys (hip_qsa_prefill_flash.inc:25-43); ple_batch, shared_pre,
    // moe_mc, moe_hdr and moe_xor64 are PLE, shared-expert and MoE projections. None adds rotary work.
    case score_rows: case ple_batch: case shared_pre: case moe_mc: case moe_hdr: case moe_xor64:
        return YarnEnvironmentPath::no_rotary;
    }
    return YarnEnvironmentPath::uncovered_rotary;
}
// Explicit controls outside the qpf2k items, one row per variable. nonrotary rows were audited when
// admitted (comments below); off_switch rows select an existing exact alternative that adds no rotary
// site; diagnostic rows only time or print on the host; refused rows change arithmetic or are not
// audited for YaRN and refuse with their reason. Unlisted variables refuse by name.
enum class YarnEnvironmentClass { nonrotary, off_switch, diagnostic, refused };
struct YarnEnvironmentRule {
    std::string_view name;
    YarnEnvironmentClass kind;
    std::string_view values; // accepted values separated by '|'; empty accepts any value
    std::string_view why = {};
};
inline constexpr std::array yarn_environment_rules{
    // Bundle scheduling and projection controls add no rotary work.
    // Route validation separately excludes unqualified MTP arithmetic.
    YarnEnvironmentRule{"STRIX_QPF2K_T16K", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_QPF2K_INDEX_BATCH", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_QPF2K_EARLY_SETUP", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_EAGER_CODE_LOAD", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"HIP_ENABLE_DEFERRED_LOADING", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_PREFILL_INDEX_PROJ_WMMA", YarnEnvironmentClass::nonrotary, ""},
    // Operational policies add no rotary work. Prefix policy uses the
    // shared parser (validates the value and applies CLI precedence);
    // serving tile/unit controls cannot execute here because the route
    // guard separately bounds explicit MTP and refuses caching and HTTP.
    YarnEnvironmentRule{"STRIX_PREFIX_BATCH_CERT", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_SERVE_PREFILL", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_BATCH_PROMPT_SMALL", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_PREFIX_CACHE_INFLIGHT", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_PREFIX_CACHE_DISK", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_PREFIX_CACHE_DURABLE", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_PREFIX_FAIR_RECLAIM_IDLE", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_PREFILL_EXACT_FP16", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_PREFILL_HC_OUTPUT_WMMA", YarnEnvironmentClass::nonrotary, ""},
    // These controls perform no rotation. MOE_DEFAULTED is a two-bit
    // provenance mask (0..3), not a boolean execution feature.
    YarnEnvironmentRule{"STRIX_QPF2K_MOE_DEFAULTED", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_QPF2K_DEFAULTS", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_PREFILL_F16", YarnEnvironmentClass::nonrotary, ""},
    // No rotary: DA16 certificate down (src/hip_moe_prefill_da16.inc:140;
    // src/hip_moe_da16_down_body.inc:103-104) only rounds route-dot storage;
    // scalar/vector merge/commit (src/hip_moe_experts_launcher.hip:2768,2773;
    // src/hip_moe_prefill_commit.inc:18,71,115-117) only widens it before FMA.
    // Match the candidate parser: empty or any 0-prefix is OFF, all other
    // nonempty values are ON. Its writer/reader/controller guards still apply.
    YarnEnvironmentRule{"STRIX_QPF2K_MOE_ROUTE_BF16", YarnEnvironmentClass::nonrotary, ""},
    // No rotary: the chunked GDN prefill route (src/hip_gdn_fp16_chunk.inc) replaces conv, recurrence and gated
    // norm of linear-attention layers only; RoPE lives in the QSA layers it does not touch.
    YarnEnvironmentRule{"STRIX_QPF2K_GDN_FP16_CHUNK", YarnEnvironmentClass::nonrotary, ""},
    // Serial decode pool controls: the pool's only rotary site is block_pool_kernel under with_qwen_rotary
    // (qsa_decode_pool_scores, bound to the table by bind_qwen_pool); the self-check re-derives the scores with
    // block_score_kernel from the same table; selection and key storage perform no rotation. The fused maxima and the
    // decode attention score only reorganize the launches of already-rotated keys and queries.
    YarnEnvironmentRule{"STRIX_QSA_DECODE_POOL", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_QSA_DECODE_POOL_CHECK", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_QSA_DECODE_POOL_BF16", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_QSA_DECODE_SELECT", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_QSA_DECODE_FUSED_MAXIMA", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_QSA_DECODE_ATTN_V2", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_QSA_DECODE_VALUE_V2", YarnEnvironmentClass::nonrotary, ""},
    // Serial MoE glue launches (clear, route checks, quantizer, merge, scan, commit, status) perform no rotation.
    YarnEnvironmentRule{"STRIX_MOE_R1_GLUE_FUSE", YarnEnvironmentClass::nonrotary, ""},
    // Last-layer cache-only QSA step (strix/qsa_cache_only.hpp): its only rotary site is the k norm/rope of the
    // full step (norm_rope group/wave/per-set kernels under with_qwen_rotary) with the q / index-query heads left out.
    YarnEnvironmentRule{"STRIX_QSA_CACHE_ONLY", YarnEnvironmentClass::nonrotary, ""},
    // Compact tiled-prefill memory controls choose where scores, maxima, rounded output images and terminal logits live; the score ring and
    // stage split reuse the same table-bound norm/RoPE, pool and scorer launches, so they add no rotary site.
    YarnEnvironmentRule{"STRIX_QPF2K_COMPACT_IMAGES", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_CTX_EXACT_MEMORY", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_QSA_SCORE_RING", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_QSA_SCORE_RING_ROWS", YarnEnvironmentClass::nonrotary, ""},
    YarnEnvironmentRule{"STRIX_TERMINAL_LOGITS_ARENA", YarnEnvironmentClass::nonrotary, ""},
    // Closed controls for the audited single-request greedy route. Values
    // are checked here, including inherited Linux environment diagnostics.
    YarnEnvironmentRule{"STRIX_PROMPT_STATE_DIGEST", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_PREFIX_CACHE_DIGEST", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_PREFIX_CACHE_DIGEST_COMPONENTS", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_FINAL_STATE_DIGEST", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_RELEASE_EVENTS", YarnEnvironmentClass::diagnostic, "0|1"},
    YarnEnvironmentRule{"STRIX_STARTUP_MARKERS", YarnEnvironmentClass::diagnostic, "0|1"},
    YarnEnvironmentRule{"STRIX_SERIAL_LOGITS_DIGEST", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_PREFILL_LOGITS_DIGEST", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_PLE_ROLLING", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_PLE_ROLLING_STATS", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_PLE_DIRECT_IO", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_PLE_DIRECT_IO_STATS", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_MTP_DRAFT_Q4_ALL", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_MTP_DRAFT_WORTH", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_MTP_CHAIN_SHADOW", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_MTP_DRAFT_POLL", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_MTP_DRAFT_TOP2", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_MTP_REQUEST_RESET", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_MTP_GREEDY_ROWS", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_MTP_CACHE_AUDIT", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_WINDOW_SETUP_PREFLIGHT", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_MTP_DIAG_FULL_CHAIN", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_CYCLE_LOG", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_MTP_DRAFT_LOG", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_QSA_POOL_SLAB", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_QSA_POOL_PACKED", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_QSA_WINDOW_POOLED", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_QSA_WINDOW_GROUPED", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_QSA_PREFILL_POOLED", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_QSA_POOLED_CHECK", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_MTP_DRAFT_CACHE", YarnEnvironmentClass::nonrotary, "mixed|fast|teacher"},
    YarnEnvironmentRule{"STRIX_MTP_DRAFT_VOCAB", YarnEnvironmentClass::nonrotary, "1|2"},
    YarnEnvironmentRule{"STRIX_MTP_CONTROLLER", YarnEnvironmentClass::nonrotary, "minp|legacy"},
    YarnEnvironmentRule{"STRIX_MTP_WINDOW_COST", YarnEnvironmentClass::nonrotary, "legacy"},
    YarnEnvironmentRule{"STRIX_MTP_DOWN_WAVES", YarnEnvironmentClass::nonrotary, "8|16|32"},
    // These source-page scheduling/accounting controls add no rotary work.
    YarnEnvironmentRule{"STRIX_PLE_RECLAIM_STRICT", YarnEnvironmentClass::nonrotary, "0|1"},
    YarnEnvironmentRule{"STRIX_PLE_STAGE_PREFETCH", YarnEnvironmentClass::nonrotary, "0|1"},
    // OFF switches of code-default exact items (docs/optimizations.json off_switch); "1" is the code default.
    // Each OFF arm is the incumbent launch with the same bits and none adds a rotary site: the union merge
    // orders selections already made (hip_indexed_attention_launcher.hip:5969,6090); SCORE_ROWS64 picks the
    // stage-batch scorer rows over already-rotated pooled keys (hip_qsa_prefill_flash.inc:45); MERGE_VEC,
    // WMMA_PIN, HC_UPCOPY and ROUTER_SELECT reorganize the MoE merge, dense WMMA registers, HC up-weight
    // reads and router selection.
    YarnEnvironmentRule{"STRIX_QSA_UNION_MERGE", YarnEnvironmentClass::off_switch, "0|1"},
    YarnEnvironmentRule{"STRIX_QSA_SCORE_ROWS64", YarnEnvironmentClass::off_switch, "0|1"},
    YarnEnvironmentRule{"STRIX_PREFILL_MOE_MERGE_VEC", YarnEnvironmentClass::off_switch, "0|1"},
    YarnEnvironmentRule{"STRIX_QPF2K_WMMA_PIN", YarnEnvironmentClass::off_switch, "0|1"},
    YarnEnvironmentRule{"STRIX_QPF2K_HC_UPCOPY", YarnEnvironmentClass::off_switch, "0|1"},
    YarnEnvironmentRule{"STRIX_PREFILL_ROUTER_SELECT", YarnEnvironmentClass::off_switch, "0|1"},
    // Unset is stage mode 2 and "1" a different split, so only the OFF arm is classified.
    YarnEnvironmentRule{"STRIX_ROUTER_STAGE_ACT", YarnEnvironmentClass::off_switch, "0"},
    // Verify-window score over already-rotated queries and keys (hip_indexed_attention_launcher.hip:6652).
    YarnEnvironmentRule{"STRIX_QSA_WINDOW_ATTN_V2", YarnEnvironmentClass::off_switch, "0|1"},
    // OFF only; caching itself stays refused by the route guard.
    YarnEnvironmentRule{"STRIX_PREFIX_CACHE", YarnEnvironmentClass::off_switch, "0"},
    // Host-only diagnostics: steady_clock spans, stderr lines and host counters; no device work or synchronization.
    YarnEnvironmentRule{"STRIX_PREFILL_TIMING", YarnEnvironmentClass::diagnostic, "0|1"},
    YarnEnvironmentRule{"STRIX_QSA_FAST_STATS", YarnEnvironmentClass::diagnostic, "0|1"},
    YarnEnvironmentRule{"STRIX_QPF2K_DENSE_TRACE", YarnEnvironmentClass::diagnostic, "0|1"},
    YarnEnvironmentRule{"STRIX_QPF2K_HC_VERBOSE", YarnEnvironmentClass::diagnostic, "0|1"},
    YarnEnvironmentRule{"STRIX_PREFLIGHT", YarnEnvironmentClass::diagnostic, "0|1"},
    // Arithmetic or unaudited controls, named so the refusal states why.
    YarnEnvironmentRule{"STRIX_HCW_OUTPUT", YarnEnvironmentClass::refused, "", "changes HC output arithmetic"},
    YarnEnvironmentRule{"STRIX_HCW_NORM16", YarnEnvironmentClass::refused, "", "changes HC norm arithmetic"},
    YarnEnvironmentRule{"STRIX_HCW_NORMFREE", YarnEnvironmentClass::refused, "", "HC norm route not qualified on YaRN"},
    YarnEnvironmentRule{"STRIX_QSA_FAST_TILE", YarnEnvironmentClass::refused, "", "QSA attend tiling not qualified on YaRN"},
    YarnEnvironmentRule{"STRIX_QSA_FAST_SPLITS", YarnEnvironmentClass::refused, "", "changes the QSA attend split combine"},
    YarnEnvironmentRule{"STRIX_QPF2K_GDN_PARTS", YarnEnvironmentClass::refused, "", "selects unqualified GDN parts"},
    YarnEnvironmentRule{"STRIX_QPF2K_QSA_TRACE", YarnEnvironmentClass::refused, "", "device trace not audited for YaRN"},
    YarnEnvironmentRule{"STRIX_QPF2K_MOE_P1STATS", YarnEnvironmentClass::refused, "", "device counters not audited for YaRN"},
    YarnEnvironmentRule{"STRIX_QPF2K_GDN_CHECK", YarnEnvironmentClass::refused, "", "device check not audited for YaRN"},
    YarnEnvironmentRule{"STRIX_GDN_PREFILL_CHECK", YarnEnvironmentClass::refused, "", "device check not audited for YaRN"},
    YarnEnvironmentRule{"STRIX_QPF2K_HC_CHECK", YarnEnvironmentClass::refused, "", "device check not audited for YaRN"},
    YarnEnvironmentRule{"STRIX_PLE_WARM", YarnEnvironmentClass::refused, "", "page residency not audited for YaRN admission"},
    YarnEnvironmentRule{"STRIX_PLE_WINDOW_WARM", YarnEnvironmentClass::refused, "", "page residency not audited for YaRN admission"},
    YarnEnvironmentRule{"STRIX_MODEL_REGISTER_MLOCK", YarnEnvironmentClass::refused, "", "page residency not audited for YaRN admission"},
};
inline const YarnEnvironmentRule* yarn_environment_rule(std::string_view name) {
    const auto it = std::find_if(yarn_environment_rules.begin(), yarn_environment_rules.end(),
        [&](const auto& rule) { return rule.name == name; });
    return it == yarn_environment_rules.end() ? nullptr : &*it;
}
inline bool yarn_environment_value_accepted(const YarnEnvironmentRule& rule, std::string_view value) {
    if (rule.kind == YarnEnvironmentClass::refused) return false;
    if (rule.values.empty()) return true;
    for (std::size_t first = 0;;) {
        const auto bar = rule.values.find('|', first);
        if (rule.values.substr(first, bar - first) == value) return true;
        if (bar == std::string_view::npos) return false;
        first = bar + 1U;
    }
}
// The only STRIX_QPF2K_SCHED value YaRN admits; a defaulted schedule resolves to it.
inline constexpr const char* yarn_qualified_schedule = "12";
inline std::string yarn_environment_refusal(const Environment& environment, unsigned factor) {
    if (const auto it = environment.find("STRIX_QPF2K_SCHED");
        it != environment.end() && it->second != yarn_qualified_schedule)
        return "YaRN requires STRIX_QPF2K_SCHED=12; got " + it->second;
    // Every offending variable is listed; a lone unclassified name keeps the historical notice.
    std::vector<std::string> unsupported;
    for (const auto& [name, value] : environment) {
        if (storage_location_environment(name)) continue;
        const auto entry = std::find_if(qpf2k::defaults.begin(), qpf2k::defaults.end(),
            [&](const auto& item) { return name == item.name; });
        if (entry != qpf2k::defaults.end() &&
            yarn_environment_path(entry->item) != YarnEnvironmentPath::uncovered_rotary) continue;
        // M1 target/predictor producers and raw checks bind with_qwen_rotary;
        // packed consumers read those already-rotated keys. No second rotation.
        if (name == "STRIX_QSA_MTP_POOL" || name == "STRIX_QSA_MTP_POOL_CHECK" || name == "STRIX_MTP_LOGITS_DIGEST" || name == "STRIX_MTP_STEP_LOGITS_DIGEST") {
            if(value=="0" || value=="1") continue;
            return "YaRN diagnostic/pool switch must be 0 or 1: "+name;
        }
        if(name=="STRIX_QWEN_YARN_HOST_RESERVE") {
            if(value=="0" || value=="1") continue;
            return "YaRN host reserve switch must be 0 or 1";
        }
        if(name=="STRIX_QWEN_YARN_MTP") {
            if(value=="0" || value=="1") continue;
            return "YaRN MTP switch must be 0 or 1";
        }
        if (const auto* rule = yarn_environment_rule(name)) {
            if (yarn_environment_value_accepted(*rule, value)) continue;
            unsupported.push_back(name + " (" + (rule->kind == YarnEnvironmentClass::refused
                ? std::string(rule->why) : "accepts " + std::string(rule->values)) + ')');
            continue;
        }
        // Disabled rolling never parses its tuning values at runtime.
        if ((name=="STRIX_PLE_ROLLING_POSITIONS" || name=="STRIX_PLE_ROLLING_CAP_BYTES") &&
            (env_is(environment,"STRIX_PLE_ROLLING","0") || env_is(environment,"STRIX_CTX_EXACT_MEMORY","0"))) continue;
        if(name=="STRIX_PLE_ROLLING_CAP_BYTES" || name=="STRIX_PLE_ROLLING_POSITIONS" ||
           name=="STRIX_PLE_RECLAIM_DECODE_BYTES" || name=="STRIX_PLE_RECLAIM_AHEAD_ROWS") {
            try { if(parse_u64(value,"YaRN PLE bound")) continue; } catch(const std::exception&) {}
        }
        if (name == "STRIX_QWEN_ROPE_YARN" && value == std::to_string(factor)) continue;
        unsupported.push_back(name);
    }
    if (unsupported.empty()) return {};
    std::string listing = "YaRN execution environment not supported: ";
    for (std::size_t i = 0; i < unsupported.size(); ++i) listing += (i ? ", " : "") + unsupported[i];
    return listing;
}
inline std::uint64_t prefill_checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::overflow_error("prefill reservation multiplication overflow");
    return a * b;
}
inline EffectiveConfig resolve_yarn_config(const RequestedOptions& request,
    const ModelContract& model, const gguf::FileInfo& info, const ModelManifest& manifest,
    std::string_view content, const PrefillCapabilities& capabilities, const PrefillMemory& memory) {
    constexpr std::uint64_t gib = UINT64_C(1) << 30U;
    auto result = initialize_prefill_config(request);
    result.options.ple_rolling_explicit_opt_in=env_is(request.environment,"STRIX_PLE_ROLLING","1");
    result.options.qsa_mtp_pool_explicit_opt_in=env_is(request.environment,"STRIX_QSA_MTP_POOL","1");
    for (const auto& name : request.options.supplied_options) result.sources[name] = OptionSource::explicit_value;
    result.rotary_identity = model.rotary_identity();
    auto& o = result.options;
    if(const auto it=result.environment.find("STRIX_PREFILL_PIPE");it!=result.environment.end())
        o.yarn_mtp_pipeline=(std::strtoul(it->second.c_str(),nullptr,0)&1U)!=0U;
    o.yarn_mtp_q4_all=!env_is(result.environment,"STRIX_MTP_DRAFT_Q4_ALL","0");
    o.yarn_mtp_greedy_rows=!env_is(result.environment,"STRIX_MTP_GREEDY_ROWS","0");
    o.yarn_mtp_draft_top2=env_is(result.environment,"STRIX_MTP_DRAFT_TOP2","1");
    o.yarn_mtp_pool_off=env_is(result.environment,"STRIX_QSA_MTP_POOL","0");
    const auto refuse = [&](std::string reason) {
        result.reason = "yarn-refused";
        result.error = std::move(reason);
        return result;
    };
    if(o.mtp.enabled && env_is(request.environment,"STRIX_QWEN_YARN_MTP","0"))
        return refuse("explicit --mtp conflicts with STRIX_QWEN_YARN_MTP=0; omit MTP options for the serial YaRN comparator");
    if (const auto why = yarn_route_refusal(o); !why.empty()) return refuse(why);
    try {
        (void)validate_qwen_rotary_request(o.rotary.factor(), o.context_explicit, o.context_tokens,
            o.baseline || env_is(request.environment, "STRIX_BASELINE", "1"));
        if (o.tokens.empty() || o.tokens.size() > o.context_tokens ||
            o.generate_tokens > o.context_tokens - o.tokens.size())
            return refuse("YaRN prompt + generation exceeds requested context");
        if (model.max_position_embeddings != RotaryDescriptor::original_max ||
            model.rope_theta != RotaryDescriptor::theta || model.rotary_dim() != RotaryDescriptor::rotary_dim ||
            model.rotary_identity().descriptor() != o.rotary || !model.validate().empty())
            return refuse("YaRN resolved rotary/checkpoint contract mismatch");
        if (!memory.available || !memory.total || !memory.gtt_limit || !memory.gtt_used || !memory.estimate)
            return refuse("YaRN memory observation unavailable (MemTotal/MemAvailable/GTT required)");
        if (!capabilities.target_qualified || !capabilities.a8_runtime || content != validated_model_content_identity ||
            qpf2k::has_gsq_gate(manifest)) return refuse("YaRN checkpoint or target route not supported");
        const auto injected = qpf2k::initialize([&](const char* name) -> const char* {
            const auto it = result.environment.find(name);
            return it == result.environment.end() ? nullptr : it->second.c_str();
        }, [&](const char* name, const char* value) {
            // A defaulted schedule stays at YaRN's audited 12 (the native default 14 adds bit 2, which YaRN's owner
            // inventory and qualification do not cover). An explicit value other than 12 still refuses below.
            if (std::string_view(name) == "STRIX_QPF2K_SCHED") value = yarn_qualified_schedule;
            result.environment.emplace(name, value); return true;
        });
        for (const auto& entry : qpf2k::defaults)
            if (result.environment.contains(entry.name)) result.sources[entry.name] =
                injected & qpf2k::bit(entry.item) ? OptionSource::automatic : OptionSource::explicit_value;
        if (result.environment.contains("STRIX_QPF2K_MOE_DEFAULTED"))
            result.sources["STRIX_QPF2K_MOE_DEFAULTED"] = request.environment.contains("STRIX_QPF2K_MOE_DEFAULTED") ?
                OptionSource::explicit_value : OptionSource::automatic;
        // Missing with defaults disabled also means execution cannot use schedule 12.
        if (!result.environment.contains("STRIX_QPF2K_SCHED"))
            return refuse("YaRN requires STRIX_QPF2K_SCHED=12; effective value is unset");
        // Classify the complete effective environment before admission callbacks.
        if (const auto why = yarn_environment_refusal(result.environment, o.rotary.factor()); !why.empty())
            return refuse(why);
        if (!o.disabled_prefill_features.empty()) return refuse("YaRN disabled prefill feature not supported");
        o.unqualified_context = true;
        const bool explicit_low_level=o.any_low_level_flag;
        // yarn_route_refusal has admitted the closed explicit MTP organization.
        if(o.mtp.enabled) o.any_low_level_flag=false;
        auto organization = resolve_optimization_profile(o, model, info, manifest, content, capabilities.a8_runtime, false);
        o.any_low_level_flag=explicit_low_level;
        if (!organization.automatic_applied) return refuse("YaRN production organization unavailable: " + organization.reason);
        o.automatic_prefill = true;
        o.known_prefix_window = o.known_prefix_full_rows = o.known_prefix_expert_union = o.known_window_router_rows = true;
        o.known_prefix_output_policy = KnownWindowOutputPolicy::terminal_row;
        apply_measured_prompt_organization(o);
        o.numerical_profile = dense_a8_profile;
        // Keep one explicit lineage across all T trials. No FP16-to-A8 fallback.
        if (capabilities.fp16_operands && !env_is(request.environment,"STRIX_PREFILL_F16","0")) {
            o.prefill_numerical_profile = DecodeNumericalProfile::fast_prefill_f16wmma_v1;
            o.prefill_profile_automatic = true;
        }
        o.known_prefill_workspace_budget = 8U * gib;
        o.known_prefill_metadata_budget = gib;
        o.known_prefill_uma_budget = std::min(120U * gib, *memory.total);
        o.known_prefill_host_headroom = env_is(result.environment,"STRIX_QWEN_YARN_HOST_RESERVE","1")
            ? yarn_legacy_host_reserve : 0U;
        o.known_prefix_prefill_rows = 8192U;
        const auto default_environment = result.environment;
        const auto default_options = o;
        // These are source/scratch storage policies; no rotary site or numerical route changes.
        const bool adjustable_storage = !request.environment.contains("STRIX_PLE_ROLLING_POSITIONS") ||
            !request.environment.contains("STRIX_PLE_ROLLING_CAP_BYTES") ||
            !request.environment.contains("STRIX_QSA_SCORE_RING_ROWS") ||
            !request.environment.contains("STRIX_PREFILL_PIPE");
        const unsigned last_storage_step = o.mtp.enabled && ple_rolling_default_qualified(o) &&
            adjustable_storage ? 2U : 0U;
        std::uint64_t fixed_credit = 0, fixed_predicted_credit = 0;
        bool fixed_credit_ready = false;
        for (const auto rows : {8192U,4096U,2048U}) {
            for (unsigned storage_step = 0; storage_step <= last_storage_step; ++storage_step) {
                result.environment = default_environment;
                o = default_options;
                const auto storage_default = [&](const char* name, const char* value) {
                    if (!request.environment.contains(name)) {
                        result.environment[name] = value;
                        result.sources[name] = OptionSource::automatic;
                    }
                };
                if (storage_step) {
                    storage_default("STRIX_PLE_ROLLING_POSITIONS", storage_step == 1U ? "1024" : "256");
                    storage_default("STRIX_PLE_ROLLING_CAP_BYTES", storage_step == 1U ? "268435456" : "33554432");
                    storage_default("STRIX_QSA_SCORE_RING_ROWS", storage_step == 1U ? "256" : "64");
                    // The tight set releases the pipeline overlap owner; numerical routes stay fixed.
                    if (storage_step == 2U) storage_default("STRIX_PREFILL_PIPE", "0");
                }
                std::uint32_t score_ring_rows = 0;
                if (o.mtp.enabled) {
                    const auto ring = result.environment.find("STRIX_QSA_SCORE_RING_ROWS");
                    score_ring_rows = QsaScoreRingGeometry(o.rotary.effective_max(), 4U,
                        qsa_score_ring_rows_override(ring == result.environment.end() ? nullptr : ring->second.c_str())).rows;
                    o.yarn_mtp_pipeline = (std::strtoul(result.environment.at("STRIX_PREFILL_PIPE").c_str(), nullptr, 0) & 1U) != 0U;
                }
                // A failed fresh verification must not expose the preceding trial's credit/owners.
                if (last_storage_step) {
                    result.admission.clear();
                    result.ple_reclaim_credit = 0;
                    result.ple_rolling_h = result.ple_rolling_bound = result.ple_rolling_cap = 0;
                }
                std::uint64_t credit = fixed_credit, predicted_credit = fixed_predicted_credit;
                // Serial and fully pinned requests retain one verification across tile trials.
                if (last_storage_step || !fixed_credit_ready) {
                    credit = memory.verify_ple_credit ? memory.verify_ple_credit(o,result.environment) : 0U;
                    predicted_credit = prefill_reclaim_credit(o,model,manifest,result.environment,memory.ple_reclaim_page,
                        &result.ple_rolling_h,&result.ple_rolling_bound,&result.ple_rolling_cap);
                    fixed_credit = credit; fixed_predicted_credit = predicted_credit; fixed_credit_ready = true;
                }
                result.ple_reclaim_credit = credit;
                o.known_prefix_prefill_rows = rows;
                apply_default_known_prefill_tails(o, "12");
                auto validation = organization;
                apply_numerical_profile(o, model, manifest, content, capabilities.a8_runtime, validation);
                // Bind storage arithmetic before compiling the admission identities.
                std::string route_error;
                resolve_moe_route_bf16(o, result.environment, false, capabilities.fp16_operands, route_error);
                // The outer resolver withdraws declined profile proposals after admission.
                if (!route_error.empty() && !moe_route_bf16_profile_owned(o)) return refuse(route_error);
                validate_effective_options(o,model);
                result.admission = {{"ple_credit_verification_attempted",memory.verify_ple_credit?1U:0U},{"context_capacity",o.context_tokens},{"window_capacity_rows",rows},{"mtp_enabled",o.mtp.enabled?1U:0U},
                    {"parked_session_bytes",0},{"capture_bytes",0U},{"restore_peak_bytes",0},{"host_reserve_bytes",o.known_prefill_host_headroom},{"verified_ple_credit_bytes",credit},
                    {"uma_limit_bytes",o.known_prefill_uma_budget},{"available_bytes",*memory.available},
                    {"gtt_limit_bytes",*memory.gtt_limit},{"gtt_used_bytes",*memory.gtt_used},{"smallest_T_tried",rows}};
                if (o.mtp.enabled) {
                    result.admission["storage_ladder_step"] = storage_step;
                    result.admission["ple_rolling_positions"] = result.ple_rolling_h;
                    result.admission["ple_rolling_cap_bytes"] = result.ple_rolling_cap;
                    result.admission["qsa_score_ring_rows"] = score_ring_rows;
                    result.admission["prefill_pipe"] = std::strtoul(result.environment.at("STRIX_PREFILL_PIPE").c_str(), nullptr, 0);
                }
                std::string failure;
                try {
                    select_mtp_bundle_golden(o,model);
                    const auto required = o.mtp.enabled && memory.estimate_with_score_ring
                        ? memory.estimate_with_score_ring(o, score_ring_rows) : memory.estimate(o);
                    auto owners = required.owners;
                    if (owners.empty()) owners = {{"model_backing",required.model_backing},{"state_d1",required.state_d1},
                        {"serial_scratch",required.serial_scratch},{"workspace",required.workspace},{"metadata",required.metadata},
                        {"runtime_payload",required.runtime_payload},{"tails",required.tails}};
                    owners["rotary_table"] = o.rotary.table_bytes();
                    const auto rolling_owner = prefill_rolling_owner_bytes(o,model,manifest,result.environment,memory.ple_reclaim_page);
                    if (rolling_owner) owners["ple_rolling_owner"] = rolling_owner;
                    for(const auto& [name,bytes]:owners) result.admission["owner_"+name+"_bytes"]=bytes;
                    for(const auto& [name,cap]:required.owner_caps) {
                        result.admission["owner_"+name+"_cap_bytes"]=cap;
                        result.admission["owner_"+name+"_actual_bytes"]=required.owner_actuals.at(name);
                    }
                    const auto peaks=prefill_owner_phase_peaks(owners,required.owner_phases,o.known_prefill_host_headroom);
                    const std::array phase_names{"setup","calibration","prompt","decode","drain"};
                    for(unsigned i=0;i<phase_names.size();++i)
                        result.admission[std::string("phase_")+phase_names[i]+"_gross_bytes"]=peaks[i];
                    const auto gross=*std::max_element(peaks.begin(),peaks.end());
                    if(required.owners.contains("mtp_target_hidden_captures")) result.admission["capture_bytes"]=required.owners.at("mtp_target_hidden_captures");
                    if (credit > required.model_backing) throw std::invalid_argument("verified credit exceeds model backing");
                    const auto net = gross - credit;
                    result.admission["gross_required_bytes"] = gross;
                    result.admission["required_bytes"] = net;
                    // File credit affects the total envelope only. MemAvailable already includes
                    // reclaimable cache, so credit is NEVER added to incremental free headroom.
                    const auto incremental = gross - required.model_backing;
                    result.admission["incremental_bytes"] = incremental;
                    const auto device = prefill_checked_add(required.device_base,
                        prefill_checked_add(required.workspace, prefill_checked_add(required.tails, o.rotary.table_bytes())));
                    result.admission["device_growth_bytes"] = device;
                    if(!required.owner_error.empty()) throw std::invalid_argument(required.owner_error);
                    auto live = UmaOwnerInventory{};
                    live.reservation_upper_bound_bytes = required.model_backing;
                    live.known_file_mapping_bytes = required.model_backing;
                    // The runtime bridge uses this same ledger against its gross live
                    // inventory. Both checks subtract verified credit once, never from
                    // an already credited live value or physical/GTT headroom.
                    (void)plan_known_prefill_reservation(live, incremental, 0,
                        {o.known_prefill_uma_budget,o.known_prefill_uma_budget},credit,0);
                    if (!prefill_physical_margin_fits(*memory.total,*memory.available,net,incremental))
                        throw std::runtime_error("YaRN physical admission must retain 4 GiB margin");
                    if (incremental > *memory.available || *memory.gtt_used > *memory.gtt_limit ||
                        device > *memory.gtt_limit - *memory.gtt_used || !required.workspace ||
                        prefill_checked_add(required.workspace,required.tails) > o.known_prefill_workspace_budget ||
                        required.metadata > o.known_prefill_metadata_budget)
                        throw std::runtime_error("owner payload exceeds available/workspace/metadata/GTT limit");
                    result.host_compilation = required.organization;
                    if(o.mtp.enabled) {
                        result.host_compilation["yarn_mtp_contract_version"]=1U;
                        result.host_compilation["yarn_mtp_predictor_source_position"]=1U;
                        result.host_compilation["yarn_mtp_serial_verifier_required"]=1U;
                    }
                    result.organization = organization;
                    result.reason = "experimental-yarn";
                    result.admitted = true;
                    return result;
                } catch (const std::overflow_error&) { throw; }
                  catch (const std::exception& error) { failure = error.what(); }
                if (!result.tile_fallback.empty()) result.tile_fallback += ',';
                result.tile_fallback += "T" + std::to_string(rows) + ":insufficient-memory";
                const bool memory_refusal = failure == "UMA retained reservation exceeds explicit budget" ||
                    failure == "UMA setup peak reservation exceeds explicit budget" ||
                    failure == "YaRN physical admission must retain 4 GiB margin" ||
                    failure == "owner payload exceeds available/workspace/metadata/GTT limit";
                if (rows == 2048U && (!memory_refusal || storage_step == last_storage_step)) {
                    const bool provisional = o.print_effective_config && !memory.verify_ple_credit && predicted_credit &&
                        (failure == "UMA retained reservation exceeds explicit budget" ||
                         failure == "UMA setup peak reservation exceeds explicit budget");
                    result.admission["zero_credit_refusal_provisional"] = provisional ? 1U : 0U;
                    std::string listing = provisional
                        ? "YaRN admission provisional zero-credit inspection refusal (runtime-setup-required): "
                        : "YaRN admission refused: ";
                    listing += failure + "; context=" + std::to_string(o.context_tokens);
                    for (const auto& [name, bytes] : result.admission) listing += "; " + name + '=' + std::to_string(bytes);
                    return refuse(listing);
                }
                if (!memory_refusal) break;
            }
        }
    } catch (const std::exception& error) { return refuse(std::string("YaRN admission refused: ") + error.what()); }
    return refuse("YaRN admission refused");
}
