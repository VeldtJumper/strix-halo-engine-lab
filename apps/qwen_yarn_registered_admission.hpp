#pragma once
#include "qwen_effective_config.hpp"

namespace strix::decode_cli {
inline void record_registered_yarn_refusal(EffectiveConfig& current,std::string_view reason,
    const PrefillMemory& observed,std::uint64_t retained,std::uint64_t incremental,std::uint64_t device) {
    const auto step=current.admission.at("storage_ladder_step");
    const auto prefix="refused_step_"+std::to_string(step)+"_T"+std::to_string(current.options.known_prefix_prefill_rows)+"_";
    const auto shortfall=prefill_physical_shortfall(observed.total.value_or(0),observed.available.value_or(0),retained,incremental);
    const auto gtt_room=observed.gtt_limit && observed.gtt_used && *observed.gtt_used<=*observed.gtt_limit
        ? *observed.gtt_limit-*observed.gtt_used : 0U;
    // Same as the admission trial: usage already over the limit adds to the request.
    const auto gtt_shortfall=observed.gtt_limit && observed.gtt_used && *observed.gtt_used>*observed.gtt_limit
        ? prefill_checked_add(*observed.gtt_used-*observed.gtt_limit,device) : device>gtt_room?device-gtt_room:0U;
    const std::map<std::string,std::uint64_t> fields{{"observed_mem_total_bytes",observed.total.value_or(0)},
        {"observed_mem_available_bytes",observed.available.value_or(0)},
        {"registered_physical_retained_bytes",retained},{"registered_physical_incremental_bytes",incremental},
        {"retained_shortfall",shortfall.retained},{"incremental_shortfall",shortfall.incremental},
        {"uma_setup_shortfall",retained>current.options.known_prefill_uma_budget?retained-current.options.known_prefill_uma_budget:0U},
        {"gtt_shortfall",gtt_shortfall}};
    if (!current.tile_fallback.empty()) current.tile_fallback+=',';
    current.tile_fallback+="post-registration:step="+std::to_string(step)+":T"+
        std::to_string(current.options.known_prefix_prefill_rows)+':'+std::string(reason);
    for (const auto& [name,bytes]:fields) {
        current.admission[prefix+name]=bytes;
        current.tile_fallback+=';'+name+'='+std::to_string(bytes);
    }
}
// Continue the exact tile ladder after registration, before allocating target
// state or prompt owners. Storage/source leases are already live: pin their
// sizes and reuse only their actually verified credit. Never credit file cache
// to MemAvailable. Trials use the runtime physical/UMA/GTT envelope.
inline EffectiveConfig readmit_registered_yarn_mtp(const RequestedOptions& original,
    const EffectiveConfig& current, const ModelContract& model, const gguf::FileInfo& info,
    const ModelManifest& manifest, std::string_view content, const PrefillCapabilities& capabilities,
    PrefillMemory post, bool next_step = false) {
    const auto& o = current.options;
    if (o.rotary.mode() != RotaryMode::yarn || !o.mtp.enabled || !current.admitted)
        throw std::invalid_argument("registered tile retry requires admitted YaRN MTP");
    auto request = original;
    for (const char* name : {"STRIX_PLE_ROLLING_POSITIONS", "STRIX_PLE_ROLLING_CAP_BYTES",
            "STRIX_QSA_SCORE_RING_ROWS", "STRIX_PREFILL_PIPE"})
        if (const auto it = current.environment.find(name); it != current.environment.end())
            request.environment[name] = it->second;
    // Step zero can use implicit sizes that have no environment entry. Pin
    // those too; retry must never resize an already constructed source owner.
    if (current.ple_rolling_h) {
        request.environment["STRIX_PLE_ROLLING_POSITIONS"] = std::to_string(current.ple_rolling_h);
        request.environment["STRIX_PLE_ROLLING_CAP_BYTES"] = std::to_string(current.ple_rolling_cap);
    }
    request.environment["STRIX_QSA_SCORE_RING_ROWS"] = std::to_string(current.admission.at("qsa_score_ring_rows"));
    request.environment["STRIX_PREFILL_PIPE"] = std::to_string(current.admission.at("prefill_pipe"));
    post.yarn_mtp_max_prefill_rows = o.known_prefix_prefill_rows;
    if (next_step) {
        if (o.known_prefix_prefill_rows<=2048U) {
            auto refused=current; refused.admitted=false;
            refused.error="YaRN post-registration storage ladder exhausted; "+current.tile_fallback;
            return refused;
        }
        post.yarn_mtp_max_prefill_rows=o.known_prefix_prefill_rows>4096U?4096U:2048U;
    }
    post.yarn_registered_storage_step=current.admission.at("storage_ladder_step");
    post.yarn_registered_rotary_bytes = o.rotary.table_bytes();
    // Missing observations fail every remaining step closed, with diagnostics,
    // rather than escaping the ladder through an unrelated exception.
    if (!post.total) post.total=0U;
    if (!post.available) post.available=0U;
    if (!post.gtt_limit) post.gtt_limit=0U;
    if (!post.gtt_used) post.gtt_used=0U;
    post.verify_ple_credit = [&](const Options&, const Environment&) { return current.ple_reclaim_credit; };
    auto result = resolve_effective_config(request,model,info,manifest,content,capabilities,post);
    // Preserve all refusals even when the last step fails and the bundle
    // unwinds into its existing clean serial fallback.
    for (const auto& [name,bytes]:current.admission)
        if (name.starts_with("refused_step_")) result.admission.emplace(name,bytes);
    result.tile_fallback=current.tile_fallback+(current.tile_fallback.empty() || result.tile_fallback.empty()?"":",")+result.tile_fallback;
    if (!result.admitted) result.error += "; "+result.tile_fallback;
    if (result.admitted) {
        result.sources = current.sources;
        for (const char* name : {"STRIX_PLE_ROLLING_POSITIONS", "STRIX_PLE_ROLLING_CAP_BYTES",
                "STRIX_QSA_SCORE_RING_ROWS", "STRIX_PREFILL_PIPE"})
            if (result.environment.contains(name) && !result.sources.contains(name))
                result.sources[name] = OptionSource::automatic;
        const auto old_step = current.admission.at("storage_ladder_step");
        result.admission["storage_ladder_step"] = old_step;
        if (result.options.known_prefix_prefill_rows < o.known_prefix_prefill_rows)
            result.admission["storage_ladder_step"] = result.options.known_prefix_prefill_rows == 4096U ? 3U : 4U;
        result.admission["post_registration_readmission"] = 1U;
        result.admission["post_registration_available_bytes"] = result.admission.at("observed_mem_available_bytes");
        result.admission["post_registration_gtt_used_bytes"] = result.admission.at("gtt_used_bytes");
        result.tile_fallback += (result.tile_fallback.empty() ? "" : ",") + std::string(
            "post-registration:T" + std::to_string(o.known_prefix_prefill_rows) + "->T" +
            std::to_string(result.options.known_prefix_prefill_rows));
    }
    return result;
}
} // namespace strix::decode_cli
