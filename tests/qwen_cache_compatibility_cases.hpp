#pragma once
#include "strix/model_content_hash.hpp"
#include <filesystem>
#include <vector>
#include <string>

inline std::string without_cache_policy(std::string text) {
        if(const auto start=text.find("\"cache_policy\":");start!=std::string::npos) {
            std::size_t end=start+15;unsigned depth=0;bool quoted=false,escape=false;
            for(;end<text.size();++end) {
                const char c=text[end];
                if(quoted) {if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='\"')quoted=false;continue;}
                if(c=='\"')quoted=true;else if(c=='{')++depth;else if(c=='}' && --depth==0) {++end;break;}
            }
            if(end<text.size() && text[end]==',')++end;
            text.erase(start,end-start);
        }
    return text;
}
// In-flight joining and certified-disk reporting are additive fields inside
// "prefix_cache"; strip them so every incumbent field is still compared.
inline std::string without_prefix_reuse_policy(std::string text) {
    const auto object=text.find("\"prefix_cache\":{");
    if(object==std::string::npos) return text;
    for(const char* key:{"disk","certified_disk","durable","inflight","inflight_requested",
            "inflight_source","inflight_reason","inflight_policy"}) {
        std::size_t close=object+16;
        for(unsigned depth=1;close<text.size() && depth;++close) {
            if(text[close]=='\"') {for(++close;close<text.size() && text[close]!='\"';++close) if(text[close]=='\\') ++close;continue;}
            if(text[close]=='{') ++depth; else if(text[close]=='}') --depth;
        }
        const auto needle=std::string("\"")+key+"\":";
        auto start=text.find(needle,object);
        if(start==std::string::npos || start>=close) continue;
        auto end=start+needle.size();
        if(end<text.size() && text[end]=='\"') {for(++end;end<text.size() && text[end]!='\"';++end) if(text[end]=='\\') ++end;++end;}
        else while(end<text.size() && text[end]!=',' && text[end]!='}') ++end;
        if(end<text.size() && text[end]==',') ++end;
        else if(start>0 && text[start-1]==',') --start;
        text.erase(start,end-start);
    }
    return text;
}

// Remove only additive reports; retain the frozen bytes of every incumbent field.
inline std::string without_added_prefill_reports(std::string text) {
    for (const char* key : {"prefill_identity", "prefill_identity_hash", "prefill_options_active",
            "compact_stage_images_requested", "hc_lowband_policy", "hc_lowband_route", "index_batch",
            "early_setup", "eager_code_load", "hip_enable_deferred_loading", "t16k_policy", "profile", "prefill_moe_route_bf16",
            "prefill_storage_identity", "prefill_storage_identity_hash",
            "prefill_moe_route_bf16_reason", "prefill_fast_components"}) {
        const auto needle = std::string("\"") + key + "\":";
        const auto start = text.find(needle);
        if (start == std::string::npos) continue;
        auto end = start + needle.size();
        unsigned depth = 0;
        bool quoted = false, escape = false;
        for (; end < text.size(); ++end) {
            const char c = text[end];
            if (quoted) {
                if (escape) escape = false;
                else if (c == '\\') escape = true;
                else if (c == '\"') quoted = false;
                continue;
            }
            if (c == '\"') quoted = true;
            else if (c == '{' || c == '[') ++depth;
            else if ((c == '}' || c == ']') && depth) --depth;
            else if (!depth && (c == ',' || c == '}')) break;
        }
        if (end < text.size() && text[end] == ',') text.erase(start, end + 1U - start);
        else text.erase(start && text[start - 1U] == ',' ? start - 1U : start,
            end - start + (start && text[start - 1U] == ',' ? 1U : 0U));
    }
    return text;
}

// Long-context items added after the frozen table ({"enabled":..,"suppression":..} objects): each is
// reported and tested on its own, so strip it from the frozen comparison.
inline std::string without_added_long_context_items(std::string text) {
    for(const char* key:{"STRIX_QSA_CACHE_ONLY","STRIX_QPF2K_GDN_CHUNK16"}) {
        // Only inside "long_context_items" (the requested environment may carry the same name).
        const auto object=text.find("\"long_context_items\":{");
        if(object==std::string::npos) return text;
        const auto close=text.find("}}",object);
        const auto needle=std::string("\"")+key+"\":{";
        auto start=text.find(needle,object);
        if(start==std::string::npos || (close!=std::string::npos && start>close)) continue;
        auto end=text.find('}',start+needle.size());
        if(end==std::string::npos) continue;
        ++end;
        if(end<text.size() && text[end]==',') ++end;
        else if(start>0 && text[start-1]==',') --start;
        text.erase(start,end-start);
    }
    return text;
}

// Public-main resolver output fingerprints. Generation uses the same fixture
// against the preserved source; expected values are in the adjacent table.
inline std::vector<std::pair<std::string,std::string>> cache_compatibility_records(const strix::ModelContract& model) {
    using namespace strix;
    using namespace strix::decode_cli;
    constexpr auto gib=UINT64_C(1)<<30;
    std::vector<std::pair<std::string,std::string>> out;
    const auto record=[&](std::string name,Options o,Environment env) {
        resolve_session_defaults(o,model,env,128*gib);
        resolve_prefix_defaults(o,model,env,128*gib);
        EffectiveConfig config;config.options=o;
        auto text=effective_config_json(config);
        // New report is additive; compare every incumbent field, including its
        // original representation/reason, rather than a handpicked subset.
        text=without_added_long_context_items(without_added_prefill_reports(without_prefix_reuse_policy(without_cache_policy(std::move(text)))));
        text+=';'+std::to_string(o.prefix_batch_cert)+';'+std::to_string(o.session_management_requested);
        text+=';'+std::to_string(ordinary_http_batch_slots(o))+';'+std::to_string(o.prefix_cache_every);
        text+=';'+std::to_string(o.prefix_cache_disk_bytes)+';'+o.prefix_cache_dir;
        text+=';'+std::to_string(o.session_capture_host_bytes);
        out.emplace_back(std::move(name),model_content_identity_hex(sha256_bytes(std::as_bytes(std::span(text)))));
    };
    for(const unsigned runner:{0U,1U,4U}) for(const unsigned suppression:{0U,1U,2U,3U}) {
        Options base;base.context_tokens=8192;base.generate_tokens=32;
        base.http_serve=runner!=0;base.http_batch=runner==4?4:0;
        base.known_prefix_window=true;base.known_prefix_full_rows=true;base.known_prefix_prefill_rows=2048;
        base.mtp.enabled=runner!=0;base.mtp.draft_chain=true;
        Environment env;
        if(suppression==1)base.baseline=true;
        if(suppression==2)env["STRIX_BASELINE"]="1";
        if(suppression==3)env["STRIX_QPF2K_DEFAULTS"]="0";
        const auto prefix=std::to_string(runner)+"-"+std::to_string(suppression)+"-";
        for(const auto* shape:{"budget","disk","cadence","budget-off","fixed","fixed-budget","session-off","session-dir","cli-io"}) {
            auto o=base;const std::string kind=shape;
            if(kind=="budget" || kind=="disk" || kind=="cadence" || kind=="budget-off") {
                o.prefix_cache_host_bytes=gib;o.supplied_options.insert("--prefix-cache-host-bytes");
                if(kind=="disk") {o.prefix_cache_dir="cache-fixture";o.prefix_cache_disk_bytes=2*gib;o.supplied_options.insert("--prefix-cache-dir");}
                if(kind=="cadence") {o.prefix_cache_every=65536;o.supplied_options.insert("--prefix-cache-every");}
                if(kind=="budget-off") {o.prefix_cache=false;o.supplied_options.insert("--prefix-cache");}
            } else {
                o.session_mtp_fixed=true;o.session_tiled_suffix=true;
                o.supplied_options.insert("--session-mtp-fixed");o.supplied_options.insert("--session-tiled-suffix");
                env["STRIX_SESSION_DISK"]="0";
                if(kind=="fixed-budget") {o.session_ram_bytes=7*gib;o.session_max_parked=3;o.session_capture_host_bytes=2*gib;
                    o.supplied_options.insert("--session-ram-bytes");o.supplied_options.insert("--session-max-parked");}
                if(kind=="session-off") {o.session_cache=false;o.supplied_options.insert("--session-cache");}
                if(kind=="session-dir") {o.session_dir="session-fixture";o.session_disk_bytes=3*gib;o.supplied_options.insert("--session-dir");o.supplied_options.insert("--session-disk-bytes");}
                if(kind=="cli-io") {o.session_save="checkpoint";o.session_load="checkpoint";o.session_live_control=true;}
            }
            record(prefix+kind,o,env);
        }
        for(const auto* spelling:{"0","1","l0","capture","l1"}) for(const bool cli:{false,true}) for(const bool store:{false,true}) {
            auto o=base;auto cert_env=env;
            if(store) {o.prefix_cache_host_bytes=gib;o.supplied_options.insert("--prefix-cache-host-bytes");}
            if(cli) {o.supplied_options.insert("--prefix-batch-cert");o.prefix_batch_cert=std::string_view(spelling)=="0"?0:std::string_view(spelling)=="capture"?2:std::string_view(spelling)=="l1"?3:1;cert_env["STRIX_PREFIX_BATCH_CERT"]="0";}
            else cert_env["STRIX_PREFIX_BATCH_CERT"]=spelling;
            record(prefix+"cert-"+spelling+(cli?"-cli":"-env")+(store?"-store":"-observer"),o,cert_env);
        }
    }
    return out;
}
