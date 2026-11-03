#include "../apps/hip_decode_options.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Env {
    std::string key,old;bool had;
    static void set(const char* k,const char* v){
#ifdef _WIN32
        _putenv_s(k,v?v:"");
#else
        if(v)setenv(k,v,1);else unsetenv(k);
#endif
    }
    Env(const char* k,const char* v):key(k),had(std::getenv(k)!=nullptr){
        if(had)old=std::getenv(k);
        set(k,v);
    }
    ~Env(){set(key.c_str(),had?old.c_str():nullptr);}
};
void depth_policy_options_contract(){
    using namespace strix::decode_cli;
    Env mode("STRIX_MTP_CONTROLLER","depth"),domain("STRIX_MTP_DOMAIN","prose"),
        threshold("STRIX_MTP_MINP_PROSE",".83");
    MtpCliOptions options;options.window_rows_set={2,4,8};
    require(std::abs(mtp_draft_min_probability(options)-.83F)<1e-6F,"domain threshold applies to an implicit min-p");
    options.spec_min_p=.61;
    require(std::abs(mtp_draft_min_probability(options)-.61F)<1e-6F,"explicit CLI min-p overrides the domain");
    options.sample_temperature=1;
    bool rejected=false;try{(void)mtp_draft_min_probability(options);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unqualified sampled depth policy is rejected");
}
strix::decode_cli::Options parse(std::vector<std::string> args){
    std::vector<char*> argv;for(auto& value:args)argv.push_back(value.data());
    return strix::decode_cli::parse_options(static_cast<int>(argv.size()),argv.data());
}
std::vector<std::string> request(){
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
void set(std::vector<std::string>& values,const std::string& key,const std::string& value){
    for(std::size_t i=0;i<values.size();++i){
        if(values[i]==key && i+1U<values.size()){values[i+1U]=value;return;}
        if(values[i].starts_with(key+"=")){values[i]=key+"="+value;return;}
    }
    throw std::logic_error("fixture key absent");
}
void reject(const std::vector<std::string>& args,const char* message){
    bool failed=false;try{(void)parse(args);}catch(const std::exception&){failed=true;}require(failed,message);
}
void reject_as(const std::vector<std::string>& args,const char* expected,const char* message){
    bool failed=false;try{(void)parse(args);}catch(const std::invalid_argument& error){failed=std::string(error.what())==expected;}
    require(failed,message);
}
void erase_option(std::vector<std::string>& args,const std::string& key){
    for(auto it=args.begin();it!=args.end();++it){
        if(*it==key){require(it+1!=args.end(),"fixture option needs a value");args.erase(it,it+2);return;}
        if(it->starts_with(key+"=")){args.erase(it);return;}
    }
    throw std::logic_error("fixture option absent");
}
std::vector<std::string> experts_request(){
    return {"decoder","target.gguf","1","2","3","--generate","20","--context","8192",
        "--optimization-profile","auto","--model-copy-tier","experts",
        "--model-copy-payload-bytes","24656","--model-retained-budget","131072",
        "--model-setup-peak-budget","131072","--coalesce-gap","0"};
}
void expert_copy_cli_contract(){
    using namespace strix;
    using namespace strix::decode_cli;
    auto args=experts_request();
    set(args,"--model-copy-tier",std::string(hip_model_copy_tier_name(HipModelCopyTier::experts)));
    const auto parsed=parse(args);
    require(parsed.model_copy_tier==HipModelCopyTier::experts && parsed.model_copy_tier_explicit &&
        parsed.model_copy_payload_bytes==24656 && parsed.model_retained_budget==131072 &&
        parsed.model_setup_peak_budget==131072 && parsed.coalesce_gap_bytes==0 &&
        parsed.numerical_profile==DecodeNumericalProfile::exact_v5 && !parsed.mtp.enabled,
        "explicit experts round-trips its tier name and preserves exact payload/budgets/gap without selecting arithmetic or MTP");
    auto equals=args;erase_option(equals,"--model-copy-tier");equals.push_back("--model-copy-tier=experts");
    require(parse(equals).model_copy_tier==HipModelCopyTier::experts,"expert tier accepts equals syntax");
    auto missing_gap=args;erase_option(missing_gap,"--coalesce-gap");
    reject_as(missing_gap,"expert model copy requires --coalesce-gap 0","experts rejects the nonzero default gap");
    for(const auto* value:{"1","4096"}){
        auto bad=args;set(bad,"--coalesce-gap",value);
        reject_as(bad,"expert model copy requires --coalesce-gap 0","experts requires zero gap explicitly");
    }
    for(const auto* key:{"--model-copy-payload-bytes","--model-retained-budget","--model-setup-peak-budget"}){
        auto missing=args;erase_option(missing,key);
        reject_as(missing,"selective model copy requires positive exact payload, retained and setup-peak budgets",
            "experts rejects each missing reservation declaration before setup");
        for(const auto* value:{"0","-1","18446744073709551616"}){
            auto bad=args;set(bad,key,value);reject(bad,"experts rejects zero, negative and overflowing budget values");
        }
        auto duplicate=args;duplicate.push_back(std::string(key)+"=1");
        reject(duplicate,"experts rejects duplicate payload and budget declarations");
    }
    for(const auto* tier:{"experts","none","normalization","hot_nonexpert"}){
        auto duplicate=args;duplicate.push_back(std::string("--model-copy-tier=")+tier);
        reject_as(duplicate,"duplicate model copy tier option","tiers cannot be repeated or combined");
    }
    auto unknown=args;set(unknown,"--model-copy-tier","EXPERTS");reject(unknown,"expert tier spelling is exact");
    auto copied=args;copied.insert(copied.end(),{"--residency","copy"});
    reject_as(copied,"selective model copy requires register residency","experts excludes whole-copy residency");
    auto graph=args;graph.insert(graph.end(),{"--execution-mode","graph"});
    reject_as(graph,"selective model copy requires eager execution","experts excludes an unqualified controller");
    auto serving=args;serving.insert(serving.end(),{"--serve-concurrency","1"});
    reject_as(serving,"selective model copy excludes serving options","even an explicit default serving selector is not unused expert admission");
    auto mtp=request();mtp.insert(mtp.end(),{"--model-copy-tier=experts","--model-copy-payload-bytes=24656",
        "--model-retained-budget=131072","--model-setup-peak-budget=131072","--coalesce-gap","0"});
    reject_as(mtp,"selective model copy excludes MTP options","experts cannot implicitly qualify target/sidecar composition");
    for(const auto* profile:{"q51-a8-down-v1","hc-deferred-norm-f32+q51-q80-a8-down-v1",
        "hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1"}){
        auto bad=args;bad.insert(bad.end(),{"--numerical-profile",profile});
        reject_as(bad,"selective model copy requires exact-v5 or an admitted explicit expert recipe",
            "experts excludes every unadmitted arithmetic recipe");
    }
    for(const auto* tier:{"normalization","hot_nonexpert"})
        for(const auto* profile:{"hc-splitk-f32+q51-q80-a8-down-v1",
            "hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1",
            "hc-deferred-norm-f32+q51-q80-a8-down-v1",
            "hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1","q51-a8-down-v1"}){
            auto bad=args;set(bad,"--model-copy-tier",tier);bad.insert(bad.end(),{"--numerical-profile",profile});
            reject_as(bad,"selective model copy requires exact-v5 or an admitted explicit expert recipe",
                "existing nonexpert tiers remain exact-v5 only");
        }
    const auto model=ModelContract::qwen38_flash_next();gguf::FileInfo info;
    ModelManifest manifest;manifest.architecture="qwen3next";
    manifest.layout_identity=model_manifest_layout_identity(manifest);
    const auto resolve=[&](Options& o,bool has_a8){return resolve_optimization_profile(o,model,info,manifest,
        validated_model_content_identity,has_a8);};
    const auto same_organization=[](const Options& a,const Options& b){
        for(auto member:{&Options::parallel_hyper_mix,&Options::staged_hyper_projections,&Options::fused_staged_hyper_mix,
            &Options::q4k_a8g32_gate_up,&Options::grouped_exact_q51_down,&Options::q51_slot_parallel_down,
            &Options::batched_moe_activation,&Options::deferred_moe_status,&Options::wave_moe_router_topk,
            &Options::wave8_q8_gdn_input,&Options::wave8_q8_gdn_output,&Options::prefetch_q8_qsa_input,
            &Options::prefetch_q8_shared_expert,&Options::prefetch_q8_lm_head,&Options::wave8_q8_lm_head,
            &Options::wave8_q8_qsa_output,&Options::gdn_islands,&Options::prepared_eager,&Options::moe_shared_overlap})
            require(a.*member==b.*member,"copy tier must not alter resolved organization flags");
        require(a.numerical_profile==b.numerical_profile,"backing must not select arithmetic");
    };
    for(const auto* organization:{"auto","manual"})for(const bool has_a8:{false,true}){
        auto selected_args=args;set(selected_args,"--optimization-profile",organization);
        auto none_args=selected_args;set(none_args,"--model-copy-tier","none");erase_option(none_args,"--model-copy-payload-bytes");
        auto none=parse(none_args);auto expert=parse(selected_args);
        auto baseline=resolve(none,has_a8);auto selected=resolve(expert,has_a8);
        require(selected.requested==baseline.requested && selected.selected==baseline.selected &&
            selected.reason==baseline.reason && selected.automatic_applied==baseline.automatic_applied,
            "experts preserves manual, automatic and missing-runtime resolution before suffixing");
        same_organization(none,expert);
        const auto baseline_name=baseline.selected;apply_model_copy_profile(none,baseline);apply_model_copy_profile(expert,selected);
        require(none.model_copy_tier==HipModelCopyTier::none && baseline.selected==baseline_name &&
            selected.selected==baseline_name+"+model-copy-tier:experts" && selected.reason==baseline.reason &&
            selected.automatic_applied==baseline.automatic_applied,
            "none remains unsuffixed and experts adds only its explicit backing suffix");
        same_organization(none,expert);
    }
    // An explicit expert tier is source admission only: each arithmetic recipe
    // must still satisfy its existing resolver/prerequisite checks independently.
    for(const auto* profile:{"hc-splitk-f32+q51-q80-a8-down-v1",
        "hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1"}){
        auto selected_args=args;selected_args.insert(selected_args.end(),{"--numerical-profile",profile});
        auto none_args=selected_args;set(none_args,"--model-copy-tier","none");erase_option(none_args,"--model-copy-payload-bytes");
        auto none=parse(none_args);auto expert=parse(selected_args);auto baseline=resolve(none,true);auto selected=resolve(expert,true);
        require(selected.automatic_applied && baseline.automatic_applied,"explicit fast expert recipes retain admitted automatic prerequisites");
        apply_numerical_profile(none,model,manifest,validated_model_content_identity,true,baseline);
        apply_numerical_profile(expert,model,manifest,validated_model_content_identity,true,selected);
        same_organization(none,expert);
        require(selected.selected==baseline.selected,"same numerical recipe resolves identically before backing suffix");
        apply_model_copy_profile(none,baseline);apply_model_copy_profile(expert,selected);
        require(selected.selected==baseline.selected+"+model-copy-tier:experts",
            "explicit expert composition preserves the numerical recipe name and adds only backing provenance");
        auto manual_args=selected_args;set(manual_args,"--optimization-profile","manual");
        auto manual=parse(manual_args);auto unresolved=resolve(manual,true);
        bool rejected=false;try{apply_numerical_profile(manual,model,manifest,validated_model_content_identity,true,unresolved);}
        catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"experts cannot invent missing manual numerical prerequisites");
    }
    for(const auto* organization:{"auto","manual"}){
        auto defaults=parse({"decoder","target.gguf","1","--optimization-profile",organization});
        auto selected=resolve(defaults,true);const auto name=selected.selected;apply_model_copy_profile(defaults,selected);
        require(defaults.model_copy_tier==HipModelCopyTier::none && !defaults.model_copy_tier_explicit &&
            !defaults.model_copy_payload_bytes && selected.selected==name,
            "no manual or automatic profile implicitly selects experts or any other selective tier");
    }
    auto implicit_args=args;erase_option(implicit_args,"--optimization-profile");auto implicit=parse(implicit_args);
    require(resolve(implicit,true).reason=="explicit-low-level-options",
        "the existing explicit-gap low-level fallback is unchanged without an explicit organization selector");
}
}
// D-055: decode defaults from the final campaign. The draft early-stop probability follows the draft's
// identity, --mtp-draft-profile fills a qualified draft's pins, and the n-gram extension is auto by default.
void campaign_defaults_contract(){
    using namespace strix;
    using namespace strix::decode_cli;
    MtpCliOptions m;
    m.window_rows_set={2U,4U,8U};
    m.sidecar_sha256=std::string(find_mtp_draft_profile("distill-g")->sidecar_sha256);
    require(mtp_draft_min_probability(m)==0.7F,"distillG defaults to min-p 0.7");
    m.sidecar_sha256=std::string(find_mtp_draft_profile("distill-d")->sidecar_sha256);
    require(mtp_draft_min_probability(m)==0.7F,"distillD defaults to min-p 0.7");
    m.sidecar_sha256=std::string(find_mtp_draft_profile("base")->sidecar_sha256);
    require(mtp_draft_min_probability(m)==0.85F,"the base draft defaults to min-p 0.85 (D-022)");
    m.sidecar_sha256=std::string(64,'1');
    require(mtp_draft_min_probability(m)==0.5F,"an unprofiled draft keeps 0.5");
    m.spec_min_p=0.6;
    require(mtp_draft_min_probability(m)==0.6F,"an explicit min-p wins");
    m.spec_min_p=-1.0;m.window_rows_set.clear();
    require(mtp_draft_min_probability(m)==0.0F,"a single window does not stop the chain early");
    require(find_mtp_draft_profile(mtp_recommended_draft_profile)!=nullptr,"the recommended profile exists");

    // --mtp-draft-profile fills the pins that are not given; a given pin must agree.
    auto args=request();
    for(const auto* key:{"--mtp-sidecar-logical-name","--mtp-sidecar-sha256","--mtp-source-repository","--mtp-source-revision",
        "--mtp-source-path","--mtp-converter-revision","--mtp-converter-sha256","--mtp-norm-converter-sha256","--mtp-sidecar-bytes"})
        erase_option(args,key);
    reject(args,"a request without the sidecar pins is rejected");
    args.insert(args.end(),{"--mtp-draft-profile","distill-g"});
    const auto profiled=parse(args);
    const auto* g=find_mtp_draft_profile("distill-g");
    require(profiled.mtp.sidecar_sha256==g->sidecar_sha256 && profiled.mtp.sidecar_logical_name==g->logical_name &&
        profiled.mtp.repository==g->repository && profiled.mtp.revision==g->revision && profiled.mtp.source_path==g->source_path &&
        profiled.mtp.converter_revision==g->converter_revision && profiled.mtp.converter_sha256==g->converter_sha256 &&
        profiled.mtp.norm_converter_sha256==g->norm_converter_sha256 && profiled.mtp.sidecar_bytes==g->sidecar_bytes,
        "the distill-g profile fills every sidecar pin");
    auto agree=args;agree.insert(agree.end(),{"--mtp-sidecar-sha256",std::string(g->sidecar_sha256)});
    (void)parse(agree);
    auto disagree=args;disagree.insert(disagree.end(),{"--mtp-sidecar-sha256",std::string(64,'1')});
    reject_as(disagree,"MTP draft profile disagrees with an explicit sidecar pin","a disagreeing pin is rejected");
    auto unknown=request();unknown.insert(unknown.end(),{"--mtp-draft-profile","other"});
    reject_as(unknown,"unknown MTP draft profile (base, distill-d, distill-g)","an unknown profile is rejected");

    // --mtp-ngram-draft[=on|off|auto]: auto (the default) runs where admissible.
    MtpCliOptions n;
    require(!mtp_ngram_draft_requested(n),"auto without the draft chain and prefix windows does not run");
    n.draft_chain=true;n.window_rows=8U;n.window_prefix=true;
    require(mtp_ngram_draft_admissible(n) && mtp_ngram_draft_requested(n),"auto runs with the chain and prefix windows");
    n.ngram_off=true;require(!mtp_ngram_draft_requested(n),"off disables it");
    n.ngram_off=false;n.sample_options=true;require(!mtp_ngram_draft_requested(n),"auto stays off when sampling");
    n.sample_options=false;n.cohort.push_back({});require(!mtp_ngram_draft_requested(n),"auto stays off with cohort requests");
    n.cohort.clear();n.golden_dump="g";require(!mtp_ngram_draft_requested(n),"auto stays off when dumping a golden");
    auto mode=request();mode.push_back("--mtp-ngram-draft=off");
    require(parse(mode).mtp.ngram_off && !parse(mode).mtp.ngram_draft,"=off parses");
    mode=request();mode.push_back("--mtp-ngram-draft=auto");
    require(!parse(mode).mtp.ngram_off && !parse(mode).mtp.ngram_draft,"=auto parses");
    require(!n.ngram_context && !n.ngram_copy_edit,"the context drafter is never a default");
    mode=request();mode.push_back("--mtp-ngram-draft=copy");
    reject(mode,"copy still requires the chain and prefix windows");
    mode=request();mode.push_back("--mtp-ngram-draft=ctx");
    reject(mode,"ctx still requires the chain and prefix windows");
    mode=request();mode.insert(mode.end(),{"--mtp-draft-chain","--mtp-target-window-rows","8","--mtp-window-prefix",
        "--mtp-window-capture-bytes","8192","--mtp-window-checkpoint-bytes","8192",
        "--mtp-window-total-bytes","32768","--mtp-window-arena-bytes","8192",
        "--mtp-window-history-intervals","32","--mtp-ngram-draft=copy"});
    set(mode,"--mtp-candidate-depth","7");
    const auto copied=parse(mode).mtp;
    require(copied.ngram_draft && copied.ngram_context && copied.ngram_copy_edit && !copied.ngram_off,
        "copy parses as a required context drafter with the exact window prerequisites");
    mode.back()="--mtp-ngram-draft=ctx";
    const auto context=parse(mode).mtp;
    require(context.ngram_draft && context.ngram_context && !context.ngram_copy_edit,"ctx parses without copy-first");
    mode.back()="--mtp-ngram-draft=wide";
    const auto wide_only=parse(mode).mtp;
    require(wide_only.ngram_draft && wide_only.ngram_context && wide_only.ngram_copy_edit && wide_only.ngram_wide_only,
        "wide parses as the extension plus an observing copy drafter");
    require(!copied.ngram_wide_only && !context.ngram_wide_only && !n.ngram_wide_only,"only =wide is observe-only");
    mode=request();mode.push_back("--mtp-ngram-draft=maybe");
    reject_as(mode,"--mtp-ngram-draft is on, off, auto, ctx, copy or wide","an unknown n-gram mode is rejected");
    mode=request();mode.insert(mode.end(),{"--mtp-ngram-draft=off","--mtp-ngram-draft"});
    reject(mode,"duplicate n-gram option rejected");
    mode=request();mode.push_back("--mtp-ngram-draft");
    reject(mode,"an explicit n-gram draft still requires the chain and prefix windows");

    // --mtp-copy-wide-rows: off by default; 32, 16 or 32,16 need =copy on the tiled MTP prompt and add their tail blocks.
    require(parse(request()).mtp.copy_wide_rows.empty(),"wide copy verify is off by default");
    auto wide=request();wide.insert(wide.end(),{"--mtp-draft-chain","--mtp-target-window-rows","8","--mtp-window-prefix",
        "--mtp-window-capture-bytes","8192","--mtp-window-checkpoint-bytes","8192",
        "--mtp-window-total-bytes","32768","--mtp-window-arena-bytes","8192",
        "--mtp-window-history-intervals","32","--mtp-ngram-draft=copy","--mtp-calibration-rows","2",
        "--known-prefix-window","--known-prefix-full-rows","--known-prefix-prefill-rows","256",
        "--known-prefix-workspace-budget-bytes","1","--known-prefix-host-metadata-budget-bytes","2",
        "--known-prefix-uma-budget-bytes","1","--known-prefix-host-headroom-bytes","0"});
    set(wide,"--mtp-candidate-depth","7");
    const auto narrow=parse(wide);
    require(narrow.mtp.copy_wide_rows.empty() && narrow.known_prefix_tail_rows==std::vector<std::uint32_t>{64U,16U},
        "the tiled copy route keeps its default tails without the wide option");
    auto wide32=wide;wide32.insert(wide32.end(),{"--mtp-copy-wide-rows","32"});
    const auto parsed_wide=parse(wide32);
    require(parsed_wide.mtp.copy_wide_rows==std::vector<std::uint32_t>{32U} && parsed_wide.known_prefix_tail_rows==std::vector<std::uint32_t>{64U,32U,16U},
        "--mtp-copy-wide-rows 32 parses and adds the 32-row tail block in descending order");
    auto off=wide;off.push_back("--mtp-copy-wide-rows=0");
    require(parse(off).mtp.copy_wide_rows.empty(),"an explicit 0 keeps the feature off");
    for(const auto* bad:{"31","64","8","16,32","32,32","32,16,16","0,32"}) {
        auto other=wide;other.insert(other.end(),{"--mtp-copy-wide-rows",bad});
        reject_as(other,"MTP copy wide rows must be 0 or a descending list of distinct 32/16",
            "wide rows other than 0 or a descending list from 32 and 16 are rejected");
    }
    auto both=wide;both.insert(both.end(),{"--mtp-copy-wide-rows","32,16"});
    const auto parsed_both=parse(both);
    require(parsed_both.mtp.copy_wide_rows==std::vector<std::uint32_t>{32U,16U} &&
        parsed_both.known_prefix_tail_rows==std::vector<std::uint32_t>{64U,32U,16U},
        "--mtp-copy-wide-rows 32,16 parses largest first and adds every listed tail block once");
    auto only16=wide;only16.push_back("--mtp-copy-wide-rows=16");
    const auto parsed16=parse(only16);
    require(parsed16.mtp.copy_wide_rows==std::vector<std::uint32_t>{16U} &&
        parsed16.known_prefix_tail_rows==std::vector<std::uint32_t>{64U,16U},"a single 16 keeps the default tails");
    auto missing16=both;missing16.insert(missing16.end(),{"--known-prefix-tail-rows","64,32"});
    reject_as(missing16,"--mtp-copy-wide-rows requires a tiled tail block of each listed size "
        "(--known-prefix-tail-rows including it, below --known-prefix-prefill-rows)","explicit tails must include every wide size");
    const auto wide_requires="--mtp-copy-wide-rows requires --mtp-ngram-draft=copy or =wide on the tiled MTP prompt, "
        "one request and no golden dump";
    auto ctx=wide32;for(auto& value:ctx)if(value=="--mtp-ngram-draft=copy")value="--mtp-ngram-draft=ctx";
    reject_as(ctx,wide_requires,"wide rows require copy mode, not ctx");
    auto no_copy=wide32;no_copy.erase(std::find(no_copy.begin(),no_copy.end(),std::string("--mtp-ngram-draft=copy")));
    reject_as(no_copy,wide_requires,"wide rows require the copy drafter");
    auto wide_mode=wide32;for(auto& value:wide_mode)if(value=="--mtp-ngram-draft=copy")value="--mtp-ngram-draft=wide";
    const auto parsed_mode=parse(wide_mode).mtp;
    require(parsed_mode.ngram_wide_only && parsed_mode.copy_wide_rows==std::vector<std::uint32_t>{32U},"wide rows accept =wide");
    auto serial_prompt=request();serial_prompt.insert(serial_prompt.end(),{"--mtp-draft-chain","--mtp-target-window-rows","8",
        "--mtp-window-prefix","--mtp-window-capture-bytes","8192","--mtp-window-checkpoint-bytes","8192",
        "--mtp-window-total-bytes","32768","--mtp-window-arena-bytes","8192","--mtp-window-history-intervals","32",
        "--mtp-ngram-draft=copy","--mtp-copy-wide-rows","32"});
    set(serial_prompt,"--mtp-candidate-depth","7");
    reject_as(serial_prompt,wide_requires,"wide rows require the tiled MTP prompt");
    auto no_tail=wide32;no_tail.insert(no_tail.end(),{"--known-prefix-tail-rows","64,16"});
    reject_as(no_tail,"--mtp-copy-wide-rows requires a tiled tail block of each listed size "
        "(--known-prefix-tail-rows including it, below --known-prefix-prefill-rows)","explicit tails must include the wide rows");
    auto with_tail=wide32;with_tail.insert(with_tail.end(),{"--known-prefix-tail-rows","128,32"});
    require(parse(with_tail).known_prefix_tail_rows==std::vector<std::uint32_t>{128U,32U},"explicit tails including 32 are kept");
}

int main(){try{
    for (const auto* profile : {"default", "fast", "quality"}) {
        auto args=request(); args.insert(args.end(), {"--profile",profile,"--allow-pending"});
        const auto product=parse(args);
        const auto plain=parse(request());
        require(product.product_profile.profile==strix::qwen::profile::parse(profile) && product.mtp.enabled &&
            product.mtp.candidate_depth==plain.mtp.candidate_depth && product.mtp.sidecar_sha256==plain.mtp.sidecar_sha256 &&
            product.numerical_profile==plain.numerical_profile && product.prefill_numerical_profile==plain.prefill_numerical_profile &&
            product.any_low_level_flag==plain.any_low_level_flag && product.route_owned==plain.route_owned,
            "product profiles preserve the pinned MTP route and arithmetic");
        auto served=request(); served.insert(served.begin()+5,13,"1");
        served.insert(served.end(), {"--profile",profile,"--http-serve","--http-batch","0"});
        const auto http=parse(served);
        require(http.http_serve && http.mtp.enabled && http.product_profile.profile==product.product_profile.profile,
            "HTTP serving retains the selected product profile with its MTP route");
        args.insert(args.end(), {"--rope-yarn","2","--context","32768"});
        args.insert(args.begin()+5, {"4","5","6","7","8","9","10"});
        require(parse(args).mtp.enabled,"product selectors are admitted on the YaRN MTP route");
    }
    {
        auto args=request();args.insert(args.begin()+5,{"4","5","6","7","8","9","10"});args.insert(args.end(),{"--rope-yarn","2","--context","32768"});
        const auto ym=parse(args);require(ym.mtp.enabled && ym.rotary.factor()==2,"explicit YMTP parser admission");
        args[args.size()-3]="4";
        reject_as(args,"YaRN MTP requires f2 at C<=524288 and f4 at C<=1048576", "small capacity uses f2 MTP band");
    }
    const auto no_speculation = parse({"decoder","m","1","--generate","1","--known-prefix-window=0"});
    require(!no_speculation.mtp.enabled && !no_speculation.known_prefix_window,
        "automatic prefill off never infers MTP from environment-only tuning");
    campaign_defaults_contract();
    const auto legacy=parse({"decoder","target.gguf","1"});
    require(!legacy.mtp.enabled&&!legacy.mtp.supplied,"legacy CLI does not select MTP");
    auto good=parse(request());
    require(good.mtp.enabled&&good.mtp.candidate_depth==2U&&good.mtp.normalized_rounding=="fp32"&&
        good.mtp.projection_rounding=="bf16-rne"&&!good.any_low_level_flag,
        "MTP flags preserve automatic profile selection and explicit cast choices");
    // Parsing precedes automatic context resolution. A normal512 output request
    // must remain admissible until the actual resolved context is known.
    good.context_tokens=8192;strix::decode_cli::validate_resolved_mtp_cli(good);
    require(!good.mtp.draft_chain && good.mtp.draft_vocab.empty() && !good.mtp.draft_context_vocab,
        "ordinary MTP keeps the full-head per-draft path");
    auto chain=request();chain.push_back("--mtp-draft-chain");
    const auto full_chain=parse(chain);
    require(full_chain.mtp.draft_chain && full_chain.mtp.draft_vocab.empty() && !full_chain.mtp.draft_context_vocab,
        "draft chain alone keeps the full draft head");
    chain.insert(chain.end(),{"--mtp-draft-vocab","base.u32","--mtp-draft-context-vocab=4096"});
    const auto subset=parse(chain);
    require(subset.mtp.draft_vocab=="base.u32" && subset.mtp.draft_context_vocab==4096U,"draft subset options parse");
    reject_as([]{auto a=request();a.insert(a.end(),{"--mtp-draft-vocab","base.u32"});return a;}(),
        "MTP draft vocabulary options require --mtp-draft-chain","draft vocabulary requires the chain");
    reject([]{auto a=request();a.insert(a.end(),{"--mtp-draft-chain","--mtp-draft-context-vocab","0"});return a;}(),
        "zero draft context cap rejected");
    reject([]{auto a=request();a.insert(a.end(),{"--mtp-draft-chain","--mtp-draft-chain"});return a;}(),"duplicate draft chain rejected");
    reject([]{auto a=request();a.push_back("--mtp-draft-chain=1");return a;}(),"draft chain takes no value");
    reject({"decoder","target.gguf","1","--mtp-draft-chain"},"draft chain cannot select MTP");
    require(!subset.mtp.draft_exact && !subset.mtp.draft_q4,"draft chain defaults to fast kernels and the Q8 head");
    chain.insert(chain.end(),{"--mtp-draft-exact","--mtp-draft-q4"});
    const auto variants=parse(chain);
    require(variants.mtp.draft_exact && variants.mtp.draft_q4,"draft exact/Q4 flags parse");
    reject_as([]{auto a=request();a.push_back("--mtp-draft-exact");return a;}(),
        "MTP draft vocabulary options require --mtp-draft-chain","draft exact requires the chain");
    reject_as([]{auto a=request();a.insert(a.end(),{"--mtp-draft-chain","--mtp-draft-q4"});return a;}(),
        "MTP draft Q4 head requires a draft vocabulary subset","Q4 head requires a subset");
    reject([]{auto a=request();a.insert(a.end(),{"--mtp-draft-chain","--mtp-draft-exact","--mtp-draft-exact"});return a;}(),"duplicate draft exact rejected");
    auto metadata=request();
    for(const auto* key:{"--mtp-golden","--mtp-golden-sha256"}){
        for(auto it=metadata.begin();it!=metadata.end();++it)if(*it==key){metadata.erase(it,it+2);break;}
    }
    reject(metadata,"ordinary MTP still requires independent golden pins");
    metadata.insert(metadata.end(),{"--mtp-metadata-only","1"});
    const auto probe=parse(metadata);
    require(probe.mtp.metadata_only && probe.mtp.golden.empty() && probe.mtp.golden_sha256.empty(),
        "explicit metadata diagnostic admits real assets without an expected-output file");
    for(const auto* unused:{"--mtp-golden","--mtp-golden-sha256"}){
        auto bad=metadata;bad.insert(bad.end(),{unused,unused==std::string_view("--mtp-golden")?"unused.bin":std::string(64,'a')});
        bool exact=false;try{(void)parse(bad);}catch(const std::invalid_argument& e){
            exact=std::string(e.what())=="MTP metadata diagnostic does not accept unused golden options";}
        require(exact,"metadata diagnostic rejects unused golden parameters explicitly");
    }
    set(metadata,"--mtp-metadata-only","0");reject(metadata,"metadata diagnostic requires its explicit value");
    auto dump=request();
    for(const auto* key:{"--mtp-golden","--mtp-golden-sha256"}){
        for(auto it=dump.begin();it!=dump.end();++it)if(*it==key){dump.erase(it,it+2);break;}
    }
    dump.insert(dump.end(),{"--mtp-golden-dump","self.golden"});
    const auto dumped=parse(dump);
    require(dumped.mtp.golden_dump=="self.golden" && dumped.mtp.golden.empty() && !dumped.mtp.metadata_only,
        "golden dump replaces the input golden pins");
    reject_as([&]{auto a=dump;a.insert(a.end(),{"--mtp-golden","in.bin"});return a;}(),
        "MTP golden dump writes a golden and does not accept an input golden","golden dump rejects an input golden");
    reject_as([&]{auto a=dump;a.insert(a.end(),{"--mtp-metadata-only","1"});return a;}(),
        "MTP metadata diagnostic does not accept unused golden options","golden dump excludes metadata diagnostic");
    reject([&]{auto a=request();a.insert(a.end(),{"--mtp-golden-dump","self.golden"});return a;}(),"dump with full golden pins rejected");
    for(const auto* flag:{"--prepared-eager","--moe-shared-overlap","--gdn-islands","--gpu-greedy-token","--prompt-lookup",
        "--prefix-cache-roundtrip","--device-queue-trace"}){
        auto bad=request();bad.push_back(flag);reject(bad,"incompatible MTP route rejects at parsing before native setup");
    }
    for(unsigned fault=0;fault<13U;++fault){auto bad=request();
        if(fault==0U)set(bad,"--mtp-sidecar-sha256",std::string(64,'0'));
        if(fault==1U)set(bad,"--mtp-golden-sha256","abc");
        if(fault==2U)set(bad,"--mtp-target-content",std::string(64,'g'));
        if(fault==3U)set(bad,"--mtp-runtime-bytes","0");
        if(fault==4U)set(bad,"--mtp-header-bytes","33554433");
        if(fault==5U)set(bad,"--mtp-candidate-depth","9");
        if(fault==6U)set(bad,"--mtp-normalized-rounding","implicit");
        if(fault==7U)set(bad,"--mtp-conversion","raw-norm");
        if(fault==8U){bad.push_back("--mtp-policy-id=8");}
        if(fault==9U){bad.insert(bad.end(),{"--context","3"});}
        if(fault==10U){bad.push_back("--mtp-target-window-rows=2");}
        if(fault==11U)set(bad,"--mtp-policy-id","0");
        if(fault==12U){bad.insert(bad.end(),{"--mtp-max-logit-error","-1"});}
        reject(bad,"MTP source pins, positive budgets, casts, bounds and duplicate options fail closed");
    }
    reject({"decoder","target.gguf","1","--mtp-runtime-bytes","1"},"MTP parameters do not silently select execution");
    auto wide=request();wide.insert(wide.end(),{"--mtp-target-window-rows=2","--mtp-window-arena-bytes=1024",
        "--mtp-window-capture-bytes=1024","--mtp-window-checkpoint-bytes=1024","--mtp-window-total-bytes=8192",
        "--mtp-window-history-intervals=128"});
    // B2: configured windows must be reachable with the admitted draft depth.
    // Check exact admission cause, not any unrelated missing-budget rejection.
    for(const auto rows:{2U,4U,8U}) {
        auto minimum=wide;
        set(minimum,"--mtp-target-window-rows",std::to_string(rows));
        set(minimum,"--mtp-candidate-depth",std::to_string(rows-1U));
        const auto admitted=parse(minimum);
        require(admitted.mtp.window_rows==rows && admitted.mtp.candidate_depth==rows-1U,
            "each closed MTP target window admits its exact minimum draft depth");
        if(rows>2U) {
            auto insufficient=minimum;
            set(insufficient,"--mtp-candidate-depth",std::to_string(rows-2U));
            bool depth_rejected=false;
            try{(void)parse(insufficient);}catch(const std::invalid_argument& e){
                depth_rejected=std::string(e.what())=="MTP target window requires candidate depth at least rows minus one";
            }
            require(depth_rejected,"insufficient draft depth rejects specifically before reserving any target window");
        }
    }
    auto unreachable=wide;set(unreachable,"--mtp-target-window-rows","8");
    bool unreachable_rejected=false;
    try{(void)parse(unreachable);}catch(const std::invalid_argument& e){
        unreachable_rejected=std::string(e.what())=="MTP target window requires candidate depth at least rows minus one";
    }
    require(unreachable_rejected,"depth2 cannot configure an unreachable eight-row target window");
    require(strix::decode_cli::mtp_target_verification_name(0U)=="serial-target-fallback" &&
        strix::decode_cli::mtp_target_verification_name(1U)=="closed-known-window-attempted-with-serial-fallback" &&
        strix::decode_cli::mtp_target_verification_name(7U)==strix::decode_cli::mtp_target_verification_name(1U),
        "receipt verification mode follows actual attempt evidence, never owner configuration alone");
    auto selected=parse(wide);selected.context_tokens=8192;
    bool failed=false;try{strix::decode_cli::validate_resolved_mtp_cli(selected);}catch(const std::invalid_argument&){failed=true;}
    require(failed,"MTP target windows cannot silently enable D1 or paired GDN optimizations");
    selected.deferred_moe_status=selected.virtual_q8_gdn_input=selected.prefetch_q8_gdn_input=selected.virtual_q8_gdn_output=
        selected.wave8_q8_gdn_input=selected.wave8_q8_gdn_output=true;
    strix::decode_cli::validate_resolved_mtp_cli(selected);
    // The union is a separate opt-in target-window consumer, never an implicit
    // prerequisite-completing optimization or a known-prefix alias.
    require(!legacy.mtp.window_expert_union && !legacy.mtp.window_expert_union_bytes_explicit &&
        legacy.mtp.window_expert_union_bytes==0 && !good.mtp.window_expert_union,
        "legacy and ordinary MTP requests do not acquire union workspace");
    auto union_request=wide;
    union_request.insert(union_request.end(),{"--mtp-window-expert-union","--mtp-window-expert-union-bytes","512"});
    for(const auto rows:{2U,4U,8U}) {
        auto args=union_request;
        set(args,"--mtp-target-window-rows",std::to_string(rows));
        set(args,"--mtp-candidate-depth",std::to_string(rows-1U));
        auto options=parse(args);
        require(options.mtp.window_expert_union && options.mtp.window_expert_union_bytes_explicit &&
            options.mtp.window_expert_union_bytes==512 && options.mtp.window_rows==rows &&
            !options.any_low_level_flag && !options.known_prefix_expert_union &&
            !options.q4k_a8g32_gate_up && !options.grouped_exact_q51_down && !options.q51_slot_parallel_down &&
            !options.deferred_moe_status,
            "real parser preserves explicit union budget and leaves all execution prerequisites unchanged");
        options.context_tokens=8192;
        options.deferred_moe_status=options.virtual_q8_gdn_input=options.prefetch_q8_gdn_input=
            options.virtual_q8_gdn_output=options.wave8_q8_gdn_input=options.wave8_q8_gdn_output=true;
        bool missing_prerequisites=false;
        try{strix::decode_cli::validate_resolved_mtp_cli(options);}catch(const std::invalid_argument& e){
            missing_prerequisites=std::string(e.what())==
                "MTP window expert union requires resolved A8, grouped Q5_1, slot-parallel down and D1";
        }
        require(missing_prerequisites,"resolved union rejects absent A8/grouped/slot prerequisites specifically");
        options.q4k_a8g32_gate_up=options.grouped_exact_q51_down=options.q51_slot_parallel_down=true;
        strix::decode_cli::validate_resolved_mtp_cli(options);
        for(unsigned fault=0;fault<4U;++fault) {
            auto bad=options;
            if(fault==0U)bad.q4k_a8g32_gate_up=false;
            if(fault==1U)bad.grouped_exact_q51_down=false;
            if(fault==2U)bad.q51_slot_parallel_down=false;
            if(fault==3U)bad.deferred_moe_status=false;
            bool rejected=false;
            try{strix::decode_cli::validate_resolved_mtp_cli(bad);}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"each missing execution prerequisite rejects without completing or disabling flags");
        }
    }
    auto equals_budget=wide;
    equals_budget.insert(equals_budget.end(),{"--mtp-window-expert-union","--mtp-window-expert-union-bytes=1024"});
    require(parse(equals_budget).mtp.window_expert_union_bytes==1024,
        "explicit union budget accepts equals syntax and inclusive existing arena ceiling");
    auto no_budget=wide;no_budget.push_back("--mtp-window-expert-union");
    reject_as(no_budget,"MTP window expert union requires an actual target window and a positive explicit budget within arena and total window budgets","union flag requires an explicit positive budget");
    auto no_flag=wide;no_flag.push_back("--mtp-window-expert-union-bytes=512");
    reject_as(no_flag,"MTP window expert union requires an actual target window and a positive explicit budget within arena and total window budgets","unused union budget cannot silently select a consumer");
    auto no_window=request();no_window.insert(no_window.end(),{"--mtp-window-expert-union","--mtp-window-expert-union-bytes=512"});
    reject_as(no_window,"MTP window expert union requires an actual target window and a positive explicit budget within arena and total window budgets","union requires an actual admitted target window");
    auto no_mtp=union_request;
    for(auto it=no_mtp.begin();it!=no_mtp.end();++it)if(*it=="--mtp"){no_mtp.erase(it);break;}
    reject(no_mtp,"union parameters cannot silently enable MTP");
    for(unsigned fault=0;fault<8U;++fault) {
        auto bad=union_request;
        if(fault==0U)set(bad,"--mtp-window-expert-union-bytes","0");
        if(fault==1U)set(bad,"--mtp-window-expert-union-bytes","1025");
        if(fault==2U)set(bad,"--mtp-window-total-bytes","256");
        if(fault==3U)set(bad,"--mtp-window-expert-union-bytes","18446744073709551616");
        if(fault==4U)bad.push_back("--mtp-window-expert-union");
        if(fault==5U)bad.push_back("--mtp-window-expert-union-bytes=512");
        if(fault==6U)bad.push_back("--mtp-window-expert-union=1");
        if(fault==7U)bad.push_back("--mtp-window-expert-union-bytes");
        reject(bad,"union zero/over-budget/overflow/duplicate/invalid flag/missing value rejects before setup");
    }
    auto metadata_union=metadata;
    set(metadata_union,"--mtp-metadata-only","1");
    metadata_union.insert(metadata_union.end(),{"--mtp-target-window-rows=2","--mtp-window-arena-bytes=1024",
        "--mtp-window-capture-bytes=1024","--mtp-window-checkpoint-bytes=1024","--mtp-window-total-bytes=8192",
        "--mtp-window-history-intervals=128","--mtp-window-expert-union","--mtp-window-expert-union-bytes=512"});
    reject_as(metadata_union,"MTP window expert union requires an actual target window and a positive explicit budget within arena and total window budgets","metadata-only route cannot accept unused union workspace");
    for (const auto* profile : {"q51-a8-down-v1",
        "hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1",
        "hc-deferred-norm-f32+q51-q80-a8-down-v1",
        "hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1"}) {
        auto numerical_mtp = request();
        numerical_mtp.insert(numerical_mtp.end(), {"--numerical-profile", profile});
        reject_as(numerical_mtp,
            "numerical profile requires ordinary eager floating C1 execution without alternate controllers or layouts",
            "MTP cannot inherit an unqualified target arithmetic atom");
    }
    auto exact_mtp = request();
    exact_mtp.insert(exact_mtp.end(), {"--numerical-profile=exact-v5"});
    require(parse(exact_mtp).numerical_profile == strix::DecodeNumericalProfile::exact_v5,
        "explicit default arithmetic leaves the existing MTP controller contract unchanged");
    auto selective_mtp = request();
    selective_mtp.insert(selective_mtp.end(), {"--model-copy-tier=normalization",
        "--model-copy-payload-bytes=3997696", "--model-retained-budget=120000000000",
        "--model-setup-peak-budget=120000000000"});
    reject_as(selective_mtp, "selective model copy excludes MTP options",
        "selective target backing cannot implicitly qualify sidecar residency");
    auto teacher_mtp = request();
    set(teacher_mtp, "--generate", "1");
    teacher_mtp.push_back("--teacher-forced-score");
    reject_as(teacher_mtp,
        "teacher-forced score requires at least two teacher inputs and generate 1 in serial eager full-output diagnostic mode",
        "teacher score never enters speculative controller execution");
    {
        auto baseline = experts_request();
        require(!parse(baseline).staged_moe_router_logits,"staged logits defaults off");
        baseline.push_back("--staged-moe-router-logits");
        reject_as(baseline,"staged router logits requires parallel projection","staged logits requires explicit parallel topology");
        baseline.push_back("--parallel-moe-router");
        require(parse(baseline).staged_moe_router_logits,"staged logits explicit opt-in round trip");
        auto mtp = request(); mtp.push_back("--parallel-moe-router"); mtp.push_back("--staged-moe-router-logits");
        reject_as(mtp,"staged router logits excludes MTP","staged logits never silently enters the direct MTP path");
        baseline.push_back("--staged-moe-router-logits");
        reject_as(baseline,"duplicate staged MoE router logits option","duplicate staged logits rejected");
    }
    {
        auto args=request();require(parse(args).http_selection_mtp,"selection MTP defaults on");
        args.insert(args.end(),{"--http-selection-mtp","off"});require(!parse(args).http_selection_mtp,"selection MTP off");
        args.back()="on";require(parse(args).http_selection_mtp,"selection MTP on");
        args.back()="invalid";reject_as(args,"--http-selection-mtp requires on|off","selection MTP invalid value");
        args.pop_back();reject_as(args,"--http-selection-mtp requires a value","selection MTP missing value");
    }
    {
        auto args=request();require(parse(args).http_selection_fast,"argmax-first selection defaults on");
        args.insert(args.end(),{"--http-selection-fast","off"});require(!parse(args).http_selection_fast,"argmax-first selection off");
        args.back()="invalid";reject_as(args,"--http-selection-fast requires on|off","argmax-first invalid value");
    }
    {
        auto args=request(); args.insert(args.begin()+5,{"4","5","6","7","8","9","10"}); args.insert(args.end(), {"--rtx-state-import-file","state.json","--rope-yarn","2","--context","524288"});
        const auto imported=parse(args);
        require(imported.rtx_state_import && imported.rtx_state_import_file=="state.json", "YaRN MTP CLI file import admitted");
        args.insert(args.end(), {"--http-serve"});
        reject(args,"file import refuses HTTP");
        args.pop_back(); args.insert(args.end(), {"--http-batch","0"});
        reject(args,"file import refuses HTTP batch");
        const auto native=parse({"decoder","target.gguf","1","2","--generate","1","--rtx-state-import-file","state.json"});
        require(native.rtx_state_import && !native.route_owned, "native import preserves automatic prefill routing");
        reject_as({"decoder","target.gguf","1","--rtx-state-import-file","state.json","--http-port","8080"},
            "--rtx-state-import-file refuses HTTP-only features", "file import refuses HTTP-only configuration");
        const auto serial=parse({"decoder","target.gguf","1","2","--generate","1","--rope-yarn","2","--context","524288",
            "--rtx-state-import-file","state.json"});
        require(serial.rtx_state_import && !serial.mtp.enabled, "YaRN serial CLI file import admitted");
    }
    expert_copy_cli_contract();
    depth_policy_options_contract();
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
