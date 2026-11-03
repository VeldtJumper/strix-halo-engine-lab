#include "../apps/qwen_effective_config.hpp"
#include <iostream>

using namespace strix;
using namespace strix::decode_cli;
namespace {
void require(bool v,const char* why) {if(!v)throw std::runtime_error(why);}
void env(const char* key,const std::string& value) {
#ifdef _WIN32
    _putenv_s(key,value.c_str());
#else
    if(value.empty())unsetenv(key);else setenv(key,value.c_str(),1);
#endif
}
Options parse(std::vector<std::string> args) {
    std::vector<char*> pointers;
    for(auto& arg:args)pointers.push_back(arg.data());
    return parse_options(static_cast<int>(pointers.size()),pointers.data());
}
template<class F> void refuses(F f) {
    bool threw=false;try{f();}catch(const std::exception&){threw=true;}
    require(threw,"expected refusal");
}
void write(const std::filesystem::path& p,const std::string& text) {
    std::ofstream f(p,std::ios::binary);f<<text;
    if(!f)throw std::runtime_error("fixture write failed");
}
}
int main(int argc,char** argv) try {
    require(argc==2,"recipe argument");
    const auto dir=std::filesystem::temp_directory_path()/("strix-mtp-bundle-test-"+
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    struct Cleanup {std::filesystem::path p;~Cleanup(){std::error_code ec;std::filesystem::remove_all(p,ec);}} cleanup{dir};
    env("STRIX_HASH_CACHE",(dir/"hash-cache").string());
    env("STRIX_QWEN_YARN_MTP","");env("STRIX_QWEN_ROPE_YARN","");
    std::ifstream recipe(argv[1]);
    auto json=serve::parse_json_unique(std::string((std::istreambuf_iterator<char>(recipe)),{}));
    auto opts=*json.find("options");
    // Same shipped options, with a tiny byte-pinned sidecar for host-only testing.
    const auto sidecar=dir/"sidecar.gguf";write(sidecar,"abc");
    const std::string sha="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    opts.set("--mtp-sidecar","sidecar.gguf");opts.set("--mtp-sidecar-sha256",sha);
    opts.set("--mtp-sidecar-bytes","3");
    opts.set("--mtp-policy-id",1);
    json.set("options",opts);
    const auto bundle=dir/"target.gguf.mtp-bundle.json";write(bundle,json.dump());
    std::vector<std::string> base{"decode",(dir/"target.gguf").string(),"1","2","3","4","5","6","7","8","9",
        "10","11","12","13","14","15","16","17","18","--rope-yarn","4","--context","1048576","--generate","256"};
    const auto bundled=parse(base);
    require(bundled.mtp.enabled && bundled.mtp_bundle_path==bundle.string(),"automatic discovery");
    require(bundled.yarn_decode_source==bundle.string(),"bundle source");
    auto import_args=base;
    import_args.insert(import_args.end(),{"--rtx-state-import-file","prefix.json"});
    const auto imported=parse(import_args);
    require(imported.mtp.enabled && imported.mtp_bundle_path==bundle.string() && imported.rtx_state_import_file=="prefix.json",
        "CLI file import preserves bundle admission");

    EffectiveConfig inspected;inspected.options=bundled;inspected.admitted=true;
    auto receipt=serve::parse_json_unique(effective_config_json(inspected));
    require(receipt.find("yarn_decode_route")->find("source")->as_string()==bundle.string(),"effective bundle source");
    require(receipt.find("yarn_mtp_assets")->find("golden")->as_string()=="pending-self-generation","cold bundle receipt");
    auto selected=bundled;
    const auto model=ModelContract::qwen38_flash_next();
    env("XDG_CACHE_HOME",(dir/"admission-cache").string());
    select_mtp_bundle_golden(selected,model);
    require(selected.mtp.golden.empty(),"cold admission leaves golden pending");
    const auto admission_id=mtp_bundle_golden_identity(selected,model);
    const auto admission_file=mtp_golden_cache_dir()/(admission_id+".golden");
    std::filesystem::create_directories(admission_file.parent_path());write(admission_file,"cached");
    select_mtp_bundle_golden(selected,model);
    require(selected.mtp.golden==admission_file.string(),"admission selects cached exact identity");
    ++selected.context_tokens;select_mtp_bundle_golden(selected,model);
    require(selected.mtp.golden.empty(),"changed geometry misses cache");
    env("XDG_CACHE_HOME","");
    auto explicit_args=base;
    explicit_args.push_back("--mtp");
    for(const auto& [key,value]:opts.as_object()) {
        explicit_args.push_back(key);
        if(value.is_bool())continue;
        auto v=value.is_string()?value.as_string():value.number_source();
        if(key=="--mtp-sidecar")v=(dir/v).string();
        if(key=="--mtp-draft-vocab")v=v.starts_with("auto:")?"auto:"+(dir/v.substr(5)).lexically_normal().string():(dir/v).string();
        explicit_args.push_back(v);
    }
    explicit_args.insert(explicit_args.end(),{"--mtp-golden","reference.golden","--mtp-golden-sha256",sha});
    const auto explicit_options=parse(explicit_args);
    inspected.options=explicit_options;
    receipt=serve::parse_json_unique(effective_config_json(inspected));
    require(receipt.find("yarn_decode_route")->find("source")->as_string()=="explicit","effective explicit source");
#define SAME(field) require(bundled.mtp.field==explicit_options.mtp.field,"bundle/explicit " #field)
    SAME(sidecar);SAME(sidecar_sha256);SAME(sidecar_logical_name);SAME(sidecar_bytes);
    SAME(target_sha256);SAME(target_content);SAME(repository);SAME(revision);SAME(source_path);
    SAME(converter_revision);SAME(converter_sha256);SAME(norm_converter_sha256);SAME(conversion);
    SAME(normalized_rounding);SAME(projection_rounding);SAME(sidecar_residency);
    SAME(header_bytes);SAME(sidecar_retained_bytes);SAME(sidecar_setup_bytes);SAME(golden_max_bytes);
    SAME(replay_host_bytes);SAME(runtime_bytes);SAME(target_capture_bytes);SAME(policy_identity);
    SAME(max_hidden_error);SAME(max_logit_error);SAME(spec_min_p);SAME(calibration_rows);SAME(draft_chain);
    SAME(draft_vocab);SAME(draft_context_vocab);
    SAME(draft_q4);SAME(candidate_depth);SAME(window_rows);SAME(window_rows_set);SAME(window_prefix);
    SAME(window_hyper_mix);SAME(window_qsa_rows);SAME(window_expert_union);SAME(window_expert_union_bytes);
    SAME(window_arena_bytes);SAME(window_capture_bytes);SAME(window_checkpoint_bytes);SAME(window_total_bytes);
    SAME(window_history_intervals);
#undef SAME
    const auto stats=file_hash_cache_stats();verify_mtp_bundle_sidecar(sidecar.string(),sha,3);
    verify_mtp_bundle_sidecar(sidecar.string(),"BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",3);
#if defined(__linux__)
    require(file_hash_cache_stats().hits>stats.hits,"persistent hash cache used");
#else
    (void)stats;
#endif
    auto explicit_bundle=base;explicit_bundle.insert(explicit_bundle.end(),{"--mtp-bundle",bundle.string()});
    require(parse(explicit_bundle).mtp.enabled,"explicit bundle");
    explicit_bundle.push_back("--mtp");refuses([&]{parse(explicit_bundle);});
    explicit_bundle.pop_back();explicit_bundle.insert(explicit_bundle.end(),{"--mtp-policy-id","1"});
    refuses([&]{parse(explicit_bundle);});
    env("STRIX_QWEN_YARN_MTP","0");
    inspected.options=parse(base);
    require(!inspected.options.mtp.enabled,"env zero serial");
    receipt=serve::parse_json_unique(effective_config_json(inspected));
    require(receipt.find("yarn_decode_route")->find("source")->as_string()=="env","effective env source");
    auto missing_bundle=base;missing_bundle.insert(missing_bundle.end(),{"--mtp-bundle",(dir/"missing.json").string()});
    require(!parse(missing_bundle).mtp.enabled,"env zero bypasses bundle assets");
    env("STRIX_QWEN_YARN_MTP","1");require(parse(base).mtp.enabled,"env one bundle");
    std::filesystem::remove(bundle);refuses([&]{parse(base);});
    env("STRIX_QWEN_YARN_MTP","");
    std::ostringstream notice;auto* old=std::cerr.rdbuf(notice.rdbuf());
    const auto serial=parse(base);std::cerr.rdbuf(old);
    require(!serial.mtp.enabled && notice.str()=="YaRN: no MTP bundle found; using serial decode\n","no bundle serial notice once");
    write(bundle,json.dump());
    {
        YarnBundleSerialRetry retry;
        require(!parse(base).mtp.enabled,"dump failure retry suppresses automatic MTP");
    }
    require(!yarn_bundle_serial_retry,"retry scope restores resolver");
    std::vector<std::string> native{"decode",(dir/"target.gguf").string(),"1","2","3","--generate","1"};
    require(!parse(native).mtp.enabled,"native ignores adjacent bundle");
    native.insert(native.end(),{"--rope-yarn","off"});
    require(!parse(native).mtp.enabled,"explicit rotary off ignores bundle");
    native.pop_back();native.push_back("4");native.push_back("--baseline");
    require(!parse(native).mtp.enabled,"baseline ignores automatic bundle");
    write(sidecar,"bad");refuses([&]{parse(base);});
    const auto cache=dir/"cache";int dumps=0;
    const auto fake=[&](const std::filesystem::path& temp){
        ++dumps;require(temp.extension().string().find("tmp-")!=std::string::npos,"temporary dump");
        require(!std::filesystem::exists(cache/"id.golden"),"not published during dump");write(temp,"v3");};
    const auto cached=ensure_mtp_bundle_golden(bundle,"id",fake,[&]{return cache;});
    require(dumps==1 && cached==cache/"id.golden","cache miss publishes");
    require(ensure_mtp_bundle_golden(bundle,"id",fake,[&]{return cache;})==cached && dumps==1,"cache hit");
    std::filesystem::create_directories(dir/"goldens");write(dir/"goldens/id.golden","shipped");
    require(ensure_mtp_bundle_golden(bundle,"id",fake,[&]{return cache;})==dir/"goldens/id.golden","shipped priority");
    refuses([&]{ensure_mtp_bundle_golden(bundle,"failed",[&](const auto& temp){write(temp,"partial");throw std::runtime_error("fake dump failure");},[&]{return cache;});});
    require(!std::filesystem::exists(cache/"failed.golden"),"failed dump not published");
    for(const auto& entry:std::filesystem::directory_iterator(cache))require(entry.path()==cached,"temporary cleaned");
    refuses([&]{ensure_mtp_bundle_golden(bundle,"rename-failure",[&](const auto& temp){
        write(temp,"v3");std::filesystem::create_directory(cache/"rename-failure.golden");
    },[&]{return cache;});});
    for(const auto& entry:std::filesystem::directory_iterator(cache))
        require(entry.path().filename().string().find(".tmp-")==std::string::npos,"rename failure cleans temp");
    write(dir/"blocked","file");
    const auto serial_on_failure=[&](const std::function<void()>& attempt) {
        bool fell_back=false;
        try {attempt();}
        catch(const std::exception& error) {
            env("STRIX_QWEN_YARN_MTP","1");
            const auto status=retry_mtp_bundle_serial(error,[&] {
                fell_back=true;require(!parse(base).mtp.enabled,"failed calibration uses fresh serial route");return 0;
            });
            require(status==0,"serial fallback succeeds");env("STRIX_QWEN_YARN_MTP","");
        }
        require(fell_back && !yarn_bundle_serial_retry,"failure fallback and scope reset");
    };
    serial_on_failure([&]{ensure_mtp_bundle_golden(bundle,"blocked",fake,[&]{return dir/"blocked/subdir";});});
    serial_on_failure([&]{ensure_mtp_bundle_golden(bundle,"dump-failed",[](const auto&){throw std::runtime_error("fake dump");},[&]{return cache;});});
    // Admission, preflight and bridge setup use the same retry seam. Fresh serial
    // resolution must see the caller's environment, not the failed MTP defaults.
    write(sidecar,"abc");
    const MtpBundleSetupRetry identity_test_state;
    detail::prefill_identity_override()={};
    env("STRIX_PREFILL_EXACT_FP16","");env("STRIX_PREFILL_HC_OUTPUT_WMMA","1");
    const auto resolve_identity=[&](const Options& options) {
        return resolve_effective_config({options,requested_prefill_environment()},model,{}, {},"",{},{});
    };
    const auto fresh_serial=[&] {
        YarnBundleSerialRetry retry;
        return resolve_identity(parse(base));
    }();
    require(fresh_serial.prefill_option_mask==kPrefillOptionsReady,"fresh serial retains READY arithmetic");
    for(const auto* failure:{"admission refusal","preflight failure","bridge setup failure","golden dump failure"}) {
        env("STRIX_PREFILL_PIPE","0");env("STRIX_QSA_SCORE_RING_ROWS","");
        MtpBundleSetupRetry state;state.selected=true;
        auto attempt=parse(base);
        require(attempt.mtp.enabled,"automatic YaRN bundle selected before setup");
        set_prefill_identity_override(attempt.prefill_exact_fp16,attempt.prefill_disabled);
        const auto resolved=resolve_identity(attempt);
        set_prefill_identity_override(resolved.options.prefill_exact_fp16,resolved.options.prefill_disabled);
        require(resolved.prefill_option_mask==0U && prefill_options_active()==0U,
            "automatic YaRN MTP resolution installs runtime arithmetic restrictions");
        env("STRIX_PREFILL_PIPE","1");env("STRIX_QSA_SCORE_RING_ROWS","64");
        const auto previous_defaulted=qpf2k::defaulted_items;
        qpf2k::defaulted_items=~previous_defaulted;
        std::ostringstream notices;auto* saved=std::cerr.rdbuf(notices.rdbuf());
        unsigned attempts=0;
        const std::runtime_error error(failure);
        const auto status=retry_mtp_bundle_setup(state,error,[&] {
            ++attempts;
            require(std::string(std::getenv("STRIX_PREFILL_PIPE"))=="0" && !std::getenv("STRIX_QSA_SCORE_RING_ROWS"),"retry restores requested environment");
            require(qpf2k::defaulted_items==previous_defaulted,"retry restores default provenance");
            require(!parse(base).mtp.enabled,"bundle failure retries serial even with adjacent bundle");
            const auto serial_config=resolve_identity(parse(base));
            set_prefill_identity_override(serial_config.options.prefill_exact_fp16,serial_config.options.prefill_disabled);
            require(detail::prefill_identity_override().disabled==fresh_serial.options.prefill_disabled &&
                detail::prefill_identity_override().exact==fresh_serial.options.prefill_exact_fp16 &&
                prefill_options_active()==fresh_serial.prefill_option_mask,
                "failed setup restores the fresh serial override and runtime mask");
            require(serial_config.prefill_option_mask==prefill_options_active() &&
                serial_config.options.resolved_prefill_option_mask==prefill_options_active() &&
                effective_config_json(serial_config).find(prefill_identity_fields(prefill_options_active()))!=std::string::npos,
                "serial retry receipt and executed arithmetic declare the same identity");
            require(!retry_mtp_bundle_setup(state,error,[&] {++attempts;return 1;}),"serial failure cannot recurse");
            return 17;
        });
        std::cerr.rdbuf(saved);
        require(status && *status==17 && attempts==1,"one serial retry returns its status");
        const auto notice_text=notices.str();
        require(notice_text.starts_with("YaRN MTP bundle setup unavailable; using serial decode: ") &&
            std::count(notice_text.begin(),notice_text.end(),'\n')==1,"one fallback notice");
        state.target_mutated=true;
        require(!retry_mtp_bundle_setup(state,error,[]{return 1;}),"post-mutation error keeps refusal");
        state.target_mutated=false;state.selected=false;
        require(!retry_mtp_bundle_setup(state,error,[]{return 1;}),"explicit MTP keeps refusal");
    }
    env("STRIX_PREFILL_PIPE","");
    identity_test_state.restore();
    // Bundle parsing/setup errors retain their retry classification. Explicit
    // argv errors remain ordinary refusals and never acquire bundle selection.
    write(sidecar,"bad");
    bool bundle_failure=false;
    try {(void)parse(base);}catch(const YarnBundleCalibrationFailure&){bundle_failure=true;}
    require(bundle_failure,"bundle asset setup failure enters serial retry");
    bool explicit_failure=false;
    auto invalid_explicit=explicit_args;
    const auto pin_option=std::find(invalid_explicit.begin(),invalid_explicit.end(),"--mtp-sidecar-sha256");
    require(pin_option!=invalid_explicit.end(),"explicit fixture has sidecar pin");
    *(pin_option+1)="invalid";
    try {(void)parse(invalid_explicit);}catch(const YarnBundleCalibrationFailure&){throw;}
    catch(const std::exception&){explicit_failure=true;}
    require(explicit_failure,"explicit MTP asset refusal stays explicit");
    env("XDG_CACHE_HOME",(dir/"xdg").string());require(mtp_golden_cache_dir()==dir/"xdg/strix-engine/goldens","XDG path");
    env("XDG_CACHE_HOME","");env("HOME",dir.string());require(mtp_golden_cache_dir()==dir/".cache/strix-engine/goldens","HOME path");
    write(bundle,"{\"version\":1,\"version\":1,\"options\":{}}");refuses([&]{read_mtp_bundle(bundle);});
    opts.set("--mtp-sidecar","/absolute/sidecar.gguf");json.set("options",opts);write(bundle,json.dump());
    refuses([&]{read_mtp_bundle(bundle);});
    write(bundle,"{\"version\":1,\"options\":{}}");refuses([&]{read_mtp_bundle(bundle);});
    std::cout<<"MTP bundle CPU contracts passed\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
