#pragma once
#include "qwen_yarn_memory.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/prefill_identity.hpp"
#include "strix/qpf2k_defaults.hpp"
#include "strix/ple_discard_safety.hpp"
#include "strix/registered_backing_lifetime.hpp"
#include <map>
#include <optional>
#include "strix/serve/json.hpp"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace strix::decode_cli {
// Only a failed bundle setup retry may bypass bundle discovery.
inline thread_local bool yarn_bundle_serial_retry = false;
class YarnBundleCalibrationFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
// Keep golden-generation classification shared with the host failure-path tests.
template<class Work> void with_mtp_bundle_golden_errors(Work&& work) {
    try { std::forward<Work>(work)(); }
    catch (const YarnMemoryPressure&) { throw; }
    catch (const ple_reclaim::DiscardSafetyError&) { throw; }
    catch (const RegisteredPageLockError&) { throw; }
    catch (const std::exception& error) {
        throw YarnBundleCalibrationFailure(error.what());
    }
}
class YarnBundleSerialRetry {
public:
    YarnBundleSerialRetry() : previous_(yarn_bundle_serial_retry) {yarn_bundle_serial_retry=true;}
    ~YarnBundleSerialRetry() {yarn_bundle_serial_retry=previous_;}
    YarnBundleSerialRetry(const YarnBundleSerialRetry&)=delete;
    YarnBundleSerialRetry& operator=(const YarnBundleSerialRetry&)=delete;
private:
    bool previous_;
};
inline int retry_mtp_bundle_serial(const std::exception& error,const std::function<int()>& run) {
    std::cerr << "YaRN MTP bundle setup unavailable; using serial decode: " << error.what() << '\n';
    YarnBundleSerialRetry retry;
    return run();
}
// Lives across the entry function's unwind so provisional owners are gone before
// retry. Restore resolver-installed settings as well as their provenance.
class MtpBundleSetupRetry {
public:
    bool selected=false;
    bool target_mutated=false;
    MtpBundleSetupRetry() {
#if defined(__linux__)
        for (auto entry=::environ; entry && *entry; ++entry) {
#elif defined(_WIN32)
        for (auto entry=::_environ; entry && *entry; ++entry) {
#endif
#if defined(__linux__) || defined(_WIN32)
            const std::string text=*entry;
            if (text.starts_with("STRIX_")) {
                const auto equal=text.find('=');
                if(equal!=std::string::npos) environment_.emplace(text.substr(0,equal),text.substr(equal+1U));
            }
        }
#endif
    }
    void restore() const {
        // Collect names before erasing: setenv/unsetenv can invalidate environ.
        std::vector<std::string> names;
#if defined(__linux__)
        for(auto entry=::environ;entry && *entry;++entry) {
#elif defined(_WIN32)
        for(auto entry=::_environ;entry && *entry;++entry) {
#endif
#if defined(__linux__) || defined(_WIN32)
            const std::string text=*entry;
            if(text.starts_with("STRIX_")) names.push_back(text.substr(0,text.find('=')));
        }
#endif
        for(const auto& name:names) if(!environment_.contains(name)) set(name,{},true);
        for(const auto& [name,value]:environment_) set(name,value);
        qpf2k::baseline=baseline_;qpf2k::defaulted_items=defaulted_;
        qpf2k::pre_injected=injected_;qpf2k::pre_injected_moe_marker=marker_;
        detail::prefill_identity_override()=prefill_identity_;
    }
private:
    static void set(const std::string& name,const std::string& value,bool erase=false) {
#ifdef _WIN32
        ::_putenv_s(name.c_str(),erase?"":value.c_str());
#else
        if(erase) ::unsetenv(name.c_str());
        else ::setenv(name.c_str(),value.c_str(),1);
#endif
    }
    std::map<std::string,std::string> environment_;
    bool baseline_=qpf2k::baseline,marker_=qpf2k::pre_injected_moe_marker;
    unsigned defaulted_=qpf2k::defaulted_items,injected_=qpf2k::pre_injected;
    detail::PrefillIdentityOverride prefill_identity_=detail::prefill_identity_override();
};
inline std::optional<int> retry_mtp_bundle_setup(const MtpBundleSetupRetry& state,
    const std::exception& error,const std::function<int()>& run) {
    // Safety refusals remain fatal through the entry function's generic catch.
    if(dynamic_cast<const ple_reclaim::DiscardSafetyError*>(&error) ||
       dynamic_cast<const RegisteredPageLockError*>(&error)) return {};
    if(!state.selected || state.target_mutated || yarn_bundle_serial_retry) return {};
    state.restore();
    return retry_mtp_bundle_serial(error,run);
}
struct MtpBundleArguments {
    std::string path;
    std::vector<std::string> argv;
};
inline MtpBundleArguments read_mtp_bundle(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("cannot read MTP bundle");
    std::string text;
    for (char c; input.get(c);) {
        if (text.size() >= 65536U) throw std::invalid_argument("MTP bundle exceeds 64 KiB");
        text += c;
    }
    const auto json = serve::parse_json_unique(text);
    if (!json.is_object() || json.size()!=2U || !json.find("version") ||
        json.find("version")->as_int()!=1 || !json.find("options"))
        throw std::invalid_argument("MTP bundle requires version 1 and options");
    MtpBundleArguments result{path.string(), {"--mtp"}};
    const auto& options=json.find("options")->as_object();
    static constexpr std::string_view required[]{
        "--mtp-sidecar",
        "--mtp-sidecar-logical-name",
        "--mtp-sidecar-sha256",
        "--mtp-target-sha256",
        "--mtp-target-content",
        "--mtp-source-repository",
        "--mtp-source-revision",
        "--mtp-source-path",
        "--mtp-converter-sha256",
        "--mtp-norm-converter-sha256",
        "--mtp-converter-revision",
        "--mtp-conversion",
        "--mtp-normalized-rounding",
        "--mtp-projection-rounding",
        "--mtp-sidecar-residency",
        "--mtp-sidecar-bytes",
        "--mtp-header-bytes",
        "--mtp-sidecar-retained-bytes",
        "--mtp-sidecar-setup-bytes",
        "--mtp-golden-max-bytes",
        "--mtp-replay-host-bytes",
        "--mtp-runtime-bytes",
        "--mtp-target-capture-bytes",
        "--mtp-policy-id",
        "--mtp-max-hidden-error",
        "--mtp-max-logit-error",
        "--mtp-spec-min-p",
        "--mtp-draft-chain",
        "--mtp-draft-vocab",
        "--mtp-draft-context-vocab",
        "--mtp-draft-q4",
        "--mtp-candidate-depth",
        "--mtp-target-window-rows",
        "--mtp-target-window-rows-set",
        "--mtp-window-prefix",
        "--mtp-window-hyper-mix",
        "--mtp-window-qsa-rows",
        "--mtp-window-expert-union",
        "--mtp-window-expert-union-bytes",
        "--mtp-window-arena-bytes",
        "--mtp-window-capture-bytes",
        "--mtp-window-checkpoint-bytes",
        "--mtp-window-total-bytes",
        "--mtp-window-history-intervals",
    };
    // Optional keys keep the explicit flag's default when absent.
    static constexpr std::string_view optional[]{"--mtp-calibration-rows"};
    std::size_t present_optional=0;
    for (auto key:optional) present_optional+=json.find("options")->find(key)?1U:0U;
    if (options.size()!=std::size(required)+present_optional) throw std::invalid_argument("incomplete MTP bundle options");
    for (auto key:required) if (!json.find("options")->find(key))
        throw std::invalid_argument("missing MTP bundle option: "+std::string(key));
    for (const auto& [key,value]:options) {
        if (!key.starts_with("--mtp-") || key=="--mtp-bundle" ||
            (key.starts_with("--mtp-golden") && key!="--mtp-golden-max-bytes") ||
            key=="--mtp-metadata-only" || key=="--mtp-draft-profile")
            throw std::invalid_argument("unsupported MTP bundle option: "+key);
        result.argv.push_back(key);
        if (value.is_bool()) {
            if (!value.as_bool()) throw std::invalid_argument("bundle switches must be true");
            continue;
        }
        std::string v=value.is_string()?value.as_string():value.number_source();
        if (key=="--mtp-sidecar" || key=="--mtp-draft-vocab") {
            // The draft vocabulary may name a directory for per-prompt selection ("auto:DIR"), as the explicit flag does.
            const bool automatic=key=="--mtp-draft-vocab" && v.starts_with("auto:");
            const std::filesystem::path relative(automatic?v.substr(5):v);
            if (relative.empty() || relative.is_absolute() || v.find(',')!=std::string::npos ||
                (!automatic && v.starts_with("auto:")))
                throw std::invalid_argument("bundle assets require one relative file or auto:directory");
            v=(automatic?"auto:":"")+(path.parent_path()/relative).lexically_normal().string();
        }
        result.argv.push_back(std::move(v));
    }
    return result;
}
inline void verify_mtp_bundle_sidecar(const std::string& path, const std::string& sha, std::uint64_t bytes) {
    auto expected=sha;
    for (auto& c:expected) if (c>='A' && c<='F') c=static_cast<char>(c-'A'+'a');
    if (std::filesystem::file_size(path)!=bytes ||
        model_content_identity_hex(sha256_file_content(path,FileHashPolicy::cached))!=expected)
        throw std::invalid_argument("MTP bundle sidecar size/SHA256 mismatch");
}
inline std::filesystem::path mtp_golden_cache_dir() {
    if (const char* xdg=std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
        return std::filesystem::path(xdg)/"strix-engine/goldens";
    if (const char* home=std::getenv("HOME"); home && *home)
        return std::filesystem::path(home)/".cache/strix-engine/goldens";
    throw std::runtime_error("no golden cache home");
}
// Dump hooks are also used by CPU tests. Publication is always in the cache,
// even when the bundle directory is writable. No partial file is a cache hit.
inline std::filesystem::path ensure_mtp_bundle_golden(const std::filesystem::path& bundle,
    const std::string& identity, const std::function<void(const std::filesystem::path&)>& dump,
    const std::function<std::filesystem::path()>& cache_dir=mtp_golden_cache_dir) {
    const auto shipped=bundle.parent_path()/"goldens"/(identity+".golden");
    if (std::filesystem::is_regular_file(shipped)) return shipped;
    const auto dir=cache_dir(), final=dir/(identity+".golden");
    if (std::filesystem::is_regular_file(final)) return final;
    std::filesystem::create_directories(dir);
    static std::atomic<std::uint64_t> sequence{0};
    const auto temp=std::filesystem::path(final.string()+".tmp-"+
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count())+"-"+
        std::to_string(sequence.fetch_add(1)));
    try {
        dump(temp);
        std::filesystem::rename(temp,final);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temp,ignored);
        throw;
    }
    std::cerr << "YaRN MTP calibration cached: " << final.string() << '\n';
    return final;
}
} // namespace strix::decode_cli
