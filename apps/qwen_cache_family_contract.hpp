#pragma once
#include "qwen_effective_config.hpp"
#include "strix/session_state_cache.hpp"

namespace strix::decode_cli {
// Family matching only. Never use this filtered contract as a storage namespace.
// `sched_defers`: whether the runner executes STRIX_QPF2K_SCHED bit 2 (cache_runner_defers). Where it does not, the
// signature binds the schedule the runner executes (bit 2 masked), as the scheduler field of the family does.
inline ModelContentIdentity cache_family_execution_signature(std::string_view cli, bool baseline, bool gsq, const RotaryIdentity& rotary = {},
    bool sched_defers = true) {
    // Length-framed, sorted runtime environment also covers launcher-only controls.
    // Include post-resolver suppression separately: injected values can remain present
    // while their readers suppress them for baseline or GSQ owners.
    // Name every code-default long-context QSA control with its resolved state, so a changed
    // default (not only a changed variable) also changes the namespace.
    std::string long_context;
    for(const char* name : {"STRIX_QSA_UNION_MERGE","STRIX_QSA_SCORE_ROWS64","STRIX_QSA_DECODE_POOL",
            "STRIX_QSA_DECODE_SELECT","STRIX_QSA_DECODE_POOL_BF16"})
        long_context+=std::string(";")+name+"="+(qpf2k::long_context_item_enabled(name,gsq)?"1":"0");
    std::vector<std::string> executed;
    if (const char* v = std::getenv("STRIX_QPF2K_SCHED")) {
        const auto raw = static_cast<std::uint64_t>(std::strtoull(v, nullptr, 0));
        if (const auto run = qpf2k::effective_sched(raw, sched_defers); run != raw)
            executed.push_back("STRIX_QPF2K_SCHED=" + std::to_string(run));
    }
    const auto canonical=session_family_numerical_config(std::string(cli)+";prefix-execution-v4;baseline="+
        std::to_string(baseline || qpf2k::baseline)+";gsq="+std::to_string(gsq)+
        ";defaulted="+std::to_string(qpf2k::defaulted_items)+long_context,rotary,executed);
    return sha256_bytes(std::as_bytes(std::span(canonical.data(),canonical.size())));
}
// Auto vocabulary selection can read any of these files as prompts change.
// Pin their bytes, never the directory spelling. Unreadable/missing assets decline.
inline constexpr std::pair<const char*,const char*> cache_mtp_vocabulary_digests[]={
    {"core32k.bin","85578fdf1c47aba24568ed60d117e215dbb176db3f50973924a5dfa6bf78ead2"},
    {"core48k.bin","785a50f70a006f5ba57430139d837e357af9ee18dbea9852ddf1ee78cc35d532"},
    {"core64k.bin","6e734f27923153487e855fcbed7edc8c956063d90833f9b6fdfcff27a697e0d1"},
    {"pack2-de.bin","4c1154228128cde72c8a45384aa5f5f204241398535c05d2eff4bf286ecc3da2"},
    {"pack2-es.bin","62c2ad417d85ca026bab4add824b9b22e9ca9a794771c7180a9f365dc845f261"},
    {"pack2-fr.bin","1ba7876bda448d88e526484854e1a20076a4c361d85a8b1bd66364e3fedbfc6b"},
    {"pack2-it.bin","b4dbcfbc59ddf8e247c172833082a9e86b0040268da46ced725e429ef871ff58"},
    {"pack2-ja.bin","bb55228979f2c2f319a7b3e1b378ab2023110f908d83d5755c917c9815678b4b"},
    {"pack2-pt.bin","ecd376f566f9bce6d007b830d152508423be5f60cec19b5df8212efc890c9c6f"},
    {"pack2-ru.bin","e533e7af546834e857cf024c472eecb0b4af8ba196605fdba73afb575362254e"},
    {"pack2-zh.bin","4b01353de01e2f876c22ab01af51f481e886ffd4519131da1f7e902aa5f10326"},
    {"pack-de16k.bin","1646667d002945442f61f9965bd9d412a5cebb8e8ac5f9767c7681260d8ca677"},
    {"pack-es16k.bin","84ac5b184a8b2212b0fcb471203a7afbfb70814eefbd549e9f11832bf226a772"},
    {"pack-fr16k.bin","c59a60b8b2100b88c08911bc2dbacda3172561aadf282c9d3f59da320efa5115"},
    {"pack-it16k.bin","797b152ef302744e51f0d494b4f6f39dae4522f86582d6e0ee4e5341bf2a2f0f"},
    {"pack-ja16k.bin","7859afeeaf063c86b46e6e65706ca7a60bf85d0df7d7fca613d7fd563ac10170"},
    {"pack-pt16k.bin","910c1dc30b18c9526e07e98b4c5dddf2b0d02b654a63ca4b1fd2bde2dc6f0b8a"},
    {"pack-ru16k.bin","f9b6164bba32c2915ff4c2d7105a236d7fe2f0b3f5b6207ba9c13dbae9861223"},
    {"pack-zh16k.bin","39f9197b1e571a7888536c15455e150e52c0cfab1af1617b3d0cf1e26b5ea381"}
};
inline bool cache_mtp_vocabulary_qualified(const Options& o,
    const std::function<std::string(const std::filesystem::path&)>& hash=[](const auto& path) {
        return model_content_identity_hex(session_cached_file_hash(path));
    }) {
    if(!o.mtp.draft_vocab.starts_with("auto:")) return false;
    const auto directory=std::filesystem::path(o.mtp.draft_vocab.substr(5));
    try {
        for(const auto& [name,digest]:cache_mtp_vocabulary_digests)
            if(hash(directory/name)!=digest) return false;
        return true;
    } catch(const std::exception&) { return false; }
}
inline std::string cache_family_execution_cli(Options options) {
    options.context_tokens=16384;
    options.unqualified_context=true;
    // effective_prefill_argv omits http_port, http_bind and http_model_id.
    // They identify the listener/API model label, not model bytes or arithmetic.
    std::string cli;
    for(const auto& arg:effective_prefill_argv(options)) cli+=std::to_string(arg.size())+":"+arg;
    return cli;
}
} // namespace strix::decode_cli
