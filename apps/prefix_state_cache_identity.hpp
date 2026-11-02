#pragma once
#include "strix/model_content_hash.hpp"
#include "strix/prefill_identity.hpp"
#include "strix/moe_route_bf16.hpp"
#include "strix/session_state_cache.hpp"
#include "strix/qpf2k_defaults.hpp"
#include <span>
#include <string>
#include <stdexcept>
#include <vector>
namespace strix::decode_cli {
// Everything that changes the state the tiled prefill produces for a token prefix. Fields that
// cannot change it under one profile (the decode identities when T divides the prefix) are kept
// anyway: a key must never be shared by two configurations whose outputs could differ.
// `sched_defers`: whether the route that owns this namespace can execute STRIX_QPF2K_SCHED bit 2. Where it cannot
// (the MTP bridge: CLI --mtp, HTTP, cohorts; a CLI prefix-state cache run with a defaulted schedule), bit 2 is
// inert: the same kernels, rows, order and save points run as with schedule 12, so the namespace binds the
// executed schedule and entries written under the former default 12 still hit. A deferring route binds the value.
inline ModelContentIdentity prefix_effective_execution_signature(std::string_view cli, bool baseline, bool gsq, const RotaryIdentity& rotary = {}, unsigned prefill_mask = prefill_options_active(),
    bool sched_defers = true) {
    // Length-framed, sorted runtime environment also covers launcher-only controls.
    // Include post-resolver suppression separately: injected values can remain present
    // while their readers suppress them for baseline or GSQ owners.
    // Name every code-default long-context QSA control with its resolved state, so a changed
    // default (not only a changed variable) also changes the namespace.
    std::string long_context;
    for(const char* name : {"STRIX_QSA_UNION_MERGE","STRIX_QSA_SCORE_ROWS64","STRIX_QSA_DECODE_POOL",
            "STRIX_QSA_DECODE_SELECT","STRIX_QSA_DECODE_POOL_BF16","STRIX_PREFILL_MOE_MERGE_VEC"})
        long_context+=std::string(";")+name+"="+(qpf2k::long_context_item_enabled(name,gsq)?"1":"0");
    // This arithmetic candidate is opt-in; an absent variable must resolve to OFF.
    const bool route_bf16 = !baseline && !qpf2k::baseline && !gsq && moe_route_bf16_environment_requested(
        std::getenv("STRIX_QPF2K_MOE_ROUTE_BF16"), std::getenv("STRIX_BASELINE"), std::getenv("STRIX_QPF2K_DEFAULTS"));
    long_context += std::string(";STRIX_QPF2K_MOE_ROUTE_BF16=") + (route_bf16 ? "1" : "0");
    std::vector<std::string> executed;
    if (const char* v = std::getenv("STRIX_QPF2K_SCHED")) {
        const auto raw = static_cast<std::uint64_t>(std::strtoull(v, nullptr, 0));
        if (const auto run = qpf2k::effective_sched(raw, sched_defers); run != raw)
            executed.push_back("STRIX_QPF2K_SCHED=" + std::to_string(run));
    }
    const auto canonical=session_numerical_config(std::string(cli)+";prefix-execution-v5;baseline="+
        std::to_string(baseline || qpf2k::baseline)+";gsq="+std::to_string(gsq)+
        ";defaulted="+std::to_string(qpf2k::defaulted_items)+long_context+
        ";prefill_mask="+std::to_string(prefill_mask)+";prefill_identity="+prefill_identity_name(prefill_mask)+
        ";prefill_identity_hash="+std::to_string(prefill_identity_hash(prefill_mask)),rotary,executed);
    return sha256_bytes(std::as_bytes(std::span(canonical.data(),canonical.size())));
}
struct PrefixStateCacheIdentityInputs {
    ModelContentIdentity model_content;
    ModelContentIdentity effective_execution; // canonical runtime controls and suppression
    ModelContentIdentity engine_build;  // digest of the running executable (zero: not bound)
    std::uint64_t state_layout_identity = 0, runtime_compatibility_identity = 0;
    std::uint64_t decode_execution_identity = 0, decode_kernel_plan_identity = 0;
    std::uint64_t prefill_execution_identity = 0, prefill_kernel_plan_identity = 0;
    std::uint64_t tiled_execution_identity = 0, tiled_kernel_plan_identity = 0;
    std::uint64_t prefill_profile_identity = 0, decode_profile_identity = 0, profile_pair_identity = 0;
    std::uint32_t block_rows = 0;
    std::vector<std::uint32_t> tail_rows;
    std::uint64_t context_tokens = 0;
};

inline ModelContentIdentity prefix_state_cache_identity(const PrefixStateCacheIdentityInputs& in) {
    if (!in.effective_execution.valid() || !in.model_content.valid() || !in.block_rows || !in.state_layout_identity || !in.runtime_compatibility_identity ||
        !in.decode_execution_identity || !in.prefill_execution_identity || !in.tiled_execution_identity)
        throw std::invalid_argument("prefix state cache identity is incomplete");
    std::vector<std::byte> frame;
    const auto u64 = [&](std::uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) frame.push_back(static_cast<std::byte>((value >> shift) & 255U));
    };
    const auto digest = [&](const ModelContentIdentity& value) {
        for (const auto byte : value.bytes) frame.push_back(static_cast<std::byte>(byte));
    };
    u64(0x5354524958504332ULL);  // format: prefix state cache v2
    digest(in.model_content);
    digest(in.engine_build);
    digest(in.effective_execution);
    for (const auto value : {in.state_layout_identity, in.runtime_compatibility_identity, in.decode_execution_identity,
             in.decode_kernel_plan_identity, in.prefill_execution_identity, in.prefill_kernel_plan_identity,
             in.tiled_execution_identity, in.tiled_kernel_plan_identity, in.prefill_profile_identity,
             in.decode_profile_identity, in.profile_pair_identity, static_cast<std::uint64_t>(in.block_rows),
             static_cast<std::uint64_t>(in.tail_rows.size()), in.context_tokens})
        u64(value);
    for (const auto rows : in.tail_rows) u64(rows);
    return sha256_bytes(frame);
}

} // namespace strix::decode_cli
