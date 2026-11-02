#pragma once
// Versioned complete-session envelope layered on PrefixStateCache buffers/accounting.
#include "strix/prefix_state_cache.hpp"
#include "strix/rotary_descriptor.hpp"
#include <span>
#include <string>
#include <string_view>

namespace strix {
struct SessionCompatibility {
    static constexpr std::uint64_t schema = 1;
    ModelContentIdentity model, executable;
    std::string numerical_config; // canonical, length-framed effective settings and plan
    bool mtp = false;
    [[nodiscard]] ModelContentIdentity digest() const;
};
struct SessionSuffixPlan { bool tiled = false; std::uint64_t serial_rows = 0; };
// New session routes align with the ordinary cached-prefix block boundary. Tail rules apply
// only after at least one qualified full block; short continuations preserve the serial route.
[[nodiscard]] inline SessionSuffixPlan plan_session_suffix(std::uint64_t position, std::uint64_t inputs,
    std::uint64_t block_rows, bool enabled) {
    if (!enabled || !block_rows) return {false, inputs};
    const auto align = (block_rows - position % block_rows) % block_rows;
    if (inputs < align || inputs - align < block_rows) return {false, inputs};
    return {true, align};
}
struct SessionFrontier {
    std::uint64_t consumed = 0, emitted = 0, accepted = 0, predictor = 0;
    std::optional<std::uint32_t> pending;
    std::string sampling = "greedy";
    std::vector<std::uint32_t> tokens; // consumed inputs only; pending is separate
};
struct SessionSection { std::string name; PrefixCacheBuffer payload; };
struct SessionCheckpoint {
    SessionCompatibility compatibility;
    SessionFrontier frontier;
    std::vector<SessionSection> sections; // rank 0, replicated (single node)
};
// Opportunistic named lookup: incompatible/corrupt envelopes are misses, never mutate a live slot.
[[nodiscard]] std::optional<SessionCheckpoint> lookup_session_checkpoint(PrefixStateCache&, const std::string&,
    const SessionCompatibility&);
// Four-copy peak for envelope, decoded sections and transient vector capacity.
// Pass current available memory after model/prefill/state backing is retained.
[[nodiscard]] inline std::uint64_t session_checkpoint_envelope_limit(std::uint64_t host_budget,
    std::uint64_t available_bytes = UINT64_MAX) noexcept {
    return (host_budget < available_bytes ? host_budget : available_bytes) / 4U;
}
struct SessionIoStats { std::uint64_t bytes = 0; double read_verify_ms = 0, write_fsync_ms = 0; };
// known_full_digest may seed an already validated mapped-file digest; subsequent calls recheck the file stamp.
[[nodiscard]] ModelContentIdentity session_cached_file_hash(const std::filesystem::path&, std::optional<ModelContentIdentity> known_full_digest = std::nullopt);
// Canonical STRIX_*, HIP_*, HSA_*, AMD_*, ROCR_*, GPU_* environment values plus caller's effective CLI/compiled plan record.
[[nodiscard]] std::string session_numerical_config(std::string_view effective_cli, const RotaryIdentity& rotary = {});
// The same, with each "NAME=VALUE" of `replace` standing for the process's NAME entry (the value the route executes).
[[nodiscard]] std::string session_numerical_config(std::string_view effective_cli, const RotaryIdentity& rotary,
    std::span<const std::string> replace);
// Qualification only: storage/certificate callers continue using the unfiltered function.
[[nodiscard]] std::string session_family_numerical_config(std::string_view effective_cli, const RotaryIdentity& rotary = {});
// The same, with each "NAME=VALUE" of `replace` standing for the process's NAME entry (the value the route executes).
[[nodiscard]] std::string session_family_numerical_config(std::string_view effective_cli, const RotaryIdentity& rotary,
    std::span<const std::string> replace);
[[nodiscard]] PrefixCacheBuffer encode_session_checkpoint(const SessionCheckpoint&, const PrefixCacheAllocator&);
// All checks complete before this returns. expected mismatch throws; lookup wrapper returns null.
[[nodiscard]] SessionCheckpoint decode_session_checkpoint(const PrefixCacheBuffer&, const SessionCompatibility& expected);
void publish_session_checkpoint(const std::filesystem::path& directory, const PrefixCacheBuffer&, SessionIoStats* = nullptr);
[[nodiscard]] PrefixCacheBuffer read_session_checkpoint(const std::filesystem::path& directory,
    const SessionCompatibility&, const PrefixCacheAllocator&, std::uint64_t max_bytes, SessionIoStats* = nullptr, SessionCheckpoint* decoded = nullptr);
// CPU preflight only: schema/hashes + executable and supplied CLI/environment closure.
// max_bytes is the four-copy host working-set budget, not the envelope byte limit.
// Does not certify model, observed device or compiled-plan compatibility; normal restore still does.
void preflight_session_checkpoint(const std::filesystem::path&, const ModelContentIdentity& executable,
    std::string_view numerical_prefix, std::uint64_t max_bytes);
} // namespace strix
