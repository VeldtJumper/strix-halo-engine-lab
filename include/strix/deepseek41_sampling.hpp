#pragma once

#include "strix/serve/engine.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace strix::deepseek41 {

// One request owns its sampler. Absolute-position RNG keeps batching and prefix reuse
// from consuming another request's draws. Greedy never reads logits through this path.
class TargetOnlySampler {
public:
    explicit TargetOnlySampler(const serve::SamplingParameters& parameters) {
        const TokenSamplingPolicy policy{static_cast<float>(parameters.temperature), parameters.top_k,
            static_cast<float>(parameters.top_p), static_cast<float>(parameters.min_p), parameters.seed};
        validate_token_sampling_policy(policy);
        if (token_sampling_enabled(policy)) sampler_.emplace(policy);
    }
    [[nodiscard]] bool enabled() const noexcept { return sampler_.has_value(); }
    [[nodiscard]] std::uint32_t choose(std::span<const float> logits, std::uint64_t position, std::uint32_t greedy) {
        return sampler_ ? sampler_->decide(logits, position, nullptr).token : greedy;
    }
private:
    std::optional<TokenSampler> sampler_;
};

// A whole-prompt cache entry has a greedy next token, not the full scores. Leave
// a qualified tail to execute so the first sample uses fresh target logits. Merely
// dropping one token can pick an exact shorter hit that leaves an undersized tail.
inline std::span<const std::uint32_t> sampling_prefix_query(std::span<const std::uint32_t> prompt, bool sampled,
                                                          std::size_t minimum_tail = 1U) {
    if (!sampled) return prompt;
    const auto tail = std::max<std::size_t>(minimum_tail, 1U);
    return prompt.first(prompt.size() > tail ? prompt.size() - tail : 0U);
}

// Never run a sampled sequence through a greedy speculative verifier. Separate
// batches also keep the greedy sequences' drafter state advancing normally.
template<class Sequence>
std::array<std::vector<Sequence*>, 2> split_sampling_batches(const std::vector<Sequence*>& sequences) {
    std::array<std::vector<Sequence*>, 2> batches;
    for (auto* sequence : sequences) batches[sequence->sampled ? 1U : 0U].push_back(sequence);
    return batches;
}

}  // namespace strix::deepseek41
