#pragma once
#include <cstdint>
namespace strix::glm::rtxpf {
// Complete pools selected by DSA; the incomplete pool is always appended.
inline constexpr unsigned dsa_selection_budget=512;
inline constexpr bool dsa_short_skip_eligible(bool enabled,unsigned rows,unsigned first) {
    return enabled&&rows==1&&(std::uint64_t(first)+1)/4<=dsa_selection_budget;
}
}
