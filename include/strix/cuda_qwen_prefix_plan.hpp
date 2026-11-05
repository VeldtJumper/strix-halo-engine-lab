#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strix::cuda::qwen_prefix {

inline void validate_block(std::uint32_t block) {
    if (!block || block % 4) throw std::invalid_argument("--prefix-cache-block must be a positive multiple of 4");
}

inline std::uint64_t lookup_limit(std::uint64_t tokens) { return tokens >= 2 ? tokens - 2 : 0; }

inline std::vector<std::uint64_t> capture_points(std::uint64_t tokens, std::uint32_t block,
                                                std::uint64_t restored, std::uint64_t stride) {
    validate_block(block);
    if (stride && stride % block) throw std::invalid_argument("--prefix-cache-stride must be 0 or a multiple of --prefix-cache-block");
    const auto last = lookup_limit(tokens) / block * block;
    std::vector<std::uint64_t> points;
    if (last <= restored) return points;
    if (stride) {
        for (auto p = (restored / stride + 1) * stride; p < last; p += stride) points.push_back(p);
    }
    points.push_back(last);
    return points;
}

inline std::uint32_t clip_chunk(std::uint64_t prefilled, std::uint32_t rows,
                                const std::vector<std::uint64_t>& points) {
    const auto next = std::upper_bound(points.begin(), points.end(), prefilled);
    return next == points.end() ? rows : static_cast<std::uint32_t>(std::min<std::uint64_t>(rows, *next - prefilled));
}

}  // namespace strix::cuda::qwen_prefix
