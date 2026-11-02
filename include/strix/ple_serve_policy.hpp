#pragma once

#include <algorithm>
#include <cstdint>

namespace strix::ple_detail {

[[nodiscard]] inline std::uint64_t serve_available(std::uint64_t available,
    std::uint64_t unlocked_registered, std::uint64_t table_bytes, bool reserve) noexcept {
    available -= std::min(available, unlocked_registered);
    if (reserve) available -= std::min(available, table_bytes);
    return available;
}

} // namespace strix::ple_detail
