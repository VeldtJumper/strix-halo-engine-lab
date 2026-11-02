#pragma once
// Lazy host promotion of streamed file-tier expert layers (CPU-only policy; the transport is ds41_expert_streamer).
// A file-tier layer read for a request goes into a new pinned buffer when the host has room for one more layer, and
// streams from there on the following requests (the same file bytes). At a request start, a host below the floor
// gives back the least recently used promoted layers (never the load-time host tier) until the shortfall is covered.
// Room and floor follow host_pin_admission: physical >= n + floor + 4 GiB, commit >= 2n + floor.
#include "strix/ds41_memory_budget.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strix::ds41 {

struct HostPromotePolicy {
    bool enabled = false;
    std::uint64_t floor_bytes = 0;  // free physical / commit kept for everything else
    std::uint64_t cap_bytes = 0;    // promoted bytes at most (0: only the floor bounds them)
};

// The bytes one more promoted layer may take now: min(physical - floor - 4 GiB, (commit - floor) / 2, cap - promoted);
// 0 when the switch is off or the telemetry is unknown.
inline std::uint64_t host_promote_room(const HostMemoryRoom& room, const HostPromotePolicy& p, std::uint64_t promoted_bytes) {
    if (!p.enabled || !room.known) return 0;
    const std::uint64_t margin = p.floor_bytes + (4ULL << 30);
    if (room.free_physical <= margin || room.free_commit <= p.floor_bytes) return 0;
    std::uint64_t r = std::min(room.free_physical - margin, (room.free_commit - p.floor_bytes) / 2U);
    if (p.cap_bytes != 0U) r = std::min(r, p.cap_bytes > promoted_bytes ? p.cap_bytes - promoted_bytes : 0U);
    return r;
}
// Never below one whole layer (the largest streamed part).
inline bool host_promote_admit(std::uint64_t room, std::uint64_t layer_bytes) { return layer_bytes != 0U && room >= layer_bytes; }

// The shortfall below the floor at a request start (0: none, or unknown telemetry).
inline std::uint64_t host_promote_deficit(const HostMemoryRoom& room, const HostPromotePolicy& p) {
    if (!room.known) return 0;
    const std::uint64_t margin = p.floor_bytes + (4ULL << 30);
    const std::uint64_t phys = room.free_physical < margin ? margin - room.free_physical : 0U;
    const std::uint64_t commit = room.free_commit < p.floor_bytes ? p.floor_bytes - room.free_commit : 0U;
    return std::max(phys, commit);
}

// The promoted layers with their last use; the load-time host tier is never in it.
class HostPromoteLru {
public:
    void add(std::uint32_t layer, std::uint64_t bytes, std::uint64_t stamp) {
        remove(layer);
        e_.push_back({layer, bytes, stamp});
        bytes_ += bytes;
    }
    void touch(std::uint32_t layer, std::uint64_t stamp) {
        for (auto& e : e_)
            if (e.layer == layer) e.stamp = stamp;
    }
    void remove(std::uint32_t layer) {
        for (auto it = e_.begin(); it != e_.end(); ++it)
            if (it->layer == layer) {
                bytes_ -= it->bytes;
                e_.erase(it);
                return;
            }
    }
    [[nodiscard]] bool contains(std::uint32_t layer) const {
        return std::any_of(e_.begin(), e_.end(), [&](const Entry& e) { return e.layer == layer; });
    }
    // Least recently used first (ties: lower layer first) until their bytes cover `deficit` (0: none).
    [[nodiscard]] std::vector<std::uint32_t> victims(std::uint64_t deficit) const {
        std::vector<std::uint32_t> out;
        if (deficit == 0U) return out;
        auto order = e_;
        std::sort(order.begin(), order.end(), [](const Entry& a, const Entry& b) { return a.stamp != b.stamp ? a.stamp < b.stamp : a.layer < b.layer; });
        std::uint64_t freed = 0;
        for (const auto& e : order) {
            if (freed >= deficit) break;
            out.push_back(e.layer);
            freed += e.bytes;
        }
        return out;
    }
    [[nodiscard]] std::size_t size() const noexcept { return e_.size(); }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

private:
    struct Entry {
        std::uint32_t layer = 0;
        std::uint64_t bytes = 0, stamp = 0;
    };
    std::vector<Entry> e_;
    std::uint64_t bytes_ = 0;
};

}  // namespace strix::ds41
