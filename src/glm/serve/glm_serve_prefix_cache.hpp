#pragma once
// Host-only prompt snapshots. Pending nodes are allocated before the paired commit;
// committing only splices a list node and removes oldest entries (no allocation).
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <list>
#include <span>
#include <string>
#include <vector>

namespace strix::glm::serve {

inline std::uint64_t prefix_hash(std::span<const std::uint32_t> ids) {
    std::uint64_t h = 14695981039346656037ULL;
    for (auto t : ids)
        for (unsigned b = 0; b < 4; ++b) h = (h ^ ((t >> (8 * b)) & 255U)) * 1099511628211ULL;
    return h;
}
inline std::uint64_t prefix_budget_bytes(const std::string& v) {
    std::size_t end = 0;
    const double g = std::stod(v, &end);
    if (end != v.size() || !std::isfinite(g) || g < 0 || g >= 17179869184.0)
        throw std::invalid_argument("--prefix-cache-gib: finite GiB >= 0, less than 17179869184");
    const auto bytes = static_cast<std::uint64_t>(g * 1073741824.0);
    if (g > 0 && bytes == 0) throw std::invalid_argument("--prefix-cache-gib: budget below one byte");
    return bytes;
}
inline bool prefix_memory_available(std::istream& f) {
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) != 0) continue;
        std::istringstream in(line.substr(13));
        std::string value, unit, extra;
        in >> value >> unit;
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos || unit != "kB" || (in >> extra)) return false;
        try { return std::stoull(value) >= 16ULL * 1024 * 1024; }
        catch (...) { return false; }
    }
    return false;
}
inline bool prefix_memory_available() {
    std::ifstream f("/proc/meminfo");
    return prefix_memory_available(f);
}

class PrefixCache {
public:
    struct Entry {
        std::uint64_t id = 0, identity = 0, hash = 0, frontier = 0;
        bool mtp = false;
        std::vector<std::uint32_t> ids;
        std::vector<std::uint8_t> state;
        std::uint64_t bytes() const { return state.size() + ids.size() * sizeof(std::uint32_t) + sizeof(Entry) + 2 * sizeof(void*); }
    };
    explicit PrefixCache(std::uint64_t budget = 0) : budget_(budget) {}
    const Entry* find(std::span<const std::uint32_t> prompt, bool mtp, std::uint64_t identity) const {
        const Entry* best = nullptr;
        for (const auto& e : entries_) {
            if (e.mtp != mtp || e.identity != identity || e.ids.size() >= prompt.size() || e.frontier >= prompt.size()) continue;
            const auto head = prompt.first(e.ids.size());
            if ((!best || e.ids.size() > best->ids.size()) && prefix_hash(head) == e.hash &&
                std::equal(e.ids.begin(), e.ids.end(), head.begin())) best = &e;
        }
        return best;
    }
    const Entry* get(std::uint64_t id) const {
        for (const auto& e : entries_) if (e.id == id) return &e;
        return nullptr;
    }
    std::uint64_t replacement_id() const {
        if (pending_.empty()) return 0;
        const auto& p = pending_.front();
        for (const auto& e : entries_)
            if (e.identity == p.identity && e.mtp == p.mtp && e.hash == p.hash && e.ids == p.ids) return e.id;
        return 0;
    }
    std::uint64_t oldest() const { return entries_.empty() ? 0 : entries_.front().id; }
    bool erase(std::uint64_t id) {
        for (auto i = entries_.begin(); i != entries_.end(); ++i) {
            if (i->id != id) continue;
            bytes_ -= i->bytes();
            entries_.erase(i);
            ++evictions;
            return true;
        }
        return false;
    }
    void touch(std::uint64_t id) {
        for (auto i = entries_.begin(); i != entries_.end(); ++i)
            if (i->id == id) { entries_.splice(entries_.end(), entries_, i); return; }
    }
    bool prepare(Entry e) {
        pending_.clear();
        if (!budget_ || e.state.empty() || e.bytes() > budget_) return false;
        pending_.push_back(std::move(e));
        return true;
    }
    void commit(bool accept) {
        if (accept && !pending_.empty()) {
            const auto n = pending_.front().bytes();
            while (const auto id = replacement_id()) (void)erase(id);
            while (bytes_ > budget_ - n) {
                bytes_ -= entries_.front().bytes();
                entries_.pop_front();
                ++evictions;
            }
            bytes_ += n;
            entries_.splice(entries_.end(), pending_);
            ++stores;
        }
        pending_.clear();
    }
    void clear() { entries_.clear(); pending_.clear(); bytes_ = 0; }
    std::uint64_t bytes() const { return bytes_; }
    std::uint64_t pending_bytes() const { return pending_.empty() ? 0 : pending_.front().bytes(); }
    std::uint64_t budget() const { return budget_; }
    std::uint64_t stores = 0, evictions = 0;
private:
    std::uint64_t budget_ = 0, bytes_ = 0;
    std::list<Entry> entries_, pending_;
};

}  // namespace strix::glm::serve
