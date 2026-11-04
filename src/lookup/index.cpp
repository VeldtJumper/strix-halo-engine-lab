#include "strix/lookup/index.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix::lookup {
namespace {
constexpr std::array<std::uint32_t, 8> levels{16, 12, 8, 6, 4, 3, 2, 1}; // longest first
constexpr std::uint64_t empty = ~0ULL;
constexpr std::size_t probes = 16; // linear-probe run; the oldest entry of a full run is evicted (bounded memory)

std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

class NgramIndex final : public Index {
public:
    explicit NgramIndex(const IndexConfig& c) : cap_(c.capacity), max_match_(c.max_match) {
        if (cap_ < 64) throw std::invalid_argument("lookup index capacity must be >= 64 tokens");
        ring_.assign(cap_, 0);
        std::size_t slots = 1;
        while (slots < std::size_t(cap_) * levels.size()) slots <<= 1;
        keys_.assign(slots, empty);
        ends_.assign(slots, 0);
        mask_ = slots - 1;
    }
    void append(std::uint32_t token) override {
        ring_[n_ % cap_] = token;
        ++n_;
        if (n_ >= 2) insert_windows(n_ - 2);
    }
    Match longest(std::span<const std::uint32_t> extra, std::uint32_t min_length, std::uint32_t want) const override {
        Match m;
        const std::uint64_t total = n_ + extra.size();
        if (total < 2 || n_ < 2) return m;
        const std::uint64_t cur = total - 1, low = n_ > cap_ ? n_ - cap_ : 0;
        const auto tok = [&](std::uint64_t i) { return i < n_ ? ring_[i % cap_] : extra[std::size_t(i - n_)]; };
        want = std::min(want, max_draft);
        for (const auto L : levels) {
            if (L < std::max(1U, min_length) || cur + 1 < L + low) continue;
            const auto key = hash(tok, cur, L);
            const auto e = find(key);
            if (e == empty || e >= cur || e + 1 < L + low) continue;
            bool same = true;
            for (std::uint32_t j = 0; j < L && same; ++j) same = tok(e - j) == tok(cur - j);
            if (!same) continue; // hash collision
            std::uint32_t len = L;
            while (len < max_match_ && e >= len + low && tok(e - len) == tok(cur - len)) ++len;
            m.length = len;
            m.distance = cur - e;
            m.source_end = e;
            for (std::uint64_t i = e + 1; i <= cur && m.available < want; ++i) m.next[m.available++] = tok(i);
            return m;
        }
        return m;
    }
    std::size_t continuation(std::uint64_t source_end, std::span<const std::uint32_t> extra,
                             std::span<std::uint32_t> out) const override {
        const std::uint64_t total = n_ + extra.size(), low = oldest();
        std::size_t count = 0;
        for (std::uint64_t i = source_end + 1; i < total && count < out.size(); ++i) {
            if (i < low) return count;
            out[count++] = i < n_ ? ring_[i % cap_] : extra[std::size_t(i - n_)];
        }
        return count;
    }
    std::uint32_t at(std::uint64_t position) const override {
        if (position >= n_ || position < oldest()) throw std::out_of_range("lookup index position outside the ring");
        return ring_[position % cap_];
    }
    std::uint64_t oldest() const override { return n_ > cap_ ? n_ - cap_ : 0; }
    std::uint64_t tokens() const override { return n_; }
    std::size_t bytes() const override { return ring_.size() * 4 + keys_.size() * 16; }

private:
    template <class Tok> static std::uint64_t hash(const Tok& tok, std::uint64_t end, std::uint32_t L) {
        std::uint64_t h = 0x9e3779b97f4a7c15ULL * (L + 1);
        for (std::uint32_t j = 0; j < L; ++j) h = mix(h ^ (std::uint64_t(tok(end - j)) + 0x632be59bd9b4e019ULL * j));
        return h | 1; // never the empty marker's low bit pattern collision: keys are odd
    }
    std::uint64_t find(std::uint64_t key) const {
        for (std::size_t p = 0; p < probes; ++p) {
            const auto s = (key + p) & mask_;
            if (keys_[s] == key) return ends_[s];
            if (keys_[s] == empty) return empty;
        }
        return empty;
    }
    void insert_windows(std::uint64_t end) {
        const std::uint64_t low = n_ > cap_ ? n_ - cap_ : 0;
        const auto tok = [&](std::uint64_t i) { return ring_[i % cap_]; };
        for (const auto L : levels) {
            if (end + 1 < L + low) continue;
            const auto key = hash(tok, end, L);
            std::size_t victim = (key) & mask_;
            std::uint64_t oldest = empty;
            bool placed = false;
            for (std::size_t p = 0; p < probes; ++p) {
                const auto s = (key + p) & mask_;
                if (keys_[s] == key || keys_[s] == empty) { keys_[s] = key; ends_[s] = end; placed = true; break; }
                if (oldest == empty || ends_[s] < oldest) { oldest = ends_[s]; victim = s; }
            }
            if (!placed) { keys_[victim] = key; ends_[victim] = end; } // bounded memory: evict the oldest of the probe run
        }
    }
    std::uint32_t cap_, max_match_;
    std::uint64_t n_ = 0;
    std::vector<std::uint32_t> ring_;
    std::vector<std::uint64_t> keys_, ends_;
    std::size_t mask_ = 0;
};
} // namespace

std::unique_ptr<Index> make_ngram_index(const IndexConfig& config) { return std::make_unique<NgramIndex>(config); }

} // namespace strix::lookup
