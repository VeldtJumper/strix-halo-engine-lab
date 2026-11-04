#pragma once
// MiniMax H3 two-node sequence-parallel schedule (D1): row split, key-block geometry and the K/V exchange message.
//
// Every rank owns a contiguous range of ALL packed rows. The ranges are the globally indexed key blocks of the
// attention: block 0 = rows [0, cut), block 1 = rows [cut, S), cut = round_down_64(S / 2) (the same cut on one and two
// nodes, so every node uses the same tile boundaries). Every op except the attention is row-local; the attention of a
// rank's query rows is the canonical merge (block 0 then block 1) of one partial per key block: the local block from
// the rank's own K/V, the other from the peer's, received through one exchange per DiT block.
//
// The exchange message of a block (both ranks send the same size, so the larger half sets it):
//   [0, k_bytes)                K  [max_rows][56][128] bf16 (token-major, rows >= rows(b) unused)
//   [k_bytes, message_bytes)    V^T [56][128][vt_pitch] bf16 (keys contiguous, keys >= rows(b) unused)
// vt_pitch = round_up(max_rows, 64), so K and V^T start 16-byte aligned and the message is a multiple of 64 bytes.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strix::h3::dist {

inline constexpr std::uint32_t kHeads = 56;
inline constexpr std::uint32_t kHeadDim = 128;
inline constexpr std::uint32_t kInner = kHeads * kHeadDim;  // 7168
inline constexpr std::uint32_t kCutAlign = 64;               // the attention kernel's key step

// The canonical cut of S packed rows (0 when S < 128: one key block).
inline std::uint32_t canonical_cut(std::uint32_t S) { return (S / 2U) / kCutAlign * kCutAlign; }

struct Split {
    std::uint32_t S = 0, cut = 0;

    static Split of(std::uint32_t S) {
        const Split s{S, canonical_cut(S)};
        if (s.cut == 0U) throw std::invalid_argument("h3 dist: sequence too short for two key blocks (S < 128)");
        return s;
    }
    // The sparse-attention identity's split (h3-attn-sparse-thr-v1): cut = round_down_128(S / 2) = 128 floor(S / 256),
    // used by that identity in BOTH worlds (its query tiles of 128 rows are then fixed global row sets).
    static Split sparse(std::uint32_t S) {
        const Split s{S, 128U * (S / 256U)};
        if (s.cut == 0U) throw std::invalid_argument("h3 dist: sequence too short for the sparse split (S < 256)");
        return s;
    }
    std::uint32_t first(unsigned b) const { return b == 0U ? 0U : cut; }
    std::uint32_t rows(unsigned b) const { return b == 0U ? cut : S - cut; }
    std::uint32_t max_rows() const { return std::max(cut, S - cut); }
    std::uint64_t vt_pitch() const { return (std::uint64_t{max_rows()} + kCutAlign - 1U) / kCutAlign * kCutAlign; }
    std::size_t k_bytes() const { return std::size_t{max_rows()} * kInner * 2U; }
    std::size_t vt_bytes() const { return std::size_t{kInner} * vt_pitch() * 2U; }
    std::size_t message_bytes() const { return k_bytes() + vt_bytes(); }
};

// Which key block (= row range) a rank owns. world 1: both blocks (one node runs the same two partials + merge).
// world 2: rank r owns block r, or block 1 - r with `swap` (the merge order is by block, never by rank).
struct Role {
    std::uint32_t rank = 0, world = 1;
    bool swap = false;
    unsigned block() const { return world == 1U ? 0U : (rank ^ (swap ? 1U : 0U)); }
    unsigned peer_block() const { return 1U - block(); }
    // Local query rows: [row0, row0 + rows).
    std::uint32_t row0(const Split& s) const { return world == 1U ? 0U : s.first(block()); }
    std::uint32_t rows(const Split& s) const { return world == 1U ? s.S : s.rows(block()); }
};

// FNV-1a 64 of a byte range (per-block residual hashes, same on every host).
inline std::uint64_t fnv1a64(const void* data, std::size_t n, std::uint64_t h = 0xcbf29ce484222325ULL) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

}  // namespace strix::h3::dist
