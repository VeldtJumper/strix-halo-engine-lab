#pragma once

// Prompt-lookup (n-gram) index shared by the engines' drafting.
//
// Index: an exact, incremental longest-suffix matcher over one stream's committed tokens (prompt + generated), with
// bounded memory. Interface kept minimal so a shared n-gram component can replace the local
// implementation: append committed tokens, query the longest earlier occurrence of the current suffix (optionally
// extended by up to 7 hypothetical tokens) and its continuation.
//
// Exactness: lookup only proposes draft tokens; greedy verification decides every emitted token, so output ids equal
// drafting off for any proposal. The index reads committed tokens only (never drafted ones), so a rejected draft
// needs no rollback.
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strix::lookup {

inline constexpr std::uint32_t max_draft = 7; // the verify pass holds at most 8 rows (anchor + 7 drafts)

struct Match {
    std::uint32_t length = 0;                    // matched suffix length (0: no earlier occurrence)
    std::uint64_t distance = 0;                  // current end - earlier occurrence end (recency)
    std::array<std::uint32_t, max_draft> next{}; // continuation after the earlier occurrence
    std::uint32_t available = 0;                 // valid entries of `next`
};

class Index {
public:
    virtual ~Index() = default;
    // Committed token at the next stream position.
    virtual void append(std::uint32_t token) = 0;
    // Longest suffix (>= min_length) of stream + extra that also ends at an earlier position of the stream; the
    // continuation is taken from the stream (+ extra) after that occurrence, up to `want` tokens.
    [[nodiscard]] virtual Match longest(std::span<const std::uint32_t> extra, std::uint32_t min_length,
                                        std::uint32_t want) const = 0;
    [[nodiscard]] virtual std::uint64_t tokens() const = 0;
    [[nodiscard]] virtual std::size_t bytes() const = 0;
};

struct IndexConfig {
    std::uint32_t capacity = 1U << 16; // tokens kept (older positions are forgotten)
    std::uint32_t max_match = 64;      // matched-length report cap (exact backward extension)
};

// Local implementation: hashed n-grams at lengths {1,2,3,4,6,8,12,16} (open addressing, newest occurrence wins,
// collisions resolved by token comparison), token ring of `capacity`.
[[nodiscard]] std::unique_ptr<Index> make_ngram_index(const IndexConfig& config = {});

} // namespace strix::lookup

