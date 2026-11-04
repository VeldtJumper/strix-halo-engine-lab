#pragma once
// Host route policy shared by launchers and the CPU census. No HIP dependencies.
// The GEMM policy below preserves the adopted strix-trellis 0.3 policy verbatim.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::minimax::route {
inline std::vector<std::vector<std::uint32_t>> expert_rows(std::span<const std::uint32_t> ids, unsigned rows,
                                                         unsigned experts, unsigned top_k) {
    if (ids.size() != std::size_t{rows} * top_k) throw std::invalid_argument("minimax prefill moe: route table size");
    std::vector<std::vector<std::uint32_t>> grouped(experts);
    for (unsigned t = 0; t < rows; ++t)
        for (unsigned k = 0; k < top_k; ++k) {
            const std::uint32_t expert = ids[std::size_t{t} * top_k + k];
            if (expert >= experts) throw std::invalid_argument("minimax prefill moe: expert id out of range");
            grouped[expert].push_back(t);
        }
    return grouped;
}
inline bool medium_prefill_candidate(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    if (rows < 32) return false;
    if (rows >= 256) return (bits == 6 && input == 5120 && output == 1280) || (bits == 5 && input == 4096 && output == 1024);
    if (bits == 6) return input == 5120 && output == 1280;
    if (bits != 5) return false;
    return (input == 1280 && output == 32768) || (input == 8192 && output == 5120) || (input == 5120 && output == 2304) ||
           (input == 2304 && output == 5120) || (input == 4096 && output == 1024);
}
inline unsigned measured_vector_splits(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    if (bits == 5) {
        if (input == 4096 && output == 1024 && (rows == 1 || rows == 4 || rows == 8)) return 16;
        if (input == 5120 && output == 2304 && rows == 8) return 8;
    }
    return output <= 1024 ? 32U : (output <= 4096 && input >= 4096 ? 16U : 4U);
}
inline unsigned gemm_tile_rows(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    return medium_prefill_candidate(input, output, bits, rows) || (rows >= 256 && output >= 2304) ||
                   (rows >= 64 && output >= 32768 && bits == 5) ? 64U : 16U;
}
inline unsigned gemm_splits(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    std::size_t requested;
    if (rows > 8) {
        const std::size_t tile_m = gemm_tile_rows(input, output, bits, rows);
        const std::size_t groups = ((rows + tile_m - 1) / tile_m) * ((output + 63U) / 64U);
        if (medium_prefill_candidate(input, output, bits, rows)) {
            if (output >= 32768) {
                requested = rows < 64 ? 4U : 1U;
            } else {
                const std::size_t target = rows > 64 && rows % 64 ? 512U : 256U;
                requested = 1;
                while (groups * requested < target && requested < 16) requested *= 2;
            }
        } else {
            requested = 1;
            while (groups * requested < 160 && requested < 32) requested *= 2;
        }
    } else {
        requested = measured_vector_splits(input, output, bits, rows);
    }
    return static_cast<unsigned>(std::min<std::size_t>(requested, input / 16));
}
inline unsigned mt_class(unsigned rows) { return rows <= 16 ? 0U : rows <= 32 ? 1U : rows <= 64 ? 2U : 3U; }
inline unsigned mt_of_class(unsigned cls) { return 1U << cls; }
struct MtFit { unsigned main, remainder, tail_class; };
inline MtFit mtfit(unsigned rows) {
    const unsigned remainder = rows % 128;
    return {rows / 128, remainder, remainder != 0 ? (remainder + 15) / 16 - 1 : 0U};
}
#if defined(__HIPCC__)
#define STRIX_MM_ROUTE_HD __attribute__((host)) __attribute__((device))
#else
#define STRIX_MM_ROUTE_HD
#endif
// Physical decode selectors; arithmetic identity is unchanged (see route registry gates).
STRIX_MM_ROUTE_HD inline constexpr bool moe_interleave(unsigned rows, bool enabled) { return enabled && rows == 8; }
STRIX_MM_ROUTE_HD inline constexpr bool moe_single_slot(unsigned rows, unsigned count, unsigned singles,
                                      bool shared, unsigned slot_rows) {
    return rows == 8 && count >= 25 && singles >= 18 && !shared && slot_rows == 1;
}
STRIX_MM_ROUTE_HD inline constexpr bool moe_single_reader(bool enabled, unsigned rows, unsigned count, unsigned singles,
                                        bool shared, unsigned slot_rows, unsigned bits, bool wide, unsigned waves, unsigned depth) {
    return enabled && depth == 4 && !wide && waves == 4 && bits == 3 &&
           moe_single_slot(rows, count, singles, shared, slot_rows);
}
// Decode MoE down partials + one stream-ordered 48-CTA finish (STRIX_MMX_MOE_DOWN_SPLIT): mode 1 = the measured
// winning regime only (N=4 wide W4 Depth4; host-known, no sync to read U/S), 2 = every v4 down, 0 = off.
STRIX_MM_ROUTE_HD inline constexpr bool moe_down_split(unsigned mode, unsigned rows, bool wide, unsigned waves,
                                                       unsigned depth) {
    return mode == 2 || (mode == 1 && rows == 4 && wide && waves == 4 && depth == 4);
}
#undef STRIX_MM_ROUTE_HD
// Decode attention grids where the measured alternatives won (t9 LDS diet, t12 v4 one head per wave; the diet is tried
// first, STRIX_MM_ATTEND_DIET, then v4, STRIX_MM_ATTEND4_DECODE): two single-row
// sparse sequences, both at >= 8K positions. Dense, N1, N4/N8 and verify windows keep v3 at the 42 KB footprint.
inline bool attend_n2_sparse(unsigned seqs, unsigned rows_per_seq, bool dense, std::uint64_t min_pos) {
    return !dense && seqs == 2 && rows_per_seq == 1 && min_pos + 1 >= 8192;
}
// Decode selection once per (row, KV head) (STRIX_MM_SELECT_ONCE): a separate selection launch instead of every chunk
// block of the attend launch repeating it; same IDs and bits. Sparse calls of >= kSelectOnceRows rows whose rows are all
// at >= kSelectOncePos positions (s1b/s2: 32K N4/N8 -1.8/-4.3 %, W4/W8 -3.5/-6.4 %, 128K N8 -1.4 %; the extra launch
// loses at N1 (+3..+8 %) and the selection is too cheap at 8K (N4 +1.1 %)).
inline constexpr unsigned kSelectOnceRows = 4;
inline constexpr std::uint64_t kSelectOncePos = 16384;
inline bool select_once(bool enabled, unsigned rows, bool dense, std::uint64_t min_pos) {
    return enabled && !dense && rows >= kSelectOnceRows && min_pos + 1 >= kSelectOncePos;
}
inline bool wide_for(unsigned rows, int mode) { return mode == 1 || (mode == 2 && rows >= 2 && rows <= 4); }
inline int scheduled_linear_version(unsigned input, unsigned output, unsigned count, unsigned rows) {
    if (output >= 65536) return 4;
    if (input == 6144 && output == 12288) return rows <= 4 ? 8 : 10;
    if (input == 12288 && output == 6144) return rows <= 2 ? 10 : 9;
    if (count == 3 || (input == 6144 && output == 4096)) return 8;
    if (output == 256) return rows <= 4 ? 4 : 8;
    if (input == 4096 && output == 6144) return rows == 1 ? 10 : (rows == 2 ? 9 : (rows <= 4 ? 4 : 8));
    if (input == 6144 && output == 6144) return rows <= 4 ? 10 : 8;
    return output % 256 == 0 ? 10 : (rows == 1 || rows == 8 ? 4 : 8);
}
struct StepLayout { std::uint32_t seqs, rows_per_seq; };
template <class Row> StepLayout step_layout(std::span<const Row> rows) {
    bool distinct = true, one_slot = true;
    for (std::size_t r = 0; r < rows.size(); ++r)
        for (std::size_t s = 0; s < r; ++s) {
            distinct &= rows[s].slot != rows[r].slot;
            one_slot &= rows[s].slot == rows[r].slot;
        }
    const auto n = static_cast<std::uint32_t>(rows.size());
    return {distinct ? n : one_slot ? 1U : 0U, distinct ? 1U : one_slot ? n : 0U};
}
template <class Row> bool step_dense(std::span<const Row> rows, std::uint64_t boundary) {
    bool dense = true;
    for (const auto& row : rows) dense &= row.pos < boundary;
    return dense;
}
inline int kv_tile_config(std::size_t rows, int mode) { return mode == -2 ? (rows >= 2048 ? 1 : 7) : mode; }
inline std::size_t unit_rows(std::size_t members, std::uint32_t max_rows, std::size_t remaining) {
    return std::min<std::size_t>(members * max_rows, remaining);
}
inline unsigned layer_end(unsigned begin, unsigned budget, unsigned layers) {
    return begin + std::min<unsigned>(budget, layers - begin);
}
inline bool fp32_profile(const char* value) { return value != nullptr && std::string_view(value) == "fp32"; }
inline std::size_t append_rows(std::size_t remaining) { return remaining >= 32 ? 32 : std::min<std::size_t>(8, remaining); }
struct IndexRoute { unsigned ring, rows_per_wave; bool lds, tiled; };
inline IndexRoute index_route(unsigned rows, unsigned ring, unsigned split, unsigned lds) {
    return {ring != 0 ? ring : rows == 1 ? 32U : 16U, rows > 2 ? split : 0U,
            rows >= 2 && rows <= 8 && lds != 0U, rows > 8};
}
// Number of complete dense windows. A straddling window takes the sparse path.
inline unsigned dense_windows(std::uint64_t pos0, unsigned rows, unsigned windows, unsigned window_rows,
                              std::uint64_t boundary) {
    unsigned dense = 0;
    for (; dense < windows; ++dense) {
        const unsigned end = (dense + 1) * window_rows < rows ? (dense + 1) * window_rows : rows;
        if (pos0 + end - 1 >= boundary) break;
    }
    return dense;
}
}  // namespace strix::minimax::route
