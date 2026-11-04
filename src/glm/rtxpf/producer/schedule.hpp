#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
namespace strix::glm::rtxpf::producer {
inline constexpr unsigned max_capacity=131072,max_row_tile=32768;
// Zero is the capacity-scaled candidate; explicit numeric tiles remain fixed.
inline unsigned effective_row_tile(unsigned capacity,unsigned requested){return requested?requested:(capacity>=32768?2048:256);}
inline bool valid_schedule(unsigned capacity,unsigned tile){return capacity>=4&&capacity<=max_capacity&&capacity%4==0&&tile>=32&&tile<=std::min(capacity,max_row_tile)&&tile%32==0;}
// Default prefill tile (tierstream service). The tile must not eat the expert
// arena: candidates start at min(capacity, default_max_row_tile) (8192 = one piece
// at 8K, 4 at 32K) rounded down to 32 and halve (rounded down to 32) to 32, each a
// valid schedule; the first one is used that the whole-device plan admits AND whose
// activation + workspace growth over tile 2048 is at most 5% of the expert arena
// planned at tile 2048 (tiles <= 2048 have no growth). Larger tiles adding file
// homes must also pass tile_file_admission below. Explicit --row-tile bypasses both
// percentage rules; the whole-device plan must still fit. 0 ends the list.
inline constexpr unsigned default_max_row_tile=8192,baseline_row_tile=2048,tile_arena_share_percent=5;
// MTP prefill runs in pieces of at most 2048 rows whatever the trunk tile (its per-row arithmetic does not
// depend on the piece: per-row router, GEMM K order and absolute DSA positions; pieces stay 4-aligned).
inline constexpr unsigned mtp_max_row_tile=2048;
inline unsigned mtp_row_tile(unsigned tile){return tile<mtp_max_row_tile?tile:mtp_max_row_tile;}
inline unsigned first_tile_candidate(unsigned capacity,unsigned cap_tile=max_row_tile){return std::min({capacity,max_row_tile,cap_tile})/32*32;}
inline unsigned next_tile_candidate(unsigned tile){return tile>32?std::max(32u,tile/2/32*32):0u;}
inline bool tile_growth_within(std::uint64_t growth,std::uint64_t baseline_arena){return growth*100<=std::uint64_t(tile_arena_share_percent)*baseline_arena;}
// Full-card receipts: ~1.5-4.6% of routed units on file at 2048, ~2.4% more at
// 8192; TTFT -30% at 32K, -9% / -5% at 8K, bitwise equal. Outside these bounds
// retain the old veto on extra file homes.
inline constexpr unsigned tile_base_max_file_percent=15,tile_extra_max_file_percent=5;
struct TileFileAdmission {bool admitted;const char* reason;};
inline TileFileAdmission tile_file_admission(std::uint64_t baseline_file,std::uint64_t candidate_file,std::uint64_t keys){
    if(candidate_file<=baseline_file)return {true,"no additional file units"};
    if(!keys||baseline_file*100>keys*tile_base_max_file_percent)
        return {false,"additional file units outside the 2048 file-share bound (15%)"};
    if((candidate_file-baseline_file)*100>keys*tile_extra_max_file_percent)
        return {false,"additional file units exceed 5% of routed units"};
    return {true,"additional file units within the 15% / 5% bounds"};
}
template<class Fits> unsigned largest_fitting_tile(unsigned capacity,Fits fits,unsigned cap_tile=max_row_tile){
    for(unsigned t=first_tile_candidate(capacity,cap_tile);t;t=next_tile_candidate(t))if(fits(t))return t;return 0;
}
// Per-request prefill tile inside the startup allocation (exact: the tile only changes the row
// chunking; the tile sweeps showed equal prefill state digests). A request whose prefill rows
// (prompt tokens - 1) are below the crossover runs at min(small_prefill_tile, allocated); at and
// above it, at the allocated tile. Crossover 0 = always the allocated tile. The crossover is the
// profile row prefill_tile_crossover (--prefill-tile-crossover); default 0 = always the allocated tile: the one-load
// sweep (allocation 8192) had 8192 fastest at every length (8K 29.2 / 30.2 / 32.5 s at 8192 / 4096 / 2048,
// 16K 36.7 / 38.6 / 43.3, 32K 53.6 / 56.6 / 66.4); the rule stays for an override or a later re-measure.
inline constexpr unsigned small_prefill_tile=4096,default_prefill_tile_crossover=0;
inline unsigned select_prefill_tile(unsigned rows,unsigned allocated,unsigned crossover){
    return crossover&&rows<crossover?std::min(small_prefill_tile,allocated):allocated;}
inline const char* prefill_tile_reason(unsigned rows,unsigned crossover){
    return !crossover?"auto: crossover off (allocated tile)":rows<crossover?"auto: rows below the crossover (small tile)":"auto: rows at or above the crossover (allocated tile)";}
struct RowTile {unsigned first,rows,bucket;};
inline RowTile row_tile(unsigned first,unsigned prefix,unsigned tile){auto n=std::min(tile,prefix-first);return {first,n,n==tile?0u:1u};}
// Prefill CUDA graph template key (graph[layer][bucket], bucket 0 = a full tile, 1 = the tail): every
// launch parameter that varies between requests is in it (first row, rows, index identity, the session's
// open KDA block), so a tile change (rows differ) recaptures the slot; equal keys = identical launches.
inline std::string prefill_binding(unsigned first,unsigned rows,const std::string& identity,bool legacy,const std::string& session){
    return std::to_string(first)+":"+std::to_string(rows)+":"+identity+":"+std::to_string(legacy?1:0)+session;}
// Persistent residuals + final hidden only. Working rows are not capacity rows.
inline std::uint64_t activation_bytes(unsigned capacity,unsigned tile){return std::uint64_t(capacity)*4096*4*5+std::uint64_t(tile)*4096*4*14+std::uint64_t(tile)*24*4+std::uint64_t(capacity+1)*4+std::uint64_t(tile)*64+4;}
}
