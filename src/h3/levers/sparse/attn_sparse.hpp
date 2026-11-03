#pragma once
// Block-sparse variant of K1 (identity h3-attn-sparse-thr-v1, FAST-class, OFF by default): a levers-owned copy of the
// K1 v1 default schedule (head dim 128, 8 waves = 128 query rows per work-group, early next-tile loads, grid launch,
// single LDS buffer) whose per-partial key loop walks a LUT of 64-key tiles instead of every tile of the block.
// Arithmetic per processed tile, partial format (mc, l, o), masking and the canonical merge are K1's (h3/attn
// b0a559848, attn.hpp / attn_ref.hpp); a LUT listing every tile of a block gives K1's dense partial bit for bit
// (gate: sparse_attn_gpu_tests).
//
// LUT of one key block b (built by the router, router.hpp build_lut with j0 / j1 = the block's tile range):
//   list (query tile x, head h) = entries[offsets[x * heads + h] .. offsets[x * heads + h + 1]),
//   entries = GLOBAL 64-key tile indices, strictly ascending, inside [tile0, tile0 + ceil(keys / 64));
//   query tile x = rows [128 x, 128 x + 128) of this call's queries (the router's query tile t0 + x).
// The first 1024 entries of a list are staged in LDS at work-group start (no global load on the step path).
// Contract checks in the kernel: an entry out of range or not strictly ascending is skipped (never read) and counted
// in *lut_errors (device counter, optional; callers assert 0). An empty list writes the complete empty partial
// (mc = -inf, l = 0, o = +0), which the merge weighs 0.
#include <cstdint>

#include "h3/attn/attn.hpp"

namespace strix::h3::levers::sparse {

struct BlockLut {
    const std::int32_t* entries = nullptr;
    const std::uint32_t* offsets = nullptr;  // (query tiles x heads) + 1
    std::uint32_t tile0 = 0;                 // global tile index of the block's key 0 (cut / 64 for block 1)
};

// Partials of blocks[0 .. nblocks) with their LUTs into slots slot0 .. (same layout as attn::launch_partials).
void launch_sparse_partials(const attn::Queries& q, const attn::KeyBlock* blocks, const BlockLut* luts, unsigned nblocks, const attn::Partials& p,
                            unsigned slot0, unsigned* lut_errors, void* stream);
// Tests only: the same kernel with a 4-entry LDS stage, so lists longer than the stage exercise the global-read path.
void launch_sparse_partials_small_stage_for_test(const attn::Queries& q, const attn::KeyBlock* blocks, const BlockLut* luts, unsigned nblocks,
                                                 const attn::Partials& p, unsigned slot0, unsigned* lut_errors, void* stream);

}  // namespace strix::h3::levers::sparse
