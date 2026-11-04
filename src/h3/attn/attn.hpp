#pragma once
// MiniMax H3 dense non-causal flash attention, head dim 128, bf16 WMMA, fp32 online softmax, split-KV partials and
// the canonical merge (gfx1151). The arithmetic contract is in attn_ref.hpp; the layouts are below. All strides
// are in elements; every row start (q, k, vt rows) must be 16-byte aligned.
//
// Operand contract (the qualified domain):
//   - Q, K, V finite bf16;
//   - ||q_row||_2 * ||k_row||_2 * log2(e) / sqrt(128) <= 2^14 for every (query row, key row, head) of a call. It
//     bounds every score |s c| (Cauchy-Schwarz), hence every running and transient maximum; H3's q and k leave a
//     per-head RMSNorm, so ||q||, ||k|| <= sqrt(128) * max|norm weight| (checkable cheaply in the producer);
//   - |V| <= 2^100: the fp32 numerator is at most keys * max|V| (p <= 1) with keys <= 2^20.
// Inside it mc = round(m c) is within 2^-10 of m c, so the p of the running maximum is within 2^(+-2^-10) of 1
// (keys of a later step below an earlier maximum may all have p ~ 0: exact behaviour), and nothing overflows.
// Telemetry: bound_rows (if given) counts each (row, head, block) where some 64-key step had |max score * c| > 2^15
// or a NaN max; callers assert 0 per forward. Zero telemetry is NOT a certification of the contract (it only sees
// step maxima beyond 2^15): the gate is the producer's admission check (the H3 block forward measures the q / k
// row norms after the per-head RMSNorm + RoPE pass and throws above 2^14). Only head dim 128 is qualified.
// Environment switches (STRIX_H3_ATTN_EARLY / WAVES / PERSIST / PINGPONG / TUNE) accept only their documented
// values; anything else throws std::invalid_argument at first use.
#include <cstddef>
#include <cstdint>

namespace strix::h3::attn {

// One globally indexed key block: keys j = 0 .. keys-1 of this block (its first key is the block's first row).
struct KeyBlock {
    const std::uint16_t* k = nullptr;   // K[h][j][d]  = k  + h * k_hs  + j * k_rs  + d (bf16)
    const std::uint16_t* vt = nullptr;  // V[h][j][d]  = vt + h * vt_hs + d * vt_ds + j (bf16, keys contiguous)
    std::uint64_t k_hs = 0, k_rs = 0, vt_hs = 0, vt_ds = 0;  // vt_ds % 8 == 0 and vt_ds >= round_up(keys, 8)
    std::uint32_t keys = 0;
};

struct Queries {
    const std::uint16_t* q = nullptr;  // Q[h][r][d] = q + h * q_hs + r * q_rs + d (bf16)
    std::uint64_t q_hs = 0, q_rs = 0;
    std::uint32_t rows = 0, heads = 0;
    std::uint32_t dim = 128;  // head dim: 128 (DiT) or 64 (VAE decoder)
};

// Partials of nslots key blocks: o[slot][h][r][dim] fp32 (16-byte aligned), ml[slot][h][r][2] = (mc, l) fp32
// (8-byte aligned).
struct Partials {
    float* o = nullptr;
    float* ml = nullptr;
    std::uint32_t rows = 0, heads = 0;  // must match the Queries
    std::uint32_t dim = 128;            // o is [slot][h][r][dim]
    unsigned* bound_rows = nullptr;     // optional device counter: rows outside the input bound (atomic add)
};

// Output: bf16 out[r * out_rs + h * dim + d] (and optionally fp32 out_f32 at the same index).
struct Output {
    std::uint16_t* out = nullptr;
    float* out_f32 = nullptr;
    std::uint64_t out_rs = 0;           // >= heads * dim, a multiple of 8
    unsigned* bound_rows = nullptr;     // optional device counter (fused path)
};

constexpr unsigned kMaxBlocksPerLaunch = 4;

// Work-group size of the flash kernel: 16 waves (256 query rows, default) or 8 (128 rows); 0 = environment
// (STRIX_H3_ATTN_WAVES). Identical bits either way (each row's arithmetic is independent of the tile).
// Default 16 because a paired per-block bench (2-node per-block shape, alternating 5-s windows) measured a +4 % median
// over 8 waves; its 6 waves/SIMD is the register-derived occupancy ceiling (226 VGPRs), not a measured residency.
void set_waves_per_wg(unsigned waves);
// The work-group size the next launch uses (setter, else STRIX_H3_ATTN_WAVES, else the default 16).
unsigned waves_per_wg_resolved();
// Next-tile loads one full step ahead (1, the default), the late schedule (0), or the environment (-1,
// STRIX_H3_ATTN_EARLY=0 selects the late schedule). Same bits.
void set_early_loads(int early);
// Persistent round-robin schedule over (block, head, row tile) (1), grid per tile (0), environment (-1,
// STRIX_H3_ATTN_PERSIST). Same bits.
void set_persistent(int on);
// Two LDS tile buffers, one barrier per 64-key step (1), single buffer (0), environment (-1,
// STRIX_H3_ATTN_PINGPONG). Same bits.
void set_pingpong(int on);
// Scheduling variants of the default schedule (bit 0 VALU P^T build, bit 1 WMMA/fragment-read interleave), -1 =
// environment (STRIX_H3_ATTN_TUNE). Same bits.
void set_tune(int mask);
// One launch per (block, head) for blocks of more than 24576 keys (automatic, -1), always (1), never (0), or the
// environment (-2, STRIX_H3_ATTN_HEAD_SPLIT=0/1). Same bits.
void set_head_split(int mode);

// Partials of blocks[0 .. nblocks) into slots slot0 .. slot0 + nblocks - 1 (one launch, grid z = block).
void launch_partials(const Queries& q, const KeyBlock* blocks, unsigned nblocks, const Partials& p, unsigned slot0, void* stream);
// Bench only (wrong results): partials with parts of the kernel removed (1 softmax, 2 tile loads, 4 PV, 8 QK).
void launch_partials_ablate(const Queries& q, const KeyBlock* blocks, unsigned nblocks, const Partials& p, unsigned ablate, void* stream);
// Canonical merge of slots 0 .. nslots-1 (in that order) into the output.
void launch_merge(const Partials& p, unsigned nslots, const Output& out, void* stream);
// Fused one-block path: one key block, normalised output directly (bitwise == launch_partials + launch_merge(1)).
void launch_fused_single(const Queries& q, const KeyBlock& block, const Output& out, void* stream);
// V [h][j][d] (v + h * v_hs + j * v_rs + d) -> V^T [h][d][j] (vt + h * vt_hs + d * vt_ds + j), keys 0 .. keys-1.
void launch_transpose_v(const std::uint16_t* v, std::uint64_t v_hs, std::uint64_t v_rs, std::uint32_t keys, std::uint32_t heads,
                        std::uint16_t* vt, std::uint64_t vt_hs, std::uint64_t vt_ds, void* stream, std::uint32_t dim = 128);

// One-call form for the block forward. Q, K token-major [rows][heads][dim] (row pitch heads * dim), V^T
// [heads][dim][vt_pitch] covering the same key rows as K; key blocks b = [cuts[b], cuts[b + 1]) (cuts[0] = 0,
// cuts[nblocks] = keys; every cut a multiple of 64 so every node uses the same tile boundaries). Partials go to
// scratch (attention_scratch_floats(rows, heads, dim, nblocks) floats), then the canonical merge writes
// out[r * out_rs + h * dim + d] (bf16). On two nodes each node calls this with its own query rows and the
// same cut table, its K/V^T pointers covering both blocks (local + received) or use launch_partials per block.
struct AttentionCall {
    const std::uint16_t* q = nullptr;
    const std::uint16_t* k = nullptr;
    const std::uint16_t* vt = nullptr;
    std::uint32_t rows = 0, heads = 56, dim = 128, keys = 0;
    std::uint64_t vt_pitch = 0;          // keys per V^T row (>= round_up(keys, 8), multiple of 8)
    const std::uint32_t* cuts = nullptr;  // nblocks + 1 entries (host memory)
    std::uint32_t nblocks = 0;            // 1 .. kMaxBlocksPerLaunch
    float* scratch = nullptr;             // 16-byte aligned, attention_scratch_floats(...) floats
    std::uint16_t* out = nullptr;         // 16-byte aligned
    std::uint64_t out_rs = 0;             // 0: heads * dim; else >= heads * dim, a multiple of 8
    unsigned* bound_rows = nullptr;       // optional device counter of rows outside the input bound
};
std::size_t attention_scratch_floats(std::uint32_t rows, std::uint32_t heads, std::uint32_t dim, std::uint32_t nblocks);
void attention(const AttentionCall& c, void* stream);

}  // namespace strix::h3::attn
