#include "device.hpp"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

// Whole-chunk schedule of the canonical g4-v1 attention (gpu::attention).
// Every value is produced by the same separately rounded FP32 operations in
// the same order as the decode kernels; only the work distribution changes:
//  - score(row, key): lane L of the decode wave accumulates
//    sum = sum + q[L + 32j] * k[L + 32j] for ascending j from 0.0F, then the
//    wave tree adds lane L + d into lane L for d = 16, 8, 4, 2, 1. Here one
//    thread owns 4 heads x 4 keys and visits the 32 lane chains in post-order
//    of that tree (lane order bit-reversed), combining finished subtrees with
//    exactly the tree's additions and operand order.
//  - maximum over visible keys: any order gives the same exp arguments (the
//    only order-dependent case, +0 versus -0, gives identical declared_exp).
//  - denominator: per absolute 64-key chunk an ascending sum from 0.0F, merged
//    from 0.0F in ascending chunk order.
//  - numerator: per chunk an ascending sum of e * v from 0.0F, merged in chunk
//    order. Rows of a block that do not see a key use e = 0: with finite V the
//    product is a signed zero, and neither a chunk partial nor the merged sum
//    can be -0, so the extra additions return their left operand unchanged.
//    Non-finite V is only reachable when some row sees it, which raises the
//    same failure bit as the per-row path; such chunks are never published.
namespace strix::gemma4::prefill {
namespace {
constexpr unsigned block_positions = 16, key_tile = 64, threads = 256;
// Two 8-wave score blocks per CU (<= 192 VGPRs) keep one block computing while the
// other waits at a stage barrier.
#define G4_CANON_OCCUPANCY __attribute__((amdgpu_waves_per_eu(8)))

__device__ inline void flag_finite(float value, unsigned* failure) {
    if ((bit_cast<unsigned>(value) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
}
// Order-preserving integer image of a non-NaN float, so atomicMax computes the
// maximum of the visible scores. Only +0 versus -0 may differ from the decode
// kernels' first-wins scan, and declared_exp(s - (+0)) == declared_exp(s - (-0))
// for every s (s = -0 gives argument -0 or +0, both reduce to r = +0, n = 0).
__device__ inline unsigned ordered(float x) {
    const auto b = bit_cast<unsigned>(x);
    return b & 0x80000000U ? ~b : b | 0x80000000U;
}
__device__ inline float unordered(unsigned e) {
    return bit_cast<float>(e & 0x80000000U ? e & 0x7fffffffU : ~e);
}
constexpr unsigned ordered_negative_infinity = 0x007fffffU;
__device__ inline std::uint64_t first_visible(const gpu::KvRead& r, std::uint64_t position) {
    return !r.global && position >= r.window ? position - r.window + 1U : 0U;
}
// gpu::attention's kv_at, including its failure bit for a missing ring entry.
__device__ const float* canonical_at(const gpu::KvRead& r, std::uint64_t p, bool value, unsigned* failure) {
    if (p >= r.frontier) {
        if (p - r.frontier >= r.candidate_rows) { atomicOr(failure, 2U); return nullptr; }
        return (value ? r.overlay_v : r.overlay_k) + std::size_t(p - r.frontier) * r.width;
    }
    const auto slot = r.global ? p : p % r.slots;
    if (slot >= r.slots || r.tags[slot] != p) { atomicOr(failure, 2U); return nullptr; }
    return (value ? r.v : r.k) + std::size_t(slot) * r.width;
}
__device__ constexpr unsigned reverse5(unsigned t) {
    return ((t & 1U) << 4) | ((t & 2U) << 2) | (t & 4U) | ((t & 8U) >> 2) | ((t & 16U) >> 4);
}
using Tile = float[4][4];
__device__ inline void keep(Tile& to, const Tile& from) {
#pragma unroll
    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
        for (unsigned b = 0; b < 4; ++b) to[a][b] = from[a][b];
}
// value = left + value: the lower lane's subtree is the left operand.
__device__ inline void join(const Tile& left, Tile& value) {
#pragma unroll
    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
        for (unsigned b = 0; b < 4; ++b) value[a][b] = left[a][b] + value[a][b];
}

// Optional pre-pass for launch one: queries and the chunk's keys copied into
// post-order lane layout, element t * J + j holding component reverse5(t) + 32j,
// so a stage of lane chains is one contiguous 128-byte run per row or key.
// Keys outside the chunk's visible union [lo, last] are zero and absent.
template <unsigned Dim>
__global__ __launch_bounds__(threads) void canonical_pack_queries(const float* q, unsigned count, float* packed) {
    constexpr unsigned J = Dim / 32U;
    const auto o = std::size_t(blockIdx.x) * threads + threadIdx.x;
    if (o >= count) return;
    const auto w = unsigned(o % Dim);
    packed[o] = q[o - w + reverse5(w / J) + 32U * (w % J)];
}
template <unsigned Dim>
__global__ __launch_bounds__(threads) void canonical_pack_keys(gpu::KvRead r, std::uint64_t base, unsigned keys,
                                                               std::uint64_t lo, std::uint64_t last, float* packed,
                                                               unsigned* present, unsigned* failure) {
    constexpr unsigned J = Dim / 32U;
    const auto o = std::size_t(blockIdx.x) * threads + threadIdx.x;
    if (o >= std::size_t(keys) * r.width) return;
    const auto key = unsigned(o / r.width), w = unsigned(o % r.width), x = w % Dim;
    const auto p = base + key;
    const float* k = p >= lo && p <= last ? canonical_at(r, p, false, failure) : nullptr;
    packed[o] = k ? k[w - x + reverse5(x / J) + 32U * (x % J)] : 0.0F;
    if (!w) present[key] = k ? 1U : 0U;
}

// Launch one: grid (key tiles of the span, 16-position row tiles, KV heads).
// Writes the score of every visible (row, head, key); missing keys score +0
// (the decode wave sums nothing) and raise failure bit 2. Packed reads the
// pre-pass layouts (q is then the packed queries).
template <unsigned Dim, bool Packed>
__global__ __launch_bounds__(threads) G4_CANON_OCCUPANCY void canonical_scores(const float* q, const float* packed_keys,
                                                             const unsigned* present_keys, gpu::KvRead r,
                                                             unsigned rows, unsigned heads, std::uint64_t base,
                                                             unsigned width, float* scores, unsigned* maxima,
                                                             unsigned* failure, bool rows_major = false, unsigned row_group = 0) {
    constexpr unsigned J = Dim / 32U, Leaves = 32U / J, Stages = 32U / Leaves, S = 68U;
    static_assert(Leaves * J == 32U && Leaves >= 2U);
    __shared__ alignas(16) float qs[32U * S]; // [leaf * J + j][position * 4 + head], padded rows
    __shared__ alignas(16) float ks[32U * S]; // [leaf * J + j][key], padded rows
    __shared__ const float* keys[key_tile];
    __shared__ unsigned present[key_tile];
    const unsigned tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U;
    const unsigned tk = lane % 16U, tq = lane / 16U + 2U * wave;
    // rows_major: the grid runs row tiles fastest, so the row tiles of one key tile run together and share its keys
    // through the caches (same work per block, same bytes).
    // row_group > 0: a 1-D grid in groups of `row_group` row tiles; inside a group the key tiles advance after every row
    // tile of the group, so the group's queries and the current keys stay cache-resident (same bytes).
    unsigned key_block = rows_major ? blockIdx.y : blockIdx.x, row_block = rows_major ? blockIdx.x : blockIdx.y;
    if (row_group) {
        const unsigned row_tiles = (rows + block_positions - 1U) / block_positions, key_tiles = width / key_tile;
        const unsigned id = blockIdx.x, first = id / (row_group * key_tiles) * row_group, size = min(row_group, row_tiles - first);
        const unsigned within = id - first * key_tiles;
        row_block = first + within % size;
        key_block = within / size;
    }
    const unsigned row0 = row_block * block_positions, kv_head = blockIdx.z;
    const unsigned count = min(block_positions, rows - row0);
    const std::uint64_t first_position = r.position + row0, last_position = first_position + count - 1U;
    const std::uint64_t lo = first_visible(r, first_position);
    const std::uint64_t k0 = base + std::uint64_t(key_tile) * key_block;
    if (k0 > last_position || k0 + key_tile - 1U < lo) return;
    if (tid < key_tile) {
        const auto p = k0 + tid;
        if constexpr (Packed) present[tid] = present_keys[p - base];
        else {
            keys[tid] = p >= lo && p <= last_position ? canonical_at(r, p, false, failure) : nullptr;
            present[tid] = keys[tid] != nullptr;
        }
    }
    __syncthreads();
    const unsigned kv_offset = kv_head * Dim, head0 = kv_head * group;
    float sq[8], sk[8];
    const auto load = [&](unsigned stage) {
        if constexpr (Packed) {
#pragma unroll
            for (unsigned i = 0; i < 2; ++i) {
                const unsigned e = tid + threads * i, item = e / 8U, part = e % 8U;
                const unsigned position = item / 4U, head = item % 4U;
                float4 x = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
                if (position < count)
                    x = *reinterpret_cast<const float4*>(q + (std::size_t(row0 + position) * heads + head0 + head) * Dim +
                                                         stage * 32U + part * 4U);
                const float4 y = *reinterpret_cast<const float4*>(
                    packed_keys + std::size_t(k0 - base + item) * (heads / group * Dim) + kv_offset + stage * 32U + part * 4U);
                sq[4 * i] = x.x; sq[4 * i + 1] = x.y; sq[4 * i + 2] = x.z; sq[4 * i + 3] = x.w;
                sk[4 * i] = y.x; sk[4 * i + 1] = y.y; sk[4 * i + 2] = y.z; sk[4 * i + 3] = y.w;
            }
        } else {
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const unsigned e = tid + threads * i, slice = e / 64U;
                const unsigned d = reverse5(stage * Leaves + slice / J) + 32U * (slice % J);
                const unsigned position = (e / 4U) % 16U, head = e % 4U;
                sq[i] = position < count ? q[(std::size_t(row0 + position) * heads + head0 + head) * Dim + d] : 0.0F;
                const float* k = keys[e % 64U];
                sk[i] = k ? k[kv_offset + d] : 0.0F;
            }
        }
    };
    const auto store = [&] {
        if constexpr (Packed) {
#pragma unroll
            for (unsigned i = 0; i < 2; ++i) {
                const unsigned e = tid + threads * i, item = e / 8U, part = e % 8U;
#pragma unroll
                for (unsigned c = 0; c < 4; ++c) {
                    qs[(part * 4U + c) * S + item] = sq[4 * i + c];
                    ks[(part * 4U + c) * S + item] = sk[4 * i + c];
                }
            }
        } else {
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const unsigned e = tid + threads * i;
                qs[(e / 64U) * S + e % 64U] = sq[i];
                ks[(e / 64U) * S + e % 64U] = sk[i];
            }
        }
    };
    Tile value, s0, s1, s2, s3, s4;
    load(0);
    for (unsigned stage = 0; stage < Stages; ++stage) {
        __syncthreads();
        store();
        __syncthreads();
        if (stage + 1U < Stages) load(stage + 1U);
#pragma unroll
        for (unsigned leaf = 0; leaf < Leaves; ++leaf) {
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned b = 0; b < 4; ++b) value[a][b] = 0.0F;
#pragma unroll 4
            for (unsigned j = 0; j < J; ++j) {
                const float4 x = *reinterpret_cast<const float4*>(qs + (leaf * J + j) * S + tq * 4U);
                const float4 y = *reinterpret_cast<const float4*>(ks + (leaf * J + j) * S + 4U * tk);
                const float qv[4] = {x.x, x.y, x.z, x.w}, kv[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
                for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                    for (unsigned b = 0; b < 4; ++b) value[a][b] = value[a][b] + qv[a] * kv[b];
            }
            // Post-order position t of lane reverse5(t); t is uniform per block.
            const unsigned t = stage * Leaves + leaf;
            if (!(t & 1U)) keep(s0, value);
            else {
                join(s0, value);
                if (!(t & 2U)) keep(s1, value);
                else {
                    join(s1, value);
                    if (!(t & 4U)) keep(s2, value);
                    else {
                        join(s2, value);
                        if (!(t & 8U)) keep(s3, value);
                        else {
                            join(s3, value);
                            if (!(t & 16U)) keep(s4, value);
                            else join(s4, value);
                        }
                    }
                }
            }
        }
    }
    const bool real = tq < count;
    const auto position = first_position + tq, first = first_visible(r, position);
    const auto row = row0 + tq;
    float m[4];
#pragma unroll
    for (unsigned a = 0; a < 4; ++a) m[a] = bit_cast<float>(0xff800000U);
#pragma unroll
    for (unsigned b = 0; b < 4; ++b) {
        const auto p = k0 + 4U * tk + b;
        if (!real || p < first || p > position) continue;
        const bool present_key = present[4U * tk + b] != 0U;
#pragma unroll
        for (unsigned a = 0; a < 4; ++a) {
            const float s = present_key ? value[a][b] : 0.0F;
            flag_finite(s, failure);
            scores[(std::size_t(row) * heads + head0 + a) * width + (p - base)] = s;
            m[a] = s > m[a] ? s : m[a];
        }
    }
    if (!maxima) return;
    // NaN never wins, as in the decode scan; the 16 lanes of a position reduce.
#pragma unroll
    for (unsigned d = 1; d < 16U; d <<= 1U)
#pragma unroll
        for (unsigned a = 0; a < 4; ++a) {
            const float other = __shfl_xor(m[a], int(d), 32);
            m[a] = other > m[a] ? other : m[a];
        }
    if (real && !tk)
#pragma unroll
        for (unsigned a = 0; a < 4; ++a)
            if (m[a] > bit_cast<float>(0xff800000U)) atomicMax(maxima + std::size_t(row) * heads + head0 + a, ordered(m[a]));
}

// Launch two: one wave per (row, head). Maximum, e = declared_exp(s - m)
// written over the score, chunk denominators and their ordered merge.
__global__ __launch_bounds__(threads) void canonical_softmax(float* scores, float* denominators, gpu::KvRead r,
                                                             unsigned rows, unsigned heads, std::uint64_t base,
                                                             unsigned width) {
    const unsigned lane = threadIdx.x % 32U, rh = blockIdx.x * (threads / 32U) + threadIdx.x / 32U;
    if (rh >= rows * heads) return;
    const auto position = r.position + rh / heads, first = first_visible(r, position);
    float* s = scores + std::size_t(rh) * width;
    float m = bit_cast<float>(0xff800000U);
    for (auto p = first + lane; p <= position; p += 32U) {
        const float x = s[p - base];
        m = x > m ? x : m;
    }
    for (unsigned d = 16; d; d >>= 1U) {
        const float other = __shfl_xor(m, int(d), 32);
        m = other > m ? other : m;
    }
    const auto first_chunk = first / attention_chunk, last_chunk = position / attention_chunk;
    float denominator = 0.0F;
    for (auto c0 = first_chunk; c0 <= last_chunk; c0 += 32U) {
        const auto c = c0 + lane;
        float partial = 0.0F;
        if (c <= last_chunk) {
            const auto begin = max(c * attention_chunk, first), end = min((c + 1U) * attention_chunk, position + 1U);
            for (auto p = begin; p < end; ++p) {
                const float e = declared_exp(s[p - base] - m);
                s[p - base] = e;
                partial = partial + e;
            }
        }
        const auto n = unsigned(min(last_chunk - c0 + 1U, std::uint64_t(32)));
        for (unsigned l = 0; l < n; ++l) denominator = denominator + __shfl(partial, int(l), 32);
    }
    if (!lane) denominators[rh] = denominator;
}

// Launch three: grid (128-column slices, 16-position row tiles, KV heads).
// Each thread owns 4 heads x 8 columns of one position and streams the
// block's keys in ascending 32-key stages, two per absolute 64-key chunk.
// Fused: scores are raw and denominators holds ordered maxima; e is computed
// while staging and every block forms the 64 chunk denominators in LDS (the
// decode partial loop's order; invisible keys stage e = +0, which adds exactly
// nothing to a sum that is never -0).
// Parts = 2: one block of 2 x 256 threads covers 2 x 128 columns; the first part stages the probabilities (and
// their exp and chunk denominators) once for both, each part stages and multiplies its own values (same bytes).
// Tall (Parts = 2): the two parts take two consecutive 16-row tiles of the same 128 columns instead, so the
// value tile is staged once for 32 rows (half the value reads); each part keeps its own probabilities and
// denominators. The block runs the union of the two tiles' stages; a stage outside a tile's own range has
// every probability 0 there and adds +0, so every output keeps its operations and order. Same bytes.
template <unsigned Dim, bool Fused, unsigned Parts = 1, bool Tall = false>
__global__ __launch_bounds__(threads * Parts) G4_CANON_OCCUPANCY void canonical_values(const float* scores, const float* denominators,
                                                             gpu::KvRead r, unsigned rows, unsigned heads,
                                                             std::uint64_t base, unsigned width, float* out,
                                                             unsigned* failure, bool rows_major = false, unsigned row_group = 0) {
    constexpr unsigned Cols = 128, Keys = 32, EStride = 68;
    constexpr unsigned RowParts = Tall ? Parts : 1U, ColParts = Tall ? 1U : Parts, Staged = 16U / RowParts;
    __shared__ alignas(16) float es[RowParts][Keys * EStride]; // [row part][key][position * 4 + head], padded rows
    __shared__ alignas(16) float vs[ColParts][Keys * Cols];   // [column part][key][column]
    __shared__ const float* values[2][Keys];
    __shared__ float merged[RowParts][64];
    const unsigned part = threadIdx.x / threads, tid = threadIdx.x % threads, lane = tid % 32U, wave = tid / 32U;
    // own: this thread stages and reduces its part's probabilities (every part when Tall, the first otherwise).
    const bool lead = part == 0U, own = Tall || lead;
    const unsigned ep = Tall ? part : 0U, vp = Tall ? 0U : part, staging = Tall ? threadIdx.x : tid;
    const unsigned cg = lane % 16U, tq = lane / 16U + 2U * wave;
    // rows_major: row tiles fastest, so the row tiles of one column slice stream the same values together.
    unsigned slice = rows_major ? blockIdx.y : blockIdx.x, row_block = rows_major ? blockIdx.x : blockIdx.y;
    // row_group > 0: 1-D grid, groups of `row_group` row tiles x every column slice, rows fastest, so neighbouring
    // blocks share both the probabilities of their rows and the values of their slices (same bytes).
    if (row_group) {
        const unsigned row_tiles = (rows + block_positions * RowParts - 1U) / (block_positions * RowParts);
        const unsigned slices = Dim / (Cols * ColParts);
        const unsigned id = blockIdx.x, first = id / (row_group * slices) * row_group, size = min(row_group, row_tiles - first);
        const unsigned within = id - first * slices;
        row_block = first + within % size;
        slice = within / size;
    }
    const unsigned block_row0 = row_block * block_positions * RowParts, row0 = block_row0 + (Tall ? part : 0U) * block_positions;
    const unsigned kv_head = blockIdx.z, c0 = (slice * ColParts + (Tall ? 0U : part)) * Cols;
    const unsigned count = row0 < rows ? min(block_positions, rows - row0) : 0U;
    const unsigned block_count = min(block_positions * RowParts, rows - block_row0);
    const std::uint64_t first_position = r.position + row0;
    const std::uint64_t last_position = r.position + block_row0 + block_count - 1U; // the block's last row
    const std::uint64_t lo = first_visible(r, r.position + block_row0);
    const std::uint64_t first_key = lo / attention_chunk * attention_chunk;
    const auto stages = unsigned((last_position / attention_chunk + 1U) * attention_chunk - first_key) / Keys;
    const unsigned kv_offset = kv_head * Dim + c0, head0 = kv_head * group;
    const auto pointers = [&](unsigned stage) {
        if (lead && tid < Keys) {
            const auto p = first_key + std::uint64_t(stage) * Keys + tid;
            values[stage & 1U][tid] = p >= lo && p <= last_position ? canonical_at(r, p, true, failure) : nullptr;
        }
    };
    // Staging offsets are block-relative 32-bit values: key kp = p - first_key,
    // row position P = position - first_key; (row, key) is visible iff
    // kp <= P and (global or P - kp < window), the decode kernel's range.
    const auto relative0 = unsigned(first_position - first_key); // own tile
    const float* row_scores = scores + (first_key - base);
    float se[8], sv[Staged], maximum[8];
    if constexpr (Fused) {
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const unsigned qr = wave + 8U * i, position = qr / 4U, head = qr % 4U;
            maximum[i] = position < count
                ? unordered(bit_cast<unsigned>(denominators[(row0 + position) * heads + head0 + head])) : 0.0F;
        }
    }
    const auto seen = [&](unsigned stage, unsigned i) {
        const unsigned kp = stage * Keys + lane, qr = wave + 8U * i, position = qr / 4U, at = relative0 + position;
        return position < count && kp <= at && (r.global || at - kp < r.window);
    };
    const auto load = [&](unsigned stage) {
        const unsigned kp = stage * Keys + lane;
        if (own)
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const unsigned qr = wave + 8U * i, position = qr / 4U, head = qr % 4U;
                se[i] = seen(stage, i) ? row_scores[((row0 + position) * heads + head0 + head) * width + kp] : 0.0F;
            }
#pragma unroll
        for (unsigned i = 0; i < Staged; ++i) {
            const float* v = values[stage & 1U][staging / Cols + 2U * RowParts * i];
            sv[i] = v ? v[kv_offset + staging % Cols] : 0.0F;
        }
    };
    float partial[4][8], numerator[4][8];
#pragma unroll
    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
        for (unsigned b = 0; b < 8; ++b) { partial[a][b] = 0.0F; numerator[a][b] = 0.0F; }
    float denominator = 0.0F, chunk = 0.0F; // Fused: row tid < 64 of the block
    pointers(0);
    __syncthreads();
    load(0);
    for (unsigned stage = 0; stage < stages; ++stage) {
        __syncthreads();
        if (own)
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                // exp is applied here, a stage after the load was issued.
                if constexpr (Fused) se[i] = seen(stage, i) ? declared_exp(se[i] - maximum[i]) : 0.0F;
                es[ep][lane * EStride + wave + 8U * i] = se[i];
            }
#pragma unroll
        for (unsigned i = 0; i < Staged; ++i) vs[vp][staging + threads * RowParts * i] = sv[i];
        if (stage + 1U < stages) pointers(stage + 1U);
        __syncthreads();
        if (stage + 1U < stages) load(stage + 1U);
        if constexpr (Fused) {
            if (own && tid < 64U) {
#pragma unroll 8
                for (unsigned key = 0; key < Keys; ++key) chunk = chunk + es[ep][key * EStride + tid];
                if (stage & 1U) { denominator = denominator + chunk; chunk = 0.0F; }
            }
        }
#pragma unroll 2
        for (unsigned key = 0; key < Keys; ++key) {
            const float4 x = *reinterpret_cast<const float4*>(es[ep] + key * EStride + 4U * tq);
            const float4 y0 = *reinterpret_cast<const float4*>(vs[vp] + key * Cols + 8U * cg);
            const float4 y1 = *reinterpret_cast<const float4*>(vs[vp] + key * Cols + 8U * cg + 4U);
            const float ev[4] = {x.x, x.y, x.z, x.w}, vv[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned b = 0; b < 8; ++b) partial[a][b] = partial[a][b] + ev[a] * vv[b];
        }
        if (stage & 1U) {
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned b = 0; b < 8; ++b) {
                    flag_finite(partial[a][b], failure);
                    numerator[a][b] = numerator[a][b] + partial[a][b];
                    partial[a][b] = 0.0F;
                }
        }
    }
    if constexpr (Fused) {
        if (own && tid < 64U) merged[ep][tid] = denominator;
        __syncthreads();
    }
    if (tq >= count) return;
    const auto row = row0 + tq;
#pragma unroll
    for (unsigned a = 0; a < 4; ++a) {
        const float total = Fused ? merged[ep][4U * tq + a] : denominators[std::size_t(row) * heads + head0 + a];
#pragma unroll
        for (unsigned b = 0; b < 8; ++b) {
            const float value = numerator[a][b] / total;
            flag_finite(value, failure);
            out[(std::size_t(row) * heads + head0 + a) * Dim + c0 + 8U * cg + b] = value;
        }
    }
}
bool same_descriptor(const gpu::KvRead& a, const gpu::KvRead& b) {
    return a.k == b.k && a.v == b.v && a.tags == b.tags && a.overlay_k == b.overlay_k && a.overlay_v == b.overlay_v &&
           a.frontier == b.frontier && a.width == b.width && a.slots == b.slots && a.candidate_rows == b.candidate_rows &&
           a.window == b.window && a.global == b.global;
}
} // namespace

bool canonical_kernel_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_CANON_KERNEL");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
// Default on: score and value grids run row tiles fastest (same bytes). Measured v1 32K prefill attention
// 19.26 -> 16.51 s (1,037 -> 1,138 tok/s, profiled), 8K unchanged (1,400 ms). OFF: STRIX_G4_PREFILL_CANON_ROWS_MAJOR=0.
bool canonical_rows_major_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_CANON_ROWS_MAJOR");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
// STRIX_G4_PREFILL_CANON_GROUP=G (1..64): grouped score grid of G row tiles; default 8 (attention -5 % at 8K with
// the wide values), 0 = off.
unsigned canonical_group_rows() {
    static const unsigned g = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_CANON_GROUP");
        const long v = p ? std::strtol(p, nullptr, 10) : 8;
        return v > 0 && v <= 64 ? unsigned(v) : 0U;
    }();
    return g;
}
// STRIX_G4_PREFILL_CANON_VGROUP=G (1..64): grouped value grid, G row tiles x all column slices (0 = off).
unsigned canonical_value_group_rows() {
    static const unsigned g = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_CANON_VGROUP");
        const long v = p ? std::strtol(p, nullptr, 10) : 0;
        return v > 0 && v <= 64 ? unsigned(v) : 0U;
    }();
    return g;
}
// STRIX_G4_PREFILL_CANON_TALL=1: D512 values in 512-thread blocks of two 16-row tiles sharing one value tile
// (same bytes; default off).
bool canonical_tall_values_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_CANON_TALL");
        return p && std::string_view(p) == "1";
    }();
    return enabled;
}
// STRIX_G4_PREFILL_CANON_WIDE (default on, 0 = off): D512 values in 512-thread blocks of two 128-column parts
// sharing one probability tile (same bytes).
bool canonical_wide_values_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_CANON_WIDE");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool canonical_pack_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_CANON_PACK");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool canonical_fuse_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_CANON_FUSE");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
CanonicalSpan canonical_span(const gpu::KvRead& first_row, unsigned rows) {
    const auto first = !first_row.global && first_row.position >= first_row.window
                           ? first_row.position - first_row.window + 1U : 0U;
    CanonicalSpan s;
    s.base = first / attention_chunk * attention_chunk;
    const auto end = (first_row.position + rows + attention_chunk - 1U) / attention_chunk * attention_chunk;
    s.width = unsigned(end - s.base);
    return s;
}
std::size_t canonical_packed_floats(unsigned rows, unsigned heads, unsigned dim, unsigned span_width) {
    return std::size_t(rows) * heads * dim + std::size_t(span_width) * (heads / group * dim + 1U);
}
bool canonical_attention(const float* q, std::span<const gpu::KvRead> reads, unsigned heads, unsigned dim, float* out,
                         std::uint32_t* failure, gpu::Queue queue, const CanonicalScratch& scratch, int pack,
                         float* phase_ms, int fuse) {
    const auto rows = unsigned(reads.size());
    if (!rows || rows > max_chunk_rows || !heads || heads % group || !q || !out || !failure || !scratch.scores ||
        !scratch.denominators)
        throw std::invalid_argument("canonical attention geometry");
    if (dim != 256U && dim != 512U) return false;
    const auto& r = reads[0];
    if (r.position >= max_positions || rows > max_positions - r.position || r.width != heads / group * dim)
        throw std::invalid_argument("canonical attention descriptor");
    for (unsigned i = 1; i < rows; ++i)
        if (!same_descriptor(r, reads[i]) || reads[i].position != r.position + i)
            throw std::invalid_argument("canonical attention rows must be consecutive positions of one stream");
    const auto span = canonical_span(r, rows);
    if (std::size_t(rows) * heads * span.width > scratch.score_floats) throw std::invalid_argument("canonical attention scratch");
    const bool packed = pack < 0 ? canonical_pack_enabled() : pack != 0;
    const bool fused = fuse < 0 ? canonical_fuse_enabled() : fuse != 0;
    const std::size_t query_floats = std::size_t(rows) * heads * dim;
    if (packed && (!scratch.packed || canonical_packed_floats(rows, heads, dim, span.width) > scratch.packed_floats))
        throw std::invalid_argument("canonical attention packing scratch");
    const auto stream = static_cast<hipStream_t>(queue);
    hipEvent_t marks[5] = {};
    const auto mark = [&](unsigned i) {
        if (!phase_ms) return;
        if (!marks[i] && hipEventCreate(&marks[i]) != hipSuccess) throw std::runtime_error("canonical attention event");
        if (hipEventRecord(marks[i], stream) != hipSuccess) throw std::runtime_error("canonical attention event");
    };
    const bool rows_major = canonical_rows_major_enabled();
    const unsigned row_group = canonical_group_rows(), value_group = canonical_value_group_rows();
    const unsigned row_tiles = (rows + block_positions - 1U) / block_positions;
    const dim3 tiles = row_group ? dim3(row_tiles * (span.width / key_tile), 1, heads / group)
                       : rows_major ? dim3(row_tiles, span.width / key_tile, heads / group) : dim3(span.width / key_tile, row_tiles, heads / group);
    // TALL (D512) wins over WIDE when both are set.
    const bool tall = dim == 512U && canonical_tall_values_enabled();
    const unsigned parts = tall || (dim == 512U && canonical_wide_values_enabled()) ? 2U : 1U;
    const unsigned nslices = dim / (128U * (tall ? 1U : parts));
    const unsigned value_tiles = tall ? (rows + 2U * block_positions - 1U) / (2U * block_positions) : row_tiles;
    const dim3 slices = value_group ? dim3(value_tiles * nslices, 1, heads / group)
                        : rows_major ? dim3(value_tiles, nslices, heads / group) : dim3(nslices, value_tiles, heads / group);
    float* packed_queries = scratch.packed;
    float* packed_keys = packed ? scratch.packed + query_floats : nullptr;
    auto* present = packed ? reinterpret_cast<unsigned*>(packed_keys + std::size_t(span.width) * r.width) : nullptr;
    const std::uint64_t lo = !r.global && r.position >= r.window ? r.position - r.window + 1U : 0U;
    mark(0);
    // Fused: the denominators scratch first holds the ordered row maxima.
    unsigned* maxima = fused ? reinterpret_cast<unsigned*>(scratch.denominators) : nullptr;
    if (fused && hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(maxima), int(ordered_negative_infinity), std::size_t(rows) * heads,
                                   stream) != hipSuccess)
        throw std::runtime_error("canonical attention maxima");
    if (packed) {
        const auto keys = std::size_t(span.width) * r.width;
        if (dim == 256U) {
            hipLaunchKernelGGL(canonical_pack_queries<256>, dim3(unsigned((query_floats + threads - 1U) / threads)), dim3(threads), 0,
                               stream, q, unsigned(query_floats), packed_queries);
            hipLaunchKernelGGL(canonical_pack_keys<256>, dim3(unsigned((keys + threads - 1U) / threads)), dim3(threads), 0,
                               stream, r, span.base, span.width, lo, r.position + rows - 1U, packed_keys, present, failure);
        } else {
            hipLaunchKernelGGL(canonical_pack_queries<512>, dim3(unsigned((query_floats + threads - 1U) / threads)), dim3(threads), 0,
                               stream, q, unsigned(query_floats), packed_queries);
            hipLaunchKernelGGL(canonical_pack_keys<512>, dim3(unsigned((keys + threads - 1U) / threads)), dim3(threads), 0,
                               stream, r, span.base, span.width, lo, r.position + rows - 1U, packed_keys, present, failure);
        }
    }
    mark(1);
    const float* query = packed ? packed_queries : q;
#define G4_CANON_SCORES(D, P)                                                                                              \
    hipLaunchKernelGGL((canonical_scores<D, P>), tiles, dim3(threads), 0, stream, query, packed_keys, present, r, rows,   \
                       heads, span.base, span.width, scratch.scores, maxima, failure, rows_major, row_group)
    if (dim == 256U) { if (packed) G4_CANON_SCORES(256, true); else G4_CANON_SCORES(256, false); }
    else { if (packed) G4_CANON_SCORES(512, true); else G4_CANON_SCORES(512, false); }
#undef G4_CANON_SCORES
    mark(2);
    if (!fused)
        hipLaunchKernelGGL(canonical_softmax, dim3((rows * heads + threads / 32U - 1U) / (threads / 32U)), dim3(threads), 0,
                           stream, scratch.scores, scratch.denominators, r, rows, heads, span.base, span.width);
    mark(3);
#define G4_CANON_VALUES(D, F, N, T)                                                                                         \
    hipLaunchKernelGGL((canonical_values<D, F, N, T>), slices, dim3(threads * N), 0, stream, scratch.scores,              \
                       scratch.denominators, r, rows, heads, span.base, span.width, out, failure, rows_major, value_group)
    if (dim == 256U) { if (fused) G4_CANON_VALUES(256, true, 1, false); else G4_CANON_VALUES(256, false, 1, false); }
    else if (tall) { if (fused) G4_CANON_VALUES(512, true, 2, true); else G4_CANON_VALUES(512, false, 2, true); }
    else if (parts == 2U) { if (fused) G4_CANON_VALUES(512, true, 2, false); else G4_CANON_VALUES(512, false, 2, false); }
    else { if (fused) G4_CANON_VALUES(512, true, 1, false); else G4_CANON_VALUES(512, false, 1, false); }
#undef G4_CANON_VALUES
    mark(4);
    if (const auto e = hipGetLastError(); e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
    if (phase_ms) {
        if (hipEventSynchronize(marks[4]) != hipSuccess) throw std::runtime_error("canonical attention event");
        for (unsigned i = 0; i < 4; ++i) {
            float ms = 0.0F;
            (void)hipEventElapsedTime(&ms, marks[i], marks[i + 1]);
            phase_ms[i] += ms;
        }
        for (auto* e : marks) (void)hipEventDestroy(e);
    }
    return true;
}
} // namespace strix::gemma4::prefill
