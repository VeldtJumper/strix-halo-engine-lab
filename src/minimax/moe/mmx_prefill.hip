// minimax-m3 prefill MoE: grouped launches over all experts of a chunk (identity: mmx_prefill.hpp).
#include "minimax/moe/mmx_prefill.hpp"
#include "minimax/engine/route_identity.hpp"

#include "minimax/moe/mmx_prefill_core.hpp"
#include "minimax/moe/mmx_trellis_device.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace strix::minimax::moe::opt {

// Adopted strix-trellis 0.3 policy, shared with the host-only route census.
unsigned gemm_tile_rows(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    return route::gemm_tile_rows(input, output, bits, rows);
}
unsigned gemm_splits(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    return route::gemm_splits(input, output, bits, rows);
}

namespace {

namespace d = dev;

constexpr unsigned kHB = kHidden / 128, kIB = kInter / 128;  // 48, 12
constexpr float kAlpha = 1.702F, kLimit = 7.0F;

// ---------------------------------------------------------------- library WMMA core pieces (gemm.hip, trellis_device.hpp)
__device__ __forceinline__ float codebook_mul1(std::uint32_t state) {
    const std::uint32_t product = state * 0x83dcd12dU;
    const unsigned sum = __builtin_amdgcn_udot4(product, 0x01010101U, 0, false);
    return __half2float(__float2half_rn(fmaf(static_cast<float>(1024 + sum), __half2float(__ushort_as_half(0x1eee)),
                                             __half2float(__ushort_as_half(0xc931)))));
}
template <int Bits> struct WaveTile {
    std::uint32_t lo, hi;
    static constexpr int words = Bits * 8;
    __device__ explicit WaveTile(const std::uint32_t* data) {
        const int lane = static_cast<int>(threadIdx.x & 31U);
        lo = lane < words ? data[lane] : 0;
        if constexpr (words > 32) hi = lane + 32 < words ? data[lane + 32] : 0;
        else hi = 0;
    }
    __device__ __forceinline__ std::uint32_t word(int index) const {
        const std::uint32_t first = __shfl(lo, index & 31, 32);
        if constexpr (words > 32) {
            const std::uint32_t second = __shfl(hi, index & 31, 32);
            return index < 32 ? first : second;
        }
        return first;
    }
    __device__ __forceinline__ std::uint32_t state(int position) const {
        const int end = (position + 1) * Bits + 256 * Bits;
        const int start = end - 16;
        const std::uint32_t earlier = word((start / 32) % words);
        const std::uint32_t later = word(((end - 1) / 32) % words);
        const std::uint64_t pair = (static_cast<std::uint64_t>(earlier) << 32) | later;
        return static_cast<std::uint32_t>(pair >> ((32 - (end & 31)) & 31)) & 65535U;
    }
};
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));

// One grouped WMMA work item = one strix_gemm call's core grid (O/64 x ceil(rows/TileM) x splits CTAs).
struct WItem {
    const std::uint32_t* trellis;
    const _Float16* x;  // transformed rows [rows][input]
    float* part;        // [splits][rows][output]
    unsigned input, output, bits, rows, splits, cta_begin, gx, gy;
};

template <int Bits, int TileM>
__device__ void wmma_item(const WItem& it, unsigned bx, unsigned by, unsigned bz, _Float16* a, _Float16* q) {
    constexpr int AStride = 16;
    const int thread = static_cast<int>(threadIdx.x), lane = thread & 31, wave = thread / 32;
    const int first_row = static_cast<int>(by) * TileM;
    const int tile_n = static_cast<int>(bx) * 4 + wave;
    const int inputs = static_cast<int>(it.input), outputs = static_cast<int>(it.output), rows = static_cast<int>(it.rows);
    const int tiles_n = outputs / 16, tiles_k = inputs / 16;
    const int begin = static_cast<int>(static_cast<std::int64_t>(tiles_k) * bz / it.splits);
    const int end = static_cast<int>(static_cast<std::int64_t>(tiles_k) * (bz + 1) / it.splits);
    Float8 acc[TileM / 16] = {};
    for (int tile_k = begin; tile_k < end; ++tile_k) {
        for (int j = thread; j < TileM * 16; j += 128) {
            const int row = first_row + j / 16;
            a[(j / 16) * AStride + j % 16] =
                row < rows ? it.x[static_cast<std::size_t>(row) * static_cast<std::size_t>(inputs) + static_cast<std::size_t>(tile_k * 16 + j % 16)]
                           : static_cast<_Float16>(0);
        }
        if (tile_n < tiles_n) {
            const WaveTile<Bits> tile(it.trellis + (static_cast<std::size_t>(tile_k) * static_cast<std::size_t>(tiles_n) + static_cast<std::size_t>(tile_n)) * (Bits * 8));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r = 2 * (lane % 4) + (j % 2) + 8 * ((j / 2) % 2);
                const int c = lane / 4 + 8 * (j / 4);
                q[wave * 256 + r * 16 + c] = static_cast<_Float16>(codebook_mul1(tile.state(8 * lane + j)));
            }
        }
        __syncthreads();
        if (tile_n < tiles_n) {
            Half16 right;
#pragma unroll
            for (int k = 0; k < 16; ++k) right[k] = q[wave * 256 + k * 16 + (lane % 16)];
#pragma unroll
            for (int m = 0; m < TileM / 16; ++m) {
                Half16 left;
#pragma unroll
                for (int k = 0; k < 16; ++k) left[k] = a[(m * 16 + lane % 16) * AStride + k];
                acc[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(left, right, acc[m]);
            }
        }
        __syncthreads();
    }
    if (tile_n < tiles_n) {
#pragma unroll
        for (int m = 0; m < TileM / 16; ++m) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int row = first_row + m * 16 + 2 * j + lane / 16;
                if (row < rows)
                    it.part[(static_cast<std::size_t>(bz) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(row)) * static_cast<std::size_t>(outputs) +
                            static_cast<std::size_t>(tile_n * 16 + lane % 16)] = acc[m][j];
            }
        }
    }
}

template <int TileM>
__global__ __launch_bounds__(128) void wmma_grouped_kernel(const WItem* items, unsigned count) {
    __shared__ _Float16 a[TileM * 16];
    __shared__ _Float16 q[4 * 16 * 16];
    unsigned lo = 0, hi = count;  // last item with cta_begin <= blockIdx.x
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[mid].cta_begin <= blockIdx.x) lo = mid;
        else hi = mid;
    }
    const WItem it = items[lo];
    const unsigned local = blockIdx.x - it.cta_begin;
    const unsigned bx = local % it.gx, by = (local / it.gx) % it.gy, bz = local / (it.gx * it.gy);
    switch (it.bits) {
        case 3: wmma_item<3, TileM>(it, bx, by, bz, a, q); break;
        case 4: wmma_item<4, TileM>(it, bx, by, bz, a, q); break;
        case 5: wmma_item<5, TileM>(it, bx, by, bz, a, q); break;
        case 6: wmma_item<6, TileM>(it, bx, by, bz, a, q); break;
        default: wmma_item<8, TileM>(it, bx, by, bz, a, q); break;
    }
}

// ---------------------------------------------------------------- tables
struct PGroup {  // one expert (or the shared expert) of the chunk
    ExpertSlices w;
    unsigned n, pair_begin, s_gu, s_dn;
    unsigned tbegin;  // first row of the group in the tile-major (v4) xt / at layout, a multiple of 16
    std::size_t part_gu, part_dn;  // float offsets: gate [s_gu][n][1536], up right after; down [s_dn][n][6144]
};
struct Tables {
    const PGroup* groups;
    const std::uint32_t* pair_row;    // [P]
    const std::uint32_t* pair_group;  // [P]
    const std::uint32_t* tok_pair;    // [T][4], ascending expert id
    const float* tok_w;               // [T][4]
    _Float16* xt;                     // [2][P][6144] (gate, up)
    _Float16* at;                     // [P][1536]
    float* part;
    unsigned pairs, rows, shared_group;
    unsigned tpairs;  // padded rows of the tile-major layout (v4); xt = [2][tpairs][6144], at = [tpairs][1536]
};

// Tile-major position (v4) of row R (padded row index), column ch of a K-wide matrix: [R/16][ch/16][R%16][ch%16].
__device__ __forceinline__ std::size_t tiled_at(unsigned R, unsigned ch, unsigned K) {
    return ((static_cast<std::size_t>(R >> 4U) * (K >> 4U) + (ch >> 4U)) << 8U) + ((R & 15U) << 4U) + (ch & 15U);
}

__global__ __launch_bounds__(256) void gather_transform_kernel(const float* y, Tables t) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.pairs * 2 * kHB) return;
    const unsigned p = q / (2 * kHB), rem = q % (2 * kHB), m = rem / kHB, b = rem % kHB;
    const PGroup& g = t.groups[t.pair_group[p]];
    const std::uint16_t* suh = m == 0 ? g.w.gate.suh : g.w.up.suh;
    d::transform_block(y + static_cast<std::size_t>(t.pair_row[p]) * kHidden, suh,
                       t.xt + (static_cast<std::size_t>(m) * t.pairs + p) * kHidden, b * 128U, lane);
}

// v4: gather + input transform as gather_transform_kernel (transform_block's operation order), stored tile-major.
__global__ __launch_bounds__(256) void gather_transform_tiled_kernel(const float* y, Tables t) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.pairs * 2 * kHB) return;
    const unsigned p = q / (2 * kHB), rem = q % (2 * kHB), m = rem / kHB, b = rem % kHB;
    const PGroup& g = t.groups[t.pair_group[p]];
    const std::uint16_t* suh = m == 0 ? g.w.gate.suh : g.w.up.suh;
    const float* x = y + static_cast<std::size_t>(t.pair_row[p]) * kHidden;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
        const float scaled = d::materialize(x[c] * d::half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    d::hadamard128(v, lane);
    _Float16* out = t.xt + static_cast<std::size_t>(m) * t.tpairs * kHidden;
    const unsigned R = g.tbegin + (p - g.pair_begin);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[tiled_at(R, b * 128U + lane + 32U * static_cast<unsigned>(j), kHidden)] = static_cast<_Float16>(v[j]);
}

// v4, token-major: one wave = one (token, 128-block). It reads the token's fp32 block once and runs
// gather_transform_tiled_kernel's per-(pair, matrix) statements for the token's 4 routed pairs and its shared pair, gate
// then up. Same values, same stores (padded rows untouched); the input block is read once instead of 10 times.
__global__ __launch_bounds__(256) void gather_transform_token_kernel(const float* y, Tables t) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.rows * kHB) return;
    const unsigned tok = q / kHB, b = q % kHB;
    const float* x = y + static_cast<std::size_t>(tok) * kHidden;
    float xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = x[b * 128U + lane + 32U * static_cast<unsigned>(j)];
#pragma unroll 1
    for (unsigned k = 0; k <= kTopE; ++k) {
        const unsigned p = k < kTopE ? t.tok_pair[tok * kTopE + k] : t.groups[t.shared_group].pair_begin + tok;
        const PGroup& g = t.groups[t.pair_group[p]];
        const unsigned R = g.tbegin + (p - g.pair_begin);
#pragma unroll
        for (unsigned m = 0; m < 2; ++m) {
            const std::uint16_t* suh = m == 0 ? g.w.gate.suh : g.w.up.suh;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
                const float scaled = d::materialize(xv[j] * d::half_bits(suh[c]));
                v[j] = static_cast<float>(static_cast<_Float16>(scaled));
            }
            d::hadamard128(v, lane);
            _Float16* out = t.xt + static_cast<std::size_t>(m) * t.tpairs * kHidden;
#pragma unroll
            for (int j = 0; j < 4; ++j) out[tiled_at(R, b * 128U + lane + 32U * static_cast<unsigned>(j), kHidden)] = static_cast<_Float16>(v[j]);
        }
    }
}

__device__ __forceinline__ float swiglu(float g, float u) {  // tests/minimax_gate_common.hpp swiglu_kernel
    const float gg = g < kLimit ? g : kLimit;
    const float uu = u < -kLimit ? -kLimit : (u > kLimit ? kLimit : u);
    return gg * (1.0F / (1.0F + expf(-kAlpha * gg))) * (uu + 1.0F);
}

// output_transform of one 128-block of one row from split partials [splits][rows][output].
__device__ __forceinline__ void finish_block(const float* part, unsigned splits, unsigned rows, unsigned row, unsigned output,
                                             unsigned b, const std::uint16_t* svh, unsigned lane, float (&v)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = 0.0F;
    for (unsigned s = 0; s < splits; ++s)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] += part[(static_cast<std::size_t>(s) * rows + row) * output + b * 128 + lane + 32U * static_cast<unsigned>(j)];
    d::hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * d::half_bits(svh[b * 128 + lane + 32U * static_cast<unsigned>(j)]);
}

__global__ __launch_bounds__(256) void gate_up_finish_kernel(Tables t) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.pairs * kIB) return;
    const unsigned p = q / kIB, b = q % kIB;
    const PGroup& g = t.groups[t.pair_group[p]];
    const unsigned i = p - g.pair_begin;
    float gv[4], uv[4];
    finish_block(t.part + g.part_gu, g.s_gu, g.n, i, kInter, b, g.w.gate.svh, lane, gv);
    finish_block(t.part + g.part_gu + static_cast<std::size_t>(g.s_gu) * g.n * kInter, g.s_gu, g.n, i, kInter, b, g.w.up.svh, lane, uv);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = b * 128 + lane + 32U * static_cast<unsigned>(j);
        const float act = swiglu(gv[j], uv[j]);
        const float scaled = d::materialize(act * d::half_bits(g.w.down.suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    d::hadamard128(v, lane);
    _Float16* out = t.at + static_cast<std::size_t>(p) * kInter;
#pragma unroll
    for (int j = 0; j < 4; ++j) out[b * 128 + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
}

// Shared epilogue: preserve output transforms, SwiGLU and both FP16 conversions.
__device__ __forceinline__ void gate_up_tiled_store(Tables t, const PGroup& g, unsigned p, unsigned i,
                                                  unsigned b, unsigned lane, const float (&gv)[4],
                                                  const float (&uv)[4], float* debug) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = b * 128 + lane + 32U * static_cast<unsigned>(j);
        const float act = swiglu(gv[j], uv[j]);
        const float scaled = d::materialize(act * d::half_bits(g.w.down.suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
        if (debug != nullptr) {
            const std::size_t idx = static_cast<std::size_t>(p) * kInter + c;
            const std::size_t stride = static_cast<std::size_t>(t.pairs) * kInter;
            debug[idx] = gv[j];
            debug[stride + idx] = uv[j];
            debug[2 * stride + idx] = act;
            debug[3 * stride + idx] = v[j];
        }
    }
    d::hadamard128(v, lane);
    const unsigned R = g.tbegin + i;
#pragma unroll
    for (int j = 0; j < 4; ++j) t.at[tiled_at(R, b * 128 + lane + 32U * static_cast<unsigned>(j), kInter)] = static_cast<_Float16>(v[j]);
}

// v4: gate_up_finish_kernel with the down input stored tile-major.
__global__ __launch_bounds__(256) void gate_up_finish_tiled_kernel(Tables t, bool fused, float* debug) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.pairs * kIB) return;
    const unsigned p = q / kIB, b = q % kIB;
    const PGroup& g = t.groups[t.pair_group[p]];
    if (fused && g.s_gu == 1) return;
    const unsigned i = p - g.pair_begin;
    float gv[4], uv[4];
    finish_block(t.part + g.part_gu, g.s_gu, g.n, i, kInter, b, g.w.gate.svh, lane, gv);
    finish_block(t.part + g.part_gu + static_cast<std::size_t>(g.s_gu) * g.n * kInter, g.s_gu, g.n, i, kInter, b, g.w.up.svh, lane, uv);
    gate_up_tiled_store(t, g, p, i, b, lane, gv, uv, debug);
}

// Items are consecutive gate/up pairs, with identical row partitions and CTA ranges.
template <int MT>
__global__ __launch_bounds__(256) void gate_up_fused_kernel(const pfk::Item* items, unsigned count, Tables t,
                                                          bool partials, float* debug, bool producer_only) {
    __shared__ float drain[2 * 16 * 128];  // 16 KiB, reused for every sixteen-row slice
    unsigned lo = 0, hi = count;
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[2 * mid].cta_begin <= blockIdx.x) lo = mid;
        else hi = mid;
    }
    const pfk::Item gate = items[2 * lo];
    const unsigned matrix = threadIdx.x >> 7U;
    const pfk::Item it = items[2 * lo + matrix];
    const unsigned local = blockIdx.x - gate.cta_begin;
    const unsigned bx = local % gate.gx, by = local / gate.gx;
    // Only wave-local control flow differs for mixed gate/up bit widths. All waves
    // converge before the first CTA barrier in the drain loop.
    pfk::Accum<MT> acc;
    switch (it.bits) {
        case 3: acc = pfk::item_acc_v3<3, MT, 0, true, 0, true>(it, bx, by, 0); break;
        case 4: acc = pfk::item_acc_v3<4, MT, 0, true, 0, true>(it, bx, by, 0); break;
        case 5: acc = pfk::item_acc_v3<5, MT, 0, true, 0, true>(it, bx, by, 0); break;
        case 6: acc = pfk::item_acc_v3<6, MT, 0, true, 0, true>(it, bx, by, 0); break;
        default: acc = pfk::item_acc_v3<8, MT, 0, true, 0, true>(it, bx, by, 0); break;
    }
    unsigned gl = 0, gh = t.shared_group + 1;
    while (gh - gl > 1) {
        const unsigned mid = (gl + gh) / 2;
        if (t.part + t.groups[mid].part_gu <= gate.part) gl = mid;
        else gh = mid;
    }
    const PGroup& g = t.groups[gl];
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    const unsigned col = (wave & 3U) * 32U + (lane & 15U);
    const unsigned first = it.row0 + by * MT * 16U;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned r = 2U * static_cast<unsigned>(j) + (lane >> 4U);
            if (!producer_only) {
                drain[matrix * 2048U + r * 128U + col] = acc.acc0[m][j];
                drain[matrix * 2048U + r * 128U + col + 16U] = acc.acc1[m][j];
            }
            if (partials && first + 16U * static_cast<unsigned>(m) + r < it.rows) {
                float* dst = it.part + static_cast<std::size_t>(first + 16U * static_cast<unsigned>(m) + r) * kInter + bx * 128U + col;
                dst[0] = acc.acc0[m][j];
                dst[16] = acc.acc1[m][j];
            }
        }
        if (producer_only) continue;
        __syncthreads();
        for (unsigned r = wave; r < 16; r += 8) {
            const unsigned row = first + 16U * static_cast<unsigned>(m) + r;
            if (row < it.rows) {
                float gv[4], uv[4];
                // finish_block retains the initial +0.0F + partial, even for one split.
                finish_block(drain, 1, 16, r, 128, 0, g.w.gate.svh + bx * 128U, lane, gv);
                finish_block(drain + 2048, 1, 16, r, 128, 0, g.w.up.svh + bx * 128U, lane, uv);
                gate_up_tiled_store(t, g, g.pair_begin + row, row, bx, lane, gv, uv, debug);
            }
        }
        __syncthreads();  // all consumers finish before the next slice overwrites LDS
    }
}

__global__ __launch_bounds__(256) void down_combine_kernel(Tables t, float* out) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.rows * kHB) return;
    const unsigned tok = q / kHB, b = q % kHB;
    float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned k = 0; k < kTopE; ++k) {
        const unsigned p = t.tok_pair[tok * kTopE + k];
        const PGroup& g = t.groups[t.pair_group[p]];
        float v[4];
        finish_block(t.part + g.part_dn, g.s_dn, g.n, p - g.pair_begin, kHidden, b, g.w.down.svh, lane, v);
        const float w = t.tok_w[tok * kTopE + k];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = __builtin_fmaf(w, v[j], acc[j]);
    }
    const PGroup& sg = t.groups[t.shared_group];
    float v[4];
    finish_block(t.part + sg.part_dn, sg.s_dn, sg.n, tok, kHidden, b, sg.w.down.svh, lane, v);
#pragma unroll
    for (int j = 0; j < 4; ++j)
        out[static_cast<std::size_t>(tok) * kHidden + b * 128 + lane + 32U * static_cast<unsigned>(j)] = acc[j] + v[j];
}

// Dense prefill linears on the v4 core (enqueue_prefill_dense): the library input transform, row-major x -> tile-major
// f16 per job (transform_block's statements, one x read for every job), and the library output transform per job.
struct DenseArgs {
    const std::uint16_t* suh[kMaxDenseJobs];
    const std::uint16_t* svh[kMaxDenseJobs];
    _Float16* xt[kMaxDenseJobs];
    const float* part[kMaxDenseJobs];
    float* out[kMaxDenseJobs];
    unsigned n[kMaxDenseJobs], splits[kMaxDenseJobs];
    unsigned count, rows, k;
};

__global__ __launch_bounds__(256) void dense_transform_kernel(const float* x, DenseArgs a) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave, kb = a.k / 128U;
    if (q >= a.rows * kb) return;
    const unsigned row = q / kb, b = q % kb;
    const float* xr = x + static_cast<std::size_t>(row) * a.k;
    float xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = xr[b * 128U + lane + 32U * static_cast<unsigned>(j)];
#pragma unroll 1
    for (unsigned m = 0; m < a.count; ++m) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
            const float scaled = d::materialize(xv[j] * d::half_bits(a.suh[m][c]));
            v[j] = static_cast<float>(static_cast<_Float16>(scaled));
        }
        d::hadamard128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a.xt[m][tiled_at(row, b * 128U + lane + 32U * static_cast<unsigned>(j), a.k)] = static_cast<_Float16>(v[j]);
    }
}

__global__ __launch_bounds__(256) void dense_finish_kernel(DenseArgs a) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    unsigned q = blockIdx.x * 8U + wave, m = 0;
    for (; m < a.count; ++m) {  // job m owns rows * n / 128 consecutive waves
        const unsigned w = a.rows * (a.n[m] / 128U);
        if (q < w) break;
        q -= w;
    }
    if (m == a.count) return;
    const unsigned nb = a.n[m] / 128U, row = q / nb, b = q % nb;
    float v[4];
    finish_block(a.part[m], a.splits[m], a.rows, row, a.n[m], b, a.svh[m], lane, v);
#pragma unroll
    for (int j = 0; j < 4; ++j)
        a.out[m][static_cast<std::size_t>(row) * a.n[m] + b * 128U + lane + 32U * static_cast<unsigned>(j)] = v[j];
}

void launched(const char* what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax prefill moe ") + what + ": " + hipGetErrorString(e));
}
void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax prefill moe ") + what + ": " + hipGetErrorString(e));
}

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }

std::size_t table_bound(std::uint32_t rows) {
    const std::size_t pairs = 5ULL * rows, groups = kExperts + 1;
    return align256(groups * sizeof(PGroup)) + 2 * align256(pairs * 4) +
           2 * align256(4ULL * rows * 4) + align256(groups * std::max(3 * sizeof(WItem), 6 * sizeof(pfk::Item)));
}

// Module state is used by one engine host thread; setters, enqueue and cleanup must not race.
// Slots protect only H2D sources. Device workspace reuse must be ordered on the caller's stream.
constexpr unsigned N_SLOTS = 4;
struct TableSlot {
    unsigned char* host = nullptr;
    std::size_t capacity = 0;
    hipEvent_t copied = nullptr;
    bool pending = false;
};
TableSlot g_table_slots[N_SLOTS];
unsigned g_table_next = 0;
int g_table_ring_override = -1;
bool table_ring_default() {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_MM_PF_TABLE_RING");
        if (e == nullptr || std::string(e) == "0") return false;
        if (std::string(e) == "1") return true;
        throw std::invalid_argument("STRIX_MM_PF_TABLE_RING must be 0 or 1");
    }();
    return on;
}
bool table_ring() { return g_table_ring_override < 0 ? table_ring_default() : g_table_ring_override != 0; }
TableSlot& next_table_slot(std::uint32_t rows) {
    TableSlot& slot = g_table_slots[g_table_next];
    if (slot.pending) {
        ok(hipEventSynchronize(slot.copied), "table slot wait");
        slot.pending = false;
    }
    // Reserve the bound's table prefix, never its large activation/partial regions.
    const std::size_t capacity = table_bound(std::max(rows, 4096U));
    if (slot.copied == nullptr) ok(hipEventCreateWithFlags(&slot.copied, hipEventDisableTiming), "table event");
    if (slot.capacity < capacity) {
        if (slot.host != nullptr) {
            ok(hipHostFree(slot.host), "table slot free");
            slot.host = nullptr;
            slot.capacity = 0;
        }
        ok(hipHostMalloc(reinterpret_cast<void**>(&slot.host), capacity), "table slot malloc");
        slot.capacity = capacity;
    }
    g_table_next = (g_table_next + 1) % N_SLOTS;
    return slot;
}

// Host plan of one call: groups, pair/token tables and the workspace layout.
struct Plan {
    std::vector<PGroup> groups;
    std::vector<std::uint32_t> pair_row, pair_group, tok_pair;
    std::vector<float> tok_w;
    std::vector<WItem> gu_items, dn_items16, dn_items64;
    unsigned gu_ctas = 0, dn_ctas16 = 0, dn_ctas64 = 0;
    std::size_t part_floats = 0;
    unsigned tpairs = 0;  // padded rows of the tile-major layout (sum of the groups' rows rounded up to 16)
    std::size_t off_groups = 0, off_pair_row = 0, off_pair_group = 0, off_tok_pair = 0, off_tok_w = 0, off_items = 0,
                off_xt = 0, off_at = 0, off_part = 0, bytes = 0;
};

Plan make_plan(const PrefillMoeView& v) {
    if (v.rows == 0) throw std::invalid_argument("minimax prefill moe: rows must be > 0");
    if (v.ids == nullptr || v.weights == nullptr || v.experts == nullptr || v.y == nullptr || v.out == nullptr)
        throw std::invalid_argument("minimax prefill moe: null pointer in view");
    const unsigned T = v.rows;
    Plan p;
    const auto erows = route::expert_rows(std::span<const std::uint32_t>(v.ids, std::size_t{T} * kTopE), T, kExperts, kTopE);
    std::vector<std::uint32_t> group_of(kExperts, ~0U);
    auto add_group = [&](const ExpertSlices& w, const std::vector<std::uint32_t>& rows) {
        PGroup g{};
        g.w = w;
        g.n = static_cast<unsigned>(rows.size());
        g.pair_begin = static_cast<unsigned>(p.pair_row.size());
        g.tbegin = p.tpairs;
        p.tpairs += (g.n + 15U) / 16U * 16U;
        g.s_gu = gemm_splits(kHidden, kInter, w.gate.bits, g.n);
        g.s_dn = gemm_splits(kInter, kHidden, w.down.bits, g.n);
        if (gemm_splits(kHidden, kInter, w.up.bits, g.n) != g.s_gu || gemm_tile_rows(kHidden, kInter, w.gate.bits, g.n) != 16 ||
            gemm_tile_rows(kHidden, kInter, w.up.bits, g.n) != 16)
            throw std::invalid_argument("minimax prefill moe: gate/up policy differs between the two matrices");
        g.part_gu = p.part_floats;
        p.part_floats += 2 * static_cast<std::size_t>(g.s_gu) * g.n * kInter;
        g.part_dn = p.part_floats;
        p.part_floats += static_cast<std::size_t>(g.s_dn) * g.n * kHidden;
        const auto gi = static_cast<std::uint32_t>(p.groups.size());
        for (const std::uint32_t r : rows) {
            p.pair_row.push_back(r);
            p.pair_group.push_back(gi);
        }
        p.groups.push_back(g);
        return gi;
    };
    for (unsigned e = 0; e < kExperts; ++e)
        if (!erows[e].empty()) group_of[e] = add_group(v.experts[e], erows[e]);
    std::vector<std::uint32_t> all(T);
    for (unsigned t = 0; t < T; ++t) all[t] = t;
    const std::uint32_t shared_group = add_group(v.shared, all);
    // Token tables: the 4 pairs of each token in ascending expert id (its position in each expert's row list).
    p.tok_pair.resize(std::size_t{T} * kTopE);
    p.tok_w.resize(std::size_t{T} * kTopE);
    std::vector<std::uint32_t> cursor(kExperts, 0);
    for (unsigned t = 0; t < T; ++t) {
        std::uint32_t id[kTopE];
        float w[kTopE];
        for (unsigned k = 0; k < kTopE; ++k) {
            id[k] = v.ids[std::size_t{t} * kTopE + k];
            w[k] = v.weights[std::size_t{t} * kTopE + k];
        }
        for (unsigned i = 1; i < kTopE; ++i)
            for (unsigned j = i; j > 0 && id[j - 1] > id[j]; --j) {
                std::swap(id[j], id[j - 1]);
                std::swap(w[j], w[j - 1]);
            }
        for (unsigned k = 0; k < kTopE; ++k) {
            const PGroup& g = p.groups[group_of[id[k]]];
            p.tok_pair[std::size_t{t} * kTopE + k] = g.pair_begin + cursor[id[k]]++;
            p.tok_w[std::size_t{t} * kTopE + k] = w[k];
        }
    }
    // Workspace layout.
    const std::size_t P = p.pair_row.size();
    std::size_t off = 0;
    auto take = [&](std::size_t bytes) {
        const std::size_t at = off;
        off += align256(bytes);
        return at;
    };
    p.off_groups = take(p.groups.size() * sizeof(PGroup));
    p.off_pair_row = take(P * 4);
    p.off_pair_group = take(P * 4);
    p.off_tok_pair = take(p.tok_pair.size() * 4);
    p.off_tok_w = take(p.tok_w.size() * 4);
    const std::size_t item_bytes = std::max(3 * sizeof(WItem), 6 * sizeof(pfk::Item));
    p.off_items = take(p.groups.size() * item_bytes);
    const std::size_t PX = std::max<std::size_t>(P, p.tpairs);  // the lib / v2 / v3 row-major or the v4 tiled layout
    p.off_xt = take(2 * PX * kHidden * 2);
    p.off_at = take(PX * kInter * 2);
    p.off_part = take(p.part_floats * 4);
    p.bytes = off;
    (void)shared_group;
    return p;
}

// STRIX_MMX_PREFILL_TILE: 0 (default) = the library's row tile per call (16 or 64 rows per CTA), 128 = 128 rows per
// CTA for every item. Same bits: every 16-row WMMA of a row tile is independent of the others, so grouping more row
// tiles per CTA only reuses the decoded weight tile (one weight pass per 128 rows instead of per 16).
unsigned prefill_tile() {
    static const unsigned t = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_TILE");
        const unsigned x = e != nullptr ? static_cast<unsigned>(std::atoi(e)) : 0U;
        if (x != 0 && x != 128) throw std::invalid_argument("STRIX_MMX_PREFILL_TILE must be 0 or 128");
        return x;
    }();
    return t;
}

// Prefill WMMA core: 0 = the library port above (wmma_item), 1 / 2 / 3 = mmx_prefill_core.hpp v2 /
// v3 / v4 (same bits). STRIX_MMX_PREFILL_CORE=v4 (default) | v3 | v2 | lib sets the process default; set_prefill_core
// overrides it (tests, ABA).
int g_core_override = -1;
int g_mtfit_override = -1;
int g_core8_override = -1;
int g_pipe_override = -1;
int g_tail_stream_override = -1;
int g_gufuse_override = -1;
int g_gather_tok_override = -1;
bool g_gufuse_debug = false;
bool g_gufuse_producer_only = false;
float* g_gufuse_intermediates = nullptr;
PrefillMarker g_marker = nullptr;  // profiling: called with a label after each launch (set_prefill_marker)
void* g_marker_user = nullptr;
void mark(const char* label, hipStream_t stream) {
    if (g_marker != nullptr) g_marker(label, stream, g_marker_user);
}
unsigned prefill_core() {
    if (g_core_override >= 0) return static_cast<unsigned>(g_core_override);
    static const unsigned c = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_CORE");
        // Default v4 (measurement: bitwise everywhere incl. every split partial; whole call 0.34-0.58x lib, 0.69-0.99x v3, 0.68-0.89x v2).
        if (e == nullptr || std::string(e) == "v4") return 3U;
        if (std::string(e) == "lib") return 0U;
        if (std::string(e) == "v2") return 1U;
        if (std::string(e) == "v3") return 2U;
        throw std::invalid_argument("STRIX_MMX_PREFILL_CORE must be lib, v2, v3 or v4");
    }();
    return c;
}

bool prefill_mtfit_default() {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_MTFIT");
        return e == nullptr || std::string(e) != "0";  // default ON (prefill measurement: exact); =0 off
    }();
    return on;
}
bool prefill_mtfit() {
    return g_mtfit_override < 0 ? prefill_mtfit_default() : g_mtfit_override != 0;
}

int prefill_core8_default() {
    static const int value = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_CORE8");
        if (e == nullptr || std::string(e) == "0") return 0;
        if (std::string(e) == "1") return 1;
        if (std::string(e) == "2") return 2;
        throw std::invalid_argument("STRIX_MMX_PREFILL_CORE8 must be 0, 1 or 2");
    }();
    return value;
}
int prefill_core8() { return g_core8_override < 0 ? prefill_core8_default() : g_core8_override; }

int prefill_pipe_default() {
    static const int value = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_PIPE");
        if (e == nullptr) return 1;  // default PIPE=1 (measurement: exact, 8K B/A 0.973/0.935/0.924; b1 bench bitwise)
        if (std::string(e) == "0") return 0;
        if (std::string(e) == "1") return 1;
        if (std::string(e) == "2") return 2;
        throw std::invalid_argument("STRIX_MMX_PREFILL_PIPE must be 0, 1 or 2");
    }();
    return value;
}
int prefill_pipe() { return g_pipe_override < 0 ? prefill_pipe_default() : g_pipe_override; }

// STRIX_MMX_PREFILL_TAIL_STREAM (v4 + mtfit): 1 (default) = the tail-class launches (MT 1..7) of each stage run on a side
// stream beside the MT8 launch, joined before the stage's consumer; 0 = every class launch on the caller's stream.
// Same items, kernels and bits; the classes write disjoint partials.
bool prefill_tail_stream_default() {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_TAIL_STREAM");
        if (e == nullptr || std::string(e) == "1") return true;
        if (std::string(e) == "0") return false;
        throw std::invalid_argument("STRIX_MMX_PREFILL_TAIL_STREAM must be 0 or 1");
    }();
    return on;
}
bool prefill_tail_stream() { return g_tail_stream_override < 0 ? prefill_tail_stream_default() : g_tail_stream_override != 0; }

bool prefill_gufuse_default() {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_GUFUSE");
        if (e == nullptr || std::string(e) == "0") return false;
        if (std::string(e) == "1") return true;
        throw std::invalid_argument("STRIX_MMX_PREFILL_GUFUSE must be 0 or 1");
    }();
    return on;
}
bool prefill_gufuse() { return g_gufuse_override < 0 ? prefill_gufuse_default() : g_gufuse_override != 0; }

// STRIX_MMX_PREFILL_GATHER_TOK: 1 (default) = v4 gather in token-major order (gather_transform_token_kernel),
// 0 = one wave per (pair, matrix, block). Same values and stores.
bool prefill_gather_tok_default() {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_GATHER_TOK");
        if (e == nullptr || std::string(e) == "1") return true;
        if (std::string(e) == "0") return false;
        throw std::invalid_argument("STRIX_MMX_PREFILL_GATHER_TOK must be 0 or 1");
    }();
    return on;
}
bool prefill_gather_tok() { return g_gather_tok_override < 0 ? prefill_gather_tok_default() : g_gather_tok_override != 0; }

// Gate/up and down lists: v4 classes are MT - 1; older cores retain the four power-of-two classes.
struct V2Items {
    std::vector<pfk::Item> gu[8], dn[8], fused[8];
    unsigned gu_ctas[8] = {}, dn_ctas[8] = {}, fused_ctas[8] = {};
};
V2Items fill_items_v2(const Plan& p, unsigned char* ws, bool tiled, bool mtfit, bool gufuse) {
    V2Items v;
    auto* xt = reinterpret_cast<_Float16*>(ws + p.off_xt);
    auto* at = reinterpret_cast<_Float16*>(ws + p.off_at);
    auto* part = reinterpret_cast<float*>(ws + p.off_part);
    const std::size_t P = p.pair_row.size();
    for (const PGroup& g : p.groups) {
        const unsigned cls = pfk::mt_class(g.n), tm = 16U * pfk::mt_of_class(cls);
        const bool fuse = gufuse && g.s_gu == 1;
        // Splits and part pointers always describe the whole group.
        auto append = [&](pfk::Item it, bool down) {
            auto* lists = down ? v.dn : fuse ? v.fused : v.gu;
            auto* ctas = down ? v.dn_ctas : fuse ? v.fused_ctas : v.gu_ctas;
            auto emit = [&](unsigned c, unsigned row0, unsigned gy) {
                it.row0 = row0;
                it.gy = gy;
                it.cta_begin = ctas[c];
                ctas[c] += it.gx * gy * it.splits;
                lists[c].push_back(it);
                if (!down && fuse) {
                    pfk::Item up = it;
                    up.trellis = g.w.up.trellis;
                    up.x += static_cast<std::size_t>(p.tpairs) * kHidden;
                    up.part += static_cast<std::size_t>(g.n) * kInter;
                    up.bits = g.w.up.bits;
                    lists[c].push_back(up);
                }
            };
            if (tiled && mtfit) {
                const auto fit = route::mtfit(g.n);
                if (fit.main != 0) emit(7, 0, fit.main);
                if (fit.remainder != 0) emit(fit.tail_class, fit.main * 128, 1);
            } else {
                emit(tiled ? pfk::mt_of_class(cls) - 1 : cls, 0, (g.n + tm - 1) / tm);
            }
        };
        for (unsigned m = 0; m < (fuse ? 1U : 2U); ++m) {
            const Exl3& w = m == 0 ? g.w.gate : g.w.up;
            pfk::Item it{};
            it.trellis = w.trellis;
            it.x = tiled ? xt + (static_cast<std::size_t>(m) * p.tpairs + g.tbegin) * kHidden
                         : xt + (static_cast<std::size_t>(m) * P + g.pair_begin) * kHidden;
            it.part = part + g.part_gu + static_cast<std::size_t>(m) * g.s_gu * g.n * kInter;
            it.input = kHidden;
            it.output = kInter;
            it.bits = w.bits;
            it.rows = g.n;
            it.splits = g.s_gu;
            it.gx = kInter / 128;
            append(it, false);
        }
        pfk::Item it{};
        it.trellis = g.w.down.trellis;
        it.x = at + static_cast<std::size_t>(tiled ? g.tbegin : g.pair_begin) * kInter;
        it.part = part + g.part_dn;
        it.input = kInter;
        it.output = kHidden;
        it.bits = g.w.down.bits;
        it.rows = g.n;
        it.splits = g.s_dn;
        it.gx = kHidden / 128;
        append(it, true);
    }
    return v;
}

// STRIX_MMX_PREFILL_TAIL_STREAM side stream (non-blocking) and its fork / join events, created on first use.
hipStream_t g_tail_side = nullptr;
hipEvent_t g_tail_forked = nullptr, g_tail_joined = nullptr;
void tail_fork(hipStream_t stream) {
    if (g_tail_side == nullptr) {  // all three or none: a failed creation leaves nothing half-initialized
        hipStream_t side = nullptr;
        hipEvent_t forked = nullptr, joined = nullptr;
        hipError_t e = hipStreamCreateWithFlags(&side, hipStreamNonBlocking);
        if (e == hipSuccess) e = hipEventCreateWithFlags(&forked, hipEventDisableTiming);
        if (e == hipSuccess) e = hipEventCreateWithFlags(&joined, hipEventDisableTiming);
        if (e != hipSuccess) {
            if (joined != nullptr) (void)hipEventDestroy(joined);
            if (forked != nullptr) (void)hipEventDestroy(forked);
            if (side != nullptr) (void)hipStreamDestroy(side);
            ok(e, "tail side stream / events");
        }
        g_tail_side = side;
        g_tail_forked = forked;
        g_tail_joined = joined;
    }
    ok(hipEventRecord(g_tail_forked, stream), "tail fork");
    ok(hipStreamWaitEvent(g_tail_side, g_tail_forked, 0), "tail fork wait");
}
void tail_join(hipStream_t stream) {
    ok(hipEventRecord(g_tail_joined, g_tail_side), "tail join");
    ok(hipStreamWaitEvent(stream, g_tail_joined, 0), "tail join wait");
}

constexpr const char* kGuLabel[8] = {"gate_up.mt1", "gate_up.mt2", "gate_up.mt3", "gate_up.mt4",
                                   "gate_up.mt5", "gate_up.mt6", "gate_up.mt7", "gate_up.mt8"};
constexpr const char* kDnLabel[8] = {"down.mt1", "down.mt2", "down.mt3", "down.mt4",
                                   "down.mt5", "down.mt6", "down.mt7", "down.mt8"};
template <int V>
void launch_class(const pfk::Item* items_d, unsigned n, unsigned ctas, unsigned cls, hipStream_t stream) {
    if constexpr (V == 4) {
        if (cls == 7 && prefill_pipe() != 0) {
            if (prefill_pipe() == 1)
                hipLaunchKernelGGL((pfk::grouped_pipe_kernel<8, 4, false>), dim3(ctas), dim3(128), 0, stream, items_d, n);
            else
                hipLaunchKernelGGL((pfk::grouped_pipe_kernel<8, 4, true>), dim3(ctas), dim3(128), 0, stream, items_d, n);
            return;
        }
        if (cls == 7 && prefill_core8() != 0) {
            if (prefill_core8() == 1)
                hipLaunchKernelGGL((pfk::grouped_core8_kernel<1>), dim3(ctas), dim3(128), 0, stream, items_d, n);
            else
                hipLaunchKernelGGL((pfk::grouped_core8_kernel<2>), dim3(ctas), dim3(128), 0, stream, items_d, n);
            return;
        }
        switch (cls) {
            case 0: hipLaunchKernelGGL((pfk::grouped_kernel<1, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
            case 1: hipLaunchKernelGGL((pfk::grouped_kernel<2, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
            case 2: hipLaunchKernelGGL((pfk::grouped_kernel<3, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
            case 3: hipLaunchKernelGGL((pfk::grouped_kernel<4, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
            case 4: hipLaunchKernelGGL((pfk::grouped_kernel<5, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
            case 5: hipLaunchKernelGGL((pfk::grouped_kernel<6, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
            case 6: hipLaunchKernelGGL((pfk::grouped_kernel<7, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
            case 7: hipLaunchKernelGGL((pfk::grouped_kernel<8, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        }
        return;
    }
    switch (cls) {
        case 0: hipLaunchKernelGGL((pfk::grouped_kernel<1, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 1: hipLaunchKernelGGL((pfk::grouped_kernel<2, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 2: hipLaunchKernelGGL((pfk::grouped_kernel<4, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        default: hipLaunchKernelGGL((pfk::grouped_kernel<8, V>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
    }
}
// core: 1 = v2, 2 = v3, 3 = v4, 8 / 9 = v3 diagnostics (no A loads / no decode)
void launch_v2(const pfk::Item* items_d, const std::vector<pfk::Item>& list, unsigned ctas, unsigned cls, unsigned core, hipStream_t stream) {
    if (list.empty()) return;
    const auto n = static_cast<unsigned>(list.size());
    switch (core) {
        case 1: launch_class<2>(items_d, n, ctas, cls, stream); break;
        case 2: launch_class<3>(items_d, n, ctas, cls, stream); break;
        case 3: launch_class<4>(items_d, n, ctas, cls, stream); break;
        case 8: launch_class<31>(items_d, n, ctas, cls, stream); break;
        case 9: launch_class<32>(items_d, n, ctas, cls, stream); break;
        default: throw std::invalid_argument("minimax prefill moe: unknown core " + std::to_string(core));
    }
}

template <int MT>
void launch_gufuse(const pfk::Item* items, unsigned count, unsigned ctas, Tables t, hipStream_t stream) {
    static const bool qualified = [] {
        hipFuncAttributes attrs{};
        ok(hipFuncGetAttributes(&attrs, reinterpret_cast<const void*>(gate_up_fused_kernel<MT>)), "gufuse resources");
        if (attrs.localSizeBytes != 0) throw std::runtime_error("minimax prefill gufuse: scratch/spilling rejected");
        if (attrs.sharedSizeBytes > 16384) throw std::runtime_error("minimax prefill gufuse: LDS budget exceeded");
        return true;
    }();
    (void)qualified;
    hipLaunchKernelGGL((gate_up_fused_kernel<MT>), dim3(ctas), dim3(256), 0, stream,
                       items, count, t, g_gufuse_debug, g_gufuse_intermediates, g_gufuse_producer_only);
}
void launch_gufuse_class(const pfk::Item* items, unsigned count, unsigned ctas, unsigned cls, Tables t, hipStream_t stream) {
    switch (cls) {
        case 0: launch_gufuse<1>(items, count, ctas, t, stream); break;
        case 1: launch_gufuse<2>(items, count, ctas, t, stream); break;
        case 2: launch_gufuse<3>(items, count, ctas, t, stream); break;
        case 3: launch_gufuse<4>(items, count, ctas, t, stream); break;
        case 4: launch_gufuse<5>(items, count, ctas, t, stream); break;
        case 5: launch_gufuse<6>(items, count, ctas, t, stream); break;
        case 6: launch_gufuse<7>(items, count, ctas, t, stream); break;
        case 7: launch_gufuse<8>(items, count, ctas, t, stream); break;
    }
}

void fill_items(Plan& p, unsigned char* ws) {
    const unsigned override_tile = prefill_tile();
    auto* xt = reinterpret_cast<_Float16*>(ws + p.off_xt);
    auto* at = reinterpret_cast<_Float16*>(ws + p.off_at);
    auto* part = reinterpret_cast<float*>(ws + p.off_part);
    const std::size_t P = p.pair_row.size();
    for (const PGroup& g : p.groups) {
        for (unsigned m = 0; m < 2; ++m) {
            const Exl3& w = m == 0 ? g.w.gate : g.w.up;
            WItem it{};
            it.trellis = w.trellis;
            it.x = xt + (static_cast<std::size_t>(m) * P + g.pair_begin) * kHidden;
            it.part = part + g.part_gu + static_cast<std::size_t>(m) * g.s_gu * g.n * kInter;
            it.input = kHidden;
            it.output = kInter;
            it.bits = w.bits;
            it.rows = g.n;
            it.splits = g.s_gu;
            it.gx = (kInter / 16 + 3) / 4;
            it.gy = (g.n + (override_tile ? override_tile : 16U) - 1) / (override_tile ? override_tile : 16U);
            it.cta_begin = p.gu_ctas;
            p.gu_ctas += it.gx * it.gy * it.splits;
            p.gu_items.push_back(it);
        }
        const unsigned tile = override_tile ? override_tile : gemm_tile_rows(kInter, kHidden, g.w.down.bits, g.n);
        WItem it{};
        it.trellis = g.w.down.trellis;
        it.x = at + static_cast<std::size_t>(g.pair_begin) * kInter;
        it.part = part + g.part_dn;
        it.input = kInter;
        it.output = kHidden;
        it.bits = g.w.down.bits;
        it.rows = g.n;
        it.splits = g.s_dn;
        it.gx = (kHidden / 16 + 3) / 4;
        it.gy = (g.n + tile - 1) / tile;
        unsigned& ctas = tile != 16 ? p.dn_ctas64 : p.dn_ctas16;
        it.cta_begin = ctas;
        ctas += it.gx * it.gy * it.splits;
        (tile != 16 ? p.dn_items64 : p.dn_items16).push_back(it);
    }
}

}  // namespace

void set_prefill_table_ring(int on) {
    if (on < -1 || on > 1) throw std::invalid_argument("set_prefill_table_ring: value must be -1, 0 or 1");
    (void)table_ring_default();  // capture once, including at Engine startup
    g_table_ring_override = on;
}
void cleanup_prefill_table_ring() noexcept {
    auto checked = [](hipError_t error) {
        if (error != hipSuccess) {
            std::fprintf(stderr, "minimax prefill moe table cleanup: %s\n", hipGetErrorString(error));
            std::abort();  // never free an in-flight source after a failed wait
        }
    };
    for (TableSlot& slot : g_table_slots) {
        if (slot.pending) checked(hipEventSynchronize(slot.copied));
        if (slot.host != nullptr) checked(hipHostFree(slot.host));
        if (slot.copied != nullptr) checked(hipEventDestroy(slot.copied));
        slot = {};
    }
    g_table_next = 0;
}

void set_prefill_core(int core) {  // exact cores only
    if (core < -1 || core > 3) throw std::invalid_argument("set_prefill_core: core must be -1, 0, 1, 2 or 3");
    g_core_override = core;
}
void set_prefill_mtfit(int on) {
    if (on < -1 || on > 1) throw std::invalid_argument("set_prefill_mtfit: value must be -1, 0 or 1");
    (void)prefill_mtfit_default();  // capture the environment once, including at Engine startup
    g_mtfit_override = on;
}
int prefill_pipe_override() { return g_pipe_override; }
void set_prefill_pipe(int value) {
    if (value < -1 || value > 2) throw std::invalid_argument("set_prefill_pipe: value must be -1, 0, 1 or 2");
    (void)prefill_pipe_default();
    g_pipe_override = value;
}
void set_prefill_tail_stream(int on) {
    if (on < -1 || on > 1) throw std::invalid_argument("set_prefill_tail_stream: value must be -1, 0 or 1");
    (void)prefill_tail_stream_default();  // capture once, including at Engine startup
    g_tail_stream_override = on;
}
void cleanup_prefill_tail_stream() noexcept {
    if (g_tail_side != nullptr) (void)hipStreamSynchronize(g_tail_side);
    for (hipEvent_t* ev : {&g_tail_forked, &g_tail_joined}) {
        if (*ev != nullptr) (void)hipEventDestroy(*ev);
        *ev = nullptr;
    }
    if (g_tail_side != nullptr) (void)hipStreamDestroy(g_tail_side);
    g_tail_side = nullptr;
}
void set_prefill_core8(int value) {
    if (value < -1 || value > 2) throw std::invalid_argument("set_prefill_core8: value must be -1, 0, 1 or 2");
    (void)prefill_core8_default();  // capture once, including at Engine startup
    g_core8_override = value;
}
void set_prefill_gufuse(int on) {
    if (on < -1 || on > 1) throw std::invalid_argument("set_prefill_gufuse: value must be -1, 0 or 1");
    (void)prefill_gufuse_default();
    g_gufuse_override = on;
}
void set_prefill_gather_tok(int on) {
    if (on < -1 || on > 1) throw std::invalid_argument("set_prefill_gather_tok: value must be -1, 0 or 1");
    (void)prefill_gather_tok_default();
    g_gather_tok_override = on;
}
void set_prefill_gufuse_debug(bool on, float* intermediates, bool producer_only) {
    g_gufuse_debug = on || producer_only;
    g_gufuse_intermediates = intermediates;
    g_gufuse_producer_only = producer_only;
}
void set_prefill_diagnostic_core(int core) {  // bench only: wrong outputs by design
    if (core != 8 && core != 9) throw std::invalid_argument("set_prefill_diagnostic_core: core must be 8 or 9");
    g_core_override = core;
}
void set_prefill_marker(PrefillMarker marker, void* user) {
    g_marker = marker;
    g_marker_user = user;
}

PrefillDebugRegions prefill_moe_debug_regions(const PrefillMoeView& v) {
    const Plan p = make_plan(v);
    PrefillDebugRegions r;
    r.xt_offset = p.off_xt;
    r.xt_bytes = 2 * p.pair_row.size() * kHidden * 2;
    r.pairs = p.pair_row.size();
    r.tpairs = p.tpairs;
    for (const PGroup& g : p.groups) r.groups.push_back({g.pair_begin, g.tbegin, g.n});
    r.at_offset = p.off_at;
    r.at_bytes = p.pair_row.size() * kInter * 2;
    r.part_offset = p.off_part;
    r.part_bytes = p.part_floats * 4;
    return r;
}

std::size_t prefill_moe_workspace_bytes(const PrefillMoeView& v) { return make_plan(v).bytes; }

std::size_t prefill_moe_workspace_bound(std::uint32_t rows) {
    // Pairs 5T; per group splits x rows <= max(8 x 16, rows) for gate/up and <= max(2 x 16, rows) for down (the policy
    // stops doubling once ceil(rows/16) x O/64 x splits >= 160), so partial floats <= 129 x 128 x (2 x 1536 + 6144)
    // + 5T x (2 x 1536 + 6144), plus tables.
    const std::size_t P = 5ULL * rows, groups = kExperts + 1;
    std::size_t b = 0;
    b += table_bound(rows);
    const std::size_t PX = P + 15ULL * groups;  // v4 pads every group to 16 rows
    b += align256(2 * PX * kHidden * 2) + align256(PX * kInter * 2);
    b += align256((groups * 128 + P) * (2ULL * kInter + kHidden) * 4);
    return b;
}

void enqueue_prefill_moe(const PrefillMoeView& v, void* workspace, std::size_t workspace_bytes, hipStream_t stream) {
    Plan p = make_plan(v);
    if (workspace == nullptr || workspace_bytes < p.bytes) throw std::invalid_argument("minimax prefill moe: workspace too small");
    auto* ws = static_cast<unsigned char*>(workspace);
    const unsigned core = prefill_core();
    const bool gufuse = core == 3 && v.rows > 8 && prefill_gufuse();
    const bool v2 = core != 0;  // the grouped v2 item layout (v2, v3 and the v3 diagnostics)
    V2Items v2i;
    std::vector<WItem> items;
    std::vector<pfk::Item> items2;  // gate/up classes, then down classes (same table region)
    if (v2) {
        v2i = fill_items_v2(p, ws, core == 3, core == 3 && prefill_mtfit(), gufuse);
        for (unsigned c = 0; c < 8; ++c) items2.insert(items2.end(), v2i.gu[c].begin(), v2i.gu[c].end());
        for (unsigned c = 0; c < 8; ++c) items2.insert(items2.end(), v2i.fused[c].begin(), v2i.fused[c].end());
        for (unsigned c = 0; c < 8; ++c) items2.insert(items2.end(), v2i.dn[c].begin(), v2i.dn[c].end());
    } else {
        fill_items(p, ws);
        // Tables: groups, pairs, tokens, items (gate/up, then down tile 16, then down tile 64).
        items = p.gu_items;
        items.insert(items.end(), p.dn_items16.begin(), p.dn_items16.end());
        items.insert(items.end(), p.dn_items64.begin(), p.dn_items64.end());
    }
    TableSlot* slot = table_ring() ? &next_table_slot(v.rows) : nullptr;
    if (slot != nullptr && p.off_xt > slot->capacity) throw std::runtime_error("minimax prefill moe: table bound exceeded");
    auto copy = [&](std::size_t offset, const void* source, std::size_t bytes, const char* label) {
        if (slot != nullptr) {
            if (offset > slot->capacity || bytes > slot->capacity - offset)
                throw std::runtime_error("minimax prefill moe: table slot overflow");
            std::memcpy(slot->host + offset, source, bytes);
            source = slot->host + offset;
        }
        ok(hipMemcpyAsync(ws + offset, source, bytes, hipMemcpyHostToDevice, stream), label);
    };
    try {
        copy(p.off_groups, p.groups.data(), p.groups.size() * sizeof(PGroup), "groups");
        copy(p.off_pair_row, p.pair_row.data(), p.pair_row.size() * 4, "pairs");
        copy(p.off_pair_group, p.pair_group.data(), p.pair_group.size() * 4, "pairs");
        copy(p.off_tok_pair, p.tok_pair.data(), p.tok_pair.size() * 4, "tokens");
        copy(p.off_tok_w, p.tok_w.data(), p.tok_w.size() * 4, "tokens");
        if (v2) copy(p.off_items, items2.data(), items2.size() * sizeof(pfk::Item), "items");
        else copy(p.off_items, items.data(), items.size() * sizeof(WItem), "items");
        if (slot != nullptr) {
            ok(hipEventRecord(slot->copied, stream), "table copies event");
            slot->pending = true;
        } else {
            ok(hipStreamSynchronize(stream), "tables");  // the host vectors die with this call
        }
    } catch (...) {
        // A partial upload or failed record has no completion event; retain sources until drained.
        if (hipStreamSynchronize(stream) != hipSuccess) std::abort();
        throw;
    }
    Tables t{};
    t.groups = reinterpret_cast<const PGroup*>(ws + p.off_groups);
    t.pair_row = reinterpret_cast<const std::uint32_t*>(ws + p.off_pair_row);
    t.pair_group = reinterpret_cast<const std::uint32_t*>(ws + p.off_pair_group);
    t.tok_pair = reinterpret_cast<const std::uint32_t*>(ws + p.off_tok_pair);
    t.tok_w = reinterpret_cast<const float*>(ws + p.off_tok_w);
    t.xt = reinterpret_cast<_Float16*>(ws + p.off_xt);
    t.at = reinterpret_cast<_Float16*>(ws + p.off_at);
    t.part = reinterpret_cast<float*>(ws + p.off_part);
    t.pairs = static_cast<unsigned>(p.pair_row.size());
    t.rows = v.rows;
    t.shared_group = static_cast<unsigned>(p.groups.size() - 1);
    t.tpairs = p.tpairs;
    const bool tiled = core == 3;
    const auto* items_d = reinterpret_cast<const WItem*>(ws + p.off_items);
    const auto n_gu = static_cast<unsigned>(p.gu_items.size()), n16 = static_cast<unsigned>(p.dn_items16.size()),
               n64 = static_cast<unsigned>(p.dn_items64.size());
    mark("start", stream);
    if (tiled && prefill_gather_tok()) hipLaunchKernelGGL(gather_transform_token_kernel, dim3((v.rows * kHB + 7) / 8), dim3(256), 0, stream, v.y, t);
    else if (tiled) hipLaunchKernelGGL(gather_transform_tiled_kernel, dim3((t.pairs * 2 * kHB + 7) / 8), dim3(256), 0, stream, v.y, t);
    else hipLaunchKernelGGL(gather_transform_kernel, dim3((t.pairs * 2 * kHB + 7) / 8), dim3(256), 0, stream, v.y, t);
    launched("gather transform");
    mark("gather", stream);
    if (v2) {
        const auto* it_d = reinterpret_cast<const pfk::Item*>(ws + p.off_items);
        std::size_t at = 0;
        // One stage's class launches; with the tail stream, classes 0..6 go to the side stream beside class 7.
        const bool tail_stream = core == 3 && prefill_mtfit() && prefill_tail_stream();
        auto stage = [&](const std::vector<pfk::Item>* lists, const unsigned* ctas, const char* const* labels) {
            bool tails = false;
            for (unsigned c = 0; c < 7; ++c) tails = tails || !lists[c].empty();
            const bool side = tail_stream && tails && !lists[7].empty();
            if (side) tail_fork(stream);
            std::size_t c_at = at;
            try {
                for (unsigned c = 0; c < 7 && side; ++c) {
                    launch_v2(it_d + c_at, lists[c], ctas[c], c, core, g_tail_side);
                    c_at += lists[c].size();
                }
                for (unsigned c = side ? 7U : 0U; c < 8; ++c) {
                    launch_v2(it_d + c_at, lists[c], ctas[c], c, core, stream);
                    c_at += lists[c].size();
                    if (!lists[c].empty()) mark(labels[core == 3 ? c : pfk::mt_of_class(c) - 1], stream);
                }
            } catch (...) {
                if (side) tail_join(stream);  // side work stays inside the caller's completion chain
                throw;
            }
            if (side) {
                tail_join(stream);
                mark(labels[6], stream);  // the joined tail classes (v4 only)
            }
            at = c_at;
        };
        stage(v2i.gu, v2i.gu_ctas, kGuLabel);
        for (unsigned c = 0; c < 8; ++c) {
            if (!v2i.fused[c].empty()) {
                launch_gufuse_class(it_d + at, static_cast<unsigned>(v2i.fused[c].size() / 2), v2i.fused_ctas[c], c, t, stream);
                mark(kGuLabel[c], stream);
            }
            at += v2i.fused[c].size();
        }
        launched("gate/up v2");
        if (tiled) hipLaunchKernelGGL(gate_up_finish_tiled_kernel, dim3((t.pairs * kIB + 7) / 8), dim3(256), 0, stream, t, gufuse && !g_gufuse_producer_only, g_gufuse_intermediates);
        else hipLaunchKernelGGL(gate_up_finish_kernel, dim3((t.pairs * kIB + 7) / 8), dim3(256), 0, stream, t);
        launched("gate/up finish");
        mark("finish", stream);
        stage(v2i.dn, v2i.dn_ctas, kDnLabel);
        launched("down v2");
        hipLaunchKernelGGL(down_combine_kernel, dim3((v.rows * kHB + 7) / 8), dim3(256), 0, stream, t, v.out);
        launched("down combine");
        mark("combine", stream);
        return;
    }
    const bool big = prefill_tile() == 128;  // every item then sits in the 'tile 64' list of the down stage
    if (big) hipLaunchKernelGGL(wmma_grouped_kernel<128>, dim3(p.gu_ctas), dim3(128), 0, stream, items_d, n_gu);
    else hipLaunchKernelGGL(wmma_grouped_kernel<16>, dim3(p.gu_ctas), dim3(128), 0, stream, items_d, n_gu);
    launched("gate/up");
    mark("gate_up", stream);
    hipLaunchKernelGGL(gate_up_finish_kernel, dim3((t.pairs * kIB + 7) / 8), dim3(256), 0, stream, t);
    launched("gate/up finish");
    mark("finish", stream);
    if (n16 > 0) {
        hipLaunchKernelGGL(wmma_grouped_kernel<16>, dim3(p.dn_ctas16), dim3(128), 0, stream, items_d + n_gu, n16);
        launched("down 16");
    }
    if (n64 > 0) {
        if (big) hipLaunchKernelGGL(wmma_grouped_kernel<128>, dim3(p.dn_ctas64), dim3(128), 0, stream, items_d + n_gu + n16, n64);
        else hipLaunchKernelGGL(wmma_grouped_kernel<64>, dim3(p.dn_ctas64), dim3(128), 0, stream, items_d + n_gu + n16, n64);
        launched("down 64");
    }
    mark("down", stream);
    hipLaunchKernelGGL(down_combine_kernel, dim3((v.rows * kHB + 7) / 8), dim3(256), 0, stream, t, v.out);
    launched("down combine");
    mark("combine", stream);
}

namespace {
// Item list of one dense call: at most kMaxDenseJobs jobs x (main + tail) items.
constexpr std::size_t kDenseItems = 2 * kMaxDenseJobs;
struct DensePlan {
    unsigned k = 0, tpad = 0;
    unsigned splits[kMaxDenseJobs] = {};
    std::size_t off_items = 0, off_xt[kMaxDenseJobs] = {}, off_part[kMaxDenseJobs] = {}, bytes = 0;
};
bool dense_bits_supported(unsigned bits) { return bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8; }
DensePlan make_dense_plan(const PrefillDenseJob* jobs, unsigned count, unsigned rows) {
    if (jobs == nullptr || count == 0 || count > kMaxDenseJobs || rows == 0)
        throw std::invalid_argument("minimax prefill dense: bad job count or rows");
    DensePlan p;
    p.k = jobs[0].w.input;
    p.tpad = (rows + 15U) / 16U * 16U;
    if (p.k == 0 || p.k % 128 != 0) throw std::invalid_argument("minimax prefill dense: input must be a multiple of 128");
    std::size_t off = 0;
    p.off_items = off;
    off += align256(kDenseItems * sizeof(pfk::Item));
    for (unsigned m = 0; m < count; ++m) {
        const Exl3& w = jobs[m].w;
        if (w.input != p.k || w.output == 0 || w.output % 128 != 0 || w.trellis == nullptr || w.suh == nullptr ||
            w.svh == nullptr || jobs[m].out == nullptr)
            throw std::invalid_argument("minimax prefill dense: jobs need one input width and outputs % 128 == 0");
        if (!dense_bits_supported(w.bits)) throw std::invalid_argument("minimax prefill dense: unsupported bit width");
        p.splits[m] = gemm_splits(w.input, w.output, w.bits, rows);
        p.off_xt[m] = off;
        off += align256(std::size_t{p.tpad} * p.k * 2);
        p.off_part[m] = off;
        off += align256(std::size_t{p.splits[m]} * rows * w.output * 4);
    }
    p.bytes = off;
    return p;
}
// Pinned host staging for the item lists: the H2D copy is stream-ordered into the call's workspace (consumed by the
// call's own launches before any later call on that stream overwrites it); a slot is reused only after its copy event.
struct DenseSlot {
    pfk::Item* host = nullptr;
    hipEvent_t copied = nullptr;
    bool pending = false;
};
constexpr unsigned kDenseSlots = 16;
DenseSlot g_dense_slots[kDenseSlots];
unsigned g_dense_next = 0;
std::mutex g_dense_mutex;  // reserve -> fill -> copy -> record, and cleanup: one transaction per thread
DenseSlot& next_dense_slot() {
    DenseSlot& slot = g_dense_slots[g_dense_next];
    g_dense_next = (g_dense_next + 1) % kDenseSlots;
    if (slot.host == nullptr) {
        ok(hipHostMalloc(reinterpret_cast<void**>(&slot.host), kDenseItems * sizeof(pfk::Item), hipHostMallocDefault), "dense items host");
        ok(hipEventCreateWithFlags(&slot.copied, hipEventDisableTiming), "dense items event");
    }
    if (slot.pending) {
        ok(hipEventSynchronize(slot.copied), "dense items slot reuse");
        slot.pending = false;
    }
    return slot;
}
}  // namespace

bool prefill_dense_supported(const Exl3& w) { return dense_bits_supported(w.bits) && w.input % 128 == 0 && w.output % 128 == 0; }

std::size_t prefill_dense_workspace_bytes(const PrefillDenseJob* jobs, unsigned count, unsigned rows) {
    return make_dense_plan(jobs, count, rows).bytes;
}

void enqueue_prefill_dense(const PrefillDenseJob* jobs, unsigned count, const float* x, unsigned rows, void* workspace,
                           std::size_t workspace_bytes, hipStream_t stream) {
    const DensePlan p = make_dense_plan(jobs, count, rows);
    if (x == nullptr || workspace == nullptr || workspace_bytes < p.bytes)
        throw std::invalid_argument("minimax prefill dense: null input or workspace too small");
    auto* ws = static_cast<unsigned char*>(workspace);
    std::vector<pfk::Item> lists[8];
    unsigned ctas[8] = {};
    for (unsigned m = 0; m < count; ++m) {
        const Exl3& w = jobs[m].w;
        pfk::Item it{};
        it.trellis = w.trellis;
        it.x = reinterpret_cast<_Float16*>(ws + p.off_xt[m]);
        it.part = reinterpret_cast<float*>(ws + p.off_part[m]);
        it.input = w.input;
        it.output = w.output;
        it.bits = w.bits;
        it.rows = rows;
        it.splits = p.splits[m];
        it.gx = w.output / 128;
        auto emit = [&](unsigned c, unsigned row0, unsigned gy) {
            it.row0 = row0;
            it.gy = gy;
            it.cta_begin = ctas[c];
            ctas[c] += it.gx * gy * it.splits;
            lists[c].push_back(it);
        };
        const auto fit = route::mtfit(rows);  // the production v4 + mtfit item shapes
        if (fit.main != 0) emit(7, 0, fit.main);
        if (fit.remainder != 0) emit(fit.tail_class, fit.main * 128, 1);
    }
    std::lock_guard<std::mutex> staging(g_dense_mutex);
    DenseSlot& slot = next_dense_slot();
    unsigned at[8] = {}, used = 0;
    for (unsigned c = 0; c < 8; ++c) {
        at[c] = used;
        for (const pfk::Item& it : lists[c]) slot.host[used++] = it;
    }
    auto* items_d = reinterpret_cast<pfk::Item*>(ws + p.off_items);
    ok(hipMemcpyAsync(items_d, slot.host, used * sizeof(pfk::Item), hipMemcpyHostToDevice, stream), "dense items");
    ok(hipEventRecord(slot.copied, stream), "dense items event");
    slot.pending = true;
    DenseArgs a{};
    a.count = count;
    a.rows = rows;
    a.k = p.k;
    unsigned waves = 0;
    for (unsigned m = 0; m < count; ++m) {
        a.suh[m] = jobs[m].w.suh;
        a.svh[m] = jobs[m].w.svh;
        a.xt[m] = reinterpret_cast<_Float16*>(ws + p.off_xt[m]);
        a.part[m] = reinterpret_cast<const float*>(ws + p.off_part[m]);
        a.out[m] = jobs[m].out;
        a.n[m] = jobs[m].w.output;
        a.splits[m] = p.splits[m];
        waves += rows * (jobs[m].w.output / 128);
    }
    hipLaunchKernelGGL(dense_transform_kernel, dim3((rows * (p.k / 128) + 7) / 8), dim3(256), 0, stream, x, a);
    launched("dense transform");
    for (unsigned c = 0; c < 8; ++c)
        if (!lists[c].empty()) launch_class<4>(items_d + at[c], static_cast<unsigned>(lists[c].size()), ctas[c], c, stream);
    launched("dense v4");
    hipLaunchKernelGGL(dense_finish_kernel, dim3((waves + 7) / 8), dim3(256), 0, stream, a);
    launched("dense finish");
}

void cleanup_prefill_dense() noexcept {
    std::lock_guard<std::mutex> staging(g_dense_mutex);
    for (DenseSlot& slot : g_dense_slots) {
        if (slot.pending) (void)hipEventSynchronize(slot.copied);
        if (slot.copied != nullptr) (void)hipEventDestroy(slot.copied);
        if (slot.host != nullptr) (void)hipHostFree(slot.host);
        slot = DenseSlot{};
    }
    g_dense_next = 0;
}

}  // namespace strix::minimax::moe::opt
