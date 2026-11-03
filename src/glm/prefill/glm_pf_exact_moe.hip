// Exact grouped routed + shared experts. Arithmetic helpers retain their original provenance.
#include "glm/prefill/glm_pf_exact_moe.hpp"
#include "glm/moe/glm_trellis_device.hpp"
#include "glm/common/glm_math.hpp"
#include "glm/prefill/glm_pf_exact_x.hpp"
#include <array>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::glm::pf::exact_moe {
namespace {
namespace d = moe::dev;
constexpr unsigned H = moe::kHidden, I = moe::kInter, E = moe::kExperts, P = 9;
constexpr std::size_t align256(std::size_t x) { return (x + 255) & ~std::size_t{255}; }
constexpr unsigned kPlanRows = 256;  // rows per device-plan block (one thread per row)
struct Layout {
    std::size_t table = 0, pairs, tiles, groups, gu, at, output, input, plan, bytes;
    explicit Layout(unsigned rows) {
        if (rows == 0 || rows > 65536) throw std::invalid_argument("exact moe: max_rows must be 1..65536");
        const std::size_t n = std::size_t{rows} * P;
        pairs = align256((E + 1) * sizeof(moe::ExpertSlices));  // [E] = the shared expert
        tiles = pairs + align256(n * sizeof(unsigned));
        groups = tiles + align256((n / 16 + E + 1) * sizeof(Tile));
        gu = groups + align256((E + 1) * sizeof(Tile));
        at = gu + align256(n * 2 * I * sizeof(float));
        output = at + align256(n * I * sizeof(std::uint16_t));
        input = output + align256(n * H * sizeof(float));
        // V2/V3 cache both matrix-specific gate/up transforms; V1 leaves this region untouched.
        plan = input + align256(n * 2 * H * sizeof(std::uint16_t));
        // Device plan: per-block expert counts/prefixes, per-assignment ranks, expert offsets and the tile count.
        const std::size_t blocks = (rows + kPlanRows - 1U) / kPlanRows;
        bytes = plan + align256(blocks * E * sizeof(unsigned)) + align256(std::size_t{rows} * 8U * sizeof(unsigned)) +
                align256((E + 2) * sizeof(unsigned));
    }
};
void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("exact moe: ") + what + ": " + hipGetErrorString(e));
}
struct Args {
    unsigned rows;
    const float* x;
    const float* weights;
    float* out;
    const moe::ExpertSlices* experts;
    moe::ExpertSlices shared;
    const unsigned* pairs;
    const Tile* tiles;
    const Tile* groups;
    float* gu;
    _Float16* at;
    float* output;
    _Float16* input;
    unsigned rcap;  // v3 row buckets on (1) / off (0)
    const unsigned* ntiles;  // device tile count (device plan) or null (host plan: grid == tiles)
};
// One global load from the device table (entry E = the shared expert): selecting a member of a by-value / kernarg
// struct with a runtime index forces the struct to scratch (328 B/lane in the first draft).
static_assert(offsetof(moe::ExpertSlices, up) == sizeof(moe::Exl3) && offsetof(moe::ExpertSlices, down) == 2 * sizeof(moe::Exl3),
              "gate, up, down contiguous");
__device__ __forceinline__ moe::Exl3 matrix(const Args& a, unsigned expert, unsigned m) {
    return (&a.experts[expert].gate)[m];
}

// Each wave transforms one H128 block exactly once per (pair, matrix). The
// materialized FP16 values have the same boundary as the V1 LDS staging.
__global__ void input_kernel(Args a) {
    if (a.ntiles && blockIdx.y >= *a.ntiles) return;
    const Tile tile = a.tiles[blockIdx.y];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned m = blockIdx.z;
    const moe::Exl3 w = matrix(a, tile.expert, m);
    // One CTA covers 1024 input channels for every row in this expert tile.
    // Eight H128 waves avoid a separate tiny CTA for each transform block.
    for (unsigned r = 0; r < tile.count; ++r) {
        const unsigned pair = a.pairs[tile.first + r];
        const unsigned begin = (blockIdx.x * 8U + wave) * 128U;
        d::transform_block_at(a.x + std::size_t{pair / P} * H, w.suh,
                              a.input + (std::size_t{pair} * 2U + m) * H + begin, begin, 0, lane);
    }
}

// One wave per 16-output tile, eight waves per H128 block. Each decoded operand is reused
// across TM independent row chains by compute_slot; the helper fixes DOT2 pair order.
// Full reference splits are consumed in order, never selected from the tile row count.
template<int Bits, int TM, bool Down, bool Cached>
__device__ __forceinline__ void core(Args a, Tile tile, moe::Exl3 w, unsigned m,
                                    _Float16* xs, float* fin) {
    constexpr unsigned KPS = Down ? 16U : 8U, Splits = Down ? 4U : 32U;
    constexpr unsigned Chunk = KPS * 16U;
    constexpr unsigned N = Down ? H : I;
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    const unsigned col = blockIdx.x * 8U + wave;
    const d::LaneGeom<Bits> geo(lane);
    const auto* packed = reinterpret_cast<const unsigned char*>(d::uniform(w.trellis));
    // V2 keeps ascending split totals in LDS, freeing TM*2 accumulator VGPRs.
    float total[Cached ? 1 : TM][2] = {};
    if constexpr (Cached) {
        for (unsigned q = threadIdx.x; q < tile.count * 128U; q += 256U) fin[q] = 0.0F;
        __syncthreads();
    }
    for (unsigned sp = 0; sp < Splits; ++sp) {
        if constexpr (Down || Cached) {
            for (unsigned q = threadIdx.x; q < tile.count * (Chunk / 2); q += 256) {
                const unsigned r = q / (Chunk / 2), c = q % (Chunk / 2);
                const unsigned pair = a.pairs[tile.first + r];
                const _Float16* input;
                if constexpr (Down) input = a.at + std::size_t{pair} * I;
                else input = a.input + (std::size_t{pair} * 2U + m) * H;
                reinterpret_cast<std::uint32_t*>(xs)[q] =
                    reinterpret_cast<const std::uint32_t*>(input)[sp * (Chunk / 2) + c];
            }
        } else {
            for (unsigned r = wave; r < tile.count; r += 8) {
                const unsigned pair = a.pairs[tile.first + r];
                d::transform_block_at(a.x + std::size_t{pair / P} * H, w.suh,
                                      xs + r * Chunk, sp * Chunk, 0, lane);
            }
        }
        __syncthreads();
        float sums[TM][2] = {};
        std::uint32_t pa = 0, pb = 0, pc = 0;
        if constexpr (Cached) {
            const auto* raw = packed + (std::size_t{sp * KPS} * (N / 16) + col) * Bits * 32;
            pa = d::load_u32(raw, geo.oa);
            pb = d::load_u32(raw, geo.ob);
            pc = d::load_u32(raw, geo.oc);
        }
        for (unsigned kk = 0; kk < KPS; ++kk) {
            const unsigned k = sp * KPS + kk;
            const unsigned char* raw = packed + (std::size_t{k} * (N / 16) + col) * Bits * 32;
            d::Slot<Bits, TM> s;
            if constexpr (Cached) { s.a = pa; s.b = pb; s.c = pc; }
            else {
                s.a = d::load_u32(raw, geo.oa);
                s.b = d::load_u32(raw, geo.ob);
                s.c = d::load_u32(raw, geo.oc);
            }
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                // Padded rows use defined zero operands; no scratch from a previous call is read.
                s.act[r][0] = d::Half2{};
                s.act[r][1] = d::Half2{};
                if (static_cast<unsigned>(r) < tile.count) {
                    const auto* p = reinterpret_cast<const std::uint32_t*>(xs + static_cast<unsigned>(r) * Chunk + kk * 16) + (lane & 3U);
                    s.act[r][0] = __builtin_bit_cast(d::Half2, p[0]);
                    s.act[r][1] = __builtin_bit_cast(d::Half2, p[4]);
                }
            }
            if constexpr (Cached) {
                // Issue the next trellis words before this tile's decode/DOT2 chain.
                // Do not prefetch past a split (especially the final allocation tile).
                if (kk + 1U < KPS) {
                    const auto* next = raw + std::size_t{N / 16} * Bits * 32;
                    pa = d::load_u32(next, geo.oa);
                    pb = d::load_u32(next, geo.ob);
                    pc = d::load_u32(next, geo.oc);
                }
            }
            d::compute_slot<Bits, TM>(s, geo, tile.count, sums);
        }
#pragma unroll
        for (int r = 0; r < TM; ++r) {
            const float v0 = d::reduce4(sums[r][0]), v1 = d::reduce4(sums[r][1]);
            if constexpr (Cached) {
                if ((lane & 3U) == 0 && static_cast<unsigned>(r) < tile.count) {
                    const unsigned o = static_cast<unsigned>(r) * 128U + wave * 16U + lane / 4U;
                    fin[o] = fin[o] + v0;
                    fin[o + 8U] = fin[o + 8U] + v1;
                }
            } else {
                total[r][0] = total[r][0] + v0;
                total[r][1] = total[r][1] + v1;
            }
        }
        __syncthreads(); // all waves have finished reading xs before its next split
    }
    if constexpr (!Cached) {
        if ((lane & 3U) == 0)
#pragma unroll
            for (int r = 0; r < TM; ++r)
                if (static_cast<unsigned>(r) < tile.count) {
                    fin[static_cast<unsigned>(r) * 128 + wave * 16 + lane / 4] = total[r][0];
                    fin[static_cast<unsigned>(r) * 128 + wave * 16 + lane / 4 + 8] = total[r][1];
                }
    }
    __syncthreads();
    for (unsigned r = wave; r < tile.count; r += 8) {
        float v[4];
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) v[j] = fin[r * 128 + lane + 32 * j];
        d::hadamard128(v, lane);
        const unsigned pair = a.pairs[tile.first + r];
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned o = blockIdx.x * 128 + lane + 32 * j;
            const float value = v[j] * d::half_bits(w.svh[o]);
            if constexpr (Down) a.output[std::size_t{pair} * H + o] = value;
            else a.gu[(std::size_t{pair} * 2 + m) * I + o] = value;
        }
    }
}
template<int TM, bool Down, bool Cached = false>
__global__ __launch_bounds__(256) void linear_kernel(Args a) {
    constexpr unsigned Chunk = Down ? 256U : 128U;
    __shared__ _Float16 xs[TM * Chunk];
    __shared__ float fin[TM * 128];
    const Tile tile = a.tiles[blockIdx.y];
    const unsigned m = Down ? 2U : blockIdx.z;
    const moe::Exl3 w = matrix(a, tile.expert, m);
    if (w.bits == 6) core<6, TM, Down, Cached>(a, tile, w, m, xs, fin);
    else core<4, TM, Down, Cached>(a, tile, w, m, xs, fin);
}
// V3: dense operand schedule, with one CTA per (expert, output block, matrix).
// The descriptor address is wave-uniform; load it once outside all row/split loops.
// Bits is a launch-time specialization, never a per-row or per-k branch. Rcap (1..16) bounds the row chains a
// tile computes; live rows run the same statements at any Rcap, rows past tile.count are never published.
template<int Bits, bool Down, unsigned Rcap = 16>
__device__ __forceinline__ void core_v3(Args a, Tile tile, moe::Exl3 w, unsigned m,
                                       std::uint32_t* act, float* total) {
    constexpr unsigned KPS = Down ? 16U : 8U, Splits = Down ? 4U : 32U;
    constexpr unsigned Pairs = KPS * 8U, N = Down ? H : I;
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    const unsigned col = blockIdx.x * 8U + wave;
    const d::LaneGeom<Bits> geo(lane);
    const auto* packed = reinterpret_cast<const unsigned char*>(w.trellis);
    for (unsigned q = threadIdx.x; q < 16U * 128U; q += 256U) total[q] = 0.0F;
    __syncthreads();
    for (unsigned sp = 0; sp < Splits; ++sp) {
        for (unsigned q = threadIdx.x; q < Rcap * Pairs; q += 256U) {
            const unsigned r = q / Pairs, c = q % Pairs;
            std::uint32_t value = 0;
            if (r < tile.count) {
                const unsigned pair = a.pairs[tile.first + r];
                const _Float16* input;
                if constexpr (Down) input = a.at + std::size_t{pair} * I;
                else input = a.input + (std::size_t{pair} * 2U + m) * H;
                __builtin_memcpy(&value, input + (sp * Pairs + c) * 2U, sizeof(value));
            }
            act[q] = value;
        }
        __syncthreads();
        float sums[Rcap][2] = {};
        for (unsigned kk = 0; kk < KPS; ++kk) {
            const auto* raw = packed + (std::size_t{sp * KPS + kk} * (N / 16U) + col) * Bits * 32U;
            const d::TileStates<Bits> st(raw, geo);
            const d::Half2 w0 = d::mul1_pair(st.template state<0>(), st.template state<1>());
            const d::Half2 w1 = d::mul1_pair(st.template state<2>(), st.template state<3>());
            const d::Half2 w2 = d::mul1_pair(st.template state<4>(), st.template state<5>());
            const d::Half2 w3 = d::mul1_pair(st.template state<6>(), st.template state<7>());
#pragma unroll
            for (unsigned r = 0; r < Rcap; ++r) {
                const unsigned p = r * Pairs + kk * 8U + (lane & 3U);
                const d::Half2 a0 = __builtin_bit_cast(d::Half2, act[p]);
                const d::Half2 a1 = __builtin_bit_cast(d::Half2, act[p + 4U]);
                sums[r][0] = __builtin_amdgcn_fdot2(w0, a0, sums[r][0], false);
                sums[r][0] = __builtin_amdgcn_fdot2(w1, a1, sums[r][0], false);
                sums[r][1] = __builtin_amdgcn_fdot2(w2, a0, sums[r][1], false);
                sums[r][1] = __builtin_amdgcn_fdot2(w3, a1, sums[r][1], false);
            }
        }
#pragma unroll
        for (unsigned r = 0; r < Rcap; ++r) {
            const float v0 = d::reduce4(sums[r][0]), v1 = d::reduce4(sums[r][1]);
            if ((lane & 3U) == 0) {
                const unsigned o = r * 128U + wave * 16U + lane / 4U;
                total[o] = total[o] + v0;
                total[o + 8U] = total[o + 8U] + v1;
            }
        }
        __syncthreads(); // finish LDS reads before staging the next split
    }
    for (unsigned r = wave; r < tile.count; r += 8U) {
        float v[4];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) v[j] = total[r * 128U + lane + 32U * j];
        d::hadamard128(v, lane);
        const unsigned pair = a.pairs[tile.first + r];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const unsigned o = blockIdx.x * 128U + lane + 32U * j;
            const float value = v[j] * d::half_bits(w.svh[o]);
            if constexpr (Down) a.output[std::size_t{pair} * H + o] = value;
            else a.gu[(std::size_t{pair} * 2U + m) * I + o] = value;
        }
    }
    __syncthreads(); // other waves must finish total before the next row tile clears it
}
constexpr unsigned kSharedRows = 64;
template<int Bits, bool Down, bool Shared>
__global__ __launch_bounds__(256) void linear_kernel_v3(Args a) {
    __shared__ std::uint32_t act[16U * (Down ? 128U : 64U)];
    __shared__ float total[16U * 128U];
    const unsigned expert = Shared ? E : blockIdx.y;
    const Tile group = a.groups[expert];
    if (group.count == 0) return; // uniform, before any barrier
    const unsigned m = Down ? 2U : blockIdx.z;
    const auto* descriptor = d::uniform((&a.experts[expert].gate) + m);
    moe::Exl3 w = *descriptor;
    w.trellis = d::uniform(w.trellis);
    w.suh = d::uniform(w.suh);
    w.svh = d::uniform(w.svh);
    // The shared expert has every row: its row tiles are spread over blockIdx.y slices of kSharedRows rows
    // (a routed expert has ~57 rows at R 2048 and keeps one CTA per output block).
    const unsigned r0 = Shared ? blockIdx.y * kSharedRows : 0U;
    const unsigned r1 = Shared ? (group.count < r0 + kSharedRows ? group.count : r0 + kSharedRows) : group.count;
    for (unsigned r = r0; r < r1; r += 16U) {
        const Tile tile{expert, group.first + r, r1 - r < 16U ? r1 - r : 16U};
        // Smallest row bucket covering the tile (uniform per CTA); STRIX_GLM_PF_EXACTMOE_V3_RCAP=0 keeps 16 rows.
        if (!a.rcap || tile.count > 8U) core_v3<Bits, Down>(a, tile, w, m, act, total);
        else if (tile.count > 4U) core_v3<Bits, Down, 8>(a, tile, w, m, act, total);
        else if (tile.count > 2U) core_v3<Bits, Down, 4>(a, tile, w, m, act, total);
        else if (tile.count > 1U) core_v3<Bits, Down, 2>(a, tile, w, m, act, total);
        else core_v3<Bits, Down, 1>(a, tile, w, m, act, total);
    }
}
template<bool Down>
void launch_linear_v3(Args a, unsigned shared_bits, hipStream_t stream) {
    constexpr unsigned N = Down ? H : I;
    linear_kernel_v3<4, Down, false><<<dim3(N / 128U, E, Down ? 1U : 2U), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "v3 routed linear");
    if (shared_bits == 6U)
        linear_kernel_v3<6, Down, true><<<dim3(N / 128U, (a.rows + kSharedRows - 1U) / kSharedRows, Down ? 1U : 2U), 256, 0, stream>>>(a);
    else
        linear_kernel_v3<4, Down, true><<<dim3(N / 128U, (a.rows + kSharedRows - 1U) / kSharedRows, Down ? 1U : 2U), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "v3 shared linear");
}
template<bool Down>
struct XRows {
    Args a;
    Tile tile;
    unsigned m;
    __device__ const _Float16* input(unsigned r) const {
        const unsigned pair = a.pairs[tile.first + r];
        if constexpr (Down) return a.at + std::size_t{pair} * I;
        else return a.input + (std::size_t{pair} * 2U + m) * H;
    }
    __device__ float* output(unsigned r) const {
        const unsigned pair = a.pairs[tile.first + r];
        if constexpr (Down) return a.output + std::size_t{pair} * H;
        else return a.gu + (std::size_t{pair} * 2U + m) * I;
    }
};
template<int Bits, bool Down, bool Shared, bool Prefetch = false>
__global__ __launch_bounds__(256) void linear_kernel_x(Args a) {
    __shared__ exact_x::Storage lds;
    const unsigned expert = Shared ? E : blockIdx.y;
    const Tile group = a.groups[expert];
    if (group.count == 0) return;
    const unsigned m = Down ? 2U : blockIdx.z;
    const auto w = *d::uniform((&a.experts[expert].gate) + m);
    const unsigned r0 = Shared ? blockIdx.y * kSharedRows : 0U;
    const unsigned r1 = Shared ? (group.count < r0 + kSharedRows ? group.count : r0 + kSharedRows) : group.count;
    for (unsigned r = r0; r < r1; r += 32U) {
        const Tile tile{expert, group.first + r, r1 - r < 32U ? r1 - r : 32U};
        exact_x::core<Bits, Prefetch>(w, XRows<Down>{a, tile, m}, tile.count, Down ? 4U : 32U, false, lds);
    }
}
template<bool Down, bool Prefetch = false>
void launch_linear_x(Args a, unsigned shared_bits, hipStream_t stream) {
    constexpr unsigned N = Down ? H : I;
    linear_kernel_x<4, Down, false, Prefetch><<<dim3(N / 128U, E, Down ? 1U : 2U), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "X routed linear");
    if (shared_bits == 6U)
        linear_kernel_x<6, Down, true, Prefetch><<<dim3(N / 128U, (a.rows + kSharedRows - 1U) / kSharedRows, Down ? 1U : 2U), 256, 0, stream>>>(a);
    else
        linear_kernel_x<4, Down, true, Prefetch><<<dim3(N / 128U, (a.rows + kSharedRows - 1U) / kSharedRows, Down ? 1U : 2U), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "X shared linear");
}
template<int Bits, bool Down, bool Shared, unsigned R>
__global__ __launch_bounds__(256) void linear_kernel_s(Args a) {
    __shared__ exact_x::StorageS<R> lds;
    const unsigned expert = Shared ? E : blockIdx.y;
    const Tile group = a.groups[expert];
    if (group.count == 0) return;
    const unsigned m = Down ? 2U : blockIdx.z;
    const auto w = *d::uniform((&a.experts[expert].gate) + m);
    const unsigned r0 = Shared ? blockIdx.y * kSharedRows : 0U;
    const unsigned r1 = Shared ? (group.count < r0 + kSharedRows ? group.count : r0 + kSharedRows) : group.count;
    for (unsigned r = r0; r < r1; r += 8U * R) {
        const Tile tile{expert, group.first + r, r1 - r < 8U * R ? r1 - r : 8U * R};
        exact_x::core_s<Bits, R>(w, XRows<Down>{a, tile, m}, tile.count, Down ? 4U : 32U, false, lds);
    }
}
// V10: per expert group, 32-row tiles (four rows per wave) for groups of up to 32 rows, 64-row tiles above.
template<int Bits, bool Down, bool Shared>
__global__ __launch_bounds__(256) void linear_kernel_s2(Args a) {
    __shared__ exact_x::StorageS<8> lds;
    const unsigned expert = Shared ? E : blockIdx.y;
    const Tile group = a.groups[expert];
    if (group.count == 0) return;
    const unsigned m = Down ? 2U : blockIdx.z;
    const auto w = *d::uniform((&a.experts[expert].gate) + m);
    const unsigned r0 = Shared ? blockIdx.y * kSharedRows : 0U;
    const unsigned r1 = Shared ? (group.count < r0 + kSharedRows ? group.count : r0 + kSharedRows) : group.count;
    if (r1 - r0 <= 32U) {
        auto& small = *reinterpret_cast<exact_x::StorageS<4>*>(&lds);
        exact_x::core_s<Bits, 4>(w, XRows<Down>{a, Tile{expert, group.first + r0, r1 - r0}, m}, r1 - r0, Down ? 4U : 32U, false, small);
        return;
    }
    for (unsigned r = r0; r < r1; r += 64U) {
        const Tile tile{expert, group.first + r, r1 - r < 64U ? r1 - r : 64U};
        exact_x::core_s<Bits, 8>(w, XRows<Down>{a, tile, m}, tile.count, Down ? 4U : 32U, false, lds);
    }
}
template<bool Down>
void launch_linear_s2(Args a, unsigned shared_bits, hipStream_t stream) {
    constexpr unsigned N = Down ? H : I;
    linear_kernel_s2<4, Down, false><<<dim3(N / 128U, E, Down ? 1U : 2U), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "X3 routed linear");
    if (shared_bits == 6U)
        linear_kernel_s2<6, Down, true><<<dim3(N / 128U, (a.rows + kSharedRows - 1U) / kSharedRows, Down ? 1U : 2U), 256, 0, stream>>>(a);
    else
        linear_kernel_s2<4, Down, true><<<dim3(N / 128U, (a.rows + kSharedRows - 1U) / kSharedRows, Down ? 1U : 2U), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "X3 shared linear");
}
// V11: each group (or 64-row shared slice) runs its whole 64-row tiles with eight rows per wave in one kernel and
// its remainder in a second kernel with the smallest bucket per tile (1, 2 or 4 rows per wave, at most 32 rows per
// tile), so computed row chains track the live rows. The two kernels write disjoint rows; per-row arithmetic is X3's.
// Rows the full kernel covers for a span: whole 64-row tiles, plus (Fold) a remainder above 32 rows as one padded
// 64-row tile (rows past the span are computed and never published).
template<bool Fold> __device__ __forceinline__ unsigned full_rows(unsigned span) {
    const unsigned full = span / 64U * 64U;
    return Fold && span - full > 32U ? span : full;
}
template<int Bits, bool Down, bool Shared, bool Fold = false>
__global__ __launch_bounds__(256) void linear_kernel_s3_full(Args a) {
    __shared__ exact_x::StorageS<8> lds;
    const unsigned expert = Shared ? E : blockIdx.y;
    const Tile group = a.groups[expert];
    const unsigned m = Down ? 2U : blockIdx.z;
    const unsigned r0 = Shared ? blockIdx.y * kSharedRows : 0U;
    if (r0 >= group.count) return;
    const unsigned span = Shared ? (group.count - r0 < kSharedRows ? group.count - r0 : kSharedRows) : group.count;
    const unsigned full = full_rows<Fold>(span);
    if (full == 0) return;  // uniform, before any barrier
    const auto w = *d::uniform((&a.experts[expert].gate) + m);
    for (unsigned r = r0; r < r0 + full; r += 64U) {
        const unsigned n = r0 + full - r < 64U ? r0 + full - r : 64U;
        exact_x::core_s<Bits, 8>(w, XRows<Down>{a, Tile{expert, group.first + r, n}, m}, n, Down ? 4U : 32U, false, lds);
    }
}
template<int Bits, bool Down, bool Shared, bool Fold = false>
__global__ __launch_bounds__(256) void linear_kernel_s3_tail(Args a) {
    __shared__ exact_x::StorageS<4> lds;
    const unsigned expert = Shared ? E : blockIdx.y;
    const Tile group = a.groups[expert];
    const unsigned m = Down ? 2U : blockIdx.z;
    const unsigned r0 = Shared ? blockIdx.y * kSharedRows : 0U;
    if (r0 >= group.count) return;
    const unsigned span = Shared ? (group.count - r0 < kSharedRows ? group.count - r0 : kSharedRows) : group.count;
    const unsigned full = full_rows<Fold>(span);
    if (full == span) return;  // uniform, before any barrier
    const auto w = *d::uniform((&a.experts[expert].gate) + m);
    auto& one = *reinterpret_cast<exact_x::StorageS<1>*>(&lds);
    auto& two = *reinterpret_cast<exact_x::StorageS<2>*>(&lds);
    for (unsigned r = r0 + full; r < r0 + span;) {
        const unsigned n = r0 + span - r < 32U ? r0 + span - r : 32U;
        const Tile tile{expert, group.first + r, n};
        if (n <= 8U) exact_x::core_s<Bits, 1>(w, XRows<Down>{a, tile, m}, n, Down ? 4U : 32U, false, one);
        else if (n <= 16U) exact_x::core_s<Bits, 2>(w, XRows<Down>{a, tile, m}, n, Down ? 4U : 32U, false, two);
        else exact_x::core_s<Bits, 4>(w, XRows<Down>{a, tile, m}, n, Down ? 4U : 32U, false, lds);
        r += n;
    }
}
template<bool Down, bool Fold = false>
void launch_linear_s3(Args a, unsigned shared_bits, hipStream_t stream) {
    constexpr unsigned N = Down ? H : I;
    const dim3 routed(N / 128U, E, Down ? 1U : 2U), shared(N / 128U, (a.rows + kSharedRows - 1U) / kSharedRows, Down ? 1U : 2U);
    linear_kernel_s3_full<4, Down, false, Fold><<<routed, 256, 0, stream>>>(a);
    linear_kernel_s3_tail<4, Down, false, Fold><<<routed, 256, 0, stream>>>(a);
    ok(hipGetLastError(), "X3 routed linear");
    if (shared_bits == 6U) {
        linear_kernel_s3_full<6, Down, true, Fold><<<shared, 256, 0, stream>>>(a);
        linear_kernel_s3_tail<6, Down, true, Fold><<<shared, 256, 0, stream>>>(a);
    } else {
        linear_kernel_s3_full<4, Down, true, Fold><<<shared, 256, 0, stream>>>(a);
        linear_kernel_s3_tail<4, Down, true, Fold><<<shared, 256, 0, stream>>>(a);
    }
    ok(hipGetLastError(), "X3 shared linear");
}
template<bool Down, unsigned R>
void launch_linear_s(Args a, unsigned shared_bits, hipStream_t stream) {
    constexpr unsigned N = Down ? H : I;
    linear_kernel_s<4, Down, false, R><<<dim3(N / 128U, E, Down ? 1U : 2U), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "X3 routed linear");
    if (shared_bits == 6U)
        linear_kernel_s<6, Down, true, R><<<dim3(N / 128U, (a.rows + kSharedRows - 1U) / kSharedRows, Down ? 1U : 2U), 256, 0, stream>>>(a);
    else
        linear_kernel_s<4, Down, true, R><<<dim3(N / 128U, (a.rows + kSharedRows - 1U) / kSharedRows, Down ? 1U : 2U), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "X3 shared linear");
}
__global__ void activation_kernel(Args a) {
    if (a.ntiles && blockIdx.y >= *a.ntiles) return;
    const Tile tile = a.tiles[blockIdx.y];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const moe::Exl3 down = matrix(a, tile.expert, 2);
    for (unsigned r = wave; r < tile.count; r += 8) {
        const unsigned pair = a.pairs[tile.first + r];
        float v[4];
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned i = blockIdx.x * 128 + lane + 32 * j;
            const float g = a.gu[(std::size_t{pair} * 2) * I + i];
            const float u = a.gu[(std::size_t{pair} * 2 + 1) * I + i];
            const float gg = g > kClamp ? kClamp : g;
            const float uu = u > kClamp ? kClamp : (u < -kClamp ? -kClamp : u);
            const float act = silu(gg) * uu;
            const float scaled = d::materialize(act * d::half_bits(down.suh[i]));
            v[j] = static_cast<float>(static_cast<_Float16>(scaled));
        }
        d::hadamard128(v, lane);
#pragma unroll
        for (unsigned j = 0; j < 4; ++j)
            a.at[std::size_t{pair} * I + blockIdx.x * 128 + lane + 32 * j] = static_cast<_Float16>(v[j]);
    }
}
__global__ void combine_kernel(Args a) {
    const unsigned row = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    float y = 0.0F;
    for (unsigned k = 0; k < 8; ++k)
        y = __builtin_fmaf(a.weights[row * 8 + k], a.output[(std::size_t{row} * P + k) * H + i], y);
    a.out[std::size_t{row} * H + i] = y + a.output[(std::size_t{row} * P + 8) * H + i];
}
template<int TM, bool Cached> void launch(Args a, unsigned tiles, hipStream_t stream) {
    if constexpr (Cached) {
        input_kernel<<<dim3(H / 1024, tiles, 2), 256, 0, stream>>>(a);
        ok(hipGetLastError(), "gate/up input transform");
    }
    linear_kernel<TM, false, Cached><<<dim3(I / 128, tiles, 2), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "gate/up");
    activation_kernel<<<dim3(I / 128, tiles), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "R15/down transform");
    linear_kernel<TM, true, Cached><<<dim3(H / 128, tiles), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "down");
    combine_kernel<<<dim3(H / 256, a.rows), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "R16");
}
// Device plan, byte-equal to the host plan in enqueue(): expert e's pairs in ascending row order (a row holds an
// expert at most once), shared pairs r*9+8 last, groups[e] = {e, offset, count}, tiles of 16 rows per expert in
// expert order. A row whose ids are not strictly ascending and < 288 sets *flag and contributes no routed pair.
struct PlanArgs {
    unsigned rows;
    const std::uint32_t* ids;
    unsigned* blk;    // [blocks][E] counts, then exclusive per-expert block prefixes
    unsigned* rank;   // [rows][8] rank among the block's earlier rows holding the same expert
    unsigned* off;    // [E + 1] expert offsets, [E + 1] tile count
    unsigned* pairs;
    Tile* tiles;
    Tile* groups;
    unsigned* flag;
};
__global__ __launch_bounds__(kPlanRows) void plan_rank_kernel(PlanArgs p) {
    __shared__ unsigned cnt[kPlanRows / 32U][E];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, r = blockIdx.x * kPlanRows + tid;
    for (unsigned i = tid; i < kPlanRows / 32U * E; i += kPlanRows) (&cnt[0][0])[i] = 0;
    unsigned e[8];
    bool bad = false;
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) {
        e[k] = r < p.rows ? p.ids[std::size_t{r} * 8U + k] : E;
        bad = bad || (r < p.rows && (e[k] >= E || (k && e[k] <= e[k - 1])));
    }
    if (bad) {
        atomicOr(p.flag, 1U);
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) e[k] = E;
    }
    __syncthreads();
    // Wave-local ranks: one pass per distinct expert held by any lane; lanes below hold earlier rows.
    unsigned rk[8] = {}, done = 0;
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) if (e[k] >= E) done |= 1U << k;
    for (;;) {
        const unsigned pending = static_cast<unsigned>(__ballot(done != 0xffU));
        if (!pending) break;
        const unsigned leader = static_cast<unsigned>(__builtin_ctz(pending));
        unsigned mine = E;
#pragma unroll
        for (unsigned k = 8U; k-- > 0;) if (!(done >> k & 1U)) mine = e[k];
        const unsigned x = static_cast<unsigned>(__shfl(static_cast<int>(mine), static_cast<int>(leader)));
        unsigned slot = 8U;
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) if (e[k] == x) slot = k;
        const unsigned holders = static_cast<unsigned>(__ballot(slot < 8U));
        if (slot < 8U) {
            rk[slot] = cnt[wave][x] + static_cast<unsigned>(__builtin_popcount(holders & ((1U << lane) - 1U)));
            done |= 1U << slot;
        }
        if (lane == leader) cnt[wave][x] += static_cast<unsigned>(__builtin_popcount(holders));
    }
    __syncthreads();
    // Waves cover consecutive rows: prefix the wave counts per expert, publish the block's counts.
    for (unsigned x = tid; x < E; x += kPlanRows) {
        unsigned run = 0;
        for (unsigned w = 0; w < kPlanRows / 32U; ++w) { const unsigned c = cnt[w][x]; cnt[w][x] = run; run += c; }
        p.blk[std::size_t{blockIdx.x} * E + x] = run;
    }
    __syncthreads();
    if (r < p.rows)
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) p.rank[std::size_t{r} * 8U + k] = e[k] < E ? rk[k] + cnt[wave][e[k]] : 0U;
}
__global__ __launch_bounds__(1024) void plan_scan_kernel(PlanArgs p, unsigned blocks) {
    __shared__ unsigned count[E + 1], tile_off[E + 1];
    const unsigned tid = threadIdx.x;
    for (unsigned x = tid; x < E; x += blockDim.x) {
        unsigned run = 0;
        for (unsigned b = 0; b < blocks; ++b) {
            unsigned* c = p.blk + std::size_t{b} * E + x;
            const unsigned v = *c;
            *c = run;
            run += v;
        }
        count[x] = run;
    }
    if (tid == 0) count[E] = p.rows;
    __syncthreads();
    if (tid == 0) {
        unsigned n = 0, t = 0;
        for (unsigned x = 0; x <= E; ++x) {
            p.off[x] = n;
            tile_off[x] = t;
            p.groups[x] = {x, n, count[x]};
            n += count[x];
            t += (count[x] + 15U) / 16U;
        }
        p.off[E + 1] = t;
    }
    __syncthreads();
    for (unsigned x = tid; x <= E; x += blockDim.x)
        for (unsigned r = 0, t = tile_off[x]; r < count[x]; r += 16U, ++t)
            p.tiles[t] = {x, p.off[x] + r, count[x] - r < 16U ? count[x] - r : 16U};
}
__global__ __launch_bounds__(kPlanRows) void plan_scatter_kernel(PlanArgs p) {
    const unsigned r = blockIdx.x * kPlanRows + threadIdx.x;
    if (r >= p.rows) return;
    unsigned e[8];
    bool bad = false;
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) {
        e[k] = p.ids[std::size_t{r} * 8U + k];
        bad = bad || e[k] >= E || (k && e[k] <= e[k - 1]);
    }
    if (!bad)
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k)
            p.pairs[p.off[e[k]] + p.blk[std::size_t{blockIdx.x} * E + e[k]] + p.rank[std::size_t{r} * 8U + k]] = r * P + k;
    p.pairs[p.off[E] + r] = r * P + 8U;
}
// The grouped variants (v3 and the X cores): input transform, gate/up, activation, down, combine. grid_tiles is the
// tile count (host plan) or its upper bound (device plan; surplus CTAs return on Args::ntiles).
void launch_grouped(const Args& a, unsigned grid_tiles, unsigned selected, const moe::ExpertSlices& shared, hipStream_t stream) {
    input_kernel<<<dim3(H / 1024, grid_tiles, 2), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "v3 gate/up input transform");
    if (selected == 14) launch_linear_s3<false, true>(a, shared.gate.bits, stream);
    else if (selected == 11) launch_linear_s3<false>(a, shared.gate.bits, stream);
    else if (selected == 10) launch_linear_s2<false>(a, shared.gate.bits, stream);
    else if (selected == 9) launch_linear_s<false, 8>(a, shared.gate.bits, stream);
    else if (selected == 8) launch_linear_s<false, 4>(a, shared.gate.bits, stream);
    else if (selected == 7) launch_linear_x<false, true>(a, shared.gate.bits, stream);
    else if (selected == 6) launch_linear_x<false>(a, shared.gate.bits, stream);
    else launch_linear_v3<false>(a, shared.gate.bits, stream);
    activation_kernel<<<dim3(I / 128, grid_tiles), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "v3 R15/down transform");
    if (selected == 14) launch_linear_s3<true, true>(a, shared.down.bits, stream);
    else if (selected == 11) launch_linear_s3<true>(a, shared.down.bits, stream);
    else if (selected == 10) launch_linear_s2<true>(a, shared.down.bits, stream);
    else if (selected == 9) launch_linear_s<true, 8>(a, shared.down.bits, stream);
    else if (selected == 8) launch_linear_s<true, 4>(a, shared.down.bits, stream);
    else if (selected == 7) launch_linear_x<true, true>(a, shared.down.bits, stream);
    else if (selected == 6) launch_linear_x<true>(a, shared.down.bits, stream);
    else launch_linear_v3<true>(a, shared.down.bits, stream);
    combine_kernel<<<dim3(H / 256, a.rows), 256, 0, stream>>>(a);
    ok(hipGetLastError(), "v3 R16");
}
void check_matrix(const moe::Exl3& w, unsigned input, unsigned output, bool shared) {
    if (!w.trellis || !w.suh || !w.svh || w.input != input || w.output != output ||
        (w.bits != 4 && !(shared && w.bits == 6)))
        throw std::invalid_argument("exact moe: invalid raw matrix");
}
} // namespace
template<bool Prefetch>
hipFuncAttributes query_x_resources(unsigned bits, bool down, bool shared) {
    hipFuncAttributes a{};
#define QUERY(B, D, S) ok(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(&linear_kernel_x<B, D, S, Prefetch>)), "X resources")
#define SELECT(B) if (down) { if (shared) { QUERY(B, true, true); } else { QUERY(B, true, false); } } \
                  else { if (shared) { QUERY(B, false, true); } else { QUERY(B, false, false); } }
    if (bits == 4) { SELECT(4); }
    else if (bits == 6 && shared) { SELECT(6); }
    else throw std::invalid_argument("exact moe: X resource bits/shared");
#undef SELECT
#undef QUERY
    return a;
}
template<unsigned R>
hipFuncAttributes query_s_resources(unsigned bits, bool down, bool shared) {
    hipFuncAttributes a{};
#define QUERY(B, D, S) ok(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(&linear_kernel_s<B, D, S, R>)), "X3 resources")
#define SELECT(B) if (down) { if (shared) { QUERY(B, true, true); } else { QUERY(B, true, false); } }                   else { if (shared) { QUERY(B, false, true); } else { QUERY(B, false, false); } }
    if (bits == 4) { SELECT(4); }
    else if (bits == 6 && shared) { SELECT(6); }
    else throw std::invalid_argument("exact moe: X3 resource bits/shared");
#undef SELECT
#undef QUERY
    return a;
}
hipFuncAttributes query_s2_resources(unsigned bits, bool down, bool shared) {
    hipFuncAttributes a{};
#define QUERY(B, D, S) ok(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(&linear_kernel_s2<B, D, S>)), "X3 resources")
#define SELECT(B) if (down) { if (shared) { QUERY(B, true, true); } else { QUERY(B, true, false); } } \
                  else { if (shared) { QUERY(B, false, true); } else { QUERY(B, false, false); } }
    if (bits == 4) { SELECT(4); }
    else if (bits == 6 && shared) { SELECT(6); }
    else throw std::invalid_argument("exact moe: X3 resource bits/shared");
#undef SELECT
#undef QUERY
    return a;
}
hipFuncAttributes query_s3_resources(unsigned bits, bool down, bool shared, bool tail) {
    hipFuncAttributes a{};
#define QUERY(B, D, S) ok(hipFuncGetAttributes(&a, tail ? reinterpret_cast<const void*>(&linear_kernel_s3_tail<B, D, S>) \
                                                    : reinterpret_cast<const void*>(&linear_kernel_s3_full<B, D, S>)), "X3 resources")
#define SELECT(B) if (down) { if (shared) { QUERY(B, true, true); } else { QUERY(B, true, false); } } \
                  else { if (shared) { QUERY(B, false, true); } else { QUERY(B, false, false); } }
    if (bits == 4) { SELECT(4); }
    else if (bits == 6 && shared) { SELECT(6); }
    else throw std::invalid_argument("exact moe: X3 resource bits/shared");
#undef SELECT
#undef QUERY
    return a;
}
hipFuncAttributes x_resources(unsigned bits, bool down, bool shared, unsigned variant) {
    if (variant == 6) return query_x_resources<false>(bits, down, shared);
    if (variant == 7) return query_x_resources<true>(bits, down, shared);
    if (variant == 8) return query_s_resources<4>(bits, down, shared);
    if (variant == 9) return query_s_resources<8>(bits, down, shared);
    if (variant == 10) return query_s2_resources(bits, down, shared);
    if (variant == 11) return query_s3_resources(bits, down, shared, false);
    if (variant == 12) return query_s3_resources(bits, down, shared, true);  // V11's tail kernel
    throw std::invalid_argument("exact moe: X resources variant must be 6..12 (12 = V11 tail kernel)");
}
Plan::Plan(unsigned max_rows) : capacity(max_rows) {
    (void)Layout(max_rows);
    pairs.reserve(std::size_t{max_rows} * P);
    tiles.reserve(std::size_t{max_rows} * P / 16 + E + 1);
}
std::size_t workspace_bytes(unsigned max_rows) { return Layout(max_rows).bytes; }
const std::uint16_t* pair_down_input(void* ws, unsigned rows, unsigned pair) {
    return reinterpret_cast<const std::uint16_t*>(static_cast<unsigned char*>(ws) + Layout(rows).at) + std::size_t{pair} * I;
}
const float* pair_gate_up(void* ws, unsigned rows, unsigned pair) {
    return reinterpret_cast<const float*>(static_cast<unsigned char*>(ws) + Layout(rows).gu) + std::size_t{pair} * 2 * I;
}
const std::uint16_t* pair_gate_up_input(void* ws, unsigned rows, unsigned pair) {
    return reinterpret_cast<const std::uint16_t*>(static_cast<unsigned char*>(ws) + Layout(rows).input) +
        std::size_t{pair} * 2U * H;
}
const float* pair_output(void* ws, unsigned rows, unsigned pair) {
    return reinterpret_cast<const float*>(static_cast<unsigned char*>(ws) + Layout(rows).output) + std::size_t{pair} * H;
}
unsigned x_auto_rows() {
    // STRIX_GLM_PF_EXACTMOE_X_MIN_ROWS overrides the auto threshold (rows per MoE call).
    const char* p = std::getenv("STRIX_GLM_PF_EXACTMOE_X_MIN_ROWS");
    if (!p) return 128;
    char* end = nullptr;
    const unsigned long n = std::strtoul(p, &end, 10);
    if (!*p || *end || n == 0 || n > 65536) throw std::invalid_argument("STRIX_GLM_PF_EXACTMOE_X_MIN_ROWS must be 1..65536");
    return static_cast<unsigned>(n);
}
unsigned tm_env() {
    const char* s = std::getenv("STRIX_GLM_PF_EXACTMOE_TM");
    if (!s || std::string(s) == "16") return 16;
    if (std::string(s) == "32") return 32;
    throw std::invalid_argument("STRIX_GLM_PF_EXACTMOE_TM must be 16 or 32");
}
bool v3_row_buckets() {
    // Default on: v3 tiles compute only the smallest power-of-two row bucket covering their rows (same bits).
    const char* p = std::getenv("STRIX_GLM_PF_EXACTMOE_V3_RCAP");
    if (!p || std::string(p) == "1") return true;
    if (std::string(p) == "0") return false;
    throw std::invalid_argument("STRIX_GLM_PF_EXACTMOE_V3_RCAP must be 0 or 1");
}
unsigned selected_variant(unsigned rows) {
    const char* variant = std::getenv("STRIX_GLM_PF_EXACTMOE_V");
    // v3 is the measured default (real pair, 1K/8K/32K TTFT logits bitwise == v1 on both ranks, -10..-11 % ms/token);
    // STRIX_GLM_PF_EXACTMOE_V=1 restores v1. auto: v14 (X3 core, row-bucketed tiles, remainders above 32 rows as one
    // padded 64-row tile) for calls >= x_auto_rows() rows, v3 below (both exact;
    // 32-row X tiles mostly pad the small expert groups of short calls).
    if (!variant) return 3;
    const std::string s(variant);
    if (s == "1") return 1;
    if (s == "2") return 2;
    if (s == "3") return 3;
    if (s == "6") return 6;
    if (s == "7") return 7;
    if (s == "8") return 8;
    if (s == "9") return 9;
    if (s == "10") return 10;
    if (s == "11") return 11;
    if (s == "14") return 14;
    if (s == "auto") return rows >= x_auto_rows() ? 14 : 3;
    throw std::invalid_argument("STRIX_GLM_PF_EXACTMOE_V must be 1, 2, 3, 6, 7, 8, 9, 10, 11, 14 or auto");
}
void enqueue(const View& v, Plan& plan, void* ws, std::size_t bytes, hipStream_t stream, unsigned tm) {
    const Layout l(plan.capacity);
    if (!v.rows || v.rows > plan.capacity || !v.experts || !v.x || !v.ids || !v.weights || !v.out || !ws || bytes < l.bytes)
        throw std::invalid_argument("exact moe: invalid view/workspace");
    if (tm == 0) tm = tm_env();
    if (tm != 16 && tm != 32) throw std::invalid_argument("exact moe: TM must be 16 or 32");
    const unsigned selected = selected_variant(v.rows);
    const bool v2 = selected == 2, v3 = selected == 3, v6 = selected == 6, v7 = selected == 7, v8 = selected == 8, v9 = selected == 9, v10 = selected == 10, v11 = selected == 11, v14 = selected == 14;
    for (unsigned e = 0; e <= E; ++e) {
        const auto& ex = e == E ? v.shared : v.experts[e];
        check_matrix(ex.gate, H, I, e == E);
        check_matrix(ex.up, H, I, e == E);
        check_matrix(ex.down, I, H, e == E);
    }
    if (v.shared.gate.bits != v.shared.up.bits)
        throw std::invalid_argument("exact moe: shared gate/up bits must match decode contract");
    std::array<Tile, E + 1> groups{};
    std::array<unsigned, E + 1> count{}, offset{}, cursor{};
    count[E] = v.rows;
    for (unsigned r = 0; r < v.rows; ++r)
        for (unsigned k = 0; k < 8; ++k) {
            const unsigned e = v.ids[r * 8 + k];
            if (e >= E || (k && e <= v.ids[r * 8 + k - 1]))
                throw std::invalid_argument("exact moe: ids must be distinct, ascending and <288");
            ++count[e];
        }
    unsigned n = 0;
    for (unsigned e = 0; e <= E; ++e) {
        offset[e] = cursor[e] = n;
        groups[e] = {e, n, count[e]};
        n += count[e];
    }
    plan.pairs.resize(n);
    plan.tiles.clear();
    for (unsigned r = 0; r < v.rows; ++r) {
        for (unsigned k = 0; k < 8; ++k) plan.pairs[cursor[v.ids[r * 8 + k]]++] = r * P + k;
        plan.pairs[cursor[E]++] = r * P + 8;
    }
    for (unsigned e = 0; e <= E; ++e)
        for (unsigned r = 0; r < count[e]; r += tm)
            plan.tiles.push_back({e, offset[e] + r, count[e] - r < tm ? count[e] - r : tm});
    auto* base = static_cast<unsigned char*>(ws);
    auto* table = reinterpret_cast<moe::ExpertSlices*>(base + l.table);
    auto* pairs = reinterpret_cast<unsigned*>(base + l.pairs);
    auto* tiles = reinterpret_cast<Tile*>(base + l.tiles);
    auto* device_groups = reinterpret_cast<Tile*>(base + l.groups);
    ok(hipMemcpyAsync(table, v.experts, E * sizeof(moe::ExpertSlices), hipMemcpyHostToDevice, stream), "expert table");
    ok(hipMemcpyAsync(table + E, &v.shared, sizeof(moe::ExpertSlices), hipMemcpyHostToDevice, stream), "shared expert");
    ok(hipMemcpyAsync(pairs, plan.pairs.data(), plan.pairs.size() * sizeof(unsigned), hipMemcpyHostToDevice, stream), "pairs");
    ok(hipMemcpyAsync(tiles, plan.tiles.data(), plan.tiles.size() * sizeof(Tile), hipMemcpyHostToDevice, stream), "tiles");
    if (v3 || v6 || v7 || v8 || v9 || v10 || v11 || v14) ok(hipMemcpyAsync(device_groups, groups.data(), sizeof(groups), hipMemcpyHostToDevice, stream), "expert groups");
    // Owns host upload lifetimes even when callers immediately reuse their table or Plan.
    ok(hipStreamSynchronize(stream), "plan upload");
    Args a{v.rows, v.x, v.weights, v.out, table, v.shared, pairs, tiles, device_groups,
           reinterpret_cast<float*>(base + l.gu), reinterpret_cast<_Float16*>(base + l.at), reinterpret_cast<float*>(base + l.output),
           reinterpret_cast<_Float16*>(base + l.input), v3_row_buckets() ? 1U : 0U, nullptr};
    const unsigned nt = static_cast<unsigned>(plan.tiles.size());
    if (v3 || v6 || v7 || v8 || v9 || v10 || v11 || v14) {
        launch_grouped(a, nt, selected, v.shared, stream);
    } else if (v2) {
        if (tm == 16) launch<16, true>(a, nt, stream);
        else launch<32, true>(a, nt, stream);
    } else {
        if (tm == 16) launch<16, false>(a, nt, stream);
        else launch<32, false>(a, nt, stream);
    }
}
const unsigned* plan_pairs(void* ws, unsigned rows) {
    return reinterpret_cast<const unsigned*>(static_cast<unsigned char*>(ws) + Layout(rows).pairs);
}
const Tile* plan_tiles(void* ws, unsigned rows) {
    return reinterpret_cast<const Tile*>(static_cast<unsigned char*>(ws) + Layout(rows).tiles);
}
const Tile* plan_groups(void* ws, unsigned rows) {
    return reinterpret_cast<const Tile*>(static_cast<unsigned char*>(ws) + Layout(rows).groups);
}
moe::ExpertSlices* workspace_table(void* ws) {
    return reinterpret_cast<moe::ExpertSlices*>(static_cast<unsigned char*>(ws) + Layout(1).table);
}
bool device_plan(unsigned rows, unsigned tm) {
    // STRIX_GLM_PF_EXACTMOE_PLAN=host keeps the host plan (route download, host grouping, upload).
    const char* p = std::getenv("STRIX_GLM_PF_EXACTMOE_PLAN");
    if (p && std::string(p) == "host") return false;
    if (p && std::string(p) != "device") throw std::invalid_argument("STRIX_GLM_PF_EXACTMOE_PLAN must be device or host");
    const unsigned s = selected_variant(rows);
    return (s == 3 || s >= 6) && (tm == 0 ? tm_env() : tm) == 16;
}
void enqueue_device(const DeviceView& v, Plan& plan, void* ws, std::size_t bytes, hipStream_t stream, unsigned* flag) {
    const Layout l(plan.capacity);
    if (!v.rows || v.rows > plan.capacity || !v.table || !v.x || !v.ids || !v.weights || !v.out || !flag || !ws || bytes < l.bytes)
        throw std::invalid_argument("exact moe: invalid device view/workspace");
    const unsigned selected = selected_variant(v.rows);
    if (!(selected == 3 || selected >= 6)) throw std::invalid_argument("exact moe: device plan needs a grouped variant");
    check_matrix(v.shared.gate, H, I, true);
    check_matrix(v.shared.up, H, I, true);
    check_matrix(v.shared.down, I, H, true);
    if (v.shared.gate.bits != v.shared.up.bits)
        throw std::invalid_argument("exact moe: shared gate/up bits must match decode contract");
    auto* base = static_cast<unsigned char*>(ws);
    const unsigned blocks = (v.rows + kPlanRows - 1U) / kPlanRows;
    const std::size_t cap_blocks = (plan.capacity + kPlanRows - 1U) / kPlanRows;
    auto* blk = reinterpret_cast<unsigned*>(base + l.plan);
    auto* rank = reinterpret_cast<unsigned*>(base + l.plan + align256(cap_blocks * E * sizeof(unsigned)));
    auto* off = rank + align256(std::size_t{plan.capacity} * 8U * sizeof(unsigned)) / sizeof(unsigned);
    const PlanArgs pa{v.rows, v.ids, blk, rank, off, reinterpret_cast<unsigned*>(base + l.pairs),
                      reinterpret_cast<Tile*>(base + l.tiles), reinterpret_cast<Tile*>(base + l.groups), flag};
    plan_rank_kernel<<<blocks, kPlanRows, 0, stream>>>(pa);
    ok(hipGetLastError(), "device plan ranks");
    plan_scan_kernel<<<1, 1024, 0, stream>>>(pa, blocks);
    ok(hipGetLastError(), "device plan scan");
    plan_scatter_kernel<<<blocks, kPlanRows, 0, stream>>>(pa);
    ok(hipGetLastError(), "device plan scatter");
    Args a{v.rows, v.x, v.weights, v.out, v.table, v.shared, pa.pairs, pa.tiles, pa.groups,
           reinterpret_cast<float*>(base + l.gu), reinterpret_cast<_Float16*>(base + l.at), reinterpret_cast<float*>(base + l.output),
           reinterpret_cast<_Float16*>(base + l.input), v3_row_buckets() ? 1U : 0U, off + E + 1};
    const auto grid_tiles = static_cast<unsigned>(std::size_t{v.rows} * P / 16U + E + 1U);
    launch_grouped(a, grid_tiles, selected, v.shared, stream);
}
} // namespace strix::glm::pf::exact_moe
