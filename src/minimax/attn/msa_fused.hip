#include "minimax/engine/route_identity.hpp"
// minimax-m3 MSA decode, fused and batched (src/minimax/attn/msa_fused.hpp). Every arithmetic step is the one of
// src/hip_minimax_msa.hip (K1-K4) in the same order, so the outputs are bit-identical; what changes is who computes
// it (fewer launches, N sequences per launch) and how the caches are read (coalesced tiles through LDS).
#include "msa_fused.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strix::minimax::msa::fused {
namespace {

constexpr unsigned kIdxThreads = kBlock;    // index launch: thread = position of a 128-position key block
constexpr unsigned kPhases = 16;             // per attend block: 8 wall-clock stamps, then 8 shader-cycle stamps
constexpr unsigned kChunks16 = kBlock * kHeadDim * 2 / 16;  // 16-byte pieces of a 128-key BF16 tile (2048)
constexpr unsigned kPartial = 2 + kHeadDim;
constexpr unsigned kCountMax = 256;         // selection: rank by counting up to this many candidates, else radix

__device__ __forceinline__ float butterfly_sum(float v) {
    for (unsigned d = 16; d >= 1; d >>= 1U) v = v + __shfl_xor(v, static_cast<int>(d), 32);
    return v;
}
__device__ __forceinline__ float max_of(float a, float b) { return a < b ? b : a; }  // std::max(a, b)
__device__ __forceinline__ float lo_bf16(std::uint32_t w) { return u2f(w << 16U); }
__device__ __forceinline__ float hi_bf16(std::uint32_t w) { return u2f(w & 0xFFFF0000U); }

// Lane l holds dims 4l..4l+3 of one head in x[4]. Norm + RoPE in place (host head_norm_rope order); identical to the
// K1 helper.
__device__ __forceinline__ void norm_rope_lane(float* x, const float* w, const RopeRow& rope, unsigned lane) {
    float acc = 0.0F;
    for (unsigned i = 0; i < kDimsPerLane; ++i) acc = __builtin_fmaf(x[i], x[i], acc);
    const float sum = butterfly_sum(acc);
    const float r = 1.0F / __builtin_sqrtf(sum / static_cast<float>(kHeadDim) + kNormEps);
    for (unsigned i = 0; i < kDimsPerLane; ++i) x[i] = (x[i] * r) * (1.0F + w[lane * kDimsPerLane + i]);
    float partner[kDimsPerLane];
    for (unsigned i = 0; i < kDimsPerLane; ++i) partner[i] = __shfl_xor(x[i], 8, 32);
    if (lane < 8) {
        for (unsigned i = 0; i < kDimsPerLane; ++i) {
            const unsigned d = lane * kDimsPerLane + i;
            x[i] = __builtin_fmaf(-partner[i], rope.sin[d], x[i] * rope.cos[d]);
        }
    } else if (lane < 16) {
        for (unsigned i = 0; i < kDimsPerLane; ++i) {
            const unsigned d = (lane - 8) * kDimsPerLane + i;
            x[i] = __builtin_fmaf(partner[i], rope.sin[d], x[i] * rope.cos[d]);
        }
    }
}
__device__ __forceinline__ void load4(float* x, const float* src, unsigned lane) {
    const float4 f = *reinterpret_cast<const float4*>(src + lane * kDimsPerLane);
    x[0] = f.x;
    x[1] = f.y;
    x[2] = f.z;
    x[3] = f.w;
}
__device__ __forceinline__ uint2 pack4(const float* x) {
    return make_uint2(static_cast<std::uint32_t>(float_to_bf16(x[0])) | (static_cast<std::uint32_t>(float_to_bf16(x[1])) << 16U),
                      static_cast<std::uint32_t>(float_to_bf16(x[2])) | (static_cast<std::uint32_t>(float_to_bf16(x[3])) << 16U));
}

// dot128 of the host reference evaluated by one thread on a BF16 key held as 64 packed words: 32 lane chains of 4
// fmaf, then the butterfly's lane-0 tree (K2/K4 order).
__device__ __forceinline__ float dot128_packed(const float* q, const std::uint32_t* kw) {
    float part[kLanes];
#pragma unroll
    for (unsigned l = 0; l < kLanes; ++l) {
        const float4 qq = *reinterpret_cast<const float4*>(q + l * kDimsPerLane);
        float a = 0.0F;
        a = __builtin_fmaf(qq.x, lo_bf16(kw[2 * l]), a);
        a = __builtin_fmaf(qq.y, hi_bf16(kw[2 * l]), a);
        a = __builtin_fmaf(qq.z, lo_bf16(kw[2 * l + 1]), a);
        a = __builtin_fmaf(qq.w, hi_bf16(kw[2 * l + 1]), a);
        part[l] = a;
    }
#pragma unroll
    for (unsigned d = 16; d >= 1; d >>= 1U)
#pragma unroll
        for (unsigned l = 0; l < d; ++l) part[l] = part[l] + part[l + d];
    return part[0];
}

// A 128-key BF16 tile in LDS: key t's 16-byte piece c lives at t * 16 + (c ^ (t & 15)), so a wave reading one row per
// lane hits distinct banks.
__device__ __forceinline__ unsigned swz(unsigned key, unsigned piece) { return key * 16U + (piece ^ (key & 15U)); }
__device__ __forceinline__ void read_row(const uint4* tile, unsigned key, std::uint32_t* kw) {
#pragma unroll
    for (unsigned c = 0; c < 16; ++c) {
        const uint4 u = tile[swz(key, c)];
        kw[4 * c] = u.x;
        kw[4 * c + 1] = u.y;
        kw[4 * c + 2] = u.z;
        kw[4 * c + 3] = u.w;
    }
}

// ---------------------------------------------------------------- index launch (K1 + K2)

struct Item {
    unsigned s, j;            // sequence, local block index
    std::uint64_t b;          // global block id
    std::uint64_t count;      // visible positions of the sequence's last row
};
__device__ __forceinline__ Item item_of(const DecodeView& v, unsigned item) {
    Item it;
    it.s = item % v.seqs;
    it.j = item / v.seqs;
    it.b = static_cast<std::uint64_t>(it.j) * v.shard_count + v.shard_rank;
    it.count = v.pos[it.s * v.rows_per_seq + v.rows_per_seq - 1] + 1;
    return it;
}
// First item >= item (stepping by the grid) whose block holds visible positions; items when none.
__device__ __forceinline__ unsigned next_valid(const DecodeView& v, unsigned item, unsigned items) {
    for (; item < items; item += gridDim.x) {
        const Item it = item_of(v, item);
        if (it.b * kBlock < it.count) return item;
    }
    return items;
}
__device__ __forceinline__ void load_tile(const DecodeView& v, unsigned item, uint4* pre) {
    const Item it = item_of(v, item);
    const uint4* src = reinterpret_cast<const uint4*>(v.ik_cache[it.s] + it.b * kBlock * kIdxDim);
#pragma unroll
    for (unsigned i = 0; i < kChunks16 / kIdxThreads; ++i) {
        const unsigned e = i * kIdxThreads + threadIdx.x;
        pre[i] = it.b * kBlock + (e >> 4U) < it.count ? src[e] : make_uint4(0U, 0U, 0U, 0U);
    }
}

// K1 work that nothing in this launch reads: K (norm + RoPE) and V rows to the caches, and (sharded) the index keys
// of rows whose block another shard scores. Items spread over all blocks and waves.
__device__ void prep_kv(const DecodeView& v) {
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U, rows = v.seqs * v.rows_per_seq;
    const unsigned per_row = 2 * v.kv_heads + (v.shard_count > 1 ? 1U : 0U);
    const unsigned n = rows * per_row, waves = gridDim.x * (kIdxThreads / 32);
    for (unsigned w = blockIdx.x * (kIdxThreads / 32) + wave; w < n; w += waves) {
        const unsigned r = w / per_row, kind = w % per_row, s = r / v.rows_per_seq;
        const std::uint64_t p = v.pos[r];
        float x[kDimsPerLane];
        if (kind < v.kv_heads) {
            load4(x, v.k + (static_cast<std::size_t>(r) * v.kv_heads + kind) * kHeadDim, lane);
            norm_rope_lane(x, v.k_w, v.rope[r], lane);
            *reinterpret_cast<uint2*>(v.k_cache[s] + (p * v.kv_heads + kind) * kHeadDim + lane * kDimsPerLane) = pack4(x);
        } else if (kind < 2 * v.kv_heads) {
            const unsigned g = kind - v.kv_heads;
            load4(x, v.v + (static_cast<std::size_t>(r) * v.kv_heads + g) * kHeadDim, lane);
            *reinterpret_cast<uint2*>(v.v_cache[s] + (p * v.kv_heads + g) * kHeadDim + lane * kDimsPerLane) = pack4(x);
        } else if ((p / kBlock) % v.shard_count != v.shard_rank) {
            load4(x, v.ik + static_cast<std::size_t>(r) * kIdxDim, lane);
            norm_rope_lane(x, v.ik_w, v.rope[r], lane);
            *reinterpret_cast<uint2*>(v.ik_cache[s] + p * kIdxDim + lane * kDimsPerLane) = pack4(x);
        }
    }
}

// Preserve local row/block coordinates and operation order inside each window.
__host__ __device__ DecodeView prefill_offset(DecodeView v, unsigned r0) {
    v.q += static_cast<std::size_t>(r0) * v.q_heads * kHeadDim;
    v.k += static_cast<std::size_t>(r0) * v.kv_heads * kHeadDim;
    v.v += static_cast<std::size_t>(r0) * v.kv_heads * kHeadDim;
    v.iq += static_cast<std::size_t>(r0) * v.idx_heads * kIdxDim;
    v.ik += static_cast<std::size_t>(r0) * kIdxDim;
    v.rope += r0;
    v.pos += r0;
    v.o += static_cast<std::size_t>(r0) * v.q_heads * kHeadDim;
    return v;
}
__device__ DecodeView kernel_view(const DecodeView& v, unsigned) { return v; }
__device__ DecodeView kernel_view(const PrefillView& p, unsigned w) {
    const unsigned r0 = (p.first_window + w) * kMaxRows;
    DecodeView v = prefill_offset(p.window, r0);
    v.seqs = 1;
    v.rows_per_seq = p.rows - r0 < kMaxRows ? p.rows - r0 : kMaxRows;
    v.scores += static_cast<std::size_t>(r0) * v.idx_heads * v.max_blocks;
    v.idx += static_cast<std::size_t>(r0) * v.kv_heads * kTopK;
    v.idx_count += static_cast<std::size_t>(r0) * v.kv_heads;
    v.partial += static_cast<std::size_t>(r0) * v.q_heads * kMaxChunks * 130;
    v.ticket += static_cast<std::size_t>(r0) * v.kv_heads * 4;
    return v;
}
// Tag only the new producer specialization; existing kernel template signatures,
// view layouts and instantiations remain unchanged for modes 0/1/2.
struct RepairedPrefillView : PrefillView {};
template <class View> inline constexpr bool repaired_prefill = false;
template <> inline constexpr bool repaired_prefill<RepairedPrefillView> = true;
template <class View> inline constexpr bool prefill_kernel = false;
template <> inline constexpr bool prefill_kernel<PrefillView> = true;
template <> inline constexpr bool prefill_kernel<RepairedPrefillView> = true;

// One writer per cache row/head. Uses the same lane norm, RoPE and packing as window prep.
__global__ __launch_bounds__(kIdxThreads) void prefill_prep_kernel(PrefillView p) {
    const DecodeView& v = p.window;
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    const unsigned per_row = 2 * v.kv_heads + 1;
    for (unsigned item = blockIdx.x * (kIdxThreads / 32) + wave; item < p.rows * per_row;
         item += gridDim.x * (kIdxThreads / 32)) {
        const unsigned r = item / per_row, kind = item % per_row;
        const std::uint64_t pos = v.pos[r];
        float x[kDimsPerLane];
        if (kind < v.kv_heads) {
            load4(x, v.k + (static_cast<std::size_t>(r) * v.kv_heads + kind) * kHeadDim, lane);
            norm_rope_lane(x, v.k_w, v.rope[r], lane);
            *reinterpret_cast<uint2*>(v.k_cache[0] + (pos * v.kv_heads + kind) * kHeadDim + lane * kDimsPerLane) = pack4(x);
        } else if (kind < 2 * v.kv_heads) {
            const unsigned g = kind - v.kv_heads;
            load4(x, v.v + (static_cast<std::size_t>(r) * v.kv_heads + g) * kHeadDim, lane);
            *reinterpret_cast<uint2*>(v.v_cache[0] + (pos * v.kv_heads + g) * kHeadDim + lane * kDimsPerLane) = pack4(x);
        } else {
            load4(x, v.ik + static_cast<std::size_t>(r) * kIdxDim, lane);
            norm_rope_lane(x, v.ik_w, v.rope[r], lane);
            *reinterpret_cast<uint2*>(v.ik_cache[0] + pos * kIdxDim + lane * kDimsPerLane) = pack4(x);
        }
    }
}

template <class View>
__global__ __launch_bounds__(kIdxThreads) void index_kernel(View input) {
    const DecodeView v = kernel_view(input, blockIdx.y);
    extern __shared__ uint4 smem[];
    uint4* tile = smem;                                          // kBlock x 16 pieces (32 KB)
    float* iqs = reinterpret_cast<float*>(smem + kChunks16);     // R x idx_heads x 128, normed
    __shared__ float wave_max[kMaxRows * kMaxIdxHeads * (kIdxThreads / 32)];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const unsigned W = v.rows_per_seq, H = v.idx_heads, rows = v.seqs * W;
    const unsigned local = (v.max_blocks + v.shard_count - 1 - v.shard_rank) / v.shard_count;
    const unsigned items = v.seqs * local;
    std::uint64_t* ph = v.phase_index != nullptr && tid == 0 ? v.phase_index + static_cast<std::size_t>(blockIdx.x) * 4 : nullptr;
    if (ph != nullptr) ph[0] = static_cast<std::uint64_t>(wall_clock64());
    std::uint64_t n_items = 0;
    if constexpr (!prefill_kernel<View>) prep_kv(v);
    for (unsigned n = wave; n < rows * H; n += kIdxThreads / 32) {  // normed index q of every row, once per block
        const unsigned r = n / H;
        float x[kDimsPerLane];
        load4(x, v.iq + static_cast<std::size_t>(n) * kIdxDim, lane);
        norm_rope_lane(x, v.iq_w, v.rope[r], lane);
        *reinterpret_cast<float4*>(iqs + static_cast<std::size_t>(n) * kIdxDim + lane * kDimsPerLane) =
            make_float4(x[0], x[1], x[2], x[3]);
    }
    if (ph != nullptr) ph[1] = static_cast<std::uint64_t>(wall_clock64());
    // No register prefetch of the next tile: with the grid cap enough blocks are resident to cover the latency, and
    // the 64 extra VGPRs made the compiler spill.
    for (unsigned item = next_valid(v, blockIdx.x, items); item < items; item = next_valid(v, item + gridDim.x, items)) {
        const Item it = item_of(v, item);
        ++n_items;
        {
            uint4 pre[kChunks16 / kIdxThreads];
            load_tile(v, item, pre);
#pragma unroll
            for (unsigned i = 0; i < kChunks16 / kIdxThreads; ++i) {
                const unsigned e = i * kIdxThreads + tid;
                tile[swz(e >> 4U, e & 15U)] = pre[i];
            }
        }
        // A block holding new rows: their slots are overwritten below, after every stale store landed.
        if (it.b >= v.pos[it.s * W] / kBlock) __syncthreads();
        // This block's new rows: index key norm + RoPE, to the cache and over the (stale) tile slot.
        for (unsigned rr = wave; rr < W; rr += kIdxThreads / 32) {
            const unsigned r = it.s * W + rr;
            const std::uint64_t p = v.pos[r];
            if (p / kBlock != it.b) continue;
            float x[kDimsPerLane];
            load4(x, v.ik + static_cast<std::size_t>(r) * kIdxDim, lane);
            norm_rope_lane(x, v.ik_w, v.rope[r], lane);
            const uint2 packed = pack4(x);
            if constexpr (!prefill_kernel<View>)
                *reinterpret_cast<uint2*>(v.ik_cache[it.s] + p * kIdxDim + lane * kDimsPerLane) = packed;
            const unsigned t = static_cast<unsigned>(p - it.b * kBlock);
            reinterpret_cast<uint2*>(tile + swz(t, lane >> 1U))[lane & 1U] = packed;
        }
        __syncthreads();
        std::uint32_t kw[64];  // packed: converted per dot (128 floats + hoisted q loads spill)
        read_row(tile, tid, kw);
        const std::uint64_t t = it.b * kBlock + tid;
#pragma unroll 1
        for (unsigned rr = 0; rr < W; ++rr) {
            const unsigned r = it.s * W + rr;
            const std::uint64_t count = v.pos[r] + 1;
            if (it.b * kBlock >= count) continue;  // uniform over the block
#pragma unroll 1
            for (unsigned h = 0; h < H; ++h) {
                float sc = -__builtin_inff();
                if (t < count) {
                    // Opaque to the optimizer: the conversions stay inside the dot instead of being hoisted.
                    for (unsigned i = 0; i < 64; ++i) asm volatile("" : "+v"(kw[i]));
                    sc = dot128_packed(iqs + (static_cast<std::size_t>(r) * H + h) * kIdxDim, kw) * kScale;
                }
                for (unsigned d = 16; d >= 1; d >>= 1U) sc = max_of(sc, __shfl_xor(sc, static_cast<int>(d), 32));
                if (lane == 0) wave_max[(rr * H + h) * (kIdxThreads / 32) + wave] = sc;
            }
        }
        __syncthreads();
        if (tid < W * H) {
            const unsigned rr = tid / H, h = tid % H, r = it.s * W + rr;
            if (it.b * kBlock < v.pos[r] + 1) {
                const float* wm = wave_max + tid * (kIdxThreads / 32);
                float m = wm[0];
                for (unsigned w = 1; w < kIdxThreads / 32; ++w) m = max_of(m, wm[w]);
                v.scores[(static_cast<std::size_t>(r) * H + h) * v.max_blocks + it.j] = m;
            }
        }
        // The next iteration's tile stores and wave_max writes come after the finishing threads are done reading
        // wave_max: one more barrier.
        __syncthreads();
    }
    if (ph != nullptr) {
        ph[2] = static_cast<std::uint64_t>(wall_clock64());
        ph[3] = n_items;
    }
}

__global__ __launch_bounds__(kIdxThreads) void prefill_index_tile_kernel(PrefillView input, unsigned query_rows) {
    const unsigned r0 = input.first_window * kMaxRows + blockIdx.y * query_rows;
    DecodeView v = kernel_view(input, blockIdx.y * (query_rows / kMaxRows));
    v.rows_per_seq = input.rows - r0 < query_rows ? input.rows - r0 : query_rows;
    extern __shared__ uint4 smem[];
    uint4* tile = smem;                                          // kBlock x 16 pieces (32 KB)
    float* iqs = reinterpret_cast<float*>(smem + kChunks16);     // R x idx_heads x 128, normed
    float* wave_max = reinterpret_cast<float*>(tile);  // keys are in kw before maxima overwrite LDS
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const unsigned W = v.rows_per_seq, H = v.idx_heads, rows = v.seqs * W;
    const unsigned local = (v.max_blocks + v.shard_count - 1 - v.shard_rank) / v.shard_count;
    const unsigned items = v.seqs * local;
    for (unsigned n = wave; n < rows * H; n += kIdxThreads / 32) {  // normed index q of every row, once per block
        const unsigned r = n / H;
        float x[kDimsPerLane];
        load4(x, v.iq + static_cast<std::size_t>(n) * kIdxDim, lane);
        norm_rope_lane(x, v.iq_w, v.rope[r], lane);
        *reinterpret_cast<float4*>(iqs + static_cast<std::size_t>(n) * kIdxDim + lane * kDimsPerLane) =
            make_float4(x[0], x[1], x[2], x[3]);
    }
    // Keep packed keys in registers across all query rows.
    for (unsigned item = next_valid(v, blockIdx.x, items); item < items; item = next_valid(v, item + gridDim.x, items)) {
        const Item it = item_of(v, item);
        {
            uint4 pre[kChunks16 / kIdxThreads];
            load_tile(v, item, pre);
#pragma unroll
            for (unsigned i = 0; i < kChunks16 / kIdxThreads; ++i) {
                const unsigned e = i * kIdxThreads + tid;
                tile[swz(e >> 4U, e & 15U)] = pre[i];
            }
        }
        __syncthreads();
        std::uint32_t kw[64];  // packed: converted per dot (128 floats + hoisted q loads spill)
        read_row(tile, tid, kw);
        for (unsigned i = 0; i < 64; ++i) asm volatile("" : "+v"(kw[i]));
        __syncthreads();  // all key reads precede the aliased wave maxima
        const std::uint64_t t = it.b * kBlock + tid;
#pragma unroll 1
        for (unsigned rr = 0; rr < W; ++rr) {
            const unsigned r = it.s * W + rr;
            const std::uint64_t count = v.pos[r] + 1;
            if (it.b * kBlock >= count) continue;  // uniform over the block
#pragma unroll 1
            for (unsigned h = 0; h < H; ++h) {
                float sc = -__builtin_inff();
                if (t < count) {
                    // Opaque to the optimizer: the conversions stay inside the dot instead of being hoisted.
                    for (unsigned i = 0; i < 64; ++i) asm volatile("" : "+v"(kw[i]));
                    sc = dot128_packed(iqs + (static_cast<std::size_t>(r) * H + h) * kIdxDim, kw) * kScale;
                }
                for (unsigned d = 16; d >= 1; d >>= 1U) sc = max_of(sc, __shfl_xor(sc, static_cast<int>(d), 32));
                if (lane == 0) wave_max[(rr * H + h) * (kIdxThreads / 32) + wave] = sc;
            }
        }
        __syncthreads();
        if (tid < W * H) {
            const unsigned rr = tid / H, h = tid % H, r = it.s * W + rr;
            if (it.b * kBlock < v.pos[r] + 1) {
                const float* wm = wave_max + tid * (kIdxThreads / 32);
                float m = wm[0];
                for (unsigned w = 1; w < kIdxThreads / 32; ++w) m = max_of(m, wm[w]);
                v.scores[(static_cast<std::size_t>(r) * H + h) * v.max_blocks + it.j] = m;
            }
        }
        // The next iteration's tile stores and wave_max writes come after the finishing threads are done reading
        // wave_max: one more barrier.
        __syncthreads();
    }
}

// Q64 macrotile: one key load/read_row per 64 queries, four LDS subtiles.
// Keep dot128_packed's per-dot widening and exact four-FMA/butterfly order.
__global__ __launch_bounds__(kIdxThreads) void prefill_index_q64_kernel(PrefillView input) {
    constexpr unsigned query_rows = 64;
    static_assert(kMaxRows * 2 * kIdxDim * sizeof(float) +
                  kMaxRows * 2 * (kIdxThreads / 32) * sizeof(float) <= kChunks16 * sizeof(uint4));
    const unsigned r0 = input.first_window * kMaxRows + blockIdx.y * query_rows;
    DecodeView v = kernel_view(input, blockIdx.y * (query_rows / kMaxRows));
    v.rows_per_seq = input.rows - r0 < query_rows ? input.rows - r0 : query_rows;
    extern __shared__ uint4 smem[];
    uint4* tile = smem;                                          // kBlock x 16 pieces (32 KB)
    // Once read_row has retired, packed keys live only in kw. Overlay the key
    // staging area with one 16-query subtile (16 KiB) and its maxima (512 B).
    // Two index heads are enforced by enqueue_prefill_index.
    float* iqs = reinterpret_cast<float*>(smem);
    float* wave_max = iqs + kMaxRows * 2 * kIdxDim;
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const unsigned W = v.rows_per_seq, H = v.idx_heads;
    const unsigned local = (v.max_blocks + v.shard_count - 1 - v.shard_rank) / v.shard_count;
    const unsigned items = v.seqs * local;
    // Keep packed keys in registers across all query rows.
    for (unsigned item = next_valid(v, blockIdx.x, items); item < items; item = next_valid(v, item + gridDim.x, items)) {
        const Item it = item_of(v, item);
        {
            uint4 pre[kChunks16 / kIdxThreads];
            load_tile(v, item, pre);
#pragma unroll
            for (unsigned i = 0; i < kChunks16 / kIdxThreads; ++i) {
                const unsigned e = i * kIdxThreads + tid;
                tile[swz(e >> 4U, e & 15U)] = pre[i];
            }
        }
        __syncthreads();
        std::uint32_t kw[64];  // packed: converted per dot (128 floats + hoisted q loads spill)
        read_row(tile, tid, kw);
        for (unsigned i = 0; i < 64; ++i) asm volatile("" : "+v"(kw[i]));
        __syncthreads();  // all key reads precede the aliased wave maxima
        const std::uint64_t t = it.b * kBlock + tid;
#pragma unroll 1
        for (unsigned q0 = 0; q0 < W; q0 += kMaxRows) {
            const unsigned subrows = W - q0 < kMaxRows ? W - q0 : kMaxRows;
            for (unsigned n = wave; n < subrows * H; n += kIdxThreads / 32) {
                const unsigned r = q0 + n / H;
                float x[kDimsPerLane];
                load4(x, v.iq + (static_cast<std::size_t>(q0) * H + n) * kIdxDim, lane);
                norm_rope_lane(x, v.iq_w, v.rope[r], lane);
                *reinterpret_cast<float4*>(iqs + static_cast<std::size_t>(n) * kIdxDim + lane * kDimsPerLane) =
                    make_float4(x[0], x[1], x[2], x[3]);
            }
            __syncthreads();  // all subtile queries precede dot reads
#pragma unroll 1
            for (unsigned rr = 0; rr < subrows; ++rr) {
                const unsigned r = q0 + rr;
                const std::uint64_t count = v.pos[r] + 1;
                if (it.b * kBlock >= count) continue;  // uniform over the block
#pragma unroll 1
                for (unsigned h = 0; h < H; ++h) {
                    float sc = -__builtin_inff();
                    if (t < count) {
                        // Opaque to the optimizer: the conversions stay inside the dot instead of being hoisted.
                        for (unsigned i = 0; i < 64; ++i) asm volatile("" : "+v"(kw[i]));
                        sc = dot128_packed(iqs + (static_cast<std::size_t>(rr) * H + h) * kIdxDim, kw) * kScale;
                    }
                    for (unsigned d = 16; d >= 1; d >>= 1U) sc = max_of(sc, __shfl_xor(sc, static_cast<int>(d), 32));
                    if (lane == 0) wave_max[(rr * H + h) * (kIdxThreads / 32) + wave] = sc;
                }
            }
            __syncthreads();
            if (tid < subrows * H) {
                const unsigned rr = tid / H, h = tid % H, r = q0 + rr;
                if (it.b * kBlock < v.pos[r] + 1) {
                    const float* wm = wave_max + tid * (kIdxThreads / 32);
                    float m = wm[0];
                    for (unsigned w = 1; w < kIdxThreads / 32; ++w) m = max_of(m, wm[w]);
                    v.scores[(static_cast<std::size_t>(r) * H + h) * v.max_blocks + it.j] = m;
                }
            }
            // Finish all maxima reads before the next subtile or key tile overwrites LDS.
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------- attend launch (K3 + K4 + combine)

__device__ __forceinline__ std::uint32_t order_key(float x) {
    const std::uint32_t b = f2u(x == 0.0F ? 0.0F : x);
    return (b & 0x80000000U) != 0U ? ~b : (b | 0x80000000U);
}
// Inclusive block scan of one flag per thread (NT threads); returns the prefix and the block total.
template <unsigned NT> __device__ __forceinline__ unsigned block_scan(bool flag, unsigned* wave_tot, unsigned& total) {
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    const unsigned mask = static_cast<unsigned>(__ballot(flag));
    const unsigned in_wave = static_cast<unsigned>(__popc(mask & ((2U << lane) - 1U)));
    if (lane == 31U) wave_tot[wave] = static_cast<unsigned>(__popc(mask));
    __syncthreads();
    unsigned before = 0;
    total = 0;
    for (unsigned w = 0; w < NT / 32; ++w) {
        if (w < wave) before += wave_tot[w];
        total += wave_tot[w];
    }
    __syncthreads();
    return before + in_wave;
}
// Marks chosen[i] for the k best of n keys by (key desc, index asc), n > k: rank by counting for small n, else a
// 4-pass 8-bit radix threshold (digit found by a parallel suffix scan of the histogram; each thread owns 256 / NT
// consecutive digits from the top) and the ties at the threshold in index order. Same total order as the K3 kernel,
// so the same set. Integer counts only.
template <unsigned NT>
__device__ void choose_topk(const std::uint32_t* keys, std::uint32_t n, std::uint32_t k, std::uint8_t* chosen,
                            unsigned* hist, unsigned* wave_tot, std::uint32_t* sh) {
    constexpr unsigned D = NT >= 256 ? 1U : 256 / NT;  // digits per thread (threads past 256 own none)
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    if (n <= kCountMax) {
        for (std::uint32_t i = tid; i < n; i += NT) {
            const std::uint32_t key = keys[i];
            std::uint32_t beaten = 0;
            for (std::uint32_t o = 0; o < n; ++o) {
                const std::uint32_t ko = keys[o];
                beaten += (ko > key || (ko == key && o < i)) ? 1U : 0U;
            }
            chosen[i] = beaten < k ? 1U : 0U;
        }
        __syncthreads();
        return;
    }
    std::uint32_t prefix = 0, prefix_mask = 0, need = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned i = tid; i < 256; i += NT) hist[i] = 0;
        __syncthreads();
        for (unsigned i = tid; i < n; i += NT)
            if ((keys[i] & prefix_mask) == prefix) atomicAdd(&hist[(keys[i] >> shift) & 255U], 1U);
        __syncthreads();
        // Thread t owns digits 255 - (t * D + i), i < D: count of keys with digit >= each, by an inclusive scan.
        unsigned h[D], own = 0;
        if (tid * D < 256) {
            for (unsigned i = 0; i < D; ++i) {
                h[i] = hist[255U - (tid * D + i)];
                own += h[i];
            }
        }
        unsigned incl = own;
        for (unsigned d = 1; d < 32; d <<= 1U) {
            const unsigned y = static_cast<unsigned>(__shfl_up(static_cast<int>(incl), d, 32));
            if (lane >= d) incl += y;
        }
        if (lane == 31U) wave_tot[wave] = incl;
        __syncthreads();
        if (tid * D < 256) {
            for (unsigned w = 0; w < wave; ++w) incl += wave_tot[w];
            unsigned above = incl - own;  // keys with a larger digit than this thread's first ("above" of the serial search)
            for (unsigned i = 0; i < D; ++i) {
                if (above < need && above + h[i] >= need) {
                    sh[0] = prefix | ((255U - (tid * D + i)) << shift);
                    sh[1] = need - above;
                }
                above += h[i];
            }
        }
        __syncthreads();
        prefix = sh[0];
        need = sh[1];
        prefix_mask |= 255U << shift;
    }
    unsigned ties_before = 0;
    for (std::uint32_t base = 0; base < n; base += NT) {
        const std::uint32_t i = base + tid;
        const std::uint32_t key = i < n ? keys[i] : 0U;
        const bool tie = i < n && key == prefix;
        unsigned tie_total = 0;
        const unsigned tie_rank = ties_before + block_scan<NT>(tie, wave_tot, tie_total);  // inclusive
        if (i < n) chosen[i] = (key > prefix || (tie && tie_rank <= need)) ? 1U : 0U;
        ties_before += tie_total;
    }
    __syncthreads();
}

template <unsigned NT> struct SelScratch {
    std::uint32_t keys[kMaxBlocks];
    std::uint8_t chosen[kMaxBlocks];
    unsigned hist[256];
    unsigned wave_tot[NT / 32];
    std::uint32_t sh[2];
};
template <unsigned NT> union AttnSmem {
    SelScratch<NT> sel;
    uint4 tile[kChunks16];  // K tile (swizzled), then V tile (row-major)
};

// Selection scratch carved from dynamic LDS for max_blocks candidates (keys, chosen flags, histogram, wave totals,
// 3 words: radix prefix / need, and the picked block id).
template <unsigned NT> struct SelView {
    std::uint32_t* keys;
    std::uint8_t* chosen;
    unsigned* hist;
    unsigned* wave_tot;
    std::uint32_t* sh;
    __device__ SelView(uint4* base, std::uint32_t max_blocks) {
        const std::uint32_t cap = (max_blocks + 15U) & ~15U;
        keys = reinterpret_cast<std::uint32_t*>(base);
        chosen = reinterpret_cast<std::uint8_t*>(keys + cap);
        hist = reinterpret_cast<unsigned*>(chosen + cap);
        wave_tot = hist + 256;
        sh = wave_tot + NT / 32;
    }
};
__host__ __device__ constexpr std::size_t sel_bytes(std::uint32_t max_blocks, unsigned nt) {
    return ((max_blocks + 15U) & ~15U) * 5U + 256U * 4U + nt / 32U * 4U + 3U * 4U;
}

__device__ __forceinline__ void stamp(std::uint64_t* phase, unsigned k) {
    if (phase != nullptr && threadIdx.x == 0) {
        std::uint64_t* p = phase + (static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x) * kPhases;
        p[k] = static_cast<std::uint64_t>(wall_clock64());
        p[8 + k] = __builtin_readcyclecounter();  // gfx11: SHADER_CYCLES, 20 bits (phase deltas only)
    }
}

// One block per (row, KV head, head part, chunk): HW waves = HW query heads of the KV group (kGroup / HW parts).
// DENSE: every row sees at most kTopK blocks (pos < 2048), no index launch: the block writes its own new K/V rows
// (and chunk 0 the row's index key) and patches them into the LDS tiles.
template <unsigned HW, bool DENSE> __global__ __launch_bounds__(HW * 32) void attend_kernel(DecodeView v, int select) {
    constexpr unsigned NT = HW * 32, PARTS = kGroup / HW;
    __shared__ AttnSmem<NT> S;
    __shared__ alignas(16) float qs[HW * kHeadDim];
    __shared__ std::uint32_t pick[2];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const unsigned rgp = blockIdx.x, part = rgp % PARTS, rg = rgp / PARTS, r = rg / v.kv_heads, g = rg % v.kv_heads;
    const unsigned c = blockIdx.y, W = v.rows_per_seq, s = r / W, q_heads = v.kv_heads * kGroup;
    const std::uint64_t count = v.pos[r] + 1;
    stamp(v.phase, 0);
    std::uint32_t cnt = 0, blk = 0;
    const std::uint32_t nb = static_cast<std::uint32_t>(block_count(count));
    if (DENSE || nb <= kTopK) {
        cnt = nb;
        blk = c;
        if (!DENSE && select != 0 && c == 0 && part == 0 && tid < nb) v.idx[static_cast<std::size_t>(rg) * kTopK + tid] = tid;
    } else if (select != 0) {
        cnt = kTopK;
        const std::uint32_t rest = nb - 1;
        const float* sc = v.scores + static_cast<std::size_t>(rg) * v.max_blocks;  // index head g = KV head g
        for (unsigned b = tid; b < rest; b += NT) S.sel.keys[b] = order_key(sc[b]);
        __syncthreads();
        choose_topk<NT>(S.sel.keys, rest, kTopK - 1, S.sel.chosen, S.sel.hist, S.sel.wave_tot, S.sel.sh);
        const bool out = c == 0 && part == 0;
        unsigned before = 0;  // ascending compaction (block scans), slot c's id is this block's chunk
        for (std::uint32_t base = 0; base < rest; base += NT) {
            const std::uint32_t i = base + tid;
            const bool ch = i < rest && S.sel.chosen[i] != 0U;
            unsigned total = 0;
            const unsigned rank = before + block_scan<NT>(ch, S.sel.wave_tot, total);
            if (ch) {
                if (rank - 1 == c) pick[0] = i;
                if (out) v.idx[static_cast<std::size_t>(rg) * kTopK + rank - 1] = i;
            }
            before += total;
        }
        if (c == kTopK - 1 && tid == 0) pick[0] = rest;  // the newest block, always last
        if (out && tid == 0) v.idx[static_cast<std::size_t>(rg) * kTopK + kTopK - 1] = rest;
        __syncthreads();
        blk = pick[0];
    } else {
        cnt = v.idx_count[rg];
        if (c < cnt) blk = v.idx[static_cast<std::size_t>(rg) * kTopK + c];
    }
    if (!DENSE && select != 0 && c == 0 && part == 0 && tid == 0) v.idx_count[rg] = cnt;
    if (c >= cnt) return;
    stamp(v.phase, 1);
    const std::uint64_t lo = static_cast<std::uint64_t>(blk) * kBlock;
    const unsigned n_keys = static_cast<unsigned>((count < lo + kBlock ? count : lo + kBlock) - lo);
    std::uint16_t* kc = v.k_cache[s];
    std::uint16_t* vc = v.v_cache[s];
    constexpr unsigned PER = kChunks16 / NT;  // 16-byte pieces per thread and tile
    {
        uint4 kreg[PER];
#pragma unroll
        for (unsigned i = 0; i < PER; ++i) {
            const unsigned e = i * NT + tid, key = e >> 4U, piece = e & 15U;
            kreg[i] = key < n_keys ? *reinterpret_cast<const uint4*>(kc + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                   : make_uint4(0U, 0U, 0U, 0U);
        }
        const unsigned qh = g * kGroup + part * HW + wave;
        float x[kDimsPerLane];
        load4(x, v.q + (static_cast<std::size_t>(r) * q_heads + qh) * kHeadDim, lane);
        norm_rope_lane(x, v.q_w, v.rope[r], lane);
        *reinterpret_cast<float4*>(qs + wave * kHeadDim + lane * kDimsPerLane) = make_float4(x[0], x[1], x[2], x[3]);
        __syncthreads();  // selection scratch is dead from here: the tile reuses it
#pragma unroll
        for (unsigned i = 0; i < PER; ++i) {
            const unsigned e = i * NT + tid;
            S.tile[swz(e >> 4U, e & 15U)] = kreg[i];
        }
    }
    if (DENSE) {
        // New rows of this sequence inside this block and visible to row r: K row norm + RoPE over the stale slot
        // (and to the cache: every block writing a row writes the same bits). Chunk 0 also stores row r's index key.
        __syncthreads();
        for (unsigned rr = wave; rr < W; rr += HW) {
            const unsigned rw = s * W + rr;
            const std::uint64_t p = v.pos[rw];
            if (p < lo || p >= lo + n_keys) continue;
            float x[kDimsPerLane];
            load4(x, v.k + (static_cast<std::size_t>(rw) * v.kv_heads + g) * kHeadDim, lane);
            norm_rope_lane(x, v.k_w, v.rope[rw], lane);
            const uint2 packed = pack4(x);
            *reinterpret_cast<uint2*>(kc + (p * v.kv_heads + g) * kHeadDim + lane * kDimsPerLane) = packed;
            reinterpret_cast<uint2*>(S.tile + swz(static_cast<unsigned>(p - lo), lane >> 1U))[lane & 1U] = packed;
        }
        if (c == 0 && g == 0 && part == 0 && wave == 0) {
            float x[kDimsPerLane];
            load4(x, v.ik + static_cast<std::size_t>(r) * kIdxDim, lane);
            norm_rope_lane(x, v.ik_w, v.rope[r], lane);
            *reinterpret_cast<uint2*>(v.ik_cache[s] + v.pos[r] * kIdxDim + lane * kDimsPerLane) = pack4(x);
        }
    }
    // V tile loads in flight during the scores.
    uint4 vreg[PER];
#pragma unroll
    for (unsigned i = 0; i < PER; ++i) {
        const unsigned e = i * NT + tid, key = e >> 4U, piece = e & 15U;
        vreg[i] = key < n_keys ? *reinterpret_cast<const uint4*>(vc + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                               : make_uint4(0U, 0U, 0U, 0U);
    }
    __syncthreads();
    stamp(v.phase, 2);
    // Scores: lane j of group k0 owns key k0 + j (dot128 in the declared order by one thread).
    const float* q = qs + wave * kHeadDim;
    float sv[kBlock / 32];
#pragma unroll
    for (unsigned grp = 0; grp < kBlock / 32; ++grp) {
        const unsigned key = grp * 32 + lane;
        sv[grp] = -__builtin_inff();
        if (key < n_keys) {
            std::uint32_t kw[64];
            read_row(S.tile, key, kw);
            sv[grp] = dot128_packed(q, kw) * kScale;
        }
    }
    __syncthreads();
    stamp(v.phase, 3);
#pragma unroll
    for (unsigned i = 0; i < PER; ++i) S.tile[i * NT + tid] = vreg[i];
    if (DENSE) {
        __syncthreads();
        for (unsigned rr = wave; rr < W; rr += HW) {
            const unsigned rw = s * W + rr;
            const std::uint64_t p = v.pos[rw];
            if (p < lo || p >= lo + n_keys) continue;
            float x[kDimsPerLane];
            load4(x, v.v + (static_cast<std::size_t>(rw) * v.kv_heads + g) * kHeadDim, lane);
            const uint2 packed = pack4(x);
            *reinterpret_cast<uint2*>(vc + (p * v.kv_heads + g) * kHeadDim + lane * kDimsPerLane) = packed;
            reinterpret_cast<uint2*>(S.tile)[static_cast<unsigned>(p - lo) * 32 + lane] = packed;
        }
    }
    __syncthreads();
    stamp(v.phase, 4);
    const uint2* vt = reinterpret_cast<const uint2*>(S.tile);  // key t, lane l: vt[t * 32 + l]
    float acc[kDimsPerLane] = {0.0F, 0.0F, 0.0F, 0.0F};
    float m = -__builtin_inff(), l = 0.0F;
    bool first = true;
#pragma unroll
    for (unsigned grp = 0; grp < kBlock / 32; ++grp) {
        const unsigned k0 = grp * 32;
        if (k0 >= n_keys) break;
        const float sg = sv[grp];
        for (unsigned t0 = k0; t0 < k0 + 32 && t0 < n_keys; t0 += kTile) {
            const unsigned nt = n_keys - t0 < kTile ? n_keys - t0 : kTile;
            float tm = -__builtin_inff();
            for (unsigned j = 0; j < nt; ++j) tm = max_of(tm, __shfl(sg, static_cast<int>(t0 - k0 + j), 32));
            const float mn = max_of(m, tm);
            const float corr = first ? 0.0F : exp_f(m - mn);
            l = l * corr;
            for (unsigned i = 0; i < kDimsPerLane; ++i) acc[i] = acc[i] * corr;
            const float p_own = exp_f(sg - mn);
            for (unsigned j = 0; j < nt; ++j) {
                const float pj = __shfl(p_own, static_cast<int>(t0 - k0 + j), 32);
                l = l + pj;
                const uint2 raw = vt[(t0 + j) * 32 + lane];
                acc[0] = __builtin_fmaf(pj, lo_bf16(raw.x), acc[0]);
                acc[1] = __builtin_fmaf(pj, hi_bf16(raw.x), acc[1]);
                acc[2] = __builtin_fmaf(pj, lo_bf16(raw.y), acc[2]);
                acc[3] = __builtin_fmaf(pj, hi_bf16(raw.y), acc[3]);
            }
            m = mn;
            first = false;
        }
    }
    const unsigned qh = g * kGroup + part * HW + wave;
    float* out = v.partial + ((static_cast<std::size_t>(r) * q_heads + qh) * kMaxChunks + c) * kPartial;
    if (lane == 0) {
        out[0] = m;
        out[1] = l;
    }
    for (unsigned i = 0; i < kDimsPerLane; ++i) out[2 + lane * kDimsPerLane + i] = acc[i];
    stamp(v.phase, 5);
    // Ticket: the last of the cnt chunk blocks of (row, KV head, part) combines. Stores drained and released at agent
    // scope before the ticket; the last block acquires before reading the other blocks' partials.
    __threadfence();
    __syncthreads();
    std::uint32_t* ticket = v.ticket + rgp;
    if (tid == 0) {
        const unsigned prev = atomicAdd(ticket, 1U);
        pick[1] = prev + 1 == cnt ? 1U : 0U;
        if (prev + 1 == cnt) atomicExch(ticket, 0U);  // ready for the next launch / graph replay
    }
    __syncthreads();
    stamp(v.phase, 6);
    if (pick[1] == 0U) return;
    __threadfence();
    const float* pt = v.partial + (static_cast<std::size_t>(r) * q_heads + qh) * kMaxChunks * kPartial;
    float big = -__builtin_inff();
    for (unsigned cc = 0; cc < cnt; ++cc) big = max_of(big, pt[cc * kPartial]);
    float L = 0.0F, e[kMaxChunks];
    for (unsigned cc = 0; cc < cnt; ++cc) {
        e[cc] = exp_f(pt[cc * kPartial] - big);
        L = __builtin_fmaf(pt[cc * kPartial + 1], e[cc], L);
    }
    float* o = v.o + (static_cast<std::size_t>(r) * q_heads + qh) * kHeadDim;
    for (unsigned i = 0; i < kDimsPerLane; ++i) {
        const unsigned d = lane * kDimsPerLane + i;
        float a = 0.0F;
        for (unsigned cc = 0; cc < cnt; ++cc) a = __builtin_fmaf(pt[cc * kPartial + 2 + d], e[cc], a);
        o[d] = a / L;
    }
    stamp(v.phase, 7);
}

// ---------------------------------------------------------------- attend v2 (same arithmetic, lane-owns-dims layout)
//
// Scores: lane l keeps its 4 dims of q in registers and reads its 8 bytes of each key row (row-major tile, no bank
// conflicts); the per-key declared dot (lane chain of 4 fmaf, then the xor butterfly 16..1) runs 8 keys at a time as a
// transposed butterfly: at xor 16 / 8 / 4 each lane keeps half of its keys and adds the partner's chain of the same key
// (own + partner: the very addition of the declared butterfly at that lane), then xor 2 and 1 as usual. Every lane of
// a group ends with the declared value (a + b == b + a), key k of the 8 in lanes with bits (4,3,2) = (k>>2, k>>1, k)&1.
// Online softmax: scores and probabilities broadcast by readlane at compile-time lanes (fully unrolled), V rows from
// LDS. Combine: the chunk partials loaded in parallel, then the declared chunk-order chains.

__device__ __forceinline__ float swz_xor(float v, int d) {
    // ds_swizzle bit mode: and 0x1F, or 0, xor d (d < 32): lane l reads lane l ^ d.
    switch (d) {
    case 16: return u2f(static_cast<std::uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int>(f2u(v)), (16 << 10) | 0x1F)));
    case 8: return u2f(static_cast<std::uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int>(f2u(v)), (8 << 10) | 0x1F)));
    case 4: return u2f(static_cast<std::uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int>(f2u(v)), (4 << 10) | 0x1F)));
    case 2: return u2f(static_cast<std::uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int>(f2u(v)), (2 << 10) | 0x1F)));
    default: return u2f(static_cast<std::uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int>(f2u(v)), (1 << 10) | 0x1F)));
    }
}
__device__ __forceinline__ float rl(float v, int lane) {
    return u2f(static_cast<std::uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(f2u(v)), lane)));
}
// Lane holding key kk (0..7) of a group of 8 after the transposed butterfly.
__host__ __device__ constexpr int key_lane(unsigned kk) {
    return static_cast<int>(((kk >> 2U) & 1U) * 16U + ((kk >> 1U) & 1U) * 8U + (kk & 1U) * 4U);
}

// Scores of keys 8 * grp .. 8 * grp + 7 (tile rows, lane's 8 bytes each): the declared dot128 of each, kScale applied.
__device__ __forceinline__ float group_scores(const uint2* tile, unsigned grp, const float* q, unsigned lane) {
    float vals[8];
#pragma unroll
    for (unsigned k = 0; k < 8; ++k) {
        const uint2 raw = tile[(grp * 8 + k) * 32 + lane];
        float a = 0.0F;
        a = __builtin_fmaf(q[0], lo_bf16(raw.x), a);
        a = __builtin_fmaf(q[1], hi_bf16(raw.x), a);
        a = __builtin_fmaf(q[2], lo_bf16(raw.y), a);
        a = __builtin_fmaf(q[3], hi_bf16(raw.y), a);
        vals[k] = a;
    }
    const bool h16 = (lane & 16U) != 0U, h8 = (lane & 8U) != 0U, h4 = (lane & 4U) != 0U;
    float a4[4];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const float own = h16 ? vals[i + 4] : vals[i], send = h16 ? vals[i] : vals[i + 4];
        a4[i] = own + swz_xor(send, 16);
    }
    float a2[2];
#pragma unroll
    for (unsigned i = 0; i < 2; ++i) {
        const float own = h8 ? a4[i + 2] : a4[i], send = h8 ? a4[i] : a4[i + 2];
        a2[i] = own + swz_xor(send, 8);
    }
    float c = (h4 ? a2[1] : a2[0]) + swz_xor(h4 ? a2[0] : a2[1], 4);
    c = c + swz_xor(c, 2);
    c = c + swz_xor(c, 1);
    return c * kScale;
}

// ROLL (version 3): the group and tile loops are rolled (scores shifted through a register window so the readlane
// lanes stay compile-time constants): same operations in the same order, a few KB of code instead of tens. At long
// contexts the index scan flushes L2/MALL and the fully unrolled version 2 refetched its straight-line code from DRAM.
template <unsigned HW, bool DENSE, bool ROLL = false, class View = DecodeView>
__global__ __launch_bounds__(HW * 32) void attend2_kernel(View input, int select) {
    const DecodeView v = kernel_view(input, blockIdx.z);
    if constexpr (prefill_kernel<View>) {
        if (blockIdx.x >= v.rows_per_seq * v.kv_heads * (kGroup / HW)) return;
    }
    constexpr unsigned NT = HW * 32, PARTS = kGroup / HW;
    // Dynamic LDS: the 32 KB K/V tile, unioned with the selection scratch sized by max_blocks (fits the tile up to
    // 6333 blocks, ~810K positions), so four 16-wave blocks share a WGP's 128 KB; the static union sized for 1M
    // positions was 42 KB, three blocks.
    extern __shared__ uint4 dyn_smem[];
    SelView<NT> sel(dyn_smem, v.max_blocks);
    uint4* const tile_mem = dyn_smem;
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const unsigned rgp = blockIdx.x, part = rgp % PARTS, rg = rgp / PARTS, r = rg / v.kv_heads, g = rg % v.kv_heads;
    const unsigned c = blockIdx.y, W = v.rows_per_seq, s = r / W, q_heads = v.kv_heads * kGroup;
    const unsigned qh = g * kGroup + part * HW + wave;
    const std::uint64_t count = v.pos[r] + 1;
    stamp(v.phase, 0);
    // The raw q of this wave's head is needed whatever the selection: load it first.
    float q[kDimsPerLane];
    load4(q, v.q + (static_cast<std::size_t>(r) * q_heads + qh) * kHeadDim, lane);
    std::uint32_t cnt = 0, blk = 0;
    const std::uint32_t nb = static_cast<std::uint32_t>(block_count(count));
    if (DENSE || nb <= kTopK) {
        cnt = nb;
        blk = c;
        if (!DENSE && select != 0 && c == 0 && part == 0 && tid < nb) v.idx[static_cast<std::size_t>(rg) * kTopK + tid] = tid;
    } else if (select != 0) {
        cnt = kTopK;
        const std::uint32_t rest = nb - 1;
        if (c == kTopK - 1) {
            blk = rest;  // the newest block, always last: no need to select
        } else {
            const float* sc = v.scores + static_cast<std::size_t>(rg) * v.max_blocks;  // index head g = KV head g
            for (unsigned b = tid; b < rest; b += NT) sel.keys[b] = order_key(sc[b]);
            __syncthreads();
            choose_topk<NT>(sel.keys, rest, kTopK - 1, sel.chosen, sel.hist, sel.wave_tot, sel.sh);
            const bool out = c == 0 && part == 0;
            unsigned before = 0;  // ascending compaction (block scans), slot c's id is this block's chunk
            for (std::uint32_t base = 0; base < rest; base += NT) {
                const std::uint32_t i = base + tid;
                const bool ch = i < rest && sel.chosen[i] != 0U;
                unsigned total = 0;
                const unsigned rank = before + block_scan<NT>(ch, sel.wave_tot, total);
                if (ch) {
                    if (rank - 1 == c) sel.sh[2] = i;
                    if (out) v.idx[static_cast<std::size_t>(rg) * kTopK + rank - 1] = i;
                }
                before += total;
            }
            if (out && tid == 0) v.idx[static_cast<std::size_t>(rg) * kTopK + kTopK - 1] = rest;
            __syncthreads();
            blk = sel.sh[2];
        }
    } else {
        cnt = v.idx_count[rg];
        if (c < cnt) blk = v.idx[static_cast<std::size_t>(rg) * kTopK + c];
    }
    if (!DENSE && select != 0 && c == 0 && part == 0 && tid == 0) v.idx_count[rg] = cnt;
    if (c >= cnt) return;
    stamp(v.phase, 1);
    const std::uint64_t lo = static_cast<std::uint64_t>(blk) * kBlock;
    const unsigned n_keys = static_cast<unsigned>((count < lo + kBlock ? count : lo + kBlock) - lo);
    std::uint16_t* kc = v.k_cache[s];
    std::uint16_t* vc = v.v_cache[s];
    constexpr unsigned PER = kChunks16 / NT;
    {
        uint4 kreg[PER];
#pragma unroll
        for (unsigned i = 0; i < PER; ++i) {
            const unsigned e = i * NT + tid, key = e >> 4U, piece = e & 15U;
            kreg[i] = key < n_keys ? *reinterpret_cast<const uint4*>(kc + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                   : make_uint4(0U, 0U, 0U, 0U);
        }
        norm_rope_lane(q, v.q_w, v.rope[r], lane);
        __syncthreads();  // selection scratch is dead from here: the tile reuses it
#pragma unroll
        for (unsigned i = 0; i < PER; ++i) tile_mem[i * NT + tid] = kreg[i];  // row-major
    }
    uint2* t2 = reinterpret_cast<uint2*>(tile_mem);  // key t, lane l: t2[t * 32 + l]
    if constexpr (DENSE && !prefill_kernel<View>) {
        __syncthreads();
        for (unsigned rr = wave; rr < W; rr += HW) {
            const unsigned rw = s * W + rr;
            const std::uint64_t p = v.pos[rw];
            if (p < lo || p >= lo + n_keys) continue;
            float x[kDimsPerLane];
            load4(x, v.k + (static_cast<std::size_t>(rw) * v.kv_heads + g) * kHeadDim, lane);
            norm_rope_lane(x, v.k_w, v.rope[rw], lane);
            const uint2 packed = pack4(x);
            *reinterpret_cast<uint2*>(kc + (p * v.kv_heads + g) * kHeadDim + lane * kDimsPerLane) = packed;
            t2[static_cast<unsigned>(p - lo) * 32 + lane] = packed;
        }
        if (c == 0 && g == 0 && part == 0 && wave == 0) {
            float x[kDimsPerLane];
            load4(x, v.ik + static_cast<std::size_t>(r) * kIdxDim, lane);
            norm_rope_lane(x, v.ik_w, v.rope[r], lane);
            *reinterpret_cast<uint2*>(v.ik_cache[s] + v.pos[r] * kIdxDim + lane * kDimsPerLane) = pack4(x);
        }
    }
    uint4 vreg[PER];
#pragma unroll
    for (unsigned i = 0; i < PER; ++i) {
        const unsigned e = i * NT + tid, key = e >> 4U, piece = e & 15U;
        vreg[i] = key < n_keys ? *reinterpret_cast<const uint4*>(vc + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                               : make_uint4(0U, 0U, 0U, 0U);
    }
    __syncthreads();
    stamp(v.phase, 2);
    float sv[kBlock / 8];  // group grp: score of key 8 * grp + (lane's key of the group)
    const unsigned my_k = ((lane >> 4U) & 1U) * 4U + ((lane >> 3U) & 1U) * 2U + ((lane >> 2U) & 1U);
    if constexpr (ROLL) {
#pragma unroll 1
        for (unsigned grp = 0; grp < kBlock / 8; ++grp) {
            const float sc = group_scores(t2, grp, q, lane);
#pragma unroll
            for (unsigned i = 0; i + 1 < kBlock / 8; ++i) sv[i] = sv[i + 1];
            sv[kBlock / 8 - 1] = grp * 8 + my_k < n_keys ? sc : -__builtin_inff();  // after 16 shifts: sv[i] = group i
        }
    } else {
#pragma unroll
        for (unsigned grp = 0; grp < kBlock / 8; ++grp) {
            const float sc = group_scores(t2, grp, q, lane);
            sv[grp] = grp * 8 + my_k < n_keys ? sc : -__builtin_inff();
        }
    }
    __syncthreads();
    stamp(v.phase, 3);
#pragma unroll
    for (unsigned i = 0; i < PER; ++i) tile_mem[i * NT + tid] = vreg[i];
    if constexpr (DENSE && !prefill_kernel<View>) {
        __syncthreads();
        for (unsigned rr = wave; rr < W; rr += HW) {
            const unsigned rw = s * W + rr;
            const std::uint64_t p = v.pos[rw];
            if (p < lo || p >= lo + n_keys) continue;
            float x[kDimsPerLane];
            load4(x, v.v + (static_cast<std::size_t>(rw) * v.kv_heads + g) * kHeadDim, lane);
            const uint2 packed = pack4(x);
            *reinterpret_cast<uint2*>(vc + (p * v.kv_heads + g) * kHeadDim + lane * kDimsPerLane) = packed;
            t2[static_cast<unsigned>(p - lo) * 32 + lane] = packed;
        }
    }
    __syncthreads();
    stamp(v.phase, 4);
    float acc[kDimsPerLane] = {0.0F, 0.0F, 0.0F, 0.0F};
    float m = -__builtin_inff(), l = 0.0F;
    // One tile of 16 keys from its two groups' scores (sa: keys t0..t0+7, sb: t0+8..t0+15).
    // NT = visible keys of the tile as a compile-time constant (16 = a full tile: no per-key predicate, so the 16 LDS
    // reads issue up front instead of one branch + LDS wait per key) or 0 (runtime nt, the partial last tile).
    auto tile_n = [&](auto ntc, unsigned t, float sa, float sb) {
        constexpr unsigned NTC = decltype(ntc)::value;
        const unsigned t0 = t * kTile;
        const unsigned nt = NTC != 0 ? NTC : (n_keys - t0 < kTile ? n_keys - t0 : kTile);
        float tm = -__builtin_inff();
#pragma unroll
        for (unsigned j = 0; j < kTile; ++j)
            if (NTC != 0 || j < nt) tm = max_of(tm, rl(j < 8 ? sa : sb, key_lane(j % 8)));
        const float mn = max_of(m, tm);
        const float corr = t == 0 ? 0.0F : exp_f(m - mn);
        l = l * corr;
        for (unsigned i = 0; i < kDimsPerLane; ++i) acc[i] = acc[i] * corr;
        const float p0 = exp_f(sa - mn), p1 = exp_f(sb - mn);
        uint2 vrow[kTile];  // full tile: all 16 V rows read before the chain (the scheduler otherwise sinks each read
                            // to its use and waits for LDS once per key)
        if constexpr (NTC != 0) {
#pragma unroll
            for (unsigned j = 0; j < kTile; ++j) vrow[j] = t2[(t0 + j) * 32 + lane];
            __builtin_amdgcn_sched_barrier(0);
        }
#pragma unroll
        for (unsigned j = 0; j < kTile; ++j) {
            if (NTC != 0 || j < nt) {
                const float pj = rl(j < 8 ? p0 : p1, key_lane(j % 8));
                l = l + pj;
                const uint2 raw = NTC != 0 ? vrow[j] : t2[(t0 + j) * 32 + lane];
                acc[0] = __builtin_fmaf(pj, lo_bf16(raw.x), acc[0]);
                acc[1] = __builtin_fmaf(pj, hi_bf16(raw.x), acc[1]);
                acc[2] = __builtin_fmaf(pj, lo_bf16(raw.y), acc[2]);
                acc[3] = __builtin_fmaf(pj, hi_bf16(raw.y), acc[3]);
            }
        }
        m = mn;
    };
    auto tile = [&](unsigned t, float sa, float sb) {
        if (v.pv_full_tiles != 0 && t * kTile + kTile <= n_keys) tile_n(std::integral_constant<unsigned, kTile>{}, t, sa, sb);
        else tile_n(std::integral_constant<unsigned, 0>{}, t, sa, sb);
    };
    if constexpr (ROLL) {
#pragma unroll 1
        for (unsigned t = 0; t < kBlock / kTile; ++t) {  // tiles of 16 keys = groups 2t, 2t + 1
            if (t * kTile < n_keys) tile(t, sv[0], sv[1]);  // uniform; tiles past the visible keys do nothing
#pragma unroll
            for (unsigned i = 0; i + 2 < kBlock / 8; ++i) sv[i] = sv[i + 2];
        }
    } else {
#pragma unroll
        for (unsigned t = 0; t < kBlock / kTile; ++t)
            if (t * kTile < n_keys) tile(t, sv[2 * t], sv[2 * t + 1]);
    }
    float* out = v.partial + ((static_cast<std::size_t>(r) * q_heads + qh) * kMaxChunks + c) * kPartial;
    if (lane == 0) {
        out[0] = m;
        out[1] = l;
    }
    for (unsigned i = 0; i < kDimsPerLane; ++i) out[2 + lane * kDimsPerLane + i] = acc[i];
    stamp(v.phase, 5);
    __threadfence();
    __syncthreads();
    std::uint32_t* ticket = v.ticket + rgp;
    // Every wave is past its PV reads (barrier above), so the tile's first word carries thread 0's flag (no static
    // LDS: __syncthreads_or would add 256 B and drop a WGP to three blocks).
    std::uint32_t* flag = reinterpret_cast<std::uint32_t*>(tile_mem);
    if (tid == 0) {
        const unsigned prev = atomicAdd(ticket, 1U);
        *flag = prev + 1 == cnt ? 1U : 0U;
        if (prev + 1 == cnt) atomicExch(ticket, 0U);
    }
    __syncthreads();
    stamp(v.phase, 6);
    if (*flag == 0U) return;
    __threadfence();
    // Combine: every chunk's (m, l) and this lane's 4 dims of acc loaded up front (independent loads), then the
    // declared chunk-order chains.
    const float* pt = v.partial + (static_cast<std::size_t>(r) * q_heads + qh) * kMaxChunks * kPartial;
    const float mc = lane < cnt ? pt[lane * kPartial] : -__builtin_inff();
    const float lc = lane < cnt ? pt[lane * kPartial + 1] : 0.0F;
    float ac[kMaxChunks][kDimsPerLane];
#pragma unroll
    for (unsigned cc = 0; cc < kMaxChunks; ++cc)
        for (unsigned i = 0; i < kDimsPerLane; ++i) ac[cc][i] = cc < cnt ? pt[cc * kPartial + 2 + lane * kDimsPerLane + i] : 0.0F;
    float big = -__builtin_inff();
#pragma unroll
    for (unsigned cc = 0; cc < kMaxChunks; ++cc)
        if (cc < cnt) big = max_of(big, rl(mc, static_cast<int>(cc)));
    const float ec_own = exp_f(mc - big);  // lane cc: e of chunk cc
    float L = 0.0F, e[kMaxChunks];
#pragma unroll
    for (unsigned cc = 0; cc < kMaxChunks; ++cc)
        if (cc < cnt) {
            e[cc] = rl(ec_own, static_cast<int>(cc));
            L = __builtin_fmaf(rl(lc, static_cast<int>(cc)), e[cc], L);
        }
    float* o = v.o + (static_cast<std::size_t>(r) * q_heads + qh) * kHeadDim;
    for (unsigned i = 0; i < kDimsPerLane; ++i) {
        float a = 0.0F;
#pragma unroll
        for (unsigned cc = 0; cc < kMaxChunks; ++cc)
            if (cc < cnt) a = __builtin_fmaf(ac[cc][i], e[cc], a);
        o[lane * kDimsPerLane + i] = a / L;
    }
    stamp(v.phase, 7);
}

// ---------------------------------------------------------------- attend v4: HPW heads per wave
//
// Version 3's arithmetic, but a wave runs HPW query heads of the KV group: every K row (scores) and V row (PV) is
// read from LDS once per wave and used for all its heads, so a block's LDS traffic drops by HPW (prefill attention
// was about half LDS-bound). Each head's operations and their order are unchanged (heads never mix), so the bits are
// those of versions 1-3. Block = all 16 heads of (row, KV head, chunk): 16 / HPW waves.

// Scores of keys 8 * grp .. 8 * grp + 7 for HPW heads: one LDS read per key, HPW lane chains, HPW transposed
// butterflies (group_scores per head).
template <unsigned HPW>
__device__ __forceinline__ void group_scores_h(const uint2* tile, unsigned grp, const float (&q)[HPW][kDimsPerLane],
                                               unsigned lane, float (&out)[HPW]) {
    float vals[HPW][8];
#pragma unroll
    for (unsigned k = 0; k < 8; ++k) {
        const uint2 raw = tile[(grp * 8 + k) * 32 + lane];
        const float k0 = lo_bf16(raw.x), k1 = hi_bf16(raw.x), k2 = lo_bf16(raw.y), k3 = hi_bf16(raw.y);
#pragma unroll
        for (unsigned h = 0; h < HPW; ++h) {
            float a = 0.0F;
            a = __builtin_fmaf(q[h][0], k0, a);
            a = __builtin_fmaf(q[h][1], k1, a);
            a = __builtin_fmaf(q[h][2], k2, a);
            a = __builtin_fmaf(q[h][3], k3, a);
            vals[h][k] = a;
        }
    }
    const bool h16 = (lane & 16U) != 0U, h8 = (lane & 8U) != 0U, h4 = (lane & 4U) != 0U;
#pragma unroll
    for (unsigned h = 0; h < HPW; ++h) {
        float a4[4];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const float own = h16 ? vals[h][i + 4] : vals[h][i], send = h16 ? vals[h][i] : vals[h][i + 4];
            a4[i] = own + swz_xor(send, 16);
        }
        float a2[2];
#pragma unroll
        for (unsigned i = 0; i < 2; ++i) {
            const float own = h8 ? a4[i + 2] : a4[i], send = h8 ? a4[i] : a4[i + 2];
            a2[i] = own + swz_xor(send, 8);
        }
        float c = (h4 ? a2[1] : a2[0]) + swz_xor(h4 ? a2[0] : a2[1], 4);
        c = c + swz_xor(c, 2);
        c = c + swz_xor(c, 1);
        out[h] = c * kScale;
    }
}

// Number of valid 128-key partials for positions [0, end).
__host__ __device__ std::uint64_t prefill_chunk_prefix(std::uint64_t end) {
    const std::uint64_t limit = kTopK * kBlock;
    const std::uint64_t dense = end < limit ? end : limit;
    const std::uint64_t blocks = dense / kBlock, tail = dense % kBlock;
    return kBlock * blocks * (blocks + 1) / 2 + tail * (blocks + 1) + (end - dense) * kTopK;
}

// One selection per (row, KV group), with IDs emitted in ascending block order.
__global__ __launch_bounds__(256) void prefill_select_kernel(PrefillView p) {
    const DecodeView v = kernel_view(p, blockIdx.z);
    const unsigned rg = blockIdx.x, r = rg / v.kv_heads, tid = threadIdx.x;
    if (r >= v.rows_per_seq) return;
    const unsigned nb = static_cast<unsigned>(block_count(v.pos[r] + 1));
    const unsigned cnt = nb < kTopK ? nb : kTopK;
    if (nb <= kTopK) {
        if (tid < cnt) v.idx[static_cast<std::size_t>(rg) * kTopK + tid] = tid;
    } else {
        extern __shared__ uint4 selection_mem[];
        SelView<256> sel(selection_mem, v.max_blocks);
        const unsigned rest = nb - 1;
        const float* sc = v.scores + static_cast<std::size_t>(rg) * v.max_blocks;
        for (unsigned b = tid; b < rest; b += 256) sel.keys[b] = order_key(sc[b]);
        __syncthreads();
        choose_topk<256>(sel.keys, rest, kTopK - 1, sel.chosen, sel.hist, sel.wave_tot, sel.sh);
        unsigned before = 0;
        for (unsigned base = 0; base < rest; base += 256) {
            const unsigned i = base + tid;
            const bool chosen = i < rest && sel.chosen[i] != 0U;
            unsigned total = 0;
            const unsigned rank = before + block_scan<256>(chosen, sel.wave_tot, total);
            if (chosen) v.idx[static_cast<std::size_t>(rg) * kTopK + rank - 1] = i;
            before += total;
        }
        if (tid == 0) v.idx[static_cast<std::size_t>(rg) * kTopK + kTopK - 1] = rest;
    }
    if (tid == 0) v.idx_count[rg] = cnt;
}

// One wave per query head. Expressions and ascending chunk order match attend4.
template <bool DENSE>
__global__ __launch_bounds__(256) void prefill_combine_kernel(PrefillView p) {
    const DecodeView v = kernel_view(p, blockIdx.z);
    const unsigned r = blockIdx.x, lane = threadIdx.x & 31U;
    if (r >= v.rows_per_seq) return;
    const unsigned q_heads = v.q_heads, qh = blockIdx.y * 8 + (threadIdx.x >> 5U);
    const unsigned g = qh / kGroup;
    const unsigned cnt = DENSE ? static_cast<unsigned>(block_count(v.pos[r] + 1)) : v.idx_count[r * v.kv_heads + g];
    const float* pt = v.partial + (static_cast<std::size_t>(r) * q_heads + qh) * kMaxChunks * kPartial;
    const float mc = lane < cnt ? pt[lane * kPartial] : -__builtin_inff();
    const float lc = lane < cnt ? pt[lane * kPartial + 1] : 0.0F;
    float ac[kMaxChunks][kDimsPerLane];
#pragma unroll
    for (unsigned cc = 0; cc < kMaxChunks; ++cc)
        for (unsigned i = 0; i < kDimsPerLane; ++i) ac[cc][i] = cc < cnt ? pt[cc * kPartial + 2 + lane * kDimsPerLane + i] : 0.0F;
    float big = -__builtin_inff();
#pragma unroll
    for (unsigned cc = 0; cc < kMaxChunks; ++cc)
        if (cc < cnt) big = max_of(big, rl(mc, static_cast<int>(cc)));
    const float ec_own = exp_f(mc - big);
    float L = 0.0F, e[kMaxChunks];
#pragma unroll
    for (unsigned cc = 0; cc < kMaxChunks; ++cc)
        if (cc < cnt) {
            e[cc] = rl(ec_own, static_cast<int>(cc));
            L = __builtin_fmaf(rl(lc, static_cast<int>(cc)), e[cc], L);
        }
    float* o = v.o + (static_cast<std::size_t>(r) * q_heads + qh) * kHeadDim;
    for (unsigned i = 0; i < kDimsPerLane; ++i) {
        float a = 0.0F;
#pragma unroll
        for (unsigned cc = 0; cc < kMaxChunks; ++cc)
            if (cc < cnt) a = __builtin_fmaf(ac[cc][i], e[cc], a);
        o[lane * kDimsPerLane + i] = a / L;
    }
}

template <unsigned HPW, bool DENSE, class View = DecodeView, bool SEPARATE = false, bool STREAM = false, bool COMPACT = SEPARATE, bool PACKED = false,
          bool CORRPACK = false, bool TAIL = !SEPARATE>
__global__ __launch_bounds__(kGroup / HPW * 32) void attend4_kernel(View input, int select) {
    constexpr bool REPAIR = repaired_prefill<View>;
    static_assert(!PACKED || (HPW == 2 && REPAIR && SEPARATE && STREAM && COMPACT));
    static_assert(!CORRPACK || PACKED);
    unsigned window = blockIdx.z, row_group = blockIdx.x, chunk = blockIdx.y;
    if constexpr (COMPACT) {
        // Invert the prefix to launch exactly the valid (row, group, chunk) items.
        const unsigned first_row = input.first_window * kMaxRows;
        if constexpr (REPAIR && !DENSE) {
            const unsigned item = blockIdx.x / input.window.kv_heads;
            const unsigned row = first_row + item / 16;
            chunk = item % 16;
            window = row / kMaxRows - input.first_window;
            row_group = (row % kMaxRows) * input.window.kv_heads + blockIdx.x % input.window.kv_heads;
        } else {
            const std::uint64_t base = prefill_chunk_prefix(input.pos0 + first_row);
            const std::uint64_t item = blockIdx.x / input.window.kv_heads;
            unsigned low = first_row, high = input.rows;
            while (low + 1 < high) {
                const unsigned mid = low + (high - low) / 2;
                if (prefill_chunk_prefix(input.pos0 + mid) - base <= item) low = mid;
                else high = mid;
            }
            window = low / kMaxRows - input.first_window;
            row_group = (low % kMaxRows) * input.window.kv_heads + blockIdx.x % input.window.kv_heads;
            chunk = static_cast<unsigned>(item - (prefill_chunk_prefix(input.pos0 + low) - base));
        }
    }
    const DecodeView v = kernel_view(input, window);
    if constexpr (prefill_kernel<View>) {
        if (row_group >= v.rows_per_seq * v.kv_heads) return;
    }
    constexpr unsigned NW = kGroup / HPW, NT = NW * 32;
    extern __shared__ uint4 dyn_smem[];
    SelView<NT> sel(dyn_smem, v.max_blocks);
    uint4* const tile_mem = dyn_smem;
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const unsigned rg = row_group, r = rg / v.kv_heads, g = rg % v.kv_heads;
    const unsigned c = chunk, W = v.rows_per_seq, s = r / W, q_heads = v.kv_heads * kGroup;
    const unsigned qh0 = g * kGroup + wave * HPW;  // this wave's heads qh0 .. qh0 + HPW - 1
    const std::uint64_t count = v.pos[r] + 1;
    stamp(v.phase, 0);
    float q[HPW][kDimsPerLane];
#pragma unroll
    for (unsigned h = 0; h < HPW; ++h) load4(q[h], v.q + (static_cast<std::size_t>(r) * q_heads + qh0 + h) * kHeadDim, lane);
    std::uint32_t cnt = 0, blk = 0;
    const std::uint32_t nb = static_cast<std::uint32_t>(block_count(count));
    if (DENSE || nb <= kTopK) {
        cnt = nb;
        blk = c;
        if (!DENSE && !SEPARATE && select != 0 && c == 0 && tid < nb) v.idx[static_cast<std::size_t>(rg) * kTopK + tid] = tid;
    } else if (!SEPARATE && select != 0) {
        cnt = kTopK;
        const std::uint32_t rest = nb - 1;
        if (c == kTopK - 1) {
            blk = rest;
        } else {
            const float* sc = v.scores + static_cast<std::size_t>(rg) * v.max_blocks;
            for (unsigned b = tid; b < rest; b += NT) sel.keys[b] = order_key(sc[b]);
            __syncthreads();
            choose_topk<NT>(sel.keys, rest, kTopK - 1, sel.chosen, sel.hist, sel.wave_tot, sel.sh);
            const bool out = c == 0;
            unsigned before = 0;
            for (std::uint32_t base = 0; base < rest; base += NT) {
                const std::uint32_t i = base + tid;
                const bool ch = i < rest && sel.chosen[i] != 0U;
                unsigned total = 0;
                const unsigned rank = before + block_scan<NT>(ch, sel.wave_tot, total);
                if (ch) {
                    if (rank - 1 == c) sel.sh[2] = i;
                    if (out) v.idx[static_cast<std::size_t>(rg) * kTopK + rank - 1] = i;
                }
                before += total;
            }
            if (out && tid == 0) v.idx[static_cast<std::size_t>(rg) * kTopK + kTopK - 1] = rest;
            __syncthreads();
            blk = sel.sh[2];
        }
    } else {
        cnt = v.idx_count[rg];
        if (c < cnt) blk = v.idx[static_cast<std::size_t>(rg) * kTopK + c];
    }
    if (!DENSE && !SEPARATE && select != 0 && c == 0 && tid == 0) v.idx_count[rg] = cnt;
    if (c >= cnt) return;
    stamp(v.phase, 1);
    const std::uint64_t lo = static_cast<std::uint64_t>(blk) * kBlock;
    const unsigned n_keys = static_cast<unsigned>((count < lo + kBlock ? count : lo + kBlock) - lo);
    std::uint16_t* kc = v.k_cache[s];
    std::uint16_t* vc = v.v_cache[s];
    static_assert(!STREAM || (SEPARATE && prefill_kernel<View> && HPW == 2));
    uint2* t2 = reinterpret_cast<uint2*>(tile_mem);
    float sv[HPW][STREAM ? 2 : kBlock / 8];
    if constexpr (!STREAM) {
        constexpr unsigned PER = kChunks16 / NT;
        {
            uint4 kreg[PER];
#pragma unroll
            for (unsigned i = 0; i < PER; ++i) {
                const unsigned e = i * NT + tid, key = e >> 4U, piece = e & 15U;
                kreg[i] = key < n_keys ? *reinterpret_cast<const uint4*>(kc + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                       : make_uint4(0U, 0U, 0U, 0U);
            }
#pragma unroll
            for (unsigned h = 0; h < HPW; ++h) norm_rope_lane(q[h], v.q_w, v.rope[r], lane);
            __syncthreads();
#pragma unroll
            for (unsigned i = 0; i < PER; ++i) tile_mem[i * NT + tid] = kreg[i];
        }
        if constexpr (DENSE && !prefill_kernel<View>) {
            __syncthreads();
            for (unsigned rr = wave; rr < W; rr += NW) {
                const unsigned rw = s * W + rr;
                const std::uint64_t p = v.pos[rw];
                if (p < lo || p >= lo + n_keys) continue;
                float x[kDimsPerLane];
                load4(x, v.k + (static_cast<std::size_t>(rw) * v.kv_heads + g) * kHeadDim, lane);
                norm_rope_lane(x, v.k_w, v.rope[rw], lane);
                const uint2 packed = pack4(x);
                *reinterpret_cast<uint2*>(kc + (p * v.kv_heads + g) * kHeadDim + lane * kDimsPerLane) = packed;
                t2[static_cast<unsigned>(p - lo) * 32 + lane] = packed;
            }
            if (c == 0 && g == 0 && wave == 0) {
                float x[kDimsPerLane];
                load4(x, v.ik + static_cast<std::size_t>(r) * kIdxDim, lane);
                norm_rope_lane(x, v.ik_w, v.rope[r], lane);
                *reinterpret_cast<uint2*>(v.ik_cache[s] + v.pos[r] * kIdxDim + lane * kDimsPerLane) = pack4(x);
            }
        }
        uint4 vreg[PER];
#pragma unroll
        for (unsigned i = 0; i < PER; ++i) {
            const unsigned e = i * NT + tid, key = e >> 4U, piece = e & 15U;
            vreg[i] = key < n_keys ? *reinterpret_cast<const uint4*>(vc + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                   : make_uint4(0U, 0U, 0U, 0U);
        }
        __syncthreads();
        stamp(v.phase, 2);
        const unsigned my_k = ((lane >> 4U) & 1U) * 4U + ((lane >> 3U) & 1U) * 2U + ((lane >> 2U) & 1U);
#pragma unroll 1
        for (unsigned grp = 0; grp < kBlock / 8; ++grp) {
            float sc[HPW];
            group_scores_h<HPW>(t2, grp, q, lane, sc);
            const bool vis = grp * 8 + my_k < n_keys;
#pragma unroll
            for (unsigned h = 0; h < HPW; ++h) {
#pragma unroll
                for (unsigned i = 0; i + 1 < kBlock / 8; ++i) sv[h][i] = sv[h][i + 1];
                sv[h][kBlock / 8 - 1] = vis ? sc[h] : -__builtin_inff();
            }
        }
        __syncthreads();
        stamp(v.phase, 3);
#pragma unroll
        for (unsigned i = 0; i < PER; ++i) tile_mem[i * NT + tid] = vreg[i];
        if constexpr (DENSE && !prefill_kernel<View>) {
            __syncthreads();
            for (unsigned rr = wave; rr < W; rr += NW) {
                const unsigned rw = s * W + rr;
                const std::uint64_t p = v.pos[rw];
                if (p < lo || p >= lo + n_keys) continue;
                float x[kDimsPerLane];
                load4(x, v.v + (static_cast<std::size_t>(rw) * v.kv_heads + g) * kHeadDim, lane);
                const uint2 packed = pack4(x);
                *reinterpret_cast<uint2*>(vc + (p * v.kv_heads + g) * kHeadDim + lane * kDimsPerLane) = packed;
                t2[static_cast<unsigned>(p - lo) * 32 + lane] = packed;
            }
        }
        __syncthreads();
        stamp(v.phase, 4);
    } else {
#pragma unroll
        for (unsigned h = 0; h < HPW; ++h) norm_rope_lane(q[h], v.q_w, v.rope[r], lane);
    }
    if constexpr (STREAM) {
        // Two 8 KiB K/V slots; the next tile is prefetched into registers.
        const unsigned key = tid / 16, piece = tid % 16;
        tile_mem[tid] = key < n_keys ? *reinterpret_cast<const uint4*>(kc + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                     : make_uint4(0U, 0U, 0U, 0U);
        tile_mem[NT + tid] = key < n_keys ? *reinterpret_cast<const uint4*>(vc + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                          : make_uint4(0U, 0U, 0U, 0U);
    }
    float acc[HPW][kDimsPerLane], m[HPW], l[HPW];
#pragma unroll
    for (unsigned h = 0; h < HPW; ++h) {
        for (unsigned i = 0; i < kDimsPerLane; ++i) acc[h][i] = 0.0F;
        m[h] = -__builtin_inff();
        l[h] = 0.0F;
    }
#pragma unroll 1
    for (unsigned t = 0; t < (REPAIR ? (n_keys + kTile - 1) / kTile : kBlock / kTile); ++t) {  // tiles of 16 keys = groups 2t, 2t + 1 (now in sv[.][0..1])
        uint4 next_k{}, next_v{};
        uint4* const current = tile_mem + (t & 1U) * (2 * NT);
        if constexpr (STREAM) {
            static_assert(kTile * (kHeadDim / 8) == NT);
            __syncthreads();  // Publish the tile staged by the previous iteration.
            const unsigned key = (t + 1) * kTile + tid / 16, piece = tid % 16;
            if (t + 1 < (REPAIR ? (n_keys + kTile - 1) / kTile : kBlock / kTile)) {
                next_k = key < n_keys ? *reinterpret_cast<const uint4*>(kc + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                      : make_uint4(0U, 0U, 0U, 0U);
                next_v = key < n_keys ? *reinterpret_cast<const uint4*>(vc + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                      : make_uint4(0U, 0U, 0U, 0U);
            }
#pragma unroll
            for (unsigned grp = 0; grp < 2; ++grp) {
                float sc[HPW];
                group_scores_h<HPW>(reinterpret_cast<const uint2*>(current), grp, q, lane, sc);
                const unsigned my_key = ((lane >> 4U) & 1U) * 4U + ((lane >> 3U) & 1U) * 2U + ((lane >> 2U) & 1U);
#pragma unroll
                for (unsigned h = 0; h < HPW; ++h)
                    sv[h][grp] = t * kTile + grp * 8 + my_key < n_keys ? sc[h] : -__builtin_inff();
            }
            t2 = reinterpret_cast<uint2*>(current + NT);
        }
        auto tile4 = [&](auto ntc) {
            constexpr unsigned NTC = decltype(ntc)::value;  // 16: full tile, no per-key predicate; 0: runtime nt
            const unsigned t0 = t * kTile;
            const unsigned nt = NTC != 0 ? NTC : (n_keys - t0 < kTile ? n_keys - t0 : kTile);
            float p0[HPW], p1[HPW], mprev[HPW];
#pragma unroll
            for (unsigned h = 0; h < HPW; ++h) {
                const float sa = sv[h][0], sb = sv[h][1];
                float tm = -__builtin_inff();
#pragma unroll
                for (unsigned j = 0; j < kTile; ++j)
                    if (NTC != 0 || j < nt) tm = max_of(tm, rl(j < 8 ? sa : sb, key_lane(j % 8)));
                const float mn = max_of(m[h], tm);
                mprev[h] = m[h];
                if constexpr (!CORRPACK) {
                    const float corr = t == 0 ? 0.0F : exp_f(m[h] - mn);
                    l[h] = l[h] * corr;
                    for (unsigned i = 0; i < kDimsPerLane; ++i) acc[h][i] = acc[h][i] * corr;
                }
                if constexpr (!PACKED) {
                    p0[h] = exp_f(sa - mn);
                    p1[h] = exp_f(sb - mn);
                }
                m[h] = mn;
            }
            if constexpr (PACKED) {
                // Preserve canonical score bits, including NaN payloads:
                // bit-mode AND 0x1c broadcasts lane (lane & ~3U) in each quad.
                // Head 0 occupies even lanes, head 1 odd lanes, in both halves.
                const float mn = (lane & 1U) != 0U ? m[1] : m[0];
#pragma unroll
                for (unsigned half = 0; half < 2; ++half) {
                    const float s0 = u2f(static_cast<std::uint32_t>(__builtin_amdgcn_ds_swizzle(
                        static_cast<int>(f2u(sv[0][half])), 0x1c)));
                    const float s1 = u2f(static_cast<std::uint32_t>(__builtin_amdgcn_ds_swizzle(
                        static_cast<int>(f2u(sv[1][half])), 0x1c)));
                    const float score = (lane & 1U) != 0U ? s1 : s0;
                    // CORRPACK (mode 6): quad offsets 2/3 of the first vector carry head 0/1's correction argument
                    // m_old - m_new instead of a duplicate probability, so the same exp_f statements give the corrections
                    // (applied below, still before any PV accumulation of this tile).
                    const float mprev_lane = (lane & 1U) != 0U ? mprev[1] : mprev[0];
                    const float arg = CORRPACK && half == 0 && (lane & 2U) != 0U ? mprev_lane - mn : score - mn;
                    const float probability = exp_f(arg);
                    if (half == 0) p0[0] = probability;
                    else p1[0] = probability;
                }
                if constexpr (CORRPACK) {
#pragma unroll
                    for (unsigned h = 0; h < HPW; ++h) {
                        const float corr = t == 0 ? 0.0F : rl(p0[0], 2 + static_cast<int>(h));
                        l[h] = l[h] * corr;
                        for (unsigned i = 0; i < kDimsPerLane; ++i) acc[h][i] = acc[h][i] * corr;
                    }
                }
            }
            uint2 vrow[kTile];  // full tile: the 16 V rows read before the chain (see version 3)
            if constexpr (NTC != 0) {
#pragma unroll
                for (unsigned j = 0; j < kTile; ++j) vrow[j] = t2[((STREAM ? 0U : t0) + j) * 32 + lane];
                __builtin_amdgcn_sched_barrier(0);
            }
#pragma unroll
            for (unsigned j = 0; j < kTile; ++j) {
                if (NTC != 0 || j < nt) {
                    const uint2 raw = NTC != 0 ? vrow[j] : t2[((STREAM ? 0U : t0) + j) * 32 + lane];
                    const float v0 = lo_bf16(raw.x), v1 = hi_bf16(raw.x), v2 = lo_bf16(raw.y), v3 = hi_bf16(raw.y);
#pragma unroll
                    for (unsigned h = 0; h < HPW; ++h) {
                        float pj;
                        if constexpr (PACKED)
                            pj = rl(j < 8 ? p0[0] : p1[0], key_lane(j % 8) + static_cast<int>(h));
                        else
                            pj = rl(j < 8 ? p0[h] : p1[h], key_lane(j % 8));
                        l[h] = l[h] + pj;
                        acc[h][0] = __builtin_fmaf(pj, v0, acc[h][0]);
                        acc[h][1] = __builtin_fmaf(pj, v1, acc[h][1]);
                        acc[h][2] = __builtin_fmaf(pj, v2, acc[h][2]);
                        acc[h][3] = __builtin_fmaf(pj, v3, acc[h][3]);
                    }
                }
            }
        };
        if (v.pv_full_tiles != 0 && t * kTile + kTile <= n_keys) tile4(std::integral_constant<unsigned, kTile>{});
        else if (t * kTile < n_keys) tile4(std::integral_constant<unsigned, 0>{});
        if constexpr (STREAM) {
            if (t + 1 < (REPAIR ? (n_keys + kTile - 1) / kTile : kBlock / kTile)) {
                uint4* const next = tile_mem + ((t + 1) & 1U) * (2 * NT);
                next[tid] = next_k;
                next[NT + tid] = next_v;
            }
        } else {
#pragma unroll
            for (unsigned h = 0; h < HPW; ++h)
#pragma unroll
                for (unsigned i = 0; i + 2 < kBlock / 8; ++i) sv[h][i] = sv[h][i + 2];
        }
    }
#pragma unroll
    for (unsigned h = 0; h < HPW; ++h) {
        float* out = v.partial + ((static_cast<std::size_t>(r) * q_heads + qh0 + h) * kMaxChunks + c) * kPartial;
        if (lane == 0) {
            out[0] = m[h];
            out[1] = l[h];
        }
        for (unsigned i = 0; i < kDimsPerLane; ++i) out[2 + lane * kDimsPerLane + i] = acc[h][i];
    }
    stamp(v.phase, 5);
    if constexpr (TAIL) {  // the last chunk block of (row, group) combines; mode 7 keeps this on the SEPARATE producer
        __threadfence();
        __syncthreads();
        std::uint32_t* ticket = v.ticket + rg;
        std::uint32_t* flag = reinterpret_cast<std::uint32_t*>(tile_mem);
        if (tid == 0) {
            const unsigned prev = atomicAdd(ticket, 1U);
            *flag = prev + 1 == cnt ? 1U : 0U;
            if (prev + 1 == cnt) atomicExch(ticket, 0U);
        }
        __syncthreads();
        stamp(v.phase, 6);
        if (*flag == 0U) return;
        __threadfence();
#pragma unroll 1
        for (unsigned h = 0; h < HPW; ++h) {  // combine, version 3's code per head
            const unsigned qh = qh0 + h;
            const float* pt = v.partial + (static_cast<std::size_t>(r) * q_heads + qh) * kMaxChunks * kPartial;
            const float mc = lane < cnt ? pt[lane * kPartial] : -__builtin_inff();
            const float lc = lane < cnt ? pt[lane * kPartial + 1] : 0.0F;
            float ac[kMaxChunks][kDimsPerLane];
#pragma unroll
            for (unsigned cc = 0; cc < kMaxChunks; ++cc)
                for (unsigned i = 0; i < kDimsPerLane; ++i) ac[cc][i] = cc < cnt ? pt[cc * kPartial + 2 + lane * kDimsPerLane + i] : 0.0F;
            float big = -__builtin_inff();
#pragma unroll
            for (unsigned cc = 0; cc < kMaxChunks; ++cc)
                if (cc < cnt) big = max_of(big, rl(mc, static_cast<int>(cc)));
            const float ec_own = exp_f(mc - big);
            float L = 0.0F, e[kMaxChunks];
#pragma unroll
            for (unsigned cc = 0; cc < kMaxChunks; ++cc)
                if (cc < cnt) {
                    e[cc] = rl(ec_own, static_cast<int>(cc));
                    L = __builtin_fmaf(rl(lc, static_cast<int>(cc)), e[cc], L);
                }
            float* o = v.o + (static_cast<std::size_t>(r) * q_heads + qh) * kHeadDim;
            for (unsigned i = 0; i < kDimsPerLane; ++i) {
                float a = 0.0F;
#pragma unroll
                for (unsigned cc = 0; cc < kMaxChunks; ++cc)
                    if (cc < cnt) a = __builtin_fmaf(ac[cc][i], e[cc], a);
                o[lane * kDimsPerLane + i] = a / L;
            }
        }
        stamp(v.phase, 7);
    }
}

// R16 tile-major producer: four banks of eight waves, four scalarized row
// slots per bank, two heads per wave. Only integer membership is shared state;
// each row/head retains the original ascending 16-key floating-point recurrence.
template <bool DENSE>
__global__ __launch_bounds__(1024) void prefill_r16_kernel(PrefillView input) {
    static_assert(kMaxRows == 16 && kGroup == 16 && kTile == 16 && kHeadDim == 128);
    const DecodeView v = kernel_view(input, blockIdx.z);
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const unsigned bank = wave / 8, g = blockIdx.y, blk = blockIdx.x;
    const unsigned qh0 = g * kGroup + (wave % 8) * 2;
    const std::uint64_t lo = static_cast<std::uint64_t>(blk) * kBlock;
    if (lo >= v.pos[v.rows_per_seq - 1] + 1) return;  // beyond this window's candidate range
    __shared__ unsigned selected_row[16], ordinal[16], visible[16];
    __shared__ unsigned members, max_keys;
    extern __shared__ uint4 tile_mem[];
    constexpr unsigned PIECES = kTile * (kHeadDim / 8);  // 256 uint4 per K or V tile
    if (tid == 0) {
        members = 0;
        max_keys = 0;
        for (unsigned r = 0; r < v.rows_per_seq; ++r) {
            const std::uint64_t count = v.pos[r] + 1;
            if (count <= lo) continue;
            const unsigned nb = static_cast<unsigned>(block_count(count));
            unsigned c = blk;
            bool found = DENSE || nb <= kTopK;
            if (!found) {
                const unsigned rg = r * v.kv_heads + g;
                for (c = 0; c < v.idx_count[rg]; ++c) {
                    if (v.idx[static_cast<std::size_t>(rg) * kTopK + c] == blk) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) continue;
            const unsigned nk = static_cast<unsigned>(count < lo + kBlock ? count - lo : kBlock);
            selected_row[members] = r;
            ordinal[members] = c;
            visible[members] = nk;
            ++members;
            if (max_keys < nk) max_keys = nk;
        }
    }
    __syncthreads();
    if (members == 0) return;  // uniform, before Q or K/V loads
    float q[4][2][kDimsPerLane], acc[4][2][kDimsPerLane];
    float m[4][2], l[4][2];
#pragma unroll
    for (unsigned slot = 0; slot < 4; ++slot) {
        const unsigned member = bank * 4 + slot;
        if (member < members) {
            const unsigned r = selected_row[member];
#pragma unroll
            for (unsigned h = 0; h < 2; ++h) {
                load4(q[slot][h], v.q + (static_cast<std::size_t>(r) * v.q_heads + qh0 + h) * kHeadDim, lane);
                norm_rope_lane(q[slot][h], v.q_w, v.rope[r], lane);
                m[slot][h] = -__builtin_inff();
                l[slot][h] = 0.0F;
#pragma unroll
                for (unsigned i = 0; i < kDimsPerLane; ++i) acc[slot][h][i] = 0.0F;
            }
        }
    }
    // Only 256 threads load each K/V tile; all 1024 threads consume it.
    if (tid < PIECES) {
        const unsigned key = tid / 16, piece = tid % 16;
        tile_mem[tid] = key < max_keys ? *reinterpret_cast<const uint4*>(v.k_cache[0] + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                              : make_uint4(0U, 0U, 0U, 0U);
        tile_mem[PIECES + tid] = key < max_keys ? *reinterpret_cast<const uint4*>(v.v_cache[0] + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                                       : make_uint4(0U, 0U, 0U, 0U);
    }
    const unsigned tiles = (max_keys + kTile - 1) / kTile;
#pragma unroll 1
    for (unsigned t = 0; t < tiles; ++t) {
        __syncthreads();  // publish current tile; inactive/finished rows also participate
        uint4 next_k{}, next_v{};
        if (tid < PIECES && t + 1 < tiles) {
            const unsigned key = (t + 1) * kTile + tid / 16, piece = tid % 16;
            next_k = key < max_keys ? *reinterpret_cast<const uint4*>(v.k_cache[0] + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                   : make_uint4(0U, 0U, 0U, 0U);
            next_v = key < max_keys ? *reinterpret_cast<const uint4*>(v.v_cache[0] + ((lo + key) * v.kv_heads + g) * kHeadDim + piece * 8)
                                   : make_uint4(0U, 0U, 0U, 0U);
        }
        const uint4* const current = tile_mem + (t & 1U) * (2 * PIECES);
        const uint2* const t2 = reinterpret_cast<const uint2*>(current + PIECES);
#pragma unroll
        for (unsigned slot = 0; slot < 4; ++slot) {
            const unsigned member = bank * 4 + slot;
            if (member < members) {
                const unsigned n_keys = visible[member];
                if (t * kTile < n_keys) {
                    float sv[2][2];
#pragma unroll
                    for (unsigned grp = 0; grp < 2; ++grp) {
                        float sc[2];
                        group_scores_h<2>(reinterpret_cast<const uint2*>(current), grp, q[slot], lane, sc);
                        const unsigned my_key = ((lane >> 4U) & 1U) * 4U + ((lane >> 3U) & 1U) * 2U + ((lane >> 2U) & 1U);
#pragma unroll
                        for (unsigned h = 0; h < 2; ++h)
                            sv[h][grp] = t * kTile + grp * 8 + my_key < n_keys ? sc[h] : -__builtin_inff();
                    }
                    auto tile4 = [&](auto ntc) {
                        constexpr unsigned NTC = decltype(ntc)::value;  // 16: full tile, no per-key predicate; 0: runtime nt
                        const unsigned t0 = t * kTile;
                        const unsigned nt = NTC != 0 ? NTC : (n_keys - t0 < kTile ? n_keys - t0 : kTile);
                        float p0[2], p1[2];
#pragma unroll
                        for (unsigned h = 0; h < 2; ++h) {
                            const float sa = sv[h][0], sb = sv[h][1];
                            float tm = -__builtin_inff();
#pragma unroll
                            for (unsigned j = 0; j < kTile; ++j)
                                if (NTC != 0 || j < nt) tm = max_of(tm, rl(j < 8 ? sa : sb, key_lane(j % 8)));
                            const float mn = max_of(m[slot][h], tm);
                            const float corr = t == 0 ? 0.0F : exp_f(m[slot][h] - mn);
                            l[slot][h] = l[slot][h] * corr;
                            for (unsigned i = 0; i < kDimsPerLane; ++i) acc[slot][h][i] = acc[slot][h][i] * corr;
                            p0[h] = exp_f(sa - mn);
                            p1[h] = exp_f(sb - mn);
                            m[slot][h] = mn;
                        }
                        uint2 vrow[kTile];  // full tile: the 16 V rows read before the chain (see version 3)
                        if constexpr (NTC != 0) {
#pragma unroll
                            for (unsigned j = 0; j < kTile; ++j) vrow[j] = t2[j * 32 + lane];
                            __builtin_amdgcn_sched_barrier(0);
                        }
#pragma unroll
                        for (unsigned j = 0; j < kTile; ++j) {
                            if (NTC != 0 || j < nt) {
                                const uint2 raw = NTC != 0 ? vrow[j] : t2[j * 32 + lane];
                                const float v0 = lo_bf16(raw.x), v1 = hi_bf16(raw.x), v2 = lo_bf16(raw.y), v3 = hi_bf16(raw.y);
#pragma unroll
                                for (unsigned h = 0; h < 2; ++h) {
                                    const float pj = rl(j < 8 ? p0[h] : p1[h], key_lane(j % 8));
                                    l[slot][h] = l[slot][h] + pj;
                                    acc[slot][h][0] = __builtin_fmaf(pj, v0, acc[slot][h][0]);
                                    acc[slot][h][1] = __builtin_fmaf(pj, v1, acc[slot][h][1]);
                                    acc[slot][h][2] = __builtin_fmaf(pj, v2, acc[slot][h][2]);
                                    acc[slot][h][3] = __builtin_fmaf(pj, v3, acc[slot][h][3]);
                                }
                            }
                        }
                    };
                    if (v.pv_full_tiles != 0 && t * kTile + kTile <= n_keys) tile4(std::integral_constant<unsigned, kTile>{});
                    else tile4(std::integral_constant<unsigned, 0>{});
                }
            }
        }
        // The alternate slot was last consumed before the current publish barrier.
        if (tid < PIECES && t + 1 < tiles) {
            uint4* const next = tile_mem + ((t + 1) & 1U) * (2 * PIECES);
            next[tid] = next_k;
            next[PIECES + tid] = next_v;
        }
    }
#pragma unroll
    for (unsigned slot = 0; slot < 4; ++slot) {
        const unsigned member = bank * 4 + slot;
        if (member < members) {
            const unsigned r = selected_row[member], c = ordinal[member];
#pragma unroll
            for (unsigned h = 0; h < 2; ++h) {
                float* out = v.partial + ((static_cast<std::size_t>(r) * v.q_heads + qh0 + h) * kMaxChunks + c) * kPartial;
                if (lane == 0) {
                    out[0] = m[slot][h];
                    out[1] = l[slot][h];
                }
#pragma unroll
                for (unsigned i = 0; i < kDimsPerLane; ++i) out[2 + lane * kDimsPerLane + i] = acc[slot][h][i];
            }
        }
    }
}

// Selection once per (row, KV head) (DecodeView::select_once): the K3 work of the attend launch's selecting blocks,
// done by one 256-thread block per (row, KV head) instead of by every chunk block. Same order keys, same choose_topk,
// same ascending compaction with the newest block last, so idx / idx_count are the bytes the attend launch writes
// with select; the attend launch then runs with select = false.
template <unsigned NT> __global__ __launch_bounds__(NT) void select_kernel(DecodeView v) {
    extern __shared__ uint4 dyn_smem[];
    SelView<NT> sel(dyn_smem, v.max_blocks);
    const unsigned tid = threadIdx.x, rg = blockIdx.x, r = rg / v.kv_heads;
    const std::uint32_t nb = static_cast<std::uint32_t>(block_count(v.pos[r] + 1));
    std::uint32_t* const idx = v.idx + static_cast<std::size_t>(rg) * kTopK;
    if (nb <= kTopK) {
        if (tid < nb) idx[tid] = tid;
        if (tid == 0) v.idx_count[rg] = nb;
        return;
    }
    const std::uint32_t rest = nb - 1;
    const float* sc = v.scores + static_cast<std::size_t>(rg) * v.max_blocks;  // index head g = KV head g
    for (unsigned b = tid; b < rest; b += NT) sel.keys[b] = order_key(sc[b]);
    __syncthreads();
    choose_topk<NT>(sel.keys, rest, kTopK - 1, sel.chosen, sel.hist, sel.wave_tot, sel.sh);
    unsigned before = 0;
    for (std::uint32_t base = 0; base < rest; base += NT) {
        const std::uint32_t i = base + tid;
        const bool ch = i < rest && sel.chosen[i] != 0U;
        unsigned total = 0;
        const unsigned rank = before + block_scan<NT>(ch, sel.wave_tot, total);
        if (ch) idx[rank - 1] = i;
        before += total;
    }
    if (tid == 0) {
        idx[kTopK - 1] = rest;  // the newest block, always last
        v.idx_count[rg] = kTopK;
    }
}

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

}  // namespace

namespace {
unsigned g_index_grid_cap = [] {
    const char* e = std::getenv("STRIX_MMATTN_INDEX_GRID");
    const long x = e != nullptr ? std::strtol(e, nullptr, 10) : 0;
    return x > 0 ? static_cast<unsigned>(x) : 160U;
}();
}  // namespace
unsigned index_grid_cap() { return g_index_grid_cap; }
void set_index_grid_cap(unsigned cap) { g_index_grid_cap = cap > 0 ? cap : 1U; }

void enqueue_index(const DecodeView& v, hipStream_t stream) {
    const unsigned rows = v.seqs * v.rows_per_seq;
    if (rows == 0 || rows > kMaxRows || v.idx_heads == 0 || v.idx_heads > kMaxIdxHeads || v.shard_count == 0 ||
        v.shard_rank >= v.shard_count || v.max_blocks == 0)
        throw std::invalid_argument("minimax fused index: bad view");
    const unsigned local = (v.max_blocks + v.shard_count - 1 - v.shard_rank) / v.shard_count;
    const unsigned items = v.seqs * local;
    const unsigned cap = index_grid_cap(), grid = items == 0 ? 1U : items < cap ? items : cap;  // >= 1: K/V rows
    const std::size_t lds = kChunks16 * sizeof(uint4) + static_cast<std::size_t>(rows) * v.idx_heads * kIdxDim * sizeof(float);
    hipLaunchKernelGGL((index_kernel<DecodeView>), dim3(grid), dim3(kIdxThreads), lds, stream, v);
    check(hipGetLastError(), "minimax fused index");
}

unsigned attend_heads(const DecodeView& v) {
    if (v.heads_per_block != 0) return v.heads_per_block;
    return kGroup;  // measured policy (qualification notes): 16 heads per block is never slower
}

std::size_t g_attend_lds_min = 43008;  // default = the measured footprint (dynamic 32 KB was mixed, measurement)
void set_attend_lds_min(std::size_t bytes) { g_attend_lds_min = bytes; }

unsigned attend_version(const DecodeView& v) { return v.attend_version != 0 ? v.attend_version : 3U; }

template <bool DENSE> void launch_attend(const DecodeView& v, bool select, hipStream_t stream) {
    const unsigned rows = v.seqs * v.rows_per_seq, hw = attend_heads(v), ver = attend_version(v);
    if (ver == 4) {  // all 16 heads per block, heads_per_wave per wave; dynamic LDS at its real size
        const unsigned hpw = v.heads_per_wave != 0 ? v.heads_per_wave : 2U;
        const std::size_t need = DENSE ? 0 : sel_bytes(v.max_blocks, kGroup / hpw * 32);
        const std::size_t lds4 = need > kChunks16 * sizeof(uint4) ? need : kChunks16 * sizeof(uint4);
        const dim3 grid4(rows * v.kv_heads, kMaxChunks);
        const int sel4 = select ? 1 : 0;
        switch (hpw) {
        case 1: hipLaunchKernelGGL((attend4_kernel<1, DENSE>), grid4, dim3(16 * 32), lds4, stream, v, sel4); break;
        case 2: hipLaunchKernelGGL((attend4_kernel<2, DENSE>), grid4, dim3(8 * 32), lds4, stream, v, sel4); break;
        case 4: hipLaunchKernelGGL((attend4_kernel<4, DENSE>), grid4, dim3(4 * 32), lds4, stream, v, sel4); break;
        default: throw std::invalid_argument("minimax fused attend: heads_per_wave must be 1, 2 or 4");
        }
        check(hipGetLastError(), "minimax fused attend v4");
        return;
    }
    const dim3 grid(rows * v.kv_heads * (kGroup / hw), kMaxChunks);
    const int sel = select ? 1 : 0;
    // versions 2/3: dynamic LDS = the 32 KB tile or the selection scratch, whichever is larger (dense: the tile)
    const std::size_t sel_need = DENSE ? 0 : sel_bytes(v.max_blocks, hw * 32);
    std::size_t lds = sel_need > kChunks16 * sizeof(uint4) ? sel_need : kChunks16 * sizeof(uint4);
    // The floor keeps the measured 42 KB footprint except where the 32 KB diet won (t9: sparse N2/W1 8K/32K/128K) and
    // the selection scratch fits the tile; only the allocation (occupancy) changes.
    const bool diet = v.lds_diet && !DENSE && sel_need <= kChunks16 * sizeof(uint4);
    if (!diet && g_attend_lds_min > lds) lds = g_attend_lds_min;
    if (ver < 1 || ver > 3) throw std::invalid_argument("minimax fused attend: attend_version must be 1, 2, 3 or 4");
    switch (hw * 10 + ver) {
    case 161: hipLaunchKernelGGL((attend_kernel<16, DENSE>), grid, dim3(16 * 32), 0, stream, v, sel); break;
    case 81: hipLaunchKernelGGL((attend_kernel<8, DENSE>), grid, dim3(8 * 32), 0, stream, v, sel); break;
    case 41: hipLaunchKernelGGL((attend_kernel<4, DENSE>), grid, dim3(4 * 32), 0, stream, v, sel); break;
    case 162: hipLaunchKernelGGL((attend2_kernel<16, DENSE>), grid, dim3(16 * 32), lds, stream, v, sel); break;
    case 82: hipLaunchKernelGGL((attend2_kernel<8, DENSE>), grid, dim3(8 * 32), lds, stream, v, sel); break;
    case 42: hipLaunchKernelGGL((attend2_kernel<4, DENSE>), grid, dim3(4 * 32), lds, stream, v, sel); break;
    case 163: hipLaunchKernelGGL((attend2_kernel<16, DENSE, true>), grid, dim3(16 * 32), lds, stream, v, sel); break;
    case 83: hipLaunchKernelGGL((attend2_kernel<8, DENSE, true>), grid, dim3(8 * 32), lds, stream, v, sel); break;
    case 43: hipLaunchKernelGGL((attend2_kernel<4, DENSE, true>), grid, dim3(4 * 32), lds, stream, v, sel); break;
    default: throw std::invalid_argument("minimax fused attend: heads_per_block must be 4, 8 or 16");
    }
    check(hipGetLastError(), "minimax fused attend");
}

void enqueue_attend(const DecodeView& v, bool select, hipStream_t stream) {
    const unsigned rows = v.seqs * v.rows_per_seq;
    if (rows == 0 || rows > kMaxRows || v.max_blocks > kMaxBlocks || (select && v.idx_heads != v.kv_heads))
        throw std::invalid_argument("minimax fused attend: bad view");
    launch_attend<false>(v, select, stream);
}

std::size_t prefill_window_bytes(std::uint32_t max_blocks) {
    return std::size_t{kMaxRows} * (2 * std::size_t{max_blocks} + 2 * kTopK + 2 +
                                  32 * kMaxChunks * 130 + 2 * 4) * sizeof(std::uint32_t);
}

namespace {
template <bool DENSE> void launch_prefill_attend(const PrefillView& p, unsigned windows, hipStream_t stream) {
    if (windows == 0) return;
    const DecodeView& v = p.window;
    if (p.msa_rework != 0) {
        PrefillView scheduled = p;
        const unsigned end_row = (p.first_window + windows) * kMaxRows;
        if (scheduled.rows > end_row) scheduled.rows = end_row;
        if constexpr (!DENSE) {
            hipLaunchKernelGGL(prefill_select_kernel, dim3(kMaxRows * v.kv_heads, 1, windows), dim3(256),
                               sel_bytes(v.max_blocks, 256), stream, scheduled);
            check(hipGetLastError(), "minimax prefill shared selection");
        }
        const auto chunks = prefill_chunk_prefix(p.pos0 + scheduled.rows) -
                            prefill_chunk_prefix(p.pos0 + p.first_window * kMaxRows);
        if (p.msa_rework == 7) {
            // Mode 6 producer with the ticket combine fused into the last chunk block of each (row, group):
            // same partials, same combine expressions and ascending chunk order, no separate combine launch.
            const RepairedPrefillView repaired{scheduled};
            hipLaunchKernelGGL((attend4_kernel<2, DENSE, RepairedPrefillView, true, true, true, true, true, true>),
                               dim3(static_cast<unsigned>(chunks * v.kv_heads)), dim3(256),
                               4 * kTile * kHeadDim * sizeof(std::uint16_t), stream, repaired, 0);
            check(hipGetLastError(), "minimax prefill fused partial producers");
            return;
        }
        if (p.msa_rework == 4) {
            const unsigned blocks = static_cast<unsigned>(block_count(p.pos0 + scheduled.rows));
            hipLaunchKernelGGL((prefill_r16_kernel<DENSE>), dim3(blocks, v.kv_heads, windows), dim3(1024),
                               4 * kTile * kHeadDim * sizeof(std::uint16_t), stream, scheduled);
        } else if (p.msa_rework == 6) {
            const RepairedPrefillView repaired{scheduled};
            hipLaunchKernelGGL((attend4_kernel<2, DENSE, RepairedPrefillView, true, true, true, true, true>),
                               dim3(static_cast<unsigned>(chunks * v.kv_heads)), dim3(256),
                               4 * kTile * kHeadDim * sizeof(std::uint16_t), stream, repaired, 0);
        } else if (p.msa_rework == 5) {
            const RepairedPrefillView repaired{scheduled};
            hipLaunchKernelGGL((attend4_kernel<2, DENSE, RepairedPrefillView, true, true, true, true>),
                               dim3(static_cast<unsigned>(chunks * v.kv_heads)), dim3(256),
                               4 * kTile * kHeadDim * sizeof(std::uint16_t), stream, repaired, 0);
        } else if (p.msa_rework == 3) {
            const RepairedPrefillView repaired{scheduled};
            hipLaunchKernelGGL((attend4_kernel<2, DENSE, RepairedPrefillView, true, true>),
                               dim3(static_cast<unsigned>(chunks * v.kv_heads)), dim3(256),
                               4 * kTile * kHeadDim * sizeof(std::uint16_t), stream, repaired, 0);
        } else if (p.msa_rework == 2) {
            hipLaunchKernelGGL((attend4_kernel<2, DENSE, PrefillView, true, true>),
                               dim3(static_cast<unsigned>(chunks * v.kv_heads)), dim3(256),
                               4 * kTile * kHeadDim * sizeof(std::uint16_t), stream, scheduled, 0);
        } else {
            hipLaunchKernelGGL((attend4_kernel<2, DENSE, PrefillView, true>),
                               dim3(static_cast<unsigned>(chunks * v.kv_heads)), dim3(256),
                               kChunks16 * sizeof(uint4), stream, scheduled, 0);
        }
        check(hipGetLastError(), "minimax prefill partial producers");
        hipLaunchKernelGGL((prefill_combine_kernel<DENSE>), dim3(kMaxRows, v.q_heads / 8, windows),
                           dim3(256), 0, stream, scheduled);
        check(hipGetLastError(), "minimax prefill separate combine");
        return;
    }
    const dim3 grid(kMaxRows * v.kv_heads, kMaxChunks, windows);
    const int select = DENSE ? 0 : 1;
    if (attend_version(v) == 4) {
        const std::size_t need = DENSE ? 0 : sel_bytes(v.max_blocks, 8 * 32);
        const std::size_t lds = need > kChunks16 * sizeof(uint4) ? need : kChunks16 * sizeof(uint4);
        hipLaunchKernelGGL((attend4_kernel<2, DENSE, PrefillView>), grid, dim3(8 * 32), lds, stream, p, select);
    } else {
        const std::size_t need = DENSE ? 0 : sel_bytes(v.max_blocks, 16 * 32);
        std::size_t lds = need > kChunks16 * sizeof(uint4) ? need : kChunks16 * sizeof(uint4);
        if (g_attend_lds_min > lds) lds = g_attend_lds_min;
        hipLaunchKernelGGL((attend2_kernel<16, DENSE, true, PrefillView>), grid, dim3(16 * 32), lds, stream, p, select);
    }
    check(hipGetLastError(), "minimax batched prefill attend");
}
}  // namespace

namespace {
template <bool DENSE> void launch_prefill_partials(const PrefillView& p, unsigned windows, hipStream_t stream) {
    if (windows == 0) return;
    const DecodeView& v = p.window;
    const auto chunks = prefill_chunk_prefix(p.pos0 + p.rows) -
                        prefill_chunk_prefix(p.pos0 + p.first_window * kMaxRows);
    if (p.msa_rework == 0) {
        hipLaunchKernelGGL((attend4_kernel<2, DENSE, PrefillView, true, false, false>),
                           dim3(kMaxRows * v.kv_heads, kMaxChunks, windows), dim3(256),
                           kChunks16 * sizeof(uint4), stream, p, 0);
    } else if (p.msa_rework == 1) {
        hipLaunchKernelGGL((attend4_kernel<2, DENSE, PrefillView, true>),
                           dim3(static_cast<unsigned>(chunks * v.kv_heads)), dim3(256),
                           kChunks16 * sizeof(uint4), stream, p, 0);
    } else if (p.msa_rework == 4) {
        const unsigned blocks = static_cast<unsigned>(block_count(p.pos0 + p.rows));
        hipLaunchKernelGGL((prefill_r16_kernel<DENSE>), dim3(blocks, v.kv_heads, windows), dim3(1024),
                           4 * kTile * kHeadDim * sizeof(std::uint16_t), stream, p);
    } else if (p.msa_rework == 6 || p.msa_rework == 7) {  // diagnostic: mode 7 = mode 6 producer (no tickets here)
        const RepairedPrefillView repaired{p};
        hipLaunchKernelGGL((attend4_kernel<2, DENSE, RepairedPrefillView, true, true, true, true, true>),
                           dim3(static_cast<unsigned>(chunks * v.kv_heads)), dim3(256),
                           4 * kTile * kHeadDim * sizeof(std::uint16_t), stream, repaired, 0);
    } else if (p.msa_rework == 5) {
        const RepairedPrefillView repaired{p};
        hipLaunchKernelGGL((attend4_kernel<2, DENSE, RepairedPrefillView, true, true, true, true>),
                           dim3(static_cast<unsigned>(chunks * v.kv_heads)), dim3(256),
                           4 * kTile * kHeadDim * sizeof(std::uint16_t), stream, repaired, 0);
    } else if (p.msa_rework == 3) {
        const RepairedPrefillView repaired{p};
        hipLaunchKernelGGL((attend4_kernel<2, DENSE, RepairedPrefillView, true, true>),
                           dim3(static_cast<unsigned>(chunks * v.kv_heads)), dim3(256),
                           4 * kTile * kHeadDim * sizeof(std::uint16_t), stream, repaired, 0);
    } else {
        hipLaunchKernelGGL((attend4_kernel<2, DENSE, PrefillView, true, true>),
                           dim3(static_cast<unsigned>(chunks * v.kv_heads)), dim3(256),
                           4 * kTile * kHeadDim * sizeof(std::uint16_t), stream, p, 0);
    }
    check(hipGetLastError(), "minimax prefill diagnostic partial producers");
}
}  // namespace

namespace {
void validate_prefill_diagnostic(const PrefillView& p) {
    const DecodeView& v = p.window;
    const unsigned windows = p.rows == 0 ? 0 : (p.rows - 1) / kMaxRows + 1;
    if (p.msa_rework > 7 || windows == 0 || windows > 65535U || p.first_window != 0 ||
        p.scratch_windows == 0 || windows > p.scratch_windows ||
        v.rows_per_seq != (p.rows < kMaxRows ? p.rows : kMaxRows) || v.seqs != 1 ||
        v.q_heads != 32 || v.kv_heads != 2 || v.max_blocks == 0 || v.max_blocks > kMaxBlocks ||
        p.pos0 >= std::uint64_t{v.max_blocks} * kBlock ||
        p.rows > std::uint64_t{v.max_blocks} * kBlock - p.pos0 ||
        v.q == nullptr || v.q_w == nullptr || v.rope == nullptr || v.pos == nullptr ||
        v.k_cache == nullptr || v.v_cache == nullptr || v.idx == nullptr || v.idx_count == nullptr ||
        v.partial == nullptr || v.phase != nullptr)
        throw std::invalid_argument("minimax prefill partials: bad view");
}
}  // namespace

// Diagnostic launch: prepared cache/selection, one scratch batch, no combine or tickets.
void enqueue_prefill_partials(const PrefillView& p, hipStream_t stream) {
    validate_prefill_diagnostic(p);
    const unsigned windows = (p.rows - 1) / kMaxRows + 1;
    // Match production's dense/sparse split within this scratch batch. Every
    // sparse window starts after position 1920, hence every row selects 16 blocks.
    unsigned dense = 0;
    for (; dense < windows; ++dense) {
        const unsigned end = (dense + 1) * kMaxRows < p.rows ? (dense + 1) * kMaxRows : p.rows;
        if (p.pos0 + end - 1 >= kTopK * kBlock) break;
    }
    PrefillView scheduled = p;
    scheduled.first_window = dense;
    launch_prefill_partials<false>(scheduled, windows - dense, stream);
    scheduled.first_window = 0;
    scheduled.rows = dense * kMaxRows < p.rows ? dense * kMaxRows : p.rows;
    launch_prefill_partials<true>(scheduled, dense, stream);
}

// Prepared scores, shared selection only. R16 membership is fused into producers.
void enqueue_prefill_selection(const PrefillView& p, hipStream_t stream) {
    validate_prefill_diagnostic(p);
    if (p.window.scores == nullptr) throw std::invalid_argument("minimax prefill selection: missing scores");
    const unsigned windows = (p.rows - 1) / kMaxRows + 1;
    unsigned dense = 0;
    for (; dense < windows; ++dense) {
        const unsigned end = (dense + 1) * kMaxRows < p.rows ? (dense + 1) * kMaxRows : p.rows;
        if (p.pos0 + end - 1 >= kTopK * kBlock) break;
    }
    if (dense == windows) return;
    PrefillView scheduled = p;
    scheduled.first_window = dense;
    hipLaunchKernelGGL(prefill_select_kernel, dim3(kMaxRows * p.window.kv_heads, 1, windows - dense),
                       dim3(256), sel_bytes(p.window.max_blocks, 256), stream, scheduled);
    check(hipGetLastError(), "minimax prefill diagnostic selection");
}

void enqueue_prefill_index(const PrefillView& p, hipStream_t stream) {
    const DecodeView& v = p.window;
    const unsigned windows = p.rows == 0 ? 0 : (p.rows - 1) / kMaxRows + 1;
    if (windows == 0 || windows > 65535U || windows > p.scratch_windows || p.first_window >= windows ||
        v.seqs != 1 || v.idx_heads != 2 || v.shard_count != 1 || v.shard_rank != 0 ||
        v.max_blocks == 0 || v.max_blocks > kMaxBlocks || v.iq == nullptr || v.iq_w == nullptr ||
        v.rope == nullptr || v.pos == nullptr || v.ik_cache == nullptr || v.scores == nullptr ||
        v.ik == nullptr || v.ik_w == nullptr ||
        p.pos0 >= std::uint64_t{v.max_blocks} * kBlock ||
        p.rows > std::uint64_t{v.max_blocks} * kBlock - p.pos0 ||
        v.phase_index != nullptr || (p.index_rows != 16 && p.index_rows != 32 && p.index_rows != 64))
        throw std::invalid_argument("minimax prefill index: bad view");
    const unsigned query_rows = p.index_rows;
    const unsigned sparse = windows - p.first_window;
    const unsigned grid = v.max_blocks < index_grid_cap() ? v.max_blocks : index_grid_cap();
    const unsigned rows = p.index_tile ? query_rows : kMaxRows;
    // Q64 overlays queries/maxima on key staging; legacy modes retain their layout.
    const std::size_t query_lds = p.index_tile && rows == 64 ? 0U
        : std::size_t{rows} * v.idx_heads * kIdxDim * sizeof(float);
    const std::size_t lds = kChunks16 * sizeof(uint4) + query_lds;
    if (p.index_tile && rows == 64) {
        const unsigned tiles = (sparse + 3U) / 4U;
        hipLaunchKernelGGL(prefill_index_q64_kernel, dim3(grid, tiles), dim3(kIdxThreads), lds, stream, p);
    } else if (p.index_tile) {
        const unsigned tiles = (sparse + rows / kMaxRows - 1) / (rows / kMaxRows);
        hipLaunchKernelGGL(prefill_index_tile_kernel, dim3(grid, tiles), dim3(kIdxThreads), lds, stream, p, rows);
    } else {
        hipLaunchKernelGGL((index_kernel<PrefillView>), dim3(grid, sparse), dim3(kIdxThreads), lds, stream, p);
    }
    check(hipGetLastError(), "minimax batched prefill index");
}

void enqueue_prefill(const PrefillView& p, hipStream_t stream, PrefillStageMarker marker, void* user) {
    const DecodeView& v = p.window;
    if (p.msa_rework > 7 || (p.index_rows != 16 && p.index_rows != 32 && p.index_rows != 64) ||
        p.rows == 0 || p.scratch_windows == 0 || p.first_window != 0 || v.seqs != 1 ||
        v.q_heads != 32 || v.kv_heads != 2 || v.idx_heads != 2 || v.shard_count != 1 || v.shard_rank != 0 ||
        v.max_blocks == 0 || v.max_blocks > kMaxBlocks || (attend_version(v) != 3 && attend_version(v) != 4) ||
        (v.heads_per_block != 0 && v.heads_per_block != 16) ||
        (attend_version(v) == 4 && v.heads_per_wave != 0 && v.heads_per_wave != 2) ||
        v.phase != nullptr || v.phase_index != nullptr ||
        p.pos0 >= std::uint64_t{v.max_blocks} * kBlock ||
        p.rows > std::uint64_t{v.max_blocks} * kBlock - p.pos0 ||
        v.q == nullptr || v.k == nullptr || v.v == nullptr || v.iq == nullptr || v.ik == nullptr ||
        v.q_w == nullptr || v.k_w == nullptr || v.iq_w == nullptr || v.ik_w == nullptr ||
        v.rope == nullptr || v.pos == nullptr || v.k_cache == nullptr || v.v_cache == nullptr || v.ik_cache == nullptr ||
        v.scores == nullptr || v.idx == nullptr || v.idx_count == nullptr || v.partial == nullptr ||
        v.ticket == nullptr || v.o == nullptr)
        throw std::invalid_argument("minimax batched prefill: bad view");
    const unsigned prep_grid = index_grid_cap();
    if (marker != nullptr) marker(user, "msa-prep", true);
    hipLaunchKernelGGL(prefill_prep_kernel, dim3(prep_grid), dim3(kIdxThreads), 0, stream, p);
    check(hipGetLastError(), "minimax batched prefill prep");
    if (marker != nullptr) marker(user, "msa-prep", false);
    const unsigned windows = (p.rows - 1) / kMaxRows + 1;
    const unsigned batch_cap = p.scratch_windows < 65535U ? p.scratch_windows : 65535U;
    for (unsigned first = 0; first < windows;) {
        const unsigned nw = windows - first < batch_cap ? windows - first : batch_cap;
        const unsigned r0 = first * kMaxRows;
        PrefillView batch{prefill_offset(v, r0), p.rows - r0, p.pos0 + r0, p.scratch_windows, 0, p.index_tile, p.index_rows, p.msa_rework};
        if (batch.rows > nw * kMaxRows) batch.rows = nw * kMaxRows;
        const unsigned dense = route::dense_windows(batch.pos0, batch.rows, nw, kMaxRows, kTopK * kBlock);
        const unsigned sparse = nw - dense;
        if (sparse != 0) {
            batch.first_window = dense;
            if (marker != nullptr) marker(user, "index-scoring", true);
            enqueue_prefill_index(batch, stream);
            if (marker != nullptr) marker(user, "index-scoring", false);
            if (marker != nullptr) marker(user, "attend", true);
            launch_prefill_attend<false>(batch, sparse, stream);
            if (marker != nullptr) marker(user, "attend", false);
        }
        batch.first_window = 0;
        if (dense != 0 && marker != nullptr) marker(user, "attend", true);
        launch_prefill_attend<true>(batch, dense, stream);
        if (dense != 0 && marker != nullptr) marker(user, "attend", false);
        first += nw;
    }
}

void enqueue_decode(const DecodeView& v, hipStream_t stream) {
    if (v.dense) {
        const unsigned rows = v.seqs * v.rows_per_seq;
        if (rows == 0 || rows > kMaxRows) throw std::invalid_argument("minimax fused dense: bad view");
        launch_attend<true>(v, false, stream);
        return;
    }
    enqueue_index(v, stream);
    if (v.select_once) {
        const unsigned rows = v.seqs * v.rows_per_seq;
        if (rows == 0 || rows > kMaxRows || v.max_blocks > kMaxBlocks || v.idx_heads != v.kv_heads || v.shard_count != 1)
            throw std::invalid_argument("minimax fused select: bad view (context-parallel views select by candidates/merge)");
        hipLaunchKernelGGL((select_kernel<256>), dim3(rows * v.kv_heads), dim3(256), sel_bytes(v.max_blocks, 256), stream, v);
        check(hipGetLastError(), "minimax fused select");
        enqueue_attend(v, false, stream);
        return;
    }
    enqueue_attend(v, true, stream);
}

}  // namespace strix::minimax::msa::fused
