// CUDA QSA layer core, bit-identical to the HIP single-A8-profile tiled prefill (see include/strix/cuda_qwen_qsa.hpp).
// HIP references: src/hip_indexed_attention_launcher.hip (H: line numbers below), ISA
// the gfx1151 disassembly of hip_indexed_attention_launcher (S: line numbers).
// Every FP operation is an explicit IEEE intrinsic (__fadd_rn, __fmul_rn, __fmaf_rn, __fdiv_rn) in the HIP order.
#include "strix/cuda_qwen_qsa.hpp"
#include "strix/cuda_vram_guard.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::cuda::qwen_qsa {
namespace {

namespace am = strix::cuda::amdmath;

constexpr unsigned kFull = 0xffffffffU;
#define kInf __uint_as_float(0x7f800000U)

__device__ __forceinline__ float bf16f(std::uint16_t b) { return __uint_as_float(static_cast<std::uint32_t>(b) << 16U); }
// H:328-343 float_to_bf16: (bits + 0x7fff + lsb) >> 16, no NaN special case.
__device__ __forceinline__ std::uint16_t to_bf16(float v) {
    const auto b = __float_as_uint(v);
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
}
__device__ __forceinline__ float round_bf16(float v) { return bf16f(to_bf16(v)); }

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("qwen_qsa: ") + what + ": " + cudaGetErrorString(e));
}

// ------------------------------------------------------------------------------------------------------------------
// Indexer projection: linear_to_float_rows_kernel<bf16,16> (H:3566-3591; S:182274 = ..._kernelILNS0_10WeightKindE2ELj16E..).
// Per (row, output o): thread t (256) keeps sum = fma(act[t+256k], w[o][t+256k], sum), k = 0..9 ascending from 0
// (S:182477-182494 v_fmac_f32 vSUM, vW, vACT: contracted); then the LDS tree strides 128, 64, ..., 1
// (scratch[t] += scratch[t+stride], S:182536-), round_bf16. One addition commutes, so any (lower, upper) operand order.
// CUDA: block = 8 rows x a slice of 64 outputs, 4 outputs per pass; the 8 activation rows staged once in shared memory
// (BF16), each activation read from shared memory feeds 4 FMAs (4 weight rows in registers); strides 128/64/32 in shared
// memory, 16..1 by shuffles (4 (row, output) pairs per warp).
constexpr unsigned kIdxRows = 8, kIdxOuts = 4, kIdxOutputs = kIndexQWidth + kIndexDim, kIdxSlice = 64;
constexpr unsigned kIdxPairs = kIdxRows * kIdxOuts;  // 32
constexpr std::size_t kIdxSmem = kIdxRows * kHidden * sizeof(std::uint16_t) + kIdxPairs * 256U * sizeof(float);
static_assert(kIdxSlice % kIdxOuts == 0 && kIdxOutputs % kIdxSlice == 0);

__global__ __launch_bounds__(256) void indexer_kernel(const std::uint16_t* __restrict__ hidden, std::uint32_t rows,
    const std::uint16_t* __restrict__ wq, const std::uint16_t* __restrict__ wk, float* __restrict__ iq, float* __restrict__ ik) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto* act = reinterpret_cast<std::uint16_t*>(smem);
    auto* red = reinterpret_cast<float*>(smem + kIdxRows * kHidden * sizeof(std::uint16_t));  // [pair = r*4+oo][256]
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5U;
    const std::uint32_t r0 = blockIdx.x * kIdxRows;
    const unsigned nr = min(kIdxRows, rows - r0);
    {
        const auto* src = reinterpret_cast<const uint4*>(hidden + static_cast<std::uint64_t>(r0) * kHidden);
        auto* dst = reinterpret_cast<uint4*>(act);
        const unsigned n = nr * kHidden / 8U, total = kIdxRows * kHidden / 8U;
        for (unsigned i = t; i < total; i += 256U) dst[i] = i < n ? src[i] : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    for (unsigned o0 = blockIdx.y * kIdxSlice; o0 < (blockIdx.y + 1U) * kIdxSlice; o0 += kIdxOuts) {
        float w[kIdxOuts][10];
#pragma unroll
        for (unsigned oo = 0; oo < kIdxOuts; ++oo) {
            const unsigned o = o0 + oo;
            const std::uint16_t* wrow = o < kIndexQWidth ? wq + static_cast<std::uint64_t>(o) * kHidden
                                                         : wk + static_cast<std::uint64_t>(o - kIndexQWidth) * kHidden;
#pragma unroll
            for (unsigned k = 0; k < 10U; ++k) w[oo][k] = bf16f(__ldg(wrow + t + 256U * k));
        }
#pragma unroll 2
        for (unsigned r = 0; r < kIdxRows; ++r) {
            float sum[kIdxOuts];
#pragma unroll
            for (unsigned oo = 0; oo < kIdxOuts; ++oo) sum[oo] = 0.0F;
#pragma unroll
            for (unsigned k = 0; k < 10U; ++k) {
                const float a = bf16f(act[r * kHidden + t + 256U * k]);
#pragma unroll
                for (unsigned oo = 0; oo < kIdxOuts; ++oo) sum[oo] = __fmaf_rn(a, w[oo][k], sum[oo]);
            }
#pragma unroll
            for (unsigned oo = 0; oo < kIdxOuts; ++oo) red[(r * kIdxOuts + oo) * 256U + t] = sum[oo];
        }
        __syncthreads();
#pragma unroll
        for (unsigned level = 0; level < 3U; ++level) {
            const unsigned half = 128U >> level, shift = 7U - level;
            for (unsigned i = t; i < kIdxPairs * half; i += 256U) {
                const unsigned pr = i >> shift, j = i & (half - 1U);
                red[pr * 256U + j] = __fadd_rn(red[pr * 256U + j], red[pr * 256U + j + half]);
            }
            __syncthreads();
        }
#pragma unroll
        for (unsigned pp = 0; pp < kIdxPairs / 8U; ++pp) {
            const unsigned pr = warp * (kIdxPairs / 8U) + pp, r = pr / kIdxOuts, o = o0 + pr % kIdxOuts;
            float v = red[pr * 256U + lane];
#pragma unroll
            for (unsigned d = 16U; d > 0U; d >>= 1U) v = __fadd_rn(v, __shfl_down_sync(kFull, v, d));
            if (lane == 0U && r < nr) {
                const auto row = static_cast<std::uint64_t>(r0 + r);
                if (o < kIndexQWidth) iq[row * kIndexQWidth + o] = round_bf16(v);
                else ik[row * kIndexDim + (o - kIndexQWidth)] = round_bf16(v);
            }
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------------------------
// norm_rope_body (H:865-996), kernel norm_rope_rows_kernel<f32,64> (S:314600-315315). One block per (head, row), one
// thread per feature:
//   red[f] = x*x (S:314638 v_mul), LDS tree strides hd/2..1 (S:314662 v_add);
//   mean = red/hd + eps (S:314676-314694: correctly rounded v_div sequence, then v_add); inv = rsqrtf(mean) (S:314696-314701
//   OCML: scale by 2^24 below 2^-126, v_rsq, scale 2^12) = amd_rsqrtf;
//   y = round_bf16((x*inv)*w) (S:314704, 314707: two v_mul);
//   RoPE f < 64, pair = f % 32: res = a*cos (S:315272 v_mul) then f >= 32: fma(b, sin, a*cos) (S:315293 v_fmac),
//   f < 32: fma(-b, sin, a*cos) (S:315302 v_fma -v0); b = y[f -/+ 32]; round_bf16.
__device__ __forceinline__ void norm_rope_kernel_body(float* __restrict__ values, std::uint64_t row_stride, std::uint32_t head_dim,
    std::uint32_t head_stride, const float* __restrict__ norm, std::uint64_t first_position, const float2* __restrict__ cs,
    am::Tables tables) {
    extern __shared__ float nsm[];
    float* source = nsm;
    float* reduction = nsm + head_dim;
    const auto head = blockIdx.x, row = blockIdx.y, f = threadIdx.x;
    const auto position = first_position + row;
    float* v = values + row * row_stride + static_cast<std::uint64_t>(head) * head_stride;
    const float input = v[f];
    source[f] = input;
    reduction[f] = __fmul_rn(input, input);
    __syncthreads();
    for (std::uint32_t stride = head_dim / 2U; stride > 0; stride >>= 1U) {
        if (f < stride) reduction[f] = __fadd_rn(reduction[f], reduction[f + stride]);
        __syncthreads();
    }
    const float mean = __fadd_rn(__fdiv_rn(reduction[0], static_cast<float>(head_dim)), kEpsilon);
    const float inv = am::amd_rsqrtf(tables, mean);
    const float y = round_bf16(__fmul_rn(__fmul_rn(input, inv), norm[f]));
    source[f] = y;
    __syncthreads();
    float result = y;
    if (f < kRotaryDim) {
        const auto pair = f % kRotaryPairs;
        const float2 c = cs[position * kRotaryPairs + pair];
        const float ac = __fmul_rn(y, c.x);
        result = f < kRotaryPairs ? __fmaf_rn(-source[f + kRotaryPairs], c.y, ac) : __fmaf_rn(source[f - kRotaryPairs], c.y, ac);
    }
    v[f] = round_bf16(result);
}
__global__ void norm_rope_kernel(float* __restrict__ values, std::uint64_t row_stride, std::uint32_t head_dim,
    std::uint32_t head_stride, const float* __restrict__ norm, std::uint64_t first_position, const float2* __restrict__ cs,
    am::Tables tables) {
    norm_rope_kernel_body(values, row_stride, head_dim, head_stride, norm, first_position, cs, tables);
}

// ------------------------------------------------------------------------------------------------------------------
// store_rows_kernel<Rows> (H:3647-3671): raw index key, K, V -> BF16 (exact for BF16-valued inputs).
__device__ __forceinline__ void store_kernel_body(const float* __restrict__ ik, const float* __restrict__ k, const float* __restrict__ v,
    std::uint64_t first_position, std::uint16_t* __restrict__ raw, std::uint16_t* __restrict__ kv) {
    const auto r = blockIdx.y;
    const auto f = blockIdx.x * blockDim.x + threadIdx.x;
    const auto position = first_position + r;
    if (f < kIndexDim) raw[position * kIndexDim + f] = to_bf16(ik[static_cast<std::uint64_t>(r) * kIndexDim + f]);
    if (f < kKvWidth) {
        const auto base = position * kKvWidth * 2ULL;
        kv[base + f] = to_bf16(k[static_cast<std::uint64_t>(r) * kKvWidth + f]);
        kv[base + kKvWidth + f] = to_bf16(v[static_cast<std::uint64_t>(r) * kKvWidth + f]);
    }
}
__global__ void store_kernel(const float* __restrict__ ik, const float* __restrict__ k, const float* __restrict__ v,
    std::uint64_t first_position, std::uint16_t* __restrict__ raw, std::uint16_t* __restrict__ kv) {
    store_kernel_body(ik, k, v, first_position, raw, kv);
}

// ------------------------------------------------------------------------------------------------------------------
// block_pool_kernel<f32> (H:4172-4208 via the listing at H:3973-4010; S:317565-318324). One block per 4-token block,
// 128 threads:
//   mean = 0 + k0 + k1 + k2 + k3 (S:317611 v_add chain), mean /= 4 (correctly rounded div S:317616-317631), round_bf16;
//   sq = mean*mean (S:317641), LDS tree 64..1; inv = rsqrtf(sq/128 + eps) (S:317677-317705);
//   round_bf16((mean*inv)*w) (S:317706, 317709); RoPE at position 4*block: fma(+-b, sin, a*cos) (S:318275, 318296,
//   318305) exactly as norm_rope; round_bf16.
__device__ __forceinline__ void block_pool_kernel_body(const std::uint16_t* __restrict__ raw, std::uint32_t first_block,
    const float* __restrict__ key_norm, const float2* __restrict__ cs, am::Tables tables, float* __restrict__ pooled_keys) {
    __shared__ float pooled[kIndexDim], scratch[kIndexDim];
    const auto block = first_block + blockIdx.x;
    const auto f = threadIdx.x;
    const auto token_start = static_cast<std::uint64_t>(block) * kRatio;
    float mean = 0.0F;
    for (std::uint32_t o = 0; o < kRatio; ++o) mean = __fadd_rn(mean, bf16f(raw[(token_start + o) * kIndexDim + f]));
    mean = __fdiv_rn(mean, static_cast<float>(kRatio));
    mean = round_bf16(mean);
    pooled[f] = mean;
    scratch[f] = __fmul_rn(mean, mean);
    __syncthreads();
    for (std::uint32_t stride = kIndexDim / 2U; stride > 0; stride >>= 1U) {
        if (f < stride) scratch[f] = __fadd_rn(scratch[f], scratch[f + stride]);
        __syncthreads();
    }
    const float inv = am::amd_rsqrtf(tables, __fadd_rn(__fdiv_rn(scratch[0], static_cast<float>(kIndexDim)), kEpsilon));
    const float y = round_bf16(__fmul_rn(__fmul_rn(mean, inv), key_norm[f]));
    pooled[f] = y;
    __syncthreads();
    float rotated = y;
    if (f < kRotaryDim) {
        const auto pair = f % kRotaryPairs;
        const float2 c = cs[token_start * kRotaryPairs + pair];
        const float ac = __fmul_rn(y, c.x);
        rotated = f < kRotaryPairs ? __fmaf_rn(-pooled[f + kRotaryPairs], c.y, ac) : __fmaf_rn(pooled[f - kRotaryPairs], c.y, ac);
    }
    pooled_keys[static_cast<std::uint64_t>(block) * kIndexDim + f] = round_bf16(rotated);
}
__global__ void block_pool_kernel(const std::uint16_t* __restrict__ raw, std::uint32_t first_block,
    const float* __restrict__ key_norm, const float2* __restrict__ cs, am::Tables tables, float* __restrict__ pooled_keys) {
    block_pool_kernel_body(raw, first_block, key_norm, cs, tables, pooled_keys);
}

// ------------------------------------------------------------------------------------------------------------------
// Lane-xor reduce-scatter qsa_scatter<P> (H:3882-3913): step D = 16 (permlanex16 identity = lane ^ 16), 8, 4, 2, 1
// (DPP row_xmask = lane ^ D) halves the carried values, the lower lane keeping the lower half; each step adds the
// partials of lanes L and L ^ D (one addition: operand order irrelevant). Per value this is the 32-lane tree
// x_L + x_{L^16}, then ^8, ^4, ^2, ^1, i.e. lane 0's shfl_down 16..1 tree.
template <unsigned P, unsigned Step = 0U> __device__ __forceinline__ void scatter(float (&v)[P], unsigned lane) {
    if constexpr (Step < 5U) {
        constexpr unsigned D = 16U >> Step, Count = (P >> Step) > 0U ? (P >> Step) : 1U;
        if constexpr (Count > 1U) {
            constexpr unsigned H = Count / 2U;
            const bool upper = (lane & D) != 0U;
#pragma unroll
            for (unsigned j = 0; j < H; ++j) {
                const float low = __shfl_xor_sync(kFull, v[j], D), high = __shfl_xor_sync(kFull, v[j + H], D);
                v[j] = upper ? __fadd_rn(v[j + H], high) : __fadd_rn(v[j], low);
            }
        } else {
            v[0] = __fadd_rn(v[0], __shfl_xor_sync(kFull, v[0], D));
        }
        scatter<P, Step + 1U>(v, lane);
    }
}
template <unsigned P> __device__ __forceinline__ unsigned scatter_head(unsigned lane) {
    unsigned head = 0, weight = P / 2U;
#pragma unroll
    for (unsigned D = 16U; D > 0U && weight > 0U; D >>= 1U, weight >>= 1U) if (lane & D) head += weight;
    return head;
}

// ------------------------------------------------------------------------------------------------------------------
// block_score_pooled_rows_kernel<Rows,4> (H:4211-4259, `#pragma clang fp contract(off)`; S:322261). One warp scores 4
// consecutive rows against 32 consecutive blocks. Per (row, block, head): lane l, p_j = q[l+32j]*k[l+32j] (j=0..3),
// partial = (p0+p2)+(p1+p3), scatter<4>; total = 0 + relu(h0) + relu(h1) + relu(h2) + relu(h3) (head order),
// score = total / sqrtf(128). Written for blocks < complete(row) of rows with complete > 512.
constexpr unsigned kScoreRowsPerWave = 4;
__device__ __forceinline__ void block_score_body(const float* __restrict__ index_query, std::uint32_t rows,
    std::uint64_t first_position, const float* __restrict__ pooled_keys, std::uint32_t scored_blocks,
    float* __restrict__ block_scores, std::uint32_t max_blocks, std::uint32_t wave_base) {
    constexpr unsigned R = kScoreRowsPerWave;
    const unsigned lane = threadIdx.x & 31U;
    const auto wave = blockIdx.x * 8U + threadIdx.x / 32U - wave_base;
    const auto chunks = (scored_blocks + 31U) / 32U;
    const auto r0 = wave / chunks * R, base = wave % chunks * 32U;
    if (r0 >= rows) return;
    float q[R][kIndexHeads][4];
    std::uint32_t blocks[R];
    bool any = false;
#pragma unroll
    for (unsigned rr = 0; rr < R; ++rr) {
        const auto r = r0 + rr;
        std::uint32_t complete = 0;
        if (r < rows) complete = static_cast<std::uint32_t>((first_position + r + 1U) / kRatio);
        blocks[rr] = complete > kTopK ? complete : 0U;
        any = any || base < blocks[rr];
#pragma unroll
        for (unsigned h = 0; h < kIndexHeads; ++h)
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j)
                q[rr][h][j] = r < rows ? index_query[static_cast<std::uint64_t>(r) * kIndexQWidth + h * kIndexDim + lane + 32U * j] : 0.0F;
    }
    if (!any) return;  // uniform per warp; such a wave writes nothing in HIP either
    float result[R];
#pragma unroll
    for (unsigned rr = 0; rr < R; ++rr) result[rr] = 0.0F;
    const float scale = __fsqrt_rn(static_cast<float>(kIndexDim));
    const auto last = min(32U, scored_blocks - base);
    for (unsigned i = 0; i < last; ++i) {
        const auto* key = pooled_keys + static_cast<std::uint64_t>(base + i) * kIndexDim + lane;
        const float k0 = key[0], k1 = key[32], k2 = key[64], k3 = key[96];
#pragma unroll
        for (unsigned rr = 0; rr < R; ++rr) {
            float partial[kIndexHeads];
#pragma unroll
            for (unsigned h = 0; h < kIndexHeads; ++h) {
                const float p0 = __fmul_rn(q[rr][h][0], k0), p1 = __fmul_rn(q[rr][h][1], k1);
                const float p2 = __fmul_rn(q[rr][h][2], k2), p3 = __fmul_rn(q[rr][h][3], k3);
                partial[h] = __fadd_rn(__fadd_rn(p0, p2), __fadd_rn(p1, p3));
            }
            scatter<kIndexHeads>(partial, lane);
            float total = 0.0F;
#pragma unroll
            for (unsigned h = 0; h < kIndexHeads; ++h)
                total = __fadd_rn(total, fmaxf(__shfl_sync(kFull, partial[0], static_cast<int>(8U * h)), 0.0F));
            if (lane == i) result[rr] = __fdiv_rn(total, scale);
        }
    }
#pragma unroll
    for (unsigned rr = 0; rr < R; ++rr)
        if (lane < last && base + lane < blocks[rr]) block_scores[static_cast<std::uint64_t>(r0 + rr) * max_blocks + base + lane] = result[rr];
}
__global__ __launch_bounds__(256) void block_score_kernel(const float* __restrict__ index_query, std::uint32_t rows,
    std::uint64_t first_position, const float* __restrict__ pooled_keys, std::uint32_t scored_blocks,
    float* __restrict__ block_scores, std::uint32_t max_blocks) {
    block_score_body(index_query, rows, first_position, pooled_keys, scored_blocks, block_scores, max_blocks, 0U);
}

// ------------------------------------------------------------------------------------------------------------------
// select_rows_kernel<Rows> (H:3756-3766) -> select_blocks_parallel_body (H:1472-1557), 1024 threads per row: radix
// top-512 over the ordered key (11/11/10 bits, +0 == -0, NaN / -inf never counted), ties to the lowest block index,
// output blocks ascending (4 tokens each), the first `zeros` slots tokens 0..3 repeated when fewer than 512 finite
// scores (block 0 then set to -inf), then the partial tail tokens. Integer histogram atomics are order-free.
constexpr unsigned kSelectThreads = 1024, kSelectWaves = 32;
__device__ __forceinline__ std::uint32_t select_key(float value) {
    auto bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}
__device__ __forceinline__ std::uint32_t select_scan(std::uint32_t value, std::uint32_t* waves, std::uint32_t& total) {
    const auto lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    for (unsigned d = 1; d < 32U; d <<= 1U) { const auto up = __shfl_up_sync(kFull, value, d); if (lane >= d) value += up; }
    if (lane == 31U) waves[wave] = value;
    __syncthreads();
    if (wave == 0) {
        auto sum = waves[lane];
        for (unsigned d = 1; d < 32U; d <<= 1U) { const auto up = __shfl_up_sync(kFull, sum, d); if (lane >= d) sum += up; }
        waves[lane] = sum;
    }
    __syncthreads();
    total = waves[kSelectWaves - 1U];
    value += wave ? waves[wave - 1U] : 0U;
    __syncthreads();
    return value;
}
__device__ __forceinline__ void select_kernel_body(float* __restrict__ all_scores, std::uint32_t max_blocks,
    std::uint64_t first_position, std::uint32_t* __restrict__ all_selected) {
    const auto r = blockIdx.x;
    const auto tid = threadIdx.x;
    const auto visible = static_cast<std::uint32_t>(first_position + r + 1U);
    const auto num_blocks = visible / kRatio;
    std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    if (num_blocks <= kTopK) {
        for (std::uint32_t index = tid; index < visible; index += kSelectThreads) selected[index] = index;
        return;
    }
    float* block_scores = all_scores + static_cast<std::uint64_t>(r) * max_blocks;
    __shared__ std::uint32_t histogram[2048];
    __shared__ std::uint32_t waves[kSelectWaves];
    __shared__ std::uint32_t found_digit, found_need, found_count;
    std::uint32_t prefix = 0, mask = 0, need = kTopK, zeros = 0;
    for (unsigned pass = 0; pass < 3U; ++pass) {
        const unsigned shift = pass == 0 ? 21U : pass == 1 ? 10U : 0U;
        const auto digits = pass == 2 ? 1024U : 2048U;
        for (auto i = tid; i < 2048U; i += kSelectThreads) histogram[i] = 0;
        __syncthreads();
        for (std::uint32_t base = 0; base < num_blocks; base += kSelectThreads) {
            const auto block = base + tid;
            const float value = block < num_blocks ? block_scores[block] : (-kInf);
            const auto key = select_key(value);
            const bool counted = value > (-kInf) && (key & mask) == prefix;
            if (counted) atomicAdd(&histogram[(key >> shift) & (digits - 1U)], 1U);
        }
        __syncthreads();
        const auto high = 2U * tid + 1U < digits ? histogram[2U * tid + 1U] : 0U;
        const auto low = 2U * tid < digits ? histogram[2U * tid] : 0U;
        std::uint32_t counted_total = 0;
        const auto inclusive = select_scan(high + low, waves, counted_total);
        const auto above = counted_total - inclusive;
        if (pass == 0 && counted_total <= need) { zeros = need - counted_total; need = counted_total; break; }
        if (above < need && need <= above + high) { found_digit = 2U * tid + 1U; found_need = need - above; found_count = high; }
        else if (above + high < need && need <= above + high + low) { found_digit = 2U * tid; found_need = need - above - high; found_count = low; }
        __syncthreads();
        prefix |= found_digit << shift; mask |= (digits - 1U) << shift; need = found_need;
        if (found_count == need) break;
    }
    for (auto token = tid; token < zeros * kRatio; token += kSelectThreads) selected[token] = token % kRatio;
    std::uint32_t greater_before = 0, equal_before = 0;
    for (std::uint32_t base = 0; base < num_blocks; base += 4U * kSelectThreads) {
        const auto first = base + 4U * tid;
        unsigned kind[4], greater = 0, equal = 0;
        for (unsigned j = 0; j < 4U; ++j) {
            const auto block = first + j;
            const float value = block < num_blocks ? block_scores[block] : (-kInf);
            const auto masked = select_key(value) & mask;
            kind[j] = !(value > (-kInf)) ? 0U : masked > prefix ? 1U : masked == prefix ? 2U : 0U;
            greater += kind[j] == 1U; equal += kind[j] == 2U;
        }
        std::uint32_t tile = 0;
        const auto packed = select_scan((greater << 16U) | equal, waves, tile);
        auto g = greater_before + (packed >> 16U) - greater, e = equal_before + (packed & 0xffffU) - equal;
        for (unsigned j = 0; j < 4U; ++j) {
            if (kind[j] == 1U || (kind[j] == 2U && e < need)) {
                const auto block = first + j, slot = zeros + g + min(e, need);
                for (std::uint32_t offset = 0; offset < kRatio; ++offset) selected[slot * kRatio + offset] = block * kRatio + offset;
                block_scores[block] = (-kInf);
            }
            g += kind[j] == 1U; e += kind[j] == 2U;
        }
        greater_before += tile >> 16U; equal_before += tile & 0xffffU;
    }
    if (zeros && tid == 0) block_scores[0] = (-kInf);
    const auto complete_tokens = num_blocks * kRatio, selected_complete = kTopK * kRatio;
    for (auto token = complete_tokens + tid; token < visible; token += kSelectThreads)
        selected[selected_complete + token - complete_tokens] = token;
}
__global__ __launch_bounds__(kSelectThreads) void select_kernel(float* __restrict__ all_scores, std::uint32_t max_blocks,
    std::uint64_t first_position, std::uint32_t* __restrict__ all_selected) {
    select_kernel_body(all_scores, max_blocks, first_position, all_selected);
}

// ------------------------------------------------------------------------------------------------------------------
// attention_score_rows_group256_kernel<64,12,16> (H:3981-4050, contract off; S:383040). One warp per (row, 12-head
// group = KV head, 16-slot run). Lane l holds q features l+32j (j=0..7) of the 12 heads; per slot k from the KV row;
// p_j = q*k; p[j] += p[j+4] (j<4); p0 += p2; p1 += p3; partial = p0 + p1; scatter<16> (heads 12..15 zero); lane with
// head h (lane & 1 == 0) writes partial / sqrtf(256) (= /16, exact). = the 256-feature tree strides 128..1.
constexpr unsigned kG = 12, kSlots = 16;
__global__ __launch_bounds__(256) void attention_score_kernel(const float* __restrict__ q_gate, std::uint32_t rows,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    float* __restrict__ all_scores, std::uint32_t slots) {
    const unsigned lane = threadIdx.x & 31U;
    const auto wave = blockIdx.x * 8U + threadIdx.x / 32U;
    const auto groups = (slots + kSlots - 1U) / kSlots;
    constexpr unsigned head_groups = kHeads / kG;
    const auto r = wave / (head_groups * groups);
    if (r >= rows) return;
    const auto count = selected_count(first_position + r);
    const auto head0 = wave / groups % head_groups * kG, first = wave % groups * kSlots;
    if (first >= count) return;
    const auto kv_head = head0 / (kHeads / kKvHeads);
    float qv[kG][8];
#pragma unroll
    for (unsigned h = 0; h < kG; ++h) {
        const float* q = q_gate + static_cast<std::uint64_t>(r) * kQGateWidth + static_cast<std::uint64_t>(head0 + h) * kHeadDim * 2ULL;
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) qv[h][j] = q[lane + 32U * j];
    }
    const float scale = __fsqrt_rn(static_cast<float>(kHeadDim));
    const auto last = min(first + kSlots, count);
    const std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    const auto tokens = lane < last - first ? selected[first + lane] : 0U;
    float* const scores = all_scores + static_cast<std::uint64_t>(r) * kHeads * kMaxSelected;
    const std::uint16_t* kbase = kv + static_cast<std::uint64_t>(kv_head) * kHeadDim + lane;
    std::uint16_t raw[8];
    const auto load = [&](unsigned index) {
        const auto token = __shfl_sync(kFull, tokens, static_cast<int>(index));
        const auto* row = kbase + static_cast<std::uint64_t>(token) * (kKvWidth * 2ULL);
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) raw[j] = row[32U * j];
    };
    load(0U);
    constexpr unsigned P = 16;
    for (auto slot = first; slot < last; ++slot) {
        float kf[8];
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) kf[j] = bf16f(raw[j]);
        if (slot + 1U < last) load(slot + 1U - first);
        float partial[P];
#pragma unroll
        for (unsigned h = 0; h < P; ++h) {
            if (h >= kG) { partial[h] = 0.0F; continue; }
            float p[8];
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) p[j] = __fmul_rn(qv[h][j], kf[j]);
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) p[j] = __fadd_rn(p[j], p[j + 4U]);
            p[0] = __fadd_rn(p[0], p[2]); p[1] = __fadd_rn(p[1], p[3]);
            partial[h] = __fadd_rn(p[0], p[1]);
        }
        scatter<P>(partial, lane);
        const auto h = scatter_head<P>(lane);
        if (h < kG && (lane & 1U) == 0U)
            scores[static_cast<std::uint64_t>(head0 + h) * kMaxSelected + slot] = __fdiv_rn(partial[0], scale);
    }
}

// Same scores, half-warp form (default): the HIP kernel's shuffles bound it on sm_120 (60 SHFL per slot for 12 heads).
// A half-warp of 16 lanes serves one slot, lane l holding features l + 16j (j = 0..15) of 6 heads; the in-lane pairs
// (j, j+8), (j, j+4), (j, j+2), (0, 1) are strides 128, 64, 32, 16 and the xor reduce-scatter over lanes 8, 4, 2, 1
// (scatter over 16 lanes, 8 values) the strides 8..1: the identical 256-feature tree (strides 128..1), now with 15 SHFL
// per (2 slots x 6 heads). Block = 4 warps = (2 head halves) x (2 slot pairs) of one (row, KV head, 32-slot run); both
// head halves read the same K rows (L1).
constexpr unsigned kHalfHeads = 6, kRun = 32;
template <unsigned P, unsigned Step = 1U> __device__ __forceinline__ void scatter16(float (&v)[P], unsigned lane) {
    // steps D = 8, 4, 2, 1 of scatter (Step 1..4); P <= 8 values over the 16 lanes of a half-warp
    if constexpr (Step < 5U) {
        constexpr unsigned D = 16U >> Step, Count = (P >> (Step - 1U)) > 0U ? (P >> (Step - 1U)) : 1U;
        if constexpr (Count > 1U) {
            constexpr unsigned H = Count / 2U;
            const bool upper = (lane & D) != 0U;
#pragma unroll
            for (unsigned j = 0; j < H; ++j) {
                const float low = __shfl_xor_sync(kFull, v[j], D), high = __shfl_xor_sync(kFull, v[j + H], D);
                v[j] = upper ? __fadd_rn(v[j + H], high) : __fadd_rn(v[j], low);
            }
        } else {
            v[0] = __fadd_rn(v[0], __shfl_xor_sync(kFull, v[0], D));
        }
        scatter16<P, Step + 1U>(v, lane);
    }
}
__device__ __forceinline__ void attention_score_half_kernel_body(const float* __restrict__ q_gate, std::uint64_t first_position,
    const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv, float* __restrict__ all_scores) {
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U, l = lane & 15U, half = lane >> 4U;
    const auto run = blockIdx.x, kvh = blockIdx.y, r = blockIdx.z;
    const auto count = selected_count(first_position + r);
    const auto first = run * kRun;
    if (first >= count) return;
    const unsigned hh = warp & 1U, sp = warp >> 1U;
    const auto head0 = kvh * kG + hh * kHalfHeads;
    float qv[kHalfHeads][16];
#pragma unroll
    for (unsigned h = 0; h < kHalfHeads; ++h) {
        const float* qp = q_gate + static_cast<std::uint64_t>(r) * kQGateWidth + static_cast<std::uint64_t>(head0 + h) * kHeadDim * 2ULL + l;
#pragma unroll
        for (unsigned j = 0; j < 16U; ++j) qv[h][j] = qp[16U * j];
    }
    const float scale = __fsqrt_rn(static_cast<float>(kHeadDim));
    const std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    float* const scores = all_scores + static_cast<std::uint64_t>(r) * kHeads * kMaxSelected;
    const std::uint16_t* kbase = kv + static_cast<std::uint64_t>(kvh) * kHeadDim + l;
    // slot of iteration i for this half-warp: first + 4i + 2sp + half (i = 0..7)
    const auto slot_of = [&](unsigned i) { return first + 4U * i + 2U * sp + half; };
    std::uint16_t raw[16];
    const auto load = [&](unsigned i) {
        const auto slot = slot_of(i);
        const auto token = slot < count ? selected[slot] : 0U;
        const auto* row = kbase + static_cast<std::uint64_t>(token) * (kKvWidth * 2ULL);
#pragma unroll
        for (unsigned j = 0; j < 16U; ++j) raw[j] = row[16U * j];
    };
    load(0U);
#pragma unroll 1
    for (unsigned i = 0; i < kRun / 4U; ++i) {
        if (first + 4U * i >= count) break;  // uniform per block
        float kf[16];
#pragma unroll
        for (unsigned j = 0; j < 16U; ++j) kf[j] = bf16f(raw[j]);
        if (i + 1U < kRun / 4U) load(i + 1U);
        float partial[8];
#pragma unroll
        for (unsigned h = 0; h < 8U; ++h) {
            if (h >= kHalfHeads) { partial[h] = 0.0F; continue; }
            float p[16];
#pragma unroll
            for (unsigned j = 0; j < 16U; ++j) p[j] = __fmul_rn(qv[h][j], kf[j]);
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) p[j] = __fadd_rn(p[j], p[j + 8U]);
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) p[j] = __fadd_rn(p[j], p[j + 4U]);
            p[0] = __fadd_rn(p[0], p[2]); p[1] = __fadd_rn(p[1], p[3]);
            partial[h] = __fadd_rn(p[0], p[1]);
        }
        scatter16<8>(partial, lane);
        const unsigned h = ((l & 8U) ? 4U : 0U) + ((l & 4U) ? 2U : 0U) + ((l & 2U) ? 1U : 0U);
        const auto slot = slot_of(i);
        if (h < kHalfHeads && (l & 1U) == 0U && slot < count)
            scores[static_cast<std::uint64_t>(head0 + h) * kMaxSelected + slot] = __fdiv_rn(partial[0], scale);
    }
}
__global__ __launch_bounds__(128) void attention_score_half_kernel(const float* __restrict__ q_gate, std::uint64_t first_position,
    const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv, float* __restrict__ all_scores) {
    attention_score_half_kernel_body(q_gate, first_position, all_selected, kv, all_scores);
}

// attention_softmax_rows_kernel<64> -> attention_softmax_body (H:1680-1786; S:408329-408725). Grid (24 heads, rows),
// 256 threads. max: fmaxf over slots l, l+256, ... then LDS fmaxf tree (order-free); local_sum += expf(s - max)
// ascending (S:408533-408553: v_sub, OCML expf, v_add); LDS tree 128..1 (S:408571-408655);
// weight = round_bf16(expf(s - max) / den) (S:408678-408712: correctly rounded division).
__device__ __forceinline__ void softmax_kernel_body(float* __restrict__ all_scores, std::uint64_t first_position, am::Tables tables) {
    __shared__ float scratch[256];
    const auto head = blockIdx.x, r = blockIdx.y, lane = threadIdx.x;
    const auto count = selected_count(first_position + r);
    float* scores = all_scores + (static_cast<std::uint64_t>(r) * kHeads + head) * kMaxSelected;
    float local_max = (-kInf);
    for (std::uint32_t i = lane; i < count; i += 256U) local_max = fmaxf(local_max, scores[i]);
    scratch[lane] = local_max;
    __syncthreads();
    for (std::uint32_t stride = 128U; stride > 0; stride >>= 1U) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float maximum = scratch[0];
    __syncthreads();
    float local_sum = 0.0F;
    // HIP recomputes expf(s - max) in the weight pass; the value is the same, so it is kept in place of the score.
    for (std::uint32_t i = lane; i < count; i += 256U) {
        const float e = am::amd_expf(tables, __fsub_rn(scores[i], maximum));
        scores[i] = e;
        local_sum = __fadd_rn(local_sum, e);
    }
    scratch[lane] = local_sum;
    __syncthreads();
    for (std::uint32_t stride = 128U; stride > 0; stride >>= 1U) {
        if (lane < stride) scratch[lane] = __fadd_rn(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float denominator = scratch[0];
    for (std::uint32_t i = lane; i < count; i += 256U)
        scores[i] = round_bf16(__fdiv_rn(scores[i], denominator));
}
__global__ __launch_bounds__(256) void softmax_kernel(float* __restrict__ all_scores, std::uint64_t first_position, am::Tables tables) {
    softmax_kernel_body(all_scores, first_position, tables);
}

// attention_output_rows_group_prefetch_kernel<64,12> (H:4102-4163; S:414951). Per (row, head, feature):
// sum = fma(w[slot], v[slot], sum) over all selected slots ascending from 0 (S: v_fmac/v_fma chain, `__builtin_fmaf`);
// attn = round_bf16(sum); gate = round_bf16(1 / (1 + expf(-g))) (S:415646-415678: OCML expf with the negation folded
// into the constants = expf(-g) exactly, v_add 1.0, correctly rounded division); out = round_bf16(attn * gate).
// CUDA: block per (row, KV head), 128 threads x 2 features x 12 heads (24 chains per thread); the weights of a chunk of
// slots are staged in shared memory as [slot][12] (3 x 16-byte broadcast loads per slot), tokens too.
constexpr unsigned kOutThreads = 128, kOutChunk = 256;
__device__ __forceinline__ void attention_output_kernel_body(const float* __restrict__ q_gate,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    __shared__ __align__(16) float ws[kOutChunk * kG];
    __shared__ std::uint32_t tok[kOutChunk];
    const auto kvh = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const auto head0 = kvh * kG;
    const auto count = selected_count(first_position + r);
    const std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    const float* scores = all_scores + (static_cast<std::uint64_t>(r) * kHeads + head0) * kMaxSelected;
    const std::uint16_t* vbase = kv + kKvWidth + kvh * kHeadDim + 2U * t;
    float s0[kG], s1[kG];
#pragma unroll
    for (unsigned h = 0; h < kG; ++h) { s0[h] = 0.0F; s1[h] = 0.0F; }
    for (std::uint32_t c0 = 0; c0 < count; c0 += kOutChunk) {
        const auto n = min(kOutChunk, count - c0);
        __syncthreads();
        for (unsigned i = t; i < n * kG; i += kOutThreads) {
            const unsigned slot = i % n, h = i / n;
            ws[slot * kG + h] = scores[static_cast<std::uint64_t>(h) * kMaxSelected + c0 + slot];
        }
        for (unsigned i = t; i < n; i += kOutThreads) tok[i] = selected[c0 + i];
        __syncthreads();
        constexpr unsigned B = 8;
        unsigned i = 0;
        for (; i + B <= n; i += B) {
            std::uint32_t raw[B];
#pragma unroll
            for (unsigned b = 0; b < B; ++b)
                raw[b] = *reinterpret_cast<const std::uint32_t*>(vbase + static_cast<std::uint64_t>(tok[i + b]) * (kKvWidth * 2ULL));
#pragma unroll
            for (unsigned b = 0; b < B; ++b) {
                const float v0 = __uint_as_float(raw[b] << 16U), v1 = __uint_as_float(raw[b] & 0xffff0000U);
                const float4* wp = reinterpret_cast<const float4*>(ws + (i + b) * kG);
                const float4 a = wp[0], bq = wp[1], cq = wp[2];
                const float w[kG] = {a.x, a.y, a.z, a.w, bq.x, bq.y, bq.z, bq.w, cq.x, cq.y, cq.z, cq.w};
#pragma unroll
                for (unsigned h = 0; h < kG; ++h) { s0[h] = __fmaf_rn(w[h], v0, s0[h]); s1[h] = __fmaf_rn(w[h], v1, s1[h]); }
            }
        }
        for (; i < n; ++i) {
            const auto raw = *reinterpret_cast<const std::uint32_t*>(vbase + static_cast<std::uint64_t>(tok[i]) * (kKvWidth * 2ULL));
            const float v0 = __uint_as_float(raw << 16U), v1 = __uint_as_float(raw & 0xffff0000U);
#pragma unroll
            for (unsigned h = 0; h < kG; ++h) {
                const float w = ws[i * kG + h];
                s0[h] = __fmaf_rn(w, v0, s0[h]); s1[h] = __fmaf_rn(w, v1, s1[h]);
            }
        }
    }
    const float* qg = q_gate + static_cast<std::uint64_t>(r) * kQGateWidth;
    float* out = output + static_cast<std::uint64_t>(r) * kAttnWidth;
#pragma unroll
    for (unsigned h = 0; h < kG; ++h) {
        const auto head = head0 + h;
#pragma unroll
        for (unsigned e = 0; e < 2U; ++e) {
            const auto f = 2U * t + e;
            const float attn = round_bf16(e ? s1[h] : s0[h]);
            const float g = qg[static_cast<std::uint64_t>(head) * kHeadDim * 2ULL + kHeadDim + f];
            const float gate = round_bf16(__fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(tables, -g))));
            out[static_cast<std::uint64_t>(head) * kHeadDim + f] = round_bf16(__fmul_rn(attn, gate));
        }
    }
}
__global__ __launch_bounds__(kOutThreads) void attention_output_kernel(const float* __restrict__ q_gate,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    attention_output_kernel_body(q_gate, first_position, all_selected, kv, all_scores, output, tables);
}

// Decode shape (few rows): one block per (query head, row), so a row runs 24 blocks instead of 2. Each thread keeps the
// same two chains sum = fma(w[slot], v[slot], sum), slots ascending from 0, and the same epilogue.
// STRIX_QRTX_DECODE_KERNELS=0 keeps the per-KV-head kernel (reference seam).
bool decode_rows_enabled() {
    static const bool on = [] { const char* e = std::getenv("STRIX_QRTX_DECODE_KERNELS"); return !(e && e[0] == '0'); }();
    return on;
}
__device__ __forceinline__ void attention_output_head_kernel_body(const float* __restrict__ q_gate,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    __shared__ float ws[kOutChunk];
    __shared__ std::uint32_t tok[kOutChunk];
    const auto head = blockIdx.x, r = blockIdx.y, t = threadIdx.x, kvh = head / kG;
    const auto count = selected_count(first_position + r);
    const std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    const float* scores = all_scores + (static_cast<std::uint64_t>(r) * kHeads + head) * kMaxSelected;
    const std::uint16_t* vbase = kv + kKvWidth + kvh * kHeadDim + 2U * t;
    float s0 = 0.0F, s1 = 0.0F;
    for (std::uint32_t c0 = 0; c0 < count; c0 += kOutChunk) {
        const auto n = min(kOutChunk, count - c0);
        __syncthreads();
        for (unsigned i = t; i < n; i += kOutThreads) { ws[i] = scores[c0 + i]; tok[i] = selected[c0 + i]; }
        __syncthreads();
        constexpr unsigned B = 8;
        unsigned i = 0;
        for (; i + B <= n; i += B) {
            std::uint32_t raw[B];
#pragma unroll
            for (unsigned b = 0; b < B; ++b)
                raw[b] = *reinterpret_cast<const std::uint32_t*>(vbase + static_cast<std::uint64_t>(tok[i + b]) * (kKvWidth * 2ULL));
#pragma unroll
            for (unsigned b = 0; b < B; ++b) {
                const float w = ws[i + b];
                s0 = __fmaf_rn(w, __uint_as_float(raw[b] << 16U), s0);
                s1 = __fmaf_rn(w, __uint_as_float(raw[b] & 0xffff0000U), s1);
            }
        }
        for (; i < n; ++i) {
            const auto raw = *reinterpret_cast<const std::uint32_t*>(vbase + static_cast<std::uint64_t>(tok[i]) * (kKvWidth * 2ULL));
            s0 = __fmaf_rn(ws[i], __uint_as_float(raw << 16U), s0);
            s1 = __fmaf_rn(ws[i], __uint_as_float(raw & 0xffff0000U), s1);
        }
    }
    const float* qg = q_gate + static_cast<std::uint64_t>(r) * kQGateWidth;
    float* out = output + static_cast<std::uint64_t>(r) * kAttnWidth;
#pragma unroll
    for (unsigned e = 0; e < 2U; ++e) {
        const auto f = 2U * t + e;
        const float attn = round_bf16(e ? s1 : s0);
        const float g = qg[static_cast<std::uint64_t>(head) * kHeadDim * 2ULL + kHeadDim + f];
        const float gate = round_bf16(__fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(tables, -g))));
        out[static_cast<std::uint64_t>(head) * kHeadDim + f] = round_bf16(__fmul_rn(attn, gate));
    }
}
__global__ __launch_bounds__(kOutThreads) void attention_output_head_kernel(const float* __restrict__ q_gate,
    std::uint64_t first_position, const std::uint32_t* __restrict__ all_selected, const std::uint16_t* __restrict__ kv,
    const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    attention_output_head_kernel_body(q_gate, first_position, all_selected, kv, all_scores, output, tables);
}

// ------------------------------------------------------------------------------------------------------------------
// Multi-segment launches (decode cohort windows, layer_core_multi): one launch per step for every session of a window.
// Every block (score kernel: every wave) belongs to one segment and runs the single-call body with that segment's
// pointers; its first_position argument is position[s] - first_row[s] (modulo 2^64), so first_position + r, with r the
// window row the block already uses for the row buffers, is the session's own position. Same code, same per-row and
// per-head arithmetic as one call per session.
__device__ __forceinline__ unsigned seg_of(const SegmentMap& m, unsigned row) {
    unsigned seg = 0;
    while (seg + 1U < m.n && row >= m.first_row[seg + 1U]) ++seg;
    return seg;
}
__device__ __forceinline__ std::uint64_t base_position(const SegmentMap& m, unsigned seg) { return m.position[seg] - m.first_row[seg]; }

__global__ void norm_rope_multi_kernel(float* __restrict__ values, std::uint64_t row_stride, std::uint32_t head_dim,
    std::uint32_t head_stride, const float* __restrict__ norm, SegmentMap map, const float2* __restrict__ cs, am::Tables tables) {
    norm_rope_kernel_body(values, row_stride, head_dim, head_stride, norm, base_position(map, seg_of(map, blockIdx.y)), cs, tables);
}
__global__ void store_multi_kernel(const float* __restrict__ ik, const float* __restrict__ k, const float* __restrict__ v, SegmentMap map) {
    const unsigned seg = seg_of(map, blockIdx.y);
    store_kernel_body(ik, k, v, base_position(map, seg), map.raw_index[seg], map.kv[seg]);
}
__global__ void block_pool_multi_kernel(SegmentMap map, const float* __restrict__ key_norm, const float2* __restrict__ cs,
    am::Tables tables) {
    unsigned seg = 0;
    while (seg + 1U < map.n && blockIdx.x >= map.pool_begin[seg + 1U]) ++seg;
    block_pool_kernel_body(map.raw_index[seg], map.pool_first[seg] - map.pool_begin[seg], key_norm, cs, tables, map.pooled[seg]);
}
__global__ __launch_bounds__(256) void block_score_multi_kernel(const float* __restrict__ index_query, SegmentMap map,
    float* __restrict__ block_scores, std::uint32_t max_blocks) {
    const auto wave = blockIdx.x * 8U + threadIdx.x / 32U;
    if (wave >= map.score_begin[map.n]) return;
    unsigned seg = 0;
    while (seg + 1U < map.n && wave >= map.score_begin[seg + 1U]) ++seg;
    const auto row = map.first_row[seg];
    block_score_body(index_query + static_cast<std::uint64_t>(row) * kIndexQWidth, map.first_row[seg + 1U] - row, map.position[seg],
        map.pooled[seg], map.scored[seg], block_scores + static_cast<std::uint64_t>(row) * max_blocks, max_blocks, map.score_begin[seg]);
}
__global__ __launch_bounds__(kSelectThreads) void select_multi_kernel(float* __restrict__ all_scores, std::uint32_t max_blocks,
    SegmentMap map, std::uint32_t* __restrict__ all_selected) {
    select_kernel_body(all_scores, max_blocks, base_position(map, seg_of(map, blockIdx.x)), all_selected);
}
__global__ __launch_bounds__(128) void attention_score_half_multi_kernel(const float* __restrict__ q_gate, SegmentMap map,
    const std::uint32_t* __restrict__ all_selected, float* __restrict__ all_scores) {
    const unsigned seg = seg_of(map, blockIdx.z);
    attention_score_half_kernel_body(q_gate, base_position(map, seg), all_selected, map.kv[seg], all_scores);
}
__global__ __launch_bounds__(256) void softmax_multi_kernel(float* __restrict__ all_scores, SegmentMap map, am::Tables tables) {
    softmax_kernel_body(all_scores, base_position(map, seg_of(map, blockIdx.y)), tables);
}
__global__ __launch_bounds__(kOutThreads) void attention_output_multi_kernel(const float* __restrict__ q_gate, SegmentMap map,
    const std::uint32_t* __restrict__ all_selected, const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    const unsigned seg = seg_of(map, blockIdx.y);
    attention_output_kernel_body(q_gate, base_position(map, seg), all_selected, map.kv[seg], all_scores, output, tables);
}
__global__ __launch_bounds__(kOutThreads) void attention_output_head_multi_kernel(const float* __restrict__ q_gate, SegmentMap map,
    const std::uint32_t* __restrict__ all_selected, const float* __restrict__ all_scores, float* __restrict__ output, am::Tables tables) {
    const unsigned seg = seg_of(map, blockIdx.y);
    attention_output_head_kernel_body(q_gate, base_position(map, seg), all_selected, map.kv[seg], all_scores, output, tables);
}

}  // namespace

// ------------------------------------------------------------------------------------------------------------------
std::size_t workspace_bytes(std::uint32_t chunk_rows, std::uint64_t max_positions) {
    const auto max_blocks = (max_positions + kRatio - 1U) / kRatio;
    return max_blocks * kIndexDim * 4ULL + static_cast<std::uint64_t>(chunk_rows) * max_blocks * 4ULL +
        static_cast<std::uint64_t>(chunk_rows) * kMaxSelected * 4ULL + static_cast<std::uint64_t>(chunk_rows) * kHeads * kMaxSelected * 4ULL;
}

Workspace allocate_workspace(std::uint32_t chunk_rows, std::uint64_t max_positions) {
    Workspace w;
    w.chunk_rows = chunk_rows;
    w.max_blocks = static_cast<std::uint32_t>((max_positions + kRatio - 1U) / kRatio);
    strix::cuda::require_vram(workspace_bytes(chunk_rows, max_positions), "qsa workspace");
    check(cudaMalloc(&w.pooled_keys, std::max<std::size_t>(1, static_cast<std::size_t>(w.max_blocks) * kIndexDim * 4U)), "alloc pooled");
    check(cudaMalloc(&w.block_scores, std::max<std::size_t>(1, static_cast<std::size_t>(chunk_rows) * w.max_blocks * 4U)), "alloc block scores");
    check(cudaMalloc(&w.selected, static_cast<std::size_t>(chunk_rows) * kMaxSelected * 4U), "alloc selected");
    check(cudaMalloc(&w.scores, static_cast<std::size_t>(chunk_rows) * kHeads * kMaxSelected * 4U), "alloc scores");
    return w;
}
void free_workspace(Workspace& w) {
    cudaFree(w.pooled_keys); cudaFree(w.block_scores); cudaFree(w.selected); cudaFree(w.scores);
    w = Workspace{};
}

void indexer_projection(const std::uint16_t* hidden, std::uint32_t rows, const LayerWeights& w, float* index_query,
    float* index_key, cudaStream_t stream) {
    if (!rows) return;
    static const bool attr = [] {
        check(cudaFuncSetAttribute(indexer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kIdxSmem)), "indexer smem attr");
        return true;
    }();
    (void)attr;
    const dim3 grid((rows + kIdxRows - 1U) / kIdxRows, (kIdxOutputs + kIdxSlice - 1U) / kIdxSlice);
    indexer_kernel<<<grid, 256, kIdxSmem, stream>>>(hidden, rows, w.index_q_proj, w.index_k_proj, index_query, index_key);
    check(cudaGetLastError(), "indexer launch");
}

void norm_rope(float* values, std::uint32_t rows, std::uint64_t row_stride, std::uint32_t heads, std::uint32_t head_dim,
    std::uint32_t head_stride, const float* norm, std::uint64_t first_position, const RopeTable* cs,
    const amdmath::Tables& tables, cudaStream_t stream) {
    if (!rows) return;
    if (head_dim != 128U && head_dim != 256U) throw std::invalid_argument("qwen_qsa norm_rope: head_dim 128 or 256");
    norm_rope_kernel<<<dim3(heads, rows), head_dim, 2U * head_dim * sizeof(float), stream>>>(values, row_stride, head_dim,
        head_stride, norm, first_position, cs, tables);
    check(cudaGetLastError(), "norm_rope launch");
}

void store_rows(const float* index_key, const float* k, const float* v, std::uint32_t rows, std::uint64_t first_position,
    const LayerState& state, cudaStream_t stream) {
    if (!rows) return;
    store_kernel<<<dim3(kKvWidth / 256U, rows), 256, 0, stream>>>(index_key, k, v, first_position, state.raw_index, state.kv);
    check(cudaGetLastError(), "store launch");
}

void block_pool(const std::uint16_t* raw_index, std::uint32_t first_block, std::uint32_t count, const float* index_k_norm,
    const RopeTable* cs, const amdmath::Tables& tables, float* pooled_keys, cudaStream_t stream) {
    if (!count) return;
    block_pool_kernel<<<count, kIndexDim, 0, stream>>>(raw_index, first_block, index_k_norm, cs, tables, pooled_keys);
    check(cudaGetLastError(), "block pool launch");
}

void block_scores(const float* index_query, std::uint32_t rows, std::uint64_t first_position, const float* pooled_keys,
    float* scores, std::uint32_t max_blocks, cudaStream_t stream) {
    if (!rows) return;
    const auto scored = static_cast<std::uint32_t>((first_position + rows) / kRatio);  // complete blocks of the last row
    if (scored <= kTopK) return;
    const auto waves = static_cast<std::uint64_t>((rows + kScoreRowsPerWave - 1U) / kScoreRowsPerWave) * ((scored + 31U) / 32U);
    block_score_kernel<<<static_cast<std::uint32_t>((waves + 7U) / 8U), 256, 0, stream>>>(index_query, rows, first_position,
        pooled_keys, scored, scores, max_blocks);
    check(cudaGetLastError(), "block score launch");
}

void select_rows(float* scores, std::uint32_t max_blocks, std::uint32_t rows, std::uint64_t first_position,
    std::uint32_t* selected, cudaStream_t stream) {
    if (!rows) return;
    select_kernel<<<rows, kSelectThreads, 0, stream>>>(scores, max_blocks, first_position, selected);
    check(cudaGetLastError(), "select launch");
}

void attention_scores(const float* q_gate, std::uint32_t rows, std::uint64_t first_position, const std::uint32_t* selected,
    const std::uint16_t* kv, float* scores, cudaStream_t stream) {
    if (!rows) return;
    std::uint32_t slots = 0;
    for (std::uint64_t p = first_position; p < first_position + rows && slots < kMaxSelected; ++p) slots = std::max(slots, selected_count(p));
    static const bool wave = [] { const char* v = std::getenv("STRIX_QSA_SCORE_WAVE"); return v && v[0] == '1'; }();
    if (wave) {
        const auto waves = static_cast<std::uint64_t>(rows) * (kHeads / kG) * ((slots + kSlots - 1U) / kSlots);
        attention_score_kernel<<<static_cast<std::uint32_t>((waves + 7U) / 8U), 256, 0, stream>>>(q_gate, rows, first_position,
            selected, kv, scores, slots);
    } else {
        attention_score_half_kernel<<<dim3((slots + kRun - 1U) / kRun, kKvHeads, rows), 128, 0, stream>>>(q_gate, first_position,
            selected, kv, scores);
    }
    check(cudaGetLastError(), "score launch");
}
void attention_softmax(std::uint32_t rows, std::uint64_t first_position, float* scores, const amdmath::Tables& tables, cudaStream_t stream) {
    if (!rows) return;
    softmax_kernel<<<dim3(kHeads, rows), 256, 0, stream>>>(scores, first_position, tables);
    check(cudaGetLastError(), "softmax launch");
}
void attention_values(const float* q_gate, std::uint32_t rows, std::uint64_t first_position, const std::uint32_t* selected,
    const std::uint16_t* kv, const float* scores, float* attention_output, const amdmath::Tables& tables, cudaStream_t stream) {
    if (!rows) return;
    if (rows <= 8U && decode_rows_enabled())
        attention_output_head_kernel<<<dim3(kHeads, rows), kOutThreads, 0, stream>>>(q_gate, first_position, selected, kv, scores,
            attention_output, tables);
    else
        attention_output_kernel<<<dim3(kKvHeads, rows), kOutThreads, 0, stream>>>(q_gate, first_position, selected, kv, scores,
            attention_output, tables);
    check(cudaGetLastError(), "output launch");
}
void attention(const float* q_gate, std::uint32_t rows, std::uint64_t first_position, const std::uint32_t* selected,
    const std::uint16_t* kv, float* scores, float* attention_output, const amdmath::Tables& tables, cudaStream_t stream) {
    attention_scores(q_gate, rows, first_position, selected, kv, scores, stream);
    attention_softmax(rows, first_position, scores, tables, stream);
    attention_values(q_gate, rows, first_position, selected, kv, scores, attention_output, tables, stream);
}

void layer_core(const RowBuffers& b, std::uint32_t rows, std::uint64_t first_position, const LayerWeights& w,
    const LayerState& state, const RopeTable* cs, const amdmath::Tables& tables, Workspace& ws, cudaStream_t stream) {
    if (!rows) return;
    if (first_position + rows > static_cast<std::uint64_t>(ws.max_blocks) * kRatio + kRatio)
        throw std::invalid_argument("qwen_qsa layer_core: workspace too small for the positions");
    indexer_projection(b.hidden, rows, w, b.index_query, b.index_key, stream);
    // H:4588-4599: index_q, q, k norms (each row only touches its own values; launch order irrelevant).
    norm_rope(b.index_query, rows, kIndexQWidth, kIndexHeads, kIndexDim, kIndexDim, w.index_q_norm, first_position, cs, tables, stream);
    norm_rope(b.q_gate, rows, kQGateWidth, kHeads, kHeadDim, kHeadDim * 2U, w.q_norm, first_position, cs, tables, stream);
    norm_rope(b.k, rows, kKvWidth, kKvHeads, kHeadDim, kHeadDim, w.k_norm, first_position, cs, tables, stream);
    // Every row's state before any row attends: rows read only positions <= their own, so storing the whole range first
    // equals the HIP group-by-group order (H:5148-5159).
    store_rows(b.index_key, b.k, b.v, rows, first_position, state, stream);
    const auto last_complete = static_cast<std::uint32_t>((first_position + rows) / kRatio);
    float* pooled = state.pooled ? state.pooled : ws.pooled_keys;
    if (last_complete > kTopK) {
        if (last_complete > ws.max_blocks) throw std::invalid_argument("qwen_qsa layer_core: max_blocks");
        std::uint32_t first = 0;
        if (state.pooled) first = std::min(*state.pooled_valid, static_cast<std::uint32_t>(first_position / kRatio));
        if (last_complete > first) block_pool(state.raw_index, first, last_complete - first, w.index_k_norm, cs, tables, pooled, stream);
        if (state.pooled) *state.pooled_valid = last_complete;
    }
    for (std::uint32_t c = 0; c < rows; c += ws.chunk_rows) {
        const auto n = std::min(ws.chunk_rows, rows - c);
        const auto p0 = first_position + c;
        block_scores(b.index_query + static_cast<std::uint64_t>(c) * kIndexQWidth, n, p0, pooled, ws.block_scores, ws.max_blocks, stream);
        select_rows(ws.block_scores, ws.max_blocks, n, p0, ws.selected, stream);
        attention(b.q_gate + static_cast<std::uint64_t>(c) * kQGateWidth, n, p0, ws.selected, state.kv, ws.scores,
            b.attention_output + static_cast<std::uint64_t>(c) * kAttnWidth, tables, stream);
    }
}

void layer_core_multi(const RowBuffers& b, const SegmentState* segs, std::uint32_t n, const LayerWeights& w, const RopeTable* cs,
    const amdmath::Tables& tables, Workspace& ws, cudaStream_t stream) {
    if (n == 0 || n > kMaxSegments) throw std::invalid_argument("qwen_qsa layer_core_multi: 1..8 segments");
    static const bool score_wave = [] { const char* v = std::getenv("STRIX_QSA_SCORE_WAVE"); return v && v[0] == '1'; }();
    // Every check and the whole segment map before the first launch; the host pooled-valid counts are written only after
    // the launches (a throw here leaves the sessions as they were).
    SegmentMap map{};
    map.n = n;
    std::uint32_t rows = 0, max_rows = 0, min_rows = 0xffffffffU;
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto& g = segs[i];
        if (g.rows == 0) throw std::invalid_argument("qwen_qsa layer_core_multi: empty segment");
        if (g.first_position + g.rows > static_cast<std::uint64_t>(ws.max_blocks) * kRatio + kRatio)
            throw std::invalid_argument("qwen_qsa layer_core: workspace too small for the positions");
        if (!g.state.pooled && n > 1) throw std::invalid_argument("qwen_qsa layer_core_multi: sessions need their own pooled cache");
        if (static_cast<std::uint32_t>((g.first_position + g.rows) / kRatio) > ws.max_blocks && (g.first_position + g.rows) / kRatio > kTopK)
            throw std::invalid_argument("qwen_qsa layer_core: max_blocks");
        map.first_row[i] = rows;
        map.position[i] = g.first_position;
        map.kv[i] = g.state.kv;
        map.raw_index[i] = g.state.raw_index;
        rows += g.rows;
        max_rows = std::max(max_rows, g.rows);
        min_rows = std::min(min_rows, g.rows);
    }
    map.first_row[n] = rows;
    // the output kernel per-segment layer_core chooses for each segment (head kernel: <= 8 rows with the decode kernels on);
    // the multi launch needs one choice for all, else per segment
    const bool head_all = max_rows <= 8U && decode_rows_enabled(), rows_all = !decode_rows_enabled() || min_rows > 8U;
    if (rows > ws.chunk_rows || score_wave || !(head_all || rows_all)) {  // outside the multi-launch shapes: one call per segment
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint64_t r = map.first_row[i];
            const RowBuffers sb{b.hidden + r * kHidden, b.q_gate + r * kQGateWidth, b.k + r * kKvWidth, b.v + r * kKvWidth,
                                b.index_query + r * kIndexQWidth, b.index_key + r * kIndexDim, b.attention_output + r * kAttnWidth};
            layer_core(sb, segs[i].rows, segs[i].first_position, w, segs[i].state, cs, tables, ws, stream);
        }
        return;
    }
    // pooled keys per session (host-side valid counts, as layer_core), block-score waves, attention slots
    std::uint32_t pool = 0, waves = 0, slots = 0;
    std::uint32_t valid_after[kMaxSegments] = {};
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto& g = segs[i];
        const auto last_complete = static_cast<std::uint32_t>((g.first_position + g.rows) / kRatio);
        map.pooled[i] = g.state.pooled ? g.state.pooled : ws.pooled_keys;
        map.pool_begin[i] = pool;
        map.pool_first[i] = 0;
        if (g.state.pooled) valid_after[i] = *g.state.pooled_valid;
        if (last_complete > kTopK) {
            std::uint32_t first = 0;
            if (g.state.pooled) first = std::min(*g.state.pooled_valid, static_cast<std::uint32_t>(g.first_position / kRatio));
            map.pool_first[i] = first;
            if (last_complete > first) pool += last_complete - first;
            valid_after[i] = last_complete;
        }
        // block scores: as block_scores() for this segment's rows
        map.score_begin[i] = waves;
        const auto scored = last_complete;  // (first_position + rows) / kRatio: complete blocks of the segment's last row
        map.scored[i] = scored;
        if (scored > kTopK) waves += static_cast<std::uint32_t>((g.rows + kScoreRowsPerWave - 1U) / kScoreRowsPerWave * ((scored + 31U) / 32U));
        for (std::uint64_t p = g.first_position; p < g.first_position + g.rows && slots < kMaxSelected; ++p) slots = std::max(slots, selected_count(p));
    }
    map.pool_begin[n] = pool;
    map.score_begin[n] = waves;
    indexer_projection(b.hidden, rows, w, b.index_query, b.index_key, stream);  // row-wise
    norm_rope_multi_kernel<<<dim3(kIndexHeads, rows), kIndexDim, 2U * kIndexDim * sizeof(float), stream>>>(b.index_query, kIndexQWidth,
        kIndexDim, kIndexDim, w.index_q_norm, map, cs, tables);
    norm_rope_multi_kernel<<<dim3(kHeads, rows), kHeadDim, 2U * kHeadDim * sizeof(float), stream>>>(b.q_gate, kQGateWidth, kHeadDim,
        kHeadDim * 2U, w.q_norm, map, cs, tables);
    norm_rope_multi_kernel<<<dim3(kKvHeads, rows), kHeadDim, 2U * kHeadDim * sizeof(float), stream>>>(b.k, kKvWidth, kHeadDim, kHeadDim,
        w.k_norm, map, cs, tables);
    check(cudaGetLastError(), "norm_rope multi launch");
    store_multi_kernel<<<dim3(kKvWidth / 256U, rows), 256, 0, stream>>>(b.index_key, b.k, b.v, map);
    check(cudaGetLastError(), "store multi launch");
    if (pool) {
        block_pool_multi_kernel<<<pool, kIndexDim, 0, stream>>>(map, w.index_k_norm, cs, tables);
        check(cudaGetLastError(), "block pool multi launch");
    }
    if (waves) {
        block_score_multi_kernel<<<(waves + 7U) / 8U, 256, 0, stream>>>(b.index_query, map, ws.block_scores, ws.max_blocks);
        check(cudaGetLastError(), "block score multi launch");
    }
    select_multi_kernel<<<rows, kSelectThreads, 0, stream>>>(ws.block_scores, ws.max_blocks, map, ws.selected);
    check(cudaGetLastError(), "select multi launch");
    attention_score_half_multi_kernel<<<dim3((slots + kRun - 1U) / kRun, kKvHeads, rows), 128, 0, stream>>>(b.q_gate, map, ws.selected,
        ws.scores);
    check(cudaGetLastError(), "score multi launch");
    softmax_multi_kernel<<<dim3(kHeads, rows), 256, 0, stream>>>(ws.scores, map, tables);
    check(cudaGetLastError(), "softmax multi launch");
    // keyed on the segment rows (as per-segment layer_core), not on the window total
    if (head_all)
        attention_output_head_multi_kernel<<<dim3(kHeads, rows), kOutThreads, 0, stream>>>(b.q_gate, map, ws.selected, ws.scores,
            b.attention_output, tables);
    else
        attention_output_multi_kernel<<<dim3(kKvHeads, rows), kOutThreads, 0, stream>>>(b.q_gate, map, ws.selected, ws.scores,
            b.attention_output, tables);
    check(cudaGetLastError(), "output multi launch");
    for (std::uint32_t i = 0; i < n; ++i)
        if (segs[i].state.pooled) *segs[i].state.pooled_valid = valid_after[i];
}

}  // namespace strix::cuda::qwen_qsa
