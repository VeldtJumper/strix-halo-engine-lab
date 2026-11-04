#include "internal.hpp"
#include "a8_device.hpp"
#include "attention_fast_decode.hpp"
#include "gemma4/kv/layout.hpp"
#include "gemma4/kv/kv8_device.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::gemma4::gpu {
namespace {
constexpr unsigned fd_dim = 512, fd_tile = 128, fd_threads = 256, fd_waves = fd_threads / 32U;
// Per (row, KV head, tile): maxima[group], denominators[group], values[group][fd_dim].
constexpr unsigned fd_partial = 2U * group + group * fd_dim;
constexpr unsigned fd_max_tiles = (max_positions + fd_tile - 1U) / fd_tile;
constexpr unsigned fd_merge_threads = 256, fd_merge_waves = fd_merge_threads / 32U, fd_slices = fd_dim / 32U;
static_assert(group == 4U && fd_tile % 64U == 0 && (fd_tile / 4U) % 4U == 0);

__device__ inline const float* fd_kv_at(const KvRead& r, std::uint64_t p, bool value, unsigned* failure) {
    if (p >= r.frontier) {
        if (p - r.frontier >= r.candidate_rows) {
            atomicOr(failure, 2U);
            return nullptr;
        }
        return (value ? r.overlay_v : r.overlay_k) + std::size_t(p - r.frontier) * r.width;
    }
    const auto slot = r.global ? p : p % r.slots;
    if (slot >= r.slots || r.tags[slot] != p) {
        atomicOr(failure, 2U);
        return nullptr;
    }
    return (value ? r.v : r.k) + std::size_t(slot) * r.width;
}
__device__ inline float half_value(unsigned bits16) { return float(__builtin_bit_cast(_Float16, (unsigned short)bits16)); }
// The mirror's conversion: RNE half; an overflowing half sets failure bit 4 as the mirror sync does.
__device__ inline float rounded(float x, unsigned* failure) {
    const auto h = static_cast<_Float16>(x);
    if ((__builtin_bit_cast(unsigned short, h) & 0x7c00U) == 0x7c00U)
        atomicOr(failure, 4U);
    return float(h);
}
// Count consecutive elements [at, at + Count) of the K (or V) row of position p, as FP32 values of the identity's
// reduced operands. Format f16: the FP16 halves (the mirror's below `mirrored`, else the FP32 row - committed or
// overlay - rounded RNE). Format kv8: d * code of the KV8 groups (kv8_device.hpp); Count 16 = one lane's own group,
// Count 8 = half a group held by two consecutive lanes (lane bit 0), whose amax they combine by one xor shuffle
// (every lane of the wave calls with the same p, so the shuffle is uniform).
template <unsigned Count, MirrorFormat Format>
__device__ inline void load_kv(const KvRead& r, const Fp16MirrorRead& m, std::uint64_t p, bool value, unsigned at,
                               float (&out)[Count], unsigned* failure) {
    static_assert(Count == 8U || Count == 16U);
    if (p < m.mirrored) {
        if constexpr (Format == MirrorFormat::f16) {
            const auto* src = static_cast<const std::uint16_t*>(value ? m.v : m.k) + std::size_t(p) * m.width + at;
            if constexpr (Count == 16U) {
#pragma unroll
                for (unsigned c = 0; c < 2; ++c) {
                    const uint4 u = *reinterpret_cast<const uint4*>(src + 8U * c);
                    const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                    for (unsigned i = 0; i < 4; ++i) {
                        out[8U * c + 2U * i] = half_value(w[i] & 0xffffU);
                        out[8U * c + 2U * i + 1U] = half_value(w[i] >> 16U);
                    }
                }
            } else {
                const uint4 u = *reinterpret_cast<const uint4*>(src);
                const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    out[2U * i] = half_value(w[i] & 0xffffU);
                    out[2U * i + 1U] = half_value(w[i] >> 16U);
                }
            }
        } else {
            const auto* codes = static_cast<const std::int8_t*>(value ? m.v : m.k) + std::size_t(p) * m.width + at;
            const float d = kv8_scale_value((value ? m.v_scale : m.k_scale)[std::size_t(p) * (m.width / kv8_group) +
                                                                            at / kv8_group]);
            unsigned w[Count / 4U];
            if constexpr (Count == 16U) {
                const uint4 u = *reinterpret_cast<const uint4*>(codes);
                w[0] = u.x; w[1] = u.y; w[2] = u.z; w[3] = u.w;
            } else {
                const uint2 u = *reinterpret_cast<const uint2*>(codes);
                w[0] = u.x; w[1] = u.y;
            }
#pragma unroll
            for (unsigned c = 0; c < Count / 4U; ++c)
#pragma unroll
                for (unsigned i = 0; i < 4; ++i)
                    out[4U * c + i] = d * float(int(std::int8_t((w[c] >> (8U * i)) & 0xffU)));
        }
        return;
    }
    const float* row = fd_kv_at(r, p, value, failure);
    if (!row) {
#pragma unroll
        for (unsigned i = 0; i < Count; ++i)
            out[i] = 0.0F;
        return;
    }
#pragma unroll
    for (unsigned c = 0; c < Count / 4U; ++c) {
        const float4 f = *reinterpret_cast<const float4*>(row + at + 4U * c);
        out[4U * c] = f.x;
        out[4U * c + 1U] = f.y;
        out[4U * c + 2U] = f.z;
        out[4U * c + 3U] = f.w;
    }
    if constexpr (Format == MirrorFormat::f16) {
#pragma unroll
        for (unsigned i = 0; i < Count; ++i)
            out[i] = rounded(out[i], failure);
    } else {
        float amax = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < Count; ++i) {
            kv8_finite(out[i], failure);
            const float a = kv8_abs(out[i]);
            amax = a > amax ? a : amax;
        }
        if constexpr (Count == 8U) {
            const float other = __shfl_xor(amax, 1, 32);
            amax = other > amax ? other : amax;
        }
        const float d = kv8_scale_value(kv8_scale_bits(amax, failure));
#pragma unroll
        for (unsigned i = 0; i < Count; ++i)
            out[i] = d * float(kv8_code(out[i], d));
    }
}
// Raw mirror operands of one lane, fetched ahead of their use (loads in flight): Count 16 = 32 B of halves or
// 16 B of codes + scale, Count 8 = 16 B of halves or 8 B of codes + scale. Decoding gives exactly load_kv's values.
template <unsigned Count, MirrorFormat Format>
struct RawKv {
    uint4 a, b;
    unsigned short scale;
};
template <unsigned Count, MirrorFormat Format>
__device__ inline RawKv<Count, Format> fetch_raw(const Fp16MirrorRead& m, std::uint64_t p, bool value, unsigned at) {
    RawKv<Count, Format> raw{};
    if constexpr (Format == MirrorFormat::f16) {
        const auto* src = static_cast<const std::uint16_t*>(value ? m.v : m.k) + std::size_t(p) * m.width + at;
        raw.a = *reinterpret_cast<const uint4*>(src);
        if constexpr (Count == 16U)
            raw.b = *reinterpret_cast<const uint4*>(src + 8U);
    } else {
        const auto* codes = static_cast<const std::int8_t*>(value ? m.v : m.k) + std::size_t(p) * m.width + at;
        if constexpr (Count == 16U)
            raw.a = *reinterpret_cast<const uint4*>(codes);
        else {
            const uint2 u = *reinterpret_cast<const uint2*>(codes);
            raw.a.x = u.x;
            raw.a.y = u.y;
        }
        raw.scale = (value ? m.v_scale : m.k_scale)[std::size_t(p) * (m.width / kv8_group) + at / kv8_group];
    }
    return raw;
}
template <unsigned Count, MirrorFormat Format>
__device__ inline void decode_raw(const RawKv<Count, Format>& raw, float (&out)[Count]) {
    if constexpr (Format == MirrorFormat::f16) {
        const unsigned w[8] = {raw.a.x, raw.a.y, raw.a.z, raw.a.w, raw.b.x, raw.b.y, raw.b.z, raw.b.w};
#pragma unroll
        for (unsigned i = 0; i < Count / 2U; ++i) {
            out[2U * i] = half_value(w[i] & 0xffffU);
            out[2U * i + 1U] = half_value(w[i] >> 16U);
        }
    } else {
        const float d = kv8_scale_value(raw.scale);
        const unsigned w[4] = {raw.a.x, raw.a.y, raw.a.z, raw.a.w};
#pragma unroll
        for (unsigned c = 0; c < Count / 4U; ++c)
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                out[4U * c + i] = d * float(int(std::int8_t((w[c] >> (8U * i)) & 0xffU)));
    }
}
// The normed and rotated query, exactly as the exact chunk kernels form it (norm, then rotation): one wave per
// (row, query head), written once per layer instead of once per tile block.
__global__ void __launch_bounds__(32) fd_query_kernel(const float* q, LinearView q_norm, const std::uint64_t* positions,
                                                      const float* table, unsigned heads_count, float* queries,
                                                      bool parallel, bool bounded, bool intervals, unsigned* failure) {
    const unsigned rh = blockIdx.x, row = rh / heads_count, lane = threadIdx.x;
    float v[fd_dim / 32U];
#pragma unroll
    for (unsigned j = 0; j < fd_dim / 32U; ++j)
        v[j] = q[std::size_t(rh) * fd_dim + lane + 32U * j];
    register_head_values<fd_dim>(v, q_norm, table + std::size_t(positions[row]) * fd_dim, parallel, bounded,
                                 intervals);
#pragma unroll
    for (unsigned j = 0; j < fd_dim / 32U; ++j) {
        queries[std::size_t(rh) * fd_dim + lane + 32U * j] = v[j];
        finite_flag(v[j], failure);
    }
}
// One block per (row, KV head, 128-key tile of absolute positions) serves the group's four query heads.
// Phase 1: wave w scores keys w, w + 8, ... (lane l holds features [16l, 16l + 16) of the key and of each query;
//          lane-ordered products, then one fixed butterfly that reduces the four heads together: xor 16 splits heads
//          {0,1} / {2,3}, xor 8 splits each pair, xor 4, 2, 1 finish; lane 8h holds head h's score).
//          Phase 2: wave h forms head h's tile maximum, exponentials and denominator (lane-strided, then the wave
//          tree). Phase 3: thread t sums features [8(t mod 64), +8) of the values over keys [32(t / 64), +32) in key
//          order; the four key quarters' sums are added in quarter order.
template <MirrorFormat Format>
__global__ void __launch_bounds__(fd_threads) fd_split_kernel(const float* queries, const KvRead* reads,
                                                              const Fp16MirrorRead* mirrors, unsigned heads_count,
                                                              float* partials, unsigned* failure) {
    const unsigned kv_count = heads_count / group, row = blockIdx.x / kv_count, kvh = blockIdx.x % kv_count;
    const unsigned tile = blockIdx.y, tiles = gridDim.y;
    const auto r = reads[row];
    const auto m = mirrors[row];
    const std::uint64_t begin = std::uint64_t(tile) * fd_tile;
    if (begin > r.position)
        return; // the merge reads only the tiles of keys [0, position]
    const unsigned count = unsigned(min(begin + fd_tile, r.position + 1U) - begin);
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    __shared__ float shared_values[3U * group * fd_dim]; // quarters 1..3's value sums
    __shared__ float score[group][fd_tile];
    const unsigned kv_offset = kvh * fd_dim;
    {
        constexpr unsigned batch = 4; // keys of one wave in flight
        float query[group][16];
        const float* qrow = queries + (std::size_t(row) * heads_count + kvh * group) * fd_dim + 16U * lane;
#pragma unroll
        for (unsigned h = 0; h < group; ++h)
#pragma unroll
            for (unsigned i = 0; i < 16; ++i)
                query[h][i] = qrow[h * fd_dim + i];
        for (unsigned j0 = wave; j0 < count; j0 += batch * fd_waves) {
            RawKv<16, Format> raw[batch];
#pragma unroll
            for (unsigned k = 0; k < batch; ++k) {
                const auto p = begin + j0 + k * fd_waves;
                if (j0 + k * fd_waves < count && p < m.mirrored)
                    raw[k] = fetch_raw<16, Format>(m, p, false, kv_offset + 16U * lane);
            }
#pragma unroll
            for (unsigned k = 0; k < batch; ++k) {
                const unsigned j = j0 + k * fd_waves;
                if (j >= count)
                    break;
                float key[16];
                if (begin + j < m.mirrored)
                    decode_raw<16, Format>(raw[k], key);
                else
                    load_kv<16, Format>(r, m, begin + j, false, kv_offset + 16U * lane, key, failure);
                float sum[group];
#pragma unroll
                for (unsigned h = 0; h < group; ++h) {
                    sum[h] = 0.0F;
#pragma unroll
                    for (unsigned i = 0; i < 16; ++i)
                            sum[h] = sum[h] + query[h][i] * key[i];
                }
                // Lanes with bit 4 keep heads 2, 3, the others 0, 1; then bit 3 picks one of the pair.
                const bool upper = (lane & 16U) != 0, odd = (lane & 8U) != 0;
                const float give0 = upper ? sum[0] : sum[2], give1 = upper ? sum[1] : sum[3];
                const float got0 = __shfl_xor(give0, 16, 32), got1 = __shfl_xor(give1, 16, 32);
                const float pair0 = (upper ? sum[2] : sum[0]) + got0, pair1 = (upper ? sum[3] : sum[1]) + got1;
                const float got = __shfl_xor(odd ? pair0 : pair1, 8, 32);
                float total = (odd ? pair1 : pair0) + got;
#pragma unroll
                for (int d = 4; d != 0; d >>= 1)
                    total = total + __shfl_xor(total, d, 32);
                if ((lane & 7U) == 0) {
                    finite_flag(total, failure);
                    score[(upper ? 2U : 0U) + (odd ? 1U : 0U)][j] = total;
                }
            }
        }
    }
    __syncthreads();
    float* partial = partials + (std::size_t(blockIdx.x) * tiles + tile) * fd_partial;
    if (wave < group) {
        float maximum = bit_cast<float>(0xff800000U);
        for (unsigned j = lane; j < count; j += 32U)
            maximum = score[wave][j] > maximum ? score[wave][j] : maximum;
#pragma unroll
        for (unsigned d = 16; d != 0; d >>= 1U) {
            const float other = __shfl_xor(maximum, int(d), 32);
            maximum = other > maximum ? other : maximum;
        }
        float denominator = 0.0F;
        for (unsigned j = lane; j < count; j += 32U) {
            const float e = __expf(score[wave][j] - maximum);
            score[wave][j] = e;
            denominator = denominator + e;
        }
        denominator = wave_sum(denominator);
        if (lane == 0) {
            partial[wave] = maximum;
            partial[group + wave] = denominator;
        }
    }
    __syncthreads();
    const unsigned feature = 8U * (threadIdx.x % 64U), quarter = threadIdx.x / 64U;
    const unsigned first = quarter * (fd_tile / 4U), last = min(first + fd_tile / 4U, count);
    float acc[group][8];
#pragma unroll
    for (unsigned h = 0; h < group; ++h)
#pragma unroll
        for (unsigned i = 0; i < 8; ++i)
            acc[h][i] = 0.0F;
    unsigned j = first;
    for (; j + 8U <= last && begin + j + 8U <= m.mirrored; j += 8U) {
        RawKv<8, Format> raw[8];
#pragma unroll
        for (unsigned k = 0; k < 8; ++k)
            raw[k] = fetch_raw<8, Format>(m, begin + j + k, true, kv_offset + feature);
#pragma unroll
        for (unsigned k = 0; k < 8; ++k) {
            float v[8];
            decode_raw<8, Format>(raw[k], v);
#pragma unroll
            for (unsigned h = 0; h < group; ++h) {
                const float e = score[h][j + k];
#pragma unroll
                for (unsigned i = 0; i < 8; ++i)
                    acc[h][i] = acc[h][i] + e * v[i];
            }
        }
    }
    for (; j < last; ++j) {
        float v[8];
        load_kv<8, Format>(r, m, begin + j, true, kv_offset + feature, v, failure);
#pragma unroll
        for (unsigned h = 0; h < group; ++h) {
            const float e = score[h][j];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
                acc[h][i] = acc[h][i] + e * v[i];
        }
    }
    if (quarter != 0U)
#pragma unroll
        for (unsigned h = 0; h < group; ++h)
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
                shared_values[((quarter - 1U) * group + h) * fd_dim + feature + i] = acc[h][i];
    __syncthreads();
    if (quarter == 0U)
#pragma unroll
        for (unsigned h = 0; h < group; ++h)
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                float value = acc[h][i];
#pragma unroll
                for (unsigned u = 0; u < 3; ++u)
                    value = value + shared_values[(u * group + h) * fd_dim + feature + i];
                finite_flag(value, failure);
                partial[2U * group + h * fd_dim + feature + i] = value;
            }
}
// Ordered merge, one block per (row, query head, 32-feature slice): M = max over the row's tiles (thread-strided,
// wave tree, waves in order), w_t = exp(m_t - M), denominator = sum w_t l_t (thread-strided in tile order, wave tree,
// waves in order). Wave w sums w_t O_t over tiles w, w + 8, ... in order (lane = feature); the eight wave sums are
// added in wave order; value = numerator / denominator. Lane l of wave 0 holds feature 32 * slice + l (A8G32 group).
template <bool Quant>
__global__ void __launch_bounds__(fd_merge_threads) fd_merge_kernel(const KvRead* reads, unsigned heads_count,
                                                                    unsigned tiles, const float* partials, float* out,
                                                                    unsigned* failure, A8Rows quant) {
    const unsigned rh = blockIdx.x, row = rh / heads_count, head = rh % heads_count;
    const unsigned kv_count = heads_count / group, kvh = head / group, index = head % group;
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const unsigned feature = blockIdx.y * 32U + lane;
    const auto r = reads[row];
    const unsigned used = unsigned(r.position / fd_tile) + 1U;
    const float* base = partials + std::size_t(row * kv_count + kvh) * tiles * fd_partial;
    __shared__ float weight[fd_max_tiles];
    __shared__ float wave_values[fd_merge_waves];
    __shared__ float wave_sums[fd_merge_waves][32];
    float maximum = bit_cast<float>(0xff800000U);
    for (unsigned t = threadIdx.x; t < used; t += fd_merge_threads) {
        const float a = base[std::size_t(t) * fd_partial + index];
        maximum = a > maximum ? a : maximum;
    }
#pragma unroll
    for (unsigned d = 16; d != 0; d >>= 1U) {
        const float other = __shfl_xor(maximum, int(d), 32);
        maximum = other > maximum ? other : maximum;
    }
    if (lane == 0)
        wave_values[wave] = maximum;
    __syncthreads();
    maximum = wave_values[0];
#pragma unroll
    for (unsigned w = 1; w < fd_merge_waves; ++w)
        maximum = wave_values[w] > maximum ? wave_values[w] : maximum;
    __syncthreads();
    float denominator = 0.0F;
    for (unsigned t = threadIdx.x; t < used; t += fd_merge_threads) {
        const float w = __expf(base[std::size_t(t) * fd_partial + index] - maximum);
        weight[t] = w;
        denominator = denominator + w * base[std::size_t(t) * fd_partial + group + index];
    }
    denominator = wave_sum(denominator);
    if (lane == 0)
        wave_values[wave] = denominator;
    __syncthreads();
    denominator = wave_values[0];
#pragma unroll
    for (unsigned w = 1; w < fd_merge_waves; ++w)
        denominator = denominator + wave_values[w];
    const float* values = base + 2U * group + index * fd_dim + feature;
    float numerator = 0.0F;
    unsigned t = wave;
    for (; t + 7U * fd_merge_waves < used; t += 8U * fd_merge_waves) {
        float o[8];
#pragma unroll
        for (unsigned k = 0; k < 8; ++k)
            o[k] = values[std::size_t(t + k * fd_merge_waves) * fd_partial];
#pragma unroll
        for (unsigned k = 0; k < 8; ++k)
            numerator = numerator + weight[t + k * fd_merge_waves] * o[k];
    }
    for (; t < used; t += fd_merge_waves)
        numerator = numerator + weight[t] * values[std::size_t(t) * fd_partial];
    wave_sums[wave][lane] = numerator;
    __syncthreads();
    if (wave != 0)
        return;
    numerator = wave_sums[0][lane];
#pragma unroll
    for (unsigned w = 1; w < fd_merge_waves; ++w)
        numerator = numerator + wave_sums[w][lane];
    const float value = numerator / denominator;
    out[std::size_t(rh) * fd_dim + feature] = value;
    finite_flag(value, failure);
    if constexpr (Quant)
        a8_quantize_lane(value, quant, row, head * fd_dim + feature);
}
} // namespace
const char* fast_decode_attention_name(MirrorFormat format) {
    return format == MirrorFormat::f16 ? fast_decode_attention_f16 : format == MirrorFormat::kv8 ? fast_decode_attention_kv8
                                                                                                : "exact";
}
MirrorFormat fast_decode_attention_format() {
    static const MirrorFormat format = [] {
        const auto* p = std::getenv("STRIX_G4_DECODE_ATTN");
        if (!p || !*p || std::string_view(p) == "exact" || std::string_view(p) == "0" || std::string_view(p) == "off")
            return MirrorFormat::none;
        if (std::string_view(p) == fast_decode_attention_f16)
            return MirrorFormat::f16;
        if (std::string_view(p) == fast_decode_attention_kv8)
            return MirrorFormat::kv8;
        throw std::invalid_argument(std::string("STRIX_G4_DECODE_ATTN must be exact, ") + fast_decode_attention_f16 +
                                    " or " + fast_decode_attention_kv8);
    }();
    return format;
}
// Scratch: the rows' normed queries (rows x heads x 512 floats), then the tile partials.
std::size_t fast_decode_attention_scratch_bytes(std::uint32_t rows, std::uint32_t stride) {
    const std::size_t tiles = (std::size_t(stride) + fd_tile - 1U) / fd_tile;
    return std::size_t(rows) * (heads * fd_dim + kv_heads * tiles * fd_partial) * sizeof(float);
}
bool fast_decode_attention(const float* q, LinearView q_norm, const std::uint64_t* positions, const float* table,
                           const KvRead* reads, const Fp16MirrorRead* mirrors, std::uint32_t rows,
                           std::uint32_t head_count, std::uint32_t dim, std::uint32_t stride, float* out,
                           float* scratch, std::size_t scratch_bytes, std::uint32_t* failure, Queue queue,
                           bool parallel, bool bounded, bool intervals, MirrorFormat format, A8Rows quant) {
    valid_rows(rows);
    if (format == MirrorFormat::none || dim != fd_dim || !head_count || head_count % group || head_count / group > kv_heads || !q_norm.data ||
        !mirrors || !scratch || !stride || stride > max_positions)
        return false;
    const unsigned tiles = (stride + fd_tile - 1U) / fd_tile;
    const std::size_t query_floats = std::size_t(rows) * head_count * fd_dim;
    if ((query_floats + std::size_t(rows) * (head_count / group) * tiles * fd_partial) * sizeof(float) > scratch_bytes)
        return false;
    float* queries = scratch;
    float* partials = scratch + query_floats;
    const bool quantize = quant.codes && quant.width == head_count * dim && dim % 256U == 0;
    const auto stream = static_cast<hipStream_t>(queue);
    hipLaunchKernelGGL(fd_query_kernel, dim3(rows * head_count), dim3(32), 0, stream, q, q_norm, positions, table,
                       head_count, queries, parallel, bounded, intervals, failure);
    launch_checked();
    if (format == MirrorFormat::f16)
        hipLaunchKernelGGL(fd_split_kernel<MirrorFormat::f16>, dim3(rows * (head_count / group), tiles),
                           dim3(fd_threads), 0, stream, queries, reads, mirrors, head_count, partials, failure);
    else
        hipLaunchKernelGGL(fd_split_kernel<MirrorFormat::kv8>, dim3(rows * (head_count / group), tiles),
                           dim3(fd_threads), 0, stream, queries, reads, mirrors, head_count, partials, failure);
    launch_checked();
    if (quantize)
        hipLaunchKernelGGL(fd_merge_kernel<true>, dim3(rows * head_count, fd_slices), dim3(fd_merge_threads), 0,
                           stream, reads, head_count, tiles, partials, out, failure, quant);
    else
        hipLaunchKernelGGL(fd_merge_kernel<false>, dim3(rows * head_count, fd_slices), dim3(fd_merge_threads), 0,
                           stream, reads, head_count, tiles, partials, out, failure, quant);
    launch_checked();
    return true;
}
} // namespace strix::gemma4::gpu
