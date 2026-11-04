#include "internal.hpp"
#include "a8_device.hpp"
#include "gemma4/kv/storage.hpp"
#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <type_traits>

namespace strix::gemma4::gpu {
namespace {
__device__ const float* kv_at(const KvRead& r, std::uint64_t p, bool value, unsigned* failure) {
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
template <bool Prefetch>
__global__ void scores_kernel(const float* q, const KvRead* reads, unsigned heads_count, unsigned dim, unsigned stride,
                              float* scores, float* maxima, unsigned* failure) {
    const unsigned rh = blockIdx.y, row = rh / heads_count, head = rh % heads_count;
    const auto r = reads[row];
    const auto range = visible(r.position, r.global, r.window);
    const std::uint64_t begin = std::uint64_t(blockIdx.x) * attention_chunk;
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    auto* out = scores + std::size_t(rh) * stride;
    float query[16];
    if constexpr (Prefetch) {
        if (begin > range.last || begin + attention_chunk <= range.first) {
            if (threadIdx.x == 0)
                maxima[std::size_t(rh) * gridDim.x + blockIdx.x] = bit_cast<float>(0xff800000U);
            return;
        }
#pragma unroll
        for (unsigned j = 0; j < 16; ++j)
            if (lane + 32U * j < dim)
                query[j] = q[std::size_t(rh) * dim + lane + 32U * j];
    }
    for (unsigned j = wave; j < attention_chunk; j += 4U) {
        const auto p = begin + j;
        if (p < range.first || p > range.last)
            continue;
        const auto* k = kv_at(r, p, false, failure);
        float sum = 0.0F;
        if (k) {
            if constexpr (Prefetch) {
                float key[16];
#pragma unroll
                for (unsigned component = 0; component < 16; ++component)
                    if (lane + 32U * component < dim)
                        key[component] = k[(head / group) * dim + lane + 32U * component];
#pragma unroll
                for (unsigned component = 0; component < 16; ++component)
                    if (lane + 32U * component < dim)
                        sum = sum + query[component] * key[component];
            } else {
                for (unsigned i = lane; i < dim; i += 32U)
                    sum = sum + q[std::size_t(rh) * dim + i] * k[(head / group) * dim + i];
            }
        }
        sum = wave_sum(sum);
        if (lane == 0) {
            finite_flag(sum, failure);
            out[p] = sum;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = bit_cast<float>(0xff800000U);
        for (unsigned j = 0; j < attention_chunk; ++j) {
            const auto p = begin + j;
            if (p >= range.first && p <= range.last)
                m = out[p] > m ? out[p] : m;
        }
        maxima[std::size_t(rh) * gridDim.x + blockIdx.x] = m;
    }
}
__global__ void maximum_kernel(const KvRead* reads, unsigned heads_count, unsigned chunks, float* maxima) {
    if (threadIdx.x)
        return;
    const auto r = reads[blockIdx.x / heads_count];
    const auto range = visible(r.position, r.global, r.window);
    float m = bit_cast<float>(0xff800000U);
    for (auto c = range.first / attention_chunk; c <= range.last / attention_chunk; ++c) {
        const float a = maxima[std::size_t(blockIdx.x) * chunks + c];
        m = a > m ? a : m;
    }
    // An extra entry per row/head avoids overwriting a chunk before another launch consumes it.
    maxima[std::size_t(gridDim.x) * chunks + blockIdx.x] = m;
}
template <bool Prefetch, bool Final = false>
__global__ void partial_kernel(const KvRead* reads, unsigned heads_count, unsigned dim, unsigned stride,
                               const float* scores, const float* maxima, float* partials, unsigned* failure) {
    const unsigned rh = blockIdx.y, head = rh % heads_count;
    const auto r = reads[rh / heads_count];
    const auto range = visible(r.position, r.global, r.window);
    const auto first = std::uint64_t(blockIdx.x) * attention_chunk;
    if (first > range.last || first + attention_chunk <= range.first)
        return;
    const auto begin = max(first, range.first), end = min(first + attention_chunk, range.last + 1);
    __shared__ float e[attention_chunk];
    __shared__ const float* values[attention_chunk];
    __shared__ float merged_denominator;
    const float negative_infinity = bit_cast<float>(0xff800000U);
    const float raw_maximum = maxima[Final ? rh : std::size_t(gridDim.y) * gridDim.x + rh];
    const float m = Final && !(raw_maximum > negative_infinity) ? negative_infinity : raw_maximum;
    for (auto p = begin + threadIdx.x; p < end; p += blockDim.x) {
        e[p - first] = declared_exp(scores[std::size_t(rh) * stride + p] - m);
        if constexpr (Prefetch)
            values[p - first] = kv_at(r, p, true, failure);
    }
    __syncthreads();
    float* out = Final ? partials + std::size_t(rh) * dim
                       : partials + (std::size_t(rh) * gridDim.x + blockIdx.x) * (dim + 1U);
    if (threadIdx.x == 0) {
        float denominator = 0.0F;
        for (auto p = begin; p < end; ++p)
            denominator = denominator + e[p - first];
        if constexpr (Final)
            merged_denominator = 0.0F + denominator;
        else
            out[dim] = denominator;
    }
    if constexpr (Final)
        __syncthreads();
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
        float numerator = 0.0F;
        if constexpr (Prefetch) {
            for (auto p = begin; p < end; p += 8U) {
                float v[8];
                unsigned present = 0;
#pragma unroll
                for (unsigned j = 0; j < 8; ++j)
                    if (p + j < end) {
                        const auto* pointer = values[p + j - first];
                        if (pointer) {
                            v[j] = pointer[(head / group) * dim + i];
                            present |= 1U << j;
                        }
                    }
                // Only loads are batched. Keep one accumulator across batches,
                // visiting every present key in its original absolute order.
#pragma unroll
                for (unsigned j = 0; j < 8; ++j)
                    if (present & (1U << j))
                        numerator = numerator + e[p + j - first] * v[j];
            }
        } else {
            for (auto p = begin; p < end; ++p) {
                const auto* v = kv_at(r, p, true, failure);
                if (v)
                    numerator = numerator + e[p - first] * v[(head / group) * dim + i];
            }
        }
        finite_flag(numerator, failure);
        if constexpr (Final) {
            // Retain the ordered single-chunk merge's initial +0 additions.
            const float merged = 0.0F + numerator;
            const float value = merged / merged_denominator;
            out[i] = value;
            finite_flag(value, failure);
        } else {
            out[i] = numerator;
        }
    }
}
__global__ void attention_merge_kernel(const KvRead* reads, unsigned heads_count, unsigned dim, unsigned chunks,
                                       const float* partials, float* out, unsigned* failure) {
    const auto r = reads[blockIdx.x / heads_count];
    const auto range = visible(r.position, r.global, r.window);
    float denominator = 0.0F;
    for (auto c = range.first / attention_chunk; c <= range.last / attention_chunk; ++c)
        denominator = denominator + partials[(std::size_t(blockIdx.x) * chunks + c) * (dim + 1U) + dim];
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
        float numerator = 0.0F;
        for (auto c = range.first / attention_chunk; c <= range.last / attention_chunk; ++c)
            numerator = numerator + partials[(std::size_t(blockIdx.x) * chunks + c) * (dim + 1U) + i];
        const float value = numerator / denominator;
        out[std::size_t(blockIdx.x) * dim + i] = value;
        finite_flag(value, failure);
    }
}
// Producer K/V epilogue: one wave per (row, KV head, K|V). K receives its
// weighted norm and rotation, V the weightless norm, with the register head
// arithmetic; results go straight to the row's overlay slot.
template <unsigned Dim>
__global__ void kv_store_kernel(const float* k, const float* v, LinearView k_norm, const std::uint64_t* positions,
                                const float* table, const KvWrite* dst, unsigned kv_heads, bool parallel,
                                bool bounded, bool intervals, unsigned* failure) {
    const unsigned head = blockIdx.x % kv_heads, value = (blockIdx.x / kv_heads) % 2U,
                   row = blockIdx.x / (2U * kv_heads), lane = threadIdx.x;
    const auto width = kv_heads * Dim;
    const float* in = (value ? v : k) + std::size_t(row) * width + head * Dim;
    float values[Dim / 32U];
#pragma unroll
    for (unsigned j = 0; j < Dim / 32U; ++j)
        values[j] = in[lane + 32U * j];
    if (value)
        register_head_values<Dim>(values, {}, nullptr, parallel, bounded, intervals);
    else
        register_head_values<Dim>(values, k_norm, table + std::size_t(positions[row]) * Dim, parallel, bounded,
                                  intervals);
    float* out = (value ? dst[row].v : dst[row].k) + std::size_t(dst[row].row) * width + head * Dim;
#pragma unroll
    for (unsigned j = 0; j < Dim / 32U; ++j) {
        out[lane + 32U * j] = values[j];
        finite_flag(values[j], failure);
    }
}
// One block per (row, query head) for a visible span of at most fused_chunks
// absolute 64-position chunks. The query norm/rotation, scores, per-chunk and
// merged maxima, per-chunk partials and their ordered merge reproduce the
// multi-launch schedule value by value; only launch boundaries are removed.
constexpr unsigned fused_chunks = 9;
// Quant (g4-a8-v1): the output row (heads * Dim values) is also quantized for the output projection;
// each wave stores whole 32-value groups in lane order.
// Lanes (STRIX_G4_ATTN_SCAN, default on): the per-chunk maxima and the per-chunk denominator partials are
// computed by one lane per chunk (each in its chunk's position order), and lane 0 folds them in chunk order:
// the same comparisons and additions in the same order as the single-thread scans.
template <unsigned Dim, bool Quant = false, bool Lanes = false>
__global__ void __launch_bounds__(256) fused_attention_kernel(const float* q, LinearView q_norm,
                                                              const std::uint64_t* positions, const float* table,
                                                              const KvRead* reads, unsigned heads_count, float* out,
                                                              bool parallel, bool bounded, bool intervals,
                                                              unsigned* failure, A8Rows quant = {}) {
    constexpr unsigned Span = fused_chunks * attention_chunk;
    const unsigned rh = blockIdx.x, row = rh / heads_count, head = rh % heads_count;
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    __shared__ float query[Dim];
    __shared__ float score[Span];
    __shared__ const float* keys[Span];
    __shared__ const float* values[Span];
    __shared__ float merged_maximum, merged_denominator;
    __shared__ float chunk_value[fused_chunks];
    const auto r = reads[row];
    const auto range = visible(r.position, r.global, r.window);
    const auto origin = (range.first / attention_chunk) * attention_chunk;
    const auto count = unsigned(range.last + 1U - range.first);
    if (wave == 0) {
        float v[Dim / 32U];
#pragma unroll
        for (unsigned j = 0; j < Dim / 32U; ++j)
            v[j] = q[std::size_t(rh) * Dim + lane + 32U * j];
        register_head_values<Dim>(v, q_norm, table + std::size_t(positions[row]) * Dim, parallel, bounded,
                                  intervals);
#pragma unroll
        for (unsigned j = 0; j < Dim / 32U; ++j) {
            query[lane + 32U * j] = v[j];
            finite_flag(v[j], failure);
        }
    }
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
        const auto p = range.first + i;
        keys[p - origin] = kv_at(r, p, false, failure);
        values[p - origin] = kv_at(r, p, true, failure);
    }
    __syncthreads();
    const auto kv_offset = (head / group) * Dim;
    // Each wave loads four keys before reducing them; every score is still one
    // independent lane-ordered dot and lane tree.
    constexpr unsigned Keys = Dim == 512U ? 2U : 4U;
    const unsigned waves = blockDim.x / 32U;
    for (unsigned base = wave; base < count; base += Keys * waves) {
        float kv[Keys][Dim / 32U];
#pragma unroll
        for (unsigned u = 0; u < Keys; ++u) {
            const auto i = base + u * waves;
            const auto* key = i < count ? keys[unsigned(range.first + i - origin)] : nullptr;
            if (key) {
#pragma unroll
                for (unsigned j = 0; j < Dim / 32U; ++j)
                    kv[u][j] = key[kv_offset + lane + 32U * j];
            }
        }
#pragma unroll
        for (unsigned u = 0; u < Keys; ++u) {
            const auto i = base + u * waves;
            if (i >= count)
                break;
            const auto slot = unsigned(range.first + i - origin);
            float sum = 0.0F;
            if (keys[slot]) {
#pragma unroll
                for (unsigned j = 0; j < Dim / 32U; ++j)
                    sum = sum + query[lane + 32U * j] * kv[u][j];
            }
            sum = wave_sum(sum);
            if (lane == 0) {
                finite_flag(sum, failure);
                score[slot] = sum;
            }
        }
    }
    __syncthreads();
    const auto first_chunk = unsigned(range.first / attention_chunk);
    const auto last_chunk = unsigned(range.last / attention_chunk);
    if constexpr (Lanes) {
        const auto chunks = last_chunk - first_chunk + 1U;
        if (threadIdx.x < chunks) {
            const auto c = first_chunk + threadIdx.x;
            float a = bit_cast<float>(0xff800000U);
            for (unsigned j = 0; j < attention_chunk; ++j) {
                const auto p = std::uint64_t(c) * attention_chunk + j;
                if (p >= range.first && p <= range.last)
                    a = score[p - origin] > a ? score[p - origin] : a;
            }
            chunk_value[threadIdx.x] = a;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = bit_cast<float>(0xff800000U);
            for (unsigned c = 0; c < chunks; ++c)
                m = chunk_value[c] > m ? chunk_value[c] : m;
            merged_maximum = m;
        }
    } else if (threadIdx.x == 0) {
        float m = bit_cast<float>(0xff800000U);
        for (auto c = first_chunk; c <= last_chunk; ++c) {
            float a = bit_cast<float>(0xff800000U);
            for (unsigned j = 0; j < attention_chunk; ++j) {
                const auto p = std::uint64_t(c) * attention_chunk + j;
                if (p >= range.first && p <= range.last)
                    a = score[p - origin] > a ? score[p - origin] : a;
            }
            m = a > m ? a : m;
        }
        merged_maximum = m;
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
        const auto slot = unsigned(range.first + i - origin);
        score[slot] = declared_exp(score[slot] - merged_maximum);
    }
    __syncthreads();
    if constexpr (Lanes) {
        const auto chunks = last_chunk - first_chunk + 1U;
        if (threadIdx.x < chunks) {
            const auto c = first_chunk + threadIdx.x;
            const auto begin = max(std::uint64_t(c) * attention_chunk, range.first);
            const auto end = min(std::uint64_t(c + 1U) * attention_chunk, range.last + 1U);
            float partial = 0.0F;
            for (auto p = begin; p < end; ++p)
                partial = partial + score[p - origin];
            chunk_value[threadIdx.x] = partial;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float denominator = 0.0F;
            for (unsigned c = 0; c < chunks; ++c)
                denominator = denominator + chunk_value[c];
            merged_denominator = denominator;
        }
    } else if (threadIdx.x == 0) {
        float denominator = 0.0F;
        for (auto c = first_chunk; c <= last_chunk; ++c) {
            const auto begin = max(std::uint64_t(c) * attention_chunk, range.first);
            const auto end = min(std::uint64_t(c + 1U) * attention_chunk, range.last + 1U);
            float partial = 0.0F;
            for (auto p = begin; p < end; ++p)
                partial = partial + score[p - origin];
            denominator = denominator + partial;
        }
        merged_denominator = denominator;
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < Dim; i += blockDim.x) {
        float numerator = 0.0F;
        for (auto c = first_chunk; c <= last_chunk; ++c) {
            const auto begin = max(std::uint64_t(c) * attention_chunk, range.first);
            const auto end = min(std::uint64_t(c + 1U) * attention_chunk, range.last + 1U);
            float partial = 0.0F;
            for (auto p = begin; p < end; p += 32U) {
                float v[32];
                unsigned present = 0;
#pragma unroll
                for (unsigned j = 0; j < 32; ++j)
                    if (p + j < end) {
                        const auto* pointer = values[p + j - origin];
                        if (pointer) {
                            v[j] = pointer[kv_offset + i];
                            present |= 1U << j;
                        }
                    }
                // Loads batched by 32; the key order of the accumulation is unchanged.
#pragma unroll
                for (unsigned j = 0; j < 32; ++j)
                    if (present & (1U << j))
                        partial = partial + score[p + j - origin] * v[j];
            }
            finite_flag(partial, failure);
            numerator = numerator + partial;
        }
        const float value = numerator / merged_denominator;
        out[std::size_t(rh) * Dim + i] = value;
        finite_flag(value, failure);
        if constexpr (Quant)
            a8_quantize_lane(value, quant, row, head * Dim + i);
    }
}
// fused_attention_kernel for spans of at most two 64-position chunks with the
// latency chain shortened, values unchanged: every wave requests all of its
// keys in one batch, each thread requests its first chunk's value column before
// the softmax (in flight during scores, maximum and denominator), and the
// ordered maximum/denominator scans read their chunk's scores into registers
// before the same sequential compare/add. Each score is still one lane-ordered
// dot and lane tree; every reduction keeps its order.
// Store: the producer K/V epilogue (kv_store_kernel) is folded in for passes in
// which each row is its transaction's only new position. Waves 1 and 2 compute
// the row's K (weighted norm + rotation) and V (weightless norm) of the block's
// KV head with kv_store_kernel's arithmetic while wave 0 normalizes the query;
// the first block of each KV group writes them to the overlay, and every block
// reads its current position from the shared copy (the same values).
struct KvStoreArgs {
    const float* k = nullptr;
    const float* v = nullptr;
    LinearView k_norm;
    const KvWrite* destinations = nullptr;
    bool seeded = false; // seeded exact rsqrt (STRIX_G4_SEEDED_RSQRT)
    A8Rows quant;        // g4-a8-v1: also quantize the output row (whole groups per wave)
};
constexpr unsigned fast_chunks = 2;
template <unsigned Dim, bool Store = false, unsigned Shape = 0>
__global__ void __launch_bounds__(256) fused_attention_fast_kernel(const float* q, LinearView q_norm,
                                                                   const std::uint64_t* positions, const float* table,
                                                                   const KvRead* reads, unsigned heads_count,
                                                                   float* out, bool parallel, bool bounded,
                                                                   bool intervals, unsigned* failure,
                                                                   KvStoreArgs store = {}) {
    constexpr unsigned Span = fast_chunks * attention_chunk;
    // Shape (STRIX_G4_ATTN_SHAPE): bit 0 doubles the keys per wave batch; bit 1
    // reads each chunk's scores into registers before the ordered maximum and
    // denominator scans; bit 2 requests each chunk's value column and scores in
    // one batch before the ordered numerator accumulation. Values and order are
    // fused_attention_kernel's for every shape; only load batching differs.
    // Bit 3 (STRIX_G4_ATTN_SCAN, default on): one lane per chunk computes that chunk's maximum and denominator
    // partial in position order and lane 0 folds them in chunk order (the same operations in the same order).
    constexpr bool Wide = (Shape & 1U) != 0, Window = (Shape & 2U) != 0, Batch = (Shape & 4U) != 0,
                   Lanes = (Shape & 8U) != 0;
    constexpr unsigned Keys = Wide ? (Dim == 512U ? 4U : 8U) : (Dim == 512U ? 2U : 4U);
    const unsigned rh = blockIdx.x, row = rh / heads_count, head = rh % heads_count;
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    __shared__ float query[Dim];
    __shared__ float score[Span];
    __shared__ const float* keys[Span];
    __shared__ const float* values[Span];
    __shared__ float merged_maximum, merged_denominator;
    __shared__ float current[Store ? 2U : 1U][Store ? kv_heads * Dim : 1U];
    __shared__ float chunk_value[fast_chunks];
    const auto r = reads[row];
    const auto range = visible(r.position, r.global, r.window);
    const auto origin = (range.first / attention_chunk) * attention_chunk;
    const auto count = unsigned(range.last + 1U - range.first);
    if constexpr (Store) {
        if (wave == 1U || wave == 2U) {
            const unsigned value = wave - 1U, kv_head = head / group, width = kv_heads * Dim;
            const float* in = (value ? store.v : store.k) + std::size_t(row) * width + kv_head * Dim;
            float x[Dim / 32U];
#pragma unroll
            for (unsigned j = 0; j < Dim / 32U; ++j)
                x[j] = in[lane + 32U * j];
            if (value)
                register_head_values<Dim>(x, {}, nullptr, parallel, bounded, intervals, store.seeded);
            else
                register_head_values<Dim>(x, store.k_norm, table + std::size_t(positions[row]) * Dim, parallel,
                                          bounded, intervals, store.seeded);
            const auto& dst = store.destinations[row];
            float* target = (value ? dst.v : dst.k) + std::size_t(dst.row) * width + kv_head * Dim;
#pragma unroll
            for (unsigned j = 0; j < Dim / 32U; ++j) {
                current[value][kv_head * Dim + lane + 32U * j] = x[j];
                if (head % group == 0U) {
                    target[lane + 32U * j] = x[j];
                    finite_flag(x[j], failure);
                }
            }
        }
    }
    if (wave == 0) {
        float v[Dim / 32U];
#pragma unroll
        for (unsigned j = 0; j < Dim / 32U; ++j)
            v[j] = q[std::size_t(rh) * Dim + lane + 32U * j];
        register_head_values<Dim>(v, q_norm, table + std::size_t(positions[row]) * Dim, parallel, bounded,
                                  intervals, store.seeded);
#pragma unroll
        for (unsigned j = 0; j < Dim / 32U; ++j) {
            query[lane + 32U * j] = v[j];
            finite_flag(v[j], failure);
        }
    }
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
        const auto p = range.first + i;
        keys[p - origin] = kv_at(r, p, false, failure);
        values[p - origin] = kv_at(r, p, true, failure);
        if constexpr (Store) {
            if (p == r.position) {
                keys[p - origin] = current[0];
                values[p - origin] = current[1];
            }
        }
    }
    __syncthreads();
    const auto kv_offset = (head / group) * Dim;
    const auto first_chunk = unsigned(range.first / attention_chunk);
    const auto last_chunk = unsigned(range.last / attention_chunk);
    const unsigned waves = blockDim.x / 32U;
    // Score batches: wave w handles positions base + waves*u, base = w + Keys*waves*batch.
    float kv[Keys][Dim / 32U];
    const auto load_keys = [&](unsigned base) {
#pragma unroll
        for (unsigned u = 0; u < Keys; ++u) {
            const auto i = base + u * waves;
            const auto* key = i < count ? keys[unsigned(range.first + i - origin)] : nullptr;
            if (key) {
#pragma unroll
                for (unsigned j = 0; j < Dim / 32U; ++j)
                    kv[u][j] = key[kv_offset + lane + 32U * j];
            }
        }
    };
    const auto store_scores = [&](unsigned base) {
#pragma unroll
        for (unsigned u = 0; u < Keys; ++u) {
            const auto i = base + u * waves;
            if (i >= count)
                break;
            const auto slot = unsigned(range.first + i - origin);
            float sum = 0.0F;
            if (keys[slot]) {
#pragma unroll
                for (unsigned j = 0; j < Dim / 32U; ++j)
                    sum = sum + query[lane + 32U * j] * kv[u][j];
            }
            sum = wave_sum(sum);
            if (lane == 0) {
                finite_flag(sum, failure);
                score[slot] = sum;
            }
        }
    };
    load_keys(wave);
    store_scores(wave);
    for (unsigned base = wave + Keys * waves; base < count; base += Keys * waves) {
        load_keys(base);
        store_scores(base);
    }
    __syncthreads();
    if constexpr (Lanes) {
        const auto chunks = last_chunk - first_chunk + 1U;
        if (threadIdx.x < chunks) {
            const auto c = first_chunk + threadIdx.x;
            const auto offset = std::uint64_t(c) * attention_chunk - origin;
            float a = bit_cast<float>(0xff800000U);
#pragma unroll
            for (unsigned j = 0; j < attention_chunk; ++j) {
                const auto p = std::uint64_t(c) * attention_chunk + j;
                if (p >= range.first && p <= range.last) {
                    const float x = score[offset + j];
                    a = x > a ? x : a;
                }
            }
            chunk_value[threadIdx.x] = a;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = bit_cast<float>(0xff800000U);
            for (unsigned c = 0; c < chunks; ++c)
                m = chunk_value[c] > m ? chunk_value[c] : m;
            merged_maximum = m;
        }
    } else if (threadIdx.x == 0) {
        float m = bit_cast<float>(0xff800000U);
        for (auto c = first_chunk; c <= last_chunk; ++c) {
            float window[attention_chunk];
            const auto offset = std::uint64_t(c) * attention_chunk - origin;
            if constexpr (Window) {
#pragma unroll
                for (unsigned j = 0; j < attention_chunk; ++j)
                    window[j] = score[offset + j];
            }
            float a = bit_cast<float>(0xff800000U);
#pragma unroll
            for (unsigned j = 0; j < attention_chunk; ++j) {
                const auto p = std::uint64_t(c) * attention_chunk + j;
                if (p >= range.first && p <= range.last) {
                    const float x = Window ? window[j] : score[offset + j];
                    a = x > a ? x : a;
                }
            }
            m = a > m ? a : m;
        }
        merged_maximum = m;
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
        const auto slot = unsigned(range.first + i - origin);
        score[slot] = declared_exp(score[slot] - merged_maximum);
    }
    __syncthreads();
    if constexpr (Lanes) {
        const auto chunks = last_chunk - first_chunk + 1U;
        if (threadIdx.x < chunks) {
            const auto c = first_chunk + threadIdx.x;
            const auto offset = std::uint64_t(c) * attention_chunk - origin;
            const auto begin = max(std::uint64_t(c) * attention_chunk, range.first);
            const auto end = min(std::uint64_t(c + 1U) * attention_chunk, range.last + 1U);
            float partial = 0.0F;
#pragma unroll
            for (unsigned j = 0; j < attention_chunk; ++j) {
                const auto p = std::uint64_t(c) * attention_chunk + j;
                if (p >= begin && p < end)
                    partial = partial + score[offset + j];
            }
            chunk_value[threadIdx.x] = partial;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float denominator = 0.0F;
            for (unsigned c = 0; c < chunks; ++c)
                denominator = denominator + chunk_value[c];
            merged_denominator = denominator;
        }
    } else if (threadIdx.x == 0) {
        float denominator = 0.0F;
        for (auto c = first_chunk; c <= last_chunk; ++c) {
            float window[attention_chunk];
            const auto offset = std::uint64_t(c) * attention_chunk - origin;
            if constexpr (Window) {
#pragma unroll
                for (unsigned j = 0; j < attention_chunk; ++j)
                    window[j] = score[offset + j];
            }
            const auto begin = max(std::uint64_t(c) * attention_chunk, range.first);
            const auto end = min(std::uint64_t(c + 1U) * attention_chunk, range.last + 1U);
            float partial = 0.0F;
#pragma unroll
            for (unsigned j = 0; j < attention_chunk; ++j) {
                const auto p = std::uint64_t(c) * attention_chunk + j;
                if (p >= begin && p < end)
                    partial = partial + (Window ? window[j] : score[offset + j]);
            }
            denominator = denominator + partial;
        }
        merged_denominator = denominator;
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < Dim; i += blockDim.x) {
        float numerator = 0.0F;
        for (auto c = first_chunk; c <= last_chunk; ++c) {
            const auto begin = max(std::uint64_t(c) * attention_chunk, range.first);
            const auto end = min(std::uint64_t(c + 1U) * attention_chunk, range.last + 1U);
            float partial = 0.0F;
            if constexpr (Batch) {
                float v[attention_chunk], w[attention_chunk];
                unsigned present[2] = {0U, 0U};
#pragma unroll
                for (unsigned j = 0; j < attention_chunk; ++j)
                    if (begin + j < end) {
                        const auto* pointer = values[begin + j - origin];
                        w[j] = score[begin + j - origin];
                        if (pointer) {
                            v[j] = pointer[kv_offset + i];
                            present[j / 32U] |= 1U << (j % 32U);
                        }
                    }
#pragma unroll
                for (unsigned j = 0; j < attention_chunk; ++j)
                    if (present[j / 32U] & (1U << (j % 32U)))
                        partial = partial + w[j] * v[j];
            } else {
                for (auto p = begin; p < end; p += 32U) {
                    float v[32];
                    unsigned present = 0;
#pragma unroll
                    for (unsigned j = 0; j < 32; ++j)
                        if (p + j < end) {
                            const auto* pointer = values[p + j - origin];
                            if (pointer) {
                                v[j] = pointer[kv_offset + i];
                                present |= 1U << j;
                            }
                        }
#pragma unroll
                    for (unsigned j = 0; j < 32; ++j)
                        if (present & (1U << j))
                            partial = partial + score[p + j - origin] * v[j];
                }
            }
            finite_flag(partial, failure);
            numerator = numerator + partial;
        }
        const float value = numerator / merged_denominator;
        out[std::size_t(rh) * Dim + i] = value;
        finite_flag(value, failure);
        if (store.quant.codes)
            a8_quantize_lane(value, store.quant, row, head * Dim + i);
    }
}
// Chunk of a long-span block: the full grid launches one slot per absolute chunk (slots == chunks); a compact
// grid (STRIX_G4_ATTN_COMPACT, sliding layers: fused_chunks slots) maps slot i to the row's first visible chunk + i.
// Per-chunk layouts keep the absolute chunk index and the full pitch, so every value is the full grid's.
__device__ inline unsigned compact_chunk(unsigned slot, unsigned slots, unsigned chunks, Visibility range) {
    return slots < chunks ? unsigned(range.first / attention_chunk) + slot : slot;
}
// Long spans, launch one of two: the chunk-parallel scores with the query
// norm/rotation recomputed per chunk block (same values as the separate head
// kernel), and per-chunk maxima.
// Shared (STRIX_G4_ATTN_KV_SHARED, default on): the grid is (row x head, chunk)
// instead of (chunk, row x head), so the blocks of every row and query head
// that read one key/value chunk run back to back and the chunk is read from
// memory once per pass instead of once per row and head. Each block computes
// exactly the values of the chunk-fastest grid.
// Local (STRIX_G4_ATTN_MERGE, default on): the chunk maximum scans the block's scores from LDS in the same
// position order instead of re-reading them from global memory (same values, same comparisons).
template <unsigned Dim, bool Local = false>
__global__ void __launch_bounds__(128) chunk_scores_kernel(const float* q, LinearView q_norm,
                                                           const std::uint64_t* positions, const float* table,
                                                           const KvRead* reads, unsigned heads_count, unsigned stride,
                                                           float* scores, float* maxima, bool parallel, bool bounded,
                                                           bool intervals, unsigned* failure, bool shared) {
    const unsigned rh = shared ? blockIdx.x : blockIdx.y, row = rh / heads_count, head = rh % heads_count;
    const auto r = reads[row];
    const auto range = visible(r.position, r.global, r.window);
    const unsigned chunks = (stride + attention_chunk - 1U) / attention_chunk;
    const unsigned chunk = compact_chunk(shared ? blockIdx.y : blockIdx.x, shared ? gridDim.y : gridDim.x, chunks, range);
    const std::uint64_t begin = std::uint64_t(chunk) * attention_chunk;
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    auto* out = scores + std::size_t(rh) * stride;
    if (begin > range.last || begin + attention_chunk <= range.first) {
        if (threadIdx.x == 0 && chunk < chunks)
            maxima[std::size_t(rh) * chunks + chunk] = bit_cast<float>(0xff800000U);
        return;
    }
    __shared__ float query[Dim];
    __shared__ float chunk_score[attention_chunk]; // Local: this block's scores, for the maximum scan
    if (wave == 0) {
        float v[Dim / 32U];
#pragma unroll
        for (unsigned j = 0; j < Dim / 32U; ++j)
            v[j] = q[std::size_t(rh) * Dim + lane + 32U * j];
        register_head_values<Dim>(v, q_norm, table + std::size_t(positions[row]) * Dim, parallel, bounded,
                                  intervals);
#pragma unroll
        for (unsigned j = 0; j < Dim / 32U; ++j) {
            query[lane + 32U * j] = v[j];
            finite_flag(v[j], failure);
        }
    }
    __syncthreads();
    const auto kv_offset = (head / group) * Dim;
    for (unsigned j = wave; j < attention_chunk; j += 4U) {
        const auto p = begin + j;
        if (p < range.first || p > range.last)
            continue;
        const auto* k = kv_at(r, p, false, failure);
        float sum = 0.0F;
        if (k) {
            float key[Dim / 32U];
#pragma unroll
            for (unsigned c = 0; c < Dim / 32U; ++c)
                key[c] = k[kv_offset + lane + 32U * c];
#pragma unroll
            for (unsigned c = 0; c < Dim / 32U; ++c)
                sum = sum + query[lane + 32U * c] * key[c];
        }
        sum = wave_sum(sum);
        if (lane == 0) {
            finite_flag(sum, failure);
            out[p] = sum;
            if constexpr (Local)
                chunk_score[j] = sum;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = bit_cast<float>(0xff800000U);
        for (unsigned j = 0; j < attention_chunk; ++j) {
            const auto p = begin + j;
            if (p >= range.first && p <= range.last) {
                const float x = Local ? chunk_score[j] : out[p];
                m = x > m ? x : m;
            }
        }
        maxima[std::size_t(rh) * chunks + chunk] = m;
    }
}
// Launch two: every chunk block merges the chunk maxima in order itself, writes
// its partial, and the last chunk block of each (row, head) merges the partials
// in chunk order. Values equal the maximum/partial/merge kernels.
// Shared: (row x head, chunk) grid as in chunk_scores_kernel, and the block
// forms the maximum with all its threads. The maximum of the visible scores
// is order independent except for +0 versus -0, and declared_exp(s - (+0)) ==
// declared_exp(s - (-0)) for every s, so every e and every byte is unchanged.
// Ordered sum of base[c * step] for c = first..last, starting at +0 and adding in increasing c; Loads > 1
// requests that many terms before adding them (the same additions in the same order).
template <unsigned Loads>
__device__ inline float ordered_sum(const float* base, std::uint64_t first, std::uint64_t last, std::size_t step) {
    float sum = 0.0F;
    for (auto c = first; c <= last; c += Loads) {
        float x[Loads];
#pragma unroll
        for (unsigned j = 0; j < Loads; ++j)
            if (c + j <= last)
                x[j] = base[(c + j) * step];
#pragma unroll
        for (unsigned j = 0; j < Loads; ++j)
            if (c + j <= last)
                sum = sum + x[j];
    }
    return sum;
}
// Batch (STRIX_G4_ATTN_MERGE, default 32; =0 restores 8): value loads requested per batch of positions before
// the batch's ordered accumulation, and the last block requests 16 chunk partials at a time before adding them
// in chunk order (Batch 8: one at a time, as before). Only the load batching changes; each numerator still adds
// its positions in position order, skipping the same absent rows, and each merge its chunks in chunk order.
// Ticket = false (STRIX_G4_ATTN_FINAL, default on): the block ends after its partial; chunk_final_kernel, the next
// launch, does the chunk-order merge for every feature in parallel instead of the last block.
template <bool Quant = false, unsigned Batch = 8, bool Ticket = true>
__global__ void __launch_bounds__(256) chunk_partial_merge_kernel(const KvRead* reads, unsigned heads_count,
                                                                  unsigned dim, unsigned stride, const float* scores,
                                                                  const float* maxima, float* partials, float* out,
                                                                  unsigned* counters, unsigned* failure,
                                                                  A8Rows quant = {}, bool shared = false) {
    const unsigned rh = shared ? blockIdx.x : blockIdx.y, head = rh % heads_count;
    const unsigned slots = shared ? gridDim.y : gridDim.x;
    const auto r = reads[rh / heads_count];
    const auto range = visible(r.position, r.global, r.window);
    const unsigned chunks = (stride + attention_chunk - 1U) / attention_chunk;
    const unsigned chunk = compact_chunk(shared ? blockIdx.y : blockIdx.x, slots, chunks, range);
    const auto first = std::uint64_t(chunk) * attention_chunk;
    __shared__ float e[attention_chunk];
    __shared__ const float* values[attention_chunk];
    __shared__ float maximum;
    __shared__ float wave_maxima[8];
    __shared__ unsigned ticket;
    const auto first_chunk = range.first / attention_chunk, last_chunk = range.last / attention_chunk;
    if (!(first > range.last || first + attention_chunk <= range.first)) {
        if (shared) {
            float m = bit_cast<float>(0xff800000U);
            for (auto c = first_chunk + threadIdx.x; c <= last_chunk; c += blockDim.x) {
                const float a = maxima[std::size_t(rh) * chunks + c];
                m = a > m ? a : m;
            }
#pragma unroll
            for (unsigned d = 16; d != 0; d >>= 1U) {
                const float a = __shfl_xor(m, int(d), 32);
                m = a > m ? a : m;
            }
            if (threadIdx.x % 32U == 0) wave_maxima[threadIdx.x / 32U] = m;
            __syncthreads();
            if (threadIdx.x == 0) {
                m = wave_maxima[0];
                for (unsigned w = 1; w < blockDim.x / 32U; ++w) m = wave_maxima[w] > m ? wave_maxima[w] : m;
                maximum = m;
            }
        } else if (threadIdx.x == 0) {
            float m = bit_cast<float>(0xff800000U);
            for (auto c = first_chunk; c <= last_chunk; ++c) {
                const float a = maxima[std::size_t(rh) * chunks + c];
                m = a > m ? a : m;
            }
            maximum = m;
        }
        __syncthreads();
        const auto begin = max(first, range.first), end = min(first + attention_chunk, range.last + 1);
        for (auto p = begin + threadIdx.x; p < end; p += blockDim.x) {
            e[p - first] = declared_exp(scores[std::size_t(rh) * stride + p] - maximum);
            values[p - first] = kv_at(r, p, true, failure);
        }
        __syncthreads();
        float* partial = partials + (std::size_t(rh) * chunks + chunk) * (dim + 1U);
        if (threadIdx.x == 0) {
            float denominator = 0.0F;
            for (auto p = begin; p < end; ++p)
                denominator = denominator + e[p - first];
            partial[dim] = denominator;
        }
        for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
            float numerator = 0.0F;
            for (auto p = begin; p < end; p += Batch) {
                float v[Batch];
                unsigned present = 0;
#pragma unroll
                for (unsigned j = 0; j < Batch; ++j)
                    if (p + j < end) {
                        const auto* pointer = values[p + j - first];
                        if (pointer) {
                            v[j] = pointer[(head / group) * dim + i];
                            present |= 1U << j;
                        }
                    }
#pragma unroll
                for (unsigned j = 0; j < Batch; ++j)
                    if (present & (1U << j))
                        numerator = numerator + e[p + j - first] * v[j];
            }
            finite_flag(numerator, failure);
            partial[i] = numerator;
        }
    }
    if constexpr (!Ticket) return;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0)
        ticket = atomicAdd(counters + rh, 1U);
    __syncthreads();
    if (ticket != slots - 1U)
        return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const float* merged = partials + std::size_t(rh) * chunks * (dim + 1U);
    constexpr unsigned Loads = Batch == 8U ? 1U : 16U;
    const float denominator = ordered_sum<Loads>(merged + dim, first_chunk, last_chunk, dim + 1U);
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
        const float numerator = ordered_sum<Loads>(merged + i, first_chunk, last_chunk, dim + 1U);
        const float value = numerator / denominator;
        out[std::size_t(rh) * dim + i] = value;
        finite_flag(value, failure);
        if constexpr (Quant)
            a8_quantize_lane(value, quant, rh / heads_count, head * dim + i);
    }
    if (threadIdx.x == 0)
        atomicExch(counters + rh, 0U);
}
// The chunk-order merge of chunk_partial_merge_kernel's last block, one thread per (row x head, feature): the
// denominator and each numerator add the chunk partials from +0 in increasing chunk order (ordered_sum batches
// only the loads), so every value is the ticketed merge's. Grid (row x head, dim / 64), 64 threads.
template <bool Quant = false>
__global__ void __launch_bounds__(64) chunk_final_kernel(const KvRead* reads, unsigned heads_count, unsigned dim,
                                                         unsigned stride, const float* partials, float* out,
                                                         unsigned* failure, A8Rows quant = {}) {
    const unsigned rh = blockIdx.x, head = rh % heads_count, i = blockIdx.y * 64U + threadIdx.x;
    const auto r = reads[rh / heads_count];
    const auto range = visible(r.position, r.global, r.window);
    const unsigned chunks = (stride + attention_chunk - 1U) / attention_chunk;
    const auto first_chunk = range.first / attention_chunk, last_chunk = range.last / attention_chunk;
    const float* merged = partials + std::size_t(rh) * chunks * (dim + 1U);
    const float denominator = ordered_sum<16>(merged + dim, first_chunk, last_chunk, dim + 1U);
    const float numerator = ordered_sum<16>(merged + i, first_chunk, last_chunk, dim + 1U);
    const float value = numerator / denominator;
    out[std::size_t(rh) * dim + i] = value;
    finite_flag(value, failure);
    if constexpr (Quant)
        a8_quantize_lane(value, quant, rh / heads_count, head * dim + i);
}
// Grouped long-span launches (STRIX_G4_ATTN_KV_SHARED=2): one block per
// (row, key/value head, chunk) serves the group's query heads, so each key and
// value is loaded once for all of them. Grid (row x kv head, chunk) as in the
// shared mode. Per (row, head) every score, maximum, e, partial and merged value
// is the chunk_scores/chunk_partial_merge value, formed by the same operations
// in the same order; only the head loop moves inside the block.
template <unsigned Dim>
__global__ void __launch_bounds__(128) group_scores_kernel(const float* q, LinearView q_norm,
                                                           const std::uint64_t* positions, const float* table,
                                                           const KvRead* reads, unsigned heads_count, unsigned stride,
                                                           float* scores, float* maxima, bool parallel, bool bounded,
                                                           bool intervals, unsigned* failure) {
    static_assert(group == 4U);
    const unsigned kv_count = heads_count / group, row = blockIdx.x / kv_count, kv_head = blockIdx.x % kv_count;
    const unsigned rh0 = row * heads_count + kv_head * group;
    const auto r = reads[row];
    const auto range = visible(r.position, r.global, r.window);
    const unsigned chunks = (stride + attention_chunk - 1U) / attention_chunk;
    const unsigned chunk = compact_chunk(blockIdx.y, gridDim.y, chunks, range);
    const std::uint64_t begin = std::uint64_t(chunk) * attention_chunk;
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    if (begin > range.last || begin + attention_chunk <= range.first) {
        if (threadIdx.x < group && chunk < chunks)
            maxima[std::size_t(rh0 + threadIdx.x) * chunks + chunk] = bit_cast<float>(0xff800000U);
        return;
    }
    __shared__ float query[group][Dim];
    {
        // Wave w forms query head kv_head * group + w (the separate head kernel's values).
        float v[Dim / 32U];
#pragma unroll
        for (unsigned j = 0; j < Dim / 32U; ++j)
            v[j] = q[std::size_t(rh0 + wave) * Dim + lane + 32U * j];
        register_head_values<Dim>(v, q_norm, table + std::size_t(positions[row]) * Dim, parallel, bounded, intervals);
#pragma unroll
        for (unsigned j = 0; j < Dim / 32U; ++j) {
            query[wave][lane + 32U * j] = v[j];
            finite_flag(v[j], failure);
        }
    }
    __syncthreads();
    const auto kv_offset = kv_head * Dim;
    for (unsigned j = wave; j < attention_chunk; j += 4U) {
        const auto p = begin + j;
        if (p < range.first || p > range.last)
            continue;
        const auto* k = kv_at(r, p, false, failure);
        float sum[group] = {};
        if (k) {
            float key[Dim / 32U];
#pragma unroll
            for (unsigned c = 0; c < Dim / 32U; ++c)
                key[c] = k[kv_offset + lane + 32U * c];
#pragma unroll
            for (unsigned h = 0; h < group; ++h)
#pragma unroll
                for (unsigned c = 0; c < Dim / 32U; ++c)
                    sum[h] = sum[h] + query[h][lane + 32U * c] * key[c];
        }
#pragma unroll
        for (unsigned h = 0; h < group; ++h) {
            const float total = wave_sum(sum[h]);
            if (lane == 0) {
                finite_flag(total, failure);
                scores[std::size_t(rh0 + h) * stride + p] = total;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < group) {
        const float* out = scores + std::size_t(rh0 + threadIdx.x) * stride;
        float m = bit_cast<float>(0xff800000U);
        for (unsigned j = 0; j < attention_chunk; ++j) {
            const auto p = begin + j;
            if (p >= range.first && p <= range.last)
                m = out[p] > m ? out[p] : m;
        }
        maxima[std::size_t(rh0 + threadIdx.x) * chunks + chunk] = m;
    }
}
template <bool Quant = false, unsigned Batch = 8>
__global__ void __launch_bounds__(256) group_partial_merge_kernel(const KvRead* reads, unsigned heads_count,
                                                                  unsigned dim, unsigned stride, const float* scores,
                                                                  const float* maxima, float* partials, float* out,
                                                                  unsigned* counters, unsigned* failure,
                                                                  A8Rows quant = {}) {
    const unsigned kv_count = heads_count / group, row = blockIdx.x / kv_count, kv_head = blockIdx.x % kv_count;
    const unsigned rh0 = row * heads_count + kv_head * group, slots = gridDim.y;
    const auto r = reads[row];
    const auto range = visible(r.position, r.global, r.window);
    const unsigned chunks = (stride + attention_chunk - 1U) / attention_chunk;
    const unsigned chunk = compact_chunk(blockIdx.y, slots, chunks, range);
    const auto first = std::uint64_t(chunk) * attention_chunk;
    __shared__ float e[group][attention_chunk];
    __shared__ const float* values[attention_chunk];
    __shared__ float maximum[group], wave_maxima[8][group];
    __shared__ unsigned ticket;
    const auto first_chunk = range.first / attention_chunk, last_chunk = range.last / attention_chunk;
    if (!(first > range.last || first + attention_chunk <= range.first)) {
        // Maximum of the chunk maxima per head, by all threads (order free, see chunk_partial_merge_kernel).
        float m[group];
#pragma unroll
        for (unsigned h = 0; h < group; ++h) m[h] = bit_cast<float>(0xff800000U);
        for (auto c = first_chunk + threadIdx.x; c <= last_chunk; c += blockDim.x)
#pragma unroll
            for (unsigned h = 0; h < group; ++h) {
                const float a = maxima[std::size_t(rh0 + h) * chunks + c];
                m[h] = a > m[h] ? a : m[h];
            }
#pragma unroll
        for (unsigned h = 0; h < group; ++h) {
#pragma unroll
            for (unsigned d = 16; d != 0; d >>= 1U) {
                const float a = __shfl_xor(m[h], int(d), 32);
                m[h] = a > m[h] ? a : m[h];
            }
            if (threadIdx.x % 32U == 0) wave_maxima[threadIdx.x / 32U][h] = m[h];
        }
        __syncthreads();
        if (threadIdx.x < group) {
            float a = wave_maxima[0][threadIdx.x];
            for (unsigned w = 1; w < blockDim.x / 32U; ++w) a = wave_maxima[w][threadIdx.x] > a ? wave_maxima[w][threadIdx.x] : a;
            maximum[threadIdx.x] = a;
        }
        __syncthreads();
        const auto begin = max(first, range.first), end = min(first + attention_chunk, range.last + 1);
        for (auto p = begin + threadIdx.x; p < end; p += blockDim.x) {
#pragma unroll
            for (unsigned h = 0; h < group; ++h)
                e[h][p - first] = declared_exp(scores[std::size_t(rh0 + h) * stride + p] - maximum[h]);
            values[p - first] = kv_at(r, p, true, failure);
        }
        __syncthreads();
        if (threadIdx.x < group) {
            float denominator = 0.0F;
            for (auto p = begin; p < end; ++p)
                denominator = denominator + e[threadIdx.x][p - first];
            partials[(std::size_t(rh0 + threadIdx.x) * chunks + chunk) * (dim + 1U) + dim] = denominator;
        }
        for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
            float numerator[group] = {};
            for (auto p = begin; p < end; p += Batch) {
                float v[Batch];
                unsigned present = 0;
#pragma unroll
                for (unsigned j = 0; j < Batch; ++j)
                    if (p + j < end) {
                        const auto* pointer = values[p + j - first];
                        if (pointer) {
                            v[j] = pointer[kv_head * dim + i];
                            present |= 1U << j;
                        }
                    }
#pragma unroll
                for (unsigned j = 0; j < Batch; ++j)
                    if (present & (1U << j))
#pragma unroll
                        for (unsigned h = 0; h < group; ++h)
                            numerator[h] = numerator[h] + e[h][p + j - first] * v[j];
            }
#pragma unroll
            for (unsigned h = 0; h < group; ++h) {
                finite_flag(numerator[h], failure);
                partials[(std::size_t(rh0 + h) * chunks + chunk) * (dim + 1U) + i] = numerator[h];
            }
        }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0)
        ticket = atomicAdd(counters + rh0, 1U);
    __syncthreads();
    if (ticket != slots - 1U)
        return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    for (unsigned h = 0; h < group; ++h) {
        const float* merged = partials + std::size_t(rh0 + h) * chunks * (dim + 1U);
        constexpr unsigned Loads = Batch == 8U ? 1U : 16U;
        const float denominator = ordered_sum<Loads>(merged + dim, first_chunk, last_chunk, dim + 1U);
        for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
            const float numerator = ordered_sum<Loads>(merged + i, first_chunk, last_chunk, dim + 1U);
            const float value = numerator / denominator;
            out[std::size_t(rh0 + h) * dim + i] = value;
            finite_flag(value, failure);
            if constexpr (Quant)
                a8_quantize_lane(value, quant, row, (kv_head * group + h) * dim + i);
        }
    }
    if (threadIdx.x == 0)
        atomicExch(counters + rh0, 0U);
}
template <unsigned Dim, bool Store, unsigned Shape>
void launch_fast_attention_shape(unsigned blocks, hipStream_t stream, const float* q, LinearView q_norm,
                                 const std::uint64_t* positions, const float* table, const KvRead* reads,
                                 unsigned heads_count, float* out, bool parallel, bool bounded, bool intervals,
                                 unsigned* failure, KvStoreArgs store) {
    hipLaunchKernelGGL((fused_attention_fast_kernel<Dim, Store, Shape>), dim3(blocks), dim3(256), 0, stream, q, q_norm,
                       positions, table, reads, heads_count, out, parallel, bounded, intervals, failure, store);
}
// STRIX_G4_ATTN_SCAN (default on, read once per process): lane-parallel per-chunk maximum and denominator
// scans in the decode attention kernels (same values); =0 restores the single-thread scans.
bool attention_lane_scans() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_ATTN_SCAN");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
template <bool Store>
void launch_fast_attention(bool global, unsigned blocks, hipStream_t stream, const float* q, LinearView q_norm,
                           const std::uint64_t* positions, const float* table, const KvRead* reads,
                           unsigned heads_count, float* out, bool parallel, bool bounded, bool intervals,
                           unsigned* failure, KvStoreArgs store) {
    static const unsigned shape = [] {
        const auto* p = std::getenv("STRIX_G4_ATTN_SHAPE");
        return (p ? unsigned(std::strtoul(p, nullptr, 10)) & 7U : 6U) | (attention_lane_scans() ? 8U : 0U);
    }();
    const auto launch = [&](auto dim, auto form) {
        launch_fast_attention_shape<decltype(dim)::value, Store, decltype(form)::value>(
            blocks, stream, q, q_norm, positions, table, reads, heads_count, out, parallel, bounded, intervals, failure,
            store);
    };
    using D256 = std::integral_constant<unsigned, 256>;
    using D512 = std::integral_constant<unsigned, 512>;
    const auto by_shape = [&](auto dim) {
        switch (shape) {
        case 0: launch(dim, std::integral_constant<unsigned, 0>{}); break;
        case 1: launch(dim, std::integral_constant<unsigned, 1>{}); break;
        case 2: launch(dim, std::integral_constant<unsigned, 2>{}); break;
        case 3: launch(dim, std::integral_constant<unsigned, 3>{}); break;
        case 4: launch(dim, std::integral_constant<unsigned, 4>{}); break;
        case 5: launch(dim, std::integral_constant<unsigned, 5>{}); break;
        case 6: launch(dim, std::integral_constant<unsigned, 6>{}); break;
        case 7: launch(dim, std::integral_constant<unsigned, 7>{}); break;
        case 8: launch(dim, std::integral_constant<unsigned, 8>{}); break;
        case 9: launch(dim, std::integral_constant<unsigned, 9>{}); break;
        case 10: launch(dim, std::integral_constant<unsigned, 10>{}); break;
        case 11: launch(dim, std::integral_constant<unsigned, 11>{}); break;
        case 12: launch(dim, std::integral_constant<unsigned, 12>{}); break;
        case 13: launch(dim, std::integral_constant<unsigned, 13>{}); break;
        case 14: launch(dim, std::integral_constant<unsigned, 14>{}); break;
        default: launch(dim, std::integral_constant<unsigned, 15>{}); break;
        }
    };
    if (global)
        by_shape(D512{});
    else
        by_shape(D256{});
}
// STRIX_G4_ATTN_MERGE (default on, read once per process): the chunk-parallel decode attention merge kernels
// request 32 value rows per batch before the ordered accumulation instead of 8 (same values); =0 restores 8.
bool merge_batch_wide() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_ATTN_MERGE");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool seeded_rsqrt() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_SEEDED_RSQRT");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
} // namespace
bool fused_attention_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_FUSED_ATTN");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
int attention_kv_shared_mode() {
    static const int mode = [] {
        const auto* p = std::getenv("STRIX_G4_ATTN_KV_SHARED");
        return p && std::string_view(p) == "0" ? 0 : p && std::string_view(p) == "1" ? 1 : p && std::string_view(p) == "2" ? 2 : 3;
    }();
    return mode;
}
bool attention_kv_shared_enabled() { return attention_kv_shared_mode() != 0; }
// STRIX_G4_ATTN_COMPACT (default on, read once per process): sliding long-span launches cover only the window's
// chunks; =0 launches one slot per chunk of the stride (same values either way).
// STRIX_G4_ATTN_FINAL (default on, read once per process): the one-row long-span merge ends each chunk block at
// its partial and merges the chunks in a separate per-feature launch; =0 keeps the ticketed last-block merge.
static bool attention_final() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_ATTN_FINAL");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
static bool attention_compact() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_ATTN_COMPACT");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool fused_attention(const float* q, LinearView q_norm, const std::uint64_t* positions, const float* table,
                     const KvRead* reads, unsigned rows, unsigned heads_count, unsigned dim, unsigned stride,
                     bool global, float* out, unsigned* failure, Queue queue, bool parallel, bool bounded,
                     bool intervals, float* scores, float* maxima, float* partials, unsigned* counters,
                     A8Rows quant, int kv_shared) {
    valid_rows(rows);
    // g4-a8-v1: quantize the attention rows in the writing launch (whole groups per wave).
    const bool quantize = quant.codes && quant.width == heads_count * dim && dim % 256U == 0;
    // One block per (row, head) serializes the whole span: it wins on short
    // spans and loses on full 512-position windows, so longer spans use the
    // chunk-parallel launches (STRIX_G4_FUSED_ATTN_CHUNKS, default 2, at most 9).
    static const unsigned limit = [] {
        const auto* p = std::getenv("STRIX_G4_FUSED_ATTN_CHUNKS");
        return p ? unsigned(std::strtoul(p, nullptr, 10)) : 2U;
    }();
    const auto chunks = (stride + attention_chunk - 1U) / attention_chunk;
    const auto span = global ? chunks : std::min(chunks, fused_chunks);
    if (!fused_attention_enabled() || !(global ? dim == 512U : dim == 256U) || heads_count % group || !q_norm.data)
        return false;
    const auto stream = static_cast<hipStream_t>(queue);
    if (span > std::min(limit, fused_chunks)) {
        if (!counters || !scores || !maxima || !partials)
            return false;
        // Default (3): one row keeps a block per query head (more blocks); more rows share the group.
        int mode = kv_shared < 0 ? attention_kv_shared_mode() : kv_shared;
        if (mode == 3) mode = rows > 1U ? 2 : 1;
        // Sliding layers see at most fused_chunks chunks: a compact grid launches only those (the stride's chunk
        // count only sizes the layouts), instead of one mostly empty block per chunk of the longest context.
        const unsigned slots = !global && attention_compact() ? span : chunks;
        if (mode == 2) {
            const dim3 groups(rows * (heads_count / group), slots);
            if (global)
                hipLaunchKernelGGL((group_scores_kernel<512>), groups, dim3(128), 0, stream, q, q_norm, positions, table, reads,
                                   heads_count, stride, scores, maxima, parallel, bounded, intervals, failure);
            else
                hipLaunchKernelGGL((group_scores_kernel<256>), groups, dim3(128), 0, stream, q, q_norm, positions, table, reads,
                                   heads_count, stride, scores, maxima, parallel, bounded, intervals, failure);
            launch_checked();
            const auto group_merge = [&](auto quantized, auto batch) {
                hipLaunchKernelGGL((group_partial_merge_kernel<decltype(quantized)::value, decltype(batch)::value>), groups,
                                   dim3(256), 0, stream, reads, heads_count, dim, stride, scores, maxima, partials, out,
                                   counters, failure, quant);
            };
            const auto group_batch = [&](auto quantized) {
                if (merge_batch_wide()) group_merge(quantized, std::integral_constant<unsigned, 32>{});
                else group_merge(quantized, std::integral_constant<unsigned, 8>{});
            };
            if (quantize) group_batch(std::true_type{});
            else group_batch(std::false_type{});
            launch_checked();
            return true;
        }
        const bool shared = mode != 0;
        const dim3 grid = shared ? dim3(rows * heads_count, slots) : dim3(slots, rows * heads_count);
        const auto chunk_scores = [&](auto dim, auto local) {
            hipLaunchKernelGGL((chunk_scores_kernel<decltype(dim)::value, decltype(local)::value>), grid, dim3(128), 0,
                               stream, q, q_norm, positions, table, reads, heads_count, stride, scores, maxima, parallel,
                               bounded, intervals, failure, shared);
        };
        const auto scores_local = [&](auto dim) {
            if (merge_batch_wide()) chunk_scores(dim, std::true_type{});
            else chunk_scores(dim, std::false_type{});
        };
        if (global) scores_local(std::integral_constant<unsigned, 512>{});
        else scores_local(std::integral_constant<unsigned, 256>{});
        launch_checked();
        const bool final_launch = attention_final();
        const auto chunk_merge = [&](auto quantized, auto batch) {
            constexpr bool q8 = decltype(quantized)::value;
            if (final_launch) {
                hipLaunchKernelGGL((chunk_partial_merge_kernel<q8, decltype(batch)::value, false>), grid, dim3(256), 0,
                                   stream, reads, heads_count, dim, stride, scores, maxima, partials, out, counters,
                                   failure, quant, shared);
                launch_checked();
                hipLaunchKernelGGL((chunk_final_kernel<q8>), dim3(rows * heads_count, dim / 64U), dim3(64), 0, stream,
                                   reads, heads_count, dim, stride, partials, out, failure, quant);
            } else
                hipLaunchKernelGGL((chunk_partial_merge_kernel<q8, decltype(batch)::value>), grid, dim3(256), 0, stream,
                                   reads, heads_count, dim, stride, scores, maxima, partials, out, counters, failure,
                                   quant, shared);
        };
        const auto chunk_batch = [&](auto quantized) {
            if (merge_batch_wide()) chunk_merge(quantized, std::integral_constant<unsigned, 32>{});
            else chunk_merge(quantized, std::integral_constant<unsigned, 8>{});
        };
        if (quantize) chunk_batch(std::true_type{});
        else chunk_batch(std::false_type{});
        launch_checked();
        return true;
    }
    static const bool fast = [] {
        const auto* p = std::getenv("STRIX_G4_FAST_ATTN");
        return !p || std::string_view(p) != "0";
    }();
    if (fast && span <= fast_chunks) {
        KvStoreArgs plain;
        plain.seeded = seeded_rsqrt();
        plain.quant = quantize ? quant : A8Rows{};
        launch_fast_attention<false>(global, rows * heads_count, stream, q, q_norm, positions, table, reads,
                                     heads_count, out, parallel, bounded, intervals, failure, plain);
    } else {
        const auto launch = [&](auto dim, auto quantized, auto lanes) {
            hipLaunchKernelGGL((fused_attention_kernel<decltype(dim)::value, decltype(quantized)::value,
                                                       decltype(lanes)::value>),
                               dim3(rows * heads_count), dim3(256), 0, stream, q, q_norm, positions, table, reads,
                               heads_count, out, parallel, bounded, intervals, failure, quant);
        };
        using D256 = std::integral_constant<unsigned, 256>;
        using D512 = std::integral_constant<unsigned, 512>;
        const auto with_quant = [&](auto dim, auto lanes) {
            if (quantize) launch(dim, std::true_type{}, lanes);
            else launch(dim, std::false_type{}, lanes);
        };
        const auto with_dim = [&](auto lanes) {
            if (global) with_quant(D512{}, lanes);
            else with_quant(D256{}, lanes);
        };
        if (attention_lane_scans()) with_dim(std::true_type{});
        else with_dim(std::false_type{});
    }
    launch_checked();
    return true;
}
bool fused_attention_store(const float* q, LinearView q_norm, const float* k, const float* v, LinearView k_norm,
                           const KvWrite* destinations, unsigned kv_head_count, const std::uint64_t* positions,
                           const float* table, const KvRead* reads, unsigned rows, unsigned heads_count, unsigned dim,
                           unsigned stride, bool global, float* out, unsigned* failure, Queue queue, bool parallel,
                           bool bounded, bool intervals, A8Rows quant) {
    valid_rows(rows);
    static const bool enabled = [] {
        const auto* fast = std::getenv("STRIX_G4_FAST_ATTN");
        const auto* p = std::getenv("STRIX_G4_ATTN_KV_STORE");
        return (!fast || std::string_view(fast) != "0") && (!p || std::string_view(p) != "0");
    }();
    const auto chunks = (stride + attention_chunk - 1U) / attention_chunk;
    const auto span = global ? chunks : std::min(chunks, fused_chunks);
    if (!enabled || !fused_attention_enabled() || !(global ? dim == 512U : dim == 256U) ||
        kv_head_count != kv_heads || heads_count != kv_heads * group || !q_norm.data || !k_norm.data ||
        span > fast_chunks || (quant.codes && (quant.width != heads_count * dim || dim % 256U != 0)))
        return false;
    const KvStoreArgs store{k, v, k_norm, destinations, seeded_rsqrt(), quant};
    launch_fast_attention<true>(global, rows * heads_count, static_cast<hipStream_t>(queue), q, q_norm, positions,
                                table, reads, heads_count, out, parallel, bounded, intervals, failure, store);
    launch_checked();
    return true;
}
bool kv_store(const float* k, const float* v, LinearView k_norm, const std::uint64_t* positions, const float* table,
              const KvWrite* destinations, unsigned rows, unsigned kv_heads, unsigned dim, bool global,
              unsigned* failure, Queue queue, bool parallel, bool bounded, bool intervals) {
    valid_rows(rows);
    if (!fused_attention_enabled() || !(global ? dim == 512U : dim == 256U) || !k_norm.data)
        return false;
    const auto stream = static_cast<hipStream_t>(queue);
    if (global)
        hipLaunchKernelGGL((kv_store_kernel<512>), dim3(rows * kv_heads * 2U), dim3(32), 0, stream, k, v, k_norm,
                           positions, table, destinations, kv_heads, parallel, bounded, intervals, failure);
    else
        hipLaunchKernelGGL((kv_store_kernel<256>), dim3(rows * kv_heads * 2U), dim3(32), 0, stream, k, v, k_norm,
                           positions, table, destinations, kv_heads, parallel, bounded, intervals, failure);
    launch_checked();
    return true;
}
void attention(const float* q, const KvRead* reads, unsigned rows, unsigned heads_count, unsigned dim, unsigned stride,
               float* scores, float* maxima, float* partials, float* out, unsigned* failure, Queue queue,
               bool allow_fusion) {
    valid_rows(rows);
    if (!heads_count || heads_count % group || !dim || dim > 512 || !stride || stride > max_positions)
        throw std::runtime_error("device attention geometry");
    const auto chunks = (stride + attention_chunk - 1U) / attention_chunk;
    const auto stream = static_cast<hipStream_t>(queue);
    static const bool prefetch = [] {
        const auto* p = std::getenv("STRIX_G4_ATTN_PREFETCH");
        return !p || std::string_view(p) != "0";
    }();
    static const bool one_chunk = [] {
        const auto* p = std::getenv("STRIX_G4_ONE_CHUNK_ATTN");
        const auto* master = std::getenv("STRIX_G4_FUSION");
        return p && std::string_view(p) == "1" && (!master || std::string_view(master) != "0");
    }();
    if (prefetch)
        hipLaunchKernelGGL((scores_kernel<true>), dim3(chunks, rows * heads_count), dim3(128), 0, stream, q, reads,
                           heads_count, dim, stride, scores, maxima, failure);
    else
        hipLaunchKernelGGL((scores_kernel<false>), dim3(chunks, rows * heads_count), dim3(128), 0, stream, q, reads,
                           heads_count, dim, stride, scores, maxima, failure);
    launch_checked();
    if (allow_fusion && one_chunk && chunks == 1U) {
        if (prefetch)
            hipLaunchKernelGGL((partial_kernel<true, true>), dim3(1, rows * heads_count), dim3(256), 0, stream, reads,
                               heads_count, dim, stride, scores, maxima, out, failure);
        else
            hipLaunchKernelGGL((partial_kernel<false, true>), dim3(1, rows * heads_count), dim3(256), 0, stream, reads,
                               heads_count, dim, stride, scores, maxima, out, failure);
        launch_checked();
        return;
    }
    hipLaunchKernelGGL(maximum_kernel, dim3(rows * heads_count), dim3(32), 0, stream, reads, heads_count, chunks,
                       maxima);
    launch_checked();
    if (prefetch)
        hipLaunchKernelGGL((partial_kernel<true>), dim3(chunks, rows * heads_count), dim3(256), 0, stream, reads,
                           heads_count, dim, stride, scores, maxima, partials, failure);
    else
        hipLaunchKernelGGL((partial_kernel<false>), dim3(chunks, rows * heads_count), dim3(256), 0, stream, reads,
                           heads_count, dim, stride, scores, maxima, partials, failure);
    launch_checked();
    hipLaunchKernelGGL(attention_merge_kernel, dim3(rows * heads_count), dim3(256), 0, stream, reads, heads_count, dim,
                       chunks, partials, out, failure);
    launch_checked();
}
} // namespace strix::gemma4::gpu
