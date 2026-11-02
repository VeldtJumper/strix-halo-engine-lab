#pragma clang fp contract(off)
#include "strix/hip_ds41_dspark.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "strix/hip_ds41_formats.hpp"

#include "dspark_common/markov_cand_kernels.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::ds41::dspark {
namespace {

void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string("ds41 dspark: ") + what + ": " + hipGetErrorString(status));
}
namespace mc = ::strix::dspark::markov_cand;
static_assert(block_size <= mc::kMaxRows && markov_rank == mc::kRank);
hipStream_t as_stream(void* stream) { return static_cast<hipStream_t>(stream); }

__device__ __forceinline__ float f16_device(std::uint16_t bits) { return __half2float(__ushort_as_half(bits)); }
// Wave-order reduction: every lane holds its ascending partial sum; butterfly d = 16..1.
__device__ __forceinline__ float butterfly(float v) {
    for (int d = 16; d != 0; d /= 2) v = v + __shfl_xor(v, d, 32);
    return v;
}
__device__ __forceinline__ float entry_value_device(const std::uint8_t* entry, unsigned i) {
    return decode_e4m3(entry[i]) * scale_value(entry[head_dim + i / 32U]);
}

__global__ __launch_bounds__(256) void main_hidden_kernel(const float* l37, const float* l38, const float* l39, float* out) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (i >= main_width) return;
    const unsigned l = i / hidden, c = i % hidden;
    const float* s = (l == 0U ? l37 : l == 1U ? l38 : l39) + static_cast<std::size_t>(r) * stream_width;
    out[static_cast<std::size_t>(r) * main_width + i] = (((s[c] + s[hidden + c]) + s[2U * hidden + c]) + s[3U * hidden + c]) * 0.25F;
}

// Chunked sum (chunk_sum): wave w of the block sums chunk w; the chunk sums are added in order.
__device__ __forceinline__ float block_chunk_sum(float lane_acc, float* partial) {
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, waves = blockDim.x / 32U;
    const float part = butterfly(lane_acc);
    if (lane == 0U) partial[wave] = part;
    __syncthreads();
    float total = partial[0];
    for (unsigned c = 1; c < waves; ++c) total = total + partial[c];
    __syncthreads();
    return total;
}

// One block per row, one wave per 640-value chunk of the row (width <= 640: one wave).
__global__ __launch_bounds__(1024) void rms_norm_kernel(const float* x, unsigned width, const float* weight, float* out) {
    __shared__ float partial[32];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const float* row = x + static_cast<std::size_t>(blockIdx.x) * width;
    const unsigned first = wave * sum_chunk, end = min(width, first + sum_chunk);
    float acc = 0.0F;
    for (unsigned i = first + lane; i < end; i += 32U) acc = acc + row[i] * row[i];
    const float rstd = rstd_of(block_chunk_sum(acc, partial), width, norm_eps);
    float* y = out + static_cast<std::size_t>(blockIdx.x) * width;
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) y[i] = (row[i] * rstd) * weight[i];
}

// blockIdx.x < 24: dot of fn row j with the row's stream; == 24: sum of squares. One wave per
// 640-value chunk (32 waves). scratch[r][25].
__global__ __launch_bounds__(1024) void hc_dot_kernel(const float* streams, const std::uint16_t* fn, float* scratch) {
    __shared__ float partial[32];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, j = blockIdx.x, r = blockIdx.y;
    const float* x = streams + static_cast<std::size_t>(r) * stream_width;
    const unsigned first = wave * sum_chunk, end = first + sum_chunk;
    float acc = 0.0F;
    if (j < mix_count) {
        const std::uint16_t* f = fn + static_cast<std::size_t>(j) * stream_width;
        for (unsigned i = first + lane; i < end; i += 32U) acc = acc + f16_device(f[i]) * x[i];
    } else {
        for (unsigned i = first + lane; i < end; i += 32U) acc = acc + x[i] * x[i];
    }
    const float total = block_chunk_sum(acc, partial);
    if (threadIdx.x == 0U) scratch[r * (mix_count + 1U) + j] = total;
}
__global__ void sinkhorn_kernel(const float* scratch, const float* scale, const float* base, unsigned rows, float* mixes) {
    const unsigned r = threadIdx.x;
    if (r >= rows) return;
    const float* raw = scratch + r * (mix_count + 1U);
    const float rstd = rstd_of(raw[mix_count], stream_width, norm_eps);
    float m[mix_count];
    for (unsigned j = 0; j < mix_count; ++j) m[j] = raw[j] * rstd;
    sinkhorn_core(m, scale, base, mixes + r * mix_count);
}

__global__ __launch_bounds__(256) void hc_pre_kernel(const float* streams, const float* coefficients, unsigned stride, float* out) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (i >= hidden) return;
    const float* x = streams + static_cast<std::size_t>(r) * stream_width;
    const float* p = coefficients + static_cast<std::size_t>(r) * stride;
    out[static_cast<std::size_t>(r) * hidden + i] =
        ((p[0] * x[i] + p[1] * x[hidden + i]) + p[2] * x[2U * hidden + i]) + p[3] * x[3U * hidden + i];
}
__global__ __launch_bounds__(256) void hc_post_kernel(const float* x, const float* residual, const float* mixes, float* out) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (i >= hidden) return;
    const float* m = mixes + static_cast<std::size_t>(r) * mix_count;
    const float* s = residual + static_cast<std::size_t>(r) * stream_width;
    const float* c = m + 2U * hc;
    const float xi = x[static_cast<std::size_t>(r) * hidden + i];
    for (unsigned k = 0; k < hc; ++k)
        out[static_cast<std::size_t>(r) * stream_width + k * hidden + i] =
            m[hc + k] * xi + (((c[0U * hc + k] * s[i] + c[1U * hc + k] * s[hidden + i]) + c[2U * hc + k] * s[2U * hidden + i]) +
                              c[3U * hc + k] * s[3U * hidden + i]);
}

__global__ __launch_bounds__(256) void rope_kernel(float* x, unsigned row_width, unsigned head_width, const float* table, bool inverse) {
    const unsigned r = blockIdx.x, pairs = row_width / head_width * (rope_dim / 2U);
    const float* t = table + static_cast<std::size_t>(r) * rope_dim;
    for (unsigned item = threadIdx.x; item < pairs; item += 256U) {
        const unsigned h = item / (rope_dim / 2U), i = item % (rope_dim / 2U);
        float* tail = x + static_cast<std::size_t>(r) * row_width + static_cast<std::size_t>(h) * head_width + (head_width - rope_dim);
        rope_pair(tail[2U * i], tail[2U * i + 1U], t[i], t[rope_dim / 2U + i], inverse);
    }
}

__global__ __launch_bounds__(32) void fp8_entries_kernel(const float* x, std::uint8_t* entries, unsigned capacity, unsigned long long first) {
    const unsigned lane = threadIdx.x, g = blockIdx.x, r = blockIdx.y;
    const float v = x[static_cast<std::size_t>(r) * head_dim + 32U * g + lane];
    float amax = fabsf(v);
    for (int d = 16; d != 0; d /= 2) amax = fmaxf(amax, __shfl_xor(amax, d, 32));
    const std::uint8_t code = fp8_scale_code(amax);
    const unsigned long long slot = capacity != 0U ? (first + r) % capacity : r;
    std::uint8_t* entry = entries + slot * entry_bytes;
    entry[32U * g + lane] = fp8_element(v, scale_value(code));
    if (lane == 0U) entry[head_dim + g] = code;
}

constexpr unsigned max_keys = window + 8U;
// Decodes the keys of a block attention once (ring positions ascending, then the block rows) into
// FP32 rows of 512 (the exact entry values), shared by every (head, row) block.
__global__ __launch_bounds__(256) void decode_keys_kernel(const std::uint8_t* ring, unsigned ring_keys, unsigned long long oldest,
                                                          const std::uint8_t* block, float* keys) {
    const unsigned j = blockIdx.x;
    const std::uint8_t* k = j < ring_keys ? ring + ((oldest + j) % window) * entry_bytes : block + static_cast<std::size_t>(j - ring_keys) * entry_bytes;
    for (unsigned d = threadIdx.x; d < head_dim; d += 256U) keys[static_cast<std::size_t>(j) * head_dim + d] = entry_value_device(k, d);
}
// Block (head, row): 8 waves score the keys (wave w takes keys w, w + 8, ...), then every thread
// forms two output dimensions in key order.
__global__ __launch_bounds__(256) void block_attention_kernel(const float* q, const float* key_values, unsigned keys, const float* sinks,
                                                              float* out) {
    __shared__ float scores[max_keys];
    __shared__ float weights[max_keys];
    __shared__ float total;
    const unsigned h = blockIdx.x, r = blockIdx.y, lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const float* qh = q + (static_cast<std::size_t>(r) * heads + h) * head_dim;
    for (unsigned j = wave; j < keys; j += 8U) {
        const float* k = key_values + static_cast<std::size_t>(j) * head_dim;
        float acc = 0.0F;
        for (unsigned d = lane; d < head_dim; d += 32U) acc = acc + qh[d] * k[d];
        acc = butterfly(acc);
        if (lane == 0U) scores[j] = acc * attention_scale;
    }
    __syncthreads();
    float m = scores[0];
    for (unsigned j = 1; j < keys; ++j) m = scores[j] > m ? scores[j] : m;
    for (unsigned j = threadIdx.x; j < keys; j += 256U) weights[j] = declared_exp(scores[j] - m);
    __syncthreads();
    if (threadIdx.x == 0U) {
        float sum = 0.0F;
        for (unsigned j = 0; j < keys; ++j) sum = sum + weights[j];
        total = sum + declared_exp(sinks[h] - m);
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < head_dim; d += 256U) {
        float acc = 0.0F;
        for (unsigned j = 0; j < keys; ++j) acc = acc + weights[j] * key_values[static_cast<std::size_t>(j) * head_dim + d];
        out[(static_cast<std::size_t>(r) * heads + h) * head_dim + d] = acc / total;
    }
}

__global__ __launch_bounds__(256) void embed_block_kernel(const std::uint16_t* table, const std::uint32_t* tokens, float* streams) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (i >= hidden) return;
    const float v = f16_device(table[static_cast<std::size_t>(tokens[r]) * hidden + i]);
    for (unsigned k = 0; k < hc; ++k) streams[static_cast<std::size_t>(r) * stream_width + k * hidden + i] = v;
}

// One wave per vocabulary entry (markov_dot order).
__global__ __launch_bounds__(256) void markov_bias_kernel(const std::uint16_t* embed, const std::uint16_t* head, const std::uint32_t* tokens,
                                                          unsigned step, float* logits, float* markov) {
    const unsigned lane = threadIdx.x % 32U, v = blockIdx.x * 8U + threadIdx.x / 32U;
    const std::uint16_t* e = embed + static_cast<std::size_t>(tokens[step]) * markov_rank;
    if (blockIdx.x == 0U)
        for (unsigned j = threadIdx.x; j < markov_rank; j += 256U) markov[static_cast<std::size_t>(step) * markov_rank + j] = f16_device(e[j]);
    if (v >= vocab) return;
    const std::uint16_t* hv = head + static_cast<std::size_t>(v) * markov_rank;
    float acc = 0.0F;
    for (unsigned k = 0; k < 8U; ++k) acc = acc + f16_device(hv[8U * lane + k]) * f16_device(e[8U * lane + k]);
    acc = butterfly(acc);
    if (lane == 0U) {
        float* l = logits + static_cast<std::size_t>(step) * vocab + v;
        *l = *l + acc;
    }
}
__global__ __launch_bounds__(1024) void argmax_kernel(const float* logits, unsigned step, std::uint32_t* tokens) {
    __shared__ float values[1024];
    __shared__ unsigned indices[1024];
    const float* l = logits + static_cast<std::size_t>(step) * vocab;
    float best = -__builtin_huge_valf();
    unsigned best_index = vocab;
    for (unsigned v = threadIdx.x; v < vocab; v += 1024U)
        if (better(l[v], v, best, best_index)) { best = l[v]; best_index = v; }
    values[threadIdx.x] = best;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    for (unsigned half = 512; half != 0U; half /= 2U) {
        if (threadIdx.x < half) {
            const unsigned o = threadIdx.x + half;
            if (better(values[o], indices[o], values[threadIdx.x], indices[threadIdx.x])) {
                values[threadIdx.x] = values[o];
                indices[threadIdx.x] = indices[o];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) tokens[step + 1U] = indices[0];
}
__global__ __launch_bounds__(32) void confidence_kernel(const float* weight, const float* x, const float* markov, float* out) {
    const unsigned lane = threadIdx.x, r = blockIdx.x;
    const float* xr = x + static_cast<std::size_t>(r) * hidden;
    const float* er = markov + static_cast<std::size_t>(r) * markov_rank;
    float acc = 0.0F;
    for (unsigned j = lane; j < confidence_width; j += 32U) acc = acc + weight[j] * (j < hidden ? xr[j] : er[j - hidden]);
    acc = butterfly(acc);
    if (lane == 0U) out[r] = acc;
}

// ---- router of the draft rows (reference_router of ds41_moe.hpp): one wave per (expert, row) for
// the scores, then one block per row for the top-k ----
__global__ __launch_bounds__(256) void draft_router_scores_kernel(const float* x, const float* weight, unsigned rows, float* scores) {
    const unsigned lane = threadIdx.x % 32U, item = blockIdx.x * 8U + threadIdx.x / 32U;
    if (item >= experts * rows) return;
    const unsigned e = item % experts, r = item / experts;
    const float* w = weight + static_cast<std::size_t>(e) * hidden;
    const float* xr = x + static_cast<std::size_t>(r) * hidden;
    float acc = 0.0F;
    for (unsigned i = lane; i < hidden; i += 32U) acc = acc + w[i] * xr[i];
    acc = butterfly(acc);
    if (lane == 0U) scores[static_cast<std::size_t>(r) * experts + e] = router_score(acc);
}
__global__ __launch_bounds__(128) void draft_router_select_kernel(const float* scores, const float* bias, std::uint32_t* ids, float* weights) {
    __shared__ float value[128];
    __shared__ unsigned index[128];
    __shared__ unsigned chosen[top_k];
    const unsigned r = blockIdx.x, t = threadIdx.x;
    const float* sc = scores + static_cast<std::size_t>(r) * experts;
    const float biased = sc[t] + bias[t];
    for (unsigned k = 0; k < top_k; ++k) {
        bool taken = false;
        for (unsigned j = 0; j < k; ++j) taken = taken || chosen[j] == t;
        value[t] = biased;
        index[t] = taken ? experts : t;
        __syncthreads();
        for (unsigned h = 64; h != 0U; h /= 2U) {
            if (t < h) {
                const float ov = value[t + h];
                const unsigned oi = index[t + h];
                const bool take = oi != experts && (index[t] == experts || ov > value[t] || (ov == value[t] && oi < index[t]));
                if (take) { value[t] = ov; index[t] = oi; }
            }
            __syncthreads();
        }
        if (t == 0U) chosen[k] = index[0];
        __syncthreads();
    }
    if (t == 0U) {
        float sum = 0.0F;
        for (unsigned k = 0; k < top_k; ++k) sum = sum + sc[chosen[k]];
        const float denominator = sum + 1e-20F;
        for (unsigned k = 0; k < top_k; ++k) {
            ids[r * top_k + k] = chosen[k];
            weights[r * top_k + k] = sc[chosen[k]] / denominator * route_scale;
        }
    }
}

// ---- MoE routes built on the device (build_window_routes of ds41_moe.hpp, one thread) ----
// The union is padded to rows * top_k experts with empty entries, so launches need no host sync;
// the window kernels skip experts without entries.
__global__ void build_routes_kernel(const std::uint32_t* ids, unsigned rows, unsigned k_top, unsigned split, unsigned slot_rows,
                                    unsigned max_slots, std::uint32_t* experts_out,
                                    std::uint32_t* begins, std::uint32_t* rows_out, std::uint32_t* slots, std::uint32_t* order,
                                    std::uint32_t* low_counts) {
    if (threadIdx.x != 0U) return;
    const unsigned n = rows * k_top;
    std::uint32_t uniq[64];
    unsigned count = 0;
    for (unsigned i = 0; i < n; ++i) {
        const std::uint32_t e = ids[i];
        unsigned at = 0;
        while (at < count && uniq[at] < e) ++at;
        if (at < count && uniq[at] == e) continue;
        for (unsigned j = count; j > at; --j) uniq[j] = uniq[j - 1U];
        uniq[at] = e;
        ++count;
    }
    // Slots of at most `slot_rows` entries: an expert with more entries takes several consecutive
    // slots (each entry's product is independent of the others, so the results do not change).
    unsigned pos = 0, slot = 0;
    begins[0] = 0;
    for (unsigned j = 0; j < count; ++j) {
        unsigned in_slot = 0;
        for (unsigned i = 0; i < n; ++i)
            if (ids[i] == uniq[j]) {
                if (in_slot == slot_rows) {
                    experts_out[slot] = uniq[j];
                    begins[++slot] = pos;
                    in_slot = 0;
                }
                rows_out[pos] = i / k_top; slots[pos] = i; ++pos; ++in_slot;
            }
        experts_out[slot] = uniq[j];
        begins[++slot] = pos;
    }
    for (; slot < max_slots; ++slot) {
        experts_out[slot] = 0U;
        begins[slot + 1U] = pos;
    }
    for (unsigned r = 0; r < rows; ++r) {
        std::uint32_t row_slots[16];
        for (unsigned k = 0; k < k_top; ++k) row_slots[k] = r * k_top + k;
        for (unsigned a = 1; a < k_top; ++a)
            for (unsigned b = a; b > 0U && ids[row_slots[b - 1U]] > ids[row_slots[b]]; --b) {
                const std::uint32_t t = row_slots[b]; row_slots[b] = row_slots[b - 1U]; row_slots[b - 1U] = t;
            }
        unsigned low = 0;
        for (unsigned k = 0; k < k_top; ++k) {
            order[r * k_top + k] = row_slots[k];
            low += ids[row_slots[k]] < split ? 1U : 0U;
        }
        low_counts[r] = low;
    }
}

// ---- Q8_0 rows of a wide input: one wave per output, activations read from memory (no LDS
// staging), the declared Q8_0 product per (row, output) ----
template <unsigned R>
__global__ __launch_bounds__(256) void q8_rows_kernel(const std::uint8_t* weights, unsigned long long row_bytes, unsigned width,
                                                      unsigned outputs, const signed char* codes, const float* deltas, unsigned rows,
                                                      float* out) {
    const unsigned lane = threadIdx.x % 32U, o = blockIdx.x * 8U + threadIdx.x / 32U;
    if (o >= outputs) return;
    const unsigned groups = width / 32U;
    const std::uint8_t* row = weights + static_cast<unsigned long long>(o) * row_bytes;
    float acc[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) acc[r] = 0.0F;
    for (unsigned g = lane; g < groups; g += 32U) {
        const std::uint8_t* block = row + 34ULL * g;
        std::uint16_t half[17];
        __builtin_memcpy(half, block, 34);
        const float scale = f16_device(half[0]);
        int w[8];
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) w[k] = static_cast<int>(half[1U + 2U * k] | (static_cast<unsigned>(half[2U + 2U * k]) << 16U));
#pragma unroll
        for (unsigned r = 0; r < R; ++r)
            if (r < rows) {
                const int4* x = reinterpret_cast<const int4*>(codes + static_cast<unsigned long long>(r) * width + 32U * g);
                const int4 x0 = x[0], x1 = x[1];
                int dot = 0;
                dot = __builtin_amdgcn_sudot4(true, w[0], true, x0.x, dot, false);
                dot = __builtin_amdgcn_sudot4(true, w[1], true, x0.y, dot, false);
                dot = __builtin_amdgcn_sudot4(true, w[2], true, x0.z, dot, false);
                dot = __builtin_amdgcn_sudot4(true, w[3], true, x0.w, dot, false);
                dot = __builtin_amdgcn_sudot4(true, w[4], true, x1.x, dot, false);
                dot = __builtin_amdgcn_sudot4(true, w[5], true, x1.y, dot, false);
                dot = __builtin_amdgcn_sudot4(true, w[6], true, x1.z, dot, false);
                dot = __builtin_amdgcn_sudot4(true, w[7], true, x1.w, dot, false);
                acc[r] = acc[r] + static_cast<float>(dot) * (scale * deltas[static_cast<unsigned long long>(r) * groups + g]);
            }
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
        if (r < rows) {
            const float v = butterfly(acc[r]);
            if (lane == 0U) out[static_cast<unsigned long long>(r) * outputs + o] = v;
        }
}

// ---- Markov chain: one pass over the Markov head per step, fused with a partial argmax ----
// The rows of this rank: j < core (logits_core[step][j], id core_base + j) then j - core < extra
// (logits_extra, id extra_ids[extra_base + j - core]). value_j = logit_j + markov_dot(head[id_j], e)
// (ds41_dspark.hpp: lane l takes elements 8l..8l+7, one 16-byte load); per block the best (value, id),
// then a pick kernel (the token, or with two ranks this rank's candidate for the exchange).
struct HeadView {
    const float* logits_core;
    unsigned core;
    unsigned core_base;
    const float* logits_extra;
    unsigned extra;
    unsigned extra_base;
    const std::uint32_t* extra_ids;
    const std::uint16_t* embed;
    const std::uint16_t* head;
    std::uint32_t* tokens;
    float* markov;
    float* best_values;          // [blocks]
    std::uint32_t* best_ids;     // [blocks]
};
constexpr unsigned markov_blocks = 512;
__device__ __forceinline__ float head_logit(const HeadView& v, unsigned step, unsigned j) {
    return j < v.core ? v.logits_core[static_cast<unsigned long long>(step) * v.core + j]
                      : v.logits_extra[static_cast<unsigned long long>(step) * v.extra + (j - v.core)];
}
__device__ __forceinline__ unsigned head_id(const HeadView& v, unsigned j) {
    return j < v.core ? v.core_base + j : v.extra_ids[v.extra_base + (j - v.core)];
}
__device__ __forceinline__ void unpack8(const uint4 q, float (&out)[8]) {
    const unsigned w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (unsigned k = 0; k < 4U; ++k) {
        out[2U * k] = f16_device(static_cast<std::uint16_t>(w[k] & 0xffffU));
        out[2U * k + 1U] = f16_device(static_cast<std::uint16_t>(w[k] >> 16U));
    }
}
__global__ __launch_bounds__(256) void markov_pass_kernel(HeadView v, unsigned step) {
    __shared__ float values[8];
    __shared__ unsigned ids[8];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const std::uint16_t* erow = v.embed + static_cast<unsigned long long>(v.tokens[step]) * markov_rank;
    float e[8];
    unpack8(reinterpret_cast<const uint4*>(erow)[lane], e);
    if (blockIdx.x == 0U && wave == 0U)
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) v.markov[static_cast<unsigned long long>(step) * markov_rank + 8U * lane + k] = e[k];
    float best = -__builtin_huge_valf();
    unsigned best_id = vocab;
    const unsigned size = v.core + v.extra, waves = gridDim.x * 8U;
    for (unsigned j = blockIdx.x * 8U + wave; j < size; j += waves) {
        const unsigned id = head_id(v, j);
        float h[8];
        unpack8(reinterpret_cast<const uint4*>(v.head + static_cast<unsigned long long>(id) * markov_rank)[lane], h);
        float acc = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) acc = acc + h[k] * e[k];
        acc = butterfly(acc);
        const float value = head_logit(v, step, j) + acc;
        if (better(value, id, best, best_id)) { best = value; best_id = id; }
    }
    if (lane == 0U) { values[wave] = best; ids[wave] = best_id; }
    __syncthreads();
    if (threadIdx.x == 0U) {
        for (unsigned w = 1; w < 8U; ++w)
            if (better(values[w], ids[w], best, best_id)) { best = values[w]; best_id = ids[w]; }
        v.best_values[blockIdx.x] = best;
        v.best_ids[blockIdx.x] = best_id;
    }
}
__global__ __launch_bounds__(markov_blocks) void markov_pick_kernel(HeadView v, unsigned step, std::uint32_t* candidate) {
    __shared__ float values[markov_blocks];
    __shared__ unsigned ids[markov_blocks];
    const unsigned t = threadIdx.x;
    values[t] = v.best_values[t];
    ids[t] = v.best_ids[t];
    __syncthreads();
    for (unsigned h = markov_blocks / 2U; h != 0U; h /= 2U) {
        if (t < h && better(values[t + h], ids[t + h], values[t], ids[t])) { values[t] = values[t + h]; ids[t] = ids[t + h]; }
        __syncthreads();
    }
    if (t == 0U) {
        if (candidate) { candidate[0] = __float_as_uint(values[0]); candidate[1] = ids[0]; }
        else v.tokens[step + 1U] = ids[0];
    }
}
// Two ranks: the better of the local and the remote candidate (a total order, so both ranks agree).
__global__ void markov_combine_kernel(const std::uint32_t* local, const std::uint32_t* remote, std::uint32_t* tokens, unsigned step) {
    const float a = __uint_as_float(local[0]), b = __uint_as_float(remote[0]);
    tokens[step + 1U] = better(b, remote[1], a, local[1]) ? remote[1] : local[1];
}
// Candidate-form chain (STRIX_DS41_DSPARK_MARKOV=cand): markov[r] = embed[tokens[r]] for the confidence (the full pass
// writes the same values).
__global__ __launch_bounds__(256) void markov_rows_kernel(const std::uint16_t* embed, const std::uint32_t* tokens, float* markov) {
    const unsigned r = blockIdx.x;
    const std::uint16_t* e = embed + static_cast<unsigned long long>(tokens[r]) * markov_rank;
    for (unsigned j = threadIdx.x; j < markov_rank; j += 256U) markov[static_cast<unsigned long long>(r) * markov_rank + j] = f16_device(e[j]);
}
// Gathers Q8_0 head rows of extra vocabulary ids into a packed table (rows first .. first + count).
__global__ __launch_bounds__(256) void gather_rows_kernel(const std::uint8_t* table, unsigned long long row_bytes, const std::uint32_t* ids,
                                                          unsigned first, std::uint8_t* out) {
    const unsigned r = first + blockIdx.y;
    const std::uint8_t* src = table + static_cast<unsigned long long>(ids[r]) * row_bytes;
    std::uint8_t* dst = out + static_cast<unsigned long long>(r) * row_bytes;
    for (unsigned long long i = blockIdx.x * 256ULL + threadIdx.x; i < row_bytes; i += 256ULL * gridDim.x) dst[i] = src[i];
}

// ---- device memory ----

template <class T>
struct Buffer {
    T* data = nullptr;
    std::size_t count = 0;
    Buffer() = default;
    explicit Buffer(std::size_t n) : count(n) {
        check(hipMalloc(reinterpret_cast<void**>(&data), std::max<std::size_t>(n, 1U) * sizeof(T)), "hipMalloc");
    }
    Buffer(const T* host, std::size_t n) : Buffer(n) {
        check(hipMemcpy(data, host, n * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept : data(o.data), count(o.count) { o.data = nullptr; o.count = 0; }
    Buffer& operator=(Buffer&& o) noexcept {
        std::swap(data, o.data);
        std::swap(count, o.count);
        return *this;
    }
    ~Buffer() {
        if (data) (void)hipFree(data);
    }
    void upload(const T* host, std::size_t n, hipStream_t stream = nullptr) {
        check(hipMemcpyAsync(data, host, n * sizeof(T), hipMemcpyHostToDevice, stream), "upload");
    }
    std::vector<T> download(hipStream_t stream = nullptr, std::size_t n = 0) const {
        std::vector<T> out(n ? n : count);
        check(hipMemcpyAsync(out.data(), data, out.size() * sizeof(T), hipMemcpyDeviceToHost, stream), "download");
        check(hipStreamSynchronize(stream), "download sync");
        return out;
    }
};
template <class T>
Buffer<T> upload_span(std::span<const T> s) { return Buffer<T>(s.data(), s.size()); }

struct DeviceQ8 {
    Buffer<std::uint8_t> bytes;
    std::uint32_t width = 0, outputs = 0;
    std::uint64_t row_bytes = 0;
    DeviceQ8() = default;
    explicit DeviceQ8(const Q8Matrix& m) : bytes(upload_span(m.bytes)), width(m.width), outputs(m.outputs), row_bytes(m.row_bytes()) {}
};
struct DeviceExperts {
    ExpertTensors host;
    Buffer<std::uint8_t> gate, up, down;
    DeviceExperts() = default;
    explicit DeviceExperts(const ExpertTensors& t) : host(t), gate(upload_span(t.gate)), up(upload_span(t.up)), down(upload_span(t.down)) {}
};
struct DeviceStage {
    Buffer<std::uint16_t> hc_attn_fn, hc_ffn_fn;
    Buffer<float> hc_attn_scale, hc_attn_base, hc_ffn_scale, hc_ffn_base, attn_norm, ffn_norm, sinks, q_norm, kv_norm;
    DeviceQ8 q_a, q_b, kv, out_a, out_b;
    Buffer<float> router, bias, bias_image;
    DeviceExperts routed, shared;
    Buffer<std::uint8_t> ring;
};

// STRIX_DS41_DSPARK_MEMLOG=1: the amdgpu GTT in use (sysfs mem_info_gtt_used of the first GPU) at each stage of the
// drafter's setup, against the bytes it allocates there (DeviceDrafter::footprint's parts). Diagnostics only.
bool memlog_on() {
    static const bool on = [] { const char* v = std::getenv("STRIX_DS41_DSPARK_MEMLOG"); return v && v[0] == '1'; }();
    return on;
}
std::uint64_t gtt_used() {
    std::error_code ec;
    for (const auto& de : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
        std::ifstream f(de.path() / "device" / "mem_info_gtt_used");
        std::uint64_t v = 0;
        if (f >> v) return v;
    }
    return 0;
}
void memlog(const char* stage, std::uint64_t& last, std::uint64_t expected) {
    if (!memlog_on()) return;
    (void)hipDeviceSynchronize();
    const auto now = gtt_used();
    std::fprintf(stderr, "dspark memlog: %-34s GTT +%9.1f MiB (allocated here %9.1f MiB)\n", stage, static_cast<double>(static_cast<std::int64_t>(now - last)) / 1048576.0,
                 static_cast<double>(expected) / 1048576.0);
    last = now;
}

// Activation rows that one product launch stages in LDS fit in 64 KB.
std::uint32_t row_chunk(std::uint32_t width) {
    for (std::uint32_t c = 8U; c > 1U; c /= 2U)
        if (static_cast<std::size_t>(c) * width + static_cast<std::size_t>(c) * (width / 32U) * 4U <= 65536U) return c;
    return 1U;
}

} // namespace

// ---- single operations ----

void enqueue_main_hidden(const float* layer37, const float* layer38, const float* layer39, std::uint32_t rows, float* out, void* stream) {
    hipLaunchKernelGGL(main_hidden_kernel, dim3((main_width + 255U) / 256U, rows), dim3(256), 0, as_stream(stream), layer37, layer38, layer39,
                       out);
    check(hipGetLastError(), "main hidden launch");
}
void enqueue_rms_norm(const float* x, std::uint32_t rows, std::uint32_t width, const float* weight, float* out, void* stream) {
    if (rows == 0U || width % 32U != 0U) throw std::invalid_argument("ds41 dspark rms norm geometry");
    if (width > sum_chunk && width % sum_chunk != 0U) throw std::invalid_argument("ds41 dspark rms norm width is not chunked");
    const unsigned waves = width <= sum_chunk ? 1U : width / sum_chunk;
    hipLaunchKernelGGL(rms_norm_kernel, dim3(rows), dim3(32U * waves), 0, as_stream(stream), x, width, weight, out);
    check(hipGetLastError(), "rms norm launch");
}
void enqueue_hc_mixes(const float* streams, std::uint32_t rows, const std::uint16_t* fn, const float* scale, const float* base, float* scratch,
                      float* mixes, void* stream) {
    if (rows == 0U || rows > 256U) throw std::invalid_argument("ds41 dspark hc mixes rows");
    hipLaunchKernelGGL(hc_dot_kernel, dim3(mix_count + 1U, rows), dim3(32U * (stream_width / sum_chunk)), 0, as_stream(stream), streams, fn,
                       scratch);
    hipLaunchKernelGGL(sinkhorn_kernel, dim3(1), dim3(rows), 0, as_stream(stream), scratch, scale, base, rows, mixes);
    check(hipGetLastError(), "hc mixes launch");
}
void enqueue_hc_pre(const float* streams, std::uint32_t rows, const float* coefficients, std::uint32_t stride, float* out, void* stream) {
    hipLaunchKernelGGL(hc_pre_kernel, dim3((hidden + 255U) / 256U, rows), dim3(256), 0, as_stream(stream), streams, coefficients, stride, out);
    check(hipGetLastError(), "hc pre launch");
}
void enqueue_hc_post(const float* x, const float* residual, std::uint32_t rows, const float* mixes, float* out, void* stream) {
    hipLaunchKernelGGL(hc_post_kernel, dim3((hidden + 255U) / 256U, rows), dim3(256), 0, as_stream(stream), x, residual, mixes, out);
    check(hipGetLastError(), "hc post launch");
}
void enqueue_rope(float* x, std::uint32_t rows, std::uint32_t row_width, std::uint32_t head_width, const float* table, bool inverse,
                  void* stream) {
    if (rows == 0U || head_width < rope_dim || row_width % head_width != 0U) throw std::invalid_argument("ds41 dspark rope geometry");
    hipLaunchKernelGGL(rope_kernel, dim3(rows), dim3(256), 0, as_stream(stream), x, row_width, head_width, table, inverse);
    check(hipGetLastError(), "rope launch");
}
void enqueue_fp8_entries(const float* x, std::uint32_t rows, std::uint8_t* entries, std::uint32_t capacity, std::uint64_t first,
                         void* stream) {
    hipLaunchKernelGGL(fp8_entries_kernel, dim3(head_dim / 32U, rows), dim3(32), 0, as_stream(stream), x, entries, capacity,
                       static_cast<unsigned long long>(first));
    check(hipGetLastError(), "fp8 entries launch");
}
void enqueue_block_attention(const float* q, std::uint32_t rows, const std::uint8_t* ring, std::uint32_t ring_keys, std::uint64_t oldest,
                             const std::uint8_t* block, const float* sinks, float* key_scratch, float* out, void* stream) {
    if (rows == 0U || rows > 8U || ring_keys > window || !key_scratch) throw std::invalid_argument("ds41 dspark block attention geometry");
    const unsigned keys = ring_keys + rows;
    hipLaunchKernelGGL(decode_keys_kernel, dim3(keys), dim3(256), 0, as_stream(stream), ring, ring_keys, static_cast<unsigned long long>(oldest),
                       block, key_scratch);
    hipLaunchKernelGGL(block_attention_kernel, dim3(heads, rows), dim3(256), 0, as_stream(stream), q, key_scratch, keys, sinks, out);
    check(hipGetLastError(), "block attention launch");
}
void enqueue_embed_block(const std::uint16_t* token_embd, const std::uint32_t* tokens, std::uint32_t rows, float* streams, void* stream) {
    hipLaunchKernelGGL(embed_block_kernel, dim3((hidden + 255U) / 256U, rows), dim3(256), 0, as_stream(stream), token_embd, tokens, streams);
    check(hipGetLastError(), "embed launch");
}
void enqueue_markov_step(const std::uint16_t* embed, const std::uint16_t* head, std::uint32_t* tokens, std::uint32_t step, float* logits,
                         float* markov, void* stream) {
    hipLaunchKernelGGL(markov_bias_kernel, dim3((vocab + 7U) / 8U), dim3(256), 0, as_stream(stream), embed, head, tokens, step, logits, markov);
    hipLaunchKernelGGL(argmax_kernel, dim3(1), dim3(1024), 0, as_stream(stream), logits, step, tokens);
    check(hipGetLastError(), "markov step launch");
}
void enqueue_confidence(const float* weight, const float* x, const float* markov, std::uint32_t rows, float* out, void* stream) {
    hipLaunchKernelGGL(confidence_kernel, dim3(rows), dim3(32), 0, as_stream(stream), weight, x, markov, out);
    check(hipGetLastError(), "confidence launch");
}

// ---- resident drafter ----

struct DeviceDrafter::Impl {
    static constexpr std::uint32_t max_rows = 128; // rows of one main_rows/append call
    std::uint64_t gtt_mark = memlog_on() ? gtt_used() : 0U;  // STRIX_DS41_DSPARK_MEMLOG: the GTT before any member
    hipStream_t stream{};
    DeviceQ8 main_proj, output;
    Buffer<float> main_norm, final_norm, confidence;
    Buffer<std::uint16_t> markov_embed, markov_head, token_embd;
    DeviceStage stage[stages];
    std::uint64_t next = 0;
    std::uint64_t start = 0;  // the first position with an appended row (skip_to); attention never reads before it
    // Scratch.
    Buffer<float> x_main{static_cast<std::size_t>(max_rows) * main_width};
    Buffer<float> proj{static_cast<std::size_t>(max_rows) * hidden}, main_x{static_cast<std::size_t>(max_rows) * hidden};
    Buffer<std::int8_t> codes{static_cast<std::size_t>(max_rows) * main_width};
    Buffer<float> deltas{static_cast<std::size_t>(max_rows) * main_width / 32U};
    Buffer<std::int32_t> sums{static_cast<std::size_t>(max_rows) * main_width / 32U};
    Buffer<float> kv{static_cast<std::size_t>(max_rows) * head_dim}, kv_normed{static_cast<std::size_t>(max_rows) * head_dim};
    Buffer<float> rope_table{static_cast<std::size_t>(max_rows) * rope_dim};
    Buffer<std::uint32_t> zero{1};
    // Draft step buffers (5 rows).
    Buffer<std::uint32_t> tokens{block_size + 1U};
    Buffer<float> streams{static_cast<std::size_t>(block_size) * stream_width}, streams2{static_cast<std::size_t>(block_size) * stream_width};
    Buffer<float> attn_mix{block_size * mix_count}, ffn_mix{block_size * mix_count}, pre_mix{block_size * mix_count},
        identity{block_size * mix_count};
    Buffer<float> mix_scratch{block_size * (mix_count + 1U)};
    Buffer<float> x{static_cast<std::size_t>(block_size) * hidden}, xn{static_cast<std::size_t>(block_size) * hidden};
    Buffer<float> qa{static_cast<std::size_t>(block_size) * q_rank}, qan{static_cast<std::size_t>(block_size) * q_rank};
    Buffer<float> q{static_cast<std::size_t>(block_size) * q_width}, o{static_cast<std::size_t>(block_size) * q_width};
    Buffer<float> low{static_cast<std::size_t>(block_size) * o_width}, y{static_cast<std::size_t>(block_size) * hidden};
    Buffer<std::uint8_t> block_entries{static_cast<std::size_t>(block_size) * entry_bytes};
    Buffer<float> key_scratch{static_cast<std::size_t>(max_keys) * head_dim};
    Buffer<std::uint32_t> wo_experts, wo_begins, wo_entries;
    // MoE.
    Buffer<float> scores{static_cast<std::size_t>(block_size) * experts}, route_weights{block_size * top_k};
    Buffer<std::uint32_t> ids{block_size * top_k};
    Buffer<std::int8_t> act_codes{static_cast<std::size_t>(block_size) * top_k * inter};
    Buffer<float> act_deltas{static_cast<std::size_t>(block_size) * top_k * inter / 32U};
    Buffer<std::int32_t> act_sums{static_cast<std::size_t>(block_size) * top_k * inter / 32U};
    Buffer<std::int8_t> shared_codes{static_cast<std::size_t>(block_size) * inter};
    Buffer<float> shared_deltas{static_cast<std::size_t>(block_size) * inter / 32U};
    Buffer<std::int32_t> shared_sums{static_cast<std::size_t>(block_size) * inter / 32U};
    Buffer<float> raw{static_cast<std::size_t>(block_size) * top_k * hidden}, shared_out{static_cast<std::size_t>(block_size) * hidden};
    Buffer<std::uint32_t> shared_experts, shared_begins, shared_rows;
    // Device routes: slots of at most 4 entries (the MaxRows 4 window kernels); 5 rows x top-3 need at
    // most 15 + 3 slots.
    static constexpr std::uint32_t route_slot_rows = 4, route_slots = block_size * top_k + 3U;
    Buffer<std::uint32_t> r_experts{route_slots}, r_begins{route_slots + 1U}, r_rows{block_size * top_k},
        r_slots{block_size * top_k}, r_order{block_size * top_k}, r_low{block_size};
    // Head and draft vocabulary.
    std::uint32_t extra_capacity = 8192;
    std::uint32_t core = vocab;
    std::vector<std::uint32_t> extra_host;
    std::vector<std::uint8_t> member = std::vector<std::uint8_t>(vocab, 0U); // extra ids
    Buffer<std::uint32_t> extra_ids{8192};
    Buffer<std::uint8_t> extra_rows;
    Buffer<float> logits{static_cast<std::size_t>(block_size) * vocab}, logits_extra{static_cast<std::size_t>(block_size) * 8192};
    // Two ranks: this rank evaluates its half of the vocabulary rows and exchanges its candidate.
    std::uint32_t rank = 0, world = 1;
    RankExchange exchange;
    Buffer<std::uint32_t> candidate{2}, remote_candidate{2};
    Buffer<float> markov{static_cast<std::size_t>(block_size) * markov_rank};
    Buffer<float> best_values{markov_blocks};
    Buffer<std::uint32_t> best_ids{markov_blocks};
    Buffer<float> confidence_out{block_size};
    // Candidate-form Markov chain (STRIX_DS41_DSPARK_MARKOV=cand; kernels in dspark_common/markov_cand_kernels.hpp). The
    // domain (core ids, then extras) splits in two halves by position as in local_rows(); every chain step of row k scores
    // each half's top-C of row k's head logits (value desc, position asc) and H, the |H| domain ids of largest Markov-head
    // row sum of squares, with the full pass's per-candidate arithmetic. One rank selects both halves; two ranks select
    // their own half and exchange the payload once per draft. Both see the same candidates and `better` is a total order,
    // so the tokens do not depend on the rank count. The full pass's token is kept whenever it is a candidate.
    DeviceDrafter::MarkovConfig mcfg = DeviceDrafter::markov_config();
    struct CandState {
        std::vector<std::uint32_t> order;            // every id by row sum of squares desc, id asc (first use)
        std::uint64_t key = 0;                       // the domain H was derived for (0: none yet)
        std::vector<std::uint32_t> hid[2], hpos[2];  // H per half: ids, positions within the half
        std::uint32_t hmax = 0;
        Buffer<std::uint32_t> d_hid[2], d_hpos[2];
        Buffer<unsigned char> pay[2], ws;  // pay[0]: the payload of this rank's half (one rank: half 0), pay[1]: the other half's
        Buffer<mc::Cand> cand;
        std::array<std::uint32_t, 2U * block_size> counts{};  // both payloads' row counts (the fallback decision)
    } cs;
    // Sequence contexts (set_sequences): what one request's drafting owns -- the stage rings, the position and the
    // draft vocabulary. The active context lives in the fields above; the others are parked here (parked[active]
    // holds nothing while it is active). Selecting swaps buffers, nothing is copied.
    struct Context {
        Buffer<std::uint8_t> ring[stages];
        std::uint64_t next = 0, start = 0;
        std::uint32_t core = vocab, extra_capacity = 0;
        std::vector<std::uint32_t> extra_host;
        std::vector<std::uint8_t> member;
        Buffer<std::uint32_t> extra_ids;
        Buffer<std::uint8_t> extra_rows;
        Buffer<float> logits_extra;
    };
    std::vector<Context> parked;
    std::uint32_t active = 0;
    void swap_context(Context& other) {
        for (std::uint32_t si = 0; si < stages; ++si) std::swap(stage[si].ring, other.ring[si]);
        std::swap(next, other.next);
        std::swap(start, other.start);
        std::swap(core, other.core);
        std::swap(extra_capacity, other.extra_capacity);
        std::swap(extra_host, other.extra_host);
        std::swap(member, other.member);
        std::swap(extra_ids, other.extra_ids);
        std::swap(extra_rows, other.extra_rows);
        std::swap(logits_extra, other.logits_extra);
    }
    // A fresh context: empty rings, position 0, the full vocabulary (as a new DeviceDrafter).
    Context fresh_context() {
        Context ctx;
        for (auto& r : ctx.ring) {
            r = Buffer<std::uint8_t>(static_cast<std::size_t>(window) * entry_bytes);
            check(hipMemsetAsync(r.data, 0, r.count, stream), "clear ring");
        }
        ctx.extra_capacity = 8192;
        ctx.member.assign(vocab, 0U);
        ctx.extra_ids = Buffer<std::uint32_t>(8192);
        ctx.extra_rows = Buffer<std::uint8_t>(static_cast<std::size_t>(8192) * output.row_bytes);
        ctx.logits_extra = Buffer<float>(static_cast<std::size_t>(block_size) * 8192);
        return ctx;
    }

    explicit Impl(const Weights& w)
        : main_proj(w.main_proj), output(w.output), main_norm(upload_span(w.main_norm)), final_norm(upload_span(w.final_norm)),
          confidence(upload_span(w.confidence)), markov_embed(upload_span(w.markov_embed.bits)), markov_head(upload_span(w.markov_head.bits)),
          token_embd(upload_span(w.token_embd.bits)) {
        if (memlog_on()) {
            const auto f = footprint(w, 1U);
            std::uint64_t stage_bytes = 0;
            for (const auto& h : w.stage)
                stage_bytes += (h.hc_attn_fn.bits.size() + h.hc_ffn_fn.bits.size()) * 2U + h.q_a.bytes.size() + h.q_b.bytes.size() + h.kv.bytes.size() +
                               h.out_a.bytes.size() + h.out_b.bytes.size() + h.routed.gate.size() + h.routed.up.size() + h.routed.down.size() +
                               h.shared.gate.size() + h.shared.up.size() + h.shared.down.size() +
                               (h.hc_attn_scale.size() + h.hc_attn_base.size() + h.hc_ffn_scale.size() + h.hc_ffn_base.size() + h.attn_norm.size() +
                                h.ffn_norm.size() + h.sinks.size() + h.q_norm.size() + h.kv_norm.size() + h.router.size() + h.bias.size() + h.bias_image.size()) * 4U;
            memlog_stages_expected = stage_bytes;
            memlog("members (head weights + scratch)", gtt_mark, f.weights - stage_bytes + f.scratch);
        }
        check(hipStreamCreate(&stream), "stream");
        for (std::uint32_t s = 0; s < stages; ++s) {
            const auto& h = w.stage[s];
            auto& d = stage[s];
            d.hc_attn_fn = upload_span(h.hc_attn_fn.bits); d.hc_ffn_fn = upload_span(h.hc_ffn_fn.bits);
            d.hc_attn_scale = upload_span(h.hc_attn_scale); d.hc_attn_base = upload_span(h.hc_attn_base);
            d.hc_ffn_scale = upload_span(h.hc_ffn_scale); d.hc_ffn_base = upload_span(h.hc_ffn_base);
            d.attn_norm = upload_span(h.attn_norm); d.ffn_norm = upload_span(h.ffn_norm); d.sinks = upload_span(h.sinks);
            d.q_norm = upload_span(h.q_norm); d.kv_norm = upload_span(h.kv_norm);
            d.q_a = DeviceQ8(h.q_a); d.q_b = DeviceQ8(h.q_b); d.kv = DeviceQ8(h.kv); d.out_a = DeviceQ8(h.out_a); d.out_b = DeviceQ8(h.out_b);
            d.router = upload_span(h.router); d.bias = upload_span(h.bias); d.bias_image = upload_span(h.bias_image);
            d.routed = DeviceExperts(h.routed); d.shared = DeviceExperts(h.shared);
            d.ring = Buffer<std::uint8_t>(static_cast<std::size_t>(window) * entry_bytes);
            check(hipMemsetAsync(d.ring.data, 0, d.ring.count, stream), "clear ring");
        }
        memlog("stages (weights + rings)", gtt_mark, memlog_stages_expected + std::uint64_t{stages} * window * entry_bytes);
        const std::uint32_t z = 0;
        zero.upload(&z, 1, stream);
        std::vector<float> ident(block_size * mix_count, 0.0F);
        for (std::uint32_t r = 0; r < block_size; ++r) ident[r * mix_count] = 1.0F;
        identity.upload(ident.data(), ident.size(), stream);
        // Grouped output projection: expert g = rows g * 1024.. of out_a, entries (r, g) in ascending row.
        std::vector<std::uint32_t> ex, begins, entries;
        begins.push_back(0);
        for (std::uint32_t g = 0; g < o_groups; ++g) {
            ex.push_back(g);
            for (std::uint32_t r = 0; r < block_size; ++r) entries.push_back(r * o_groups + g);
            begins.push_back(static_cast<std::uint32_t>(entries.size()));
        }
        wo_experts = Buffer<std::uint32_t>(ex.data(), ex.size());
        wo_begins = Buffer<std::uint32_t>(begins.data(), begins.size());
        wo_entries = Buffer<std::uint32_t>(entries.data(), entries.size());
        // Shared expert: every row on expert 0.
        std::vector<std::uint32_t> sb{0U, block_size}, rows_list;
        for (std::uint32_t r = 0; r < block_size; ++r) rows_list.push_back(r);
        shared_experts = Buffer<std::uint32_t>(&z, 1);
        shared_begins = Buffer<std::uint32_t>(sb.data(), sb.size());
        shared_rows = Buffer<std::uint32_t>(rows_list.data(), rows_list.size());
        extra_rows = Buffer<std::uint8_t>(static_cast<std::size_t>(extra_capacity) * output.row_bytes);
        const auto flag = [](const char* name) {
            const char* v = std::getenv(name);
            return !(v && v[0] == '0');
        };
        if (const char* v = std::getenv("STRIX_DS41_DSPARK_WIDE_ROWS")) wide_rows = static_cast<unsigned>(v[0] - '0');
        gpu_routes = flag("STRIX_DS41_DSPARK_GPU_ROUTES");
        if (mcfg.cand) cand_init();
        check(hipDeviceSynchronize(), "upload");
        memlog("route tables + extra rows", gtt_mark, std::uint64_t{extra_capacity} * output.row_bytes);
    }
    std::uint64_t memlog_stages_expected = 0;
    ~Impl() { (void)hipStreamDestroy(stream); }

    // y[rows][outputs] = Q8_0 linear of x (A8 per row), launched in row chunks whose staged rows fit in LDS.
    void q8_linear(const DeviceQ8& w, const float* xin, std::uint32_t rows, float* out) {
        enqueue_quantize_a8(xin, w.width, rows, codes.data, deltas.data, sums.data, stream);
        const auto chunk = row_chunk(w.width);
        if ((chunk < rows && wide_rows == 1U) || (rows > 1U && wide_rows == 2U)) {
            for (std::uint32_t r0 = 0; r0 < rows; r0 += 8U) {
                const std::uint32_t n = std::min(8U, rows - r0);
                hipLaunchKernelGGL(q8_rows_kernel<8>, dim3((w.outputs + 7U) / 8U), dim3(256), 0, stream, w.bytes.data, w.row_bytes, w.width,
                                   w.outputs, reinterpret_cast<const signed char*>(codes.data) + static_cast<std::size_t>(r0) * w.width,
                                   deltas.data + static_cast<std::size_t>(r0) * (w.width / 32U), n,
                                   out + static_cast<std::size_t>(r0) * w.outputs);
            }
            check(hipGetLastError(), "q8 rows launch");
            return;
        }
        for (std::uint32_t r0 = 0; r0 < rows; r0 += chunk) {
            ProductView v;
            v.weights = w.bytes.data; v.expert_stride_bytes = 0; v.row_bytes = w.row_bytes; v.width = w.width; v.outputs = w.outputs;
            v.experts = zero.data; v.expert_count = 1; v.codes = codes.data + static_cast<std::size_t>(r0) * w.width;
            v.deltas = deltas.data + static_cast<std::size_t>(r0) * (w.width / 32U); v.rows = std::min(chunk, rows - r0);
            v.out = out + static_cast<std::size_t>(r0) * w.outputs;
            enqueue_product(WeightFormat::q8_0, v, stream);
        }
    }
    // STRIX_DS41_DSPARK_WIDE_ROWS: 0 = LDS-staged product in row chunks, 1 (default) = the wide-row
    // kernel when the staged rows do not fit in LDS, 2 = the wide-row kernel for every multi-row linear.
    unsigned wide_rows = 1;
    bool gpu_routes = true;     // STRIX_DS41_DSPARK_GPU_ROUTES=0 builds the MoE routes on the host
    std::vector<float> rope_host;
    void upload_rope(std::uint64_t first, std::uint32_t rows) {
        check(hipStreamSynchronize(stream), "rope table reuse");
        auto& t = rope_host;
        t.assign(static_cast<std::size_t>(rows) * rope_dim, 0.0F);
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto row = rope_row(first + r);
            std::copy(row.c.begin(), row.c.end(), t.begin() + static_cast<std::ptrdiff_t>(r) * rope_dim);
            std::copy(row.s.begin(), row.s.end(), t.begin() + static_cast<std::ptrdiff_t>(r) * rope_dim + rope_dim / 2U);
        }
        rope_table.upload(t.data(), t.size(), stream);
    }
    // kv entries of `rows` rows of xin (stage input) at positions first..: into the ring or `linear`.
    void kv_entries(const DeviceStage& s, const float* xin, std::uint32_t rows, std::uint64_t first, std::uint8_t* linear) {
        q8_linear(s.kv, xin, rows, kv.data);
        enqueue_rms_norm(kv.data, rows, head_dim, s.kv_norm.data, kv_normed.data, stream);
        enqueue_rope(kv_normed.data, rows, head_dim, head_dim, rope_table.data, false, stream);
        if (linear) enqueue_fp8_entries(kv_normed.data, rows, linear, 0U, 0U, stream);
        else enqueue_fp8_entries(kv_normed.data, rows, s.ring.data, window, first, stream);
    }
    void append_device(const float* mx, std::uint32_t rows) {
        upload_rope(next, rows);
        for (auto& s : stage) kv_entries(s, mx, rows, next, nullptr);
        next += rows;
    }
    void moe(DeviceStage& s, const float* xin, float* out) {
        constexpr std::uint32_t rows = block_size;
        RouterView rv;
        rv.x = xin; rv.weight = s.router.data; rv.bias = s.bias.data; rv.bias_image = s.bias_image.data; rv.rows = rows; rv.hidden = hidden;
        rv.experts = experts; rv.top_k = top_k; rv.route_scale = route_scale; rv.scores = scores.data; rv.ids = ids.data;
        rv.weights = route_weights.data;
        if (gpu_routes) {
            hipLaunchKernelGGL(draft_router_scores_kernel, dim3((experts * rows + 7U) / 8U), dim3(256), 0, stream, xin, s.router.data, rows,
                               scores.data);
            hipLaunchKernelGGL(draft_router_select_kernel, dim3(rows), dim3(128), 0, stream, scores.data, s.bias.data, ids.data,
                               route_weights.data);
            check(hipGetLastError(), "router launch");
        } else {
            enqueue_router(rv, stream);
        }
        enqueue_quantize_a8(xin, hidden, rows, codes.data, deltas.data, sums.data, stream);
        // Shared expert (Q8_0, expert 0 for every row).
        const auto& st = s.shared.host;
        RoutesView shared_routes{shared_experts.data, shared_begins.data, shared_rows.data, shared_rows.data, 1U, rows};
        GateUpView sgu;
        sgu.gate = s.shared.gate.data; sgu.up = s.shared.up.data; sgu.expert_stride_bytes = st.gate_row_bytes() * st.inter;
        sgu.row_bytes = st.gate_row_bytes(); sgu.hidden = hidden; sgu.inter = st.inter; sgu.routes = shared_routes;
        sgu.codes = codes.data; sgu.deltas = deltas.data; sgu.limit = st.limit;
        sgu.act_codes = shared_codes.data; sgu.act_deltas = shared_deltas.data; sgu.act_sums = shared_sums.data;
        enqueue_gate_up(st.gate_up_format, sgu, stream);
        RoutesProductView sdown;
        sdown.weights = s.shared.down.data; sdown.expert_stride_bytes = st.down_row_bytes() * st.hidden; sdown.row_bytes = st.down_row_bytes();
        sdown.width = st.inter; sdown.outputs = hidden; sdown.routes = shared_routes; sdown.codes = shared_codes.data;
        sdown.deltas = shared_deltas.data; sdown.out = shared_out.data;
        enqueue_routes_product(st.down_format, sdown, stream);
        if (gpu_routes) {
            hipLaunchKernelGGL(build_routes_kernel, dim3(1), dim3(32), 0, stream, ids.data, rows, top_k, experts / 2U, route_slot_rows,
                               route_slots, r_experts.data, r_begins.data, r_rows.data, r_slots.data, r_order.data, r_low.data);
            routed_window(s, route_slots, route_slot_rows, out);
            return;
        }
        // Routes built on the host from the router's ids (one sync).
        const auto host_ids = ids.download(stream);
        const auto routes = build_window_routes(host_ids, rows, top_k, experts / 2U);
        r_experts.upload(routes.experts.data(), routes.experts.size(), stream);
        r_begins.upload(routes.begins.data(), routes.begins.size(), stream);
        r_rows.upload(routes.rows.data(), routes.rows.size(), stream);
        r_slots.upload(routes.slots.data(), routes.slots.size(), stream);
        r_order.upload(routes.order.data(), routes.order.size(), stream);
        r_low.upload(routes.low_counts.data(), routes.low_counts.size(), stream);
        routed_window(s, static_cast<std::uint32_t>(routes.experts.size()), routes.max_rows, out);
        check(hipStreamSynchronize(stream), "moe"); // the host route vectors must outlive the uploads
    }
    void routed_window(DeviceStage& s, std::uint32_t expert_count, std::uint32_t entry_rows, float* out) {
        constexpr std::uint32_t rows = block_size;
        const auto& t = s.routed.host;
        GateUpView gu;
        gu.gate = s.routed.gate.data; gu.up = s.routed.up.data; gu.expert_stride_bytes = t.gate_row_bytes() * t.inter;
        gu.row_bytes = t.gate_row_bytes(); gu.hidden = hidden; gu.inter = t.inter;
        gu.routes = RoutesView{r_experts.data, r_begins.data, r_rows.data, r_slots.data, expert_count, entry_rows};
        gu.codes = codes.data; gu.deltas = deltas.data; gu.limit = t.limit;
        gu.act_codes = act_codes.data; gu.act_deltas = act_deltas.data; gu.act_sums = act_sums.data;
        enqueue_gate_up(t.gate_up_format, gu, stream);
        RoutesProductView down;
        down.weights = s.routed.down.data; down.expert_stride_bytes = t.down_row_bytes() * hidden; down.row_bytes = t.down_row_bytes();
        down.width = t.inter; down.outputs = hidden;
        down.routes = RoutesView{r_experts.data, r_begins.data, r_slots.data, r_slots.data, expert_count, entry_rows};
        down.codes = act_codes.data; down.deltas = act_deltas.data; down.out = raw.data;
        enqueue_routes_product(t.down_format, down, stream);
        CombineView c;
        c.raw = raw.data; c.shared = shared_out.data; c.weights = route_weights.data; c.order = r_order.data; c.low_counts = r_low.data;
        c.rows = rows; c.top_k = top_k; c.hidden = hidden; c.out = out;
        enqueue_combine(c, stream);
    }
    // This rank's vocabulary rows: the domain (core ids, then extras) split in two halves by position.
    struct LocalRows { std::uint32_t core_base, core_count, extra_base, extra_count; };
    LocalRows local_rows() const {
        const auto size = core + static_cast<std::uint32_t>(extra_host.size());
        const std::uint32_t half = (size + 1U) / 2U;
        const std::uint32_t lo = world == 1U ? 0U : rank == 0U ? 0U : half, hi = world == 1U ? size : rank == 0U ? half : size;
        const auto clamp_core = [&](std::uint32_t at) { return at < core ? at : core; };
        const std::uint32_t cb = clamp_core(lo), cc = clamp_core(hi) - cb;
        const std::uint32_t eb = (lo > core ? lo : core) - core, ee = (hi > core ? hi : core) - core;
        return {cb, cc, eb, ee - eb};
    }
    // Grows the packed extra rows (keeping the gathered ones) to hold `needed` ids.
    void reserve_extra(std::uint32_t needed) {
        if (needed <= extra_capacity) return;
        const std::uint32_t cap = needed + 16384U;
        Buffer<std::uint32_t> ids_new(cap);
        Buffer<std::uint8_t> rows_new(static_cast<std::size_t>(cap) * output.row_bytes);
        const auto have = static_cast<std::size_t>(extra_host.size());
        if (have) {
            check(hipMemcpyAsync(ids_new.data, extra_ids.data, have * 4U, hipMemcpyDeviceToDevice, stream), "grow ids");
            check(hipMemcpyAsync(rows_new.data, extra_rows.data, have * output.row_bytes, hipMemcpyDeviceToDevice, stream), "grow rows");
        }
        check(hipStreamSynchronize(stream), "grow");
        extra_ids = std::move(ids_new);
        extra_rows = std::move(rows_new);
        logits_extra = Buffer<float>(static_cast<std::size_t>(block_size) * cap);
        extra_capacity = cap;
    }
    // Appends extra ids (already checked and deduplicated) and gathers their head rows.
    void push_extra(const std::vector<std::uint32_t>& fresh) {
        if (fresh.empty()) return;
        reserve_extra(static_cast<std::uint32_t>(extra_host.size() + fresh.size()));
        const auto first = static_cast<std::uint32_t>(extra_host.size());
        extra_host.insert(extra_host.end(), fresh.begin(), fresh.end());
        for (const auto id : fresh) member[id] = 1U;
        extra_ids_host_stage = fresh;
        check(hipMemcpyAsync(extra_ids.data + first, extra_ids_host_stage.data(), fresh.size() * 4U, hipMemcpyHostToDevice, stream), "extra ids");
        hipLaunchKernelGGL(gather_rows_kernel, dim3(8, static_cast<unsigned>(fresh.size())), dim3(256), 0, stream, output.bytes.data,
                           static_cast<unsigned long long>(output.row_bytes), extra_ids.data, first, extra_rows.data);
        check(hipGetLastError(), "gather launch");
        check(hipStreamSynchronize(stream), "extra rows");
    }
    std::vector<std::uint32_t> extra_ids_host_stage;
    // Head logits of this rank's vocabulary rows, then the Markov chain and the confidence.
    void head(std::uint32_t rows) {
        ProductView v;
        enqueue_quantize_a8(xn.data, hidden, rows, codes.data, deltas.data, sums.data, stream);
        v.weights = output.bytes.data; v.expert_stride_bytes = 0; v.row_bytes = output.row_bytes; v.width = hidden; v.outputs = core;
        v.experts = zero.data; v.expert_count = 1; v.codes = codes.data; v.deltas = deltas.data; v.rows = rows; v.out = logits.data;
        const auto [cb, cc, eb, ec] = local_rows();
        if (cc != 0U) {
            v.weights = output.bytes.data + static_cast<std::size_t>(cb) * output.row_bytes; v.outputs = cc;
            enqueue_product(WeightFormat::q8_0, v, stream);
        }
        if (ec != 0U) {
            v.weights = extra_rows.data + static_cast<std::size_t>(eb) * output.row_bytes; v.outputs = ec; v.out = logits_extra.data;
            enqueue_product(WeightFormat::q8_0, v, stream);
        }
        HeadView h{logits.data, cc, cb, logits_extra.data, ec, eb, extra_ids.data, markov_embed.data, markov_head.data, tokens.data,
                   markov.data, best_values.data, best_ids.data};
        const std::uint32_t full_steps = mcfg.cand && cand_head(rows) ? 0U : rows;  // the candidate chain did them all
        for (std::uint32_t i = 0; i < full_steps; ++i) {
            hipLaunchKernelGGL(markov_pass_kernel, dim3(markov_blocks), dim3(256), 0, stream, h, i);
            hipLaunchKernelGGL(markov_pick_kernel, dim3(1), dim3(markov_blocks), 0, stream, h, i, world > 1U ? candidate.data : nullptr);
            if (world > 1U) {
                exchange(stream, candidate.data, remote_candidate.data, 2U * sizeof(std::uint32_t));
                hipLaunchKernelGGL(markov_combine_kernel, dim3(1), dim3(1), 0, stream, candidate.data, remote_candidate.data, tokens.data, i);
            }
        }
        check(hipGetLastError(), "markov launch");
        enqueue_confidence(confidence.data, x.data, markov.data, rows, confidence_out.data, stream);
    }
    // This rank's logits of the domain positions [lo, hi) (inside local_rows()).
    mc::TwoSegments half_view(std::uint32_t lo, std::uint32_t hi, const LocalRows& l) const {
        const std::uint32_t clo = std::min(lo, core), chi = std::min(hi, core), elo = std::max(lo, core) - core, ehi = std::max(hi, core) - core;
        return {logits.data, l.core_count, clo - l.core_base, chi - clo, clo, logits_extra.data, l.extra_count, elo - l.extra_base, extra_ids.data,
                elo, (chi - clo) + (ehi - elo)};
    }
    // H of the current domain (re-derived when the core or the extras change).
    void cand_h(std::uint32_t half) {
        std::uint64_t key = 14695981039346656037ULL;
        const auto mix = [&](std::uint64_t v) { key = (key ^ v) * 1099511628211ULL; };
        mix(core);
        mix(extra_host.size());
        for (const auto id : extra_host) mix(id);
        key |= 1U;
        if (key == cs.key) return;
        std::vector<std::uint32_t> extra_at;  // extra id -> its index (with extras only)
        if (!extra_host.empty()) {
            extra_at.assign(vocab, 0xffffffffU);
            for (std::uint32_t i = 0; i < extra_host.size(); ++i) extra_at[extra_host[i]] = i;
        }
        for (std::uint32_t h = 0; h < 2U; ++h) cs.hid[h].clear(), cs.hpos[h].clear();
        std::uint32_t found = 0;
        for (const auto id : cs.order) {
            if (found == mcfg.H) break;
            const std::uint32_t pos = id < core ? id : !extra_at.empty() && extra_at[id] != 0xffffffffU ? core + extra_at[id] : 0xffffffffU;
            if (pos == 0xffffffffU) continue;
            const std::uint32_t h = pos < half ? 0U : 1U;
            cs.hid[h].push_back(id);
            cs.hpos[h].push_back(pos - (h == 0U ? 0U : half));
            ++found;
        }
        cs.hmax = static_cast<std::uint32_t>(std::max(cs.hid[0].size(), cs.hid[1].size()));
        for (std::uint32_t h = 0; h < 2U; ++h) {
            cs.d_hid[h].upload(cs.hid[h].data(), cs.hid[h].size(), stream);
            cs.d_hpos[h].upload(cs.hpos[h].data(), cs.hpos[h].size(), stream);
        }
        cs.key = key;
    }
    // Top-C + owned-H payload of one half (h: which half's H).
    void cand_select(const mc::TwoSegments& lg, std::uint32_t rows, std::uint32_t h, unsigned char* pay) {
        const std::uint32_t nb = (lg.n + mc::kCh - 1U) / mc::kCh;
        unsigned char* ws = cs.ws.data;
        hipLaunchKernelGGL(mc::v2_hist_kernel<mc::TwoSegments>, dim3(nb, rows), dim3(mc::kV2Threads), 0, stream, lg, block_size, ws, nb);
        hipLaunchKernelGGL(mc::v2_compact_kernel<mc::TwoSegments>, dim3(nb, rows), dim3(mc::kV2Threads), 0, stream, lg, mcfg.C, block_size, ws, nb);
        hipLaunchKernelGGL(mc::v2_refine_kernel, dim3(rows), dim3(1024), 0, stream, lg.n, mcfg.C, block_size, ws, nb, pay);
        hipLaunchKernelGGL(mc::v2_emit_kernel<mc::TwoSegments>, dim3(nb, rows), dim3(mc::kV2Threads), 0, stream, lg, mcfg.C, block_size,
                           cs.d_hpos[h].data, static_cast<std::uint32_t>(cs.hpos[h].size()), cs.hmax, ws, nb, pay);
    }
    // At construction: the buffers and the H order (Markov-head row norms; one device sync, not in a draft).
    void cand_init() {
        const std::uint32_t n_max = (vocab + 1U) / 2U, nb_max = (n_max + mc::kCh - 1U) / mc::kCh;
        cs.ws = Buffer<unsigned char>(mc::ws_bytes(nb_max, n_max, block_size));
        check(hipMemsetAsync(cs.ws.data, 0, mc::ws_words(nb_max, block_size) * 4U, stream), "markov cand ws");
        for (std::uint32_t h = 0; h < 2U; ++h) {
            cs.pay[h] = Buffer<unsigned char>(mc::payload_bytes(mcfg.C, block_size, mcfg.H));
            cs.d_hid[h] = Buffer<std::uint32_t>(mcfg.H);
            cs.d_hpos[h] = Buffer<std::uint32_t>(mcfg.H);
        }
        cs.cand = Buffer<mc::Cand>((2U * mcfg.C + mcfg.H + 7U) / 8U);
        Buffer<float> row_norms(vocab);
        hipLaunchKernelGGL(mc::norm_kernel<mc::F16>, dim3((vocab + 7U) / 8U), dim3(256), 0, stream, markov_head.data, vocab, row_norms.data);
        check(hipGetLastError(), "markov cand norms");
        const auto n = row_norms.download(stream);
        cs.order.resize(vocab);
        for (std::uint32_t v = 0; v < vocab; ++v) cs.order[v] = v;
        std::sort(cs.order.begin(), cs.order.end(), [&](std::uint32_t ia, std::uint32_t ib) { return n[ia] > n[ib] || (n[ia] == n[ib] && ia < ib); });
    }
    // The chain over the candidates (tokens[1..rows], markov[0..rows)). false: the full passes run - a half holds fewer
    // than C positions, or a row of either half has a nonfinite logit (its count is not C). Both ranks decide alike:
    // same domain, and the counts of both halves are in both ranks' payloads.
    bool cand_head(std::uint32_t rows) {
        const auto size = core + static_cast<std::uint32_t>(extra_host.size()), half = (size + 1U) / 2U;
        if (size - half < mcfg.C) return false;
        cand_h(half);
        const auto l = local_rows();
        unsigned char *own = cs.pay[0].data, *peer = cs.pay[1].data;
        const std::uint32_t mine = world == 1U ? 0U : rank, other = 1U - mine;
        if (world == 1U) {
            cand_select(half_view(0U, half, l), rows, 0U, own);
            cand_select(half_view(half, size, l), rows, 1U, peer);
        } else {
            cand_select(rank == 0U ? half_view(0U, half, l) : half_view(half, size, l), rows, mine, own);
        }
        check(hipGetLastError(), "markov cand select");
        if (world > 1U) exchange(stream, own, peer, mc::payload_bytes(mcfg.C, block_size, cs.hmax));
        const std::size_t counts_at = std::size_t{block_size} * mcfg.C * 4U;  // Payload::counts
        check(hipMemcpyAsync(cs.counts.data(), own + counts_at, block_size * 4U, hipMemcpyDeviceToHost, stream), "markov cand counts");
        check(hipMemcpyAsync(cs.counts.data() + block_size, peer + counts_at, block_size * 4U, hipMemcpyDeviceToHost, stream), "markov cand counts");
        check(hipStreamSynchronize(stream), "markov cand counts");
        for (std::uint32_t k = 0; k < rows; ++k)
            if (cs.counts[k] != mcfg.C || cs.counts[block_size + k] != mcfg.C) return false;
        const auto nh_own = static_cast<std::uint32_t>(cs.hid[mine].size()), nh_peer = static_cast<std::uint32_t>(cs.hid[other].size());
        const std::uint32_t nb = (2U * mcfg.C + nh_own + nh_peer + 7U) / 8U;
        for (std::uint32_t k = 0; k < rows; ++k) {
            hipLaunchKernelGGL((mc::cand_kernel<mc::F16>), dim3(nb), dim3(256), 0, stream, markov_embed.data, markov_head.data, tokens.data + k,
                               k, block_size, own, peer, mcfg.C, cs.d_hid[mine].data, nh_own, cs.d_hid[other].data, nh_peer, cs.hmax, cs.cand.data);
            hipLaunchKernelGGL(mc::pick_kernel, dim3(1), dim3(256), 0, stream, cs.cand.data, nb, nullptr, tokens.data + k + 1U);
        }
        hipLaunchKernelGGL(markov_rows_kernel, dim3(rows), dim3(256), 0, stream, markov_embed.data, tokens.data, markov.data);
        check(hipGetLastError(), "markov cand chain");
        return true;
    }
    void output_projection(DeviceStage& s, float* out) {
        constexpr std::uint32_t rows = block_size;
        enqueue_quantize_a8(o.data, o_group_width, rows * o_groups, codes.data, deltas.data, sums.data, stream);
        RoutesProductView v;
        v.weights = s.out_a.bytes.data; v.expert_stride_bytes = s.out_a.row_bytes * o_rank; v.row_bytes = s.out_a.row_bytes;
        v.width = o_group_width; v.outputs = o_rank;
        v.routes = RoutesView{wo_experts.data, wo_begins.data, wo_entries.data, wo_entries.data, o_groups, rows};
        v.codes = codes.data; v.deltas = deltas.data; v.out = low.data;
        enqueue_routes_product(WeightFormat::q8_0, v, stream);
        q8_linear(s.out_b, low.data, rows, out);
    }
};

DeviceDrafter::DeviceDrafter(const Weights& weights) : impl_(std::make_unique<Impl>(weights)) {}
DeviceDrafter::~DeviceDrafter() = default;
std::uint64_t DeviceDrafter::next() const noexcept { return impl_->next; }
void DeviceDrafter::set_vocabulary(const Vocab& vocabulary) {
    auto& m = *impl_;
    if (vocabulary.core > vocab || vocabulary.size() == 0U) throw std::invalid_argument("ds41 dspark draft vocabulary");
    std::vector<std::uint8_t> seen(vocab, 0U);
    for (const auto id : vocabulary.extra) {
        if (id < vocabulary.core || id >= vocab || seen[id]) throw std::invalid_argument("ds41 dspark draft vocabulary: bad extra id");
        seen[id] = 1U;
    }
    m.core = vocabulary.core;
    m.extra_host.clear();
    std::fill(m.member.begin(), m.member.end(), 0U);
    m.push_extra(vocabulary.extra);
}
std::size_t DeviceDrafter::add_vocabulary(std::span<const std::uint32_t> ids) {
    auto& m = *impl_;
    std::vector<std::uint32_t> fresh;
    for (const auto id : ids)
        if (id < vocab && id >= m.core && !m.member[id]) {
            m.member[id] = 1U;
            fresh.push_back(id);
        }
    for (const auto id : fresh) m.member[id] = 0U; // push_extra sets them again
    m.push_extra(fresh);
    return fresh.size();
}
void DeviceDrafter::set_rank(std::uint32_t rank, std::uint32_t world, RankExchange exchange) {
    if (world == 0U || world > 2U || rank >= world || (world == 2U && !exchange)) throw std::invalid_argument("ds41 dspark rank");
    impl_->rank = rank;
    impl_->world = world;
    impl_->exchange = std::move(exchange);
}
Vocab DeviceDrafter::vocabulary() const { return Vocab{impl_->core, impl_->extra_host}; }
DeviceDrafter::MarkovConfig DeviceDrafter::markov_config() {
    MarkovConfig c;
    const char* m = std::getenv("STRIX_DS41_DSPARK_MARKOV");
    const std::string mode = m != nullptr && *m != '\0' ? m : "cand";
    if (mode != "full" && mode != "cand") throw std::invalid_argument("STRIX_DS41_DSPARK_MARKOV must be full or cand");
    c.cand = mode == "cand";
    const auto number = [](const char* name, std::uint32_t fallback, std::uint32_t most) {
        const char* v = std::getenv(name);
        if (v == nullptr || *v == '\0') return fallback;
        char* end = nullptr;
        const unsigned long n = std::strtoul(v, &end, 10);
        if (*end != '\0' || n == 0UL || n > most) throw std::invalid_argument(std::string(name) + " must be 1.." + std::to_string(most));
        return static_cast<std::uint32_t>(n);
    };
    c.C = number("STRIX_DS41_DSPARK_MARKOV_C", c.C, (vocab + 1U) / 2U);  // a half's size: every position a candidate (= full)
    c.H = number("STRIX_DS41_DSPARK_MARKOV_H", c.H, vocab);
    return c;
}
DeviceDrafter::MarkovConfig DeviceDrafter::markov() const noexcept { return impl_->mcfg; }
ContextSnapshot DeviceDrafter::export_context() const {
    if (sequences() != 1U) throw std::logic_error("DSpark export requires a single context");
    const auto& m = *impl_;
    check(hipStreamSynchronize(m.stream), "export context");
    ContextSnapshot out;
    out.start = m.start; out.next = m.next; out.core = m.core; out.extra = m.extra_host;
    for (std::uint32_t i = 0; i < stages; ++i) out.rings[i] = m.stage[i].ring.download(m.stream);
    out.validate();
    return out;
}
void DeviceDrafter::import_context(const ContextSnapshot& in) {
    in.validate(); // CPU validation before any device mutation
    if (sequences() != 1U) throw std::logic_error("DSpark import requires a single context");
    auto& m = *impl_;
    check(hipStreamSynchronize(m.stream), "import context");
    set_vocabulary(Vocab{in.core, in.extra}); // reconstruct membership, ids and gathered head rows
    for (std::uint32_t i = 0; i < stages; ++i)
        m.stage[i].ring.upload(in.rings[i].data(), in.rings[i].size(), m.stream);
    check(hipStreamSynchronize(m.stream), "import rings");
    m.start = in.start; m.next = in.next;
}
void DeviceDrafter::reset() {
    for (auto& s : impl_->stage) check(hipMemsetAsync(s.ring.data, 0, s.ring.count, impl_->stream), "clear ring");
    check(hipStreamSynchronize(impl_->stream), "reset");
    impl_->next = 0;
    impl_->start = 0;
}
void DeviceDrafter::skip_to(std::uint64_t position) {
    if (impl_->next != 0U) throw std::logic_error("ds41 dspark: skip_to needs an empty context");
    impl_->next = position;
    impl_->start = position;
}
std::vector<std::uint8_t> DeviceDrafter::ring(std::uint32_t s) const { return impl_->stage[s].ring.download(impl_->stream); }
void DeviceDrafter::set_sequences(std::uint32_t count) {
    auto& m = *impl_;
    if (count == 0U) throw std::invalid_argument("ds41 dspark: at least one sequence context");
    check(hipStreamSynchronize(m.stream), "sequences");
    if (m.active != 0U) select(0U);
    m.parked.resize(1U);  // context 0 is the active one; contexts beyond `count` are dropped
    while (m.parked.size() < count) m.parked.push_back(m.fresh_context());
    check(hipStreamSynchronize(m.stream), "sequences");
    if (memlog_on()) {
        const auto one = std::uint64_t{stages} * window * entry_bytes + 8192U * 4U + 8192U * m.output.row_bytes + std::uint64_t{block_size} * 8192U * 4U;
        memlog("sequence contexts", m.gtt_mark, one * (count - 1U));
    }
}
DeviceDrafter::Footprint DeviceDrafter::footprint(const Weights& w, std::uint32_t sequences) {
    // keep in step with Impl (its members and constructor) and Impl::fresh_context
    Footprint f;
    const auto q8 = [](const Q8Matrix& m) { return static_cast<std::uint64_t>(m.bytes.size()); };
    const auto f32 = [](std::span<const float> v) { return static_cast<std::uint64_t>(v.size()) * 4U; };
    const auto f16 = [](const F16Matrix& m) { return static_cast<std::uint64_t>(m.bits.size()) * 2U; };
    f.weights = q8(w.main_proj) + q8(w.output) + f32(w.main_norm) + f32(w.final_norm) + f32(w.confidence) + f16(w.markov_embed) +
                f16(w.markov_head) + f16(w.token_embd);
    for (const auto& h : w.stage)
        f.weights += f16(h.hc_attn_fn) + f16(h.hc_ffn_fn) + f32(h.hc_attn_scale) + f32(h.hc_attn_base) + f32(h.hc_ffn_scale) + f32(h.hc_ffn_base) +
                     f32(h.attn_norm) + f32(h.ffn_norm) + f32(h.sinks) + f32(h.q_norm) + f32(h.kv_norm) + q8(h.q_a) + q8(h.q_b) + q8(h.kv) +
                     q8(h.out_a) + q8(h.out_b) + f32(h.router) + f32(h.bias) + f32(h.bias_image) + h.routed.gate.size() + h.routed.up.size() +
                     h.routed.down.size() + h.shared.gate.size() + h.shared.up.size() + h.shared.down.size();
    using u64 = std::uint64_t;
    const u64 R = Impl::max_rows, B = block_size, K = top_k, slots = Impl::route_slots;
    u64 e4 = 0, e1 = 0;  // 4-byte and 1-byte elements of the scratch buffers
    e4 += R * main_width + 2U * R * hidden + 2U * (R * main_width / 32U) + 2U * R * head_dim + R * rope_dim + 1U;  // x_main .. zero
    e1 += R * main_width;                                                                                          // codes
    e4 += (B + 1U) + 2U * B * stream_width + 4U * B * mix_count + B * (mix_count + 1U) + 2U * B * hidden + 2U * B * q_rank + 2U * B * q_width +
          B * o_width + B * hidden + u64{max_keys} * head_dim + o_groups + (o_groups + 1U) + u64{o_groups} * B;      // tokens .. wo_entries
    e1 += B * entry_bytes;                                                                                         // block_entries
    e4 += B * experts + 2U * B * K + 2U * (B * K * inter / 32U) + 2U * (B * inter / 32U) + B * K * hidden + B * hidden + 1U + 2U + B;
    e1 += B * K * inter + B * inter;                                                                               // act / shared codes
    e4 += slots + (slots + 1U) + 3U * B * K + B;                                                                   // device routes
    e4 += B * vocab + 2U + 2U + B * markov_rank + 2U * u64{markov_blocks} + B;                                     // logits .. confidence_out
    if (const auto mk = markov_config(); mk.cand) {  // Impl::cand_head's buffers
        const u64 n = (vocab + 1U) / 2U, nb = (n + mc::kCh - 1U) / mc::kCh;
        e1 += mc::ws_bytes(static_cast<std::uint32_t>(nb), static_cast<std::uint32_t>(n), block_size) + 2U * mc::payload_bytes(mk.C, block_size, mk.H);
        e4 += 4U * u64{mk.H} + 2U * ((2U * u64{mk.C} + mk.H + 7U) / 8U);
    }
    f.scratch = 4U * e4 + e1;
    // a context: the stage rings, the extra vocabulary ids / head rows and their logits (the active one lives in Impl)
    const u64 context = u64{stages} * window * entry_bytes + 8192U * 4U + 8192U * w.output.row_bytes() + B * 8192U * 4U;
    f.contexts = context * std::max<std::uint32_t>(sequences, 1U);
    return f;
}
std::uint32_t DeviceDrafter::sequences() const noexcept { return impl_->parked.empty() ? 1U : static_cast<std::uint32_t>(impl_->parked.size()); }
void DeviceDrafter::select(std::uint32_t sequence) {
    auto& m = *impl_;
    if (sequence >= sequences()) throw std::invalid_argument("ds41 dspark: no such sequence context");
    if (sequence == m.active) return;
    // the kernels already enqueued keep the device pointers they were given; later work on the stream is ordered
    m.swap_context(m.parked[m.active]);
    m.swap_context(m.parked[sequence]);
    m.active = sequence;
}
std::uint32_t DeviceDrafter::selected() const noexcept { return impl_->active; }

std::vector<float> DeviceDrafter::main_rows(std::span<const float> main_hidden, std::uint32_t rows) {
    auto& m = *impl_;
    if (rows == 0U || rows > Impl::max_rows || main_hidden.size() != static_cast<std::size_t>(rows) * main_width)
        throw std::invalid_argument("ds41 dspark main rows geometry");
    m.x_main.upload(main_hidden.data(), main_hidden.size(), m.stream);
    m.q8_linear(m.main_proj, m.x_main.data, rows, m.proj.data);
    enqueue_rms_norm(m.proj.data, rows, hidden, m.main_norm.data, m.main_x.data, m.stream);
    return m.main_x.download(m.stream, static_cast<std::size_t>(rows) * hidden);
}
void DeviceDrafter::append(std::span<const float> main_x, std::uint32_t rows) {
    auto& m = *impl_;
    if (rows == 0U || rows > Impl::max_rows || main_x.size() != static_cast<std::size_t>(rows) * hidden)
        throw std::invalid_argument("ds41 dspark append geometry");
    m.main_x.upload(main_x.data(), main_x.size(), m.stream);
    m.append_device(m.main_x.data, rows);
    check(hipStreamSynchronize(m.stream), "append");
}
void DeviceDrafter::append_hidden(std::span<const float> main_hidden, std::uint32_t rows) {
    auto& m = *impl_;
    if (rows == 0U || rows > Impl::max_rows || main_hidden.size() != static_cast<std::size_t>(rows) * main_width)
        throw std::invalid_argument("ds41 dspark append geometry");
    m.x_main.upload(main_hidden.data(), main_hidden.size(), m.stream);
    m.q8_linear(m.main_proj, m.x_main.data, rows, m.proj.data);
    enqueue_rms_norm(m.proj.data, rows, hidden, m.main_norm.data, m.main_x.data, m.stream);
    m.append_device(m.main_x.data, rows);
    check(hipStreamSynchronize(m.stream), "append");
}

void DeviceDrafter::append_hidden_device(const float* main_hidden, std::uint32_t rows, void* stream) {
    auto& m = *impl_;
    if (rows == 0U || rows > Impl::max_rows || !main_hidden) throw std::invalid_argument("ds41 dspark append geometry");
    if (stream) check(hipStreamSynchronize(as_stream(stream)), "producer");
    else check(hipDeviceSynchronize(), "producer");
    m.q8_linear(m.main_proj, main_hidden, rows, m.proj.data);
    enqueue_rms_norm(m.proj.data, rows, hidden, m.main_norm.data, m.main_x.data, m.stream);
    m.append_device(m.main_x.data, rows);
    check(hipStreamSynchronize(m.stream), "append");
}

Draft DeviceDrafter::draft(std::uint32_t token, bool keep_logits, DraftTimes* times) {
    using clock = std::chrono::steady_clock;
    auto& m = *impl_;
    constexpr std::uint32_t rows = block_size;
    if (token >= vocab) throw std::invalid_argument("ds41 dspark draft token");
    const auto start = clock::now();
    const std::uint64_t first = m.next, oldest = std::max(first > window ? first - window : std::uint64_t{0}, m.start);
    const auto ring_keys = static_cast<std::uint32_t>(first - oldest);
    std::uint32_t block_tokens[block_size + 1U] = {token, noise_token, noise_token, noise_token, noise_token, 0U};
    m.tokens.upload(block_tokens, block_size + 1U, m.stream);
    m.upload_rope(first, rows);
    enqueue_embed_block(m.token_embd.data, m.tokens.data, rows, m.streams.data, m.stream);
    const float* pre = m.identity.data;
    Draft out;
    auto stage_start = clock::now();
    for (std::uint32_t si = 0; si < stages; ++si) {
        auto& s = m.stage[si];
        enqueue_hc_mixes(m.streams.data, rows, s.hc_attn_fn.data, s.hc_attn_scale.data, s.hc_attn_base.data, m.mix_scratch.data, m.attn_mix.data,
                         m.stream);
        enqueue_hc_pre(m.streams.data, rows, pre, mix_count, m.x.data, m.stream);
        enqueue_rms_norm(m.x.data, rows, hidden, s.attn_norm.data, m.xn.data, m.stream);
        m.q8_linear(s.q_a, m.xn.data, rows, m.qa.data);
        enqueue_rms_norm(m.qa.data, rows, q_rank, s.q_norm.data, m.qan.data, m.stream);
        m.q8_linear(s.q_b, m.qan.data, rows, m.q.data);
        enqueue_rope(m.q.data, rows, q_width, head_dim, m.rope_table.data, false, m.stream);
        m.kv_entries(s, m.xn.data, rows, first, m.block_entries.data);
        enqueue_block_attention(m.q.data, rows, s.ring.data, ring_keys, oldest, m.block_entries.data, s.sinks.data, m.key_scratch.data, m.o.data,
                                m.stream);
        enqueue_rope(m.o.data, rows, q_width, head_dim, m.rope_table.data, true, m.stream);
        m.output_projection(s, m.y.data);
        enqueue_hc_post(m.y.data, m.streams.data, rows, m.attn_mix.data, m.streams2.data, m.stream);
        enqueue_hc_mixes(m.streams2.data, rows, s.hc_ffn_fn.data, s.hc_ffn_scale.data, s.hc_ffn_base.data, m.mix_scratch.data, m.ffn_mix.data,
                         m.stream);
        enqueue_hc_pre(m.streams2.data, rows, m.attn_mix.data, mix_count, m.x.data, m.stream);
        enqueue_rms_norm(m.x.data, rows, hidden, s.ffn_norm.data, m.xn.data, m.stream);
        m.moe(s, m.xn.data, m.y.data);
        enqueue_hc_post(m.y.data, m.streams2.data, rows, m.ffn_mix.data, m.streams.data, m.stream);
        // The next stage's attention collapses with this FFN's pre coefficients.
        check(hipMemcpyAsync(m.pre_mix.data, m.ffn_mix.data, m.ffn_mix.count * sizeof(float), hipMemcpyDeviceToDevice, m.stream), "pre mix");
        pre = m.pre_mix.data;
        if (keep_logits) out.stage_out[si] = m.streams.download(m.stream);
        if (times) {
            check(hipStreamSynchronize(m.stream), "stage");
            const auto now = clock::now();
            times->stages_ms[si] = std::chrono::duration<double, std::milli>(now - stage_start).count();
            stage_start = now;
        }
    }
    const auto head_start = clock::now();
    // Head: hc_pre with the last stage's FFN pre coefficients, final norm, target output head, Markov chain.
    enqueue_hc_pre(m.streams.data, rows, pre, mix_count, m.x.data, m.stream);
    enqueue_rms_norm(m.x.data, rows, hidden, m.final_norm.data, m.xn.data, m.stream);
    m.head(rows);
    const auto tokens = m.tokens.download(m.stream);
    const auto conf = m.confidence_out.download(m.stream);
    std::copy(tokens.begin(), tokens.end(), out.tokens.begin());
    std::copy(conf.begin(), conf.end(), out.confidence.begin());
    const auto local = m.local_rows();
    out.candidates.fill(local.core_count + local.extra_count);
    if (keep_logits) {
        // This rank's rows (all rows with one rank), core part then extras.
        const auto cc = local.core_count, ec = local.extra_count, size = cc + ec;
        const auto core_logits = m.logits.download(m.stream, static_cast<std::size_t>(rows) * cc);
        const auto extra_logits = m.logits_extra.download(m.stream, static_cast<std::size_t>(rows) * ec);
        out.logits.resize(static_cast<std::size_t>(rows) * size);
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::copy_n(core_logits.begin() + static_cast<std::ptrdiff_t>(r) * cc, cc, out.logits.begin() + static_cast<std::ptrdiff_t>(r) * size);
            std::copy_n(extra_logits.begin() + static_cast<std::ptrdiff_t>(r) * ec, ec,
                        out.logits.begin() + static_cast<std::ptrdiff_t>(r) * size + cc);
        }
        out.head_in = m.x.download(m.stream, static_cast<std::size_t>(rows) * hidden);
        out.markov = m.markov.download(m.stream);
    }
    if (times) {
        const auto end = clock::now();
        times->head_ms = std::chrono::duration<double, std::milli>(end - head_start).count();
        times->total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }
    return out;
}

} // namespace strix::ds41::dspark
