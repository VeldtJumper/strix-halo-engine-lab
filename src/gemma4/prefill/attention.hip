#include "device.hpp"
#include "attention.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <string_view>

namespace strix::gemma4::prefill {
namespace {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using Half8 = _Float16 __attribute__((ext_vector_type(8)));
__device__ _Float16 half(float x, unsigned* failure) {
    const auto h = static_cast<_Float16>(x);
    if ((bit_cast<unsigned short>(h) & 0x7c00U) == 0x7c00U) atomicOr(failure, 4U);
    return h;
}
__device__ void finite(float value, unsigned* failure) {
    if ((bit_cast<unsigned>(value) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
}
__device__ const float* at(const gpu::KvRead& r, std::uint64_t p, bool value, unsigned* failure) {
    if (p >= r.frontier) {
        if (p - r.frontier >= r.candidate_rows) { atomicOr(failure, 2U); return nullptr; }
        return (value ? r.overlay_v : r.overlay_k) + std::size_t(p - r.frontier) * r.width;
    }
    const auto slot = r.global ? p : p % r.slots;
    if (slot >= r.slots || r.tags[slot] != p) { atomicOr(failure, 2U); return nullptr; }
    return (value ? r.v : r.k) + std::size_t(slot) * r.width;
}
__global__ void pack(const gpu::KvRead* reads, unsigned rows, unsigned width, unsigned stride,
                     _Float16* packed, unsigned* failure) {
    const auto r = reads[0];
    const auto first = !r.global && r.position >= r.window ? r.position - r.window + 1 : 0;
    const auto last = r.position + rows - 1;
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (last >= stride || r.width != width) { if (!index) atomicOr(failure, 2U); return; }
    if (index >= (last - first + 1) * width) return;
    const auto p = first + index / width;
    const auto d = unsigned(index % width);
    const auto* k = at(r, p, false, failure);
    const auto* v = at(r, p, true, failure);
    packed[std::size_t(p) * width + d] = half(k ? k[d] : 0.0F, failure);
    packed[(std::size_t(stride) + p) * width + d] = half(v ? v[d] : 0.0F, failure);
}
// One workgroup owns sixteen adjacent queries of one Q head. Only wave zero
// computes QK; all four waves reuse its probabilities for disjoint output D16
// tiles. Every output keeps its own accumulator through increasing key tiles.
template <unsigned Dim>
__global__ __launch_bounds__(128) void flash(const float* q, const gpu::KvRead* reads, unsigned rows,
                                            unsigned head_count, float* out, unsigned* failure,
                                            const _Float16* packed, unsigned packed_stride) {
    constexpr unsigned padded = (Dim + 15U) / 16U * 16U;
    __shared__ _Float16 query[padded * 16], keys[padded * 16], values[16 * Dim], probability[16 * 16];
    __shared__ float scores[16 * 16], maximum[16], denominator[16], alpha[16];
    const unsigned tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const unsigned row0 = blockIdx.x * 16, head = blockIdx.y;
    const auto r = reads[row0];
    if (r.position >= max_positions || rows - row0 > max_positions - r.position) {
        if (!tid) atomicOr(failure, 2U);
        return;
    }
    const auto position0 = unsigned(r.position);
    const auto last = position0 + min(15U, rows - row0 - 1);
    if (r.width != head_count / group * Dim || (packed && last >= packed_stride)) {
        if (!tid) atomicOr(failure, 2U);
        return;
    }
    const auto first = !r.global && position0 >= r.window ? position0 - r.window + 1 : 0;
    for (unsigned i = tid; i < Dim * 16; i += 128) {
        const auto d = i / 16, row = i % 16;
        query[i] = half(row0 + row < rows ? q[(std::size_t(row0 + row) * head_count + head) * Dim + d] : 0.0F, failure);
    }
    for (unsigned i = Dim * 16 + tid; i < padded * 16; i += 128) query[i] = _Float16(0);
    if (tid < 16) { maximum[tid] = bit_cast<float>(0xff800000U); denominator[tid] = 0.0F; }
    constexpr unsigned tiles = (Dim + 63) / 64;
    Float8 accum[tiles]{};
    __syncthreads();
    for (auto base = first / 16 * 16; base <= last; base += 16) {
        for (unsigned i = tid; i < Dim * 16; i += 128) {
            const auto d = i / 16, j = i % 16;
            const auto p = base + j;
            const bool valid = p >= first && p <= last;
            if (packed) {
                const auto offset = std::size_t(p) * r.width + (head / group) * Dim + d;
                keys[i] = valid ? packed[offset] : _Float16(0);
                values[j * Dim + d] = valid ? packed[std::size_t(packed_stride) * r.width + offset] : _Float16(0);
            } else {
                const auto* k = valid ? at(r, p, false, failure) : nullptr;
                const auto* v = valid ? at(r, p, true, failure) : nullptr;
                keys[i] = half(k ? k[(head / group) * Dim + d] : 0.0F, failure);
                values[j * Dim + d] = half(v ? v[(head / group) * Dim + d] : 0.0F, failure);
            }
        }
        for (unsigned i = Dim * 16 + tid; i < padded * 16; i += 128) keys[i] = _Float16(0);
        __syncthreads();
        if (!wave) {
            Float8 score{};
#pragma unroll 1
            for (unsigned d0 = 0; d0 < Dim; d0 += 16) {
                Half16 a, b;
#pragma unroll
                for (unsigned d = 0; d < 16; ++d) {
                    a[d] = query[(d0 + d) * 16 + lane % 16]; b[d] = keys[(d0 + d) * 16 + lane % 16];
                }
                score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, score);
            }
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) scores[(2 * e + lane / 16) * 16 + lane % 16] = score[e];
        }
        __syncthreads();
        if (tid < 16) {
            const unsigned row = row0 + tid;
            const auto position = position0 + tid;
            const auto lo = !r.global && position >= r.window ? position - r.window + 1 : 0;
            float m = maximum[tid];
            bool present = false;
            for (unsigned j = 0; j < 16; ++j) if (row < rows && base + j >= lo && base + j <= position) {
                const float s = scores[tid * 16 + j]; finite(s, failure); m = s > m ? s : m; present = true;
            }
            // Empty leading/trailing tiles do not introduce an extra WMMA:
            // use alpha=1 and skip the instruction for those query rows below.
            const float a = !present ? 1.0F : denominator[tid] == 0.0F ? 0.0F : declared_exp(maximum[tid] - m);
            alpha[tid] = a;
            float sum = denominator[tid] * a;
            for (unsigned j = 0; j < 16; ++j) {
                float p = 0.0F;
                if (row < rows && base + j >= lo && base + j <= position) {
                    p = declared_exp(scores[tid * 16 + j] - m); sum = sum + p;
                }
                probability[j * 16 + tid] = half(p, failure);
            }
            maximum[tid] = m; denominator[tid] = sum; finite(sum, failure);
        }
        __syncthreads();
        Half16 a;
#pragma unroll
        for (unsigned j = 0; j < 16; ++j) a[j] = probability[j * 16 + lane % 16];
        constexpr unsigned tile_unroll = Dim == 256 ? 1 : 8;
#pragma unroll tile_unroll
        for (unsigned t = 0; t < tiles; ++t) {
            const unsigned d = (wave * tiles + t) * 16 + lane % 16;
            Half16 b;
#pragma unroll
            for (unsigned j = 0; j < 16; ++j) b[j] = d < Dim ? values[j * Dim + d] : _Float16(0);
            Half16 positive, negative;
#pragma unroll
            for (unsigned j = 0; j < 16; ++j) {
                const auto bits = bit_cast<unsigned short>(b[j]);
                positive[j] = bit_cast<_Float16>(static_cast<unsigned short>(bits & 0x8000U ? 0U : bits));
                negative[j] = bit_cast<_Float16>(static_cast<unsigned short>(bits & 0x8000U ? bits & 0x7fffU : 0U));
            }
            Float8 zero{};
            const auto pos = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, positive, zero);
            const auto neg = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, negative, zero);
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const unsigned row = row0 + 2 * e + lane / 16;
                const auto position = position0 + 2 * e + lane / 16;
                const auto lo = !r.global && position >= r.window ? position - r.window + 1 : 0;
                // A zero matrix instruction can change bits; exclude absent
                // tiles per row after execution, preserving the frozen recipe.
                if (row < rows && base <= position && base + 15 >= lo) {
                    const float part = pos[e] - neg[e];
                    accum[t][e] = accum[t][e] * alpha[2 * e + lane / 16] + part;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned t = 0; t < tiles; ++t) {
        const unsigned d = (wave * tiles + t) * 16 + lane % 16;
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const unsigned row = row0 + 2 * e + lane / 16;
            if (row < rows && d < Dim) {
                const float value = accum[t][e] / denominator[2 * e + lane / 16];
                finite(value, failure); out[(std::size_t(row) * head_count + head) * Dim + d] = value;
            }
        }
    }
}
// Layout of the split packed operands, each region split_stride(stride) keys:
// FP16 K rows [key][width], then V transposed per component and already
// divided into its nonnegative positive and negated-negative halves,
// [width][key] each. Keys of the K16 tiles that hold [first, last] but lie
// outside it are written as zeros, so every operand a tile reads is finite.
__host__ __device__ constexpr unsigned split_stride(unsigned stride) { return (stride + 15U) / 16U * 16U; }
__global__ __launch_bounds__(256) void pack_split(const gpu::KvRead* reads, unsigned rows, unsigned width, unsigned stride,
                                                  _Float16* packed, unsigned* failure) {
    __shared__ unsigned short positive[32][33], negative[32][33];
    const auto r = reads[0];
    const auto first = !r.global && r.position >= r.window ? r.position - r.window + 1 : 0;
    const auto last = r.position + rows - 1;
    const unsigned tx = threadIdx.x % 32, ty = threadIdx.x / 32;
    if (last >= stride || r.width != width) { if (!threadIdx.x && !blockIdx.x && !blockIdx.y) atomicOr(failure, 2U); return; }
    const auto begin = first / 16 * 16, end = last / 16 * 16 + 15;
    const auto p0 = begin + std::uint64_t(blockIdx.x) * 32;
    const unsigned d0 = blockIdx.y * 32U;
    if (p0 > end) return;
    const auto vstride = split_stride(stride);
    auto* vpos = packed + std::size_t(vstride) * width;
    auto* vneg = vpos + std::size_t(vstride) * width;
    for (unsigned i = 0; i < 4; ++i) {
        const auto p = p0 + ty + 8 * i;
        const auto d = d0 + tx;
        if (p > end || d >= width) continue;
        unsigned short bits = 0;
        _Float16 key = _Float16(0);
        if (p >= first && p <= last) {
            const auto* k = at(r, p, false, failure);
            const auto* v = at(r, p, true, failure);
            key = half(k ? k[d] : 0.0F, failure);
            bits = bit_cast<unsigned short>(half(v ? v[d] : 0.0F, failure));
        }
        packed[std::size_t(p) * width + d] = key;
        positive[tx][ty + 8 * i] = bits & 0x8000U ? 0U : bits;
        negative[tx][ty + 8 * i] = bits & 0x8000U ? bits & 0x7fffU : 0U;
    }
    __syncthreads();
    for (unsigned i = 0; i < 4; ++i) {
        const auto d = d0 + ty + 8 * i;
        const auto p = p0 + tx;
        if (p > end || d >= width) continue;
        vpos[std::size_t(d) * vstride + p] = bit_cast<_Float16>(positive[ty + 8 * i][tx]);
        vneg[std::size_t(d) * vstride + p] = bit_cast<_Float16>(negative[ty + 8 * i][tx]);
    }
}
__device__ inline bool nonfinite(float value) { return (bit_cast<unsigned>(value) & 0x7f800000U) == 0x7f800000U; }
__device__ inline bool overflowed(_Float16 h) { return (bit_cast<unsigned short>(h) & 0x7c00U) == 0x7c00U; }
// The flash recurrence reorganized for throughput, same instructions and order
// per row: each wave owns the score chain of its K16 tiles and the row softmax
// in registers (lane%16 is the query row; both half-waves compute identical
// values), reading FP16 K rows and the pre-split transposed V straight from the
// packed cache. Masks are selects rather than branches, and keys outside a
// row's range contribute +0 products exactly as zeroed operands did. For D512
// two waves own disjoint output halves; they alternate key tiles for the score
// chain and both replay the identical row recurrence.
template <unsigned Dim, unsigned Group = 2, unsigned Waves = (Dim > 256 ? 2U : 1U)>
__global__ __launch_bounds__(256) void flash_split(const float* q, const gpu::KvRead* reads, unsigned rows,
                                                                  unsigned head_count, float* out, unsigned* failure,
                                                                  const _Float16* packed, unsigned packed_stride) {
    // Query heads of one KV group share a workgroup (four for D256, two for
    // D512 within the LDS budget) so their identical K/V tile loads coincide.
    constexpr unsigned padded = (Dim + 15U) / 16U * 16U, waves = Waves, shared = Dim > 256 ? 2U : 4U;
    constexpr unsigned tiles = padded / 16U / waves, qstride = padded + 8U;
    __shared__ _Float16 queries[shared][16 * qstride];
    __shared__ float tiles_scores[shared][waves][16 * 17];
    const unsigned tid = threadIdx.x, lane = tid % 32, member = tid / 32 / waves, wave = tid / 32 % waves, rr = lane % 16;
    auto* query = queries[member];
    auto* scores = tiles_scores[member];
    // Heads vary fastest so the four query heads of a KV group (and both
    // groups) read the same K/V tiles together; the longest row blocks start first.
    const unsigned head = blockIdx.x * shared + member, row0 = (gridDim.y - 1U - blockIdx.y) * 16;
    const auto r = reads[row0];
    if (r.position >= max_positions || rows - row0 > max_positions - r.position) {
        if (!tid) atomicOr(failure, 2U);
        return;
    }
    const auto position0 = unsigned(r.position);
    const auto last = position0 + min(15U, rows - row0 - 1);
    const unsigned width = head_count / group * Dim, kv = (head / group) * Dim;
    if (r.width != width || last >= packed_stride) {
        if (!tid) atomicOr(failure, 2U);
        return;
    }
    const auto first = !r.global && position0 >= r.window ? position0 - r.window + 1 : 0;
    for (unsigned i = tid; i < shared * 16 * padded; i += 32 * waves * shared) {
        const auto h = i / (16 * padded), row = i / padded % 16, d = i % padded, source = blockIdx.x * shared + h;
        queries[h][row * qstride + d] = d < Dim ? half(row0 + row < rows ? q[(std::size_t(row0 + row) * head_count + source) * Dim + d] : 0.0F, failure)
                                                : _Float16(0);
    }
    const auto vstride = split_stride(packed_stride);
    const auto* vpos = packed + std::size_t(vstride) * width;
    const auto* vneg = vpos + std::size_t(vstride) * width;
    const bool live_r = row0 + rr < rows;
    const unsigned position_r = position0 + rr;
    const unsigned lo_r = !r.global && position_r >= r.window ? position_r - r.window + 1 : 0;
    float maximum = bit_cast<float>(0xff800000U), denominator = 0.0F;
    bool bad_value = false, bad_half = false;
    Float8 accum[tiles]{};
    __syncthreads();
    for (auto base = first / 16 * 16; base <= last; base += 16 * waves) {
        const auto mine = base + 16 * wave;
        if (mine <= last) {
            const auto* k = packed + std::size_t(mine + rr) * width + kv;
            Float8 score{};
#pragma unroll 4
            for (unsigned d0 = 0; d0 < padded; d0 += 16) {
                const auto* qa = &query[rr * qstride + d0];
                const Half8 a0 = *reinterpret_cast<const Half8*>(qa), a1 = *reinterpret_cast<const Half8*>(qa + 8);
                Half16 a, b;
#pragma unroll
                for (unsigned d = 0; d < 8; ++d) { a[d] = a0[d]; a[d + 8] = a1[d]; }
                if constexpr (Dim % 16 == 0) {
                    b = *reinterpret_cast<const Half16*>(k + d0);
                } else {
#pragma unroll
                    for (unsigned d = 0; d < 16; ++d) b[d] = d0 + d < Dim ? k[d0 + d] : _Float16(0);
                }
                score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, score);
                if (d0 % 64 == 48) __builtin_amdgcn_sched_barrier(0);
            }
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) scores[wave][(2 * e + lane / 16) * 17 + rr] = score[e];
        }
        __syncthreads();
        for (unsigned s = 0; s < waves; ++s) {
            const auto tile = base + 16 * s;
            if (tile > last) break;
            float sc[16];
#pragma unroll
            for (unsigned j = 0; j < 16; ++j) sc[j] = scores[s][rr * 17 + j];
            float m = maximum;
            bool present = false;
#pragma unroll
            for (unsigned j = 0; j < 16; ++j) {
                const bool v = live_r && tile + j >= lo_r && tile + j <= position_r;
                bad_value |= v && nonfinite(sc[j]);
                m = v && sc[j] > m ? sc[j] : m;
                present |= v;
            }
            const float alpha = !present ? 1.0F : denominator == 0.0F ? 0.0F : declared_exp(maximum - m);
            float sum = denominator * alpha;
            // Both half-waves hold the same row: each evaluates the exponentials
            // of its eight keys and takes the other eight from its partner lane.
            // The denominator still adds all sixteen in key order.
            const unsigned half_wave = lane / 16;
            float own[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const unsigned j = 8 * half_wave + i;
                const bool v = live_r && tile + j >= lo_r && tile + j <= position_r;
                const float e = declared_exp(sc[j] - m);
                float p = v ? e : 0.0F;
                // Two roundings, as declared: never fold the exponential's last
                // multiply into the FP16 conversion (v_fma_mix).
                asm volatile("" : "+v"(p));
                own[i] = p;
                // Keep one exponential live at a time; interleaving them costs
                // more registers than the accumulators leave.
                __builtin_amdgcn_sched_barrier(0);
            }
            Half16 a;
#pragma unroll
            for (unsigned j = 0; j < 16; ++j) {
                const float other = __shfl_xor(own[j % 8], 16, 32);
                const float p = j / 8 == half_wave ? own[j % 8] : other;
                const bool v = live_r && tile + j >= lo_r && tile + j <= position_r;
                sum = v ? sum + p : sum;
                a[j] = static_cast<_Float16>(p);
                bad_half |= overflowed(a[j]);
            }
            maximum = m; denominator = sum; bad_value |= nonfinite(sum);
            float scale[8];
            bool update[8];
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const unsigned position = position0 + 2 * e + lane / 16;
                const unsigned lo = !r.global && position >= r.window ? position - r.window + 1 : 0;
                scale[e] = __shfl(alpha, int(2 * e + lane / 16), 32);
                update[e] = row0 + 2 * e + lane / 16 < rows && tile <= position && tile + 15 >= lo;
            }
#pragma unroll
            for (unsigned t = 0; t < tiles; ++t) {
                // Bound the live V operands to two tiles ahead of their use.
                if (Group && t % Group == 0) __builtin_amdgcn_sched_barrier(0);
                const unsigned d = (wave * tiles + t) * 16 + rr;
                Half16 positive{}, negative{};
                if (Dim % 16 == 0 || d < Dim) {
                    const auto offset = std::size_t(kv + d) * vstride + tile;
                    positive = *reinterpret_cast<const Half16*>(vpos + offset);
                    negative = *reinterpret_cast<const Half16*>(vneg + offset);
                }
                Float8 zero{};
                const auto pos = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, positive, zero);
                const auto neg = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, negative, zero);
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e) {
                        const float part = pos[e] - neg[e];
                        const float next = accum[t][e] * scale[e] + part;
                        accum[t][e] = update[e] ? next : accum[t][e];
                    }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned e = 0; e < 8; ++e) {
        const unsigned row = row0 + 2 * e + lane / 16;
        const float total = __shfl(denominator, int(2 * e + lane / 16), 32);
#pragma unroll
        for (unsigned t = 0; t < tiles; ++t) {
            const unsigned d = (wave * tiles + t) * 16 + rr;
            if (row < rows && d < Dim) {
                const float value = accum[t][e] / total;
                bad_value |= nonfinite(value); out[(std::size_t(row) * head_count + head) * Dim + d] = value;
            }
        }
    }
    if (bad_value) atomicOr(failure, 1U);
    if (bad_half) atomicOr(failure, 4U);
}
} // namespace
// Diagnostic: V operand prefetch group of the split kernel (identical bits).
unsigned attention_group() {
    static const unsigned group = [] { const auto* p = std::getenv("STRIX_G4_PREFILL_ATTN_GROUP"); return p ? unsigned(std::atoi(p)) : 2U; }();
    return group;
}
// Waves per head doubled by default (D256: 2, D512: 4; identical bits):
// fewer accumulators per wave, twice the occupancy. STRIX_G4_PREFILL_ATTN_WIDE=0
// restores one/two waves per head.
bool attention_wide() {
    static const bool wide = [] { const auto* p = std::getenv("STRIX_G4_PREFILL_ATTN_WIDE"); return !p || std::string_view(p) != "0"; }();
    return wide;
}
bool attention_split_enabled() {
    static const bool enabled = [] { const auto* p = std::getenv("STRIX_G4_PREFILL_ATTN_SPLIT"); return !p || std::string_view(p) != "0"; }();
    return enabled;
}
std::size_t attention_packed_bytes(unsigned packed_stride, unsigned width) {
    const auto split = 3ULL * split_stride(packed_stride) * width;
    return std::max(split, 2ULL * packed_stride * width) * sizeof(std::uint16_t);
}
void attention(const float* q, const gpu::KvRead* reads, unsigned rows, unsigned heads, unsigned dim,
               float* out, unsigned* failure, gpu::Queue queue, void* packed, unsigned packed_stride, unsigned key_capacity, bool repack) {
    if (!rows || rows > max_chunk_rows || !heads || heads % group || !q || !reads || !out || !failure)
        throw std::invalid_argument("WMMA attention geometry");
    const auto stream = static_cast<hipStream_t>(queue);
    const bool split = packed && attention_split_enabled();
    if (packed) {
        if (!packed_stride || packed_stride > max_positions) throw std::invalid_argument("WMMA attention packed stride");
        // The coordinator's absolute score stride bounds every descriptor's
        // position. Extra workgroups exit before touching masked ring entries.
        const auto width = heads / group * dim;
        if (!key_capacity) key_capacity = packed_stride;
        if (key_capacity > packed_stride) throw std::invalid_argument("WMMA attention key capacity");
        // pack=false: packed already holds these descriptors' operands (a
        // layer sharing its producer's KV within the same chunk).
        if (!repack) {}
        else if (split)
            hipLaunchKernelGGL(pack_split, dim3((key_capacity + 61U) / 32U, (width + 31U) / 32U), dim3(256), 0, stream,
                               reads, rows, width, packed_stride, static_cast<_Float16*>(packed), failure);
        else
            hipLaunchKernelGGL(pack, dim3(unsigned((std::size_t(key_capacity) * width + 255) / 256)), dim3(256), 0, stream,
                               reads, rows, width, packed_stride, static_cast<_Float16*>(packed), failure);
    }
    switch (dim) {
#define G4_FLASH(D) case D:         if (split && (D == 256 || D == 512) && attention_wide()) hipLaunchKernelGGL((flash_split<D, 2, (D > 256 ? 4U : D == 256 ? 2U : 1U)>), dim3(heads / (D > 256 ? 2U : 4U), (rows + 15U) / 16U), dim3(D >= 256 ? 256 : 128), 0, stream, q, reads, rows, heads, out, failure, static_cast<const _Float16*>(packed), packed_stride);         else if (split && (D == 256 || D == 512) && attention_group() == 4) hipLaunchKernelGGL((flash_split<D, 4>), dim3(heads / (D > 256 ? 2U : 4U), (rows + 15U) / 16U), dim3(128), 0, stream, q, reads, rows, heads, out, failure, static_cast<const _Float16*>(packed), packed_stride);         else if (split && (D == 256 || D == 512) && attention_group() == 1) hipLaunchKernelGGL((flash_split<D, 1>), dim3(heads / (D > 256 ? 2U : 4U), (rows + 15U) / 16U), dim3(128), 0, stream, q, reads, rows, heads, out, failure, static_cast<const _Float16*>(packed), packed_stride);         else if (split) hipLaunchKernelGGL((flash_split<D>), dim3(heads / (D > 256 ? 2U : 4U), (rows + 15U) / 16U), dim3(128), 0, stream, q, reads, rows, heads, out, failure, static_cast<const _Float16*>(packed), packed_stride);         else hipLaunchKernelGGL((flash<D>), dim3((rows + 15U) / 16U, heads), dim3(128), 0, stream, q, reads, rows, heads, out, failure, static_cast<const _Float16*>(packed), packed_stride);         break
        G4_FLASH(8); G4_FLASH(16); G4_FLASH(32); G4_FLASH(64); G4_FLASH(256); G4_FLASH(512);
#undef G4_FLASH
        default: throw std::invalid_argument("WMMA attention head dimension");
    }
    if (const auto e = hipGetLastError(); e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
} // namespace strix::gemma4::prefill
