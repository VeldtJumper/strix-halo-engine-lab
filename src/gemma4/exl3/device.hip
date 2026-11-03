// Adapted mathematical codec and transforms; see notices/ for retained attribution.
#include "device.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace strix::gemma4::exl3 {
namespace {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
void checked() {
    if (const auto e = hipGetLastError(); e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
__device__ void flag(float value, unsigned* failure) {
    if ((bit_cast<unsigned>(value) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
}
// Keep an RN32 boundary before RN16: mixed FP16-output instructions can
// otherwise remove an observable rounding step on double-rounding ties.
__device__ float materialize(float v) { asm volatile("" : "+v"(v)); return v; }
__device__ _Float16 half(float v, unsigned* failure) {
    const auto h = _Float16(materialize(v));
    if ((bit_cast<unsigned short>(h) & 0x7c00U) == 0x7c00U) atomicOr(failure, 4U);
    return h;
}
__device__ void transform(float (&values)[4], unsigned* failure) {
    const auto lane = threadIdx.x % 32U;
#pragma unroll
    for (unsigned stride = 1; stride < 32; stride *= 2)
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const auto peer = __shfl_xor(values[j], int(stride), 32);
            values[j] = lane & stride ? peer - values[j] : values[j] + peer;
            flag(values[j], failure);
        }
#pragma unroll
    for (unsigned stride = 1; stride < 4; stride *= 2)
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) if (!(j & stride)) {
            const auto a = values[j], b = values[j + stride];
            values[j] = a + b; values[j + stride] = a - b;
            flag(values[j], failure); flag(values[j + stride], failure);
        }
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) { values[j] = values[j] * hscale; flag(values[j], failure); }
}
// panels = false: rows x inputs; true: the prefill GEMMs' [K16][row][16] input panels.
__device__ __forceinline__ void input_body(LinearView w, const float* x, unsigned rows, _Float16* out,
                                           unsigned* failure, unsigned* maximum, std::size_t group, bool panels = false) {
    if (group >= std::size_t(rows) * (w.inputs / 128)) return;
    const auto row = unsigned(group / (w.inputs / 128)), begin = unsigned(group % (w.inputs / 128)) * 128;
    const auto lane = threadIdx.x % 32;
    float v[4];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const auto i = begin + lane + 32 * j;
        const float a = x[std::size_t(row) * w.inputs + i];
        if (maximum) atomicMax(maximum, bit_cast<unsigned>(a) & 0x7fffffffU);
        const float scaled = materialize(a * fp16_value(load16(w.data + 2ULL * i)));
        v[j] = float(half(scaled, failure));
    }
    transform(v, failure);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const auto i = begin + lane + 32 * j;
        out[panels ? std::size_t(i / 16) * rows * 16 + row * 16 + i % 16 : std::size_t(row) * w.inputs + i] = half(v[j], failure);
    }
}
template <bool Panels = false>
__global__ void input_kernel(LinearView w, const float* x, unsigned rows, _Float16* out,
                             unsigned* failure, unsigned* maximum) {
    input_body(w, x, rows, out, failure, maximum, std::size_t(blockIdx.x) * 4 + threadIdx.x / 32, Panels);
}
// Output transform and scale of one row's 128 core values v[j] (output begin + lane + 32j).
__device__ __forceinline__ void output_values(LinearView w, unsigned begin, float (&v)[4], unsigned* failure) {
    const auto lane = threadIdx.x % 32;
    transform(v, failure);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        v[j] = v[j] * fp16_value(load16(w.data + 2ULL * (w.inputs + begin + lane + 32 * j)));
        flag(v[j], failure);
    }
}
__device__ __forceinline__ void output_body(LinearView w, float* out, unsigned rows, unsigned* failure, std::size_t group) {
    if (group >= std::size_t(rows) * (w.outputs / 128)) return;
    const auto row = unsigned(group / (w.outputs / 128)), begin = unsigned(group % (w.outputs / 128)) * 128;
    const auto lane = threadIdx.x % 32;
    float v[4];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) v[j] = out[std::size_t(row) * w.outputs + begin + lane + 32 * j];
    output_values(w, begin, v, failure);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) out[std::size_t(row) * w.outputs + begin + lane + 32 * j] = v[j];
}
__global__ void output_kernel(LinearView w, float* out, unsigned rows, unsigned* failure) {
    output_body(w, out, rows, failure, std::size_t(blockIdx.x) * 4 + threadIdx.x / 32);
}
// Linears sharing one input: blockIdx.y selects the member. Each member's
// transform is exactly the single-linear transform above.
struct Group {
    LinearView w[max_group];
    float* out[max_group];
};
__global__ void input_group_kernel(Group g, const float* x, unsigned rows, _Float16* scratch, unsigned* failure) {
    const auto t = blockIdx.y;
    input_body(g.w[t], x, rows, scratch + std::size_t(t) * rows * g.w[0].inputs, failure, nullptr,
               std::size_t(blockIdx.x) * 4 + threadIdx.x / 32);
}
__global__ void output_group_kernel(Group g, unsigned rows, unsigned* failure) {
    const auto t = blockIdx.y;
    output_body(g.w[t], g.out[t], rows, failure, std::size_t(blockIdx.x) * 4 + threadIdx.x / 32);
}
__device__ __forceinline__ float native_codebook(unsigned s) {
    const unsigned product = s * 0x83dcd12dU;
    const unsigned sum = __builtin_amdgcn_udot4(product, 0x01010101U, 0, false);
    const float value = float(1024 + sum) * fp16_value(0x1eeeU) + fp16_value(0xc931U);
    return float(_Float16(materialize(value)));
}
template <unsigned K> __device__ float decoded(LinearView w, unsigned input, unsigned output) {
    const auto* tile = w.data + 2ULL * (w.inputs + w.outputs) +
        (std::size_t(input / 16) * (w.outputs / 16) + output / 16) * (32 * K);
    return native_codebook(state(tile, K, stream_position(input % 16, output % 16)));
}
// One logical column touches K or K+1 ring words across its sixteen input
// coordinates. Cache them before decode; the row dot still visits e=0..15.
// High is uniform for each output tile, keeping register indices constant
// after unrolling instead of introducing dynamically indexed private arrays.
template <unsigned K, bool High> struct Column {
    unsigned words[K + 1];
    __device__ __forceinline__ void load(LinearView w, unsigned chunk, unsigned output) {
        const auto* tile = w.data + 2ULL * (w.inputs + w.outputs) +
            (std::size_t(chunk) * (w.outputs / 16) + output / 16) * (32 * K);
#pragma unroll
        for (unsigned j = 0; j <= K; ++j) {
            const int index = int((output & 7U) * K + j) - (High ? 0 : 1);
            const auto wrapped = unsigned(index < 0 ? index + int(8 * K) : index >= int(8 * K) ? index - int(8 * K) : index);
            words[j] = word(tile + 4ULL * wrapped);
        }
    }
    __device__ __forceinline__ float at(unsigned e) const {
        const auto position = ((e & 6U) << 2U) | (e & 1U) | ((e & 8U) >> 2U);
        const auto start = (High ? 5 * K - 16 : K + 16) + position * K;
        const auto pair = (std::uint64_t(words[start / 32]) << 32U) | words[(start + 15) / 32];
        const auto s = unsigned(pair >> ((32U - ((start + 16) & 31U)) & 31U)) & 65535U;
        return native_codebook(s);
    }
};
template <unsigned K, unsigned Rows, unsigned Outputs, bool Cached, bool High>
__device__ __forceinline__ void core_body(LinearView w, const _Float16* x, float* out, unsigned* failure, unsigned first, unsigned lane) {
    float acc[Outputs][Rows] = {};
    for (unsigned c = lane; c < w.inputs / 16; c += 32) {
        Column<K, High> columns[Outputs];
        if constexpr (Cached)
#pragma unroll
            for (unsigned o = 0; o < Outputs; ++o) columns[o].load(w, c, first + o);
#pragma unroll
        for (unsigned e = 0; e < 16; ++e) {
            const auto i = c * 16 + e;
#pragma unroll
            for (unsigned o = 0; o < Outputs; ++o) {
                const float weight = Cached ? columns[o].at(e) : decoded<K>(w, i, first + o);
#pragma unroll
                for (unsigned r = 0; r < Rows; ++r)
                    acc[o][r] = acc[o][r] + weight * float(x[std::size_t(r) * w.inputs + i]);
            }
        }
    }
#pragma unroll
    for (unsigned o = 0; o < Outputs; ++o)
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            auto value = acc[o][r];
            for (unsigned d = 16; d; d >>= 1U) value = value + __shfl_down(value, d, 32);
            if (!lane) { out[std::size_t(r) * w.outputs + first + o] = value; flag(value, failure); }
        }
}
template <unsigned K, unsigned Rows, unsigned Outputs, bool Cached>
__global__ void core_kernel(LinearView w, const _Float16* x, float* out, unsigned* failure) {
    const auto first = (blockIdx.x * 4 + threadIdx.x / 32) * Outputs, lane = threadIdx.x % 32;
    if (first >= w.outputs) return;
    if constexpr (Cached) {
        if (first & 8U) core_body<K, Rows, Outputs, true, true>(w, x, out, failure, first, lane);
        else core_body<K, Rows, Outputs, true, false>(w, x, out, failure, first, lane);
    } else core_body<K, Rows, Outputs, false, false>(w, x, out, failure, first, lane);
}
template <unsigned Rows, unsigned Outputs, bool Cached>
void launch_core_tile(LinearView w, const _Float16* x, float* out, unsigned* failure, hipStream_t q) {
    switch (bits(w)) {
#define G4_EXL3_CORE(K) case K: hipLaunchKernelGGL((core_kernel<K, Rows, Outputs, Cached>), dim3(w.outputs / (4 * Outputs)), dim3(128), 0, q, w, x, out, failure); break
        G4_EXL3_CORE(4); G4_EXL3_CORE(5); G4_EXL3_CORE(6); G4_EXL3_CORE(8);
#undef G4_EXL3_CORE
    }
}
template <unsigned Rows> void launch_core(LinearView w, const _Float16* x, float* out, unsigned* failure, hipStream_t q);
// Lane-tile route. The arithmetic is exactly core_body's: lane l owns the
// partial over chunks c = l (mod 32), visits e = 0..15 in order and adds one
// exact FP16 product at a time, followed by the unchanged lane butterfly.
// Only the data movement differs: each lane reads an aligned window of its own
// tile with 16-byte loads (G column groups, 2G outputs), prefetches the next
// chunk's window, and decodes each coefficient with one FP16 FMA. That FMA
// equals RN16(RN32(.)) for all 1021 codebook sums (exact product, no
// double-rounding tie); the operator bench enumerates all 65,536 states.
using Word4 = std::uint32_t __attribute__((ext_vector_type(4)));
using Half8 = _Float16 __attribute__((ext_vector_type(8)));
template <unsigned N, class F> __device__ __forceinline__ void unrolled(F&& f) {
    [&]<unsigned... I>(std::integer_sequence<unsigned, I...>) { (f.template operator()<I>(), ...); }(
        std::make_integer_sequence<unsigned, N>{});
}
constexpr unsigned position_of(unsigned row, unsigned col) {
    return ((col & 7U) << 5U) | ((row & 6U) << 2U) | (row & 1U) | ((row & 8U) >> 2U) | ((col & 8U) >> 1U);
}
template <unsigned K, unsigned G, unsigned G0> struct LaneTile {
    static_assert(G == 8 ? G0 == 0 : G == 4 && (G0 == 0 || G0 == 4));
    static constexpr unsigned ring = 8 * K;
    // A half tile also needs the word preceding its first group (ring wrap for G0=0).
    static constexpr unsigned base = G == 8 ? 0 : (G0 * K + ring - 4) % ring;
    static constexpr unsigned vectors = G == 8 ? 2 * K : G * K / 4 + 1;
    Word4 v[vectors];
    __device__ __forceinline__ void load(const std::uint32_t* tile) {
#pragma unroll
        for (unsigned i = 0; i < vectors; ++i) v[i] = *reinterpret_cast<const Word4*>(tile + (base + 4 * i) % ring);
    }
    template <unsigned J> __device__ __forceinline__ std::uint32_t word() const {
        static_assert(J < 4 * vectors);
        return v[J / 4][J % 4];
    }
    // Same bits as format.hpp state(): 16 ring bits ending at (P+1)K, MSB first.
    template <unsigned P> __device__ __forceinline__ unsigned state() const {
        constexpr unsigned start = ((P + 1) * K + 256 * K - 16) % (256 * K), off = start % 32;
        constexpr unsigned first = (start / 32 + ring - base) % ring;
        if constexpr (off <= 16) return (word<first>() >> (16 - off)) & 65535U;
        else {
            constexpr unsigned second = (start / 32 + 1 + ring - base) % ring;
            return ((word<first>() << (off - 16)) | (word<second>() >> (48 - off))) & 65535U;
        }
    }
};
// One FP16 FMA per coefficient. (A packed half2 form was observed to be
// miscompiled for gfx1151 when its halves are extracted; scalar FMA avoids it.)
__device__ __forceinline__ _Float16 codebook_half(unsigned s) {
    // 0x6400 + sum is the FP16 encoding of 1024 + sum (sum <= 1020).
    const unsigned sum = __builtin_amdgcn_udot4(s * 0x83dcd12dU, 0x01010101U, 0x6400U, false);
    const auto scale = __builtin_bit_cast(_Float16, std::uint16_t(0x1eee));
    const auto bias = __builtin_bit_cast(_Float16, std::uint16_t(0xc931));
    return __builtin_elementwise_fma(__builtin_bit_cast(_Float16, std::uint16_t(sum)), scale, bias);
}
__global__ void codebook_table_kernel(std::uint16_t* out) {
    const unsigned s = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    out[s] = __builtin_bit_cast(std::uint16_t, codebook_half(s));
    out[s + 1] = __builtin_bit_cast(std::uint16_t, codebook_half(s + 1));
}
// g4-exl3-v2 core (format.hpp): one wave per 16-output tile column and split.
// Lane t holds output columns t/4 and t/4+8 and inputs (2a, 2a+1, 2a+8, 2a+9)
// with a = t%4, i.e. the eight consecutive ring states 8t..8t+7 of each tile,
// read from three aligned words per lane (coalesced across the wave).
template <unsigned K> struct CoopLane {
    std::uint32_t high, low, tail;
    __device__ __forceinline__ void load(const std::uint32_t* data, unsigned lane) {
        constexpr unsigned count = 8 * K;
        const unsigned start = (lane * 8 + 1) * K + 256 * K - 16, index = start / 32, shift = start & 31U; // shift != 0
        const auto a = data[index % count], b = data[(index + 1) % count], c = data[(index + 2) % count];
        high = (a << shift) | (b >> (32 - shift));
        low = (b << shift) | (c >> (32 - shift));
        tail = c << shift;
    }
    // The same window from words fetched earlier (CoopWords): alignment is
    // deferred to the chunk's use so pending prefetches are not waited on.
    __device__ __forceinline__ void align(std::uint32_t a, std::uint32_t b, std::uint32_t c, unsigned lane) {
        const unsigned shift = ((lane * 8 + 1) * K + 256 * K - 16) & 31U;
        high = (a << shift) | (b >> (32 - shift));
        low = (b << shift) | (c >> (32 - shift));
        tail = c << shift;
    }
    // State 8t+J: the 16 ring bits starting J*K bits into the aligned window.
    template <unsigned J> __device__ __forceinline__ unsigned state() const {
        constexpr unsigned offset = J * K;
        if constexpr (offset <= 16) return (high >> (16 - offset)) & 65535U;
        else if constexpr (offset < 32) return ((high << (offset - 16)) | (low >> (48 - offset))) & 65535U;
        else if constexpr (offset <= 48) return (low >> (48 - offset)) & 65535U;
        else return ((low << (offset - 48)) | (tail >> (80 - offset))) & 65535U;
    }
};
// Raw trellis words of one lane and chunk (CoopLane::load's three reads).
template <unsigned K> struct CoopWords {
    std::uint32_t a, b, c;
    __device__ __forceinline__ void fetch(const std::uint32_t* data, unsigned lane) {
        constexpr unsigned count = 8 * K;
        const unsigned index = ((lane * 8 + 1) * K + 256 * K - 16) / 32;
        a = data[index % count]; b = data[(index + 1) % count]; c = data[(index + 2) % count];
    }
};
// Accumulates one split of one tile column; returns the quad-reduced value in
// lane a == 0 (lanes 4c..4c+3 hold output column c and c+8 in acc[0]/acc[1]).
// One chunk of the cooperative core: lane chains in the identity's order.
template <unsigned K, unsigned Rows>
__device__ __forceinline__ void coop_chunk(const CoopLane<K>& t, const _Float16 (&xs)[Rows][4], float (&acc)[2][Rows]) {
    const float c0 = float(codebook_half(t.template state<0>())), c1 = float(codebook_half(t.template state<1>()));
    const float c2 = float(codebook_half(t.template state<2>())), c3 = float(codebook_half(t.template state<3>()));
    const float c4 = float(codebook_half(t.template state<4>())), c5 = float(codebook_half(t.template state<5>()));
    const float c6 = float(codebook_half(t.template state<6>())), c7 = float(codebook_half(t.template state<7>()));
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
        const float x0 = float(xs[r][0]), x1 = float(xs[r][1]), x8 = float(xs[r][2]), x9 = float(xs[r][3]);
        acc[0][r] = __builtin_fmaf(c0, x0, acc[0][r]); acc[0][r] = __builtin_fmaf(c1, x1, acc[0][r]);
        acc[0][r] = __builtin_fmaf(c2, x8, acc[0][r]); acc[0][r] = __builtin_fmaf(c3, x9, acc[0][r]);
        acc[1][r] = __builtin_fmaf(c4, x0, acc[1][r]); acc[1][r] = __builtin_fmaf(c5, x1, acc[1][r]);
        acc[1][r] = __builtin_fmaf(c6, x8, acc[1][r]); acc[1][r] = __builtin_fmaf(c7, x9, acc[1][r]);
    }
}
// Depth > 1 keeps Depth chunks of trellis words in flight per wave (decode is
// latency-bound at one chunk); chunks are consumed in the same order with the
// same arithmetic, so the bytes equal Depth 1. Spans not divisible by Depth
// use Depth 1.
template <unsigned K, unsigned Rows, unsigned Depth = 1>
__device__ __forceinline__ void coop_body(LinearView w, const _Float16* x, unsigned tile, unsigned split, unsigned splits,
                                          unsigned lane, float (&acc)[2][Rows]) {
    const unsigned a = lane & 3U, chunks = w.inputs / 16, span = chunks / splits;
    const unsigned begin = split * span, end = begin + span, tiles = w.outputs / 16;
    const auto* core = reinterpret_cast<const std::uint32_t*>(w.data + 2ULL * (w.inputs + w.outputs));
    if constexpr (Depth > 1) {
        if (span % Depth == 0) {
            CoopWords<K> pipe[Depth];
#pragma unroll
            for (unsigned j = 0; j < Depth; ++j) pipe[j].fetch(core + (std::size_t(begin + j) * tiles + tile) * (8 * K), lane);
            for (unsigned k = begin; k < end; k += Depth) {
#pragma unroll
                for (unsigned j = 0; j < Depth; ++j) {
                    _Float16 xs[Rows][4];
#pragma unroll
                    for (unsigned r = 0; r < Rows; ++r) {
                        const auto* row = x + std::size_t(r) * w.inputs + (k + j) * 16 + 2 * a;
                        xs[r][0] = row[0]; xs[r][1] = row[1]; xs[r][2] = row[8]; xs[r][3] = row[9];
                    }
                    CoopLane<K> t;
                    t.align(pipe[j].a, pipe[j].b, pipe[j].c, lane);
                    if (k + j + Depth < end) pipe[j].fetch(core + (std::size_t(k + j + Depth) * tiles + tile) * (8 * K), lane);
                    coop_chunk<K, Rows>(t, xs, acc);
                }
            }
#pragma unroll
            for (unsigned h = 0; h < 2; ++h)
#pragma unroll
                for (unsigned r = 0; r < Rows; ++r) {
                    acc[h][r] = acc[h][r] + __shfl_down(acc[h][r], 2, 4);
                    acc[h][r] = acc[h][r] + __shfl_down(acc[h][r], 1, 4);
                }
            return;
        }
    }
    // The next chunk's words are requested after this chunk's activations, so
    // waiting on the activations never drains the prefetch.
    CoopLane<K> t;
    t.load(core + (std::size_t(begin) * tiles + tile) * (8 * K), lane);
    for (unsigned k = begin; k < end; ++k) {
        _Float16 xs[Rows][4];
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const auto* row = x + std::size_t(r) * w.inputs + k * 16 + 2 * a;
            xs[r][0] = row[0]; xs[r][1] = row[1]; xs[r][2] = row[8]; xs[r][3] = row[9];
        }
        CoopLane<K> next;
        if (k + 1 < end) next.load(core + (std::size_t(k + 1) * tiles + tile) * (8 * K), lane);
        coop_chunk<K, Rows>(t, xs, acc);
        t = next;
    }
#pragma unroll
    for (unsigned h = 0; h < 2; ++h)
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            acc[h][r] = acc[h][r] + __shfl_down(acc[h][r], 2, 4);
            acc[h][r] = acc[h][r] + __shfl_down(acc[h][r], 1, 4);
        }
}
template <unsigned K, unsigned Rows, unsigned Depth = 1>
__global__ __launch_bounds__(128) void coop_kernel(LinearView w, const _Float16* x, float* out, float* partial,
                                                   unsigned splits, unsigned* failure) {
    const unsigned tile = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (tile >= w.outputs / 16) return;
    const unsigned a = lane & 3U, column = lane >> 2U;
    float acc[2][Rows] = {};
    coop_body<K, Rows, Depth>(w, x, tile, blockIdx.y, splits, lane, acc);
#pragma unroll
    for (unsigned h = 0; h < 2; ++h)
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const float v = acc[h][r];
            if (!a) {
                const auto o = tile * 16 + column + 8 * h;
                if (splits == 1) { out[std::size_t(r) * w.outputs + o] = v; flag(v, failure); }
                else partial[(std::size_t(blockIdx.y) * Rows + r) * w.outputs + o] = v;
            }
        }
}
// Sums split partials in increasing split order, then applies output_body.
__global__ void coop_output_kernel(LinearView w, const float* partial, unsigned splits, float* out, unsigned rows,
                                   unsigned* failure) {
    const auto group = std::size_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (group >= std::size_t(rows) * (w.outputs / 128)) return;
    const auto row = unsigned(group / (w.outputs / 128)), begin = unsigned(group % (w.outputs / 128)) * 128;
    const auto lane = threadIdx.x % 32;
    float v[4];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const auto o = begin + lane + 32 * j;
        v[j] = partial[std::size_t(row) * w.outputs + o];
        for (unsigned s = 1; s < splits; ++s) v[j] = v[j] + partial[(std::size_t(s) * rows + row) * w.outputs + o];
        flag(v[j], failure);
    }
    output_values(w, begin, v, failure);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) out[std::size_t(row) * w.outputs + begin + lane + 32 * j] = v[j];
}
// Fused v2 linear: one block owns 128 outputs (eight tile columns) and all
// S splits (8*S waves). With In it first forms the transformed FP16 rows in
// LDS (input_body); after the cores it sums split partials in increasing s
// and applies output_body's transform and scale. Same arithmetic as the
// three-launch form.
template <unsigned K, unsigned Rows, unsigned S, bool In>
__global__ __launch_bounds__(1024) void coop_fused_kernel(LinearView w, const _Float16* x, const float* source, float* out,
                                                          unsigned* failure) {
    __shared__ float part[S * Rows * 128];
    extern __shared__ Word4 dynamic_lds[];
    const unsigned wave = threadIdx.x / 32, lane = threadIdx.x % 32, first = blockIdx.x * 128;
    if constexpr (In) {
        auto* transformed = reinterpret_cast<_Float16*>(dynamic_lds);
        const unsigned groups = Rows * (w.inputs / 128), waves = blockDim.x / 32;
        for (unsigned group = wave; group < groups; group += waves)
            input_body(w, source, Rows, transformed, failure, nullptr, group);
        __syncthreads();
        x = transformed;
    }
    const unsigned column = wave % 8, split = wave / 8;
    float acc[2][Rows] = {};
    coop_body<K, Rows>(w, x, blockIdx.x * 8 + column, split, S, lane, acc);
    if (!(lane & 3U))
#pragma unroll
        for (unsigned h = 0; h < 2; ++h)
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) part[(split * Rows + r) * 128 + column * 16 + (lane >> 2U) + 8 * h] = acc[h][r];
    __syncthreads();
    if (wave >= Rows) return;
    float v[4];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const unsigned o = lane + 32 * j;
        v[j] = part[wave * 128 + o];
        for (unsigned s2 = 1; s2 < S; ++s2) v[j] = v[j] + part[(s2 * Rows + wave) * 128 + o];
        flag(v[j], failure);
    }
    transform(v, failure);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const auto o = first + lane + 32 * j;
        const float value = v[j] * fp16_value(load16(w.data + 2ULL * (w.inputs + o)));
        out[std::size_t(wave) * w.outputs + o] = value; flag(value, failure);
    }
}
// Returns false when In does not fit LDS (nothing launched).
template <unsigned Rows, unsigned S>
bool launch_coop_fused_s(LinearView w, const _Float16* x, const float* source, float* out, unsigned* failure, hipStream_t q) {
    const dim3 blocks(w.outputs / 128), threads(256 * S);
    const std::size_t dynamic = source ? std::size_t(Rows) * w.inputs * sizeof(_Float16) : 0;
    if (source && S * Rows * 512 + dynamic > 64 * 1024) return false;
    switch (bits(w)) {
#define G4_EXL3_CF(K) case K: if (source) hipLaunchKernelGGL((coop_fused_kernel<K, Rows, S, true>), blocks, threads, dynamic, q, w, nullptr, source, out, failure); \
                       else hipLaunchKernelGGL((coop_fused_kernel<K, Rows, S, false>), blocks, threads, 0, q, w, x, nullptr, out, failure); break
        G4_EXL3_CF(4); G4_EXL3_CF(5); G4_EXL3_CF(6); G4_EXL3_CF(8);
#undef G4_EXL3_CF
    }
    return true;
}
template <unsigned Rows>
bool launch_coop_fused(LinearView w, const _Float16* x, const float* source, float* out, unsigned* failure, hipStream_t q) {
    switch (v2_splits(w.inputs, w.outputs)) {
    case 1: return launch_coop_fused_s<Rows, 1>(w, x, source, out, failure, q);
    case 2: return launch_coop_fused_s<Rows, 2>(w, x, source, out, failure, q);
    case 4: return launch_coop_fused_s<Rows, 4>(w, x, source, out, failure, q);
    default: return false; // eight splits exceed one block; the three-launch form runs
    }
}
// Grouped v2/v3 epilogue: blockIdx.y selects the member; each member's
// partial sum and output transform are exactly coop_output_kernel /
// output_kernel's (members with one split transform their core output in place).
struct CoopGroup {
    LinearView w[max_group];
    float* out[max_group];
    const float* partial[max_group];
    unsigned splits[max_group];
};
__global__ void coop_output_group_kernel(CoopGroup g, unsigned rows, unsigned* failure) {
    const auto t = blockIdx.y;
    const auto w = g.w[t];
    const auto group = std::size_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (group >= std::size_t(rows) * (w.outputs / 128)) return;
    if (g.splits[t] > 1) {
        const auto row = unsigned(group / (w.outputs / 128)), begin = unsigned(group % (w.outputs / 128)) * 128;
        const auto lane = threadIdx.x % 32;
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const auto o = begin + lane + 32 * j;
            float v = g.partial[t][std::size_t(row) * w.outputs + o];
            for (unsigned s2 = 1; s2 < g.splits[t]; ++s2) v = v + g.partial[t][(std::size_t(s2) * rows + row) * w.outputs + o];
            flag(v, failure);
            g.out[t][std::size_t(row) * w.outputs + o] = v;
        }
    }
    output_body(w, g.out[t], rows, failure, group);
}
// Trellis prefetch depth of the decode cores: 4 by default (fastest measured C1
// decode), 2 selectable; STRIX_G4_EXL3_COOP_DEPTH=1 restores one aligned chunk.
unsigned coop_depth() {
    static const unsigned depth = [] {
        const auto* p = std::getenv("STRIX_G4_EXL3_COOP_DEPTH");
        if (!p || std::string_view(p) == "4") return 4U;
        if (std::string_view(p) == "1") return 1U;
        if (std::string_view(p) == "2") return 2U;
        throw std::invalid_argument("STRIX_G4_EXL3_COOP_DEPTH must be 1, 2 or 4");
    }();
    return depth;
}
template <unsigned Rows> void launch_coop(LinearView w, const _Float16* x, float* out, float* partial, unsigned splits,
                                          unsigned* failure, hipStream_t q) {
    const dim3 grid((w.outputs / 16 + 3) / 4, splits);
    const unsigned depth = coop_depth();
    switch (bits(w)) {
#define G4_EXL3_COOP(K) case K: \
        if (depth == 4) hipLaunchKernelGGL((coop_kernel<K, Rows, 4>), grid, dim3(128), 0, q, w, x, out, partial, splits, failure); \
        else if (depth == 2) hipLaunchKernelGGL((coop_kernel<K, Rows, 2>), grid, dim3(128), 0, q, w, x, out, partial, splits, failure); \
        else hipLaunchKernelGGL((coop_kernel<K, Rows>), grid, dim3(128), 0, q, w, x, out, partial, splits, failure); break
        G4_EXL3_COOP(4); G4_EXL3_COOP(5); G4_EXL3_COOP(6); G4_EXL3_COOP(8);
#undef G4_EXL3_COOP
    }
}
template <unsigned Rows> struct ChunkInput {
    Half8 v[Rows][2];
    // Issued before the next tile's loads so waiting on it does not drain them.
    __device__ __forceinline__ void load(const _Float16* x, unsigned inputs, unsigned c) {
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const auto* row = reinterpret_cast<const Half8*>(x + std::size_t(r) * inputs + c * 16);
            v[r][0] = row[0]; v[r][1] = row[1];
        }
    }
};
template <unsigned K, unsigned Rows, unsigned G, unsigned G0, class Tile>
__device__ __forceinline__ void lane_chunk(const Tile& current, const ChunkInput<Rows>& xv, float (&acc)[G][2][Rows]) {
    unrolled<16 * G>([&]<unsigned I>() {
        constexpr unsigned g = I / 16, e = I % 16;
        const float low = float(codebook_half(current.template state<position_of(e, G0 + g)>()));
        const float high = float(codebook_half(current.template state<position_of(e, G0 + g + 8)>()));
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const float input = float(xv.v[r][e / 8][e % 8]);
            acc[g][0][r] = __builtin_fmaf(low, input, acc[g][0][r]);
            acc[g][1][r] = __builtin_fmaf(high, input, acc[g][1][r]);
        }
    });
}
// Staged variant: the wave loads the windows of its 32 lanes cooperatively
// (adjacent lanes read adjacent 16-byte vectors of one tile), transposes them
// through a wave-private LDS area, and each lane then decodes its own window
// exactly as in the direct variant. Coefficients and per-lane order are unchanged.
template <unsigned K, unsigned G> constexpr unsigned staged_stride() {
    constexpr unsigned v = G == 8 ? 2 * K : G * K / 4 + 1;
    return v % 2 ? v : v + 1; // odd 16-byte stride avoids LDS bank conflicts
}
template <unsigned K, unsigned Rows, unsigned G, unsigned G0, bool Staged>
__device__ __forceinline__ void lane_body(LinearView w, const _Float16* x, float* out, std::size_t out_stride,
                                          unsigned out_first, unsigned* failure, unsigned n, unsigned lane, Word4* lds) {
    using Tile = LaneTile<K, G, G0>;
    constexpr unsigned V = Tile::vectors, S = staged_stride<K, G>();
    const unsigned chunks = w.inputs / 16, tiles = w.outputs / 16;
    const auto* core = reinterpret_cast<const std::uint32_t*>(w.data + 2ULL * (w.inputs + w.outputs));
    float acc[G][2][Rows] = {};
    Tile current;
    if constexpr (Staged) {
        Word4 raw[V];
        const auto gather = [&](unsigned first) {
#pragma unroll
            for (unsigned m = 0; m < V; ++m) {
                const unsigned j = lane + 32 * m, t = j / V, v = j % V, chunk = first + t;
                unsigned offset = Tile::base + 4 * v;
                if (offset >= Tile::ring) offset -= Tile::ring;
                raw[m] = chunk < chunks ? *reinterpret_cast<const Word4*>(core + (std::size_t(chunk) * tiles + n) * (8 * K) + offset)
                                        : Word4{};
            }
        };
        const auto stage = [&] {
#pragma unroll
            for (unsigned m = 0; m < V; ++m) {
                const unsigned j = lane + 32 * m;
                lds[(j / V) * S + j % V] = raw[m];
            }
#pragma unroll
            for (unsigned i = 0; i < V; ++i) current.v[i] = lds[lane * S + i];
        };
        gather(0);
        stage();
        for (unsigned first = 0; first < chunks; first += 32) {
            const bool more = first + 32 < chunks, active = first + lane < chunks;
            ChunkInput<Rows> xv;
            if (active) xv.load(x, w.inputs, first + lane);
            if (more) gather(first + 32);
            if (active) lane_chunk<K, Rows, G, G0>(current, xv, acc);
            if (more) stage();
        }
    } else {
        if (lane < chunks) current.load(core + (std::size_t(lane) * tiles + n) * (8 * K));
        for (unsigned c = lane; c < chunks; c += 32) {
            ChunkInput<Rows> xv;
            xv.load(x, w.inputs, c);
            Tile next;
            if (c + 32 < chunks) next.load(core + (std::size_t(c + 32) * tiles + n) * (8 * K));
            lane_chunk<K, Rows, G, G0>(current, xv, acc);
            current = next;
        }
    }
#pragma unroll
    for (unsigned g = 0; g < G; ++g)
#pragma unroll
        for (unsigned h = 0; h < 2; ++h)
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) {
                auto value = acc[g][h][r];
                for (unsigned d = 16; d; d >>= 1U) value = value + __shfl_down(value, d, 32);
                if (!lane) {
                    out[std::size_t(r) * out_stride + n * 16 + G0 + g + 8 * h - out_first] = value;
                    flag(value, failure);
                }
            }
}
template <unsigned K, unsigned Rows, unsigned G, bool Staged>
__global__ __launch_bounds__(128) void lane_kernel(LinearView w, const _Float16* x, float* out, unsigned* failure) {
    const unsigned wave = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    constexpr unsigned S = staged_stride<K, G>();
    __shared__ Word4 staging[Staged ? 4 * 32 * S : 1];
    Word4* lds = staging + (Staged ? (threadIdx.x / 32) * 32 * S : 0);
    if constexpr (G == 8) {
        if (wave < w.outputs / 16) lane_body<K, Rows, 8, 0, Staged>(w, x, out, w.outputs, 0, failure, wave, lane, lds);
    } else {
        if (wave / 2 >= w.outputs / 16) return;
        if (wave & 1U) lane_body<K, Rows, 4, 4, Staged>(w, x, out, w.outputs, 0, failure, wave / 2, lane, lds);
        else lane_body<K, Rows, 4, 0, Staged>(w, x, out, w.outputs, 0, failure, wave / 2, lane, lds);
    }
}
template <unsigned Rows, unsigned G, bool Staged>
void launch_lane(LinearView w, const _Float16* x, float* out, unsigned* failure, hipStream_t q) {
    // Whole tiles exceed the register file beyond four rows; use half tiles.
    if constexpr (G == 8 && Rows > 4) launch_lane<Rows, 4, Staged>(w, x, out, failure, q);
    else {
        const auto blocks = dim3(w.outputs / 16 * (8 / G) / 4);
        switch (bits(w)) {
#define G4_EXL3_LANE(K) case K: hipLaunchKernelGGL((lane_kernel<K, Rows, G, Staged>), blocks, dim3(128), 0, q, w, x, out, failure); break
            G4_EXL3_LANE(4); G4_EXL3_LANE(5); G4_EXL3_LANE(6); G4_EXL3_LANE(8);
#undef G4_EXL3_LANE
        }
    }
}
// Fused output epilogue: one block owns 128 outputs (eight tile columns), so
// after each wave's unchanged butterfly the block applies exactly
// output_body's H128 transform and signed scale from LDS instead of a
// separate launch. Core values, transform order and scaling are identical.
template <unsigned K, unsigned G> constexpr std::size_t fused_staging_bytes() {
    return std::size_t(8) * (8 / G) * 32 * staged_stride<K, G>() * sizeof(Word4);
}
// With In, the block first forms the transformed FP16 rows in LDS with
// exactly input_body's arithmetic (one wave per 128-input group), replacing
// the separate input launch and its scratch round trip.
template <unsigned K, unsigned Rows, unsigned G, bool Staged, bool In>
__device__ __forceinline__ void fused_body(LinearView w, const _Float16* x, const float* source, float* out,
                                           unsigned* failure, unsigned block, float* result, Word4* staging,
                                           _Float16* transformed) {
    constexpr unsigned per_tile = 8 / G, S = staged_stride<K, G>();
    const unsigned wave = threadIdx.x / 32, lane = threadIdx.x % 32, first = block * 128;
    if constexpr (In) {
        const unsigned groups = Rows * (w.inputs / 128), waves = blockDim.x / 32;
        for (unsigned group = wave; group < groups; group += waves)
            input_body(w, source, Rows, transformed, failure, nullptr, group);
        __syncthreads();
        x = transformed;
    }
    const unsigned n = block * 8 + wave / per_tile;
    Word4* lds = staging + (Staged ? wave * 32 * S : 0);
    if constexpr (G == 8) lane_body<K, Rows, 8, 0, Staged>(w, x, result, 128, first, failure, n, lane, lds);
    else if (wave % 2) lane_body<K, Rows, 4, 4, Staged>(w, x, result, 128, first, failure, n, lane, lds);
    else lane_body<K, Rows, 4, 0, Staged>(w, x, result, 128, first, failure, n, lane, lds);
    __syncthreads();
    if (wave >= Rows) return;
    float v[4];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) v[j] = result[wave * 128 + lane + 32 * j];
    transform(v, failure);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const auto o = first + lane + 32 * j;
        const float value = v[j] * fp16_value(load16(w.data + 2ULL * (w.inputs + o)));
        out[std::size_t(wave) * w.outputs + o] = value; flag(value, failure);
    }
}
template <unsigned K, unsigned Rows, unsigned G, bool Staged, bool In>
__global__ __launch_bounds__(512) void fused_kernel(LinearView w, const _Float16* x, const float* source, float* out,
                                                    unsigned* failure) {
    __shared__ float result[Rows * 128];
    __shared__ Word4 staging[Staged ? 8 * (8 / G) * 32 * staged_stride<K, G>() : 1];
    extern __shared__ Word4 dynamic_lds[];
    fused_body<K, Rows, G, Staged, In>(w, x, source, out, failure, blockIdx.x, result, staging,
                                       reinterpret_cast<_Float16*>(dynamic_lds));
}
// Several half-tile members (same inputs, any K) in one launch: block ranges
// select the member, whose body is exactly the single-member fused body. One
// ramp/tail instead of one per member; K8 members do not use staging.
struct FusedGroup {
    LinearView w[max_group];
    float* out[max_group];
    const _Float16* x[max_group];
    unsigned first[max_group + 1];
    unsigned count;
};
constexpr unsigned group_stride = staged_stride<6, 4>();
template <unsigned Rows, bool Staged, bool In>
__global__ __launch_bounds__(512) void group_fused_kernel(FusedGroup g, const float* source, unsigned* failure) {
    __shared__ float result[Rows * 128];
    __shared__ Word4 staging[Staged ? 16 * 32 * group_stride : 1];
    extern __shared__ Word4 dynamic_lds[];
    unsigned t = 0;
    while (t + 1 < g.count && blockIdx.x >= g.first[t + 1]) ++t;
    const unsigned block = blockIdx.x - g.first[t];
    auto* transformed = reinterpret_cast<_Float16*>(dynamic_lds);
    switch (bits(g.w[t])) {
    case 4: fused_body<4, Rows, 4, Staged, In>(g.w[t], g.x[t], source, g.out[t], failure, block, result, staging, transformed); break;
    case 5: fused_body<5, Rows, 4, Staged, In>(g.w[t], g.x[t], source, g.out[t], failure, block, result, staging, transformed); break;
    case 6: fused_body<6, Rows, 4, Staged, In>(g.w[t], g.x[t], source, g.out[t], failure, block, result, staging, transformed); break;
    default: fused_body<8, Rows, 4, false, In>(g.w[t], g.x[t], source, g.out[t], failure, block, result, staging, transformed); break;
    }
}
constexpr std::size_t lds_budget = 64 * 1024;
template <unsigned K, unsigned Rows, unsigned G, bool Staged> constexpr std::size_t fused_static_bytes() {
    return Rows * 128 * sizeof(float) + (Staged ? fused_staging_bytes<K, G>() : sizeof(Word4));
}
// Returns false (nothing launched) when source != nullptr and the transformed
// rows do not fit beside the block's static LDS; the caller then transforms first.
template <unsigned K, unsigned Rows, unsigned G, bool Staged>
bool launch_fused_k(LinearView w, const _Float16* x, const float* source, float* out, unsigned* failure, hipStream_t q) {
    // Staging is used for at most two rows and when it fits the block's LDS.
    if constexpr (Staged && (Rows > 2 || fused_staging_bytes<K, G>() > 60 * 1024))
        return launch_fused_k<K, Rows, G, false>(w, x, source, out, failure, q);
    else {
        const dim3 blocks(w.outputs / 128), threads(32 * 8 * (8 / G));
        if (source) {
            const std::size_t dynamic = std::size_t(Rows) * w.inputs * sizeof(_Float16);
            if (fused_static_bytes<K, Rows, G, Staged>() + dynamic > lds_budget) return false;
            hipLaunchKernelGGL((fused_kernel<K, Rows, G, Staged, true>), blocks, threads, dynamic, q, w, nullptr, source, out, failure);
        } else hipLaunchKernelGGL((fused_kernel<K, Rows, G, Staged, false>), blocks, threads, 0, q, w, x, nullptr, out, failure);
        return true;
    }
}
template <unsigned Rows, unsigned G, bool Staged>
bool launch_fused(LinearView w, const _Float16* x, const float* source, float* out, unsigned* failure, hipStream_t q) {
    if constexpr (G == 8 && Rows > 4) return launch_fused<Rows, 4, Staged>(w, x, source, out, failure, q);
    else switch (bits(w)) {
#define G4_EXL3_FUSED(K) case K: return launch_fused_k<K, Rows, G, Staged>(w, x, source, out, failure, q)
        G4_EXL3_FUSED(4); G4_EXL3_FUSED(5); G4_EXL3_FUSED(6); G4_EXL3_FUSED(8);
#undef G4_EXL3_FUSED
    }
    return false;
}
bool env_on(const char* name) {
    const auto* p = std::getenv(name);
    return !p || std::string_view(p) != "0";
}
// Measured-shape policy: whole tiles where outputs supply enough waves and
// at most two rows keep the register footprint spill-free; otherwise half tiles.
unsigned lane_groups(LinearView w, unsigned rows) {
    static const unsigned forced = [] {
        const auto* p = std::getenv("STRIX_G4_EXL3_LANE_GROUPS");
        return p && std::string_view(p) == "4" ? 4U : p && std::string_view(p) == "8" ? 8U : 0U;
    }();
    if (forced) return forced;
    return w.outputs >= 65536 && rows == 1 ? 8U : 4U;
}
// DRAM-cold operator measurements: staged windows help up to two rows;
// direct half-tile windows are as fast or faster beyond. STRIX_G4_EXL3_STAGED=0 disables.
// STRIX_G4_PREFILL_EXACT_MANYROW=0 keeps EXL3 prompt chunks on the eight-row passes.
bool rows_enabled() {
    static const bool enabled = env_on("STRIX_G4_PREFILL_EXACT_MANYROW");
    return enabled;
}
bool lane_staged(unsigned rows) {
    static const bool enabled = env_on("STRIX_G4_EXL3_STAGED");
    return enabled && rows <= 2;
}
template <unsigned Rows> void launch_core_route(LinearView w, const _Float16* x, float* out, unsigned* failure,
                                                CoreRoute route, hipStream_t q) {
    if (route == CoreRoute::automatic) {
        static const bool lane_tile = env_on("STRIX_G4_EXL3_LANE_TILE");
        const bool eight = lane_groups(w, Rows) == 8;
        route = !lane_tile ? CoreRoute::legacy
              : lane_staged(Rows) ? (eight ? CoreRoute::staged8 : CoreRoute::staged4)
                                  : (eight ? CoreRoute::lane8 : CoreRoute::lane4);
    }
    if (route == CoreRoute::lane8) { launch_lane<Rows, 8, false>(w, x, out, failure, q); return; }
    if (route == CoreRoute::lane4) { launch_lane<Rows, 4, false>(w, x, out, failure, q); return; }
    if (route == CoreRoute::staged8) { launch_lane<Rows, 8, true>(w, x, out, failure, q); return; }
    if (route == CoreRoute::staged4) { launch_lane<Rows, 4, true>(w, x, out, failure, q); return; }
    launch_core<Rows>(w, x, out, failure, q);
}
template <unsigned Rows> void launch_core(LinearView w, const _Float16* x, float* out, unsigned* failure, hipStream_t q) {
    static const bool tiled = [] {
        const auto* p = std::getenv("STRIX_G4_EXL3_OUTPUT_TILE");
        return !p || std::string_view(p) != "0";
    }();
    static const bool cached = [] {
        const auto* p = std::getenv("STRIX_G4_EXL3_COLUMN_CACHE");
        return !p || std::string_view(p) != "0";
    }();
    if (cached) {
        if (tiled) launch_core_tile<Rows, 2, true>(w, x, out, failure, q);
        else launch_core_tile<Rows, 1, true>(w, x, out, failure, q);
    } else {
        if (tiled) launch_core_tile<Rows, 2, false>(w, x, out, failure, q);
        else launch_core_tile<Rows, 1, false>(w, x, out, failure, q);
    }
}
// Four waves share the same K16 progression and own a 64x64 output tile.
// Panels are optional: both paths form exactly the same FP16 operands.
template <unsigned K> __global__ void wmma_kernel(LinearView w, const _Float16* x, unsigned rows,
                                                  float* out, unsigned* failure, const _Float16* panels) {
    const auto lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const auto m0 = blockIdx.y * 64 + (wave / 2) * 32, n0 = blockIdx.x * 64 + (wave % 2) * 32;
    Float8 c[2][2]{};
    for (unsigned k0 = 0; k0 < w.inputs; k0 += 16) {
        Half16 a[2], b[2];
#pragma unroll
        for (unsigned t = 0; t < 2; ++t) {
            const auto r = m0 + t * 16 + lane % 16, n = n0 + t * 16 + lane % 16;
#pragma unroll
            for (unsigned e = 0; e < 16; ++e) {
                a[t][e] = r < rows ? x[std::size_t(r) * w.inputs + k0 + e] : _Float16(0);
                b[t][e] = panels ? panels[(std::size_t(n / 16) * (w.inputs / 16) + k0 / 16) * 256 + (n % 16) * 16 + e]
                                  : _Float16(decoded<K>(w, k0 + e, n));
            }
        }
#pragma unroll
        for (unsigned i = 0; i < 2; ++i)
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) c[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], c[i][j]);
    }
#pragma unroll
    for (unsigned i = 0; i < 2; ++i)
#pragma unroll
        for (unsigned j = 0; j < 2; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m0 + i * 16 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
                if (r < rows) { out[std::size_t(r) * w.outputs + n] = c[i][j][e]; flag(c[i][j][e], failure); }
            }
}
}
namespace {
// Lane-tile core with the output transform fused; false when the default
// route is not lane-tile or STRIX_G4_EXL3_FUSED_OUTPUT=0.
void coop_core(LinearView w, const _Float16* x, unsigned rows, float* out, float* partial, unsigned splits,
               unsigned* failure, hipStream_t q) {
    switch (rows) {
#define G4_EXL3_CROWS(R) case R: launch_coop<R>(w, x, out, partial, splits, failure, q); break
        G4_EXL3_CROWS(1); G4_EXL3_CROWS(2); G4_EXL3_CROWS(3); G4_EXL3_CROWS(4);
        G4_EXL3_CROWS(5); G4_EXL3_CROWS(6); G4_EXL3_CROWS(7); G4_EXL3_CROWS(8);
#undef G4_EXL3_CROWS
    }
    checked();
}
// g4-exl3-v2 linear: v1 input transform, cooperative core, ordered split sum
// fused into the v1 output transform. Partials live at scratch + v2_partial_offset.
bool coop_fused(LinearView w, const _Float16* x, const float* source, unsigned rows, float* out, unsigned* failure,
                hipStream_t q) {
    bool launched = false;
    switch (rows) {
#define G4_EXL3_CFR(R) case R: launched = launch_coop_fused<R>(w, x, source, out, failure, q); break
        G4_EXL3_CFR(1); G4_EXL3_CFR(2); G4_EXL3_CFR(3); G4_EXL3_CFR(4);
        G4_EXL3_CFR(5); G4_EXL3_CFR(6); G4_EXL3_CFR(7); G4_EXL3_CFR(8);
#undef G4_EXL3_CFR
    }
    if (launched) checked();
    return launched;
}
// Grouped v2/v3 members over one input: one input-transform launch, the member
// cores, one split-sum/output launch. STRIX_G4_EXL3_V2_GROUP=0 issues them as
// separate linears. Returns false when partials do not fit (nothing launched).
bool device_group_v2(const LinearView* w, unsigned count, const float* x, unsigned rows, float* const* out, void* scratch,
                     unsigned* failure, hipStream_t q) {
    static const bool enabled = env_on("STRIX_G4_EXL3_V2_GROUP");
    if (!enabled || count < 2 || count > max_group) return false;
    CoopGroup g{};
    Group transforms{};
    std::size_t offset = 0;
    for (unsigned t = 0; t < count; ++t) {
        if (!geometry(w[t]) || w[t].inputs != w[0].inputs || !out[t]) throw std::invalid_argument("EXL3 group geometry");
        g.w[t] = transforms.w[t] = w[t]; g.out[t] = out[t];
        g.splits[t] = v2_splits(w[t].inputs, w[t].outputs);
        g.partial[t] = reinterpret_cast<const float*>(static_cast<std::uint8_t*>(scratch) + v2_partial_offset + offset);
        if (g.splits[t] > 1) offset += std::size_t(g.splits[t]) * rows * w[t].outputs * sizeof(float);
    }
    if (offset > v2_partial_bytes || std::size_t(count) * rows * w[0].inputs * sizeof(_Float16) > v2_partial_offset) return false;
    auto* transformed = static_cast<_Float16*>(scratch);
    hipLaunchKernelGGL(input_group_kernel, dim3((rows * (w[0].inputs / 128) + 3) / 4, count), dim3(128), 0, q, transforms, x,
                       rows, transformed, failure);
    checked();
    unsigned outputs = 0;
    for (unsigned t = 0; t < count; ++t) {
        coop_core(w[t], transformed + std::size_t(t) * rows * w[0].inputs, rows, out[t], const_cast<float*>(g.partial[t]),
                  g.splits[t], failure, q);
        outputs = std::max(outputs, w[t].outputs);
    }
    hipLaunchKernelGGL(coop_output_group_kernel, dim3((rows * (outputs / 128) + 3) / 4, count), dim3(128), 0, q, g, rows, failure);
    checked();
    return true;
}
void device_linear_v2(LinearView w, const float* x, unsigned rows, float* out, void* scratch, unsigned* failure,
                      bool shared, hipStream_t q) {
    // Fused single-launch form, opt-in (STRIX_G4_EXL3_V2_FUSED=1): measured slower
    // than the three-launch form at C1..C8 (1024-thread blocks). Same bytes. The
    // input transform is fused only below the head's size (redundant per block).
    static const bool fused = [] {
        const auto* p = std::getenv("STRIX_G4_EXL3_V2_FUSED");
        return p && std::string_view(p) == "1";
    }();
    if (fused && shared && w.outputs <= 65536 && coop_fused(w, nullptr, x, rows, out, failure, q)) return;
    auto* transformed = static_cast<_Float16*>(scratch);
    hipLaunchKernelGGL(input_kernel<>, dim3((rows * (w.inputs / 128) + 3) / 4), dim3(128), 0, q, w, x, rows, transformed,
                       failure, nullptr);
    checked();
    const unsigned splits = v2_splits(w.inputs, w.outputs);
    auto* partial = reinterpret_cast<float*>(static_cast<std::uint8_t*>(scratch) + v2_partial_offset);
    if (splits > 1 && std::size_t(splits) * rows * w.outputs * sizeof(float) > v2_partial_bytes)
        throw std::invalid_argument("EXL3 v2 partial scratch");
    if (!shared) {
        for (unsigned r = 0; r < rows; ++r) {
            coop_core(w, transformed + std::size_t(r) * w.inputs, 1, out + std::size_t(r) * w.outputs, partial, splits, failure, q);
            if (splits > 1) {
                hipLaunchKernelGGL(coop_output_kernel, dim3((w.outputs / 128 + 3) / 4), dim3(128), 0, q, w, partial, splits,
                                   out + std::size_t(r) * w.outputs, 1U, failure);
                checked();
            }
        }
        if (splits == 1) {
            hipLaunchKernelGGL(output_kernel, dim3((rows * (w.outputs / 128) + 3) / 4), dim3(128), 0, q, w, out, rows, failure);
            checked();
        }
        return;
    }
    if (fused && coop_fused(w, transformed, nullptr, rows, out, failure, q)) return;
    coop_core(w, transformed, rows, out, partial, splits, failure, q);
    if (splits > 1)
        hipLaunchKernelGGL(coop_output_kernel, dim3((rows * (w.outputs / 128) + 3) / 4), dim3(128), 0, q, w, partial, splits,
                           out, rows, failure);
    else
        hipLaunchKernelGGL(output_kernel, dim3((rows * (w.outputs / 128) + 3) / 4), dim3(128), 0, q, w, out, rows, failure);
    checked();
}
// source != nullptr additionally fuses the input transform (when it fits LDS).
bool fused_core(LinearView w, const _Float16* x, const float* source, unsigned rows, float* out, unsigned* failure,
                hipStream_t q) {
    static const bool enabled = env_on("STRIX_G4_EXL3_FUSED_OUTPUT") && env_on("STRIX_G4_EXL3_LANE_TILE");
    static const bool input = env_on("STRIX_G4_EXL3_FUSED_INPUT");
    // Measured: the 128-output blocks lose to the narrow lane kernels beyond two rows.
    if (!enabled || rows > 2 || (source && !input)) return false;
    bool launched = false;
    switch (rows) {
#define G4_EXL3_FROWS(R) case R:                                                                                  \
        if (lane_groups(w, R) == 8) launched = lane_staged(R) ? launch_fused<R, 8, true>(w, x, source, out, failure, q)  \
                                                              : launch_fused<R, 8, false>(w, x, source, out, failure, q); \
        else launched = lane_staged(R) ? launch_fused<R, 4, true>(w, x, source, out, failure, q)                        \
                                       : launch_fused<R, 4, false>(w, x, source, out, failure, q);                       \
        break
        G4_EXL3_FROWS(1); G4_EXL3_FROWS(2); G4_EXL3_FROWS(3); G4_EXL3_FROWS(4);
        G4_EXL3_FROWS(5); G4_EXL3_FROWS(6); G4_EXL3_FROWS(7); G4_EXL3_FROWS(8);
#undef G4_EXL3_FROWS
    }
    if (launched) checked();
    return launched;
}
// Core plus output transform for one linear over its transformed scratch.
// Returns whether the output transform was fused into the core launch(es).
bool core_output(LinearView w, const _Float16* x, unsigned rows, float* out, unsigned* failure, bool shared,
                 gpu::Queue queue, bool output) {
    const auto q = static_cast<hipStream_t>(queue);
    bool fused = true;
    if (!shared) {
        for (unsigned r = 0; r < rows; ++r) {
            const auto* xr = x + std::size_t(r) * w.inputs;
            auto* yr = out + std::size_t(r) * w.outputs;
            if (!fused_core(w, xr, nullptr, 1, yr, failure, q)) { fused = false; device_core(w, xr, 1, yr, failure, CoreRoute::automatic, queue); }
        }
    } else if (!fused_core(w, x, nullptr, rows, out, failure, q)) {
        fused = false;
        device_core(w, x, rows, out, failure, CoreRoute::automatic, queue);
    }
    if (!fused && output) {
        hipLaunchKernelGGL(output_kernel, dim3((rows * (w.outputs / 128) + 3) / 4), dim3(128), 0, q, w, out, rows, failure);
        checked();
    }
    return fused;
}
// All members in one fused launch (R <= 2, half-tile members); false launches nothing.
// STRIX_G4_EXL3_GROUP_LAUNCH=0 issues one fused launch per member instead.
bool group_launch(const LinearView* w, unsigned count, const float* x, unsigned rows, float* const* out, void* scratch,
                  unsigned* failure, hipStream_t q) {
    static const bool enabled = env_on("STRIX_G4_EXL3_GROUP_LAUNCH") && env_on("STRIX_G4_EXL3_FUSED_OUTPUT") &&
                                env_on("STRIX_G4_EXL3_LANE_TILE");
    static const bool input = env_on("STRIX_G4_EXL3_FUSED_INPUT");
    if (!enabled || rows > 2 || count < 2 || count > max_group) return false;
    FusedGroup g{};
    g.count = count;
    for (unsigned t = 0; t < count; ++t) {
        if (!geometry(w[t]) || w[t].inputs != w[0].inputs || !out[t]) throw std::invalid_argument("EXL3 group geometry");
        if (lane_groups(w[t], rows) != 4) return false;
        g.w[t] = w[t]; g.out[t] = out[t];
        g.first[t + 1] = g.first[t] + w[t].outputs / 128;
    }
    const bool staged = lane_staged(rows);
    const std::size_t fixed = rows * 128 * sizeof(float) + (staged ? 16 * 32 * group_stride * sizeof(Word4) : sizeof(Word4));
    const std::size_t dynamic = std::size_t(rows) * w[0].inputs * sizeof(_Float16);
    const bool in = input && fixed + dynamic <= lds_budget;
    if (!in) {
        Group transforms{};
        for (unsigned t = 0; t < count; ++t) transforms.w[t] = w[t];
        auto* half_scratch = static_cast<_Float16*>(scratch);
        hipLaunchKernelGGL(input_group_kernel, dim3((rows * (w[0].inputs / 128) + 3) / 4, count), dim3(128), 0, q,
                           transforms, x, rows, half_scratch, failure);
        checked();
        for (unsigned t = 0; t < count; ++t) g.x[t] = half_scratch + std::size_t(t) * rows * w[0].inputs;
    }
    const dim3 blocks(g.first[count]), threads(512);
    const auto* source = in ? x : nullptr;
    const auto bytes = in ? dynamic : 0;
#define G4_EXL3_GROUP(R, S, I) hipLaunchKernelGGL((group_fused_kernel<R, S, I>), blocks, threads, bytes, q, g, source, failure)
    if (rows == 1) {
        if (staged) { if (in) G4_EXL3_GROUP(1, true, true); else G4_EXL3_GROUP(1, true, false); }
        else { if (in) G4_EXL3_GROUP(1, false, true); else G4_EXL3_GROUP(1, false, false); }
    } else {
        if (staged) { if (in) G4_EXL3_GROUP(2, true, true); else G4_EXL3_GROUP(2, true, false); }
        else { if (in) G4_EXL3_GROUP(2, false, true); else G4_EXL3_GROUP(2, false, false); }
    }
#undef G4_EXL3_GROUP
    checked();
    return true;
}
}
void device_input(LinearView w, const float* x, unsigned rows, void* scratch, unsigned* failure, gpu::Queue queue) {
    if (!geometry(w) || !rows || rows > 8 || !x || !scratch || !failure) throw std::invalid_argument("EXL3 device input arguments");
    hipLaunchKernelGGL(input_kernel<>, dim3((rows * (w.inputs / 128) + 3) / 4), dim3(128), 0, static_cast<hipStream_t>(queue), w, x,
                       rows, static_cast<_Float16*>(scratch), failure, nullptr);
    checked();
}
void device_codebook_table(void* out, gpu::Queue queue) {
    hipLaunchKernelGGL(codebook_table_kernel, dim3(128), dim3(256), 0, static_cast<hipStream_t>(queue), static_cast<std::uint16_t*>(out));
    checked();
}
void device_core(LinearView w, const void* transformed, unsigned rows, float* out, unsigned* failure,
                 CoreRoute route, gpu::Queue queue) {
    if (!geometry(w) || !rows || rows > 8 || !transformed || !out || !failure) throw std::invalid_argument("EXL3 device core arguments");
    const auto q = static_cast<hipStream_t>(queue);
    const auto* x = static_cast<const _Float16*>(transformed);
    switch (rows) {
#define G4_EXL3_ROWS(R) case R: launch_core_route<R>(w, x, out, failure, route, q); break
        G4_EXL3_ROWS(1); G4_EXL3_ROWS(2); G4_EXL3_ROWS(3); G4_EXL3_ROWS(4);
        G4_EXL3_ROWS(5); G4_EXL3_ROWS(6); G4_EXL3_ROWS(7); G4_EXL3_ROWS(8);
#undef G4_EXL3_ROWS
    }
    checked();
}
void device_linear(LinearView w, const float* x, unsigned rows, float* out, void* scratch,
                   unsigned* failure, bool shared, gpu::Queue queue) {
    if (!geometry(w) || !rows || rows > 8 || !x || !out || !scratch || !failure) throw std::invalid_argument("EXL3 device linear arguments");
    const auto q = static_cast<hipStream_t>(queue);
    if (v2_active()) { device_linear_v2(w, x, rows, out, scratch, failure, shared, q); return; }
    if (shared && fused_core(w, nullptr, x, rows, out, failure, q)) return;
    hipLaunchKernelGGL(input_kernel<>, dim3((rows * (w.inputs / 128) + 3) / 4), dim3(128), 0, q, w, x, rows,
                       static_cast<_Float16*>(scratch), failure, nullptr);
    checked();
    core_output(w, static_cast<_Float16*>(scratch), rows, out, failure, shared, queue, true);
}
void device_linear_group(const LinearView* w, unsigned count, const float* x, unsigned rows, float* const* out,
                         void* scratch, unsigned* failure, bool shared, gpu::Queue queue) {
    static const bool grouped = env_on("STRIX_G4_EXL3_GROUP");
    if (grouped && shared && count > 1 && v2_active() && rows <= 8 && x && scratch && failure &&
        device_group_v2(w, count, x, rows, out, scratch, failure, static_cast<hipStream_t>(queue)))
        return;
    if (!grouped || count == 1 || v2_active()) {
        for (unsigned t = 0; t < count; ++t) device_linear(w[t], x, rows, out[t], scratch, failure, shared, queue);
        return;
    }
    if (!count || count > max_group || !rows || rows > 8 || !x || !scratch || !failure) throw std::invalid_argument("EXL3 group arguments");
    if (shared && group_launch(w, count, x, rows, out, scratch, failure, static_cast<hipStream_t>(queue))) return;
    if (shared) {
        // Fully fused members need no shared transform launch; others take device_linear.
        bool all = true, done[max_group] = {};
        for (unsigned t = 0; t < count; ++t) {
            if (!geometry(w[t]) || w[t].inputs != w[0].inputs || !out[t]) throw std::invalid_argument("EXL3 group geometry");
            done[t] = fused_core(w[t], nullptr, x, rows, out[t], failure, static_cast<hipStream_t>(queue));
            all = all && done[t];
        }
        if (all) return;
        if (std::any_of(done, done + count, [](bool d) { return d; })) {
            for (unsigned t = 0; t < count; ++t)
                if (!done[t]) device_linear(w[t], x, rows, out[t], scratch, failure, shared, queue);
            return;
        }
    }
    Group g{};
    unsigned outputs = 0;
    for (unsigned t = 0; t < count; ++t) {
        if (!geometry(w[t]) || w[t].inputs != w[0].inputs || !out[t]) throw std::invalid_argument("EXL3 group geometry");
        g.w[t] = w[t]; g.out[t] = out[t]; outputs = std::max(outputs, w[t].outputs);
    }
    const auto q = static_cast<hipStream_t>(queue);
    auto* half_scratch = static_cast<_Float16*>(scratch);
    hipLaunchKernelGGL(input_group_kernel, dim3((rows * (w[0].inputs / 128) + 3) / 4, count), dim3(128), 0, q, g, x, rows,
                       half_scratch, failure);
    checked();
    // Fusion depends only on rows and switches, so it is uniform across members.
    bool fused = true;
    for (unsigned t = 0; t < count; ++t)
        fused = core_output(w[t], half_scratch + std::size_t(t) * rows * w[0].inputs, rows, out[t], failure, shared, queue,
                            false) && fused;
    if (!fused) {
        hipLaunchKernelGGL(output_group_kernel, dim3((rows * (outputs / 128) + 3) / 4, count), dim3(128), 0, q, g, rows, failure);
        checked();
    }
}
namespace {
// Many-row cores for prompt chunks. Every output keeps its identity's order:
// v1 is lane_body's (lane l owns chunks l mod 32, e ascending, one exact FP16
// product per FP32 add, then the lane butterfly); v2/v3 is coop_body's (split
// ranges, lanes a = 0..3 over (2a, 2a+1, 2a+8, 2a+9), quad reduction, split
// sum in increasing order). A workgroup decodes each lane's coefficients once
// and applies them to R rows whose transformed FP16 inputs are staged in LDS.
// No row affects another row's value, so the bytes equal the eight-row passes.
constexpr unsigned rows_v1 = 16, rows_v2 = 32;
// One K block (32 chunks) of one lane's half tile against R staged rows; the
// pair layout [row][e/2][lane] gives each lane its own consecutive word.
template <unsigned K, unsigned R, unsigned G0>
__device__ __forceinline__ void rows_lane_block(const std::uint32_t* tile, const std::uint32_t* staged, unsigned lane,
                                                float (&acc)[4][2][R]) {
    LaneTile<K, 4, G0> current;
    current.load(tile);
    unrolled<8>([&]<unsigned P>() {
        float wl[2][4], wh[2][4];
        unrolled<2>([&]<unsigned E>() {
            unrolled<4>([&]<unsigned g>() {
                wl[E][g] = float(codebook_half(current.template state<position_of(2 * P + E, G0 + g)>()));
                wh[E][g] = float(codebook_half(current.template state<position_of(2 * P + E, G0 + g + 8)>()));
            });
        });
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            const std::uint32_t pair = staged[(r * 8 + P) * 32 + lane];
            const float in0 = float(__builtin_bit_cast(_Float16, std::uint16_t(pair))),
                        in1 = float(__builtin_bit_cast(_Float16, std::uint16_t(pair >> 16U)));
#pragma unroll
            for (unsigned g = 0; g < 4; ++g) {
                acc[g][0][r] = __builtin_fmaf(wl[0][g], in0, acc[g][0][r]);
                acc[g][1][r] = __builtin_fmaf(wh[0][g], in0, acc[g][1][r]);
            }
#pragma unroll
            for (unsigned g = 0; g < 4; ++g) {
                acc[g][0][r] = __builtin_fmaf(wl[1][g], in1, acc[g][0][r]);
                acc[g][1][r] = __builtin_fmaf(wh[1][g], in1, acc[g][1][r]);
            }
        }
    });
}
// Eight waves = four 16-output tile columns (two half tiles each) x R rows.
template <unsigned K, unsigned R>
__global__ __launch_bounds__(256) void rows_lane_kernel(LinearView w, const _Float16* x, unsigned rows, float* out,
                                                        unsigned* failure) {
    __shared__ std::uint32_t staged[R * 8 * 32];
    const unsigned wave = threadIdx.x / 32, lane = threadIdx.x % 32, row0 = blockIdx.x * R;
    const unsigned n = blockIdx.y * 4 + wave / 2, chunks = w.inputs / 16, tiles = w.outputs / 16;
    const bool active_tile = n < tiles;
    const auto* core = reinterpret_cast<const std::uint32_t*>(w.data + 2ULL * (w.inputs + w.outputs));
    const auto* words = reinterpret_cast<const std::uint32_t*>(x);
    float acc[4][2][R] = {};
    for (unsigned b = 0; b < (chunks + 31) / 32; ++b) {
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < R * 8 * 32 / 256; ++i) {
            // Global order (row, lane, pair) keeps the loads contiguous.
            const unsigned j = threadIdx.x + 256 * i, r = j / 256, l = (j / 8) % 32, p = j % 8, c = b * 32 + l;
            if (c < chunks)
                staged[(r * 8 + p) * 32 + l] = words[(std::size_t(min(row0 + r, rows - 1)) * w.inputs + c * 16) / 2 + p];
        }
        __syncthreads();
        const unsigned c = b * 32 + lane;
        if (active_tile && c < chunks) {
            const auto* tile = core + (std::size_t(c) * tiles + n) * (8 * K);
            if (wave & 1U) rows_lane_block<K, R, 4>(tile, staged, lane, acc);
            else rows_lane_block<K, R, 0>(tile, staged, lane, acc);
        }
    }
    if (!active_tile) return;
    const unsigned g0 = (wave & 1U) * 4;
#pragma unroll
    for (unsigned g = 0; g < 4; ++g)
#pragma unroll
        for (unsigned h = 0; h < 2; ++h)
#pragma unroll
            for (unsigned r = 0; r < R; ++r) {
                auto value = acc[g][h][r];
                for (unsigned d = 16; d; d >>= 1U) value = value + __shfl_down(value, d, 32);
                if (!lane && row0 + r < rows) {
                    out[std::size_t(row0 + r) * w.outputs + n * 16 + g0 + g + 8 * h] = value;
                    flag(value, failure);
                }
            }
}
// Eight waves = eight tile columns (128 outputs) of one split x R rows. Chunks
// of the split are staged 32 at a time as [row][chunk][pair].
// Paired: each staged chunk keeps lane a's low/high activation words adjacent
// (one 64-bit LDS read per row with an immediate row offset) and the next
// chunk's raw trellis words are fetched before the current chunk's FMAs. Same
// values in the same chains and order as Paired = false.
template <unsigned K, unsigned R, bool Paired = false>
__global__ __launch_bounds__(256) void rows_coop_kernel(LinearView w, const _Float16* x, unsigned rows, float* out,
                                                        float* partial, unsigned splits, unsigned* failure) {
    __shared__ alignas(8) std::uint32_t staged[R * 32 * 8];
    const unsigned wave = threadIdx.x / 32, lane = threadIdx.x % 32, row0 = blockIdx.x * R;
    const unsigned tile = blockIdx.y * 8 + wave, split = blockIdx.z, tiles = w.outputs / 16;
    const unsigned a = lane & 3U, column = lane >> 2U, chunks = w.inputs / 16, span = chunks / splits;
    const unsigned begin = split * span, end = begin + span;
    const bool active_tile = tile < tiles;
    const auto* core = reinterpret_cast<const std::uint32_t*>(w.data + 2ULL * (w.inputs + w.outputs));
    const auto* words = reinterpret_cast<const std::uint32_t*>(x);
    float acc[2][R] = {};
    for (unsigned k0 = begin; k0 < end; k0 += 32) {
        const unsigned count = min(32U, end - k0);
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < R * 32 * 8 / 256; ++i) {
            const unsigned j = threadIdx.x + 256 * i, r = j / 256, kk = (j / 8) % 32, p = j % 8;
            if (kk < count)
                staged[(r * 32 + kk) * 8 + (Paired ? 2 * (p & 3U) + (p >> 2U) : p)] =
                    words[(std::size_t(min(row0 + r, rows - 1)) * w.inputs + (k0 + kk) * 16) / 2 + p];
        }
        __syncthreads();
        if (!active_tile) continue;
        if constexpr (Paired) {
            using Pair = std::uint32_t __attribute__((ext_vector_type(2)));
            CoopWords<K> next;
            next.fetch(core + (std::size_t(k0) * tiles + tile) * (8 * K), lane);
            for (unsigned kk = 0; kk < count; ++kk) {
                CoopLane<K> t;
                t.align(next.a, next.b, next.c, lane);
                if (kk + 1 < count) next.fetch(core + (std::size_t(k0 + kk + 1) * tiles + tile) * (8 * K), lane);
                const float c0 = float(codebook_half(t.template state<0>())), c1 = float(codebook_half(t.template state<1>()));
                const float c2 = float(codebook_half(t.template state<2>())), c3 = float(codebook_half(t.template state<3>()));
                const float c4 = float(codebook_half(t.template state<4>())), c5 = float(codebook_half(t.template state<5>()));
                const float c6 = float(codebook_half(t.template state<6>())), c7 = float(codebook_half(t.template state<7>()));
                const auto* chunk = reinterpret_cast<const Pair*>(staged + kk * 8 + 2 * a);
#pragma unroll
                for (unsigned r = 0; r < R; ++r) {
                    const Pair pair = chunk[r * 32 * 8 / 2];
                    const std::uint32_t low = pair.x, high = pair.y;
                    const float x0 = float(__builtin_bit_cast(_Float16, std::uint16_t(low))),
                                x1 = float(__builtin_bit_cast(_Float16, std::uint16_t(low >> 16U)));
                    const float x8 = float(__builtin_bit_cast(_Float16, std::uint16_t(high))),
                                x9 = float(__builtin_bit_cast(_Float16, std::uint16_t(high >> 16U)));
                    acc[0][r] = __builtin_fmaf(c0, x0, acc[0][r]); acc[0][r] = __builtin_fmaf(c1, x1, acc[0][r]);
                    acc[0][r] = __builtin_fmaf(c2, x8, acc[0][r]); acc[0][r] = __builtin_fmaf(c3, x9, acc[0][r]);
                    acc[1][r] = __builtin_fmaf(c4, x0, acc[1][r]); acc[1][r] = __builtin_fmaf(c5, x1, acc[1][r]);
                    acc[1][r] = __builtin_fmaf(c6, x8, acc[1][r]); acc[1][r] = __builtin_fmaf(c7, x9, acc[1][r]);
                }
            }
            continue;
        }
        for (unsigned kk = 0; kk < count; ++kk) {
            CoopLane<K> t;
            t.load(core + (std::size_t(k0 + kk) * tiles + tile) * (8 * K), lane);
            const float c0 = float(codebook_half(t.template state<0>())), c1 = float(codebook_half(t.template state<1>()));
            const float c2 = float(codebook_half(t.template state<2>())), c3 = float(codebook_half(t.template state<3>()));
            const float c4 = float(codebook_half(t.template state<4>())), c5 = float(codebook_half(t.template state<5>()));
            const float c6 = float(codebook_half(t.template state<6>())), c7 = float(codebook_half(t.template state<7>()));
#pragma unroll
            for (unsigned r = 0; r < R; ++r) {
                const std::uint32_t low = staged[(r * 32 + kk) * 8 + a], high = staged[(r * 32 + kk) * 8 + a + 4];
                const float x0 = float(__builtin_bit_cast(_Float16, std::uint16_t(low))),
                            x1 = float(__builtin_bit_cast(_Float16, std::uint16_t(low >> 16U)));
                const float x8 = float(__builtin_bit_cast(_Float16, std::uint16_t(high))),
                            x9 = float(__builtin_bit_cast(_Float16, std::uint16_t(high >> 16U)));
                acc[0][r] = __builtin_fmaf(c0, x0, acc[0][r]); acc[0][r] = __builtin_fmaf(c1, x1, acc[0][r]);
                acc[0][r] = __builtin_fmaf(c2, x8, acc[0][r]); acc[0][r] = __builtin_fmaf(c3, x9, acc[0][r]);
                acc[1][r] = __builtin_fmaf(c4, x0, acc[1][r]); acc[1][r] = __builtin_fmaf(c5, x1, acc[1][r]);
                acc[1][r] = __builtin_fmaf(c6, x8, acc[1][r]); acc[1][r] = __builtin_fmaf(c7, x9, acc[1][r]);
            }
        }
    }
    if (!active_tile) return;
#pragma unroll
    for (unsigned h = 0; h < 2; ++h)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            acc[h][r] = acc[h][r] + __shfl_down(acc[h][r], 2, 4);
            acc[h][r] = acc[h][r] + __shfl_down(acc[h][r], 1, 4);
        }
#pragma unroll
    for (unsigned h = 0; h < 2; ++h)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            const float v = acc[h][r];
            if (!a && row0 + r < rows) {
                const auto o = tile * 16 + column + 8 * h;
                if (splits == 1) { out[std::size_t(row0 + r) * w.outputs + o] = v; flag(v, failure); }
                else partial[(std::size_t(split) * rows + row0 + r) * w.outputs + o] = v;
            }
        }
}
// FFN middle stage of the many-row route: gate and up core values (split
// partials summed in increasing split order), their output transforms, then
// gelu_tanh(gate) * up and down's input transform, all in registers. The
// operations are coop_output/output_kernel's, gpu::gelu_product's and
// input_kernel's, without their FP32 round trips through memory.
__device__ __forceinline__ void core_values(LinearView w, const float* source, unsigned splits, unsigned rows, unsigned row,
                                            unsigned begin, float (&v)[4], unsigned* failure) {
    const auto lane = threadIdx.x % 32;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const auto o = begin + lane + 32 * j;
        v[j] = source[std::size_t(row) * w.outputs + o];
        if (splits > 1) {
            for (unsigned s = 1; s < splits; ++s) v[j] = v[j] + source[(std::size_t(s) * rows + row) * w.outputs + o];
            flag(v[j], failure);
        }
    }
    output_values(w, begin, v, failure);
}
// Panels/maximum: down's input as the prefill GEMMs' [K16][row][16] panels and
// the product's abs-bits maximum (input_kernel's record of an FP32 product).
template <bool Panels = false>
__global__ void ffn_middle_kernel(LinearView gate, LinearView up, LinearView down, const float* g, unsigned gate_splits,
                                  const float* u, unsigned up_splits, unsigned rows, _Float16* out, unsigned* failure,
                                  unsigned* maximum = nullptr) {
    const auto group = std::size_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (group >= std::size_t(rows) * (down.inputs / 128)) return;
    const auto row = unsigned(group / (down.inputs / 128)), begin = unsigned(group % (down.inputs / 128)) * 128;
    const auto lane = threadIdx.x % 32;
    float a[4], b[4], v[4];
    core_values(gate, g, gate_splits, rows, row, begin, a, failure);
    core_values(up, u, up_splits, rows, row, begin, b, failure);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const auto i = begin + lane + 32 * j;
        const float product = gelu_tanh(a[j]) * b[j];
        if (maximum) atomicMax(maximum, bit_cast<unsigned>(product) & 0x7fffffffU);
        v[j] = float(half(materialize(product * fp16_value(load16(down.data + 2ULL * i))), failure));
    }
    transform(v, failure);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const auto i = begin + lane + 32 * j;
        out[Panels ? std::size_t(i / 16) * rows * 16 + row * 16 + i % 16 : std::size_t(row) * down.inputs + i] = half(v[j], failure);
    }
}
std::size_t transformed_bytes(LinearView w, unsigned rows) {
    return (std::size_t(rows) * w.inputs * sizeof(_Float16) + 255) / 256 * 256;
}
std::size_t partial_bytes(LinearView w, unsigned rows) {
    const unsigned splits = v2_splits(w.inputs, w.outputs);
    return splits > 1 ? std::size_t(splits) * rows * w.outputs * sizeof(float) : 0;
}
// Paired LDS layout + next-chunk raw codec prefetch in the many-row core (same
// bytes); STRIX_G4_EXL3_ROWS_PAIRED=0 runs the previous schedule.
bool rows_paired() {
    static const bool enabled = env_on("STRIX_G4_EXL3_ROWS_PAIRED");
    return enabled;
}
// v2/v3 many-row core: raw values into out (one split) or split partials.
void rows_core(LinearView w, const _Float16* transformed, unsigned rows, float* out, float* partial, unsigned splits,
               unsigned* failure, hipStream_t q) {
    const dim3 grid((rows + rows_v2 - 1) / rows_v2, (w.outputs / 16 + 7) / 8, splits);
    switch (bits(w)) {
#define G4_EXL3_RC(K) case K: \
        if (rows_paired()) hipLaunchKernelGGL((rows_coop_kernel<K, rows_v2, true>), grid, dim3(256), 0, q, w, transformed, rows, out, partial, splits, failure); \
        else hipLaunchKernelGGL((rows_coop_kernel<K, rows_v2>), grid, dim3(256), 0, q, w, transformed, rows, out, partial, splits, failure); break
        G4_EXL3_RC(4); G4_EXL3_RC(5); G4_EXL3_RC(6); G4_EXL3_RC(8);
#undef G4_EXL3_RC
    }
    checked();
}
bool ffn_fuse_enabled() {
    static const bool enabled = env_on("STRIX_G4_EXL3_FFN_FUSE");
    return enabled;
}
} // namespace
std::size_t rows_scratch_bytes(LinearView w, unsigned rows) {
    const unsigned splits = v2_active() ? v2_splits(w.inputs, w.outputs) : 1U;
    return transformed_bytes(w, rows) + (splits > 1 ? std::size_t(splits) * rows * w.outputs * sizeof(float) : 0);
}
bool device_linear_rows(LinearView w, const float* x, unsigned rows, float* out, void* scratch, std::size_t scratch_bytes,
                        unsigned* failure, gpu::Queue queue) {
    if (!rows_enabled() || rows <= 8 || !geometry(w)) return false;
    if (!x || !out || !scratch || !failure || scratch_bytes < rows_scratch_bytes(w, rows))
        throw std::invalid_argument("EXL3 many-row arguments");
    const auto q = static_cast<hipStream_t>(queue);
    auto* transformed = static_cast<_Float16*>(scratch);
    hipLaunchKernelGGL(input_kernel<>, dim3((rows * (w.inputs / 128) + 3) / 4), dim3(128), 0, q, w, x, rows, transformed,
                       failure, nullptr);
    checked();
    if (v2_active()) {
        const unsigned splits = v2_splits(w.inputs, w.outputs);
        auto* partial = reinterpret_cast<float*>(static_cast<std::uint8_t*>(scratch) + transformed_bytes(w, rows));
        rows_core(w, transformed, rows, out, partial, splits, failure, q);
        if (splits > 1)
            hipLaunchKernelGGL(coop_output_kernel, dim3((rows * (w.outputs / 128) + 3) / 4), dim3(128), 0, q, w, partial, splits,
                               out, rows, failure);
        else
            hipLaunchKernelGGL(output_kernel, dim3((rows * (w.outputs / 128) + 3) / 4), dim3(128), 0, q, w, out, rows, failure);
        checked();
        return true;
    }
    const dim3 grid((rows + rows_v1 - 1) / rows_v1, (w.outputs / 16 + 3) / 4);
    switch (bits(w)) {
#define G4_EXL3_RL(K) case K: hipLaunchKernelGGL((rows_lane_kernel<K, rows_v1>), grid, dim3(256), 0, q, w, transformed, rows, out, failure); break
        G4_EXL3_RL(4); G4_EXL3_RL(5); G4_EXL3_RL(6); G4_EXL3_RL(8);
#undef G4_EXL3_RL
    }
    checked();
    hipLaunchKernelGGL(output_kernel, dim3((rows * (w.outputs / 128) + 3) / 4), dim3(128), 0, q, w, out, rows, failure);
    checked();
    return true;
}
std::size_t ffn_rows_scratch_bytes(LinearView gate, LinearView up, LinearView down, unsigned rows) {
    if (!v2_active()) return 0;
    return transformed_bytes(gate, rows) + transformed_bytes(up, rows) + transformed_bytes(down, rows) +
           partial_bytes(gate, rows) + partial_bytes(up, rows) + partial_bytes(down, rows);
}
bool device_ffn_rows(LinearView gate, LinearView up, LinearView down, const float* x, unsigned rows, float* gate_out,
                     float* up_out, float* out, void* scratch, std::size_t scratch_bytes, unsigned* failure, gpu::Queue queue) {
    if (!rows_enabled() || !ffn_fuse_enabled() || !v2_active() || rows <= 8 || !geometry(gate) || !geometry(up) ||
        !geometry(down) || gate.inputs != up.inputs || gate.outputs != up.outputs || down.inputs != gate.outputs)
        return false;
    if (!x || !gate_out || !up_out || !out || !scratch || !failure || scratch_bytes < ffn_rows_scratch_bytes(gate, up, down, rows))
        throw std::invalid_argument("EXL3 many-row FFN arguments");
    const auto q = static_cast<hipStream_t>(queue);
    auto* at = static_cast<std::uint8_t*>(scratch);
    const auto take = [&](std::size_t bytes) { auto* p = at; at += bytes; return p; };
    auto* tg = reinterpret_cast<_Float16*>(take(transformed_bytes(gate, rows)));
    auto* tu = reinterpret_cast<_Float16*>(take(transformed_bytes(up, rows)));
    auto* td = reinterpret_cast<_Float16*>(take(transformed_bytes(down, rows)));
    auto* pg = reinterpret_cast<float*>(take(partial_bytes(gate, rows)));
    auto* pu = reinterpret_cast<float*>(take(partial_bytes(up, rows)));
    auto* pd = reinterpret_cast<float*>(take(partial_bytes(down, rows)));
    const unsigned sg = v2_splits(gate.inputs, gate.outputs), su = v2_splits(up.inputs, up.outputs);
    const unsigned sd = v2_splits(down.inputs, down.outputs);
    hipLaunchKernelGGL(input_kernel<>, dim3((rows * (gate.inputs / 128) + 3) / 4), dim3(128), 0, q, gate, x, rows, tg, failure, nullptr);
    hipLaunchKernelGGL(input_kernel<>, dim3((rows * (up.inputs / 128) + 3) / 4), dim3(128), 0, q, up, x, rows, tu, failure, nullptr);
    checked();
    rows_core(gate, tg, rows, gate_out, pg, sg, failure, q);
    rows_core(up, tu, rows, up_out, pu, su, failure, q);
    hipLaunchKernelGGL(ffn_middle_kernel<>, dim3((rows * (down.inputs / 128) + 3) / 4), dim3(128), 0, q, gate, up, down,
                       sg > 1 ? pg : gate_out, sg, su > 1 ? pu : up_out, su, rows, td, failure);
    checked();
    rows_core(down, td, rows, out, pd, sd, failure, q);
    if (sd > 1)
        hipLaunchKernelGGL(coop_output_kernel, dim3((rows * (down.outputs / 128) + 3) / 4), dim3(128), 0, q, down, pd, sd, out,
                           rows, failure);
    else
        hipLaunchKernelGGL(output_kernel, dim3((rows * (down.outputs / 128) + 3) / 4), dim3(128), 0, q, down, out, rows, failure);
    checked();
    return true;
}
void device_input_panels(LinearView w, const float* x, unsigned rows, void* scratch, unsigned* failure, unsigned* maximum,
                         gpu::Queue queue) {
    if (!geometry(w) || !rows || rows > prefill::max_chunk_rows || !x || !scratch || !failure)
        throw std::invalid_argument("EXL3 panel input arguments");
    hipLaunchKernelGGL(input_kernel<true>, dim3((rows * (w.inputs / 128) + 3) / 4), dim3(128), 0, static_cast<hipStream_t>(queue),
                       w, x, rows, static_cast<_Float16*>(scratch), failure, maximum);
    checked();
}
void device_output(LinearView w, float* out, unsigned rows, unsigned* failure, gpu::Queue queue) {
    if (!geometry(w) || !rows || !out || !failure) throw std::invalid_argument("EXL3 output arguments");
    hipLaunchKernelGGL(output_kernel, dim3((rows * (w.outputs / 128) + 3) / 4), dim3(128), 0, static_cast<hipStream_t>(queue),
                       w, out, rows, failure);
    checked();
}
void device_ffn_middle_panels(LinearView gate, LinearView up, LinearView down, const float* gate_core, const float* up_core,
                              unsigned rows, void* product, unsigned* failure, unsigned* maximum, gpu::Queue queue) {
    if (!geometry(gate) || !geometry(up) || !geometry(down) || gate.inputs != up.inputs || gate.outputs != up.outputs ||
        down.inputs != gate.outputs || !rows || rows > prefill::max_chunk_rows || !gate_core || !up_core || !product || !failure)
        throw std::invalid_argument("EXL3 FFN middle arguments");
    hipLaunchKernelGGL(ffn_middle_kernel<true>, dim3((rows * (down.inputs / 128) + 3) / 4), dim3(128), 0,
                       static_cast<hipStream_t>(queue), gate, up, down, gate_core, 1U, up_core, 1U, rows,
                       static_cast<_Float16*>(product), failure, maximum);
    checked();
}
void device_prefill(LinearView w, const float* x, unsigned rows, float* out, void* scratch,
                    unsigned* failure, unsigned* maximum, const void* panels, gpu::Queue queue, void* converted_event) {
    if (!geometry(w) || !rows || rows > prefill::max_chunk_rows || !x || !out || !scratch || !failure || !maximum)
        throw std::invalid_argument("EXL3 WMMA arguments");
    const auto q = static_cast<hipStream_t>(queue);
    hipLaunchKernelGGL(input_kernel<>, dim3((rows * (w.inputs / 128) + 3) / 4), dim3(128), 0, q, w, x, rows,
                       static_cast<_Float16*>(scratch), failure, maximum);
    checked();
    if (converted_event) {
        const auto e = hipEventRecord(static_cast<hipEvent_t>(converted_event), q);
        if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
    }
    switch (bits(w)) {
#define G4_EXL3_WMMA(K) case K: hipLaunchKernelGGL((wmma_kernel<K>), dim3(w.outputs / 64, (rows + 63) / 64), dim3(128), 0, q, w, static_cast<_Float16*>(scratch), rows, out, failure, static_cast<const _Float16*>(panels)); break
        G4_EXL3_WMMA(4); G4_EXL3_WMMA(5); G4_EXL3_WMMA(6); G4_EXL3_WMMA(8);
#undef G4_EXL3_WMMA
    }
    checked();
    hipLaunchKernelGGL(output_kernel, dim3((rows * (w.outputs / 128) + 3) / 4), dim3(128), 0, q, w, out, rows, failure);
    checked();
}
} // namespace strix::gemma4::exl3
