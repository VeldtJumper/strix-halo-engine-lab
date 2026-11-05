// g4-a8-v1 device kernels: A8G32 activation quantization and integer-dot linears (contract: a8.hpp).
#include "internal.hpp"
#include "chunk.hpp"
#include "a8_ops.hpp"
#include "a8_device.hpp"
#include <cstdlib>
#include <string_view>

namespace strix::gemma4::gpu {
namespace {
using a8::Meta;

__device__ inline int dot4(int w, int a, int c) { return __builtin_amdgcn_sudot4(true, w, true, a, c, false); }

// One decoded 32-element weight group: eight words of signed bytes plus its FP32 scale fields.
// The integers equal the scalar decoders of a8.hpp; only their evaluation is vectorized.
template <WeightFormat Format, bool Streaming = false> struct Group;
template <bool Streaming> struct Group<WeightFormat::q8_0, Streaming> {
    int w[8];
    float d;
    __device__ Group() = default;
    __device__ Group(const std::uint8_t* row, unsigned g, bool hardware) {
        const auto* p = row + std::size_t(g) * 34U;
        d = half_scale(load16(p), hardware);
        const uint4 a = chunk_bytes<Streaming>(p + 2), b = chunk_bytes<Streaming>(p + 18);
        w[0] = int(a.x), w[1] = int(a.y), w[2] = int(a.z), w[3] = int(a.w);
        w[4] = int(b.x), w[5] = int(b.y), w[6] = int(b.z), w[7] = int(b.w);
    }
    __device__ float term(const int (&a)[8], Meta m) const {
        int integer = 0;
#pragma unroll
        for (unsigned k = 0; k < 8; ++k)
            integer = dot4(w[k], a[k], integer);
        return a8::scaled_term(d, integer, m);
    }
};
template <bool Streaming> struct Group<WeightFormat::q4_k, Streaming> {
    int w[8];
    float d, dmin;
    int sc, m;
    __device__ Group() = default;
    __device__ Group(const std::uint8_t* row, unsigned g, bool hardware) {
        const auto* p = row + std::size_t(g / 8U) * 144U;
        const unsigned j = g % 8U;
        const uint4 h = chunk_bytes<Streaming>(p);
        sc = int(j < 4U ? chunk_byte(h, 4U + j) & 63U : (chunk_byte(h, 8U + j) & 15U) | ((chunk_byte(h, j) >> 6U) << 4U));
        m = int(j < 4U ? chunk_byte(h, 8U + j) & 63U
                       : (chunk_byte(h, 8U + j) >> 4U) | ((chunk_byte(h, 4U + j) >> 6U) << 4U));
        d = half_scale(std::uint16_t(h.x), hardware);
        dmin = half_scale(std::uint16_t(h.x >> 16U), hardware);
        const auto* q = p + 16U + (j / 2U) * 32U;
        const uint4 a = chunk_bytes<Streaming>(q), b = chunk_bytes<Streaming>(q + 16);
        const unsigned shift = (j & 1U) * 4U;
        const unsigned words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (unsigned k = 0; k < 8; ++k)
            w[k] = int((words[k] >> shift) & 0x0f0f0f0fU);
    }
    __device__ float term(const int (&a)[8], Meta meta) const {
        int integer = 0;
#pragma unroll
        for (unsigned k = 0; k < 8; ++k)
            integer = dot4(w[k], a[k], integer);
        return a8::q4_k_term(d, dmin, sc, m, integer, meta);
    }
};
template <bool Streaming> struct Group<WeightFormat::q6_k, Streaming> {
    int w[8];
    float d;
    int s0, s1;
    __device__ Group() = default;
    __device__ Group(const std::uint8_t* row, unsigned g, bool hardware) {
        const auto* p = row + std::size_t(g / 8U) * 210U;
        const unsigned j = g % 8U, half = j / 4U, quarter = j % 4U;
        const auto* low = p + half * 64U + (quarter & 1U) * 32U;
        const auto* high = p + 128U + half * 32U;
        const uint4 l0 = chunk_bytes<Streaming>(low), l1 = chunk_bytes<Streaming>(low + 16), h0 = chunk_bytes<Streaming>(high), h1 = chunk_bytes<Streaming>(high + 16);
        const unsigned lw[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
        const unsigned hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const unsigned ls = quarter >= 2U ? 4U : 0U, hs = 2U * quarter;
#pragma unroll
        for (unsigned k = 0; k < 8; ++k) {
            const unsigned q = ((lw[k] >> ls) & 0x0f0f0f0fU) | (((hw[k] >> hs) & 0x03030303U) << 4U);
            // Bytewise q - 32 without borrows between bytes: ((q | 0x80) - 0x20) ^ 0x80.
            w[k] = int(((q | 0x80808080U) - 0x20202020U) ^ 0x80808080U);
        }
        const auto s = load16(p + 192U + 2U * j);
        s0 = int(std::int8_t(s & 255U));
        s1 = int(std::int8_t(s >> 8U));
        d = half_scale(load16(p + 208), hardware);
    }
    __device__ float term(const int (&a)[8], Meta meta) const {
        int first = 0, second = 0;
#pragma unroll
        for (unsigned k = 0; k < 4; ++k) {
            first = dot4(w[k], a[k], first);
            second = dot4(w[k + 4U], a[k + 4U], second);
        }
        return a8::scaled_term(d, s0 * first + s1 * second, meta);
    }
};
// Half-group schedule (STRIX_G4_A8_HALF=1): one output per wave; lanes 2p and 2p+1 hold the two
// 16-element halves of slot p's group, exchange their exact integer partials and both form the same
// FP32 term, so slot p accumulates exactly the declared terms in the declared order.
template <WeightFormat Format> struct HalfGroup;
template <> struct HalfGroup<WeightFormat::q8_0> {
    int w[4];
    float d;
    __device__ HalfGroup() = default;
    __device__ HalfGroup(const std::uint8_t* row, unsigned g, unsigned h, bool hardware) {
        const auto* p = row + std::size_t(g) * 34U;
        d = half_scale(load16(p), hardware);
        const uint4 a = chunk_bytes(p + 2 + 16U * h);
        w[0] = int(a.x), w[1] = int(a.y), w[2] = int(a.z), w[3] = int(a.w);
    }
    __device__ float term(const int (&a)[4], Meta m) const {
        int part = 0;
#pragma unroll
        for (unsigned k = 0; k < 4; ++k)
            part = dot4(w[k], a[k], part);
        return a8::scaled_term(d, part + __shfl_xor(part, 1, 32), m);
    }
};
template <> struct HalfGroup<WeightFormat::q4_k> {
    int w[4];
    float d, dmin;
    int sc, m;
    __device__ HalfGroup() = default;
    __device__ HalfGroup(const std::uint8_t* row, unsigned g, unsigned h, bool hardware) {
        const auto* p = row + std::size_t(g / 8U) * 144U;
        const unsigned j = g % 8U;
        const uint4 hd = chunk_bytes(p);
        sc = int(j < 4U ? chunk_byte(hd, 4U + j) & 63U
                        : (chunk_byte(hd, 8U + j) & 15U) | ((chunk_byte(hd, j) >> 6U) << 4U));
        m = int(j < 4U ? chunk_byte(hd, 8U + j) & 63U
                       : (chunk_byte(hd, 8U + j) >> 4U) | ((chunk_byte(hd, 4U + j) >> 6U) << 4U));
        d = half_scale(std::uint16_t(hd.x), hardware);
        dmin = half_scale(std::uint16_t(hd.x >> 16U), hardware);
        const uint4 a = chunk_bytes(p + 16U + (j / 2U) * 32U + 16U * h);
        const unsigned shift = (j & 1U) * 4U;
        const unsigned words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (unsigned k = 0; k < 4; ++k)
            w[k] = int((words[k] >> shift) & 0x0f0f0f0fU);
    }
    __device__ float term(const int (&a)[4], Meta meta) const {
        int part = 0;
#pragma unroll
        for (unsigned k = 0; k < 4; ++k)
            part = dot4(w[k], a[k], part);
        return a8::q4_k_term(d, dmin, sc, m, part + __shfl_xor(part, 1, 32), meta);
    }
};
template <> struct HalfGroup<WeightFormat::q6_k> {
    int w[4];
    float d;
    int s;
    __device__ HalfGroup() = default;
    __device__ HalfGroup(const std::uint8_t* row, unsigned g, unsigned h, bool hardware) {
        const auto* p = row + std::size_t(g / 8U) * 210U;
        const unsigned j = g % 8U, half = j / 4U, quarter = j % 4U;
        const uint4 l0 = chunk_bytes(p + half * 64U + (quarter & 1U) * 32U + 16U * h),
                    h0 = chunk_bytes(p + 128U + half * 32U + 16U * h);
        const unsigned lw[4] = {l0.x, l0.y, l0.z, l0.w}, hw[4] = {h0.x, h0.y, h0.z, h0.w};
        const unsigned ls = quarter >= 2U ? 4U : 0U, hs = 2U * quarter;
#pragma unroll
        for (unsigned k = 0; k < 4; ++k) {
            const unsigned q = ((lw[k] >> ls) & 0x0f0f0f0fU) | (((hw[k] >> hs) & 0x03030303U) << 4U);
            w[k] = int(((q | 0x80808080U) - 0x20202020U) ^ 0x80808080U);
        }
        s = int(std::int8_t(p[192U + 2U * j + h]));
        d = half_scale(load16(p + 208), hardware);
    }
    __device__ float term(const int (&a)[4], Meta meta) const {
        int part = 0;
#pragma unroll
        for (unsigned k = 0; k < 4; ++k)
            part = dot4(w[k], a[k], part);
        part = s * part;
        return a8::scaled_term(d, part + __shfl_xor(part, 1, 32), meta);
    }
};
__device__ inline void load_codes(const A8Rows& x, unsigned r, unsigned g, int (&a)[8], Meta& meta) {
    const auto* p = reinterpret_cast<const uint4*>(x.codes + std::size_t(r) * x.width + std::size_t(g) * 32U);
    const uint4 c0 = p[0], c1 = p[1];
    a[0] = int(c0.x), a[1] = int(c0.y), a[2] = int(c0.z), a[3] = int(c0.w);
    a[4] = int(c1.x), a[5] = int(c1.y), a[6] = int(c1.z), a[7] = int(c1.w);
    const float2 m = reinterpret_cast<const float2*>(x.meta)[std::size_t(r) * (x.width / 32U) + g];
    meta.delta = m.x;
    meta.sum = __builtin_bit_cast(int, m.y);
}
__device__ float store_value(float* out, unsigned row, unsigned o, unsigned outputs, float value,
                             const LinearEpilogue& ep, unsigned* failure) {
    if (ep.product) {
        finite_flag(value, failure);
        value = gelu_tanh(value) * ep.product[std::size_t(row) * ep.product_stride + o];
    } else if (ep.cap) {
        finite_flag(value, failure);
        value = softcap(value);
    }
    out[std::size_t(row) * outputs + o] = value;
    finite_flag(value, failure);
    return value;
}
// Row-split form of the declared slot tree (STRIX_G4_A8_SPLIT_TREE, default on; same bytes). Lane slot l
// holds v[l] of every row. At distance d, lanes l and l ^ d still share P rows: the lower lane keeps the first
// half and the upper lane the second; each adds its partner's copy, so every kept row gets v[l] + v[l + d]
// (FP32 addition is commutative). Once one row is left per lane the remaining distances are plain exchanges.
// Returns the row whose total va[0] (vb[0]) now holds; lanes with slot % (16 / P) == 0 hold distinct rows.
__host__ __device__ constexpr unsigned split_rows(unsigned rows) {
    return rows <= 1U ? 1U : rows <= 2U ? 2U : rows <= 4U ? 4U : 8U;
}
template <unsigned Rows, bool Gate>
__device__ inline unsigned split_tree(float (&va)[split_rows(Rows)], float (&vb)[split_rows(Rows)], unsigned slot) {
    unsigned row = 0;
#pragma unroll
    for (unsigned d = 8, count = split_rows(Rows); d; d >>= 1U) {
        if (count > 1U) {
            const unsigned half = count / 2U;
            const bool upper = slot & d;
#pragma unroll
            for (unsigned i = 0; i < half; ++i) {
                const float give = upper ? va[i] : va[i + half], keep = upper ? va[i + half] : va[i];
                va[i] = keep + __shfl_xor(give, int(d), 16);
                if constexpr (Gate) {
                    const float gb = upper ? vb[i] : vb[i + half], kb = upper ? vb[i + half] : vb[i];
                    vb[i] = kb + __shfl_xor(gb, int(d), 16);
                }
            }
            row += upper ? half : 0U;
            count = half;
        } else {
            va[0] = va[0] + __shfl_xor(va[0], int(d), 16);
            if constexpr (Gate)
                vb[0] = vb[0] + __shfl_xor(vb[0], d, 16);
        }
    }
    return row;
}
struct FfnPrologue {
    const float* branch = nullptr;
    LinearView norm;
    float* residual = nullptr;
    unsigned* ticket = nullptr;
    bool prefetch = true, bounded = true, intervals = false, seeded = true;
};
// Each block consumes the old residual into private LDS before taking a ticket. Only the last
// consumer publishes global residual/A8 bytes; all dots read private codes (no inter-block races).
template <unsigned Rows, unsigned Threads>
__device__ void ffn_prologue(const FfnPrologue& p, A8Rows published, A8Rows local, float (&x)[Rows][2560],
                             unsigned* failure) {
    constexpr unsigned J = 2560U / Threads;
    __shared__ float factor;
    for (unsigned r = 0; r < Rows; ++r) {
        float b[J], h[J], g[J];
        if (threadIdx.x < Threads) {
#pragma unroll
            for (unsigned j = 0; j < J; ++j) {
                const unsigned i = threadIdx.x + Threads * j;
                b[j] = p.branch[std::size_t(r) * 2560U + i];
                h[j] = p.residual[std::size_t(r) * 2560U + i];
                g[j] = p.norm.format == WeightFormat::f32
                           ? reinterpret_cast<const float*>(p.norm.data)[i]
                           : bf16_value(reinterpret_cast<const std::uint16_t*>(p.norm.data)[i]);
                x[r][i] = b[j];
            }
        }
        __syncthreads();
        if (threadIdx.x < 32U) {
            const float f = epilogue_factor<true>(x[r], 2560U, p.prefetch, p.bounded, p.intervals, p.seeded);
            if (threadIdx.x == 0)
                factor = f;
        }
        __syncthreads();
        if (threadIdx.x < Threads) {
#pragma unroll
            for (unsigned j = 0; j < J; ++j) {
                const unsigned i = threadIdx.x + Threads * j;
                const float normalized = (b[j] * factor) * g[j];
                finite_flag(normalized, failure);
                const float value = h[j] + normalized;
                finite_flag(value, failure);
                x[r][i] = value;
                a8_quantize_lane(value, local, r, i);
            }
        }
        __syncthreads();
    }
    __shared__ bool last;
    if (threadIdx.x == 0)
        last = atomicAdd(p.ticket, 1U) == gridDim.x - 1U;
    __syncthreads();
    if (last) {
        for (unsigned r = 0; r < Rows; ++r) {
            for (unsigned i = threadIdx.x; i < 2560U; i += blockDim.x) {
                p.residual[std::size_t(r) * 2560U + i] = x[r][i];
                published.codes[std::size_t(r) * 2560U + i] = local.codes[std::size_t(r) * 2560U + i];
            }
            for (unsigned g = threadIdx.x; g < 80U; g += blockDim.x)
                published.meta[std::size_t(r) * 80U + g] = local.meta[std::size_t(r) * 80U + g];
        }
        // The next launch/replay is stream ordered after every block, including these stores.
        if (threadIdx.x == 0)
            *p.ticket = 0U;
    }
}
// Waves of four per block, two outputs per wave: lanes 0..15 own the first output, 16..31 the second.
__device__ inline unsigned select_view(const LinearEpilogue& ep, LinearView& w, float*& out) {
    unsigned block = blockIdx.x;
    if (ep.extra_out[0] && block >= ep.extra_block[0]) {
        const bool second = ep.extra_out[1] && block >= ep.extra_block[1];
        w = second ? ep.extra[1] : ep.extra[0];
        out = second ? ep.extra_out[1] : ep.extra_out[0];
        block -= second ? ep.extra_block[1] : ep.extra_block[0];
    }
    return block;
}
// Quant: sixteen waves cover one 32-output group whose stored values are also quantized (A8G32) by
// the block; the grid is exact (outputs a multiple of 32), so no wave leaves before the barrier.
// Mode bit 1 (STRIX_G4_A8_PIPE): the next visit's weight groups are loaded before the current one is
// accumulated (as pipelined_linear_kernel). Mode bit 2 (STRIX_G4_A8_STREAM): weights use nontemporal loads so
// they do not displace the reused activation codes. Neither changes a visit or an operation.
template <unsigned Rows, bool Gate, WeightFormat Format, bool Quant, unsigned Mode, bool Ffn = false>
__global__ void __launch_bounds__(Quant ? 512 : 128) a8_linear_kernel(LinearView w, LinearView up, A8Rows x, float* out,
                                                                      unsigned* failure, LinearEpilogue ep, A8Rows quant,
                                                                      FfnPrologue prologue) {
    if constexpr (Ffn) {
        __shared__ float residual_rows[Rows][2560];
        __shared__ __align__(16) std::int8_t codes[Rows * 2560U];
        __shared__ __align__(16) Meta meta[Rows * 80U];
        const A8Rows local{codes, meta, 2560U};
        ffn_prologue<Rows, Quant ? 256U : 128U>(prologue, x, local, residual_rows, failure);
        x = local;
    }
    constexpr bool pipe = Mode & 1U, stream = Mode & 2U;
    using G = Group<Format, stream>;
    constexpr unsigned waves = Quant ? 16U : 4U;
    const unsigned lane = threadIdx.x % 32U, slot = lane % 16U;
    const unsigned o = ((Quant ? blockIdx.x : select_view(ep, w, out)) * waves + threadIdx.x / 32U) * 2U + lane / 16U;
    if (!Quant && o - (lane / 16U) >= w.outputs)
        return; // whole wave beyond the matrix
    // A tail half-wave reuses a valid row for its loads; it never stores.
    const unsigned row = min(o, w.outputs - 1U), groups = w.inputs / 32U;
    const auto* wp = w.data + std::size_t(row) * w.row_bytes;
    const auto* uptr = Gate ? up.data + std::size_t(row) * up.row_bytes : wp;
    float acc[Rows] = {}, uacc[Rows] = {};
    __shared__ float stored[Quant ? 8 : 1][32];
    G a, b;
    if (pipe && slot < groups) {
        a = G(wp, slot, ep.hardware_half);
        if constexpr (Gate)
            b = G(uptr, slot, ep.hardware_half);
    }
    for (unsigned g = slot; g < groups; g += 16U) {
        G na, nb;
        if constexpr (pipe) {
            if (g + 16U < groups) {
                na = G(wp, g + 16U, ep.hardware_half);
                if constexpr (Gate)
                    nb = G(uptr, g + 16U, ep.hardware_half);
            }
        } else {
            a = G(wp, g, ep.hardware_half);
            if constexpr (Gate)
                b = G(uptr, g, ep.hardware_half);
        }
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            int codes[8];
            Meta meta;
            load_codes(x, r, g, codes, meta);
            acc[r] = acc[r] + a.term(codes, meta);
            if constexpr (Gate)
                uacc[r] = uacc[r] + b.term(codes, meta);
        }
        if constexpr (pipe) {
            a = na;
            if constexpr (Gate)
                b = nb;
        }
    }
    if (ep.split_tree) {
        float va[split_rows(Rows)] = {}, vb[split_rows(Rows)] = {};
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r)
            va[r] = acc[r], vb[r] = uacc[r];
        const unsigned r = split_tree<Rows, Gate>(va, vb, slot);
        if (r < Rows && slot % (16U / split_rows(Rows)) == 0 && o < w.outputs) {
            const float value = store_value(out, r, o, w.outputs, Gate ? gelu_tanh(va[0]) * vb[0] : va[0], ep, failure);
            if constexpr (Quant)
                stored[r][o % 32U] = value;
        }
    } else {
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            // Declared tree over the 16 slots: v[l] + v[l + d], d = 8, 4, 2, 1.
            float sa = acc[r], sb = uacc[r];
            for (unsigned d = 8; d; d >>= 1U) {
                sa = sa + __shfl_down(sa, d, 16);
                if constexpr (Gate)
                    sb = sb + __shfl_down(sb, d, 16);
            }
            if (slot == 0 && o < w.outputs) {
                const float value = store_value(out, r, o, w.outputs, Gate ? gelu_tanh(sa) * sb : sa, ep, failure);
                if constexpr (Quant)
                    stored[r][o % 32U] = value; // the stored value is what the next linear quantizes
            }
        }
    }
    if constexpr (Quant) {
        __syncthreads();
        const unsigned wave = threadIdx.x / 32U;
        if (wave < Rows)
            a8_quantize_lane(stored[wave][lane], quant, wave, blockIdx.x * 32U + lane);
    }
}
template <unsigned Rows, bool Gate, WeightFormat Format, bool Quant>
__global__ void __launch_bounds__(Quant ? 1024 : 128) a8_half_kernel(LinearView w, LinearView up, A8Rows x, float* out,
                                                                     unsigned* failure, LinearEpilogue ep, A8Rows quant) {
    constexpr unsigned waves = Quant ? 32U : 4U;
    const unsigned lane = threadIdx.x % 32U, slot = lane / 2U, h = lane % 2U;
    const unsigned o = (Quant ? blockIdx.x : select_view(ep, w, out)) * waves + threadIdx.x / 32U;
    if (!Quant && o >= w.outputs)
        return;
    const unsigned groups = w.inputs / 32U;
    const auto* wp = w.data + std::size_t(o) * w.row_bytes;
    const auto* uptr = Gate ? up.data + std::size_t(o) * up.row_bytes : wp;
    float acc[Rows] = {}, uacc[Rows] = {};
    __shared__ float stored[Quant ? 8 : 1][32];
    for (unsigned g = slot; g < groups; g += 16U) {
        const HalfGroup<Format> a(wp, g, h, ep.hardware_half);
        HalfGroup<Format> b;
        if constexpr (Gate)
            b = HalfGroup<Format>(uptr, g, h, ep.hardware_half);
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const uint4 c = *reinterpret_cast<const uint4*>(x.codes + std::size_t(r) * x.width + std::size_t(g) * 32U +
                                                           16U * h);
            const int codes[4] = {int(c.x), int(c.y), int(c.z), int(c.w)};
            const float2 mv = reinterpret_cast<const float2*>(x.meta)[std::size_t(r) * groups + g];
            const Meta meta{mv.x, __builtin_bit_cast(int, mv.y)};
            acc[r] = acc[r] + a.term(codes, meta);
            if constexpr (Gate)
                uacc[r] = uacc[r] + b.term(codes, meta);
        }
    }
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
        // Lanes 2p and 2p+1 hold slot p: distances 16, 8, 4, 2 are the declared 8, 4, 2, 1 slot tree.
        float sa = acc[r], sb = uacc[r];
        for (unsigned d = 16; d >= 2U; d >>= 1U) {
            sa = sa + __shfl_down(sa, d, 32);
            if constexpr (Gate)
                sb = sb + __shfl_down(sb, d, 32);
        }
        if (lane == 0) {
            const float value = store_value(out, r, o, w.outputs, Gate ? gelu_tanh(sa) * sb : sa, ep, failure);
            if constexpr (Quant)
                stored[r][o % 32U] = value;
        }
    }
    if constexpr (Quant) {
        __syncthreads();
        const unsigned wave = threadIdx.x / 32U;
        if (wave < Rows)
            a8_quantize_lane(stored[wave][lane], quant, wave, blockIdx.x * 32U + lane);
    }
}
// One wave per 32-element group of the row-major [rows, width] input.
__global__ void __launch_bounds__(256) a8_quantize_kernel(const float* x, unsigned groups, unsigned width, A8Rows q) {
    const unsigned g = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    if (g >= groups)
        return;
    const unsigned row = g / (width / 32U), i = (g % (width / 32U)) * 32U + lane;
    a8_quantize_lane(x[std::size_t(row) * width + i], q, row, i);
}
bool split_tree_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_A8_SPLIT_TREE");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool hardware_half() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_HARDWARE_HALF");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
// Opt-in half-group schedule (same bytes): one output per wave instead of two.
bool half_groups() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_A8_HALF");
        return p && std::string_view(p) == "1";
    }();
    return enabled;
}
// Pair-schedule mode bits (same bytes), both opt-in: 1 = pipelined weight loads (STRIX_G4_A8_PIPE=1),
// 2 = nontemporal weight loads (STRIX_G4_A8_STREAM=1). Measured: pipe C1 -2 %, C8 -2.4 %; stream C1 -13 %.
unsigned pair_mode() {
    static const unsigned mode = [] {
        const auto* p = std::getenv("STRIX_G4_A8_PIPE");
        const auto* s = std::getenv("STRIX_G4_A8_STREAM");
        return (p && std::string_view(p) == "1" ? 1U : 0U) | (s && std::string_view(s) == "1" ? 2U : 0U);
    }();
    return mode;
}
unsigned blocks_for(unsigned outputs) {
    const unsigned per_block = half_groups() ? 4U : 8U;
    return (outputs + per_block - 1U) / per_block;
}
template <unsigned R, bool Gate, WeightFormat Format, bool Quant>
void pair(LinearView w, LinearView up, A8Rows x, float* out, unsigned* failure, const LinearEpilogue& ep, unsigned blocks,
          Queue q, A8Rows quant) {
    const auto stream = static_cast<hipStream_t>(q);
    const dim3 threads(Quant ? 512 : 128);
    const auto run = [&](auto kernel) {
        hipLaunchKernelGGL(kernel, dim3(blocks), threads, 0, stream, w, up, x, out, failure, ep, quant, FfnPrologue{});
    };
    switch (pair_mode()) {
    case 0:
        run(a8_linear_kernel<R, Gate, Format, Quant, 0>);
        break;
    case 1:
        run(a8_linear_kernel<R, Gate, Format, Quant, 1>);
        break;
    case 2:
        run(a8_linear_kernel<R, Gate, Format, Quant, 2>);
        break;
    default:
        run(a8_linear_kernel<R, Gate, Format, Quant, 3>);
        break;
    }
}
template <bool Gate, WeightFormat Format, bool Quant>
void launch(LinearView w, LinearView up, A8Rows x, unsigned rows, float* out, unsigned* failure, const LinearEpilogue& ep,
            unsigned blocks, Queue q, A8Rows quant) {
#define G4_A8_LAUNCH(R)                                                                                                    \
    case R:                                                                                                                \
        if (half_groups())                                                                                                 \
            hipLaunchKernelGGL((a8_half_kernel<R, Gate, Format, Quant>), dim3(blocks), dim3(Quant ? 1024 : 128), 0,       \
                               static_cast<hipStream_t>(q), w, up, x, out, failure, ep, quant);                            \
        else                                                                                                               \
            pair<R, Gate, Format, Quant>(w, up, x, out, failure, ep, blocks, q, quant);                                    \
        break
    switch (rows) {
        G4_A8_LAUNCH(1);
        G4_A8_LAUNCH(2);
        G4_A8_LAUNCH(3);
        G4_A8_LAUNCH(4);
        G4_A8_LAUNCH(5);
        G4_A8_LAUNCH(6);
        G4_A8_LAUNCH(7);
        G4_A8_LAUNCH(8);
    }
#undef G4_A8_LAUNCH
    launch_checked();
}
template <bool Gate, bool Quant>
bool dispatch_format(LinearView w, LinearView up, A8Rows x, unsigned rows, float* out, unsigned* failure,
                     const LinearEpilogue& ep, unsigned blocks, Queue q, A8Rows quant) {
    switch (w.format) {
    case WeightFormat::q8_0:
        launch<Gate, WeightFormat::q8_0, Quant>(w, up, x, rows, out, failure, ep, blocks, q, quant);
        return true;
    case WeightFormat::q4_k:
        launch<Gate, WeightFormat::q4_k, Quant>(w, up, x, rows, out, failure, ep, blocks, q, quant);
        return true;
    case WeightFormat::q6_k:
        launch<Gate, WeightFormat::q6_k, Quant>(w, up, x, rows, out, failure, ep, blocks, q, quant);
        return true;
    default:
        return false;
    }
}
template <bool Gate>
bool dispatch(LinearView w, LinearView up, A8Rows x, unsigned rows, float* out, unsigned* failure, const LinearEpilogue& ep,
              unsigned blocks, Queue q, A8Rows quant) {
    if (!quant.codes)
        return dispatch_format<Gate, false>(w, up, x, rows, out, failure, ep, blocks, q, quant);
    if (w.outputs % 32U || quant.width != w.outputs || ep.extra_out[0] || !quant.meta)
        throw std::runtime_error("g4-a8-v1 quantized-output geometry");
    return dispatch_format<Gate, true>(w, up, x, rows, out, failure, ep, w.outputs / 32U, q, quant);
}
template <WeightFormat Format, bool Quant>
void launch_ffn(LinearView gate, A8Rows input, unsigned rows, float* out, unsigned* failure,
                LinearEpilogue ep, Queue q, A8Rows quant, const FfnPrologue& p) {
#define G4_A8_FFN(R)                                                                                                    \
    case R:                                                                                                             \
        hipLaunchKernelGGL((a8_linear_kernel<R, false, Format, Quant, 0, true>),                                           \
                           dim3(Quant ? gate.outputs / 32U : (gate.outputs + 7U) / 8U),                                  \
                           dim3(Quant ? 512 : 128), 0, static_cast<hipStream_t>(q), gate, LinearView{}, input, out,      \
                           failure, ep, quant, p);                                                                      \
        break
    switch (rows) {
        G4_A8_FFN(1);
        G4_A8_FFN(2);
        G4_A8_FFN(3);
        G4_A8_FFN(4);
    }
#undef G4_A8_FFN
}
template <bool Quant>
bool dispatch_ffn(LinearView gate, A8Rows input, unsigned rows, float* out, unsigned* failure,
                  LinearEpilogue ep, Queue q, A8Rows quant, const FfnPrologue& p) {
    switch (gate.format) {
    case WeightFormat::q8_0:
        launch_ffn<WeightFormat::q8_0, Quant>(gate, input, rows, out, failure, ep, q, quant, p);
        break;
    case WeightFormat::q4_k:
        launch_ffn<WeightFormat::q4_k, Quant>(gate, input, rows, out, failure, ep, q, quant, p);
        break;
    case WeightFormat::q6_k:
        launch_ffn<WeightFormat::q6_k, Quant>(gate, input, rows, out, failure, ep, q, quant, p);
        break;
    default:
        return false;
    }
    launch_checked();
    return true;
}
void check_input(LinearView w, const A8Rows& x, unsigned rows, const unsigned* failure) {
    valid_rows(rows);
    if (!w.data || !w.outputs || !w.inputs || w.inputs % 256U || x.width != w.inputs || !x.codes || !x.meta || !failure)
        throw std::runtime_error("g4-a8-v1 linear geometry");
}
} // namespace
bool a8_ffn_ple_gate(const float* branch, LinearView norm, float* residual, LinearView gate,
                      const float* ple, unsigned product_stride, unsigned rows, float* out, A8Rows input,
                      unsigned* failure, unsigned* ticket, Queue q, A8Rows quant) {
    const auto on = [](const char* name) {
        const auto* v = std::getenv(name);
        return !v || std::string_view(v) != "0";
    };
    const auto* early = std::getenv("STRIX_G4_EPILOGUE_EARLY_LOADS");
    const auto overlap = [](const void* a, std::size_t na, const void* b, std::size_t nb) {
        const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
        return x < y + nb && y < x + na;
    };
    const auto bytes = std::size_t(rows) * 2560U * sizeof(float);
    if (!rows || rows > 4U || gate.inputs != 2560U || !gate.outputs || !gate.data || !branch || !residual || !out ||
        !ple || product_stride < gate.outputs || !failure || !ticket || !norm.data || norm.outputs != 1U ||
        norm.inputs != 2560U || (norm.format != WeightFormat::f32 && norm.format != WeightFormat::bf16) ||
        reinterpret_cast<std::uintptr_t>(norm.data) % (norm.format == WeightFormat::f32 ? 4U : 2U) ||
        !input.codes || !input.meta || input.width != 2560U || half_groups() || pair_mode() != 0U ||
        !on("STRIX_G4_FAST_EPILOGUE") || !on("STRIX_G4_NORM_STAGE") || !on("STRIX_G4_NORM_SEARCH") ||
        (early && std::string_view(early) == "1") || overlap(branch, bytes, residual, bytes) ||
        overlap(out, std::size_t(rows) * gate.outputs * 4U, branch, bytes) ||
        overlap(out, std::size_t(rows) * gate.outputs * 4U, residual, bytes) ||
        (quant.codes && (!quant.meta || quant.width != gate.outputs || gate.outputs % 32U ||
                         overlap(input.codes, std::size_t(rows) * input.width, quant.codes,
                                 std::size_t(rows) * quant.width) ||
                         overlap(input.meta, std::size_t(rows) * input.width / 32U * sizeof(Meta), quant.meta,
                                 std::size_t(rows) * quant.width / 32U * sizeof(Meta)))))
        return false;
    LinearEpilogue ep;
    ep.product = ple;
    ep.product_stride = product_stride;
    ep.hardware_half = hardware_half();
    ep.split_tree = split_tree_enabled();
    const auto* intervals = std::getenv("STRIX_G4_NORM_INTERVALS");
    const FfnPrologue p{branch, norm, residual, ticket, on("STRIX_G4_NORM_PREFETCH"), on("STRIX_G4_NORM_BOUNDS"),
                         intervals && std::string_view(intervals) == "1", on("STRIX_G4_SEEDED_RSQRT")};
    return quant.codes ? dispatch_ffn<true>(gate, input, rows, out, failure, ep, q, quant, p)
                       : dispatch_ffn<false>(gate, input, rows, out, failure, ep, q, quant, p);
}
void a8_quantize(const float* x, unsigned rows, unsigned width, A8Rows out, Queue q) {
    valid_rows(rows);
    if (!width || width % 256U || out.width != width || !out.codes || !out.meta)
        throw std::runtime_error("g4-a8-v1 quantization geometry");
    const unsigned groups = rows * (width / 32U);
    hipLaunchKernelGGL(a8_quantize_kernel, dim3((groups + 7U) / 8U), dim3(256), 0, static_cast<hipStream_t>(q), x, groups,
                       width, out);
    launch_checked();
}
bool a8_linear(LinearView w, A8Rows x, unsigned rows, float* out, const LinearEpilogue& epilogue, Queue q,
               unsigned* failure, A8Rows quant) {
    if (!a8::quantized(w.format))
        return false;
    check_input(w, x, rows, failure);
    auto ep = epilogue;
    ep.hardware_half = hardware_half();
    ep.split_tree = split_tree_enabled();
    unsigned blocks = blocks_for(w.outputs);
    if (ep.extra_out[0]) {
        if (ep.product || ep.cap)
            return false;
        for (unsigned t = 0; t < 2; ++t) {
            if (!ep.extra_out[t])
                break;
            const auto v = ep.extra[t];
            if (!v.data || v.format != w.format || v.inputs != w.inputs || !v.outputs)
                return false;
            ep.extra_block[t] = blocks;
            blocks += blocks_for(v.outputs);
        }
        ep.blocks = blocks;
    }
    return dispatch<false>(w, {}, x, rows, out, failure, ep, blocks, q, quant);
}
bool a8_gate_up(LinearView gate, LinearView up, A8Rows x, unsigned rows, float* out, Queue q, unsigned* failure,
                A8Rows quant) {
    if (!a8::quantized(gate.format) || gate.format != up.format)
        return false;
    check_input(gate, x, rows, failure);
    if (!up.data || up.outputs != gate.outputs || up.inputs != gate.inputs)
        throw std::runtime_error("g4-a8-v1 gate/up geometry");
    LinearEpilogue ep;
    ep.hardware_half = hardware_half();
    ep.split_tree = split_tree_enabled();
    return dispatch<true>(gate, up, x, rows, out, failure, ep, blocks_for(gate.outputs), q, quant);
}
} // namespace strix::gemma4::gpu
