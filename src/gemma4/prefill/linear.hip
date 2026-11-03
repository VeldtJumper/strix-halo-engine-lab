#include "device.hpp"
#include "gemma4/exl3/device.hpp"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace strix::gemma4::prefill {
namespace {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
// A WMMA operand as eight dwords: two 16-byte integer loads, reinterpreted as
// Half16 only at the instruction, so the compiler keeps the loaded registers
// instead of rebuilding the 16-element half vector (same bits).
using Raw16 = unsigned __attribute__((ext_vector_type(8)));
using Raw8 = unsigned __attribute__((ext_vector_type(4)));
__device__ inline Raw16 raw16(const _Float16* p) {
    const auto* q = reinterpret_cast<const Raw8*>(p);
    const Raw8 a = q[0], b = q[1];
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7);
}
// Operand through a raw buffer resource: the workgroup-uniform per-K16 byte
// offset in the scalar offset, the lane's byte offset in one VGPR (no 64-bit
// address per load). Same bytes as raw16() at base + element.
__device__ inline __amdgpu_buffer_rsrc_t panel_rsrc(const _Float16* base) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<_Float16*>(base), 0, 0x7fffffff, 0x31004000);
}
__device__ inline Raw8 raw8_buf(__amdgpu_buffer_rsrc_t rsrc, unsigned lane_element, unsigned uniform_bytes) {
    return __builtin_bit_cast(Raw8, __builtin_amdgcn_raw_buffer_load_b128(rsrc, int(lane_element * 2U), int(uniform_bytes), 0));
}
__device__ inline Raw16 raw16_buf(__amdgpu_buffer_rsrc_t rsrc, unsigned lane_element, unsigned uniform_bytes) {
    const auto v = int(lane_element * 2U), s0 = int(uniform_bytes);
    const Raw8 a = __builtin_bit_cast(Raw8, __builtin_amdgcn_raw_buffer_load_b128(rsrc, v, s0, 0));
    const Raw8 b = __builtin_bit_cast(Raw8, __builtin_amdgcn_raw_buffer_load_b128(rsrc, v + 16, s0, 0));
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7);
}
// Global operand at element offset `element` from a workgroup-uniform base,
// through a buffer resource (32-bit offset VGPR, no 64-bit address).
__device__ inline Raw16 raw16_at(const _Float16* base, unsigned element) { return raw16_buf(panel_rsrc(base), element, 0); }
__device__ inline Float8 wmma(Raw16 a, Raw16 b, Float8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(Half16, a), __builtin_bit_cast(Half16, b), c);
}
void checked() {
    if (const auto e = hipGetLastError(); e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
__device__ _Float16 rounded(float x, unsigned* failure) {
    // Native conversion is RNE under the target's fixed mode; fixture-tested
    // against half_bits, including half subnormals and the overflow midpoint.
    const auto h = static_cast<_Float16>(x);
    if ((__builtin_bit_cast(unsigned short, h) & 0x7c00U) == 0x7c00U) atomicOr(failure, 4U);
    return h;
}
template <bool Panels>
__global__ void convert(const float* x, _Float16* out, unsigned count, unsigned width, unsigned stride,
                        unsigned* failure, unsigned* maximum) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned bits = 0;
    if (i < std::size_t(count) * stride) {
        const unsigned r = unsigned(i / stride), c = unsigned(i % stride);
        const float a = c < width ? x[std::size_t(r) * width + c] : 0.0F;
        bits = __builtin_bit_cast(unsigned, a) & 0x7fffffffU;
        // Panels [K16][row][K] let adjacent matrix lanes load adjacent rows.
        const auto index = Panels ? std::size_t(c / 16) * count * 16 + r * 16 + c % 16 : i;
        out[index] = rounded(a, failure);
    }
    for (unsigned d = 16; d; d >>= 1U) bits = max(bits, __shfl_down(bits, d, 32));
    __shared__ unsigned maxima[8];
    if (threadIdx.x % 32U == 0) maxima[threadIdx.x / 32U] = bits;
    __syncthreads();
    if (!threadIdx.x) {
        unsigned result = 0;
        for (unsigned j = 0; j < 8; ++j) result = max(result, maxima[j]);
        atomicMax(maximum, result);
    }
}
// g4-wmma-pf-v2 activation split (see emulate_linear_v2): high panels at out,
// low panels at out + count * stride, both in the [K16][row][16] panel layout.
__global__ void convert_split(const float* x, _Float16* out, unsigned count, unsigned width, unsigned stride,
                              unsigned* failure, unsigned* maximum) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned bits = 0;
    if (i < std::size_t(count) * stride) {
        const unsigned r = unsigned(i / stride), c = unsigned(i % stride);
        const float a = c < width ? x[std::size_t(r) * width + c] : 0.0F;
        bits = __builtin_bit_cast(unsigned, a) & 0x7fffffffU;
        const auto index = std::size_t(c / 16) * count * 16 + r * 16 + c % 16;
        const auto high = rounded(a, failure);
        out[index] = high;
        out[std::size_t(count) * stride + index] = rounded((a - float(high)) * v2_low_scale, failure);
    }
    for (unsigned d = 16; d; d >>= 1U) bits = max(bits, __shfl_down(bits, d, 32));
    __shared__ unsigned maxima[8];
    if (threadIdx.x % 32U == 0) maxima[threadIdx.x / 32U] = bits;
    __syncthreads();
    if (!threadIdx.x) {
        unsigned result = 0;
        for (unsigned j = 0; j < 8; ++j) result = max(result, maxima[j]);
        atomicMax(maximum, result);
    }
}
// One block's high/low accumulators into the output state: part = h + l*2^-12,
// then (blocked) s = s + part over the blocks in increasing K. Each operation is
// a separately rounded FP32 op.
template <bool Blocked>
__device__ inline void v2_merge(float h, float l, float& s) {
    float scaled = l * v2_low_unscale;
    asm volatile("" : "+v"(scaled));
    const float part = h + scaled;
    s = Blocked ? s + part : part;
}
// Corrected linear over prepared FP16 weight panels: four waves own TM x TN
// output tiles of 16x16 each (2x2 waves); every output keeps independent high
// and low accumulators over increasing K16 and, when blocked, restarts them at
// +0 every v2_block inputs.
template <bool Blocked, unsigned TM, unsigned TN, bool Guard>
__global__ __launch_bounds__(128) void hilo_gemm(const _Float16* weights, const _Float16* x, unsigned rows, unsigned outputs,
                                                unsigned stride, float* out, unsigned* failure) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m0 = blockIdx.y * (32 * TM) + (wave / 2) * (16 * TM), n0 = blockIdx.x * (32 * TN) + (wave % 2) * (16 * TN);
    const auto* xl = x + std::size_t(rows) * stride;
    Float8 h[TM][TN]{}, l[TM][TN]{};
    float s[TM][TN][8]{};
    // Workgroup-uniform bases advance per K16; lane offsets stay 32-bit.
    unsigned wo[TN];
#pragma unroll
    for (unsigned j = 0; j < TN; ++j) {
        const auto n = n0 + j * 16 + lane % 16;
        wo[j] = (n / 16) * (stride / 16) * 256 + (n % 16) * 16;
    }
    const auto rx = panel_rsrc(x), rxl = panel_rsrc(xl), rw = panel_rsrc(weights);
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        const unsigned xs = k0 * rows * 2U, ws = (k0 / 16) * 512U;
        Raw16 ah[TM], al[TM], b[TN];
#pragma unroll
        for (unsigned i = 0; i < TM; ++i) {
            const auto r = m0 + i * 16 + lane % 16;
            ah[i] = !Guard || r < rows ? raw16_buf(rx, r * 16, xs) : Raw16{};
            al[i] = !Guard || r < rows ? raw16_buf(rxl, r * 16, xs) : Raw16{};
        }
#pragma unroll
        for (unsigned j = 0; j < TN; ++j) {
            const auto n = n0 + j * 16 + lane % 16;
            b[j] = !Guard || n < outputs ? raw16_buf(rw, wo[j], ws) : Raw16{};
        }
#pragma unroll
        for (unsigned i = 0; i < TM; ++i)
#pragma unroll
            for (unsigned j = 0; j < TN; ++j) {
                h[i][j] = wmma(ah[i], b[j], h[i][j]);
                l[i][j] = wmma(al[i], b[j], l[i][j]);
            }
        if (Blocked && ((k0 + 16) % v2_block == 0 || k0 + 16 >= stride)) {
#pragma unroll
            for (unsigned i = 0; i < TM; ++i)
#pragma unroll
                for (unsigned j = 0; j < TN; ++j) {
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e) v2_merge<true>(h[i][j][e], l[i][j][e], s[i][j][e]);
                    h[i][j] = Float8{}; l[i][j] = Float8{};
                }
        }
    }
    unsigned bad = 0;
#pragma unroll
    for (unsigned i = 0; i < TM; ++i)
#pragma unroll
        for (unsigned j = 0; j < TN; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m0 + i * 16 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
                float y;
                if constexpr (Blocked) y = s[i][j][e];
                else { v2_merge<false>(h[i][j][e], l[i][j][e], y); }
                if (r < rows && n < outputs) {
                    bad |= (bit_cast<unsigned>(y) & 0x7f800000U) == 0x7f800000U;
                    out[std::size_t(r) * outputs + n] = y;
                }
            }
    if (bad) atomicOr(failure, 1U);
}
// Unblocked v2 (hilo) in the register-only style of direct_gemm: eight waves,
// each owning 64 rows x 32 outputs with high and low accumulators (16 WMMAs
// per 10 operand loads per K16), 128 rows x 128 outputs per workgroup.
// Guard = false: full tiles only (rows % 128 == 0, outputs % 128 == 0), no
// per-lane load predicates.
template <bool Guard>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8))) void hilo_direct(const _Float16* weights, const _Float16* x, unsigned rows, unsigned outputs,
                                                  unsigned stride, float* out, unsigned* failure) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m0 = blockIdx.y * 128 + (wave / 4) * 64, n0 = blockIdx.x * 128 + (wave % 4) * 32;
    const auto* xl = x + std::size_t(rows) * stride;
    Float8 h[4][2]{}, l[4][2]{};
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        Raw16 b[2];
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const auto n = n0 + j * 16 + lane % 16;
            b[j] = !Guard || n < outputs ? raw16_at(weights, unsigned((std::size_t(n / 16) * (stride / 16) + k0 / 16) * 256 + (n % 16) * 16)) : Raw16{};
        }
        // One row tile's operands live at a time: 192 VGPRs (occupancy 8) without spills.
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const auto r = m0 + i * 16 + lane % 16;
            const Raw16 ah = !Guard || r < rows ? raw16_at(x, unsigned(std::size_t(k0) * rows + r * 16)) : Raw16{};
            const Raw16 al = !Guard || r < rows ? raw16_at(xl, unsigned(std::size_t(k0) * rows + r * 16)) : Raw16{};
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                h[i][j] = wmma(ah, b[j], h[i][j]);
                l[i][j] = wmma(al, b[j], l[i][j]);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    unsigned bad = 0;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 2; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m0 + i * 16 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
                float y;
                v2_merge<false>(h[i][j][e], l[i][j][e], y);
                if (r < rows && n < outputs) {
                    bad |= (bit_cast<unsigned>(y) & 0x7f800000U) == 0x7f800000U;
                    out[std::size_t(r) * outputs + n] = y;
                }
            }
    if (bad) atomicOr(failure, 1U);
}
// Corrected (hilo) gate and up of the same split input in one workgroup: each
// wave owns 32 rows x 32 outputs of both with high/low accumulators, the
// projections finish as two hilo linears would (v2_merge), then the epilogue
// is gelu_pair_gemm's (FP32 product, checks, high and optional low panels).
// 128 rows x 64 outputs per workgroup.
// Guard = false: full tiles only (rows % 128 == 0, outputs % 64 == 0).
template <bool Low, bool Guard>
__global__ __launch_bounds__(256) void gelu_pair_hilo(const _Float16* gate, const _Float16* up, const _Float16* x,
                                                     unsigned rows, unsigned outputs, unsigned stride,
                                                     _Float16* product, unsigned* failure, unsigned* maximum) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m0 = blockIdx.y * 128 + (wave / 2) * 32, n0 = blockIdx.x * 64 + (wave % 2) * 32;
    const auto* xl = x + std::size_t(rows) * stride;
    Float8 gh[2][2]{}, gl[2][2]{}, uh[2][2]{}, ul[2][2]{};
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        Raw16 ah[2], al[2], bg[2], bu[2];
#pragma unroll
        for (unsigned i = 0; i < 2; ++i) {
            const auto r = m0 + i * 16 + lane % 16;
            ah[i] = !Guard || r < rows ? raw16_at(x, unsigned(std::size_t(k0) * rows + r * 16)) : Raw16{};
            al[i] = !Guard || r < rows ? raw16_at(xl, unsigned(std::size_t(k0) * rows + r * 16)) : Raw16{};
        }
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const auto n = n0 + j * 16 + lane % 16;
            const auto offset = (std::size_t(n / 16) * (stride / 16) + k0 / 16) * 256 + (n % 16) * 16;
            bg[j] = !Guard || n < outputs ? raw16_at(gate, unsigned(offset)) : Raw16{};
            bu[j] = !Guard || n < outputs ? raw16_at(up, unsigned(offset)) : Raw16{};
        }
#pragma unroll
        for (unsigned i = 0; i < 2; ++i)
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                gh[i][j] = wmma(ah[i], bg[j], gh[i][j]);
                gl[i][j] = wmma(al[i], bg[j], gl[i][j]);
                uh[i][j] = wmma(ah[i], bu[j], uh[i][j]);
                ul[i][j] = wmma(al[i], bu[j], ul[i][j]);
            }
    }
    unsigned bad = 0, high = 0;
#pragma unroll
    for (unsigned i = 0; i < 2; ++i)
#pragma unroll
        for (unsigned j = 0; j < 2; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m0 + i * 16 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
                if (r < rows && n < outputs) {
                    float g, u;
                    v2_merge<false>(gh[i][j][e], gl[i][j][e], g);
                    v2_merge<false>(uh[i][j][e], ul[i][j][e], u);
                    bad |= (bit_cast<unsigned>(g) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                    bad |= (bit_cast<unsigned>(u) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                    float p = gelu_tanh(g) * u;
                    asm volatile("" : "+v"(p));
                    bad |= (bit_cast<unsigned>(p) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                    high = max(high, bit_cast<unsigned>(p) & 0x7fffffffU);
                    const auto hp = static_cast<_Float16>(p);
                    bad |= (bit_cast<unsigned short>(hp) & 0x7c00U) == 0x7c00U ? 4U : 0U;
                    product[std::size_t(n / 16) * rows * 16 + std::size_t(r) * 16 + n % 16] = hp;
                    if constexpr (Low) {
                        const auto lp = static_cast<_Float16>((p - float(hp)) * v2_low_scale);
                        bad |= (bit_cast<unsigned short>(lp) & 0x7c00U) == 0x7c00U ? 4U : 0U;
                        product[std::size_t(rows) * outputs + std::size_t(n / 16) * rows * 16 + std::size_t(r) * 16 + n % 16] = lp;
                    }
                }
            }
    if (bad) atomicOr(failure, bad);
    for (unsigned d = 16; d; d >>= 1U) high = max(high, __shfl_down(high, d, 32));
    if (lane == 0) atomicMax(maximum, high);
}
// Blocked (hilo_k128) gate and up of the same split input in one workgroup:
// each wave owns 16 rows x 32 outputs of both, the high/low chains restart
// every v2_block inputs and their block parts merge as two blocked hilo
// linears would (v2_merge<true>); the epilogue is
// gelu_pair_hilo's. 128 rows x 32 outputs per workgroup.
template <bool Low, bool Guard>
__global__ __launch_bounds__(256) void gelu_pair_hilo_k128(const _Float16* gate, const _Float16* up, const _Float16* x,
                                                          unsigned rows, unsigned outputs, unsigned stride,
                                                          _Float16* product, unsigned* failure, unsigned* maximum) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m0 = blockIdx.y * 128 + wave * 16, n0 = blockIdx.x * 32;
    const auto* xl = x + std::size_t(rows) * stride;
    Float8 gh[2]{}, gl[2]{}, uh[2]{}, ul[2]{};
    float sg[2][8]{}, su[2][8]{};
    const auto r = m0 + lane % 16;
    // Workgroup-uniform bases advance per K16; lane offsets stay 32-bit.
    const unsigned xo = r * 16;
    unsigned wo[2];
#pragma unroll
    for (unsigned j = 0; j < 2; ++j) {
        const auto n = n0 + j * 16 + lane % 16;
        wo[j] = (n / 16) * (stride / 16) * 256 + (n % 16) * 16;
    }
    const auto rx = panel_rsrc(x), rxl = panel_rsrc(xl), rg = panel_rsrc(gate), ru = panel_rsrc(up);
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        const unsigned xs = k0 * rows * 2U, ws = (k0 / 16) * 512U;
        const Raw16 ah = !Guard || r < rows ? raw16_buf(rx, xo, xs) : Raw16{};
        const Raw16 al = !Guard || r < rows ? raw16_buf(rxl, xo, xs) : Raw16{};
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const auto n = n0 + j * 16 + lane % 16;
            const Raw16 bg = !Guard || n < outputs ? raw16_buf(rg, wo[j], ws) : Raw16{};
            const Raw16 bu = !Guard || n < outputs ? raw16_buf(ru, wo[j], ws) : Raw16{};
            gh[j] = wmma(ah, bg, gh[j]); gl[j] = wmma(al, bg, gl[j]);
            uh[j] = wmma(ah, bu, uh[j]); ul[j] = wmma(al, bu, ul[j]);
        }
        if ((k0 + 16) % v2_block == 0 || k0 + 16 >= stride) {
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) {
                    v2_merge<true>(gh[j][e], gl[j][e], sg[j][e]);
                    v2_merge<true>(uh[j][e], ul[j][e], su[j][e]);
                }
                gh[j] = Float8{}; gl[j] = Float8{}; uh[j] = Float8{}; ul[j] = Float8{};
            }
        }
    }
    unsigned bad = 0, high = 0;
#pragma unroll
    for (unsigned j = 0; j < 2; ++j)
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const auto row = m0 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
            if (row < rows && n < outputs) {
                const float g = sg[j][e], u = su[j][e];
                bad |= (bit_cast<unsigned>(g) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                bad |= (bit_cast<unsigned>(u) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                float p = gelu_tanh(g) * u;
                asm volatile("" : "+v"(p));
                bad |= (bit_cast<unsigned>(p) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                high = max(high, bit_cast<unsigned>(p) & 0x7fffffffU);
                const auto hp = static_cast<_Float16>(p);
                bad |= (bit_cast<unsigned short>(hp) & 0x7c00U) == 0x7c00U ? 4U : 0U;
                product[std::size_t(n / 16) * rows * 16 + std::size_t(row) * 16 + n % 16] = hp;
                if constexpr (Low) {
                    const auto lp = static_cast<_Float16>((p - float(hp)) * v2_low_scale);
                    bad |= (bit_cast<unsigned short>(lp) & 0x7c00U) == 0x7c00U ? 4U : 0U;
                    product[std::size_t(rows) * outputs + std::size_t(n / 16) * rows * 16 + std::size_t(row) * 16 + n % 16] = lp;
                }
            }
        }
    if (bad) atomicOr(failure, bad);
    for (unsigned d = 16; d; d >>= 1U) high = max(high, __shfl_down(high, d, 32));
    if (lane == 0) atomicMax(maximum, high);
}
// LDS-staged v2 linear for long-K projections (down): eight waves own 32 x 32
// outputs each (4 x 2 waves, 128 rows x 64 outputs per workgroup) with high and
// low accumulators and, when blocked, the block sum. High and low input
// panels and the weight panel are staged K32 at a time in double-buffered LDS,
// the next stage prefetched into registers. Same K16 chains, restarts and merge
// order as hilo_gemm / emulate_linear_v2. Requires stride % 32 == 0.
template <bool Blocked>
__global__ __launch_bounds__(256) void staged_hilo(const _Float16* weights, const _Float16* x, unsigned rows, unsigned outputs,
                                                  unsigned stride, float* out, unsigned* failure) {
    constexpr unsigned threads = 256, tile_m = 128, tile_n = 64;
    constexpr unsigned a_loads = 2 * tile_m * 32 / 8 / threads, b_loads = tile_n * 32 / 8 / threads;
    using Vec = unsigned __attribute__((ext_vector_type(4)));
    // [stage][k16 of the stage][high/low][row][16] and [stage][k16][output][16]
    __shared__ _Float16 la[2][2][2][tile_m * 16], lb[2][2][tile_n * 16];
    const unsigned tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const unsigned m_tile = blockIdx.y * tile_m, n_tile = blockIdx.x * tile_n;
    const unsigned wm = (wave / 2) * 32, wn = (wave % 2) * 32;
    const unsigned k16s = stride / 16, n16s = (outputs + 15U) / 16U;
    const auto* xl = x + std::size_t(rows) * stride;
    Vec ra[a_loads], rb[b_loads];
    // Buffer resources: the stage's K16 offset is workgroup-uniform (scalar),
    // each thread keeps one 32-bit element offset per load.
    const auto rx = panel_rsrc(x), rxl = panel_rsrc(xl), rw = panel_rsrc(weights);
    auto fetch = [&](unsigned k0) {
#pragma unroll
        for (unsigned i = 0; i < a_loads; ++i) {
            // [half][k16][row][2 parts]
            const unsigned v = tid + i * threads, which = v / (tile_m * 4), h = (v / (tile_m * 2)) % 2;
            const unsigned row = (v / 2) % tile_m, part = v % 2, r = m_tile + row;
            const unsigned uniform = (k0 / 16 + h) * rows * 32U;
            ra[i] = r < rows ? __builtin_bit_cast(Vec, raw8_buf(which ? rxl : rx, r * 16 + part * 8, uniform)) : Vec{};
        }
#pragma unroll
        for (unsigned i = 0; i < b_loads; ++i) {
            const unsigned v = tid + i * threads, h = v / (tile_n * 2), n = (v / 2) % tile_n, part = v % 2;
            const unsigned j = (n_tile + n) / 16;
            rb[i] = j < n16s ? __builtin_bit_cast(Vec, raw8_buf(rw, j * k16s * 256 + (n % 16) * 16 + part * 8, (k0 / 16 + h) * 512U))
                             : Vec{};
        }
    };
    auto store = [&](unsigned slot) {
#pragma unroll
        for (unsigned i = 0; i < a_loads; ++i) {
            const unsigned v = tid + i * threads, which = v / (tile_m * 4), h = (v / (tile_m * 2)) % 2;
            const unsigned row = (v / 2) % tile_m, part = v % 2;
            *reinterpret_cast<Vec*>(&la[slot][h][which][row * 16 + part * 8]) = ra[i];
        }
#pragma unroll
        for (unsigned i = 0; i < b_loads; ++i) {
            const unsigned v = tid + i * threads, h = v / (tile_n * 2), n = (v / 2) % tile_n, part = v % 2;
            *reinterpret_cast<Vec*>(&lb[slot][h][n * 16 + part * 8]) = rb[i];
        }
    };
    Float8 hi[2][2]{}, lo[2][2]{};
    float s[2][2][8]{};
    fetch(0); store(0);
    __syncthreads();
    for (unsigned k0 = 0; k0 < stride; k0 += 32) {
        const unsigned slot = (k0 / 32) % 2;
        if (k0 + 32 < stride) fetch(k0 + 32);
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            Raw16 ah[2], al[2], b[2];
#pragma unroll
            for (unsigned i = 0; i < 2; ++i) {
                ah[i] = raw16(&la[slot][h][0][(wm + i * 16 + lane % 16) * 16]);
                al[i] = raw16(&la[slot][h][1][(wm + i * 16 + lane % 16) * 16]);
                b[i] = raw16(&lb[slot][h][(wn + i * 16 + lane % 16) * 16]);
            }
#pragma unroll
            for (unsigned i = 0; i < 2; ++i)
#pragma unroll
                for (unsigned j = 0; j < 2; ++j) {
                    hi[i][j] = wmma(ah[i], b[j], hi[i][j]);
                    lo[i][j] = wmma(al[i], b[j], lo[i][j]);
                }
            const unsigned k_end = k0 + 16 * (h + 1);
            if (Blocked && (k_end % v2_block == 0 || k_end >= stride)) {
#pragma unroll
                for (unsigned i = 0; i < 2; ++i)
#pragma unroll
                    for (unsigned j = 0; j < 2; ++j) {
#pragma unroll
                        for (unsigned e = 0; e < 8; ++e) v2_merge<true>(hi[i][j][e], lo[i][j][e], s[i][j][e]);
                        hi[i][j] = Float8{}; lo[i][j] = Float8{};
                    }
            }
        }
        if (k0 + 32 < stride) store(slot ^ 1U);
        __syncthreads();
    }
    unsigned bad = 0;
#pragma unroll
    for (unsigned i = 0; i < 2; ++i)
#pragma unroll
        for (unsigned j = 0; j < 2; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m_tile + wm + i * 16 + 2 * e + lane / 16, n = n_tile + wn + j * 16 + lane % 16;
                float y;
                if constexpr (Blocked) y = s[i][j][e];
                else { v2_merge<false>(hi[i][j][e], lo[i][j][e], y); }
                if (r < rows && n < outputs) {
                    bad |= (bit_cast<unsigned>(y) & 0x7f800000U) == 0x7f800000U;
                    out[std::size_t(r) * outputs + n] = y;
                }
            }
    if (bad) atomicOr(failure, 1U);
}
// Blocked v2 (hilo_k128) register variant for long-K projections: eight waves,
// each owning 32 rows x 32 outputs with high/low accumulators and the
// block sum, 128 rows x 64 outputs per workgroup, one row tile's
// operands live at a time.
__global__ __launch_bounds__(256) void hilo_direct_k128(const _Float16* weights, const _Float16* x, unsigned rows, unsigned outputs,
                                                       unsigned stride, float* out, unsigned* failure) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m0 = blockIdx.y * 128 + (wave / 2) * 32, n0 = blockIdx.x * 64 + (wave % 2) * 32;
    const auto* xl = x + std::size_t(rows) * stride;
    Float8 h[2][2]{}, l[2][2]{};
    float s[2][2][8]{};
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        Raw16 b[2];
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const auto n = n0 + j * 16 + lane % 16;
            b[j] = n < outputs ? raw16_at(weights, unsigned((std::size_t(n / 16) * (stride / 16) + k0 / 16) * 256 + (n % 16) * 16)) : Raw16{};
        }
#pragma unroll
        for (unsigned i = 0; i < 2; ++i) {
            const auto r = m0 + i * 16 + lane % 16;
            const Raw16 ah = r < rows ? raw16_at(x, unsigned(std::size_t(k0) * rows + r * 16)) : Raw16{};
            const Raw16 al = r < rows ? raw16_at(xl, unsigned(std::size_t(k0) * rows + r * 16)) : Raw16{};
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                h[i][j] = wmma(ah, b[j], h[i][j]);
                l[i][j] = wmma(al, b[j], l[i][j]);
            }
        }
        if ((k0 + 16) % v2_block == 0 || k0 + 16 >= stride) {
#pragma unroll
            for (unsigned i = 0; i < 2; ++i)
#pragma unroll
                for (unsigned j = 0; j < 2; ++j) {
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e) v2_merge<true>(h[i][j][e], l[i][j][e], s[i][j][e]);
                    h[i][j] = Float8{}; l[i][j] = Float8{};
                }
        }
    }
    unsigned bad = 0;
#pragma unroll
    for (unsigned i = 0; i < 2; ++i)
#pragma unroll
        for (unsigned j = 0; j < 2; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m0 + i * 16 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
                const float y = s[i][j][e];
                if (r < rows && n < outputs) {
                    bad |= (bit_cast<unsigned>(y) & 0x7f800000U) == 0x7f800000U;
                    out[std::size_t(r) * outputs + n] = y;
                }
            }
    if (bad) atomicOr(failure, 1U);
}
// Panels are [N16][K16][N][K]. Packing is outside prompt timing and consumes
// exactly the same decoded FP32 coefficients and FP16 conversion as gemm.
template <WeightFormat Format>
__global__ void prepare_kernel(LinearView w, _Float16* out, unsigned stride, unsigned* failure) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto count = std::size_t((w.outputs + 15U) / 16U) * 16 * stride;
    if (i >= count) return;
    const auto n = unsigned(i / (16ULL * stride)) * 16 + unsigned(i / 16) % 16;
    const auto k = unsigned(i / 256) % (stride / 16) * 16 + unsigned(i % 16);
    w.format = Format;
    out[i] = rounded(n < w.outputs && k < w.inputs ? gpu::coefficient(w, n, k) : 0.0F, failure);
}
// Stage four K16 panels per barrier. Four waves share a 64x64 output tile;
// each wave owns four accumulators throughout the increasing K16 sequence.
// A transposed LDS layout makes lanes visit adjacent rows/columns, avoiding
// the previous stride-16 bank conflicts. Arithmetic is still issued K16 by K16.
__global__ __launch_bounds__(128) void prepared_gemm(const _Float16* weights, const _Float16* x,
                                                    unsigned rows, unsigned outputs, unsigned stride,
                                                    float* out, unsigned* failure) {
    __shared__ _Float16 ax[2][64 * 64], bw[2][64 * 64];
    const unsigned tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const unsigned m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const unsigned wm = (wave / 2) * 32, wn = (wave % 2) * 32;
    Float8 c00{}, c01{}, c10{}, c11{};
    for (unsigned base = 0; base < stride; base += 64) {
        const unsigned slot = (base / 64) % 2;
        for (unsigned i = tid; i < 64 * 64; i += 128) {
            const unsigned r = i / 64, k = i % 64;
            ax[slot][k * 64 + r] = m0 + r < rows && base + k < stride
                ? x[std::size_t(m0 + r) * stride + base + k] : _Float16(0);
            bw[slot][k * 64 + r] = n0 + r < outputs && base + k < stride
                ? weights[(std::size_t((n0 + r) / 16) * (stride / 16) + (base + k) / 16) * 256 + (r % 16) * 16 + k % 16]
                : _Float16(0);
        }
        __syncthreads();
#pragma unroll
        for (unsigned t = 0; t < 4; ++t) {
            if (base + t * 16 >= stride) break;
            Half16 a0, a1, b0, b1;
#pragma unroll
            for (unsigned k = 0; k < 16; ++k) {
                const auto index = (t * 16 + k) * 64 + lane % 16;
                a0[k] = ax[slot][index + wm]; a1[k] = ax[slot][index + wm + 16];
                b0[k] = bw[slot][index + wn]; b1[k] = bw[slot][index + wn + 16];
            }
            c00 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, c00);
            c01 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, c01);
            c10 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, c10);
            c11 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, c11);
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned e = 0; e < 8; ++e) {
        const auto r = m0 + wm + 2 * e + lane / 16, n = n0 + wn + lane % 16;
        const float values[4] = {c00[e], c01[e], c10[e], c11[e]};
#pragma unroll
        for (unsigned i = 0; i < 2; ++i)
#pragma unroll
            for (unsigned j = 0; j < 2; ++j)
                if (r + i * 16 < rows && n + j * 16 < outputs) {
                    const float a = values[i * 2 + j];
                    if ((__builtin_bit_cast(unsigned, a) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
                    out[std::size_t(r + i * 16) * outputs + n + j * 16] = a;
                }
    }
}
// A register-only alternative: four waves own four independent 64x64 tiles in
// a 128x128 workgroup. Contiguous panel vectors feed sixteen WMMA accumulators
// per wave; the device caches provide immutable operand reuse across waves.
template <unsigned RowWaves, bool Panels = false>
__global__ __launch_bounds__(RowWaves * 64) void direct_gemm(const _Float16* weights, const _Float16* x,
                                                  unsigned rows, unsigned outputs, unsigned stride,
                                                  float* out, unsigned* failure) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m0 = blockIdx.y * (RowWaves * 64) + (wave / 2) * 64, n0 = blockIdx.x * 128 + (wave % 2) * 64;
    Float8 c[4][4]{};
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        Raw16 a[4], b[4];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const auto r = m0 + i * 16 + lane % 16, n = n0 + i * 16 + lane % 16;
            const auto index = Panels ? std::size_t(k0) * rows + r * 16 : std::size_t(r) * stride + k0;
            a[i] = r < rows ? raw16_at(x, unsigned(index)) : Raw16{};
            b[i] = n < outputs ? raw16_at(weights, unsigned((std::size_t(n / 16) * (stride / 16) + k0 / 16) * 256 + (n % 16) * 16)) : Raw16{};
        }
#pragma unroll
        for (unsigned i = 0; i < 4; ++i)
#pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                c[i][j] = wmma(a[i], b[j], c[i][j]);
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m0 + i * 16 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
                if (r < rows && n < outputs) {
                    const float a = c[i][j][e];
                    if ((bit_cast<unsigned>(a) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
                    out[std::size_t(r) * outputs + n] = a;
                }
            }
}
// Narrow projections need more workgroups than a 128-column output tile gives.
// Four waves own a 64x64 tile without changing any output's K16 recurrence.
__global__ __launch_bounds__(128) void narrow_gemm(const _Float16* weights, const _Float16* x,
                                                  unsigned rows, unsigned outputs, unsigned stride,
                                                  float* out, unsigned* failure) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m0 = blockIdx.y * 64 + (wave / 2) * 32, n0 = blockIdx.x * 64 + (wave % 2) * 32;
    Float8 c[2][2]{};
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        Raw16 a[2], b[2];
#pragma unroll
        for (unsigned i = 0; i < 2; ++i) {
            const auto r = m0 + i * 16 + lane % 16, n = n0 + i * 16 + lane % 16;
            a[i] = r < rows ? raw16_at(x, unsigned(std::size_t(k0) * rows + r * 16)) : Raw16{};
            b[i] = n < outputs ? raw16_at(weights, unsigned((std::size_t(n / 16) * (stride / 16) + k0 / 16) * 256 + (n % 16) * 16)) : Raw16{};
        }
#pragma unroll
        for (unsigned i = 0; i < 2; ++i)
#pragma unroll
            for (unsigned j = 0; j < 2; ++j)
                c[i][j] = wmma(a[i], b[j], c[i][j]);
    }
#pragma unroll
    for (unsigned i = 0; i < 2; ++i)
#pragma unroll
        for (unsigned j = 0; j < 2; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m0 + i * 16 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
                if (r < rows && n < outputs) {
                    const float a = c[i][j][e];
                    if ((bit_cast<unsigned>(a) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
                    out[std::size_t(r) * outputs + n] = a;
                }
            }
}
// Small last-row head calls also reuse the prepared panels. One wave owns N16,
// avoiding the padded row work of the large workgroup while retaining K order.
__global__ __launch_bounds__(32) void prepared_rows(const _Float16* weights, const _Float16* x,
                                                   unsigned rows, unsigned outputs, unsigned stride,
                                                   float* out, unsigned* failure) {
    const unsigned lane = threadIdx.x, r = blockIdx.y * 16 + lane % 16, n = blockIdx.x * 16 + lane % 16;
    Float8 c{};
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        const Raw16 a = r < rows ? raw16_at(x, unsigned(std::size_t(k0) * rows + r * 16)) : Raw16{};
        const Raw16 b = n < outputs ? raw16_at(weights, unsigned((std::size_t(n / 16) * (stride / 16) + k0 / 16) * 256 + (n % 16) * 16)) : Raw16{};
        c = wmma(a, b, c);
    }
#pragma unroll
    for (unsigned e = 0; e < 8; ++e) {
        const auto row = blockIdx.y * 16 + 2 * e + lane / 16;
        if (row < rows && n < outputs) {
            if ((bit_cast<unsigned>(c[e]) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
            out[std::size_t(row) * outputs + n] = c[e];
        }
    }
}
// Four waves, each with 2x2 independent 16x16 accumulators. LDS shares each
// decoded weight tile across 64 prompt rows; only increasing K tiles are issued.
template <WeightFormat Format>
__global__ __launch_bounds__(128) void gemm(LinearView w, const _Float16* x, unsigned rows, unsigned stride,
                                           float* out, unsigned* failure) {
    __shared__ _Float16 ax[64 * 16], bw[64 * 16];
    const unsigned tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const unsigned m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const unsigned wm = (wave / 2) * 32, wn = (wave % 2) * 32;
    Float8 c00{}, c01{}, c10{}, c11{};
    w.format = Format;
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        for (unsigned i = tid; i < 64 * 16; i += 128) {
            const auto r = i / 16, k = i % 16;
            ax[i] = m0 + r < rows ? x[std::size_t(m0 + r) * stride + k0 + k] : _Float16(0);
            const float b = n0 + r < w.outputs && k0 + k < w.inputs ? gpu::coefficient(w, n0 + r, k0 + k) : 0.0F;
            bw[i] = rounded(b, failure);
        }
        __syncthreads();
        Half16 a0, a1, b0, b1;
#pragma unroll
        for (unsigned k = 0; k < 16; ++k) {
            a0[k] = ax[(wm + lane % 16) * 16 + k];
            a1[k] = ax[(wm + 16 + lane % 16) * 16 + k];
            b0[k] = bw[(wn + lane % 16) * 16 + k];
            b1[k] = bw[(wn + 16 + lane % 16) * 16 + k];
        }
        c00 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, c00);
        c01 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, c01);
        c10 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, c10);
        c11 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, c11);
        __syncthreads();
    }
#pragma unroll
    for (unsigned e = 0; e < 8; ++e) {
        const auto r = m0 + wm + 2 * e + lane / 16, n = n0 + wn + lane % 16;
        const float values[4] = {c00[e], c01[e], c10[e], c11[e]};
#pragma unroll
        for (unsigned i = 0; i < 2; ++i)
#pragma unroll
            for (unsigned j = 0; j < 2; ++j)
                if (r + i * 16 < rows && n + j * 16 < w.outputs) {
                    const auto a = values[i * 2 + j];
                    if ((__builtin_bit_cast(unsigned, a) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
                    out[std::size_t(r + i * 16) * w.outputs + n + j * 16] = a;
                }
    }
}
template <WeightFormat Format>
__global__ __launch_bounds__(32) void reference_gemm(LinearView w, const _Float16* x, unsigned rows, unsigned stride,
                                                    float* out, unsigned* failure) {
    const auto lane = threadIdx.x, r = blockIdx.y * 16 + lane % 16, n = blockIdx.x * 16 + lane % 16;
    Float8 c{};
    w.format = Format;
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        Half16 a, b;
#pragma unroll
        for (unsigned k = 0; k < 16; ++k) {
            a[k] = r < rows ? x[std::size_t(r) * stride + k0 + k] : _Float16(0);
            b[k] = rounded(n < w.outputs && k0 + k < w.inputs ? gpu::coefficient(w, n, k0 + k) : 0.0F, failure);
        }
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
#pragma unroll
    for (unsigned e = 0; e < 8; ++e) {
        const auto row = blockIdx.y * 16 + 2 * e + lane / 16;
        if (row < rows && n < w.outputs) {
            if ((__builtin_bit_cast(unsigned, c[e]) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
            out[std::size_t(row) * w.outputs + n] = c[e];
        }
    }
}
// LDS-staged variant of direct_gemm<.., true>: a workgroup of 2 x WavesN waves
// stages 128 rows x (64*WavesN) outputs x K32 of the panel operands into
// double-buffered LDS (next stage prefetched into registers while the current
// one is multiplied), so each operand is fetched once per workgroup instead of
// once per wave. Every accumulator still issues the same increasing K16
// instructions on the same FP16 operands. Requires stride % 32 == 0.
template <unsigned WavesN>
__global__ __launch_bounds__(WavesN * 64) void staged_gemm(const _Float16* weights, const _Float16* x,
                                                          unsigned rows, unsigned outputs, unsigned stride,
                                                          float* out, unsigned* failure) {
    constexpr unsigned threads = WavesN * 64, tile_m = 128, tile_n = 64 * WavesN;
    constexpr unsigned a_vectors = tile_m * 32 / 8, b_vectors = tile_n * 32 / 8; // 16-byte vectors per stage
    constexpr unsigned a_loads = a_vectors / threads, b_loads = b_vectors / threads;
    using Vec = unsigned __attribute__((ext_vector_type(4)));
    // [stage][k16 of the stage][row or output][16]
    __shared__ _Float16 la[2][2][tile_m * 16], lb[2][2][tile_n * 16];
    const unsigned tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const unsigned m_tile = blockIdx.y * tile_m, n_tile = blockIdx.x * tile_n;
    const unsigned wm = (wave / WavesN) * 64, wn = (wave % WavesN) * 64;
    const unsigned k16s = stride / 16, n16s = (outputs + 15U) / 16U;
    Vec ra[a_loads], rb[b_loads];
    // Vector v of a stage: k16 half h = v / (rows*2), then row and 16-byte half.
    auto fetch = [&](unsigned k0) {
#pragma unroll
        for (unsigned i = 0; i < a_loads; ++i) {
            const unsigned v = tid + i * threads, h = v / (tile_m * 2), row = (v / 2) % tile_m, part = v % 2;
            const unsigned r = m_tile + row;
            ra[i] = r < rows ? *reinterpret_cast<const Vec*>(x + (std::size_t(k0 / 16 + h) * rows + r) * 16 + part * 8) : Vec{};
        }
#pragma unroll
        for (unsigned i = 0; i < b_loads; ++i) {
            const unsigned v = tid + i * threads, h = v / (tile_n * 2), n = (v / 2) % tile_n, part = v % 2;
            const unsigned j = (n_tile + n) / 16;
            rb[i] = j < n16s ? *reinterpret_cast<const Vec*>(weights + (std::size_t(j) * k16s + k0 / 16 + h) * 256 + (n % 16) * 16 + part * 8)
                             : Vec{};
        }
    };
    auto store = [&](unsigned slot) {
#pragma unroll
        for (unsigned i = 0; i < a_loads; ++i) {
            const unsigned v = tid + i * threads, h = v / (tile_m * 2), row = (v / 2) % tile_m, part = v % 2;
            *reinterpret_cast<Vec*>(&la[slot][h][row * 16 + part * 8]) = ra[i];
        }
#pragma unroll
        for (unsigned i = 0; i < b_loads; ++i) {
            const unsigned v = tid + i * threads, h = v / (tile_n * 2), n = (v / 2) % tile_n, part = v % 2;
            *reinterpret_cast<Vec*>(&lb[slot][h][n * 16 + part * 8]) = rb[i];
        }
    };
    Float8 c[4][4]{};
    fetch(0); store(0);
    __syncthreads();
    for (unsigned k0 = 0; k0 < stride; k0 += 32) {
        const unsigned slot = (k0 / 32) % 2;
        if (k0 + 32 < stride) fetch(k0 + 32);
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            Half16 a[4], b[4];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                a[i] = __builtin_bit_cast(Half16, raw16(&la[slot][h][(wm + i * 16 + lane % 16) * 16]));
                b[i] = __builtin_bit_cast(Half16, raw16(&lb[slot][h][(wn + i * 16 + lane % 16) * 16]));
            }
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                    c[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], c[i][j]);
        }
        if (k0 + 32 < stride) store(slot ^ 1U);
        __syncthreads();
    }
    unsigned bad = 0;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m_tile + wm + i * 16 + 2 * e + lane / 16, n = n_tile + wn + j * 16 + lane % 16;
                if (r < rows && n < outputs) {
                    const float a = c[i][j][e];
                    bad |= (bit_cast<unsigned>(a) & 0x7f800000U) == 0x7f800000U;
                    out[std::size_t(r) * outputs + n] = a;
                }
            }
    if (bad) atomicOr(failure, 1U);
}
// Gate and up projections of the same panel input in one workgroup, in the
// register-only style of direct_gemm: eight waves, each owning 64 rows x 32
// outputs of both matrices, with the same increasing K16 chains as two
// separate GEMMs. The epilogue forms gelu_tanh(gate) * up in FP32 exactly as
// the product kernel, checks it as check_finite would, and stores it straight
// into the next linear's FP16 input panels with convert's RNE/overflow rule
// and abs-bits maximum. 256 rows x 64 outputs per workgroup.
// Low: also store the g4-wmma-pf-v2 low panel RNE16((p - high) * 2^12) at
// product + rows * outputs, so a corrected down keeps this fusion.
template <bool Low>
__global__ __launch_bounds__(256) void gelu_pair_gemm(const _Float16* gate, const _Float16* up, const _Float16* x,
                                                     unsigned rows, unsigned outputs, unsigned stride,
                                                     _Float16* product, unsigned* failure, unsigned* maximum) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m0 = blockIdx.y * 256 + (wave / 2) * 64, n0 = blockIdx.x * 64 + (wave % 2) * 32;
    Float8 cg[4][2]{}, cu[4][2]{};
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        Raw16 a[4], bg[2], bu[2];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const auto r = m0 + i * 16 + lane % 16;
            a[i] = r < rows ? raw16_at(x, unsigned(std::size_t(k0) * rows + r * 16)) : Raw16{};
        }
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const auto n = n0 + j * 16 + lane % 16;
            const auto offset = (std::size_t(n / 16) * (stride / 16) + k0 / 16) * 256 + (n % 16) * 16;
            bg[j] = n < outputs ? raw16_at(gate, unsigned(offset)) : Raw16{};
            bu[j] = n < outputs ? raw16_at(up, unsigned(offset)) : Raw16{};
        }
#pragma unroll
        for (unsigned i = 0; i < 4; ++i)
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                cg[i][j] = wmma(a[i], bg[j], cg[i][j]);
                cu[i][j] = wmma(a[i], bu[j], cu[i][j]);
            }
    }
    unsigned bad = 0, high = 0;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 2; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m0 + i * 16 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
                if (r < rows && n < outputs) {
                    const float g = cg[i][j][e], u = cu[i][j][e];
                    bad |= (bit_cast<unsigned>(g) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                    bad |= (bit_cast<unsigned>(u) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                    float p = gelu_tanh(g) * u;
                    // The product is an FP32 value first (it was stored), then
                    // converted: never fold the multiply into v_fma_mix.
                    asm volatile("" : "+v"(p));
                    bad |= (bit_cast<unsigned>(p) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                    high = max(high, bit_cast<unsigned>(p) & 0x7fffffffU);
                    const auto hp = static_cast<_Float16>(p);
                    bad |= (bit_cast<unsigned short>(hp) & 0x7c00U) == 0x7c00U ? 4U : 0U;
                    product[std::size_t(n / 16) * rows * 16 + std::size_t(r) * 16 + n % 16] = hp;
                    if constexpr (Low) {
                        const auto lp = static_cast<_Float16>((p - float(hp)) * v2_low_scale);
                        bad |= (bit_cast<unsigned short>(lp) & 0x7c00U) == 0x7c00U ? 4U : 0U;
                        product[std::size_t(rows) * outputs + std::size_t(n / 16) * rows * 16 + std::size_t(r) * 16 + n % 16] = lp;
                    }
                }
            }
    if (bad) atomicOr(failure, bad);
    for (unsigned d = 16; d; d >>= 1U) high = max(high, __shfl_down(high, d, 32));
    if (lane == 0) atomicMax(maximum, high);
}
} // namespace
bool hilo_narrow_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_HILO_NARROW");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool hilo_direct_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_HILO_DIRECT");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
// Unguarded full-tile instantiations of the hilo kernels (same bytes);
// STRIX_G4_PREFILL_FULL_TILES=0 keeps the guarded loads everywhere.
bool full_tiles_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_FULL_TILES");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool staged_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_STAGED");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool tiled_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_TILING");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool prepared_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_PREPARE");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
unsigned direct_mode() {
    static const unsigned mode = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_DIRECT");
        if (!p || std::string_view(p) == "3") return 3U;
        if (std::string_view(p) == "4") return 4U;
        if (std::string_view(p) == "2") return 2U;
        if (std::string_view(p) == "0") return 0U;
        if (std::string_view(p) == "1") return 1U;
        throw std::invalid_argument("unknown prefill direct tile");
    }();
    return mode;
}
namespace {
// Widest output count routed to the 64x64 narrow tiles (same K16 chain, same
// bytes). 2048 also moves q (2048 outputs) off the 128x128 tiles: +0.7 % at
// 2K, flat at 512/8K (one node, two interleaved runs); STRIX_G4_PREFILL_NARROW_MAX
// overrides it for tile A/Bs.
unsigned narrow_outputs() {
    static const unsigned limit = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_NARROW_MAX");
        return p ? unsigned(std::strtoul(p, nullptr, 10)) : 2048U;
    }();
    return limit;
}
// Every panel GEMM issues the same increasing K16 WMMA chain per output.
void panel_gemm(const PreparedLinear& prepared, const _Float16* x, unsigned rows, float* out, unsigned* failure,
                hipStream_t q, unsigned mode) {
    const auto w = prepared.weights();
    const auto stride = ((w.inputs + 15U) / 16U) * 16U;
    // The staged kernel measured faster only for long-K projections (down:
    // 33 vs 29 TF at M512); wide outputs keep the register-only tiles.
    const bool staged = mode >= 3 && staged_enabled() && stride % 32 == 0 && stride >= 8192 && w.outputs > 1024 && w.outputs <= 4096 && rows > 8;
    const auto direct = mode == 4 ? 3U : mode;
    if (staged && w.outputs >= 4096)
        hipLaunchKernelGGL(staged_gemm<4>, dim3((w.outputs + 255U) / 256U, (rows + 127U) / 128U), dim3(256), 0, q,
                           prepared.panels().as<_Float16>(), x, rows, w.outputs, stride, out, failure);
    else if (staged)
        hipLaunchKernelGGL(staged_gemm<2>, dim3((w.outputs + 127U) / 128U, (rows + 127U) / 128U), dim3(128), 0, q,
                           prepared.panels().as<_Float16>(), x, rows, w.outputs, stride, out, failure);
    else if (direct == 3 && rows <= 8)
        hipLaunchKernelGGL(prepared_rows, dim3((w.outputs + 15U) / 16U, (rows + 15U) / 16U), dim3(32), 0, q,
                           prepared.panels().as<_Float16>(), x, rows, w.outputs, stride, out, failure);
    else if (direct == 3 && w.outputs <= narrow_outputs())
        hipLaunchKernelGGL(narrow_gemm, dim3((w.outputs + 63U) / 64U, (rows + 63U) / 64U), dim3(128), 0, q,
                           prepared.panels().as<_Float16>(), x, rows, w.outputs, stride, out, failure);
    else if (direct == 3 && w.outputs >= 8192)
        hipLaunchKernelGGL((direct_gemm<4, true>), dim3((w.outputs + 127U) / 128U, (rows + 255U) / 256U), dim3(256), 0, q,
                           prepared.panels().as<_Float16>(), x, rows, w.outputs, stride, out, failure);
    else if (direct == 3)
        hipLaunchKernelGGL((direct_gemm<2, true>), dim3((w.outputs + 127U) / 128U, (rows + 127U) / 128U), dim3(128), 0, q,
                           prepared.panels().as<_Float16>(), x, rows, w.outputs, stride, out, failure);
    else if (direct == 2 && w.outputs > 512)
        hipLaunchKernelGGL((direct_gemm<4>), dim3((w.outputs + 127U) / 128U, (rows + 255U) / 256U), dim3(256), 0, q,
                           prepared.panels().as<_Float16>(), x, rows, w.outputs, stride, out, failure);
    else if (direct)
        hipLaunchKernelGGL((direct_gemm<2>), dim3((w.outputs + 127U) / 128U, (rows + 127U) / 128U), dim3(128), 0, q,
                           prepared.panels().as<_Float16>(), x, rows, w.outputs, stride, out, failure);
    else
        hipLaunchKernelGGL(prepared_gemm, dim3((w.outputs + 63U) / 64U, (rows + 63U) / 64U), dim3(128), 0, q,
                           prepared.panels().as<_Float16>(), x, rows, w.outputs, stride, out, failure);
    checked();
}
bool exl3_panel_route(const PreparedLinear* prepared, bool tiled) {
    return prepared && tiled && tiled_enabled() && direct_mode() >= 3 && exl3_gemm_enabled();
}
} // namespace
bool exl3_gemm_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_EXL3_PREFILL_GEMM");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
PreparedLinear::PreparedLinear(LinearView w, gpu::Queue queue) : weights_(w) {
    if (!w.data || !w.inputs || w.inputs > 10752 || !w.outputs || w.outputs > vocab ||
        (w.format == WeightFormat::exl3_mul1 ? !exl3::geometry(w) : w.row_bytes < packed_row_bytes(w.format, w.inputs))) throw std::invalid_argument("prepared linear geometry");
    const auto stride = (w.inputs + 15U) / 16U * 16U;
    panels_.resize(std::size_t((w.outputs + 15U) / 16U) * 16 * stride * sizeof(std::uint16_t));
    gpu::Buffer failure(sizeof(unsigned)); failure.zero();
    const auto q = static_cast<hipStream_t>(queue);
    switch (w.format) {
#define G4_PREPARE(F) case WeightFormat::F: \
        hipLaunchKernelGGL((prepare_kernel<WeightFormat::F>), dim3(unsigned((panels_.size() / 2 + 255) / 256)), dim3(256), 0, q, \
                           w, panels_.as<_Float16>(), stride, failure.as<unsigned>()); break
        G4_PREPARE(bf16); G4_PREPARE(f32); G4_PREPARE(q8_0); G4_PREPARE(q4_k); G4_PREPARE(q6_k);
        G4_PREPARE(exl3_mul1);
#undef G4_PREPARE
        default: throw std::invalid_argument("prepared linear format");
    }
    checked(); gpu::synchronize(queue);
    unsigned flag = 0; failure.download(&flag, sizeof(flag));
    if (flag) throw std::runtime_error("prepared linear FP16 overflow/nonfinite coefficient");
}
void linear(LinearView w, const float* x, unsigned rows, float* out, void* scratch, unsigned* failure,
            unsigned* maximum, gpu::Queue queue, bool tiled, const PreparedLinear* prepared, void* converted_event, bool converted,
            LinearPrecision precision) {
    if (precision == LinearPrecision::hilo || precision == LinearPrecision::hilo_k128) {
        if (!rows || rows > max_chunk_rows || !w.data || !w.inputs || !w.outputs || !out || !scratch || !failure || !maximum || !prepared ||
            w.format == WeightFormat::exl3_mul1 || (!converted && !x))
            throw std::invalid_argument("WMMA v2 linear geometry, scratch or prepared panels");
        const auto p = prepared->weights();
        if (p.data != w.data || p.inputs != w.inputs || p.outputs != w.outputs || p.row_bytes != w.row_bytes || p.format != w.format)
            throw std::invalid_argument("prepared linear tensor mismatch");
        const auto stride = ((w.inputs + 15U) / 16U) * 16U;
        const auto q = static_cast<hipStream_t>(queue);
        if (!converted)
            hipLaunchKernelGGL(convert_split, dim3(unsigned((std::size_t(rows) * stride + 255) / 256)), dim3(256), 0, q,
                               x, static_cast<_Float16*>(scratch), rows, w.inputs, stride, failure, maximum);
        checked();
        if (converted_event)
            if (const auto e = hipEventRecord(static_cast<hipEvent_t>(converted_event), q); e != hipSuccess)
                throw std::runtime_error(hipGetErrorString(e));
        const auto* panels = prepared->panels().as<_Float16>();
        const auto* input = static_cast<const _Float16*>(scratch);
        const bool blocked = precision == LinearPrecision::hilo_k128;
#define G4_HILO(B, TM, TN) do { \
            const dim3 grid((w.outputs + 32 * TN - 1) / (32 * TN), (rows + 32 * TM - 1) / (32 * TM)); \
            if (rows % (32 * TM) == 0 && w.outputs % (32 * TN) == 0 && full_tiles_enabled()) \
                hipLaunchKernelGGL((hilo_gemm<B, TM, TN, false>), grid, dim3(128), 0, q, panels, input, rows, w.outputs, stride, out, failure); \
            else hipLaunchKernelGGL((hilo_gemm<B, TM, TN, true>), grid, dim3(128), 0, q, panels, input, rows, w.outputs, stride, out, failure); \
        } while (false)
        // Long-K blocked projections (down under K128) stage through LDS; the
        // unblocked staged form measured slower than hilo_direct (314 vs 291 ms
        // for down at 2K), so it is not used.
        const bool staged = blocked && staged_enabled() && stride % 32 == 0 && stride >= 8192 && rows > 16;
        if (blocked && rows > 16 && stride >= 8192 && !staged_enabled())
            hipLaunchKernelGGL(hilo_direct_k128, dim3((w.outputs + 63U) / 64U, (rows + 127U) / 128U), dim3(256), 0, q,
                               panels, input, rows, w.outputs, stride, out, failure);
        else if (staged && blocked)
            hipLaunchKernelGGL(staged_hilo<true>, dim3((w.outputs + 63U) / 64U, (rows + 127U) / 128U), dim3(256), 0, q,
                               panels, input, rows, w.outputs, stride, out, failure);
        else if (staged)
            hipLaunchKernelGGL(staged_hilo<false>, dim3((w.outputs + 63U) / 64U, (rows + 127U) / 128U), dim3(256), 0, q,
                               panels, input, rows, w.outputs, stride, out, failure);
        else if (rows <= 16) { if (blocked) G4_HILO(true, 1, 4); else G4_HILO(false, 1, 4); }
        else if (blocked) G4_HILO(true, 2, 2);
        // Narrow projections (K/V, PLE gate) need more workgroups than 128x128 tiles give.
        else if (w.outputs <= 1024 && hilo_narrow_enabled()) G4_HILO(false, 2, 2);
        else if (hilo_direct_enabled())
        {
            const dim3 grid((w.outputs + 127U) / 128U, (rows + 127U) / 128U);
            if (rows % 128 == 0 && w.outputs % 128 == 0 && full_tiles_enabled())
                hipLaunchKernelGGL(hilo_direct<false>, grid, dim3(256), 0, q, panels, input, rows, w.outputs, stride, out, failure);
            else
                hipLaunchKernelGGL(hilo_direct<true>, grid, dim3(256), 0, q, panels, input, rows, w.outputs, stride, out, failure);
        }
        else G4_HILO(false, 2, 4);
#undef G4_HILO
        checked();
        return;
    }
    if (w.format == WeightFormat::exl3_mul1) {
        if (prepared) {
            const auto p = prepared->weights();
            if (p.data != w.data || p.inputs != w.inputs || p.outputs != w.outputs || p.row_bytes != w.row_bytes || p.format != w.format)
                throw std::invalid_argument("EXL3 prepared linear mismatch");
        }
        if (exl3_panel_route(prepared, tiled)) {
            const auto q = static_cast<hipStream_t>(queue);
            if (!converted) exl3::device_input_panels(w, x, rows, scratch, failure, maximum, queue);
            if (converted_event)
                if (const auto e = hipEventRecord(static_cast<hipEvent_t>(converted_event), q); e != hipSuccess)
                    throw std::runtime_error(hipGetErrorString(e));
            panel_gemm(*prepared, static_cast<const _Float16*>(scratch), rows, out, failure, q, direct_mode());
            exl3::device_output(w, out, rows, failure, queue);
            return;
        }
        if (converted) throw std::invalid_argument("pre-converted EXL3 input requires panel GEMMs");
        exl3::device_prefill(w, x, rows, out, scratch, failure, maximum,
            prepared ? prepared->panels().data() : nullptr, queue, converted_event);
        return;
    }
    if (!rows || rows > max_chunk_rows || !w.data || !w.inputs || !w.outputs || !x || !out || !scratch || !failure || !maximum)
        throw std::invalid_argument("WMMA linear geometry or scratch");
    if (w.inputs > 10752 || w.outputs > vocab || w.row_bytes < packed_row_bytes(w.format, w.inputs))
        throw std::invalid_argument("WMMA linear tensor bounds");
    if (w.format != WeightFormat::bf16 && w.format != WeightFormat::f32 && w.format != WeightFormat::q8_0 &&
        w.format != WeightFormat::q4_k && w.format != WeightFormat::q6_k)
        throw std::invalid_argument("WMMA linear weight format");
    const auto stride = ((w.inputs + 15U) / 16U) * 16U;
    const auto q = static_cast<hipStream_t>(queue);
    tiled &= tiled_enabled();
    const auto mode = prepared && tiled ? direct_mode() : 0;
    const auto direct = mode == 4 ? 3U : mode;
    if (converted && direct != 3) throw std::invalid_argument("pre-converted input requires panel GEMMs");
    if (converted) {}
    else if (direct == 3)
        hipLaunchKernelGGL((convert<true>), dim3(unsigned((std::size_t(rows) * stride + 255) / 256)), dim3(256), 0, q,
                           x, static_cast<_Float16*>(scratch), rows, w.inputs, stride, failure, maximum);
    else
        hipLaunchKernelGGL((convert<false>), dim3(unsigned((std::size_t(rows) * stride + 255) / 256)), dim3(256), 0, q,
                           x, static_cast<_Float16*>(scratch), rows, w.inputs, stride, failure, maximum);
    checked();
    if (converted_event)
        if (const auto e = hipEventRecord(static_cast<hipEvent_t>(converted_event), q); e != hipSuccess)
            throw std::runtime_error(hipGetErrorString(e));
    if (prepared && tiled) {
        const auto p = prepared->weights();
        if (p.data != w.data || p.inputs != w.inputs || p.outputs != w.outputs || p.row_bytes != w.row_bytes || p.format != w.format)
            throw std::invalid_argument("prepared linear tensor mismatch");
        panel_gemm(*prepared, static_cast<const _Float16*>(scratch), rows, out, failure, q, mode);
        return;
    }
    switch (w.format) {
#define G4_PF_FORMAT(F) case WeightFormat::F: \
    if (tiled) \
        hipLaunchKernelGGL((gemm<WeightFormat::F>), dim3((w.outputs + 63U) / 64U, (rows + 63U) / 64U), dim3(128), 0, q, \
                           w, static_cast<const _Float16*>(scratch), rows, stride, out, failure); \
    else \
        hipLaunchKernelGGL((reference_gemm<WeightFormat::F>), dim3((w.outputs + 15U) / 16U, (rows + 15U) / 16U), dim3(32), 0, q, \
                           w, static_cast<const _Float16*>(scratch), rows, stride, out, failure); break
        G4_PF_FORMAT(bf16);
        G4_PF_FORMAT(f32);
        G4_PF_FORMAT(q8_0);
        G4_PF_FORMAT(q4_k);
        G4_PF_FORMAT(q6_k);
#undef G4_PF_FORMAT
        case WeightFormat::exl3_mul1: break; // returned above (EXL3 packs have their own route)
    }
    checked();
}
bool fused_ffn_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_FUSE_FFN");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool gelu_pair_supported(const PreparedLinear& gate, const PreparedLinear& up) {
    const auto g = gate.weights(), u = up.weights();
    return g.inputs == u.inputs && g.outputs == u.outputs && g.format != WeightFormat::exl3_mul1 &&
           u.format != WeightFormat::exl3_mul1 && g.outputs % 16U == 0;
}
void gelu_pair(const PreparedLinear& gate, const PreparedLinear& up, const float* x, unsigned rows, void* scratch,
               void* product, unsigned* failure, unsigned* maximum, gpu::Queue queue, bool low, bool split, bool blocked) {
    const auto g = gate.weights();
    if (!rows || rows > max_chunk_rows || !x || !scratch || !product || !failure || !maximum || !gelu_pair_supported(gate, up))
        throw std::invalid_argument("fused gate/up geometry");
    const auto stride = (g.inputs + 15U) / 16U * 16U;
    const auto q = static_cast<hipStream_t>(queue);
    if (split)
        hipLaunchKernelGGL(convert_split, dim3(unsigned((std::size_t(rows) * stride + 255) / 256)), dim3(256), 0, q,
                           x, static_cast<_Float16*>(scratch), rows, g.inputs, stride, failure, maximum);
    else
        hipLaunchKernelGGL((convert<true>), dim3(unsigned((std::size_t(rows) * stride + 255) / 256)), dim3(256), 0, q,
                           x, static_cast<_Float16*>(scratch), rows, g.inputs, stride, failure, maximum);
    checked();
    // Both projections read the same converted input: same recorded maximum.
    if (const auto e = hipMemcpyAsync(maximum + 1, maximum, sizeof(unsigned), hipMemcpyDeviceToDevice, q); e != hipSuccess)
        throw std::runtime_error(hipGetErrorString(e));
    if (split && blocked) {
        const dim3 grid((g.outputs + 31U) / 32U, (rows + 127U) / 128U);
        const bool full = rows % 128 == 0 && g.outputs % 32 == 0 && full_tiles_enabled();
#define G4_PAIR_K128(L, G) hipLaunchKernelGGL((gelu_pair_hilo_k128<L, G>), grid, dim3(256), 0, q, gate.panels().as<_Float16>(), up.panels().as<_Float16>(), \
                               static_cast<const _Float16*>(scratch), rows, g.outputs, stride, static_cast<_Float16*>(product), failure, maximum + 2)
        if (low) { if (full) G4_PAIR_K128(true, false); else G4_PAIR_K128(true, true); }
        else { if (full) G4_PAIR_K128(false, false); else G4_PAIR_K128(false, true); }
#undef G4_PAIR_K128
        checked();
        return;
    }
    if (split) {
        const dim3 grid((g.outputs + 63U) / 64U, (rows + 127U) / 128U);
        const bool full = rows % 128 == 0 && g.outputs % 64 == 0 && full_tiles_enabled();
#define G4_PAIR_HILO(L, G) hipLaunchKernelGGL((gelu_pair_hilo<L, G>), grid, dim3(256), 0, q, gate.panels().as<_Float16>(), up.panels().as<_Float16>(), \
                               static_cast<const _Float16*>(scratch), rows, g.outputs, stride, static_cast<_Float16*>(product), failure, maximum + 2)
        if (low) { if (full) G4_PAIR_HILO(true, false); else G4_PAIR_HILO(true, true); }
        else { if (full) G4_PAIR_HILO(false, false); else G4_PAIR_HILO(false, true); }
#undef G4_PAIR_HILO
        checked();
        return;
    }
    const dim3 grid((g.outputs + 63U) / 64U, (rows + 255U) / 256U);
    if (low)
        hipLaunchKernelGGL(gelu_pair_gemm<true>, grid, dim3(256), 0, q, gate.panels().as<_Float16>(), up.panels().as<_Float16>(),
                           static_cast<const _Float16*>(scratch), rows, g.outputs, stride, static_cast<_Float16*>(product), failure, maximum + 2);
    else
        hipLaunchKernelGGL(gelu_pair_gemm<false>, grid, dim3(256), 0, q, gate.panels().as<_Float16>(), up.panels().as<_Float16>(),
                           static_cast<const _Float16*>(scratch), rows, g.outputs, stride, static_cast<_Float16*>(product), failure, maximum + 2);
    checked();
}
bool exl3_gelu_pair_supported(const PreparedLinear& gate, const PreparedLinear& up, LinearView down) {
    const auto g = gate.weights(), u = up.weights();
    return fused_ffn_enabled() && exl3_panel_route(&gate, true) && g.format == WeightFormat::exl3_mul1 &&
           u.format == WeightFormat::exl3_mul1 && down.format == WeightFormat::exl3_mul1 && g.inputs == u.inputs &&
           g.outputs == u.outputs && down.inputs == g.outputs && exl3::geometry(g) && exl3::geometry(u) && exl3::geometry(down);
}
void exl3_gelu_pair(const PreparedLinear& gate, const PreparedLinear& up, LinearView down, const float* x, unsigned rows,
                    void* scratch, float* gate_core, float* up_core, void* product, unsigned* failure, unsigned* maximum,
                    gpu::Queue queue) {
    if (!rows || rows > max_chunk_rows || !x || !scratch || !gate_core || !up_core || !product || !failure || !maximum ||
        !exl3_gelu_pair_supported(gate, up, down))
        throw std::invalid_argument("EXL3 fused gate/up geometry");
    const auto q = static_cast<hipStream_t>(queue);
    const auto g = gate.weights(), u = up.weights();
    // gate and up scale the input differently: one transform each, one scratch in stream order.
    exl3::device_input_panels(g, x, rows, scratch, failure, maximum, queue);
    panel_gemm(gate, static_cast<const _Float16*>(scratch), rows, gate_core, failure, q, direct_mode());
    exl3::device_input_panels(u, x, rows, scratch, failure, maximum + 1, queue);
    panel_gemm(up, static_cast<const _Float16*>(scratch), rows, up_core, failure, q, direct_mode());
    exl3::device_ffn_middle_panels(g, u, down, gate_core, up_core, rows, product, failure, maximum + 2, queue);
}
} // namespace strix::gemma4::prefill
