#include "a8_linear.hpp"
#include "prefill.hpp"
#include "gemma4/kernels/a8.hpp"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace strix::gemma4::prefill {
namespace {
using Int4 = int __attribute__((ext_vector_type(4)));
using Int8 = int __attribute__((ext_vector_type(8)));
constexpr unsigned group_width = a8::group_width;
// int32 accumulators start at the bits of 1.5 * 2^23: adding |I| < 2^22 keeps the
// value in [2^23, 2^24), where the float with those bits is exactly 12582912 + I.
constexpr int magic = 0x4B400000;
constexpr float magic_value = 12582912.0F;

void checked() {
    if (const auto e = hipGetLastError(); e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
unsigned padded_rows(unsigned rows) { return (rows + 127U) / 128U * 128U; }
// Kernel form: 0 = unstaged control, 1 = staged 32x32, 2 = staged 32x64, 3 = staged 64x64, 4 = 32x64 with
// two stages of loads in flight; wide (8/16-wave) forms 5 = 64x64 (default, measured fastest: 1,768 vs 1,633 tok/s
// for 32x64 at 2K), 6 = 32x128, 7 = 64x128, 9 = 128x64 (4-group stages), 8 = 64x64 (2-group stages). All give the
// same bytes.
// Startup default from STRIX_G4_PREFILL_A8_KERNEL (5 when unset); set_a8_kernel overrides it (benchmarks/tests).
unsigned& a8_kernel_choice() {
    static unsigned choice = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_A8_KERNEL");
        return p && *p >= '0' && *p <= '9' ? unsigned(*p - '0') : 5U;
    }();
    return choice;
}
unsigned a8_kernel() { return a8_kernel_choice(); }

__host__ __device__ inline unsigned reverse4(unsigned i) { return ((i & 1U) << 3U) | ((i & 2U) << 1U) | ((i & 4U) >> 1U) | ((i & 8U) >> 3U); }
// Groups of slot p: p, p+16, ... below `groups`.
__host__ __device__ inline unsigned slot_groups(unsigned p, unsigned groups) { return p < groups ? (groups - p + 15U) / 16U : 0U; }
// Slot-major storage: the groups are stored slot by slot in visit order (0, 8, 4, 12, 2, ...), each
// slot's groups ascending, so a slot's accumulation reads consecutive groups. Returns the stored
// position of logical group g.
__host__ __device__ inline unsigned slot_position(unsigned g, unsigned groups) {
    const unsigned visit = reverse4(g % 16U);
    unsigned position = g / 16U;
    for (unsigned s = 0; s < visit; ++s) position += slot_groups(reverse4(s), groups);
    return position;
}

// One thread per (output, group): the group's 32 weight integers and its FP32 pair, stored slot-major.
template <WeightFormat Format>
__global__ void prepare_a8(LinearView w, unsigned outputs16, std::int8_t* panels, float2* scales) {
    const unsigned groups = w.inputs / group_width;
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= std::size_t(outputs16) * groups) return;
    const auto o = unsigned(index / groups), g = unsigned(index % groups);
    const auto slot = std::size_t(o) * groups + slot_position(g, groups);
    auto* out = panels + slot * group_width;
    if (o >= w.outputs) {
        for (unsigned i = 0; i < group_width; ++i) out[i] = 0;
        scales[slot] = {0.0F, 0.0F};
        return;
    }
    const auto* row = w.data + std::size_t(o) * w.row_bytes;
    if constexpr (Format == WeightFormat::q4_k) {
        const auto* b = row + std::size_t(g / 8U) * 144U;
        const auto j = g % 8U;
        int sc = 0, m = 0;
        a8::q4_k_factors(b, j, sc, m);
        for (unsigned i = 0; i < group_width; ++i) out[i] = std::int8_t((b[16U + (j / 2U) * 32U + i] >> ((j & 1U) * 4U)) & 15U);
        // FP16 values carry at most 11 significant bits and sc, m < 64: both products are exact.
        scales[slot] = {fp16_value(load16(b)) * float(sc), fp16_value(load16(b + 2)) * float(m)};
    } else if constexpr (Format == WeightFormat::q6_k) {
        const auto* b = row + std::size_t(g / 8U) * 210U;
        const auto j = g % 8U, half = j / 4U, quarter = j % 4U;
        for (unsigned i = 0; i < group_width; ++i) {
            const auto low = b[half * 64U + i + (quarter & 1U) * 32U];
            const auto high = b[128U + half * 32U + i];
            out[i] = std::int8_t((((low >> (quarter >= 2U ? 4U : 0U)) & 15) | (((high >> (2U * quarter)) & 3) << 4)) - 32);
        }
        const unsigned s = unsigned(b[192U + 2U * j]) | (unsigned(b[193U + 2U * j]) << 8U);
        scales[slot] = {fp16_value(load16(b + 208)), bit_cast<float>(s)};
    } else {
        const auto* b = row + std::size_t(g) * 34U;
        for (unsigned i = 0; i < group_width; ++i) out[i] = std::int8_t(b[2U + i]);
        scales[slot] = {fp16_value(load16(b)), 0.0F};
    }
}

// One wave per (row, group); rows past `rows` up to the padded count get codes 0 and {1, 0}.
// Same operations as a8::quantize_group (exact maximum, IEEE division, rint, clamp, int sum).
// Product = true: the value is the decode target's gelu_tanh(gate) * up (only its codes are kept; the down linear
// is its only consumer).
template <bool Product = false>
__global__ void __launch_bounds__(256) quantize_a8(const float* x, unsigned rows, unsigned padded, unsigned width,
                                                   std::int8_t* codes, float2* meta, unsigned* failure, const float* gate = nullptr,
                                                   const float* up = nullptr) {
    const unsigned groups = width / group_width;
    const unsigned item = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    if (item >= padded * groups) return;
    const unsigned row = item / groups, g = item % groups;
    const auto at = std::size_t(row) * width + g * group_width + lane;
    float v = 0.0F;
    if constexpr (Product) {
        if (row < rows) v = gelu_tanh(gate[at]) * up[at];
    } else if (row < rows) v = x[at];
    if (!__builtin_isfinite(v)) atomicOr(failure, 1U);
    float amax = bit_cast<float>(bit_cast<unsigned>(v) & 0x7fffffffU);
    for (int d = 16; d; d >>= 1) {
        const float other = __shfl_xor(amax, d, 32);
        amax = other > amax ? other : amax;
    }
    const float delta = amax / 127.0F;
    int code = 0;
    if (delta != 0.0F) {
        float c = __builtin_rintf(v / delta);
        c = c > 127.0F ? 127.0F : (c < -127.0F ? -127.0F : c);
        code = int(c);
    }
    int sum = code;
    for (int d = 16; d; d >>= 1) sum += __shfl_xor(sum, d, 32);
    const auto slot = std::size_t(row) * groups + slot_position(g, groups);
    codes[slot * group_width + lane] = std::int8_t(code);
    if (!lane) meta[slot] = {delta != 0.0F ? delta : 1.0F, delta != 0.0F ? float(sum) : 0.0F};
}

// Each wave: TM row tiles x 16 outputs; a workgroup of 4 waves covers 16*TM rows x 64 outputs.
// The 16 slots run in bit-reversed order (0, 8, 4, 12, 2, ...): slot p accumulates groups p, p+16, ...
// ascending from +0, and a four-level carry chain applies the 8/4/2/1 tree as soon as both halves of a
// node exist (left operand = lower slots), so at most four finished partials are live.
template <WeightFormat Format, unsigned TM>
__global__ void __launch_bounds__(128) a8_gemm(const std::int8_t* panels, const float2* scales, const std::int8_t* codes,
                                               const float2* meta, unsigned rows, unsigned outputs, unsigned width, float* out) {
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, i = lane % 16U, h = lane / 16U;
    const unsigned groups = width / group_width;
    const unsigned m0 = blockIdx.y * 16U * TM, n = (blockIdx.x * 4U + wave) * 16U + i;
    const auto* wrow = panels + std::size_t(n) * width;
    const auto* srow = scales + std::size_t(n) * groups;
    float level[4][TM][8];
    float acc[TM][8];
    for (unsigned s = 0; s < a8::slots; ++s) {
        const unsigned p = reverse4(s);
#pragma unroll
        for (unsigned t = 0; t < TM; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[t][e] = 0.0F;
        const unsigned first = slot_position(p, groups);
        for (unsigned g = first; g < first + slot_groups(p, groups); ++g) {
            const auto* wg = reinterpret_cast<const Int4*>(wrow + std::size_t(g) * group_width);
            const Int4 b0 = wg[0], b1 = wg[1];
            const float2 ws = srow[g];
#pragma unroll
            for (unsigned t = 0; t < TM; ++t) {
                const auto* ag = reinterpret_cast<const Int4*>(codes + std::size_t(m0 + 16U * t + i) * width + std::size_t(g) * group_width);
                const Int4 a0 = ag[0], a1 = ag[1];
                float value[8];
                if constexpr (Format == WeightFormat::q6_k) {
                    const Int8 zero{};
                    const Int8 c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, zero, false);
                    const Int8 c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, zero, false);
                    const unsigned s01 = bit_cast<unsigned>(ws.y);
                    const int s0 = int(std::int8_t(s01 & 255U)), s1 = int(std::int8_t(s01 >> 8U));
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) value[e] = ws.x * float(s0 * c0[e] + s1 * c1[e]);
                } else {
                    const Int8 start{magic, magic, magic, magic, magic, magic, magic, magic};
                    constexpr bool sign = Format == WeightFormat::q8_0;
                    Int8 c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, sign, b0, start, false);
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, sign, b1, c, false);
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) value[e] = ws.x * (bit_cast<float>(c[e]) - magic_value);
                }
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    const float2 a = meta[std::size_t(m0 + 16U * t + 2U * e + h) * groups + g];
                    const float term = Format == WeightFormat::q4_k ? a.x * (value[e] - ws.y * a.y) : a.x * value[e];
                    acc[t][e] = acc[t][e] + term;
                }
            }
        }
        bool placed = false;
#pragma unroll
        for (unsigned l = 0; l < 4U; ++l)
            if (!placed) {
                if ((s >> l) & 1U) {
#pragma unroll
                    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) acc[t][e] = level[l][t][e] + acc[t][e];
                } else {
#pragma unroll
                    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
                        for (unsigned e = 0; e < 8U; ++e) level[l][t][e] = acc[t][e];
                    placed = true;
                }
            }
    }
    if (n >= outputs) return;
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned r = m0 + 16U * t + 2U * e + h;
            if (r < rows) out[std::size_t(r) * outputs + n] = acc[t][e];
        }
}
// Staged form: a workgroup of 4 waves computes 32*TM rows x 32*TN outputs, each wave a 16*TM x 16*TN tile.
// Slot-major storage makes the K order one contiguous stream; Stage groups (codes of 32 rows and
// 32*TN outputs plus their FP32 pairs) are staged in LDS per barrier, double-buffered. Same group
// terms and bit-reversed post-order slot fold as a8_gemm, so the same bytes.
// Ahead = 2 keeps two stages of global loads in flight (two register sets), Ahead = 1 one.
template <WeightFormat Format, unsigned Stage, unsigned TM, unsigned TN, unsigned Ahead = 1>
__global__ void __launch_bounds__(128) a8_staged(const std::int8_t* panels, const float2* scales, const std::int8_t* codes,
                                                 const float2* meta, unsigned rows, unsigned outputs, unsigned width, float* out) {
    static_assert(Stage % 2U == 0U && Stage <= 4U && (TM == 1U || TM == 2U) && (TN == 1U || TN == 2U) && (Ahead == 1U || Ahead == 2U));
    constexpr unsigned R = 32U * TM, N = 32U * TN;
    struct Buffer {
        Int4 a[Stage][2][R], b[Stage][2][N];
        alignas(16) float2 am[Stage][R];
        float2 bs[Stage][N];
    };
    __shared__ Buffer lds[2];
    const unsigned tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U, i = lane % 16U, h = lane / 16U;
    const unsigned groups = width / group_width, stages = groups / Stage;
    const unsigned m0 = blockIdx.y * R, n0 = blockIdx.x * N, rt = wave / 2U, ct = wave % 2U;
    // Loader: per group pair, 32 lines (rows or outputs) x 4 parts of 16 bytes; thread = (line, part).
    const unsigned line = tid / 4U, part = tid % 4U, mline = tid % 32U, mq = tid / 32U;
    const std::int8_t* arow[TM];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t) arow[t] = codes + std::size_t(m0 + 32U * t + line) * width;
    const std::int8_t* brow[TN];
#pragma unroll
    for (unsigned u = 0; u < TN; ++u) brow[u] = panels + std::size_t(n0 + 32U * u + line) * width;
    Int4 ra[Ahead][TM][Stage / 2U], rb[Ahead][TN][Stage / 2U];
    float2 rm[Ahead][TM]{}, rs[Ahead][TN]{};
    const auto fetch = [&](unsigned st) {
        const unsigned k = Ahead == 2U ? st & 1U : 0U;
#pragma unroll
        for (unsigned q = 0; q < Stage / 2U; ++q) {
            const auto offset = std::size_t(st * Stage + 2U * q) * group_width + 16U * part;
#pragma unroll
            for (unsigned t = 0; t < TM; ++t) ra[k][t][q] = *reinterpret_cast<const Int4*>(arow[t] + offset);
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) rb[k][u][q] = *reinterpret_cast<const Int4*>(brow[u] + offset);
        }
        if (mq < Stage) {
#pragma unroll
            for (unsigned t = 0; t < TM; ++t) rm[k][t] = meta[std::size_t(m0 + 32U * t + mline) * groups + st * Stage + mq];
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) rs[k][u] = scales[std::size_t(n0 + 32U * u + mline) * groups + st * Stage + mq];
        }
    };
    const auto store = [&](Buffer& b, unsigned st) {
        const unsigned k = Ahead == 2U ? st & 1U : 0U;
#pragma unroll
        for (unsigned q = 0; q < Stage / 2U; ++q) {
#pragma unroll
            for (unsigned t = 0; t < TM; ++t) b.a[2U * q + part / 2U][part % 2U][32U * t + line] = ra[k][t][q];
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) b.b[2U * q + part / 2U][part % 2U][32U * u + line] = rb[k][u][q];
        }
        if (mq < Stage) {
#pragma unroll
            // Row r of a 16-row tile is stored at (r % 2) * 8 + r / 2: a lane's eight accumulator rows
            // (2e + h) become eight consecutive pairs, read with four 16-byte loads.
            for (unsigned t = 0; t < TM; ++t) b.am[mq][32U * t + (mline & 16U) + ((mline & 1U) << 3U) + ((mline & 15U) >> 1U)] = rm[k][t];
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) b.bs[mq][32U * u + mline] = rs[k][u];
        }
    };
    constexpr unsigned T = TM * TN;
    // With four tiles per wave the top stack level (written and read once per output) lives in LDS:
    // registers hold levels 0-2 and the leaf, which keeps the 64x64 form inside 256 VGPRs.
    constexpr unsigned Held = T >= 4U ? 3U : 4U;
    __shared__ float upper[T >= 4U ? 4U : 1U][4U - Held + (T >= 4U ? 0U : 1U)][T][8][32];
    float level[Held][T][8], acc[T][8];
#pragma unroll
    for (unsigned u = 0; u < T; ++u)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) acc[u][e] = 0.0F;
    unsigned s = 0, left = slot_groups(0, groups);
    // Folds finished leaves (slot visit s) into the stack; empty slots fold +0 leaves.
    const auto finish = [&]() {
        while (left == 0U && s < a8::slots) {
            bool placed = false;
#pragma unroll
            for (unsigned l = 0; l < 4U; ++l)
                if (!placed) {
                    if ((s >> l) & 1U) {
#pragma unroll
                        for (unsigned u = 0; u < T; ++u)
#pragma unroll
                            for (unsigned e = 0; e < 8U; ++e) {
                                const float left_value = l < Held ? level[l < Held ? l : 0][u][e] : upper[wave][l >= Held ? l - Held : 0][u][e][lane];
                                acc[u][e] = left_value + acc[u][e];
                            }
                    } else {
#pragma unroll
                        for (unsigned u = 0; u < T; ++u)
#pragma unroll
                            for (unsigned e = 0; e < 8U; ++e) {
                                if (l < Held) level[l < Held ? l : 0][u][e] = acc[u][e];
                                else upper[wave][l >= Held ? l - Held : 0][u][e][lane] = acc[u][e];
                            }
                        placed = true;
                    }
                }
            if (++s < a8::slots) {
                left = slot_groups(reverse4(s), groups);
#pragma unroll
                for (unsigned u = 0; u < T; ++u)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) acc[u][e] = 0.0F;
            }
        }
    };
    finish();
    fetch(0);
    store(lds[0], 0);
    if (Ahead == 2U && stages > 1U) fetch(1);
    __syncthreads();
    for (unsigned st = 0; st < stages; ++st) {
        if (st + Ahead < stages) fetch(st + Ahead);
        const Buffer& b = lds[st & 1U];
#pragma unroll
        for (unsigned q = 0; q < Stage; ++q) {
#pragma unroll
            for (unsigned t = 0; t < TM; ++t) {
            const unsigned row = 16U * (TM * rt + t);
            const Int4 a0 = b.a[q][0][row + i], a1 = b.a[q][1][row + i];
            float2 am[8];
            const auto* pairs = reinterpret_cast<const float4*>(&b.am[q][row + 8U * h]);
#pragma unroll
            for (unsigned e = 0; e < 4U; ++e) {
                const float4 v = pairs[e];
                am[2U * e] = {v.x, v.y};
                am[2U * e + 1U] = {v.z, v.w};
            }
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) {
                const unsigned col = 16U * (TN * ct + u) + i;
                const Int4 b0 = b.b[q][0][col], b1 = b.b[q][1][col];
                const float2 ws = b.bs[q][col];
                float value[8];
                if constexpr (Format == WeightFormat::q6_k) {
                    const Int8 zero{};
                    const Int8 c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, zero, false);
                    const Int8 c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, zero, false);
                    const unsigned s01 = bit_cast<unsigned>(ws.y);
                    const int s0 = int(std::int8_t(s01 & 255U)), s1 = int(std::int8_t(s01 >> 8U));
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) value[e] = ws.x * float(s0 * c0[e] + s1 * c1[e]);
                } else {
                    const Int8 start{magic, magic, magic, magic, magic, magic, magic, magic};
                    constexpr bool sign = Format == WeightFormat::q8_0;
                    Int8 c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, sign, b0, start, false);
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, sign, b1, c, false);
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) value[e] = ws.x * (bit_cast<float>(c[e]) - magic_value);
                }
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    const float term = Format == WeightFormat::q4_k ? am[e].x * (value[e] - ws.y * am[e].y) : am[e].x * value[e];
                    acc[t * TN + u][e] = acc[t * TN + u][e] + term;
                }
            }
            }
            --left;
            finish();
        }
        if (st + 1U < stages) store(lds[(st + 1U) & 1U], st + 1U);
        __syncthreads();
    }
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
    for (unsigned u = 0; u < TN; ++u) {
        const unsigned n = n0 + 16U * (TN * ct + u) + i;
        if (n < outputs)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = m0 + 16U * (TM * rt + t) + 2U * e + h;
                if (r < rows) out[std::size_t(r) * outputs + n] = acc[t * TN + u][e];
            }
    }
}
// Wide form: WR x WC waves, each a 16 x 32 tile (two output tiles), so a workgroup covers 16*WR rows x 32*WC
// outputs and every staged byte feeds more WMMAs than in a8_staged (the 32x64 form measured 36 % of its wave time
// waiting for loads). Stage groups of codes and FP32 pairs per barrier, double-buffered; same group terms and slot
// fold as a8_gemm, so the same bytes.
template <WeightFormat Format, unsigned Stage, unsigned WR, unsigned WC>
__global__ void __launch_bounds__(32 * WR * WC) a8_tiled(const std::int8_t* panels, const float2* scales, const std::int8_t* codes,
                                                       const float2* meta, unsigned rows, unsigned outputs, unsigned width, float* out) {
    constexpr unsigned TN = 2U, R = 16U * WR, N = 32U * WC, Threads = 32U * WR * WC;
    static_assert(Stage % 2U == 0U && Stage <= 4U);
    struct Buffer {
        Int4 a[Stage][2][R], b[Stage][2][N];
        alignas(16) float2 am[Stage][R];
        float2 bs[Stage][N];
    };
    __shared__ Buffer lds[2];
    const unsigned tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U, i = lane % 16U, h = lane / 16U;
    const unsigned groups = width / group_width, stages = groups / Stage;
    const unsigned m0 = blockIdx.y * R, n0 = blockIdx.x * N, rt = wave / WC, ct = wave % WC;
    // Loader: per group pair a line (row or output) is 64 bytes = 4 chunks of 16; chunk c of the R + N lines.
    constexpr unsigned Chunks = (R + N) * 4U * (Stage / 2U), Per = (Chunks + Threads - 1U) / Threads;
    constexpr unsigned MetaItems = (R + N) * Stage, MetaPer = (MetaItems + Threads - 1U) / Threads;
    Int4 rc[Per];
    float2 rp[MetaPer];
    const auto fetch = [&](unsigned st) {
#pragma unroll
        for (unsigned k = 0; k < Per; ++k) {
            const unsigned c = tid + Threads * k;
            if (Chunks % Threads == 0U || c < Chunks) {
                const unsigned q = c / ((R + N) * 4U), line = (c / 4U) % (R + N), part = c % 4U;
                const auto offset = std::size_t(st * Stage + 2U * q) * group_width + 16U * part;
                const std::int8_t* base = line < R ? codes + std::size_t(m0 + line) * width : panels + std::size_t(n0 + line - R) * width;
                rc[k] = *reinterpret_cast<const Int4*>(base + offset);
            }
        }
#pragma unroll
        for (unsigned k = 0; k < MetaPer; ++k) {
            const unsigned c = tid + Threads * k;
            if (MetaItems % Threads == 0U || c < MetaItems) {
                const unsigned g = c / (R + N), line = c % (R + N);
                rp[k] = line < R ? meta[std::size_t(m0 + line) * groups + st * Stage + g] : scales[std::size_t(n0 + line - R) * groups + st * Stage + g];
            }
        }
    };
    const auto store = [&](Buffer& b) {
#pragma unroll
        for (unsigned k = 0; k < Per; ++k) {
            const unsigned c = tid + Threads * k;
            if (Chunks % Threads == 0U || c < Chunks) {
                const unsigned q = c / ((R + N) * 4U), line = (c / 4U) % (R + N), part = c % 4U;
                const unsigned g = 2U * q + part / 2U, half = part % 2U;
                if (line < R) b.a[g][half][line] = rc[k];
                else b.b[g][half][line - R] = rc[k];
            }
        }
#pragma unroll
        for (unsigned k = 0; k < MetaPer; ++k) {
            const unsigned c = tid + Threads * k;
            if (MetaItems % Threads == 0U || c < MetaItems) {
                const unsigned g = c / (R + N), line = c % (R + N);
                // Row r of a 16-row tile at (r % 2) * 8 + r / 2: a lane's eight accumulator rows are consecutive.
                if (line < R) b.am[g][(line & ~15U) + ((line & 1U) << 3U) + ((line & 15U) >> 1U)] = rp[k];
                else b.bs[g][line - R] = rp[k];
            }
        }
    };
    float level[4][TN][8], acc[TN][8];
#pragma unroll
    for (unsigned u = 0; u < TN; ++u)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) acc[u][e] = 0.0F;
    unsigned s = 0, left = slot_groups(0, groups);
    const auto finish = [&]() {
        while (left == 0U && s < a8::slots) {
            bool placed = false;
#pragma unroll
            for (unsigned l = 0; l < 4U; ++l)
                if (!placed) {
                    if ((s >> l) & 1U) {
#pragma unroll
                        for (unsigned u = 0; u < TN; ++u)
#pragma unroll
                            for (unsigned e = 0; e < 8U; ++e) acc[u][e] = level[l][u][e] + acc[u][e];
                    } else {
#pragma unroll
                        for (unsigned u = 0; u < TN; ++u)
#pragma unroll
                            for (unsigned e = 0; e < 8U; ++e) level[l][u][e] = acc[u][e];
                        placed = true;
                    }
                }
            if (++s < a8::slots) {
                left = slot_groups(reverse4(s), groups);
#pragma unroll
                for (unsigned u = 0; u < TN; ++u)
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) acc[u][e] = 0.0F;
            }
        }
    };
    finish();
    fetch(0);
    store(lds[0]);
    __syncthreads();
    for (unsigned st = 0; st < stages; ++st) {
        if (st + 1U < stages) fetch(st + 1U);
        const Buffer& b = lds[st & 1U];
#pragma unroll
        for (unsigned q = 0; q < Stage; ++q) {
            const unsigned row = 16U * rt;
            const Int4 a0 = b.a[q][0][row + i], a1 = b.a[q][1][row + i];
            float2 am[8];
            const auto* pairs = reinterpret_cast<const float4*>(&b.am[q][row + 8U * h]);
#pragma unroll
            for (unsigned e = 0; e < 4U; ++e) {
                const float4 v = pairs[e];
                am[2U * e] = {v.x, v.y};
                am[2U * e + 1U] = {v.z, v.w};
            }
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) {
                const unsigned col = 16U * (TN * ct + u) + i;
                const Int4 b0 = b.b[q][0][col], b1 = b.b[q][1][col];
                const float2 ws = b.bs[q][col];
                float value[8];
                if constexpr (Format == WeightFormat::q6_k) {
                    const Int8 zero{};
                    const Int8 c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, zero, false);
                    const Int8 c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, zero, false);
                    const unsigned s01 = bit_cast<unsigned>(ws.y);
                    const int s0 = int(std::int8_t(s01 & 255U)), s1 = int(std::int8_t(s01 >> 8U));
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) value[e] = ws.x * float(s0 * c0[e] + s1 * c1[e]);
                } else {
                    const Int8 start{magic, magic, magic, magic, magic, magic, magic, magic};
                    constexpr bool sign = Format == WeightFormat::q8_0;
                    Int8 c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, sign, b0, start, false);
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, sign, b1, c, false);
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) value[e] = ws.x * (bit_cast<float>(c[e]) - magic_value);
                }
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    const float term = Format == WeightFormat::q4_k ? am[e].x * (value[e] - ws.y * am[e].y) : am[e].x * value[e];
                    acc[u][e] = acc[u][e] + term;
                }
            }
            --left;
            finish();
        }
        if (st + 1U < stages) store(lds[(st + 1U) & 1U]);
        __syncthreads();
    }
#pragma unroll
    for (unsigned u = 0; u < TN; ++u) {
        const unsigned n = n0 + 16U * (TN * ct + u) + i;
        if (n < outputs)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned r = m0 + 16U * rt + 2U * e + h;
                if (r < rows) out[std::size_t(r) * outputs + n] = acc[u][e];
            }
    }
}
} // namespace

void set_a8_kernel(unsigned form) {
    if (form > 9) throw std::invalid_argument("a8 kernel form");
    a8_kernel_choice() = form;
}
bool a8_prefill_supported(LinearView w) {
    return a8::quantized(w.format) && w.data && w.outputs && w.inputs && w.inputs % 256U == 0 && w.inputs <= 16384 &&
           w.row_bytes >= packed_row_bytes(w.format, w.inputs);
}
A8Prepared::A8Prepared(LinearView w, gpu::Queue queue) : weights_(w) {
    if (!a8_prefill_supported(w)) throw std::invalid_argument("g4-a8-v1 prefill linear geometry");
    const unsigned outputs16 = (w.outputs + 127U) / 128U * 128U, groups = w.inputs / group_width;
    panels_.resize(std::size_t(outputs16) * w.inputs);
    scales_.resize(std::size_t(outputs16) * groups * sizeof(float2));
    const auto q = static_cast<hipStream_t>(queue);
    const auto items = std::size_t(outputs16) * groups;
    const dim3 grid(unsigned((items + 255) / 256));
    if (w.format == WeightFormat::q4_k)
        hipLaunchKernelGGL(prepare_a8<WeightFormat::q4_k>, grid, dim3(256), 0, q, w, outputs16, panels_.as<std::int8_t>(), scales_.as<float2>());
    else if (w.format == WeightFormat::q6_k)
        hipLaunchKernelGGL(prepare_a8<WeightFormat::q6_k>, grid, dim3(256), 0, q, w, outputs16, panels_.as<std::int8_t>(), scales_.as<float2>());
    else
        hipLaunchKernelGGL(prepare_a8<WeightFormat::q8_0>, grid, dim3(256), 0, q, w, outputs16, panels_.as<std::int8_t>(), scales_.as<float2>());
    checked();
    gpu::synchronize(queue);
}
std::size_t a8_scratch_bytes(unsigned rows, unsigned inputs) {
    // codes [padded][inputs] bytes, then {delta, float(S)} [padded][inputs / 32]
    return std::size_t(padded_rows(rows)) * (inputs + std::size_t(inputs / group_width) * sizeof(float2));
}
void a8_linear(const A8Prepared& prepared, const float* x, unsigned rows, float* out, void* scratch, std::uint32_t* failure,
               gpu::Queue queue, bool quantize) {
    const auto w = prepared.weights();
    if (!rows || rows > max_chunk_rows || !x || !out || !scratch || !failure) throw std::invalid_argument("g4-a8-v1 prefill linear arguments");
    const unsigned padded = padded_rows(rows), groups = w.inputs / group_width;
    auto* codes = static_cast<std::int8_t*>(scratch);
    auto* meta = reinterpret_cast<float2*>(codes + std::size_t(padded) * w.inputs);
    const auto q = static_cast<hipStream_t>(queue);
    if (quantize) {
        hipLaunchKernelGGL(quantize_a8<false>, dim3((padded * groups + 7U) / 8U), dim3(256), 0, q, x, rows, padded, w.inputs, codes, meta,
                           failure, nullptr, nullptr);
        checked();
    }
    if (const unsigned form = a8_kernel(); form >= 5U) {
        const auto* p = prepared.panels().as<std::int8_t>();
        const auto* sc = prepared.scales().as<float2>();
#define G4_A8_TILED(F, S, WR, WC) hipLaunchKernelGGL((a8_tiled<WeightFormat::F, S, WR, WC>), \
            dim3((w.outputs + 32U * WC - 1U) / (32U * WC), (padded + 16U * WR - 1U) / (16U * WR)), dim3(32 * WR * WC), 0, q, p, sc, codes, meta, rows, \
            w.outputs, w.inputs, out)
#define G4_A8_WIDE(F) do { if (form == 5) G4_A8_TILED(F, 4, 4, 2); else if (form == 6) G4_A8_TILED(F, 4, 2, 4); else if (form == 7) G4_A8_TILED(F, 4, 4, 4); \
                           else if (form == 8) G4_A8_TILED(F, 2, 4, 2); else G4_A8_TILED(F, 4, 8, 2); } while (false)
        if (w.format == WeightFormat::q4_k) G4_A8_WIDE(q4_k);
        else if (w.format == WeightFormat::q6_k) G4_A8_WIDE(q6_k);
        else G4_A8_WIDE(q8_0);
#undef G4_A8_WIDE
#undef G4_A8_TILED
        checked();
        return;
    }
    if (const unsigned tn = a8_kernel(); tn) {
        const auto* p = prepared.panels().as<std::int8_t>();
        const auto* sc = prepared.scales().as<float2>();
#define G4_A8_STAGED(F, TM, TN, A) hipLaunchKernelGGL((a8_staged<WeightFormat::F, 4, TM, TN, A>), \
            dim3((w.outputs + 32U * TN - 1U) / (32U * TN), (padded + 32U * TM - 1U) / (32U * TM)), dim3(128), 0, q, p, sc, codes, meta, rows, w.outputs, w.inputs, out)
#define G4_A8_FORMS(F) do { if (tn == 4) G4_A8_STAGED(F, 1, 2, 2); else if (tn == 3) G4_A8_STAGED(F, 2, 2, 1); else if (tn == 2) G4_A8_STAGED(F, 1, 2, 1); \
                               else G4_A8_STAGED(F, 1, 1, 1); } while (false)
        if (w.format == WeightFormat::q4_k) G4_A8_FORMS(q4_k);
        else if (w.format == WeightFormat::q6_k) G4_A8_FORMS(q6_k);
        else G4_A8_FORMS(q8_0);
#undef G4_A8_FORMS
#undef G4_A8_STAGED
        checked();
        return;
    }
    const dim3 grid((w.outputs + 63U) / 64U, padded / 32U);
    const auto* panels = prepared.panels().as<std::int8_t>();
    const auto* scales = prepared.scales().as<float2>();
    if (w.format == WeightFormat::q4_k)
        hipLaunchKernelGGL((a8_gemm<WeightFormat::q4_k, 2>), grid, dim3(128), 0, q, panels, scales, codes, meta, rows, w.outputs, w.inputs, out);
    else if (w.format == WeightFormat::q6_k)
        hipLaunchKernelGGL((a8_gemm<WeightFormat::q6_k, 2>), grid, dim3(128), 0, q, panels, scales, codes, meta, rows, w.outputs, w.inputs, out);
    else
        hipLaunchKernelGGL((a8_gemm<WeightFormat::q8_0, 2>), grid, dim3(128), 0, q, panels, scales, codes, meta, rows, w.outputs, w.inputs, out);
    checked();
}

void a8_product_linear(const A8Prepared& prepared, float* gate, const float* up, unsigned rows, float* out, void* scratch,
                       std::uint32_t* failure, gpu::Queue queue) {
    const auto w = prepared.weights();
    if (!rows || rows > max_chunk_rows || !gate || !up || !out || !scratch || !failure) throw std::invalid_argument("g4-a8-v1 product linear arguments");
    const unsigned padded = padded_rows(rows), groups = w.inputs / group_width;
    auto* codes = static_cast<std::int8_t*>(scratch);
    auto* meta = reinterpret_cast<float2*>(codes + std::size_t(padded) * w.inputs);
    hipLaunchKernelGGL(quantize_a8<true>, dim3((padded * groups + 7U) / 8U), dim3(256), 0, static_cast<hipStream_t>(queue), nullptr, rows, padded,
                       w.inputs, codes, meta, failure, gate, up);
    checked();
    a8_linear(prepared, gate, rows, out, scratch, failure, queue, false);
}

} // namespace strix::gemma4::prefill
