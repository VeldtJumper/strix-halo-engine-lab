// CUDA Qwen prefill, family "hc": hyper-connection mixes, elementwise steps, MoE router and LM head, bit-identical to the
// HIP A8-profile tiled prefill on gfx1151 (engine bf60d9d2). See include/strix/cuda_qwen_hc.hpp for the API.
//
// Every floating-point operation below is one explicit IEEE operation (__fmaf_rn / __fadd_rn / __fmul_rn / __fdiv_rn)
// mirroring the HIP instruction, as read in the gfx1151 ISA (disassembly of each translation unit; ISA line numbers are
// relative to the kernel symbol's label line):
//   [N] rows_prepare_hyper_norm_kernel<u16,HyperTileRows> (hip_hyper_mix_launcher.s:398955): `sum += v*v` is
//       v_dual_fmac_f32 v3, v5, v5 (+66); LDS tree v_add_f32 strides 128..1 (+86..+185); `red/2560` is the
//       v_div_scale/fmas/fixup sequence (correctly rounded, +199..+216); `+ eps` v_add_f32 (+217); rsqrtf = the
//       2^24 / v_rsq_f32 / 2^12 sequence (+219..+226, amd_rsqrtf); `x*inverse*w` two v_mul_f32 (+257, +259).
//   [P] wide2_splitk_hyper_projection_kernel<8,true,false> (hip_hyper_mix_launcher.s:312631): chains are the
//       generated v_dual_fmac_f32 of hip_hc_wide2_vopd.inc (fma(x, c, acc), j = 0..3 per 128-feature chunk); fold =
//       DPP row_xmask 1/2/4/8 + v_permlanex16 then v_add_f32 (+822..+1025); epilogue: total/streams correctly rounded
//       (+1124..+1139, +2540..+2547), expf(-scaled) as amd_expf with negated constants (+2548..+2560), 1 + e
//       (v_add_f32), 1/(1+e) correctly rounded (+2561..+2574), low = scaled * sig v_mul_f32 (+2576), injection =
//       sig + sig v_add_f32 (+1177).
//   [O] wide2_hyper_output_kernel<u16,32,true,false> (hip_hyper_mix_launcher.s:340714): gate chains = the generated
//       v_dual_fmac_f32 of hip_hc_wide2_output_vopd.inc (fma(low, c, g), k ascending); the stream mix is CONTRACTED:
//       v_fma_f32 v3, n0, sig0, 0 then v_fmac_f32 for s = 1, 2, 3 (+838..+848); `/4` is v_mul_f32 0x3e800000 (+849);
//       BF16 RNE v_bfe/v_add3 (+851..+853).
//   [E] embedding_kernel<u16,q8_0> (hip_embedding_launcher.s:3165): v_fma_mix_f32 scale, code, -0 = the exact product.
//   [I] inject_rows_kernel<u16,ElementwiseTileRowsArgs> (hip_runtime_launchers.s:2602) and inject_kernel<u16> (:1560):
//       CONTRACTED, v_fmac_f32 hyper, weight, hidden. [C] combine_rows_kernel<u16,TileRows> (:3235): v_add_f32.
//   [S] rows_moe_route_select_kernel<wave,RouterTileRowsArgs> (hip_moe_route_launcher.s:107647): v_max_f32 (+86..+138),
//       expf(l - max) = amd_expf (+159..+175), adds (+178, +220), correctly rounded division (+236..+252), compare/
//       select top-k with ds_bpermute xor butterflies (+697..).
//   [L] prefetch8_wave_q8_lm_head_kernel<u16,8> (hip_lm_head_launcher.s:180867): coefficient = v_fma_mix_f32 scale,
//       code, -0 (exact product, +103..+123), leaf chain CONTRACTED v_fmac_f32 partial, coef, x (+114..+129),
//       __shfl_down tree adds (+146..).
#include "strix/cuda_qwen_hc.hpp"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::cuda::qwen_hc {
namespace {

namespace am = strix::cuda::amdmath;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("qwen_hc ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float bf16_to_float(std::uint16_t v) { return __uint_as_float(static_cast<std::uint32_t>(v) << 16U); }
// HIP store_activation (hip_hyper_mix_launcher.hip:202-214, hip_runtime_launchers.hip:118-128): RNE, no NaN case.
// NaN: gfx1151 arithmetic writes the canonical 0x7fc00000 for a generated NaN (-> BF16 0x7fc0), CUDA writes 0x7fffffff,
// which the rounding trick would wrap to 0x8000 (-0, a silently lost NaN). So a NaN is stored as HIP's canonical NaN
// (payloads of propagated input NaNs are not reproduced; plan rules 3-4).
__device__ __forceinline__ std::uint16_t float_to_bf16(float v) {
    if (v != v) return 0x7fc0U;
    const std::uint32_t b = __float_as_uint(v);
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
}
__device__ __forceinline__ float half_bits_to_float(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }
// sigmoid(x) = 1.0F / (1.0F + expf(-x)) (hip_hyper_mix_launcher.hip:399-404; ISA [P], [O]).
__device__ __forceinline__ float amd_sigmoid(const am::Tables& t, float x) {
    return __fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(t, -x)));
}

// ---------------------------------------------------------------------------------------------------------------------
// HC pack (replaces wide2_pack_down_kernel, hip_hyper_mix_launcher.hip:2887-2905: data movement only).
constexpr std::size_t kDownCodeBytes = std::size_t{kRank} * kWide;                  // int8
constexpr std::size_t kDownScaleBytes = std::size_t{kRank} * (kWide / 32U) * 4U;
constexpr std::size_t kUpCodeBytes = std::size_t{kRank / 4U} * kStreams * kHidden * 4U;
constexpr std::size_t kUpScaleBytes = std::size_t{kRank / 32U} * kStreams * kHidden * 4U;

__global__ void hc_pack_kernel(const unsigned char* __restrict__ down, const unsigned char* __restrict__ up, HcPacked p) {
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr unsigned kDownItems = kRank * (kWide / 32U), kUpItems = kWide * (kRank / 32U);
    if (id < kDownItems) {
        const unsigned o = id / (kWide / 32U), b = id % (kWide / 32U);
        const unsigned char* block = down + (static_cast<std::size_t>(o) * (kWide / 32U) + b) * 34U;
        p.down_scales[id] = half_bits_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8U)));
        std::uint32_t* dst = reinterpret_cast<std::uint32_t*>(p.down_codes + static_cast<std::size_t>(o) * kWide + b * 32U);
#pragma unroll
        for (unsigned w = 0; w < 8U; ++w)
            dst[w] = static_cast<std::uint32_t>(block[2 + 4 * w]) | (static_cast<std::uint32_t>(block[3 + 4 * w]) << 8U) |
                (static_cast<std::uint32_t>(block[4 + 4 * w]) << 16U) | (static_cast<std::uint32_t>(block[5 + 4 * w]) << 24U);
    } else if (id < kDownItems + kUpItems) {
        const unsigned u = id - kDownItems, row = u / (kRank / 32U), b = u % (kRank / 32U);
        const unsigned s = row / kHidden, f = row % kHidden;
        const unsigned char* block = up + (static_cast<std::size_t>(row) * (kRank / 32U) + b) * 34U;
        p.up_scales[(static_cast<std::size_t>(b) * kStreams + s) * kHidden + f] =
            half_bits_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8U)));
#pragma unroll
        for (unsigned w = 0; w < 8U; ++w)
            p.up_codes[(static_cast<std::size_t>(b * 8U + w) * kStreams + s) * kHidden + f] =
                static_cast<std::uint32_t>(block[2 + 4 * w]) | (static_cast<std::uint32_t>(block[3 + 4 * w]) << 8U) |
                (static_cast<std::uint32_t>(block[4 + 4 * w]) << 16U) | (static_cast<std::uint32_t>(block[5 + 4 * w]) << 24U);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// [N] rows_prepare_hyper_norm_kernel (hip_hyper_mix_launcher.hip:1757-1776): one block per (row, stream), 256 threads.
__global__ __launch_bounds__(256) void hc_norm_kernel(const std::uint16_t* __restrict__ input, const float* __restrict__ norm,
    float epsilon, float* __restrict__ normalized, am::Tables t) {
    const unsigned row = blockIdx.x, stream = blockIdx.y, lane = threadIdx.x;
    const std::uint16_t* in = input + static_cast<std::size_t>(row) * kWide + stream * kHidden;
    float* out = normalized + static_cast<std::size_t>(row) * kNormalizedPitch + stream * kHidden;
    __shared__ float reduction[256];
    __shared__ float inverse;
    float values[kHidden / 256U];
    float sum = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < kHidden / 256U; ++i) {  // features lane, lane + 256, ... ascending
        values[i] = bf16_to_float(in[lane + 256U * i]);
        sum = __fmaf_rn(values[i], values[i], sum);
    }
    reduction[lane] = sum;
    __syncthreads();
    // LDS tree :1768-1771: reduction[l] += reduction[l + stride], stride 128, 64, ..., 1, barrier after each level.
    for (unsigned stride = 128U; stride > 16U; stride >>= 1U) {
        if (lane < stride) reduction[lane] = __fadd_rn(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    if (lane < 32U) {  // strides 16..1 inside warp 0: the same pairs (l, l + stride) as the LDS levels
        float v = __fadd_rn(reduction[lane], reduction[lane + 16U]);
#pragma unroll
        for (unsigned stride = 8U; stride > 0U; stride >>= 1U) v = __fadd_rn(v, __shfl_down_sync(0xffffffffU, v, stride));
        if (lane == 0U) inverse = am::amd_rsqrtf(t, __fadd_rn(__fdiv_rn(v, static_cast<float>(kHidden)), epsilon));
    }
    __syncthreads();
    const float inv = inverse;
    const float* w = norm + stream * kHidden;
#pragma unroll
    for (unsigned i = 0; i < kHidden / 256U; ++i)
        out[lane + 256U * i] = __fmul_rn(__fmul_rn(values[i], inv), w[lane + 256U * i]);
}

// ---------------------------------------------------------------------------------------------------------------------
// [P] wide2_splitk_hyper_projection_kernel<8, true> (hip_hyper_mix_launcher.hip:2907-3050).
// Per (row, output) the HIP DAG is (window lane l = 0..31, tile T = 0..15, sub-chunk s = 0..4, j = 0..3):
//   leaf(T, l)  = fma chain from 0 over s, j ascending: acc = fma(x[f], c[f], acc), f = 640 T + 128 s + 4 l + j
//   tile(T)     = xor butterfly over l with distances 16, 8, 4, 2, 1 (wide2_lane_tree :2865-2876 on bit-reversed
//                 hardware lanes = window-lane distances 16, 8, 4, 2, 1; same as the serial __shfl_xor 16..1 :1081-1086)
//   total       = ((0 + tile(0)) + tile(1)) + ... + tile(15)   (:3000-3006)
//   low         = (total/4) * sigmoid(total/4);  injection = 2 * sigmoid(total/4)   (:3016-3030)
// CUDA mapping (same as HIP): block = 8 warps x 8 outputs, 16 rows; hardware lane = window lane; 128 accumulators
// (16 rows x 8 outputs) per lane. The butterfly is a transpose-reduce: at each level a lane keeps the half of its
// items selected by its lane bit and receives its partner's value of the same items, so after 5 levels lane L holds
// 4 finished (row, output) pairs: row q = 8 b4 + 4 b3 + 2 b2 + b1, outputs 4 b0 + i (b = bits of L).
constexpr unsigned kProjRows = 16U, kProjOutputs = 8U, kProjWarps = 8U, kProjItems = kProjRows * kProjOutputs;
constexpr unsigned kProjLowBlocks = kRank / (kProjWarps * kProjOutputs);  // 5

// One butterfly level at distance 16 >> Level: items [0, half) keep the lane's half and add the partner's same items.
template <unsigned Level> __device__ __forceinline__ void fold_level(float (&acc)[kProjItems], unsigned lane) {
    constexpr unsigned distance = 16U >> Level, half = 64U >> Level;
    const bool upper = (lane & distance) != 0U;
#pragma unroll
    for (unsigned i = 0; i < half; ++i) {
        const float keep = upper ? acc[i + half] : acc[i];
        const float send = upper ? acc[i] : acc[i + half];
        acc[i] = __fadd_rn(keep, __shfl_xor_sync(0xffffffffU, send, distance));
    }
}

// x_rows and low_rows are the same workspace (normalized part read, low part written): no __restrict__ on them.
__global__ __launch_bounds__(256, 1) void hc_projection_kernel(const float* x_rows, std::uint32_t rows,
    const std::int8_t* __restrict__ down_codes, const float* __restrict__ down_scales, const float* __restrict__ inject,
    float* low_rows, float* __restrict__ injection, am::Tables t) {
    __shared__ float4 xs[2][kProjRows][32];
    const unsigned warp = threadIdx.x / 32U, lane = threadIdx.x % 32U, row0 = blockIdx.y * kProjRows;
    const bool is_inject = blockIdx.x >= kProjLowBlocks;
    const unsigned first = is_inject ? 0U : (blockIdx.x * kProjWarps + warp) * kProjOutputs;
    const unsigned total_outputs = is_inject ? (warp == 0U ? kStreams : 0U) : kRank;
    const bool active = first < total_outputs;

    const auto stage = [&](unsigned chunk, unsigned buffer) {
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m) {
            const unsigned e = threadIdx.x + 256U * m, r = e / 32U, k = e % 32U;
            float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            if (row0 + r < rows)
                v = *reinterpret_cast<const float4*>(x_rows + static_cast<std::size_t>(row0 + r) * kNormalizedPitch +
                    (chunk / 5U) * 640U + (chunk % 5U) * 128U + k * 4U);
            xs[buffer][r][k] = v;
        }
    };
    const auto fetch = [&](unsigned chunk, float4 (&next)[2]) {
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m) {
            const unsigned e = threadIdx.x + 256U * m, r = e / 32U, k = e % 32U;
            next[m] = row0 + r < rows ? *reinterpret_cast<const float4*>(x_rows + static_cast<std::size_t>(row0 + r) *
                kNormalizedPitch + (chunk / 5U) * 640U + (chunk % 5U) * 128U + k * 4U) : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        }
    };

    float acc[kProjItems], total[4];
#pragma unroll
    for (unsigned i = 0; i < kProjItems; ++i) acc[i] = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) total[i] = 0.0F;
    // Weights of the next chunk are loaded one chunk ahead (as HIP's load_weights, :2975): raw[] holds, for the Q8
    // blocks, the 8 code words then the 8 scales (bits); for the injection block, the 4 x 4 FP32 weights.
    std::uint32_t raw[16];
    const auto load_raw = [&](unsigned chunk) {
        const unsigned feature = (chunk / 5U) * 640U + (chunk % 5U) * 128U + lane * 4U;
        if (is_inject) {
#pragma unroll
            for (unsigned p = 0; p < 4U; ++p) {
                const float4 w = first + p < total_outputs ?
                    *reinterpret_cast<const float4*>(inject + static_cast<std::size_t>(first + p) * kWide + feature) :
                    make_float4(0.0F, 0.0F, 0.0F, 0.0F);
                raw[4U * p] = __float_as_uint(w.x); raw[4U * p + 1U] = __float_as_uint(w.y);
                raw[4U * p + 2U] = __float_as_uint(w.z); raw[4U * p + 3U] = __float_as_uint(w.w);
            }
        } else {
#pragma unroll
            for (unsigned p = 0; p < kProjOutputs; ++p) {
                raw[p] = *reinterpret_cast<const std::uint32_t*>(down_codes + static_cast<std::size_t>(first + p) * kWide + feature);
                raw[8U + p] = __float_as_uint(down_scales[static_cast<std::size_t>(first + p) * (kWide / 32U) + feature / 32U]);
            }
        }
    };
    stage(0U, 0U);
    if (active) load_raw(0U);
    __syncthreads();
    for (unsigned chunk = 0; chunk < 80U; ++chunk) {
        float4 next[2];
        if (chunk + 1U < 80U) fetch(chunk + 1U, next);
        if (active) {
            float c[kProjOutputs][4];
#pragma unroll
            for (unsigned p = 0; p < kProjOutputs; ++p)
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j)
                    // injection: FP32 weights (outputs 4..7 of the injection warp are zero, as HIP's padding);
                    // Q8: coefficient = scale * float(code), exact (:2967-2971)
                    c[p][j] = is_inject ? (p < 4U ? __uint_as_float(raw[4U * p + j]) : 0.0F) :
                        __fmul_rn(__uint_as_float(raw[8U + p]), static_cast<float>(static_cast<signed char>((raw[p] >> (8U * j)) & 255U)));
            if (chunk + 1U < 80U) load_raw(chunk + 1U);
            const float4 (*x)[32] = xs[chunk & 1U];
#pragma unroll
            for (unsigned r = 0; r < kProjRows; ++r) {
                const float4 v = x[r][lane];
                const float in[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j)
#pragma unroll
                    for (unsigned p = 0; p < kProjOutputs; ++p) acc[r * kProjOutputs + p] = __fmaf_rn(in[j], c[p][j], acc[r * kProjOutputs + p]);
            }
            if (chunk % 5U == 4U) {
                // Tile fold: xor butterfly, distances 16, 8, 4, 2, 1 (transpose-reduce form, see above).
                fold_level<0>(acc, lane); fold_level<1>(acc, lane); fold_level<2>(acc, lane); fold_level<3>(acc, lane);
                fold_level<4>(acc, lane);
#pragma unroll
                for (unsigned i = 0; i < 4U; ++i) total[i] = __fadd_rn(total[i], acc[i]);
#pragma unroll
                for (unsigned i = 0; i < kProjItems; ++i) acc[i] = 0.0F;
            }
        }
        if (chunk + 1U < 80U) {  // buffer (chunk + 1) & 1 was last read in chunk - 1, before the barrier below
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m) {
                const unsigned e = threadIdx.x + 256U * m;
                xs[(chunk + 1U) & 1U][e / 32U][e % 32U] = next[m];
            }
        }
        __syncthreads();
    }
    if (!active) return;
    const unsigned b = lane;
    const unsigned r = 8U * ((b >> 4U) & 1U) + 4U * ((b >> 3U) & 1U) + 2U * ((b >> 2U) & 1U) + ((b >> 1U) & 1U);
    const unsigned out_base = first + 4U * (b & 1U);
    if (row0 + r >= rows) return;
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) {
        const unsigned o = out_base + i;
        if (o >= total_outputs) continue;
        const float scaled = __fdiv_rn(total[i], static_cast<float>(kStreams));
        const float sig = amd_sigmoid(t, scaled);
        if (is_inject) injection[static_cast<std::size_t>(row0 + r) * kStreams + o] = __fadd_rn(sig, sig);
        else low_rows[static_cast<std::size_t>(row0 + r) * kNormalizedPitch + kWide + o] = __fmul_rn(scaled, sig);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// [O] wide2_hyper_output_kernel<u16, 32, true> (hip_hyper_mix_launcher.hip:3193-3268). Thread = one output feature f,
// 32 rows x 4 streams gate chains g = fma(low[k], c_s(k), g), k = 0..319 ascending from 0; then
// mixed = fma(n3, sig3, fma(n2, sig2, fma(n1, sig1, fma(n0, sig0, 0)))) and bf16(mixed * 0.25).
// The 32 rows' low values sit in shared memory (HIP reads a transposed copy low_t through scalar loads; the values
// are the same FP32 lows, so this kernel has no 8192-row pitch limit: above 8192 rows HIP switches to
// wide_lane_chain_hyper_output_kernel<u16,32> (:3075-3180), whose ISA (hip_hyper_mix_launcher.s:355815) also
// contracts `gate += low * c` to v_fmac_f32 and the mix to v_fma(.., 0)/v_fmac: the same DAG, so no change).
constexpr unsigned kOutRows = 32U, kOutThreads = 128U;

template <bool Fused>
__global__ __launch_bounds__(128) void hc_output_kernel(const float* __restrict__ normalized, std::uint32_t rows,
    const std::uint32_t* __restrict__ up_codes, const float* __restrict__ up_scales, std::uint16_t* __restrict__ output,
    float* __restrict__ gates, am::Tables t) {
    __shared__ __align__(16) float lows[kRank][kOutRows];
    const unsigned row0 = blockIdx.x * kOutRows, f = blockIdx.y * kOutThreads + threadIdx.x;
    for (unsigned idx = threadIdx.x; idx < kRank * kOutRows; idx += kOutThreads) {
        const unsigned r = idx / kRank, k = idx % kRank;
        lows[k][r] = row0 + r < rows ? normalized[static_cast<std::size_t>(row0 + r) * kNormalizedPitch + kWide + k] : 0.0F;
    }
    __syncthreads();
    float g[kStreams][kOutRows];
#pragma unroll
    for (unsigned s = 0; s < kStreams; ++s)
#pragma unroll
        for (unsigned r = 0; r < kOutRows; ++r) g[s][r] = 0.0F;
    float scale_next[kStreams];
    std::uint32_t word_next[kStreams];
    const auto load_q = [&](unsigned q) {
#pragma unroll
        for (unsigned s = 0; s < kStreams; ++s) {
            scale_next[s] = up_scales[(static_cast<std::size_t>(q / 8U) * kStreams + s) * kHidden + f];
            word_next[s] = up_codes[(static_cast<std::size_t>(q) * kStreams + s) * kHidden + f];
        }
    };
    load_q(0U);
#pragma unroll 1
    for (unsigned q = 0; q < kRank / 4U; ++q) {
        float scale[kStreams];
        std::uint32_t word[kStreams];
#pragma unroll
        for (unsigned s = 0; s < kStreams; ++s) { scale[s] = scale_next[s]; word[s] = word_next[s]; }
        if (q + 1U < kRank / 4U) load_q(q + 1U);  // one k-quad ahead
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            float c[kStreams];
#pragma unroll
            for (unsigned s = 0; s < kStreams; ++s)  // scale * (float(code ^ 0x80) - 128) == scale * float(int8), exact
                c[s] = __fmul_rn(scale[s], static_cast<float>(static_cast<signed char>((word[s] >> (8U * j)) & 255U)));
            const float4* low = reinterpret_cast<const float4*>(lows[4U * q + j]);
#pragma unroll
            for (unsigned r4 = 0; r4 < kOutRows / 4U; ++r4) {
                const float4 v = low[r4];
#pragma unroll
                for (unsigned s = 0; s < kStreams; ++s) {
                    g[s][4U * r4 + 0U] = __fmaf_rn(v.x, c[s], g[s][4U * r4 + 0U]);
                    g[s][4U * r4 + 1U] = __fmaf_rn(v.y, c[s], g[s][4U * r4 + 1U]);
                    g[s][4U * r4 + 2U] = __fmaf_rn(v.z, c[s], g[s][4U * r4 + 2U]);
                    g[s][4U * r4 + 3U] = __fmaf_rn(v.w, c[s], g[s][4U * r4 + 3U]);
                }
            }
        }
    }
    if constexpr (!Fused) {
#pragma unroll
        for (unsigned r = 0; r < kOutRows; ++r) {
            if (row0 + r >= rows) break;
#pragma unroll
            for (unsigned s = 0; s < kStreams; ++s) gates[(static_cast<std::size_t>(row0 + r) * kStreams + s) * kHidden + f] = g[s][r];
        }
        return;
    }
#pragma unroll
    for (unsigned r = 0; r < kOutRows; ++r) {
        if (row0 + r >= rows) break;
        const float* n = normalized + static_cast<std::size_t>(row0 + r) * kNormalizedPitch + f;
        float mixed = 0.0F;
#pragma unroll
        for (unsigned s = 0; s < kStreams; ++s) mixed = __fmaf_rn(n[s * kHidden], amd_sigmoid(t, g[s][r]), mixed);
        output[static_cast<std::size_t>(row0 + r) * kHidden + f] = float_to_bf16(__fmul_rn(mixed, 0.25F));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode shapes (rows <= 8): the [P], [O] and router DAGs above, laid out for few rows.
// STRIX_QRTX_DECODE_KERNELS=0 keeps the tiled kernels (reference seam).
bool decode_rows_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QRTX_DECODE_KERNELS"); return !(v && v[0] == '0'); }();
    return on;
}

// [P] for R rows: block = one output (320 low outputs + 1 injection block), warp w = tile w (16 warps). Lane l keeps the
// leaf chain over s = 0..4, j = 0..3 of feature 640 w + 128 s + 4 l + j; the xor butterfly 16..1 gives tile(w) (every
// lane the same sum: IEEE addition is commutative); thread (o, r) then adds the tiles in order from 0.
template <unsigned R>
__global__ __launch_bounds__(512) void hc_projection_rows_kernel(const float* x_rows, const std::int8_t* __restrict__ down_codes,
    const float* __restrict__ down_scales, const float* __restrict__ inject, float* low_rows, float* __restrict__ injection,
    am::Tables t) {
    __shared__ float tiles[16][4][R];
    const unsigned warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const bool is_inject = blockIdx.x >= kRank;
    const unsigned outputs = is_inject ? kStreams : 1U;
    float acc[4][R];
#pragma unroll
    for (unsigned o = 0; o < 4U; ++o)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) acc[o][r] = 0.0F;
#pragma unroll
    for (unsigned sub = 0; sub < 5U; ++sub) {
        const unsigned f = 640U * warp + 128U * sub + 4U * lane;
        float x[R][4];
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            const float4 v = *reinterpret_cast<const float4*>(x_rows + static_cast<std::size_t>(r) * kNormalizedPitch + f);
            x[r][0] = v.x; x[r][1] = v.y; x[r][2] = v.z; x[r][3] = v.w;
        }
        if (is_inject) {
#pragma unroll
            for (unsigned o = 0; o < 4U; ++o) {
                const float4 w = *reinterpret_cast<const float4*>(inject + static_cast<std::size_t>(o) * kWide + f);
                const float c[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j)
#pragma unroll
                    for (unsigned r = 0; r < R; ++r) acc[o][r] = __fmaf_rn(x[r][j], c[j], acc[o][r]);
            }
        } else {
            const unsigned o = blockIdx.x;
            const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(down_codes + static_cast<std::size_t>(o) * kWide + f);
            const float scale = down_scales[static_cast<std::size_t>(o) * (kWide / 32U) + f / 32U];
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                const float c = __fmul_rn(scale, static_cast<float>(static_cast<signed char>((word >> (8U * j)) & 255U)));
#pragma unroll
                for (unsigned r = 0; r < R; ++r) acc[0][r] = __fmaf_rn(x[r][j], c, acc[0][r]);
            }
        }
    }
#pragma unroll
    for (unsigned o = 0; o < 4U; ++o) {
        if (o >= outputs) break;
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            float v = acc[o][r];
#pragma unroll
            for (unsigned d = 16U; d > 0U; d >>= 1U) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffU, v, d));
            if (lane == 0U) tiles[warp][o][r] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x >= outputs * R) return;
    const unsigned o = threadIdx.x / R, r = threadIdx.x % R;
    float total = 0.0F;
#pragma unroll
    for (unsigned w = 0; w < 16U; ++w) total = __fadd_rn(total, tiles[w][o][r]);
    const float scaled = __fdiv_rn(total, static_cast<float>(kStreams));
    const float sig = amd_sigmoid(t, scaled);
    if (is_inject) injection[static_cast<std::size_t>(r) * kStreams + o] = __fadd_rn(sig, sig);
    else low_rows[static_cast<std::size_t>(r) * kNormalizedPitch + kWide + blockIdx.x] = __fmul_rn(scaled, sig);
}

// [O] for R rows: thread = (feature f, stream s), lane = 4 f' + s; the 4 streams of a feature are adjacent lanes, so
// the stream mix fma(n_s, sig_s, ..) s ascending from 0 is done by lane s = 0 after shuffles.
template <unsigned R>
__global__ __launch_bounds__(128) void hc_output_rows_kernel(const float* __restrict__ normalized,
    const std::uint32_t* __restrict__ up_codes, const float* __restrict__ up_scales, std::uint16_t* __restrict__ output,
    am::Tables t) {
    __shared__ float lows[kRank][R];
    for (unsigned i = threadIdx.x; i < kRank * R; i += blockDim.x)
        lows[i / R][i % R] = normalized[static_cast<std::size_t>(i % R) * kNormalizedPitch + kWide + i / R];
    __syncthreads();
    const unsigned s = threadIdx.x % 4U, f = blockIdx.x * (blockDim.x / 4U) + threadIdx.x / 4U;
    float g[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) g[r] = 0.0F;
#pragma unroll 4
    for (unsigned q = 0; q < kRank / 4U; ++q) {
        const float scale = up_scales[(static_cast<std::size_t>(q / 8U) * kStreams + s) * kHidden + f];
        const std::uint32_t word = up_codes[(static_cast<std::size_t>(q) * kStreams + s) * kHidden + f];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const float c = __fmul_rn(scale, static_cast<float>(static_cast<signed char>((word >> (8U * j)) & 255U)));
#pragma unroll
            for (unsigned r = 0; r < R; ++r) g[r] = __fmaf_rn(lows[4U * q + j][r], c, g[r]);
        }
    }
    const unsigned base = (threadIdx.x % 32U) & ~3U;
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        const float term = amd_sigmoid(t, g[r]);
        const float n = normalized[static_cast<std::size_t>(r) * kNormalizedPitch + s * kHidden + f];
        float sig[4], nv[4];
#pragma unroll
        for (unsigned k = 0; k < 4U; ++k) {
            sig[k] = __shfl_sync(0xffffffffU, term, base + k);
            nv[k] = __shfl_sync(0xffffffffU, n, base + k);
        }
        if (s == 0U) {
            float mixed = 0.0F;
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) mixed = __fmaf_rn(nv[k], sig[k], mixed);
            output[static_cast<std::size_t>(r) * kHidden + f] = float_to_bf16(__fmul_rn(mixed, 0.25F));
        }
    }
}

// Split epilogue: the same stream mix per (row, feature) from the stored gates.
__global__ __launch_bounds__(256) void hc_mix_epilogue_kernel(const float* __restrict__ normalized, const float* __restrict__ gates,
    std::uint32_t rows, std::uint16_t* __restrict__ output, am::Tables t) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * kHidden) return;
    const std::size_t row = i / kHidden, f = i % kHidden;
    const float* n = normalized + row * kNormalizedPitch + f;
    const float* g = gates + row * kStreams * kHidden + f;
    float mixed = 0.0F;
#pragma unroll
    for (unsigned s = 0; s < kStreams; ++s) mixed = __fmaf_rn(n[s * kHidden], amd_sigmoid(t, g[s * kHidden]), mixed);
    output[i] = float_to_bf16(__fmul_rn(mixed, 0.25F));
}

// ---------------------------------------------------------------------------------------------------------------------
// Elementwise.
__global__ void embedding_kernel(const unsigned char* __restrict__ table, const std::uint32_t* __restrict__ ids,
    std::uint16_t* __restrict__ out) {
    const unsigned row = blockIdx.x;
    const std::size_t token = ids[row];
    for (unsigned f = threadIdx.x; f < kHidden; f += blockDim.x) {
        const unsigned char* block = table + (token * (kHidden / 32U) + f / 32U) * 34U;
        const float scale = half_bits_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8U)));
        out[static_cast<std::size_t>(row) * kHidden + f] =
            float_to_bf16(__fmul_rn(scale, static_cast<float>(static_cast<signed char>(block[2 + f % 32U]))));
    }
}
__global__ void expand_kernel(const std::uint16_t* __restrict__ in, std::uint16_t* __restrict__ out, std::size_t total) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const std::size_t row = i / kWide, f = i % kHidden;
    out[i] = float_to_bf16(bf16_to_float(in[row * kHidden + f]));  // BF16 -> FP32 -> BF16: the identity
}
__global__ void inject_kernel(const std::uint16_t* __restrict__ hyper, const std::uint16_t* __restrict__ hidden,
    const float* __restrict__ weights, std::uint16_t* __restrict__ out, std::size_t total) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const std::size_t row = i / kWide, within = i % kWide, s = within / kHidden, f = within % kHidden;
    out[i] = float_to_bf16(__fmaf_rn(bf16_to_float(hidden[row * kHidden + f]), weights[row * kStreams + s], bf16_to_float(hyper[i])));
}
__global__ void combine_kernel(const std::uint16_t* __restrict__ a, const std::uint16_t* __restrict__ b,
    std::uint16_t* __restrict__ out, std::size_t total) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = float_to_bf16(__fadd_rn(bf16_to_float(a[i]), bf16_to_float(b[i])));
}

// ---------------------------------------------------------------------------------------------------------------------
// Router logits (lanes_tiles_moe_route_logits_kernel, hip_moe_route_launcher.hip:1280-1336): per (row, expert) one
// chain sum = fma(x_f, w_f, sum), f = 0..2559 ascending from 0. CUDA: 64 rows x 64 experts per block, 4 x 4 outputs
// per thread, K staged 32 at a time in shared memory; each output keeps its own sequential chain.
constexpr unsigned kRtRows = 64U, kRtExperts = 64U, kRtK = 32U;
__global__ __launch_bounds__(256) void route_logits_kernel(const std::uint16_t* __restrict__ hidden, const float* __restrict__ w,
    std::uint32_t rows, float* __restrict__ logits) {
    __shared__ __align__(16) float xs[kRtK][kRtRows + 4U];  // +4: fewer bank conflicts on the transposing stores
    __shared__ __align__(16) float ws[kRtK][kRtExperts + 4U];
    const unsigned tx = threadIdx.x % 16U, ty = threadIdx.x / 16U;
    const unsigned row0 = blockIdx.x * kRtRows, e0 = blockIdx.y * kRtExperts;
    float acc[4][4];
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) acc[i][j] = 0.0F;
    for (unsigned k0 = 0; k0 < kHidden; k0 += kRtK) {
#pragma unroll
        for (unsigned m = 0; m < 8U; ++m) {
            const unsigned idx = threadIdx.x + 256U * m, r = idx / kRtK, k = idx % kRtK;
            xs[k][r] = row0 + r < rows ? bf16_to_float(hidden[static_cast<std::size_t>(row0 + r) * kHidden + k0 + k]) : 0.0F;
            ws[k][r] = w[static_cast<std::size_t>(e0 + r) * kHidden + k0 + k];
        }
        __syncthreads();
#pragma unroll
        for (unsigned k = 0; k < kRtK; ++k) {
            const float4 a = *reinterpret_cast<const float4*>(&xs[k][ty * 4U]);
            const float4 b = *reinterpret_cast<const float4*>(&ws[k][tx * 4U]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) acc[i][j] = __fmaf_rn(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) {
        const unsigned r = row0 + ty * 4U + i;
        if (r >= rows) continue;
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) logits[static_cast<std::size_t>(r) * kExperts + e0 + tx * 4U + j] = acc[i][j];
    }
}

// Router logits for R rows: block = 4 experts (128 blocks); its 4 x 2560 FP32 weights and the R rows are staged in
// shared memory in one go, then thread e < 4 keeps the sequential chain over f ascending from 0 for each row.
constexpr unsigned kRtePerBlock = 4U, kRtePitch = kHidden + 1U;   // +1: the 4 chains read distinct banks
template <unsigned R>
__global__ __launch_bounds__(256) void route_logits_rows_kernel(const std::uint16_t* __restrict__ hidden, const float* __restrict__ w,
    float* __restrict__ logits) {
    extern __shared__ float rt_smem[];
    float* ws = rt_smem;                                   // [4][kRtePitch]
    auto* xs = reinterpret_cast<std::uint16_t*>(rt_smem + kRtePerBlock * kRtePitch);   // BF16 [R][2560]
    const unsigned e0 = blockIdx.x * kRtePerBlock;
    for (unsigned i = threadIdx.x; i < kRtePerBlock * kHidden; i += 256U)
        ws[(i / kHidden) * kRtePitch + i % kHidden] = w[static_cast<std::size_t>(e0) * kHidden + i];
    for (unsigned i = threadIdx.x; i < R * kHidden; i += 256U) xs[i] = hidden[i];
    __syncthreads();
    if (threadIdx.x >= kRtePerBlock) return;
    const float* we = ws + threadIdx.x * kRtePitch;
    float acc[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) acc[r] = 0.0F;
#pragma unroll 8
    for (unsigned k = 0; k < kHidden; ++k) {
        const float wv = we[k];
#pragma unroll
        for (unsigned r = 0; r < R; ++r) acc[r] = __fmaf_rn(bf16_to_float(xs[r * kHidden + k]), wv, acc[r]);
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r) logits[static_cast<std::size_t>(r) * kExperts + e0 + threadIdx.x] = acc[r];
}

// [S] wave_parallel_moe_route_select_row (hip_moe_route_launcher.hip:866-1010), one 256-thread block per row.
__device__ __forceinline__ bool better(float lv, unsigned li, float rv, unsigned ri) {  // :271-279
    return lv > rv || (lv == rv && li < ri);
}
__global__ __launch_bounds__(256) void route_select_kernel(const float* __restrict__ workspace, std::uint32_t top_k,
    bool normalize, std::uint32_t* __restrict__ ids, float* __restrict__ weights, am::Tables t) {
    constexpr unsigned kThreads = 256U, kWave = 32U, kCandidates = kExperts / kWave;
    const unsigned row = blockIdx.x, lane = threadIdx.x;
    __shared__ float logits[kExperts];
    __shared__ float scratch[kThreads];
    __shared__ float best_values[32];
    __shared__ unsigned best_indices[32];
    for (unsigned e = lane; e < kExperts; e += kThreads) logits[e] = workspace[static_cast<std::size_t>(row) * kExperts + e];
    __syncthreads();
    float local_max = -INFINITY;
    for (unsigned e = lane; e < kExperts; e += kThreads) local_max = fmaxf(local_max, logits[e]);
    scratch[lane] = local_max;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride > 0U; stride >>= 1U) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float maximum = scratch[0];
    __syncthreads();
    float local_sum = 0.0F;
    for (unsigned e = lane; e < kExperts; e += kThreads) {
        const float p = am::amd_expf(t, __fsub_rn(logits[e], maximum));
        logits[e] = p;
        local_sum = __fadd_rn(local_sum, p);
    }
    scratch[lane] = local_sum;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride > 0U; stride >>= 1U) {
        if (lane < stride) scratch[lane] = __fadd_rn(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float denominator = scratch[0];
    for (unsigned e = lane; e < kExperts; e += kThreads) logits[e] = __fdiv_rn(logits[e], denominator);
    __syncthreads();
    if (lane < kWave) {
        float values[kCandidates];
        unsigned available = 0;
#pragma unroll
        for (unsigned i = 0; i < kCandidates; ++i) {
            values[i] = logits[lane + kWave * i];
            if (values[i] == values[i]) available |= 1U << i;  // NaN probabilities are excluded (:957)
        }
        for (unsigned slot = 0; slot < top_k; ++slot) {
            bool has = false;
            float best = -INFINITY;
            unsigned index = kExperts;
#pragma unroll
            for (unsigned i = 0; i < kCandidates; ++i)
                if ((available & (1U << i)) != 0U) {
                    const unsigned expert = lane + kWave * i;
                    if (!has || better(values[i], expert, best, index)) { has = true; best = values[i]; index = expert; }
                }
#pragma unroll
            for (unsigned mask = kWave / 2U; mask > 0U; mask >>= 1U) {
                const float other = __shfl_xor_sync(0xffffffffU, best, static_cast<int>(mask));
                const unsigned other_index = __shfl_xor_sync(0xffffffffU, index, static_cast<int>(mask));
                const int other_has = __shfl_xor_sync(0xffffffffU, static_cast<int>(has), static_cast<int>(mask));
                if (other_has != 0 && (!has || better(other, other_index, best, index))) { has = true; best = other; index = other_index; }
            }
            if (lane == 0U) { best_values[slot] = has ? best : -INFINITY; best_indices[slot] = has ? index : kExperts; }
            if (has && index % kWave == lane) available &= ~(1U << (index / kWave));
        }
    }
    __syncthreads();
    if (lane == 0U) {
        float selected_sum = 0.0F;
        for (unsigned slot = 0; slot < top_k; ++slot) selected_sum = __fadd_rn(selected_sum, best_values[slot]);
        if (!normalize) selected_sum = 1.0F;
        for (unsigned slot = 0; slot < top_k; ++slot) {
            ids[static_cast<std::size_t>(row) * top_k + slot] = best_indices[slot];
            weights[static_cast<std::size_t>(row) * top_k + slot] = __fdiv_rn(best_values[slot], selected_sum);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// [L] prefetch8_wave_q8_lm_head_kernel<u16, 8> (hip_lm_head_launcher.hip:475-520): warp per vocab row; lane L leaf l
// accumulates partial[l] = fma(x[c + 8L + l], scale * code, partial[l]) over chunks c = 0, 256, ..., 2304; then
// for d = 16, 8, 4, 2, 1: lanes < d add lane + d; lane 0: p0+=p4, p1+=p5, p2+=p6, p3+=p7; p0+=p2, p1+=p3; p0+=p1.
__global__ __launch_bounds__(256) void lm_head_kernel(const std::uint16_t* __restrict__ hidden, const unsigned char* __restrict__ w,
    std::uint32_t vocab, float* __restrict__ logits) {
    __shared__ float x[kHidden];
    for (unsigned f = threadIdx.x; f < kHidden; f += blockDim.x) x[f] = bf16_to_float(hidden[f]);
    __syncthreads();
    const unsigned row = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    if (row >= vocab) return;
    const unsigned char* blocks = w + static_cast<std::size_t>(row) * (kHidden / 32U) * 34U;
    float partial[8];
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) partial[l] = 0.0F;
#pragma unroll
    for (unsigned chunk = 0; chunk < kHidden; chunk += 256U) {
        const unsigned feature = chunk + 8U * lane;
        const unsigned char* block = blocks + (feature / 32U) * 34U;
        const std::uint16_t* h = reinterpret_cast<const std::uint16_t*>(block);  // block offsets are even
        const float scale = half_bits_to_float(h[0]);
        const unsigned o = 1U + (feature % 32U) / 2U;
        const std::uint16_t c01 = h[o], c23 = h[o + 1U], c45 = h[o + 2U], c67 = h[o + 3U];
        const std::uint16_t pairs[4] = {c01, c23, c45, c67};
#pragma unroll
        for (unsigned l = 0; l < 8U; ++l) {
            const signed char code = static_cast<signed char>((pairs[l / 2U] >> (8U * (l % 2U))) & 255U);
            partial[l] = __fmaf_rn(x[feature + l], __fmul_rn(scale, static_cast<float>(code)), partial[l]);
        }
    }
#pragma unroll
    for (unsigned d = 16U; d > 0U; d /= 2U)
#pragma unroll
        for (unsigned l = 0; l < 8U; ++l) {
            const float partner = __shfl_down_sync(0xffffffffU, partial[l], d);
            if (lane < d) partial[l] = __fadd_rn(partial[l], partner);
        }
    if (lane == 0U) {
        partial[0] = __fadd_rn(partial[0], partial[4]); partial[1] = __fadd_rn(partial[1], partial[5]);
        partial[2] = __fadd_rn(partial[2], partial[6]); partial[3] = __fadd_rn(partial[3], partial[7]);
        partial[0] = __fadd_rn(partial[0], partial[2]); partial[1] = __fadd_rn(partial[1], partial[3]);
        partial[0] = __fadd_rn(partial[0], partial[1]);
        logits[row] = partial[0];
    }
}

// lm_head_kernel for R rows sharing each weight read: row r's partials, leaf order and reduction tree are exactly the
// one-row kernel's (the HIP rows variant keeps the same DAG, hip_lm_head_launcher.hip:573-664).
template <unsigned R>
__global__ __launch_bounds__(256) void lm_head_rows_kernel(const std::uint16_t* __restrict__ hidden, std::uint64_t hidden_stride,
    const unsigned char* __restrict__ w, std::uint32_t vocab, float* __restrict__ logits, std::uint64_t logits_stride) {
    __shared__ std::uint16_t xs[R][kHidden];
    for (unsigned i = threadIdx.x; i < R * kHidden; i += blockDim.x)
        xs[i / kHidden][i % kHidden] = hidden[static_cast<std::size_t>(i / kHidden) * hidden_stride + i % kHidden];
    __syncthreads();
    const unsigned row = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    if (row >= vocab) return;
    const unsigned char* blocks = w + static_cast<std::size_t>(row) * (kHidden / 32U) * 34U;
    float partial[R][8];
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
#pragma unroll
        for (unsigned l = 0; l < 8U; ++l) partial[r][l] = 0.0F;
#pragma unroll
    for (unsigned chunk = 0; chunk < kHidden; chunk += 256U) {
        const unsigned feature = chunk + 8U * lane;
        const unsigned char* block = blocks + (feature / 32U) * 34U;
        const std::uint16_t* h = reinterpret_cast<const std::uint16_t*>(block);
        const float scale = half_bits_to_float(h[0]);
        const unsigned o = 1U + (feature % 32U) / 2U;
        const std::uint16_t pairs[4] = {h[o], h[o + 1U], h[o + 2U], h[o + 3U]};
        float coef[8];
#pragma unroll
        for (unsigned l = 0; l < 8U; ++l)
            coef[l] = __fmul_rn(scale, static_cast<float>(static_cast<signed char>((pairs[l / 2U] >> (8U * (l % 2U))) & 255U)));
#pragma unroll
        for (unsigned r = 0; r < R; ++r)
#pragma unroll
            for (unsigned l = 0; l < 8U; ++l)
                partial[r][l] = __fmaf_rn(bf16_to_float(xs[r][feature + l]), coef[l], partial[r][l]);
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
#pragma unroll
        for (unsigned d = 16U; d > 0U; d /= 2U)
#pragma unroll
            for (unsigned l = 0; l < 8U; ++l) {
                const float partner = __shfl_down_sync(0xffffffffU, partial[r][l], d);
                if (lane < d) partial[r][l] = __fadd_rn(partial[r][l], partner);
            }
        if (lane == 0U) {
            float* p = partial[r];
            p[0] = __fadd_rn(p[0], p[4]); p[1] = __fadd_rn(p[1], p[5]);
            p[2] = __fadd_rn(p[2], p[6]); p[3] = __fadd_rn(p[3], p[7]);
            p[0] = __fadd_rn(p[0], p[2]); p[1] = __fadd_rn(p[1], p[3]);
            p[0] = __fadd_rn(p[0], p[1]);
            logits[static_cast<std::size_t>(r) * logits_stride + row] = p[0];
        }
    }
}

// Greedy selection on the device (select_greedy_token semantics): first maximum, lowest index on ties (+0 == -0);
// a non-finite logit sets out[row] = 0xffffffff.
__global__ __launch_bounds__(1024) void first_max_kernel(const float* __restrict__ logits, std::uint64_t stride, std::uint32_t count,
    std::uint32_t* __restrict__ out) {
    const float* x = logits + static_cast<std::size_t>(blockIdx.x) * stride;
    float best = 0.0F;
    unsigned index = 0xffffffffU;
    bool bad = false;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
        const float v = x[i];
        if (!isfinite(v)) bad = true;
        else if (index == 0xffffffffU || v > best) { best = v; index = i; }
    }
    __shared__ float bv[1024];
    __shared__ unsigned bi[1024];
    __shared__ int any_bad;
    if (threadIdx.x == 0) any_bad = 0;
    __syncthreads();
    if (bad) any_bad = 1;
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = index;
    __syncthreads();
    for (unsigned d = blockDim.x / 2U; d > 0U; d /= 2U) {
        if (threadIdx.x < d) {
            const float v = bv[threadIdx.x + d];
            const unsigned j = bi[threadIdx.x + d];
            if (j != 0xffffffffU && (bi[threadIdx.x] == 0xffffffffU || v > bv[threadIdx.x] || (v == bv[threadIdx.x] && j < bi[threadIdx.x]))) {
                bv[threadIdx.x] = v;
                bi[threadIdx.x] = j;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = any_bad || bi[0] == 0xffffffffU ? 0xffffffffU : bi[0];
}

unsigned blocks_for(std::size_t total, unsigned threads) { return static_cast<unsigned>((total + threads - 1U) / threads); }

}  // namespace

std::size_t hc_packed_bytes() noexcept { return kDownCodeBytes + kDownScaleBytes + kUpCodeBytes + kUpScaleBytes; }
HcPacked hc_packed_view(void* base) noexcept {
    auto* b = static_cast<unsigned char*>(base);
    HcPacked p;
    p.down_codes = reinterpret_cast<std::int8_t*>(b);
    p.down_scales = reinterpret_cast<float*>(b + kDownCodeBytes);
    p.up_codes = reinterpret_cast<std::uint32_t*>(b + kDownCodeBytes + kDownScaleBytes);
    p.up_scales = reinterpret_cast<float*>(b + kDownCodeBytes + kDownScaleBytes + kUpCodeBytes);
    return p;
}
void hc_pack(const void* down_q8, const void* up_q8, const HcPacked& packed, cudaStream_t stream) {
    const unsigned items = kRank * (kWide / 32U) + kWide * (kRank / 32U);
    hc_pack_kernel<<<blocks_for(items, 256U), 256U, 0, stream>>>(static_cast<const unsigned char*>(down_q8),
        static_cast<const unsigned char*>(up_q8), packed);
    check(cudaGetLastError(), "pack launch");
}

namespace {
template <unsigned R> void hc_mix_rows_r(const HcMixArgs& a, const am::Tables& tables, cudaStream_t stream) {
    hc_projection_rows_kernel<R><<<kRank + (a.inject ? 1U : 0U), 512U, 0, stream>>>(a.normalized, a.packed.down_codes,
        a.packed.down_scales, a.inject, a.normalized, a.injection, tables);
    check(cudaGetLastError(), "projection rows launch");
    hc_output_rows_kernel<R><<<kHidden / 32U, 128U, 0, stream>>>(a.normalized, a.packed.up_codes, a.packed.up_scales, a.output, tables);
    check(cudaGetLastError(), "output rows launch");
}
void hc_mix_rows(const HcMixArgs& a, const am::Tables& tables, cudaStream_t stream) {
    switch (a.rows) {
    case 1: hc_mix_rows_r<1>(a, tables, stream); break;
    case 2: hc_mix_rows_r<2>(a, tables, stream); break;
    case 3: hc_mix_rows_r<3>(a, tables, stream); break;
    case 4: hc_mix_rows_r<4>(a, tables, stream); break;
    case 5: hc_mix_rows_r<5>(a, tables, stream); break;
    case 6: hc_mix_rows_r<6>(a, tables, stream); break;
    case 7: hc_mix_rows_r<7>(a, tables, stream); break;
    default: hc_mix_rows_r<8>(a, tables, stream); break;
    }
}
template <unsigned R> void route_rows_r(const std::uint16_t* hidden, const float* w, float* logits, cudaStream_t stream) {
    constexpr unsigned smem = kRtePerBlock * kRtePitch * 4U + R * kHidden * 2U;
    static const cudaError_t attr = cudaFuncSetAttribute(route_logits_rows_kernel<R>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    check(attr, "route logits rows smem");
    route_logits_rows_kernel<R><<<kExperts / kRtePerBlock, 256U, smem, stream>>>(hidden, w, logits);
}
}  // namespace

void hc_mix(const HcMixArgs& a, const am::Tables& tables, cudaStream_t stream, unsigned stages) {
    if (!a.rows) return;
    if (a.rows > 65535U * kProjRows) throw std::invalid_argument("qwen_hc mix: too many rows");
    if (stages & 1U) hc_norm_kernel<<<dim3(a.rows, kStreams), 256U, 0, stream>>>(a.input, a.norm, a.epsilon, a.normalized, tables);
    check(cudaGetLastError(), "norm launch");
    const unsigned out_blocks = kProjLowBlocks + (a.inject ? 1U : 0U);
    // Output blocks fastest (HIP default order): the 6 blocks of one row block run together and share its x rows in L2.
    if (a.rows <= 8U && stages == 15U && !a.gates && decode_rows_enabled()) {
        hc_mix_rows(a, tables, stream);
        return;
    }
    if (stages & 2U) hc_projection_kernel<<<dim3(out_blocks, (a.rows + kProjRows - 1U) / kProjRows), 256U, 0, stream>>>(a.normalized, a.rows,
        a.packed.down_codes, a.packed.down_scales, a.inject, a.normalized, a.injection, tables);
    check(cudaGetLastError(), "projection launch");
    const dim3 out_grid((a.rows + kOutRows - 1U) / kOutRows, kHidden / kOutThreads);
    if ((stages & 4U) && !a.gates)
        hc_output_kernel<true><<<out_grid, kOutThreads, 0, stream>>>(a.normalized, a.rows, a.packed.up_codes, a.packed.up_scales,
            a.output, nullptr, tables);
    if ((stages & 4U) && a.gates)
        hc_output_kernel<false><<<out_grid, kOutThreads, 0, stream>>>(a.normalized, a.rows, a.packed.up_codes, a.packed.up_scales,
            a.output, a.gates, tables);
    if ((stages & 8U) && a.gates)
        hc_mix_epilogue_kernel<<<blocks_for(std::size_t{a.rows} * kHidden, 256U), 256U, 0, stream>>>(a.normalized, a.gates, a.rows,
            a.output, tables);
    check(cudaGetLastError(), "output launch");
}

void token_embedding(const void* table, const std::uint32_t* ids, std::uint32_t rows, std::uint16_t* out, cudaStream_t stream) {
    if (!rows) return;
    embedding_kernel<<<rows, 256U, 0, stream>>>(static_cast<const unsigned char*>(table), ids, out);
    check(cudaGetLastError(), "embedding launch");
}
void hyper_expand(const std::uint16_t* in, std::uint32_t rows, std::uint16_t* out, cudaStream_t stream) {
    const std::size_t total = std::size_t{rows} * kWide;
    if (!total) return;
    expand_kernel<<<blocks_for(total, 256U), 256U, 0, stream>>>(in, out, total);
    check(cudaGetLastError(), "expand launch");
}
void inject(const std::uint16_t* hyper, const std::uint16_t* hidden, const float* injection, std::uint32_t rows,
    std::uint16_t* out, cudaStream_t stream) {
    const std::size_t total = std::size_t{rows} * kWide;
    if (!total) return;
    inject_kernel<<<blocks_for(total, 256U), 256U, 0, stream>>>(hyper, hidden, injection, out, total);
    check(cudaGetLastError(), "inject launch");
}
void combine(const std::uint16_t* shared, const std::uint16_t* routed, std::uint32_t rows, std::uint16_t* out, cudaStream_t stream) {
    const std::size_t total = std::size_t{rows} * kHidden;
    if (!total) return;
    combine_kernel<<<blocks_for(total, 256U), 256U, 0, stream>>>(shared, routed, out, total);
    check(cudaGetLastError(), "combine launch");
}
void route_logits(const std::uint16_t* hidden, const float* gate_inp, std::uint32_t rows, float* logits, cudaStream_t stream) {
    if (!rows) return;
    if (rows <= 8U && decode_rows_enabled()) {
        switch (rows) {
        case 1: route_rows_r<1>(hidden, gate_inp, logits, stream); break;
        case 2: route_rows_r<2>(hidden, gate_inp, logits, stream); break;
        case 3: route_rows_r<3>(hidden, gate_inp, logits, stream); break;
        case 4: route_rows_r<4>(hidden, gate_inp, logits, stream); break;
        case 5: route_rows_r<5>(hidden, gate_inp, logits, stream); break;
        case 6: route_rows_r<6>(hidden, gate_inp, logits, stream); break;
        case 7: route_rows_r<7>(hidden, gate_inp, logits, stream); break;
        default: route_rows_r<8>(hidden, gate_inp, logits, stream); break;
        }
        check(cudaGetLastError(), "route logits rows launch");
        return;
    }
    route_logits_kernel<<<dim3((rows + kRtRows - 1U) / kRtRows, kExperts / kRtExperts), 256U, 0, stream>>>(hidden, gate_inp, rows, logits);
    check(cudaGetLastError(), "route logits launch");
}
void route_select(const float* logits, std::uint32_t rows, std::uint32_t top_k, bool normalize, const am::Tables& tables,
    std::uint32_t* ids, float* weights, cudaStream_t stream) {
    if (top_k == 0U || top_k > 32U) throw std::invalid_argument("qwen_hc route_select: top_k must be 1..32");
    if (!rows) return;
    route_select_kernel<<<rows, 256U, 0, stream>>>(logits, top_k, normalize, ids, weights, tables);
    check(cudaGetLastError(), "route select launch");
}
void lm_head(const std::uint16_t* hidden, const void* output_q8, std::uint32_t vocab, float* logits, cudaStream_t stream) {
    if (!vocab) return;
    lm_head_kernel<<<(vocab + 7U) / 8U, 256U, 0, stream>>>(hidden, static_cast<const unsigned char*>(output_q8), vocab, logits);
    check(cudaGetLastError(), "lm head launch");
}
void lm_head_rows(const std::uint16_t* hidden, std::uint64_t hidden_stride, std::uint32_t rows, const void* output_q8,
    std::uint32_t vocab, float* logits, std::uint64_t logits_stride, cudaStream_t stream) {
    if (!vocab || !rows) return;
    const auto* w = static_cast<const unsigned char*>(output_q8);
    const unsigned grid = (vocab + 7U) / 8U;
    switch (rows) {
    case 1: lm_head_rows_kernel<1><<<grid, 256U, 0, stream>>>(hidden, hidden_stride, w, vocab, logits, logits_stride); break;
    case 2: lm_head_rows_kernel<2><<<grid, 256U, 0, stream>>>(hidden, hidden_stride, w, vocab, logits, logits_stride); break;
    case 3: lm_head_rows_kernel<3><<<grid, 256U, 0, stream>>>(hidden, hidden_stride, w, vocab, logits, logits_stride); break;
    case 4: lm_head_rows_kernel<4><<<grid, 256U, 0, stream>>>(hidden, hidden_stride, w, vocab, logits, logits_stride); break;
    case 5: lm_head_rows_kernel<5><<<grid, 256U, 0, stream>>>(hidden, hidden_stride, w, vocab, logits, logits_stride); break;
    case 6: lm_head_rows_kernel<6><<<grid, 256U, 0, stream>>>(hidden, hidden_stride, w, vocab, logits, logits_stride); break;
    case 7: lm_head_rows_kernel<7><<<grid, 256U, 0, stream>>>(hidden, hidden_stride, w, vocab, logits, logits_stride); break;
    case 8: lm_head_rows_kernel<8><<<grid, 256U, 0, stream>>>(hidden, hidden_stride, w, vocab, logits, logits_stride); break;
    default: throw std::invalid_argument("qwen_hc lm_head_rows: rows must be 1..8");
    }
    check(cudaGetLastError(), "lm head rows launch");
}
void first_max_rows(const float* logits, std::uint64_t stride, std::uint32_t rows, std::uint32_t count, std::uint32_t* out,
    cudaStream_t stream) {
    if (!rows) return;
    first_max_kernel<<<rows, 1024U, 0, stream>>>(logits, stride, count, out);
    check(cudaGetLastError(), "first max launch");
}
std::uint32_t first_max_token(const float* logits, std::size_t count) {  // select_greedy_token, src/serial_request.cpp:12-55
    if (count == 0U || count > 0x100000000ULL) throw std::invalid_argument("greedy selection vocabulary size is invalid");
    std::uint32_t best = 0;
    float best_value = 0.0F;
    bool initialized = false;
    for (std::size_t i = 0; i < count; ++i) {
        const float v = logits[i];
        if (!std::isfinite(v)) throw std::invalid_argument("greedy selection logits must be finite");
        if (!initialized || v > best_value) { initialized = true; best_value = v; best = static_cast<std::uint32_t>(i); }
    }
    return best;
}

}  // namespace strix::cuda::qwen_hc
