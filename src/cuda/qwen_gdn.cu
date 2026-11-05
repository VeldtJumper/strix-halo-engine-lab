// CUDA GDN prefill core, bit-identical to the HIP tiled prefill (see include/strix/cuda_qwen_gdn.hpp).
//
// Every floating-point operation is one explicit IEEE intrinsic with the HIP operands; the HIP order comes from the gfx1151
// ISA (gfx1151 disassembly of hip_gdn_launcher, engine bf60d9d2):
//   gates  _ZN5strix12_GLOBAL__N_124gdn_prefill_gates_kernelILj8ELj2EEEvPKtmPKfS5_NS0_14GdnPrefillRowsEPKvNS0_15SmallWeightKindES8_S9_jj
//          (.s:113457): lane chains v_fmac_f32 / v_dual_fmac_f32 (:113561-113585); asm v_add_f32 trees (:113630-113879);
//          softplus: !(x > 20) (:114006), expf (v_mul 0x3fb8aa3b, v_fma, v_exp, v_ldexp), log1pf (v_frexp_mant :114032 ...),
//          a*softplus v_mul; beta: expf(-x) with negated constants (:114180-114185), 1 + e, IEEE division (v_div_fmas :114213);
//   conv   _ZN5strix12_GLOBAL__N_123gdn_prefill_conv_kernelILj8ELj32EEEvNS0_14GdnPrefillRowsEjPtPKvNS0_15SmallWeightKindEjjf
//          (.s:116154): q_scale = v_rsq_f32(128) with subnormal scaling (:117327); tap chain v_fma_f32 x,w,0 then v_fmac
//          (:117473-117503); silu division v_div_fixup (:117617); bf16 RNE v_bfe/v_add3/v_and; squares v_mul; key_sum
//          v_permlanex16 (:117659) + row_shl 8,4,2,1 (:117681) + readfirstlane (:117685); s_add_f32 eps (:117686);
//          v_rsq_f32 (:117694); (inv*value) then *q_scale (:117703);
//   lanes  _ZN5strix12_GLOBAL__N_124gdn_prefill_lanes_kernelILNS_19GdnRecurrentStorageE0ELj4ELj8ELj8EEEvNS0_14GdnPrefillRowsEjPfj
//          (.s:152370): decay = expf(gate) at commit (:152642-152660); v_mul S*decay, v_mul S*k (:152831-152899), in-lane
//          v_add tree, row_xmask:2 then :1 (:152913-152918), v_sub value-pred (:152921), v_mul *beta (:152924),
//          v_fmac S += k*delta (:152927-152971), then S*q and the same tree;
//   norm   _ZN5strix12_GLOBAL__N_128gdn_prefill_norm_rows_kernelILj8ELb0EEEvNS0_14GdnPrefillRowsEjjPKvNS0_15SmallWeightKindEjfPaPf
//          (.s:156427): key_sum (v_permlanex16 :156512, readfirstlane :156539); IEEE division by 128 (v_div_fixup :156644);
//          v_add eps (:156647); v_rsq_f32 (:156656); inv*v (:156667), *w (:156670), gate*(.) (:156674) with
//          gate = 1/(1+expf(-z)).
#include "strix/cuda_qwen_gdn.hpp"

#include <stdexcept>

namespace strix::cuda::qwen_gdn {
namespace {

namespace am = strix::cuda::amdmath;
constexpr unsigned kFull = 0xffffffffU;

__device__ __forceinline__ float bf16_to_float(std::uint16_t b) { return __uint_as_float(static_cast<std::uint32_t>(b) << 16); }
// hip_gdn_launcher.hip:307-317 (no NaN special case)
__device__ __forceinline__ std::uint16_t float_to_bf16(float v) {
    const std::uint32_t b = __float_as_uint(v);
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16);
}
__device__ __forceinline__ float round_bf16(float v) { return bf16_to_float(float_to_bf16(v)); }

// gdn_window_key_sum (hip_gdn_launcher.hip:2255-2262): lane l holds channels l, l+32, l+64, l+96 as p[0..3];
// s = (p0+p2)+(p1+p3); then lane l (< d) adds lane l+d for d = 16 (permlanex16), 8, 4, 2, 1 (DPP row_shl); the root is
// lane 0's value (readfirstlane). Lane 0's cone only reads lanes l+d with l+d inside the 16-lane row, so the xor
// butterfly gives lane 0 the same operands at every level, and (commutativity) every lane the same root bits.
__device__ __forceinline__ float key_sum(const float (&p)[4]) {
    float s = __fadd_rn(__fadd_rn(p[0], p[2]), __fadd_rn(p[1], p[3]));
#pragma unroll
    for (int d = 16; d; d >>= 1) s = __fadd_rn(s, __shfl_xor_sync(kFull, s, d));
    return __shfl_sync(kFull, s, 0);
}

// ---------------------------------------------------------------------------------------------------------------- K3
// gdn_prefill_gates_kernel (hip_gdn_launcher.hip:2750-2802). Per (row, head): lane l of 256 chains features l, l+256, ...
// (sum = fma(x, w, sum) from +0, v_fmac_f32); LDS tree strides 128 then 64 (asm v_add_f32), then per wave
// s[l] + s[l+32] and __shfl_down 16..1 (lane 0's cone == xor butterfly). Heads per block do not change any value.
template <unsigned R, unsigned H>
__global__ __launch_bounds__(256) void gates_kernel(const std::uint16_t* __restrict__ input, std::uint64_t input_stride,
    Rows rows, std::uint32_t count, const float* __restrict__ beta_w, const float* __restrict__ alpha_w,
    const float* __restrict__ dt, const float* __restrict__ a, am::Tables t) {
    constexpr unsigned S = R * H;
    __shared__ float scratch[S][256];
    const unsigned lane = threadIdx.x, first = blockIdx.y * R, head0 = blockIdx.x * H;
    const bool alpha = head0 >= kValueHeads;
    const unsigned out0 = head0 - (alpha ? kValueHeads : 0U);
    const float* weights = (alpha ? alpha_w : beta_w) + static_cast<std::uint64_t>(out0) * kHidden;
    float sum[H][R];
#pragma unroll
    for (unsigned h = 0; h < H; ++h)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) sum[h][r] = 0.0F;
    for (unsigned f = lane; f < kHidden; f += 256U) {
        float x[R];
#pragma unroll
        for (unsigned r = 0; r < R; ++r)
            x[r] = first + r < count ? bf16_to_float(input[static_cast<std::uint64_t>(first + r) * input_stride + f]) : 0.0F;
#pragma unroll
        for (unsigned h = 0; h < H; ++h) {
            const float w = weights[static_cast<std::uint64_t>(h) * kHidden + f];
#pragma unroll
            for (unsigned r = 0; r < R; ++r) sum[h][r] = __fmaf_rn(x[r], w, sum[h][r]);
        }
    }
#pragma unroll
    for (unsigned h = 0; h < H; ++h)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) scratch[h * R + r][lane] = sum[h][r];
    __syncthreads();
    for (unsigned stride = 128U; stride > 32U; stride >>= 1U) {
        if (lane < stride)
#pragma unroll
            for (unsigned i = 0; i < S; ++i) scratch[i][lane] = __fadd_rn(scratch[i][lane], scratch[i][lane + stride]);
        __syncthreads();
    }
    const unsigned wave = lane / 32U, l = lane % 32U;
    __shared__ float totals[S];
    for (unsigned i = wave; i < S; i += 8U) {
        float total = __fadd_rn(scratch[i][l], scratch[i][l + 32U]);
#pragma unroll
        for (int d = 16; d; d >>= 1) total = __fadd_rn(total, __shfl_xor_sync(kFull, total, d));
        if (l == 0U) totals[i] = total;
    }
    __syncthreads();
    // the transforms of the block's S outputs run on S threads at once (HIP: lane 0 of each wave in turn; same values)
    if (lane < S) {
        const unsigned i = lane, out = out0 + i / R, row = first + i % R;
        if (row < count) {
            const float raw = round_bf16(totals[i]);
            const std::uint64_t offset = static_cast<std::uint64_t>(row) * rows.stride + out;
            if (alpha) {
                // softplus (:1141-1151): x > 20 -> x; x < -20 -> expf(x); else log1pf(expf(x)). ISA: one expf shared.
                const float biased = __fadd_rn(raw, dt[out]);
                float sp;
                if (biased > 20.0F) sp = biased;
                else {
                    const float e = am::amd_expf(t, biased);
                    sp = biased < -20.0F ? e : am::amd_log1pf(t, e);
                }
                rows.gate[offset] = __fmul_rn(a[out], sp);
            } else {
                rows.beta[offset] = round_bf16(__fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(t, -raw))));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------- K4
// The HIP kernel walks all rows of a head in one block with an LDS ring. Rows are independent given their 3 predecessors'
// raw BF16 inputs, so here a block owns (head, 32-row chunk); a prepass saves each chunk's 3 preceding raw rows (from the
// workspace or the old history) and writes the new history, before any row is overwritten in place.
__device__ __forceinline__ void conv_prepass_body(const Rows& rows, std::uint32_t count, std::uint16_t* conv_state,
    std::uint16_t* __restrict__ boundary) {
    const unsigned ch = blockIdx.x * 256U + threadIdx.x;
    if (ch >= kConvWidth) return;
    std::uint16_t old[kConvKernel];
#pragma unroll
    for (unsigned i = 0; i < kConvKernel; ++i) old[i] = conv_state[static_cast<std::uint64_t>(ch) * kConvKernel + i];
    const auto raw = [&](int row) -> std::uint16_t {  // row >= -3; rows -3..-1 are taps 1..3 (:2823-2824)
        return row >= 0 ? float_to_bf16(rows.qkv[static_cast<std::uint64_t>(row) * rows.stride + ch]) : old[row + 4];
    };
    const unsigned chunks = (count + kConvChunk - 1U) / kConvChunk;
    for (unsigned c = 0; c < chunks; ++c)
#pragma unroll
        for (unsigned j = 0; j < 3U; ++j)
            boundary[(static_cast<std::uint64_t>(c) * 3U + j) * kConvWidth + ch] = raw(static_cast<int>(c * kConvChunk + j) - 3);
    // history = raw inputs of rows count-4 .. count-1 (:2858-2859; ring slot of row -k is tap 4-k)
#pragma unroll
    for (unsigned i = 0; i < kConvKernel; ++i)
        conv_state[static_cast<std::uint64_t>(ch) * kConvKernel + i] = raw(static_cast<int>(count) - 4 + static_cast<int>(i));
}
__global__ __launch_bounds__(256) void conv_prepass_kernel(Rows rows, std::uint32_t count, DevRef<std::uint16_t> conv_ref,
    std::uint16_t* __restrict__ boundary) {
    conv_prepass_body(rows, count, conv_ref.get(), boundary);
}
// Segment views: the rows of segment `seg` start at its row (every field shifts by the same stride).
__device__ __forceinline__ Rows segment_rows(const Rows& rows, std::uint32_t row) {
    const std::uint64_t o = static_cast<std::uint64_t>(row) * rows.stride;
    return Rows{rows.qkv + o, rows.z + o, rows.beta + o, rows.gate + o, rows.core + o, rows.stride};
}
// one grid slice (blockIdx.y) per segment; boundary of segment i at boundary + i * 3 * kConvWidth (one chunk: count <= 32)
__global__ __launch_bounds__(256) void conv_prepass_multi_kernel(Rows rows, Segments segs, std::uint16_t* __restrict__ boundary) {
    const Segment& s = segs.s[blockIdx.y];
    conv_prepass_body(segment_rows(rows, s.row), s.count, s.conv.get(), boundary + static_cast<std::uint64_t>(blockIdx.y) * 3U * kConvWidth);
}

// Conv history after the first `keep` rows of a window: the prepass rule (raw inputs of rows keep-4 .. keep-1, older taps
// from the history before the window), from the saved raw rows (launch_core_split raw_qkv).
__global__ __launch_bounds__(256) void conv_rollback_kernel(const float* __restrict__ raw_qkv, std::uint32_t keep,
    const std::uint16_t* __restrict__ before, std::uint16_t* __restrict__ conv_state) {
    const unsigned ch = blockIdx.x * 256U + threadIdx.x;
    if (ch >= kConvWidth) return;
#pragma unroll
    for (unsigned i = 0; i < kConvKernel; ++i) {
        const int row = static_cast<int>(keep) - 4 + static_cast<int>(i);
        conv_state[static_cast<std::uint64_t>(ch) * kConvKernel + i] = row >= 0
            ? float_to_bf16(raw_qkv[static_cast<std::uint64_t>(row) * kConvWidth + ch])
            : before[static_cast<std::uint64_t>(ch) * kConvKernel + static_cast<unsigned>(row + 4)];
    }
}

// gdn_prefill_conv_kernel body (:2838-2856): sum = fma(x0,w0,0), fma(x1,w1,.), ... oldest tap first; silu = sum/(1+expf(-sum));
// value = round_bf16(silu); q/k: inv = rsqrtf(key_sum(value^2) + eps), value = (value*inv)*q_scale, q_scale = rsqrtf(128) or 1.
__device__ __forceinline__ void conv_body(const Rows& rows, std::uint32_t count, const std::uint16_t* __restrict__ boundary,
    const float* __restrict__ conv_w, float epsilon, const am::Tables& t, unsigned chunk) {
    __shared__ std::uint16_t raw[kConvChunk + 3U][kHeadDim];
    const unsigned thread = threadIdx.x, lane = thread % 32U, wave = thread / 32U;
    const unsigned block = blockIdx.x, base = block * kHeadDim, first = chunk * kConvChunk;
    const bool is_qk = block < kKeyHeads * 2U, is_key = is_qk && block >= kKeyHeads;
    for (unsigned i = thread; i < (kConvChunk + 3U) * kHeadDim; i += 256U) {
        const unsigned rr = i / kHeadDim, c = i % kHeadDim;
        if (rr < 3U) raw[rr][c] = boundary[(static_cast<std::uint64_t>(chunk) * 3U + rr) * kConvWidth + base + c];
        else if (first + rr - 3U < count)
            raw[rr][c] = float_to_bf16(rows.qkv[static_cast<std::uint64_t>(first + rr - 3U) * rows.stride + base + c]);
    }
    float weight[4][kConvKernel];
#pragma unroll
    for (unsigned part = 0; part < 4U; ++part)
#pragma unroll
        for (unsigned i = 0; i < kConvKernel; ++i)
            weight[part][i] = conv_w[static_cast<std::uint64_t>(base + part * 32U + lane) * kConvKernel + i];
    const float q_scale = is_key ? 1.0F : am::amd_rsqrtf(t, static_cast<float>(kHeadDim));
    __syncthreads();
    for (unsigned local = wave; local < kConvChunk && first + local < count; local += 8U) {
        float value[4], square[4];
#pragma unroll
        for (unsigned part = 0; part < 4U; ++part) {
            const unsigned channel = part * 32U + lane;
            float sum = __fmaf_rn(bf16_to_float(raw[local][channel]), weight[part][0], 0.0F);
#pragma unroll
            for (unsigned i = 1; i < kConvKernel; ++i) sum = __fmaf_rn(bf16_to_float(raw[local + i][channel]), weight[part][i], sum);
            const float silu = __fdiv_rn(sum, __fadd_rn(1.0F, am::amd_expf(t, -sum)));
            value[part] = round_bf16(silu);
            square[part] = __fmul_rn(value[part], value[part]);
        }
        if (is_qk) {  // warp-uniform
            const float inverse = am::amd_rsqrtf(t, __fadd_rn(key_sum(square), epsilon));
#pragma unroll
            for (unsigned part = 0; part < 4U; ++part) value[part] = __fmul_rn(__fmul_rn(value[part], inverse), q_scale);
        }
        float* out = rows.qkv + static_cast<std::uint64_t>(first + local) * rows.stride + base + lane;
#pragma unroll
        for (unsigned part = 0; part < 4U; ++part) out[part * 32U] = value[part];
    }
}
__global__ __launch_bounds__(256) void conv_kernel(Rows rows, std::uint32_t count, const std::uint16_t* __restrict__ boundary,
    const float* __restrict__ conv_w, float epsilon, am::Tables t) {
    conv_body(rows, count, boundary, conv_w, epsilon, t, blockIdx.y);
}
__global__ __launch_bounds__(256) void conv_multi_kernel(Rows rows, Segments segs, const std::uint16_t* __restrict__ boundary,
    const float* __restrict__ conv_w, float epsilon, am::Tables t) {
    const Segment& s = segs.s[blockIdx.y];
    conv_body(segment_rows(rows, s.row), s.count, boundary + static_cast<std::uint64_t>(blockIdx.y) * 3U * kConvWidth, conv_w, epsilon,
        t, 0U);
}

// ---------------------------------------------------------------------------------------------------------------- K5
// gdn_prefill_lanes_kernel (:2896-2994, #pragma clang fp contract(off)). Lane `sub` of a G-lane column group owns keys
// sub + j*G. Per row, in this order: S = S*decay; p = S*k; pred = tree(p); delta = (v - pred)*beta; S = fma(k, delta, S);
// out = tree(S*q). Tree = the 128-leaf pairing of keys i, i+d for d = 64, 32, ..., 1: in-lane j, j+half (keys differ
// half*G) for half = K/2..1, then xor G/2..1 inside the group (HIP: DPP row_xmask G/2..1, every lane gets the root).
// decay = expf(gate) (computed when the row is staged, :2951). Rows strictly ascending; state read once, written once.
// Mapping (CUDA): a block owns 32 columns of one value head (32*G threads; 48*4 = 192 blocks); thread = column*G + sub.
// q/k of Chunk rows are staged in shared memory transposed by key residue (key[r][sub][j] = k[sub + j*G], pitch K+4 so
// the G distinct float4 reads of a warp hit distinct banks), v/beta/decay alongside; double-buffered through registers
// (the next chunk's global loads are in flight while the current chunk computes), as the HIP kernel stages them.
// decay = expf(gate) is computed by decay_kernel for all (row, head) before the recurrence (same value as the HIP commit-time
// expf; an fp64-based amd_expf inside the serial loop cost ~0.7 ms of 3.4 at T=8192). Rows are unrolled by 2 so the
// out tree of row r can overlap the delta chain of row r+1 (the arithmetic of each row is unchanged).
template <unsigned G> struct RecShape {
    static constexpr unsigned Columns = 32, Threads = 32 * G, K = kHeadDim / G, Pitch = K + 4, Chunk = G >= 16 ? 8 : 16;
    static constexpr unsigned KQ = Chunk * 64 / Threads;                                  // float4 of k|q per thread
    static constexpr unsigned VQ = (Chunk * Columns / 4 + Threads - 1) / Threads;          // float4 of v per thread
};
template <unsigned G> struct RecStage {
    using S = RecShape<G>;
    alignas(16) float key[S::Chunk][G][S::Pitch];
    alignas(16) float query[S::Chunk][G][S::Pitch];
    alignas(16) float value[S::Chunk][S::Columns];
    float beta[S::Chunk], decay[S::Chunk];
};

__global__ __launch_bounds__(256) void decay_kernel(Rows rows, std::uint32_t count, float* __restrict__ decay, am::Tables t) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x;
    if (i >= count * kValueHeads) return;
    decay[i] = am::amd_expf(t, rows.gate[static_cast<std::uint64_t>(i / kValueHeads) * rows.stride + i % kValueHeads]);
}
template <unsigned G>
__device__ __forceinline__ void recurrence_body(const Rows& rows, std::uint32_t count, const float* state_in, float* state_out,
    const float* __restrict__ decays) {
    using Sh = RecShape<G>;
    constexpr unsigned Columns = Sh::Columns, Threads = Sh::Threads, K = Sh::K, Chunk = Sh::Chunk, KQ = Sh::KQ, VQ = Sh::VQ;
    static_assert(Chunk * 64 % Threads == 0 && 2 * Chunk <= Threads);
    __shared__ RecStage<G> stage[2];
    const unsigned thread = threadIdx.x, sub = thread % G, local = thread / G;
    constexpr unsigned Blocks = kHeadDim / Columns;
    const unsigned value_head = blockIdx.x / Blocks, column_base = blockIdx.x % Blocks * Columns, column = column_base + local;
    const unsigned key_offset = value_head % kKeyHeads * kHeadDim;  // value head h reads key head h mod 16 (:2922)
    const unsigned value_offset = 2U * kKeyWidth + value_head * kHeadDim + column_base;
    float4 kq[KQ], vv[VQ];
    float scalar = 0.0F;
    const auto fetch = [&](unsigned first) {
#pragma unroll
        for (unsigned i = 0; i < KQ; ++i) {
            const unsigned q = thread + i * Threads, r = q / 64U, w = q % 64U;
            kq[i] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            if (first + r < count) {
                const float* row = rows.qkv + static_cast<std::uint64_t>(first + r) * rows.stride + key_offset;
                kq[i] = *reinterpret_cast<const float4*>(w < 32U ? row + kKeyWidth + w * 4U : row + (w - 32U) * 4U);
            }
        }
#pragma unroll
        for (unsigned i = 0; i < VQ; ++i) {
            const unsigned q = thread + i * Threads, r = q / (Columns / 4U);
            if (q < Chunk * Columns / 4U && first + r < count)
                vv[i] = *reinterpret_cast<const float4*>(rows.qkv + static_cast<std::uint64_t>(first + r) * rows.stride +
                    value_offset + q % (Columns / 4U) * 4U);
        }
        if (thread < 2U * Chunk && first + thread % Chunk < count)
            scalar = thread < Chunk ? rows.beta[static_cast<std::uint64_t>(first + thread) * rows.stride + value_head]
                                    : decays[(first + thread - Chunk) * kValueHeads + value_head];
    };
    const auto commit = [&](RecStage<G>& s) {
#pragma unroll
        for (unsigned i = 0; i < KQ; ++i) {
            const unsigned q = thread + i * Threads, r = q / 64U, w = q % 64U, key = (w % 32U) * 4U;
            float (&target)[G][Sh::Pitch] = w < 32U ? s.key[r] : s.query[r];
            const float parts[4] = {kq[i].x, kq[i].y, kq[i].z, kq[i].w};
#pragma unroll
            for (unsigned c = 0; c < 4U; ++c) target[(key + c) % G][(key + c) / G] = parts[c];
        }
#pragma unroll
        for (unsigned i = 0; i < VQ; ++i) {
            const unsigned q = thread + i * Threads;
            if (q < Chunk * Columns / 4U) *reinterpret_cast<float4*>(&s.value[q / (Columns / 4U)][q % (Columns / 4U) * 4U]) = vv[i];
        }
        if (thread < Chunk) s.beta[thread] = scalar;
        else if (thread < 2U * Chunk) s.decay[thread - Chunk] = scalar;
    };
    float S[K];
#pragma unroll
    for (unsigned j = 0; j < K; ++j)
        S[j] = state_in[(static_cast<std::uint64_t>(value_head) * kHeadDim + sub + j * G) * kHeadDim + column];
    const unsigned chunks = (count + Chunk - 1U) / Chunk;
    fetch(0U);
    commit(stage[0]);
    __syncthreads();
    for (unsigned c = 0; c < chunks; ++c) {
        const bool more = c + 1U < chunks;
        if (more) fetch((c + 1U) * Chunk);
        const RecStage<G>& s = stage[c & 1U];
        const unsigned first = c * Chunk, n = min(Chunk, count - first);
        float* core = rows.core + static_cast<std::uint64_t>(first) * rows.stride + value_head * kHeadDim + column;
#pragma unroll 2
        for (unsigned r = 0; r < n; ++r) {
            const float4* key = reinterpret_cast<const float4*>(s.key[r][sub]);
            const float4* query = reinterpret_cast<const float4*>(s.query[r][sub]);
            const float decay = s.decay[r], beta = s.beta[r], value = s.value[r][local];
            float p[K];
#pragma unroll
            for (unsigned m = 0; m < K / 4U; ++m) {
                const float4 k4 = key[m];
                const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
                for (unsigned c4 = 0; c4 < 4U; ++c4) {
                    const unsigned j = 4U * m + c4;
                    S[j] = __fmul_rn(S[j], decay);
                    p[j] = __fmul_rn(S[j], kk[c4]);
                }
            }
#pragma unroll
            for (unsigned half = K / 2U; half; half /= 2U)
#pragma unroll
                for (unsigned j = 0; j < half; ++j) p[j] = __fadd_rn(p[j], p[j + half]);
            float pred = p[0];
#pragma unroll
            for (unsigned m = G / 2U; m; m /= 2U) pred = __fadd_rn(pred, __shfl_xor_sync(kFull, pred, m));
            const float delta = __fmul_rn(__fsub_rn(value, pred), beta);
#pragma unroll
            for (unsigned m = 0; m < K / 4U; ++m) {
                const float4 k4 = key[m], q4 = query[m];
                const float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
                for (unsigned c4 = 0; c4 < 4U; ++c4) {
                    const unsigned j = 4U * m + c4;
                    S[j] = __fmaf_rn(kk[c4], delta, S[j]);
                    p[j] = __fmul_rn(S[j], qq[c4]);
                }
            }
#pragma unroll
            for (unsigned half = K / 2U; half; half /= 2U)
#pragma unroll
                for (unsigned j = 0; j < half; ++j) p[j] = __fadd_rn(p[j], p[j + half]);
            float out = p[0];
#pragma unroll
            for (unsigned m = G / 2U; m; m /= 2U) out = __fadd_rn(out, __shfl_xor_sync(kFull, out, m));
            if (sub == 0U) *core = out;
            core += rows.stride;
        }
        if (more) commit(stage[(c + 1U) & 1U]);
        __syncthreads();
    }
#pragma unroll
    for (unsigned j = 0; j < K; ++j)
        state_out[(static_cast<std::uint64_t>(value_head) * kHeadDim + sub + j * G) * kHeadDim + column] = S[j];
}
template <unsigned G>
__global__ __launch_bounds__(RecShape<G>::Threads) void recurrence_kernel(Rows rows, std::uint32_t count,
    DevRef<float> in_ref, DevRef<float> out_ref, const float* __restrict__ decays) {
    recurrence_body<G>(rows, count, in_ref.get(), out_ref.get(), decays);
}
// one grid slice (blockIdx.y) per segment: its rows, its state, its rows' decays (decay rows are window rows)
template <unsigned G>
__global__ __launch_bounds__(RecShape<G>::Threads) void recurrence_multi_kernel(Rows rows, Segments segs, const float* __restrict__ decays) {
    const Segment& s = segs.s[blockIdx.y];
    recurrence_body<G>(segment_rows(rows, s.row), s.count, s.state_in.get(), s.state_out.get(),
        decays + static_cast<std::uint64_t>(s.row) * kValueHeads);
}


// ---------------------------------------------------------------------------------------------------------------- K6
// gdn_prefill_norm_rows_kernel (:3000-3029): one warp per (row, head); inv = rsqrtf(key_sum(v^2)/128 + eps);
// result = ((v*inv)*w[c])*(1/(1+expf(-z))). A8: dense_a8_quantize_lane (hip_q8_rows_gemv.hpp:1147-1159, contract off).
__device__ __forceinline__ void quantize_lane(float x, std::int8_t* code, float* delta, unsigned lane) {
    const bool bad = __any_sync(kFull, !isfinite(x));
    float amax = fabsf(x);
#pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(kFull, amax, d));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float scale = bad ? __uint_as_float(0x7fc00000U) : amax == 0.0F || tiny ? 1.0F : __fdiv_rn(amax, 127.0F);
    const float rounded = bad || tiny ? 0.0F : rintf(__fdiv_rn(x, scale));
    *code = static_cast<std::int8_t>(static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded))));
    if (lane == 0U) *delta = scale;
}
template <bool A8>
__global__ __launch_bounds__(256) void norm_kernel(Rows rows, std::uint32_t count, const float* __restrict__ norm_w,
    float epsilon, am::Tables t, std::int8_t* codes, float* deltas) {
    const unsigned lane = threadIdx.x % 32U;
    const unsigned item = blockIdx.x * 8U + threadIdx.x / 32U;
    if (item >= count * kValueHeads) return;  // whole warp
    const unsigned row = item / kValueHeads, head = item % kValueHeads;
    const std::uint64_t base = static_cast<std::uint64_t>(row) * rows.stride + head * kHeadDim;
    float value[4], square[4];
#pragma unroll
    for (unsigned part = 0; part < 4U; ++part) {
        value[part] = rows.core[base + part * 32U + lane];
        square[part] = __fmul_rn(value[part], value[part]);
    }
    const float inv = am::amd_rsqrtf(t, __fadd_rn(__fdiv_rn(key_sum(square), static_cast<float>(kHeadDim)), epsilon));
#pragma unroll
    for (unsigned part = 0; part < 4U; ++part) {
        const unsigned channel = part * 32U + lane;
        const float gate = __fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(t, -rows.z[base + channel])));
        const float result = __fmul_rn(__fmul_rn(__fmul_rn(value[part], inv), norm_w[channel]), gate);
        rows.core[base + channel] = result;
        if constexpr (A8) {
            const std::uint64_t feature = head * kHeadDim + channel;
            quantize_lane(result, codes + static_cast<std::uint64_t>(row) * kValueWidth + feature,
                deltas + static_cast<std::uint64_t>(row / kA8Pitch) * (kValueWidth / 32U * kA8Pitch) + feature / 32U * kA8Pitch +
                    row % kA8Pitch, lane);
        }
    }
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("qwen_gdn ") + what + ": " + cudaGetErrorString(e));
}
void check_rows(const Rows& rows) {
    if (!rows.qkv || !rows.z || !rows.beta || !rows.gate || !rows.core || !rows.stride)
        throw std::invalid_argument("qwen_gdn: missing row workspace");
}

}  // namespace

std::size_t conv_scratch_bytes(std::uint32_t count) {
    return static_cast<std::size_t>((count + kConvChunk - 1U) / kConvChunk) * 3U * kConvWidth * sizeof(std::uint16_t);
}

void launch_gates(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    const Weights& w, const amdmath::Tables& t, cudaStream_t stream) {
    check_rows(rows);
    if (!count) return;
    constexpr unsigned R = 8, H = 4;
    gates_kernel<R, H><<<dim3(kValueHeads * 2U / H, (count + R - 1U) / R), 256, 0, stream>>>(input, input_stride, rows, count,
        w.beta, w.alpha, w.dt, w.a, t);
    check(cudaGetLastError(), "gates launch");
}

namespace {
void launch_conv_ref(const Rows& rows, std::uint32_t count, DevRef<std::uint16_t> conv_state, void* scratch, const Weights& w,
    const amdmath::Tables& t, cudaStream_t stream);
void launch_recurrence_ref(const Rows& rows, std::uint32_t count, DevRef<float> state_in, DevRef<float> state_out, void* scratch,
    const amdmath::Tables& t, cudaStream_t stream, Tuning tuning);
}  // namespace

void launch_conv(const Rows& rows, std::uint32_t count, std::uint16_t* conv_state, void* scratch, const Weights& w,
    const amdmath::Tables& t, cudaStream_t stream) {
    launch_conv_ref(rows, count, DevRef<std::uint16_t>{conv_state}, scratch, w, t, stream);
}

namespace {
void launch_conv_ref(const Rows& rows, std::uint32_t count, DevRef<std::uint16_t> conv_state, void* scratch, const Weights& w,
    const amdmath::Tables& t, cudaStream_t stream) {
    check_rows(rows);
    if (!count) return;
    if (rows.stride % 4U || reinterpret_cast<std::uintptr_t>(rows.qkv) % 16U)
        throw std::invalid_argument("qwen_gdn: qkv rows must be 16-byte aligned");
    auto* boundary = static_cast<std::uint16_t*>(scratch);
    conv_prepass_kernel<<<(kConvWidth + 255U) / 256U, 256, 0, stream>>>(rows, count, conv_state, boundary);
    check(cudaGetLastError(), "conv prepass launch");
    conv_kernel<<<dim3(kKeyHeads * 2U + kValueHeads, (count + kConvChunk - 1U) / kConvChunk), 256, 0, stream>>>(rows, count,
        boundary, w.conv, w.qk_l2_epsilon, t);
    check(cudaGetLastError(), "conv launch");
}
}  // namespace

void launch_conv_rollback(const float* raw_qkv, std::uint32_t keep, const std::uint16_t* before, std::uint16_t* conv_state,
    cudaStream_t stream) {
    conv_rollback_kernel<<<(kConvWidth + 255U) / 256U, 256, 0, stream>>>(raw_qkv, keep, before, conv_state);
    check(cudaGetLastError(), "conv rollback launch");
}

std::size_t decay_scratch_bytes(std::uint32_t count) { return static_cast<std::size_t>(count) * kValueHeads * sizeof(float); }
std::size_t core_scratch_bytes(std::uint32_t count) {
    return (conv_scratch_bytes(count) + 255U) / 256U * 256U + decay_scratch_bytes(count);
}

void launch_recurrence(const Rows& rows, std::uint32_t count, float* state, void* scratch, const amdmath::Tables& t,
    cudaStream_t stream, Tuning tuning) {
    launch_recurrence_split(rows, count, state, state, scratch, t, stream, tuning);
}

void launch_recurrence_split(const Rows& rows, std::uint32_t count, const float* state_in, float* state_out, void* scratch,
    const amdmath::Tables& t, cudaStream_t stream, Tuning tuning) {
    launch_recurrence_ref(rows, count, DevRef<float>{const_cast<float*>(state_in)}, DevRef<float>{state_out}, scratch, t, stream, tuning);
}

namespace {
void launch_recurrence_ref(const Rows& rows, std::uint32_t count, DevRef<float> state_in, DevRef<float> state_out, void* scratch,
    const amdmath::Tables& t, cudaStream_t stream, Tuning tuning) {
    check_rows(rows);
    if (!count) return;
    if (rows.stride % 4U || reinterpret_cast<std::uintptr_t>(rows.qkv) % 16U)
        throw std::invalid_argument("qwen_gdn: qkv rows must be 16-byte aligned");
    auto* decay = static_cast<float*>(scratch);
    decay_kernel<<<(count * kValueHeads + 255U) / 256U, 256, 0, stream>>>(rows, count, decay, t);
    check(cudaGetLastError(), "decay launch");
    const auto go = [&](auto kernel, unsigned g) {
        kernel<<<kValueHeads * (kHeadDim / 32U), 32U * g, 0, stream>>>(rows, count, state_in, state_out, decay);
    };
    switch (tuning.lanes) {
    case 2: go(recurrence_kernel<2>, 2); break;
    case 4: go(recurrence_kernel<4>, 4); break;
    case 8: go(recurrence_kernel<8>, 8); break;
    case 16: go(recurrence_kernel<16>, 16); break;
    default: throw std::invalid_argument("qwen_gdn: lanes must be 2, 4, 8 or 16");
    }
    check(cudaGetLastError(), "recurrence launch");
}
}  // namespace

void launch_norm(const Rows& rows, std::uint32_t count, const Weights& w, const amdmath::Tables& t, cudaStream_t stream,
    const A8Out* a8) {
    check_rows(rows);
    if (!count) return;
    const unsigned blocks = (count * kValueHeads + 7U) / 8U;
    if (a8) norm_kernel<true><<<blocks, 256, 0, stream>>>(rows, count, w.norm, w.rms_epsilon, t, a8->codes, a8->deltas);
    else norm_kernel<false><<<blocks, 256, 0, stream>>>(rows, count, w.norm, w.rms_epsilon, t, nullptr, nullptr);
    check(cudaGetLastError(), "norm launch");
}

void launch_core(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    std::uint16_t* conv_state, float* state, void* scratch, const Weights& w, const amdmath::Tables& t, cudaStream_t stream,
    const A8Out* a8, Tuning tuning) {
    launch_core_split(input, input_stride, rows, count, conv_state, state, state, nullptr, scratch, w, t, stream, a8, tuning);
}

void launch_core_split(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    std::uint16_t* conv_state, const float* state_in, float* state_out, float* raw_qkv, void* scratch, const Weights& w,
    const amdmath::Tables& t, cudaStream_t stream, const A8Out* a8, Tuning tuning) {
    launch_gates(input, input_stride, rows, count, w, t, stream);
    if (raw_qkv)
        check(cudaMemcpy2DAsync(raw_qkv, std::size_t{kConvWidth} * 4U, rows.qkv, rows.stride * 4U, std::size_t{kConvWidth} * 4U, count,
            cudaMemcpyDeviceToDevice, stream), "raw qkv copy");
    launch_conv(rows, count, conv_state, scratch, w, t, stream);
    launch_recurrence_split(rows, count, state_in, state_out,
        static_cast<unsigned char*>(scratch) + (conv_scratch_bytes(count) + 255U) / 256U * 256U, t, stream, tuning);
    launch_norm(rows, count, w, t, stream, a8);
}

std::size_t multi_scratch_bytes() {
    return (std::size_t{kMaxSegments} * 3U * kConvWidth * sizeof(std::uint16_t) + 255U) / 256U * 256U +
           decay_scratch_bytes(kMaxSegments * kConvChunk);
}

void launch_core_multi(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, const Segments& segs,
    float* raw_qkv, void* scratch, const Weights& w, const amdmath::Tables& t, cudaStream_t stream, Tuning tuning) {
    check_rows(rows);
    if (segs.n == 0 || segs.n > kMaxSegments) throw std::invalid_argument("qwen_gdn multi: 1..8 segments");
    std::uint32_t total = 0;
    for (std::uint32_t i = 0; i < segs.n; ++i) {
        if (segs.s[i].row != total || segs.s[i].count == 0 || segs.s[i].count > kConvChunk)
            throw std::invalid_argument("qwen_gdn multi: segments must be contiguous from row 0, 1..32 rows each");
        total += segs.s[i].count;
    }
    if (rows.stride % 4U || reinterpret_cast<std::uintptr_t>(rows.qkv) % 16U)
        throw std::invalid_argument("qwen_gdn: qkv rows must be 16-byte aligned");
    // K3 over all rows (row-wise)
    launch_gates(input, input_stride, rows, total, w, t, stream);
    if (raw_qkv)
        check(cudaMemcpy2DAsync(raw_qkv, std::size_t{kConvWidth} * 4U, rows.qkv, rows.stride * 4U, std::size_t{kConvWidth} * 4U, total,
            cudaMemcpyDeviceToDevice, stream), "raw qkv copy");
    // K4 per segment (its conv history)
    auto* boundary = static_cast<std::uint16_t*>(scratch);
    auto* decay = reinterpret_cast<float*>(static_cast<unsigned char*>(scratch) +
        (std::size_t{kMaxSegments} * 3U * kConvWidth * sizeof(std::uint16_t) + 255U) / 256U * 256U);
    conv_prepass_multi_kernel<<<dim3((kConvWidth + 255U) / 256U, segs.n), 256, 0, stream>>>(rows, segs, boundary);
    check(cudaGetLastError(), "conv prepass multi launch");
    conv_multi_kernel<<<dim3(kKeyHeads * 2U + kValueHeads, segs.n), 256, 0, stream>>>(rows, segs, boundary, w.conv, w.qk_l2_epsilon, t);
    check(cudaGetLastError(), "conv multi launch");
    // decay over all rows (row-wise), K5 per segment (its state)
    decay_kernel<<<(total * kValueHeads + 255U) / 256U, 256, 0, stream>>>(rows, total, decay, t);
    check(cudaGetLastError(), "decay launch");
    const auto go = [&](auto kernel, unsigned g) {
        kernel<<<dim3(kValueHeads * (kHeadDim / 32U), segs.n), 32U * g, 0, stream>>>(rows, segs, decay);
    };
    switch (tuning.lanes) {
    case 2: go(recurrence_multi_kernel<2>, 2); break;
    case 4: go(recurrence_multi_kernel<4>, 4); break;
    case 8: go(recurrence_multi_kernel<8>, 8); break;
    case 16: go(recurrence_multi_kernel<16>, 16); break;
    default: throw std::invalid_argument("qwen_gdn: lanes must be 2, 4, 8 or 16");
    }
    check(cudaGetLastError(), "recurrence multi launch");
    // K6 over all rows (row-wise)
    launch_norm(rows, total, w, t, stream, nullptr);
}

void launch_core_split_ref(const std::uint16_t* input, std::uint64_t input_stride, const Rows& rows, std::uint32_t count,
    DevRef<std::uint16_t> conv_state, DevRef<float> state_in, DevRef<float> state_out, float* raw_qkv, void* scratch,
    const Weights& w, const amdmath::Tables& t, cudaStream_t stream, const A8Out* a8, Tuning tuning) {
    launch_gates(input, input_stride, rows, count, w, t, stream);
    if (raw_qkv)
        check(cudaMemcpy2DAsync(raw_qkv, std::size_t{kConvWidth} * 4U, rows.qkv, rows.stride * 4U, std::size_t{kConvWidth} * 4U, count,
            cudaMemcpyDeviceToDevice, stream), "raw qkv copy");
    launch_conv_ref(rows, count, conv_state, scratch, w, t, stream);
    launch_recurrence_ref(rows, count, state_in, state_out,
        static_cast<unsigned char*>(scratch) + (conv_scratch_bytes(count) + 255U) / 256U * 256U, t, stream, tuning);
    launch_norm(rows, count, w, t, stream, a8);
}

}  // namespace strix::cuda::qwen_gdn
