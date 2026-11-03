// GLM-5.3-Flash chunked prefill, Tier A kernels (see glm_prefill.hpp). Each kernel repeats the per-row expressions,
// chains and trees of the decode kernels in glm_tier_a.hip / glm_dsa.hip; only the work placement differs. Gate:
// tests/hip_glm_prefill_tests.hip (prefill == decode row by row, bitwise). Build with -ffp-contract=off -fno-fast-math
// -fno-gpu-flush-denormals-to-zero.
#include "glm/prefill/glm_prefill.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::glm::pf {

namespace {

using dev::DsaSlot;
using dev::IndexerDev;
using dev::KdaDev;

// Constant-address-space views for wave-uniform operands read through scalar loads (s_load) into SGPRs.
using ConstF = const __attribute__((address_space(4))) float;
typedef float SVec16 __attribute__((ext_vector_type(16)));
using ConstV16 = const __attribute__((address_space(4))) SVec16;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template <unsigned L> __device__ __forceinline__ float lds_tree(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = L / 2; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}
// lds_tree<256>'s sum for lane 0 only, one barrier: wave 0 forms levels 128 / 64 / 32 from LDS in the same pairing, then
// levels 16 .. 1 by __shfl_down (lane l < h receives lane l + h). Other lanes return an unspecified value.
__device__ __forceinline__ float lds_tree256_lane0(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    float w = 0.0F;
    if (l < 32) {
        const float t0 = s[l] + s[l + 128], t1 = s[l + 64] + s[l + 192], t2 = s[l + 32] + s[l + 160], t3 = s[l + 96] + s[l + 224];
        w = (t0 + t1) + (t2 + t3);
#pragma unroll
        for (unsigned h = 16; h > 0; h >>= 1U) w = w + __shfl_down(w, h, 32);
    }
    return w;
}

// ---------------------------------------------------------------- KDA
// 4a for C rows of one slot: grid (head, row). The conv of row r reads the pre-conv inputs of positions t - 3 .. t - 1
// from the chunk (row r - j >= 0) or from the slot's conv state (oldest first); the state itself is written afterwards
// by kda_conv_commit_kernel. Expressions = kda_prep_kernel.
__global__ __launch_bounds__(128) void kda_prep_rows_kernel(KdaDev k, const float* qkv, const float* f, const float* b,
                                                            const float* conv_state, float* scratch, float* beta) {
    __shared__ float s[128];
    const unsigned h = blockIdx.x, r = blockIdx.y, i = threadIdx.x, H = k.heads, C = H * kKdaDim;
    float out[3];
    for (unsigned part = 0; part < 3; ++part) {
        const unsigned ch = part * C + h * kKdaDim + i;
        const float* st = conv_state + static_cast<std::size_t>(ch) * 3;
        const std::uint16_t* cw = k.conv_w + static_cast<std::size_t>(ch) * kConvK;
        float x[3];
        for (unsigned j = 0; j < 3; ++j) {  // x[j] = input of position t - 3 + j
            const int m = static_cast<int>(r) - 3 + static_cast<int>(j);
            x[j] = m >= 0 ? qkv[static_cast<std::size_t>(m) * 3 * C + ch] : st[3 + m];
        }
        const float nw = qkv[static_cast<std::size_t>(r) * 3 * C + ch];
        float acc = __builtin_fmaf(bf16_to_float(cw[0]), x[0], 0.0F);
        acc = __builtin_fmaf(bf16_to_float(cw[1]), x[1], acc);
        acc = __builtin_fmaf(bf16_to_float(cw[2]), x[2], acc);
        acc = __builtin_fmaf(bf16_to_float(cw[3]), nw, acc);
        out[part] = silu(acc);
    }
    const float sq = lds_tree<128>(s, __builtin_fmaf(out[0], out[0], 0.0F));
    const float sk = lds_tree<128>(s, __builtin_fmaf(out[1], out[1], 0.0F));
    const float nq = declared_sqrt(sq + kL2Eps), nk = declared_sqrt(sk + kL2Eps);
    float* sc = scratch + (static_cast<std::size_t>(r) * H + h) * 4 * kKdaDim;
    sc[i] = (out[0] / nq) * kInvSqrt128;
    sc[kKdaDim + i] = out[1] / nk;
    sc[2 * kKdaDim + i] = out[2];
    const float expA = declared_exp(k.A_log[h]);
    sc[3 * kKdaDim + i] = kGateLow * sigmoid(expA * (f[static_cast<std::size_t>(r) * C + h * kKdaDim + i] + k.dt_bias[h * kKdaDim + i]));
    if (i == 0) beta[static_cast<std::size_t>(r) * H + h] = sigmoid(b[static_cast<std::size_t>(r) * H + h]);
}

// Conv state after row C - 1: the inputs of the last three positions, oldest first.
__global__ void kda_conv_commit_kernel(const float* qkv, float* conv_state, unsigned channels, unsigned C) {
    const unsigned ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= channels) return;
    float* st = conv_state + static_cast<std::size_t>(ch) * 3;
    const float o0 = st[0], o1 = st[1], o2 = st[2];
#pragma unroll
    for (unsigned j = 0; j < 3; ++j) {
        const int m = static_cast<int>(C) - 3 + static_cast<int>(j);
        st[j] = m >= 0 ? qkv[static_cast<std::size_t>(m) * channels + ch] : (m == -3 ? o0 : (m == -2 ? o1 : o2));
    }
}

// 4b over C tokens, token-sequential: grid (head, value tile of kScanV columns); a quad of lanes owns column v and
// holds S[:, v] (128 fp32) in registers for the whole chunk: lane q of the quad owns keys g * 32 + q * 8 + j
// (g = 0..3, j = 0..7). Per token, the decode kda_update_kernel expressions for column v:
//   sp_k = S_kv * dec_k; u = lds_tree<128> pairing of (+0 + sp_k * k_k); d = (v - u) * beta; S_kv = fmaf(k_k, d, sp_k);
//   o_v = the same tree of (+0 + S_kv * q_k).
// Tree pairing as lds_tree<128>: levels 64 and 32 pair keys inside a lane; level 16 pairs lanes q and q ^ 2, level 8
// lanes q and q ^ 1 (shuffles; IEEE addition is commutative, so both lanes of a pair get the same bits); levels 4 .. 1
// inside each lane.
constexpr unsigned kScanV = 64, kScanTB = 16, kScanLanes = 4, kOwn = kKdaDim / kScanLanes;  // 32 keys per lane
__device__ __forceinline__ unsigned scan_key(unsigned q, unsigned i) {  // own key i in [0, 32)
    return (i / 8) * 32 + q * 8 + (i % 8);
}
__device__ __forceinline__ float scan_tree(const float* st, const float* w, unsigned q) {
    float tr[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)  // level 64: (l, l + 64) for l = g * 32 + q * 8 + j, g = 0, 1
        tr[i] = (0.0F + st[i] * w[scan_key(q, i)]) + (0.0F + st[i + 16] * w[scan_key(q, i + 16)]);
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) tr[j] = tr[j] + tr[j + 8];  // level 32: (l, l + 32), l = q * 8 + j
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) tr[j] = tr[j] + __shfl_xor(tr[j], 2);  // level 16
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) tr[j] = tr[j] + __shfl_xor(tr[j], 1);  // level 8
#pragma unroll
    for (unsigned hh = 4; hh > 0; hh >>= 1U)
#pragma unroll
        for (unsigned l = 0; l < hh; ++l) tr[l] = tr[l] + tr[l + hh];
    return tr[0];
}
__global__ __launch_bounds__(kScanLanes * kScanV) void kda_scan_kernel(unsigned H, float* S, const float* scratch,
                                                                       const float* beta, float* o, unsigned* flag,
                                                                       unsigned C) {
    __shared__ float sq[kScanTB][kKdaDim], sk[kScanTB][kKdaDim], sd[kScanTB][kKdaDim];
    const unsigned h = blockIdx.x, tid = threadIdx.x, q = tid % kScanLanes, v = blockIdx.y * kScanV + tid / kScanLanes;
    float* Sh = S + static_cast<std::size_t>(h) * kKdaDim * kKdaDim;
    float st[kOwn];
#pragma unroll
    for (unsigned i = 0; i < kOwn; ++i) st[i] = Sh[scan_key(q, i) * kKdaDim + v];
    bool bad = false;
    for (unsigned b0 = 0; b0 < C; b0 += kScanTB) {
        const unsigned nb = C - b0 < kScanTB ? C - b0 : kScanTB;
        __syncthreads();
        for (unsigned idx = tid; idx < nb * kKdaDim; idx += kScanLanes * kScanV) {
            const unsigned j = idx / kKdaDim, kk = idx % kKdaDim;
            const float* sc = scratch + (static_cast<std::size_t>(b0 + j) * H + h) * 4 * kKdaDim;
            sq[j][kk] = sc[kk];
            sk[j][kk] = sc[kKdaDim + kk];
            sd[j][kk] = declared_exp(sc[3 * kKdaDim + kk]);
        }
        __syncthreads();
        for (unsigned j = 0; j < nb; ++j) {
            const std::size_t rh = static_cast<std::size_t>(b0 + j) * H + h;
            const float vv = scratch[rh * 4 * kKdaDim + 2 * kKdaDim + v];
            const float bt = beta[rh];
#pragma unroll
            for (unsigned i = 0; i < kOwn; ++i) st[i] = st[i] * sd[j][scan_key(q, i)];
            const float dlt = (vv - scan_tree(st, sk[j], q)) * bt;
#pragma unroll
            for (unsigned i = 0; i < kOwn; ++i) {
                st[i] = __builtin_fmaf(sk[j][scan_key(q, i)], dlt, st[i]);
                bad = bad || !is_finite(st[i]);
            }
            const float ov = scan_tree(st, sq[j], q);
            if (q == 0) o[rh * kKdaDim + v] = ov;
        }
    }
#pragma unroll
    for (unsigned i = 0; i < kOwn; ++i) Sh[scan_key(q, i) * kKdaDim + v] = st[i];
    if (bad && flag) atomicOr(flag, 1U);
}

// Scan v3: v1's grid, per-lane chains and tree with the two cross-lane levels (q ^ 2, q ^ 1) read through quad DPP
// moves instead of ds_bpermute shuffles (same operand, same order: own value + partner's), which takes four LDS round
// trips off every token's serial path. Measured 1.86-1.93 -> 1.24-1.28 us/row per KDA layer at t0 0 / 4096 / 28672
// (32 columns per CTA, twice the CTAs, was slower: 2.02-2.07).
template <int Ctrl> __device__ __forceinline__ float quad_dpp(float x) {  // quad_perm: 0xB1 = lane ^ 1, 0x4E = lane ^ 2
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), Ctrl, 0xF, 0xF, true));
}
__device__ __forceinline__ float scan_tree_dpp(const float* st, const float* w, unsigned q) {
    float tr[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)
        tr[i] = (0.0F + st[i] * w[scan_key(q, i)]) + (0.0F + st[i + 16] * w[scan_key(q, i + 16)]);
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) tr[j] = tr[j] + tr[j + 8];
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) tr[j] = tr[j] + quad_dpp<0x4E>(tr[j]);  // level 16
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) tr[j] = tr[j] + quad_dpp<0xB1>(tr[j]);  // level 8
#pragma unroll
    for (unsigned hh = 4; hh > 0; hh >>= 1U)
#pragma unroll
        for (unsigned l = 0; l < hh; ++l) tr[l] = tr[l] + tr[l + hh];
    return tr[0];
}
__global__ __launch_bounds__(kScanLanes * kScanV) void kda_scan_v3_kernel(unsigned H, float* S, const float* scratch,
                                                                          const float* beta, float* o, unsigned* flag, unsigned C) {
    __shared__ float sq[kScanTB][kKdaDim], sk[kScanTB][kKdaDim], sd[kScanTB][kKdaDim];
    const unsigned h = blockIdx.x, tid = threadIdx.x, q = tid % kScanLanes, v = blockIdx.y * kScanV + tid / kScanLanes;
    float* Sh = S + static_cast<std::size_t>(h) * kKdaDim * kKdaDim;
    float st[kOwn];
#pragma unroll
    for (unsigned i = 0; i < kOwn; ++i) st[i] = Sh[scan_key(q, i) * kKdaDim + v];
    bool bad = false;
    for (unsigned b0 = 0; b0 < C; b0 += kScanTB) {
        const unsigned nb = C - b0 < kScanTB ? C - b0 : kScanTB;
        __syncthreads();
        for (unsigned idx = tid; idx < nb * kKdaDim; idx += kScanLanes * kScanV) {
            const unsigned j = idx / kKdaDim, kk = idx % kKdaDim;
            const float* sc = scratch + (static_cast<std::size_t>(b0 + j) * H + h) * 4 * kKdaDim;
            sq[j][kk] = sc[kk];
            sk[j][kk] = sc[kKdaDim + kk];
            sd[j][kk] = declared_exp(sc[3 * kKdaDim + kk]);
        }
        __syncthreads();
        for (unsigned j = 0; j < nb; ++j) {
            const std::size_t rh = static_cast<std::size_t>(b0 + j) * H + h;
            const float vv = scratch[rh * 4 * kKdaDim + 2 * kKdaDim + v];
            const float bt = beta[rh];
#pragma unroll
            for (unsigned i = 0; i < kOwn; ++i) st[i] = st[i] * sd[j][scan_key(q, i)];
            const float dlt = (vv - scan_tree_dpp(st, sk[j], q)) * bt;
#pragma unroll
            for (unsigned i = 0; i < kOwn; ++i) {
                st[i] = __builtin_fmaf(sk[j][scan_key(q, i)], dlt, st[i]);
                bad = bad || !is_finite(st[i]);
            }
            const float ov = scan_tree_dpp(st, sq[j], q);
            if (q == 0) o[rh * kKdaDim + v] = ov;
        }
    }
#pragma unroll
    for (unsigned i = 0; i < kOwn; ++i) Sh[scan_key(q, i) * kKdaDim + v] = st[i];
    if (bad && flag) atomicOr(flag, 1U);
}

// Scan v5: v3's per-token work with the batch fill off the serial path: the next batch's q / k / g are loaded into
// registers before the current batch's tokens run and stored (decay widened) into the other half of a double LDS
// buffer after them, so one barrier per batch remains and the global latency hides behind the token loop. 8 tokens per
// buffer keep LDS at v3's 24 KB. Same values and expressions as v3. Measured 1.23-1.25 -> 1.18-1.20 us/row per KDA layer
// at t0 0 / 4096 / 28672 (-3.7 %). (Staging v and beta in LDS instead was slower: 1.41-2.24.)
constexpr unsigned kScan5TB = 8, kScan5Per = kScan5TB * kKdaDim / (kScanLanes * kScanV);  // 4 elements per thread
__global__ __launch_bounds__(kScanLanes * kScanV) void kda_scan_v5_kernel(unsigned H, float* S, const float* scratch,
                                                                          const float* beta, float* o, unsigned* flag, unsigned C) {
    __shared__ float sq[2][kScan5TB][kKdaDim], sk[2][kScan5TB][kKdaDim], sd[2][kScan5TB][kKdaDim];
    const unsigned h = blockIdx.x, tid = threadIdx.x, q = tid % kScanLanes, v = blockIdx.y * kScanV + tid / kScanLanes;
    float* Sh = S + static_cast<std::size_t>(h) * kKdaDim * kKdaDim;
    float st[kOwn];
#pragma unroll
    for (unsigned i = 0; i < kOwn; ++i) st[i] = Sh[scan_key(q, i) * kKdaDim + v];
    float rq[kScan5Per] = {}, rk[kScan5Per] = {}, rg[kScan5Per] = {};
    auto load = [&](unsigned b0) {
#pragma unroll
        for (unsigned m = 0; m < kScan5Per; ++m) {
            const unsigned idx = tid + kScanLanes * kScanV * m, j = idx / kKdaDim, kk = idx % kKdaDim;
            if (b0 + j < C) {
                const float* sc = scratch + (static_cast<std::size_t>(b0 + j) * H + h) * 4 * kKdaDim;
                rq[m] = sc[kk];
                rk[m] = sc[kKdaDim + kk];
                rg[m] = sc[3 * kKdaDim + kk];
            }
        }
    };
    auto store = [&](unsigned buf) {  // entries past C hold stale values and are never read
#pragma unroll
        for (unsigned m = 0; m < kScan5Per; ++m) {
            const unsigned idx = tid + kScanLanes * kScanV * m, j = idx / kKdaDim, kk = idx % kKdaDim;
            sq[buf][j][kk] = rq[m];
            sk[buf][j][kk] = rk[m];
            sd[buf][j][kk] = declared_exp(rg[m]);
        }
    };
    load(0);
    store(0);
    __syncthreads();
    bool bad = false;
    unsigned cur = 0;
    for (unsigned b0 = 0; b0 < C; b0 += kScan5TB, cur ^= 1U) {
        const unsigned nb = C - b0 < kScan5TB ? C - b0 : kScan5TB;
        const bool next = b0 + kScan5TB < C;
        if (next) load(b0 + kScan5TB);
        for (unsigned j = 0; j < nb; ++j) {
            const std::size_t rh = static_cast<std::size_t>(b0 + j) * H + h;
            const float vv = scratch[rh * 4 * kKdaDim + 2 * kKdaDim + v];
            const float bt = beta[rh];
#pragma unroll
            for (unsigned i = 0; i < kOwn; ++i) st[i] = st[i] * sd[cur][j][scan_key(q, i)];
            const float dlt = (vv - scan_tree_dpp(st, sk[cur][j], q)) * bt;
#pragma unroll
            for (unsigned i = 0; i < kOwn; ++i) {
                st[i] = __builtin_fmaf(sk[cur][j][scan_key(q, i)], dlt, st[i]);
                bad = bad || !is_finite(st[i]);
            }
            const float ov = scan_tree_dpp(st, sq[cur][j], q);
            if (q == 0) o[rh * kKdaDim + v] = ov;
        }
        if (next) store(cur ^ 1U);
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < kOwn; ++i) Sh[scan_key(q, i) * kKdaDim + v] = st[i];
    if (bad && flag) atomicOr(flag, 1U);
}

// 4c: = kda_out_kernel (row-parallel already).
__global__ __launch_bounds__(128) void kda_out_rows_kernel(KdaDev k, const float* o, const float* gate, float* y) {
    __shared__ float s[128];
    const unsigned h = blockIdx.x, r = blockIdx.y, i = threadIdx.x, H = k.heads;
    const std::size_t at = (static_cast<std::size_t>(r) * H + h) * kKdaDim + i;
    const float v = o[at];
    const float ms = lds_tree<128>(s, __builtin_fmaf(v, v, 0.0F)) / static_cast<float>(kKdaDim);
    const float rr = inv_sqrt(ms + kEps);
    y[at] = ((v * rr) * bf16_to_float(k.o_norm[i])) * sigmoid(gate[at]);
}

// ---------------------------------------------------------------- DSA append
// dsa_store_kernel without the pooling (row-parallel: the pool reads other rows), position t0 + r.
__global__ __launch_bounds__(128) void dsa_store_rows_kernel(DsaSlot sl, unsigned t0, const float* latent, IndexerDev w,
                                                             const float* key, const float* gs, unsigned* flag) {
    __shared__ float s[128];
    const unsigned r = blockIdx.x, i = threadIdx.x, t = t0 + r;
    const float k = key[static_cast<std::size_t>(r) * kIdxDim + i];
    const float mu = lds_tree<128>(s, 0.0F + k) / static_cast<float>(kIdxDim);
    const float d = k - mu;
    const float var = lds_tree<128>(s, __builtin_fmaf(d, d, 0.0F)) / static_cast<float>(kIdxDim);
    const float rr = inv_sqrt(var + kLnEps);
    const std::uint16_t kb = float_to_bf16((d * rr) * f16_to_float(w.k_norm_w[i]) + f16_to_float(w.k_norm_b[i]));
    const std::uint16_t gb = float_to_bf16(gs[static_cast<std::size_t>(r) * kIdxDim + i]);
    if (!bf16_is_finite(kb) || !bf16_is_finite(gb)) atomicOr(flag, 1U);
    sl.ikey[static_cast<std::size_t>(t) * kIdxDim + i] = kb;
    sl.igate[static_cast<std::size_t>(t) * kIdxDim + i] = gb;
    for (unsigned l = i; l < kKvLora; l += 128) {
        const std::uint16_t lb = float_to_bf16(latent[static_cast<std::size_t>(r) * kKvLora + l]);
        if (!bf16_is_finite(lb)) atomicOr(flag, 1U);
        sl.latent[static_cast<std::size_t>(t) * kKvLora + l] = lb;
    }
}
// R12 for pools p0 + blockIdx.x (channel = lane), = the pooling branch of dsa_store_kernel.
__global__ __launch_bounds__(128) void dsa_pool_kernel(DsaSlot sl, unsigned p0, IndexerDev w, unsigned* flag) {
    const unsigned p = p0 + blockIdx.x, i = threadIdx.x, t0 = p * kKpool;
    float l4[kKpool], kv[kKpool], e[kKpool], pr[kKpool];
    for (unsigned j = 0; j < kKpool; ++j) {
        l4[j] = bf16_to_float(sl.igate[static_cast<std::size_t>(t0 + j) * kIdxDim + i]) + w.ape[j * kIdxDim + i];
        kv[j] = bf16_to_float(sl.ikey[static_cast<std::size_t>(t0 + j) * kIdxDim + i]);
    }
    float m = l4[0];
    for (unsigned j = 1; j < kKpool; ++j) m = l4[j] > m ? l4[j] : m;
    for (unsigned j = 0; j < kKpool; ++j) e[j] = declared_exp(l4[j] - m);
    const float sum = ((e[0] + e[1]) + e[2]) + e[3];
    for (unsigned j = 0; j < kKpool; ++j) pr[j] = e[j] / sum;
    const float K = ((pr[0] * kv[0] + pr[1] * kv[1]) + pr[2] * kv[2]) + pr[3] * kv[3];
    if (!is_finite(K)) atomicOr(flag, 1U);
    sl.pooled[static_cast<std::size_t>(p) * kIdxDim + i] = K;
}

// ---------------------------------------------------------------- indexer (R11, R5)
__global__ void scale_kernel(float* v, unsigned n, float c) {
    const unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) v[i] = v[i] * c;
}
// Scores of a tile of kScoreRows rows x 128 pools: thread = pool p, its pooled key in registers; the row's qI is staged
// in LDS. Per (row, pool) = idx_score_kernel's chain. Rows with P_t <= 512 select every pool and need no scores.
constexpr unsigned kScoreRows = 16;
__global__ __launch_bounds__(128) void idx_score_rows_kernel(const float* pooled, unsigned t0, unsigned C, const float* qI,
                                                             const float* wh, std::uint32_t* keys, unsigned cap) {
    __shared__ float q[kIdxHeads * kIdxDim];
    __shared__ float w[kIdxHeads];
    const unsigned p = blockIdx.x * 128 + threadIdx.x, r0 = blockIdx.y * kScoreRows;
    const unsigned Pmax = (t0 + (r0 + kScoreRows < C ? r0 + kScoreRows : C)) / kKpool;  // pools of the tile's last row
    if (blockIdx.x * 128 >= Pmax) return;  // uniform per block
    float K[kIdxDim];
    const bool have = p < Pmax;
#pragma unroll
    for (unsigned i = 0; i < kIdxDim; ++i) K[i] = have ? pooled[static_cast<std::size_t>(p) * kIdxDim + i] : 0.0F;
    for (unsigned r = r0; r < r0 + kScoreRows && r < C; ++r) {
        const unsigned P = (t0 + r + 1) / kKpool;
        if (P <= kIdxPools) continue;  // uniform per block
        __syncthreads();
        for (unsigned i = threadIdx.x; i < kIdxHeads * kIdxDim; i += 128) q[i] = qI[static_cast<std::size_t>(r) * kIdxHeads * kIdxDim + i];
        if (threadIdx.x < kIdxHeads) w[threadIdx.x] = wh[static_cast<std::size_t>(r) * kIdxHeads + threadIdx.x];
        __syncthreads();
        if (p >= P) continue;
        float acc = 0.0F;
        for (unsigned h = 0; h < kIdxHeads; ++h) {
            float dot = 0.0F;
#pragma unroll
            for (unsigned i = 0; i < kIdxDim; ++i) dot = __builtin_fmaf(q[h * kIdxDim + i], K[i], dot);
            const float sc = dot * kInvSqrt128;
            acc = __builtin_fmaf(w[h], sc > 0.0F ? sc : 0.0F, acc);
        }
        keys[static_cast<std::size_t>(r) * cap + p] = ordered_key(acc);
    }
}

// Scores v2: CTA = 64 pools x 4 rows, all 32 heads. Phase 1: the per-(row, head, pool) dot chains (i = 0..127
// ascending from +0: idx_score_rows_kernel's fmaf(q, K, dot)) as a register-blocked tile, thread = 4 pools x 8 (row,
// head) pairs, the pooled keys [i][pool] and qI [i][(row, head)] staged in LDS per 32-dim slice. Phase 2: per (row,
// pool) the head chain acc = fmaf(w[h], relu(dot * kInvSqrt128), acc) for h = 0..31 ascending from +0, from the dots
// staged in LDS. Same bits as the v1 kernel; rows with P_t <= 512 and pools p >= P_t are skipped as there.
constexpr unsigned kIs2Pools = 64, kIs2Rows = 4, kIs2K = 32, kIs2RH = kIs2Rows * kIdxHeads;  // 128 (row, head) pairs
__global__ __launch_bounds__(256) void idx_score_rows_v2_kernel(const float* pooled, unsigned t0, unsigned C, const float* qI,
                                                                const float* wh, std::uint32_t* keys, unsigned cap) {
    __shared__ alignas(16) float ks[kIs2K][kIs2Pools];  // 8 KB
    __shared__ alignas(16) float qs[kIs2K][kIs2RH];     // 16 KB
    __shared__ float dots[kIs2RH][kIs2Pools + 1];        // 33 KB (+1 column against bank conflicts)
    __shared__ float ws[kIs2RH];
    const unsigned p0 = blockIdx.x * kIs2Pools, r0 = blockIdx.y * kIs2Rows;
    const unsigned rlast = (r0 + kIs2Rows < C ? r0 + kIs2Rows : C) - 1;
    const unsigned Pmax = (t0 + rlast + 1) / kKpool;  // pools of the tile's last row (earlier rows have fewer)
    if (p0 >= Pmax || Pmax <= kIdxPools) return;    // uniform per block
    const unsigned tid = threadIdx.x, pg = tid % 16, rg = tid / 16;  // pools 4 pg .. + 3, (row, head) pairs 8 rg .. + 7
    float acc[4][8];
#pragma unroll
    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
        for (unsigned b = 0; b < 8; ++b) acc[a][b] = 0.0F;
    for (unsigned k0 = 0; k0 < kIdxDim; k0 += kIs2K) {
        __syncthreads();  // previous slice consumed
        for (unsigned x = tid; x < (kIs2K / 4) * kIs2Pools; x += 256) {  // lanes walk pools: conflict-free stores
            const unsigned pp = x % kIs2Pools, i4 = x / kIs2Pools;
            float4 v{0.0F, 0.0F, 0.0F, 0.0F};
            if (p0 + pp < Pmax) v = *reinterpret_cast<const float4*>(pooled + static_cast<std::size_t>(p0 + pp) * kIdxDim + k0 + 4 * i4);
            ks[4 * i4 + 0][pp] = v.x;
            ks[4 * i4 + 1][pp] = v.y;
            ks[4 * i4 + 2][pp] = v.z;
            ks[4 * i4 + 3][pp] = v.w;
        }
        for (unsigned x = tid; x < (kIs2K / 4) * kIs2RH; x += 256) {  // lanes walk (row, head): conflict-free stores
            const unsigned rh = x % kIs2RH, i4 = x / kIs2RH, r = r0 + rh / kIdxHeads, h = rh % kIdxHeads;
            float4 v{0.0F, 0.0F, 0.0F, 0.0F};
            if (r < C) v = *reinterpret_cast<const float4*>(qI + (static_cast<std::size_t>(r) * kIdxHeads + h) * kIdxDim + k0 + 4 * i4);
            qs[4 * i4 + 0][rh] = v.x;
            qs[4 * i4 + 1][rh] = v.y;
            qs[4 * i4 + 2][rh] = v.z;
            qs[4 * i4 + 3][rh] = v.w;
        }
        __syncthreads();
#pragma unroll 4
        for (unsigned i = 0; i < kIs2K; ++i) {
            const float4 kv = *reinterpret_cast<const float4*>(&ks[i][4 * pg]);
            const float4 qa = *reinterpret_cast<const float4*>(&qs[i][8 * rg]), qb = *reinterpret_cast<const float4*>(&qs[i][8 * rg + 4]);
            const float kk[4] = {kv.x, kv.y, kv.z, kv.w}, qq[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned b = 0; b < 8; ++b) acc[a][b] = __builtin_fmaf(qq[b], kk[a], acc[a][b]);
        }
    }
    __syncthreads();
#pragma unroll
    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
        for (unsigned b = 0; b < 8; ++b) dots[8 * rg + b][4 * pg + a] = acc[a][b];
    if (tid < kIs2RH) {
        const unsigned r = r0 + tid / kIdxHeads;
        ws[tid] = r < C ? wh[static_cast<std::size_t>(r) * kIdxHeads + tid % kIdxHeads] : 0.0F;
    }
    __syncthreads();
    const unsigned rr = tid / kIs2Pools, pp = tid % kIs2Pools, r = r0 + rr, p = p0 + pp;  // phase 2: thread = (row, pool)
    if (r >= C) return;
    const unsigned P = (t0 + r + 1) / kKpool;
    if (P <= kIdxPools || p >= P) return;
    float a2 = 0.0F;
    for (unsigned h = 0; h < kIdxHeads; ++h) {
        const float sc = dots[rr * kIdxHeads + h][pp] * kInvSqrt128;
        a2 = __builtin_fmaf(ws[rr * kIdxHeads + h], sc > 0.0F ? sc : 0.0F, a2);
    }
    keys[static_cast<std::size_t>(r) * cap + p] = ordered_key(a2);
}

// Scores v3: CTA = 64 pools x 8 rows; the pools' keys are staged once in LDS ([pool][dim], padded) and shared by the 8
// rows; wave w = row r0 + w, lane = pools p0 + lane and p0 + 32 + lane, all 32 heads' dots in registers. qI and wh of
// the row are wave-uniform and come through scalar loads, so the inner loop is FMAs with an SGPR operand. Per (row,
// head, pool) the dot chain over i ascending from +0 and per (row, pool) the head chain are v2's: same keys.
constexpr unsigned kIs3Rows = 8, kIs3Pools = 64, kIs3Pad = kIdxDim + 4, kIs3L = 16;
__global__ __launch_bounds__(32 * kIs3Rows) void idx_score_rows_v3_kernel(const float* pooled, unsigned t0, unsigned C, const float* qI,
                                                                         const float* wh, std::uint32_t* keys, unsigned cap) {
    __shared__ alignas(16) float ks[kIs3Pools][kIs3Pad];  // 33 KB
    const unsigned p0 = blockIdx.x * kIs3Pools, r0 = blockIdx.y * kIs3Rows, tid = threadIdx.x, lane = tid % 32;
    const unsigned rlast = (r0 + kIs3Rows < C ? r0 + kIs3Rows : C) - 1;
    const unsigned Pmax = (t0 + rlast + 1) / kKpool;  // pools of the tile's last row
    if (p0 >= Pmax || Pmax <= kIdxPools) return;    // uniform per block
    for (unsigned x = tid; x < kIs3Pools * (kIdxDim / 4); x += 32 * kIs3Rows) {
        const unsigned pp = x / (kIdxDim / 4), i4 = x % (kIdxDim / 4);
        float4 v{0.0F, 0.0F, 0.0F, 0.0F};
        if (p0 + pp < Pmax) v = *reinterpret_cast<const float4*>(pooled + static_cast<std::size_t>(p0 + pp) * kIdxDim + 4 * i4);
        *reinterpret_cast<float4*>(&ks[pp][4 * i4]) = v;
    }
    __syncthreads();
    const unsigned w = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(tid / 32))), r = r0 + w;
    if (r >= C) return;  // uniform per wave (no barrier follows)
    const unsigned P = (t0 + r + 1) / kKpool;
    if (P <= kIdxPools || p0 >= P) return;
    const auto* qr = (ConstF*)(qI + static_cast<std::size_t>(r) * kIdxHeads * kIdxDim);  // NOLINT: address-space cast
    float acc[2][kIdxHeads];
#pragma unroll
    for (unsigned e = 0; e < 2; ++e)
#pragma unroll
        for (unsigned h = 0; h < kIdxHeads; ++h) acc[e][h] = 0.0F;
    for (unsigned i0 = 0; i0 < kIdxDim; i0 += kIs3L) {
        float kv[2][kIs3L];
#pragma unroll
        for (unsigned e = 0; e < 2; ++e)
#pragma unroll
            for (unsigned i = 0; i < kIs3L; i += 4) {
                const float4 v = *reinterpret_cast<const float4*>(&ks[lane + 32 * e][i0 + i]);
                kv[e][i] = v.x;
                kv[e][i + 1] = v.y;
                kv[e][i + 2] = v.z;
                kv[e][i + 3] = v.w;
            }
#pragma unroll
        for (unsigned h = 0; h < kIdxHeads; ++h) {
            const SVec16 q = *reinterpret_cast<ConstV16*>(qr + static_cast<std::size_t>(h) * kIdxDim + i0);
#pragma unroll
            for (unsigned i = 0; i < kIs3L; ++i)
#pragma unroll
                for (unsigned e = 0; e < 2; ++e) acc[e][h] = __builtin_fmaf(q[i], kv[e][i], acc[e][h]);
        }
    }
    const auto* wr = (ConstF*)(wh + static_cast<std::size_t>(r) * kIdxHeads);  // NOLINT: address-space cast
#pragma unroll
    for (unsigned e = 0; e < 2; ++e) {
        float a2 = 0.0F;
#pragma unroll
        for (unsigned h = 0; h < kIdxHeads; ++h) {
            const float sc = acc[e][h] * kInvSqrt128;
            a2 = __builtin_fmaf(wr[h], sc > 0.0F ? sc : 0.0F, a2);
        }
        const unsigned p = p0 + lane + 32 * e;
        if (p < P) keys[static_cast<std::size_t>(r) * cap + p] = ordered_key(a2);
    }
}

// idx_select_kernel per row (blockIdx.x = row, t = t0 + row). P <= 512: every pool is selected, the list is 0 .. t.
__global__ __launch_bounds__(1024) void idx_select_rows_kernel(const std::uint32_t* keys_all, unsigned cap, unsigned t0,
                                                               std::uint32_t* tokens_all, std::uint32_t* n_all) {
    __shared__ unsigned hist[256];
    __shared__ unsigned sh_prefix, sh_mask, sh_remaining, sh_carry_eq, sh_carry_out;
    __shared__ unsigned scan_eq[1024], scan_sel[1024];
    const unsigned l = threadIdx.x, row = blockIdx.x, t = t0 + row;
    const unsigned P = (t + 1) / kKpool;
    std::uint32_t* tokens = tokens_all + static_cast<std::size_t>(row) * kMaxTokens;
    if (P <= kIdxPools) {
        for (unsigned pos = l; pos <= t; pos += 1024) tokens[pos] = pos;
        if (l == 0) n_all[row] = t + 1;
        return;
    }
    const std::uint32_t* keys = keys_all + static_cast<std::size_t>(row) * cap;
    const unsigned k = kIdxPools;
    if (l == 0) {
        sh_prefix = 0;
        sh_mask = 0;
        sh_remaining = k;
        sh_carry_eq = 0;
        sh_carry_out = 0;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned d = l; d < 256; d += 1024) hist[d] = 0;
        __syncthreads();
        const unsigned prefix = sh_prefix, mask = sh_mask;
        for (unsigned p = l; p < P; p += 1024)
            if ((keys[p] & mask) == prefix) atomicAdd(&hist[(keys[p] >> shift) & 255U], 1U);
        __syncthreads();
        if (l == 0) {
            unsigned rem = sh_remaining;
            for (int d = 255; d >= 0; --d) {
                if (hist[d] >= rem) {
                    sh_prefix = prefix | (static_cast<unsigned>(d) << shift);
                    break;
                }
                rem -= hist[d];
            }
            sh_remaining = rem;
            sh_mask = mask | (255U << shift);
        }
        __syncthreads();
    }
    const unsigned kstar = sh_prefix, take_eq = sh_remaining;
    for (unsigned base = 0; base < P; base += 1024) {
        const unsigned p = base + l;
        const unsigned key = p < P ? keys[p] : 0U;
        const unsigned eq = (p < P && key == kstar) ? 1U : 0U;
        scan_eq[l] = eq;
        __syncthreads();
        for (unsigned off = 1; off < 1024; off <<= 1U) {
            const unsigned v = l >= off ? scan_eq[l - off] : 0U;
            __syncthreads();
            scan_eq[l] += v;
            __syncthreads();
        }
        const unsigned eq_rank = sh_carry_eq + scan_eq[l] - eq;
        const unsigned sel = (p < P && (key > kstar || (eq && eq_rank < take_eq))) ? 1U : 0U;
        scan_sel[l] = sel;
        __syncthreads();
        for (unsigned off = 1; off < 1024; off <<= 1U) {
            const unsigned v = l >= off ? scan_sel[l - off] : 0U;
            __syncthreads();
            scan_sel[l] += v;
            __syncthreads();
        }
        if (sel) {
            const unsigned at = sh_carry_out + scan_sel[l] - 1;
            for (unsigned j = 0; j < kKpool; ++j) tokens[at * kKpool + j] = p * kKpool + j;
        }
        __syncthreads();
        if (l == 1023) {
            sh_carry_eq += scan_eq[1023];
            sh_carry_out += scan_sel[1023];
        }
        __syncthreads();
    }
    if (l == 0) {
        unsigned n = sh_carry_out * kKpool;
        for (unsigned pos = P * kKpool; pos <= t; ++pos) tokens[n++] = pos;
        n_all[row] = n;
    }
}

// Select v2: the same list as idx_select_rows_kernel (an integer set/order property: the same top-512 keys, ties broken
// by ascending pool, pools in ascending order, then the tail), with 256 threads per row. Radix digit search: the 256
// counts are scanned in parallel (descending digit) instead of by one thread; compaction: 1024 keys per round, 4
// consecutive pools per thread, exclusive counts from a wave scan plus the per-wave totals (4 barriers per round
// instead of 40).
constexpr unsigned kSel2Threads = 256, kSel2Waves = kSel2Threads / 32;
// Exclusive prefix of v over the CTA in thread order; `total` = the CTA sum. ws: kSel2Waves words of LDS.
__device__ __forceinline__ unsigned sel2_excl_scan(unsigned v, unsigned* ws, unsigned& total) {
    const unsigned lane = threadIdx.x % 32, w = threadIdx.x / 32;
    unsigned x = v;
#pragma unroll
    for (unsigned off = 1; off < 32; off <<= 1U) {
        const unsigned y = __shfl_up(x, off, 32);
        if (lane >= off) x += y;
    }
    if (lane == 31) ws[w] = x;
    __syncthreads();
    unsigned before = 0, tot = 0;
#pragma unroll
    for (unsigned i = 0; i < kSel2Waves; ++i) {
        const unsigned s = ws[i];
        tot += s;
        before += i < w ? s : 0U;
    }
    __syncthreads();  // ws is reused by the next scan
    total = tot;
    return before + x - v;
}
__global__ __launch_bounds__(kSel2Threads) void idx_select_rows_v2_kernel(const std::uint32_t* keys_all, unsigned cap, unsigned t0,
                                                                         std::uint32_t* tokens_all, std::uint32_t* n_all) {
    __shared__ unsigned hist[256];
    __shared__ unsigned ws[kSel2Waves];
    __shared__ unsigned sh_digit, sh_rem;
    const unsigned l = threadIdx.x, row = blockIdx.x, t = t0 + row;
    const unsigned P = (t + 1) / kKpool;
    std::uint32_t* tokens = tokens_all + static_cast<std::size_t>(row) * kMaxTokens;
    if (P <= kIdxPools) {
        for (unsigned pos = l; pos <= t; pos += kSel2Threads) tokens[pos] = pos;
        if (l == 0) n_all[row] = t + 1;
        return;
    }
    const std::uint32_t* keys = keys_all + static_cast<std::size_t>(row) * cap;
    unsigned prefix = 0, mask = 0, rem = kIdxPools;
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[l] = 0;
        __syncthreads();
        for (unsigned p = l; p < P; p += kSel2Threads) {
            const unsigned key = keys[p];
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255U], 1U);
        }
        __syncthreads();
        // thread l holds digit d = 255 - l; cum = count of matching keys with digit >= d. The chosen digit is the one
        // where cum first reaches rem (= the descending walk of the v1 kernel); rem becomes rem - (cum - h).
        const unsigned h = hist[255 - l];
        unsigned tot;
        const unsigned above = sel2_excl_scan(h, ws, tot);
        if (above < rem && above + h >= rem) {
            sh_digit = 255 - l;
            sh_rem = rem - above;
        }
        __syncthreads();
        prefix |= sh_digit << shift;
        mask |= 255U << shift;
        rem = sh_rem;
        __syncthreads();  // sh_* and hist are rewritten by the next pass
    }
    const unsigned kstar = prefix, take_eq = rem;
    unsigned carry_eq = 0, carry_out = 0;
    for (unsigned base = 0; base < P; base += 4 * kSel2Threads) {
        unsigned key[4], eq_n = 0;
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned p = base + 4 * l + i;
            key[i] = p < P ? keys[p] : 0U;
            eq_n += (p < P && key[i] == kstar) ? 1U : 0U;
        }
        unsigned eq_tot;
        unsigned eq_rank = carry_eq + sel2_excl_scan(eq_n, ws, eq_tot);
        bool sel[4];
        unsigned sel_n = 0;
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned p = base + 4 * l + i;
            const bool eq = p < P && key[i] == kstar;
            sel[i] = p < P && (key[i] > kstar || (eq && eq_rank < take_eq));
            eq_rank += eq ? 1U : 0U;
            sel_n += sel[i] ? 1U : 0U;
        }
        unsigned sel_tot;
        unsigned at = carry_out + sel2_excl_scan(sel_n, ws, sel_tot);
#pragma unroll
        for (unsigned i = 0; i < 4; ++i)
            if (sel[i]) {
                const unsigned p = base + 4 * l + i;
                for (unsigned j = 0; j < kKpool; ++j) tokens[at * kKpool + j] = p * kKpool + j;
                ++at;
            }
        carry_eq += eq_tot;
        carry_out += sel_tot;
    }
    if (l == 0) {
        unsigned n = carry_out * kKpool;
        for (unsigned pos = P * kKpool; pos <= t; ++pos) tokens[n++] = pos;
        n_all[row] = n;
    }
}

// ---------------------------------------------------------------- MLA (R13), split in four row-parallel stages
// (a) q-tilde = mla_absorb_kernel per row.
__global__ void mla_absorb_rows_kernel(const float* q, const std::uint16_t* kv_b, unsigned heads, float* qt) {
    const unsigned l = blockIdx.x * 256 + threadIdx.x, h = blockIdx.y, r = blockIdx.z;
    if (l >= kKvLora) return;
    const float* qr = q + (static_cast<std::size_t>(r) * heads + h) * kMlaDim;
    float acc = 0.0F;
    for (unsigned d = 0; d < kMlaDim; ++d)
        acc = __builtin_fmaf(qr[d], f16_to_float(kv_b[static_cast<std::size_t>(h * 2 * kMlaDim + d) * kKvLora + l]), acc);
    qt[(static_cast<std::size_t>(r) * heads + h) * kKvLora + l] = acc;
}
// (b) scores: grid (token block of 256, row); thread = list index j, all heads at once (one latent read per token);
// per (head, token) the chain over l ascending = mla_attn_kernel's, times the scale.
constexpr unsigned kLatBlk = 64;
template <unsigned H>
__global__ __launch_bounds__(256) void mla_scores_rows_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                              const float* qt, float* e_all) {
    __shared__ float qs[H][kLatBlk];
    const unsigned r = blockIdx.y, j = blockIdx.x * 256 + threadIdx.x, n = n_all[r];
    if (blockIdx.x * 256 >= n) return;  // uniform per block
    const bool act = j < n;
    const std::uint16_t* lat = sl.latent + static_cast<std::size_t>(act ? tokens_all[static_cast<std::size_t>(r) * kMaxTokens + j] : 0) * kKvLora;
    const float* qr = qt + static_cast<std::size_t>(r) * H * kKvLora;
    float acc[H];
#pragma unroll
    for (unsigned h = 0; h < H; ++h) acc[h] = 0.0F;
    for (unsigned i0 = 0; i0 < kKvLora; i0 += kLatBlk) {
        __syncthreads();
        for (unsigned x = threadIdx.x; x < H * kLatBlk; x += 256) qs[x / kLatBlk][x % kLatBlk] = qr[(x / kLatBlk) * kKvLora + i0 + x % kLatBlk];
        __syncthreads();
        if (act)
            for (unsigned i = 0; i < kLatBlk; ++i) {
                const float c = bf16_to_float(lat[i0 + i]);
#pragma unroll
                for (unsigned h = 0; h < H; ++h) acc[h] = __builtin_fmaf(qs[h][i], c, acc[h]);
            }
    }
    if (act) {
        float* e = e_all + static_cast<std::size_t>(r) * H * kEStride;
#pragma unroll
        for (unsigned h = 0; h < H; ++h) e[h * kEStride + j] = acc[h] * kMlaScale;
    }
}
// (c) softmax per (head, row): mla_attn_kernel's max / exp / lane-strided sum / tree / divide, in place.
// kLane (o-bar v5): the normalized p are written lane-major instead, slot 9 * bitrev8(t % 256) + t / 256 (lane l's
// rounds m = 0..8 contiguous, lanes in k = bitrev8(l) order; slots of t >= n are +0, so all 2304 slots are written,
// each by its lane's thread). Same values and arithmetic; every thread reads only its own t, and the permuted stores
// start after a barrier that follows the last read.
__device__ __forceinline__ unsigned bitrev8(unsigned x) { return __brev(x) >> 24; }
template <bool kLane>
__global__ __launch_bounds__(256) void mla_softmax_rows_kernel(const std::uint32_t* n_all, unsigned heads, float* e_all) {
    __shared__ float s[256];
    const unsigned h = blockIdx.x, r = blockIdx.y, l = threadIdx.x, n = n_all[r];
    float* e = e_all + (static_cast<std::size_t>(r) * heads + h) * kEStride;
    float mx = -__builtin_huge_valf();
    for (unsigned t = l; t < n; t += 256) mx = e[t] > mx ? e[t] : mx;
    s[l] = mx;
    __syncthreads();
    for (unsigned hh = 128; hh > 0; hh >>= 1U) {
        if (l < hh) s[l] = s[l + hh] > s[l] ? s[l + hh] : s[l];
        __syncthreads();
    }
    const float m = s[0];
    __syncthreads();
    float a = 0.0F;
    for (unsigned t = l; t < n; t += 256) {
        e[t] = declared_exp(e[t] - m);
        a = a + e[t];
    }
    const float sum = lds_tree<256>(s, a);
    if constexpr (!kLane) {
        for (unsigned t = l; t < n; t += 256) e[t] = e[t] / sum;
    } else {
        float q[kObRounds];
#pragma unroll
        for (unsigned i = 0; i < kObRounds; ++i) q[i] = l + 256 * i < n ? e[l + 256 * i] / sum : 0.0F;
        __syncthreads();  // every thread's reads of e are done
        float* lane = e + kObRounds * bitrev8(l);
#pragma unroll
        for (unsigned i = 0; i < kObRounds; ++i) lane[i] = q[i];
    }
}
// (d) o-bar: grid (latent block of kObLat dims, row); lane j accumulates p_t * c_t[l] over t = j, j + 256, ... (fmaf,
// ascending t) for two heads x kObLat dims at a time, then 2 * kObLat lds_tree<256> pairings evaluated together.
constexpr unsigned kObLat = 16, kObTrees = 2 * kObLat;
__global__ __launch_bounds__(256) void mla_obar_rows_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                            unsigned heads, const float* e_all, float* ob) {
    __shared__ float s[kObTrees][256];
    const unsigned lb = blockIdx.x * kObLat, r = blockIdx.y, j = threadIdx.x, n = n_all[r];
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    for (unsigned h0 = 0; h0 < heads; h0 += 2) {
        const float* e0 = e_all + (static_cast<std::size_t>(r) * heads + h0) * kEStride;
        const float* e1 = e0 + kEStride;
        float acc[2][kObLat];
#pragma unroll
        for (unsigned c = 0; c < kObLat; ++c) acc[0][c] = acc[1][c] = 0.0F;
        for (unsigned t = j; t < n; t += 256) {
            const std::uint16_t* lat = sl.latent + static_cast<std::size_t>(tok[t]) * kKvLora + lb;
            const float p0 = e0[t], p1 = e1[t];
#pragma unroll
            for (unsigned c = 0; c < kObLat; ++c) {
                const float cv = bf16_to_float(lat[c]);
                acc[0][c] = __builtin_fmaf(p0, cv, acc[0][c]);
                acc[1][c] = __builtin_fmaf(p1, cv, acc[1][c]);
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned c = 0; c < kObLat; ++c) {
            s[c][j] = acc[0][c];
            s[kObLat + c][j] = acc[1][c];
        }
        __syncthreads();
        for (unsigned hh = 128; hh > 0; hh >>= 1U) {
            for (unsigned x = j; x < kObTrees * hh; x += 256) {
                const unsigned q = x / hh, l = x % hh;
                s[q][l] = s[q][l] + s[q][l + hh];
            }
            __syncthreads();
        }
        if (j < kObTrees) ob[(static_cast<std::size_t>(r) * heads + h0 + j / kObLat) * kKvLora + lb + j % kObLat] = s[j][0];
    }
}
// (e) output chain per (head, row), = the tail of mla_attn_kernel.
__global__ __launch_bounds__(256) void mla_out_rows_kernel(const float* ob, const std::uint16_t* kv_b, unsigned heads, float* o) {
    const unsigned h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const float* obr = ob + (static_cast<std::size_t>(r) * heads + h) * kKvLora;
    float acc = 0.0F;
    for (unsigned i = 0; i < kKvLora; ++i)
        acc = __builtin_fmaf(obr[i], f16_to_float(kv_b[static_cast<std::size_t>(h * 2 * kMlaDim + kMlaDim + d) * kKvLora + i]), acc);
    o[(static_cast<std::size_t>(r) * heads + h) * kMlaDim + d] = acc;
}

// ---------------------------------------------------------------- MLA v2 (default): same per-output expressions, other
// placement. The absorb and output chains become tiled VALU GEMMs over rows (kv_b read once per 64 rows instead of per
// row); scores read q-tilde transposed with vector LDS loads; o-bar stages the row's latent tile once in LDS and runs the
// lds_tree<256> pairing as 3 LDS levels (128, 64, 32) + 5 shuffle levels (16 .. 1) inside wave 0.
constexpr unsigned kGT = 64, kGK = 32;  // GEMM tile: 64 rows x 64 cols, K step 32; thread = 4 x 4 outputs
// Y[r][h][c] = chain over k ascending of fmaf(A[r][h][k], float(W(h, k, c)), acc), k < K.
//   absorb: A = q [C][H][256], W(h, d, l) = kv_b[(h*512 + d)*512 + l], c = l < 512, K = 256
//   out:    A = ob [C][H][512], W(h, l, d) = kv_b[(h*512 + 256 + d)*512 + l], c = d < 256, K = 512
template <bool kOut>
__global__ __launch_bounds__(256) void mla_gemm_rows_kernel(const float* A, const std::uint16_t* kv_b, unsigned heads, unsigned C, float* Y) {
    constexpr unsigned K = kOut ? kKvLora : kMlaDim, NC = kOut ? kMlaDim : kKvLora;
    __shared__ float as[kGT][kGK + 1];
    __shared__ float ws[kGK][kGT + 1];
    const unsigned r0 = blockIdx.x * kGT, c0 = blockIdx.y * kGT, h = blockIdx.z, tid = threadIdx.x, tr = tid / 16, tc = tid % 16;
    float acc[4][4];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) acc[i][j] = 0.0F;
    for (unsigned k0 = 0; k0 < K; k0 += kGK) {
        __syncthreads();
        for (unsigned x = tid; x < kGT * kGK; x += 256) {  // A tile [64 rows][32 k]
            const unsigned rr = x / kGK, kk = x % kGK, r = r0 + rr;
            as[rr][kk] = r < C ? A[(static_cast<std::size_t>(r) * heads + h) * K + k0 + kk] : 0.0F;
        }
        for (unsigned x = tid; x < kGK * kGT; x += 256) {  // W tile [32 k][64 c]
            std::size_t at;
            unsigned kk, cc;
            if (kOut) {  // kv_b row (h*512 + 256 + d), column l: consecutive x -> consecutive l (coalesced)
                cc = x / kGK;
                kk = x % kGK;
                at = static_cast<std::size_t>(h * 2 * kMlaDim + kMlaDim + c0 + cc) * kKvLora + k0 + kk;
            } else {     // kv_b row (h*512 + d), column l
                kk = x / kGT;
                cc = x % kGT;
                at = static_cast<std::size_t>(h * 2 * kMlaDim + k0 + kk) * kKvLora + c0 + cc;
            }
            ws[kk][cc] = f16_to_float(kv_b[at]);
        }
        __syncthreads();
#pragma unroll 4
        for (unsigned kk = 0; kk < kGK; ++kk) {
            float a[4], w[4];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) a[i] = as[tr * 4 + i][kk];
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) w[j] = ws[kk][tc * 4 + j];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) acc[i][j] = __builtin_fmaf(a[i], w[j], acc[i][j]);
        }
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned r = r0 + tr * 4 + i;
        if (r >= C) continue;
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) Y[(static_cast<std::size_t>(r) * heads + h) * NC + c0 + tc * 4 + j] = acc[i][j];
    }
}
// GEMM v2: mla_gemm_rows_kernel's outputs and per-output chain (k ascending from +0, fmaf(A, W, acc)) with a 128-row x
// 64-column tile and 8 x 4 outputs per thread. A is staged transposed ([k][row], padded) and W as f32 ([k][col], padded),
// so one k step is 3 ds_load_b128 + 32 FMAs; the next K stage is loaded into registers while the current one is used.
constexpr unsigned kG2R = 128, kG2C = 64, kG2K = 32, kG2PadR = kG2R + 4, kG2PadC = kG2C + 4;
template <bool kOut>
__global__ __launch_bounds__(256) void mla_gemm_rows_v2_kernel(const float* A, const std::uint16_t* kv_b, unsigned heads, unsigned C, float* Y) {
    constexpr unsigned K = kOut ? kKvLora : kMlaDim, NC = kOut ? kMlaDim : kKvLora;
    __shared__ alignas(16) float as[kG2K][kG2PadR];  // 16.5 KB
    __shared__ alignas(16) float ws[kG2K][kG2PadC];  // 8.5 KB
    const unsigned r0 = blockIdx.x * kG2R, c0 = blockIdx.y * kG2C, h = blockIdx.z, tid = threadIdx.x, tr = tid / 16, tc = tid % 16;
    float4 ar[4];
    uint4 wr;
    auto load = [&](unsigned k0) {
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {  // A tile [128 rows][32 k]: x -> row x / 8, k quad x % 8 (128 B per row)
            const unsigned x = tid + 256 * i, rr = x / 8, q = x % 8, r = r0 + rr;
            ar[i] = r < C ? *reinterpret_cast<const float4*>(A + (static_cast<std::size_t>(r) * heads + h) * K + k0 + 4 * q)
                          : float4{0.0F, 0.0F, 0.0F, 0.0F};
        }
        std::size_t at;
        if (kOut) {  // column c (kv_b row h*512 + 256 + c), 8 consecutive k
            const unsigned cc = tid / 4, k8 = tid % 4;
            at = static_cast<std::size_t>(h * 2 * kMlaDim + kMlaDim + c0 + cc) * kKvLora + k0 + 8 * k8;
        } else {     // k row (kv_b row h*512 + k), 8 consecutive c
            const unsigned kk = tid / 8, c8 = tid % 8;
            at = static_cast<std::size_t>(h * 2 * kMlaDim + k0 + kk) * kKvLora + c0 + 8 * c8;
        }
        wr = *reinterpret_cast<const uint4*>(kv_b + at);
    };
    auto store = [&]() {
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned x = tid + 256 * i, rr = x / 8, q = x % 8;
            as[4 * q + 0][rr] = ar[i].x;
            as[4 * q + 1][rr] = ar[i].y;
            as[4 * q + 2][rr] = ar[i].z;
            as[4 * q + 3][rr] = ar[i].w;
        }
        const std::uint32_t wv[4] = {wr.x, wr.y, wr.z, wr.w};
        float f[8];
#pragma unroll
        for (unsigned ii = 0; ii < 8; ++ii) f[ii] = f16_to_float(static_cast<std::uint16_t>(wv[ii / 2] >> (16 * (ii % 2))));
        if (kOut) {
            const unsigned cc = tid / 4, k8 = tid % 4;
#pragma unroll
            for (unsigned ii = 0; ii < 8; ++ii) ws[8 * k8 + ii][cc] = f[ii];
        } else {
            const unsigned kk = tid / 8, c8 = tid % 8;
            *reinterpret_cast<float4*>(&ws[kk][8 * c8]) = float4{f[0], f[1], f[2], f[3]};
            *reinterpret_cast<float4*>(&ws[kk][8 * c8 + 4]) = float4{f[4], f[5], f[6], f[7]};
        }
    };
    float acc[8][4];
#pragma unroll
    for (unsigned i = 0; i < 8; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) acc[i][j] = 0.0F;
    load(0);
    for (unsigned k0 = 0; k0 < K; k0 += kG2K) {
        __syncthreads();  // previous stage consumed; this stage's loads complete
        store();
        __syncthreads();
        if (k0 + kG2K < K) load(k0 + kG2K);
#pragma unroll 4
        for (unsigned kk = 0; kk < kG2K; ++kk) {
            const float4 a0 = *reinterpret_cast<const float4*>(&as[kk][tr * 8]), a1 = *reinterpret_cast<const float4*>(&as[kk][tr * 8 + 4]);
            const float4 w4 = *reinterpret_cast<const float4*>(&ws[kk][tc * 4]);
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, w[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) acc[i][j] = __builtin_fmaf(a[i], w[j], acc[i][j]);
        }
    }
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned r = r0 + tr * 8 + i;
        if (r < C)
            *reinterpret_cast<float4*>(Y + (static_cast<std::size_t>(r) * heads + h) * NC + c0 + tc * 4) =
                float4{acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
    }
}

// Scores v2: thread = list index j; q-tilde of the row staged transposed [i][H] so 4 heads come in one LDS load; the
// token's latent read 8 dims at a time. Per (head, token) the chain over l ascending is unchanged.
template <unsigned H>
__global__ __launch_bounds__(256) void mla_scores_v2_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                            const float* qt, float* e_all) {
    __shared__ float4 qs[kLatBlk][H / 4];
    const unsigned r = blockIdx.y, j = blockIdx.x * 256 + threadIdx.x, n = n_all[r];
    if (blockIdx.x * 256 >= n) return;  // uniform per block
    const bool act = j < n;
    const uint4* lat = reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(act ? tokens_all[static_cast<std::size_t>(r) * kMaxTokens + j] : 0) * kKvLora);
    const float* qr = qt + static_cast<std::size_t>(r) * H * kKvLora;
    float acc[H];
#pragma unroll
    for (unsigned h = 0; h < H; ++h) acc[h] = 0.0F;
    for (unsigned i0 = 0; i0 < kKvLora; i0 += kLatBlk) {
        __syncthreads();
        for (unsigned x = threadIdx.x; x < kLatBlk * H; x += 256) {  // qs[i][h] = qt[h][i0 + i]
            const unsigned i = x % kLatBlk, h = x / kLatBlk;
            reinterpret_cast<float*>(qs[i])[h] = qr[h * kKvLora + i0 + i];
        }
        __syncthreads();
        if (act)
            for (unsigned i = 0; i < kLatBlk; i += 8) {
                const uint4 v = lat[(i0 + i) / 8];
                const std::uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (unsigned ii = 0; ii < 8; ++ii) {
                    const float c = bf16_to_float(static_cast<std::uint16_t>(wv[ii / 2] >> (16 * (ii % 2))));
#pragma unroll
                    for (unsigned hq = 0; hq < H / 4; ++hq) {
                        const float4 q4 = qs[i + ii][hq];
                        acc[4 * hq + 0] = __builtin_fmaf(q4.x, c, acc[4 * hq + 0]);
                        acc[4 * hq + 1] = __builtin_fmaf(q4.y, c, acc[4 * hq + 1]);
                        acc[4 * hq + 2] = __builtin_fmaf(q4.z, c, acc[4 * hq + 2]);
                        acc[4 * hq + 3] = __builtin_fmaf(q4.w, c, acc[4 * hq + 3]);
                    }
                }
            }
    }
    if (act) {
        float* e = e_all + static_cast<std::size_t>(r) * H * kEStride;
#pragma unroll
        for (unsigned h = 0; h < H; ++h) e[h * kEStride + j] = acc[h] * kMlaScale;
    }
}
// Scores v3: CTA = 64 list entries x all heads of one row; thread = 2 entries (tp, tp + 32) x H / 8 heads (one wave
// shares its heads, so q-tilde comes as LDS broadcasts). Per 64-dim block the latent tile [dim][entry] (f32, the exact
// widening of the bf16 cache) and q-tilde [dim][head] are staged in LDS. Per (head, entry) the chain over l ascending
// from +0 is v2's (fmaf(q, c, acc)), times the scale: same bits.
constexpr unsigned kS3Tok = 64, kS3K = 64;
template <unsigned H>
__global__ __launch_bounds__(256) void mla_scores_v3_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                            const float* qt, float* e_all) {
    constexpr unsigned HPT = H / 8;  // heads per thread: 4 (TP2) or 8
    __shared__ float lt[kS3K][kS3Tok];
    __shared__ alignas(16) float qs[kS3K][H];
    const unsigned r = blockIdx.y, t0 = blockIdx.x * kS3Tok, n = n_all[r];
    if (t0 >= n) return;  // uniform per block
    const unsigned tid = threadIdx.x, tp = tid % 32, hg = tid / 32;
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    const float* qr = qt + static_cast<std::size_t>(r) * H * kKvLora;
    float acc[2][HPT];
#pragma unroll
    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
        for (unsigned b = 0; b < HPT; ++b) acc[a][b] = 0.0F;
    for (unsigned k0 = 0; k0 < kKvLora; k0 += kS3K) {
        __syncthreads();  // previous block's tiles consumed
        for (unsigned x = tid; x < kS3Tok * (kS3K / 8); x += 256) {  // lanes walk entries: conflict-free LDS stores
            const unsigned tt = x % kS3Tok, c8 = x / kS3Tok, j = t0 + tt;
            uint4 v{0U, 0U, 0U, 0U};
            if (j < n) v = *reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(tok[j]) * kKvLora + k0 + c8 * 8);
            const std::uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (unsigned ii = 0; ii < 8; ++ii) lt[c8 * 8 + ii][tt] = bf16_to_float(static_cast<std::uint16_t>(wv[ii / 2] >> (16 * (ii % 2))));
        }
        for (unsigned x = tid; x < (kS3K / 4) * H; x += 256) {  // qs[i][h] = qt[h][k0 + i]; lanes walk heads: conflict-free stores
            const unsigned h = x % H, i4 = x / H;
            const float4 v = *reinterpret_cast<const float4*>(qr + static_cast<std::size_t>(h) * kKvLora + k0 + 4 * i4);
            qs[4 * i4 + 0][h] = v.x;
            qs[4 * i4 + 1][h] = v.y;
            qs[4 * i4 + 2][h] = v.z;
            qs[4 * i4 + 3][h] = v.w;
        }
        __syncthreads();
#pragma unroll 4
        for (unsigned i = 0; i < kS3K; ++i) {
            const float c0 = lt[i][tp], c1 = lt[i][tp + 32];
#pragma unroll
            for (unsigned a = 0; a < HPT; a += 4) {
                const float4 q4 = *reinterpret_cast<const float4*>(&qs[i][hg * HPT + a]);
                acc[0][a + 0] = __builtin_fmaf(q4.x, c0, acc[0][a + 0]);
                acc[0][a + 1] = __builtin_fmaf(q4.y, c0, acc[0][a + 1]);
                acc[0][a + 2] = __builtin_fmaf(q4.z, c0, acc[0][a + 2]);
                acc[0][a + 3] = __builtin_fmaf(q4.w, c0, acc[0][a + 3]);
                acc[1][a + 0] = __builtin_fmaf(q4.x, c1, acc[1][a + 0]);
                acc[1][a + 1] = __builtin_fmaf(q4.y, c1, acc[1][a + 1]);
                acc[1][a + 2] = __builtin_fmaf(q4.z, c1, acc[1][a + 2]);
                acc[1][a + 3] = __builtin_fmaf(q4.w, c1, acc[1][a + 3]);
            }
        }
    }
    float* e = e_all + static_cast<std::size_t>(r) * H * kEStride;
#pragma unroll
    for (unsigned sidx = 0; sidx < 2; ++sidx) {
        const unsigned j = t0 + tp + 32 * sidx;
        if (j < n)
#pragma unroll
            for (unsigned a = 0; a < HPT; ++a) e[static_cast<std::size_t>(hg * HPT + a) * kEStride + j] = acc[sidx][a] * kMlaScale;
    }
}

// Scores v4: v3 output ownership and ascending FMAs, with 32-dim double-buffered tiles.
// Prefetch the next latent/q tile into registers while consuming the current LDS tile.
// LDS = 2 * 32 * (64 + H) * 4: 24 / 32 KiB for H = 32 / 64.
constexpr unsigned kS4K = 32;
template <unsigned H>
__global__ __launch_bounds__(256) void mla_scores_v4_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                            const float* qt, float* e_all) {
    constexpr unsigned HPT = H / 8;  // heads per thread: 4 (TP2) or 8
    __shared__ float lt[2][kS4K][kS3Tok];
    __shared__ alignas(16) float qs[2][kS4K][H];
    const unsigned r = blockIdx.y, t0 = blockIdx.x * kS3Tok, n = n_all[r];
    if (t0 >= n) return;  // uniform per block
    const unsigned tid = threadIdx.x, tp = tid % 32, hg = tid / 32;
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    const float* qr = qt + static_cast<std::size_t>(r) * H * kKvLora;
    float acc[2][HPT];
#pragma unroll
    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
        for (unsigned b = 0; b < HPT; ++b) acc[a][b] = 0.0F;
    uint4 lr;
    float4 qr4[H / 32];
    auto load = [&](unsigned k0) {
        const unsigned tt = tid % kS3Tok, c8 = tid / kS3Tok, j = t0 + tt;
        lr = j < n ? *reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(tok[j]) * kKvLora + k0 + c8 * 8)
                   : uint4{0, 0, 0, 0};
#pragma unroll
        for (unsigned i = 0; i < H / 32; ++i) {
            const unsigned x = tid + 256 * i, h = x % H, i4 = x / H;
            qr4[i] = *reinterpret_cast<const float4*>(qr + static_cast<std::size_t>(h) * kKvLora + k0 + 4 * i4);
        }
    };
    auto store = [&](unsigned buf) {
        const unsigned tt = tid % kS3Tok, c8 = tid / kS3Tok;
        const std::uint32_t wv[4] = {lr.x, lr.y, lr.z, lr.w};
#pragma unroll
        for (unsigned ii = 0; ii < 8; ++ii)
            lt[buf][c8 * 8 + ii][tt] = bf16_to_float(static_cast<std::uint16_t>(wv[ii / 2] >> (16 * (ii % 2))));
#pragma unroll
        for (unsigned i = 0; i < H / 32; ++i) {
            const unsigned x = tid + 256 * i, h = x % H, i4 = x / H;
            qs[buf][4 * i4 + 0][h] = qr4[i].x;
            qs[buf][4 * i4 + 1][h] = qr4[i].y;
            qs[buf][4 * i4 + 2][h] = qr4[i].z;
            qs[buf][4 * i4 + 3][h] = qr4[i].w;
        }
    };
    load(0);
    store(0);
    __syncthreads();
    for (unsigned k0 = 0; k0 < kKvLora; k0 += kS4K) {
        const unsigned cur = (k0 / kS4K) & 1U;
        if (k0 + kS4K < kKvLora) load(k0 + kS4K);
#pragma unroll 4
        for (unsigned i = 0; i < kS4K; ++i) {
            const float c0 = lt[cur][i][tp], c1 = lt[cur][i][tp + 32];
#pragma unroll
            for (unsigned a = 0; a < HPT; a += 4) {
                const float4 q4 = *reinterpret_cast<const float4*>(&qs[cur][i][hg * HPT + a]);
                acc[0][a + 0] = __builtin_fmaf(q4.x, c0, acc[0][a + 0]);
                acc[0][a + 1] = __builtin_fmaf(q4.y, c0, acc[0][a + 1]);
                acc[0][a + 2] = __builtin_fmaf(q4.z, c0, acc[0][a + 2]);
                acc[0][a + 3] = __builtin_fmaf(q4.w, c0, acc[0][a + 3]);
                acc[1][a + 0] = __builtin_fmaf(q4.x, c1, acc[1][a + 0]);
                acc[1][a + 1] = __builtin_fmaf(q4.y, c1, acc[1][a + 1]);
                acc[1][a + 2] = __builtin_fmaf(q4.z, c1, acc[1][a + 2]);
                acc[1][a + 3] = __builtin_fmaf(q4.w, c1, acc[1][a + 3]);
            }
        }
        if (k0 + kS4K < kKvLora) store(cur ^ 1U);
        __syncthreads();
    }
    float* e = e_all + static_cast<std::size_t>(r) * H * kEStride;
#pragma unroll
    for (unsigned sidx = 0; sidx < 2; ++sidx) {
        const unsigned j = t0 + tp + 32 * sidx;
        if (j < n)
#pragma unroll
            for (unsigned a = 0; a < HPT; ++a) e[static_cast<std::size_t>(hg * HPT + a) * kEStride + j] = acc[sidx][a] * kMlaScale;
    }
}

// Scores v5: lane = list entries j and j + 32 of a 64-entry wave tile (one entry for H = 64), every head's accumulator
// in registers. q-tilde is the same for the whole CTA (one row), so it is read through scalar loads (16 consecutive dims
// of one head per load) and each inner-loop FMA takes its q operand from an SGPR; the entries' latent comes straight from
// global memory, 16 bf16 per entry per step, prefetched one step ahead. No LDS, no barriers. Per (head, entry) the chain
// over l ascending from +0 is v4's fmaf(q, c, acc), times the scale: same bits.
constexpr unsigned kS5L = 16, kS5Waves = 8;
template <unsigned H>
__global__ __launch_bounds__(32 * kS5Waves) void mla_scores_v5_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                                      const float* qt, float* e_all) {
    constexpr unsigned E = 64 / H;  // entries per lane: 2 (TP2) or 1
    const unsigned r = blockIdx.y, n = n_all[r];
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / 32))), lane = threadIdx.x % 32;
    const unsigned j0 = (blockIdx.x * kS5Waves + wave) * 32 * E;
    if (j0 >= n) return;  // uniform per wave
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    const uint4* lat[E];
#pragma unroll
    for (unsigned x = 0; x < E; ++x) {
        const unsigned j = j0 + lane + 32 * x;  // entries past n read position 0's latent (always cached) and store nothing
        lat[x] = reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(j < n ? tok[j] : 0U) * kKvLora);
    }
    const auto* qr = (ConstF*)(qt + static_cast<std::size_t>(r) * H * kKvLora);  // NOLINT: address-space cast
    float acc[E][H];
#pragma unroll
    for (unsigned x = 0; x < E; ++x)
#pragma unroll
        for (unsigned h = 0; h < H; ++h) acc[x][h] = 0.0F;
    uint4 raw[E][2];
#pragma unroll
    for (unsigned x = 0; x < E; ++x) {
        raw[x][0] = lat[x][0];
        raw[x][1] = lat[x][1];
    }
    for (unsigned l0 = 0; l0 < kKvLora; l0 += kS5L) {
        float c[E][kS5L];
#pragma unroll
        for (unsigned x = 0; x < E; ++x) {
            const std::uint32_t wv[8] = {raw[x][0].x, raw[x][0].y, raw[x][0].z, raw[x][0].w, raw[x][1].x, raw[x][1].y, raw[x][1].z, raw[x][1].w};
#pragma unroll
            for (unsigned i = 0; i < kS5L; ++i) c[x][i] = bf16_to_float(static_cast<std::uint16_t>(wv[i / 2] >> (16 * (i % 2))));
        }
        if (l0 + kS5L < kKvLora)
#pragma unroll
            for (unsigned x = 0; x < E; ++x) {
                raw[x][0] = lat[x][(l0 + kS5L) / 8];
                raw[x][1] = lat[x][(l0 + kS5L) / 8 + 1];
            }
#pragma unroll
        for (unsigned h = 0; h < H; ++h) {
            const SVec16 q = *reinterpret_cast<ConstV16*>(qr + static_cast<std::size_t>(h) * kKvLora + l0);
#pragma unroll
            for (unsigned i = 0; i < kS5L; ++i)
#pragma unroll
                for (unsigned x = 0; x < E; ++x) acc[x][h] = __builtin_fmaf(q[i], c[x][i], acc[x][h]);
        }
    }
    float* e = e_all + static_cast<std::size_t>(r) * H * kEStride;
#pragma unroll
    for (unsigned x = 0; x < E; ++x) {
        const unsigned j = j0 + lane + 32 * x;
        if (j < n)
#pragma unroll
            for (unsigned h = 0; h < H; ++h) e[static_cast<std::size_t>(h) * kEStride + j] = acc[x][h] * kMlaScale;
    }
}

// Scores v6 (heads 32): v5's placement, loads and chains with the inner step hand-paired as VOPD (the two entries of a
// lane share each q SGPR; four heads interleaved per block; tools/gen_glm_scores_vopd.py has the compact register
// layout, 125 VGPRs). a[2 h + x] = v5's acc[x][h]. Measured 4.57 -> 3.40, 9.27 -> 8.27, 11.39 -> 10.38 us/row per DSA
// layer at t0 0 / 4096 / 28672 (a first layout pinned up to v133, 158 VGPRs and two heads per block, was slower).
#include "glm/prefill/glm_pf_scores_vopd.inc"
template <unsigned P = 0>
__device__ __forceinline__ void scores_vopd_step(float (&a)[64], const float (&c)[32], const ConstF* qr, unsigned l0) {
    const SVec16 qa = *reinterpret_cast<ConstV16*>(qr + static_cast<std::size_t>(4 * P) * kKvLora + l0);
    const SVec16 qb = *reinterpret_cast<ConstV16*>(qr + static_cast<std::size_t>(4 * P + 1) * kKvLora + l0);
    const SVec16 qc = *reinterpret_cast<ConstV16*>(qr + static_cast<std::size_t>(4 * P + 2) * kKvLora + l0);
    const SVec16 qd = *reinterpret_cast<ConstV16*>(qr + static_cast<std::size_t>(4 * P + 3) * kKvLora + l0);
    scores_vopd_heads<P>(a, c, qa, qb, qc, qd);
    if constexpr (P + 1 < 8) scores_vopd_step<P + 1>(a, c, qr, l0);
}
// Scores v7 = v6 with G = kS7Group: a 1-D grid in groups of G rows x the 5 list tiles, rows fastest inside a group, so the
// CTAs in flight are up to G adjacent rows' same list tile (their sorted selections overlap and share latent lines in
// L2) while a row's absorbed q is still reused within its group. Same work per CTA. Measured per DSA layer at t0 0 /
// 4096 / 12288 / 28672 (C 4096): 4.96 / 8.42 / 9.41 / 10.46 -> 4.90 / 6.08 / 7.87 / 9.89 us/row (groups of 16 or 64
// rows gained less, all rows at once lost 38 % at t0 0). G = 0: v6's (tile, row) grid.
constexpr unsigned kS6Tiles = (kMaxTokens + 32 * kS5Waves * 2 - 1) / (32 * kS5Waves * 2), kS7Group = 256;
template <unsigned G>
__global__ __launch_bounds__(32 * kS5Waves) void mla_scores_v6_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                                      const float* qt, float* e_all, unsigned C) {
    constexpr unsigned H = 32;
    const unsigned idx = blockIdx.x % (G ? G * kS6Tiles : 1U);
    const unsigned r = G ? blockIdx.x / (G * kS6Tiles) * G + idx % G : blockIdx.y, tile = G ? idx / G : blockIdx.x;
    if (r >= C) return;  // uniform per CTA (partial last group)
    const unsigned n = n_all[r];
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / 32))), lane = threadIdx.x % 32;
    const unsigned j0 = (tile * kS5Waves + wave) * 64;
    if (j0 >= n) return;  // uniform per wave
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    const uint4* lat[2];
#pragma unroll
    for (unsigned x = 0; x < 2; ++x) {
        const unsigned j = j0 + lane + 32 * x;  // entries past n read position 0's latent (always cached) and store nothing
        lat[x] = reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(j < n ? tok[j] : 0U) * kKvLora);
    }
    const auto* qr = (ConstF*)(qt + static_cast<std::size_t>(r) * H * kKvLora);  // NOLINT: address-space cast
    float a[64];
#pragma unroll
    for (unsigned i = 0; i < 64; ++i) a[i] = 0.0F;
    uint4 raw[2][2];
#pragma unroll
    for (unsigned x = 0; x < 2; ++x) {
        raw[x][0] = lat[x][0];
        raw[x][1] = lat[x][1];
    }
    for (unsigned l0 = 0; l0 < kKvLora; l0 += kS5L) {
        float c[32];  // c[2 i + x] = v5's c[x][i]
#pragma unroll
        for (unsigned x = 0; x < 2; ++x) {
            const std::uint32_t wv[8] = {raw[x][0].x, raw[x][0].y, raw[x][0].z, raw[x][0].w, raw[x][1].x, raw[x][1].y, raw[x][1].z, raw[x][1].w};
#pragma unroll
            for (unsigned i = 0; i < kS5L; ++i) c[2 * i + x] = bf16_to_float(static_cast<std::uint16_t>(wv[i / 2] >> (16 * (i % 2))));
        }
        if (l0 + kS5L < kKvLora)
#pragma unroll
            for (unsigned x = 0; x < 2; ++x) {
                raw[x][0] = lat[x][(l0 + kS5L) / 8];
                raw[x][1] = lat[x][(l0 + kS5L) / 8 + 1];
            }
        scores_vopd_step(a, c, qr, l0);
    }
    float* e = e_all + static_cast<std::size_t>(r) * H * kEStride;
#pragma unroll
    for (unsigned x = 0; x < 2; ++x) {
        const unsigned j = j0 + lane + 32 * x;
        if (j < n)
#pragma unroll
            for (unsigned h = 0; h < H; ++h) e[static_cast<std::size_t>(h) * kEStride + j] = a[2 * h + x] * kMlaScale;
    }
}

// o-bar v4: no cross-thread reduction tree. tree_fma<256>'s fold s[l] += s[l + h] (h = 128 .. 1) is the adjacent-pair
// balanced tree over k = bitrev8(l) (operand order included; design notes section 2, checked symbolically), so one
// thread evaluates it exactly by walking lanes in k order and merging their chains in a binary counter (left = the
// older, lower-k value). CTA = 8 heads x 64 latent dims of one row; thread = 2 heads x 4 dims (8 outputs) of one
// quarter of k: sub-group s owns k in [64 s, 64 s + 64), i.e. lanes l = bitrev8(k) (l mod 4 fixed), and builds that
// subtree (6 levels); the 4 roots then combine as (R0 + R1) + (R2 + R3), the tree's top 2 levels. Per lane the chain is
// the contract's: fmaf(p_t, c_t, a) over t = l, l + 256, ... ascending from +0 (lanes without tokens stay +0).
// Staging per batch of 8 k-steps: the 32 lanes' tokens (9 slots per lane: t = l + 256 m, m < 9 since n <= 2051), their
// 64 latent dims (bf16) and the 8 heads' p, in LDS.
constexpr unsigned kOb4Heads = 8, kOb4Dims = 64, kOb4Steps = 8, kOb4M = 9, kOb4Slots = 4 * kOb4Steps * kOb4M;  // 288
static_assert((kMaxTokens + 255) / 256 <= kOb4M, "o-bar v4: 9 rounds of 256 cover the list");
__global__ __launch_bounds__(256) void mla_obar_v4_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all, unsigned heads,
                                                          const float* e_all, float* ob) {
    __shared__ alignas(16) std::uint16_t lt[kOb4Slots][kOb4Dims];  // 36 KB
    __shared__ float ps[kOb4Heads][kOb4Slots];                      // 9 KB
    __shared__ float roots[4][kOb4Heads][kOb4Dims];                 // 8 KB: subtree roots of sub-groups 0..3
    const unsigned hb = blockIdx.x * kOb4Heads, d0 = blockIdx.y * kOb4Dims, r = blockIdx.z, n = n_all[r];
    const unsigned tid = threadIdx.x, dq = tid % 16, hp = (tid / 16) % 4, sg = tid / 64;
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    float stk[2][4][7];  // per output: binary-counter stack of the sub-group's 64-lane subtree
    unsigned cnt = 0;
    for (unsigned b = 0; b < 64 / kOb4Steps; ++b) {
        __syncthreads();  // previous batch consumed
        // stage: slot = (s * 8 + j) * 9 + m  <->  token t = bitrev8(64 s + 8 b + j) + 256 m
        for (unsigned x = tid; x < kOb4Slots * (kOb4Dims / 8); x += 256) {
            const unsigned slot = x / (kOb4Dims / 8), c8 = x % (kOb4Dims / 8);
            const unsigned lane = slot / kOb4M, m = slot % kOb4M;
            const unsigned t = bitrev8(64 * (lane / kOb4Steps) + kOb4Steps * b + lane % kOb4Steps) + 256 * m;
            if (t < n)
                *reinterpret_cast<uint4*>(&lt[slot][c8 * 8]) =
                    *reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(tok[t]) * kKvLora + d0 + c8 * 8);
        }
        for (unsigned x = tid; x < kOb4Heads * kOb4Slots; x += 256) {
            const unsigned h = x / kOb4Slots, slot = x % kOb4Slots;
            const unsigned lane = slot / kOb4M, m = slot % kOb4M;
            const unsigned t = bitrev8(64 * (lane / kOb4Steps) + kOb4Steps * b + lane % kOb4Steps) + 256 * m;
            if (t < n) ps[h][slot] = e_all[(static_cast<std::size_t>(r) * heads + hb + h) * kEStride + t];
        }
        __syncthreads();
        for (unsigned j = 0; j < kOb4Steps; ++j) {
            const unsigned l = bitrev8(64 * sg + kOb4Steps * b + j), slot0 = (sg * kOb4Steps + j) * kOb4M;
            float c[2][4];
#pragma unroll
            for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                for (unsigned q = 0; q < 4; ++q) c[a][q] = 0.0F;
            for (unsigned m = 0; m < kOb4M && l + 256 * m < n; ++m) {
                const uint2 v = *reinterpret_cast<const uint2*>(&lt[slot0 + m][dq * 4]);
                const float cv[4] = {bf16_to_float(static_cast<std::uint16_t>(v.x)), bf16_to_float(static_cast<std::uint16_t>(v.x >> 16)),
                                     bf16_to_float(static_cast<std::uint16_t>(v.y)), bf16_to_float(static_cast<std::uint16_t>(v.y >> 16))};
                const float p0 = ps[2 * hp][slot0 + m], p1 = ps[2 * hp + 1][slot0 + m];
#pragma unroll
                for (unsigned q = 0; q < 4; ++q) {
                    c[0][q] = __builtin_fmaf(p0, cv[q], c[0][q]);
                    c[1][q] = __builtin_fmaf(p1, cv[q], c[1][q]);
                }
            }
            // binary-counter merge of this lane's chains: carry through the counter's set low bits, place at the first
            // clear one. Levels are compile-time indices (registers, no scratch); cnt is uniform, so no divergence.
            bool placed = false;
#pragma unroll
            for (unsigned L = 0; L < 7; ++L) {
                if (placed) continue;
                if ((cnt >> L) & 1U) {
#pragma unroll
                    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) c[a][q] = stk[a][q][L] + c[a][q];
                } else {
#pragma unroll
                    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) stk[a][q][L] = c[a][q];
                    placed = true;
                }
            }
            ++cnt;
        }
    }
    // 64 lanes merged: the sub-group's subtree root sits at level 6
#pragma unroll
    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
        for (unsigned q = 0; q < 4; ++q) roots[sg][2 * hp + a][dq * 4 + q] = stk[a][q][6];
    __syncthreads();
    for (unsigned x = tid; x < kOb4Heads * kOb4Dims; x += 256) {
        const unsigned h = x / kOb4Dims, d = x % kOb4Dims;
        const float v = (roots[0][h][d] + roots[1][h][d]) + (roots[2][h][d] + roots[3][h][d]);
        ob[(static_cast<std::size_t>(r) * heads + hb + h) * kKvLora + d0 + d] = v;
    }
}

// o-bar v5 (heads 32): CTA = all 32 heads x 64 latent dims of one row (latent gathered once per row); thread = 2 heads
// x 4 dims over the full 256-lane tree. Lanes in k order (l = bitrev8(k)): lane l's chain fmaf(p_t, c_t, a) over
// t = l + 256 m ascending from +0, only for t < n (no padding FMAs), then the binary-counter merge (older / lower-k
// subtree + newer one) of design notes section 2: the 256-lane adjacent-pair tree = tree_fma<256>'s fold.
// p comes lane-major from mla_softmax_rows_kernel<true> (a batch of 8 k is 72 contiguous slots per head). Staging is
// double-buffered: batch b + 1's p / latent are loaded into registers before batch b is computed and stored to the
// other buffer after it; one barrier per batch.
constexpr unsigned kOb5Dims = 64, kOb5Steps = 8, kOb5Slots = kOb5Steps * kObRounds;  // 72
constexpr unsigned kOb5PVec = 32 * kOb5Slots / 4, kOb5LVec = kOb5Slots * kOb5Dims / 8;  // 576 float4, 576 uint4
__global__ __launch_bounds__(256) void mla_obar_v5_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                          const float* e_all, float* ob) {
    __shared__ alignas(16) std::uint16_t lt[2][kOb5Slots][kOb5Dims];  // 18 KB
    __shared__ alignas(16) float ps[2][kOb5Slots][32];                 // 18 KB
    const unsigned d0 = blockIdx.x * kOb5Dims, r = blockIdx.y, n = n_all[r];
    const unsigned tid = threadIdx.x, dq = tid % 16, hp = tid / 16;  // dims d0 + 4 dq .. + 3, heads 2 hp, 2 hp + 1
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    const float* ep = e_all + static_cast<std::size_t>(r) * 32 * kEStride;
    float4 pr[3];
    uint4 lr[3];
    auto load = [&](unsigned b) {  // batch b (k = 8 b .. 8 b + 7) into registers
#pragma unroll
        for (unsigned i = 0; i < 3; ++i) {
            const unsigned x = tid + 256 * i;
            if (x < kOb5PVec) {
                const unsigned h = x / (kOb5Slots / 4), q = x % (kOb5Slots / 4);
                pr[i] = *reinterpret_cast<const float4*>(ep + static_cast<std::size_t>(h) * kEStride + kOb5Slots * b + 4 * q);
            }
            if (x < kOb5LVec) {
                const unsigned slot = x / (kOb5Dims / 8), c8 = x % (kOb5Dims / 8);
                const unsigned t = bitrev8(kOb5Steps * b + slot / kObRounds) + 256 * (slot % kObRounds);
                lr[i] = t < n ? *reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(tok[t]) * kKvLora + d0 + 8 * c8)
                              : uint4{0, 0, 0, 0};
            }
        }
    };
    auto store = [&](unsigned buf) {
#pragma unroll
        for (unsigned i = 0; i < 3; ++i) {
            const unsigned x = tid + 256 * i;
            if (x < kOb5PVec) {
                const unsigned h = x / (kOb5Slots / 4), q = x % (kOb5Slots / 4);
                ps[buf][4 * q + 0][h] = pr[i].x;
                ps[buf][4 * q + 1][h] = pr[i].y;
                ps[buf][4 * q + 2][h] = pr[i].z;
                ps[buf][4 * q + 3][h] = pr[i].w;
            }
            if (x < kOb5LVec) {
                const unsigned slot = x / (kOb5Dims / 8), c8 = x % (kOb5Dims / 8);
                *reinterpret_cast<uint4*>(&lt[buf][slot][8 * c8]) = lr[i];
            }
        }
    };
    float stk[2][4][8];  // binary-counter levels 0..7; the root (level 8) is formed by the last merge (k = 255)
    float root[2][4];
    load(0);
    store(0);
    __syncthreads();
    for (unsigned b = 0; b < 256 / kOb5Steps; ++b) {
        const unsigned cur = b & 1U;
        if (b + 1 < 256 / kOb5Steps) load(b + 1);
        for (unsigned j = 0; j < kOb5Steps; ++j) {
            const unsigned k = kOb5Steps * b + j, l = bitrev8(k);
            float c[2][4];
#pragma unroll
            for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                for (unsigned q = 0; q < 4; ++q) c[a][q] = 0.0F;
            for (unsigned m = 0; m < kObRounds && l + 256 * m < n; ++m) {
                const unsigned slot = kObRounds * j + m;
                const uint2 v = *reinterpret_cast<const uint2*>(&lt[cur][slot][4 * dq]);
                const float cv[4] = {bf16_to_float(static_cast<std::uint16_t>(v.x)), bf16_to_float(static_cast<std::uint16_t>(v.x >> 16)),
                                     bf16_to_float(static_cast<std::uint16_t>(v.y)), bf16_to_float(static_cast<std::uint16_t>(v.y >> 16))};
                const float2 pp = *reinterpret_cast<const float2*>(&ps[cur][slot][2 * hp]);
#pragma unroll
                for (unsigned q = 0; q < 4; ++q) {
                    c[0][q] = __builtin_fmaf(pp.x, cv[q], c[0][q]);
                    c[1][q] = __builtin_fmaf(pp.y, cv[q], c[1][q]);
                }
            }
            // merge: carry through the set low bits of k (left operand = the stored, lower-k subtree), place at the
            // first clear bit; k = 255 carries through all 8 levels into the root. Levels are compile-time indices.
            bool placed = false;
#pragma unroll
            for (unsigned L = 0; L < 8; ++L) {
                if (placed) continue;
                if ((k >> L) & 1U) {
#pragma unroll
                    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) c[a][q] = stk[a][q][L] + c[a][q];
                } else {
#pragma unroll
                    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) stk[a][q][L] = c[a][q];
                    placed = true;
                }
            }
            if (!placed)
#pragma unroll
                for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                    for (unsigned q = 0; q < 4; ++q) root[a][q] = c[a][q];
        }
        if (b + 1 < 256 / kOb5Steps) store(cur ^ 1U);
        __syncthreads();
    }
#pragma unroll
    for (unsigned a = 0; a < 2; ++a)
        *reinterpret_cast<float4*>(ob + (static_cast<std::size_t>(r) * 32 + 2 * hp + a) * kKvLora + d0 + 4 * dq) =
            float4{root[a][0], root[a][1], root[a][2], root[a][3]};
}

// o-bar v6: four independent 64-lane subtrees, each with the v5 lane chains and ordered
// binary-counter merges. Only the final two levels cross sub-groups through LDS.
// CTA = 16 heads x 16 dims; thread = 2 heads x 2 dims of one subtree (28 stack floats).
// Double-buffer 4 k-steps per subtree, prefetching the next gather before computation.
// Token-order probabilities support both 32 and 64 heads. LDS: 9 + 18 + 4 = 31 KiB.
constexpr unsigned kOb6Heads = 16, kOb6Dims = 16, kOb6Steps = 4, kOb6M = kObRounds;
constexpr unsigned kOb6Slots = 4 * kOb6Steps * kOb6M;  // 144
__global__ __launch_bounds__(256) void mla_obar_v6_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all, unsigned heads,
                                                          const float* e_all, float* ob) {
    __shared__ alignas(16) std::uint16_t lt[2][kOb6Slots][kOb6Dims];
    __shared__ float ps[2][kOb6Heads][kOb6Slots];
    __shared__ float roots[4][kOb6Heads][kOb6Dims];
    const unsigned hb = blockIdx.x * kOb6Heads, d0 = blockIdx.y * kOb6Dims, r = blockIdx.z, n = n_all[r];
    const unsigned tid = threadIdx.x, dq = tid % 8, hp = (tid / 8) % 8, sg = tid / 64;
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    float stk[2][2][7];  // per output: binary-counter stack of the sub-group's 64-lane subtree
    float pr[kOb6Heads * kOb6Slots / 256];
    uint4 lr[2];
    auto load = [&](unsigned b) {
#pragma unroll
        for (unsigned i = 0; i < 2; ++i) {
            const unsigned x = tid + 256 * i;
            if (x < kOb6Slots * (kOb6Dims / 8)) {
                const unsigned slot = x / (kOb6Dims / 8), c8 = x % (kOb6Dims / 8);
                const unsigned lane = slot / kOb6M, m = slot % kOb6M;
                const unsigned t = bitrev8(64 * (lane / kOb6Steps) + kOb6Steps * b + lane % kOb6Steps) + 256 * m;
                lr[i] = t < n ? *reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(tok[t]) * kKvLora + d0 + c8 * 8)
                              : uint4{0, 0, 0, 0};
            }
        }
#pragma unroll
        for (unsigned i = 0; i < kOb6Heads * kOb6Slots / 256; ++i) {
            const unsigned x = tid + 256 * i, h = x / kOb6Slots, slot = x % kOb6Slots;
            const unsigned lane = slot / kOb6M, m = slot % kOb6M;
            const unsigned t = bitrev8(64 * (lane / kOb6Steps) + kOb6Steps * b + lane % kOb6Steps) + 256 * m;
            pr[i] = t < n ? e_all[(static_cast<std::size_t>(r) * heads + hb + h) * kEStride + t] : 0.0F;
        }
    };
    auto store = [&](unsigned buf) {
#pragma unroll
        for (unsigned i = 0; i < 2; ++i) {
            const unsigned x = tid + 256 * i;
            if (x < kOb6Slots * (kOb6Dims / 8)) {
                const unsigned slot = x / (kOb6Dims / 8), c8 = x % (kOb6Dims / 8);
                *reinterpret_cast<uint4*>(&lt[buf][slot][c8 * 8]) = lr[i];
            }
        }
#pragma unroll
        for (unsigned i = 0; i < kOb6Heads * kOb6Slots / 256; ++i) {
            const unsigned x = tid + 256 * i;
            ps[buf][x / kOb6Slots][x % kOb6Slots] = pr[i];
        }
    };
    unsigned cnt = 0;
    load(0);
    store(0);
    __syncthreads();
    for (unsigned b = 0; b < 64 / kOb6Steps; ++b) {
        const unsigned cur = b & 1U;
        if (b + 1 < 64 / kOb6Steps) load(b + 1);
        for (unsigned j = 0; j < kOb6Steps; ++j) {
            const unsigned l = bitrev8(64 * sg + kOb6Steps * b + j), slot0 = (sg * kOb6Steps + j) * kOb6M;
            float c[2][2];
#pragma unroll
            for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                for (unsigned q = 0; q < 2; ++q) c[a][q] = 0.0F;
            for (unsigned m = 0; m < kOb6M && l + 256 * m < n; ++m) {
                const std::uint32_t v = *reinterpret_cast<const std::uint32_t*>(&lt[cur][slot0 + m][dq * 2]);
                const float cv[2] = {bf16_to_float(static_cast<std::uint16_t>(v)), bf16_to_float(static_cast<std::uint16_t>(v >> 16))};
                const float p0 = ps[cur][2 * hp][slot0 + m], p1 = ps[cur][2 * hp + 1][slot0 + m];
#pragma unroll
                for (unsigned q = 0; q < 2; ++q) {
                    c[0][q] = __builtin_fmaf(p0, cv[q], c[0][q]);
                    c[1][q] = __builtin_fmaf(p1, cv[q], c[1][q]);
                }
            }
            // binary-counter merge of this lane's chains: carry through the counter's set low bits, place at the first
            // clear one. Levels are compile-time indices for scalar registers; cnt is uniform, so no divergence.
            bool placed = false;
#pragma unroll
            for (unsigned L = 0; L < 7; ++L) {
                if (placed) continue;
                if ((cnt >> L) & 1U) {
#pragma unroll
                    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 2; ++q) c[a][q] = stk[a][q][L] + c[a][q];
                } else {
#pragma unroll
                    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 2; ++q) stk[a][q][L] = c[a][q];
                    placed = true;
                }
            }
            ++cnt;
        }
        if (b + 1 < 64 / kOb6Steps) store(cur ^ 1U);
        __syncthreads();
    }
    // 64 lanes merged: the sub-group's subtree root sits at level 6
#pragma unroll
    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
        for (unsigned q = 0; q < 2; ++q) roots[sg][2 * hp + a][dq * 2 + q] = stk[a][q][6];
    __syncthreads();
    for (unsigned x = tid; x < kOb6Heads * kOb6Dims; x += 256) {
        const unsigned h = x / kOb6Dims, d = x % kOb6Dims;
        const float v = (roots[0][h][d] + roots[1][h][d]) + (roots[2][h][d] + roots[3][h][d]);
        ob[(static_cast<std::size_t>(r) * heads + hb + h) * kKvLora + d0 + d] = v;
    }
}

// o-bar v7 (heads 32): v5's lane chains and 256-lane binary-counter merge with another placement. CTA = all 32 heads x
// 128 latent dims of one row; wave w owns heads 4 w .. 4 w + 3, lane the dims d0 + 4 lane .. + 3 (16 outputs per
// thread). The probabilities are wave-uniform, so they come through scalar loads from the lane-major layout (9
// contiguous slots per head and k); the latent of a batch of 4 k (36 slots, 128 dims) is staged in LDS as f32 (the exact
// bf16 widening, done once per CTA instead of per thread), double buffered, one barrier per batch. Inner step: one LDS
// read + 16 FMAs with an SGPR operand.
constexpr unsigned kOb7Dims = 128, kOb7Steps = 4, kOb7Slots = kOb7Steps * kObRounds;  // 36
constexpr unsigned kOb7LVec = kOb7Slots * kOb7Dims / 8, kOb7Ld = (kOb7LVec + 255) / 256;  // 576 uint4 (8 bf16), 3 per thread
__global__ __launch_bounds__(256) void mla_obar_v7_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                          const float* e_all, float* ob) {
    __shared__ alignas(16) float lt[2][kOb7Slots][kOb7Dims];  // 36 KB
    const unsigned d0 = blockIdx.x * kOb7Dims, r = blockIdx.y, n = n_all[r];
    const unsigned tid = threadIdx.x, lane = tid % 32, w = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(tid / 32)));
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    const auto* pw = (ConstF*)(e_all + (static_cast<std::size_t>(r) * 32 + 4 * w) * kEStride);  // NOLINT: address-space cast (heads 4 w ..)
    uint4 lr[kOb7Ld];
    auto load = [&](unsigned b) {  // batch b (k = 4 b .. 4 b + 3) into registers
#pragma unroll
        for (unsigned i = 0; i < kOb7Ld; ++i) {
            const unsigned x = tid + 256 * i;
            if (x < kOb7LVec) {
                const unsigned slot = x / (kOb7Dims / 8), c8 = x % (kOb7Dims / 8);
                const unsigned t = bitrev8(kOb7Steps * b + slot / kObRounds) + 256 * (slot % kObRounds);
                lr[i] = t < n ? *reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(tok[t]) * kKvLora + d0 + 8 * c8)
                              : uint4{0, 0, 0, 0};
            }
        }
    };
    auto store = [&](unsigned buf) {
#pragma unroll
        for (unsigned i = 0; i < kOb7Ld; ++i) {
            const unsigned x = tid + 256 * i;
            if (x < kOb7LVec) {
                const unsigned slot = x / (kOb7Dims / 8), c8 = x % (kOb7Dims / 8);
                const std::uint32_t wv[4] = {lr[i].x, lr[i].y, lr[i].z, lr[i].w};
                float f[8];
#pragma unroll
                for (unsigned ii = 0; ii < 8; ++ii) f[ii] = bf16_to_float(static_cast<std::uint16_t>(wv[ii / 2] >> (16 * (ii % 2))));
                float4* dst = reinterpret_cast<float4*>(&lt[buf][slot][8 * c8]);
                dst[0] = float4{f[0], f[1], f[2], f[3]};
                dst[1] = float4{f[4], f[5], f[6], f[7]};
            }
        }
    };
    float stk[4][4][8];  // binary-counter levels 0..7; the root (level 8) is formed by the last merge (k = 255) and stored
    load(0);
    store(0);
    __syncthreads();
    for (unsigned b = 0; b < 256 / kOb7Steps; ++b) {
        const unsigned cur = b & 1U;
        if (b + 1 < 256 / kOb7Steps) load(b + 1);
        for (unsigned j = 0; j < kOb7Steps; ++j) {
            const unsigned k = kOb7Steps * b + j, l = bitrev8(k);
            float p[4][kObRounds];
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned m = 0; m < kObRounds; ++m) p[a][m] = pw[a * kEStride + kObRounds * k + m];  // padding slots are +0
            float c[4][4];
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned q = 0; q < 4; ++q) c[a][q] = 0.0F;
#pragma unroll
            for (unsigned m = 0; m < kObRounds; ++m) {
                if (l + 256 * m < n) {  // uniform; true for a prefix of m
                    const float4 cv4 = *reinterpret_cast<const float4*>(&lt[cur][kObRounds * j + m][4 * lane]);
                    const float cv[4] = {cv4.x, cv4.y, cv4.z, cv4.w};
#pragma unroll
                    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) c[a][q] = __builtin_fmaf(p[a][m], cv[q], c[a][q]);
                }
            }
            // merge as v5: carry through the set low bits of k (left operand = the stored, lower-k subtree), place at
            // the first clear bit; k = 255 carries through all 8 levels and its result is the root.
            bool placed = false;
#pragma unroll
            for (unsigned L = 0; L < 8; ++L) {
                if (placed) continue;
                if ((k >> L) & 1U) {
#pragma unroll
                    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) c[a][q] = stk[a][q][L] + c[a][q];
                } else {
#pragma unroll
                    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) stk[a][q][L] = c[a][q];
                    placed = true;
                }
            }
            if (!placed)
#pragma unroll
                for (unsigned a = 0; a < 4; ++a)
                    *reinterpret_cast<float4*>(ob + (static_cast<std::size_t>(r) * 32 + 4 * w + a) * kKvLora + d0 + 4 * lane) =
                        float4{c[a][0], c[a][1], c[a][2], c[a][3]};
        }
        if (b + 1 < 256 / kOb7Steps) store(cur ^ 1U);
        __syncthreads();
    }
}

// o-bar v2: grid (latent block of 8 dims, row). The row's latent tile [n][8] bf16 is staged in LDS once; per pass of 4
// heads, lane j accumulates 4 x 8 fmaf chains over t = j, j + 256, ... (ascending t), then lds_tree<256>'s pairing:
// levels 128 / 64 / 32 through LDS, levels 16 .. 1 by __shfl_down inside wave 0 (lane l < h receives lane l + h).
constexpr unsigned kOb2Lat = 8, kOb2Heads = 4, kOb2Vals = kOb2Lat * kOb2Heads;
__global__ __launch_bounds__(256) void mla_obar_v2_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all, unsigned heads,
                                                          const float* e_all, float* ob) {
    __shared__ uint4 lt[kMaxTokens];             // latent dims [lb, lb + 8) of each listed token (16 B)
    __shared__ float buf[kOb2Vals][128];
    const unsigned lb = blockIdx.x * kOb2Lat, r = blockIdx.y, j = threadIdx.x, n = n_all[r];
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    for (unsigned t = j; t < n; t += 256) lt[t] = *reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(tok[t]) * kKvLora + lb);
    for (unsigned h0 = 0; h0 < heads; h0 += kOb2Heads) {
        float acc[kOb2Heads][kOb2Lat];
#pragma unroll
        for (unsigned a = 0; a < kOb2Heads; ++a)
#pragma unroll
            for (unsigned c = 0; c < kOb2Lat; ++c) acc[a][c] = 0.0F;
        __syncthreads();  // lt staged (first pass) / buf free (later passes)
        for (unsigned t = j; t < n; t += 256) {
            const uint4 v = lt[t];
            const std::uint32_t wv[4] = {v.x, v.y, v.z, v.w};
            float p[kOb2Heads];
#pragma unroll
            for (unsigned a = 0; a < kOb2Heads; ++a) p[a] = e_all[(static_cast<std::size_t>(r) * heads + h0 + a) * kEStride + t];
#pragma unroll
            for (unsigned c = 0; c < kOb2Lat; ++c) {
                const float cv = bf16_to_float(static_cast<std::uint16_t>(wv[c / 2] >> (16 * (c % 2))));
#pragma unroll
                for (unsigned a = 0; a < kOb2Heads; ++a) acc[a][c] = __builtin_fmaf(p[a], cv, acc[a][c]);
            }
        }
        for (unsigned hh = 128; hh >= 32; hh >>= 1U) {  // cross-wave levels: lanes [hh, 2hh) hand their values to l - hh
            if (j >= hh && j < 2 * hh)
#pragma unroll
                for (unsigned a = 0; a < kOb2Heads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kOb2Lat; ++c) buf[a * kOb2Lat + c][j - hh] = acc[a][c];
            __syncthreads();
            if (j < hh)
#pragma unroll
                for (unsigned a = 0; a < kOb2Heads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kOb2Lat; ++c) acc[a][c] = acc[a][c] + buf[a * kOb2Lat + c][j];
            __syncthreads();
        }
        if (j < 32) {
#pragma unroll
            for (unsigned hh = 16; hh > 0; hh >>= 1U)
#pragma unroll
                for (unsigned a = 0; a < kOb2Heads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kOb2Lat; ++c) acc[a][c] = acc[a][c] + __shfl_down(acc[a][c], hh, 32);
            if (j == 0)
#pragma unroll
                for (unsigned a = 0; a < kOb2Heads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kOb2Lat; ++c) ob[(static_cast<std::size_t>(r) * heads + h0 + a) * kKvLora + lb + c] = acc[a][c];
        }
    }
}

// R1 chains over rows: tiled VALU GEMM, 64 rows x 64 outputs per CTA, K step 32, thread = 4 x 4 outputs; per output
// the chain of dense_kernel (fmaf(x[i], w[o][i], acc), i ascending, from +0).
__device__ __forceinline__ float wload_pf(const void* w, dev::WType t, std::size_t k) {
    switch (t) {
        case dev::WType::F16: return f16_to_float(static_cast<const std::uint16_t*>(w)[k]);
        case dev::WType::BF16: return bf16_to_float(static_cast<const std::uint16_t*>(w)[k]);
        default: return static_cast<const float*>(w)[k];
    }
}
__global__ __launch_bounds__(256) void dense_rows_kernel(const float* x, const void* w, dev::WType wt, unsigned in, unsigned o0, unsigned n,
                                                         float* y, unsigned rows) {
    __shared__ float as[kGT][kGK + 1];
    __shared__ float ws[kGK][kGT + 1];
    const unsigned r0 = blockIdx.x * kGT, c0 = blockIdx.y * kGT, tid = threadIdx.x, tr = tid / 16, tc = tid % 16;
    float acc[4][4];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) acc[i][j] = 0.0F;
    for (unsigned k0 = 0; k0 < in; k0 += kGK) {
        __syncthreads();
        for (unsigned v = tid; v < kGT * kGK; v += 256) {
            const unsigned rr = v / kGK, kk = v % kGK, r = r0 + rr;
            as[rr][kk] = r < rows ? x[static_cast<std::size_t>(r) * in + k0 + kk] : 0.0F;
            const unsigned o = c0 + rr;  // the same index pair for the weight tile: row o, column k0 + kk
            ws[kk][rr] = o < n ? wload_pf(w, wt, static_cast<std::size_t>(o0 + o) * in + k0 + kk) : 0.0F;
        }
        __syncthreads();
#pragma unroll 4
        for (unsigned kk = 0; kk < kGK; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) a[i] = as[tr * 4 + i][kk];
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) b[j] = ws[kk][tc * 4 + j];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) acc[i][j] = __builtin_fmaf(a[i], b[j], acc[i][j]);
        }
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned r = r0 + tr * 4 + i;
        if (r >= rows) continue;
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned o = c0 + tc * 4 + j;
            if (o < n) y[static_cast<std::size_t>(r) * n + o] = acc[i][j];
        }
    }
}

// hc_pre split in three launches (expressions of hc_pre_kernel): (1) per-row unweighted norm -> F; (2) the 24 chains
// z = F . fn as dense_rows; (3) per-row R6 on lane 0 + R7 collapse.
__global__ __launch_bounds__(256) void hc_norm_rows_kernel(const float* X, float* F) {
    constexpr unsigned n = kHc * kHidden;
    __shared__ float s[256];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * n;
    float* Fr = F + static_cast<std::size_t>(r) * n;
    float a = 0.0F;
    for (unsigned i = l; i < n; i += 256) a = __builtin_fmaf(Xr[i], Xr[i], a);
    const float ss = lds_tree<256>(s, a);
    const float rr = inv_sqrt(ss / static_cast<float>(n) + kEps);
    for (unsigned i = l; i < n; i += 256) Fr[i] = Xr[i] * rr;
}
// hc_norm_rows_kernel's factor only (same chain and tree), for the split form STRIX_GLM_PF_HC_FUSE=2: F is not written.
__global__ __launch_bounds__(256) void hc_norm_factor_rows_kernel(const float* X, float* rr_out) {
    constexpr unsigned n = kHc * kHidden;
    __shared__ float s[256];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * n;
    float a = 0.0F;
    for (unsigned i = l; i < n; i += 256) a = __builtin_fmaf(Xr[i], Xr[i], a);
    const float ss = lds_tree256_lane0(s, a);
    if (l == 0) rr_out[r] = inv_sqrt(ss / static_cast<float>(n) + kEps);
}
// hc_expand (dev::hc_expand_kernel expressions) fused with the next site's hc_norm_rows_kernel: lane l keeps the 64
// outputs it owns in the norm (element l + 256 j, j = stream * 16 + block) and runs the same fmaf chain in ascending j
// and the same lds_tree; it stores rr per row and F is never materialised: the chain kernel forms F = X * rr on load
// (one product, as hc_norm_rows writes it). The file is built with -ffp-contract=off, as is the expand kernel.
__global__ __launch_bounds__(256) void hc_expand_norm_rows_kernel(const float* X, const float* mix, const float* y, float* Xo,
                                                                  float* rr_out, unsigned* flag) {
    constexpr unsigned n = kHc * kHidden, B = kHidden / 256;
    __shared__ float s[256];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* m = mix + static_cast<std::size_t>(r) * kHcMix;
    const float* c = m + 2 * kHc;
    const float* Xr = X + static_cast<std::size_t>(r) * n;
    float* Xor = Xo + static_cast<std::size_t>(r) * n;
    float v[kHc - 1][B];  // streams 1 .. 3; stream 0 enters the chain as produced (its terms come first in ascending j)
    float a = 0.0F;
    bool bad = false;
#pragma unroll
    for (unsigned b = 0; b < B; ++b) {
        const unsigned d = b * 256 + l;
        const float yd = y[static_cast<std::size_t>(r) * kHidden + d];
#pragma unroll
        for (unsigned i = 0; i < kHc; ++i) {
            const float mx = ((c[0 * kHc + i] * Xr[d] + c[1 * kHc + i] * Xr[kHidden + d]) + c[2 * kHc + i] * Xr[2 * kHidden + d]) +
                             c[3 * kHc + i] * Xr[3 * kHidden + d];
            const float o = m[kHc + i] * yd + mx;
            bad = bad || !is_finite(o);
            Xor[i * kHidden + d] = o;
            if (i == 0) a = __builtin_fmaf(o, o, a);
            else v[i - 1][b] = o;
        }
    }
    if (flag && bad) atomicOr(flag, 1U);
#pragma unroll
    for (unsigned i = 0; i < kHc - 1; ++i)
#pragma unroll
        for (unsigned b = 0; b < B; ++b) a = __builtin_fmaf(v[i][b], v[i][b], a);
    const float ss = lds_tree256_lane0(s, a);
    if (l == 0) rr_out[r] = inv_sqrt(ss / static_cast<float>(n) + kEps);
}
// Serial R6, shared by both work placements. Preserve every expression and loop order.
__device__ __forceinline__ void hc_mix_r6(const float* z, const float* base, const float* scale, float* m) {
    for (unsigned j = 0; j < kHc; ++j) m[j] = sigmoid(z[j] * scale[0] + base[j]) + kHcEps;
    for (unsigned j = 0; j < kHc; ++j) m[kHc + j] = 2.0F * sigmoid(z[kHc + j] * scale[1] + base[kHc + j]);
    float* c = m + 2 * kHc;
    for (unsigned k = 0; k < kHc * kHc; ++k) c[k] = z[2 * kHc + k] * scale[2] + base[2 * kHc + k];
    for (unsigned j = 0; j < kHc; ++j) {
        float mx = c[j * kHc];
        for (unsigned k = 1; k < kHc; ++k) mx = c[j * kHc + k] > mx ? c[j * kHc + k] : mx;
        float e[kHc];
        for (unsigned k = 0; k < kHc; ++k) e[k] = declared_exp(c[j * kHc + k] - mx);
        const float sum = ((e[0] + e[1]) + e[2]) + e[3];
        for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = e[k] / sum + kHcEps;
    }
    for (unsigned it = 0; it < kHcIters; ++it) {
        if (it > 0) {
            float sr[kHc];
            for (unsigned j = 0; j < kHc; ++j)
                sr[j] = (((c[j * kHc + 0] + c[j * kHc + 1]) + c[j * kHc + 2]) + c[j * kHc + 3]) + kHcEps;
            for (unsigned j = 0; j < kHc; ++j)
                for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sr[j];
        }
        float sc[kHc];
        for (unsigned k = 0; k < kHc; ++k)
            sc[k] = (((c[0 * kHc + k] + c[1 * kHc + k]) + c[2 * kHc + k]) + c[3 * kHc + k]) + kHcEps;
        for (unsigned j = 0; j < kHc; ++j)
            for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sc[k];
    }
}
__global__ __launch_bounds__(256) void hc_mix_rows_kernel(const float* X, const float* zg, const float* base, const float* scale, float* mix,
                                                          float* xc) {
    __shared__ float m[kHcMix];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
    if (l == 0) {  // R6 on lane 0 (host hc_mix order) = hc_pre_kernel
        const float* z = zg + static_cast<std::size_t>(r) * kHcMix;
        hc_mix_r6(z, base, scale, m);
        for (unsigned k = 0; k < kHcMix; ++k) mix[static_cast<std::size_t>(r) * kHcMix + k] = m[k];
    }
    __syncthreads();
    float* xr = xc + static_cast<std::size_t>(r) * kHidden;
    for (unsigned d = l; d < kHidden; d += 256)
        xr[d] = ((m[0] * Xr[d] + m[1] * Xr[kHidden + d]) + m[2] * Xr[2 * kHidden + d]) + m[3] * Xr[3 * kHidden + d];
}

// V2 R1: 64 rows x 24 outputs, K=32, 128 threads, 4 x 3 live chains per thread.
// Each stage is loaded once per CTA. Prefetch the next tile to registers before consuming
// the current LDS stage, then publish to the other stage. The tile-end barrier protects
// both reuse of the old stage and reads of the newly filled stage.
constexpr unsigned kHcK = 32, kHcMixRows = 16;
static_assert(kHc * kHidden == 16384 && kHcMix == 24);
// Rows-row tiles (64 / 32 / 16), 2 * Rows threads, 4 x 3 chains per thread: smaller tiles give more CTAs (the K chain
// cannot be split), e.g. 2048 rows -> 32 / 64 / 128 CTAs. Every tile size gives the same bits (one fmaf chain per
// (row, output), k ascending from +0). Cols = 24 (all outputs per CTA) or 8 (grid.y = 3 output groups, 4 x 1 chains per
// thread: three times the CTAs for short calls, activations loaded once per group; same bits).
template <unsigned Rows, bool FromX = false, unsigned Cols = kHcMix>
__device__ __forceinline__ void hc_chain_prefetch(const float* F, const float* fn, float* next_a, float* next_w,
                                                 unsigned r0, unsigned rows, unsigned k0) {
    constexpr unsigned n = kHc * kHidden, T = 2 * Rows;
    const unsigned tid = threadIdx.x;
#pragma unroll
    for (unsigned q = 0; q < Rows * kHcK / T; ++q) {
        const unsigned v = tid + q * T, r = v / kHcK, k = v % kHcK;
        // FromX: raw X here; the stage forms F = X * rr when it publishes to LDS, so the loads stay in flight
        // behind the current tile's FMAs
        next_a[q] = r0 + r < rows ? F[static_cast<std::size_t>(r0 + r) * n + k0 + k] : 0.0F;
    }
#pragma unroll
    for (unsigned q = 0; q < Cols * kHcK / T; ++q) {
        const unsigned v = tid + q * T, o = (Cols == kHcMix ? 0U : blockIdx.y * Cols) + v / kHcK, k = v % kHcK;
        next_w[q] = fn[static_cast<std::size_t>(o) * n + k0 + k];
    }
}
template <unsigned Rows, unsigned Cols = kHcMix, bool FromX = false>
__device__ __forceinline__ void hc_chain_stage(const float* next_a, const float* next_w,
                                              float as[][kHcK + 1], float ws[][Cols + 1], const float* rrq = nullptr) {
    constexpr unsigned T = 2 * Rows;
    const unsigned tid = threadIdx.x;
#pragma unroll
    for (unsigned q = 0; q < Rows * kHcK / T; ++q) {
        const unsigned v = tid + q * T;
        // F = X * rr exactly as hc_norm_rows_kernel stores it (rrq[q]: this element's row factor; padding rows 0 * 0 = +0)
        if constexpr (FromX) as[v / kHcK][v % kHcK] = next_a[q] * rrq[q];
        else as[v / kHcK][v % kHcK] = next_a[q];
    }
#pragma unroll
    for (unsigned q = 0; q < Cols * kHcK / T; ++q) {
        const unsigned v = tid + q * T;
        ws[v % kHcK][v / kHcK] = next_w[q];
    }
}
template <unsigned Rows, bool FromX = false, unsigned Cols = kHcMix>
__global__ __launch_bounds__(2 * Rows) void hc_chain_rows_v2_kernel(const float* F, const float* fn, float* z, unsigned rows,
                                                                    const float* rr = nullptr) {
    constexpr unsigned n = kHc * kHidden, T = 2 * Rows, J = Cols / 8;
    static_assert(Cols * kHcK % T == 0 && Rows * kHcK % T == 0 && Cols % 8 == 0);
    __shared__ float as[2][Rows][kHcK + 1];
    __shared__ float ws[2][kHcK][Cols + 1];
    const unsigned r0 = blockIdx.x * Rows, tid = threadIdx.x, tr = tid / 8, tc = tid % 8, c0 = Cols == kHcMix ? 0U : blockIdx.y * Cols;
    float acc[4][J];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < J; ++j) acc[i][j] = 0.0F;
    float next_a[Rows * kHcK / T], next_w[Cols * kHcK / T];
    float rrq[Rows * kHcK / T] = {};  // a thread's rows are the same at every k0: load their factors once
    if constexpr (FromX) {
#pragma unroll
        for (unsigned q = 0; q < Rows * kHcK / T; ++q) {
            const unsigned r = (tid + q * T) / kHcK;
            rrq[q] = r0 + r < rows ? rr[r0 + r] : 0.0F;
        }
    }
    hc_chain_prefetch<Rows, FromX, Cols>(F, fn, next_a, next_w, r0, rows, 0);
    hc_chain_stage<Rows, Cols, FromX>(next_a, next_w, as[0], ws[0], rrq);
    __syncthreads();
    for (unsigned k0 = 0; k0 < n; k0 += kHcK) {
        const unsigned stage = (k0 / kHcK) % 2;
        if (k0 + kHcK < n)
            hc_chain_prefetch<Rows, FromX, Cols>(F, fn, next_a, next_w, r0, rows, k0 + kHcK);
#pragma unroll 4
        for (unsigned k = 0; k < kHcK; ++k) {
            float a[4], b[J];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) a[i] = as[stage][tr * 4 + i][k];
#pragma unroll
            for (unsigned j = 0; j < J; ++j) b[j] = ws[stage][k][tc * J + j];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                for (unsigned j = 0; j < J; ++j) acc[i][j] = __builtin_fmaf(a[i], b[j], acc[i][j]);
        }
        if (k0 + kHcK < n) hc_chain_stage<Rows, Cols, FromX>(next_a, next_w, as[1 - stage], ws[1 - stage], rrq);
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned r = r0 + tr * 4 + i;
        if (r < rows)
#pragma unroll
            for (unsigned j = 0; j < J; ++j) z[static_cast<std::size_t>(r) * kHcMix + c0 + tc * J + j] = acc[i][j];
    }
}
bool hc_cols8() {  // STRIX_GLM_PF_HC_COLS = 8 (opt-in, same bits): three output groups per row tile; default 24
    static const bool on = [] {
        const char* e = std::getenv("STRIX_GLM_PF_HC_COLS");
        if (e == nullptr || *e == '\0' || std::string(e) == "24") return false;
        if (std::string(e) != "8") throw std::invalid_argument("STRIX_GLM_PF_HC_COLS must be 24 or 8");
        return true;
    }();
    return on;
}
// One launch of the v2 chain for a tile size and column split (the launch shape follows the template arguments).
template <unsigned Rows, bool FromX>
void hc_chain_launch(const float* F, const float* fn, float* z, unsigned rows, const float* rr, bool cols8, hipStream_t s) {
    if (cols8) hc_chain_rows_v2_kernel<Rows, FromX, 8><<<dim3((rows + Rows - 1) / Rows, kHcMix / 8), 2 * Rows, 0, s>>>(F, fn, z, rows, rr);
    else hc_chain_rows_v2_kernel<Rows, FromX><<<(rows + Rows - 1) / Rows, 2 * Rows, 0, s>>>(F, fn, z, rows, rr);
}
unsigned hc_rows_tile() {  // STRIX_GLM_PF_HC_TILE = 64 | 32 | 16 (same bits), default 32
    static const unsigned t = [] {
        const char* e = std::getenv("STRIX_GLM_PF_HC_TILE");
        if (e && std::string(e) == "64") return 64U;
        if (e && std::string(e) == "16") return 16U;
        return 32U;
    }();
    return t;
}

// One lane per row executes the shared serial R6; all lanes then share the collapse.
__global__ __launch_bounds__(256) void hc_mix_rows_v2_kernel(const float* X, const float* zg, const float* base,
                                                            const float* scale, float* mix, float* xc, unsigned rows) {
    __shared__ float m[kHcMixRows][kHcMix];
    const unsigned r0 = blockIdx.x * kHcMixRows, tid = threadIdx.x;
    if (tid < kHcMixRows && r0 + tid < rows) {
        const unsigned r = r0 + tid;
        hc_mix_r6(zg + static_cast<std::size_t>(r) * kHcMix, base, scale, m[tid]);
    }
    __syncthreads();
    for (unsigned v = tid; v < kHcMixRows * kHcMix; v += 256) {
        const unsigned rr = v / kHcMix, k = v % kHcMix;
        if (r0 + rr < rows) mix[static_cast<std::size_t>(r0 + rr) * kHcMix + k] = m[rr][k];
    }
    for (unsigned v = tid; v < kHcMixRows * kHidden; v += 256) {
        const unsigned rr = v / kHidden, d = v % kHidden, r = r0 + rr;
        if (r < rows) {
            const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
            xc[static_cast<std::size_t>(r) * kHidden + d] =
                ((m[rr][0] * Xr[d] + m[rr][1] * Xr[kHidden + d]) + m[rr][2] * Xr[2 * kHidden + d]) + m[rr][3] * Xr[3 * kHidden + d];
        }
    }
}

unsigned hc_version() {  // default v2 (measurement: PF-K 30/0 + synthetic 6-layer engine gate PASS, 2.72 -> 1.75 us/row/site); STRIX_GLM_PF_HC=1: v1
    static const unsigned v = [] {
        const char* e = std::getenv("STRIX_GLM_PF_HC");
        return e && std::string(e) == "1" ? 1U : 2U;
    }();
    return v;
}

bool mla_v1() {
    static const bool v1 = [] {
        const char* e = std::getenv("STRIX_GLM_PF_MLA");
        return e && std::string(e) == "1";
    }();
    return v1;
}
unsigned idx_version() {  // default v3 (bitwise == v2, -35..-44 % at t0 4096..28672); STRIX_GLM_PF_IDX=2 / 1: v2 / v1 (same keys)
    static const unsigned v = [] {
        const char* e = std::getenv("STRIX_GLM_PF_IDX");
        if (e && std::string(e) == "1") return 1U;
        if (e && std::string(e) == "2") return 2U;
        return 3U;
    }();
    return v;
}
unsigned select_version() {  // default v2 (same lists as v1, -60..-85 % at t0 4096..28672); STRIX_GLM_PF_SELECT=1: v1
    static const unsigned v = [] {
        const char* e = std::getenv("STRIX_GLM_PF_SELECT");
        return e && std::string(e) == "1" ? 1U : 2U;
    }();
    return v;
}
unsigned obar_version() {  // default 7 (heads 32; heads 64 runs v2); STRIX_GLM_PF_OBAR=5 / 2 / 4 / 6: v5 / v2 / v4 / v6 (all same bits)
    // measurement: v7 9.0-10.2 vs v5 16.6-17.3 us/row per DSA layer at t0 4096..28672 (6.6 vs 11.3 at t0 0); v5 18.1 vs v2
    // 53.9 at the 32K chunk; v4 5 % slower than v2; v6 2.4x slower than v5
    static const unsigned v = [] {
        const char* e = std::getenv("STRIX_GLM_PF_OBAR");
        if (e && std::string(e) == "6") return 6U;
        if (e && std::string(e) == "5") return 5U;
        if (e && std::string(e) == "4") return 4U;
        if (e && std::string(e) == "2") return 2U;
        return 7U;
    }();
    return v;
}
unsigned mla_gemm_version() {  // default v2 absorb / out GEMM (same bits, -20..-25 % per call); STRIX_GLM_PF_MLA_GEMM=1: v1
    static const unsigned v = [] {
        const char* e = std::getenv("STRIX_GLM_PF_MLA_GEMM");
        return e && std::string(e) == "1" ? 1U : 2U;
    }();
    return v;
}
unsigned scores_version() {  // default v7 (v6 in 256-row groups, bitwise == v6 / v5, -1..-28 % vs v6); STRIX_GLM_PF_SCORES=6 / 5 / 4 / 3 / 2
    static const unsigned v = [] {
        const char* e = std::getenv("STRIX_GLM_PF_SCORES");
        if (e && std::string(e) == "6") return 6U;
        if (e && std::string(e) == "5") return 5U;
        if (e && std::string(e) == "4") return 4U;
        if (e && std::string(e) == "3") return 3U;
        if (e && std::string(e) == "2") return 2U;
        return 7U;
    }();
    return v;
}

}  // namespace

namespace {
hipEvent_t* g_kda_prof = nullptr;  // 5 events: before prep, conv commit, scan, out, after
void kda_mark(unsigned i, hipStream_t s) {
    if (g_kda_prof) ok(hipEventRecord(g_kda_prof[i], s), "glm pf kda profile event");
}
}  // namespace
void set_kda_profile(hipEvent_t* ev5) { g_kda_prof = ev5; }
namespace {
unsigned g_kda_scan_v = 0;  // explicit selector (gates / bench); 0 = kda_scan_version()
unsigned kda_scan_version() {  // default v5 (v3 + fill off the serial path, same bits, -3.7 % vs v3); STRIX_GLM_PF_KDA_SCAN=3 / 1: v3 / v1
    static const unsigned v = [] {
        const char* e = std::getenv("STRIX_GLM_PF_KDA_SCAN");
        if (e && std::string(e) == "3") return 3U;
        return e && std::string(e) == "1" ? 1U : 5U;
    }();
    return v;
}
}  // namespace
void set_kda_scan_version(unsigned v) { g_kda_scan_v = v; }

void kda_prefill(const KdaDev& k, const float* qkv, const float* f, const float* gate, const float* b, float* S,
                 float* conv_state, float* scratch, float* beta, float* o, float* y, unsigned* flag, unsigned C,
                 hipStream_t s) {
    if (C == 0) return;
    const unsigned channels = 3 * k.heads * kKdaDim;
    kda_mark(0, s);
    kda_prep_rows_kernel<<<dim3(k.heads, C), 128, 0, s>>>(k, qkv, f, b, conv_state, scratch, beta);
    ok(hipGetLastError(), "glm pf kda_prep_rows");
    kda_mark(1, s);
    kda_conv_commit_kernel<<<(channels + 255) / 256, 256, 0, s>>>(qkv, conv_state, channels, C);
    ok(hipGetLastError(), "glm pf kda_conv_commit");
    kda_mark(2, s);
    const unsigned sv = g_kda_scan_v ? g_kda_scan_v : kda_scan_version();
    if (sv == 1) kda_scan_kernel<<<dim3(k.heads, kKdaDim / kScanV), kScanLanes * kScanV, 0, s>>>(k.heads, S, scratch, beta, o, flag, C);
    else if (sv == 3) kda_scan_v3_kernel<<<dim3(k.heads, kKdaDim / kScanV), kScanLanes * kScanV, 0, s>>>(k.heads, S, scratch, beta, o, flag, C);
    else kda_scan_v5_kernel<<<dim3(k.heads, kKdaDim / kScanV), kScanLanes * kScanV, 0, s>>>(k.heads, S, scratch, beta, o, flag, C);
    ok(hipGetLastError(), "glm pf kda_scan");
    kda_mark(3, s);
    kda_out_rows_kernel<<<dim3(k.heads, C), 128, 0, s>>>(k, o, gate, y);
    ok(hipGetLastError(), "glm pf kda_out_rows");
    kda_mark(4, s);
}

void dsa_append_rows(const DsaSlot& slot, unsigned t0, const float* latent, const float* key, const float* gate,
                     const IndexerDev& w, unsigned* flag, unsigned C, hipStream_t s) {
    if (C == 0) return;
    dsa_store_rows_kernel<<<C, 128, 0, s>>>(slot, t0, latent, w, key, gate, flag);
    ok(hipGetLastError(), "glm pf dsa_store_rows");
    const unsigned p_lo = t0 / kKpool, p_hi = (t0 + C) / kKpool;
    if (p_hi > p_lo) {
        dsa_pool_kernel<<<p_hi - p_lo, 128, 0, s>>>(slot, p_lo, w, flag);
        ok(hipGetLastError(), "glm pf dsa_pool");
    }
}

namespace {
hipEvent_t* g_idx_prof = nullptr;  // 3 events: before scores, before select, after
void idx_mark(unsigned i, hipStream_t s) {
    if (g_idx_prof) ok(hipEventRecord(g_idx_prof[i], s), "glm pf indexer profile event");
}
}  // namespace
void set_idx_profile(hipEvent_t* ev3) { g_idx_prof = ev3; }

void indexer_select_rows(const DsaSlot& slot, unsigned t0, const float* qI, float* wh, std::uint32_t* keys,
                         unsigned kpool_cap, std::uint32_t* tokens, std::uint32_t* n, unsigned C, hipStream_t s) {
    indexer_select_rows_versions(slot, t0, qI, wh, keys, kpool_cap, tokens, n, C, s, idx_version(), select_version());
}

void indexer_select_rows_versions(const DsaSlot& slot, unsigned t0, const float* qI, float* wh, std::uint32_t* keys,
                                  unsigned kpool_cap, std::uint32_t* tokens, std::uint32_t* n, unsigned C, hipStream_t s,
                                  unsigned score_v, unsigned select_v) {
    if ((score_v != 1 && score_v != 2 && score_v != 3) || (select_v != 1 && select_v != 2))
        throw std::runtime_error("glm pf indexer_select_rows_versions: unsupported versions");
    if (C == 0) return;
    idx_mark(0, s);
    const unsigned Pmax = (t0 + C) / kKpool;
    if (Pmax > kpool_cap) throw std::runtime_error("glm pf indexer_select_rows: key workspace too small");
    if (Pmax > kIdxPools) {
        scale_kernel<<<(C * kIdxHeads + 255) / 256, 256, 0, s>>>(wh, C * kIdxHeads, kInvSqrt32);
        ok(hipGetLastError(), "glm pf scale");
        if (score_v == 1)
            idx_score_rows_kernel<<<dim3((Pmax + 127) / 128, (C + kScoreRows - 1) / kScoreRows), 128, 0, s>>>(
                slot.pooled, t0, C, qI, wh, keys, kpool_cap);
        else if (score_v == 3)
            idx_score_rows_v3_kernel<<<dim3((Pmax + kIs3Pools - 1) / kIs3Pools, (C + kIs3Rows - 1) / kIs3Rows), 32 * kIs3Rows, 0, s>>>(
                slot.pooled, t0, C, qI, wh, keys, kpool_cap);
        else
            idx_score_rows_v2_kernel<<<dim3((Pmax + kIs2Pools - 1) / kIs2Pools, (C + kIs2Rows - 1) / kIs2Rows), 256, 0, s>>>(
                slot.pooled, t0, C, qI, wh, keys, kpool_cap);
        ok(hipGetLastError(), "glm pf idx_score_rows");
    }
    idx_mark(1, s);
    if (select_v == 2) idx_select_rows_v2_kernel<<<C, kSel2Threads, 0, s>>>(keys, kpool_cap, t0, tokens, n);
    else idx_select_rows_kernel<<<C, 1024, 0, s>>>(keys, kpool_cap, t0, tokens, n);
    ok(hipGetLastError(), "glm pf idx_select_rows");
    idx_mark(2, s);
}

void dense_rows(const float* x, const void* w, dev::WType wt, unsigned in, unsigned o0, unsigned n, float* y, unsigned rows,
                hipStream_t s) {
    if (rows == 0 || n == 0) return;
    if (in == 0 || in % kGK) throw std::runtime_error("glm pf dense_rows: in must be a positive multiple of 32");
    dense_rows_kernel<<<dim3((rows + kGT - 1) / kGT, (n + kGT - 1) / kGT), 256, 0, s>>>(x, w, wt, in, o0, n, y, rows);
    ok(hipGetLastError(), "glm pf dense_rows");
}

void hc_pre_rows_v1(const float* X, const float* fn, const float* base, const float* scale, float* F, float* z, float* mix, float* xc,
                 unsigned rows, hipStream_t s) {
    if (rows == 0) return;
    hc_norm_rows_kernel<<<rows, 256, 0, s>>>(X, F);
    ok(hipGetLastError(), "glm pf hc_norm_rows");
    dense_rows(F, fn, dev::WType::F32, kHc * kHidden, 0, kHcMix, z, rows, s);
    hc_mix_rows_kernel<<<rows, 256, 0, s>>>(X, z, base, scale, mix, xc);
    ok(hipGetLastError(), "glm pf hc_mix_rows");
}

void hc_pre_rows_v2(const float* X, const float* fn, const float* base, const float* scale, float* F, float* z, float* mix, float* xc,
                    unsigned rows, hipStream_t s) {
    if (rows == 0) return;
    hc_norm_rows_kernel<<<rows, 256, 0, s>>>(X, F);
    ok(hipGetLastError(), "glm pf hc_norm_rows v2");
    const unsigned t = hc_rows_tile();
    if (t == 64) hc_chain_launch<64, false>(F, fn, z, rows, nullptr, hc_cols8(), s);
    else if (t == 16) hc_chain_launch<16, false>(F, fn, z, rows, nullptr, hc_cols8(), s);
    else hc_chain_launch<32, false>(F, fn, z, rows, nullptr, hc_cols8(), s);
    ok(hipGetLastError(), "glm pf hc_chain_rows v2");
    hc_mix_rows_v2_kernel<<<(rows + kHcMixRows - 1) / kHcMixRows, 256, 0, s>>>(X, z, base, scale, mix, xc, rows);
    ok(hipGetLastError(), "glm pf hc_mix_rows v2");
}

void hc_pre_rows(const float* X, const float* fn, const float* base, const float* scale, float* F, float* z, float* mix, float* xc,
                 unsigned rows, hipStream_t s) {
    if (hc_version() == 2) hc_pre_rows_v2(X, fn, base, scale, F, z, mix, xc, rows, s);
    else hc_pre_rows_v1(X, fn, base, scale, F, z, mix, xc, rows, s);
}

namespace {
unsigned hc_fuse_mode() {
    static const unsigned m = [] {
        const char* e = std::getenv("STRIX_GLM_PF_HC_FUSE");
        if (e == nullptr || *e == '\0') return 1U;  // default on: exact, pair ABBA -1.0 % 1K / -1.4 % 8K; 0 restores the unfused sites
        const std::string v(e);
        if (v != "0" && v != "1" && v != "2") throw std::invalid_argument("STRIX_GLM_PF_HC_FUSE must be 0, 1 or 2");
        return static_cast<unsigned>(v[0] - '0');
    }();
    return m;
}
}  // namespace

bool hc_expand_norm_enabled() { return hc_fuse_mode() != 0 && hc_version() == 2; }
bool hc_expand_norm_split() { return hc_fuse_mode() == 2; }

void hc_norm_factor_rows(const float* X, float* rr, unsigned rows, hipStream_t s) {
    if (rows == 0) return;
    hc_norm_factor_rows_kernel<<<rows, 256, 0, s>>>(X, rr);
    ok(hipGetLastError(), "glm pf hc_norm_factor_rows");
}

void hc_expand_norm_rows(const float* X, const float* mix, const float* y, float* Xo, float* rr, unsigned* flag, unsigned rows,
                         hipStream_t s) {
    if (rows == 0) return;
    hc_expand_norm_rows_kernel<<<rows, 256, 0, s>>>(X, mix, y, Xo, rr, flag);
    ok(hipGetLastError(), "glm pf hc_expand_norm_rows");
}

void hc_pre_rows_normed(const float* X, const float* fn, const float* base, const float* scale, const float* rr, float* z, float* mix,
                        float* xc, unsigned rows, hipStream_t s) {
    if (rows == 0) return;
    if (hc_version() != 2) throw std::logic_error("glm pf hc_pre_rows_normed: v2 sites only");
    const unsigned t = hc_rows_tile();
    if (t == 64) hc_chain_launch<64, true>(X, fn, z, rows, rr, hc_cols8(), s);
    else if (t == 16) hc_chain_launch<16, true>(X, fn, z, rows, rr, hc_cols8(), s);
    else hc_chain_launch<32, true>(X, fn, z, rows, rr, hc_cols8(), s);
    ok(hipGetLastError(), "glm pf hc_chain_rows normed");
    hc_mix_rows_v2_kernel<<<(rows + kHcMixRows - 1) / kHcMixRows, 256, 0, s>>>(X, z, base, scale, mix, xc, rows);
    ok(hipGetLastError(), "glm pf hc_mix_rows normed");
}

namespace {
hipEvent_t* g_mla_prof = nullptr;  // 6 events: before absorb, scores, softmax, o-bar, out, after
void mla_mark(unsigned i, hipStream_t s) {
    if (g_mla_prof) ok(hipEventRecord(g_mla_prof[i], s), "glm pf mla profile event");
}
}  // namespace
void set_mla_profile(hipEvent_t* ev6) { g_mla_prof = ev6; }
bool mla_e_lane_major(unsigned heads) { return !mla_v1() && (obar_version() == 5 || obar_version() == 7) && heads == 32; }

void mla_attention_rows_versions(const DsaSlot& slot, const std::uint32_t* tokens, const std::uint32_t* n, const float* q,
                        const std::uint16_t* kv_b, unsigned heads, float* qt, float* e, float* ob, float* o, unsigned C,
                        hipStream_t s, unsigned scores_v, unsigned obar_v, unsigned gemm_v) {
    if (C == 0) return;
    if (heads != 32 && heads != 64) throw std::runtime_error("glm pf mla_attention_rows: heads must be 32 or 64");
    if (gemm_v != 1 && gemm_v != 2) throw std::runtime_error("glm pf mla_attention_rows_versions: unsupported gemm version");
    if ((scores_v != 1 && scores_v != 2 && scores_v != 3 && scores_v != 4 && scores_v != 5 && scores_v != 6 && scores_v != 7) ||
        (obar_v != 1 && obar_v != 2 && obar_v != 4 && obar_v != 5 && obar_v != 6 && obar_v != 7) ||
        ((scores_v == 1) != (obar_v == 1)))
        throw std::runtime_error("glm pf mla_attention_rows_versions: unsupported version pair");
    const bool v1 = scores_v == 1;  // explicit legacy pair
    const dim3 sg((kMaxTokens + 255) / 256, C);
    mla_mark(0, s);
    if (v1) {
        mla_absorb_rows_kernel<<<dim3(kKvLora / 256, heads, C), 256, 0, s>>>(q, kv_b, heads, qt);
        mla_mark(1, s);
        if (heads == 32) mla_scores_rows_kernel<32><<<sg, 256, 0, s>>>(slot, tokens, n, qt, e);
        else mla_scores_rows_kernel<64><<<sg, 256, 0, s>>>(slot, tokens, n, qt, e);
    } else {
        if (gemm_v == 2) mla_gemm_rows_v2_kernel<false><<<dim3((C + kG2R - 1) / kG2R, kKvLora / kG2C, heads), 256, 0, s>>>(q, kv_b, heads, C, qt);
        else mla_gemm_rows_kernel<false><<<dim3((C + kGT - 1) / kGT, kKvLora / kGT, heads), 256, 0, s>>>(q, kv_b, heads, C, qt);
        mla_mark(1, s);
        if (scores_v == 2) {
            if (heads == 32) mla_scores_v2_kernel<32><<<sg, 256, 0, s>>>(slot, tokens, n, qt, e);
            else mla_scores_v2_kernel<64><<<sg, 256, 0, s>>>(slot, tokens, n, qt, e);
        } else if (scores_v >= 5) {  // v6 / v7: heads 32 only (heads 64 runs v5)
            constexpr unsigned tile = 32 * kS5Waves * 2;  // entries per CTA at H = 32 (half that at H = 64)
            if (heads == 32 && scores_v == 7)
                mla_scores_v6_kernel<kS7Group><<<(C + kS7Group - 1) / kS7Group * kS7Group * kS6Tiles, 32 * kS5Waves, 0, s>>>(slot, tokens, n, qt, e, C);
            else if (heads == 32 && scores_v == 6)
                mla_scores_v6_kernel<0><<<dim3((kMaxTokens + tile - 1) / tile, C), 32 * kS5Waves, 0, s>>>(slot, tokens, n, qt, e, C);
            else if (heads == 32) mla_scores_v5_kernel<32><<<dim3((kMaxTokens + tile - 1) / tile, C), 32 * kS5Waves, 0, s>>>(slot, tokens, n, qt, e);
            else mla_scores_v5_kernel<64><<<dim3((kMaxTokens + tile / 2 - 1) / (tile / 2), C), 32 * kS5Waves, 0, s>>>(slot, tokens, n, qt, e);
        } else {
            const dim3 g3((kMaxTokens + kS3Tok - 1) / kS3Tok, C);
            if (scores_v == 4 && heads == 32) mla_scores_v4_kernel<32><<<g3, 256, 0, s>>>(slot, tokens, n, qt, e);
            else if (scores_v == 4) mla_scores_v4_kernel<64><<<g3, 256, 0, s>>>(slot, tokens, n, qt, e);
            else if (heads == 32) mla_scores_v3_kernel<32><<<g3, 256, 0, s>>>(slot, tokens, n, qt, e);
            else mla_scores_v3_kernel<64><<<g3, 256, 0, s>>>(slot, tokens, n, qt, e);
        }
    }
    ok(hipGetLastError(), "glm pf mla absorb/scores");
    mla_mark(2, s);
    const unsigned obv = v1 ? 1U : (obar_v == 5 || obar_v == 7) && heads == 64 ? 2U : obar_v;  // v5 / v7: heads 32 only
    if (obv == 5 || obv == 7) mla_softmax_rows_kernel<true><<<dim3(heads, C), 256, 0, s>>>(n, heads, e);
    else mla_softmax_rows_kernel<false><<<dim3(heads, C), 256, 0, s>>>(n, heads, e);
    ok(hipGetLastError(), "glm pf mla_softmax_rows");
    mla_mark(3, s);
    if (v1) {
        mla_obar_rows_kernel<<<dim3(kKvLora / kObLat, C), 256, 0, s>>>(slot, tokens, n, heads, e, ob);
        mla_mark(4, s);
        mla_out_rows_kernel<<<dim3(heads, C), 256, 0, s>>>(ob, kv_b, heads, o);
    } else {
        if (obv == 7) mla_obar_v7_kernel<<<dim3(kKvLora / kOb7Dims, C), 256, 0, s>>>(slot, tokens, n, e, ob);
        else if (obv == 6) mla_obar_v6_kernel<<<dim3(heads / kOb6Heads, kKvLora / kOb6Dims, C), 256, 0, s>>>(slot, tokens, n, heads, e, ob);
        else if (obv == 5) mla_obar_v5_kernel<<<dim3(kKvLora / kOb5Dims, C), 256, 0, s>>>(slot, tokens, n, e, ob);
        else if (obv == 4) mla_obar_v4_kernel<<<dim3(heads / kOb4Heads, kKvLora / kOb4Dims, C), 256, 0, s>>>(slot, tokens, n, heads, e, ob);
        else mla_obar_v2_kernel<<<dim3(kKvLora / kOb2Lat, C), 256, 0, s>>>(slot, tokens, n, heads, e, ob);
        mla_mark(4, s);
        if (gemm_v == 2) mla_gemm_rows_v2_kernel<true><<<dim3((C + kG2R - 1) / kG2R, kMlaDim / kG2C, heads), 256, 0, s>>>(ob, kv_b, heads, C, o);
        else mla_gemm_rows_kernel<true><<<dim3((C + kGT - 1) / kGT, kMlaDim / kGT, heads), 256, 0, s>>>(ob, kv_b, heads, C, o);
    }
    ok(hipGetLastError(), "glm pf mla obar/out");
    mla_mark(5, s);
}

void mla_attention_rows(const DsaSlot& slot, const std::uint32_t* tokens, const std::uint32_t* n, const float* q,
                        const std::uint16_t* kv_b, unsigned heads, float* qt, float* e, float* ob, float* o, unsigned C,
                        hipStream_t s) {
    mla_attention_rows_versions(slot, tokens, n, q, kv_b, heads, qt, e, ob, o, C, s,
                                mla_v1() ? 1U : scores_version(), mla_v1() ? 1U : obar_version(), mla_gemm_version());
}

}  // namespace strix::glm::pf
