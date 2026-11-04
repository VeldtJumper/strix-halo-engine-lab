#include "h3/vae/fast_decoder.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "h3/vae/vae_kernels.hpp"
#include "h3/vae/vae_plan.hpp"

namespace strix::h3::vae::fast {

namespace {
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("h3 vae fast decoder: ") + what + ": " + hipGetErrorString(e));
}
std::vector<std::pair<int, int>> spans(int n, int shift, int size) {
    std::vector<std::pair<int, int>> out;
    const int count = (n + shift + size - 1) / size;
    for (int i = 0; i < count; ++i) {
        const int lo = std::max(i * size - shift, 0), hi = std::min((i + 1) * size - shift, n);
        if (hi > lo) out.emplace_back(lo, hi);
    }
    return out;
}
float window_id(int i, int o, int m) {  // ((i + o) % m + 0.5) / m * 2 - 1 in FP32, then FP16
    float v = float((i + o) % m) + 0.5F;
    v = v / float(m);
    v = v * 2.0F;
    v = v - 1.0F;
    return rh(v);
}
}  // namespace

Geometry build_geometry(int T, int H, int W) {
    constexpr int tb = 5, ws = 16, S = kSuffix;
    Geometry g;
    g.T = T; g.H = H; g.W = W;
    g.nb = (T + tb - 1) / tb;
    const int HW = H * W;
    for (int b = 0; b <= g.nb; ++b) g.block_row0.push_back(std::min(b * tb, T) * HW + b * S);
    g.token_row.resize(std::size_t(T) * HW);
    for (int t = 0; t < T; ++t)
        for (int q = 0; q < HW; ++q) g.token_row[std::size_t(t) * HW + q] = g.block_row0[std::size_t(t / tb)] + (t % tb) * HW + q;
    const int shifts[2][2] = {{0, 0}, {ws / 2, 2}};
    for (int si = 0; si < 2; ++si) {
        const int s = shifts[si][0], st = shifts[si][1];
        Geometry::Shift& sh = g.shift[si];
        sh.ids.assign(std::size_t(g.rows()) * 3, 0.0F);
        for (int t = 0; t < T; ++t)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const std::size_t r = std::size_t(g.token_row[(std::size_t(t) * H + y) * W + x]);
                    sh.ids[r * 3 + 0] = window_id(t, st, tb);
                    sh.ids[r * 3 + 1] = window_id(y, s, ws);
                    sh.ids[r * 3 + 2] = window_id(x, s, ws);
                }
        for (const auto& [t0, t1] : spans(T, st, tb)) {
            std::vector<int> blocks = {t0 / tb, (t1 - 1) / tb};
            if (blocks[0] == blocks[1]) blocks.pop_back();
            for (const auto& [y0, y1] : spans(H, s, ws))
                for (const auto& [x0, x1] : spans(W, s, ws)) {
                    Problem p;
                    p.qoff = unsigned(sh.qidx.size());
                    p.koff = unsigned(sh.kidx.size());
                    for (int t = t0; t < t1; ++t)
                        for (int y = y0; y < y1; ++y)
                            for (int x = x0; x < x1; ++x) {
                                const unsigned r = unsigned(g.token_row[(std::size_t(t) * H + y) * W + x]);
                                sh.qidx.push_back(r);
                                sh.kidx.push_back(r);
                            }
                    p.nq = unsigned(sh.qidx.size()) - p.qoff;
                    for (int b : blocks)
                        for (int k = 0; k < S; ++k) sh.kidx.push_back(unsigned(g.block_row0[std::size_t(b) + 1] - S + k));
                    p.nk = unsigned(sh.kidx.size()) - p.koff;
                    sh.problems.push_back(p);
                }
        }
        for (int b = 0; b < g.nb; ++b) {  // global rows attend to their whole block (all rows, in row order)
            Problem p;
            p.qoff = unsigned(sh.qidx.size());
            p.koff = unsigned(sh.kidx.size());
            for (int k = 0; k < S; ++k) sh.qidx.push_back(unsigned(g.block_row0[std::size_t(b) + 1] - S + k));
            for (int r = g.block_row0[std::size_t(b)]; r < g.block_row0[std::size_t(b) + 1]; ++r) sh.kidx.push_back(unsigned(r));
            p.nq = S;
            p.nk = unsigned(sh.kidx.size()) - p.koff;
            sh.problems.push_back(p);
        }
    }
    return g;
}

void rope_from_ids(const std::vector<float>& ids, int rows, std::vector<float>& c, std::vector<float>& s) {
    float inv[8];
    for (int f = 0; f < 8; ++f) inv[f] = rh(1.0F / std::pow(100.0F, float(f) * 0.125F));
    const float two_pi = float(2.0 * 3.14159265358979323846);
    c.assign(std::size_t(rows) * kRotPairs, 1.0F);
    s.assign(std::size_t(rows) * kRotPairs, 0.0F);
    for (int r = 0; r < rows; ++r)
        for (int a = 0; a < 3; ++a)
            for (int f = 0; f < 8; ++f) {
                const float ang = (two_pi * ids[std::size_t(r) * 3 + a]) * inv[f];
                c[std::size_t(r) * kRotPairs + a * 8 + f] = rh(std::cos(ang));
                s[std::size_t(r) * kRotPairs + a * 8 + f] = rh(std::sin(ang));
            }
}

namespace {
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ _Float16 r16(float x) { return static_cast<_Float16>(x); }
__device__ __forceinline__ h16 ld16(const _Float16* p) {
    const u32x4 lo = *reinterpret_cast<const u32x4*>(p), hi = *reinterpret_cast<const u32x4*>(p + 8);
    return __builtin_bit_cast(h16, u32x8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w});
}
__device__ __forceinline__ void st16(_Float16* p, h16 v) {
    const u32x8 u = __builtin_bit_cast(u32x8, v);
    *reinterpret_cast<u32x4*>(p) = u32x4{u[0], u[1], u[2], u[3]};
    *reinterpret_cast<u32x4*>(p + 8) = u32x4{u[4], u[5], u[6], u[7]};
}
__device__ __forceinline__ _Float16 nan_to_num16(_Float16 v) {
    const unsigned short b = __builtin_bit_cast(unsigned short, v);
    if ((b & 0x7c00U) != 0x7c00U) return v;
    if (b & 0x03ffU) return r16(0.0F);
    return __builtin_bit_cast(_Float16, static_cast<unsigned short>((b & 0x8000U) | 0x7bffU));
}

// Gathered non-causal attention, head dim 64: the D64 kernel of the 36-layer decoder (S^T = K Q^T, online softmax in
// FP32, P in FP16, ascending key order of the problem's key list) with query and key rows taken from index lists.
// Block job = (problem, 128-query slice); grid.y = head. Q/K/V [32][Rp][64]; O [R][2048].
struct Job { unsigned problem, q0; };
constexpr unsigned kAQ = 128U, kAK = 64U, kAPitch = 72U;
__global__ __launch_bounds__(256) void gather_attention_kernel(const _Float16* __restrict__ Q, const _Float16* __restrict__ Kh,
    const _Float16* __restrict__ V, _Float16* __restrict__ O, unsigned Rp, const Problem* __restrict__ probs,
    const Job* __restrict__ jobs, const unsigned* __restrict__ qidx, const unsigned* __restrict__ kidx) {
    __shared__ __align__(16) _Float16 ks[kAK * kAPitch];
    __shared__ __align__(16) _Float16 vt[64U * kAPitch];
    const unsigned t = threadIdx.x, lane = t % 32U, wave = t / 32U, hd = blockIdx.y;
    const Job job = jobs[blockIdx.x];
    const Problem pr = probs[job.problem];
    const unsigned q0 = job.q0 + wave * 16U;
    const _Float16* qh = Q + std::size_t(hd) * Rp * 64U;
    const _Float16* kh = Kh + std::size_t(hd) * Rp * 64U;
    const _Float16* vh = V + std::size_t(hd) * Rp * 64U;
    const unsigned myq = qidx[pr.qoff + min(q0 + lane % 16U, pr.nq - 1U)];
    h16 qf[4];
#pragma unroll
    for (unsigned dt = 0; dt < 4U; ++dt) qf[dt] = ld16(qh + std::size_t(myq) * 64U + 16U * dt);
    f8 o[4];
#pragma unroll
    for (unsigned dt = 0; dt < 4U; ++dt) o[dt] = f8{};
    float mrow = -INFINITY, lrow = 0.0F;
    const float sl2 = 0.125F * 1.4426950408889634F;
    const bool upper = lane >= 16U;
    for (unsigned k0 = 0; k0 < pr.nk; k0 += kAK) {
        __syncthreads();
        {
            const unsigned kr = t / 4U, kc = 16U * (t % 4U);
            h16 kv = {}, vv = {};
            if (k0 + kr < pr.nk) {
                const std::size_t g = std::size_t(kidx[pr.koff + k0 + kr]) * 64U + kc;
                kv = ld16(kh + g);
                vv = ld16(vh + g);
            }
            st16(ks + kr * kAPitch + kc, kv);
#pragma unroll
            for (unsigned d = 0; d < 16U; ++d) vt[(kc + d) * kAPitch + kr] = vv[d];
        }
        __syncthreads();
        f8 s[4];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            s[j] = f8{};
#pragma unroll
            for (unsigned dt = 0; dt < 4U; ++dt)
                s[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ld16(ks + (16U * j + lane % 16U) * kAPitch + 16U * dt), qf[dt], s[j]);
        }
        float mx = mrow;
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned key = k0 + 16U * j + 2U * e + (upper ? 1U : 0U);
                s[j][e] = key < pr.nk ? s[j][e] * sl2 : -INFINITY;
                mx = fmaxf(mx, s[j][e]);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float alpha = exp2f(mrow - mx);
        float ps = 0.0F;
        h16 pf[4];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const float p = exp2f(s[j][e] - mx);
                s[j][e] = p;
                ps += p;
            }
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const float other = __shfl_xor(s[j][e], 16, 32);
                pf[j][2U * e] = r16(upper ? other : s[j][e]);
                pf[j][2U * e + 1U] = r16(upper ? s[j][e] : other);
            }
        }
        ps += __shfl_xor(ps, 16, 32);
        lrow = lrow * alpha + ps;
        mrow = mx;
#pragma unroll
        for (unsigned dt = 0; dt < 4U; ++dt) {
            o[dt] *= alpha;
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j)
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ld16(vt + (16U * dt + lane % 16U) * kAPitch + 16U * j), pf[j], o[dt]);
        }
    }
    const unsigned qi = q0 + lane % 16U;
    if (qi >= pr.nq) return;
    const unsigned row = qidx[pr.qoff + qi];
    const float inv = 1.0F / lrow;
#pragma unroll
    for (unsigned dt = 0; dt < 4U; ++dt)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned d = 16U * dt + 2U * e + (upper ? 1U : 0U);
            O[std::size_t(row) * 2048U + hd * 64U + d] = nan_to_num16(r16(o[dt][e] * inv));
        }
}

// x_embedder on a contiguous row view (one rounding): x[row] = fp16(z W^T + b), z channel-major [24][P]
__global__ __launch_bounds__(256) void embed1_kernel(const _Float16* __restrict__ zin, const _Float16* __restrict__ w,
                                                     const _Float16* __restrict__ b, const unsigned* __restrict__ rows,
                                                     _Float16* __restrict__ x, unsigned P) {
    const std::size_t i = std::size_t(blockIdx.x) * 256U + threadIdx.x;
    if (i >= std::size_t(P) * 2048U) return;
    const unsigned p = unsigned(i / 2048U), n = unsigned(i % 2048U);
    float acc[8] = {};
#pragma unroll
    for (unsigned c = 0; c < 24U; ++c) acc[c % 8U] += static_cast<float>(zin[std::size_t(c) * P + p]) * static_cast<float>(w[n * 24U + c]);
    const float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    x[std::size_t(rows[p]) * 2048U + n] = r16(s + static_cast<float>(b[n]));
}

struct DevLinear { _Float16 *w = nullptr, *b = nullptr; };
// per-call device scratch, released on return or exception
struct Scratch {
    std::vector<void*> p;
    Scratch() = default;
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    ~Scratch() { for (void* q : p) (void)hipFree(q); }
    template <class Tp> Tp* get(std::size_t n) {
        void* q = nullptr;
        ck(hipMalloc(&q, std::max<std::size_t>(n * sizeof(Tp), 16)), "hipMalloc");
        p.push_back(q);
        return static_cast<Tp*>(q);
    }
};
const float kLatentsMean[24] = {0.858090341091156F, -0.9606591463088989F, 1.0661640167236328F, -0.5090325474739075F,
    -0.2727581858634949F, -1.3675414323806763F, -0.2553254961967468F, -0.26907554268836975F, -0.5376840829849243F,
    -0.0464097298681736F, 0.6657370328903198F, 0.19690127670764923F, -0.5460608005523682F, -0.4035342037677765F,
    -0.23683024942874908F, 0.25928452610969543F, -0.30133944749832153F, 0.211341992020607F, -1.1206848621368408F,
    0.3581933379173279F, -0.04225143790245056F, 0.2604829967021942F, 0.22864092886447906F, 0.7056031823158264F};
const float kLatentsStd[24] = {1.2223774194717407F, 1.2767263650894165F, 1.68317747116088865F, 1.7549455165863037F,
    1.5636216402053833F, 2.194143533706665F, 0.96531379222869875F, 1.05698859691619875F, 0.841948926448822F,
    0.7729952931404114F, 1.8955937623977661F, 0.946841835975647F, 0.7996809482574463F, 0.44988900423049925F,
    0.7197399735450745F, 0.69362932443618775F, 2.961095094680786F, 2.7694199085235595F, 3.0496184825897215F,
    2.1088054180145265F, 3.276226282119751F, 3.1627357006073F, 2.28168129920959475F, 2.6127843856811525F};
}  // namespace

struct GpuDecoder::Dev {
    struct Layer { _Float16 *norm1, *norm2, *scale1, *scale2; DevLinear qkv, out, w1, w2; };
    std::vector<Layer> layers;
    DevLinear x_embed, proj;
    _Float16 *registers = nullptr, *ln_w = nullptr, *ln_b = nullptr;
    hipStream_t st = nullptr;
    std::vector<void*> owned;
    ~Dev() {
        (void)hipDeviceSynchronize();
        for (void* p : owned) (void)hipFree(p);
        if (st) (void)hipStreamDestroy(st);
    }
    void* alloc_bytes(std::size_t n) {
        void* p = nullptr;
        ck(hipMalloc(&p, std::max<std::size_t>(n, 16)), "hipMalloc");
        owned.push_back(p);
        return p;
    }
    _Float16* upload(const H16& v) {
        auto* p = static_cast<_Float16*>(alloc_bytes(v.size() * 2));
        ck(hipMemcpy(p, v.data(), v.size() * 2, hipMemcpyHostToDevice), "upload");
        return p;
    }
    DevLinear upload(const LinearH& l) { return {upload(l.w), upload(l.b)}; }
};

GpuDecoder::GpuDecoder(const std::string& weights) {
    SafeTensors st(weights);
    head_ = load_head(st, "");
    head_.latents_mean.resize(24);
    head_.latents_std.resize(24);
    for (int c = 0; c < 24; ++c) { head_.latents_mean[std::size_t(c)] = f2h(kLatentsMean[c]); head_.latents_std[std::size_t(c)] = f2h(kLatentsStd[c]); }
    auto dev = std::make_unique<Dev>();
    Dev& d = *dev;
    ck(hipStreamCreate(&d.st), "stream");
    for (int l = 0; l < 24; ++l) {
        const LayerH w = load_layer(st, l, "blocks.");
        LinearH w1 = w.w1, p = w.w1;  // fused gated SiLU order
        for (unsigned r = 0; r < unsigned(w1.out); ++r) {
            const unsigned src = gpu::swiglu_source_row(r, unsigned(w1.out));
            std::memcpy(p.w.data() + std::size_t(r) * w1.in, w1.w.data() + std::size_t(src) * w1.in, std::size_t(w1.in) * 2);
            p.b[r] = w1.b[src];
        }
        d.layers.push_back({d.upload(w.norm1), d.upload(w.norm2), d.upload(w.scale1), d.upload(w.scale2), d.upload(w.qkv), d.upload(w.out),
                            d.upload(p), d.upload(w.w2)});
    }
    d.x_embed = d.upload(head_.x_embed);
    d.proj = d.upload(head_.proj);
    d.registers = d.upload(head_.registers);
    d.ln_w = d.upload(head_.norm_out_w);
    d.ln_b = d.upload(head_.norm_out_b);
    d_ = dev.release();
}

GpuDecoder::~GpuDecoder() { delete d_; }

std::vector<float> GpuDecoder::decode(const H16& z_norm, int T, int H, int W, double* seconds) {
    if (T < 1 || H < 1 || W < 1) throw std::invalid_argument("h3 vae fast decoder: T, H, W must be >= 1");
    if (z_norm.size() != std::size_t(24) * std::size_t(T) * std::size_t(H) * std::size_t(W))
        throw std::invalid_argument("h3 vae fast decoder: latent length != 24 x T x H x W");
    const auto t0 = std::chrono::steady_clock::now();
    Dev& d = *d_;
    hipStream_t st = d.st;
    const Geometry g = build_geometry(T, H, W);
    const unsigned R = unsigned(g.rows()), Rp = (R + 63U) / 64U * 64U, HW = unsigned(H * W);
    // denormalize (FP16, two roundings) and post_quant_conv on the host, channel-major per block
    H16 zd(z_norm.size());
    for (int c = 0; c < 24; ++c) {
        const float sd = h2f(head_.latents_std[std::size_t(c)]), mu = h2f(head_.latents_mean[std::size_t(c)]);
        for (std::size_t q = 0; q < std::size_t(T) * HW; ++q) {
            const std::size_t i = std::size_t(c) * T * HW + q;
            zd[i] = f2h(rh(h2f(z_norm[i]) * sd) + mu);
        }
    }
    const H16 zin = post_quant(head_, zd, int(T * HW));  // [24][T * HW], tokens in (t, y, x) order
    std::vector<unsigned> trow(g.token_row.begin(), g.token_row.end());
    // per-call device scratch (released on return or exception)
    Scratch sc;
    _Float16 *x = sc.get<_Float16>(std::size_t(R) * kDim), *a = sc.get<_Float16>(std::size_t(R) * kDim);
    _Float16 *qkv = sc.get<_Float16>(std::size_t(R) * kQkv), *o = sc.get<_Float16>(std::size_t(R) * kDim);
    _Float16 *gb = sc.get<_Float16>(std::size_t(R) * kFf);
    _Float16 *q = sc.get<_Float16>(std::size_t(32) * Rp * 64), *k = sc.get<_Float16>(std::size_t(32) * Rp * 64), *v = sc.get<_Float16>(std::size_t(32) * Rp * 64);
    _Float16* dz = sc.get<_Float16>(zin.size());
    unsigned* drow = sc.get<unsigned>(trow.size());
    ck(hipMemcpyAsync(dz, zin.data(), zin.size() * 2, hipMemcpyHostToDevice, st), "z");
    ck(hipMemcpyAsync(drow, trow.data(), trow.size() * 4, hipMemcpyHostToDevice, st), "rows");
    // global rows: 4 registers + a zero row per block
    H16 glob(std::size_t(kSuffix) * kDim, 0);
    std::memcpy(glob.data(), head_.registers.data(), std::size_t(kRegisters) * kDim * 2);
    for (int b = 0; b < g.nb; ++b)
        ck(hipMemcpyAsync(x + std::size_t(g.block_row0[std::size_t(b) + 1] - kSuffix) * kDim, glob.data(), glob.size() * 2, hipMemcpyHostToDevice, st), "glob");
    embed1_kernel<<<unsigned((std::size_t(T) * HW * 2048U + 255U) / 256U), 256, 0, st>>>(dz, d.x_embed.w, d.x_embed.b, drow, x, unsigned(T) * HW);
    ck(hipGetLastError(), "embed");
    // per-shift device tables: rope (FP16 c, s) and the attention problems / jobs / index lists
    struct ShiftDev { _Float16 *rc, *rs; Problem* probs; Job* jobs; unsigned *qidx, *kidx; unsigned njobs; } sd[2];
    for (int si = 0; si < 2; ++si) {
        const auto& sh = g.shift[si];
        std::vector<float> c, s;
        rope_from_ids(sh.ids, int(R), c, s);
        H16 c16(c.size()), s16(s.size());
        for (std::size_t i = 0; i < c.size(); ++i) { c16[i] = f2h(c[i]); s16[i] = f2h(s[i]); }
        std::vector<Job> jobs;
        for (unsigned p = 0; p < sh.problems.size(); ++p)
            for (unsigned q0 = 0; q0 < sh.problems[p].nq; q0 += kAQ) jobs.push_back({p, q0});
        sd[si].rc = sc.get<_Float16>(c16.size()); sd[si].rs = sc.get<_Float16>(s16.size());
        sd[si].probs = sc.get<Problem>(sh.problems.size()); sd[si].jobs = sc.get<Job>(jobs.size());
        sd[si].qidx = sc.get<unsigned>(sh.qidx.size()); sd[si].kidx = sc.get<unsigned>(sh.kidx.size());
        sd[si].njobs = unsigned(jobs.size());
        ck(hipMemcpyAsync(sd[si].rc, c16.data(), c16.size() * 2, hipMemcpyHostToDevice, st), "rope");
        ck(hipMemcpyAsync(sd[si].rs, s16.data(), s16.size() * 2, hipMemcpyHostToDevice, st), "rope");
        ck(hipMemcpyAsync(sd[si].probs, sh.problems.data(), sh.problems.size() * sizeof(Problem), hipMemcpyHostToDevice, st), "problems");
        ck(hipMemcpyAsync(sd[si].jobs, jobs.data(), jobs.size() * sizeof(Job), hipMemcpyHostToDevice, st), "jobs");
        ck(hipMemcpyAsync(sd[si].qidx, sh.qidx.data(), sh.qidx.size() * 4, hipMemcpyHostToDevice, st), "qidx");
        ck(hipMemcpyAsync(sd[si].kidx, sh.kidx.data(), sh.kidx.size() * 4, hipMemcpyHostToDevice, st), "kidx");
        ck(hipStreamSynchronize(st), "sync");  // host vectors go out of scope
    }
    for (int li = 0; li < 24; ++li) {
        const auto& w = d.layers[std::size_t(li)];
        const ShiftDev& s = sd[li % 2];
        gpu::rms_norm(x, w.norm1, a, R, st);
        gpu::gemm(a, kDim, w.qkv.w, w.qkv.b, qkv, kQkv, R, kQkv, kDim, st);
        gpu::rms_rope(qkv, s.rc, s.rs, q, k, v, R, Rp, st);
        gather_attention_kernel<<<dim3(s.njobs, 32U), 256, 0, st>>>(q, k, v, o, Rp, s.probs, s.jobs, s.qidx, s.kidx);
        ck(hipGetLastError(), "gather attention");
        gpu::gemm_residual(o, kDim, w.out.w, w.out.b, w.scale1, x, kDim, R, kDim, kDim, st);
        gpu::rms_norm(x, w.norm2, a, R, st);
        gpu::gemm_swiglu(a, kDim, w.w1.w, w.w1.b, gb, kFf, R, kFf2, kDim, st);
        gpu::gemm_residual(gb, kFf, w.w2.w, w.w2.b, w.scale2, x, kDim, R, kDim, kFf, st);
    }
    gpu::layer_norm(x, d.ln_w, d.ln_b, a, R, st);
    _Float16* pr = qkv;  // reuse: R x 3072 <= R x 6144
    gpu::gemm(a, kDim, d.proj.w, d.proj.b, pr, kProj, R, kProj, kDim, st);
    H16 rows(std::size_t(R) * kProj);
    ck(hipMemcpyAsync(rows.data(), pr, rows.size() * 2, hipMemcpyDeviceToHost, st), "proj");
    ck(hipStreamSynchronize(st), "sync");
    // per block: unpatchify its tokens, drop the first 3 frames, finalize
    const int F = frames(T), Hp = 16 * H, Wp = 16 * W;
    const float mean16[3] = {rh(0.485F), rh(0.456F), rh(0.406F)}, std16[3] = {rh(0.229F), rh(0.224F), rh(0.225F)};
    std::vector<float> out(std::size_t(3) * F * Hp * Wp);
    int f0 = 0;
    for (int b = 0; b < g.nb; ++b) {
        const int nt = std::min(5, T - 5 * b);
        const H16 br(rows.begin() + std::ptrdiff_t(std::size_t(g.block_row0[std::size_t(b)]) * kProj),
                     rows.begin() + std::ptrdiff_t(std::size_t(g.block_row0[std::size_t(b)] + nt * int(HW)) * kProj));
        const H16 px = unpatchify(br, nt, H, W);  // [3][4 nt][Hp][Wp]
        const int keep = 4 * nt - 3;
        for (int c = 0; c < 3; ++c)
            for (int f = 0; f < keep; ++f)
                for (std::size_t p = 0; p < std::size_t(Hp) * Wp; ++p) {
                    float val = h2f(px[(std::size_t(c) * 4 * nt + 3 + f) * Hp * Wp + p]) * std16[c];
                    val = val + mean16[c];
                    val = val < 0.0F ? 0.0F : (val > 1.0F ? 1.0F : val);
                    out[(std::size_t(c) * F + f0 + f) * Hp * Wp + p] = val;
                }
        f0 += keep;
    }
    if (seconds) *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace strix::h3::vae::fast
