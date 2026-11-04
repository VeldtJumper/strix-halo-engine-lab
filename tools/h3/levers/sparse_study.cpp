// Sparse-attention quality study (identity h3-attn-sparse-thr-v1, study only):
// a dense 1-node sampling run (block package DeviceForward, gemm + flash attention) at a mid shape; in every DiT
// self-attention call the REAL post-K4 Q / K / V^T also go through the threshold router + the sparse K1 variant for
// tau in {0.5, 0.75, 1.0} (lever cut 128 floor(S / 256), two key blocks, canonical merge) and the sparse output is
// compared with the dense output of the same call (the forward itself continues dense, so every call sees the
// dense trajectory: per-call error, not chained drift). Logged per (step, block, tau): rel L2 over all rows and over
// the video rows, selected-tile density, union inflation (|A u B| / (|A| + |B|)), sparse vs dense partial time.
//   usage: strix-h3-levers-sparse-study --ckpt <dit> --te-hidden <te_hidden_f32.bin> [--width 896] [--height 512]
//          [--frames 73] [--steps 8] [--seed 42] [--detail-steps 0,3,7]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/attn/attn.hpp"
#include "h3/block/device_forward.hpp"
#include "h3/block/layout.hpp"
#include "h3/block/safetensors.hpp"
#include "h3/block/sampler.hpp"
#include "h3/block/torch_rng.hpp"
#include "h3/block/weights.hpp"
#include "h3/levers/sparse/attn_sparse.hpp"
#include "h3/levers/sparse/router.hpp"

namespace blk = strix::h3::block;
namespace sp = strix::h3::levers::sparse;
namespace attn = strix::h3::attn;

namespace {

void hc(hipError_t e, const char* w) {
    if (e != hipSuccess) throw std::runtime_error(std::string("hip ") + w + ": " + hipGetErrorString(e));
}
struct Dev {
    void* p = nullptr;
    std::size_t bytes = 0;
    void need(std::size_t b) {
        if (b <= bytes) return;
        if (p) hc(hipFree(p), "free");
        hc(hipMalloc(&p, b), "malloc");
        bytes = b;
    }
    template <class T> T* as() const { return static_cast<T*>(p); }
    ~Dev() {
        if (p) static_cast<void>(hipFree(p));
    }
};

// sum (a - b)^2 and sum b^2 over rows [r0, r1) of [rows][7168] bf16 (fp64 atomics: telemetry, not bitwise)
__global__ void rel_l2_kernel(const std::uint16_t* a, const std::uint16_t* b, std::uint64_t e0, std::uint64_t e1, double* acc) {
    __shared__ double s0[256], s1[256];
    double x = 0.0, y = 0.0;
    for (std::uint64_t i = e0 + static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x; i < e1; i += static_cast<std::uint64_t>(gridDim.x) * 256U) {
        const float fa = __uint_as_float(static_cast<unsigned>(a[i]) << 16U), fb = __uint_as_float(static_cast<unsigned>(b[i]) << 16U);
        const double d = static_cast<double>(fa) - static_cast<double>(fb);
        x += d * d;
        y += static_cast<double>(fb) * static_cast<double>(fb);
    }
    s0[threadIdx.x] = x;
    s1[threadIdx.x] = y;
    __syncthreads();
    for (unsigned s = 128; s > 0; s >>= 1U) {
        if (threadIdx.x < s) {
            s0[threadIdx.x] += s0[threadIdx.x + s];
            s1[threadIdx.x] += s1[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&acc[0], s0[0]);
        atomicAdd(&acc[1], s1[0]);
    }
}

struct CallStat {
    std::uint32_t step = 0, block = 0;
    float tau = 0;
    double rel_all = 0, rel_video = 0, density = 0, inflation = 0, ms_sparse = 0, ms_dense = 0;
};

class StudyAttention final : public blk::DeviceAttentionOp {
public:
    std::vector<float> taus{0.5F, 0.75F, 1.0F};
    std::uint32_t P = 0;          // prefix rows (before the target video)
    bool study = false;           // DiT calls of this forward are studied
    std::uint32_t step = 0, block = 0;
    std::vector<CallStat> stats;
    blk::FlashDeviceAttention dense;

    void run(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* vt, std::uint64_t vt_pitch, std::uint32_t rows, std::uint32_t keys,
        bool split, std::uint16_t* out, void* stream) override {
        const auto s = static_cast<hipStream_t>(stream);
        hipEvent_t e0, e1;
        hc(hipEventCreate(&e0), "event");
        hc(hipEventCreate(&e1), "event");
        hc(hipEventRecord(e0, s), "record");
        dense.run(q, k, vt, vt_pitch, rows, keys, split, out, stream);
        hc(hipEventRecord(e1, s), "record");
        if (!split || !study || rows != keys) {
            hc(hipEventDestroy(e0), "event");
            hc(hipEventDestroy(e1), "event");
            return;
        }
        hc(hipEventSynchronize(e1), "sync");
        float dense_ms = 0;
        hc(hipEventElapsedTime(&dense_ms, e0, e1), "elapsed");
        study_call(q, k, vt, vt_pitch, rows, out, s, dense_ms, e0, e1);
        hc(hipEventDestroy(e0), "event");
        hc(hipEventDestroy(e1), "event");
        ++block;
    }
    std::uint64_t bound_rows(void* stream) override { return dense.bound_rows(stream); }
    void reset_bound_rows(void* stream) override { dense.reset_bound_rows(stream); }

private:
    Dev kc_, qbar_, st_, mean_, var_, thr_, part_, out_, acc_;
    void study_call(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* vt, std::uint64_t vt_pitch, std::uint32_t S, const std::uint16_t* dense_out,
        hipStream_t s, double dense_ms, hipEvent_t e0, hipEvent_t e1) {
        const sp::Geometry g = sp::geometry(S, P);
        if (g.dense) return;
        constexpr std::uint64_t W = 7168U;
        const std::uint32_t jc = g.cut / 64U;
        kc_.need(static_cast<std::size_t>(g.KT) * W * 2);
        qbar_.need(static_cast<std::size_t>(g.KT) * W * 4);
        st_.need(4U * W * 4);
        mean_.need(W * 4);
        var_.need(W * 4);
        thr_.need(static_cast<std::size_t>(g.KT) * 56U * 4);
        part_.need(attn::attention_scratch_floats(S, 56, 128, 2) * 4);
        out_.need(static_cast<std::size_t>(S) * W * 2);
        acc_.need(4 * 8);
        std::uint16_t* kc = kc_.as<std::uint16_t>();
        sp::launch_key_centroids(k, W, g, 0, jc, kc, s);
        sp::launch_key_centroids(k + static_cast<std::size_t>(g.cut) * W, W, g, jc, g.KT, kc + static_cast<std::size_t>(jc) * W, s);
        float* st = st_.as<float>();
        sp::launch_stats_partial(kc, jc, st, st + W, s);
        sp::launch_stats_partial(kc + static_cast<std::size_t>(jc) * W, g.KT - jc, st + 2 * W, st + 3 * W, s);
        sp::launch_stats_finalize(st, st + W, st + 2 * W, st + 3 * W, g.KT, mean_.as<float>(), var_.as<float>(), s);
        sp::launch_query_centroids(q, W, g, 0, g.KT, qbar_.as<float>(), s);
        float* o = part_.as<float>();
        const attn::Partials parts{o, o + static_cast<std::size_t>(2) * 56U * S * 128U, S, 56, 128, nullptr};
        const attn::Queries qq{q, 128, W, S, 56, 128};
        const attn::KeyBlock blocks[2] = {attn::KeyBlock{k, vt, 128, W, 128 * vt_pitch, vt_pitch, g.cut},
                                          attn::KeyBlock{k + static_cast<std::size_t>(g.cut) * W, vt + g.cut, 128, W, 128 * vt_pitch, vt_pitch, S - g.cut}};
        sp::Params prm;
        for (const float tau : taus) {
            prm.tau = tau;
            sp::launch_thresholds(qbar_.as<float>(), g.KT, mean_.as<float>(), var_.as<float>(), tau, thr_.as<float>(), s);
            const sp::DeviceLut l0 = sp::build_lut(g, prm, 0, g.QT, qbar_.as<float>(), thr_.as<float>(), 0, jc, kc, s);
            const sp::DeviceLut l1 = sp::build_lut(g, prm, 0, g.QT, qbar_.as<float>(), thr_.as<float>(), jc, g.KT, kc + static_cast<std::size_t>(jc) * W, s);
            std::int32_t* dummy = nullptr;
            hc(hipMalloc(&dummy, 16), "malloc");
            const sp::BlockLut luts[2] = {sp::BlockLut{l0.entries_n ? l0.entries.as<std::int32_t>() : dummy, l0.offsets.as<std::uint32_t>(), 0},
                                          sp::BlockLut{l1.entries_n ? l1.entries.as<std::int32_t>() : dummy, l1.offsets.as<std::uint32_t>(), jc}};
            hc(hipEventRecord(e0, s), "record");
            sp::launch_sparse_partials(qq, blocks, luts, 2, parts, 0, nullptr, s);
            hc(hipEventRecord(e1, s), "record");
            attn::launch_merge(parts, 2, attn::Output{out_.as<std::uint16_t>(), nullptr, W, nullptr}, s);
            double r[2][2];
            const std::uint64_t ranges[2][2] = {{0, static_cast<std::uint64_t>(S) * W}, {static_cast<std::uint64_t>(P) * W, static_cast<std::uint64_t>(S) * W}};
            for (int c = 0; c < 2; ++c) {
                hc(hipMemsetAsync(acc_.p, 0, 16, s), "memset");
                hipLaunchKernelGGL(rel_l2_kernel, dim3(1024), dim3(256), 0, s, out_.as<std::uint16_t>(), dense_out, ranges[c][0], ranges[c][1], acc_.as<double>());
                hc(hipMemcpyAsync(r[c], acc_.p, 16, hipMemcpyDeviceToHost, s), "copy");
                hc(hipStreamSynchronize(s), "sync");
            }
            float sms = 0;
            hc(hipEventElapsedTime(&sms, e0, e1), "elapsed");
            hc(hipFree(dummy), "free");
            CallStat cs;
            cs.step = step;
            cs.block = block;
            cs.tau = tau;
            cs.rel_all = std::sqrt(r[0][0] / std::max(r[0][1], 1e-300));
            cs.rel_video = std::sqrt(r[1][0] / std::max(r[1][1], 1e-300));
            cs.density = static_cast<double>(l0.entries_n + l1.entries_n) / (static_cast<double>(g.QT) * 56.0 * g.KT);
            const unsigned long long us = l0.union_sum + l1.union_sum, ps = l0.parts_sum + l1.parts_sum;
            cs.inflation = ps ? static_cast<double>(us) / static_cast<double>(ps) : 0.0;
            cs.ms_sparse = sms;
            cs.ms_dense = dense_ms;
            stats.push_back(cs);
        }
    }
};

std::vector<float> read_f32(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + p);
    const auto n = static_cast<std::size_t>(f.tellg());
    std::vector<float> v(n / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    return v;
}
std::uint16_t bf(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
std::vector<std::uint16_t> to_bf16(const std::vector<float>& v) {
    std::vector<std::uint16_t> o(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) o[i] = bf(v[i]);
    return o;
}
std::vector<float> widen(const std::vector<std::uint16_t>& v) {
    std::vector<float> o(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        const std::uint32_t u = static_cast<std::uint32_t>(v[i]) << 16U;
        std::memcpy(&o[i], &u, 4);
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    std::string ckpt, te_path, detail = "0,3,7";
    int width = 896, height = 512, frames = 73, steps = 8;
    unsigned long long seed = 42;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--ckpt") ckpt = v;
        else if (a == "--te-hidden") te_path = v;
        else if (a == "--width") width = std::stoi(v);
        else if (a == "--height") height = std::stoi(v);
        else if (a == "--frames") frames = std::stoi(v);
        else if (a == "--steps") steps = std::stoi(v);
        else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--detail-steps") detail = v;
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (ckpt.empty() || te_path.empty()) {
        std::printf("usage: strix-h3-levers-sparse-study --ckpt <dit> --te-hidden <te_hidden_f32.bin> [--width 896] [--height 512] [--frames 73] [--steps 8]\n");
        return 2;
    }
    std::set<int> detail_steps;
    {
        std::stringstream ss(detail);
        std::string t;
        while (std::getline(ss, t, ',')) if (!t.empty()) detail_steps.insert(std::stoi(t));
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    const int n = ((frames - 5) / 17) * 17 + 5;
    const std::uint32_t lt = static_cast<std::uint32_t>(n <= 5 ? 2 : ((n - 5) / 17) * 5 + 2);
    const std::uint32_t at = static_cast<std::uint32_t>(std::nearbyint(static_cast<double>(n) / 24.0 * 40.0));
    const std::uint32_t lh = static_cast<std::uint32_t>(height / 16), lw = static_cast<std::uint32_t>(width / 16);
    const std::vector<float> hidden = read_f32(te_path);
    const std::uint32_t L = static_cast<std::uint32_t>(hidden.size() / blk::kTextDim);
    if (!L || hidden.size() != static_cast<std::size_t>(L) * blk::kTextDim) throw std::runtime_error("te hidden size");
    const std::vector<std::uint16_t> text = to_bf16(hidden);
    blk::SafeTensors st(ckpt);
    const blk::GlobalWeights g = blk::load_global(st);
    blk::FileBlockSource blocks(st, g.basis, false);
    const blk::PackedLayout lay = blk::PackedLayout::build(L, lt, lh, lw, at);
    const std::uint32_t S = lay.seq_len, P = lay.find(blk::SegKind::video).start;
    std::printf("[study] %dx%d, %d frames -> latent [24][%u][%u][%u], audio %u, text %u: S = %u, prefix P = %u, lever cut %u; %d Euler steps, seed %llu\n",
        width, height, n, lt, lh, lw, at, L, S, P, sp::geometry(S, P).cut, steps, seed);
    std::fflush(stdout);
    blk::DeviceForward dev(g);
    std::vector<std::uint16_t> refined;
    {
        blk::FlashDeviceAttention rfa;
        refined = dev.refine_text(text.data(), L, rfa);
    }
    blk::GemmDeviceLinear lin(S);
    StudyAttention att;
    att.P = P;
    const std::size_t nv = static_cast<std::size_t>(24) * lt * lh * lw, na = static_cast<std::size_t>(32) * 2 * at;
    const std::vector<float> sig = blk::simple_sigmas(12.0F, static_cast<std::uint32_t>(steps));
    blk::ForwardPlanConfig pcfg;
    blk::TorchCpuGenerator gen(seed);
    const std::vector<float> noise_v = gen.randn(nv), noise_a = gen.randn(na);
    blk::SamplerCarry carry;
    carry.x = blk::initial_state(sig[0], noise_v, noise_a);
    const float audio_scale = 12.0F / 3.0F;
    int step = 0;
    blk::AvModelFn model = [&](const blk::AvState& x, float sigma) {
        const double ts = secs();
        const blk::ForwardPlan p = blk::ForwardPlan::build(lay, sigma, {}, pcfg);
        blk::StreamInputs in;
        in.T = lt;
        in.H = lh;
        in.W = lw;
        in.audio_t = at;
        in.video_x = to_bf16(x.video);
        const std::vector<std::uint16_t> a_carried = to_bf16(x.audio);
        in.audio_x = blk::uncarry_audio(a_carried, p);
        in.text = refined;
        att.study = true;
        att.step = static_cast<std::uint32_t>(step);
        att.block = 0;
        const std::size_t first = att.stats.size();
        const blk::StreamOutputs so = dev.run(blocks, lay, p, in, lin, att);
        const auto va = blk::carry_audio_velocity(a_carried, so.audio_neg_v, p, audio_scale);
        // summary of this step: per tau, over the sparse layers (blocks >= 2)
        std::printf("[step %d/%d] sigma %.5f: forward + study %.1f s (%u blocks studied)\n", step + 1, steps, static_cast<double>(sigma), secs() - ts,
            att.block);
        for (const float tau : att.taus) {
            double sa = 0, sv = 0, mx = 0, dn = 0, inf = 0, ms_s = 0, ms_d = 0;
            std::uint32_t c = 0, mxb = 0;
            for (std::size_t i = first; i < att.stats.size(); ++i) {
                const CallStat& cs = att.stats[i];
                if (cs.tau != tau || cs.block < 2U) continue;
                sa += cs.rel_all;
                sv += cs.rel_video;
                if (cs.rel_video > mx) { mx = cs.rel_video; mxb = cs.block; }
                dn += cs.density;
                inf += cs.inflation;
                ms_s += cs.ms_sparse;
                ms_d += cs.ms_dense;
                ++c;
            }
            if (!c) continue;
            std::printf("  tau %.2f: rel L2 mean all %.4f video %.4f, max video %.4f (block %u), density mean %.3f, union / (|A|+|B|) %.3f, "
                        "sparse partials %.1f ms vs dense attention %.1f ms per block\n",
                static_cast<double>(tau), sa / c, sv / c, mx, mxb, dn / c, inf / c, ms_s / c, ms_d / c);
        }
        if (detail_steps.count(step)) {
            for (std::size_t i = first; i < att.stats.size(); i += att.taus.size()) {
                std::printf("    block %2u:", att.stats[i].block);
                for (std::size_t j = 0; j < att.taus.size() && i + j < att.stats.size(); ++j)
                    std::printf("  tau %.2f rel %.4f (video %.4f) dens %.3f", static_cast<double>(att.stats[i + j].tau), att.stats[i + j].rel_all,
                        att.stats[i + j].rel_video, att.stats[i + j].density);
                std::printf("\n");
            }
        }
        std::fflush(stdout);
        ++step;
        return blk::AvState{widen(so.video_neg_v), widen(va)};
    };
    while (carry.next < sig.size() - 1) blk::sample_steps(model, carry, sig, blk::SamplerKind::euler, carry.next + 1);
    std::printf("[study] done in %.1f s\n", secs());
    return 0;
}
