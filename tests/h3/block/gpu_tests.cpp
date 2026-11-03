// MiniMax H3 block glue: GPU tests (gfx1151).
// usage: strix-h3-block-gpu-tests [kernels] [forward]
//   kernels: every glue kernel bitwise vs the host reference on random data + batch invariance (no files needed)
//   forward: device forward (glue on the GPU, host-reference linears / attention) vs the host forward, bitwise per block,
//            and G2 vs the capture (env H3_FIXTURES, H3_CKPT, H3_ATTN=capture|f64)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>

#include "h3/block/device_forward.hpp"
#include "h3/block/glue.hpp"
#include "h3/block/host_ref.hpp"
#include "h3/block/layout.hpp"
#include "h3/block/sampler.hpp"
#include "test_util.hpp"

using namespace strix::h3::block;
using namespace h3t;

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
void hc(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::printf("hip error %s: %s\n", w, hipGetErrorString(e));
        std::exit(3);
    }
}
template <class T> T* dput(const std::vector<T>& v) {
    T* p = nullptr;
    hc(hipMalloc(&p, std::max<std::size_t>(16, v.size() * sizeof(T))), "malloc");
    hc(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return p;
}
template <class T> T* dalloc(std::size_t n) {
    T* p = nullptr;
    hc(hipMalloc(&p, std::max<std::size_t>(16, n * sizeof(T))), "malloc");
    hc(hipMemset(p, 0, std::max<std::size_t>(16, n * sizeof(T))), "memset");
    return p;
}
template <class T> std::vector<T> dget(const T* p, std::size_t n) {
    hc(hipDeviceSynchronize(), "sync");
    std::vector<T> v(n);
    hc(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return v;
}

// bf16 values with a wide spread (normal * 2^e, e in [-20, 12]) incl. zeros and massive outliers.
std::vector<std::uint16_t> rand_bf16(std::size_t n, std::mt19937& rng, float spread = 1.0F) {
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_int_distribution<int> ex(-20, 12), pick(0, 999);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) {
        const int p = pick(rng);
        float f = nd(rng) * spread;
        if (p < 5) f = 0.0F;
        else if (p < 15) f *= std::ldexp(1.0F, ex(rng));
        else if (p < 17) f *= 3.0e4F;
        x = tob(f);
    }
    return v;
}
std::vector<float> rand_f32(std::size_t n, std::mt19937& rng, float scale = 1.0F) {
    std::normal_distribution<float> nd(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}
template <class T> std::size_t ndiff(const std::vector<T>& a, const std::vector<T>& b) {
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return d + (a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

void test_kernels() {
    std::puts("[kernels]");
    std::mt19937 rng(1234);
    const std::uint32_t R = 1000, H = kHidden, NM = 9;
    // ---- K3 rmsnorm + modulation, gated residual
    auto x = rand_bf16(static_cast<std::size_t>(R) * H, rng, 20.0F);
    auto w = rand_bf16(H, rng);
    auto s1 = rand_bf16(static_cast<std::size_t>(NM) * H, rng), sh = rand_bf16(static_cast<std::size_t>(NM) * H, rng);
    std::vector<std::uint32_t> mrow(R);
    for (std::uint32_t r = 0; r < R; ++r) mrow[r] = (r / 37U) % NM;
    std::vector<std::uint16_t> ref(x.size()), refp(x.size());
    for (std::uint32_t r = 0; r < R; ++r) rmsnorm_row(x.data() + r * H, w.data(), H, kNormEps, ref.data() + r * H);
    refp = ref;
    for (std::uint32_t r = 0; r < R; ++r) mod_scale_shift_row(ref.data() + r * H, s1.data() + mrow[r] * H, sh.data() + mrow[r] * H, H);
    auto *dx = dput(x), *dw = dput(w), *ds1 = dput(s1), *dsh = dput(sh);
    auto* dm = dput(mrow);
    auto* dh = dalloc<std::uint16_t>(x.size());
    // diagnostics: per-row sum of squares and inverse RMS, GPU vs host reference order vs fp64
    {
        auto* dss = dalloc<float>(R);
        auto* dinv = dalloc<float>(R);
        gpu::launch_row_rms_debug(dx, R, H, kNormEps, dss, dinv, nullptr);
        const auto gss = dget(dss, R), ginv = dget(dinv, R);
        std::size_t bad_ss = 0, bad_inv = 0;
        for (std::uint32_t r = 0; r < R; ++r) {
            float sq[kHidden];
            double s64 = 0;
            for (std::uint32_t i = 0; i < H; ++i) {
                const float f = bf(x[r * H + i]);
                sq[i] = f * f;
                s64 += static_cast<double>(f) * f;
            }
            const float hs = ref_sum_f32(sq, H);
            const float hinv = rms_inv_bf16(x.data() + r * H, H, kNormEps);
            if (hs != gss[r] || hinv != ginv[r]) {
                if (bad_ss + bad_inv < 6)
                    std::printf("   row %u: ss host %.9g gpu %.9g f64 %.9g | inv host %.9g gpu %.9g\n", r, hs, gss[r], s64, hinv, ginv[r]);
                bad_ss += hs != gss[r];
                bad_inv += hinv != ginv[r];
            }
        }
        std::printf("  info  K3 rows with ss mismatch %zu, inv mismatch %zu (of %u)\n", bad_ss, bad_inv, R);
        hc(hipFree(dss), "f");
        hc(hipFree(dinv), "f");
    }
    gpu::launch_rmsnorm_mod(dx, dw, dm, ds1, dsh, R, H, kNormEps, dh, nullptr);
    {
        const auto got = dget(dh, x.size());
        std::size_t nd = 0;
        for (std::size_t i = 0; i < got.size(); ++i)
            if (got[i] != ref[i] && nd++ < 4)
                std::printf("   elem %zu (row %zu col %zu): host %04x gpu %04x x %04x\n", i, i / H, i % H, ref[i], got[i], x[i]);
        std::printf("  info  K3 element mismatches %zu\n", nd);
        check(nd == 0, "K3 rmsnorm + modulation bitwise (1000 rows, 9 mod rows)");
    }
    gpu::launch_rmsnorm_mod(dx, dw, dm, nullptr, nullptr, R, H, kNormEps, dh, nullptr);
    check(ndiff(dget(dh, x.size()), refp) == 0, "K3 plain rmsnorm bitwise");
    // batch invariance: rows 300..376 alone
    {
        auto* dh2 = dalloc<std::uint16_t>(static_cast<std::size_t>(77) * H);
        gpu::launch_rmsnorm_mod(dx + 300 * H, dw, dm + 300, ds1, dsh, 77, H, kNormEps, dh2, nullptr);
        const auto sub = dget(dh2, static_cast<std::size_t>(77) * H);
        check(std::equal(sub.begin(), sub.end(), ref.begin() + 300 * H), "K3 batch invariance (77 rows at offset 300 == full run)");
        hc(hipFree(dh2), "free");
    }
    {
        auto o = rand_bf16(x.size(), rng, 50.0F);
        auto gt = rand_bf16(static_cast<std::size_t>(NM) * H, rng, 0.3F);
        auto xr = x;
        for (std::uint32_t r = 0; r < R; ++r) gate_row(xr.data() + r * H, o.data() + r * H, gt.data() + mrow[r] * H, H);
        auto *dxx = dput(x), *dob = dput(o), *dg = dput(gt);
        gpu::launch_gate_residual(dxx, dob, dm, dg, R, H, nullptr);
        check(ndiff(dget(dxx, x.size()), xr) == 0, "K3 gated residual bitwise");
        hc(hipFree(dxx), "f"); hc(hipFree(dob), "f"); hc(hipFree(dg), "f");
    }
    // ---- K4 q/k norm + rope, V^T
    {
        const std::uint32_t S = 333;
        auto qkv = rand_bf16(static_cast<std::size_t>(S) * 3 * kInner, rng, 5.0F);
        auto qw = rand_bf16(kHeadDim, rng), kw = rand_bf16(kHeadDim, rng);
        std::vector<double> pos;
        for (std::uint32_t r = 0; r < S; ++r) {
            pos.push_back(r * 0.37);
            pos.push_back((r % 17) * 1.3 - 9.0);
            pos.push_back((r % 23) * 0.7 - 6.0);
        }
        std::vector<float> invf(16);
        for (int f = 0; f < 16; ++f) invf[f] = static_cast<float>(std::pow(10000.0, -f / 16.0));
        const RopeTable rt = rope_table(rope_angles(pos, invf));
        std::vector<std::uint16_t> q(static_cast<std::size_t>(S) * kInner), k(q.size()), vt(q.size()), q0(q.size()), k0(q.size()), vt0(q.size());
        qk_prepare(qkv.data(), S, qw.data(), kw.data(), &rt, q.data(), k.data(), vt.data());
        qk_prepare(qkv.data(), S, qw.data(), kw.data(), nullptr, q0.data(), k0.data(), vt0.data());
        auto *dqkv = dput(qkv), *dqw = dput(qw), *dkw = dput(kw), *dc = dput(rt.c), *dsn = dput(rt.s);
        auto *dq = dalloc<std::uint16_t>(q.size()), *dk = dalloc<std::uint16_t>(q.size()), *dvt = dalloc<std::uint16_t>(q.size());
        gpu::launch_qk_norm_rope(dqkv, S, dqw, dkw, dc, dsn, kNormEps, dq, dk, nullptr);
        gpu::launch_transpose_v(dqkv, S, dvt, S, nullptr);
        check(ndiff(dget(dq, q.size()), q) == 0, "K4 q RMSNorm + split-half RoPE bitwise (S=333, token-major)");
        check(ndiff(dget(dk, k.size()), k) == 0, "K4 k RMSNorm + split-half RoPE bitwise");
        check(ndiff(dget(dvt, vt.size()), vt) == 0, "K4 V^T [56][128][S] bitwise");
        {
            const std::uint64_t pitch = vt_pitch_for(S);  // 336
            auto* dvp = dalloc<std::uint16_t>(static_cast<std::size_t>(kInner) * pitch);
            gpu::launch_transpose_v(dqkv, S, dvp, pitch, nullptr);
            const auto got = dget(dvp, static_cast<std::size_t>(kInner) * pitch);
            std::size_t nd = 0;
            for (std::size_t r = 0; r < kInner; ++r)
                for (std::uint32_t j = 0; j < S; ++j) nd += got[r * pitch + j] != vt[r * S + j];
            check(nd == 0, "K4 V^T with the attention pitch (336 for S=333) bitwise");
            hc(hipFree(dvp), "f");
        }
        gpu::launch_qk_norm_rope(dqkv, S, dqw, dkw, nullptr, nullptr, kNormEps, dq, dk, nullptr);
        check(ndiff(dget(dq, q.size()), q0) == 0 && ndiff(dget(dk, k.size()), k0) == 0, "K4 without rotation (refiner form) bitwise");
        for (auto* p : {dqkv, dqw, dkw, dc, dsn, dq, dk, dvt}) hc(hipFree(p), "f");
    }
    // ---- refiner glue: bf16 linear (fp64 k-ascending) and the residual add
    {
        Bf16Linear l;
        l.n = 320;
        l.k = 5376;
        l.w = rand_bf16(static_cast<std::size_t>(l.n) * l.k, rng, 0.05F);
        l.b = rand_bf16(l.n, rng);
        const std::uint32_t mr = 37;
        auto xx = rand_bf16(static_cast<std::size_t>(mr) * l.k, rng, 3.0F);
        std::vector<std::uint16_t> yh(static_cast<std::size_t>(mr) * l.n);
        bf16_linear_host(l, xx.data(), mr, yh.data());
        auto *dwl = dput(l.w), *dbl = dput(l.b), *dxl = dput(xx);
        auto* dyl = dalloc<std::uint16_t>(yh.size());
        gpu::launch_bf16_linear(dxl, mr, dwl, dbl, l.n, l.k, dyl, nullptr);
        check(ndiff(dget(dyl, yh.size()), yh) == 0, "refiner bf16 linear (37 x 320 x 5376, bias) bitwise");
        gpu::launch_bf16_linear(dxl, mr, dwl, nullptr, l.n, l.k, dyl, nullptr);
        l.b.clear();
        bf16_linear_host(l, xx.data(), mr, yh.data());
        check(ndiff(dget(dyl, yh.size()), yh) == 0, "refiner bf16 linear without bias bitwise");
        auto o = rand_bf16(yh.size(), rng, 4.0F);
        auto xr = yh;
        for (std::size_t i = 0; i < xr.size(); ++i) xr[i] = tob(bf(o[i]) + bf(xr[i]));
        auto* dob = dput(o);
        gpu::launch_add_residual(dyl, dob, yh.size(), nullptr);
        check(ndiff(dget(dyl, yh.size()), xr) == 0, "refiner residual add bitwise");
        for (auto* p : {dwl, dbl, dxl, dyl, dob}) hc(hipFree(p), "f");
    }
    // ---- K5 swiglu over every bf16 gate value
    {
        const std::uint32_t rows = 4, ffn = 16384;
        std::vector<std::uint16_t> f1(static_cast<std::size_t>(rows) * 2 * ffn);
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t i = 0; i < ffn; ++i) {
                f1[static_cast<std::size_t>(r) * 2 * ffn + i] = static_cast<std::uint16_t>(r * ffn + i);  // all 65536 gates
                f1[static_cast<std::size_t>(r) * 2 * ffn + ffn + i] = tob(static_cast<float>((i % 97) - 48) * 0.37F);
            }
        std::vector<std::uint16_t> ref2(static_cast<std::size_t>(rows) * ffn);
        swiglu_host(f1.data(), rows, ffn, ref2.data());
        const auto& tab = strix::h3::gemm::silu_table();
        auto* dt = dput(std::vector<std::uint16_t>(tab.begin(), tab.end()));
        auto* df = dput(f1);
        auto* dout = dalloc<std::uint16_t>(ref2.size());
        gpu::launch_swiglu(df, rows, ffn, dt, dout, nullptr);
        check(ndiff(dget(dout, ref2.size()), ref2) == 0, "K5 SwiGLU bitwise (all 65536 gate values)");
        for (auto* p : {dt, df, dout}) hc(hipFree(p), "f");
    }
    // ---- K6 fp64 linear (adaLN shape and head shape), mod vectors, patch projection, final mod
    {
        F32Linear ad;
        ad.n = 96768;
        ad.k = 8;
        ad.w = rand_f32(static_cast<std::size_t>(ad.n) * ad.k, rng, 0.3F);
        ad.b = rand_f32(ad.n, rng, 0.5F);
        auto te = rand_f32(2 * 8, rng);
        const auto ada = f32_linear(ad, te.data(), 2);
        auto *dwt = dput(ad.w), *dbt = dput(ad.b), *dte = dput(te);
        auto* dada = dalloc<float>(ada.size());
        gpu::launch_linear_f64(dte, 8, 2, dwt, dbt, ad.n, 8, dada, nullptr);
        check(ndiff(dget(dada, ada.size()), ada) == 0, "K6 adaLN linear (fp64 dot, 2 x 96768 x 8) bitwise");
        const ModVectors mv = mod_vectors(ada, 6);
        std::uint16_t* dmod[6];
        for (auto& p : dmod) p = dalloc<std::uint16_t>(static_cast<std::size_t>(6) * H);
        gpu::launch_mod_vectors(dada, 6, dmod, nullptr);
        const std::vector<std::uint16_t>* hm[6] = {&mv.shift_msa, &mv.s1_msa, &mv.gate_msa, &mv.shift_mlp, &mv.s1_mlp, &mv.gate_mlp};
        bool ok = true;
        for (int j = 0; j < 6; ++j) ok &= ndiff(dget(dmod[j], hm[j]->size()), *hm[j]) == 0;
        check(ok, "K6 mod vectors (shift, 1 + scale, gate as bf16) bitwise");
        for (auto* p : dmod) hc(hipFree(p), "f");
        F32Linear hd;
        hd.n = 96;
        hd.k = H;
        hd.w = rand_f32(static_cast<std::size_t>(hd.n) * H, rng, 0.02F);
        hd.b = rand_f32(96, rng);
        auto f = rand_f32(static_cast<std::size_t>(50) * H, rng, 3.0F);
        const auto y = f32_linear(hd, f.data(), 50);
        auto *dhw = dput(hd.w), *dhb = dput(hd.b), *dfx = dput(f);
        auto* dy = dalloc<float>(y.size());
        gpu::launch_linear_f64(dfx, H, 50, dhw, dhb, 96, H, dy, nullptr);
        check(ndiff(dget(dy, y.size()), y) == 0, "K6 output head (fp64 dot, 50 x 96 x 5376) bitwise");
        // patch projection into scattered rows
        F32Linear pp;
        pp.n = H;
        pp.k = 96;
        for (std::size_t i = 0; i < static_cast<std::size_t>(H) * 96; ++i) pp.w.push_back(bf(tob(rand_f32(1, rng, 0.1F)[0])));
        pp.b = rand_f32(H, rng, 0.1F);
        auto rows = rand_f32(static_cast<std::size_t>(40) * 96, rng);
        std::vector<std::int32_t> dst(40);
        for (int i = 0; i < 40; ++i) dst[i] = 100 + 3 * i;
        std::vector<std::uint16_t> hh(static_cast<std::size_t>(300) * H, 0), hdev;
        for (int i = 0; i < 40; ++i) {
            const auto yy = patch_linear_row(pp, rows.data() + i * 96);
            for (std::uint32_t c = 0; c < H; ++c) hh[static_cast<std::size_t>(dst[i]) * H + c] = tob(yy[c]);
        }
        auto *dpw = dput(pp.w), *dpb = dput(pp.b), *drw = dput(rows);
        auto* dds = dput(dst);
        auto* dhh = dalloc<std::uint16_t>(hh.size());
        gpu::launch_patch_proj(drw, 40, 96, dpw, dpb, dds, dhh, nullptr);
        check(ndiff(dget(dhh, hh.size()), hh) == 0, "K6 patch projection (fp32 k-ascending) + row scatter bitwise");
        // final norm + modulation
        auto fw = rand_bf16(H, rng);
        auto fsh = rand_f32(H, rng, 0.3F), fsc = rand_f32(H, rng, 0.3F);
        std::vector<float> fref(static_cast<std::size_t>(64) * H);
        for (std::uint32_t r = 0; r < 64; ++r) {
            std::vector<std::uint16_t> nrm(H);
            rmsnorm_row(x.data() + (200 + r) * H, fw.data(), H, kNormEps, nrm.data());
            for (std::uint32_t c = 0; c < H; ++c) fref[r * H + c] = (bf(nrm[c]) * (1.0F + fsc[c])) + fsh[c];
        }
        auto* dfw = dput(fw);
        auto *dfsh = dput(fsh), *dfsc = dput(fsc);
        auto* dff = dalloc<float>(fref.size());
        gpu::launch_final_mod(dx, 200, 64, dfw, dfsh, dfsc, kNormEps, dff, nullptr);
        check(ndiff(dget(dff, fref.size()), fref) == 0, "K6 final norm + modulation (fp32) bitwise");
        for (void* p : {static_cast<void*>(dwt), static_cast<void*>(dbt), static_cast<void*>(dte), static_cast<void*>(dada),
                 static_cast<void*>(dhw), static_cast<void*>(dhb), static_cast<void*>(dfx), static_cast<void*>(dy), static_cast<void*>(dpw),
                 static_cast<void*>(dpb), static_cast<void*>(drw), static_cast<void*>(dds), static_cast<void*>(dhh), static_cast<void*>(dfw),
                 static_cast<void*>(dfsh), static_cast<void*>(dfsc), static_cast<void*>(dff)})
            hc(hipFree(p), "f");
    }
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dw), static_cast<void*>(ds1), static_cast<void*>(dsh),
             static_cast<void*>(dm), static_cast<void*>(dh)})
        hc(hipFree(p), "f");
}

struct Model {
    SafeTensors st;
    GlobalWeights g;
    FileBlockSource blocks;
    Model() : st(ckpt_path()), g(load_global(st)), blocks(st, g.basis, true) {}
};
Model& model() {
    static Model m;
    return m;
}
StreamInputs small_inputs(const fx::Manifest& m, const std::string& fwd) {
    StreamInputs in;
    in.T = 2;
    in.H = 16;
    in.W = 16;
    in.audio_t = 8;
    in.video_x = bf16v(m, fwd + "/stream_in/video_x");
    in.audio_x = bf16v(m, fwd + "/stream_in/audio_x");
    in.text = bf16v(m, fwd + "/model_in/context");
    return in;
}

// Device forward at S=400: (1) gemm linears + host capture-emulating attention == host forward bitwise per block;
// (2) gemm linears + flash attention: numbers vs the capture and the forward time (weights resident).
void test_forward() {
    std::puts("[forward dit_t2va_small f0]");
    const fx::Manifest m = fx::Manifest::load(fixtures_root() + "/dit_t2va_small");
    Model& md = model();
    const PackedLayout L = PackedLayout::build(256, 2, 16, 16, 8);
    const ForwardPlan p = ForwardPlan::build(L, f32v(m, "sampler/sigmas_simple20")[0], {});
    const StreamInputs in = small_inputs(m, "f0");
    HostLinear hl;
    CaptureAttention ha;
    std::vector<std::vector<std::uint16_t>> res_h, res_d;
    ForwardTaps th;
    th.blocks = &res_h;
    double t0 = now_s();
    const StreamOutputs oh = stream_forward_host(md.g, md.blocks, L, p, in, hl, ha, &th);
    std::printf("  host forward %.1f s (capture-emulating attention, incl. block loads)\n", now_s() - t0);
    DeviceForward dev(md.g);
    GemmDeviceLinear dl(2304);
    {
        HostFallbackAttention da(std::unique_ptr<AttentionOp>(new CaptureAttention));
        DeviceForwardTaps td;
        td.blocks = &res_d;
        t0 = now_s();
        const StreamOutputs od = dev.run(md.blocks, L, p, in, dl, da, &td);
        std::printf("  device forward, gemm linears + host attention: %.1f s (incl. uploading %.2f GB of int8 weights)\n", now_s() - t0,
            static_cast<double>(dl.resident_bytes()) / 1e9);
        std::size_t bad = 0;
        for (std::size_t i = 0; i < res_h.size(); ++i) bad += res_d.size() > i && res_d[i] == res_h[i] ? 0 : 1;
        check(bad == 0, "device (gemm linears) residual == host residual bitwise after every block (" + std::to_string(res_h.size() - bad) + "/50)");
        check(od.video_neg_v == oh.video_neg_v && od.audio_neg_v == oh.audio_neg_v, "device (gemm linears) velocities == host bitwise");
        std::printf("  info  flushed rotation groups: device %llu, host %llu (the gemm exact-rotation window)\n",
            static_cast<unsigned long long>(td.flushed_groups), static_cast<unsigned long long>(hl.inexact_groups()));
        check(td.flushed_groups == hl.inexact_groups(), "device and host flush the same rotation groups");
    }
    FlashDeviceAttention fa;
    std::vector<std::vector<std::uint16_t>> res_f;
    DeviceForwardTaps tf;
    tf.blocks = &res_f;
    const StreamOutputs of = dev.run(md.blocks, L, p, in, dl, fa, &tf);
    std::printf("   block  flash device vs capture rel_l2 | vs host (capture-emulating attention) rel_l2\n");
    for (std::size_t i = 0; i < res_f.size(); i += (i < 4 || i > 45) ? 1 : 5) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "f0/blocks/out_%02zu", i);
        std::printf("   %5zu  %.3e | %.3e\n", i, compare_bf16(res_f[i], bf16v(m, nm)).rel_l2, compare_bf16(res_f[i], res_h[i]).rel_l2);
    }
    const Cmp cv = compare_bf16(of.video_neg_v, bf16v(m, "f0/stream_out/video_neg_v"));
    const Cmp ca = compare_bf16(of.audio_neg_v, bf16v(m, "f0/stream_out/audio_neg_v"));
    std::printf("  info  flash forward velocities vs capture: video rel_l2 %.3e cos %.7f, audio rel_l2 %.3e cos %.7f\n", cv.rel_l2, cv.cosine,
        ca.rel_l2, ca.cosine);
    {
        // G2(a) with the flash kernel: teacher forced (each block from the captured residual of the previous block)
        std::vector<std::vector<std::uint16_t>> teacher, res_t;
        for (int i = 0; i < 50; ++i) {
            char nm[64];
            std::snprintf(nm, sizeof nm, "f0/blocks/out_%02d", i);
            teacher.push_back(bf16v(m, nm));
        }
        DeviceForwardTaps tt;
        tt.teacher = &teacher;
        tt.blocks = &res_t;
        const StreamOutputs ot = dev.run(md.blocks, L, p, in, dl, fa, &tt);
        double mx = 0;
        std::size_t worst = 0;
        for (std::size_t i = 0; i < res_t.size(); ++i) {
            const double r = compare_bf16(res_t[i], teacher[i]).rel_l2;
            if (r > mx) {
                mx = r;
                worst = i;
            }
        }
        std::printf("  info  G2(a) flash teacher-forced: max block rel_l2 %.3e (block %zu); blocks 0/24/49: %.3e %.3e %.3e\n", mx, worst,
            compare_bf16(res_t[0], teacher[0]).rel_l2, compare_bf16(res_t[24], teacher[24]).rel_l2, compare_bf16(res_t[49], teacher[49]).rel_l2);
        check(mx <= 2.5e-2, "G2(a) t2va f0 teacher-forced with the flash kernel: all 50 blocks rel_l2 <= 2.5e-2 (exact/flash bound)");
        const Cmp bv = compare_bf16(ot.video_neg_v, bf16v(m, "f0/stream_out/video_neg_v"));
        check(bv.mism * 1000 <= bv.n, "G2(b) device final layer from the captured residual: video mism " + std::to_string(bv.mism));
    }
    for (int rep = 0; rep < 3; ++rep) {
        t0 = now_s();
        const StreamOutputs o2 = dev.run(md.blocks, L, p, in, dl, fa);
        const double dt = now_s() - t0;
        std::printf("  info  S=400 forward (gemm + flash, weights resident) rep %d: %.1f ms\n", rep, dt * 1e3);
        if (rep == 0) check(o2.video_neg_v == of.video_neg_v && o2.audio_neg_v == of.audio_neg_v, "repeat forward bitwise identical");
    }
    std::vector<double> ms;
    DeviceForwardTaps tt;
    tt.block_ms = &ms;
    dev.run(md.blocks, L, p, in, dl, fa, &tt);
    double sum = 0;
    for (double v : ms) sum += v;
    std::printf("  info  per-block (synchronized) mean %.2f ms, block 0 %.2f ms, sum %.1f ms\n", sum / static_cast<double>(ms.size()), ms[0], sum);
}

// Device token refiner: == host refiner bitwise with the same (host) attention; with flash attention vs the capture.
void test_refiner() {
    std::puts("[refiner dit_t2va_small]");
    const fx::Manifest m = fx::Manifest::load(fixtures_root() + "/dit_t2va_small");
    Model& md = model();
    const auto text = bf16v(m, "in/text_states");
    const std::uint32_t L = 256;
    HostAttention ha;
    double t0 = now_s();
    const auto rh = text_refine_host(md.g, text.data(), L, ha);
    std::printf("  host refiner %.1f s\n", now_s() - t0);
    DeviceForward dev(md.g);
    HostFallbackAttention da(std::unique_ptr<AttentionOp>(new HostAttention));
    t0 = now_s();
    const auto rd = dev.refine_text(text.data(), L, da);
    std::printf("  device refiner (host attention) %.1f s incl. weight upload\n", now_s() - t0);
    check(rd == rh, "device refiner == host refiner bitwise (same attention)");
    FlashDeviceAttention fa;
    const auto rf0 = dev.refine_text(text.data(), L, fa);
    t0 = now_s();
    const auto rf = dev.refine_text(text.data(), L, fa);
    std::printf("  info  refiner attention contract (measured): %.1f, limit 16384\n", dev.last_refiner_score());
    check(dev.last_refiner_score() > 0.0, "refiner contract check ran");
    const double dt = now_s() - t0;
    check(rf == rf0, "device refiner (flash) deterministic");
    const Cmp c = compare_bf16(rf, bf16v(m, "text/refined"));
    check(c.rel_l2 < 1e-2 && c.cosine > 0.9999, "device refiner (flash attention, one key block) vs capture " + c.str());
    std::printf("  info  device refiner (flash, resident) %.1f ms; vs host-attention refiner rel_l2 %.3e\n", dt * 1e3, compare_bf16(rf, rh).rel_l2);
}

// S=2122 (dit_t2va_mid) block 0: gemm linears + host capture-emulating attention == host block 0 bitwise (V^T pitch
// 2128); gemm + flash: block 0 vs the capture and its time.
void test_mid() {
    std::puts("[mid dit_t2va_mid block 0, S=2122]");
    const fx::Manifest m = fx::Manifest::load(fixtures_root() + "/dit_t2va_mid");
    Model& md = model();
    const PackedLayout L = PackedLayout::build(256, 7, 32, 32, 37);
    const ForwardPlan p = ForwardPlan::build(L, f32v(m, "sampler/sigmas_simple20")[0], {});
    StreamInputs in;
    in.T = 7;
    in.H = 32;
    in.W = 32;
    in.audio_t = 37;
    in.video_x = bf16v(m, "f0/stream_in/video_x");
    in.audio_x = bf16v(m, "f0/stream_in/audio_x");
    in.text = bf16v(m, "f0/model_in/context");
    const auto b0 = md.blocks.block(0);
    std::vector<std::uint16_t> xh = embed_and_pack(md.g, L, in);
    {
        const Cmp ce = compare_bf16(xh, bf16v(m, "f0/blocks/in_00"));
        check(ce.mism * 10000 <= ce.n, "S=2122 embedding vs capture " + ce.str());
    }
    const auto te = curve_t_emb(md.g, p.t_vals);
    const ModVectors mv = mod_vectors(adaln_block_f32(*b0, te, p.n_t()), p.n_mod_rows());
    const RopeTable rope = rope_table(rope_angles(L.position_ids, md.g.inv_freq));
    HostLinear hl;
    CaptureAttention ca;
    double t0 = now_s();
    block_forward_host(*b0, mv, mod_row_of(p.mod_segments, L.seq_len), rope, xh, L.seq_len, hl, ca);
    std::printf("  host block 0 %.1f s\n", now_s() - t0);
    DeviceForward dev(md.g);
    GemmDeviceLinear dl(2304);
    std::vector<std::vector<std::uint16_t>> rd;
    DeviceForwardTaps td;
    td.blocks = &rd;
    td.stop_after_block = 0;
    {
        HostFallbackAttention da(std::unique_ptr<AttentionOp>(new CaptureAttention));
        dev.run(md.blocks, L, p, in, dl, da, &td);
    }
    check(rd.size() == 1 && rd[0] == xh, "S=2122 device block 0 (gemm linears, host attention) == host block 0 bitwise");
    {
        const Cmp c = compare_bf16(rd[0], bf16v(m, "f0/blocks/out_00"));
        check(c.rel_l2 < 5e-3, "S=2122 block 0 (capture-emulating attention) vs capture " + c.str());
    }
    FlashDeviceAttention fa;
    rd.clear();
    dev.run(md.blocks, L, p, in, dl, fa, &td);
    const Cmp c = compare_bf16(rd[0], bf16v(m, "f0/blocks/out_00"));
    check(c.rel_l2 < 1e-2, "S=2122 block 0 (gemm + flash) vs capture " + c.str());
    std::vector<double> ms;
    DeviceForwardTaps tt;
    tt.block_ms = &ms;
    tt.stop_after_block = 0;
    for (int rep = 0; rep < 3; ++rep) dev.run(md.blocks, L, p, in, dl, fa, &tt);
    std::printf("  info  S=2122 block 0 (gemm + flash, synchronized) %.2f / %.2f / %.2f ms\n", ms[0], ms[1], ms[2]);
}

// ------------------------------------------------------------------------------------------------ G2 with flash attention
struct GpuCase {
    fx::Manifest m;
    PackedLayout L;
    std::vector<std::int64_t> tags;
    std::vector<float> cond_video;
    static GpuCase load(const std::string& set) {
        GpuCase c{fx::Manifest::load(fixtures_root() + "/" + set), {}, {}, {}};
        const auto& sig = c.m.meta.at("layout").at("signature").arr();
        std::vector<Keyframe> kfs;
        const int nkf = static_cast<int>(c.m.meta.at("config").at("keyframes").num());
        for (int i = 0; i < nkf; ++i) {
            const std::string n = "in/keyframe" + std::to_string(i) + "_latent";
            const auto& sh = c.m.find(n)->shape;
            kfs.push_back({0.0, static_cast<std::uint32_t>(sh[2]), 0});
            const auto lat = f32v(c.m, n);
            const auto noise = f32v(c.m, "noise/cond_video_rows0");
            const auto rows = cond_video_rows(lat.data(), static_cast<std::uint32_t>(sh[2]), static_cast<std::uint32_t>(sh[3]),
                static_cast<std::uint32_t>(sh[4]), kVisualCondT, noise.data());
            c.cond_video.insert(c.cond_video.end(), rows.begin(), rows.end());
        }
        c.L = PackedLayout::build(static_cast<std::uint32_t>(sig[0].num()), static_cast<std::uint32_t>(sig[1].num()),
            static_cast<std::uint32_t>(sig[2].num()), static_cast<std::uint32_t>(sig[3].num()), static_cast<std::uint32_t>(sig[4].num()), kfs);
        c.tags = as_vec<std::int64_t>(c.m.read("in/text_token_tags"));
        bool all_text = true;
        for (auto t : c.tags) all_text &= t == 1;
        if (all_text) c.tags.clear();
        return c;
    }
    ForwardPlan plan(int f) const { return ForwardPlan::build(L, f32v(m, "sampler/sigmas_simple20")[f], tags); }
    StreamInputs inputs(int f) const {
        const std::string pre = "f" + std::to_string(f);
        StreamInputs in;
        in.T = L.latent_t;
        in.H = L.latent_h;
        in.W = L.latent_w;
        in.audio_t = L.audio_t;
        in.video_x = bf16v(m, pre + "/stream_in/video_x");
        in.audio_x = bf16v(m, pre + "/stream_in/audio_x");
        in.text = bf16v(m, pre + "/model_in/context");
        in.cond_video = cond_video;
        return in;
    }
};

// G2 with the flash kernel on the device, both small sets: (a) teacher-forced per block <= 2.5e-2 (exact /
// flash attention bound), f0 and f1; (b) final layer from the captured block-49 residual; (c) chained velocity <= 2 x the
// median one-ulp drift over 3 perturbations (redrawn until the block-0 output changes).
void test_g2flash() {
    Model& md = model();
    DeviceForward dev(md.g);
    GemmDeviceLinear dl(2304);
    FlashDeviceAttention fa;
    for (const char* set : {"dit_t2va_small", "dit_fl2va_small"}) {
        std::printf("[g2flash %s]\n", set);
        const GpuCase c = GpuCase::load(set);
        for (int f = 0; f < 2; ++f) {
            const std::string pre = "f" + std::to_string(f);
            std::vector<std::vector<std::uint16_t>> teacher, res;
            for (int i = 0; i < 50; ++i) {
                char nm[64];
                std::snprintf(nm, sizeof nm, "%s/blocks/out_%02d", pre.c_str(), i);
                teacher.push_back(bf16v(c.m, nm));
            }
            DeviceForwardTaps tt;
            tt.teacher = &teacher;
            tt.blocks = &res;
            const StreamOutputs o = dev.run(md.blocks, c.L, c.plan(f), c.inputs(f), dl, fa, &tt);
            double mx = 0;
            std::size_t worst = 0;
            for (std::size_t i = 0; i < res.size(); ++i) {
                const double r = compare_bf16(res[i], teacher[i]).rel_l2;
                if (r > mx) {
                    mx = r;
                    worst = i;
                }
            }
            char b[200];
            std::snprintf(b, sizeof b, "G2(a) %s %s flash teacher-forced: all 50 blocks rel_l2 <= 2.5e-2 (max %.3e at block %zu)", set,
                pre.c_str(), mx, worst);
            check(mx <= 2.5e-2, b);
            const Cmp bv = compare_bf16(o.video_neg_v, bf16v(c.m, pre + "/stream_out/video_neg_v"));
            const Cmp ba = compare_bf16(o.audio_neg_v, bf16v(c.m, pre + "/stream_out/audio_neg_v"));
            check(bv.mism * 1000 <= bv.n && bv.rel_l2 <= 1e-5 && ba.mism * 1000 <= ba.n && ba.rel_l2 <= 1e-5,
                std::string("G2(b) ") + set + " " + pre + " final layer from the captured residual: video mism " + std::to_string(bv.mism) +
                    ", audio mism " + std::to_string(ba.mism));
        }
        // (c)
        const ForwardPlan p = c.plan(0);
        const StreamInputs in = c.inputs(0);
        DeviceForwardTaps tb;
        const StreamOutputs base = dev.run(md.blocks, c.L, p, in, dl, fa, &tb);
        std::printf("  info  flushed rotation groups in the chained forward: %llu\n", static_cast<unsigned long long>(tb.flushed_groups));
        std::printf("  info  attention contract (measured): max||q|| max||k|| log2(e)/sqrt(128) = %.1f (block %u), limit 16384\n",
            tb.attn_score_max, tb.attn_score_block);
        check(tb.attn_score_max > 0.0 && tb.attn_score_max <= kAttnScoreLimit, "attention input contract holds in the chained forward");
        check(tb.attn_bound_rows == 0, "attention bound_rows telemetry 0 in the chained forward (" + std::to_string(tb.attn_bound_rows) + ")");
        const Cmp ev = compare_bf16(base.video_neg_v, bf16v(c.m, "f0/stream_out/video_neg_v"));
        const Cmp ea = compare_bf16(base.audio_neg_v, bf16v(c.m, "f0/stream_out/audio_neg_v"));
        std::vector<std::vector<std::uint16_t>> b0a;
        DeviceForwardTaps t0;
        t0.stop_after_block = 0;
        t0.blocks = &b0a;
        dev.run(md.blocks, c.L, p, in, dl, fa, &t0);
        std::vector<double> dv, da;
        for (std::uint32_t seed = 1; seed <= 3; ++seed) {
            std::mt19937 rng(seed);
            StreamInputs pin;
            std::size_t pos = 0;
            for (unsigned tries = 0;; ++tries) {
                pin = in;
                pos = std::uniform_int_distribution<std::size_t>(0, pin.text.size() - 1)(rng);
                const bool up = (rng() & 1U) != 0U;
                std::uint16_t& v = pin.text[pos];
                v = static_cast<std::uint16_t>(up || (v & 0x7fffU) == 0 ? v + 1 : v - 1);
                std::vector<std::vector<std::uint16_t>> b0b;
                DeviceForwardTaps t1;
                t1.stop_after_block = 0;
                t1.blocks = &b0b;
                dev.run(md.blocks, c.L, p, pin, dl, fa, &t1);
                if (b0b[0] != b0a[0]) break;
                std::printf("  info  seed %u draw %u: text value %zu absorbed, drawing again\n", seed, tries, pos);
            }
            const StreamOutputs o = dev.run(md.blocks, c.L, p, pin, dl, fa);
            const Cmp cv = compare_bf16(o.video_neg_v, base.video_neg_v), ca = compare_bf16(o.audio_neg_v, base.audio_neg_v);
            dv.push_back(cv.rel_l2);
            da.push_back(ca.rel_l2);
            std::printf("  info  perturbation seed %u: text value %zu (row %zu) -> drift video %.3e, audio %.3e\n", seed, pos, pos / kHidden,
                cv.rel_l2, ca.rel_l2);
        }
        std::sort(dv.begin(), dv.end());
        std::sort(da.begin(), da.end());
        char b1[200], b2[200];
        std::snprintf(b1, sizeof b1, "G2(c) %s flash video: chained rel_l2 %.3e (cos %.7f) <= 2 x median drift %.3e = %.3e", set, ev.rel_l2,
            ev.cosine, dv[1], 2 * dv[1]);
        std::snprintf(b2, sizeof b2, "G2(c) %s flash audio: chained rel_l2 %.3e (cos %.7f) <= 2 x median drift %.3e = %.3e", set, ea.rel_l2,
            ea.cosine, da[1], 2 * da[1]);
        check(ev.rel_l2 <= 2 * dv[1], b1);
        check(ea.rel_l2 <= 2 * da[1], b2);
    }
}

// ------------------------------------------------------------------------------------------------ real-shape timing
double gpu_ms(const std::function<void()>& fn, int reps = 3) {
    hc(hipDeviceSynchronize(), "sync");
    fn();  // warm
    hc(hipDeviceSynchronize(), "sync");
    const double t0 = now_s();
    for (int r = 0; r < reps; ++r) fn();
    hc(hipDeviceSynchronize(), "sync");
    return (now_s() - t0) * 1e3 / reps;
}

// Profiling target (run under rocprofv3 --kernel-trace --stats): block 0 at the 38K shape, 4 synchronized reps (the
// first is a warm-up), then the four linears and the attention alone at the same shape (as in `shape`).
void test_prof() {
    std::puts("[prof: block 0 at S ~38K, 4 reps + isolated ops]");
    Model& md = model();
    std::mt19937 rng(77);
    const PackedLayout L = PackedLayout::build(256, 37, 48, 84, 207);
    const std::uint32_t S = L.seq_len;
    const ForwardPlan p = ForwardPlan::build(L, 0.9F, {});
    StreamInputs in;
    in.T = 37;
    in.H = 48;
    in.W = 84;
    in.audio_t = 207;
    in.video_x = rand_bf16(static_cast<std::size_t>(24) * 37 * 48 * 84, rng);
    in.audio_x = rand_bf16(static_cast<std::size_t>(32) * 2 * 207, rng);
    in.text = rand_bf16(static_cast<std::size_t>(256) * kHidden, rng);
    DeviceForward dev(md.g);
    GemmDeviceLinear dl(S);
    FlashDeviceAttention fa;
    std::vector<double> ms;
    DeviceForwardTaps tt;
    tt.stop_after_block = 0;
    tt.block_ms = &ms;
    if (std::getenv("H3_PROF_ISO") == nullptr) {  // the block run (profile 1)
        for (int rep = 0; rep < 4; ++rep) dev.run(md.blocks, L, p, in, dl, fa, &tt);
        std::printf("  info  S=%u block ms: %.1f / %.1f / %.1f / %.1f\n", S, ms[0], ms[1], ms[2], ms[3]);
        return;
    }
    dev.run(md.blocks, L, p, in, dl, fa, &tt);  // profile 2: prepare block 0, then the isolated ops
    auto* dx = dput(rand_bf16(static_cast<std::size_t>(S) * kFfn, rng, 0.5F));
    auto* dy = dalloc<std::uint16_t>(static_cast<std::size_t>(S) * 3 * kInner);
    const double t_lin = gpu_ms([&] {
        dl.run(0, LinearKind::qkv, dx, S, dy, nullptr);
        dl.run(0, LinearKind::out, dx, S, dy, nullptr);
        dl.run(0, LinearKind::fc1, dx, S, dy, nullptr);
        dl.run(0, LinearKind::fc2, dx, S, dy, nullptr);
    }, 2);
    const std::uint64_t vtp = vt_pitch_for(S);
    auto* dq = dput(rand_bf16(static_cast<std::size_t>(S) * kInner, rng, 0.3F));
    auto* dk = dput(rand_bf16(static_cast<std::size_t>(S) * kInner, rng, 0.3F));
    auto* dvt = dput(rand_bf16(static_cast<std::size_t>(kInner) * vtp, rng));
    auto* dao = dalloc<std::uint16_t>(static_cast<std::size_t>(S) * kInner);
    const double t_att = gpu_ms([&] { fa.run(dq, dk, dvt, vtp, S, S, true, dao, nullptr); }, 2);
    std::printf("  info  isolated: 4 linears %.1f ms, attention %.1f ms\n", t_lin, t_att);
    for (void* q : {static_cast<void*>(dx), static_cast<void*>(dy), static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dvt),
             static_cast<void*>(dao)})
        hc(hipFree(q), "f");
}

// One block at real row counts (synthetic inputs, block 0 weights resident): driver block time, per-op breakdown (the
// four linears and the attention timed alone; glue = remainder), host preparation, final layer, sampler step, refiner.
void test_shape() {
    std::puts("[shape: one block at real row counts, synthetic inputs]");
    Model& md = model();
    std::mt19937 rng(77);
    struct Shape {
        const char* name;
        std::uint32_t text, lt, lh, lw, at;
    };
    const Shape shapes[] = {{"~19K (half clip)", 256, 18, 48, 84, 100}, {"~38K (5 s 768x1344)", 256, 37, 48, 84, 207}};
    for (const Shape& sh : shapes) {
        const PackedLayout L = PackedLayout::build(sh.text, sh.lt, sh.lh, sh.lw, sh.at);
        const std::uint32_t S = L.seq_len;
        std::printf(" shape %s: S = %u rows (text %u, audio %u, video %u)\n", sh.name, S, sh.text, sh.at * 2, sh.lt * (sh.lh / 2) * (sh.lw / 2));
        const ForwardPlan p = ForwardPlan::build(L, 0.9F, {});
        StreamInputs in;
        in.T = sh.lt;
        in.H = sh.lh;
        in.W = sh.lw;
        in.audio_t = sh.at;
        in.video_x = rand_bf16(static_cast<std::size_t>(24) * sh.lt * sh.lh * sh.lw, rng);
        in.audio_x = rand_bf16(static_cast<std::size_t>(32) * 2 * sh.at, rng);
        in.text = rand_bf16(static_cast<std::size_t>(sh.text) * kHidden, rng);
        DeviceForward dev(md.g);
        GemmDeviceLinear dl(S);
        FlashDeviceAttention fa;
        std::vector<double> ms;
        DeviceForwardTaps tt;
        tt.stop_after_block = 0;
        tt.block_ms = &ms;
        for (int rep = 0; rep < 3; ++rep) dev.run(md.blocks, L, p, in, dl, fa, &tt);
        std::printf("  info  S=%u block (glue + gemm + flash, synchronized): %.1f / %.1f / %.1f ms; host prep %.1f ms; flushed %llu\n", S,
            ms[0], ms[1], ms[2], tt.host_prep_ms, static_cast<unsigned long long>(tt.flushed_groups));
        std::printf("  info  S=%u attention contract (measured, random inputs): %.1f\n", S, tt.attn_score_max);
        std::printf("  info  S=%u device patch projections %.1f ms\n", S, tt.embed_proj_ms);
        // per-op breakdown on device buffers
        auto* dx = dput(rand_bf16(static_cast<std::size_t>(S) * kFfn, rng, 0.5F));
        auto* dy = dalloc<std::uint16_t>(static_cast<std::size_t>(S) * 3 * kInner);
        const double t_qkv = gpu_ms([&] { dl.run(0, LinearKind::qkv, dx, S, dy, nullptr); });
        const double t_out = gpu_ms([&] { dl.run(0, LinearKind::out, dx, S, dy, nullptr); });
        const double t_fc1 = gpu_ms([&] { dl.run(0, LinearKind::fc1, dx, S, dy, nullptr); });
        const double t_fc2 = gpu_ms([&] { dl.run(0, LinearKind::fc2, dx, S, dy, nullptr); });
        const std::uint64_t vtp = vt_pitch_for(S);
        auto* dq = dput(rand_bf16(static_cast<std::size_t>(S) * kInner, rng, 0.3F));
        auto* dk = dput(rand_bf16(static_cast<std::size_t>(S) * kInner, rng, 0.3F));
        auto* dvt = dput(rand_bf16(static_cast<std::size_t>(kInner) * vtp, rng));
        auto* dao = dalloc<std::uint16_t>(static_cast<std::size_t>(S) * kInner);
        const double t_att = gpu_ms([&] { fa.run(dq, dk, dvt, vtp, S, S, true, dao, nullptr); }, 2);
        {
            // K4 cost with / without the attention-contract norm maxima
            auto* dw = dput(rand_bf16(kHeadDim, rng));
            auto* dnm = dalloc<std::uint32_t>(2 * kHeads);
            hc(hipMemset(dnm, 0, 2 * kHeads * 4), "nm");
            const double k4a = gpu_ms([&] { gpu::launch_qk_norm_rope(dy, S, dw, dw, nullptr, nullptr, kNormEps, dq, dk, nullptr); });
            const double k4b = gpu_ms([&] { gpu::launch_qk_norm_rope(dy, S, dw, dw, nullptr, nullptr, kNormEps, dq, dk, nullptr, dnm); });
            std::printf("  info  S=%u K4 qk_norm_rope %.2f ms, with the contract maxima %.2f ms\n", S, k4a, k4b);
            hc(hipFree(dw), "f");
            hc(hipFree(dnm), "f");
        }
        const double lin = t_qkv + t_out + t_fc1 + t_fc2;
        const double flops_lin = 2.0 * S * 385.4e6, flops_att = 4.0 * S * static_cast<double>(S) * kInner;
        std::printf("  info  breakdown: qkv %.1f, out %.1f, fc1+swiglu %.1f, fc2 %.1f (linears %.1f ms = %.1f TOPS), attention %.1f ms (%.1f TF),"
                    " glue+adaLN %.1f ms (%.1f %%)\n",
            t_qkv, t_out, t_fc1, t_fc2, lin, flops_lin / lin / 1e9, t_att, flops_att / t_att / 1e9, ms[2] - lin - t_att,
            100.0 * (ms[2] - lin - t_att) / ms[2]);
        const double fwd_est = 50.0 * ms[2];
        std::printf("  info  forward estimate 50 x block = %.2f s; host prep %.3f %% of it\n", fwd_est / 1e3, 100.0 * tt.host_prep_ms / fwd_est);
        for (void* q : {static_cast<void*>(dx), static_cast<void*>(dy), static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dvt),
                 static_cast<void*>(dao)})
            hc(hipFree(q), "f");
        // sampler step on the host at this latent size
        AvState x{rand_f32(in.video_x.size(), rng), rand_f32(in.audio_x.size(), rng)};
        AvState den{rand_f32(in.video_x.size(), rng), rand_f32(in.audio_x.size(), rng)};
        const std::vector<float> sig = {0.9F, 0.8F, 0.7F};
        int calls = 0;
        AvModelFn fake = [&](const AvState& xi, float sigma) {
            ++calls;
            (void)sigma;
            AvState o = xi;
            for (std::size_t i = 0; i < o.video.size(); ++i) o.video[i] = den.video[i];
            for (std::size_t i = 0; i < o.audio.size(); ++i) o.audio[i] = den.audio[i];
            return o;
        };
        const double ts0 = now_s();
        const AvState fin = sample(fake, x, sig, SamplerKind::res_multistep);
        const double tsamp = (now_s() - ts0) * 1e3 / 2.0;
        std::printf("  info  sampler step (host, res_multistep, %zu latent values incl. the fake model copy): %.1f ms = %.3f %% of a forward\n",
            fin.video.size() + fin.audio.size(), tsamp, 100.0 * tsamp / fwd_est);
        // refiner at two text lengths (once per sampling run)
        for (std::uint32_t Lt : {256U, 1024U}) {
            const auto txt = rand_bf16(static_cast<std::size_t>(Lt) * kTextDim, rng);
            dev.refine_text(txt.data(), Lt, fa);
            const double t0 = now_s();
            dev.refine_text(txt.data(), Lt, fa);
            const double tr = (now_s() - t0) * 1e3;
            std::printf("  info  device refiner L=%u: %.0f ms = %.2f %% of one forward\n", Lt, tr, 100.0 * tr / fwd_est);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, bool> want;
    for (int i = 1; i < argc; ++i) want[argv[i]] = true;
    if (want.empty()) want["kernels"] = true;
    {
        bool needs = false;
        for (const auto& kv : want) needs |= kv.second && kv.first != "kernels";
        if (needs && !assets_present(true)) {
            std::printf("SKIPPED: forward / refiner / mid / g2flash / shape / prof need H3_FIXTURES (with FORMAT.md) and H3_CKPT\n");
            return kSkipped;
        }
    }
    try {
        if (want["kernels"]) test_kernels();
        if (want["forward"]) test_forward();
        if (want["refiner"]) test_refiner();
        if (want["mid"]) test_mid();
        if (want["g2flash"]) test_g2flash();
        if (want["shape"]) test_shape();
        if (want["prof"]) test_prof();
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
