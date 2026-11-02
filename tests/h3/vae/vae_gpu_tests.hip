// GPU qualification of the H3 ViT3D video decoder against the seeded python fixtures (gfx1151).
// usage: strix-h3-vae-gpu-tests [weights.safetensors] [fixture root] [--real]
//   (defaults: H3_VAE_WEIGHTS, H3_VAE_FIXTURES). Qualification: missing weights or any fixture is a FAILURE.
// Gates: layer 0 / 36 layers / decoder-call output vs fixture, frames PSNR >= 65 dB vs the python FP16 decoder on
// t7 (1 tile), tiles (2 x 2 tiles x 2 chunks), odd (9 x 13 latents, one non-square partial tile), oddtiles
// (21 x 19 latents, 2 x 2 tiles with 176 / 208 px overlaps); all metrics finite; tiled GEMM == simple GEMM bitwise;
// determinism; shrinking-shape call sequence bitwise; attention NaN/Inf sanitization; decoder destruction
// returns its device memory; call-range partition (2 ranks, sequential on this device) == 1 rank bitwise.
// --real: one random 37 x 48 x 84 latent (5 s 768x1344): finite frames, 2-rank partition == 1 rank bitwise,
// frames hash printed. (A true two-process / two-node equality check belongs to the distribution package.)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

#include "h3/vae/vae_decoder.hpp"
#include "h3/vae/vae_kernels.hpp"
#include "h3/vae/vae_shard.hpp"
#include "h3/vae/vae_threads.hpp"

using namespace strix::h3::vae;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
void report(const char* name, const Diff& d) {
    std::printf("  %-26s n=%zu mismatch=%zu nonfinite=%zu max_abs=%.4g rel_l2=%.4g psnr(1)=%.2f dB\n", name, d.n, d.mismatches,
                d.nonfinite, d.max_abs, d.rel_l2(), d.psnr(1.0));
    std::fflush(stdout);
}
H16 z16_of(const Fixture& fx) {
    const auto zf = fx.f32("z_norm_f32");
    H16 z(zf.size());
    for (std::size_t i = 0; i < zf.size(); ++i) z[i] = f2h(zf[i]);  // the host wrapper casts to FP16
    return z;
}
bool all_finite(const std::vector<float>& v) {
    for (float f : v) if (!std::isfinite(f)) return false;
    return true;
}
std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* p = reinterpret_cast<const unsigned char*>(v.data());
    for (std::size_t i = 0; i < v.size() * 4; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
std::size_t free_bytes() {
    std::size_t f = 0, t = 0;
    (void)hipDeviceSynchronize();
    (void)hipMemGetInfo(&f, &t);
    return f;
}

// Frames of one fixture case vs the python frames; also the 2-rank partition when the plan has > 1 call.
void frames_case(GpuDecoder& dec, const std::string& root, const char* fcase) {
    const std::string dir = root + "/" + fcase;
    if (!file_exists(dir + "/manifest.json")) { check(false, std::string(fcase) + ": fixture present"); return; }
    const Fixture fx(dir);
    const auto& zi = fx.info("z_norm_f32");
    const int T = int(zi.shape[2]), H = int(zi.shape[3]), W = int(zi.shape[4]);
    const H16 z = z16_of(fx);
    double sec = 0;
    const auto frames = dec.decode(z, T, H, W, &sec);
    const auto ref = fx.f32("frames_f32");
    const Diff d = diff_f32(frames, ref);
    report((std::string(fcase) + " frames").c_str(), d);
    const DecodePlan p = make_plan(T, H, W);
    std::printf("  %s decode (%d calls) %.2f s\n", fcase, p.calls(), sec);
    check(d.valid() && d.psnr(1.0) >= 65.0, std::string(fcase) + ": frames PSNR >= 65 dB vs python decoder, all finite");
    if (p.calls() > 1) {
        const H16 zp = denormalize_and_pad(p, z, dec.head().latents_mean.data(), dec.head().latents_std.data());
        auto r0 = dec.decode_calls(p, zp, partition(p, 0, 2));
        auto r1 = dec.decode_calls(p, zp, partition(p, 1, 2));
        r0.insert(r0.end(), r1.begin(), r1.end());
        check(diff_f32(blend_calls(p, r0), frames).bitwise(), std::string(fcase) + ": 2-rank call partition == 1 rank (bitwise)");
        if (p.num_chunks > 1) {  // chunk split (the two-node entry point's unit): canvases of both halves -> frames
            auto c0 = dec.decode_chunks(p, zp, {0, p.num_chunks / 2});
            auto c1 = dec.decode_chunks(p, zp, {p.num_chunks / 2, p.num_chunks});
            for (auto& c : c1) c0.push_back(std::move(c));
            check(diff_f32(finish_frames(p, std::move(c0)), frames).bitwise(), std::string(fcase) + ": 2-rank chunk split == 1 rank (bitwise)");
        }
        // two-node gather at a mid-chunk cut (rank 1's pieces packed / unpacked, pushed first) == 1 rank
        const int cut = p.calls() / 2 - (p.calls() > 3 ? 1 : 0);
        FrameAssembler fa(p);
        std::vector<std::vector<std::uint8_t>> wire;
        dec.decode_pieces(p, zp, split_calls(p, 1, 2, cut - p.calls() / 2), [&](DecodePiece&& x) { wire.push_back(pack_piece(x)); });
        for (const auto& b : wire) fa.push(unpack_piece(p, b.data(), b.size()));
        dec.decode_pieces(p, zp, split_calls(p, 0, 2, cut - p.calls() / 2), [&](DecodePiece&& x) { fa.push(std::move(x)); });
        check(fa.complete() && diff_f32(fa.take(), frames).bitwise(), std::string(fcase) + ": 2-rank gather at call " + std::to_string(cut) + " == 1 rank (bitwise)");
    }
}

// Attention NaN / Inf sanitization: one head-shaped problem with a V row of +Inf (output +Inf -> 65504) and one of NaN
// in another head (output NaN -> 0).
void test_sanitize() {
    const unsigned L = 70, Lp = 128;
    std::vector<_Float16> q(32U * Lp * 64U, _Float16(0.0F)), k = q, v = q;
    for (unsigned h = 0; h < 32U; ++h)
        for (unsigned t = 0; t < L; ++t)
            for (unsigned d = 0; d < 64U; ++d) {
                q[(h * Lp + t) * 64U + d] = _Float16(0.01F * float((t + d) % 7));
                k[(h * Lp + t) * 64U + d] = _Float16(0.02F * float((t * 3 + d) % 5));
                v[(h * Lp + t) * 64U + d] = _Float16(0.1F * float((t + 2 * d) % 9) - 0.4F);
            }
    for (unsigned d = 0; d < 64U; ++d) {
        v[(0U * Lp + 5U) * 64U + d] = _Float16(INFINITY);
        v[(1U * Lp + 9U) * 64U + d] = _Float16(NAN);
    }
    _Float16 *dq, *dk, *dv, *dout;
    (void)hipMalloc(&dq, q.size() * 2); (void)hipMalloc(&dk, k.size() * 2); (void)hipMalloc(&dv, v.size() * 2);
    (void)hipMalloc(&dout, std::size_t(L) * 2048U * 2U);
    (void)hipMemcpy(dq, q.data(), q.size() * 2, hipMemcpyHostToDevice);
    (void)hipMemcpy(dk, k.data(), k.size() * 2, hipMemcpyHostToDevice);
    (void)hipMemcpy(dv, v.data(), v.size() * 2, hipMemcpyHostToDevice);
    gpu::attention(dq, dk, dv, dout, L, Lp, nullptr);
    std::vector<std::uint16_t> o(std::size_t(L) * 2048U);
    (void)hipMemcpy(o.data(), dout, o.size() * 2, hipMemcpyDeviceToHost);
    (void)hipFree(dq); (void)hipFree(dk); (void)hipFree(dv); (void)hipFree(dout);
    bool head0 = true, head1 = true, finite = true;
    for (unsigned t = 0; t < L; ++t)
        for (unsigned d = 0; d < 64U; ++d) {
            head0 &= o[std::size_t(t) * 2048U + d] == 0x7bffU;               // +Inf -> +65504
            head1 &= o[std::size_t(t) * 2048U + 64U + d] == 0x0000U;         // NaN -> 0
            for (unsigned h = 0; h < 32U; ++h) finite &= (o[std::size_t(t) * 2048U + 64U * h + d] & 0x7c00U) != 0x7c00U;
        }
    check(head0 && head1 && finite, "attention output sanitized (NaN -> 0, +Inf -> 65504, no non-finite stored)");
}

// Padding rows [L, Lp) of Q / K / V must not influence any output bit: zero padding vs garbage padding (NaN, +-Inf,
// large values) give the same output (masked keys are staged as zeros, never loaded).
void test_padding_garbage() {
    for (const unsigned L : {70U, 129U, 1797U}) {
        const unsigned Lp = (L + 63U) / 64U * 64U + 64U;  // at least one whole padded key block
        std::mt19937 rng(L);
        std::normal_distribution<float> nd;
        std::vector<_Float16> q(std::size_t(32) * Lp * 64U, _Float16(0.0F)), k = q, v = q;
        for (unsigned h = 0; h < 32U; ++h)
            for (unsigned t = 0; t < L; ++t)
                for (unsigned d = 0; d < 64U; ++d) {
                    const std::size_t i = (std::size_t(h) * Lp + t) * 64U + d;
                    q[i] = _Float16(nd(rng)); k[i] = _Float16(nd(rng)); v[i] = _Float16(nd(rng));
                }
        std::vector<_Float16> qg = q, kg = k, vg = v;
        const _Float16 junk[4] = {_Float16(NAN), _Float16(INFINITY), _Float16(-INFINITY), _Float16(60000.0F)};
        for (unsigned h = 0; h < 32U; ++h)
            for (unsigned t = L; t < Lp; ++t)
                for (unsigned d = 0; d < 64U; ++d) {
                    const std::size_t i = (std::size_t(h) * Lp + t) * 64U + d;
                    qg[i] = junk[(t + d) % 4U]; kg[i] = junk[(t + 2U * d + 1U) % 4U]; vg[i] = junk[(3U * t + d + 2U) % 4U];
                }
        const auto run = [&](const std::vector<_Float16>& a, const std::vector<_Float16>& b, const std::vector<_Float16>& c) {
            _Float16 *dq, *dk, *dv, *dout;
            (void)hipMalloc(&dq, a.size() * 2); (void)hipMalloc(&dk, b.size() * 2); (void)hipMalloc(&dv, c.size() * 2);
            (void)hipMalloc(&dout, std::size_t(L) * 2048U * 2U);
            (void)hipMemcpy(dq, a.data(), a.size() * 2, hipMemcpyHostToDevice);
            (void)hipMemcpy(dk, b.data(), b.size() * 2, hipMemcpyHostToDevice);
            (void)hipMemcpy(dv, c.data(), c.size() * 2, hipMemcpyHostToDevice);
            gpu::attention(dq, dk, dv, dout, L, Lp, nullptr);
            std::vector<std::uint16_t> o(std::size_t(L) * 2048U);
            (void)hipMemcpy(o.data(), dout, o.size() * 2, hipMemcpyDeviceToHost);
            (void)hipFree(dq); (void)hipFree(dk); (void)hipFree(dv); (void)hipFree(dout);
            return o;
        };
        const auto clean = run(q, k, v), dirty = run(qg, kg, vg);
        bool finite = true;
        for (auto x : clean) finite &= (x & 0x7c00U) != 0x7c00U;
        check(finite && clean == dirty, "attention L=" + std::to_string(L) + ": garbage (NaN/Inf/large) padding rows == zero padding (bitwise)");
    }
}

// GEMM qualification + tile bench at the decoder / encoder shapes (M = 1797): for every epilogue (plain, residual
// addcmul, residual add, fused w1 + gated SiLU) and every block tile, the output must equal the simple (old) kernel's
// output bitwise; for the fused form the reference is the old two-step path (simple GEMM on the unpermuted weights,
// then the gated SiLU kernel). Residual outputs start from the same initial matrix for every run.
void gemm_bench() {
    struct Shape { const char* name; unsigned N, K, mode; };  // mode 0 plain, 1 residual addcmul, 2 residual add, 3 swiglu
    // enc-k9216 / enc-k9216-add: the encoder's 1024-channel 3x3 convs (K = 9216, routed to the 256 x 256 tile)
    const Shape shapes[7] = {{"qkv", 6144, 2048, 0}, {"out", 2048, 2048, 1}, {"w1+swiglu", 16384, 2048, 3}, {"w2", 2048, 8192, 1},
                             {"enc-add", 256, 1152, 2}, {"enc-k9216", 1024, 9216, 0}, {"enc-k9216-add", 1024, 9216, 2}};
    const unsigned M = 1797;
    std::mt19937 rng(9);
    std::normal_distribution<float> nd;
    for (const Shape& sh : shapes) {
        std::vector<_Float16> a(std::size_t(M) * sh.K), w(std::size_t(sh.N) * sh.K), b(sh.N), sc(sh.N);
        for (auto& v : a) v = _Float16(nd(rng));
        for (auto& v : w) v = _Float16(0.02F * nd(rng));
        for (auto& v : b) v = _Float16(0.1F * nd(rng));
        for (auto& v : sc) v = _Float16(0.5F + 0.1F * nd(rng));
        const unsigned outN = sh.mode == 3U ? sh.N / 2U : sh.N;
        std::vector<_Float16> init(std::size_t(M) * outN), resid(std::size_t(M) * outN);
        for (auto& v : init) v = _Float16(nd(rng));
        for (auto& v : resid) v = _Float16(nd(rng));
        _Float16 *da, *dw, *db, *ds, *dc, *dr, *dwu = nullptr, *dbu = nullptr, *dh = nullptr;
        (void)hipMalloc(&da, a.size() * 2); (void)hipMalloc(&dw, w.size() * 2); (void)hipMalloc(&db, b.size() * 2);
        (void)hipMalloc(&ds, sc.size() * 2); (void)hipMalloc(&dc, init.size() * 2); (void)hipMalloc(&dr, resid.size() * 2);
        (void)hipMemcpy(da, a.data(), a.size() * 2, hipMemcpyHostToDevice); (void)hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice);
        (void)hipMemcpy(db, b.data(), b.size() * 2, hipMemcpyHostToDevice); (void)hipMemcpy(ds, sc.data(), sc.size() * 2, hipMemcpyHostToDevice);
        (void)hipMemcpy(dr, resid.data(), resid.size() * 2, hipMemcpyHostToDevice);
        if (sh.mode == 3U) {  // w is the permuted (fused) weight: build the unpermuted one for the two-step reference
            std::vector<_Float16> wu(w.size()), bu(b.size());
            for (unsigned p = 0; p < sh.N; ++p) {
                const unsigned src = gpu::swiglu_source_row(p, sh.N);
                std::memcpy(wu.data() + std::size_t(src) * sh.K, w.data() + std::size_t(p) * sh.K, std::size_t(sh.K) * 2);
                bu[src] = b[p];
            }
            (void)hipMalloc(&dwu, wu.size() * 2); (void)hipMalloc(&dbu, bu.size() * 2); (void)hipMalloc(&dh, std::size_t(M) * sh.N * 2);
            (void)hipMemcpy(dwu, wu.data(), wu.size() * 2, hipMemcpyHostToDevice); (void)hipMemcpy(dbu, bu.data(), bu.size() * 2, hipMemcpyHostToDevice);
        }
        const auto run = [&] {
            if (sh.mode == 0U) gpu::gemm(da, sh.K, dw, db, dc, outN, M, sh.N, sh.K, nullptr);
            else if (sh.mode == 1U) gpu::gemm_residual(da, sh.K, dw, db, ds, dc, outN, M, sh.N, sh.K, nullptr);
            else if (sh.mode == 2U) gpu::gemm_add(da, sh.K, dw, db, dr, dc, outN, M, sh.N, sh.K, nullptr);
            else gpu::gemm_swiglu(da, sh.K, dw, db, dc, outN, M, sh.N, sh.K, nullptr);
        };
        const auto fetch = [&] {
            std::vector<std::uint16_t> out(init.size());
            (void)hipMemcpy(out.data(), dc, out.size() * 2, hipMemcpyDeviceToHost);
            return out;
        };
        const auto reset = [&] { (void)hipMemcpy(dc, init.data(), init.size() * 2, hipMemcpyHostToDevice); };
        // reference: the simple kernel (and, for the fused form, the old two-step path)
        gpu::set_gemm_tiled(0);
        reset();
        if (sh.mode == 3U) {
            gpu::gemm(da, sh.K, dwu, dbu, dh, sh.N, M, sh.N, sh.K, nullptr);
            gpu::swiglu(dh, dc, M, nullptr);
        } else run();
        const std::vector<std::uint16_t> ref = fetch();
        gpu::set_gemm_tiled(-1);
        bool finite = true;
        for (auto v : ref) finite &= (v & 0x7c00U) != 0x7c00U;
        for (int tile = 0; tile < 4; ++tile) {
            gpu::set_gemm_tile(tile);
            reset();
            run();
            const std::vector<std::uint16_t> out = fetch();
            hipEvent_t e0, e1;
            (void)hipEventCreate(&e0); (void)hipEventCreate(&e1);
            (void)hipEventRecord(e0, nullptr);
            for (int r = 0; r < 10; ++r) run();
            (void)hipEventRecord(e1, nullptr);
            (void)hipEventSynchronize(e1);
            float ms = 0;
            (void)hipEventElapsedTime(&ms, e0, e1);
            (void)hipEventDestroy(e0); (void)hipEventDestroy(e1);
            const double tf = 2.0 * M * sh.N * sh.K / (ms / 10.0 * 1e-3) / 1e12;
            std::printf("  gemm %-10s tile %d: %.3f ms, %.1f TF/s\n", sh.name, tile, ms / 10.0, tf);
            check(finite && out == ref, std::string("gemm ") + sh.name + " tile " + std::to_string(tile) +
                                           (sh.mode == 3U ? " == old gemm + swiglu (bitwise)" : " == simple kernel (bitwise)"));
        }
        gpu::set_gemm_tile(-1);
        (void)hipFree(da); (void)hipFree(dw); (void)hipFree(db); (void)hipFree(ds); (void)hipFree(dc); (void)hipFree(dr);
        if (dwu) { (void)hipFree(dwu); (void)hipFree(dbu); (void)hipFree(dh); }
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::string weights = std::getenv("H3_VAE_WEIGHTS") ? std::getenv("H3_VAE_WEIGHTS") : "";
    std::string root = std::getenv("H3_VAE_FIXTURES") ? std::getenv("H3_VAE_FIXTURES") : "";
    bool real = false, bench_only = false;
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--real")) real = true;
        else if (!std::strcmp(argv[i], "--gemm-bench")) bench_only = true;
        else if (pos == 0) { weights = argv[i]; ++pos; }
        else if (pos == 1) { root = argv[i]; ++pos; }
    }
    if (weights.empty() || root.empty() || !file_exists(weights) || !file_exists(root + "/t7/manifest.json")) {
        std::printf("FAIL qualification needs the weights and fixtures (args or H3_VAE_WEIGHTS / H3_VAE_FIXTURES)\nFAILED (1 failures)\n");
        return 1;
    }
    if (bench_only) {
        gemm_bench();
        std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
        return failures ? 1 : 0;
    }
    {   // device memory is returned on destruction: a warm-up cycle first (runtime / code-object state that loads on
        // first use and stays for the process), then the measured construct + destroy cycle must leave nothing behind
        { GpuDecoder warm(weights, 2); }
        const std::size_t before = free_bytes();
        { GpuDecoder tmp(weights, 2); }
        const std::size_t after = free_bytes();
        const double leak = double(before) - double(after);
        std::printf("  free before %.1f MiB, after construct+destroy %.1f MiB\n", before / 1048576.0, after / 1048576.0);
        check(leak < 64.0 * 1048576.0, "decoder destruction returns its device memory (< 64 MiB difference)");
    }
    const auto tl0 = std::chrono::steady_clock::now();
    GpuDecoder dec(weights);
    std::printf("weights loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tl0).count());
    {   // inputs are validated before any allocation or indexing
        const auto throws = [](auto&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
        const H16 z7(std::size_t(24) * 7 * 16 * 16, 0);
        check(throws([&] { (void)dec.decode(H16(z7.begin(), z7.end() - 1), 7, 16, 16); }) &&   // truncated latent
              throws([&] { (void)dec.decode(H16{}, 7, 16, 16); }) &&                           // empty latent
              throws([&] { (void)dec.decode(H16(std::size_t(24) * 16 * 16, 0), 1, 16, 16); }) &&  // T = 1 (not covered)
              throws([&] { (void)dec.decode(H16{}, 7, 0, 16); }) &&                            // zero side
              throws([&] { (void)dec.decode_call(H16(10, 0), 7, 16, 16); }) &&                 // call input length
              throws([&] { (void)dec.decode_call(H16(std::size_t(24) * 8 * 16 * 16, 0), 8, 16, 16); }),  // call too large
              "video decoder rejects empty / truncated latents, T < 2, zero sides, bad call inputs");
    }
    test_sanitize();
    test_padding_garbage();
    gemm_bench();  // every epilogue x tile vs the simple kernel / the old two-step w1 path, bitwise

    const Fixture t7(root + "/t7");
    const int nt = 7, ny = 16, nx = 16;
    {
        const H16 hin = t7.f16("tap_h_in");
        H16 h1, h36;
        dec.decode_call(H16{}, nt, ny, nx, nullptr, &h1, 1, &hin);
        const Diff d1 = diff_f16(h1, t7.f16("tap_h_after_l0"));
        report("layer 0 residual", d1);
        check(d1.valid() && d1.rel_l2() < 5e-4, "layer 0 vs fixture (rel_l2 < 5e-4; measured 9.9e-5)");
        dec.decode_call(H16{}, nt, ny, nx, nullptr, &h36, -1, &hin);
        const Diff d36 = diff_f16(h36, t7.f16("tap_h_after_l35"));
        report("after 36 layers", d36);
        check(d36.valid() && d36.rel_l2() < 3e-3, "36 layers vs fixture (rel_l2 < 3e-3; measured 1.25e-3)");
    }
    {
        CallTiming tm;
        const H16 out = dec.decode_call(t7.f16("tap_dec_in"), nt, ny, nx, &tm);
        const Diff d = diff_f16(out, t7.f16("tap_dec_out"));
        report("decoder call output", d);
        check(d.valid() && d.psnr(1.0) >= 52.0, "decoder call output PSNR >= 52 dB (measured 57.5)");
        std::printf("  call 7x16x16: embed %.2f ms, 36 layers %.1f ms, head %.2f ms, wall %.1f ms\n", tm.embed_ms, tm.layers_ms, tm.head_ms, tm.total_ms);
        gpu::set_gemm_tiled(0);
        CallTiming ts;
        const H16 simple = dec.decode_call(t7.f16("tap_dec_in"), nt, ny, nx, &ts);
        gpu::set_gemm_tiled(-1);
        std::printf("  simple GEMM form: 36 layers %.1f ms\n", ts.layers_ms);
        check(diff_f16(simple, out).bitwise(), "tiled GEMM == simple GEMM (decoder call output bitwise)");
        {
            GpuDecoder unfused(weights, kLayers36, false);
            CallTiming tu;
            gpu::set_gemm_tiled(0);  // the complete old path: simple GEMM for every linear + separate gated SiLU
            const H16 two_step = unfused.decode_call(t7.f16("tap_dec_in"), nt, ny, nx, &tu);
            gpu::set_gemm_tiled(-1);
            std::printf("  unfused w1 + swiglu: 36 layers %.1f ms\n", tu.layers_ms);
            check(diff_f16(two_step, out).bitwise(), "default path (tiled, fused w1+swiglu) == old path (simple GEMM, separate swiglu): decoder call output bitwise");
        }
        // shrinking / growing shape sequence: a small call after a full one gives the bits of a small call alone
        const DecodePlan po = make_plan(7, 9, 13);
        const auto calls = decoder_calls(po);
        H16 zsmall(std::size_t(24) * 7 * 9 * 13);
        std::mt19937 rng(11);
        std::normal_distribution<float> nd;
        for (auto& v : zsmall) v = f2h(nd(rng));
        const H16 zp = denormalize_and_pad(po, zsmall, dec.head().latents_mean.data(), dec.head().latents_std.data());
        const H16 in_small = call_input(dec.head(), po, zp, calls[0]);
        GpuDecoder fresh(weights, 4);  // small call on a decoder that never saw a larger shape
        const H16 a = fresh.decode_call(in_small, 7, 9, 13, nullptr, nullptr, 4);
        (void)fresh.decode_call(t7.f16("tap_dec_in"), nt, ny, nx, nullptr, nullptr, 4);
        const H16 b = fresh.decode_call(in_small, 7, 9, 13, nullptr, nullptr, 4);
        check(diff_f16(a, b).bitwise(), "small call after a full call == small call alone (bitwise; padding rewritten per call)");
        for (int rep = 0; rep < 2; ++rep) {
            dec.decode_call(t7.f16("tap_dec_in"), nt, ny, nx, &tm);
            std::printf("  call repeat %d: 36 layers %.1f ms, wall %.1f ms\n", rep, tm.layers_ms, tm.total_ms);
        }
    }
    {
        const auto f1 = dec.decode(z16_of(t7), 7, 16, 16);
        const auto f2 = dec.decode(z16_of(t7), 7, 16, 16);
        check(diff_f32(f1, f2).bitwise(), "t7 decode deterministic (two runs bitwise, finite)");
        if (file_exists(root + "/t7f32/manifest.json")) {
            const Fixture f32(root + "/t7f32");
            report("t7 frames vs fp32 python", diff_f32(f1, f32.f32("frames_f32")));
            report("python fp16 vs python fp32", diff_f32(t7.f32("frames_f32"), f32.f32("frames_f32")));
        }
    }
    for (const char* c : {"t7", "tiles", "odd", "oddtiles"}) frames_case(dec, root, c);
    {   // a failure in any worker or in the GPU loop surfaces as an exception (no terminate), the decoder stays usable
        const Fixture tf(root + "/tiles");
        const H16 z = z16_of(tf);
        const auto clean = dec.decode(z, 12, 20, 20);
        const std::pair<FaultStage, const char*> stages[4] = {{FaultStage::call_input, "call input worker"},
            {FaultStage::unpatchify, "unpatchify worker"}, {FaultStage::gpu_loop, "GPU loop"}, {FaultStage::chunk_assembly, "chunk assembly worker"}};
        for (const auto& [stage, name] : stages) {
            inject_failure(stage, stage == FaultStage::chunk_assembly ? 1 : 3);
            bool thrown = false;
            try { (void)dec.decode(z, 12, 20, 20); } catch (const std::runtime_error&) { thrown = true; }
            inject_failure(stage, -1);
            const auto after = dec.decode(z, 12, 20, 20);
            check(thrown && diff_f32(after, clean).bitwise(), std::string("injected failure in the ") + name + " surfaces as an exception; next decode bitwise");
        }
    }
    if (real) {  // 5 s 768x1344: 37 x 48 x 84 latents, 196 calls
        std::mt19937 rng(7);
        std::normal_distribution<float> nd;
        H16 z(std::size_t(24) * 37 * 48 * 84);
        for (auto& v : z) v = f2h(nd(rng));
        double sec = 0;
        const auto frames = dec.decode(z, 37, 48, 84, &sec);
        std::printf("  real 37x48x84 decode: %.1f s (196 calls, one node), frames hash %016llx\n", sec, (unsigned long long)fnv(frames));
        check(frames.size() == std::size_t(3) * 124 * 768 * 1344 && all_finite(frames), "real shape: 124 finite frames of 768x1344");
        const DecodePlan p = make_plan(37, 48, 84);
        const H16 zp = denormalize_and_pad(p, z, dec.head().latents_mean.data(), dec.head().latents_std.data());
        // real shape: the two-node gather at the balanced cut (98 + 98 calls; rank 0 cuts chunk 3 after 14 of its 28
        // tiles). Sequential on this device: rank 1 first (its pieces packed as for the send), then rank 0 decoding into
        // the assembler, which finalizes its whole chunks while the GPU runs; rank 1's pieces are then pushed (in the
        // real run they arrive during rank 0's decode). Projection: max(rank 0, rank 1 + send) + work after the last piece.
        const int shift = std::getenv("H3_VAE_SPLIT_SHIFT") ? std::atoi(std::getenv("H3_VAE_SPLIT_SHIFT")) : 0;
        const CallRange g0 = split_calls(p, 0, 2, shift), g1 = split_calls(p, 1, 2, shift);
        std::vector<std::vector<std::uint8_t>> wire;
        double pack_s = 0;
        const auto t1 = std::chrono::steady_clock::now();
        dec.decode_pieces(p, zp, g1, [&](DecodePiece&& x) {
            const auto tp = std::chrono::steady_clock::now();
            wire.push_back(pack_piece(x));
            pack_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
        });
        const double s1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        std::size_t wire_bytes = 0;
        for (const auto& b : wire) wire_bytes += b.size();
        FrameAssembler fa(p);
        const auto t0 = std::chrono::steady_clock::now();
        dec.decode_pieces(p, zp, g0, [&](DecodePiece&& x) { fa.push(std::move(x)); });
        const double s0 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const int done0 = fa.chunks_done();
        double last_s = 0, push_s = 0;
        for (const auto& b : wire) {
            const auto tp = std::chrono::steady_clock::now();
            fa.push(unpack_piece(p, b.data(), b.size()));
            const double one = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
            last_s = std::max(last_s, one);
            push_s += one;
        }
        const auto cf = fa.take();
        std::printf("  gather: cut %d (shift %d): rank 0 %d calls %.1f s (%d chunks finalized during the decode), rank 1 %d calls %.1f s "
                    "(pack %.2f s, %zu pieces, %.0f MB), pushes of rank 1 pieces %.2f s (max one %.2f s), frames hash %016llx\n",
                    g0.last, shift, g0.last - g0.first, s0, done0, g1.last - g1.first, s1, pack_s, wire.size(), double(wire_bytes) / 1e6,
                    push_s, last_s, (unsigned long long)fnv(cf));
        check(diff_f32(cf, frames).bitwise(), "real shape: 2-rank gather (balanced call cut, packed pieces) == 1 rank (bitwise)");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
