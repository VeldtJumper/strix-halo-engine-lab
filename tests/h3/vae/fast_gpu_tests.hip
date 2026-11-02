// GPU qualification of the FAST-class lever h3-vae-fast24-v1 (distilled 24-layer non-causal decoder).
// usage: strix-h3-fast-gpu-tests [fast weights] [fixture root] [--real]  (defaults H3_VAE_FAST_WEIGHTS, H3_VAE_FIXTURES)
// Gates: frames vs the python fast decoder (fixture fast, 24 x 7 x 16 x 16: 2 temporal blocks, shifted windows across
// the block boundary) PSNR >= 60 dB, finite, deterministic. Reported (lever quality, not gated here): PSNR of the fast
// decoder against the 36-layer reference decoder on the same latent (python fast vs python 36L, ours vs python 36L).
// --real: 37 x 48 x 84 (5 s 768x1344) decode timed.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

#include "h3/vae/fast_decoder.hpp"

using namespace strix::h3::vae;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
}  // namespace

int main(int argc, char** argv) {
    std::string weights = std::getenv("H3_VAE_FAST_WEIGHTS") ? std::getenv("H3_VAE_FAST_WEIGHTS") : "";
    std::string root = std::getenv("H3_VAE_FIXTURES") ? std::getenv("H3_VAE_FIXTURES") : "";
    bool real = false;
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--real")) real = true;
        else if (pos++ == 0) weights = argv[i];
        else root = argv[i];
    }
    if (weights.empty() || root.empty() || !file_exists(weights) || !file_exists(root + "/fast/manifest.json")) {
        std::printf("FAIL qualification needs the fast decoder weights and the fast fixture\nFAILED (1 failures)\n");
        return 1;
    }
    std::printf("lever identity %s (opt-in only)\n", fast::kIdentity);
    fast::GpuDecoder dec(weights);
    {
        bool r = false;
        try { (void)dec.decode(H16(10, 0), 7, 16, 16); } catch (const std::invalid_argument&) { r = true; }
        check(r, "fast decoder rejects a latent of the wrong length");
    }
    const Fixture fx(root + "/fast");
    const auto zf = fx.f32("z_norm_f32");
    H16 z(zf.size());
    for (std::size_t i = 0; i < zf.size(); ++i) z[i] = f2h(zf[i]);
    double sec = 0;
    const auto frames = dec.decode(z, 7, 16, 16, &sec);
    const Diff d = diff_f32(frames, fx.f32("frames_f32"));
    std::printf("  fast frames vs python fast: n=%zu max_abs=%.4g psnr=%.2f dB (%.2f s)\n", d.n, d.max_abs, d.psnr(1.0), sec);
    check(d.valid() && d.psnr(1.0) >= 60.0, "fast decoder frames PSNR >= 60 dB vs the python fast decoder, finite");
    check(diff_f32(dec.decode(z, 7, 16, 16), frames).bitwise(), "fast decoder deterministic (bitwise rerun)");
    if (file_exists(root + "/t7/manifest.json")) {  // same seeded latent through the 36-layer reference decoder
        const Fixture t7(root + "/t7");
        const auto ref36 = t7.f32("frames_f32");
        std::printf("  lever delta: python fast vs python 36L %.2f dB; ours fast vs python 36L %.2f dB\n",
                    diff_f32(fx.f32("frames_f32"), ref36).psnr(1.0), diff_f32(frames, ref36).psnr(1.0));
    }
    if (real) {
        std::mt19937 rng(7);
        std::normal_distribution<float> nd;
        H16 zr(std::size_t(24) * 37 * 48 * 84);
        for (auto& v : zr) v = f2h(nd(rng));
        double s = 0;
        const auto fr = dec.decode(zr, 37, 48, 84, &s);
        bool finite = fr.size() == std::size_t(3) * dec.frames(37) * 768 * 1344;
        for (float f : fr) finite &= std::isfinite(f);
        std::printf("  real 37x48x84 fast decode: %d frames, %.1f s (one node)\n", dec.frames(37), s);
        check(finite, "real shape: finite frames");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
