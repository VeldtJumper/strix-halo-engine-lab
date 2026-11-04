// GPU qualification of the H3 video VAE encoder (single-image path) against the seeded fixtures.
// usage: strix-h3-enc-gpu-tests [video weights] [fixture root]  (defaults H3_VAE_WEIGHTS, H3_VAE_FIXTURES);
// missing evidence is a FAILURE. Gates: enc_small (64 x 64) / enc1h (256 x 256) vs the FP16 python encoder SNR >= 50 dB;
// enc1 (256 x 256), enc_odd (176 x 240), enc_tiles (320 x 400, 2 x 2 tiles) vs the FP32 python encoder SNR >= 35 dB;
// all finite; deterministic; a 768 x 1344 keyframe timed (28 tiles).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>

#include "h3/vae/enc_gpu.hpp"

using namespace strix::h3::vae;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
// the shared validated metric: -inf on non-finite values or a length mismatch
double snr_db(const std::vector<float>& got, const std::vector<float>& ref) { return diff_f32(got, ref).snr_db(); }
}  // namespace

int main(int argc, char** argv) {
    std::string weights = std::getenv("H3_VAE_WEIGHTS") ? std::getenv("H3_VAE_WEIGHTS") : "";
    std::string root = std::getenv("H3_VAE_FIXTURES") ? std::getenv("H3_VAE_FIXTURES") : "";
    if (argc > 1) weights = argv[1];
    if (argc > 2) root = argv[2];
    if (weights.empty() || root.empty() || !file_exists(weights) || !file_exists(root + "/enc_small/manifest.json")) {
        std::printf("FAIL qualification needs the video weights and the encoder fixtures\nFAILED (1 failures)\n");
        return 1;
    }
    enc::GpuEncoder e(weights);
    {   // unsupported shapes and inconsistent buffers are rejected before any dispatch
        const auto rejects = [&](int H, int W, std::size_t n) {
            try { (void)e.encode(std::vector<float>(n, 0.0F), H, W); } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        enc::Map tiny;
        tiny.C = 3; tiny.H = 16; tiny.W = 64; tiny.v.assign(std::size_t(3) * 16 * 64, 0);
        bool tile_rejected = false;
        try { (void)e.moments(tiny); } catch (const std::invalid_argument&) { tile_rejected = true; }
        check(rejects(16, 64, std::size_t(3) * 16 * 64) && rejects(64, 16, std::size_t(3) * 64 * 16) && rejects(48, 40, std::size_t(3) * 48 * 40) &&
              rejects(64, 64, std::size_t(3) * 64 * 64 - 1) && tile_rejected,
              "encoder rejects sides < 32 or not multiples of 16, and buffer length mismatches");
    }
    for (const char* fcase : {"enc_small", "enc1h", "enc1", "enc_odd", "enc_tiles"}) {
        if (!file_exists(root + "/" + fcase + "/manifest.json")) { check(false, std::string(fcase) + ": fixture present"); continue; }
        const Fixture fx(root + "/" + fcase);
        const auto& xi = fx.info("x_f32");
        const int H = int(xi.shape[3]), W = int(xi.shape[4]);
        double sec = 0;
        const auto z = e.encode(fx.f32("x_f32"), H, W, &sec);
        const auto ref = fx.f32("latent_f32");
        const Diff d = diff_f32(z, ref);
        const double snr = snr_db(z, ref);
        const bool fp16_ref = std::string(fcase) == "enc_small" || std::string(fcase) == "enc1h";
        const double floor = fp16_ref ? 50.0 : 35.0;
        std::printf("  %s (%dx%d): latent n=%zu max_abs=%.3g snr=%.1f dB, %.2f s\n", fcase, H, W, d.n, d.max_abs, snr, sec);
        check(d.valid() && snr >= floor, std::string(fcase) + ": latent SNR >= " + std::to_string(int(floor)) + " dB vs python encoder, finite");
        const auto again = e.encode(fx.f32("x_f32"), H, W);
        check(diff_f32(again, z).bitwise(), std::string(fcase) + ": deterministic (bitwise rerun)");
    }
    // multi-frame path (references / control videos)
    for (const char* fcase : {"encv_small", "encv", "encv_tiles"}) {
        if (!file_exists(root + "/" + fcase + "/manifest.json")) { check(false, std::string(fcase) + ": fixture present"); continue; }
        const Fixture fx(root + "/" + fcase);
        const auto& xi = fx.info("x_f32");
        const int T = int(xi.shape[2]), H = int(xi.shape[3]), W = int(xi.shape[4]);
        double sec = 0;
        const auto z = e.encode_video(fx.f32("x_f32"), T, H, W, &sec);
        const Diff d = diff_f32(z, fx.f32("latent_f32"));
        const double floor = std::string(fcase) == "encv_small" ? 50.0 : 35.0;  // FP16 vs FP16 python / vs FP32 python
        std::printf("  %s (%dx%dx%d): latent n=%zu max_abs=%.3g snr=%.1f dB, %.2f s\n", fcase, T, H, W, d.n, d.max_abs, d.snr_db(), sec);
        check(d.valid() && d.snr_db() >= floor, std::string(fcase) + ": video latent SNR >= " + std::to_string(int(floor)) + " dB vs python encoder, finite");
        check(diff_f32(e.encode_video(fx.f32("x_f32"), T, H, W), z).bitwise(), std::string(fcase) + ": video encode deterministic (bitwise rerun)");
        enc::set_implicit_conv(0);  // the per-output-frame im2col + GEMM form
        double sec_old = 0;
        const auto z_old = e.encode_video(fx.f32("x_f32"), T, H, W, &sec_old);
        enc::set_implicit_conv(-1);
        std::printf("    per-frame form %.2f s\n", sec_old);
        check(diff_f32(z_old, z).bitwise(), std::string(fcase) + ": implicit-GEMM convs == per-frame im2col + GEMM (bitwise)");
    }
    {
        bool r = false;
        try { (void)e.encode_video(std::vector<float>(std::size_t(3) * 64 * 64), 1, 64, 64); } catch (const std::invalid_argument&) { r = true; }
        check(r, "video encoder rejects T < 2 (the image path)");
    }
    {   // a 17-frame reference clip at the real canvas: 768 x 1344 (28 tiles)
        std::mt19937 rng(4);
        std::uniform_real_distribution<float> ud(-1.0F, 1.0F);
        std::vector<float> x(std::size_t(3) * 17 * 768 * 1344);
        for (auto& v : x) v = ud(rng);
        double sec = 0;
        const auto z = e.encode_video(x, 17, 768, 1344, &sec);
        bool finite = z.size() == std::size_t(24) * 2 * 48 * 84;
        for (float f : z) finite &= std::isfinite(f);
        std::printf("  reference clip 17 x 768 x 1344 (28 tiles): %.1f s\n", sec);
        check(finite, "reference clip 17 x 768 x 1344: finite latent 24 x 2 x 48 x 84");
    }
    {   // keyframe at the real canvas: 768 x 1344 (4 x 7 tiles)
        std::mt19937 rng(3);
        std::uniform_real_distribution<float> ud(-1.0F, 1.0F);
        std::vector<float> x(std::size_t(3) * 768 * 1344);
        for (auto& v : x) v = ud(rng);
        double sec = 0;
        const auto z = e.encode(x, 768, 1344, &sec);
        bool finite = z.size() == std::size_t(24) * 48 * 84;
        for (float f : z) finite &= std::isfinite(f);
        std::printf("  keyframe 768x1344 (28 tiles): %.2f s\n", sec);
        check(finite, "keyframe 768x1344: finite latent 24 x 48 x 84");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
