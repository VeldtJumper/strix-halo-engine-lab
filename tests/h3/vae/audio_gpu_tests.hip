// GPU qualification of the H3 audio decoder against the seeded python fixtures.
// usage: strix-h3-audio-gpu-tests [audio weights] [fixture root]   (defaults H3_AUDIO_VAE_WEIGHTS, H3_VAE_FIXTURES)
// Missing weights or fixtures is a FAILURE. Gates vs the python FP32 decoder: wave SNR >= 90 dB (primary) and max abs
// <= 5e-4 (secondary, -66 dBFS), finite, deterministic (bitwise rerun), on audio (8 latents), audio50 (50, per-level
// taps), audio207 (the 5 s clip). Calibration: the reference itself drifts with the CPU thread count (conv reduction
// order): on audio207, 8 / 4 / 1 threads vs the 16-thread fixture = SNR 111.0 / 111.0 / 105.1 dB, max abs 2.5e-5 /
// 2.5e-5 / 3.6e-5; ours (FP32 VALU convs, another fixed order) 96.2 dB / 1.2e-4. Input validation; random 5 s timing.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>

#include "h3/vae/audio_decoder.hpp"
#include "h3/vae/vae_ref.hpp"

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
    std::string weights = std::getenv("H3_AUDIO_VAE_WEIGHTS") ? std::getenv("H3_AUDIO_VAE_WEIGHTS") : "";
    std::string root = std::getenv("H3_VAE_FIXTURES") ? std::getenv("H3_VAE_FIXTURES") : "";
    if (argc > 1) weights = argv[1];
    if (argc > 2) root = argv[2];
    if (weights.empty() || root.empty() || !file_exists(weights) || !file_exists(root + "/audio/manifest.json") ||
        !file_exists(root + "/audio50/manifest.json")) {
        std::printf("FAIL qualification needs the audio weights and the audio / audio50 fixtures\nFAILED (1 failures)\n");
        return 1;
    }
    audio::GpuDecoder dec(weights);
    {
        const auto throws = [](auto&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
        check(throws([&] { (void)dec.decode({}, 8); }) && throws([&] { (void)dec.decode(std::vector<float>(32 * 2 * 8 - 1), 8); }) &&
              throws([&] { (void)dec.decode({}, 0); }),
              "audio decoder rejects empty / truncated latents and T < 1");
    }
    if (!file_exists(root + "/audio207/manifest.json")) check(false, "audio207 fixture present (5 s clip reference)");
    for (const char* fcase : {"audio", "audio50", "audio207"}) {
        if (!file_exists(root + "/" + fcase + "/manifest.json")) continue;
        const Fixture fx(root + "/" + fcase);
        const int T = int(fx.info("z_norm_f32").shape[3]);
        std::vector<std::vector<float>> taps;
        double ms = 0;
        const auto z = fx.f32("z_norm_f32");
        const auto wave = dec.decode(z, T, &ms, &taps);
        const auto ref = fx.f32("wave_f32");
        const Diff d = diff_f32(wave, ref);
        const double snr = snr_db(wave, ref);
        std::printf("  %s: T=%d wave n=%zu max_abs=%.3g snr=%.1f dB, gpu %.1f ms\n", fcase, T, d.n, d.max_abs, snr, ms);
        for (std::size_t lvl = 0; lvl < taps.size(); ++lvl) {
            const std::string name = "tap_level" + std::to_string(lvl);
            if (fx.has(name)) std::printf("    level %zu snr %.1f dB\n", lvl, snr_db(taps[lvl], fx.f32(name)));
        }
        check(d.valid() && snr >= 90.0 && d.max_abs <= 5e-4, std::string(fcase) + ": wave SNR >= 90 dB (primary), max abs <= 5e-4 (secondary), finite");
        const auto again = dec.decode(z, T);
        check(diff_f32(again, wave).bitwise(), std::string(fcase) + ": deterministic (bitwise rerun)");
    }
    {   // 5 s clip: 207 latents (40 / s + padding as the DiT's audio rows)
        const int T = 207;
        std::mt19937 rng(5);
        std::normal_distribution<float> nd;
        std::vector<float> z(std::size_t(32) * 2 * T);
        for (auto& v : z) v = nd(rng);
        double ms = 0;
        const auto wave = dec.decode(z, T, &ms);
        bool finite = wave.size() == std::size_t(2) * 800 * T;
        for (float f : wave) finite &= std::isfinite(f);
        std::printf("  5 s clip (T=207, 165600 samples x 2): gpu %.1f ms\n", ms);
        check(finite, "5 s clip: finite stereo wave of 800 T samples");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
