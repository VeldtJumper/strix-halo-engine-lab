// CPU test of the H3 audio decoder host reference against the seeded fixtures.
// Needs H3_VAE_FIXTURES (fixture root) and H3_AUDIO_VAE_WEIGHTS (audio VAE fp32 safetensors); unset or missing ->
// SKIPPED, or FAILED with --require (qualification). Metrics are the shared validated Diff (non-finite values or a
// length mismatch fail every gate).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "h3/vae/audio_ref.hpp"
#include "h3/vae/vae_ref.hpp"

using namespace strix::h3::vae;

namespace {
int failures = 0, skipped = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
}  // namespace

int main(int argc, char** argv) {
    const bool require = argc > 1 && !std::strcmp(argv[1], "--require");
    const char* root = std::getenv("H3_VAE_FIXTURES");
    const char* wpath = std::getenv("H3_AUDIO_VAE_WEIGHTS");
    {   // the shared metric rejects NaN and length mismatches
        const std::vector<float> a = {0.5F, 0.25F}, b = {0.5F, NAN};
        const Diff dn = diff_f32(b, a);
        check(!dn.valid() && dn.snr_db() == -INFINITY && !(dn.snr_db() >= 80.0), "audio metric: NaN can never pass the SNR gate");
    }
    {
        bool r1 = false, r2 = false;
        try { audio::check_latent({}, 8); } catch (const std::invalid_argument&) { r1 = true; }
        try { audio::check_latent(std::vector<float>(32 * 2 * 8), 0); } catch (const std::invalid_argument&) { r2 = true; }
        check(r1 && r2, "audio latent validation rejects empty latents and T < 1");
    }
    if (!root || !wpath || !file_exists(std::string(root) + "/audio/manifest.json") || !file_exists(wpath)) {
        std::printf("SKIPPED audio host tests (H3_VAE_FIXTURES / H3_AUDIO_VAE_WEIGHTS unset or missing)\n");
        ++skipped;
    } else {
        SafeTensors st(wpath);
        const audio::Weights w = audio::load(st);
        for (const char* fcase : {"audio", "audio50"}) {
            const std::string dir = std::string(root) + "/" + fcase;
            if (!file_exists(dir + "/manifest.json")) { std::printf("SKIPPED %s (no fixture)\n", fcase); ++skipped; continue; }
            Fixture fx(dir);
            const int T = int(fx.info("z_norm_f32").shape[3]);
            std::vector<std::vector<audio::Sig>> taps;
            const auto wave = audio::decode(w, fx.f32("z_norm_f32"), T, &taps);
            const Diff d = diff_f32(wave, fx.f32("wave_f32"));
            std::printf("  %s wave: n=%zu nonfinite=%zu max_abs=%.3g snr=%.1f dB\n", fcase, d.n, d.nonfinite, d.max_abs, d.snr_db());
            for (int lvl = 0; lvl < 7; ++lvl) {
                const std::string name = "tap_level" + std::to_string(lvl);
                if (!fx.has(name)) break;
                std::vector<float> got;
                for (int r = 0; r < 2; ++r) got.insert(got.end(), taps[std::size_t(r)][std::size_t(lvl)].begin(), taps[std::size_t(r)][std::size_t(lvl)].end());
                std::printf("  %s level %d: snr=%.1f dB\n", fcase, lvl, diff_f32(got, fx.f32(name)).snr_db());
            }
            check(d.valid() && d.snr_db() >= 90.0 && d.max_abs <= 5e-4, std::string(fcase) + ": host audio decoder vs reference (SNR >= 90 dB primary, max abs <= 5e-4, finite)");
        }
    }
    if (require && skipped) {
        std::printf("FAIL qualification: %d groups SKIPPED (missing fixtures or weights)\n", skipped);
        failures += skipped;
    }
    std::printf("%s (%d failures, %d groups SKIPPED)\n", failures ? "FAILED" : skipped ? "PASS WITH SKIPS" : "ALL PASS", failures, skipped);
    return failures ? 1 : 0;
}
