// CPU tests of the H3 video VAE encoder host reference (single-image path) against the seeded fixtures.
// Needs H3_VAE_FIXTURES and H3_VAE_WEIGHTS; unset -> SKIPPED (with --require: FAIL).
//   enc_small (FP16 reference, 64 x 64): per-stage taps and the latent; enc_tiles (FP32 reference, 320 x 400,
//   2 x 2 tiles): tile blending of the reference moments -> latent.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "h3/vae/enc_ref.hpp"

using namespace strix::h3::vae;
using namespace strix::h3::vae::enc;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
Map tap_map(const Fixture& fx, const std::string& name) {  // [1][C][1][H][W] (FP16 or FP32) -> FP16 map
    const auto& ti = fx.info(name);
    Map m;
    m.C = int(ti.shape[1]);
    m.H = int(ti.shape[ti.shape.size() - 2]);
    m.W = int(ti.shape[ti.shape.size() - 1]);
    if (ti.dtype == "f16") m.v = fx.f16(name);
    else {
        const auto f = fx.f32(name);
        m.v.resize(f.size());
        for (std::size_t i = 0; i < f.size(); ++i) m.v[i] = f2h(f[i]);
    }
    return m;
}
void report(const char* n, const Diff& d) {
    std::printf("  %-22s n=%zu mismatch=%zu max_abs=%.3g rel_l2=%.3g\n", n, d.n, d.mismatches, d.max_abs, d.rel_l2());
}
#include "encv_tests.inc"
}  // namespace

int main(int argc, char** argv) {
    const bool require = argc > 1 && !std::strcmp(argv[1], "--require");
    const char* root = std::getenv("H3_VAE_FIXTURES");
    const char* wp = std::getenv("H3_VAE_WEIGHTS");
    if (!root || !wp || !file_exists(std::string(root) + "/enc_small/manifest.json") || !file_exists(wp)) {
        std::printf("SKIPPED encoder host tests (H3_VAE_FIXTURES / H3_VAE_WEIGHTS unset or missing)\n");
        std::printf("%s\n", require ? "FAILED (qualification: missing evidence)" : "PASS WITH SKIPS (0 failures, 1 groups SKIPPED)");
        return require ? 1 : 0;
    }
    {
        bool r1 = false, r2 = false;
        try { check_image(16, 64, std::size_t(3) * 16 * 64); } catch (const std::invalid_argument&) { r1 = true; }
        try { check_image(64, 64, 5); } catch (const std::invalid_argument&) { r2 = true; }
        check(r1 && r2, "encoder shape / buffer validation rejects 16 px sides and length mismatches");
    }
    SafeTensors st(wp);
    const Weights w = load(st);
    {
        const Fixture fx(std::string(root) + "/enc_small");
        const auto& xi = fx.info("x_f32");
        const int H = int(xi.shape[3]), W = int(xi.shape[4]);
        const Map px = normalize_pixels(fx.f32("x_f32"), H, W);
        const Diff dn = diff_f16(px.v, tap_map(fx, "tap_call00_in").v);
        report("normalize", dn);
        check(dn.bitwise(), "encoder pixel normalization bitwise");
        const Map ci = conv(w.conv_in, px, 1);
        const Diff dc = diff_f16(ci.v, tap_map(fx, "tap_c0_conv_in").v);
        report("conv_in", dc);
        check(dc.valid() && dc.rel_l2() < 2e-3, "conv_in (3x3 reflect) vs fixture");
        const Map b0 = resblock(w.blocks[0][0], tap_map(fx, "tap_c0_conv_in"));
        const Diff db = diff_f16(b0.v, tap_map(fx, "tap_c0_l0_b0").v);
        report("level 0 block 0", db);
        check(db.valid() && db.rel_l2() < 2e-3, "resnet block (gn + silu + conv x2 + residual) vs fixture");
        const Map dn0 = conv(w.down[0], tap_map(fx, "tap_c0_l0_b1"), 2);
        const Diff dd = diff_f16(dn0.v, tap_map(fx, "tap_c0_l0_down").v);
        report("downsample 0", dd);
        check(dd.valid() && dd.rel_l2() < 2e-3, "downsample (3x3 stride 2, reflect right/bottom) vs fixture");
        const Map nb = resblock(w.blocks[1][0], tap_map(fx, "tap_c0_l0_down"));
        const Diff dnb = diff_f16(nb.v, tap_map(fx, "tap_c0_l1_b0").v);
        report("level 1 block 0 (nin)", dnb);
        check(dnb.valid() && dnb.rel_l2() < 2e-3, "resnet block with 1x1 shortcut vs fixture");
        const Map mo = encode_tile(w, px);
        const Diff dm = diff_f16(mo.v, tap_map(fx, "tap_call00_moments").v);
        report("moments (full tile)", dm);
        check(dm.valid() && dm.rel_l2() < 1e-2, "whole host encoder tile -> moments vs fixture (rel_l2 < 1e-2)");
        const auto z = latent_of(w, tap_map(fx, "tap_call00_moments"));
        const Diff dz = diff_f32(z, fx.f32("latent_f32"));
        report("latent from moments", dz);
        check(dz.bitwise(), "latent normalization of the reference moments bitwise");
    }
    if (file_exists(std::string(root) + "/enc_tiles/manifest.json")) {
        const Fixture fx(std::string(root) + "/enc_tiles");
        const auto& xi = fx.info("x_f32");
        const int H = int(xi.shape[3]), W = int(xi.shape[4]);
        const Tiles1D ty = split_tiles(H), tx = split_tiles(W);
        std::vector<std::vector<Map>> tiles(ty.start.size());
        int k = 0;
        for (std::size_t i = 0; i < ty.start.size(); ++i)
            for (std::size_t j = 0; j < tx.start.size(); ++j) {
                char name[40];
                std::snprintf(name, sizeof name, "tap_call%02d_moments", k++);
                tiles[i].push_back(tap_map(fx, name));
            }
        const Map m = blend_tiles(tiles, ty, tx);
        const auto z = latent_of(w, m);
        const Diff d = diff_f32(z, fx.f32("latent_f32"));
        report("enc_tiles latent", d);
        // the reference blends in FP32 (FP32 fixture); this path blends the FP16-rounded moments in FP16
        check(d.valid() && d.max_abs < 2e-2 && d.rel_l2() < 2e-3, "tile blending of the reference moments (2 x 2 tiles) -> latent");
    } else {
        std::printf("SKIPPED enc_tiles (no fixture)\n");
        if (require) ++failures;
    }
    encv_host_checks(w, root, require, failures, check, report);
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
