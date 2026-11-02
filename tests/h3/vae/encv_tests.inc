// Multi-frame encoder checks shared by the host test (included into enc_host_tests.cpp).
// encv_small: FP16 python, 3 x 5 x 32 x 32 (one clip padded to 17 frames, one tile): pixel prep bitwise, conv_in,
// level-0 block, level-1 temporal downsample from the reference's own inputs, the whole clip tile -> moments, latent.
// encv (FP32, 3 x 22 x 64 x 96: 2 clips) / encv_tiles (FP32, 3 x 9 x 320 x 288: 2 x 2 tiles): clip / tile assembly of
// the reference moments -> latent.

inline Vid tap_vid(const Fixture& fx, const std::string& name) {  // [1][C][T][H][W] (FP16 or FP32) -> FP16 Vid
    const auto& ti = fx.info(name);
    Vid m;
    m.C = int(ti.shape[1]); m.T = int(ti.shape[2]); m.H = int(ti.shape[3]); m.W = int(ti.shape[4]);
    if (ti.dtype == "f16") m.v = fx.f16(name);
    else {
        const auto f = fx.f32(name);
        m.v.resize(f.size());
        for (std::size_t i = 0; i < f.size(); ++i) m.v[i] = f2h(f[i]);
    }
    return m;
}

inline void encv_host_checks(const Weights& w, const std::string& root, bool require, int& failures,
                             void (*check)(bool, const std::string&), void (*report)(const char*, const Diff&)) {
    {
        bool r = false;
        try { check_video(1, 64, 64, std::size_t(3) * 64 * 64); } catch (const std::invalid_argument&) { r = true; }
        check(r, "video encoder validation rejects T < 2");
    }
    if (file_exists(root + "/encv_small/manifest.json")) {
        const Fixture fx(root + "/encv_small");
        const auto& xi = fx.info("x_f32");
        const int T = int(xi.shape[2]), H = int(xi.shape[3]), W = int(xi.shape[4]);
        const Vid px = clip_pixels(fx.f32("x_f32"), T, H, W, 0);
        const Diff dp = diff_f16(px.v, tap_vid(fx, "tap_call00_in").v);
        report("video pixels (clip 0)", dp);
        check(dp.bitwise(), "video clip pixels (17-frame fill + normalization) bitwise");
        const Diff dc = diff_f16(conv3(w.conv_in, px, 1, 1).v, tap_vid(fx, "tap_c0_conv_in").v);
        report("video conv_in", dc);
        check(dc.valid() && dc.rel_l2() < 2e-3, "causal 3-D conv_in vs fixture");
        const Diff db = diff_f16(resblock3(w.blocks[0][0], tap_vid(fx, "tap_c0_conv_in")).v, tap_vid(fx, "tap_c0_l0_b0").v);
        report("video level 0 block 0", db);
        check(db.valid() && db.rel_l2() < 2e-3, "causal resnet block (per-frame GN) vs fixture");
        const Diff dd = diff_f16(conv3(w.down[1], tap_vid(fx, "tap_c0_l1_b1"), 2, 2).v, tap_vid(fx, "tap_c0_l1_down").v);
        report("video downsample 1 (t/2)", dd);
        check(dd.valid() && dd.rel_l2() < 2e-3, "temporal + spatial downsample vs fixture");
        const Vid mo = encode_clip_tile(w, px);
        const Diff dm = diff_f16(mo.v, tap_vid(fx, "tap_call00_moments").v);
        report("video moments (clip tile)", dm);
        check(dm.valid() && dm.rel_l2() < 1e-2, "whole host video encoder clip tile -> moments vs fixture");
        const Diff dz = diff_f32(video_latent(w, {tap_vid(fx, "tap_call00_moments")}), fx.f32("latent_f32"));
        report("video latent from moments", dz);
        check(dz.bitwise(), "video latent assembly (5 per clip, last 3 dropped) of the reference moments bitwise");
    } else {
        std::printf("SKIPPED encv_small (no fixture)\n");
        if (require) ++failures;
    }
    for (const char* fcase : {"encv", "encv_tiles"}) {
        if (!file_exists(root + "/" + fcase + "/manifest.json")) {
            std::printf("SKIPPED %s (no fixture)\n", fcase);
            if (require) ++failures;
            continue;
        }
        const Fixture fx(root + "/" + fcase);
        const auto& xi = fx.info("x_f32");
        const int T = int(xi.shape[2]), H = int(xi.shape[3]), W = int(xi.shape[4]);
        const Tiles1D ty = split_tiles(H), tx = split_tiles(W);
        const int clips = (T + 16) / 17, per = int(ty.start.size() * tx.start.size());
        std::vector<Vid> cm;
        for (int c = 0; c < clips; ++c) {
            std::vector<std::vector<Vid>> tiles(ty.start.size());
            for (std::size_t i = 0; i < ty.start.size(); ++i)
                for (std::size_t j = 0; j < tx.start.size(); ++j) {
                    char name[40];
                    std::snprintf(name, sizeof name, "tap_call%02d_moments", c * per + int(i * tx.start.size() + j));
                    tiles[i].push_back(tap_vid(fx, name));
                }
            cm.push_back(blend_tiles3(tiles, ty, tx));
        }
        const Diff d = diff_f32(video_latent(w, cm), fx.f32("latent_f32"));
        report((std::string(fcase) + " latent").c_str(), d);
        check(d.valid() && d.rel_l2() < 2e-3, std::string(fcase) + ": clip x tile assembly of the reference moments -> latent");
    }
}
