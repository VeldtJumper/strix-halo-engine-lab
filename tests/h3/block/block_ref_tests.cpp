// MiniMax H3 block glue: host reference tests against the reference capture (h3fx fixtures) and the real checkpoint.
// usage: strix-h3-block-ref-tests [unit] [weights] [layout] [block0] [refiner] [forward] [forward-fl2va] [sampler]
//   (no argument: unit only - no fixtures or checkpoint needed). Env: H3_FIXTURES (fixture root), H3_CKPT
//   (checkpoint path), H3_HOST_THREADS.
#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <string>

#include "h3/block/carry_io.hpp"
#include "h3/block/host_ref.hpp"
#include "h3/block/layout.hpp"
#include "h3/block/sampler.hpp"
#include "h3/block/weights.hpp"
#include "test_util.hpp"

using namespace strix::h3::block;
using namespace h3t;

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------------------------------------ checkpoint / API rejects
// Serialized split runs (save_carry -> file -> load_carry -> continue) == one uninterrupted run, bitwise, for both
// samplers and every split point; malformed / foreign checkpoints and inconsistent stream inputs are rejected.
void test_carry_and_rejects() {
    std::puts("[carry / rejects]");
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("h3_carry_test_" + std::to_string(static_cast<long long>(now_s() * 1e6)));
    fs::create_directories(dir);
    const fs::path f = dir / "dit_state.bin";
    const std::size_t nv = 24 * 2 * 4 * 4, na = 32 * 2 * 5;
    // a deterministic, non-trivial stand-in model (velocity-like output from x and sigma)
    const AvModelFn model = [](const AvState& x, float s) {
        AvState o;
        o.video.resize(x.video.size());
        o.audio.resize(x.audio.size());
        for (std::size_t i = 0; i < x.video.size(); ++i) o.video[i] = std::sin(1.3F * x.video[i] + s) - 0.25F * x.video[i];
        for (std::size_t i = 0; i < x.audio.size(); ++i) o.audio[i] = std::cos(0.7F * x.audio[i] - s) + 0.1F * x.audio[i];
        return o;
    };
    std::vector<float> nvv(nv), naa(na);
    for (std::size_t i = 0; i < nv; ++i) nvv[i] = std::sin(0.37F * static_cast<float>(i)) * 1.5F;
    for (std::size_t i = 0; i < na; ++i) naa[i] = std::cos(0.11F * static_cast<float>(i));
    for (SamplerKind kind : {SamplerKind::euler, SamplerKind::res_multistep}) {
        const char* kn = kind == SamplerKind::euler ? "euler" : "res_multistep";
        const std::vector<float> sig = simple_sigmas(12.0F, 8);
        SamplerCarry full;
        full.x = initial_state(sig[0], nvv, naa);
        sample_steps(model, full, sig, kind, 8);
        int ok = 0;
        for (std::size_t k = 1; k < 8; ++k) {
            SamplerCarry a;
            a.x = initial_state(sig[0], nvv, naa);
            sample_steps(model, a, sig, kind, k);
            save_carry(f, a);
            SamplerCarry b;
            const bool loaded = load_carry(f, b, nv, na, sig);
            sample_steps(model, b, sig, kind, 8);
            ok += loaded && b.next == 8 && b.x.video == full.x.video && b.x.audio == full.x.audio;
        }
        check(ok == 7, std::string(kn) + ": serialized split at every step 1..7 == uninterrupted, bitwise (" + std::to_string(ok) + "/7)");
        // malformed / foreign checkpoints, each must throw
        SamplerCarry good;
        good.x = initial_state(sig[0], nvv, naa);
        sample_steps(model, good, sig, kind, 3);
        save_carry(f, good);
        std::vector<char> bytes;
        {
            std::ifstream in(f, std::ios::binary);
            bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        auto rejects = [&](const std::vector<char>& b, std::size_t vsz, std::size_t asz, const std::vector<float>& sg) {
            {
                std::ofstream o(f, std::ios::binary | std::ios::trunc);
                o.write(b.data(), static_cast<std::streamsize>(b.size()));
            }
            SamplerCarry c;
            try {
                load_carry(f, c, vsz, asz, sg);
            } catch (const std::exception&) {
                return true;
            }
            return false;
        };
        auto with_u64 = [&](std::size_t off, std::uint64_t v) {
            std::vector<char> b = bytes;
            std::memcpy(b.data() + off, &v, 8);
            return b;
        };
        auto with_f32 = [&](std::size_t off, float v) {
            std::vector<char> b = bytes;
            std::memcpy(b.data() + off, &v, 4);
            return b;
        };
        const std::size_t hist0 = 44 + (nv + na) * 4;
        int rej = 0, cases = 0;
        auto expect = [&](bool r, const char* what) {
            ++cases;
            rej += r;
            if (!r) std::printf("  NOT rejected: %s %s\n", kn, what);
        };
        expect(rejects(std::vector<char>(bytes.begin(), bytes.begin() + 20), nv, na, sig), "short header");
        expect(rejects(with_u64(0, 0x1234ULL), nv, na, sig), "bad magic");
        expect(rejects(with_u64(8, 0), nv, na, sig), "next = 0");
        expect(rejects(with_u64(8, 9), nv, na, sig), "next beyond the schedule");
        expect(rejects(with_u64(16, 0), nv, na, sig), "history flag 0");
        expect(rejects(with_u64(16, 2), nv, na, sig), "history flag 2");
        expect(rejects(with_f32(40, sig[3] * 0.5F), nv, na, sig), "history sigma != schedule");
        expect(rejects(with_f32(40, NAN), nv, na, sig), "non-finite history sigma");
        expect(rejects(with_f32(44, NAN), nv, na, sig), "NaN in the state");
        expect(rejects(with_f32(hist0 + 4, INFINITY), nv, na, sig), "inf in the history");
        expect(rejects(std::vector<char>(bytes.begin(), bytes.end() - 4), nv, na, sig), "truncated payload");
        expect(rejects(bytes, nv + 1, na, sig), "latent size of another run");
        expect(rejects(bytes, nv, na, simple_sigmas(12.0F, 6)), "another schedule (6 steps)");
        expect(rejects(bytes, nv, na, simple_sigmas(10.0F, 8)), "another shift");
        check(rej == cases, std::string(kn) + ": malformed / foreign checkpoints rejected (" + std::to_string(rej) + "/" + std::to_string(cases) + ")");
        // the unmodified file still loads
        {
            std::ofstream o(f, std::ios::binary | std::ios::trunc);
            o.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        SamplerCarry c;
        check(load_carry(f, c, nv, na, sig) && c.next == 3, std::string(kn) + ": the intact checkpoint loads (next 3)");
    }
    fs::remove(f);
    check(!load_carry(f, *std::make_unique<SamplerCarry>(), nv, na, simple_sigmas(12.0F, 8)), "no checkpoint file -> fresh run");
    // embed_rows: shape-derived ranges validated before any read
    {
        const PackedLayout L = PackedLayout::build(4, 2, 4, 4, 5);
        StreamInputs in;
        in.T = 2;
        in.H = 4;
        in.W = 4;
        in.audio_t = 5;
        in.video_x.assign(24 * 2 * 4 * 4, 0);
        in.audio_x.assign(32 * 2 * 5, 0);
        std::vector<float> av, aa;
        bool good = true;
        try {
            embed_rows(L, in, av, aa);
        } catch (const std::exception&) {
            good = false;
        }
        check(good, "embed_rows accepts consistent inputs");
        int rj = 0;
        auto bad = [&](StreamInputs b) {
            try {
                embed_rows(L, b, av, aa);
            } catch (const std::invalid_argument&) {
                ++rj;
            }
        };
        StreamInputs b1 = in; b1.video_x.pop_back(); bad(b1);
        StreamInputs b2 = in; b2.audio_x.push_back(0); bad(b2);
        StreamInputs b3 = in; b3.T = 3; bad(b3);
        StreamInputs b4 = in; b4.cond_video.assign(96, 0.0F); bad(b4);
        StreamInputs b5 = in; b5.cond_audio.assign(32, 0.0F); bad(b5);
        check(rj == 5, "embed_rows rejects wrong latent sizes / shape / cond rows before reading (" + std::to_string(rj) + "/5)");
    }
    // malformed norm assets: every norm vector of a refiner / DiT block and both final norms are shape-checked
    {
        int rj = 0, cases = 0;
        auto thr = [&](auto fn) {
            ++cases;
            try {
                fn();
            } catch (const std::runtime_error&) {
                ++rj;
            }
        };
        BlockWeights ok;
        ok.norm1.assign(kHidden, 0);
        ok.norm2.assign(kHidden, 0);
        ok.q_norm.assign(kHeadDim, 0);
        ok.k_norm.assign(kHeadDim, 0);
        bool good = true;
        try {
            detail::check_norms(ok, "block");
        } catch (const std::exception&) {
            good = false;
        }
        RefinerBlockWeights r;
        r.norm1 = ok.norm1;
        r.norm2 = ok.norm2;
        r.q_norm = ok.q_norm;
        r.k_norm = ok.k_norm;
        r.k_norm.pop_back();
        thr([&] { detail::check_norms(r, "refiner"); });
        BlockWeights b1 = ok;
        b1.norm1.pop_back();
        thr([&] { detail::check_norms(b1, "block"); });
        BlockWeights b2 = ok;
        b2.q_norm.push_back(0);
        thr([&] { detail::check_norms(b2, "block"); });
        GlobalWeights g;
        g.refiner_final_norm.assign(kHidden, 0);
        g.final_norm.assign(kHidden, 0);
        try {
            detail::check_final_norms(g);
        } catch (const std::exception&) {
            good = false;
        }
        GlobalWeights g1 = g;
        g1.final_norm.resize(kHidden - 1);
        thr([&] { detail::check_final_norms(g1); });
        GlobalWeights g2 = g;
        g2.refiner_final_norm.clear();
        thr([&] { detail::check_final_norms(g2); });
        check(good && rj == cases,
            "malformed norm assets rejected (" + std::to_string(rj) + "/" + std::to_string(cases) + "), intact ones accepted");
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
}

// ------------------------------------------------------------------------------------------------ unit (no data)
void test_unit() {
    test_carry_and_rejects();
    std::puts("[unit]");
    // ref_sum_f32 equals the plain sum on exactly representable inputs
    std::vector<float> v(5376);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i % 7);
    double s = 0;
    for (float f : v) s += f;
    check(ref_sum_f32(v.data(), 5376) == static_cast<float>(s), "ref_sum_f32 exact on small integers (n=5376)");
    double s128 = 0;
    for (int i = 0; i < 128; ++i) s128 += v[i];
    check(ref_sum_f32(v.data(), 128) == static_cast<float>(s128), "ref_sum_f32 n=128");
    // layout of the small t2va shape
    const PackedLayout L = PackedLayout::build(256, 2, 16, 16, 8);
    check(L.seq_len == 400 && L.segments.size() == 3 && L.segments[1].start == 256 && L.segments[2].start == 272, "t2va layout 400 rows");
    const ForwardPlan p = ForwardPlan::build(L, 1.0F, {});
    check(p.n_t() == 1 && p.mod_segments.size() == 3 && p.mod_segments[0].row == 1 && p.mod_segments[1].row == 2, "plan at sigma 1");
    // patchify round trip
    std::vector<float> vid(24 * 2 * 4 * 6);
    for (std::size_t i = 0; i < vid.size(); ++i) vid[i] = static_cast<float>(i);
    const auto r = patchify_video(vid.data(), 2, 4, 6);
    check(unpatchify_video(r.data(), 2, 4, 6) == vid, "patchify / unpatchify round trip");
    std::vector<float> aud(32 * 2 * 5);
    for (std::size_t i = 0; i < aud.size(); ++i) aud[i] = static_cast<float>(i);
    check(unpack_audio(pack_audio(aud.data(), 5).data(), 5) == aud, "pack / unpack audio round trip");
    check(pack_audio(aud.data(), 5)[1] == aud[1 * 10], "audio channel-major packing");
}

// ------------------------------------------------------------------------------------------------ weights
void test_weights(const SafeTensors& st, const GlobalWeights& g) {
    std::puts("[weights]");
    check(st.tensors().size() == 932, "932 tensors in the checkpoint (" + std::to_string(st.tensors().size()) + ")");
    check(count_blocks(st) == 50, "50 blocks");
    check(g.basis == 8 && g.curve_grid == 1025, "curve table [1025, 8]");
    check(g.pdd_steps == 1, "single output head (pdd_steps 1)");
    check(g.refiner.size() == 2, "2 refiner blocks");
    check(g.inv_freq.size() == 16, "rope inv_freq [16]");
    const double t0 = now_s();
    const BlockWeights b0 = load_block(st, 0, g.basis);
    const BlockWeights b49 = load_block(st, 49, g.basis);
    std::printf("  two blocks loaded in %.2f s\n", now_s() - t0);
    // codes and scales vs the quantiser fixture (first 64 output channels of block 0 qkv / fc2)
    const fx::Manifest q = fx::Manifest::load(fixtures_root() + "/quant");
    const auto wq = as_vec<std::int8_t>(q.read("K5376/normal/w_q"));
    const auto ws = f32v(q, "K5376/normal/w_scale");
    check(std::equal(wq.begin(), wq.end(), b0.qkv.w.begin()) && std::equal(ws.begin(), ws.end(), b0.qkv.ws.begin()),
        "block 0 qkv codes + scales == quant fixture (64 channels)");
    const auto wq2 = as_vec<std::int8_t>(q.read("K14336/normal/w_q"));
    const auto ws2 = f32v(q, "K14336/normal/w_scale");
    check(std::equal(wq2.begin(), wq2.end(), b0.fc2.w.begin()) && std::equal(ws2.begin(), ws2.end(), b0.fc2.ws.begin()),
        "block 0 fc2 codes + scales == quant fixture (64 channels)");
    check(b49.fc1.n == 28672 && b49.fc1.k == 5376 && b49.adaln.n == 96768, "block 49 shapes");
    {
        // attention input contract: static bound of the scaled q.k score from every block's q / k norm weights
        double worst = 0;
        std::uint32_t wb = 0, nsafe = 0;
        for (std::uint32_t i = 0; i < count_blocks(st); ++i) {
            const std::string p = "blocks." + std::to_string(i) + ".attn.";
            const double v = qk_score_bound(st.read_vec<std::uint16_t>(p + "q_norm.weight", StDtype::bf16),
                st.read_vec<std::uint16_t>(p + "k_norm.weight", StDtype::bf16));
            if (v <= kAttnScoreLimit) ++nsafe;
            if (v > worst) {
                worst = v;
                wb = i;
            }
        }
        // informational: attained only by one-hot head inputs; the hard check is the measured one in the device forward
        std::printf("  info  attention contract: static (weight-only) q.k score bound max over blocks %.1f (block %u), %u of %u "
                    "blocks <= 2^14 statically\n", worst, wb, nsafe, count_blocks(st));
    }
}

// ------------------------------------------------------------------------------------------------ layout / plan
struct CaseInfo {
    std::string set;
    PackedLayout L;
    std::vector<std::int64_t> tags;
    fx::Manifest m;
};
CaseInfo load_case(const std::string& set) {
    CaseInfo c{set, {}, {}, fx::Manifest::load(fixtures_root() + "/" + set)};
    const auto& sig = c.m.meta.at("layout").at("signature").arr();
    std::vector<Keyframe> kfs;
    const double nkf = c.m.meta.at("config").at("keyframes").num();
    for (int i = 0; i < static_cast<int>(nkf); ++i) {
        const auto& sh = c.m.find("in/keyframe" + std::to_string(i) + "_latent")->shape;
        kfs.push_back({0.0, static_cast<std::uint32_t>(sh[2]), 0});
    }
    c.L = PackedLayout::build(static_cast<std::uint32_t>(sig[0].num()), static_cast<std::uint32_t>(sig[1].num()),
        static_cast<std::uint32_t>(sig[2].num()), static_cast<std::uint32_t>(sig[3].num()), static_cast<std::uint32_t>(sig[4].num()), kfs);
    c.tags = as_vec<std::int64_t>(c.m.read("in/text_token_tags"));
    bool all_text = true;
    for (auto t : c.tags) all_text &= t == 1;
    if (all_text) c.tags.clear();
    return c;
}

[[maybe_unused]] float fixture_sigma(const fx::Manifest& m, int fwd) {
    return f32v(m, "f" + std::to_string(fwd) + "/model_in/timestep")[0] / 1000.0F;  // exact: sigma * 1000 / 1000 re-derived below
}

void test_layout(const GlobalWeights& g, const std::string& set) {
    std::printf("[layout %s]\n", set.c_str());
    CaseInfo c = load_case(set);
    const auto pos = as_vec<double>(c.m.read("layout/position_ids"));
    check(c.L.seq_len == static_cast<std::uint32_t>(c.m.meta.at("layout").at("seq_len").num()), "seq_len");
    check(pos == c.L.position_ids, "position_ids bitwise (f64)");
    const auto& segs = c.m.meta.at("layout").at("segments").arr();
    bool seg_ok = segs.size() == c.L.segments.size();
    for (std::size_t i = 0; seg_ok && i < segs.size(); ++i)
        seg_ok = segs[i].arr()[0].num() == c.L.segments[i].start && segs[i].arr()[1].num() == c.L.segments[i].stop &&
                 segs[i].arr()[2].str() == seg_kind_name(c.L.segments[i].kind);
    check(seg_ok, "segment table");
    if (c.m.find("layout/img_pos"))
        check(as_vec<std::int64_t>(c.m.read("layout/img_pos")) == c.L.img_pos && as_vec<std::int64_t>(c.m.read("layout/audio_pos")) == c.L.audio_pos,
            "img_pos / audio_pos");
    const auto ang = rope_angles(c.L.position_ids, g.inv_freq);
    check(ang == f32v(c.m, "layout/rope_angles"), "rope angles bitwise (fp32)");
    const RopeTable rt = rope_table(ang);
    const auto tab = bf16v(c.m, "f0/rope/table");  // [S][48][2][2] = c, -s, s, c
    std::size_t mism = 0;
    for (std::size_t r = 0; r < rt.seq; ++r)
        for (unsigned i = 0; i < 48; ++i) {
            const std::size_t o = (r * 48 + i) * 4;
            mism += tab[o] != rt.c[r * 48 + i] || tab[o + 2] != rt.s[r * 48 + i] || tab[o + 3] != rt.c[r * 48 + i] ||
                    tab[o + 1] != static_cast<std::uint16_t>(rt.s[r * 48 + i] ^ 0x8000U);
        }
    check(mism == 0, "rope table bitwise (bf16), mismatching entries " + std::to_string(mism));
    // timestep plans of every captured forward
    for (int f = 0; f < 2; ++f) {
        const std::string pre = "f" + std::to_string(f);
        if (!c.m.find(pre + "/model_in/timestep")) continue;
        const float ts = f32v(c.m, pre + "/model_in/timestep")[0];
        // the sampler sigma that produces this timestep (sigmas_simple20[f])
        const float sigma = f32v(c.m, "sampler/sigmas_simple20")[f];
        check(sigma * 1000.0F == ts, pre + " timestep == fp32(sigma * 1000)");
        const ForwardPlan p = ForwardPlan::build(c.L, sigma, c.tags);
        const auto* ms = c.m.meta.at("mod_segments").find(std::to_string(f));
        if (ms) {
            bool ok = ms->arr().size() == p.mod_segments.size();
            for (std::size_t i = 0; ok && i < p.mod_segments.size(); ++i)
                ok = ms->arr()[i].arr()[0].num() == p.mod_segments[i].start && ms->arr()[i].arr()[1].num() == p.mod_segments[i].stop &&
                     ms->arr()[i].arr()[2].num() == p.mod_segments[i].row;
            check(ok, pre + " mod segments");
        }
        if (c.m.find(pre + "/adaln/t_emb")) {
            const auto te = curve_t_emb(g, p.t_vals);
            check(te == f32v(c.m, pre + "/adaln/t_emb"), pre + " curve t_emb bitwise (n_t=" + std::to_string(p.n_t()) + ")");
        }
    }
}

// ------------------------------------------------------------------------------------------------ block 0 sub-ops
void test_block0(const SafeTensors& st, const GlobalWeights& g, const std::string& set) {
    std::printf("[block0 %s]\n", set.c_str());
    CaseInfo c = load_case(set);
    const fx::Manifest& m = c.m;
    const std::uint32_t S = c.L.seq_len;
    const BlockWeights b = load_block(st, 0, g.basis);
    const float sigma = f32v(m, "sampler/sigmas_simple20")[0];
    const ForwardPlan p = ForwardPlan::build(c.L, sigma, c.tags);
    const auto te = curve_t_emb(g, p.t_vals);
    const auto ada = adaln_block_f32(b, te, p.n_t());
    // adaLN fp32 rows vs the capture; and at the consumed precision (bf16)
    for (int j = 0; j < 6; ++j) {
        const auto ref = f32v(m, "f0/b0/adaln_out" + std::to_string(j));
        std::vector<float> got(ref.size());
        std::size_t bmism = 0;
        for (std::uint32_t r = 0; r < p.n_mod_rows(); ++r)
            for (std::uint32_t k = 0; k < kHidden; ++k) {
                got[r * kHidden + k] = ada[(static_cast<std::size_t>(r) * 6 + j) * kHidden + k];
                bmism += tob(got[r * kHidden + k]) != tob(ref[r * kHidden + k]);
            }
        const Cmp cm = compare_f32(got, ref);
        check(cm.rel_l2 < 1e-6 && bmism * 10000 < ref.size(),
            "adaln_out" + std::to_string(j) + " fp32 " + cm.str() + " | bf16-cast mism " + std::to_string(bmism));
    }
    const ModVectors mv = mod_vectors(ada, p.n_mod_rows());
    const auto mrow = mod_row_of(p.mod_segments, S);
    // teacher-forced sub-ops from the captured inputs
    const auto x_in = bf16v(m, "f0/b0/norm1_in");
    std::vector<std::uint16_t> h(x_in.size());
    for (std::uint32_t r = 0; r < S; ++r) rmsnorm_row(x_in.data() + r * kHidden, b.norm1.data(), kHidden, kNormEps, h.data() + r * kHidden);
    check(compare_bf16(h, bf16v(m, "f0/b0/norm1_out")).mism == 0, "norm1 bitwise " + compare_bf16(h, bf16v(m, "f0/b0/norm1_out")).str());
    h = bf16v(m, "f0/b0/norm1_out");
    for (std::uint32_t r = 0; r < S; ++r)
        mod_scale_shift_row(h.data() + r * kHidden, mv.s1_msa.data() + mrow[r] * kHidden, mv.shift_msa.data() + mrow[r] * kHidden, kHidden);
    { const Cmp cm = compare_bf16(h, bf16v(m, "f0/b0/attn_in")); check(cm.mism * 10000 <= cm.n, "modulation msa " + cm.str()); }
    HostLinear lin;
    const auto qin = bf16v(m, "f0/b0/qkv_proj_in");
    std::vector<std::uint16_t> qkv(static_cast<std::size_t>(S) * 3 * kInner);
    double t0 = now_s();
    lin.run(b.qkv, qin.data(), S, qkv.data());
    const double tq = now_s() - t0;
    { const Cmp cm = compare_bf16(qkv, bf16v(m, "f0/b0/qkv_proj_out")); check(cm.mism == 0 || cm.rel_l2 < 1e-4, "qkv_proj bitwise (or a few library-rotation ulp code flips) " + cm.str() + " (" + std::to_string(tq) + " s)"); }
    qkv = bf16v(m, "f0/b0/qkv_proj_out");
    std::vector<std::uint16_t> q(static_cast<std::size_t>(S) * kInner), k(q.size()), vt(q.size());
    const RopeTable rope = rope_table(rope_angles(c.L.position_ids, g.inv_freq));
    qk_prepare(qkv.data(), S, b.q_norm.data(), b.k_norm.data(), &rope, q.data(), k.data(), vt.data());
    // fixture q/k/v are [56][S][128]
    auto to_tm = [&](const std::vector<std::uint16_t>& hm) {
        std::vector<std::uint16_t> tm(hm.size());
        for (std::uint32_t hh = 0; hh < kHeads; ++hh)
            for (std::uint32_t r = 0; r < S; ++r)
                std::memcpy(tm.data() + (static_cast<std::size_t>(r) * kHeads + hh) * kHeadDim, hm.data() + (static_cast<std::size_t>(hh) * S + r) * kHeadDim, kHeadDim * 2);
        return tm;
    };
    const auto qref = to_tm(bf16v(m, "f0/attn/b00_q")), kref = to_tm(bf16v(m, "f0/attn/b00_k"));
    const auto vhm = bf16v(m, "f0/attn/b00_v");
    { const Cmp cm = compare_bf16(q, qref); check(cm.mism == 0, "q norm+rope bitwise " + cm.str()); }
    { const Cmp cm = compare_bf16(k, kref); check(cm.mism == 0, "k norm+rope bitwise " + cm.str()); }
    std::size_t vm = 0;
    for (std::uint32_t hh = 0; hh < kHeads; ++hh)
        for (std::uint32_t d = 0; d < kHeadDim; ++d)
            for (std::uint32_t r = 0; r < S; ++r)
                vm += vt[(static_cast<std::size_t>(hh) * kHeadDim + d) * S + r] != vhm[(static_cast<std::size_t>(hh) * S + r) * kHeadDim + d];
    check(vm == 0, "V^T layout bitwise");
    HostAttention att;
    std::vector<std::uint16_t> ao(static_cast<std::size_t>(S) * kInner);
    att.run(qref.data(), kref.data(), vt.data(), S, S, ao.data());
    { const Cmp cm = compare_bf16(ao, bf16v(m, "f0/attn/b00_out")); check(cm.rel_l2 < 2e-2, "attention fp64 vs capture (bf16 scores) " + cm.str()); }
    {
        CaptureAttention cap;
        std::vector<std::uint16_t> ac(ao.size());
        cap.run(qref.data(), kref.data(), vt.data(), S, S, ac.data());
        const Cmp cm = compare_bf16(ac, bf16v(m, "f0/attn/b00_out"));
        check(cm.rel_l2 < 1e-3, "capture-emulating attention vs capture " + cm.str());
    }
    std::vector<std::uint16_t> o(static_cast<std::size_t>(S) * kHidden);
    lin.run(b.out, bf16v(m, "f0/b0/out_proj_in").data(), S, o.data());
    { const Cmp cm = compare_bf16(o, bf16v(m, "f0/b0/out_proj_out")); check(cm.mism == 0 || cm.rel_l2 < 1e-4, "out_proj bitwise (or a few library-rotation ulp code flips) " + cm.str()); }
    auto x = bf16v(m, "f0/blocks/in_00");
    const auto ao_ref = bf16v(m, "f0/b0/attn_out");
    for (std::uint32_t r = 0; r < S; ++r) gate_row(x.data() + r * kHidden, ao_ref.data() + r * kHidden, mv.gate_msa.data() + mrow[r] * kHidden, kHidden);
    { const Cmp cm = compare_bf16(x, bf16v(m, "f0/b0/norm2_in")); check(cm.mism * 10000 <= cm.n, "gated residual msa " + cm.str()); }
    x = bf16v(m, "f0/b0/norm2_in");
    for (std::uint32_t r = 0; r < S; ++r) rmsnorm_row(x.data() + r * kHidden, b.norm2.data(), kHidden, kNormEps, h.data() + r * kHidden);
    { const Cmp cm = compare_bf16(h, bf16v(m, "f0/b0/norm2_out")); check(cm.mism == 0, "norm2 bitwise " + cm.str()); }
    h = bf16v(m, "f0/b0/norm2_out");
    for (std::uint32_t r = 0; r < S; ++r)
        mod_scale_shift_row(h.data() + r * kHidden, mv.s1_mlp.data() + mrow[r] * kHidden, mv.shift_mlp.data() + mrow[r] * kHidden, kHidden);
    { const Cmp cm = compare_bf16(h, bf16v(m, "f0/b0/mlp_in")); check(cm.mism * 10000 <= cm.n, "modulation mlp " + cm.str()); }
    std::vector<std::uint16_t> f1(static_cast<std::size_t>(S) * 2 * kFfn);
    lin.run(b.fc1, bf16v(m, "f0/b0/mlp_in").data(), S, f1.data());
    { const Cmp cm = compare_bf16(f1, bf16v(m, "f0/b0/fc1_out")); check(cm.mism == 0 || cm.rel_l2 < 1e-4, "fc1 bitwise (or a few library-rotation ulp code flips) " + cm.str()); }
    f1 = bf16v(m, "f0/b0/fc1_out");
    std::vector<std::uint16_t> sw(static_cast<std::size_t>(S) * kFfn);
    swiglu_host(f1.data(), S, kFfn, sw.data());
    if (set == "dit_t2va_small") {
        const fx::Manifest qm = fx::Manifest::load(fixtures_root() + "/quant");
        const auto xr = bf16v(qm, "K14336/real_b0_fc2_in/x");
        { const Cmp cm = compare_bf16(sw, xr); check(cm.mism == 0, "swiglu vs quant fixture fc2 input " + cm.str()); }
        const auto xrot = bf16v(qm, "K14336/real_b0_fc2_in/x_rot");
        const auto qcodes = as_vec<std::int8_t>(qm.read("K14336/real_b0_fc2_in/q"));
        std::vector<std::uint16_t> rot(xr.size());
        std::vector<std::int8_t> codes(xr.size());
        for (std::uint32_t r = 0; r < S; ++r)
            strix::h3::gemm::rotate_quantize_row(xr.data() + static_cast<std::size_t>(r) * kFfn, kFfn, codes.data() + static_cast<std::size_t>(r) * kFfn, rot.data() + static_cast<std::size_t>(r) * kFfn);
        std::size_t cmism = 0;
        for (std::size_t i = 0; i < codes.size(); ++i) cmism += codes[i] != qcodes[i];
        std::printf("  info  fc2 rotation vs capture: %s; code mismatches %zu\n", compare_bf16(rot, xrot).str().c_str(), cmism);
    }
    lin.run(b.fc2, sw.data(), S, o.data());
    { const Cmp cm = compare_bf16(o, bf16v(m, "f0/b0/mlp_out")); check(cm.rel_l2 < 1e-3, "swiglu + fc2 (exact rotation; the capture's rotation differs in a few ulps) " + cm.str()); }
    x = bf16v(m, "f0/b0/norm2_in");
    const auto mo = bf16v(m, "f0/b0/mlp_out");
    for (std::uint32_t r = 0; r < S; ++r) gate_row(x.data() + r * kHidden, mo.data() + r * kHidden, mv.gate_mlp.data() + mrow[r] * kHidden, kHidden);
    { const Cmp cm = compare_bf16(x, bf16v(m, "f0/blocks/out_00")); check(cm.mism * 10000 <= cm.n, "gated residual mlp " + cm.str()); }
    // the whole block from in_00 with the capture-emulating attention, then with the host attention
    {
        x = bf16v(m, "f0/blocks/in_00");
        CaptureAttention cap;
        BlockTaps tp;
        std::vector<std::uint16_t> t_n1, t_ain, t_qkv, t_att, t_op, t_n2i, t_mlp;
        tp.norm1_out = &t_n1;
        tp.attn_in = &t_ain;
        tp.qkv = &t_qkv;
        tp.attn = &t_att;
        tp.out_proj = &t_op;
        tp.norm2_in = &t_n2i;
        tp.mlp_out = &t_mlp;
        block_forward_host(b, mv, mrow, rope, x, S, lin, cap, &tp);
        std::printf("  info  chain norm1 %s\n", compare_bf16(t_n1, bf16v(m, "f0/b0/norm1_out")).str().c_str());
        std::printf("  info  chain attn_in %s\n", compare_bf16(t_ain, bf16v(m, "f0/b0/attn_in")).str().c_str());
        std::printf("  info  chain qkv %s\n", compare_bf16(t_qkv, bf16v(m, "f0/b0/qkv_proj_out")).str().c_str());
        std::printf("  info  chain attn %s\n", compare_bf16(t_att, bf16v(m, "f0/attn/b00_out")).str().c_str());
        std::printf("  info  chain out_proj %s\n", compare_bf16(t_op, bf16v(m, "f0/b0/out_proj_out")).str().c_str());
        std::printf("  info  chain norm2_in %s\n", compare_bf16(t_n2i, bf16v(m, "f0/b0/norm2_in")).str().c_str());
        std::printf("  info  chain mlp_out %s\n", compare_bf16(t_mlp, bf16v(m, "f0/b0/mlp_out")).str().c_str());
        const Cmp cm = compare_bf16(x, bf16v(m, "f0/blocks/out_00"));
        check(cm.rel_l2 < 5e-3, "block 0 end to end (capture-emulating attention) " + cm.str());
    }
    x = bf16v(m, "f0/blocks/in_00");
    t0 = now_s();
    block_forward_host(b, mv, mrow, rope, x, S, lin, att);
    { const Cmp cm = compare_bf16(x, bf16v(m, "f0/blocks/out_00")); check(cm.rel_l2 < 1e-2, "block 0 end to end (fp64 attention) " + cm.str() + " (" + std::to_string(now_s() - t0) + " s)"); }
}

}  // namespace

int run_forward_tests(const SafeTensors& st, const GlobalWeights& g, const std::string& set, bool chained);
int run_refiner_test(const SafeTensors& st, const GlobalWeights& g, const std::string& set);
int run_g2_drift_test(const SafeTensors& st, const GlobalWeights& g, const std::string& set);
int run_sampler_tests(const SafeTensors* st, const GlobalWeights* g, const std::string& set, bool with_model);

int main(int argc, char** argv) {
    std::map<std::string, bool> want;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    for (int i = 1; i < argc; ++i) want[argv[i]] = true;
    if (want.empty()) want["unit"] = true;
    {
        bool needs = false;
        for (const auto& kv : want) needs |= kv.second && kv.first != "unit";
        if (needs && !assets_present(true)) {
            std::printf("SKIPPED: fixture modes need H3_FIXTURES (with FORMAT.md) and H3_CKPT\n");
            if (want["unit"]) {
                test_unit();
                std::printf("unit: %s (%d failures)\n", g_failures ? "FAILED" : "PASSED", g_failures);
                if (g_failures) return 1;
            }
            return kSkipped;
        }
    }
    try {
        if (want["unit"]) test_unit();
        if (want["sampler-unit"]) run_sampler_tests(nullptr, nullptr, "dit_t2va_small", false);
        const bool need_model = want["weights"] || want["layout"] || want["block0"] || want["block0-mid"] || want["block0-fl2va"] ||
                                want["refiner"] || want["forward"] || want["forward-tf"] || want["forward-fl2va"] || want["sampler"] || want["g2-drift"] || want["forward-tf-fl2va"] || want["g2-drift-fl2va"] || want["g2"];
        if (need_model) {
            SafeTensors st(ckpt_path());
            const double t0 = now_s();
            const GlobalWeights g = load_global(st);
            std::printf("global weights loaded in %.2f s\n", now_s() - t0);
            if (want["weights"]) test_weights(st, g);
            if (want["layout"]) {
                test_layout(g, "dit_t2va_small");
                test_layout(g, "dit_fl2va_small");
                test_layout(g, "dit_t2va_mid");
            }
            if (want["block0"]) test_block0(st, g, "dit_t2va_small");
            if (want["block0-fl2va"]) test_block0(st, g, "dit_fl2va_small");
            if (want["block0-mid"]) test_block0(st, g, "dit_t2va_mid");
            if (want["refiner"]) run_refiner_test(st, g, "dit_t2va_small");
            if (want["forward"]) run_forward_tests(st, g, "dit_t2va_small", true);
            if (want["forward-tf"]) run_forward_tests(st, g, "dit_t2va_small", false);
            if (want["forward-fl2va"]) run_forward_tests(st, g, "dit_fl2va_small", true);
            if (want["sampler"]) run_sampler_tests(&st, &g, "dit_t2va_small", true);
            if (want["g2-drift"]) run_g2_drift_test(st, g, "dit_t2va_small");
            if (want["g2-drift-fl2va"]) run_g2_drift_test(st, g, "dit_fl2va_small");
            if (want["g2"]) {  // the whole G2 gate on both small sets (H3_ATTN selects the attention)
                run_forward_tests(st, g, "dit_t2va_small", false);
                run_forward_tests(st, g, "dit_fl2va_small", false);
                run_g2_drift_test(st, g, "dit_t2va_small");
                run_g2_drift_test(st, g, "dit_fl2va_small");
            }
            if (want["forward-tf-fl2va"]) run_forward_tests(st, g, "dit_fl2va_small", false);
        }
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
