// MiniMax H3: whole-forward, refiner and sampler gates of the host reference against the reference capture.
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <random>
#include <string>

#include "h3/block/host_ref.hpp"
#include "h3/block/layout.hpp"
#include "h3/block/sampler.hpp"
#include "h3/block/torch_rng.hpp"
#include "h3/block/weights.hpp"
#include "test_util.hpp"

using namespace strix::h3::block;
using namespace h3t;

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

std::vector<float> widen(const std::vector<std::uint16_t>& b) {
    std::vector<float> f(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) f[i] = bf(b[i]);
    return f;
}
std::vector<std::uint16_t> narrow(const std::vector<float>& f) {
    std::vector<std::uint16_t> b(f.size());
    for (std::size_t i = 0; i < f.size(); ++i) b[i] = tob(f[i]);
    return b;
}

struct Case {
    fx::Manifest m;
    PackedLayout L;
    std::vector<std::int64_t> tags;
    std::vector<float> cond_video;
};
Case make_case(const std::string& set) {
    Case c{fx::Manifest::load(fixtures_root() + "/" + set), {}, {}, {}};
    const auto& sig = c.m.meta.at("layout").at("signature").arr();
    std::vector<Keyframe> kfs;
    const int nkf = static_cast<int>(c.m.meta.at("config").at("keyframes").num());
    for (int i = 0; i < nkf; ++i) {
        const std::string n = "in/keyframe" + std::to_string(i) + "_latent";
        const auto& sh = c.m.find(n)->shape;  // [1,24,vt,H,W]
        kfs.push_back({0.0, static_cast<std::uint32_t>(sh[2]), 0});
        const auto lat = f32v(c.m, n);
        // every keyframe restarts the same noise stream (seed), dumped once for keyframe 0
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

void print_g2(const std::string& what, const Cmp& c, bool gate) {
    const bool ok = c.rel_l2 <= 1e-2 && c.cosine >= 0.9999;
    char b[160];
    std::snprintf(b, sizeof b, "G2 %-28s rel_l2=%.3e cos=%.7f max_abs=%.3g", what.c_str(), c.rel_l2, c.cosine, c.max_abs);
    if (gate) check(ok, b);
    else std::printf("  info  %s\n", b);
}

}  // namespace

int run_refiner_test(const SafeTensors&, const GlobalWeights& g, const std::string& set) {
    std::printf("[refiner %s]\n", set.c_str());
    const fx::Manifest m = fx::Manifest::load(fixtures_root() + "/" + set);
    const auto text = bf16v(m, "in/text_states");
    const std::uint32_t L = static_cast<std::uint32_t>(m.find("in/text_states")->shape[1]);
    HostAttention att;
    std::vector<std::vector<std::uint16_t>> stages;
    const double t0 = now_s();
    const auto refined = text_refine_host(g, text.data(), L, att, &stages);
    std::printf("  refiner host %.1f s\n", now_s() - t0);
    const char* names[] = {"text/condition_proj_out", "text/refiner_block0_out", "text/refiner_block1_out"};
    for (int i = 0; i < 3; ++i) {
        const Cmp c = compare_bf16(stages[i], bf16v(m, names[i]));
        check(c.rel_l2 < 1e-2 && c.cosine > 0.9999, std::string(names[i]) + " " + c.str());
    }
    const Cmp c = compare_bf16(refined, bf16v(m, "text/refined"));
    check(c.rel_l2 < 1e-2 && c.cosine > 0.9999, "text/refined " + c.str());
    return 0;
}

// Sampler arithmetic alone: replay the dumped model outputs through our sampler (bitwise on every state).
int run_sampler_tests(const SafeTensors* st, const GlobalWeights* g, const std::string& set, bool with_model) {
    (void)st;
    (void)g;
    (void)with_model;
    std::printf("[sampler-unit %s]\n", set.c_str());
    const fx::Manifest m = fx::Manifest::load(fixtures_root() + "/" + set);
    {
        // seeded noise: the global generator (seed) draws the video noise then the audio noise
        TorchCpuGenerator gen(static_cast<std::uint64_t>(m.meta.at("rng").at("seed").num()));
        const auto nv = gen.randn(f32v(m, "noise/video").size());
        const auto na = gen.randn(f32v(m, "noise/audio").size());
        const Cmp cv = compare_f32(nv, f32v(m, "noise/video")), ca = compare_f32(na, f32v(m, "noise/audio"));
        check(cv.mism == 0, "seeded video noise bitwise " + cv.str());
        check(ca.mism == 0, "seeded audio noise bitwise (same generator) " + ca.str());
        const fx::Manifest mf = fx::Manifest::load(fixtures_root() + "/dit_fl2va_small");
        TorchCpuGenerator g2(static_cast<std::uint64_t>(mf.meta.at("config").at("seed").num()));
        const auto ref = f32v(mf, "noise/cond_video_rows0");
        const Cmp cc = compare_f32(g2.randn(ref.size()), ref);
        check(cc.mism == 0, "keyframe cond-row noise (fresh generator, seed) bitwise " + cc.str());
        TorchCpuGenerator g3(static_cast<std::uint64_t>(mf.meta.at("config").at("seed").num()));
        const Cmp c3 = compare_f32(g3.randn(f32v(mf, "noise/video").size()), f32v(mf, "noise/video"));
        check(c3.mism == 0, "fl2va video noise bitwise " + c3.str());
    }
    const auto s20 = f32v(m, "sampler/sigmas_simple20");
    check(simple_sigmas(12.0F, 20) == s20, "simple schedule 20 steps, shift 12 bitwise");
    check(simple_sigmas(12.0F, 8) == f32v(m, "sampler/sigmas_simple8"), "simple schedule 8 steps bitwise");
    const std::vector<float> sig(s20.begin(), s20.begin() + 3);
    AvState x = initial_state(sig[0], f32v(m, "noise/video"), f32v(m, "noise/audio"));
    check(x.video == f32v(m, "sampler/step0_x_video") && x.audio == f32v(m, "sampler/step0_x_audio"), "initial state bitwise");
    int fwd = 0;
    AvModelFn model = [&](const AvState& xi, float sigma) {
        const std::string pre = "f" + std::to_string(fwd++);
        check(narrow(xi.video) == bf16v(m, pre + "/model_in/video_x") && narrow(xi.audio) == bf16v(m, pre + "/model_in/audio_x_carried"),
            pre + " model input (bf16 cast) bitwise");
        check(sigma * 1000.0F == f32v(m, pre + "/model_in/timestep")[0], pre + " timestep");
        // model-level carry conversion from the stream outputs
        const PackedLayout L = PackedLayout::build(256, 2, 16, 16, 8);
        const ForwardPlan p = ForwardPlan::build(L, sigma, {});
        check(uncarry_audio(narrow(xi.audio), p) == bf16v(m, pre + "/stream_in/audio_x"), pre + " audio un-carry bitwise");
        const auto va = carry_audio_velocity(narrow(xi.audio), bf16v(m, pre + "/stream_out/audio_neg_v"), p, 4.0F);
        check(va == bf16v(m, pre + "/model_out/audio_v_carried"), pre + " audio velocity carry bitwise");
        return AvState{widen(bf16v(m, pre + "/model_out/video_v")), widen(bf16v(m, pre + "/model_out/audio_v_carried"))};
    };
    SamplerTrace tr;
    const AvState fin = sample(model, x, sig, SamplerKind::res_multistep, &tr);
    for (int i = 0; i < 2; ++i) {
        const std::string s = "sampler/step" + std::to_string(i);
        check(tr.x[i].video == f32v(m, s + "_x_video") && tr.x[i].audio == f32v(m, s + "_x_audio"), s + " x bitwise");
        check(tr.denoised[i].video == f32v(m, s + "_denoised_video") && tr.denoised[i].audio == f32v(m, s + "_denoised_audio"),
            s + " denoised bitwise");
    }
    const AvState out = finalize(fin, sig.back(), 4.0F);
    check(out.video == f32v(m, "sampler/final_video"), "final video (res_multistep 2nd order + unscale) bitwise");
    check(out.audio == f32v(m, "sampler/final_audio"), "final audio bitwise");
    {
        // resumable form: step 0, then step 1 from the kept carry (the clip runner's checkpoints) == one call
        fwd = 0;
        SamplerCarry c;
        c.x = x;
        sample_steps(model, c, sig, SamplerKind::res_multistep, 1);
        SamplerCarry r = c;  // as restored from a checkpoint
        sample_steps(model, r, sig, SamplerKind::res_multistep, 2);
        check(r.next == 2 && r.x.video == fin.video && r.x.audio == fin.audio, "sample_steps split 1 + 1 == sample bitwise");
    }
    return 0;
}

// G2 gate, part (c): the chained velocity error vs the capture must be at most
// 2x the one-ulp drift of our own forward, the drift being the MEDIAN over 3 single-ulp perturbations of one text value
// (positions drawn from seeds 1, 2, 3), measured with the same attention mode (H3_ATTN). Video and audio separately.
// Parts (a) teacher-forced per-block <= 1e-2 and (b) the final layer from the captured block-49 residual are asserted
// in run_forward_tests (teacher-forced mode).
int run_g2_drift_test(const SafeTensors& st, const GlobalWeights& g, const std::string& set) {
    const bool cap = env_or("H3_ATTN", "f64") == "capture";
    std::printf("[g2-drift %s, attention %s]\n", set.c_str(), cap ? "capture-emulating" : "fp64");
    Case c = make_case(set);
    const fx::Manifest& m = c.m;
    FileBlockSource blocks(st, g.basis, true);
    HostLinear lin;
    std::unique_ptr<AttentionOp> attp = cap ? std::unique_ptr<AttentionOp>(new CaptureAttention) : std::unique_ptr<AttentionOp>(new HostAttention);
    const ForwardPlan p = ForwardPlan::build(c.L, f32v(m, "sampler/sigmas_simple20")[0], c.tags);
    StreamInputs in;
    in.T = c.L.latent_t;
    in.H = c.L.latent_h;
    in.W = c.L.latent_w;
    in.audio_t = c.L.audio_t;
    in.video_x = bf16v(m, "f0/stream_in/video_x");
    in.audio_x = bf16v(m, "f0/stream_in/audio_x");
    in.text = bf16v(m, "f0/model_in/context");
    in.cond_video = c.cond_video;
    const StreamOutputs base = stream_forward_host(g, blocks, c.L, p, in, lin, *attp);
    const Cmp ev = compare_bf16(base.video_neg_v, bf16v(m, "f0/stream_out/video_neg_v"));
    const Cmp ea = compare_bf16(base.audio_neg_v, bf16v(m, "f0/stream_out/audio_neg_v"));
    std::vector<double> dv, da;
    for (std::uint32_t seed = 1; seed <= 3; ++seed) {
        // A one-ulp change can be absorbed entirely (its RMSNorm output and its gated residual both round back to the
        // same bf16): such a draw is not a perturbation of the forward, so the seed's generator draws again until the
        // block-0 residual changes (checked with block 0 alone; skipped draws are reported).
        std::mt19937 rng(seed);
        StreamInputs pin;
        std::size_t pos = 0;
        for (unsigned tries = 0;; ++tries) {
            pin = in;
            pos = std::uniform_int_distribution<std::size_t>(0, pin.text.size() - 1)(rng);
            const bool up = (rng() & 1U) != 0U;
            std::uint16_t& v = pin.text[pos];
            v = static_cast<std::uint16_t>(up || (v & 0x7fffU) == 0 ? v + 1 : v - 1);  // one ulp in magnitude
            const auto b0 = blocks.block(0);
            const auto te = curve_t_emb(g, p.t_vals);
            const ModVectors mv = mod_vectors(adaln_block_f32(*b0, te, p.n_t()), p.n_mod_rows());
            const RopeTable rope = rope_table(rope_angles(c.L.position_ids, g.inv_freq));
            const auto mrow = mod_row_of(p.mod_segments, c.L.seq_len);
            std::vector<std::uint16_t> xa = embed_and_pack(g, c.L, in), xb = embed_and_pack(g, c.L, pin);
            block_forward_host(*b0, mv, mrow, rope, xa, c.L.seq_len, lin, *attp);
            block_forward_host(*b0, mv, mrow, rope, xb, c.L.seq_len, lin, *attp);
            if (xa != xb) break;
            std::printf("  info  seed %u draw %u: text value %zu absorbed (block-0 residual unchanged), drawing again\n", seed, tries, pos);
        }
        const StreamOutputs o = stream_forward_host(g, blocks, c.L, p, pin, lin, *attp);
        const Cmp cv = compare_bf16(o.video_neg_v, base.video_neg_v), ca = compare_bf16(o.audio_neg_v, base.audio_neg_v);
        dv.push_back(cv.rel_l2);
        da.push_back(ca.rel_l2);
        std::printf("  info  perturbation seed %u: text value %zu (row %zu) -> drift video rel_l2 %.3e cos %.7f, audio %.3e cos %.7f\n", seed,
            pos, pos / kHidden, cv.rel_l2, cv.cosine, ca.rel_l2, ca.cosine);
    }
    auto median3 = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[1];
    };
    const double mv = median3(dv), ma = median3(da);
    char b1[200], b2[200];
    std::snprintf(b1, sizeof b1, "G2(c) %s video: chained rel_l2 %.3e (cos %.7f) <= 2 x median drift %.3e = %.3e", set.c_str(), ev.rel_l2,
        ev.cosine, mv, 2 * mv);
    std::snprintf(b2, sizeof b2, "G2(c) %s audio: chained rel_l2 %.3e (cos %.7f) <= 2 x median drift %.3e = %.3e", set.c_str(), ea.rel_l2,
        ea.cosine, ma, 2 * ma);
    check(ev.rel_l2 <= 2 * mv, b1);
    check(ea.rel_l2 <= 2 * ma, b2);
    return 0;
}

// Whole forward through the sampler: the host model runs both captured forwards (f0: Euler step, f1: res_multistep
// step); per-block residual table, final rows, G2 on the velocities, sampler states. chained = our own residual stream
// (the real gate); otherwise teacher forced (every block starts from the captured residual of the previous block).
int run_forward_tests(const SafeTensors& st, const GlobalWeights& g, const std::string& set, bool chained) {
    std::printf("[forward %s %s]\n", set.c_str(), chained ? "chained" : "teacher-forced");
    Case c = make_case(set);
    const fx::Manifest& m = c.m;
    FileBlockSource blocks(st, g.basis, true);
    HostLinear lin;
    // H3_ATTN=capture: the capture-emulating attention (isolates every other op); default fp64 exact softmax.
    const bool cap = env_or("H3_ATTN", "f64") == "capture";
    std::unique_ptr<AttentionOp> attp = cap ? std::unique_ptr<AttentionOp>(new CaptureAttention) : std::unique_ptr<AttentionOp>(new HostAttention);
    AttentionOp& att = *attp;
    std::printf("  attention: %s\n", cap ? "capture-emulating (bf16 scores)" : "fp64 exact softmax");
    const auto text = bf16v(m, "f0/model_in/context");
    const auto s20 = f32v(m, "sampler/sigmas_simple20");
    const std::vector<float> sig(s20.begin(), s20.begin() + 3);
    const float audio_scale = static_cast<float>(m.meta.at("audio_scale").num());
    double tf_max = 0.0;
    // G2(a) bound: 1e-2 with the capture-mimicking attention, 2.5e-2 with exact / flash attention
    // (the capture's CPU attention rounds its scores to bf16: 6e-3 .. 1e-2 off exact per call).
    const double tf_bound = cap ? 1e-2 : 2.5e-2;
    int fwd = 0;
    AvModelFn model = [&](const AvState& xi, float sigma) {
        const std::string pre = "f" + std::to_string(fwd++);
        std::printf(" forward %s sigma=%.6f\n", pre.c_str(), sigma);
        const ForwardPlan p = ForwardPlan::build(c.L, sigma, c.tags);
        StreamInputs in;
        in.T = c.L.latent_t;
        in.H = c.L.latent_h;
        in.W = c.L.latent_w;
        in.audio_t = c.L.audio_t;
        // every forward is gated on the captured model inputs (G2 isolates the forward; the sampler arithmetic is
        // gated bitwise in sampler-unit); report how far our own sampler state drifted
        in.video_x = bf16v(m, pre + "/model_in/video_x");
        const auto a_carried = bf16v(m, pre + "/model_in/audio_x_carried");
        std::printf("  info  %s own sampler input vs captured: video %s\n", pre.c_str(), compare_bf16(narrow(xi.video), in.video_x).str().c_str());
        in.audio_x = uncarry_audio(a_carried, p);
        in.text = text;
        in.cond_video = c.cond_video;
        std::vector<std::uint16_t> h_in;
        std::vector<std::vector<std::uint16_t>> res;
        FinalRows fr;
        ForwardTaps taps;
        taps.h_in = &h_in;
        taps.blocks = &res;
        taps.final_rows = &fr;
        std::printf("   block  mism%%     max_abs     ref_max     rel_l2      cos\n");
        taps.after_block = [&](std::uint32_t i, std::vector<std::uint16_t>& x) {
            char nm[64];
            std::snprintf(nm, sizeof nm, "%s/blocks/out_%02u", pre.c_str(), i);
            if (!m.find(nm)) return;
            const auto ref = bf16v(m, nm);
            const Cmp cm = compare_bf16(x, ref);
            std::printf("   %5u %6.2f %11.4g %11.4g  %.3e  %.8f\n", i, 100.0 * static_cast<double>(cm.mism) / static_cast<double>(cm.n), cm.max_abs, cm.ref_max, cm.rel_l2, cm.cosine);
            std::fflush(stdout);
            if (!chained) {
                char b[160];
                std::snprintf(b, sizeof b, "G2(a) %s teacher-forced block %u rel_l2 %.3e <= %.1e", pre.c_str(), i, cm.rel_l2, tf_bound);
                if (cm.rel_l2 > tf_bound) check(false, b);
                tf_max = std::max(tf_max, cm.rel_l2);
                x = ref;
            }
        };
        const double t0 = now_s();
        const StreamOutputs so = stream_forward_host(g, blocks, c.L, p, in, lin, att, &taps);
        std::printf("   forward %.1f s, inexact rotation groups so far %llu\n", now_s() - t0, static_cast<unsigned long long>(lin.inexact_groups()));
        {
            const Cmp cm = compare_bf16(h_in, bf16v(m, pre + "/blocks/in_00"));
            check(cm.mism * 10000 <= cm.n, pre + " embed (in_00) " + cm.str());
        }
        if (chained) {
            const Cmp last = compare_bf16(res.back(), bf16v(m, pre + "/blocks/out_49"));
            check(last.rel_l2 <= 1e-2, pre + " residual after block 49 " + last.str());
        }
        const Cmp fv = compare_f32(fr.video, f32v(m, pre + "/final/video_rows"));
        const Cmp fa = compare_f32(fr.audio, f32v(m, pre + "/final/audio_rows"));
        std::printf("  info  %s final video rows %s\n  info  %s final audio rows %s\n", pre.c_str(), fv.str().c_str(), pre.c_str(), fa.str().c_str());
        if (!chained) {
            check(tf_max <= tf_bound, "G2(a) " + pre + " teacher-forced: all 50 blocks rel_l2 <= " + (cap ? std::string("1e-2") : std::string("2.5e-2")) +
                " (max " + std::to_string(tf_max) + ")");
            tf_max = 0.0;
            // (b) the final layer from the captured block-49 residual: bf16 velocities equal to the capture except
            // rare 1-ulp casts of fp32 head values (the library's fp32 head order is not reproduced; ours is fp64)
            const Cmp bv = compare_bf16(so.video_neg_v, bf16v(m, pre + "/stream_out/video_neg_v"));
            const Cmp ba = compare_bf16(so.audio_neg_v, bf16v(m, pre + "/stream_out/audio_neg_v"));
            check(bv.mism * 1000 <= bv.n && bv.rel_l2 <= 1e-5 && ba.mism * 1000 <= ba.n && ba.rel_l2 <= 1e-5,
                "G2(b) " + pre + " final layer from the captured residual: video mism " + std::to_string(bv.mism) + "/" + std::to_string(bv.n) +
                    ", audio mism " + std::to_string(ba.mism) + "/" + std::to_string(ba.n) + " (1-ulp head casts)");
        }
        print_g2(pre + " video velocity", compare_bf16(so.video_neg_v, bf16v(m, pre + "/stream_out/video_neg_v")), chained);
        print_g2(pre + " audio velocity", compare_bf16(so.audio_neg_v, bf16v(m, pre + "/stream_out/audio_neg_v")), chained);
        const auto va = carry_audio_velocity(a_carried, so.audio_neg_v, p, audio_scale);
        print_g2(pre + " audio velocity (carried)", compare_bf16(va, bf16v(m, pre + "/model_out/audio_v_carried")), chained);
        return AvState{widen(so.video_neg_v), widen(va)};
    };
    AvState x = initial_state(sig[0], f32v(m, "noise/video"), f32v(m, "noise/audio"));
    SamplerTrace tr;
    const AvState fin = finalize(sample(model, x, sig, SamplerKind::res_multistep, &tr), sig.back(), audio_scale);
    for (int i = 0; i < 2; ++i) {
        const std::string s = "sampler/step" + std::to_string(i);
        print_g2(s + " denoised video", compare_f32(tr.denoised[i].video, f32v(m, s + "_denoised_video")), chained);
        print_g2(s + " denoised audio", compare_f32(tr.denoised[i].audio, f32v(m, s + "_denoised_audio")), chained);
    }
    print_g2("final video", compare_f32(fin.video, f32v(m, "sampler/final_video")), chained);
    print_g2("final audio", compare_f32(fin.audio, f32v(m, "sampler/final_audio")), chained);
    return 0;
}
