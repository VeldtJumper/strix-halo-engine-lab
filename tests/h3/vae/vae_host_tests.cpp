// CPU tests of the H3 VAE package: decode plan, blending (bitwise vs the reference frames), and the
// host reference decoder ops against seeded fixtures. Fixture and weight tests need H3_VAE_FIXTURES (fixture root)
// and H3_VAE_WEIGHTS (video VAE fp16 safetensors); unset or missing -> reported SKIPPED, never PASS.
// --require (qualification) turns every SKIPPED group into a failure.
// --quick skips the full-layer test (one 36-layer block on ~1.8 K tokens on the CPU).

#include <algorithm>
#include <cmath>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "h3/vae/vae_plan.hpp"
#include "h3/vae/vae_ref.hpp"
#include "h3/vae/vae_decoder.hpp"
#include "h3/vae/vae_shard.hpp"
#include "h3/vae/vae_threads.hpp"

using namespace strix::h3::vae;

namespace {
int failures = 0, skipped = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
std::string env_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v && *v ? v : fallback;
}
void report(const char* name, const Diff& d) {
    std::printf("  %-28s n=%zu mismatch=%zu max_abs=%.3g rel_l2=%.3g\n", name, d.n, d.mismatches, d.max_abs, d.rel_l2());
}
H16 transpose_heads(const H16& bhld, int L) {  // [32][L][64] -> [L][2048]
    H16 out(bhld.size());
    for (int h = 0; h < kHeads; ++h)
        for (int r = 0; r < L; ++r)
            std::memcpy(out.data() + std::size_t(r) * kDim + h * kHeadDim, bhld.data() + (std::size_t(h) * L + r) * kHeadDim, kHeadDim * 2);
    return out;
}

void test_workers() {
    bool rethrown = false;
    {
        Workers w;
        for (int i = 0; i < 4; ++i) w.run([i] { if (i == 2) throw std::runtime_error("boom"); });
        try { w.join(); } catch (const std::runtime_error&) { rethrown = true; }
    }
    {   // never joined explicitly: the destructor joins, a pending exception is dropped, no terminate
        Workers w;
        w.run([] { throw std::runtime_error("ignored"); });
    }
    bool blend_thrown = false;
    {
        const DecodePlan p = make_plan(12, 20, 20);
        std::vector<Pix16> outs(std::size_t(p.calls()));
        for (auto& o : outs) { o.C = 3; o.F = 28; o.Hh = 256; o.Ww = 256; o.v.assign(std::size_t(3) * 28 * 256 * 256, 0); }
        inject_failure(FaultStage::chunk_assembly, 1);
        try { (void)blend_calls(p, outs); } catch (const std::runtime_error&) { blend_thrown = true; }
        inject_failure(FaultStage::chunk_assembly, -1);
    }
    check(rethrown && blend_thrown, "worker exceptions rethrown after join (blend chunk worker included), destructor joins without terminate");
}

void test_metrics() {
    check(nan_to_num16(0x7e00U) == 0 && nan_to_num16(0x7c00U) == 0x7bffU && nan_to_num16(0xfc00U) == 0xfbffU && nan_to_num16(0x3c00U) == 0x3c00U,
          "nan_to_num16: NaN -> 0, +-Inf -> +-65504, finite unchanged");
    const std::vector<float> a = {0.5F, 0.25F}, b = {0.5F, NAN}, c = {0.5F};
    const Diff dn = diff_f32(b, a), ds = diff_f32(c, a), ok = diff_f32(a, a);
    check(!dn.valid() && dn.psnr(1.0) == -INFINITY && !(dn.rel_l2() < 1.0) && !ds.valid() && !ds.bitwise() && ok.bitwise() && ok.psnr(1.0) >= 999.0,
          "metrics: NaN or length mismatch can never pass a threshold");
}

void test_plan() {
    const DecodePlan p = make_plan(37, 48, 84);
    check(p.num_chunks == 7 && p.pad_tokens == 0 && p.out_frames == 124, "plan 37x48x84: 7 chunks, 124 frames");
    check(p.ty.start == std::vector<int>({0, 160, 336, 512}) && p.ty.overlap == std::vector<int>({96, 80, 80}), "plan 768 px rows: 4 tiles");
    check(p.tx.start == std::vector<int>({0, 176, 352, 528, 704, 896, 1088}) && p.tx.overlap == std::vector<int>({80, 80, 80, 80, 64, 64}),
          "plan 1344 px columns: 7 tiles");
    check(p.calls() == 196, "plan 37x48x84: 196 decoder calls (28 tiles x 7 chunks)");
    const DecodePlan s = make_plan(12, 20, 20);
    check(s.num_chunks == 2 && s.out_frames == 39 && s.ty.overlap == std::vector<int>({192}), "plan 12x20x20");
    const DecodePlan t2 = make_plan(2, 16, 16);
    check(t2.pad_tokens == 5 && t2.num_chunks == 1 && t2.out_frames == 5, "plan 2x16x16 (padded chunk)");
    check(make_plan(67, 48, 84).num_chunks == 13 && make_plan(67, 48, 84).out_frames == 226, "plan 67x48x84");
    const auto r0 = partition(p, 0, 2), r1 = partition(p, 1, 2);
    check(r0.first == 0 && r0.last == r1.first && r1.last == 196, "two-node partition covers the calls once");
}

// Two-node gather: pieces of a call range from per-call outputs (canonical order), as a rank would emit them.
std::vector<DecodePiece> pieces_of(const DecodePlan& p, const std::vector<Pix16>& outs, CallRange r) {
    std::vector<DecodePiece> v;
    const int nx = int(p.tx.start.size());
    for (const PieceSpan& sp : piece_spans(p, r)) {
        DecodePiece x;
        x.chunk = sp.chunk; x.call_first = sp.call_first; x.call_last = sp.call_last; x.whole = sp.whole;
        if (sp.whole) x.canvas = assemble_chunk(p, [&](int i, int j) { return outs[std::size_t(sp.call_first + i * nx + j)]; });
        else x.tiles.assign(outs.begin() + sp.call_first, outs.begin() + sp.call_last);
        v.push_back(std::move(x));
    }
    return v;
}
// Frames through the assembler for a 2-rank cut: rank 1's pieces go through pack/unpack; order 0 canonical,
// 1 reversed, 2 rank 1 first, 3 shuffled.
std::vector<float> gathered(const DecodePlan& p, const std::vector<Pix16>& outs, int cut, int order) {
    auto all = pieces_of(p, outs, {0, cut});
    auto r1 = pieces_of(p, outs, {cut, p.calls()});
    std::vector<DecodePiece> wire;
    for (const auto& x : r1) {
        const auto bytes = pack_piece(x);
        wire.push_back(unpack_piece(p, bytes.data(), bytes.size()));
    }
    if (order == 2) { wire.insert(wire.end(), std::make_move_iterator(all.begin()), std::make_move_iterator(all.end())); all = std::move(wire); }
    else all.insert(all.end(), std::make_move_iterator(wire.begin()), std::make_move_iterator(wire.end()));
    if (order == 1) std::reverse(all.begin(), all.end());
    if (order == 3) { std::mt19937 g{static_cast<unsigned>(cut)}; std::shuffle(all.begin(), all.end(), g); }
    FrameAssembler fa(p);
    for (auto& x : all) fa.push(std::move(x));
    return fa.take();
}

// Synthetic (no fixtures): random tiles for a 3-chunk 2 x 2-tile plan; every cut x 4 arrival orders == assemble_video.
void test_gather() {
    const DecodePlan p = make_plan(17, 20, 20);
    const auto calls = decoder_calls(p);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> ud(-1.2F, 1.2F);
    std::vector<Pix16> outs;
    for (const auto& c : calls) {
        Pix16 t;
        t.C = 3; t.F = 4 * c.nt; t.Hh = 16 * c.ny; t.Ww = 16 * c.nx;
        t.v.resize(std::size_t(3) * t.F * t.Hh * t.Ww);
        for (auto& h : t.v) h = f2h(ud(rng));
        outs.push_back(std::move(t));
    }
    const int nx = int(p.tx.start.size()), ny = int(p.ty.start.size());
    const auto ref = assemble_video(p, [&](int c) { return assemble_chunk(p, [&](int i, int j) { return outs[std::size_t((c * ny + i) * nx + j)]; }); });
    bool ok = true;
    for (int cut = 0; cut <= p.calls(); ++cut)
        for (int order = 0; order < 4; ++order) ok &= diff_f32(gathered(p, outs, cut, order), ref).bitwise();
    check(p.num_chunks == 3 && p.calls() == 12 && ok, "gather: every 2-rank cut x 4 arrival orders (pack/unpack on rank 1) == one node, bitwise (synthetic 3 chunks)");
    const auto s0 = split_calls(p, 0, 2, 1), s1 = split_calls(p, 1, 2, 1), w = split_calls(p, 0, 1);
    check(s0.first == 0 && s0.last == 7 && s1.first == 7 && s1.last == 12 && w.last == 12 && split_calls(make_plan(37, 48, 84), 0, 2).last == 98,
          "split_calls: cut n/2 + shift, world 1 = all");
    bool rej = true;
    {   // duplicate call, frames before completion, malformed bytes
        FrameAssembler fa(p);
        auto pcs = pieces_of(p, outs, {0, 6});
        auto again = pieces_of(p, outs, {5, 6});
        for (auto& x : pcs) fa.push(std::move(x));
        bool t1 = false, t2 = false, t3 = false, t4 = false;
        try { fa.push(std::move(again[0])); } catch (const std::invalid_argument&) { t1 = true; }
        try { (void)fa.take(); } catch (const std::logic_error&) { t2 = true; }
        auto bytes = pack_piece(pieces_of(p, outs, {6, 7})[0]);
        bytes.pop_back();
        try { (void)unpack_piece(p, bytes.data(), bytes.size()); } catch (const std::invalid_argument&) { t3 = true; }
        auto b2 = pack_piece(pieces_of(p, outs, {6, 7})[0]);
        b2[8] = 9;  // chunk index outside the plan
        try { (void)unpack_piece(p, b2.data(), b2.size()); } catch (const std::invalid_argument&) { t4 = true; }
        rej = t1 && t2 && t3 && t4 && fa.chunks_done() == 1;
    }
    check(rej, "gather: duplicate call, early take, truncated / out-of-plan piece bytes rejected");
}

// Blending: the decoder outputs of the fixture's calls -> frames, bitwise vs the reference frames.
void test_blend(const std::string& root, const char* fcase) {
    const std::string dir = root + "/" + fcase;
    if (!file_exists(dir + "/manifest.json")) { std::printf("SKIPPED blend %s (H3_VAE_FIXTURES unset or no fixture)\n", fcase); ++skipped; return; }
    Fixture fx(dir);
    if (!fx.has("tap_call00_out")) { std::printf("SKIPPED blend %s (no per-call outputs)\n", fcase); ++skipped; return; }
    const auto& z = fx.info("z_norm_f32");
    const DecodePlan p = make_plan(int(z.shape[2]), int(z.shape[3]), int(z.shape[4]));
    const auto calls = decoder_calls(p);
    const auto out = assemble_video(p, [&](int c) {
        return assemble_chunk(p, [&](int i, int j) {
            // the reference batches up to 4 tiles of a tile row per decoder call (batch dim = tile column)
            const int nx = int(p.tx.start.size()), per_row = (nx + 3) / 4;
            const int idx = (c * int(p.ty.start.size()) + i) * per_row + j / 4;
            char name[32];
            std::snprintf(name, sizeof name, "tap_call%02d_out", idx);
            const auto& ti = fx.info(name);
            Pix16 t;
            t.C = 3; t.F = int(ti.shape[2]); t.Hh = int(ti.shape[3]); t.Ww = int(ti.shape[4]);
            const auto all = fx.f16(name);
            const std::size_t per = std::size_t(3) * t.F * t.Hh * t.Ww;
            t.v.assign(all.begin() + std::ptrdiff_t(per * std::size_t(j % 4)), all.begin() + std::ptrdiff_t(per * std::size_t(j % 4 + 1)));
            return t;
        });
    });
    const auto ref = fx.f32("frames_f32");
    const Diff d = diff_f32(out, ref);
    report("frames", d);
    check(out.size() == ref.size() && d.bitwise(), std::string("blend ") + fcase + ": frames bitwise from the per-call outputs");
    {   // the same per-call outputs through the two-node gather (cut mid-chunk when possible, reversed arrival)
        std::vector<Pix16> outs;
        const int nx = int(p.tx.start.size()), per_row = (nx + 3) / 4;
        for (const auto& c : calls) {
            const int idx = (c.chunk * int(p.ty.start.size()) + c.ti) * per_row + c.tj / 4;
            char name[32];
            std::snprintf(name, sizeof name, "tap_call%02d_out", idx);
            const auto& ti = fx.info(name);
            Pix16 t;
            t.C = 3; t.F = int(ti.shape[2]); t.Hh = int(ti.shape[3]); t.Ww = int(ti.shape[4]);
            const auto all = fx.f16(name);
            const std::size_t per = std::size_t(3) * t.F * t.Hh * t.Ww;
            t.v.assign(all.begin() + std::ptrdiff_t(per * std::size_t(c.tj % 4)), all.begin() + std::ptrdiff_t(per * std::size_t(c.tj % 4 + 1)));
            outs.push_back(std::move(t));
        }
        bool ok = true;
        for (int cut = 0; cut <= p.calls(); ++cut) ok &= diff_f32(gathered(p, outs, cut, 1), ref).bitwise();
        check(ok, std::string("gather ") + fcase + ": every 2-rank cut (reversed arrival) == reference frames, bitwise");
    }
}

void test_ops(const std::string& root, const std::string& weights, bool quick) {
    const std::string dir = root + "/t7";
    if (!file_exists(dir + "/manifest.json") || !file_exists(weights)) { std::printf("SKIPPED ops (H3_VAE_FIXTURES or H3_VAE_WEIGHTS unset or missing)\n"); ++skipped; return; }
    Fixture fx(dir);
    SafeTensors st(weights);
    const HeadH head = load_head(st);
    const int nt = 7, ny = 16, nx = 16, P = nt * ny * nx, L = P + kSuffix;

    // RoPE table [1, L, 1, 24, 2, 2] = c, -s, s, c
    {
        const RopeTable r = rope_table(nt, ny, nx);
        const auto ref = fx.f16("tap_rope_table");
        std::size_t bad = 0;
        for (int t = 0; t < L; ++t)
            for (int i = 0; i < kRotPairs; ++i) {
                const half_bits* e = ref.data() + (std::size_t(t) * kRotPairs + i) * 4;
                const std::size_t q = std::size_t(t) * kRotPairs + i;
                bad += e[0] != f2h(r.c[q]) || e[1] != f2h(-r.s[q]) || e[2] != f2h(r.s[q]) || e[3] != f2h(r.c[q]);
            }
        std::printf("  rope entries mismatching: %zu of %d\n", bad, L * kRotPairs);
        check(bad == 0, "rope table bitwise");
    }
    // Denormalize + post_quant_conv -> decoder input
    {
        const auto zf = fx.f32("z_norm_f32");
        H16 z16(zf.size());
        for (std::size_t i = 0; i < zf.size(); ++i) z16[i] = f2h(zf[i]);
        const DecodePlan p = make_plan(nt, ny, nx);
        const H16 zd = denormalize_and_pad(p, z16, head.latents_mean.data(), head.latents_std.data());
        const H16 dec_in = post_quant(head, zd, P);
        const Diff d = diff_f16(dec_in, fx.f16("tap_dec_in"));
        report("post_quant", d);
        check(d.valid() && d.max_abs <= 2e-3 && d.mismatches * 100 <= d.n, "denormalize + post_quant_conv (<= 1 % ulp flips)");
    }
    // x_embedder + registers
    {
        const H16 x = embed(head, fx.f16("tap_dec_in"), nt, ny, nx);
        const Diff d = diff_f16(H16(x.begin(), x.begin() + std::ptrdiff_t(std::size_t(P) * kDim)), fx.f16("tap_x_embed"));
        report("x_embed", d);
        const Diff dh = diff_f16(x, fx.f16("tap_h_in"));
        report("h_in", dh);
        check(d.rel_l2() < 1e-3 && dh.rel_l2() < 1e-3, "x_embedder + register tokens");
    }
    // Layer 0 pieces from the fixture's own inputs
    const LayerH l0 = load_layer(st, 0);
    const H16 h_in = fx.f16("tap_h_in");
    {
        const H16 a = rms_norm(h_in, L, kDim, &l0.norm1, 1e-5F);
        const H16 qkv = linear(a, L, l0.qkv);
        const Diff d = diff_f16(qkv, fx.f16("tap_l0_qkv"));
        report("l0 qkv", d);
        check(d.rel_l2() < 2e-3, "layer 0 rms_norm + to_qkv");
    }
    {
        const RopeTable r = rope_table(nt, ny, nx);
        H16 q, k;
        const H16 o = attention(fx.f16("tap_l0_qkv"), L, r, &q, &k);
        const Diff dq = diff_f16(q, transpose_heads(fx.f16("tap_l0_q"), L));
        const Diff dk = diff_f16(k, transpose_heads(fx.f16("tap_l0_k"), L));
        report("l0 q (rms+rope)", dq);
        report("l0 k (rms+rope)", dk);
        check(dq.valid() && dk.valid() && dq.mismatches * 1000 <= dq.n && dk.mismatches * 1000 <= dk.n, "layer 0 per-head rms + split-half rope (<= 0.1 % flips)");
        const Diff da = diff_f16(o, fx.f16("tap_l0_attn"));
        report("l0 attention", da);
        check(da.rel_l2() < 2e-3, "layer 0 attention (fp64 softmax vs reference SDPA)");
    }
    if (!quick) {
        H16 x = h_in;
        layer_forward(l0, x, L, rope_table(nt, ny, nx));
        const Diff d = diff_f16(x, fx.f16("tap_h_after_l0"));
        report("l0 full layer", d);
        check(d.rel_l2() < 2e-3, "layer 0 full block");
    }
    // Head: norm_out + proj_out + unpatchify
    {
        H16 pr;
        const H16 out = head_out(head, fx.f16("tap_h_after_l35"), nt, ny, nx, &pr);
        const auto ref_pr = fx.f16("tap_proj_out");
        const Diff d = diff_f16(pr, H16(ref_pr.begin(), ref_pr.begin() + std::ptrdiff_t(std::size_t(P) * kProj)));
        report("norm_out+proj_out", d);
        check(d.rel_l2() < 2e-3, "norm_out + proj_out");
        const Diff du = diff_f16(unpatchify(H16(ref_pr.begin(), ref_pr.begin() + std::ptrdiff_t(std::size_t(P) * kProj)), nt, ny, nx),
                                 fx.f16("tap_dec_out"));
        check(du.bitwise() && du.n == fx.info("tap_dec_out").numel(), "unpatchify bitwise");
    }
}
}  // namespace

int main(int argc, char** argv) {
    bool quick = false, require = false;  // --require (qualification): a skipped group is a failure
    for (int i = 1; i < argc; ++i) {
        quick |= std::strcmp(argv[i], "--quick") == 0;
        require |= std::strcmp(argv[i], "--require") == 0;
    }
    const std::string root = env_or("H3_VAE_FIXTURES", "");
    const std::string weights = env_or("H3_VAE_WEIGHTS", "");
    test_plan();
    test_blend(root, "t7");
    test_blend(root, "tiles");
    test_blend(root, "odd");
    test_blend(root, "oddtiles");
    test_gather();
    test_metrics();
    test_workers();
    test_ops(root, weights, quick);
    if (require && skipped) {
        std::printf("FAIL qualification: %d groups SKIPPED (missing fixtures or weights)\n", skipped);
        failures += skipped;
    }
    std::printf("%s (%d failures, %d groups SKIPPED)\n", failures ? "FAILED" : skipped ? "PASS WITH SKIPS" : "ALL PASS", failures, skipped);
    return failures ? 1 : 0;
}
