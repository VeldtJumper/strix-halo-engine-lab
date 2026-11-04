// strix-h3-dist-forward: the MiniMax H3 two-node schedule (D1) on the device, one process per rank (or both ranks in
// threads of one process on one GPU with --transport emulate).
//
//   strix-h3-dist-forward --ckpt FILE --transport usb4auto|emulate [--rank R]
//       --case NAME=fixture:DIR[:BLOCKS] | NAME=synth:TEXT,LT,LH,LW,AT[:BLOCKS]   (repeatable)
//       [--forms w1,w2,w2swap] [--timing NAME] [--bench-attn NAME:REPS] [--seed N] [--forward]
//       [--inject-fail KIND:RANK:BLOCK]   failure injection instead of G3 (world 2, first case; KIND = throw | nanq |
//                                         vbig | bound; see run_inject)
//       [--accel-check [--turbo LORA,MERGED_FC2,EGRID[,STRENGTH]] [--fbc THRESHOLD] [--fc1-plain]
//        [--expect-dense NAME=VIDEOHASH:AUDIOHASH]]   FAST-class accelerators (--forward, see run_accel_cases)
//
// Per case and form: per-block FNV-1a of each owned key block's residual rows; G3 lines compare, on each rank, the
// world-2 (and swapped) hashes of the rank's own key block with the same rank's world-1 run, and the two ranks'
// world-1 hashes with each other (exchanged over the communicator). --timing NAME: per-block timing of that case
// (world 2, no per-block downloads). --bench-attn: real-shape attention / exchange bench at that case's shape.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "h3/block/layout.hpp"
#include "h3/block/weights.hpp"
#include "h3/common/fixture.hpp"
#include "h3/dist/device_dist.hpp"
#include "h3/dist/host_dist.hpp"
#include "strix/dist/communicator.hpp"

using namespace strix::h3;
using namespace strix::h3::dist;
namespace fx = strix::h3::fixture;

namespace {

struct CaseSpec {
    std::string name, kind, arg;
    std::uint32_t blocks = 0;
    bool blocks_hashes = true;  // forward mode: per-block residual hashes (synth cases: off, for the wall time)
};
struct Args {
    std::string ckpt, transport = "emulate";
    std::uint32_t rank = 0;
    std::vector<CaseSpec> cases;
    std::vector<std::string> forms{"w1", "w2", "w2swap"};
    std::string timing, bench;
    unsigned bench_reps = 3;
    std::uint64_t seed = 1234;
    bool forward = false;  // whole forward from the stream inputs (else: the block stack from the captured block input)
    std::int32_t inject_rank = -1, inject_block = -1;  // --inject-fail KIND:RANK:BLOCK (failure injection; replaces the G3 run)
    DeviceRunOptions::Inject inject_kind = DeviceRunOptions::Inject::none;
    bool inject_fbc = false;  // --inject-fail fbc:RANK:CALL (FirstBlockCache failure inside after_block0 of call CALL)
    bool accel_check = false, fc1_plain = false;
    std::string turbo_lora, turbo_fc2, turbo_egrid;
    float turbo_strength = 0.70F;
    double fbc_threshold = -1.0;                     // < 0: FirstBlockCache off
    std::map<std::string, std::array<std::uint64_t, 2>> expect_dense;
    unsigned exposure_runs = 0;
    bool profile = false;  // --profile: per-kernel time split of one world-2 forward of the --timing case (see run_profile)
    bool sparse = false;  // --sparse TAU [--sparse-dense-calls N] [--sparse-dense-layers LIST] (identity h3-attn-sparse-thr-v1)
    strix::h3::levers::sparse::Params sparse_params;  // --exposure-runs K: performance qualification of the --timing case (see run_exposure)  // case -> dense velocity hashes (video, audio)
};

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, c);) v.push_back(t);
    return v;
}

struct Case {
    CaseSpec spec;
    block::PackedLayout L;
    block::ForwardPlan p;
    ForwardTables t;
    std::vector<std::uint16_t> x;  // all S rows (block-stack mode)
    std::uint32_t blocks = 0;
    // whole-forward mode: stream inputs (text = raw text states [L][5120], refined on every rank)
    block::StreamInputs in;
    std::vector<std::uint16_t> text_states;
    std::vector<std::int64_t> tags;  // plan inputs (to rebuild the plan at another sigma)
    float sigma0 = 0.9F;
};

template <class T> std::vector<T> tensor_vec(const fx::Manifest& m, const std::string& n) {
    const fx::Tensor t = m.read(n);
    std::vector<T> v(t.bytes.size() / sizeof(T));
    std::memcpy(v.data(), t.bytes.data(), v.size() * sizeof(T));
    return v;
}

Case make_case(const CaseSpec& cs, const block::GlobalWeights& g, std::uint64_t seed, bool forward) {
    Case c;
    c.spec = cs;
    if (cs.kind == "fixture") {
        const fx::Manifest m = fx::Manifest::load(cs.arg);
        const auto& sig = m.meta.at("layout").at("signature").arr();
        c.L = block::PackedLayout::build(static_cast<std::uint32_t>(sig[0].num()), static_cast<std::uint32_t>(sig[1].num()),
            static_cast<std::uint32_t>(sig[2].num()), static_cast<std::uint32_t>(sig[3].num()), static_cast<std::uint32_t>(sig[4].num()));
        const fx::Tensor tt = m.read("in/text_token_tags");
        std::vector<std::int64_t> tags(tt.bytes.size() / 8);
        std::memcpy(tags.data(), tt.bytes.data(), tags.size() * 8);
        bool all_text = true;
        for (auto v : tags) all_text &= v == 1;
        if (all_text) tags.clear();
        const fx::Tensor sg = m.read("sampler/sigmas_simple20");
        float sigma0;
        std::memcpy(&sigma0, sg.bytes.data(), 4);
        c.p = block::ForwardPlan::build(c.L, sigma0, tags);
        c.tags = tags;
        c.sigma0 = sigma0;
        if (forward) {
            c.in.video_x = tensor_vec<std::uint16_t>(m, "f0/stream_in/video_x");
            c.in.audio_x = tensor_vec<std::uint16_t>(m, "f0/stream_in/audio_x");
            c.text_states = tensor_vec<std::uint16_t>(m, "in/text_states");
        } else {
            c.x = tensor_vec<std::uint16_t>(m, "f0/blocks/in_00");
        }
    } else if (cs.kind == "synth") {
        const auto d = split(cs.arg, ',');
        if (d.size() != 5) throw std::invalid_argument("synth:TEXT,LT,LH,LW,AT");
        c.L = block::PackedLayout::build(static_cast<std::uint32_t>(std::stoul(d[0])), static_cast<std::uint32_t>(std::stoul(d[1])),
            static_cast<std::uint32_t>(std::stoul(d[2])), static_cast<std::uint32_t>(std::stoul(d[3])), static_cast<std::uint32_t>(std::stoul(d[4])));
        c.p = block::ForwardPlan::build(c.L, 0.9F, {});
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd(0.0F, 1.0F);
        if (forward) {  // seeded synthetic stream inputs: latents ~ N(0, 1), text states ~ N(0, 1) [L][5120]
            c.in.video_x.resize(std::size_t{24} * c.L.latent_t * c.L.latent_h * c.L.latent_w);
            c.in.audio_x.resize(std::size_t{32} * 2 * c.L.audio_t);
            c.text_states.resize(std::size_t{c.L.text_len} * block::kTextDim);
            for (auto* v : {&c.in.video_x, &c.in.audio_x, &c.text_states})
                for (auto& e : *v) e = block::tob(nd(rng));
        } else {
            c.x.resize(std::size_t{c.L.seq_len} * block::kHidden);
            for (auto& e : c.x) e = block::tob(nd(rng));
        }
    } else {
        throw std::invalid_argument("case kind fixture | synth");
    }
    if (!forward && c.x.size() != std::size_t{c.L.seq_len} * block::kHidden) throw std::runtime_error("case " + cs.name + ": residual size");
    c.in.T = c.L.latent_t;
    c.in.H = c.L.latent_h;
    c.in.W = c.L.latent_w;
    c.in.audio_t = c.L.audio_t;
    c.t = ForwardTables::build(g, c.L, c.p);
    c.blocks = cs.blocks;
    return c;
}

std::vector<std::uint16_t> rows_of(const Case& c, const Split& sp, const Role& r) {
    const std::size_t off = std::size_t{r.row0(sp)} * block::kHidden, n = std::size_t{r.rows(sp)} * block::kHidden;
    return std::vector<std::uint16_t>(c.x.begin() + static_cast<std::ptrdiff_t>(off), c.x.begin() + static_cast<std::ptrdiff_t>(off + n));
}

struct Shared {
    std::unique_ptr<DeviceModel> model;
    block::GlobalWeights g;
    std::vector<Case> cases;
};

std::uint64_t hash16(const std::vector<std::uint16_t>& v) { return fnv1a64(v.data(), v.size() * 2U); }

using Digests = std::vector<std::array<std::uint64_t, 2>>;

// Per-block device digests (every run, also the timing ones): w1 = the reference (and identical on both nodes);
// w2 / w2swap: the owned key block == the reference on every block. A missing reference is a failure.
template <class Gate>
void digest_gates(const std::string& name, const std::string& form, const Role& role, const Digests& d, Digests& ref, std::uint32_t nblocks,
    strix::dist::Communicator* comm, Gate&& gate) {
    if (role.world == 1U) {
        ref = d;
        if (comm) {
            Digests peer(d.size());
            comm->exchange_host(std::as_bytes(std::span(d)), std::as_writable_bytes(std::span(peer)),
                strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xA4U, 0});
            gate(peer == d, name + " w1 per-block device digests identical on both nodes (" + std::to_string(nblocks) + " blocks)");
        }
        return;
    }
    if (ref.size() != d.size()) {
        gate(false, name + " " + form + " NO-REFERENCE (no w1 run of this case)");
        return;
    }
    const unsigned kb = role.block();
    std::uint32_t bad = 0;
    for (std::uint32_t i = 0; i < nblocks; ++i) bad += d[i][kb] != ref[i][kb] || d[i][kb] == 0U;
    gate(bad == 0, name + " " + form + " key block " + std::to_string(kb) + " device digest == w1 after every block (" + std::to_string(nblocks - bad) +
        "/" + std::to_string(nblocks) + ")");
}

// The < 3 % link-exposure gate: (GPU wait in finish + the local partial's slowdown under the exchange) / block time.
template <class Line, class Gate>
void exposure_gate(const std::string& name, double block_ms, double exposed_ms, const DeviceDistRank::AttnBench& b, Line&& line, Gate&& gate) {
    const double cost = exposed_ms + std::max(0.0, b.local_slowdown_ms);
    line("exposure %s: (exposed %.2f ms + paired partial slowdown %.2f ms) / block %.2f ms = %.2f %%", name.c_str(), exposed_ms,
        std::max(0.0, b.local_slowdown_ms), block_ms, 100.0 * cost / block_ms);
    gate(block_ms > 0 && cost / block_ms < 0.03, name + " link exposure < 3 % of the block");
}

template <class Line>
DeviceDistRank::AttnBench run_bench(const std::string& name, const Split& sp, unsigned reps, DeviceDistRank& R, Line&& line) {
    const auto b = R.bench_attention(reps);
    line("bench %s (median of %u): attention alone (local + remote + merge) %.1f ms | local partial alone %.1f ms | local partial "
         "with the %.1f MB exchange in flight %.1f ms (%+.1f %%) | exchange alone %.1f ms (%.2f GB/s per direction) | "
         "start->finish wall with local partial %.1f ms, finish wait %.1f ms",
        name.c_str(), reps, b.attn_ms, b.local_ms, sp.message_bytes() / 1e6, b.local_with_exchange_ms, 100.0 * (b.local_with_exchange_ms / b.local_ms - 1.0),
        b.exchange_ms, sp.message_bytes() / b.exchange_ms / 1e6, b.overlapped_wall_ms, b.finish_wait_ms);
    return b;
}

// Mean over blocks 1.. (block 0 = warm-up) of the w2 timing; returns {block_ms, exposed_ms}.
std::array<double, 2> mean_timing(const std::vector<DeviceBlockTiming>& tm) {
    double sb = 0, se = 0;
    for (std::size_t i = 1; i < tm.size(); ++i) { sb += tm[i].block_ms; se += tm[i].exposed_ms; }
    const double k = tm.size() > 1 ? 1.0 / static_cast<double>(tm.size() - 1) : 0.0;
    return {sb * k, se * k};
}

// Whole forward per case and form: refiner (host, replicated on every rank and form), embedding, blocks, final layer,
// velocity gather. G3: the velocity of every form == w1 on the same rank, w1 identical on both ranks, gemm flush counts
// add up, attention contract telemetry 0 (hard).
int run_forward_cases(const Args& a, std::uint32_t rank, Shared& sh, strix::dist::Communicator* comm) {
    int failures = 0;
    auto line = [&](const char* fmt, auto... v) {
        char b[1400];
        std::snprintf(b, sizeof b, fmt, v...);
        std::printf("r%u %s\n", rank, b);
        std::fflush(stdout);
    };
    auto gate = [&](bool ok, const std::string& what) {
        line("%s %s", ok ? "G3 ok  " : "G3 FAIL", what.c_str());
        if (!ok) ++failures;
    };
    for (const Case& c : sh.cases) {
        const Split sp = Split::of(c.L.seq_len);
        line("forward case %s S=%u cut=%u rows %u + %u blocks %u text %u message %.1f MB", c.spec.name.c_str(), sp.S, sp.cut, sp.rows(0), sp.rows(1),
            c.blocks, c.L.text_len, sp.message_bytes() / 1e6);
        std::vector<std::uint16_t> v1;  // the w1 velocity, complete (video then audio, bf16 bits)
        bool have_v1 = false;
        std::uint64_t flush1 = 0;
        std::vector<std::array<std::uint64_t, 2>> h1;
        Digests d1;
        const bool timing_case = c.spec.name == a.timing;
        std::vector<std::pair<std::string, std::array<double, 2>>> w2_timings;  // world-2 forms of the timing case
        auto cat = [](const block::StreamOutputs& o) {
            std::vector<std::uint16_t> v(o.video_neg_v);
            v.insert(v.end(), o.audio_neg_v.begin(), o.audio_neg_v.end());
            return v;
        };
        for (const auto& form : a.forms) {
            const Role role = form == "w1" ? Role{rank, 1, false} : Role{rank, 2, form == "w2swap"};
            if (role.world == 2U && !comm) continue;
            // replicated refiner: identical inputs and code on every rank and form
            const auto r0 = std::chrono::steady_clock::now();
            block::HostAttention ratt;
            block::StreamInputs in = c.in;  // per rank thread (emulate: the ranks share the case)
            in.text = block::text_refine_host(sh.g, c.text_states.data(), c.L.text_len, ratt);
            const double refine_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count();
            DeviceRunOptions opt;
            opt.hashes = c.spec.blocks_hashes && !timing_case;
            opt.timing = timing_case;
            DeviceDistRank R(*sh.model, sp, role, role.world == 2U ? comm : nullptr);
            const auto f = R.forward(c.L, c.p, in, c.t, c.blocks, opt);
            const std::vector<std::uint16_t> vel = cat(f.out);
            line("%s %s: forward %.3f s (+ refiner %.2f s on the host) = embed %.3f + blocks %.3f + final %.3f + gather %.3f s (%zu B); "
                 "velocity video %016llx audio %016llx (%zu values); text refined %016llx; flushed %llu; attn bound rows %llu%s%s",
                c.spec.name.c_str(), form.c_str(), f.wall_s, refine_s, f.embed_s, f.blocks_s, f.final_s, f.gather_s, f.gather_bytes,
                static_cast<unsigned long long>(hash16(f.out.video_neg_v)), static_cast<unsigned long long>(hash16(f.out.audio_neg_v)), vel.size(),
                static_cast<unsigned long long>(hash16(in.text)), static_cast<unsigned long long>(f.run.flushed_groups),
                static_cast<unsigned long long>(f.run.attn_bound_rows), f.run.comm_stats.empty() ? "" : " | ", f.run.comm_stats.c_str());
            gate(f.run.attn_bound_rows == 0, c.spec.name + " " + form + " attention contract telemetry (bound rows) == 0");
            line("info %s %s: measured attention operand bound %.1f (block %u) <= 16384, max |V| %.3g <= 2^100", c.spec.name.c_str(), form.c_str(),
                f.run.admission_bound, f.run.admission_block, f.run.v_max);
            digest_gates(c.spec.name, form, role, f.run.digest, d1, c.blocks, role.world == 1U ? comm : nullptr, gate);
            if (opt.timing) {
                const auto mt = mean_timing(f.run.timing);
                line("timing %s %s mean(blocks 1..): block %.2f ms exposed %.2f ms (%.2f %%)", c.spec.name.c_str(), form.c_str(), mt[0], mt[1],
                    100.0 * mt[1] / std::max(mt[0], 1e-9));
                {  // exposure distribution (diagnostics): blocks with a GPU wait, the largest waits, local partial range
                    std::vector<double> ex;
                    double lmin = 1e30, lmax = 0;
                    for (std::size_t i = 1; i < f.run.timing.size(); ++i) {
                        ex.push_back(f.run.timing[i].exposed_ms);
                        lmin = std::min(lmin, f.run.timing[i].local_ms);
                        lmax = std::max(lmax, f.run.timing[i].local_ms);
                    }
                    std::sort(ex.begin(), ex.end());
                    const auto waits = std::count_if(ex.begin(), ex.end(), [](double v) { return v > 1.0; });
                    const std::size_t m = ex.size();
                    line("timing %s %s exposure distribution: %lld of %zu blocks wait > 1 ms; max %.1f, p90 %.1f, median %.1f ms; local partial %.1f .. %.1f ms",
                        c.spec.name.c_str(), form.c_str(), static_cast<long long>(waits), m, m ? ex[m - 1] : 0.0, m ? ex[(m * 9) / 10] : 0.0,
                        m ? ex[m / 2] : 0.0, lmin, lmax);
                }
                if (role.world == 2U) w2_timings.emplace_back(form, mt);
            }
            const std::string nv = std::to_string(vel.size());
            if (role.world == 1U) {
                have_v1 = true;
                v1 = vel;
                flush1 = f.run.flushed_groups;
                h1 = f.run.hash;
                if (comm) {  // the complete w1 velocity of the other node
                    std::vector<std::uint16_t> pv(vel.size(), 0);
                    comm->exchange_host(std::as_bytes(std::span(vel)), std::as_writable_bytes(std::span(pv)),
                        strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xA3U, 0});
                    gate(pv == vel, c.spec.name + " w1 velocity identical on both nodes (complete bf16 vectors, " + nv + " values)");
                }
            } else {
                if (!have_v1) gate(false, c.spec.name + " " + form + " NO-REFERENCE (no w1 velocity)");
                else gate(vel == v1, c.spec.name + " " + form + " velocity (video + audio, after the gather) == w1 (complete bf16 vectors, " + nv + " values)");
                std::uint64_t mine = f.run.flushed_groups, peer = 0;
                comm->exchange_host(std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(&peer, 1)),
                    strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xA2U, 0});
                gate(mine + peer == flush1, c.spec.name + " " + form + " flushed groups rank0 + rank1 == 1-node (" + std::to_string(mine) + " + " +
                    std::to_string(peer) + " vs " + std::to_string(flush1) + ")");
                if (opt.hashes && !h1.empty()) {
                    const unsigned kb = role.block();
                    std::uint32_t bad = 0;
                    for (std::uint32_t i = 0; i < c.blocks; ++i) bad += f.run.hash[i][kb] != h1[i][kb];
                    gate(bad == 0, c.spec.name + " " + form + " key block " + std::to_string(kb) + " residual FNV == w1 after every block (" +
                        std::to_string(c.blocks - bad) + "/" + std::to_string(c.blocks) + ")");
                }
                if (!timing_case) {
                    // Runner reuse: one rank object per clip, forward() once per sampler step.
                    // (a) the same inputs again on the same object
                    const auto f2 = R.forward(c.L, c.p, in, c.t, c.blocks, opt);
                    gate(cat(f2.out) == vel && f2.run.digest == f.run.digest && f2.run.attn_bound_rows == 0,
                        c.spec.name + " " + form + " reuse: second forward on the same object == first (complete velocity, digests, bound rows 0)");
                    // (b) another sampler step: latent sign-flipped, sigma halved, an ODD block count (TX / RX parity
                    // ends shifted): the reused object == a fresh object
                    const std::uint32_t nb2 = c.blocks % 2U ? c.blocks : c.blocks - 1U;
                    block::StreamInputs in2 = in;
                    for (auto& e : in2.video_x) e ^= 0x8000U;
                    for (auto& e : in2.audio_x) e ^= 0x8000U;
                    const block::ForwardPlan p2 = block::ForwardPlan::build(c.L, c.sigma0 * 0.5F, c.tags);
                    const ForwardTables t2 = ForwardTables::build(sh.g, c.L, p2);
                    const auto g_reuse = R.forward(c.L, p2, in2, t2, nb2, opt);
                    std::vector<std::uint16_t> v_fresh;
                    Digests d_fresh;
                    std::uint64_t b_fresh = 1;
                    {
                        DeviceDistRank R2(*sh.model, sp, role, comm);
                        const auto g_fresh = R2.forward(c.L, p2, in2, t2, nb2, opt);
                        v_fresh = cat(g_fresh.out);
                        d_fresh = g_fresh.run.digest;
                        b_fresh = g_fresh.run.attn_bound_rows;
                    }
                    const std::vector<std::uint16_t> v_reuse = cat(g_reuse.out);
                    gate(v_reuse == v_fresh && g_reuse.run.digest == d_fresh && g_reuse.run.attn_bound_rows == 0 && b_fresh == 0 && v_reuse != vel,
                        c.spec.name + " " + form + " reuse: changed latent + sigma, " + std::to_string(nb2) +
                            " blocks: reused object == fresh object (complete velocity, digests, bound rows 0; differs from step 1)");
                    // (c) the original inputs again after the odd-length run
                    const auto f3 = R.forward(c.L, c.p, in, c.t, c.blocks, opt);
                    gate(cat(f3.out) == vel && f3.run.digest == f.run.digest && f3.run.attn_bound_rows == 0,
                        c.spec.name + " " + form + " reuse: original inputs after the odd-length run == first (complete velocity, digests)");
                }
            }
        }
        if (!a.bench.empty() && c.spec.name == a.bench && comm) {
            DeviceDistRank R(*sh.model, sp, Role{rank, 2, false}, comm);
            const auto b = run_bench(c.spec.name, sp, a.bench_reps, R, line);
            if (w2_timings.empty()) gate(false, c.spec.name + " exposure: NO-REFERENCE (no world-2 timing run of this case)");
            for (const auto& [form, mt] : w2_timings) exposure_gate(c.spec.name + " " + form, mt[0], mt[1], b, line, gate);
        }
    }
    line("%s (%d failures)", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}

// Failure injection through the device schedule (world 2, first case), rank `inject_rank`, block `inject_block`:
//   throw  the rank throws InjectedFailure right after its exchange of the block started (transfer in flight).
//          PASS: the injecting rank caught InjectedFailure, its hook fired and its last started exchange is that block;
//          the peer failed with another error AT that block's exchange (last started = the same block: no early or
//          unrelated failure), within 15 s on USB4 (stream close; 60 s emulated: the test transport has no abort).
//   nanq / vbig / bound  a contract violation on one rank (NaN in Q / |V| = 2^101 / bound-rows counter 1).
//          PASS: BOTH ranks ran every block, exchanged their measurements and threw ContractViolation with the expected
//          reason (naming the block and the injecting rank for nanq / vbig); the hook fired on the injecting rank only.
// Every rank: teardown completed (rank object destroyed after the transport was cancelled) and the device is healthy.
int run_inject(const Args& a, std::uint32_t rank, Shared& sh, strix::dist::Communicator* comm) {
    using Inject = DeviceRunOptions::Inject;
    const Case& c = sh.cases.front();
    const Split sp = Split::of(c.L.seq_len);
    const Role role{rank, 2, false};
    const bool injecting = static_cast<std::int32_t>(rank) == a.inject_rank;
    auto report = [&](bool ok, const std::string& what) {
        std::printf("r%u %s %s\n", rank, ok ? "FAILINJ ok  " : "FAILINJ FAIL", what.c_str());
        std::printf("r%u %s (%d failures)\n", rank, ok ? "PASSED" : "FAILED", ok ? 0 : 1);
        std::fflush(stdout);
        return ok ? 0 : 1;
    };
    if (a.inject_block < 0 || static_cast<std::uint32_t>(a.inject_block) >= c.blocks)
        return report(false, "inject block " + std::to_string(a.inject_block) + " outside [0, " + std::to_string(c.blocks) + ")");
    DeviceRunOptions opt;
    opt.hashes = false;
    opt.timing = true;  // per-block events live through the failure (released by their owner on every path)
    if (injecting) {
        opt.inject = a.inject_kind;
        opt.inject_block = a.inject_block;
    }
    enum { kNone, kInjected, kContract, kArgument, kOther } caught = kNone;
    std::string why;
    std::unique_ptr<DeviceDistRank> R = std::make_unique<DeviceDistRank>(*sh.model, sp, role, comm);
    block::StreamInputs in = c.in;
    if (a.forward) {
        block::HostAttention ratt;
        in.text = block::text_refine_host(sh.g, c.text_states.data(), c.L.text_len, ratt);
    }
    const auto t0 = std::chrono::steady_clock::now();
    try {
        if (a.forward) static_cast<void>(R->forward(c.L, c.p, in, c.t, c.blocks, opt));
        else static_cast<void>(R->run(c.t, sh.g.basis, rows_of(c, sp, role), c.blocks, opt));
    } catch (const InjectedFailure& e) {
        caught = kInjected;
        why = e.what();
    } catch (const ContractViolation& e) {
        caught = kContract;
        why = e.what();
    } catch (const std::invalid_argument& e) {
        caught = kArgument;
        why = e.what();
    } catch (const std::exception& e) {
        caught = kOther;
        why = e.what();
    }
    const bool fired = R->injection_fired();
    const std::int64_t last = R->last_started_block();
    R.reset();  // teardown (buffers, events) after the cancel
    const bool healthy = device_synchronize_ok();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();  // incl. teardown
    const double limit = a.transport == "usb4auto" ? 15.0 : 60.0;
    bool ok = healthy && s < limit && fired == injecting;
    std::string expect;
    if (a.inject_kind == Inject::throw_in_flight) {
        ok = ok && last == a.inject_block && (injecting ? caught == kInjected : caught == kOther);
        expect = injecting ? "InjectedFailure" : "a transport error at the same block";
    } else {
        const std::string at = " on block " + std::to_string(a.inject_block) + " (rank " + std::to_string(a.inject_rank) + ")";
        expect = a.inject_kind == Inject::nan_q ? "non-finite ||q||" + at : a.inject_kind == Inject::v_big ? "> 2^100" + at : "bound rows (must be 0)";
        ok = ok && caught == kContract && why.find(expect) != std::string::npos && last == static_cast<std::int64_t>(c.blocks) - 1;
    }
    static const char* names[] = {"none", "InjectedFailure", "ContractViolation", "invalid_argument", "other"};
    char b[1600];
    std::snprintf(b, sizeof b,
        "injection (%s rank, block %d, case %s S=%u): caught %s after %.2f s (limit %.0f s), hook fired %d, last started block %lld, "
        "device healthy %d; expected %s: %s",
        injecting ? "injecting" : "peer", a.inject_block, c.spec.name.c_str(), sp.S, names[caught], s, limit, fired ? 1 : 0,
        static_cast<long long>(last), healthy ? 1 : 0, expect.c_str(), why.c_str());
    return report(ok, b);
}

// FAST-class accelerators (--accel-check, whole forward). Per case and form (one DistAccel per form and rank):
//   off: a forward with an all-off DistAccel == the dense forward (complete velocity, digests); unfused fc1 (plain
//        linear + SwiGLU on the permuted halves, no Turbo) == the fused dense forward.
//   sequence (Turbo if --turbo, FirstBlockCache at --fbc): c1 original inputs (cold: full + commit), c2 original with
//        forced reuse, c3 changed latent / sigma with forced refresh (full + commit), c4 changed, natural decision
//        (identical r -> score 0 -> reuse), c5 original, natural decision.
//   G3 within the lever: every world-2 form == w1 per call (complete velocity, action, score bits, owned key block
//   digests); w1 identical on both nodes (all calls' complete velocities exchanged).
int run_accel_cases(const Args& a, std::uint32_t rank, Shared& sh, strix::dist::Communicator* comm) {
    int failures = 0;
    auto line = [&](const char* fmt, auto... v) {
        char b[1400];
        std::snprintf(b, sizeof b, fmt, v...);
        std::printf("r%u %s\n", rank, b);
        std::fflush(stdout);
    };
    auto gate = [&](bool ok, const std::string& what) {
        line("%s %s", ok ? "G3 ok  " : "G3 FAIL", what.c_str());
        if (!ok) ++failures;
    };
    auto cat = [](const block::StreamOutputs& o) {
        std::vector<std::uint16_t> v(o.video_neg_v);
        v.insert(v.end(), o.audio_neg_v.begin(), o.audio_neg_v.end());
        return v;
    };
    const double thr = a.fbc_threshold;
    const bool fbc = thr >= 0.0;
    // one TurboV4 per rank (its device scratch serves one stream): built in the rank's own thread / process
    std::unique_ptr<block::TurboV4> turbo;
    if (!a.turbo_lora.empty()) {
        std::uint32_t max_s = 0;
        for (const Case& c : sh.cases) max_s = std::max(max_s, c.L.seq_len);
        turbo = std::make_unique<block::TurboV4>(a.turbo_lora, a.turbo_egrid, a.turbo_strength, max_s, sh.model->num_blocks(), 0U);
        line("turbo %s strength %.2f, %u blocks, %.2f GB device", block::TurboV4::kIdentity, static_cast<double>(turbo->strength()), sh.model->num_blocks(),
            turbo->device_bytes() / 1e9);
    }
    if (!fbc && !turbo && !a.sparse) {
        line("accel-check needs --turbo, --fbc and / or --sparse");
        return 1;
    }
    for (const Case& c : sh.cases) {
        const Split sp = Split::of(c.L.seq_len);
        line("accel case %s S=%u blocks %u turbo %s fbc %s fc1 %s sparse %s", c.spec.name.c_str(), sp.S, c.blocks, turbo ? "on" : "off",
            fbc ? std::to_string(thr).c_str() : "off", sh.model->fc1_plain() ? "plain (unfused + K5)" : "fused",
            a.sparse ? ("tau " + std::to_string(a.sparse_params.tau) + " dense calls " + std::to_string(a.sparse_params.dense_calls) + " (cut " +
                           std::to_string(Split::sparse(c.L.seq_len).cut) + ")").c_str() : "off");
        block::StreamInputs in2 = c.in;
        for (auto& e : in2.video_x) e ^= 0x8000U;
        for (auto& e : in2.audio_x) e ^= 0x8000U;
        const block::ForwardPlan p2 = block::ForwardPlan::build(c.L, c.sigma0 * 0.5F, c.tags);
        const ForwardTables t2 = ForwardTables::build(sh.g, c.L, p2);
        struct Call {
            std::vector<std::uint16_t> v;
            Digests d;
            int action = -1;
            std::uint32_t score_bits = 0, blocks_run = 0;
            std::uint32_t sparse_call = 0, sparse_layers = 0;
            std::vector<std::uint64_t> lutd, lute;
        };
        std::vector<Call> ref;  // w1
        for (const auto& form : a.forms) {
            const Role role = form == "w1" ? Role{rank, 1, false} : Role{rank, 2, form == "w2swap"};
            if (role.world == 2U && !comm) continue;
            block::HostAttention ratt;
            block::StreamInputs in = c.in;
            in.text = block::text_refine_host(sh.g, c.text_states.data(), c.L.text_len, ratt);
            in2.text = in.text;
            DeviceRunOptions opt;
            opt.hashes = false;
            DeviceDistRank R(*sh.model, sp, role, role.world == 2U ? comm : nullptr);
            const std::string tag = c.spec.name + " " + form;
            // the model's forward without accelerators; an attached all-off accelerator must not change a bit
            const auto fd = R.forward(c.L, c.p, in, c.t, c.blocks, opt);
            const std::vector<std::uint16_t> dense = cat(fd.out);
            {
                DistAccel off(nullptr, false);
                const auto fo = R.forward(c.L, c.p, in, c.t, c.blocks, opt, &off);
                gate(cat(fo.out) == dense && fo.run.digest == fd.run.digest, tag + " accelerators attached but OFF == no accelerator (complete velocity, digests)");
            }
            const auto ed = a.expect_dense.find(c.spec.name);
            if (ed != a.expect_dense.end())  // the dense bits of the reference runs (fused fc1, base fc2)
                gate(hash16(fd.out.video_neg_v) == ed->second[0] && hash16(fd.out.audio_neg_v) == ed->second[1],
                    tag + " model without accelerators == the recorded dense velocity (fc1 " + (sh.model->fc1_plain() ? "plain" : "fused") + ")");
            DistAccel acc(turbo.get(), fbc, fbc ? thr : 0.08);
            if (a.sparse) acc.set_sparse(a.sparse_params);
            acc.begin_run();  // a sampling run, on both ranks
            // the sparse identity runs on its own split (cut 128 floor(S / 256)) in both worlds
            std::unique_ptr<DeviceDistRank> Rs;
            if (a.sparse) Rs = std::make_unique<DeviceDistRank>(*sh.model, Split::sparse(c.L.seq_len), role, role.world == 2U ? comm : nullptr);
            DeviceDistRank& RA = a.sparse ? *Rs : R;
            struct Spec {
                bool changed;
                DistAccel::FbcForce force;
                const char* name;
            };
            const Spec seq[] = {{false, DistAccel::FbcForce::none, "c1 original"}, {false, DistAccel::FbcForce::reuse, "c2 original, forced reuse"},
                {true, DistAccel::FbcForce::refresh, "c3 changed, forced refresh"}, {true, DistAccel::FbcForce::none, "c4 changed, natural"},
                {false, DistAccel::FbcForce::none, "c5 original, natural"}};
            std::vector<Call> calls;
            for (const Spec& sp_ : seq) {
                acc.set_fbc_force(fbc ? sp_.force : DistAccel::FbcForce::none);
                const auto f = sp_.changed ? RA.forward(c.L, p2, in2, t2, c.blocks, opt, &acc) : RA.forward(c.L, c.p, in, c.t, c.blocks, opt, &acc);
                Call k;
                k.v = cat(f.out);
                k.d = f.run.digest;
                k.action = acc.last().fbc_action;
                std::memcpy(&k.score_bits, &acc.last().fbc_score, 4);
                k.blocks_run = acc.last().blocks_run;
                k.sparse_call = acc.last().sparse_call;
                k.sparse_layers = acc.last().sparse_layers;
                k.lutd = acc.last().lut_digest;
                k.lute = acc.last().lut_entries;
                if (a.sparse) {
                    std::uint64_t ent = 0;
                    for (auto e : k.lute) ent += e;
                    const std::uint32_t KT = (c.L.seq_len + 63U) / 64U, QT = (role.rows(Split::sparse(c.L.seq_len)) + 127U) / 128U;
                    line("%s %s: sparse model call %u, %u sparse layers, this rank's LUT entries %llu (density of its tiles %.3f over the sparse layers)",
                        tag.c_str(), sp_.name, k.sparse_call, k.sparse_layers, static_cast<unsigned long long>(ent),
                        k.sparse_layers ? static_cast<double>(ent) / (static_cast<double>(k.sparse_layers) * QT * 56.0 * KT) : 0.0);
                }
                std::string rows;
                for (auto r : acc.last().adaln_idx) rows += std::to_string(r) + " ";
                line("%s %s: action %d (%s) score %.6g blocks run %u, velocity %016llx, bound rows %llu, E-grid rows [ %s]", tag.c_str(), sp_.name,
                    k.action, k.action == 0 ? "full+commit" : k.action == 1 ? "reuse" : k.action == 2 ? "full, no commit" : "no fbc",
                    static_cast<double>(acc.last().fbc_score), k.blocks_run, static_cast<unsigned long long>(hash16(k.v)),
                    static_cast<unsigned long long>(f.run.attn_bound_rows), rows.c_str());
                calls.push_back(std::move(k));
            }
            if (turbo) gate(calls[0].v != dense, tag + " Turbo changes the velocity (c1 != the model without accelerators)");
            if (fbc) {
                const bool exp = calls[0].action == 0 && calls[1].action == 1 && calls[2].action == 0 && calls[3].action == 1 &&
                    calls[1].blocks_run == 1U && calls[0].blocks_run == c.blocks && calls[2].blocks_run == c.blocks;
                gate(exp, tag + " FBC actions c1..c4 = full+commit (cold), reuse (forced, 1 block run), full+commit (forced refresh), reuse (natural, score " +
                    std::to_string(calls[3].score_bits) + " bits)");
            }
            if (a.sparse) {
                std::uint32_t sl = 0;
                for (const auto& k : calls) sl += k.sparse_layers;
                gate(sl > 0, tag + " sparse layers ran (" + std::to_string(sl) + " over the 5 calls)");
            }
            if (a.sparse && comm) {
                // LUT digests / entries: w1 identical on both nodes; world 2: rank 0 + rank 1 == w1 per block and call
                std::vector<std::uint64_t> mine, peer;
                for (const auto& k : calls) {
                    mine.insert(mine.end(), k.lutd.begin(), k.lutd.end());
                    mine.insert(mine.end(), k.lute.begin(), k.lute.end());
                }
                peer.assign(mine.size(), 0);
                comm->exchange_host(std::as_bytes(std::span(mine)), std::as_writable_bytes(std::span(peer)),
                    strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xABU, 0});
                if (role.world == 1U) {
                    gate(peer == mine, tag + " sparse LUT digests + entries of every block and call identical on both nodes");
                } else if (ref.size() == calls.size()) {
                    std::uint32_t bad = 0, n = 0;
                    std::size_t at = 0;
                    for (std::size_t j = 0; j < calls.size(); ++j) {
                        for (std::size_t i = 0; i < calls[j].lutd.size(); ++i, ++n) bad += mine[at + i] + peer[at + i] != ref[j].lutd[i];
                        at += calls[j].lutd.size();
                        for (std::size_t i = 0; i < calls[j].lute.size(); ++i, ++n) bad += mine[at + i] + peer[at + i] != ref[j].lute[i];
                        at += calls[j].lute.size();
                    }
                    gate(bad == 0 && n > 0, tag + " sparse LUT digest and entry count of rank 0 + rank 1 == w1 for every block of every call (" +
                        std::to_string(n - bad) + "/" + std::to_string(n) + ")");
                }
            }
            if (role.world == 1U) {
                ref = calls;
                if (comm) {
                    std::vector<std::uint16_t> all, peer;
                    for (const auto& k : calls) all.insert(all.end(), k.v.begin(), k.v.end());
                    peer.assign(all.size(), 0);
                    comm->exchange_host(std::as_bytes(std::span(all)), std::as_writable_bytes(std::span(peer)),
                        strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xA8U, 0});
                    gate(peer == all, tag + " accelerated velocities of the 5 calls identical on both nodes (complete bf16 vectors, " +
                        std::to_string(all.size()) + " values)");
                }
            } else if (ref.size() != calls.size()) {
                gate(false, tag + " NO-REFERENCE (no accelerated w1 run)");
            } else {
                const unsigned kb = role.block();
                for (std::size_t j = 0; j < calls.size(); ++j) {
                    std::uint32_t bad = 0;
                    for (std::size_t i = 0; i < calls[j].d.size(); ++i) bad += calls[j].d[i][kb] != ref[j].d[i][kb];
                    gate(calls[j].v == ref[j].v && calls[j].action == ref[j].action && calls[j].score_bits == ref[j].score_bits &&
                            calls[j].blocks_run == ref[j].blocks_run && calls[j].sparse_call == ref[j].sparse_call &&
                            calls[j].sparse_layers == ref[j].sparse_layers && bad == 0,
                        tag + " " + seq[j].name + ": == w1 (complete velocity, action, score bits, blocks run, key block " + std::to_string(kb) +
                            " digests)");
                }
            }
        }
    }
    line("%s (%d failures)", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}

// FirstBlockCache pair-wide abort (--inject-fail fbc:RANK:CALL, --forward --fbc THR): both ranks run CALL + 1 accelerated
// forwards (cold, then natural decisions) after reset_fbc(); on call CALL the injecting rank's FBC holder fails inside
// after_block0. PASS: the injecting rank threw the FBC error, the peer failed at the decision exchange of the same call
// (transport closed / timeout, not an FBC error of its own), both well before the deadline on USB4, devices healthy, and
// on both ranks the accelerator is dead (a further forward / reset refuses: no local-only recovery).
int run_inject_fbc(const Args& a, std::uint32_t rank, Shared& sh, strix::dist::Communicator* comm) {
    const Case& c = sh.cases.front();
    const Split sp = Split::of(c.L.seq_len);
    const Role role{rank, 2, false};
    const bool injecting = static_cast<std::int32_t>(rank) == a.inject_rank;
    auto report = [&](bool ok, const std::string& what) {
        std::printf("r%u %s %s\n", rank, ok ? "FAILINJ ok  " : "FAILINJ FAIL", what.c_str());
        std::printf("r%u %s (%d failures)\n", rank, ok ? "PASSED" : "FAILED", ok ? 0 : 1);
        std::fflush(stdout);
        return ok ? 0 : 1;
    };
    if (!a.forward || a.fbc_threshold < 0.0) return report(false, "fbc injection needs --forward and --fbc");
    if (a.inject_block > 8) return report(false, "fbc injection call index above 8");
    block::HostAttention ratt;
    block::StreamInputs in = c.in;
    in.text = block::text_refine_host(sh.g, c.text_states.data(), c.L.text_len, ratt);
    DeviceRunOptions opt;
    opt.hashes = false;
    DistAccel acc(nullptr, true, a.fbc_threshold);
    acc.reset_fbc();
    auto R = std::make_unique<DeviceDistRank>(*sh.model, sp, role, comm);
    std::string why;
    bool fbc_error = false, threw = false;
    std::int32_t threw_at = -1;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::int32_t k = 0; k <= a.inject_block && !threw; ++k) {
        if (injecting && k == a.inject_block) acc.inject_fbc_failure_for_test();
        try {
            static_cast<void>(R->forward(c.L, c.p, in, c.t, c.blocks, opt, &acc));
        } catch (const std::exception& e) {
            threw = true;
            threw_at = k;
            why = e.what();
            fbc_error = why.find("h3 fbc:") != std::string::npos;
        }
    }
    const bool dead = acc.failed();
    bool refused = false;
    try {
        static_cast<void>(R->forward(c.L, c.p, in, c.t, c.blocks, opt, &acc));
    } catch (const std::logic_error&) {
        refused = true;
    } catch (const std::exception&) {
    }
    R.reset();
    const bool healthy = device_synchronize_ok();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();  // incl. teardown
    const double limit = a.transport == "usb4auto" ? 15.0 : 90.0;
    const bool ok = threw && threw_at == a.inject_block && (injecting ? fbc_error : !fbc_error) && dead && refused && healthy && s < limit;
    char b[1400];
    std::snprintf(b, sizeof b,
        "FBC injection (%s rank, call %d, case %s S=%u): threw on call %d after %.2f s (limit %.0f s), %s error, accelerator dead %d, "
        "later forward refused %d, device healthy %d: %s",
        injecting ? "injecting" : "peer", a.inject_block, c.spec.name.c_str(), sp.S, threw_at, s, limit, fbc_error ? "FBC" : "non-FBC", dead ? 1 : 0,
        refused ? 1 : 0, healthy ? 1 : 0, why.c_str());
    return report(ok, b);
}

// Link-exposure performance qualification (--exposure-runs K --timing NAME --forward, world 2 only): K runs, each a whole
// forward of both world-2 forms in ROTATED order (run r even: w2, w2swap; odd: w2swap, w2), every forward followed on the
// same rank object by the matching attention / exchange bench (same ownership, the block loop's scans included). Per
// rank, form and run: ratio = (mean GPU wait for RX after the local partial + max(0, paired local slowdown under the
// exchange)) / mean block (blocks 1..). Gate: max over ranks and forms of the per-(rank, form) MEDIAN run ratio < 3 %;
// spread (min .. max) and the wait tails are reported. Also asserted: every forward's velocity is the same bits.
int run_exposure(const Args& a, std::uint32_t rank, Shared& sh, strix::dist::Communicator* comm) {
    int failures = 0;
    auto line = [&](const char* fmt, auto... v) {
        char b[1400];
        std::snprintf(b, sizeof b, fmt, v...);
        std::printf("r%u %s\n", rank, b);
        std::fflush(stdout);
    };
    auto gate = [&](bool ok, const std::string& what) {
        line("%s %s", ok ? "PERF ok  " : "PERF FAIL", what.c_str());
        if (!ok) ++failures;
    };
    const Case* cp = nullptr;
    for (const Case& c : sh.cases)
        if (c.spec.name == a.timing) cp = &c;
    if (!cp || !comm || !a.forward) {
        line("exposure runs need --forward, two ranks and --timing NAME of a case");
        return 1;
    }
    const Case& c = *cp;
    const Split sp = a.sparse ? Split::sparse(c.L.seq_len) : Split::of(c.L.seq_len);
    if (a.sparse)
        line("exposure with sparse attention: tau %.3f, dense calls %u, cut %u", static_cast<double>(a.sparse_params.tau), a.sparse_params.dense_calls, sp.cut);
    block::HostAttention ratt;
    block::StreamInputs in = c.in;
    in.text = block::text_refine_host(sh.g, c.text_states.data(), c.L.text_len, ratt);
    std::map<std::string, std::vector<double>> ratios;
    std::uint64_t v0 = 0;
    bool same_bits = true;
    const auto w0 = std::chrono::steady_clock::now();
    for (unsigned r = 0; r < a.exposure_runs; ++r) {
        const std::array<std::string, 2> order = r % 2U ? std::array<std::string, 2>{"w2swap", "w2"} : std::array<std::string, 2>{"w2", "w2swap"};
        for (const auto& form : order) {
            const Role role{rank, 2, form == "w2swap"};
            DeviceDistRank R(*sh.model, sp, role, comm);
            DeviceRunOptions opt;
            opt.hashes = false;
            opt.timing = true;
            std::unique_ptr<DistAccel> acc;
            if (a.sparse) {
                acc = std::make_unique<DistAccel>(nullptr, false);
                acc->set_sparse(a.sparse_params);
                acc->begin_run();
            }
            const auto f = R.forward(c.L, c.p, in, c.t, c.blocks, opt, acc.get());
            if (acc) line("exposure run %u %s: %u sparse layers", r, form.c_str(), acc->last().sparse_layers);
            const std::uint64_t vh = hash16(f.out.video_neg_v) ^ (hash16(f.out.audio_neg_v) * 0x9e3779b97f4a7c15ULL);
            if (r == 0 && form == order[0]) v0 = vh;
            same_bits = same_bits && vh == v0;
            const auto mt = mean_timing(f.run.timing);
            std::vector<double> ex;
            for (std::size_t i = 1; i < f.run.timing.size(); ++i) ex.push_back(f.run.timing[i].exposed_ms);
            std::sort(ex.begin(), ex.end());
            const auto b = R.bench_attention(a.bench_reps);
            const double ratio = (mt[1] + std::max(0.0, b.local_slowdown_ms)) / mt[0];
            ratios[form].push_back(ratio);
            line("exposure run %u %s (key block %u): forward %.2f s, block %.2f ms, exposed mean %.2f ms (max %.1f, p90 %.1f), paired slowdown %.2f ms "
                 "(bench: local %.1f, with exchange %.1f, exchange alone %.1f ms) -> %.2f %%",
                r, form.c_str(), role.block(), f.wall_s, mt[0], mt[1], ex.empty() ? 0.0 : ex.back(), ex.empty() ? 0.0 : ex[(ex.size() * 9) / 10],
                b.local_slowdown_ms, b.local_ms, b.local_with_exchange_ms, b.exchange_ms, 100.0 * ratio);
        }
    }
    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::array<double, 2> mine{med(ratios["w2"]), med(ratios["w2swap"])}, peer{0, 0};
    comm->exchange_host(std::as_bytes(std::span(mine)), std::as_writable_bytes(std::span(peer)),
        strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xA9U, 0});
    for (const char* form : {"w2", "w2swap"}) {
        const auto& v = ratios[form];
        line("exposure %s: median %.2f %% over %zu runs, spread %.2f .. %.2f %%", form, 100.0 * med(v), v.size(),
            100.0 * *std::min_element(v.begin(), v.end()), 100.0 * *std::max_element(v.begin(), v.end()));
    }
    const double worst = std::max({mine[0], mine[1], peer[0], peer[1]});
    line("exposure medians: r%u w2 %.2f %% w2swap %.2f %% | r%u w2 %.2f %% w2swap %.2f %% | pair wall %.1f s", rank, 100.0 * mine[0], 100.0 * mine[1],
        1U - rank, 100.0 * peer[0], 100.0 * peer[1], std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count());
    gate(same_bits, "every forward of every run and form gave the same velocity bits");
    gate(worst < 0.03, "max over ranks / forms of the median run exposure = " + std::to_string(100.0 * worst) + " % < 3 %");
    line("%s (%d failures)", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}

// Per-kernel time split of ONE world-2 whole forward of the --timing case (--profile --forward; Turbo with --turbo,
// sparse with --sparse): GPU time per kernel group summed over the blocks (consecutive stream events), int8 GEMM TOPS
// from the GEMM time (group time minus the standalone quantizer time x blocks), attention TF (dense: 4 x rows x S x 128
// x 56 per block over local + remote partial time), and the host-side stages of the forward.
int run_profile(const Args& a, std::uint32_t rank, Shared& sh, strix::dist::Communicator* comm) {
    auto line = [&](const char* fmt, auto... v) {
        char b[1400];
        std::snprintf(b, sizeof b, fmt, v...);
        std::printf("r%u %s\n", rank, b);
        std::fflush(stdout);
    };
    const Case* cp = nullptr;
    for (const Case& c : sh.cases)
        if (c.spec.name == a.timing) cp = &c;
    if (!cp || !comm || !a.forward) {
        line("profile needs --forward, two ranks and --timing NAME of a case");
        return 1;
    }
    const Case& c = *cp;
    const Split sp = a.sparse ? Split::sparse(c.L.seq_len) : Split::of(c.L.seq_len);
    const Role role{rank, 2, false};
    std::unique_ptr<block::TurboV4> turbo;
    if (!a.turbo_lora.empty()) turbo = std::make_unique<block::TurboV4>(a.turbo_lora, a.turbo_egrid, a.turbo_strength, c.L.seq_len, sh.model->num_blocks(), 0U);
    std::unique_ptr<DistAccel> acc;
    if (turbo || a.sparse) {
        acc = std::make_unique<DistAccel>(turbo.get(), false);
        if (a.sparse) acc->set_sparse(a.sparse_params);
        acc->begin_run();
    }
    block::HostAttention ratt;
    block::StreamInputs in = c.in;
    in.text = block::text_refine_host(sh.g, c.text_states.data(), c.L.text_len, ratt);
    DeviceDistRank R(*sh.model, sp, role, comm);
    DeviceRunOptions opt;
    opt.hashes = false;
    opt.profile = true;
    const auto f = R.forward(c.L, c.p, in, c.t, c.blocks, opt, acc.get());
    const auto& km = f.run.kernel_ms;
    double total = 0;
    for (double v : km) total += v;
    const std::uint32_t n = role.rows(sp), nb = c.blocks;
    line("profile case %s S=%u, rank %u rows %u (key block %u), %u blocks, turbo %s, sparse %s: forward %.2f s = embed %.3f + blocks %.3f + final %.3f + "
         "gather %.3f s; GPU time in the block groups %.1f ms (%.2f ms / block)",
        c.spec.name.c_str(), sp.S, rank, n, role.block(), nb, turbo ? "on" : "off", a.sparse ? "on" : "off", f.wall_s, f.embed_s, f.blocks_s, f.final_s,
        f.gather_s, total, total / nb);
    for (std::size_t k = 0; k < km.size(); ++k)
        if (km[k] > 0)
            line("profile  %-58s %9.1f ms  %7.2f ms/block  %5.1f %%", f.run.kernel_names[k].c_str(), km[k], km[k] / nb, 100.0 * km[k] / total);
    // GEMM TOPS (group index: qkv 2, out 11, fc1 13, fc2 15; quantizer K 5376 / 7168 / 14336)
    struct G {
        int idx, q;
        const char* name;
        double N, K;
    };
    const G gs[4] = {{2, 0, "qkv", 21504, 5376}, {11, 1, "out", 5376, 7168}, {13, 0, "fc1", 28672, 5376}, {15, 2, "fc2", 5376, 14336}};
    double ops_all = 0, gemm_all = 0, quant_all = 0;
    for (const G& g : gs) {
        const double qms = f.run.quantize_ms[g.q] * nb, gms = km[static_cast<std::size_t>(g.idx)] - qms;
        const double ops = 2.0 * n * g.N * g.K * nb;
        ops_all += ops;
        gemm_all += gms;
        quant_all += qms;
        line("profile  int8 %-4s: quantize %.2f ms/block, GEMM %.2f ms/block -> %.1f TOPS (GEMM only), %.1f TOPS (with quantize)", g.name,
            qms / nb, gms / nb, ops / (gms * 1e9), ops / (km[static_cast<std::size_t>(g.idx)] * 1e9));
    }
    line("profile  int8 all four: %.1f TOPS GEMM only, %.1f TOPS with quantize (quantize %.1f ms / block)", ops_all / (gemm_all * 1e9),
        ops_all / ((gemm_all + quant_all) * 1e9), quant_all / nb);
    const double attn_ms = km[8] + km[10];
    const double fl = 4.0 * n * static_cast<double>(sp.S) * 128.0 * 56.0 * nb;
    line("profile  attention: local + remote + merge %.2f ms/block -> %.1f TF dense-equivalent (4 x rows x S x 128 x 56 per block), exposed wait %.2f ms/block",
        attn_ms / nb, fl / (attn_ms * 1e9), km[9] / nb);
    line("PASSED (0 failures)");
    return 0;
}

int run_rank(const Args& a, std::uint32_t rank, Shared& sh, strix::dist::Communicator* comm) {
    if (a.profile) return run_profile(a, rank, sh, comm);
    if (a.exposure_runs) return run_exposure(a, rank, sh, comm);
    if (a.inject_fbc) return run_inject_fbc(a, rank, sh, comm);
    if (a.inject_block >= 0) return run_inject(a, rank, sh, comm);
    if (a.accel_check) return run_accel_cases(a, rank, sh, comm);
    if (a.forward) return run_forward_cases(a, rank, sh, comm);
    int failures = 0;
    auto line = [&](const char* fmt, auto... v) {
        char b[1024];
        std::snprintf(b, sizeof b, fmt, v...);
        std::printf("r%u %s\n", rank, b);
        std::fflush(stdout);
    };
    auto gate = [&](bool ok, const std::string& what) {
        line("%s %s", ok ? "G3 ok  " : "G3 FAIL", what.c_str());
        if (!ok) ++failures;
    };
    for (const Case& c : sh.cases) {
        const Split sp = Split::of(c.L.seq_len);
        const bool timing_case = c.spec.name == a.timing;
        line("case %s S=%u cut=%u rows %u + %u blocks %u message %.1f MB", c.spec.name.c_str(), sp.S, sp.cut, sp.rows(0), sp.rows(1), c.blocks,
            sp.message_bytes() / 1e6);
        std::vector<std::array<std::uint64_t, 2>> h1;
        std::array<std::uint64_t, 2> f1{0, 0};  // world-1 final residual hash of each key block
        bool have_f1 = false;
        Digests d1;
        std::vector<std::pair<std::string, std::array<double, 2>>> w2_timings_bs;
        std::uint64_t flush1 = 0;                // world-1 gemm flush count
        for (const auto& form : a.forms) {
            const Role role = form == "w1" ? Role{rank, 1, false} : Role{rank, 2, form == "w2swap"};
            if (role.world == 2U && !comm) continue;
            DeviceRunOptions opt;
            opt.hashes = !timing_case;
            opt.timing = timing_case;
            DeviceDistRank R(*sh.model, sp, role, role.world == 2U ? comm : nullptr);
            const DeviceRunResult res = R.run(c.t, sh.g.basis, rows_of(c, sp, role), c.blocks, opt);
            line("%s %s: %u rows (block %s) %.3f s, flushed groups %llu%s%s", c.spec.name.c_str(), form.c_str(), role.rows(sp),
                role.world == 1U ? "0+1" : (role.block() ? "1" : "0"), res.wall_s, static_cast<unsigned long long>(res.flushed_groups),
                res.comm_stats.empty() ? "" : " | ", res.comm_stats.c_str());
            // gemm flush counter = telemetry (ruled): info only; the two ranks' counts must add up to the 1-node count
            line("info %s %s: %llu flushed rotation groups (gemm telemetry)", c.spec.name.c_str(), form.c_str(),
                static_cast<unsigned long long>(res.flushed_groups));
            gate(res.attn_bound_rows == 0, c.spec.name + " " + form + " attention contract telemetry (bound rows) == 0");
            digest_gates(c.spec.name, form, role, res.digest, d1, c.blocks, role.world == 1U ? comm : nullptr, gate);
            if (role.world == 1U) {
                flush1 = res.flushed_groups;
            } else {
                std::uint64_t mine = res.flushed_groups, peer = 0;
                comm->exchange_host(std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(&peer, 1)),
                    strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xA2U, 0});
                gate(mine + peer == flush1, c.spec.name + " " + form + " flushed groups rank0 + rank1 (" + std::to_string(mine) + " + " +
                    std::to_string(peer) + ") == 1-node (" + std::to_string(flush1) + ")");
            }
            {  // final residual of the owned key block(s), every case (also the timing case, which skips per-block hashes)
                std::array<std::uint64_t, 2> f{0, 0};
                for (unsigned kb = 0; kb < 2; ++kb) {
                    if (role.world == 2U && kb != role.block()) continue;
                    const std::size_t off = role.world == 2U ? 0U : std::size_t{sp.first(kb)} * block::kHidden;
                    f[kb] = fnv1a64(res.x.data() + off, std::size_t{sp.rows(kb)} * block::kHidden * 2U);
                }
                line("final %s %s kb0 %016llx kb1 %016llx", c.spec.name.c_str(), form.c_str(), static_cast<unsigned long long>(f[0]),
                    static_cast<unsigned long long>(f[1]));
                if (role.world == 1U) {
                    f1 = f;
                    have_f1 = true;
                } else if (!have_f1) {
                    gate(false, c.spec.name + " " + form + " NO-REFERENCE (no w1 final residual)");
                } else {
                    gate(f[role.block()] == f1[role.block()], c.spec.name + " " + form + " final residual of key block " +
                        std::to_string(role.block()) + " == w1");
                }
            }
            if (opt.hashes) {
                for (std::uint32_t i = 0; i < c.blocks; ++i)
                    line("hash %s %s block %u kb0 %016llx kb1 %016llx", c.spec.name.c_str(), form.c_str(), i,
                        static_cast<unsigned long long>(res.hash[i][0]), static_cast<unsigned long long>(res.hash[i][1]));
                if (role.world == 1U) {
                    h1 = res.hash;
                    if (comm) {  // the other rank's world-1 hashes (same code on the other node)
                        std::vector<std::array<std::uint64_t, 2>> peer(h1.size());
                        comm->exchange_host(std::as_bytes(std::span(h1)), std::as_writable_bytes(std::span(peer)),
                            strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xA1U, 0});
                        gate(peer == h1, c.spec.name + " w1 per-block hashes identical on both ranks (both key blocks, " +
                            std::to_string(c.blocks) + " blocks)");
                    }
                } else if (h1.empty()) {
                    gate(false, c.spec.name + " " + form + " NO-REFERENCE (no w1 per-block hashes)");
                } else {
                    const unsigned kb = role.block();
                    std::uint32_t bad = 0;
                    for (std::uint32_t i = 0; i < c.blocks; ++i) bad += res.hash[i][kb] != h1[i][kb];
                    gate(bad == 0, c.spec.name + " " + form + " rank " + std::to_string(rank) + " key block " + std::to_string(kb) +
                        " == w1, every block (" + std::to_string(c.blocks - bad) + "/" + std::to_string(c.blocks) + ")");
                }
            }
            if (opt.timing) {
                double sb = 0, sa = 0, se = 0, sl = 0;
                line("timing %s %s: block_ms attn_ms local_ms exposed_ms finish_host_ms", c.spec.name.c_str(), form.c_str());
                for (std::uint32_t i = 0; i < c.blocks; ++i) {
                    const auto& tm = res.timing[i];
                    line("timing %s %s block %u %.2f %.2f %.2f %.2f %.2f", c.spec.name.c_str(), form.c_str(), i, tm.block_ms, tm.attn_ms, tm.local_ms,
                        tm.exposed_ms, tm.finish_host_ms);
                    if (i > 0) { sb += tm.block_ms; sa += tm.attn_ms; se += tm.exposed_ms; sl += tm.local_ms; }  // block 0: warm-up
                }
                if (role.world == 2U) w2_timings_bs.emplace_back(form, mean_timing(res.timing));
                if (c.blocks > 1) {
                    const double k = 1.0 / (c.blocks - 1);
                    line("timing %s %s mean(blocks 1..): block %.2f ms attn %.2f ms local %.2f ms exposed %.2f ms = %.2f %% of the block",
                        c.spec.name.c_str(), form.c_str(), sb * k, sa * k, sl * k, se * k, 100.0 * se / std::max(sb, 1e-9));
                }
            }
        }
        if (!a.bench.empty() && c.spec.name == a.bench && comm) {
            DeviceDistRank R(*sh.model, sp, Role{rank, 2, false}, comm);
            const auto b = run_bench(c.spec.name, sp, a.bench_reps, R, line);
            for (const auto& [form, mt] : w2_timings_bs) exposure_gate(c.spec.name + " " + form, mt[0], mt[1], b, line, gate);
        }
    }
    line("%s (%d failures)", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) try {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        if (s == "--ckpt") a.ckpt = next();
        else if (s == "--transport") a.transport = next();
        else if (s == "--rank") a.rank = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--forms") {
            a.forms = split(next(), ',');
            if (!valid_forms(a.forms)) throw std::invalid_argument("--forms: w1 first, then w2 and / or w2swap (each once)");
        }
        else if (s == "--timing") a.timing = next();
        else if (s == "--seed") a.seed = std::stoull(next());
        else if (s == "--forward") a.forward = true;
        else if (s == "--accel-check") a.accel_check = true;
        else if (s == "--fbc") a.fbc_threshold = std::stod(next());
        else if (s == "--fc1-plain") a.fc1_plain = true;
        else if (s == "--exposure-runs") a.exposure_runs = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--profile") a.profile = true;
        else if (s == "--sparse") {
            a.sparse = true;
            a.sparse_params.tau = std::stof(next());
        } else if (s == "--sparse-dense-calls") a.sparse_params.dense_calls = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--sparse-dense-layers") a.sparse_params.dense_layer_mask = strix::h3::levers::sparse::layer_mask(next());
        else if (s == "--expect-dense") {
            const std::string v = next();
            const auto eq = v.find('='), col = v.find(':');
            if (eq == std::string::npos || col == std::string::npos || col < eq) throw std::invalid_argument("--expect-dense NAME=VIDEOHASH:AUDIOHASH");
            a.expect_dense[v.substr(0, eq)] = {std::stoull(v.substr(eq + 1, col - eq - 1), nullptr, 16), std::stoull(v.substr(col + 1), nullptr, 16)};
        }
        else if (s == "--turbo") {
            const auto v = split(next(), ',');
            if (v.size() < 3 || v.size() > 4) throw std::invalid_argument("--turbo LORA,MERGED_FC2,EGRID[,STRENGTH]");
            a.turbo_lora = v[0];
            a.turbo_fc2 = v[1];
            a.turbo_egrid = v[2];
            if (v.size() == 4) a.turbo_strength = std::stof(v[3]);
        }
        else if (s == "--inject-fail") {
            const auto v = split(next(), ':');
            if (v.size() != 3) throw std::invalid_argument("--inject-fail KIND:RANK:BLOCK");
            using Inject = DeviceRunOptions::Inject;
            a.inject_fbc = v[0] == "fbc";
            a.inject_kind = v[0] == "throw" ? Inject::throw_in_flight : v[0] == "nanq" ? Inject::nan_q : v[0] == "vbig" ? Inject::v_big
                : v[0] == "bound" ? Inject::bound_rows : Inject::none;
            if (a.inject_kind == Inject::none && !a.inject_fbc) throw std::invalid_argument("--inject-fail KIND: throw | nanq | vbig | bound | fbc");
            a.inject_rank = static_cast<std::int32_t>(std::stoi(v[1]));
            a.inject_block = static_cast<std::int32_t>(std::stoi(v[2]));
            if (a.inject_rank < 0 || a.inject_rank > 1 || a.inject_block < 0) throw std::invalid_argument("--inject-fail KIND:RANK (0 | 1):BLOCK");
        }
        else if (s == "--bench-attn") {
            const auto v = split(next(), ':');
            a.bench = v.at(0);
            if (v.size() > 1) a.bench_reps = static_cast<unsigned>(std::stoul(v[1]));
        } else if (s == "--case") {
            const std::string v = next();
            const auto eq = v.find('='), col = v.find(':', eq);
            if (eq == std::string::npos || col == std::string::npos) throw std::invalid_argument("--case NAME=KIND:ARG[:BLOCKS]");
            CaseSpec cs;
            cs.name = v.substr(0, eq);
            cs.kind = v.substr(eq + 1, col - eq - 1);
            std::string rest = v.substr(col + 1);
            const auto last = rest.rfind(':');
            cs.blocks = cs.kind == "synth" ? 3U : 50U;
            if (last != std::string::npos) {
                cs.blocks = static_cast<std::uint32_t>(std::stoul(rest.substr(last + 1)));
                rest = rest.substr(0, last);
            }
            cs.arg = rest;
            cs.blocks_hashes = cs.kind != "synth";
            a.cases.push_back(cs);
        } else throw std::invalid_argument("unknown option " + s);
    }
    if (a.ckpt.empty() || a.cases.empty()) throw std::invalid_argument("--ckpt and at least one --case are required");
    if (a.accel_check && !a.forward) throw std::invalid_argument("--accel-check needs --forward");
    const auto t0 = std::chrono::steady_clock::now();
    auto since = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    Shared sh;
    block::SafeTensors st(a.ckpt);
    sh.g = block::load_global(st);
    std::uint32_t nres = 0;
    std::size_t max_msg = 0;
    for (const auto& cs : a.cases) {
        sh.cases.push_back(make_case(cs, sh.g, a.seed, a.forward));
        sh.cases.back().blocks = std::min(sh.cases.back().blocks, block::count_blocks(st));
        nres = std::max(nres, sh.cases.back().blocks);
        max_msg = std::max(max_msg, Split::of(sh.cases.back().L.seq_len).message_bytes());
    }
    if (!a.turbo_lora.empty()) {  // Turbo: merged fc2 + fc1 in checkpoint order (the block package's objects)
        block::SafeTensors merged(a.turbo_fc2);
        block::MergedFc2Source src(st, merged, sh.g.basis, a.turbo_strength);
        sh.model = std::make_unique<DeviceModel>(sh.g, src, nres, true);
    } else {
        block::FileBlockSource src(st, sh.g.basis, false);
        sh.model = std::make_unique<DeviceModel>(sh.g, src, nres, a.fc1_plain);
    }
    std::printf("loaded %u blocks (%.1f GB resident) in %.1f s; largest K/V message %.1f MB; weight-only operand bound %.1f (info)\n", nres,
        sh.model->resident_bytes() / 1e9, since(), max_msg / 1e6, sh.model->weight_bound());

    std::fflush(stdout);
    const strix::dist::Group group0{0, 2, 0x4833U, 0x4833443144ULL}, group1{1, 2, 0x4833U, 0x4833443144ULL};
    if (a.transport == "emulate") {
        auto [t0p, t1p] = strix::dist::memory_pair();
        int rc[2] = {0, 0};
        std::exception_ptr err[2];
        std::unique_ptr<strix::dist::Transport> tp[2] = {std::move(t0p), std::move(t1p)};
        std::thread th[2];
        for (std::uint32_t r = 0; r < 2; ++r)
            th[r] = std::thread([&, r] {
                try {
                    strix::dist::Communicator comm(r ? group1 : group0, std::move(tp[r]), max_msg);
                    rc[r] = run_rank(a, r, sh, &comm);
                } catch (...) {
                    err[r] = std::current_exception();
                }
            });
        for (auto& t : th) t.join();
        for (auto& e : err)
            if (e) std::rethrow_exception(e);
        std::printf("done %.1f s\n", since());
        return rc[0] | rc[1];
    }
    if (a.transport != "usb4auto") throw std::invalid_argument("--transport emulate | usb4auto");
    strix::dist::Usb4AutoOptions o;
    o.rank = a.rank;
    o.timeout = std::chrono::milliseconds(240000);  // the ranks finish loading weights at different times
    strix::dist::Usb4Topology topo;
    auto t = strix::dist::usb4_host_auto(o, &topo);
    std::printf("r%u transport %s (%.1f s)\n%s\n", a.rank, t->name().c_str(), since(), topo.receipt().c_str());
    strix::dist::Communicator comm(a.rank ? group1 : group0, std::move(t), max_msg);
    const int rc = run_rank(a, a.rank, sh, &comm);
    std::printf("r%u done %.1f s\n", a.rank, since());
    return rc;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
