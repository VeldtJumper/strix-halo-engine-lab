// Multi-session (cohort) decode gate and bench of the CUDA Qwen engine (one model load).
//   qwen_cohort_e2e <tables dir> <rope table> <max positions> <tokens file[,file2,...]> <prompt N> <generate G> [gate|bench]
//
// gate (default): session i decodes a different prompt at a different position: the tokens file i % files, its first
// N - 5 * (i / files) tokens. Reference = each prompt ALONE through the single-session API (session 0): serial greedy
// steps (T = 1, MTP off), the FP32 logits row SHA-256 of every generated token (node STRIX_SERIAL_LOGITS_DIGEST) and the
// step chain. Then for S in STRIX_COHORT_S (default 2,3,4,8) and graphs on/off (STRIX_COHORT_GRAPHS=both|on|off), with S
// fresh sessions (odd sessions prefilled in 256-row chunks through prefill_chunk):
//   lockstep  one row per session per window (S rows), every session in one cohort;
//   mixed     teacher-forced windows of 1..4 rows per session packed into <= 8 rows (cohort_plan), partial commits
//             (keep cycles 1..rows) so every session rolls back inside shared windows; every computed row is compared;
//   mtp       (sidecar set) the serve's step: per-session draft chains, packed windows, greedy acceptance;
// and every row must equal the session alone: argmax and logits SHA-256. The reference runs eager (set_graphs(false)).
// Graphs-on lockstep/mixed arms must show replays (graph counter deltas per arm). Extra arms: a short (64) + long
// (STRIX_COHORT_LONG, e.g. 4K) context in one window; LRU eviction (cache cap 2); session churn (destroy/create) with
// session 0 in the cohort and 100/1024-row prefill chunks; decode-first admission (a session prefilled in 100-row chunks
// between the other sessions' windows, then joining). Prints COHORT GATE PASS/FAIL.
//
// bench: for C in STRIX_COHORT_C (default 1,2,3,4,8): C sessions (prompts as above), G tokens each through the serve's
// step (MTP when a sidecar is set; STRIX_COHORT_BENCH_NOMTP=1 adds an MTP-off arm), aggregate and per-stream tok/s,
// windows, rows/window, draft and verify ms. No logits leave the device.
//
// Env: STRIX_QWEN_GGUF, STRIX_QWEN_MTP, STRIX_QWEN_DRAFT_VOCAB, STRIX_QWEN_MIN_P (0.7), STRIX_COHORT_DEPTH (7),
// STRIX_COHORT_POLICY (fair|max).
#include "strix/cuda_qwen_cohort.hpp"
#include "strix/cuda_qwen_prefill.hpp"
#include "strix/model_content_hash.hpp"

#include <cuda_profiler_api.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace qp = strix::cuda::qwen_prefill;
namespace qc = strix::cuda::qwen_cohort;
using Clock = std::chrono::steady_clock;

static std::vector<std::uint32_t> tokens_of(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "missing %s\n", path.c_str()); std::exit(2); }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (auto& ch : s) if (ch == ',' || ch == '[' || ch == ']') ch = ' ';
    std::istringstream in(s);
    std::vector<std::uint32_t> t;
    std::uint32_t v;
    while (in >> v) t.push_back(v);
    return t;
}
static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (const char ch : s) {
        if (ch == sep) { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur.push_back(ch);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}
static std::vector<std::uint32_t> list_env(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    std::vector<std::uint32_t> out;
    for (const auto& x : split(v ? v : fallback, ',')) out.push_back(static_cast<std::uint32_t>(std::stoul(x)));
    return out;
}

using Digest = strix::ModelContentIdentity;
static Digest row_sha(const float* row, std::size_t count) {
    return strix::sha256_bytes(std::span<const std::byte>(reinterpret_cast<const std::byte*>(row), count * 4));
}
static std::string hex(const Digest& d) {
    static const char* x = "0123456789abcdef";
    std::string s;
    for (auto b : d.bytes) { s.push_back(x[b >> 4]); s.push_back(x[b & 15]); }
    return s;
}
static Digest chain_step(const Digest& prev, std::uint64_t position, const Digest& row) {
    std::array<std::byte, 72> in{};
    for (std::size_t j = 0; j < 32; ++j) { in[j] = std::byte{prev.bytes[j]}; in[40 + j] = std::byte{row.bytes[j]}; }
    for (unsigned j = 0; j < 8; ++j) in[32 + j] = static_cast<std::byte>((position >> (8U * j)) & 255U);
    return strix::sha256_bytes(in);
}

constexpr std::size_t kVocab = 248320;

struct Prompt {
    std::vector<std::uint32_t> tokens;  // prompt (n tokens); the decode starts at position n - 1
    std::vector<std::uint32_t> seq;     // prompt + reference greedy tokens
    std::vector<Digest> digest;         // per generated row
    std::uint64_t n = 0;
    std::string label;
};
constexpr std::uint32_t kNone = 0xffffffffU;

// The session's state after the prompt rows [0, n - 1): run() or 256-row prefill_chunk calls.
static void prefill(qp::Prefill& p, qp::SessionId s, const Prompt& pr, bool chunked) {
    if (!chunked) {
        p.run(s, pr.tokens, pr.n - 1, [](const qp::Piece&) {});
        return;
    }
    std::uint64_t at = 0;
    while (at < pr.n - 1) at = p.prefill_chunk(s, pr.tokens, pr.n - 1, at, 256);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: qwen_cohort_e2e tables rope max_positions tokens[,tokens2..] N G [gate|bench]\n");
        return 1;
    }
    const std::string mode = argc > 7 ? argv[7] : "gate";
    qp::Config c;
    c.model_gguf = std::getenv("STRIX_QWEN_GGUF") ? std::getenv("STRIX_QWEN_GGUF") : "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf";
    if (const char* s = std::getenv("STRIX_QWEN_MTP")) c.mtp_sidecar_gguf = s;
    if (const char* v = std::getenv("STRIX_QWEN_DRAFT_VOCAB")) c.draft_vocab = v;
    c.tables_dir = argv[1];
    c.rope_table = argv[2];
    c.max_positions = static_cast<std::uint32_t>(std::atoi(argv[3]));
    c.release_by_layer = false;
    c.decode = true;
    c.export_state = false;  // decode sessions (prefill_chunk needs it); run() then keeps the state on the device
    const auto files = split(argv[4], ',');
    const auto n_base = static_cast<std::uint64_t>(std::atoll(argv[5]));
    const auto g = static_cast<std::uint32_t>(std::atoi(argv[6]));
    std::vector<std::vector<std::uint32_t>> sources;
    for (const auto& f : files) sources.push_back(tokens_of(f));
    const bool mtp = !c.mtp_sidecar_gguf.empty();
    const float min_p = std::getenv("STRIX_QWEN_MIN_P") ? static_cast<float>(std::atof(std::getenv("STRIX_QWEN_MIN_P"))) : 0.7F;
    const auto depth = std::getenv("STRIX_COHORT_DEPTH") ? static_cast<std::uint32_t>(std::atoi(std::getenv("STRIX_COHORT_DEPTH"))) : 7U;
    const auto policy = qc::depth_policy_of(std::getenv("STRIX_COHORT_POLICY") ? std::getenv("STRIX_COHORT_POLICY") : "fair");
    std::uint32_t max_sessions = 0;
    const auto s_list = list_env("STRIX_COHORT_S", "2,3,4,8");
    const auto c_list = list_env("STRIX_COHORT_C", "1,2,3,4,8");
    for (const auto s : mode == "bench" ? c_list : s_list) max_sessions = std::max(max_sessions, s);
    if (max_sessions == 0 || g == 0) { std::fprintf(stderr, "bad S/G\n"); return 1; }
    std::vector<Prompt> prompts(max_sessions);
    for (std::uint32_t i = 0; i < max_sessions; ++i) {
        const auto& src = sources[i % sources.size()];
        const std::uint64_t shift = 5ULL * (i / sources.size());
        auto& pr = prompts[i];
        pr.n = std::min<std::uint64_t>(n_base, src.size());
        if (pr.n < shift + 2) { std::fprintf(stderr, "prompt %u too short\n", i); return 1; }
        pr.n -= shift;
        if (pr.n + g + 8 > c.max_positions) { std::fprintf(stderr, "max_positions too small\n"); return 1; }
        pr.tokens.assign(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(pr.n));
        pr.label = files[i % sources.size()] + " n " + std::to_string(pr.n);
    }
    // gate extras: a short context (the first 64 tokens of the first file) and STRIX_COHORT_LONG (e.g. a 4K prompt, all of it)
    std::uint32_t short_prompt = kNone, long_prompt = kNone, long3k_prompt = kNone;
    if (mode != "bench") {
        Prompt sp;
        sp.n = std::min<std::uint64_t>(64, sources[0].size());
        sp.tokens.assign(sources[0].begin(), sources[0].begin() + static_cast<std::ptrdiff_t>(sp.n));
        sp.label = "short " + std::to_string(sp.n);
        short_prompt = static_cast<std::uint32_t>(prompts.size());
        prompts.push_back(sp);
        if (const char* lf = std::getenv("STRIX_COHORT_LONG")) {
            Prompt lp;
            lp.tokens = tokens_of(lf);
            lp.n = lp.tokens.size();
            if (lp.n + g + 8 > c.max_positions) { std::fprintf(stderr, "STRIX_COHORT_LONG longer than max_positions allows\n"); return 1; }
            lp.label = std::string("long ") + lf + " n " + std::to_string(lp.n);
            long_prompt = static_cast<std::uint32_t>(prompts.size());
            prompts.push_back(lp);
            if (lp.n > 3000) {  // a second long context (its first 3000 tokens): two sessions past 2048 positions in one window
                Prompt l3;
                l3.n = 3000;
                l3.tokens.assign(lp.tokens.begin(), lp.tokens.begin() + 3000);
                l3.label = std::string("long ") + lf + " n 3000";
                long3k_prompt = static_cast<std::uint32_t>(prompts.size());
                prompts.push_back(l3);
            }
        }
    }
    std::printf("loading...\n");
    std::fflush(stdout);
    qp::Prefill p(c);
    std::printf("loaded %s session_bytes %.1f MB\n", p.describe().c_str(), static_cast<double>(p.session_bytes()) / 1e6);
    std::vector<qp::SessionId> sid;
    for (std::uint32_t i = 0; i < max_sessions; ++i) sid.push_back(p.create_session());
    std::fflush(stdout);
    int bad = 0;

    if (mode == "bench") {
        std::vector<int> mtp_arms;
        if (mtp) mtp_arms.push_back(1);
        if (!mtp || (std::getenv("STRIX_COHORT_BENCH_NOMTP") && std::getenv("STRIX_COHORT_BENCH_NOMTP")[0] == '1')) mtp_arms.push_back(0);
        for (const int arm_mtp : mtp_arms) {
            for (const auto cn : c_list) {
                std::vector<qc::Stream> streams;
                for (std::uint32_t i = 0; i < cn; ++i) {
                    prefill(p, sid[i], prompts[i], false);
                    streams.push_back({sid[i], prompts[i].tokens[prompts[i].n - 1], prompts[i].n - 1, 0});
                }
                qc::StepOptions o;
                o.mtp = arm_mtp != 0;
                o.depth = depth;
                o.min_p = min_p;
                o.policy = policy;
                o.max_positions = c.max_positions;
                // warm-up: a few steps (graph captures of the common layouts), then the timed decode from fresh prefills
                for (int w = 0; w < 3; ++w) (void)qc::step(p, streams, o);
                streams.clear();
                for (std::uint32_t i = 0; i < cn; ++i) {
                    prefill(p, sid[i], prompts[i], false);
                    streams.push_back({sid[i], prompts[i].tokens[prompts[i].n - 1], prompts[i].n - 1, 0});
                }
                std::vector<std::uint32_t> produced(cn, 0);
                std::vector<double> done_s(cn, 0);
                std::vector<std::uint32_t> live(cn);
                for (std::uint32_t i = 0; i < cn; ++i) live[i] = i;
                qc::StepTiming tm;
                std::uint64_t steps = 0;
                // STRIX_COHORT_PROFILE_RANGE=1: nsys --capture-range=cudaProfilerApi sees only the timed decode
                static const bool profile_range = [] { const char* v = std::getenv("STRIX_COHORT_PROFILE_RANGE"); return v && v[0] == '1'; }();
                if (profile_range) cudaProfilerStart();
                const auto t0 = Clock::now();
                while (!live.empty()) {
                    std::vector<qc::Stream> active;
                    for (const auto i : live) active.push_back(streams[i]);
                    const auto r = qc::step(p, active, o, &tm);
                    ++steps;
                    std::vector<std::uint32_t> next;
                    for (std::size_t j = 0; j < live.size(); ++j) {
                        const auto i = live[j];
                        streams[i] = active[j];
                        produced[i] += static_cast<std::uint32_t>(r[j].tokens.size());
                        if (produced[i] >= g) done_s[i] = std::chrono::duration<double>(Clock::now() - t0).count();
                        else next.push_back(i);
                    }
                    live = next;
                }
                const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
                if (profile_range) cudaProfilerStop();
                std::uint64_t total = 0;
                std::string per;
                for (std::uint32_t i = 0; i < cn; ++i) {
                    total += produced[i];
                    char b[48];
                    std::snprintf(b, sizeof(b), "%s%.1f", i ? "," : "", produced[i] / done_s[i]);
                    per += b;
                }
                std::printf("BENCH C%u mtp %s: %llu tokens %.3f s aggregate %.1f tok/s per-stream [%s] steps %llu windows %u "
                            "rows/window %.2f tokens/step/session %.2f draft %.2f ms/step verify %.2f ms/step\n",
                            cn, o.mtp ? "on" : "off", static_cast<unsigned long long>(total), wall, total / wall, per.c_str(),
                            static_cast<unsigned long long>(steps), tm.windows, tm.windows ? static_cast<double>(tm.rows) / tm.windows : 0.0,
                            static_cast<double>(total) / static_cast<double>(steps) / cn, tm.draft_ms / steps, tm.verify_ms / steps);
                std::fflush(stdout);
            }
        }
        std::printf("graph %s\n", p.graph_stats().c_str());
        return 0;
    }

    // ---- reference: every prompt alone through session 0 (the single-session API), serial T = 1, MTP off, eager
    p.set_graphs(false);
    for (std::uint32_t i = 0; i < prompts.size(); ++i) {
        auto& pr = prompts[i];
        pr.seq = pr.tokens;
        p.run(pr.seq, pr.n - 1, [](const qp::Piece&) {});
        Digest chain{};
        for (std::uint32_t k = 0; k < g; ++k) {
            const auto pos = pr.n - 1 + k;
            const auto r = p.step(&pr.seq[pos], 1, pos, true);
            pr.digest.push_back(row_sha(r.logits.data(), r.logits.size()));
            chain = chain_step(chain, pos, pr.digest.back());
            pr.seq.push_back(r.argmax[0]);
        }
        std::printf("alone %u (%s): n %llu chain %s first %u last %u\n", i, pr.label.c_str(), static_cast<unsigned long long>(pr.n),
                    hex(chain).c_str(), pr.seq[pr.n], pr.seq.back());
    }
    std::fflush(stdout);
    // the row of prompt i at absolute position pos (n - 1 <= pos < n - 1 + g) must equal the reference
    const auto check_row = [&](std::uint32_t i, std::uint64_t pos, std::uint32_t token, const float* row, std::uint32_t& diff,
                               const std::string& arm) {
        const auto& pr = prompts[i];
        const auto k = pos - (pr.n - 1);
        if (k >= g) return;
        if (token != pr.seq[pr.n + k] || row_sha(row, kVocab) != pr.digest[k]) {
            if (diff == 0) std::printf("  %s: prompt %u first difference at generated row %llu\n", arm.c_str(), i, static_cast<unsigned long long>(k));
            ++diff;
        }
    };
    // a cohort member: engine session, prompt, prefill (0 = run(), else prefill_chunk rows)
    // chunk2: the rows of every chunk after the first (0: chunk again), e.g. the serve's auto sizes 128 then 48
    struct Member { qp::SessionId sid; std::uint32_t prompt; std::uint32_t chunk; std::uint32_t chunk2 = 0; };
    const auto prefill_member = [&](const Member& mb) {
        const auto& pr = prompts[mb.prompt];
        if (mb.chunk == 0) { p.run(mb.sid, pr.tokens, pr.n - 1, [](const qp::Piece&) {}); return; }
        std::uint64_t at = 0;
        while (at < pr.n - 1) at = p.prefill_chunk(mb.sid, pr.tokens, pr.n - 1, at, mb.chunk);
    };
    // arm bookkeeping: verdict + graph counter deltas (replays required where the arm must prove graphs ran)
    const auto report = [&](const std::string& arm, std::uint32_t diff, std::uint32_t failed, const qp::GraphCounters& g0,
                            bool graphs, bool need_replay, bool need_evict, const std::string& extra, bool need_static = false) {
        const auto g1 = p.graph_counters();
        const auto replays = g1.replays - g0.replays, eager = g1.eager_windows - g0.eager_windows;
        const auto evictions = g1.evictions - g0.evictions;
        bool ok = diff == 0 && failed == 0;
        std::string why;
        if (graphs && need_replay && replays == 0) { ok = false; why += " NO-REPLAY"; }
        if (need_evict && evictions == 0) { ok = false; why += " NO-EVICTION"; }
        // reuse proof: every window replayed a graph captured for other sessions (no warm-up, no capture)
        if (need_static && (g1.captures != g0.captures || eager != 0)) { ok = false; why += " NOT-REUSED"; }
        std::printf("%s graphs %s %s%s (%s; %u differing rows, %u failed entries; graph +%llu captures +%llu replays +%llu eager "
                    "+%llu evictions, eager ratio %.2f)\n", arm.c_str(), graphs ? "on" : "off",
                    diff == 0 && failed == 0 ? "BIT-IDENTICAL" : "DIFFER", why.c_str(), extra.c_str(), diff, failed,
                    static_cast<unsigned long long>(g1.captures - g0.captures), static_cast<unsigned long long>(replays),
                    static_cast<unsigned long long>(eager), static_cast<unsigned long long>(evictions),
                    replays + eager ? static_cast<double>(eager) / static_cast<double>(replays + eager) : 0.0);
        std::fflush(stdout);
        bad += !ok;
    };
    // lockstep: one row per member, members split into windows of <= 8 (the plan's split)
    const auto lockstep = [&](const std::string& arm, const std::vector<Member>& ms, bool graphs, bool need_replay, bool need_evict,
                              bool need_static = false) {
        p.set_graphs(graphs);
        for (const auto& mb : ms) prefill_member(mb);
        const auto g0 = p.graph_counters();
        std::uint32_t diff = 0, failed = 0, windows = 0;
        const auto S = static_cast<std::uint32_t>(ms.size());
        for (std::uint32_t k = 0; k < g; ++k) {
            std::uint32_t first = 0;
            for (const auto size : qc::window_sizes(S)) {
                std::vector<qp::CohortRows> cohort;
                for (std::uint32_t i = first; i < first + size; ++i) {
                    const auto& pr = prompts[ms[i].prompt];
                    const auto pos = pr.n - 1 + k;
                    cohort.push_back({ms[i].sid, &pr.seq[pos], 1, pos});
                }
                const auto r = p.verify_cohort(cohort, true);
                for (std::uint32_t j = 0; j < size; ++j) {
                    if (!r[j].ok) { ++failed; std::printf("  %s: entry failed: %s\n", arm.c_str(), r[j].error.c_str()); continue; }
                    check_row(ms[first + j].prompt, cohort[j].position, r[j].argmax[0], r[j].logits.data(), diff, arm);
                }
                p.commit_cohort(std::vector<std::uint32_t>(size, 1U));
                first += size;
                ++windows;
            }
        }
        report(arm, diff, failed, g0, graphs, need_replay, need_evict, std::to_string(windows) + " windows, " + std::to_string(S * g) + " rows",
               need_static);
    };
    // mixed: teacher-forced windows of 1..4 rows per member packed into <= 8 rows, partial commits (rollback in shared windows)
    // budget: window rows of the packing (16: windows of 9..16 rows, which must then occur)
    const auto mixed = [&](const std::string& arm, const std::vector<Member>& ms, bool graphs, bool need_replay, bool need_evict,
                           bool need_static = false, std::uint32_t budget = qc::kWindowRows) {
        std::uint32_t widest = 0;
        p.set_graphs(graphs);
        for (const auto& mb : ms) prefill_member(mb);
        const auto g0 = p.graph_counters();
        const auto S = static_cast<std::uint32_t>(ms.size());
        std::vector<std::uint64_t> pos(S);
        for (std::uint32_t i = 0; i < S; ++i) pos[i] = prompts[ms[i].prompt].n - 1;
        std::uint32_t diff = 0, failed = 0, windows = 0, step_no = 0, computed = 0;
        const auto done = [&](std::uint32_t i) { return pos[i] >= prompts[ms[i].prompt].n - 1 + g; };
        for (;;) {
            std::vector<std::uint32_t> live;
            for (std::uint32_t i = 0; i < S; ++i) if (!done(i)) live.push_back(i);
            if (live.empty()) break;
            std::vector<std::uint32_t> want;
            for (const auto i : live)  // rows stay inside the reference sequence
                want.push_back(static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(1U + (i + step_no) % 4U, prompts[ms[i].prompt].n - 1 + g - pos[i])));
            for (const auto& w : qc::pack(want, budget)) {
                widest = std::max(widest, w.rows);
                std::vector<qp::CohortRows> cohort;
                for (const auto& e : w.entries) {
                    const auto i = live[e.index];
                    cohort.push_back({ms[i].sid, &prompts[ms[i].prompt].seq[pos[i]], e.rows, pos[i]});
                }
                const auto r = p.verify_cohort(cohort, true);
                std::vector<std::uint32_t> keep;
                for (std::size_t j = 0; j < cohort.size(); ++j) {
                    const auto i = live[w.entries[j].index];
                    if (!r[j].ok) { ++failed; keep.push_back(0); std::printf("  %s: entry failed: %s\n", arm.c_str(), r[j].error.c_str()); continue; }
                    for (std::uint32_t k = 0; k < cohort[j].rows; ++k)
                        check_row(ms[i].prompt, pos[i] + k, r[j].argmax[k], r[j].logits.data() + std::uint64_t{k} * kVocab, diff, arm);
                    computed += cohort[j].rows;
                    keep.push_back(std::min(cohort[j].rows, 1U + (step_no + i) % cohort[j].rows));
                }
                p.commit_cohort(keep);
                for (std::size_t j = 0; j < cohort.size(); ++j) pos[live[w.entries[j].index]] += keep[j];
                ++windows;
            }
            ++step_no;
            if (failed) break;
        }
        if (budget > 8U && widest <= 8U) { std::printf("  %s: NO-WIDE-WINDOW (widest %u rows)\n", arm.c_str(), widest); ++bad; }
        report(arm, diff, failed, g0, graphs, need_replay, need_evict, std::to_string(windows) + " windows, " + std::to_string(computed) +
               " rows computed, widest " + std::to_string(widest), need_static);
    };
    // the serve's step: per-session drafts, packing, greedy acceptance, failure isolation
    const auto mtp_arm = [&](const std::string& arm, const std::vector<Member>& ms, bool graphs, int draft_batch = -1,
                             std::uint32_t budget = 0) {
        std::uint32_t widest = 0;
        if (!mtp) return;
        p.set_graphs(graphs);
        for (const auto& mb : ms) prefill_member(mb);
        const auto g0 = p.graph_counters();
        const auto S = static_cast<std::uint32_t>(ms.size());
        std::vector<qc::Stream> streams;
        for (const auto& mb : ms) {
            const auto& pr = prompts[mb.prompt];
            streams.push_back({mb.sid, pr.tokens[pr.n - 1], pr.n - 1, 0});
        }
        qc::StepOptions o;
        o.mtp = true;
        o.depth = depth;
        o.min_p = min_p;
        o.policy = policy;
        o.want_logits = true;  // max_positions 0: the engine's
        o.draft_batch = draft_batch;
        if (budget) o.budget = budget;
        std::vector<std::uint32_t> produced(S, 0), live(S);
        for (std::uint32_t i = 0; i < S; ++i) live[i] = i;
        std::uint32_t diff = 0, failed = 0, steps = 0, retried = 0;
        qc::StepTiming tm;
        while (!live.empty()) {
            std::vector<qc::Stream> active;
            for (const auto i : live) active.push_back(streams[i]);
            const auto before = active;
            const auto r = qc::step(p, active, o, &tm);
            std::vector<std::uint32_t> next;
            for (std::size_t j = 0; j < live.size(); ++j) {
                const auto i = live[j];
                retried += r[j].retried;
                widest = std::max(widest, r[j].window_rows);
                if (r[j].failed) { ++failed; std::printf("  %s: session failed: %s\n", arm.c_str(), r[j].error.c_str()); continue; }
                for (std::size_t k = 0; k < r[j].tokens.size(); ++k)
                    check_row(ms[i].prompt, before[j].position + k, r[j].tokens[k], r[j].logits.data() + k * kVocab, diff, arm);
                produced[i] += static_cast<std::uint32_t>(r[j].tokens.size());
                streams[i] = active[j];
                if (produced[i] < g) next.push_back(i);
            }
            live = next;
            ++steps;
        }
        std::uint64_t total = 0;
        for (const auto x : produced) total += x;
        char extra[200];
        std::snprintf(extra, sizeof(extra), "depth %u, %u steps, %u windows, %.2f rows/window, widest %u, %.2f tokens/step/session, %u retried",
                      depth, steps, tm.windows, tm.windows ? static_cast<double>(tm.rows) / tm.windows : 0.0, widest,
                      static_cast<double>(total) / steps / S, retried);
        if (budget > 8U && widest <= 8U) { std::printf("  %s: NO-WIDE-WINDOW (widest %u rows)\n", arm.c_str(), widest); ++bad; }
        report(arm, diff, failed, g0, graphs, false, false, extra);
    };
    // decode-first admission: the decoders step in lockstep while the joiner is prefilled one chunk between windows; it
    // then joins the cohort. Every row of every member is compared.
    // overlap: the joiner's chunk is enqueued on the prefill stream (prefill_chunk_async) before each decode window and
    // waited for after it, so it computes concurrently with the window
    // poll (with overlap): the decoders keep stepping while the chunk runs (prefill_poll after each window); the chunk is
    // waited for only when no decoder is left; at least 2 windows must run while a chunk is in flight
    const auto admit_arm = [&](const std::string& arm, const std::vector<Member>& decoders, const Member& joiner, bool graphs,
                               bool overlap = false, bool poll = false) {
        std::uint32_t pending_windows = 0;
        p.set_graphs(graphs);
        for (const auto& mb : decoders) prefill_member(mb);
        const auto g0 = p.graph_counters();
        std::vector<Member> ms = decoders;
        ms.push_back(joiner);
        std::vector<std::uint64_t> k(ms.size(), 0);  // generated rows done per member
        const auto& jp = prompts[joiner.prompt];
        std::uint64_t at = 0;
        bool joined = false;
        std::uint32_t diff = 0, failed = 0, windows = 0, chunks = 0;
        for (;;) {
            std::vector<std::uint32_t> live;
            for (std::uint32_t i = 0; i < ms.size(); ++i)
                if (k[i] < g && (i + 1 < ms.size() || joined)) live.push_back(i);
            if (live.empty() && joined) break;
            const std::uint32_t chunk_rows = chunks == 0 || joiner.chunk2 == 0 ? joiner.chunk : joiner.chunk2;
            if (!joined && overlap && !p.prefill_pending()) p.prefill_chunk_async(joiner.sid, jp.tokens, jp.n - 1, at, chunk_rows);
            if (!live.empty()) {
                std::vector<qp::CohortRows> cohort;
                for (const auto i : live) {
                    const auto& pr = prompts[ms[i].prompt];
                    const auto pos = pr.n - 1 + k[i];
                    cohort.push_back({ms[i].sid, &pr.seq[pos], 1, pos});
                }
                const auto r = p.verify_cohort(cohort, true);
                for (std::size_t j = 0; j < live.size(); ++j) {
                    if (!r[j].ok) { ++failed; continue; }
                    check_row(ms[live[j]].prompt, cohort[j].position, r[j].argmax[0], r[j].logits.data(), diff, arm);
                }
                p.commit_cohort(std::vector<std::uint32_t>(live.size(), 1U));
                for (const auto i : live) ++k[i];
                ++windows;
            }
            if (!joined) {  // one chunk of the joiner after each decode window
                if (overlap && poll && !live.empty()) {
                    ++pending_windows;  // the window above ran while the chunk was in flight
                    const auto r = p.prefill_poll();
                    if (!r) continue;   // still running: the decoders take another step
                    at = r->length;
                } else {
                    at = overlap ? p.prefill_wait().length : p.prefill_chunk(joiner.sid, jp.tokens, jp.n - 1, at, chunk_rows);
                }
                ++chunks;
                joined = at == jp.n - 1;
            }
        }
        if (poll && pending_windows < 2) { std::printf("  %s: TOO-FEW-WINDOWS-DURING-CHUNK (%u)\n", arm.c_str(), pending_windows); ++bad; }
        report(arm, diff, failed, g0, graphs, false, false,
               std::to_string(windows) + " windows, joiner prefilled in " + std::to_string(chunks) + " chunks of " + std::to_string(joiner.chunk) +
               (poll ? ", " + std::to_string(pending_windows) + " windows while a chunk was in flight" : std::string()));
    };

    std::vector<int> graph_arms;
    {
        const char* v = std::getenv("STRIX_COHORT_GRAPHS");
        const std::string gm = v ? v : "both";
        if (gm != "off") graph_arms.push_back(1);
        if (gm != "on") graph_arms.push_back(0);
    }
    const auto members = [&](std::uint32_t S, std::uint32_t chunk_parity) {
        std::vector<Member> ms;
        for (std::uint32_t i = 0; i < S; ++i) ms.push_back({sid[i], i, i % 2 == chunk_parity ? 256U : 0U});
        return ms;
    };
    for (const int graphs : graph_arms) {
        for (const auto S : s_list) {
            const auto tag = "S=" + std::to_string(S);
            lockstep(tag + " lockstep", members(S, 1), graphs != 0, true, false);
            mixed(tag + " mixed+rollback", members(S, 0), graphs != 0, true, false);
            mtp_arm(tag + " mtp", members(S, 1), graphs != 0);
        }
    }
    // a short (64) and a long (4K) context in the same windows (QSA below and above the pooled-selection regime)
    if (short_prompt != kNone) {
        std::vector<Member> ms{{sid[0], short_prompt, 0}, {sid[1], 0, 0}};
        if (long_prompt != kNone) ms.push_back({sid[2], long_prompt, 0});
        for (const int graphs : graph_arms) {
            lockstep("context-mix lockstep", ms, graphs != 0, true, false);
            mixed("context-mix mixed+rollback", ms, graphs != 0, false, false);
        }
    }
    // two or more sessions past 2048 positions in one window, not in slot 0, around a short one: the multi-segment block
    // pool / block score launches then map pooled blocks and score waves of several sessions (non-zero prefixes, a
    // session without work between two with work)
    if (long3k_prompt != kNone && max_sessions >= 4) {
        const std::vector<Member> a{{sid[0], long_prompt, 0}, {sid[1], short_prompt, 0}, {sid[2], long3k_prompt, 0}};
        const std::vector<Member> b{{sid[0], short_prompt, 0}, {sid[1], long_prompt, 0}, {sid[2], 0, 0}, {sid[3], long3k_prompt, 0}};
        for (const int graphs : graph_arms) {
            lockstep("two-long {4K,64,3K} lockstep", a, graphs != 0, false, false);
            mixed("two-long {4K,64,3K} mixed+rollback", a, graphs != 0, false, false);
            lockstep("two-long {64,4K,1K,3K} lockstep", b, graphs != 0, false, false);
            mixed("two-long {64,4K,1K,3K} mixed+rollback", b, graphs != 0, false, false);
        }
        mtp_arm("two-long {4K,64,3K} mtp", a, true);
        mtp_arm("two-long {64,4K,1K,3K} mtp", b, true);
    }
    // LRU eviction inside the gate: a cache of 2 graphs under changing layouts
    if (std::find(graph_arms.begin(), graph_arms.end(), 1) != graph_arms.end()) {
        const auto cap = p.graph_counters().cap;
        p.set_graph_cap(2);
        const auto S = std::min<std::uint32_t>(4, max_sessions);
        mixed("eviction(cap 2) mixed+rollback", members(S, 0), true, false, true);
        lockstep("eviction(cap 2) lockstep", members(S, 1), true, false, false);
        p.set_graph_cap(cap);
    }
    // session churn (destroy + create mid-gate: graph purge), session 0 in a cohort, unaligned (100) and 1024-row chunks
    if (max_sessions >= 3) {
        p.destroy_session(sid[1]);
        sid[1] = p.create_session();
        const std::vector<Member> ms{{0, 0, 100}, {sid[1], 1, 1024}, {sid[2], 2, 0}};
        for (const int graphs : graph_arms) {
            lockstep("churn+session0 lockstep", ms, graphs != 0, true, false);
            mixed("churn+session0 mixed+rollback", ms, graphs != 0, false, false);
        }
        mtp_arm("churn+session0 mtp", ms, true);
    }
    // batched drafts == per-session drafts (tokens and probability bits), chains of mixed depths (drop-outs at every
    // step), then a serve step (batched drafts) whose rows must equal alone
    const auto draft_compare = [&](const std::string& arm, const std::vector<Member>& ms) {
        if (!mtp) return;
        p.set_graphs(true);
        for (const auto& mb : ms) prefill_member(mb);
        const auto g0 = p.graph_counters();
        const auto S = static_cast<std::uint32_t>(ms.size());
        std::vector<qc::Stream> streams;
        for (const auto& mb : ms) {
            const auto& pr = prompts[mb.prompt];
            streams.push_back({mb.sid, pr.tokens[pr.n - 1], pr.n - 1, 0});
        }
        qc::StepOptions o;
        o.mtp = true;
        o.depth = depth;
        o.min_p = min_p;
        o.policy = policy;
        o.want_logits = true;
        o.draft_batch = 1;
        std::vector<std::uint32_t> produced(S, 0), live(S);
        for (std::uint32_t i = 0; i < S; ++i) live[i] = i;
        std::uint32_t diff = 0, failed = 0, steps = 0, draft_diff = 0, chains = 0;
        while (!live.empty()) {
            std::vector<qp::DraftRequest> req;
            for (std::size_t j = 0; j < live.size(); ++j) {
                // step 0: depth 0 next to live chains {0,3,3,1,..}; step 1: a single live chain (the draft() fallback);
                // later: depths 1..7 rotating, so chains drop out at different steps
                std::uint32_t d = 1U + (live[j] + steps) % 7U;
                if (steps == 0) d = j == 0 ? 0U : j == live.size() - 1 ? 1U : 3U;
                if (steps == 1) d = j == 1 ? 2U : 0U;
                req.push_back({streams[live[j]].session, streams[live[j]].pending, d, min_p});
            }
            std::vector<qp::Draft> alone;
            for (const auto& r : req) alone.push_back(p.draft(r.session, r.pending, r.depth, r.min_probability));
            const auto batched = p.draft_cohort(req);
            for (std::size_t j = 0; j < req.size(); ++j) {
                ++chains;
                const bool same = alone[j].tokens == batched[j].tokens &&
                    std::memcmp(alone[j].probability.data(), batched[j].probability.data(), 4 * std::min(alone[j].probability.size(),
                        batched[j].probability.size())) == 0 && alone[j].probability.size() == batched[j].probability.size();
                if (!same) {
                    if (draft_diff == 0) std::printf("  %s: step %u chain of prompt %u differs (%zu vs %zu drafts)\n", arm.c_str(), steps,
                                                     ms[live[j]].prompt, alone[j].tokens.size(), batched[j].tokens.size());
                    ++draft_diff;
                }
            }
            std::vector<qc::Stream> active;
            for (const auto i : live) active.push_back(streams[i]);
            const auto before = active;
            const auto r = qc::step(p, active, o);
            std::vector<std::uint32_t> next;
            for (std::size_t j = 0; j < live.size(); ++j) {
                const auto i = live[j];
                if (r[j].failed) { ++failed; continue; }
                for (std::size_t k = 0; k < r[j].tokens.size(); ++k)
                    check_row(ms[i].prompt, before[j].position + k, r[j].tokens[k], r[j].logits.data() + k * kVocab, diff, arm);
                produced[i] += static_cast<std::uint32_t>(r[j].tokens.size());
                streams[i] = active[j];
                if (produced[i] < g) next.push_back(i);
            }
            live = next;
            ++steps;
        }
        std::printf("%s drafts %s (%u chains compared, %u differ)\n", arm.c_str(), draft_diff == 0 ? "BIT-IDENTICAL" : "DIFFER", chains, draft_diff);
        bad += draft_diff != 0;
        report(arm, diff, failed, g0, true, false, false, std::to_string(steps) + " steps");
    };
    draft_compare("draft-batch S=4", members(std::min<std::uint32_t>(4, max_sessions), 1));
    if (max_sessions >= 8) draft_compare("draft-batch S=8", members(8, 0));
    // per-session drafts (STRIX_QRTX_DRAFT_BATCH=0 path) in the serve step
    mtp_arm("S=4 mtp draft-batch off", members(std::min<std::uint32_t>(4, max_sessions), 1), true, 0);
    // graphs that bake session pointers (STRIX_QRTX_GRAPH_TABLE=0) vs the descriptor table: both must equal alone
    {
        const auto S = std::min<std::uint32_t>(4, max_sessions);
        p.set_graph_table(false);
        lockstep("table-off S=" + std::to_string(S) + " lockstep", members(S, 1), true, true, false);
        mixed("table-off S=" + std::to_string(S) + " mixed+rollback", members(S, 0), true, true, false);
        mtp_arm("table-off S=" + std::to_string(S) + " mtp", members(S, 1), true);
        p.set_graph_table(true);
        // per-session GDN/QSA launches (STRIX_QRTX_MULTISEG=0); every other cohort arm runs the multi-segment cores
        p.set_multiseg(false);
        lockstep("multiseg-off S=" + std::to_string(S) + " lockstep", members(S, 1), true, true, false);
        mixed("multiseg-off S=" + std::to_string(S) + " mixed+rollback", members(S, 0), true, false, false);
        mtp_arm("multiseg-off S=" + std::to_string(S) + " mtp", members(S, 1), false);
        p.set_multiseg(true);
        // table graphs are reused by other sessions: warm the layouts with sessions A (lockstep once; mixed twice, so
        // every layout seen once in the first pass is captured in the second), then the same layouts from sessions B (other
        // sessions, other GDN parities per slot) must replay only: no warm-up window, no capture
        std::vector<Member> warm, rot;
        for (std::uint32_t i = 0; i < S; ++i) {
            warm.push_back({sid[i], i, 0});
            rot.push_back({sid[(i + (max_sessions >= 2 * S ? S : 1)) % max_sessions], i, 0});
        }
        lockstep("table-on warm S=" + std::to_string(S) + " lockstep", warm, true, true, false);
        mixed("table-on warm S=" + std::to_string(S) + " mixed+rollback (1)", warm, true, false, false);
        mixed("table-on warm S=" + std::to_string(S) + " mixed+rollback (2)", warm, true, false, false);
        lockstep("table-on rotated sessions S=" + std::to_string(S) + " lockstep", rot, true, true, false, true);
        mixed("table-on rotated sessions S=" + std::to_string(S) + " mixed+rollback (mixed parity)", rot, true, true, false, true);
    }
    // windows of 9..16 rows (STRIX_QRTX_WINDOW_ROWS=16): the row-independent families on the prefill-shaped kernels over 16
    // padded rows; every row of every session must still equal it alone
    if (max_sessions >= 8) {
        for (const int graphs : graph_arms) {
            mixed("wide16 S=8 mixed+rollback", members(8, 0), graphs != 0, false, false, false, 16);
            mixed("wide16 S=4 mixed+rollback", members(4, 1), graphs != 0, false, false, false, 16);
        }
        mtp_arm("wide16 S=8 mtp (1-draft windows)", members(8, 1), true, -1, 16);
        mtp_arm("wide16 S=4 mtp", members(4, 0), true, -1, 16);
        mtp_arm("wide16 S=8 mtp graphs off", members(8, 0), false, -1, 16);
    }
    // decode-first admission: prefill_chunk(100) of a new session between the other sessions' windows
    if (max_sessions >= 4)
        for (const int graphs : graph_arms)
            admit_arm("admit(chunk 100) lockstep", {{sid[0], 0, 0}, {sid[2], 2, 256}}, Member{sid[3], 3, 100}, graphs != 0);
    // the same with the joiner's chunks computing concurrently on the prefill stream (enable_prefill_overlap)
    if (max_sessions >= 4) {
        p.enable_prefill_overlap(1024);
        for (const int graphs : graph_arms) {
            admit_arm("admit-overlap(chunk 100) lockstep", {{sid[0], 0, 0}, {sid[2], 2, 0}}, Member{sid[3], 3, 100}, graphs != 0, true);
            admit_arm("admit-overlap(chunk 256) lockstep", {{sid[1], 1, 0}, {sid[3], 3, 0}}, Member{sid[0], 0, 256}, graphs != 0, true);
            // the serve's overlap1024 + polling: one 1024-row logical block, several decode windows while it runs
            admit_arm("admit-overlap-poll(chunk 1024) lockstep", {{sid[0], 0, 0}, {sid[1], 1, 0}, {sid[2], 2, 0}}, Member{sid[3], 3, 1024},
                      graphs != 0, true, true);
        }
        if (long_prompt != kNone)
            admit_arm("admit-overlap-poll long(chunk 1024) lockstep", {{sid[0], 0, 0}, {sid[2], 2, 0}}, Member{sid[1], long_prompt, 1024}, true,
                      true, true);
    }
    // the long (4K) prompt chunked past 2048 positions (the QSA pooled-selection regime, pooled-cache recompute at chunk
    // boundaries): overlapped in 100-row chunks, overlapped 128 then 48s (the serve's auto sizes), synchronous 37-row chunks
    // (a partial pooled block at every boundary)
    if (max_sessions >= 4 && long_prompt != kNone) {
        admit_arm("admit-overlap long(chunk 100) lockstep", {{sid[0], 0, 0}, {sid[2], 2, 0}}, Member{sid[3], long_prompt, 100}, true, true);
        admit_arm("admit-overlap long(chunk 128,48..) lockstep", {{sid[0], 0, 0}, {sid[2], 2, 0}}, Member{sid[1], long_prompt, 128, 48},
                  true, true);
        admit_arm("admit long(chunk 37) lockstep", {{sid[0], 0, 0}, {sid[2], 2, 0}}, Member{sid[3], long_prompt, 37}, true, false);
    }
    // overlap beside the serve's real decode: the other sessions run MTP steps (batched drafts, multi-row windows) while the
    // joiner's chunks are in flight, with the graph cache emptied first so captures happen during chunks (depth 1: four
    // layouts, so some layout repeats while the chunks run); at the join, the joiner's MTP state (held row, MTP KV/index)
    // must draft exactly as the same prompt prefilled alone (tokens and probability bits); then it decodes in the cohort,
    // its drafts compared with per-session drafts, every row compared with alone.
    const auto admit_mtp = [&](const std::string& arm, const std::vector<Member>& decoders, const Member& joiner, qp::SessionId ref) {
        if (!mtp) return;
        p.set_graphs(true);
        for (const auto& mb : decoders) prefill_member(mb);
        const auto& jp = prompts[joiner.prompt];
        p.run(ref, jp.tokens, jp.n - 1, [](const qp::Piece&) {});  // the joiner's reference state (prefilled alone, synchronously)
        const auto cap = p.graph_counters().cap;
        p.set_graph_cap(1);
        p.set_graph_cap(cap);
        const auto g0 = p.graph_counters();
        std::vector<Member> ms = decoders;
        ms.push_back(joiner);
        const auto J = static_cast<std::uint32_t>(ms.size() - 1);
        std::vector<qc::Stream> streams;
        for (const auto& mb : ms) {
            const auto& pr = prompts[mb.prompt];
            streams.push_back({mb.sid, pr.tokens[pr.n - 1], pr.n - 1, 0});
        }
        qc::StepOptions o;
        o.mtp = true;
        o.depth = 1;
        o.min_p = min_p;
        o.want_logits = true;
        o.draft_batch = 1;
        std::vector<std::uint32_t> produced(ms.size(), 0);
        std::uint32_t diff = 0, failed = 0, steps = 0, chunks = 0, draft_diff = 0, overlapped_captures = 0;
        std::uint64_t at = 0;
        bool joined = false;
        for (;;) {
            std::vector<std::uint32_t> live;
            for (std::uint32_t i = 0; i < ms.size(); ++i)
                if (produced[i] < g && (i < J || joined)) live.push_back(i);
            if (live.empty() && joined) break;
            if (!joined) p.prefill_chunk_async(joiner.sid, jp.tokens, jp.n - 1, at, joiner.chunk);
            const auto c0 = p.graph_counters().captures;
            if (!live.empty()) {
                if (joined && produced[J] < g) {  // the joiner's chain batched with the others == its chain alone
                    std::vector<qp::DraftRequest> req;
                    for (const auto i : live) req.push_back({streams[i].session, streams[i].pending, 3, min_p});
                    const auto batched = p.draft_cohort(req);
                    for (std::size_t j = 0; j < live.size(); ++j) {
                        const auto alone = p.draft(req[j].session, req[j].pending, 3, min_p);
                        if (alone.tokens != batched[j].tokens || alone.probability != batched[j].probability) ++draft_diff;
                    }
                }
                std::vector<qc::Stream> active;
                for (const auto i : live) active.push_back(streams[i]);
                const auto before = active;
                const auto r = qc::step(p, active, o);
                for (std::size_t j = 0; j < live.size(); ++j) {
                    const auto i = live[j];
                    if (r[j].failed) { ++failed; continue; }
                    for (std::size_t k = 0; k < r[j].tokens.size(); ++k)
                        check_row(ms[i].prompt, before[j].position + k, r[j].tokens[k], r[j].logits.data() + k * kVocab, diff, arm);
                    produced[i] += static_cast<std::uint32_t>(r[j].tokens.size());
                    streams[i] = active[j];
                }
                ++steps;
            }
            if (!joined) {
                overlapped_captures += static_cast<std::uint32_t>(p.graph_counters().captures - c0);
                at = p.prefill_wait().length;
                ++chunks;
                joined = at == jp.n - 1;
                if (joined) {  // the joiner's MTP state == the same prompt prefilled alone
                    const auto pend = jp.tokens[jp.n - 1];
                    const auto a = p.draft(ref, pend, 7, min_p);
                    const auto b = p.draft(joiner.sid, pend, 7, min_p);
                    const bool same = a.tokens == b.tokens && a.probability == b.probability;
                    std::printf("  %s: joiner draft chain after the overlapped prefill %s (%zu drafts)\n", arm.c_str(),
                                same ? "BIT-IDENTICAL to alone" : "DIFFERS from alone", b.tokens.size());
                    draft_diff += same ? 0U : 1U;
                }
            }
        }
        const bool captured = overlapped_captures > 0;
        if (!captured) std::printf("  %s: NO-CAPTURE-DURING-CHUNK\n", arm.c_str());
        std::printf("%s drafts %s (%u differ)\n", arm.c_str(), draft_diff == 0 ? "BIT-IDENTICAL" : "DIFFER", draft_diff);
        bad += (draft_diff != 0) + !captured;
        report(arm, diff, failed, g0, true, true, false,
               std::to_string(steps) + " steps, " + std::to_string(chunks) + " overlapped chunks, " + std::to_string(overlapped_captures) +
               " captures during chunks");
    };
    if (max_sessions >= 8) admit_mtp("admit-overlap mtp(chunk 48)", {{sid[0], 0, 0}, {sid[2], 2, 0}}, Member{sid[3], 3, 48}, sid[7]);
    std::printf("graph %s\n", p.graph_stats().c_str());
    std::printf("%s\n", bad == 0 ? "COHORT GATE PASS" : "COHORT GATE FAIL");
    return bad == 0 ? 0 : 3;
}
