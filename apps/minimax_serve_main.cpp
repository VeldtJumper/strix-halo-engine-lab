// strix-minimax-serve: the MiniMax M3.1 Flash TP2 engine behind an OpenAI-compatible HTTP API (both ranks run it;
// rank 0 serves HTTP). The DS serve recipe: rank 0 owns the queue and a continuous-batching scheduler over the
// engine's sequence slots and sends one fixed-size control block per round (prefill / unit / step / draft / heartbeat /
// shutdown); rank 1 replays it, so both ranks make the same engine calls and produce the same tokens (greedy, exact).
//
// usage: strix-minimax-serve MODEL_DIR --rank R --transport T [--port 8080] [--bind 127.0.0.1] [--slots 8]
//        [--capacity 8256] [--max-rows 4096] [--model-id minimax-m3.1-flash] [--max-tokens 4096]
//        [--dspark-mode off|auto|on] [--dspark DIR [--dspark-exl3 DIR]] [--admin] [--resume-check P.u32[:k/k...][,...] [--resume-check-nested SHORT.u32 [--resume-check-child-layers K]] [--resume-check-tokens 256] [--check-only]]
//   TRANSPORT: usb4-auto | tcp-listen:PORT | tcp-connect:HOST:PORT
// Routes (rank 0): POST /v1/chat/completions, POST /v1/completions (stream or not), GET /v1/models, GET /health,
// GET /metrics (metrics text), GET /api/status[?detail=1] (strix-serve-status-v1), POST /tokenize; with --admin,
// GET / POST /admin/sched (scheduler policy at runtime, for A/B in one load). The API and its observability are in
// src/minimax/serve/minimax_api.* and minimax_observe.* (host only, HTTP threads); this file gives them the scheduler's
// per-request timings, its mm_* lines and /admin/sched. Sampling parameters are accepted and ignored: decoding is
// greedy (said in the reply).
//
// Scheduler (qualification notes), rank 0's decisions only (rank 1 runs the commands):
//   STRIX_MM_SERVE_SCHED=off|interleave (default interleave; GPU qualification pending): off = an admitted prompt's whole prefill runs at
//     once and the decoding requests wait for all of it (the first scheduler). interleave = the prefill runs as a
//     resumable task in LAYER units (Engine::prefill_begin / prefill_resume: the same kernel calls and rows as one
//     prefill, so the bits are those of the request alone; row units or batched prompts are not exact here: the prefill
//     split-K policy follows the call's row count). A decoder protects a candidate only when its first emission
//     strictly predates that candidate's submission. Without protection or warranted nesting, use OFF's FIFO whole
//     admission and admit again before decoding. Short-first requires a protected FIFO head. Bounded tasks use
//     STRIX_MM_SERVE_SERVICE_MS (250) when idle; with decoders they use the remaining gap budget, recent layer-cost
//     upper samples and a control margin. Rank 1 only replays these decisions.
//   Default auto uses structural-v1 admission; prose stays plain. --dspark-prior unqualified-v1 restores conservative plain.
//   --dspark-mode off disables drafting, taps and appends; on retains the legacy opt-in scheduler.
//   --dspark supplies assets, independently of mode; optional local MODEL_DIR/dspark discovery never downloads.
//   --dspark: DSpark windows inside the batch steps, sharing the step's 8 rows (n decoding rows + drafts <= 8: C1 up to 7
//     drafts, C8 none). Requests admitted while <= STRIX_MM_SERVE_DSPARK_INFLIGHT (default 1: measurement C2 measured slower) are in flight keep a
//     drafter context (prefill taps + every committed row); above it appends stop and those contexts are stale for good.
//     Each cycle a gate per decoding count decides whether to draft; the drafting slots' windows are allocated jointly
//     (expected tokens per measured cycle cost over every feasible total). Drafts change acceptance only, never output.
//     OFF by default (measurement: C1 +8 % aggregate but prose -10 %, C2 -3 %): STRIX_MM_SERVE_DSPARK=1 (or /admin/sched) turns
//     the loaded drafter on.
#include "minimax/serve/minimax_api.hpp"
#include "minimax/serve/minimax_sched.hpp"
#include "minimax/serve/minimax_lead.hpp"
#include "minimax/serve/minimax_control.hpp"
#include "minimax/serve/minimax_observe.hpp"
#include "minimax/serve/minimax_tokenizer.hpp"
#include "strix/minimax/engine.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "minimax/serve/minimax_dspark_policy.hpp"
#include <filesystem>
#include "strix/dist/communicator.hpp"
#include "minimax/xfer/minimax_xfer.hpp"
#include "hip_minimax_dspark.hpp"
#include "minimax/session/runtime.hpp"
#include "strix/minimax/session_startup.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/session_api.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

namespace mm = strix::minimax::engine;
namespace sv = strix::serve;
namespace ms = strix::minimax::serve;
namespace ds = strix::minimax::dspark;
using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
constexpr std::uint32_t kStepRows = 8;  // Engine::step's row limit

std::uint64_t fnv(const void* p, std::size_t bytes, std::uint64_t h = 14695981039346656037ULL) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}
std::string hex(std::uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}
double env_num(const char* name, double def) {
    const char* v = std::getenv(name);
    return v == nullptr || *v == 0 ? def : std::atof(v);
}

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, std::string& receipt) {
    const std::chrono::seconds limit(300);
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon), rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "tcp-listen") return strix::dist::tcp_listen(static_cast<std::uint16_t>(std::stoul(rest)), limit);
    if (kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return strix::dist::tcp_connect(rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), limit);
    }
    if (kind == "usb4-auto") {
        strix::dist::Usb4AutoOptions o;
        o.minimax_compat = true;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        receipt = topology.receipt();
        return t;
    }
    throw std::invalid_argument("unknown transport " + spec);
}

// The transport shared in turn by the communicator and the exchange module (as strix-minimax-rank).
class Borrowed final : public strix::dist::Transport {
public:
    explicit Borrowed(strix::dist::Transport& t) : t_(t) {}
    void send(std::span<const std::byte> m) override { t_.send(m); }
    void receive(std::span<std::byte> m) override { t_.receive(m); }
    void exchange(std::span<const std::byte> o, std::span<std::byte> i) override { t_.exchange(o, i); }
    [[nodiscard]] std::string name() const override { return t_.name(); }
    void abort(const std::string& why) override { t_.abort(why); }

private:
    strix::dist::Transport& t_;
};

// ---- control channel: one 512-byte block per round, rank 0 -> rank 1 (rank 1 answers with its own) ----
// prefill: opens the resumable prefill of `plen` tokens (they follow in kTokensPerMessage chunks) in `pslot`; pflags bit 0
//   = the prefill's taps go to the slot's drafter context. unit: the next `layers` layers of the open prefill. step: the
//   rows (one row per slot, or a slot's verify window: its anchor then its drafts at consecutive positions); `append` bit
//   s = slot s's committed rows (anchor + accepted drafts, from the step's tokens on each rank) are appended to its drafter
//   context. draft: the 7-row DSpark block + 7 Markov steps of each of `dslots` slots from its anchor at its position.
//   retire: append is a mask of contexts whose staging/prefill append ownership is revoked at this boundary.
//   taps: the engine's draft taps on (pflags 1) / off (0), only with no prefill open (DSpark switched at runtime).
using Command = ms::Command;
using ms::Control;
using ms::exchange;
constexpr std::size_t kTokensPerMessage = 16384;  // 64 KiB per control exchange

// Prompt tokens of a prefill, rank 0 -> rank 1.
void exchange_tokens(mm::Engine& eng, std::vector<std::uint32_t>& tokens, std::size_t n, bool sender) try {
    if (!sender) tokens.assign(n, 0);
    std::vector<std::uint32_t> in(kTokensPerMessage);
    for (std::size_t c0 = 0; c0 < n; c0 += kTokensPerMessage) {
        const std::size_t m = std::min(kTokensPerMessage, n - c0);
        std::vector<std::uint32_t> out(m, 0);
        if (sender) std::copy_n(tokens.begin() + static_cast<long>(c0), m, out.begin());
        eng.exchange_bytes(out.data(), in.data(), m * 4);
        if (!sender) std::copy_n(in.begin(), m, tokens.begin() + static_cast<long>(c0));
    }
} catch (...) {
    eng.abort("control channel: protocol failure");
    throw;
}

// ---- the engine calls of every command, made identically by both ranks ----
using ms::lead::Drafts;
struct Runner {
    void send_prompt(std::vector<std::uint32_t>& prompt) { exchange_tokens(eng, prompt, prompt.size(), true); }
    static void synchronize() { ds::DeviceDrafter::device_synchronize(); }
    mm::Engine& eng;
    ds::DeviceDrafter* dd = nullptr;  // DSpark drafter (both ranks), nullptr = none
    std::uint32_t v0 = 0, vn = 0;     // this rank's vocabulary half (drafter Markov steps)

    std::uint64_t local_id = 0, control_round = 0;
    std::uint32_t last_next = 0, contexts = 0;
    void retire_contexts(std::uint32_t mask) {
        contexts &= ~mask;
        if (dd != nullptr) for (std::uint32_t s = 0; s < eng.slots(); ++s)
            if ((mask >> s) & 1U) dd->drop_staged(s);
    }
    void prefill_begin(std::uint32_t slot, std::span<const std::uint32_t> prompt, bool context,
                       std::uint64_t id, std::uint64_t parent = 0, std::uint64_t generation = 0) {
        contexts = (contexts & ~(1U << slot)) | (context ? 1U << slot : 0U);
        mm::Engine::PrefillTapSink sink;
        if (dd != nullptr && context)
            sink = [this](const float* taps, std::size_t n, std::uint32_t s, std::uint64_t pos0) { if ((contexts >> s) & 1U) dd->append(s, taps, n, pos0, nullptr); };
        if (id == 0) throw std::runtime_error("prefill begin: missing task ID");
        eng.prefill_begin_task(id, parent, generation, slot, prompt, std::move(sink));
    }
    void prefill_check(std::uint64_t id, std::uint32_t next) const {
        const auto p = eng.prefill_progress();
        const auto result = eng.prefill_result();
        if (p.open || p.id != id || result.id != id || result.next != next || last_next != next) {
            eng.abort("control channel: completion mismatch");
            throw std::runtime_error("control channel: completion identity / token mismatch");
        }
    }
    void require_active(std::uint64_t id, unsigned bank) const {
        if (!eng.prefill_active() || eng.prefill_progress().id != id || eng.prefill_progress().bank != bank)
            throw std::runtime_error("control channel: wrong active task");
    }
    bool prefill_unit(unsigned layers, std::uint32_t* next) {
        const bool done = eng.prefill_resume(ms::resume_layers(layers), &last_next);
        if (done && next) *next = last_next;
        return done;
    }
    void prefill_cancel(std::uint64_t id) {
        if (id == 0) throw std::runtime_error("prefill cancel: missing task ID");
        eng.prefill_cancel(id);
    }
    void cancel_all() {
        if (eng.prefill_active()) prefill_cancel(eng.prefill_progress().id);
        if (eng.prefill_progress(true).open) prefill_cancel(eng.prefill_progress(true).id);
    }
    // DSpark block + Markov chain (one 8-byte candidate exchange per step) + confidences, as strix-minimax-rank --spec auto.
    Drafts draft(std::uint32_t slot, std::uint32_t anchor, std::uint64_t pos) {
        if (slot >= eng.slots() || ((contexts >> slot) & 1U) == 0) throw std::runtime_error("draft: retired drafter context");
        Drafts out;
        dd->forward_block(slot, eng.embed_row(anchor), eng.embed_row(ds::noise_token), pos, nullptr);
        const float* logits = eng.head_logits(dd->y(), ds::block);
        std::array<std::uint32_t, ds::block> prev{};
        std::uint32_t t = anchor;
        for (std::uint32_t i = 0; i < ds::block; ++i) {
            prev[i] = t;
            const ds::Candidate mine = dd->markov_step_host(t, logits + static_cast<std::size_t>(i) * vn, v0, v0 + vn, nullptr);
            const auto theirs = eng.peer2(__builtin_bit_cast(std::uint32_t, mine.value), mine.id);
            t = ds::combine(mine, ds::Candidate{__builtin_bit_cast(float, theirs[0]), theirs[1]}).id;
            out.tokens[i] = t;
        }
        out.confidence = dd->confidence_host(prev, nullptr);
        return out;
    }
    // One step; then each slot of `append` gets its committed rows appended (its anchor + the drafts the target accepted:
    // the window's contiguous tap rows, before the next call overwrites them).
    std::vector<std::uint32_t> step(std::span<const mm::Row> rows, std::uint32_t append) {
        if ((append & ~contexts) != 0) throw std::runtime_error("step: retired drafter context");
        auto g = eng.step(rows);
        if (dd != nullptr && append != 0) {
            const float* taps = eng.last_taps();
            for (std::size_t i0 = 0; i0 < rows.size();) {
                std::size_t i1 = i0 + 1;
                while (i1 < rows.size() && rows[i1].slot == rows[i0].slot) ++i1;
                if ((append >> rows[i0].slot) & 1U) {
                    std::vector<std::uint32_t> d;
                    for (std::size_t i = i0 + 1; i < i1; ++i) d.push_back(rows[i].token);
                    const auto v = ds::verify_greedy(d, std::span<const std::uint32_t>(g.data() + i0, i1 - i0));
                    dd->append(rows[i0].slot, taps + i0 * ds::taps * 6144, v.accepted + 1, rows[i0].pos, nullptr);
                }
                i0 = i1;
            }
        }
        return g;
    }
};

using ms::lead::Policy;
using ms::lead::Shared;
using ms::lead::Counters;
using ms::lead::Extra;
using ms::lead::Sched;
using ms::lead::Lead;

void follow(mm::Engine& eng, Runner& run, strix::minimax::session::Runtime* session = nullptr) try {
    ms::SessionCounts counts;
    bool taps = run.dd != nullptr && env_num("STRIX_MM_SERVE_DSPARK", 0) != 0;
    std::uint64_t round = run.control_round;
    for (;;) {
        Control mine;
        mine.ack_round = round == 0 ? ~std::uint64_t{0} : round - 1;
        mine.plen = eng.prefill_progress().open ? 0 : run.last_next;
        const Control c = exchange(eng, mine);
        if (c.round != round++) {
            eng.abort("control channel: duplicate / stale round");
            throw std::runtime_error("control channel: duplicate / stale round");
        }
        if (static_cast<Command>(c.command) == Command::session) {
            const auto body = ms::session_wire(eng, {}, c.plen);
            const auto available = eng.peer(session ? 1U : 0U);
            if (session && available) (void)ms::session_execute(eng, *session, body, counts);
            eng.set_taps(taps);
            continue;
        }
        if (static_cast<Command>(c.command) == Command::taps) taps = (c.pflags & 1U) != 0;
        if (!ms::replay_command(c, eng, run, [&](std::size_t n) {
                std::vector<std::uint32_t> prompt;
                exchange_tokens(eng, prompt, n, false);
                return prompt;
            })) return;
    }
} catch (...) {
    eng.abort("control channel: protocol failure");
    throw;
}

// ---- both ranks, before READY: the resumable prefill vs prefill() with decode steps of other slots in between ----
// Each request has a complete, fixed-length continuation (stop IDs are retained). Compare all IDs, committed states,
// prompt output rows and continuation logits. Include idle bounded resumes and two queued prompts in reversed order.
int resume_check(mm::Engine& eng, Runner& run, const std::string& files, unsigned tokens, const std::string& nested_file, unsigned child_layers) try {
    if (eng.slots() < 3 || tokens == 0) throw std::invalid_argument("resume-check: requires at least 3 slots and positive continuation length");
    struct Result {
        std::array<std::vector<std::uint32_t>, 3> ids;
        std::array<std::uint64_t, 3> state{}, logits{}, out{};
        std::array<std::uint32_t, ds::block> draft{};
        bool operator==(const Result&) const = default;
    };
    const auto receipt = [&](const Result& a, const Result& ref) {
        for (unsigned slot = 0; slot < 3; ++slot) {
            if (ref.ids[slot].empty()) continue;
            std::printf("rank %u resume-check slot=%u ids=%zu/%zu ids_fnv=%s/%s state=%s/%s out=%s/%s logits=%s/%s\n",
                        eng.rank(), slot, a.ids[slot].size(), ref.ids[slot].size(),
                        hex(fnv(a.ids[slot].data(), a.ids[slot].size() * 4)).c_str(),
                        hex(fnv(ref.ids[slot].data(), ref.ids[slot].size() * 4)).c_str(),
                        hex(a.state[slot]).c_str(), hex(ref.state[slot]).c_str(),
                        hex(a.out[slot]).c_str(), hex(ref.out[slot]).c_str(),
                        hex(a.logits[slot]).c_str(), hex(ref.logits[slot]).c_str());
        }
    };
    eng.set_prefill_check_outputs(true);
    int bad = 0;
    bool nested_ran = false;
    std::size_t at = 0;
    while (at <= files.size()) {
        const auto comma = files.find(',', at);
        std::string f = files.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        at = comma == std::string::npos ? files.size() + 1 : comma + 1;
        if (f.empty()) continue;
        std::vector<unsigned> ks{1, 7, 60};
        if (const auto colon = f.rfind(':'); colon != std::string::npos) {
            ks.clear();
            for (std::size_t a = colon + 1; a < f.size();) {
                const auto slash = f.find('/', a);
                ks.push_back(static_cast<unsigned>(std::stoul(f.substr(a, slash == std::string::npos ? std::string::npos : slash - a))));
                a = slash == std::string::npos ? f.size() : slash + 1;
            }
            f.resize(colon);
        }
        std::ifstream in(f, std::ios::binary);
        std::vector<std::uint32_t> p;
        for (std::uint32_t t; in.read(reinterpret_cast<char*>(&t), 4);) p.push_back(t);
        if (p.size() < 80 || p.size() + tokens >= eng.capacity())
            throw std::invalid_argument("resume-check: prompt needs >=80 tokens and room for the complete continuation");
        const std::vector<std::uint32_t> q(p.begin(), p.begin() + 64);
        const bool ctx = run.dd != nullptr && p.size() <= 4096;
        auto hash_logits = [&](Result& r, unsigned slot, std::size_t rows, std::size_t row) {
            const auto logits = eng.last_logits(rows);
            const auto width = logits.size() / rows;
            r.logits[slot] = fnv(logits.data() + row * width, width * sizeof(float),
                                  r.logits[slot] == 0 ? 14695981039346656037ULL : r.logits[slot]);
        };
        // Capture before another prefill, decode or draft can replace the prompt output/head.
        const auto completed = [&](Result& r, unsigned slot, std::uint32_t next) {
            r.ids[slot].push_back(next);
            r.out[slot] = eng.prefill_check_output_fnv();
            hash_logits(r, slot, 1, 0);
        };
        const auto arm = [&](unsigned k, bool idle) {
            Result r;
            std::array<std::uint64_t, 3> pos{p.size(), q.size(), q.size()};
            auto backgrounds = [&] {
                for (unsigned slot = 1; slot <= 2; ++slot) {
                    std::uint32_t next = 0;
                    run.prefill_begin(slot, q, false, ++run.local_id);
                    (void)run.prefill_unit(~0U, &next);
                    completed(r, slot, next);
                }
            };
            auto step = [&] {
                const std::array<mm::Row, 2> rows{mm::Row{1, r.ids[1].back(), pos[1]}, mm::Row{2, r.ids[2].back(), pos[2]}};
                const auto g = run.step(rows, 0);
                for (unsigned slot = 1; slot <= 2; ++slot) {
                    ++pos[slot];
                    r.ids[slot].push_back(g[slot - 1]);
                    hash_logits(r, slot, 2, slot - 1);
                }
            };
            if (!idle) backgrounds();
            if (k != 0 && !idle && tokens > 1) step();
            run.prefill_begin(0, p, ctx, ++run.local_id);
            unsigned units = 0;
            for (bool done = false; !done;) {
                unsigned budget = k == 0 ? ~0U : k;
                if (idle && k != 0) {
                    const ms::LayerModel model;
                    budget = ms::resume_layers(std::min(budget, ms::unit_layers(250, false, 1500, 0, model.fixed_ms,
                        model.layer_ms(std::max<std::uint64_t>(1, eng.prefill_unit_rows())))));
                }
                std::uint32_t next = 0;
                done = run.prefill_unit(budget, &next);
                ++units;
                if (done) completed(r, 0, next);
                if (!idle && k != 0 && r.ids[1].size() < tokens) step();
            }
            if (ctx) r.draft = run.draft(0, r.ids[0].back(), p.size()).tokens;
            if (idle) backgrounds();
            while (r.ids[1].size() < tokens) step();
            for (unsigned i = 1; i < tokens; ++i) {
                const mm::Row row{0, r.ids[0].back(), pos[0]++};
                const auto g = run.step(std::span(&row, 1), 0);
                r.ids[0].push_back(g[0]);
                hash_logits(r, 0, 1, 0);
            }
            for (unsigned slot = 0; slot < 3; ++slot) r.state[slot] = eng.state_fnv(slot, pos[slot]);
            if (idle && k != 0 && *eng.prefill_unlimited_reason() == 0 && units < 2) ++bad;
            std::printf("rank %u resume-check prompt_tokens=%zu k=%u idle=%u units=%u fallback=%s\n",
                        eng.rank(), p.size(), k, idle ? 1U : 0U, units, eng.prefill_unlimited_reason());
            return r;
        };
        const Result ref = arm(0, false);
        for (unsigned k : ks) {
            if (k == 0) throw std::invalid_argument("resume-check: k >= 1");
            for (bool idle : {false, true}) {
                const auto a = arm(k, idle);
                const bool ok = a == ref;
                receipt(a, ref);
                bad += ok ? 0 : 1;
                std::printf("rank %u resume-check k=%u idle=%u continuation=%u all IDs/state/prompt outputs/first+continuation logits/draft: %s\n",
                            eng.rank(), k, idle ? 1U : 0U, tokens, ok ? "IDENTICAL" : "MISMATCH");
            }
        }
        // Two waiting prompts submitted together. Rank 0 selects; the follower receives only the selected slot.
        const auto reorder = [&](bool short_first) {
            Result r;
            sv::GenerationQueue queue;
            if (eng.rank() == 0) {
                sv::GenerationJob long_job, short_job;
                long_job.prompt = p; short_job.prompt = q;
                (void)queue.submit(std::move(long_job)); (void)queue.submit(std::move(short_job));
            }
            for (unsigned i = 0; i < 2; ++i) {
                std::optional<sv::TakenJob> job;
                unsigned slot = 0;
                if (eng.rank() == 0) {
                    job = queue.take(false, short_first ? sv::QueueSelection::short_first : sv::QueueSelection::fifo);
                    slot = job->job.prompt.size() == p.size() ? 0U : 1U;
                }
                slot = eng.agree(slot);
                if (slot != (short_first ? 1 - i : i)) ++bad;
                const auto& prompt = slot == 0 ? p : q;
                run.prefill_begin(slot, prompt, false, ++run.local_id);
                std::uint32_t next = 0;
                (void)run.prefill_unit(~0U, &next);
                completed(r, slot, next);
                std::uint64_t pos = prompt.size();
                for (unsigned j = 1; j < tokens; ++j) {
                    const mm::Row row{slot, r.ids[slot].back(), pos++};
                    r.ids[slot].push_back(run.step(std::span(&row, 1), 0)[0]);
                    hash_logits(r, slot, 1, 0);
                }
                r.state[slot] = eng.state_fnv(slot, pos);
                if (job) queue.finish(*job, {});
            }
            return r;
        };
        const auto fifo = reorder(false);
        const auto shortest = reorder(true);
        receipt(shortest, fifo);
        const bool ordered = shortest == fifo;
        bad += ordered ? 0 : 1;
        std::printf("rank %u resume-check short-first complete continuations/state/prompt outputs/first+continuation logits: %s\n", eng.rank(), ordered ? "IDENTICAL" : "MISMATCH");
        if (nested_file.empty()) continue;
        nested_ran = true;
        // Explicit nested qualification never truncates the original fixture.
        std::ifstream short_in(nested_file, std::ios::binary);
        if (!short_in) throw std::invalid_argument("resume-check nested: cannot open fixture");
        const auto short_prompt = ms::nested_fixture(short_in, eng.nested_rows(), eng.capacity(), tokens);
        if (*eng.prefill_unlimited_reason() != 0)
            throw std::runtime_error("resume-check nested: requires bounded path (NOT a pass)");
        const std::array<std::vector<std::uint32_t>, 3> prompts{p, short_prompt, q};
        const auto nested_arm = [&](unsigned checkpoint, bool background_step) {
            Result r;
            std::array<std::uint64_t, 3> pos{p.size(), short_prompt.size(), q.size()};
            auto& check_round = run.control_round;
            const auto command = [&](Command op, std::uint64_t id, std::uint64_t parent = 0,
                                     unsigned slot = 0, unsigned layers = 0) {
                Control out;
                out.ack_round = check_round == 0 ? ~std::uint64_t{0} : check_round - 1;
                if (eng.rank() == 0) {
                    out.command = static_cast<std::uint32_t>(op);
                    out.round = check_round;
                    out.task_id = id; out.parent_task_id = parent; out.slot_generation = 1;
                    out.pslot = slot; out.layers = layers;
                    out.bank = op == Command::nest ? 1U : eng.prefill_progress().bank;
                    if (op == Command::check) out.plen = run.last_next;
                    if (op == Command::prefill || op == Command::nest) out.plen = static_cast<std::uint32_t>(prompts[slot].size());
                }
                const auto peer = exchange(eng, out);
                const auto& c = eng.rank() == 0 ? out : peer;
                if (c.round != check_round++) {
                    eng.abort("nested check replay round");
                    throw std::runtime_error("nested check replay round");
                }
                (void)ms::replay_command(c, eng, run, [&](std::size_t n) {
                    auto prompt = eng.rank() == 0 ? prompts[slot] : std::vector<std::uint32_t>{};
                    exchange_tokens(eng, prompt, n, eng.rank() == 0);
                    return prompt;
                });
            };
            const auto unit = [&](std::uint64_t id, unsigned layers, std::uint32_t* next) {
                command(Command::unit, id, 0, 0, layers);
                *next = run.last_next;
                return !eng.prefill_active();
            };
            const auto begin = [&](unsigned slot, std::uint64_t id, std::uint64_t parent) {
                command(parent == 0 ? Command::prefill : Command::nest, id, parent, slot);
            };
            const auto checked_completion = [&](unsigned slot, std::uint32_t next) {
                command(Command::check, 100 + slot);
                completed(r, slot, next);
            };
            const auto advance = [&](unsigned slot) {
                const mm::Row row{slot, r.ids[slot].back(), pos[slot]++};
                r.ids[slot].push_back(run.step(std::span(&row, 1), 0)[0]);
                hash_logits(r, slot, 1, 0);
            };
            const auto complete = [&](unsigned slot) {
                while (r.ids[slot].size() < tokens) advance(slot);
                r.state[slot] = eng.state_fnv(slot, pos[slot]);
            };
            std::uint32_t next = 0;
            if (checkpoint == 0) {
                for (unsigned slot = 0; slot < 3; ++slot) {
                    begin(slot, 100 + slot, 0);
                    (void)unit(100 + slot, ~0U, &next);
                    checked_completion(slot, next);
                    complete(slot);
                }
            } else {
                begin(2, 102, 0);
                (void)unit(102, ~0U, &next);
                checked_completion(2, next);
                begin(0, 100, 0);
                if (unit(100, checkpoint, &next)) throw std::runtime_error("nested check did not park an incomplete long");
                const auto before = eng.prefill_progress();
                const auto live = eng.prefill_activation_fnv(100);
                if (background_step && tokens > 1) advance(2); // overwrite the shared eight-row save's source
                command(Command::park, 100);
                const auto frozen = eng.prefill_activation_fnv(100);
                if (live != frozen) ++bad;
                begin(1, 101, 100);
                unsigned child_units = 0;
                while (!unit(101, child_layers, &next)) {
                    ++child_units;
                    if (background_step && r.ids[2].size() < tokens) advance(2);
                    if (eng.prefill_activation_fnv(100) != frozen) ++bad;
                }
                if (child_layers < 60 && child_units == 0) ++bad;
                checked_completion(1, next); // consume child output/head before rebinding
                if (eng.prefill_activation_fnv(100) != frozen) ++bad;
                command(Command::resume_parked, 100);
                if (eng.prefill_progress() != before) throw std::runtime_error("nested checkpoint changed");
                (void)unit(100, ~0U, &next);
                checked_completion(0, next);
                for (;;) {
                    std::vector<mm::Row> rows;
                    for (unsigned slot = 0; slot < 3; ++slot)
                        if (r.ids[slot].size() < tokens) rows.push_back({slot, r.ids[slot].back(), pos[slot]++});
                    if (rows.empty()) break;
                    const auto g = run.step(rows, 0);
                    for (std::size_t i = 0; i < rows.size(); ++i) {
                        r.ids[rows[i].slot].push_back(g[i]);
                        hash_logits(r, rows[i].slot, rows.size(), i);
                    }
                }
                for (unsigned slot = 0; slot < 3; ++slot) complete(slot);
            }
            return r;
        };
        if (std::none_of(ks.begin(), ks.end(), [](unsigned k) { return k > 0 && k < 60; }))
            throw std::invalid_argument("resume-check nested: supply at least one park layer in 1..59");
        const auto standalone = nested_arm(0, false);
        for (unsigned checkpoint : ks) {
            if (checkpoint >= 60) continue; // existing k=60 is a unit size, not a valid first-group park layer
            for (bool background_step : {false, true}) {
                const auto nested = nested_arm(checkpoint, background_step);
                receipt(nested, standalone);
                const bool identical = nested == standalone;
                bad += identical ? 0 : 1;
                std::printf("rank %u resume-check nested parent_rows=%zu short_rows=%zu park_layer=%u background_step=%u standalone B1 IDs/state/output/first+continuation logits: %s\n",
                    eng.rank(), p.size(), short_prompt.size(), checkpoint, background_step ? 1U : 0U, identical ? "IDENTICAL" : "MISMATCH");
            }
        }
        std::fflush(stdout);
    }
    if (!nested_file.empty() && !nested_ran) throw std::invalid_argument("resume-check nested: no parent fixtures");
    eng.set_prefill_check_outputs(false);
    return bad + static_cast<int>(eng.peer(static_cast<std::uint32_t>(bad)));
} catch (...) {
    eng.abort("control channel: protocol failure");
    throw;
}

// ---- rank 0: the HTTP API's scheduler side (ms::MinimaxApi serves the routes) ----
sv::Json policy_json(Shared& shared, bool drafter) {
    std::lock_guard<std::mutex> l(shared.m);
    const Policy& p = shared.policy;
    const Counters& c = shared.counters;
    sv::Json b = sv::Json::object();
    b.set("sched", sv::Json(p.sched == Sched::interleave ? "interleave" : "off"));
    b.set("effective_sched", sv::Json(p.sched == Sched::off ? "off" : shared.unlimited_reason.empty() ? "interleave" : "unlimited"));
    b.set("nest", sv::Json(p.nest));
    b.set("effective_nest", sv::Json(p.nest && p.sched == Sched::interleave && shared.nest_rows > 0 && shared.unlimited_reason.empty()));
    b.set("nest_reason", sv::Json(!p.nest ? "disabled" : p.sched == Sched::off ? "sched_off" :
        !shared.unlimited_reason.empty() ? shared.unlimited_reason : shared.nest_reason));
    b.set("nest_rows", sv::Json(static_cast<std::int64_t>(shared.nest_rows)));
    b.set("nest_poll_ms", sv::Json(std::int64_t{0}));  // Phase A gap-budget units; no extra arrival cap
    b.set("nest_max_per_parent", sv::Json(std::int64_t{2}));
    b.set("nest_bank_bytes", sv::Json(static_cast<std::int64_t>((shared.nest_rows == 0 ? 0U : std::max(8U, shared.nest_rows)) * 6144ULL * 8)));
    b.set("nest_tap_bytes", sv::Json(static_cast<std::int64_t>(shared.nest_taps && shared.nest_rows != 0 ? std::max(8U, shared.nest_rows) * 6144ULL * 20 : 0)));
    b.set("active_task", sv::Json(static_cast<std::int64_t>(shared.active_id)));
    b.set("parked_task", sv::Json(static_cast<std::int64_t>(shared.parked_id)));
    b.set("nest_qualification", sv::Json("pending extended resume-check"));
    b.set("service_ms", sv::Json(p.service_ms));
    b.set("short_first", sv::Json(p.sched == Sched::interleave && p.short_first));
    b.set("fallback_reason", sv::Json(p.sched == Sched::off ? "sched_off" : shared.unlimited_reason));
    b.set("max_unit_layers", sv::Json(static_cast<std::int64_t>(c.max_unit_layers)));
    b.set("qualification", sv::Json("pending"));
    b.set("gap_ms", sv::Json(p.gap_ms));
    b.set("dspark", sv::Json(drafter && (p.dspark_mode == ds::Mode::on || shared.dspark_contexts != 0)));
    b.set("dspark_mode", sv::Json(ds::mode_name(p.dspark_mode)));
    b.set("lookup", sv::Json(p.lookup));
    b.set("dspark_effective", sv::Json(p.dspark_mode == ds::Mode::on && drafter ? "on" : p.dspark_mode == ds::Mode::automatic && p.dspark_prior == ds::Prior::structural_v1 && drafter ? "per_request" : "plain"));
    b.set("dspark_assets", sv::Json(drafter));
    b.set("dspark_live", sv::Json(static_cast<std::int64_t>(shared.dspark_live)));
    b.set("dspark_contexts", sv::Json(static_cast<std::int64_t>(shared.dspark_contexts)));
    b.set("dspark_concurrency_eligible", sv::Json(shared.dspark_live == 1 && shared.active_id == 0 && shared.parked_id == 0));
    b.set("dspark_prior", sv::Json(ds::prior_name(p.dspark_prior)));
    b.set("dspark_reason", sv::Json(p.dspark_mode == ds::Mode::off ? "plain:off" : !drafter ? "plain:assets_missing" :
                                     p.dspark_mode == ds::Mode::automatic ? (p.dspark_prior == ds::Prior::unqualified_v1 ? "plain:prior_unqualified" : "auto:per_request") : "legacy"));
    b.set("dspark_inflight", sv::Json(static_cast<std::int64_t>(p.inflight)));
    b.set("draft_slots", sv::Json(static_cast<std::int64_t>(p.draft_slots)));
    b.set("epoch", sv::Json(static_cast<std::int64_t>(p.epoch)));
    sv::Json k = sv::Json::object();
    k.set("step_s", sv::Json(c.step_s));
    k.set("prefill_s", sv::Json(c.prefill_s));
    k.set("draft_s", sv::Json(c.draft_s));
    k.set("idle_s", sv::Json(c.idle_s));
    k.set("stall_s", sv::Json(c.stall_s));
    k.set("steps", sv::Json(static_cast<std::int64_t>(c.steps)));
    k.set("step_rows", sv::Json(static_cast<std::int64_t>(c.step_rows)));
    k.set("units", sv::Json(static_cast<std::int64_t>(c.units)));
    k.set("prefills", sv::Json(static_cast<std::int64_t>(c.prefills)));
    k.set("cancels", sv::Json(static_cast<std::int64_t>(c.cancels)));
    k.set("multi_windows", sv::Json(static_cast<std::int64_t>(c.multi_windows)));
    k.set("unequal_windows", sv::Json(static_cast<std::int64_t>(c.unequal_windows)));
    k.set("rejections", sv::Json(static_cast<std::int64_t>(c.rejections)));
    k.set("draft_cycles", sv::Json(static_cast<std::int64_t>(c.draft_cycles)));
    k.set("plain_cycles", sv::Json(static_cast<std::int64_t>(c.plain_cycles)));
    k.set("drafts", sv::Json(static_cast<std::int64_t>(c.drafts)));
    k.set("drafted", sv::Json(static_cast<std::int64_t>(c.drafted)));
    k.set("accepted", sv::Json(static_cast<std::int64_t>(c.accepted)));
    k.set("lookup_windows", sv::Json(static_cast<std::int64_t>(c.lookup_windows)));
    k.set("lookup_drafted", sv::Json(static_cast<std::int64_t>(c.lookup_drafted)));
    k.set("lookup_accepted", sv::Json(static_cast<std::int64_t>(c.lookup_accepted)));
    k.set("lookup_s", sv::Json(c.lookup_s));
    sv::Json h = sv::Json::array();
    for (auto v : c.rows_hist) h.as_array().push_back(sv::Json(static_cast<std::int64_t>(v)));
    k.set("rows_hist", std::move(h));
    b.set("counters", std::move(k));
    return b;
}

// The mm_* value lines of GET /metrics: the queue's (from the snapshot the observer renders too) and the scheduler's
// counters (the observer adds their HELP/TYPE).
std::string mm_metric_lines(const sv::QueueSnapshot& s, Shared& shared) {
    Counters c;
    {
        std::lock_guard<std::mutex> l(shared.m);
        c = shared.counters;
    }
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "mm_queued %llu\nmm_running %llu\nmm_completed %llu\nmm_failed %llu\nmm_prompt_tokens %llu\n"
                  "mm_generated_tokens %llu\nmm_busy_seconds %.3f\nmm_step_seconds %.3f\nmm_prefill_seconds %.3f\n"
                  "mm_draft_seconds %.3f\nmm_idle_seconds %.3f\nmm_decode_stall_seconds %.3f\nmm_steps %llu\nmm_step_rows %llu\n"
                  "mm_prefill_units %llu\nmm_draft_cycles %llu\nmm_drafted %llu\nmm_accepted %llu\n",
                  static_cast<unsigned long long>(s.queued), static_cast<unsigned long long>(s.running),
                  static_cast<unsigned long long>(s.completed), static_cast<unsigned long long>(s.failed),
                  static_cast<unsigned long long>(s.prompt_tokens), static_cast<unsigned long long>(s.generated_tokens),
                  s.busy_seconds, c.step_s, c.prefill_s, c.draft_s, c.idle_s, c.stall_s, static_cast<unsigned long long>(c.steps),
                  static_cast<unsigned long long>(c.step_rows), static_cast<unsigned long long>(c.units),
                  static_cast<unsigned long long>(c.draft_cycles), static_cast<unsigned long long>(c.drafted),
                  static_cast<unsigned long long>(c.accepted));
    return buf;
}

// GET / POST /admin/sched (with --admin): the policy and counters; POST changes them (a bad value throws: a 400).
sv::Json admin_sched(const sv::HttpRequest& req, Shared& shared, bool drafter) {
    if (req.method == "POST") {
        const sv::Json body = sv::parse_request_json(req.body);
        std::lock_guard<std::mutex> l(shared.m);
        Policy p = shared.policy;
        if (const auto* v = body.find("sched"); v && v->is_string()) {
            if (v->as_string() == "off") p.sched = Sched::off;
            else if (v->as_string() == "interleave") p.sched = Sched::interleave;
            else throw std::invalid_argument("sched: off | interleave");
        }
        if (const auto* v = body.find("gap_ms"); v && v->is_number()) p.gap_ms = ms::gap_ms(v->dump().c_str());
        if (const auto* v = body.find("service_ms"); v && v->is_number()) p.service_ms = ms::service_ms(v->dump().c_str());
        if (const auto* v = body.find("nest")) {
            if (!v->is_bool()) throw std::invalid_argument("nest: boolean required");
            p.nest = v->as_bool();
        }
        if (const auto* v = body.find("short_first"); v && v->is_bool()) p.short_first = v->as_bool();
        p.dspark_mode = ms::dspark_mode_update(p.dspark_mode, body);
        if (const auto* v = body.find("dspark_inflight")) {
            if (!v->is_number()) throw std::invalid_argument("dspark_inflight: 0..8");
            p.inflight = ds::parse_inflight(v->dump().c_str());
        }
        if (const auto* v = body.find("lookup")) {
            if (!v->is_bool()) throw std::invalid_argument("lookup: boolean required");
            p.lookup = v->as_bool();
        }
        if (const auto* v = body.find("draft_slots"); v && v->is_number()) p.draft_slots = static_cast<std::uint32_t>(std::clamp<std::int64_t>(v->as_int(), 1, 8));
        if (body.find("reset_counters")) shared.counters = Counters{};
        ++p.epoch;
        shared.policy = p;
    }
    return policy_json(shared, drafter);
}

}  // namespace

int main(int argc, char** argv) {
    // Defaults-on resolver and validation precede any device work, including inspection-only calls.
    try {
        if (strix::minimax::defaults::print_or_apply("serve", argc, argv)) return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "minimax serve: %s\n", e.what());
        return 2;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR --rank R --transport T [--port P] [--bind ADDR] [--slots S] [--capacity C] "
                             "[--max-rows R] [--model-id ID] [--max-tokens N] [--dspark-mode off|auto|on] [--dspark DIR [--dspark-exl3 DIR]] [--admin] "
                             "[--resume-check P.u32[,...]]\n", argv[0]);
        return 2;
    }
    try {
        (void)::setenv("STRIX_DIST_USB4_TIMEOUT_MS", "60000", 0);
        (void)::setenv("STRIX_DIST_WAIT_TIMEOUT_MS", "60000", 0);
        // DS production host-path settings (CRC on parallel workers, CRC overlap, 1 MiB host chunks): transport
        // only, bits unchanged; measurement ABA x3 TTFT 1K -12.8 %, decode -1 %. An explicit value wins (=0 disables each).
        (void)::setenv("STRIX_DIST_CRC_PARALLEL", "1", 0);
        (void)::setenv("STRIX_DIST_HOST_CRC_OVERLAP", "1", 0);
        (void)::setenv("STRIX_DIST_HOST_CHUNK_BYTES", "1048576", 0);
        // deferred CRC ON by default (measurement: 7 processes inline/deferred alternated, 384-step 1K
        // sessions, tokens identical, each deferred arm below its two inline neighbours by 0.04-0.12 ms/token).
        // STRIX_MMX_CRC: 0 = off, 1 = inline, deferred = deferred check.
        (void)::setenv("STRIX_MMX_CRC", "deferred", 0);
        strix::minimax::session::RunOptions session_options;
        mm::Options opt;
        opt.model_dir = argv[1];
        opt.slots = 8;
        std::string transport_spec = "usb4-auto", bind = "127.0.0.1", model_id = "minimax-m3.1-flash", dspark_dir, dspark_exl3, check, check_nested;
        std::uint16_t port = 8080;
        std::uint32_t max_tokens = 4096, check_tokens = 256, check_child_layers = ~0U;
        bool admin = false, check_only = false;
        std::string dspark_mode_value, dspark_prior_value;
        const auto dspark_inflight = ds::parse_inflight(std::getenv("STRIX_MM_SERVE_DSPARK_INFLIGHT"));
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if (a == "--rank") opt.rank = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--transport") transport_spec = next();
            else if (a == "--port") port = static_cast<std::uint16_t>(std::stoul(next()));
            else if (a == "--bind") bind = next();
            else if (a == "--slots") opt.slots = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--capacity") opt.capacity = std::stoull(next());
            else if (a == "--max-rows") opt.max_rows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--model-id") model_id = next();
            else if (a == "--max-tokens") max_tokens = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--dspark") dspark_dir = next();
            else if (a == "--dspark-prior") dspark_prior_value = next();
            else if (a == "--dspark-mode") dspark_mode_value = next();
            else if (a == "--dspark-exl3") dspark_exl3 = next();
            else if (a == "--admin") admin = true;
            else if (a == "--resume-check-child-layers") check_child_layers = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--resume-check-nested") {
                check_nested = next();
                if (check_nested.empty()) throw std::invalid_argument("--resume-check-nested requires a fixture");
            }
            else if (a == "--resume-check") check = next();
            else if (a == "--resume-check-tokens") check_tokens = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--check-only") check_only = true;
            else if (strix::minimax::session::parse_option(session_options,a,next)) {}
            else throw std::invalid_argument("unknown option " + a);
        }
        const auto dspark_prior = ds::parse_prior(dspark_prior_value.empty() ? std::getenv("STRIX_MM_DSPARK_PRIOR") : dspark_prior_value.c_str());
        const auto dspark_mode = ds::parse_mode(dspark_mode_value.empty() ? std::getenv("STRIX_MM_SERVE_DSPARK") : dspark_mode_value.c_str());
        if (check_child_layers == 0 || (check_child_layers != ~0U && check_child_layers >= 60))
            throw std::invalid_argument("--resume-check-child-layers: 1..59");
        if ((!check_nested.empty() || check_child_layers != ~0U) && check.empty())
            throw std::invalid_argument("nested qualification requires --resume-check");
        if (check_child_layers != ~0U && check_nested.empty())
            throw std::invalid_argument("bounded child requires --resume-check-nested");
        if (check_child_layers != ~0U && check_tokens < 2 + 59 / check_child_layers)
            throw std::invalid_argument("bounded child requires enough continuation tokens for every background step");
        if(!session_options.save.empty() && !session_options.load.empty()) throw std::invalid_argument("choose session save or load");
        if (opt.slots == 0 || opt.slots > 8) throw std::invalid_argument("--slots 1..8");
        if (!dspark_exl3.empty() && dspark_dir.empty()) dspark_dir = opt.model_dir + "/dspark";
        if (dspark_dir.empty() && dspark_mode != ds::Mode::off && std::filesystem::is_directory(opt.model_dir + "/dspark"))
            for (const auto& entry : std::filesystem::directory_iterator(opt.model_dir + "/dspark"))
                if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                    dspark_dir = opt.model_dir + "/dspark";
                    break;
                }
        const strix::minimax::defaults::Lookup lookup = [](const char* key) { return std::getenv(key); };
        strix::minimax::defaults::Cli cli; cli.app = "serve"; cli.transport = transport_spec; cli.slots = opt.slots; cli.capacity = opt.capacity; cli.max_rows = opt.max_rows;
        strix::minimax::defaults::SessionConfig session_config;
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        std::string receipt;
        auto transport = open_transport(transport_spec, opt.rank, receipt);
        strix::dist::Group group;
        group.minimax_compat = true;
        group.rank = opt.rank;
        group.world = 2;
        group.plan = 0x4d4d5331ULL;  // "MMS1"
        const std::size_t max_bytes = std::max<std::size_t>(std::size_t{opt.max_rows} * 6144, 256 * 782 * 128) * sizeof(float);
        // exchange module for the decode steps (default on, STRIX_MM_XFER=off disables)
        const char* xfer_env = std::getenv("STRIX_MM_XFER");
        std::unique_ptr<strix::minimax::xfer::Exchanger> exchanger;
        if (xfer_env == nullptr || std::string(xfer_env) != "off") {
            strix::minimax::xfer::Options xo;
            xo.rank = opt.rank;
            xo.capacity = strix::minimax::xfer::MmSchedule::row_bytes(8);
            xo.stream = nullptr;
            xo.from_env();
            exchanger = std::make_unique<strix::minimax::xfer::Exchanger>(std::make_unique<Borrowed>(*transport), xo);
        }
        strix::dist::Communicator comm(group, std::make_unique<Borrowed>(*transport), max_bytes);
        std::printf("connected: rank %u transport %s\n", opt.rank, comm.transport_name().c_str());
        std::fflush(stdout);
        mm::Engine eng(opt, comm);
        if (exchanger) eng.set_exchanger(exchanger.get(), true);
        const auto lr = eng.load();
        const std::uint32_t dspark_config = 0x4d4d0400U | (static_cast<std::uint32_t>(dspark_prior) << 7) | (static_cast<std::uint32_t>(dspark_mode) << 5) |
                                            (dspark_dir.empty() ? 0U : 16U) | dspark_inflight;
        if (eng.peer(dspark_config) != dspark_config) {
            eng.abort("DSpark startup mode / assets / protocol mismatch");
            throw std::runtime_error("DSpark startup mode / assets / protocol mismatch");
        }
        std::printf("rank %u: load %.1f s, device used %.1f GiB\n", opt.rank, lr.seconds, lr.device_gib);
        std::fflush(stdout);
        // DSpark drafter on both ranks (both run every draft command; the target's residual taps are replicated)
        std::unique_ptr<ds::DraftFile> draft_file;
        std::unique_ptr<ds::DeviceDrafter> drafter;
        Runner run{eng};
        if (!dspark_dir.empty()) {
            const auto t0 = Clock::now();
            draft_file = ds::DraftFile::open(dspark_dir, dspark_exl3.empty());
            ds::DeviceDrafter::Options dopt;
            dopt.slots = opt.slots;
            dopt.capacity = opt.capacity;
            if (const char* e = std::getenv("STRIX_MM_DSPARK_ATTN")) dopt.attn_variant = static_cast<std::uint32_t>(std::atoi(e));  // as the rank app
            drafter = std::make_unique<ds::DeviceDrafter>(draft_file->weights(), dopt);
            if (!dspark_exl3.empty()) eng.attach_draft_exl3(dspark_exl3, *drafter);
            eng.set_taps(true);  // for the drafter-context gate; then the startup policy below (both ranks)
            run.dd = drafter.get();
            std::tie(run.v0, run.vn) = eng.vocab_range();
            std::printf("rank %u: dspark %s%s loaded in %.1f s (%.2f GB device, rings %.0f MB, %u slots)\n", opt.rank, dspark_dir.c_str(),
                        dspark_exl3.empty() ? "" : (" + EXL3 " + dspark_exl3).c_str(), secs(t0, Clock::now()),
                        static_cast<double>(drafter->weight_bytes()) / 1e9, static_cast<double>(drafter->ring_bytes()) / 1e6, opt.slots);
            std::fflush(stdout);
        }
        (void)eng.reserve_nested(drafter != nullptr);
        // A protocol/capacity handshake before any check or READY, independent of rank-1 environment policy.
        const auto peer_capacity = eng.peer(eng.nested_rows());
        if (peer_capacity != eng.nested_rows()) {
            eng.abort("nested capacity mismatch");
            throw std::runtime_error("nested capacity mismatch");
        }
        std::unique_ptr<strix::minimax::session::Runtime> session_runtime;
        const bool session_cli = !session_options.prompt.empty() || !session_options.load.empty();
        try { session_config = strix::minimax::defaults::session_policy(cli, lookup, session_cli); }
        catch (const std::exception& e) { session_config.enabled = false; session_config.reason = e.what(); }
        const char* session_switch = std::getenv("STRIX_SESSION_CACHE");
        const bool session_explicit = session_cli || (session_switch &&
            (std::string(session_switch) == "1" || std::string(session_switch) == "on"));
        const bool session_vote = ms::session_hello(eng, session_config.reason != "baseline" &&
            (session_config.enabled || session_explicit));
        if (session_vote) {
            try {
                if (session_cli) {
                    session_config = strix::minimax::defaults::session_config(cli, lookup, UINT64_MAX, true,
                        !session_options.save.empty() ? session_options.save : !session_options.load.empty() ? session_options.load : session_options.directory);
                    if ((!session_options.save.empty() || !session_options.load.empty()) && !session_config.disk) {
                        session_config.enabled = false; session_config.reason = session_config.disk_reason;
                    }
                } else session_config = strix::minimax::defaults::session_config(cli, lookup);
            } catch (const std::exception& e) { session_config.enabled = false; session_config.reason = e.what(); }
            if (!session_options.budget) session_options.budget = session_config.ram_bytes;
            if (session_cli || session_options.directory.empty()) session_options.directory = session_config.directory;
            session_options.max_parked = session_config.max_parked;
            session_options.disk_budget = session_config.disk_bytes;
            session_options.disk_enabled = session_config.disk;
            const bool session_on = session_config.enabled;

            if (!session_cli) session_options.draft = drafter != nullptr;
            session_runtime = strix::minimax::session::admit_runtime_pair(
                {session_on, session_explicit, session_config.reason}, eng.rank(),
                [&](const strix::minimax::session::StartupVote& vote) { return strix::minimax::session::exchange_startup(eng, vote); }, [&] {
                    std::fprintf(stderr, "session cache admission: ram_bytes=%llu disk_bytes=%llu disk=%s\n",
                        static_cast<unsigned long long>(session_options.budget), static_cast<unsigned long long>(session_options.disk_budget),
                        session_options.disk_enabled ? "on" : "off");
                    return std::make_unique<strix::minimax::session::Runtime>(eng,drafter.get(),opt,"/proc/self/exe",session_options,dspark_dir,dspark_exl3,draft_file ? draft_file->loader_snapshot() : strix::LoaderSnapshot{});
                }, [](const std::string& line) {
                    std::fprintf(stderr, "%s\n", line.c_str());
                });
        } else {
            std::fprintf(stderr, "session cache: off (%s)\n", session_config.reason.c_str());
        }
        if (session_runtime && session_cli) {
            strix::minimax::session::run_cli(*session_runtime,session_options); return 0;
        }

        if (!check.empty()) {
            const int bad = resume_check(eng, run, check, check_tokens, check_nested, check_child_layers);
            std::printf("rank %u resume-check: %s\n", opt.rank, bad == 0 ? "ALL IDENTICAL" : "MISMATCH");
            std::fflush(stdout);
            if (bad != 0) return 4;
            if (check_only) {  // structural gates only (e.g. a small --max-rows: groups then a single chunk), no serving
                std::printf("rank %u: check only, STOPPED\n", opt.rank);
                return 0;
            }
        }
        // taps only while DSpark is on (both ranks read the same STRIX_MM_SERVE_DSPARK; runtime switches via Command::taps)
        const bool dspark_on = drafter != nullptr && dspark_mode == ds::Mode::on;
        if (drafter != nullptr) eng.set_taps(dspark_on);
        if (opt.rank != 0) {
            follow(eng, run, session_runtime.get());
            std::printf("rank 1: STOPPED\n");
            return 0;
        }
        Shared shared;
        {
            shared.policy.nest = env_num("STRIX_MM_SERVE_NEST", 1) != 0;
            shared.nest_rows = eng.nested_rows();
            shared.nest_reason = eng.nested_reason();
            shared.nest_taps = drafter != nullptr;
            shared.policy.sched = ms::sched_mode(std::getenv("STRIX_MM_SERVE_SCHED"));
            shared.policy.service_ms = ms::service_ms(std::getenv("STRIX_MM_SERVE_SERVICE_MS"));
            shared.policy.short_first = env_num("STRIX_MM_SERVE_SHORT_FIRST", 1) != 0;
            shared.unlimited_reason = eng.prefill_unlimited_reason();
            shared.policy.gap_ms = ms::gap_ms(std::getenv("STRIX_MM_SERVE_GAP_MS"));
            shared.policy.dspark_mode = dspark_mode;
            const char* lookup_env = std::getenv("STRIX_MM_LOOKUP");
            shared.policy.lookup = lookup_env == nullptr || std::strcmp(lookup_env, "0") != 0;
            shared.policy.dspark_prior = dspark_prior;
            shared.policy.inflight = dspark_inflight;
            shared.policy.draft_slots = static_cast<std::uint32_t>(std::clamp(env_num("STRIX_MM_SERVE_DRAFT_SLOTS", 8), 1.0, 8.0));
        }
        sv::GenerationQueue queue(64);
        ms::SessionService session_service;
        ms::SessionCounts session_counts;
        auto refresh_session_metadata = [&] {
            try {
                session_service.tasks.publish_metadata(ms::session_metadata(*session_runtime, session_counts, true),
                                                       ms::session_metadata(*session_runtime, session_counts, false));
            } catch (const std::exception& e) {
                auto error = sv::session_error("unavailable", std::string(e.what()) + "; retry after 1 second", 503);
                session_service.tasks.publish_metadata(error, error);
            }
        };
        if (session_runtime) {
            refresh_session_metadata();
            session_service.execute = [&](const std::string& body, const ms::SessionService::Publish& publish) {
                auto reply = ms::session_execute(eng, *session_runtime, body, session_counts, publish);
                refresh_session_metadata();
                return reply;
            };
        }
        auto& session_tasks = session_service.tasks;
        ms::ObserverInfo oi;
        oi.model = model_id;
        oi.context_tokens = opt.capacity;
        oi.slots = opt.slots;
        oi.transport = comm.transport_name();
        oi.load_seconds = lr.seconds;
        oi.device_used_bytes = static_cast<std::uint64_t>(std::max(0.0, lr.device_gib) * 1073741824.0);
        oi.draft = drafter != nullptr;
        ms::Observer observer(std::move(oi));
        ms::MinimaxApiOptions ao;
        ao.session_enabled = session_runtime != nullptr;
        ao.model_id = model_id;
        ao.default_max_tokens = max_tokens;
        ao.context_tokens = opt.capacity;
        // the scheduler's side: its per-request extras (inserted by Lead::finish before the ticket finishes), its mm_*
        // lines and, with --admin, /admin/sched
        ao.take_extra = [&shared](const sv::GenerationTicket& t) -> std::optional<Extra> {
            std::lock_guard<std::mutex> l(shared.m);
            const auto it = shared.extras.find(&t);
            if (it == shared.extras.end()) return std::nullopt;
            Extra x = it->second;
            shared.extras.erase(it);
            return x;
        };
        ao.legacy_metrics = [&shared](const sv::QueueSnapshot& q) { return mm_metric_lines(q, shared); };
        if (admin) {
            const bool has_drafter = drafter != nullptr;
            ao.admin_sched = [&shared, has_drafter](const sv::HttpRequest& r) { return admin_sched(r, shared, has_drafter); };
        }
        const auto tokenizer = ms::MinimaxTokenizer::load(opt.model_dir + "/tokenizer.json");
        ms::MinimaxApi api(ms::api_tokenizer(tokenizer), queue, ao, observer);
        sv::HttpServerConfig hc;
        hc.bind_address = bind;
        hc.port = port;
        sv::HttpServer http(hc, [&](const sv::HttpRequest& r, sv::HttpResponseWriter& w) {
            if (session_runtime && (r.path.starts_with("/v1/sessions") || r.path == "/v1/cache/stats")) {
                try {
                    const auto command = sv::session_command(r.method,r.path,r.body);
                    auto reply = session_tasks.submit(command);
                    int status = 200;
                    if (const auto* v = reply.find("_status")) status = static_cast<int>(v->as_int());
                    if (const auto* error = reply.find("error")) {
                        if (status == 200 || status == 503) w.add_response_header("Retry-After", "1");
                        w.send(status == 200 ? 503 : status,"application/json",sv::session_error(error->find("code")->as_string(),error->find("message")->as_string()).dump());
                    } else w.send(status,"application/json",reply.dump());
                } catch (const sv::SessionApiError& e) {
                    if (e.status == 503) w.add_response_header("Retry-After", "1");
                    w.send(e.status,"application/json",sv::session_error(e.code,e.what()).dump()); }
                catch (const std::exception& e) { w.send(400,"application/json",sv::session_error("invalid_request",e.what()).dump()); }
                return;
            }
            api.handle(r,w);
        });
        http.listen();
        std::thread http_thread([&] { http.serve(); });
        std::printf("minimax serve policy: %s\n", policy_json(shared, drafter != nullptr).dump().c_str());
        std::printf("READY rank 0 http://%s:%u model %s slots %u context %llu sched %s gap %.0f ms dspark %s\n", bind.c_str(), http.port(),
                    model_id.c_str(), opt.slots, static_cast<unsigned long long>(opt.capacity),
                    shared.policy.sched == Sched::interleave ? "interleave" : "off", shared.policy.gap_ms, ds::mode_name(shared.policy.dspark_mode));
        std::fflush(stdout);
        Lead lead(eng, run, queue, shared, opt.max_rows, dspark_on, g_stop,
                  [tokenizer](const std::vector<std::uint32_t>& ids) { return tokenizer->decode(ids, true); },
                  session_runtime ? &session_service : nullptr);
        try { lead.loop(); } catch (...) { session_tasks.close(); queue.shutdown(); http.stop(); http_thread.join(); throw; }
        session_tasks.close();
        queue.shutdown();
        http.stop();
        http_thread.join();
        std::printf("rank 0: STOPPED\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-minimax-serve error: %s\n", e.what());
        return 2;
    }
}
