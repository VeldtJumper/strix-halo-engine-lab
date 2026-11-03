// strix-glm5-serve host-only tests: control block encode/decode, hello / row checks, finish decisions, the slot table,
// and the lead + follower protocol over an in-process loopback with a deterministic fake engine (batched streams must
// equal their alone runs; serial mode; world 1; context limit; a cross-rank token mismatch aborts both ranks).
#include "glm/serve/glm_serve_control.hpp"
#include "glm/serve/glm_serve_lead.hpp"
#include "glm/serve/glm_serve_selftest.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <optional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace gs = strix::glm::serve;
namespace sv = strix::serve;

namespace {

int failures = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);        \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

template<class F>
bool throws_with(F&& f, const std::string& needle) {
    try {
        f();
    } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

// Deterministic per-slot sequence model: a slot's tokens depend only on its own history (slot id excluded), as the
// engine's batch invariance promises. Positions are checked against the slot's frontier. The fake MTP path emits the
// same chain 1..3 tokens per verify cycle, with the real adapter's rules: a decoder opens only from a fresh seed (MTP
// prefill or the last step's row, checked against that step), no other engine call while it is open. Every call is
// logged so both ranks' call sequences can be compared.
struct FakeEng {
    std::uint32_t slots;
    bool stops = true;
    std::uint32_t corrupt_call = 0;  // > 0: the n-th step returns a different first token
    unsigned delay_us = 0;           // per step / verify cycle
    std::vector<std::uint64_t> h, frontier;
    std::vector<bool> ready;
    std::uint64_t calls = 0, steps = 0, cycles = 0, prefills = 0;
    std::vector<std::size_t> step_rows;
    std::vector<std::string> log;
    struct Seed { std::uint32_t slot, y0; std::uint64_t p; };
    std::optional<Seed> seed;
    struct Dec { std::uint32_t slot, last, budget, emitted; };
    std::optional<Dec> dec;
    std::vector<gs::WireRow> last_rows;
    std::vector<std::uint32_t> last_out;
    // batched self-speculation: per slot drafter state + the slot's anchor (its last emitted token)
    std::vector<gs::SlotSpec> spec;
    std::vector<std::uint32_t> anchor;
    std::uint64_t spec_verifies = 0, spec_drafts = 0, multi_slot_verifies = 0;
    explicit FakeEng(std::uint32_t s) : slots(s), h(s, 0), frontier(s, 0), ready(s, false), spec(s), anchor(s, 0) {}
    // the fake drafter: the true next token after the chain state x, wrong for a quarter of the states
    std::pair<std::uint32_t, float> guess(std::uint64_t x) const {
        const std::uint32_t t = tok(x);
        const bool ok = ((x >> 17) % 4U) != 0;
        return {ok ? t : t ^ 0x2BU, ok ? 0.85F : 0.35F};
    }
    std::uint64_t chain(std::uint32_t s) const {  // state after the anchor and the slot's drafts
        std::uint64_t x = fold(h[s], anchor[s]);
        for (const auto d : spec[s].drafts) x = fold(x, d);
        return x;
    }
    void seed_spec(std::uint32_t s) {
        spec[s] = {};
        spec[s].seeded = true;
        spec[s].p = frontier[s];
        const auto g = guess(fold(h[s], anchor[s]));
        spec[s].drafts.push_back(g.first);
        spec[s].conf.push_back(g.second);
    }
    [[nodiscard]] const gs::SlotSpec& spec_state(std::uint32_t s) const { return spec.at(s); }
    std::uint32_t spec_draft(std::uint32_t s) {
        plain_call();
        if (!spec[s].seeded || spec[s].drafts.empty()) throw std::runtime_error("fake spec_draft: no chain");
        const auto g = guess(chain(s));
        spec[s].drafts.push_back(g.first);
        spec[s].conf.push_back(g.second);
        ++spec_drafts;
        log.push_back("draft " + std::to_string(s));
        return g.first;
    }
    std::vector<std::uint32_t> spec_verify(std::span<const gs::WireRow> rows, std::span<const std::uint32_t> budgets) {
        plain_call();
        const auto win = gs::windows_of(rows, slots, 1U << 30);
        if (budgets.size() != win.size()) throw std::runtime_error("fake spec_verify: budgets");
        std::vector<std::uint32_t> out;
        std::string l = "verify";
        for (std::size_t i = 0; i < win.size(); ++i) {
            const auto& w = win[i];
            const auto s = w.slot;
            if (rows[w.row0].pos != frontier[s] || rows[w.row0].token != anchor[s]) throw std::runtime_error("fake spec_verify: anchor");
            std::vector<std::uint32_t> drafts, target;
            for (std::uint32_t j = 1; j < w.n; ++j) {
                drafts.push_back(rows[w.row0 + j].token);
                if (j > spec[s].drafts.size() || spec[s].drafts[j - 1] != drafts.back()) throw std::runtime_error("fake spec_verify: drafts");
            }
            std::uint64_t x = h[s];
            for (std::uint32_t j = 0; j < w.n; ++j) {
                x = fold(x, rows[w.row0 + j].token);
                target.push_back(tok(x));
            }
            const auto acc = strix::glm::spec::accept_greedy(drafts, target, strix::glm::spec::kNoFlag,
                                                             strix::glm::spec::Terminal{strix::glm::spec::kNoFlag, budgets[i]});
            for (unsigned j = 0; j < acc.rows; ++j) (void)row(s, rows[w.row0 + j].token);
            std::vector<std::uint32_t> emitted(drafts.begin(), drafts.begin() + acc.m);
            if (acc.has_bonus) emitted.push_back(acc.bonus);
            anchor[s] = emitted.back();
            out.push_back(static_cast<std::uint32_t>(emitted.size()));
            out.insert(out.end(), emitted.begin(), emitted.end());
            if (acc.stop) spec[s] = {};
            else seed_spec(s);
            l += " " + std::to_string(s) + "@" + std::to_string(rows[w.row0].pos) + "+" + std::to_string(w.n);
        }
        ++spec_verifies;
        if (win.size() > 1) ++multi_slot_verifies;
        log.push_back(l);
        wait();
        return out;
    }
    std::uint32_t tok(std::uint64_t x) const {
        if (stops && x % 53 == 7) return 154820U;
        return static_cast<std::uint32_t>((x >> 11) % 150000U) + 1U;
    }
    static std::uint64_t fold(std::uint64_t x, std::uint32_t t) { return (x ^ t) * 1099511628211ULL + 0x9e3779b97f4a7c15ULL; }
    void plain_call() {
        if (dec) throw std::runtime_error("fake: engine call while an MTP decoder is open");
        seed.reset();
        last_rows.clear();
        last_out.clear();
    }
    void wait() const {
        if (delay_us) std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    }
    // one trunk row of `slot` at its frontier
    std::uint32_t row(std::uint32_t s, std::uint32_t token) {
        h[s] = fold(h[s], token);
        ++frontier[s];
        return tok(h[s]);
    }
    void reset(std::uint32_t s) {
        plain_call();
        spec.at(s) = {};
        h.at(s) = 14695981039346656037ULL;
        frontier.at(s) = 0;
        ready.at(s) = true;
        ++calls;
        log.push_back("reset " + std::to_string(s));
    }
    std::uint32_t prefill(std::uint32_t s, std::span<const std::uint32_t> t, std::uint64_t pos0) {
        plain_call();
        if (!ready.at(s) || pos0 != frontier[s]) throw std::runtime_error("fake prefill: not at the slot frontier / without reset");
        for (const auto x : t) h[s] = fold(h[s], x);
        frontier[s] = pos0 + t.size();
        ++calls;
        ++prefills;
        log.push_back("prefill " + std::to_string(s) + " @" + std::to_string(pos0) + " +" + std::to_string(t.size()));
        spec[s] = {};
        anchor[s] = tok(h[s]);
        if (delay_us) std::this_thread::sleep_for(std::chrono::microseconds(delay_us * (t.size() / 32 + 1)));
        return tok(h[s]);
    }
    std::vector<std::uint32_t> step(std::span<const gs::WireRow> rows) {
        plain_call();
        std::vector<std::uint32_t> out;
        ++steps;
        step_rows.push_back(rows.size());
        std::string l = "step";
        for (const auto& r : rows) {
            if (r.pos != frontier.at(r.slot)) throw std::runtime_error("fake step: position not at the slot frontier");
            out.push_back(row(r.slot, r.token));
            spec[r.slot] = {};
            l += " " + std::to_string(r.slot) + "@" + std::to_string(r.pos);
        }
        if (corrupt_call != 0 && steps == corrupt_call) out[0] ^= 1U;
        for (std::size_t i = 0; i < rows.size(); ++i) anchor[rows[i].slot] = out[i];
        ++calls;
        log.push_back(l);
        last_rows.assign(rows.begin(), rows.end());
        last_out = out;
        wait();
        return out;
    }
    std::uint32_t prefill_mtp(std::uint32_t s, std::span<const std::uint32_t> t, std::uint64_t pos0, bool last) {
        const auto y0 = prefill(s, t, pos0);
        log.back() += last ? " mtp last" : " mtp";
        if (last) {
            seed = Seed{s, y0, pos0 + t.size()};
            seed_spec(s);
        }
        return y0;
    }
    void mtp_enter(std::uint32_t r, const gs::WireRow& w, bool from_seed) {
        if (from_seed) {
            const auto& st = spec.at(w.slot);
            if (!st.seeded || st.drafts.size() != 1 || st.p != w.pos || anchor[w.slot] != w.token || frontier[w.slot] != w.pos)
                throw std::runtime_error("fake mtp_enter: no fresh batched seed");
            plain_call();
            seed = Seed{w.slot, w.token, w.pos};
            spec[w.slot] = {};
            log.push_back("enter-seed " + std::to_string(w.slot) + "@" + std::to_string(w.pos));
            return;
        }
        if (r >= last_rows.size() || last_rows[r].slot != w.slot || last_rows[r].pos != w.pos || last_out[r] != w.token)
            throw std::runtime_error("fake mtp_enter: not the last step's row");
        plain_call();
        if (frontier[w.slot] != w.pos + 1) throw std::runtime_error("fake mtp_enter: frontier");
        seed = Seed{w.slot, w.token, w.pos + 1};
        log.push_back("enter " + std::to_string(w.slot) + "@" + std::to_string(w.pos));
    }
    std::vector<std::uint32_t> mtp_run(std::uint32_t s, std::uint32_t budget, std::uint32_t n) {
        if (!dec) {
            if (!seed || seed->slot != s || frontier[s] != seed->p) throw std::runtime_error("fake mtp_run without a fresh seed");
            dec = Dec{s, seed->y0, budget, 1};
            seed.reset();
            spec[s] = {};
        } else if (dec->slot != s) {
            throw std::runtime_error("fake mtp_run: other slot");
        }
        std::vector<std::uint32_t> out;
        for (std::uint32_t c = 0; c < n && dec->emitted < dec->budget; ++c) {
            const unsigned k = 1U + static_cast<unsigned>(h[s] % 3U);
            for (unsigned i = 0; i < k && dec->emitted < dec->budget; ++i) {
                dec->last = row(s, dec->last);
                out.push_back(dec->last);
                ++dec->emitted;
            }
            ++cycles;
            wait();
        }
        anchor[s] = dec->last;
        log.push_back("run " + std::to_string(s) + " -> " + std::to_string(out.size()));
        return out;
    }
    void mtp_leave(std::uint32_t s) {
        if (dec && dec->slot != s) throw std::runtime_error("fake mtp_leave: other slot");
        dec.reset();
        seed.reset();
        spec.at(s) = {};
        log.push_back("leave " + std::to_string(s));
    }
};

// In-process wire: one message queue per direction; exchange = send, then receive (5 s timeout).
class Loopback {
public:
    gs::Channel::Wire side(int s) {
        return [this, s](std::span<const std::byte> out, std::span<std::byte> in, std::uint32_t, std::uint64_t) {
            {
                std::lock_guard<std::mutex> l(m_);
                q_[s].emplace_back(out.begin(), out.end());
                cv_.notify_all();
            }
            std::unique_lock<std::mutex> l(m_);
            if (!cv_.wait_for(l, std::chrono::seconds(5), [&] { return !q_[1 - s].empty(); }))
                throw std::runtime_error("loopback: peer timeout");
            auto msg = std::move(q_[1 - s].front());
            q_[1 - s].pop_front();
            if (msg.size() != in.size()) throw std::runtime_error("loopback: size mismatch");
            std::copy(msg.begin(), msg.end(), in.begin());
        };
    }
private:
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::vector<std::byte>> q_[2];
};

std::vector<std::vector<std::uint32_t>> prompts(unsigned n) {
    std::vector<std::vector<std::uint32_t>> p;
    for (unsigned i = 0; i < n; ++i) {
        std::vector<std::uint32_t> t;
        for (unsigned k = 0; k < 5 + 7 * i; ++k) t.push_back(1000U + 13U * i + k);
        p.push_back(std::move(t));
    }
    return p;
}

gs::Hello hello_for(std::uint32_t slots, std::uint64_t cap, std::uint32_t world) {
    gs::Hello h;
    h.model_hash = 0x1234;
    h.identity_hash = 0x5678;
    h.capacity = cap;
    h.slots = slots;
    h.max_rows = 4096;
    h.step_rows = gs::kMaxRows;
    h.world = world;
    return h;
}

struct PairRun {
    gs::SelftestResult result;
    std::string lead_error, follow_error, driver_error;
    gs::LeadStats stats;
    std::vector<std::size_t> rows0, rows1;
    std::vector<std::string> log0, log1;
};

PairRun run_pair(unsigned n, std::uint32_t slots, bool batch, std::uint32_t tokens, std::uint32_t corrupt = 0,
                 gs::Hello h1 = hello_for(8, 4096, 2), bool mtp = false, unsigned chunk = 0, unsigned long_n = 0,
                 unsigned delay_us = 0, bool skip_basic = false, bool g2 = false, std::uint64_t* multi = nullptr) {
    PairRun out;
    Loopback lb;
    FakeEng e0(slots), e1(slots);
    e1.corrupt_call = corrupt;
    e0.delay_us = e1.delay_us = delay_us;
    gs::Channel c0(lb.side(0), 0), c1(lb.side(1), 1);
    const gs::Hello h0 = hello_for(slots, 4096, 2);
    if (h1.slots == 8 && slots != 8) h1.slots = slots;
    std::thread follower([&] {
        try {
            gs::follow(e1, c1, h1, slots, 4096);
        } catch (const std::exception& e) {
            out.follow_error = e.what();
        }
    });
    std::atomic<bool> stop{false};
    sv::GenerationQueue queue(64);
    gs::LeadConfig lc;
    lc.slots = slots;
    lc.capacity = 4096;
    lc.batch = batch;
    lc.heartbeat_seconds = 0.01;
    lc.mtp = mtp;
    lc.mtp_cycles = 2;
    lc.prefill_chunk = chunk;
    lc.g2 = g2;
    try {
        c0.hello(h0);
        gs::Lead<FakeEng> lead(e0, c0, queue, lc, stop);
        lead.set_trace(true);
        std::thread driver([&] {
            try {
                gs::SelftestHooks hooks;
                if (mtp) hooks.set_mtp = [&lead](bool on) { lead.set_mtp(on); };
                hooks.take_trace = [&lead] { return lead.take_trace(); };
                hooks.set_prefill_chunk = [&lead](unsigned c) { lead.set_prefill_chunk(c); };
                hooks.progress = [&lead] { return lead.progress(); };
                hooks.prefill_chunk = chunk;
                hooks.extra_chunks = {64U};
                hooks.skip_basic = skip_basic;
                if (g2) hooks.set_g2 = [&lead](bool on) { lead.set_g2(on); };
                for (unsigned i = 0; i < long_n; ++i) {
                    std::vector<std::uint32_t> t;
                    for (unsigned k = 0; k < 1000 + 37 * i; ++k) t.push_back(2000U + 7U * i + k % 911U);
                    hooks.long_prompts.push_back(std::move(t));
                }
                out.result = gs::run_selftest(queue, prompts(n), tokens, gs::default_stop_ids(), slots, [](const std::string& l) {
                    std::printf("  %s\n", l.c_str());
                }, hooks);
            } catch (const std::exception& e) {
                out.driver_error = e.what();
            }
            stop.store(true);
        });
        try {
            lead.loop();
        } catch (const std::exception& e) {
            out.lead_error = e.what();
            queue.shutdown();
        }
        driver.join();
        out.stats = lead.stats();
    } catch (const std::exception& e) {
        out.lead_error = e.what();
    }
    follower.join();
    out.rows0 = e0.step_rows;
    out.rows1 = e1.step_rows;
    out.log0 = e0.log;
    out.log1 = e1.log;
    if (multi) *multi = e0.multi_slot_verifies;
    return out;
}

}  // namespace

int main() {
    // ---- encode / decode
    {
        gs::Control c;
        c.cmd = static_cast<std::uint32_t>(gs::Cmd::step);
        c.nrows = 2;
        c.row[0] = {0, 5, 10};
        c.row[1] = {1, 6, 11};
        c.resets = 0x5;
        c.hello = hello_for(8, 8192, 2);
        const auto b = gs::encode(c);
        const auto d = gs::decode(b);
        CHECK(d.cmd == c.cmd && d.nrows == 2 && d.row[1] == c.row[1] && d.resets == 0x5 && d.hello == c.hello);
        auto bad = b;
        bad[0] = std::byte{0};
        CHECK(throws_with([&] { (void)gs::decode(bad); }, "bad magic"));
        gs::Control v = c;
        v.version = 99;
        CHECK(throws_with([&] { (void)gs::decode(gs::encode(v)); }, "protocol version"));
        gs::Control r = c;
        r.nrows = 9;
        CHECK(throws_with([&] { (void)gs::decode(gs::encode(r)); }, "more than 8 rows"));
        auto pad = b;
        pad[gs::kControlBytes - 1] = std::byte{1};
        CHECK(throws_with([&] { (void)gs::decode(pad); }, "padding"));
        gs::Control k = c;
        k.cmd = 77;
        CHECK(throws_with([&] { (void)gs::decode(gs::encode(k)); }, "unknown command"));
    }
    // ---- hello, rows
    {
        const auto a = hello_for(8, 8192, 2);
        auto b = a;
        CHECK(gs::hello_mismatch(a, b).empty());
        b.slots = 4;
        b.identity_hash = 1;
        const auto m = gs::hello_mismatch(a, b);
        CHECK(m.find("slots") != std::string::npos && m.find("identity_hash") != std::string::npos);
        std::vector<gs::WireRow> rows = {{0, 1, 3}, {1, 1, 3}, {2, 1, 9}};
        CHECK(gs::consecutive_slots(rows));
        gs::check_rows(rows, 8, 16);
        rows[2].slot = 4;
        CHECK(!gs::consecutive_slots(rows));
        rows[2].slot = 1;
        CHECK(throws_with([&] { gs::check_rows(rows, 8, 16); }, "ascending"));
        rows[2] = {9, 1, 0};
        CHECK(throws_with([&] { gs::check_rows(rows, 8, 16); }, "slot out of range"));
        rows[2] = {3, 1, 16};
        CHECK(throws_with([&] { gs::check_rows(rows, 8, 16); }, "capacity"));
        CHECK(throws_with([&] { gs::check_rows({}, 8, 16); }, "1..8"));
    }
    // ---- decide, slot table
    {
        const auto& st = gs::default_stop_ids();
        CHECK(gs::decide(154827U, 1, 10, 5, 100, st) == gs::Next::stop);
        CHECK(gs::decide(42U, 10, 10, 5, 100, st) == gs::Next::length);
        CHECK(gs::decide(42U, 3, 10, 100, 100, st) == gs::Next::context);
        CHECK(gs::decide(42U, 3, 10, 99, 100, st) == gs::Next::decode);
        gs::SlotTable t(4);
        CHECK(*t.alloc() == 0 && *t.alloc() == 1 && *t.alloc() == 2);
        t.release(1);
        CHECK(*t.alloc() == 1);
        CHECK(*t.alloc() == 3 && !t.alloc() && t.full());
        t.release(0);
        CHECK(t.live() == 3 && *t.alloc() == 0);
    }
    // ---- channel: rounds, tokens, echo (world 1 mirror and loopback)
    {
        gs::Channel solo({}, 0);
        solo.hello(hello_for(1, 64, 1));
        gs::Control c;
        c.cmd = static_cast<std::uint32_t>(gs::Cmd::heartbeat);
        (void)solo.exchange(c);
        CHECK(solo.round() == 2);
        Loopback lb;
        gs::Channel a(lb.side(0), 0), b(lb.side(1), 1);
        std::vector<std::uint32_t> big(40000), got;
        for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<std::uint32_t>(i * 7);
        std::thread t([&] { b.tokens(got, big.size(), false); });
        a.tokens(big, big.size(), true);
        t.join();
        CHECK(got == big);
    }
    // ---- lead + follower over the loopback
    std::printf("batched, 6 prompts, 4 slots:\n");
    {
        const auto r = run_pair(6, 4, true, 40);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact);
        CHECK(r.rows0 == r.rows1);  // the follower made exactly the lead's step calls
        bool multi = false;
        for (const auto n : r.rows0) multi = multi || n > 1;
        CHECK(multi);  // steps really batched rows
        CHECK(r.stats.active == 0 && r.stats.finished > 0);
        if (!r.lead_error.empty()) std::printf("  lead error: %s\n", r.lead_error.c_str());
        if (!r.follow_error.empty()) std::printf("  follow error: %s\n", r.follow_error.c_str());
    }
    std::printf("batched, 8 prompts, 8 slots, no early stops in some streams:\n");
    {
        const auto r = run_pair(8, 8, true, 64);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.result.exact && r.rows0 == r.rows1);
        for (const auto& l : r.result.levels) {
            if (!l.window.valid) continue;  // early stops can end a stream before the last prefill
            CHECK(l.window.tokens > 0 && l.window.seconds > 0);
            std::uint64_t sum = 0;
            for (const auto n : l.window.per_stream) sum += n;
            CHECK(sum == l.window.tokens && l.window.per_stream.size() == l.streams.size());
            if (l.c > 1) CHECK(l.window.steps == 0 || l.window.rows >= l.window.steps);
        }
    }
    std::printf("serial (OFF switch), 4 prompts:\n");
    {
        const auto r = run_pair(4, 4, false, 24);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.result.exact);
        for (const auto n : r.rows0) CHECK(n == 1);
    }
    std::printf("cross-rank mismatch (rank 1 corrupts its 5th step):\n");
    {
        const auto r = run_pair(3, 4, true, 30, 5);
        CHECK(r.lead_error.find("cross-rank token mismatch") != std::string::npos);
        CHECK(r.follow_error.find("cross-rank token mismatch") != std::string::npos);
        CHECK(!r.result.exact);
        std::printf("  lead: %s\n  follower: %s\n", r.lead_error.c_str(), r.follow_error.c_str());
    }
    std::printf("hello mismatch (capacity):\n");
    {
        auto h1 = hello_for(4, 8192, 2);
        const auto r = run_pair(2, 4, true, 8, 0, h1);
        CHECK(r.lead_error.find("hello mismatch") != std::string::npos && r.follow_error.find("hello mismatch") != std::string::npos);
    }
    std::printf("C1 MTP + batched, 6 prompts, 4 slots:\n");
    {
        const auto r = run_pair(6, 4, true, 48, 0, hello_for(8, 4096, 2), true);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact);
        CHECK(r.log0 == r.log1);  // the follower made exactly the lead's engine calls
        CHECK(r.stats.mtp_rounds > 0 && r.stats.mtp_bootstraps > 0 && r.stats.mtp_enters > 0 && r.stats.mtp_leaves > 0);
        bool multi = false;
        for (const auto x : r.rows0) multi = multi || x > 1;
        CHECK(multi);
        std::printf("  mtp rounds %llu tokens %llu enters %llu leaves %llu, steps %llu\n",
                    static_cast<unsigned long long>(r.stats.mtp_rounds), static_cast<unsigned long long>(r.stats.mtp_tokens),
                    static_cast<unsigned long long>(r.stats.mtp_enters), static_cast<unsigned long long>(r.stats.mtp_leaves),
                    static_cast<unsigned long long>(r.stats.steps));
        if (!r.lead_error.empty()) std::printf("  lead error: %s\n", r.lead_error.c_str());
        if (!r.follow_error.empty()) std::printf("  follow error: %s\n", r.follow_error.c_str());
    }
    std::printf("C1 MTP, serial, 3 prompts:\n");
    {
        const auto r = run_pair(3, 4, false, 40, 0, hello_for(8, 4096, 2), true);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.result.exact && r.log0 == r.log1);
        CHECK(r.stats.mtp_rounds > 0);
    }
    std::printf("prefill interleave: chunk 128, long prompts, MTP, 4 slots:\n");
    {
        const auto r = run_pair(4, 4, true, 256, 0, hello_for(8, 4096, 2), true, 128, 4, 200);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact);
        CHECK(r.log0 == r.log1);
        bool chunked = false;
        for (const auto& line : r.log0) chunked = chunked || line.find(" @128 +128") != std::string::npos;
        CHECK(chunked);  // prompts really went in chunks at pos0 > 0
        CHECK(r.result.stall_whole_ms > 2.0 * r.result.stall_chunked_ms && r.result.stall_chunked_ms > 0);
        // adaptive: a prompt admitted with nothing decoding goes in one call (the first long C=1 runs)
        bool whole_alone = false;
        for (const auto& line : r.log0) whole_alone = whole_alone || line.rfind("prefill 0 @0 +1000", 0) == 0;
        CHECK(whole_alone);
        if (!r.lead_error.empty()) std::printf("  lead error: %s\n", r.lead_error.c_str());
        if (!r.follow_error.empty()) std::printf("  follow error: %s\n", r.follow_error.c_str());
        if (!r.driver_error.empty()) std::printf("  driver error: %s\n", r.driver_error.c_str());
    }
    std::printf("G2 batched self-speculation, 6 prompts, 4 slots, MTP:\n");
    {
        std::uint64_t multi = 0;
        const auto r = run_pair(6, 4, true, 64, 0, hello_for(8, 4096, 2), true, 0, 0, 0, false, true, &multi);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact);
        CHECK(r.log0 == r.log1);  // the follower made exactly the lead's engine calls
        CHECK(r.stats.g2_cycles > 0 && multi > 0 && r.stats.g2_drafts > 0 && r.stats.g2_accepted > 0);
        std::printf("  g2 cycles %llu rows %llu (%.2f/cycle) drafts %llu accepted %llu extension calls %llu, multi-slot verifies %llu\n",
                    static_cast<unsigned long long>(r.stats.g2_cycles), static_cast<unsigned long long>(r.stats.g2_rows),
                    r.stats.g2_cycles ? static_cast<double>(r.stats.g2_rows) / static_cast<double>(r.stats.g2_cycles) : 0.0,
                    static_cast<unsigned long long>(r.stats.g2_drafts), static_cast<unsigned long long>(r.stats.g2_accepted),
                    static_cast<unsigned long long>(r.stats.g2_draft_calls), static_cast<unsigned long long>(multi));
        if (!r.lead_error.empty()) std::printf("  lead error: %s\n", r.lead_error.c_str());
        if (!r.follow_error.empty()) std::printf("  follow error: %s\n", r.follow_error.c_str());
        if (!r.driver_error.empty()) std::printf("  driver error: %s\n", r.driver_error.c_str());
    }
    std::printf("G2 + chunked long prompts + staggered arrivals, 8 slots:\n");
    {
        const auto r = run_pair(8, 8, true, 96, 0, hello_for(8, 4096, 2), true, 128, 4, 0, false, true);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty() && r.result.exact && r.log0 == r.log1);
        CHECK(r.stats.g2_cycles > 0);
        // C=6 / C=8: plain steps (no room for drafts), G2 again with fewer slots after a re-seed
        CHECK(r.stats.g2_plain > 0 && r.stats.g2_reseeds > 0);
        std::size_t wide_spec = 0;
        for (const auto& line : r.log0) {
            if (line.rfind("verify", 0) != 0) continue;
            std::size_t windows = 0;
            for (std::size_t p = line.find('@'); p != std::string::npos; p = line.find('@', p + 1)) ++windows;
            if (windows > 4) ++wide_spec;
        }
        CHECK(wide_spec == 0);  // never a windowed verify of more than 4 slots
        std::printf("  g2 cycles %llu, handover plain steps %llu, re-seeds %llu, steps %llu\n",
                    static_cast<unsigned long long>(r.stats.g2_cycles), static_cast<unsigned long long>(r.stats.g2_plain),
                    static_cast<unsigned long long>(r.stats.g2_reseeds), static_cast<unsigned long long>(r.stats.steps));
        if (!r.lead_error.empty()) std::printf("  lead error: %s\n", r.lead_error.c_str());
        if (!r.driver_error.empty()) std::printf("  driver error: %s\n", r.driver_error.c_str());
    }
    std::printf("skip basic: only the long reference and the staggered arms:\n");
    {
        const auto r = run_pair(3, 4, true, 40, 0, hello_for(8, 4096, 2), true, 128, 3, 0, true);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty() && r.result.exact && r.log0 == r.log1);
        CHECK(r.result.levels.empty());  // no short levels ran
    }
    std::printf("prefill interleave: chunk 100 (not a divisor), plain:\n");
    {
        const auto r = run_pair(3, 4, true, 40, 0, hello_for(8, 4096, 2), false, 100, 3);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty() && r.result.exact && r.log0 == r.log1);
    }
    // ---- decode-first: a running stream gets one decode round between consecutive prefill chunks of a new prompt
    {
        FakeEng e(4);
        e.stops = false;
        gs::Channel ch({}, 0);
        ch.hello(hello_for(4, 4096, 1));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 4;
        lc.capacity = 4096;
        lc.heartbeat_seconds = 0.01;
        lc.prefill_chunk = 64;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        auto job = [&](std::vector<std::uint32_t> p, std::uint32_t max) {
            sv::GenerationJob j;
            j.prompt = std::move(p);
            j.max_tokens = max;
            return queue.submit(std::move(j));
        };
        auto ta = job(prompts(1)[0], 400);
        std::vector<std::uint32_t> a;
        bool done = false;
        while (a.size() < 5 && !done) {
            auto g = ta->wait(std::chrono::milliseconds(100), done);
            a.insert(a.end(), g.begin(), g.end());
        }
        std::vector<std::uint32_t> big(640, 5U);
        auto tb = job(big, 4);
        bool done_b = false;
        while (!done_b) (void)tb->wait(std::chrono::milliseconds(100), done_b);
        while (!done) (void)ta->wait(std::chrono::milliseconds(100), done);
        stop.store(true);
        runner.join();
        // between two chunk calls of the big prompt there is exactly one step
        int last_chunk = -1, violations = 0, chunks = 0;
        for (std::size_t i = 0; i < e.log.size(); ++i) {
            if (e.log[i].rfind("prefill 1 ", 0) == 0) {
                if (last_chunk >= 0) {
                    int st = 0;
                    for (std::size_t j = static_cast<std::size_t>(last_chunk) + 1; j < i; ++j) st += e.log[j].rfind("step", 0) == 0 ? 1 : 0;
                    if (st != 1) ++violations;
                }
                last_chunk = static_cast<int>(i);
                ++chunks;
            }
        }
        CHECK(chunks == 10 && violations == 0);
        std::printf("decode-first interleave: %d chunks of the 640-token prompt, one step between consecutive chunks: %s\n", chunks,
                    violations == 0 ? "yes" : "NO");
    }
    // ---- a second request arriving while the first decodes on the MTP path: leave, batched steps, enter again
    {
        FakeEng e(4);
        e.stops = false;
        e.delay_us = 300;
        gs::Channel ch({}, 0);
        ch.hello(hello_for(4, 4096, 1));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 4;
        lc.capacity = 4096;
        lc.heartbeat_seconds = 0.01;
        lc.mtp = true;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        auto job = [&](std::vector<std::uint32_t> p, std::uint32_t max) {
            sv::GenerationJob j;
            j.prompt = std::move(p);
            j.max_tokens = max;
            return queue.submit(std::move(j));
        };
        const auto pa = prompts(2)[0], pb = prompts(2)[1];
        auto ta = job(pa, 300);
        std::vector<std::uint32_t> a;
        bool done = false;
        while (a.size() < 20 && !done) {
            auto g = ta->wait(std::chrono::milliseconds(100), done);
            a.insert(a.end(), g.begin(), g.end());
        }
        auto tb = job(pb, 40);
        std::vector<std::uint32_t> b;
        bool done_b = false;
        while (!done_b) {
            auto g = tb->wait(std::chrono::milliseconds(100), done_b);
            b.insert(b.end(), g.begin(), g.end());
        }
        while (!done) {
            auto g = ta->wait(std::chrono::milliseconds(100), done);
            a.insert(a.end(), g.begin(), g.end());
        }
        stop.store(true);
        runner.join();
        // reference: the plain chain of each prompt alone
        auto plain = [](const std::vector<std::uint32_t>& p, std::size_t n) {
            FakeEng f(1);
            f.stops = false;
            f.reset(0);
            std::vector<std::uint32_t> t{f.prefill(0, p, 0)};
            while (t.size() < n) {
                const gs::WireRow r{0, t.back(), f.frontier[0]};
                t.push_back(f.step({&r, 1})[0]);
            }
            return t;
        };
        CHECK(a == plain(pa, 300));
        CHECK(b == plain(pb, 40));
        const auto st = lead.stats();
        CHECK(st.mtp_leaves >= 1 && st.mtp_enters >= 1 && st.steps > 0 && st.mtp_rounds > 0);
        std::printf("arrival during MTP: A %zu tokens, B %zu tokens, mtp rounds %llu enters %llu leaves %llu, steps %llu\n", a.size(),
                    b.size(), static_cast<unsigned long long>(st.mtp_rounds), static_cast<unsigned long long>(st.mtp_enters),
                    static_cast<unsigned long long>(st.mtp_leaves), static_cast<unsigned long long>(st.steps));
    }
    // ---- world 1, context limit, rejected prompt, max_tokens
    {
        FakeEng e(2);
        e.stops = false;
        gs::Channel ch({}, 0);
        ch.hello(hello_for(2, 64, 1));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 2;
        lc.capacity = 64;
        lc.heartbeat_seconds = 0.01;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        auto job = [&](std::size_t n, std::uint32_t max) {
            sv::GenerationJob j;
            j.prompt.assign(n, 7U);
            j.max_tokens = max;
            return queue.submit(std::move(j));
        };
        auto drain = [](const std::shared_ptr<sv::GenerationTicket>& t) {
            std::vector<std::uint32_t> all;
            bool done = false;
            while (!done) {
                auto g = t->wait(std::chrono::milliseconds(100), done);
                all.insert(all.end(), g.begin(), g.end());
            }
            return all;
        };
        auto t1 = job(40, 1000);
        auto t2 = job(64, 10);
        auto t3 = job(10, 5);
        const auto a1 = drain(t1), a2 = drain(t2), a3 = drain(t3);
        CHECK(a1.size() == 25 && t1->stats().finish == "context");  // first token + rows at positions 40..63
        CHECK(a2.empty() && t2->stats().error_code == "context_length_exceeded");
        CHECK(a3.size() == 5 && t3->stats().finish == "length");
        stop.store(true);
        runner.join();
    }
    if (failures == 0) std::printf("glm serve cpu tests: ALL PASS\n");
    else std::printf("glm serve cpu tests: %d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
