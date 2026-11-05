// strix-glm5-serve host-only tests: control block encode/decode, hello / row checks, finish decisions, the slot table,
// and the lead + follower protocol over an in-process loopback with a deterministic fake engine (batched streams must
// equal their alone runs; serial mode; world 1; context limit; a cross-rank token mismatch aborts both ranks).
#include "glm/serve/glm_serve_control.hpp"
#include "glm/serve/glm_serve_lead.hpp"
#include "glm/serve/glm_serve_import.hpp"
#include "glm/serve/glm_serve_selftest.hpp"
#include "glm/rtxpf/import/glm_import_socket.hpp"

#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <atomic>
#include <functional>
#include <map>
#include <optional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
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
// In-process pair exchange for the 'DSA once' relay: each side posts its block and waits for the other's (lock-step,
// as Communicator::exchange_host); remote empty = not wanted.
struct Pipe {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::vector<std::byte>> q[2];  // q[r]: blocks sent by rank r, not yet taken
    void xchg(unsigned rank, std::span<const std::byte> local, std::span<std::byte> remote) {
        std::unique_lock<std::mutex> l(m);
        q[rank].emplace_back(local.begin(), local.end());
        cv.notify_all();
        if (!cv.wait_for(l, std::chrono::seconds(10), [&] { return !q[1 - rank].empty(); }))
            throw std::runtime_error("pipe: the peer never exchanged");
        auto b = std::move(q[1 - rank].front());
        q[1 - rank].pop_front();
        if (b.size() != local.size()) throw std::runtime_error("pipe: block sizes differ");
        if (!remote.empty()) std::copy(b.begin(), b.end(), remote.begin());
    }
};
// A synthetic per-rank layout (KDA, DSA, KDA, DSA, DSA) and its payloads: KDA bytes differ per rank, DSA bytes are the
// same for both (as the RTX producer's).
const std::vector<gs::StateComponent> kRelayLayout = {{false, 40}, {true, 100}, {false, 24}, {true, 300}, {true, 7}};
std::vector<std::uint8_t> relay_part(bool dsa, unsigned rank) {
    std::vector<std::uint8_t> v(static_cast<std::size_t>(gs::component_bytes(kRelayLayout, dsa)));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::uint8_t>(dsa ? 0x80U + i * 7U : rank * 16U + i * 3U);
    return v;
}
std::vector<std::uint8_t> relay_full(unsigned rank) {
    return gs::merge_dsa(relay_part(false, rank), relay_part(true, 0), kRelayLayout);
}

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
    bool refuse_prefix = false, fail_prefix_restore = false;
    bool prefix_store_allowed() const { return !refuse_prefix; }
    std::vector<std::uint8_t> state_bytes_upto(std::uint32_t s, std::uint64_t p) {
        if (frontier.at(s) != p) throw std::runtime_error("fake snapshot frontier");
        std::vector<std::uint8_t> bytes(sizeof(std::uint64_t));
        std::memcpy(bytes.data(), &h.at(s), bytes.size());
        return bytes;
    }
    void restore_state(std::uint32_t s, const std::vector<std::uint8_t>& bytes, std::uint64_t p) {
        if (fail_prefix_restore) throw std::runtime_error("fake prefix restore failure");
        if (!ready.at(s) || frontier[s] != 0 || bytes.size() != sizeof(std::uint64_t)) throw std::runtime_error("fake prefix restore");
        std::memcpy(&h.at(s), bytes.data(), bytes.size());
        frontier[s] = p;
        spec[s] = {};
        log.push_back("restore " + std::to_string(s) + " @" + std::to_string(p));
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
    // import: a staged payload = the state of a prefix (hash + frontier), as Engine::restore_state would write it.
    struct Staged { std::uint64_t h, p; };
    std::map<std::uint32_t, Staged> staged;
    void stage(std::uint32_t id, std::span<const std::uint32_t> prefix) {
        std::uint64_t x = 14695981039346656037ULL;
        for (const auto t : prefix) x = fold(x, t);
        staged[id] = Staged{x, prefix.size()};
    }
    // import_expect: the fake "receives" at once (the prefix state of the announced prompt); fail_expect makes it fail.
    std::uint32_t expected_id = 0, expected_status = 0;
    bool fail_expect = false, fail_restore = false;
    std::optional<gs::ImportReport> import_done;
    std::string abort_why;
    void import_expect(std::uint32_t id, std::span<const std::uint32_t> t) {
        log.push_back("expect " + std::to_string(id) + " " + std::to_string(t.size()));
        expected_id = id;
        import_done.reset();
        if (fail_expect) {
            expected_status = static_cast<std::uint32_t>(gs::ImportStatus::failed);
            return;
        }
        stage(id, t.first(t.size() - 1));
        expected_status = static_cast<std::uint32_t>(gs::ImportStatus::ready);
    }
    const std::atomic<bool>* hold = nullptr;  // while true, a staged import reports listening (the RTX still sending)
    gs::ImportReport import_report() const {
        if (import_done && import_done->id == expected_id) return *import_done;
        if (hold && hold->load() && expected_status == static_cast<std::uint32_t>(gs::ImportStatus::ready))
            return {expected_id, static_cast<std::uint32_t>(gs::ImportStatus::listening), 0};
        return {expected_id, expected_status, 0};
    }
    // state_digest (import gate): the slot's state words at a frontier (no state change)
    std::vector<std::uint32_t> state_digest(std::uint32_t s, std::uint64_t f) const {
        if (frontier.at(s) != f) throw std::runtime_error("fake state_digest: not at the slot frontier");
        return {static_cast<std::uint32_t>(h[s]), static_cast<std::uint32_t>(h[s] >> 32), static_cast<std::uint32_t>(f), 0U};
    }
    // import_abort (replay, after import_state threw): the slot reset, the failure reported.
    void import_abort(std::uint32_t s, std::uint32_t id, const std::string& why) {
        abort_why = why;
        seed.reset();
        reset(s);
        log.back() += " (import abort)";
        import_done = gs::ImportReport{id, static_cast<std::uint32_t>(gs::ImportStatus::failed), 0};
    }
    // 'DSA once': both fake ranks share `pipe`; rank 0 stages the full synthetic payload, rank 1 its KDA part only.
    Pipe* pipe = nullptr;
    unsigned rank = 0;
    bool relay_fail = false;  // this rank cannot take part in the relay (nothing usable staged)
    gs::RelayStats last_relay{};
    std::vector<std::uint8_t> relayed;  // rank 1: the merged payload it restored
    std::uint8_t binding_salt = 0;  // != 0: this rank staged a hand-off with another binding (both must abort)
    unsigned suffix_collectives = 0;  // pair exchanges made by the suffix (the engine's cross-rank argmax, modelled)
    // As EngineOps: gs::import_transaction (prepare vote with the binding, relay, restore, restored vote, then the
    // suffix, which pairs up with the peer like the engine's argmax collective). Votes ride `pipe` (none: one rank).
    std::uint32_t import_state(std::uint32_t s, std::uint32_t id, std::span<const std::uint32_t> t, bool mtp, bool dsa) {
        plain_call();
        bool prepared = true;
        std::string why;
        gs::Binding b{};
        const auto it = staged.find(id);
        try {
            if (!ready.at(s) || frontier[s] != 0) throw std::runtime_error("fake import without reset");
            if (it == staged.end()) throw std::runtime_error("fake import: nothing staged under this id");
            if (it->second.p + 1 != t.size()) throw std::runtime_error("fake import: staged frontier != prompt - 1");
            for (std::size_t k = 0; k < 8; ++k) b[k] = static_cast<std::uint8_t>(it->second.h >> (8 * k));
            b[31] = binding_salt;
        } catch (const std::exception& e) {
            prepared = false;
            why = e.what();
        }
        std::function<void(std::span<const std::byte>, std::span<std::byte>, std::uint64_t)> xchg;
        if (pipe) xchg = [this](std::span<const std::byte> a, std::span<std::byte> r, std::uint64_t) { pipe->xchg(rank, a, r); };
        return gs::import_transaction(
            prepared, why, b, t.size() >= 1 ? t.size() - 1 : 0, xchg,
            [&] {
                if (!dsa) return;
                if (!pipe) throw std::runtime_error("fake relay: no pipe");
                const auto mine = rank == 0 ? relay_full(0) : relay_part(false, 1);
                relayed = gs::dsa_relay(rank, !relay_fail, mine, kRelayLayout, 64, xchg, &last_relay);
                if (rank == 1 && relayed != relay_full(1)) throw std::runtime_error("fake relay: rank 1 payload differs from its full state");
            },
            [&] {
                if (fail_restore) {  // a half-written slot, as a restore_state that throws midway
                    h[s] = 12345;
                    frontier[s] = 7;
                    throw std::runtime_error("fake restore failure");
                }
                h[s] = fold(it->second.h, t.back());
                frontier[s] = t.size();
            },
            [&]() -> std::uint32_t {
                if (pipe) {  // the engine's argmax over both ranks: a rank alone here would wait for its peer forever
                    const std::uint32_t mine = tok(h[s]);
                    std::uint32_t peer = 0;
                    pipe->xchg(rank, std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(&peer, 1)));
                    ++suffix_collectives;
                    if (peer != mine) throw std::runtime_error("fake suffix: the ranks' first tokens differ");
                }
                ++calls;
                log.push_back("import " + std::to_string(s) + " " + std::to_string(t.size()) + (mtp ? " mtp" : "") + (dsa ? " dsa" : ""));
                spec[s] = {};
                anchor[s] = tok(h[s]);
                if (mtp) {
                    seed = Seed{s, tok(h[s]), t.size()};
                    seed_spec(s);  // batched self-speculation starts from the import's seed, as after the slot's own prefill
                }
                import_done = gs::ImportReport{id, static_cast<std::uint32_t>(gs::ImportStatus::restored), tok(h[s])};
                return tok(h[s]);
            });
    }
    std::vector<std::uint32_t> dropped;  // import_drop ids (fallback / cancel)
    void import_drop(std::uint32_t id) {
        dropped.push_back(id);
        staged.erase(id);
        if (expected_id == id) expected_status = static_cast<std::uint32_t>(gs::ImportStatus::failed);
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
                 unsigned delay_us = 0, bool skip_basic = false, bool g2 = false, std::uint64_t* multi = nullptr,
                 std::uint64_t import_min = 0, bool fail_import = false, bool fail_restore = false, std::uint8_t binding_salt = 0, bool prefix = false, bool refuse_prefix = false, std::uint64_t prefix_budget = 1ULL << 20) {
    PairRun out;
    Loopback lb;
    FakeEng e0(slots), e1(slots);
    Pipe pipe;  // the import commit's votes and the suffix collective pair up through it
    e0.pipe = e1.pipe = &pipe;
    e1.rank = 1;
    e1.corrupt_call = corrupt;
    e0.delay_us = e1.delay_us = delay_us;
    e1.fail_expect = fail_import;  // the follower's staging fails -> every import falls back to its own prefill
    e1.fail_restore = fail_restore;  // the follower's restore fails after both staged -> reset + fallback, still exact
    e1.binding_salt = binding_salt;  // the follower staged another hand-off -> the prepare vote aborts both
    gs::Channel c0(lb.side(0), 0), c1(lb.side(1), 1);
    gs::Hello h0 = hello_for(slots, 4096, 2);
    if (prefix) h0.prefix_cache_bytes = h1.prefix_cache_bytes = prefix_budget;
    e1.refuse_prefix = refuse_prefix;
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
    lc.prefix_cache_bytes = h0.prefix_cache_bytes;
    lc.slots = slots;
    lc.capacity = 4096;
    lc.batch = batch;
    lc.heartbeat_seconds = 0.01;
    lc.mtp = mtp;
    lc.mtp_cycles = 2;
    lc.prefill_chunk = chunk;
    lc.g2 = g2;
    if (import_min != 0) {  // route on: the fake ranks stage at import_expect, the 'RTX' always accepts
        lc.rtx_prefill = [](std::span<const std::uint32_t>, std::uint32_t) { return true; };
        lc.import_min_tokens = import_min;
        lc.import_timeout_s = 5.0;
    }
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
                if (prefix) out.result.exact = gs::run_prefix_selftest(queue, prompts(n), tokens, gs::default_stop_ids(), {17, 23},
                    [&lead] { lead.clear_prefix_cache(); }, [&lead] { return lead.stats().prefix_hits; },
                    [](const std::string& line) { std::printf("%s\n", line.c_str()); });
                else out.result = gs::run_selftest(queue, prompts(n), tokens, gs::default_stop_ids(), slots, [](const std::string& l) {
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
    // ---- prefix key semantics, byte LRU, full-token collision checks and admission parsing
    {
        auto entry = [](std::uint64_t id, std::vector<std::uint32_t> ids, bool mtp = false, std::uint64_t identity = 7) {
            gs::PrefixCache::Entry e;
            e.id = id; e.ids = std::move(ids); e.hash = gs::prefix_hash(e.ids);
            e.mtp = mtp; e.identity = identity; e.frontier = e.ids.size() - (mtp ? 1U : 0U);
            e.state.assign(16, 42);
            return e;
        };
        gs::PrefixCache c(4096);
        CHECK(c.prepare(entry(1, {1, 2}))); c.commit(true);
        CHECK(c.prepare(entry(2, {1, 2, 3}))); c.commit(true);
        const std::vector<std::uint32_t> p{1, 2, 3, 4};
        CHECK(c.find(p, false, 7)->id == 2);
        CHECK(c.find(std::span(p).first(3), false, 7)->id == 1);  // exact key forbidden
        CHECK(!c.find(std::span(p).first(2), false, 7));
        CHECK(!c.find(p, true, 7) && !c.find(p, false, 8));
        CHECK(c.prepare(entry(3, {1, 2, 3}, true))); c.commit(true);
        CHECK(c.find(p, true, 7)->frontier == 2);  // A + y0 key; restore at |A|
        auto collision = entry(4, {9, 9, 9});
        collision.hash = gs::prefix_hash(std::span(p).first(3));
        CHECK(c.prepare(std::move(collision))); c.commit(true);
        CHECK(c.find(p, false, 7)->id == 2);  // matching hash alone is insufficient
        const auto e = entry(5, {2, 3});
        gs::PrefixCache lru(e.bytes() * 2);
        CHECK(lru.prepare(entry(1, {1, 2}))); lru.commit(true);
        CHECK(lru.prepare(entry(2, {2, 3}))); lru.commit(true);
        lru.touch(1);
        CHECK(lru.prepare(entry(3, {3, 4}))); lru.commit(true);
        CHECK(lru.get(1) && !lru.get(2) && lru.get(3) && lru.evictions == 1);
        auto huge = entry(4, {4, 5}); huge.state.resize(4096);
        CHECK(!lru.prepare(std::move(huge)) && lru.bytes() == e.bytes() * 2);
        CHECK(lru.prepare(entry(5, {5, 6}))); lru.commit(false);
        CHECK(!lru.get(5) && lru.get(1));
        CHECK(lru.prepare(entry(6, {1, 2}))); lru.commit(true);
        CHECK(!lru.get(1) && lru.get(6) && lru.get(3) && lru.bytes() == e.bytes() * 2);
        CHECK(gs::prefix_budget_bytes("0") == 0 && gs::prefix_budget_bytes("0.5") == (1ULL << 29));
        for (const auto* v : {"-1", "nan", "inf", "1x", "17179869184", "1e-20"})
            CHECK(throws_with([&] { (void)gs::prefix_budget_bytes(v); }, "--prefix-cache-gib"));
        for (const auto* v : {"", "MemAvailable: -1 kB", "MemAvailable: 16777215 kB", "MemAvailable: bad kB"}) {
            std::istringstream in(v); CHECK(!gs::prefix_memory_available(in));
        }
        std::istringstream in("MemFree: 0 kB\nMemAvailable: 16777216 kB\n");
        CHECK(gs::prefix_memory_available(in));
        auto hello = hello_for(2, 4096, 2), peer = hello;
        peer.prefix_cache_bytes = 1;
        CHECK(gs::hello_mismatch(hello, peer).find("prefix_cache_bytes") != std::string::npos);
    }
    {
        // A budget holding only one snapshot forces the lead to publish explicit eviction victims.
        const auto r = run_pair(3, 4, true, 16, 0, hello_for(4, 4096, 2), false, 7, 0, 0, false, false,
                                nullptr, 0, false, false, 0, true, false, 256);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.stats.prefix_evictions > 0 && r.stats.prefix_bytes <= 256 && r.log0 == r.log1);
        // Native prefix caching must not replace the optional imported numerical identity.
        const auto imported = run_pair(2, 4, true, 16, 0, hello_for(4, 4096, 2), false, 7, 0, 0, false, false,
                                       nullptr, 1, false, false, 0, true);
        CHECK(imported.lead_error.empty() && imported.follow_error.empty() && imported.driver_error.empty());
        CHECK(imported.stats.imports > 0 && imported.stats.prefix_hits == 0 && imported.stats.prefix_stores == 0);
    }
    // Missing peer key and a local restore exception must be observed on BOTH ranks before suffix collectives.
    for (const bool missing : {false, true}) {
        Loopback lb;
        gs::Channel c0(lb.side(0), 0), c1(lb.side(1), 1);
        FakeEng e0(1), e1(1);
        gs::PrefixCache k0(4096), k1(4096);
        gs::PrefixCache::Entry e;
        e.id = 1; e.identity = 7; e.frontier = 2; e.ids = {11, 22}; e.hash = gs::prefix_hash(e.ids);
        e0.reset(0); (void)e0.prefill(0, e.ids, 0);
        e.state = e0.state_bytes_upto(0, 2);
        CHECK(k0.prepare(e)); k0.commit(true);
        if (!missing) { CHECK(k1.prepare(e)); k1.commit(true); e1.fail_prefix_restore = true; }
        gs::Control c;
        c.cmd = static_cast<std::uint32_t>(gs::Cmd::prefix_restore);
        c.prefix_id = 1; c.prefix_identity = 7; c.ppos = 2;
        bool failed0 = false, failed1 = false;
        std::thread peer([&] {
            gs::Control reply;
            const auto cmd = c1.exchange(reply);
            std::vector<std::uint32_t> p;
            failed1 = throws_with([&] { (void)gs::replay(e1, c1, cmd, p, false, 1, 64, &k1); }, "paired restore failed");
        });
        (void)c0.exchange(c);
        std::vector<std::uint32_t> p;
        failed0 = throws_with([&] { (void)gs::replay(e0, c0, c, p, true, 1, 64, &k0); }, "paired restore failed");
        peer.join();
        CHECK(failed0 && failed1 && e1.prefills == 0);
    }
    // PC2 through the actual lead/follower replay, plain and MTP/G2; independent rank-1 refusal must discard both.
    for (const bool mtp : {false, true}) {
        auto r = run_pair(3, 4, true, 16, 0, hello_for(4, 4096, 2), mtp, 7, 0, 0, false, mtp,
                          nullptr, 0, false, false, 0, true);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact && r.stats.prefix_hits >= 3 && r.stats.prefix_stores > 0);
        CHECK(r.log0 == r.log1);
        auto f = run_pair(2, 4, true, 16, 0, hello_for(4, 4096, 2), mtp, 7, 0, 0, false, mtp,
                          nullptr, 0, false, false, 0, true, true);
        CHECK(f.lead_error.empty() && f.follow_error.empty() && f.driver_error.empty());
        CHECK(!f.result.exact && f.stats.prefix_hits == 0 && f.stats.prefix_stores == 0);
    }

    // ---- import commit votes: both ranks fail together on any rank's failure, binding or frontier mismatch
    {
        std::printf("import commit votes:\n");
        auto vote_pair = [](bool ok0, bool ok1, std::uint8_t salt1, std::uint64_t f1) {
            Pipe pipe;
            gs::Binding b0{}, b1{};
            b0[0] = b1[0] = 42;
            b1[1] = salt1;
            std::string e[2];
            std::thread t([&] {
                try {
                    gs::import_vote("prepare", ok1, "local", b1, f1, [&](std::span<const std::byte> a, std::span<std::byte> r, std::uint64_t) { pipe.xchg(1, a, r); }, 1);
                } catch (const std::exception& x) { e[1] = x.what(); }
            });
            try {
                gs::import_vote("prepare", ok0, "local", b0, 100, [&](std::span<const std::byte> a, std::span<std::byte> r, std::uint64_t) { pipe.xchg(0, a, r); }, 1);
            } catch (const std::exception& x) { e[0] = x.what(); }
            t.join();
            return std::pair<std::string, std::string>{e[0], e[1]};
        };
        {
            const auto [a, b] = vote_pair(true, true, 0, 100);
            CHECK(a.empty() && b.empty());
        }
        {
            const auto [a, b] = vote_pair(true, false, 0, 100);
            CHECK(a.find("peer rank failed") != std::string::npos && b.find("this rank: local") != std::string::npos);
        }
        {
            const auto [a, b] = vote_pair(true, true, 7, 100);  // different binding: both refuse
            CHECK(a.find("different hand-offs") != std::string::npos && b.find("different hand-offs") != std::string::npos);
        }
        {
            const auto [a, b] = vote_pair(true, true, 0, 99);  // different frontier: both refuse
            CHECK(a.find("different hand-offs") != std::string::npos && b.find("different hand-offs") != std::string::npos);
        }
        // the transaction: a one-rank restore failure aborts both before the suffix (no rank alone in a collective);
        // a failure inside the suffix is ImportFatal (the engine's fatal contract), never an import failure.
        auto txn = [](bool fail_restore1, bool fail_suffix0) {
            Pipe pipe;
            const gs::Binding b{};
            std::string e[2];
            bool fatal[2] = {false, false};
            unsigned suffix[2] = {0, 0};
            auto run = [&](unsigned r) {
                try {
                    (void)gs::import_transaction(
                        true, "", b, 10, [&](std::span<const std::byte> a, std::span<std::byte> x, std::uint64_t) { pipe.xchg(r, a, x); },
                        [] {}, [&] { if (r == 1 && fail_restore1) throw std::runtime_error("restore broke"); },
                        [&]() -> std::uint32_t {
                            ++suffix[r];
                            if (r == 0 && fail_suffix0) throw std::runtime_error("collective broke");
                            return 5;
                        });
                } catch (const gs::ImportFatal& x) {
                    fatal[r] = true;
                    e[r] = x.what();
                } catch (const std::exception& x) {
                    e[r] = x.what();
                }
            };
            std::thread t([&] { run(1); });
            run(0);
            t.join();
            return std::make_tuple(e[0], e[1], fatal[0], fatal[1], suffix[0], suffix[1]);
        };
        {
            const auto [a, b, f0, f1, s0, s1] = txn(true, false);
            CHECK(a.find("restore: the peer rank failed") != std::string::npos && b.find("restore: this rank: restore broke") != std::string::npos);
            CHECK(!f0 && !f1 && s0 == 0 && s1 == 0);
            std::printf("  rank 1 restore fails: both abort before the suffix (rank 0: %s)\n", a.c_str());
        }
        {
            const auto [a, b, f0, f1, s0, s1] = txn(false, true);
            CHECK(f0 && !f1 && a.find("fatal") != std::string::npos && b.empty() && s0 == 1 && s1 == 1);
        }
        {
            const auto [a, b, f0, f1, s0, s1] = txn(false, false);
            CHECK(a.empty() && b.empty() && s0 == 1 && s1 == 1);
        }
    }
#ifndef _WIN32
    // ---- import receiver: a peer trickling bytes is cut by cancellation and by the aggregate deadline
    {
        std::printf("import receiver, slow-drip peer:\n");
        namespace gi = strix::glm::import;
        const gi::Bytes key(32, std::byte{7});
        gi::Begin expect;
        expect.frontier = 1;
        expect.total = 2;
        for (const int mode : {0, 1}) {  // 0: cancelled after 400 ms, 1: aggregate deadline 1 s
            const std::string ep = mode == 0 ? "127.0.0.1:47991" : "127.0.0.1:47992";
            std::atomic<bool> cancel{false}, bound{false}, stop_drip{false};
            std::string err;
            double secs = 0;
            std::thread rx([&] {
                const auto t0 = std::chrono::steady_clock::now();
                try {
                    (void)gi::receive(ep, key, expect, &cancel, [&] { bound.store(true); }, mode == 0 ? 900U : 1U);
                } catch (const std::exception& e) {
                    err = e.what();
                }
                secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            });
            for (int k = 0; k < 200 && !bound.load(); ++k) std::this_thread::sleep_for(std::chrono::milliseconds(10));
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_port = htons(static_cast<std::uint16_t>(mode == 0 ? 47991 : 47992));
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            const bool connected = bound.load() && ::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0;
            CHECK(connected);
            std::thread drip([&] {  // reads the nonce, then one byte of the length field every 150 ms
                char nonce[32];
                std::size_t got = 0;
                while (connected && got < sizeof nonce) {
                    const auto n = ::recv(fd, nonce + got, sizeof nonce - got, 0);
                    if (n <= 0) break;
                    got += static_cast<std::size_t>(n);
                }
                const char one = 0;
                for (int k = 0; k < 40 && !stop_drip.load(); ++k) {
                    if (k < 3) (void)::send(fd, &one, 1, MSG_NOSIGNAL);  // 3 of the 4 length bytes, then nothing more
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                }
            });
            if (mode == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(400));
                cancel.store(true);
            }
            rx.join();
            stop_drip.store(true);
            drip.join();
            ::close(fd);
            CHECK(mode == 0 ? err.find("cancelled") != std::string::npos : err.find("deadline") != std::string::npos);
            CHECK(secs < 3.0);
            std::printf("  %s: '%s' after %.2f s\n", mode == 0 ? "cancel" : "deadline 1 s", err.c_str(), secs);
        }
    }
#endif
    // ---- 'DSA once' relay protocol: rank 0's DSA components reach rank 1 in chunks, merged in layout order
    {
        std::printf("dsa relay protocol:\n");
        const auto run = [](bool ok0, bool ok1, std::size_t chunk, std::vector<std::uint8_t> mine0) {
            Pipe pipe;
            std::vector<std::uint8_t> out[2];
            std::string err[2];
            gs::RelayStats st[2];
            const std::vector<std::uint8_t> mine[2] = {std::move(mine0), relay_part(false, 1)};
            std::thread t1([&] {
                try {
                    out[1] = gs::dsa_relay(1, ok1, mine[1], kRelayLayout, chunk,
                                           [&](std::span<const std::byte> a, std::span<std::byte> b, std::uint64_t) { pipe.xchg(1, a, b); }, &st[1]);
                } catch (const std::exception& e) {
                    err[1] = e.what();
                }
            });
            try {
                out[0] = gs::dsa_relay(0, ok0, mine[0], kRelayLayout, chunk,
                                       [&](std::span<const std::byte> a, std::span<std::byte> b, std::uint64_t) { pipe.xchg(0, a, b); }, &st[0]);
            } catch (const std::exception& e) {
                err[0] = e.what();
            }
            t1.join();
            return std::make_tuple(out[0], out[1], err[0], err[1], st[1]);
        };
        {  // chunks smaller than the components, a partial last chunk: rank 1 ends with its KDA + rank 0's DSA, in order
            const auto [o0, o1, e0, e1, s1] = run(true, true, 64, relay_full(0));
            CHECK(e0.empty() && e1.empty() && o0.empty() && o1 == relay_full(1));
            CHECK(s1.bytes == 407 && s1.chunks == 7);
            CHECK(gs::gather_dsa(relay_full(1), kRelayLayout) == relay_part(true, 0));
            std::printf("  407 DSA bytes in %llu chunks of <= 64: rank 1 == its full state: %s\n",
                        static_cast<unsigned long long>(s1.chunks), o1 == relay_full(1) ? "yes" : "no");
        }
        {  // one chunk
            const auto [o0, o1, e0, e1, s1] = run(true, true, 1U << 20, relay_full(0));
            CHECK(e0.empty() && e1.empty() && o1 == relay_full(1) && s1.chunks == 1);
        }
        {  // rank 1 has nothing usable: both fail at the header (neither waits alone)
            const auto [o0, o1, e0, e1, s1] = run(true, false, 64, relay_full(0));
            CHECK(!e0.empty() && !e1.empty() && o1.empty());
            std::printf("  rank 1 not ready: rank 0 '%s' / rank 1 '%s'\n", e0.c_str(), e1.c_str());
        }
        {  // rank 0's staged payload has the wrong size: both fail
            auto bad = relay_full(0);
            bad.pop_back();
            const auto [o0, o1, e0, e1, s1] = run(true, true, 64, bad);
            CHECK(!e0.empty() && !e1.empty());
        }
    }
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
        // RTX import settings: receiver on / 'DSA once', wire version and the attested model SHA must agree on every rank
        {
            auto i0 = a;
            i0.import_flags = 3;
            i0.import_wire = 2;
            i0.import_model_hash = gs::fnv1a64(std::string("ab12"));
            auto i1 = i0;
            CHECK(gs::hello_mismatch(i0, i1).empty());
            i1.import_flags = 1;  // the peer has no 'DSA once'
            i1.import_wire = 1;
            CHECK(gs::hello_mismatch(i0, i1).find("import_flags") != std::string::npos &&
                  gs::hello_mismatch(i0, i1).find("import_wire") != std::string::npos);
            i1 = i0;
            i1.import_model_hash = gs::fnv1a64(std::string("cd34"));  // another attested model
            CHECK(gs::hello_mismatch(i0, i1).find("import_model_hash") != std::string::npos);
            i1 = a;  // the peer has no import receiver at all
            CHECK(gs::hello_mismatch(i0, i1).find("import_flags") != std::string::npos);
            // the fields travel in the control block: a mismatch over the loopback refuses on both ranks at hello
            Loopback lb;
            gs::Channel c0(lb.side(0), 0), c1(lb.side(1), 1);
            std::string e1;
            std::thread t([&] {
                try {
                    c1.hello(i1);
                } catch (const std::exception& e) {
                    e1 = e.what();
                }
            });
            std::string e0;
            try {
                c0.hello(i0);
            } catch (const std::exception& e) {
                e0 = e.what();
            }
            t.join();
            CHECK(e0.find("import_flags") != std::string::npos && e1.find("import_flags") != std::string::npos);
            std::printf("hello with different import settings: refused on both ranks (%s)\n", e0.c_str());
            // the attested model SHA is hashed canonically: the same digest in upper / lower case agrees
            const std::string lower = "d4b733a5d05d587c65a5a0f3d73b65c6bd8f62cb6fc2b6e48f6646826e3d1d3a";
            std::string upper = lower;
            for (auto& ch : upper) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            CHECK(gs::attested_model_hash(lower) == gs::attested_model_hash(upper) && upper != lower);
            CHECK(gs::attested_model_hash(lower) != gs::attested_model_hash(std::string(64, '0')));
            CHECK(throws_with([] { (void)gs::attested_model_hash("abc"); }, "64 hex digits"));
            CHECK(throws_with([&] { (void)gs::attested_model_hash(std::string(63, 'a') + "g"); }, "not a hex digit"));
        }
        // a peer with the previous (v5) block layout is refused at the version check, before any later field is read
        {
            gs::Control c;
            c.cmd = static_cast<std::uint32_t>(gs::Cmd::heartbeat);
            auto blk = gs::encode(c);
            const std::uint32_t v5 = 5;
            std::memcpy(blk.data() + offsetof(gs::Control, version), &v5, sizeof v5);
            CHECK(throws_with([&] { (void)gs::decode(blk); }, "protocol version 5 from the peer, expected 7"));
            CHECK(gs::decode(gs::encode(c)).version == gs::kControlVersion);
            std::printf("v5-layout peer block: refused at the version check\n");
        }
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
    // ---- import_state: replay restores the staged prefix and runs the last token; == the slot's own prefill
    {
        const std::vector<std::uint32_t> prompt{11, 22, 33, 44, 55};
        for (const bool mtp : {false, true}) {
            FakeEng own(2), imp(2);
            own.reset(1);
            const auto y_own = mtp ? own.prefill_mtp(1, prompt, 0, true) : own.prefill(1, prompt, 0);
            imp.stage(7, std::span<const std::uint32_t>(prompt).first(prompt.size() - 1));
            imp.expected_id = 7;
            gs::Channel solo({}, 0);
            solo.hello(hello_for(2, 64, 1));
            gs::Control c;
            c.cmd = static_cast<std::uint32_t>(gs::Cmd::import_state);
            c.pslot = 1;
            c.resets = 1U << 1;
            c.plen = static_cast<std::uint32_t>(prompt.size());
            c.import_id = 7;
            c.pflags = mtp ? gs::kPrefillMtp : 0U;
            CHECK(gs::decode(gs::encode(c)).import_id == 7);
            auto p = prompt;
            const auto out = gs::replay(imp, solo, c, p, true, 2, 64);
            CHECK(out.size() == 1 && out[0] == y_own && imp.h[1] == own.h[1] && imp.frontier[1] == own.frontier[1]);
            CHECK(!mtp || (imp.seed && imp.seed->y0 == y_own && imp.seed->p == prompt.size()));
            CHECK(imp.import_report().status == static_cast<std::uint32_t>(gs::ImportStatus::restored) && imp.import_report().token == y_own);
            c.import_id = 8;  // nothing staged under it: the rank's failure is local (slot reset, kImportFailed, reported)
            auto q = prompt;
            imp.expected_id = 8;
            const auto bad = gs::replay(imp, solo, c, q, true, 2, 64);
            CHECK(bad.size() == 1 && bad[0] == gs::kImportFailed && imp.abort_why.find("nothing staged") != std::string::npos);
            CHECK(imp.frontier[1] == 0 && imp.ready[1] && !imp.seed);
            CHECK(imp.import_report().status == static_cast<std::uint32_t>(gs::ImportStatus::failed));
            c.import_id = 7;
            c.resets = 0;  // the same command must reset the slot
            CHECK(throws_with([&] { (void)gs::replay(imp, solo, c, q, true, 2, 64); }, "without its slot reset"));
        }
        FakeEng short_one(2);
        short_one.stage(3, std::vector<std::uint32_t>{1, 2});
        gs::Control c;
        c.cmd = static_cast<std::uint32_t>(gs::Cmd::import_state);
        c.pslot = 0;
        c.resets = 1;
        c.plen = 4;  // staged frontier 2 != 4 - 1
        c.import_id = 3;
        gs::Channel solo({}, 0);
        solo.hello(hello_for(2, 64, 1));
        std::vector<std::uint32_t> p{1, 2, 3, 4};
        CHECK(gs::replay(short_one, solo, c, p, true, 2, 64).at(0) == gs::kImportFailed && short_one.abort_why.find("staged frontier") != std::string::npos);
    }
    // ---- ImportStager : receives off the caller's thread; ready / failed per id; payload only when ready
    {
        std::mutex gm;
        std::condition_variable gcv;
        bool go = false;
        gs::ImportStager st([&](std::uint32_t id, std::span<const std::uint32_t> t, const std::atomic<bool>& cancel,
                                const std::function<void()>& listening) {
            if (id == 3) {  // never connected: waits until cancelled
                listening();
                while (!cancel.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                throw std::runtime_error("import cancelled");
            }
            std::unique_lock<std::mutex> l(gm);
            if (id == 1) listening();
            gcv.wait(l, [&] { return go; });
            if (id == 2) throw std::runtime_error("model hash mismatch");
            return std::vector<std::uint8_t>(t.size() * 3, static_cast<std::uint8_t>(id));
        });
        st.expect(1, {5, 6, 7});
        st.expect(2, {5, 6});
        for (int i = 0; i < 2000 && st.status(1) != gs::ImportStatus::listening; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(st.status(1) == gs::ImportStatus::listening && st.status(2) == gs::ImportStatus::pending && st.status(9) == gs::ImportStatus::none);
        CHECK(throws_with([&] { (void)st.payload(1); }, "nothing staged"));
        CHECK(throws_with([&] { st.expect(1, {1, 2}); }, "already expected"));
        {
            std::lock_guard<std::mutex> l(gm);
            go = true;
        }
        gcv.notify_all();
        for (int i = 0; i < 2000 && (st.status(1) == gs::ImportStatus::pending || st.status(2) == gs::ImportStatus::pending); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(st.status(1) == gs::ImportStatus::ready && st.payload(1).size() == 9 && st.payload(1)[0] == 1);
        CHECK(st.status(2) == gs::ImportStatus::failed && st.error(2).find("model hash") != std::string::npos);
        st.drop(1);
        CHECK(st.status(1) == gs::ImportStatus::none);
        // dropping a receive that never connects returns at once (cancelled, reaped later); its id never reappears
        st.expect(3, {1, 2, 3});
        for (int i = 0; i < 2000 && st.status(3) != gs::ImportStatus::listening; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const auto t0 = std::chrono::steady_clock::now();
        st.drop(3);
        CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(50) && st.status(3) == gs::ImportStatus::none);
        for (int i = 0; i < 2000 && st.retired() != 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            st.drop(99);  // reaps
        }
        CHECK(st.retired() == 0 && st.status(3) == gs::ImportStatus::none);
    }
    // ---- drop never stalls the replay; a new import is refused (falls back) while a dropped receive still closes the
    // endpoint, and binds again once that receive has ended
    {
        std::atomic<int> started{0};
        gs::ImportStager st([&](std::uint32_t, std::span<const std::uint32_t>, const std::atomic<bool>& cancel,
                                const std::function<void()>& listening) -> std::vector<std::uint8_t> {
            ++started;
            listening();
            while (!cancel.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::this_thread::sleep_for(std::chrono::milliseconds(300));  // a slow close of the endpoint
            throw std::runtime_error("import cancelled");
        });
        st.expect(1, {1, 2, 3});
        for (int i = 0; i < 2000 && st.status(1) != gs::ImportStatus::listening; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const auto t0 = std::chrono::steady_clock::now();
        st.drop(1);
        st.expect(2, {1, 2, 3});  // immediately: the dropped receive still holds the endpoint
        const auto dt = std::chrono::steady_clock::now() - t0;
        CHECK(dt < std::chrono::milliseconds(50));
        CHECK(st.status(2) == gs::ImportStatus::failed && st.error(2).find("still closing") != std::string::npos && started.load() == 1);
        st.drop(2);
        for (int i = 0; i < 3000 && st.retired() != 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            st.drop(99);  // reaps
        }
        CHECK(st.retired() == 0);
        st.expect(3, {1, 2, 3});  // the endpoint is free again: a new receive starts
        for (int i = 0; i < 2000 && st.status(3) != gs::ImportStatus::listening; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(st.status(3) == gs::ImportStatus::listening && started.load() == 2);
        std::printf("drop + immediate expect: refused in %.1f ms (no stall), next expect after close: %s\n",
                    std::chrono::duration<double, std::milli>(dt).count(), st.status(3) == gs::ImportStatus::listening ? "listening" : "NO");
        st.drop(3);
    }
    // ---- import route over the loopback: imported slots decode exactly as prefilled ones; failures fall back
    for (const bool mtp : {false, true}) {
        std::printf("import route, 6 prompts, 4 slots%s:\n", mtp ? ", MTP" : "");
        const auto r = run_pair(6, 4, true, 40, 0, hello_for(8, 4096, 2), mtp, 0, 0, 0, false, false, nullptr, 1);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact && r.rows0 == r.rows1);
        CHECK(r.stats.prefills > 0 && r.stats.imports == r.stats.prefills && r.stats.import_fallbacks == 0);  // every prompt imported
        const auto f = run_pair(6, 4, true, 40, 0, hello_for(8, 4096, 2), mtp, 0, 0, 0, false, false, nullptr, 1, true);
        CHECK(f.lead_error.empty() && f.follow_error.empty() && f.driver_error.empty());
        CHECK(f.result.exact && f.stats.prefills > 0 && f.stats.imports == 0 && f.stats.import_fallbacks == f.stats.prefills);
        // both ranks staged, the follower's restore fails: its slot is reset, both fall back, the server keeps serving
        const auto g = run_pair(6, 4, true, 40, 0, hello_for(8, 4096, 2), mtp, 128, 0, 0, false, false, nullptr, 1, false, true);
        CHECK(g.lead_error.empty() && g.follow_error.empty() && g.driver_error.empty());
        CHECK(g.result.exact && g.stats.prefills > 0 && g.stats.imports == 0 && g.stats.import_fallbacks == g.stats.prefills);
        bool aborted = false, aborted0 = false;
        for (const auto& line : g.log1) aborted = aborted || line.find("(import abort)") != std::string::npos;
        for (const auto& line : g.log0) aborted0 = aborted0 || line.find("(import abort)") != std::string::npos;
        CHECK(aborted && aborted0);  // the restored vote: rank 0 restored but aborts too, before any suffix collective
        // the ranks staged different hand-offs (binding): the prepare vote aborts both, every prompt prefills itself
        const auto m = run_pair(6, 4, true, 40, 0, hello_for(8, 4096, 2), mtp, 128, 0, 0, false, false, nullptr, 1, false, false, 1);
        CHECK(m.lead_error.empty() && m.follow_error.empty() && m.driver_error.empty());
        CHECK(m.result.exact && m.stats.prefills > 0 && m.stats.imports == 0 && m.stats.import_fallbacks == m.stats.prefills);
    }
    // ---- import route beside the chunked prefill: long prompts import (no chunk), short ones prefill in chunks
    for (const bool mtp : {false, true}) {
        std::printf("import route + prefill interleave (chunk 128, long prompts import)%s:\n", mtp ? ", MTP" : "");
        const auto r = run_pair(4, 4, true, 64, 0, hello_for(8, 4096, 2), mtp, 128, 3, 0, false, false, nullptr, 500);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact && r.log0 == r.log1);
        CHECK(r.stats.imports > 0 && r.stats.import_fallbacks == 0 && r.stats.imports < r.stats.prefills);
        bool long_chunked = false, imported = false;
        for (const auto& line : r.log0) {
            long_chunked = long_chunked || line.find(" @128 +128") != std::string::npos;
            imported = imported || line.rfind("import ", 0) == 0;
        }
        CHECK(imported && !long_chunked);  // the long prompts skipped the chunked prefill
        const auto f = run_pair(4, 4, true, 64, 0, hello_for(8, 4096, 2), mtp, 128, 3, 0, false, false, nullptr, 500, true);
        CHECK(f.lead_error.empty() && f.follow_error.empty() && f.driver_error.empty());
        CHECK(f.result.exact && f.log0 == f.log1 && f.stats.imports == 0 && f.stats.import_fallbacks > 0);
        // a failed import prefills on the pair: in chunks while others decode, whole when it is alone (adaptive chunking)
        bool fell_back_prefilled = false;
        for (const auto& line : f.log0) {
            if (line.find(" @128 +128") != std::string::npos) fell_back_prefilled = true;
            const auto at = line.find(" @0 +");
            if (line.rfind("prefill ", 0) == 0 && at != std::string::npos && std::stoul(line.substr(at + 5)) >= 500) fell_back_prefilled = true;
        }
        CHECK(fell_back_prefilled);
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
    // ---- per-request route receipts: an imported prompt reports imported_tokens + route rtx; a refused one, its fallback
    for (const int variant : {0, 1, 2, 3}) {  // imported / staging fails / imported 'DSA once' / 'DSA once', rank 1 cannot relay
        const bool fail = variant == 1, dsa = variant >= 2, relay_fail = variant == 3;
        Loopback lb;
        FakeEng e(4), e1(4);
        Pipe pipe;
        e.stops = e1.stops = false;
        e.fail_expect = fail;
        e.pipe = e1.pipe = &pipe;
        e1.rank = 1;
        e1.relay_fail = relay_fail;
        gs::Channel ch(lb.side(0), 0), c1(lb.side(1), 1);
        std::thread follower([&] { gs::follow(e1, c1, hello_for(4, 4096, 2), 4, 4096); });
        ch.hello(hello_for(4, 4096, 2));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 4;
        lc.capacity = 4096;
        lc.heartbeat_seconds = 0.01;
        lc.prefill_chunk = 64;
        lc.rtx_prefill = [](std::span<const std::uint32_t>, std::uint32_t) { return true; };
        lc.import_min_tokens = 100;
        lc.state_digest = true;
        lc.import_dsa_once = dsa;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        sv::GenerationJob j;
        j.prompt.assign(300, 9U);
        j.max_tokens = 4;
        auto t = queue.submit(std::move(j));
        sv::GenerationJob k;
        k.prompt = prompts(1)[0];  // short: never routed
        k.max_tokens = 4;
        auto u = queue.submit(std::move(k));
        bool done = false, done_u = false;
        while (!done) (void)t->wait(std::chrono::milliseconds(100), done);
        while (!done_u) (void)u->wait(std::chrono::milliseconds(100), done_u);
        stop.store(true);
        runner.join();
        follower.join();
        const auto st = t->stats(), su = u->stats();
        std::string route;
        for (const auto& [name, value] : st.engine_extra)
            if (name == "prefill_route") route = value;
        if (!fail && !relay_fail) CHECK(st.imported_tokens == 299 && route.find("\"route\":\"rtx\"") != std::string::npos && lead.stats().imported_tokens == 299);
        if (fail) {  // the fallback drops the hand-off on both ranks at once (import_drop)
            CHECK(st.imported_tokens == 0 && route.find("\"fallback\":\"rank0_staging_failed\"") != std::string::npos);
            CHECK(!e.dropped.empty() && e.dropped == e1.dropped);
        }
        if (dsa && !relay_fail) {  // both ranks relayed inside import_state's replay; rank 1 restored its full state
            CHECK(route.find("\"dsa_once\":true") != std::string::npos && route.find("\"relay_bytes\":407") != std::string::npos);
            CHECK(e1.relayed == relay_full(1) && e1.last_relay.chunks == 7);
            bool logged = false;
            for (const auto& line : e1.log) logged = logged || (line.rfind("import ", 0) == 0 && line.find(" dsa") != std::string::npos);
            CHECK(logged);
        }
        if (relay_fail) {  // the relay failed on both ranks at its header: import_abort on both, the prompt prefilled itself
            CHECK(st.imported_tokens == 0 && route.find("\"fallback\":\"both_restore_failed\"") != std::string::npos);
            CHECK(lead.stats().imports == 0 && lead.stats().import_fallbacks == 1);
            bool aborted0 = false, aborted1 = false;
            for (const auto& line : e.log) aborted0 = aborted0 || line.find("(import abort)") != std::string::npos;
            for (const auto& line : e1.log) aborted1 = aborted1 || line.find("(import abort)") != std::string::npos;
            CHECK(aborted0 && aborted1);
        }
        CHECK(su.imported_tokens == 0 && su.engine_extra.empty());
        const auto dg = lead.take_digests();  // one per request, right after its prefill, rank 1's gathered by rank 0
        CHECK(dg.size() == 2 && dg[0].rank0.size() == 4 && dg[0].rank0 == dg[0].rank1 && dg[1].rank0 == dg[1].rank1);
        CHECK(!route.empty() && sv::parse_json(route).is_object());
        static const char* names[] = {"imported", "staging fails", "imported, DSA once", "DSA once, rank 1 cannot relay"};
        std::printf("route receipt (%s): %s\n", names[variant], route.c_str());
    }
    // ---- a prompt waiting for its RTX state does not hold the admission: a short prompt behind it is served meanwhile
    {
        Loopback lb;
        FakeEng e(4), e1(4);
        e.stops = e1.stops = false;
        std::atomic<bool> hold{true};
        e.hold = e1.hold = &hold;
        gs::Channel ch(lb.side(0), 0), c1(lb.side(1), 1);
        std::thread follower([&] { gs::follow(e1, c1, hello_for(4, 4096, 2), 4, 4096); });
        ch.hello(hello_for(4, 4096, 2));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 4;
        lc.capacity = 4096;
        lc.heartbeat_seconds = 0.01;
        lc.prefill_chunk = 64;
        lc.rtx_prefill = [](std::span<const std::uint32_t>, std::uint32_t) { return true; };
        lc.import_min_tokens = 100;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        sv::GenerationJob j;
        j.prompt.assign(300, 9U);
        j.max_tokens = 4;
        auto t = queue.submit(std::move(j));
        sv::GenerationJob k;
        k.prompt.assign(90, 4U);  // two chunks, not routed
        k.max_tokens = 4;
        auto u = queue.submit(std::move(k));
        bool done = false, done_u = false;
        while (!done_u) (void)u->wait(std::chrono::milliseconds(100), done_u);
        (void)t->wait(std::chrono::milliseconds(1), done);
        CHECK(!done);  // the routed prompt still waits for its state
        hold.store(false);
        while (!done) (void)t->wait(std::chrono::milliseconds(100), done);
        stop.store(true);
        runner.join();
        follower.join();
        CHECK(t->stats().imported_tokens == 299 && u->stats().imported_tokens == 0 && e.log == e1.log);
        std::printf("import wait beside the prefill: short prompt served meanwhile: %s\n", done_u ? "yes" : "NO");
    }
    // ---- import gate shape (8192-token prompt, chunked admission on, the slot reused across arms): the imported arm's
    // tokens equal the own-prefill arm's from the first one on (no extra / stale first token), plain and MTP
    for (const bool mtp : {false, true}) {
        Loopback lb;
        FakeEng e(2), e1(2);
        e.stops = e1.stops = false;
        gs::Channel ch(lb.side(0), 0), c1(lb.side(1), 1);
        std::thread follower([&] { gs::follow(e1, c1, hello_for(2, 16384, 2), 2, 16384); });
        ch.hello(hello_for(2, 16384, 2));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 2;
        lc.capacity = 16384;
        lc.heartbeat_seconds = 0.01;
        lc.prefill_chunk = 512;
        lc.mtp = mtp;
        lc.mtp_cycles = 2;
        lc.state_digest = true;
        lc.rtx_prefill = [](std::span<const std::uint32_t>, std::uint32_t) { return true; };
        lc.import_min_tokens = 512;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        std::vector<std::uint32_t> q8k;
        for (unsigned i = 0; q8k.size() < 8188; ++i) q8k.insert(q8k.end(), {11U, 100000U + i % 997U});
        q8k.insert(q8k.end(), {102340U, 11U, 102573U, 22U});
        auto run = [&](bool routed) {
            lead.set_route(routed);
            sv::GenerationJob j;
            j.prompt = q8k;
            j.max_tokens = 32;
            auto t = queue.submit(std::move(j));
            std::vector<std::uint32_t> got;
            bool done = false;
            while (!done) {
                auto g = t->wait(std::chrono::milliseconds(100), done);
                got.insert(got.end(), g.begin(), g.end());
            }
            return std::make_pair(got, t->stats());
        };
        const auto b0 = run(false), a = run(true), b1 = run(false);
        stop.store(true);
        runner.join();
        follower.join();
        CHECK(a.second.imported_tokens == 8191 && b0.second.imported_tokens == 0);
        CHECK(a.first.size() == 32 && a.first == b0.first && b0.first == b1.first);
        CHECK(e.log == e1.log);
        bool import_line = false;
        for (const auto& line : e.log) import_line = import_line || line.rfind("import 0 8192", 0) == 0 || line.rfind("import 1 8192", 0) == 0;
        CHECK(import_line);
        const auto dg = lead.take_digests();
        CHECK(dg.size() == 3 && dg[0].rank0 == dg[1].rank0 && dg[1].rank0 == dg[2].rank0 && dg[1].frontier == 8192);
        std::printf("import gate shape, 8192-token prompt%s: imported == own prefill from the first token: %s\n", mtp ? ", MTP" : "",
                    a.first == b0.first ? "yes" : "NO");
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
