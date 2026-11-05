// strix-glm5-serve: GLM-5.3-Flash TP2 HTTP server with continuous batched decode (both ranks run this binary).
//
//   strix-glm5-serve MODEL_DIR --world 2 --rank 0 --transport usb4-auto [--host H --port P] [options]   lead + HTTP
//   strix-glm5-serve MODEL_DIR --world 2 --rank 1 --transport usb4-auto [options]                        follower
//   strix-glm5-serve MODEL_DIR --world 1 [options]                                                       one node
//
// Rank 0 serves OpenAI chat/completions + completions (SSE), /health, /metrics through ApiService and runs the lead
// loop (glm/serve/glm_serve_lead.hpp): admission between steps into the lowest free slot, ONE Engine::step for every
// decoding request per iteration. Rank 1 replays rank 0's control blocks (hello, prefill, step, heartbeat, shutdown;
// glm/serve/glm_serve_control.hpp) over the Communicator's host path (control tag) and echoes its tokens, so a
// cross-rank disagreement aborts both ranks.
// Options (both ranks must pass the same engine options; hello refuses a mismatch):
//   --slots S (default 8, <= 8 rows per step) --capacity C (positions per slot, default 8192)
//   --serial or STRIX_GLM_SERVE_BATCH=0: serial admission (one request at a time; the OFF switch of batching)
//   --prefill-chunk N (decode-first prefill interleave: while requests decode, one N-token chunk of a new prompt per
//   decode step, default 512; with nothing decoding the whole prompt in one call; 0 = always whole, the OFF switch) --prefill-chunk-mtp 0|1 (chunk MTP-bootstrap prompts too, default 1)
//   --heartbeat-ms MS (idle heartbeat, default 1000)
//   --identity / --prefill / --max-rows / --decode-xch / --baseline / --print-effective-config as strix-glm5-rank
//   rank 0: --host ADDR (127.0.0.1) --port P (8080) --model-id ID --max-tokens N (default max_tokens) --api-key-file F
//   rank 0: --selftest N [--selftest-tokens M] [--selftest-long L (multi-chunk prompt tokens, default 2048; 0 = skip the
//           interleave levels)] [--selftest-skip-basic (only the long-prompt reference and the staggered arms)]: no HTTP; N built-in prompts at once, then each alone (C=1), then the
//           first C together (C = 2, 4, 8): EXACT/FAIL per level with aggregate and per-stream tok/s; exit 0 EXACT, 3 FAIL
//   --prefix-cache-gib G: per-rank host LRU GiB (default 4, 0 = off); both ranks must agree.
//     Exact trunk restore and emitted tokens; MTP rows retain the existing chunk scheduling rule.
//     MTP snapshots use prompt + first greedy token as key, restoring at the prompt frontier.
//   --selftest-prefix: with --selftest N and cache ON, compare two-turn streams against cleared-cache replays
//           (--selftest-prefix-long L: first turns of ~L tokens, with warm vs cold timing lines).
//   --mtp-cycles N (verify cycles per control round on the C1 MTP path, default 1), --mtp-k K / --mtp-adaptive 0|1 /
//   --mtp-spec 0|1 as strix-glm5-rank; STRIX_GLM_SERVE_MTP=0 disables the C1 MTP path (plain decode at every C).
//   Batched self-speculation (G2, with MTP on): every decoding slot's window (anchor + adaptive drafts) in ONE multi-slot
//   verify of <= 8 rows when more than one request decodes; STRIX_GLM_SERVE_G2=0 = batched plain steps (the OFF switch).
//   Needs an attention seam with multi-slot verify (Engine::verify_multi_supported; off otherwise, said at startup).
// RTX prefill import (route off unless --rtx-prefill is set on rank 0):
//   every rank: --import-listen IPV4:PORT (this rank's receiver, the address the RTX producer connects to)
//               --import-key FILE (32 raw bytes, the producer's key) --import-model-sha HEX (the producer's model_sha256:
//               an operator attestation compared with the producer's hand-off header, NOT computed from the tensors this
//               rank loaded; every rank must pass the same one, the hello refuses a mismatch)
//               [--import-node r1|r1b: must equal --identity; r1b-all cannot import]
//   rank 0: --rtx-prefill HOST:PORT (producer, POST /v1/prefill) --import-peer IPV4:PORT (= rank 1's --import-listen)
//           --import-min-tokens N (default 4096) --import-timeout-s S (default 120)
//   rank 0: --import-gate P.u32[,P2.u32] [--import-gate-tokens N (64)]: no HTTP; per prompt file and decode mode (plain,
//           + MTP when on) arm A imports from the RTX, arm B prefills itself; prints route, tokens, TTFT split, tok/s,
//           I1 (A tokens == B tokens) and I0 (per-rank state digests per component after the prefill); exit 0 EQUAL, 3 DIFF
// C1 MTP: with one request in flight (none waiting) it decodes through the engine's exact single-slot self-speculation
// (verify / commit / adaptive K, the CLI --decode route); rank 1 replays mtp_enter / mtp_run / mtp_leave. Any second
// request ends the decoder at a cycle boundary (deferred rows flushed) and batched plain steps follow.
#include "glm_rank_defaults.hpp"
#include "glm_rank_startup.hpp"
#include "glm_rank_transport.hpp"

#include "strix/dist/communicator.hpp"
#include "strix/glm/engine.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/glm_chat.hpp"
#include "strix/serve/http_server.hpp"

#include "glm/engine/glm_mem.hpp"
#include "glm/prefill/glm_pf_path.hpp"
#include "glm/rtxpf/import/glm_import_socket.hpp"
#include "glm/serve/glm_serve_control.hpp"
#include "glm/serve/glm_serve_import.hpp"
#include "glm/serve/glm_serve_lead.hpp"
#include "glm/serve/glm_serve_selftest.hpp"
#include "glm/spec/glm_spec_run.hpp"

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace ge = strix::glm::engine;
namespace gs = strix::glm::serve;
namespace sv = strix::serve;
namespace rd = strix::glm::rank_defaults;
namespace sp = strix::glm::spec;
namespace gi = strix::glm::import;

std::atomic<bool> g_stop{false};
std::atomic<int> g_signal{0};
void on_signal(int sig) {
    g_signal.store(sig);
    g_stop.store(true);
}

// The engine calls of the control protocol (ge::Row and WireRow share their layout; converted field by field), the
// same on both ranks. The MTP state (drafter seed, open decoder, merged control state) lives here: every decision of
// the decoder comes from values merged across the ranks, so both ranks make the same engine calls.
// One blocking HTTP/1.1 POST (Connection: close) to the RTX producer; returns the status code (0 = no answer) and the
// response body in `reply` (when given).
int http_post(const std::string& host_port, const std::string& path, const std::string& body, int timeout_s,
              std::string* reply = nullptr) {
    const auto colon = host_port.rfind(':');
    if (colon == std::string::npos) throw std::invalid_argument("--rtx-prefill wants host:port");
    const std::string host = host_port.substr(0, colon), port = host_port.substr(colon + 1);
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) return 0;
    const int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return 0; }
    timeval tv{timeout_s, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    const bool connected = connect(fd, res->ai_addr, res->ai_addrlen) == 0;
    freeaddrinfo(res);
    if (!connected) { close(fd); return 0; }
    const std::string req = "POST " + path + " HTTP/1.1\r\nHost: " + host + "\r\nContent-Type: application/json\r\nContent-Length: " +
                            std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    for (std::size_t off = 0; off < req.size();) {
        const auto n = send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
        if (n <= 0) { close(fd); return 0; }
        off += static_cast<std::size_t>(n);
    }
    std::string resp;
    char buf[4096];
    for (;;) {
        const auto n = recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        resp.append(buf, static_cast<std::size_t>(n));
        if (resp.size() > (1U << 20)) break;
    }
    close(fd);
    if (resp.rfind("HTTP/1.", 0) != 0 || resp.size() < 12) return 0;
    if (reply) {
        const auto b = resp.find("\r\n\r\n");
        *reply = b == std::string::npos ? std::string() : resp.substr(b + 4);
    }
    return std::atoi(resp.c_str() + 9);
}

// Import gate (I0): the components of Engine::state_bytes_upto(slot, frontier) in its order (per layer: KDA S / conv,
// or DSA latent / index key / index gate / closed pools; layer 45 = MTP), sized for `world` ranks. Empty when the sizes
// do not add up to `total` (the gate then digests the whole state as one component).
struct StateComponent {
    std::string name;
    std::size_t bytes = 0;
    bool mtp = false;
};
std::vector<StateComponent> state_components(std::size_t total, unsigned world, std::uint64_t frontier) {
    const std::size_t heads = 64U / world, s = heads * 128U * 128U * 4U, conv = 3U * heads * 128U * 3U * 4U;
    const std::size_t latent = frontier * 512U * 2U, idx = frontier * 128U * 2U, pooled = (frontier / 4U) * 128U * 4U;
    std::vector<StateComponent> out;
    std::size_t sum = 0;
    for (unsigned l = 0; l < 46; ++l) {
        if (l == 45 && sum == total) break;  // no MTP layer in this engine
        const std::string ln = "L" + std::to_string(l) + (l == 45 ? "(mtp) " : " ");
        if (l < 45 && l % 4 != 3) {
            out.push_back({ln + "kda_S", s, false});
            out.push_back({ln + "kda_conv", conv, false});
        } else {
            out.push_back({ln + "dsa_latent", latent, l == 45});
            out.push_back({ln + "dsa_index_key", idx, l == 45});
            out.push_back({ln + "dsa_index_gate", idx, l == 45});
            out.push_back({ln + "dsa_pools", pooled, l == 45});
        }
        for (std::size_t k = out.size() - (l < 45 && l % 4 != 3 ? 2U : 4U); k < out.size(); ++k) sum += out[k].bytes;
    }
    if (sum != total) return {};
    return out;
}
// FNV-1a over 64-bit words (the tail bytes as one more word), seeded with the size.
std::uint64_t fnv64_words(const std::uint8_t* p, std::size_t n) {
    std::uint64_t h = 14695981039346656037ULL ^ n;
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        std::uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 1099511628211ULL;
    }
    if (i < n) {
        std::uint64_t w = 0;
        std::memcpy(&w, p + i, n - i);
        h = (h ^ w) * 1099511628211ULL;
    }
    return h;
}

struct EngineOps {
    ge::Engine& eng;
    sp::DecodePolicy policy;           // kmax / adaptive / capacity of the CLI route
    sp::BootstrapPolicy bootstrap = sp::BootstrapPolicy::Chunked;
    sp::ControlState control;
    std::optional<sp::Seed> seed;      // fresh only until the next engine call
    std::uint32_t seed_slot = 0;
    std::unique_ptr<sp::GreedyDecoder> dec;
    std::uint32_t dec_slot = 0;
    // batched self-speculation: per slot drafter state; its drafter rows live in Engine::mtp_hidden(slot) row hp_row
    std::vector<gs::SlotSpec> spec;
    std::vector<unsigned> hp_row;
    std::uint32_t prefix_restored = 0, prefix_bootstrap_pending = 0;

    void plain_call() {
        if (dec) throw std::logic_error("engine call while an MTP decoder is open");
        seed.reset();
    }
    void reset(std::uint32_t slot) {
        plain_call();
        prefix_restored &= ~(1U << slot);
        prefix_bootstrap_pending &= ~(1U << slot);
        eng.reset(slot);
        spec.at(slot) = {};
    }
    std::vector<std::uint8_t> state_bytes_upto(std::uint32_t slot, std::uint64_t p) {
        return eng.state_bytes_upto(slot, p);
    }
    void restore_state(std::uint32_t slot, const std::vector<std::uint8_t>& bytes, std::uint64_t p) {
        // reset() above has cleared SlotSpec and the old seed. pos0 > 0 does not reset bootstrap control.
        control.calibration = sp::Calibration{};
        control.cycle = control.step = 0;
        hp_row.at(slot) = 0;
        prefix_restored |= 1U << slot;
        prefix_bootstrap_pending |= 1U << slot;
        eng.restore_state(slot, bytes, p);
    }
    std::uint32_t prefill(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0) {
        plain_call();
        spec.at(slot) = {};
        return eng.prefill(slot, tokens, pos0);
    }
    std::vector<std::uint32_t> step(std::span<const gs::WireRow> rows) {
        plain_call();
        ge::Row r[gs::kMaxRows];
        for (std::size_t i = 0; i < rows.size(); ++i) {
            r[i] = ge::Row{rows[i].slot, rows[i].token, rows[i].pos};
            spec.at(rows[i].slot) = {};  // its drafter seed no longer follows the slot's frontier
        }
        return eng.step(std::span<const ge::Row>(r, rows.size()));
    }
    [[nodiscard]] const gs::SlotSpec& spec_state(std::uint32_t slot) const { return spec.at(slot); }
    void seed_slot_spec(std::uint32_t slot, const sp::Seed& s) {
        gs::SlotSpec& st = spec.at(slot);
        st = {};
        st.seeded = true;
        st.disabled = s.flagged;
        st.p = s.p;
        if (!s.flagged) {
            st.drafts.push_back(s.draft);
            st.conf.push_back(s.confidence);
        }
        hp_row.at(slot) = 0;  // the seed's drafter call had one row
    }
    // One more draft of the slot's chain: the MTP extension step of decode_greedy (h' of its last draft, that draft).
    std::uint32_t spec_draft(std::uint32_t slot) {
        plain_call();
        gs::SlotSpec& st = spec.at(slot);
        if (!st.seeded || st.disabled || st.drafts.empty()) throw std::logic_error("spec_draft: no drafter chain for the slot");
        const std::uint32_t token = st.drafts.back();
        auto c = control.next(3);
        const auto m = eng.mtp(slot, eng.mtp_hidden(slot) + static_cast<std::size_t>(hp_row.at(slot)) * 4096, {&token, 1},
                               st.p + st.drafts.size() - 1, c);
        control.consume(c);
        if (m.flagged) {
            st.disabled = true;
            return 0;
        }
        st.drafts.push_back(m.id.at(0));
        st.conf.push_back(m.confidence);
        hp_row[slot] = 0;
        return st.drafts.back();
    }
    // ONE verify of every window, then per window: greedy acceptance (decode_greedy's rule, the window's budget),
    // commit, and the drafter catch-up from the window's verify rows. Out per window: count, tokens.
    std::vector<std::uint32_t> spec_verify(std::span<const gs::WireRow> rows, std::span<const std::uint32_t> budgets) {
        plain_call();
        const auto win = gs::windows_of(rows, static_cast<std::uint32_t>(spec.size()), policy.capacity);
        std::vector<ge::Row> r(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) r[i] = ge::Row{rows[i].slot, rows[i].token, rows[i].pos};
        auto cv = control.next(1);
        const auto v = eng.verify_multi(r, cv);
        control.consume(cv);
        if (v.replay_flag) throw std::runtime_error("non-finite committed KDA row (replay)");
        std::vector<std::uint32_t> out;
        for (std::size_t i = 0; i < win.size(); ++i) {
            const auto& w = win[i];
            std::vector<std::uint32_t> drafts, target(v.id.begin() + w.row0, v.id.begin() + w.row0 + w.n);
            for (std::uint32_t j = 1; j < w.n; ++j) drafts.push_back(rows[w.row0 + j].token);
            unsigned first_flag = sp::kNoFlag;
            for (std::uint32_t j = 0; j < w.n; ++j)
                if (v.flag.at(w.row0 + j) != 0 && first_flag == sp::kNoFlag) first_flag = j;
            const auto acc = sp::accept_greedy(drafts, target, first_flag, sp::Terminal{sp::kNoFlag, budgets[i]});
            if (acc.abort) throw std::runtime_error("non-finite required row: request aborted (as plain greedy)");
            const std::uint64_t p = rows[w.row0].pos;
            eng.commit(w.slot, acc.rows, p);
            std::vector<std::uint32_t> emitted(drafts.begin(), drafts.begin() + acc.m);
            if (acc.has_bonus) emitted.push_back(acc.bonus);
            out.push_back(static_cast<std::uint32_t>(emitted.size()));
            out.insert(out.end(), emitted.begin(), emitted.end());
            gs::SlotSpec& st = spec.at(w.slot);
            const bool disabled = st.disabled;
            st = {};
            if (acc.stop) continue;
            // catch-up: pairs (h_j, x_{j+1}) of the committed rows, as decode_greedy
            std::vector<std::uint32_t> cu(drafts.begin(), drafts.begin() + acc.m);
            cu.push_back(acc.bonus);
            auto c = control.next(2);
            const auto m = eng.mtp(w.slot, eng.last_hidden() + static_cast<std::size_t>(w.row0) * 4096, cu, p, c);
            control.consume(c);
            st.seeded = true;
            st.disabled = disabled || m.flagged;
            st.p = p + acc.rows;
            if (!st.disabled) {
                st.drafts.push_back(m.id.at(acc.m));
                st.conf.push_back(m.confidence);
            }
            hp_row.at(w.slot) = acc.m;
        }
        return out;
    }
    // The CLI bootstrap: exact chunked prompt + MTP layer (or reference rows), seed for the slot's first cycle. A
    // chunk of a longer prompt (pos0 > 0 or not last) runs Engine::prefill_mtp on the chunk at pos0 in R1, as the
    // bootstrap does for the whole prompt: same trunk rows; the drafter pair at each chunk's last row takes the chunk's
    // greedy token instead of the next prompt token (drafts only; output exactness never depends on the drafter).
    std::uint32_t prefill_mtp(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint64_t pos0, bool last) {
        plain_call();
        if ((prefix_bootstrap_pending >> slot) & 1U) {
            // Admission may have decoded other slots since restore: reset at this request's first suffix call too.
            control.calibration = sp::Calibration{};
            control.cycle = control.step = 0;
            spec.at(slot) = {};
            hp_row.at(slot) = 0;
            prefix_bootstrap_pending &= ~(1U << slot);
        }
        if (pos0 == 0 && last) {
            seed = sp::bootstrap(eng, slot, tokens, bootstrap, control);
            seed_slot = slot;
            seed_slot_spec(slot, *seed);
            return seed->y0;
        }
        if (bootstrap != sp::BootstrapPolicy::Chunked) {
            if (((prefix_restored >> slot) & 1U) == 0) throw std::logic_error("chunked MTP prefill needs the exact chunked prefill path");
            // Reference bootstrap indexes the complete prompt at pos0; only its suffix is read.
            std::vector<std::uint32_t> prompt(static_cast<std::size_t>(pos0), 0);
            prompt.insert(prompt.end(), tokens.begin(), tokens.end());
            const auto sd = sp::bootstrap(eng, slot, prompt, bootstrap, control, pos0);
            if (last) {
                seed = sd;
                seed_slot = slot;
                seed_slot_spec(slot, sd);
            }
            return sd.y0;
        }
        if (pos0 == 0) {  // a new prompt: as sp::bootstrap
            control.calibration = sp::Calibration{};
            control.cycle = control.step = 0;
        }
        const auto saved = eng.identity_mode();
        eng.set_identity_mode(ge::IdentityMode::kR1);
        ge::Engine::MtpPrefillOut b;
        try {
            auto c = control.next(2);
            b = eng.prefill_mtp(slot, tokens, pos0, c);
            control.consume(c);
        } catch (...) {
            eng.set_identity_mode(saved);
            throw;
        }
        eng.set_identity_mode(saved);
        if (last) {
            sp::Seed sd;
            sd.y0 = b.y0;
            sd.draft = b.mtp.id.at(0);
            sd.confidence = b.mtp.confidence;
            sd.hp = b.hp;
            sd.flagged = b.mtp.flagged;
            sd.p = pos0 + tokens.size();
            seed = sd;
            seed_slot = slot;
            seed_slot_spec(slot, sd);
        }
        return b.y0;
    }
    // import: the payload staged under `id` (the RTX producer's per-rank state at frontier P = |tokens| - 1, the
    // Engine::restore_state layout), then the last prompt token through the same call the slot's own prefill makes for
    // it (prefill at P, or the MTP bootstrap from P with its seed kept). Returns the first generated token.
    std::function<std::span<const std::uint8_t>(std::uint32_t id)> staged{};
    std::unique_ptr<gs::ImportStager> stager{};  // --import-listen: this rank's RTX hand-offs (off: imports refused)
    std::uint32_t import_last = 0;
    unsigned world = 2;                                   // state component sizes (import gate)
    double last_restore_ms = 0, last_token_ms = 0;        // this rank's last import_state: restore / last-token prefill
    // Import gate (I0): per component of state_bytes_upto(slot, frontier): fnv64 lo/hi, size lo/hi. Reads only.
    std::vector<std::uint32_t> state_digest(std::uint32_t slot, std::uint64_t frontier) {
        const auto b = eng.state_bytes_upto(slot, frontier);
        auto comps = state_components(b.size(), world, frontier);
        if (comps.empty()) comps.push_back({"all", b.size(), false});
        std::vector<std::uint32_t> out;
        std::size_t o = 0;
        for (const auto& c : comps) {
            const std::uint64_t h = fnv64_words(b.data() + o, c.bytes), z = c.bytes;
            out.insert(out.end(), {static_cast<std::uint32_t>(h), static_cast<std::uint32_t>(h >> 32), static_cast<std::uint32_t>(z),
                                   static_cast<std::uint32_t>(z >> 32)});
            o += c.bytes;
        }
        return out;
    }
    std::optional<gs::ImportReport> import_done{};  // import_state's outcome for import_last (restored / failed + token)
    // Never throws (a rank that cannot stage reports failed; the lead falls back on both ranks).
    void import_expect(std::uint32_t id, std::span<const std::uint32_t> tokens) {
        if (stager && import_last != 0 && import_last != id) stager->drop(import_last);  // a fallen-back import's leftovers
        import_last = id;
        import_done.reset();
        try {
            if (!stager) throw std::runtime_error("this rank has no --import-listen");
            // host-memory admission of the staging (the receive assembles this rank's full restore payload in RAM)
            if (state_layout && import_mem_guard != 0 && tokens.size() >= 2) {
                std::uint64_t need = 0;
                for (const auto& c : state_layout(tokens.size() - 1, tokens.size())) need += c.bytes;
                const auto avail = mem_available();
                if (avail && *avail < need + import_mem_guard)
                    throw std::runtime_error("host memory: " + std::to_string(*avail >> 20) + " MiB available, staging needs " +
                                             std::to_string(need >> 20) + " MiB + the guard");
            }
            stager->expect(id, std::vector<std::uint32_t>(tokens.begin(), tokens.end()));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "rank %u: import %u refused: %s\n", control.rank, id, e.what());
            import_done = gs::ImportReport{id, static_cast<std::uint32_t>(gs::ImportStatus::failed), 0};
        }
    }
    gs::ImportReport import_report() const {
        if (import_done && import_done->id == import_last) return *import_done;
        return {import_last, stager ? static_cast<std::uint32_t>(stager->status(import_last)) : 0U, 0};
    }
    // 'DSA once' (--import-dsa-once on every rank): the receivers expect hand-off version 2 and import_state relays rank
    // 0's DSA components to rank 1 (relay: one pair exchange; layout: the full per-rank component list at frontier P).
    bool dsa_once = false;
    std::function<void(std::span<const std::byte> local, std::span<std::byte> remote, std::uint64_t seq)> relay{};
    std::function<std::vector<gs::StateComponent>(std::uint64_t frontier, std::uint64_t total)> state_layout{};
    std::size_t relay_chunk = 0;
    gs::RelayStats last_relay{};
    // Host-memory admission (staging, and the bytes the relay / merge still allocate on this rank): refused when
    // MemAvailable minus them would drop below import_mem_guard (0 = no check). Only MISSING telemetry skips the check
    // (no /proc/meminfo, or no MemAvailable line); a measured MemAvailable of 0 is refused like any short one.
    std::uint64_t import_mem_guard = std::uint64_t{4} << 30;
    static std::optional<std::uint64_t> mem_available() {
        std::ifstream f("/proc/meminfo");
        std::string k, unit;
        std::uint64_t v = 0;
        while (f >> k >> v >> unit)
            if (k == "MemAvailable:") return v * 1024ULL;
        return std::nullopt;
    }
    std::function<gs::Binding(std::uint32_t id)> staged_binding{};
    // The pair vote / relay exchange (the communicator's host path, its own control layers; empty with one rank).
    std::function<void(std::span<const std::byte> local, std::span<std::byte> remote, std::uint64_t seq)> vote{};
    std::uint32_t import_state(std::uint32_t slot, std::uint32_t id, std::span<const std::uint32_t> tokens, bool mtp, bool dsa) {
        plain_call();
        // Prepare (local): staged under the announced id, the relay mode both ranks were started with, the binding the
        // hand-off was authenticated under, and host memory for what the relay still allocates. The vote makes both
        // ranks abort together on any of them (gs::import_transaction).
        bool prepared = true;
        std::string why;
        gs::Binding binding{};
        std::span<const std::uint8_t> mine;
        std::vector<gs::StateComponent> comps;
        const std::uint64_t p = tokens.size() >= 2 ? tokens.size() - 1 : 0;
        try {
            if (!staged || !staged_binding) throw std::runtime_error("no staging on this rank");
            if (id != import_last) throw std::runtime_error("not the announced import");
            if (tokens.size() < 2) throw std::runtime_error("prompt shorter than two tokens");
            if (dsa != dsa_once) throw std::runtime_error(dsa ? "the lead asked for the DSA relay, this rank staged a v1 hand-off"
                                                              : "this rank staged a 'DSA once' hand-off, the lead did not ask for the relay");
            if (dsa && (!relay || !state_layout)) throw std::runtime_error("no pair relay on this rank");
            mine = staged(id);
            binding = staged_binding(id);
            std::uint64_t need = 0;
            if (dsa) {
                comps = state_layout(p, tokens.size());
                const auto d = gs::component_bytes(comps, true);
                need = control.rank == 0 ? d : d + mine.size() + d;  // rank 1: DSA buffer + merged payload
            }
            if (import_mem_guard != 0 && need != 0) {
                const auto avail = mem_available();
                if (avail && *avail < need + import_mem_guard)
                    throw std::runtime_error("host memory: " + std::to_string(*avail >> 20) + " MiB available, the relay needs " +
                                             std::to_string(need >> 20) + " MiB + the guard");
            }
        } catch (const std::exception& e) {
            prepared = false;
            why = e.what();
        }
        last_relay = {};
        std::vector<std::uint8_t> merged;  // rank 1 with the relay: its KDA payload + rank 0's DSA components
        std::uint32_t y0 = 0;
        std::chrono::steady_clock::time_point t0, t1;
        y0 = gs::import_transaction(
            prepared, why, binding, p, vote,
            [&] {
                if (!dsa) return;
                merged = gs::dsa_relay(control.rank, true, mine, comps, relay_chunk, relay, &last_relay);
                std::printf("rank %u: import %u DSA relay %s %llu bytes in %llu chunks, %.1f ms\n", control.rank, id,
                            control.rank == 0 ? "sent" : "received", static_cast<unsigned long long>(last_relay.bytes),
                            static_cast<unsigned long long>(last_relay.chunks), last_relay.ms);
            },
            [&] {
                t0 = std::chrono::steady_clock::now();
                if (!merged.empty()) eng.restore_state(slot, merged, p);
                else eng.restore_state(slot, mine, p);
                t1 = std::chrono::steady_clock::now();
            },
            [&]() -> std::uint32_t {
                if (stager) stager->drop(id);  // both ranks restored: the engine holds the state now
                std::vector<std::uint8_t>().swap(merged);  // release the merged copy (the engine holds the state now)
                if (mtp) {
                    seed = sp::bootstrap(eng, slot, tokens, bootstrap, control, p);
                    seed_slot = slot;
                    seed_slot_spec(slot, *seed);  // batched self-speculation starts from the same seed as after its own prefill
                    return seed->y0;
                }
                spec.at(slot) = {};
                return eng.prefill(slot, tokens.subspan(static_cast<std::size_t>(p)), p);
            });
        last_restore_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_token_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        std::printf("rank %u: import %u restored P=%llu rows (slot %u), last-token prefill pos %llu token %u%s -> first token %u"
                    " (restore %.1f ms, last token %.1f ms)\n",
                    control.rank, id, static_cast<unsigned long long>(p), slot, static_cast<unsigned long long>(p), tokens[p],
                    mtp ? " (MTP bootstrap)" : "", y0, last_restore_ms, last_token_ms);
        std::fflush(stdout);
        import_done = gs::ImportReport{id, static_cast<std::uint32_t>(gs::ImportStatus::restored), y0};
        return y0;
    }
    // import_drop (both ranks): forget the hand-off staged / still being received under `id` (fallback, cancel).
    void import_drop(std::uint32_t id) {
        if (stager) stager->drop(id);
        if (import_last == id) import_done = gs::ImportReport{id, static_cast<std::uint32_t>(gs::ImportStatus::failed), 0};
    }
    // import_state failed on this rank (replay caught it): the slot is reset and the failure reported; a failing reset
    // propagates (the engine is unusable).
    void import_abort(std::uint32_t slot, std::uint32_t id, const std::string& why) {
        std::fprintf(stderr, "rank %u: import %u not restored (slot %u reset, prefill fallback): %s\n", control.rank, id, slot, why.c_str());
        if (dec) throw std::logic_error("import_abort while an MTP decoder is open");
        seed.reset();
        if (stager) stager->drop(id);
        eng.reset(slot);
        spec.at(slot) = {};
        import_done = gs::ImportReport{id, static_cast<std::uint32_t>(gs::ImportStatus::failed), 0};
    }
    // Seed from the last step's row r (it produced row.token at row.pos): the drafter pair (h_r, token) at row.pos,
    // as decode_greedy's catch-up does after a verify.
    // from_seed: the C1 decoder takes over the slot's batched-spec drafter seed (its first draft, h' row hp_row).
    void mtp_enter(std::uint32_t r, const gs::WireRow& row, bool from_seed) {
        plain_call();
        if (from_seed) {
            gs::SlotSpec& st = spec.at(row.slot);
            if (!st.seeded || st.disabled || st.drafts.size() != 1 || st.p != row.pos)
                throw std::logic_error("mtp_enter: no fresh batched-spec seed for the slot");
            sp::Seed s;
            s.y0 = row.token;
            s.draft = st.drafts[0];
            s.confidence = st.conf[0];
            s.hp = eng.mtp_hidden(row.slot) + static_cast<std::size_t>(hp_row.at(row.slot)) * 4096;
            s.p = row.pos;
            seed = s;
            seed_slot = row.slot;
            st = {};
            return;
        }
        spec.at(row.slot) = {};
        auto c = control.next(2);
        const auto m = eng.mtp(row.slot, eng.last_hidden() + static_cast<std::size_t>(r) * 4096, {&row.token, 1}, row.pos, c);
        control.consume(c);
        sp::Seed s;
        s.y0 = row.token;
        s.draft = m.id.at(0);
        s.confidence = m.confidence;
        s.hp = eng.mtp_hidden();
        s.flagged = m.flagged;
        s.p = row.pos + 1;
        seed = s;
        seed_slot = row.slot;
    }
    // Up to `cycles` verify cycles; the first call opens the decoder from the fresh seed (budget includes the seed's y0,
    // already emitted). Returns the tokens after the ones already returned.
    std::vector<std::uint32_t> mtp_run(std::uint32_t slot, std::uint32_t budget, std::uint32_t cycles) {
        if (!dec) {
            if (!seed || seed_slot != slot) throw std::logic_error("mtp_run without a fresh drafter seed");
            dec = std::make_unique<sp::GreedyDecoder>(eng, slot, *seed, policy, budget, control);
            dec_slot = slot;
            seed.reset();
            spec.at(slot) = {};
        } else if (dec_slot != slot) {
            throw std::logic_error("mtp_run on another slot than the open decoder");
        }
        const std::size_t before = dec->result().tokens.size();
        for (std::uint32_t i = 0; i < cycles && !dec->done(); ++i) (void)dec->cycle();
        const auto& t = dec->result().tokens;
        return {t.begin() + static_cast<std::ptrdiff_t>(before), t.end()};
    }
    void mtp_leave(std::uint32_t slot) {
        seed.reset();
        spec.at(slot) = {};
        if (dec) {
            if (dec_slot != slot) throw std::logic_error("mtp_leave of another slot than the open decoder");
            dec->finish();
            dec.reset();
        } else {
            eng.flush(slot);
        }
    }
};

// Metadata only (no tensor payload): config.json bytes, then every *.json / *.safetensors name and size, sorted.
std::uint64_t model_hash(const std::string& dir) {
    std::vector<std::pair<std::string, std::uint64_t>> files;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const auto ext = e.path().extension().string();
        if (ext == ".json" || ext == ".safetensors") files.emplace_back(e.path().filename().string(), e.file_size());
    }
    std::sort(files.begin(), files.end());
    std::uint64_t h = 14695981039346656037ULL;
    std::ifstream f(dir + "/config.json", std::ios::binary);
    const std::string cfg{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    h = gs::fnv1a64(cfg, h);
    for (const auto& [name, size] : files) {
        h = gs::fnv1a64(name, h);
        h = gs::fnv1a64(&size, sizeof size, h);
    }
    return h;
}

std::string file_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read " + path);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// Built-in self-test prompts (varied lengths: rows of one step sit at different positions).
std::vector<std::vector<std::uint32_t>> selftest_prompts(const sv::BpeTokenizer& tok, unsigned n) {
    static const char* texts[] = {
        "What is the capital of France? Answer in one sentence.",
        "Write a short poem about the sea at night.",
        "Explain in three bullet points why the sky is blue.",
        "List the first ten prime numbers and explain how you found them.",
        "Translate to Spanish: The quick brown fox jumps over the lazy dog, and then it runs into the forest to find its "
        "friends, who are waiting near the old oak tree by the river.",
        "Write a Python function that returns the n-th Fibonacci number iteratively, then explain its complexity.",
        "Summarize the plot of a heist story in which a group of retired engineers tries to steal a famous painting from a "
        "museum, but everything goes wrong because one of them forgot the date of the exhibition change. Keep it short.",
        "Give me a recipe for a simple tomato soup with exact quantities.",
    };
    std::vector<std::vector<std::uint32_t>> out;
    for (unsigned i = 0; i < n; ++i) {
        std::string text = texts[i % std::size(texts)];
        if (i >= std::size(texts)) text += " (variant " + std::to_string(i / std::size(texts)) + ")";
        sv::ChatTemplateInput in;
        in.messages = sv::Json::array();
        sv::Json m = sv::Json::object();
        m.set("role", sv::Json("user"));
        m.set("content", sv::Json(text));
        in.messages.push(std::move(m));
        out.push_back(tok.encode(sv::render_glm_chat(in), true));
    }
    return out;
}

// Multi-chunk self-test prompts (prefill interleave): a user message of about `tokens` tokens, the built-in texts
// repeated (each prompt starts at another text).
std::vector<std::vector<std::uint32_t>> selftest_long_prompts(const sv::BpeTokenizer& tok, unsigned n, std::size_t tokens) {
    static const char* texts[] = {
        "The capital of France is Paris, a city on the Seine known for its museums and cafes.",
        "The sea at night reflects the moon; waves roll slowly toward the dark shore.",
        "The sky looks blue because air molecules scatter short wavelengths of sunlight more strongly.",
        "The first ten primes are 2, 3, 5, 7, 11, 13, 17, 19, 23 and 29; each has exactly two divisors.",
        "An iterative Fibonacci function keeps two running values and updates them n times.",
        "A group of retired engineers planned to borrow a painting, but they forgot the exhibition date.",
        "A simple tomato soup needs tomatoes, onion, garlic, olive oil, stock, salt and pepper.",
        "The quick brown fox jumps over the lazy dog and runs into the forest by the river.",
    };
    std::vector<std::vector<std::uint32_t>> out;
    for (unsigned i = 0; i < n; ++i) {
        std::string text = "Read the following notes, then answer the last question in two sentences.\n";
        for (unsigned k = 0; tok.encode(text, false).size() + 64 < tokens; ++k)
            text += "Note " + std::to_string(k) + ": " + texts[(i + k) % std::size(texts)] + "\n";
        text += "Question: which note mentions a river, and what happens there?";
        sv::ChatTemplateInput in;
        in.messages = sv::Json::array();
        sv::Json m = sv::Json::object();
        m.set("role", sv::Json("user"));
        m.set("content", sv::Json(text));
        in.messages.push(std::move(m));
        out.push_back(tok.encode(sv::render_glm_chat(in), true));
    }
    return out;
}

}  // namespace

// --import-gate with no prompt file after parsing (e.g. ","): nothing would be qualified, so the verdict is FAIL.
static int import_gate_empty(const std::string& list) {
    std::printf("rank 0: import-gate: no prompt files in '%s', coverage FAIL, verdict FAIL, STOPPED\n", list.c_str());
    return 4;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    try {
        if (argc < 2) {
            std::fprintf(stderr, "usage: %s MODEL_DIR --world 1|2 [--rank R --transport T] [--slots S] [--capacity C] [--serial] "
                                 "[--host H] [--port P] [--prefix-cache-gib G] [--selftest N [--selftest-tokens M] [--selftest-prefix [--selftest-prefix-long L]]] "
                                 "[--import-listen IP:PORT --import-key FILE --import-model-sha HEX (operator attestation of the producer's "
                                 "model, compared with its hand-off header, not computed from the loaded tensors)]\n", argv[0]);
            return 2;
        }
        const auto t_start = std::chrono::steady_clock::now();
        ge::Options opt;
        opt.model_dir = argv[1];
        opt.slots = 8;
        std::string transport_spec, host = "127.0.0.1", model_id = "glm-5.3-flash", api_key_file;
        unsigned port = 8080, selftest = 0, selftest_tokens = 128, heartbeat_ms = 1000, mtp_cycles = 1, prefill_chunk = 512;
        std::uint64_t prefix_cache_bytes = 4ULL << 30;  // default 4 GiB per rank; --prefix-cache-gib 0 = off
        bool selftest_prefix = false;
        unsigned selftest_long = 2048, selftest_prefix_long = 0;
        bool prefill_chunk_mtp = true, selftest_skip_basic = false;
        bool serve_mtp = true;
        std::uint32_t max_tokens = 4096;
        bool serial = false, print_config = false;
        // RTX prefill route: every rank receives its hand-off on --import-listen; rank 0 asks the RTX producer
        // (--rtx-prefill host:port) for prompts of >= --import-min-tokens and names rank 1's --import-peer endpoint.
        std::string import_listen, import_key_file, import_model_sha, import_node, rtx_prefill, import_peer;
        std::uint64_t import_min_tokens = 4096;
        double import_timeout_s = 120.0;
        std::string import_gate;              // rank 0: PROMPT.u32[,PROMPT2.u32] (I0 / I1 gate, no HTTP)
        bool import_gate_on = false;          // --import-gate given (an empty list is a gate that FAILs, not serving)
        unsigned import_gate_tokens = 64;
        bool import_dsa_once = false;         // every rank: hand-off v2, rank 0 relays the DSA components to rank 1
        std::vector<std::string> rank_args;  // the options the rank defaults depend on (rd::scan)
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if (a == "--world") { rank_args.push_back(a); rank_args.push_back(next()); opt.world = static_cast<std::uint32_t>(std::stoul(rank_args.back())); }
            else if (a == "--identity" || a == "--prefill" || a == "--max-rows" || a == "--decode-xch" || a == "--mtp-k" ||
                     a == "--mtp-adaptive" || a == "--mtp-spec") {
                rank_args.push_back(a);
                rank_args.push_back(next());
            }
            else if (a == "--baseline") rank_args.push_back(a);
            else if (a == "--print-effective-config") { rank_args.push_back(a); print_config = true; }
            else if (a == "--rank") opt.rank = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--transport") transport_spec = next();
            else if (a == "--slots") opt.slots = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--capacity") opt.capacity = std::stoull(next());
            else if (a == "--serial") serial = true;
            else if (a == "--mtp-cycles") mtp_cycles = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--prefill-chunk") prefill_chunk = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--prefill-chunk-mtp") prefill_chunk_mtp = std::stoul(next()) != 0;
            else if (a == "--prefix-cache-gib") {
                prefix_cache_bytes = gs::prefix_budget_bytes(next());
            }
            else if (a == "--selftest-prefix") selftest_prefix = true;
            else if (a == "--selftest-prefix-long") selftest_prefix_long = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--selftest-long") selftest_long = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--selftest-skip-basic") selftest_skip_basic = true;
            else if (a == "--heartbeat-ms") heartbeat_ms = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--host") host = next();
            else if (a == "--port") port = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--model-id") model_id = next();
            else if (a == "--max-tokens") max_tokens = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--api-key-file") api_key_file = next();
            else if (a == "--selftest") selftest = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--selftest-tokens") selftest_tokens = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--import-listen") import_listen = next();
            else if (a == "--import-key") import_key_file = next();
            else if (a == "--import-model-sha") import_model_sha = next();
            else if (a == "--import-node") import_node = next();
            else if (a == "--rtx-prefill") rtx_prefill = next();
            else if (a == "--import-peer") import_peer = next();
            else if (a == "--import-min-tokens") import_min_tokens = std::stoull(next());
            else if (a == "--import-timeout-s") import_timeout_s = std::stod(next());
            else if (a == "--import-gate") {
                import_gate = next();
                import_gate_on = true;
            }
            else if (a == "--import-gate-tokens") import_gate_tokens = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--import-dsa-once") import_dsa_once = true;
            else throw std::invalid_argument("unknown option " + a);
        }
        if (const char* b = std::getenv("STRIX_GLM_SERVE_BATCH"); b != nullptr && std::string(b) == "0") serial = true;
        if (const char* m = std::getenv("STRIX_GLM_SERVE_MTP"); m != nullptr && std::string(m) == "0") serve_mtp = false;
        bool serve_g2 = true;
        if (const char* g = std::getenv("STRIX_GLM_SERVE_G2"); g != nullptr && std::string(g) == "0") serve_g2 = false;
        if (mtp_cycles == 0 || mtp_cycles > gs::kMaxMtpCycles) throw std::invalid_argument("--mtp-cycles 1..64");
        if (opt.slots == 0 || opt.slots > gs::kMaxRows) throw std::invalid_argument("--slots must be 1..8 (rows per step)");
        if (opt.capacity == 0 || opt.capacity % 4 != 0) throw std::invalid_argument("--capacity must be a positive multiple of 4");
        if (opt.world != 1 && opt.world != 2) throw std::invalid_argument("--world 1|2");
        if (opt.rank >= opt.world) throw std::invalid_argument("--rank must be < --world");
        if (import_gate_on && import_gate.find_first_not_of(',') == std::string::npos)
            return import_gate_empty(import_gate);  // --import-gate "" or ",": an explicit empty suite FAILs before loading
        if (import_gate_on && (opt.rank != 0 || rtx_prefill.empty()))  // before the follower dispatch: rank 1 must not exit 0
            throw std::invalid_argument("--import-gate: rank 0 with --rtx-prefill (arm A imports, arm B prefills itself)");
        if (selftest_prefix && (!selftest || !prefix_cache_bytes || selftest_tokens == 0))
            throw std::invalid_argument("--selftest-prefix needs --selftest N, --selftest-tokens > 0 and --prefix-cache-gib > 0");
        if (selftest_prefix_long > 0 && !selftest_prefix) throw std::invalid_argument("--selftest-prefix-long L needs --selftest-prefix");
        if (selftest > 32) throw std::invalid_argument("--selftest N: 1..32");

        // Engine configuration: the rank app's resolution of a normal decode (MTP self-speculation and its exact chunked
        // bootstrap when the model has a complete MTP head; STRIX_GLM_SERVE_MTP=0 resolves as --mtp-spec 0). No decode
        // handshake here (both ranks resolve on their own and hello compares the result), so automatic max-rows takes its
        // fixed fallback.
        rank_args.push_back("--decode");
        rank_args.push_back("1");
        if (!serve_mtp) {
            rank_args.push_back("--mtp-spec");
            rank_args.push_back("0");
        }
        const auto request = rd::scan(rank_args);
        auto eff = rd::resolve(request, rd::process_env(), rd::probe_model_mtp(argv[1]));
        rd::fallback_without_handshake(eff, request, rd::process_env());
        const bool mtp_on = eff.mtp_spec.value == "1" && (eff.prefill == "gemv" || eff.prefill == "ref");
        if (rd::max_rows_auto(eff)) eff.max_rows = {std::to_string(rd::kFallbackMaxRows), "auto-4096-serve"};
        if (print_config) {
            std::fputs(rd::print(eff).c_str(), stdout);
            std::printf("serve slots=%u capacity=%llu batch=%s prefill-chunk=%u (mtp %d) c1-mtp=%s kmax=%u adaptive=%d cycles/round=%u\n",
                        opt.slots, static_cast<unsigned long long>(opt.capacity), serial ? "serial" : "on", prefill_chunk,
                        prefill_chunk_mtp ? 1 : 0,
                        mtp_on ? "on" : "off", eff.kmax, eff.adaptive ? 1 : 0, mtp_cycles);
            // The RTX import route (FAST identity glm-pf-rtx-native-v1: imported prompts are not bitwise the pair's own
            // prefill; off unless --rtx-prefill / --import-listen are given).
            std::printf("serve import-route=%s import-listen=%s wire=v%u identity=%s\n", rtx_prefill.empty() ? "off" : "rtx",
                        import_listen.empty() ? "off" : "on", import_dsa_once ? 2U : 1U,
                        rtx_prefill.empty() && import_listen.empty() ? "exact" : "exact+FAST(glm-pf-rtx-native-v1, imported prompts)");
            return 0;
        }
        rd::apply_env(eff, [](const char* k, const char* v) { (void)::setenv(k, v, 0); });
        opt.mtp = mtp_on;
        opt.spec = mtp_on;
        opt.max_rows = static_cast<unsigned>(std::stoul(eff.max_rows.value));
        opt.auto_rows = false;
        opt.exchange_rows = opt.max_rows;
        const std::string& id = eff.identity.value;
        if (id != "r1" && id != "r1b" && id != "r1b-all") throw std::invalid_argument("--identity r1|r1b|r1b-all");
        opt.identity = id == "r1" ? ge::IdentityMode::kR1 : id == "r1b" ? ge::IdentityMode::kR1b : ge::IdentityMode::kR1bAll;
        opt.decode_xfer = eff.decode_xch.value == "xfer";
        std::string prefill_mode = eff.prefill;
        if (prefill_mode != "ref" && prefill_mode != "gemv" && prefill_mode != "wmma")
            throw std::invalid_argument("prefill must resolve to ref, gemv or wmma");
        const char* oom = std::getenv("STRIX_GLM_OOM_SCORE_ADJ");
        if (!(oom != nullptr && std::string(oom) == "keep") && !ge::mem::set_oom_score_adj(oom ? std::atoi(oom) : 1000))
            std::fprintf(stderr, "warning: could not set oom_score_adj\n");
        std::printf("rank %u: pid %d, slots %u capacity %llu max-rows %u prefill %s identity %s batch %s\n", opt.rank,
                    static_cast<int>(::getpid()), opt.slots, static_cast<unsigned long long>(opt.capacity), opt.max_rows,
                    prefill_mode.c_str(), id.c_str(), serial ? "serial" : "on");

        std::unique_ptr<strix::dist::Communicator> comm;
        if (opt.world == 2) {
            if (transport_spec.empty()) throw std::invalid_argument("--world 2 needs --transport");
            (void)::setenv("STRIX_DIST_USB4_TIMEOUT_MS", "600000", 0);
            (void)::setenv("STRIX_DIST_WAIT_TIMEOUT_MS", "600000", 0);
            std::string receipt;
            auto transport = strix::glm::rank_transport::open_transport(transport_spec, opt.rank, receipt);
            if (!receipt.empty()) std::printf("transport: %s\n", receipt.c_str());
            comm = strix::glm::rank_transport::make_communicator(opt.rank, std::move(transport), opt.exchange_rows);
            std::printf("connected: rank %u transport %s %.1f s\n", opt.rank, comm->transport_name().c_str(),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());
        }
        ge::Engine eng(opt, comm.get());
        const auto lr = eng.load([&](const ge::mem::LoadAdmission&) { return opt.max_rows; });
        if (prefill_mode != "ref") {
            namespace pf = strix::glm::pf;
            const unsigned heads = 64 / opt.world;
            eng.set_prefill(pf::make_prefill_path(heads, heads, opt.capacity, opt.max_rows,
                                                  prefill_mode == "gemv" ? pf::TierB::gemv : pf::TierB::wmma));
        }
        std::printf("rank %u: load %.1f s, %.1f GiB read, device %.1f GiB, identity %s, decode-xch %s, prefill %s pipeline %u\n",
                    opt.rank, lr.seconds, lr.gib_read, lr.device_gib, ge::identity_mode_name(eng.identity_mode()),
                    opt.world == 1 ? "local" : eng.decode_xfer_enabled() ? "xfer" : "comm", prefill_mode.c_str(), eng.prefill_pipeline());

        // Control channel: the Communicator's host path, control tag. Between engine calls only: every step / prefill has
        // returned (Xfer drained by the step's argmax, prefill completed), so the transport is idle.
        gs::Channel::Wire wire;
        if (comm) {
            wire = [&comm](std::span<const std::byte> out, std::span<std::byte> in, std::uint32_t layer, std::uint64_t req) {
                strix::dist::Tag tag;
                tag.kind = static_cast<std::uint32_t>(strix::dist::TagKind::control);
                tag.layer = 0x474C0000U | layer;
                tag.request = req;
                comm->exchange_host(out, in, tag);
            };
        }
        gs::Channel ch(std::move(wire), opt.rank);
        gs::Hello hello;
        hello.prefix_cache_bytes = prefix_cache_bytes;
        hello.model_hash = model_hash(opt.model_dir);
        {
            std::string ident = rd::print(eff);
            ident += std::string("engine identity ") + ge::identity_mode_name(eng.identity_mode()) + " decode-xfer " +
                     (eng.decode_xfer_enabled() ? "1" : "0") + " prefill " + prefill_mode + " pipeline " +
                     std::to_string(eng.prefill_pipeline()) + " mtp " + std::to_string(mtp_on ? 1 : 0) + " g2 " + std::to_string(serve_g2 ? 1 : 0) +
                     " kmax " + std::to_string(eff.kmax) + " adaptive " + std::to_string(eff.adaptive ? 1 : 0) + "\n";
            hello.identity_hash = gs::fnv1a64(ident);
        }
        hello.capacity = opt.capacity;
        hello.slots = opt.slots;
        hello.max_rows = opt.max_rows;
        hello.step_rows = gs::kMaxRows;
        hello.world = opt.world;
        hello.mtp = mtp_on ? 1U : 0U;
        hello.mtp_kmax = eff.kmax;
        const bool g2_on = mtp_on && serve_g2 && eng.verify_multi_supported();
        if (mtp_on && serve_g2 && !g2_on)
            std::printf("rank %u: batched self-speculation off (the attention seam verifies one slot per call)\n", opt.rank);
        hello.g2 = g2_on ? 1U : 0U;
        EngineOps ops{eng, {eff.kmax, eff.adaptive, eff.kmax, opt.capacity},
                      prefill_mode == "gemv" ? sp::BootstrapPolicy::Chunked : sp::BootstrapPolicy::Reference, {}, {}, 0, {}, 0, {}, {}};
        ops.control.rank = opt.rank;
        ops.spec.assign(opt.slots, {});
        ops.hp_row.assign(opt.slots, 0);
        ops.world = opt.world;
        if (!import_listen.empty()) {
            if (import_key_file.empty() || import_model_sha.empty())
                throw std::invalid_argument("--import-listen needs --import-key and --import-model-sha (the RTX producer's model)");
            // The hand-off's node identity is this engine's: r1 and r1b prefill prompts in R1 (the RTX producer's
            // state); r1b-all prefills in R1b, so its state would differ: refused. --import-node, if given, must agree.
            if (id != "r1" && id != "r1b")
                throw std::invalid_argument("--import-listen: identity " + id + " cannot import RTX prefill state (r1 / r1b only)");
            if (!import_node.empty() && import_node != id)
                throw std::invalid_argument("--import-node " + import_node + " differs from this engine's identity " + id);
            import_node = id;
            const auto key = gi::read_key(import_key_file);
            const auto model = gi::unhex(import_model_sha);
            const unsigned rank = opt.rank;
            const std::string ep = import_listen, node = import_node;
            const unsigned version = import_dsa_once ? 2U : 1U;
            // One hand-off per import_expect: the header must equal this rank's expectation (model, prompt hash,
            // frontier, layout, rank, node); the frames' payloads, in order, are the Engine::restore_state bytes.
            ops.stager = std::make_unique<gs::ImportStager>([key, model, rank, ep, node, version](std::uint32_t iid, std::span<const std::uint32_t> t,
                                                                                                const std::atomic<bool>& cancel,
                                                                                                const std::function<void()>& listening) {
                gi::Begin b;
                b.version = version;
                b.rank = rank;
                b.frontier = t.size() - 1;
                b.total = t.size();
                b.model = model;
                b.prompt = gi::hash(std::as_bytes(t));
                b.producer = "glm-pf-rtx-native-v1";
                b.node = node;
                try {
                    auto st = gi::receive(ep, key, b, &cancel, listening);
                    std::vector<std::uint8_t> out;
                    out.reserve(static_cast<std::size_t>(st->raw_bytes));
                    st->apply([&](const gi::Frame&, std::span<const std::uint8_t> p) { out.insert(out.end(), p.begin(), p.end()); });
                    // the authenticated Begin's common fields (transaction included, rank excluded): both ranks compare
                    // it before either applies its state
                    const auto d = gi::hash(st->begin.encode(true));
                    gs::Binding binding{};
                    std::copy(std::begin(d.bytes), std::end(d.bytes), binding.begin());
                    return gs::StagedPayload(std::move(out), binding);
                } catch (const std::exception& e) {
                    if (!cancel.load()) std::fprintf(stderr, "rank %u: import %u not staged: %s\n", rank, iid, e.what());
                    throw;
                }
            });
            ops.staged = [&ops](std::uint32_t iid) { return ops.stager->payload(iid); };
            ops.staged_binding = [&ops](std::uint32_t iid) { return ops.stager->binding(iid); };
            if (!mtp_on || !opt.mtp)
                throw std::invalid_argument("--import-listen needs the engine loaded with its MTP layer (46 layers): the RTX hand-off "
                                            "carries the MTP state; this engine resolved MTP off");
            if (comm) {  // import commit votes: the communicator's host path, their own control layer ("GV")
                ops.vote = [&comm](std::span<const std::byte> out, std::span<std::byte> in, std::uint64_t seq) {
                    strix::dist::Tag tag;
                    tag.kind = static_cast<std::uint32_t>(strix::dist::TagKind::control);
                    tag.layer = 0x47560000U;
                    tag.request = seq;
                    comm->exchange_host(out, in, tag);
                };
            }
            ops.dsa_once = import_dsa_once;
            // The full per-rank component list of a v1 hand-off (= this rank's restore_state layout) at frontier P.
            ops.state_layout = [rank](std::uint64_t frontier, std::uint64_t total) {
                gi::Begin b;
                b.rank = rank;
                b.frontier = frontier;
                b.total = total;
                std::vector<gs::StateComponent> out;
                for (const auto& c : gi::layout(b)) out.push_back({c.id >= 2, c.rows * c.stride});
                return out;
            };
            if (import_dsa_once) {
                if (!comm) throw std::invalid_argument("--import-dsa-once needs the pair (--world 2)");
                // The relay rides the communicator's host path (its own control layer) inside import_state's replay:
                // between engine calls, so the transport is idle. Chunks fit the communicator's buffers.
                ops.relay_chunk = std::min<std::size_t>(comm->max_bytes(), std::size_t{32} << 20);
                ops.relay = [&comm](std::span<const std::byte> out, std::span<std::byte> in, std::uint64_t seq) {
                    strix::dist::Tag tag;
                    tag.kind = static_cast<std::uint32_t>(strix::dist::TagKind::control);
                    tag.layer = 0x47440000U;  // "GD": import DSA relay
                    tag.request = seq;
                    comm->exchange_host(out, in, tag);
                };
            }
            std::printf("rank %u: import receiver on %s (node identity %s%s)\n", rank, ep.c_str(), node.c_str(),
                        import_dsa_once ? ", DSA once: rank 0 relays the DSA components" : "");
        }

        // RTX import settings every rank must share: a mismatch refuses at hello, before serving.
        if (!import_listen.empty()) {
            hello.import_flags = 1U | (import_dsa_once ? 2U : 0U);
            hello.import_wire = import_dsa_once ? 2U : 1U;
            hello.import_model_hash = gs::attested_model_hash(import_model_sha);  // canonical: the parsed 32 bytes
        }
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        std::signal(SIGPIPE, SIG_IGN);
        if (opt.rank != 0) {
            std::printf("rank %u: following (model %016llx identity %016llx)\n", opt.rank,
                        static_cast<unsigned long long>(hello.model_hash), static_cast<unsigned long long>(hello.identity_hash));
            gs::follow(ops, ch, hello, opt.slots, opt.capacity);
            std::printf("rank %u: STOPPED after %llu rounds\n", opt.rank, static_cast<unsigned long long>(ch.round()));
            return 0;
        }
        ch.hello(hello);
        std::printf("rank 0: hello OK (model %016llx identity %016llx)\n", static_cast<unsigned long long>(hello.model_hash),
                    static_cast<unsigned long long>(hello.identity_hash));

        const auto tokenizer = sv::BpeTokenizer::load_hf_json(opt.model_dir + "/tokenizer.json");
        sv::GenerationQueue queue(64);
        gs::LeadConfig lc;
        lc.slots = opt.slots;
        lc.capacity = opt.capacity;
        lc.batch = !serial;
        lc.prefix_cache_bytes = prefix_cache_bytes;
        lc.prefix_identity = gs::fnv1a64(&hello.model_hash, sizeof(hello.model_hash), hello.identity_hash);
        lc.prefill_chunk = prefill_chunk;
        lc.prefill_chunk_mtp = prefill_chunk_mtp;
        lc.mtp = mtp_on;
        lc.mtp_cycles = mtp_cycles;
        lc.g2 = g2_on;
        lc.kmax = eff.kmax;
        lc.heartbeat_seconds = heartbeat_ms / 1000.0;
        struct RtxState { std::mutex m; std::set<std::uint32_t> ids; std::string last_reply; };
        auto rtx_state = std::make_shared<RtxState>();  // failed RTX requests; the producer's last receipt (gate)
        lc.state_digest = import_gate_on;
        if (!rtx_prefill.empty()) {
            if (import_listen.empty() || import_peer.empty())
                throw std::invalid_argument("--rtx-prefill needs --import-listen (this rank) and --import-peer (rank 1's endpoint)");
            if (opt.world != 2) throw std::invalid_argument("--rtx-prefill needs --world 2 (the hand-off is the TP2 per-rank state)");
            const auto key = gi::read_key(import_key_file);
            static const char* hexd = "0123456789abcdef";
            std::string key_hex;
            for (const auto b : key) { key_hex += hexd[(static_cast<unsigned>(b) >> 4) & 15U]; key_hex += hexd[static_cast<unsigned>(b) & 15U]; }
            auto failed = rtx_state;
            const std::string rtx = rtx_prefill, ep0 = import_listen, ep1 = import_peer, node = import_node;
            // The RTX producer prefills and sends each rank its state (POST /v1/prefill, the hand-off of service.cpp);
            // it answers only after both ranks sealed, so the call runs off the lead's thread. Any non-200 marks the
            // import failed and the slot prefills itself at once.
            lc.rtx_prefill = [failed, rtx, ep0, ep1, node, key_hex, dsa_once = import_dsa_once](std::span<const std::uint32_t> prompt, std::uint32_t iid) {
                std::string body = "{\"tokens\":[";
                for (std::size_t i = 0; i < prompt.size(); ++i) body += (i ? "," : "") + std::to_string(prompt[i]);
                body += "],\"identity\":\"glm-pf-rtx-native-v1\",\"handoff\":{\"rank0\":\"" + ep0 + "\",\"rank1\":\"" + ep1 + "\",\"key\":\"" +
                        key_hex + "\",\"node_identity\":\"" + node + "\"" + (dsa_once ? ",\"dsa_once\":true" : "") + "}}";
                std::thread([failed, rtx, body, iid] {
                    int status = 0;
                    std::string reply;
                    try { status = http_post(rtx, "/v1/prefill", body, 600, &reply); } catch (...) {}
                    {
                        std::lock_guard<std::mutex> l(failed->m);
                        failed->last_reply = reply;
                    }
                    if (status != 200) {
                        std::fprintf(stderr, "import %u: RTX /v1/prefill answered %d; falling back to the pair prefill\n", iid, status);
                        std::lock_guard<std::mutex> l(failed->m);
                        failed->ids.insert(iid);
                    }
                }).detach();
                return true;
            };
            lc.rtx_failed = [failed](std::uint32_t iid) { std::lock_guard<std::mutex> l(failed->m); return failed->ids.count(iid) != 0; };
            lc.import_min_tokens = import_min_tokens;
            lc.import_timeout_s = import_timeout_s;
            lc.import_dsa_once = import_dsa_once;
            std::printf("rank 0: RTX prefill route via %s for prompts >= %llu tokens\n", rtx.c_str(),
                        static_cast<unsigned long long>(import_min_tokens));
        }
        gs::Lead<EngineOps> lead(ops, ch, queue, lc, g_stop);
        if (selftest > 0) lead.set_trace(true);

        if (import_gate_on) {
            // Import gate: per prompt file (raw little-endian u32 token ids), per decode mode (plain; MTP when on), arm A
            // through the RTX import and arm B with its own prefill, one at a time. I1: A's tokens == B's. I0: right after
            // each arm's prefill both ranks digest the slot's state (per component); A == B on each rank.
            std::vector<std::string> files;
            for (std::size_t a = 0, b; a <= import_gate.size(); a = b + 1) {
                b = import_gate.find(',', a);
                if (b == std::string::npos) b = import_gate.size();
                if (b > a) files.push_back(import_gate.substr(a, b - a));
            }
            if (files.empty()) return import_gate_empty(import_gate);  // a suite of zero prompts qualifies nothing
            bool all_equal = true, gate_covered = true;
            unsigned gate_arms = 0;  // completed, covered A/B arm pairs (the verdict needs at least one)
            std::string driver_error;
            auto fnv_tokens = [](const std::vector<std::uint32_t>& t) { return gs::fnv1a64(t.data(), t.size() * 4); };
            auto print_coverage = [](const std::string& f, bool mtp, bool covered, const std::string& route_a, const std::string& fallback_a,
                                     std::uint64_t imported_a, std::size_t want, const std::string& route_b) {
                std::printf("import-gate %s %s coverage %s (arm A route %s fallback %s imported %llu of %zu, arm B route %s)\n",
                            f.c_str(), mtp ? "mtp" : "plain", covered ? "OK" : "FAIL", route_a.c_str(), fallback_a.c_str(),
                            static_cast<unsigned long long>(imported_a), want, route_b.c_str());
            };
            std::thread driver([&] {
                try {
                    for (const auto& f : files) {
                        std::ifstream in(f, std::ios::binary);
                        if (!in) throw std::runtime_error("--import-gate: cannot open " + f);
                        std::string raw((std::istreambuf_iterator<char>(in)), {});
                        if (raw.size() % 4 != 0 || raw.size() < 8) throw std::runtime_error("--import-gate: " + f + " is not u32 tokens");
                        std::vector<std::uint32_t> prompt(raw.size() / 4);
                        std::memcpy(prompt.data(), raw.data(), raw.size());
                        if (prompt.size() + import_gate_tokens + 1 >= opt.capacity)
                            throw std::runtime_error("--import-gate: " + f + " + tokens does not fit --capacity");
                        if (prompt.size() < lc.import_min_tokens)
                            throw std::runtime_error("--import-gate: " + f + " is shorter than --import-min-tokens");
                        std::vector<bool> modes{false};
                        if (mtp_on) modes.push_back(true);
                        for (const bool mtp : modes) {
                            lead.set_mtp(mtp);
                            struct Arm {
                                std::vector<std::uint32_t> tokens;
                                std::vector<std::uint32_t> d0, d1;
                                std::uint64_t frontier = 0;
                            } arm[2];
                            std::string arm_route[2] = {"-", "-"}, arm_fallback[2] = {"-", "-"};
                            std::uint64_t arm_imported[2] = {0, 0};
                            bool arm_ok[2] = {false, false};
                            for (int k = 0; k < 2; ++k) {
                                const bool routed = k == 0;
                                lead.set_route(routed);
                                {
                                    std::lock_guard<std::mutex> l(rtx_state->m);
                                    rtx_state->last_reply.clear();
                                }
                                (void)lead.take_digests();
                                sv::GenerationJob j;
                                j.prompt = prompt;
                                j.max_tokens = import_gate_tokens;
                                auto t = queue.submit(std::move(j));
                                bool done = false;
                                while (!done) {
                                    auto g = t->wait(std::chrono::milliseconds(200), done);
                                    arm[k].tokens.insert(arm[k].tokens.end(), g.begin(), g.end());
                                }
                                const auto st = t->stats();
                                for (auto& d : lead.take_digests()) {
                                    arm[k].d0 = std::move(d.rank0);
                                    arm[k].d1 = std::move(d.rank1);
                                    arm[k].frontier = d.frontier;
                                }
                                std::string route = "{}";
                                for (const auto& [name, value] : st.engine_extra)
                                    if (name == "prefill_route") route = value;
                                const auto rj = sv::parse_json(route);
                                auto num = [](const sv::Json& o, const char* key) {
                                    const auto* v = o.is_object() ? o.find(key) : nullptr;
                                    return v && v->is_number() ? v->as_double() : -1.0;
                                };
                                auto str = [](const sv::Json& o, const char* key) {
                                    const auto* v = o.is_object() ? o.find(key) : nullptr;
                                    return v && v->is_string() ? v->as_string() : std::string("-");
                                };
                                arm_route[k] = str(rj, "route");
                                arm_fallback[k] = str(rj, "fallback");
                                arm_imported[k] = st.imported_tokens;
                                arm_ok[k] = st.error_code.empty() && st.finish != "cancelled" && st.finish != "error";
                                const auto& tk = arm[k].tokens;
                                std::string first8;
                                for (std::size_t i = 0; i < tk.size() && i < 8; ++i) first8 += (i ? "," : "") + std::to_string(tk[i]);
                                const double tps = st.decode_ms > 0 && tk.size() > 1 ? 1000.0 * static_cast<double>(tk.size() - 1) / st.decode_ms : 0.0;
                                std::printf("import-gate %s %s arm %s (%s): route %s fallback %s, tokens %zu fnv64 %016llx first8 [%s], "
                                            "TTFT %.1f ms, decode %.2f tok/s\n",
                                            f.c_str(), mtp ? "mtp" : "plain", routed ? "A" : "B", routed ? "rtx import" : "own prefill",
                                            str(rj, "route").c_str(), str(rj, "fallback").c_str(), tk.size(),
                                            static_cast<unsigned long long>(fnv_tokens(tk)), first8.c_str(), st.prefill_ms, tps);
                                if (str(rj, "route") == "rtx") {
                                    std::string reply;
                                    {
                                        std::lock_guard<std::mutex> l(rtx_state->m);
                                        reply = rtx_state->last_reply;
                                    }
                                    sv::Json pj;
                                    try { pj = sv::parse_json(reply); } catch (...) {}
                                    const double prod = num(pj, "producer_ms"), total = num(pj, "total_ms");
                                    std::printf("import-gate   TTFT split: announce->listening %.1f ms, rtx prefill %.1f ms, transfer tail %.1f ms "
                                                "(staged both %.1f ms after the request), restore %.1f ms, last-token prefill %.1f ms (rank 0), "
                                                "commit round %.1f ms, wire %.0f bytes\n",
                                                num(rj, "listen_ms"), prod, prod >= 0 && total >= 0 ? total - prod : -1.0, num(rj, "staged_ms"),
                                                ops.last_restore_ms, ops.last_token_ms, num(rj, "commit_ms"), num(pj, "wire_bytes"));
                                } else {
                                    std::printf("import-gate   TTFT split: own prefill %.1f ms\n", st.prefill_ms);
                                }
                            }
                            // Coverage: arm A must really have imported (RTX route, no fallback, every prompt token but the
                            // last restored, the request completed); otherwise A == B would only compare two own prefills.
                            const bool covered = arm_route[0] == "rtx" && arm_fallback[0] == "-" &&
                                                 arm_imported[0] == prompt.size() - 1 && arm_ok[0] && !arm[0].tokens.empty() &&
                                                 arm_route[1] != "rtx" && arm_ok[1] && !arm[1].tokens.empty();
                            print_coverage(f, mtp, covered, arm_route[0], arm_fallback[0], arm_imported[0], prompt.size() - 1, arm_route[1]);
                            gate_covered = gate_covered && covered;
                            if (covered) ++gate_arms;
                            const bool i1 = arm[0].tokens == arm[1].tokens && !arm[0].tokens.empty();
                            std::printf("import-gate %s %s I1 %s\n", f.c_str(), mtp ? "mtp" : "plain", i1 ? "EQUAL" : "DIFF");
                            // I0: per rank, A's components vs B's
                            const auto comps = [&] {
                                std::size_t total = 0;
                                for (std::size_t i = 2; i + 1 < arm[1].d0.size(); i += 4)
                                    total += static_cast<std::size_t>(arm[1].d0[i]) | (static_cast<std::size_t>(arm[1].d0[i + 1]) << 32);
                                auto c = state_components(total, opt.world, arm[1].frontier);
                                if (c.empty()) c.push_back({"all", total, false});
                                return c;
                            }();
                            bool i0 = true;
                            for (int r = 0; r < 2; ++r) {
                                const auto& a = r == 0 ? arm[0].d0 : arm[0].d1;
                                const auto& b = r == 0 ? arm[1].d0 : arm[1].d1;
                                std::string first = "none";
                                unsigned ndiff = 0, mtp_diff = 0;
                                const bool shape = !a.empty() && a.size() == b.size() && a.size() == comps.size() * 4 &&
                                                   arm[0].frontier == arm[1].frontier;
                                for (std::size_t c = 0; shape && c < comps.size(); ++c) {
                                    const auto o = static_cast<std::ptrdiff_t>(4 * c);
                                    if (std::equal(a.begin() + o, a.begin() + o + 4, b.begin() + o)) continue;
                                    // a plain own prefill does not write the MTP layer: compared, reported, not gating
                                    if (comps[c].mtp && !mtp) {
                                        ++mtp_diff;
                                        continue;
                                    }
                                    if (ndiff++ == 0) first = comps[c].name;
                                }
                                const bool eq = shape && ndiff == 0;
                                i0 = i0 && eq;
                                std::printf("import-gate %s %s I0 rank %d: %s (%zu components, frontier %llu, %u differ, first %s%s)\n",
                                            f.c_str(), mtp ? "mtp" : "plain", r, eq ? "EQUAL" : "DIFF", comps.size(),
                                            static_cast<unsigned long long>(arm[0].frontier), ndiff, shape ? first.c_str() : "digest shape mismatch",
                                            mtp_diff ? (", MTP layer differs: " + std::to_string(mtp_diff) + " components, not gating in plain mode").c_str() : "");
                            }
                            std::printf("import-gate %s %s I0 %s\n", f.c_str(), mtp ? "mtp" : "plain", i0 ? "EQUAL" : "DIFF");
                            all_equal = all_equal && i0 && i1;
                        }
                    }
                } catch (const std::exception& e) {
                    driver_error = e.what();
                    all_equal = false;
                }
                g_stop.store(true);
            });
            try {
                lead.loop();
            } catch (...) {
                queue.shutdown();
                driver.join();
                throw;
            }
            queue.shutdown();
            driver.join();
            if (!driver_error.empty()) std::printf("import-gate error: %s\n", driver_error.c_str());
            // FAIL (exit 4) whenever an arm A did not import: equality alone is not a qualification of the import.
            const bool pass = gate_covered && gate_arms > 0 && driver_error.empty();
            std::printf("rank 0: import-gate %s, coverage %s, verdict %s, STOPPED\n", all_equal ? "EQUAL" : "DIFF",
                        gate_covered ? "OK" : "FAIL", !pass ? "FAIL" : all_equal ? "PASS" : "DIFF");
            return !pass ? 4 : all_equal ? 0 : 3;
        }

        if (selftest > 0) {
            const auto prompts = selftest_prompts(*tokenizer, selftest);
            gs::SelftestResult result;
            std::string driver_error;
            std::thread driver([&] {
                try {
                    gs::SelftestHooks hooks;
                    if (mtp_on) hooks.set_mtp = [&lead](bool on) { lead.set_mtp(on); };
                    if (g2_on) hooks.set_g2 = [&lead](bool on) { lead.set_g2(on); };
                    hooks.take_trace = [&lead] { return lead.take_trace(); };
                    hooks.set_prefill_chunk = [&lead](unsigned c) { lead.set_prefill_chunk(c); };
                    hooks.progress = [&lead] { return lead.progress(); };
                    hooks.prefill_chunk = prefill_chunk;
                    hooks.extra_chunks = {256U};
                    hooks.skip_basic = selftest_skip_basic;
                    // interleave levels: up to 4 multi-chunk prompts (batched mode, chunking on, room in the context)
                    const std::uint64_t room = opt.capacity > selftest_tokens + 64U ? opt.capacity - selftest_tokens - 64U : 0U;
                    const std::size_t long_tokens = static_cast<std::size_t>(std::min<std::uint64_t>(selftest_long, room));
                    if (!serial && prefill_chunk > 0 && selftest_long > 0 && long_tokens > prefill_chunk)
                        hooks.long_prompts = selftest_long_prompts(*tokenizer, std::min(4U, opt.slots), long_tokens);
                    for (const auto& lp : hooks.long_prompts)
                        if (lp.size() + selftest_tokens >= opt.capacity) throw std::runtime_error("selftest: long prompt does not fit");
                    if (!hooks.long_prompts.empty())
                        std::printf("selftest long prompts: %zu x ~%zu tokens (chunk %u)\n", hooks.long_prompts.size(),
                                    hooks.long_prompts[0].size(), prefill_chunk);
                    if (selftest_prefix) {
                        // --selftest-prefix-long L: multi-chunk first turns (L tokens each) so the timing lines measure a real restore.
                        const auto prefix_prompts =
                            selftest_prefix_long > 0 ? selftest_long_prompts(*tokenizer, std::min(selftest, opt.slots), selftest_prefix_long) : prompts;
                        result.exact = gs::run_prefix_selftest(queue, prefix_prompts, selftest_tokens, gs::default_stop_ids(), tokenizer->encode("\nPlease continue briefly.", false),
                            [&lead] { lead.clear_prefix_cache(); }, [&lead] { return lead.stats().prefix_hits; },
                            [](const std::string& line) { std::printf("%s\n", line.c_str()); });
                    } else result = gs::run_selftest(queue, prompts, selftest_tokens, gs::default_stop_ids(), serial ? 1U : opt.slots,
                                              [](const std::string& line) { std::printf("%s\n", line.c_str()); }, hooks);
                } catch (const std::exception& e) {
                    driver_error = e.what();
                    result.exact = false;
                }
                g_stop.store(true);
            });
            try {
                lead.loop();
            } catch (...) {
                queue.shutdown();
                driver.join();
                throw;
            }
            queue.shutdown();
            driver.join();
            if (!driver_error.empty()) std::printf("selftest error: %s\n", driver_error.c_str());
            if (g_signal.load() != 0) std::printf("selftest interrupted by signal %d\n", g_signal.load());
            const auto st = lead.stats();
            std::printf("selftest engine: steps %llu rows %llu (R1..8: %llu %llu %llu %llu %llu %llu %llu %llu) gap steps %llu, "
                        "prefills %llu (%llu tokens in %llu calls, %.2f s), step time %.2f s, rounds %llu\n",
                        static_cast<unsigned long long>(st.steps), static_cast<unsigned long long>(st.rows),
                        static_cast<unsigned long long>(st.rows_hist[1]), static_cast<unsigned long long>(st.rows_hist[2]),
                        static_cast<unsigned long long>(st.rows_hist[3]), static_cast<unsigned long long>(st.rows_hist[4]),
                        static_cast<unsigned long long>(st.rows_hist[5]), static_cast<unsigned long long>(st.rows_hist[6]),
                        static_cast<unsigned long long>(st.rows_hist[7]), static_cast<unsigned long long>(st.rows_hist[8]),
                        static_cast<unsigned long long>(st.gap_steps), static_cast<unsigned long long>(st.prefills),
                        static_cast<unsigned long long>(st.prefill_tokens), static_cast<unsigned long long>(st.prefill_chunks),
                        st.prefill_seconds, st.step_seconds,
                        static_cast<unsigned long long>(st.rounds));
            std::printf("selftest c1-mtp: %s, bootstraps %llu, enters %llu, leaves %llu, rounds %llu, tokens %llu in %.2f s (%.2f tok/s on "
                        "the MTP path)\n", mtp_on ? "on" : "off", static_cast<unsigned long long>(st.mtp_bootstraps),
                        static_cast<unsigned long long>(st.mtp_enters), static_cast<unsigned long long>(st.mtp_leaves),
                        static_cast<unsigned long long>(st.mtp_rounds), static_cast<unsigned long long>(st.mtp_tokens), st.mtp_seconds,
                        st.mtp_seconds > 0 ? static_cast<double>(st.mtp_tokens) / st.mtp_seconds : 0.0);
            std::printf("selftest g2: %s, cycles %llu rows %llu (%.2f rows/cycle, %llu windows), drafts %llu accepted %llu, drafter "
                        "extension calls %llu, tokens %llu in %.2f s (%.2f tok/s on the batched-spec path), handover plain steps "
                        "%llu, re-seeds %llu\n",
                        g2_on ? "on" : "off", static_cast<unsigned long long>(st.g2_cycles), static_cast<unsigned long long>(st.g2_rows),
                        st.g2_cycles ? static_cast<double>(st.g2_rows) / static_cast<double>(st.g2_cycles) : 0.0,
                        static_cast<unsigned long long>(st.g2_windows), static_cast<unsigned long long>(st.g2_drafts),
                        static_cast<unsigned long long>(st.g2_accepted), static_cast<unsigned long long>(st.g2_draft_calls),
                        static_cast<unsigned long long>(st.g2_tokens), st.g2_seconds,
                        st.g2_seconds > 0 ? static_cast<double>(st.g2_tokens) / st.g2_seconds : 0.0,
                        static_cast<unsigned long long>(st.g2_plain), static_cast<unsigned long long>(st.g2_reseeds));
            std::printf("rank 0: selftest %s, STOPPED\n", result.exact ? "EXACT" : "FAIL");
            return result.exact ? 0 : 3;
        }

        sv::ApiConfig config;
        config.model_id = model_id;
        config.default_max_tokens = max_tokens;
        config.http_slots = serial ? 1U : opt.slots;
        config.default_thinking = true;  // the GLM generation prompt ends inside <think>
        config.sampling_policy = "greedy";
        config.chat_renderer = [](const sv::ChatTemplateInput& in) { return sv::render_glm_chat(in); };
        config.output_parser = [](const std::string& t, bool thinking, const sv::Json& tools) { return sv::parse_glm_output(t, thinking, tools); };
        config.tool_call_marker = sv::glm_tool_marker;
        config.stop_tokens.assign(sv::glm_stop_tokens.begin(), sv::glm_stop_tokens.end());
        config.engine_description = "GLM-5.3-Flash TP2 (" + std::to_string(opt.world) + " ranks), " + std::to_string(opt.slots) +
                                    " slots, " + (serial ? "serial" : "batched") + " decode" +
                                    (mtp_on ? ", MTP self-speculation at C1" : "") +
                                    (g2_on ? " and in the batch" : "") + ", greedy, exact per request" +
                                    (rtx_prefill.empty() ? std::string()
                                                         : std::string("; prompts of >= ") + std::to_string(import_min_tokens) +
                                                               " tokens may take the RTX prefill route (FAST identity glm-pf-rtx-native-v1, wire v" +
                                                               (import_dsa_once ? "2" : "1") + ", not bitwise the pair's own prefill; "
                                                               "each response's prefill_route names its route)");
        if (!api_key_file.empty()) {
            auto key = file_text(api_key_file);
            while (!key.empty() && (key.back() == '\n' || key.back() == '\r' || key.back() == ' ')) key.pop_back();
            if (key.empty()) throw std::invalid_argument("--api-key-file is empty");
            config.api_key = key;
        }
        config.status_extra = [&lead, &opt, serial](sv::Json& status) {
            const auto st = lead.stats();
            sv::Json g = sv::Json::object();
            g.set("slots", sv::Json(static_cast<std::uint64_t>(opt.slots)));
            g.set("batch", sv::Json(!serial));
            g.set("active", sv::Json(st.active));
            g.set("steps", sv::Json(st.steps));
            g.set("gap_steps", sv::Json(st.gap_steps));
            status.set("glm_serve", std::move(g));
        };
        config.metrics_extra = [&lead, prefix_cache_bytes]() {
            const auto st = lead.stats();
            std::ostringstream m;
            auto counter = [&](const char* name, const char* help, auto v) {
                m << "# HELP " << name << ' ' << help << "\n# TYPE " << name << " counter\n" << name << ' ' << v << '\n';
            };
            counter("strix_glm_serve_steps_total", "Decode steps (one Engine::step for every decoding request).", st.steps);
            counter("strix_glm_serve_rows_total", "Rows of the decode steps.", st.rows);
            counter("strix_glm_serve_gap_steps_total", "Steps whose slots were not consecutive (per-row KDA path).", st.gap_steps);
            counter("strix_glm_serve_prefills_total", "Prompt prefills (each stalls decode for its duration).", st.prefills);
            counter("strix_glm_serve_prefill_tokens_total", "Prompt tokens prefilled.", st.prefill_tokens);
            counter("strix_glm_serve_prefill_seconds_total", "Wall time of the prefills.", st.prefill_seconds);
            counter("strix_glm_serve_step_seconds_total", "Wall time of the decode steps.", st.step_seconds);
            counter("strix_glm_serve_control_rounds_total", "Control rounds with rank 1.", st.rounds);
            counter("strix_glm_serve_mtp_rounds_total", "C1 MTP control rounds (verify cycles of the only request).", st.mtp_rounds);
            counter("strix_glm_serve_mtp_tokens_total", "Tokens emitted on the C1 MTP path.", st.mtp_tokens);
            counter("strix_glm_serve_mtp_seconds_total", "Wall time on the C1 MTP path.", st.mtp_seconds);
            counter("strix_glm_serve_mtp_enters_total", "Entries into the MTP path from plain steps.", st.mtp_enters);
            counter("strix_glm_serve_mtp_leaves_total", "Exits from the MTP path (decoder closed at a cycle boundary).", st.mtp_leaves);
            counter("strix_glm_serve_g2_cycles_total", "Batched self-speculation cycles (one multi-slot verify each).", st.g2_cycles);
            counter("strix_glm_serve_g2_rows_total", "Rows of the batched self-speculation verifies.", st.g2_rows);
            counter("strix_glm_serve_g2_drafts_total", "Drafts verified in batched self-speculation.", st.g2_drafts);
            counter("strix_glm_serve_g2_accepted_total", "Drafts accepted in batched self-speculation.", st.g2_accepted);
            if (prefix_cache_bytes) {
                counter("strix_glm_serve_prefix_hits_total", "Prefix cache hits.", st.prefix_hits);
                counter("strix_glm_serve_prefix_misses_total", "Prefix cache misses.", st.prefix_misses);
                counter("strix_glm_serve_prefix_stores_total", "Paired prefix stores.", st.prefix_stores);
                counter("strix_glm_serve_prefix_evictions_total", "Prefix LRU evictions.", st.prefix_evictions);
                m << "# TYPE strix_glm_serve_prefix_bytes gauge\nstrix_glm_serve_prefix_bytes " << st.prefix_bytes << '\n';
            }
            counter("strix_glm_serve_imports_total", "Prompts restored from the RTX prefill state (per request: strix.prefill_route).", st.imports);
            counter("strix_glm_serve_import_fallbacks_total", "RTX imports abandoned for the pair's own prefill.", st.import_fallbacks);
            counter("strix_glm_serve_imported_tokens_total", "Prompt tokens restored from the RTX prefill state.", st.imported_tokens);
            counter("strix_glm_serve_import_wait_seconds_total", "Announce to both ranks staged, imported prompts.", st.import_wait_seconds);
            m << "# HELP strix_glm_serve_active Requests holding a slot.\n# TYPE strix_glm_serve_active gauge\nstrix_glm_serve_active "
              << st.active << '\n';
            return m.str();
        };
        sv::ApiService api(config, tokenizer, queue, opt.capacity, false);
        sv::HttpServerConfig hc;
        hc.bind_address = host;
        hc.port = static_cast<std::uint16_t>(port);
        sv::HttpServer http(hc, [&](const sv::HttpRequest& r, sv::HttpResponseWriter& w) { api.handle(r, w); });
        http.listen();
        std::thread http_thread([&] { http.serve(); });
        api.set_state("ready");
        std::printf("READY rank 0 http://%s:%u model %s slots %u context %llu batch %s\n", host.c_str(), http.port(), model_id.c_str(),
                    opt.slots, static_cast<unsigned long long>(opt.capacity), serial ? "serial" : "on");
        try {
            lead.loop();
        } catch (...) {
            api.set_state("failed");
            queue.shutdown();
            http.stop();
            http_thread.join();
            throw;
        }
        queue.shutdown();
        http.stop();
        http_thread.join();
        std::printf("rank 0: STOPPED\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-glm5-serve error: %s\n", e.what());
        return 2;
    }
}
