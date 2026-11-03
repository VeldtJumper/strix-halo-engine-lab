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
//   --mtp-cycles N (verify cycles per control round on the C1 MTP path, default 1), --mtp-k K / --mtp-adaptive 0|1 /
//   --mtp-spec 0|1 as strix-glm5-rank; STRIX_GLM_SERVE_MTP=0 disables the C1 MTP path (plain decode at every C).
//   Batched self-speculation (G2, with MTP on): every decoding slot's window (anchor + adaptive drafts) in ONE multi-slot
//   verify of <= 8 rows when more than one request decodes; STRIX_GLM_SERVE_G2=0 = batched plain steps (the OFF switch).
//   Needs an attention seam with multi-slot verify (Engine::verify_multi_supported; off otherwise, said at startup).
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
#include "glm/serve/glm_serve_control.hpp"
#include "glm/serve/glm_serve_lead.hpp"
#include "glm/serve/glm_serve_selftest.hpp"
#include "glm/spec/glm_spec_run.hpp"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
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

std::atomic<bool> g_stop{false};
std::atomic<int> g_signal{0};
void on_signal(int sig) {
    g_signal.store(sig);
    g_stop.store(true);
}

// The engine calls of the control protocol (ge::Row and WireRow share their layout; converted field by field), the
// same on both ranks. The MTP state (drafter seed, open decoder, merged control state) lives here: every decision of
// the decoder comes from values merged across the ranks, so both ranks make the same engine calls.
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

    void plain_call() {
        if (dec) throw std::logic_error("engine call while an MTP decoder is open");
        seed.reset();
    }
    void reset(std::uint32_t slot) {
        plain_call();
        eng.reset(slot);
        spec.at(slot) = {};
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
        if (pos0 == 0 && last) {
            seed = sp::bootstrap(eng, slot, tokens, bootstrap, control);
            seed_slot = slot;
            seed_slot_spec(slot, *seed);
            return seed->y0;
        }
        if (bootstrap != sp::BootstrapPolicy::Chunked) throw std::logic_error("chunked MTP prefill needs the exact chunked prefill path");
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

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    try {
        if (argc < 2) {
            std::fprintf(stderr, "usage: %s MODEL_DIR --world 1|2 [--rank R --transport T] [--slots S] [--capacity C] [--serial] "
                                 "[--host H] [--port P] [--selftest N [--selftest-tokens M]]\n", argv[0]);
            return 2;
        }
        const auto t_start = std::chrono::steady_clock::now();
        ge::Options opt;
        opt.model_dir = argv[1];
        opt.slots = 8;
        std::string transport_spec, host = "127.0.0.1", model_id = "glm-5.3-flash", api_key_file;
        unsigned port = 8080, selftest = 0, selftest_tokens = 128, heartbeat_ms = 1000, mtp_cycles = 1, prefill_chunk = 512;
        unsigned selftest_long = 2048;
        bool prefill_chunk_mtp = true, selftest_skip_basic = false;
        bool serve_mtp = true;
        std::uint32_t max_tokens = 4096;
        bool serial = false, print_config = false;
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
        lc.prefill_chunk = prefill_chunk;
        lc.prefill_chunk_mtp = prefill_chunk_mtp;
        lc.mtp = mtp_on;
        lc.mtp_cycles = mtp_cycles;
        lc.g2 = g2_on;
        lc.kmax = eff.kmax;
        lc.heartbeat_seconds = heartbeat_ms / 1000.0;
        gs::Lead<EngineOps> lead(ops, ch, queue, lc, g_stop);
        if (selftest > 0) lead.set_trace(true);

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
                    result = gs::run_selftest(queue, prompts, selftest_tokens, gs::default_stop_ids(), serial ? 1U : opt.slots,
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
                                    (g2_on ? " and in the batch" : "") + ", greedy, exact per request";
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
        config.metrics_extra = [&lead]() {
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
