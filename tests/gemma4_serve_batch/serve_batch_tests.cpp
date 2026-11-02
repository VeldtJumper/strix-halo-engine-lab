// Serve continuous batching through engine::Api with a row-invariant fake backend and the real tokenizer: concurrent
// HTTP chat requests (greedy, seeded sampling, thinking-budget continuation, streaming, a client that disconnects
// mid-stream, prompts long enough for many-row prefill chunks) must return the same generated ids as each request
// served alone with STRIX_G4_SERVE_BATCH=0.
#include "gemma4/engine/api.hpp"
#include "gemma4/serve/tokenizer.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace g = strix::gemma4;
namespace e = g::engine;
using Json = strix::serve::Json;

#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ")+#x+" line "+std::to_string(__LINE__)); } while (false)

namespace {
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
std::uint64_t mix(std::uint64_t h, std::uint32_t t) {
    h ^= t + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    h *= 0xff51afd7ed558ccdULL;
    return h ^ (h >> 33U);
}
struct Stream;
struct Txn final : g::KvTxn {
    Stream* s;
    std::uint64_t start;
    std::vector<std::uint32_t> tokens;
    Txn(Stream* stream, std::uint64_t f) : s(stream), start(f) {}
    std::uint64_t frontier() const override { return start; }
    void reserve_rows(std::size_t) override {}
    void commit(std::size_t n) override;
};
struct Stream final : e::StreamState {
    std::vector<std::uint64_t> hashes{0x1234567ULL};
    std::unique_ptr<g::KvTxn> begin() override { return std::make_unique<Txn>(this, hashes.size()-1); }
    std::uint64_t frontier() const override { return hashes.size()-1; }
    bool poisoned = false; // as a device stream after a failed pass: no canonical state to hash
    std::string digest() const override {
        if (poisoned) throw std::runtime_error("failed stream has no canonical state");
        return std::to_string(hashes.back());
    }
    std::unique_ptr<e::StreamState> fork(std::uint32_t) const override {
        auto out = std::make_unique<Stream>();
        out->hashes = hashes;
        return out;
    }
};
void Txn::commit(std::size_t n) {
    if (n > tokens.size()) throw std::runtime_error("fake commit");
    for (std::size_t i = 0; i < n; ++i) s->hashes.push_back(mix(s->hashes.back(), tokens[i]));
}
// Each row's logits/taps depend only on its own stream's committed tokens and the row's own token chain.
struct Backend final : e::Backend {
    g::ModelConfig c;
    std::vector<std::uint32_t> words; // ordinary text tokens the fake model emits
    std::uint32_t stop = 1;
    // A prefill whose rows differ from decode rows (as an FP32 prefill beside an int8-activation decode): prompt rows
    // it writes carry a salt into every later hash, so any change of which rows it writes changes the outputs.
    bool same = true;
    std::uint32_t salt = 0;
    std::atomic<std::uint64_t> passes{0}, max_rows{0};
    std::atomic<int> fail_in{-1}; // > 0: the fail_in-th next pass fails and poisons its streams
    std::uint32_t pick(std::uint64_t h) const { return h % 37 == 0 ? stop : words[h % words.size()]; }
    std::uint64_t row_hash(Txn& tx, std::uint64_t position, std::uint32_t token) const {
        if (position != tx.start + tx.tokens.size()) throw std::runtime_error("fake rows must be contiguous");
        tx.tokens.push_back(token);
        std::uint64_t h = tx.s->hashes.back();
        for (const auto t : tx.tokens) h = mix(h, t);
        return h;
    }
    std::unique_ptr<e::StreamState> create_stream() override { return std::make_unique<Stream>(); }
    g::ForwardResult forward_rows(std::span<const g::RowPlan> plans) override {
        ++passes;
        if (fail_in > 0 && --fail_in == 0) {
            for (const auto& p : plans) static_cast<Txn*>(p.transaction)->s->poisoned = true;
            throw std::runtime_error("injected device failure");
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500)); // lets concurrent requests overlap
        if (plans.size() > max_rows) max_rows = plans.size();
        g::ForwardResult f;
        f.rows = plans.size(); f.vocab = c.vocab; f.tap_width = 5 * c.hidden;
        f.logits.assign(f.rows * c.vocab, 0.0F); f.taps.assign(f.rows * f.tap_width, 0.5F);
        for (const auto& p : plans) {
            const auto h = row_hash(*static_cast<Txn*>(p.transaction), p.position, p.token);
            f.logits[p.output_row * c.vocab + pick(h)] = 8.0F;
            f.logits[p.output_row * c.vocab + pick(mix(h, 7))] += 7.5F;
        }
        return f;
    }
    std::size_t prefill_chunk_rows() const override { return 16; }
    bool prefill_matches_decode() const override { return same; }
    e::PrefillRows prefill_rows(g::KvTxn& transaction, std::span<const std::uint32_t> tokens, bool hidden,
                                bool taps) override {
        auto& tx = static_cast<Txn&>(transaction);
        for (std::size_t i = 0; i < tokens.size(); ++i) (void)row_hash(tx, tx.start + tx.tokens.size(), tokens[i] ^ salt);
        e::PrefillRows rows;
        rows.tap_width = 5 * c.hidden;
        if (taps) rows.taps.assign(tokens.size() * rows.tap_width, 0.5F);
        if (hidden) rows.hidden.assign(tokens.size() * c.hidden, 0.0F);
        return rows;
    }
    const g::ModelConfig& config() const override { return c; }
    Json identity() const override { return Json::object(); }
};
class Writer final : public strix::serve::HttpResponseWriter {
public:
    int status = 0;
    std::string body;
    std::size_t gone_after = SIZE_MAX, writes = 0; // the client disconnects after this many SSE writes
    void send(int s, std::string_view, std::string_view b, const std::vector<std::pair<std::string,std::string>>&) override {
        status = s; body = b;
    }
    bool begin_stream(int s, std::string_view, const std::vector<std::pair<std::string,std::string>>&) override {
        status = s; return true;
    }
    bool write(std::string_view chunk) override {
        if (client_gone()) return false;
        ++writes; body += chunk; return true;
    }
    void end_stream() override {}
    bool client_gone() const override { return writes >= gone_after; }
};
struct Case {
    std::string name, body;
    bool streaming = false;
    std::size_t gone_after = SIZE_MAX;
};
std::string chat(const std::string& text, const std::string& extra) {
    return R"({"messages":[{"role":"user","content":")" + text + R"("}])" + extra + "}";
}
// Turn 1 of each conversation, then turn 2 (turn 1, its answer text, a follow-up): generated ids + finish of both
// turns, and the prompt rows turn 2 restored from the prefix cache.
std::pair<std::vector<std::string>, std::uint64_t> conversations(Backend& backend,
        std::shared_ptr<const g::serve::Tokenizer> tokenizer, const std::vector<std::string>& questions) {
    e::Api api(backend, tokenizer);
    std::vector<std::string> out(2 * questions.size());
    std::uint64_t cached = 0;
    std::vector<std::thread> threads;
    std::mutex m;
    for (std::size_t i = 0; i < questions.size(); ++i)
        threads.emplace_back([&, i] {
            strix::serve::HttpRequest req; req.method = "POST"; req.path = "/v1/chat/completions";
            req.body = chat(questions[i], R"(,"max_tokens":24)");
            Writer first; api.handle(req, first);
            CHECK(first.status == 200);
            const auto j = strix::serve::parse_json_unique(first.body);
            const auto answer = j.find("choices")->as_array()[0].find("message")->find("content")->as_string();
            auto turn2 = Json(Json::Object{{"messages", Json(Json::Array{
                Json(Json::Object{{"role", "user"}, {"content", questions[i]}}),
                Json(Json::Object{{"role", "assistant"}, {"content", answer}}),
                Json(Json::Object{{"role", "user"}, {"content", "And then?"}})})}, {"max_tokens", std::uint64_t(24)}});
            req.body = turn2.dump();
            Writer second; api.handle(req, second);
            CHECK(second.status == 200);
            const auto k = strix::serve::parse_json_unique(second.body);
            std::lock_guard lock(m);
            out[2 * i] = j.find("strix")->find("generated_ids")->dump();
            out[2 * i + 1] = k.find("strix")->find("generated_ids")->dump();
            for (const auto& phase : k.find("strix")->find("phases")->as_array())
                if (const auto* p = phase.find("serve_prefix")) cached += std::uint64_t(p->find("cached_tokens")->as_int());
        });
    for (auto& t : threads) t.join();
    return {out, cached};
}
// generated_ids + finish of a finished request (the final SSE chunk carries the same receipt).
std::string outcome(const Writer& w, bool streaming) {
    if (!streaming) {
        const auto j = strix::serve::parse_json_unique(w.body);
        return j.find("strix")->find("generated_ids")->dump() + " " + j.find("strix")->find("budget_continued")->dump() +
               " " + j.find("choices")->as_array()[0].find("finish_reason")->dump();
    }
    const auto at = w.body.rfind("\"strix\"");
    CHECK(at != std::string::npos);
    const auto begin = w.body.rfind("data: ", at) + 6, end = w.body.find("\n\n", at);
    const auto j = strix::serve::parse_json_unique(w.body.substr(begin, end - begin));
    return j.find("strix")->find("generated_ids")->dump() + " " + j.find("strix")->find("budget_continued")->dump() +
           " " + j.find("choices")->as_array()[0].find("finish_reason")->dump();
}
std::map<std::string, std::string> serve(Backend& backend, std::shared_ptr<const g::serve::Tokenizer> tokenizer,
                                         const std::vector<Case>& cases, bool concurrent) {
    e::Api api(backend, tokenizer);
    std::vector<Writer> writers(cases.size());
    std::vector<std::thread> threads;
    const auto run = [&](std::size_t i) {
        strix::serve::HttpRequest req; req.method = "POST"; req.path = "/v1/chat/completions"; req.body = cases[i].body;
        writers[i].gone_after = cases[i].gone_after;
        api.handle(req, writers[i]);
    };
    for (std::size_t i = 0; i < cases.size(); ++i) {
        if (concurrent) threads.emplace_back(run, i);
        else run(i);
    }
    for (auto& t : threads) t.join();
    std::map<std::string, std::string> out;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        CHECK(writers[i].status == 200);
        out[cases[i].name] = cases[i].gone_after == SIZE_MAX ? outcome(writers[i], cases[i].streaming) : "cancelled";
    }
    return out;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--tokenizer") {
            std::cout << "SKIP: --tokenizer TOKENIZER_JSON required\n";
            return 0;
        }
        const auto tokenizer = g::serve::Tokenizer::load(argv[2]);
        Backend backend;
        backend.c.hidden = 8; backend.c.vocab = std::uint32_t(tokenizer->vocab_size()); backend.c.max_positions = 4096;
        backend.words = tokenizer->encode(" the quick brown fox jumps over a lazy dog while seven small birds sing");
        CHECK(backend.words.size() > 8);
        std::string long_text;
        for (int i = 0; i < 12; ++i) long_text += "Paragraph " + std::to_string(i) + " repeats a plain sentence about rivers and hills. ";
        const std::vector<Case> cases = {
            {"greedy", chat("Hello there.", R"(,"max_tokens":48)")},
            {"long", chat(long_text, R"(,"max_tokens":40)")},
            {"long2", chat(long_text + "Another one.", R"(,"max_tokens":24,"stream":true)"), true},
            {"sampled", chat("Tell me a story.", R"(,"max_tokens":40,"temperature":0.7,"seed":5)")},
            {"budget", chat("Think first.", R"(,"max_tokens":40,"enable_thinking":true,"thinking_budget":8)")},
            {"stream", chat("Stream this.", R"(,"max_tokens":56,"stream":true)"), true},
            {"short", chat("Hi.", R"(,"max_tokens":4)")},
            {"gone", chat("Leave early.", R"(,"max_tokens":64,"stream":true)"), true, 3},
            {"ninth", chat("Wait for a slot.", R"(,"max_tokens":32)")},
            {"tenth", chat(long_text, R"(,"max_tokens":16,"temperature":0.9,"seed":11)")},
        };
        set_env("STRIX_G4_SERVE_BATCH", "0");
        const auto alone = serve(backend, tokenizer, cases, false);
        const auto serial_passes = backend.passes.load();
        CHECK(backend.max_rows <= 8);
        set_env("STRIX_G4_SERVE_BATCH", nullptr);
        backend.max_rows = 0;
        const auto batched = serve(backend, tokenizer, cases, true);
        for (const auto& [name, ids] : alone) {
            if (batched.at(name) != ids) throw std::runtime_error("batched ids differ for " + name + ": " + batched.at(name) + " vs " + ids);
        }
        CHECK(backend.max_rows > 1);       // requests really shared target passes
        CHECK(alone.at("budget").find(" true ") != std::string::npos); // the thinking budget ran its continuation
        set_env("STRIX_G4_SERVE_PREFILL_PER_STEP", "0");
        const auto unpaced = serve(backend, tokenizer, cases, true);
        set_env("STRIX_G4_SERVE_PREFILL_PER_STEP", nullptr);
        for (const auto& [name, ids] : alone) CHECK(unpaced.at(name) == ids);
        for (const char* rows : {"4", "0"}) { // prompt chunks cut to 4 rows while others decode, and whole chunks
            set_env("STRIX_G4_SERVE_PREFILL_ROWS", rows);
            const auto budget = serve(backend, tokenizer, cases, true);
            for (const auto& [name, ids] : alone) CHECK(budget.at(name) == ids);
        }
        set_env("STRIX_G4_SERVE_PREFILL_ROWS", nullptr);
        // Prefill rows != decode rows: the budget must keep the stream-alone chunk ends (a 4-row budget would move
        // which prompt tails go through decode passes and change the salted outputs).
        backend.same = false; backend.salt = 0x40000000U;
        set_env("STRIX_G4_SERVE_BATCH", "0");
        const auto alone_mixed = serve(backend, tokenizer, cases, false);
        set_env("STRIX_G4_SERVE_BATCH", nullptr);
        const char* const budgets[] = {"4", nullptr};
        for (const char* rows : budgets) {
            set_env("STRIX_G4_SERVE_PREFILL_ROWS", rows);
            const auto budget = serve(backend, tokenizer, cases, true);
            for (const auto& [name, ids] : alone_mixed) CHECK(budget.at(name) == ids);
        }
        set_env("STRIX_G4_SERVE_PREFILL_ROWS", nullptr);
        backend.same = true; backend.salt = 0;
        // A failed pass fails its requests cleanly (no state hash of a poisoned stream) and the server keeps serving.
        {
            e::Api api(backend, tokenizer);
            backend.fail_in = 3;
            strix::serve::HttpRequest req; req.method = "POST"; req.path = "/v1/chat/completions";
            req.body = chat("Hello there.", R"(,"max_tokens":16)");
            Writer failed; api.handle(req, failed);
            CHECK(failed.status == 500 && failed.body.find("injected device failure") != std::string::npos);
            backend.fail_in = -1;
            Writer next; api.handle(req, next);
            CHECK(next.status == 200);
        }
        std::cout << "PASS serve failure: a failed pass fails its request, the next request is served\n";
        std::string doc;
        for (int i = 0; i < 20; ++i) doc += "Line " + std::to_string(i) + " of a shared report about bridges. ";
        const std::vector<std::string> questions = {"Hello there.", doc + "Summarize it.", "Write a haiku.", doc + "List the lines."};
        set_env("STRIX_G4_SERVE_PREFIX", "0");
        const auto plain = conversations(backend, tokenizer, questions);
        set_env("STRIX_G4_SERVE_PREFIX", nullptr);
        const auto cached = conversations(backend, tokenizer, questions);
        CHECK(plain.second == 0);
        CHECK(cached.first == plain.first);  // both turns' ids equal with and without the prefix cache
        CHECK(cached.second > 0);            // turn 2 restored its conversation prefix
        std::cout << "PASS serve prefix cache: " << questions.size() << " two-turn conversations == cache off, "
                  << cached.second << " prompt rows restored\n";
        // Prefill rows != decode rows: only prefill-written states at a chunk boundary are reused, never the decode-written
        // end of a request, so both turns still equal the cache off.
        backend.same = false; backend.salt = 0x40000000U;
        set_env("STRIX_G4_SERVE_PREFIX", "0");
        const auto plain_mixed = conversations(backend, tokenizer, questions);
        set_env("STRIX_G4_SERVE_PREFIX", nullptr);
        const auto cached_mixed = conversations(backend, tokenizer, questions);
        backend.same = true; backend.salt = 0;
        CHECK(cached_mixed.first == plain_mixed.first);
        CHECK(cached_mixed.second > 0 && cached_mixed.second % backend.prefill_chunk_rows() == 0);
        std::cout << "PASS serve prefix cache, prefill != decode rows: == cache off, " << cached_mixed.second
                  << " prompt rows restored at chunk boundaries\n";
        std::cout << "PASS serve batching: " << cases.size() << " concurrent requests == alone (greedy, sampled, budget "
                  << "continuation, streaming, disconnect, prefill chunks paced/unpaced, row budget 4/64/whole, prefill != decode rows); serial passes " << serial_passes
                  << ", peak rows " << backend.max_rows << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FAIL " << ex.what() << "\n";
        return 1;
    }
}
