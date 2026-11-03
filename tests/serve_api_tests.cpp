#include "../apps/qwen_rotary_capabilities.hpp"
// HTTP API contract tests with a scripted engine (no GPU): OpenAI chat/completions (plain and
// SSE), tool calls, stop strings, Anthropic messages (plain and events), errors and API keys,
// two concurrent clients through the FIFO queue, and cancellation when a client disconnects.
#include "strix/serve/api.hpp"
#include "strix/serve/constraint_selection.hpp"

#include <algorithm>
#include <limits>
#include <cmath>
#include "strix/serve/deepseek_chat.hpp"
#include "../apps/state_cache_stats.hpp"
#include "../apps/hip_session_store.hpp"
#include "../apps/qwen_effective_config.hpp"
#include "strix/serve/http_server.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <iostream>
#include <netinet/in.h>
#include <mutex>
#include <set>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace strix::serve;
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
}

// Byte alphabet, a few merges and the chat specials (types: 1 normal, 3 control, 4 user-defined).
std::shared_ptr<const BpeTokenizer> tiny_tokenizer() {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    // GPT-2 byte alphabet in byte order.
    for (int b = 0; b < 256; ++b) {
        int cp = b;
        static int extra = 0;
        const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        if (!direct) cp = 256 + extra++;
        std::string s;
        if (cp < 0x80) s.push_back(static_cast<char>(cp));
        else { s.push_back(static_cast<char>(0xC0 | (cp >> 6))); s.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        tokens.push_back(s);
        types.push_back(1);
    }
    std::vector<std::string> merges{"H e", "He l", "Hel l", "Hell o"};
    tokens.push_back("He"); tokens.push_back("Hel"); tokens.push_back("Hell"); tokens.push_back("Hello");
    for (int i = 0; i < 4; ++i) types.push_back(1);
    for (const char* special : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) { tokens.push_back(special); types.push_back(3); }
    for (const char* special : {"<tool_call>", "</tool_call>", "<think>", "</think>"}) { tokens.push_back(special); types.push_back(4); }
    return BpeTokenizer::from_parts(std::move(tokens), std::move(types), merges, "qwen35");
}

class ScriptedEngine final : public GenerationEngine {
public:
    explicit ScriptedEngine(std::shared_ptr<const BpeTokenizer> t) : tokenizer_(std::move(t)), trie_(std::make_shared<TokenByteTrie>(*tokenizer_)) {}
    std::atomic<bool> budget_headers{false};
    bool supports_constraints() const override { return true; }
    bool supports_thinking_budget() const override { return true; }
    std::vector<std::string> constraint_execution_routes() const override { return {"serial-host"}; }
    GenerationStats generate(GenerationJob& job) override {
        last_retention.store(job.cache_retention);
        { const std::lock_guard lock(mutex_); prompts_.push_back({job.prompt, job.stable_prefix}); sampling_.push_back(job.sampling); }
        if (job.constraint && !job.constraint->program && !job.constraint->thinking) return scored(job);
        if (job.constraint && job.constraint->thinking) return thinking_budget(job);
        if (job.constraint) return constrained(job);
        if(job.session_id=="late_budget_refusal") {
            job.on_admitted();
            // The client acknowledges receipt of headers, so the late refusal is
            // deterministic even if the HTTP thread is delayed by the scheduler.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!budget_headers.load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if(job.session_id=="budget_refusal" || job.session_id=="late_budget_refusal") {
            GenerationStats stats; stats.error_status=507; stats.error_code="session_budget";
            stats.finish="admission_refused";
            throw GenerationRequestError(std::move(stats),"session capture exceeds host peak budget");
        }
        if (job.on_admitted) job.on_admitted();
        const auto prompt = tokenizer_->decode(job.prompt);
        const bool progress = prompt.find("PROGRESS") != std::string::npos;
        if (progress) {
            const auto total = static_cast<std::uint64_t>(job.prompt.size());
            for (std::uint64_t step = 0; step <= 4; ++step) {
                if (job.on_progress) job.on_progress({GenerationPhase::prefill, (total - 1) * step / 4, 1, total});
                std::this_thread::sleep_for(std::chrono::milliseconds(240));
            }
        }
        if (prompt.find("SLOW") != std::string::npos) std::this_thread::sleep_for(std::chrono::milliseconds(750));
        if (prompt.find("ENGINE_ERROR") != std::string::npos) throw std::runtime_error("scripted failure");
        std::string script = "Let me think.\n</think>\n\nHello there, friend.<|im_end|>";
        if (prompt.find("TOOL") != std::string::npos)
            script = "Need weather.\n</think>\n\nChecking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n"
                     "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>";
        if (prompt.find("LONG") != std::string::npos) { script.clear(); for (int i = 0; i < 400; ++i) script += "word "; }
        if (prompt.find("<think>\n\n</think>") != std::string::npos) script = "Plain answer.<|im_end|>";
        const auto ids = tokenizer_->encode(script, true);
        GenerationStats stats;
        stats.prompt_tokens = job.prompt.size();
        stats.sampling = "greedy";
        if (!job.session_id.empty()) stats.engine_extra.emplace_back("session_echo", Json(job.session_id).dump());
        std::size_t at = 0;
        while (at < ids.size()) {
            const auto n = std::min<std::size_t>(3U, ids.size() - at);
            std::vector<std::uint32_t> step(ids.begin() + static_cast<std::ptrdiff_t>(at), ids.begin() + static_cast<std::ptrdiff_t>(at + n));
            if (stats.generated_tokens + step.size() >= job.max_tokens) { step.resize(job.max_tokens - stats.generated_tokens); }
            bool stop = false;  // after the budget cut: a stop token beyond the budget is not generated
            for (std::size_t k = 0; k < step.size(); ++k)
                if (std::find(job.stop_tokens.begin(), job.stop_tokens.end(), step[k]) != job.stop_tokens.end()) { step.resize(k + 1U); stop = true; break; }
            stats.generated_tokens += step.size();
            ++stats.cycles;
            calls.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (!job.on_tokens(step)) { stats.finish = "cancelled"; cancelled.fetch_add(1); return stats; }
            if (progress && at == 0) std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (stop) { stats.finish = "stop"; return stats; }
            if (stats.generated_tokens >= job.max_tokens) { stats.finish = "length"; return stats; }
            at += n;
        }
        stats.finish = "length";
        return stats;
    }
    std::uint64_t context_tokens() const override { return 4096; }
    std::atomic<CacheRetention> last_retention{CacheRetention::server_policy};
    std::atomic<int> calls{0}, cancelled{0};
    using Captured = std::pair<std::vector<std::uint32_t>, std::uint32_t>;
    std::vector<SamplingParameters> sampling() { const std::lock_guard lock(mutex_); return sampling_; }
    std::vector<Captured> prompts() { const std::lock_guard lock(mutex_); return prompts_; }
private:
    GenerationStats scored(GenerationJob& job) {
        auto selection = std::make_shared<ConstraintSelection>(job.constraint, trie_, tokenizer_);
        ConstraintRowSelection row(selection); GenerationStats stats; stats.prompt_tokens = job.prompt.size();
        std::vector<float> logits(tokenizer_->vocab_size(), -100); logits['a'] = logits['b'] = 2; logits['c'] = -1;
        while (stats.generated_tokens < job.max_tokens) {
            row.prepare(); const auto id = row.select(logits); row.commit(); ++stats.generated_tokens;
            if (!job.on_tokens(std::span<const std::uint32_t>(&id, 1))) { stats.finish = "cancelled"; break; }
        }
        const auto text=tokenizer_->decode(job.prompt);
        if(text.find("ROUTE_MTP")!=std::string::npos)row.set_execution("mtp-host");
        if(text.find("ROUTE_OFF")!=std::string::npos)row.set_execution("serial-host","selection-mtp-off");
        stats.constraint = row.verdict(stats.finish); return stats;
    }
    GenerationStats thinking_budget(GenerationJob& job) {
        auto selection = std::make_shared<ConstraintSelection>(job.constraint, trie_, tokenizer_);
        ConstraintRowSelection row(selection);
        GenerationStats stats; stats.prompt_tokens = job.prompt.size();
        const auto prompt = tokenizer_->decode(job.prompt);
        const auto answer = tokenizer_->encode("{\"a\":\"ok\"}", false);
        std::size_t answer_at = 0;
        while (stats.generated_tokens < job.max_tokens && !selection->state().terminal) {
            const auto& state = selection->state();
            std::uint32_t desired = job.constraint->eos_ids.front();
            if (state.reasoning) {
                desired = 'w';
                if (prompt.find("NATURAL") != std::string::npos && state.reasoning_tokens == 2) desired = *job.constraint->end_thinking_id;
                if (prompt.find("REASON_EOS") != std::string::npos) desired = job.constraint->eos_ids.front();
            } else if (answer_at < answer.size()) desired = answer[answer_at++];
            std::vector<float> logits(tokenizer_->vocab_size(), -100.0F); logits[desired] = 100;
            row.prepare();
            const auto id = row.select(logits);
            row.commit();
            ++stats.generated_tokens;
            if (!job.on_tokens(std::span<const std::uint32_t>(&id, 1))) { stats.finish = "cancelled"; break; }
        }
        if (stats.finish != "cancelled") stats.finish = selection->state().terminal ? "stop" : "length";
        stats.constraint = row.verdict(stats.finish);
        return stats;
    }
    GenerationStats constrained(GenerationJob& job) {
        ConstraintSelection selector(job.constraint, trie_, tokenizer_);
        GenerationStats stats; stats.prompt_tokens = job.prompt.size();
        const auto prompt = tokenizer_->decode(job.prompt);
        std::vector<std::uint32_t> desired;
        if (job.constraint->starts_in_reasoning) {
            desired = tokenizer_->encode("why", false);
            desired.push_back(*job.constraint->end_thinking_id);
        }
        const auto answer = tokenizer_->encode(" \n{\"a\":\"<tool_call> </think> caf\xC3\xA9\"}", false);
        desired.insert(desired.end(), answer.begin(), answer.end());
        desired.push_back(job.constraint->eos_ids.front());
        try {
            for (std::size_t i = 0; stats.generated_tokens < job.max_tokens; ++i) {
                if (i == 1 && prompt.find("CONSTRAINT_CONTEXT") != std::string::npos) { stats.finish = "context"; break; }
                if ((prompt.find("EXECUTION_ANSWER") != std::string::npos && selector.verdict("length", "serial-host").accepting) ||
                    (prompt.find("EXECUTION_REASONING") != std::string::npos && i == 1)) {
                    stats.finish = "execution_failure";
                    stats.constraint = selector.verdict(stats.finish, "serial-host");
                    throw GenerationRequestError(std::move(stats), "injected execution failure");
                }
                auto prepared = selector.prepare();
                std::vector<float> logits(tokenizer_->vocab_size(), -100.0F);
                logits[desired.at(i)] = 10.0F;
                if (!prepared.state.reasoning) {
                    logits[tokenizer_->find("<tool_call>")] = 100.0F; // invalid control wins without masking
                    logits[job.constraint->eos_ids.front()] = 90.0F; // blocked until accepting
                    if (desired.at(i) == static_cast<std::uint32_t>('a')) logits['b'] = 10.0F;
                }
                if (prompt.find("CONSTRAINT_FAILURE") != std::string::npos) logits[0] = std::numeric_limits<float>::infinity();
                if (prompt.find("CONSTRAINT_REASON_EOS") != std::string::npos) logits[job.constraint->eos_ids.front()] = 200;
                const auto staged = selector.select(logits, prepared);
                selector.commit(staged); // commit before publishing the pending output token
                ++stats.generated_tokens; ++stats.cycles; calls.fetch_add(1);
                if (!job.on_tokens(std::span<const std::uint32_t>(&staged.token, 1))) { stats.finish = "cancelled"; break; }
                if (selector.state().terminal) { stats.finish = "stop"; break; }
            }
            stats.constraint = selector.verdict(stats.finish, "serial-host");
        } catch (const ConstraintError&) {
            stats.constraint = selector.verdict("constraint_failure", "serial-host");
        }
        return stats;
    }
    std::mutex mutex_;
    std::vector<Captured> prompts_;
    std::vector<SamplingParameters> sampling_;
    std::shared_ptr<const BpeTokenizer> tokenizer_;
    std::shared_ptr<const TokenByteTrie> trie_;
};

// Member access that names the missing key and the document instead of dereferencing null.
const Json& at(const Json& json, std::string_view key) {
    if (const auto* value = json.find(key)) return *value;
    throw std::runtime_error("missing key " + std::string(key) + " in " + json.dump());
}
struct Reply { int status = 0; std::string headers, body; };
// Plain HTTP/1.1 client; decodes chunked bodies. `abort_after` closes the socket after that many body bytes.
Reply http(std::uint16_t port, const std::string& method, const std::string& path, const std::string& body,
    const std::string& extra = {}, std::size_t abort_after = 0, std::atomic<bool>* headers_seen = nullptr) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) { ::close(fd); return {}; }
    const auto request = method + " " + path + " HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\n" + extra +
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    (void)::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
    std::string raw;
    char buffer[4096];
    for (;;) {
        const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) break;
        raw.append(buffer, static_cast<std::size_t>(n));
        if (headers_seen && raw.find("\r\n\r\n") != std::string::npos) headers_seen->store(true);
        if (abort_after && raw.size() > abort_after) break;
    }
    ::close(fd);
    Reply out;
    const auto end = raw.find("\r\n\r\n");
    if (end == std::string::npos) return out;
    out.headers = raw.substr(0, end);
    out.status = std::atoi(raw.c_str() + 9);
    auto rest = raw.substr(end + 4U);
    if (out.headers.find("Transfer-Encoding: chunked") == std::string::npos) { out.body = rest; return out; }
    for (std::size_t at = 0; at < rest.size();) {
        const auto line = rest.find("\r\n", at);
        if (line == std::string::npos) break;
        const auto size = std::stoul(rest.substr(at, line - at), nullptr, 16);
        if (!size) break;
        out.body += rest.substr(line + 2U, size);
        at = line + 2U + size + 2U;
    }
    return out;
}
std::vector<Json> sse_data(const std::string& body) {
    std::vector<Json> out;
    for (std::size_t at = 0; (at = body.find("data: ", at)) != std::string::npos;) {
        const auto end = body.find("\n\n", at);
        const auto payload = body.substr(at + 6U, end - at - 6U);
        if (payload != "[DONE]") out.push_back(parse_json(payload));
        at = end;
    }
    return out;
}
}  // namespace

// The pull interface of batching engines: FIFO order, jobs cancelled while queued finished and skipped,
// tokens published into the ticket, totals per finished job, nothing after shutdown.
// Compare complete caller-visible traces against the pre-rebase selection rules.
// The model deliberately does not call the shared selector under test.
void test_queue_caller_orders() {
    for (unsigned caller = 0; caller < 4; ++caller) {
        for (unsigned seed = 0; seed < 81; ++seed) {
            struct Model { unsigned id, rows, age; bool cancelled; };
            std::vector<Model> model;
            std::vector<std::shared_ptr<GenerationTicket>> tickets;
            GenerationQueue queue;
            unsigned cancellations = 0;
            auto submit = [&](unsigned rows, bool cancelled) {
                GenerationJob job;
                job.prompt.resize(rows);
                auto ticket = queue.submit(std::move(job));
                check(ticket != nullptr, "trace submission");
                if (cancelled) ticket->cancel();
                model.push_back({static_cast<unsigned>(tickets.size()), rows, 0, cancelled});
                tickets.push_back(ticket);
            };
            unsigned sizes = seed;
            for (unsigned i = 0; i < 4; ++i) {
                submit(1U + (sizes % 3U) * 20U, ((seed + i) % 5U) == 0);
                sizes /= 3U;
            }
            for (unsigned step = 0; step < 40 && !model.empty(); ++step) {
                if (step < 20) submit(1, step % 3U == 0);
                unsigned expected = static_cast<unsigned>(tickets.size());
                while (!model.empty()) {
                    std::size_t pick = 0;
                    const bool short_first = caller >= 2;
                    bool cancelled_pick = false;
                    if (caller == 3) {
                        const auto it = std::find_if(model.begin(), model.end(), [](const Model& e) { return e.cancelled; });
                        if (it != model.end()) {
                            pick = static_cast<std::size_t>(it - model.begin());
                            cancelled_pick = true;
                        }
                    }
                    if (short_first && !cancelled_pick) {
                        for (std::size_t i = 0; i < model.size(); ++i) {
                            if (model[i].age >= 8U) { pick = i; break; }
                            if (model[i].rows < model[pick].rows) pick = i;
                        }
                        for (std::size_t i = 0; i < pick; ++i) ++model[i].age;
                    }
                    const auto chosen = model[pick];
                    model.erase(model.begin() + static_cast<std::ptrdiff_t>(pick));
                    if (chosen.cancelled) { ++cancellations; continue; }
                    expected = chosen.id;
                    break;
                }
                auto taken = caller == 0 ? queue.take(false) : caller == 1 ? queue.take(false, false) :
                    caller == 2 ? queue.take(false, true) : queue.take(false, QueueSelection::short_first);
                check(static_cast<bool>(taken) == (expected < tickets.size()), "caller trace presence");
                if (taken) {
                    check(taken->ticket == tickets[expected], "caller trace matches original arrival/size/aging/cancellation order");
                    check(taken->ticket->queue_position() == 0, "taken position resets");
                    queue.finish(*taken, {});
                }
                check(queue.snapshot().cancelled == cancellations, "caller trace cancellation completion order");
                check(queue.snapshot().queued == model.size() && queue.snapshot().running == 0, "caller trace accounting");
                for (std::size_t i = 0; i < model.size(); ++i)
                    check(tickets[model[i].id]->queue_position() == i, "caller trace waiting arrival positions");
            }
            check(model.empty(), "caller trace drained");
        }
        GenerationQueue idle;
        unsigned controls = 0;
        idle.set_idle_control([&] { ++controls; });
        idle.wake_control();
        auto empty = caller == 0 ? idle.take(true) : caller == 1 ? idle.take(true, false) :
            caller == 2 ? idle.take(true, true) : idle.take(true, QueueSelection::short_first);
        check(!empty && controls == 1, "all callers preserve idle control wake");
        idle.shutdown();
    }
    // Switching APIs must share the same aging debt and the same bound.
    GenerationQueue mixed;
    GenerationJob long_job;
    long_job.prompt.resize(100);
    const auto owner = mixed.submit(std::move(long_job));
    for (unsigned i = 0; i < 9; ++i) {
        GenerationJob short_job;
        short_job.prompt.resize(1);
        const auto newcomer = mixed.submit(std::move(short_job));
        auto taken = i % 2U == 0 ? mixed.take(false, true) : mixed.take(false, QueueSelection::short_first);
        check(taken && taken->ticket == (i < 8 ? newcomer : owner), "both APIs share one overtake counter and bound");
        if (taken) mixed.finish(*taken, {});
    }
    auto last = mixed.take(false);
    check(last.has_value(), "mixed APIs retain unselected newcomer");
    if (last) mixed.finish(*last, {});

}

void test_batched_queue() {
    GenerationQueue small(1);
    check(small.submit({}) && !small.submit({}), "explicit small queue retains rejection");
    GenerationQueue queue(8);
    check(!queue.has_queued_work(), "empty queue hint");
    const auto job = [](std::uint32_t token) { GenerationJob j; j.prompt = {token, token + 1U}; j.cache_retention=CacheRetention::no_store; return j; };
    const auto t1 = queue.submit(job(1)), t2 = queue.submit(job(3)), t3 = queue.submit(job(5));
    check(t1 && t2 && t3, "batched queue submit");
    check(queue.has_queued_work(), "queued work hint");
    check(t1->queue_position() == 0U && t2->queue_position() == 1U && t3->queue_position() == 2U,
        "initial position counts waiting jobs ahead");
    t2->cancel();
    auto a = queue.take(false);
    check(a && a->ticket == t1 && a->job.prompt.front() == 1U, "batched queue takes the oldest job");
    check(t3->queue_position() == 1U && t3->submitted_queue_position() == 2U,
        "updated position excludes running job and preserves submission snapshot");
    const auto t4 = queue.submit(job(9));
    check(t4->submitted_queue_position() == 2U, "submit excludes running job");
    t4->cancel();
    check(a->job.cache_retention==CacheRetention::no_store,"retention survives batch queue move");
    auto b = queue.take(false);
    check(b && b->ticket == t3, "batched queue skips a job cancelled while queued");
    bool done = false;
    static_cast<void>(t2->wait(std::chrono::milliseconds(0), done));
    check(done && t2->stats().finish == "cancelled", "batched queue finishes the cancelled job");
    check(!queue.take(false), "batched queue is empty");
    check(!queue.has_queued_work(), "running jobs do not count as queued work");
    const std::uint32_t tokens[] = {7, 8};
    check(a->job.on_tokens(tokens), "batched job publishes");
    done = false;
    const auto got = t1->wait(std::chrono::milliseconds(0), done);
    check(got.size() == 2U && got[0] == 7U && !done, "batched job tokens reach the ticket");
    GenerationStats stats;
    stats.generated_tokens = 2;
    stats.finish = "stop";
    queue.finish(*a, stats);
    queue.finish(*b, {}, "engine failure");
    const auto snap = queue.snapshot();
    check(snap.completed == 1U && snap.failed == 1U && snap.cancelled == 2U && snap.running == 0U && snap.queued == 0U &&
          snap.generated_tokens == 2U, "batched queue totals");
    static_cast<void>(t1->wait(std::chrono::milliseconds(0), done));
    check(done && t1->stats().finish == "stop" && t3->error() == "engine failure", "batched tickets finished");
    queue.shutdown();
    check(!queue.take(true) && queue.stopping(), "batched queue after shutdown");
    check(!queue.has_queued_work(), "shutdown queue hint");
}

// Destroying a cancelled job's token callback acts as a deterministic producer between
// cleanup iterations. It stays armed until take returns, with a hard cap so regressions fail.
void test_queue_cancel_replenishment() {
    for (const auto selection : {QueueSelection::fifo, QueueSelection::short_first}) {
        for (const bool wait : {false, true}) {
            GenerationQueue queue;
            GenerationJob live;
            live.prompt.resize(100);
            const auto aged = queue.submit(std::move(live));
            for (unsigned i = 0; i < kMaxQueueOvertakes; ++i) {
                GenerationJob shorter;
                shorter.prompt.resize(1);
                const auto ticket = queue.submit(std::move(shorter));
                auto t = queue.take(false, QueueSelection::short_first);
                check(t && t->ticket == ticket, "age the persistent live request");
                if (t) queue.finish(*t, {});
            }
            unsigned produced = 0;
            bool producing = true;
            std::function<void()> replenish;
            replenish = [&] {
                if (!producing) return;
                if (produced == 32) {
                    producing = false;
                    GenerationJob escape;
                    escape.prompt.resize(1);
                    check(queue.submit(std::move(escape)) != nullptr, "cap replenishment without blocking a waiting take");
                    return;
                }
                ++produced;
                GenerationJob cancelled;
                cancelled.prompt.resize(1);
                auto trigger = std::shared_ptr<void>(nullptr, [&](void*) { replenish(); });
                cancelled.on_tokens = [trigger](std::span<const std::uint32_t>) { return false; };
                const auto ticket = queue.submit(std::move(cancelled));
                check(ticket != nullptr, "replenishing producer submitted");
                if (ticket) ticket->cancel();
            };
            replenish();
            replenish();
            auto t = queue.take(wait, selection);
            producing = false;
            check(t && t->ticket == aged, "aged admission returns control under replenishing cancellations");
            const bool short_first = selection == QueueSelection::short_first;
            check(produced == (short_first ? 4U : 2U) && queue.snapshot().cancelled == (short_first ? 2U : 0U),
                  "short-first cancellation precedence and FIFO arrival order are preserved within the batch");
            if (t) queue.finish(*t, {});
            check(!queue.take(false, selection) && queue.snapshot().queued == 0, "drain remaining cancellations");
            produced = 0;
            producing = true;
            replenish();
            const auto before = queue.snapshot().cancelled;
            auto empty = queue.take(wait, selection);
            producing = false;
            check(!empty && produced == 2 && queue.snapshot().cancelled == before + 1 && queue.snapshot().queued == 1,
                  "a cancellation-only batch returns control even when wait is true");
            if (empty) queue.finish(*empty, {});
            check(queue.has_queued_work(), "cancelled replenishment still needs cleanup after nullopt");
            check(!queue.take(false, selection) && !queue.has_queued_work(),
                  "retry drains the remaining cancellation without a queued-work spin");
            // The lead can now decode, heartbeat or check shutdown before taking again.
            queue.shutdown();
            check(queue.stopping() && queue.snapshot().queued == 0 && queue.snapshot().running == 0,
                  "scheduler regains control and can shut down");
        }
    }
}

// A fake target owns the predicted frontier, just like the HIP ticket. The
// production serial adapter must stage, replace that frontier, then commit before
// publication; aborting a row must not change the committed cursor.
void test_constraint_row(const std::shared_ptr<const BpeTokenizer>& tokenizer) {
    const auto trie=std::make_shared<TokenByteTrie>(*tokenizer);
    auto spec=std::make_shared<ConstraintSpec>();
    spec->program=std::make_shared<ConstraintProgram>(compile_json_object());
    spec->eos_ids={tokenizer->find("<|im_end|>")};
    const auto selection=std::make_shared<ConstraintSelection>(spec,trie,tokenizer);
    auto adapter=std::make_shared<ConstraintRowSelection>(selection);
    std::vector<float> row(tokenizer->vocab_size(),-100.0F);
    row['x']=100;row['{']=10;
    std::uint32_t pending='x', consumed=0;
    const auto execute=[&](std::uint32_t input) {
        consumed=input;
        // Target finite first-max predicts x; the host callback replaces it.
        std::uint32_t predicted='x';
        predicted=adapter->select(row);
        return predicted;
    };
    auto first=execute(7);
    check(first=='{' && !selection->prepare().mask.contains('}'), "terminal row stages constrained ticket without commit");
    adapter->commit();pending=first;
    check(adapter->verdict("length").reason=="max_output_tokens", "budget one reports incomplete first token");
    row['{']=-100;row['}']=10;
    adapter->prepare();auto next=execute(pending);
    check(consumed=='{' && next=='}' && !selection->state().cursor.complete(), "serial target consumes constrained pending token before staging successor");
    adapter->commit();pending=next;
    check(selection->state().cursor.complete() && adapter->verdict("context").reason=="context", "serial commit advances exact output frontier");
    check(adapter->verdict("cancelled").reason=="cancelled", "row cancellation receipt");
    const auto execution = adapter->verdict("execution_failure");
    check(execution.status=="failed" && execution.reason=="execution_failure" && execution.accepting && execution.phase=="answer",
        "execution failure snapshots accepting committed row before EOS");
    row[spec->eos_ids.front()]=200;
    adapter->prepare();next=execute(pending);adapter->commit();
    check(next==spec->eos_ids.front() && adapter->verdict("stop").status=="complete", "serial EOS completes only after commit");
    // A dropped callback/state never leaks into a reused slot sharing this trie.
    std::weak_ptr<ConstraintRowSelection> retired=adapter;adapter.reset();
    check(retired.expired(), "request row adapter released at finish");
    auto fresh=std::make_shared<ConstraintSelection>(spec,trie,tokenizer);
    {
        ConstraintRowSelection aborted(fresh);
        row.assign(row.size(),-100);row['{']=10;
        check(aborted.select(row)=='{', "aborted row stages token");
    }
    check(!fresh->prepare().mask.contains('}'), "aborted row leaves next request frontier unchanged");
    ConstraintRowSelection failed(fresh);
    row['x']=std::numeric_limits<float>::quiet_NaN();
    bool refused=false;
    try {(void)failed.select(row);} catch(const ConstraintError&) {refused=true;}
    check(refused && failed.verdict("constraint_failure").status=="failed" && !fresh->prepare().mask.contains('}'),
        "row refusal preserves committed state and failed receipt");
}

void test_constraint_request_failure() {
    class FailingRequest final : public GenerationEngine {
    public:
        std::uint64_t context_tokens() const override {return 32;}
        GenerationStats generate(GenerationJob& job) override {
            GenerationStats stats;
            if(job.prompt.front()==1U) {
                stats.finish="constraint_failure";
                stats.constraint=ConstraintVerdict{"failed",false,"answer","constraint_failure","serial-host"};
                throw GenerationRequestError(std::move(stats),"empty constraint mask");
            }
            stats.finish="stop";
            return stats;
        }
    } engine;
    GenerationQueue queue;
    GenerationJob failed;failed.prompt={1};
    GenerationJob healthy;healthy.prompt={2};
    const auto first=queue.submit(std::move(failed)),second=queue.submit(std::move(healthy));
    std::thread worker([&]{queue.run(engine);});
    for(const auto& ticket:{first,second}) {
        bool done=false;
        while(!done) (void)ticket->wait(std::chrono::milliseconds(50),done);
    }
    queue.shutdown();worker.join();
    const auto stats=first->stats();
    check(first->error()=="empty constraint mask" && stats.finish=="constraint_failure" && stats.constraint &&
        stats.constraint->status=="failed" && stats.constraint->execution=="serial-host", "single queue retains failure receipt and error");
    check(second->error().empty() && second->stats().finish=="stop", "constraint refusal does not poison next queued request");
}

void test_constraint_late_cancellation() {
    for (const bool batched : {false, true}) for (const bool reasoning : {false, true}) {
        GenerationQueue queue;
        GenerationJob job; job.prompt = {1};
        auto ticket = queue.submit(std::move(job));
        GenerationStats stats; stats.finish = "stop";
        stats.constraint = ConstraintVerdict{"complete", !reasoning, reasoning ? "reasoning" : "answer", "", "serial-host"};
        if (batched) {
            auto taken = queue.take(false);
            ticket->cancel();
            queue.finish(*taken, stats);
        } else {
            class LateCancel final : public GenerationEngine {
            public:
                std::uint64_t context_tokens() const override { return 32; }
                GenerationStats generate(GenerationJob&) override {
                    ticket->cancel(); queue->shutdown(); return stats;
                }
                GenerationQueue* queue;
                std::shared_ptr<GenerationTicket> ticket;
                GenerationStats stats;
            } engine;
            engine.queue = &queue; engine.ticket = ticket; engine.stats = stats;
            queue.run(engine);
        }
        const auto result = ticket->stats();
        check(result.finish == "cancelled" && result.constraint && result.constraint->status == "incomplete" &&
            result.constraint->reason == "cancelled" && result.constraint->phase == stats.constraint->phase &&
            result.constraint->accepting == stats.constraint->accepting && result.constraint->execution == "serial-host",
            "late cancellation reconciles single/batched stats while preserving frontier");
    }
}

void test_constraint_selection(const std::shared_ptr<const BpeTokenizer>& tokenizer) {
    const auto trie = std::make_shared<TokenByteTrie>(*tokenizer);
    auto spec = std::make_shared<ConstraintSpec>();
    spec->program = std::make_shared<ConstraintProgram>(compile_json_object());
    spec->eos_ids = {tokenizer->find("<|im_end|>")}; spec->end_thinking_id = tokenizer->find("</think>");
    spec->identity = "json-object";
    const auto throws = [](const auto& operation) { try { operation(); } catch (const ConstraintError&) { return true; } return false; };
    ConstraintSelection selector(spec, trie, tokenizer);
    std::vector<float> row(tokenizer->vocab_size(), -100.0F);
    row['{'] = 1; row[spec->eos_ids.front()] = 100;
    auto first = selector.select(row, selector.prepare());
    check(first.token == '{' && !selector.state().cursor.complete(), "first position masks EOS without committing");
    check(!selector.prepare().mask.contains('}'), "staged brace has not advanced live cursor");
    auto branch = selector.prepare(first.state);
    check(branch.mask.contains('}'), "branch has staged cursor");
    row['{'] = -100; row['}'] = 1;
    auto second = selector.select(row, branch);
    check(second.state.cursor.complete() && !selector.state().cursor.complete(), "branch accepts without advancing live cursor");
    selector.commit(second);
    check(selector.state().cursor.complete(), "accepted prefix commits correction token");
    check(throws([&] { selector.commit(second); }) && throws([&] { selector.commit(first); }), "duplicate and stale commits refused");
    auto eos = selector.select(row, selector.prepare());
    check(eos.token == spec->eos_ids.front(), "accepting boundary admits EOS"); selector.commit(eos);
    check(selector.verdict("stop", "serial-host").status == "complete", "committed EOS completes constraint");
    check(throws([&] { (void)selector.prepare(); }), "no selection after EOS");
    ConstraintSelection tie(spec, trie, tokenizer);
    std::fill(row.begin(), row.end(), -100); row[' '] = -0.0F; row['{'] = 0.0F;
    check(tie.select(row, tie.prepare()).token == ' ', "signed-zero tie chooses lowest ID");
    row[tokenizer->find("<tool_call>")] = std::numeric_limits<float>::quiet_NaN();
    check(throws([&] { (void)tie.select(row, tie.prepare()); }), "nonfinite even in masked entry fails");
    auto empty = tie.prepare(); std::fill(empty.mask.words.begin(), empty.mask.words.end(), 0);
    std::fill(row.begin(), row.end(), 0);
    check(throws([&] { (void)tie.select(row, empty); }), "empty mask fails explicitly");
    auto thinking_spec = std::make_shared<ConstraintSpec>(*spec); thinking_spec->starts_in_reasoning = true;
    ConstraintSelection thinking(thinking_spec, trie, tokenizer);
    const auto execution = thinking.verdict("execution_failure", "serial-host");
    check(execution.status == "failed" && execution.reason == "execution_failure" && execution.phase == "reasoning" && !execution.accepting,
        "execution failure preserves reasoning frontier");
    row[*spec->end_thinking_id] = 10;
    auto closing = thinking.select(row, thinking.prepare());
    check(thinking.state().reasoning && !closing.state.reasoning, "end-thinking transition is staged");
    thinking.commit(closing);
    check(!thinking.state().reasoning && !thinking.prepare().mask.contains('x'), "next position uses answer mask");
    ConstraintSelection early(thinking_spec, trie, tokenizer);
    row[spec->eos_ids.front()] = 20; auto stop = early.select(row, early.prepare()); early.commit(stop);
    const auto early_verdict = early.verdict("stop", "serial-host");
    check(early_verdict.status == "incomplete" && early_verdict.reason == "reasoning_eos", "reasoning EOS stays unmasked and incomplete");
    check(thinking.verdict("context", "serial-host").reason == "context", "context verdict");
    check(thinking.verdict("cancelled", "serial-host").reason == "cancelled", "cancellation verdict");
    check(thinking.verdict("constraint_failure", "serial-host").status == "failed", "constraint failure verdict");
    ConstraintSelection accepting(spec, trie, tokenizer);
    row.assign(row.size(), -100); row['{'] = 1; accepting.commit(accepting.select(row, accepting.prepare()));
    row['{'] = -100; row['}'] = 1; accepting.commit(accepting.select(row, accepting.prepare()));
    const auto budget = accepting.verdict("length", "serial-host");
    check(budget.status == "incomplete" && budget.accepting && budget.reason == "max_output_tokens", "accepting budget boundary is still incomplete");
}

void test_thinking_frontier(const std::shared_ptr<const BpeTokenizer>& tokenizer) {
    auto spec = std::make_shared<ConstraintSpec>();
    spec->logprobs = true; spec->transform.presence = .1;
    spec->starts_in_reasoning = true;
    spec->end_thinking_id = tokenizer->find("</think>");
    spec->eos_ids = {tokenizer->find("<|im_end|>")};
    ThinkingPolicy policy; policy.max_thinking_tokens = 2; policy.reasoning_limit = 2;
    policy.closure_ids = tokenizer->encode("</think>\n\n", true); spec->thinking = policy;
    auto selection = std::make_shared<ConstraintSelection>(spec, std::make_shared<TokenByteTrie>(*tokenizer), tokenizer);
    ConstraintRowSelection adapter(selection);
    std::vector<float> logits(tokenizer->vocab_size(), -100); logits['x'] = 1;
    for (int i = 0; i < 2; ++i) { adapter.prepare(); check(adapter.select(logits) == 'x', "reasoning frontier selected"); adapter.commit(); }
    auto staged = selection->select(logits, selection->prepare());
    check(staged.forced_control && staged.token == *spec->end_thinking_id && selection->state().forced_ids.empty(), "forced control is staged before target commit");
    std::vector<std::uint32_t> consumed;
    // A serial target consumes each selected pending frontier before the next row.
    for (const auto expected : policy.closure_ids) {
        adapter.prepare(); const auto pending = adapter.select(logits);
        check(pending == expected, "target pending frontier is exact forced ID");
        consumed.push_back(pending); adapter.commit();
        check(!spec->scores->tokens.back().logprob && spec->scores->tokens.back().forced_control,
            "forced control has null model score and explicit origin");
    }
    check(selection->state().reply_counts.at('x') == 2 && selection->state().reply_counts.at(*spec->end_thinking_id) == 1, "reasoning and forced controls enter committed counts");
    check(consumed == policy.closure_ids && !selection->state().reasoning && selection->state().reasoning_tokens == 2,
        "entire closure commits before answer phase");
    adapter.prepare(); check(adapter.select(logits) == 'x', "ordinary answer greedy restored after closure"); adapter.commit();
    check(selection->state().reasoning_tokens == 2, "answer does not increment reasoning count");
}

void test_scoring_transforms(const std::shared_ptr<const BpeTokenizer>& tokenizer) {
    const auto trie = std::make_shared<TokenByteTrie>(*tokenizer);
    auto spec = std::make_shared<ConstraintSpec>(); spec->logprobs = true; spec->top_logprobs = 20;
    auto selection = std::make_shared<ConstraintSelection>(spec, trie, tokenizer);
    ConstraintRowSelection adapter(selection);
    std::vector<float> logits(tokenizer->vocab_size(), -100); logits['a'] = logits['b'] = 2; logits['c'] = -1;
    double sum = 0; for (auto z : logits) sum += std::exp(static_cast<double>(z) - 2);
    const double reference = -std::log(sum);
    adapter.prepare(); check(adapter.select(logits) == 'a', "scores preserve lowest-ID greedy tie");
    check(spec->scores->tokens.empty(), "scores not published before commit");
    adapter.prepare(); check(adapter.select(logits) == 'a', "discarded row selects identically"); adapter.commit();
    check(spec->scores->tokens.size() == 1 && spec->scores->tokens.front().logprob == reference, "first terminal prompt row exact reference log-softmax");
    const auto& top = spec->scores->tokens.front().top;
    check(top.size() == 20 && top[0].first == 'a' && top[1].first == 'b' && top[2].first == 'c' && top[3].first == 0, "top logprobs descending with ID ties");
    auto masked_spec = std::make_shared<ConstraintSpec>(); masked_spec->logprobs = true;
    masked_spec->program = std::make_shared<ConstraintProgram>(compile_json_object());
    masked_spec->transform.bias = {{'a', 100}};
    auto masked = std::make_shared<ConstraintSelection>(masked_spec, trie, tokenizer);
    const auto prepared = masked->prepare();
    std::vector<float> masked_row(logits.size(), -100); masked_row['{'] = 2; masked_row['a'] = 100;
    auto staged = masked->select(masked_row, prepared);
    double masked_sum = 0;
    for (std::uint32_t i = 0; i < masked_row.size(); ++i) if (prepared.mask.contains(i)) masked_sum += std::exp(static_cast<double>(masked_row[i]) - 2);
    check(staged.token == '{' && staged.score.logprob == -std::log(masked_sum), "JSON mask follows bias and defines log-softmax support");
    masked->commit(staged);
    const auto run = [&](LogitTransform transform, std::vector<std::uint32_t> prompt, std::vector<float> row) {
        auto p = std::make_shared<ConstraintSpec>(); p->transform = transform; p->prompt_ids = prompt;
        auto sel = std::make_shared<ConstraintSelection>(p, trie, tokenizer);
        ConstraintRowSelection r(sel); r.prepare(); const auto first = r.select(row);
        check(sel->state().reply_counts.empty(), "counts unchanged by uncommitted selection");
        r.prepare(); check(r.select(row) == first, "discarded transform selection has no count effect"); r.commit();
        check(sel->state().reply_counts.at(first) == 1, "only committed token counted");
        r.prepare(); const auto second = r.select(row); r.commit();
        r.prepare(); const auto third = r.select(row);
        return std::vector<std::uint32_t>{first, second, third};
    };
    LogitTransform t; t.presence = 1;
    logits['a'] = 2; logits['b'] = .5F;
    check(run(t, {}, logits) == std::vector<std::uint32_t>{'a','a','a'}, "presence subtracts once regardless of count");
    t.presence = 0; t.frequency = 1;
    check(run(t, {}, logits) == std::vector<std::uint32_t>{'a','a','b'}, "frequency subtracts per committed reply occurrence");
    t.frequency = 0; t.repetition = 2; logits['b'] = 1.5F;
    check(run(t, {'a'}, logits).front() == 'b', "repetition divides positive prompt-seen logits");
    logits.assign(logits.size(), -100); logits['a'] = -1; logits['b'] = -1.5F;
    check(run(t, {'a'}, logits).front() == 'b', "repetition multiplies negative prompt-seen logits");
    t.repetition = 1; t.bias = {{'b', 1}};
    check(run(t, {}, logits).front() == 'b', "bias added after penalties");
    t.presence = .25; t.frequency = .25; t.repetition = 2; t.bias = {{'a', 1.75}};
    // Repetition first on the raw logit, then presence/frequency, then bias. After n committed a:
    // -1 * 2 - .25 - .25 * n + 1.75 = -.5 - .25 n: -0.75, -1.0, -1.25 > b = -1.5 (b is never seen: unchanged).
    check(run(t, {}, logits) == std::vector<std::uint32_t>{'a','a','a'}, "repetition presence frequency bias composition order");
    t.bias = {{'a', 1.25}};  // after 2 committed a: -2 - .25 - .5 + 1.25 = -1.5 ties b; the lower id 'a' wins
    check(run(t, {}, logits) == std::vector<std::uint32_t>{'a','a','a'}, "composition tie resolves to the lowest id");
}

int main() try {
    {
        // Large objects: repeated keys keep their first position and the last value (serving parser) or fail (unique
        // parser), with or without the key index; a large object parses in linear time.
        std::string wide = "{";
        for (int i = 0; i < 200; ++i) wide += (i ? ",\"k" : "\"k") + std::to_string(i) + "\":" + std::to_string(i);
        const auto replaced = parse_json(wide + ",\"k5\":-5,\"k150\":-150}");
        check(replaced.size() == 200U && replaced.as_object()[5].first == "k5" && replaced.find("k5")->as_int() == -5 &&
                  replaced.as_object()[150].second.as_int() == -150, "indexed object replaces repeated keys in place");
        check(parse_json(R"({"a":1,"b":2,"a":3})").find("a")->as_int() == 3, "small object replaces repeated keys");
        bool unique_refused = false;
        try { (void)parse_json_unique(wide + ",\"k150\":1}"); } catch (const JsonError&) { unique_refused = true; }
        check(unique_refused, "unique parser refuses a repeated key in an indexed object");
        std::string huge = "{";
        for (int i = 0; i < 200000; ++i) huge += (i ? ",\"m" : "\"m") + std::to_string(i) + "\":0";
        huge += "}";
        const auto begin = std::chrono::steady_clock::now();
        check(parse_json(huge).size() == 200000U && parse_json_unique(huge).size() == 200000U, "200k-member object parses");
        check(std::chrono::steady_clock::now() - begin < std::chrono::seconds(10), "200k-member object parses in linear time");
        // A request body holds at most kMaxRequestJsonValues values (parse_request_json).
        std::string many = "[0";
        many.reserve(2U * kMaxRequestJsonValues + 2U);
        for (std::size_t i = 1; i < kMaxRequestJsonValues - 1U; ++i) many += ",0";
        check(parse_request_json(many + "]").size() == kMaxRequestJsonValues - 1U, "a document at the value limit parses");
        bool over = false;
        try { (void)parse_request_json(many + ",0]"); } catch (const JsonError&) { over = true; }
        check(over, "a request body over the value limit is refused");
        check(parse_json(many + ",0]").size() == kMaxRequestJsonValues, "files and trusted inputs keep no value limit");
    }
    {
        // Request bodies share one in-flight budget: a request that does not fit is refused with 503 before its body
        // is read; the budget is returned when the handler finishes.
        HttpServerConfig budget_config;
        budget_config.port = 0;
        budget_config.max_inflight_body_bytes = 1000;
        std::atomic<bool> release{false};
        HttpServer budget(budget_config, [&](const HttpRequest& r, HttpResponseWriter& w) {
            if (r.path == "/hold") while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
            w.send(200, "text/plain", "ok");
        });
        budget.listen();
        std::thread budget_thread([&] { budget.serve(); });
        std::thread holder([&] { (void)http(budget.port(), "POST", "/hold", std::string(800, 'x')); });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));  // the 800-byte body is held by its handler
        check(http(budget.port(), "POST", "/ok", std::string(300, 'y')).status == 503, "a body over the shared budget answers 503");
        check(http(budget.port(), "POST", "/ok", std::string(150, 'y')).status == 200, "a body within the remaining budget is served");
        check(http(budget.port(), "GET", "/ok", "").status == 200, "a request without a body always fits");
        release.store(true);
        holder.join();
        check(http(budget.port(), "POST", "/ok", std::string(900, 'z')).status == 200, "the budget is returned after the handler");
        budget.stop();
        budget_thread.join();
    }
    {
        // A handler that throws a non-standard type fails that request with 500; the server keeps serving.
        HttpServerConfig boom_config;
        boom_config.port = 0;
        HttpServer boom(boom_config, [](const HttpRequest& r, HttpResponseWriter& w) {
            if (r.path == "/boom") throw 42;
            if (r.path == "/stream-boom") {
                (void)w.begin_stream(200, "text/event-stream");
                (void)w.write("data: x\n\n");
                throw 42;
            }
            w.send(200, "text/plain", "ok");
        });
        boom.listen();
        std::thread boom_thread([&] { boom.serve(); });
        check(http(boom.port(), "GET", "/boom", "").status == 500, "non-standard handler exception answers 500");
        check(http(boom.port(), "GET", "/ok", "").status == 200, "server survives a non-standard handler exception");
        {
            // Pipelined keep-alive requests: a stream interrupted by an exception must end the connection, so the
            // second response can never appear inside the unfinished chunked body.
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_port = htons(boom.port());
            ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
            check(::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "pipelined client connects");
            timeval receive_limit{};
            receive_limit.tv_sec = 20;  // a regression keeps the connection open: fail fast instead of the 300 s idle wait
            ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &receive_limit, sizeof(receive_limit));
            const std::string requests = "GET /stream-boom HTTP/1.1\r\nHost: x\r\n\r\nGET /ok HTTP/1.1\r\nHost: x\r\n\r\n";
            (void)::send(fd, requests.data(), requests.size(), MSG_NOSIGNAL);
            std::string raw;
            char buffer[4096];
            for (;;) {
                const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
                if (n <= 0) break;
                raw.append(buffer, static_cast<std::size_t>(n));
            }
            ::close(fd);
            std::size_t responses = 0;
            for (auto at = raw.find("HTTP/1.1 "); at != std::string::npos; at = raw.find("HTTP/1.1 ", at + 1U)) ++responses;
            check(responses == 1 && raw.find("data: x") != std::string::npos, "interrupted stream closes the connection");
        }
        boom.stop();
        boom_thread.join();
    }
    {
        // Request and write deadlines: a stalled request gets 408, a client that stops reading is noticed within the
        // write timeout, and the request deadline starts only at a request's first byte (idle keep-alive is unchanged).
        HttpServerConfig timed_config;
        timed_config.port = 0;
        timed_config.request_timeout_seconds = 1;
        timed_config.write_timeout_seconds = 1;
        std::atomic<int> flood_result{0};  // 0 running, 1 write failed (client gone), 2 wrote everything
        HttpServer timed(timed_config, [&](const HttpRequest& r, HttpResponseWriter& w) {
            if (r.path == "/flood") {
                (void)w.begin_stream(200, "text/event-stream");
                const std::string block(1U << 16U, 'x');
                for (int i = 0; i < 4096; ++i)
                    if (!w.write(block)) { flood_result.store(1); return; }
                w.end_stream();
                flood_result.store(2);
                return;
            }
            w.send(200, "text/plain", "ok");
        });
        timed.listen();
        std::thread timed_thread([&] { timed.serve(); });
        const auto connect_local = [&] {
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_port = htons(timed.port());
            ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
            timeval receive_limit{};
            receive_limit.tv_sec = 20;
            ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &receive_limit, sizeof(receive_limit));
            check(::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "deadline client connects");
            return fd;
        };
        const auto read_all = [](int fd) {
            std::string raw;
            char buffer[4096];
            for (;;) {
                const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
                if (n <= 0) break;
                raw.append(buffer, static_cast<std::size_t>(n));
            }
            return raw;
        };
        {
            const int fd = connect_local();
            const std::string partial = "GET /ok HTTP/1.1\r\nHo";
            (void)::send(fd, partial.data(), partial.size(), MSG_NOSIGNAL);
            const auto raw = read_all(fd);
            ::close(fd);
            check(raw.rfind("HTTP/1.1 408", 0) == 0, "stalled request header answers 408");
        }
        {
            // A kept-alive connection: a first request, idle longer than the request deadline, then a second request
            // that is answered; a third that stalls is refused with 408 even though the connection already responded.
            const int fd = connect_local();
            const auto read_response = [&] {
                std::string raw;
                char buffer[4096];
                while (raw.find("\r\n\r\nok") == std::string::npos) {
                    const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
                    if (n <= 0) break;
                    raw.append(buffer, static_cast<std::size_t>(n));
                }
                return raw;
            };
            const std::string request = "GET /ok HTTP/1.1\r\nHost: x\r\n\r\n";
            (void)::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
            check(read_response().rfind("HTTP/1.1 200", 0) == 0, "kept-alive first request answered");
            std::this_thread::sleep_for(std::chrono::milliseconds(2500));  // idle longer than the request deadline
            (void)::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
            check(read_response().rfind("HTTP/1.1 200", 0) == 0, "request deadline starts at the first byte, not while idle");
            const std::string partial = "GET /ok HTTP/1.1\r\nHo";
            (void)::send(fd, partial.data(), partial.size(), MSG_NOSIGNAL);
            const auto raw = read_all(fd);
            check(raw.rfind("HTTP/1.1 408", 0) == 0, "a stalled request on a kept-alive connection answers 408");
            ::close(fd);
        }
        {
            // A body that stops arriving is refused once the request deadline passes.
            const int fd = connect_local();
            const std::string partial = "POST /ok HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\nabc";
            (void)::send(fd, partial.data(), partial.size(), MSG_NOSIGNAL);
            const auto raw = read_all(fd);
            ::close(fd);
            check(raw.rfind("HTTP/1.1 408", 0) == 0, "stalled request body answers 408");
        }
        {
            // Framing that cannot be read unambiguously is refused and the connection closed (one response only).
            const auto answer = [&](const std::string& raw_request) {
                const int fd = connect_local();
                (void)::send(fd, raw_request.data(), raw_request.size(), MSG_NOSIGNAL);
                const auto raw = read_all(fd);
                ::close(fd);
                std::size_t responses = 0;
                for (auto at = raw.find("HTTP/1.1 "); at != std::string::npos; at = raw.find("HTTP/1.1 ", at + 1U)) ++responses;
                return responses == 1U ? raw.substr(9, 3) : "x" + std::to_string(responses);
            };
            const std::string get_ok = "GET /ok HTTP/1.1\r\nHost: x\r\n\r\n";
            check(answer("POST /ok HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\nabc" + get_ok) == "400", "conflicting Content-Length answers 400");
            check(answer("POST /ok HTTP/1.1\r\nContent-Length: 5abc\r\n\r\nabcde") == "400", "non-decimal Content-Length answers 400");
            check(answer("POST /ok HTTP/1.1\r\nContent-Length: +5\r\n\r\nabcde") == "400", "signed Content-Length answers 400");
            check(answer("POST /ok HTTP/1.1\r\nContent-Length : 5\r\n\r\nabcde") == "400", "whitespace before the colon answers 400");
            check(answer("POST /ok HTTP/1.1\r\nTransfer-Encoding: gzip\r\nContent-Length: 1\r\n\r\na") == "501", "non-identity transfer encoding answers 501");
            check(answer("GET /ok HTTP/2.0\r\n\r\n") == "400", "unsupported HTTP version answers 400");
            check(answer("POST /ok HTTP/1.1\r\nTransfer-Encoding:\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n" + get_ok) == "501",
                  "every Transfer-Encoding field counts, even an empty first one");
            check(answer(std::string("POST /ok HTTP/1.1\r\nContent-Length\x0b: 20\r\n\r\n") + get_ok) == "400", "control byte in a field name answers 400");
            check(answer("GET   HTTP/1.1\r\n\r\n") == "400", "whitespace-only target answers 400");
            check(answer("GET /o\x0bk HTTP/1.1\r\n\r\n") == "400" && answer("GET /o\x0ck HTTP/1.1\r\n\r\n") == "400", "control bytes in the target answer 400");
            check(answer("G(T /ok HTTP/1.1\r\n\r\n") == "400", "non-token method answers 400");
            {
                // The refusal survives unread request bytes (no reset before the client reads it) and is complete JSON.
                const int fd = connect_local();
                const std::string raw_request = "POST /ok HTTP/1.1\r\nContent-Length: x\r\n\r\n" + std::string(200000, 'z');
                (void)::send(fd, raw_request.data(), raw_request.size(), MSG_NOSIGNAL);
                const auto raw = read_all(fd);
                ::close(fd);
                const auto body_at = raw.find("\r\n\r\n");
                bool json = false;
                try { json = body_at != std::string::npos && parse_json(raw.substr(body_at + 4U)).find("error") != nullptr; } catch (...) {}
                check(raw.rfind("HTTP/1.1 400", 0) == 0 && json, "refusal with unread bytes is delivered whole");
            }
            check(answer("POST /ok HTTP/1.1\r\nContent-Length: 3\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc") == "200", "agreeing Content-Length fields are accepted");
            // A header beyond the limit is refused even when its terminator arrives in the same read.
            HttpServerConfig small_config;
            small_config.port = 0;
            small_config.max_header_bytes = 1024;
            HttpServer small(small_config, [](const HttpRequest&, HttpResponseWriter& w) { w.send(200, "text/plain", "ok"); });
            small.listen();
            std::thread small_thread([&] { small.serve(); });
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_port = htons(small.port());
            ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
            timeval receive_limit{};
            receive_limit.tv_sec = 20;
            ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &receive_limit, sizeof(receive_limit));
            check(::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "header-limit client connects");
            const std::string big = "GET /ok HTTP/1.1\r\nX-Pad: " + std::string(4000, 'a') + "\r\n\r\n";
            (void)::send(fd, big.data(), big.size(), MSG_NOSIGNAL);
            const auto raw = read_all(fd);
            ::close(fd);
            check(raw.rfind("HTTP/1.1 431", 0) == 0, "header over the limit answers 431 whatever the read sizes");
            // A header of exactly the limit gets the same answer whole and split inside its terminator.
            const auto at_limit = [&](std::size_t extra, bool split) {
                const std::string prefix = "GET /ok HTTP/1.1\r\nConnection: close\r\nX-Pad: ";
                const std::string head = prefix + std::string(1024U + extra - prefix.size(), 'a');
                const std::string request = head + "\r\n\r\n";
                const int c = ::socket(AF_INET, SOCK_STREAM, 0);
                ::setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &receive_limit, sizeof(receive_limit));
                (void)::connect(c, reinterpret_cast<sockaddr*>(&address), sizeof(address));
                const auto cut = split ? request.size() - 2U : request.size();
                (void)::send(c, request.data(), cut, MSG_NOSIGNAL);
                if (split) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    (void)::send(c, request.data() + cut, request.size() - cut, MSG_NOSIGNAL);
                }
                const auto reply = read_all(c);
                ::close(c);
                return reply.substr(0, 12);
            };
            check(at_limit(0, false) == "HTTP/1.1 200" && at_limit(0, true) == "HTTP/1.1 200", "a header at the limit is accepted whole or split");
            check(at_limit(1, false) == "HTTP/1.1 431" && at_limit(1, true) == "HTTP/1.1 431", "one byte over the limit is refused whole or split");
            small.stop();
            small_thread.join();
        }
        {
            const int fd = connect_local();
            const int small = 4096;
            ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &small, sizeof(small));
            const std::string request = "GET /flood HTTP/1.1\r\nHost: x\r\n\r\n";
            (void)::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
            const auto begin = std::chrono::steady_clock::now();
            while (flood_result.load() == 0 && std::chrono::steady_clock::now() - begin < std::chrono::seconds(30))
                std::this_thread::sleep_for(std::chrono::milliseconds(50));  // never read: the server's send stalls
            check(flood_result.load() == 1, "a client that stops reading is marked gone within the write timeout");
            ::close(fd);
        }
        timed.stop();
        timed_thread.join();
    }
    {
        GenerationTicket ticket;
        check(ticket.progress().phase == GenerationPhase::queued, "ticket starts queued");
        for (std::uint64_t n = 0; n <= 100; ++n) ticket.report_progress({GenerationPhase::prefill, n, 10, 110});
        const auto progress = ticket.progress();
        check(progress.prefilled_tokens == 100 && progress.cached_tokens == 10 && progress.total_tokens == 110, "ticket coalesces completed units");
        const std::array<std::uint32_t, 1> token{42};
        ticket.publish(token);
        check(ticket.progress().phase == GenerationPhase::decode && ticket.progress().prefilled_tokens == 100, "first token changes phase only");
        bool done = false;
        check(ticket.wait(std::chrono::milliseconds(0), done) == std::vector<std::uint32_t>{42}, "progress preserves token publication");
        GenerationTicket silent; silent.publish(token);
        check(silent.progress().phase == GenerationPhase::queued, "no fabricated engine report");
    }
    test_batched_queue();
    test_queue_caller_orders();
    test_queue_cancel_replenishment();
    const auto tokenizer = tiny_tokenizer();
    test_scoring_transforms(tokenizer);
    test_constraint_selection(tokenizer);
    test_thinking_frontier(tokenizer);
    test_constraint_row(tokenizer);
    test_constraint_request_failure();
    test_constraint_late_cancellation();
    ScriptedEngine engine(tokenizer);
    GenerationQueue queue;
    ApiConfig config;
    config.model_id = "test-model";
    config.http_slots = 9;
    config.capabilities_extra=[](Json& out) {
        strix::decode_cli::Options modes;
        modes.session_tiled_suffix_requested=true; modes.session_mtp_fixed_requested=true;
        modes.session_modes_suppression="baseline";
        out.set("session_numerical_modes",strix::decode_cli::session_modes_json(modes));
    };
    config.stop_tokens = {tokenizer->find("<|im_end|>"), tokenizer->find("<|endoftext|>")};
    config.api_key = "secret";
    config.sse_keepalive_seconds = 0.3;
    config.engine_progress = true;
    config.sse_progress_seconds = 0.15;
    auto mirror=std::make_shared<strix::decode_cli::StateCacheStatsMirror>();
    strix::StateCacheSnapshotV1 snap; snap.prefix_enabled=true; snap.prefix.no_store_skips=7; snap.prefix.evictions_superseded=3;
    mirror->publish(snap);
    config.state_cache_stats=[mirror] { return mirror->read(); };
    ApiService api(config, tokenizer, queue, engine.context_tokens(), false);
    auto constrained_config = config;
    constrained_config.engine_logprobs = constrained_config.engine_logit_transforms = true;
    constrained_config.engine_constraints = engine.supports_constraints();
    constrained_config.engine_thinking_budget = engine.supports_thinking_budget();
    constrained_config.enforce_anthropic_thinking_budget = true;  // the operator opted in (default: stated, not enforced)
    constrained_config.constraint_execution_routes = engine.constraint_execution_routes();
    ApiService constrained_api(constrained_config, tokenizer, queue, engine.context_tokens(), true);
    ApiService constrained_greedy_api(constrained_config, tokenizer, queue, engine.context_tokens(), false);
    constrained_config.stream_receipts = false;
    ApiService constrained_quiet_api(constrained_config, tokenizer, queue, engine.context_tokens(), true);
    auto session_config = config; session_config.sessions_enabled = true;
    const auto allocator=[](std::uint64_t bytes) -> strix::PrefixCacheBuffer {
        return {std::shared_ptr<std::byte>(new std::byte[static_cast<std::size_t>(bytes ? bytes : 1)],std::default_delete<std::byte[]>()),bytes};
    };
    strix::decode_cli::HttpSessionStore startup_store({},1U<<20,3,allocator,{},1234);
    auto session_mirror=std::make_shared<strix::decode_cli::StateCacheStatsMirror>();
    session_mirror->publish(startup_store.snapshot(),parse_json(startup_store.command("stats","","")));
    session_config.state_cache_stats=[session_mirror] {return session_mirror->read();};
    ApiService session_api(session_config, tokenizer, queue, engine.context_tokens(), false);
    auto constrained_session_config = constrained_config; constrained_session_config.sessions_enabled = true;
    constrained_session_config.stream_receipts = true;
    ApiService constrained_session_api(constrained_session_config, tokenizer, queue, engine.context_tokens(), false);
    auto off_config = config;
    off_config.capabilities = false; off_config.responses = false;
    ApiService off_api(off_config, tokenizer, queue, engine.context_tokens(), false);
    auto legacy_config = config;
    legacy_config.stream_receipts = false; legacy_config.sse_keepalive_seconds = 0;
    ApiService legacy_api(legacy_config, tokenizer, queue, engine.context_tokens(), false);
    auto quiet_config = config; quiet_config.sse_keepalive_seconds = 0;
    ApiService quiet_api(quiet_config, tokenizer, queue, engine.context_tokens(), false);
    auto progress_off_config = config; progress_off_config.sse_progress_seconds = 0;
    ApiService progress_off_api(progress_off_config, tokenizer, queue, engine.context_tokens(), false);
    auto sample_config = config; sample_config.sampling_policy = "sample";
    sample_config.capabilities_extra = [](Json& doc) {
        doc.set("engine_feature", Json(true));
        strix::decode_cli::qwen_rotary_capabilities(doc);
    };
    ApiService sample_api(sample_config, tokenizer, queue, engine.context_tokens(), true);
    auto defaults_config = sample_config;
    defaults_config.default_temperature = 0.7; defaults_config.default_top_k = 20;
    defaults_config.default_top_p = 0.9; defaults_config.default_min_p = 0.1;
    ApiService defaults_api(defaults_config, tokenizer, queue, engine.context_tokens(), true);
    auto reject_config = config; reject_config.sampling_policy = "reject"; reject_config.responses = false;
    ApiService reject_api(reject_config, tokenizer, queue, engine.context_tokens(), false);
    HttpServerConfig http_config;
    http_config.port = 0;
    HttpServer server(http_config, [&](const HttpRequest& r, HttpResponseWriter& w) {
        const auto mode = r.header("x-test-config");
        if (r.header("x-test-sessions") == "1" && mode == "constraints") constrained_session_api.handle(r, w);
        else if (r.header("x-test-sessions") == "1") session_api.handle(r, w);
        else if (mode == "constraints") constrained_api.handle(r, w);
        else if (mode == "constraints-greedy") constrained_greedy_api.handle(r, w);
        else if (mode == "constraints-quiet") constrained_quiet_api.handle(r, w);
        else if (mode == "off") off_api.handle(r, w);
        else if (mode == "legacy") legacy_api.handle(r, w);
        else if (mode == "progress-off") progress_off_api.handle(r, w);
        else if (mode == "quiet") quiet_api.handle(r, w);
        else if (mode == "defaults") defaults_api.handle(r, w);
        else if (mode == "sample") sample_api.handle(r, w);
        else if (mode == "reject") reject_api.handle(r, w);
        else api.handle(r, w);
    });
    server.listen();
    const auto port = server.port();
    std::thread http_thread([&] { server.serve(); });
    std::thread engine_thread([&] { queue.run(engine); });
    const std::string auth = "Authorization: Bearer secret\r\n";
    const auto first_choice = [](const Json& j) -> const Json& { return at(j, "choices").as_array().at(0); };
    const auto stream_text = [](const std::string& body, std::string* reasoning, std::string* finish, bool* usage, bool* tools) {
        std::string content;
        for (const auto& event : sse_data(body)) {
            if (const auto* u = event.find("usage"); usage && u && !u->is_null()) *usage = true;
            const auto& choices = at(event, "choices").as_array();
            if (choices.empty()) continue;
            const auto& delta = at(choices[0], "delta");
            if (const auto* c = delta.find("content"); c && c->is_string()) content += c->as_string();
            if (const auto* c = delta.find("reasoning_content"); reasoning && c && c->is_string()) *reasoning += c->as_string();
            if (const auto* f = choices[0].find("finish_reason"); finish && f && f->is_string()) *finish = f->as_string();
            if (tools && delta.find("tool_calls")) *tools = true;
        }
        return content;
    };

    try {
    {
        const auto mode = auth + "X-Test-Config: constraints\r\n";
        const auto call = [&](const std::string& path, Json body, bool stream) {
            body.set("stream", Json(stream)); return http(port, "POST", path, body.dump(), mode);
        };
        const auto base = parse_json(R"({"messages":[{"role":"user","content":"Hi"}],"enable_thinking":false,"max_tokens":3,"logprobs":true,"top_logprobs":3})");
        auto reply = call("/v1/chat/completions", base, false);
        check(reply.status == 200, "chat score request executes");
        auto result = parse_json(reply.body);
        check(at(at(result,"strix"),"selection_execution").as_string()=="serial-host" &&
            at(at(result,"strix"),"selection_fallback").is_null(),"score-only resolved route receipt");
        const auto scores = at(at(first_choice(result), "logprobs"), "content");
        check(scores.size() == 3 && at(scores.as_array()[0], "token").as_string() == "a", "chat first-token and every score emitted");
        double sum = 0; for (std::size_t i = 0; i < tokenizer->vocab_size(); ++i) sum += std::exp((i == 'a' || i == 'b' ? 2.0 : i == 'c' ? -1.0 : -100.0) - 2);
        check(at(scores.as_array()[0], "logprob").as_double() == -std::log(sum), "wire logprob exactly equals reference");
        check(at(at(result, "strix"), "scoring").find("basis")->as_string() == "target-row-log-softmax", "score basis receipt");
        for(const auto* marker:{"ROUTE_MTP","ROUTE_OFF"}) {
            auto routed=base;routed.set("messages",parse_json(std::string("[{\"role\":\"user\",\"content\":\"")+marker+"\"}]"));
            const auto wire=parse_json(call("/v1/chat/completions",routed,false).body);
            const auto& receipt=at(wire,"strix");
            check(at(receipt,"selection_execution").as_string()==(std::string(marker)=="ROUTE_MTP"?"mtp-host":"serial-host"),"wire resolved selection route");
            check(std::string(marker)=="ROUTE_MTP"?at(receipt,"selection_fallback").is_null():at(receipt,"selection_fallback").as_string()=="selection-mtp-off","wire selection fallback reason");
        }
        Json streamed = Json::array();
        for (const auto& e : sse_data(call("/v1/chat/completions", base, true).body)) {
            const auto* choices = e.find("choices"); if (!choices || !choices->size()) continue;
            const auto* lp = choices->as_array()[0].find("logprobs"); if (lp && lp->is_object())
                for (const auto& score : at(*lp, "content").as_array()) streamed.push(score);
        }
        check(streamed.dump() == scores.dump(), "chat stream and nonstream scores identical");
        auto completion = parse_json(R"({"prompt":"Hi","max_tokens":3,"logprobs":3})");
        auto c = parse_json(call("/v1/completions", completion, false).body);
        const auto lp = at(first_choice(c), "logprobs");
        check(at(lp,"tokens").size() == 3 && at(lp,"text_offset").as_array()[2].as_int() == 2, "completions scores and offsets");
        std::size_t count = 0;
        for (const auto& e : sse_data(call("/v1/completions", completion, true).body)) {
            const auto& v = at(first_choice(e), "logprobs"); if (v.is_object()) count += at(v,"tokens").size();
        }
        check(count == 3, "completion stream scores exactly once");
        auto response_body = parse_json(R"({"input":"Hi","reasoning":{"effort":"none"},"max_output_tokens":3,"top_logprobs":3,"include":["message.output_text.logprobs"]})");
        auto r = parse_json(call("/v1/responses", response_body, false).body);
        const auto& part = at(at(r,"output").as_array()[0],"content").as_array()[0];
        check(at(part,"logprobs").dump() == scores.dump(), "Responses output_text scores");
        Json response_stream = Json::array();
        for (const auto& e : sse_data(call("/v1/responses", response_body, true).body))
            if (at(e,"type").as_string() == "response.output_text.delta") for (const auto& score : at(e,"logprobs").as_array()) response_stream.push(score);
        check(response_stream.dump() == scores.dump(), "Responses delta scores match nonstream");
        for (const auto n : {-1,21}) { auto b = parse_json(base.dump()); b.set("top_logprobs",Json(n)); check(call("/v1/chat/completions",b,false).status == 400,"top logprobs bounds"); }
        for (const auto n : {-1,6}) { auto b = parse_json(completion.dump()); b.set("logprobs",Json(n)); check(call("/v1/completions",b,false).status == 400,"completion logprobs bounds"); }
        for (const auto* field : {"presence_penalty","frequency_penalty","repetition_penalty"}) {
            for (const auto value : {-3.0, 11.0}) { auto b = parse_json(base.dump()); b.set(field,Json(value)); check(call("/v1/chat/completions",b,false).status == 400,"transform bounds"); }
        }
        for (const auto* bad : {R"({"-1":0})",R"({"267":1})",R"({"97":101})",R"({"97":-101})",R"({"a":0})",R"({"1.0":0})",R"({"097":0})"}) {
            auto b = parse_json(base.dump()); b.set("logit_bias",parse_json(bad)); check(call("/v1/chat/completions",b,false).status == 400,"bias ID and value validation");
        }
        auto oversized = parse_json(base.dump()); Json huge_bias = Json::object();
        for (int i = 0; i < 1025; ++i) huge_bias.set(std::to_string(i),Json(0));
        oversized.set("logit_bias",huge_bias); check(call("/v1/chat/completions",oversized,false).status == 400,"bias entry cap");
        auto bad_top = parse_json(base.dump()); bad_top.set("top_logprobs",Json(.5));
        check(call("/v1/chat/completions",bad_top,false).status == 400,"fractional top logprobs refused");
        auto score_top = parse_json(base.dump()); score_top.set("logprobs",Json(false));
        check(at(at(first_choice(parse_json(call("/v1/chat/completions",score_top,false).body)),"logprobs"),"content").size() == 3,"positive top logprobs independently routes");
        auto bytes = parse_json(base.dump()); bytes.set("max_tokens",Json(1));
        bytes.set("logit_bias",parse_json(R"({"97":-100,"98":-100,"99":-100,"195":100})"));
        const auto byte_reply = parse_json(call("/v1/chat/completions",bytes,false).body);
        const auto& byte_score = at(at(first_choice(byte_reply),"logprobs"),"content").as_array()[0];
        check(at(byte_score,"bytes").as_array()[0].as_int() == 195 && at(byte_score,"token").as_string() == "\xEF\xBF\xBD", "token bytes preserve split UTF8 with lossy text");
        auto transformed = parse_json(base.dump()); transformed.set("presence_penalty",Json(1)); transformed.set("frequency_penalty",Json(1));
        auto tr = parse_json(call("/v1/chat/completions",transformed,false).body);
        check(at(at(tr,"strix"),"sampling").as_string() == "greedy-logit-transform-v1", "transform identity distinct from exact greedy");
        check(!at(at(tr,"strix"),"unconstrained_greedy_equal").as_bool(), "transform receipt denies untransformed greedy equality");
        check(at(at(first_choice(tr),"message"),"content").as_string() == "aba", "reply counts affect committed selection");
        auto transform_only = parse_json(R"({"messages":[{"role":"user","content":"Hi"}],"enable_thinking":false,"max_tokens":1,"logit_bias":{"98":1}})");
        check(http(port,"POST","/v1/chat/completions",transform_only.dump(),auth).status == 400,"transform off separately refuses bias");
        auto zero = parse_json(completion.dump()); zero.set("logprobs",Json(0));
        check(at(at(first_choice(parse_json(call("/v1/completions",zero,false).body)),"logprobs"),"top_logprobs").as_array()[0].size() == 0,"completion zero requests chosen score with no alternatives");
        auto sampled = parse_json(base.dump()); sampled.set("temperature",Json(.7));
        check(call("/v1/chat/completions",sampled,false).status == 400,"sampled score policy refused");
        check(http(port,"POST","/v1/chat/completions",base.dump(),auth).status == 400,"logprobs off refuses request");
        check(http(port,"POST","/v1/chat/completions",transformed.dump(),auth).status == 400,"transforms off refuses request");
        const auto caps = parse_json(http(port,"GET","/v1/capabilities","",mode).body);
        check(at(caps,"logprobs").as_bool() && at(caps,"penalties").as_bool() && at(caps,"logit_bias").as_bool(),"executing capability switches");
        auto neutral = parse_json(R"({"messages":[{"role":"user","content":"Hi"}],"enable_thinking":false,"max_tokens":3,"presence_penalty":0,"frequency_penalty":0,"repetition_penalty":1,"logit_bias":{},"logprobs":false})");
        check(at(at(first_choice(parse_json(call("/v1/chat/completions",neutral,false).body)),"message"),"content").as_string() == "Pla", "neutral policy retains ordinary route");
    }
    {
        const std::string mode = auth + "X-Test-Config: constraints\r\n";
        const std::string quiet = auth + "X-Test-Config: constraints-quiet\r\n";
        const auto request = [](const char* prompt, int n, int room, int budget) {
            auto body = parse_json(R"({"messages":[{"role":"user","content":"FORCE"}],"max_tokens":64,"strix":{"thinking_policy":"thinking-budget-v1"}})");
            Json messages = Json::array(), message = Json::object();
            message.set("role", Json("user")); message.set("content", Json(prompt)); messages.push(std::move(message));
            body.set("messages", std::move(messages));
            body.set("max_tokens", Json(budget));
            Json options = *body.find("strix");
            options.set("max_thinking_tokens", Json(n)); options.set("answer_room_tokens", Json(room));
            body.set("strix", std::move(options));
            return body;
        };
        const auto run = [&](const Json& body, const std::string& headers = "") {
            return http(port, "POST", "/v1/chat/completions", body.dump(), headers.empty() ? mode : headers);
        };
        for (const auto* name : {"NATURAL", "FORCE"}) {
            auto body = request(name, 5, 20, 64);
            auto result = run(body, quiet);
            check(result.status == 200, "thinking request executes with receipts off");
            auto out = parse_json(result.body);
            const auto& t = at(at(out, "strix"), "thinking");
            const bool forced = std::string(name) == "FORCE";
            check(at(t, "closure").as_string() == (forced ? "forced" : "natural"), "closure provenance");
            check(at(t, "reasoning_tokens").as_int() == (forced ? 5 : 2), "committed reasoning count");
            const auto& message = at(first_choice(out), "message");
            check(at(message, "content").as_string() == (forced ? "\n\n" : "") + std::string("{\"a\":\"ok\"}"), "answer follows exact closure");
            if (forced) {
                const auto& ids = at(t, "forced_ids").as_array();
                auto closure = tokenizer->encode("</think>\n\n", true);
                check(ids.size() == closure.size(), "forced suffix length from tokenizer");
                for (std::size_t i = 0; i < ids.size(); ++i) check(ids[i].as_int() == closure[i], "exact forced closure id");
            }
            body.set("stream", Json(true));
            auto streamed = run(body, quiet);
            std::string streamed_reasoning;
            check(stream_text(streamed.body, &streamed_reasoning, nullptr, nullptr, nullptr) == at(message, "content").as_string(), "policy stream answer bytes equal non-stream");
            check(streamed_reasoning == std::string(forced ? 5U : 2U, 'w'), "policy stream splits committed thinking ID");
            auto events = sse_data(streamed.body);
            check(!events.empty() && at(at(events.back(), "strix"), "thinking").find("forced_ids"), "terminal stream thinking metadata with receipts off");
        }
        auto bounded = request("FORCE", 50, 20, 26);
        auto bounded_out = parse_json(run(bounded).body);
        const auto c = tokenizer->encode("</think>\n\n", true).size();
        check(at(at(at(bounded_out, "strix"), "thinking"), "reasoning_tokens").as_int() == 26 - 20 - static_cast<int>(c), "answer room bound wins");
        auto immediate = parse_json(run(request("FORCE", 5, 20, 20 + static_cast<int>(c))).body);
        check(at(at(at(immediate, "strix"), "thinking"), "reasoning_tokens").as_int() == 0, "zero remaining reasoning allowance closes immediately");
        auto combined = request("FORCE", 3, 20, 64);
        combined.set("response_format", parse_json(R"({"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object","properties":{"a":{"const":"ok"}},"required":["a"],"additionalProperties":false}}})"));
        combined.set("logprobs", Json(true)); combined.set("top_logprobs", Json(3));
        combined.set("presence_penalty", Json(.25)); combined.set("frequency_penalty", Json(.5));
        combined.set("repetition_penalty", Json(1.1)); combined.set("logit_bias", parse_json(R"({"97":-1})"));
        const auto composed = parse_json(run(combined).body);
        const auto& composed_scores = at(at(first_choice(composed), "logprobs"), "content");
        bool forced_score = false;
        for (const auto& value : composed_scores.as_array()) if (value.find("forced_control")) {
            forced_score = true; check(at(value,"logprob").is_null(), "composed forced record has null logprob");
        }
        check(forced_score && at(at(at(composed,"strix"),"scoring"),"constrained").as_bool(), "JSON thinking transform scoring compose");
        check(at(at(at(composed, "strix"), "constraint"), "status").as_string() == "complete", "JSON starts after forced suffix");
        const auto answer = at(at(first_choice(composed), "message"), "content").as_string();
        check(at(parse_json(answer), "a").as_string() == "ok", "composed answer validates");
        auto eos = parse_json(run(request("REASON_EOS", 5, 20, 64)).body);
        check(at(at(at(eos, "strix"), "thinking"), "reason").as_string() == "reasoning_eos", "reasoning EOS incomplete");
        auto responses = request("FORCE", 4, 20, 64);
        responses.set("input", Json("FORCE")); responses.set("max_output_tokens", Json(64));
        auto rr = http(port, "POST", "/v1/responses", responses.dump(), mode);
        check(rr.status == 200 && at(at(parse_json(rr.body), "strix"), "thinking").find("forced_ids"), "Responses executes policy");
        const auto before = engine.prompts().size();
        for (auto invalid : {request("FORCE", 0, 20, 64), request("FORCE", 5, 64, 64), request("FORCE", 5, 20, 1)})
            check(run(invalid).status == 400, "impossible budget rejected");
        auto omitted_room = request("FORCE", 5, 20, 64);
        omitted_room.set("strix", parse_json(R"({"thinking_policy":"thinking-budget-v1","max_thinking_tokens":5})"));
        const auto omitted = run(omitted_room);  // omitted room: the answer gets what remains after N and the closure
        check(omitted.status == 200, "omitted answer room resolves to the remainder");
        if (omitted.status == 200) {
            const auto omitted_json = parse_json(omitted.body);
            const auto& ot = at(at(omitted_json, "strix"), "thinking");
            check(at(ot, "answer_room_tokens").as_int() == 64 - 5 - static_cast<std::int64_t>(at(ot, "forced_ids").size()), "remainder answer room");
        }
        auto no_room = request("FORCE", 5, 20, 64);
        no_room.set("strix", parse_json(R"({"thinking_policy":"thinking-budget-v1","max_thinking_tokens":64})"));
        check(run(no_room).status == 400, "a budget that leaves no answer room is refused");
        auto disabled = request("FORCE", 5, 20, 64); disabled.set("enable_thinking", Json(false));
        check(run(disabled).status == 400, "disabled thinking rejected");
        auto unknown = request("FORCE", 5, 20, 64);
        auto unknown_options = *unknown.find("strix"); unknown_options.set("thinking_policy", Json("unknown")); unknown.set("strix", std::move(unknown_options));
        check(run(unknown).status == 400, "unknown policy rejected");
        check(run(request("FORCE", 5, 20, 64), auth).status == 400, "unsupported engine rejects policy");
        check(engine.prompts().size() == before + 1U, "invalid policies refused before submission (only the valid omitted-room request ran)");
        auto anthropic = parse_json(R"({"messages":[{"role":"user","content":"FORCE"}],"max_tokens":1200,"thinking":{"type":"enabled","budget_tokens":4}})");
        const auto ar = http(port, "POST", "/v1/messages", anthropic.dump(), quiet);
        check(ar.status == 200, "Anthropic budget mapped");
        check(!ApiConfig{}.enforce_anthropic_thinking_budget, "Anthropic budget enforcement is an operator opt-in");
        const auto anthropic_out = parse_json(ar.body);
        const auto& t = at(at(anthropic_out, "strix"), "thinking");
        check(at(t, "answer_room_tokens").as_int() == 1200 - 4 - static_cast<std::int64_t>(at(t, "forced_ids").size()), "default answer room is the remainder");
        check(at(t, "reasoning_tokens").as_int() == 4 && at(t, "closure").as_string() == "forced", "Anthropic budget enforced");
        anthropic.set("stream", Json(true));
        const auto as = http(port, "POST", "/v1/messages", anthropic.dump(), quiet);
        check(as.body.find("forced_ids") != std::string::npos && as.body.find("not enforced") == std::string::npos, "Anthropic stream final policy");
        for (const auto& headers : {auth, mode}) {
            const auto caps = parse_json(http(port, "GET", "/v1/capabilities", "", headers).body);
            check(at(caps, "thinking_budget").as_bool() == (headers == mode), "thinking capability matches execution");
            check(headers == mode ? at(caps, "thinking_budget_policy").as_string() == "thinking-budget-v1" : at(caps, "thinking_budget_policy").is_null(), "capability policy name gated");
        }
    }
    {
        const std::string mode = auth + "X-Test-Config: constraints\r\n";
        const std::string quiet = auth + "X-Test-Config: constraints-quiet\r\n";
        const std::string expected = " \n{\"a\":\"<tool_call> </think> caf\xC3\xA9\"}";
        const auto schema = parse_json(R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false})");
        const auto validate = [&](const std::string& answer) {
            const auto parsed = parse_json(answer);
            check(at(parsed, "a").as_string() == "<tool_call> </think> caf\xC3\xA9", "marker bytes and lowest-id tie retained");
            auto program = compile_json_schema(schema); ConstraintCursor cursor(program);
            cursor = advance_token(cursor, answer); check(cursor.complete(), "answer independently validates against schema");
        };
        const auto verdict = [&](const Json& out, const std::string& status) {
            const auto& v = at(at(out, "strix"), "constraint");
            check(at(v, "status").as_string() == status && at(v, "execution").as_string() == "serial-host", "engine constraint verdict");
            check(at(v, "selection").as_string() == "greedy-lowest-id", "selection identity");
            const auto& profile = at(v, "host_profile");
            check(at(profile, "rows").is_number() && at(profile, "mask_ms").is_number() && at(profile, "select_ms").is_number() &&
                at(profile, "window_rows").is_number() && at(profile, "window_row_ms").is_number(), "host selection profile reported");
        };
        for (bool thinking : {false, true}) for (bool schema_mode : {false, true}) {
            Json format = parse_json(R"({"type":"json_object"})");
            Json chat_format = format;
            if (schema_mode) {
                format.set("type", Json("json_schema")); format.set("name", Json("answer")); format.set("schema", schema); format.set("strict", Json(false));
                chat_format = parse_json(R"({"type":"json_schema"})");
                Json envelope = format; chat_format.set("json_schema", envelope);
            }
            auto chat = parse_json(R"({"messages":[{"role":"user","content":"JSON"}],"max_tokens":256})");
            chat.set("enable_thinking", Json(thinking)); chat.set("response_format", chat_format);
            auto responses = parse_json(R"({"input":"JSON","max_output_tokens":256})");
            responses.set("enable_thinking", Json(thinking)); Json text = Json::object(); text.set("format", format); responses.set("text", text);
            const auto chat_reply = http(port, "POST", "/v1/chat/completions", chat.dump(), mode);
            check(chat_reply.status == 200, "constrained chat admitted");
            const auto c = parse_json(chat_reply.body); verdict(c, "complete");
            const auto answer = at(at(first_choice(c), "message"), "content").as_string();
            check(answer == expected, "exact constrained chat bytes incl leading whitespace"); validate(answer);
            const auto response_reply = http(port, "POST", "/v1/responses", responses.dump(), mode);
            check(response_reply.status == 200, "constrained Responses admitted");
            const auto r = parse_json(response_reply.body); verdict(r, "complete");
            check(at(r, "status").as_string() == "completed", "completed constrained Responses");
            const auto& output = at(r, "output").as_array();
            check(at(at(output.back(), "content").as_array().front(), "text").as_string() == answer, "chat and Responses bytes match");
            const auto prompts = engine.prompts(); check(prompts[prompts.size()-1].first == prompts[prompts.size()-2].first, "constrained endpoint prompt IDs equal");
            chat.set("stream", Json(true)); responses.set("stream", Json(true));
            for (const auto& headers : {mode, quiet}) {
                const auto cs = http(port, "POST", "/v1/chat/completions", chat.dump(), headers);
                check(stream_text(cs.body, nullptr, nullptr, nullptr, nullptr) == answer, "chat SSE bytes equal nonstream bytes");
                const auto events = sse_data(cs.body); verdict(events.front(), "in_progress"); verdict(events.back(), "complete");
                if (headers == quiet) check(at(events.front(), "strix").size() == 3 && at(events.back(), "strix").size() == 3, "receipts off retains only mandatory constraint metadata");
                const auto rs = sse_data(http(port, "POST", "/v1/responses", responses.dump(), headers).body);
                std::string joined;
                for (const auto& event : rs) if (at(event, "type").as_string() == "response.output_text.delta") joined += at(event, "delta").as_string();
                check(joined == answer, "Responses SSE bytes equal nonstream bytes");
                verdict(at(rs.front(), "response"), "in_progress"); verdict(at(rs.back(), "response"), "complete");
            }
            chat.set("stream", Json(false)); chat.set("max_tokens", Json(1));
            const auto cut = parse_json(http(port, "POST", "/v1/chat/completions", chat.dump(), quiet).body); verdict(cut, "incomplete");
            check(at(first_choice(cut), "finish_reason").as_string() == "length", "chat budget keeps finish vocabulary");
            responses.set("stream", Json(false)); responses.set("max_output_tokens", Json(1));
            const auto cut_r = parse_json(http(port, "POST", "/v1/responses", responses.dump(), quiet).body); verdict(cut_r, "incomplete");
            check(at(cut_r, "status").as_string() == "incomplete" && at(at(cut_r, "incomplete_details"), "reason").as_string() == "max_output_tokens", "Responses budget verdict");
        }
        for (const bool response_endpoint : {false, true}) {
            const std::string path = response_endpoint ? "/v1/responses" : "/v1/chat/completions";
            auto body = response_endpoint ? parse_json(R"({"input":"JSON"})") : parse_json(R"({"messages":[{"role":"user","content":"JSON"}]})");
            const auto set_format = [&](const Json& f) {
                if (response_endpoint) { Json text = Json::object(); text.set("format", f); body.set("text", text); }
                else body.set("response_format", f);
            };
            const auto before = engine.prompts().size();
            auto bad_schema = parse_json(R"({"type":"json_schema","name":"answer","schema":{"type":"object","properties":{"age":{"type":"integer","minimum":0}}},"strict":false})");
            Json envelope = bad_schema;
            if (!response_endpoint) { envelope = parse_json(R"({"type":"json_schema"})"); envelope.set("json_schema", bad_schema); }
            set_format(envelope);
            auto error = parse_json(http(port, "POST", path, body.dump(), mode).body);
            const auto& e = at(error, "error");
            check(at(e, "param").as_string() == (response_endpoint ? "text.format.schema.properties.age.minimum" : "response_format.json_schema.schema.properties.age.minimum"), "full schema error path");
            check(at(e, "keyword").as_string() == "minimum" && at(e, "code").as_string() == "unsupported_schema_keyword", "schema error code and keyword");
            bad_schema.set("strict", Json("yes"));
            if (!response_endpoint) envelope.set("json_schema", bad_schema); else envelope = bad_schema;
            set_format(envelope); check(http(port, "POST", path, body.dump(), mode).status == 400, "strict requires boolean");
            set_format(parse_json(R"({"type":"json_object"})"));
            check(http(port, "POST", path, body.dump(), auth).status == 400, "unsupported engine refuses constraints");
            for (const auto& extra : {parse_json(R"({"stop":["x"]})"), parse_json(R"({"temperature":0.5})"), parse_json(R"({"tools":[{"type":"function","name":"f","function":{"name":"f","parameters":{}}}]})")}) {
                auto request = parse_json(body.dump()); for (const auto& [key, value] : extra.as_object()) request.set(key, value);
                check(http(port, "POST", path, request.dump(), mode).status == 400, "unsupported constraint combination refused");
            }
            check(engine.prompts().size() == before, "constraint refusals never submitted");
            for (const auto& tools : {Json(), Json::array()}) {
                auto request = parse_json(body.dump()); request.set("tools", tools);
                check(http(port, "POST", path, request.dump(), mode).status == 200, "null and empty tools admitted with constraints");
            }
            for (const auto& headers : {mode, quiet}) for (const auto& scenario : {"CONSTRAINT_FAILURE", "EXECUTION_ANSWER", "EXECUTION_REASONING"}) {
                auto request = parse_json(body.dump());
                const bool execution = std::string(scenario) != "CONSTRAINT_FAILURE";
                const bool reasoning = std::string(scenario) == "EXECUTION_REASONING";
                request.set("enable_thinking", Json(reasoning));
                if (response_endpoint) request.set("input", Json(scenario));
                else request.set("messages", parse_json(std::string("[{\"role\":\"user\",\"content\":\"") + scenario + "\"}]"));
                const auto reply = http(port, "POST", path, request.dump(), headers);
                check(reply.status == 500, "terminal constrained failure retains HTTP 500");
                const auto failure_body = parse_json(reply.body); verdict(failure_body, "failed");
                const auto& v = at(at(failure_body, "strix"), "constraint");
                check(at(v, "reason").as_string() == (execution ? "execution_failure" : "constraint_failure"), "HTTP preserves engine failure reason");
                check(at(v, "phase").as_string() == (reasoning ? "reasoning" : "answer") &&
                    at(v, "accepting").as_bool() == (execution && !reasoning), "HTTP preserves engine failure frontier");
                check(at(failure_body, "error").is_object(), "failure envelope retains error detail");
                if (headers == quiet) check(at(failure_body, "strix").size() == 3, "mandatory failure verdict with receipts off");
            }
        }
        {
            auto body = parse_json(R"({"messages":[{"role":"user","content":"JSON"}],"response_format":{"type":"json_object"},"enable_thinking":false,"stop":"","temperature":0.7,"tools":[],"tool_choice":"none"})");
            const auto reply = http(port, "POST", "/v1/chat/completions", body.dump(), auth + "X-Test-Config: constraints-greedy\r\n");
            check(reply.status == 200, "effective greedy downgrade, empty stop and disabled tools admitted"); verdict(parse_json(reply.body), "complete");
        }
        for (const auto& scenario : {std::string("CONSTRAINT_FAILURE"), std::string("CONSTRAINT_CONTEXT"), std::string("CONSTRAINT_REASON_EOS")}) {
            auto body = parse_json(R"({"input":"JSON","text":{"format":{"type":"json_object"}},"max_output_tokens":128})");
            body.set("input", Json(scenario));
            const bool failed = scenario == "CONSTRAINT_FAILURE";
            const std::string status = failed ? "failed" : "incomplete";
            const std::string reason = failed ? "constraint_failure" : scenario == "CONSTRAINT_CONTEXT" ? "context" : "reasoning_eos";
            for (const bool stream : {false, true}) {
                body.set("stream", Json(stream));
                const auto reply = http(port, "POST", "/v1/responses", body.dump(), quiet);
                if (failed && !stream) {  // a failure is not a reply: 500, as for unconstrained requests
                    check(reply.status == 500, "non-streamed constraint failure answers 500");
                    continue;
                }
                const auto events = stream ? sse_data(reply.body) : std::vector<Json>{};
                const auto terminal = stream ? at(events.back(), "response") : parse_json(reply.body);
                verdict(terminal, status);
                check(at(terminal, "status").as_string() == status, "Responses terminal failure/context/reasoning status");
                check(at(at(at(terminal, "strix"), "constraint"), "reason").as_string() == reason, "terminal constraint reason retained");
                if (!failed) check(at(at(terminal, "incomplete_details"), "reason").as_string() == reason, "Responses incomplete reason retained");
            }
            auto chat = parse_json(R"({"messages":[{"role":"user","content":"CONSTRAINT_FAILURE"}],"response_format":{"type":"json_object"},"stream":true})");
            if (failed) {
                const auto events = sse_data(http(port, "POST", "/v1/chat/completions", chat.dump(), quiet).body);
                verdict(events.back(), "failed");
            }
        }
        for (const auto& headers : {auth, mode}) {
            const auto caps = parse_json(http(port, "GET", "/v1/capabilities", "", headers).body);
            const bool enabled = headers == mode;
            check(at(caps, "json_object").as_bool() == enabled && at(caps, "json_schema").as_bool() == enabled, "capabilities follow engine execution");
            check(at(at(caps, "constraints"), "identities").size() == (enabled ? 2U : 0U), "constraint identity inventory");
        }
    }

    {
        const auto request = [&](const std::string& fields, const std::string& mode = "sample") {
            return http(port, "POST", "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1" + fields + "}",
                auth + "X-Test-Config: " + mode + "\r\n");
        };
        const auto filters_equal = [&](const Json& receipt, const std::string& source, double temperature, int top_k, double top_p, double min_p) {
            const auto& filters = at(receipt, "filters");
            for (const auto& [key, value] : std::vector<std::pair<std::string, double>>{{"temperature", temperature}, {"top_k", top_k}, {"top_p", top_p}, {"min_p", min_p}}) {
                check(at(at(filters, key), "value").as_double() == value, key + " resolved value");
                check(at(at(filters, key), "source").as_string() == source, key + " resolved source");
            }
        };
        const auto builtin = parse_json(request("").body);
        filters_equal(at(builtin, "strix"), "builtin", 0, 0, 1, 0);
        check(!at(builtin, "strix").find("seed"), "greedy receipt has no seed");
        const auto defaults = parse_json(request("", "defaults").body);
        const auto& receipt = at(defaults, "strix");
        filters_equal(receipt, "server", 0.7, 20, 0.9, 0.1);
        const auto captured = engine.sampling().back();
        check(captured.temperature == 0.7 && captured.top_k == 20 && captured.top_p == 0.9 && captured.min_p == 0.1, "engine receives server filters");
        check(captured.seeded && captured.seed == static_cast<std::uint64_t>(at(receipt, "seed").as_int()), "API draws and echoes effective seed");
        const auto explicit_values = parse_json(request(",\"temperature\":0.5,\"top_k\":5,\"top_p\":0.8,\"min_p\":0.2,\"seed\":42", "defaults").body);
        filters_equal(at(explicit_values, "strix"), "request", 0.5, 5, 0.8, 0.2);
        check(at(at(explicit_values, "strix"), "seed").as_int() == 42 && engine.sampling().back().seed == 42, "explicit seed echoed and submitted");
        for (const auto* key : {"temperature", "top_k", "top_p", "min_p"}) {
            const auto mixed = parse_json(request(std::string(",\"") + key + "\":null", "defaults").body);
            filters_equal(at(mixed, "strix"), "server", 0.7, 20, 0.9, 0.1);
        }
        for (const auto& [key, value] : std::vector<std::pair<std::string, std::string>>{{"temperature", "0"}, {"top_k", "0"}, {"top_p", "1"}, {"min_p", "0"}}) {
            for (const auto* mode : {"sample", "defaults"}) {
                const auto mixed = parse_json(request(",\"" + key + "\":" + value, mode).body);
                const auto& filters = at(at(mixed, "strix"), "filters");
                for (const auto* field : {"temperature", "top_k", "top_p", "min_p"})
                    check(at(at(filters, field), "source").as_string() == (key == field ? "request" : std::string(mode) == "defaults" ? "server" : "builtin"), "independent filter precedence");
            }
        }
        const auto ignored = parse_json(request(",\"temperature\":0,\"top_k\":10").body);
        check(at(at(ignored, "strix"), "note").as_string() == "filters ignored at temperature 0 (greedy)", "greedy filter note");
        check(!at(ignored, "strix").find("seed"), "greedy filters do not create seed");
        const auto downgraded = parse_json(request(",\"temperature\":1", "quiet").body);
        check(at(at(downgraded, "strix"), "note").as_string() == "temperature ignored: this engine decodes greedy (exact)", "downgrade note unchanged");
        for (const auto& [key, values] : std::vector<std::pair<std::string, std::vector<std::string>>>{
                {"temperature", {"-1", "2.1", "1e999", "\"1\"", "true", "[]", "{}"}},
                {"top_p", {"0", "-0.1", "1.1", "1e999", "\"1\"", "false", "[]", "{}"}},
                {"min_p", {"-0.1", "1", "1e999", "\"0\"", "false", "[]", "{}"}},
                {"top_k", {"1.5", "-1", "4294967296", "1e999", "\"1\"", "true", "[]", "{}"}},
                {"seed", {"-1", "42.5", "9223372036854775808", "18446744073709551616", "\"1\"", "true", "[]", "{}"}}}) {
            for (const auto& value : values) {
                const auto before = engine.prompts().size();
                const auto bad = request(",\"" + key + "\":" + value);
                check(bad.status == 400 && at(at(parse_json(bad.body), "error"), "param").as_string() == key, key + " rejects " + value);
                check(engine.prompts().size() == before, "invalid " + key + " never submitted");
            }
        }
        // integral values in any spelling are integers
        for (const auto* spelling : {"10.0", "1e1", "10"}) {
            const auto ok = request(std::string(",\"temperature\":1,\"top_k\":") + spelling + ",\"seed\":" + spelling);
            check(ok.status == 200 && engine.sampling().back().top_k == 10U && engine.sampling().back().seed == 10U, std::string("integral spelling ") + spelling);
        }
        check(request(",\"temperature\":1,\"top_k\":-0").status == 200 && engine.sampling().back().top_k == 0U, "top_k -0 is 0");
        // integer spellings are read exactly (never through a double)
        for (const auto& [spelling, expected] : std::vector<std::pair<std::string, std::uint64_t>>{
                 {"9007199254740993.0", 9007199254740993ULL}, {"9007199254740994.0", 9007199254740994ULL},
                 {"9.223372036854775807e18", 9223372036854775807ULL}, {"100e-2", 1ULL}, {"0.0e5", 0ULL}}) {
            const auto ok = request(",\"temperature\":1,\"seed\":" + spelling);
            check(ok.status == 200 && engine.sampling().back().seed == expected, "exact integer spelling " + spelling);
        }
        for (const char* spelling : {"9007199254740992.1", "9.223372036854775808e18", "1e-1", "100e-3", "-1e0"}) {
            const auto bad = request(std::string(",\"temperature\":1,\"seed\":") + spelling);
            check(bad.status == 400, std::string("non-integer or out-of-range seed refused ") + spelling);
        }
        check(request(",\"temperature\":1,\"top_k\":1.0000000000000001").status == 400, "top_k fraction below double precision refused");
        // the engine samples with float parameters: values that narrow to an invalid float are refused up front
        for (const auto& [extra, param] : std::vector<std::pair<std::string, std::string>>{
                 {",\"temperature\":1e-50", "temperature"}, {",\"temperature\":1,\"top_p\":1e-50", "top_p"},
                 {",\"temperature\":1,\"min_p\":0.99999999", "min_p"}}) {
            const auto bad = request(extra);
            check(bad.status == 400 && at(at(parse_json(bad.body), "error"), "param").as_string() == param, "float narrowing refused " + extra);
        }
        for (const auto* seed : {"0", "9223372036854775807"}) {
            const auto accepted = parse_json(request(std::string(",\"temperature\":1,\"seed\":") + seed).body);
            check(at(at(accepted, "strix"), "seed").number_source() == seed, "seed boundary accepted");
        }
        const auto caps = parse_json(http(port, "GET", "/v1/capabilities", "", auth + "X-Test-Config: defaults\r\n").body);
        Json cap_receipt = Json::object(); cap_receipt.set("filters", at(at(caps, "sampling"), "defaults"));
        filters_equal(cap_receipt, "server", 0.7, 20, 0.9, 0.1);
        check(at(at(caps, "sampling"), "seed_echo").as_bool(), "capabilities seed echo");
        for (const auto* route : {"/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses"}) {
            std::string body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"prompt\":\"Hi\",\"input\":\"Hi\",\"max_tokens\":1,\"max_output_tokens\":1,\"stream\":true}";
            const auto events = sse_data(http(port, "POST", route, body, auth + "X-Test-Config: defaults\r\n").body);
            std::vector<Json> receipts;
            for (const auto& event : events) {
                const Json* object = &event;
                if (const auto* r = event.find("response")) object = r;
                if (const auto* m = event.find("message")) object = m;
                if (const auto* r = object->find("strix")) receipts.push_back(*r);
            }
            check(receipts.size() >= 2, std::string(route) + " stream end receipts");
            if (receipts.size() >= 2) {
                check(at(receipts.front(), "seed").as_int() == at(receipts.back(), "seed").as_int(), "stream seed is stable");
                filters_equal(receipts.front(), "server", 0.7, 20, 0.9, 0.1);
                filters_equal(receipts.back(), "server", 0.7, 20, 0.9, 0.1);
            }
            const auto plain = parse_json(body);
            Json nonstream = Json::object();
            for (const auto& [key, value] : plain.as_object()) if (key != "stream") nonstream.set(key, value);
            const auto legacy = parse_json(http(port, "POST", route, nonstream.dump(), auth + "X-Test-Config: legacy\r\n").body);
            check(!at(legacy, "strix").find("filters") && !at(legacy, "strix").find("seed"), "receipt off omits new policy fields");
        }
    }
    {
        const auto r = http(port, "GET", "/v1/capabilities", "", auth);
        const auto doc = parse_json(r.body);
        for(const auto* name:{"tiled_suffix","mtp_fixed"}) {
            const auto& mode=at(at(doc,"session_numerical_modes"),name);
            check(at(mode,"requested").as_bool() && !at(mode,"effective").as_bool() &&
                at(mode,"suppression").as_string()=="baseline" && at(mode,"bit_changing").as_bool(),
                "capabilities disclose requested/effective numerical session modes");
        }
        const auto id = at(doc, "capabilities_id").as_string();
        check(r.status == 200 && at(doc, "object").as_string() == "strix.capabilities" && at(doc, "version").as_int() == 1, "capabilities default");
        check(http(port, "GET", "/v1/capabilities", "").status == 401, "capabilities authenticated");
        check(at(parse_json(http(port, "GET", "/v1/capabilities", "", auth).body), "capabilities_id").as_string() == id, "capabilities stable id");
        Json canonical = Json::object();
        for (const auto& [key, value] : doc.as_object()) if (key != "capabilities_id") canonical.set(key, value);
        std::uint64_t hash = 14695981039346656037ULL;
        for (const unsigned char c : canonical.dump()) { hash ^= c; hash *= 1099511628211ULL; }
        char hex[17]; std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(hash));
        check(id == hex, "capabilities FNV-1a canonical dump");
        for (const char* path : {"/api/status", "/health"})
            check(at(parse_json(http(port, "GET", path, "", auth).body), "capabilities_id").as_string() == id, "capabilities reference " + std::string(path));
        const auto models = parse_json(http(port, "GET", "/v1/models", "", auth).body);
        check(at(at(models, "data").as_array()[0], "capabilities_id").as_string() == id, "models capabilities reference");
        for (const char* key : {"json_schema", "strict_tools", "penalties", "logit_bias", "logprobs", "retry", "thinking_budget"})
            check(!at(doc, key).as_bool(), "unavailable capability " + std::string(key));
        check(!at(at(doc, "sampling"), "sample").as_bool(), "greedy-only capability");
        for (const auto* field : {"temperature", "top_k", "top_p", "min_p"}) {
            const auto& entry = at(at(at(doc, "sampling"), "defaults"), field);
            check(at(entry, "source").as_string() == "builtin", "capabilities builtin source");
            check(at(entry, "value").as_double() == (std::string(field) == "top_p" ? 1.0 : 0.0), "capabilities builtin value");
        }
        check(ApiConfig{}.sse_keepalive_seconds == 10.0, "default keepalive interval");
        check(ApiConfig{}.sse_progress_seconds == 1.0 && !ApiConfig{}.engine_progress, "default progress policy");
        const auto& progress = at(at(doc, "stream"), "progress");
        check(at(progress, "interval_seconds").as_double() == config.sse_progress_seconds && at(progress, "engine_reports").as_bool() && at(doc, "prefill_progress").as_bool(), "progress capability");
        const auto sampled = parse_json(http(port, "GET", "/v1/capabilities", "", auth + "X-Test-Config: sample\r\n").body);
        check(at(at(sampled, "sampling"), "sample").as_bool() && at(sampled, "engine_feature").as_bool() && at(sampled, "capabilities_id").as_string() != id, "sampling capabilities and engine hook");
        check(at(sampled,"rotary").dump() ==
            R"({"rope_mode":"native","scope":"server","yarn":{"supported":false,"scope":"CLI-only"}})",
            "Qwen HTTP rotary capabilities declare native and CLI-only YaRN");
        const auto rejected = parse_json(http(port, "GET", "/v1/capabilities", "", auth + "X-Test-Config: reject\r\n").body);
        check(at(at(rejected, "sampling"), "positive_temperature").as_string() == "reject" && !at(at(rejected, "endpoints"), "responses").as_bool(), "reject policy and disabled endpoint capabilities");
        const auto before = engine.prompts().size();
        check(http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"Hi"}],"temperature":1})", auth + "X-Test-Config: reject\r\n").status == 400 && engine.prompts().size() == before, "sampling reject semantics unchanged");
        const auto off = auth + "X-Test-Config: off\r\n";
        check(http(port, "GET", "/v1/capabilities", "", off).status == 404, "capabilities off");
        check(http(port, "POST", "/v1/responses", "{}", off).status == 404, "responses off");
        for (const char* path : {"/health", "/api/status"})
            check(!parse_json(http(port, "GET", path, "", off).body).find("capabilities_id"), "off omits id " + std::string(path));
        const auto off_models = parse_json(http(port, "GET", "/v1/models", "", off).body);
        check(!at(off_models, "data").as_array()[0].find("capabilities_id"), "off models omit id");
    }
    {
        // The real routes submit the same complete prompt and stable prefix, including literal argument bytes.
        std::vector<std::pair<Json, Json>> pairs;
        const auto pair = [&](const char* chat, const char* responses) { pairs.emplace_back(parse_json(chat), parse_json(responses)); };
        pair(R"({"messages":[{"role":"user","content":"Hi"}]})", R"({"input":"Hi"})");
        pair(R"({"messages":[{"role":"system","content":"Rules"},{"role":"user","content":"Hi"}]})", R"({"instructions":"Rules","input":"Hi"})");
        pair(R"({"messages":[{"role":"user","content":[{"type":"text","text":"  a\n"},{"type":"text","text":"\u03bb <think> <tool_call>  "}]}]})",
             R"({"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"  a\n"},{"type":"output_text","text":"\u03bb <think> <tool_call>  "}]}]})");
        pair(R"({"messages":[{"role":"developer","content":"Rules"},{"role":"user","content":"Hi"}]})", R"({"input":[{"role":"developer","content":"Rules"},{"role":"user","content":"Hi"}]})");
        pair(R"({"messages":[{"role":"user","content":"Hi"}],"tools":[{"type":"function","function":{"description":"desc","name":"f","parameters":{"type":"object"},"strict":false}}],"tool_choice":"none"})",
             R"({"input":"Hi","tools":[{"type":"function","description":"desc","name":"f","parameters":{"type":"object"},"strict":false}],"tool_choice":"none","n":1,"previous_response_id":null,"store":false,"background":false})");
        pair(R"({"messages":[{"role":"user","content":"Hi"}],"tools":[{"type":"function","function":{"description":"desc","name":"f","parameters":{"type":"object"},"strict":false}}],"tool_choice":"auto"})",
             R"({"input":"Hi","tools":[{"type":"function","description":"desc","name":"f","parameters":{"type":"object"},"strict":false}],"tool_choice":"auto"})");
        pair(R"({"messages":[{"role":"user","content":"First"},{"role":"assistant","content":"Answer","reasoning_content":"Reason\n"},{"role":"user","content":"Next"}],"preserve_thinking":true})",
             R"({"input":[{"role":"user","content":"First"},{"type":"reasoning","content":[{"type":"reasoning_text","text":"Reason\n"}]},{"role":"assistant","content":"Answer"},{"role":"user","content":"Next"}],"preserve_thinking":true})");
        pair(R"({"messages":[{"role":"user","content":"First"},{"role":"assistant","content":"Answer","reasoning_content":"Summary"},{"role":"user","content":"Next"}]})",
             R"({"input":[{"role":"user","content":"First"},{"type":"reasoning","summary":[{"type":"summary_text","text":"Summary"}]},{"role":"assistant","content":"Answer"},{"role":"user","content":"Next"}]})");
        for (const char* effort : {"none", "low", "high"}) {
            auto c = parse_json(R"({"messages":[{"role":"user","content":"Hi"}]})"); c.set("reasoning_effort", Json(effort));
            auto r = parse_json(R"({"input":"Hi"})"); Json reasoning = Json::object(); reasoning.set("effort", Json(effort)); r.set("reasoning", std::move(reasoning));
            pairs.emplace_back(std::move(c), std::move(r));
        }
        auto c = parse_json(R"({"messages":[{"role":"user","content":"Call"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":""}}]},{"role":"tool","tool_call_id":"c1","content":"result"}],"tools":[{"type":"function","function":{"name":"f","description":"function","parameters":{"type":"object","properties":{"s":{"type":"string"}}}}}]})");
        auto r = parse_json(R"({"input":[{"role":"user","content":"Call"},{"type":"function_call","call_id":"c1","name":"f","arguments":""},{"type":"function_call_output","call_id":"c1","output":"result"}],"tools":[{"type":"function","name":"f","description":"function","parameters":{"type":"object","properties":{"s":{"type":"string"}}}}]})");
        const std::string args = " {\n\"s\": \"quote \\\" backslash \\\\ newline \\n </tool_call> </parameter>\"\n} ";
        auto call = parse_json(R"({"id":"c1","type":"function","function":{"name":"f"}})");
        Json function = Json::object(); function.set("name", Json("f")); function.set("arguments", Json(args)); call.set("function", std::move(function));
        auto& messages = c.as_object()[0].second.as_array(); Json calls = Json::array(); calls.push(std::move(call)); messages[1].set("tool_calls", std::move(calls));
        r.as_object()[0].second.as_array()[1].set("arguments", Json(args));
        pairs.emplace_back(std::move(c), std::move(r));
        // an assistant message followed by two calls is one assistant turn with two tool calls
        pair(R"({"messages":[{"role":"user","content":"Both"},{"role":"assistant","content":"Calling","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{\"s\":\"1\"}"}},{"id":"b","type":"function","function":{"name":"f","arguments":"{\"s\":\"2\"}"}}]},{"role":"tool","tool_call_id":"a","content":"one"},{"role":"tool","tool_call_id":"b","content":"two"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]})",
             R"({"input":[{"role":"user","content":"Both"},{"role":"assistant","content":"Calling"},{"type":"function_call","call_id":"a","name":"f","arguments":"{\"s\":\"1\"}"},{"type":"function_call","call_id":"b","name":"f","arguments":"{\"s\":\"2\"}"},{"type":"function_call_output","call_id":"a","output":"one"},{"type":"function_call_output","call_id":"b","output":"two"}],"tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]})");
        pair(R"({"messages":[{"role":"user","content":"Both"},{"role":"assistant","content":"","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"b","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"a","content":"one"},{"role":"tool","tool_call_id":"b","content":"two"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]})",
             R"({"input":[{"role":"user","content":"Both"},{"type":"function_call","call_id":"a","name":"f","arguments":"{}"},{"type":"function_call","call_id":"b","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"a","output":"one"},{"type":"function_call_output","call_id":"b","output":"two"}],"tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]})");
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            const auto before = engine.prompts().size();
            const auto a = http(port, "POST", "/v1/chat/completions", pairs[i].first.dump(), auth);
            const auto b = http(port, "POST", "/v1/responses", pairs[i].second.dump(), auth);
            const auto captured = engine.prompts();
            check(a.status == 200 && b.status == 200, "responses prompt pair accepted " + std::to_string(i) + " " + b.body);
            check(captured.size() == before + 2U && captured[before] == captured[before + 1U], "responses_prompt_ids_equal_chat " + std::to_string(i));
        }
    }
    {
        const std::vector<std::pair<std::string, std::string>> invalid{
            {R"({"previous_response_id":"resp_x"})", "previous_response_id"}, {R"({"store":true})", "store"},
            {R"({"conversation":"conv_1"})", "conversation"}, {R"({"background":true})", "background"},
            {R"({"input":[{"type":"unknown"}]})", "input[0].type"},
            {R"({"input":[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]})", "input[0].content"},
            {R"({"input":[{"role":"user","content":[{"type":"input_file","file_id":"x"}]}]})", "input[0].content"},
            {R"({"tools":[{"type":"web_search"}]})", "tools[0].type"},
            {R"({"input":[{"type":"reasoning","encrypted_content":"x"}]})", "input[0]"},
            {R"({"input":[{"type":"reasoning","summary":[],"encrypted_content":"x"}]})", "input[0]"},
            {R"({"input":[{"type":"function_call_output","call_id":"missing","output":"x"}]})", "input[0].call_id"},
            {R"({"input":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]})", "input[1].call_id"},
            {R"({"input":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"x"},{"type":"function_call_output","call_id":"c","output":"x"}]})", "input[2].call_id"},
            {R"({"text":{"format":{"type":"json_schema"}}})", "text.format"}, {R"({"text":{"format":{"type":"json_object"}}})", "text.format"},
            {R"({"text":{"format":{"type":"other"}}})", "text.format"},
            {R"({"tools":[{"type":"function","name":"f","strict":true}]})", "tools[0].strict"},
            {R"({"tool_choice":"required"})", "tool_choice"}, {R"({"tool_choice":{"type":"function","name":"f"}})", "tool_choice"},
            {R"({"n":"1"})", "n"}, {R"({"n":2})", "n"}, {R"({"instructions":[]})", "instructions"}
        };
        const auto before = engine.prompts().size();
        for (const auto& [body, param] : invalid) {
            auto request = parse_json(body); if (!request.find("input")) request.set("input", Json("Hi"));
            const auto reply = http(port, "POST", "/v1/responses", request.dump(), auth);
            check(reply.status == 400 && at(at(parse_json(reply.body), "error"), "param").as_string() == param, "responses_validation " + param + " " + reply.body);
        }
        check(engine.prompts().size() == before, "responses_validation never invokes engine");
    }
    {
        const auto plain = http(port, "POST", "/v1/responses", R"({"input":"Hi","temperature":1,"text":{"format":{"type":"text"}}})", auth);
        const auto full = parse_json(plain.body);
        check(plain.status == 200 && at(full, "object").as_string() == "response" && at(full, "status").as_string() == "completed", "responses non-stream shape");
        check(at(full, "id").as_string().starts_with("resp_") && at(full, "created_at").is_number(), "responses id and time");
        const auto& output = at(full, "output").as_array();
        check(output.size() == 2U && at(output[0], "type").as_string() == "reasoning" && at(output[0], "summary").size() == 0U, "responses reasoning item");
        check(at(output[1], "role").as_string() == "assistant" && at(output[1], "status").as_string() == "completed", "responses message item");
        const auto expected_text = at(at(output[1], "content").as_array()[0], "text").as_string();
        const auto expected_reasoning = at(at(output[0], "content").as_array()[0], "text").as_string();
        check(at(at(output[1], "content").as_array()[0], "annotations").size() == 0U, "responses annotations");
        const auto& usage = at(full, "usage");
        check(at(usage, "input_tokens").as_int() + at(usage, "output_tokens").as_int() == at(usage, "total_tokens").as_int() &&
            at(at(usage, "input_tokens_details"), "cached_tokens").as_int() == 0, "responses usage");
        check(at(at(full, "strix"), "sampling_requested").as_string() == "sample" && at(at(full, "strix"), "sampling_effective").as_string() == "greedy" &&
            !at(at(full, "strix"), "note").as_string().empty(), "responses downgrade receipt");
        const auto incomplete = parse_json(http(port, "POST", "/v1/responses", R"({"input":"Hi","max_output_tokens":4})", auth).body);
        check(at(incomplete, "status").as_string() == "incomplete" && at(at(incomplete, "incomplete_details"), "reason").as_string() == "max_output_tokens", "responses non-stream incomplete");
        const auto status = parse_json(http(port, "GET", "/api/status", "").body);
        check(at(at(status, "requests").as_array()[0], "route").as_string() == "responses", "responses recent request route");
        for (const char* input : {"Hi", "TOOL", "ENGINE_ERROR", "length"}) {
            Json request = Json::object(); request.set("input", Json(input)); request.set("stream", Json(true));
            if (std::string(input) == "TOOL") request.set("tools", parse_json(R"([{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"days":{"type":"integer"}}}}])"));
            if (std::string(input) == "length") request.set("max_output_tokens", Json(4));
            const auto r = http(port, "POST", "/v1/responses", request.dump(), auth);
            check(r.status == 200 && r.body.find("[DONE]") == std::string::npos, "responses SSE transport");
            const auto events = sse_data(r.body);
            check(at(events[0], "type").as_string() == "response.created" && at(events[1], "type").as_string() == "response.in_progress", "responses created/progress order");
            check(at(events[0], "response").find("strix"), "responses first receipt");
            std::set<std::int64_t> added, parts, texts_done, parts_done, items_done;
            std::string text, reasoning, arguments;
            std::int64_t sequence = 0;
            int terminal = 0;
            for (const auto& e : events) {
                const auto type = at(e, "type").as_string();
                check(at(e, "sequence_number").as_int() == sequence++, "responses sequence_number");
                check(r.body.find("event: " + type + "\ndata: " + e.dump() + "\n\n") != std::string::npos, "responses event == data.type");
                if (type == "response.output_item.added") added.insert(at(e, "output_index").as_int());
                if (type == "response.content_part.added") {
                    const auto i = at(e, "output_index").as_int(); check(added.count(i) && !items_done.count(i), "part follows item added"); parts.insert(i);
                }
                if (type.ends_with(".delta")) {
                    const auto i = at(e, "output_index").as_int(); check(added.count(i) && !items_done.count(i) && !texts_done.count(i), "delta ordering");
                    if (type == "response.output_text.delta") { check(parts.count(i), "text part added before delta"); text += at(e, "delta").as_string(); }
                    if (type == "response.reasoning_text.delta") reasoning += at(e, "delta").as_string();
                    if (type == "response.function_call_arguments.delta") arguments += at(e, "delta").as_string();
                }
                if (type == "response.output_text.done" || type == "response.reasoning_text.done" || type == "response.function_call_arguments.done") texts_done.insert(at(e, "output_index").as_int());
                if (type == "response.content_part.done") {
                    const auto i = at(e, "output_index").as_int(); check(texts_done.count(i), "part done after text done"); parts_done.insert(i);
                }
                if (type == "response.output_item.done") {
                    const auto i = at(e, "output_index").as_int(); check(added.count(i) && texts_done.count(i) && (!parts.count(i) || parts_done.count(i)), "item done ordering"); items_done.insert(i);
                }
                if (type == "response.completed" || type == "response.incomplete" || type == "response.failed") {
                    ++terminal; check(items_done == added && at(e, "response").find("strix"), "terminal complete output and receipt");
                }
            }
            check(terminal == 1, "exactly one responses terminal");
            if (std::string(input) == "Hi") check(text == expected_text && reasoning == expected_reasoning, "responses deltas equal non-stream text");
            if (std::string(input) == "TOOL") {
                const auto& result = at(events.back(), "response"); const auto& calls = at(result, "output").as_array();
                check(at(calls.back(), "type").as_string() == "function_call" && !at(calls.back(), "call_id").as_string().empty() && at(calls.back(), "arguments").as_string() == arguments, "responses tool item and arguments");
                request.set("stream", Json(false)); const auto nonstream = parse_json(http(port, "POST", "/v1/responses", request.dump(), auth).body);
                check(at(at(nonstream, "output").as_array().back(), "arguments").as_string() == arguments, "responses tool arguments equal non-stream");
            }
            if (std::string(input) == "length") check(at(at(at(events.back(), "response"), "incomplete_details"), "reason").as_string() == "max_output_tokens", "responses incomplete reason");
            if (std::string(input) == "ENGINE_ERROR") check(at(events.back(), "type").as_string() == "response.failed", "responses engine failure terminal");
        }
        check(http(port, "POST", "/v1/responses", R"({"input":"ENGINE_ERROR"})", auth).status == 500, "responses non-stream engine failure");
    }
    {
        const auto body = R"({"messages":[{"role":"user","content":"Hi"}],"stream":true,"temperature":1,"max_tokens":1000})";
        const auto events = sse_data(http(port, "POST", "/v1/chat/completions", body, auth).body);
        const auto& receipt = at(events.front(), "strix");
        check(at(receipt, "sampling_requested").as_string() == "sample" && at(receipt, "sampling_effective").as_string() == "greedy", "first chat receipt without usage");
        check(at(receipt, "max_tokens_requested").as_int() == 1000 && at(receipt, "max_tokens_effective").as_int() == 1000 && receipt.find("note"), "first chat policy limit and note");
        check(events.back().find("strix"), "final chat receipt without usage");
        const auto old = sse_data(http(port, "POST", "/v1/chat/completions", body, auth + "X-Test-Config: legacy\r\n").body);
        check(!old.front().find("strix") && !old.back().find("strix"), "receipts off");
        // Build every old chunk with the original ordered fields, normalizing only the random id and timestamp.
        const auto normalize = [](const Json& value) {
            Json out = parse_json(value.dump()); out.set("id", Json("ID")); out.set("created", Json(0)); return out.dump();
        };
        for (const auto& e : old) {
            Json golden = Json::object(); golden.set("id", Json("ID")); golden.set("object", Json("chat.completion.chunk")); golden.set("created", Json(0));
            golden.set("model", Json("test-model")); golden.set("choices", at(e, "choices"));
            check(normalize(e) == golden.dump(), "old chat chunk golden bytes");
        }
        check(events.size() == old.size(), "receipt does not add chat chunks");
        for (std::size_t i = 0; i < events.size() && i < old.size(); ++i)
            check(at(events[i], "choices").dump() == at(old[i], "choices").dump(), "receipt leaves chat deltas unchanged");
    }
    {
        for (const char* route : {"/v1/chat/completions", "/v1/messages", "/v1/completions", "/v1/responses"}) {
            auto request = parse_json(R"({"stream":true,"max_tokens":1000,"return_token_ids":true})");
            const std::string path = route;
            if (path == "/v1/responses") request.set("input", Json("PROGRESS"));
            else if (path == "/v1/completions") request.set("prompt", Json("PROGRESS"));
            else request.set("messages", parse_json(R"([{"role":"user","content":"PROGRESS"}])"));
            const auto active = http(port, "POST", path, request.dump(), auth);
            const auto off = http(port, "POST", path, request.dump(), auth + "X-Test-Config: progress-off\r\n");
            request.set("strix", parse_json(R"({"progress":false})"));
            const auto disabled = http(port, "POST", path, request.dump(), auth);
            check(off.body.find(": progress ") == std::string::npos && disabled.body.find(": progress ") == std::string::npos, "progress off switches " + path);
            std::int64_t prefilled = 0, cached = 0;
            bool saw_prefill = false, saw_decode = false;
            for (auto pos = active.body.find(": progress "); pos != std::string::npos; pos = active.body.find(": progress ", pos + 1)) {
                const auto value = parse_json(active.body.substr(pos + 11, active.body.find("\n\n", pos) - pos - 11));
                const auto n = at(value, "prefilled_tokens").as_int(), c = at(value, "cached_tokens").as_int(), t = at(value, "total_tokens").as_int();
                check(n >= prefilled && c >= cached && n + c <= t, "progress counters " + path);
                check(value.dump() == "{\"phase\":\"" + at(value, "phase").as_string() + "\",\"prefilled_tokens\":" + std::to_string(n) + ",\"cached_tokens\":" + std::to_string(c) + ",\"total_tokens\":" + std::to_string(t) + "}", "compact ordered progress JSON " + path);
                prefilled = n; cached = c;
                if (at(value, "phase").as_string() == "prefill") { check(!saw_decode, "progress phase order " + path); saw_prefill = true; }
                else { check(saw_prefill, "prefill before decode " + path); saw_decode = true; }
            }
            check(saw_prefill && saw_decode, "progress phases " + path);
            check(active.body.find(": keepalive") == std::string::npos, "progress resets keepalive " + path);
            const auto a = sse_data(active.body), b = sse_data(off.body), c = sse_data(disabled.body);
            const std::function<Json(const Json&)> payload = [&](const Json& value) -> Json {
                if (value.is_array()) { Json out = Json::array(); for (const auto& v : value.as_array()) out.push(payload(v)); return out; }
                if (!value.is_object()) return value;
                Json out = Json::object();
                for (const auto& [key, v] : value.as_object())
                    if (key != "id" && key != "item_id" && key != "created" && key != "created_at" && key != "strix") out.set(key, payload(v));
                return out;
            };
            check(a.size() == b.size() && a.size() == c.size(), "progress data count " + path);
            for (std::size_t i = 0; i < a.size() && i < b.size() && i < c.size(); ++i) {
                check(payload(a[i]).dump() == payload(b[i]).dump() && payload(a[i]).dump() == payload(c[i]).dump(), "progress data events identical " + path);
                // IDs and timestamps are request-local; compare the generation payloads and final token IDs.
                for (const char* key : {"choices", "delta", "token_ids"})
                    if (const auto* v = a[i].find(key)) check(b[i].find(key) && c[i].find(key) && v->dump() == at(b[i], key).dump() && v->dump() == at(c[i], key).dump(), "progress data unchanged " + path);
            }
            const auto ids = [&](const std::vector<Json>& values) {
                const auto& last = values.back();
                return at(path == "/v1/responses" ? at(at(last, "response"), "strix") : at(last, "strix"), "token_ids").dump();
            };
            check(ids(a) == ids(b) && ids(a) == ids(c), "progress token IDs identical " + path);
            request.set("stream", Json(false)); request.set("strix", Json::object());
            const auto plain = http(port, "POST", path, request.dump(), auth);
            check(plain.body.find(": progress ") == std::string::npos, "non-stream has no progress " + path);
        }
    }
    {
        for (const char* route : {"/v1/chat/completions", "/v1/messages", "/v1/completions", "/v1/responses"}) {
            auto request = parse_json(R"({"stream":true,"max_tokens":1000,"return_token_ids":true})");
            const std::string path = route;
            if (path == "/v1/responses") request.set("input", Json("SLOW"));
            else if (path == "/v1/completions") request.set("prompt", Json("SLOW"));
            else request.set("messages", parse_json(R"([{"role":"user","content":"SLOW"}])"));
            const auto active = http(port, "POST", path, request.dump(), auth);
            // the opening events (role chunk, message_start, response.created + in_progress) are written at once; the
            // slow engine's first text comes after keepalive comments
            const auto keepalive = active.body.find(": keepalive\n\n");
            std::size_t opening = 0;
            for (auto at_ = active.body.find("data: "); at_ < keepalive; at_ = active.body.find("data: ", at_ + 1)) ++opening;
            const std::size_t expected = path == "/v1/responses" ? 2U : 1U;
            check(active.body.find(": progress ") == std::string::npos, "no fabricated progress " + path);
            check(keepalive != std::string::npos && opening == expected, "keepalive before the first generated text " + path);
            const auto quiet = http(port, "POST", path, request.dump(), auth + "X-Test-Config: quiet\r\n");
            request.set("strix", parse_json(R"({"keepalive":false})"));
            const auto disabled = http(port, "POST", path, request.dump(), auth);
            check(quiet.body.find(": keepalive") == std::string::npos && disabled.body.find(": keepalive") == std::string::npos, "keepalive off switches " + path);
            const auto a = sse_data(active.body), b = sse_data(quiet.body), c = sse_data(disabled.body);
            check(a.size() == b.size() && a.size() == c.size(), "keepalive event count unchanged " + path);
            const auto ids = [&](const std::vector<Json>& values) {
                const auto& last = values.back();
                return at(path == "/v1/responses" ? at(at(last, "response"), "strix") : at(last, "strix"), "token_ids").dump();
            };
            check(ids(a) == ids(b) && ids(a) == ids(c), "keepalive generated ids identical " + path);
            for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
                if (path == "/v1/chat/completions" || path == "/v1/completions") check(at(a[i], "choices").dump() == at(b[i], "choices").dump(), "keepalive deltas unchanged " + path);
                if (const auto* delta = a[i].find("delta")) check(delta->dump() == at(b[i], "delta").dump(), "keepalive delta unchanged " + path);
            }
            if (path == "/v1/messages") check(at(a.front(), "message").find("strix") && a.back().find("strix"), "messages first and final receipt");
            if (path == "/v1/completions") check(a.front().find("strix") && a.back().find("strix"), "completions first and final receipt");
        }
    }
    {
        // A call is completed only when its frame closed: as the budget grows, every call status goes incomplete ->
        // completed exactly once (streamed or not), even while the response itself is still incomplete.
        const auto tools = R"([{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}])";
        for (const bool stream : {false, true}) {
            std::string history;  // per limit: '-' no call, 'i' incomplete call, 'c' completed call
            bool completed_in_incomplete_response = false;
            for (int limit = 20; limit <= 140; ++limit) {
                auto request = parse_json(R"({"input":"TOOL"})"); request.set("tools", parse_json(tools));
                request.set("max_output_tokens", Json(limit)); request.set("stream", Json(stream));
                const auto r = http(port, "POST", "/v1/responses", request.dump(), auth);
                const Json response = stream ? at(sse_data(r.body).back(), "response") : parse_json(r.body);
                char state = '-';
                for (const auto& item : at(response, "output").as_array())
                    if (at(item, "type").as_string() == "function_call") state = at(item, "status").as_string() == "completed" ? 'c' : 'i';
                if (state == 'c' && at(response, "status").as_string() == "incomplete") completed_in_incomplete_response = true;
                history += state;
            }
            const auto first_i = history.find('i'), first_c = history.find('c');
            check(first_i != std::string::npos && first_c != std::string::npos && first_i < first_c &&
                  history.find('-', first_i) == std::string::npos && history.find('i', first_c) == std::string::npos,
                  std::string("call status monotone in the budget ") + (stream ? "stream " : "") + history);
            check(completed_in_incomplete_response, "a closed call stays completed when the budget then ends the response");
        }
        // A stop string inside the arguments cuts the frame: the call is incomplete although the response stopped.
        for (const bool stream : {false, true}) {
            auto request = parse_json(R"({"input":"TOOL","stop":["slo"]})"); request.set("tools", parse_json(tools)); request.set("stream", Json(stream));
            const auto r = http(port, "POST", "/v1/responses", request.dump(), auth);
            const Json response = stream ? at(sse_data(r.body).back(), "response") : parse_json(r.body);
            int calls = 0;
            for (const auto& item : at(response, "output").as_array())
                if (at(item, "type").as_string() == "function_call") { ++calls; check(at(item, "status").as_string() == "incomplete", "stop-cut call incomplete"); }
            check(calls == 1, "stop-cut call reported " + r.body.substr(0, 200));
        }
        const auto cut = parse_assistant_output("x</think>\n\n<tool_call>\n<function=f>\n<parameter=a>\nv", true, Json());
        const auto closed = parse_assistant_output("<tool_call>\n<function=f>\n</function>\n</tool_call><tool_call>\n<function=g>", false, Json());
        check(cut.call_complete == std::vector<bool>{false} && closed.call_complete == std::vector<bool>{true, false}, "parser call_complete");
        check(parse_assistant_output("only reasoning", true, Json()).tool_calls.is_array(), "tool_calls array inside thinking");
    }
    {
        // Generation-affecting fields the server does not execute are refused before the engine runs; neutral values pass.
        const auto before = engine.prompts().size();
        const std::vector<std::pair<std::string, std::string>> refused = {
            {"prefill_exact_fp16", "true"}, {"prefill_exact_fp16", "false"},
            {"prefill_disable", R"(["gdn_chunked"])"},
            {"presence_penalty", R"(0.5)"}, {"frequency_penalty", R"(-1)"}, {"repetition_penalty", R"(1.1)"},
            {"logit_bias", R"({"5":-100})"}, {"logprobs", R"(true)"}, {"top_logprobs", R"(3)"},
            {"response_format", R"({"type":"json_object"})"}, {"parallel_tool_calls", R"(false)"}, {"tool_choice", R"("required")"},
            {"tool_choice", R"({"type":"function","function":{"name":"f"}})"},
        };
        for (const auto& [key, value] : refused)
            for (const char* route : {"/v1/chat/completions", "/v1/responses"}) {
                if (std::string(route) == "/v1/responses" && key == "tool_choice") continue;  // the Responses translation refuses it itself
                auto request = parse_json(std::string(route) == "/v1/responses" ? R"({"input":"Hi"})" : R"({"messages":[{"role":"user","content":"Hi"}]})");
                request.set(key, parse_json(value));
                const auto r = http(port, "POST", route, request.dump(), auth);
                check(r.status == 400 && at(at(parse_json(r.body), "error"), "param").as_string() == key, "refused " + key + " on " + route + ": " + r.body);
            }
        auto strict = parse_json(R"({"messages":[{"role":"user","content":"Hi"}],"tools":[{"type":"function","function":{"name":"f","strict":true,"parameters":{"type":"object"}}}]})");
        check(http(port, "POST", "/v1/chat/completions", strict.dump(), auth).status == 400, "strict chat tool refused");
        const auto any = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tool_choice":{"type":"any"}})", auth);
        check(any.status == 400, "messages forced tool_choice refused");
        for (const char* body : {R"({"input":"Hi","include":["reasoning.encrypted_content"]})", R"({"input":"Hi","truncation":"auto"})", R"({"input":"Hi","max_tool_calls":2})"})
            check(http(port, "POST", "/v1/responses", body, auth).status == 400, std::string("responses refuses ") + body);
        check(engine.prompts().size() == before, "refused requests never reach the engine");
        const auto neutral = http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"Hi"}],"presence_penalty":0,"frequency_penalty":0,"logit_bias":{},"logprobs":false,"top_logprobs":0,"response_format":{"type":"text"},"parallel_tool_calls":true,"tool_choice":"auto"})", auth);
        check(neutral.status == 200, "neutral values pass " + neutral.body);
        const auto neutral_messages = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tool_choice":{"type":"auto"}})", auth);
        check(neutral_messages.status == 200, "messages auto tool_choice passes " + neutral_messages.body);
    }
    {
        // A tool-call header cut before any call parses: no call, but the response is incomplete (unfinished_tool_call).
        const auto header = parse_assistant_output("<tool_call>\n<function=nex", false, Json());
        check(header.tool_calls.size() == 0U && header.unfinished_call, "parser keeps the unfinished header");
        for (const bool stream : {false, true}) {
            auto request = parse_json(R"({"input":"TOOL","stop":["_weather"],"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object"}}]})");
            request.set("stream", Json(stream));
            const auto r = http(port, "POST", "/v1/responses", request.dump(), auth);
            const Json response = stream ? at(sse_data(r.body).back(), "response") : parse_json(r.body);
            check(at(response, "status").as_string() == "incomplete" &&
                  at(at(response, "incomplete_details"), "reason").as_string() == "unfinished_tool_call", "cut header => incomplete " + r.body.substr(0, 300));
        }
        // Without tools an unclosed marker is plain text: the reply stays completed.
        for (const bool stream : {false, true}) {
            auto request = parse_json(R"({"input":"TOOL","stop":["_weather"]})"); request.set("stream", Json(stream));
            const auto r = http(port, "POST", "/v1/responses", request.dump(), auth);
            const Json response = stream ? at(sse_data(r.body).back(), "response") : parse_json(r.body);
            check(at(response, "status").as_string() == "completed", "literal marker without tools stays completed");
        }
        // DeepSeek: a calls block that never closed is an unfinished call; a closed malformed one is not.
        const std::string dsml = "\xEF\xBD\x9C" "DSML\xEF\xBD\x9C";
        const auto ds_cut = parse_deepseek_v41_output("Answer.\n\n<" + dsml + " calls>\n<" + dsml + " invoke name=\"f\">\n", false, Json());
        check(ds_cut.unfinished_call && ds_cut.tool_calls.size() == 0U, "deepseek unfinished calls block");
        check(!parse_deepseek_v41_output("Answer.", false, Json()).unfinished_call, "deepseek plain answer finished");
        check(http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"output_config":{"effort":null,"format":null}})", auth).status == 200,
              "messages output_config null members pass");
        // Anthropic: tool_choice none renders no tools; unexecuted constraints are refused; the budget is stated.
        const auto before = engine.prompts().size();
        const auto plain = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}]})", auth);
        const auto none = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tools":[{"name":"f","input_schema":{"type":"object"}}],"tool_choice":{"type":"none"}})", auth);
        const auto prompts = engine.prompts();
        check(plain.status == 200 && none.status == 200 && prompts.size() == before + 2U && prompts[before] == prompts[before + 1U], "messages tool_choice none renders no tools");
        for (const char* body : {
                 R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tool_choice":{"type":"auto","disable_parallel_tool_use":true}})",
                 R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tools":[{"name":"f","strict":true,"input_schema":{"type":"object"}}]})",
                 R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"thinking":{"type":"adaptive"}})",
                 R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"output_config":{"effort":"low"}})"})
            check(http(port, "POST", "/v1/messages", body, auth).status == 400, std::string("messages refuses ") + body);
        const auto budget = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"thinking":{"type":"enabled","budget_tokens":1024}})", auth);
        check(budget.status == 200 && budget.body.find("budget_tokens not enforced") != std::string::npos, "thinking budget stated as not enforced");
        const auto budget_stream = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"stream":true,"messages":[{"role":"user","content":"Hi"}],"thinking":{"type":"enabled","budget_tokens":1024}})",
            auth + "X-Test-Config: legacy\r\n");
        check(budget_stream.body.find("budget_tokens not enforced") != std::string::npos, "budget note kept in a stream without receipts");
        check(http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tool_choice":{"type":"auto","disable_parallel_tool_use":false}})", auth).status == 200,
              "disable_parallel_tool_use false passes");
        // Completions: one candidate, no suffix, no scores; chat: no legacy functions.
        for (const char* body : {R"({"prompt":"Hi","max_tokens":4,"n":2})", R"({"prompt":"Hi","max_tokens":4,"best_of":2})",
                 R"({"prompt":"Hi","max_tokens":4,"suffix":"end"})", R"({"prompt":"Hi","max_tokens":4,"logprobs":0})"})
            check(http(port, "POST", "/v1/completions", body, auth).status == 400, std::string("completions refuses ") + body);
        check(http(port, "POST", "/v1/completions", R"({"prompt":"Hi","max_tokens":4,"n":1,"best_of":1,"suffix":null,"logprobs":null})", auth).status == 200, "completions neutral values pass");
        for (const char* body : {R"({"messages":[{"role":"user","content":"Hi"}],"functions":[{"name":"f","parameters":{"type":"object"}}]})",
                 R"({"messages":[{"role":"user","content":"Hi"}],"function_call":{"name":"f"}})"})
            check(http(port, "POST", "/v1/chat/completions", body, auth).status == 400, std::string("chat refuses ") + body);
        // Responses: null is absent; verbosity other than medium is refused.
        const auto nulls = http(port, "POST", "/v1/responses", R"({"input":"Hi","conversation":null,"instructions":null,"reasoning":{"effort":null},"text":{"verbosity":null,"format":null},"tools":[{"type":"function","name":"f","strict":null,"parameters":{"type":"object"}}],"tool_choice":null,"previous_response_id":null})", auth);
        check(nulls.status == 200, "responses nullable fields " + nulls.body.substr(0, 200));
        check(http(port, "POST", "/v1/responses", R"({"input":"Hi","reasoning":null})", auth).status == 200, "responses reasoning null");
        check(http(port, "POST", "/v1/responses", R"({"input":"Hi","text":{"verbosity":"low"}})", auth).status == 400, "responses refuses verbosity low");
    }
    {
        // A failed stream: response.failed, and no item carries a failed status.
        const auto r = http(port, "POST", "/v1/responses", R"({"input":"ENGINE_ERROR","stream":true})", auth);
        const auto events = sse_data(r.body);
        check(at(events.back(), "type").as_string() == "response.failed", "failed stream ends response.failed");
        for (const auto& e : events)
            if (const auto* item = e.find("item"); item && item->find("status"))
                check(at(*item, "status").as_string() != "failed", "items never carry a failed status");
        // Output text carries (empty) logprobs arrays while token scores are unavailable.
        const auto ok = sse_data(http(port, "POST", "/v1/responses", R"({"input":"Hi","stream":true})", auth).body);
        int text_events = 0;
        for (const auto& e : ok) {
            const auto& type = at(e, "type").as_string();
            if (type == "response.output_text.delta" || type == "response.output_text.done") { ++text_events; check(at(e, "logprobs").is_array(), "text event logprobs " + type); }
            if (type == "response.content_part.done" && at(at(e, "part"), "type").as_string() == "output_text") check(at(at(e, "part"), "logprobs").is_array(), "output_text part logprobs");
        }
        check(text_events >= 2, "output text events seen");
        // Reasoning summaries are refused before the engine runs.
        const auto before = engine.prompts().size();
        for (const char* body : {R"({"input":"Hi","reasoning":{"summary":"detailed"}})", R"({"input":"Hi","reasoning":{"generate_summary":"auto"}})"}) {
            const auto rejected = http(port, "POST", "/v1/responses", body, auth);
            check(rejected.status == 400 && rejected.body.find("reasoning.") != std::string::npos, "reasoning summary refused");
        }
        check(engine.prompts().size() == before, "refused summaries never reach the engine");
    }
    {
        // Prefix-only statistics are available even with sessions disabled.
        const auto reply=http(port,"GET","/v1/cache/stats","",auth);
        const auto stats=parse_json(reply.body);
        const auto& state=at(at(stats,"engine"),"state_cache");
        check(reply.status==200 && at(state,"schema").as_int()==1 && at(state,"prefix_enabled").as_bool() &&
            !at(state,"session_enabled").as_bool() && !at(state,"pool_present").as_bool(),"cache snapshot schema and flags");
        check(at(at(state,"prefix"),"no_store_skips").as_int()==7 && at(stats,"parked").as_int()==0,"cache snapshot counters");
        check(at(at(state,"prefix"),"evictions_superseded").as_int()==3,"cache successor eviction counter");
        const auto startup=http(port,"GET","/v1/cache/stats","",auth+"x-test-sessions: 1\r\n");
        const auto session_stats=parse_json(startup.body);
        const auto& session_engine=at(session_stats,"engine");
        const auto& session_state=at(session_engine,"state_cache");
        check(startup.status==200 && at(session_engine,"max_parked").as_int()==3 &&
            at(session_engine,"tombstones").as_int()==0 && at(session_engine,"tombstone_bytes").as_int()==0 &&
            !at(session_engine,"disk_fallback_disabled").as_bool() && at(session_engine,"evicted").as_int()==0,
            "session startup endpoint preserves complete legacy fields");
        check(at(session_state,"published").as_bool() && at(session_state,"sequence").as_int()==1 &&
            at(at(session_state,"disk"),"budget_bytes").as_int()==1234,"session startup publication and disk budget");
        for (const auto& path : {"/v1/completions","/v1/chat/completions","/v1/messages"}) {
            const auto base=std::string(path)=="/v1/completions" ? R"({"prompt":"Hi","max_tokens":100})" :
                R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":100})";
            for (const auto& field : {"false","true","absent"}) {
                auto body=parse_json(base);
                if (std::string(field)!="absent") {
                    Json policy=Json::object(); policy.set("retain",Json(std::string(field)=="true")); body.set("state_cache",std::move(policy));
                }
                const auto r=http(port,"POST",path,body.dump(),auth);
                const auto expected=std::string(field)=="false" ? CacheRetention::no_store :
                    std::string(field)=="true" ? CacheRetention::retain : CacheRetention::server_policy;
                check(r.status==200 && engine.last_retention.load()==expected,"retention propagation "+std::string(path)+" "+field);
            }
            for (const auto& invalid : {"null","false","[]","{}",R"({"retain":0})",R"({"retain":"false"})",R"({"retain":null})"}) {
                auto body=parse_json(base); body.set("state_cache",parse_json(invalid));
                const auto r=http(port,"POST",path,body.dump(),auth);
                check(r.status==400 && r.body.find("invalid_request_error")!=std::string::npos &&
                    r.body.find("boolean retain")!=std::string::npos,"invalid retention shape "+std::string(path)+invalid);
            }
        }
    }
    {
        const auto r = http(port, "GET", "/health", "");
        check(r.status == 200 && at(parse_json(r.body), "status").as_string() == "ok", "health");
        check(http(port, "GET", "/v1/models", "").status == 401, "models needs the API key");
        const auto models = http(port, "GET", "/v1/models", "", auth);
        check(models.status == 200 && at(at(parse_json(models.body), "data").as_array().at(0), "id").as_string() == "test-model", "models");
    }
    {
        const auto r = http(port, "POST", "/v1/chat/completions", R"({"model":"x","messages":[{"role":"user","content":"Hi"}]})", auth);
        check(r.headers.find("X-Strix-Http-Slots: 9") != std::string::npos &&
              r.headers.find("X-Strix-Queue-Position: 0") != std::string::npos, "capacity and submission headers");
        check(r.status == 200, "chat status " + std::to_string(r.status) + " " + r.body);
        const auto j = parse_json(r.body);
        const auto& message = at(first_choice(j), "message");
        check(at(message, "content").as_string() == "Hello there, friend.", "chat content");
        check(at(message, "reasoning_content").as_string() == "Let me think.", "chat reasoning");
        check(at(first_choice(j), "finish_reason").as_string() == "stop", "chat finish stop");
        check(at(at(j, "usage"), "prompt_tokens").as_int() > 0, "chat usage");
    }
    {
        const auto r = http(port, "POST", "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"Hi"}],"stream":true,"stream_options":{"include_usage":true}})", auth);
        std::string reasoning, finish;
        bool usage = false;
        const auto content = stream_text(r.body, &reasoning, &finish, &usage, nullptr);
        check(content == "Hello there, friend." && reasoning == "Let me think.", "chat stream deltas [" + content + "] [" + reasoning + "]");
        check(finish == "stop" && usage && r.body.ends_with("data: [DONE]\n\n"), "chat stream finish, usage and [DONE]");
    }
    {
        const std::string body = R"({"messages":[{"role":"user","content":"TOOL please"}],"tools":[{"type":"function","function":{"name":"get_weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]})";
        const auto j = parse_json(http(port, "POST", "/v1/chat/completions", body, auth).body);
        const auto& choice = first_choice(j);
        const auto& message = at(choice, "message");
        const auto* calls = message.find("tool_calls");
        check(at(choice, "finish_reason").as_string() == "tool_calls" && calls && calls->size() == 1U, "tool call finish " + j.dump());
        if (calls && calls->size() == 1U) {
            const auto args = parse_json(at(at(calls->as_array()[0], "function"), "arguments").as_string());
            check(args.find("days") && at(args, "city").as_string() == "Oslo" && at(args, "days").as_int() == 2, "tool call arguments " + j.dump());
        }
        check(at(message, "content").is_string() && at(message, "content").as_string() == "Checking.", "content before the tool call");
        std::string stream_body = body;
        stream_body.insert(1, R"("stream":true,)");
        bool tools = false;
        const auto streamed = stream_text(http(port, "POST", "/v1/chat/completions", stream_body, auth).body, nullptr, nullptr, nullptr, &tools);
        check(streamed == "Checking." && tools, "streamed tool call [" + streamed + "]");
    }
    {
        const auto j = parse_json(http(port, "POST", "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"Hi"}],"enable_thinking":false})", auth).body);
        check(at(at(first_choice(j), "message"), "content").as_string() == "Plain answer.", "thinking off answer");
        const auto s = parse_json(http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"Hi"}],"stop":"there"})", auth).body);
        check(at(at(first_choice(s), "message"), "content").as_string() == "Hello" && at(first_choice(s), "finish_reason").as_string() == "stop",
            "stop string truncates " + s.dump());
    }
    {
        const auto j = parse_json(http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"LONG"}],"max_tokens":20})", auth).body);
        check(at(first_choice(j), "finish_reason").as_string() == "length" && at(at(j, "usage"), "completion_tokens").as_int() == 20,
            "max_tokens finish length " + j.dump());
    }
    {
        const auto j = parse_json(http(port, "POST", "/v1/completions", R"({"prompt":"Hello","max_tokens":50})", auth).body);
        check(at(j, "object").as_string() == "text_completion" &&
            at(first_choice(j), "text").as_string().find("Hello there") != std::string::npos, "completions " + j.dump());
        const auto ids = parse_json(http(port, "POST", "/v1/completions", R"({"prompt":[72,101],"max_tokens":5,"return_token_ids":true})", auth).body);
        check(at(at(ids, "usage"), "prompt_tokens").as_int() == 2, "completions with token IDs " + ids.dump());
        check(at(at(ids, "strix"), "token_ids").size() == 5U, "returned token IDs " + ids.dump());
    }
    {
        const std::string body = R"({"model":"x","max_tokens":100,"system":"Be brief.","thinking":{"type":"enabled","budget_tokens":100},
            "messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]})";
        const auto j = parse_json(http(port, "POST", "/v1/messages", body, "x-api-key: secret\r\n").body);
        const auto& content = at(j, "content").as_array();
        check(at(j, "type").as_string() == "message" && content.size() == 2U && at(content[0], "type").as_string() == "thinking" &&
            at(content[1], "text").as_string() == "Hello there, friend." && at(j, "stop_reason").as_string() == "end_turn", "anthropic message " + j.dump());
        std::string stream_body = body;
        stream_body.insert(1, R"("stream":true,)");
        const auto events = http(port, "POST", "/v1/messages", stream_body, "x-api-key: secret\r\n").body;
        check(events.find("event: message_start") == 0 && events.find("event: content_block_start") != std::string::npos &&
            events.find("thinking_delta") != std::string::npos && events.find("text_delta") != std::string::npos &&
            events.rfind("event: message_stop") != std::string::npos, "anthropic event stream");
        const auto tool = parse_json(http(port, "POST", "/v1/messages", R"({"max_tokens":1000,"messages":[{"role":"user","content":"TOOL"}],
            "tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"days":{"type":"integer"}}}}]})", "x-api-key: secret\r\n").body);
        bool used = false;
        for (const auto& block : at(tool, "content").as_array())
            if (at(block, "type").as_string() == "tool_use" && at(at(block, "input"), "days").as_int() == 2) used = true;
        check(used && at(tool, "stop_reason").as_string() == "tool_use", "anthropic tool_use " + tool.dump());
    }
    {
        const std::string request = R"({"prompt":[1,2],"max_tokens":1,"session":"session_A"})";
        // Default-off ignores every session value, including invalid names/types, as before.
        // Compare serialized responses after removing only volatile IDs/timestamps/timings.
        const auto stable = [](const std::string& text) {
            auto response = parse_json(text);
            response.set("id", Json("volatile")); response.set("created", Json(0));
            auto extra = *response.find("strix");
            for (const auto* key : {"queue_ms", "prefill_ms", "decode_ms", "decode_tokens_per_second"})
                extra.set(key, Json(0));
            // Main's host selection profile includes wall times; retain its deterministic counters.
            if (const auto* verdict = extra.find("constraint")) {
                auto constraint = *verdict;
                if (const auto* measured = constraint.find("host_profile")) {
                    auto profile = *measured;
                    for (const auto* key : {"mask_ms", "select_ms", "window_row_ms"}) profile.set(key, Json(0));
                    constraint.set("host_profile", std::move(profile));
                }
                extra.set("constraint", std::move(constraint));
            }
            response.set("strix", std::move(extra));
            return response.dump();
        };
        const auto baseline = http(port, "POST", "/v1/completions", R"({"prompt":[1,2],"max_tokens":1})", auth);
        for (const auto* value : {R"("session_A")", R"("../escape")", "17", "{}", "[]", "null"}) {
            const auto ignored = http(port, "POST", "/v1/completions",
                std::string(R"({"prompt":[1,2],"max_tokens":1,"session":)") + value + "}", auth);
            check(ignored.status == baseline.status && stable(ignored.body) == stable(baseline.body),
                "disabled session field changed response bytes");
        }
        // Unnamed streams keep the public-main wire: their code path is unchanged, and
        // enabling session support must not change a single event (ids/timestamps aside).
        // Event lines and [DONE] verbatim; each JSON payload without its per-request ids,
        // timestamps and timing receipts, at any depth (message_start nests its message id).
        const std::function<void(Json&)> strip_volatile = [&](Json& value) {
            if (value.is_object()) {
                std::erase_if(value.as_object(), [](const Json::Member& m) {
                    return m.first == "id" || m.first == "created" || m.first == "strix";
                });
                for (auto& member : value.as_object()) strip_volatile(member.second);
            } else if (value.is_array()) for (auto& item : value.as_array()) strip_volatile(item);
        };
        const auto stable_wire = [&](const std::string& wire) {
            std::string out;
            for (std::size_t at = 0; at < wire.size();) {
                auto end = wire.find('\n', at);
                if (end == std::string::npos) end = wire.size();
                const auto line = wire.substr(at, end - at);
                at = end + 1U;
                if (line.starts_with("data: ") && line != "data: [DONE]") {
                    auto event = parse_json(line.substr(6U));
                    strip_volatile(event);
                    out += "data: " + event.dump() + "\n";
                } else out += line + "\n";
            }
            return out;
        };
        for (const auto* path : {"/v1/completions", "/v1/chat/completions", "/v1/messages"}) {
            const bool messages = std::string_view(path) == "/v1/messages";
            const bool chat = std::string_view(path) == "/v1/chat/completions";
            for (const bool failure : {false,true}) {
            std::string disabled_wire;
            for (const bool enabled : {false,true}) {
                const auto prompt = failure ? "ENGINE_ERROR" : "Hello";
                auto body = parse_json(std::string_view(path) == "/v1/completions"
                    ? R"({"prompt":"","stream":true,"max_tokens":16})"
                    : R"({"messages":[{"role":"user","content":""}],"stream":true,"max_tokens":16})");
                if (!messages && !chat) body.set("prompt",Json(prompt));
                else {
                    auto messages_body = *body.find("messages");
                    messages_body.as_array()[0].set("content",Json(prompt));
                    body.set("messages",std::move(messages_body));
                }
                const auto wire = http(port,"POST",path,body.dump(),auth + (enabled ? "X-Test-Sessions: 1\r\n" : ""));
                check(wire.status == 200, "unnamed stream status");
                if (!failure)
                    check(stable_wire(wire.body).ends_with(messages ? "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
                                                                    : "data: [DONE]\n\n"), "unnamed stream success terminal");
                if (!enabled) disabled_wire = stable_wire(wire.body);
                else check(stable_wire(wire.body) == disabled_wire, "session support changes an unnamed stream");
                int errors = 0;
                for (const auto& event : sse_data(wire.body)) if (const auto* error = event.find("error")) {
                    ++errors;
                    check(error->dump() == R"({"message":"scripted failure","type":"server_error"})" &&
                        !event.find("strix"), "unnamed chat error retains exact legacy payload");
                }
                check(errors == (failure && chat ? 1 : 0), "unnamed endpoint preserves legacy error event count");
            }
            }
        }
        const auto session_auth = auth + "X-Test-Sessions: 1\r\n";
        const auto budget=http(port,"POST","/v1/completions",
            R"({"prompt":[1,2],"max_tokens":1,"session":"budget_refusal"})",session_auth);
        check(budget.status==507 && at(at(parse_json(budget.body),"error"),"code").as_string()=="session_budget",
            "session budget refusal is a structured 507");
        for (const auto* path : {"/v1/completions", "/v1/chat/completions", "/v1/messages"}) {
            const bool messages = std::string_view(path) == "/v1/messages";
            auto request_body = parse_json(std::string_view(path) == "/v1/completions"
                ? R"({"prompt":[1,2],"max_tokens":16})"
                : R"({"messages":[{"role":"user","content":"Hello"}],"max_tokens":16})");
            request_body.set("session", Json("budget_refusal"));
            const auto plain = http(port, "POST", path, request_body.dump(), session_auth);
            request_body.set("stream", Json(true));
            const auto early = http(port, "POST", path, request_body.dump(), session_auth);
            check(plain.status == 507 && early.status == 507 && early.body == plain.body &&
                early.headers.find("text/event-stream") == std::string::npos,
                std::string(path) + " streaming preflight returns identical structured 507");
            request_body.set("session", Json("session_A"));
            check(http(port, "POST", path, request_body.dump(), session_auth).status == 200,
                std::string(path) + " serves after preflight refusal");
            request_body.set("session", Json("late_budget_refusal"));
            engine.budget_headers.store(false);
            const auto late = http(port, "POST", path, request_body.dump(), session_auth, 0, &engine.budget_headers);
            int errors = 0;
            for (const auto& event : sse_data(late.body)) {
                if (const auto* error = event.find("error")) {
                    ++errors;
                    check(at(*error, "status").as_int() == 507 && at(*error, "code").as_string() == "session_budget" &&
                        at(*error, "type").as_string() == (messages ? "api_error" : "server_error"),
                        std::string(path) + " late refusal retains status/code/type");
                    if (messages) check(at(event, "type").as_string() == "error", "messages error event type");
                }
                if (const auto* choices = event.find("choices")) for (const auto& choice : choices->as_array())
                    check(at(choice, "finish_reason").is_null(), "no success finish reason after refusal");
            }
            check(late.status == 200 && errors == 1 && late.body.find("[DONE]") == std::string::npos &&
                late.body.find("event: message_stop") == std::string::npos &&
                late.body.find("event: message_delta") == std::string::npos &&
                (!messages || late.body.find("event: error\n") != std::string::npos),
                std::string(path) + " exactly one SSE error and no success terminal");
            request_body.set("session", Json("session_A"));
            check(http(port, "POST", path, request_body.dump(), session_auth).status == 200,
                std::string(path) + " serves after late refusal");
        }
        const auto unnamed_on = http(port, "POST", "/v1/completions", R"({"prompt":[1,2],"max_tokens":1})", session_auth);
        check(unnamed_on.status == baseline.status && stable(unnamed_on.body) == stable(baseline.body),
            "defaults ON unnamed request differs from session cache OFF");
        for (const auto* field : {"ngram_draft", "tool_draft", "ngram_peers", "draft_depth_max"}) {
            const auto body = std::string(R"({"prompt":[1,2],"max_tokens":1,")") + field +
                (std::string_view(field)=="draft_depth_max" ? "\":0}" : "\":true}");
            const auto off = http(port, "POST", "/v1/completions", body, auth);
            const auto on = http(port, "POST", "/v1/completions", body, session_auth);
            check(on.status == 200 && off.status == on.status && stable(off.body) == stable(on.body),
                "unnamed draft override changes when sessions enabled");
        }
        const auto named = http(port, "POST", "/v1/completions", request, session_auth);
        check(named.status == 200 && at(at(parse_json(named.body), "strix"), "session_echo").as_string() == "session_A",
            "named session reaches engine job");
        check(http(port, "POST", "/v1/completions", R"({"prompt":[1,2],"session":"../escape"})", session_auth).status == 400,
            "invalid session path rejected before queue admission");
        check(http(port, "POST", "/v1/completions", R"({"prompt":[1,2],"session":17})", session_auth).status == 400,
            "nonstring session rejected");
        check(http(port, "POST", "/v1/completions", R"({"prompt":[1,2],"session":"A","ngram_draft":true})", session_auth).status == 400,
            "session draft override rejected before queue admission");
        const auto constraint_auth = auth + "X-Test-Config: constraints\r\n";
        const std::string constrained_body = R"({"messages":[{"role":"user","content":"object"}],"max_tokens":32,"response_format":{"type":"json_object"}})";
        const auto ordinary_constraint = http(port, "POST", "/v1/chat/completions", constrained_body, constraint_auth);
        const auto unnamed_constraint = http(port, "POST", "/v1/chat/completions", constrained_body,
            constraint_auth + "X-Test-Sessions: 1\r\n");
        check(ordinary_constraint.status == 200 && unnamed_constraint.status == 200 &&
            stable(ordinary_constraint.body) == stable(unnamed_constraint.body),
            "unnamed constrained output changes when sessions enabled");
        auto named_constraint_body = parse_json(constrained_body); named_constraint_body.set("session", Json("A"));
        const auto named_constraint = http(port, "POST", "/v1/chat/completions", named_constraint_body.dump(),
            constraint_auth + "X-Test-Sessions: 1\r\n");
        check(named_constraint.status == 400 && at(at(parse_json(named_constraint.body), "error"), "param").as_string() == "session",
            "named constrained output rejected before queue admission");
    }
    {
        const auto bad = http(port, "POST", "/v1/chat/completions", "{nope", auth);
        check(bad.status == 400 && at(at(parse_json(bad.body), "error"), "type").as_string() == "invalid_request_error", "bad JSON");
        check(http(port, "POST", "/v1/nowhere", "{}", auth).status == 404, "unknown route");
        const auto anthropic = http(port, "POST", "/v1/messages", R"({"messages":[]})", auth);
        check(anthropic.status == 400 && at(parse_json(anthropic.body), "type").as_string() == "error", "anthropic error format");
    }
    {
        // Two concurrent clients: both complete with their own outputs (FIFO on one engine).
        Reply a, b;
        std::thread ta([&] { a = http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"LONG"}],"max_tokens":60})", auth); });
        std::thread tb([&] { b = http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"Hi"}],"stream":true})", auth); });
        ta.join(); tb.join();
        check(a.status == 200 && at(at(parse_json(a.body), "usage"), "completion_tokens").as_int() == 60, "concurrent client A");
        check(b.headers.find("X-Strix-Http-Slots: 9") != std::string::npos &&
              b.headers.find("X-Strix-Queue-Position:") != std::string::npos, "stream submission headers");
        check(b.status == 200 && stream_text(b.body, nullptr, nullptr, nullptr, nullptr) == "Hello there, friend.", "concurrent client B");
    }
    {
        const auto before = engine.cancelled.load();
        (void)http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"LONG"}],"stream":true,"max_tokens":400})", auth, 300);
        for (int i = 0; i < 200 && engine.cancelled.load() == before; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        check(engine.cancelled.load() > before, "client disconnect cancels the generation");
        const auto status = parse_json(http(port, "GET", "/api/status", "").body);
        check(at(status, "schema").as_string() == "strix-serve-status-v1" && at(status, "requests").size() > 0U, "status");
        const auto metrics = http(port, "GET", "/metrics", "").body;
        check(metrics.find("strix_generated_tokens_total") != std::string::npos, "metrics");
    }
    } catch (const std::exception& error) {
        check(false, std::string("exception: ") + error.what());
    }
    server.stop();
    queue.shutdown();
    http_thread.join();
    engine_thread.join();
    std::cout << "serve API tests: failures=" << failures << '\n';
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "serve API tests: " << error.what() << '\n';
    return 1;
}
