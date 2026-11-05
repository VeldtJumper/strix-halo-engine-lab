#include "api.hpp"
#include "gemma4/features/features.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <list>
#include <map>
#include <mutex>
#include <thread>

namespace strix::gemma4::engine {
namespace {
using Json = strix::serve::Json;
using strix::serve::HttpRequest;
using strix::serve::HttpResponseWriter;
const Json& get(const Json& j, const char* name) {
    static const Json empty;
    const auto* p = j.find(name); return p ? *p : empty;
}
std::uint64_t integer(const Json& j, const char* field, std::uint64_t maximum = UINT32_MAX) {
    const auto n = j.as_int();
    if (n < 0 || std::uint64_t(n) > maximum) throw std::invalid_argument(std::string("invalid ")+field);
    return std::uint64_t(n);
}
Json wire_calls(const Json& calls, bool streaming) {
    auto out = Json::array();
    for (std::size_t i = 0; i < calls.size(); ++i) {
        const auto& c = calls.as_array()[i];
        auto fn = get(c,"function");
        fn = Json(fn.as_object());
        fn.set("arguments",get(fn,"arguments").dump());
        Json call(Json::Object{{"id",get(c,"id")},{"type","function"},{"function",fn}});
        if (streaming) call.set("index",std::uint64_t(i));
        out.push(call);
    }
    return out;
}
Json output_message(const gemma4::serve::Output& output) {
    Json message(Json::Object{{"role","assistant"},{"content",output.content}});
    if (!output.reasoning.empty()) message.set("reasoning_content",output.reasoning);
    if (output.tool_calls.size()) message.set("tool_calls",wire_calls(output.tool_calls,false));
    return message;
}
std::size_t env_count(const char* name, std::size_t fallback, std::size_t maximum) {
    const auto* p = std::getenv(name);
    if (!p) return fallback;
    const auto n = std::strtoull(p,nullptr,10);
    return n > maximum ? maximum : std::size_t(n);
}
bool serve_batched() { const auto* p = std::getenv("STRIX_G4_SERVE_BATCH"); return !p || std::string(p) != "0"; }
}
std::size_t serve_slots() { return serve_batched() ? std::max<std::size_t>(1,env_count("STRIX_G4_SERVE_SLOTS",8,8)) : 1; }
ChatRequest parse_chat_request(const Json& j, DraftMode default_mode, const Request& defaults) {
    if (!j.is_object()) throw std::invalid_argument("chat request must be an object");
    for (const auto& [key,value] : j.as_object()) {
        (void)value;
        bool known = false;
        for (const auto* name : {"model","messages","tools","tool_choice","stream","stream_options","max_tokens",
                "max_completion_tokens","temperature","top_p","top_k","seed","enable_thinking","preserve_thinking",
                "conversational_mode","thinking_budget","chat_template_kwargs","dflash","n"}) known |= key == name;
        if (!known) throw std::invalid_argument("unsupported field: "+key);
    }
    ChatRequest r;
    r.generation = defaults;
    r.generation.dflash = default_mode;
    auto messages = Json(get(j,"messages").as_array());
    // Deep-copy modified message/function objects; Json containers share storage.
    for (auto& m : messages.as_array()) {
        m = Json(m.as_object());
        if (const auto* calls = m.find("tool_calls")) {
            auto converted = Json(calls->as_array());
            for (auto& call : converted.as_array()) {
                call = Json(call.as_object());
                auto fn = Json(get(call,"function").as_object());
                if (get(fn,"arguments").is_string()) fn.set("arguments",strix::serve::parse_json_unique(get(fn,"arguments").as_string()));
                call.set("function",fn);
            }
            m.set("tool_calls",converted);
        }
    }
    Json kwargs = get(j,"chat_template_kwargs").is_null() ? Json::object() : Json(get(j,"chat_template_kwargs").as_object());
    for (const auto* name : {"enable_thinking","preserve_thinking","conversational_mode"}) if (j.find(name)) {
        if (kwargs.find(name)) throw std::invalid_argument(std::string("duplicate template option: ")+name);
        (void)get(j,name).as_bool(); kwargs.set(name,get(j,name));
    }
    r.input = gemma4::serve::chat_input(messages,kwargs);
    if (j.find("tools")) r.input.tools = get(j,"tools");
    if (j.find("tool_choice")) {
        const auto& choice = get(j,"tool_choice").as_string();
        if (choice == "none") r.input.tools = Json();
        else if (choice != "auto") throw std::invalid_argument("unsupported tool_choice");
    }
    if (j.find("n") && integer(get(j,"n"),"n") != 1) throw std::invalid_argument("n must be 1");
    if (j.find("stream")) (void)get(j,"stream").as_bool();
    if (j.find("stream_options")) {
        for (const auto& [key,value] : get(j,"stream_options").as_object())
            if (key != "include_usage" || !value.is_bool()) throw std::invalid_argument("unsupported stream_options");
    }
    if (j.find("max_tokens") && j.find("max_completion_tokens")) throw std::invalid_argument("choose one max token field");
    for (const auto* name : {"max_tokens","max_completion_tokens"}) if (j.find(name)) r.generation.max_tokens = integer(get(j,name),name);
    if (!r.generation.max_tokens) throw std::invalid_argument("max_tokens must be positive");
    if (j.find("temperature")) r.generation.sampling.temperature = float(get(j,"temperature").as_double());
    if (j.find("top_p")) r.generation.sampling.top_p = float(get(j,"top_p").as_double());
    if (j.find("top_k")) r.generation.sampling.top_k = std::uint32_t(integer(get(j,"top_k"),"top_k"));
    if (j.find("seed")) r.generation.sampling.seed = integer(get(j,"seed"),"seed",INT64_MAX);
    r.generation.sampling.validate();
    if (j.find("thinking_budget")) r.thinking_budget = std::uint32_t(integer(get(j,"thinking_budget"),"thinking_budget"));
    if (r.thinking_budget && (!r.input.enable_thinking || r.thinking_budget >= r.generation.max_tokens))
        throw std::invalid_argument("thinking_budget requires thinking and must be below max_tokens");
    if (j.find("dflash")) {
        const auto& mode = get(j,"dflash").as_string();
        if (mode != "on" && mode != "off" && mode != "auto") throw std::invalid_argument("dflash must be on/off/auto");
        r.generation.dflash = mode == "on" ? DraftMode::on : mode == "off" ? DraftMode::off : DraftMode::automatic;
    }
    (void)gemma4::serve::render_chat(r.input);
    return r;
}
struct Api::Impl final : strix::serve::GenerationEngine {
    struct Task {
        ChatRequest request;
        ChatResult result;
        ChatSink sink;
        std::atomic<bool> cancelled{false};
        // Continuous batching: deltas the batcher produced, written to the client by the handler thread.
        std::mutex deltas_mutex;
        std::vector<gemma4::serve::OutputDelta> deltas;
    };
    Backend& backend;
    gemma4::serve::Adapter adapter;
    DraftFactory drafts;
    DraftMode mode;
    Request defaults;
    std::string model;
    strix::serve::GenerationQueue queue;
    std::shared_ptr<CostModel> costs = std::make_shared<CostModel>();
    std::mutex mutex;
    std::mutex backend_mutex;
    std::unique_ptr<features::Exporter> feature_exporter;
    std::map<std::string,std::shared_ptr<Task>> tasks;
    std::atomic<std::uint64_t> sequence{0};
    // STRIX_G4_SERVE_BATCH=0: one chat job at a time (queue.run + run_chat). Otherwise up to STRIX_G4_SERVE_SLOTS
    // (default 8) jobs share one Coordinator and decode together; STRIX_G4_SERVE_PREFILL_PER_STEP (default 1, 0 = all)
    // bounds the prompt chunks one step may prefill. Each job's ids equal the job run alone.
    const bool batched = serve_batched();
    const std::size_t slots = serve_slots();
    const std::size_t prefill_per_step = env_count("STRIX_G4_SERVE_PREFILL_PER_STEP",1,8);
    // Prompt rows per prefill chunk while other requests decode (0 = whole 512-row chunks).
    const std::size_t prefill_rows = env_count("STRIX_G4_SERVE_PREFILL_ROWS",64,512);
    // Prompt prefix cache (STRIX_G4_SERVE_PREFIX=0 off): committed states at the end of each prompt and of each
    // finished request, reused by a later prompt that starts with exactly those tokens (a conversation's next turn).
    // LRU within STRIX_G4_SERVE_PREFIX_MB (default 4096) of KV and 16 entries.
    const bool prefix_cache = batched && [] { const auto* p = std::getenv("STRIX_G4_SERVE_PREFIX"); return !p || std::string(p) != "0"; }();
    const std::size_t prefix_bytes = env_count("STRIX_G4_SERVE_PREFIX_MB",4096,1U << 20U) << 20U;
    std::thread worker;
    Impl(Backend& b, std::shared_ptr<const gemma4::serve::Tokenizer> t, DraftFactory d, DraftMode m, std::string name, Request settings)
        : backend(b), adapter(std::move(t)), drafts(std::move(d)), mode(m), defaults(std::move(settings)), model(std::move(name)),
          worker([this] { if (batched) batch_loop(); else queue.run(*this); }) {}
    ~Impl() override {
        { std::lock_guard lock(mutex); for (auto& [id,t] : tasks) { (void)id; t->cancelled = true; } }
        queue.shutdown(); worker.join();
    }
    std::uint64_t context_tokens() const override { return backend.config().max_positions; }
    bool supports_sampling() const override { return true; }
    bool supports_thinking_budget() const override { return true; }
    strix::serve::GenerationStats generate(strix::serve::GenerationJob& job) override {
        std::shared_ptr<Task> task;
        { std::lock_guard lock(mutex); task = tasks.at(job.session_id); }
        {
            std::lock_guard lock(backend_mutex);
            task->result = run_chat(backend,adapter,task->request,drafts,task->sink,[task] { return task->cancelled.load(); },costs);
        }
        strix::serve::GenerationStats stats;
        stats.prompt_tokens = task->result.prompt_tokens; stats.generated_tokens = task->result.ids.size(); stats.finish = task->result.finish;
        if (job.on_tokens) (void)job.on_tokens(task->result.ids);
        return stats;
    }
    void batch_loop() {
        struct Live {
            strix::serve::TakenJob taken;
            std::shared_ptr<Task> task;
            std::unique_ptr<ChatJob> chat;
            std::size_t slot = 0, prompt = 0;
            // Per phase, for the receipt: steps while live, summed and widest batch, steps that also prefilled
            // another request's prompt chunk.
            std::uint64_t admitted = 0, steps = 0, width_sum = 0, width_max = 0, shared_prefill = 0;
            bool prompt_kept = false;
        };
        Coordinator coordinator(backend,drafts,costs);
        coordinator.set_prefill_limit(prefill_per_step);
        coordinator.set_prefill_budget(prefill_rows);
        struct Kept { std::shared_ptr<const PrefixSnapshot> snapshot; std::size_t bytes = 0; };
        std::list<Kept> kept; // most recently used first
        std::size_t kept_bytes = 0;
        std::uint64_t prefix_hits = 0, prefix_misses = 0;
        // Device KV bytes of a state with `positions` committed rows: global slots grow with it, sliding rings do not.
        const auto state_bytes = [&](std::size_t positions) {
            const auto& c = backend.config();
            std::size_t bytes = 0;
            for (std::uint32_t l = 0; l < c.layer.size(); ++l)
                if (c.layer[l].producer == l) {
                    const std::size_t width = c.kv_heads * c.layer[l].head_dim;
                    const std::size_t rows = c.layer[l].attention == AttentionKind::global ? positions : c.window;
                    bytes += rows * (2 * width * sizeof(float) + sizeof(std::uint64_t));
                }
            return bytes;
        };
        // When prefill rows differ from decode rows, only states the prefill alone wrote, at a chunk boundary, are
        // reusable: a request without the cache writes exactly those rows the same way (decode-written rows - the
        // answer, a short prompt tail - would be prefilled instead).
        const bool decode_states = backend.prefill_matches_decode();
        const std::size_t chunk = backend.prefill_chunk_rows();
        const auto reusable = [&](const PrefixSnapshot& p) {
            return decode_states || (p.prefill_written && chunk && p.tokens.size() % chunk == 0);
        };
        const auto keep = [&](std::size_t slot) {
            std::shared_ptr<PrefixSnapshot> snapshot;
            {
                std::lock_guard lock(backend_mutex);
                snapshot = coordinator.snapshot(slot);
            }
            if (!snapshot || !reusable(*snapshot)) return;
            for (auto it = kept.begin(); it != kept.end(); ++it)
                if (it->snapshot->tokens == snapshot->tokens) { kept_bytes -= it->bytes; kept.erase(it); break; }
            const auto bytes = state_bytes(snapshot->tokens.size());
            kept.push_front({std::move(snapshot),bytes});
            kept_bytes += bytes;
            while (kept.size() > 16 || (kept.size() > 1 && kept_bytes > prefix_bytes)) {
                kept_bytes -= kept.back().bytes;
                kept.pop_back();
            }
        };
        // The longest kept state whose tokens start the prompt, leaving at least its last token to decode from.
        const auto lookup = [&](const std::vector<std::uint32_t>& prompt) -> std::shared_ptr<const PrefixSnapshot> {
            auto best = kept.end();
            for (auto it = kept.begin(); it != kept.end(); ++it) {
                const auto& t = it->snapshot->tokens;
                if (reusable(*it->snapshot) && t.size() < prompt.size() && std::equal(t.begin(),t.end(),prompt.begin()) &&
                    (best == kept.end() || t.size() > best->snapshot->tokens.size()))
                    best = it;
            }
            if (best == kept.end()) { ++prefix_misses; return nullptr; }
            ++prefix_hits;
            kept.splice(kept.begin(),kept,best);
            return kept.front().snapshot;
        };
        std::uint64_t steps = 0;
        std::vector<Live> live;
        const auto finish = [&](Live& l, const std::string& error) {
            strix::serve::GenerationStats stats;
            if (error.empty()) {
                l.task->result = l.chat->take();
                stats.prompt_tokens = l.task->result.prompt_tokens; stats.generated_tokens = l.task->result.ids.size();
                stats.finish = l.task->result.finish;
                for (const auto& phase : l.task->result.receipts)
                    if (const auto* cached = phase.find("serve_prefix")) stats.cached_tokens += cached->find("cached_tokens")->as_int();
                if (l.taken.job.on_tokens) (void)l.taken.job.on_tokens(l.task->result.ids);
            }
            queue.finish(l.taken,stats,error);
        };
        const auto receipt = [&](const Live& l) {
            auto r = coordinator.receipt(l.slot);
            r.set("serve_batch",Json(Json::Object{{"mode","continuous"},{"slots",std::uint64_t(slots)},
                {"prefill_per_step",std::uint64_t(prefill_per_step)},{"prefill_rows",std::uint64_t(prefill_rows)},{"slot",std::uint64_t(l.slot)},
                {"admitted_at_step",l.admitted},{"steps",l.steps},{"max_batch",l.width_max},
                {"mean_batch",l.steps ? double(l.width_sum)/double(l.steps) : 0.0},
                {"steps_sharing_prefill",l.shared_prefill}}));
            if (prefix_cache)
                r.set("serve_prefix",Json(Json::Object{{"cached_tokens",coordinator.result(l.slot).cached_tokens},
                    {"entries",std::uint64_t(kept.size())},{"bytes",std::uint64_t(kept_bytes)},
                    {"hits",prefix_hits},{"misses",prefix_misses}}));
            return r;
        };
        const auto submit = [&](Live& l) {
            auto job = l.chat->phase_request();
            l.prompt = job.prompt.size(); l.admitted = steps;
            l.steps = l.width_sum = l.width_max = l.shared_prefill = 0;
            l.prompt_kept = false;
            auto prefix = prefix_cache ? lookup(job.prompt) : nullptr;
            std::lock_guard lock(backend_mutex);
            l.slot = coordinator.submit(std::move(job),std::move(prefix));
        };
        const auto prefilling = [&](const Live& l) { return coordinator.result(l.slot).frontier+1 < l.prompt; };
        for (;;) {
            while (live.size() < slots) {
                auto taken = queue.take(live.empty());
                if (!taken) break;
                Live l{std::move(*taken),nullptr,nullptr};
                try {
                    { std::lock_guard lock(mutex); l.task = tasks.at(l.taken.job.session_id); }
                    l.chat = std::make_unique<ChatJob>(adapter,l.task->request,l.task->sink);
                    submit(l);
                } catch (const std::exception& e) {
                    queue.finish(l.taken,{},e.what());
                    continue;
                }
                live.push_back(std::move(l));
            }
            if (live.empty()) {
                if (queue.stopping()) return;
                continue;
            }
            std::size_t prefills = 0;
            for (auto& l : live) {
                if (l.task->cancelled || l.chat->disconnected()) coordinator.cancel(l.slot);
                prefills += prefilling(l) ? 1 : 0;
            }
            for (auto& l : live) {
                ++l.steps; l.width_sum += live.size(); l.width_max = std::max<std::uint64_t>(l.width_max,live.size());
                l.shared_prefill += prefills > (prefilling(l) ? 1U : 0U) ? 1 : 0;
            }
            ++steps;
            {
                std::lock_guard lock(backend_mutex);
                coordinator.step();
            }
            for (std::size_t i = 0; i < live.size();) {
                auto& l = live[i];
                if (!l.chat->absorb(coordinator.result(l.slot))) coordinator.cancel(l.slot);
                const auto& result = coordinator.result(l.slot);
                // End of the prompt: the conversation so far, which the next turn renders again as its prefix. When
                // prefill rows differ from decode rows: the last prefill-written chunk boundary before the prompt's end.
                if (prefix_cache && !l.prompt_kept && result.status != Status::failed && result.frontier > result.cached_tokens &&
                    (decode_states ? result.frontier+1 >= l.prompt :
                        chunk && result.prefill_frontier == result.frontier && result.frontier % chunk == 0 &&
                        l.prompt-1-result.frontier < chunk)) {
                    l.prompt_kept = true;
                    keep(l.slot);
                }
                if (result.status == Status::running) { ++i; continue; }
                // End of a request: prompt and answer, for a prompt that continues the raw output (decode-written rows).
                if (prefix_cache && decode_states && (result.status == Status::stopped || result.status == Status::length))
                    keep(l.slot);
                std::string error;
                try {
                    if (l.chat->end_phase(coordinator.result(l.slot),receipt(l))) {
                        submit(l);
                        ++i;
                        continue;
                    }
                } catch (const std::exception& e) {
                    error = e.what();
                }
                finish(l,error);
                live.erase(live.begin()+std::ptrdiff_t(i));
            }
        }
    }
    void handle(const HttpRequest& req, HttpResponseWriter& writer) {
        bool streaming = false;
        int failure_status = 400;
        std::string task_id;
        try {
            if (req.method == "GET" && req.path == "/health") { writer.send(200,"application/json","{\"status\":\"ok\"}"); return; }
            if (req.method == "GET" && req.path == "/v1/models") {
                writer.send(200,"application/json",Json(Json::Object{{"object","list"},{"data",Json(Json::Array{
                    Json(Json::Object{{"id",model},{"object","model"},{"owned_by","local"}})})}}).dump()); return;
            }
            if (req.method == "GET" && req.path == "/v1/capabilities") {
                const auto config = effective_config(backend,defaults,bool(drafts));
                writer.send(200,"application/json",Json(Json::Object{{"model",model},{"identity",*config.find("decode_identity")},
                    {"prefill",*config.find("prefill_identity")},{"sampling","splitmix64-cdf-v1"},{"thinking_budget","checked-two-job-continuation"},
                    {"tools",true},{"reasoning_content",true},{"queue",batched ? "continuous batching; up to "+std::to_string(slots)+" chat jobs decode together" :
                        std::string("FIFO; one chat job at a time")},
                    {"features",backend.identity().find("feature_identity") ? *backend.identity().find("feature_identity") : Json()},
                    {"dflash",config},
                    {"unsupported",Json(Json::Array{"response_format","logprobs","stop","penalties","forced tool_choice"})}}).dump()); return;
            }
            if (req.method == "POST" && req.path == "/v1/gemma4/features") {
                const auto request = features::parse_request(strix::serve::parse_json_unique(req.body),&adapter);
                features::Result result;
                try {
                    std::lock_guard lock(backend_mutex);
                    if (!feature_exporter) feature_exporter = std::make_unique<features::Exporter>(backend,features::backend_identity(backend));
                    result = feature_exporter->export_features(request);
                } catch (const std::invalid_argument&) {
                    throw;
                } catch (...) {
                    failure_status = 500; throw;
                }
                writer.send(200,"application/json",result.json().dump()); return;
            }
            if (req.method != "POST" || req.path != "/v1/chat/completions") {
                writer.send(404,"application/json","{\"error\":{\"message\":\"unknown endpoint\"}}"); return;
            }
            const auto body = strix::serve::parse_json_unique(req.body);
            if (body.find("model") && get(body,"model").as_string() != model) throw std::invalid_argument("unknown model");
            auto task = std::make_shared<Task>(); task->request = parse_chat_request(body,mode,defaults);
            const auto prepared = adapter.prepare(task->request.input);
            if (prepared.ids.size()+task->request.generation.max_tokens > backend.config().max_positions)
                throw std::invalid_argument("request exceeds context capacity");
            if (task->request.generation.dflash == DraftMode::on && !drafts && dflash::enabled_from_environment() &&
                task->request.generation.sampling.temperature == 0)
                throw std::invalid_argument("DFlash on requires a drafter");
            task_id = "chatcmpl-g4-"+std::to_string(++sequence);
            const auto created = std::uint64_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
            const auto envelope = [&](Json choice, const char* object) {
                return Json(Json::Object{{"id",task_id},{"object",object},{"created",created},{"model",model},{"choices",Json(Json::Array{choice})}});
            };
            streaming = body.find("stream") && get(body,"stream").as_bool();
            bool first = true;
            std::size_t tool_index = 0;
            ChatSink emit;
            if (streaming) {
                if (!writer.begin_stream(200,"text/event-stream",{{"Cache-Control","no-cache"}})) return;
                emit = [&](const gemma4::serve::OutputDelta& delta) {
                    if (writer.client_gone()) return false;
                    Json d = Json::object();
                    if (first) { d.set("role","assistant"); first = false; }
                    if (!delta.content.empty()) d.set("content",delta.content);
                    if (!delta.reasoning.empty()) d.set("reasoning_content",delta.reasoning);
                    if (delta.tool_calls.size()) {
                        auto calls = wire_calls(delta.tool_calls,true);
                        for (auto& call : calls.as_array()) { call.set("index",std::uint64_t(tool_index++)); }
                        d.set("tool_calls",calls);
                    }
                    if (!d.size()) return true;
                    return writer.write("data: "+envelope(Json(Json::Object{{"index",0},{"delta",d},{"finish_reason",Json()}}),"chat.completion.chunk").dump()+"\n\n");
                };
                // Batched: the batcher only buffers; this thread writes, so a slow client never stalls other jobs.
                task->sink = batched ? ChatSink([task = task.get()](const gemma4::serve::OutputDelta& delta) {
                    std::lock_guard lock(task->deltas_mutex);
                    task->deltas.push_back(delta);
                    return !task->cancelled.load();
                }) : emit;
            }
            const auto drain = [&] {
                std::vector<gemma4::serve::OutputDelta> deltas;
                { std::lock_guard lock(task->deltas_mutex); deltas.swap(task->deltas); }
                for (const auto& delta : deltas)
                    if (!task->cancelled && emit && !emit(delta)) task->cancelled = true;
            };
            { std::lock_guard lock(mutex); tasks.emplace(task_id,task); }
            auto job = prepared.job(std::uint32_t(task->request.generation.max_tokens)); job.session_id = task_id;
            const auto ticket = queue.submit(std::move(job));
            if (!ticket) { failure_status = 503; throw std::runtime_error("generation queue is full or shutting down"); }
            failure_status = 500;
            bool done = false;
            for (;;) {
                (void)ticket->wait(std::chrono::milliseconds(batched && streaming ? 5 : 50),done,false);
                drain();
                if (writer.client_gone() || task->cancelled) { task->cancelled = true; ticket->cancel(); }
                if (done) break;
            }
            drain();
            { std::lock_guard lock(mutex); tasks.erase(task_id); }
            if (!ticket->error().empty()) throw std::runtime_error(ticket->error());
            if (writer.client_gone()) return;
            auto receipt = Json::object();
            auto phases = Json::array(); for (const auto& phase : task->result.receipts) phases.push(phase);
            receipt.set("phases",phases); receipt.set("budget_continued",task->result.budget_continued);
            auto generated_ids = Json::array();
            for (const auto id : task->result.ids) generated_ids.push(std::uint64_t(id));
            receipt.set("generated_ids",generated_ids);
            const auto usage = Json(Json::Object{{"prompt_tokens",std::uint64_t(task->result.prompt_tokens)},
                {"completion_tokens",std::uint64_t(task->result.ids.size())},{"total_tokens",std::uint64_t(task->result.prompt_tokens+task->result.ids.size())}});
            if (streaming) {
                auto final = envelope(Json(Json::Object{{"index",0},{"delta",Json::object()},{"finish_reason",task->result.finish}}),"chat.completion.chunk");
                final.set("strix",receipt);
                if (!writer.write("data: "+final.dump()+"\n\n")) return;
                if (get(get(body,"stream_options"),"include_usage").is_bool() && get(get(body,"stream_options"),"include_usage").as_bool()) {
                    auto u = envelope(Json(),"chat.completion.chunk"); u.set("choices",Json::array()); u.set("usage",usage);
                    if (!writer.write("data: "+u.dump()+"\n\n")) return;
                }
                (void)writer.write("data: [DONE]\n\n"); writer.end_stream();
            } else {
                auto response = envelope(Json(Json::Object{{"index",0},{"message",output_message(task->result.output)},{"finish_reason",task->result.finish}}),"chat.completion");
                response.set("usage",usage); response.set("strix",receipt);
                writer.send(200,"application/json",response.dump());
            }
        } catch (const std::exception& e) {
            if (!task_id.empty()) { std::lock_guard lock(mutex); tasks.erase(task_id); }
            const auto error = Json(Json::Object{{"error",Json(Json::Object{{"message",e.what()},
                {"type",failure_status == 400 ? "invalid_request_error" : "server_error"}})}}).dump();
            if (streaming) { (void)writer.write("data: "+error+"\n\n"); (void)writer.write("data: [DONE]\n\n"); writer.end_stream(); }
            else writer.send(failure_status,"application/json",error);
        }
    }
};
Api::Api(Backend& b, std::shared_ptr<const gemma4::serve::Tokenizer> t, DraftFactory d, DraftMode m, std::string name, Request defaults)
    : impl_(std::make_unique<Impl>(b,std::move(t),std::move(d),m,std::move(name),std::move(defaults))) {}
Api::~Api() = default;
void Api::handle(const HttpRequest& r, HttpResponseWriter& w) { impl_->handle(r,w); }
} // namespace strix::gemma4::engine
