#pragma once
#include "strix/serve/session_api.hpp"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <vector>
namespace strix::minimax::serve {
namespace sv = strix::serve;
// Session commands share the ordinary control channel and run only at an idle boundary.
// HTTP threads read published metadata or enqueue bounded tasks; the engine owns the cache/device.
// A stop token ends the request even when it is also the last allowed token; otherwise reaching the limit (max_tokens
// or the end of the context) is 'length'.
inline const char* session_finish(std::span<const std::uint32_t> generated, std::size_t limit, std::span<const std::uint32_t> stops) {
    if (!generated.empty() && std::find(stops.begin(), stops.end(), generated.back()) != stops.end()) return "stop";
    return generated.size() < limit ? "stop" : "length";
}
struct SessionTasks {
    struct Task { std::string body; std::promise<sv::Json> result; bool dispatched = false; };
    std::mutex mutex;
    std::deque<std::shared_ptr<Task>> tasks;
    bool accepting = true;
    sv::Json sessions = sv::Json::object(), stats = sv::Json::object();
    void publish_metadata(sv::Json list, sv::Json counters) {
        std::lock_guard<std::mutex> lock(mutex);
        sessions = std::move(list); stats = std::move(counters);
    }
    std::atomic<bool> pending{false};
    ~SessionTasks() { close(); }
    void close() {
        std::lock_guard<std::mutex> lock(mutex); accepting = false;
        for (auto& t : tasks) t->result.set_value(sv::session_error("unavailable", "server stopped"));
        tasks.clear(); pending.store(false, std::memory_order_relaxed);
    }
    sv::Json submit(const sv::Json& command,
                    std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!accepting) throw sv::SessionApiError(503, "unavailable", "server stopped; retry after 1 second");
            const auto op = command.find("op")->as_string();
            if (op == "list") return sessions;
            if (op == "stats") return stats;
        }
        auto task = std::make_shared<Task>(); task->body = command.dump();
        auto future = task->result.get_future();
        { std::lock_guard<std::mutex> lock(mutex);
          if (!accepting || tasks.size() >= 16) throw sv::SessionApiError(503, "unavailable", "session queue closed/full; retry after 1 second");
          tasks.push_back(task); pending.store(true, std::memory_order_relaxed); }
        if (future.wait_for(timeout) != std::future_status::ready) {
            std::lock_guard<std::mutex> lock(mutex);
            // An undispatched task must never mutate state after a busy response.
            const auto found = std::find(tasks.begin(), tasks.end(), task);
            if (!task->dispatched && found != tasks.end()) {
                tasks.erase(found);
                pending.store(!tasks.empty(), std::memory_order_relaxed);
                throw sv::SessionApiError(503, "unavailable", "session busy; retry after 1 second");
            }
        }
        return future.get();
    }
    std::shared_ptr<Task> take() {
        std::lock_guard<std::mutex> lock(mutex);
        if (tasks.empty()) return {};
        auto task = tasks.front(); tasks.pop_front(); task->dispatched = true;
        pending.store(!tasks.empty(), std::memory_order_relaxed); return task;
    }
};
template<class Engine>
std::string session_wire(Engine& eng, std::string body, std::size_t size) {
    body.resize(size);
    std::string peer(size, ' ');
    for (std::size_t i = 0; i < size; i += 65536)
        eng.exchange_bytes(body.data() + i, peer.data() + i, std::min<std::size_t>(65536, size - i));
    return eng.rank() == 0 ? body : peer;
}
struct SessionCounts { std::uint64_t hits = 0, misses = 0, refusals = 0; };
template<class Runtime>
sv::Json session_metadata(Runtime& rt, const SessionCounts& counts, bool list) {
    auto reply = sv::Json::object();
    if (list) {
        auto sessions = sv::Json::array();
        for (const auto& c : rt.describe()) {
            auto item = sv::Json::object(); item.set("id",sv::Json(c.id));
            item.set("position",sv::Json(c.frontier.consumed)); item.set("consumed",sv::Json(c.frontier.consumed));
            item.set("pending",sv::Json(std::uint64_t(c.frontier.pending)));
            item.set("bytes",sv::Json(c.bytes)); item.set("state",sv::Json(rt.in_ram(c.id) ? "parked" : "disk")); item.set("durability",sv::Json(rt.on_disk(c.id) ? "disk" : "ram"));
            sessions.push(std::move(item));
        }
        reply.set("sessions",std::move(sessions));
    } else {
        const auto st = rt.stats();
        reply.set("ram_bytes",sv::Json(st.bytes)); reply.set("ram_budget",sv::Json(rt.ram_budget()));
        reply.set("parked",sv::Json(st.entries)); reply.set("disk_bytes",sv::Json(st.disk_bytes));
        reply.set("hits",sv::Json(counts.hits)); reply.set("misses",sv::Json(counts.misses)); reply.set("refusals",sv::Json(counts.refusals));
    }
    return reply;
}
template<class Engine, class Runtime>
sv::Json session_execute(Engine& eng, Runtime& rt,
                         const std::string& body, SessionCounts& counts,
                         const std::function<bool(std::span<const std::uint32_t>)>& publish = {}) {
    auto reply = sv::Json::object();
    try {
        const auto req = sv::parse_json(body);
        const auto op = req.find("op")->as_string();
        const auto id = req.find("id")->as_string();
        if (op == "generate") {
            std::vector<std::uint32_t> prompt;
            for (const auto& t : req.find("tokens")->as_array()) prompt.push_back(static_cast<std::uint32_t>(t.as_int()));
            const bool hit = rt.begin_request(id, prompt);
            hit ? ++counts.hits : ++counts.misses;
            auto ids = sv::Json::array();
            std::vector<std::uint32_t> stops;
            for (const auto& t : req.find("stops")->as_array()) stops.push_back(static_cast<std::uint32_t>(t.as_int()));
            // Generation stops at the end of the context like a max_tokens limit (finish "length").
            const auto room = rt.generation_room();
            if (room == 0) { rt.park(id); throw sv::SessionApiError(400, "context_length_exceeded", "the session fills the context; nothing can be generated"); }
            const auto n = static_cast<unsigned>(std::min<std::uint64_t>(static_cast<std::uint64_t>(req.find("count")->as_int()), room));
            const auto generated = rt.generate(n, {}, stops, publish);
            for (const auto token : generated) ids.push(sv::Json(std::uint64_t(token)));
            // defer_park: the caller answers the request first and then sends park_active before anything else runs.
            const auto* defer = req.find("defer_park");
            if (!(defer && defer->is_bool() && defer->as_bool())) rt.park(id);
            reply.set("token_ids", std::move(ids)); reply.set("hit", sv::Json(hit));
            reply.set("finish", sv::Json(session_finish(generated, n, stops)));
        } else if (op == "park_active") {
            // Parks the session a deferred generate left active (no restore: its state is still in the slot).
            if (!rt.active()) throw sv::SessionApiError(409, "session_incompatible", "no active session to park");
            rt.park(id);
        } else if (op == "checkpoint" || op == "ram_checkpoint") {
            // Finished requests are parked; resume before checkpointing their committed state.
            rt.restore_auto(id);
            rt.checkpoint(id, op == "checkpoint");
        } else if (op == "resume" || op == "park" || op == "delete") {
            if (op == "delete") rt.erase(id);
            else {
                rt.restore_auto(id);
                // Device slots remain available to ordinary jobs after every operation.
                rt.park(id);
            }
        } else if (op == "list" || op == "stats") {
            reply = session_metadata(rt, counts, op == "list");
        }
    } catch (const std::exception& e) {
        ++counts.refusals;
        const auto* api_error = dynamic_cast<const sv::SessionApiError*>(&e);
        const auto error = api_error ? *api_error : sv::session_failure(e.what());
        reply = sv::session_error(error.code, error.what()); reply.set("_status",sv::Json(std::int64_t(error.status)));
    }
    // Every local-only refusal is visible on both ranks before the next command.
    const auto ok = reply.find("error") == nullptr;
    const auto peer = eng.peer(ok ? 1U : 0U);
    if (ok && !peer) { reply = sv::session_error("session_incompatible","peer refused operation"); reply.set("_status",sv::Json(std::int64_t(409))); }
    return reply;
}

struct SessionService {
    using Publish = std::function<bool(std::span<const std::uint32_t>)>;
    SessionTasks tasks;
    std::function<sv::Json(const std::string&, const Publish&)> execute;
};
} // namespace strix::minimax::serve
