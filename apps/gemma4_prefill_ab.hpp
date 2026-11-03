#pragma once
// strix-gemma4-decode --ab-switch ...: in-process A/B of one exact prompt-prefill lever through the engine path that
// strix-gemma4-decode times: the model loads once, then each prompt length runs
// the arms in order (default off,on,off) with the lever's OFF switch set or
// cleared before each request. Every arm must reproduce the first arm's ids,
// state digest and receipt (timings and the prefill_levers counters excepted).
// Prefill levers read their switch per chunk; nothing else reads it.
#include "gemma4/engine/runtime.hpp"
#include "gemma4/kernels/capture_guard.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <iostream>
#include <sstream>

namespace prefill_ab {
namespace e = strix::gemma4::engine;
using Json = strix::serve::Json;
// A group entry NAME=0 is held off for every arm of that group (a lever the
// one under test depends on); plain NAME entries are the levers under test.
inline void set_switch(const std::string& entry, bool on) {
    const auto eq = entry.find('=');
    const auto name = entry.substr(0, eq);
    if (eq != std::string::npos) on = false;
#ifdef _WIN32
    (void)_putenv_s(name.c_str(), on ? "" : "0");
#else
    if (on) (void)unsetenv(name.c_str()); else (void)setenv(name.c_str(), "0", 1);
#endif
}
inline std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> out; std::stringstream in(text); std::string item;
    while (std::getline(in, item, ',')) if (!item.empty()) out.push_back(item);
    return out;
}
// The receipt without wall-clock timings and lever engagement fields.
inline std::string comparable(const Json& receipt) {
    Json out = Json::object();
    for (const auto& [key, value] : receipt.as_object())
        if (key != "timings" && key != "prefill_levers" && key != "prefill_taps" && key != "ab_run_ms" && key != "ab_graphs") out.set(key, value);
    return out.dump();
}
inline double number(const Json& j, const char* object, const char* key) {
    const auto* o = j.find(object); const auto* v = o ? o->find(key) : nullptr;
    return v ? v->as_double() : 0.0;
}
inline int run(int argc, char** argv) {
    try {
        std::vector<std::vector<std::string>> groups;
        std::vector<std::string> arms{"off", "on", "off"};
        std::vector<std::size_t> lengths;
        unsigned warmup = 1, streams = 1;
        bool decode = false; // --ab-decode: time the whole run (prompt + generation) and report generated tokens/s
        std::vector<char*> rest{argv[0]};
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value = [&]() -> std::string { if (++i == argc) throw std::invalid_argument("missing value for " + arg); return argv[i]; };
            if (arg == "--ab-switch") groups.push_back(split(value()));
            else if (arg == "--ab-lengths") for (const auto& l : split(value())) lengths.push_back(std::stoull(l));
            else if (arg == "--ab-arms") arms = split(value());
            else if (arg == "--ab-warmup") warmup = unsigned(std::stoul(value()));
            else if (arg == "--ab-streams") streams = unsigned(std::stoul(value()));
            else if (arg == "--ab-decode") decode = true;
            else rest.push_back(argv[i]);
        }
        for (const auto& a : arms) if (a != "on" && a != "off") throw std::invalid_argument("--ab-arms takes on/off");
        if (groups.empty() || lengths.empty() || arms.empty())
            throw std::invalid_argument("usage: strix-gemma4-decode --ab-switch STRIX_G4_X[,..] --ab-lengths 512,2048 [--ab-arms off,on,off] "
                                        "[--ab-warmup 1] [--ab-streams N] [--ab-decode] <strix-gemma4-decode options with --ids IDS.txt>");
        for (const auto& g : groups) for (const auto& s : g)
            if (!s.starts_with("STRIX_G4_")) throw std::invalid_argument("lever switches are STRIX_G4_*");
        const auto options = e::parse_options(int(rest.size()), rest.data());
        if (options.ids.empty()) throw std::invalid_argument("--ids IDS.txt required");
        std::vector<std::uint32_t> source;
        { std::istringstream in(e::read_text(options.ids)); std::string w; while (in >> w) source.push_back(std::uint32_t(std::stoul(w))); }
        if (source.empty()) throw std::invalid_argument("empty --ids");
        e::Runtime runtime(options);
        const auto run = [&](const std::vector<std::string>& switches, std::size_t length, bool on) {
            for (const auto& s : switches) set_switch(s, on);
            e::Coordinator coordinator(*runtime.backend, runtime.drafts);
            // --ab-streams N: N concurrent requests, stream i's prompt starting i * 97 tokens into the ids.
            std::vector<std::size_t> slots;
            for (unsigned k = 0; k < streams; ++k) {
                auto request = options.request;
                request.prompt.resize(length);
                for (std::size_t i = 0; i < length; ++i) request.prompt[i] = source[(i + std::size_t(k) * 97U) % source.size()];
                slots.push_back(coordinator.submit(std::move(request)));
            }
            auto& graphs = strix::gemma4::gpu::capture_guard_state();
            const std::uint64_t before[7] = {graphs.target_captures.load(), graphs.target_capture_ns.load(), graphs.target_replays.load(),
                                             graphs.draft_captures.load(), graphs.draft_capture_ns.load(), graphs.draft_replays.load(),
                                             graphs.eager_passes.load()};
            const auto begin = std::chrono::steady_clock::now();
            coordinator.run();
            const double run_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            auto ids = Json::array();
            std::string digests;
            std::uint64_t generated = 0;
            for (const auto slot : slots) {
                if (coordinator.result(slot).status == e::Status::failed) throw std::runtime_error("request failed: " + coordinator.result(slot).error);
                for (const auto id : coordinator.result(slot).ids) ids.push(std::uint64_t(id));
                generated += coordinator.result(slot).ids.size();
                digests += coordinator.receipt(slot).find("state_digest")->as_string();
            }
            auto receipt = coordinator.receipt(slots.front());
            receipt.set("ids", ids);
            if (streams > 1) receipt.set("state_digest", digests);
            receipt.set("ab_run_ms", run_ms); receipt.set("ab_generated", generated);
            // Graph work of this run (timing evidence, excluded from receipt equality).
            auto counted = Json::object();
            counted.set("target_captures", graphs.target_captures.load() - before[0]);
            counted.set("target_capture_ms", double(graphs.target_capture_ns.load() - before[1]) / 1e6);
            counted.set("target_replays", graphs.target_replays.load() - before[2]);
            counted.set("draft_captures", graphs.draft_captures.load() - before[3]);
            counted.set("draft_capture_ms", double(graphs.draft_capture_ns.load() - before[4]) / 1e6);
            counted.set("draft_replays", graphs.draft_replays.load() - before[5]);
            counted.set("eager_passes", graphs.eager_passes.load() - before[6]);
            receipt.set("ab_graphs", counted);
            return receipt;
        };
        for (unsigned w = 0; w < warmup; ++w) (void)run(groups.front(), std::min<std::size_t>(lengths.front(), 512), true);
        bool exact = true;
        // One reference per length across every group: all arms of all groups must agree.
        struct Reference { Json receipt; std::string digest, ids; bool set = false; };
        std::map<std::size_t, Reference> references;
        for (const auto& switches : groups) {
            const auto name = [&] { std::string n; for (const auto& s : switches) n += (n.empty() ? "" : ",") + s; return n; }();
            for (const auto length : lengths) {
                auto& first = references[length];
                std::vector<double> off, on;
                bool same = true, receipts = true;
                for (const auto& arm : arms) {
                    auto receipt = run(switches, length, arm == "on");
                    const auto ms = decode ? receipt.find("ab_run_ms")->as_double() : number(receipt, "timings", "prefill_ms");
                    const auto state = receipt.find("state_digest")->as_string(), produced = receipt.find("ids") ? receipt.find("ids")->dump() : "";
                    if (!first.set) { first.receipt = receipt; first.digest = state; first.ids = produced; first.set = true; }
                    const auto text = comparable(receipt), reference = comparable(first.receipt);
                    const bool equal_state = state == first.digest && produced == first.ids, equal_receipt = text == reference;
                    same &= equal_state; receipts &= equal_receipt;
                    (arm == "on" ? on : off).push_back(ms);
                    Json line = Json::object();
                    line.set("ab_switch", name); line.set("arm", arm); line.set("rows", std::uint64_t(length));
                    if (decode) {
                        const auto generated = receipt.find("ab_generated")->as_double();
                        line.set("streams", std::uint64_t(streams)); line.set("run_ms", ms); line.set("generated", generated);
                        line.set("generated_per_s", ms > 0 ? generated * 1000.0 / ms : 0.0);
                    } else {
                        line.set("prefill_ms", ms); line.set("tokens_per_s", ms > 0 ? double(length - 1) * 1000.0 / ms : 0.0);
                    }
                    line.set("prefill_host_ms", number(receipt, "timings", "prefill_host_ms"));
                line.set("context_ms", number(receipt, "timings", "context_ms")); line.set("commit_ms", number(receipt, "timings", "commit_ms"));
                    line.set("target_calls", *receipt.find("target_calls"));
                    line.set("graphs", *receipt.find("ab_graphs"));
                    line.set("state_digest", state); line.set("state_equal", equal_state); line.set("receipt_equal", equal_receipt);
                    if (const auto* levers = receipt.find("prefill_levers")) line.set("prefill_levers", *levers);
                if (const auto* taps = receipt.find("prefill_taps")) line.set("prefill_taps", *taps);
                    std::cout << line.dump() << std::endl;
                    if (!equal_receipt) std::cerr << "receipt differs (" << arm << "):\n" << text << "\nreference:\n" << reference << '\n';
                }
                const auto median = [](std::vector<double> v) {
                    if (v.empty()) return 0.0;
                    std::sort(v.begin(), v.end());
                    const auto h = v.size() / 2;
                    return v.size() % 2 ? v[h] : (v[h - 1] + v[h]) / 2.0;
                };
                Json verdict = Json::object();
                verdict.set("ab_switch", name); verdict.set("rows", std::uint64_t(length));
                verdict.set("off_ms", median(off)); verdict.set("on_ms", median(on));
                verdict.set("speedup", median(on) > 0 ? median(off) / median(on) : 0.0);
                verdict.set("verdict", same && receipts ? "EXACT" : same ? "EXACT-IDS-STATE (receipt differs)" : "MISMATCH");
                std::cout << verdict.dump() << std::endl;
                exact &= same;
            }
            for (const auto& s : switches) set_switch(s.substr(0, s.find('=')), true);
        }
        return exact ? 0 : 3;
    } catch (const std::exception& error) { std::cerr << "gemma4 prefill ab: " << error.what() << '\n'; return 1; }
}
} // namespace prefill_ab
