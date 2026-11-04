#pragma once
// Opt-in decode timeline (--decode-profile): the runtime records one CUDA event on
// the compute stream after each phase it enqueues and the host time spent enqueuing
// it; after the token's synchronization the event intervals are turned into device
// milliseconds per (layer, phase). This header holds the host-only part (phase names,
// the per-token JSON and the per-arm totals) so it is tested without a device.
// Events only: no kernel, copy or ordering changes, so tokens and digests are the
// same with or without the timeline (timings include the events' small cost).
#include "strix/serve/json.hpp"
#include <array>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::glm::rtxpf::producer {
using strix::serve::Json;

enum class DecodePhase : unsigned {
    embed,         // token write + embedding
    hc_pre_attn,   // hyper-connection pre-mix + input norm
    kda,           // KDA layer: open-block checkpoint replay (state/conv copies) + the chunked KDA call
    dsa,           // DSA layer: index scoring over the context + sparse attention (one call)
    hc_post_attn,  // hyper-connection expand after attention
    hc_pre_ffn,    // pre-mix + post-attention norm
    dense_ffn,     // layers 0..2: dense FFN
    router,        // MoE router (top-8 + rates)
    export_wait,   // cache export + device-side wait for the layer's units + descriptor bind
    predict,       // predicted prefetch fork (input copy + side-stream router), when on
    routed_moe,    // routed experts (CUTLASS grouped GEMMs)
    shared_expert, // shared expert
    combine,       // routed + shared add, release
    hc_post_ffn,   // hyper-connection expand after the FFN
    head,          // final mix + norm + lm_head + argmax + result copies
    host_gap,      // device idle while the host synchronizes, checks and reads the result
    mtp,           // MTP maintenance (its own export/wait and experts) + its synchronization
    stage,         // prefill: the layer's staging (trunk H2D, expert-cache make_resident / release); device idle while the host stages
    state_export,  // prefill: the layer's state export (stream synchronization, D2H copies, the sink)
    count
};
inline constexpr std::array<const char*, std::size_t(DecodePhase::count)> decode_phase_names{
    "embed", "hc_pre_attn", "kda", "dsa", "hc_post_attn", "hc_pre_ffn", "dense_ffn", "router", "export_wait",
    "predict", "routed_moe", "shared_expert", "combine", "hc_post_ffn", "head", "host_gap", "mtp", "stage", "state_export"};

struct TimelineMark {
    int layer = -1;  // model layer, -1 = outside the layer loop
    DecodePhase phase = DecodePhase::embed;
    double host_ms = 0;  // host time from the previous mark to this one (enqueue cost, or a synchronization)
};

// One token. gpu_ms[i] = device time from the previous event (the token's start event
// for i = 0) to mark i's event; span_ms = first to last event.
inline Json decode_timeline_json(const std::vector<TimelineMark>& marks, const std::vector<double>& gpu_ms, double span_ms) {
    if (marks.size() != gpu_ms.size()) throw std::invalid_argument("decode timeline: one device interval per mark");
    constexpr std::size_t P = std::size_t(DecodePhase::count);
    std::array<double, P> gpu{}, host{};
    std::array<unsigned, P> n{};
    std::map<int, std::array<double, P>> layers;
    double gpu_total = 0, host_total = 0;
    for (std::size_t i = 0; i < marks.size(); ++i) {
        const auto p = std::size_t(marks[i].phase);
        if (p >= P) throw std::invalid_argument("decode timeline: phase");
        gpu[p] += gpu_ms[i];
        host[p] += marks[i].host_ms;
        ++n[p];
        gpu_total += gpu_ms[i];
        host_total += marks[i].host_ms;
        if (marks[i].layer >= 0) layers[marks[i].layer][p] += gpu_ms[i];
    }
    auto phases = Json::object();
    for (std::size_t p = 0; p < P; ++p) {
        if (!n[p]) continue;
        auto o = Json::object();
        o.set("gpu_ms", gpu[p]);
        o.set("host_ms", host[p]);
        o.set("count", std::uint64_t(n[p]));
        phases.set(decode_phase_names[p], o);
    }
    auto per_layer = Json::array();
    for (const auto& [layer, row] : layers) {
        auto o = Json::object();
        o.set("layer", layer);
        for (std::size_t p = 0; p < P; ++p)
            if (row[p] != 0) o.set(decode_phase_names[p], row[p]);
        per_layer.push(o);
    }
    auto j = Json::object();
    j.set("span_ms", span_ms);
    j.set("gpu_ms_total", gpu_total);
    j.set("host_ms_total", host_total);
    j.set("phases", phases);
    j.set("layers", per_layer);
    return j;
}

// A bench arm: per phase the device/host milliseconds summed over the steps that carry
// a timeline and their per-token means, plus the mean span; empty object = none carried one.
inline Json aggregate_decode_timelines(const Json& steps) {
    constexpr std::size_t P = std::size_t(DecodePhase::count);
    std::array<double, P> gpu{}, host{};
    double span = 0;
    std::map<std::string,double> sub_ms;
    std::map<std::string,std::uint64_t> sub_calls,sub_skipped;
    std::uint64_t tokens = 0;
    for (const auto& s : steps.as_array()) {
        const Json* t = s.find("timeline");
        if (!t) continue;
        ++tokens;
        span += t->find("span_ms")->as_double();
        if(const auto* sub=t->find("subcounters"))for(const auto& [name,v]:sub->as_object()){
            sub_ms[name]+=v.find("gpu_ms")->as_double();sub_calls[name]+=std::uint64_t(v.find("calls")->as_int());sub_skipped[name]+=std::uint64_t(v.find("skipped_calls")->as_int());}
        const Json* ph = t->find("phases");
        for (std::size_t p = 0; p < P; ++p)
            if (const Json* o = ph->find(decode_phase_names[p])) {
                gpu[p] += o->find("gpu_ms")->as_double();
                host[p] += o->find("host_ms")->as_double();
            }
    }
    auto j = Json::object();
    if (!tokens) return j;
    j.set("tokens", tokens);
    j.set("span_ms_per_token", span / double(tokens));
    if(!sub_ms.empty()){auto sub=Json::object();for(const auto& [name,ms]:sub_ms){auto v=Json::object();v.set("gpu_ms_total",ms);v.set("gpu_ms_per_token",ms/double(tokens));v.set("calls",sub_calls[name]);v.set("skipped_calls",sub_skipped[name]);sub.set(name,v);}j.set("subcounters",sub);}
    auto phases = Json::object();
    for (std::size_t p = 0; p < P; ++p) {
        if (gpu[p] == 0 && host[p] == 0) continue;
        auto o = Json::object();
        o.set("gpu_ms_per_token", gpu[p] / double(tokens));
        o.set("host_ms_per_token", host[p] / double(tokens));
        o.set("gpu_ms_total", gpu[p]);
        phases.set(decode_phase_names[p], o);
    }
    j.set("phases", phases);
    return j;
}
// Prefill (--prefill-profile): one routed-MoE call (a layer's row tile), its 16-expert groups' device times summed
// (launch_cutlass's Profile events). decode_ms runs on the decode stream beside the GEMMs; decode_wait_ms is the
// compute stream waiting for a group's decoded FP16 weights, so decode_wait_ms > 0 means the call is decode-bound.
struct MoeCallTimes {
    int layer = -1;
    unsigned rows = 0, groups = 0;
    double transform_ms = 0, decode_ms = 0, decode_wait_ms = 0, setup_ms = 0, gate_up_ms = 0, epilogue_ms = 0, down_ms = 0, merge_ms = 0;
};
// Totals, per layer, and the GEMM rates: gate+up = 4 x hidden x inter and down = 2 x hidden x inter FLOP per
// (row, routed expert) pair, rows x topk pairs per call.
inline Json prefill_moe_json(const std::vector<MoeCallTimes>& calls, unsigned topk, unsigned hidden, unsigned inter) {
    struct Sum { double transform = 0, decode = 0, wait = 0, setup = 0, gu = 0, epi = 0, down = 0, merge = 0, gu_flop = 0, down_flop = 0; std::uint64_t rows = 0, calls = 0; };
    auto add = [&](Sum& t, const MoeCallTimes& c) {
        t.transform += c.transform_ms; t.decode += c.decode_ms; t.wait += c.decode_wait_ms; t.setup += c.setup_ms;
        t.gu += c.gate_up_ms; t.epi += c.epilogue_ms; t.down += c.down_ms; t.merge += c.merge_ms;
        const double pairs = double(c.rows) * topk;
        t.gu_flop += 4.0 * hidden * inter * pairs; t.down_flop += 2.0 * hidden * inter * pairs;
        t.rows += c.rows; ++t.calls;
    };
    auto json = [](const Sum& t) {
        auto o = Json::object();
        o.set("calls", t.calls); o.set("rows", t.rows);
        o.set("transform_ms", t.transform); o.set("weight_decode_ms", t.decode); o.set("decode_wait_ms", t.wait); o.set("setup_ms", t.setup);
        o.set("gate_up_gemm_ms", t.gu); o.set("epilogue_ms", t.epi); o.set("down_gemm_ms", t.down); o.set("merge_ms", t.merge);
        o.set("gate_up_tflop", t.gu_flop / 1e12); o.set("down_tflop", t.down_flop / 1e12);
        o.set("gate_up_tflops", t.gu > 0 ? t.gu_flop / (t.gu * 1e9) : 0.0); o.set("down_tflops", t.down > 0 ? t.down_flop / (t.down * 1e9) : 0.0);
        return o;
    };
    Sum total;
    std::map<int, Sum> layers;
    for (const auto& c : calls) { add(total, c); add(layers[c.layer], c); }
    auto per_layer = Json::array();
    for (const auto& [layer, t] : layers) { auto o = json(t); o.set("layer", layer); per_layer.push(o); }
    auto j = json(total);
    j.set("layers", per_layer);
    return j;
}
}  // namespace strix::glm::rtxpf::producer
