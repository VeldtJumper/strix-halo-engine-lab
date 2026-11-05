// g4-a8-v1 device bytes: quantization and every quantized format/epilogue against the host oracle for
// R=1..8, then the complete synthetic graph against the host graph, R rows against serial passes.
#include "gemma4/kernels/a8_host.hpp"
#include "gemma4/kernels/a8_ops.hpp"
#include "gemma4/kernels/capture_guard.hpp"
#include "gemma4/kernels/device.hpp"
#include "../gemma4_gpu/fixture.hpp"
#include <limits>
#include <cstdlib>
// This file also belongs to the host-only declaration/CLI compile target (no HIP headers).
#if __has_include(<hip/hip_runtime.h>)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime.h>
#define G4_A8_GRAPH_TESTS 1
#endif

using namespace strix::gemma4;
using namespace strix::gemma4::gpu;
namespace {
bool enabled(const char* name) {
    const auto* v = std::getenv(name);
    return !v || std::string_view(v) != "0";
}
bool one(const char* name) {
    const auto* v = std::getenv(name);
    return v && std::string_view(v) == "1";
}
bool ffn_ple_admitted() {
    return enabled("STRIX_G4_FAST_EPILOGUE") && enabled("STRIX_G4_NORM_STAGE") &&
           enabled("STRIX_G4_NORM_SEARCH") && !one("STRIX_G4_EPILOGUE_EARLY_LOADS") &&
           !one("STRIX_G4_A8_HALF") && !one("STRIX_G4_A8_PIPE") && !one("STRIX_G4_A8_STREAM");
}
int environment(const char* name, const char* value) {
#ifdef _WIN32
    return _putenv_s(name, value ? value : "");
#else
    return value ? setenv(name, value, 1) : unsetenv(name);
#endif
}
struct ScopedEnv {
    const char* name;
    bool present;
    std::string value;
    ScopedEnv(const char* key, const char* replacement) : name(key), present(std::getenv(key) != nullptr) {
        if (present) value = std::getenv(key);
        CHECK(environment(name, replacement) == 0);
    }
    ~ScopedEnv() { (void)environment(name, present ? value.c_str() : nullptr); }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;
};
std::vector<float> download(const Buffer& b, std::size_t count) {
    std::vector<float> v(count);
    b.download(v.data(), count * sizeof(float));
    return v;
}
LinearView device_view(Buffer& storage, const std::vector<std::uint8_t>& bytes, WeightFormat f, unsigned outputs,
                       unsigned width) {
    storage.resize(bytes.size());
    storage.upload(bytes.data(), bytes.size());
    return {storage.as<std::uint8_t>(), outputs, width, packed_row_bytes(f, width), f};
}
void kernel_tests() {
    std::mt19937 rng(91);
    for (const auto f : {WeightFormat::q8_0, WeightFormat::q4_k, WeightFormat::q6_k})
        for (const unsigned width : {256U, 768U, 2560U}) {
            const unsigned outputs = 37;
            const auto wb = fixture::packed(f, outputs, width, 7U * width + unsigned(f)),
                       ub = fixture::packed(f, outputs, width, 9U * width + unsigned(f)),
                       eb = fixture::packed(f, 21, width, 11U * width + unsigned(f));
            const LinearView hw{wb.data(), outputs, width, packed_row_bytes(f, width), f},
                hu{ub.data(), outputs, width, packed_row_bytes(f, width), f},
                he{eb.data(), 21, width, packed_row_bytes(f, width), f};
            Buffer ws, us, es;
            const auto dw = device_view(ws, wb, f, outputs, width), du = device_view(us, ub, f, outputs, width),
                       de = device_view(es, eb, f, 21, width);
            std::vector<float> x(8ULL * width), product(8ULL * outputs);
            for (auto& v : x)
                v = std::ldexp(float(int(rng() % 4001) - 2000), int(rng() % 9) - 12);
            for (unsigned i = 0; i < 64; ++i)
                x[rng() % x.size()] = 0.0F; // zero groups and signed zeros
            for (unsigned i = 0; i < 32; ++i)
                x[width + i] = i % 2 ? -0.0F : 0.0F;
            for (auto& v : product)
                v = float(int(rng() % 201) - 100) * 0.01F;
            Buffer dx(x.size() * 4), dp(product.size() * 4), codes(8ULL * width), meta(8ULL * width / 32U * 8U),
                out(8ULL * outputs * 4), extra(8ULL * 21 * 4), failure(4);
            dx.upload(x.data(), x.size() * 4);
            dp.upload(product.data(), product.size() * 4);
            failure.zero();
            const A8Rows rows_view{codes.as<std::int8_t>(), meta.as<a8::Meta>(), width};
            for (unsigned rows = 1; rows <= 8; ++rows) {
                std::vector<std::int8_t> hc;
                std::vector<a8::Meta> hm;
                a8::quantize_rows(x.data(), rows, width, hc, hm);
                a8_quantize(dx.as<float>(), rows, width, rows_view);
                synchronize();
                std::vector<std::int8_t> dc(hc.size());
                std::vector<a8::Meta> dm(hm.size());
                codes.download(dc.data(), dc.size());
                meta.download(dm.data(), dm.size() * sizeof(a8::Meta));
                CHECK(std::memcmp(dc.data(), hc.data(), hc.size()) == 0);
                CHECK(std::memcmp(dm.data(), hm.data(), hm.size() * sizeof(a8::Meta)) == 0);
                const auto groups = width / 32U;
                auto host_dot = [&](LinearView v, unsigned o, unsigned r) {
                    return a8::dot(v, o, hc.data() + std::size_t(r) * width, hm.data() + std::size_t(r) * groups);
                };
                // Plain, PLE-product, softcap, gate/up and a three-matrix shared grid.
                for (unsigned mode = 0; mode < 5; ++mode) {
                    LinearEpilogue ep;
                    if (mode == 1) {
                        ep.product = dp.as<float>();
                        ep.product_stride = outputs;
                    }
                    ep.cap = mode == 2;
                    Buffer second(8ULL * outputs * 4);
                    if (mode == 4) {
                        ep.extra[0] = de;
                        ep.extra_out[0] = extra.as<float>();
                        ep.extra[1] = dw;
                        ep.extra_out[1] = second.as<float>();
                    }
                    if (mode == 3)
                        CHECK(a8_gate_up(dw, du, rows_view, rows, out.as<float>(), nullptr, failure.as<unsigned>()));
                    else
                        CHECK(a8_linear(dw, rows_view, rows, out.as<float>(), ep, nullptr, failure.as<unsigned>()));
                    synchronize();
                    const auto got = download(out, std::size_t(rows) * outputs);
                    std::vector<float> want(got.size());
                    for (unsigned r = 0; r < rows; ++r)
                        for (unsigned o = 0; o < outputs; ++o) {
                            float v = host_dot(hw, o, r);
                            if (mode == 1)
                                v = gelu_tanh(v) * product[std::size_t(r) * outputs + o];
                            if (mode == 2)
                                v = softcap(v);
                            if (mode == 3)
                                v = gelu_tanh(v) * host_dot(hu, o, r);
                            want[std::size_t(r) * outputs + o] = v;
                        }
                    bytes(got, want);
                    if (mode == 4) {
                        const auto e = download(extra, std::size_t(rows) * 21);
                        std::vector<float> ew(e.size());
                        for (unsigned r = 0; r < rows; ++r)
                            for (unsigned o = 0; o < 21; ++o)
                                ew[std::size_t(r) * 21 + o] = host_dot(he, o, r);
                        bytes(e, ew);
                        const auto s = download(second, std::size_t(rows) * outputs);
                        std::vector<float> sw(s.size());
                        for (unsigned r = 0; r < rows; ++r)
                            for (unsigned o = 0; o < outputs; ++o)
                                sw[std::size_t(r) * outputs + o] = host_dot(hw, o, r);
                        bytes(s, sw);
                    }
                }
            }
            unsigned flag = 1;
            failure.download(&flag, 4);
            CHECK(flag == 0);
        }
    // Quantized outputs (blocks of 32 outputs) equal the separate quantization of the stored values.
    for (const auto f : {WeightFormat::q4_k, WeightFormat::q6_k}) {
        const unsigned width = 512, outputs = 256;
        const auto wb = fixture::packed(f, outputs, width, 41U + unsigned(f)),
                   ub = fixture::packed(f, outputs, width, 43U + unsigned(f));
        Buffer ws, us;
        const auto dw = device_view(ws, wb, f, outputs, width), du = device_view(us, ub, f, outputs, width);
        std::vector<float> x(8ULL * width);
        for (auto& v : x)
            v = float(int(rng() % 2001) - 1000) * 0.001F;
        Buffer dx(x.size() * 4), codes(8ULL * width), meta(8ULL * width / 4U), out(8ULL * outputs * 4),
            qcodes(8ULL * outputs), qmeta(8ULL * outputs / 4U), failure(4);
        dx.upload(x.data(), x.size() * 4);
        failure.zero();
        const A8Rows in{codes.as<std::int8_t>(), meta.as<a8::Meta>(), width},
            quant{qcodes.as<std::int8_t>(), qmeta.as<a8::Meta>(), outputs};
        for (unsigned rows = 1; rows <= 8; ++rows)
            for (const bool gate : {false, true}) {
                a8_quantize(dx.as<float>(), rows, width, in);
                if (gate)
                    CHECK(a8_gate_up(dw, du, in, rows, out.as<float>(), nullptr, failure.as<unsigned>(), quant));
                else
                    CHECK(a8_linear(dw, in, rows, out.as<float>(), {}, nullptr, failure.as<unsigned>(), quant));
                synchronize();
                const auto values = download(out, std::size_t(rows) * outputs);
                std::vector<std::int8_t> hc, dc(std::size_t(rows) * outputs);
                std::vector<a8::Meta> hm, dm(std::size_t(rows) * outputs / 32U);
                a8::quantize_rows(values.data(), rows, outputs, hc, hm);
                qcodes.download(dc.data(), dc.size());
                qmeta.download(dm.data(), dm.size() * sizeof(a8::Meta));
                CHECK(std::memcmp(dc.data(), hc.data(), hc.size()) == 0);
                CHECK(std::memcmp(dm.data(), hm.data(), hm.size() * sizeof(a8::Meta)) == 0);
            }
    }
    // BF16 views are not quantized linears.
    const auto bb = fixture::packed(WeightFormat::bf16, 4, 256);
    Buffer bs, codes(256), meta(64), out(16), failure(4);
    const auto bv = device_view(bs, bb, WeightFormat::bf16, 4, 256);
    CHECK(!a8_linear(bv, {codes.as<std::int8_t>(), meta.as<a8::Meta>(), 256}, 1, out.as<float>(), {}, nullptr,
                     failure.as<unsigned>()));
}
// The consumer prologue must preserve both slots, not just its gate outputs. Keep the residual
// evolving across launches/replays: resetting inputs or tickets between passes would hide races.
void ffn_ple_tests() {
    constexpr unsigned width = 2560;
    const bool admitted = ffn_ple_admitted();
    std::mt19937 rng(817);
    for (const auto f : {WeightFormat::q8_0, WeightFormat::q4_k, WeightFormat::q6_k})
        for (const auto nf : {WeightFormat::f32, WeightFormat::bf16})
            for (unsigned n = 1; n <= 4; ++n)
                for (unsigned outputs : {37U, 256U}) {
                    const auto wb = fixture::packed(f, outputs, width, 991U + unsigned(f));
                    const auto nb = fixture::packed(nf, 1, width, 125U);
                    Buffer ws, ns;
                    const auto gate = device_view(ws, wb, f, outputs, width),
                               norm_weight = device_view(ns, nb, nf, 1, width);
                    const auto count = std::size_t(n) * width, out_count = std::size_t(n) * outputs;
                    std::vector<float> branch(count), residual(count), product(out_count + n * 7U);
                    Buffer db(count * 4), dp(product.size() * 4), r[2], out[2], codes[2], meta[2], qc[2], qm[2], flag[2], ticket(4);
                    A8Rows in[2], quant[2];
                    for (unsigned arm = 0; arm < 2; ++arm) {
                        r[arm].resize(count * 4); out[arm].resize(out_count * 4);
                        codes[arm].resize(count); meta[arm].resize(count / 32U * sizeof(a8::Meta));
                        qc[arm].resize(out_count); qm[arm].resize(out_count / 32U * sizeof(a8::Meta));
                        flag[arm].resize(4);
                        in[arm] = {codes[arm].as<std::int8_t>(), meta[arm].as<a8::Meta>(), width};
                        if (outputs % 32U == 0)
                            quant[arm] = {qc[arm].as<std::int8_t>(), qm[arm].as<a8::Meta>(), outputs};
                    }
                    LinearEpilogue ep;
                    ep.product = dp.as<float>(); ep.product_stride = outputs + 7U;
                    const auto fused = [&](Queue q) {
                        return a8_ffn_ple_gate(db.as<float>(), norm_weight, r[1].as<float>(), gate, dp.as<float>(),
                                               outputs + 7U, n, out[1].as<float>(), in[1], flag[1].as<unsigned>(),
                                               ticket.as<unsigned>(), q, quant[1]);
                    };
                    const auto separate = [&] {
                        CHECK(norm_residual(db.as<float>(), norm_weight, r[0].as<float>(), n, width, {}, nullptr, {},
                                            flag[0].as<unsigned>(), nullptr, in[0]));
                        CHECK(a8_linear(gate, in[0], n, out[0].as<float>(), ep, nullptr, flag[0].as<unsigned>(), quant[0]));
                    };
                    const auto same = [](const Buffer& a, const Buffer& b, std::size_t size) {
                        std::vector<std::uint8_t> av(size), bv(size);
                        a.download(av.data(), size); b.download(bv.data(), size);
                        CHECK(av == bv);
                    };
                    const auto compare = [&] {
                        synchronize();
                        same(r[0], r[1], count * 4); same(out[0], out[1], out_count * 4);
                        same(codes[0], codes[1], count); same(meta[0], meta[1], meta[0].size());
                        same(flag[0], flag[1], 4);
                        if (quant[0].codes) { same(qc[0], qc[1], out_count); same(qm[0], qm[1], qm[0].size()); }
                        unsigned t = 9; ticket.download(&t, 4); CHECK(t == 0);
                    };
                    for (unsigned kind = 0; kind < 8; ++kind) {
                        for (auto& v : branch) v = float(int(rng() % 2001) - 1000) * 0.00390625F;
                        for (auto& v : residual) v = float(int(rng() % 2001) - 1000) * 0.0078125F;
                        for (auto& v : product) v = float(int(rng() % 201) - 100) * 0.01F;
                        if (kind == 1) {
                            for (std::size_t i = 0; i < count; ++i) branch[i] = residual[i] = i % 2 ? -0.0F : 0.0F;
                        }
                        if (kind == 2) {
                            for (std::size_t i = 0; i < count; ++i)
                                branch[i] = residual[i] = (i % 2 ? -1.0F : 1.0F) * std::numeric_limits<float>::denorm_min();
                        }
                        if (kind == 3) {
                            for (auto& v : branch) v *= 1.0e15F;
                            for (auto& v : residual) v *= 1.0e30F;
                        }
                        if (kind == 4) branch[count / 2U] = 1.0e30F; // squared sum overflows
                        if (kind == 5) residual[count / 2U] = std::numeric_limits<float>::infinity();
                        if (kind == 6) branch[count / 2U] = std::numeric_limits<float>::quiet_NaN();
                        if (kind == 7) product[outputs / 2U] = std::numeric_limits<float>::infinity();
                        db.upload(branch.data(), db.size()); dp.upload(product.data(), dp.size());
                        for (unsigned arm = 0; arm < 2; ++arm) {
                            r[arm].upload(residual.data(), r[arm].size());
                            const unsigned sticky = 4; flag[arm].upload(&sticky, 4);
                        }
                        ticket.zero();
                        if (!admitted) { CHECK(!fused(nullptr)); return; }
                        for (unsigned repeat = 0; repeat < 2; ++repeat) { separate(); CHECK(fused(nullptr)); compare(); }
#ifdef G4_A8_GRAPH_TESTS
                        // Capture without a ticket clear inside the graph. Replay the same graph twice on an
                        // explicit stream; the reference keeps advancing from the preceding launches as well.
                        hipStream_t stream = nullptr; hipGraph_t graph = nullptr; hipGraphExec_t exec = nullptr;
                        CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess);
                        CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal) == hipSuccess);
                        CHECK(fused(stream));
                        CHECK(hipStreamEndCapture(stream, &graph) == hipSuccess);
                        CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == hipSuccess);
                        for (unsigned repeat = 0; repeat < 2; ++repeat) {
                            separate();
                            CHECK(hipGraphLaunch(exec, stream) == hipSuccess);
                            synchronize(stream); compare();
                        }
                        CHECK(hipGraphExecDestroy(exec) == hipSuccess);
                        CHECK(hipGraphDestroy(graph) == hipSuccess);
                        CHECK(hipStreamDestroy(stream) == hipSuccess);
#endif
                    }
                    // Unsupported rows/width/format must enqueue nothing (in particular no ticket increment).
                    CHECK(!a8_ffn_ple_gate(db.as<float>(), norm_weight, r[1].as<float>(), gate, dp.as<float>(),
                                           outputs + 7U, 5, out[1].as<float>(), in[1], flag[1].as<unsigned>(),
                                           ticket.as<unsigned>(), nullptr));
                    auto unsupported = gate; unsupported.inputs = 256;
                    CHECK(!a8_ffn_ple_gate(db.as<float>(), norm_weight, r[1].as<float>(), unsupported, dp.as<float>(),
                                           outputs + 7U, n, out[1].as<float>(), in[1], flag[1].as<unsigned>(),
                                           ticket.as<unsigned>(), nullptr));
                    unsupported = gate; unsupported.format = WeightFormat::bf16;
                    CHECK(!a8_ffn_ple_gate(db.as<float>(), norm_weight, r[1].as<float>(), unsupported, dp.as<float>(),
                                           outputs + 7U, n, out[1].as<float>(), in[1], flag[1].as<unsigned>(),
                                           ticket.as<unsigned>(), nullptr));
                }
}

template <class Stream>
ForwardResult ingest(Target& target, Stream& stream, unsigned n, unsigned token, bool permute = false) {
    auto tx = stream.begin();
    std::vector<RowPlan> plans;
    for (unsigned r = 0; r < n; ++r)
        plans.push_back({13, 7, stream.frontier() + r, (token + r) % 37, tx.get(), r, true});
    if (permute)
        std::reverse(plans.begin(), plans.end());
    const auto out = target.forward_rows(plans);
    tx->commit(n);
    return out;
}
void compare(const ForwardResult& a, const ForwardResult& b) {
    CHECK(a.rows == b.rows);
    bytes(a.logits, b.logits);
    bytes(a.taps, b.taps);
}
void ffn_ple_target_tests() {
    // The existing CTest fallback variants intentionally disable admission. Some kernel controls
    // are process-cached, so do not override them after the operator tests have initialized them.
    if (!ffn_ple_admitted() || !enabled("STRIX_G4_FUSION") || !enabled("STRIX_G4_NORM_GROUP") ||
        !enabled("STRIX_G4_STORE_FUSION") || !enabled("STRIX_G4_SHARED_WEIGHTS")) {
        std::cout << "FFN/PLE admitted target gate skipped: explicit fallback schedule\n";
        return;
    }
    const ScopedEnv fuse_env("STRIX_G4_A8_FFN_PLE_FUSE", nullptr), graphs_env("STRIX_G4_GRAPHS", nullptr);
    // Six is the minimum layer count that exports a real tap (layer 5); it also includes
    // sliding and global attention. Only hidden grows; FFN and PLE widths remain 256.
    const auto model = fixture::tiny(6, true, 2560);
    const auto& c = model.config;
    CHECK(c.layer.front().attention == AttentionKind::sliding);
    CHECK(c.layer.back().attention == AttentionKind::global);
    fixture::Tables tables(64);
    const auto weights = bind_host_weights(model);
    for (const auto& layer : weights.layer)
        CHECK(a8::quantized(layer.ple_gate.format));
    for (const bool graphs : {false, true}) {
        DeviceOptions options;
        options.identity = a8::arithmetic_id;
        options.graphs = graphs;
        options.a8_ffn_ple_fuse = true;
        DeviceTarget fused(weights, tables.view(), options);
        options.a8_ffn_ple_fuse = false;
        DeviceTarget separate(weights, tables.view(), options);
        CHECK(fused.stats().identity == a8::arithmetic_id);
        CHECK(separate.stats().identity == a8::arithmetic_id);
        CHECK(fused.stats().a8_ffn_ple_fuse);
        CHECK(!separate.stats().a8_ffn_ple_fuse);
        CHECK(fused.stats().fusion && fused.stats().norm_group && fused.stats().store_fusion &&
              fused.stats().shared_weights);
        for (unsigned n = 1; n <= 4; ++n) {
            DeviceStream on(c, 64), off(c, 64);
            // Keep the same targets, row count and stride bucket: one capture, then three
            // cached replays, with changing tokens and growing committed KV on every pass.
            for (unsigned step = 0; step < 4; ++step) {
                const auto launches = fused.stats().a8_ffn_ple_launches;
                auto& counters = capture_guard_state();
                const auto captures = counters.target_captures.load(), replays = counters.target_replays.load();
                const auto out = ingest(fused, on, n, 3 + step * 5, true),
                           expected = ingest(separate, off, n, 3 + step * 5, true);
                compare(out, expected);
                CHECK(out.logits.size() == std::size_t(n) * c.vocab);
                CHECK(out.taps.size() == std::size_t(n) * 5U * c.hidden);
                CHECK(std::any_of(out.taps.begin(), out.taps.begin() + c.hidden, [](float x) { return x != 0.0F; }));
                CHECK(on.digest() == off.digest());
                CHECK(fused.stats().graphs == graphs && separate.stats().graphs == graphs);
                CHECK(fused.stats().a8_ffn_ple_launches == launches + (graphs && step ? 0U : c.layers));
                CHECK(separate.stats().a8_ffn_ple_launches == 0);
                CHECK(counters.target_captures.load() == captures + (graphs && !step ? 2U : 0U));
                CHECK(counters.target_replays.load() == replays + (graphs && step ? 2U : 0U));
            }
        }
        if (!graphs) {
            // An environment ON cannot override programmatic OFF. Restore it even on failure.
            const ScopedEnv force_env("STRIX_G4_A8_FFN_PLE_FUSE", "1");
            DeviceTarget forced_off(weights, tables.view(), options);
            CHECK(!forced_off.stats().a8_ffn_ple_fuse);
            DeviceStream reference(c, 64), off(c, 64);
            compare(ingest(separate, reference, 4, 19, true), ingest(forced_off, off, 4, 19, true));
            CHECK(reference.digest() == off.digest());
            CHECK(forced_off.stats().a8_ffn_ple_launches == 0);
        }
    }
}
void graph_tests(unsigned count) {
    const auto model = fixture::tiny(count, true);
    fixture::Tables tables;
    const auto weights = bind_host_weights(model);
    HostTarget host(weights, tables.view(), 2, a8::host_recipe());
    DeviceOptions options;
    options.identity = a8::arithmetic_id;
    DeviceTarget device(weights, tables.view(), options);
    CHECK(device.stats().identity == a8::arithmetic_id);
    DeviceStream batch(model.config, 1024), serial(model.config, 1024);
    HostStream reference(model.config);
    for (unsigned n = 1; n <= 8; ++n) {
        const auto out = ingest(device, batch, n, 2 + n, true), expected = ingest(host, reference, n, 2 + n, true);
        compare(out, expected);
        const auto selected = device.selected();
        for (unsigned r = 0; r < n; ++r) {
            const auto* logits = out.logits.data() + std::size_t(r) * model.config.vocab;
            unsigned best = 0;
            for (unsigned i = 1; i < model.config.vocab; ++i)
                if (better(logits[i], i, logits[best], best))
                    best = i;
            CHECK(selected[r].id == best);
            const auto single = ingest(device, serial, 1, 2 + n + r);
            CHECK(std::memcmp(single.logits.data(), out.logits.data() + std::size_t(r) * model.config.vocab,
                              single.logits.size() * 4) == 0);
            CHECK(std::memcmp(single.taps.data(), out.taps.data() + std::size_t(r) * 5 * model.config.hidden,
                              single.taps.size() * 4) == 0);
        }
        CHECK(batch.digest() == serial.digest());
        CHECK(batch.digest() == reference.digest());
    }
    DeviceOptions wrong;
    wrong.identity = "g4-a8-v0";
    refuses([&] { DeviceTarget invalid(weights, tables.view(), wrong); });
}
} // namespace
int main() {
    try {
        require_device();
        kernel_tests();
        ffn_ple_tests();
        graph_tests(1);
        graph_tests(6);
        ffn_ple_target_tests();
        std::cout << "g4-a8-v1 device: quantization, Q8_0/Q4_K/Q6_K epilogues, FFN/PLE C1..C4 bytes/replay and admitted target gate, graph R=1..8 host/serial bytes passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
