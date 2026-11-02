// MTP assistant drafter device tests: DeviceDrafter == HostDrafter bit for bit (tokens, probabilities, margins) for
// random weights (BF16 and Q8_0), positions around the 64-chunk/512-window edges, k = 1..7, HIP graphs on and off,
// target sliding KV through the engine's tagged 512-slot ring; r5b kernels (GQA-tiled attention, bitonic top-k,
// skipped statistics) == pre-r5b kernels (STRIX_G4_MTP_KERNELS=legacy) bit for bit; optional HF fixture;
// --bench for ms/step: legacy (old) vs r5b (new).
// usage: strix-gemma4-eagle-mtp-device-tests [fixture_dir assistant_dir]
//        strix-gemma4-eagle-mtp-device-tests --bench <assistant_dir|random> [contexts...]
#include "mtp_common.hpp"
#include "gemma4/eagle/mtp/device.hpp"

#include <chrono>
#include <cstring>

using namespace mtp_test;

namespace {
struct Dev {
    gpu::Buffer b;
    template <class T> explicit Dev(const std::vector<T>& v) : b(v.size() * sizeof(T)) { b.upload(v.data(), b.size()); }
    template <class T> const T* as() const { return b.as<T>(); }
};
bool same(float a, float b) { return bit_cast<std::uint32_t>(a) == bit_cast<std::uint32_t>(b); }
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
void set_graphs(bool on) { set_env("STRIX_G4_MTP_GRAPHS", on ? "1" : "0"); }
void set_legacy(bool on) { set_env("STRIX_G4_MTP_KERNELS", on ? "legacy" : "r5b"); }

// Device KV views of a TargetKv: full arrays for the full layer, the engine ring for the sliding layer at p.
struct DeviceKv {
    Dev kf, vf;
    gpu::Buffer rk, rv, tags;
    explicit DeviceKv(const TargetKv& kv) : kf(kv.kf), vf(kv.vf), rk(512ULL * 512 * 4), rv(512ULL * 512 * 4), tags(512 * 8) {}
    std::pair<mt::KvLayer, mt::KvLayer> at(TargetKv& kv, std::uint64_t p) {
        const auto host = kv.ring(p);
        rk.upload(host.k, rk.size()); rv.upload(host.v, rv.size()); tags.upload(host.tags, tags.size());
        return {{rk.as<float>(), rv.as<float>(), tags.as<std::uint64_t>(), 512, 512, false},
                {kf.as<float>(), vf.as<float>(), nullptr, 1024, kv.T, true}};
    }
};

void device_vs_host(const Table& table, const gpu::Buffer& dtable, WeightFormat format, const char* name) {
    const auto w = mt::Weights::random(mt::Config{}, 21, format);
    Rope rope(1024);
    TargetKv kv(800, 6);
    std::vector<float> hidden(800ULL * 2560);
    Rng r{8};
    for (auto& x : hidden) x = 2 * r.uniform();
    mt::HostDrafter host(w, rope.view(), table.view());
    const LinearView dview{dtable.as<std::uint8_t>(), strix::gemma4::vocab, strix::gemma4::hidden, 2ULL * strix::gemma4::hidden, WeightFormat::bf16};
    auto model = std::make_shared<mt::DeviceModel>(w, rope.view(), dview);
    set_graphs(true);
    mt::DeviceDrafter graphed(model);
    set_legacy(true);
    mt::DeviceDrafter legacy(model); // pre-r5b kernels, graphs on
    set_legacy(false);
    mt::DeviceDrafter quiet(model);  // r5b, no statistics (the plain propose_window path)
    quiet.set_statistics(0);
    set_graphs(false);
    mt::DeviceDrafter plain(model);
    DeviceKv dkv(kv);
    std::size_t depths = 0;
    std::uint32_t k = 1;
    for (const std::uint64_t p : {1ULL, 2ULL, 63ULL, 64ULL, 65ULL, 511ULL, 512ULL, 513ULL, 600ULL, 799ULL, 700ULL, 799ULL}) {
        const auto anchor = r.below(strix::gemma4::vocab);
        const auto hrow = std::span<const float>(hidden).subspan((p - 1) * 2560, 2560);
        mt::Inputs hin{anchor, p, hrow, kv.ring(p), kv.full()};
        const auto h = host.draft(hin, k);
        const auto views = dkv.at(kv, p);
        mt::Inputs din{anchor, p, hrow, views.first, views.second};
        for (auto* d : {&graphed, &plain, &legacy}) {
            const auto x = d->draft(din, k);
            for (std::uint32_t i = 0; i < k; ++i)
                EAGLE_CHECK(x.tokens[i] == h.tokens[i] && same(x.prob[i], h.prob[i]) && same(x.margin[i], h.margin[i]),
                            "%s %s p %llu depth %u: device %u/%a host %u/%a", name,
                            d == &graphed ? "graph" : d == &plain ? "plain" : "legacy", (unsigned long long)p, i + 1,
                            x.tokens[i], double(x.prob[i]), h.tokens[i], double(h.prob[i]));
            EAGLE_CHECK(d->timings().launches == 44ULL * k, "%s launches %llu for k %u", name,
                        (unsigned long long)d->timings().launches, k);
        }
        const auto y = quiet.draft(din, k);
        for (std::uint32_t i = 0; i < k; ++i)
            EAGLE_CHECK(y.tokens[i] == h.tokens[i] && y.prob[i] == 0.0F && y.margin[i] == 0.0F,
                        "%s no-statistics p %llu depth %u: device %u host %u", name, (unsigned long long)p, i + 1, y.tokens[i], h.tokens[i]);
        depths += k;
        k = k % 7 + 1;
    }
    std::printf("device == host (%s): %zu depths bit-identical, r5b graphs on and off, legacy kernels, no-statistics "
                "tokens\n", name, depths);
    // extend(k) continues the device chain ([depth, k) segment graphs): bit-identical to the host chain from scratch
    // (= device draft(in, k) from scratch), graphs on and off, for several chunkings, launches = 44 x new depths.
    std::size_t extended = 0;
    for (const std::uint64_t p : {1ULL, 65ULL, 513ULL, 799ULL}) {
        const auto anchor = r.below(strix::gemma4::vocab);
        const auto hrow = std::span<const float>(hidden).subspan((p - 1) * 2560, 2560);
        const auto h = host.draft({anchor, p, hrow, kv.ring(p), kv.full()}, 7);
        const auto views = dkv.at(kv, p);
        const mt::Inputs din{anchor, p, hrow, views.first, views.second};
        for (auto* d : {&graphed, &plain}) {
            for (const auto& chain : {std::vector<std::uint32_t>{1, 2, 4, 7}, std::vector<std::uint32_t>{2, 3, 5}, std::vector<std::uint32_t>{3, 7}}) {
                auto x = d->draft(din, chain[0]);
                for (std::size_t c = 1; c < chain.size(); ++c) {
                    x = d->extend(chain[c]);
                    EAGLE_CHECK(x.depth == chain[c] && d->timings().launches == 44ULL * (chain[c] - chain[c - 1]),
                                "%s extend %u->%u depth %u launches %llu", name, chain[c - 1], chain[c], x.depth,
                                (unsigned long long)d->timings().launches);
                    for (std::uint32_t i = 0; i < chain[c]; ++i)
                        EAGLE_CHECK(x.tokens[i] == h.tokens[i] && same(x.prob[i], h.prob[i]) && same(x.margin[i], h.margin[i]),
                                    "%s %s extend %u->%u p %llu depth %u: device %u/%a host %u/%a", name,
                                    d == &graphed ? "graph" : "plain", chain[c - 1], chain[c], (unsigned long long)p, i + 1,
                                    x.tokens[i], double(x.prob[i]), h.tokens[i], double(h.prob[i]));
                    ++extended;
                }
                EAGLE_CHECK(throws([&] { (void)d->extend(chain.back()); }), "%s extend to the same depth refuses", name);
            }
        }
    }
    std::printf("device extend == draft from scratch (%s): %zu extensions bit-identical, graphs on and off\n", name, extended);
}

void device_fixture(const std::filesystem::path& dir, const std::filesystem::path& model_dir, const gpu::Buffer& dtable) {
    Fixture f;
    if (!Fixture::load(dir, f) || !std::filesystem::exists(model_dir / "model.safetensors")) {
        std::printf("SKIP device fixture\n");
        return;
    }
    auto w = mt::Weights::load(model_dir);
    w.config.sliding_keys = f.sliding_keys;
    w.config.top_k = 32; // HF reference selects config.centroid_intermediate_top_k = 32
    Rope rope(1024);
    const LinearView dview{dtable.as<std::uint8_t>(), strix::gemma4::vocab, strix::gemma4::hidden, 2ULL * strix::gemma4::hidden, WeightFormat::bf16};
    auto model = std::make_shared<mt::DeviceModel>(w, rope.view(), dview);
    mt::DeviceDrafter d(model);
    Dev ks(f.kv.ks), vs(f.kv.vs), kf(f.kv.kf), vf(f.kv.vf);
    const mt::KvLayer sl{ks.as<float>(), vs.as<float>(), nullptr, 512, f.kv.T, true}, fl{kf.as<float>(), vf.as<float>(), nullptr, 1024, f.kv.T, true};
    std::size_t same_depths = 0, total = 0;
    for (std::size_t ci = 0; ci < f.cases.size(); ++ci) {
        const auto& c = f.cases[ci];
        for (std::size_t i = 0; i < c.tokens.size(); ++i) {
            const auto h = std::span<const float>(f.step_hidden).subspan((ci * f.K + i) * 2560, 2560);
            const auto out = d.draft({c.inputs[i], c.p, h, sl, fl}, 1);
            ++total;
            if (out.tokens[0] == c.tokens[i]) { ++same_depths; continue; }
            EAGLE_CHECK(c.margins[i] < 1e-3, "device vs HF p %llu depth %zu", (unsigned long long)c.p, i + 1);
        }
    }
    std::printf("device HF fixture: %zu/%zu depths identical\n", same_depths, total);
}

void bench(const std::string& source, std::vector<std::uint32_t> contexts, const gpu::Buffer& dtable) {
    using Clock = std::chrono::steady_clock;
    if (contexts.empty()) contexts = {512, 4096, 32768};
    std::uint32_t maxp = 0;
    for (const auto c : contexts) maxp = std::max(maxp, c);
    for (const auto format : {WeightFormat::bf16, WeightFormat::q8_0})
    for (const std::uint32_t topk : {32U, 64U}) {
        auto w = source == "random" ? mt::Weights::random(mt::Config{}, 1, format) : mt::Weights::load(source, format);
        w.config.top_k = topk;
        Rope rope(maxp + 8);
        const LinearView dview{dtable.as<std::uint8_t>(), strix::gemma4::vocab, strix::gemma4::hidden, 2ULL * strix::gemma4::hidden, WeightFormat::bf16};
        auto model = std::make_shared<mt::DeviceModel>(w, rope.view(), dview);
        TargetKv kv(maxp, 3);
        DeviceKv dkv(kv);
        std::vector<float> hidden(2560, 0.7F);
        for (const bool graphs : {true, false})
        for (const bool old : {true, false}) {
            set_graphs(graphs);
            set_legacy(old);
            mt::DeviceDrafter d(model);
            if (!old) d.set_statistics(0); // production propose_window path (legacy always computes prob + margin)
            std::printf("[%s %s top%u graphs=%d kernels=%s] weight bytes/step %.2f MB\n", source == "random" ? "random" : "checkpoint",
                        format == WeightFormat::q8_0 ? "q8_0" : "bf16", topk, graphs ? 1 : 0, old ? "legacy" : "r5b",
                        double(model->step_weight_bytes()) / 1e6);
            for (const auto p : contexts) {
                const auto views = dkv.at(kv, p);
                mt::Inputs in{1234, p, hidden, views.first, views.second};
                double ms[8] = {}, wall[8] = {};
                for (std::uint32_t k = 1; k <= mt::max_depth; ++k) {
                    for (int i = 0; i < 3; ++i) (void)d.draft(in, k);
                    for (int i = 0; i < 15; ++i) {
                        const auto a = Clock::now();
                        (void)d.draft(in, k);
                        wall[k] += std::chrono::duration<double, std::milli>(Clock::now() - a).count() / 15;
                        ms[k] += d.timings().draft_ms / 15;
                    }
                }
                std::printf("  %s ctx %6u: GPU k1 %.3f k2 %.3f k4 %.3f k7 %.3f ms -> %.3f ms/depth; wall k1 %.3f k7 %.3f ms\n", old ? "old" : "new", p, ms[1],
                            ms[2], ms[4], ms[7], (ms[7] - ms[1]) / 6, wall[1], wall[7]);
            }
        }
        set_legacy(false);
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        gpu::require_device();
        const Table table;
        gpu::Buffer dtable(table.bytes.size());
        dtable.upload(table.bytes.data(), table.bytes.size());
        if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
            std::vector<std::uint32_t> contexts;
            for (int i = 3; i < argc; ++i) contexts.push_back(std::uint32_t(std::stoul(argv[i])));
            bench(argc > 2 ? argv[2] : "random", contexts, dtable);
        } else {
            device_vs_host(table, dtable, WeightFormat::bf16, "random bf16");
            device_vs_host(table, dtable, WeightFormat::q8_0, "random q8_0");
            if (argc > 2) device_fixture(argv[1], argv[2], dtable);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(eagle_test::failures ? "FAILED %d checks\n" : "ALL OK\n", eagle_test::failures);
    return eagle_test::failures ? 1 : 0;
}
