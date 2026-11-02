// DSpark drafter fixtures: every device kernel against the host reference bitwise on synthetic data,
// then (with STRIX_DS41_Q2_GGUF and STRIX_DS41_DSPARK_GGUF) the whole drafter on the real weights:
// main rows, ring seeding with wrap-around, draft steps (tokens, confidence, logits, every stage's
// streams), appends after acceptance, and a single-run smoke timing of a draft step.
// STRIX_DS41_DSPARK_DUMP=<dir> writes the inputs and outputs of the real draft steps as raw files.

#include "strix/gguf_reader.hpp"
#include "strix/hip_ds41_dspark.hpp"
#include "ds41_dspark_model.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/ds41_dspark_vocab.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <exception>
#include <thread>
#include <vector>

namespace {

using namespace strix::ds41::dspark;
using namespace ds41_dspark_test;
using strix::ds41::ExpertTensors;
using strix::ds41::WeightFormat;
int failures = 0;
void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}
void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}

template <class T>
struct Device {
    T* data = nullptr;
    std::size_t count = 0;
    explicit Device(std::size_t n) : count(n) { check(hipMalloc(reinterpret_cast<void**>(&data), std::max<std::size_t>(n, 1U) * sizeof(T)), "hipMalloc"); }
    Device(const std::vector<T>& host) : Device(host.size()) { upload(host); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { (void)hipFree(data); }
    void upload(const std::vector<T>& host) { check(hipMemcpy(data, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<T> download() const {
        check(hipDeviceSynchronize(), "sync");
        std::vector<T> out(count);
        check(hipMemcpy(out.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return out;
    }
};

template <class T>
std::size_t mismatches(const std::vector<T>& a, const std::vector<T>& b) {
    std::size_t bad = a.size() != b.size();
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if constexpr (std::is_same_v<T, float>) bad += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]);
        else bad += a[i] != b[i];
    }
    return bad;
}
void report(const std::string& label, std::size_t bad) {
    expect(bad == 0U, label + ": " + std::to_string(bad) + " values differ");
    std::cout << label << ": " << (bad == 0U ? "bit-exact" : "MISMATCH") << '\n';
}

std::uint16_t f16_bits(float x) {
    // Round-to-nearest F16 of a value well inside the normal range (synthetic weights only).
    const auto b = std::bit_cast<std::uint32_t>(x);
    const std::uint32_t sign = (b >> 16U) & 0x8000U;
    const int e = static_cast<int>((b >> 23U) & 0xffU) - 127;
    if (e < -14) return static_cast<std::uint16_t>(sign);
    const std::uint32_t m = (b & 0x7fffffU) >> 13U;
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e + 15) << 10U) | m);
}
std::vector<std::uint16_t> f16_values(std::size_t n, float sigma, unsigned seed) {
    const auto v = normal_values(n, sigma, seed);
    std::vector<std::uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = f16_bits(v[i]);
    return out;
}

// ---- synthetic kernels ----

void test_synthetic() {
    const std::uint32_t rows = block_size;
    // RMSNorm at the drafter's widths.
    for (const std::uint32_t width : {head_dim, q_rank, hidden}) {
        const auto x = normal_values(static_cast<std::size_t>(rows) * width, 2.0F, width);
        const auto w = normal_values(width, 1.0F, width + 1U);
        Device<float> dx(x), dw(w), dy(x.size());
        enqueue_rms_norm(dx.data, rows, width, dw.data, dy.data, nullptr);
        report("rms norm width " + std::to_string(width), mismatches(dy.download(), rms_norm(x, rows, w)));
    }
    // Main hidden rows.
    {
        const auto a = normal_values(static_cast<std::size_t>(rows) * stream_width, 1.0F, 1), b = normal_values(a.size(), 1.0F, 2),
                   c = normal_values(a.size(), 1.0F, 3);
        Device<float> da(a), db(b), dc(c), out(static_cast<std::size_t>(rows) * main_width);
        enqueue_main_hidden(da.data, db.data, dc.data, rows, out.data, nullptr);
        std::vector<float> ref;
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto row = main_hidden_row(std::span<const float>(a).subspan(r * stream_width, stream_width),
                                             std::span<const float>(b).subspan(r * stream_width, stream_width),
                                             std::span<const float>(c).subspan(r * stream_width, stream_width));
            ref.insert(ref.end(), row.begin(), row.end());
        }
        report("main hidden rows", mismatches(out.download(), ref));
    }
    // Hyper-connections: mixes (dots + Sinkhorn), pre, post.
    {
        const auto streams = normal_values(static_cast<std::size_t>(rows) * stream_width, 1.5F, 7);
        const auto fn = f16_values(static_cast<std::size_t>(mix_count) * stream_width, 0.02F, 8);
        const auto scale = normal_values(3, 0.5F, 9), base = normal_values(mix_count, 1.0F, 10);
        Device<float> ds(streams), dscale(scale), dbase(base), scratch(rows * (mix_count + 1U)), mixes(rows * mix_count);
        Device<std::uint16_t> dfn(fn);
        enqueue_hc_mixes(ds.data, rows, dfn.data, dscale.data, dbase.data, scratch.data, mixes.data, nullptr);
        const auto ref = hc_mixes(streams, rows, F16Matrix{fn, stream_width, mix_count}, scale, base);
        report("hc mixes (dots, rstd, Sinkhorn)", mismatches(mixes.download(), ref));
        Device<float> pre(static_cast<std::size_t>(rows) * hidden);
        enqueue_hc_pre(ds.data, rows, mixes.data, mix_count, pre.data, nullptr);
        report("hc pre", mismatches(pre.download(), hc_pre(streams, rows, ref, mix_count)));
        const auto x = normal_values(static_cast<std::size_t>(rows) * hidden, 1.0F, 11);
        Device<float> dx(x), post(streams.size());
        enqueue_hc_post(dx.data, ds.data, rows, mixes.data, post.data, nullptr);
        report("hc post", mismatches(post.download(), hc_post(x, streams, rows, ref)));
    }
    // RoPE (forward on q-shaped rows, inverse), FP8 entries into a ring with wrap-around.
    {
        const std::uint64_t first = 126;
        auto q = normal_values(static_cast<std::size_t>(rows) * q_width, 1.0F, 12);
        std::vector<float> table;
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto row = rope_row(first + r);
            table.insert(table.end(), row.c.begin(), row.c.end());
            table.insert(table.end(), row.s.begin(), row.s.end());
        }
        Device<float> dq(q), dt(table);
        enqueue_rope(dq.data, rows, q_width, head_dim, dt.data, false, nullptr);
        auto ref = q;
        apply_rope(ref, rows, head_dim, first, false);
        report("rope forward", mismatches(dq.download(), ref));
        enqueue_rope(dq.data, rows, q_width, head_dim, dt.data, true, nullptr);
        apply_rope(ref, rows, head_dim, first, true);
        report("rope inverse", mismatches(dq.download(), ref));
        const auto kv = normal_values(static_cast<std::size_t>(rows) * head_dim, 4.0F, 13);
        Device<float> dkv(kv);
        Device<std::uint8_t> ring(static_cast<std::size_t>(window) * entry_bytes);
        check(hipMemset(ring.data, 0, ring.count), "memset");
        enqueue_fp8_entries(dkv.data, rows, ring.data, window, first, nullptr);
        std::vector<std::uint8_t> expected(ring.count, 0U);
        for (std::uint32_t r = 0; r < rows; ++r)
            fp8_entry(std::span<const float>(kv).subspan(r * head_dim, head_dim),
                      std::span<std::uint8_t>(expected).subspan(((first + r) % window) * entry_bytes, entry_bytes));
        report("fp8 ring entries (wrap-around)", mismatches(ring.download(), expected));
    }
    // Block attention (before the inverse RoPE) for an empty, a partial and a full wrapped ring.
    for (const std::uint64_t first : {0ULL, 9ULL, 300ULL}) {
        Ring ring;
        const auto values = normal_values(static_cast<std::size_t>(window) * head_dim, 1.0F, 20U + static_cast<unsigned>(first));
        for (std::uint32_t s = 0; s < window; ++s)
            fp8_entry(std::span<const float>(values).subspan(s * head_dim, head_dim),
                      std::span<std::uint8_t>(ring.entries).subspan(s * entry_bytes, entry_bytes));
        ring.next = first;
        std::vector<std::uint8_t> block(static_cast<std::size_t>(rows) * entry_bytes);
        const auto bv = normal_values(static_cast<std::size_t>(rows) * head_dim, 1.0F, 30);
        for (std::uint32_t r = 0; r < rows; ++r)
            fp8_entry(std::span<const float>(bv).subspan(r * head_dim, head_dim),
                      std::span<std::uint8_t>(block).subspan(r * entry_bytes, entry_bytes));
        const auto q = normal_values(static_cast<std::size_t>(rows) * q_width, 2.0F, 31), sinks = normal_values(heads, 1.0F, 32);
        Device<float> dq(q), ds(sinks), out(q.size());
        Device<std::uint8_t> dr(ring.entries), db(block);
        const std::uint64_t oldest = first > window ? first - window : 0U;
        Device<float> scratch(static_cast<std::size_t>(window + 8U) * head_dim);
        enqueue_block_attention(dq.data, rows, dr.data, static_cast<std::uint32_t>(first - oldest), oldest, db.data, ds.data, scratch.data,
                                out.data, nullptr);
        auto got = out.download();
        apply_rope(got, rows, head_dim, first, true);
        report("block attention, ring next " + std::to_string(first), mismatches(got, block_attention(q, rows, ring, block, sinks, first)));
    }
}

void compare_drafts(const Draft& dev, const Draft& host, const std::string& label) {
    std::size_t stage_bad = 0;
    for (std::uint32_t s = 0; s < stages; ++s) stage_bad += mismatches(dev.stage_out[s], host.stage_out[s]);
    const std::vector<std::uint32_t> dt(dev.tokens.begin(), dev.tokens.end()), ht(host.tokens.begin(), host.tokens.end());
    const std::vector<float> dc(dev.confidence.begin(), dev.confidence.end()), hc_(host.confidence.begin(), host.confidence.end());
    report(label + ": stage streams", stage_bad);
    report(label + ": head input, Markov embeddings, logits", mismatches(dev.head_in, host.head_in) + mismatches(dev.markov, host.markov) +
                                                                  mismatches(dev.logits, host.logits));
    report(label + ": drafted tokens (pruned Markov argmax on the device, every bias on the host) and confidence",
           mismatches(dt, ht) + mismatches(dc, hc_));
    std::cout << label << ": vocabulary " << dev.candidates[0] << "; tokens";
    for (const auto t : dev.tokens) std::cout << ' ' << t;
    std::cout << ", confidence";
    for (const auto c : dev.confidence) std::cout << ' ' << c;
    std::cout << '\n';
}

void test_real(const Files& f) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto model = load_model(f);
    const auto& w = model->w;
    DeviceDrafter device(w);
    std::cout << "real weights loaded and uploaded in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
    const char* dump_env = std::getenv("STRIX_DS41_DSPARK_DUMP");
    const std::string dir = dump_env ? dump_env : "";

    // Main rows: 130 committed rows of synthetic target hidden states (positions 0..129).
    const std::uint32_t seed_rows = 130;
    const auto hidden_rows = normal_values(static_cast<std::size_t>(seed_rows) * main_width, 1.0F, 77);
    dump(dir, "main_hidden_seed.f32", hidden_rows.data(), hidden_rows.size() * 4U);
    const auto host_main = main_rows(w, hidden_rows, seed_rows);
    std::vector<float> dev_main;
    for (std::uint32_t r0 = 0; r0 < seed_rows; r0 += 64U) {
        const auto n = std::min(64U, seed_rows - r0);
        const auto part = device.main_rows(std::span<const float>(hidden_rows).subspan(static_cast<std::size_t>(r0) * main_width,
                                                                                      static_cast<std::size_t>(n) * main_width), n);
        dev_main.insert(dev_main.end(), part.begin(), part.end());
    }
    report("main_proj + main_norm, 130 rows", mismatches(dev_main, host_main));

    // Seed the rings: 128 rows, then 2 (wrap-around), on both sides.
    State state;
    append(w, state, std::span<const float>(host_main).first(static_cast<std::size_t>(128) * hidden), 128);
    append(w, state, std::span<const float>(host_main).subspan(static_cast<std::size_t>(128) * hidden), 2);
    device.append(std::span<const float>(host_main).first(static_cast<std::size_t>(128) * hidden), 128);
    device.append(std::span<const float>(host_main).subspan(static_cast<std::size_t>(128) * hidden), 2);
    std::size_t ring_bad = 0;
    for (std::uint32_t s = 0; s < stages; ++s) ring_bad += mismatches(device.ring(s), state.rings[s].entries);
    report("ring seeding, 130 positions over 128 slots, 3 stages", ring_bad);

    // Draft step 1 from token 1000 at position 130.
    const auto host1 = draft(w, state, 1000U, true);
    DraftTimes times;
    const auto dev1 = device.draft(1000U, true, &times);
    compare_drafts(dev1, host1, "draft at position 130");
    dump(dir, "draft1_logits.f32", host1.logits.data(), host1.logits.size() * 4U);
    dump(dir, "draft1_tokens.u32", host1.tokens.data(), host1.tokens.size() * 4U);
    dump(dir, "draft1_stage0.f32", host1.stage_out[0].data(), host1.stage_out[0].size() * 4U);
    dump(dir, "draft1_stage2.f32", host1.stage_out[2].data(), host1.stage_out[2].size() * 4U);
    dump(dir, "draft1_confidence.f32", host1.confidence.data(), host1.confidence.size() * 4U);

    // Verify accepts two drafts: rows 0..2 of the window (3 main rows) are committed, appended on
    // the device directly from hidden states; the host appends their main rows. Next token at 133.
    const auto accepted_hidden = normal_values(static_cast<std::size_t>(3) * main_width, 1.0F, 78);
    dump(dir, "main_hidden_accepted.f32", accepted_hidden.data(), accepted_hidden.size() * 4U);
    append(w, state, main_rows(w, accepted_hidden, 3), 3);
    {
        Device<float> rows_on_device(accepted_hidden);
        device.append_hidden_device(rows_on_device.data, 3);
    }
    ring_bad = 0;
    for (std::uint32_t s = 0; s < stages; ++s) ring_bad += mismatches(device.ring(s), state.rings[s].entries);
    report("append after a verify (3 committed rows from hidden states in device memory)", ring_bad);
    const auto host2 = draft(w, state, host1.tokens[3], true);
    const auto dev2 = device.draft(host1.tokens[3], true);
    compare_drafts(dev2, host2, "draft at position 133");
    dump(dir, "draft2_logits.f32", host2.logits.data(), host2.logits.size() * 4U);
    dump(dir, "draft2_tokens.u32", host2.tokens.data(), host2.tokens.size() * 4U);

    // A short context (ring not yet full): the drafter reset, 7 rows appended as 3 + 4.
    {
        device.reset();
        State s7;
        append(w, s7, std::span<const float>(host_main).first(static_cast<std::size_t>(7) * hidden), 7);
        device.append(std::span<const float>(host_main).first(static_cast<std::size_t>(3) * hidden), 3);
        device.append(std::span<const float>(host_main).subspan(static_cast<std::size_t>(3) * hidden, static_cast<std::size_t>(4) * hidden), 4);
        std::size_t bad = 0;
        for (std::uint32_t s = 0; s < stages; ++s) bad += mismatches(device.ring(s), s7.rings[s].entries);
        report("7 rows appended as 3 + 4 equal 7 at once (append order does not matter)", bad);
        compare_drafts(device.draft(42U, true), draft(w, s7, 42U, true), "draft at position 7");
        // Reduced draft vocabulary: the first 32,768 ids plus three context ids.
        Vocab v;
        v.core = 32768;
        v.extra = {40000U, 100000U, 128000U};
        device.set_vocabulary(v);
        compare_drafts(device.draft(42U, true), draft(w, s7, 42U, true, v), "draft at position 7, vocabulary 32768 + 3");
        // The real draft vocabulary (STRIX_DS41_DSPARK_VOCAB_DIR): automatic choice for a prompt made of
        // pack-es tokens (core32k + pack-es + context rows), then context rows added during generation.
        if (const char* vocab_dir = std::getenv("STRIX_DS41_DSPARK_VOCAB_DIR"); vocab_dir && *vocab_dir) {
            const auto es = read_vocabulary_file(std::string(vocab_dir) + "/pack-es16k.bin");
            const std::vector<std::uint32_t> prompt(es.begin(), es.begin() + 40);
            auto choice = choose_vocabulary(std::string("auto:") + vocab_dir, prompt);
            std::cout << "automatic draft vocabulary: " << choice.label << '\n';
            expect(choice.label.rfind("core32k+es", 0) == 0, "a prompt of Spanish pack tokens selects core32k + pack-es");
            device.set_vocabulary(choice.vocabulary);
            compare_drafts(device.draft(42U, true), draft(w, s7, 42U, true, choice.vocabulary), "draft at position 7, " + choice.label);
            const std::vector<std::uint32_t> context{120000U, 120001U, 5U, 120000U};
            const auto added = device.add_vocabulary(context);
            const auto added_host = add_context(choice.vocabulary, context);
            expect(added == added_host && added == 2U, "context rows: the two new ids are added once");
            compare_drafts(device.draft(43U, true), draft(w, s7, 43U, true, choice.vocabulary), "draft at position 7, + 2 context rows");
            const auto core48 = choose_vocabulary(std::string("auto:") + vocab_dir, std::vector<std::uint32_t>{1U, 2U, 3U});
            expect(core48.label.rfind("core48k", 0) == 0, "a prompt without a language pack selects core48k");
        }
        device.set_vocabulary(Vocab{});
    }

    // Smoke timing (single runs, warm): draft steps without keeping logits.
    for (int i = 0; i < 3; ++i) {
        DraftTimes t;
        (void)device.draft(1000U + static_cast<std::uint32_t>(i), false, &t);
        std::cout << "smoke draft step " << i << ": total " << t.total_ms << " ms (stages " << t.stages_ms[0] << ' ' << t.stages_ms[1] << ' '
                  << t.stages_ms[2] << " ms, head " << t.head_ms << " ms)\n";
    }
    const auto a0 = std::chrono::steady_clock::now();
    device.append_hidden(accepted_hidden, 3);
    std::cout << "smoke append of 3 rows from hidden states: "
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a0).count() << " ms\n";
}

// Head split over two ranks, emulated in one process (two drafters, two threads, the in-memory
// transport of the communicator): both ranks draft exactly the tokens of the host reference.
void test_two_ranks(const Files& f) {
    const auto model = load_model(f);
    const auto& w = model->w;
    const auto seed = normal_values(static_cast<std::size_t>(130) * main_width, 1.0F, 77);
    const auto accepted = normal_values(static_cast<std::size_t>(3) * main_width, 1.0F, 78);
    Vocab restricted;
    restricted.core = 0;
    for (std::uint32_t id = 0; id < 40000U; id += 2U) restricted.extra.push_back(id); // an arbitrary non-prefix set
    restricted.extra.push_back(1001U);
    State host;
    const auto host_main = main_rows(w, seed, 130);
    append(w, host, std::span<const float>(host_main).first(static_cast<std::size_t>(128) * hidden), 128);
    append(w, host, std::span<const float>(host_main).subspan(static_cast<std::size_t>(128) * hidden), 2);
    const auto h1 = draft(w, host, 1000U, false);
    append(w, host, main_rows(w, accepted, 3), 3);
    const auto h2 = draft(w, host, h1.tokens[3], false);
    const auto h3 = draft(w, host, h1.tokens[3], false, restricted);
    auto [ta, tb] = strix::dist::memory_pair();
    std::unique_ptr<strix::dist::Transport> transports[2] = {std::move(ta), std::move(tb)};
    Draft results[2][3];
    double head_ms[2] = {};
    std::exception_ptr errors[2];
    std::thread ranks[2];
    for (std::uint32_t r = 0; r < 2U; ++r)
        ranks[r] = std::thread([&, r] {
            try {
                strix::dist::Communicator comm({r, 2U, 1U, 88U}, std::move(transports[r]), 64U);
                std::uint64_t request = 0;
                DeviceDrafter d(w);
                d.set_rank(r, 2U, [&](void* stream, const void* local, void* remote, std::size_t bytes) {
                    comm.exchange(static_cast<hipStream_t>(stream), local, remote, bytes,
                                  strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::argmax), 0U, request++});
                });
                d.append_hidden(std::span<const float>(seed).first(static_cast<std::size_t>(128) * main_width), 128);
                d.append_hidden(std::span<const float>(seed).subspan(static_cast<std::size_t>(128) * main_width), 2);
                DraftTimes t;
                results[r][0] = d.draft(1000U, false, &t);
                head_ms[r] = t.head_ms;
                d.append_hidden(accepted, 3);
                results[r][1] = d.draft(h1.tokens[3], false);
                d.set_vocabulary(restricted);
                results[r][2] = d.draft(h1.tokens[3], false);
            } catch (...) { errors[r] = std::current_exception(); }
        });
    for (auto& t : ranks) t.join();
    for (auto& e : errors) if (e) std::rethrow_exception(e);
    const Draft* host_drafts[3] = {&h1, &h2, &h3};
    for (std::uint32_t r = 0; r < 2U; ++r)
        for (std::uint32_t k = 0; k < 3U; ++k) {
            const std::vector<std::uint32_t> dt(results[r][k].tokens.begin(), results[r][k].tokens.end()),
                ht(host_drafts[k]->tokens.begin(), host_drafts[k]->tokens.end());
            const std::vector<float> dc(results[r][k].confidence.begin(), results[r][k].confidence.end()),
                hc_(host_drafts[k]->confidence.begin(), host_drafts[k]->confidence.end());
            report("two ranks (emulated), rank " + std::to_string(r) + ", draft " + std::to_string(k) + ": tokens and confidence = one-rank host",
                   mismatches(dt, ht) + mismatches(dc, hc_));
        }
    std::cout << "two ranks (emulated): head " << head_ms[0] << " / " << head_ms[1] << " ms with the in-process transport; rank rows "
              << results[0][0].candidates[0] << " + " << results[1][0].candidates[0] << '\n';
}

// Candidate-form Markov chain (STRIX_DS41_DSPARK_MARKOV=cand). With C = a half's size every position is a candidate:
// the drafts equal the one-rank host draft (full vocabulary), and a half smaller than C runs the full chain (restricted
// vocabulary). With the default C / |H|: one rank and two ranks give the same tokens and confidence (full and restricted
// vocabulary: core ids then extras, H filtered to the domain), and the agreement with the full chain is printed.
void test_cand(const Files& f) {
    const auto model = load_model(f);
    const auto& w = model->w;
    const auto seed = normal_values(static_cast<std::size_t>(130) * main_width, 1.0F, 77);
    const auto accepted = normal_values(static_cast<std::size_t>(3) * main_width, 1.0F, 78);
    Vocab restricted;
    restricted.core = 0;
    for (std::uint32_t id = 0; id < 40000U; id += 2U) restricted.extra.push_back(id);
    restricted.extra.push_back(1001U);
    State host;
    const auto host_main = main_rows(w, seed, 130);
    append(w, host, std::span<const float>(host_main).first(static_cast<std::size_t>(128) * hidden), 128);
    append(w, host, std::span<const float>(host_main).subspan(static_cast<std::size_t>(128) * hidden), 2);
    const auto h1 = draft(w, host, 1000U, false);
    append(w, host, main_rows(w, accepted, 3), 3);
    const auto h2 = draft(w, host, h1.tokens[3], false);
    const auto h3 = draft(w, host, h1.tokens[3], false, restricted);
    const Draft* host_drafts[3] = {&h1, &h2, &h3};
    // the three drafts of test_two_ranks on `world` ranks (world 1: one drafter)
    const auto run = [&](std::uint32_t world) {
        std::vector<std::array<Draft, 3>> out(world);
        auto [ta, tb] = strix::dist::memory_pair();
        std::unique_ptr<strix::dist::Transport> transports[2] = {std::move(ta), std::move(tb)};
        std::exception_ptr errors[2];
        std::vector<std::thread> ranks;
        for (std::uint32_t r = 0; r < world; ++r)
            ranks.emplace_back([&, r] {
                try {
                    std::unique_ptr<strix::dist::Communicator> comm;
                    std::uint64_t request = 0;
                    DeviceDrafter d(w);
                    if (world == 2U) {
                        comm = std::make_unique<strix::dist::Communicator>(strix::dist::Group{r, 2U, 1U, 89U}, std::move(transports[r]), std::size_t{4} << 20U);
                        d.set_rank(r, 2U, [&](void* stream, const void* local, void* remote, std::size_t bytes) {
                            comm->exchange(static_cast<hipStream_t>(stream), local, remote, bytes,
                                           strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::argmax), 0U, request++});
                        });
                    }
                    d.append_hidden(std::span<const float>(seed).first(static_cast<std::size_t>(128) * main_width), 128);
                    d.append_hidden(std::span<const float>(seed).subspan(static_cast<std::size_t>(128) * main_width), 2);
                    out[r][0] = d.draft(1000U, false);
                    d.append_hidden(accepted, 3);
                    out[r][1] = d.draft(h1.tokens[3], false);
                    d.set_vocabulary(restricted);
                    out[r][2] = d.draft(h1.tokens[3], false);
                } catch (...) { errors[r] = std::current_exception(); }
            });
        for (auto& t : ranks) t.join();
        for (auto& e : errors) if (e) std::rethrow_exception(e);
        return out;
    };
    const auto same = [](const Draft& a, const Draft& b) {
        const std::vector<std::uint32_t> at(a.tokens.begin(), a.tokens.end()), bt(b.tokens.begin(), b.tokens.end());
        const std::vector<float> ac(a.confidence.begin(), a.confidence.end()), bc(b.confidence.begin(), b.confidence.end());
        return mismatches(at, bt) + mismatches(ac, bc);
    };
    setenv("STRIX_DS41_DSPARK_MARKOV", "cand", 1);
    setenv("STRIX_DS41_DSPARK_MARKOV_C", std::to_string((vocab + 1U) / 2U).c_str(), 1);
    for (const std::uint32_t world : {1U, 2U}) {
        const auto out = run(world);
        for (std::uint32_t r = 0; r < world; ++r)
            for (std::uint32_t k = 0; k < 3U; ++k)
                report("markov cand C = half, " + std::to_string(world) + " rank(s), rank " + std::to_string(r) + ", draft " + std::to_string(k) +
                           ": tokens and confidence = full one-rank host",
                       same(out[r][k], *host_drafts[k]));
    }
    unsetenv("STRIX_DS41_DSPARK_MARKOV_C");
    const auto one = run(1U), two = run(2U);
    std::size_t agree = 0;
    for (std::uint32_t k = 0; k < 3U; ++k) {
        for (std::uint32_t r = 0; r < 2U; ++r)
            report("markov cand default C / |H|, two ranks, rank " + std::to_string(r) + ", draft " + std::to_string(k) + " = one rank",
                   same(two[r][k], one[0][k]));
        for (std::uint32_t j = 1; j <= block_size; ++j) agree += one[0][k].tokens[j] == host_drafts[k]->tokens[j] ? 1U : 0U;
    }
    std::cout << "markov cand default C / |H|: " << agree << " of " << 3U * block_size << " draft tokens = the full chain's\n";
    setenv("STRIX_DS41_DSPARK_MARKOV", "full", 1);
}

// Bench mode (STRIX_DS41_DSPARK_BENCH=1): no host reference; seed 130 rows, then draft steps and
// 3-row appends, for one-pass kernel traces.
void bench(const Files& f) {
    const auto model = load_model(f);
    DeviceDrafter device(model->w);
    if (const char* v = std::getenv("STRIX_DS41_DSPARK_VOCAB"); v && *v) {
        // auto: specs see a prompt of pack-es tokens when STRIX_DS41_DSPARK_BENCH_LANG=es, else none.
        std::vector<std::uint32_t> prompt;
        if (const char* lang = std::getenv("STRIX_DS41_DSPARK_BENCH_LANG"); lang && *lang && std::string(v).rfind("auto:", 0) == 0) {
            const auto pack = read_vocabulary_file(std::string(v).substr(5) + "/pack-" + lang + "16k.bin");
            prompt.assign(pack.begin(), pack.begin() + 64);
        }
        const auto choice = choose_vocabulary(v, prompt);
        device.set_vocabulary(choice.vocabulary);
        std::cout << "draft vocabulary: " << choice.label << '\n';
    }
    const auto hidden_rows = normal_values(static_cast<std::size_t>(130) * main_width, 1.0F, 77);
    device.append_hidden(std::span<const float>(hidden_rows).first(static_cast<std::size_t>(128) * main_width), 128);
    device.append_hidden(std::span<const float>(hidden_rows).subspan(static_cast<std::size_t>(128) * main_width), 2);
    const auto accepted = normal_values(static_cast<std::size_t>(3) * main_width, 1.0F, 78);
    std::uint32_t token = 1000;
    for (int i = 0; i < 8; ++i) {
        DraftTimes t;
        const auto d = device.draft(token, false, &t);
        const auto a0 = std::chrono::steady_clock::now();
        device.append_hidden(accepted, 3);
        const double append_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a0).count();
        std::cout << "bench step " << i << ": draft " << t.total_ms << " ms (stages " << t.stages_ms[0] << ' ' << t.stages_ms[1] << ' '
                  << t.stages_ms[2] << ", head " << t.head_ms << "), append 3 rows " << append_ms << " ms, tokens";
        for (const auto x : d.tokens) std::cout << ' ' << x;
        std::cout << '\n';
        token = d.tokens[3];
    }
}

} // namespace

int main() try {
    Files f;
    if (const char* q2 = std::getenv("STRIX_DS41_Q2_GGUF")) f.q2 = q2;
    if (const char* d = std::getenv("STRIX_DS41_DSPARK_GGUF")) f.dspark = d;
    if (const char* b = std::getenv("STRIX_DS41_DSPARK_BENCH"); b && *b == '1' && !f.q2.empty() && !f.dspark.empty()) {
        bench(f);
        return 0;
    }
    setenv("STRIX_DS41_DSPARK_MARKOV", "full", 1);  // the host reference drafts with the full chain (test_cand sets cand)
    test_synthetic();
    if (!f.q2.empty() && !f.dspark.empty()) {
        test_real(f);
        test_two_ranks(f);
        test_cand(f);
    }
    else std::cout << "real drafter checks skipped (set STRIX_DS41_Q2_GGUF and STRIX_DS41_DSPARK_GGUF)\n";
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ds41 dspark device tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
