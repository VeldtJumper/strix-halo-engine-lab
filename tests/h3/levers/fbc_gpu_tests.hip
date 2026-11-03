// GPU test of FirstBlockCache (src/h3/levers/fbc/fbc.hpp): device chunk sums == host mirror bitwise (incl. adversarial
// magnitudes); two holders over [0, cut) / [cut, S) (rank order and swapped launch order) give the same block sums and
// decision as one full holder; a call sequence (cold, reuse, reuse, refresh, non-finite, cold, reuse) device == FbcHost
// bitwise; phase / stream / shape guards; reset; allocation-failure injection without leaks. Timing at 38K rows.
// usage: strix-h3-levers-fbc-gpu-tests [--bench]
#include "h3/levers/fbc/fbc.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strix::h3::levers::fbc;
namespace g = strix::h3::gemm;

namespace {
int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
#define HIP(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

struct Rng {
    std::uint64_t s;
    std::uint64_t next() { std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL; return z ^ (z >> 31U); }
    float uni() { return static_cast<float>(next() >> 40U) * 0x1p-24F * 2.0F - 1.0F; }
};
std::vector<std::uint16_t> randbf(std::size_t n, float scale, std::uint64_t seed, bool adversarial = false) {
    Rng r{seed};
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) {
        float m = scale;
        if (adversarial) {
            const std::uint64_t k = r.next() % 4U;
            m = k == 0 ? 0x1p60F : k == 1 ? 0x1p-60F : k == 2 ? 0x1p-130F : 1.0F;
        }
        x = g::f32_to_bf16(r.uni() * m);
    }
    return v;
}
std::uint16_t* up(const std::vector<std::uint16_t>& v) { std::uint16_t* p; HIP(hipMalloc(&p, v.size() * 2)); HIP(hipMemcpy(p, v.data(), v.size() * 2, hipMemcpyHostToDevice)); return p; }
std::vector<std::uint16_t> down(const std::uint16_t* p, std::size_t n) { std::vector<std::uint16_t> v(n); HIP(hipMemcpy(v.data(), p, n * 2, hipMemcpyDeviceToHost)); return v; }
template <class F> bool throws(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
bool same_sums(const Sums& a, const Sums& b) { return std::memcmp(&a.n, &b.n, 8) == 0 && std::memcmp(&a.d, &b.d, 8) == 0; }

// One holder through two calls (cold with h1a, then h1b) -> chunk sums of the second call.
std::vector<Sums> two_calls(FirstBlockCacheDevice& f, const std::uint16_t* h0, const std::uint16_t* h1a, const std::uint16_t* h1b, std::uint32_t rows) {
    f.begin(h0, rows, nullptr);
    f.after_block0(h1a, rows, nullptr);
    std::uint16_t* tmp;
    HIP(hipMalloc(&tmp, static_cast<std::size_t>(rows) * 5376U * 2U));
    if (f.resolve(Decision{}, tmp, nullptr) != Action::kFullCommit) throw std::logic_error("cold call must be kFullCommit");
    f.commit(h1a, rows, nullptr);
    HIP(hipFree(tmp));
    f.begin(h0, rows, nullptr);
    return f.after_block0(h1b, rows, nullptr);
}

void test_chunks_and_ranks(bool adversarial) {
    constexpr std::uint32_t rows = 333U, w = 5376U, cut = 128U;
    const std::size_t n = static_cast<std::size_t>(rows) * w, off = static_cast<std::size_t>(cut) * w;
    const auto h0 = randbf(n, 2.0F, 1, adversarial), h1a = randbf(n, 2.0F, 2, adversarial), h1b = randbf(n, 2.0F, 3, adversarial);
    auto *d0 = up(h0), *da = up(h1a), *db = up(h1b);
    FirstBlockCacheDevice full(rows, w);
    const auto ch = two_calls(full, d0, da, db, rows);
    // host
    std::vector<std::uint16_t> ra(n), rb(n);
    for (std::size_t i = 0; i < n; ++i) { ra[i] = sub_bf16(h1a[i], h0[i]); rb[i] = sub_bf16(h1b[i], h0[i]); }
    const auto hc = chunk_sums_all(rb, ra, rows, w);
    std::size_t bad = hc.size() != ch.size();
    for (std::size_t c = 0; c < std::min(hc.size(), ch.size()); ++c) bad += !same_sums(hc[c], ch[c]);
    // two holders; launch rank 1 first (swapped order), combine in block order
    FirstBlockCacheDevice r1(rows - cut, w), r0(cut, w);
    const auto c1 = two_calls(r1, d0 + off, da + off, db + off, rows - cut);
    const auto c0 = two_calls(r0, d0, da, db, cut);
    const auto fb = split_block_sums(ch, rows, cut);
    const Sums b0 = block_sums(c0, 0, c0.size()), b1 = block_sums(c1, 0, c1.size());
    const Decision df = decide(fb, n, 0.08), dr = decide({b0, b1}, n, 0.08);
    const bool ranks_ok = same_sums(fb[0], b0) && same_sums(fb[1], b1) && std::memcmp(&df.score, &dr.score, 4) == 0 && df.reuse == dr.reuse;
    std::printf("fbc chunk sums 333 x 5376%s: device vs host %zu differ; two holders (swapped launch) vs full: %s (score %g)\n",
        adversarial ? " (adversarial magnitudes)" : "", bad, ranks_ok ? "identical" : "DIFFER", static_cast<double>(df.score));
    CHECK(bad == 0 && ranks_ok, "chunk sums / two ranks");
    HIP(hipFree(d0)); HIP(hipFree(da)); HIP(hipFree(db));
}

// device holder vs FbcHost over a sequence of calls
void test_sequence() {
    constexpr std::uint32_t rows = 333U, w = 5376U, cut = 128U;
    const std::size_t n = static_cast<std::size_t>(rows) * w;
    FirstBlockCacheDevice dev(rows, w);
    FbcHost host(0.08, w);
    const auto h0 = randbf(n, 2.0F, 11), b0 = randbf(n, 0.5F, 12), tail = randbf(n, 1.0F, 13), tail2 = randbf(n, 1.0F, 14), nz = randbf(n, 0.01F, 15),
               bg = randbf(n, 0.5F, 16);
    std::vector<std::uint16_t> b0s(n), b0big(n), b0nan(n);
    for (std::size_t i = 0; i < n; ++i) { b0s[i] = add_bf16(b0[i], nz[i]); b0big[i] = add_bf16(b0[i], bg[i]); }
    b0nan = b0s;
    b0nan[12345] = 0x7fc0U;
    struct C { const std::vector<std::uint16_t>* b; const std::vector<std::uint16_t>* t; Action want; };
    const C calls[] = {{&b0, &tail, Action::kFullCommit},   {&b0s, &tail, Action::kReuse},         {&b0s, &tail, Action::kReuse},
                       {&b0big, &tail2, Action::kFullCommit}, {&b0nan, &tail, Action::kFullNoCommit}, {&b0s, &tail, Action::kFullCommit},
                       {&b0s, &tail, Action::kReuse}};
    int k = 0;
    std::uint16_t* d;
    HIP(hipMalloc(&d, n * 2));
    for (const C& c : calls) {
        std::vector<std::uint16_t> h1(n);
        for (std::size_t i = 0; i < n; ++i) h1[i] = add_bf16(h0[i], (*c.b)[i]);
        // host
        std::vector<std::uint16_t> hh = h1;
        Decision dh;
        const Action ah = host.after_block0(h0, hh, rows, cut, &dh);
        if (ah != Action::kReuse) {
            for (std::size_t i = 0; i < n; ++i) hh[i] = add_bf16(hh[i], (*c.t)[i]);
            if (ah == Action::kFullCommit) host.commit(hh);
        }
        // device
        HIP(hipMemcpy(d, h0.data(), n * 2, hipMemcpyHostToDevice));
        dev.begin(d, rows, nullptr);
        HIP(hipMemcpy(d, h1.data(), n * 2, hipMemcpyHostToDevice));  // "block 0"
        const auto ch = dev.after_block0(d, rows, nullptr);
        Decision dd;
        if (dev.ready()) dd = decide(split_block_sums(ch, rows, cut), n, 0.08);
        const Action ad = dev.resolve(dd, d, nullptr);
        if (ad != Action::kReuse) {
            auto hd = down(d, n);
            for (std::size_t i = 0; i < n; ++i) hd[i] = add_bf16(hd[i], (*c.t)[i]);
            HIP(hipMemcpy(d, hd.data(), n * 2, hipMemcpyHostToDevice));
            if (ad == Action::kFullCommit) dev.commit(d, rows, nullptr);
        }
        HIP(hipDeviceSynchronize());
        const auto out = down(d, n);
        std::size_t bad = (ad != ah) + (std::memcmp(&dd.score, &dh.score, 4) != 0);
        for (std::size_t i = 0; i < n; ++i) bad += out[i] != hh[i];
        std::printf("fbc call %d: action %d (want %d) score %g, device vs host %zu differ\n", k++, static_cast<int>(ad), static_cast<int>(c.want),
            static_cast<double>(dd.score), bad);
        CHECK(bad == 0 && ad == c.want, "fbc call");
    }
    HIP(hipFree(d));
}

void test_guards() {
    constexpr std::uint32_t rows = 128U, w = 256U;
    const std::size_t n = static_cast<std::size_t>(rows) * w;
    std::uint16_t *a, *b;
    HIP(hipMalloc(&a, n * 2)); HIP(hipMalloc(&b, n * 2));
    HIP(hipMemset(a, 0x3c, n * 2)); HIP(hipMemset(b, 0x3d, n * 2));
    hipStream_t s2;
    HIP(hipStreamCreate(&s2));
    FirstBlockCacheDevice f(rows, w);
    int rej = 0;
    rej += throws([&] { f.after_block0(b, rows, nullptr); });           // without begin
    rej += throws([&] { f.commit(b, rows, nullptr); });                 // commit before block 0
    rej += throws([&] { f.begin(a, rows + 1U, nullptr); });             // rows > max
    rej += throws([&] { f.begin(nullptr, rows, nullptr); });            // null
    f.begin(a, rows, nullptr);
    rej += throws([&] { f.after_block0(b, rows, s2); });                // other stream
    rej += throws([&] { f.after_block0(b, rows - 64U, nullptr); });     // rows differ
    f.after_block0(b, rows, nullptr);
    rej += throws([&] { f.commit(b, rows, nullptr); });                 // commit before resolve
    if (f.resolve(Decision{}, b, nullptr) != Action::kFullCommit) ++failures;
    rej += throws([&] { f.resolve(Decision{}, b, nullptr); });          // resolve twice
    f.commit(b, rows, nullptr);
    rej += throws([&] { f.commit(b, rows, nullptr); });                 // double commit
    // reuse call: commit after reuse rejected
    f.begin(a, rows, nullptr);
    const auto ch = f.after_block0(b, rows, nullptr);
    const Decision d = decide(split_block_sums(ch, rows, 64U), n, 0.08);
    if (f.resolve(d, b, nullptr) != Action::kReuse) ++failures;
    rej += throws([&] { f.commit(b, rows, nullptr); });                 // commit after reuse
    rej += throws([&] { f.begin(a, 64U, nullptr); });                   // shape change without reset
    f.reset();
    f.begin(a, 64U, nullptr);                                           // after reset: accepted
    f.after_block0(b, 64U, nullptr);
    const bool cold = f.resolve(Decision{}, b, nullptr) == Action::kFullCommit;
    std::printf("fbc guards: %d of 11 invalid calls rejected, reset + new shape cold %d\n", rej, cold ? 1 : 0);
    CHECK(rej == 11 && cold, "guards");
    HIP(hipStreamDestroy(s2)); HIP(hipFree(a)); HIP(hipFree(b));
}

void test_alloc_failure() {
    HIP(hipDeviceSynchronize());
    std::size_t free0, total, free1;
    HIP(hipMemGetInfo(&free0, &total));
    int thrown = 0;
    for (int at = 0; at < 6; ++at) {
        try {
            FirstBlockCacheDevice f(4096U, 5376U, at);
        } catch (const std::runtime_error&) {
            ++thrown;
        }
    }
    HIP(hipDeviceSynchronize());
    HIP(hipMemGetInfo(&free1, &total));
    const long long leak = static_cast<long long>(free0) - static_cast<long long>(free1);
    std::printf("fbc allocation-failure injection: %d of 6 constructions threw, free memory change %lld bytes (each holder 220 MB)\n", thrown, leak);
    CHECK(thrown == 6 && leak < (16LL << 20), "allocation failure leaks");
    // real failure: an impossible size
    CHECK(throws([] { FirstBlockCacheDevice f(1U << 30U, 5376U); }), "huge allocation throws");
}

// The holder is bound to the stream of its first begin: a full sequence on one non-blocking stream works and equals
// the host; any call on a second non-blocking stream (incl. begin right after a commit whose tail kernel may still be
// pending) is rejected before anything is enqueued.
void test_streams() {
    constexpr std::uint32_t rows = 200U, w = 512U, cut = 64U;
    const std::size_t n = static_cast<std::size_t>(rows) * w;
    hipStream_t s1, s2;
    HIP(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
    HIP(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
    const auto h0 = randbf(n, 2.0F, 61), b0 = randbf(n, 0.5F, 62), nz = randbf(n, 0.01F, 63), tail = randbf(n, 1.0F, 64);
    FirstBlockCacheDevice f(rows, w);
    FbcHost host(0.08, w);
    std::uint16_t *d0 = up(h0), *d;
    HIP(hipMalloc(&d, n * 2));
    std::size_t bad = 0;
    int rej = 0;
    for (int call = 0; call < 3; ++call) {
        std::vector<std::uint16_t> h1(n);
        for (std::size_t i = 0; i < n; ++i) h1[i] = add_bf16(add_bf16(h0[i], b0[i]), call ? nz[i] : static_cast<std::uint16_t>(0));
        std::vector<std::uint16_t> hh = h1;
        const Action ah = host.after_block0(h0, hh, rows, cut, nullptr);
        if (ah != Action::kReuse) {
            for (std::size_t i = 0; i < n; ++i) hh[i] = add_bf16(hh[i], tail[i]);
            if (ah == Action::kFullCommit) host.commit(hh);
        }
        if (call == 1) rej += throws([&] { f.begin(d0, rows, s2); });  // right after the commit of call 0 (tail kernel possibly pending on s1)
        f.begin(d0, rows, s1);
        HIP(hipMemcpyAsync(d, h1.data(), n * 2, hipMemcpyHostToDevice, s1));
        rej += throws([&] { f.after_block0(d, rows, s2); });
        const auto ch = f.after_block0(d, rows, s1);
        Decision dd;
        if (f.ready()) dd = decide(split_block_sums(ch, rows, cut), n, 0.08);
        rej += throws([&] { f.resolve(dd, d, s2); });
        const Action ad = f.resolve(dd, d, s1);
        if (ad != Action::kReuse) {
            HIP(hipStreamSynchronize(s1));
            auto hd = down(d, n);
            for (std::size_t i = 0; i < n; ++i) hd[i] = add_bf16(hd[i], tail[i]);
            HIP(hipMemcpyAsync(d, hd.data(), n * 2, hipMemcpyHostToDevice, s1));
            HIP(hipStreamSynchronize(s1));
            if (ad == Action::kFullCommit) {
                rej += throws([&] { f.commit(d, rows, s2); });
                f.commit(d, rows, s1);
            }
        }
        HIP(hipStreamSynchronize(s1));
        const auto out = down(d, n);
        bad += ad != ah;
        for (std::size_t i = 0; i < n; ++i) bad += out[i] != hh[i];
    }
    std::printf("fbc streams (two non-blocking): 3-call sequence on the bound stream vs host %zu differ, %d of 8 calls on the other stream rejected\n", bad,
        rej);
    CHECK(bad == 0 && rej == 8, "stream binding");
    HIP(hipFree(d0)); HIP(hipFree(d)); HIP(hipStreamDestroy(s1)); HIP(hipStreamDestroy(s2));
}

// Consecutive calls on the bound (non-blocking) stream with NO completion wait between them (the only
// synchronisation is the API's own chunk-sum read inside after_block0 of a ready holder); the other-stream begin is
// attempted IMMEDIATELY after each commit / resolve, before any hipStreamSynchronize; begin while a kFullCommit call
// awaits its commit is rejected. Every call's output (copied stream-ordered into its own buffer) == the host mirror.
void test_stream_no_wait() {
    constexpr std::uint32_t rows = 200U, w = 512U, cut = 64U, ncalls = 6U;
    const std::size_t n = static_cast<std::size_t>(rows) * w;
    hipStream_t s1, s2;
    HIP(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
    HIP(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
    const auto h0 = randbf(n, 2.0F, 71), b0 = randbf(n, 0.5F, 72), nz = randbf(n, 0.01F, 73), bg = randbf(n, 0.5F, 74), tail = randbf(n, 1.0F, 75);
    // per call: block-0 perturbation (0: none, 1: small, 2: large)
    const int pert[ncalls] = {0, 1, 1, 2, 1, 2};
    FirstBlockCacheDevice f(rows, w);
    FbcHost host(0.08, w);
    std::vector<std::vector<std::uint16_t>> h1s(ncalls), hls(ncalls), want(ncalls);
    std::vector<Action> want_act(ncalls);
    for (std::uint32_t c = 0; c < ncalls; ++c) {
        h1s[c].resize(n);
        for (std::size_t i = 0; i < n; ++i)
            h1s[c][i] = add_bf16(add_bf16(h0[i], b0[i]), pert[c] == 0 ? static_cast<std::uint16_t>(0) : pert[c] == 1 ? nz[i] : bg[i]);
        hls[c].resize(n);
        for (std::size_t i = 0; i < n; ++i) hls[c][i] = add_bf16(h1s[c][i], tail[i]);  // "blocks 1.."
        std::vector<std::uint16_t> hh = h1s[c];
        want_act[c] = host.after_block0(h0, hh, rows, cut, nullptr);
        if (want_act[c] == Action::kFullCommit) {
            hh = hls[c];
            host.commit(hh);
        } else if (want_act[c] == Action::kFullNoCommit) {
            hh = hls[c];
        }
        want[c] = hh;
    }
    std::uint16_t *dh0 = up(h0), *d, *outs;
    HIP(hipMalloc(&d, n * 2));
    HIP(hipMalloc(&outs, ncalls * n * 2));
    int rej = 0, act_bad = 0;
    for (std::uint32_t c = 0; c < ncalls; ++c) {
        f.begin(dh0, rows, s1);
        HIP(hipMemcpyAsync(d, h1s[c].data(), n * 2, hipMemcpyHostToDevice, s1));  // "block 0"
        const auto ch = f.after_block0(d, rows, s1);
        Decision dd;
        if (f.ready()) dd = decide(split_block_sums(ch, rows, cut), n, 0.08);
        const Action a = f.resolve(dd, d, s1);
        act_bad += a != want_act[c];
        if (a == Action::kFullCommit) {
            rej += throws([&] { f.begin(dh0, rows, s1); });  // begin while awaiting commit
            HIP(hipMemcpyAsync(d, hls[c].data(), n * 2, hipMemcpyHostToDevice, s1));
            f.commit(d, rows, s1);
        } else if (a == Action::kFullNoCommit) {
            HIP(hipMemcpyAsync(d, hls[c].data(), n * 2, hipMemcpyHostToDevice, s1));
        }
        rej += throws([&] { f.begin(dh0, rows, s2); });  // right after commit / resolve, no synchronisation
        HIP(hipMemcpyAsync(outs + c * n, d, n * 2, hipMemcpyDeviceToDevice, s1));
    }
    HIP(hipStreamSynchronize(s1));
    std::size_t bad = 0;
    for (std::uint32_t c = 0; c < ncalls; ++c) {
        const auto o = down(outs + c * n, n);
        for (std::size_t i = 0; i < n; ++i) bad += o[i] != want[c][i];
    }
    std::uint32_t full = 0;
    for (std::uint32_t c = 0; c < ncalls; ++c) full += want_act[c] == Action::kFullCommit;
    std::printf("fbc no-wait sequence (%u calls, %u full commits, bound non-blocking stream): outputs vs host %zu differ, actions %d differ, "
                "%d of %u rejections (other stream right after each call + begin while awaiting commit)\n",
        ncalls, full, bad, act_bad, rej, ncalls + full);
    CHECK(bad == 0 && act_bad == 0 && rej == static_cast<int>(ncalls + full) && full >= 2U, "no-wait stream sequence");
    HIP(hipFree(dh0)); HIP(hipFree(d)); HIP(hipFree(outs)); HIP(hipStreamDestroy(s1)); HIP(hipStreamDestroy(s2));
}

// Injected launch / copy failure: the call throws, the cache is invalidated (ready() == false), the holder recovers
// (next call cold -> commit -> reuse works). Also: begin while a call is in progress is rejected; rows bound.
void test_failure_injection() {
    constexpr std::uint32_t rows = 128U, w = 256U;
    const std::size_t n = static_cast<std::size_t>(rows) * w;
    std::uint16_t *a, *b;
    HIP(hipMalloc(&a, n * 2)); HIP(hipMalloc(&b, n * 2));
    HIP(hipMemset(a, 0x3c, n * 2)); HIP(hipMemset(b, 0x3d, n * 2));
    FirstBlockCacheDevice f(rows, w);
    const auto full_call = [&]() {
        f.begin(a, rows, nullptr);
        const auto ch = f.after_block0(b, rows, nullptr);
        Decision d;
        if (f.ready()) d = decide(split_block_sums(ch, rows, 64U), n, 0.08);
        const Action act = f.resolve(d, b, nullptr);
        if (act == Action::kFullCommit) f.commit(b, rows, nullptr);
        return act;
    };
    int ok = 0;
    ok += full_call() == Action::kFullCommit && f.ready();
    // failure inside after_block0 of a ready holder
    f.begin(a, rows, nullptr);
    f.inject_failure_for_test();
    const bool t1 = throws([&] { f.after_block0(b, rows, nullptr); });
    ok += t1 && !f.ready();
    ok += full_call() == Action::kFullCommit && f.ready();  // recovery: cold
    ok += full_call() == Action::kReuse;
    // failure inside commit
    f.reset();
    f.begin(a, rows, nullptr);
    f.after_block0(b, rows, nullptr);
    f.resolve(Decision{}, b, nullptr);
    f.inject_failure_for_test();
    const bool t2 = throws([&] { f.commit(b, rows, nullptr); });
    ok += t2 && !f.ready();
    ok += full_call() == Action::kFullCommit && f.ready();
    ok += full_call() == Action::kReuse;
    // begin while a call is in progress
    f.begin(a, rows, nullptr);
    ok += throws([&] { f.begin(a, rows, nullptr); });
    f.after_block0(b, rows, nullptr);
    ok += throws([&] { f.begin(a, rows, nullptr); });
    f.reset();
    ok += !throws([&] { f.begin(a, rows, nullptr); });
    // rows bound
    ok += throws([] { FirstBlockCacheDevice g(kMaxRows + 1U, 1U); });
    std::printf("fbc failure injection / phases / rows bound: %d of 11 checks\n", ok);
    CHECK(ok == 11, "failure injection");
    HIP(hipFree(a)); HIP(hipFree(b));
}

void bench() {
    constexpr std::uint32_t rows = 38000U, w = 5376U;
    const std::size_t n = static_cast<std::size_t>(rows) * w;
    std::uint16_t *a, *b;
    HIP(hipMalloc(&a, n * 2)); HIP(hipMalloc(&b, n * 2));
    HIP(hipMemset(a, 0x3c, n * 2)); HIP(hipMemset(b, 0x3d, n * 2));
    FirstBlockCacheDevice f(rows, w);
    f.begin(a, rows, nullptr); f.after_block0(b, rows, nullptr); f.resolve(Decision{}, b, nullptr); f.commit(b, rows, nullptr);
    HIP(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    f.begin(a, rows, nullptr);
    const auto ch = f.after_block0(b, rows, nullptr);
    const auto t1 = std::chrono::steady_clock::now();
    const Decision d = decide(split_block_sums(ch, rows, 18944U), n, 0.08);
    const Action act = f.resolve(d, b, nullptr);
    HIP(hipDeviceSynchronize());
    const auto t2 = std::chrono::steady_clock::now();
    const auto ms = [](auto x, auto y) { return std::chrono::duration<double, std::milli>(y - x).count(); };
    std::printf("bench fbc rows %u: begin + after_block0 (incl. chunk sums, sync) %.2f ms, decide + resolve (%d) %.2f ms, %zu chunks, device %zu MB\n", rows,
        ms(t0, t1), static_cast<int>(act), ms(t1, t2), ch.size(), f.device_bytes() >> 20U);
    HIP(hipFree(a)); HIP(hipFree(b));
}
}  // namespace

int main(int argc, char** argv) {
    test_chunks_and_ranks(false);
    test_chunks_and_ranks(true);
    test_sequence();
    test_guards();
    test_alloc_failure();
    test_streams();
    test_stream_no_wait();
    test_failure_injection();
    std::printf("fbc correctness %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    if (argc > 1 && std::string(argv[1]) == "--bench") bench();
    return failures ? 1 : 0;
}
