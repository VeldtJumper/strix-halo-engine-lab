// Native exactness gate and microbench for the Engram F16 prefill projection (enqueue_engram_f16_rows).
// No arguments: representative full 6144 -> 25600 shapes across rows 1..8192, plus stride/fallback guards.
// --bench [--rows N] [--iters N] [--copies N]: gate first, then old / pf event medians.
// Without --iters, use one timed launch per arm/round at rows >=1024; five balanced rounds.
#include "strix/hip_ds41_layer.hpp"
#include "../src/hip_ds41_engram_diag.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace strix::ds41;
constexpr unsigned width = 6144, count = 25600;
constexpr std::size_t weight_elements = std::size_t{width} * count;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void select(bool on) {
    if (on ? unsetenv("STRIX_DS41_ENGRAM_F16_PF") : setenv("STRIX_DS41_ENGRAM_F16_PF", "0", 1))
        throw std::runtime_error("set Engram switch");
}
template <class T> struct Device {
    T* p = nullptr;
    explicit Device(std::size_t n) { check(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "allocate"); }
    ~Device() { if (p) (void)hipFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Events {
    hipEvent_t a{}, b{};
    Events() { check(hipEventCreate(&a), "event a"); check(hipEventCreate(&b), "event b"); }
    ~Events() { (void)hipEventDestroy(a); (void)hipEventDestroy(b); }
};

bool equal(const float* a, const float* b, std::size_t n) {
    constexpr std::size_t chunk = 262144;
    std::vector<std::uint32_t> ha(chunk), hb(chunk);
    for (std::size_t i = 0; i < n; i += chunk) {
        const std::size_t k = std::min(chunk, n - i);
        check(hipMemcpy(ha.data(), a + i, k * sizeof(float), hipMemcpyDeviceToHost), "read old");
        check(hipMemcpy(hb.data(), b + i, k * sizeof(float), hipMemcpyDeviceToHost), "read new");
        if (std::memcmp(ha.data(), hb.data(), k * sizeof(float)) != 0) {
            for (std::size_t j = 0; j < k; ++j)
                if (ha[j] != hb[j]) {
                    std::fprintf(stderr, "first mismatch element %zu: %08x vs %08x\n", i + j, ha[j], hb[j]);
                    break;
                }
            return false;
        }
    }
    return true;
}

// Arm 1 = the unset (auto) tile choice. Extra STRIX_DS41_ENGRAM_PF_TILE values (comma list in STRIX_DS41_ENGRAM_PF_BENCH_TILES): gated bitwise and timed as arms 2..
std::vector<std::string> extra_tiles() {
    std::vector<std::string> out;
    const char* e = std::getenv("STRIX_DS41_ENGRAM_PF_BENCH_TILES");
    std::string s = e ? e : "";
    for (std::size_t at = 0; at < s.size();) {
        const std::size_t end = std::min(s.find(',', at), s.size());
        if (end > at) out.push_back(s.substr(at, end - at));
        at = end + 1U;
    }
    return out;
}
// Always gated: the forced 48 x 64 tile (scan + fused form) and the 48 x 32 tile at every admitted shape, then the extras.
std::vector<std::string> gate_tiles() {
    std::vector<std::string> out{"48", "48x32"};
    for (const auto& t : extra_tiles()) out.push_back(t);
    return out;
}
// A "-sep" suffix (e.g. 48x32-sep, or -sep alone for the automatic tile) also sets STRIX_DS41_ENGRAM_PF_FMA=0.
void tile(const std::string* t) {
    std::string name = t ? *t : "";
    const bool sep = name.size() >= 4U && name.compare(name.size() - 4U, 4U, "-sep") == 0;
    if (sep) name.resize(name.size() - 4U);
    if (!name.empty() ? setenv("STRIX_DS41_ENGRAM_PF_TILE", name.c_str(), 1) : unsetenv("STRIX_DS41_ENGRAM_PF_TILE"))
        throw std::runtime_error("set Engram tile");
    if (sep ? setenv("STRIX_DS41_ENGRAM_PF_FMA", "0", 1) : unsetenv("STRIX_DS41_ENGRAM_PF_FMA"))
        throw std::runtime_error("set Engram fma");
}

void enqueue_arm(unsigned arm, const F16RowsView& v) {
    // main() clears the old generic controls. Benchmark explicit kernels, not the switchable wrapper.
    static const std::vector<std::string> tiles = extra_tiles();
    if (arm == 0U) enqueue_f16_rows(v, nullptr);
    else {
        tile(arm >= 2U ? &tiles[arm - 2U] : nullptr);
        enqueue_engram_f16_pf(v, nullptr);
        tile(nullptr);
    }
}

void verify(F16RowsView v, float* other, unsigned copies) {
    const std::size_t n = std::size_t{v.rows} * v.out_stride;
    for (unsigned c = 0; c < copies; ++c) {
        F16RowsView a = v, b = v;
        a.weight += c * weight_elements;
        b.weight = a.weight;
        b.out = other;
        // Different poison patterns detect missing stores; padding must retain the identical sentinel below.
        check(hipMemset(a.out, 0xa5, n * sizeof(float)), "poison old");
        check(hipMemset(b.out, 0x5a, n * sizeof(float)), "poison new");
        if (v.out_stride > v.count) {
            // Identical guard values let the full-buffer comparison include padding.
            check(hipMemset(a.out, 0xa5, n * sizeof(float)), "guard old");
            check(hipMemset(b.out, 0xa5, n * sizeof(float)), "guard new");
        }
        select(false);
        enqueue_engram_f16_rows(a, nullptr);
        select(true);
        enqueue_engram_f16_rows(b, nullptr);
        check(hipDeviceSynchronize(), "bitwise sync");
        if (!equal(a.out, b.out, n)) throw std::runtime_error("old/default mismatch");
        if (engram_f16_pf_admitted(v))
            for (const auto& t : gate_tiles()) {
                check(hipMemset(b.out, v.out_stride > v.count ? 0xa5 : 0x5a, n * sizeof(float)), "poison tile");
                tile(&t);
                enqueue_engram_f16_rows(b, nullptr);
                tile(nullptr);
                check(hipDeviceSynchronize(), "tile sync");
                if (!equal(a.out, b.out, n)) throw std::runtime_error("old/tile " + t + " mismatch");
            }
        if (engram_f16_pf_admitted(v)) {  // the separate multiply and add on every slab
            check(hipMemset(b.out, v.out_stride > v.count ? 0xa5 : 0x5a, n * sizeof(float)), "poison fma off");
            if (setenv("STRIX_DS41_ENGRAM_PF_FMA", "0", 1)) throw std::runtime_error("set fma off");
            enqueue_engram_f16_rows(b, nullptr);
            if (unsetenv("STRIX_DS41_ENGRAM_PF_FMA")) throw std::runtime_error("clear fma off");
            check(hipDeviceSynchronize(), "fma off sync");
            if (!equal(a.out, b.out, n)) throw std::runtime_error("old/fma off mismatch");
        }
        if (v.out_stride > v.count) {
            std::vector<std::uint32_t> row(v.out_stride);
            for (unsigned r = 0; r < v.rows; ++r) {
                check(hipMemcpy(row.data(), b.out + std::size_t{r} * v.out_stride, row.size() * sizeof(float), hipMemcpyDeviceToHost), "guards");
                for (unsigned o = v.count; o < v.out_stride; ++o)
                    if (row[o] != 0xa5a5a5a5U) throw std::runtime_error("output padding overwritten");
            }
        }
    }
    std::printf("6144 -> %u rows %u strides %u/%u: BITWISE EQUAL (%s, %u copies)\n", v.count, v.rows, v.x_stride, v.out_stride,
                engram_f16_pf_supported(v) ? "pf" : "fallback", copies);
    std::fflush(stdout);
}

unsigned bench_iters(unsigned rows) {
    if (rows <= 128U) return 6U;
    if (rows <= 512U) return 2U;
    return 1U;
}

void bench(F16RowsView v, unsigned iters, unsigned copies) {
    if (!engram_f16_pf_supported(v)) throw std::invalid_argument("bench requires rows 32..8192 and a full aligned view");
    const unsigned arms = 2U + static_cast<unsigned>(extra_tiles().size());
    std::vector<std::vector<float>> samples(arms);
    Events ev;
    // Alternating order, reversed every round; all arms use the same copies (including with iters == 1).
    for (unsigned round = 0; round < 5; ++round)
        for (unsigned j = 0; j < arms; ++j) {
            const unsigned arm = (round % 2U ? arms - 1U - j : j);
            auto warm = v;
            warm.weight += ((round + copies - 1U) % copies) * weight_elements;
            enqueue_arm(arm, warm);
            check(hipDeviceSynchronize(), "warmup");
            check(hipEventRecord(ev.a), "start");
            for (unsigned i = 0; i < iters; ++i) {
                auto t = v;
                t.weight += ((i + round) % copies) * weight_elements;
                enqueue_arm(arm, t);
            }
            check(hipEventRecord(ev.b), "stop");
            check(hipEventSynchronize(ev.b), "timing sync");
            float ms = 0.0F;
            check(hipEventElapsedTime(&ms, ev.a, ev.b), "elapsed");
            samples[arm].push_back(ms / static_cast<float>(iters));
        }
    for (auto& s : samples) std::sort(s.begin(), s.end());
    std::printf("BENCH rows %u old %.6f ms pf %.6f ms pf/old %.4f copies %u iters %u (median of 5)\n", v.rows, samples[0][2],
                samples[1][2], samples[1][2] / samples[0][2], copies, iters);
    const auto tiles = extra_tiles();
    for (unsigned a = 2; a < arms; ++a)
        std::printf("BENCH rows %u tile %s %.6f ms tile/pf %.4f (median of 5)\n", v.rows, tiles[a - 2U].c_str(), samples[a][2],
                    samples[a][2] / samples[1][2]);
    std::fflush(stdout);
}

constexpr std::size_t xe_bytes(unsigned w) { return std::size_t{1024} * w * sizeof(float); }

// One planted x value per case: the scan must classify it exactly as the exact-product set says.
void scan_cases(F16RowsView v, const std::vector<float>& base, float* x) {
    struct Case { std::uint32_t bits; bool exact; const char* name; };
    const Case cases[] = {
        {0x00000000U, true, "+0"}, {0x80000000U, true, "-0"}, {0x0D800000U, true, "2^-100"}, {0x8D800000U, true, "-2^-100"},
        {0x0D7F0000U, false, "BF16 below 2^-100"}, {0x00010000U, false, "BF16 subnormal"}, {0x71FF0000U, true, "BF16 below 2^101"},
        {0xF1FF0000U, true, "-BF16 below 2^101"}, {0x72000000U, false, "2^101"}, {0xF2000000U, false, "-2^101"},
        {0x7F800000U, false, "+Inf"}, {0xFF800000U, false, "-Inf"}, {0x7FC00000U, false, "NaN"}, {0xFFC00001U, false, "-NaN"},
        {0x3F800001U, false, "1 + 2^-23"}, {0x3F808000U, false, "1 + 2^-8 (bit 15)"}, {0x3F810000U, true, "1 + 2^-7"}};
    const std::size_t last = std::size_t{v.rows - 1U} * v.x_stride + v.width - 1U;
    for (const Case& c : cases)
        for (const std::size_t at : {std::size_t{0}, std::size_t{v.x_stride} * (v.rows / 2U) + 2944U, last}) {
            std::vector<float> t(base);
            t[at] = std::bit_cast<float>(c.bits);
            check(hipMemcpy(x, t.data(), t.size() * sizeof(float), hipMemcpyHostToDevice), "scan case");
            if (engram_f16_pf_x_exact(v) != c.exact)
                throw std::runtime_error(std::string("scan misclassified ") + c.name + " at " + std::to_string(at));
        }
    if (v.x_stride > v.width) {  // padding is not part of the view
        std::vector<float> t(base);
        for (unsigned r = 0; r < v.rows; ++r) t[std::size_t{r} * v.x_stride + v.width] = std::bit_cast<float>(0x7FC00000U);
        check(hipMemcpy(x, t.data(), t.size() * sizeof(float), hipMemcpyHostToDevice), "scan padding");
        if (!engram_f16_pf_x_exact(v)) throw std::runtime_error("scan read x padding");
    }
    check(hipMemcpy(x, base.data(), base.size() * sizeof(float), hipMemcpyHostToDevice), "scan restore");
    std::printf("scan rows %u stride %u: %zu categories x 3 positions classified\n", v.rows, v.x_stride, std::size(cases));
}

// Full-mantissa x: the unchecked fused form must differ from the separate one somewhere (else the gate cannot tell the
// forms apart), while the default and forced tile calls still match the separate form bit for bit.
void distinguish(F16RowsView v, float* other, float* x, std::mt19937& rng) {
    const std::size_t n = std::size_t{v.rows} * v.out_stride;
    std::vector<float> hx(std::size_t{v.rows} * v.x_stride);
    for (auto& f : hx)
        f = std::bit_cast<float>((static_cast<std::uint32_t>(rng()) & 0x807fffffU) | ((static_cast<std::uint32_t>(rng()) % 25U + 115U) << 23U));
    check(hipMemcpy(x, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice), "full-mantissa inputs");
    if (engram_f16_pf_x_exact(v)) throw std::runtime_error("full-mantissa x classified exact");
    F16RowsView a = v, b = v;
    b.out = other;
    select(false);
    enqueue_engram_f16_rows(a, nullptr);
    select(true);
    check(hipMemcpy(b.out, a.out, n * sizeof(float), hipMemcpyDeviceToDevice), "seed unchecked");
    enqueue_engram_f16_pf_fused_unchecked(b, nullptr);
    check(hipDeviceSynchronize(), "unchecked sync");
    std::vector<std::uint32_t> ha(n), hb(n);
    check(hipMemcpy(ha.data(), a.out, n * sizeof(float), hipMemcpyDeviceToHost), "read separate");
    check(hipMemcpy(hb.data(), b.out, n * sizeof(float), hipMemcpyDeviceToHost), "read unchecked");
    std::size_t diff = 0;
    for (std::size_t i = 0; i < n; ++i) diff += ha[i] != hb[i];
    if (diff == 0U) throw std::runtime_error("unchecked fused form equals the separate form on full-mantissa x");
    std::printf("full-mantissa rows %u: unchecked fused differs in %zu of %zu outputs; gated arms must not\n", v.rows, diff, n);
    verify(v, other, 1);
}

// Forced 48 x 64 tile on the null stream: same bytes as enqueue_f16_rows, and the production flag says which form ran.
void routed(F16RowsView v, float* other, int want) {
    const std::size_t n = std::size_t{v.rows} * v.out_stride;
    F16RowsView b = v;
    b.out = other;
    check(hipMemset(v.out, 0xa5, n * sizeof(float)), "poison old");
    check(hipMemset(b.out, 0xa5, n * sizeof(float)), "poison new");
    enqueue_f16_rows(v, nullptr);
    const std::string t = "48";
    tile(&t);
    enqueue_engram_f16_pf(b, nullptr);
    tile(nullptr);
    const int flag = engram_f16_pf_stream_flag(nullptr);
    if (!equal(v.out, b.out, n)) throw std::runtime_error("routed mismatch");
    if (flag != want)
        throw std::runtime_error("route flag " + std::to_string(flag) + " expected " + std::to_string(want) + " at rows " +
                                 std::to_string(v.rows));
    std::printf("route rows %u stride %u: %s form\n", v.rows, v.x_stride, flag == 0 ? "fused" : "separate");
}

// Several host threads submitting on one explicit stream (one exact and one wide input), the per-thread default
// stream, and graphs captured cold on fresh streams replayed concurrently: every output equals enqueue_f16_rows.
void concurrency(F16RowsView v, const float* exact_x, const float* wide_x) {
    constexpr unsigned rows = 96, reps = 6;
    v.rows = rows;
    const std::size_t n = std::size_t{rows} * v.out_stride;
    Device<float> ref_e(n), ref_w(n), out(n * 4U);
    F16RowsView e = v, w = v;
    e.x = exact_x; w.x = wide_x;
    e.out = ref_e.p; w.out = ref_w.p;
    enqueue_f16_rows(e, nullptr);
    enqueue_f16_rows(w, nullptr);
    check(hipDeviceSynchronize(), "references");
    const std::string t = "48";
    tile(&t);
    const auto compare = [&](const char* what) {
        check(hipDeviceSynchronize(), what);
        for (unsigned k = 0; k < 4U; ++k)
            if (!equal(k % 2U ? ref_w.p : ref_e.p, out.p + k * n, n)) throw std::runtime_error(std::string(what) + " mismatch");
    };
    const auto submit = [&](void* stream, unsigned k) {
        F16RowsView o = k % 2U ? w : e;
        o.out = out.p + k * n;
        for (unsigned r = 0; r < reps; ++r) enqueue_engram_f16_pf(o, stream);
    };
    hipStream_t s{};
    check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
    for (unsigned round = 0; round < 3U; ++round) {
        check(hipMemset(out.p, 0x5a, 4U * n * sizeof(float)), "poison threads");
        std::vector<std::thread> th;
        for (unsigned k = 0; k < 4U; ++k) th.emplace_back(submit, static_cast<void*>(s), k);
        for (auto& x : th) x.join();
        check(hipStreamSynchronize(s), "thread stream");
        compare("one-stream threads");
    }
    check(hipMemset(out.p, 0x5a, 4U * n * sizeof(float)), "poison per-thread");
    {
        std::vector<std::thread> th;
        for (unsigned k = 0; k < 4U; ++k)
            th.emplace_back([&, k] { submit(static_cast<void*>(hipStreamPerThread), k); check(hipStreamSynchronize(hipStreamPerThread), "per-thread sync"); });
        for (auto& x : th) x.join();
    }
    compare("per-thread default stream");
    if (engram_f16_pf_stream_flag(static_cast<void*>(hipStreamPerThread)) != -1)
        throw std::runtime_error("per-thread default stream took the scanned route");
    hipStream_t cs[2]{}, rs[2]{};
    hipGraph_t g[2]{};
    hipGraphExec_t ge[2]{};
    check(hipMemset(out.p, 0x5a, 4U * n * sizeof(float)), "poison graphs");
    for (unsigned k = 0; k < 2U; ++k) {
        check(hipStreamCreateWithFlags(&cs[k], hipStreamNonBlocking), "capture stream");
        check(hipStreamCreateWithFlags(&rs[k], hipStreamNonBlocking), "replay stream");
        check(hipStreamBeginCapture(cs[k], hipStreamCaptureModeThreadLocal), "begin capture");
        F16RowsView o = k ? w : e;
        o.out = out.p + k * n;
        enqueue_engram_f16_pf(o, cs[k]);
        check(hipStreamEndCapture(cs[k], &g[k]), "end capture");
        check(hipGraphInstantiate(&ge[k], g[k], nullptr, nullptr, 0), "instantiate");
        if (engram_f16_pf_stream_flag(cs[k]) != -1) throw std::runtime_error("capturing stream took the scanned route");
    }
    for (unsigned r = 0; r < 3U; ++r)
        for (unsigned k = 0; k < 2U; ++k) check(hipGraphLaunch(ge[k], rs[k]), "replay");
    check(hipDeviceSynchronize(), "graphs");
    for (unsigned k = 0; k < 2U; ++k)
        if (!equal(k ? ref_w.p : ref_e.p, out.p + k * n, n)) throw std::runtime_error("graph replay mismatch");
    for (unsigned k = 0; k < 2U; ++k) {
        (void)hipGraphExecDestroy(ge[k]); (void)hipGraphDestroy(g[k]);
        (void)hipStreamDestroy(cs[k]); (void)hipStreamDestroy(rs[k]);
    }
    (void)hipStreamDestroy(s);
    tile(nullptr);
    std::printf("concurrency rows %u: 4 threads x %u calls on one stream (3 rounds), per-thread default stream, 2 cold captures replayed concurrently: BITWISE EQUAL\n", rows, reps);
}

void rejected(const F16RowsView& v) {
    if (engram_f16_pf_supported(v)) throw std::runtime_error("unsupported view admitted");
    // Guard the explicit entry as well as its predicate; F8 never reaches the tile.
    bool threw = false;
    try {
        enqueue_engram_f16_pf(v, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    if (!threw) throw std::runtime_error("unsupported explicit launch accepted");
}
}  // namespace

int main(int argc, char** argv) try {
    bool timing = false;
    unsigned only_rows = 0, iters = 0, copies = 1;  // zero internally means automatic per-shape iteration count
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bench") { timing = true; continue; }
        if ((arg != "--rows" && arg != "--iters" && arg != "--copies") || i + 1 == argc)
            throw std::invalid_argument("usage: [--bench] [--rows 1..8192] [--iters 1..10000] [--copies 1..8]");
        const auto n = std::stoul(argv[++i]);
        if (n == 0 || n > (arg == "--rows" ? 8192U : arg == "--copies" ? 8U : 10000U))
            throw std::invalid_argument("argument out of range");
        if (arg == "--rows") only_rows = static_cast<unsigned>(n);
        else if (arg == "--copies") copies = static_cast<unsigned>(n);
        else iters = static_cast<unsigned>(n);
    }
    int devices = 0;
    const auto device_status = hipGetDeviceCount(&devices);
    if (device_status == hipErrorNoDevice || (device_status == hipSuccess && devices == 0)) return 77;
    check(device_status, "device count");
    // Fix the old arm to the production dispatch, independent of the invoking shell's tuning overrides.
    if (unsetenv("STRIX_DS41_F16_TILED") || unsetenv("STRIX_DS41_F16_TILE")) throw std::runtime_error("clear old tile controls");
    if (unsetenv("STRIX_DS41_ENGRAM_F16_PF") || unsetenv("STRIX_DS41_ENGRAM_PF_TILE") || unsetenv("STRIX_DS41_ENGRAM_PF_FMA"))
        throw std::runtime_error("clear Engram controls");
    if (timing && only_rows != 0U && only_rows < 32U) throw std::invalid_argument("bench rows must be >=32");
    std::vector<unsigned> shapes = only_rows ? std::vector<unsigned>{only_rows} : timing ?
        std::vector<unsigned>{32, 128, 512, 1024, 2048, 4096, 8192} :
        std::vector<unsigned>{1, 2, 4, 8, 9, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129,
                              511, 512, 513, 1023, 1024, 1025, 2048, 4096, 8192};
    const unsigned max_rows = std::max(1024U, *std::max_element(shapes.begin(), shapes.end()));
    Device<std::uint16_t> w(weight_elements * copies);
    Device<float> x(std::size_t{max_rows} * (width + 4U) + 4U), a(std::size_t{max_rows} * (count + 4U)),
                  b(std::size_t{max_rows} * (count + 4U));
    std::mt19937 rng(41030);
    std::vector<std::uint16_t> hw(weight_elements);
    for (unsigned c = 0; c < copies; ++c) {
        for (auto& h : hw) h = static_cast<std::uint16_t>((rng() % 0x7c00U) | ((rng() & 1U) << 15U));
        check(hipMemcpy(w.p + c * weight_elements, hw.data(), hw.size() * sizeof(hw[0]), hipMemcpyHostToDevice), "weights");
    }
    std::vector<float> hx(std::size_t{max_rows} * (width + 4U) + 4U);
    // Engram rows are BF16-rounded FP32. Include both zeros and a broad finite exponent/mantissa range.
    for (auto& f : hx) {
        const auto exponent = (static_cast<std::uint32_t>(rng()) % 25U + 115U) << 23U;
        f = std::bit_cast<float>((static_cast<std::uint32_t>(rng()) & 0x807f0000U) | exponent);
    }
    for (std::size_t i = 0; i < hx.size(); i += 37U) hx[i] = i % 3U == 0U ? 0.0F : (i % 3U == 1U ? -0.0F : 0x1p-100F);
    check(hipMemcpy(x.p, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice), "inputs");
    F16RowsView v;
    v.weight = w.p; v.width = width; v.count = count; v.x = x.p; v.x_stride = width; v.out_stride = count; v.out = a.p;
    // Pin admission independently of byte equality: falling back everywhere would pass equality too.
    for (const unsigned rows : {1U, 16U, 31U, 32U, 33U, 48U, 64U, 1024U, 8192U, 8193U}) {
        v.rows = rows;
        if (engram_f16_pf_supported(v) != (rows >= 32U && rows <= 8192U)) throw std::runtime_error("row admission changed");
        const bool admitted = rows >= 32U && rows <= 8192U && rows != 64U && rows != 8193U;  // 64: 96 padded rows vs 64
        if (engram_f16_pf_admitted(v) != admitted) throw std::runtime_error("default admission changed");
    }
    for (const unsigned rows : shapes) {
        v.rows = rows;
        verify(v, b.p, copies);
        if (timing) bench(v, iters ? iters : bench_iters(rows), copies);
    }
    if (!timing) {
        // x outside the exact-product set (non-BF16, below 2^-100, at or above 2^101) in mid-width slabs of some rows: the
        // scan must route the dispatch to the separate multiply and add; a fused run would show as mismatches.
        std::vector<float> wx(hx);
        const float wide[] = {0x1p-110F, 0x1.000002p+0F, 0x1p+110F, -0x1.8p-120F, 0x1.4p-101F};
        for (unsigned r = 0; r < 1024U; r += 7U) wx[std::size_t{r} * width + 2944U + r % 300U] = wide[r % 5U];
        v.rows = 1024;
        if (!engram_f16_pf_x_exact(v)) throw std::runtime_error("exact inputs classified wide");
        verify(v, b.p, 1);
        routed(v, b.p, 0);  // fused -> separate -> fused on one stream: each call resets the flag
        Device<float> xe(std::size_t{1024} * width), xw(std::size_t{1024} * width);
        check(hipMemcpy(xe.p, hx.data(), xe_bytes(width), hipMemcpyHostToDevice), "exact copy");
        check(hipMemcpy(x.p, wx.data(), wx.size() * sizeof(float), hipMemcpyHostToDevice), "wide inputs");
        check(hipMemcpy(xw.p, wx.data(), xe_bytes(width), hipMemcpyHostToDevice), "wide copy");
        v.rows = 1000;  // tail row tile; row 994 carries a wide value
        routed(v, b.p, 1);
        for (const unsigned rows : {100U, 1024U}) {
            v.rows = rows;
            if (engram_f16_pf_x_exact(v)) throw std::runtime_error("wide inputs classified exact");
            verify(v, b.p, copies);
        }
        check(hipMemcpy(x.p, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice), "inputs");
        verify(v, b.p, 1);
        routed(v, b.p, 0);
        concurrency(v, xe.p, xw.p);
        v.rows = 96;
        scan_cases(v, std::vector<float>(hx.begin(), hx.begin() + std::size_t{96} * width), x.p);
        v.rows = 256;
        distinguish(v, b.p, x.p, rng);
        check(hipMemcpy(x.p, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice), "inputs");
    }
    // Padded strides and guards: a row predicate cannot mask a bad view guard.
    v.rows = 65; v.x_stride = width + 4U; v.out_stride = count + 4U;
    verify(v, b.p, 1);
    if (!timing) {
        std::vector<float> base(hx.begin(), hx.begin() + std::size_t{65} * (width + 4U));
        scan_cases(v, base, x.p);
        routed(v, b.p, 0);
        std::vector<float> t(base);
        for (unsigned r = 0; r < 65U; ++r) t[std::size_t{r} * (width + 4U) + width] = std::bit_cast<float>(0x7FC00000U);
        check(hipMemcpy(x.p, t.data(), t.size() * sizeof(float), hipMemcpyHostToDevice), "padding NaN");
        routed(v, b.p, 0);
        t[std::size_t{64} * (width + 4U) + width - 1U] = 0x1p+110F;
        check(hipMemcpy(x.p, t.data(), t.size() * sizeof(float), hipMemcpyHostToDevice), "padded wide");
        routed(v, b.p, 1);
        check(hipMemcpy(x.p, base.data(), base.size() * sizeof(float), hipMemcpyHostToDevice), "padded exact");
        routed(v, b.p, 0);
    }
    for (const unsigned rows : {48U, 1024U}) {
        v.rows = rows;
        if (!engram_f16_pf_supported(v)) throw std::runtime_error("padded view rejected");
        verify(v, b.p, 1);
        v.x = x.p + 1;
        rejected(v);
        verify(v, b.p, 1);
        v.x = x.p; v.weight = w.p + 1;
        rejected(v);
        v.weight = w.p; v.first = 1; v.count = count - 1U;
        rejected(v);
        verify(v, b.p, 1);
        v.first = 0; v.count = count; v.weight_f8 = reinterpret_cast<const std::uint8_t*>(w.p);
        rejected(v);  // deliberately no F8 execution: w contains F16, not packed F8 data
        v.weight_f8 = nullptr;
        v.x_stride = width + 1U;
        rejected(v);
        v.x_stride = width - 4U;
        rejected(v);
        v.x_stride = width + 4U; v.out_stride = count - 1U;
        rejected(v);
        v.out_stride = count + 4U;
    }
    std::printf("ALL BITWISE EQUAL\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
