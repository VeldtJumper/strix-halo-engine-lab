// GPU tests of the H3 W8A8 convrot projection (src/h3/gemm/kernels.hpp) against the host reference (ref.hpp).
//   correctness: quantizer (rotated codes + fp32 row scales) bitwise, int32 accumulators bitwise, bf16 outputs bitwise,
//     both tile widths, M not a multiple of 256, fused SwiGLU vs the unfused chain, q | k | v segment output, batch
//     invariance (rows of a short call equal the same rows inside a long call);
//   dumps (argv[1] or STRIX_H3_GEMM_DUMPS): every step vs the python chain's files (B_* exact-rotation chain);
//   timing (--bench): quantizer + GEMM for qkv / out / fc1 (+swiglu) / fc2 at M = 19000 and 38000.
#include "h3/gemm/api.hpp"
#include "g1_gate.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace strix::h3::gemm;

namespace {
int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
#define HIP(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

struct SplitMix {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31U);
    }
};
template <class T> struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Dev(std::size_t count) : n(count) { HIP(hipMalloc(&p, std::max<std::size_t>(1, count) * sizeof(T))); }
    Dev(const std::vector<T>& v) : Dev(v.size()) { HIP(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); }
    ~Dev() { static_cast<void>(hipFree(p)); }
    std::vector<T> get() const { std::vector<T> v(n); HIP(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return v; }
    Dev(const Dev&) = delete;
};
template <class T> std::size_t ndiff(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return ~std::size_t{0};
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return d;
}
// activations with outlier channels and a row-magnitude spread (bf16 bits)
std::vector<std::uint16_t> activations(std::uint32_t m, std::uint32_t k, std::uint64_t seed) {
    SplitMix g{seed};
    std::vector<float> ch(k);
    for (auto& c : ch) c = std::ldexp(1.0F, static_cast<int>(g.next() % 7U) - 3);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(m) * k);
    for (std::uint32_t r = 0; r < m; ++r) {
        const float rs = std::ldexp(1.0F, static_cast<int>(g.next() % 9U) - 4);
        for (std::uint32_t i = 0; i < k; ++i) {
            const double u1 = (static_cast<double>(g.next() >> 11U) + 0.5) * 0x1p-53, u2 = static_cast<double>(g.next() >> 11U) * 0x1p-53;
            const float z = static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
            x[static_cast<std::size_t>(r) * k + i] = f32_to_bf16(z * rs * ch[i]);
        }
    }
    if (m > 3U) for (std::uint32_t i = 0; i < k; ++i) x[3U * k + i] = 0;  // an all-zero row
    return x;
}
std::vector<std::int8_t> weights(std::uint32_t n, std::uint32_t k, std::uint64_t seed) {
    SplitMix g{seed};
    std::vector<std::int8_t> w(static_cast<std::size_t>(n) * k);
    for (auto& v : w) v = static_cast<std::int8_t>(static_cast<std::uint8_t>(g.next()));
    return w;
}
std::vector<float> channel_scales(std::uint32_t n, std::uint64_t seed) {
    SplitMix g{seed};
    std::vector<float> s(n);
    for (auto& v : s) v = std::ldexp(1.0F + static_cast<float>(g.next() % 1000U) / 1000.0F, -12);
    return s;
}

struct DeviceRun {
    std::vector<std::int8_t> q;
    std::vector<float> scale;
    std::vector<std::int32_t> acc;
    std::vector<std::uint16_t> y;
    unsigned flushed = 0;
};
template <unsigned BN>
DeviceRun run_device(const std::vector<std::uint16_t>& x, std::uint32_t m, std::uint32_t k, const std::vector<std::int8_t>& w,
    const std::vector<float>& ws, std::uint32_t n, bool with_acc) {
    Dev<std::uint16_t> dx(x), dy(static_cast<std::size_t>(m) * n);
    Dev<std::int8_t> dq(static_cast<std::size_t>(m) * k), dw(w);
    Dev<float> ds(m), dws(ws);
    Dev<unsigned> dfl(1);
    HIP(hipMemset(dfl.p, 0, 4));
    launch_quantize(dx.p, k, m, k, dq.p, ds.p, dfl.p, nullptr);
    dev::GemmArgs a;
    a.a = dq.p; a.b = dw.p; a.row_scale = ds.p; a.ch_scale = dws.p; a.m = m; a.n = n; a.k = k;
    a.out.ptr[0] = dy.p; a.out.ld = n; a.out.seg_cols = n;
    launch_gemm<BN, dev::Epi::plain>(a, nullptr);
    DeviceRun r;
    if (with_acc) {
        Dev<std::int32_t> dacc(static_cast<std::size_t>(m) * n);
        a.acc_out = dacc.p;
        launch_gemm<BN, dev::Epi::raw>(a, nullptr);
        HIP(hipDeviceSynchronize());
        r.acc = dacc.get();
    }
    HIP(hipDeviceSynchronize());
    r.q = dq.get(); r.scale = ds.get(); r.y = dy.get(); r.flushed = dfl.get()[0];
    return r;
}

template <unsigned BN> void test_projection(std::uint32_t m, std::uint32_t k, std::uint32_t n, std::uint64_t seed) {
    const auto x = activations(m, k, seed);
    const auto w = weights(n, k, seed + 1U);
    const auto ws = channel_scales(n, seed + 2U);
    const Projection ref = project(x.data(), m, k, w.data(), ws.data(), n);
    const DeviceRun d = run_device<BN>(x, m, k, w, ws, n, true);
    const std::size_t dq = ndiff(d.q, ref.q), ds = ndiff(d.scale, ref.row_scale), da = ndiff(d.acc, ref.acc), dy = ndiff(d.y, ref.y);
    std::printf("projection BN %u M %u K %u N %u: codes %zu, row scales %zu, acc %zu, out %zu differ (flushed groups %u / host %u)\n", BN, m,
        k, n, dq, ds, da, dy, d.flushed, ref.inexact_groups);
    CHECK(dq == 0 && ds == 0 && da == 0 && dy == 0 && d.flushed == ref.inexact_groups, "projection BN %u M %u K %u N %u", BN, m, k, n);
}

// Fused SwiGLU (fc1 with permuted rows) == host swiglu(epilogue(gate), epilogue(up)); also q | k | v segments.
template <unsigned BN> void test_swiglu_and_segments(std::uint32_t m, std::uint64_t seed) {
    constexpr std::uint32_t k = 768U, half = 512U, n = 2U * half;
    const auto x = activations(m, k, seed);
    const auto w = weights(n, k, seed + 1U);
    const auto ws = channel_scales(n, seed + 2U);
    const Projection ref = project(x.data(), m, k, w.data(), ws.data(), n);
    std::vector<std::uint16_t> h_ref(static_cast<std::size_t>(m) * half);
    for (std::uint32_t r = 0; r < m; ++r)
        for (std::uint32_t j = 0; j < half; ++j) h_ref[r * half + j] = swiglu(ref.y[r * n + j], ref.y[r * n + half + j]);
    std::vector<std::int8_t> wp(w.size());
    std::vector<float> wsp(n);
    for (std::uint32_t p = 0; p < n; ++p) {
        const std::uint32_t src = swiglu_source_row(p, n);
        std::memcpy(wp.data() + static_cast<std::size_t>(p) * k, w.data() + static_cast<std::size_t>(src) * k, k);
        wsp[p] = ws[src];
    }
    std::vector<std::uint16_t> table(silu_table().begin(), silu_table().end());
    Dev<std::uint16_t> dx(x), dh(static_cast<std::size_t>(m) * half), dtab(table);
    Dev<std::int8_t> dq(static_cast<std::size_t>(m) * k), dw(wp), dw0(w);
    Dev<float> ds(m), dws(wsp), dws0(ws);
    launch_quantize(dx.p, k, m, k, dq.p, ds.p, nullptr, nullptr);
    dev::GemmArgs a;
    a.a = dq.p; a.b = dw.p; a.row_scale = ds.p; a.ch_scale = dws.p; a.silu = dtab.p; a.m = m; a.n = n; a.k = k;
    a.out.ptr[0] = dh.p; a.out.ld = half; a.out.seg_cols = half;
    launch_gemm<BN, dev::Epi::swiglu>(a, nullptr);
    // segments: the unpermuted projection into 4 buffers of n / 4 columns with a padded pitch
    const std::uint32_t seg = n / 4U, ld = seg + 8U;
    Dev<std::uint16_t> s0(static_cast<std::size_t>(m) * ld), s1(static_cast<std::size_t>(m) * ld), s2(static_cast<std::size_t>(m) * ld),
        s3(static_cast<std::size_t>(m) * ld);
    dev::GemmArgs b = a;
    b.b = dw0.p; b.ch_scale = dws0.p; b.out = {};
    b.out.ptr[0] = s0.p; b.out.ptr[1] = s1.p; b.out.ptr[2] = s2.p; b.out.ptr[3] = s3.p; b.out.ld = ld; b.out.seg_cols = seg;
    launch_gemm<BN, dev::Epi::plain>(b, nullptr);
    HIP(hipDeviceSynchronize());
    const std::size_t dhs = ndiff(dh.get(), h_ref);
    std::size_t dseg = 0;
    const std::vector<std::uint16_t> sv[4] = {s0.get(), s1.get(), s2.get(), s3.get()};
    for (std::uint32_t r = 0; r < m; ++r)
        for (std::uint32_t c = 0; c < n; ++c) dseg += sv[c / seg][static_cast<std::size_t>(r) * ld + c % seg] != ref.y[static_cast<std::size_t>(r) * n + c];
    std::printf("swiglu BN %u M %u: fused h %zu of %zu differ from the unfused chain; 4-segment output %zu differ\n", BN, m, dhs, h_ref.size(), dseg);
    CHECK(dhs == 0 && dseg == 0, "swiglu / segments BN %u", BN);
}

// Rows 0..m_short-1 of a long call equal a short call (quantizer + GEMM): batch invariance.
void test_batch_invariance() {
    constexpr std::uint32_t k = 1024U, n = 256U, m_long = 1000U, m_short = 77U;
    const auto x = activations(m_long, k, 99U);
    const auto w = weights(n, k, 100U);
    const auto ws = channel_scales(n, 101U);
    const std::vector<std::uint16_t> xs(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(m_short) * k);
    const DeviceRun L = run_device<128U>(x, m_long, k, w, ws, n, false), S = run_device<256U>(xs, m_short, k, w, ws, n, false);
    const std::vector<std::uint16_t> head(L.y.begin(), L.y.begin() + static_cast<std::ptrdiff_t>(m_short) * n);
    const std::size_t d = ndiff(head, S.y);
    std::printf("batch invariance: rows 0..%u of M %u (BN 128) vs M %u (BN 256): %zu differ\n", m_short - 1U, m_long, m_short, d);
    CHECK(d == 0, "batch invariance");
}

template <class T> std::vector<T> load(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    std::vector<T> v(static_cast<std::size_t>(f.tellg()) / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    return v;
}
// Qualification bookkeeping (--qualify: every required case must have run; a skip is a failure).
std::set<std::string> seen;

// N padded to the next 128 with zero weight rows (scale 1); the first n columns are compared.
DeviceRun run_padded(const std::vector<std::uint16_t>& x, std::uint32_t m, std::uint32_t k, const std::vector<std::int8_t>& w,
    const std::vector<float>& ws, std::uint32_t n, Projection* host) {
    const std::uint32_t np = (n + 127U) / 128U * 128U;
    std::vector<std::int8_t> wp(static_cast<std::size_t>(np) * k, 0);
    std::copy(w.begin(), w.end(), wp.begin());
    std::vector<float> wsp(np, 1.0F);
    std::copy(ws.begin(), ws.end(), wsp.begin());
    DeviceRun d = run_device<128U>(x, m, k, wp, wsp, np, true);
    if (host) *host = project(x.data(), m, k, wp.data(), wsp.data(), np);
    if (np != n) {
        std::vector<std::uint16_t> y(static_cast<std::size_t>(m) * n);
        std::vector<std::int32_t> acc(y.size());
        for (std::uint32_t r = 0; r < m; ++r)
            for (std::uint32_t c = 0; c < n; ++c) {
                y[static_cast<std::size_t>(r) * n + c] = d.y[static_cast<std::size_t>(r) * np + c];
                acc[static_cast<std::size_t>(r) * n + c] = d.acc[static_cast<std::size_t>(r) * np + c];
            }
        d.y = std::move(y);
        d.acc = std::move(acc);
    }
    return d;
}

// Host rotated rows (the device quantizer's rotation is the same arithmetic; its codes are compared bitwise to the host).
std::vector<std::uint16_t> host_rotated(const std::vector<std::uint16_t>& x, std::uint32_t m, std::uint32_t k) {
    std::vector<std::uint16_t> xr(x.size());
    std::vector<std::int8_t> q(k);
    for (std::uint32_t r = 0; r < m; ++r) rotate_quantize_row(x.data() + static_cast<std::size_t>(r) * k, k, q.data(), xr.data() + static_cast<std::size_t>(r) * k);
    return xr;
}
// G1 (asserted in --qualify runs too): rotation within the fp32-order bound of the reference, device code flips in budget.
void g1_assert(const std::string& name, const std::vector<std::uint16_t>& x, std::uint32_t m, std::uint32_t k, const std::vector<std::uint16_t>& ref_rot,
    std::size_t device_code_flips, std::size_t codes) {
    const g1::RotGate gt = g1::rotation_gate(x, host_rotated(x, m, k), ref_rot);
    std::printf("  G1 %s: rotation max %u ulp, %zu > 1 ulp, worst / bound %.3f; device code flips %zu (budget %zu)\n", name.c_str(), gt.max_ulp,
        gt.over_1ulp, gt.worst, device_code_flips, (codes + 500000U) / 250000U);
    CHECK(gt.ok && g1::flips_ok(device_code_flips, codes), "G1 %s", name.c_str());
}

void test_dumps(const std::filesystem::path& root) {
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        const auto dir = entry.path();
        if (!std::filesystem::exists(dir / "meta.txt")) continue;
        std::map<std::string, std::vector<std::size_t>> sh;
        {
            std::ifstream f(dir / "meta.txt");
            std::string line;
            while (std::getline(f, line)) { std::istringstream is(line); std::string nm; is >> nm; std::size_t d; while (is >> d) sh[nm].push_back(d); }
        }
        const auto m = static_cast<std::uint32_t>(sh.at("x")[0]), k = static_cast<std::uint32_t>(sh.at("x")[1]), n = static_cast<std::uint32_t>(sh.at("w")[0]);
        const bool g1_only = std::filesystem::exists(dir / "g1_only");  // groups beyond the exact window: no fp64-exact gate
        Projection host;
        const DeviceRun d = run_padded(load<std::uint16_t>(dir / "x.bin"), m, k, load<std::int8_t>(dir / "w.bin"), load<float>(dir / "ws.bin"), n, &host);
        {   // GPU == host reference (incl. flushed groups and their count)
            const std::size_t np = host.y.size() / m;
            std::vector<std::uint16_t> hy(static_cast<std::size_t>(m) * n);
            std::vector<std::int32_t> hacc(hy.size());
            for (std::uint32_t r = 0; r < m; ++r)
                for (std::uint32_t c = 0; c < n; ++c) {
                    hy[static_cast<std::size_t>(r) * n + c] = host.y[r * np + c];
                    hacc[static_cast<std::size_t>(r) * n + c] = host.acc[r * np + c];
                }
            const std::size_t dq = ndiff(d.q, host.q), dy = ndiff(d.y, hy), ds = ndiff(d.scale, host.row_scale), da = ndiff(d.acc, hacc);
            std::printf("dump %s GPU vs host: codes %zu, row scales %zu, int32 acc %zu, out %zu, flushed groups %u / host %u\n", dir.filename().string().c_str(), dq, ds, da, dy,
                d.flushed, host.inexact_groups);
            CHECK(dq == 0 && ds == 0 && da == 0 && dy == 0 && d.flushed == host.inexact_groups, "dump %s: GPU != host", dir.filename().string().c_str());
        }
        for (const char* pre : {"B_", "A_"}) {
            const std::string p(pre);
            const std::size_t dq = ndiff(d.q, load<std::int8_t>(dir / (p + "q.bin"))), ds = ndiff(d.scale, load<float>(dir / (p + "xs.bin")));
            const std::size_t da = ndiff(d.acc, load<std::int32_t>(dir / (p + "acc.bin"))), dy = ndiff(d.y, load<std::uint16_t>(dir / (p + "out.bin")));
            std::printf("dump %s (M %u K %u N %u) GPU vs python %s: codes %zu / %zu, row scales %zu, acc %zu / %zu, out %zu\n",
                dir.filename().string().c_str(), m, k, n, pre[0] == 'B' ? "exact-rotation chain" : "verbatim chain", dq, d.q.size(), ds, da, d.acc.size(), dy);
            if (pre[0] == 'B') { if (!g1_only) CHECK(dq == 0 && ds == 0 && da == 0 && dy == 0, "dump %s", dir.filename().string().c_str()); }
            else g1_assert("dump " + dir.filename().string(), load<std::uint16_t>(dir / "x.bin"), m, k, load<std::uint16_t>(dir / "A_xrot.bin"), dq, d.q.size());
        }
        seen.insert("dump/" + dir.filename().string());
    }
}

// Public API (api.hpp): linear and linear_fc1_swiglu vs the host reference, padded input / output pitches; argument
// validation (each must throw before queuing anything); flush telemetry.
void test_api() {
    constexpr std::uint32_t k = 1280U, n = 1024U, m = 301U, xld = k + 64U, yld = n + 8U;
    const auto x0 = activations(m, k, 31U);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(m) * xld, 0x7fc0U);
    for (std::uint32_t r = 0; r < m; ++r)
        std::copy(x0.begin() + static_cast<std::ptrdiff_t>(r) * k, x0.begin() + static_cast<std::ptrdiff_t>(r + 1U) * k, x.begin() + static_cast<std::ptrdiff_t>(r) * xld);
    const auto w = weights(n, k, 32U);
    const auto ws = channel_scales(n, 33U);
    const Projection ref = project(x0.data(), m, k, w.data(), ws.data(), n);
    const auto h_ref = project_fc1_swiglu(x0.data(), m, k, w.data(), ws.data(), n);
    const SwiGLUWeights perm = permute_fc1_for_swiglu(w.data(), ws.data(), n, k);
    Dev<std::uint16_t> dx(x), dy(static_cast<std::size_t>(m) * yld), dh(static_cast<std::size_t>(m) * (n / 2U));
    Dev<std::int8_t> dw(w), dwp(perm.codes);
    Dev<float> dws(ws), dwsp(perm.scales);
    Workspace work(m, k);
    const Weights wt{dw.p, dws.p, n, k};
    linear(dx.p, xld, m, wt, work, dy.p, yld, nullptr);
    linear_fc1_swiglu(dx.p, xld, m, Weights{dwp.p, dwsp.p, n, k}, work, dh.p, n / 2U, nullptr);
    HIP(hipDeviceSynchronize());
    const auto y = dy.get();
    std::size_t dyc = 0;
    for (std::uint32_t r = 0; r < m; ++r)
        for (std::uint32_t c = 0; c < n; ++c) dyc += y[static_cast<std::size_t>(r) * yld + c] != ref.y[static_cast<std::size_t>(r) * n + c];
    const std::size_t dhc = ndiff(dh.get(), h_ref);
    const unsigned fl = work.flush_count(nullptr);
    std::printf("api: linear %zu of %zu differ, linear_fc1_swiglu %zu of %zu differ, flush count %u\n", dyc, ref.y.size(), dhc, h_ref.size(), fl);
    CHECK(dyc == 0 && dhc == 0 && fl == 0U, "api");
    // validation
    unsigned rejected = 0, cases = 0;
    const auto expect_throw = [&](const char* what, const std::function<void()>& f) {
        ++cases;
        try { f(); std::printf("  validation: %s NOT rejected\n", what); } catch (const std::invalid_argument&) { ++rejected; }
    };
    expect_throw("k = 0", [&] { linear(dx.p, xld, m, Weights{dw.p, dws.p, n, 0U}, work, dy.p, yld, nullptr); });
    expect_throw("n = 0", [&] { linear(dx.p, xld, m, Weights{dw.p, dws.p, 0U, k}, work, dy.p, yld, nullptr); });
    expect_throw("rows = 0", [&] { linear(dx.p, xld, 0U, wt, work, dy.p, yld, nullptr); });
    expect_throw("k % 256", [&] { linear(dx.p, xld, m, Weights{dw.p, dws.p, n, 1152U}, work, dy.p, yld, nullptr); });
    expect_throw("n % 256", [&] { linear(dx.p, xld, m, Weights{dw.p, dws.p, 640U, k}, work, dy.p, yld, nullptr); });
    expect_throw("x_ld < k", [&] { linear(dx.p, k - 8U, m, wt, work, dy.p, yld, nullptr); });
    expect_throw("x row pitch not 16-byte", [&] { linear(dx.p, xld + 4U, m, wt, work, dy.p, yld, nullptr); });
    expect_throw("x misaligned", [&] { linear(dx.p + 4, xld, m, wt, work, dy.p, yld, nullptr); });
    expect_throw("y_ld < n", [&] { linear(dx.p, xld, m, wt, work, dy.p, n - 1U, nullptr); });
    expect_throw("rows > workspace", [&] { linear(dx.p, xld, m + 1U, wt, work, dy.p, yld, nullptr); });
    expect_throw("swiglu n % 512", [&] { linear_fc1_swiglu(dx.p, xld, m, Weights{dwp.p, dwsp.p, 768U, k}, work, dh.p, n / 2U, nullptr); });
    expect_throw("swiglu h_ld < n / 2", [&] { linear_fc1_swiglu(dx.p, xld, m, Weights{dwp.p, dwsp.p, n, k}, work, dh.p, n / 2U - 1U, nullptr); });
    expect_throw("segments > 4", [&] {
        dev::GemmArgs a; a.a = work.codes(); a.b = dw.p; a.row_scale = work.row_scale(); a.ch_scale = dws.p; a.m = m; a.n = n; a.k = k;
        a.out.ptr[0] = a.out.ptr[1] = a.out.ptr[2] = a.out.ptr[3] = dy.p; a.out.ld = yld; a.out.seg_cols = n / 5U;
        launch_gemm<128U, dev::Epi::plain>(a, nullptr); });
    expect_throw("segment pointer unset", [&] {
        dev::GemmArgs a; a.a = work.codes(); a.b = dw.p; a.row_scale = work.row_scale(); a.ch_scale = dws.p; a.m = m; a.n = n; a.k = k;
        a.out.ptr[0] = dy.p; a.out.ld = yld; a.out.seg_cols = n / 2U;
        launch_gemm<128U, dev::Epi::plain>(a, nullptr); });
    expect_throw("seg_cols = 0", [&] {
        dev::GemmArgs a; a.a = work.codes(); a.b = dw.p; a.row_scale = work.row_scale(); a.ch_scale = dws.p; a.m = m; a.n = n; a.k = k;
        a.out.ptr[0] = dy.p; a.out.ld = yld; a.out.seg_cols = 0U;
        launch_gemm<128U, dev::Epi::plain>(a, nullptr); });
    expect_throw("rows * k >= 2^32", [&] { linear(dx.p, 16384U, 262145U, Weights{dw.p, dws.p, n, 16384U}, work, dy.p, yld, nullptr); });
    expect_throw("gemm m * k >= 2^32", [&] {
        dev::GemmArgs a; a.a = work.codes(); a.b = dw.p; a.row_scale = work.row_scale(); a.ch_scale = dws.p; a.m = 262145U; a.n = n; a.k = 16384U;
        a.out.ptr[0] = dy.p; a.out.ld = yld; a.out.seg_cols = n;
        launch_gemm<128U, dev::Epi::plain>(a, nullptr); });
    expect_throw("gemm k > 16384", [&] {
        dev::GemmArgs a; a.a = work.codes(); a.b = dw.p; a.row_scale = work.row_scale(); a.ch_scale = dws.p; a.m = 4U; a.n = n; a.k = 16512U;
        a.out.ptr[0] = dy.p; a.out.ld = yld; a.out.seg_cols = n;
        launch_gemm<128U, dev::Epi::plain>(a, nullptr); });
    expect_throw("gemm raster_m 2^31", [&] {
        dev::GemmArgs a; a.a = work.codes(); a.b = dw.p; a.row_scale = work.row_scale(); a.ch_scale = dws.p; a.m = m; a.n = n; a.k = k;
        a.out.ptr[0] = dy.p; a.out.ld = yld; a.out.seg_cols = n; a.raster_m = 0x80000000U;
        launch_gemm<128U, dev::Epi::plain>(a, nullptr); });
    expect_throw("gemm raster_m 0", [&] {
        dev::GemmArgs a; a.a = work.codes(); a.b = dw.p; a.row_scale = work.row_scale(); a.ch_scale = dws.p; a.m = m; a.n = n; a.k = k;
        a.out.ptr[0] = dy.p; a.out.ld = yld; a.out.seg_cols = n; a.raster_m = 0U;
        launch_gemm<128U, dev::Epi::plain>(a, nullptr); });
    std::printf("api validation: %u of %u invalid calls rejected\n", rejected, cases);
    CHECK(rejected == cases, "api validation");
    // flush telemetry: rows whose groups span more than 47 binades are counted (host reference counts the same groups)
    {
        std::vector<std::uint16_t> xf(static_cast<std::size_t>(4) * xld, 0);
        for (unsigned r = 0; r < 4U; ++r)
            for (unsigned i = 0; i < k; ++i) xf[r * xld + i] = f32_to_bf16(1.0F + static_cast<float>(i % 7U));
        xf[0 * xld + 5] = 0x0080U;              // 2^-126 next to ~1: 126 binades -> group 0 of row 0 flushed
        xf[2 * xld + 300] = 0x7f00U;            // 2^127 among ~1 -> group 1 of row 2 flushed
        std::vector<std::uint16_t> xc(static_cast<std::size_t>(4) * k);
        for (unsigned r = 0; r < 4U; ++r) std::copy(xf.begin() + r * xld, xf.begin() + r * xld + k, xc.begin() + r * k);
        const Projection hf = project(xc.data(), 4U, k, w.data(), ws.data(), n);
        Dev<std::uint16_t> dxf(xf);
        // producer = a NON-BLOCKING stream with a queue of work in front: the read must follow it without any device sync
        hipStream_t nb = nullptr;
        HIP(hipStreamCreateWithFlags(&nb, hipStreamNonBlocking));
        work.reset_flush_count(nb);
        for (int rep = 0; rep < 40; ++rep) linear(dx.p, xld, m, wt, work, dy.p, yld, nb);
        linear(dxf.p, xld, 4U, wt, work, dy.p, yld, nb);
        const unsigned c1 = work.flush_count(nb);
        work.reset_flush_count(nb);
        const unsigned c0 = work.flush_count(nb);
        // fresh workspace -> immediate out-of-domain call on the non-blocking stream: the counter's initialization must
        // already be complete (repeated, each with a new workspace)
        unsigned fresh_bad = 0;
        for (int rep = 0; rep < 8; ++rep) {
            Workspace fresh(4U, k);
            linear(dxf.p, xld, 4U, wt, fresh, dy.p, yld, nb);
            fresh_bad += fresh.flush_count(nb) != 2U;
        }
        HIP(hipStreamDestroy(nb));
        const auto yf = dy.get();
        std::size_t dfy = 0;
        for (std::uint32_t r = 0; r < 4U; ++r)
            for (std::uint32_t c = 0; c < n; ++c) dfy += yf[static_cast<std::size_t>(r) * yld + c] != hf.y[static_cast<std::size_t>(r) * n + c];
        std::printf("api flush telemetry: device %u groups, host %u groups, after reset %u; outputs vs host %zu differ\n", c1, hf.inexact_groups, c0, dfy);
        std::printf("api flush telemetry: fresh workspaces on a non-blocking stream: %u of 8 miscounted\n", fresh_bad);
        CHECK(c1 == 2U && hf.inexact_groups == 2U && c0 == 0U && dfy == 0 && fresh_bad == 0U, "flush telemetry");
    }
}

// Real checkpoint fc1 (block 0 gate rows 0..255 + up rows 14336..14591 as a 512-row [gate | up] matrix) through the fused
// SwiGLU API vs the host fc1 + swiglu; activations = the captured block-0 qkv input if available (real rows).
bool read_safetensor_rows(const std::string& path, const std::string& name, std::size_t row_bytes, std::size_t first, std::size_t count, void* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::uint64_t hsize = 0;
    f.read(reinterpret_cast<char*>(&hsize), 8);
    std::string header(hsize, '\0');
    f.read(header.data(), static_cast<std::streamsize>(hsize));
    const auto key = header.find("\"" + name + "\"");
    if (key == std::string::npos) return false;
    const auto off = header.find("\"data_offsets\"", key);
    const auto lb = header.find('[', off);
    const std::uint64_t begin = std::strtoull(header.c_str() + lb + 1, nullptr, 10);
    f.seekg(static_cast<std::streamoff>(8 + hsize + begin + first * row_bytes));
    f.read(static_cast<char*>(out), static_cast<std::streamsize>(count * row_bytes));
    return static_cast<bool>(f);
}
void test_real_fc1_swiglu(const char* ckpt, const char* quant_set) {
    constexpr std::uint32_t k = 5376U, half = 256U, n = 2U * half, full_half = 14336U;
    std::vector<std::int8_t> w(static_cast<std::size_t>(n) * k);
    std::vector<float> ws(n);
    bool ok = read_safetensor_rows(ckpt, "blocks.0.mlp.fc1.weight", k, 0, half, w.data()) &&
        read_safetensor_rows(ckpt, "blocks.0.mlp.fc1.weight", k, full_half, half, w.data() + static_cast<std::size_t>(half) * k) &&
        read_safetensor_rows(ckpt, "blocks.0.mlp.fc1.weight_scale", 4, 0, half, ws.data()) &&
        read_safetensor_rows(ckpt, "blocks.0.mlp.fc1.weight_scale", 4, full_half, half, ws.data() + half);
    if (!ok) { std::printf("real fc1: checkpoint not readable, skipped\n"); return; }
    std::vector<std::uint16_t> x;
    if (quant_set) x = load<std::uint16_t>(std::filesystem::path(quant_set) / "K5376" / "real_b0_qkv_in" / "x.bin");
    const bool real_x = !x.empty();
    if (!real_x) x = activations(400U, k, 77U);
    const auto m = static_cast<std::uint32_t>(x.size() / k);
    const auto h_ref = project_fc1_swiglu(x.data(), m, k, w.data(), ws.data(), n);
    const SwiGLUWeights perm = permute_fc1_for_swiglu(w.data(), ws.data(), n, k);
    Dev<std::uint16_t> dx(x), dh(static_cast<std::size_t>(m) * half);
    Dev<std::int8_t> dwp(perm.codes);
    Dev<float> dwsp(perm.scales);
    Workspace work(m, k);
    linear_fc1_swiglu(dx.p, k, m, Weights{dwp.p, dwsp.p, n, k}, work, dh.p, half, nullptr);
    HIP(hipDeviceSynchronize());
    const std::size_t d = ndiff(dh.get(), h_ref);
    std::printf("real fc1 + swiglu (block 0 rows, %s activations M %u): %zu of %zu differ, flush %u\n", real_x ? "captured" : "synthetic", m, d, h_ref.size(), work.flush_count(nullptr));
    CHECK(d == 0 && work.flush_count(nullptr) == 0U, "real fc1 swiglu");
    seen.insert(real_x ? "real_fc1" : "real_fc1_synthetic_x");
}

// Quantiser fixture set: device chain from x == host reference (gate); differences to the set's codes / outputs reported.
void test_quant_set(const std::filesystem::path& root) {
    for (const std::uint32_t k : {5376U, 14336U}) {
        const auto kdir = root / ("K" + std::to_string(k));
        if (!std::filesystem::is_directory(kdir)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(kdir)) {
            const auto dir = entry.path();
            const auto x = load<std::uint16_t>(dir / "x.bin");
            const auto w = load<std::int8_t>(dir / "w_q.bin");
            const auto ws = load<float>(dir / "w_scale.bin");
            const auto m = static_cast<std::uint32_t>(x.size() / k), n = static_cast<std::uint32_t>(ws.size());
            // pad N to one 128-output tile with zero weights
            std::vector<std::int8_t> wp(std::size_t{128} * k, 0);
            std::copy(w.begin(), w.end(), wp.begin());
            std::vector<float> wsp(128, 1.0F);
            std::copy(ws.begin(), ws.end(), wsp.begin());
            const DeviceRun d = run_device<128U>(x, m, k, wp, wsp, 128U, true);
            const Projection p = project(x.data(), m, k, wp.data(), wsp.data(), 128U);
            std::vector<std::uint16_t> y(std::size_t{m} * n);
            std::vector<std::int32_t> acc(y.size());
            for (std::uint32_t r = 0; r < m; ++r)
                for (std::uint32_t c = 0; c < n; ++c) { y[std::size_t{r} * n + c] = d.y[std::size_t{r} * 128U + c]; acc[std::size_t{r} * n + c] = d.acc[std::size_t{r} * 128U + c]; }
            const std::size_t host[4] = {ndiff(d.q, p.q), ndiff(d.scale, p.row_scale), ndiff(d.acc, p.acc), ndiff(d.y, p.y)};
            const std::size_t set[4] = {ndiff(d.q, load<std::int8_t>(dir / "q.bin")), ndiff(d.scale, load<float>(dir / "row_scale.bin")),
                ndiff(acc, load<std::int32_t>(dir / "acc.bin")), ndiff(y, load<std::uint16_t>(dir / "out.bin"))};
            const std::string name = "K" + std::to_string(k) + "/" + dir.filename().string();
            std::printf("quant set %-22s M %3u: GPU vs host codes %zu scales %zu acc %zu out %zu | GPU vs set codes %zu / %zu, scales %zu, acc %zu, out %zu / %zu\n",
                name.c_str(), m, host[0], host[1], host[2], host[3], set[0], d.q.size(), set[1], set[2], set[3], y.size());
            CHECK(host[0] == 0 && host[1] == 0 && host[2] == 0 && host[3] == 0, "quant set %s: GPU != host", name.c_str());
            g1_assert("quant " + name, x, m, k, load<std::uint16_t>(dir / "x_rot.bin"), set[0], d.q.size());
            seen.insert("quant/" + name);
        }
    }
}

// Quantizer variants (baseline int64 / fast int32-when-narrow) must give identical codes, scales and flush counts.
struct VariantStats { std::size_t rows = 0, values = 0, diffs = 0; unsigned flush0 = 0, flush1 = 0; };
VariantStats variant_stats;
void compare_variants(const std::string& name, const std::vector<std::uint16_t>& x, std::uint32_t m, std::uint32_t k) {
    Dev<std::uint16_t> dx(x);
    Dev<std::int8_t> q0(static_cast<std::size_t>(m) * k), q1(static_cast<std::size_t>(m) * k);
    Dev<float> s0(m), s1(m);
    Dev<unsigned> f0(1), f1(1);
    HIP(hipMemset(f0.p, 0, 4)); HIP(hipMemset(f1.p, 0, 4));
    launch_quantize(dx.p, k, m, k, q0.p, s0.p, f0.p, nullptr, 0);
    launch_quantize(dx.p, k, m, k, q1.p, s1.p, f1.p, nullptr, 1);
    HIP(hipDeviceSynchronize());
    const std::size_t d = ndiff(q0.get(), q1.get()) + ndiff(s0.get(), s1.get());
    const unsigned a = f0.get()[0], b = f1.get()[0];
    variant_stats.rows += m; variant_stats.values += x.size(); variant_stats.diffs += d; variant_stats.flush0 += a; variant_stats.flush1 += b;
    CHECK(d == 0 && a == b, "quantizer variants differ on %s (%zu diffs, flush %u vs %u)", name.c_str(), d, a, b);
}
void test_quant_variants(const char* dumps, const char* qs) {
    for (const std::uint32_t k : {256U, 1280U, 5376U, 7168U, 14336U, 16384U})
        for (const std::uint32_t m : {1U, 77U, 300U}) compare_variants("random K " + std::to_string(k), activations(m, k, 1000U + k + m), m, k);
    {   // extreme groups: carry group, subnormals, huge/tiny mixes (flushed), spans around 15 and 47 binades
        constexpr std::uint32_t k = 4096U, m = 64U;
        std::vector<std::uint16_t> x(static_cast<std::size_t>(m) * k, 0);
        SplitMix g{4711};
        for (std::uint32_t r = 0; r < m; ++r)
            for (std::uint32_t i = 0; i < k; ++i) {
                const std::uint64_t u = g.next();
                const unsigned span = (r % 8U) * 8U + (i / 256U) % 4U;  // 0..59 binades
                const unsigned base = 100U + (r * 7U) % 40U, e = base + static_cast<unsigned>((u >> 40U) % (span + 1U));
                x[static_cast<std::size_t>(r) * k + i] = (u & 3U) == 0U ? 0U : static_cast<std::uint16_t>(((u >> 63U) << 15U) | (std::min(e, 254U) << 7U) | ((u >> 8U) & 127U));
            }
        for (std::uint32_t i = 0; i < 256U; ++i) x[i] = 0;
        x[0] = 0x437fU; x[1] = 0x4380U;                                           // the carry group
        for (std::uint32_t i = 256U; i < 512U; ++i) x[i] = static_cast<std::uint16_t>(i & 0x7fU);  // subnormals only
        compare_variants("extreme groups", x, m, k);
    }
    const auto from_dir = [&](const std::filesystem::path& file, std::uint32_t k, const std::string& name) {
        const auto x = load<std::uint16_t>(file);
        if (!x.empty() && x.size() % k == 0) compare_variants(name, x, static_cast<std::uint32_t>(x.size() / k), k);
    };
    if (dumps && std::filesystem::is_directory(dumps))
        for (const auto& entry : std::filesystem::directory_iterator(dumps)) {
            const auto dir = entry.path();
            if (!std::filesystem::exists(dir / "meta.txt")) continue;
            std::ifstream f(dir / "meta.txt");
            std::string nm; std::size_t r = 0, kk = 0;
            while (f >> nm) { if (nm == "x") { f >> r >> kk; break; } std::string rest; std::getline(f, rest); }
            if (kk % 256U == 0U && kk) from_dir(dir / "x.bin", static_cast<std::uint32_t>(kk), "dump " + dir.filename().string());
        }
    if (qs && std::filesystem::is_directory(qs))
        for (const std::uint32_t k : {5376U, 14336U}) {
            const auto kdir = std::filesystem::path(qs) / ("K" + std::to_string(k));
            if (!std::filesystem::is_directory(kdir)) continue;
            for (const auto& entry : std::filesystem::directory_iterator(kdir)) from_dir(entry.path() / "x.bin", k, "quant " + entry.path().filename().string());
        }
    std::printf("quantizer variants: %zu rows, %zu values, %zu differences, flushed groups %u (baseline) / %u (fast)\n", variant_stats.rows,
        variant_stats.values, variant_stats.diffs, variant_stats.flush0, variant_stats.flush1);
    seen.insert("quant_variants");
}

// ------------------------------------------------------------------------------------------------------------------ bench
float time_ms(const std::function<void()>& f, int reps) {
    hipEvent_t e0, e1;
    HIP(hipEventCreate(&e0)); HIP(hipEventCreate(&e1));
    f(); HIP(hipDeviceSynchronize());
    std::vector<float> v;
    for (int r = 0; r < reps; ++r) {
        HIP(hipEventRecord(e0)); f(); HIP(hipEventRecord(e1)); HIP(hipEventSynchronize(e1));
        float ms; HIP(hipEventElapsedTime(&ms, e0, e1)); v.push_back(ms);
    }
    std::sort(v.begin(), v.end());
    static_cast<void>(hipEventDestroy(e0)); static_cast<void>(hipEventDestroy(e1));
    return v[v.size() / 2];
}
void bench(unsigned reps, unsigned bn_sel, unsigned raster, const char* dumps) {
    struct Shape { const char* name; std::uint32_t n, k; bool swiglu; };
    const Shape shapes[] = {{"qkv", 21504U, 5376U, false}, {"out", 5376U, 7168U, false}, {"fc1", 28672U, 5376U, false},
        {"fc1+swiglu", 28672U, 5376U, true}, {"fc2", 5376U, 14336U, false}};
    constexpr std::uint32_t kMaxM = 38000U, kMaxK = 14336U, kMaxN = 28672U;
    Dev<std::uint16_t> dx(static_cast<std::size_t>(kMaxM) * kMaxK), dy(static_cast<std::size_t>(kMaxM) * kMaxN);
    Dev<std::int8_t> dq(static_cast<std::size_t>(kMaxM) * kMaxK), dw(static_cast<std::size_t>(kMaxN) * 5376U);
    Dev<float> ds(kMaxM), dws(kMaxN);
    std::vector<std::uint16_t> table(silu_table().begin(), silu_table().end());
    Dev<std::uint16_t> dtab(table);
    {   // fill: activations ~N(0,1)-like bf16 (one row pattern repeated), weights / scales random
        const auto xr = activations(64U, kMaxK, 5U);
        for (std::uint32_t r = 0; r < kMaxM; r += 64U)
            HIP(hipMemcpy(dx.p + static_cast<std::size_t>(r) * kMaxK, xr.data(), std::min<std::size_t>(64U, kMaxM - r) * kMaxK * 2U, hipMemcpyHostToDevice));
        const auto w = weights(kMaxN, 5376U, 6U);
        HIP(hipMemcpy(dw.p, w.data(), w.size(), hipMemcpyHostToDevice));
        const auto s = channel_scales(kMaxN, 7U);
        HIP(hipMemcpy(dws.p, s.data(), s.size() * 4U, hipMemcpyHostToDevice));
    }
    Workspace work(kMaxM, kMaxK);
    if (dumps) {  // quantizer on wide-span SwiGLU rows (fc2_wide_synth, 400 rows repeated): the int64 path dominates there
        const auto xw = load<std::uint16_t>(std::filesystem::path(dumps) / "fc2_wide_synth" / "x.bin");
        if (xw.size() == 400U * 14336U) {
            for (std::uint32_t r = 0; r < 19000U; r += 400U)
                HIP(hipMemcpy(dx.p + static_cast<std::size_t>(r) * 14336U, xw.data(), std::min<std::size_t>(400U, 19000U - r) * 14336U * 2U, hipMemcpyHostToDevice));
            const float t0 = time_ms([&] { launch_quantize(dx.p, 14336U, 19000U, 14336U, dq.p, ds.p, nullptr, nullptr, 0); }, static_cast<int>(reps));
            const float t1 = time_ms([&] { launch_quantize(dx.p, 14336U, 19000U, 14336U, dq.p, ds.p, nullptr, nullptr, 1); }, static_cast<int>(reps));
            std::printf("quant wide-span fc2 rows M 19000 K 14336: baseline %7.3f ms, fast %7.3f ms (%.2fx)\n", t0, t1, t0 / t1);
            const auto xr = activations(64U, kMaxK, 5U);  // restore the synthetic fill
            for (std::uint32_t r = 0; r < kMaxM; r += 64U)
                HIP(hipMemcpy(dx.p + static_cast<std::size_t>(r) * kMaxK, xr.data(), std::min<std::size_t>(64U, kMaxM - r) * kMaxK * 2U, hipMemcpyHostToDevice));
        }
    }
    for (const std::uint32_t m : {19000U, 38000U})
        for (const auto& sh : shapes) {
            const float tq = time_ms([&] { launch_quantize(dx.p, sh.k, m, sh.k, dq.p, ds.p, nullptr, nullptr, 0); }, static_cast<int>(reps));
            const float tq1 = time_ms([&] { launch_quantize(dx.p, sh.k, m, sh.k, dq.p, ds.p, nullptr, nullptr, 1); }, static_cast<int>(reps));
            std::printf("quant M %5u K %5u: baseline %7.3f ms, fast %7.3f ms (%.2fx)\n", m, sh.k, tq, tq1, tq / tq1);
            dev::GemmArgs a;
            a.a = dq.p; a.b = dw.p; a.row_scale = ds.p; a.ch_scale = dws.p; a.silu = dtab.p; a.m = m; a.n = sh.n; a.k = sh.k; a.raster_m = raster;
            const std::uint32_t outc = sh.swiglu ? sh.n / 2U : sh.n;
            a.out.ptr[0] = dy.p; a.out.ld = outc; a.out.seg_cols = outc;
            const double ops = 2.0 * m * sh.n * sh.k;
            for (const unsigned bn : {128U, 256U}) {
                if (bn_sel && bn != bn_sel) continue;
                const float tg = time_ms([&] {
                    if (bn == 128U) { if (sh.swiglu) launch_gemm<128U, dev::Epi::swiglu>(a, nullptr); else launch_gemm<128U, dev::Epi::plain>(a, nullptr); }
                    else { if (sh.swiglu) launch_gemm<256U, dev::Epi::swiglu>(a, nullptr); else launch_gemm<256U, dev::Epi::plain>(a, nullptr); }
                }, static_cast<int>(reps));
                std::printf("bench M %5u %-11s N %5u K %5u BN %u raster %u: quantize %7.3f ms (%5.1f GB/s), gemm %8.3f ms = %5.2f TOPS, both %5.2f TOPS\n", m,
                    sh.name, sh.n, sh.k, bn, raster, tq, 3.0 * m * sh.k / (tq * 1e6), tg, ops / (tg * 1e9), ops / ((tg + tq) * 1e9));
            }
            for (const int variant : {0, 1}) {   // end to end through the public API (quantize + GEMM, the API's tile choice)
                set_quantizer_variant(variant);
                const Weights wt{dw.p, dws.p, sh.n, sh.k};
                const float te = time_ms([&] {
                    if (sh.swiglu) linear_fc1_swiglu(dx.p, sh.k, m, wt, work, dy.p, sh.n / 2U, nullptr);
                    else linear(dx.p, sh.k, m, wt, work, dy.p, sh.n, nullptr);
                }, static_cast<int>(reps));
                std::printf("e2e   M %5u %-11s quant %-8s quantize + GEMM %8.3f ms = %5.2f TOPS\n", m, sh.name, variant ? "fast" : "baseline", te, ops / (te * 1e9));
            }
            set_quantizer_variant(kQuantVariantAuto);
        }
}
}  // namespace

// Usage: gpu_tests [dumps_dir] [--qualify] [--bench] [--reps=N] [--bn=128|256] [--raster=N]
// Environment: STRIX_H3_GEMM_DUMPS (dumps dir), STRIX_H3_QUANT_SET (quantiser fixture set), H3_CKPT (int8 convrot checkpoint).
// --qualify: the dumps (6 cases incl. silu is host-only, so 5 here), the 8 quant families and the real-checkpoint fc1
// case with captured activations are required; anything missing fails.
int main(int argc, char** argv) {
    bool do_bench = false, qualify = false;
    unsigned reps = 5, bn = 0, raster = 8;
    const char* dumps = std::getenv("STRIX_H3_GEMM_DUMPS");
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--bench") do_bench = true;
        else if (s == "--qualify") qualify = true;
        else if (s.rfind("--reps=", 0) == 0) reps = static_cast<unsigned>(std::atoi(s.c_str() + 7));
        else if (s.rfind("--bn=", 0) == 0) bn = static_cast<unsigned>(std::atoi(s.c_str() + 5));
        else if (s.rfind("--raster=", 0) == 0) raster = static_cast<unsigned>(std::atoi(s.c_str() + 9));
        else dumps = argv[i];
    }
    const char* qs = std::getenv("STRIX_H3_QUANT_SET");
    const char* ckpt = std::getenv("H3_CKPT");
    const auto t0 = std::chrono::steady_clock::now();
    test_projection<128U>(333U, 5376U, 384U, 11U);
    test_projection<256U>(333U, 5376U, 512U, 12U);
    test_projection<128U>(70U, 14336U, 128U, 13U);
    test_projection<256U>(300U, 7168U, 256U, 14U);
    test_projection<256U>(70U, 14336U, 256U, 15U);    // BN 256 at K 14336
    test_projection<128U>(1U, 5376U, 128U, 16U);      // M = 1
    test_projection<256U>(1U, 14336U, 256U, 17U);
    test_projection<128U>(2400U, 512U, 256U, 18U);    // 10 row tiles: grouped raster crosses its 8-tile groups
    test_projection<256U>(2305U, 256U, 512U, 19U);    // 10 row tiles, 2 output tiles, tail of 1 row
    test_swiglu_and_segments<128U>(259U, 21U);
    test_swiglu_and_segments<256U>(259U, 22U);
    test_swiglu_and_segments<256U>(2100U, 23U);       // 9 row tiles
    test_batch_invariance();
    test_api();
    if (dumps && std::filesystem::is_directory(dumps)) test_dumps(dumps);
    else std::printf("dumps: skipped\n");
    if (qs && std::filesystem::is_directory(qs)) test_quant_set(qs);
    else std::printf("quant set: skipped (STRIX_H3_QUANT_SET)\n");
    if (ckpt && std::filesystem::exists(ckpt)) test_real_fc1_swiglu(ckpt, qs && std::filesystem::is_directory(qs) ? qs : nullptr);
    else std::printf("real fc1: skipped (H3_CKPT)\n");
    test_quant_variants(dumps, qs);
    if (qualify) {
        std::vector<std::string> need = {"quant_variants", "dump/qkv_real", "dump/out_real", "dump/fc1_real", "dump/fc2_real", "dump/random_wide", "dump/fc2_wide_synth", "real_fc1"};
        for (const char* f : {"normal", "heavy", "edge"}) { need.push_back(std::string("quant/K5376/") + f); need.push_back(std::string("quant/K14336/") + f); }
        need.push_back("quant/K5376/real_b0_qkv_in");
        need.push_back("quant/K14336/real_b0_fc2_in");
        std::size_t present = 0;
        for (const auto& nm : need) {
            present += seen.count(nm);
            CHECK(seen.count(nm) == 1, "qualification: %s missing", nm.c_str());
        }
        std::printf("qualification: %zu of %zu required cases present\n", present, need.size());
    }
    std::printf("correctness %s (%d failures, %.1f s)\n", failures ? "FAILED" : "PASSED", failures,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (do_bench) bench(reps, bn, raster, dumps);
    return failures ? 1 : 0;
}
