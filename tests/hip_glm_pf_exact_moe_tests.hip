// Exact MoE admission gate: qualified decode seam R=1, including every pair intermediate.
#include "glm/prefill/glm_pf_exact_moe.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace M = strix::glm::moe;
namespace X = strix::glm::pf::exact_moe;
namespace {
constexpr unsigned H = 4096, I = 1024, E = 288, Top = 8, MaxR = 4096;
unsigned checks = 0, failures = 0;
std::string only_versions;  // --versions a,b,...: gate/exception arms limited to these (split long gates)
bool selected(const char* version) {
    return only_versions.empty() || ("," + only_versions + ",").find(std::string(",") + version + ",") != std::string::npos;
}
void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void check(bool pass, const std::string& name) {
    ++checks;
    failures += pass ? 0U : 1U;
    std::printf("%s %s\n", pass ? "OK  " : "FAIL", name.c_str());
}
// enqueue reads this setting per call, so every fixture exercises all four paths.
struct Variant {
    std::string saved;
    bool present;
    Variant() : present(std::getenv("STRIX_GLM_PF_EXACTMOE_V") != nullptr) {
        if (present) saved = std::getenv("STRIX_GLM_PF_EXACTMOE_V");
    }
    static void set(const char* value) {
#ifdef _WIN32
        if (_putenv_s("STRIX_GLM_PF_EXACTMOE_V", value ? value : "") != 0)
#else
        if ((value ? setenv("STRIX_GLM_PF_EXACTMOE_V", value, 1) : unsetenv("STRIX_GLM_PF_EXACTMOE_V")) != 0)
#endif
            throw std::runtime_error("setting exact MoE variant");
    }
    ~Variant() {
        try { set(present ? saved.c_str() : nullptr); } catch (...) {}
    }
};
void rcap_env(const char* value) {
#ifdef _WIN32
    if (_putenv_s("STRIX_GLM_PF_EXACTMOE_V3_RCAP", value ? value : "") != 0)
#else
    if ((value ? setenv("STRIX_GLM_PF_EXACTMOE_V3_RCAP", value, 1) : unsetenv("STRIX_GLM_PF_EXACTMOE_V3_RCAP")) != 0)
#endif
        throw std::runtime_error("setting v3 row buckets");
}
struct Arena {
    std::vector<void*> p;
    ~Arena() { for (void* v : p) (void)hipFree(v); }
    template<class T> T* alloc(std::size_t n) {
        T* v = nullptr;
        ok(hipMalloc(&v, n * sizeof(T)), "alloc");
        p.push_back(v);
        return v;
    }
};
std::uint16_t half(float x) {
    const _Float16 h = static_cast<_Float16>(x);
    std::uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}
M::Exl3 make_matrix(Arena& mem, std::mt19937& rng, unsigned K, unsigned N, unsigned bits) {
    std::vector<std::uint32_t> tr(std::size_t{K} * N * bits / 32);
    for (auto& v : tr) v = static_cast<std::uint32_t>(rng());
    std::vector<std::uint16_t> suh(K), svh(N);
    // Deliberately different suh for every expert/matrix: no accidental scale sharing.
    for (auto& v : suh) v = half((rng() & 1U ? 1.0F : -1.0F) * (0.5F + static_cast<float>(rng() % 1024) / 2048.0F));
    for (auto& v : svh) v = half(rng() & 1U ? 0.02F : -0.02F);
    auto* dt = mem.alloc<std::uint32_t>(tr.size());
    auto* ds = mem.alloc<std::uint16_t>(K);
    auto* dv = mem.alloc<std::uint16_t>(N);
    ok(hipMemcpy(dt, tr.data(), tr.size() * 4, hipMemcpyHostToDevice), "trellis");
    ok(hipMemcpy(ds, suh.data(), K * 2, hipMemcpyHostToDevice), "suh");
    ok(hipMemcpy(dv, svh.data(), N * 2, hipMemcpyHostToDevice), "svh");
    return {dt, ds, dv, K, N, bits};
}
M::ExpertSlices make_expert(Arena& mem, std::mt19937& rng, unsigned bits) {
    return {make_matrix(mem, rng, H, I, bits), make_matrix(mem, rng, H, I, bits), make_matrix(mem, rng, I, H, bits)};
}
struct Routes {
    std::vector<std::uint32_t> ids;
    std::vector<float> weights;
};
Routes routes(unsigned R, unsigned mode, std::mt19937& rng) {
    std::vector<std::vector<unsigned>> rows(R);
    if (mode == 1) for (auto& row : rows) row = {0, 1, 2, 3, 4, 5, 6, 287};
    if (mode == 2) {
        // Exact group boundaries; filler never uses these expert ids.
        const unsigned counts[] = {1, 15, 16, 17, 31, 32, 33};
        unsigned expert = 0;
        for (unsigned count : counts) {
            if (count <= R) for (unsigned r = 0; r < count; ++r) rows[r].push_back(expert);
            ++expert;
        }
    }
    if (mode == 4) {
        // Disjoint assignments ensure no row exceeds Top while every group tail is present.
        unsigned first = 0;
        for (unsigned count = 1; count <= 33; ++count) {
            for (unsigned r = first; r < first + count; ++r) rows[r].push_back(count - 1);
            first += count;
        }
    }
    if (mode >= 5) {
        const unsigned count = 2047U + mode - 5U;
        for (unsigned r = 0; r < count; ++r) rows[r].push_back(0);
    }
    std::vector<double> popularity(E);
    std::normal_distribution<double> normal(0.0, 1.2);
    for (auto& p : popularity) p = std::exp(normal(rng));
    std::discrete_distribution<unsigned> lognormal(popularity.begin(), popularity.end());
    for (auto& row : rows) {
        while (row.size() < Top) {
            const unsigned e = mode == 4 ? 33U + static_cast<unsigned>(rng() % (E - 33)) :
                               mode >= 5 ? 1U + static_cast<unsigned>(rng() % (E - 1)) :
                               mode == 2 ? 7U + static_cast<unsigned>(rng() % (E - 7)) :
                               mode == 3 ? lognormal(rng) : static_cast<unsigned>(rng() % E);
            if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
        }
        std::sort(row.begin(), row.end());
    }
    Routes out;
    for (const auto& row : rows) for (unsigned e : row) {
        out.ids.push_back(e);
        out.weights.push_back(0.02F + static_cast<float>(rng() % 1000) * 0.0005F);
    }
    return out;
}
std::uint32_t bits(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return u; }
std::uint32_t ordered(std::uint32_t u) { return (u & 0x80000000U) ? ~u : u ^ 0x80000000U; }
bool equal32(const std::vector<float>& a, const std::vector<float>& b, const std::string& name) {
    std::size_t count = 0, first = 0;
    std::uint32_t ulp = 0;
    bool finite = true;
    for (std::size_t i = 0; i < a.size(); ++i) {
        finite = finite && std::isfinite(a[i]) && std::isfinite(b[i]);
        const auto u = bits(a[i]), v = bits(b[i]);
        if (u != v) {
            if (!count) first = i;
            ++count;
            const auto x = ordered(u), y = ordered(v);
            ulp = std::max(ulp, x > y ? x - y : y - x);
        }
    }
    if (count) std::printf("FAIL %s mismatches=%zu first=%zu ref=%08x got=%08x max_ulp=%u\n",
                           name.c_str(), count, first, bits(a[first]), bits(b[first]), ulp);
    check(count == 0 && finite, name);
    return count == 0 && finite;
}
// Capture the R=1 seam's expert-sorted pairs into the chunk's fixed row/slot layout.
__global__ void capture(const float* src_o, const std::uint16_t* src_a, float* dst_o,
                        std::uint16_t* dst_a, const float* src_gu, float* dst_gu, unsigned pair) {
    const unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i < H) dst_o[std::size_t{pair} * H + i] = src_o[i];
    if (i < 2U * I) dst_gu[std::size_t{pair} * 2U * I + i] = src_gu[i];
    if (i < I) dst_a[std::size_t{pair} * I + i] = src_a[i];
}
struct Fixture {
    Arena model_mem, scratch;
    std::array<M::ExpertSlices, E> experts;
    M::ExpertSlices shared;
    const M::ExpertSlices* device_experts;
    hipStream_t stream{};
    X::Plan plan{MaxR};
    void* ws;
    std::size_t wsb = X::workspace_bytes(MaxR);
    M::MoeWorkspace ref_ws;
    float *x, *weights, *out, *ref_out, *ref_pairs, *ref_gu, *debug_gu;
    std::uint32_t* ids;
    std::uint16_t* ref_at;
    M::ExpertSlices* device_table;  // [289] for the device plan (entry 288 = shared)
    unsigned* flag;
    Fixture(std::mt19937& rng) {
        ok(hipStreamCreate(&stream), "stream");
        for (auto& e : experts) e = make_expert(model_mem, rng, 4);
        shared = make_expert(model_mem, rng, 6);
        auto* table = model_mem.alloc<M::ExpertSlices>(E);
        ok(hipMemcpy(table, experts.data(), sizeof(experts), hipMemcpyHostToDevice), "table");
        device_experts = table;
        ws = scratch.alloc<unsigned char>(wsb);
        ref_ws.base = scratch.alloc<unsigned char>(M::moe_workspace_bytes());
        ok(hipMemset(ref_ws.base, 0, M::moe_workspace_bytes()), "zero counters");
        x = scratch.alloc<float>(std::size_t{MaxR} * H);
        weights = scratch.alloc<float>(std::size_t{MaxR} * Top);
        ids = scratch.alloc<std::uint32_t>(std::size_t{MaxR} * Top);
        out = scratch.alloc<float>(std::size_t{MaxR} * H);
        ref_out = scratch.alloc<float>(std::size_t{MaxR} * H);
        ref_pairs = scratch.alloc<float>(std::size_t{MaxR} * 9 * H);
        ref_gu = scratch.alloc<float>(std::size_t{MaxR} * 9 * 2 * I);
        debug_gu = scratch.alloc<float>(std::size_t{M::kMaxPairs} * 2 * I);
        ref_at = scratch.alloc<std::uint16_t>(std::size_t{MaxR} * 9 * I);
        device_table = scratch.alloc<M::ExpertSlices>(E + 1);
        flag = scratch.alloc<unsigned>(1);
    }
    ~Fixture() { (void)hipStreamSynchronize(stream); (void)hipStreamDestroy(stream); }
    void upload(unsigned R, const Routes& rt, std::mt19937& rng) {
        std::vector<float> host(std::size_t{R} * H);
        std::normal_distribution<float> normal(0, 1);
        for (auto& v : host) v = normal(rng);
        // Include +/- zero, FP32 and FP16 subnormals and rounding-boundary inputs.
        const std::uint32_t edge[] = {0, 0x80000000U, 1, 0x80000001U, 0x007fffffU, 0x33800000U, 0xb3800000U, 0x3f801000U};
        for (std::size_t i = 0; i < host.size(); i += 97) {
            const auto u = edge[(i / 97) % 8]; std::memcpy(&host[i], &u, 4);
        }
        for (unsigned r = 0; R >= 7 && r < 3; ++r) for (unsigned c = 0; c < H; ++c) {
            std::uint32_t u = r == 0 ? 0U : r == 1 ? 1U : 0x33800000U;
            if (c & 1U) u |= 0x80000000U;
            std::memcpy(&host[std::size_t{r} * H + c], &u, 4);
        }
        ok(hipMemcpy(x, host.data(), host.size() * 4, hipMemcpyHostToDevice), "x");
        ok(hipMemcpy(ids, rt.ids.data(), rt.ids.size() * 4, hipMemcpyHostToDevice), "ids");
        ok(hipMemcpy(weights, rt.weights.data(), rt.weights.size() * 4, hipMemcpyHostToDevice), "weights");
    }
    X::View view(unsigned R, const Routes& rt) {
        return {R, experts.data(), shared, x, rt.ids.data(), rt.weights.data(), weights, out};
    }
    void reference(unsigned R, unsigned block, bool diagnostics) {
        for (unsigned r = 0; r < R; r += block) {
            M::MoeView v;
            v.rows = std::min(block, R - r);
            v.y = x + std::size_t{r} * H;
            v.ids = ids + r * Top;
            v.weights = weights + r * Top;
            v.experts = device_experts;
            v.shared = shared;
            v.out = ref_out + std::size_t{r} * H;
            v.ws = ref_ws;
            v.debug_gate_up = diagnostics ? debug_gu : nullptr;
            M::enqueue_experts(v, stream);
            if (diagnostics) for (unsigned slot = 0; slot < 9; ++slot) {
                const unsigned p = slot == 8 ? 0 : slot + 1;
                capture<<<H / 256, 256, 0, stream>>>(M::moe_pair_output(ref_ws, p), M::moe_pair_down_input(ref_ws, p),
                                                     ref_pairs, ref_at, debug_gu + std::size_t{p} * 2 * I, ref_gu, r * 9 + slot);
                ok(hipGetLastError(), "capture");
            }
        }
    }
    bool gate(unsigned R, const Routes& rt, const std::string& label, std::mt19937& rng, bool added = false) {
        upload(R, rt, rng);
        reference(R, 1, true);
        ok(hipStreamSynchronize(stream), "reference");
        std::vector<float> yr(std::size_t{R} * H), pr(std::size_t{R} * 9 * H);
        std::vector<std::uint16_t> ar(std::size_t{R} * 9 * I);
        ok(hipMemcpy(yr.data(), ref_out, yr.size() * 4, hipMemcpyDeviceToHost), "ref out");
        ok(hipMemcpy(pr.data(), ref_pairs, pr.size() * 4, hipMemcpyDeviceToHost), "ref pairs");
        ok(hipMemcpy(ar.data(), ref_at, ar.size() * 2, hipMemcpyDeviceToHost), "ref at");
        std::vector<float> gr(std::size_t{R} * 9 * 2 * I), gu(gr.size());
        ok(hipMemcpy(gr.data(), ref_gu, gr.size() * 4, hipMemcpyDeviceToHost), "ref gate/up");
        std::vector<float> y(yr.size()), pairs(pr.size());
        std::vector<std::uint16_t> at(ar.size());
        bool pass = true;
        Variant variant;
        for (const char* version : {"1", "2", "3", "3r0", "6", "7", "8", "9", "10", "11", "14", "auto"}) for (unsigned tm : {16U, 32U}) for (unsigned repeat = 0; repeat < 2; ++repeat) {
            // Existing checks retain every arm/repeat. Added large cases qualify v3/v6/v7/v8/v9 against the seam.
            if (!selected(version)) continue;
            if (added && (std::strcmp(version, "3r0") == 0 || std::strcmp(version, "1") == 0 || std::strcmp(version, "2") == 0 || tm != 16 || repeat != 0)) continue;
            // 3r0 = v3 with its row buckets off (STRIX_GLM_PF_EXACTMOE_V3_RCAP=0, the previous fixed 16 rows).
            const bool r0 = std::strcmp(version, "3r0") == 0;
            Variant::set(r0 ? "3" : version);
            rcap_env(r0 ? "0" : nullptr);
            ok(hipMemsetAsync(ws, 0xff, wsb, stream), "poison workspace");
            ok(hipMemsetAsync(out, 0xff, std::size_t{MaxR} * H * 4, stream), "poison output");
            X::enqueue(view(R, rt), plan, ws, wsb, stream, tm);
            ok(hipStreamSynchronize(stream), "exact");
            ok(hipMemcpy(y.data(), out, y.size() * 4, hipMemcpyDeviceToHost), "out");
            ok(hipMemcpy(pairs.data(), X::pair_output(ws, MaxR, 0), pairs.size() * 4, hipMemcpyDeviceToHost), "pairs");
            ok(hipMemcpy(gu.data(), X::pair_gate_up(ws, MaxR, 0), gu.size() * 4, hipMemcpyDeviceToHost), "gate/up");
            ok(hipMemcpy(at.data(), X::pair_down_input(ws, MaxR, 0), at.size() * 2, hipMemcpyDeviceToHost), "at");
            if (R < MaxR) {
                std::uint32_t tail_out = 0, tail_pair = 0, tail_gu = 0;
                std::uint16_t tail_at = 0;
                ok(hipMemcpy(&tail_out, out + std::size_t{R} * H, 4, hipMemcpyDeviceToHost), "tail out");
                ok(hipMemcpy(&tail_pair, X::pair_output(ws, MaxR, R * 9), 4, hipMemcpyDeviceToHost), "tail pair");
                ok(hipMemcpy(&tail_at, X::pair_down_input(ws, MaxR, R * 9), 2, hipMemcpyDeviceToHost), "tail at");
                ok(hipMemcpy(&tail_gu, X::pair_gate_up(ws, MaxR, R * 9), 4, hipMemcpyDeviceToHost), "tail gate/up");
                const bool untouched = tail_gu == 0xffffffffU && tail_out == 0xffffffffU && tail_pair == 0xffffffffU && tail_at == 0xffffU;
                check(untouched, label + " V=" + version + " padded rows untouched TM=" + std::to_string(tm));
                pass = pass && untouched;
            }
            const auto name = label + " V=" + version + " TM=" + std::to_string(tm) + " repeat=" + std::to_string(repeat);
            const bool out_pass = equal32(yr, y, name + " R16");
            const bool gu_pass = equal32(gr, gu, name + " gate/up");
            const bool pair_pass = equal32(pr, pairs, name + " pair-output");
            std::size_t mismatches = 0, first = 0;
            unsigned max_ulp = 0;
            for (std::size_t i = 0; i < at.size(); ++i) if (at[i] != ar[i]) {
                if (!mismatches) first = i;
                ++mismatches;
                const unsigned a = ar[i] & 0x8000U ? static_cast<unsigned>(~ar[i] & 0xffffU) : ar[i] ^ 0x8000U;
                const unsigned b = at[i] & 0x8000U ? static_cast<unsigned>(~at[i] & 0xffffU) : at[i] ^ 0x8000U;
                max_ulp = std::max(max_ulp, a > b ? a - b : b - a);
            }
            if (mismatches) std::printf("FAIL down-input mismatches=%zu first=%zu ref=%04x got=%04x max_ulp=%u\n",
                                        mismatches, first, static_cast<unsigned>(ar[first]), static_cast<unsigned>(at[first]), max_ulp);
            check(mismatches == 0, name + " down-input FP16");
            pass = pass && gu_pass && out_pass && pair_pass && mismatches == 0;
            if (tm == 16 && repeat == 0 && X::device_plan(R, 16)) pass = device_arm(R, rt, name, yr, gr, pr, ar) && pass;
        }
        return pass;
    }
    // Device plan for the same call: plan bytes equal the host plan just built, outputs equal the seam, flag clear.
    bool device_arm(unsigned R, const Routes& rt, const std::string& name, const std::vector<float>& yr,
                    const std::vector<float>& gr, const std::vector<float>& pr, const std::vector<std::uint16_t>& ar) {
        const std::size_t np = plan.pairs.size(), nt = plan.tiles.size();
        std::vector<unsigned> hp(np);
        std::vector<X::Tile> ht(nt), hg(E + 1);
        ok(hipMemcpy(hp.data(), X::plan_pairs(ws, MaxR), np * 4, hipMemcpyDeviceToHost), "host pairs");
        ok(hipMemcpy(ht.data(), X::plan_tiles(ws, MaxR), nt * sizeof(X::Tile), hipMemcpyDeviceToHost), "host tiles");
        ok(hipMemcpy(hg.data(), X::plan_groups(ws, MaxR), hg.size() * sizeof(X::Tile), hipMemcpyDeviceToHost), "host groups");
        std::vector<M::ExpertSlices> table(experts.begin(), experts.end());
        table.push_back(shared);
        ok(hipMemcpy(device_table, table.data(), table.size() * sizeof(M::ExpertSlices), hipMemcpyHostToDevice), "device table");
        ok(hipMemsetAsync(ws, 0xff, wsb, stream), "device poison workspace");
        ok(hipMemsetAsync(out, 0xff, std::size_t{MaxR} * H * 4, stream), "device poison output");
        ok(hipMemsetAsync(flag, 0, 4, stream), "flag");
        X::DeviceView dv;
        dv.rows = R; dv.table = device_table; dv.shared = shared; dv.x = x; dv.ids = ids; dv.weights = weights; dv.out = out;
        X::enqueue_device(dv, plan, ws, wsb, stream, flag);
        ok(hipStreamSynchronize(stream), "device plan");
        (void)rt;
        std::vector<unsigned> dp(np);
        std::vector<X::Tile> dt(nt), dg(E + 1);
        unsigned f = 1;
        ok(hipMemcpy(dp.data(), X::plan_pairs(ws, MaxR), np * 4, hipMemcpyDeviceToHost), "device pairs");
        ok(hipMemcpy(dt.data(), X::plan_tiles(ws, MaxR), nt * sizeof(X::Tile), hipMemcpyDeviceToHost), "device tiles");
        ok(hipMemcpy(dg.data(), X::plan_groups(ws, MaxR), dg.size() * sizeof(X::Tile), hipMemcpyDeviceToHost), "device groups");
        ok(hipMemcpy(&f, flag, 4, hipMemcpyDeviceToHost), "flag");
        const bool plan_same = dp == hp && std::memcmp(dt.data(), ht.data(), nt * sizeof(X::Tile)) == 0 &&
                               std::memcmp(dg.data(), hg.data(), dg.size() * sizeof(X::Tile)) == 0 && f == 0;
        check(plan_same, name + " device plan == host plan, flag clear");
        std::vector<float> y(yr.size()), gu(gr.size()), pairs(pr.size());
        std::vector<std::uint16_t> at(ar.size());
        ok(hipMemcpy(y.data(), out, y.size() * 4, hipMemcpyDeviceToHost), "device out");
        ok(hipMemcpy(pairs.data(), X::pair_output(ws, MaxR, 0), pairs.size() * 4, hipMemcpyDeviceToHost), "device pairs out");
        ok(hipMemcpy(gu.data(), X::pair_gate_up(ws, MaxR, 0), gu.size() * 4, hipMemcpyDeviceToHost), "device gate/up");
        ok(hipMemcpy(at.data(), X::pair_down_input(ws, MaxR, 0), at.size() * 2, hipMemcpyDeviceToHost), "device at");
        const bool out_pass = equal32(yr, y, name + " device-plan R16");
        const bool gu_pass = equal32(gr, gu, name + " device-plan gate/up");
        const bool pair_pass = equal32(pr, pairs, name + " device-plan pair-output");
        const bool at_pass = at == ar;
        check(at_pass, name + " device-plan down-input FP16");
        return plan_same && out_pass && gu_pass && pair_pass && at_pass;
    }
    // Invalid ids (duplicate within a row, id >= 288) on the device plan set the flag instead of throwing.
    void device_invalid(std::mt19937& rng) {
        const unsigned R = 300;
        Routes rt = routes(R, 0, rng);
        rt.ids[17 * 8 + 3] = rt.ids[17 * 8 + 2];
        rt.ids[201 * 8 + 7] = 300;
        upload(R, rt, rng);
        std::vector<M::ExpertSlices> table(experts.begin(), experts.end());
        table.push_back(shared);
        ok(hipMemcpy(device_table, table.data(), table.size() * sizeof(M::ExpertSlices), hipMemcpyHostToDevice), "device table");
        ok(hipMemsetAsync(flag, 0, 4, stream), "flag");
        X::DeviceView dv;
        dv.rows = R; dv.table = device_table; dv.shared = shared; dv.x = x; dv.ids = ids; dv.weights = weights; dv.out = out;
        X::enqueue_device(dv, plan, ws, wsb, stream, flag);
        ok(hipStreamSynchronize(stream), "device invalid");
        unsigned f = 0;
        ok(hipMemcpy(&f, flag, 4, hipMemcpyDeviceToHost), "flag");
        check(f == 1, "device plan invalid ids set the flag");
        bool threw = false;
        try { X::enqueue(view(R, rt), plan, ws, wsb, stream, 16); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "host plan invalid ids throw");
    }
    void exceptional(std::mt19937& rng) {
        constexpr unsigned R = 33;
        const auto rt = routes(R, 1, rng); // All eight routed groups and shared span a 32-row tile.
        Variant variant;
        for (const char* kind : {"signed-zero", "fp16-subnormal", "overflow", "nan-payload"}) {
            upload(R, rt, rng);
            std::vector<float> host(std::size_t{R} * H, 0.0F);
            for (unsigned r = 0; r < R; ++r) {
                if (std::strcmp(kind, "signed-zero") == 0) {
                    for (unsigned c = 0; c < H; ++c) {
                        const std::uint32_t u = (c + r) & 1U ? 0x80000000U : 0;
                        std::memcpy(&host[std::size_t{r} * H + c], &u, 4);
                    }
                } else for (unsigned c : {2048U, 2112U}) { // Gate/up split boundary and mid-split.
                    const bool negative = (r + c / 64U) & 1U;
                    float v;
                    if (std::strcmp(kind, "fp16-subnormal") == 0) v = negative ? -0x1p-18F : 0x1p-18F;
                    else if (std::strcmp(kind, "overflow") == 0) v = negative ? -1.0e10F : 1.0e10F;
                    else {
                        const std::uint32_t u = (negative ? 0xffc00000U : 0x7fc00000U) | ((r * 2U + c / 64U + 1U) << 13U);
                        std::memcpy(&v, &u, 4);
                    }
                    host[std::size_t{r} * H + c] = v;
                }
            }
            ok(hipMemcpy(x, host.data(), host.size() * 4, hipMemcpyHostToDevice), "exception x");
            // Zero effective weights carry both signs. Restore these private fixture scales below
            // before subnormal/overflow/NaN cases and the existing gates use them again.
            std::vector<M::Exl3> zero_weights;
            std::vector<std::vector<std::uint16_t>> saved_scales;
            if (std::strcmp(kind, "signed-zero") == 0) {
                for (unsigned e : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 287U, 288U}) {
                    const auto& expert = e == E ? shared : experts[e];
                    for (const auto& matrix : {expert.gate, expert.up, expert.down}) {
                        zero_weights.push_back(matrix);
                        saved_scales.emplace_back(matrix.output);
                        auto& saved = saved_scales.back();
                        ok(hipMemcpy(saved.data(), matrix.svh, saved.size() * 2, hipMemcpyDeviceToHost), "save exception weights");
                        std::vector<std::uint16_t> zeros(matrix.output);
                        for (unsigned c = 0; c < matrix.output; ++c) zeros[c] = c & 1U ? 0x8000U : 0U;
                        ok(hipMemcpy(const_cast<std::uint16_t*>(matrix.svh), zeros.data(), zeros.size() * 2,
                                     hipMemcpyHostToDevice), "signed-zero weights");
                    }
                }
            }
            std::vector<float> yr, gr, pr;
            std::vector<std::uint16_t> ar;
            for (const char* version : {"3", "6", "7", "8", "9", "10", "11", "14"}) {
                // v3 is this case's reference and always runs.
                if (std::strcmp(version, "3") != 0 && !selected(version)) continue;
                Variant::set(version);
                ok(hipMemsetAsync(ws, 0xff, wsb, stream), "exception poison");
                ok(hipMemsetAsync(out, 0xff, std::size_t{R} * H * 4, stream), "exception poison output");
                X::enqueue(view(R, rt), plan, ws, wsb, stream, 16);
                ok(hipStreamSynchronize(stream), "exception sync");
                std::vector<float> y(std::size_t{R} * H), gu(std::size_t{R} * 9 * 2 * I), pairs(std::size_t{R} * 9 * H);
                std::vector<std::uint16_t> at(std::size_t{R} * 9 * I), staged(std::size_t{R} * 9 * 2 * H);
                ok(hipMemcpy(y.data(), out, y.size() * 4, hipMemcpyDeviceToHost), "exception out");
                ok(hipMemcpy(gu.data(), X::pair_gate_up(ws, MaxR, 0), gu.size() * 4, hipMemcpyDeviceToHost), "exception gu");
                ok(hipMemcpy(pairs.data(), X::pair_output(ws, MaxR, 0), pairs.size() * 4, hipMemcpyDeviceToHost), "exception pairs");
                ok(hipMemcpy(at.data(), X::pair_down_input(ws, MaxR, 0), at.size() * 2, hipMemcpyDeviceToHost), "exception at");
                ok(hipMemcpy(staged.data(), X::pair_gate_up_input(ws, MaxR, 0), staged.size() * 2, hipMemcpyDeviceToHost), "exception staged t");
                const std::string label = std::string("exception ") + kind + " V=" + version;
                if (std::strcmp(kind, "fp16-subnormal") == 0)
                    check(std::any_of(staged.begin(), staged.end(), [](std::uint16_t h) {
                        return (h & 0x7c00U) == 0 && (h & 0x03ffU) != 0;
                    }), label + " staged t contains FP16 subnormals");
                if (std::strcmp(kind, "overflow") == 0)
                    check(std::find(staged.begin(), staged.end(), 0x7c00U) != staged.end() &&
                          std::find(staged.begin(), staged.end(), 0xfc00U) != staged.end(), label + " staged t contains +/-Inf");
                if (std::strcmp(kind, "nan-payload") == 0)
                    check(std::any_of(staged.begin(), staged.end(), [](std::uint16_t h) {
                        return (h & 0x7c00U) == 0x7c00U && (h & 0x03ffU) != 0;
                    }), label + " staged t contains NaN");
                if (std::strcmp(version, "3") == 0) {
                    yr = y; gr = gu; pr = pairs; ar = at;
                    if (std::strcmp(kind, "signed-zero") == 0)
                        check(std::all_of(gu.begin(), gu.end(), [](float v) { return (bits(v) & 0x7fffffffU) == 0; }),
                              label + " exact-zero linear sums");
                } else {
                    // Raw bytes deliberately include signed zero and all nonfinite payload bits.
                    check(std::memcmp(yr.data(), y.data(), y.size() * 4) == 0, label + " v3 R16 bytes");
                    const bool gu_same = std::memcmp(gr.data(), gu.data(), gu.size() * 4) == 0;
                    if (std::strcmp(kind, "overflow") == 0) {
                        // Documented contract: bitwise for all outputs; in this FP16-overflow case only, NaNs generated
                        // inside a chain (+Inf meets -Inf) may carry different payload bits in this INTERNAL workspace.
                        std::size_t payload_only = 0;
                        bool other_diff = false;
                        for (std::size_t i = 0; i < gu.size(); ++i) {
                            const std::uint32_t a = bits(gr[i]), b = bits(gu[i]);
                            if (a == b) continue;
                            if (std::isnan(gr[i]) && std::isnan(gu[i]) && (a >> 31) == (b >> 31)) ++payload_only;
                            else other_diff = true;
                        }
                        std::printf("INFO %s internal gate/up NaN-payload-only words: %zu\n", label.c_str(), payload_only);
                        check(!other_diff, label + " v3 gate/up bytes (NaN payload bits excepted)");
                    } else check(gu_same, label + " v3 gate/up bytes");
                    if (!gu_same) {
                        // Classify differing words so a nonfinite-only difference is distinguishable from a value change.
                        std::size_t nan_nan = 0, nan_inf = 0, other = 0, shown = 0;
                        for (std::size_t i = 0; i < gu.size(); ++i) {
                            const std::uint32_t a = bits(gr[i]), b = bits(gu[i]);
                            if (a == b) continue;
                            const bool an = std::isnan(gr[i]), bn = std::isnan(gu[i]);
                            if (an && bn) ++nan_nan;
                            else if ((an && std::isinf(gu[i])) || (bn && std::isinf(gr[i]))) ++nan_inf;
                            else ++other;
                            if (shown++ < 6) std::printf("DIFF gate/up word %zu v3 %08x got %08x\n", i, a, b);
                        }
                        std::printf("DIFF gate/up words: nan-vs-nan %zu, nan-vs-inf %zu, other %zu\n", nan_nan, nan_inf, other);
                    }
                    check(std::memcmp(pr.data(), pairs.data(), pairs.size() * 4) == 0, label + " v3 pair-output bytes");
                    check(std::memcmp(ar.data(), at.data(), at.size() * 2) == 0, label + " v3 down-input FP16 bytes");
                }
            }
            for (std::size_t i = 0; i < zero_weights.size(); ++i)
                ok(hipMemcpy(const_cast<std::uint16_t*>(zero_weights[i].svh), saved_scales[i].data(),
                             saved_scales[i].size() * 2, hipMemcpyHostToDevice), "restore exception weights");
        }
    }
    void timing(unsigned R, const Routes& rt, std::mt19937& rng) {
        upload(R, rt, rng);
        // Timed workload is qualified too before reporting any timing.
        reference(R, 1, false);
        ok(hipStreamSynchronize(stream), "timing reference");
        std::vector<float> expected(std::size_t{R} * H), got(expected.size());
        ok(hipMemcpy(expected.data(), ref_out, expected.size() * 4, hipMemcpyDeviceToHost), "expected");
        Variant variant;
        const X::View v = view(R, rt);
        for (unsigned tm : {16U, 32U}) {
            std::array<std::array<double, 3>, 6> times{};
            for (const char* version : {"1", "2", "3", "6", "7"}) {
                Variant::set(version);
                X::enqueue(v, plan, ws, wsb, stream, tm);
            }
            reference(R, 8, false);
            ok(hipStreamSynchronize(stream), "warmup");
            // Rotate all six arms, including host planning/upload/synchronization.
            for (unsigned rep = 0; rep < 3; ++rep) for (unsigned order = 0; order < 6; ++order) {
                const unsigned arm = (rep + order) % 6;
                if (arm < 5) Variant::set(arm == 0 ? "1" : arm == 1 ? "2" : arm == 2 ? "3" : arm == 3 ? "6" : "7");
                const auto start = std::chrono::steady_clock::now();
                if (arm < 5) X::enqueue(v, plan, ws, wsb, stream, tm);
                else reference(R, 8, false);
                ok(hipStreamSynchronize(stream), "timed");
                const auto stop = std::chrono::steady_clock::now();
                times[arm][rep] = std::chrono::duration<double, std::milli>(stop - start).count();
                ok(hipMemcpy(got.data(), arm < 5 ? out : ref_out, got.size() * 4, hipMemcpyDeviceToHost), "timed output");
                if (!equal32(expected, got, "timed output R=" + std::to_string(R) + " arm=" + std::to_string(arm))) return;
            }
            for (auto& arm : times) std::sort(arm.begin(), arm.end());
            const double a = times[0][1], b = times[1][1], c = times[2][1], xt = times[3][1], x2t = times[4][1], seam = times[5][1];
            // Useful GEMM FLOPs, including shared; excludes transforms, decoding and R15/R16.
            const double flops = static_cast<double>(R) * 9.0 * 6.0 * H * I;
            std::printf("TIME R=%u TM=%u rotated median whole-call: "
                        "v1 %.3f ms/chunk %.6f ms/row %.3f TFLOP/s; "
                        "v2 %.3f ms/chunk %.6f ms/row %.3f TFLOP/s; "
                        "v3 %.3f ms/chunk %.6f ms/row %.3f TFLOP/s; "
                        "v6 %.3f ms/chunk %.6f ms/row %.3f TFLOP/s; "
                        "v7 %.3f ms/chunk %.6f ms/row %.3f TFLOP/s; "
                        "decode R8 %.3f ms/chunk; v2/v1 speedup %.3fx; v3/v1 speedup %.3fx; v3/seam speedup %.3fx; v6/v3 speedup %.3fx; v7/v3 speedup %.3fx\n",
                        R, tm, a, a / R, flops / (a * 1e9), b, b / R, flops / (b * 1e9),
                        c, c / R, flops / (c * 1e9), xt, xt / R, flops / (xt * 1e9), x2t, x2t / R, flops / (x2t * 1e9), seam, a / b, a / c, seam / c, c / xt, c / x2t);
            if (tm == 16) check(a / c >= 1.6, "v3 >=1.6x v1 R=" + std::to_string(R));
        }
    }
};
} // namespace
int main(int argc, char** argv) {
    try {
        bool timing = true, timing_only = false;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--no-timing") == 0) timing = false;
            else if (std::strcmp(argv[i], "--timing-only") == 0) timing_only = true;
            else if (std::strcmp(argv[i], "--versions") == 0 && i + 1 < argc) only_versions = argv[++i];
            else throw std::invalid_argument("usage: strix-hip-glm5-pf-exact-moe-tests [--no-timing | --timing-only] [--versions LIST]");
        }
        if (!timing && timing_only) throw std::invalid_argument("conflicting timing options");
        std::mt19937 rng(0x5eed1234U);
        for (unsigned variant : {6U, 7U, 8U, 9U, 10U, 11U, 12U}) for (bool shared : {false, true}) for (unsigned bits : {4U, 6U}) for (bool down : {false, true}) {
            if (bits == 6 && !shared) continue;
            const auto a = X::x_resources(bits, down, shared, variant);
            const auto name = "v" + std::to_string(variant) + " K" + std::to_string(bits) + (shared ? " shared" : " routed") + (down ? " down" : " gate/up");
            std::printf("RESOURCE %s private=%zu B/lane VGPR=%d LDS=%zu\n", name.c_str(), a.localSizeBytes, a.numRegs, a.sharedSizeBytes);
            check(a.localSizeBytes == 0, "zero spills " + name);
        }
        {
            // enqueue dispatches through selected_variant(); auto must pick v14 exactly at the threshold.
            Variant variant;
            auto min_rows = [](const char* value) {
#ifdef _WIN32
                if (_putenv_s("STRIX_GLM_PF_EXACTMOE_X_MIN_ROWS", value ? value : "") != 0)
#else
                if ((value ? setenv("STRIX_GLM_PF_EXACTMOE_X_MIN_ROWS", value, 1) : unsetenv("STRIX_GLM_PF_EXACTMOE_X_MIN_ROWS")) != 0)
#endif
                    throw std::runtime_error("setting exact MoE auto threshold");
            };
            min_rows(nullptr);
            Variant::set(nullptr);
            check(X::selected_variant(4096) == 3, "selected variant unset -> 3");
            for (const char* v : {"1", "2", "3", "6", "7", "8", "9", "10", "11", "14"}) {
                Variant::set(v);
                check(X::selected_variant(2048) == static_cast<unsigned>(std::atoi(v)), std::string("selected variant ") + v);
            }
            Variant::set("auto");
            for (unsigned rows : {1U, 127U, 128U, 129U, 2048U, 4096U})
                check(X::selected_variant(rows) == (rows >= 128 ? 14U : 3U), "selected variant auto rows=" + std::to_string(rows));
            min_rows("100");
            check(X::selected_variant(99) == 3 && X::selected_variant(100) == 14, "selected variant auto MIN_ROWS=100");
            for (const char* bad : {"0", "65537", "abc"}) {
                min_rows(bad);
                bool threw = false;
                try { (void)X::selected_variant(4096); } catch (const std::invalid_argument&) { threw = true; }
                check(threw, std::string("auto MIN_ROWS=") + bad + " rejected");
            }
            min_rows(nullptr);
            check(X::v3_row_buckets(), "v3 row buckets default on");
            rcap_env("0");
            check(!X::v3_row_buckets(), "v3 row buckets =0 off");
            rcap_env("2");
            bool rcap_threw = false;
            try { (void)X::v3_row_buckets(); } catch (const std::invalid_argument&) { rcap_threw = true; }
            check(rcap_threw, "v3 row buckets =2 rejected");
            rcap_env(nullptr);
            Variant::set("5");
            bool threw = false;
            try { (void)X::selected_variant(1); } catch (const std::invalid_argument&) { threw = true; }
            check(threw, "selected variant 5 rejected");
        }
        Fixture f(rng);
        std::printf("exact workspace: %zu bytes at Rmax=%u; random K4 x288 + shared K6\n", f.wsb, MaxR);
        if (!timing_only) {
            for (unsigned R : {1U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 32U, 33U, 64U, 300U, 1024U, 2048U}) {
                for (unsigned mode = 0; mode < 4; ++mode) {
                    const Routes r = routes(R, mode, rng);
                    const auto label = "R=" + std::to_string(R) + " routes=" + std::to_string(mode);
                    if (!f.gate(R, r, label, rng)) {
                        std::printf("checks: %u failures: %u\n", checks, failures);
                        return 1;
                    }
                }
            }
            f.exceptional(rng);
            f.device_invalid(rng);
            const auto tails = routes(1024, 4, rng);
            bool all_tails = true;
            for (unsigned e = 0; e < 33; ++e)
                all_tails = all_tails && std::count(tails.ids.begin(), tails.ids.end(), e) == e + 1;
            check(all_tails, "routing contains exact group sizes 1..33");
            f.gate(1024, tails, "all group tails 1..33", rng, true);
            for (unsigned mode : {0U, 1U, 5U, 6U, 7U}) {
                const auto rt = routes(4096, mode, rng);
                if (mode == 1) for (unsigned e : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 287U})
                    check(std::count(rt.ids.begin(), rt.ids.end(), e) == 4096, "concentrated group R=4096 expert=" + std::to_string(e));
                if (mode >= 5)
                    check(std::count(rt.ids.begin(), rt.ids.end(), 0U) == 2047U + mode - 5U,
                          "expert 0 group=" + std::to_string(2047U + mode - 5U));
                f.gate(4096, rt, "R=4096 routes=" + std::to_string(mode), rng, true);
            }
            // Shared K4 follows the same qualified code path and must also pass the gate.
            f.shared = f.experts[0];
            if (!f.gate(33, routes(33, 2, rng), "shared K4", rng)) timing = false;
        }
        // Restore K6 for the real-shape timing workloads.
        if (!timing_only) f.shared = make_expert(f.model_mem, rng, 6);
        if (!failures && timing) for (unsigned R : {1024U, 2048U}) {
            const auto r = routes(R, 3, rng);
            if (!timing_only && !f.gate(R, r, "timing lognormal R=" + std::to_string(R), rng)) break;
            f.timing(R, r, rng);
        }
    } catch (const std::exception& e) {
        check(false, e.what());
    }
    std::printf("checks: %u failures: %u\n", checks, failures);
    return failures ? 1 : 0;
}
