// Prefill micro-benchmarks of the deepseek41 layer sites (not a ctest), T rows at once:
//  - Q8_0 x A8 projections in the WMMA iu8 batch form (enqueue_q8_0_a8_prefill), checked bitwise
//    against product_q8_0 on sampled rows, in TOPS (2 * T * width * outputs) and weight GB/s;
//  - the per-row kernels (hc mixes, collapse + norm + A8, expansion, query RoPE, inverse RoPE + A8,
//    Engram injection) in GB/s of the bytes they must move.
// usage: strix-hip-ds41-layer-prefill-bench [iterations]

#include "strix/ds41_formats.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace {

using namespace strix::ds41;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
template <class T> T* dev(std::size_t n) { T* p = nullptr; check(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "hipMalloc"); return p; }
template <class T> T* dev(const std::vector<T>& h) { T* p = dev<T>(h.size()); check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); return p; }
template <class T> std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> h(n);
    check(hipDeviceSynchronize(), "sync");
    check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return h;
}
std::uint16_t to_f16(float x) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x)); }
std::vector<std::uint8_t> random_q8(std::mt19937& rng, std::size_t rows, std::size_t width) {
    std::vector<std::uint8_t> w(rows * width / 32 * 34);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.001F, 0.05F);
    for (std::size_t b = 0; b < rows * width / 32; ++b) {
        const auto s = to_f16(sc(rng));
        w[b * 34] = static_cast<std::uint8_t>(s & 0xffU);
        w[b * 34 + 1] = static_cast<std::uint8_t>(s >> 8U);
        for (std::size_t k = 0; k < 32; ++k) w[b * 34 + 2 + k] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(rng)));
    }
    return w;
}
std::vector<float> random_vector(std::mt19937& rng, std::size_t n, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
bool sites_mode() {
    const char* e = std::getenv("STRIX_BENCH_SITES");
    return e && e[0] == '1';
}

std::vector<unsigned> tall_forms() {
    std::vector<unsigned> forms;
    if (const char* e = std::getenv("STRIX_BENCH_TALL"))
        for (const char* q = e; *q != '\0';) {
            char* end = nullptr;
            forms.push_back(static_cast<unsigned>(std::strtoul(q, &end, 10)));
            q = *end == ',' ? end + 1 : end;
            if (end == q && *q != '\0') break;
        }
    return forms;
}

double time_ms(int iterations, const std::function<void()>& launch) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    for (int i = 0; i < 2; ++i) launch();
    std::vector<float> t;
    for (int i = 0; i < iterations; ++i) {
        check(hipEventRecord(a, nullptr), "record");
        launch();
        check(hipEventRecord(b, nullptr), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        t.push_back(ms);
    }
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

struct DeviceRoutes {
    std::vector<std::uint32_t*> owned;
    PrefillRoutesView view;
    explicit DeviceRoutes(const PrefillRoutes& r) {
        auto up = [&](const std::vector<std::uint32_t>& v) { auto* p = dev(v); owned.push_back(p); return p; };
        view.begins = up(r.begins);
        view.entry_rows = up(r.entry_rows);
        view.entry_slots = up(r.entry_slots);
        view.batch_expert = up(r.batch_expert);
        view.batch_begin = up(r.batch_begin);
        view.batch_count = up(r.batch_count);
        view.batch_total = up(std::vector<std::uint32_t>{static_cast<std::uint32_t>(r.batch_expert.size())});
        view.max_batches = static_cast<std::uint32_t>(r.batch_expert.size());
        view.batch_rows = r.batch_rows;
    }
    ~DeviceRoutes() { for (auto* p : owned) (void)hipFree(p); }
};

struct Site { const char* name; std::uint32_t width, groups, per_group; };

void projection(std::mt19937& rng, const Site& s, std::uint32_t T, int iterations) {
    const std::uint32_t outputs = s.groups * s.per_group, in_row = s.groups * s.width;
    const auto w = random_q8(rng, outputs, s.width);
    auto* dw = dev(w);
    const auto xf = random_vector(rng, std::size_t{T} * in_row, 1.0F);
    auto* dx = dev(xf);
    auto* codes = dev<std::int8_t>(std::size_t{T} * in_row);
    auto* deltas = dev<float>(std::size_t{T} * in_row / 32);
    auto* sums = dev<std::int32_t>(std::size_t{T} * in_row / 32);
    enqueue_quantize_a8(dx, in_row, T, codes, deltas, sums, nullptr);
    auto* out = dev<float>(std::size_t{T} * outputs);
    const auto hc = down(codes, std::size_t{T} * in_row);
    const auto hd = down(deltas, std::size_t{T} * in_row / 32);
    // Bitwise check of three rows against product_q8_0 (host); split: the two K-halves of attn_output_b summed as
    // production sums them (q8_0_a8_split_rows: h0, then + h1).
    auto exact_rows = [&](bool split = false) {
        const auto got = down(out, std::size_t{T} * outputs);
        bool exact = true;
        for (const std::uint32_t t : {0U, T / 2U + 1U, T - 1U}) {
            A8Activation full;
            full.codes.assign(hc.begin() + static_cast<std::ptrdiff_t>(std::size_t{t} * in_row), hc.begin() + static_cast<std::ptrdiff_t>(std::size_t{t + 1} * in_row));
            full.deltas.assign(hd.begin() + static_cast<std::ptrdiff_t>(std::size_t{t} * in_row / 32), hd.begin() + static_cast<std::ptrdiff_t>(std::size_t{t + 1} * in_row / 32));
            full.sums.assign(in_row / 32, 0);
            std::vector<float> ref(outputs);
            if (split) q8_0_a8_split_rows(w.data(), s.width, outputs, 2, full, ref.data());
            else q8_0_a8_rows(w.data(), s.width, 0, outputs, s.per_group, full, ref.data());
            exact = exact && std::memcmp(ref.data(), got.data() + std::size_t{t} * outputs, outputs * 4) == 0;
        }
        return exact;
    };
    const double ops = 2.0 * T * s.width * static_cast<double>(outputs);
    auto print = [&](const char* kernel, double ms, bool exact) {
        std::printf("{\"site\":\"%s\",\"kernel\":\"%s\",\"rows\":%u,\"ms\":%.4f,\"tops\":%.2f,\"weight_gbs\":%.1f,\"exact\":%s}\n", s.name,
                    kernel, T, ms, ops / (ms * 1e-3) / 1e12, static_cast<double>(w.size()) / (ms * 1e-3) / 1e9, exact ? "true" : "false");
        std::fflush(stdout);
    };
    if (s.width <= 5120) {
        DeviceRoutes routes(prefill_site_routes(T, s.groups, sites_mode() ? 16U : 32U));
        Q8A8PrefillView v;
        v.weight = dw; v.width = s.width; v.groups = s.groups; v.per_group = s.per_group; v.routes = routes.view;
        v.codes = codes; v.deltas = deltas; v.out = out;
        check(hipMemset(out, 0, std::size_t{T} * outputs * 4), "memset");
        enqueue_q8_0_a8_prefill(v, nullptr);
        const bool exact = exact_rows();
        print("prefill-batch-wmma", time_ms(iterations, [&] { enqueue_q8_0_a8_prefill(v, nullptr); }), exact);
    }
    if (s.groups == 1) {
        Q8A8DensePrefillView v;
        v.weight = dw; v.width = s.width; v.count = outputs; v.codes = codes; v.deltas = deltas; v.rows = T; v.x_stride = s.width;
        v.out = out; v.out_stride = outputs;
        check(hipMemset(out, 0, std::size_t{T} * outputs * 4), "memset");
        enqueue_q8_0_a8_dense_prefill(v, nullptr);
        const bool exact = exact_rows();
        print("dense-wmma", time_ms(iterations, [&] { enqueue_q8_0_a8_dense_prefill(v, nullptr); }), exact);
    }
    // STRIX_BENCH_TALL=form,form,...: the dense kernel per form (0 = wide) as production calls it: one call per o-group
    // (o_a), the two K-halves of attn_output_b (h0 into out, h1 accumulated), else one call
    std::vector<float> first_out;
    for (const unsigned form : tall_forms()) {
        const bool halves = s.width > 5120;
        const auto launch = [&] {
            for (std::uint32_t g = 0; g < s.groups; ++g)
                for (std::uint32_t part = 0; part < (halves ? 2U : 1U); ++part) {
                    Q8A8DensePrefillView v;
                    v.weight = dw + std::size_t{g} * s.per_group * (s.width / 32U) * 34U;
                    v.width = halves ? s.width / 2U : s.width;
                    v.row_width = halves ? s.width : 0U;
                    v.k_offset = part * v.width;
                    v.accumulate = part == 1U;
                    v.count = s.per_group;
                    v.codes = codes + g * s.width + part * v.width;
                    v.deltas = deltas + (g * s.width + part * v.width) / 32U;
                    v.rows = T;
                    v.x_stride = in_row;
                    v.out = out + g * s.per_group;
                    v.out_stride = outputs;
                    v.tall = form;
                    enqueue_q8_0_a8_dense_prefill(v, nullptr);
                }
        };
        check(hipMemset(out, 0, std::size_t{T} * outputs * 4), "memset");
        launch();
        // the three rows against the host, and every output bit for bit against the first form of the list
        bool exact = exact_rows(halves);
        auto all = down(out, std::size_t{T} * outputs);
        if (first_out.empty()) first_out = std::move(all);
        else exact = exact && std::memcmp(first_out.data(), all.data(), first_out.size() * 4) == 0;
        char name[32];
        std::snprintf(name, sizeof name, "dense-tall-%u", form);
        print(name, time_ms(iterations, launch), exact);
    }
    (void)hipFree(dw); (void)hipFree(dx); (void)hipFree(codes); (void)hipFree(deltas); (void)hipFree(sums); (void)hipFree(out);
}

void per_row(std::mt19937& rng, std::uint32_t T, int iterations) {
    const std::uint32_t H = 5120, n = H * hc_copies, heads = 64, D = 512;
    auto* s = dev(random_vector(rng, std::size_t{T} * n, 1.0F));
    auto* s2 = dev<float>(std::size_t{T} * n);
    std::vector<std::uint16_t> fn(std::size_t{hc_mix_count} * n);
    for (auto& x : fn) x = to_f16(std::normal_distribution<float>(0.0F, 0.02F)(rng));
    auto* dfn = dev(fn);
    auto* base = dev(random_vector(rng, hc_mix_count, 0.5F));
    auto* scale = dev(std::vector<float>{0.7F, 1.3F, 2.1F});
    auto* mixes = dev<HcMix>(T);
    auto* partial = dev<float>(hc_mix_partial_floats(T, H));
    auto* w = dev(random_vector(rng, H, 1.0F));
    auto* xn = dev<float>(std::size_t{T} * H);
    auto* codes = dev<std::int8_t>(std::size_t{T} * heads * D);
    auto* deltas = dev<float>(std::size_t{T} * heads * D / 32);
    auto* sums = dev<std::int32_t>(std::size_t{T} * heads * D / 32);
    auto* q = dev(random_vector(rng, std::size_t{T} * heads * D, 1.0F));
    const auto table = rope_table(RopeConfig{64, 160000.0, 65536.0, 16.0, 32.0, 1.0}, 0, T);
    auto* cs = dev(table.cos);
    auto* sn = dev(table.sin);
    std::vector<std::uint32_t> pos(T);
    for (std::uint32_t i = 0; i < T; ++i) pos[i] = i;
    auto* dpos = dev(pos);
    auto* kv = dev(random_vector(rng, std::size_t{T} * H * (hc_copies + 1), 0.7F));
    auto report = [&](const char* name, double bytes, double ms) {
        std::printf("{\"kernel\":\"%s\",\"rows\":%u,\"ms\":%.4f,\"gbs\":%.1f,\"us_per_row\":%.3f}\n", name, T, ms, bytes / (ms * 1e-3) / 1e9,
                    ms * 1e3 / T);
        std::fflush(stdout);
    };
    HcMixView mv;
    mv.stream = s; mv.fn = dfn; mv.base = base; mv.scale = scale; mv.rows = T; mv.hidden = H; mv.out = mixes; mv.partial = partial;
    report("hc-mixes", static_cast<double>(T) * n * 4 + static_cast<double>(fn.size()) * 2, time_ms(iterations, [&] { enqueue_hc_mixes(mv, nullptr); }));
    NormView nv;
    nv.hc_stream = s; nv.mixes = mixes; nv.weight = w; nv.rows = T; nv.n = H; nv.eps = 1e-20F; nv.out = xn; nv.out_stride = H;
    nv.codes = codes; nv.deltas = deltas; nv.sums = sums; nv.a8_stride = H;
    report("collapse-norm-a8", static_cast<double>(T) * (n * 4 + H * 4 + H + H / 8), time_ms(iterations, [&] { enqueue_norm(nv, nullptr); }));
    if (T <= 8) {
        NormView a = nv;
        a.codes = nullptr; a.deltas = nullptr; a.sums = nullptr;
        report("collapse-norm", static_cast<double>(T) * (n * 4 + H * 4), time_ms(iterations, [&] { enqueue_norm(a, nullptr); }));
        a.hc_stream = nullptr; a.mixes = nullptr; a.x = s; a.x_stride = n;
        report("norm", static_cast<double>(T) * H * 8, time_ms(iterations, [&] { enqueue_norm(a, nullptr); }));
        a.codes = codes; a.deltas = deltas; a.sums = sums;
        report("norm-a8", static_cast<double>(T) * H * 9, time_ms(iterations, [&] { enqueue_norm(a, nullptr); }));
        report("empty-launch-ref(collapse)", 0.0, time_ms(iterations, [&] { enqueue_hc_collapse(s, mixes, T, 32, xn, nullptr); }));
        // Site boundary: expansion + mixes + Sinkhorn + collapse + norm + A8, separate launches vs one.
        auto* res = dev(random_vector(rng, std::size_t{T} * n, 1.0F));
        auto* sub = dev(random_vector(rng, std::size_t{T} * H, 1.0F));
        auto* counters = dev(std::vector<std::uint32_t>(hc_site_counter_count(T), 0U));
        auto* mixes_out = dev<HcMix>(T);
        const double separate = time_ms(iterations, [&] {
            enqueue_hc_expand(sub, res, mixes, T, H, s2, nullptr);
            HcMixView m2 = mv;
            m2.stream = s2; m2.out = mixes_out;
            enqueue_hc_mixes(m2, nullptr);
            NormView n2 = nv;
            n2.hc_stream = s2;
            enqueue_norm(n2, nullptr);
        });
        report("site-separate(expand+mixes+norm-a8)", static_cast<double>(T) * (3.0 * n * 4), separate);
        HcSiteView sv;
        sv.rows = T; sv.hidden = H; sv.sub = sub; sv.residual = res; sv.expand_mixes = mixes; sv.stream = s2;
        sv.fn = dfn; sv.base = base; sv.scale = scale; sv.mixes_out = mixes_out; sv.partial = partial; sv.counters = counters;
        sv.collapse_mixes = mixes; sv.norm_weight = w; sv.out = xn; sv.out_stride = H; sv.codes = codes; sv.deltas = deltas;
        sv.sums = sums; sv.a8_stride = H;
        sv.launches = 1;
        report("site-one-launch", static_cast<double>(T) * (3.0 * n * 4), time_ms(iterations, [&] { enqueue_hc_site(sv, nullptr); }));
        sv.launches = 2;
        report("site-separate-via-site", static_cast<double>(T) * (3.0 * n * 4), time_ms(iterations, [&] { enqueue_hc_site(sv, nullptr); }));
        sv.launches = 1;
        HcSiteView head = sv;
        head.fn = nullptr;
        head.launches = 1;
        report("site-head(expand+norm-a8)", static_cast<double>(T) * (3.0 * n * 4), time_ms(iterations, [&] { enqueue_hc_site(head, nullptr); }));
        (void)hipFree(res); (void)hipFree(sub); (void)hipFree(counters); (void)hipFree(mixes_out);
    }
    report("hc-expand", static_cast<double>(T) * (2.0 * n * 4 + H * 4), time_ms(iterations, [&] { enqueue_hc_expand(xn, s, mixes, T, H, s2, nullptr); }));
    RopeView rv;
    rv.x = q; rv.rows = T; rv.vectors = heads; rv.row_stride = heads * D; rv.vector_stride = D; rv.offset = D - 64; rv.pairs = 32;
    rv.cos = cs; rv.sin = sn; rv.positions = dpos;
    report("query-rope", static_cast<double>(T) * heads * 64 * 4 * 2, time_ms(iterations, [&] { enqueue_rope(rv, nullptr); }));
    InverseRopeA8View iv;
    iv.o = q; iv.rows = T; iv.heads = heads; iv.head_dim = D; iv.pairs = 32; iv.rope_cos = cs; iv.rope_sin = sn; iv.positions = dpos;
    iv.codes = codes; iv.deltas = deltas; iv.sums = sums;
    report("inverse-rope-a8", static_cast<double>(T) * heads * D * (4 + 1 + 1.0 / 4), time_ms(iterations, [&] { enqueue_inverse_rope_a8(iv, nullptr); }));
    auto* qn = dev(random_vector(rng, n, 1.0F));
    auto* kn = dev(random_vector(rng, n, 1.0F));
    report("engram-inject", static_cast<double>(T) * (2.0 * n * 4 + H * (hc_copies + 1) * 4),
           time_ms(iterations, [&] { enqueue_engram_inject(s2, kv, qn, kn, T, H, 1e-20F, nullptr); }));
    for (void* p : std::initializer_list<void*>{s, s2, dfn, base, scale, mixes, partial, w, xn, codes, deltas, sums, q, cs, sn, dpos, kv, qn, kn})
        (void)hipFree(p);
}

} // namespace

// STRIX_BENCH_OLOW=1: o_a as the prefill runs it (one dense call per o-group, tall auto, chunk rows) followed by the
// separate quantize_a8 of o_low, against the same calls writing o_low's A8 themselves; codes / deltas / sums compared.
void olow(std::mt19937& rng, std::uint32_t T, std::uint32_t groups, int iterations) {
    const std::uint32_t width = 4096, rank = 1024, oll = groups * rank, in_row = groups * width;
    const auto w = random_q8(rng, oll, width);
    auto* dw = dev(w);
    auto* dx = dev(random_vector(rng, std::size_t{T} * in_row, 1.0F));
    auto* codes = dev<std::int8_t>(std::size_t{T} * in_row);
    auto* deltas = dev<float>(std::size_t{T} * in_row / 32);
    auto* sums = dev<std::int32_t>(std::size_t{T} * in_row / 32);
    enqueue_quantize_a8(dx, in_row, T, codes, deltas, sums, nullptr);
    auto* out = dev<float>(std::size_t{T} * oll);
    std::int8_t* lc[2] = {dev<std::int8_t>(std::size_t{T} * oll), dev<std::int8_t>(std::size_t{T} * oll)};
    float* ld[2] = {dev<float>(std::size_t{T} * oll / 32), dev<float>(std::size_t{T} * oll / 32)};
    std::int32_t* ls[2] = {dev<std::int32_t>(std::size_t{T} * oll / 32), dev<std::int32_t>(std::size_t{T} * oll / 32)};
    auto run = [&](bool fused) {
        bool all = fused;
        for (std::uint32_t g = 0; g < groups; ++g) {
            Q8A8DensePrefillView v;
            v.weight = dw + std::size_t{g} * rank * (width / 32U) * 34U;
            v.width = width; v.count = rank; v.rows = T; v.x_stride = in_row;
            v.codes = codes + g * width; v.deltas = deltas + g * width / 32U;
            v.out = out + g * rank; v.out_stride = oll; v.tall = 1U;
            if (fused) { v.a8_codes = lc[1] + g * rank; v.a8_deltas = ld[1] + g * rank / 32U; v.a8_sums = ls[1] + g * rank / 32U; v.a8_stride = oll; }
            const bool written = enqueue_q8_0_a8_dense_prefill(v, nullptr);
            all = all && written;
        }
        if (!all) enqueue_quantize_a8(out, oll, T, lc[fused ? 1 : 0], ld[fused ? 1 : 0], ls[fused ? 1 : 0], nullptr);
        return all;
    };
    run(false);
    const bool written = run(true);
    check(hipDeviceSynchronize(), "sync");
    const bool same = down(lc[0], std::size_t{T} * oll) == down(lc[1], std::size_t{T} * oll) &&
                      std::memcmp(down(ld[0], std::size_t{T} * oll / 32).data(), down(ld[1], std::size_t{T} * oll / 32).data(), std::size_t{T} * oll / 32 * 4) == 0 &&
                      down(ls[0], std::size_t{T} * oll / 32) == down(ls[1], std::size_t{T} * oll / 32);
    double sep = 0, fus = 0;
    for (int round = 0; round < 3; ++round) {  // interleaved
        sep += time_ms(iterations, [&] { run(false); });
        fus += time_ms(iterations, [&] { run(true); });
    }
    std::printf("{\"kernel\":\"o_a+o_low A8\",\"rows\":%u,\"groups\":%u,\"separate_ms\":%.4f,\"fused_ms\":%.4f,\"written\":%d,\"bitwise\":%d}\n", T,
                groups, sep / 3, fus / 3, written ? 1 : 0, same ? 1 : 0);
    (void)hipFree(dw); (void)hipFree(dx); (void)hipFree(codes); (void)hipFree(deltas); (void)hipFree(sums); (void)hipFree(out);
    for (int i = 0; i < 2; ++i) { (void)hipFree(lc[i]); (void)hipFree(ld[i]); (void)hipFree(ls[i]); }
}

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 10;
    std::mt19937 rng(7);
    const Site sites[] = {
        {"attn_q_a", 5120, 1, 1280},
        {"attn_kv", 5120, 1, 512},
        {"attn_q_b", 1280, 1, 32768},
        {"attn_output_a", 4096, 8, 1024},
        {"attn_output_b", 8192, 1, 5120},
    };
    // STRIX_BENCH_SITES=1: the attention sites only, at the production rows of a rank (row split: q_a / kv up to 4096
    // rows of an 8K prompt; q_b / o_a / o_b per chunk of at most chunk_rows = 1024; batch routes of 16 rows as the
    // prefill builds them)
    if (const char* e = std::getenv("STRIX_BENCH_OLOW"); e && e[0] == '1') {
        for (const std::uint32_t groups : {4U, 8U})
            for (const std::uint32_t T : {256U, 512U, 1024U}) olow(rng, T, groups, iterations);
        return 0;
    }
    if (sites_mode()) {
        for (const std::uint32_t T : {512U, 1024U, 2048U, 3072U, 4096U})
            for (const auto& s : {sites[0], sites[1]}) projection(rng, s, T, iterations);
        for (const std::uint32_t T : {256U, 512U, 1024U})
            for (const auto& s : {sites[2], sites[3], sites[4]}) projection(rng, s, T, iterations);
        return 0;
    }
    for (const std::uint32_t T : {512U, 2048U})
        for (const auto& s : sites) projection(rng, s, T, iterations);
    for (const std::uint32_t T : {1U, 2U, 4U, 8U, 512U, 2048U}) per_row(rng, T, std::max(iterations, T <= 8U ? 50 : 1));
    // Greedy selection over a rank's head slice (64,640 logits per row).
    for (const std::uint32_t R : {1U, 8U}) {
        auto* logits = dev(random_vector(rng, std::size_t{R} * 64640, 3.0F));
        auto* partial = dev<ArgmaxCandidate>(std::size_t{R} * argmax_partial_blocks(64640));
        auto* cand = dev<ArgmaxCandidate>(R);
        const double ms = time_ms(50, [&] { enqueue_argmax(logits, R, 64640, 64640, 0, partial, cand, nullptr); });
        std::printf("{\"kernel\":\"head-slice-argmax\",\"rows\":%u,\"ms\":%.4f}\n", R, ms);
        (void)hipFree(logits); (void)hipFree(partial); (void)hipFree(cand);
    }
    return 0;
}
