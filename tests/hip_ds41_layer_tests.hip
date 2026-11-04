// Device kernels of the deepseek41 layer sites against their host reference, bitwise. Synthetic
// cases for every kernel; with STRIX_DEEPSEEK41_GGUF a chain over real layer-1 weights in the decode
// data flow: embedding, Engram (hash, file reads, decode, projection, gated injection), the
// attention-site mixes, collapse + attn_norm + A8, q_a, q_norm + A8, q_b + RoPE, kv, kv_norm + RoPE, a
// stand-in attention output through inverse RoPE + A8, the grouped output projection, A8, output_b,
// both expansions, the FFN collapse + norm + A8, the final collapse + norm + A8 and the
// vocabulary-split head with its greedy selection.

#include "strix/deepseek41_model.hpp"
#include "strix/ds41_engram.hpp"
#include "strix/ds41_packed.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace strix::ds41;
int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::cerr << what << ": " << hipGetErrorString(e) << '\n'; std::exit(2); }
}

template <class T>
struct Device {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t count) : n(count) { check(hipMalloc(&p, std::max<std::size_t>(1, count) * sizeof(T)), "hipMalloc"); }
    explicit Device(const std::vector<T>& h) : Device(h.size()) { up(h); }
    ~Device() { (void)hipFree(p); }
    Device(const Device&) = delete;
    void up(const std::vector<T>& h) { check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        check(hipDeviceSynchronize(), "sync");
        check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
        return h;
    }
};

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
bool same_mix(const std::vector<HcMix>& a, const std::vector<HcMix>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(HcMix)) == 0;
}

std::uint16_t to_f16(float x) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x)); }
std::vector<float> random_vector(std::mt19937& rng, std::size_t n, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
std::vector<std::uint16_t> random_f16(std::mt19937& rng, std::size_t n, float scale) {
    std::vector<std::uint16_t> v(n);
    std::normal_distribution<float> d(0.0F, scale);
    for (auto& x : v) x = to_f16(d(rng));
    return v;
}
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

// ---- A8 activations of several rows ----------------------------------------------------------------
struct HostA8 {
    std::vector<A8Activation> rows;
    std::vector<std::int8_t> codes() const { std::vector<std::int8_t> c; for (const auto& r : rows) c.insert(c.end(), r.codes.begin(), r.codes.end()); return c; }
    std::vector<float> deltas() const { std::vector<float> d; for (const auto& r : rows) d.insert(d.end(), r.deltas.begin(), r.deltas.end()); return d; }
    std::vector<std::int32_t> sums() const { std::vector<std::int32_t> s; for (const auto& r : rows) s.insert(s.end(), r.sums.begin(), r.sums.end()); return s; }
};
HostA8 host_a8(const std::vector<float>& x, std::uint32_t rows, std::uint32_t n) {
    HostA8 out;
    for (std::uint32_t r = 0; r < rows; ++r)
        out.rows.push_back(quantize_a8(std::span<const float>(x.data() + std::size_t{r} * n, n)));
    return out;
}
struct DevA8 {
    Device<std::int8_t> codes;
    Device<float> deltas;
    Device<std::int32_t> sums;
    DevA8(std::uint32_t rows, std::uint32_t n) : codes(std::size_t{rows} * n), deltas(std::size_t{rows} * n / 32), sums(std::size_t{rows} * n / 32) {}
    explicit DevA8(const HostA8& h) : codes(h.codes()), deltas(h.deltas()), sums(h.sums()) {}
    bool equals(const HostA8& h) const {
        return same_bits(codes.down(), h.codes()) && same_bits(deltas.down(), h.deltas()) && same_bits(sums.down(), h.sums());
    }
};

// ---- host references over rows --------------------------------------------------------------------
std::vector<HcMix> host_mixes(const std::vector<float>& s, std::uint32_t rows, std::uint32_t hidden, const std::vector<std::uint16_t>& fn,
                              const std::vector<float>& base, const std::vector<float>& scale, float hc_eps, float norm_eps) {
    std::vector<HcMix> out(rows);
    const std::size_t n = std::size_t{hidden} * hc_copies;
    for (std::uint32_t r = 0; r < rows; ++r)
        out[r] = hc_mixes(std::span<const float>(s.data() + r * n, n), HcSiteWeights{fn, base, scale}, 20, hc_eps, norm_eps);
    return out;
}
std::vector<float> host_collapse(const std::vector<float>& s, const std::vector<HcMix>* mixes, std::uint32_t rows, std::uint32_t hidden) {
    std::vector<float> out(std::size_t{rows} * hidden);
    const std::size_t n = std::size_t{hidden} * hc_copies;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto pre = mixes ? (*mixes)[r].pre : hc_identity_pre();
        hc_collapse(std::span<const float>(s.data() + r * n, n), std::span<const float, hc_copies>(pre.data(), hc_copies),
                    std::span<float>(out.data() + std::size_t{r} * hidden, hidden));
    }
    return out;
}
std::vector<float> host_expand(const std::vector<float>& sub, const std::vector<float>& res, const std::vector<HcMix>& mixes,
                               std::uint32_t rows, std::uint32_t hidden) {
    const std::size_t n = std::size_t{hidden} * hc_copies;
    std::vector<float> out(rows * n);
    for (std::uint32_t r = 0; r < rows; ++r)
        hc_expand(std::span<const float>(sub.data() + std::size_t{r} * hidden, hidden), std::span<const float>(res.data() + r * n, n),
                  mixes[r], std::span<float>(out.data() + r * n, n));
    return out;
}
std::vector<float> host_norm(const std::vector<float>& x, const std::vector<float>& w, std::uint32_t rows, std::uint32_t n, float eps) {
    std::vector<float> out(x.size());
    for (std::uint32_t r = 0; r < rows; ++r)
        rms_norm(std::span<const float>(x.data() + std::size_t{r} * n, n), w, eps, std::span<float>(out.data() + std::size_t{r} * n, n));
    return out;
}
std::vector<float> host_q8(const std::vector<std::uint8_t>& w, std::uint32_t width, std::uint32_t first, std::uint32_t count,
                           std::uint32_t per_group, const HostA8& x) {
    std::vector<float> out(x.rows.size() * count);
    for (std::size_t r = 0; r < x.rows.size(); ++r) q8_0_a8_rows(w.data(), width, first, count, per_group, x.rows[r], out.data() + r * count);
    return out;
}
std::vector<float> host_f16(const std::vector<std::uint16_t>& w, std::uint32_t width, std::uint32_t count, const std::vector<float>& x,
                            std::uint32_t rows) {
    std::vector<float> out(std::size_t{rows} * count);
    for (std::uint32_t r = 0; r < rows; ++r) f16_rows(w.data(), width, 0, count, x.data() + std::size_t{r} * width, out.data() + std::size_t{r} * count);
    return out;
}
void host_rope(std::vector<float>& x, std::uint32_t rows, std::uint32_t vectors, std::uint32_t row_stride, std::uint32_t vector_stride,
               std::uint32_t offset, const RopeTable& t, const std::vector<std::uint32_t>& positions, bool inverse) {
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t v = 0; v < vectors; ++v)
            apply_rope(std::span<float>(x.data() + std::size_t{r} * row_stride + std::size_t{v} * vector_stride + offset, 2 * t.pairs), t,
                       t.first + positions[r], inverse);
}

// ---- device wrappers -------------------------------------------------------------------------------
std::vector<HcMix> dev_mixes(const Device<float>& s, std::uint32_t rows, std::uint32_t hidden, const Device<std::uint16_t>& fn,
                             const Device<float>& base, const Device<float>& scale, float hc_eps, float norm_eps) {
    Device<HcMix> out(rows);
    Device<float> partial(hc_mix_partial_floats(rows, hidden));
    HcMixView v;
    v.stream = s.p; v.fn = fn.p; v.base = base.p; v.scale = scale.p; v.rows = rows; v.hidden = hidden; v.iters = 20;
    v.hc_eps = hc_eps; v.norm_eps = norm_eps; v.out = out.p; v.partial = partial.p;
    enqueue_hc_mixes(v, nullptr);
    return out.down();
}
std::vector<float> dev_q8(const Device<std::uint8_t>& w, std::uint32_t width, std::uint32_t first, std::uint32_t count, std::uint32_t per_group,
                          const DevA8& x, std::uint32_t rows, std::uint32_t x_stride) {
    Device<float> out(std::size_t{rows} * count);
    Q8A8View v;
    v.weight = w.p; v.width = width; v.first = first; v.count = count; v.per_group = per_group; v.codes = x.codes.p;
    v.deltas = x.deltas.p; v.rows = rows; v.x_stride = x_stride; v.out_stride = count; v.out = out.p;
    enqueue_q8_0_a8(v, nullptr);
    return out.down();
}
std::vector<float> dev_f16(const Device<std::uint16_t>& w, std::uint32_t width, std::uint32_t count, const Device<float>& x, std::uint32_t rows) {
    Device<float> out(std::size_t{rows} * count);
    F16RowsView v;
    v.weight = w.p; v.width = width; v.count = count; v.x = x.p; v.rows = rows; v.x_stride = width; v.out_stride = count; v.out = out.p;
    enqueue_f16_rows(v, nullptr);
    return out.down();
}
// Fused norm: input x (or hc stream + mixes), optional RoPE, outputs FP32 and A8.
void dev_norm(NormView v, Device<float>& out, DevA8* a8) {
    v.out = out.p; v.out_stride = v.n;
    if (a8) { v.codes = a8->codes.p; v.deltas = a8->deltas.p; v.sums = a8->sums.p; v.a8_stride = v.n; }
    enqueue_norm(v, nullptr);
    check(hipDeviceSynchronize(), "norm sync");
}
void dev_rope(Device<float>& x, std::uint32_t rows, std::uint32_t vectors, std::uint32_t row_stride, std::uint32_t vector_stride,
              std::uint32_t offset, const RopeTable& t, const std::vector<std::uint32_t>& positions, bool inverse) {
    Device<float> c(t.cos), s(t.sin);
    Device<std::uint32_t> pos(positions);
    RopeView v;
    v.x = x.p; v.rows = rows; v.vectors = vectors; v.row_stride = row_stride; v.vector_stride = vector_stride; v.offset = offset;
    v.pairs = t.pairs; v.cos = c.p; v.sin = s.p; v.positions = pos.p; v.inverse = inverse;
    enqueue_rope(v, nullptr);
    check(hipDeviceSynchronize(), "rope sync");
}
ArgmaxCandidate dev_argmax_slice(const Device<float>& logits, std::uint32_t count, std::uint32_t first_index) {
    Device<ArgmaxCandidate> partial(argmax_partial_blocks(count)), out(1);
    enqueue_argmax(logits.p, 1, count, count, first_index, partial.p, out.p, nullptr);
    return out.down()[0];
}

__global__ void f16_select_kernel(std::uint32_t* out) {
    const unsigned h = blockIdx.x * blockDim.x + threadIdx.x;
    const auto a = __builtin_bit_cast(std::uint32_t, f16_to_float(static_cast<std::uint16_t>(h)));
    const auto b = __builtin_bit_cast(std::uint32_t, f16_to_float_select(static_cast<std::uint16_t>(h)));
    out[h] = a == b ? a : ~a;
}

// ---- synthetic --------------------------------------------------------------------------------------
// Explicit F8 views: do not rely on a model tensor happening to pass the loader's
// lossless-packing check. Exercise both wave counts, partial output blocks and F16
// subnormals against the original F16 weights and the serial host operator.
void f8_rows16(std::mt19937& rng) {
    for (const std::uint32_t rows : {9U, 16U})
        for (const auto [width, count] : {std::pair{6144U, 37U}, std::pair{256U, 8193U}}) {
            std::vector<std::uint16_t> weight(std::size_t{width} * count);
            for (std::size_t i = 0; i < weight.size(); ++i)
                weight[i] = to_f16(std::ldexp(static_cast<float>(static_cast<int>(i % 16U) - 8),
                                             static_cast<int>((i / 32U) % 10U) - 20));
            const auto packed = pack_f16_as_f8(weight, width);
            expect(packed.has_value(), "F8 rows16 fixture is losslessly representable");
            if (!packed) continue;
            const auto input = random_vector(rng, std::size_t{rows} * width, 1.0F);
            Device<std::uint8_t> dw(*packed);
            Device<float> dx(input), out(std::size_t{rows} * count);
            F16RowsView v;
            v.weight_f8 = dw.p; v.width = width; v.count = count; v.rows = rows;
            v.x = dx.p; v.x_stride = width; v.out = out.p; v.out_stride = count;
            enqueue_f16_rows(v, nullptr);
            expect(same_bits(out.down(), host_f16(weight, width, count, input, rows)),
                   "F8 projection equals original F16 operator, rows=" + std::to_string(rows) +
                       " width=" + std::to_string(width) + " count=" + std::to_string(count));
        }
    std::cout << "explicit F8 rows9/16 checked against original F16 operator\n";
}

void synthetic(std::mt19937& rng) {
    {
        // The select form of the half conversion (the fast site kernel) is f16_to_float bitwise on the device too.
        Device<std::uint32_t> d(65536);
        hipLaunchKernelGGL(f16_select_kernel, dim3(256), dim3(256), 0, nullptr, d.p);
        const auto got = d.down();
        bool same = true;
        for (std::uint32_t h = 0; h < 65536U; ++h)
            same = same && got[h] == __builtin_bit_cast(std::uint32_t, f16_to_float(static_cast<std::uint16_t>(h)));
        expect(same, "device f16_to_float_select = host f16_to_float for every half");
    }
    const std::uint32_t hidden = 5120, rows = 5, n = hidden * hc_copies;
    const auto s = random_vector(rng, std::size_t{rows} * n, 1.0F);
    const auto fn = random_f16(rng, std::size_t{hc_mix_count} * n, 0.02F);
    const auto base = random_vector(rng, hc_mix_count, 0.5F);
    const std::vector<float> scale{0.7F, 1.3F, 2.1F};
    Device<float> ds(s), dbase(base), dscale(scale);
    Device<std::uint16_t> dfn(fn);
    const auto hm = host_mixes(s, rows, hidden, fn, base, scale, 1e-6F, 1e-20F);
    expect(same_mix(dev_mixes(ds, rows, hidden, dfn, dbase, dscale, 1e-6F, 1e-20F), hm), "hc mixes (chunk partials) + Sinkhorn bitwise");
    Device<HcMix> dmix(hm);
    for (const bool identity : {true, false}) {
        Device<float> out(std::size_t{rows} * hidden);
        enqueue_hc_collapse(ds.p, identity ? nullptr : dmix.p, rows, hidden, out.p, nullptr);
        expect(same_bits(out.down(), host_collapse(s, identity ? nullptr : &hm, rows, hidden)), "hc collapse bitwise");
    }
    const auto sub = random_vector(rng, std::size_t{rows} * hidden, 1.0F);
    Device<float> dsub(sub), dexp(std::size_t{rows} * n);
    enqueue_hc_expand(dsub.p, ds.p, dmix.p, rows, hidden, dexp.p, nullptr);
    expect(same_bits(dexp.down(), host_expand(sub, s, hm, rows, hidden)), "hc expand bitwise");

    // Site launches (expansion + mixes + Sinkhorn + collapse + norm + A8), every variant twice (the counters
    // must come back to zero), against the host composition.
    for (const std::uint32_t sr : {1U, 2U, 3U, 6U, 8U, 9U}) {
        const auto res = random_vector(rng, std::size_t{sr} * n, 1.0F), sb = random_vector(rng, std::size_t{sr} * hidden, 1.0F);
        const auto em = host_mixes(res, sr, hidden, fn, base, scale, 1e-6F, 1e-20F);
        const auto cm = host_mixes(random_vector(rng, std::size_t{sr} * n, 1.0F), sr, hidden, fn, base, scale, 1e-6F, 1e-20F);
        const auto w = random_vector(rng, hidden, 1.0F);
        Device<float> dres(res), dsb(sb), dw(w), partial(hc_mix_partial_floats(sr, hidden)), st(std::size_t{sr} * n),
            out(std::size_t{sr} * hidden);
        Device<HcMix> dem(em), dcm(cm), mixes(sr);
        Device<std::uint32_t> counters(std::vector<std::uint32_t>(hc_site_counter_count(sr), 0U));
        for (const int variant : {0, 1, 2, 3}) {
            // 0: expand + mixes + norm; 1: stored stream + mixes + identity collapse; 2: expand + norm (head); 3: expand + mixes
            const bool expand = variant != 1, mix = variant != 2, norm = variant != 3;
            const auto stream_ref = expand ? host_expand(sb, res, em, sr, hidden) : res;
            const auto mix_ref = host_mixes(stream_ref, sr, hidden, fn, base, scale, 1e-6F, 1e-20F);
            const auto y_ref = host_norm(host_collapse(stream_ref, variant == 1 ? nullptr : &cm, sr, hidden), w, sr, hidden, 1e-20F);
            // each one-launch form twice (counters back at zero): the previous kernel, the fast one (by the row count, then
            // every tile with norm parts 1-8 in turn); then separate kernels
            for (int pass = 0; pass < 10; ++pass) {
                st.up(expand ? std::vector<float>(std::size_t{sr} * n, -1.0F) : res);  // an expansion must store every value
                DevA8 a8(sr, hidden);
                HcSiteView v;
                v.rows = sr; v.hidden = hidden; v.stream = st.p; v.launches = pass < 9 ? 1U : 2U; v.form = pass < 2 ? 1U : 2U;
                if (pass >= 4 && pass < 9) {
                    static constexpr std::uint32_t tiles[] = {1, 2, 4, 8, 1}, parts[] = {3, 5, 1, 8, 2};
                    v.tile = tiles[pass - 4];
                    v.norm_part = parts[pass - 4];
                }
                if (expand) { v.sub = dsb.p; v.residual = dres.p; v.expand_mixes = dem.p; }
                if (mix) { v.fn = dfn.p; v.base = dbase.p; v.scale = dscale.p; v.mixes_out = mixes.p; v.partial = partial.p; v.counters = counters.p; }
                if (norm) {
                    v.collapse_mixes = variant == 1 ? nullptr : dcm.p; v.norm_weight = dw.p; v.out = out.p; v.out_stride = hidden;
                    v.codes = a8.codes.p; v.deltas = a8.deltas.p; v.sums = a8.sums.p; v.a8_stride = hidden;
                }
                enqueue_hc_site(v, nullptr);
                {
                    // Every pointer the kernel receives lies in a declared range.
                    const auto access = launch_access(v);
                    bool covered = access.count < access.ranges.size();
                    for (const void* p : {static_cast<const void*>(v.sub), static_cast<const void*>(v.residual),
                                          static_cast<const void*>(v.expand_mixes), static_cast<const void*>(v.stream),
                                          static_cast<const void*>(v.fn), static_cast<const void*>(v.base), static_cast<const void*>(v.scale),
                                          static_cast<const void*>(v.mixes_out), static_cast<const void*>(v.partial),
                                          static_cast<const void*>(v.counters), static_cast<const void*>(v.collapse_mixes),
                                          static_cast<const void*>(v.norm_weight), static_cast<const void*>(v.out),
                                          static_cast<const void*>(v.codes), static_cast<const void*>(v.deltas),
                                          static_cast<const void*>(v.sums)}) {
                        if (!p) continue;
                        bool in = false;
                        for (std::uint32_t i = 0; i < access.count; ++i) {
                            const auto lo = reinterpret_cast<std::uintptr_t>(access.ranges[i].pointer);
                            const auto q = reinterpret_cast<std::uintptr_t>(p);
                            in = in || (q >= lo && q < lo + access.ranges[i].bytes);
                        }
                        covered = covered && in;
                    }
                    expect(covered, "site launch access covers every pointer");
                }
                bool ok = same_bits(st.down(), stream_ref);
                if (mix) ok = ok && same_mix(mixes.down(), mix_ref);
                if (norm) ok = ok && same_bits(out.down(), y_ref) && a8.equals(host_a8(y_ref, sr, hidden));
                if (mix) ok = ok && counters.down() == std::vector<std::uint32_t>(hc_site_counter_count(sr), 0U);
                expect(ok, "site launch bitwise, rows=" + std::to_string(sr) + " variant " + std::to_string(variant) + " pass " +
                               std::to_string(pass));
            }
        }
    }
    // Expansion + mixes + the collapse with the expansion's pre (collapse_out), every tile twice (counters back at zero),
    // against the host expansion / mixes / collapse; the inputs carry signed zeros, subnormals, infinities and NaNs.
    for (const std::uint32_t sr : {1U, 2U, 3U, 6U, 8U, 9U, 17U}) {
        auto res = random_vector(rng, std::size_t{sr} * n, 1.0F), sb = random_vector(rng, std::size_t{sr} * hidden, 1.0F);
        const float specials[] = {-0.0F, 0.0F, 1e-40F, -1e-40F, INFINITY, -INFINITY, NAN};
        for (std::size_t i = 0; i < sb.size(); i += 997) sb[i] = specials[(i / 997) % 7];
        for (std::size_t i = 3; i < res.size(); i += 1999) res[i] = specials[(i / 1999) % 7];
        const auto em = host_mixes(random_vector(rng, std::size_t{sr} * n, 1.0F), sr, hidden, fn, base, scale, 1e-6F, 1e-20F);
        Device<float> dres(res), dsb(sb), partial(hc_mix_partial_floats(sr, hidden)), st(std::size_t{sr} * n), x(std::size_t{sr} * hidden),
            sref(std::size_t{sr} * n), xref(std::size_t{sr} * hidden);
        Device<HcMix> dem(em), mixes(sr);
        // the separate kernels (STRIX_DS41_PREFILL_HC_FUSE=0): expansion, mixes, collapse
        enqueue_hc_expand(dsb.p, dres.p, dem.p, sr, hidden, sref.p, nullptr);
        enqueue_hc_collapse(sref.p, dem.p, sr, hidden, xref.p, nullptr);
        const auto stream_ref = sref.down(), x_ref = xref.down();
        const auto mix_ref = dev_mixes(sref, sr, hidden, dfn, dbase, dscale, 1e-6F, 1e-20F);
        Device<std::uint32_t> counters(std::vector<std::uint32_t>(hc_site_counter_count(sr), 0U));
        for (int pass = 0; pass < 8; ++pass) {
            st.up(std::vector<float>(std::size_t{sr} * n, -1.0F));
            x.up(std::vector<float>(std::size_t{sr} * hidden, -1.0F));
            HcSiteView v;
            v.rows = sr; v.hidden = hidden; v.stream = st.p; v.launches = 1; v.form = 2;
            static constexpr std::uint32_t tiles[] = {1, 2, 4, 8};
            v.tile = tiles[pass % 4];
            v.sub = dsb.p; v.residual = dres.p; v.expand_mixes = dem.p;
            v.fn = dfn.p; v.base = dbase.p; v.scale = dscale.p; v.mixes_out = mixes.p; v.partial = partial.p; v.counters = counters.p;
            v.collapse_out = x.p;
            enqueue_hc_site(v, nullptr);
            const auto access = launch_access(v);
            bool covered = false;
            for (std::uint32_t i = 0; i < access.count; ++i)
                covered = covered || access.ranges[i].pointer == static_cast<const void*>(v.collapse_out);
            expect(covered, "site launch access covers collapse_out");
            const bool ok = same_bits(st.down(), stream_ref) && same_mix(mixes.down(), mix_ref) && same_bits(x.down(), x_ref) &&
                            counters.down() == std::vector<std::uint32_t>(hc_site_counter_count(sr), 0U);
            expect(ok, "site launch with collapse bitwise, rows=" + std::to_string(sr) + " tile " + std::to_string(v.tile) + " pass " +
                           std::to_string(pass));
        }
    }
    {
        // In place (x == out): one block per row, never the split form.
        const auto x = random_vector(rng, std::size_t{2} * hidden, 2.0F), w = random_vector(rng, hidden, 1.0F);
        Device<float> dx(x), dw(w);
        enqueue_rms_norm(dx.p, dw.p, 2, hidden, hidden, 1e-20F, dx.p, nullptr);
        expect(same_bits(dx.down(), host_norm(x, w, 2, hidden, 1e-20F)), "RMSNorm in place bitwise");
    }

    // Norms: plain, with A8, fused with the collapse, with RoPE.
    for (const std::uint32_t nn : {512U, 1280U, 5120U}) {
        const auto x = random_vector(rng, std::size_t{rows} * nn, 2.0F), w = random_vector(rng, nn, 1.0F);
        Device<float> dx(x), dw(w), out(std::size_t{rows} * nn);
        DevA8 a8(rows, nn);
        NormView v; v.x = dx.p; v.x_stride = nn; v.weight = dw.p; v.rows = rows; v.n = nn; v.eps = 1e-20F;
        dev_norm(v, out, &a8);
        const auto ref = host_norm(x, w, rows, nn, 1e-20F);
        expect(same_bits(out.down(), ref) && a8.equals(host_a8(ref, rows, nn)), "norm + A8 bitwise n=" + std::to_string(nn));
    }
    {
        // More than 8 rows: one block per row instead of the split form.
        const std::uint32_t many = 12;
        const auto x = random_vector(rng, std::size_t{many} * hidden, 2.0F), w = random_vector(rng, hidden, 1.0F);
        Device<float> dx(x), dw(w), out(std::size_t{many} * hidden);
        DevA8 a8(many, hidden);
        NormView v; v.x = dx.p; v.x_stride = hidden; v.weight = dw.p; v.rows = many; v.n = hidden; v.eps = 1e-20F;
        dev_norm(v, out, &a8);
        const auto ref = host_norm(x, w, many, hidden, 1e-20F);
        expect(same_bits(out.down(), ref) && a8.equals(host_a8(ref, many, hidden)), "norm + A8 bitwise, 12 rows");
    }
    {
        const auto w = random_vector(rng, hidden, 1.0F);
        Device<float> dw(w), out(std::size_t{rows} * hidden);
        for (const bool identity : {true, false}) {
            DevA8 a8(rows, hidden);
            NormView v; v.hc_stream = ds.p; v.mixes = identity ? nullptr : dmix.p; v.weight = dw.p; v.rows = rows; v.n = hidden; v.eps = 1e-20F;
            dev_norm(v, out, &a8);
            const auto ref = host_norm(host_collapse(s, identity ? nullptr : &hm, rows, hidden), w, rows, hidden, 1e-20F);
            expect(same_bits(out.down(), ref) && a8.equals(host_a8(ref, rows, hidden)), "collapse + norm + A8 bitwise");
        }
    }
    const RopeConfig yarn{64, 160000.0, 65536.0, 16.0, 32.0, 1.0};
    const auto table = rope_table(yarn, 1000, 64);
    const std::vector<std::uint32_t> pos{0, 5, 17, 40, 63};
    Device<float> dcos(table.cos), dsin(table.sin);
    Device<std::uint32_t> dpos(pos);
    {
        const auto x = random_vector(rng, std::size_t{rows} * 512, 2.0F), w = random_vector(rng, 512, 1.0F);
        Device<float> dx(x), dw(w), out(std::size_t{rows} * 512);
        NormView v; v.x = dx.p; v.x_stride = 512; v.weight = dw.p; v.rows = rows; v.n = 512; v.eps = 1e-20F;
        v.rope_cos = dcos.p; v.rope_sin = dsin.p; v.positions = dpos.p; v.rope_offset = 448; v.pairs = 32;
        dev_norm(v, out, nullptr);
        auto ref = host_norm(x, w, rows, 512, 1e-20F);
        host_rope(ref, rows, 1, 512, 512, 448, table, pos, false);
        expect(same_bits(out.down(), ref), "kv norm + RoPE bitwise");
    }
    for (const bool inverse : {false, true}) {
        auto q = random_vector(rng, std::size_t{rows} * 64 * 512, 1.0F);
        Device<float> dq(q);
        host_rope(q, rows, 64, 64 * 512, 512, 448, table, pos, inverse);
        dev_rope(dq, rows, 64, 64 * 512, 512, 448, table, pos, inverse);
        expect(same_bits(dq.down(), q), std::string("RoPE bitwise (YaRN, 64 heads, tail at 448)") + (inverse ? " inverse" : ""));
    }
    {
        const auto o = random_vector(rng, std::size_t{rows} * 64 * 512, 1.0F);
        Device<float> dO(o);
        DevA8 a8(rows, 64 * 512);
        InverseRopeA8View v; v.o = dO.p; v.rows = rows; v.heads = 64; v.head_dim = 512; v.pairs = 32; v.rope_cos = dcos.p;
        v.rope_sin = dsin.p; v.positions = dpos.p; v.codes = a8.codes.p; v.deltas = a8.deltas.p; v.sums = a8.sums.p;
        enqueue_inverse_rope_a8(v, nullptr);
        HostA8 ref;
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::vector<float> row(o.begin() + static_cast<std::ptrdiff_t>(r) * 32768, o.begin() + static_cast<std::ptrdiff_t>(r + 1) * 32768);
            ref.rows.push_back(inverse_rope_a8(row, 64, 512, table, table.first + pos[r]));
        }
        expect(a8.equals(ref) && same_bits(dO.down(), o), "inverse RoPE + A8 bitwise (input untouched)");
    }

    // Q8_0 x A8 sites: plain, slice, grouped; rows 1..8.
    for (const std::uint32_t r : {1U, 3U, 8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U, 16U}) {
        const std::uint32_t width = 4096, count = 2048, groups = 2;
        const auto w = random_q8(rng, count, width);
        const auto xf = random_vector(rng, std::size_t{r} * groups * width, 1.0F);
        const auto xa = host_a8(xf, r, groups * width);
        Device<std::uint8_t> dw(w);
        DevA8 dx(xa);
        expect(same_bits(dev_q8(dw, width, 0, count, count / groups, dx, r, groups * width), host_q8(w, width, 0, count, count / groups, xa)),
               "Q8_0 x A8 grouped bitwise, rows=" + std::to_string(r));
        HostA8 first_slice;
        for (const auto& row : xa.rows) first_slice.rows.push_back(a8_slice(row, 0, width));
        expect(same_bits(dev_q8(dw, width, 100, 700, 700, dx, r, groups * width), host_q8(w, width, 100, 700, 700, first_slice)),
               "Q8_0 x A8 row slice bitwise, rows=" + std::to_string(r));
        const auto wf = random_f16(rng, std::size_t{300} * 6144, 0.02F);
        const auto xff = random_vector(rng, std::size_t{r} * 6144, 1.0F);
        Device<std::uint16_t> dwf(wf);
        Device<float> dxf(xff);
        expect(same_bits(dev_f16(dwf, 6144, 300, dxf, r), host_f16(wf, 6144, 300, xff, r)), "F16 projection bitwise, rows=" + std::to_string(r));
        if (r == 8U) {
            // Many rows (tiled kernel): rows not a multiple of the tile, outputs not a multiple of the block.
            const auto xm = random_vector(rng, std::size_t{37} * 6144, 1.0F);
            Device<float> dxm(xm);
            expect(same_bits(dev_f16(dwf, 6144, 300, dxm, 37), host_f16(wf, 6144, 300, xm, 37)), "F16 projection bitwise, rows=37");
            // LDS-tiled kernel over several row groups and a partial output tile; a width off the 128 slab (old kernel)
            for (const auto& [tw, tc, tm] : {std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>{1024U, 45U, 545U}, {1056U, 40U, 70U}}) {
                const auto wt = random_f16(rng, std::size_t{tc} * tw, 0.02F);
                const auto xt = random_vector(rng, std::size_t{tm} * tw, 1.0F);
                Device<std::uint16_t> dwt(wt);
                Device<float> dxt(xt);
                expect(same_bits(dev_f16(dwt, tw, tc, dxt, tm), host_f16(wt, tw, tc, xt, tm)),
                       "F16 projection bitwise, rows=" + std::to_string(tm) + " width=" + std::to_string(tw));
            }
        }
    }

    // CB4 rows (requantization candidates): the same sites on 4-bit indices into an int8 table give the bits of the
    // equivalent Q8_0 blocks (= product_q8_0 on the host): plain/grouped, row slice, K-split, every row count and form.
    {
        std::vector<std::int8_t> cb_table{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        std::shuffle(cb_table.begin(), cb_table.end(), rng);  // any order: the kernels look entries up, never assume sorting
        const auto cb4_pair = [&](std::uint32_t count, std::uint32_t width) {
            const auto q8 = random_q8(rng, count, width);  // only its scales are kept
            std::vector<std::uint8_t> emu(q8.size()), packed(std::size_t{count} * width / 32 * cb4_block_bytes);
            std::uniform_int_distribution<int> idx(0, 15);
            for (std::size_t b = 0; b < q8.size() / 34; ++b) {
                std::uint8_t* e = emu.data() + b * 34;
                std::uint8_t* p = packed.data() + b * cb4_block_bytes;
                e[0] = p[0] = q8[b * 34];
                e[1] = p[1] = q8[b * 34 + 1];
                int k[32];
                for (int& v : k) v = idx(rng);
                for (int j = 0; j < 32; ++j) e[2 + j] = static_cast<std::uint8_t>(cb_table[static_cast<std::size_t>(k[j])]);
                for (int j = 0; j < 16; ++j) p[2 + j] = static_cast<std::uint8_t>(k[j] | (k[j + 16] << 4));
            }
            return std::pair{emu, packed};
        };
        const auto cb4_view = [&](Q8A8View& v) {
            v.packed = 4;
            std::memcpy(v.table, cb_table.data(), 16);
        };
        for (const std::uint32_t r : {1U, 2U, 3U, 5U, 8U, 9U, 12U, 16U}) {
            const std::uint32_t width = 4096, count = 2048, groups = 2;
            const auto [emu, packed] = cb4_pair(count, width);
            const auto xa = host_a8(random_vector(rng, std::size_t{r} * groups * width, 1.0F), r, groups * width);
            Device<std::uint8_t> dw(packed);
            DevA8 dx(xa);
            for (const auto& [first, cnt, per] : {std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>{0U, count, count / groups},
                                                 {100U, 700U, 700U}, {0U, count, count}}) {
                HostA8 xs = xa;
                if (per == cnt) {  // one group: the first width of every row
                    xs.rows.clear();
                    for (const auto& row : xa.rows) xs.rows.push_back(a8_slice(row, 0, width));
                }
                Device<float> out(std::size_t{r} * cnt);
                Q8A8View v;
                v.weight = dw.p; v.width = width; v.first = first; v.count = cnt; v.per_group = per; v.codes = dx.codes.p;
                v.deltas = dx.deltas.p; v.rows = r; v.x_stride = groups * width; v.out_stride = cnt; v.out = out.p;
                cb4_view(v);
                enqueue_q8_0_a8(v, nullptr);
                expect(same_bits(out.down(), host_q8(emu, width, first, cnt, per, xs)),
                       "CB4 x A8 = Q8_0 form bitwise, rows=" + std::to_string(r) + " first=" + std::to_string(first) +
                           " count=" + std::to_string(cnt) + " per_group=" + std::to_string(per));
            }
        }
        // U5 / U6 rows: uniform codes [-16, 15] / [-32, 31] (bits 4.. of every weight in the high-bit words)
        for (const unsigned bits : {5U, 6U}) {
            const std::uint32_t width = 4096, count = 2048, groups = 2, bytes = bits == 5U ? 22U : 26U;
            const auto q8 = random_q8(rng, count, width);
            std::vector<std::uint8_t> emu(q8.size()), packed(std::size_t{count} * width / 32 * bytes);
            std::uniform_int_distribution<int> val(0, (1 << bits) - 1);
            for (std::size_t b = 0; b < q8.size() / 34; ++b) {
                std::uint8_t* e = emu.data() + b * 34;
                std::uint8_t* p = packed.data() + b * bytes;
                e[0] = p[0] = q8[b * 34];
                e[1] = p[1] = q8[b * 34 + 1];
                unsigned v[32];
                std::uint64_t high = 0;
                for (unsigned& x : v) x = static_cast<unsigned>(val(rng));
                for (unsigned j = 0; j < 32; ++j) {
                    e[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(v[j]) - (1 << (bits - 1))));
                    high |= static_cast<std::uint64_t>(v[j] >> 4U) << (bits == 5U ? j : 2U * j);
                }
                for (unsigned j = 0; j < 16; ++j) p[2 + j] = static_cast<std::uint8_t>((v[j] & 15U) | ((v[j + 16] & 15U) << 4U));
                std::memcpy(p + 18, &high, bits == 5U ? 4U : 8U);
            }
            Device<std::uint8_t> dw(packed);
            for (const std::uint32_t r : {1U, 3U, 8U, 9U, 12U, 16U}) {
                const auto xa = host_a8(random_vector(rng, std::size_t{r} * groups * width, 1.0F), r, groups * width);
                DevA8 dx(xa);
                Device<float> out(std::size_t{r} * count);
                Q8A8View v;
                v.weight = dw.p; v.width = width; v.first = 0; v.count = count; v.per_group = count / groups; v.codes = dx.codes.p;
                v.deltas = dx.deltas.p; v.rows = r; v.x_stride = groups * width; v.out_stride = count; v.out = out.p; v.packed = bits;
                enqueue_q8_0_a8(v, nullptr);
                expect(same_bits(out.down(), host_q8(emu, width, 0, count, count / groups, xa)),
                       "U" + std::to_string(bits) + " x A8 = Q8_0 form bitwise, rows=" + std::to_string(r));
            }
        }
        {  // K-split (attn_output_b over two ranks)
            const std::uint32_t width = 8192, count = 256, half = width / 2;
            const auto [emu, packed] = cb4_pair(count, width);
            Device<std::uint8_t> dw(packed);
            for (const std::uint32_t many : {1U, 4U, 8U, 9U, 12U, 16U}) {
                const auto xa = host_a8(random_vector(rng, std::size_t{many} * width, 1.0F), many, width);
                DevA8 dx(xa);
                std::vector<float> ref(std::size_t{many} * count);
                for (std::uint32_t r = 0; r < many; ++r)
                    q8_0_a8_split_rows(emu.data(), width, count, 2, xa.rows[r], ref.data() + std::size_t{r} * count);
                Device<float> out(std::size_t{many} * count);
                for (std::uint32_t p = 0; p < 2; ++p) {
                    Q8A8View v;
                    v.weight = dw.p; v.width = half; v.count = count; v.codes = dx.codes.p + p * half; v.deltas = dx.deltas.p + p * half / 32;
                    v.rows = many; v.x_stride = width; v.out_stride = count; v.out = out.p; v.row_width = width; v.k_offset = p * half;
                    v.accumulate = p > 0;
                    cb4_view(v);
                    enqueue_q8_0_a8(v, nullptr);
                }
                expect(same_bits(out.down(), ref), "CB4 K-split attn_output_b (2 parts) bitwise, rows=" + std::to_string(many));
            }
        }
        {  // the output A8 written by the product kernel (or its fallback) = enqueue_quantize_a8 of the output, bitwise
            const std::uint32_t width = 1024, count = 2048;
            const auto w = random_q8(rng, count, width);
            const auto [emu, packed] = cb4_pair(count, width);
            Device<std::uint8_t> dw(w), pw(packed);
            for (const bool cb4 : {false, true})
                for (const std::uint32_t r : {1U, 2U, 3U, 4U, 5U, 8U}) {
                    const auto xa = host_a8(random_vector(rng, std::size_t{r} * width, 1.0F), r, width);
                    DevA8 dx(xa), mine(r, count), ref(r, count);
                    Device<float> out(std::size_t{r} * count), plain(std::size_t{r} * count);
                    Q8A8View v;
                    v.weight = cb4 ? pw.p : dw.p; v.width = width; v.count = count; v.per_group = count; v.codes = dx.codes.p;
                    v.deltas = dx.deltas.p; v.rows = r; v.x_stride = width; v.out_stride = count; v.out = plain.p;
                    if (cb4) cb4_view(v);
                    enqueue_q8_0_a8(v, nullptr);
                    enqueue_quantize_a8(plain.p, count, r, ref.codes.p, ref.deltas.p, ref.sums.p, nullptr);
                    v.out = out.p; v.a8_codes = mine.codes.p; v.a8_deltas = mine.deltas.p; v.a8_sums = mine.sums.p;
                    enqueue_q8_0_a8(v, nullptr);
                    expect(same_bits(out.down(), plain.down()) && same_bits(mine.codes.down(), ref.codes.down()) &&
                               same_bits(mine.deltas.down(), ref.deltas.down()) && same_bits(mine.sums.down(), ref.sums.down()),
                           std::string("Q8_0 x A8 output A8 = quantize_a8 of the output bitwise, ") + (cb4 ? "CB4" : "Q8_0") +
                               " rows=" + std::to_string(r));
                }
        }
        {  // q_a + kv as one pair launch = the two launches, bitwise (Q8_0 and CB4, every row count)
            const std::uint32_t width = 5120, ca = 1280, cb = 512;
            const auto wa = random_q8(rng, ca, width), wb = random_q8(rng, cb, width);
            const auto [emu_a, packed_a] = cb4_pair(ca, width);
            const auto [emu_b, packed_b] = cb4_pair(cb, width);
            Device<std::uint8_t> da(wa), db(wb), pa(packed_a), pb(packed_b);
            for (const bool cb4 : {false, true})
                for (const std::uint32_t r : {1U, 2U, 3U, 4U, 5U, 8U}) {
                    const auto xa = host_a8(random_vector(rng, std::size_t{r} * width, 1.0F), r, width);
                    DevA8 dx(xa);
                    Device<float> oa(std::size_t{r} * ca), ob(std::size_t{r} * cb), pa_out(std::size_t{r} * ca), pb_out(std::size_t{r} * cb);
                    const auto view = [&](const std::uint8_t* w, std::uint32_t count, float* out) {
                        Q8A8View v;
                        v.weight = w; v.width = width; v.count = count; v.codes = dx.codes.p; v.deltas = dx.deltas.p; v.rows = r;
                        v.x_stride = width; v.out_stride = count; v.out = out;
                        if (cb4) cb4_view(v);
                        return v;
                    };
                    enqueue_q8_0_a8(view(cb4 ? pa.p : da.p, ca, oa.p), nullptr);
                    enqueue_q8_0_a8(view(cb4 ? pb.p : db.p, cb, ob.p), nullptr);
                    enqueue_q8_0_a8_pair(view(cb4 ? pa.p : da.p, ca, pa_out.p), view(cb4 ? pb.p : db.p, cb, pb_out.p), nullptr);
                    expect(same_bits(pa_out.down(), oa.down()) && same_bits(pb_out.down(), ob.down()),
                           std::string("q_a + kv pair launch = two launches bitwise, ") + (cb4 ? "CB4" : "Q8_0") + " rows=" + std::to_string(r));
                }
        }
    }

    // Dense prefill form (WMMA iu8, any width): rows not a multiple of the tile, uneven lane classes.
    for (const auto& [width, count, many] : {std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>{8192U, 128U, 40U},
                                             {1280U, 80U, 37U}, {1024U, 64U, 16U}}) {
        const auto w = random_q8(rng, count, width);
        const auto xa = host_a8(random_vector(rng, std::size_t{many} * width, 1.0F), many, width);
        Device<std::uint8_t> dw(w);
        DevA8 dx(xa);
        Device<float> out(std::size_t{many} * count);
        Q8A8DensePrefillView v;
        v.weight = dw.p; v.width = width; v.count = count; v.codes = dx.codes.p; v.deltas = dx.deltas.p; v.rows = many;
        v.x_stride = width; v.out = out.p; v.out_stride = count;
        enqueue_q8_0_a8_dense_prefill(v, nullptr);
        expect(same_bits(out.down(), host_q8(w, width, 0, count, count, xa)),
               "dense prefill (WMMA) = product_q8_0 bitwise, width=" + std::to_string(width) + " rows=" + std::to_string(many));
    }

    // Dense prefill with the outputs' A8 (o_a -> o_low): every tall form that reports it written gives the outputs of the same
    // call without A8 and the separate enqueue_quantize_a8 of them (GPU against GPU), at an output / A8 stride of two groups'
    // width with the second group's offset (the o_low layout). The second input set overflows some rows (activation deltas
    // 3e38): +-Inf outputs and all-NaN groups. The finite set's outputs are also product_q8_0 bitwise.
    {
        const std::uint32_t width = 1024;
        for (const std::uint32_t count : {128U, 96U})  // 96: an output tail inside the 64-output tiles
        for (const bool overflow : {false, true})
            for (const std::uint32_t many : {16U, 37U, 300U}) {
                const auto w = random_q8(rng, count, width);
                auto xa = host_a8(random_vector(rng, std::size_t{many} * width, 1.0F), many, width);
                if (overflow)
                    for (std::uint32_t r = 0; r < many; r += 3)
                        for (auto& d : xa.rows[r].deltas) d = 3e38F;
                const std::uint32_t stride = 2 * count;
                Device<std::uint8_t> dw(w);
                DevA8 dx(xa);
                for (const std::uint32_t form : {22U, 14U, 114U, 112U, 214U, 212U, 314U, 312U, 414U, 412U, 514U, 614U, 612U, 714U, 814U,
                                                 812U, 914U, 422U, 822U, 1514U, 1522U, 1614U, 1714U, 1722U, 1814U, 1914U, 1922U, 2014U,
                                                 2114U, 2122U, 2214U, 12U, 41U, 0U}) {
                    Q8A8DensePrefillView v;
                    v.weight = dw.p; v.width = width; v.count = count; v.codes = dx.codes.p; v.deltas = dx.deltas.p; v.rows = many;
                    v.x_stride = width; v.out_stride = stride; v.tall = form;
                    // the reference: the same form without A8, then the separate quantize of its outputs (contiguous)
                    Device<float> ref_out(std::vector<float>(std::size_t{many} * stride, -1.0F));
                    v.out = ref_out.p + count;
                    expect(!enqueue_q8_0_a8_dense_prefill(v, nullptr), "dense prefill without A8 reports none written");
                    const auto ro = ref_out.down();
                    std::vector<float> packed(std::size_t{many} * count);
                    for (std::uint32_t r = 0; r < many; ++r)
                        std::memcpy(packed.data() + std::size_t{r} * count, ro.data() + std::size_t{r} * stride + count, count * 4U);
                    if (!overflow && form == 0U)
                        expect(same_bits(packed, host_q8(w, width, 0, count, count, xa)), "dense prefill wide = product_q8_0 bitwise");
                    Device<float> dpacked(packed);
                    DevA8 ref_a8(many, count);
                    enqueue_quantize_a8(dpacked.p, count, many, ref_a8.codes.p, ref_a8.deltas.p, ref_a8.sums.p, nullptr);
                    const auto rc = ref_a8.codes.down();
                    const auto rd = ref_a8.deltas.down();
                    const auto rs = ref_a8.sums.down();
                    Device<float> out(std::vector<float>(std::size_t{many} * stride, -1.0F));
                    Device<std::int8_t> codes(std::vector<std::int8_t>(std::size_t{many} * stride, 0));
                    Device<float> deltas(std::vector<float>(std::size_t{many} * stride / 32U, -1.0F));
                    Device<std::int32_t> sums(std::vector<std::int32_t>(std::size_t{many} * stride / 32U, -1));
                    v.out = out.p + count;
                    v.a8_codes = codes.p + count; v.a8_deltas = deltas.p + count / 32U; v.a8_sums = sums.p + count / 32U; v.a8_stride = stride;
                    const bool written = enqueue_q8_0_a8_dense_prefill(v, nullptr);
                    const auto o = out.down();
                    const auto c = codes.down();
                    const auto d = deltas.down();
                    const auto sm = sums.down();
                    bool ok = same_bits(o, ro);
                    if (written)
                        for (std::uint32_t r = 0; r < many; ++r) {
                            ok = ok && std::memcmp(c.data() + std::size_t{r} * stride + count, rc.data() + std::size_t{r} * count, count) == 0;
                            ok = ok && std::memcmp(d.data() + std::size_t{r} * stride / 32U + count / 32U, rd.data() + std::size_t{r} * count / 32U,
                                                   count / 32U * 4U) == 0;
                            ok = ok && std::memcmp(sm.data() + std::size_t{r} * stride / 32U + count / 32U, rs.data() + std::size_t{r} * count / 32U,
                                                   count / 32U * 4U) == 0;
                        }
                    bool nan_group = false;
                    for (const float x : rd) nan_group = nan_group || x != x;
                    if (overflow) expect(nan_group, "overflow rows give all-NaN groups (NaN deltas)");
                    const bool expect_written = form != 41U && form != 0U;
                    expect(ok && written == expect_written, "dense prefill + output A8 bitwise, form " + std::to_string(form) + " count=" + std::to_string(count) + " rows=" +
                                                                std::to_string(many) + (overflow ? " overflow" : "") +
                                                                (written ? " (A8 written)" : " (A8 not written)"));
                }
            }
    }

    // o_b's two K-halves in one call (k_parts 2): the bits of the two calls (h0, then h1 accumulated), for the one-launch
    // form (auto, from 896 rows) and the two-launch fallback, with row and output tails.
    {
        const std::uint32_t width = 4096, row_width = 8192;
        for (const std::uint32_t count : {320U, 304U})
            for (const std::uint32_t many : {64U, 900U, 1030U}) {
                const auto w = random_q8(rng, count, row_width);
                const auto xa = host_a8(random_vector(rng, std::size_t{many} * row_width, 1.0F), many, row_width);
                Device<std::uint8_t> dw(w);
                DevA8 dx(xa);
                Device<float> ref(std::vector<float>(std::size_t{many} * count, -1.0F)), got(std::vector<float>(std::size_t{many} * count, -1.0F));
                Q8A8DensePrefillView v;
                v.weight = dw.p; v.width = width; v.row_width = row_width; v.count = count; v.rows = many; v.x_stride = row_width;
                v.out_stride = count; v.tall = 1U;
                for (std::uint32_t part = 0; part < 2U; ++part) {
                    auto q = v;
                    q.k_offset = part * width; q.codes = dx.codes.p + part * width; q.deltas = dx.deltas.p + part * width / 32U;
                    q.accumulate = part == 1U; q.out = ref.p;
                    enqueue_q8_0_a8_dense_prefill(q, nullptr);
                }
                auto q = v;
                q.k_parts = 2U; q.codes = dx.codes.p; q.deltas = dx.deltas.p; q.out = got.p;
                enqueue_q8_0_a8_dense_prefill(q, nullptr);
                expect(same_bits(got.down(), ref.down()), "o_b K-halves in one call = two calls bitwise, count=" + std::to_string(count) +
                                                              " rows=" + std::to_string(many));
            }
    }
    // The same with special activation deltas in either half or both (signed zero, subnormal, overflowing, infinite and
    // NaNs of different payloads and signs): output bytes against the two calls; and k_parts 2 views past the row rejected.
    {
        const std::uint32_t width = 4096, row_width = 8192, count = 320, many = 1030;
        const auto w = random_q8(rng, count, row_width);
        auto xa = host_a8(random_vector(rng, std::size_t{many} * row_width, 1.0F), many, row_width);
        const float specials[] = {-0.0F, 0.0F, 1e-40F, -1e-40F, 3e38F, INFINITY, -INFINITY,
                                  __builtin_bit_cast(float, 0x7fc00001U), __builtin_bit_cast(float, 0xffc12345U),
                                  __builtin_bit_cast(float, 0x7f800123U)};
        for (std::uint32_t r = 0; r < many; ++r) {
            auto& d = xa.rows[r].deltas;  // 256 groups: 0..127 in h0, 128..255 in h1
            const std::uint32_t kind = r % 4U;  // 0: none, 1: h0, 2: h1, 3: both
            for (std::uint32_t g = 0; g < 256U; g += 7U)
                if ((kind == 1U && g < 128U) || (kind == 2U && g >= 128U) || kind == 3U) d[g] = specials[(r + g) % 10U];
        }
        Device<std::uint8_t> dw(w);
        DevA8 dx(xa);
        Device<float> ref(std::size_t{many} * count), got(std::size_t{many} * count);
        Q8A8DensePrefillView v;
        v.weight = dw.p; v.width = width; v.row_width = row_width; v.count = count; v.rows = many; v.x_stride = row_width;
        v.out_stride = count; v.tall = 1U;
        for (std::uint32_t part = 0; part < 2U; ++part) {
            auto q = v;
            q.k_offset = part * width; q.codes = dx.codes.p + part * width; q.deltas = dx.deltas.p + part * width / 32U;
            q.accumulate = part == 1U; q.out = ref.p;
            enqueue_q8_0_a8_dense_prefill(q, nullptr);
        }
        auto q = v;
        q.k_parts = 2U; q.codes = dx.codes.p; q.deltas = dx.deltas.p; q.out = got.p;
        enqueue_q8_0_a8_dense_prefill(q, nullptr);
        const auto r0 = ref.down();
        bool special = false;
        for (const float x : r0) special = special || x != x || x == INFINITY || x == -INFINITY;
        expect(special, "special-delta fixture reaches NaN / Inf outputs");
        {
            const auto g0 = got.down();
            std::size_t diff = 0, nan_payload = 0, first = g0.size();
            for (std::size_t i = 0; i < g0.size(); ++i)
                if (std::memcmp(&g0[i], &r0[i], 4) != 0) {
                    ++diff;
                    if (g0[i] != g0[i] && r0[i] != r0[i]) ++nan_payload;
                    if (first == g0.size()) first = i;
                }
            if (diff)
                std::printf("special deltas: %zu differ (%zu both NaN, other payload); first %zu: one %08x two %08x\n", diff, nan_payload, first,
                            __builtin_bit_cast(std::uint32_t, g0[first]), __builtin_bit_cast(std::uint32_t, r0[first]));
            // every non-NaN output bitwise and NaN where the two calls give NaN; which NaN payload survives a two-NaN sum
            // follows the compiler's operand order inside the kernel (not controllable from source)
            expect(diff == nan_payload, "o_b K-halves in one call = two calls (finite bitwise, same NaN positions), special deltas");
        }
        {
            // NaN-free special deltas: one group per row (group 5 in h0, 133 in h1, or both) set to +0, a subnormal, 3e38
            // (overflowing products) or +Inf by row; every output bitwise (NaN only where the two calls give NaN), and the
            // fixture reaches infinite outputs and finite outputs of the zero / subnormal rows.
            auto xs = host_a8(random_vector(rng, std::size_t{many} * row_width, 1.0F), many, row_width);
            const float cat[] = {0.0F, 1e-40F, 3e38F, INFINITY};
            for (std::uint32_t r = 0; r < many; ++r) {
                const std::uint32_t c = r % 4U, place = (r / 4U) % 3U;
                if (place != 1U) xs.rows[r].deltas[5] = cat[c];
                if (place != 0U) xs.rows[r].deltas[133] = cat[c];
            }
            DevA8 dsx(xs);
            Device<float> ref2(std::size_t{many} * count), got2(std::size_t{many} * count);
            for (std::uint32_t part = 0; part < 2U; ++part) {
                auto q2 = v;
                q2.k_offset = part * width; q2.codes = dsx.codes.p + part * width; q2.deltas = dsx.deltas.p + part * width / 32U;
                q2.accumulate = part == 1U; q2.out = ref2.p;
                enqueue_q8_0_a8_dense_prefill(q2, nullptr);
            }
            auto q2 = v;
            q2.k_parts = 2U; q2.codes = dsx.codes.p; q2.deltas = dsx.deltas.p; q2.out = got2.p;
            enqueue_q8_0_a8_dense_prefill(q2, nullptr);
            const auto a = got2.down(), b = ref2.down();
            std::size_t diff = 0, both_nan = 0, inf = 0, finite_low = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (std::memcmp(&a[i], &b[i], 4) != 0) { ++diff; if (a[i] != a[i] && b[i] != b[i]) ++both_nan; }
                if (b[i] == INFINITY || b[i] == -INFINITY) ++inf;
                if ((i / count) % 4U < 2U && std::isfinite(b[i])) ++finite_low;
            }
            expect(inf > 0 && finite_low > 0, "NaN-free special deltas reach infinite and finite (zero / subnormal) outputs");
            expect(diff == both_nan, "o_b K-halves in one call = two calls, NaN-free special deltas (+0, subnormal, 3e38, +Inf)");
        }
        for (const std::uint32_t big : {0x80000000U, 0xC0000000U}) {
            auto b = q;
            b.width = big; b.row_width = big; b.x_stride = big;
            bool rejected = false;
            try { enqueue_q8_0_a8_dense_prefill(b, nullptr); } catch (const std::exception&) { rejected = true; }
            expect(rejected, "k_parts 2 view past the row rejected (width " + std::to_string(big) + ")");
        }
    }

    // K-split site (attn_output_b over two ranks): part 0 then part 1 accumulated, = q8_0_a8_split_rows.
    {
        const std::uint32_t width = 8192, count = 256, half = width / 2;
        const auto w = random_q8(rng, count, width);
        Device<std::uint8_t> dw(w);
        for (const std::uint32_t many : {1U, 3U, 8U, 40U}) {
            const auto xa = host_a8(random_vector(rng, std::size_t{many} * width, 1.0F), many, width);
            DevA8 dx(xa);
            std::vector<float> ref(std::size_t{many} * count);
            for (std::uint32_t r = 0; r < many; ++r) q8_0_a8_split_rows(w.data(), width, count, 2, xa.rows[r], ref.data() + std::size_t{r} * count);
            Device<float> out(std::size_t{many} * count);
            for (std::uint32_t p = 0; p < 2; ++p) {
                if (many <= 8) {
                    Q8A8View v;
                    v.weight = dw.p; v.width = half; v.count = count; v.codes = dx.codes.p + p * half; v.deltas = dx.deltas.p + p * half / 32;
                    v.rows = many; v.x_stride = width; v.out_stride = count; v.out = out.p; v.row_width = width; v.k_offset = p * half;
                    v.accumulate = p > 0;
                    enqueue_q8_0_a8(v, nullptr);
                } else {
                    Q8A8DensePrefillView v;
                    v.weight = dw.p; v.width = half; v.count = count; v.codes = dx.codes.p + p * half; v.deltas = dx.deltas.p + p * half / 32;
                    v.rows = many; v.x_stride = width; v.out = out.p; v.out_stride = count; v.row_width = width; v.k_offset = p * half;
                    v.accumulate = p > 0;
                    enqueue_q8_0_a8_dense_prefill(v, nullptr);
                }
            }
            expect(same_bits(out.down(), ref), "K-split attn_output_b (2 parts) bitwise, rows=" + std::to_string(many));
        }
    }

    // Argmax: ties inside and across blocks, a NaN, two-rank combine.
    auto logits = random_vector(rng, 129280, 3.0F);
    logits[90000] = logits[70000] = logits[4100] = 50.0F;
    Device<float> dl(logits);
    const auto single = argmax_slice(logits, 0);
    const auto d0 = dev_argmax_slice(dl, 129280, 0);
    expect(d0.index == single.index && d0.value == single.value && d0.index == 4100, "argmax: lowest index of the tie");
    Device<float> da(std::vector<float>(logits.begin(), logits.begin() + 64640)), db(std::vector<float>(logits.begin() + 64640, logits.end()));
    const auto ca = dev_argmax_slice(da, 64640, 0), cb = dev_argmax_slice(db, 64640, 64640);
    Device<ArgmaxCandidate> pa(std::vector<ArgmaxCandidate>{ca}), pb(std::vector<ArgmaxCandidate>{cb}), pc(1);
    enqueue_combine_argmax(pb.p, pa.p, 1, pc.p, nullptr);
    expect(pc.down()[0].index == 4100, "device combine of rank slices = single argmax");
    logits[123] = std::nanf("");
    dl.up(logits);
    expect(!dev_argmax_slice(dl, 129280, 0).finite, "argmax flags a NaN");

    // Embedding and Engram injection.
    const auto table16 = random_f16(rng, std::size_t{100} * hidden, 1.0F);
    const std::vector<std::uint32_t> toks{3, 99, 0, 3, 42};
    Device<std::uint16_t> dt(table16);
    Device<std::uint32_t> dtok(toks);
    Device<float> demb(std::size_t{rows} * n);
    enqueue_embedding(dt.p, dtok.p, rows, hidden, demb.p, nullptr);
    std::vector<float> hemb(std::size_t{rows} * n);
    for (std::uint32_t r = 0; r < rows; ++r) embed_token(table16.data(), hidden, toks[r], std::span<float>(hemb.data() + r * n, n));
    expect(same_bits(demb.down(), hemb), "embedding bitwise");
    const auto kv = random_vector(rng, std::size_t{rows} * hidden * (hc_copies + 1), 0.7F);
    const auto qn = random_vector(rng, n, 1.0F), kn = random_vector(rng, n, 1.0F);
    auto hs = s;
    for (std::uint32_t r = 0; r < rows; ++r)
        engram_inject(std::span<float>(hs.data() + r * n, n), std::span<const float>(kv.data() + std::size_t{r} * hidden * 5, hidden * 5), qn, kn, 1e-20F);
    Device<float> dkv(kv), dqn(qn), dkn(kn), dst(s);
    enqueue_engram_inject(dst.p, dkv.p, dqn.p, dkn.p, rows, hidden, 1e-20F, nullptr);
    expect(same_bits(dst.down(), hs), "Engram injection (chunked) bitwise");
    std::cout << "synthetic layer-site kernels checked\n";
}

// ---- real file chain ---------------------------------------------------------------------------------
struct File {
    const char* path;
    const strix::deepseek41::TargetModel* model;
    const strix::ManifestTensor& tensor(const std::string& name) const {
        for (const auto& t : model->manifest.tensors) if (t.name == name) return t;
        throw std::runtime_error("tensor not found: " + name);
    }
    std::vector<std::uint8_t> bytes(const std::string& name) const {
        const auto& t = tensor(name);
        std::ifstream in(path, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(t.absolute_offset));
        std::vector<std::uint8_t> out(t.byte_size);
        in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
        if (!in) throw std::runtime_error("short read: " + name);
        return out;
    }
    template <class T> std::vector<T> as(const std::string& name) const {
        const auto b = bytes(name);
        std::vector<T> out(b.size() / sizeof(T));
        std::memcpy(out.data(), b.data(), out.size() * sizeof(T));
        return out;
    }
};

void real_chain(const char* path) {
    const auto model = strix::deepseek41::validate_target(strix::deepseek41::read_file(path));
    const File f{path, &model};
    const auto& c = model.config;
    const std::uint32_t H = c.hidden_size, n = H * hc_copies, R = 3, V = c.vocab_size;
    const float eps = static_cast<float>(c.rms_norm_eps), hc_eps = static_cast<float>(c.hc_eps);
    const std::string p = "blk.1.";
    std::vector<std::uint32_t> prefix{0, 3923, 1514, 3923, 1514, 77, 5};
    const std::vector<std::uint32_t> tokens{prefix[4], prefix[5], prefix[6]};
    const std::vector<std::uint32_t> positions{4, 5, 6};

    // 1. embedding
    const auto embd = f.as<std::uint16_t>("token_embd.weight");
    std::vector<float> hs(std::size_t{R} * n);
    for (std::uint32_t r = 0; r < R; ++r) embed_token(embd.data(), H, tokens[r], std::span<float>(hs.data() + r * n, n));
    Device<std::uint16_t> dembd(embd);
    Device<std::uint32_t> dtok(tokens);
    Device<float> ds(std::size_t{R} * n);
    enqueue_embedding(dembd.p, dtok.p, R, H, ds.p, nullptr);
    expect(same_bits(ds.down(), hs), "real: embedding bitwise");

    // 2. Engram at layer 1
    EngramHasher hasher(model.engram);
    const auto all_rows = hasher.hash_sequence(prefix);
    const std::uint32_t cols = hasher.columns(), per_token = hasher.layers() * cols;
    std::vector<EngramRowRequest> req;
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::uint32_t k = 0; k < cols; ++k) req.push_back({0, all_rows[positions[r] * per_token + k]});
    EngramRowReader reader(path, engram_tables(model), 16, true);
    std::vector<std::uint8_t> raw(req.size() * engram_row_bytes);
    reader.read(req, raw);
    std::vector<float> erows(req.size() * engram_row_elements);
    engram_rows_to_bf16(raw, erows);
    Device<std::uint8_t> draw(raw);
    Device<float> derows(erows.size());
    enqueue_engram_rows_bf16(draw.p, static_cast<std::uint32_t>(req.size()), derows.p, nullptr);
    expect(same_bits(derows.down(), erows), "real: Engram rows decoded to bf16 bitwise");
    const auto ekv = f.as<std::uint16_t>(p + "engram_kv.weight");
    const std::uint32_t ein = cols * engram_row_elements, eout = H * (hc_copies + 1);
    const auto hkv = host_f16(ekv, ein, eout, erows, R);
    Device<std::uint16_t> dekv(ekv);
    expect(same_bits(dev_f16(dekv, ein, eout, derows, R), hkv), "real: engram_kv projection bitwise");
    const auto qn = f.as<float>(p + "engram_q_norm.weight"), kn = f.as<float>(p + "engram_k_norm.weight");
    for (std::uint32_t r = 0; r < R; ++r)
        engram_inject(std::span<float>(hs.data() + r * n, n), std::span<const float>(hkv.data() + std::size_t{r} * eout, eout), qn, kn, eps);
    Device<float> dkv(hkv), dqn(qn), dkn(kn);
    enqueue_engram_inject(ds.p, dkv.p, dqn.p, dkn.p, R, H, eps, nullptr);
    expect(same_bits(ds.down(), hs), "real: Engram injection bitwise");

    // 3. attention-site mixes; collapse (identity) + attn_norm + A8
    const auto afn = f.as<std::uint16_t>(p + "hc_attn_fn.weight");
    const auto abase = f.as<float>(p + "hc_attn_base.weight"), ascale = f.as<float>(p + "hc_attn_scale.weight");
    const auto hmA = host_mixes(hs, R, H, afn, abase, ascale, hc_eps, eps);
    Device<std::uint16_t> dafn(afn);
    Device<float> dabase(abase), dascale(ascale);
    expect(same_mix(dev_mixes(ds, R, H, dafn, dabase, dascale, hc_eps, eps), hmA), "real: attention mixes bitwise");
    const auto anw = f.as<float>(p + "attn_norm.weight");
    const auto xn = host_norm(host_collapse(hs, nullptr, R, H), anw, R, H, eps);
    const auto xa = host_a8(xn, R, H);
    Device<float> danw(anw), dxn(std::size_t{R} * H);
    DevA8 dxa(R, H);
    {
        NormView v; v.hc_stream = ds.p; v.weight = danw.p; v.rows = R; v.n = H; v.eps = eps;
        dev_norm(v, dxn, &dxa);
    }
    expect(same_bits(dxn.down(), xn) && dxa.equals(xa), "real: collapse + attn_norm + A8 bitwise");

    // 4. Q / KV (layer 1: window-only, plain theta)
    const RopeConfig rope{c.qk_rope_head_dim, c.rope_theta, 0.0, 1.0, 32.0, 1.0};
    const auto table = rope_table(rope, 0, 8);
    Device<float> dcos(table.cos), dsin(table.sin);
    Device<std::uint32_t> dpos(positions);
    const std::uint32_t D = c.head_dim, heads = c.num_attention_heads, tail = D - c.qk_rope_head_dim, QA = c.q_lora_rank;
    const auto wqa = f.bytes(p + "attn_q_a.weight");
    Device<std::uint8_t> dwqa(wqa);
    const auto qa = host_q8(wqa, H, 0, QA, QA, xa);
    Device<float> dqa(dev_q8(dwqa, H, 0, QA, QA, dxa, R, H));
    expect(same_bits(dqa.down(), qa), "real: attn_q_a (A8) bitwise");
    const auto qnw = f.as<float>(p + "attn_q_a_norm.weight");
    const auto qan = host_norm(qa, qnw, R, QA, eps);
    const auto qaa = host_a8(qan, R, QA);
    Device<float> dqnw(qnw), dqan(std::size_t{R} * QA);
    DevA8 dqaa(R, QA);
    {
        NormView v; v.x = dqa.p; v.x_stride = QA; v.weight = dqnw.p; v.rows = R; v.n = QA; v.eps = eps;
        dev_norm(v, dqan, &dqaa);
    }
    expect(same_bits(dqan.down(), qan) && dqaa.equals(qaa), "real: q_a norm (FP32 + A8) bitwise");
    const auto wqb = f.bytes(p + "attn_q_b.weight");
    Device<std::uint8_t> dwqb(wqb);
    auto q = host_q8(wqb, QA, 0, heads * D, heads * D, qaa);
    Device<float> dq(dev_q8(dwqb, QA, 0, heads * D, heads * D, dqaa, R, QA));
    expect(same_bits(dq.down(), q), "real: attn_q_b (A8) bitwise");
    host_rope(q, R, heads, heads * D, D, tail, table, positions, false);
    {
        RopeView v; v.x = dq.p; v.rows = R; v.vectors = heads; v.row_stride = heads * D; v.vector_stride = D; v.offset = tail;
        v.pairs = table.pairs; v.cos = dcos.p; v.sin = dsin.p; v.positions = dpos.p;
        enqueue_rope(v, nullptr);
    }
    expect(same_bits(dq.down(), q), "real: query RoPE bitwise");
    const auto wkv = f.bytes(p + "attn_kv.weight");
    Device<std::uint8_t> dwkv(wkv);
    const auto kv = host_q8(wkv, H, 0, D, D, xa);
    Device<float> dkvr(dev_q8(dwkv, H, 0, D, D, dxa, R, H));
    expect(same_bits(dkvr.down(), kv), "real: attn_kv (A8) bitwise");
    const auto kvnw = f.as<float>(p + "attn_kv_a_norm.weight");
    auto kvn = host_norm(kv, kvnw, R, D, eps);
    host_rope(kvn, R, 1, D, D, tail, table, positions, false);
    Device<float> dkvnw(kvnw), dkvn(std::size_t{R} * D);
    {
        NormView v; v.x = dkvr.p; v.x_stride = D; v.weight = dkvnw.p; v.rows = R; v.n = D; v.eps = eps;
        v.rope_cos = dcos.p; v.rope_sin = dsin.p; v.positions = dpos.p; v.rope_offset = tail; v.pairs = table.pairs;
        dev_norm(v, dkvn, nullptr);
    }
    expect(same_bits(dkvn.down(), kvn), "real: kv norm + RoPE bitwise");

    // 5. stand-in attention output o = q: inverse RoPE + A8, grouped output_a, A8, output_b
    HostA8 oa8;
    for (std::uint32_t r = 0; r < R; ++r) {
        std::vector<float> row(q.begin() + static_cast<std::ptrdiff_t>(r) * heads * D, q.begin() + static_cast<std::ptrdiff_t>(r + 1) * heads * D);
        oa8.rows.push_back(inverse_rope_a8(row, heads, D, table, positions[r]));
    }
    DevA8 doa8(R, heads * D);
    {
        InverseRopeA8View v; v.o = dq.p; v.rows = R; v.heads = heads; v.head_dim = D; v.pairs = table.pairs; v.rope_cos = dcos.p;
        v.rope_sin = dsin.p; v.positions = dpos.p; v.codes = doa8.codes.p; v.deltas = doa8.deltas.p; v.sums = doa8.sums.p;
        enqueue_inverse_rope_a8(v, nullptr);
    }
    expect(doa8.equals(oa8), "real: inverse RoPE + A8 of the attention output bitwise");
    const std::uint32_t groups = c.o_groups, gin = heads * D / groups, gout = c.o_lora_rank;
    const auto woa = f.bytes(p + "attn_output_a.weight");
    Device<std::uint8_t> dwoa(woa);
    const auto oa = host_q8(woa, gin, 0, groups * gout, gout, oa8);
    Device<float> doa(dev_q8(dwoa, gin, 0, groups * gout, gout, doa8, R, heads * D));
    expect(same_bits(doa.down(), oa), "real: grouped attn_output_a (A8) bitwise");
    const auto oaa = host_a8(oa, R, groups * gout);
    DevA8 doaa(R, groups * gout);
    enqueue_quantize_a8(doa.p, groups * gout, R, doaa.codes.p, doaa.deltas.p, doaa.sums.p, nullptr);
    expect(doaa.equals(oaa), "real: A8 of the grouped output bitwise");
    const auto wob = f.bytes(p + "attn_output_b.weight");
    Device<std::uint8_t> dwob(wob);
    const auto ob = host_q8(wob, groups * gout, 0, H, H, oaa);
    Device<float> dob(dev_q8(dwob, groups * gout, 0, H, H, doaa, R, groups * gout));
    expect(same_bits(dob.down(), ob), "real: attn_output_b (A8) bitwise");

    // 6. expansion, FFN-site mixes, collapse (attention pre) + ffn_norm + A8, expansion (MoE stand-in)
    const auto s1 = host_expand(ob, hs, hmA, R, H);
    Device<HcMix> dmA(hmA);
    Device<float> ds1(std::size_t{R} * n);
    enqueue_hc_expand(dob.p, ds.p, dmA.p, R, H, ds1.p, nullptr);
    expect(same_bits(ds1.down(), s1), "real: attention expansion bitwise");
    const auto ffn = f.as<std::uint16_t>(p + "hc_ffn_fn.weight");
    const auto fbase = f.as<float>(p + "hc_ffn_base.weight"), fscale = f.as<float>(p + "hc_ffn_scale.weight");
    const auto hmF = host_mixes(s1, R, H, ffn, fbase, fscale, hc_eps, eps);
    Device<std::uint16_t> dffn(ffn);
    Device<float> dfbase(fbase), dfscale(fscale);
    expect(same_mix(dev_mixes(ds1, R, H, dffn, dfbase, dfscale, hc_eps, eps), hmF), "real: FFN mixes bitwise");
    const auto fnw = f.as<float>(p + "ffn_norm.weight");
    const auto yn = host_norm(host_collapse(s1, &hmA, R, H), fnw, R, H, eps);
    Device<float> dfnw(fnw), dyn(std::size_t{R} * H);
    DevA8 dya(R, H);
    {
        NormView v; v.hc_stream = ds1.p; v.mixes = dmA.p; v.weight = dfnw.p; v.rows = R; v.n = H; v.eps = eps;
        dev_norm(v, dyn, &dya);
    }
    expect(same_bits(dyn.down(), yn) && dya.equals(host_a8(yn, R, H)), "real: collapse (attention pre) + ffn_norm + A8 bitwise");
    const auto s2 = host_expand(yn, s1, hmF, R, H);
    Device<HcMix> dmF(hmF);
    Device<float> ds2(std::size_t{R} * n);
    enqueue_hc_expand(dyn.p, ds1.p, dmF.p, R, H, ds2.p, nullptr);
    expect(same_bits(ds2.down(), s2), "real: FFN expansion bitwise");

    // 7. final collapse + output_norm + A8, vocabulary-split head and argmax
    const auto onw = f.as<float>("output_norm.weight");
    const auto zn = host_norm(host_collapse(s2, &hmF, R, H), onw, R, H, eps);
    const auto za = host_a8(zn, R, H);
    Device<float> donw(onw), dzn(std::size_t{R} * H);
    DevA8 dza(R, H);
    {
        NormView v; v.hc_stream = ds2.p; v.mixes = dmF.p; v.weight = donw.p; v.rows = R; v.n = H; v.eps = eps;
        dev_norm(v, dzn, &dza);
    }
    expect(same_bits(dzn.down(), zn) && dza.equals(za), "real: final collapse + output_norm + A8 bitwise");

    // 6b. the same three boundaries as single site launches
    {
        Device<std::uint32_t> counters(std::vector<std::uint32_t>(hc_site_counter_count(R), 0U));
        Device<float> partial(hc_mix_partial_floats(R, H)), st(std::size_t{R} * n), out(std::size_t{R} * H);
        Device<HcMix> mixes(R);
        auto site = [&](const float* sub, const float* res, const HcMix* em, const float* in_stream, const std::uint16_t* fn,
                        const float* base, const float* scale, const HcMix* cm, const float* w, DevA8& a8, std::uint32_t form) {
            if (!sub) check(hipMemcpy(st.p, in_stream, std::size_t{R} * n * 4, hipMemcpyDeviceToDevice), "copy");
            HcSiteView v;
            v.rows = R; v.hidden = H; v.sub = sub; v.residual = res; v.expand_mixes = em; v.stream = st.p;
            v.fn = fn; v.base = base; v.scale = scale; v.hc_eps = hc_eps; v.mix_norm_eps = eps; v.mixes_out = mixes.p;
            v.partial = partial.p; v.counters = counters.p; v.collapse_mixes = cm; v.norm_weight = w; v.norm_eps = eps;
            v.out = out.p; v.out_stride = H; v.codes = a8.codes.p; v.deltas = a8.deltas.p; v.sums = a8.sums.p; v.a8_stride = H;
            v.launches = 1;
            v.form = form;
            enqueue_hc_site(v, nullptr);
            check(hipDeviceSynchronize(), "site sync");
        };
        for (const std::uint32_t form : {1U, 2U}) {
            const std::string fs = form == 1 ? " (previous kernel)" : " (fast kernel)";
            DevA8 a1(R, H), a2(R, H), a3(R, H);
            site(nullptr, nullptr, nullptr, ds.p, dafn.p, dabase.p, dascale.p, nullptr, danw.p, a1, form);
            expect(same_mix(mixes.down(), hmA) && same_bits(out.down(), xn) && a1.equals(xa),
                   "real: site launch = attention mixes + identity collapse + attn_norm + A8" + fs);
            site(dob.p, ds.p, dmA.p, nullptr, dffn.p, dfbase.p, dfscale.p, dmA.p, dfnw.p, a2, form);
            expect(same_bits(st.down(), s1) && same_mix(mixes.down(), hmF) && same_bits(out.down(), yn) && a2.equals(host_a8(yn, R, H)),
                   "real: site launch = attention expansion + FFN mixes + collapse + ffn_norm + A8" + fs);
            site(dyn.p, ds1.p, dmF.p, nullptr, nullptr, nullptr, nullptr, dmF.p, donw.p, a3, form);
            expect(same_bits(st.down(), s2) && same_bits(out.down(), zn) && a3.equals(za),
                   "real: site launch = FFN expansion + final collapse + output_norm + A8" + fs);
        }
    }
    const auto head = f.bytes("output.weight");
    Device<std::uint8_t> dhead(head);
    const auto full = host_q8(head, H, 0, V, V, za);
    std::vector<ArgmaxCandidate> combined(R);
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        const auto sl = vocab_slice(V, 2, rank);
        Device<float> part(dev_q8(dhead, H, sl.first, sl.count, sl.count, dza, R, H));
        const auto got = part.down();
        bool same = true;
        for (std::uint32_t r = 0; r < R; ++r)
            same = same && std::memcmp(got.data() + std::size_t{r} * sl.count, full.data() + std::size_t{r} * V + sl.first, sl.count * 4) == 0;
        expect(same, "real: head slice of rank " + std::to_string(rank) + " (A8) bitwise");
        Device<ArgmaxCandidate> partial(std::size_t{R} * argmax_partial_blocks(sl.count)), cand(R);
        enqueue_argmax(part.p, R, sl.count, sl.count, sl.first, partial.p, cand.p, nullptr);
        const auto cands = cand.down();
        for (std::uint32_t r = 0; r < R; ++r) combined[r] = rank ? combine_argmax(combined[r], cands[r]) : cands[r];
    }
    for (std::uint32_t r = 0; r < R; ++r) {
        const auto single = argmax_slice(std::span<const float>(full.data() + std::size_t{r} * V, V), 0);
        expect(combined[r].index == single.index && combined[r].value == single.value && single.finite,
               "real: two-rank argmax = single-node argmax");
        std::cout << "row " << r << " greedy token " << single.index << " logit " << single.value << '\n';
    }
    std::cout << "real layer-1 chain checked\n";
}

} // namespace

// STRIX_DS41_F16_BENCH=ROWS: warm time of the F16 projection at the prefill shapes (Engram 6144 -> 20480, compressor
// 4096 -> 512, indexer q_b 1024 -> 4096) for ROWS rows; STRIX_DS41_F16_TILED=0 times the previous kernel.
void f16_bench(std::uint32_t rows) {
    std::mt19937 rng(7);
    for (const auto& [width, count] : {std::pair<std::uint32_t, std::uint32_t>{6144U, 20480U}, {4096U, 512U}, {1024U, 4096U}}) {
        const Device<std::uint16_t> w(random_f16(rng, std::size_t{count} * width, 0.02F));
        const Device<float> x(random_vector(rng, std::size_t{rows} * width, 1.0F));
        Device<float> out(std::size_t{rows} * count);
        F16RowsView v;
        v.weight = w.p; v.width = width; v.count = count; v.x = x.p; v.rows = rows; v.x_stride = width; v.out_stride = count; v.out = out.p;
        hipEvent_t e0, e1;
        (void)hipEventCreate(&e0);
        (void)hipEventCreate(&e1);
        enqueue_f16_rows(v, nullptr);
        (void)hipEventRecord(e0, nullptr);
        constexpr int reps = 5;
        for (int k = 0; k < reps; ++k) enqueue_f16_rows(v, nullptr);
        (void)hipEventRecord(e1, nullptr);
        (void)hipEventSynchronize(e1);
        float ms = 0.0F;
        (void)hipEventElapsedTime(&ms, e0, e1);
        ms /= reps;
        std::cout << "F16 rows " << rows << " x " << width << " -> " << count << ": " << ms << " ms, "
                  << 2.0 * rows * width * count / (ms * 1e-3) / 1e12 << " TFLOPS\n";
        (void)hipEventDestroy(e0);
        (void)hipEventDestroy(e1);
    }
}

int main() {
    if (const char* b = std::getenv("STRIX_DS41_F16_BENCH")) {
        f16_bench(static_cast<std::uint32_t>(std::stoul(b)));
        return 0;
    }
    std::mt19937 rng(20261019);
    if (std::getenv("STRIX_DS41_F8_ROWS16_ONLY")) f8_rows16(rng);
    else {
        synthetic(rng);
        f8_rows16(rng);
        if (const char* path = std::getenv("STRIX_DEEPSEEK41_GGUF")) real_chain(path);
    }
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "hip ds41 layer tests passed\n";
    return 0;
}
