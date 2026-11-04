// g4-dattn-f1 / f2 (FAST decode attention of the global layers, FP16 / KV8 operands) and the per-stream KV mirror.
//   mirror: sync converts committed rows in 64-row steps, halves == host RNE casts, a bad tag sets bit 2;
//   kernel: bytes independent of the mirror's sync state (none / 128 rows / synced), of the rows sharing the pass
//           (R = 1 vs R = 4 consecutive positions with overlay rows, and mixed streams), run to run; close to the
//           exact FP32 attention; poisoned tag of an unmirrored row sets bit 2.
// --bench: kernel time of the exact long-span path vs g4-dattn-f1 at 8K / 32K / 64K positions, one row.
#include <hip/hip_runtime.h>
#include "gemma4/kernels/attention_fast_decode.hpp"
#include "gemma4/kernels/a8_ops.hpp"
#include "gemma4/kernels/internal.hpp"
#include "gemma4/kv/fp16_mirror.hpp"
#include "fixture.hpp"
#include <cmath>
#include <limits>
#include <cstring>
#include <iostream>
#include <random>
#include <string_view>

using namespace strix::gemma4;
using namespace strix::gemma4::gpu;
namespace {
template <class T> Buffer device(const std::vector<T>& values) {
    Buffer b(values.size() * sizeof(T));
    b.upload(values.data(), b.size());
    return b;
}
std::vector<float> read(const Buffer& b, std::size_t count) {
    std::vector<float> v(count);
    b.download(v.data(), count * sizeof(float));
    return v;
}
std::vector<float> noise(std::size_t n, unsigned seed, float scale) {
    std::vector<float> x(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-scale, scale);
    for (auto& a : x)
        a = u(rng);
    return x;
}
std::uint16_t half_bits(float x) {
    const _Float16 h = static_cast<_Float16>(x);
    std::uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
unsigned flag_of(const Buffer& failure) {
    unsigned f = 0;
    failure.download(&f, 4);
    return f;
}
constexpr unsigned hcount = 8, dim = 512, width = 2 * dim;
struct Stream {
    DeviceStream::Impl impl;
    std::vector<float> keys, values; // positions [0, total)
    Buffer overlay_k, overlay_v;
    std::uint64_t frontier = 0;
    Stream(unsigned slots, std::uint64_t committed, unsigned total, unsigned seed) : frontier(committed) {
        keys = noise(std::size_t(total) * width, seed, 1.5F);
        values = noise(std::size_t(total) * width, seed + 1, 1.0F);
        impl.cache.resize(1);
        auto& c = impl.cache[0];
        c.width = width;
        c.slots = slots;
        c.global = true;
        std::vector<float> k(std::size_t(slots) * width), v(k.size());
        std::vector<std::uint64_t> tags(slots, UINT64_MAX);
        for (std::uint64_t p = 0; p < committed; ++p) {
            tags[p] = p;
            std::copy_n(keys.data() + p * width, width, k.data() + p * width);
            std::copy_n(values.data() + p * width, width, v.data() + p * width);
        }
        c.k = device(k);
        c.v = device(v);
        c.tags = device(tags);
        impl.frontier = committed;
        std::vector<float> ok(keys.begin() + std::ptrdiff_t(committed * width), keys.end()),
            ov(values.begin() + std::ptrdiff_t(committed * width), values.end());
        ok.resize(std::max<std::size_t>(ok.size(), width));
        ov.resize(ok.size());
        overlay_k = device(ok);
        overlay_v = device(ov);
    }
    KvRead read(std::uint64_t position, unsigned candidates) const {
        const auto& c = impl.cache[0];
        return {c.k.as<float>(), c.v.as<float>(), c.tags.as<std::uint64_t>(), overlay_k.as<float>(),
                overlay_v.as<float>(), frontier, position, width, c.slots, candidates, 512, true};
    }
};
struct Fixture {
    fixture::Tables tables;
    Buffer table, norm_weight, failure, scratch, out, scores, maxima, partials, counters, positions_d, reads_d, mirrors_d, q_d;
    LinearView q_norm;
    MirrorFormat format = MirrorFormat::f16;
    std::vector<float> q;
    explicit Fixture(unsigned table_positions) : tables(table_positions) {
        table = device(tables.global);
        const auto packed = fixture::packed(WeightFormat::f32, 1, dim);
        norm_weight = device(packed);
        q_norm = {norm_weight.as<std::uint8_t>(), 1, dim, packed_row_bytes(WeightFormat::f32, dim), WeightFormat::f32};
        failure.resize(4);
        out.resize(8ULL * hcount * dim * 4);
        q = noise(8ULL * hcount * dim, 77, 2.0F);
        q_d = device(q);
        positions_d.resize(8 * 8);
        reads_d.resize(8 * sizeof(KvRead));
        mirrors_d.resize(8 * sizeof(Fp16MirrorRead));
    }
    void ensure(unsigned stride) {
        const auto need = fast_decode_attention_scratch_bytes(8, stride);
        if (scratch.size() < need)
            scratch.resize(need);
        const auto chunks = (stride + 63U) / 64U;
        if (scores.size() < 8ULL * hcount * stride * 4) scores.resize(8ULL * hcount * stride * 4);
        if (maxima.size() < 8ULL * hcount * (chunks + 1) * 4) maxima.resize(8ULL * hcount * (chunks + 1) * 4);
        if (partials.size() < 8ULL * hcount * chunks * 513 * 4) partials.resize(8ULL * hcount * chunks * 513 * 4);
        if (!counters.size()) { counters.resize(8ULL * hcount * 4); counters.zero(); }
    }
    void stage(const std::vector<std::uint64_t>& positions, const std::vector<KvRead>& reads,
               const std::vector<Fp16MirrorRead>& mirrors, const std::vector<unsigned>& q_rows) {
        positions_d.upload(positions.data(), positions.size() * 8);
        reads_d.upload(reads.data(), reads.size() * sizeof(KvRead));
        mirrors_d.upload(mirrors.data(), mirrors.size() * sizeof(Fp16MirrorRead));
        std::vector<float> rows_q;
        for (const auto r : q_rows)
            rows_q.insert(rows_q.end(), q.begin() + std::ptrdiff_t(r) * hcount * dim,
                          q.begin() + std::ptrdiff_t(r + 1) * hcount * dim);
        q_d.upload(rows_q.data(), rows_q.size() * 4);
    }
    std::vector<float> fast(unsigned rows, unsigned stride, unsigned* flag = nullptr, A8Rows quant = {}) {
        ensure(stride);
        failure.zero();
        CHECK(fast_decode_attention(q_d.as<float>(), q_norm, positions_d.as<std::uint64_t>(), table.as<float>(),
                                    reads_d.as<KvRead>(), mirrors_d.as<Fp16MirrorRead>(), rows, hcount, dim, stride,
                                    out.as<float>(), scratch.as<float>(), scratch.size(), failure.as<unsigned>(),
                                    nullptr, true, true, true, format, quant));
        synchronize();
        const auto f = flag_of(failure);
        if (flag) *flag = f;
        else CHECK(f == 0);
        return read(out, std::size_t(rows) * hcount * dim);
    }
    std::vector<float> exact(unsigned rows, unsigned stride) {
        ensure(stride);
        failure.zero();
        CHECK(fused_attention(q_d.as<float>(), q_norm, positions_d.as<std::uint64_t>(), table.as<float>(),
                              reads_d.as<KvRead>(), rows, hcount, dim, stride, true, out.as<float>(),
                              failure.as<unsigned>(), nullptr, true, true, true, scores.as<float>(),
                              maxima.as<float>(), partials.as<float>(), counters.as<unsigned>()));
        synchronize();
        CHECK(flag_of(failure) == 0);
        return read(out, std::size_t(rows) * hcount * dim);
    }
};
unsigned bucket(std::uint64_t last) { return unsigned((last + 1U + 63U) / 64U * 64U); }
// Host KV8 group (the declared operations): d = half(amax / 127), code = clamp(rint(x / d)).
void kv8_host(const float* x, std::int8_t* codes, std::uint16_t& scale) {
    float amax = 0.0F;
    for (unsigned j = 0; j < kv8_group; ++j) amax = std::max(amax, std::abs(x[j]));
    scale = half_bits(amax / 127.0F);
    _Float16 h;
    std::memcpy(&h, &scale, 2);
    const float d = float(h);
    for (unsigned j = 0; j < kv8_group; ++j) {
        float c = d == 0.0F ? 0.0F : std::nearbyint(x[j] / d);
        c = std::min(127.0F, std::max(-127.0F, c));
        codes[j] = std::int8_t(c);
    }
}
void mirror_tests(MirrorFormat format) {
    Stream s(4096, 3001, 3009, 11);
    Buffer failure(4);
    failure.zero();
    fp16_mirror_sync(s.impl, format, failure.as<unsigned>(), nullptr);
    synchronize();
    CHECK(flag_of(failure) == 0);
    fp16_mirror_confirm(s.impl);
    const auto m = fp16_mirror_read(s.impl, 0);
    CHECK(m.mirrored == 2944 && m.width == width && m.format == unsigned(format));
    const auto n = std::size_t(m.mirrored) * width;
    if (format == MirrorFormat::f16) {
        std::vector<std::uint16_t> hk(n), hv(n);
        CHECK(hipMemcpy(hk.data(), m.k, n * 2, hipMemcpyDeviceToHost) == hipSuccess);
        CHECK(hipMemcpy(hv.data(), m.v, n * 2, hipMemcpyDeviceToHost) == hipSuccess);
        for (std::size_t i = 0; i < n; ++i) {
            CHECK(hk[i] == half_bits(s.keys[i]));
            CHECK(hv[i] == half_bits(s.values[i]));
        }
    } else {
        std::vector<std::int8_t> ck(n), cv(n), ek(kv8_group);
        std::vector<std::uint16_t> sk(n / kv8_group), sv(sk.size());
        CHECK(hipMemcpy(ck.data(), m.k, n, hipMemcpyDeviceToHost) == hipSuccess);
        CHECK(hipMemcpy(cv.data(), m.v, n, hipMemcpyDeviceToHost) == hipSuccess);
        CHECK(hipMemcpy(sk.data(), m.k_scale, sk.size() * 2, hipMemcpyDeviceToHost) == hipSuccess);
        CHECK(hipMemcpy(sv.data(), m.v_scale, sv.size() * 2, hipMemcpyDeviceToHost) == hipSuccess);
        for (std::size_t g = 0; g < n / kv8_group; ++g) {
            std::uint16_t scale = 0;
            kv8_host(s.keys.data() + g * kv8_group, ek.data(), scale);
            CHECK(scale == sk[g] && std::memcmp(ek.data(), ck.data() + g * kv8_group, kv8_group) == 0);
            kv8_host(s.values.data() + g * kv8_group, ek.data(), scale);
            CHECK(scale == sv[g] && std::memcmp(ek.data(), cv.data() + g * kv8_group, kv8_group) == 0);
        }
    }
    // Idempotent below the next step; the format is fixed per stream.
    fp16_mirror_sync(s.impl, format, failure.as<unsigned>(), nullptr);
    CHECK(fp16_mirror_read(s.impl, 0).mirrored == 2944);
    const auto other = format == MirrorFormat::f16 ? MirrorFormat::kv8 : MirrorFormat::f16;
    refuses([&] { fp16_mirror_sync(s.impl, other, failure.as<unsigned>(), nullptr); });
    // A position tag that does not match its row: bit 2.
    Stream bad(512, 200, 200, 5);
    std::uint64_t wrong = 7;
    CHECK(hipMemcpy(bad.impl.cache[0].tags.as<std::uint64_t>() + 70, &wrong, 8, hipMemcpyHostToDevice) == hipSuccess);
    failure.zero();
    fp16_mirror_sync(bad.impl, format, failure.as<unsigned>(), nullptr);
    synchronize();
    CHECK(flag_of(failure) & 2U);
    // Not confirmed: the next sync converts (and checks) the same rows again.
    failure.zero();
    fp16_mirror_sync(bad.impl, format, failure.as<unsigned>(), nullptr);
    synchronize();
    CHECK((flag_of(failure) & 2U) && bad.impl.fp16_mirror->layer[0].mirrored == 0);
}
bool same_bytes(const float* a, const float* b, std::size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }
bool same_bytes(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && same_bytes(a.data(), b.data(), a.size());
}
double max_error(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(double(a[i]) - double(b[i])));
    return e;
}
void kernel_tests(MirrorFormat format, double tolerance) {
    Fixture f(4096);
    f.format = format;
    // One stream: committed 3001 rows, overlay rows 3001..3004 (an MTP-like verify pass of four rows).
    Stream s(4096, 3001, 3005, 21);
    Buffer failure(4);
    failure.zero();
    const auto none = Fp16MirrorRead{};
    fp16_mirror_sync(s.impl, format, failure.as<unsigned>(), nullptr);
    synchronize();
    const auto synced = fp16_mirror_read(s.impl, 0);
    auto partial = synced;
    partial.mirrored = 128;
    std::vector<std::vector<float>> per_row(4);
    for (unsigned r = 0; r < 4; ++r) {
        const auto p = 3001U + r;
        std::vector<std::vector<float>> got;
        for (const auto& m : {none, partial, synced}) {
            f.stage({p}, {s.read(p, 4)}, {m}, {r});
            got.push_back(f.fast(1, bucket(p)));
        }
        CHECK(same_bytes(got[0], got[1]) && same_bytes(got[1], got[2]));
        // Run to run.
        CHECK(same_bytes(f.fast(1, bucket(p)), got[2]));
        // A larger attention stride bucket (another row of the pass further on) changes nothing.
        CHECK(same_bytes(f.fast(1, bucket(p) + 512), got[2]));
        per_row[r] = got[2];
        const auto exact = f.exact(1, bucket(p));
        const auto error = max_error(exact, got[2]);
        std::cout << fast_decode_attention_name(format) << " position " << p << " max |fast - exact| " << error << '\n';
        CHECK(error < tolerance);
    }
    // R = 4 in one pass == each row alone.
    f.stage({3001, 3002, 3003, 3004}, {s.read(3001, 4), s.read(3002, 4), s.read(3003, 4), s.read(3004, 4)},
            {synced, synced, synced, synced}, {0, 1, 2, 3});
    const auto four = f.fast(4, bucket(3004));
    for (unsigned r = 0; r < 4; ++r)
        CHECK(same_bytes(per_row[r].data(), four.data() + std::size_t(r) * hcount * dim, per_row[r].size()));
    // g4-a8-v1: the merge writes the A8G32 rows of its output; bytes == quantizing the FP32 output separately.
    {
        const unsigned width_q = hcount * dim;
        Buffer codes(4ULL * width_q), meta(4ULL * (width_q / 32U) * sizeof(a8::Meta)), ref_codes(codes.size()),
            ref_meta(meta.size());
        const A8Rows fused{codes.as<std::int8_t>(), meta.as<a8::Meta>(), width_q};
        const auto with_quant = f.fast(4, bucket(3004), nullptr, fused);
        CHECK(same_bytes(with_quant, four));
        a8_quantize(f.out.as<float>(), 4, width_q, {ref_codes.as<std::int8_t>(), ref_meta.as<a8::Meta>(), width_q});
        synchronize();
        std::vector<std::uint8_t> a(codes.size()), b(codes.size()), am(meta.size()), bm(meta.size());
        codes.download(a.data(), a.size());
        ref_codes.download(b.data(), b.size());
        meta.download(am.data(), am.size());
        ref_meta.download(bm.data(), bm.size());
        CHECK(a == b && am == bm);
    }
    // Mixed streams in one pass (C4-like): a short stream beside the long one, each row == alone.
    Stream t(4096, 70, 71, 31);
    fp16_mirror_sync(t.impl, format, failure.as<unsigned>(), nullptr);
    synchronize();
    const auto tm = fp16_mirror_read(t.impl, 0);
    f.stage({70}, {t.read(70, 1)}, {tm}, {1});
    const auto short_alone = f.fast(1, bucket(70));
    f.stage({3001, 70}, {s.read(3001, 4), t.read(70, 1)}, {synced, tm}, {0, 1});
    const auto mixed = f.fast(2, bucket(3001));
    CHECK(same_bytes(per_row[0].data(), mixed.data(), per_row[0].size()));
    CHECK(same_bytes(short_alone.data(), mixed.data() + std::size_t(hcount) * dim, short_alone.size()));
    // Position 0 (one key) and a tile edge.
    for (const unsigned p : {0U, 127U, 128U, 255U}) {
        Stream e(1024, p, p + 1, 41 + p);
        f.stage({p}, {e.read(p, 1)}, {none}, {2});
        const auto got = f.fast(1, bucket(p));
        const auto exact = f.exact(1, bucket(p));
        CHECK(max_error(exact, got) < tolerance);
    }
    // A NaN in an unmirrored committed key row (read from FP32), and in the mirror sync of the same row: never silently accepted.
    {
        Stream n(1024, 300, 301, 61);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        CHECK(hipMemcpy(n.impl.cache[0].k.as<float>() + std::size_t(200) * width + 3, &nan, 4, hipMemcpyHostToDevice) ==
              hipSuccess);
        f.stage({300}, {n.read(300, 1)}, {Fp16MirrorRead{}}, {0});
        unsigned bad = 0;
        (void)f.fast(1, bucket(300), &bad);
        CHECK(bad & 1U);
        Buffer sync_failure(4);
        sync_failure.zero();
        fp16_mirror_sync(n.impl, format, sync_failure.as<unsigned>(), nullptr);
        synchronize();
        CHECK(flag_of(sync_failure) != 0);
    }
    // A poisoned tag of an unmirrored committed row: bit 2.
    std::uint64_t wrong = 1;
    CHECK(hipMemcpy(s.impl.cache[0].tags.as<std::uint64_t>() + 2990, &wrong, 8, hipMemcpyHostToDevice) == hipSuccess);
    f.stage({3001}, {s.read(3001, 4)}, {synced}, {0});
    unsigned flag = 0;
    (void)f.fast(1, bucket(3001), &flag);
    CHECK(flag & 2U);
}
void bench() {
    Fixture f(65536);
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a) == hipSuccess && hipEventCreate(&b) == hipSuccess);
    for (const unsigned n : {8192U, 32768U, 65408U}) {
        Stream s(65536, n, n + 1, 3);
        Buffer failure(4);
        f.ensure(bucket(n));
        for (const auto form : {MirrorFormat::none, MirrorFormat::f16, MirrorFormat::kv8}) {
            f.format = form;
            if (form != MirrorFormat::none) {
                s.impl.fp16_mirror.reset();
                failure.zero();
                fp16_mirror_sync(s.impl, form, failure.as<unsigned>(), nullptr);
                synchronize();
                CHECK(flag_of(failure) == 0);
            }
            f.stage({n}, {s.read(n, 1)}, {fp16_mirror_read(s.impl, 0)}, {0});
            for (unsigned i = 0; i < 3; ++i)
                form != MirrorFormat::none ? (void)f.fast(1, bucket(n)) : (void)f.exact(1, bucket(n));
            constexpr unsigned reps = 20;
            CHECK(hipEventRecord(a, nullptr) == hipSuccess);
            for (unsigned i = 0; i < reps; ++i) {
                if (form != MirrorFormat::none)
                    fast_decode_attention(f.q_d.as<float>(), f.q_norm, f.positions_d.as<std::uint64_t>(),
                                          f.table.as<float>(), f.reads_d.as<KvRead>(), f.mirrors_d.as<Fp16MirrorRead>(),
                                          1, hcount, dim, bucket(n), f.out.as<float>(), f.scratch.as<float>(),
                                          f.scratch.size(), f.failure.as<unsigned>(), nullptr, true, true, true, form);
                else
                    fused_attention(f.q_d.as<float>(), f.q_norm, f.positions_d.as<std::uint64_t>(), f.table.as<float>(),
                                    f.reads_d.as<KvRead>(), 1, hcount, dim, bucket(n), true, f.out.as<float>(),
                                    f.failure.as<unsigned>(), nullptr, true, true, true, f.scores.as<float>(),
                                    f.maxima.as<float>(), f.partials.as<float>(), f.counters.as<unsigned>());
            }
            CHECK(hipEventRecord(b, nullptr) == hipSuccess);
            CHECK(hipEventSynchronize(b) == hipSuccess);
            float ms = 0;
            CHECK(hipEventElapsedTime(&ms, a, b) == hipSuccess);
            const double per = ms / reps, fp32 = double(n) * width * 4 * 2;
            const double bytes = form == MirrorFormat::none ? fp32 : form == MirrorFormat::f16 ? fp32 / 2 : fp32 * 9 / 32;
            std::cout << "bench position " << n << ' ' << fast_decode_attention_name(form) << ' ' << per * 1000.0
                      << " us/layer, KV " << bytes / (per * 1e6) << " GB/s" << std::endl;
        }
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        require_device();
        if (argc == 2 && std::string_view(argv[1]) == "--bench") {
            bench();
            return 0;
        }
        mirror_tests(MirrorFormat::f16);
        mirror_tests(MirrorFormat::kv8);
        kernel_tests(MirrorFormat::f16, 1e-2);
        kernel_tests(MirrorFormat::kv8, 6e-2);
        std::cout << "g4-dattn-f1/f2: mirror halves / KV8 groups, sync-state / row / run invariance, exact closeness, poison passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
