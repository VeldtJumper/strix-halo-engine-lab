// GPU tests and bench of the H3 attention kernel (gfx1151).
//   strix-h3-attn-gpu-tests            correctness: G1 vs the fp64 host reference (sizes with partial tiles),
//                                      bitwise: block launch order, row split across "nodes", fused == split(1), reruns
//   strix-h3-attn-gpu-tests --bench    timing at S = 19 K x 2 blocks and 38 K x 2 blocks (+ 38 K fused)
#include "h3/attn/attn.hpp"
#include "h3/attn/attn_ref.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace strix::h3::attn;

namespace {
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
template <class T> T* dalloc(std::size_t n) { void* p = nullptr; CK(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T))); return static_cast<T*>(p); }
unsigned H = 56;                 // heads of the current case (56 except the long-sequence case)
unsigned* g_bound = nullptr;     // device counter given to every partial launch (input-bound telemetry)
std::uint64_t round8(std::uint64_t x) { return (x + 7U) / 8U * 8U; }

struct DeviceQkv {
    std::uint16_t *q = nullptr, *k = nullptr, *vt = nullptr;
    std::uint64_t q_hs = 0, q_rs = 0, k_hs = 0, k_rs = 0, vt_hs = 0, vt_ds = 0;
    unsigned q_rows = 0, k_rows = 0;
    void release() const { CK(hipFree(q)); CK(hipFree(k)); CK(hipFree(vt)); }
    Queries queries(unsigned r0, unsigned rows) const { return Queries{q + r0 * q_rs, q_hs, q_rs, rows, H}; }
    KeyBlock block(unsigned j0, unsigned j1) const {
        return KeyBlock{k + j0 * k_rs, vt + j0, k_hs, k_rs, vt_hs, vt_ds, j1 - j0};
    }
};
// Upload: Q and K head-major or token-major, V through the transpose kernel into V^T [h][128][round8(keys)].
DeviceQkv upload(const HostQkv& x, bool token_major, int pad_byte = 0xff) {
    DeviceQkv d;
    d.q_rows = x.q_rows; d.k_rows = x.k_rows;
    const auto put = [&](const std::vector<std::uint16_t>& src, unsigned rows, std::uint64_t& hs, std::uint64_t& rs) {
        std::vector<std::uint16_t> lay(src.size());
        hs = token_major ? kHeadDim : std::uint64_t(rows) * kHeadDim;
        rs = token_major ? std::uint64_t(H) * kHeadDim : kHeadDim;
        for (unsigned h = 0; h < H; ++h)
            for (unsigned r = 0; r < rows; ++r)
                std::memcpy(&lay[h * hs + r * rs], &src[(std::size_t(h) * rows + r) * kHeadDim], kHeadDim * 2);
        auto* p = dalloc<std::uint16_t>(lay.size());
        CK(hipMemcpy(p, lay.data(), lay.size() * 2, hipMemcpyHostToDevice));
        return p;
    };
    d.q = put(x.q, x.q_rows, d.q_hs, d.q_rs);
    d.k = put(x.k, x.k_rows, d.k_hs, d.k_rs);
    std::uint64_t v_hs, v_rs;
    auto* v = put(x.v, x.k_rows, v_hs, v_rs);
    d.vt_ds = round8(x.k_rows);
    d.vt_hs = kHeadDim * d.vt_ds;
    d.vt = dalloc<std::uint16_t>(H * d.vt_hs);
    CK(hipMemset(d.vt, pad_byte, H * d.vt_hs * 2));  // pad (default NaN): the kernel must stage padded keys as zeros
    launch_transpose_v(v, v_hs, v_rs, x.k_rows, H, d.vt, d.vt_hs, d.vt_ds, nullptr);
    CK(hipDeviceSynchronize());
    CK(hipFree(v));
    return d;
}

struct Result {
    std::vector<std::uint16_t> bf;
    std::vector<float> f32;
    std::vector<float> po, pml;  // partials [slot][h][rows][128], [slot][h][rows][2] (split runs)
    unsigned rows = 0, nb = 0;
};
// Partials of the cut blocks (one launch per call of `order`), merge; rows [r0, r0 + rows) of the queries.
Result run_split(const DeviceQkv& d, const std::vector<unsigned>& cuts, unsigned r0, unsigned rows, bool one_launch, bool reverse) {
    const unsigned nb = static_cast<unsigned>(cuts.size() - 1);
    Partials p{dalloc<float>(std::size_t(nb) * H * rows * kHeadDim), dalloc<float>(std::size_t(nb) * H * rows * 2), rows, H, kHeadDim, g_bound};
    std::vector<KeyBlock> blocks;
    for (unsigned b = 0; b < nb; ++b) blocks.push_back(d.block(cuts[b], cuts[b + 1]));
    const Queries q = d.queries(r0, rows);
    if (one_launch) launch_partials(q, blocks.data(), nb, p, 0, nullptr);
    else
        for (unsigned i = 0; i < nb; ++i) {
            const unsigned b = reverse ? nb - 1 - i : i;
            launch_partials(q, &blocks[b], 1, p, b, nullptr);
        }
    Output out{dalloc<std::uint16_t>(std::size_t(rows) * H * kHeadDim), dalloc<float>(std::size_t(rows) * H * kHeadDim), std::uint64_t(H) * kHeadDim};
    launch_merge(p, nb, out, nullptr);
    CK(hipDeviceSynchronize());
    Result r;
    r.bf.resize(std::size_t(rows) * H * kHeadDim);
    r.f32.resize(r.bf.size());
    CK(hipMemcpy(r.bf.data(), out.out, r.bf.size() * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(r.f32.data(), out.out_f32, r.f32.size() * 4, hipMemcpyDeviceToHost));
    r.rows = rows; r.nb = nb;
    r.po.resize(std::size_t(nb) * H * rows * kHeadDim);
    r.pml.resize(std::size_t(nb) * H * rows * 2);
    CK(hipMemcpy(r.po.data(), p.o, r.po.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(r.pml.data(), p.ml, r.pml.size() * 4, hipMemcpyDeviceToHost));
    CK(hipFree(p.o)); CK(hipFree(p.ml)); CK(hipFree(out.out)); CK(hipFree(out.out_f32));
    return r;
}
Result run_fused(const DeviceQkv& d, unsigned* bound_rows = nullptr) {
    const unsigned rows = d.q_rows;
    Output out{dalloc<std::uint16_t>(std::size_t(rows) * H * kHeadDim), dalloc<float>(std::size_t(rows) * H * kHeadDim), std::uint64_t(H) * kHeadDim,
               bound_rows};
    launch_fused_single(d.queries(0, rows), d.block(0, d.k_rows), out, nullptr);
    CK(hipDeviceSynchronize());
    Result r;
    r.bf.resize(std::size_t(rows) * H * kHeadDim);
    r.f32.resize(r.bf.size());
    CK(hipMemcpy(r.bf.data(), out.out, r.bf.size() * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(r.f32.data(), out.out_f32, r.f32.size() * 4, hipMemcpyDeviceToHost));
    CK(hipFree(out.out)); CK(hipFree(out.out_f32));
    return r;
}
bool same(const Result& a, const Result& b, std::size_t off = 0) {
    return std::memcmp(a.bf.data() + off, b.bf.data(), b.bf.size() * 2) == 0 &&
           std::memcmp(a.f32.data() + off, b.f32.data(), b.f32.size() * 4) == 0;
}

// Full partial bytes (mc, l, o) of every (slot, head, row) of `part` (rows r0 ..) equal those of `whole`.
bool same_partials(const Result& whole, const Result& part, unsigned r0) {
    for (unsigned b = 0; b < whole.nb; ++b)
        for (unsigned h = 0; h < H; ++h) {
            const std::size_t w = (std::size_t(b) * H + h) * whole.rows + r0, q = (std::size_t(b) * H + h) * part.rows;
            if (std::memcmp(&whole.po[w * kHeadDim], &part.po[q * kHeadDim], std::size_t(part.rows) * kHeadDim * 4) ||
                std::memcmp(&whole.pml[w * 2], &part.pml[q * 2], std::size_t(part.rows) * 2 * 4))
                return false;
        }
    return true;
}

// G1 of sampled rows (all heads) against fp64, and the fp32 kernel-order emulation on a subset.
void gate(const HostQkv& x, const std::vector<unsigned>& cuts, const Result& r, unsigned row_step, const std::string& name) {
    std::vector<unsigned> rows;
    for (unsigned i = 0; i < x.q_rows; i += row_step) rows.push_back(i);
    if (rows.back() != x.q_rows - 1) rows.push_back(x.q_rows - 1);
    std::atomic<std::size_t> next{0};
    std::vector<double> e32(rows.size() * H), ebf(rows.size() * H), eem(rows.size() * H, 0.0);
    const auto work = [&] {
        for (std::size_t i; (i = next++) < rows.size() * H;) {
            const unsigned row = rows[i / H], h = static_cast<unsigned>(i % H);
            double ref[kHeadDim];
            reference_row_f64(x, h, row, ref);
            const std::size_t o = (std::size_t(row) * H + h) * kHeadDim;
            e32[i] = row_error(&r.f32[o], ref);
            float bf[kHeadDim];
            for (unsigned dd = 0; dd < kHeadDim; ++dd) bf[dd] = bf16_to_float(r.bf[o + dd]);
            ebf[i] = row_error(bf, ref);
            if ((i / H) % 4 == 0 && h % 7 == 0) {  // kernel-order fp32 emulation
                std::vector<Partial> p;
                for (std::size_t b = 0; b + 1 < cuts.size(); ++b) p.push_back(partial_row_f32(x, h, row, cuts[b], cuts[b + 1]));
                float em[kHeadDim];
                merge_f32(p.data(), static_cast<unsigned>(p.size()), em);
                double emd[kHeadDim];
                for (unsigned dd = 0; dd < kHeadDim; ++dd) emd[dd] = em[dd];
                eem[i] = row_error(&r.f32[o], emd);
            }
        }
    };
    std::vector<std::thread> th;
    const unsigned nt = std::max(1U, std::min(24U, std::thread::hardware_concurrency()));
    for (unsigned t = 0; t < nt; ++t) th.emplace_back(work);
    for (auto& t : th) t.join();
    const double m32 = *std::max_element(e32.begin(), e32.end()), mbf = *std::max_element(ebf.begin(), ebf.end());
    const double mem = *std::max_element(eem.begin(), eem.end());
    double mean = 0.0;
    for (double e : e32) mean += e;
    mean /= static_cast<double>(e32.size());
    char buf[320];
    // G1 is on the merged fp32 value (limit 2^-8 of the row max); the bf16 output adds its own rounding (unit
    // roundoff 2^-8), so it is held to 2^-7.
    std::snprintf(buf, sizeof buf, "G1 %s: %zu rows x %u heads, max err fp32 out %.3e (mean %.2e, limit 2^-8 = 3.906e-03), bf16 out %.3e (limit 2^-7); vs fp32 emulation %.2e",
                  name.c_str(), rows.size(), H, m32, mean, mbf, mem);
    check(m32 <= 1.0 / 256.0 && mbf <= 1.0 / 128.0, buf);
}

void schedule(unsigned w, int early, int persistent, int pp = 0, int tn = 0) {
    set_waves_per_wg(w);
    set_early_loads(early);
    set_persistent(persistent);
    set_pingpong(pp);
    set_tune(tn);
}
void schedule_default() { schedule(16, 1, 0); }  // the library default (W16 early)
constexpr unsigned kSchedules = 19;
const char* schedule_name(unsigned v) {
    static const char* names[kSchedules] = {"W8 late", "W16 late", "W8 early", "W16 early", "W8 late persistent", "W16 late persistent",
                                            "W8 early persistent", "W16 early persistent", "W8 pingpong", "W16 pingpong",
                                            "W8 pingpong persistent", "W16 pingpong persistent",
                                            "W8 early tune1 (permlane P^T)", "W8 early tune2 (interleave)", "W8 early tune3",
                                            "W8 early tune4 (swizzled single buffer)", "W16 early tune2 (interleave)",
                                            "W16 early tune3", "W16 early tune4"};
    return names[v % kSchedules];
}
// 0..7: waves x load timing x persistent; 8..11: ping-pong x waves x persistent
void schedule_variant(unsigned v) {
    if (v < 8U) schedule(v & 1U ? 16U : 8U, v & 2U ? 1 : 0, v & 4U ? 1 : 0, 0);
    else if (v < 12U) schedule(v & 1U ? 16U : 8U, 1, v & 2U ? 1 : 0, 1);
    else if (v < 16U) schedule(8U, 1, 0, 0, static_cast<int>(v - 11U));  // W8 tune 1..4
    else schedule(16U, 1, 0, 0, static_cast<int>(v - 14U));              // W16 tune 2..4
}

void correctness_case_x(const std::string& name, const HostQkv& x, const std::vector<unsigned>& cuts, bool token_major, unsigned row_step) {
    const unsigned sq = x.q_rows, sk = x.k_rows;
    schedule_default();
    const auto d = upload(x, token_major);
    const Result one = run_split(d, cuts, 0, sq, true, false);
    gate(x, cuts, one, row_step, name);
    const unsigned nb = static_cast<unsigned>(cuts.size() - 1);
    if (nb > 1) {
        const Result rev = run_split(d, cuts, 0, sq, false, true);
        check(same(one, rev) && same_partials(one, rev, 0), name + ": per-block launches in reverse order == one launch (outputs + partial bytes)");
    }
    const unsigned s0 = sq / 2 + 13;  // the row split of two nodes (arbitrary, not tile aligned)
    const Result a = run_split(d, cuts, 0, s0, false, false), b = run_split(d, cuts, s0, sq - s0, false, true);
    check(same(one, a, 0) && same(one, b, std::size_t(s0) * H * kHeadDim), name + ": rows split [0," + std::to_string(s0) + ") + rest == 1-node (bitwise, G3 form)");
    check(same_partials(one, a, 0) && same_partials(one, b, s0), name + ": row-split partials (mc, l, o) bytes == 1-node");
    const Result again = run_split(d, cuts, 0, sq, true, false);
    check(same(one, again) && same_partials(one, again, 0), name + ": rerun (outputs + partial bytes)");
    {   // every advertised schedule (waves x load timing x persistent), one launch and per-block reverse launches
        bool all = true;
        std::string bad;
        for (unsigned v = 0; v < kSchedules; ++v) {
            schedule_variant(v);
            const Result r1 = run_split(d, cuts, 0, sq, true, false);
            const Result r2 = run_split(d, cuts, 0, sq, false, true);
            const bool ok = same(one, r1) && same_partials(one, r1, 0) && same(one, r2) && same_partials(one, r2, 0);
            if (!ok) bad += std::string(" ") + schedule_name(v);
            all &= ok;
        }
        schedule_default();
        set_head_split(1);  // one launch per (block, head), default schedule and W16 early persistent
        const Result hs1 = run_split(d, cuts, 0, sq, true, false);
        schedule_variant(7);
        const Result hs2 = run_split(d, cuts, 0, sq, true, false);
        schedule_default();
        set_head_split(0);
        const Result hs0 = run_split(d, cuts, 0, sq, true, false);
        set_head_split(-1);
        const bool hs_ok = same(one, hs1) && same_partials(one, hs1, 0) && same(one, hs2) && same_partials(one, hs2, 0) &&
                           same(one, hs0) && same_partials(one, hs0, 0);
        if (!hs_ok) bad += " head-split";
        all &= hs_ok;
        check(all, name + ": all 19 schedules (W8/W16 x late/early/pingpong x grid/persistent + 3 tuned) == default (outputs + partial bytes)" + bad);
    }
    bool cuts64 = true;
    for (std::size_t b = 1; b + 1 < cuts.size(); ++b) cuts64 &= cuts[b] % 64U == 0U;
    if (token_major && cuts64) {  // the one-call op (interior cuts multiples of 64) == the explicit launches
        const std::size_t n = std::size_t(sq) * H * kHeadDim;
        float* scratch = dalloc<float>(attention_scratch_floats(sq, H, kHeadDim, nb));
        std::uint16_t* out = dalloc<std::uint16_t>(n);
        AttentionCall c;
        c.q = d.q; c.k = d.k; c.vt = d.vt; c.rows = sq; c.heads = H; c.keys = sk; c.vt_pitch = d.vt_ds;
        c.cuts = cuts.data(); c.nblocks = nb; c.scratch = scratch; c.out = out;
        attention(c, nullptr);
        CK(hipDeviceSynchronize());
        std::vector<std::uint16_t> got(n);
        CK(hipMemcpy(got.data(), out, n * 2, hipMemcpyDeviceToHost));
        check(std::memcmp(got.data(), one.bf.data(), n * 2) == 0, name + ": attention() one-call op == launches (bitwise)");
        CK(hipFree(scratch)); CK(hipFree(out));
    }
    if (nb == 1) {
        bool all = true;
        for (unsigned v = 0; v < kSchedules; ++v) {
            schedule_variant(v);
            all &= same(one, run_fused(d));
        }
        schedule_default();
        set_head_split(1);
        all &= same(one, run_fused(d));
        set_head_split(-1);
        check(all, name + ": fused single block == partial + merge, all 19 schedules + head split (bitwise)");
    }
    d.release();
}
void correctness_case(const std::string& name, unsigned sq, unsigned sk, const std::vector<unsigned>& cuts, float q_scale, bool token_major,
                      unsigned row_step, std::uint64_t seed) {
    correctness_case_x(name, make_qkv(H, sq, sk, seed, q_scale), cuts, token_major, row_step);
}
unsigned read_bound() {
    unsigned v = 0;
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(&v, g_bound, 4, hipMemcpyDeviceToHost));
    CK(hipMemset(g_bound, 0, 4));
    return v;
}
template <class F> bool throws_invalid(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}

// ---- bench ----
__attribute__((noinline)) void fill_random(std::vector<std::uint16_t>& v, std::uint64_t seed) {
    std::uint64_t s = seed * 0x9e3779b97f4a7c15ULL + 1;
    for (auto& e : v) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        // sum of two uniforms in [-1, 1): roughly unit-variance-ish, bf16 by truncation (bench data only)
        const float a = static_cast<float>(static_cast<std::int32_t>(s >> 32)) * (1.0F / 2147483648.0F);
        const float b = static_cast<float>(static_cast<std::int32_t>(s & 0xffffffffU)) * (1.0F / 2147483648.0F);
        const float f = (a + b) * 1.2F;
        std::uint32_t u; std::memcpy(&u, &f, 4);
        e = static_cast<std::uint16_t>(u >> 16);
    }
}
// mode: "all" (the shapes + ablations), "sweep" (18955 queries x one block of 1 K .. 37.9 K keys), or one launch for
// a profiler: "p2" (2-node shape, block 0), "pf" (fused 37910 x 37910).
void bench(const std::string& mode) {
    bool mode_pp = false;
    const unsigned s_full = 37910, s_half = 18955;
    const unsigned cut = 18944;  // block boundary: a multiple of 64 (block 0 = rows [0, 18944), block 1 = the rest)
    std::vector<std::uint16_t> host(std::size_t(s_full) * H * kHeadDim);
    DeviceQkv d;
    d.q_rows = d.k_rows = s_full;
    d.q_hs = d.k_hs = kHeadDim; d.q_rs = d.k_rs = std::uint64_t(H) * kHeadDim;  // token-major [S][56][128]
    d.vt_ds = round8(s_full); d.vt_hs = kHeadDim * d.vt_ds;
    d.q = dalloc<std::uint16_t>(host.size()); d.k = dalloc<std::uint16_t>(host.size()); d.vt = dalloc<std::uint16_t>(H * d.vt_hs);
    fill_random(host, 1); CK(hipMemcpy(d.q, host.data(), host.size() * 2, hipMemcpyHostToDevice));
    fill_random(host, 2); CK(hipMemcpy(d.k, host.data(), host.size() * 2, hipMemcpyHostToDevice));
    {
        std::vector<std::uint16_t> vt(H * d.vt_hs);
        fill_random(vt, 3);
        CK(hipMemcpy(d.vt, vt.data(), vt.size() * 2, hipMemcpyHostToDevice));
    }
    Partials p{dalloc<float>(2ULL * H * s_full * kHeadDim), dalloc<float>(2ULL * H * s_full * 2), s_full, H};
    Output out{dalloc<std::uint16_t>(std::size_t(s_full) * H * kHeadDim), nullptr, std::uint64_t(H) * kHeadDim};
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const auto timed = [&](const char* name, double flops, auto&& fn) {
        fn(); CK(hipDeviceSynchronize());
        std::vector<float> ms;
        for (int r = 0; r < 5; ++r) {
            CK(hipEventRecord(e0)); fn(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float t; CK(hipEventElapsedTime(&t, e0, e1)); ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("BENCH %-58s median %9.3f ms (min %9.3f)  %6.2f TFLOP/s\n", name, ms[2], ms[0], flops > 0 ? flops / (ms[2] * 1e9) : 0.0);
        std::fflush(stdout);
    };
    const auto fl = [](double q, double k) { return 4.0 * q * k * kHeadDim * H; };
    const KeyBlock b0 = d.block(0, cut), b1 = d.block(cut, s_full);
    const KeyBlock both[2] = {b0, b1};
    if (mode == "sweep") {
        const Queries qh = d.queries(0, s_half);
        Partials ph{p.o, p.ml, s_half, H};
        for (unsigned v = 0; v < 3; ++v) {
            const unsigned w = v == 1U ? 16U : 8U;
            set_waves_per_wg(w);
            set_persistent(v == 2U ? 1 : 0);
            for (unsigned keys : {1024U, 2048U, 4096U, 9472U, 18944U, 37888U}) {
                const KeyBlock kb = d.block(0, keys);
                char name[112];
                std::snprintf(name, sizeof name, "sweep W%u%s: 18955 q x 1 block of %u keys (K/V/head %.1f MB)", w, v == 2U ? " persist" : "", keys, keys * 512.0 / 1e6);
                timed(name, fl(s_half, keys), [&] { launch_partials(qh, &kb, 1, ph, 0, nullptr); });
            }
        }
        set_waves_per_wg(8);
        set_persistent(0);
        d.release();
        return;
    }
    if (mode == "p2pp") { set_pingpong(1); mode_pp = true; }
    if (mode == "p2t3") { set_tune(3); mode_pp = true; }
    if (mode.rfind("keys", 0) == 0) {  // "keys<N>": one launch of 18955 q x 1 block of N keys (counters below saturation)
        const unsigned nk = static_cast<unsigned>(std::atoi(mode.c_str() + 4));
        const KeyBlock kb = d.block(0, nk);
        Partials ph{p.o, p.ml, s_half, H};
        launch_partials(d.queries(0, s_half), &kb, 1, ph, 0, nullptr);
        CK(hipDeviceSynchronize());
        d.release();
        return;
    }
    if (mode == "fusedmall") {  // fused 37910^2: all heads in one launch vs one head per launch (no head transitions)
        const KeyBlock whole = d.block(0, s_full);
        const Queries qf = d.queries(0, s_full);
        set_head_split(0);
        timed("fused 37910^2, 56 heads in one launch (head split off)", fl(s_full, s_full), [&] { launch_fused_single(qf, whole, out, nullptr); });
        set_head_split(-1);
        timed("fused 37910^2, automatic (head split on above 24576 keys)", fl(s_full, s_full), [&] { launch_fused_single(qf, whole, out, nullptr); });
        {
            const KeyBlock big[2] = {d.block(0, 18944), d.block(18944, s_full)};
            timed("1-node 37910 x 2 blocks, automatic (no split: blocks <= 24576 keys)", fl(s_full, s_full), [&] { launch_partials(qf, big, 2, p, 0, nullptr); });
            set_head_split(1);
            timed("1-node 37910 x 2 blocks, head split forced", fl(s_full, s_full), [&] { launch_partials(qf, big, 2, p, 0, nullptr); });
            set_head_split(-1);
        }
        timed("fused 37910^2, 56 launches of 1 head", fl(s_full, s_full), [&] {
            for (unsigned h = 0; h < H; ++h) {
                Queries q1 = qf;
                q1.q += h * qf.q_hs;
                q1.heads = 1;
                KeyBlock b1 = whole;
                b1.k += h * whole.k_hs;
                b1.vt += h * whole.vt_hs;
                launch_fused_single(q1, b1, Output{out.out + h * kHeadDim, nullptr, out.out_rs}, nullptr);
            }
        });
        const KeyBlock both2[2] = {d.block(0, cut), d.block(cut, s_full)};
        timed("1-node 37910 x 2 blocks, one launch", fl(s_full, s_full), [&] { launch_partials(qf, both2, 2, p, 0, nullptr); });
        timed("1-node 37910 x 2 blocks, 56 x 2 launches (head h: block 0 then block 1)", fl(s_full, s_full), [&] {
            for (unsigned h = 0; h < H; ++h)
                for (unsigned b = 0; b < 2; ++b) {
                    Queries q1 = qf;
                    q1.q += h * qf.q_hs;
                    q1.heads = 1;
                    KeyBlock b1 = both2[b];
                    b1.k += h * b1.k_hs;
                    b1.vt += h * b1.vt_hs;
                    Partials p1{p.o + (std::size_t(b) * H + h) * s_full * kHeadDim, p.ml + (std::size_t(b) * H + h) * s_full * 2, s_full, 1};
                    launch_partials(q1, &b1, 1, p1, 0, nullptr);
                }
        });
        d.release();
        return;
    }
    if (mode == "p2s" || mode == "p2spp") {  // short launch (18955 q x 1024 keys) for counter-range checks
        if (mode == "p2spp") set_pingpong(1);
        const KeyBlock shortb = d.block(0, 1024);
        Partials ph{p.o, p.ml, s_half, H};
        launch_partials(d.queries(0, s_half), &shortb, 1, ph, 0, nullptr);
        CK(hipDeviceSynchronize());
        d.release();
        return;
    }
    if (mode == "p2" || mode == "pf" || mode_pp) {
        const KeyBlock first = d.block(0, cut), whole = d.block(0, s_full);
        Partials ph{p.o, p.ml, s_half, H};
        for (int r = 0; r < 2; ++r) {
            if (mode == "p2" || mode_pp) launch_partials(d.queries(0, s_half), &first, 1, ph, 0, nullptr);
            else launch_fused_single(d.queries(0, s_full), whole, out, nullptr);
        }
        CK(hipDeviceSynchronize());
        d.release();
        return;
    }
    if (mode == "sustain") {  // >= 60 s continuous at the 2-node per-block shape; TF per ~5 s window with clock + temp
        const auto probe = [](const char* cmd) {
            FILE* f = popen(cmd, "r");
            char buf[128] = {};
            if (f) { if (!std::fgets(buf, sizeof buf, f)) buf[0] = 0; pclose(f); }
            for (char* c = buf; *c; ++c) if (*c == '\n') *c = 0;
            return std::string(buf[0] ? buf : "n/a");
        };
        const Queries qh = d.queries(0, s_half);
        Partials ph{p.o, p.ml, s_half, H};
        const double flops = fl(s_half, cut);
        const auto t_start = std::chrono::steady_clock::now();
        double total_ms = 0.0;
        unsigned total_n = 0;
        for (unsigned window = 0; ; ++window) {
            const auto w0 = std::chrono::steady_clock::now();
            unsigned n = 0;
            CK(hipEventRecord(e0));
            while (std::chrono::steady_clock::now() - w0 < std::chrono::seconds(5)) {
                launch_partials(qh, &b0, 1, ph, 0, nullptr);
                ++n;
                CK(hipDeviceSynchronize());
            }
            CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            total_ms += ms; total_n += n;
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            std::printf("SUSTAIN t=%5.1f s  %3u launches  %6.2f TFLOP/s  sclk %s  temp %s mC  power %s uW\n", secs, n, flops * n / (ms * 1e9),
                        probe("cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep '[*]'").c_str(),
                        probe("cat /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1").c_str(),
                        probe("cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_average 2>/dev/null | head -1").c_str());
            std::fflush(stdout);
            if (secs >= 70.0) break;
        }
        std::printf("SUSTAIN mean over %.0f s: %.2f TFLOP/s\n", total_ms / 1e3, flops * total_n / (total_ms * 1e9));
        d.release();
        return;
    }
    if (mode.rfind("sustainab", 0) == 0) {  // "sustainab<R>": R rounds, each variant 5 s in turn at the 2-node per-block shape
        const unsigned rounds = mode.size() > 9 ? static_cast<unsigned>(std::atoi(mode.c_str() + 9)) : 4U;
        const unsigned vars[] = {2U, 15U, 3U, 16U, 17U, 18U};
        constexpr unsigned NV = sizeof vars / sizeof vars[0];
        double tms[NV] = {}, tn[NV] = {}, ghz_sum[NV] = {};
        unsigned ghz_n[NV] = {};
        const Queries qh = d.queries(0, s_half);
        Partials ph{p.o, p.ml, s_half, H};
        const double flops = fl(s_half, cut);
        const auto sclk_mhz = [] {
            FILE* f = popen("cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep '[*]'", "r");
            char buf[128] = {};
            if (f) { if (!std::fgets(buf, sizeof buf, f)) buf[0] = 0; pclose(f); }
            const char* c = std::strchr(buf, ':');
            return c ? std::atof(c + 1) : 0.0;
        };
        for (unsigned r = 0; r < rounds; ++r)
            for (unsigned i = 0; i < NV; ++i) {
                schedule_variant(vars[i]);
                launch_partials(qh, &b0, 1, ph, 0, nullptr);  // warm-up launch per window
                CK(hipDeviceSynchronize());
                const double mhz0 = sclk_mhz();
                const auto w0 = std::chrono::steady_clock::now();
                unsigned n = 0;
                CK(hipEventRecord(e0));
                while (std::chrono::steady_clock::now() - w0 < std::chrono::seconds(5)) {
                    launch_partials(qh, &b0, 1, ph, 0, nullptr);
                    ++n;
                    CK(hipDeviceSynchronize());
                }
                CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1));
                const double mhz = 0.5 * (mhz0 + sclk_mhz());
                tms[i] += ms; tn[i] += n;
                if (mhz > 0) { ghz_sum[i] += mhz / 1e3; ++ghz_n[i]; }
                std::printf("SUSTAINAB round %u %-12s %3u launches %6.2f TFLOP/s sclk ~%.0f MHz\n", r, schedule_name(vars[i]), n, flops * n / (ms * 1e9), mhz);
                std::fflush(stdout);
            }
        for (unsigned i = 0; i < NV; ++i) {
            const double tf = flops * tn[i] / (tms[i] * 1e9), g = ghz_n[i] ? ghz_sum[i] / ghz_n[i] : 0.0;
            std::printf("SUSTAINAB mean %-12s %.0f s  %6.2f TFLOP/s  sclk %.2f GHz  %.2f TF/GHz\n", schedule_name(vars[i]), tms[i] / 1e3, tf, g, g > 0 ? tf / g : 0.0);
        }
        schedule_default();
        d.release();
        return;
    }
    if (mode.rfind("pairab", 0) == 0) {  // "pairab<R>": W8 early vs W16 early, R paired rounds (order alternates), 5 s each
        const unsigned rounds = mode.size() > 6 ? static_cast<unsigned>(std::atoi(mode.c_str() + 6)) : 8U;
        const unsigned va = 2U, vb = 3U;  // W8 early, W16 early (default)
        const Queries qh = d.queries(0, s_half), qf = d.queries(0, s_full);
        Partials ph{p.o, p.ml, s_half, H};
        const double flops = fl(s_half, cut);
        const auto window = [&](unsigned v) {
            schedule_variant(v);
            launch_partials(qh, &b0, 1, ph, 0, nullptr);  // warm-up launch per window
            CK(hipDeviceSynchronize());
            const auto w0 = std::chrono::steady_clock::now();
            unsigned n = 0;
            CK(hipEventRecord(e0));
            while (std::chrono::steady_clock::now() - w0 < std::chrono::seconds(5)) {
                launch_partials(qh, &b0, 1, ph, 0, nullptr);
                ++n;
                CK(hipDeviceSynchronize());
            }
            CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            return flops * n / (ms * 1e9);
        };
        std::vector<double> gain;
        double sa = 0, sb = 0;
        for (unsigned r = 0; r < rounds; ++r) {
            double ta, tb;
            if (r & 1U) { tb = window(vb); ta = window(va); } else { ta = window(va); tb = window(vb); }
            sa += ta; sb += tb;
            gain.push_back(100.0 * (tb - ta) / ta);
            std::printf("PAIRAB round %2u (%s first)  W8 early %6.2f  W16 early %6.2f TFLOP/s  gain %+6.2f %%\n", r, r & 1U ? "W16" : "W8", ta, tb, gain.back());
            std::fflush(stdout);
        }
        std::vector<double> g = gain;
        std::sort(g.begin(), g.end());
        const double med = g.size() % 2 ? g[g.size() / 2] : 0.5 * (g[g.size() / 2 - 1] + g[g.size() / 2]);
        std::printf("PAIRAB %u rounds: mean W8 early %.2f, W16 early %.2f TFLOP/s; paired gain median %+.2f %%, min %+.2f %%, max %+.2f %%, rounds >= 0: %u/%u\n",
                    rounds, sa / rounds, sb / rounds, med, g.front(), g.back(),
                    static_cast<unsigned>(std::count_if(g.begin(), g.end(), [](double x) { return x >= 0.0; })), rounds);
        const auto hash = [](const void* dev, std::size_t bytes) {
            std::vector<std::uint64_t> h((bytes + 7) / 8, 0);
            CK(hipMemcpy(h.data(), dev, bytes, hipMemcpyDeviceToHost));
            std::uint64_t x = 0xcbf29ce484222325ULL;
            for (const std::uint64_t w : h) { x ^= w; x *= 0x100000001b3ULL; x ^= x >> 29; }
            return x;
        };
        for (unsigned v : {va, vb}) {  // same hashes as --bench all: 2-node partial bytes, 1-node merged out, fused out
            schedule_variant(v);
            launch_partials(qh, &b0, 1, ph, 0, nullptr);
            CK(hipDeviceSynchronize());
            const std::uint64_t h2 = hash(ph.o, std::size_t(H) * s_half * kHeadDim * 4) ^ hash(ph.ml, std::size_t(H) * s_half * 8) * 3;
            launch_partials(qf, both, 2, p, 0, nullptr); launch_merge(p, 2, out, nullptr);
            CK(hipDeviceSynchronize());
            const std::uint64_t h1 = hash(out.out, std::size_t(s_full) * H * kHeadDim * 2);
            launch_fused_single(qf, d.block(0, s_full), out, nullptr);
            CK(hipDeviceSynchronize());
            const std::uint64_t hf = hash(out.out, std::size_t(s_full) * H * kHeadDim * 2);
            std::printf("PAIRAB hashes %-10s %016llx %016llx %016llx\n", schedule_name(v), static_cast<unsigned long long>(h2),
                        static_cast<unsigned long long>(h1), static_cast<unsigned long long>(hf));
        }
        schedule_default();
        d.release();
        return;
    }
    if (mode == "order") {  // thermal / order check: cold fused first, the shapes, then the fused again
        const auto clock = [] {
            FILE* f = popen("cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep '[*]'", "r");
            char buf[128] = {};
            if (f) { if (!std::fgets(buf, sizeof buf, f)) buf[0] = 0; pclose(f); }
            std::printf("CLOCK %s", buf[0] ? buf : "n/a (no sclk file)\x0a");
            std::fflush(stdout);
        };
        const Queries qf = d.queries(0, s_full), qh = d.queries(0, s_half);
        Partials ph{p.o, p.ml, s_half, H};
        const KeyBlock whole = d.block(0, s_full);
        clock();
        timed("cold: fused 37910 x 37910", fl(s_full, s_full), [&] { launch_fused_single(qf, whole, out, nullptr); });
        clock();
        timed("fused 18955 q x 37910 k", fl(s_half, s_full), [&] { launch_fused_single(qh, whole, out, nullptr); });
        timed("1-node: 37910 q x 2 blocks", fl(s_full, s_full), [&] { launch_partials(qf, both, 2, p, 0, nullptr); });
        timed("1-node: 37910 q x block 0 only", fl(s_full, cut), [&] { launch_partials(qf, &b0, 1, p, 0, nullptr); });
        timed("2-node: 18955 q x block 0", fl(s_half, cut), [&] { launch_partials(qh, &b0, 1, ph, 0, nullptr); });
        timed("2-node: 18955 q x 2 blocks (one launch)", fl(s_half, s_full), [&] { launch_partials(qh, both, 2, ph, 0, nullptr); });
        clock();
        timed("again: fused 37910 x 37910", fl(s_full, s_full), [&] { launch_fused_single(qf, whole, out, nullptr); });
        clock();
        d.release();
        return;
    }
    // 2-node per-node shape: 18955 local queries against block 0 and block 1 (one launch each), then the merge.
    const Queries qh = d.queries(0, s_half);
    Partials ph{p.o, p.ml, s_half, H};
    timed("2-node shape: 18955 q x block 0 (18944 keys)", fl(s_half, cut), [&] { launch_partials(qh, &b0, 1, ph, 0, nullptr); });
    timed("2-node shape: 18955 q x block 1 (18966 keys)", fl(s_half, s_full - cut), [&] { launch_partials(qh, &b1, 1, ph, 1, nullptr); });
    timed("2-node shape: merge 2 slots, 18955 rows", 0, [&] { launch_merge(ph, 2, out, nullptr); });
    timed("2-node shape: total 18955 q x 37910 k (2 launches + merge)", fl(s_half, s_full), [&] {
        launch_partials(qh, &b0, 1, ph, 0, nullptr); launch_partials(qh, &b1, 1, ph, 1, nullptr); launch_merge(ph, 2, out, nullptr);
    });
    // 1-node canonical: 37910 queries x both blocks (one launch, z = 2), then the merge.
    const Queries qf = d.queries(0, s_full);
    timed("1-node: 37910 q x 2 blocks (one launch)", fl(s_full, s_full), [&] { launch_partials(qf, both, 2, p, 0, nullptr); });
    timed("1-node: total 37910 x 37910 (partials + merge)", fl(s_full, s_full), [&] {
        launch_partials(qf, both, 2, p, 0, nullptr); launch_merge(p, 2, out, nullptr);
    });
    for (unsigned ab : {1U, 2U, 3U, 4U, 7U, 11U}) {
        if (std::getenv("STRIX_H3_ATTN_BENCH_ABLATE") == nullptr) break;  // opt-in (compiler artefacts)
        char name[96];
        std::snprintf(name, sizeof name, "ablation %u (1 softmax 2 loads 4 PV 8 QK), 1-node 2 blocks", ab);
        timed(name, fl(s_full, s_full), [&] { launch_partials_ablate(qf, both, 2, p, ab, nullptr); });
    }
    const KeyBlock whole = d.block(0, s_full);
    // device bytes -> 64-bit hash (bitwise check of every variant against the default at the bench shapes)
    const auto hash = [](const void* dev, std::size_t bytes) {
        std::vector<std::uint64_t> h((bytes + 7) / 8, 0);
        CK(hipMemcpy(h.data(), dev, bytes, hipMemcpyDeviceToHost));
        std::uint64_t x = 0xcbf29ce484222325ULL;
        for (const std::uint64_t w : h) { x ^= w; x *= 0x100000001b3ULL; x ^= x >> 29; }
        return x;
    };
    std::uint64_t ref2 = 0, ref1 = 0, reff = 0;
    for (unsigned v : {2U, 15U, 3U, 16U, 17U, 18U}) {  // W8 early, W8 tune4, W16 early (default), W16 tune2/3/4
        schedule_variant(v);
        char name[96];
        std::snprintf(name, sizeof name, "%s: 2-node 18955 q x block 0", schedule_name(v));
        timed(name, fl(s_half, cut), [&] { launch_partials(qh, &b0, 1, ph, 0, nullptr); });
        CK(hipDeviceSynchronize());
        const std::uint64_t h2 = hash(ph.o, std::size_t(H) * s_half * kHeadDim * 4) ^ hash(ph.ml, std::size_t(H) * s_half * 8) * 3;
        std::snprintf(name, sizeof name, "%s: 1-node 37910 q x 2 blocks + merge", schedule_name(v));
        timed(name, fl(s_full, s_full), [&] { launch_partials(qf, both, 2, p, 0, nullptr); launch_merge(p, 2, out, nullptr); });
        CK(hipDeviceSynchronize());
        const std::uint64_t h1 = hash(out.out, std::size_t(s_full) * H * kHeadDim * 2);
        std::snprintf(name, sizeof name, "%s: fused 37910 x 37910", schedule_name(v));
        timed(name, fl(s_full, s_full), [&] { launch_fused_single(qf, d.block(0, s_full), out, nullptr); });
        CK(hipDeviceSynchronize());
        const std::uint64_t hf = hash(out.out, std::size_t(s_full) * H * kHeadDim * 2);
        if (v == 2U) { ref2 = h2; ref1 = h1; reff = hf; }
        std::printf("BITWISE %-12s vs W8 early: 2-node partial bytes %s, 1-node merged out %s, fused out %s (%016llx %016llx %016llx)\n", schedule_name(v),
                    h2 == ref2 ? "SAME" : "DIFF", h1 == ref1 ? "SAME" : "DIFF", hf == reff ? "SAME" : "DIFF",
                    static_cast<unsigned long long>(h2), static_cast<unsigned long long>(h1), static_cast<unsigned long long>(hf));
        std::fflush(stdout);
    }
    schedule_default();
    timed("1-node fused single block 37910 x 37910", fl(s_full, s_full), [&] { launch_fused_single(qf, whole, out, nullptr); });
    d.release();
    CK(hipFree(p.o)); CK(hipFree(p.ml)); CK(hipFree(out.out));
}
}  // namespace

// --env-check: with a malformed STRIX_H3_ATTN_* value in the environment, the first launch must throw
// std::invalid_argument before enqueueing anything (run by CTest with the variable set).
int env_check() {
    const auto x = make_qkv(H, 64, 64, 120, 1.0F);
    const auto d = upload(x, true);
    Partials p{dalloc<float>(std::size_t(H) * 64 * kHeadDim), dalloc<float>(std::size_t(H) * 64 * 2), 64, H};
    const KeyBlock b = d.block(0, 64);
    bool threw = false;
    try {
        launch_partials(d.queries(0, 64), &b, 1, p, 0, nullptr);
        CK(hipDeviceSynchronize());
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::printf("rejected: %s\n", e.what());
    }
    std::printf("%s malformed environment switch rejected at first use\n", threw ? "PASS" : "FAIL");
    return threw ? 0 : 1;
}

int main(int argc, char** argv) {
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s, %d CUs, LDS/WG %zu, clock %d kHz\n", prop.gcnArchName, prop.multiProcessorCount, prop.sharedMemPerBlock, prop.clockRate);
    CK(hipMalloc(&g_bound, 4));
    CK(hipMemset(g_bound, 0, 4));
    if (argc > 1 && std::string(argv[1]) == "--env-check") return env_check();
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        bench(argc > 2 ? argv[2] : "all");
        return 0;
    }
    {   // default resolution with nothing set (setter cleared, STRIX_H3_ATTN_WAVES unset) -> 16 waves; same bits as W8
        if (std::getenv("STRIX_H3_ATTN_WAVES") != nullptr) {
            std::printf("SKIP unset-environment resolver check (STRIX_H3_ATTN_WAVES is set)\n");
        } else {
            set_waves_per_wg(0);
            check(waves_per_wg_resolved() == 16U, "STRIX_H3_ATTN_WAVES unset, no setter -> 16 waves per work-group (default)");
            const auto x = make_qkv(H, 300, 1000, 122, 1.0F);
            const auto d = upload(x, true);
            const Result r_env = run_split(d, {0, 512, 1000}, 0, 300, true, false);
            set_waves_per_wg(8);
            const Result r8 = run_split(d, {0, 512, 1000}, 0, 300, true, false);
            check(same(r_env, r8) && same_partials(r_env, r8, 0), "unset-environment default == W8 (outputs + partial bytes)");
            d.release();
        }
        schedule_default();
    }
    correctness_case("S256 1 block head-major", 256, 256, {0, 256}, 1.0F, false, 1, 101);
    {   // empty blocks: neutral in the merge; all-empty -> zeros
        const auto x = make_qkv(H, 300, 1000, 108, 1.0F);
        const auto d = upload(x, true);
        const Result with_empty = run_split(d, {0, 512, 512, 1000}, 0, 300, true, false), plain = run_split(d, {0, 512, 1000}, 0, 300, true, false);
        check(same(with_empty, plain), "empty middle block is neutral in the merge (bitwise)");
        const Result none = run_split(d, {512, 512}, 0, 300, true, false);
        bool zeros = true;
        for (float v : none.f32) zeros &= v == 0.0F;
        for (std::size_t i = 0; i < none.pml.size(); i += 2) zeros &= none.pml[i] == -__builtin_inff() && none.pml[i + 1] == 0.0F;
        for (float v : none.po) zeros &= v == 0.0F && !std::signbit(v);
        for (std::uint16_t v : none.bf) zeros &= v == 0U;
        check(zeros, "all-empty merge -> zeros (bf16, fp32); empty partial bytes = (-inf, 0, +0 x 128)");
        d.release();
    }
    correctness_case("S1000 2 blocks token-major", 1000, 1000, {0, 512, 1000}, 1.0F, true, 1, 102);
    correctness_case("S1000 3 blocks q x4", 1000, 1000, {0, 320, 704, 1000}, 4.0F, true, 2, 103);
    correctness_case("S1000 1 block (fused check)", 1000, 1000, {0, 1000}, 2.0F, true, 3, 104);
    correctness_case("Sq777 x Sk1300 2 blocks", 777, 1300, {0, 648, 1300}, 1.0F, false, 3, 105);
    correctness_case("S4093 2 blocks token-major", 4093, 4093, {0, 2048, 4093}, 1.0F, true, 29, 106);
    correctness_case("S4093 1 block q x4", 4093, 4093, {0, 4093}, 4.0F, true, 61, 107);
    correctness_case("S1000 4 blocks (last 40 keys)", 1000, 1000, {0, 320, 704, 960, 1000}, 2.0F, true, 2, 109);
    correctness_case("Sq300 x Sk529 2 blocks (last block 17 keys)", 300, 529, {0, 512, 529}, 1.0F, true, 1, 110);
    correctness_case("Sq300 x Sk17 1 block of 17 keys", 300, 17, {0, 17}, 1.0F, true, 1, 111);
    {   // ties: every key of a head equal -> all scores tie, out = mean of V
        auto x = make_qkv(H, 300, 777, 112, 3.0F);
        for (unsigned h = 0; h < H; ++h)
            for (unsigned j = 1; j < x.k_rows; ++j)
                std::memcpy(&x.k[(std::size_t(h) * x.k_rows + j) * kHeadDim], &x.k[std::size_t(h) * x.k_rows * kHeadDim], kHeadDim * 2);
        correctness_case_x("ties: all keys equal, Sk777 2 blocks", x, {0, 384, 777}, true, 1);
    }
    {   // extreme scores at the operand contract: q = k = 356 e0 for one key (||q|| ||k|| c = 16162 <= 2^14), the
        // negated q on half the rows (extreme negative scores), other keys spread
        auto x = make_qkv(H, 300, 1000, 113, 1.0F);
        const std::uint16_t big = bf16_from_float(356.0F);
        for (unsigned h = 0; h < H; ++h) {
            for (unsigned r = 0; r < x.q_rows; ++r) {
                std::uint16_t* q = &x.q[(std::size_t(h) * x.q_rows + r) * kHeadDim];
                for (unsigned dd = 0; dd < kHeadDim; ++dd) q[dd] = 0;
                q[0] = r % 2 ? big : static_cast<std::uint16_t>(big | 0x8000U);  // half the rows see the key at -s
            }
            x.k[(std::size_t(h) * x.k_rows + 700) * kHeadDim] = big;
        }
        read_bound();
        correctness_case_x("extreme scores at the operand contract (|s c| = 16162), Sk1000 2 blocks", x, {0, 512, 1000}, true, 1);
        check(read_bound() == 0U, "input-bound telemetry: 0 rows flagged at the operand contract");
        // beyond: q = k = 1024 e0 for rows 0..9 -> s c = 133718 > 2^15 for block 1 only
        for (unsigned h = 0; h < H; ++h) {
            for (unsigned r = 0; r < x.q_rows; ++r) x.q[(std::size_t(h) * x.q_rows + r) * kHeadDim] = r < 10 ? bf16_from_float(1024.0F) : 0;
            x.k[(std::size_t(h) * x.k_rows + 700) * kHeadDim] = bf16_from_float(1024.0F);
        }
        const auto d = upload(x, true);
        (void)run_split(d, {0, 512, 1000}, 0, 300, true, false);
        const unsigned over = read_bound();
        check(over == 10U * H, "input-bound telemetry counts the rows beyond the bound: " + std::to_string(over) + " == " + std::to_string(10U * H));
        for (int split : {0, 1}) {  // fused single block (key 700 inside it): the same 10 rows per head, once each
            set_head_split(split);
            read_bound();
            (void)run_fused(d, g_bound);
            const unsigned fo = read_bound();
            check(fo == 10U * H, std::string("fused path telemetry counts the rows beyond the bound (head split ") + (split ? "on" : "off") +
                                     "): " + std::to_string(fo) + " == " + std::to_string(10U * H));
        }
        set_head_split(-1);
        d.release();
    }
    {   // in contract, entirely negative first tile: q = 8 e0, keys 0..63 = -50 e0 (s c = -51), later keys spread
        auto x = make_qkv(H, 300, 1000, 119, 1.0F);
        for (unsigned h = 0; h < H; ++h) {
            for (unsigned r = 0; r < x.q_rows; ++r) {
                std::uint16_t* q = &x.q[(std::size_t(h) * x.q_rows + r) * kHeadDim];
                for (unsigned dd = 0; dd < kHeadDim; ++dd) q[dd] = 0;
                q[0] = bf16_from_float(8.0F);
            }
            for (unsigned j = 0; j < 64; ++j) {
                std::uint16_t* k = &x.k[(std::size_t(h) * x.k_rows + j) * kHeadDim];
                for (unsigned dd = 0; dd < kHeadDim; ++dd) k[dd] = 0;
                k[0] = bf16_from_float(-50.0F);
            }
        }
        read_bound();
        correctness_case_x("negative first tile (in contract): keys 0..63 at s c = -51, Sk1000 2 blocks", x, {0, 512, 1000}, true, 1);
        check(read_bound() == 0U, "negative first tile: 0 rows flagged");
    }
    {   // out of contract, transient: first tile K = -999424 e0 (s c = -1.27e5 for q = e0), then zero keys -> every
        // (row, head) of the block is flagged once even though the final max is 0
        auto x = make_qkv(H, 128, 200, 117, 1.0F);
        for (unsigned h = 0; h < H; ++h) {
            for (unsigned r = 0; r < x.q_rows; ++r) {
                std::uint16_t* q = &x.q[(std::size_t(h) * x.q_rows + r) * kHeadDim];
                for (unsigned dd = 0; dd < kHeadDim; ++dd) q[dd] = 0;
                q[0] = bf16_from_float(1.0F);
            }
            for (unsigned j = 0; j < x.k_rows; ++j) {
                std::uint16_t* k = &x.k[(std::size_t(h) * x.k_rows + j) * kHeadDim];
                for (unsigned dd = 0; dd < kHeadDim; ++dd) k[dd] = 0;
                if (j < 64) k[0] = bf16_from_float(-999424.0F);
            }
        }
        read_bound();
        const auto d = upload(x, true);
        (void)run_split(d, {0, 200}, 0, 128, true, false);
        const unsigned flagged = read_bound();
        check(flagged == 128U * H, "telemetry latches a transient out-of-contract step (K = -999424 e0 then 0): " + std::to_string(flagged) +
                                       " == " + std::to_string(128U * H));
        read_bound();
        (void)run_fused(d, g_bound);
        const unsigned ff = read_bound();
        check(ff == 128U * H, "fused path telemetry latches the transient step: " + std::to_string(ff) + " == " + std::to_string(128U * H));
        d.release();
    }
    {   // invalid API calls are rejected before anything is enqueued
        const auto x = make_qkv(H, 64, 128, 114, 1.0F);
        const auto d = upload(x, true);
        float* scratch = dalloc<float>(attention_scratch_floats(64, H, kHeadDim, 4) + 4);
        std::uint16_t* out = dalloc<std::uint16_t>(std::size_t(64) * H * kHeadDim);
        const std::uint32_t cuts2[3] = {0, 64, 128}, cuts_bad[3] = {0, 100, 128}, cuts5[6] = {0, 64, 64, 64, 64, 128};
        const auto call = [&] {
            AttentionCall c;
            c.q = d.q; c.k = d.k; c.vt = d.vt; c.rows = 64; c.keys = 128; c.vt_pitch = d.vt_ds;
            c.cuts = cuts2; c.nblocks = 2; c.scratch = scratch; c.out = out;
            return c;
        };
        bool ok = true;
        ok &= !throws_invalid([&] { attention(call(), nullptr); CK(hipDeviceSynchronize()); });
        ok &= throws_invalid([&] { auto c = call(); c.vt_pitch = 64; attention(c, nullptr); });
        ok &= throws_invalid([&] { auto c = call(); c.vt_pitch = 132; attention(c, nullptr); });
        ok &= throws_invalid([&] { auto c = call(); c.out_rs = 8; attention(c, nullptr); });
        ok &= throws_invalid([&] { auto c = call(); c.out_rs = std::uint64_t(H) * kHeadDim + 4; attention(c, nullptr); });
        ok &= throws_invalid([&] { auto c = call(); c.scratch = scratch + 1; attention(c, nullptr); });
        ok &= throws_invalid([&] { auto c = call(); c.out = out + 1; attention(c, nullptr); });
        ok &= throws_invalid([&] { auto c = call(); c.cuts = cuts_bad; attention(c, nullptr); });
        ok &= throws_invalid([&] { auto c = call(); c.cuts = cuts5; c.nblocks = 5; attention(c, nullptr); });
        ok &= throws_invalid([&] { auto c = call(); c.keys = 127; attention(c, nullptr); });
        ok &= throws_invalid([&] { auto c = call(); c.dim = 64; attention(c, nullptr); });
        ok &= throws_invalid([&] {
            Queries q = d.queries(0, 64);
            q.dim = 64;
            const KeyBlock b = d.block(0, 128);
            launch_partials(q, &b, 1, Partials{reinterpret_cast<float*>(scratch), scratch, 64, H, 64}, 0, nullptr);
        });
        ok &= throws_invalid([&] {
            const Partials p{scratch, scratch + 2ULL * H * 64 * kHeadDim, 64, H};
            launch_merge(p, 2, Output{out, nullptr, 8}, nullptr);
        });
        ok &= throws_invalid([&] {
            launch_fused_single(d.queries(0, 64), d.block(0, 128), Output{out, nullptr, 64}, nullptr);
        });
        CK(hipDeviceSynchronize());
        check(ok, "invalid API calls (vt_pitch, out_rs, alignment, cuts, nblocks, keys, dim 64, merge/fused out_rs) rejected");
        CK(hipFree(scratch)); CK(hipFree(out));
        d.release();
    }
    {   // padded / masked keys are staged as zeros: V^T pad filled with NaN, huge (0x7f7f = 3.4e38) or zero, same bits;
        // every schedule; a block's result does not depend on the keys after its end (block 0 of 648 keys alone).
        const auto x = make_qkv(H, 300, 1300 - 3, 116, 2.0F);  // 1297 keys: pad of 7 in the last 8-key chunk
        const std::vector<unsigned> cuts = {0, 648, 1297};    // block 0 ends inside a 64-key tile
        bool ok = true;
        std::vector<Result> ref;
        for (int pad : {0x00, 0x7f, 0xff}) {
            const auto d = upload(x, true, pad);
            for (unsigned v = 0; v < kSchedules; ++v) {
                schedule_variant(v);
                Result r = run_split(d, cuts, 0, 300, true, false);
                if (ref.empty()) ref.push_back(r);
                ok &= same(ref[0], r) && same_partials(ref[0], r, 0);
            }
            schedule_default();
            d.release();
        }
        check(ok, "V^T pad = 0 / huge / NaN, all 19 schedules: identical outputs + partial bytes");
        HostQkv x0 = make_qkv(H, 300, 648, 116, 2.0F);  // same Q (same seed and rows): replace K/V by block 0 only
        x0.q = x.q;
        for (unsigned h = 0; h < H; ++h) {
            std::memcpy(&x0.k[std::size_t(h) * 648 * kHeadDim], &x.k[std::size_t(h) * x.k_rows * kHeadDim], 648 * kHeadDim * 2);
            std::memcpy(&x0.v[std::size_t(h) * 648 * kHeadDim], &x.v[std::size_t(h) * x.k_rows * kHeadDim], 648 * kHeadDim * 2);
        }
        const auto d0 = upload(x0, true, 0x7f);
        const Result alone = run_split(d0, {0, 648}, 0, 300, true, false);
        bool same0 = true;
        for (unsigned h = 0; h < H; ++h) {
            const std::size_t w = std::size_t(h) * 300;  // slot 0
            same0 &= std::memcmp(&ref[0].po[w * kHeadDim], &alone.po[w * kHeadDim], 300 * kHeadDim * 4) == 0 &&
                     std::memcmp(&ref[0].pml[w * 2], &alone.pml[w * 2], 300 * 2 * 4) == 0;
        }
        check(same0, "block 0 (648 keys) partial bytes: with block 1's keys after it in memory == alone with huge pad");
        d0.release();
    }
    {   // a long sequence: 110,000 keys in 2 blocks (cut 55040), 300 queries, 4 heads; then V at the 2^100 headroom
        H = 4;
        correctness_case("long: Sq300 x Sk110000 2 blocks, 4 heads", 300, 110000, {0, 55040, 110000}, 1.0F, true, 7, 115);
        auto x = make_qkv(H, 64, 110000, 118, 1.0F);
        for (auto& v : x.v)
            if (v & 0x7f80U) v = static_cast<std::uint16_t>(v + (97U << 7));  // V * 2^97 exactly (|V| < 2^3 -> < 2^100)
        bool finite = true;
        for (auto v : x.v) finite &= (v & 0x7f80U) < ((127U + 100U) << 7);  // |V| < 2^100
        check(finite, "V headroom data: V * 2^97 finite, max |V| < 2^100");
        correctness_case_x("V headroom: |V| ~ 2^100 x 110000 keys, 2 blocks, 4 heads", x, {0, 55040, 110000}, true, 3);
        H = 56;
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
