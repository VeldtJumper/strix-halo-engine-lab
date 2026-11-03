// FastAttn multi-slot speculative verify (SpecRows::seg, Engine::verify_multi): one call with the windows of several
// slots must equal the same windows verified as single-slot calls, bit for bit: every slot's S and conv state, both
// replay sets (qkv / scr / beta), the layer output of every row and the row / replay flags. Two identical states run
// the two forms through three rounds; rounds 2 and 3 replay the rows committed by the previous round (n_rep > 0, cur
// alternating). Tier B linears go through a row-independent FP32 test runner (Linear::prepared set), so the check is
// about the window plumbing (offsets, per-slot state and replay sets), not the linear kernels.
#include "glm/common/glm_math.hpp"
#include "glm/engine/glm_fast_attn.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix::glm;
namespace E = strix::glm::engine;

namespace {

int failures = 0, checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    std::printf("%s %s\n", ok ? "OK  " : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) throw std::runtime_error(std::string(w) + ": " + hipGetErrorString(e));
}

std::vector<void*> g_allocs;
template <class T> T* dalloc(std::size_t n) {
    void* p = nullptr;
    hok(hipMalloc(&p, n * sizeof(T)), "malloc");
    g_allocs.push_back(p);
    return static_cast<T*>(p);
}
template <class T> T* up(const std::vector<T>& v) {
    T* d = dalloc<T>(v.size());
    hok(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return d;
}
template <class T> std::vector<T> down(const T* d, std::size_t n) {
    std::vector<T> v(n);
    hok(hipDeviceSynchronize(), "sync");
    hok(hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return v;
}
template <class T> bool same_dev(const T* a, const T* b, std::size_t n) {
    const auto x = down(a, n), y = down(b, n);
    return std::memcmp(x.data(), y.data(), n * sizeof(T)) == 0;
}

std::uint16_t f16_bits(float f) {  // round to nearest even, finite small values only
    const _Float16 h = static_cast<_Float16>(f);
    std::uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
std::uint16_t bf16_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}
std::vector<float> rnd(std::size_t n, std::mt19937& g, float s) {
    std::normal_distribution<float> d(0.0F, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}
std::vector<std::uint16_t> rnd16(std::size_t n, std::mt19937& g, float s, bool bf) {
    std::vector<std::uint16_t> v(n);
    const auto f = rnd(n, g, s);
    for (std::size_t i = 0; i < n; ++i) v[i] = bf ? bf16_bits(f[i]) : f16_bits(f[i]);
    return v;
}

// Row-independent FP32 test linear: out[r][o] = fmaf chain over i ascending of in[r][i] * W[o][i] from +0.
struct TestMat {
    const float* W;
    unsigned in, out;
};
__global__ void test_linear_kernel(const float* W, const float* x, float* y, unsigned in, unsigned out) {
    const unsigned o = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (o >= out) return;
    const float* w = W + static_cast<std::size_t>(o) * in;
    const float* xr = x + static_cast<std::size_t>(r) * in;
    float a = 0.0F;
    for (unsigned i = 0; i < in; ++i) a = __builtin_fmaf(xr[i], w[i], a);
    y[static_cast<std::size_t>(r) * out + o] = a;
}
class TestRunner final : public E::LinearRunner {
public:
    void run(const E::Linear& l, const float* in, float* out, unsigned rows, hipStream_t s) const override {
        const auto* m = static_cast<const TestMat*>(l.prepared);
        test_linear_kernel<<<dim3((m->out + 255) / 256, rows), 256, 0, s>>>(m->W, in, out, m->in, m->out);
        hok(hipGetLastError(), "test linear");
    }
};

constexpr unsigned kHk = 32, kC = kHk * kKdaDim, kSlots = 3, kMR = 8;
constexpr std::size_t kSZ = static_cast<std::size_t>(kHk) * kKdaDim * kKdaDim, kCZ = static_cast<std::size_t>(3) * kC * 3;
constexpr std::size_t kRepQkv = static_cast<std::size_t>(kSlots) * 2 * kMR * 3 * kC;
constexpr std::size_t kRepScr = static_cast<std::size_t>(kSlots) * 2 * kMR * kHk * 4 * kKdaDim;
constexpr std::size_t kRepBeta = static_cast<std::size_t>(kSlots) * 2 * kMR * kHk;

struct State {
    E::KdaState st;
    explicit State(const State* from) {
        st.S = dalloc<float>(kSlots * kSZ);
        st.conv = dalloc<float>(kSlots * kCZ);
        st.rep_qkv = dalloc<float>(kRepQkv);
        st.rep_scr = dalloc<float>(kRepScr);
        st.rep_beta = dalloc<float>(kRepBeta);
        if (from) {
            hok(hipMemcpy(st.S, from->st.S, kSlots * kSZ * 4, hipMemcpyDeviceToDevice), "S");
            hok(hipMemcpy(st.conv, from->st.conv, kSlots * kCZ * 4, hipMemcpyDeviceToDevice), "conv");
            hok(hipMemcpy(st.rep_qkv, from->st.rep_qkv, kRepQkv * 4, hipMemcpyDeviceToDevice), "rep qkv");
            hok(hipMemcpy(st.rep_scr, from->st.rep_scr, kRepScr * 4, hipMemcpyDeviceToDevice), "rep scr");
            hok(hipMemcpy(st.rep_beta, from->st.rep_beta, kRepBeta * 4, hipMemcpyDeviceToDevice), "rep beta");
        }
    }
};

struct Win {
    std::uint32_t slot;
    unsigned n, commit;  // rows of the window; rows committed afterwards (the slot's n_rep for the next round)
};

}  // namespace

int main() {
    try {
        std::mt19937 g(20261103);
        // weights: Tier B through the test runner, Tier A small denses / conv / norms on the device
        std::vector<TestMat> mats;
        mats.reserve(4);
        E::KdaLayerWeights w{};
        for (unsigned p = 0; p < 3; ++p) {
            mats.push_back({up(rnd(static_cast<std::size_t>(kC) * kHidden, g, 0.02F)), kHidden, kC});
            w.qkv[p].prepared = &mats.back();
        }
        mats.push_back({up(rnd(static_cast<std::size_t>(kHidden) * kC, g, 0.02F)), kC, kHidden});
        w.o.prepared = &mats.back();
        w.f_a = up(rnd16(static_cast<std::size_t>(kKdaDim) * kHidden, g, 0.02F, false));
        w.g_a = up(rnd16(static_cast<std::size_t>(kKdaDim) * kHidden, g, 0.02F, false));
        w.f_b = up(rnd16(static_cast<std::size_t>(kC) * kKdaDim, g, 0.05F, false));
        w.g_b = up(rnd16(static_cast<std::size_t>(kC) * kKdaDim, g, 0.05F, false));
        w.b = up(rnd16(static_cast<std::size_t>(kHk) * kHidden, g, 0.02F, false));
        w.conv = up(rnd16(static_cast<std::size_t>(3) * kC * kConvK, g, 0.3F, true));
        {
            std::vector<float> a(kHk);
            std::uniform_real_distribution<float> u(0.0F, 2.5F);
            for (auto& x : a) x = u(g);
            w.A_log = up(a);
        }
        w.dt_bias = up(rnd(kC, g, 0.5F));
        w.o_norm = up(rnd16(kKdaDim, g, 1.0F, true));
        w.heads = kHk;

        State A(nullptr);
        hok(hipMemcpy(A.st.S, rnd(kSlots * kSZ, g, 0.01F).data(), kSlots * kSZ * 4, hipMemcpyHostToDevice), "S0");
        hok(hipMemcpy(A.st.conv, rnd(kSlots * kCZ, g, 0.1F).data(), kSlots * kCZ * 4, hipMemcpyHostToDevice), "conv0");
        hok(hipMemset(A.st.rep_qkv, 0, kRepQkv * 4), "rq0");
        hok(hipMemset(A.st.rep_scr, 0, kRepScr * 4), "rs0");
        hok(hipMemset(A.st.rep_beta, 0, kRepBeta * 4), "rb0");
        State B(&A);

        auto attn = E::fast_attention(kHk, 32, 4096);
        const TestRunner runner;
        unsigned* flag = dalloc<unsigned>(1);
        unsigned* rf_m = dalloc<unsigned>(kMR);
        unsigned* rf_s = dalloc<unsigned>(kMR);
        unsigned* rep_m = dalloc<unsigned>(1);
        unsigned* rep_s = dalloc<unsigned>(1);
        float* y_m = dalloc<float>(static_cast<std::size_t>(kMR) * kHidden);
        float* y_s = dalloc<float>(static_cast<std::size_t>(kMR) * kHidden);
        std::uint32_t* slot_d = dalloc<std::uint32_t>(kMR);
        std::uint64_t* pos_d = dalloc<std::uint64_t>(kMR);

        // rounds: windows sorted by slot (Engine::verify_multi's contract), 8 rows at most
        const std::vector<std::vector<Win>> rounds = {{{0, 3, 2}, {1, 1, 1}, {2, 4, 3}},
                                                      {{0, 4, 4}, {2, 2, 1}},
                                                      {{0, 1, 1}, {1, 5, 2}, {2, 2, 2}}};
        unsigned n_rep[kSlots] = {}, cur[kSlots] = {}, pos[kSlots] = {10, 200, 3000};
        for (std::size_t k = 0; k < rounds.size(); ++k) {
            const auto& wins = rounds[k];
            unsigned R = 0;
            std::vector<E::Row> rows;
            std::vector<E::SpecSeg> seg;
            for (const Win& wi : wins) {
                seg.push_back(E::SpecSeg{wi.slot, n_rep[wi.slot], cur[wi.slot], R, wi.n});
                for (unsigned j = 0; j < wi.n; ++j) rows.push_back(E::Row{wi.slot, 1000U + j, pos[wi.slot] + j});
                R += wi.n;
            }
            std::vector<std::uint32_t> hs;
            std::vector<std::uint64_t> hp;
            for (const auto& r : rows) {
                hs.push_back(r.slot);
                hp.push_back(r.pos);
            }
            hok(hipMemcpy(slot_d, hs.data(), R * 4, hipMemcpyHostToDevice), "slots");
            hok(hipMemcpy(pos_d, hp.data(), R * 8, hipMemcpyHostToDevice), "pos");
            const float* x = up(rnd(static_cast<std::size_t>(R) * kHidden, g, 0.5F));
            for (unsigned* f : {flag, rep_m, rep_s}) hok(hipMemset(f, 0, 4), "flag");
            hok(hipMemset(rf_m, 0, kMR * 4), "rf");
            hok(hipMemset(rf_s, 0, kMR * 4), "rf");

            // multi-slot form on B: one call
            E::SpecRows sm{seg[0].slot, seg[0].n_rep, seg[0].cur, rf_m, rep_m};
            sm.seg = seg.data();
            sm.nseg = static_cast<unsigned>(seg.size());
            E::CallCtx cm{&runner, rows, slot_d, pos_d, flag, nullptr, true, &sm};
            attn->kda(w, B.st, cm, x, y_m);
            // single-slot form on A: one call per window, in window order
            for (const auto& sg : seg) {
                E::SpecRows ss{sg.slot, sg.n_rep, sg.cur, rf_s + sg.row0, rep_s};
                E::CallCtx cs{&runner, std::span<const E::Row>(rows).subspan(sg.row0, sg.n), slot_d + sg.row0, pos_d + sg.row0, flag,
                              nullptr, true, &ss};
                attn->kda(w, A.st, cs, x + static_cast<std::size_t>(sg.row0) * kHidden, y_s + static_cast<std::size_t>(sg.row0) * kHidden);
            }
            const std::string tag = "round " + std::to_string(k + 1) + " (" + std::to_string(seg.size()) + " windows, " +
                                    std::to_string(R) + " rows)";
            check(same_dev(y_m, y_s, static_cast<std::size_t>(R) * kHidden), tag + ": layer output of every row");
            check(same_dev(B.st.S, A.st.S, kSlots * kSZ) && same_dev(B.st.conv, A.st.conv, kSlots * kCZ),
                  tag + ": S and conv of every slot");
            check(same_dev(B.st.rep_qkv, A.st.rep_qkv, kRepQkv) && same_dev(B.st.rep_scr, A.st.rep_scr, kRepScr) &&
                      same_dev(B.st.rep_beta, A.st.rep_beta, kRepBeta),
                  tag + ": both replay sets (qkv, scr, beta) of every slot");
            check(same_dev(rf_m, rf_s, R) && same_dev(rep_m, rep_s, 1), tag + ": row flags and replay flag");
            const auto f = down(rf_m, R);
            bool clean = down(rep_m, 1)[0] == 0;
            for (unsigned v : f) clean = clean && v == 0;
            check(clean, tag + ": no non-finite flag (finite test data)");
            {
                const auto ym = down(y_m, static_cast<std::size_t>(R) * kHidden);
                bool finite = true, nonzero = false;
                for (float v : ym) {
                    finite = finite && std::isfinite(v);
                    nonzero = nonzero || v != 0.0F;
                }
                check(finite && nonzero, tag + ": outputs finite and not all zero");
            }
            for (const Win& wi : wins) {  // commit: the next call replays wi.commit rows from the set just written
                n_rep[wi.slot] = wi.commit;
                cur[wi.slot] ^= 1U;
                pos[wi.slot] += wi.commit;
            }
        }
        for (void* p : g_allocs) (void)hipFree(p);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("checks: %d failures: %d\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
