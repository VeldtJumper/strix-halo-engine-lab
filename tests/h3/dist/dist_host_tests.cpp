// MiniMax H3 two-node schedule (D1): CPU tests of the host reference with the in-process K/V exchange.
//   strix-h3-dist-host-tests unit            no checkpoint: CRC, geometry, partial arithmetic, exchange hazards, and the
//                                            whole schedule on random weights (S = 300, 2 blocks)
//   strix-h3-dist-host-tests forward [mid]   fixtures + checkpoint (environment H3_FIXTURES = the fixture root, H3_CKPT =
//                                            the DiT checkpoint; SKIPPED when unset): dit_t2va_small (S = 400, H3_DIST_BLOCKS
//                                            blocks, default 50) [and dit_t2va_mid, S = 2122, default 4]
// Exit status: 0 PASSED, 1 FAILED, 77 SKIPPED (a requested fixture mode had no assets; never reported as PASSED).
// Gate (G3 on the host): 1 node (block_forward_host with the canonical 2-block split) == world 1 == world 2
// (rank 0 | rank 1 concatenated) == world 2 with swapped ranks, bitwise, per block (FNV-1a of each key block's rows).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "h3/attn/attn_ref.hpp"
#include "h3/block/host_ref.hpp"
#include "h3/block/layout.hpp"
#include "h3/block/weights.hpp"
#include "h3/common/fixture.hpp"
#include "h3/dist/admission.hpp"
#include "h3/dist/host_attn.hpp"
#include "h3/dist/host_dist.hpp"
#include "h3/dist/kv_exchange.hpp"

using namespace strix::h3;
using namespace strix::h3::dist;
namespace fx = strix::h3::fixture;

namespace {

int g_fail = 0;
int g_skipped = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++g_fail;
}
std::string env_or(const char* n, const char* d) {
    const char* e = std::getenv(n);
    return e && *e ? std::string(e) : std::string(d);
}
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <class F> bool throws(F&& f, std::string* what = nullptr) {
    try {
        f();
    } catch (const std::exception& e) {
        if (what) *what = e.what();
        return true;
    }
    return false;
}

// ------------------------------------------------------------------------------------------------ random model
std::uint16_t rbf(std::mt19937_64& g, double sd, double mean = 0.0) {
    std::normal_distribution<double> nd(mean, sd);
    return block::tob(static_cast<float>(nd(g)));
}
block::QLinear rand_q(std::mt19937_64& g, std::uint32_t n, std::uint32_t k) {
    block::QLinear l;
    l.n = n; l.k = k;
    l.w.resize(std::size_t{n} * k);
    std::uniform_int_distribution<int> u(-127, 127);
    for (auto& v : l.w) v = static_cast<std::int8_t>(u(g));
    l.ws.assign(n, static_cast<float>(1.0 / (127.0 * std::sqrt(static_cast<double>(k)))));
    return l;
}
struct RandomBlocks final : block::BlockSource {
    std::uint32_t n, basis;
    std::vector<std::shared_ptr<const block::BlockWeights>> cache;
    RandomBlocks(std::uint32_t nb, std::uint32_t bs) : n(nb), basis(bs), cache(nb) {}
    std::shared_ptr<const block::BlockWeights> block(std::uint32_t i) override {
        if (cache.at(i)) return cache[i];
        std::mt19937_64 g(0x48334453ULL + i);
        auto b = std::make_shared<block::BlockWeights>();
        b->index = i;
        for (auto* v : {&b->norm1, &b->norm2}) { v->resize(block::kHidden); for (auto& e : *v) e = rbf(g, 0.2, 1.0); }
        for (auto* v : {&b->q_norm, &b->k_norm}) { v->resize(block::kHeadDim); for (auto& e : *v) e = rbf(g, 0.2, 1.0); }
        b->adaln.n = 6U * block::kHidden * 3U;
        b->adaln.k = basis;
        b->adaln.w.resize(std::size_t{b->adaln.n} * basis);
        b->adaln.b.assign(b->adaln.n, 0.0F);
        std::normal_distribution<float> nd(0.0F, 0.05F);
        for (auto& e : b->adaln.w) e = block::bf(block::tob(nd(g)));
        b->qkv = rand_q(g, 3U * block::kInner, block::kHidden);
        b->out = rand_q(g, block::kHidden, block::kInner);
        b->fc1 = rand_q(g, 2U * block::kFfn, block::kHidden);
        b->fc2 = rand_q(g, block::kHidden, block::kFfn);
        cache[i] = b;
        return b;
    }
    std::uint32_t num_blocks() const override { return n; }
};

// ------------------------------------------------------------------------------------------------ runners
// 1-node reference: the block package's block_forward_host with the canonical two-block split attention.
std::vector<std::uint16_t> run_reference(block::BlockSource& blocks, std::uint32_t nb, const ForwardTables& t, std::uint32_t S,
    std::vector<std::uint16_t> x, block::LinearOp& lin, std::vector<std::array<std::uint64_t, 2>>* hashes) {
    SplitKvHostAttention att;
    const Split sp = Split::of(S);
    for (std::uint32_t i = 0; i < nb; ++i) {
        const auto b = blocks.block(i);
        const block::ModVectors mv = block::mod_vectors(block::adaln_block_f32(*b, t.t_emb, t.n_t), t.n_t * 3U);
        block::block_forward_host(*b, mv, t.mod_row, t.rope, x, S, lin, att);
        if (hashes)
            hashes->push_back({fnv1a64(x.data(), std::size_t{sp.rows(0)} * block::kHidden * 2U),
                fnv1a64(x.data() + std::size_t{sp.cut} * block::kHidden, std::size_t{sp.rows(1)} * block::kHidden * 2U)});
    }
    return x;
}

std::vector<std::uint16_t> run_world1(block::BlockSource& blocks, std::uint32_t nb, const ForwardTables& t, std::uint32_t S,
    std::vector<std::uint16_t> x, block::LinearOp& lin, DistTrace* tr) {
    dist_blocks_host(blocks, nb, t, Split::of(S), Role{0, 1, false}, x, lin, nullptr, tr);
    return x;
}

// Two ranks in threads over the in-process pair; returns rank rows concatenated in row order.
std::vector<std::uint16_t> run_world2(block::BlockSource& blocks, std::uint32_t nb, const ForwardTables& t, std::uint32_t S,
    const std::vector<std::uint16_t>& x, block::LinearOp& lin, bool swap, DistTrace tr[2], std::chrono::microseconds delay) {
    const Split sp = Split::of(S);
    MemoryKvPair pair(delay, 8);
    SharedBlockSource shared(blocks, 3);
    std::vector<std::uint16_t> xr[2];
    std::exception_ptr err[2];
    std::thread th[2];
    for (unsigned r = 0; r < 2; ++r) {
        th[r] = std::thread([&, r] {
            const Role role{r, 2, swap};
            const std::size_t off = std::size_t{role.row0(sp)} * block::kHidden;
            xr[r].assign(x.begin() + static_cast<std::ptrdiff_t>(off),
                x.begin() + static_cast<std::ptrdiff_t>(off + std::size_t{role.rows(sp)} * block::kHidden));
            try {
                dist_blocks_host(shared, nb, t, sp, role, xr[r], lin, &pair.endpoint(r), &tr[r]);
            } catch (const std::exception& e) {
                err[r] = std::current_exception();
                pair.endpoint(r).abort(e.what());
            }
        });
    }
    for (auto& h : th) h.join();
    for (auto& e : err)
        if (e) std::rethrow_exception(e);
    std::vector<std::uint16_t> out(x.size());
    for (unsigned r = 0; r < 2; ++r) {
        const Role role{r, 2, swap};
        std::memcpy(out.data() + std::size_t{role.row0(sp)} * block::kHidden, xr[r].data(), xr[r].size() * 2U);
    }
    const KvPairStats s0 = pair.stats(0), s1 = pair.stats(1);
    std::printf("  info  world 2%s: %llu + %llu exchanges, %.1f MB each way, finish wait %.3f / %.3f s\n", swap ? " (swapped)" : "",
        static_cast<unsigned long long>(s0.exchanges), static_cast<unsigned long long>(s1.exchanges), static_cast<double>(s0.bytes) / 1e6, s0.finish_wait_s,
        s1.finish_wait_s);
    return out;
}

// A fast row-local stand-in linear for the failure tests: y[r][c] = x[r][c % k] (exact copies; optional delay / throw).
struct FastLinear final : block::LinearOp {
    std::chrono::milliseconds delay{0};
    int throw_after = -1;  // throw on that call (counting from 0), -1 never
    int calls = 0;
    void run(const block::QLinear& w, const std::uint16_t* x, std::uint32_t m, std::uint16_t* y) override {
        if (throw_after >= 0 && calls++ == throw_after) throw std::runtime_error("injected rank failure");
        if (delay.count()) std::this_thread::sleep_for(delay);
        for (std::uint32_t r = 0; r < m; ++r)
            for (std::uint32_t c = 0; c < w.n; ++c) y[std::size_t{r} * w.n + c] = x[std::size_t{r} * w.k + c % w.k];
    }
};

struct PairOutcome {
    bool ok[2] = {false, false};
    std::string why[2];
    double seconds = 0;
    std::vector<std::uint16_t> x;  // rank rows concatenated (when both ok)
};
// Two ranks in threads; `setup` gets the pair and each rank's linear before the run (failure injection), `during`
// runs on a third thread while the ranks are working (e.g. an external cancellation).
template <class Setup, class During>
PairOutcome run_pair_inject(block::BlockSource& blocks, std::uint32_t nb, const ForwardTables& t, std::uint32_t S, const std::vector<std::uint16_t>& x,
    Setup&& setup, During&& during) {
    const Split sp = Split::of(S);
    MemoryKvPair pair(std::chrono::microseconds(2000), 4, std::chrono::milliseconds(15000));
    FastLinear lin[2];
    setup(pair, lin);
    SharedBlockSource shared(blocks, 3);
    PairOutcome out;
    std::vector<std::uint16_t> xr[2];
    const double t0 = now_s();
    std::thread th[2];
    for (unsigned r = 0; r < 2; ++r)
        th[r] = std::thread([&, r] {
            const Role role{r, 2, false};
            const std::size_t off = std::size_t{role.row0(sp)} * block::kHidden;
            xr[r].assign(x.begin() + static_cast<std::ptrdiff_t>(off),
                x.begin() + static_cast<std::ptrdiff_t>(off + std::size_t{role.rows(sp)} * block::kHidden));
            try {
                dist_blocks_host(shared, nb, t, sp, role, xr[r], lin[r], &pair.endpoint(r), nullptr);
                out.ok[r] = true;
            } catch (const std::exception& e) {
                out.why[r] = e.what();
                pair.endpoint(r).abort(e.what());  // the rank's failure handler: stop the pair so the peer fails too
            }
        });
    during(pair);
    for (auto& h : th) h.join();
    out.seconds = now_s() - t0;
    if (out.ok[0] && out.ok[1]) {
        out.x.resize(x.size());
        for (unsigned r = 0; r < 2; ++r) std::memcpy(out.x.data() + std::size_t{Role{r, 2, false}.row0(sp)} * block::kHidden, xr[r].data(), xr[r].size() * 2U);
    }
    return out;
}

void failure_tests(block::BlockSource& blocks, const ForwardTables& t, std::uint32_t S, const std::vector<std::uint16_t>& x) {
    std::printf("[failure injection through D1: host driver, in-process pair, 4 blocks, S=%u]\n", S);
    const std::uint32_t nb = 4;
    auto none = [](MemoryKvPair&) {};
    auto both_failed = [](const PairOutcome& o) { return !o.ok[0] && !o.ok[1]; };
    // reference: the same stand-in linear on one rank (world 1)
    FastLinear ref_lin;
    std::vector<std::uint16_t> ref = x;
    dist_blocks_host(blocks, nb, t, Split::of(S), Role{0, 1, false}, ref, ref_lin, nullptr, nullptr);
    {  // delayed reader: rank 0 slow (every linear sleeps), rank 1 races ahead: RX double buffers + one exchange in
       // flight must keep the result exact
        const PairOutcome o = run_pair_inject(blocks, nb, t, S, x, [](MemoryKvPair&, FastLinear* l) { l[0].delay = std::chrono::milliseconds(40); }, none);
        check(o.ok[0] && o.ok[1] && o.x == ref, "delayed reader (rank 0 slow): both ranks finish and rank0|rank1 == world 1 bitwise");
    }
    {  // corruption on the wire after the sender's CRC: the receiver rejects it, the pair stops, both ranks fail
        const PairOutcome o = run_pair_inject(blocks, nb, t, S, x, [](MemoryKvPair& p, FastLinear*) { p.corrupt(1, 2); }, none);
        check(both_failed(o) && o.seconds < 14.0, "corrupted block-2 message from rank 1: both ranks fail (" + o.why[0] + " | " + o.why[1] + ")");
    }
    {  // a rank dies mid-forward (the peer-close case): the other rank must not hang
        const PairOutcome o = run_pair_inject(blocks, nb, t, S, x, [](MemoryKvPair&, FastLinear* l) { l[1].throw_after = 5; }, none);
        check(both_failed(o) && o.seconds < 14.0, "rank 1 fails in block 1: both ranks fail without waiting for the timeout (" + o.why[0] + " | " +
            o.why[1] + ")");
    }
    {  // external cancellation while both ranks are in flight
        const PairOutcome o = run_pair_inject(blocks, nb, t, S, x, [](MemoryKvPair&, FastLinear* l) { l[0].delay = l[1].delay = std::chrono::milliseconds(20); },
            [](MemoryKvPair& p) {
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
                p.endpoint(0).abort("external cancellation");
            });
        check(both_failed(o), "external cancellation mid-forward: both ranks fail (" + o.why[0] + " | " + o.why[1] + ")");
    }
    {  // peer silent (never starts): finish times out on the waiting rank, which then stops the pair
        MemoryKvPair pair(std::chrono::microseconds(0), 1, std::chrono::milliseconds(1500));
        FastLinear lin;
        const Split sp = Split::of(S);
        std::vector<std::uint16_t> xr(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(std::size_t{sp.rows(0)} * block::kHidden));
        std::string why;
        const double t0 = now_s();
        const bool threw = throws([&] { dist_blocks_host(blocks, nb, t, sp, Role{0, 2, false}, xr, lin, &pair.endpoint(0), nullptr); }, &why);
        check(threw && now_s() - t0 < 10.0, "silent peer: the waiting rank fails at the exchange deadline (" + why + ")");
    }
}

// Runs all four forms and gates them against each other.
void g3_case(const std::string& name, block::BlockSource& blocks, std::uint32_t nb, const ForwardTables& t, std::uint32_t S,
    const std::vector<std::uint16_t>& x0, block::LinearOp& lin, std::chrono::microseconds delay) {
    const Split sp = Split::of(S);
    std::printf("[%s] S=%u cut=%u (key blocks %u + %u) blocks=%u message %.2f MB\n", name.c_str(), S, sp.cut, sp.rows(0), sp.rows(1), nb,
        static_cast<double>(sp.message_bytes()) / 1e6);
    double t0 = now_s();
    std::vector<std::array<std::uint64_t, 2>> href;
    const auto ref = run_reference(blocks, nb, t, S, x0, lin, &href);
    std::printf("  info  1-node block_forward_host + split attention %.1f s\n", now_s() - t0);
    t0 = now_s();
    DistTrace t1;
    const auto w1 = run_world1(blocks, nb, t, S, x0, lin, &t1);
    std::printf("  info  world 1 %.1f s\n", now_s() - t0);
    DistTrace t2[2], t3[2];
    t0 = now_s();
    const auto w2 = run_world2(blocks, nb, t, S, x0, lin, false, t2, delay);
    std::printf("  info  world 2 %.1f s\n", now_s() - t0);
    t0 = now_s();
    const auto w3 = run_world2(blocks, nb, t, S, x0, lin, true, t3, delay);
    std::printf("  info  world 2 swapped %.1f s\n", now_s() - t0);
    std::size_t bad1 = 0, bad2 = 0, bad3 = 0;
    std::printf("   block  key-block-0 hash      key-block-1 hash      1n  w1  w2  w2swap\n");
    for (std::uint32_t i = 0; i < nb; ++i) {
        const bool e1 = t1.hash[i] == href[i];
        // rank r owns block r (w2) / block 1 - r (swapped)
        const bool e2 = t2[0].hash[i][0] == href[i][0] && t2[1].hash[i][1] == href[i][1];
        const bool e3 = t3[1].hash[i][0] == href[i][0] && t3[0].hash[i][1] == href[i][1];
        bad1 += !e1; bad2 += !e2; bad3 += !e3;
        if (i < 4 || i + 2 >= nb || !e1 || !e2 || !e3)
            std::printf("   %5u  %016llx  %016llx  ref %s %s %s\n", i, static_cast<unsigned long long>(href[i][0]),
                static_cast<unsigned long long>(href[i][1]), e1 ? "==" : "!=", e2 ? "==" : "!=", e3 ? "==" : "!=");
    }
    check(bad1 == 0, name + ": world 1 == 1-node reference on every block (both key blocks)");
    check(bad2 == 0, name + ": world 2, rank 0 (block 0) and rank 1 (block 1) hashes == 1-node on every block");
    check(bad3 == 0, name + ": world 2 swapped (rank 0 owns block 1) == 1-node on every block");
    check(w1 == ref && w2 == ref && w3 == ref, name + ": final residual rank0|rank1 concatenated == 1-node, bitwise (all forms)");
    std::uint64_t full = fnv1a64(ref.data(), ref.size() * 2U);
    std::printf("  info  %s final residual fnv %016llx\n", name.c_str(), static_cast<unsigned long long>(full));
}

// ------------------------------------------------------------------------------------------------ unit
// Attention operand admission (shared verdict of both ranks) and the rank tool's --forms rule.
void admission_tests() {
    std::printf("[admission + forms]\n");
    auto fb = [](float v) {
        unsigned b;
        std::memcpy(&b, &v, 4);
        return b;
    };
    const unsigned nan_bits = 0x7fc00000U, neg_nan_bits = 0xffc00000U, inf_bits = 0x7f800000U;
    // 3 blocks, ||q|| = ||k|| = 100: bound 100 * 100 * log2(e) / sqrt(128) = 1275.3; |V| 4.0 (bf16 0x4080)
    const std::vector<unsigned> good{fb(100.F), fb(100.F), 0x4080U, fb(100.F), fb(100.F), 0x4080U, fb(100.F), fb(100.F), 0x4080U};
    auto with = [&](std::size_t at, unsigned v) {
        std::vector<unsigned> m = good;
        m[at] = v;
        return m;
    };
    auto verdict = [&](const std::vector<unsigned>& r0, const std::vector<unsigned>& r1, std::uint64_t b0, std::uint64_t b1, std::string* why) {
        try {
            const AdmissionResult r = admit({&r0, &r1}, {b0, b1}, 3);
            if (why) *why = "admitted " + std::to_string(r.worst);
            return 0;
        } catch (const ContractViolation& e) {
            if (why) *why = e.what();
            return 1;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            return 2;
        }
    };
    std::string why;
    int v = verdict(good, good, 0, 0, &why);
    check(v == 0, "in-contract operands admitted (" + why + ")");
    {
        const AdmissionResult r = admit({&good}, {0}, 3);
        check(std::fabs(r.worst - 100.0 * 100.0 * 1.4426950408889634 / std::sqrt(128.0)) < 1e-9 && r.v_max == 4.0, "bound and max |V| values (one rank)");
    }
    struct Bad {
        std::size_t at;
        unsigned v;
        int rank;
        const char* expect;
        const char* name;
    };
    const Bad bad[] = {
        {3, nan_bits, 0, "non-finite ||q|| on block 1 (rank 0)", "NaN ||q|| on rank 0"},
        {3, nan_bits, 1, "non-finite ||q|| on block 1 (rank 1)", "NaN ||q|| on the PEER (would be dropped by std::max(x, NaN))"},
        {7, neg_nan_bits, 1, "non-finite ||k|| on block 2 (rank 1)", "negative NaN ||k|| on the peer"},
        {1, inf_bits, 0, "non-finite ||k|| on block 0 (rank 0)", "Inf ||k||"},
        {0, fb(-1.F), 0, "non-finite ||q|| on block 0 (rank 0)", "negative norm"},
        {2, 0x7fc0U, 1, "non-finite V on block 0 (rank 1)", "NaN V"},
        {5, 0xff80U, 0, "non-finite V on block 1 (rank 0)", "-Inf V"},
        {8, 0x7200U, 1, "> 2^100 on block 2 (rank 1)", "|V| = 2^101"},
        {8, 0xf181U, 0, "> 2^100 on block 2 (rank 0)", "V = -(2^100 + 1 ulp)"},
        {0, fb(2000.F), 1, "outside the contract: block 0", "score bound over 2^14 (2000 x 100 x log2 e / sqrt 128 = 25,506)"},
    };
    for (const Bad& b : bad) {
        const std::vector<unsigned> m = with(b.at, b.v);
        v = b.rank == 0 ? verdict(m, good, 0, 0, &why) : verdict(good, m, 0, 0, &why);
        check(v == 1 && why.find(b.expect) != std::string::npos, std::string("rejected: ") + b.name + " (" + why + ")");
    }
    v = verdict(with(8, 0x7180U), good, 0, 0, &why);
    check(v == 0, "|V| = 2^100 exactly is admitted (" + why + ")");
    v = verdict(good, good, 0, 1, &why);
    check(v == 1 && why.find("1 bound rows") != std::string::npos, "peer bound rows 1 rejected (" + why + ")");
    v = verdict(good, std::vector<unsigned>(2), 0, 0, &why);
    check(v == 2, "measurement size mismatch refused (" + why + ")");
    // --forms
    using V = std::vector<std::string>;
    check(valid_forms(V{"w1"}) && valid_forms(V{"w1", "w2"}) && valid_forms(V{"w1", "w2swap", "w2"}), "--forms accepted: w1 | w1,w2 | w1,w2swap,w2");
    const V rejected[] = {V{}, V{"w2"}, V{"w2", "w1"}, V{"w1", "w1"}, V{"w1", "w2", "w2"}, V{"w1", "w3"}, V{"w1", ""}, V{"W1"}};
    int n_rej = 0;
    for (const V& f : rejected) n_rej += !valid_forms(f);
    check(n_rej == 8, "--forms rejected: empty, w2, w2,w1, w1,w1, w1,w2,w2, w1,w3, w1,<empty>, W1 (" + std::to_string(n_rej) + "/8)");
}

void unit_tests() {
    std::printf("[unit]\n");
    admission_tests();
    check(crc32c_sw("123456789", 9) == 0xE3069283U, "crc32c check value");
    {
        const Split a = Split::of(400), b = Split::of(37910);
        check(a.cut == 192 && a.rows(1) == 208 && a.vt_pitch() == 256 && a.message_bytes() % 64 == 0, "split S=400: cut 192, 208 rows, pitch 256");
        check(b.cut == 18944 && b.rows(1) == 18966 && b.message_bytes() % 64 == 0, "split S=37910: cut 18944 / 18966 (the attention bench shape)");
        std::printf("  info  S=37910 message %.1f MB (K %.1f + V^T %.1f)\n", static_cast<double>(b.message_bytes()) / 1e6, static_cast<double>(b.k_bytes()) / 1e6, static_cast<double>(b.vt_bytes()) / 1e6);
        check(throws([] { (void)Split::of(127); }), "split rejects S < 128");
        const Role r0{0, 2, false}, r1{1, 2, true};
        check(r0.block() == 0 && r1.block() == 0 && r1.row0(a) == 0 && Role{0, 2, true}.row0(a) == 192, "roles and swap");
    }
    {
        // the token-major partial == the attention package's partial_row_f32 (bit for bit), merge identical
        const unsigned R = 6, K = 200;
        const attn::HostQkv x = attn::make_qkv(kHeads, R, K, 77, 1.5F);
        std::vector<std::uint16_t> q(std::size_t{R} * kInner), k(std::size_t{K} * kInner), vt(std::size_t{kInner} * K);
        for (unsigned h = 0; h < kHeads; ++h)
            for (unsigned d = 0; d < kHeadDim; ++d) {
                for (unsigned r = 0; r < R; ++r) q[std::size_t{r} * kInner + h * kHeadDim + d] = x.qv(h, r, d);
                for (unsigned j = 0; j < K; ++j) {
                    k[std::size_t{j} * kInner + h * kHeadDim + d] = x.kv(h, j, d);
                    vt[(std::size_t{h} * kHeadDim + d) * K + j] = x.vv(h, j, d);
                }
            }
        bool same = true;
        for (unsigned h = 0; h < kHeads; h += 11)
            for (unsigned r = 0; r < R; ++r) {
                const unsigned cuts[3] = {0, 128, K};
                for (unsigned bl = 0; bl < 2; ++bl) {
                    const attn::Partial a = attn::partial_row_f32(x, h, r, cuts[bl], cuts[bl + 1]);
                    const attn::Partial b = host_partial(q.data() + std::size_t{r} * kInner + h * kHeadDim,
                        KeyView{k.data() + std::size_t{cuts[bl]} * kInner, vt.data() + cuts[bl], K, cuts[bl + 1] - cuts[bl]}, h);
                    same &= std::memcmp(&a, &b, sizeof a) == 0;
                }
            }
        check(same, "host_partial (token-major) == attn::partial_row_f32 bitwise (mc, l, o), 2 key blocks");
    }
    {
        // exchange: delivery, progress without finish, hazards
        std::vector<std::byte> tx[2], rx[2];  // declared before the pair: its threads may still read them at teardown
        MemoryKvPair pair(std::chrono::microseconds(20000), 4);
        for (unsigned r = 0; r < 2; ++r) {
            tx[r].resize(1 << 20);
            rx[r].resize(1 << 20);
            for (std::size_t i = 0; i < tx[r].size(); ++i) tx[r][i] = static_cast<std::byte>((i * 131 + r * 7) & 0xFF);
        }
        pair.endpoint(0).start(tx[0].data(), tx[0].size(), 5);
        pair.endpoint(1).start(tx[1].data(), tx[1].size(), 5);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        check(pair.arrived(0) && pair.arrived(1), "start progresses without finish (both messages landed before finish)");
        pair.endpoint(0).finish(rx[0].data());
        pair.endpoint(1).finish(rx[1].data());
        check(rx[0] == tx[1] && rx[1] == tx[0], "exchange delivers the peer's bytes");
        check(throws([&] { pair.endpoint(0).finish(rx[0].data()); }), "finish without start is rejected");
        pair.endpoint(0).start(tx[0].data(), tx[0].size(), 6);
        check(throws([&] { pair.endpoint(0).start(tx[0].data(), tx[0].size(), 6); }), "a second start while one is in flight is rejected");
    }
    {
        std::vector<std::byte> tx0(1 << 16, std::byte{1}), tx1(1 << 16, std::byte{2}), rx(1 << 16);
        MemoryKvPair pair(std::chrono::microseconds(40000), 4, std::chrono::milliseconds(3000));
        pair.endpoint(0).start(tx0.data(), tx0.size(), 1);
        pair.endpoint(1).start(tx1.data(), tx1.size(), 1);
        tx0[100] = std::byte{9};  // the hazard: TX overwritten while in flight
        std::string w0, w1;
        const bool a = throws([&] { pair.endpoint(0).finish(rx.data()); }, &w0);
        const bool b = throws([&] { pair.endpoint(1).finish(rx.data()); }, &w1);
        check(a && b, "TX changed during the flight fails both ranks (" + w0 + " | " + w1 + ")");
    }
    {
        std::vector<std::byte> tx(4096, std::byte{3}), rx(4096);
        MemoryKvPair pair(std::chrono::microseconds(0), 1, std::chrono::milliseconds(3000));
        pair.endpoint(0).start(tx.data(), tx.size(), 1);
        pair.endpoint(1).start(tx.data(), tx.size(), 2);
        std::string w;
        const bool t = throws([&] { pair.endpoint(0).finish(rx.data()); }, &w);
        check(t, "mismatched block (layer) tag is detected (" + w + ")");
    }
    {
        std::vector<std::byte> tx(4096), rx(4096);
        MemoryKvPair pair(std::chrono::microseconds(0), 1, std::chrono::milliseconds(500));
        pair.endpoint(0).start(tx.data(), tx.size(), 1);
        std::string w;
        const double t0 = now_s();
        const bool t = throws([&] { pair.endpoint(0).finish(rx.data()); }, &w);
        check(t && now_s() - t0 < 5.0, "a silent peer times out (" + w + ")");
    }
    {
        // the whole schedule on random weights (no checkpoint), S = 300: key blocks 128 + 172
        const std::uint32_t S = 300, nb = 2, basis = 4;
        RandomBlocks blocks(nb, basis);
        ForwardTables t;
        t.n_t = 2;
        std::mt19937_64 g(99);
        std::normal_distribution<float> nd(0.0F, 1.0F);
        t.t_emb.resize(std::size_t{t.n_t} * basis);
        for (auto& e : t.t_emb) e = nd(g);
        std::vector<float> ang(std::size_t{S} * block::kRotDim);
        for (std::uint32_t r = 0; r < S; ++r)
            for (unsigned i = 0; i < block::kRotDim; ++i) ang[std::size_t{r} * block::kRotDim + i] = 0.05F * static_cast<float>(r) * (1.0F + static_cast<float>(i % 16)) / 16.0F;
        t.rope = block::rope_table(ang);
        t.mod_row.resize(S);
        for (std::uint32_t r = 0; r < S; ++r) t.mod_row[r] = r < 40 ? 1U : (r < 60 ? 5U : 3U);
        std::vector<std::uint16_t> x(std::size_t{S} * block::kHidden);
        for (auto& e : x) e = block::tob(nd(g));
        block::HostLinear lin;
        g3_case("random-weights S=300", blocks, nb, t, S, x, lin, std::chrono::microseconds(30000));
        RandomBlocks fblocks(4, basis);
        failure_tests(fblocks, t, S, x);
    }
}

// ------------------------------------------------------------------------------------------------ fixtures
void fixture_case(const std::string& set, std::uint32_t nb_default) {
    const std::string root = env_or("H3_FIXTURES", ""), ckpt = env_or("H3_CKPT", "");
    if (root.empty() || ckpt.empty()) {
        std::printf("[%s] SKIPPED (set H3_FIXTURES and H3_CKPT)\n", set.c_str());
        ++g_skipped;
        return;
    }
    const fx::Manifest m = fx::Manifest::load(root + "/" + set);
    const auto& sig = m.meta.at("layout").at("signature").arr();
    const block::PackedLayout L = block::PackedLayout::build(static_cast<std::uint32_t>(sig[0].num()), static_cast<std::uint32_t>(sig[1].num()),
        static_cast<std::uint32_t>(sig[2].num()), static_cast<std::uint32_t>(sig[3].num()), static_cast<std::uint32_t>(sig[4].num()));
    auto rd = [&](const std::string& n) { return m.read(n); };
    const fx::Tensor tt = rd("in/text_token_tags");
    std::vector<std::int64_t> tags(tt.bytes.size() / 8);
    std::memcpy(tags.data(), tt.bytes.data(), tags.size() * 8);
    bool all_text = true;
    for (auto v : tags) all_text &= v == 1;
    if (all_text) tags.clear();
    const fx::Tensor sg = rd("sampler/sigmas_simple20");
    float sigma0;
    std::memcpy(&sigma0, sg.bytes.data(), 4);
    const block::ForwardPlan p = block::ForwardPlan::build(L, sigma0, tags);
    const fx::Tensor xin = rd("f0/blocks/in_00");
    std::vector<std::uint16_t> x(xin.bytes.size() / 2);
    std::memcpy(x.data(), xin.bytes.data(), x.size() * 2);
    if (x.size() != std::size_t{L.seq_len} * block::kHidden) throw std::runtime_error("fixture: in_00 size");
    block::SafeTensors st(ckpt);
    const block::GlobalWeights g = block::load_global(st);
    block::FileBlockSource blocks(st, g.basis, false);
    const ForwardTables t = ForwardTables::build(g, L, p);
    const std::uint32_t nb = static_cast<std::uint32_t>(std::stoul(env_or(set == "dit_t2va_small" ? "H3_DIST_BLOCKS" : "H3_DIST_MID_BLOCKS",
        std::to_string(nb_default).c_str())));
    block::HostLinear lin;
    g3_case(set + " f0", blocks, std::min(nb, blocks.num_blocks()), t, L.seq_len, x, lin, std::chrono::microseconds(0));
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> modes;
    for (int i = 1; i < argc; ++i) modes.emplace_back(argv[i]);
    if (modes.empty()) modes.emplace_back("unit");
    try {
        for (const auto& md : modes) {
            if (md == "unit") unit_tests();
            else if (md == "forward") fixture_case("dit_t2va_small", 50);
            else if (md == "mid") fixture_case("dit_t2va_mid", 4);
            else { std::printf("unknown mode %s\n", md.c_str()); return 2; }
        }
    } catch (const std::exception& e) {
        std::printf("  FAIL exception: %s\n", e.what());
        return 1;
    }
    if (g_fail) {
        std::printf("FAILED (%d failures)\n", g_fail);
        return 1;
    }
    if (g_skipped) {  // absent qualification data must never look like a pass
        std::printf("SKIPPED (%d requested fixture set%s without assets; 0 failures in what ran)\n", g_skipped, g_skipped == 1 ? "" : "s");
        return 77;
    }
    std::printf("PASSED (0 failures)\n");
    return 0;
}
