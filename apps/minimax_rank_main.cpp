// strix-minimax-rank: one rank of the MiniMax M3.1 Flash TP2 engine (both ranks run the same command line).
// The DS bench roles in one binary: teacher-forced gates, single-user sessions (TTFT, prefill tok/s, decode tok/s) and
// batched decode N = 1..8 with the per-row "batch == alone" check.
//
// usage: strix-minimax-rank MODEL_DIR RANK TRANSPORT OUT_DIR [options]
//   TRANSPORT: usb4-auto | usb4gpu (DS production: GPU-direct streams + host path, size routes) | tcp-listen:PORT |
//              tcp-connect:HOST:PORT
//   STRIX_MM_STAGING=fused: exchanged partials written / read in the communicator's pinned buffers (host-path routes)
// STRIX_MM_PFX=off|load|on (default on): prefill exchange module (src/minimax/pfx) constructed
//                                  and switched off (load) / on; STRIX_MM_PF_INTERLEAVE (default on; =0 serial, =comm
//                                  communicator even with the module on) interleaves the max_rows chunks of a prompt;
//                                  STRIX_MM_STATE_FNV=0 skips the state hash prints
//   --gate ORACLE_DIR --gate-layers N [--gold DIR]  residual after layer N-1 over ORACLE_DIR/tokens.u32 (G4 smoke form:
//                                                     same fnv as strix-hip-minimax-g4; metrics vs DIR/h{N-1}.f32)
//   --tf ORACLE_DIR [--tf-write]                     full teacher-forced logits of ORACLE_DIR/rows.u32 (G4 form): fnv,
//                                                     top-1 vs the real next token; --tf-write saves logits_pair.f32
//   --session NAME:PROMPT.u32:STEPS                  prefill + STEPS greedy decode steps (repeatable)
//   --profile NAME:PROMPT.u32:STEPS                  the same with synchronised per-phase times (slower)
//   --batch PROMPT.u32:STEPS:N1,N2,...               batched decode (rows = prefixes of the prompt), checked vs alone
//   --verify-trace NAME:PROMPT.u32:REPS:R1,R2,...    verify-shaped steps (one slot, R rows) timed by R (xfer on/off
//                                                     ABA) + per-phase GPU trace and distinct routed experts per layer
//   --ab FLAG:PROMPT.u32:BLOCK                       ABA x3 of a runtime flag (staging | shared | attn | linears | xfer | attnpf | index | idxdec | idxattn | idxconc | xbwait | xbinplace | xbboth | xbprep | xbpub | xbwarm | xbwarmo | moegrid | moepipe | xbilp | xbset | pfx | pfil | pfxip | pfcore | pfattn4 | pfmsa | pfmsarework | mtfit | core8 | pfpipe | pftab | pfgufuse | pfrouter | pfrouterrows | core8v2 | pfidxconc | pfkvtile | pftiled | pfidx | pftxuc | pfall): TTFT and decode blocks
//                                                    (pfidxconc: 0 vs STRIX_MM_AB_IDXCONC=1|2, default 2; 7 hashed prefills)
//                                                    (pfkvtile: off vs STRIX_MM_AB_KVTILE=auto|<index>, default auto; 7 hashed prefills)
//                                                    (pftxuc: PFX copy-to-host-coherent vs producer-to-uncached TX; 7 hashed prefills)
//   --xb-check PROMPT.u32:K:N[:v]                exchange boundary check: decode logits of the 4 (wait, inplace) configurations bitwise == 00 (N slots, or :v one N-row verify window)
//                                                    (pfcore: prefill MoE WMMA core, off = lib, on = STRIX_MM_AB_PFCORE, default 3 = v4)
//                                                    (pfgufuse: single-split gate/up fusion off/on; 7 prefills with state and final-row hashes)
//                                                    (pftab: pinned prefill MoE tables off/on; 7 prefills with state and final-row hashes)
//                                                    (mtfit: v4 row tiles, off/on; 7 prefills with state and final-row hashes)
//                                                    (pfpipe: production vs STRIX_MM_AB_PIPE=1|2, default 2; 7 hashed prefills)
//                                                    (core8: v4 MT8, 0 vs STRIX_MM_AB_CORE8, default 1; forces mtfit; 7 hashed prefills)
//   --spec NAME:PROMPT.u32:STEPS:POLICY              speculative decode with the DSpark draft (POLICY fixed:K | auto);
//                                                     auto: structural-v1 admission, prose stays plain; --dspark-prior unqualified-v1 restores plain; auto:legacy restores the former controller
//                                                     tokens compared with the --session of the same NAME (exactness)
//   --dspark DIR                                     draft checkpoint (default MODEL_DIR/dspark)
//   --dspark-exl3 DIR                                draft linears from an EXL3 requant (strix-trellis gemv), rest from --dspark
//   --dspark-head DIR (repeatable)                   draft-only head: EXL3 requant of the lm_head (ref/quant_head.py); the target's
//                                                    head stays for decode / verify (STRIX_MM_DRAFT_HEAD=lib|sched|sched:V;
//                                                    STRIX_MM_DRAFT_HEAD_AB: per prompt the spec runs rotate target,
//                                                    head 0, head 1, ...; otherwise the last one attached)
//   --decode-check PROMPT.u32:K                      decode steps (true tokens) vs teacher-forced logits, last K rows
//   --deadline SECONDS                               skip later parts once rank 0 has run this long
//   --slots S --capacity C --max-rows R              engine sizes (defaults 8 / 8256 / 4096)
// --offload NAME:PROMPT.u32:STEPS RTX-prefill reference form: prefill rows
//                                                     [0, P-1), export this rank's state rows to
//                                                     OUT_DIR/state-NAME-rank<r>.bin, then the last prompt row as a decode
//                                                     step + STEPS greedy steps
//   --offload-import NAME:PROMPT.u32:STATE_PREFIX:STEPS  the same from STATE_PREFIX-rank<r>.bin (an RTX PRO prefill)
//                                                     instead of the local prefill; tokens compared with --offload NAME
//   --offload-listen NAME:PROMPT.u32:PORT:STEPS:COUNT  live hand-off from mm-rtx-prefill --send: rank r listens on
//                                                     PORT + r; per request the layers arrive as they finish and are
//                                                     imported at once; rank 0 returns the last row's token to the
//                                                     sender (its TTFT clock); repeated COUNT times (one connection each)
#include "strix/dist/gpu_direct.hpp"
#include "strix/minimax/engine.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/bp_codec.hpp"
#include "minimax/moe/mmx_prefill.hpp"
#include "minimax/xfer/minimax_xfer.hpp"
#include "minimax/pfx/pf_exchange.hpp"
#include "hip_minimax_dspark.hpp"
#include "minimax/dspark/minimax_auto.hpp"
#include "minimax/serve/minimax_tokenizer.hpp"
#include "markov_cand.hpp"
#include "minimax/session/runtime.hpp"
#include "strix/minimax/session_startup.hpp"
#include "minimax_ngram.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace mm = strix::minimax::engine;
namespace ds = strix::minimax::dspark;
using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

constexpr std::size_t kHidden = 6144, kVocab = 200064;

std::uint64_t fnv(const void* p, std::size_t bytes, std::uint64_t h = 14695981039346656037ULL) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}

// The transport shared in turn by the communicator and the exchange module (shared exchange form).
class Borrowed final : public strix::dist::Transport {
public:
    explicit Borrowed(strix::dist::Transport& t) : t_(t) {}
    void send(std::span<const std::byte> m) override { t_.send(m); }
    void receive(std::span<std::byte> m) override { t_.receive(m); }
    void exchange(std::span<const std::byte> o, std::span<std::byte> i) override { t_.exchange(o, i); }
    [[nodiscard]] std::string name() const override { return t_.name(); }
    void abort(const std::string& why) override { t_.abort(why); }

private:
    strix::dist::Transport& t_;
};

// usb4gpu: the device exchange (GPU-direct streams) built next to the host-path control transport (as the DS apps).
std::unique_ptr<strix::dist::DeviceExchange> gpu_direct;

std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, std::string& receipt) {
    const std::chrono::seconds limit(180);
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon), rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "tcp-listen") return strix::dist::tcp_listen(static_cast<std::uint16_t>(std::stoul(rest)), limit);
    if (kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return strix::dist::tcp_connect(rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), limit);
    }
    if (kind == "usb4-auto") {
        strix::dist::Usb4AutoOptions o;
        o.minimax_compat = true;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        receipt = topology.receipt();
        return t;
    }
    if (kind == "usb4gpu") {
        strix::dist::Usb4AutoOptions o;
        o.minimax_compat = true;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        receipt = topology.receipt();
        std::string why_direct, direct_receipt;
        if (strix::dist::usb4_gpu_direct_streams(o, *t, &why_direct) == 0U) {  // both ranks agree: host path only
            receipt += "; gpu-direct unavailable (" + why_direct + "): host path only";
            return t;
        }
        gpu_direct = strix::dist::usb4_gpu_direct(o, topology, *t, {}, &direct_receipt);
        receipt += "; usb4 " + direct_receipt;
        return t;
    }
    throw std::invalid_argument("unknown transport " + spec);
}

std::vector<std::uint32_t> read_u32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("read " + path);
    std::vector<std::uint32_t> v(static_cast<std::size_t>(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    return v;
}
std::vector<float> read_f32(const std::string& path, std::size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
    if (!f) throw std::runtime_error("read " + path);
    return v;
}

struct Metrics {
    double rel_l2, tok_p50, tok_p99, tok_max;
};
Metrics compare(const std::vector<float>& a, const std::vector<float>& ref, std::size_t t, std::size_t d) {  // as G4
    double num = 0, den = 0;
    std::vector<double> tok(t);
    for (std::size_t i = 0; i < t; ++i) {
        double n2 = 0, d2 = 0;
        for (std::size_t j = 0; j < d; ++j) {
            const double x = a[i * d + j], r = ref[i * d + j];
            n2 += (x - r) * (x - r);
            d2 += r * r;
        }
        num += n2;
        den += d2;
        tok[i] = std::sqrt(n2 / std::max(d2, 1e-300));
    }
    std::sort(tok.begin(), tok.end());
    return {std::sqrt(num / den), tok[t / 2], tok[std::min(t - 1, t * 99 / 100)], tok[t - 1]};
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

// Synchronised per-phase times between two snapshots of the engine's cumulative profile, divided by `per` and scaled
// by `scale` (prefill: s per prompt; decode: ms per token).
void print_profile(const mm::Engine& eng, const char* what, const mm::Profile& a, const mm::Profile& b, double per, double scale,
                   const char* unit) {
    if (b.ffn == a.ffn) return;  // profiling was off
    auto d = [&](double mm::Profile::*f) { return (b.*f - a.*f) / per * scale; };
    const double total = d(&mm::Profile::embed) + d(&mm::Profile::attn_proj) + d(&mm::Profile::msa) + d(&mm::Profile::o_proj) +
                         d(&mm::Profile::exchange) + d(&mm::Profile::ffn) + d(&mm::Profile::route_host) + d(&mm::Profile::head);
    const double total2 = total + d(&mm::Profile::post) + d(&mm::Profile::dense);
    std::printf("rank %u profile %s (%s): embed %.3f attn_proj %.3f msa %.3f o_proj %.3f exchange %.3f post %.3f ffn(moe) %.3f "
                "dense %.3f route_host %.3f head %.3f | sum %.3f\n", eng.rank(), what, unit, d(&mm::Profile::embed),
                d(&mm::Profile::attn_proj), d(&mm::Profile::msa), d(&mm::Profile::o_proj), d(&mm::Profile::exchange),
                d(&mm::Profile::post), d(&mm::Profile::ffn), d(&mm::Profile::dense), d(&mm::Profile::route_host),
                d(&mm::Profile::head), total2);
    std::uint64_t pairs = 0;
    for (std::size_t i = 0; i < 8; ++i) pairs += b.rows_per_expert[i] - a.rows_per_expert[i];
    if (pairs != 0) {
        static const char* const kB[8] = {"0", "1-8", "9-16", "17-32", "33-64", "65-128", "129-256", ">256"};
        std::printf("rank %u profile %s: (MoE layer, expert) pairs by routed rows:", eng.rank(), what);
        for (std::size_t i = 0; i < 8; ++i)
            std::printf(" %s:%llu", kB[i], static_cast<unsigned long long>(b.rows_per_expert[i] - a.rows_per_expert[i]));
        std::printf("\n");
    }
}

// Set when an A/B run's arms differ (tokens, state or final rows): the process exits 4.
bool g_exact_fail = false;
// The prefill exchange module of this process (stats prints), or null.
strix::minimax::pfx::Exchanger* g_pfx = nullptr;
bool state_fnv_on() {
    const char* v = std::getenv("STRIX_MM_STATE_FNV");
    return v == nullptr || v[0] != '0';
}

// Prefill then `steps` greedy steps in slot 0. Returns the generated ids (first = the prefill's token).
std::vector<std::uint32_t> session(mm::Engine& eng, const std::string& name, const std::vector<std::uint32_t>& prompt, int steps,
                                   const std::string& out_dir) {
    const auto st0 = eng.comm_stats();
    const mm::Profile pf0 = eng.profile();
    const auto t0 = Clock::now();
    std::vector<std::uint32_t> gen{eng.prefill(0, prompt, 0)};
    const double ttft = secs(t0, Clock::now());
    const mm::Profile pf1 = eng.profile();
    const auto st1 = eng.comm_stats();
    std::printf("rank %u %s: prompt %zu tokens, TTFT %.3f s, prefill %.1f tok/s (exchange %.3f s, %llu exchanges)\n", eng.rank(),
                name.c_str(), prompt.size(), ttft, static_cast<double>(prompt.size()) / ttft, st1.seconds - st0.seconds,
                static_cast<unsigned long long>(st1.exchanges - st0.exchanges));
    std::printf("rank %u %s: communicator split (s): stage %.3f wire %.3f check %.3f overlap %.3f finish_wait %.3f\n", eng.rank(),
                name.c_str(), st1.stage - st0.stage, st1.wire - st0.wire, st1.check - st0.check, st1.overlap - st0.overlap,
                st1.finish_wait - st0.finish_wait);
    if (g_pfx != nullptr) {
        static strix::minimax::pfx::Stats last{};
        const auto px = g_pfx->stats();
        std::printf("rank %u %s: prefill module (cumulative delta): %llu exchanges, %.1f MB, producer_wait %.3f s, wire %.3f s, tail %.3f s\n",
                    eng.rank(), name.c_str(), static_cast<unsigned long long>(px.exchanges - last.exchanges),
                    static_cast<double>(px.bytes - last.bytes) / 1e6, px.producer_wait - last.producer_wait, px.wire - last.wire,
                    px.tail - last.tail);
        last = px;
    }
    if (state_fnv_on())
        std::printf("rank %u %s: state fnv %016llx (slot 0, %zu positions, K/V/index key, 60 layers), final rows fnv %016llx\n",
                    eng.rank(), name.c_str(), static_cast<unsigned long long>(eng.state_fnv(0, prompt.size())), prompt.size(),
                    static_cast<unsigned long long>(eng.prefill_out_fnv()));
    std::fflush(stdout);
    std::vector<double> dt;
    for (int s = 1; s <= steps; ++s) {
        const auto a = Clock::now();
        const mm::Row row{0, gen.back(), prompt.size() + gen.size() - 1};
        gen.push_back(eng.step({&row, 1})[0]);
        dt.push_back(secs(a, Clock::now()));
    }
    const auto st2 = eng.comm_stats();
    if (steps > 0) {
        const std::size_t from = std::min<std::size_t>(15, dt.size() - 1);
        const std::vector<double> tail(dt.begin() + static_cast<long>(from), dt.end());
        const double med = median(tail);
        std::printf("rank %u %s: decode %d steps at ctx %zu: median(steps %zu-%d) %.2f ms/token = %.2f tok/s; mean %.2f ms; "
                    "min %.2f max %.2f; exchange %.2f ms/token (%llu exchanges/token)\n",
                    eng.rank(), name.c_str(), steps, prompt.size(), from + 1, steps, med * 1e3, 1.0 / med,
                    std::accumulate(dt.begin(), dt.end(), 0.0) / steps * 1e3, *std::min_element(tail.begin(), tail.end()) * 1e3,
                    *std::max_element(tail.begin(), tail.end()) * 1e3, (st2.seconds - st1.seconds) / steps * 1e3,
                    static_cast<unsigned long long>((st2.exchanges - st1.exchanges) / static_cast<std::uint64_t>(steps)));
    }
    std::printf("rank %u %s: tokens fnv %016llx ids:", eng.rank(), name.c_str(), static_cast<unsigned long long>(fnv(gen.data(), gen.size() * 4)));
    for (auto t : gen) std::printf(" %u", t);
    std::printf("\n");
    print_profile(eng, (name + " prefill").c_str(), pf0, pf1, 1.0, 1.0, "s per prompt");
    if (steps > 0) print_profile(eng, (name + " decode").c_str(), pf1, eng.profile(), steps, 1e3, "ms per token");
    std::fflush(stdout);
    std::ofstream o(out_dir + "/gen-" + name + "-rank" + std::to_string(eng.rank()) + ".u32", std::ios::binary);
    o.write(reinterpret_cast<const char*>(gen.data()), static_cast<std::streamsize>(gen.size() * 4));
    return gen;
}

// RTX-prefill hand-off form: rows [0, P-1) come from the local prefill (and are exported) or from
// an imported state blob (Engine::export_rows layout); the last prompt row runs as a decode step, then `steps` greedy
// steps. Returns the generated ids (first = the last row's token). The time to first token covers prefill or import +
// the last row.
std::vector<std::uint32_t> offload_session(mm::Engine& eng, const std::string& name, const std::vector<std::uint32_t>& prompt, int steps,
                                           const std::string& out_dir, const std::string& import_prefix) {
    if (prompt.size() < 2) throw std::invalid_argument("offload: prompt needs >= 2 tokens");
    const std::size_t rows = prompt.size() - 1;
    std::vector<std::uint8_t> blob(rows * mm::Engine::kStateRowBytes);
    const std::string rk = "-rank" + std::to_string(eng.rank()) + ".bin";
    double t_load = 0;
    if (!import_prefix.empty()) {
        std::ifstream f(import_prefix + rk, std::ios::binary);
        if (!f.read(reinterpret_cast<char*>(blob.data()), static_cast<std::streamsize>(blob.size())))
            throw std::runtime_error("offload: cannot read " + import_prefix + rk);
    }
    (void)eng.agree(0);
    const auto t0 = Clock::now();
    if (import_prefix.empty()) {
        (void)eng.prefill(0, std::span<const std::uint32_t>(prompt.data(), rows), 0);
        t_load = secs(t0, Clock::now());
        eng.export_rows(0, 0, rows, blob.data());
    } else {
        eng.import_rows(0, 0, rows, blob.data());
        t_load = secs(t0, Clock::now());
    }
    const auto t1 = Clock::now();
    const mm::Row last{0, prompt.back(), rows};
    std::vector<std::uint32_t> gen{eng.step({&last, 1})[0]};
    const double t_last = secs(t1, Clock::now());
    for (int s = 1; s <= steps; ++s) {
        const mm::Row row{0, gen.back(), prompt.size() + gen.size() - 1};
        gen.push_back(eng.step({&row, 1})[0]);
    }
    std::printf("rank %u offload%s %s: prompt %zu tokens, %s %.3f s, last row %.3f s, state %.1f MB fnv %016llx; tokens fnv %016llx ids:",
                eng.rank(), import_prefix.empty() ? "" : "-import", name.c_str(), prompt.size(), import_prefix.empty() ? "prefill" : "import",
                t_load, t_last, static_cast<double>(blob.size()) / 1e6, static_cast<unsigned long long>(fnv(blob.data(), blob.size())),
                static_cast<unsigned long long>(fnv(gen.data(), gen.size() * 4)));
    for (auto t : gen) std::printf(" %u", t);
    std::printf("\n");
    std::fflush(stdout);
    if (import_prefix.empty()) {
        std::ofstream o(out_dir + "/state-" + name + rk, std::ios::binary);
        o.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
    }
    std::ofstream o(out_dir + "/gen-offload" + (import_prefix.empty() ? "" : "-import") + "-" + name + rk.substr(0, rk.size() - 4) + ".u32",
                    std::ios::binary);
    o.write(reinterpret_cast<const char*>(gen.data()), static_cast<std::streamsize>(gen.size() * 4));
    return gen;
}

// Live hand-off receiver: one TCP connection per request from mm-rtx-prefill --send. Frames:
// {u32 magic 'MMRX', u32 layer, u64 rows, u64 bytes} + bytes (one layer's state block, Engine::import_layer layout);
// layer 0xffffffff = end of the prompt; magic 'MMRB': the bytes are one bp-v1 frame of the block (lossless, XXH64 per
// block, decoded on a worker pool). Rank 0 answers with the last row's token (u32). Returns the generated ids.
void read_full(int fd, void* p, std::size_t n) {
    auto* b = static_cast<std::uint8_t*>(p);
    while (n) {
        const ssize_t r = ::recv(fd, b, n, 0);
        if (r <= 0) throw std::runtime_error("offload-listen: connection closed");
        b += r;
        n -= static_cast<std::size_t>(r);
    }
}
int listen_on(std::uint16_t port) {
    const int s = ::socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 || ::listen(s, 4) != 0) throw std::runtime_error("offload-listen: bind/listen");
    return s;
}
std::vector<std::uint32_t> listen_session(mm::Engine& eng, int lfd, const std::string& name, const std::vector<std::uint32_t>& prompt, int steps,
                                          const std::string& out_dir, unsigned req) {
    const std::size_t rows = prompt.size() - 1;
    std::printf("rank %u offload-listen %s #%u: waiting\n", eng.rank(), name.c_str(), req);
    std::fflush(stdout);
    const int fd = ::accept(lfd, nullptr, nullptr);
    if (fd < 0) throw std::runtime_error("offload-listen: accept");
    const int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    int buf = 32 << 20;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
    std::vector<std::uint8_t> block(rows * 1280), wire;
    static strix::bp_codec::WorkerPool pool(4);
    const auto t0 = Clock::now();
    double t_first = -1, t_import = 0, t_decode = 0;
    std::uint64_t wire_bytes = 0;
    unsigned layers = 0;
    for (;;) {
        std::uint32_t hdr32[2];
        std::uint64_t hdr64[2];
        read_full(fd, hdr32, 8);
        read_full(fd, hdr64, 16);
        const bool bp = hdr32[0] == 0x42524d4dU;
        if (hdr32[0] != 0x58524d4dU && !bp) throw std::runtime_error("offload-listen: bad frame");
        if (hdr32[1] == 0xffffffffU) break;
        if (hdr64[0] != rows || (!bp && hdr64[1] != block.size())) throw std::runtime_error("offload-listen: row count differs from the prompt");
        wire_bytes += hdr64[1];
        if (bp) {
            wire.resize(hdr64[1]);
            read_full(fd, wire.data(), wire.size());
            const auto d0 = Clock::now();
            if (strix::bp_codec::decoded_size(wire) != block.size()) throw std::runtime_error("offload-listen: bp-v1 size differs");
            strix::bp_codec::decode(wire, block, &pool);
            t_decode += secs(d0, Clock::now());
        } else {
            read_full(fd, block.data(), block.size());
        }
        if (t_first < 0) t_first = secs(t0, Clock::now());
        const auto a = Clock::now();
        eng.import_layer(0, hdr32[1], 0, rows, block.data());
        t_import += secs(a, Clock::now());
        ++layers;
    }
    const double t_recv = secs(t0, Clock::now());
    (void)eng.agree(layers);
    const auto t1 = Clock::now();
    const mm::Row last{0, prompt.back(), rows};
    std::vector<std::uint32_t> gen{eng.step({&last, 1})[0]};
    const double t_last = secs(t1, Clock::now());
    if (eng.rank() == 0) {
        const std::uint32_t tok = gen[0];
        if (::send(fd, &tok, 4, 0) != 4) throw std::runtime_error("offload-listen: reply");
    }
    ::close(fd);
    for (int s = 1; s <= steps; ++s) {
        const mm::Row row{0, gen.back(), prompt.size() + gen.size() - 1};
        gen.push_back(eng.step({&row, 1})[0]);
    }
    std::printf("rank %u offload-listen %s #%u: %u layers, %.1f MB on the wire (%.3f of plain), first frame %.3f s, all received %.3f s "
                "after accept, decode %.3f s, import %.3f s, last row %.3f s; tokens fnv %016llx ids:",
                eng.rank(), name.c_str(), req, layers, static_cast<double>(wire_bytes) / 1e6,
                static_cast<double>(wire_bytes) / static_cast<double>(std::max<std::size_t>(1, layers * block.size())), t_first, t_recv, t_decode,
                t_import, t_last,
                static_cast<unsigned long long>(fnv(gen.data(), gen.size() * 4)));
    for (auto t : gen) std::printf(" %u", t);
    std::printf("\n");
    std::fflush(stdout);
    std::ofstream o(out_dir + "/gen-listen-" + name + "-" + std::to_string(req) + "-rank" + std::to_string(eng.rank()) + ".u32", std::ios::binary);
    o.write(reinterpret_cast<const char*>(gen.data()), static_cast<std::streamsize>(gen.size() * 4));
    return gen;
}

// Speculative decoding with the DSpark draft (src/minimax/dspark, qualification notes): prefill (taps of every
// chunk appended to the draft context), then cycles of draft (block pass on both ranks, head on this rank's vocabulary
// half, 7 Markov steps with one 8-byte candidate exchange each) -> verify window of k + 1 rows of slot 0 in ONE step()
// -> greedy acceptance -> the accepted rows' taps appended. policy: fixed:K (K = 0..7 DSpark drafts per cycle), auto
// (DSpark window by confidence and measured verify costs), ngram:K (copy drafter only: prompt lookup, no forward pass),
// hybrid:K (copy drafts when the copy drafter agrees on >= 4 tokens, else DSpark fixed:K), chain:K (
// copy INSIDE the DSpark chain - after each neural draft token, a 3-token lookup over history + drafts; a match of >= 12
// fills the rest of the block with the continuation; once a copy is established (its whole window accepted) the next
// cycle tries the copy first with >= 5 and skips the neural draft). Returns the generated ids
// (first = the prefill's token), `steps` + 1 of them.
// DSpark Markov chain form. STRIX_MM_MARKOV=cand (default; STRIX_MM_BASELINE=1 -> full): ds::CandMarkov, top-C per
// half + |H| high-norm W2 rows, one payload exchange per draft, chain on device; X1: acceptance only, verify unchanged |
// full (the former default: 7 x (half-vocabulary step + 8-byte exchange)) | ab (alternates full, cand, ... per spec session,
// for ABA in one load). STRIX_MM_MARKOV_CHECK=0 skips the cross-rank chain hash.
// STRIX_MM_MARKOV_TOPC=v2 (default; baseline v1) | v1: the cand form's top-C selection kernels (same canonical set, same picks).
// STRIX_MM_MARKOV_C / STRIX_MM_MARKOV_H: candidate-set size of the cand form (default 4096 / 8192; larger sets agree with the
// full chain more often, offline res/approx1-K3.json: C4096/H8192 0 acceptance changes on 1857 anchors).
unsigned markov_cand_env(const char* name, unsigned dflt) {
    const char* e = std::getenv(name);
    if (e == nullptr || *e == '\0') return dflt;
    char* end = nullptr;
    const unsigned long v = std::strtoul(e, &end, 10);
    if (*end != '\0' || v == 0 || v > (1UL << 20)) throw std::invalid_argument(std::string(name) + " must be a positive integer");
    return static_cast<unsigned>(v);
}
// STRIX_MM_MARKOV_XCHG=device (default: one device-to-device payload exchange) | host (the original 64 KiB control-channel
// chunks through host memory). Same payload bytes either way -> same picks.
bool markov_xchg_device() {
    static const bool dev = [] {
        const char* e = std::getenv("STRIX_MM_MARKOV_XCHG");
        const std::string m = e != nullptr && *e != '\0' ? e : "device";
        if (m != "device" && m != "host") throw std::invalid_argument("STRIX_MM_MARKOV_XCHG must be device or host");
        return m == "device";
    }();
    return dev;
}
// STRIX_MM_DSPARK_DESKEW=1 (diagnostic, only with STRIX_MM_DSPARK_PROFILE): an 8-byte peer rendezvous after the head lap so
// arrival skew between the ranks is reported as 'rendezvous' instead of landing in the first Markov exchange (full and cand alike).
bool dspark_deskew() {
    static const bool on = std::getenv("STRIX_MM_DSPARK_DESKEW") != nullptr && std::getenv("STRIX_MM_DSPARK_DESKEW")[0] == '1';
    return on;
}
// Defaults ON: the cand form at C 4096 / |H| 8192 with top-C v2 (measured: acceptance unchanged, verify exact).
// STRIX_MM_MARKOV=full, or the master STRIX_MM_BASELINE=1 (an explicit STRIX_MM_MARKOV wins), restores the full chain.
// The four STRIX_MM_MARKOV* items live in the mm_defaults table: print_or_apply() exports their resolved values (explicit
// env > STRIX_MM_BASELINE > default) before any reader runs, so the fallbacks below only matter without apply().
bool mm_baseline() { return strix::minimax::defaults::baseline([](const char* n) { return std::getenv(n); }); }
std::string markov_mode() {
    const char* e = std::getenv("STRIX_MM_MARKOV");
    const std::string m = e != nullptr && *e != '\0' ? e : (mm_baseline() ? "full" : "cand");
    if (m != "full" && m != "cand" && m != "ab") throw std::invalid_argument("STRIX_MM_MARKOV must be full, cand or ab");
    return m;
}
bool markov_cand_session() {
    static int n = 0;
    const std::string m = markov_mode();
    return m == "cand" || (m == "ab" && (n++ % 2) == 1);
}
std::unique_ptr<ds::CandMarkov> g_cand;
// The candidate set (H by W2 row norms over the whole vocabulary, host sort) and the device buffers, built once per process.
// Called right after the drafter loads (prewarm: no spec session pays it) and, as a fallback, by the first cand session.
// Local to the rank (no peer exchange), so the call site does not change the ranks' lockstep.
void ensure_cand(mm::Engine& eng, ds::DeviceDrafter& dd) {
    if (g_cand) return;
    const auto t0 = Clock::now();
    const auto [v0, vn] = eng.vocab_range();
    const auto& w = dd.device_weights();
    const unsigned C = markov_cand_env("STRIX_MM_MARKOV_C", 4096), nh = markov_cand_env("STRIX_MM_MARKOV_H", 8192);
    g_cand = std::make_unique<ds::CandMarkov>(w.markov_w1, w.markov_w2, ds::vocab, v0, vn, C, nh);
    std::printf("rank %u: markov cand ready (C %u, |H| %u, exchange %s, topc %s, %u candidates, payload %.1f KB, setup %.1f ms)\n",
                eng.rank(), C, nh, markov_xchg_device() ? "device" : "host",
                g_cand->topc() == ds::CandMarkov::TopC::v2 ? "v2" : "v1", g_cand->candidates(),
                static_cast<double>(g_cand->payload_bytes()) / 1024.0, secs(t0, Clock::now()) * 1e3);
}

ds::Prior g_auto_prior = ds::Prior::structural_v1;
std::string g_auto_model;
std::shared_ptr<const strix::minimax::serve::MinimaxTokenizer> g_auto_tokenizer;

struct SpecStats {
    std::size_t cycles = 0, drafted = 0, accepted = 0, emitted = 0, copy_cycles = 0, copy_accepted = 0, copy_drafted = 0;
    std::size_t n_markov_fallback = 0;
    std::array<std::size_t, 8> hist{};
    std::array<std::size_t, 7> depth_try{}, depth_ok{};
    double t_draft = 0, t_verify = 0, t_append = 0, wall = 0;
    double t_fwd = 0, t_head = 0, t_markov = 0, t_conf = 0;  // STRIX_MM_DSPARK_PROFILE=1 (synchronised)
    // markov detail (always; host clock reads at existing syncs only): cand chain-hash exchange; full path steps executed,
    // their markov_step_host (step incl. its sync) and peer2 + combine (exchange) time
    double t_hash = 0, t_fstep = 0, t_fx = 0;
    double t_rdv = 0;  // STRIX_MM_DSPARK_DESKEW=1 (with PROFILE): rendezvous after the head lap, reported apart
    std::uint64_t draft_fnv = 14695981039346656037ULL;  // every cycle's verify inputs (window size + drafted tokens)
    std::size_t n_full_steps = 0, n_full_drafts = 0;
    std::size_t n_neural = 0;
    std::array<double, 9> t_rows{};
    std::array<std::size_t, 9> n_rows{};
};

std::vector<std::uint32_t> spec_session(mm::Engine& eng, ds::DeviceDrafter& dd, const std::string& name,
                                        const std::vector<std::uint32_t>& prompt, int steps, const std::string& policy,
                                        const std::string& out_dir) {
    if (policy == "auto") {
        ds::AutoRequest automatic;
        if (eng.rank() == 0) {
            ds::Features f;
            if (g_auto_prior == ds::Prior::structural_v1 && prompt.size() < ds::long_prompt_tokens) {
                if (!g_auto_tokenizer) g_auto_tokenizer = strix::minimax::serve::MinimaxTokenizer::load(g_auto_model + "/tokenizer.json");
                f = ds::features(g_auto_tokenizer->decode(prompt, true));
            }
            automatic.admit({ds::Mode::automatic, true, 0, false, prompt.size(), prompt.size(),
                             static_cast<std::uint64_t>(std::max(0, steps)), true, g_auto_prior, f});
            std::printf("rank %u %s: dspark auto %s prior %s\n", eng.rank(), name.c_str(), automatic.reason, ds::prior_name(g_auto_prior));
        }
        const auto action = eng.agree(automatic.state == ds::AutoRequest::State::active ? 1U : 0U);
        if (action > 1) throw std::runtime_error("auto: unsupported agreed action");
        if (action == 1) return spec_session(eng, dd, name, prompt, steps, "auto:legacy", out_dir);
        eng.set_taps(false);
        eng.set_prefill_tap_sink({});
        dd.drop_staged(0);
        const auto gen = session(eng, name, prompt, steps, out_dir);
        std::ofstream output(out_dir + "/spec-" + name + "-rank" + std::to_string(eng.rank()) + ".u32", std::ios::binary);
        output.write(reinterpret_cast<const char*>(gen.data()), static_cast<std::streamsize>(gen.size() * 4));
        return gen;
    }
    const bool dump_y = std::getenv("STRIX_MM_DSPARK_DUMP_Y") != nullptr && eng.rank() == 0;
    std::ofstream ydump;
    if (dump_y) ydump.open(out_dir + "/y-" + name + "-" + std::to_string(steps) + ".f32", std::ios::binary | std::ios::app);
    // auto:legacy = the original SurvivalController + DraftGate; auto:nogate = the window controller alone
    // probe = M12's stride-1 matched-continuation diagnostic: a full draft (7 Markov steps + confidences) at EVERY token
    // position, verified with k = 0 (plain greedy decoding plus the draft's cost); afterwards each position's drafts are
    // compared with the greedy continuation -> uncensored accepted length 0..7 per position (out_dir/probe-NAME.tsv), so
    // any causal gate / window policy can be replayed offline at token positions
    const bool probe = policy == "probe";
    const bool gated = policy == "auto:legacy", autop = gated || policy == "auto:nogate", ngram = policy.rfind("ngram:", 0) == 0,
               hybrid = policy.rfind("hybrid:", 0) == 0, chain = policy.rfind("chain:", 0) == 0;
    const std::uint32_t fixed_k =
        autop || probe ? 7U : static_cast<std::uint32_t>(std::stoul(policy.substr(policy.find(':') + 1)));
    if ((!autop && !probe && !ngram && !hybrid && !chain && policy.rfind("fixed:", 0) != 0) || fixed_k > ds::block)
        throw std::invalid_argument("spec policy: fixed:K | auto | auto:legacy | auto:nogate | probe | ngram:K | hybrid:K | chain:K (K <= 7)");
    struct ProbeRow {
        std::array<std::uint32_t, ds::block> d;
        std::array<float, ds::block> conf;
        double draft_ms;
    };
    std::vector<ProbeRow> probes;
    bool copy_established = false;
    ds::CopyDrafter copier(8);
    copier.append(prompt);
    eng.set_taps(true);
    dd.drop_staged(0);  // rows staged by a previous sequence's paused cycles do not belong to this one
    eng.set_prefill_tap_sink([&](const float* taps, std::size_t n, std::uint32_t slot, std::uint64_t pos0) {
        dd.append(slot, taps, n, pos0, nullptr);
    });
    const auto t0 = Clock::now();
    std::vector<std::uint32_t> gen{eng.prefill(0, prompt, 0)};
    const double ttft = secs(t0, Clock::now());
    eng.set_prefill_tap_sink({});
    const auto [v0, vn] = eng.vocab_range();
    SpecStats st;
    const bool use_cand = markov_cand_session();
    const bool markov_check = std::getenv("STRIX_MM_MARKOV_CHECK") == nullptr || std::getenv("STRIX_MM_MARKOV_CHECK")[0] != '0';
    if (use_cand) ensure_cand(eng, dd);
    const ds::CandMarkov::Stats cs0 = g_cand ? g_cand->stats() : ds::CandMarkov::Stats{};
    std::vector<double> cost(ds::block + 1, 0.0);  // cost[k] = draft (k > 0) + verify of k + 1 rows, measured online
    // auto: window by ds::SurvivalController on measured costs (verify EMA per row count; unmeasured row counts scaled from
    // the measured ones by 1 + 0.12 (R - 1), the measurement slope; draft EMA)
    ds::SurvivalController survival(16);
    const bool profile = std::getenv("STRIX_MM_DSPARK_PROFILE") != nullptr;
    const auto lin0 = dd.linear_seconds();  // per-session deltas (the drafter is shared by the sessions of a load)
    const auto app0 = dd.append_linear_seconds();
    // per-cycle trace on rank 0 (gate / controller design on real sequences)
    // (STRIX_MM_DSPARK_TRACE=1; repeated sessions of one prompt and policy append, each starting with its header)
    std::FILE* trace = eng.rank() == 0 && std::getenv("STRIX_MM_DSPARK_TRACE") != nullptr
                           ? std::fopen((out_dir + "/trace-" + name + "-" + policy + ".tsv").c_str(), "a")
                           : nullptr;
    if (trace) std::fputs("cycle\tdrafted\tk\taccepted\temitted\tdraft_ms\tverify_ms\tappend_ms\tconf0\tconf1\tconf2\tconf3\tconf4\tconf5\tconf6\td1ok\n", trace);
    ds::DraftGate gate(1.0, 12, true);  // agreed form: the pause is agreed once per drafting cycle with the window
    std::array<double, ds::block + 2> vcost{};
    double dcost = 0;
    std::array<float, ds::block> conf_now{};
    // Unmeasured row counts: least-squares line a + b (R - 1) through the measured ones (perf review: the
    // old 1 + 0.12 (R - 1) scaling priced R2 at ~83 ms on prose-es against 70 measured). One measured point: b = 0.267 a
    // (s6-s11: ~14.7 ms per extra row on a 55 ms step); none: 55 + 14.7 (R - 1) ms.
    const auto verify_cost = [&](std::uint32_t R) {
        if (vcost[R] > 0) return vcost[R];
        double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (std::uint32_t r = 1; r <= ds::block + 1; ++r)
            if (vcost[r] > 0) {
                const double x = r - 1.0;
                n += 1;
                sx += x;
                sy += vcost[r];
                sxx += x * x;
                sxy += x * vcost[r];
            }
        const double x = R - 1.0;
        if (n == 0) return 0.055 + 0.0147 * x;
        if (n == 1 || n * sxx - sx * sx <= 0) {
            const double a = sy / n / (1.0 + 0.267 * sx / n);
            return a * (1.0 + 0.267 * x);
        }
        const double b = (n * sxy - sx * sy) / (n * sxx - sx * sx), a = (sy - b * sx) / n;
        return std::max(a + b * x, 0.5 * a);
    };
    std::uint64_t P = prompt.size();
    std::uint32_t anchor = gen[0];
    copier.append(anchor);
    const auto t_loop = Clock::now();
    while (gen.size() < static_cast<std::size_t>(steps) + 1) {
        std::uint32_t k = std::min<std::uint32_t>(fixed_k, static_cast<std::uint32_t>(steps + 1 - gen.size()));
        std::array<std::uint32_t, ds::block> d{};
        auto a = Clock::now();
        bool via_copy = false;
        if (gated && k > 0 && !gate.draft()) k = 0;  // local: the pause length was agreed when it was decided
        std::uint32_t pause = 0;
        const bool drafted = k > 0;
        if (chain && copy_established && k > 0) {  // resumed copy: >= 5, the neural draft skipped
            const auto c = copier.continuation({}, k, 5);
            if (c.size() == k) {
                via_copy = true;
                std::copy(c.begin(), c.end(), d.begin());
            }
        }
        if ((ngram || hybrid) && k > 0) {
            const auto c = copier.propose(k);
            if (ngram || c.size() >= 4) {
                via_copy = true;
                k = static_cast<std::uint32_t>(c.size());
                std::copy(c.begin(), c.end(), d.begin());
            }
        }
        if (k > 0 && !via_copy) {
            auto p0 = Clock::now();
            const auto lap = [&](double& acc) {
                if (!profile) return;
                ds::DeviceDrafter::device_synchronize();
                const auto now = Clock::now();
                acc += secs(p0, now);
                p0 = now;
            };
            ++st.n_neural;
            dd.forward_block(0, eng.embed_row(anchor), eng.embed_row(ds::noise_token), P, nullptr);
            lap(st.t_fwd);
            const float* logits = eng.head_logits(dd.y(), ds::block);
            if (dump_y) {  // STRIX_MM_DSPARK_DUMP_Y: the head's 7 input rows of every neural draft (rank 0; draft-only head calibration)
                std::vector<float> y(static_cast<std::size_t>(ds::block) * ds::hidden);
                if (hipMemcpy(y.data(), dd.y(), y.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
                    throw std::runtime_error("dump y: hipMemcpy");
                ydump.write(reinterpret_cast<const char*>(y.data()), static_cast<std::streamsize>(y.size() * sizeof(float)));
            }
            lap(st.t_head);
            if (profile && dspark_deskew()) {  // diagnostic: both ranks rendezvous after the head, Markov timer restarts after it
                const auto r0 = Clock::now();
                (void)eng.peer(0);
                p0 = Clock::now();
                st.t_rdv += secs(r0, p0);
            }
            std::array<std::uint32_t, ds::block> prev{};
            std::uint32_t t = anchor;
            const std::uint32_t n_steps = autop || probe ? ds::block : k;  // auto needs every row's confidence
            bool cand_done = false;
            if (use_cand && !chain) {  // chain:K keeps the per-step copy continuation of the full form
                std::array<std::uint32_t, ds::block> ct{};
                cand_done = markov_xchg_device()
                                ? g_cand->draft(logits, anchor,
                                                ds::CandMarkov::DeviceExchange([&](const void* o, void* in, std::size_t b, hipStream_t s) {
                                                    eng.exchange_device(o, in, b, s);
                                                }),
                                                ct, nullptr)
                                : g_cand->draft(logits, anchor,
                                                ds::CandMarkov::Exchange([&](const void* o, void* in, std::size_t b) {
                                                    eng.exchange_bytes(o, in, b);
                                                }),
                                                ct, nullptr);
                if (cand_done && markov_check) {  // both ranks must hold the same chain (one 8-byte exchange)
                    const std::uint64_t h = fnv(ct.data(), ct.size() * 4);
                    const auto h0 = Clock::now();
                    const auto th = eng.peer2(static_cast<std::uint32_t>(h), static_cast<std::uint32_t>(h >> 32U));
                    st.t_hash += secs(h0, Clock::now());
                    if (th[0] != static_cast<std::uint32_t>(h) || th[1] != static_cast<std::uint32_t>(h >> 32U))
                        throw std::runtime_error("markov cand: ranks disagree on the draft chain");
                }
                if (cand_done) {
                    for (std::uint32_t i = 0; i < n_steps; ++i) {
                        prev[i] = t;
                        t = ct[i];
                        d[i] = t;
                    }
                } else {
                    ++st.n_markov_fallback;
                }
            }
            if (!cand_done) ++st.n_full_drafts;
            for (std::uint32_t i = 0; i < n_steps && !cand_done; ++i) {
                prev[i] = t;
                const auto f0 = Clock::now();
                const ds::Candidate mine = dd.markov_step_host(t, logits + static_cast<std::size_t>(i) * vn, v0, v0 + vn, nullptr);
                const auto f1 = Clock::now();
                const auto theirs = eng.peer2(__builtin_bit_cast(std::uint32_t, mine.value), mine.id);
                t = ds::combine(mine, ds::Candidate{__builtin_bit_cast(float, theirs[0]), theirs[1]}).id;
                st.t_fstep += secs(f0, f1);
                st.t_fx += secs(f1, Clock::now());
                ++st.n_full_steps;
                d[i] = t;
                if (chain && i + 1 < k) {  // copy inside the chain: fill the rest of the block on a >= 12 match
                    const auto c = copier.continuation(std::span<const std::uint32_t>(d.data(), i + 1), k - i - 1, 12);
                    if (!c.empty()) {
                        std::copy(c.begin(), c.end(), d.begin() + i + 1);
                        via_copy = true;
                        break;
                    }
                }
            }
            lap(st.t_markov);
            if (autop) {
                conf_now = dd.confidence_host(prev, nullptr);
                std::vector<double> cst(k + 1);
                // the draft is already paid: every window, k = 0 included, carries it (this cycle's throughput)
                for (std::uint32_t j = 0; j <= k; ++j) cst[j] = (dcost > 0 ? dcost : 0.017) + verify_cost(j + 1);
                // both ranks hold the same confidences; costs are measured per rank, so rank 0's choice is agreed
                // one agreement carries the window (bits 0-7) and the gate's pause after this cycle (bits 8+)
                const std::uint32_t agreed =
                    eng.agree(survival.choose(std::span<const float>(conf_now.data(), k), cst) | (gated ? gate.proposal() << 8 : 0U));
                k = agreed & 0xFFU;
                pause = agreed >> 8;
            }
            if (probe) {
                conf_now = dd.confidence_host(prev, nullptr);
                probes.push_back({d, conf_now, secs(a, Clock::now()) * 1e3});
                k = 0;  // decode exactly as plain greedy
            }
            lap(st.t_conf);
        }
        const double td = secs(a, Clock::now());
        std::vector<mm::Row> rows{{0, anchor, P}};
        for (std::uint32_t i = 0; i < k; ++i) rows.push_back({0, d[i], P + 1 + i});
        st.draft_fnv = fnv(&k, sizeof k, st.draft_fnv);  // verify inputs of this cycle: window size + drafted tokens
        st.draft_fnv = fnv(d.data(), std::size_t{k} * 4, st.draft_fnv);
        a = Clock::now();
        const auto g = eng.step(rows);
        const double tv = secs(a, Clock::now());
        const ds::Verdict v = ds::verify_greedy(std::span<const std::uint32_t>(d.data(), k), g);
        a = Clock::now();
        if (gated && !drafted)
            dd.append_deferred(0, eng.last_taps(), v.accepted + 1, P, nullptr);  // batched at the next draft
        else
            dd.append(0, eng.last_taps(), v.accepted + 1, P, nullptr);
        const double ta = secs(a, Clock::now());
        if (trace) {
            std::fprintf(trace, "%zu\t%d\t%u\t%u\t%zu\t%.3f\t%.3f\t%.3f", st.cycles, drafted ? 1 : 0, k, v.accepted, v.emitted.size(),
                         td * 1e3, tv * 1e3, ta * 1e3);
            for (const float c : conf_now) std::fprintf(trace, "\t%.4f", drafted ? c : -1.0F);
            std::fprintf(trace, "\t%d\n", drafted ? (d[0] == g[0] ? 1 : 0) : -1);  // depth-1 outcome, also when k = 0
        }
        for (auto t : v.emitted) gen.push_back(t);
        copier.append(v.emitted);
        copy_established = chain && via_copy && v.accepted == k;
        if (via_copy) {
            ++st.copy_cycles;
            st.copy_drafted += k;
            st.copy_accepted += v.accepted;
        }
        anchor = g[v.accepted];
        P += v.accepted + 1;
        ++st.cycles;
        st.drafted += k;
        st.accepted += v.accepted;
        st.emitted += v.emitted.size();
        ++st.hist[v.accepted];
        for (std::uint32_t i = 0; i < k; ++i) {
            ++st.depth_try[i];
            if (i < v.accepted) ++st.depth_ok[i];
            if (i >= v.accepted) break;
        }
        st.t_draft += td;
        st.t_verify += tv;
        st.t_append += ta;
        st.t_rows[k + 1] += tv;
        ++st.n_rows[k + 1];
        // a transient stall (s11 prose-story: one verify of 702 ms) must not reprice windows or feed the gate
        const bool stall = (vcost[k + 1] > 0 && tv > 3 * vcost[k + 1]) || (drafted && dcost > 0 && td > 3 * dcost);
        if (autop) {
            if (drafted && k == 0)  // the draft was paid and verify gave the next token: depth 1 is observed anyway
                survival.observe(std::span<const float>(conf_now.data(), 1), 1, d[0] == g[0] ? 1U : 0U);
            else
                survival.observe(std::span<const float>(conf_now.data(), k), k, v.accepted);
            if (!stall) {
                vcost[k + 1] = vcost[k + 1] <= 0 ? tv : 0.8 * vcost[k + 1] + 0.2 * tv;
                if (drafted && !via_copy) dcost = dcost <= 0 ? td : 0.8 * dcost + 0.2 * td;  // neural drafts only
            }
        }
        if (gated) {
            if (!stall) gate.observe(drafted, static_cast<std::uint32_t>(v.emitted.size()), td + tv + ta, verify_cost(1));
            gate.pause(pause);
        }
        const double c_new = (k > 0 ? td : 0.0) + tv;
        cost[k] = cost[k] <= 0 ? c_new : 0.8 * cost[k] + 0.2 * c_new;
        if (k == 0) cost[0] = cost[0] <= 0 ? tv : 0.8 * cost[0] + 0.2 * tv;
    }
    st.wall = secs(t_loop, Clock::now());
    if (trace) std::fclose(trace);
    if (probe && eng.rank() == 0) {
        // cycle i drafted from anchor gen[i]; its depth-j draft is right iff it equals gen[i + 1 + j] (k = 0: one token per cycle)
        std::FILE* pf = std::fopen((out_dir + "/probe-" + name + ".tsv").c_str(), "w");
        if (pf) {
            std::fputs("pos\taccepted\tdraft_ms\tconf0\tconf1\tconf2\tconf3\tconf4\tconf5\tconf6\n", pf);
            for (std::size_t i = 0; i < probes.size(); ++i) {
                std::uint32_t acc = 0;
                while (acc < ds::block && i + 1 + acc < gen.size() && probes[i].d[acc] == gen[i + 1 + acc]) ++acc;
                // decided: all 7 matched, or a mismatch was seen inside the available continuation (probe 1)
                const bool complete = acc == ds::block || i + 1 + acc < gen.size();
                std::fprintf(pf, "%zu\t%d\t%.3f", i, complete ? static_cast<int>(acc) : -1, probes[i].draft_ms);
                for (const float c : probes[i].conf) std::fprintf(pf, "\t%.9g", static_cast<double>(c));
                std::fputc('\n', pf);
            }
            std::fclose(pf);
        }
    }
    gen.resize(static_cast<std::size_t>(steps) + 1);
    eng.set_taps(false);
    const double tau = static_cast<double>(st.emitted) / static_cast<double>(std::max<std::size_t>(1, st.cycles));
    std::printf("rank %u spec %s (%s): prompt %zu, TTFT %.3f s; %d tokens in %.3f s = %.2f tok/s; cycles %zu, tau %.3f "
                "(emitted/cycle), drafted %zu accepted %zu (%.1f %%); per cycle draft %.2f ms verify %.2f ms append %.2f ms\n",
                eng.rank(), name.c_str(), policy.c_str(), prompt.size(), ttft, steps, st.wall, steps / st.wall, st.cycles, tau, st.drafted,
                st.accepted, 100.0 * static_cast<double>(st.accepted) / static_cast<double>(std::max<std::size_t>(1, st.drafted)),
                st.t_draft / static_cast<double>(st.cycles) * 1e3, st.t_verify / static_cast<double>(st.cycles) * 1e3,
                st.t_append / static_cast<double>(st.cycles) * 1e3);
    std::printf("rank %u spec %s: drafts fnv %016llx (verify inputs of all %zu cycles)\n", eng.rank(), name.c_str(),
                static_cast<unsigned long long>(st.draft_fnv), st.cycles);
    if (st.n_neural > 0) {
        const ds::CandMarkov::Stats cs1 = g_cand ? g_cand->stats() : ds::CandMarkov::Stats{};
        const std::size_t cd = cs1.drafts - cs0.drafts, fd = std::max<std::size_t>(1, st.n_full_drafts);
        const double cn = static_cast<double>(std::max<std::size_t>(1, cd)) / 1e3;
        std::printf("rank %u spec %s: markov detail: cand drafts %zu (ms each) select %.3f exchange %.3f (%.2f chunks, payload %zu B) "
                    "chain %.3f hash %.3f | full drafts %zu steps %.2f each, step %.3f exchange %.3f ms each draft\n",
                    eng.rank(), name.c_str(), cd, (cs1.select - cs0.select) / cn, (cs1.exchange - cs0.exchange) / cn,
                    static_cast<double>(cs1.exchanges - cs0.exchanges) / static_cast<double>(std::max<std::size_t>(1, cd)),
                    g_cand ? g_cand->payload_bytes() : std::size_t{0}, (cs1.chain - cs0.chain) / cn, st.t_hash / cn, st.n_full_drafts,
                    static_cast<double>(st.n_full_steps) / static_cast<double>(fd), st.t_fstep * 1e3 / static_cast<double>(fd),
                    st.t_fx * 1e3 / static_cast<double>(fd));
    }
    if (profile && st.n_neural > 0) {
        const double n = static_cast<double>(st.n_neural) / 1e3;
        std::printf("rank %u spec %s: draft attribution over %zu neural drafts (ms each): block %.2f head %.2f markov+exchange %.2f "
                    "confidence+choose %.2f rendezvous %.2f\n",
                    eng.rank(), name.c_str(), st.n_neural, st.t_fwd / n, st.t_head / n, st.t_markov / n, st.t_conf / n,
                    st.t_rdv / n);
        const auto& ls = dd.linear_seconds();
        double lin = 0;
        for (int i = 0; i < 7; ++i) lin += ls[i] - lin0[i];
        std::printf("rank %u spec %s: block linears %.2f ms each (q %.2f k %.2f v %.2f o %.2f gate %.2f up %.2f down %.2f), rest of "
                    "the block %.2f; context appends (fc + k/v linears) %.2f ms per cycle\n",
                    eng.rank(), name.c_str(), lin / n, (ls[0] - lin0[0]) / n, (ls[1] - lin0[1]) / n, (ls[2] - lin0[2]) / n,
                    (ls[3] - lin0[3]) / n, (ls[4] - lin0[4]) / n, (ls[5] - lin0[5]) / n, (ls[6] - lin0[6]) / n,
                    st.t_fwd / n - lin / n, [&] {
                        const auto& as = dd.append_linear_seconds();
                        double s = 0;
                        for (int i = 0; i < 8; ++i) s += as[i] - app0[i];
                        return s / static_cast<double>(st.cycles) * 1e3;
                    }());
    }
    if (gated) std::printf("rank %u spec %s: draft gate paused %u cycles\n", eng.rank(), name.c_str(), gate.paused_cycles());
    std::printf("rank %u spec %s: markov %s (fallbacks %zu)\n", eng.rank(), name.c_str(), use_cand ? "cand" : "full", st.n_markov_fallback);
    std::printf("rank %u spec %s: copy-drafter cycles %zu (drafted %zu accepted %zu)\n", eng.rank(), name.c_str(), st.copy_cycles,
                st.copy_drafted, st.copy_accepted);
    std::printf("rank %u spec %s: accepted histogram", eng.rank(), name.c_str());
    for (auto h : st.hist) std::printf(" %zu", h);
    std::printf("; depth acceptance");
    for (std::size_t i = 0; i < 7; ++i)
        if (st.depth_try[i]) std::printf(" %.3f", static_cast<double>(st.depth_ok[i]) / static_cast<double>(st.depth_try[i]));
    std::printf("; verify ms by rows");
    for (std::size_t r = 1; r <= 8; ++r)
        if (st.n_rows[r]) std::printf(" R%zu=%.2f(x%zu)", r, st.t_rows[r] / static_cast<double>(st.n_rows[r]) * 1e3, st.n_rows[r]);
    std::printf("\nrank %u spec %s: tokens fnv %016llx ids:", eng.rank(), name.c_str(), static_cast<unsigned long long>(fnv(gen.data(), gen.size() * 4)));
    for (auto t : gen) std::printf(" %u", t);
    std::printf("\n");
    std::fflush(stdout);
    std::ofstream o(out_dir + "/spec-" + name + "-rank" + std::to_string(eng.rank()) + ".u32", std::ios::binary);
    o.write(reinterpret_cast<const char*>(gen.data()), static_cast<std::streamsize>(gen.size() * 4));
    return gen;
}

// ABA x3 of one runtime flag (A = off, B = on; staging = STRIX_MM_STAGING fused, shared = STRIX_MM_SHARED_FIRST) within
// one load: prefill TTFT A B A B A B, then one continuing greedy
// sequence decoded in blocks of `block` steps A B A B A B (the arms are bitwise equal, so the sequence is the same as
// one arm alone). Prints per-arm medians and every cell.
// Tokens of the last plain session per prompt path (the reference an ABA run's sequence must reproduce).
std::map<std::string, std::vector<std::uint32_t>> g_session_tokens;

// local trellis build with the tiled WMMA core switch (qualification notes);
// weak, so binaries linked against a library without it still link (`--ab pftiled` then refuses).
extern "C" void strix_trellis_lab_set_wmma_tiled(int on) __attribute__((weak));

std::uint64_t ab_flag(mm::Engine& eng, const std::string& flag, const std::vector<std::uint32_t>& prompt, int block,
                      const std::vector<std::uint32_t>* reference) {
    const int pipe0 = strix::minimax::moe::opt::prefill_pipe_override();
    struct RestorePipe {
        bool active;
        int value;
        ~RestorePipe() { if (active) strix::minimax::moe::opt::set_prefill_pipe(value); }
    } restore_pipe{flag == "pfpipe", pipe0};
    const int kv_tile0 = eng.prefill_kv_tile();
    const int idx_conc0 = eng.prefill_index_concurrent();
    int kv_tile_b = -2;
    if (flag == "pfkvtile") {
        if (const char* value = std::getenv("STRIX_MM_AB_KVTILE")) {
            if (std::strcmp(value, "auto") != 0) {
                const char* end = value + std::strlen(value);
                const auto parsed = std::from_chars(value, end, kv_tile_b);
                if (parsed.ec != std::errc{} || parsed.ptr != end || kv_tile_b < 0)
                    throw std::invalid_argument("STRIX_MM_AB_KVTILE must be auto or a nonnegative config index");
            }
        }
    }
    int idx_conc_b = 2;
    if (flag == "pfidxconc") {
        if (const char* value = std::getenv("STRIX_MM_AB_IDXCONC")) {
            const std::string mode(value);
            if (mode != "1" && mode != "2") throw std::invalid_argument("STRIX_MM_AB_IDXCONC must be 1 or 2");
            idx_conc_b = mode == "1" ? 1 : 2;
        }
    }
    const unsigned rework0 = eng.prefill_msa_rework();
    const bool fused_prefill0 = eng.fused_prefill_attention();
    unsigned rework_mode = 1U, rework_base = 0U;
    if (flag == "pfmsarework") {
        if (const char* mode = std::getenv("STRIX_MM_AB_REWORK")) {
            if (std::string(mode) != "1" && std::string(mode) != "2" && std::string(mode) != "3" && std::string(mode) != "4" && std::string(mode) != "5" && std::string(mode) != "6")
                throw std::invalid_argument("STRIX_MM_AB_REWORK must be 1, 2, 3, 4, 5 or 6");
            rework_mode = std::string(mode) == "6" ? 6U : std::string(mode) == "5" ? 5U : std::string(mode) == "4" ? 4U : std::string(mode) == "3" ? 3U : std::string(mode) == "2" ? 2U : 1U;
        }
        if (const char* base = std::getenv("STRIX_MM_AB_REWORK_BASE")) {
            const std::string mode = base;
            if (mode != "0" && mode != "1" && mode != "2" && mode != "3" && mode != "4" && mode != "5" && mode != "6")
                throw std::invalid_argument("STRIX_MM_AB_REWORK_BASE must be 0, 1, 2, 3, 4, 5 or 6");
            rework_base = static_cast<unsigned>(mode[0] - '0');
        }
        std::printf("pfmsarework modes A=%u B=%u\n", rework_base, rework_mode);
    }
    const std::string pipe_on = std::getenv("STRIX_MM_AB_PIPE") != nullptr ? std::getenv("STRIX_MM_AB_PIPE") : "2";  // moepipe B form
    const char* pipe_env = std::getenv("STRIX_MMX_MOE_PIPE");  // restored after the A/B
    const std::string moe_pipe0 = pipe_env != nullptr ? pipe_env : "";
    const bool had_moe_pipe = pipe_env != nullptr;
    auto set = [&](bool on) {
        if (flag == "staging") eng.set_fused_staging(on);
        else if (flag == "shared") eng.set_shared_first(on);
        else if (flag == "attn") eng.set_fused_attention(on);
        else if (flag == "linears") eng.set_linears_opt(on);
        else if (flag == "xfer") eng.set_xfer(on);
        else if (flag == "attnpf") eng.set_fused_prefill_attention(on);
        else if (flag == "index") eng.set_index_tiled(on);
        else if (flag == "idxdec") eng.set_index_decode(on);
        else if (flag == "idxattn") eng.set_index_attn(on);
        else if (flag == "idxconc") eng.set_index_concurrent(on);
        else if (flag == "xbwait") { eng.set_xb_wait(on); eng.set_xb_inplace(false); }  // companion pinned off
        else if (flag == "xbinplace") { eng.set_xb_inplace(on); eng.set_xb_wait(false); }
        else if (flag == "xbboth") { eng.set_xb_wait(on); eng.set_xb_inplace(on); }
        else if (flag == "xbprep") eng.set_xb_prep(on);
        else if (flag == "xbpub") eng.set_xb_publish(on);  // on top of the process XB_INPLACE (env)
        else if (flag == "xbilp") eng.set_xb_postilp(on);
        else if (flag == "xbset") {  // a whole flag set: STRIX_MM_AB_XBSET letters w i p u m l (B) vs all six off (A)
            const char* e = std::getenv("STRIX_MM_AB_XBSET");
            const std::string t = e != nullptr ? e : "wim";  // default: the ON-eligible set (X1, X2, X5)
            auto has = [&](char x) { return on && t.find(x) != std::string::npos; };
            const char* m = std::getenv("STRIX_MM_AB_WARM");
            eng.set_xb_wait(has('w'));
            eng.set_xb_inplace(has('i'));
            eng.set_xb_prep(has('p'));
            eng.set_xb_publish(has('u'));
            eng.set_xb_warm(has('m') ? (m != nullptr ? static_cast<unsigned>(std::strtoul(m, nullptr, 10)) : 8U) : 0U);
            eng.set_xb_postilp(has('l'));
        }
        else if (flag == "xbwarm") {  // exchange-wait warm: 0 vs STRIX_MM_AB_WARM MB (default 8)
            const char* m = std::getenv("STRIX_MM_AB_WARM");
            eng.set_xb_warm(on ? (m != nullptr ? static_cast<unsigned>(std::strtoul(m, nullptr, 10)) : 8U) : 0U);
        }  // the other xb flags keep the process setting (env)
        else if (flag == "moegrid") setenv("STRIX_MMX_MOE_TIGHT_GRID", on ? "1" : "0", 1);  // MoE v4 grids: 33 slot rows (A) vs 4 N + 1 (B)
        else if (flag == "moepipe") setenv("STRIX_MMX_MOE_PIPE", on ? pipe_on.c_str() : "0", 1);  // routed gate/up + down one launch
        else if (flag == "xbwarmo") {  // MSA-window o_proj warm: 0 vs STRIX_MM_AB_WARM_O MB (default 8)
            const char* m = std::getenv("STRIX_MM_AB_WARM_O");
            eng.set_xb_warm_o(on ? (m != nullptr ? static_cast<unsigned>(std::strtoul(m, nullptr, 10)) : 8U) : 0U);
        }
        else if (flag == "pfx") eng.set_prefill_xchg(on);
        else if (flag == "pfidx") {
            eng.set_prefill_msa_batch(true);  // isolate index tiling in both arms
            eng.set_prefill_index_tile(on);
        }
        else if (flag == "pfmsarework") {
            eng.set_fused_prefill_attention(true);
            eng.set_prefill_msa_batch(true);  // isolate scheduling in both arms
            eng.set_prefill_msa_rework(on ? rework_mode : rework_base);
        }
        else if (flag == "pftxuc") {
            const char* mode = std::getenv("STRIX_MM_PF_INTERLEAVE");
            if (mode != nullptr && std::string(mode) == "comm")
                throw std::invalid_argument("--ab pftxuc requires the PFX backend (STRIX_MM_PF_INTERLEAVE != comm)");
            eng.set_prefill_xchg(true);
            eng.set_prefill_inplace(false);
            eng.set_prefill_tx_uncached(on);
        }
        else if (flag == "pfmsa") eng.set_prefill_msa_batch(on);
        else if (flag == "pfattn4") eng.set_prefill_attend4(on);
        else if (flag == "pfidxconc") eng.set_prefill_index_concurrent(on ? idx_conc_b : 0);
        else if (flag == "pfkvtile") eng.set_prefill_kv_tile(on ? kv_tile_b : -1);
        else if (flag == "pfrouter") eng.set_prefill_router_batch(on);
        else if (flag == "pfrouterrows") {  // batch router: 8 rows (A) vs STRIX_MM_AB_ROUTER_ROWS (B, default 32 = tiled)
            const char* e = std::getenv("STRIX_MM_AB_ROUTER_ROWS");
            const unsigned rows = e != nullptr ? static_cast<unsigned>(std::strtoul(e, nullptr, 10)) : 32U;
            eng.set_prefill_router_batch(true);
            eng.set_prefill_router_rows(on ? rows : 8U);
        }
        else if (flag == "core8v2") {  // production MT8 (A: process PIPE/core8) vs PIPE 0 + core8 STRIX_MM_AB_CORE8 (B, default 2)
            const char* e = std::getenv("STRIX_MM_AB_CORE8");
            const std::string value = e != nullptr ? e : "2";
            if (value != "1" && value != "2") throw std::invalid_argument("STRIX_MM_AB_CORE8 must be 1 or 2");
            strix::minimax::moe::opt::set_prefill_core(3);
            eng.set_prefill_mtfit(1);
            eng.set_prefill_pipe(on ? 0 : -1);
            eng.set_prefill_core8(on ? (value == "1" ? 1 : 2) : -1);
        }
        else if (flag == "pftab") eng.set_prefill_table_ring(on ? 1 : 0);
        else if (flag == "pfgufuse") eng.set_prefill_gufuse(on ? 1 : 0);
        else if (flag == "pfil") eng.set_prefill_interleave(on);
        else if (flag == "pfxip") eng.set_prefill_inplace(on);
        else if (flag == "pfpipe") {
            const char* e = std::getenv("STRIX_MM_AB_PIPE");
            const std::string value = e == nullptr ? "2" : e;
            if (value != "1" && value != "2") throw std::invalid_argument("STRIX_MM_AB_PIPE must be 1 or 2");
            eng.set_prefill_pipe(on ? (value == "1" ? 1 : 2) : 0);
        }
        else if (flag == "core8") {
            const char* e = std::getenv("STRIX_MM_AB_CORE8");
            const std::string value = e != nullptr ? e : "1";
            if (value != "1" && value != "2") throw std::invalid_argument("STRIX_MM_AB_CORE8 must be 1 or 2");
            strix::minimax::moe::opt::set_prefill_core(3);
            eng.set_prefill_mtfit(1);
            eng.set_prefill_core8(on ? (value == "1" ? 1 : 2) : 0);
        }
        else if (flag == "mtfit") {
            strix::minimax::moe::opt::set_prefill_core(3);
            eng.set_prefill_mtfit(on ? 1 : 0);
        }
        else if (flag == "pftiled") {  // dense prefill strix_gemm tiled core off (A) / on (B)
            if (strix_trellis_lab_set_wmma_tiled == nullptr) throw std::invalid_argument("--ab pftiled: trellis library without the lab switch");
            strix_trellis_lab_set_wmma_tiled(on ? 1 : 0);
        }
        else if (flag == "pfall") {  // every prefill item together (msa batch, attend v4, router batch, mtfit, tiled)
            if (strix_trellis_lab_set_wmma_tiled == nullptr) throw std::invalid_argument("--ab pfall: trellis library without the lab switch");
            eng.set_prefill_msa_batch(on);
            eng.set_prefill_attend4(on);
            eng.set_prefill_router_batch(on);
            strix::minimax::moe::opt::set_prefill_core(3);
            eng.set_prefill_mtfit(on ? 1 : 0);
            strix_trellis_lab_set_wmma_tiled(on ? 1 : 0);
        }
        else if (flag == "pfcore") {  //: prefill MoE core, lib (off) vs STRIX_MM_AB_PFCORE (on)
            const char* c = std::getenv("STRIX_MM_AB_PFCORE");
            const int core = c != nullptr ? std::atoi(c) : 3;
            if (core < 1 || core > 3) throw std::invalid_argument("STRIX_MM_AB_PFCORE must be 1, 2 or 3 (v2, v3, v4)");
            strix::minimax::moe::opt::set_prefill_core(on ? core : 0);
        } else throw std::invalid_argument("--ab: unknown flag " + flag);
    };
    const bool start_arm = flag == "staging" ? eng.fused_staging() : flag == "attn" ? eng.fused_attention()
                         : flag == "linears" ? eng.linears_opt() : flag == "xfer" ? eng.xfer()
                         : flag == "attnpf" ? eng.fused_prefill_attention() : flag == "index" ? eng.index_tiled()
                         : flag == "idxdec" ? eng.index_decode() : flag == "idxattn" ? eng.index_attn()
                         : flag == "idxconc" ? eng.index_concurrent()
                         : flag == "xbwait" ? eng.xb_wait() : flag == "xbinplace" ? eng.xb_inplace()
                         : flag == "xbboth" ? eng.xb_wait() : flag == "xbprep" ? eng.xb_prep() : flag == "xbpub" ? eng.xb_publish() : flag == "xbwarm" ? eng.xb_warm() != 0 : flag == "xbwarmo" ? eng.xb_warm_o() != 0 : flag == "moegrid" ? true : flag == "moepipe" ? false : flag == "xbilp" ? eng.xb_postilp() : flag == "xbset" ? eng.xb_wait()
                         : flag == "pfx" ? eng.prefill_xchg() : flag == "pfil" ? eng.prefill_interleave()
                         : flag == "pfidx" ? eng.prefill_index_tile()
                         : flag == "pftxuc" ? eng.prefill_tx_uncached()
                         : flag == "pfmsa" ? eng.prefill_msa_batch()
                         : flag == "pfattn4" ? eng.prefill_attend4()
                         : flag == "pfidxconc" ? eng.prefill_index_concurrent() != 0
                         : flag == "pfrouter" ? eng.prefill_router_batch()
                         : flag == "pfxip" ? eng.prefill_inplace()
                         : eng.shared_first();
    const bool pfx0 = eng.prefill_xchg(), pfxip0 = eng.prefill_inplace();
    const bool msa0 = eng.prefill_msa_batch(), att0 = eng.prefill_attend4(), rtr0 = eng.prefill_router_batch();  // pfall restores these
    const unsigned router_rows0 = eng.prefill_router_rows();  // pfrouterrows restores it
    const bool xb = flag == "xbwait" || flag == "xbinplace" || flag == "xbboth" || flag == "xbprep" || flag == "xbpub" || flag == "xbwarm" || flag == "xbwarmo" || flag == "moegrid" || flag == "moepipe" || flag == "xbilp" || flag == "xbset";
    const bool xb_w0 = eng.xb_wait(), xb_i0 = eng.xb_inplace(), xb_p0 = eng.xb_prep(), xb_u0 = eng.xb_publish();
    const unsigned xb_m0 = eng.xb_warm(), xb_o0 = eng.xb_warm_o();
    const auto xb_oc0 = eng.xb_warm_o_count();
    const char* grid_env = std::getenv("STRIX_MMX_MOE_TIGHT_GRID");
    const std::string grid0 = grid_env != nullptr ? grid_env : "";
    const bool xb_l0 = eng.xb_postilp();
    const auto xb_lc0 = eng.xb_postilp_count();
    const auto xb_pc0 = eng.xb_prep_count(), xb_uc0 = eng.xb_publish_count(), xb_mc0 = eng.xb_warm_count();  // decode: both flags restored as they were (saved before any set())
    const auto xb_c0 = eng.xb_counts();
    std::vector<double> ttft[2], step_ms[2];
    std::vector<std::uint32_t> first(6);
    std::vector<std::uint64_t> states, outs;  // per arm: K/V/IK state FNV, final-rows FNV (this rank only; never in the cross-rank hash)
    const bool hash_state = (flag != "pfcore" && (flag == "pfcore" || flag == "pfattn4" || flag == "pfmsa" || flag == "pfmsarework" || flag == "mtfit" || flag == "core8" || flag == "pftab" || flag == "pfgufuse" || flag == "pfidx" || flag == "pftxuc" || flag == "pfpipe" || flag == "pfrouter" || flag == "pfrouterrows" || flag == "core8v2" || flag == "pfidxconc" || flag == "pfkvtile" || flag == "pftiled" || flag == "pfall")) || (state_fnv_on() && (flag == "pfx" || flag == "pfil" || flag == "pfxip" || flag == "pfcore" || std::getenv("STRIX_MM_AB_STATE") != nullptr));
    // pfcore / pfattn4 / pfmsa / pfmsarework / mtfit / core8 / pftab / pfgufuse / pfidx / pftxuc / pfrouter / pfidxconc / pfkvtile / pftiled: A B A B A B A (every B bracketed by two A prefills); the other prefill flags: A B A B A B
    const int prefills = flag == "attn" || flag == "linears" || flag == "xfer" || flag == "idxdec" || flag == "idxconc" || flag == "xbwait" || flag == "xbinplace" || flag == "xbboth" || flag == "xbprep" || flag == "xbpub" || flag == "xbwarm" || flag == "xbwarmo" || flag == "moegrid" || flag == "moepipe" || flag == "xbilp" || flag == "xbset" ? 1 : (flag == "pfcore" || flag == "pfattn4" || flag == "pfmsa" || flag == "pfmsarework" || flag == "mtfit" || flag == "core8" || flag == "pftab" || flag == "pfgufuse" || flag == "pfidx" || flag == "pftxuc" || flag == "pfpipe" || flag == "pfrouter" || flag == "pfrouterrows" || flag == "core8v2" || flag == "pfidxconc" || flag == "pfkvtile" || flag == "pftiled" || flag == "pfall") ? 7 : 6;  // idxattn: both  // decode-only flags: one prefill, no TTFT cells
    first.resize(static_cast<std::size_t>(prefills));
    for (int r = 0; r < prefills; ++r) {
        set(r % 2 == 1);
        const auto t0 = Clock::now();
        first[static_cast<std::size_t>(r)] = eng.prefill(0, prompt, 0);
        ttft[r % 2].push_back(secs(t0, Clock::now()));
        if (hash_state) {
            states.push_back(eng.state_fnv(0, prompt.size()));
            outs.push_back(eng.prefill_out_fnv());
        }
    }
    // xb flags: A B A B A B A (every B bracketed), and matched replay: every block re-decodes the SAME positions with
    // the tokens of block 0 (teacher-forced from block 0's greedy output), so each arm does identical work (context,
    // routed experts); every replayed step must reproduce block 0's next token.
    const int blocks = xb ? 7 : 6;
    std::vector<std::uint32_t> gen{first[0]};
    std::vector<double> cell[7];
    std::size_t replay_diff = 0;
    for (int b = 0; b < blocks; ++b) {
        set(b % 2 == 1);
        for (int s = 0; s < block; ++s) {
            const auto a = Clock::now();
            if (xb && b > 0) {
                const mm::Row row{0, gen[static_cast<std::size_t>(s)], prompt.size() + static_cast<std::size_t>(s)};
                replay_diff += eng.step({&row, 1})[0] != gen[static_cast<std::size_t>(s) + 1] ? 1U : 0U;
            } else {
                const mm::Row row{0, gen.back(), prompt.size() + gen.size() - 1};
                gen.push_back(eng.step({&row, 1})[0]);
            }
            cell[b].push_back(secs(a, Clock::now()) * 1e3);
        }
    }
    if (xb) {
        std::printf("rank %u ab-%s: matched replay: %zu of %d replayed steps differ from block 0%s" "\n", eng.rank(), flag.c_str(), replay_diff,
                    (blocks - 1) * block, replay_diff == 0 ? " (IDENTICAL)" : " (DIFFER)");
        if (replay_diff != 0) g_exact_fail = true;
        for (int b = 0; b < blocks; ++b) {
            std::printf("rank %u ab-%s: cells block %d %c:", eng.rank(), flag.c_str(), b, b % 2 ? 'B' : 'A');
            for (double v : cell[b]) std::printf(" %.3f", v);
            std::printf("\n");
        }
    }
    if (flag == "pfmsarework") {
        eng.set_prefill_msa_rework(rework0);
        eng.set_prefill_msa_batch(msa0);
        eng.set_fused_prefill_attention(fused_prefill0);
    }
    else if (flag == "pftxuc") {
        eng.set_prefill_tx_uncached(start_arm);
        eng.set_prefill_inplace(pfxip0);
        eng.set_prefill_xchg(pfx0);
    }
    else if (flag == "pftab") eng.set_prefill_table_ring(-1);
    else if (flag == "pfgufuse") eng.set_prefill_gufuse(-1);
    else if (flag == "pfrouterrows") {
        eng.set_prefill_router_batch(rtr0);
        eng.set_prefill_router_rows(router_rows0);
    }
    else if (flag == "core8v2") {
        eng.set_prefill_core8(-1);
        eng.set_prefill_pipe(-1);
        eng.set_prefill_mtfit(-1);
        strix::minimax::moe::opt::set_prefill_core(-1);
    }
    else if (flag == "pfpipe") eng.set_prefill_pipe(pipe0);
    else if (flag == "core8") {
        eng.set_prefill_core8(-1);
        eng.set_prefill_mtfit(-1);
        strix::minimax::moe::opt::set_prefill_core(-1);
    }
    else if (flag == "mtfit") {
        eng.set_prefill_mtfit(-1);
        strix::minimax::moe::opt::set_prefill_core(-1);
    }
    else if (flag == "pfcore") strix::minimax::moe::opt::set_prefill_core(-1);  // back to the process default
    else if (flag == "pfidx") {
        eng.set_prefill_index_tile(start_arm);
        eng.set_prefill_msa_batch(msa0);
    }
    else if (flag == "pfkvtile") eng.set_prefill_kv_tile(kv_tile0);
    else if (flag == "pfidxconc") eng.set_prefill_index_concurrent(idx_conc0);
    else if (flag == "pftiled") strix_trellis_lab_set_wmma_tiled(-1);
    else if (flag == "pfall") {
        eng.set_prefill_msa_batch(msa0);
        eng.set_prefill_attend4(att0);
        eng.set_prefill_router_batch(rtr0);
        eng.set_prefill_mtfit(-1);
        strix::minimax::moe::opt::set_prefill_core(-1);
        strix_trellis_lab_set_wmma_tiled(-1);
    }
    else if (xb) {
        eng.set_xb_wait(xb_w0);
        eng.set_xb_inplace(xb_i0);
        eng.set_xb_prep(xb_p0);
        eng.set_xb_publish(xb_u0);
        eng.set_xb_warm(xb_m0);
        eng.set_xb_warm_o(xb_o0);
        if (flag == "moepipe") {
            if (had_moe_pipe) setenv("STRIX_MMX_MOE_PIPE", moe_pipe0.c_str(), 1);
            else unsetenv("STRIX_MMX_MOE_PIPE");
        }
        if (flag == "moegrid") {  // the process setting as it was
            if (grid0.empty()) unsetenv("STRIX_MMX_MOE_TIGHT_GRID");
            else setenv("STRIX_MMX_MOE_TIGHT_GRID", grid0.c_str(), 1);
        }
        eng.set_xb_postilp(xb_l0);
        std::printf("rank %u ab-%s: prep fold %llu layers, self-publish %llu exchanges, warm %llu, post-ilp %llu\n", eng.rank(), flag.c_str(),
                    static_cast<unsigned long long>(eng.xb_prep_count() - xb_pc0), static_cast<unsigned long long>(eng.xb_publish_count() - xb_uc0),
                    static_cast<unsigned long long>(eng.xb_warm_count() - xb_mc0), static_cast<unsigned long long>(eng.xb_postilp_count() - xb_lc0));
        const char* wo = std::getenv("STRIX_MM_AB_WARM_O");
        std::printf("rank %u ab-%s: o_proj warms %llu (B = %s MB)\n", eng.rank(), flag.c_str(),
                    static_cast<unsigned long long>(eng.xb_warm_o_count() - xb_oc0), wo != nullptr ? wo : "8");
        const auto c1 = eng.xb_counts();
        std::printf("rank %u ab-%s: path activation: in-kernel wait %llu, in-place producer %llu exchanges\n", eng.rank(), flag.c_str(),
                    static_cast<unsigned long long>(c1.first - xb_c0.first), static_cast<unsigned long long>(c1.second - xb_c0.second));
    } else set(start_arm);
    if (reference != nullptr && !reference->empty()) {
        const std::size_t m = std::min(reference->size(), gen.size());
        const bool same = std::equal(gen.begin(), gen.begin() + static_cast<long>(m), reference->begin());
        std::printf("rank %u ab-%s: first %zu tokens vs the plain session: %s\n", eng.rank(), flag.c_str(), m,
                    same ? "IDENTICAL" : "DIFFER");
        if (!same) g_exact_fail = true;
    }
    bool same_first = true;
    for (auto f : first) same_first &= f == first[0];
    std::printf("rank %u ab-%s: prompt %zu, TTFT off", eng.rank(), flag.c_str(), prompt.size());
    for (double t : ttft[0]) std::printf(" %.3f", t);
    std::printf(" s | on");
    for (double t : ttft[1]) std::printf(" %.3f", t);
    std::printf(" s; prefill tokens identical across arms: %s\n", same_first ? "yes" : "NO");
    if (!states.empty()) {
        bool same_state = true, same_out = true;
        for (auto st : states) same_state &= st == states[0];
        for (auto o : outs) same_out &= o == outs[0];
        std::printf("rank %u ab-%s: prefill state fnv per arm (A B A B ...):", eng.rank(), flag.c_str());
        for (auto st : states) std::printf(" %016llx", static_cast<unsigned long long>(st));
        std::printf(" -> identical: %s\n", same_state ? "yes" : "NO");
        std::printf("rank %u ab-%s: prefill final-rows fnv per arm (A B A B ...):", eng.rank(), flag.c_str());
        for (auto o : outs) std::printf(" %016llx", static_cast<unsigned long long>(o));
        std::printf(" -> identical: %s\n", same_out ? "yes" : "NO");
        if (!same_state || !same_out) g_exact_fail = true;
    }
    if (!same_first) g_exact_fail = true;
    std::printf("rank %u ab-%s: decode ms/token median per block (A=off, B=on):", eng.rank(), flag.c_str());
    for (int b = 0; b < blocks; ++b) {
        // drop the block's first 2 steps (arm switch)
        std::vector<double> v(cell[b].begin() + std::min<long>(2, static_cast<long>(cell[b].size()) - 1), cell[b].end());
        std::printf(" %c %.2f", b % 2 ? 'B' : 'A', median(v));
    }
    std::printf("\nrank %u ab-%s: tokens fnv %016llx\n", eng.rank(), flag.c_str(), static_cast<unsigned long long>(fnv(gen.data(), gen.size() * 4)));
    std::fflush(stdout);
    return fnv(gen.data(), gen.size() * 4, fnv(first.data(), first.size() * 4));
}

// Decode vs teacher-forced (equivalence check): the logits of decode step t (prefill of the first L - K tokens, then steps
// fed with the TRUE tokens) must match the teacher-forced logits of row t (one prefill of all L tokens, G4 form).
// Identities differ (decode strix_gemv, prefill strix_gemm), so the rows agree to rounding, not bitwise; a decode-path
// bug shows as a large relative difference and argmax disagreements. Rank 0 compares its vocabulary half.
void decode_check(mm::Engine& eng, const std::string& name, const std::vector<std::uint32_t>& prompt, std::size_t k) {
    const std::size_t L = prompt.size();
    if (k == 0 || k >= L) throw std::invalid_argument("--decode-check: 0 < K < prompt length");
    std::vector<std::uint32_t> rows(k);
    for (std::size_t j = 0; j < k; ++j) rows[j] = static_cast<std::uint32_t>(L - k + j);
    const auto full = eng.gate_logits(prompt, rows);  // rank 0: [k][vocab]
    const auto [id0, vn] = eng.vocab_range();
    (void)eng.prefill(0, std::span(prompt).first(L - k), 0);
    double worst = 0, sum = 0;
    std::size_t agree = 0, agree_tf_text = 0, agree_dec_text = 0;
    std::printf("rank %u decode-check %s: L %zu K %zu (rank 0 half: ids [%u, %u))\n", eng.rank(), name.c_str(), L, k, id0, id0 + vn);
    for (std::size_t j = 0; j < k; ++j) {
        const std::uint64_t pos = L - k + j;
        const mm::Row row{0, prompt[pos], pos};
        const std::uint32_t dec = eng.step({&row, 1})[0];
        const auto half = eng.last_logits(1);
        if (eng.rank() != 0) continue;
        const float* tf = full.data() + j * 200064;
        double num = 0, den = 0, mx = 0;
        for (std::uint32_t i = 0; i < vn; ++i) {
            const double d = static_cast<double>(half[i]) - tf[id0 + i];
            num += d * d;
            den += static_cast<double>(tf[id0 + i]) * tf[id0 + i];
            mx = std::max(mx, std::fabs(d));
        }
        const auto tf_arg = static_cast<std::uint32_t>(std::max_element(tf, tf + 200064) - tf);
        const double rel = std::sqrt(num / std::max(den, 1e-300));
        worst = std::max(worst, rel);
        sum += rel;
        agree += dec == tf_arg ? 1U : 0U;
        const std::uint32_t text = pos + 1 < L ? prompt[pos + 1] : 0xFFFFFFFFU;
        agree_tf_text += tf_arg == text ? 1U : 0U;
        agree_dec_text += dec == text ? 1U : 0U;
        std::printf("rank 0 decode-check %s: pos %llu rel_l2 %.3e max_abs %.3e argmax decode %u tf %u text %u\n", name.c_str(),
                    static_cast<unsigned long long>(pos), rel, mx, dec, tf_arg, text);
    }
    if (eng.rank() == 0)
        std::printf("rank 0 decode-check %s: SUMMARY rel_l2 mean %.3e max %.3e; argmax decode == tf %zu/%zu; top-1 == text: tf %zu, decode %zu\n",
                    name.c_str(), sum / static_cast<double>(k), worst, agree, k, agree_tf_text, agree_dec_text);
    std::fflush(stdout);
}

// Batched decode: row i starts at position L - 8 i (teacher-forced first input = prompt[L - 8 i]) in slot i, whose
// cache holds positions < L - 8 i (one prefill of the prompt in slot 0, copied). Each row's tokens in an N-row step
// must equal the same row run alone (batch invariance).
// Per-row cost of a speculative verify: verify-shaped steps = one slot, R consecutive rows (a
// fully accepted DSpark window, as spec_session's step(rows)). After the prefill, 8 plain greedy steps give the true
// tokens gen[0..8] at positions P..P+7 and their logits; a window of R rows (tokens gen[0..R-1]) must return gen[1..R]
// (tokens == plain greedy) and is timed in alternating rounds with the exchange module on / off (when loaded). Then one
// traced pass per arm (StepTrace: GPU time per phase summed over layers, routed experts per MoE layer -> distinct
// experts per layer, i.e. the expert slices read). Writes OUT/vtrace-NAME-rank<r>.tsv (one line per traced step and layer).
std::uint64_t verify_trace(mm::Engine& eng, const std::string& name, const std::vector<std::uint32_t>& prompt, int reps,
                           const std::vector<unsigned>& rs, const std::string& out_dir) {
    const std::uint64_t P = prompt.size();
    std::vector<std::uint32_t> gen{eng.prefill(0, prompt, 0)};
    std::vector<std::uint64_t> alone_fnv;  // logits of each plain step (row alone)
    const auto vr = eng.vocab_range();
    for (int i = 0; i < 8; ++i) {
        const mm::Row row{0, gen.back(), P + gen.size() - 1};
        gen.push_back(eng.step({&row, 1})[0]);
        const auto lg = eng.last_logits(1);
        alone_fnv.push_back(fnv(lg.data(), lg.size() * 4));
    }
    std::uint64_t h = fnv(gen.data(), gen.size() * 4);
    std::size_t mismatched = 0, windows = 0;
    std::map<unsigned, std::uint64_t> win_fnv;   // logits of the R-row window (untraced), for the traced == untraced check
    std::map<unsigned, std::size_t> alone_same;  // rows of the window whose logits are bitwise those of the row alone
    auto window = [&](unsigned R) {
        std::vector<mm::Row> rows;
        for (unsigned i = 0; i < R; ++i) rows.push_back({0, gen[i], P + i});
        const auto a = Clock::now();
        const auto g = eng.step(rows);
        const double ms = secs(a, Clock::now()) * 1e3;
        ++windows;
        for (unsigned i = 0; i < R; ++i) mismatched += g[i] != gen[i + 1] ? 1U : 0U;
        return ms;
    };
    auto logits_fnv = [&](unsigned R) {
        const auto lg = eng.last_logits(R);
        std::size_t same = 0;
        for (unsigned i = 0; i < R; ++i) same += fnv(lg.data() + i * vr.second, vr.second * 4) == alone_fnv[i] ? 1U : 0U;
        alone_same[R] = same;
        return fnv(lg.data(), lg.size() * 4);
    };
    const bool can_ab = eng.xfer();
    const int arms = can_ab ? 2 : 1;
    std::map<std::pair<int, unsigned>, std::vector<double>> cell;  // (arm: 0 = the load's state, 1 = xfer off), R -> ms
    for (unsigned R : rs) {  // warm every shape once, and the reference logits
        window(R);
        win_fnv[R] = logits_fnv(R);
    }
    for (int round = 0; round < 3 * arms; ++round) {  // A B A B A B (A = the load's state)
        const int arm = round % arms;
        if (can_ab) eng.set_xfer(arm == 0);
        for (std::size_t j = 0; j < rs.size(); ++j) {  // ascending on even rounds of an arm, descending on odd ones
            const unsigned R = (round / arms) % 2 == 0 ? rs[j] : rs[rs.size() - 1 - j];
            for (int k = 0; k < reps; ++k) cell[{arm, R}].push_back(window(R));
        }
    }
    if (can_ab) eng.set_xfer(true);
    const double r1 = rs.empty() ? 0.0 : median(cell[{0, rs[0]}]);
    for (int arm = 0; arm < arms; ++arm)
        for (unsigned R : rs) {
            const auto& c = cell[{arm, R}];
            const double m = median(c);
            std::printf("rank %u vt %s: R=%u %s median %.2f ms (min %.2f, max %.2f, n %zu); vs R=%u %+.2f ms = %+.2f ms/extra row\n",
                        eng.rank(), name.c_str(), R, arm == 0 ? (can_ab ? "xfer=on " : "start   ") : "xfer=off", m,
                        *std::min_element(c.begin(), c.end()), *std::max_element(c.begin(), c.end()), c.size(), rs[0],
                        m - r1, R > rs[0] ? (m - r1) / (R - rs[0]) : 0.0);
        }
    std::ofstream tsv(out_dir + "/vtrace-" + name + "-rank" + std::to_string(eng.rank()) + ".tsv");
    tsv << "arm\tR\tlayer\tattn_proj\tmsa\to_proj\tx_attn\tffn\tx_ffn\tdistinct_experts\n";
    eng.set_trace(true);
    for (int arm = 0; arm < arms; ++arm) {
        if (can_ab) eng.set_xfer(arm == 0);
        for (unsigned R : rs) {
            window(R);  // two traced steps: the second one is reported
            window(R);
            const auto lf = logits_fnv(R);
            const mm::StepTrace& t = eng.last_trace();
            constexpr unsigned NP = mm::StepTrace::kPhases;
            std::array<double, NP> sum{};
            double ffn_dense = 0;
            std::size_t slices = 0, dmin = 1000, dmax = 0, moe_layers = 0;
            const std::size_t layers = t.layer_ms.size() / NP;
            for (std::size_t L = 0; L < layers; ++L) {
                for (unsigned p = 0; p < NP; ++p) sum[p] += t.layer_ms[L * NP + p];
                std::array<bool, 128> seen{};
                std::size_t d = 0;
                for (std::size_t i = 0; i < std::size_t{R} * 4; ++i) {
                    const std::uint32_t e = t.routes[L * R * 4 + i];
                    if (e < 128 && !seen[e]) {
                        seen[e] = true;
                        ++d;
                    }
                }
                const bool moe = L >= 3;
                if (moe) {
                    slices += d;
                    dmin = std::min(dmin, d);
                    dmax = std::max(dmax, d);
                    ++moe_layers;
                } else {
                    ffn_dense += t.layer_ms[L * NP + 4];
                }
                tsv << (arm == 0 ? (can_ab ? "on" : "start") : "off") << '\t' << R << '\t' << L;
                for (unsigned p = 0; p < NP; ++p) tsv << '\t' << t.layer_ms[L * NP + p];
                tsv << '\t' << (moe ? d : 0) << '\n';
            }
            std::printf("rank %u vt %s trace R=%u %s: GPU %.2f ms = attn_proj %.2f msa %.2f o_proj %.2f x_attn %.2f ffn %.2f "
                        "(dense %.2f, moe %.2f) x_ffn %.2f head %.2f; distinct experts/MoE layer %.2f (min %zu max %zu), "
                        "%zu expert slices; logits %s untraced, rows == alone %zu/%u\n",
                        eng.rank(), name.c_str(), R, arm == 0 ? (can_ab ? "xfer=on " : "start   ") : "xfer=off", t.total_ms, sum[0],
                        sum[1], sum[2], sum[3], sum[4], ffn_dense, sum[4] - ffn_dense, sum[5], t.head_ms,
                        static_cast<double>(slices) / static_cast<double>(std::max<std::size_t>(moe_layers, 1)), dmin, dmax, slices,
                        lf == win_fnv[R] ? "==" : "!=", alone_same[R], R);
            // logits are this rank's vocabulary half: only the per-rank verdicts enter the cross-rank hash
            const std::size_t verdict = (lf == win_fnv[R] ? 1U : 0U) + 2U * alone_same[R];
            h = fnv(&verdict, sizeof(verdict), h);
        }
    }
    eng.set_trace(false);
    if (can_ab) eng.set_xfer(true);
    std::printf("rank %u vt %s: %zu windows, %zu rows != plain greedy -> tokens %s\n", eng.rank(), name.c_str(), windows,
                mismatched, mismatched == 0 ? "IDENTICAL" : "DIFFER");
    std::fflush(stdout);
    h = fnv(&mismatched, sizeof(mismatched), h);
    return h;
}

std::uint64_t batch(mm::Engine& eng, const std::vector<std::uint32_t>& prompt, int steps, const std::vector<unsigned>& ns,
                    std::uint32_t slots) {
    const unsigned nmax = *std::max_element(ns.begin(), ns.end());
    if (nmax > slots || nmax > 8) throw std::invalid_argument("batch N > slots");
    const std::size_t L = prompt.size() - 1;
    (void)eng.prefill(0, std::span(prompt).first(L), 0);  // slot i keeps positions < L - 8 i of it
    for (unsigned i = 1; i < nmax; ++i) eng.copy_slot(i, 0, L);
    auto run = [&](const std::vector<unsigned>& which, std::vector<double>& dt) {
        std::vector<std::vector<std::uint32_t>> toks(which.size());
        std::vector<mm::Row> rows(which.size());
        for (std::size_t r = 0; r < which.size(); ++r) rows[r] = {which[r], prompt[L - 8 * which[r]], L - 8 * which[r]};
        for (int s = 0; s < steps; ++s) {
            const auto a = Clock::now();
            const auto out = eng.step(rows);
            dt.push_back(secs(a, Clock::now()));
            for (std::size_t r = 0; r < rows.size(); ++r) {
                toks[r].push_back(out[r]);
                rows[r].token = out[r];
                ++rows[r].pos;
            }
        }
        return toks;
    };
    std::vector<std::vector<std::uint32_t>> alone(nmax);
    std::uint64_t h = 14695981039346656037ULL;
    for (unsigned i = 0; i < nmax; ++i) {
        std::vector<double> dt;
        alone[i] = run({i}, dt)[0];
        h = fnv(alone[i].data(), alone[i].size() * 4, h);
    }
    bool all_ok = true;
    for (unsigned n : ns) {
        std::vector<unsigned> which(n);
        std::iota(which.begin(), which.end(), 0U);
        std::vector<double> dt;
        const auto toks = run(which, dt);
        unsigned same = 0;
        for (unsigned i = 0; i < n; ++i) same += toks[i] == alone[i] ? 1U : 0U;
        all_ok &= same == n;
        const std::size_t from = std::min<std::size_t>(7, dt.size() - 1);
        const double med = median(std::vector<double>(dt.begin() + static_cast<long>(from), dt.end()));
        std::printf("rank %u batch N=%u: %d steps at ctx ~%zu: median %.2f ms/step = %.2f tok/s aggregate (%.2f per row); "
                    "rows == alone: %u/%u\n", eng.rank(), n, steps, L, med * 1e3, n / med, 1.0 / med, same, n);
        std::uint64_t hn = 14695981039346656037ULL;  // this N's tokens, every row (cross-process comparisons)
        for (const auto& t : toks) hn = fnv(t.data(), t.size() * 4, hn);
        std::printf("rank %u batch N=%u: tokens fnv %016llx\n", eng.rank(), n, static_cast<unsigned long long>(hn));
        std::fflush(stdout);
    }
    std::printf("rank %u batch: alone tokens fnv %016llx\n", eng.rank(), static_cast<unsigned long long>(h));
    std::printf("rank %u batch: batch invariance %s\n", eng.rank(), all_ok ? "IDENTICAL" : "DIFFER");
    return h;
}

}  // namespace

// decode exchange boundary exactness (--xb-check PROMPT.u32:K:N[:v]): teacher-forced decode of K steps with N rows per step,
// once per flag configuration (in-kernel wait, in-place producer) = 00, 10, 01, 11; this rank's logits of every step
// must be bitwise those of configuration 00 (same process, same state). Rows: N slots with staggered prefixes
// (slot i prefilled with the first L - K - i tokens), or with ':v' one slot and N consecutive positions per step (a
// verify window). Prints the activation counts of each configuration (the paths really ran).
void xb_check(mm::Engine& eng, const std::string& name, const std::vector<std::uint32_t>& prompt, std::size_t k, unsigned n, bool verify) {
    const std::size_t L = std::min<std::size_t>(prompt.size(), 192);  // short prefixes: configurations x N prefills per check
    const std::size_t need = verify ? k * n + 1 : k + n + 1;
    if (n == 0 || n > 8 || L <= need) throw std::invalid_argument("--xb-check: 1 <= N <= 8 and the prompt must be longer than the window");
    const bool w0 = eng.xb_wait(), i0 = eng.xb_inplace(), p0x = eng.xb_prep(), u0 = eng.xb_publish();
    const unsigned m0 = eng.xb_warm();
    const bool l0 = eng.xb_postilp();
    const char* cw = std::getenv("STRIX_MM_XB_CHECK_WARM");  // MB used by the warm configurations (default 8)
    const unsigned warm_mb = cw != nullptr ? static_cast<unsigned>(std::strtoul(cw, nullptr, 10)) : 8U;
    const unsigned o0 = eng.xb_warm_o();
    const char* co = std::getenv("STRIX_MM_XB_CHECK_WARM_O");  // MB used by the o_proj warm configurations (default 8)
    const unsigned warm_o_mb = co != nullptr ? static_cast<unsigned>(std::strtoul(co, nullptr, 10)) : 8U;
    const char* grid_env = std::getenv("STRIX_MMX_MOE_TIGHT_GRID");
    const std::string grid0 = grid_env != nullptr ? grid_env : "";
    // Configurations: STRIX_MM_XB_CHECK_SET = comma list of flag sets (w wait, i inplace, p prep, u self-publish, m warm;
    // o o_proj warm, g tight MoE grid, f routed gate/up + down pipeline (form STRIX_MM_XB_CHECK_PIPE, default 2); '-' none; the first one is the reference), e.g. "-,w,i,iu,p,m,ip,wiupm";
    // unset: the 4 (wait, inplace) sets.
    const std::string pipe_form = std::getenv("STRIX_MM_XB_CHECK_PIPE") != nullptr ? std::getenv("STRIX_MM_XB_CHECK_PIPE") : "2";
    const char* xpipe_env = std::getenv("STRIX_MMX_MOE_PIPE");
    const std::string xpipe0 = xpipe_env != nullptr ? xpipe_env : "";
    const bool had_xpipe = xpipe_env != nullptr;
    struct Cfg { std::string label; bool w, i, p, u, m, l, o, g, f; };
    std::vector<Cfg> cfgs;
    std::string set_list = std::getenv("STRIX_MM_XB_CHECK_SET") != nullptr ? std::getenv("STRIX_MM_XB_CHECK_SET") : "";
    const bool legacy = set_list.empty();
    if (legacy) set_list = "-,w,i,wi";
    for (std::size_t pos = 0; pos <= set_list.size();) {
        const std::size_t c = std::min(set_list.find(',', pos), set_list.size());
        const std::string t = set_list.substr(pos, c - pos);
        auto has = [&](char x) { return t.find(x) != std::string::npos; };
        if (!t.empty()) cfgs.push_back({t, has('w'), has('i'), has('p'), has('u'), has('m'), has('l'), has('o'), has('g'), has('f')});
        pos = c + 1;
    }
    std::uint64_t ref = 0;
    bool all_same = true;
    for (std::size_t ci = 0; ci < cfgs.size(); ++ci) {
        const Cfg& g = cfgs[ci];
        eng.set_xb_wait(g.w);
        eng.set_xb_inplace(g.i);
        eng.set_xb_prep(g.p);
        eng.set_xb_publish(g.u);
        eng.set_xb_warm(g.m ? warm_mb : 0U);
        eng.set_xb_postilp(g.l);
        eng.set_xb_warm_o(g.o ? warm_o_mb : 0U);
        setenv("STRIX_MMX_MOE_TIGHT_GRID", g.g ? "1" : "0", 1);
        setenv("STRIX_MMX_MOE_PIPE", g.f ? pipe_form.c_str() : "0", 1);
        const auto co0 = eng.xb_warm_o_count();
        const auto cl0 = eng.xb_postilp_count();
        const auto c0 = eng.xb_counts();
        const auto cp0 = eng.xb_prep_count(), cu0 = eng.xb_publish_count(), cm0 = eng.xb_warm_count();
        std::uint64_t h = 14695981039346656037ULL;
        std::vector<mm::Row> rows(n);
        if (verify) {
            const std::size_t p0 = L - k * n;
            (void)eng.prefill(0, std::span(prompt).first(p0), 0);
            for (std::size_t j = 0; j < k; ++j) {
                for (unsigned r = 0; r < n; ++r) {
                    const std::size_t pos = p0 + j * n + r;
                    rows[r] = mm::Row{0, prompt[pos], pos};
                }
                (void)eng.step(rows);
                const auto lg = eng.last_logits(n);
                h = fnv(lg.data(), lg.size() * 4, h);
            }
        } else {
            for (unsigned i = 0; i < n; ++i) (void)eng.prefill(i, std::span(prompt).first(L - k - i), 0);
            for (std::size_t j = 0; j < k; ++j) {
                for (unsigned i = 0; i < n; ++i) {
                    const std::size_t pos = L - k - i + j;
                    rows[i] = mm::Row{i, prompt[pos], pos};
                }
                (void)eng.step(rows);
                const auto lg = eng.last_logits(n);
                h = fnv(lg.data(), lg.size() * 4, h);
            }
        }
        const auto c1 = eng.xb_counts();
        if (ci == 0) ref = h;
        all_same &= h == ref;
        char label[64];
        if (legacy) std::snprintf(label, sizeof label, "wait=%u inplace=%u", g.w ? 1U : 0U, g.i ? 1U : 0U);
        else std::snprintf(label, sizeof label, "set=%s", g.label.c_str());
        std::printf("rank %u xb-check %s N%u%s K%zu cfg %s: logits fnv %016llx %s (in-kernel wait %llu, in-place %llu exchanges, prep %llu layers, self-publish %llu exchanges, warm %llu, post-ilp %llu)\n",
                    eng.rank(), name.c_str(), n, verify ? " verify" : "", k, label, static_cast<unsigned long long>(h),
                    h == ref ? "== cfg 00" : "DIFFER", static_cast<unsigned long long>(c1.first - c0.first),
                    static_cast<unsigned long long>(c1.second - c0.second), static_cast<unsigned long long>(eng.xb_prep_count() - cp0),
                    static_cast<unsigned long long>(eng.xb_publish_count() - cu0), static_cast<unsigned long long>(eng.xb_warm_count() - cm0),
                    static_cast<unsigned long long>(eng.xb_postilp_count() - cl0));
        if (g.o || g.g)
            std::printf("rank %u xb-check %s N%u%s K%zu cfg %s: o_proj warms %llu, tight grid %u\n", eng.rank(), name.c_str(), n,
                        verify ? " verify" : "", k, label, static_cast<unsigned long long>(eng.xb_warm_o_count() - co0), g.g ? 1U : 0U);
    }
    eng.set_xb_warm_o(o0);
    if (grid0.empty()) unsetenv("STRIX_MMX_MOE_TIGHT_GRID");
    else setenv("STRIX_MMX_MOE_TIGHT_GRID", grid0.c_str(), 1);
    if (had_xpipe) setenv("STRIX_MMX_MOE_PIPE", xpipe0.c_str(), 1);
    else unsetenv("STRIX_MMX_MOE_PIPE");
    eng.set_xb_postilp(l0);
    eng.set_xb_prep(p0x);
    eng.set_xb_publish(u0);
    eng.set_xb_warm(m0);
    eng.set_xb_wait(w0);
    eng.set_xb_inplace(i0);
    std::printf("rank %u xb-check %s N%u%s: %s\n", eng.rank(), name.c_str(), n, verify ? " verify" : "", all_same ? "ALL IDENTICAL" : "EXACTNESS FAILURE");
    std::fflush(stdout);
    if (!all_same) g_exact_fail = true;
}

int main(int argc, char** argv) {
    // Defaults-on resolver before anything reads a switch; --print-effective-config prints the resolved configuration.
    if (strix::minimax::defaults::print_or_apply("rank", argc, argv)) return 0;
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s MODEL_DIR RANK TRANSPORT OUT_DIR [options]\n", argv[0]);
        return 2;
    }
    try {
        const auto t_start = Clock::now();
        strix::minimax::session::RunOptions session_options;
        mm::Options opt;
        opt.model_dir = argv[1];
        g_auto_model = opt.model_dir;
        const auto auto_cli = strix::minimax::defaults::parse_cli("rank", std::vector<std::string>(argv + 1, argv + argc));
        g_auto_prior = ds::parse_prior(auto_cli.dspark_prior.empty() ? std::getenv("STRIX_MM_DSPARK_PRIOR") : auto_cli.dspark_prior.c_str());
        opt.rank = static_cast<std::uint32_t>(std::stoul(argv[2]));
        opt.slots = 8;
        const std::string out_dir = argv[4];
        std::string gate_dir, gold_dir, tf_dir;
        unsigned gate_layers = 6;
        bool tf_write = false, session_hash_only = false;
        double deadline = 1e9;
        std::vector<std::string> sessions, batches, ab_runs, specs, dchecks, vtraces, offloads, offload_imports, offload_listens, xbchecks;
        std::vector<std::string> session_hash_files;
        std::string dspark_dir = std::string(argv[1]) + "/dspark", dspark_exl3;
        std::vector<std::string> dspark_heads;
        for (int i = 5; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if (a == "--gate") gate_dir = next();
            else if (a == "--gate-layers") gate_layers = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--gold") gold_dir = next();
            else if (a == "--tf") tf_dir = next();
            else if (a == "--tf-write") tf_write = true;
            else if (a == "--session") sessions.push_back(next());
            else if (a == "--offload") offloads.push_back(next());
            else if (a == "--offload-import") offload_imports.push_back(next());
            else if (a == "--offload-listen") offload_listens.push_back(next());
            else if (a == "--profile") sessions.push_back("@" + next());  // a session with per-phase timing
            else if (a == "--batch") batches.push_back(next());
            else if (a == "--verify-trace") vtraces.push_back(next());
            else if (a == "--ab") ab_runs.push_back(next());
            else if (a == "--dspark-prior") (void)next();
            else if (a == "--decode-check") dchecks.push_back(next());
            else if (a == "--xb-check") xbchecks.push_back(next());
            else if (a == "--spec") specs.push_back(next());
            else if (a == "--session-hash-file") session_hash_files.push_back(next());
            else if (a == "--session-hash-only") session_hash_only = true;
            else if (a == "--dspark") dspark_dir = next();
            else if (a == "--dspark-exl3") dspark_exl3 = next();
            else if (a == "--dspark-head") dspark_heads.push_back(next());
            else if (a == "--deadline") deadline = std::stod(next());
            else if (a == "--slots") opt.slots = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--capacity") opt.capacity = std::stoull(next());
            else if (a == "--max-rows") opt.max_rows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (strix::minimax::session::parse_option(session_options,a,next)) {}
            else throw std::invalid_argument("unknown option " + a);
        }
        if(!session_options.save.empty() && !session_options.load.empty()) throw std::invalid_argument("choose session save or load");
        if (session_hash_only) {
            // CPU only: no transport, HIP runtime query, engine or drafter.
            const auto warm = strix::minimax::session::prewarm(opt.model_dir,
                session_options.draft ? dspark_dir : "", dspark_exl3, "/proc/self/exe", session_hash_files);
            std::printf("session prewarm rank %u wall_s %.6f second_s %.6f cache_hits %llu cache_misses 0 all_hits=1\n",
                opt.rank, warm.wall_seconds, warm.second_seconds,
                static_cast<unsigned long long>(warm.hits));
            return 0;
        }
        std::string receipt;
        // The ranks load at different speeds (measurement: 44 s vs 166 s under page-cache pressure): the first collective after
        // the load must not hit the dist layer's 120 s default. An explicit value wins.
        (void)::setenv("STRIX_DIST_USB4_TIMEOUT_MS", "600000", 0);
        (void)::setenv("STRIX_DIST_WAIT_TIMEOUT_MS", "600000", 0);
        // DS production host-path settings (CRC on parallel workers, CRC overlap, 1 MiB host chunks): transport
        // only, bits unchanged; measurement ABA x3 TTFT 1K -12.8 %, decode -1 %. An explicit value wins (=0 disables each).
        (void)::setenv("STRIX_DIST_CRC_PARALLEL", "1", 0);
        (void)::setenv("STRIX_DIST_HOST_CRC_OVERLAP", "1", 0);
        (void)::setenv("STRIX_DIST_HOST_CHUNK_BYTES", "1048576", 0);
        // deferred CRC ON by default (measurement: 7 processes inline/deferred alternated, 384-step 1K
        // sessions, tokens identical, each deferred arm below its two inline neighbours by 0.04-0.12 ms/token).
        // STRIX_MMX_CRC: 0 = off, 1 = inline, deferred = deferred check.
        (void)::setenv("STRIX_MMX_CRC", "deferred", 0);
        // Prefill MoE table ring ON in this app only (qualified here, single engine owner; serve keeps it off).
        // Measurement: exact, 1K -0.5 %, 8K not slower. An explicit value or STRIX_MM_BASELINE=1 wins.
        if (const char* base = std::getenv("STRIX_MM_BASELINE"); base == nullptr || std::string(base) != "1")
            (void)::setenv("STRIX_MM_PF_TABLE_RING", "1", 0);
        auto transport = open_transport(argv[3], opt.rank, receipt);
        if (!receipt.empty()) std::printf("transport: %s\n", receipt.c_str());
        strix::dist::Group group;
        group.minimax_compat = true;
        group.rank = opt.rank;
        group.world = 2;
        group.plan = 0x4d4d4531ULL;  // "MME1"
        const std::size_t max_bytes = std::max<std::size_t>(std::size_t{opt.max_rows} * kHidden, 256 * 782 * 128) * sizeof(float);
        // STRIX_MM_XFER=load|on: exchange module for decode steps (shares the host transport)
        const char* xfer_env = std::getenv("STRIX_MM_XFER");
        const bool xfer_load = (xfer_env == nullptr || std::string(xfer_env) != "off") && !gpu_direct;  // default on (measurement: bitwise, never slower); =off; =load
        std::unique_ptr<strix::minimax::xfer::Exchanger> exchanger;
        if (xfer_load) {
            strix::minimax::xfer::Options xo;
            xo.rank = opt.rank;
            xo.capacity = strix::minimax::xfer::MmSchedule::row_bytes(8);
            xo.stream = nullptr;
            xo.from_env();
            exchanger = std::make_unique<strix::minimax::xfer::Exchanger>(std::make_unique<Borrowed>(*transport), xo);
            std::printf("rank %u: exchange module %s\n", opt.rank, exchanger->name().c_str());
        }
        // STRIX_MM_PFX=off|load|on (default on: exact, GPU-gated prefill exchange, 1K -3.5 %, 8K -1.5 %): the prefill
        // exchange module (shares the host transport)
        const char* pfx_env = std::getenv("STRIX_MM_PFX");
        const bool pfx_load = (pfx_env == nullptr || std::string(pfx_env) != "off") && !gpu_direct;
        // Unset = on only on the qualified host-path USB4 transport; elsewhere unset = load (constructed, off).
        const bool pfx_default_on = std::string(argv[3]).rfind("usb4", 0) == 0;
        const bool pfx_on = pfx_env == nullptr ? pfx_default_on : std::string(pfx_env) == "on";
        std::unique_ptr<strix::minimax::pfx::Exchanger> pfx;
        if (pfx_load) {
            strix::minimax::pfx::Options po;
            po.rank = opt.rank;
            po.capacity = std::size_t{opt.max_rows} * kHidden * sizeof(float);
            po.stream = nullptr;
            po.from_env();
            pfx = std::make_unique<strix::minimax::pfx::Exchanger>(*transport, po);
            g_pfx = pfx.get();
            std::printf("rank %u: prefill exchange module %s, %s, interleave %s\n", opt.rank, pfx->name().c_str(),
                        pfx_on ? "on" : "loaded (off)",
                        std::getenv("STRIX_MM_PF_INTERLEAVE") != nullptr && std::getenv("STRIX_MM_PF_INTERLEAVE")[0] == '1' ? "on" : "off");
        }
        const bool borrow = xfer_load || pfx_load;
        auto comm_owner = gpu_direct ? std::make_unique<strix::dist::Communicator>(group, std::move(transport), std::move(gpu_direct), max_bytes)
                        : borrow ? std::make_unique<strix::dist::Communicator>(group, std::make_unique<Borrowed>(*transport), max_bytes)
                                 : std::make_unique<strix::dist::Communicator>(group, std::move(transport), max_bytes);
        strix::dist::Communicator& comm = *comm_owner;
        std::printf("connected: rank %u transport %s routes %s %.1f s\n", opt.rank, comm.transport_name().c_str(),
                    comm.route_text().c_str(), secs(t_start, Clock::now()));
        std::fflush(stdout);
        mm::Engine eng(opt, comm);
        if (exchanger) eng.set_exchanger(exchanger.get(), xfer_env == nullptr || std::string(xfer_env) == "on");
        if (pfx) eng.set_prefill_exchanger(pfx.get(), pfx_on);
        const auto lr = eng.load();
        std::printf("rank %u: load %.1f s, %.1f GiB read (%s), device used %.1f GiB\n", opt.rank, lr.seconds, lr.gib_read,
                    lr.direct_io ? "O_DIRECT" : "buffered", lr.device_gib);
        std::fflush(stdout);
        const bool session_cli = !session_options.prompt.empty() || !session_options.load.empty();
        if(session_cli) {
            strix::minimax::defaults::Cli cli; cli.slots=opt.slots; cli.capacity=opt.capacity; cli.max_rows=opt.max_rows;
            strix::minimax::defaults::SessionConfig config;
            try {
                config = strix::minimax::defaults::session_config(cli, [](const char* key) { return std::getenv(key); }, UINT64_MAX, true,
                    !session_options.save.empty() ? session_options.save : !session_options.load.empty() ? session_options.load : session_options.directory);
                if ((!session_options.save.empty() || !session_options.load.empty()) && !config.disk) {
                    config.enabled = false; config.reason = config.disk_reason;
                }
            } catch (const std::exception& e) { config.enabled = false; config.reason = e.what(); }
            if (!session_options.budget) session_options.budget=config.ram_bytes;
            session_options.directory=config.directory;
            session_options.max_parked=config.max_parked;
            session_options.disk_budget=config.disk_bytes;
            session_options.disk_enabled=config.disk;
            std::unique_ptr<ds::DraftFile> sf; std::unique_ptr<ds::DeviceDrafter> sd;
            auto runtime = strix::minimax::session::admit_runtime_pair({config.enabled, true, config.reason}, eng.rank(),
                [&](const strix::minimax::session::StartupVote& vote) { return strix::minimax::session::exchange_startup(eng, vote); }, [&] {
                std::fprintf(stderr, "session cache admission: ram_bytes=%llu disk_bytes=%llu disk=%s\n",
                    static_cast<unsigned long long>(session_options.budget), static_cast<unsigned long long>(session_options.disk_budget),
                    session_options.disk_enabled ? "on" : "off");
                if (session_options.draft) {
                    sf = ds::DraftFile::open(dspark_dir, dspark_exl3.empty());
                    ds::DeviceDrafter::Options so;
                    so.slots = opt.slots; so.capacity = opt.capacity;
                    sd = std::make_unique<ds::DeviceDrafter>(sf->weights(), so);
                    if (!dspark_exl3.empty()) eng.attach_draft_exl3(dspark_exl3, *sd);
                }
                return std::make_unique<strix::minimax::session::Runtime>(eng,sd.get(),opt,"/proc/self/exe",session_options,dspark_dir,dspark_exl3,sf ? sf->loader_snapshot() : strix::LoaderSnapshot{});
            }, [](const std::string& line) { std::fprintf(stderr, "%s\n", line.c_str()); });
            strix::minimax::session::run_cli(*runtime,session_options); return 0;
        }
        std::uint64_t hs = 14695981039346656037ULL;
        auto in_time = [&]() { return eng.agree(secs(t_start, Clock::now()) < deadline ? 1U : 0U) != 0; };

        if (!gate_dir.empty()) {
            const auto tokens = read_u32(gate_dir + "/tokens.u32");
            const auto t0 = Clock::now();
            const auto res = eng.gate_residual(tokens, gate_layers);
            const std::uint64_t rf = fnv(res.data(), res.size() * 4);
            hs = fnv(&rf, 8, hs);
            std::printf("rank %u gate: residual after layer %u fnv %016llx (T %zu, %.1f s)\n", opt.rank, gate_layers - 1,
                        static_cast<unsigned long long>(rf), tokens.size(), secs(t0, Clock::now()));
            if (!gold_dir.empty()) {
                const std::size_t T = tokens.size();
                const auto ref = read_f32(gold_dir + "/h" + std::to_string(gate_layers - 1) + ".f32", T * kHidden);
                const Metrics m = compare(res, ref, T, kHidden);
                std::printf("rank %u gate: layer %u vs oracle rel_l2 %.3e tok_p50 %.3e tok_p99 %.3e tok_max %.3e\n", opt.rank,
                            gate_layers - 1, m.rel_l2, m.tok_p50, m.tok_p99, m.tok_max);
            }
            std::fflush(stdout);
        }
        if (!tf_dir.empty() && in_time()) {
            const auto tokens = read_u32(tf_dir + "/tokens.u32");
            const auto rows = read_u32(tf_dir + "/rows.u32");
            const auto t0 = Clock::now();
            const auto full = eng.gate_logits(tokens, rows);
            if (opt.rank == 0) {
                bool finite = true;
                for (float f : full) finite &= std::isfinite(f);
                std::size_t scored = 0, hit = 0;
                for (std::size_t s = 0; s < rows.size(); ++s) {
                    if (rows[s] + 1 >= tokens.size()) continue;
                    const float* l = full.data() + s * kVocab;
                    const auto best = static_cast<std::uint32_t>(std::max_element(l, l + kVocab) - l);
                    ++scored;
                    hit += best == tokens[rows[s] + 1] ? 1U : 0U;
                }
                const std::uint64_t lf = fnv(full.data(), full.size() * 4);
                std::printf("rank 0 tf: logits fnv %016llx finite %s; top-1 == real next token %zu/%zu (%.1f %%); %.1f s\n",
                            static_cast<unsigned long long>(lf), finite ? "yes" : "NO", hit, scored,
                            100.0 * static_cast<double>(hit) / static_cast<double>(std::max<std::size_t>(scored, 1)), secs(t0, Clock::now()));
                if (tf_write) {
                    std::ofstream o(out_dir + "/logits_pair.f32", std::ios::binary);
                    o.write(reinterpret_cast<const char*>(full.data()), static_cast<std::streamsize>(full.size() * 4));
                }
            } else {
                std::printf("rank 1 tf: %.1f s\n", secs(t0, Clock::now()));
            }
            std::fflush(stdout);
        }
        std::map<std::string, std::vector<std::uint32_t>> plain;  // greedy ids of each --session, by name
        // decode: activation receipts of the exchange boundary paths (deltas per session / spec, totals at the end)
        auto xb_now = [&] {
            const auto c = eng.xb_counts();
            return std::array<std::uint64_t, 6>{c.first, c.second, eng.xb_prep_count(), eng.xb_publish_count(), eng.xb_warm_count(), eng.xb_postilp_count()};
        };
        auto xb_receipt = [&](const std::string& what, const std::array<std::uint64_t, 6>& a0) {
            const auto a1 = xb_now();
            std::printf("rank %u %s: xb activation: in-kernel wait %llu, in-place %llu exchanges, prep %llu layers, self-publish %llu exchanges, warm %llu, post-ilp %llu\n",
                        opt.rank, what.c_str(), static_cast<unsigned long long>(a1[0] - a0[0]), static_cast<unsigned long long>(a1[1] - a0[1]),
                        static_cast<unsigned long long>(a1[2] - a0[2]), static_cast<unsigned long long>(a1[3] - a0[3]),
                        static_cast<unsigned long long>(a1[4] - a0[4]), static_cast<unsigned long long>(a1[5] - a0[5]));
            std::fflush(stdout);
        };
        const auto xb_start = xb_now();
        for (const auto& s : sessions) {
            if (!in_time()) {
                std::printf("rank %u: session %s skipped (deadline)\n", opt.rank, s.c_str());
                continue;
            }
            const auto xb_s0 = xb_now();
            const auto c1 = s.find(':'), c2 = s.rfind(':');
            const bool prof = s[0] == '@';
            const std::string name = s.substr(prof ? 1 : 0, c1 - (prof ? 1 : 0)), path = s.substr(c1 + 1, c2 - c1 - 1);
            eng.set_profiling(prof || std::getenv("STRIX_MM_PROFILE") != nullptr);
            const int steps = std::stoi(s.substr(c2 + 1));
            const auto g = session(eng, name, read_u32(path), steps, out_dir);
            if (!prof) g_session_tokens[path] = g;
            plain[name] = g;
            hs = fnv(g.data(), g.size() * 4, hs);
            xb_receipt("session " + name, xb_s0);
        }
        // RTX-prefill hand-off: reference form (local prefill + export), then imports compared with it by NAME.
        std::map<std::string, std::vector<std::uint32_t>> offloaded;
        for (const auto& s : offloads) {
            const auto c1 = s.find(':'), c2 = s.rfind(':');
            const std::string name = s.substr(0, c1);
            offloaded[name] = offload_session(eng, name, read_u32(s.substr(c1 + 1, c2 - c1 - 1)), std::stoi(s.substr(c2 + 1)), out_dir, "");
            hs = fnv(offloaded[name].data(), offloaded[name].size() * 4, hs);
        }
        for (const auto& s : offload_imports) {
            const auto c1 = s.find(':'), c2 = s.find(':', c1 + 1), c3 = s.rfind(':');
            const std::string name = s.substr(0, c1);
            const auto g = offload_session(eng, name, read_u32(s.substr(c1 + 1, c2 - c1 - 1)), std::stoi(s.substr(c3 + 1)), out_dir,
                                           s.substr(c2 + 1, c3 - c2 - 1));
            const auto it = offloaded.find(name);
            if (it != offloaded.end()) {
                std::size_t first = 0;
                while (first < std::min(it->second.size(), g.size()) && it->second[first] == g[first]) ++first;
                const bool same = it->second.size() == g.size() && first == g.size();
                std::printf("rank %u offload-import %s: tokens == local offload session: %s", opt.rank, name.c_str(), same ? "YES" : "NO");
                if (!same) std::printf(" (first difference at %zu)", first);
                std::printf("\n");
                std::fflush(stdout);
            }
            hs = fnv(g.data(), g.size() * 4, hs);
        }
        for (const auto& s : offload_listens) {  // NAME:PROMPT:PORT:STEPS:COUNT
            std::vector<std::string> f;
            for (std::size_t p = 0, q; p <= s.size(); p = q + 1) {
                q = s.find(':', p);
                if (q == std::string::npos) q = s.size();
                f.push_back(s.substr(p, q - p));
            }
            if (f.size() != 5) throw std::invalid_argument("--offload-listen NAME:PROMPT:PORT:STEPS:COUNT");
            const auto prompt = read_u32(f[1]);
            const int lfd = listen_on(static_cast<std::uint16_t>(std::stoi(f[2]) + static_cast<int>(opt.rank)));
            for (unsigned i = 0; i < static_cast<unsigned>(std::stoi(f[4])); ++i) {
                const auto g = listen_session(eng, lfd, f[0], prompt, std::stoi(f[3]), out_dir, i);
                const auto it = offloaded.find(f[0]);
                if (it != offloaded.end()) {
                    std::size_t first = 0;
                    while (first < std::min(it->second.size(), g.size()) && it->second[first] == g[first]) ++first;
                    const bool same = it->second.size() == g.size() && first == g.size();
                    std::printf("rank %u offload-listen %s #%u: tokens == local offload session: %s", opt.rank, f[0].c_str(), i, same ? "YES" : "NO");
                    if (!same) std::printf(" (first difference at %zu)", first);
                    std::printf("\n");
                    std::fflush(stdout);
                }
                hs = fnv(g.data(), g.size() * 4, hs);
            }
            ::close(lfd);
        }
        // DSpark speculative sessions: same prompt and length as the --session of the same NAME must give the same ids.
        std::unique_ptr<ds::DraftFile> draft_file;
        std::unique_ptr<ds::DeviceDrafter> drafter;
        std::map<std::string, std::size_t> spec_count;
        for (const auto& sp : specs) {
            if (!in_time()) {
                std::printf("rank %u: spec %s skipped (deadline)\n", opt.rank, sp.c_str());
                continue;
            }
            const auto xb_p0 = xb_now();
            if (!drafter) {
                const auto t0 = Clock::now();
                // with --dspark-exl3 the block linears come from the EXL3 pack: the MXFP8 ones are not read (1.66 GB)
                draft_file = ds::DraftFile::open(dspark_dir, dspark_exl3.empty());
                ds::DeviceDrafter::Options dopt;
                dopt.capacity = opt.capacity;
                // drafter attention kernels (1..3 gated bit-identical to 0, 4 / 5 not gated): STRIX_MM_DSPARK_ATTN=0..5 (DeviceDrafter::Options::attn_variant, default 2; 0 = the d4 kernels)
                if (const char* e = std::getenv("STRIX_MM_DSPARK_ATTN")) dopt.attn_variant = static_cast<std::uint32_t>(std::atoi(e));
                drafter = std::make_unique<ds::DeviceDrafter>(draft_file->weights(), dopt);
                if (!dspark_exl3.empty()) eng.attach_draft_exl3(dspark_exl3, *drafter);
                for (const std::string& h : dspark_heads) eng.attach_draft_head(h);
                drafter->set_profile(std::getenv("STRIX_MM_DSPARK_PROFILE") != nullptr);
                if (markov_mode() != "full") ensure_cand(eng, *drafter);  // prewarm (cand or ab)
                std::printf("rank %u: dspark %s loaded in %.1f s (%.2f GB device, ring %.0f MB)\n", opt.rank, dspark_dir.c_str(),
                            secs(t0, Clock::now()), static_cast<double>(drafter->weight_bytes()) / 1e9,
                            static_cast<double>(drafter->ring_bytes()) / 1e6);
                std::fflush(stdout);
            }
            const std::string spec_name = sp.substr(0, sp.find(':'));
            const std::size_t spec_index = spec_count[spec_name]++;  // per prompt name: each prompt's sessions run A B A B ...
            if (std::getenv("STRIX_MM_DRAFT_AB") != nullptr) {  // with STRIX_MM_DRAFT_LIN=load (A = strix_gemv)
                const bool on = spec_index % 2 == 1;
                eng.set_draft_linears_opt(on);
                // STRIX_MM_DSPARK_ATTN_B=N: attention follows the same A/B (A = STRIX_MM_DSPARK_ATTN or 2, B = N)
                if (const char* att = std::getenv("STRIX_MM_DSPARK_ATTN_B")) {
                    const char* a0 = std::getenv("STRIX_MM_DSPARK_ATTN");
                    drafter->set_attn_variant(static_cast<std::uint32_t>(std::atoi(on ? att : (a0 ? a0 : "2"))));
                }
                std::printf("rank %u: spec %s #%zu drafter %s\n", opt.rank, spec_name.c_str(), spec_index,
                            on ? "B (enqueue_linears_scheduled, attention B)" : "A (strix_gemv, attention A)");
            }
            if (std::getenv("STRIX_MM_DRAFT_HEAD_AB") != nullptr) {  // with H x --dspark-head: per prompt the spec runs rotate
                const int h = static_cast<int>(spec_index % static_cast<std::size_t>(eng.draft_heads() + 1)) - 1;  // -1 = target
                eng.set_draft_head(h);
                std::printf("rank %u: spec %s #%zu head %d (%s)\n", opt.rank, spec_name.c_str(), spec_index, h,
                            h < 0 ? "target" : dspark_heads[static_cast<std::size_t>(h)].c_str());
            }
            const auto c1 = sp.find(':'), c2 = sp.find(':', c1 + 1), c3 = sp.find(':', c2 + 1);
            const std::string name = sp.substr(0, c1);
            const int steps = std::stoi(sp.substr(c2 + 1, c3 - c2 - 1));
            const auto g = spec_session(eng, *drafter, name, read_u32(sp.substr(c1 + 1, c2 - c1 - 1)), steps, sp.substr(c3 + 1), out_dir);
            const auto it = plain.find(name);
            if (it != plain.end()) {
                const auto& ref = it->second;
                std::size_t first = 0;
                while (first < std::min(ref.size(), g.size()) && ref[first] == g[first]) ++first;
                const bool same = ref.size() == g.size() && first == g.size();
                std::printf("rank %u spec %s: tokens == plain greedy session: %s", opt.rank, name.c_str(), same ? "YES" : "NO");
                if (!same) std::printf(" (first difference at %zu of %zu)", first, std::min(ref.size(), g.size()));
                std::printf("\n");
                std::fflush(stdout);
            }
            hs = fnv(g.data(), g.size() * 4, hs);
            xb_receipt("spec " + sp.substr(0, sp.find(':')), xb_p0);
        }
        for (const auto& d : dchecks) {
            if (!in_time()) continue;
            const auto c = d.rfind(':');
            const std::string path = d.substr(0, c);
            decode_check(eng, path.substr(path.rfind('/') + 1), read_u32(path), std::stoul(d.substr(c + 1)));
        }
        for (const auto& x : xbchecks) {  // PROMPT.u32:K:N[:v]
            if (!in_time()) {
                std::printf("rank %u: xb-check %s skipped (deadline)\n", opt.rank, x.c_str());
                continue;
            }
            const bool v = x.size() > 2 && x.compare(x.size() - 2, 2, ":v") == 0;
            const std::string y = v ? x.substr(0, x.size() - 2) : x;
            const auto cn = y.rfind(':'), ck = y.rfind(':', cn - 1);
            const std::string path = y.substr(0, ck);
            xb_check(eng, path.substr(path.rfind('/') + 1), read_u32(path), std::stoul(y.substr(ck + 1, cn - ck - 1)),
                     static_cast<unsigned>(std::stoul(y.substr(cn + 1))), v);
        }
        for (const auto& b : ab_runs) {
            if (g_exact_fail && b.rfind("xb", 0) == 0) {  // decode: no exchange boundary timing after an exactness failure
                std::printf("rank %u: ab %s skipped (exactness failure earlier in this process)" "\n", opt.rank, b.c_str());
                continue;
            }
            if (!in_time()) {
                std::printf("rank %u: ab %s skipped (deadline)\n", opt.rank, b.c_str());
                continue;
            }
            const auto c0 = b.find(':'), c = b.rfind(':');
            const std::string ppath = b.substr(c0 + 1, c - c0 - 1);
            const auto ref = g_session_tokens.find(ppath);
            std::string name = b.substr(0, c0);
            const char* wo_env = std::getenv("STRIX_MM_AB_WARM_O");
            const std::string wo0 = wo_env != nullptr ? wo_env : "";
            const bool wo_set = name.rfind("xbwarmo=", 0) == 0;
            if (wo_set) {  // xbwarmo=MB: this run's B value (several sizes in one process); the env is restored below
                setenv("STRIX_MM_AB_WARM_O", name.substr(8).c_str(), 1);
                name = "xbwarmo";
            }
            const std::uint64_t ah = ab_flag(eng, name, read_u32(ppath), std::stoi(b.substr(c + 1)),
                                             ref == g_session_tokens.end() ? nullptr : &ref->second);
            if (wo_set) {
                if (wo_env == nullptr) unsetenv("STRIX_MM_AB_WARM_O");
                else setenv("STRIX_MM_AB_WARM_O", wo0.c_str(), 1);
            }
            hs = fnv(&ah, 8, hs);
        }
        for (const auto& v : vtraces) {
            if (!in_time()) {
                std::printf("rank %u: verify-trace %s skipped (deadline)\n", opt.rank, v.c_str());
                continue;
            }
            const auto c1 = v.find(':'), c2 = v.find(':', c1 + 1), c3 = v.find(':', c2 + 1);
            std::vector<unsigned> rs;
            for (std::size_t p = c3 + 1; p < v.size();) {
                const auto e = v.find(',', p);
                rs.push_back(static_cast<unsigned>(std::stoul(v.substr(p, e - p))));
                p = e == std::string::npos ? v.size() : e + 1;
            }
            for (unsigned R : rs)
                if (R == 0 || R > 8) throw std::invalid_argument("--verify-trace: R in 1..8");
            const std::uint64_t vh = verify_trace(eng, v.substr(0, c1), read_u32(v.substr(c1 + 1, c2 - c1 - 1)),
                                                  std::stoi(v.substr(c2 + 1, c3 - c2 - 1)), rs, out_dir);
            hs = fnv(&vh, 8, hs);
        }
        for (const auto& b : batches) {
            if (!in_time()) {
                std::printf("rank %u: batch %s skipped (deadline)\n", opt.rank, b.c_str());
                continue;
            }
            const auto c1 = b.find(':'), c2 = b.find(':', c1 + 1);
            const auto prompt = read_u32(b.substr(0, c1));
            const int steps = std::stoi(b.substr(c1 + 1, c2 - c1 - 1));
            std::vector<unsigned> ns;
            for (std::size_t p = c2 + 1; p < b.size();) {
                const auto e = b.find(',', p);
                ns.push_back(static_cast<unsigned>(std::stoul(b.substr(p, e - p))));
                p = e == std::string::npos ? b.size() : e + 1;
            }
            const std::uint64_t bh = batch(eng, prompt, steps, ns, opt.slots);
            hs = fnv(&bh, 8, hs);
        }
        // Both ranks must have produced the same tokens / residuals: compare the hashes through rank 0.
        const auto mine = static_cast<std::uint32_t>(hs ^ (hs >> 32U));
        const std::uint32_t other = eng.peer(mine);
        std::printf("rank %u: result hash %08x, peer %08x -> ranks %s\n", opt.rank, mine, other, mine == other ? "AGREE" : "DIFFER");
        const auto& st = comm.stats();
        std::printf("rank %u: exchanges %llu, %.1f MB sent, %.2f s in exchange; total %.1f s\n", opt.rank,
                    static_cast<unsigned long long>(st.exchanges), static_cast<double>(st.bytes_sent) / 1e6, st.seconds,
                    secs(t_start, Clock::now()));
        xb_receipt("process totals", xb_start);
        if (g_exact_fail) std::printf("rank %u: EXACTNESS FAILURE across A/B arms (see the ab lines)\n", opt.rank);
        g_cand.reset();  // device buffers go while the HIP runtime is still up (not in static teardown)
        return mine != other ? 3 : g_exact_fail ? 4 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-minimax-rank error: %s\n", e.what());
        return 2;
    }
}
