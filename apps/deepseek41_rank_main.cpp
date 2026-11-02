#include "strix/ds41_dspark_vocab.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/deepseek41_describe.hpp"
#include "strix/deepseek41_device_forward.hpp"
#include "strix/deepseek41_pair_defaults.hpp"
#include "strix/ds41_engram_tp.hpp"
#include "strix/deepseek41_session_cache.hpp"
#include "strix/ds41_pipeline_dspark.hpp"
#include "strix/ds41_pipeline_front.hpp"
#include "strix/ds41_spec_tail.hpp"
#include "strix/ds41_dspark_driver.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/ds41_touch_batch.hpp"
#include "strix/ds41_exact_flags.hpp"
#include "strix/hip_ds41_attention.hpp"
#include "strix/hip_ds41_formats.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// One rank of the two-node deepseek41 decode (expert-parallel halves, split-vocabulary head), or both ranks in one
// process over the in-memory transport (--transport emulate).
//   strix-deepseek41-rank MODEL.gguf --rank R --transport usb4auto | usb4gpu | usb4:/dev/tbstreamA,/dev/tbstreamB | tcp-listen:PORT |
//       tcp-connect:HOST:PORT | emulate  --layers N --prompt-ids a,b,c [--generate G] [--op-hashes] [--logits OUT.f32]
//       [--timing] [--engram-direct] [--no-gather] [--op-timing] [--window-rows R[,R...] --windows N] [--op-timing-out FILE]
// --fill N --fill-ids FILE [--fill-window R[,R...]]: N context tokens before the prompt in verify windows of R rows
// (the list cycled; default 8), every row committed; --fill-keep K[,K...]: commit only the first K rows of each window (a
// rejected draft, cycled) and feed the rejected tokens again (the context stays the same).
// --op-timing: per-op device time, host exchange time and weight bytes of the decode steps (R1), then, with
// --window-rows, of N verify windows of R rows after the generation (rows cycled from the sequence, all committed).
// --window-rows R[,R...]: the windows of each size in turn; without --op-timing only their mean step wall is printed
// (with STRIX_DS41_STEP_TRACE=1|2 the device forward writes its host/device timeline, see hip_ds41_step_trace.hpp).
//       [--timing] [--engram-direct] [--no-gather]
//   strix-deepseek41-rank MODEL.gguf --describe-json [--max-positions N]: tensormeld/engine-describe-v1 from the file header
//       (no weight is read; one JSON line on stdout)
//   Pipeline TAIL (strix/ds41-pipeline-v1, docs/models/deepseek-v4.1/pipeline-stages.md): layers [K, layers) and the head on the stream a
//   FRONT sends; rank 0 listens, rank 1 (world 2) follows rank 0 over the communicator:
//   strix-deepseek41-rank MODEL.gguf --role pipeline-stage --first-layer K [--layers 40] --rank R --transport T|single
//       --stage-listen HOST:PORT --stage-token-file PATH [--stage-once | --stage-sessions N] [--max-positions N] [--op-hashes] [--stream-experts]
//   TAIL wire: busy-poll receive (--stage-no-busy-poll to block in recv), each STEP received straight into pinned host
//   memory (the stream is uploaded from there), the RESULT sent before any printing; --stage-timeline FILE writes one CSV
//   row per step (receive, decode, rank-1 forward, compute, send in microseconds); a summary at the end of each session.
//   --stage-logits FILE (rank 0, gathered logits): every one-row step's full logits appended as u32 position + vocab F32.
//   --stage-peer-chunk C (both ranks, world 2): cut-through forward to rank 1, the STEP in parts of C bytes while it
//   arrives (see to_peer); the timeline then also splits the forward's cost (communicator staging / wire / checks).
//   Speculative TAIL (ds41_pipeline_dspark.hpp): --stage-dspark DRAFT.gguf (rank 0 drafts alone; the forward keeps the
//   inputs of its last three layers), --stage-dspark-vocab SPEC (choose_vocabulary spec, default full),
//   --stage-dspark-policy controller|fixed:N (default controller), --stage-dspark-costs FILE.json (the window costs of
//   this split, {"V": {"R": ms}, "D": ms, "A": ms}: the controller's prior, 4 samples per size; measured cycles are added),
//   --stage-fault-drop-accept N (tests: every N-th decision is not sent; the FRONT's resend gets it),
//   --stage-dspark-placement split|rank0 (world 2: split = both ranks draft, the head split by vocabulary, rank 1 learns
//   each decision (kept rows, published tokens, draft or not) in one control exchange before the draft; rank0 = rank 0
//   drafts alone, rank 1 only commits).
//   --stage-peer-thread (rank 0, with --stage-peer-chunk): the parts' exchanges run on a second thread, so the STEP's
//   receive from the FRONT does not wait for them (same messages to rank 1, same verdict block; default off).
//   Recorded FRONT (tests, timing): --role pipeline-record --layers K --record FILE --prompt-ids a,b,c: layers [0, K)
//   without the head on this group (world 1 or 2, same flags as a whole run), every fed token's boundary (the STEP
//   payload a FRONT would send) appended to FILE; strix-deepseek41-pipeline-replay sends them to a TAIL.
//   Whole run continuing a saved state: --import-state FILE (the state's next token is fed first; --prompt-ids optional);
//   --export-state FILE saves the state at the end of the run (prefill here, decode elsewhere).
// --window-check R[,R...] [--window-reject N]: the prompt tokens fed once in verify windows (the first row alone, then
// windows of the listed sizes, cycled; every N-th window keeps only some rows and the rest is fed again), printing per
// row "wc rank r pos p rows R kept k argmax t slice s fnv": the FNV of each vocabulary slice of the head this process
// computes (world 1: both). With R = 1 it is the R1 reference: every pos must have the same argmax and slice FNVs.
// --state-in FILE: start from a saved attention state (strix-deepseek41-longctx / --state-out) instead of position 0.
// --aba N [--aba-flags filter,direct,seghpw,fold,f16,live,engtp]: the prompt + generation and the --window-rows windows 2N times in this
// process from the same start (--state-in or position 0), the listed attention flags (STRIX_DS41_IDX_CAND_FILTER,
// STRIX_DS41_IDX_DIRECT, STRIX_DS41_ATTN_SEG_HPW, STRIX_DS41_ATTN_WINDOW_FOLD,
// STRIX_DS41_ATTN_TILE_F16, STRIX_DS41_ATTN_LIVE_CHUNKS, STRIX_DS41_ENGRAM_TP; touch = STRIX_DS41_TOUCH_BATCH, rowviews =
// STRIX_DS41_TRELLIS_ROW_VIEWS, normilp = STRIX_DS41_NORM_ILP, routes = STRIX_DS41_R1FUSE_ROUTES, cfast = STRIX_DS41_COMPRESS_FAST,
// swz = STRIX_DS41_ATTN_LDS_SWIZZLE) off in even arms and on in odd ones; per arm one "aba arm" line
// (R1 ms/token, window ms, fingerprints of the R1 tokens + local logits slice, the window tokens and the attention state)
// and at the end "aba exact IDENTICAL|DIFFER" (every arm's fingerprints == arm 0's).
//   Route recording: --routes-seqs FILE --routes OUT [--max-positions N] teacher-forces every sequence of FILE from a
//   reset state (verify windows of 8 rows) and writes the router's top-k per row and layer, for placement studies.
// The per-step lines (and --op-hashes lines) use the format of strix-deepseek41-trimmed, so a rank's output diffs
// against the host forward's directly (ignore the routed_experts ":raw" lines: a rank holds only its expert slots).
namespace {

struct Args {
    std::string model, prompt, transport = "emulate", logits_path, routes_path, state_path, state_in;
    std::uint32_t aba = 0;           // --aba N: 2N arms, flags off / on alternately
    std::string aba_flags = "filter,direct,seghpw";
    std::uint32_t rank = 0, layers = 40, generate = 8, max_positions = 4096, fill = 0, windows = 16;
    std::vector<std::uint32_t> window_rows;  // --window-rows R[,R...]
    bool op_timing = false;
    std::string op_timing_out;
    std::string fill_ids;  // --fill: tokens cycled to fill the context before the prompt
    std::string forced_windows;  // --forced-windows FILE
    std::vector<std::uint32_t> fill_windows;  // --fill-window R[,R...]: window sizes of the fill, cycled (default 8)
    std::vector<std::uint32_t> fill_keep;     // --fill-keep K[,K...]: rows committed per fill window, cycled (default all)
    std::vector<std::uint32_t> window_check;  // --window-check R[,R...]
    std::uint32_t window_reject = 0;          // --window-reject N
    std::string routes_seqs;  // --routes-seqs: teacher-forced sequences whose routes go to --routes (no prompt run)
    unsigned connect_timeout = 60;  // seconds for opening the transport (a dead USB4 link blocks open())
    bool describe = false;
    std::string role, stage_listen, stage_token_file, import_state, export_state;
    std::string session_save, session_load, session_id = "gate", session_suffix;
    std::uint64_t session_bytes = 0;
    bool session_dspark = false;
    std::uint32_t session_suffix_tokens = 64;
    bool session_validate_only = false;
    bool session_hash_only = false;
    std::uint32_t first_layer = 0;
    bool stage_once = false;
    unsigned stage_sessions = 0;   // --stage-sessions N: exit after N sessions (0: serve until killed; --stage-once = 1)
    unsigned stage_timeout = 600;  // seconds a TAIL waits inside a session for the FRONT's next request
    bool stage_busy_poll = true;   // TAIL: spin on the socket while waiting for the next STEP (--stage-no-busy-poll: block)
    std::string stage_timeline;    // TAIL: per-step timestamps as CSV (--stage-timeline FILE)
    std::string stage_logits;      // TAIL rank 0: every one-row step's full logits appended (--stage-logits FILE: u32 position, vocab F32)
    std::size_t stage_peer_chunk = 0;  // TAIL world 2: cut-through forward to rank 1 in chunks of N bytes (0: whole STEP after its receive)
    bool stage_peer_thread = false;    // TAIL rank 0: the cut-through exchanges on a second thread (--stage-peer-thread)
    std::string record;            // --role pipeline-record: the boundaries of every step into this file
    std::string stage_dspark, stage_dspark_vocab = "full", stage_dspark_policy = "controller", stage_dspark_costs;
    std::uint32_t stage_fault_drop_accept = 0;
    std::string stage_dspark_placement = "split";  // split: both TAIL ranks draft (head split by vocabulary); rank0
    bool op_hashes = false, timing = false, engram_direct = false, gather = true, tp = false, split_shared = false, stream_experts = false;
};

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

// Sums of DeviceForward::last_op_timing over steps (one entry per plan step).
struct OpSums {
    std::vector<strix::deepseek41::OpTiming> sum;
    std::size_t steps = 0;
    double step_ms = 0.0;
    void add(const std::vector<strix::deepseek41::OpTiming>& t, double ms) {
        if (sum.empty()) sum = t;
        else
            for (std::size_t i = 0; i < t.size(); ++i) {
                sum[i].ms += t[i].ms;
                sum[i].exchange_ms += t[i].exchange_ms;
                sum[i].weight_bytes += t[i].weight_bytes;
            }
        ++steps;
        step_ms += ms;
    }
    // Per op kind (and site), summed over the layers, mean per step: ms, host exchange ms, GB read, GB/s; then per layer.
    void print(const char* label, std::uint32_t rows, const std::string& csv) const {
        if (steps == 0U) return;
        const double n = static_cast<double>(steps);
        struct Row {
            double ms = 0.0, ex = 0.0, bytes = 0.0;
        };
        std::map<std::string, Row> kinds;
        std::map<std::int32_t, Row> layers;
        Row total;
        for (const auto& o : sum) {
            const std::string k = std::string(strix::deepseek41::to_string(o.op)) + (o.site.empty() ? "" : "/" + o.site);
            for (Row* r : {&kinds[k], &layers[o.layer], &total}) {
                r->ms += o.ms / n;
                r->ex += o.exchange_ms / n;
                r->bytes += static_cast<double>(o.weight_bytes) / n;
            }
        }
        std::printf("op timing, %s (%u rows per step, mean over %zu steps; step wall %.2f ms, device events sum %.2f ms):\n", label, rows,
                    steps, step_ms / n, total.ms);
        std::printf("  %-22s %9s %7s %9s %8s %9s\n", "op", "ms", "share", "host-ex", "GB", "GB/s");
        for (const auto& [k, r] : kinds)
            std::printf("  %-22s %9.3f %6.1f%% %9.3f %8.4f %9.1f\n", k.c_str(), r.ms, 100.0 * r.ms / total.ms, r.ex, r.bytes / 1e9,
                        r.ms > 0.0 ? r.bytes / 1e9 / (r.ms / 1e3) : 0.0);
        std::printf("  %-22s %9.3f %6.1f%% %9.3f %8.4f %9.1f\n", "total", total.ms, 100.0, total.ex, total.bytes / 1e9,
                    total.bytes / 1e9 / (total.ms / 1e3));
        std::printf("  per layer ms / GB:");
        for (const auto& [l, r] : layers) std::printf(" %d:%.3f/%.4f", l, r.ms, r.bytes / 1e9);
        std::printf("\n");
        if (!csv.empty()) {
            std::ofstream f(csv, std::ios::app);
            f << "# " << label << " rows " << rows << " steps " << steps << "\nop,layer,site,ms,host_exchange_ms,weight_bytes\n";
            for (const auto& o : sum)
                f << strix::deepseek41::to_string(o.op) << ',' << o.layer << ',' << o.site << ',' << o.ms / n << ',' << o.exchange_ms / n << ','
                  << static_cast<double>(o.weight_bytes) / n << '\n';
        }
    }
};

// usb4gpu: the device exchange (GPU-direct streams) built next to the host-path control transport.
std::unique_ptr<strix::dist::DeviceExchange> gpu_direct;

std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, unsigned timeout) {
    const std::chrono::seconds limit(timeout);
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon), rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "tcp-listen") return strix::dist::tcp_listen(static_cast<std::uint16_t>(std::stoul(rest)), limit);
    if (kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return strix::dist::tcp_connect(rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), limit);
    }
    if (kind == "usb4auto") {  // every live stream of every cable, paired by handshake (the communicator's discovery)
        strix::dist::Usb4AutoOptions o;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        std::printf("usb4 topology: %s\n", topology.receipt().c_str());
        return t;
    }
    if (kind == "usb4gpu") {  // GPU-direct exchanges over the gpu_direct streams, control over the host-path ones
        strix::dist::Usb4AutoOptions o;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        std::printf("usb4 topology: %s\n", topology.receipt().c_str());
        std::string receipt;
        std::string why_direct;
        if (strix::dist::usb4_gpu_direct_streams(o, *t, &why_direct) == 0U) {  // both ranks agree: host path only
            std::printf("gpu-direct unavailable (%s): host path only\n", why_direct.c_str());
            return t;
        }
        gpu_direct = strix::dist::usb4_gpu_direct(o, topology, *t, {}, &receipt);
        std::printf("usb4 %s\n", receipt.c_str());
        return t;
    }
    if (kind == "usb4") {
        std::vector<std::string> devices;
        std::stringstream ss(rest);
        for (std::string d; std::getline(ss, d, ',');) devices.push_back(d);
        return strix::dist::usb4_host(devices);
    }
    (void)rank;
    throw std::invalid_argument("unknown or unavailable transport " + spec);
}

// ---- pipeline TAIL ----
namespace pipe = strix::ds41::pipe;

// Rank 0 -> rank 1 before every operation of a TAIL session that changes the forward (world 2): one frame of
// kPeerFrame bytes = a 64-byte control block followed by the payload when it fits (one exchange per step; a one-row
// STEP is ~82-90 KB); a larger payload follows in a second exchange. Rank 1 sends zeros of the same size back.
// The frame size follows the STEPs: every control block carries the size of the next frame (reserved[0]); after a STEP
// it is that STEP's size rounded up to 4 KB (128 KB .. 1 MB), so the next verify window of the same row count goes to
// rank 1 in one exchange.
//
// Cut-through forward (--stage-peer-chunk C > 0 on both ranks): the first exchange of every operation is 64 + C bytes
// (the control block and the payload's first C bytes; rank 1 learns the size from it). A STEP is forwarded while it
// arrives from the FRONT: rank 0 receives C bytes, sends them to rank 1 in one exchange, receives the next C bytes, and
// so on, so the TCP receive and the USB4 forward overlap and only the last part is forwarded after the last byte. The
// last exchange carries a 64-byte verdict block after the payload: rank 0 decodes and checks the whole STEP before that
// exchange, and rank 1 runs the step only when the verdict says the STEP passed (a refused or broken STEP is skipped by
// both ranks). Other operations (commit, reset, import, quit) carry the verdict in the control block and the payload
// beyond the first C bytes in one more exchange. Bits: rank 1 receives the same payload bytes as before; only the
// order of the host transfers changes.
struct StageControl {
    std::uint32_t magic = 0x31475453U;  // "STG1"
    std::uint32_t kind = 0;             // StageOp
    std::uint64_t bytes = 0;
    std::uint32_t arg = 0, pad = 0;
    std::uint64_t reserved[5] = {};     // [0] next frame (whole-frame protocol); cut-through: [1] mode, [2] verdict
};
static_assert(sizeof(StageControl) == 64U);
enum StageOp : std::uint32_t { op_step = 1, op_commit = 2, op_reset = 3, op_import = 4, op_quit = 5, op_spec = 6, op_spec_begin = 7 };
// op_spec / op_spec_begin payload (split drafter): u32 kept (0 at begin), u32 rows of the step, u32 next token, u32 draft (0/1),
// u32 count, then the published ids (the drafter's vocabulary context)
std::vector<std::uint8_t> spec_op_payload(std::uint32_t kept, std::uint32_t rows, std::uint32_t next, bool draft, const std::vector<std::uint32_t>& published) {
    std::vector<std::uint32_t> w{kept, rows, next, draft ? 1U : 0U, static_cast<std::uint32_t>(published.size())};
    w.insert(w.end(), published.begin(), published.end());
    std::vector<std::uint8_t> b(w.size() * 4U);
    std::memcpy(b.data(), w.data(), b.size());
    return b;
}
constexpr std::size_t kMinPeerFrame = std::size_t{128} << 10U, kMaxPeerFrame = std::size_t{1} << 20U;
// Per rank thread (--transport emulate runs both ranks in one process; the forward thread copies its rank's values).
thread_local std::size_t kPeerFrame = kMinPeerFrame;      // the current frame (both ranks agree through the control blocks)
thread_local std::size_t kPeerChunk = 0;                  // --stage-peer-chunk (0: the whole-frame protocol above)
thread_local strix::dist::Communicator* kComm = nullptr;  // the forward's cost breakdown (counter deltas)
constexpr std::uint64_t kModeParts = 1U, kModeWhole = 2U;
constexpr std::uint64_t kVerdictOk = 0x4b4f544349445256ULL, kVerdictRefused = 0x4553554645525256ULL;  // "VRDICTOK", "VRREFUSE"
std::size_t frame_after_step(std::size_t payload_bytes) {
    return std::clamp<std::size_t>((64U + payload_bytes + 4095U) / 4096U * 4096U, kMinPeerFrame, kMaxPeerFrame);
}

// The pinned buffer a TAIL rank receives into: [64-byte control | payload | 64-byte verdict], at least one frame.
std::span<std::uint8_t> stage_buffer(strix::deepseek41::DeviceForward& fwd, std::size_t payload_bytes) {
    const std::size_t first = kPeerChunk ? 64U + kPeerChunk + 64U : kPeerFrame;
    return fwd.receive_buffer(std::max(first, 64U + payload_bytes + 64U));
}

// The rank-1 forward's cost from the communicator's counters (rank 0): exchanges, wall, staging, wire, checks.
struct PeerCost {
    std::uint64_t exchanges = 0;
    double seconds = 0.0, stage = 0.0, wire = 0.0, check = 0.0;
};
strix::dist::CommunicatorStats peer_stats() { return kComm ? kComm->stats() : strix::dist::CommunicatorStats{}; }
void add_cost(PeerCost& c, const strix::dist::CommunicatorStats& before) {
    const auto after = peer_stats();
    c.exchanges += after.exchanges - before.exchanges;
    c.seconds += after.seconds - before.seconds;
    c.stage += after.stage - before.stage;
    c.wire += after.wire - before.wire;
    c.check += after.check - before.check;
}

// `frame` = stage_buffer(...) with the payload already at [64, 64 + bytes).
void to_peer(strix::deepseek41::DeviceForward& fwd, StageOp kind, std::uint32_t arg, std::span<std::uint8_t> frame, std::size_t bytes) {
    StageControl c;
    c.kind = kind;
    c.arg = arg;
    c.bytes = bytes;
    if (kPeerChunk) {
        c.reserved[1] = kModeWhole;
        c.reserved[2] = kVerdictOk;
        std::memcpy(frame.data(), &c, 64);
        fwd.exchange_host(std::as_bytes(frame.first(64U + kPeerChunk)), {});
        if (bytes > kPeerChunk) fwd.exchange_host(std::as_bytes(frame.subspan(64U + kPeerChunk, bytes - kPeerChunk)), {});
        return;
    }
    const std::size_t next = kind == op_step ? frame_after_step(bytes) : kPeerFrame;
    c.reserved[0] = next;
    std::memcpy(frame.data(), &c, 64);
    fwd.exchange_host(std::as_bytes(frame.first(kPeerFrame)), {});  // rank 1's bytes back are not needed
    if (64U + bytes > kPeerFrame) fwd.exchange_host(std::as_bytes(frame.subspan(64, bytes)), {});
    kPeerFrame = next;
}
void to_peer(strix::deepseek41::DeviceForward& fwd, StageOp kind, std::uint32_t arg, std::span<const std::uint8_t> payload) {
    std::vector<std::uint8_t> frame(std::max({kPeerFrame, 64U + kPeerChunk + 64U, 64U + payload.size()}));
    std::copy(payload.begin(), payload.end(), frame.begin() + 64);
    to_peer(fwd, kind, arg, frame, payload.size());
}

// Rank 0, --stage-peer-thread: the cut-through exchanges on a second (persistent) thread. The receiving thread lists the
// STEP's parts (begin), marks each one sendable as soon as its bytes are in the frame (publish), and waits for the last
// exchange at the end (finish); the forward thread sends the parts in order, each as soon as it is published. Same
// parts, same order, same bytes as the inline forward: only the thread that calls exchange_host changes, and the receiving
// thread makes no HIP or communicator call while a STEP is in flight (the check after the last byte is host-only).
class PeerForwarder {
public:
    explicit PeerForwarder(strix::deepseek41::DeviceForward& fwd)
        : fwd_(fwd), chunk_(kPeerChunk), frame_(kPeerFrame), comm_(kComm), thread_([this] { run(); }) {}
    ~PeerForwarder() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }
    PeerForwarder(const PeerForwarder&) = delete;
    PeerForwarder& operator=(const PeerForwarder&) = delete;
    void begin(std::vector<std::span<const std::uint8_t>> parts) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            parts_ = std::move(parts);
            ready_.store(0, std::memory_order_relaxed);
            all_received_.store(false, std::memory_order_relaxed);
            done_.store(false, std::memory_order_relaxed);
            total_ = exposed_ = PeerCost{};
            error_ = nullptr;
            job_ = true;
        }
        cv_.notify_one();
    }
    void publish(std::size_t parts_ready) { ready_.store(parts_ready, std::memory_order_release); }
    void received() { all_received_.store(true, std::memory_order_release); }
    // Waits for the last exchange; the cost of every exchange and of those that started after the last byte.
    void finish(PeerCost& total, PeerCost& exposed) {
        while (!done_.load(std::memory_order_acquire)) relax();
        total = total_;
        exposed = exposed_;
        if (error_) std::rethrow_exception(error_);
    }

private:
    static void relax() {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#else
        std::this_thread::yield();
#endif
    }
    void run() {
        kPeerChunk = chunk_;  // this thread's copies of the rank's values
        kPeerFrame = frame_;
        kComm = comm_;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return job_ || stop_; });
                if (stop_) return;
                job_ = false;
            }
            try {
                for (std::size_t i = 0; i < parts_.size(); ++i) {
                    while (ready_.load(std::memory_order_acquire) <= i) relax();
                    const bool late = all_received_.load(std::memory_order_acquire);
                    const auto before = peer_stats();
                    fwd_.exchange_host(std::as_bytes(parts_[i]), {});
                    add_cost(total_, before);
                    if (late) add_cost(exposed_, before);
                }
            } catch (...) {
                error_ = std::current_exception();
            }
            done_.store(true, std::memory_order_release);
        }
    }
    strix::deepseek41::DeviceForward& fwd_;
    std::size_t chunk_, frame_;
    strix::dist::Communicator* comm_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool job_ = false, stop_ = false;
    std::vector<std::span<const std::uint8_t>> parts_;
    std::atomic<std::size_t> ready_{0};
    std::atomic<bool> all_received_{false}, done_{true};
    PeerCost total_, exposed_;
    std::exception_ptr error_;
    std::thread thread_;  // last: starts after the members above exist
};

// Rank 0, cut-through: receive a STEP of `bytes` bytes from the FRONT into frame[64, 64 + bytes) while forwarding it to
// rank 1 part by part. `check` (decode + validation of the whole payload) runs after the last byte and before the last
// exchange; its verdict goes to rank 1. A receive error (the FRONT went away) does not stop the forward: the missing
// bytes are zeros, the verdict is "refused", and the error is reported after rank 1 has the whole message.
// `worker` (--stage-peer-thread): the exchanges run on the forward thread while this thread keeps receiving.
struct CutThrough {
    std::chrono::steady_clock::time_point recv, checked, forwarded;
    PeerCost total, exposed;  // every exchange of the STEP; the exchanges after the last byte
    std::size_t parts = 0;
    bool passed = false;
    std::string receive_error;
};
template <class Check>
CutThrough receive_forward_step(pipe::Connection& conn, strix::deepseek41::DeviceForward& fwd, std::span<std::uint8_t> frame, std::size_t bytes,
                                const Check& check, PeerForwarder* worker = nullptr) {
    CutThrough out;
    const std::size_t C = kPeerChunk;
    const auto recv = [&](std::size_t at, std::size_t n) {
        if (out.receive_error.empty()) {
            try {
                conn.receive_payload(frame.subspan(64U + at, n));
                return;
            } catch (const std::exception& e) {
                out.receive_error = e.what();
            }
        }
        std::memset(frame.data() + 64 + at, 0, n);
    };
    const auto verdict = [&] {
        out.recv = std::chrono::steady_clock::now();
        if (worker) worker->received();
        out.passed = out.receive_error.empty() && check();
        out.checked = std::chrono::steady_clock::now();
        return out.passed ? kVerdictOk : kVerdictRefused;
    };
    std::size_t published = 0;
    const auto send = [&](std::span<const std::uint8_t> part, bool last) {
        ++out.parts;
        if (worker) {
            worker->publish(++published);
            return;
        }
        const auto before = peer_stats();
        fwd.exchange_host(std::as_bytes(part), {});
        add_cost(out.total, before);
        if (last) add_cost(out.exposed, before);
    };
    if (worker) {  // the same parts as below, sent by the forward thread as they are published
        std::vector<std::span<const std::uint8_t>> parts;
        parts.push_back(frame.first(64U + C));
        if (bytes > C) {
            std::size_t at = C;
            for (; bytes - at > C; at += C) parts.push_back(frame.subspan(64U + at, C));
            parts.push_back(frame.subspan(64U + at, bytes - at + 64U));
        }
        worker->begin(std::move(parts));
    }
    StageControl c;
    c.kind = op_step;
    c.bytes = bytes;
    c.reserved[1] = kModeParts;
    if (bytes <= C) {  // one exchange: the control block carries the verdict
        recv(0, bytes);
        c.reserved[2] = verdict();
        std::memcpy(frame.data(), &c, 64);
        send(frame.first(64U + C), true);
    } else {
        recv(0, C);
        std::memcpy(frame.data(), &c, 64);
        send(frame.first(64U + C), false);
        std::size_t at = C;
        for (; bytes - at > C; at += C) {
            recv(at, C);
            send(frame.subspan(64U + at, C), false);
        }
        const std::size_t n = bytes - at;
        recv(at, n);
        const std::uint64_t v = verdict();
        std::memset(frame.data() + 64 + bytes, 0, 64);
        std::memcpy(frame.data() + 64 + bytes, &v, 8);
        send(frame.subspan(64U + at, n + 64U), true);
    }
    if (worker) worker->finish(out.total, out.exposed);
    out.forwarded = std::chrono::steady_clock::now();
    return out;
}

void follow_stage(const Args& a, strix::deepseek41::DeviceForward& fwd, strix::dist::Communicator* comm) {
    const auto geo = pipe::boundary_geometry(strix::deepseek41::validate_target(strix::deepseek41::read_file(a.model)).config);
    std::vector<std::byte> zeros(kMaxPeerFrame + 128U);
    // split drafter: rank 1 drafts with rank 0 (the head's vocabulary halves), on the same committed rows
    std::unique_ptr<strix::ds41::dspark::SpecTailDrafter> drafter;
    std::vector<std::uint32_t> context, last_tokens;
    if (!a.stage_dspark.empty() && a.stage_dspark_placement == "split") {
        drafter = std::make_unique<strix::ds41::dspark::SpecTailDrafter>(a.model, a.stage_dspark, a.stage_dspark_vocab);
        drafter->set_split(1U, *comm);  // this rank's communicator (both ranks share the globals under --transport emulate)
    }
    {  // rank 0 announces READY only after this: both ranks are past their load
        std::array<std::byte, 64> out{}, in{};
        fwd.exchange_host(out, in);
    }
    const auto zero = [&](std::size_t n) { return std::span<const std::byte>(zeros).first(n); };
    for (;;) {
        auto frame = stage_buffer(fwd, 0);  // pinned: a step's stream is uploaded from here
        StageControl c;
        if (kPeerChunk) {
            const std::size_t C = kPeerChunk;
            fwd.exchange_host(zero(64U + C), std::as_writable_bytes(frame.first(64U + C)));
            std::memcpy(&c, frame.data(), 64);
            if (c.magic != StageControl{}.magic || (c.reserved[1] != kModeParts && c.reserved[1] != kModeWhole))
                throw std::runtime_error("pipeline TAIL rank 1: bad control block");
            if (64U + c.bytes + 64U > frame.size()) {  // a larger buffer (a new allocation): keep the first part
                std::vector<std::uint8_t> head(frame.begin(), frame.begin() + static_cast<std::ptrdiff_t>(64U + C));
                frame = stage_buffer(fwd, c.bytes);
                std::copy(head.begin(), head.end(), frame.begin());
            }
            std::uint64_t verdict = c.reserved[2];
            if (c.reserved[1] == kModeParts) {
                if (c.bytes > C) {
                    std::size_t at = C;
                    for (; c.bytes - at > C; at += C) fwd.exchange_host(zero(C), std::as_writable_bytes(frame.subspan(64U + at, C)));
                    const std::size_t n = c.bytes - at;
                    fwd.exchange_host(zero(n + 64U), std::as_writable_bytes(frame.subspan(64U + at, n + 64U)));
                    std::memcpy(&verdict, frame.data() + 64 + c.bytes, 8);
                }
            } else if (c.bytes > C) {
                std::vector<std::byte> more(c.bytes - C);
                fwd.exchange_host(more, std::as_writable_bytes(frame.subspan(64U + C, c.bytes - C)));
            }
            if (verdict != kVerdictOk) {
                if (verdict != kVerdictRefused) throw std::runtime_error("pipeline TAIL rank 1: bad verdict");
                std::printf("rank 1: STEP of %llu bytes skipped (rank 0 refused it or lost the FRONT during it)\n", static_cast<unsigned long long>(c.bytes));
                std::fflush(stdout);
                continue;  // rank 0 refused the STEP (or lost the FRONT during it): skipped on both ranks
            }
        } else {
            fwd.exchange_host(zero(kPeerFrame), std::as_writable_bytes(frame.first(kPeerFrame)));
            std::memcpy(&c, frame.data(), 64);
            if (c.magic != StageControl{}.magic) throw std::runtime_error("pipeline TAIL rank 1: bad control block");
            const std::size_t this_frame = kPeerFrame;
            if (c.reserved[0] >= kMinPeerFrame && c.reserved[0] <= kMaxPeerFrame) kPeerFrame = c.reserved[0];  // the next frame
            if (64U + c.bytes > this_frame) {  // the payload follows (the buffer may move: the control block is read)
                frame = stage_buffer(fwd, c.bytes);
                std::vector<std::byte> more(c.bytes);
                fwd.exchange_host(more, std::as_writable_bytes(frame.subspan(64, c.bytes)));
            }
        }
        const auto payload = frame.subspan(64, c.bytes);
        if (c.kind == op_quit) return;
        if (c.kind == op_step) {
            const auto sm = pipe::decode_step(payload, geo, strix::deepseek41::DeviceForward::max_window_rows, true);
            (void)fwd.step_boundary(sm.boundary);
            if (sm.boundary.rows > 1U && (a.gather || a.op_hashes) && (sm.flags & pipe::kStepWantRowFnv) != 0U) (void)fwd.window_logits();
            last_tokens = sm.boundary.tokens;
            if (drafter && (sm.flags & pipe::kStepFill) != 0U) {  // fill rows: all committed (op_commit follows for R > 1)
                drafter->stage(fwd.captured_streams(), sm.boundary.rows, fwd.stream());
                context.insert(context.end(), sm.boundary.tokens.begin(), sm.boundary.tokens.end());
            }
        }
        else if (c.kind == op_commit) fwd.commit_window(c.arg);
        else if (c.kind == op_spec || c.kind == op_spec_begin) {
            if (!drafter || payload.size() < 20U) throw std::runtime_error("pipeline TAIL rank 1: speculative operation without the split drafter");
            std::uint32_t h[5];
            std::memcpy(h, payload.data(), 20);
            std::vector<std::uint32_t> published(h[4]);
            if (payload.size() != 20U + 4U * std::size_t{h[4]}) throw std::runtime_error("pipeline TAIL rank 1: speculative operation size");
            std::memcpy(published.data(), payload.data() + 20, 4U * std::size_t{h[4]});
            if (c.kind == op_spec_begin) {
                (void)drafter->begin(context, h[2]);
            } else {
                const std::uint32_t kept = h[0], rows = h[1];
                if (rows > 1U) fwd.commit_window(kept);
                drafter->stage(fwd.captured_streams(), kept, fwd.stream());
                drafter->add_vocabulary(published);
                context.insert(context.end(), last_tokens.begin(), last_tokens.begin() + kept);
            }
            if (h[3] != 0U) (void)drafter->draft(h[2], fwd.stream());  // collective with rank 0's draft
        }
        else if (c.kind == op_reset) {
            fwd.reset();
            if (drafter) {
                drafter->reset();
                context.clear();
            }
        }
        else if (c.kind == op_import) {
            const auto st = strix::deepseek41::deserialize_attention_state(payload);
            fwd.import_state(st);
            if (drafter) {
                drafter->reset();
                context = st.tokens;
            }
        }
        else throw std::runtime_error("pipeline TAIL rank 1: unknown operation");
    }
}

// Per-step timestamps of a TAIL session (rank 0): waiting for the STEP, its first byte, the whole payload, decoded and
// checked, forwarded to rank 1, the forward's step done (argmax, logits gathered when kept), RESULT sent.
struct StageTimeline {
    using Clock = std::chrono::steady_clock;
    struct Row {
        std::uint32_t position = 0, rows = 0;
        std::uint64_t bytes = 0;
        Clock::time_point wait, first, recv, decoded, peer, computed, sent;
        double device_step = 0.0;  // DeviceForward::last_step_seconds (inside the step: uploads .. final sync)
        // the rank-1 forward (communicator counters): every exchange of it, and the part after the last byte received
        // (cut-through; the whole forward otherwise); parts = exchanges of the forward
        PeerCost fwd_total, fwd_exposed;
        std::size_t fwd_parts = 0;
    };
    std::vector<Row> rows;
    void add(const Row& r) { rows.push_back(r); }
    static double us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
    void summary(const std::string& csv) const {
        if (rows.empty()) return;
        if (!csv.empty()) {
            std::ofstream f(csv, std::ios::app);
            f << "position,rows,bytes,idle_us,recv_us,decode_us,peer_us,compute_us,send_us,tail_us,device_step_us,fwd_parts,fwd_exchanges,"
                 "fwd_total_us,fwd_stage_us,fwd_wire_us,fwd_check_us,exposed_us,exposed_stage_us,exposed_wire_us,exposed_check_us\n";
            for (const auto& r : rows)
                f << r.position << ',' << r.rows << ',' << r.bytes << ',' << us(r.wait, r.first) << ',' << us(r.first, r.recv) << ','
                  << us(r.recv, r.decoded) << ',' << us(r.decoded, r.peer) << ',' << us(r.peer, r.computed) << ',' << us(r.computed, r.sent) << ','
                  << us(r.first, r.sent) << ',' << r.device_step * 1e6 << ',' << r.fwd_parts << ',' << r.fwd_total.exchanges << ','
                  << r.fwd_total.seconds * 1e6 << ',' << r.fwd_total.stage * 1e6 << ',' << r.fwd_total.wire * 1e6 << ',' << r.fwd_total.check * 1e6 << ','
                  << r.fwd_exposed.seconds * 1e6 << ',' << r.fwd_exposed.stage * 1e6 << ',' << r.fwd_exposed.wire * 1e6 << ','
                  << r.fwd_exposed.check * 1e6 << '\n';
        }
        // medians over the steps of the most common row count after the first two (one-time costs)
        std::map<std::uint32_t, std::size_t> by_rows;
        for (std::size_t i = 2; i < rows.size(); ++i) ++by_rows[rows[i].rows];
        std::uint32_t R = 1;
        std::size_t most = 0;
        for (const auto& [r, n] : by_rows)
            if (n > most) most = n, R = r;
        std::vector<std::array<double, 7>> v;
        for (std::size_t i = 2; i < rows.size(); ++i)
            if (rows[i].rows == R) {
                const auto& r = rows[i];
                v.push_back({us(r.first, r.recv), us(r.recv, r.decoded), us(r.decoded, r.peer), us(r.peer, r.computed), us(r.computed, r.sent),
                             us(r.first, r.sent), r.device_step * 1e6});
            }
        if (v.empty()) return;
        std::array<double, 7> med{};
        for (std::size_t k = 0; k < 7; ++k) {
            std::vector<double> c;
            for (const auto& x : v) c.push_back(x[k]);
            std::nth_element(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(c.size() / 2U), c.end());
            med[k] = c[c.size() / 2U];
        }
        double mean_tail = 0.0;
        for (const auto& x : v) mean_tail += x[5];
        mean_tail /= static_cast<double>(v.size());
        std::printf("tail timeline (R%u steps, median of %zu, us): recv %.1f decode %.1f peer %.1f compute %.1f send %.1f | tail %.1f (mean %.1f) | "
                    "device step %.1f\n",
                    R, v.size(), med[0], med[1], med[2], med[3], med[4], med[5], mean_tail, med[6]);
    }
};

// The speculative side of a TAIL session (rank 0): drafter, window controller, decision log, the context since the last
// reset / import (vocabulary choice) and the costs the controller sees.
namespace ds = strix::ds41::dspark;
struct SpecTail {
    std::unique_ptr<ds::SpecTailDrafter> drafter;
    ds::WindowController ctl;
    bool use_ctl = true;
    std::uint32_t fixed = ds::block_size;
    pipe::SpecDecisionLog log;
    std::vector<std::uint32_t> context;  // tokens committed since the last reset / import (the draft vocabulary's context)
    std::vector<std::uint32_t> eos;
    std::uint32_t remaining = 0, next = 0, flags = 0, fault_drop = 0, decisions = 0;
    // the last plan: its draft (observed after the verify), drafted or not, when it was sent
    ds::Draft draft{};
    bool drafted = false;
    std::uint32_t k = 0;
    std::chrono::steady_clock::time_point plan_sent{};
    // controller costs (seconds): a prior of 4 samples per tabled size, then every measured window
    std::array<double, ds::block_size + 2U> cost_sum{};
    std::array<std::uint32_t, ds::block_size + 2U> cost_n{};
    double draft_sum = 0.0, extra_sum = 0.0;
    std::uint32_t draft_n = 0;
    // session statistics
    std::uint32_t windows = 0, drafted_windows = 0, published = 0, dup = 0, stale = 0, dropped = 0;
    std::map<std::uint32_t, std::pair<std::uint32_t, double>> by_rows;

    void push_costs() {
        auto c = ctl.costs();
        for (std::uint32_t r = 1; r <= ds::block_size + 1U; ++r) c.verify[r] = cost_n[r] ? static_cast<float>(cost_sum[r] / cost_n[r]) : 0.0F;
        c.draft = draft_n ? static_cast<float>(draft_sum / draft_n) : 0.0F;
        c.extra = draft_n ? static_cast<float>(extra_sum / draft_n) : 0.0F;
        ctl.set_costs(c);
    }
    bool split = false;  // rank 1 drafts with rank 0 (it learns each decision through op_spec before the draft)
    // Whether the next window is drafted (the controller's gate; fixed:N drafts unless N = 0).
    bool decide() {
        drafted = use_ctl ? ctl.draft_next() : fixed != 0U;
        return drafted;
    }
    // The plan after the committed position, after decide(): [next] or [next, d_1..d_k]
    std::vector<std::uint32_t> plan(void* stream, float& draft_ms) {
        const auto t0 = std::chrono::steady_clock::now();
        k = 0;
        draft_ms = 0.0F;
        if (drafted) {
            draft = drafter->draft(next, stream);
            const double ds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            draft_ms = static_cast<float>(ds_ * 1e3);
            draft_sum += ds_;
            ++draft_n;
            k = use_ctl ? ctl.choose(draft) : fixed;
        }
        k = std::min<std::uint32_t>(k, remaining - 1U);  // no rows past the end
        std::vector<std::uint32_t> p{next};
        for (std::uint32_t j = 1; j <= k; ++j) p.push_back(draft.tokens[j]);
        return p;
    }
};

void serve_stage(const Args& a, strix::deepseek41::DeviceForward& fwd, std::uint32_t world, strix::dist::Communicator* comm) {
    const auto model = strix::deepseek41::validate_target(strix::deepseek41::read_file(a.model));
    const auto& c = model.config;
    const auto geo = pipe::boundary_geometry(c);
    const auto identity = model.manifest.layout_identity;
    const auto token = pipe::read_token_file(a.stage_token_file);
    const auto [host, port] = pipe::split_host_port(a.stage_listen);
    auto listener = std::make_unique<pipe::Listener>(host, port);
    const bool full = a.gather || a.op_hashes;
    // the drafter (both ranks with the split placement) is loaded before the startup exchange: READY means loaded
    SpecTail spec;
    // a verify window must be the next window of an open speculative sequence (VERIFY flag <=> request with an epoch)
    const auto spec_check = [&](const pipe::StepMsg& sm, std::uint64_t request) {
        const bool verify = (sm.flags & pipe::kStepVerify) != 0U;
        if (verify != (pipe::spec_epoch(request) != 0U)) throw pipe::WireError("a verify STEP carries its epoch and window in the request id");
        if ((sm.flags & pipe::kStepFill) != 0U && verify) throw pipe::WireError("a STEP is either fill or verify");
        if (!verify) return;
        if (!spec.drafter) throw pipe::WireError("verify window: this TAIL has no drafter");
        if (spec.log.classify(request) != pipe::SpecDecisionLog::Kind::fresh) throw pipe::WireError("verify window outside the open sequence");
        if (sm.boundary.rows > ds::block_size + 1U) throw pipe::WireError("verify window of more than 6 rows");
    };
    if (!a.stage_dspark.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        spec.drafter = std::make_unique<ds::SpecTailDrafter>(a.model, a.stage_dspark, a.stage_dspark_vocab);
        if (a.stage_dspark_placement != "split" && a.stage_dspark_placement != "rank0") throw std::invalid_argument("--stage-dspark-placement split | rank0");
        spec.split = world == 2U && a.stage_dspark_placement == "split";
        if (spec.split) spec.drafter->set_split(0U, *comm);
        if (a.stage_dspark_policy.rfind("fixed:", 0) == 0) {
            spec.use_ctl = false;
            spec.fixed = static_cast<std::uint32_t>(std::stoul(a.stage_dspark_policy.substr(6)));
            if (spec.fixed > ds::block_size) throw std::invalid_argument("--stage-dspark-policy fixed:0..5");
        } else if (a.stage_dspark_policy != "controller") {
            throw std::invalid_argument("--stage-dspark-policy controller | fixed:N");
        }
        if (!a.stage_dspark_costs.empty()) {
            const auto t = pipe::read_spec_cost_table(a.stage_dspark_costs);
            for (std::uint32_t r = 1; r <= ds::block_size + 1U; ++r)
                if (t.verify[r] > 0.0) spec.cost_sum[r] = 4.0 * t.verify[r], spec.cost_n[r] = 4U;
            spec.draft_sum = 4.0 * (t.draft > 0.0 ? t.draft : 7.48e-3);
            spec.extra_sum = 4.0 * (t.extra > 0.0 ? t.extra : 0.57e-3);
            spec.draft_n = 4U;
            std::printf("dspark cost prior (%s): V1..V6", t.label.c_str());
            for (std::uint32_t r = 1; r <= ds::block_size + 1U; ++r) std::printf(" %.2f", spec.cost_n[r] ? spec.cost_sum[r] / spec.cost_n[r] * 1e3 : 0.0);
            std::printf(" ms, D %.2f, A %.2f ms\n", spec.draft_sum / 4.0 * 1e3, spec.extra_sum / 4.0 * 1e3);
        }
        spec.push_costs();
        spec.fault_drop = a.stage_fault_drop_accept;
        std::printf("dspark drafter loaded in %.1f s (%s; vocabulary %s, policy %s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    spec.split ? "both ranks draft, the head split by vocabulary" : "rank 0 drafts alone", a.stage_dspark_vocab.c_str(), a.stage_dspark_policy.c_str());
        std::fflush(stdout);
    }
    if (world == 2U) {  // both ranks loaded before the first session (rank 1 enters follow_stage)
        std::array<std::byte, 64> out{}, in{};
        fwd.exchange_host(out, in);
    }
    std::printf("READY {\"role\":\"pipeline-stage\",\"layers\":[%u,%u],\"listen\":\"%s:%u\",\"world\":%u,\"protocol\":\"%s\",\"model_identity\":\"%016llx\"}\n",
                a.first_layer, a.layers, host.c_str(), listener->port(), world, pipe::kProtocol, static_cast<unsigned long long>(identity));
    std::fflush(stdout);
    const auto quit = [&] {
        if (world == 2U) to_peer(fwd, op_quit, 0, {});
    };
    // --stage-peer-thread: the cut-through parts go to rank 1 from a second thread (persistent; idle between STEPs, so
    // every other exchange of this thread (commit, reset, import, quit, the step's collectives) never overlaps it)
    std::unique_ptr<PeerForwarder> forwarder;
    unsigned sessions = 0;
    if (world == 2U && kPeerChunk != 0U && a.stage_peer_thread) forwarder = std::make_unique<PeerForwarder>(fwd);
    for (;;) {
        auto conn = listener->accept(std::chrono::milliseconds(0), std::chrono::seconds(a.stage_timeout));
        if (!conn) continue;
        std::printf("session from %s\n", conn->peer().c_str());
        std::fflush(stdout);
        bool open = true;
        try {
            const auto first = conn->receive(1U << 20U);
            if (first.header.type != pipe::MsgType::hello) throw pipe::WireError("expected HELLO");
            const auto h = pipe::decode_hello(first.payload);
            pipe::HelloAck ack;
            ack.first_layer = a.first_layer;
            ack.end_layer = a.layers;
            ack.world = world;
            ack.max_positions = a.max_positions;
            ack.model_identity = identity;
            ack.engine = "strix-deepseek41-rank";
            ack.numeric_identity = "ds41-exact-v1";
            if (!pipe::same_token(h.token, token)) ack.status = 1, ack.message = "wrong session token";
            else if (h.cut != a.first_layer) ack.status = 2, ack.message = "the FRONT cuts at " + std::to_string(h.cut) + ", this TAIL starts at " + std::to_string(a.first_layer);
            else if (h.layers != a.layers || h.hidden != c.hidden_size || h.hc != c.hc_mult || h.vocab != c.vocab_size)
                ack.status = 3, ack.message = "model geometry or layer count differs";
            else if (h.model_identity != identity) ack.status = 4, ack.message = "the model files differ (layout identity)";
            else if (h.max_positions > a.max_positions) ack.status = 5, ack.message = "the FRONT's max_positions exceeds the TAIL's";
            conn->send(pipe::MsgType::hello_ack, first.header.request, pipe::encode(ack));
            std::printf("hello from %s backend %s (%s): %s\n", conn->peer().c_str(), h.backend.c_str(), h.numeric_identity.c_str(),
                        ack.status == 0U ? "accepted" : ack.message.c_str());
            std::fflush(stdout);
            if (ack.status != 0U) open = false;
        } catch (const std::exception& e) {
            std::printf("session refused: %s\n", e.what());
            open = false;
        }
        conn->set_busy_poll(a.stage_busy_poll);
        StageTimeline timeline;
        auto t_wait = std::chrono::steady_clock::now();
        while (open) {
            pipe::Message m;  // payload: STEP in the forward's pinned buffer after the peer control block, else in m.payload
            std::span<const std::uint8_t> payload;
            std::span<std::uint8_t> frame;
            auto t_first = std::chrono::steady_clock::now();
            bool cut = false;  // cut-through: the STEP is received and forwarded to rank 1 part by part (below)
            try {
                m.header = conn->receive_header(pipe::kMaxMessageBytes, &t_first);
                std::span<std::uint8_t> into;
                // a speculative VERIFY that is not the next window (a resend after a lost decision, another epoch): its
                // payload is read and dropped here, the stored decision (or a stale status) goes back, nothing runs
                if (m.header.type == pipe::MsgType::step && spec.drafter && pipe::spec_epoch(m.header.request) != 0U &&
                    spec.log.classify(m.header.request) != pipe::SpecDecisionLog::Kind::fresh) {
                    m.payload.resize(static_cast<std::size_t>(m.header.bytes));
                    conn->receive_payload(m.payload);
                    if (spec.log.classify(m.header.request) == pipe::SpecDecisionLog::Kind::duplicate) {
                        ++spec.dup;
                        conn->send(pipe::MsgType::spec_accept, m.header.request, spec.log.last());
                        std::printf("verify window %u again: the stored decision sent again\n", pipe::spec_window(m.header.request));
                    } else {
                        ++spec.stale;
                        pipe::SpecAccept sa;
                        sa.epoch = pipe::spec_epoch(m.header.request);
                        sa.window = pipe::spec_window(m.header.request);
                        sa.status = pipe::SpecStatus::stale;
                        conn->send(pipe::MsgType::spec_accept, m.header.request, pipe::encode(sa));
                        std::printf("verify epoch %u window %u is stale (epoch %u, last window %u): not run\n", sa.epoch, sa.window, spec.log.epoch(),
                                    spec.log.window());
                    }
                    std::fflush(stdout);
                    t_wait = std::chrono::steady_clock::now();
                    continue;
                }
                if (m.header.type == pipe::MsgType::step && m.header.bytes <= (std::uint64_t{64} << 20U)) {
                    frame = stage_buffer(fwd, static_cast<std::size_t>(m.header.bytes));
                    into = frame.subspan(64, static_cast<std::size_t>(m.header.bytes));
                    cut = world == 2U && kPeerChunk != 0U;
                } else {
                    m.payload.resize(static_cast<std::size_t>(m.header.bytes));
                    into = m.payload;
                }
                if (!cut) conn->receive_payload(into);
                payload = into;
            } catch (const std::exception& e) {
                std::printf("session ended: %s\n", e.what());
                break;
            }
            CutThrough ct;
            pipe::StepMsg ct_sm;
            std::string ct_why;
            if (cut) {
                ct = receive_forward_step(*conn, fwd, frame, static_cast<std::size_t>(m.header.bytes), [&] {
                    try {
                        ct_sm = pipe::decode_step(payload, geo, strix::deepseek41::DeviceForward::max_window_rows, true);
                        pipe::validate_boundary(ct_sm.boundary, fwd.boundary_plan(), geo);
                        if (ct_sm.boundary.position != fwd.next_position())
                            throw pipe::WireError("step at position " + std::to_string(ct_sm.boundary.position) + ", the TAIL is at " +
                                                  std::to_string(fwd.next_position()));
                        if (ct_sm.boundary.position + ct_sm.boundary.rows > a.max_positions) throw pipe::WireError("context full");
                        spec_check(ct_sm, m.header.request);
                        return true;
                    } catch (const std::exception& e) {
                        ct_why = e.what();
                        return false;
                    }
                }, forwarder.get());
                if (!ct.receive_error.empty()) {
                    std::printf("session ended: %s\n", ct.receive_error.c_str());
                    break;
                }
            }
            const auto t_recv = cut ? ct.recv : std::chrono::steady_clock::now();
            const auto id = m.header.request;
            const auto fail = [&](const std::string& why) {
                std::printf("request %llu refused: %s\n", static_cast<unsigned long long>(id), why.c_str());
                conn->send(pipe::MsgType::error, id, pipe::encode_text(why));
                open = false;
            };
            // Requests are checked before anything reaches rank 1; a failure after that would leave the ranks apart: fatal.
            switch (m.header.type) {
            case pipe::MsgType::step: {
                pipe::StepMsg sm;
                if (cut && !ct.passed) {  // rank 1 skipped it too (verdict)
                    fail(ct_why);
                    break;
                }
                if (cut) sm = std::move(ct_sm);  // decoded and checked before the last part went to rank 1
                else try {
                    sm = pipe::decode_step(payload, geo, strix::deepseek41::DeviceForward::max_window_rows, true);
                    pipe::validate_boundary(sm.boundary, fwd.boundary_plan(), geo);
                    if (sm.boundary.position != fwd.next_position()) throw pipe::WireError("step at position " + std::to_string(sm.boundary.position) +
                                                                                           ", the TAIL is at " + std::to_string(fwd.next_position()));
                    if (sm.boundary.position + sm.boundary.rows > a.max_positions) throw pipe::WireError("context full");
                    spec_check(sm, id);
                } catch (const std::exception& e) {
                    fail(e.what());
                    break;
                }
                const auto t_decoded = cut ? ct.checked : std::chrono::steady_clock::now();
                PeerCost fwd_total = ct.total;
                if (world == 2U && !cut) {
                    const auto before = peer_stats();
                    to_peer(fwd, op_step, 0, frame, payload.size());
                    add_cost(fwd_total, before);
                }
                const auto t_peer = cut ? ct.forwarded : std::chrono::steady_clock::now();
                const bool verify = (sm.flags & pipe::kStepVerify) != 0U, fill = (sm.flags & pipe::kStepFill) != 0U;
                const auto tokens = fwd.step_boundary(sm.boundary);
                const auto t_computed = std::chrono::steady_clock::now();
                const auto& logits = fwd.last_logits();
                pipe::Result r;
                r.position = sm.boundary.position;
                r.tokens = tokens;
                r.logits_fnv.assign(tokens.size(), 0U);
                const bool have = sm.boundary.rows == 1U && full && !logits.empty();
                if (have) r.logits_fnv[0] = pipe::fnv_floats(logits);
                std::vector<float> rows_logits;  // R > 1 with kStepWantRowFnv: every row's logits (both ranks gather)
                if (sm.boundary.rows > 1U && full && (sm.flags & pipe::kStepWantRowFnv) != 0U) {
                    rows_logits = fwd.window_logits();
                    const std::size_t V = rows_logits.size() / sm.boundary.rows;
                    for (std::uint32_t row = 0; row < sm.boundary.rows; ++row)
                        r.logits_fnv[row] = pipe::fnv_floats(std::span<const float>(rows_logits).subspan(row * V, V));
                }
                if (have && (sm.flags & pipe::kStepWantLogits) != 0U) r.logits = logits;
                const std::uint32_t R = sm.boundary.rows;
                std::uint32_t peer_commit = 0;  // rows rank 1 commits after the answer (R > 1)
                if (verify) {
                    // ---- the decision of window w and the plan of w + 1
                    const auto td = std::chrono::steady_clock::now();
                    const double vs = std::chrono::duration<double>(t_computed - spec.plan_sent).count();  // FRONT step + wire + TAIL
                    const auto o = pipe::spec_outcome(sm.boundary.tokens, tokens, spec.remaining, spec.eos);
                    if (R > 1U) fwd.commit_window(o.kept);
                    if (R > 1U && world == 2U && !spec.split) peer_commit = o.kept;
                    spec.drafter->stage(fwd.captured_streams(), o.kept, fwd.stream());
                    spec.drafter->add_vocabulary(o.published);
                    spec.context.insert(spec.context.end(), sm.boundary.tokens.begin(), sm.boundary.tokens.begin() + o.kept);
                    if (spec.drafted) spec.ctl.observe(spec.draft, spec.k, o.accepted);
                    if (R <= ds::block_size + 1U) {
                        spec.cost_sum[R] += vs, ++spec.cost_n[R];
                        auto costs = spec.ctl.costs();
                        ++costs.samples[R];
                        spec.ctl.set_costs(costs);
                    }
                    auto& br = spec.by_rows[R];
                    ++br.first;
                    br.second += vs;
                    spec.remaining -= static_cast<std::uint32_t>(o.published.size());
                    spec.published += static_cast<std::uint32_t>(o.published.size());
                    spec.next = o.published.back();
                    ++spec.windows;
                    if (spec.drafted) ++spec.drafted_windows;
                    pipe::SpecAccept sa;
                    sa.epoch = pipe::spec_epoch(id);
                    sa.window = pipe::spec_window(id);
                    sa.position = sm.boundary.position;
                    sa.rows = R;
                    sa.kept = o.kept;
                    sa.argmax = tokens;
                    sa.row_fnv = r.logits_fnv;
                    sa.committed = fwd.next_position();
                    sa.next_token = spec.next;
                    sa.published = o.published;
                    sa.done = o.done;
                    const auto tc = std::chrono::steady_clock::now();
                    bool told = false;  // split: rank 1 got this decision (op_spec)
                    if (o.done == pipe::SpecDone::more) {
                        spec.push_costs();
                        const bool d = spec.decide();
                        if (spec.split) {
                            to_peer(fwd, op_spec, 0, spec_op_payload(o.kept, R, spec.next, d, o.published));
                            told = true;
                        }
                        sa.plan = spec.plan(fwd.stream(), sa.draft_ms);
                    }
                    const double extra = std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count() - sa.draft_ms * 1e-3;
                    if (spec.drafted) spec.extra_sum += extra;
                    sa.control_ms = static_cast<float>(extra * 1e3);
                    (void)tc;
                    auto bytes = pipe::encode(sa);
                    spec.log.record(sa.window, bytes);
                    ++spec.decisions;
                    if (spec.fault_drop != 0U && spec.decisions % spec.fault_drop == 0U) {
                        ++spec.dropped;
                        std::printf("fault: decision of window %u not sent\n", sa.window);
                    } else {
                        conn->send(pipe::MsgType::spec_accept, id, bytes);
                    }
                    spec.plan_sent = std::chrono::steady_clock::now();
                    if (spec.split && !told) to_peer(fwd, op_spec, 0, spec_op_payload(o.kept, R, spec.next, false, o.published));  // the last window
                    std::printf("window %u pos %u R %u kept %u published", sa.window, sa.position, R, o.kept);
                    for (const auto t : o.published) std::printf(" %u", t);
                    std::printf(" | next plan R %zu (drafted %d) draft %.2f ms control %.2f ms verify cycle %.2f ms%s\n", sa.plan.size(), spec.drafted ? 1 : 0,
                                sa.draft_ms, sa.control_ms, vs * 1e3, o.done == pipe::SpecDone::eos ? " EOS" : o.done == pipe::SpecDone::max_new ? " max_new" : "");
                    if (o.done != pipe::SpecDone::more) spec.log.finish();
                } else {
                    if (fill && R > 1U) {
                        fwd.commit_window(R);
                        if (world == 2U) peer_commit = R;
                    }
                    conn->send(pipe::MsgType::result, id, pipe::encode(r));  // first: the FRONT waits for it
                    if (spec.drafter) {
                        spec.drafter->stage(fwd.captured_streams(), R, fwd.stream());
                        spec.context.insert(spec.context.end(), sm.boundary.tokens.begin(), sm.boundary.tokens.end());
                        spec.next = tokens.back();
                    }
                }
                if (peer_commit) to_peer(fwd, op_commit, peer_commit, {});  // after the answer: rank 1 is idle until the next STEP
                const auto t_sent = std::chrono::steady_clock::now();
                timeline.add({sm.boundary.position, sm.boundary.rows, m.header.bytes, t_wait, t_first, t_recv, t_decoded, t_peer, t_computed, t_sent,
                              fwd.last_step_seconds(), fwd_total, cut ? ct.exposed : fwd_total, cut ? ct.parts : (fwd_total.exchanges ? 1U : 0U)});
                for (std::uint32_t row = 0; row < sm.boundary.rows; ++row) {
                    const double top = have ? static_cast<double>(logits[tokens[row]])
                        : !rows_logits.empty() ? static_cast<double>(rows_logits[row * (rows_logits.size() / sm.boundary.rows) + tokens[row]]) : 0.0;
                    std::printf("pos %u fed %u argmax %u top %.4f logits_fnv %016llx%s\n", sm.boundary.position + row, sm.boundary.tokens[row], tokens[row],
                                top, static_cast<unsigned long long>(r.logits_fnv[row]), sm.boundary.rows > 1U ? " (window)" : "");
                }
                if (a.op_hashes)
                    for (const auto& [name, hash] : fwd.last_step_hashes()) std::printf("  %s %016llx\n", name.c_str(), static_cast<unsigned long long>(hash));
                if (have && !a.stage_logits.empty()) {  // quality gates (KL against another identity), after the RESULT
                    static std::ofstream logits_dump(a.stage_logits, std::ios::binary | std::ios::app);
                    const std::uint32_t pos = sm.boundary.position;
                    logits_dump.write(reinterpret_cast<const char*>(&pos), 4);
                    logits_dump.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * 4U));
                    logits_dump.flush();
                }
                std::fflush(stdout);
                t_wait = std::chrono::steady_clock::now();
                break;
            }
            case pipe::MsgType::commit: {
                std::uint32_t kept = 0;
                try {
                    kept = pipe::decode_u32(payload);
                    if (kept == 0U || kept > strix::deepseek41::DeviceForward::max_window_rows) throw pipe::WireError("commit keeps 1..8 rows");
                } catch (const std::exception& e) {
                    fail(e.what());
                    break;
                }
                if (world == 2U) to_peer(fwd, op_commit, kept, {});
                fwd.commit_window(kept);
                conn->send(pipe::MsgType::ack, id, {});
                break;
            }
            case pipe::MsgType::reset:
                if (world == 2U) to_peer(fwd, op_reset, 0, {});
                fwd.reset();
                if (spec.drafter) {
                    spec.drafter->reset();
                    spec.context.clear();
                    spec.log.end();
                }
                std::printf("reset\n");
                conn->send(pipe::MsgType::ack, id, {});
                break;
            case pipe::MsgType::export_state: {
                const auto st = fwd.export_state();
                pipe::StateMsg sm;
                sm.manifest = pipe::state_manifest(st, a.first_layer, a.layers, identity, "strix-deepseek41-rank");
                sm.state = strix::deepseek41::serialize_attention_state(st);
                std::printf("export state: position %u, %zu bytes, digest %016llx\n", st.position, sm.state.size(),
                            static_cast<unsigned long long>(st.digest()));
                conn->send(pipe::MsgType::state, id, pipe::encode(sm));
                break;
            }
            case pipe::MsgType::import_state: {
                pipe::StateMsg sm;
                strix::deepseek41::AttentionState st;
                try {
                    sm = pipe::decode_state(payload);
                    st = strix::deepseek41::deserialize_attention_state(sm.state);
                    if (st.position + 1U > a.max_positions) throw pipe::WireError("the state leaves no room in max_positions");
                    if (st.layers.size() < a.layers) throw pipe::WireError("the state has fewer layers than the TAIL");
                } catch (const std::exception& e) {
                    fail(e.what());
                    break;
                }
                if (world == 2U) to_peer(fwd, op_import, 0, sm.state);
                fwd.import_state(st);
                if (spec.drafter) {  // the drafter starts cold at an imported state (its context: the state's tokens)
                    spec.drafter->reset();
                    spec.context = st.tokens;
                    spec.log.end();
                }
                std::printf("import state: position %u, digest %016llx\n", st.position, static_cast<unsigned long long>(st.digest()));
                conn->send(pipe::MsgType::ack, id, {});
                break;
            }
            case pipe::MsgType::spec_begin: {
                pipe::SpecBegin b;
                try {
                    b = pipe::decode_spec_begin(payload);
                    if (b.epoch == 0U) throw pipe::WireError("SPEC_BEGIN: epoch 0 is reserved");
                    if (b.max_new < 2U) throw pipe::WireError("SPEC_BEGIN: max_new >= 2");
                } catch (const std::exception& e) {
                    fail(e.what());
                    break;
                }
                pipe::SpecAccept sa;
                sa.epoch = b.epoch;
                sa.window = 0;
                if (!spec.drafter) {
                    sa.status = pipe::SpecStatus::off;  // the FRONT decodes with one-row STEPs
                    conn->send(pipe::MsgType::spec_accept, id, pipe::encode(sa));
                    std::printf("spec begin refused: no drafter (fallback: one-row steps)\n");
                    break;
                }
                if (b.position != fwd.next_position()) {
                    sa.status = pipe::SpecStatus::refused;
                    conn->send(pipe::MsgType::spec_accept, id, pipe::encode(sa));
                    std::printf("spec begin refused: position %u, the TAIL is at %u\n", b.position, fwd.next_position());
                    break;
                }
                spec.log.begin(b.epoch);
                spec.remaining = b.max_new - 1U;  // next_token is published by the FRONT already
                spec.next = b.next_token;
                spec.eos = b.eos;
                spec.flags = b.flags;
                spec.windows = spec.drafted_windows = spec.published = spec.dup = spec.stale = spec.dropped = 0;
                spec.by_rows.clear();
                const auto label = spec.drafter->begin(spec.context, b.next_token);
                spec.push_costs();
                sa.committed = fwd.next_position();
                sa.next_token = spec.next;
                {
                    const bool d = spec.decide();
                    if (spec.split) to_peer(fwd, op_spec_begin, 0, spec_op_payload(0U, 0U, spec.next, d, {}));
                }
                sa.plan = spec.plan(fwd.stream(), sa.draft_ms);
                auto bytes = pipe::encode(sa);
                spec.log.record(0, bytes);
                conn->send(pipe::MsgType::spec_accept, id, bytes);
                spec.plan_sent = std::chrono::steady_clock::now();
                std::printf("spec begin epoch %u at %u, next %u, max_new %u, vocabulary %s, context %zu: plan R %zu\n", b.epoch, b.position, b.next_token,
                            b.max_new, label.c_str(), spec.context.size(), sa.plan.size());
                break;
            }
            case pipe::MsgType::spec_commit: {
                std::pair<std::uint32_t, std::uint32_t> wm;
                try {
                    wm = pipe::decode_spec_commit(payload);
                } catch (const std::exception& e) {
                    fail(e.what());
                    break;
                }
                const bool agree = wm.first == spec.log.epoch() && wm.second == spec.log.window();
                std::printf("spec commit epoch %u window %u (TAIL: epoch %u window %u%s), position %u; windows %u (drafted %u), published %u, "
                            "duplicates %u, stale %u, dropped %u\n",
                            wm.first, wm.second, spec.log.epoch(), spec.log.window(), agree ? "" : ", DIFFERENT", fwd.next_position(), spec.windows,
                            spec.drafted_windows, spec.published, spec.dup, spec.stale, spec.dropped);
                for (const auto& [R, v] : spec.by_rows) std::printf("  verify cycle R%u: %u windows, %.2f ms each\n", R, v.first, v.second * 1e3 / v.first);
                spec.log.end();
                conn->send(pipe::MsgType::ack, id, pipe::encode_u32(fwd.next_position()));
                break;
            }
            case pipe::MsgType::ping: {
                char buf[256];
                std::snprintf(buf, sizeof(buf), "{\"status\":\"ok\",\"role\":\"pipeline-stage\",\"layers\":[%u,%u],\"position\":%u,\"world\":%u}", a.first_layer,
                              a.layers, fwd.next_position(), world);
                conn->send(pipe::MsgType::pong, id, pipe::encode_text(buf));
                break;
            }
            case pipe::MsgType::close:
                if (a.stage_once || (a.stage_sessions != 0U && sessions + 1U >= a.stage_sessions)) listener.reset();  // before the ACK: the FRONT's next session never reaches this TAIL
                conn->send(pipe::MsgType::ack, id, {});
                open = false;
                break;
            default: fail(std::string("unexpected ") + pipe::to_string(m.header.type));
            }
            std::fflush(stdout);
        }
        timeline.summary(a.stage_timeline);
        std::printf("session closed\n");
        std::fflush(stdout);
        ++sessions;
        if (a.stage_once || (a.stage_sessions != 0U && sessions >= a.stage_sessions)) break;
    }
    listener.reset();  // (if still open) closed before rank 1's quit
    quit();
}

// --role pipeline-record: this group runs layers [0, layers) without the head over the prompt ids (one row per step) and
// rank 0 appends each step's boundary to the record file: "DS4R" u32 version 1, the HELLO a FRONT would send (token
// zeroed) as u64 length + bytes, then per step u64 length + the STEP payload.
void record_front(const Args& a, strix::deepseek41::DeviceForward& fwd, std::uint32_t rank, std::uint32_t world, bool print) {
    const auto model = strix::deepseek41::validate_target(strix::deepseek41::read_file(a.model));
    const auto& c = model.config;
    std::vector<std::uint32_t> seq;
    // --import-state FILE: the record starts from that state (its next token first) and is written as a message log
    // ("DS4M" v1: u32 type + u64 length + payload; HELLO, IMPORT_STATE, the STEPs, CLOSE) for pipeline-replay --msg-log
    const bool msg_log = !a.import_state.empty();
    std::vector<std::uint8_t> state_bytes;
    std::uint32_t state_position = 0;
    if (msg_log) {
        const auto state = strix::deepseek41::load_attention_state(a.import_state);
        fwd.import_state(state);
        seq.push_back(state.next_token);
        state_position = state.position;
        state_bytes = strix::deepseek41::serialize_attention_state(state);
    }
    std::stringstream ss(a.prompt);
    if (a.prompt != "-")
        for (std::string t; std::getline(ss, t, ',');) seq.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    std::ofstream out;
    pipe::MsgType put_type = pipe::MsgType::step;
    const auto put = [&](std::span<const std::uint8_t> bytes) {
        if (msg_log) {
            const auto t = static_cast<std::uint32_t>(put_type);
            out.write(reinterpret_cast<const char*>(&t), 4);
        }
        const std::uint64_t n = bytes.size();
        out.write(reinterpret_cast<const char*>(&n), 8);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(n));
    };
    if (rank == 0U) {
        out.open(a.record, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write " + a.record);
        const std::uint32_t head[2] = {msg_log ? 0x4D345344U : 0x52345344U, 1U};  // "DS4M" / "DS4R", version
        out.write(reinterpret_cast<const char*>(head), 8);
        pipe::Hello h;
        h.cut = a.layers;
        h.layers = c.num_hidden_layers;
        h.hidden = c.hidden_size;
        h.hc = c.hc_mult;
        h.vocab = c.vocab_size;
        h.max_positions = a.max_positions;
        h.model_identity = model.manifest.layout_identity;
        h.backend = "hip-recorded";
        h.numeric_identity = "ds41-exact-v1";
        put_type = pipe::MsgType::hello;
        put(pipe::encode(h));
        if (msg_log) {
            pipe::StateMsg sm;
            sm.manifest = "{\"layers\":[0," + std::to_string(c.num_hidden_layers) + "],\"position\":" + std::to_string(state_position) + "}";
            sm.state = state_bytes;
            put_type = pipe::MsgType::import_state;
            put(pipe::encode(sm));
        }
        put_type = pipe::MsgType::step;
    }
    // --window-rows R: the first token alone, then windows of R rows (every row committed: a verifier's shape)
    const std::uint32_t R = a.window_rows.empty() ? 1U : a.window_rows[0];
    if (R == 0U || R > strix::deepseek41::DeviceForward::max_window_rows) throw std::invalid_argument("--window-rows 1..8");
    if (print)
        std::printf("recording the boundary at cut %u (world %u) over %zu tokens in steps of %u rows into %s\n", a.layers, world, seq.size(), R,
                    a.record.c_str());
    double seconds = 0.0;
    std::uint64_t bytes = 0;
    std::size_t steps = 0;
    for (std::size_t i = 0; i < seq.size();) {
        const std::size_t n = i == 0U ? 1U : std::min<std::size_t>(R, seq.size() - i);
        (void)fwd.step_window(std::span<const std::uint32_t>(seq).subspan(i, n));
        if (steps >= 1U) seconds += fwd.last_step_seconds();
        if (rank == 0U) {
            pipe::StepMsg sm;
            sm.boundary = fwd.export_boundary();
            const auto payload = pipe::encode(sm);
            bytes += payload.size();
            put(payload);
        }
        if (n > 1U) fwd.commit_window(static_cast<std::uint32_t>(n));
        i += n;
        ++steps;
    }
    if (rank == 0U && msg_log) {
        put_type = pipe::MsgType::close;
        put({});
    }
    if (print)
        std::printf("recorded %zu steps, %.1f KB/step, FRONT [0, %u) %.2f ms/step (steps after the first)\n", steps,
                    steps ? static_cast<double>(bytes) / 1e3 / static_cast<double>(steps) : 0.0, a.layers,
                    steps > 1U ? seconds * 1e3 / static_cast<double>(steps - 1U) : 0.0);
}

// --routes-seqs FILE: one sequence per line, "NAME<TAB>id,id,..." (or just the ids). Each sequence starts from a
// reset state: the first row alone, then teacher-forced verify windows of up to 8 rows, every row committed (each
// row bitwise the row of single steps). Per sequence the routes file gets "# seq NAME rows N" and then one line per
// row in the --routes format; it is flushed after every sequence, so a run cut by its time limit keeps whole ones.
void record_route_sequences(const Args& a, strix::deepseek41::DeviceForward& fwd, std::ofstream* out) {
    std::ifstream f(a.routes_seqs);
    if (!f) throw std::invalid_argument("--routes-seqs: cannot open " + a.routes_seqs);
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t done = 0, rows = 0;
    for (std::string line; std::getline(f, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::string name = std::to_string(done);
        const auto tab = line.find('\t');
        if (tab != std::string::npos) {
            name = line.substr(0, tab);
            line = line.substr(tab + 1);
        }
        std::vector<std::uint32_t> ids;
        std::stringstream ss(line);
        for (std::string t; std::getline(ss, t, ',');)
            if (!t.empty()) ids.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        if (ids.empty()) continue;
        if (ids.size() > a.max_positions) ids.resize(a.max_positions);
        fwd.reset();
        std::string text = "# seq " + name + " rows " + std::to_string(ids.size()) + "\n";
        const auto emit = [&](std::size_t R) {
            const auto& routes = fwd.last_step_routes();
            const std::size_t K = routes.empty() || R == 0 ? 0 : routes[0].size() / R;
            for (std::size_t r = 0; r < R; ++r) {
                for (std::size_t l = 0; l < routes.size(); ++l) {
                    text += (l ? " " : "") + std::to_string(l) + ':';
                    for (std::size_t k = 0; k < K; ++k) text += (k ? "," : "") + std::to_string(routes[l][r * K + k]);
                }
                text += '\n';
            }
        };
        (void)fwd.step(ids[0]);
        emit(1);
        std::vector<std::uint32_t> window;
        for (std::size_t i = 1; i < ids.size();) {
            window.clear();
            while (window.size() < strix::deepseek41::DeviceForward::max_window_rows && i < ids.size()) window.push_back(ids[i++]);
            (void)fwd.step_window(window);
            emit(window.size());
            fwd.commit_window(static_cast<std::uint32_t>(window.size()));
        }
        ++done;
        rows += ids.size();
        if (out != nullptr) {
            *out << text;
            out->flush();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("routes seq %s: %zu rows (total %zu sequences, %zu rows, %.1f s, %.1f rows/s)\n", name.c_str(), ids.size(), done, rows, s,
                        static_cast<double>(rows) / s);
            std::fflush(stdout);
        }
    }
}

strix::deepseek41::SessionCompat rank_session_compat(const Args& a, std::uint32_t rank) {
    namespace ds = strix::deepseek41;
    std::map<std::string, std::string> cfg{{"world", "2"},
                                        {"rank", std::to_string(rank)},
                                        {"engine", "tp2-rank-target-r1"},
                                        {"layers", std::to_string(a.layers)},
                                        {"context", std::to_string(a.max_positions)},
                                        {"tp-attention", std::to_string(a.tp)},
                                        {"split-shared", std::to_string(a.split_shared)},
                                        {"engram-direct", std::to_string(a.engram_direct)},
                                        {"stream-experts", std::to_string(a.stream_experts)},
                                        {"gather", std::to_string(a.gather)},
                                        {"dspark", "off"},
                                        {"sampling", "greedy"},
                                        {"producer-plan", "target-r1-every-input-v1"},
                                        {"layout", "fp8-ring136x528-kv288-idx68-fp32carry-v1"}};
    if (a.session_dspark) {
        cfg["engine"] = "tp2-rank-dspark-r2";
        cfg["dspark"] = "on";
        cfg["producer-plan"] = "dspark-target-r1-append-every-row-v2";
        cfg["draft-schema"] = "strix/ds41-dspark-context-v2";
        cfg["draft-vocab"] = a.stage_dspark_vocab;
        cfg["draft-policy"] = a.stage_dspark_policy;
        cfg["draft-placement"] = "split";
        cfg["draft-cost-update"] = "frozen-prior-survival-adapts-v1";
    }
    std::map<std::string, std::filesystem::path> assets{{"model", a.model},
        {"draft", a.session_dspark ? a.stage_dspark : ""}};
    if (a.session_dspark) assets["draft-costs"] = a.stage_dspark_costs;
    if (a.session_dspark) {
        std::uint32_t i = 0;
        for (const auto& path : strix::ds41::dspark::vocabulary_asset_files(a.stage_dspark_vocab))
            assets["draft-vocab:" + std::to_string(i++)] = path;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto snapshot = strix::LoaderCapture::snapshot();
    auto result = ds::session_compat(std::move(cfg), assets, "/proc/self/exe",
        (a.session_hash_only || a.session_validate_only) ? nullptr : &snapshot);
    const auto stats = strix::file_hash_cache_stats();
    std::printf("session hashing rank %u seconds %.6f cache_hits %llu cache_misses %llu\n", rank,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
        static_cast<unsigned long long>(stats.hits), static_cast<unsigned long long>(stats.misses));
    return result;
}

void run_session(const Args &a, strix::deepseek41::DeviceForward &fwd, std::uint32_t rank, strix::dist::Communicator* comm) {
    namespace ds = strix::deepseek41;
    using Sessions = ds::PrefixCache::Sessions;
    const ds::SessionExchange ex = [&](std::span<const std::uint8_t> b) {
        return ds::session_peer_exchange(b, [&](auto out, auto in) { fwd.exchange_host(out, in); });
    };
    const auto clock = [] { return std::chrono::steady_clock::now(); };
    const auto elapsed = [&](auto t) { return std::chrono::duration<double>(clock() - t).count(); };
    std::uint64_t emitted = 0;
    namespace spark = strix::ds41::dspark;
    std::unique_ptr<spark::SpecTailDrafter> drafter;
    spark::WindowController controller;
    ds::SessionDraftState draft_state;
    if (a.session_dspark) {
        const auto draft_load_start = clock();
        drafter = std::make_unique<spark::SpecTailDrafter>(a.model, a.stage_dspark, a.stage_dspark_vocab);
        drafter->set_split(rank, *comm);
        std::printf("session drafter load rank %u seconds %.6f\n", rank, elapsed(draft_load_start));
        spark::WindowController::Costs costs;
        for (std::uint32_t r = 1; r <= spark::WindowController::max_rows; ++r) {
            costs.verify[r] = 1.0F + 0.15F * static_cast<float>(r - 1U);
            costs.samples[r] = 1;
        }
        if (!a.stage_dspark_costs.empty()) {
            const auto table = strix::ds41::pipe::read_spec_cost_table(a.stage_dspark_costs);
            for (std::uint32_t r = 1; r <= spark::WindowController::max_rows; ++r)
                if (r < table.verify.size() && table.verify[r] > 0) costs.verify[r] = static_cast<float>(table.verify[r]);
            costs.draft = static_cast<float>(table.draft); costs.extra = static_cast<float>(table.extra);
        }
        controller.set_costs(costs);
    }
    auto compat = rank_session_compat(a, rank);
    Sessions cache(std::move(compat), rank, (a.session_bytes ? a.session_bytes : (512ULL << 20U)), a.session_load.empty() ? a.session_save : a.session_load);
    std::optional<ds::SessionCheckpoint> loaded;
    if (!a.session_load.empty()) {
        const auto t = clock();
        loaded = cache.resume(a.session_id, true, ex);
        const auto device_start = clock();
        bool ok = true;
        try {
            // resume() validated BOTH CPU sections before either device import.
            if (drafter) {
                if (!loaded->draft) throw std::runtime_error("missing DSpark state");
                drafter->import_context(loaded->draft->context, loaded->draft->exchange_request);
                controller.import_state(loaded->draft->controller);
                draft_state = *loaded->draft;
            }
            fwd.import_state(loaded->attention);
        } catch (...) {
            ok = false;
        }
        Sessions::agree(ok, ex, "fresh-process target import");
        emitted = loaded->frontier.emitted;
        const auto device_seconds = elapsed(device_start);
        const auto timing = cache.last_restore();
        std::printf("session restore rank %u seconds %.6f read_seconds %.6f verify_seconds %.6f peer_seconds %.6f copy_agree_seconds %.6f bytes %llu\n",
                    rank, elapsed(t), timing.read_seconds, timing.verify_seconds, timing.peer_seconds, device_seconds,
                    static_cast<unsigned long long>(timing.payload_bytes));
    }
    const auto parse = [](const std::string &text) {
        std::vector<std::uint32_t> out;
        std::stringstream ss(text);
        for (std::string t; std::getline(ss, t, ',');)
            if (!t.empty() && t != "-")
                out.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        return out;
    };
    if (drafter && !loaded) {
        const auto initial = parse(a.prompt);
        if (initial.empty()) throw std::invalid_argument("DSpark session needs a prompt");
        (void)drafter->begin(initial, initial.front());
    }
    const auto step = [&](std::uint32_t token) {
        auto row = fwd.step(token);
        if (drafter) {
            drafter->stage(fwd.captured_streams(), 1U, fwd.stream());
            drafter->add_vocabulary(std::span(&token, 1));
        }
        return row;
    };
    std::ofstream logits;
    if (!a.logits_path.empty())
        logits.open(a.logits_path + "-r" + std::to_string(rank) + ".f32", std::ios::binary);
    if (!a.logits_path.empty() && !logits)
        throw std::runtime_error("cannot open session logits dump");
    const auto turn = [&](std::vector<std::uint32_t> input, std::uint32_t count, bool dump) {
        if (input.empty() || count == 0)
            throw std::invalid_argument("session turn requires input and positive generated-token count");
        if (std::uint64_t{fwd.next_position()} + input.size() + count >= a.max_positions)
            throw std::invalid_argument("session turn exceeds context");
        const auto t = clock();
        std::vector<float> row;
        for (auto token : input)
            row = step(token);
        const auto input_seconds = elapsed(t);
        const auto emit = [&](std::uint32_t i) {
            const auto token = fwd.next_token();
            if (dump) {
                std::printf("session token rank %u index %u id %u\n", rank, i, token);
                if (logits)
                    logits.write(reinterpret_cast<const char *>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
            }
            ++emitted;
        };
        if (!drafter) {
            for (std::uint32_t i = 0; i < count; ++i) {
                emit(i);
                if (i + 1 < count) row = step(fwd.next_token());
            }
        } else {
            std::uint32_t i = 0;
            emit(i++);
            while (i < count) {
                const bool use = a.stage_dspark_policy != "controller" || controller.draft_next();
                spark::Draft draft;
                std::uint32_t k = 0, verified = 0, accepted = 0;
                if (use) {
                    draft = drafter->draft(fwd.next_token(), fwd.stream());
                    k = a.stage_dspark_policy == "controller" ? controller.choose(draft) :
                        static_cast<std::uint32_t>(std::stoul(a.stage_dspark_policy.substr(6)));
                    k = std::min(k, count - i);
                    ++draft_state.windows;
                }
                // R1 verification: a rejected draft is never fed. The target remains the exact R1 producer.
                for (std::uint32_t j = 0;; ++j) {
                    row = step(fwd.next_token());
                    emit(i++);
                    if (use && j < k) {
                        ++verified;
                        if (fwd.next_token() == draft.tokens[j + 1U]) ++accepted;
                        else break;
                    }
                    if (i == count || j >= k) break;
                }
                if (use) controller.observe(draft, verified, accepted);
                draft_state.verified += verified; draft_state.accepted += accepted;
            }
            // Also done on the never-saved arm: capture has no hidden-row flush side effect.
            drafter->flush(fwd.stream());
            if (drafter->next_position() != fwd.next_position()) throw std::logic_error("DSpark turn frontier");
        }
        if (logits) {
            logits.flush();
            if (!logits)
                throw std::runtime_error("session logits write failed");
        }
        std::printf("session turn rank %u seconds %.6f input_seconds %.6f input %zu generated %u\n", rank, elapsed(t), input_seconds, input.size(), count);
    };
    auto prompt = parse(a.prompt);
    if (loaded)
        prompt.insert(prompt.begin(), loaded->frontier.pending);
    turn(std::move(prompt), a.generate, a.session_suffix.empty());
    if (!a.session_save.empty()) {
        const auto t = clock();
        ds::SessionCheckpoint c;
        c.id = a.session_id;
        c.rank = rank;
        c.compat_record = cache.compat().record();
        std::string gen = rank == 0 ? ds::session_generation() : std::string{};
        const auto other = ex(ds::SessionBytes(gen.begin(), gen.end()));
        if (rank == 1)
            gen.assign(other.begin(), other.end());
        c.generation = gen;
        c.attention = fwd.export_state();
        c.frontier.consumed = c.frontier.accepted = c.attention.position;
        c.frontier.pending = c.attention.next_token;
        c.frontier.emitted = emitted;
        if (drafter) {
            draft_state.context = drafter->export_context();
            draft_state.controller = controller.export_state();
            draft_state.exchange_request = drafter->exchange_request();
            c.draft = draft_state;
            c.frontier.drafter = draft_state.context.next;
            c.frontier.lineage = "ds41-dspark-r1-committed-turn-v2";
        }
        cache.capture(c, ex);
        const auto captured = elapsed(t);
        const auto disk_start = clock();
        cache.checkpoint(c.id, ex);
        const auto saved_bytes = std::filesystem::file_size(std::filesystem::path(a.session_save) /
            (c.id + "." + c.generation + ".rank" + std::to_string(rank) + ".state"));
        std::printf("session checkpoint rank %u capture_seconds %.6f write_fsync_seconds %.6f bytes %llu generation %s\n", rank, captured, elapsed(disk_start),
                    static_cast<unsigned long long>(saved_bytes), gen.c_str());
    }
    if (!a.session_suffix.empty()) {
        auto suffix = parse(a.session_suffix);
        suffix.insert(suffix.begin(), fwd.next_token());
        turn(std::move(suffix), a.session_suffix_tokens, true);
    }
    const auto state = fwd.export_state();
    ds::SessionCheckpoint final;
    final.attention = state;
    if (drafter) {
        draft_state.context = drafter->export_context();
        draft_state.controller = controller.export_state();
        draft_state.exchange_request = drafter->exchange_request();
        final.draft = draft_state;
        std::printf("session acceptance rank %u windows %llu verified %llu accepted %llu\n", rank,
            static_cast<unsigned long long>(draft_state.windows), static_cast<unsigned long long>(draft_state.verified),
            static_cast<unsigned long long>(draft_state.accepted));
    }
    std::printf("session final rank %u canonical_sha256 %s\n", rank, ds::session_state_sha256(final).c_str());
    if (!a.state_path.empty())
        ds::save_attention_state(a.state_path + "-r" + std::to_string(rank) + ".state", state);
    if (rank == 0 && !a.export_state.empty())
        ds::save_attention_state(a.export_state, state);
}

void run_rank(const Args& a, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport, bool print) {
    const auto t0 = std::chrono::steady_clock::now();
    // no transport: one node alone (world 1; with --stream-experts the experts are read from the file per layer)
    const bool single = transport == nullptr;
    std::unique_ptr<strix::dist::Communicator> comm_owner;
    // max message 1 MB: a verify window's STEP frame or its rows' logits slices (<= 1 MB control chunks) in one exchange
    if (!single && gpu_direct)
        comm_owner = std::make_unique<strix::dist::Communicator>(strix::dist::Group{rank, 2U, 1U, 0x64733431ULL}, std::move(transport),
                                                                 std::move(gpu_direct), std::size_t{1} << 20U);
    else if (!single) comm_owner = std::make_unique<strix::dist::Communicator>(strix::dist::Group{rank, 2U, 1U, 0x64733431ULL}, std::move(transport), std::size_t{1} << 20U);
    if (comm_owner) {  // both ranks must run the same product profile
        auto& comm = *comm_owner;
        strix::profile::check_peer(strix::deepseek41::pair_defaults::profile_resolution(), [&](std::span<const std::byte> profile_out, std::span<std::byte> profile_in) {
            comm.exchange_host(profile_out, profile_in, strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0x70726f66U, 0U});
        });
    }
    const strix::dist::CommunicatorStats no_stats{};
    strix::deepseek41::DeviceForwardOptions o;
    o.layers = a.layers;
    o.max_positions = a.max_positions;
    o.rank = rank;
    o.world = single ? 1U : 2U;
    o.communicator = comm_owner.get();
    o.stream_experts = a.stream_experts;
    o.op_hashes = a.op_hashes;
    o.gather_logits = a.gather;
    o.engram_direct_io = a.engram_direct;
    o.layer_timing = a.timing;
    o.op_timing = a.op_timing;
    o.tp_attention = a.tp;
    o.split_shared = a.split_shared;
    o.record_routes = !a.routes_path.empty();
    o.first_layer = a.first_layer;
    o.head = a.role != "pipeline-record";
    o.capture_main_hidden = a.session_dspark || (a.role == "pipeline-stage" && !a.stage_dspark.empty() && (rank == 0U || a.stage_dspark_placement == "split"));  // the drafter input
    if (a.session_validate_only) {
        namespace ds = strix::deepseek41;
        using Sessions = ds::PrefixCache::Sessions;
        // Host validation and peer agreement only: no DeviceForward or drafter upload.
        const ds::SessionExchange ex = [&](std::span<const std::uint8_t> b) {
            return ds::session_peer_exchange(b, [&](auto out, auto in) {
                for (std::size_t at = 0; at < out.size(); at += 65536U) {
                    const auto n = std::min<std::size_t>(65536U, out.size() - at);
                    comm_owner->exchange_host(out.subspan(at, n), in.subspan(at, n),
                        strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xC1U, at});
                }
            });
        };
        Sessions cache(rank_session_compat(a, rank), rank, (a.session_bytes ? a.session_bytes : (512ULL << 20U)), a.session_load);
        const auto io_start = std::chrono::steady_clock::now();
        const auto report_io = [&] {
            std::printf("session checkpoint io rank %u seconds %.6f\n", rank,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count());
        };
        try { (void)cache.resume(a.session_id, true, ex); }
        catch (...) { report_io(); throw; }
        report_io();
        std::printf("session validation rank %u passed before model upload\n", rank);
        return;
    }
    strix::deepseek41::DeviceForward fwd(a.model, o);
    if (a.role == "pipeline-record") {
        record_front(a, fwd, rank, single ? 1U : 2U, print);
        return;
    }
    if (a.role == "pipeline-stage") {
        if (print)
            std::printf("rank %u of %u (%s): pipeline TAIL, layers [%u, %u) + head, device weights %.2f GiB loaded in %.1f s\n", rank, single ? 1U : 2U,
                        single ? (a.stream_experts ? "one node, experts streamed from the file" : "one node") : comm_owner->transport_name().c_str(),
                        a.first_layer, a.layers, static_cast<double>(fwd.weight_bytes()) / 1073741824.0, fwd.load_seconds());
        kPeerChunk = a.stage_peer_chunk;
        kComm = comm_owner.get();
        if (rank == 0U) serve_stage(a, fwd, single ? 1U : 2U, comm_owner.get());
        else follow_stage(a, fwd, comm_owner.get());
        return;
    }
    if (a.session_bytes || !a.session_save.empty() || !a.session_load.empty()) {
        std::printf("session model load rank %u seconds %.6f\n", rank, fwd.load_seconds());
        run_session(a, fwd, rank, comm_owner.get());
        return;
    }
    const auto save_state = [&] {
        if (a.state_path.empty()) return;
        const auto state = fwd.export_state();
        const auto path = a.state_path + "-r" + std::to_string(rank) + ".state";
        strix::deepseek41::save_attention_state(path, state);
        std::printf("state rank %u position %u digest %016llx file %s\n", rank, state.position,
                    static_cast<unsigned long long>(state.digest()), path.c_str());
    };
    if (print)
        std::printf("rank %u of %u (%s): layers [0, %u), device weights %.2f GiB loaded in %.1f s\n", rank, single ? 1U : 2U,
                    single ? (a.stream_experts ? "one node, experts streamed from the file" : "one node") : comm_owner->transport_name().c_str(), a.layers,
                    static_cast<double>(fwd.weight_bytes()) / 1073741824.0, fwd.load_seconds());
    if (print && !fwd.overlay_description().empty()) std::printf("weights overlay: %s\n", fwd.overlay_description().c_str());
    if (print) std::printf("storage: %s\n", fwd.storage_description().c_str());
    std::ofstream logits_out;
    if (print && !a.logits_path.empty()) logits_out.open(a.logits_path, std::ios::binary);
    std::ofstream routes_out;  // one line per step: layer:id,id,... (the router's selection, both ranks the same)
    if (print && !a.routes_path.empty()) routes_out.open(a.routes_path);
    if (!a.routes_seqs.empty()) {  // route recording: every sequence from position 0, the same calls on both ranks
        record_route_sequences(a, fwd, print ? &routes_out : nullptr);
        return;
    }
    std::vector<std::uint32_t> seq;
    std::size_t base = 0;  // position of seq[0] (an imported state's)
    if (!a.import_state.empty()) {
        const auto state = strix::deepseek41::load_attention_state(a.import_state);
        fwd.import_state(state);
        base = state.position;
        seq.push_back(state.next_token);
        if (print)
            std::printf("imported state %s: position %u, next token %u, digest %016llx\n", a.import_state.c_str(), state.position, state.next_token,
                        static_cast<unsigned long long>(state.digest()));
    }
    std::stringstream ss(a.prompt);
    if (a.prompt != "-")
        for (std::string t; std::getline(ss, t, ',');) seq.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    for (std::string t; std::getline(ss, t, ',');) seq.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    strix::deepseek41::AttentionState start;
    if (!a.state_in.empty()) {
        start = strix::deepseek41::load_attention_state(a.state_in);
        fwd.import_state(start);
        if (print) std::printf("state in: %s, position %u, digest %016llx\n", a.state_in.c_str(), start.position,
                               static_cast<unsigned long long>(start.digest()));
    }
    if (!a.window_check.empty()) {  // R1 / verify-window logits check (see the header)
        constexpr std::uint32_t V = 129280U;
        std::vector<strix::ds41::VocabSlice> slices;
        std::vector<std::uint32_t> slice_ids;
        for (std::uint32_t r = 0; r < 2U; ++r)
            if (single || r == rank) {
                slices.push_back(strix::ds41::vocab_slice(V, 2U, r));
                slice_ids.push_back(r);
            }
        const auto stream = static_cast<hipStream_t>(fwd.stream());
        std::vector<float> host;
        const auto report = [&](std::size_t pos, std::uint32_t rows, std::uint32_t kept, const std::vector<std::uint32_t>& toks) {
            std::vector<std::vector<std::uint64_t>> h(slices.size());
            for (std::size_t k = 0; k < slices.size(); ++k) {
                const auto& sl = slices[k];
                host.resize(std::size_t{rows} * sl.count);
                const auto lr = fwd.device_logits_rows(sl.first);
                if (hipMemcpy2DAsync(host.data(), std::size_t{sl.count} * 4U, lr.p, lr.stride * 4U, std::size_t{sl.count} * 4U, rows,
                                     hipMemcpyDeviceToHost, stream) != hipSuccess ||
                    hipStreamSynchronize(stream) != hipSuccess)
                    throw std::runtime_error("window check: logits download");
                for (std::uint32_t r = 0; r < rows; ++r)
                    h[k].push_back(fnv(std::vector<float>(host.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * sl.count),
                                                          host.begin() + static_cast<std::ptrdiff_t>(std::size_t{r + 1U} * sl.count))));
                // With --window-check each rank writes its vocabulary slice for every physical row,
                // including rejected rows. Join rank slices in vocabulary order for full-logit KL.
                if (logits_out) logits_out.write(reinterpret_cast<const char*>(host.data()),
                                                 static_cast<std::streamsize>(host.size() * sizeof(float)));
            }
            for (std::uint32_t r = 0; r < rows; ++r) {
                std::string line = "wc rank " + std::to_string(rank) + " pos " + std::to_string(pos + r) + " rows " + std::to_string(rows) +
                                   " kept " + std::to_string(kept) + " argmax " + std::to_string(toks[r]);
                for (std::size_t k = 0; k < slices.size(); ++k) {
                    char b[40];
                    std::snprintf(b, sizeof b, " slice%u %016llx", slice_ids[k], static_cast<unsigned long long>(h[k][r]));
                    line += b;
                }
                std::printf("%s\n", line.c_str());
            }
            // Preserve the R1 operation evidence in a mixed R1/verify-window exactness gate.
            if (a.op_hashes && rows == 1U)
                for (const auto& [name, hash] : fwd.last_step_hashes())
                    std::printf("  %s %016llx\n", name.c_str(), static_cast<unsigned long long>(hash));
            std::fflush(stdout);
        };
        (void)fwd.step(seq[0]);  // a window may not start before position 0
        report(0, 1U, 1U, fwd.last_tokens());
        std::vector<std::uint32_t> window;
        for (std::size_t pos = 1, k = 0; pos < seq.size(); ++k) {
            const std::uint32_t cap = a.window_check[k % a.window_check.size()];
            window.assign(seq.begin() + static_cast<std::ptrdiff_t>(pos),
                          seq.begin() + static_cast<std::ptrdiff_t>(std::min(seq.size(), pos + cap)));
            const auto rows = static_cast<std::uint32_t>(window.size());
            const auto toks = fwd.step_window(window);
            std::uint32_t kept = rows;
            if (a.window_reject != 0U && rows > 1U && k % a.window_reject == a.window_reject - 1U)
                kept = 1U + static_cast<std::uint32_t>((k / a.window_reject) % (rows - 1U));  // 1 .. rows - 1
            report(pos, rows, kept, toks);
            fwd.commit_window(kept);
            pos += kept;
        }
        save_state();
        return;
    }
    const std::vector<std::uint32_t> prompt_seq = seq;
    const std::uint32_t arms = a.aba == 0U ? 1U : 2U * a.aba;
    std::vector<std::array<std::uint64_t, 3>> arm_prints;
    const bool aba_filter = a.aba_flags.find("filter") != std::string::npos, aba_direct = a.aba_flags.find("direct") != std::string::npos,
               aba_seghpw = a.aba_flags.find("seghpw") != std::string::npos, aba_fold = a.aba_flags.find("fold") != std::string::npos,
               aba_f16 = a.aba_flags.find("f16") != std::string::npos, aba_live = a.aba_flags.find("live") != std::string::npos,
               aba_engtp = a.aba_flags.find("engtp") != std::string::npos, aba_touch = a.aba_flags.find("touch") != std::string::npos,
               aba_rowviews = a.aba_flags.find("rowviews") != std::string::npos, aba_normilp = a.aba_flags.find("normilp") != std::string::npos,
               aba_routes = a.aba_flags.find("routes") != std::string::npos, aba_cfast = a.aba_flags.find("cfast") != std::string::npos,
               aba_swz = a.aba_flags.find("swz") != std::string::npos, aba_erows = a.aba_flags.find("erows") != std::string::npos;
    const int erows_start = strix::ds41::moe_kernel_tuning().rows;  // STRIX_DS41_EXPERT_ROWS as started
    const auto slice = strix::ds41::vocab_slice(129280U, single ? 1U : 2U, single ? 0U : rank);
    std::vector<float> slice_host(slice.count);
    for (std::uint32_t arm = 0; arm < arms; ++arm) {
    if (a.aba != 0U) {
        const int on = static_cast<int>(arm % 2U);
        namespace ah = strix::ds41::attn::hip;
        ah::set_index_candidate_filter(aba_filter ? on : -1);
        ah::set_index_direct(aba_direct ? on : -1);
        ah::set_attention_segment_hpw(aba_seghpw ? on : -1);
        ah::set_attention_window_fold(aba_fold ? on : -1);   // STRIX_DS41_ATTN_WINDOW_FOLD
        ah::set_attention_tile_f16(aba_f16 ? on : -1);     // STRIX_DS41_ATTN_TILE_F16
        ah::set_attention_live_chunks(aba_live ? on : -1);   // STRIX_DS41_ATTN_LIVE_CHUNKS
        strix::ds41::set_engram_tp(aba_engtp ? on : -1);     // STRIX_DS41_ENGRAM_TP
        strix::ds41::set_touch_batch(aba_touch ? on : -1);  // touch = STRIX_DS41_TOUCH_BATCH
        strix::ds41::set_trellis_row_views(aba_rowviews ? on : -1);  // STRIX_DS41_TRELLIS_ROW_VIEWS
        strix::ds41::set_norm_ilp(aba_normilp ? on : -1);            // STRIX_DS41_NORM_ILP
        strix::ds41::set_r1fuse_routes(aba_routes ? on : -1);        // STRIX_DS41_R1FUSE_ROUTES
        ah::set_compress_fast(aba_cfast ? on : -1);                  // STRIX_DS41_COMPRESS_FAST
        ah::set_attention_lds_swizzle(aba_swz ? on : -1);            // STRIX_DS41_ATTN_LDS_SWIZZLE
        strix::ds41::moe_kernel_tuning().rows = aba_erows ? on : erows_start;  // STRIX_DS41_EXPERT_ROWS
        if (arm != 0U) {
            seq = prompt_seq;
            if (!a.state_in.empty()) fwd.import_state(start);
            else fwd.reset();
        }
    }
    std::uint64_t fp_r1 = 14695981039346656037ULL, fp_win = 14695981039346656037ULL;
    const auto mix = [](std::uint64_t& h, std::uint64_t v) { h = (h ^ v) * 1099511628211ULL; };
    const std::size_t total = seq.size() + a.generate;
    double decode_seconds = 0.0;
    std::size_t decode_steps = 0;
    std::vector<double> layer_sum(a.layers, 0.0), exch_sum(a.layers, 0.0);
    std::size_t timed = 0;
    OpSums op_r1;
    if (a.fill != 0U) {  // context before the prompt, in verify windows of up to 8 rows (every row committed)
        std::vector<std::uint32_t> filler;
        std::ifstream f(a.fill_ids);
        for (std::string t; std::getline(f, t, ',');) filler.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        if (filler.empty()) throw std::invalid_argument("--fill needs --fill-ids FILE with comma-separated token ids");
        const auto f0 = std::chrono::steady_clock::now();
        std::vector<std::uint32_t> window;
        // the first row alone: a window may not start before the first compressed entry exists (position 0)
        fwd.step(filler[0]);
        for (std::uint32_t i = 1, k = 0; i < a.fill; ++k) {
            window.clear();
            const std::uint32_t cap = a.fill_windows.empty() ? strix::deepseek41::DeviceForward::max_window_rows
                                                             : a.fill_windows[k % a.fill_windows.size()];
            while (window.size() < cap && i < a.fill) window.push_back(filler[i++ % filler.size()]);
            fwd.step_window(window);
            // --fill-keep: commit the first K rows only (a rejected draft); the rejected tokens are fed again next window
            const auto size = static_cast<std::uint32_t>(window.size());
            const std::uint32_t keep = a.fill_keep.empty() ? size : std::min(size, a.fill_keep[k % a.fill_keep.size()]);
            fwd.commit_window(keep);
            i -= size - keep;
        }
        if (print)
            std::printf("filled %u context tokens in %.1f s (%.1f tok/s in windows of 8)\n", a.fill,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count(),
                        a.fill / std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count());
    }
    if (!a.forced_windows.empty()) seq.clear();  // the windows start at the imported state's position (its next token first)
    for (std::size_t pos = 0; pos < seq.size(); ++pos) {
        const auto& logits = fwd.step(seq[pos]);
        const auto next = fwd.next_token();
        if (a.aba != 0U) {  // fingerprint: the token and this process's vocabulary slice of the logits
            const auto lr = fwd.device_logits_rows(slice.first);
            if (hipMemcpyAsync(slice_host.data(), lr.p, std::size_t{slice.count} * 4U, hipMemcpyDeviceToHost, static_cast<hipStream_t>(fwd.stream())) != hipSuccess ||
                hipStreamSynchronize(static_cast<hipStream_t>(fwd.stream())) != hipSuccess)
                throw std::runtime_error("aba: logits download");
            mix(fp_r1, next);
            mix(fp_r1, fnv(slice_host));
        }
        if (pos >= 1U) {  // the first step pays one-time costs
            decode_seconds += fwd.last_step_seconds();
            ++decode_steps;
        }
        if (a.op_timing && pos >= 1U) op_r1.add(fwd.last_op_timing(), fwd.last_step_seconds() * 1e3);
        if (a.timing && pos >= 1U) {
            for (std::uint32_t l = 0; l < a.layers; ++l) {
                layer_sum[l] += fwd.last_layer_timing()[l].seconds;
                exch_sum[l] += fwd.last_layer_timing()[l].exchange;
            }
            ++timed;
        }
        if (print) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const bool full = a.gather || a.op_hashes;
            std::printf("pos %zu fed %u argmax %u top %.4f logits_fnv %016llx (%.1f s)\n", base + pos, seq[pos], next,
                        full ? static_cast<double>(logits[next]) : 0.0, static_cast<unsigned long long>(full ? fnv(logits) : 0ULL), s);
            if (a.op_hashes)
                for (const auto& [name, hash] : fwd.last_step_hashes()) std::printf("  %s %016llx\n", name.c_str(), static_cast<unsigned long long>(hash));
            if (a.timing)
                std::printf("  step %.2f ms (engram read %.2f ms, head+argmax %.2f ms, expert copies %.2f ms)\n", fwd.last_step_seconds() * 1e3,
                            fwd.last_engram_seconds() * 1e3, fwd.last_head_seconds() * 1e3, fwd.last_stream_seconds() * 1e3);
            if (logits_out) logits_out.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * 4U));
            if (routes_out) {
                const auto& routes = fwd.last_step_routes();
                for (std::size_t l = 0; l < routes.size(); ++l) {
                    routes_out << (l ? " " : "") << l << ':';
                    for (std::size_t k = 0; k < routes[l].size(); ++k) routes_out << (k ? "," : "") << routes[l][k];
                }
                routes_out << '\n';
            }
            std::fflush(stdout);
        }
        if (pos + 1U == seq.size() && seq.size() < total) seq.push_back(next);
    }
    if (!a.forced_windows.empty()) {
        // --forced-windows FILE: verify windows with given inputs, one per line "KEEP T0,T1,..,TR-1" (R 1..8): the rows at
        // the current positions, every row's argmax and full-logits FNV printed, then commit_window(KEEP) (R = 1: a step).
        // The pair-only reference of a pipeline's forced windows (same state, same inputs, same commits).
        std::ifstream f(a.forced_windows);
        if (!f) throw std::invalid_argument("cannot read --forced-windows " + a.forced_windows);
        std::string line;
        std::uint32_t w = 0;
        std::map<std::uint32_t, std::vector<double>> wall;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            std::uint32_t keep = 0;
            std::string toks;
            ls >> keep >> toks;
            std::vector<std::uint32_t> window;
            std::stringstream ts(toks);
            for (std::string t; std::getline(ts, t, ',');) window.push_back(static_cast<std::uint32_t>(std::stoul(t)));
            const auto R = static_cast<std::uint32_t>(window.size());
            if (R == 0U || R > strix::deepseek41::DeviceForward::max_window_rows || keep == 0U || keep > R)
                throw std::invalid_argument("--forced-windows: bad line " + line);
            const std::uint32_t pos0 = fwd.next_position();
            std::vector<std::uint32_t> argmax;
            std::vector<float> rows_logits;
            if (R == 1U) {
                (void)fwd.step(window[0]);
                argmax = {fwd.next_token()};
            } else {
                argmax = fwd.step_window(window);
            }
            const double ms = fwd.last_step_seconds() * 1e3;
            if (a.gather || a.op_hashes) rows_logits = fwd.window_logits();  // both ranks
            if (R > 1U) fwd.commit_window(keep);
            if (w >= 1U) wall[R].push_back(ms);
            if (print) {
                const std::size_t V = rows_logits.empty() ? 0U : rows_logits.size() / R;
                for (std::uint32_t r = 0; r < R; ++r) {
                    const std::uint64_t h = V ? pipe::fnv_floats(std::span<const float>(rows_logits).subspan(r * V, V)) : 0U;
                    std::printf("win %u R %u keep %u row %u pos %u fed %u argmax %u top %.4f logits_fnv %016llx\n", w, R, keep, r, pos0 + r, window[r], argmax[r],
                                V ? static_cast<double>(rows_logits[r * V + argmax[r]]) : 0.0, static_cast<unsigned long long>(h));
                }
                std::printf("window %u R %u keep %u step %.2f ms\n", w, R, keep, ms);
                std::fflush(stdout);
            }
            ++w;
        }
        if (print)
            for (auto& [R, v] : wall) {
                std::sort(v.begin(), v.end());
                double mean = 0.0;
                for (const auto x : v) mean += x;
                std::printf("forced windows R%u: %zu windows, step median %.2f ms, mean %.2f ms\n", R, v.size(), v.empty() ? 0.0 : v[v.size() / 2U],
                            v.empty() ? 0.0 : mean / static_cast<double>(v.size()));
            }
    }
    std::vector<std::pair<std::uint32_t, OpSums>> op_windows;  // per window size
    std::vector<double> window_means;
    for (const std::uint32_t rows : a.window_rows) {  // verify windows of R rows (the DSpark verifier's shape), every row committed
        if (rows == 0U || rows > strix::deepseek41::DeviceForward::max_window_rows) throw std::invalid_argument("--window-rows 1..8");
        OpSums& sums = op_windows.emplace_back(rows, OpSums{}).second;
        double wall = 0.0;
        std::vector<std::uint32_t> window;
        for (std::uint32_t w = 0; w < a.windows + 1U; ++w) {
            window.clear();
            for (std::uint32_t r = 0; r < rows; ++r) window.push_back(seq[(w * rows + r) % seq.size()]);
            for (const auto t : fwd.step_window(window)) mix(fp_win, t);
            fwd.commit_window(rows);
            if (w == 0U) continue;  // the first window pays one-time costs
            if (a.op_timing) sums.add(fwd.last_op_timing(), fwd.last_step_seconds() * 1e3);
            wall += fwd.last_step_seconds() * 1e3;
        }
        if (print && !a.op_timing && a.windows != 0U)
            std::printf("verify windows of %u rows: mean step wall %.2f ms over %u windows\n", rows, wall / a.windows, a.windows);
        window_means.push_back(a.windows != 0U ? wall / a.windows : 0.0);
    }
    if (a.aba != 0U) {
        std::uint64_t fp_state = 14695981039346656037ULL;
        for (const auto& [name, h] : fwd.state_hashes()) mix(fp_state, h);
        arm_prints.push_back({fp_r1, fp_win, fp_state});
        if (print) {
            std::printf("aba arm %u flags %s %s r1_ms %.3f", arm, a.aba_flags.c_str(), arm % 2U ? "on" : "off",
                        decode_steps ? decode_seconds * 1e3 / static_cast<double>(decode_steps) : 0.0);
            for (std::size_t k = 0; k < window_means.size(); ++k) std::printf(" r%u_ms %.3f", a.window_rows[k], window_means[k]);
            std::printf(" fp_r1 %016llx fp_win %016llx fp_state %016llx\n", static_cast<unsigned long long>(fp_r1),
                        static_cast<unsigned long long>(fp_win), static_cast<unsigned long long>(fp_state));
            std::fflush(stdout);
        }
        if (arm + 1U < arms) continue;
        bool same = true;
        for (const auto& f : arm_prints) same = same && f == arm_prints[0];
        if (print) std::printf("aba exact %s (%u arms; R1 tokens + logits slice, window tokens, attention state)\n", same ? "IDENTICAL" : "DIFFER", arms);
    }
    if (print && !a.export_state.empty()) {  // the state after every fed position (the last generated token is next_token)
        const auto state = fwd.export_state();
        strix::deepseek41::save_attention_state(a.export_state, state);
        std::printf("exported state %s: position %u, next token %u, digest %016llx\n", a.export_state.c_str(), state.position, state.next_token,
                    static_cast<unsigned long long>(state.digest()));
    }
    if (!print) continue;
    op_r1.print("decode steps", 1U, a.op_timing_out);
    for (const auto& [rows, sums] : op_windows) sums.print("verify windows", rows, a.op_timing_out);
    save_state();
    const auto& st = single ? no_stats : comm_owner->stats();
    std::printf("decode: %zu steps after the first, %.2f tok/s (%.1f ms/token); exchanges %llu, %.1f MB sent, %.2f s in exchange (stage %.2f, wire %.2f, check %.2f)\n",
                decode_steps, decode_steps ? static_cast<double>(decode_steps) / decode_seconds : 0.0,
                decode_steps ? decode_seconds * 1e3 / static_cast<double>(decode_steps) : 0.0, static_cast<unsigned long long>(st.exchanges),
                static_cast<double>(st.bytes_sent) / 1e6, st.seconds, st.stage, st.wire, st.check);
    if (a.timing && timed != 0U) {
        std::printf("per layer (mean over %zu steps): layer total ms / exchange ms / compute ms\n", timed);
        double tot = 0.0, ex = 0.0;
        for (std::uint32_t l = 0; l < a.layers; ++l) {
            const double t = layer_sum[l] / static_cast<double>(timed) * 1e3, e = exch_sum[l] / static_cast<double>(timed) * 1e3;
            tot += t;
            ex += e;
            std::printf("  layer %2u %7.3f %7.3f %7.3f\n", l, t, e, t - e);
        }
        std::printf("  sum      %7.3f %7.3f %7.3f\n", tot, ex, tot - ex);
    }
    }
}

// What this binary can do, for TensorMeld (tensormeld/engine-describe-v1).
std::string describe(const Args& a) {
    strix::deepseek41::EngineDescribeOptions o;
    o.engine_name = "strix-deepseek41-rank";
    o.backend = "hip";
    o.roles = {"whole", "pipeline-stage", "prefill", "decode"};
    o.state_export = o.state_import = true;
    o.state_transports = {"file", "tcp"};
    o.max_positions = a.max_positions;
    namespace serve = strix::serve;
    serve::Json argv = serve::Json::object();
    argv.set("whole", serve::Json(serve::Json::Array{serve::Json("{executable}"), serve::Json("{model}"), serve::Json("--rank"), serve::Json("{rank}"),
                                                      serve::Json("--transport"), serve::Json("usb4auto"), serve::Json("--layers"), serve::Json("40"),
                                                      serve::Json("--prompt-ids"), serve::Json("{prompt_ids}")}));
    argv.set("pipeline-stage", serve::Json(serve::Json::Array{
                                   serve::Json("{executable}"), serve::Json("{model}"), serve::Json("--rank"), serve::Json("{rank}"), serve::Json("--transport"),
                                   serve::Json("usb4auto"), serve::Json("--role"), serve::Json("pipeline-stage"), serve::Json("--first-layer"), serve::Json("{cut}"),
                                   serve::Json("--layers"), serve::Json("40"), serve::Json("--stage-listen"), serve::Json("{listen}"),
                                   serve::Json("--stage-token-file"), serve::Json("{token_file}"), serve::Json("--max-positions"), serve::Json("{max_positions}")}));
    argv.set("pipeline-front", serve::Json(serve::Json::Array{
                                   serve::Json("strix-deepseek41-pipeline-front"), serve::Json("{model}"), serve::Json("--backend"), serve::Json("hip"),
                                   serve::Json("--layers"), serve::Json("{cut}"), serve::Json("--stage-connect"), serve::Json("{tail}"),
                                   serve::Json("--stage-token-file"), serve::Json("{token_file}"), serve::Json("--prompt-ids"), serve::Json("{prompt_ids}")}));
    argv.set("prefill", serve::Json(serve::Json::Array{serve::Json("{executable}"), serve::Json("{model}"), serve::Json("--transport"), serve::Json("single"),
                                                        serve::Json("--layers"), serve::Json("40"), serve::Json("--prompt-ids"), serve::Json("{prompt_ids}"),
                                                        serve::Json("--generate"), serve::Json("0"), serve::Json("--export-state"), serve::Json("{state_file}")}));
    argv.set("decode", serve::Json(serve::Json::Array{serve::Json("{executable}"), serve::Json("{model}"), serve::Json("--rank"), serve::Json("{rank}"),
                                                       serve::Json("--transport"), serve::Json("usb4auto"), serve::Json("--import-state"), serve::Json("{state_file}"),
                                                       serve::Json("--generate"), serve::Json("{max_tokens}")}));
    o.argv_template = argv;
    o.notes = "pipeline-stage: the TAIL [cut, 40) + head; the FRONT is strix-deepseek41-pipeline-front (layers [0, cut)); state over TCP "
              "(EXPORT_STATE / IMPORT_STATE of the pipeline wire) or files (--export-state / --import-state)";
    return strix::deepseek41::describe_engine_file(a.model, o);
}

}  // namespace

int main(int argc, char** argv) try {
    const strix::deepseek41::pair_defaults::Effective production_defaults(strix::deepseek41::pair_defaults::App::rank, argc, argv, strix::deepseek41::pair_defaults::this_build());  // before any option or env read
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        if (s == "--rank") a.rank = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--transport") a.transport = next();
        else if (s == "--layers") a.layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--prompt-ids") a.prompt = next();
        else if (s == "--generate") a.generate = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--max-positions") a.max_positions = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--logits") a.logits_path = next();
        else if (s == "--state-out") a.state_path = next();
        else if (s == "--state-in") a.state_in = next();
        else if (s == "--aba") a.aba = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--aba-flags") a.aba_flags = next();
        else if (s == "--op-hashes") a.op_hashes = true;
        else if (s == "--timing") a.timing = true;
        else if (s == "--engram-direct") a.engram_direct = true;
        else if (s == "--engram-mapped") a.engram_direct = false;
        else if (s == "--no-gather") a.gather = false;
        else if (s == "--tp-attention") a.tp = true;
        else if (s == "--no-tp-attention") a.tp = false;
        else if (s == "--split-shared") a.split_shared = true;
        else if (s == "--stream-experts") a.stream_experts = true;
        else if (s == "--routes") a.routes_path = next();
        else if (s == "--routes-seqs") a.routes_seqs = next();
        else if (s == "--fill") a.fill = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--fill-ids") a.fill_ids = next();
        else if (s == "--fill-window") {
            std::stringstream ws(next());
            for (std::string t; std::getline(ws, t, ',');) {
                const auto r = static_cast<std::uint32_t>(std::stoul(t));
                if (r == 0U || r > strix::deepseek41::DeviceForward::max_window_rows) throw std::invalid_argument("--fill-window 1..8");
                a.fill_windows.push_back(r);
            }
        }
        else if (s == "--fill-keep") {
            std::stringstream ws(next());
            for (std::string t; std::getline(ws, t, ',');) {
                const auto r = static_cast<std::uint32_t>(std::stoul(t));
                if (r == 0U) throw std::invalid_argument("--fill-keep 1..8");
                a.fill_keep.push_back(r);
            }
        }
        else if (s == "--describe-json") a.describe = true;
        else if (s == "--role") a.role = next();
        else if (s == "--first-layer") a.first_layer = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--stage-listen") a.stage_listen = next();
        else if (s == "--stage-token-file") a.stage_token_file = next();
        else if (s == "--stage-once") a.stage_once = true;
        else if (s == "--stage-sessions") a.stage_sessions = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--forced-windows") a.forced_windows = next();
        else if (s == "--stage-timeout") a.stage_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--stage-no-busy-poll") a.stage_busy_poll = false;
        else if (s == "--stage-timeline") a.stage_timeline = next();
        else if (s == "--stage-logits") a.stage_logits = next();
        else if (s == "--stage-peer-chunk") {
            a.stage_peer_chunk = static_cast<std::size_t>(std::stoull(next()));
            if (a.stage_peer_chunk != 0U && (a.stage_peer_chunk < 4096U || a.stage_peer_chunk > (std::size_t{1} << 20U) - 128U))
                throw std::invalid_argument("--stage-peer-chunk: 0 or 4096 .. 1 MB - 128 bytes");
        }
        else if (s == "--stage-peer-thread") a.stage_peer_thread = true;
        else if (s == "--record") a.record = next();
        else if (s == "--stage-dspark") a.stage_dspark = next();
        else if (s == "--stage-dspark-vocab") a.stage_dspark_vocab = next();
        else if (s == "--stage-dspark-policy") a.stage_dspark_policy = next();
        else if (s == "--stage-dspark-costs") a.stage_dspark_costs = next();
        else if (s == "--stage-dspark-placement") a.stage_dspark_placement = next();
        else if (s == "--stage-fault-drop-accept") a.stage_fault_drop_accept = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--session-dspark") a.session_dspark = true;
        else if (s == "--session-save")
            a.session_save = next();
        else if (s == "--session-load")
            a.session_load = next();
        else if (s == "--session-id")
            a.session_id = next();
        else if (s == "--session-hash-only") a.session_hash_only = true;
        else if (s == "--session-validate-only")
            a.session_validate_only = true;
        else if (s == "--session-cache-mb")
            a.session_bytes = std::stoull(next()) << 20U;
        else if (s == "--session-ram-bytes") a.session_bytes = std::stoull(next());
        else if (s == "--session-suffix-ids")
            a.session_suffix = next();
        else if (s == "--session-suffix-tokens")
            a.session_suffix_tokens = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--import-state") a.import_state = next();
        else if (s == "--export-state") a.export_state = next();
        else if (s == "--window-check") {
            std::stringstream ws(next());
            for (std::string t; std::getline(ws, t, ',');) {
                const auto r = static_cast<std::uint32_t>(std::stoul(t));
                if (r == 0U || r > strix::deepseek41::DeviceForward::max_window_rows) throw std::invalid_argument("--window-check 1..8");
                a.window_check.push_back(r);
            }
        }
        else if (s == "--window-reject") a.window_reject = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--op-timing") a.op_timing = true;
        else if (s == "--window-rows") {
            std::stringstream ws(next());
            for (std::string t; std::getline(ws, t, ',');) a.window_rows.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        }
        else if (s == "--windows") a.windows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--op-timing-out") a.op_timing_out = next();
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    if (a.describe) {
        if (a.model.empty()) throw std::invalid_argument("usage: strix-deepseek41-rank MODEL.gguf --describe-json");
        std::printf("%s\n", describe(a).c_str());
        return 0;
    }
    if (a.session_hash_only) {
        if (a.model.empty()) throw std::invalid_argument("hash-only requires MODEL");
        (void)rank_session_compat(a, a.rank);
        return 0;  // CPU only: before transport, DeviceForward and drafter construction.
    }
    strix::LoaderCapture loader_capture(a.session_bytes || !a.session_save.empty() || !a.session_load.empty());
    if (a.session_validate_only && a.session_load.empty())
        throw std::invalid_argument("--session-validate-only requires --session-load");
    if (a.session_dspark && !a.session_bytes && a.session_save.empty() && a.session_load.empty())
        throw std::invalid_argument("--session-dspark requires --session-cache-mb, --session-save or --session-load");
    if (a.session_bytes || !a.session_save.empty() || !a.session_load.empty()) {
        if (a.transport == "single" || a.role == "pipeline-stage" || a.role == "pipeline-record")
            throw std::invalid_argument("session cache requires whole TP2 ranks");
        if ((!a.stage_dspark.empty()) != a.session_dspark)
            throw std::invalid_argument("session DSpark requires --session-dspark and --stage-dspark FILE together");
        if (a.session_dspark && (a.layers != 40U || a.stage_dspark_placement != "split" ||
            (a.stage_dspark_policy != "controller" && a.stage_dspark_policy.rfind("fixed:", 0) != 0)))
            throw std::invalid_argument("session DSpark requires 40 layers, split placement, controller or fixed:0..5");
        if (a.session_dspark && a.stage_dspark_policy != "controller" && std::stoul(a.stage_dspark_policy.substr(6)) > 5U)
            throw std::invalid_argument("session DSpark fixed drafts must be 0..5");
        if (!a.session_save.empty() && !a.session_load.empty())
            throw std::invalid_argument("choose --session-save or --session-load");
        if (!a.session_load.empty() && !a.session_suffix.empty())
            throw std::invalid_argument("restore suffix goes in --prompt-ids");
        if (!a.import_state.empty() || !a.state_in.empty() || a.aba || a.fill || !a.window_rows.empty() || !a.window_check.empty() ||
            !a.forced_windows.empty() || !a.gather)
            throw std::invalid_argument("session CLI needs greedy R1, gathered logits and no other state/benchmark modes");
        if (a.generate == 0 || a.session_bytes > (std::uint64_t{1} << 40U))
            throw std::invalid_argument("invalid session generated count or RAM budget");
        strix::deepseek41::validate_session_id(a.session_id);
    }
    if (!a.role.empty() && a.role != "pipeline-stage" && a.role != "whole" && a.role != "pipeline-record")
        throw std::invalid_argument("--role is whole, pipeline-stage or pipeline-record");
    if ((a.role == "pipeline-record") != !a.record.empty()) throw std::invalid_argument("--role pipeline-record goes with --record FILE");
    if (a.role == "pipeline-stage") {
        if (a.first_layer == 0U || a.first_layer >= a.layers) throw std::invalid_argument("--role pipeline-stage needs --first-layer K in [1, layers)");
        if (a.stage_listen.empty() || a.stage_token_file.empty()) throw std::invalid_argument("--role pipeline-stage needs --stage-listen and --stage-token-file");
        if (a.prompt.empty()) a.prompt = "0";  // unused
    } else if (a.first_layer != 0U) {
        throw std::invalid_argument("--first-layer needs --role pipeline-stage");
    }
    if (!a.routes_seqs.empty()) {
        if (a.routes_path.empty() || !a.role.empty()) throw std::invalid_argument("--routes-seqs needs --routes FILE (whole model, no --role)");
        if (a.prompt.empty()) a.prompt = "0";  // unused
    }
    if (!a.import_state.empty() && a.prompt.empty()) a.prompt = "-";  // the state's next token starts the run
    if (a.model.empty() || a.prompt.empty()) throw std::invalid_argument("usage: strix-deepseek41-rank MODEL.gguf --rank R --transport T --prompt-ids a,b,c ...");
    if (a.transport == "single") {  // one node alone
        run_rank(a, 0U, nullptr, true);
        return 0;
    }
    if (a.transport == "emulate") {  // both ranks in this process over the in-memory pair; rank 0 prints
        auto [t0, t1] = strix::dist::memory_pair();
        std::exception_ptr failure;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                run_rank(a, 1U, std::move(t), false);
            } catch (...) {
                failure = std::current_exception();
            }
        });
        try {
            run_rank(a, 0U, std::move(t0), true);
        } catch (...) {
            peer.join();
            throw;
        }
        peer.join();
        if (failure) std::rethrow_exception(failure);
        return 0;
    }
    // A blocked open() of a stream on a dead link would hold this node (and the peer) indefinitely: the alarm's
    // default action ends the process if the transport is not open within the timeout.
    ::alarm(a.connect_timeout);
    auto transport = open_transport(a.transport, a.rank, a.connect_timeout);
    ::alarm(0);
    std::printf("connected (%s)\n", transport->name().c_str());
    std::fflush(stdout);
    run_rank(a, a.rank, std::move(transport), true);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
