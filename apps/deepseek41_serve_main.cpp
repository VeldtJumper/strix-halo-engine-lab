// strix-deepseek41-serve: DeepSeek V4.1 Flash served by a two-node engine group (EP experts, TP attention).
// Rank 0 runs the OpenAI/Anthropic-compatible HTTP front end (strix_serve) and the continuous-batching scheduler; rank
// 1 has no API and is driven by rank 0 over the communicator: before every step rank 0 sends a control block (the
// rows of the step and the slots to reset), both ranks run BatchForward::step on the same rows, and both hold the
// same greedy tokens. Requests are admitted between steps while slots are free (continuous admission); their prompts
// are read in chunks of rows next to the decoding requests. Greedy decoding only: every request's tokens are those
// of running it alone.
//
//   strix-deepseek41-serve MODEL.gguf --rank R --transport usb4auto|usb4gpu|tcp-listen:P|tcp-connect:H:P|emulate
//       [--group-manifest FILE] [--layers 40] [--context 4096] [--slots 4] [--prompt-rows 8] [--tp-attention]
//       [--pool-positions N] (a shared KV pool of N positions over all slots: a request holds prompt + max_tokens of it)
//       [--engram-direct] [--http-bind 127.0.0.1] [--http-port 0] [--http-api-key-file F] [--http-max-queue 64]
//       [--http-max-connections 64] [--http-responses on|off] [--http-capabilities on|off]
//       [--http-stream-receipts on|off] [--http-sse-keepalive 10] [--model-id deepseek-v4.1-flash] [--default-max-tokens 1024]
//       [--connect-timeout 60] [--peer-timeout 60]
//       [--prefill-path rows|device] [--prefill-rows 4096] [--prefill-min 0]
//       [--prefix-cache-mb 512] [--prefix-block 256] [--prefix-reserve-mb 8192] [--import-parked 1]
//       [--prefix-cache-dir DIR [--prefix-cache-disk-mb 0]] [--prefix-before-tail] [--prefix-session-snapshots]
//       [--cache-profile agent|off] [--no-prefix-cache] [--no-prefix-disk] [--no-prefix-before-tail]
//       [--no-prefix-session-snapshots] [--prefix-inrun-snapshots [--prefix-inrun-reach 0]] [--no-prefix-inrun-snapshots] (agent default)
//       [--dspark DSPARK.gguf [--dspark-window controller|fixed:R] [--dspark-vocab SPEC] [--dspark-ranks 2|1]
//        [--dspark-max-inflight 4]]   (drafts on both ranks by default; plain steps while > N requests are in flight)
//       [--hetero-listen HOST:PORT --hetero-token-file FILE [--hetero-cut 22] [--hetero-groups 2|1]]   (split decode, see below)
//   strix-deepseek41-serve MODEL.gguf --describe-json: tensormeld/engine-describe-v1 (roles, capabilities, group mode and the
//       model's parts descriptor, from the file header only)
//   strix-deepseek41-serve MODEL.gguf --describe-capabilities-json [--context N] [--slots N] [--layers 40]: the group
//       launch description (strix/engine-capabilities-v1: memory per rank, API, argv template)
//
// Split decode (opt-in, ds41_hetero_stage.hpp, strix/ds41-pipeline-multi-v1): with --hetero-listen, rank 0 waits for a
// FRONT of sequence slots (strix-cuda-ds41-multi-front on the workstation GPU) before READY; every batch step's layers
// [0, cut) run there, one SEQ_STEP per request segment, and the pair runs [cut, 40) and the head on the boundaries
// (BatchForward::step_boundary, rank 0 forwards them to rank 1). A device prefill runs on the pair as usual; its state
// goes to the FRONT's slot (layers [0, cut)) before the request's first step; prompts read in rows go through the FRONT.
// DSpark windows work as before (the FRONT drops a window's rejected rows with the next step of the slot); the
// controller costs each step with its whole wall. Not with the prefix cache (the pair does not keep layers [0, cut)).
// --hetero-groups 2 (default): the decoding requests in two groups (slot parity); a group's next rows (its DSpark plan
// parked in the speculator) go to the FRONT as soon as its step ends, so the FRONT runs one group while the pair steps the
// other. Every admission, cancellation, import or drain first runs the plans in flight (in their order).
// Every request's tokens are those of the pair alone (bitwise in the identity without the trellis shared expert,
// which the FRONT runs in IEEE order).
//
// Events (one JSON line each on stdout): {"event":"READY",...} when a rank serves (rank 0 after its HTTP listener is
// up), {"event":"POISONED","reason":...} before a non-zero exit (peer lost, engine failure, group mismatch),
// {"event":"STOPPED"} after a clean stop (SIGINT/SIGTERM: stop admitting, finish the running requests, stop rank 1).
// The group manifest (TensorMeld, strix/group-manifest-v1) gives group_id, epoch, rank, transport, http, context and
// slots; both ranks check group_id, epoch, the engine executable's SHA-256 and the model file size in a hello.
//
// --prefill-path device: an admitted request's prompt (of at least --prefill-min tokens) is run by the two-rank device
// prefill (DevicePrefill, both ranks; rank 0 sends a prefill command and the prompt over the control channel) instead
// of rows of the batch steps; the prefill reads the BatchForward's weights in place (one copy per node), runs pieces
// of at most --prefill-rows rows, and its attention state is imported into the request's slot, which then decodes as
// usual. Same tokens as the rows path (the prefill's rows equal the decode step's). The other requests wait during the
// prefill (no decode steps run).
// --prefix-cache-mb M (device path): a prefix cache of attention states on the host, M MiB (deepseek41_prefix_cache.hpp):
// a prompt whose beginning was prefilled before resumes from the longest cached block boundary; each rank keeps its own
// copy (the state is replicated), rank 0 decides hits and both ranks agree capture admission from fresh MemAvailable
// after predicted allocations and --prefix-reserve-mb. Rank 1 checks it holds the same entry. The cache lives on the host, apart from the KV
// pool: it never takes room from live requests.
// --prefix-cache-dir DIR (with --prefix-cache-mb): a disk tier of the cache in DIR/<namespace tag>/rank<R> on each rank's
// own disk (the tag covers the weights overlay, trellis, decode identity: prefix_namespace_from_env; both ranks must agree), at
// most --prefix-cache-disk-mb MiB (0: no limit): every inserted block and snapshot is written through (writer thread),
// a lookup also finds what only the disk holds and reads it back (then promoted into RAM). At start each rank rebuilds
// its index from its files (another model, bad checksums, orphans ignored) and the ranks keep the entries both hold.
// A hit that reads the disk is all-or-nothing on the two ranks: after reading, they exchange whether it verified; if
// either failed, both drop that path from the disk tier and run the prompt as a miss.
// --prefix-before-tail: a prompt's prefill also ends a run (and inserts a snapshot) one block before its last full
// block, so an edited last block resumes close to the end (one more run split per prompt).
// --dspark: DSpark windows inside the batch steps (BatchSpeculator): rank 0 drafts for every decoding request (one
// drafting context per slot) and verifies the windows in the steps; every request still gets exactly its greedy
// tokens alone. A request whose prompt the device prefill read drafts from the rows it generates (its context starts
// at the prompt's end). The commits that dropped window rows travel in the next control block.
// --prefix-inrun-snapshots (with a prefix cache): the snapshots at the shared block, the last full block and (before-tail)
// one block before it are taken inside the prefill runs (DevicePrefill::set_snapshot_points / export_state_at) instead
// of ending a run there: a run split reads every routed expert once more and replays the decoder cone again (~0.2 s
// per split on the pair). The runs end only every --prefill-rows rows and at the prompt's end. Same states, bitwise;
// such a snapshot does not know the token after it (never a whole-prompt hit). Chat requests also snapshot the last
// block boundary of their system prompt + tools (GenerationJob::stable_prefix), so the next conversation with the same
// system prompt resumes there from its first turn.
// --prefix-session-snapshots: when a decoding request reaches a block boundary, its slot's state is inserted (only the
// KV entries the cache lacks are copied from the device): a conversation's next turn resumes after the generated
// tokens when the client sends them back unchanged. The snapshot does not know its logits (FNV 0).
//
// Imported prefill states (a prefill that ran elsewhere, e.g. on another device): POST /admin/import-state on rank 0
// with the bytes of a strix/ds41-attention-state-v1 file as the body (application/octet-stream; optional headers
// X-State-Format, X-State-Sha256 checked when given, X-State-Reserve-Tokens = KV pool room after the state, default
// --default-max-tokens). Rank 0 sends the bytes to rank 1 over the communicator, both ranks import them into a free
// slot (BatchForward::import_state) and the slot stays parked; the reply is {"imported":true,"position":P,"sha256":...}
// with the server's timings. The next request whose prompt starts with the state's tokens takes the parked slot
// without a prefill: a prompt equal to the state's tokens gets the state's next token as its first token at once and
// decodes from there; a longer prompt reads only its remaining tokens. At most --import-parked N states are parked
// (default 1, the oldest goes first); a request that needs a slot while none is free takes a parked one's slot.
// The import runs between batch steps (the decoding requests wait for it). Nothing changes when no state is imported.
//
// STRIX_DS41_SERVE_SCHED (device path; default interleave): decode-priority scheduling, decided by rank 0 alone
// (deepseek41_serve_sched.hpp). defer: a prompt whose device prefill would stall decoding requests waits in an admission
// queue until nobody decodes, it waited STRIX_DS41_SERVE_SCHED_STARVE_MS, or N x T_dev(P) <= the decoders' remaining
// time; a state imported for it meanwhile is taken without a prefill. interleave: the prefill runs as a task of units
// (prefill command with the task flag, then prefill_unit commands), one unit after each batch step of the decoding
// requests; rank 1 runs the same units in the same order. STRIX_DS41_SERVE_PREFILL_LOG=1: rank 1 logs its prefills'
// next token and logits FNV too (exactness gates).
#include "strix/deepseek41_attention_state.hpp"
#include "strix/deepseek41_state_frames.hpp"
#include "strix/handoff.hpp"
#include "strix/deepseek41_batch_dspark.hpp"

#include "ds41_hetero_stage.hpp"
#include "strix/deepseek41_batch_forward.hpp"
#include "strix/deepseek41_batch_schedule.hpp"
#include "strix/deepseek41_pair_defaults.hpp"
#include "strix/ds41_dspark_files.hpp"
#include "strix/hip_ds41_dspark.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/deepseek41_serve_cache_policy.hpp"
#include "strix/deepseek41_cache_lease.hpp"
#include "strix/deepseek41_session_cache.hpp"
#include "strix/serve/session_api.hpp"
#include "strix/deepseek41_describe.hpp"
#include "strix/deepseek41_serve_observe.hpp"
#include "strix/deepseek41_serve_sched.hpp"
#include "strix/ds41_expert_tp.hpp"
#include "strix/ds41_trellis.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/deepseek_chat.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

using strix::deepseek41::BatchForward;
using strix::deepseek41::BatchRow;
using strix::deepseek41::BatchSequence;
namespace serve = strix::serve;
using Clock = std::chrono::steady_clock;

struct Args : strix::deepseek41::ServeCachePolicy {
    std::string model, transport = "usb4auto", manifest, http_bind = "127.0.0.1", api_key_file, model_id = "deepseek-v4.1-flash";
    std::string group_id = "local";
    std::uint64_t epoch = 0;
    std::uint32_t rank = 0, layers = 40, context = 4096, slots = 4, prompt_rows = 8, http_port = 0, max_queue = 64, max_connections = 64;
    std::uint32_t pool = 0;  // BatchForwardOptions::pool_positions (0: every slot owns `context` positions)
    std::uint32_t default_max_tokens = 1024;
    bool http_responses = true, http_capabilities = true, http_stream_receipts = true;  // ApiConfig defaults (on)
    double http_sse_progress_interval = 1.0;
    double http_sse_keepalive = 10.0;                                                    // seconds, 0: off
    std::uint32_t import_parked = 1;  // imported states parked in slots at most (POST /admin/import-state)
    std::uint32_t import_frames_port = 0;  // --import-frames PORT: the same imports as strix::handoff sessions (0: off)
    unsigned connect_timeout = 60, peer_timeout = 60;
    bool tp = true, engram_direct = true, describe = false, describe_parts = false;
    bool device_prefill = true;                      // --prefill-path device (default since P15; rows = the batch steps)
    std::uint32_t prefill_rows = 4096, prefill_min = 0;
    bool prefill_rows_auto = false;  // --prefill-rows auto: resolved after the weights load (auto_prefill_rows)
    std::string session_dir;
    std::uint64_t session_ram_bytes = 0; // injected by pair_defaults after route/memory checks
    std::uint64_t session_max_parked = 8, session_disk_bytes = 0;

    bool session_explicit = false;
    // DSpark windows inside the batch steps (--dspark DSPARK.gguf): the drafter on rank 0, one drafting context per slot
    std::string dspark, dspark_window = "controller", dspark_vocab = "full";
    std::uint32_t dspark_ranks = 2;  // 2 (default since P17.2): a drafter on each rank, requests draft on rank slot % 2; 1: rank 0
    std::uint32_t dspark_max_inflight = 4;  // --dspark-max-inflight: draft only while <= N requests are in flight (0: always)
    std::uint32_t dspark_seed_rows = 128;  // the device prefill keeps the inputs of its last rows for the drafter (0: none)
    std::string hetero_listen, hetero_token;  // split decode (ds41_hetero_stage.hpp)
    std::uint32_t hetero_cut = 22, hetero_groups = 2;
};

// Fill only compatible implicit settings, then apply order-independent OFF vetoes.
void apply_cache_profile(Args& a) {
    const char* dir = std::getenv("STRIX_DS41_PREFIX_DIR");
    const char* home = std::getenv("HOME");
    a.resolve(a.device_prefill, !a.hetero_listen.empty(), !a.dspark.empty(), dir ? dir : "", home ? home : "");
}

void event(const serve::Json& e) {
    std::printf("%s\n", e.dump().c_str());
    std::fflush(stdout);
}
[[noreturn]] void poisoned(std::uint32_t rank, const std::string& reason) {
    serve::Json e = serve::Json::object();
    e.set("event", serve::Json("POISONED"));
    e.set("rank", serve::Json(static_cast<std::uint64_t>(rank)));
    e.set("reason", serve::Json(reason));
    event(e);
    std::_Exit(3);
}

std::string file_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("cannot read " + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::uint64_t fnv(std::string_view s) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const char c : s) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ULL;
    return h;
}

std::string executable_sha256() {
    static const std::string sha = strix::model_content_identity_hex(strix::sha256_file_content("/proc/self/exe"));
    return sha;
}

// strix/group-manifest-v1 (TensorMeld): the fields this engine uses override the command line.
void apply_manifest(Args& a) {
    const auto m = serve::parse_json(file_text(a.manifest));
    const auto str = [&](const serve::Json* v) { return v && v->is_string() ? v->as_string() : std::string(); };
    const auto num = [&](const serve::Json* v, std::uint64_t fallback) { return v && v->is_number() ? static_cast<std::uint64_t>(v->as_int()) : fallback; };
    if (const auto schema = str(m.find("schema")); schema != "strix/group-manifest-v1")
        throw std::invalid_argument("group manifest: schema must be strix/group-manifest-v1, not '" + schema + "'");
    a.group_id = str(m.find("group_id"));
    if (a.group_id.empty()) throw std::invalid_argument("group manifest: group_id is required");
    a.epoch = num(m.find("epoch"), 0U);
    a.rank = static_cast<std::uint32_t>(num(m.find("rank"), a.rank));
    if (num(m.find("world"), 2U) != 2U) throw std::invalid_argument("group manifest: this engine group has world 2");
    if (const auto* t = m.find("transport"); t && t->is_object()) {
        const auto kind = str(t->find("kind"));
        if (kind == "usb4_host_auto") a.transport = "usb4auto";
        else if (kind == "usb4_gpu_auto") a.transport = "usb4gpu";  // GPU-direct streams where the layout has them
        else if (!kind.empty()) throw std::invalid_argument("group manifest: unsupported transport kind " + kind);
        a.connect_timeout = static_cast<unsigned>(num(t->find("connect_timeout_s"), a.connect_timeout));
    }
    if (const auto* h = m.find("http"); h && h->is_object()) {
        if (const auto b = str(h->find("bind")); !b.empty()) a.http_bind = b;
        a.http_port = static_cast<std::uint32_t>(num(h->find("port"), a.http_port));
        if (const auto k = str(h->find("api_key_file")); !k.empty()) a.api_key_file = k;
    }
    a.context = static_cast<std::uint32_t>(num(m.find("context"), a.context));
    a.slots = static_cast<std::uint32_t>(num(m.find("slots"), a.slots));
    a.pool = static_cast<std::uint32_t>(num(m.find("pool_positions"), a.pool));
    if (const auto* model = m.find("model"); model && model->is_object())
        if (const auto id = str(model->find("id")); !id.empty()) a.model_id = id;
}

strix::deepseek41::BatchForwardOptions forward_options(const Args& a, strix::dist::Communicator* comm, std::uint32_t rank) {
    strix::deepseek41::BatchForwardOptions o;
    o.layers = a.layers;
    o.max_positions = a.context;
    o.slots = a.slots;
    o.pool_positions = a.pool;
    o.rank = rank;
    o.world = 2U;
    o.communicator = comm;
    o.tp_attention = a.tp;
    o.engram_direct_io = a.engram_direct;
    o.capture_main_hidden = !a.dspark.empty() && (rank == 0U || a.dspark_ranks == 2U);  // the drafters' input rows
    return o;
}

// ---- the control channel: one fixed-size block per round, rank 0 -> rank 1 (rank 1 answers with its own) ----
enum class Command : std::uint32_t { hello = 1, step = 2, heartbeat = 3, shutdown = 4, prefill = 5, import_state = 6, prefill_unit = 7, quiesce = 8,
    session = 9
};
struct Control {
    std::uint32_t magic = 0x43425344U;  // "DSBC"
    std::uint32_t command = 0;
    std::uint64_t round = 0;
    std::uint32_t rows = 0, resets = 0;  // resets: bit s = reset slot s before the step
    BatchRow row[BatchForward::max_rows]{};
    // hello
    std::uint64_t group_hash = 0, epoch = 0, model_bytes = 0, exe_hash = 0;
    std::uint32_t layers = 0, slots = 0, context = 0, tp = 0;
    // KV pool: rank 0's reserve/release operations since the last step, in order (op = kind << 30 | slot << 24 |
    // positions; kind 1 release, 2 reserve), replayed by rank 1 before the step
    std::uint32_t pool = 0, ops = 0;
    std::uint32_t op[63]{};  // 63: the block keeps 512 bytes with the scheduler fields
    // prefill: slot, prompt tokens (they follow in exchanges of kTokensPerMessage), the cache's resume boundary, shared
    // boundary and key as rank 0 found them, flags (bit 0: insert the boundaries); hello: the prefill configuration
    std::uint32_t pslot = 0, plen = 0, presume = 0, pshared = 0, pflags = 0;
    std::uint32_t commits = 0;  // DSpark windows: the count of commit[] below
    std::uint32_t pbudget = 0;  // STRIX_DS41_SERVE_SCHED=interleave: the prefill_unit budget
    std::uint64_t pkey = 0;
    std::uint32_t prefill_mode = 0, prefill_rows = 0, prefix_block = 0;
    std::uint32_t pend = 0;  // STRIX_DS41_SERVE_SCHED=interleave: the prefill_unit end row; step: prompt-lookup window slots
    std::uint64_t prefix_bytes = 0;
    // disk tier and session snapshots (hello: configuration; step/prefill/heartbeat: snaps = slots whose state at a
    // block boundary rank 0 inserted after the last step, inserted by rank 1 before anything else)
    std::uint64_t prefix_disk = 0;
    std::uint32_t prefix_flags = 0, snaps = 0;
    std::uint64_t prefix_ns = 0;  // hello: FNV of the serve identity (sched::serve_identity_text: weights content, decode, DSpark)
    // import_state: the slot in pslot, the state's byte count in prefix_bytes, its digest in pkey (the bytes follow in
    // messages of kImportChunk bytes; the pool operations come first). Rank 1's reply of the next round carries the
    // digest it imported in pkey.
    // DSpark windows: the commits of rank 0's last step that dropped rows (slot << 16 | kept), applied by rank 1 before
    // anything else of this round (their count, commits, sits above)
    std::uint32_t commit[BatchForward::max_rows]{};
    // STRIX_DS41_SERVE_SCHED=interleave (pend, pbudget above): a prefill command with pflags bit 2 starts a PrefillTask;
    // prefill_unit runs its next unit up to row pend with budget pbudget (pflags bit 3: abort the task instead; bit 4:
    // completion in the boundary acknowledgement, with rows in presume and unit position in pbudget).
    // Extension only exchanged for steps with >8 rows; the original 512-byte block
    // and all of its field offsets are retained for smaller steps.
    BatchRow extra_rows[BatchForward::extended_max_rows - BatchForward::max_rows]{};
};
constexpr std::uint32_t kPrefillTask = 4U, kPrefillAbort = 8U, kPrefillDone = 16U;
// prefill command, pflags bits 8..31: the request's stable prefix (system prompt + tools) in cache blocks
constexpr std::uint32_t kPrefillHintShift = 8U;
constexpr std::uint32_t kTokensPerMessage = 16384U;  // 64 KiB per exchange_host
constexpr std::uint32_t kPoolRelease = 1U, kPoolReserve = 2U;
constexpr std::size_t kImportChunk = std::size_t{64} << 10U;  // BatchForward::exchange_host's message limit
static_assert(offsetof(Control, extra_rows) <= 512U);

Control exchange(BatchForward& fwd, const Control& out) {
    std::array<std::byte, 512> o{}, i{};
    std::memcpy(o.data(), &out, offsetof(Control, extra_rows));
    fwd.exchange_host(o, i);
    Control in;
    std::copy_n(i.begin(), offsetof(Control, extra_rows), std::as_writable_bytes(std::span(&in, 1)).begin());
    if (in.magic != out.magic) throw std::runtime_error("control channel: bad magic from the peer");
    if (std::max(out.rows, in.rows) > BatchForward::max_rows) {
        if (std::max(out.rows, in.rows) > fwd.row_limit()) throw std::runtime_error("control channel: step row limit differs");
        fwd.exchange_host(std::as_bytes(std::span(out.extra_rows)), std::as_writable_bytes(std::span(in.extra_rows)));
    }
    return in;
}

// Both workers are drained here; agree before finish() or the next unit can enter collectives.
void check_unit(BatchForward& fwd, std::uint32_t rank, strix::deepseek41::sched::UnitCompletion mine) {
    Control ack;
    ack.command = static_cast<std::uint32_t>(Command::prefill_unit);
    ack.pflags = mine.done ? kPrefillDone : 0U;
    ack.presume = mine.rows;
    ack.pbudget = mine.units;
    const auto peer = exchange(fwd, ack);
    const strix::deepseek41::sched::UnitCompletion other{(peer.pflags & kPrefillDone) != 0U, peer.presume, peer.pbudget};
    if (peer.command != ack.command || !(mine == other)) poisoned(rank, "prefill task: unit completion differs between ranks");
}

// The drafter's device bytes on a drafting rank (DeviceDrafter::footprint over a.slots contexts + the speculator's
// kept-row staging), against what the device actually lost while they were set up (logged before READY).
std::uint64_t dspark_footprint_bytes(const strix::ds41::dspark::Weights& w, const Args& a, std::string* detail = nullptr) {
    const auto f = strix::ds41::dspark::DeviceDrafter::footprint(w, a.slots);
    const auto staging = strix::deepseek41::BatchSpeculator::staging_bytes(a.slots);
    const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / 1048576.0; };
    if (detail) {
        char b[200];
        std::snprintf(b, sizeof b, "weights %.1f + scratch %.1f + contexts %.1f + staging %.1f MiB", mib(f.weights), mib(f.scratch), mib(f.contexts),
                      mib(staging));
        *detail = b;
    }
    return f.total() + staging;
}
// The same drop once the host copy of the drafter's weights is freed (just before READY): what the drafter keeps.
void log_dspark_resident(std::uint32_t rank, std::uint64_t free_before, std::uint64_t free_now) {
    const auto fell = free_before > free_now ? free_before - free_now : std::uint64_t{0};
    std::fprintf(stderr, "rank %u: DSpark memory before READY (host copy freed): device free %.1f MiB below its level before the drafter\n", rank,
                 static_cast<double>(fell) / 1048576.0);
}
void log_dspark_memory(std::uint32_t rank, const strix::ds41::dspark::Weights& w, const Args& a, std::uint64_t free_before, std::uint64_t free_after) {
    std::string detail;
    const auto est = dspark_footprint_bytes(w, a, &detail);
    const auto fell = free_before > free_after ? free_before - free_after : std::uint64_t{0};
    std::fprintf(stderr, "rank %u: DSpark memory: footprint %.1f MiB (%s); device free fell by %.1f MiB\n", rank, static_cast<double>(est) / 1048576.0,
                 detail.c_str(), static_cast<double>(fell) / 1048576.0);
}

// DSpark with a drafter on each rank: a request drafting alone in a step is drafted by both ranks, each over half the
// output head (BatchSpecOptions::coop; exact, the same draft). STRIX_DS41_DSPARK_COOP=0: by its owner rank alone.
// Both ranks must agree: the hello compares it.
bool dspark_coop_on() {
    const char* env = std::getenv("STRIX_DS41_DSPARK_COOP");
    return env == nullptr || std::strcmp(env, "0") != 0;
}

Control hello(const Args& a) {
    Control c;
    c.command = static_cast<std::uint32_t>(Command::hello);
    c.group_hash = fnv(a.group_id);
    c.epoch = a.epoch;
    c.model_bytes = std::filesystem::file_size(a.model);
    c.exe_hash = fnv(executable_sha256());
    c.layers = a.layers;
    c.slots = a.slots;
    c.context = a.context;
    c.tp = (a.tp ? 1U : 0U) | (BatchForward::configured_row_limit() > BatchForward::max_rows ? 2U : 0U) | (dspark_coop_on() ? 4U : 0U);
    c.pool = a.pool;
    c.prefill_mode = a.device_prefill ? 1U : 0U;
    c.prefill_rows = a.prefill_rows_auto ? 0U : a.prefill_rows;  // auto: resolved by both ranks after the load
    c.prefix_block = a.prefix_block;
    c.prefix_bytes = a.prefix_cache_mb << 20U;
    c.prefix_disk = a.prefix_dir.empty() ? 0U : (a.prefix_disk_mb << 20U) + 1U;
    c.prefix_flags = (a.prefix_before_tail ? 1U : 0U) | (a.prefix_session ? 2U : 0U) | (a.prefix_inrun ? 4U : 0U) |
                     (a.prefix_inrun ? std::min<std::uint32_t>(a.prefix_inrun_reach, 0xFFFFFFU) << 8U : 0U);  // the cone: both ranks alike
    // serve keeps expert TP off unless STRIX_DS41_EXPERT_TP=1 (then the rows path): the decode identity of fwd.expert_tp().
    // Always compared (not only with a disk tier): ranks with different weights content, decode identity or DSpark setup
    // exchange the same bytes and tags, which no CRC or framing check can tell apart.
    const auto ns = strix::deepseek41::prefix_namespace_from_env(strix::ds41::expert_tp::enabled_from_env()).text;
    c.prefix_ns = fnv(strix::deepseek41::sched::serve_identity_text(
        ns, a.dspark.empty() ? 0U : std::filesystem::file_size(a.dspark), a.dspark_ranks, a.dspark_window, a.dspark_vocab));
    // Resource limits occupy previously zero hello fields. OFF keeps the original
    // 512 bytes; defaults sampled on different hosts negotiate a common minimum.
    c.pkey = a.session_ram_bytes;
    if (c.pkey) {
        c.plen = static_cast<std::uint32_t>(a.session_max_parked);
        c.pflags = a.session_dir.empty() ? 0U : 1U;
        c.pshared = static_cast<std::uint32_t>(a.session_disk_bytes);
        c.presume = static_cast<std::uint32_t>(a.session_disk_bytes >> 32U);
    }
    return c;
}

void agree_session_limits(Args& a, const Control& peer) {
    const auto before = a.session_ram_bytes;
    const bool had_disk = !a.session_dir.empty();
    a.session_ram_bytes = std::min(a.session_ram_bytes, peer.pkey);
    a.session_max_parked = std::min(a.session_max_parked, static_cast<std::uint64_t>(peer.plen));
    if (!a.session_max_parked) a.session_ram_bytes = 0;
    if (!a.session_ram_bytes || !peer.pflags) a.session_dir.clear();
    const auto peer_disk = static_cast<std::uint64_t>(peer.pshared) | (static_cast<std::uint64_t>(peer.presume) << 32U);
    a.session_disk_bytes = std::min(a.session_disk_bytes, peer_disk);
    if (!a.session_disk_bytes) a.session_dir.clear();
    if (before && !a.session_ram_bytes)
        std::fprintf(stderr, "session cache: off (peer precondition or off switch)\n");
    else if (had_disk && a.session_dir.empty())
        std::fprintf(stderr, "session disk: off (peer precondition or off switch)\n");
}

std::string hello_mismatch(const Control& mine, const Control& peer) {
    if (peer.command != static_cast<std::uint32_t>(Command::hello)) return "the peer did not answer the hello";
    if (peer.group_hash != mine.group_hash) return "group_id differs between the ranks";
    if (peer.epoch != mine.epoch) return "epoch differs between the ranks";
    if (peer.exe_hash != mine.exe_hash) return "the engine executables differ (SHA-256)";
    if (peer.model_bytes != mine.model_bytes) return "the model files differ (size)";
    if ((peer.tp ^ mine.tp) & 4U) return "STRIX_DS41_DSPARK_COOP differs between the ranks";
    if (peer.layers != mine.layers || peer.slots != mine.slots || peer.context != mine.context || peer.tp != mine.tp || peer.pool != mine.pool)
        return "layers, slots, context, KV pool or tensor-parallel attention differ between the ranks";
    if (peer.prefill_mode != mine.prefill_mode || peer.prefill_rows != mine.prefill_rows || peer.prefix_block != mine.prefix_block ||
        peer.prefix_bytes != mine.prefix_bytes || peer.prefix_disk != mine.prefix_disk || peer.prefix_flags != mine.prefix_flags)
        return "the prefill path or the prefix cache differ between the ranks";
    if (peer.prefix_ns != mine.prefix_ns)
        return "the serve identities differ between the ranks (weights overlay / trellis / EP map content, EP vs expert TP, or DSpark drafter / ranks / window / vocabulary)";
    return {};
}

// Watchdog: the process ends (POISONED) if a round with the peer has not completed for `limit` seconds.
class Watchdog {
public:
    Watchdog(std::uint32_t rank, unsigned limit) : rank_(rank), limit_(limit), thread_([this] { run(); }) {}
    ~Watchdog() {
        stop_.store(true);
        thread_.join();
    }
    void progress() { last_.store(Clock::now().time_since_epoch().count()); }
    void pause(bool p) {
        paused_.store(p);
        progress();
    }

private:
    void run() {
        while (!stop_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (paused_.load()) continue;
            const auto idle = Clock::now().time_since_epoch().count() - last_.load();
            if (std::chrono::duration<double>(Clock::duration(idle)).count() > limit_)
                poisoned(rank_, "no round with the peer for " + std::to_string(limit_) + " s");
        }
    }
    std::uint32_t rank_;
    unsigned limit_;
    std::atomic<std::int64_t> last_{Clock::now().time_since_epoch().count()};
    std::atomic<bool> stop_{false}, paused_{false};
    std::thread thread_;
};

// ---- the device prefill of a prompt into a request's slot: both ranks run it with the same arguments ----
std::uint64_t fnv_floats(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

// MemAvailable of this node in MiB (0 when unknown).
std::uint64_t mem_available_mb() {
    std::ifstream in("/proc/meminfo");
    for (std::string line; std::getline(in, line);)
        if (line.rfind("MemAvailable:", 0) == 0) return std::stoull(line.substr(13)) / 1024U;
    return 0;
}

struct PrefillOutcome {
    std::uint32_t next_token = 0, resume = 0, pieces = 0, inserted = 0;
    strix::deepseek41::DevicePrefill::DraftSeed seed;  // the last piece's seed rows (none for a whole-prompt cache hit)
    std::uint64_t logits = 0;
    double state_seconds = 0.0, prefill_seconds = 0.0, handoff_seconds = 0.0, seconds = 0.0;
    double insert_seconds = 0.0;  // part of prefill_seconds: the state exports and cache inserts at piece ends
};

// The row-split prefill of two ranks gives each 8-row window to one rank alternately: a run needs more than 8 rows so
// that each rank has a row. Shorter prompts take the rows path; pieces and resume points leave at least this many rows.
constexpr std::uint32_t kMinRows = 9U;

// The serve's device prefill options (both ranks; the weights read in place from the batch forward).
strix::deepseek41::PrefillOptions slot_prefill_options(const Args& a, BatchForward& fwd, strix::dist::Communicator& comm, std::uint32_t rank) {
    strix::deepseek41::PrefillOptions o;
    o.layers = a.layers;
    o.max_positions = a.context;
    o.max_rows = std::min(a.prefill_rows, a.context);
    o.rank = rank;
    o.world = 2U;
    o.communicator = &comm;
    o.engram_direct_io = a.engram_direct;
    o.shared_weights = fwd.file_form_weights();
    o.expert_tp = strix::ds41::expert_tp::enabled_from_env();  // the batch forward's expert TP (dup layout)
    if (!a.dspark.empty()) o.draft_rows = a.dspark_seed_rows;  // the drafter's seed (both ranks hold it)
    return o;
}

class SlotPrefill {
public:
    SlotPrefill(const Args& a, BatchForward& fwd, strix::dist::Communicator& comm, std::uint32_t rank) : a_(a), fwd_(fwd) {
        const auto o = slot_prefill_options(a, fwd, comm, rank);
        prefill_ = std::make_unique<strix::deepseek41::DevicePrefill>(a.model, o);
        if (a.prefix_cache_mb != 0U) {
            strix::deepseek41::PrefixCacheOptions co;
            co.block = a.prefix_block;
            host_lease_ = std::make_unique<strix::deepseek41::PrefixHostLease>(a.prefix_cache_mb);
            host_budget_mb_ = host_lease_->budget_mb();
            co.budget_bytes = std::max<std::uint64_t>(1U, host_budget_mb_ << 20U);
            if (!a.prefix_dir.empty()) {
                // one directory per namespace: states of another configuration (or of code without a namespace) are never read
                const auto ns = strix::deepseek41::prefix_namespace_from_env(fwd.expert_tp());
                const std::filesystem::path base = std::filesystem::path(a.prefix_dir) / ns.tag;
                co.dir = base / ("rank" + std::to_string(rank));
                co.disk_budget_bytes = a.prefix_disk_mb << 20U;
                co.min_free_bytes = 16384ULL << 20U;
                co.quota_dir = a.prefix_dir;
                co.promote = false;  // retention admission is collective; disk reads never insert unilaterally
                co.disk_admission = [&](bool local) {
                    std::array<std::uint64_t, 1> mine{local ? 1U : 0U}, peer{};
                    fwd_.exchange_host(std::as_bytes(std::span(mine)), std::as_writable_bytes(std::span(peer)));
                    return local && peer[0] != 0U;
                };
                co.model_id = strix::deepseek41::prefix_model_id(a.model, a.layers, ns);
                std::filesystem::create_directories(base);
                std::ofstream(base / "NAMESPACE") << ns.text << '\n';  // each node's copy, for people
                std::fprintf(stderr, "rank %u: prefix cache namespace %s: %s\n", rank, ns.tag.c_str(), ns.text.c_str());
            }
            cache_ = std::make_unique<strix::deepseek41::PrefixCache>(co);
            if (cache_->has_disk()) {
                const auto st = cache_->stats();
                std::fprintf(stderr, "rank %u: prefix cache disk tier %s: %llu files read in %.3f s, %llu ignored; %llu blocks + %llu snapshots (%.1f MB)\n", rank,
                             co.dir.string().c_str(), static_cast<unsigned long long>(st.scanned_files), st.scan_seconds,
                             static_cast<unsigned long long>(st.ignored_files), static_cast<unsigned long long>(st.disk_blocks),
                             static_cast<unsigned long long>(st.disk_snapshots), static_cast<double>(st.disk_bytes) / 1e6);
            }
        }
        hist_.resize(a.slots);
        last_out_.assign(a.slots, 0U);
        const std::string cache = cache_ ? ", prefix cache " + std::to_string(a.prefix_cache_mb) + " MiB in blocks of " + std::to_string(a.prefix_block) : "";
        std::fprintf(stderr, "rank %u: device prefill (pieces of %u rows) reads %.2f GiB of the batch forward's weights in place, loaded %.2f GiB%s\n", rank,
                     o.max_rows, static_cast<double>(prefill_->shared_weight_bytes()) / 1073741824.0,
                     static_cast<double>(prefill_->weight_bytes()) / 1073741824.0, cache.c_str());
    }
    void record_export(const strix::deepseek41::AttentionState& state, unsigned kind) {
        if (!diagnostics_) return;
        export_bytes_ += state.bytes() + state.tokens.size() * 4U;
        ++exports_.at(kind);
    }
    // Fresh observations from both nodes, before either plans a capture or exports a slot.
    bool admit(std::uint32_t tokens, bool requested = true, std::uint32_t slots = 1) {
        const std::uint64_t rings = std::uint64_t{a_.layers} * (136U * 528U + 4096U + 4096U);
        const std::uint64_t state = std::uint64_t{tokens} * (std::uint64_t{a_.layers} * 356U + 4U) + rings;
        // Full export + retained copy + snapshot rings and disk serialization, conservatively
        // treating every layer as a ratio-1 KV owner. Existing retention is already in MemAvailable.
        const std::uint64_t predicted = (host_budget_mb_ << 20U) + slots * (state * 3U + rings * 8U);
        std::array<std::uint64_t, 3> mine{mem_available_mb(), predicted, 0}, peer{};
        mine[2] = requested && cache_ && host_budget_mb_ != 0U &&
                  strix::deepseek41::prefix_memory_admitted(mine[0], a_.prefix_reserve_mb, predicted);
        fwd_.exchange_host(std::as_bytes(std::span(mine)), std::as_writable_bytes(std::span(peer)));
        ++capture_checks_;
        capture_peak_ = std::max(capture_peak_, predicted);
        const bool ok = mine[2] != 0U && peer[2] != 0U;
        if (!ok) {
            ++capture_bypasses_;
            std::fprintf(stderr, "prefix capture bypass: local/peer available %llu/%llu MiB, predicted %llu/%llu bytes, reserve %llu MiB, host ceiling %llu MiB\n",
                         static_cast<unsigned long long>(mine[0]), static_cast<unsigned long long>(peer[0]),
                         static_cast<unsigned long long>(predicted), static_cast<unsigned long long>(peer[1]),
                         static_cast<unsigned long long>(a_.prefix_reserve_mb), static_cast<unsigned long long>(host_budget_mb_));
        }
        return ok;
    }
    [[nodiscard]] bool has_cache() const noexcept { return cache_ != nullptr; }
    void set_moe_chunk_rows(std::uint32_t r) noexcept { moe_chunk_rows_ = r; }
    [[nodiscard]] std::uint64_t weight_bytes() const noexcept { return prefill_->weight_bytes(); }  // loaded (owned) by the prefill
    strix::deepseek41::PrefixLookup lookup(std::span<const std::uint32_t> prompt) {
        return cache_ ? cache_->lookup(prompt, kMinRows) : strix::deepseek41::PrefixLookup{};
    }
    [[nodiscard]] strix::deepseek41::PrefixCacheStats cache_stats() const {
        auto st = cache_ ? cache_->stats() : strix::deepseek41::PrefixCacheStats{};
        st.capture_checks = capture_checks_; st.capture_bypasses = capture_bypasses_; st.capture_predicted_peak = capture_peak_;
        st.host_ceiling_bytes = host_budget_mb_ << 20U;
        st.export_bytes = export_bytes_; st.export_inrun = exports_[0]; st.export_run = exports_[1];
        st.export_handoff = exports_[2]; st.export_session = exports_[3];
        st.row_layers = row_layers_; st.planned_splits = planned_splits_;
        return st;
    }
    [[nodiscard]] bool has_disk() const noexcept { return cache_ && cache_->has_disk(); }

    // After a restart: both ranks keep the disk entries both hold (same bytes). Both ranks call it after the hello.
    void reconcile(BatchForward& fwd, std::uint32_t rank) {
        if (!has_disk()) return;
        const auto mine = cache_->inventory();
        constexpr std::size_t kMsg = 65536U, kItem = sizeof(strix::deepseek41::PrefixDiskItem);
        std::vector<std::byte> o(kMsg), i(kMsg);
        const std::uint64_t n = mine.size();
        std::memcpy(o.data(), &n, 8);
        fwd.exchange_host(o, i);
        std::uint64_t peer_n = 0;
        std::memcpy(&peer_n, i.data(), 8);
        std::vector<strix::deepseek41::PrefixDiskItem> peer(peer_n);
        const std::size_t per = kMsg / kItem;
        for (std::size_t at = 0; at < std::max<std::uint64_t>(n, peer_n); at += per) {
            std::fill(o.begin(), o.end(), std::byte{0});
            if (at < n) std::memcpy(o.data(), mine.data() + at, std::min<std::size_t>(per, n - at) * kItem);
            fwd.exchange_host(o, i);
            if (at < peer_n) std::memcpy(peer.data() + at, i.data(), std::min<std::size_t>(per, peer_n - at) * kItem);
        }
        const auto dropped = cache_->retain(peer);
        const auto st = cache_->stats();
        std::fprintf(stderr, "rank %u: prefix cache disk tier reconciled with the peer (%llu vs %llu items): %llu dropped, %llu blocks + %llu snapshots kept\n", rank,
                     static_cast<unsigned long long>(n), static_cast<unsigned long long>(peer_n), static_cast<unsigned long long>(dropped),
                     static_cast<unsigned long long>(st.disk_blocks), static_cast<unsigned long long>(st.disk_snapshots));
    }

    // Reads the hit's state (RAM and disk) without changing the cache; false when a disk file failed.
    bool prepare(std::span<const std::uint32_t> prompt, const strix::deepseek41::PrefixLookup& hit) {
        loaded_ = {};
        disk_bytes_ = 0;
        disk_seconds_ = 0.0;
        if (!cache_ || hit.resume == 0U) return true;
        try {
            loaded_ = cache_->load(prompt, hit.resume, hit.key);
            disk_bytes_ = cache_->last_load().disk_bytes;
            disk_seconds_ = cache_->last_load().disk_seconds;
            return true;
        } catch (const strix::deepseek41::PrefixCacheReadError& e) {
            std::fprintf(stderr, "prefix cache: %s\n", e.what());
            return false;
        }
    }
    void forget(std::span<const std::uint32_t> prompt, const strix::deepseek41::PrefixLookup& hit) {
        loaded_ = {};
        cache_->forget(prompt, hit.resume);
    }
    [[nodiscard]] std::uint64_t disk_bytes() const noexcept { return disk_bytes_; }
    [[nodiscard]] double disk_seconds() const noexcept { return disk_seconds_; }

    // ---- session snapshots: each slot's tokens, tracked the same way on both ranks ----
    void on_reset(std::uint32_t slot) { hist_.at(slot).clear(); }
    void on_prefill(std::uint32_t slot, const std::vector<std::uint32_t>& prompt) { hist_.at(slot) = prompt; }
    // After a step: the rows' tokens appended; returns the slots now exactly at a block boundary (candidates).
    std::uint32_t on_step(std::span<const BatchRow> rows, const std::vector<std::uint32_t>& tokens) {
        std::uint32_t touched = 0;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            hist_.at(rows[r].slot).push_back(rows[r].token);
            last_out_.at(rows[r].slot) = tokens[r];
            touched |= 1U << rows[r].slot;
        }
        if (!cache_ || !a_.prefix_session) return 0U;
        std::uint32_t out = 0;
        for (std::uint32_t slot = 0; slot < a_.slots; ++slot)
            if (((touched >> slot) & 1U) != 0U) {
                const std::uint32_t p = fwd_.position(slot);
                if (p != 0U && p % a_.prefix_block == 0U && hist_[slot].size() == p) out |= 1U << slot;
            }
        return out;
    }
    // Inserts the state of every slot of `mask` (at a block boundary since the last step).
    void snapshot(std::uint32_t mask) {
        std::uint32_t tokens = 0, slots = 0;
        for (std::uint32_t slot = 0; slot < a_.slots; ++slot)
            if (((mask >> slot) & 1U) != 0U) { tokens = std::max(tokens, fwd_.position(slot)); ++slots; }
        if (!admit(tokens, true, slots)) return;
        for (std::uint32_t slot = 0; slot < a_.slots; ++slot) {
            if (((mask >> slot) & 1U) == 0U) continue;
            const auto t0 = Clock::now();
            const std::uint32_t p = fwd_.position(slot);
            const auto& h = hist_.at(slot);
            if (h.size() != p) throw std::runtime_error("session snapshot: token history out of step");
            const std::uint32_t from = cache_->covered(h, p);
            auto st = fwd_.export_state(slot, from);
            st.tokens = h;
            st.next_token = last_out_[slot];
            record_export(st, 3);
            const bool ok = cache_->insert(st, 0U, from);
            std::fprintf(stderr, "session snapshot slot %u at %u (KV from %u): %s in %.3f s\n", slot, p, from, ok ? "inserted" : "refused",
                         std::chrono::duration<double>(Clock::now() - t0).count());
        }
    }

    // The prompt's state into `slot` (reserved and reset) as a task of units (deepseek41_serve_sched.hpp PrefillTask):
    // resumed from `hit` (a lookup of this cache), then the prefill pieces (DevicePrefill run / extend) up to the ends
    // the scheduler gives; every end of the default piece plan on a block boundary is inserted when `insert`.
    // finish() imports the state into the slot. run() is the whole prefill in the default pieces.
    class Task final : public strix::deepseek41::sched::PrefillTask {
    public:
        Task(SlotPrefill& sp, std::uint32_t slot, const std::vector<std::uint32_t>& prompt, const strix::deepseek41::PrefixLookup& hit, bool insert,
             std::uint32_t stable = 0)
            : sp_(sp), slot_(slot), prompt_(prompt), hit_(hit), insert_(insert), t0_(Clock::now()) {
            namespace ds = strix::deepseek41;
            const auto N = static_cast<std::uint32_t>(prompt_.size());
            const std::uint32_t rows = std::min(sp_.a_.prefill_rows, sp_.a_.context);
            out_.resume = hit.resume;
            at_ = hit.resume;
            if (hit.resume == N) {  // the whole prompt is a cached boundary (read by prepare())
                sp_.cache_->use(sp_.loaded_, hit.key);
                state_ = std::move(sp_.loaded_);
                sp_.loaded_ = {};
                out_.next_token = hit.next_token;
                out_.logits = hit.logits_fnv;
                out_.state_seconds = std::chrono::duration<double>(Clock::now() - t0_).count();
                return;
            }
            if (hit.resume > 0U) {
                sp_.cache_->use(sp_.loaded_, hit.key);
                sp_.prefill_->import_state(sp_.loaded_);
                sp_.loaded_ = {};
            }
            out_.state_seconds = std::chrono::duration<double>(Clock::now() - t0_).count();
            auto plan = ds::select_prefix_prefill_plan(hit.resume, hit.shared, N, sp_.a_.prefix_block, rows, kMinRows,
                                                       sp_.cache_ && insert_, sp_.a_.prefix_inrun, sp_.a_.prefix_before_tail,
                                                       sp_.a_.prefix_inrun_reach, stable);
            ends_ = std::move(plan.ends);
            points_ = std::move(plan.points);
            if (sp_.diagnostics_) {
                const auto baseline = ds::plan_prefill_pieces(hit.resume, 0U, N, rows, rows, kMinRows);
                for (const auto end : ends_)
                    sp_.planned_splits_ += std::binary_search(baseline.begin(), baseline.end(), end) ? 0U : 1U;
            }
        }
        // budget 0: the piece [at, end) in one DevicePrefill run / extend. budget k > 0 (phase B): k units of the
        // piece's resumable task (DevicePrefill::begin_run / begin_extend), which may stop inside the piece.
        bool step(std::uint32_t end, std::uint32_t budget) override {
            const auto N = static_cast<std::uint32_t>(prompt_.size());
            if (at_ == N) return true;
            // after a piece that deferred its decoder (DevicePrefill::set_decoder_deferral) the continuation it was admitted
            // for runs at least that far (a shorter one would read decoder ring entries the deferred piece never wrote)
            if (!ptask_ && pin_end_ > at_ && end < pin_end_) end = pin_end_;
            if (ptask_ ? end != piece_end_ : (end <= at_ || end > N || (end != N && end % 8U != 0U)))
                throw std::runtime_error("prefill task: bad unit end " + std::to_string(end));
            const auto t1 = Clock::now();
            const std::uint32_t from = at_;
            // in-run snapshot points of this run (both ranks: they widen the cone, so the ranks must agree)
            if (!ptask_) {
                std::vector<std::uint32_t> inside;
                for (const auto p : points_)
                    if (p > at_ && p < end) inside.push_back(p);
                // a piece whose decode state, logits and seed nobody reads (no cache insert at its end, no in-run
                // snapshot in it or in the next piece) defers its decoder cone to the next piece of the default plan
                defer_next_ = 0;
                if (end < N && inside.empty()) {
                    const auto it = std::upper_bound(ends_.begin(), ends_.end(), end);
                    const std::uint32_t next = it == ends_.end() ? N : *it;
                    const bool insert_here = sp_.cache_ && insert_ && end % sp_.a_.prefix_block == 0U && std::binary_search(ends_.begin(), ends_.end(), end);
                    const bool points_next = std::any_of(points_.begin(), points_.end(), [&](std::uint32_t p) { return p > end && p < next; });
                    if (!insert_here && !points_next) {
                        sp_.prefill_->set_decoder_deferral(next - end);
                        defer_next_ = next;
                    }
                }
                sp_.prefill_->set_snapshot_points(std::move(inside));
                // the next piece of the default plan: its Engram rows are read while this piece runs
                if (end < N) {
                    const auto it = std::upper_bound(ends_.begin(), ends_.end(), end);
                    const std::uint32_t next = it == ends_.end() ? N : *it;
                    sp_.prefill_->set_engram_lookahead(std::span<const std::uint32_t>(prompt_.data() + end, next - end));
                }
            }
            if (budget == 0U && !ptask_) {
                const std::span<const std::uint32_t> piece(prompt_.data() + at_, end - at_);
                r_ = at_ == 0U ? sp_.prefill_->run(piece) : sp_.prefill_->extend(piece);
            } else {
                if (!ptask_) {
                    const std::span<const std::uint32_t> piece(prompt_.data() + at_, end - at_);
                    strix::deepseek41::PrefillTaskOptions o;
                    o.level = strix::deepseek41::PrefillUnitLevel::finest;
                    o.moe_chunk_rows = sp_.moe_chunk_rows_;
                    ptask_ = at_ == 0U ? sp_.prefill_->begin_run(piece, o) : sp_.prefill_->begin_extend(piece, o);
                    piece_end_ = end;
                }
                (void)ptask_->step(budget == 0U ? 0xFFFFFFFFU : budget);
                for (const double w : ptask_->step_walls()) step_max_ms_ = std::max(step_max_ms_, w);
                for (const auto& u : ptask_->unit_timings()) unit_max_ms_ = std::max(unit_max_ms_, u.device_ms);
                if (!ptask_->done()) {
                    out_.prefill_seconds += std::chrono::duration<double>(Clock::now() - t1).count();
                    return false;
                }
                r_ = ptask_->result();
                ptask_.reset();
            }
            at_ = end;
            ++out_.pieces;
            pin_end_ = r_.decoder_deferred ? defer_next_ : 0U;
            if (sp_.diagnostics_) sp_.row_layers_ += r_.row_layers;
            if (end != N && sp_.cache_ && insert_ && end % sp_.a_.prefix_block == 0U && std::binary_search(ends_.begin(), ends_.end(), end)) {
                const auto ti = Clock::now();
                auto s = sp_.prefill_->export_state();
                s.tokens.assign(prompt_.begin(), prompt_.begin() + end);
                s.next_token = r_.next_token;
                sp_.record_export(s, 1);
                out_.inserted += sp_.cache_->insert(s, fnv_floats(r_.logits)) ? 1U : 0U;
                out_.insert_seconds += std::chrono::duration<double>(Clock::now() - ti).count();
            }
            if (sp_.cache_ && insert_)
                for (const auto p : points_) {
                    if (p <= from || p >= end) continue;
                    const auto ti = Clock::now();
                    auto s = sp_.prefill_->export_state_at(p);
                    s.tokens.assign(prompt_.begin(), prompt_.begin() + p);
                    s.next_token = strix::deepseek41::kPrefixUnknownNextToken;
                    sp_.record_export(s, 0);
                    out_.inserted += sp_.cache_->insert(s, 0U) ? 1U : 0U;
                    out_.insert_seconds += std::chrono::duration<double>(Clock::now() - ti).count();
                }
            out_.prefill_seconds += std::chrono::duration<double>(Clock::now() - t1).count();
            if (at_ == N) {
                out_.next_token = r_.next_token;
                out_.logits = fnv_floats(r_.logits);
                out_.seed = sp_.prefill_->draft_seed();  // the drafter's seed (dspark-batch)
            }
            return at_ == N;
        }
        [[nodiscard]] bool mid_piece() const noexcept override { return ptask_ != nullptr; }
        [[nodiscard]] std::uint32_t piece_end() const noexcept override { return ptask_ ? piece_end_ : at_; }
        [[nodiscard]] std::uint32_t unit_position() const noexcept { return ptask_ ? ptask_->units_done() : 0U; }
        // rank 0: resumable units of the current piece predicted to fit in budget_ms (1 before the piece's task exists)
        [[nodiscard]] std::uint32_t units_within(double budget_ms) const { return ptask_ ? std::max(1U, ptask_->units_within(budget_ms)) : 1U; }
        // The longest resumable step wall so far (ms); it may contain several indivisible units.
        [[nodiscard]] double step_max_ms() const noexcept { return step_max_ms_; }
        [[nodiscard]] double unit_max_ms() const noexcept { return unit_max_ms_; }
        // Abandon at the drained boundary. Only a fresh run or validated import restores the sequence.
        void cancel() {
            if (ptask_) {
                ptask_->cancel();
                ptask_.reset();
            }
        }
        [[nodiscard]] std::uint32_t rows() const noexcept override { return at_; }
        [[nodiscard]] std::uint32_t total() const noexcept override { return static_cast<std::uint32_t>(prompt_.size()); }
        [[nodiscard]] std::span<const std::uint32_t> default_ends() const noexcept { return ends_; }
        [[nodiscard]] std::uint32_t slot() const noexcept { return slot_; }
        [[nodiscard]] const std::vector<std::uint32_t>& prompt() const noexcept { return prompt_; }
        // after the last unit: the state into the slot
        PrefillOutcome finish() {
            if (!done()) throw std::runtime_error("prefill task: finish before the last unit");
            const auto t2 = Clock::now();
            double final_insert = 0.0;
            if (state_.layers.empty()) {
                state_ = sp_.prefill_->export_state();
                state_.tokens = prompt_;
                state_.next_token = r_.next_token;
                sp_.record_export(state_, 2);
                if (sp_.cache_ && insert_ && total() % sp_.a_.prefix_block == 0U) {
                    const auto ti = Clock::now();
                    out_.inserted += sp_.cache_->insert(state_, out_.logits) ? 1U : 0U;
                    final_insert = std::chrono::duration<double>(Clock::now() - ti).count();
                    out_.insert_seconds += final_insert;
                    out_.prefill_seconds += final_insert;
                }
            }
            sp_.fwd_.import_state(slot_, state_);
            out_.handoff_seconds = std::chrono::duration<double>(Clock::now() - t2).count() - final_insert;
            out_.seconds = std::chrono::duration<double>(Clock::now() - t0_).count();
            return out_;
        }

    private:
        SlotPrefill& sp_;
        std::uint32_t slot_;
        std::vector<std::uint32_t> prompt_;
        strix::deepseek41::PrefixLookup hit_;
        bool insert_;
        Clock::time_point t0_;
        std::uint32_t at_ = 0;
        std::vector<std::uint32_t> ends_;
        std::uint32_t defer_next_ = 0, pin_end_ = 0;  // decoder deferral: the next piece's end asked for / admitted
        std::vector<std::uint32_t> points_;  // --prefix-inrun-snapshots: snapshot points inside the runs (plan_prefill_runs)
        strix::deepseek41::PrefillResult r_;
        strix::deepseek41::AttentionState state_;
        PrefillOutcome out_;
        std::unique_ptr<strix::deepseek41::PrefillTask> ptask_;  // phase B: the current piece's resumable task
        std::uint32_t piece_end_ = 0;
        double step_max_ms_ = 0.0, unit_max_ms_ = 0.0;
    };
    std::unique_ptr<Task> begin(std::uint32_t slot, const std::vector<std::uint32_t>& prompt, const strix::deepseek41::PrefixLookup& hit, bool insert,
                                std::uint32_t stable = 0) {
        return std::make_unique<Task>(*this, slot, prompt, hit, insert, stable);
    }
    // One dummy prompt through the device prefill on both ranks (first launches, scratch first touch). Nothing enters
    // the prefix cache or a slot; the next run is a fresh sequence.
    void warm(std::uint32_t rows) {
        std::vector<std::uint32_t> dummy(std::min({rows, a_.prefill_rows, a_.context}));
        for (std::uint32_t i = 0; i < dummy.size(); ++i) dummy[i] = 1000U + i;
        (void)prefill_->run(dummy);
    }
    PrefillOutcome run(std::uint32_t slot, const std::vector<std::uint32_t>& prompt, const strix::deepseek41::PrefixLookup& hit, bool insert,
                       std::uint32_t stable = 0) {
        auto t = begin(slot, prompt, hit, insert, stable);
        while (!t->done()) (void)t->step(strix::deepseek41::sched::next_unit_end(t->rows(), t->default_ends(), 0U, kMinRows), 0U);
        return t->finish();
    }

private:
    const Args& a_;
    BatchForward& fwd_;
    std::unique_ptr<strix::deepseek41::PrefixHostLease> host_lease_;
    std::uint64_t host_budget_mb_ = 0, capture_checks_ = 0, capture_bypasses_ = 0, capture_peak_ = 0;
    const bool diagnostics_ = std::getenv("STRIX_DS41_PREFIX_DIAGNOSTICS") && std::string_view(std::getenv("STRIX_DS41_PREFIX_DIAGNOSTICS")) == "1";
    std::array<std::uint64_t, 4> exports_{};
    std::uint64_t export_bytes_ = 0, row_layers_ = 0, planned_splits_ = 0;
    std::unique_ptr<strix::deepseek41::DevicePrefill> prefill_;
    std::unique_ptr<strix::deepseek41::PrefixCache> cache_;
    strix::deepseek41::AttentionState loaded_;  // prepare()'s state
    std::uint64_t disk_bytes_ = 0;
    double disk_seconds_ = 0.0;
    std::vector<std::vector<std::uint32_t>> hist_;  // per slot: the tokens of positions [0, position)
    std::vector<std::uint32_t> last_out_;           // per slot: the greedy token of its last row
    std::uint32_t moe_chunk_rows_ = 0;              // phase B tasks' MoE chunk (STRIX_DS41_SERVE_SCHED_MOE_CHUNK)
};

// The prompt of a prefill command, rank 0 -> rank 1 in messages of kTokensPerMessage tokens (rank 1: received).
void send_prompt(BatchForward& fwd, std::span<const std::uint32_t> prompt, std::vector<std::uint32_t>* received) {
    std::vector<std::byte> o(std::size_t{kTokensPerMessage} * 4U), i(o.size());
    const std::size_t n = received ? received->size() : prompt.size();
    for (std::size_t at = 0; at < n; at += kTokensPerMessage) {
        const std::size_t take = std::min<std::size_t>(kTokensPerMessage, n - at);
        std::fill(o.begin(), o.end(), std::byte{0});
        if (!received) std::memcpy(o.data(), prompt.data() + at, take * 4U);
        fwd.exchange_host(o, i);
        if (received) std::memcpy(received->data() + at, i.data(), take * 4U);
    }
}

std::atomic<bool>& stop_requested() {
    static std::atomic<bool> s{false};
    return s;
}
extern "C" void on_signal(int) { stop_requested().store(true); }

// An imported state's bytes, rank 0 -> rank 1 in messages of kImportChunk bytes (rank 1 sends zeros; `in` filled).
void transfer_state(BatchForward& fwd, std::span<const std::uint8_t> out, std::vector<std::uint8_t>* in) {
    const std::size_t n = in ? in->size() : out.size();
    std::vector<std::byte> zeros, back;
    for (std::size_t at = 0; at < n; at += kImportChunk) {
        const std::size_t take = std::min(kImportChunk, n - at);
        back.resize(take);
        if (in) {
            zeros.assign(take, std::byte{0});
            fwd.exchange_host(zeros, back);
            std::memcpy(in->data() + at, back.data(), take);
        } else {
            fwd.exchange_host(std::as_bytes(out.subspan(at, take)), back);
        }
    }
}

// The trellis plans (site, buffers, row count; o_a group plans; head rows) are created by the first step that submits
// them and every creation synchronises: 2,885 per rank, ~0.4 s, in the first requests' steps and at the first step of
// each new row count; the device prefill's first use costs ~0.9 s more in the first prompt. The warm-up (both ranks,
// before READY) runs one step per row count 1..row_limit() (dummy tokens; each dummy slot first takes its indexer's lead
// rows one per step; segments of at most max_rows on slots 0, 1, ...) and one dummy device prefill, then resets the
// slots (with a pool their room goes back): no step after READY creates a plan (logged at STOPPED and in the loop
// profile). Rows never depend on other rows or on earlier steps of other slots (BatchForward's contract): no output
// changes (qualification: ids, prefill logits and prefix hits equal with/without warm-up). Both ranks must agree (they exchange
// the setting first). Not warmed: the prefill tile kernels (runs of >= 2048 rows), the non-direct indexer and the
// DSpark drafter's first use.
// STRIX_DS41_SERVE_WARM: default on (=0: off). With it the DSpark speculator's cost calibration is on by default too
// (BatchSpecOptions::calib; STRIX_DS41_DSPARK_CALIB overrides): the warm-up alone gives the controller true prices
// that the old estimator mishandles (measured: C8 after C1-C4 TTFT +15 %, prefix-hit -13 % tok/s), the calibration alone
// loses cold C8 TTFT (+26 %); together, never slower in qualification (measured cohort:
// first request TTFT -19 %, prefix hit +3.7 % tok/s, held-out C1 +4.8 %). Off: the previous behaviour.
bool serve_warm_on() {
    const char* env = std::getenv("STRIX_DS41_SERVE_WARM");
    return env == nullptr || std::strcmp(env, "0") != 0;
}
constexpr std::uint32_t kWarmPrefillRows = 64;
strix::ds41::TrellisPlanStats plans_at_ready;

double rss_mib() {  // this process's resident host memory (VmRSS)
    std::ifstream in("/proc/self/status");
    for (std::string line; std::getline(in, line);)
        if (line.rfind("VmRSS:", 0) == 0) return static_cast<double>(std::stoull(line.substr(6))) / 1024.0;
    return 0.0;
}

void warm_up(const Args& a, BatchForward& fwd, SlotPrefill* prefill, const strix::ds41::hetero::Stage* stage, Watchdog& dog, std::uint32_t rank) {
    const bool on = serve_warm_on();
    // A multi-row segment must start at or after its indexer's first compressed entry (position ratio - 1 of the KV
    // source): each dummy slot first takes `lead` rows from position 0 one per step, then the measured segment.
    const auto& layers = fwd.model().layers;
    std::uint32_t lead = 0;
    for (const auto& L : layers)
        if (L.index_source && L.kv_source_layer >= 0) lead = std::max(lead, layers.at(static_cast<std::size_t>(L.kv_source_layer)).compress_ratio - 1U);
    const std::uint32_t seg = a.context > lead ? std::min(BatchForward::max_rows, a.context - lead) : 0U;  // rows of one dummy segment
    // split decode (--hetero-listen): the steps go through the FRONT (Stage::step), so the warm-up is skipped there
    std::uint32_t rows = on && stage == nullptr ? std::min(fwd.row_limit(), seg * a.slots) : 0U;
    // the dummy slots must fit the KV pool (a small --pool-positions can hold real requests but not lead + seg per slot):
    // the default warm-up is skipped then (same decision on both ranks: same arguments); STRIX_DS41_SERVE_WARM=1 throws below
    if (rows != 0U && std::getenv("STRIX_DS41_SERVE_WARM") == nullptr) {
        const std::uint32_t need = (rows + seg - 1U) / seg;
        bool fits = true;
        std::uint32_t held = 0;
        for (; held < need && fits; ++held) fits = fwd.reserve(held, lead + seg);
        for (std::uint32_t slot = 0; slot < held; ++slot) {
            fwd.release(slot);
            fwd.reset(slot);
        }
        if (!fits) {
            std::fprintf(stderr, "rank %u: warm-up skipped: the KV pool has no room for %u dummy slots of %u positions\n", rank, need, lead + seg);
            rows = 0U;
        }
    }
    // the device prefill's row split needs more than 8 rows (kMinRows); a smaller limit leaves it cold
    const std::uint32_t prefill_rows = std::min({kWarmPrefillRows, a.prefill_rows, a.context});
    const bool warm_prefill = prefill != nullptr && prefill_rows >= kMinRows;
    std::array<std::byte, 8> mine{}, peer{};
    std::memcpy(mine.data(), &rows, 4);
    fwd.exchange_host(mine, peer);
    dog.progress();
    if (std::memcmp(mine.data(), peer.data(), mine.size()) != 0) poisoned(rank, "STRIX_DS41_SERVE_WARM or the step row limit differs between the ranks");
    if (rows != 0U) {
        const auto t0 = Clock::now();
        const auto before = strix::ds41::trellis_plan_stats();
        const auto free_before = fwd.device_memory().first;
        const double rss_before = rss_mib();
        std::vector<BatchRow> step;
        std::uint32_t used = 0;
        for (std::uint32_t r = 1; r <= rows; ++r) {
            step.clear();
            for (std::uint32_t slot = 0; step.size() < r; ++slot) {
                const std::uint32_t n = std::min<std::uint32_t>(seg, r - static_cast<std::uint32_t>(step.size()));
                if (!fwd.reserve(slot, lead + seg)) throw std::runtime_error("serve warm-up: no KV room for slot " + std::to_string(slot));
                for (std::uint32_t k = 0; k < lead; ++k) {  // positions [0, lead) one row per step, as decode (never a multi-row segment)
                    const BatchRow one{slot, 1000U + k};
                    (void)fwd.step(std::span(&one, 1));
                    dog.progress();
                }
                for (std::uint32_t k = 0; k < n; ++k) step.push_back(BatchRow{slot, 1000U + lead + k});
                used = std::max(used, slot + 1U);
            }
            (void)fwd.step(step);
            dog.progress();
        }
        for (std::uint32_t slot = 0; slot < used; ++slot) {
            fwd.release(slot);
            fwd.reset(slot);
        }
        const auto steps = strix::ds41::trellis_plan_stats();
        if (warm_prefill) {
            prefill->warm(prefill_rows);
            dog.progress();
        }
        const auto after = strix::ds41::trellis_plan_stats();
        std::fprintf(stderr, "rank %u: warm-up before READY: steps R1..R%u + device prefill %u rows in %.2f s; trellis plans %llu created in %.2f s (steps %llu), "
                     "device free %.1f MiB lower, host RSS %+.1f MiB\n", rank, rows, warm_prefill ? prefill_rows : 0U,
                     std::chrono::duration<double>(Clock::now() - t0).count(), static_cast<unsigned long long>(after.created - before.created),
                     after.seconds - before.seconds, static_cast<unsigned long long>(steps.created - before.created),
                     (static_cast<double>(free_before) - static_cast<double>(fwd.device_memory().first)) / 1048576.0, rss_mib() - rss_before);
    }
    plans_at_ready = strix::ds41::trellis_plan_stats();
}

// Plans created after READY (misses of the warm-up; all of them when it is off).
std::string plans_since_ready() {
    const auto now = strix::ds41::trellis_plan_stats();
    char b[96];
    std::snprintf(b, sizeof b, "trellis plans after READY %llu (%.2f s)", static_cast<unsigned long long>(now.created - plans_at_ready.created),
                  now.seconds - plans_at_ready.seconds);
    return b;
}

using Sessions = strix::deepseek41::PrefixCache::Sessions;
using SessionCheckpoint = strix::deepseek41::SessionCheckpoint;
using SessionBytes = strix::deepseek41::SessionBytes;
strix::deepseek41::SessionExchange session_exchange(BatchForward &fwd) {
    return [&fwd](std::span<const std::uint8_t> b) {
        return strix::deepseek41::session_peer_exchange(b, [&](auto out, auto in) { fwd.exchange_host(out, in); });
    };
}
std::unique_ptr<Sessions> make_sessions(const Args &a, std::uint32_t rank) {
    if (!a.session_ram_bytes)
        return {};
    return strix::deepseek41::admit_session_startup([&] {
        std::map<std::string, std::string> cfg{{"world", "2"},
                                            {"rank", std::to_string(rank)},
                                            {"engine", "tp2-http-target"},
                                            {"layers", std::to_string(a.layers)},
                                            {"context", std::to_string(a.context)},
                                            {"slots", std::to_string(a.slots)},
                                            {"pool", std::to_string(a.pool)},
                                            {"prompt-rows", std::to_string(a.prompt_rows)},
                                            {"tp-attention", std::to_string(a.tp)},
                                            {"engram-direct", std::to_string(a.engram_direct)},
                                            {"prefill-device", std::to_string(a.device_prefill)},
                                            {"prefill-rows", std::to_string(a.prefill_rows)},
                                            {"prefill-rows-auto", std::to_string(a.prefill_rows_auto)},
                                            {"prefill-min", std::to_string(a.prefill_min)},
                                            {"prefix-block", std::to_string(a.prefix_block)},
                                            {"prefix-inrun", std::to_string(a.prefix_inrun)},
                                            {"prefix-before-tail", std::to_string(a.prefix_before_tail)},
                                            {"prefix-session", std::to_string(a.prefix_session)},
                                            {"prefix-inrun-reach", std::to_string(a.prefix_inrun_reach)},
                                            {"dspark", "off"},
                                            {"sampling", "greedy"},
                                            {"layout", "fp8-ring136x528-kv288-idx68-fp32carry-v1"}};
        const auto snapshot = strix::LoaderCapture::snapshot();
        auto compat = strix::deepseek41::session_compat(std::move(cfg), {{"model", a.model}, {"draft", a.dspark}}, "/proc/self/exe", &snapshot);
        return std::make_unique<Sessions>(std::move(compat), rank, a.session_ram_bytes, a.session_dir, a.session_max_parked, a.session_disk_bytes);
    }, [](const char* reason) { std::fprintf(stderr, "session cache: off (%s)\n", reason); });
}
// A startup refusal on either rank must disable the API on both ranks.
void agree_session_startup(Args& a, BatchForward& fwd, std::unique_ptr<Sessions>& sessions) {
    if (!a.session_ram_bytes) return;
    if (strix::deepseek41::session_startup_ready(bool(sessions), a.session_explicit,
        [&](auto out, auto in) { fwd.exchange_host(out, in); })) return;
    if (sessions) std::fprintf(stderr, "session cache: off (session setup failed on peer)\n");
    sessions.reset();
    a.session_ram_bytes = 0;
    a.session_dir.clear();
}
// Every command runs only between committed target steps. Restore writes an unused slot,
// then the import barrier makes it eligible for rank 0's scheduler.
std::optional<SessionCheckpoint> session_operation(const Args &a, BatchForward &fwd, Sessions &cache, const serve::Json &message, std::uint32_t slot,
                                                   std::uint32_t rank) {
    (void)a;
    const auto op = message.find("op")->as_string();
    const auto id = message.find("id")->as_string();
    const auto ex = session_exchange(fwd);
    if (op == "capture") {
        SessionCheckpoint c;
        c.id = id;
        c.generation = message.find("generation")->as_string();
        c.rank = rank;
        c.compat_record = cache.compat().record();
        bool ok = true;
        try {
            c.attention = fwd.export_state(slot);
        } catch (...) {
            ok = false;
        }
        Sessions::agree(ok, ex, "device export");
        c.attention.next_token = static_cast<std::uint32_t>(message.find("pending")->as_int());
        for (const auto &t : message.find("history")->as_array())
            c.attention.tokens.push_back(static_cast<std::uint32_t>(t.as_int()));
        c.frontier.consumed = c.frontier.accepted = c.attention.position;
        c.frontier.pending = c.attention.next_token;
        c.frontier.emitted = static_cast<std::uint64_t>(message.find("emitted")->as_int());
        cache.capture(c, ex);
    } else if (op == "checkpoint")
        cache.checkpoint(id, ex);
    else if (op == "delete")
        cache.erase(id, ex);
    else if (op == "resume" || op == "restore" || op == "lookup" || op == "lookup_disk") {
        const bool disk = op == "restore" || op == "lookup_disk";
        auto c = cache.resume(id, disk, ex);
        if (op == "lookup" || op == "lookup_disk") return c;
        if (disk)
            cache.capture(c, ex);
        bool ok = true;
        try {
            fwd.import_state(slot, c.attention);
        } catch (...) {
            ok = false;
        }
        Sessions::agree(ok, ex, "staged device import");
        return c;
    } else if (op == "host_restore") {
        cache.capture(cache.resume(id, true, ex), ex);
    } else if (op == "park" || op == "ram_checkpoint") {
        (void)cache.resume(id, false, ex);
    } else
        throw std::invalid_argument("unknown session operation");
    return {};
}

// ---- rank 1: no API; runs what rank 0 sends ----
void follow(const Args& initial, BatchForward& fwd, Watchdog& dog, SlotPrefill* prefill, strix::ds41::hetero::Stage* stage) {
    Args a = initial;
    const auto mine = hello(a);
    const auto first = exchange(fwd, mine);
    dog.progress();
    if (const auto why = hello_mismatch(mine, first); !why.empty()) poisoned(1U, why);
    agree_session_limits(a, first);
    dog.pause(true);
    auto sessions = make_sessions(a, 1U);
    agree_session_startup(a, fwd, sessions);
    dog.pause(false);
    std::unique_ptr<strix::ds41::dspark::DeviceDrafter> drafter;
    std::unique_ptr<strix::deepseek41::BatchSpeculator> spec;
    std::uint64_t dspark_free_before = 0;  // device free memory before the drafter (0: no drafter on this rank)
    if (!a.dspark.empty() && a.dspark_ranks == 2U) {  // this rank drafts its half of the device-prefilled requests
        const auto w = strix::ds41::dspark::load_weights(a.model, a.dspark);
        const auto free_before = dspark_free_before = fwd.device_memory().first;
        drafter = std::make_unique<strix::ds41::dspark::DeviceDrafter>(w->weights);
        strix::deepseek41::BatchSpecOptions so;
        so.vocabulary = a.dspark_vocab;
        so.rank = 1U;
        so.world = 2U;
        so.split_started = false;
        so.coop = dspark_coop_on();
        so.exchange = [&](std::span<const std::byte> out, std::span<std::byte> in) { fwd.exchange_host(out, in); };
        so.calib = serve_warm_on() && stage == nullptr;  // with the warm-up (split decode skips it)
        spec = std::make_unique<strix::deepseek41::BatchSpeculator>(fwd, *drafter, a.slots, so);
        log_dspark_memory(1U, w->weights, a, free_before, fwd.device_memory().first);
    }
    if (dspark_free_before != 0U) log_dspark_resident(1U, dspark_free_before, fwd.device_memory().first);
    serve::Json ready = serve::Json::object();
    ready.set("event", serve::Json("READY"));
    ready.set("rank", serve::Json(1));
    ready.set("group_id", serve::Json(a.group_id));
    ready.set("epoch", serve::Json(a.epoch));
    if (prefill) prefill->reconcile(fwd, 1U);
    warm_up(a, fwd, prefill, stage, dog, 1U);
    event(ready);
    Control reply;
    std::vector<BatchRow> last_rows;  // the last step's rows and tokens: the prefix bookkeeping waits for its commits
    std::vector<std::uint32_t> last_tokens;
    std::vector<BatchRow> spec_rows;  // the same for the drafter (staged once, at the next round or control block)
    std::uint64_t spec_step = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> round_commits;  // commits that came in draft rounds
    std::unique_ptr<SlotPrefill::Task> task;  // STRIX_DS41_SERVE_SCHED=interleave: the prefill rank 0 runs in units
    const bool log_prefill = std::getenv("STRIX_DS41_SERVE_PREFILL_LOG") != nullptr && std::string(std::getenv("STRIX_DS41_SERVE_PREFILL_LOG")) == "1";
    const auto log_outcome = [&](std::uint32_t slot, std::uint32_t tokens, const PrefillOutcome& o) {
        if (log_prefill)
            std::fprintf(stderr, "rank 1 prefill slot %u: %u tokens, resume %u, %u pieces; token %u logits_fnv %016llx; %.3f s\n", slot, tokens, o.resume,
                         o.pieces, o.next_token, static_cast<unsigned long long>(o.logits), o.seconds);
    };
    for (;;) {
        reply.round += 1U;
        std::array<std::byte, 512> out{}, in{};
        std::memcpy(out.data(), &reply, offsetof(Control, extra_rows));
        fwd.exchange_host(out, in);
        reply.pkey = 0;  // the import digest echo goes out once
        dog.progress();
        std::uint32_t magic = 0;
        std::memcpy(&magic, in.data(), 4U);
        if (spec && magic == strix::deepseek41::BatchSpeculator::round_magic) {  // a draft round of rank 0's next step
            (void)spec->follow_packet(std::span<const std::byte>(in.data(), strix::deepseek41::BatchSpeculator::round_bytes), spec_rows,
                                      last_tokens, spec_step, &round_commits);
            continue;
        }
        Control c;
        std::copy_n(in.begin(), offsetof(Control, extra_rows), std::as_writable_bytes(std::span(&c, 1)).begin());
        if (c.magic != reply.magic) throw std::runtime_error("control channel: bad magic from the peer");
        if (std::max(reply.rows, c.rows) > BatchForward::max_rows) {  // as exchange(): the rows past 8 follow
            if (std::max(reply.rows, c.rows) > fwd.row_limit()) throw std::runtime_error("control channel: step row limit differs");
            fwd.exchange_host(std::as_bytes(std::span(reply.extra_rows)), std::as_writable_bytes(std::span(c.extra_rows)));
        }
        if (c.commits > std::size(c.commit)) throw std::runtime_error("control channel: too many commits");
        std::array<std::uint32_t, 64> kept{};
        kept.fill(0xFFFFFFFFU);
        for (std::uint32_t i = 0; i < c.commits; ++i) {
            if (spec) spec->follow_commit(c.commit[i] >> 16U, c.commit[i] & 0xFFFFU);
            else fwd.commit(c.commit[i] >> 16U, c.commit[i] & 0xFFFFU);
            kept.at(c.commit[i] >> 16U) = c.commit[i] & 0xFFFFU;
        }
        for (const auto& [slot, k] : round_commits) kept.at(slot) = k;
        round_commits.clear();
        if (spec) spec->follow_stage(spec_rows, last_tokens, spec_step);  // before anything moves the slots
        if (prefill && !last_rows.empty()) {
            std::vector<BatchRow> rows;
            std::vector<std::uint32_t> toks;
            std::array<std::uint32_t, 64> seen{};
            for (std::size_t r = 0; r < last_rows.size(); ++r)
                if (seen.at(last_rows[r].slot)++ < kept[last_rows[r].slot]) rows.push_back(last_rows[r]), toks.push_back(last_tokens[r]);
            (void)prefill->on_step(rows, toks);
        }
        last_rows.clear();
        if (c.snaps != 0U) {  // rank 0's session snapshots of the last step, before anything changes the slots
            if (prefill == nullptr) throw std::runtime_error("control channel: snapshots without a prefix cache");
            prefill->snapshot(c.snaps);
        }
        if (c.command == static_cast<std::uint32_t>(Command::shutdown)) return;
        if (c.command == static_cast<std::uint32_t>(Command::heartbeat)) continue;
        if (c.command == static_cast<std::uint32_t>(Command::quiesce)) {  // rank 0 went idle: no GPU receive left posted
            fwd.quiesce();
            continue;
        }
        if (c.command == static_cast<std::uint32_t>(Command::session)) {
            if (!sessions || c.pslot >= a.slots || c.prefix_bytes == 0 || c.prefix_bytes > (16U << 20U) || c.ops > std::size(c.op))
                throw std::runtime_error("invalid session control command");
            for (std::uint32_t i = 0; i < c.ops; ++i) {
                const auto kind = c.op[i] >> 30U, slot = (c.op[i] >> 24U) & 63U, positions = c.op[i] & 0xFFFFFFU;
                if (kind == kPoolRelease)
                    fwd.release(slot);
                else if (kind != kPoolReserve || !fwd.reserve(slot, positions))
                    throw std::runtime_error("session pool operation failed");
            }
            std::vector<std::uint8_t> bytes(c.prefix_bytes);
            transfer_state(fwd, {}, &bytes);
            const auto message = serve::parse_json(std::string(bytes.begin(), bytes.end()));
            dog.pause(true);
            try {
                const auto restored = session_operation(a, fwd, *sessions, message, c.pslot, 1U);
                if (restored && prefill)
                    prefill->on_prefill(c.pslot, restored->attention.tokens);
                reply.pkey = 1;
            } catch (const std::exception &e) {
                std::fprintf(stderr, "session rank 1: %s\n", e.what());
                reply.pkey = 0;
            }
            dog.pause(false);
            dog.progress();
            continue;
        }
        if (c.command == static_cast<std::uint32_t>(Command::import_state)) {  // POST /admin/import-state on rank 0
            if (c.ops > std::size(c.op) || c.pslot >= a.slots || c.prefix_bytes == 0U || c.prefix_bytes > (std::uint64_t{1} << 32U))
                throw std::runtime_error("control channel: bad import command");
            for (std::uint32_t i = 0; i < c.ops; ++i) {
                const std::uint32_t kind = c.op[i] >> 30U, slot = (c.op[i] >> 24U) & 63U, positions = c.op[i] & 0xFFFFFFU;
                if (kind == kPoolRelease) fwd.release(slot);
                else if (kind != kPoolReserve || !fwd.reserve(slot, positions)) throw std::runtime_error("control channel: pool operation failed on rank 1");
            }
            std::vector<std::uint8_t> bytes(c.prefix_bytes);
            transfer_state(fwd, {}, &bytes);
            dog.progress();
            const auto st = strix::deepseek41::deserialize_attention_state(bytes);
            const auto digest = st.digest();
            if (digest != c.pkey) poisoned(1U, "imported state: the digest differs from rank 0's");
            if (a.pool == 0U) fwd.reset(c.pslot);  // with a pool the reserve above reset it
            fwd.import_state(c.pslot, st);
            if (spec) spec->forget(c.pslot);  // its drafting contexts no longer in step on both ranks
            if (prefill) prefill->on_prefill(c.pslot, st.tokens);  // the slot's token history (session snapshots)
            reply.pkey = digest;  // rank 0's acknowledgement round
            continue;
        }
        if (c.command == static_cast<std::uint32_t>(Command::prefill_unit)) {  // the next unit of the task (or its abort)
            if (!task) throw std::runtime_error("control channel: prefill unit without a prefill task");
            if ((c.pflags & kPrefillAbort) != 0U) {
                dog.pause(true);
                (void)strix::deepseek41::sched::apply_unit(*task, 0U, 0U, true);
                dog.pause(false);
                task.reset();
                continue;
            }
            dog.pause(true);
            const bool done = strix::deepseek41::sched::apply_unit(*task, c.pend, c.pbudget, false) == strix::deepseek41::sched::UnitResult::complete;
            dog.pause(false);
            check_unit(fwd, 1U, {done, task->rows(), task->unit_position()});
            if (done) {
                const auto o = task->finish();
                prefill->on_prefill(task->slot(), task->prompt());
                if (spec) spec->imported(task->slot(), task->prompt(), task->total(), o.next_token, o.seed.streams, o.seed.rows, fwd.stream());
                log_outcome(task->slot(), task->total(), o);
                task.reset();
            }
            continue;
        }
        const bool is_prefill = c.command == static_cast<std::uint32_t>(Command::prefill);
        if (is_prefill ? (prefill == nullptr || c.plen == 0U || c.plen >= a.context || c.pslot >= a.slots)
                       : (c.command != static_cast<std::uint32_t>(Command::step) || c.rows == 0U || c.rows > fwd.row_limit()))
            throw std::runtime_error("control channel: unexpected command");
        if (c.ops > std::size(c.op)) throw std::runtime_error("control channel: too many pool operations");
        for (std::uint32_t i = 0; i < c.ops; ++i) {  // the pool operations of rank 0, in its order
            const std::uint32_t kind = c.op[i] >> 30U, slot = (c.op[i] >> 24U) & 63U, positions = c.op[i] & 0xFFFFFFU;
            if (kind == kPoolRelease) fwd.release(slot);
            else if (kind != kPoolReserve || !fwd.reserve(slot, positions)) throw std::runtime_error("control channel: pool operation failed on rank 1");
        }
        for (std::uint32_t s = 0; s < a.slots; ++s)
            if ((c.resets >> s) & 1U) {
                fwd.reset(s);
                if (prefill) prefill->on_reset(s);
                if (spec) spec->release(s);
            }
        if (is_prefill) {  // the prompt, then the same prefill as rank 0 (the cache must give rank 0's hit)
            std::vector<std::uint32_t> prompt(c.plen);
            send_prompt(fwd, {}, &prompt);
            auto hit = prefill->lookup(prompt);
            if (hit.resume != c.presume || hit.shared != c.pshared || hit.key != c.pkey)
                throw std::runtime_error("prefix cache: rank 1 holds other entries than rank 0 (resume " + std::to_string(hit.resume) + " vs " +
                                         std::to_string(c.presume) + ")");
            dog.pause(true);  // a long prompt's prefill is one round
            const bool ok = prefill->prepare(prompt, hit);
            if (hit.disk) {  // all or nothing: both ranks read their files
                Control ack;
                ack.command = static_cast<std::uint32_t>(Command::prefill);
                ack.round = reply.round;
                ack.pflags = ok ? 0U : 2U;
                const auto peer = exchange(fwd, ack);
                if (!ok || (peer.pflags & 2U) != 0U) {
                    prefill->forget(prompt, hit);
                    hit = strix::deepseek41::PrefixLookup{};
                }
            }
            const bool capture = prefill->has_cache() && hit.resume < c.plen && c.plen >= a.prefix_block;
            const bool insert = capture && prefill->admit(c.plen, (c.pflags & 1U) != 0U);
            if ((c.pflags & kPrefillTask) != 0U) {  // interleave: units follow (prefill_unit)
                if (task) throw std::runtime_error("control channel: a prefill task is already running");
                task = prefill->begin(c.pslot, prompt, hit, insert, (c.pflags >> kPrefillHintShift) * a.prefix_block);
                if (task->done()) {  // the whole prompt was cached: no unit (rank 0 does the same)
                    const auto o = task->finish();
                    prefill->on_prefill(c.pslot, prompt);
                    if (spec) spec->imported(c.pslot, prompt, c.plen, o.next_token, o.seed.streams, o.seed.rows, fwd.stream());
                    log_outcome(c.pslot, c.plen, o);
                    task.reset();
                }
                dog.pause(false);
                continue;
            }
            const auto o = prefill->run(c.pslot, prompt, hit, insert, (c.pflags >> kPrefillHintShift) * a.prefix_block);
            prefill->on_prefill(c.pslot, prompt);
            if (spec) spec->imported(c.pslot, prompt, c.plen, o.next_token, o.seed.streams, o.seed.rows, fwd.stream());
            log_outcome(c.pslot, c.plen, o);
            dog.pause(false);
            continue;
        }
        std::array<BatchRow, BatchForward::extended_max_rows> step_rows;
        std::copy_n(c.row, std::min(c.rows, BatchForward::max_rows), step_rows.begin());
        if (c.rows > BatchForward::max_rows) std::copy_n(c.extra_rows, c.rows - BatchForward::max_rows, step_rows.begin() + BatchForward::max_rows);
        const auto rows = std::span<const BatchRow>(step_rows.data(), c.rows);
        if (spec) spec->follow_lookup_windows(c.pend);  // for follow_stage of these rows (after this step)
        const auto& tokens = stage ? stage->step(rows) : fwd.step(rows);
        last_rows.assign(rows.begin(), rows.end());
        spec_rows = last_rows;
        last_tokens = tokens;
        ++spec_step;
        dog.progress();
    }
}

// ---- rank 0: HTTP + the scheduler ----
struct Running {
    serve::TakenJob job;
    BatchSequence seq;
    std::size_t published = 0;
    Clock::time_point start, first_token;
    bool has_first = false;
    std::uint32_t prefill_overtakes = 0;
    bool device_prefill = false;  // the prompt goes through the device prefill before the next step
    // every request: prompt tokens resumed from the prefix cache / continued from an imported state (disjoint)
    strix::deepseek41::serve_observe::PromptAccount account;
    // observed requests (submitted after the instrumentation armed): DSpark drafts proposed / accepted
    std::uint64_t proposed = 0, accepted = 0;
    std::uint64_t session_emitted = 0;
};

struct ServeState {
    std::mutex mutex;
    std::uint64_t steps = 0, rows = 0, decode_rows = 0, prompt_rows = 0, admitted = 0;
    double step_seconds = 0.0;
    std::uint64_t prefills = 0, prefill_tokens = 0, cached_tokens = 0;
    double prefill_seconds = 0.0;
    std::uint64_t prefill_cancellations = 0;
    double scheduler_max_unit_ms = 0.0;
    strix::deepseek41::PrefixCacheStats cache;
    std::uint32_t busy = 0;
    std::atomic<bool> draining{false}, failed{false};
    // copies the engine thread publishes in the per-step block (the HTTP threads read only these): the communicator's
    // counters at every step while armed, else at the first step after a status or metrics read asked for them
    // (stats_wanted, which also refreshes the KV pool's largest free reservation); the speculator's while armed.
    // The reads ask at most once a second (wanted_at, by the clock read under the step lock, so the asks are a second
    // apart in lock order whatever the readers' delays) and pending asks coalesce: however often the status is polled,
    // the engine thread does, over any interval of T seconds, at most T + 2 communicator copies (88 bytes) and free-list
    // walks (one map node per free range, at most slots + 1; none without a KV pool) for it. Two of them can run on
    // consecutive steps (an ask left pending by a slow step, then the next one). The reads still take the step lock for
    // their copy, as before (and now one clock read under it).
    strix::dist::CommunicatorStats comm;
    std::atomic<bool> stats_wanted{true};
    Clock::time_point wanted_at{};
    std::uint64_t pool_free = 0;
    strix::deepseek41::BatchSpecStats spec;
    bool spec_published = false;
    // what a status or metrics read copies under the mutex (it formats after unlocking)
    struct Copy {
        std::uint32_t busy = 0;
        std::uint64_t steps = 0, rows = 0, decode_rows = 0, prompt_rows = 0;
        double step_seconds = 0.0;
        std::uint64_t prefills = 0, prefill_tokens = 0, cached_tokens = 0;
        double prefill_seconds = 0.0;
        std::uint64_t prefill_cancellations = 0;
        double scheduler_max_unit_ms = 0.0;
        strix::deepseek41::PrefixCacheStats cache;
        strix::dist::CommunicatorStats comm;
        std::uint64_t pool_free = 0;
        bool spec_published = false;
    };
    // under mutex: the counters and the communicator's copy (and asks the next step for a fresh one when the last ask is
    // a second old, by the clock read here under the lock, so asks are a second apart in lock order); the speculator's
    // copy only while armed
    Copy copy_locked(bool armed, strix::deepseek41::BatchSpecStats& spec_out) {
        if (const auto now = Clock::now(); now - wanted_at >= std::chrono::seconds(1)) {
            wanted_at = now;
            stats_wanted.store(true, std::memory_order_relaxed);
        }
        if (armed) spec_out = spec;
        return Copy{busy, steps, rows, decode_rows, prompt_rows, step_seconds, prefills, prefill_tokens, cached_tokens,
                    prefill_seconds, prefill_cancellations, scheduler_max_unit_ms, cache, comm, pool_free, armed && spec_published};
    }
};

// POST /admin/import-state: the HTTP thread checks and parses the body, the scheduler imports it between steps.
struct ImportResult {
    int status = 200;
    std::string error;
    std::uint32_t slot = 0;
    double queue_seconds = 0.0, replicate_seconds = 0.0, device_seconds = 0.0;
};
struct ImportJob {
    strix::deepseek41::AttentionState state;
    std::vector<std::uint8_t> bytes;
    std::string sha256;
    std::uint64_t digest = 0;
    std::uint32_t reserve_tokens = 0;
    Clock::time_point queued;
    std::promise<ImportResult> done;
};
struct ImportBox {
    std::mutex mutex;
    std::deque<std::shared_ptr<ImportJob>> jobs;
};

struct SessionJob {
    serve::Json message;
    std::promise<std::pair<int, std::string>> done;
};
struct SessionBox {
    std::mutex mutex;
    std::deque<std::shared_ptr<SessionJob>> jobs;
};
void handle_session(const Args &a, const serve::ApiConfig &config, SessionBox &box, const serve::HttpRequest &request, serve::HttpResponseWriter &response) {
    const auto fail = [&](int status, const std::string &text) {
        auto out = serve::Json::object();
        out = serve::session_error(status == 401 ? "authentication_error" : "invalid_request", text);
        response.send(status, "application/json", out.dump());
    };
    if (!a.session_ram_bytes)
        return fail(404, "session cache is disabled");
    if (!config.api_key.empty() && request.header("authorization") != "Bearer " + config.api_key && request.header("x-api-key") != config.api_key)
        return fail(401, "invalid or missing API key");
    try {
        auto j = std::make_shared<SessionJob>();
        j->message = serve::session_command(request.method, request.path, request.body);
        auto future = j->done.get_future();
        {
            const std::lock_guard lock(box.mutex);
            if (box.jobs.size() >= a.max_queue)
                return fail(429, "session queue full");
            box.jobs.push_back(j);
        }
        if (future.wait_for(std::chrono::seconds(300)) != std::future_status::ready)
            return fail(503, "session operation timed out; query status before retrying");
        const auto result = future.get();
        response.send(result.first, "application/json", result.second);
    } catch (const serve::SessionApiError &e) {
        response.send(e.status, "application/json", serve::session_error(e.code, e.what()).dump());
    } catch (const std::exception &e) {
        fail(400, e.what());
    }
}

// An imported state parked in a slot until a request whose prompt starts with its tokens takes it.
struct Parked {
    std::string session_id;
    std::uint64_t session_emitted = 0;
    std::uint32_t slot = 0, position = 0, next_token = 0;
    std::vector<std::uint32_t> tokens;
    std::string sha256;
};

std::string hex16(std::uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
    return b;
}

// The route itself (rank 0's HTTP thread): checks, SHA-256, parse, then waits for the scheduler's import.
void handle_import(const Args& a, const serve::ApiConfig& config, ServeState& state, ImportBox& box, const serve::HttpRequest& request,
                   serve::HttpResponseWriter& response) {
    const auto t0 = Clock::now();
    const auto fail = [&](int status, const std::string& message) {
        serve::Json e = serve::Json::object();
        serve::Json inner = serve::Json::object();
        inner.set("message", serve::Json(message));
        inner.set("type", serve::Json(status == 401 ? "authentication_error" : "invalid_request_error"));
        e.set("error", std::move(inner));
        e.set("imported", serve::Json(false));
        response.send(status, "application/json", e.dump());
    };
    if (request.method != "POST") return fail(405, "POST the state file's bytes");
    if (!config.api_key.empty() && request.header("authorization") != "Bearer " + config.api_key && request.header("x-api-key") != config.api_key)
        return fail(401, "invalid or missing API key");
    if (state.draining.load() || state.failed.load()) return fail(503, "the server is draining or failed");
    if (const auto f = request.header("x-state-format"); !f.empty() && f != "strix/ds41-attention-state-v1")
        return fail(415, "X-State-Format must be strix/ds41-attention-state-v1");
    if (request.body.empty()) return fail(400, "empty body: send the state file's bytes");
    auto job = std::make_shared<ImportJob>();
    job->sha256 = strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(request.body.data(), request.body.size()))));
    auto want = request.header("x-state-sha256");
    std::transform(want.begin(), want.end(), want.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (!want.empty() && want != job->sha256) return fail(400, "X-State-Sha256 does not match the body (" + job->sha256 + ")");
    const double sha_s = std::chrono::duration<double>(Clock::now() - t0).count();
    const auto t1 = Clock::now();
    try {
        job->bytes.assign(request.body.begin(), request.body.end());
        job->state = strix::deepseek41::deserialize_attention_state(job->bytes);
    } catch (const std::exception& e) {
        return fail(400, std::string("not a valid state: ") + e.what());
    }
    const auto& st = job->state;
    if (st.layers.size() != a.layers) return fail(400, "the state has " + std::to_string(st.layers.size()) + " layers, the server " + std::to_string(a.layers));
    for (std::uint32_t l = 0; l < a.layers; ++l)
        if (!strix::deepseek41::attention_layer_present(st.layers[l])) return fail(400, "layer " + std::to_string(l) + " is missing from the state");
    if (st.position == 0U || st.tokens.size() != st.position) return fail(400, "the state's token history does not match its position");
    if (st.position + 1U >= a.context) return fail(413, "the state fills the context (" + std::to_string(a.context) + ")");
    job->reserve_tokens = a.default_max_tokens;
    if (const auto r = request.header("x-state-reserve-tokens"); !r.empty()) {
        try {
            job->reserve_tokens = static_cast<std::uint32_t>(std::stoul(r));
        } catch (const std::exception&) {
            return fail(400, "X-State-Reserve-Tokens must be a number");
        }
    }
    job->digest = st.digest();
    const double parse_s = std::chrono::duration<double>(Clock::now() - t1).count();
    job->queued = Clock::now();
    auto future = job->done.get_future();
    {
        const std::lock_guard lock(box.mutex);
        box.jobs.push_back(job);
    }
    if (future.wait_for(std::chrono::seconds(300)) != std::future_status::ready) return fail(503, "the import did not run within 300 s");
    const auto res = future.get();
    if (res.status != 200) return fail(res.status, res.error);
    serve::Json out = serve::Json::object();
    out.set("imported", serve::Json(true));
    out.set("position", serve::Json(static_cast<std::uint64_t>(st.position)));
    out.set("sha256", serve::Json(job->sha256));
    out.set("next_token", serve::Json(static_cast<std::uint64_t>(st.next_token)));
    out.set("digest", serve::Json(hex16(job->digest)));
    out.set("bytes", serve::Json(static_cast<std::uint64_t>(job->bytes.size())));
    out.set("slot", serve::Json(static_cast<std::uint64_t>(res.slot)));
    serve::Json timing = serve::Json::object();
    timing.set("sha256_s", serve::Json(sha_s));
    timing.set("parse_s", serve::Json(parse_s));
    timing.set("queue_s", serve::Json(res.queue_seconds));
    timing.set("replicate_s", serve::Json(res.replicate_seconds));
    timing.set("device_s", serve::Json(res.device_seconds));
    timing.set("server_s", serve::Json(std::chrono::duration<double>(Clock::now() - t0).count()));
    out.set("timing", std::move(timing));
    response.send(200, "application/json", out.dump());
}


// --import-frames PORT (streaming protocol): the PC streams the state file as
// strix::handoff pieces while its prefill runs: piece kind 1 = layer l's record (as serialize_attention_state writes it;
// sent when the prefill finished layer l), kind 2 = head | tail (next_token, tokens, digest, checksum; sent last). HELLO
// model "deepseek-v4.1/attn-state-v1", meta = u32 layers | u32 position | u32 reserve_tokens (0: the default). The bytes
// head | records | tail are the very state file (deepseek41_state_pieces.hpp); they then take the POST
// /admin/import-state path unchanged (deserialize with its checksum and digest checks, the scheduler's import between
// steps, the bytes to rank 1 behind the import command, rank 1's digest acknowledgement). The reply carries the same
// JSON as the route; any failure aborts the session (the sender's finish() throws with the reason).
void frames_import_session(const Args& a, ServeState& state, ImportBox& box, strix::handoff::Session& s,
                           strix::deepseek41::pieces::FramesImportListener& lis) {
    const auto t0 = Clock::now();
    if (state.draining.load() || state.failed.load()) throw std::runtime_error("the server is draining or failed");
    auto rx = strix::deepseek41::pieces::receive_state_file(s, a.layers);  // the PC's state file, rebuilt
    const double t_first = rx.first_piece_s, t_recv = rx.received_s;
    const std::uint32_t meta[3] = {rx.meta.layers, rx.meta.position, rx.meta.reserve_tokens};
    auto job = std::make_shared<ImportJob>();
    job->bytes = std::move(rx.bytes);
    job->state = strix::deepseek41::deserialize_attention_state(job->bytes);  // checksum + digest: the PC's exact file
    const auto& st = job->state;
    if (st.layers.size() != a.layers) throw std::runtime_error("layer count differs");
    for (std::uint32_t l = 0; l < a.layers; ++l)
        if (!strix::deepseek41::attention_layer_present(st.layers[l])) throw std::runtime_error("layer " + std::to_string(l) + " is missing from the state");
    if (st.position == 0U || st.tokens.size() != st.position || st.position != meta[1]) throw std::runtime_error("the state's position / tokens do not match");
    if (st.position + 1U >= a.context) throw std::runtime_error("the state fills the context (" + std::to_string(a.context) + ")");
    job->reserve_tokens = meta[2] != 0U ? meta[2] : a.default_max_tokens;
    job->digest = st.digest();
    job->sha256 = "frames-" + hex16(job->digest);
    const double t_parse = std::chrono::duration<double>(Clock::now() - t0).count() - t_recv;
    job->queued = Clock::now();
    auto future = job->done.get_future();
    // serialized with the listener's shutdown: no import is queued once shutdown began
    if (!lis.publish([&] {
            const std::lock_guard lock(box.mutex);
            box.jobs.push_back(job);
        }))
        throw std::runtime_error("the server is shutting down");
    for (int waited = 0; future.wait_for(std::chrono::seconds(1)) != std::future_status::ready; ++waited)
        if (lis.stopping() || waited >= 300) throw std::runtime_error("the import did not run (server stopping or 300 s)");
    const auto res = future.get();
    if (res.status != 200) throw std::runtime_error("import: " + res.error);
    serve::Json out = serve::Json::object();
    out.set("imported", serve::Json(true));
    out.set("position", serve::Json(static_cast<std::uint64_t>(st.position)));
    out.set("next_token", serve::Json(static_cast<std::uint64_t>(st.next_token)));
    out.set("digest", serve::Json(hex16(job->digest)));
    out.set("digest_without_topk", serve::Json(hex16(st.digest(true))));
    out.set("bytes", serve::Json(static_cast<std::uint64_t>(job->bytes.size())));
    out.set("slot", serve::Json(static_cast<std::uint64_t>(res.slot)));
    serve::Json timing = serve::Json::object();
    const auto fs = s.stats();
    timing.set("first_piece_s", serve::Json(t_first));
    timing.set("received_s", serve::Json(t_recv));
    timing.set("decode_s", serve::Json(fs.decode_seconds));
    timing.set("parse_s", serve::Json(t_parse));
    timing.set("queue_s", serve::Json(res.queue_seconds));
    timing.set("replicate_s", serve::Json(res.replicate_seconds));
    timing.set("device_s", serve::Json(res.device_seconds));
    timing.set("server_s", serve::Json(std::chrono::duration<double>(Clock::now() - t0).count()));
    out.set("timing", std::move(timing));
    const std::string body = out.dump();
    s.reply(0, std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(body.data()), body.size()));
}

void lead(const Args& initial, BatchForward& fwd, strix::dist::Communicator& comm, Watchdog& dog, SlotPrefill* prefill, strix::ds41::hetero::Stage* stage) {
    Args a = initial;
    const auto mine = hello(a);
    const auto peer = exchange(fwd, mine);
    dog.progress();
    if (const auto why = hello_mismatch(mine, peer); !why.empty()) poisoned(0U, why);
    agree_session_limits(a, peer);
    dog.pause(true);
    auto sessions = make_sessions(a, 0U);
    agree_session_startup(a, fwd, sessions);
    dog.pause(false);
    if (prefill) prefill->reconcile(fwd, 0U);
    const auto sched = strix::deepseek41::sched::options_from_env(prefill != nullptr);  // STRIX_DS41_SERVE_SCHED (device: interleave unless =off)
    if (sched.mode != strix::deepseek41::sched::Mode::off) {
        if (!prefill) throw std::invalid_argument("STRIX_DS41_SERVE_SCHED needs --prefill-path device");
        std::fprintf(stderr, "sched: %s (starvation %.1f s, T_dev = %.3f + %.5f P s, unit %u rows, budget %u units / %.0f ms, MoE chunk %u, tasks from %u rows)\n", strix::deepseek41::sched::mode_name(sched.mode),
                     sched.starve_seconds, sched.tdev_a, sched.tdev_b, sched.unit_rows, sched.unit_budget, sched.target_ms(), sched.moe_chunk_rows, sched.min_rows);
    }

    std::fprintf(stderr, "sched: effective %s, service target %.1f ms, budget source %s, short-first %u, nesting disabled\n",
                 strix::deepseek41::sched::mode_name(sched.mode), sched.service_ms, sched.budget_source(), sched.short_first ? 1U : 0U);
    const auto tokenizer = serve::BpeTokenizer::load_gguf(a.model);
    // the drafter is loaded before READY (requests may come as soon as READY is out)
    std::unique_ptr<strix::ds41::dspark::DeviceDrafter> drafter;
    std::unique_ptr<strix::deepseek41::BatchSpeculator> spec;
    std::uint64_t dspark_free_before = 0;
    double hetero_step_seconds = 0.0;  // split decode: the wall of the last batch step
    if (!a.dspark.empty()) {
        const auto w = strix::ds41::dspark::load_weights(a.model, a.dspark);
        const auto free_before = dspark_free_before = fwd.device_memory().first;
        drafter = std::make_unique<strix::ds41::dspark::DeviceDrafter>(w->weights);
        strix::deepseek41::BatchSpecOptions so;
        so.vocabulary = a.dspark_vocab;
        so.prompt_rows = a.prompt_rows;
        so.align = a.prefix_session ? a.prefix_block : 0U;
        if (a.dspark_ranks == 2U) {
            so.world = 2U;
            so.split_started = false;  // rank 1 sees a prompt read in rows only row by row
            so.coop = dspark_coop_on();
            so.exchange = [&](std::span<const std::byte> out, std::span<std::byte> in) { fwd.exchange_host(out, in); };
        }
        if (a.dspark_window.rfind("fixed:", 0) == 0) so.fixed_rows = static_cast<std::uint32_t>(std::stoul(a.dspark_window.substr(6)));
        if (stage) so.step_seconds = [&] { return hetero_step_seconds; };  // the split step's wall (FRONT wait, forward, TAIL)
        so.calib = serve_warm_on() && stage == nullptr;  // with the warm-up (split decode skips it)
        spec = std::make_unique<strix::deepseek41::BatchSpeculator>(fwd, *drafter, a.slots, so);
        log_dspark_memory(0U, w->weights, a, free_before, fwd.device_memory().first);
        std::cerr << "strix serve: DSpark windows in the batch steps (drafter " << a.dspark << ", window " << a.dspark_window << ", ranks "
                  << a.dspark_ranks << ", drafting while <= " << a.dspark_max_inflight << " requests in flight (0: always))" << std::endl;
    }
    warm_up(a, fwd, prefill, stage, dog, 0U);
    serve::ApiConfig config;
    config.model_id = a.model_id;
    config.default_max_tokens = a.default_max_tokens;
    config.default_thinking = false;  // chat mode unless the request enables thinking
    config.sampling_policy = "greedy";
    config.responses = a.http_responses;
    config.capabilities = a.http_capabilities;
    config.stream_receipts = a.http_stream_receipts;
    config.sse_keepalive_seconds = a.http_sse_keepalive;
    config.sse_progress_seconds = a.http_sse_progress_interval;
    config.sessions_enabled = a.session_ram_bytes != 0;
    config.engine_description = "deepseek41 engine group (2 ranks, EP experts" + std::string(a.tp ? " + TP attention" : "") + "), " +
                                std::to_string(a.slots) + " slots, greedy, exact per request";
    config.capabilities_extra = [sched](serve::Json& out) {
        serve::Json policy = serve::Json::object();
        policy.set("effective", serve::Json(strix::deepseek41::sched::mode_name(sched.mode)));
        policy.set("service_target_ms", serve::Json(sched.service_ms));
        policy.set("effective_unit_target_ms", serve::Json(sched.mode == strix::deepseek41::sched::Mode::interleave ? sched.target_ms() : 0.0));
        policy.set("budget_source", serve::Json(sched.budget_source()));
        policy.set("short_first", serve::Json(sched.short_first));
        policy.set("nested_prefill", serve::Json(false));
        policy.set("direct_prefill_below_rows", serve::Json(static_cast<std::uint64_t>(sched.min_rows)));
        policy.set("qualification", serve::Json("pending-both-rank-gates"));
        out.set("scheduler", std::move(policy));
        if (const auto& profile = strix::deepseek41::pair_defaults::profile_json(); !profile.empty())
            out.set("profile", serve::parse_json(profile));
    };
    config.stop_tokens = {tokenizer->eos_token()};
    config.chat_renderer = [](const serve::ChatTemplateInput& in) { return serve::render_deepseek_v41_chat(in); };
    config.output_parser = [](const std::string& t, bool thinking, const serve::Json& tools) { return serve::parse_deepseek_v41_output(t, thinking, tools); };
    config.tool_call_marker = serve::deepseek_v41_tool_marker;
    if (!a.api_key_file.empty()) {
        auto key = file_text(a.api_key_file);
        while (!key.empty() && (key.back() == '\n' || key.back() == '\r' || key.back() == ' ')) key.pop_back();
        if (key.empty()) throw std::invalid_argument("--http-api-key-file is empty");
        config.api_key = key;
    }
    ServeState state;
    state.comm = comm.stats();  // the start-up exchanges, before any HTTP thread reads the copy
    if (a.pool != 0U) state.pool_free = fwd.pool_free_positions();
    serve::GenerationQueue queue(a.max_queue);
    config.on_drain = [&] {
        state.draining.store(true);
        queue.shutdown();  // queued requests end, new ones are refused; running ones finish
    };
    const auto weights = fwd.weight_bytes(), per_slot = fwd.state_bytes_per_slot(), workspace = fwd.workspace_bytes();
    const auto trellis = fwd.trellis_bytes() + fwd.trellis_library_bytes();
    const auto trellis_id = fwd.trellis_identity();
    const std::uint64_t pool_bytes = a.pool != 0U ? fwd.pool_bytes() : 0U;  // the pool's arenas, allocated at start-up
    config.status_extra = [&](serve::Json& out) {
        const auto [free, total] = fwd.device_memory();
        serve::Json mem = serve::Json::object();
        mem.set("weights_bytes", serve::Json(weights));
        mem.set("state_bytes_per_slot", serve::Json(per_slot));
        mem.set("state_bytes", serve::Json(per_slot * a.slots));
        mem.set("workspace_bytes", serve::Json(workspace));
        if (!trellis_id.empty()) {
            mem.set("trellis_bytes", serve::Json(trellis));
            mem.set("trellis_identity", serve::Json(trellis_id));
        }
        mem.set("device_free_bytes", serve::Json(static_cast<std::uint64_t>(free)));
        mem.set("device_total_bytes", serve::Json(static_cast<std::uint64_t>(total)));
        out.set("memory", std::move(mem));
        // copies under the step lock, formatting after it (the engine takes this lock before every publication)
        const bool armed = serve::observe_armed();
        strix::deepseek41::BatchSpecStats ss;
        ServeState::Copy st;
        {
            const std::lock_guard lock(state.mutex);
            st = state.copy_locked(armed, ss);
        }
        const bool spec_published = st.spec_published;
        serve::Json slots = serve::Json::object();
        slots.set("total", serve::Json(static_cast<std::uint64_t>(a.slots)));
        slots.set("busy", serve::Json(static_cast<std::uint64_t>(st.busy)));
        slots.set("free", serve::Json(static_cast<std::uint64_t>(a.slots - st.busy)));
        slots.set("context_tokens", serve::Json(static_cast<std::uint64_t>(a.context)));
        if (a.pool != 0U) {
            slots.set("pool_positions", serve::Json(static_cast<std::uint64_t>(a.pool)));
            slots.set("pool_bytes", serve::Json(pool_bytes));  // fixed at start-up
            slots.set("pool_largest_free", serve::Json(st.pool_free));  // as of the last step that refreshed it
        }
        out.set("slots", std::move(slots));
        serve::Json batch = serve::Json::object();
        batch.set("steps", serve::Json(st.steps));
        batch.set("rows", serve::Json(st.rows));
        batch.set("decode_rows", serve::Json(st.decode_rows));
        batch.set("prompt_rows", serve::Json(st.prompt_rows));
        batch.set("mean_rows_per_step", serve::Json(st.steps ? static_cast<double>(st.rows) / static_cast<double>(st.steps) : 0.0));
        batch.set("mean_step_ms", serve::Json(st.steps ? st.step_seconds * 1e3 / static_cast<double>(st.steps) : 0.0));
        out.set("batch", std::move(batch));
        serve::Json pf = serve::Json::object();
        pf.set("scheduler", serve::Json(strix::deepseek41::sched::mode_name(sched.mode)));
        pf.set("service_target_ms", serve::Json(sched.service_ms));
        pf.set("effective_unit_target_ms", serve::Json(sched.mode == strix::deepseek41::sched::Mode::interleave ? sched.target_ms() : 0.0));
        pf.set("budget_source", serve::Json(sched.budget_source()));
        pf.set("short_first", serve::Json(sched.short_first));
        pf.set("max_unit_ms", serve::Json(st.scheduler_max_unit_ms));
        pf.set("cancellations", serve::Json(st.prefill_cancellations));
        pf.set("direct_prefill_below_rows", serve::Json(static_cast<std::uint64_t>(sched.min_rows)));
        pf.set("fallback_reason", serve::Json(sched.mode != strix::deepseek41::sched::Mode::interleave ? "scheduler-not-interleave" :
            sched.legacy_units ? "explicit-legacy-budget" : sched.service_ms == 0.0 ? "service-disabled" : ""));
        pf.set("path", serve::Json(a.device_prefill ? "device" : "rows"));
        pf.set("prefills", serve::Json(st.prefills));
        pf.set("prefill_tokens", serve::Json(st.prefill_tokens));
        pf.set("cached_tokens", serve::Json(st.cached_tokens));
        pf.set("prefill_seconds", serve::Json(st.prefill_seconds));
        if (a.prefix_cache_mb != 0U) {
            const auto& cs = st.cache;
            serve::Json pc = serve::Json::object();
            pc.set("profile", serve::Json(a.cache_profile));
            pc.set("inrun", serve::Json(a.prefix_inrun));
            pc.set("inrun_reach", serve::Json(static_cast<std::uint64_t>(a.prefix_inrun_reach)));
            pc.set("before_tail", serve::Json(a.prefix_before_tail));
            pc.set("session", serve::Json(a.prefix_session));
            pc.set("budget_bytes", serve::Json(a.prefix_cache_mb << 20U));
            pc.set("bytes", serve::Json(cs.bytes));
            pc.set("blocks", serve::Json(cs.blocks));
            pc.set("snapshots", serve::Json(cs.snapshots));
            pc.set("lookups", serve::Json(cs.lookups));
            pc.set("hits", serve::Json(cs.hits));
            pc.set("hit_tokens", serve::Json(cs.hit_tokens));
            pc.set("inserts", serve::Json(cs.inserts));
            pc.set("evicted_snapshots", serve::Json(cs.evicted_snapshots));
            pc.set("diagnostics", serve::Json(std::getenv("STRIX_DS41_PREFIX_DIAGNOSTICS") && std::string_view(std::getenv("STRIX_DS41_PREFIX_DIAGNOSTICS")) == "1"));
            pc.set("export_bytes", serve::Json(cs.export_bytes));
            pc.set("export_inrun", serve::Json(cs.export_inrun));
            pc.set("export_run", serve::Json(cs.export_run));
            pc.set("export_handoff", serve::Json(cs.export_handoff));
            pc.set("export_session", serve::Json(cs.export_session));
            pc.set("row_layers", serve::Json(cs.row_layers));
            pc.set("planned_cache_splits", serve::Json(cs.planned_splits));
            pc.set("capture_checks", serve::Json(cs.capture_checks));
            pc.set("capture_bypasses", serve::Json(cs.capture_bypasses));
            pc.set("capture_predicted_peak_bytes", serve::Json(cs.capture_predicted_peak));
            pc.set("host_ceiling_bytes", serve::Json(cs.host_ceiling_bytes));
            pc.set("refused", serve::Json(cs.refused));
            if (!a.prefix_dir.empty()) {
                serve::Json d = serve::Json::object();
                d.set("budget_bytes", serve::Json(a.prefix_disk_mb << 20U));
                d.set("tier_unlimited", serve::Json(a.prefix_disk_mb == 0U));
                d.set("service_directory_budget_bytes", serve::Json(std::uint64_t{4096} << 20U));
                d.set("queue_budget_bytes", serve::Json(std::uint64_t{128} << 20U));
                d.set("min_free_bytes", serve::Json(std::uint64_t{16384} << 20U));
                d.set("promote", serve::Json(false));
                d.set("bytes", serve::Json(cs.disk_bytes));
                d.set("blocks", serve::Json(cs.disk_blocks));
                d.set("snapshots", serve::Json(cs.disk_snapshots));
                d.set("hits", serve::Json(cs.disk_hits));
                d.set("written_bytes", serve::Json(cs.written_bytes));
                d.set("write_seconds", serve::Json(cs.write_seconds));
                d.set("write_errors", serve::Json(cs.write_errors));
                d.set("queued_bytes", serve::Json(cs.queued_bytes));
                d.set("queue_peak_bytes", serve::Json(cs.queue_peak_bytes));
                d.set("resource_refused", serve::Json(cs.disk_resource_refused));
                d.set("read_bytes", serve::Json(cs.read_bytes));
                d.set("read_seconds", serve::Json(cs.read_seconds));
                d.set("read_errors", serve::Json(cs.read_errors));
                d.set("scanned_files", serve::Json(cs.scanned_files));
                d.set("ignored_files", serve::Json(cs.ignored_files));
                d.set("dropped", serve::Json(cs.disk_dropped));
                pc.set("disk", std::move(d));
            }
            pf.set("prefix_cache", std::move(pc));
        }
        out.set("prefill", std::move(pf));
        const auto& cs = st.comm;  // as of the last step that copied it
        serve::Json group = serve::Json::object();
        group.set("group_id", serve::Json(a.group_id));
        group.set("epoch", serve::Json(a.epoch));
        group.set("rank", serve::Json(0));
        group.set("world", serve::Json(2));
        group.set("state", serve::Json(state.failed.load() ? "poisoned" : state.draining.load() ? "draining" : "ok"));
        group.set("transport", serve::Json(comm.transport_name()));
        group.set("exchanges", serve::Json(cs.exchanges));
        group.set("bytes_sent", serve::Json(cs.bytes_sent));
        group.set("exchange_seconds", serve::Json(cs.seconds));
        group.set("mean_exchange_us", serve::Json(cs.exchanges ? cs.seconds * 1e6 / static_cast<double>(cs.exchanges) : 0.0));
        if (armed) {  // the exchange time split (stage: device to host, wire, checks)
            const auto mean_us = [&](double s) { return serve::Json(cs.exchanges ? s * 1e6 / static_cast<double>(cs.exchanges) : 0.0); };
            group.set("stage_seconds", serve::Json(cs.stage));
            group.set("wire_seconds", serve::Json(cs.wire));
            group.set("check_seconds", serve::Json(cs.check));
            group.set("overlap_seconds", serve::Json(cs.overlap));
            group.set("finish_wait_seconds", serve::Json(cs.finish_wait));
            group.set("mean_stage_us", mean_us(cs.stage));
            group.set("mean_wire_us", mean_us(cs.wire));
            group.set("mean_check_us", mean_us(cs.check));
        }
        out.set("group", std::move(group));
        if (armed && spec && spec_published) {  // DSpark windows (the speculator's lifetime counters)
            serve::Json d = serve::Json::object();
            d.set("steps", serve::Json(ss.steps));
            d.set("drafted_steps", serve::Json(ss.drafted_steps));
            d.set("windows", serve::Json(ss.windows));
            d.set("plain_rows", serve::Json(ss.plain_rows));
            d.set("rows", serve::Json(ss.rows));
            d.set("drafts_verified", serve::Json(ss.drafts_verified));
            d.set("drafts_accepted", serve::Json(ss.drafts_accepted));
            d.set("acceptance", ss.drafts_verified ? serve::Json(static_cast<double>(ss.drafts_accepted) / static_cast<double>(ss.drafts_verified))
                                                   : serve::Json());
            d.set("decode_steps", serve::Json(ss.decode_steps));
            d.set("decode_tokens", serve::Json(ss.decode_tokens));
            d.set("draft_seconds", serve::Json(ss.draft_seconds));
            d.set("plan_seconds", serve::Json(ss.plan_seconds));
            d.set("step_seconds", serve::Json(ss.step_seconds));
            d.set("finish_seconds", serve::Json(ss.finish_seconds));
            d.set("decode_seconds", serve::Json(ss.decode_seconds));
            serve::Json rows = serve::Json::array();  // index = rows of a window (1: a plain row)
            for (const auto n : ss.window_rows) rows.push(serve::Json(n));
            d.set("window_rows", std::move(rows));
            serve::Json depth = serve::Json::array();  // depth j + 1: verified after j correct drafts, correct
            for (std::size_t j = 0; j < ss.reached.size(); ++j) {
                serve::Json e = serve::Json::object();
                e.set("depth", serve::Json(static_cast<std::uint64_t>(j + 1U)));
                e.set("reached", serve::Json(ss.reached[j]));
                e.set("hit", serve::Json(ss.hit[j]));
                e.set("hit_rate", ss.reached[j] ? serve::Json(static_cast<double>(ss.hit[j]) / static_cast<double>(ss.reached[j])) : serve::Json());
                depth.push(std::move(e));
            }
            d.set("depth", std::move(depth));
            out.set("dspark", std::move(d));
        }
    };
    config.metrics_extra = [&] {
        std::ostringstream m;
        m.precision(std::numeric_limits<double>::max_digits10);  // exact values: integers as integers, doubles round-trip
        const bool armed = serve::observe_armed();
        strix::deepseek41::BatchSpecStats ss;
        ServeState::Copy st;
        {
            const std::lock_guard lock(state.mutex);  // copies only; formatting after it
            st = state.copy_locked(armed, ss);
        }
        m << "# HELP strix_slots_busy Request slots in use.\n# TYPE strix_slots_busy gauge\nstrix_slots_busy " << st.busy << '\n';
        m << "# HELP strix_batch_steps_total Forward steps.\n# TYPE strix_batch_steps_total counter\nstrix_batch_steps_total " << st.steps << '\n';
        m << "# HELP strix_batch_rows_total Rows of the forward steps.\n# TYPE strix_batch_rows_total counter\nstrix_batch_rows_total " << st.rows << '\n';
        m << "# HELP strix_group_exchanges_total Exchanges with the peer rank.\n# TYPE strix_group_exchanges_total counter\nstrix_group_exchanges_total "
          << st.comm.exchanges << '\n';
        if (!armed) return m.str();
        const auto& cs = st.comm;
        const double step_seconds = st.step_seconds;
        const bool spec_published = st.spec_published;
        const auto counter = [&](const char* name, const char* help, auto value) {
            m << "# HELP " << name << ' ' << help << "\n# TYPE " << name << " counter\n" << name << ' ' << value << '\n';
        };
        counter("strix_group_bytes_sent_total", "Bytes sent to the peer rank.", cs.bytes_sent);
        counter("strix_group_exchange_seconds_total", "Wall time inside the exchanges with the peer rank.", cs.seconds);
        counter("strix_group_exchange_stage_seconds_total", "Exchange time staging device data to host memory.", cs.stage);
        counter("strix_group_exchange_wire_seconds_total", "Exchange time on the transport, waiting for the peer included.", cs.wire);
        counter("strix_group_exchange_check_seconds_total", "Exchange time in checks and host to device copies.", cs.check);
        counter("strix_batch_step_seconds_total", "Wall time of the forward steps.", step_seconds);
        if (spec && spec_published) {
            counter("strix_dspark_drafts_proposed_total", "DSpark drafts verified.", ss.drafts_verified);
            counter("strix_dspark_drafts_accepted_total", "DSpark drafts accepted.", ss.drafts_accepted);
            counter("strix_dspark_windows_total", "DSpark windows (steps of a request with drafts).", ss.windows);
            m << "# HELP strix_dspark_depth_reached_total Drafts verified at each depth (after every earlier draft was correct).\n"
                 "# TYPE strix_dspark_depth_reached_total counter\n";
            for (std::size_t j = 0; j < ss.reached.size(); ++j) m << "strix_dspark_depth_reached_total{depth=\"" << j + 1U << "\"} " << ss.reached[j] << '\n';
            m << "# HELP strix_dspark_depth_hit_total Drafts correct at each depth.\n# TYPE strix_dspark_depth_hit_total counter\n";
            for (std::size_t j = 0; j < ss.hit.size(); ++j) m << "strix_dspark_depth_hit_total{depth=\"" << j + 1U << "\"} " << ss.hit[j] << '\n';
        }
        return m.str();
    };
    serve::ApiService api(config, tokenizer, queue, a.context, false);
    serve::HttpServerConfig http;
    http.bind_address = a.http_bind;
    http.port = static_cast<std::uint16_t>(a.http_port);
    http.max_connections = a.max_connections;
    ImportBox imports;
    SessionBox session_jobs;
    serve::HttpServer server(http, [&](const serve::HttpRequest& request, serve::HttpResponseWriter& response) {
        if (strix::deepseek41::sched::session_admin_route(a.session_ram_bytes != 0, request.path))
            return handle_session(a, config, session_jobs, request, response);
        if (request.path == "/admin/import-state") return handle_import(a, config, state, imports, request, response);
        api.handle(request, response);
    });
    server.listen();
    // --import-frames PORT: the frames import listener lives as long as this function (destroyed on every exit path)
    std::unique_ptr<strix::deepseek41::pieces::FramesImportListener> frames;
    if (a.import_frames_port != 0U) {
        frames = std::make_unique<strix::deepseek41::pieces::FramesImportListener>(
            static_cast<std::uint16_t>(a.import_frames_port),
            [&](strix::handoff::Session& s, strix::deepseek41::pieces::FramesImportListener& lis) { frames_import_session(a, state, imports, s, lis); });
        std::cerr << "strix serve: import frames listener on port " << frames->port() << std::endl;
    }
    api.set_state("ready");
    std::cerr << "strix serve: listening on " << a.http_bind << ':' << server.port() << " model " << a.model_id << std::endl;
    {
        if (dspark_free_before != 0U) log_dspark_resident(0U, dspark_free_before, fwd.device_memory().first);
        serve::Json ready = serve::Json::object();
        ready.set("event", serve::Json("READY"));
        ready.set("rank", serve::Json(0));
        ready.set("http", serve::Json(a.http_bind + ":" + std::to_string(server.port())));
        ready.set("model", serve::Json(a.model_id));
        ready.set("slots", serve::Json(static_cast<std::uint64_t>(a.slots)));
        ready.set("context", serve::Json(static_cast<std::uint64_t>(a.context)));
        ready.set("group_id", serve::Json(a.group_id));
        ready.set("epoch", serve::Json(a.epoch));
        event(ready);
    }
    std::thread http_thread([&server] { server.serve(); });

    std::vector<std::uint32_t> pending_commits;  // rank 0's commits of the last step (rank 1 applies them first)
    const auto take_commits = [&](Control& c) {
        if (spec && a.dspark_ranks == 2U) {  // the commits no draft round carried (a round carries those made before it)
            pending_commits = spec->take_commits();
        }
        c.commits = static_cast<std::uint32_t>(pending_commits.size());
        for (std::size_t i = 0; i < pending_commits.size(); ++i) c.commit[i] = pending_commits[i];
        pending_commits.clear();
    };
    // where rank 0's decode loop spends its time (stderr every 100 steps and at the end)
    struct Profile {
        std::uint64_t steps = 0;
        double plan = 0.0, control = 0.0, step = 0.0, finish = 0.0, between = 0.0;
        Clock::time_point last_end;
        // steps by row count (1..16) and by requests running (0..8+), and the steps whose drafting was allowed
        std::array<std::uint64_t, 17> by_rows{};
        std::array<std::uint64_t, 9> by_running{};
        std::uint64_t drafts_allowed = 0;
        void count(std::size_t rows, std::size_t in_flight, bool allowed) {
            ++by_rows[std::min<std::size_t>(rows, 16U)];
            ++by_running[std::min<std::size_t>(in_flight, 8U)];
            drafts_allowed += allowed ? 1U : 0U;
        }
        static std::string hist(std::span<const std::uint64_t> h) {
            std::string s;
            for (std::size_t i = 0; i < h.size(); ++i)
                if (h[i] != 0U) s += (s.empty() ? "" : ",") + std::to_string(i) + ":" + std::to_string(h[i]);
            return s.empty() ? "-" : s;
        }
        void print(const strix::deepseek41::BatchSpeculator* sp) const {
            const double n = static_cast<double>(steps);
            std::fprintf(stderr, "loop profile: %llu steps; per step ms: plan %.2f, control %.2f, step %.2f, finish %.2f, between steps %.2f",
                         static_cast<unsigned long long>(steps), plan * 1e3 / n, control * 1e3 / n, step * 1e3 / n, finish * 1e3 / n, between * 1e3 / n);
            if (sp) {
                const auto& st = sp->stats();
                const auto& c = sp->controller().costs();
                std::fprintf(stderr, "; dspark: %llu drafted steps, draft %.1f s total, plan %.1f s, finish %.1f s, windows %llu, plain rows %llu,"
                             " drafts %llu accepted of %llu (cooperative %llu), lookup windows %llu (%llu accepted of %llu, host %.1f ms); costs ms plain[1..4] %.1f %.1f %.1f %.1f"
                             " row %.2f draft %.2f round %.2f coop %.2f",
                             static_cast<unsigned long long>(st.drafted_steps), st.draft_seconds, st.plan_seconds, st.finish_seconds,
                             static_cast<unsigned long long>(st.windows), static_cast<unsigned long long>(st.plain_rows),
                             static_cast<unsigned long long>(st.drafts_accepted), static_cast<unsigned long long>(st.drafts_verified),
                             static_cast<unsigned long long>(st.coop_drafts), static_cast<unsigned long long>(st.lookup_windows), static_cast<unsigned long long>(st.lookup_accepted),
                             static_cast<unsigned long long>(st.lookup_verified), st.lookup_seconds * 1e3, c.plain[1] * 1e3, c.plain[2] * 1e3, c.plain[3] * 1e3, c.plain[4] * 1e3,
                             c.row * 1e3, c.draft * 1e3, c.round * 1e3, c.coop * 1e3);
            }
            std::fprintf(stderr, "; steps by rows %s, by running %s, drafts allowed %llu; %s\n", hist(by_rows).c_str(), hist(by_running).c_str(),
                         static_cast<unsigned long long>(drafts_allowed), plans_since_ready().c_str());
        }
    } prof;
    std::vector<std::unique_ptr<Running>> running;
    std::vector<std::uint32_t> free_slots;
    for (std::uint32_t s = a.slots; s-- > 0U;) free_slots.push_back(s);
    std::uint32_t pending_resets = 0;
    std::uint32_t pending_snaps = 0;  // session snapshots rank 0 took after the last step (rank 1 takes them on the next control)
    std::vector<std::uint32_t> pool_ops;              // KV pool operations since the last step (sent to rank 1)
    std::deque<serve::TakenJob> pool_waiting;         // taken from the queue, waiting for room in the KV pool (FIFO)
    const auto pool_release = [&](std::uint32_t slot) {
        if (a.pool == 0U) return;
        fwd.release(slot);
        pool_ops.push_back(kPoolRelease << 30U | slot << 24U);
    };
    std::deque<Parked> parked;  // imported states waiting for their request (oldest first)
    auto last_round = Clock::now();
    // Local call and rank-1 command share the same transaction. Completion is acknowledged
    // before a captured turn is reported complete or a restored slot becomes schedulable.
    const auto session_call = [&](const serve::Json &message, std::uint32_t slot) -> std::optional<SessionCheckpoint> {
        Control c;
        c.command = static_cast<std::uint32_t>(Command::session);
        c.pslot = slot;
        take_commits(c);
        c.snaps = pending_snaps;
        pending_snaps = 0;
        if (pool_ops.size() > std::size(c.op))
            throw std::runtime_error("too many session pool operations");
        c.ops = static_cast<std::uint32_t>(pool_ops.size());
        std::copy(pool_ops.begin(), pool_ops.end(), c.op);
        pool_ops.clear();
        const auto text = message.dump();
        SessionBytes bytes(text.begin(), text.end());
        c.prefix_bytes = bytes.size();
        (void)exchange(fwd, c);
        transfer_state(fwd, bytes, nullptr);
        std::optional<SessionCheckpoint> restored;
        std::string error;
        dog.pause(true);
        try {
            restored = session_operation(a, fwd, *sessions, message, slot, 0U);
        } catch (const std::exception &e) {
            error = e.what();
        }
        dog.pause(false);
        Control ack;
        ack.command = static_cast<std::uint32_t>(Command::heartbeat);
        const auto peer_ack = exchange(fwd, ack);
        dog.progress();
        last_round = Clock::now();
        if (!error.empty() || peer_ack.pkey != 1)
            throw std::runtime_error(error.empty() ? "session operation refused on peer" : error);
        return restored;
    };
    const auto finish = [&](std::size_t i, const std::string& reason, const std::string& error) {
        auto& r = *running[i];
        serve::GenerationStats st;
        st.prompt_tokens = r.seq.prompt.size();
        r.account.report(st);  // every request (the prefill rate); imported reported as a field only when observed
        st.generated_tokens = r.seq.generated.size();
        st.cycles = r.seq.generated.size();
        const auto now = Clock::now();
        st.prefill_ms = std::chrono::duration<double, std::milli>((r.has_first ? r.first_token : now) - r.start).count();
        st.decode_ms = r.has_first ? std::chrono::duration<double, std::milli>(now - r.first_token).count() : 0.0;
        st.finish = reason;
        st.sampling = "greedy";
        if (r.job.job.observation_complete()) {  // its armed period held to here: the drafts of every step were attributed
            st.proposed = r.proposed;
            st.accepted = r.accepted;
            if (r.has_first) st.first_token_ms = std::chrono::duration<double, std::milli>(r.first_token - r.job.submitted).count();
        }
        std::string session_error = error;
        if (sessions && !r.job.job.session_id.empty() && error.empty() && reason != "cancelled" && !r.seq.generated.empty()) {
            auto message = serve::Json::object();
            message.set("op", serve::Json("capture"));
            message.set("id", serve::Json(r.job.job.session_id));
            message.set("generation", serve::Json(strix::deepseek41::session_generation()));
            message.set("pending", serve::Json(static_cast<std::uint64_t>(r.seq.generated.back())));
            message.set("emitted", serve::Json(r.session_emitted + r.seq.generated.size()));
            auto history = r.seq.prompt;
            history.insert(history.end(), r.seq.generated.begin(), r.seq.generated.end() - 1);
            auto ids = serve::Json::array();
            for (auto t : history)
                ids.push(serve::Json(static_cast<std::uint64_t>(t)));
            message.set("history", std::move(ids));
            try {
                session_call(message, r.seq.slot);
            } catch (const std::exception &e) {
                session_error = e.what();
            }
        }
        if (!r.job.job.session_id.empty() && !session_error.empty()) {
            const auto failure = serve::session_failure(session_error);
            st.error_status = failure.status;
            st.error_code = failure.code;
        }
        queue.finish(r.job, st, session_error);
        pool_release(r.seq.slot);
        free_slots.push_back(r.seq.slot);
        running.erase(running.begin() + static_cast<std::ptrdiff_t>(i));
        const std::lock_guard lock(state.mutex);
        state.busy = static_cast<std::uint32_t>(running.size());
    };
    // publish the new tokens, then end the requests that are done
    const auto publish = [&] {
        for (std::size_t i = running.size(); i-- > 0U;) {
            auto& r = *running[i];
            if (r.seq.generated.size() == r.published) continue;
            {  // a window emits several tokens: the ones after a stop token or past max_tokens are dropped
                auto& g = r.seq.generated;
                const auto& stops = r.job.job.stop_tokens;
                std::size_t end = std::min<std::size_t>(g.size(), std::max<std::size_t>(r.job.job.max_tokens, r.published + 1U));
                for (std::size_t k = r.published; k < end; ++k)
                    if (std::find(stops.begin(), stops.end(), g[k]) != stops.end()) {
                        end = k + 1U;
                        break;
                    }
                g.resize(end);
            }
            if (!r.has_first) {
                r.has_first = true;
                r.first_token = Clock::now();
            }
            const std::span<const std::uint32_t> fresh(r.seq.generated.data() + r.published, r.seq.generated.size() - r.published);
            r.published = r.seq.generated.size();
            const bool more = r.job.job.on_tokens ? r.job.job.on_tokens(fresh) : true;
            const auto& stops = r.job.job.stop_tokens;
            const bool stop = std::find(stops.begin(), stops.end(), fresh.back()) != stops.end();
            if (!more) finish(i, "cancelled", {});
            else if (stop) finish(i, "stop", {});
            else if (r.seq.generated.size() >= r.job.job.max_tokens) finish(i, "length", {});
            else if (r.seq.position + 1U >= a.context) finish(i, "context", {});
        }
    };
    // a parked state's slot back to the free ones
    const auto unpark_oldest = [&] {
        const auto p = parked.front();
        parked.pop_front();
        pool_release(p.slot);
        free_slots.push_back(p.slot);
        std::fprintf(stderr, "import: parked state of slot %u (position %u) dropped\n", p.slot, p.position);
    };
    // POST /admin/import-state: the state into a free slot on both ranks (rank 1 gets the command with the pending pool
    // operations, then the bytes), parked for the request that continues it
    const auto import_one = [&](ImportJob& j) {
        ImportResult res;
        res.queue_seconds = std::chrono::duration<double>(Clock::now() - j.queued).count();
        const auto& st = j.state;
        while (!parked.empty() && parked.size() >= std::max<std::uint32_t>(a.import_parked, 1U)) unpark_oldest();
        if (free_slots.empty() && !parked.empty()) unpark_oldest();
        if (free_slots.empty()) {
            res.status = 503;
            res.error = "no free slot for the state";
            j.done.set_value(res);
            return;
        }
        const std::uint32_t slot = free_slots.back();
        std::uint32_t want = 0;
        if (a.pool != 0U) {
            want = static_cast<std::uint32_t>(std::min<std::uint64_t>(a.context, std::uint64_t{st.position} + j.reserve_tokens + 1U));
            if (!fwd.reserve(slot, want)) {
                res.status = 503;
                res.error = "the KV pool has no room for the state";
                j.done.set_value(res);
                return;
            }
            pool_ops.push_back(kPoolReserve << 30U | slot << 24U | want);
        }
        free_slots.pop_back();
        pending_resets &= ~(1U << slot);  // a reset still pending from a request that left before its first step
        const auto t0 = Clock::now();
        Control c;
        c.command = static_cast<std::uint32_t>(Command::import_state);
        if (pool_ops.size() > std::size(c.op)) throw std::runtime_error("too many KV pool operations in one round");
        c.ops = static_cast<std::uint32_t>(pool_ops.size());
        std::copy(pool_ops.begin(), pool_ops.end(), c.op);
        pool_ops.clear();
        c.pslot = slot;
        c.prefix_bytes = j.bytes.size();
        c.pkey = j.digest;
        c.snaps = pending_snaps;  // rank 1 takes the last step's session snapshots before the import may reuse a slot
        pending_snaps = 0;
        take_commits(c);  // rank 1 applies commits at every control block (before it stages the last step's rows)
        (void)exchange(fwd, c);
        transfer_state(fwd, j.bytes, nullptr);
        dog.progress();
        const auto t1 = Clock::now();
        if (a.pool == 0U) fwd.reset(slot);
        fwd.import_state(slot, st);
        if (stage) stage->import_state(slot, st);
        if (spec) spec->forget(slot);  // its drafting contexts no longer in step on both ranks
        if (prefill) prefill->on_prefill(slot, st.tokens);
        Control ack;
        ack.command = static_cast<std::uint32_t>(Command::heartbeat);
        const auto acked = exchange(fwd, ack);  // rank 1 answers with the digest it imported
        dog.progress();
        last_round = Clock::now();
        if (acked.pkey != j.digest) throw std::runtime_error("imported state: rank 1 did not import the same state");
        res.slot = slot;
        res.replicate_seconds = std::chrono::duration<double>(t1 - t0).count();
        res.device_seconds = std::chrono::duration<double>(last_round - t1).count();
        Parked p;
        p.slot = slot;
        p.position = st.position;
        p.next_token = st.next_token;
        p.tokens = st.tokens;
        p.sha256 = j.sha256;
        parked.push_back(std::move(p));
        std::fprintf(stderr, "import: slot %u position %u next token %u, %zu bytes, digest %s, sha256 %.16s..; queue %.4f s, to rank 1 %.4f s, import %.4f s\n",
                     slot, st.position, st.next_token, j.bytes.size(), hex16(j.digest).c_str(), j.sha256.c_str(), res.queue_seconds,
                     res.replicate_seconds, res.device_seconds);
        j.done.set_value(res);
    };
    // the parked state a prompt continues (the longest one whose tokens begin the prompt and whose slot has room)
    const auto match_parked = [&](const std::vector<std::uint32_t>& prompt, std::uint32_t max_tokens, const std::string &session_id = {}) -> std::ptrdiff_t {
        std::ptrdiff_t best = -1;
        for (std::size_t i = 0; i < parked.size(); ++i) {
            const auto& p = parked[i];
            if (p.session_id != session_id)
                continue;
            if (!session_id.empty() && (prompt.size() <= p.position || prompt[p.position] != p.next_token))
                continue;
            if (prompt.size() < p.position || !std::equal(p.tokens.begin(), p.tokens.end(), prompt.begin())) continue;
            if (a.pool != 0U && fwd.capacity(p.slot) < std::min<std::uint64_t>(a.context, prompt.size() + std::uint64_t{max_tokens} + 1U)) continue;
            if (best < 0 || p.position > parked[static_cast<std::size_t>(best)].position) best = static_cast<std::ptrdiff_t>(i);
        }
        return best;
    };
    // the device prefill of a request's prompt on both ranks (rank 1 gets the command with the pending resets and pool
    // operations, then the prompt); as_task (STRIX_DS41_SERVE_SCHED=interleave): only started, its units follow
    const auto prefill_done = [&](Running& r, const PrefillOutcome& o, const strix::deepseek41::PrefixLookup& hit, bool disk_failed,
                                  Clock::time_point t0) {
        const auto& prompt = r.seq.prompt;
        const auto plen = static_cast<std::uint32_t>(prompt.size());
        r.seq.position = plen;
        r.seq.fed = prompt.size();
        r.seq.generated.assign(1U, o.next_token);
        r.device_prefill = false;
        if (stage) {  // the split decode: layers [0, cut) of the prefilled state go to the FRONT's slot
            auto st = fwd.export_state(r.seq.slot);
            st.tokens.assign(prompt.begin(), prompt.end());
            st.next_token = o.next_token;
            stage->import_state(r.seq.slot, st);
        }
        if (spec) spec->imported(r.seq.slot, prompt, plen, o.next_token, o.seed.streams, o.seed.rows, fwd.stream());
        r.account.resumed(hit.resume);
        const double total = std::chrono::duration<double>(Clock::now() - t0).count();
        std::fprintf(stderr, "prefill slot %u: %u tokens, resume %u (shared %u%s), %u pieces, %u inserted; token %u logits_fnv %016llx; %.3f s (state %.3f, "
                     "prefill %.3f [inserts %.3f], slot import %.3f, control + prompt %.3f)",
                     r.seq.slot, plen, hit.resume, hit.shared, hit.disk ? ", from disk" : "", o.pieces, o.inserted, o.next_token,
                     static_cast<unsigned long long>(o.logits), total, o.state_seconds, o.prefill_seconds, o.insert_seconds, o.handoff_seconds, total - o.seconds);
        if (prefill->disk_bytes() != 0U)
            std::fprintf(stderr, " disk read %.1f MB in %.3f s (%.2f GB/s)", static_cast<double>(prefill->disk_bytes()) / 1e6, prefill->disk_seconds(),
                         prefill->disk_seconds() > 0.0 ? static_cast<double>(prefill->disk_bytes()) / prefill->disk_seconds() / 1e9 : 0.0);
        if (disk_failed) std::fprintf(stderr, " DISK READ FAILED on a rank: path forgotten, ran as a miss");
        std::fprintf(stderr, "\n");
        const std::lock_guard lock(state.mutex);
        ++state.prefills;
        state.prefill_tokens += plen;
        state.cached_tokens += hit.resume;
        state.prefill_seconds += total;
        state.cache = prefill->cache_stats();
    };
    struct ActiveTask {  // interleave: the prefill running in units (one at a time: DevicePrefill holds one sequence)
        Running* r = nullptr;
        std::unique_ptr<SlotPrefill::Task> task;
        strix::deepseek41::PrefixLookup hit;
        bool disk_failed = false;
        Clock::time_point t0;
        std::uint32_t units = 0, steps_between = 0;
        double unit_max = 0.0, unit_sum = 0.0;
    };
    std::optional<ActiveTask> active;
    const auto finish_task = [&] {
        auto& at = *active;
        const auto o = at.task->finish();
        prefill->on_prefill(at.r->seq.slot, at.r->seq.prompt);
        last_round = Clock::now();
        prefill_done(*at.r, o, at.hit, at.disk_failed, at.t0);
        if (sched.log)
            std::fprintf(stderr, "sched: prefill slot %u done in %u units (max %.3f s, sum %.3f s; longest resumable step %.1f ms), %u decode steps between, %.3f s wall\n", at.r->seq.slot,
                         at.units, at.unit_max, at.unit_sum, at.task->step_max_ms(), at.steps_between, std::chrono::duration<double>(Clock::now() - at.t0).count());
        active.reset();
    };
    const auto device_prefill = [&](Running& r, bool as_task) {
        const auto& prompt = r.seq.prompt;
        const auto t0 = Clock::now();
        auto hit = prefill->lookup(prompt);
        bool insert = prefill->has_cache();
        Control c;
        c.command = static_cast<std::uint32_t>(Command::prefill);
        c.resets = pending_resets;
        if (pool_ops.size() > std::size(c.op)) throw std::runtime_error("too many KV pool operations in one round");
        c.ops = static_cast<std::uint32_t>(pool_ops.size());
        std::copy(pool_ops.begin(), pool_ops.end(), c.op);
        pool_ops.clear();
        c.pslot = r.seq.slot;
        c.plen = static_cast<std::uint32_t>(prompt.size());
        c.presume = hit.resume;
        c.pshared = hit.shared;
        c.pkey = hit.key;
        // the stable prefix in blocks (only the in-run snapshots use it); rank 1 plans the same snapshot points
        const std::uint32_t stable = a.prefix_inrun ? std::min<std::uint32_t>(r.job.job.stable_prefix, static_cast<std::uint32_t>(prompt.size())) : 0U;
        const std::uint32_t stable_blocks = std::min<std::uint32_t>(stable / a.prefix_block, 0xFFFFFFU);
        c.pflags = (insert ? 1U : 0U) | (as_task ? kPrefillTask : 0U) | (stable_blocks << kPrefillHintShift);
        c.snaps = pending_snaps;
        pending_snaps = 0;
        take_commits(c);
        (void)exchange(fwd, c);
        for (std::uint32_t s = 0; s < a.slots; ++s)
            if ((pending_resets >> s) & 1U) {
                fwd.reset(s);
                if (stage) stage->reset(s);
                prefill->on_reset(s);
            }
        pending_resets = 0;
        send_prompt(fwd, prompt, nullptr);
        dog.pause(true);
        bool disk_failed = false;
        const bool ok = prefill->prepare(prompt, hit);
        if (hit.disk) {  // all or nothing: both ranks read their files
            Control ack;
            ack.command = static_cast<std::uint32_t>(Command::prefill);
            ack.pflags = ok ? 0U : 2U;
            const auto answer = exchange(fwd, ack);
            if (!ok || (answer.pflags & 2U) != 0U) {
                prefill->forget(prompt, hit);
                hit = strix::deepseek41::PrefixLookup{};
                disk_failed = true;
            }
        }
        const bool capture = prefill->has_cache() && hit.resume < prompt.size() && prompt.size() >= a.prefix_block;
        insert = capture && prefill->admit(static_cast<std::uint32_t>(prompt.size()), insert);
        if (as_task) {
            active.emplace();
            active->r = &r;
            active->task = prefill->begin(r.seq.slot, prompt, hit, insert, stable_blocks * a.prefix_block);
            active->hit = hit;
            active->disk_failed = disk_failed;
            active->t0 = t0;
            dog.pause(false);
            last_round = Clock::now();
            if (active->task->done()) finish_task();  // the whole prompt was cached (rank 1 finishes it too)
            return;
        }
        const auto o = prefill->run(r.seq.slot, prompt, hit, insert, stable_blocks * a.prefix_block);
        prefill->on_prefill(r.seq.slot, prompt);
        dog.pause(false);
        last_round = Clock::now();
        prefill_done(r, o, hit, disk_failed, t0);
    };
    // interleave: the active task's next unit on both ranks (budget_rows 0: up to its next default piece end)
    // A resumable task stopped inside a piece continues that piece, including while idle.
    const auto run_unit = [&](std::uint32_t budget_rows, bool decoding) {
        auto& at = *active;
        const std::uint32_t end = at.task->mid_piece() ? at.task->piece_end()
                                                       : strix::deepseek41::sched::next_unit_end(at.task->rows(), at.task->default_ends(), budget_rows, kMinRows);
        const bool others_waiting = std::any_of(running.begin(), running.end(), [&](const auto& r) { return r->device_prefill && r.get() != at.r; });
        const std::uint32_t budget = strix::deepseek41::sched::tick_budget(sched, decoding, others_waiting,
            [&](double ms) { return at.task->units_within(ms); });
        Control c;
        c.command = static_cast<std::uint32_t>(Command::prefill_unit);
        take_commits(c);  // rank 1 applies commits at every control block
        c.pend = end;
        c.pbudget = budget;
        c.snaps = pending_snaps;
        pending_snaps = 0;
        const auto t0 = Clock::now();
        const std::uint32_t from = at.task->rows();
        (void)exchange(fwd, c);
        dog.pause(true);
        const bool done = strix::deepseek41::sched::apply_unit(*at.task, end, budget, false) == strix::deepseek41::sched::UnitResult::complete;
        dog.pause(false);
        check_unit(fwd, 0U, {done, at.task->rows(), at.task->unit_position()});
        last_round = Clock::now();
        const double dt = std::chrono::duration<double>(last_round - t0).count();
        {
            const std::lock_guard lock(state.mutex);
            state.scheduler_max_unit_ms = std::max(state.scheduler_max_unit_ms, at.task->unit_max_ms());
        }
        ++at.units;
        at.unit_max = std::max(at.unit_max, dt);
        at.unit_sum += dt;
        if (sched.log) std::fprintf(stderr, "sched: unit slot %u rows %u..%u (%u) %.3f s\n", at.r->seq.slot, from, at.task->rows(), at.task->rows() - from, dt);
        if (done) finish_task();
    };
    const auto abort_task = [&] {
        Control c;
        c.command = static_cast<std::uint32_t>(Command::prefill_unit);
        take_commits(c);  // rank 1 applies commits at every control block
        c.pflags = kPrefillAbort;
        c.snaps = pending_snaps;
        pending_snaps = 0;
        (void)exchange(fwd, c);
        dog.progress();
        last_round = Clock::now();
        if (sched.log) std::fprintf(stderr, "sched: prefill task of slot %u aborted at row %u%s\n", active->r->seq.slot, active->task->rows(),
                                    active->task->mid_piece() ? " (cancelled at boundary)" : "");
        dog.pause(true);
        (void)strix::deepseek41::sched::apply_unit(*active->task, 0U, 0U, true);
        {
            const std::lock_guard lock(state.mutex);
            ++state.prefill_cancellations;
        }
        dog.pause(false);
        active.reset();
    };
    // a running request that waits for its device prefill continues a parked (imported) state instead
    const auto take_parked = [&](Running& r, std::size_t m) {
        if (active && active->r == &r) abort_task();
        const auto p = parked[m];
        parked.erase(parked.begin() + static_cast<std::ptrdiff_t>(m));
        pool_release(r.seq.slot);
        free_slots.push_back(r.seq.slot);
        r.seq.slot = p.slot;
        r.session_emitted = p.session_emitted;
        r.seq.position = p.position;
        r.seq.fed = p.position;
        r.device_prefill = false;
        r.account.continued(p.position);  // usage.prompt_tokens_details.cached_tokens reports them (none read again)
        if (r.seq.prompt.size() == p.position) r.seq.generated.assign(1U, p.next_token);
        std::fprintf(stderr, "import: request of %zu prompt tokens waiting for its prefill continues the state in slot %u (position %u)\n",
                     r.seq.prompt.size(), p.slot, p.position);
    };
    // one batch step of `seqs` on both ranks, then the new tokens published
    double step_ema = 0.0;  // seconds per step (recent), for the deferred admission's W
    bool worked = false;  // a step or a prefill since the last quiesce (GPU-direct: a receive may be posted early)
    const bool comm_direct = comm.device_exchange() != nullptr;
    strix::deepseek41::serve_observe::StepDraftTally<BatchSequence> tally;  // armed: drafts per request of a step
    // the next step's rows of `seqs` (DSpark windows or plain rows)
    const auto plan_seqs = [&](std::vector<BatchSequence*>& seqs) {
        strix::deepseek41::PlannedStep planned;
        if (spec) {  // a window never emits past max_tokens or the context
            std::vector<std::uint32_t> remaining;
            for (const auto& r : running) {
                if (std::find(seqs.begin(), seqs.end(), &r->seq) == seqs.end()) continue;  // not in this step (interleave)
                const std::uint64_t left = r->job.job.max_tokens > r->seq.generated.size() ? r->job.job.max_tokens - r->seq.generated.size() : 1U;
                const std::uint64_t room = a.context > r->seq.position + 1U ? a.context - r->seq.position - 1U : 1U;
                remaining.push_back(static_cast<std::uint32_t>(std::max<std::uint64_t>(1U, std::min(left, room))));
            }
            // the drafting cap: above N requests in flight the steps are plain (DSpark gains little there and delays
            // the prefills of new requests: campaign P17.1, C8 TTFT +45 %)
            spec->allow_drafts(a.dspark_max_inflight == 0U || running.size() <= a.dspark_max_inflight);
            planned = spec->plan(seqs, remaining);
        } else {
            planned = strix::deepseek41::plan_step(seqs, a.prompt_rows, fwd.row_limit());
        }
        return planned;
    };
    // pre (split decode, groups in flight): a plan made (and submitted to the FRONT) earlier, its DSpark state parked
    const auto batch_step = [&](std::vector<BatchSequence*>& seqs, const strix::deepseek41::PlannedStep* pre = nullptr,
                                const std::shared_ptr<strix::deepseek41::BatchSpeculator::ParkedPlan>& parked_plan = {}) {
        worked = true;
        const bool armed = serve::observe_armed();  // one load per step: the draft tally and the published snapshot
        const auto t_loop = Clock::now();
        if (prof.steps != 0U) prof.between += std::chrono::duration<double>(t_loop - prof.last_end).count();
        strix::deepseek41::PlannedStep planned;
        if (pre) {
            planned = *pre;
            if (spec) spec->resume(parked_plan);
        } else {
            planned = plan_seqs(seqs);
        }
        const auto& plan = planned;
        const auto t_planned = Clock::now();
        prof.plan += std::chrono::duration<double>(t_planned - t_loop).count();
        Control c;
        c.command = static_cast<std::uint32_t>(Command::step);
        c.rows = static_cast<std::uint32_t>(plan.rows.size());
        c.pend = spec ? spec->lookup_window_slots() : 0U;  // step: the slots whose window came from prompt lookup
        c.resets = pending_resets;
        c.snaps = pending_snaps;
        pending_snaps = 0;
        const auto first_rows = std::min<std::size_t>(plan.rows.size(), BatchForward::max_rows);
        std::copy_n(plan.rows.begin(), first_rows, c.row);
        if (plan.rows.size() > first_rows) std::copy(plan.rows.begin() + first_rows, plan.rows.end(), c.extra_rows);
        if (pool_ops.size() > std::size(c.op)) throw std::runtime_error("too many KV pool operations in one round");
        c.ops = static_cast<std::uint32_t>(pool_ops.size());
        std::copy(pool_ops.begin(), pool_ops.end(), c.op);
        pool_ops.clear();
        take_commits(c);
        (void)exchange(fwd, c);
        for (std::uint32_t s = 0; s < a.slots; ++s)
            if ((pending_resets >> s) & 1U) {
                fwd.reset(s);
                if (stage) stage->reset(s);
                if (prefill) prefill->on_reset(s);
            }
        pending_resets = 0;
        const auto t0 = Clock::now();
        prof.control += std::chrono::duration<double>(t0 - t_planned).count();
        if (stage && !pre) stage->submit(plan.rows);
        const auto& tokens = stage ? stage->step(plan.rows) : fwd.step(plan.rows);
        dog.progress();
        last_round = Clock::now();
        hetero_step_seconds = std::chrono::duration<double>(last_round - t0).count();
        prof.count(plan.rows.size(), running.size(), spec && (a.dspark_max_inflight == 0U || running.size() <= a.dspark_max_inflight));
        std::uint32_t decode = 0;
        for (std::size_t r = 0; r < plan.rows.size(); ++r) decode += seqs[plan.owner[r]]->reading_prompt() ? 0U : 1U;
        std::vector<BatchRow> kept_rows(plan.rows.begin(), plan.rows.end());
        std::vector<std::uint32_t> kept_tokens(tokens.begin(), tokens.end());
        if (spec) {  // verify the windows, commit, advance the sequences; the rows the commits dropped leave the history
            if (armed) tally.begin(seqs, plan.owner);
            const auto commits = spec->finish(seqs, tokens);
            if (armed) {
                // seqs is a subsequence of `running` in its order (every caller builds it walking `running`): one merge
                // walk pairs each sequence with its request
                std::size_t i = 0;
                for (auto& r : running) {
                    if (i == seqs.size()) break;
                    if (seqs[i] != &r->seq) continue;
                    if (r->job.job.observed) {
                        const auto d = tally.after(i, r->seq);
                        r->proposed += d.proposed;
                        r->accepted += d.accepted;
                    }
                    ++i;
                }
                for (; i < seqs.size(); ++i)  // not reached by the callers: a sequence out of order is paired by a search
                    for (auto& r : running)
                        if (seqs[i] == &r->seq) {
                            if (r->job.job.observed) {
                                const auto d = tally.after(i, r->seq);
                                r->proposed += d.proposed;
                                r->accepted += d.accepted;
                            }
                            break;
                        }
            }
            std::array<std::uint32_t, 64> kept{}, seen{};
            kept.fill(0xFFFFFFFFU);
            for (const auto& [slot, k] : commits) {
                kept.at(slot) = k;
                pending_commits.push_back(slot << 16U | k);
                if (stage) stage->commit(slot, k);  // the FRONT drops the window's rejected rows with the slot's next step
            }
            kept_rows.clear();
            kept_tokens.clear();
            for (std::size_t r = 0; r < plan.rows.size(); ++r)
                if (seen.at(plan.rows[r].slot)++ < kept[plan.rows[r].slot]) kept_rows.push_back(plan.rows[r]), kept_tokens.push_back(tokens[r]);
        }
        if (prefill) {  // session snapshots: slots now at a block boundary, taken before any of them finishes
            const auto cand = prefill->on_step(kept_rows, kept_tokens);
            if (cand != 0U) {
                Control snap;
                snap.command = static_cast<std::uint32_t>(Command::heartbeat);
                snap.snaps = cand;
                take_commits(snap);
                (void)exchange(fwd, snap);
                prefill->snapshot(cand);  // both ranks admit before either exports
                dog.progress();
            }
        }
        if (!spec) strix::deepseek41::apply_step(seqs, plan, tokens);
        prof.step += std::chrono::duration<double>(last_round - t0).count();
        prof.finish += std::chrono::duration<double>(Clock::now() - last_round).count();
        prof.last_end = Clock::now();
        if (++prof.steps % 100U == 0U) prof.print(spec.get());
        const double dt = std::chrono::duration<double>(last_round - t0).count();
        step_ema = step_ema == 0.0 ? dt : 0.8 * step_ema + 0.2 * dt;  // the deferred admission's W
        // a status or metrics read asked for fresh copies (one relaxed load per step otherwise); the pool's free list
        // changes on this thread only, so its scan runs here, before the lock
        const bool wanted = state.stats_wanted.load(std::memory_order_relaxed);
        std::uint64_t pool_free = 0;
        if (wanted) {
            state.stats_wanted.store(false, std::memory_order_relaxed);
            if (a.pool != 0U) pool_free = fwd.pool_free_positions();
        }
        {
            const std::lock_guard lock(state.mutex);
            ++state.steps;
            state.rows += plan.rows.size();
            state.decode_rows += decode;
            state.prompt_rows += plan.rows.size() - decode;
            state.step_seconds += dt;
            if (armed || wanted) state.comm = comm.stats();  // unarmed: one branch; the copy only when a read asked for it
            if (wanted) state.pool_free = pool_free;
            if (armed && spec) {
                state.spec = spec->stats();
                state.spec_published = true;
            }
        }
        publish();
    };
    // split decode, --hetero-groups 2: the plans in flight (submitted to the FRONT, in step order)
    struct HeteroGroup {
        std::uint32_t parity = 0;
        std::vector<std::uint32_t> slots;  // the group's requests at plan time, in `running` order
        strix::deepseek41::PlannedStep plan;
        std::shared_ptr<strix::deepseek41::BatchSpeculator::ParkedPlan> parked;
    };
    std::deque<HeteroGroup> in_flight;
    const auto group_seqs = [&](std::uint32_t parity) {  // the decoding requests of a group (slot parity), `running` order
        std::vector<BatchSequence*> v;
        for (auto& r : running)
            if (!r->device_prefill && r->seq.slot % 2U == parity) v.push_back(&r->seq);
        return v;
    };
    const auto plan_group = [&](std::uint32_t parity) {  // plan, submit to the FRONT, park
        auto seqs = group_seqs(parity);
        if (seqs.empty()) return;
        HeteroGroup g;
        g.parity = parity;
        for (const auto* q : seqs) g.slots.push_back(q->slot);
        g.plan = plan_seqs(seqs);
        stage->submit(g.plan.rows);
        if (spec) g.parked = spec->park();
        in_flight.push_back(std::move(g));
    };
    const auto step_group = [&](bool replan) {  // the oldest plan in flight; then (replan) the group's next plan
        auto g = std::move(in_flight.front());
        in_flight.pop_front();
        std::vector<BatchSequence*> seqs;
        for (const auto slot : g.slots)
            for (auto& r : running)
                if (r->seq.slot == slot && !r->device_prefill) seqs.push_back(&r->seq);
        if (seqs.size() != g.slots.size()) throw std::logic_error("split decode: a request of a plan in flight left before its step");
        batch_step(seqs, &g.plan, g.parked);
        if (replan) plan_group(g.parity);
    };
    const auto hetero_flush = [&] {  // every plan in flight runs (before anything changes the slots or the requests)
        while (!in_flight.empty()) step_group(false);
    };
    const bool hetero_pipeline = stage != nullptr && a.hetero_groups == 2U;
    const auto decode_step = [&](std::vector<BatchSequence*>& seqs) {  // one step of the decoding requests
        if (!hetero_pipeline) {
            batch_step(seqs);
            return;
        }
        // every group with decoding requests has its plan in flight (a group alone: its plan only). A group that gets its
        // first decoding request while the other one is in flight (its prefill ended meanwhile) is planned here, behind
        // the plans in flight: measured C2, it waited for the other request to finish (TTFT p50 2.7 s).
        for (const std::uint32_t parity : {0U, 1U})
            if (std::none_of(in_flight.begin(), in_flight.end(), [&](const HeteroGroup& g) { return g.parity == parity; })) plan_group(parity);
        if (in_flight.empty()) return;
        step_group(true);
    };
    // deferred admission (STRIX_DS41_SERVE_SCHED=defer): taken jobs waiting to be admitted, and the ones admitted again
    struct Deferred {
        serve::TakenJob job;
        Clock::time_point taken;
    };
    std::deque<Deferred> deferred, admit_ready;
    const auto device_path = [&](const std::vector<std::uint32_t>& prompt) {
        return prefill != nullptr && prompt.size() >= std::max(a.prefill_min, kMinRows);
    };
    const auto decide = [&](const std::vector<std::uint32_t>& prompt, double waited) {
        std::uint32_t decoding = 0;
        std::vector<std::uint64_t> remaining;
        for (const auto& r : running)
            if (!r->device_prefill) {
                ++decoding;
                const auto g = r->seq.generated.size();
                remaining.push_back(r->job.job.max_tokens > g ? r->job.job.max_tokens - g : 0U);
            }
        const double W = strix::deepseek41::sched::wait_estimate(remaining, step_ema > 0.0 ? step_ema : 0.06);
        const auto d = strix::deepseek41::sched::admit(sched, prompt.size(), decoding, waited, W, state.draining.load());
        using strix::deepseek41::sched::Admit;
        if (sched.log && (d == Admit::defer ? waited == 0.0 : (d != Admit::idle || waited > 0.0)))
            std::fprintf(stderr, "sched: %s request of %zu tokens after %.3f s: %u decoding, cost N x T_dev %.2f s vs W %.2f s\n",
                         strix::deepseek41::sched::admit_name(d), prompt.size(), waited, decoding,
                         decoding * strix::deepseek41::sched::tdev_seconds(sched, prompt.size()), W);
        return d;
    };
    std::uint64_t session_hits = 0, session_misses = 0, session_refusals = 0;
    std::string failure;
    try {
        for (;;) {
            if (sessions) {
                std::deque<std::shared_ptr<SessionJob>> jobs;
                {
                    const std::lock_guard lock(session_jobs.mutex);
                    jobs.swap(session_jobs.jobs);
                }
                for (auto &j : jobs) {
                    auto out = serve::Json::object();
                    const auto refused_before = sessions->stats().refused;
                    int status = 200;
                    try {
                        const auto op = j->message.find("op")->as_string();
                        if (op == "list") {
                            std::map<std::string, serve::Json> rows;
                            for (const auto &id : sessions->list()) {
                                if (std::any_of(running.begin(), running.end(), [&](const auto& r) { return r->job.job.session_id == id; })) continue;
                                const auto info = sessions->inspect(id);
                                const auto &c = info.checkpoint;
                                auto row = serve::Json::object();
                                row.set("id", serve::Json(id));
                                row.set("position", serve::Json(c.frontier.consumed));
                                row.set("consumed", serve::Json(c.frontier.consumed));
                                row.set("pending", c.frontier.pending_valid ? serve::Json(static_cast<std::uint64_t>(c.frontier.pending)) : serve::Json());
                                row.set("bytes", serve::Json(info.bytes));
                                row.set("emitted", serve::Json(c.frontier.emitted));
                                row.set("state", serve::Json(info.ram ? "parked" : "disk"));
                                row.set("durability", serve::Json(info.disk ? "disk" : "ram"));
                                auto engine = serve::Json::object(); engine.set("generation", serve::Json(c.generation));
                                row.set("engine", std::move(engine)); rows.insert_or_assign(id, std::move(row));
                            }
                            for (const auto& p : parked)
                                serve::session_inventory_live(rows, p.session_id, p.position, p.next_token, p.session_emitted, false);
                            for (const auto& r : running) {
                                const auto position = active && active->r == r.get() ? active->task->rows() : r->seq.position;
                                const auto pending = r->seq.generated.empty() ? std::optional<std::uint32_t>{} : r->seq.generated.back();
                                serve::session_inventory_live(rows, r->job.job.session_id, position, pending,
                                    r->session_emitted + r->seq.generated.size(), true);
                            }
                            auto ids = serve::Json::array();
                            for (auto& [id, row] : rows) ids.push(std::move(row));
                            out.set("sessions", std::move(ids));
                        } else if (op != "stats") {
                            const auto id = j->message.find("id")->as_string();
                            if (std::any_of(running.begin(), running.end(), [&](const auto &r) { return r->job.job.session_id == id; }))
                                throw std::runtime_error("session has an active turn");
                            const auto names = sessions->list();
                            if (std::find(names.begin(), names.end(), id) == names.end())
                                throw std::runtime_error("session_not_found");
                            const bool restore = op == "resume";
                            auto command = j->message;
                            if (restore) {
                                // Explicit resume admits a validated host checkpoint. Device import
                                // waits for the next named request, so parked sessions hold no
                                // extra device-resident copy outside the RAM budget.
                                command.set("op", serve::Json(sessions->inspect(id).ram ? "ram_checkpoint" : "host_restore"));
                            } else if (op != "delete" && !sessions->inspect(id).ram) {
                                auto load = serve::Json::object(); load.set("id", serve::Json(id));
                                load.set("op", serve::Json("host_restore")); session_call(load, 0);
                            }
                            session_call(command, 0);
                            if (restore) {
                                const auto info = sessions->inspect(id);
                                auto engine = serve::Json::object();
                                engine.set("position", serve::Json(info.checkpoint.frontier.consumed));
                                engine.set("generation", serve::Json(info.checkpoint.generation));
                                out.set("engine", std::move(engine));
                            }
                            if (op == "park" || op == "delete") {
                                for (auto it = parked.begin(); it != parked.end();) {
                                    if (it->session_id == id) {
                                        pool_release(it->slot);
                                        free_slots.push_back(it->slot);
                                        it = parked.erase(it);
                                    } else
                                        ++it;
                                }
                            }
                            out.set("id", serve::Json(id));
                            out.set(op == "delete" ? "deleted" : "parked", serve::Json(true));
                        }
                        if (op == "stats") {
                            const auto st = sessions->stats();
                            out.set("ram_bytes", serve::Json(st.bytes));
                            out.set("ram_budget", serve::Json(a.session_ram_bytes));
                            out.set("parked", serve::Json(st.entries));
                            out.set("disk_bytes", serve::Json(sessions->disk_bytes()));
                            out.set("hits", serve::Json(session_hits)); out.set("misses", serve::Json(session_misses));
                            out.set("refusals", serve::Json(session_refusals + st.refused));
                        }
                    } catch (const std::exception &e) {
                        if (sessions->stats().refused == refused_before) ++session_refusals;
                        const auto error = serve::session_failure(e.what());
                        status = error.status; out = serve::session_error(error.code, e.what());
                    }
                    j->done.set_value({status, out.dump()});
                }
            }
            {  // imported states, between steps
                std::deque<std::shared_ptr<ImportJob>> jobs;
                {
                    const std::lock_guard lock(imports.mutex);
                    jobs.swap(imports.jobs);
                }
                if (hetero_pipeline && !jobs.empty()) hetero_flush();
                for (auto& j : jobs) {
                    try {
                        import_one(*j);
                    } catch (...) {
                        ImportResult res;
                        res.status = 500;
                        res.error = "the import failed on the engine";
                        j->done.set_value(res);
                        throw;
                    }
                }
            }
            const bool stopping = stop_requested().load();
            if (hetero_pipeline && (stopping || state.draining.load())) hetero_flush();
            if (stopping && !state.draining.load()) config.on_drain();
            if (state.draining.load())  // deferred jobs end with the drain (as the ones waiting for pool room)
                for (auto* d : {&deferred, &admit_ready})
                    for (; !d->empty(); d->pop_front()) {
                        serve::GenerationStats st;
                        st.prompt_tokens = d->front().job.job.prompt.size();
                        st.finish = "cancelled";
                        queue.finish(d->front().job, st, "draining");
                    }
            if (stopping && running.empty()) break;  // SIGINT/SIGTERM: drained
            if (sched.mode == strix::deepseek41::sched::Mode::defer)  // deferred jobs: cancelled, continuing a parked state, or admitted now
                for (auto it = deferred.begin(); it != deferred.end();) {
                    const auto& job = it->job.job;
                    if (it->job.ticket->cancelled()) {
                        serve::GenerationStats st;
                        st.prompt_tokens = job.prompt.size();
                        st.finish = "cancelled";
                        queue.finish(it->job, st, {});
                        it = deferred.erase(it);
                        continue;
                    }
                    const bool imported = !parked.empty() && match_parked(job.prompt, job.max_tokens, job.session_id) >= 0;
                    if (imported && sched.log) std::fprintf(stderr, "sched: deferred request of %zu tokens continues an imported state\n", job.prompt.size());
                    if (imported || decide(job.prompt, std::chrono::duration<double>(Clock::now() - it->taken).count()) !=
                                        strix::deepseek41::sched::Admit::defer) {
                        admit_ready.push_back(std::move(*it));
                        it = deferred.erase(it);
                    } else {
                        ++it;
                    }
                }
            // admission between steps: oldest queued jobs into free slots
            bool imported_first = false;
            while (!state.draining.load() && (!free_slots.empty() || !parked.empty())) {
                std::optional<serve::TakenJob> job;
                bool from_queue = false;
                Clock::time_point taken = Clock::now();
                if (!pool_waiting.empty()) {
                    job = std::move(pool_waiting.front());
                    pool_waiting.pop_front();
                } else if (!admit_ready.empty()) {
                    job = std::move(admit_ready.front().job);
                    taken = admit_ready.front().taken;
                    admit_ready.pop_front();
                } else {
                    job = queue.take(false, sched.short_first);
                    from_queue = true;
                }
                if (!job) break;
                if (hetero_pipeline) hetero_flush();  // the new request may take a slot: the plans in flight first
                if (from_queue) taken = Clock::now();
                auto r = std::make_unique<Running>();
                r->job = std::move(*job);
                r->start = taken;
                const auto& prompt = r->job.job.prompt;
                if (prompt.empty() || prompt.size() + 1U >= a.context) {
                    serve::GenerationStats st;
                    st.prompt_tokens = prompt.size();
                    st.finish = "context";
                    queue.finish(r->job, st, "prompt is empty or fills the context");
                    continue;
                }
                if (!r->job.job.session_id.empty()) {
                    const auto &id = r->job.job.session_id;
                    if (std::any_of(running.begin(), running.end(), [&](const auto &run) { return run->job.job.session_id == id; })) {
                        serve::GenerationStats st; st.finish = "cancelled";
                        st.error_status = 409; st.error_code = "session_incompatible";
                        queue.finish(r->job, st, "session has an active turn"); continue;
                    }
                    if (match_parked(prompt, r->job.job.max_tokens, id) < 0) {
                        // A changed transcript is a normal miss. Release stale staged slots.
                        for (auto it = parked.begin(); it != parked.end();) {
                            if (it->session_id == id) { pool_release(it->slot); free_slots.push_back(it->slot); it = parked.erase(it); }
                            else ++it;
                        }
                        bool reserved = false;
                        std::uint32_t slot = 0;
                        try {
                            const auto info = sessions->inspect(id);
                            auto command = serve::Json::object(); command.set("id", serve::Json(id));
                            command.set("op", serve::Json(info.ram ? "lookup" : "lookup_disk"));
                            const auto c = session_call(command, 0);
                            if (strix::deepseek41::session_prompt_extends(*c, prompt) && !free_slots.empty()) {
                                slot = free_slots.back();
                                if (a.pool && !fwd.reserve(slot, a.context)) throw std::runtime_error("session_budget");
                                reserved = true;
                                if (a.pool) pool_ops.push_back(kPoolReserve << 30U | slot << 24U | a.context);
                                command.set("op", serve::Json(info.ram ? "resume" : "restore"));
                                const auto restored = session_call(command, slot);
                                free_slots.pop_back(); pending_resets &= ~(1U << slot);
                                if (prefill) prefill->on_prefill(slot, restored->attention.tokens);
                                Parked p; p.slot = slot; p.position = restored->attention.position;
                                p.next_token = restored->attention.next_token; p.tokens = restored->attention.tokens;
                                p.session_id = id; p.session_emitted = restored->frontier.emitted;
                                parked.push_back(std::move(p)); reserved = false;
                            }
                        } catch (const std::exception &) {
                            if (reserved) pool_release(slot);
                            // Explicit resume reports validation errors; chat lookup falls back to prefill.
                        }
                    }
                    if (match_parked(prompt, r->job.job.max_tokens, id) >= 0) ++session_hits;
                    else ++session_misses;
                }
                if (const auto m = parked.empty() ? -1 : match_parked(prompt, r->job.job.max_tokens, r->job.job.session_id); m >= 0) {  // continues an imported state
                    const auto p = parked[static_cast<std::size_t>(m)];
                    parked.erase(parked.begin() + m);
                    r->seq.slot = p.slot;
                    r->session_emitted = p.session_emitted;
                    r->seq.prompt = prompt;
                    r->seq.position = p.position;
                    r->seq.fed = p.position;
                    r->account.continued(p.position);  // usage.prompt_tokens_details.cached_tokens reports them
                    if (prompt.size() == p.position) {
                        r->seq.generated.assign(1U, p.next_token);
                        imported_first = true;
                    }
                    std::fprintf(stderr, "import: request of %zu prompt tokens continues the state in slot %u (position %u)\n", prompt.size(), p.slot,
                                 p.position);
                    if (!r->job.job.session_id.empty() && r->job.job.on_admitted) r->job.job.on_admitted();
                    running.push_back(std::move(r));
                    worked = true;
                    const std::lock_guard lock(state.mutex);
                    state.busy = static_cast<std::uint32_t>(running.size());
                    ++state.admitted;
                    continue;
                }
                if (sched.mode == strix::deepseek41::sched::Mode::defer && from_queue && device_path(prompt) &&
                    decide(prompt, 0.0) == strix::deepseek41::sched::Admit::defer) {  // its prefill would stall the decoding requests: wait
                    deferred.push_back({std::move(r->job), taken});
                    continue;
                }
                if (free_slots.empty()) unpark_oldest();  // a parked state gives its slot to a request that needs one
                if (a.pool != 0U) {  // room for the prompt and every token it may generate, or wait (FIFO)
                    const auto want = static_cast<std::uint32_t>(std::min<std::uint64_t>(a.context, prompt.size() + std::uint64_t{r->job.job.max_tokens} + 1U));
                    if (!fwd.reserve(free_slots.back(), want)) {
                        if (want > a.pool) {  // never fits
                            serve::GenerationStats st;
                            st.prompt_tokens = prompt.size();
                            st.finish = "context";
                            queue.finish(r->job, st, "prompt + max_tokens exceed the KV pool");
                            continue;
                        }
                        pool_waiting.push_front(std::move(r->job));
                        break;
                    }
                    pool_ops.push_back(kPoolReserve << 30U | free_slots.back() << 24U | want);
                }
                r->seq.slot = free_slots.back();
                free_slots.pop_back();
                r->seq.prompt = prompt;
                r->device_prefill = device_path(prompt);
                if (spec && !r->device_prefill) spec->start(r->seq.slot, prompt);  // device prefill: imported() after it
                pending_resets |= 1U << r->seq.slot;
                if (!r->job.job.session_id.empty() && r->job.job.on_admitted) r->job.job.on_admitted();
                running.push_back(std::move(r));
                worked = true;
                const std::lock_guard lock(state.mutex);
                state.busy = static_cast<std::uint32_t>(running.size());
                ++state.admitted;
            }
            if (state.draining.load())  // taken jobs still waiting for pool room end with the drain
                for (; !pool_waiting.empty(); pool_waiting.pop_front()) {
                    serve::GenerationStats st;
                    st.prompt_tokens = pool_waiting.front().job.prompt.size();
                    st.finish = "cancelled";
                    queue.finish(pool_waiting.front(), st, "draining");
                }
            if (imported_first) publish();  // the first token of a request that continues an imported state is known
            // cancelled requests leave before the step
            if (hetero_pipeline && !in_flight.empty())
                for (const auto& r : running)
                    if (r->job.ticket->cancelled()) {
                        hetero_flush();
                        break;
                    }
            for (std::size_t i = running.size(); i-- > 0U;)
                if (running[i]->job.ticket->cancelled()) {
                    if (active && active->r == running[i].get()) abort_task();
                    finish(i, "cancelled", {});
                }
            if (sched.mode == strix::deepseek41::sched::Mode::interleave && prefill) {
                // a request waiting for its prefill whose state was imported meanwhile (e.g. prefilled on another device)
                bool swapped = false;
                if (!parked.empty())
                    for (auto& r : running)
                        if (r->device_prefill)
                            if (const auto m = match_parked(r->seq.prompt, r->job.job.max_tokens, r->job.job.session_id); m >= 0) {
                                take_parked(*r, static_cast<std::size_t>(m));
                                swapped = true;
                            }
                if (swapped) publish();
                if (!active) {
                    std::vector<Running*> waiting;
                    for (auto& r : running)
                        if (r->device_prefill) waiting.push_back(r.get());
                    std::stable_sort(waiting.begin(), waiting.end(), [](const auto* a0, const auto* b0) {
                        return a0->job.submitted < b0->job.submitted;
                    });
                    if (!waiting.empty()) {
                        const auto pick = serve::select_waiting(waiting, sched.short_first,
                            [](const auto* r) { return r->seq.prompt.size(); },
                            [](auto* r) -> std::uint32_t& { return r->prefill_overtakes; });
                        auto& r = *waiting[pick];
                        device_prefill(r, r.seq.prompt.size() >= sched.min_rows);
                        if (!active) publish();
                    }
                }
                // one decode step of the requests that decode (or read their prompt in rows), then one unit of the prefill
                std::vector<BatchSequence*> seqs;
                for (auto& r : running)
                    if (!r->device_prefill) seqs.push_back(&r->seq);
                if (!seqs.empty()) {
                    decode_step(seqs);
                    if (active) ++active->steps_between;
                }
                if (active) {
                    // phase B (resumable): the default pieces, a few units per tick; else pieces of unit_rows rows
                    run_unit(strix::deepseek41::sched::row_budget(sched, seqs.empty()), !seqs.empty());
                    if (!active) publish();  // its first token
                    continue;
                }
                if (!running.empty()) continue;
            } else {
                // device prefills of the admitted prompts, then their first tokens
                bool prefilled = false;
                for (auto& r : running)
                    if (r->device_prefill) {
                        device_prefill(*r, false);
                        prefilled = true;
                    }
                if (prefilled) {
                    publish();
                    continue;  // admit again, then step the requests that are decoding
                }
            }
            if (running.empty()) {
                if (worked && comm_direct) {  // going idle: both ranks consume an early-posted GPU receive (driver DMA deadline)
                    Control q;
                    q.command = static_cast<std::uint32_t>(Command::quiesce);
                    q.snaps = pending_snaps;
                    pending_snaps = 0;
                    take_commits(q);
                    (void)exchange(fwd, q);
                    fwd.quiesce();
                    dog.progress();
                    last_round = Clock::now();
                    worked = false;
                }
                if (std::chrono::duration<double>(Clock::now() - last_round).count() > 1.0) {  // keep rank 1 (and the link) alive
                    Control hb;
                    hb.command = static_cast<std::uint32_t>(Command::heartbeat);
                    hb.snaps = pending_snaps;
                    pending_snaps = 0;
                    take_commits(hb);
                    (void)exchange(fwd, hb);
                    dog.progress();
                    last_round = Clock::now();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            std::vector<BatchSequence*> seqs;
            for (auto& r : running) seqs.push_back(&r->seq);
            decode_step(seqs);
        }
    } catch (const std::exception& e) {
        failure = e.what();
    }
    if (!failure.empty()) {
        state.failed.store(true);
        api.set_state("failed");
        while (!running.empty()) finish(running.size() - 1U, "length", "engine failure: " + failure);
    } else {
        if (prof.steps != 0U) prof.print(spec.get());
        Control bye;
        bye.command = static_cast<std::uint32_t>(Command::shutdown);
        take_commits(bye);
        try {
            (void)exchange(fwd, bye);
        } catch (const std::exception& e) {
            failure = std::string("shutdown of rank 1: ") + e.what();
        }
    }
    server.stop();
    queue.shutdown();
    http_thread.join();
    if (!failure.empty()) poisoned(0U, failure);
}

// usb4gpu: the device exchange (GPU-direct streams) built next to the host-path control transport (null: host path).
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
    if (kind == "usb4auto") {
        strix::dist::Usb4AutoOptions o;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        std::fprintf(stderr, "usb4 topology: %s\n", topology.receipt().c_str());
        return t;
    }
    if (kind == "usb4gpu") {  // GPU-direct exchanges over the gpu_direct streams, control over the host-path ones
        strix::dist::Usb4AutoOptions o;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        std::fprintf(stderr, "usb4 topology: %s\n", topology.receipt().c_str());
        std::string why;
        if (strix::dist::usb4_gpu_direct_streams(o, *t, &why) == 0U) {  // both ranks agree from the same exchanged counts
            std::fprintf(stderr, "gpu-direct unavailable (%s): host path only\n", why.c_str());
            return t;
        }
        std::string receipt;
        gpu_direct = strix::dist::usb4_gpu_direct(o, topology, *t, {}, &receipt);
        std::fprintf(stderr, "usb4 %s\n", receipt.c_str());
        return t;
    }
    throw std::invalid_argument("unknown or unavailable transport " + spec);
}

// --prefill-rows auto (the pair default): the largest piece of 16384 / 8192 / 4096 rows (at most the context) that both
// ranks admit after the batch forward's weights (sched::auto_prefill_rows: the piece's scratch, the prefill's own weights
// from the tensor inventory, the drafter's peak on a drafting rank and a margin, against min(device free, MemAvailable)),
// else 2048 when that fits; both ranks take the smaller choice, and when either rank fits no piece both refuse before
// READY with the numbers (an explicit --prefill-rows N skips the check). Emulation (both ranks in one process, one device) keeps 2048. OFF: an
// explicit --prefill-rows N. Pair, context 49152, fresh prompts: 16384 vs 2048 rows, 8,399 tokens 14.57 -> 10.34 s,
// 32,499 tokens 60.8 -> 39.1 s, the same logits and tokens.
std::uint32_t auto_prefill_rows(const Args& a, BatchForward& fwd, strix::dist::Communicator& comm, std::uint32_t rank) {
    namespace sc = strix::deepseek41::sched;
    std::uint32_t rows = std::max<std::uint32_t>(32U, std::min(sc::kAutoPrefillFallback, a.context / 8U * 8U));
    sc::AutoPrefillRoom room;
    if (a.transport != "emulate") {
        room.device_free = fwd.device_memory().first;
        room.mem_available = mem_available_mb() << 20U;
        auto o = slot_prefill_options(a, fwd, comm, rank);
        o.max_rows = std::min(sc::kAutoPrefillRows[0], a.context);
        room.own_weights = strix::deepseek41::DevicePrefill::owned_weight_bytes(a.model, o);
        if (!a.dspark.empty() && (rank == 0U || a.dspark_ranks == 2U)) room.drafter = sc::drafter_peak_bytes(std::filesystem::file_size(a.dspark));
        rows = sc::auto_prefill_rows(a.context, room);
    }
    std::array<std::uint32_t, 1> mine{rows}, peer{};
    fwd.exchange_host(std::as_bytes(std::span(mine)), std::as_writable_bytes(std::span(peer)));
    const std::uint32_t agreed = std::min(rows, peer[0]);
    const double g = 1073741824.0;
    std::fprintf(stderr, "rank %u: --prefill-rows auto -> %u (this rank %u: device free %.1f, MemAvailable %.1f GiB; reserved own weights %.2f, "
                 "drafter %.1f, margin %.1f GiB)%s\n", rank, agreed, rows, static_cast<double>(room.device_free) / g,
                 static_cast<double>(room.mem_available) / g, static_cast<double>(room.own_weights) / g, static_cast<double>(room.drafter) / g,
                 static_cast<double>(room.margin) / g, a.transport == "emulate" ? " [emulate: fixed]" : "");
    if (agreed == 0U)  // both ranks see the same min: both refuse here, before the drafter loads and before READY
        throw std::runtime_error("--prefill-rows auto: no prefill piece fits on " + std::string(rows == 0U ? "this rank" : "the peer rank") +
                                 " (even " + std::to_string(std::max<std::uint32_t>(32U, std::min(sc::kAutoPrefillFallback, a.context / 8U * 8U))) +
                                 " rows need more than min(device free, MemAvailable) less the reserved weights, drafter and margin above);"
                                 " free memory, lower --context, or pass an explicit --prefill-rows N");
    return agreed;
}

void run_rank(Args a, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport) {
    // the device prefill exchanges a chunk's MoE half sums (<= 21 MB) in one message
    strix::dist::Communicator comm(strix::dist::Group{rank, 2U, a.epoch, fnv(a.group_id)}, std::move(transport), std::move(gpu_direct),
                                   a.device_prefill ? std::size_t{22} << 20U : std::size_t{4} << 20U);
    std::fprintf(stderr, "rank %u: dist %s\n", rank, comm.transport_name().c_str());
    strix::profile::check_peer(strix::deepseek41::pair_defaults::profile_resolution(), [&](std::span<const std::byte> profile_out, std::span<std::byte> profile_in) {
        comm.exchange_host(profile_out, profile_in, strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0x70726f66U, 0U});
    });  // both ranks must run the same product profile
    Watchdog dog(rank, a.peer_timeout);
    dog.pause(true);  // loading the weights is not a round
    BatchForward fwd(a.model, forward_options(a, &comm, rank));
    std::fprintf(stderr, "rank %u: weights %.2f GiB loaded in %.1f s, %u slots of %u positions (%.1f MiB state each), workspace %.1f MiB\n", rank,
                 static_cast<double>(fwd.weight_bytes()) / 1073741824.0, fwd.load_seconds(), a.slots, a.context,
                 static_cast<double>(fwd.state_bytes_per_slot()) / 1048576.0, static_cast<double>(fwd.workspace_bytes()) / 1048576.0);
    if (const auto id = fwd.trellis_identity(); !id.empty())
        std::fprintf(stderr, "rank %u: trellis %s, %.1f MiB (+ %.1f MiB library workspaces)\n", rank, id.c_str(),
                     static_cast<double>(fwd.trellis_bytes()) / 1048576.0, static_cast<double>(fwd.trellis_library_bytes()) / 1048576.0);
    std::unique_ptr<SlotPrefill> prefill;
    if (a.device_prefill && a.prefill_rows_auto) a.prefill_rows = auto_prefill_rows(a, fwd, comm, rank);
    if (a.device_prefill) {
        const auto before = fwd.device_memory().first;
        prefill = std::make_unique<SlotPrefill>(a, fwd, comm, rank);
        std::fprintf(stderr, "rank %u: device prefill of %u-row pieces: device free %.2f GiB lower (scratch estimate %.2f GiB + own weights %.2f GiB)\n", rank, a.prefill_rows,
                     (static_cast<double>(before) - static_cast<double>(fwd.device_memory().first)) / 1073741824.0,
                     static_cast<double>(strix::deepseek41::sched::prefill_piece_bytes(a.prefill_rows, a.context)) / 1073741824.0,
                     static_cast<double>(prefill->weight_bytes()) / 1073741824.0);
    }
    if (prefill) prefill->set_moe_chunk_rows(strix::deepseek41::sched::options_from_env(true).moe_chunk_rows);  // phase B tasks (both ranks)
    dog.pause(false);
    std::unique_ptr<strix::ds41::hetero::Stage> stage;
    std::uint64_t hetero_exchanges = 0;
    if (!a.hetero_listen.empty()) {
        strix::ds41::hetero::StageOptions so;
        so.cut = a.hetero_cut;
        so.rank = rank;
        so.world = 2U;
        so.listen = a.hetero_listen;
        so.token_file = a.hetero_token;
        so.slots = a.slots;
        so.engine = "strix-deepseek41-serve";
        stage = std::make_unique<strix::ds41::hetero::Stage>(fwd, so, [&comm, &hetero_exchanges](std::span<const std::byte> out, std::span<std::byte> in) {
            comm.exchange_host(out, in, strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0x4842U, hetero_exchanges++});
        });
        dog.pause(true);  // the FRONT may still be loading
        stage->connect();
        if (rank == 0U && stage->front_positions() < a.context)
            throw std::invalid_argument("--hetero-listen: the FRONT holds " + std::to_string(stage->front_positions()) + " positions per slot, --context is " +
                                        std::to_string(a.context));
        dog.pause(false);
    }
    if (rank == 0U) lead(a, fwd, comm, dog, prefill.get(), stage.get());
    else follow(a, fwd, dog, prefill.get(), stage.get());
    if (stage) stage->close();
    std::fprintf(stderr, "rank %u: %s\n", rank, plans_since_ready().c_str());
    serve::Json stopped = serve::Json::object();
    stopped.set("event", serve::Json("STOPPED"));
    stopped.set("rank", serve::Json(static_cast<std::uint64_t>(rank)));
    event(stopped);
}

void describe(const Args& a) {
    namespace fs = std::filesystem;
    serve::Json out = serve::Json::object();
    out.set("schema", serve::Json("strix/engine-capabilities-v1"));
    out.set("engine", serve::Json("strix-deepseek41-serve"));
    out.set("executable_sha256", serve::Json(executable_sha256()));
    serve::Json model = serve::Json::object();
    model.set("id", serve::Json(a.model_id));
    model.set("architecture", serve::Json("deepseek41"));
    model.set("file", serve::Json(fs::path(a.model).filename().string()));
    model.set("bytes", serve::Json(static_cast<std::uint64_t>(fs::file_size(a.model))));
    out.set("model", std::move(model));
    serve::Json modes = serve::Json::array();
    serve::Json mode = serve::Json::object();
    mode.set("name", serve::Json(a.tp ? "ep2-tp" : "ep2"));
    mode.set("world", serve::Json(2));
    mode.set("numerical_profile", serve::Json("ds41-exact-v1 (A8 x Q8_0 sites, K-split wo_b, canonical (shared + low) + high)"));
    if (const auto ns = strix::ds41::trellis_weights_namespace(); !ns.empty()) mode.set("trellis", serve::Json(ns));
    mode.set("transports", serve::Json(serve::Json::Array{serve::Json("usb4_host_auto"), serve::Json("usb4_gpu_auto")}));
    serve::Json ranks = serve::Json::array();
    // --dspark: the drafter on rank 0, and on rank 1 too with --dspark-ranks 2 (weights, scratch, a.slots contexts, staging)
    std::shared_ptr<const strix::ds41::dspark::LoadedWeights> dw;
    if (!a.dspark.empty()) dw = strix::ds41::dspark::load_weights(a.model, a.dspark);
    for (std::uint32_t r = 0; r < 2U; ++r) {
        auto o = forward_options(a, nullptr, r);
        const auto f = BatchForward::footprint(a.model, o);
        serve::Json mem = serve::Json::object();
        mem.set("rank", serve::Json(static_cast<std::uint64_t>(r)));
        mem.set("resident_weights_bytes", serve::Json(f.weights));
        mem.set("state_bytes_per_slot", serve::Json(f.state_per_slot));
        mem.set("workspace_bytes", serve::Json(f.workspace));
        mem.set("kv_pool_bytes", serve::Json(f.pool));
        if (f.trellis != 0U) mem.set("trellis_bytes", serve::Json(f.trellis));  // + the library workspaces (queried at load)
        std::uint64_t dspark = 0;
        if (dw && (r == 0U || a.dspark_ranks == 2U)) {
            const auto df = strix::ds41::dspark::DeviceDrafter::footprint(dw->weights, a.slots);
            mem.set("dspark_weights_bytes", serve::Json(df.weights));
            mem.set("dspark_scratch_bytes", serve::Json(df.scratch));
            mem.set("dspark_context_bytes", serve::Json(df.contexts));
            mem.set("dspark_staging_bytes", serve::Json(strix::deepseek41::BatchSpeculator::staging_bytes(a.slots)));
            dspark = dspark_footprint_bytes(dw->weights, a);
        }
        // the target (BatchForward::footprint) + the drafter; the trellis library workspaces and hipMalloc rounding are not in it
        mem.set("total_bytes", serve::Json(f.weights + f.state_per_slot * a.slots + f.workspace + f.pool + f.trellis + dspark));
        ranks.push(std::move(mem));
    }
    mode.set("memory", std::move(ranks));
    mode.set("context_tokens", serve::Json(static_cast<std::uint64_t>(a.context)));
    mode.set("slots", serve::Json(static_cast<std::uint64_t>(a.slots)));
    mode.set("max_slots", serve::Json(static_cast<std::uint64_t>(BatchForward::configured_row_limit())));
    modes.push(std::move(mode));
    out.set("modes", std::move(modes));
    serve::Json api = serve::Json::object();
    api.set("openai", serve::Json(true));
    api.set("anthropic", serve::Json(true));
    api.set("sampling", serve::Json(serve::Json::Array{serve::Json("greedy")}));
    api.set("endpoints", serve::Json(serve::Json::Array{serve::Json("/v1/chat/completions"), serve::Json("/v1/completions"), serve::Json("/v1/messages"),
                                                        serve::Json("/v1/models"), serve::Json("/health"), serve::Json("/metrics"),
                                                        serve::Json("/api/status"), serve::Json("/admin/drain"), serve::Json("/tokenize"),
                                                        serve::Json("/detokenize")}));
    out.set("api", std::move(api));
    out.set("argv_template", serve::Json(serve::Json::Array{serve::Json("{executable}"), serve::Json("{model}"), serve::Json("--group-manifest"),
                                                            serve::Json("{manifest}")}));
    out.set("events", serve::Json(serve::Json::Array{serve::Json("READY"), serve::Json("POISONED"), serve::Json("STOPPED")}));
    std::printf("%s\n", out.dump().c_str());
}

}  // namespace

int main(int argc, char** argv) try {
    bool session_explicit = std::getenv("STRIX_SESSION_CACHE") && std::string_view(std::getenv("STRIX_SESSION_CACHE")) != "0";
    for (int i = 1; i < argc; ++i)
        if (std::string_view(argv[i]).starts_with("--session-")) session_explicit = true;
    const strix::deepseek41::pair_defaults::Effective production_defaults(strix::deepseek41::pair_defaults::App::serve, argc, argv, strix::deepseek41::pair_defaults::this_build());  // before any option or env read
    // The dist layer's own deadlines below the serve watchdog (120 s): a rare stall ends in the communicator's state
    // report (collective in flight, phase, per-stream bytes) instead of a bare POISONED. An explicit value wins.
    (void)::setenv("STRIX_DIST_USB4_TIMEOUT_MS", "60000", 0);
    (void)::setenv("STRIX_DIST_WAIT_TIMEOUT_MS", "60000", 0);
    Args a;
    a.session_explicit = session_explicit;
    bool tp_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        const auto u32 = [&] { return static_cast<std::uint32_t>(std::stoul(next())); };
        if (s == "--rank") a.rank = u32();
        else if (s == "--transport") a.transport = next();
        else if (s == "--group-manifest") a.manifest = next();
        else if (s == "--layers") a.layers = u32();
        else if (s == "--context") a.context = u32();
        else if (s == "--slots") a.slots = u32();
        else if (s == "--pool-positions") a.pool = u32();
        else if (s == "--prompt-rows") a.prompt_rows = u32();
        else if (s == "--dspark") a.dspark = next();
        else if (s == "--no-dspark") a.dspark.clear();
        else if (s == "--dspark-window") a.dspark_window = next();
        else if (s == "--dspark-vocab") a.dspark_vocab = next();
        else if (s == "--dspark-ranks") a.dspark_ranks = u32();
        else if (s == "--dspark-max-inflight") a.dspark_max_inflight = u32();
        else if (s == "--dspark-seed-rows") a.dspark_seed_rows = u32();
        else if (s == "--hetero-listen") a.hetero_listen = next();
        else if (s == "--hetero-token-file") a.hetero_token = next();
        else if (s == "--hetero-cut") a.hetero_cut = u32();
        else if (s == "--hetero-groups") {
            a.hetero_groups = u32();
            if (a.hetero_groups != 1U && a.hetero_groups != 2U) throw std::invalid_argument("--hetero-groups is 1 or 2");
        }
        else if (s == "--tp-attention") a.tp = tp_set = true;
        else if (s == "--no-tp-attention") a.tp = false, tp_set = true;
        else if (s == "--engram-direct") a.engram_direct = true;
        else if (s == "--engram-mapped") a.engram_direct = false;
        else if (s == "--http-bind") a.http_bind = next();
        else if (s == "--http-port") a.http_port = u32();
        else if (s == "--http-api-key-file") a.api_key_file = next();
        else if (s == "--http-max-queue") a.max_queue = u32();
        else if (s == "--http-max-connections") a.max_connections = u32();
        else if (s == "--http-responses" || s == "--http-capabilities" || s == "--http-stream-receipts") {
            const auto v = next();
            if (v != "on" && v != "off") throw std::invalid_argument(s + " requires on|off");
            (s == "--http-responses" ? a.http_responses : s == "--http-capabilities" ? a.http_capabilities : a.http_stream_receipts) = v == "on";
        }
        else if (s == "--http-sse-progress-interval") {
            const auto v = next();
            std::size_t used = 0;
            a.http_sse_progress_interval = std::stod(v, &used);
            if (used != v.size() || !std::isfinite(a.http_sse_progress_interval) || a.http_sse_progress_interval < 0.0)
                throw std::invalid_argument("--http-sse-progress-interval needs seconds >= 0");
        }
        else if (s == "--http-sse-keepalive") {
            const auto v = next();
            std::size_t used = 0;
            a.http_sse_keepalive = std::stod(v, &used);
            if (used != v.size() || !std::isfinite(a.http_sse_keepalive) || a.http_sse_keepalive < 0.0)
                throw std::invalid_argument("--http-sse-keepalive needs seconds >= 0");
        }
        else if (s == "--model-id") a.model_id = next();
        else if (s == "--default-max-tokens") a.default_max_tokens = u32();
        else if (s == "--import-parked") a.import_parked = u32();
        else if (s == "--import-frames") a.import_frames_port = u32();
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--peer-timeout") a.peer_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--describe-json") a.describe_parts = true;
        else if (s == "--describe-capabilities-json") a.describe = true;
        else if (s == "--prefill-path") {
            const auto v = next();
            if (v != "rows" && v != "device") throw std::invalid_argument("--prefill-path is rows or device");
            a.device_prefill = v == "device";
        }
        else if (s == "--prefill-rows") {
            const auto v = next();
            a.prefill_rows_auto = v == "auto";
            if (!a.prefill_rows_auto) a.prefill_rows = static_cast<std::uint32_t>(std::stoul(v));
        }
        else if (s == "--prefill-min") a.prefill_min = u32();
        else if (s == "--session-cache-dir")
            a.session_dir = next();
        else if (s == "--session-cache-mb")
            a.session_ram_bytes = std::stoull(next()) << 20U;
        else if (s == "--session-ram-bytes") a.session_ram_bytes = std::stoull(next());
        else if (s == "--session-max-parked") a.session_max_parked = std::stoull(next());
        else if (s == "--session-disk-bytes") a.session_disk_bytes = std::stoull(next());
        else if (a.parse(s, strix::deepseek41::ServeCachePolicy::takes_value(s) ? next() : std::string{})) {}
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    (void)tp_set;
    if (a.model.empty()) throw std::invalid_argument("usage: strix-deepseek41-serve MODEL.gguf --rank R --transport T | --group-manifest FILE | --describe-json");
    if (!a.manifest.empty()) apply_manifest(a);
    if (a.session_ram_bytes && a.session_ram_bytes < (16ULL << 20U) + std::uint64_t{a.context} * 2048ULL) {
        a.session_ram_bytes = 0; a.session_dir.clear();
        std::fprintf(stderr, "session cache: off (effective manifest context exceeds RAM budget)\n");
    }
    // The device prefill runs the expert-TP identity too (d647d6d4) and the cache namespace records the decode identity
    // (prefix_namespace_from_env), so the agent profile applies with STRIX_DS41_EXPERT_TP=1 as well.
    apply_cache_profile(a);
    std::fprintf(stderr, "prefix policy: profile %s, host %llu MiB/rank, block %u, reserve %llu MiB after predicted allocations, inrun %u reach %u, before-tail %u, session %u, disk %s limit %s\n",
                 a.cache_profile.c_str(), static_cast<unsigned long long>(a.prefix_cache_mb), a.prefix_block,
                 static_cast<unsigned long long>(a.prefix_reserve_mb), a.prefix_inrun ? 1U : 0U, a.prefix_inrun_reach,
                 a.prefix_before_tail ? 1U : 0U, a.prefix_session ? 1U : 0U, a.prefix_dir.empty() ? "off" : "explicit",
                 a.prefix_dir.empty() ? "disabled" : a.prefix_disk_mb == 0U ? "unlimited per tier (numeric zero)" : (std::to_string(a.prefix_disk_mb) + " MiB").c_str());
    if (!a.prefix_dir.empty()) std::fprintf(stderr, "prefix disk guard: service directory quota 4096 MiB, queued payloads 128 MiB, free-space floor 16384 MiB; disk promotion off\n");
    if (a.slots == 0U || a.slots > BatchForward::configured_row_limit()) throw std::invalid_argument("--slots exceeds STRIX_DS41_BATCH_ROWS16 step limit (8, or 16 when enabled)");
    if (a.slots > strix::deepseek41::serve_observe::kMaxStepOwners)  // the per-step draft accounting covers every slot
        throw std::invalid_argument("--slots exceeds the per-step draft accounting limit (64)");
    // DSpark windows (controller, spec packet, staging) are qualified at 8 rows: not combined with ROWS16
    if (!a.dspark.empty() && BatchForward::configured_row_limit() > BatchForward::max_rows)
        throw std::invalid_argument("--dspark is not qualified with STRIX_DS41_BATCH_ROWS16=1 (DSpark windows stay at 8 rows)");
    if (a.prompt_rows == 0U || a.prompt_rows > BatchForward::max_rows) throw std::invalid_argument("--prompt-rows must be in 1..8");
    if (a.rank > 1U) throw std::invalid_argument("--rank is 0 or 1");
    if (a.prefill_rows < 32U || a.prefill_rows % 8U != 0U) throw std::invalid_argument("--prefill-rows must be a multiple of 8, at least 32");
    if (a.prefix_block == 0U || a.prefix_block % 8U != 0U) throw std::invalid_argument("--prefix-block must be a positive multiple of 8");
    if (a.prefix_cache_mb != 0U && !a.device_prefill) throw std::invalid_argument("--prefix-cache-mb needs --prefill-path device");
    (void)strix::deepseek41::sched::options_from_env(a.device_prefill);  // STRIX_DS41_SERVE_SCHED: checked before connect / load
    if (a.prefix_inrun && a.prefix_cache_mb == 0U) throw std::invalid_argument("--prefix-inrun-snapshots needs --prefix-cache-mb (or --cache-profile agent)");
    if (a.prefix_session && !a.dspark.empty())
        throw std::invalid_argument("--prefix-session-snapshots is not exact with --dspark (verify-window rows leave another state than the prefill)");
    if (a.describe_parts) {
        strix::deepseek41::EngineDescribeOptions o;
        o.engine_name = "strix-deepseek41-serve";
        o.backend = "hip";
        o.roles = {"whole"};
        o.max_positions = a.context;
        serve::Json launch = serve::Json::object();
        launch.set("whole", serve::Json(serve::Json::Array{serve::Json("{executable}"), serve::Json("{model}"), serve::Json("--group-manifest"),
                                                          serve::Json("{manifest}")}));
        o.argv_template = launch;
        o.notes = "HTTP serving on rank 0 of the ep2-tp group; --describe-capabilities-json gives the memory per rank";
        std::printf("%s\n", strix::deepseek41::describe_engine_file(a.model, o).c_str());
        return 0;
    }
    if ((!a.prefix_dir.empty() || a.prefix_disk_mb != 0U || a.prefix_before_tail || a.prefix_session) && a.prefix_cache_mb == 0U)
        throw std::invalid_argument("--prefix-cache-dir / --prefix-cache-disk-mb / --prefix-before-tail / --prefix-session-snapshots need --prefix-cache-mb");
    if (a.prefix_disk_mb != 0U && a.prefix_dir.empty()) throw std::invalid_argument("--prefix-cache-disk-mb needs --prefix-cache-dir");
    if (!a.hetero_listen.empty() && a.hetero_token.empty()) throw std::invalid_argument("--hetero-listen needs --hetero-token-file");
    if (!a.hetero_listen.empty() && a.prefix_cache_mb != 0U)
        throw std::invalid_argument("--hetero-listen does not take the prefix cache (the pair does not keep layers [0, cut) of a split request)");
    // Expert TP stays off here unless asked (STRIX_DS41_EXPERT_TP=1). With it and the device prefill, the experts load in
    // the dup layout (whole Q2_K blocks) and the device prefill runs the expert-TP identity on the batch forward's own
    // tensors (read in place, no second copy): prefill and decode give the same bits as the expert-TP decode alone.
    strix::ds41::expert_tp::set_decode_default(false);
    if (a.device_prefill && strix::ds41::expert_tp::enabled_from_env()) strix::ds41::expert_tp::set_packed(0);
    if (a.describe) {
        describe(a);
        return 0;
    }
    strix::LoaderCapture loader_capture(a.session_ram_bytes != 0);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    if (a.transport == "emulate") {  // both ranks in this process over the in-memory pair (one-node tests, few layers)
        auto [t0, t1] = strix::dist::memory_pair();
        std::exception_ptr failure;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                run_rank(a, 1U, std::move(t));
            } catch (...) {
                failure = std::current_exception();
            }
        });
        run_rank(a, 0U, std::move(t0));
        peer.join();
        if (failure) std::rethrow_exception(failure);
        return 0;
    }
    ::alarm(a.connect_timeout);
    auto transport = open_transport(a.transport, a.rank, a.connect_timeout);
    ::alarm(0);
    std::fprintf(stderr, "connected (%s)\n", transport->name().c_str());
    std::printf("connected (%s)\n", transport->name().c_str());
    std::fflush(stdout);
    run_rank(a, a.rank, std::move(transport));
    return 0;
} catch (const std::exception& e) {
    serve::Json ev = serve::Json::object();
    ev.set("event", serve::Json("POISONED"));
    ev.set("reason", serve::Json(std::string(e.what())));
    std::printf("%s\n", ev.dump().c_str());
    std::fflush(stdout);
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
