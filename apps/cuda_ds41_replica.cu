// strix-cuda-ds41-replica: the deepseek41 R1 decode as an independent replica on one CUDA device with streamed routed
// experts (design notes): every layer's dense part here (CudaPrefill::step, graphs), the routed experts
// in tierstream's device-gated arena of `--arena-gib` (least recently used out at token boundaries) and every other
// expert in its pinned host tier, missing / predicted experts copied in over PCIe by the library at each MoE site (TierSite).
//
//   strix-cuda-ds41-replica MODEL.gguf --import-state STATE --ref PAIR_LOG [--steps N] --arena-gib G [--spares N]
//       [--arena-gib auto|G (default 80)] [--arena-headroom-gib G (auto: default 1.5)] [--positions N (alias --max-positions)]
//       [--host-spares N] [--predict K] [--prefetch-layer N] [--prefetch-token N] [--pressure-spares N]
//       [--phases exact0,pressure,exact8,chats8,chats0] [--routes ROUTES [--seqmod M:K]] [--chats I,J,..
//       --chat-prompt P --chat-steps N --seqs-file SEQS] [--timing] [--json OUT] [--timeline OUT] [--verify N]
//       [--engram MODE] [--host-margin-gib G] [--fuse-hc] [--fuse-shared] [--no-lean-state]
//
// Defaults: dsrtxk, fusions, deferred records, router form 2, qakv-s1, unpadded q8-r1,
// keep-64, warm controller (chat scope), deferred WB, Engram RAM 512 MB, pipelined load.
// Off: --dsrtxk-off --fuse-off --records-inline --router-pf0 --qakv-auto --q8-r1-off
// (--q8-r1-pad: padded form), --draft-keep 0 --controller-prior none --controller-costs none
// --no-wb-defer --engram-cache-mb 0; DG_LOAD_SERIAL=1. --host-spares defaults to 48.
// Arena GiB: keep64=74.8, keep32=76.5, full=71.5, no draft=80; --arena-gib overrides.
// Vocab: auto directory from STRIX_DS41_DRAFT_VOCAB or beside draft, else full.
// Serve: --warmup-graphs 0,8, auto model.gguf.residency seed/save (STRIX_DS41_RESIDENCY override).
// --warmup-graphs none / --residency-seed none / --residency-save none disable each.
// --baseline restores old defaults at that argument; later flags override each item.
// STRIX_DS41_RTX_BASELINE=1 applies baseline before all arguments. --print-config emits
// resolved JSON and exits before CUDA, fixtures or model reads (dummy model paths allowed).
// --profile default|fast|quality (STRIX_DS41_PROFILE; strix/ds41_rtx_profile.hpp): one switch for every lever. default (= no
// --profile) is today's configuration; quality = default (no higher-precision expert pack on the RTX); fast = default +
// the qualified fast-only levers (none yet: they are listed as pending and not used). Per-lever options stay as expert
// overrides (CLI > environment > profile > builtin); --allow-pending (STRIX_DS41_PROFILE_ALLOW_PENDING=1) runs a pending
// lever that is in this build, for benchmarks only (receipts "unqualified: <rows>"). The resolution is the "profile"
// object of --print-config / the census effective_config and the run's first "profile" event.
//
// Phases, in the given order, in one loaded process (each re-imports the state):
//   exact0 / exact<K>   greedy steps from the imported state with prediction off / K predicted experts per layer:
//                       every step compared with the pair's greedy log (fed token, argmax, logits FNV); the phase fails
//                       unless all `steps` positions are in the log and all match
//   pressure            exact0 with the free-slot target lowered to --pressure-spares (demand scratch banks, mid-token
//                       write-back collection and resurrections exercised); restored afterwards
//   chats<K>            the sequences of --chats: the first P tokens fed as R1 steps, then N greedy steps (timing)
// A device-side wait timeout fails the step at once. --routes: the seed order (the most used experts of the recorded
// routes outside test fold K of M start on the device). One JSON line per step (wall ms including the token boundary);
// --timeline: one line per (step, layer). --verify N: after the run, N expert byte checks against the file.
// tierstream is the only cache (the replica's own StreamCache + StreamSite were removed after the bitwise and A/B/A
// checks); --cache tierstream is accepted for the older scripts.
//
// Smoke mode (--layers N < 40, a small device): layers [0, N) only, no head; phases smoke<K> / smokep<K> feed the
// reference log's tokens and hash each step's output stream; smokep runs under the library's pressure knob (size the
// arena small with --arena-gib for forced misses, banks, write-backs). Every smoke phase must give the first smoke
// phase's hashes bitwise.
//
// Verify windows and DSpark speculation (design notes; --max-rows R: windows of up to R rows through the stream
// site, demand banks of a whole window union):
//   wforce<K>   windows of every (rows, kept) pair sequentially: rows 0..kept-1 the reference's tokens, the rest wrong tokens;
//               every kept row compared with the reference (fed, argmax, logits FNV), commit_window(kept); ends at the
//               same position as exact<K> with the same attention-state digest
//   spec<K>     greedy speculative decoding with the DSpark drafter (--draft FILE [--draft-device N] [--draft-vocab
//               SPEC] [--window controller|fixed:N|probe]): every committed row compared with the reference, the same end
//               position and state digest as exact<K>
//   schats<K>[b<B>]  the chats with speculation: the prompt fed in windows of R rows (all committed), then speculative
//               decoding of --chat-steps tokens (timing, one JSON line per window)
//   wsmoke<K>   (smoke mode) the wforce schedule on the output streams: every kept row equals the smoke phase's hash
// Acceptance tools (window sizes and logging only, the committed tokens stay the target's greedy tokens):
//   --window probe              the drafter runs at every position and the target verifies no draft (R1 windows): with
//                               --draft-trace, every position's five drafts and confidences (an uncensored acceptance
//                               record of the PRO's own greedy text, replayable offline for any window policy)
//   --draft-trace FILE          one JSON line per drafted window: phase, pos, input token t, verified drafts k, rows,
//                               accepted a, the five drafts d, their confidences c, the target's greedy token of every row g
//   --controller-prior FILE[:CAP]  every fresh window controller starts from a stored survival table
//                               (ds41_dspark_driver.hpp read_survival_table; CAP = observations kept per depth, default 64)
//   --controller-save FILE      the controller's survival table after every speculative phase / chat
//   --controller-costs C1,..,C6[,D]  with the prior: seeded window costs in ms (verify[R] = window cycle minus the draft
//                               step, R = 1..6; D = draft step), no initial sweep, exploration every
//                               --controller-explore N decisions (default 0 = never), costs frozen unless
//                               --controller-learn (EWMA as today). Applied wherever the prior is applied.
//   --controller-prior-alt      with --controller-prior and --controller-scope chat: the prior only in occurrences 1, 2
//                               (mod 4) of each schats phase (A B B A in one process: cold, warm, warm, cold)
//   --probe-phase NAME[,NAME]   probe windows (as --window probe) only in those phases (e.g. spec0 for the state digest,
//                               schats0b0 for the record), the other phases as --window says
// The trace covers probe, fixed-window and compared (spec) phases only, never a timed controller arm; the survival table
// is saved after the phase's last window is credited (outside every timed cycle).
#include "strix/ds41_replica_census.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/cuda_ds41_dspark.hpp"
#include "strix/cuda_ds41_attention.hpp"
#include "strix/cuda_ds41_core.hpp"
#include "strix/cuda_ds41_hot_residency.hpp"
#include "strix/cuda_ds41_prefill.hpp"
#include "strix/cuda_ds41_router_v2.hpp"
#include "strix/cuda_ds41_tier_site.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/deepseek41_model.hpp"
#include "strix/ds41_dspark_driver.hpp"
#include "strix/ds41_dspark_files.hpp"
#include "strix/ds41_dspark_vocab.hpp"
#include "strix/ds41_rtx_schedule.hpp"
#include "strix/ds41_rtx_budget.hpp"
#include "strix/ds41_rtx_profile.hpp"

#include "cuda_ds41_replica_serve.hpp"
#include "cuda_ds41_session_draft.hpp"
#include "cuda_ds41_replica_budget.hpp"

#include "strix/cuda_ds41_moe_v2.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace strix::ds41::cuda;

// --serve: the options a request may change (rs::runtime_options), restored to their launch values before each request
#define PERSIST_RUNTIME_VARS \
    qual_dump, state_file, ref_file, phases_arg, chats_arg, seqs_file, json_out, timeline_out, steps, chat_prompt, chat_steps, verify_n, warm, \
        chat_logits, pressure_free, pressure_recent, pf_layer, pf_token, window_arg, controller_scope, controller_prior, \
        controller_save, controller_prior_alt, controller_costs, controller_explore, controller_learn, probe_phase, draft_vocab, \
        draft_trace, winpf_knobs, wb_alt, reset_cache, req_env, qual, wb_defer, flush_ready, controller_costs_given, evict, census

namespace {

namespace rs = strix::ds41::replica_serve;
// --serve: once the worker is loaded, a usage error belongs to the request (rc 2), never ends the worker
bool g_serving = false;

[[noreturn]] void usage(const char* why) {
    if (g_serving) throw rs::RequestReject(why);
    std::fprintf(stderr, "strix-cuda-ds41-replica: %s\n", why);
    std::fprintf(stderr, "  --weight-dirs LIST (; separated; empty disables) --drive-readers N (4) --weight-verify cached|always (cached)\n  Environment when absent: STRIX_WEIGHT_DIRS, STRIX_WEIGHT_VERIFY\n");
    std::fprintf(stderr, "  tiers: --vram-expert-gib G|auto (= --arena-gib) --ram-expert-gib G|auto (auto) --ram-inclusive-gib G|auto|0 (0)\n"
                         "         --file-tier|--no-file-tier (on; STRIX_DS41_FILE_TIER=0) --disk-prefetch --disk-staging N (8) --disk-queue N (4; STRIX_DS41_DISK_QUEUE; 0 = library default, 1 = one read at a time)\n"
                         "  whole-device budgets: --vram-total-gib N|none --ram-total-gib M|none (STRIX_DS41_VRAM_TOTAL_GIB / STRIX_DS41_RAM_TOTAL_GIB;\n"
                         "         default none = today) --vram-margin-gib G (1.5) --ram-margin-gib G (1) --budget-receipt --budget-sample-ms N (100; 0 = stages only)\n");
    std::exit(2);
}

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::istringstream is(s);
    std::string v;
    while (std::getline(is, v, c)) if (!v.empty()) out.push_back(v);
    return out;
}

struct RefStep {
    std::uint32_t pos = 0, fed = 0, argmax = 0;
    std::uint64_t fnv = 0;
};
std::map<std::uint32_t, RefStep> read_ref(const std::string& path) {
    std::ifstream in(path);
    if (!in) usage("cannot read the reference log");
    std::map<std::uint32_t, RefStep> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("pos ", 0) != 0) continue;
        std::istringstream s(line);
        std::string w;
        RefStep r;
        while (s >> w) {
            if (w == "pos") s >> r.pos;
            else if (w == "fed") s >> r.fed;
            else if (w == "argmax") s >> r.argmax;
            else if (w == "logits_fnv") s >> std::hex >> r.fnv >> std::dec;
        }
        out.emplace(r.pos, r);
    }
    return out;
}

std::string hex(std::uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}
std::uint64_t census_hash(const void* bytes, std::size_t n, std::uint64_t h = 14695981039346656037ULL) {
    const auto* p = static_cast<const unsigned char*>(bytes);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}
std::string census_file_hash(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot fingerprint census input");
    std::vector<std::byte> buf(1U << 20);
    strix::Sha256Fast hash;
    while (in) {
        in.read(reinterpret_cast<char*>(buf.data()), buf.size());
        hash.update(std::span<const std::byte>(buf.data(), static_cast<std::size_t>(in.gcount())));
    }
    if (!in.eof()) throw std::runtime_error("failed to fingerprint census input");
    return strix::model_content_identity_hex(hash.finalize());
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())))];
}
double mean(const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }

// keys (layer * E + id) by recorded use outside test fold K of M, most used first
std::vector<std::uint32_t> seed_order(const std::string& routes, const std::string& seqmod, std::uint32_t L, std::uint32_t E) {
    std::vector<double> freq(std::size_t{L} * E, 0.0);
    const auto mk = split(seqmod, ':');
    const auto M = std::stoul(mk.at(0)), Kf = std::stoul(mk.at(1));
    std::ifstream rf(routes);
    if (!rf) usage("cannot read --routes");
    std::string line;
    long seq = -1;
    while (std::getline(rf, line)) {
        if (line.rfind("# seq", 0) == 0) { ++seq; continue; }
        if (line.empty() || line[0] == '#' || (seq >= 0 && static_cast<unsigned long>(seq) % M == Kf)) continue;
        std::istringstream ls(line);
        std::string item;
        while (ls >> item) {
            const auto colon = item.find(':');
            if (colon == std::string::npos) continue;
            const auto l = static_cast<std::uint32_t>(std::stoul(item.substr(0, colon)));
            if (l >= L) continue;
            for (const auto& v : split(item.substr(colon + 1), ',')) {
                const auto e = static_cast<std::uint32_t>(std::stoul(v));
                if (e < E) freq[std::size_t{l} * E + e] += 1.0;
            }
        }
    }
    std::vector<std::uint32_t> idx(std::size_t{L} * E);
    std::iota(idx.begin(), idx.end(), 0U);
    std::stable_sort(idx.begin(), idx.end(), [&](std::uint32_t a, std::uint32_t b) { return freq[a] > freq[b]; });
    return idx;
}

// The model receipt (one full SHA-256 of this file) must bind the file being
// loaded (size, modification time) to the admitted hash of the pinned target; otherwise nothing is loaded.
std::string json_field(const std::string& j, const std::string& key) {
    const auto k = j.find("\"" + key + "\"");
    if (k == std::string::npos) return {};
    auto v = j.find(':', k);
    if (v == std::string::npos) return {};
    ++v;
    while (v < j.size() && (j[v] == ' ' || j[v] == '"')) ++v;
    auto e = v;
    while (e < j.size() && j[e] != '"' && j[e] != ',' && j[e] != '\n' && j[e] != '}') ++e;
    return j.substr(v, e - v);
}
void check_receipt(const std::string& model, const std::string& receipt) {
    std::ifstream in(receipt);
    if (!in) usage("cannot read --receipt");
    const std::string j((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto& pin = strix::deepseek41::pinned_file(strix::deepseek41::PinnedFileKind::target_q2);
    std::uint64_t size = 0, mtime_ns = 0;
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA fa{};
    if (!GetFileAttributesExA(model.c_str(), GetFileExInfoStandard, &fa)) usage("cannot stat the model file");
    size = (std::uint64_t{fa.nFileSizeHigh} << 32U) | fa.nFileSizeLow;
    const std::uint64_t ticks = (std::uint64_t{fa.ftLastWriteTime.dwHighDateTime} << 32U) | fa.ftLastWriteTime.dwLowDateTime;
    mtime_ns = (ticks - 116444736000000000ULL) * 100ULL;
#endif
    const bool ok = json_field(j, "sha256") == pin.sha256_hex && json_field(j, "admitted_sha256") == pin.sha256_hex && json_field(j, "match") == "true" &&
                    json_field(j, "bytes") == std::to_string(pin.bytes) && size == pin.bytes && json_field(j, "mtime_ns") == std::to_string(mtime_ns);
    if (!ok) usage("the model receipt does not bind this file to the admitted SHA-256 (size / mtime / hash)");
}

constexpr std::uint32_t dspark_rows() { return strix::ds41::dspark::block_size + 1U; }  // t + 5 drafts

// The rows of the next window: (rows, kept) pairs R = 2..max_rows, kept = 1..R sequentially (wforce / wsmoke).
struct WindowSchedule {
    std::uint32_t max_rows = 2, at = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
    explicit WindowSchedule(std::uint32_t m) : max_rows(m) {
        for (std::uint32_t r = 2; r <= m; ++r)
            for (std::uint32_t k = 1; k <= r; ++k) pairs.emplace_back(r, k);
    }
    std::pair<std::uint32_t, std::uint32_t> next() { return pairs[at++ % pairs.size()]; }
};

// Calls on the drafter's device (the target's device is restored afterwards).
struct DeviceScope {
    int prev = 0;
    bool changed = false;
    explicit DeviceScope(int dev) {
        STRIX_CUDA_CHECK(cudaGetDevice(&prev));
        if (dev >= 0 && dev != prev) { STRIX_CUDA_CHECK(cudaSetDevice(dev)); changed = true; }
    }
    ~DeviceScope() { if (changed) cudaSetDevice(prev); }
};

double host_available_gib() {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) return static_cast<double>(ms.ullAvailPhys) / 1073741824.0;
#endif
    return 1e9;
}

}  // namespace

// Only the pinned model identity and requested geometry are consulted; never reads the model.
static std::vector<std::uint32_t> residency_keys(const std::string& path, std::uint32_t L, std::uint32_t E) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read residency seed");
    std::string head;
    std::getline(in, head);
    const auto& pin = strix::deepseek41::pinned_file(strix::deepseek41::PinnedFileKind::target_q2);
    const std::string want = "ds41-residency-v1 sha256=" + std::string(pin.sha256_hex) + " layers=" + std::to_string(L) + " experts=" + std::to_string(E);
    if (head.rfind(want + " ", 0) != 0) throw std::runtime_error("residency seed: model / geometry mismatch");
    std::vector<std::uint32_t> keys;
    std::string tok;
    while (in >> tok) {
        std::size_t used = 0;
        const auto k = std::stoull(tok, &used);
        if (used != tok.size() || k >= std::uint64_t{L} * E) throw std::runtime_error("residency seed: invalid key");
        keys.push_back(static_cast<std::uint32_t>(k));
    }
    if (keys.empty()) throw std::runtime_error("residency seed: no keys");
    return keys;
}

static std::optional<strix::ds41::dspark::WindowController::Table> resolve_prior(const std::string& spec, std::uint32_t* cap_out = nullptr) {
    if (spec.empty() || spec == "none") return std::nullopt;
    std::string path = spec;
    std::uint32_t cap = 64;
    if (const auto colon = path.rfind(':'); colon != std::string::npos && colon > 1U && colon + 1U < path.size() &&
        path.find_first_not_of("0123456789", colon + 1U) == std::string::npos) {
        cap = static_cast<std::uint32_t>(std::stoul(path.substr(colon + 1U)));
        path.erase(colon);
    }
    if (cap_out) *cap_out = cap;
    return path == "builtin" ? strix::ds41::dspark::builtin_survival_table(cap) : strix::ds41::dspark::read_survival_table(path, cap);
}

int main(int argc, char** argv) {
    // lazy module loading + a kernel spinning on a host flag can deadlock the first launch of another kernel
#ifdef _WIN32
    _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
#endif
    if (argc < 2) usage("MODEL.gguf [options]");
    const std::string model = argv[1];
    std::string state_file, ref_file, routes, seqmod = "5:0", chats_arg, seqs_file, json_out, timeline_out, phases_arg = "exact0,pressure,exact8,chats8,chats0";
    std::uint32_t steps = 64, chat_prompt = 96, chat_steps = 128, spares = 96, host_spares = 48, predict = 8, verify_n = 0, warm = 4, pressure_spares = 8;
    std::uint32_t pf_layer = 4, pf_token = 48, layers_arg = 40, pressure_free = 2, pressure_recent = 12;
    double arena_gib = -1.0, host_margin_gib = 6.0, max_pinned_gib = 68.0;
    bool arena_auto = false;
    double arena_headroom_gib = 1.5;
    // Three-tier expert placement (rtx::tier_plan; tierstream SS_DG_DISK_TIER): --vram-expert-gib = --arena-gib;
    // --ram-expert-gib G|auto = the pinned tier (spares and staging included; auto = what the host can pin now within
    // --host-margin-gib and --max-pinned-gib); the rest of the experts stays in the model file (read on demand,
    // through --weight-dirs stripes). The file tier replaces the old refusal when RAM cannot pin every non-resident
    // expert: default ON, --no-file-tier / STRIX_DS41_FILE_TIER=0 / --baseline = refuse as before. With budgets that
    // cover every expert nothing changes (same tiers, same library path). --ram-inclusive-gib G|auto|0: pinned blocks
    // device experts keep as clean copies (eviction = a drop, no write-back); default 0 until measured.
    double ram_expert_gib = -1.0, ram_inclusive_gib = 0.0;
    const char* file_tier_env = std::getenv("STRIX_DS41_FILE_TIER");
    bool file_tier = !(file_tier_env && std::string(file_tier_env) == "0");
    bool disk_prefetch = false;       // --disk-prefetch: predictions may read file-tier experts (default: demand only)
    std::uint32_t disk_staging = 8;   // --disk-staging N: pinned staging buffers of the file reads
    // --disk-queue auto|N: file reads in flight (0 = library default, 1 = one at a time). Default auto = the library's
    // clamp(3 x read locations, 4, 8): one drive 4 (measured at 24+32 on tierstream's serialization fixes: 4 in flight
    // +20 % plain / +26 % DSpark vs one at a time, 16 = 4, identical bytes; one drive saturates at 4), two 6, three or more 8;
    // a library without the auto value gets 4. STRIX_DS41_DISK_QUEUE=auto|N sets it, --disk-queue 1 / --baseline = one at a time.
    const char* disk_queue_env = std::getenv("STRIX_DS41_DISK_QUEUE");
    std::uint32_t disk_queue = disk_queue_env && std::string(disk_queue_env) != "auto"
                                   ? static_cast<std::uint32_t>(std::strtoul(disk_queue_env, nullptr, 10)) : strix::ds41::cuda::kDiskQueueAuto;
    // Whole-device budgets (strix/ds41_rtx_budget.hpp; tierstream ss_budget_plan / ss_budget_ledger): --vram-total-gib N = the WHOLE device
    // (CUDA context + trunk + attention state for --max-positions + drafter + workspaces + arena + margin), --ram-total-gib
    // M = the process's whole commit (pinned included). The arena and the pinned tier get what the measured fixed parts
    // leave (never more than --vram-expert-gib / --ram-expert-gib ask: a cap only shrinks a tier); refused at startup
    // with the minimum N / M when the fixed part does not fit. Peaks are measured (cudaMemGetInfo stages + NVML sampler,
    // process commit) and checked against the caps in the 'budget' events. Unset (default) = today, nothing measured.
    // A load-time setting (not a request option); --baseline keeps it (a device limit, not a tuning).
    strix::ds41::rtx::BudgetCaps caps;
    std::uint32_t budget_sample_ms = 100;  // --budget-sample-ms N: the NVML sampler's period (0: stage samples only)
    const auto total_gib_value = [](const std::string& name, const std::string& v) {
        if (v == "none" || v == "0") return -1.0;
        double g = -1.0;
        try { g = std::stod(v); } catch (const std::exception&) { g = -1.0; }
        if (!std::isfinite(g) || g <= 0.0 || g > 65536.0) usage(("bad value of " + name + ": " + v + " (GiB > 0, or none)").c_str());
        return g;
    };
    if (const char* v = std::getenv("STRIX_DS41_VRAM_TOTAL_GIB"); v && *v) caps.vram_total_gib = total_gib_value("STRIX_DS41_VRAM_TOTAL_GIB", v);
    if (const char* v = std::getenv("STRIX_DS41_RAM_TOTAL_GIB"); v && *v) caps.ram_total_gib = total_gib_value("STRIX_DS41_RAM_TOTAL_GIB", v);
    std::string receipt, cache_kind = "tierstream";
    bool chat_logits = false, moe_v1b3 = false, router_form0 = false, qakv_auto = false, dsrtxk_on = true, fuse_on = true, flush_ready = false;
    std::uint32_t wb_alt = 0;  // --wb-alt N: chats phases alternate the spare target --spares / 0 every N decode tokens
    // window prefetch (tierstream winpf): on in phases whose name ends in 'w' (wforce<K>w, wforcep<K>w, spec<K>w,
    // schats<K>[b<B>][f<N>]w); --winpf-prior FILE (tierstream/hint-prior-v1), knobs below; --wb-defer = dg_wb_defer
    std::string winpf_prior_file;
    TierWinpf winpf_knobs;
    // auto (scored victims after verify windows) measured: DSpark -14 % misses/token, +6.4 % tok/s (7 chats), plain
    // unchanged; a tierstream without the auto policy keeps lru (notice below)
    std::string evict = TierSite::can_auto_evict() ? "auto" : "lru";
    if (!TierSite::can_auto_evict()) std::fprintf(stderr, "eviction: auto needs a tierstream with ts_dg_end_rows; using lru\n");
    bool census = false;  // --census: per-cycle census in the summaries (~15x the JSON of a request, measured)
    bool wb_defer = true;  // adopted default: deferred write-backs; --no-wb-defer = boundary write-backs
    const char* ce_env = std::getenv("STRIX_DS41_CE_PACING");
    bool ce_pacing = !(ce_env && std::string(ce_env) == "0");  // default ON (PRO A/B: exact, not slower)
    // tierstream dg_wb_stream: deferred write-backs on their own D2H stream (overlapping the next layer's demand H2D)
    // instead of the demand stream; default ON (PRO copy bench: duplex 43.3 -> 44.8 GB/s; transport only)
    const char* wbs_env = std::getenv("STRIX_DS41_WB_STREAM");
    bool wb_stream = !(wbs_env && std::string(wbs_env) == "0");
    std::string draft_file, draft_vocab = "auto", window_arg = "controller", controller_scope = "chat";
    std::string draft_keep_from;  // --draft-keep-from FILE: route_stats lines of an earlier run (summed)
    std::uint32_t draft_keep = 64;  // --draft-keep N: the N most routed experts per drafter stage stay (0: all); adopted default 64
    bool route_stats = false;  // --route-stats: the drafter's routed-expert usage per speculative phase (pruning study)
    std::string draft_trace, controller_prior = "builtin:64", controller_save;  // survival-table warm start: adopted default = the
    // built-in node prior; "none" = the cold controller; FILE[:CAP] = a stored table
    bool controller_costs_given = false;
    std::string probe_phase;     // --probe-phase NAME
    bool controller_prior_alt = false;
    std::string controller_costs = "14.8,18.7,26.6,28.9,33.2,37.3,1.6";  // seeded costs (ms), adopted default; "none" = unseeded
    std::uint32_t controller_explore = 0;
    bool controller_learn = false;
    int draft_device = -1;
    std::uint32_t max_rows = 6;
    std::uint32_t attn_warps = 0;  // 0: the attention default (--attn1 / --attn-h1 / replica default)
    bool records_inline = false;   // --records-inline: no deferred step records (the earlier order of host work)
    constexpr std::uint32_t kQualifySteps = 64;  // an exact phase qualifies only with at least this many compared steps
    bool timing = false;
    CudaPrefillOptions o;
    bool weight_dirs_given = false, weight_verify_given = false;
    o.engram_cache_mb = 512;  // decode Engram rows from RAM after their first read (the same bytes; storage stalls hit once)
    // --serve (design notes): the persistent worker; a request = the launch args + its runtime args
    std::string serve_dir, reset_cache = "warm";
    double serve_idle_s = 300.0;
    std::uint32_t serve_max = 0, max_positions_arg = 0, serve_session_mib = 0;
    bool test_hooks_arg = false, serve_keep_on_fail = false;
    std::string residency_save = "auto", residency_seed = "auto";  // the residency at shutdown / the next load's seed
    std::string warmup_graphs = "auto";                   // K list; every (K, rows, logits, capture) graph before ready
    std::string qual_dump;
    strix::ds41::session::Config session_config;
    bool qual = false;  // --qual: qualification records (every window row's token + logits FNV, prompt windows, chat token
                        // vectors and end states, draft confidences as bits); off = today's lines byte for byte
    std::vector<std::string> req_env;  // --env NAME=VALUE (request variables, applied at the request's start)
    bool print_config = false, baseline_arena = false, residency_auto = false;
    bool q8_r1 = true, q8_r1_pad = false, router_pf_explicit = false, hc_cluster = false;
    const auto baseline = [&]() {
        session_config.off("baseline");
        dsrtxk_on = fuse_on = false;
        o.decode_fuse_hc = o.decode_fuse_shared = false;
        o.decode_router_v2 = o.decode_ob_one = o.decode_shared_fork = o.decode_attn_fork = false;
        o.decode_ob_expand = o.decode_moe_expand = o.decode_qakv_s1 = moe_v1b3 = false;
        attn_warps = 0;
        records_inline = router_form0 = qakv_auto = true;
        router_pf_explicit = hc_cluster = false;
        q8_r1 = false;
        q8_r1_pad = false;
        draft_keep = 0;
        controller_prior = controller_costs = "none";
        controller_costs_given = false;
        wb_defer = false;
        ce_pacing = false;
        wb_stream = false;
        file_tier = false;          // the old refusal when the pinned tier does not fit
        ram_expert_gib = -1.0;
        ram_inclusive_gib = 0.0;
        disk_prefetch = false;
        disk_staging = 8;
        disk_queue = 1;
        evict = "lru";
        controller_scope = "run";
        host_spares = 96;
        pressure_spares = 4;
        max_rows = 0;
        arena_gib = -1.0;
        arena_auto = false;
        baseline_arena = true;
        o.lean_state = false;
        draft_vocab = "full";
        warmup_graphs = residency_seed = residency_save = "none";
        o.engram_cache_mb = 0;
        // Respect a caller's explicit load policy, including DG_LOAD_SERIAL=0.
        if (!std::getenv("DG_LOAD_SERIAL")) {
#ifdef _WIN32
            _putenv_s("DG_LOAD_SERIAL", "1");
#else
            setenv("DG_LOAD_SERIAL", "1", 0);
#endif
        }
    };
    const char* baseline_env = std::getenv("STRIX_DS41_RTX_BASELINE");
    if (baseline_env && std::string(baseline_env) == "1") baseline();
    const auto parse_args = [&](const std::vector<std::string>& args) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string a = args[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= args.size()) usage(("missing value of " + a).c_str());
            return args[++i];
        };
        const auto u32 = [&]() {
            const auto v = next();
            try {
                return static_cast<std::uint32_t>(std::stoul(v));
            } catch (const std::exception&) {
                usage(("bad value of " + a + ": " + v).c_str());
            }
        };
        bool known = true;  // two chains: one else-if chain this long nests too deeply for MSVC inside a lambda
        if (a == "--baseline") baseline();
        else if (a == "--print-config" || a == "--print-effective-config") print_config = true;
        else if (a == "--profile") next();             // resolved before the options (strix/ds41_rtx_profile.hpp)
        else if (a == "--allow-pending") {}            // idem
        else if (a == "--ce-pacing") ce_pacing = true;
        else if (a == "--no-ce-pacing") ce_pacing = false;
        else if (a == "--session-cache" || a == "--session-disk") {
            const auto v=next();if(v!="on"&&v!="off")usage("session switch expects on|off");
            if(a=="--session-cache"){session_config.enabled=v=="on";session_config.explicit_opt_in=v=="on";session_config.reason=v=="off"?"switch":"";}
            else session_config.disk=v=="on";
        }
        else if (a == "--session-ram-bytes") session_config.ram=strix::ds41::session::unsigned_value(next());
        else if (a == "--session-max-parked") session_config.max_parked=strix::ds41::session::unsigned_value(next());
        else if (a == "--session-dir") session_config.dir=next();
        else if (a == "--session-disk-bytes") {session_config.disk_bytes=strix::ds41::session::unsigned_value(next());session_config.disk_explicit=true;}
        else if (a == "--wb-stream") wb_stream = true;
        else if (a == "--no-wb-stream") wb_stream = false;
        else if (a == "--q8-r1") q8_r1 = true;
        else if (a == "--q8-r1-off") q8_r1 = false;
        else if (a == "--q8-r1-pad") q8_r1_pad = true;
        else if (a == "--q8-r1-nopad") q8_r1_pad = false;
        else if (a == "--dsrtxk") dsrtxk_on = true;
        else if (a == "--fuse") fuse_on = true;
        else if (a == "--records-deferred") records_inline = false;
        else if (a == "--import-state") state_file = next();
        else if (a == "--ref") ref_file = next();
        else if (a == "--steps") steps = u32();
        else if (a == "--arena-gib") {
            const auto value = next();
            arena_auto = value == "auto";
            if (!arena_auto) arena_gib = std::stod(value);
        }
        else if (a == "--arena-headroom-gib") arena_headroom_gib = std::stod(next());
        else if (a == "--spares") spares = u32();
        else if (a == "--host-spares") host_spares = u32();
        else if (a == "--predict") predict = u32();
        else if (a == "--prefetch-layer") pf_layer = u32();
        else if (a == "--prefetch-token") pf_token = u32();
        else if (a == "--pressure-spares") pressure_spares = u32();
        else if (a == "--phases") phases_arg = next();
        else if (a == "--routes") routes = next();
        else if (a == "--seqmod") seqmod = next();
        else if (a == "--chats") chats_arg = next();
        else if (a == "--chat-prompt") chat_prompt = u32();
        else if (a == "--chat-steps") chat_steps = u32();
        else if (a == "--seqs-file") seqs_file = next();
        else if (a == "--no-lean-state") o.lean_state = false;
        else if (a == "--lean-state") o.lean_state = true;
        else if (a == "--timing") timing = true;
        else if (a == "--qual-dump") qual_dump = next();
        else if (a == "--json") json_out = next();
        else if (a == "--timeline") timeline_out = next();
        else if (a == "--verify") verify_n = u32();
        else if (a == "--warm") warm = u32();
        else if (a == "--host-margin-gib") host_margin_gib = std::stod(next());
        else if (a == "--weight-dirs") { o.weight_read.weight_dirs = next(); weight_dirs_given = true; }
        else if (a == "--drive-readers") {
            try { o.weight_read.set_drive_readers(next()); }
            catch (const std::exception& e) { usage(e.what()); }
        }
        else if (a == "--weight-verify") { o.weight_read.weight_verify = next(); weight_verify_given = true; }
        else if (a == "--engram") o.engram = next();
        else if (a == "--layers") layers_arg = u32();
        else if (a == "--max-pinned-gib") max_pinned_gib = std::stod(next());
        else if (a == "--receipt") receipt = next();
        else if (a == "--chat-logits") chat_logits = true;
        else if (a == "--pressure-free") pressure_free = u32();
        else if (a == "--pressure-recent") pressure_recent = u32();
        else if (a == "--fuse-hc") o.decode_fuse_hc = true;          // bitwise (cuda-decode-fuse gates)
        else if (a == "--fuse-shared") o.decode_fuse_shared = true;  // bitwise at R1 (cuda-decode-fuse gates)
        else known = false;
        if (known) continue;
        known = true;  // three-tier placement (its own chain: MSVC C1061)
        const auto gib_or_auto = [&](double auto_value) {
            const auto v = next();
            if (v == "auto") return auto_value;
            double g = -1.0;
            try { g = std::stod(v); } catch (const std::exception&) { g = -1.0; }
            if (!std::isfinite(g) || g < 0.0 || g > 65536.0) usage(("bad value of " + a + ": " + v).c_str());
            return g;
        };
        if (a == "--vram-expert-gib") {  // = --arena-gib
            const double g = gib_or_auto(-2.0);
            arena_auto = g == -2.0;
            if (!arena_auto) arena_gib = g;
        }
        else if (a == "--ram-expert-gib") ram_expert_gib = gib_or_auto(-1.0);
        else if (a == "--ram-inclusive-gib") ram_inclusive_gib = gib_or_auto(-1.0);
        else if (a == "--file-tier") file_tier = true;
        else if (a == "--no-file-tier") file_tier = false;
        else if (a == "--disk-prefetch") disk_prefetch = true;
        else if (a == "--no-disk-prefetch") disk_prefetch = false;
        else if (a == "--disk-staging") {
            disk_staging = u32();
            if (disk_staging < 1 || disk_staging > 256) usage("--disk-staging must be 1..256");
        } else if (a == "--disk-queue") {
            const std::string v = next();
            disk_queue = v == "auto" ? strix::ds41::cuda::kDiskQueueAuto : static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
            if (disk_queue != strix::ds41::cuda::kDiskQueueAuto && (disk_queue > 64 || v.find_first_not_of("0123456789") != std::string::npos))
                usage("--disk-queue must be auto or 0..64");
        }
        else if (a == "--vram-total-gib") caps.vram_total_gib = total_gib_value(a, next());
        else if (a == "--ram-total-gib") caps.ram_total_gib = total_gib_value(a, next());
        else if (a == "--vram-margin-gib" || a == "--ram-margin-gib") {
            const double g = gib_or_auto(-1.0);
            if (g < 0.0) usage(("bad value of " + a + " (GiB >= 0)").c_str());
            (a == "--vram-margin-gib" ? caps.vram_margin_gib : caps.ram_margin_gib) = g;
        }
        else if (a == "--budget-receipt") caps.receipt = true;
        else if (a == "--budget-sample-ms") budget_sample_ms = u32();
        else known = false;
        if (known) continue;
        if (a == "--draft") draft_file = next();
        else if (a == "--draft-device") draft_device = static_cast<int>(std::stol(next()));
        else if (a == "--draft-vocab") draft_vocab = next();
        else if (a == "--window") window_arg = next();
        else if (a == "--controller-scope") controller_scope = next();
        else goto more_options;  // MSVC C1061: one else-if chain of all options nests too deeply; split in two
        continue;
    more_options:
        if (a == "--route-stats") route_stats = true;
        else if (a == "--draft-keep") draft_keep = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--draft-keep-from") draft_keep_from = next();
        else if (a == "--draft-trace") draft_trace = next();
        else if (a == "--controller-prior") controller_prior = next();
        else if (a == "--controller-save") controller_save = next();
        else if (a == "--controller-prior-alt") controller_prior_alt = true;
        else if (a == "--controller-costs") { controller_costs = next(); controller_costs_given = true; }
        else if (a == "--controller-explore") controller_explore = u32();
        else if (a == "--controller-learn") controller_learn = true;
        else if (a == "--probe-phase") probe_phase = next();
        else if (a == "--max-rows") max_rows = u32();
        else if (a == "--stream-hashes") o.stream_hashes = true;  // eager steps: FNV of the last row's stream after each layer (csmoke)
        else if (a == "--cache") cache_kind = next();  // tierstream (the only cache)
        else if (a == "--flush-ready") flush_ready = true;  // tierstream dg_flush_ready (diagnostic A/B)
        else if (a == "--no-flush-ready") flush_ready = false;  // persist: a request turns it off again
        else if (a == "--engram-cache-mb") o.engram_cache_mb = u32();  // Engram records kept in RAM (default 512; 0 = off)
        else if (a == "--winpf-prior") winpf_prior_file = next();
        else if (a == "--winpf-l1-cap") winpf_knobs.l1_cap = u32();
        else if (a == "--winpf-pmin") winpf_knobs.l1_pmin = std::stof(next());
        else if (a == "--winpf-prior-cap") winpf_knobs.prior_cap = u32();
        else if (a == "--winpf-prior-pmin") winpf_knobs.prior_pmin = std::stof(next());
        else if (a == "--winpf-revert") winpf_knobs.revert_unused = u32() != 0U;
        else if (a == "--evict") {
            evict = next();
            if (const auto* error = rs::eviction_error(evict, TierSite::can_set_policy(), TierSite::can_auto_evict())) usage(error);
        }
        else if (a == "--no-census") census = false;
        else if (a == "--census") census = true;
        else if (a == "--wb-defer") wb_defer = true;
        else if (a == "--no-wb-defer") wb_defer = false;  // persist: a request turns it off again
        else if (a == "--wb-alt") wb_alt = u32();           // write-back probe: spare target --spares / 0 every N tokens
        else if (a == "--router-v2") o.decode_router_v2 = true;      // router v2 (bitwise, kern-bench router)
        else if (a == "--router-pf") { router_form0 = false; router_pf_explicit = true; }  // router v2 loads moved earlier (same bits)
        else if (a == "--router-pf0") router_form0 = true;  // router v2 form 0 (the earlier default)
        else if (a == "--pdl") o.decode_pdl = true;                  // PDL + L2 prefetch for the dense GEMVs (sm_90+)
        else if (a == "--pdl-np") {  // PDL for the dense GEMVs without the L2 weight prefetch (sm_90+)
            o.decode_pdl = true;
            strix::ds41::cuda::set_q8_pdl_prefetch(false);
        }
        else if (a == "--pdl-regs") {  // PDL with the first step's weights in registers before the wait (sm_90+)
            o.decode_pdl = true;
            strix::ds41::cuda::set_q8_pdl_mode(2);
        }
        else if (a == "--shared-fork") o.decode_shared_fork = true;  // shared expert on a side branch (same bits)
        else if (a == "--qakv-s1") { o.decode_qakv_s1 = true; qakv_auto = false; }
        else if (a == "--qakv-auto") qakv_auto = true;  // qa_kv with the automatic split (the earlier default)
        else if (a == "--attn-fork") o.decode_attn_fork = true;      // attention prep concurrent with q_b (same bits)
        else if (a == "--ob-expand") o.decode_ob_expand = true;      // with --ob-one, hc_expand in o_b's epilogue (same bits)
        else if (a == "--moe-expand") o.decode_moe_expand = true;    // the FFN hc_expand in the site's combine (same bits)
        else if (a == "--ob-one") o.decode_ob_one = true;            // o_b's two K-halves in one launch (same bits)
        else if (a == "--attn1") attn_warps = 1;                     // 1-warp attention blocks (same bits)
        else if (a == "--attn-h1") attn_warps = 4;                   // 1 head per warp, 4 per block (same bits)
        else if (a == "--attn-pf") strix::ds41::cuda::attn::set_attention_prefetch(true);  // attention loads at once (same bits)
        // clustered hc split/norm (sm_90+); same bits. The 25-warp partials stay: with the cluster the 5-warp form
        // measured slower on the PRO (hc mixes+norm 11.26 vs 10.88 us); --hc-partial5 keeps it opt-in
        else if (a == "--hc-cluster") hc_cluster = true;
        else if (a == "--hc-partial5") strix::ds41::cuda::set_hc_partial_warps(5);  // 5-warp hc partials (same bits)
        else if (a == "--moe-v1b3") moe_v1b3 = true;                  // v1b gate/up + v3 down (same bits)
        else if (a == "--dsrtxk-off") dsrtxk_on = false;              // the nine dsrtxk options below off (baseline arm)
        else if (a == "--fuse-off") fuse_on = false;                  // --fuse-hc / --fuse-shared off
        else if (a == "--records-inline") records_inline = true;      // each step's record written before the next step (A/B)
        else if (a == "--serve-session-mib") {
            serve_session_mib = u32();session_config.ram=std::uint64_t{serve_session_mib}*1024*1024;
            session_config.enabled=serve_session_mib!=0;session_config.explicit_opt_in=serve_session_mib!=0;session_config.reason=serve_session_mib?"":"legacy zero budget";
        }
        else if (a == "--serve") serve_dir = next();                  // persist: the worker's spool directory
        else if (a == "--serve-idle") serve_idle_s = std::stod(next());  // persist: exit after S s without a request
        else if (a == "--serve-max") serve_max = u32();               // persist: exit after N requests (0: no limit)
        else if (a == "--max-positions" || a == "--positions") max_positions_arg = u32();   // capacity: absolute position bound (frozen at load)
        else if (a == "--serve-keep-on-fail") serve_keep_on_fail = true;  // persist: a failed gate does not end the worker
        else if (a == "--qual") qual = true;                          // qualification records (runtime)
        else if (a == "--residency-save") residency_save = next();    // write the residency at exit
        else if (a == "--residency-seed") residency_seed = next();    // seed the placement from such a file
        else if (a == "--warmup-graphs") warmup_graphs = next();      // capture the common graphs at load
        else if (a == "--test-hooks") test_hooks_arg = true;          // open tierstream with dg_test_hooks (pressure)
        else if (a == "--reset-cache") reset_cache = next();          // persist: warm (default) | open
        else if (a == "--env") req_env.push_back(next());             // persist: a request variable NAME=VALUE
        else usage(("unknown option " + a).c_str());
    }
    try {
        o.weight_read.environment(weight_dirs_given, weight_verify_given, model);
        o.weight_read.require_supported(model);
    } catch (const std::exception& e) { usage(e.what()); }

    };
    const std::vector<std::string> launch_args(argv + 2, argv + argc);
    // --profile: resolved on the launch options and the environment before they are parsed (a profile value is the
    // lowest explicit layer: the parser's options and the levers' variables already read above win over it)
    namespace rprof = strix::ds41::rtx::profile;
    rprof::Resolution prof;
    try { prof = rprof::resolve(rprof::App::replica, rprof::Inputs{launch_args, {}}); }
    catch (const std::exception& e) { usage(e.what()); }
    {
        // a profile value other than the builtin form: today none (every ready row is the replica's default)
        const auto apply_lever = [&](std::string_view lever, const std::string& v) {
            const bool on = v == "on";
            if (v != "on" && v != "off") return false;
            if (lever == "lean_state") o.lean_state = on;
            else if (lever == "q8_r1") { q8_r1 = on; q8_r1_pad = false; }
            else if (lever == "dsrtxk") dsrtxk_on = on;
            else if (lever == "fuse") fuse_on = on;
            else if (lever == "decode_forms") records_inline = router_form0 = qakv_auto = !on;
            else if (lever == "evict_auto") evict = on && TierSite::can_auto_evict() ? "auto" : "lru";
            else if (lever == "controller") {
                controller_prior = on ? "builtin:64" : "none";
                controller_costs = on ? "14.8,18.7,26.6,28.9,33.2,37.3,1.6" : "none";
                controller_scope = on ? "chat" : "run";
                controller_costs_given = false;
            }
            else if (lever == "draft_keep") draft_keep = on ? 64U : 0U;
            else if (lever == "draft_vocab_auto") draft_vocab = on ? "auto" : "full";
            else if (lever == "wb_defer") wb_defer = on;
            else if (lever == "ce_pacing") ce_pacing = on;
            else if (lever == "wb_stream") wb_stream = on;
            else if (lever == "file_tier") file_tier = on;
            else if (lever == "residency_auto") residency_seed = residency_save = on ? "auto" : "none";
            else if (lever == "warmup_graphs") warmup_graphs = on ? "auto" : "none";
            else if (lever == "engram_cache") o.engram_cache_mb = on ? 512U : 0U;
            else return false;
            return true;
        };
        for (const auto& e : prof.entries) {
            if ((e.status != rprof::Status::profile && e.status != rprof::Status::unqualified) || e.value == e.lever->builtin) continue;
            if (!apply_lever(e.lever->name, e.value))
                usage(("--profile " + std::string(rprof::name(prof.name)) + ": lever " + std::string(e.lever->name) + " = " + e.value +
                       " has no binding in this build").c_str());
        }
        for (const auto& e : prof.entries)
            if (e.status == rprof::Status::pending)
                std::fprintf(stderr, "profile %s: %s pending, not used (%s)\n", rprof::name(prof.name), std::string(e.lever->name).c_str(),
                             std::string(e.lever->source).c_str());
        if (prof.qualification() != "qualified")
            std::fprintf(stderr, "profile %s: %s (--allow-pending: benchmarks only)\n", rprof::name(prof.name), prof.qualification().c_str());
    }
    parse_args(launch_args);
    namespace fs = std::filesystem;
    if (arena_gib < 0) arena_gib = draft_file.empty() ? 80.0 : baseline_arena ? 71.5 : draft_keep == 64 ? 74.8 : draft_keep == 32 ? 76.5 : 71.5;
    if (draft_vocab == "auto") {
        const char* dir = std::getenv("STRIX_DS41_DRAFT_VOCAB");
        std::error_code ec;
        fs::path vp = dir && *dir ? fs::path(dir) : fs::path();
        if (vp.empty() || !fs::is_directory(vp, ec)) vp = fs::path(draft_file).parent_path() / "ds41-draft-vocab";
        ec.clear();
        if (!draft_file.empty() && fs::is_directory(vp, ec)) draft_vocab = "auto:" + vp.string();
        else {
            draft_vocab = "full";
            if (!draft_file.empty()) std::fprintf(stderr, "draft vocab: no auto directory; using full\n");
        }
    }
    const char* residency_env = std::getenv("STRIX_DS41_RESIDENCY");
    const std::string auto_path = residency_env && *residency_env ? residency_env : model + ".residency";
    if (residency_seed == "auto") {
        residency_auto = true;
        residency_seed.clear();
        std::error_code ec;
        if (fs::exists(auto_path, ec)) {
            try {
                // The admitted target has 384 routed experts per layer. Rechecked against the reader below.
                residency_keys(auto_path, layers_arg, 384);
                residency_seed = auto_path;
                std::fprintf(stderr, "residency: auto seed %s\n", auto_path.c_str());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "residency: %s; using routes order\n", e.what());
            }
        }
    }
    const bool residency_save_auto = residency_save == "auto", warmup_graphs_auto = warmup_graphs == "auto";
    if (residency_save == "auto") residency_save = serve_dir.empty() ? "" : auto_path;
    if (warmup_graphs == "auto") warmup_graphs = serve_dir.empty() ? "" : "0,8";
    if (residency_seed == "none") residency_seed.clear();
    if (!residency_auto && !residency_seed.empty()) {
        try { residency_keys(residency_seed, layers_arg, 384); }
        catch (const std::exception& e) { usage(e.what()); }
    }
    if (residency_save == "none") residency_save.clear();
    if (warmup_graphs == "none") warmup_graphs.clear();
    const auto resolve_keep = [&](const auto& emit) {
        CudaDrafterOptions dopt;
        if (draft_keep != 0U && draft_keep_from.empty()) {  // the built-in keep-64 lists (held-out route statistics)
            if (draft_keep != 64U) usage("--draft-keep N other than 64 needs --draft-keep-from FILE (route_stats lines)");
            static constexpr std::uint16_t kKeep64[3][64] = {
{97, 86, 43, 85, 36, 8, 117, 5, 47, 3, 33, 19, 124, 92, 123, 95, 61, 31, 27, 2, 72, 6, 17, 14, 99, 101, 23, 110, 24, 93, 68, 34, 4, 88, 109, 15, 62, 21, 74, 54, 70, 60, 26, 89, 11, 57, 63, 96, 38, 64, 13, 71, 58, 66, 113, 59, 49, 106, 77, 53, 115, 65, 118, 20},
{121, 98, 51, 109, 61, 58, 78, 31, 113, 114, 82, 53, 56, 52, 122, 24, 26, 55, 41, 66, 5, 62, 25, 27, 45, 10, 97, 32, 115, 79, 111, 28, 39, 35, 68, 49, 11, 4, 21, 77, 63, 1, 19, 47, 118, 89, 126, 76, 37, 7, 101, 116, 100, 64, 36, 112, 102, 125, 18, 110, 71, 12, 43, 80},
{101, 19, 48, 81, 58, 1, 103, 127, 97, 78, 6, 98, 9, 12, 17, 38, 102, 16, 122, 51, 83, 88, 53, 27, 96, 21, 111, 15, 84, 120, 93, 67, 14, 80, 13, 49, 45, 66, 112, 69, 77, 20, 87, 35, 94, 24, 59, 11, 91, 82, 119, 10, 47, 52, 79, 116, 118, 0, 2, 3, 4, 5, 7, 8},
            };
            dopt.keep.assign(3, {});
            for (std::uint32_t s = 0; s < 3U; ++s) dopt.keep[s].assign(kKeep64[s], kKeep64[s] + 64);
            emit("{\"event\":\"draft_keep\",\"keep\":64,\"source\":\"builtin\"}");
        } else if (draft_keep != 0U) {  // expert pruning: the N most routed per stage in the stats file (ties: lower id)
            namespace dsp = strix::ds41::dspark;
            const std::uint32_t E = dsp::experts, S = dsp::stages;
            std::vector<double> cnt(std::size_t{S} * E, 0.0);
            std::ifstream kf(draft_keep_from);
            std::string line;
            std::uint32_t lines = 0;
            while (std::getline(kf, line)) {
                if (line.find("\"event\":\"route_stats\"") == std::string::npos) continue;
                const auto a = line.find("\"counts\":["), b = line.find(']', a);
                if (a == std::string::npos || b == std::string::npos) continue;
                std::istringstream cs(line.substr(a + 10, b - a - 10));
                std::string tok;
                std::size_t i = 0;
                while (std::getline(cs, tok, ',') && i < cnt.size()) cnt[i++] += std::stod(tok);
                if (i != cnt.size()) throw std::runtime_error("--draft-keep-from: a route_stats line without " + std::to_string(cnt.size()) + " counts");
                ++lines;
            }
            if (lines == 0) throw std::runtime_error("--draft-keep-from: no route_stats line in " + draft_keep_from);
            std::ostringstream ks;
            ks << "{\"event\":\"draft_keep\",\"keep\":" << draft_keep << ",\"lines\":" << lines << ",\"route_share\":[";
            dopt.keep.resize(S);
            for (std::uint32_t s = 0; s < S; ++s) {
                std::vector<std::uint32_t> ids(E);
                for (std::uint32_t e = 0; e < E; ++e) ids[e] = e;
                const double* c = cnt.data() + std::size_t{s} * E;
                std::stable_sort(ids.begin(), ids.end(), [&](std::uint32_t x, std::uint32_t y) { return c[x] > c[y]; });
                ids.resize(std::min(draft_keep, E));
                double tot = 0, kept = 0;
                for (std::uint32_t e = 0; e < E; ++e) tot += c[e];
                for (const auto e : ids) kept += c[e];
                ks << (s ? "," : "") << (tot > 0 ? kept / tot : 0.0);
                dopt.keep[s] = std::move(ids);
            }
            ks << "]}";
            emit(ks.str());
        }
        return dopt.keep;
    };
    if (max_rows == 0 && serve_dir.empty()) {
        bool windows = false;
        for (const auto& ph : split(phases_arg, ','))
            windows = windows || ph.rfind("wforce", 0) == 0 || ph.rfind("spec", 0) == 0 || ph.rfind("schats", 0) == 0 || ph.rfind("wsmoke", 0) == 0;
        max_rows = windows ? 6U : 1U;
    }
    // the profile receipt: the resolution plus every lever's effective form after the options ("on" = today's default)
    const auto profile_json = [&]() {
        const auto onoff = [](bool b) { return std::string(b ? "on" : "off"); };
        return prof.json([&](std::string_view lever) -> std::string {
            if (lever == "lean_state") return onoff(o.lean_state);
            if (lever == "q8_r1") return q8_r1 ? (q8_r1_pad ? "pad" : "on") : "off";
            if (lever == "dsrtxk") return onoff(dsrtxk_on);
            if (lever == "fuse") return onoff(fuse_on);
            if (lever == "decode_forms") {
                if (!records_inline && !router_form0 && !qakv_auto) return "on";
                return records_inline && router_form0 && qakv_auto ? "off" : "custom";
            }
            if (lever == "evict_auto") return evict == "auto" ? "on" : evict;
            if (lever == "controller") {
                const bool on = window_arg == "controller" && controller_prior == "builtin:64" && controller_scope == "chat" &&
                                controller_costs == "14.8,18.7,26.6,28.9,33.2,37.3,1.6";
                const bool off = (controller_prior.empty() || controller_prior == "none") && controller_costs == "none" && controller_scope == "run";
                return on ? "on" : off ? "off" : "custom";
            }
            if (lever == "draft_keep") return draft_keep == 64U && draft_keep_from.empty() ? "on" : draft_keep == 0U ? "off" : std::to_string(draft_keep);
            if (lever == "draft_vocab_auto") return draft_vocab.rfind("auto", 0) == 0 ? "on" : draft_vocab == "full" ? "off" : draft_vocab;
            if (lever == "wb_defer") return onoff(wb_defer);
            if (lever == "ce_pacing") return onoff(ce_pacing);
            if (lever == "wb_stream") return onoff(wb_stream);
            if (lever == "file_tier") return onoff(file_tier);
            if (lever == "residency_auto")
                return residency_auto && residency_save_auto ? "on" : !residency_auto && residency_seed.empty() && residency_save.empty() ? "off" : "custom";
            if (lever == "warmup_graphs")
                return warmup_graphs_auto ? (serve_dir.empty() ? "on (serve only)" : "on") : warmup_graphs.empty() ? "off" : "custom";
            if (lever == "engram_cache")
                return o.engram_cache_mb == 512U ? "on" : o.engram_cache_mb == 0U ? "off" : std::to_string(o.engram_cache_mb);
            if (lever == "prompt_layer_major") return "off";
            return "";
        });
    };
    const auto effective_config = [&](bool session_policy = false) {
        // Resolve the same feature expansion as below, without calling any CUDA entry point.
        std::ostringstream j;
        j << std::boolalpha << "{";
        bool first = true;
        const auto field = [&](const char* k, const auto& v) {
            if (!first) j << ',';
            first = false;
            j << rs::json_str(k) << ':' << v;
        };
        const auto str = [&](const char* k, const std::string& v) { field(k, rs::json_str(v)); };
        field("dsrtxk", dsrtxk_on); field("fuse", fuse_on); field("records_inline", records_inline);
        field("router_form", router_form0 ? 0 : (dsrtxk_on || router_pf_explicit) ? 2 : 0);
        field("qakv_s1", !qakv_auto && (dsrtxk_on || o.decode_qakv_s1));
        field("q8_r1", q8_r1); field("q8_r1_pad", q8_r1_pad);
        field("draft_keep", draft_keep); field("arena_gib", arena_gib); field("arena_auto", arena_auto); field("lean_state", o.lean_state);
        str("draft_vocab", draft_vocab);
        str("controller_scope", controller_scope);
        // Source filenames are provenance, not different runtime settings. Compare effective data below in CPU tests.
        std::string prior = "none";
        if (!controller_prior.empty() && controller_prior != "none") {
            const auto t = *resolve_prior(controller_prior);
            std::ostringstream ps;
            for (const auto& row : t.seen) for (auto v : row) ps << v << ',';
            ps << '/';
            for (const auto& row : t.hits) for (auto v : row) ps << v << ',';
            prior = ps.str();
        }
        str("controller_prior", prior); str("controller_costs", controller_costs);
        const auto keep = resolve_keep([](const std::string&) {});
        std::ostringstream kj;
        kj << '[';
        for (std::size_t st = 0; st < keep.size(); ++st) {
            kj << (st ? ",[" : "[");
            for (std::size_t n = 0; n < keep[st].size(); ++n) kj << (n ? "," : "") << keep[st][n];
            kj << ']';
        }
        kj << ']';
        field("draft_keep_ids", kj.str());
        field("controller_explore", controller_explore); field("controller_learn", controller_learn);
        field("controller_prior_alt", controller_prior_alt); str("controller_save", controller_save);
        field("spares", spares); field("host_spares", host_spares); field("predict", predict);
        field("max_rows", max_rows); str("window", window_arg); field("pressure_spares", pressure_spares);
        field("pressure_free", pressure_free); field("pressure_recent", pressure_recent);
        field("prefetch_layer", pf_layer); field("prefetch_token", pf_token);
        field("host_margin_gib", host_margin_gib); field("max_pinned_gib", max_pinned_gib);
        str("engram", o.engram); field("engram_cache_mb", o.engram_cache_mb);
        str("evict", evict); field("census", census);
        field("wb_defer", wb_defer); field("wb_alt", wb_alt); field("flush_ready", flush_ready);
        field("ce_pacing", ce_pacing);
        field("wb_stream", wb_stream);
        field("file_tier", file_tier);
        const auto gib_str = [](double g) {
            if (g < 0) return std::string("auto");
            std::ostringstream gs;
            gs << g;
            return gs.str();
        };
        str("ram_expert_gib", gib_str(ram_expert_gib));
        str("ram_inclusive_gib", gib_str(ram_inclusive_gib));
        field("disk_prefetch", disk_prefetch); field("disk_staging", disk_staging); if (disk_queue == strix::ds41::cuda::kDiskQueueAuto) str("disk_queue", "auto"); else field("disk_queue", disk_queue);
        str("vram_total_gib", caps.device_capped() ? gib_str(caps.vram_total_gib) : std::string("none"));
        str("ram_total_gib", caps.host_capped() ? gib_str(caps.ram_total_gib) : std::string("none"));
        field("vram_margin_gib", caps.vram_margin_gib); field("ram_margin_gib", caps.ram_margin_gib);
        field("budget_receipt", caps.measuring()); field("budget_sample_ms", budget_sample_ms);
        // the plan itself needs the measured fixed parts (load time: the 'budget' event, stage "plan"); here its rule
        str("budget_plan", !caps.capped() ? std::string("none: today's arena (--vram-expert-gib) and pinned tier (--ram-expert-gib)") :
            std::string("at load: arena slots = min(--vram-expert-gib, (") + (caps.device_capped() ? gib_str(caps.vram_total_gib) : std::string("no cap")) +
            " GiB - measured context/trunk/drafter/workspaces - " + gib_str(caps.vram_margin_gib) + " margin - bank/site/graph reserves) / expert); pinned blocks = min(--ram-expert-gib, (" +
            (caps.host_capped() ? gib_str(caps.ram_total_gib) : std::string("no cap")) + " GiB - measured commit - " + gib_str(caps.ram_margin_gib) +
            " margin - load staging) / expert); the rest in the model file; refused with the minimum when the fixed part does not fit");
        str("warmup_graphs", warmup_graphs.empty() ? "none" : warmup_graphs);
        str("residency_seed", residency_seed.empty() ? "none" : (residency_auto ? "auto:" : "") + residency_seed);
        str("residency_save", residency_save.empty() ? "none" : residency_save);
        const char* serial = std::getenv("DG_LOAD_SERIAL");
        field("load_serial", serial && std::string(serial) == "1");
        field("layers", layers_arg); field("warm", warm); field("draft_device", draft_device);
        field("attn_warps", attn_warps ? attn_warps : dsrtxk_on ? 4 : 0);
        field("max_positions", max_positions_arg); str("cache", cache_kind); str("reset_cache", reset_cache);
        field("decode_fuse_hc", fuse_on || o.decode_fuse_hc);
        field("decode_fuse_shared", fuse_on || o.decode_fuse_shared);
        field("decode_router_v2", dsrtxk_on || o.decode_router_v2);
        field("decode_ob_one", dsrtxk_on || o.decode_ob_one);
        field("decode_shared_fork", dsrtxk_on || o.decode_shared_fork);
        field("decode_attn_fork", dsrtxk_on || o.decode_attn_fork);
        field("decode_ob_expand", dsrtxk_on || o.decode_ob_expand);
        field("decode_moe_expand", dsrtxk_on || o.decode_moe_expand);
        field("moe_v1b3", dsrtxk_on || moe_v1b3);
        field("hc_cluster", dsrtxk_on || hc_cluster);
        field("decode_pdl", o.decode_pdl);
        str("winpf_prior", winpf_prior_file);
        field("winpf_l1_cap", winpf_knobs.l1_cap); field("winpf_pmin", winpf_knobs.l1_pmin);
        field("winpf_prior_cap", winpf_knobs.prior_cap); field("winpf_prior_pmin", winpf_knobs.prior_pmin);
        field("winpf_revert", winpf_knobs.revert_unused);
        field("serve", !serve_dir.empty()); field("serve_idle", serve_idle_s); field("serve_max", serve_max);
        field("profile", profile_json());
        if (session_policy) {
        field("session_cache",session_config.enabled);field("session_disk",session_config.enabled&&session_config.disk);
        field("session_ram_bytes",session_config.ram==UINT64_MAX?0:session_config.ram);field("session_max_parked",session_config.max_parked);
        field("session_explicit_opt_in",session_config.explicit_opt_in);str("session_route","rtx-tp1-decode-rows");
        field("session_disk_bytes",session_config.disk_bytes);str("session_dir",session_config.dir.string());
        str("session_off_reason",session_config.reason);
        }
        j << "}";
        return j.str();
    };
    if(serve_dir.empty())session_config.off("not serving");
    else if(layers_arg!=40)session_config.off("requires 40 layers");
    else if(!session_config.ram||!session_config.max_parked)session_config.off("zero RAM budget or parked limit");
    strix::ds41::session::qualify(session_config,"rtx-tp1-decode-rows");
    if(!session_config.enabled&&session_config.reason.empty())session_config.reason="environment";
    if (print_config) {
        session_config.resolve_ram();session_config.resolve_directory();
        std::puts(effective_config(true).c_str());
        return 0;
    }
    const bool launch_wb_defer = wb_defer, launch_flush_ready = flush_ready;  // a worker without ts_dg_set_sched keeps them
    const auto set_env = [](const std::string& n, const std::string& v) {  // "\x01" = unset
#ifdef _WIN32
        _putenv_s(n.c_str(), v == "\x01" ? "" : v.c_str());
#else
        if (v == "\x01") unsetenv(n.c_str()); else setenv(n.c_str(), v.c_str(), 1);
#endif
    };
    // the request variables as the process started (before any --env): every request rebuilds its environment from it
    std::map<std::string, std::string> start_env;
    for (const char* n : {"STRIX_DS41_CUDA_ENGRAM_CHECK", "STRIX_DS41_REPLICA_WINDOW_ENGRAM_PREFETCH"}) {
        const char* v = std::getenv(n);
        start_env[n] = v ? std::string(v) : std::string("\x01");
    }
    const auto apply_env = [&]() {  // start values, then every --env of the launch args and the request, in order
        for (const auto& [n, v] : start_env) set_env(n, v);
        for (const auto& e : req_env) {
            const auto eq = e.find('=');
            if (eq == std::string::npos || eq == 0 || !rs::is_request_env(e.substr(0, eq))) usage(("--env NAME=VALUE with a request variable, got " + e).c_str());
            set_env(e.substr(0, eq), e.substr(eq + 1));
        }
    };
    apply_env();
    if (!std::isfinite(arena_headroom_gib) || arena_headroom_gib < 0.0 || arena_headroom_gib > 1024.0)
        usage("--arena-headroom-gib must be finite in 0..1024");
    const bool serving = !serve_dir.empty();
    if (serve_session_mib && !serving) usage("--serve-session-mib requires --serve");
    if (serving && max_positions_arg == 0) usage("--serve needs --max-positions N (the absolute capacity every request and its fresh reference share)");
    if (serving && max_rows == 0) usage("--serve needs --max-rows R");
    // Default ON: every option is bitwise; full-model qualification measured
    // all-hit 10.50 vs 12.62 ms/token; full-loop improvement +5.8..+10.8 %, all-hit 10.51 ms.
    // The flags above stay accepted; --dsrtxk-off / --fuse-off restore the measured baselines.
    if (fuse_on) o.decode_fuse_hc = o.decode_fuse_shared = true;
    if (dsrtxk_on) {
        o.decode_router_v2 = o.decode_ob_one = o.decode_shared_fork = o.decode_attn_fork = true;
        o.decode_ob_expand = o.decode_moe_expand = moe_v1b3 = true;
        if (!attn_warps) attn_warps = 4;                        // --attn-h1 unless --attn1 was given
        strix::ds41::cuda::set_hc_norm_cluster(true);           // --hc-cluster
        // --router-pf (form 2) default ON (measured: bitwise rows 1..8; R1 15.4 vs 17.3 us,
        // R2 17.4 vs 21.4 us; 8-layer smokes -0.5 %); --router-pf0 restores form 0
        strix::ds41::cuda::set_router_v2_form(2);
        // --qakv-s1 default ON (measured: smokes == reference, -1.54/-1.84/-1.94 % device ms
        // vs the bracketing parents; bench q_a S=1 11.4 vs S=2 12.9 us, bitwise); --qakv-auto restores the automatic split
        o.decode_qakv_s1 = true;
    }
    strix::ds41::cuda::set_q8_r1(q8_r1);
    strix::ds41::cuda::set_q8_r1_pad(q8_r1_pad);
    strix::ds41::cuda::set_hc_norm_cluster(dsrtxk_on || hc_cluster);
    if (router_pf_explicit) strix::ds41::cuda::set_router_v2_form(2);
    if (router_form0) strix::ds41::cuda::set_router_v2_form(0);
    if (qakv_auto) o.decode_qakv_s1 = false;  // --qakv-auto wins over --qakv-s1 and the default  // --router-pf0 wins over --router-pf and the default
    if (attn_warps) strix::ds41::cuda::attn::set_attention_chunk_warps(attn_warps);
    std::fprintf(stderr, "decode options: fuse %s, dsrtxk %s\n", fuse_on ? "on" : "off", dsrtxk_on ? "on (default)" : "off");
    if (state_file.empty()) usage("--import-state is required");
    if (cache_kind != "tierstream") usage("--cache: tierstream (the replica's own cache was removed)");
    // ---- per-request configuration: phases, fixtures, checks (the launch args once; with --serve every request) ----
    std::vector<std::string> phases;
    bool any_exact = false, any_chats = false, any_window = false, any_draft = false;
    // the imported state (re-read when a request names another --import-state; capacity checked against the load's)
    auto imported0 = strix::deepseek41::load_attention_state(state_file);
    strix::ds41::session::LaunchSeed<decltype(imported0)> session_seed_holder(session_config.enabled,state_file,
        [](const std::filesystem::path& p){return strix::deepseek41::load_attention_state(p);});
    std::string imported_path = state_file;
    std::map<std::uint32_t, RefStep> ref;
    std::vector<std::pair<std::string, std::vector<std::uint32_t>>> seqs;
    // phase names: a trailing 'w' (window phases only) turns window prefetch on for that phase
    const auto winpf_phase = [](const std::string& ph) {
        return !ph.empty() && ph.back() == 'w' &&
               (ph.rfind("wforce", 0) == 0 || ph.rfind("spec", 0) == 0 || ph.rfind("schats", 0) == 0);
    };
    const auto needed_extra64 = [&]() {  // rows above the imported position the phases need (the capacity), no wrap
        std::uint64_t extra = std::uint64_t{steps} + 16U;
        if (any_chats) extra = std::max(extra, std::uint64_t{chat_prompt} + chat_steps + 16U);
        return extra;
    };
    const auto needed_extra = [&]() {
        const std::uint64_t e = needed_extra64();
        if (e > 0x7fffffffULL) usage("--steps / --chat-prompt / --chat-steps too large");
        return static_cast<std::uint32_t>(e);
    };
    const auto prepare_request = [&]() {
    phases = split(phases_arg, ',');
    any_exact = any_chats = any_window = any_draft = false;
    for (const auto& ph : phases) {
        if (ph.rfind("exact", 0) == 0 || ph.rfind("pressure", 0) == 0) any_exact = true;
        else if (ph.rfind("wforce", 0) == 0) any_exact = any_window = true;
        else if (ph.rfind("spec", 0) == 0) any_exact = any_window = any_draft = true;
        else if (ph.rfind("schats", 0) == 0) any_chats = any_window = any_draft = true;
        else if (ph.rfind("chats", 0) == 0 || ph.rfind("dchats", 0) == 0) any_chats = true;
        else if (ph.rfind("wsmoke", 0) == 0) { any_window = true; if (layers_arg >= 40) usage("smoke phases need --layers < 40"); }
        else if (ph.rfind("gsmoke", 0) == 0) { any_window = true; if (layers_arg >= 40 || layers_arg < 3) usage("gsmoke needs --layers 3..39"); }
        else if (ph.rfind("csmoke", 0) == 0) {
            any_window = true;
            if (layers_arg >= 40 || layers_arg < 3) usage("csmoke needs --layers 3..39");
            if (!o.stream_hashes) usage("csmoke needs --stream-hashes");
        }
        else if (ph.rfind("smoke", 0) == 0) { if (layers_arg >= 40) usage("smoke phases need --layers < 40"); }
        else usage(("unknown phase " + ph).c_str());
    }
    if (max_rows == 0) max_rows = any_window ? 6U : 1U;
    if (max_rows > 8U) usage("--max-rows 1..8");
    if (any_window && max_rows < 2U) usage("window phases need --max-rows >= 2");
    if (any_draft && draft_file.empty()) usage("spec / schats phases need --draft");
    if (any_draft && max_rows > dspark_rows()) usage("--max-rows above the drafter's window (6)");
    if (any_exact && ref_file.empty()) usage("exact phases need --ref");
    if (layers_arg < 40 && (any_exact || any_chats)) usage("--layers < 40 runs smoke phases only");
    if (layers_arg < 40 && ref_file.empty()) usage("smoke phases need --ref (its fed tokens)");
    if (any_chats && (chats_arg.empty() || seqs_file.empty())) usage("chats phases need --chats and --seqs-file");
    // fixtures before any load: the reference covers every compared position, the chats exist
    if (g_serving || state_file != imported_path) {  // the worker re-reads it for every request (content, not path)
        session_seed_holder.before_replace(imported0,session_config.enabled,state_file);
        imported0 = strix::deepseek41::load_attention_state(state_file);
        session_seed_holder.after_replace(state_file);
        imported_path = state_file;
    }
    ref = ref_file.empty() ? std::map<std::uint32_t, RefStep>{} : read_ref(ref_file);
    if (!ref_file.empty())
        for (std::uint32_t i = 0; i < steps; ++i)
            if (!ref.count(static_cast<std::uint32_t>(imported0.position) + i)) usage("the reference log does not cover every step position");
    if (any_exact && steps < kQualifySteps) usage("exact phases need --steps >= 64 (qualification)");
    seqs.clear();
    if (any_chats) {
        std::ifstream sf(seqs_file);
        if (!sf) usage("cannot read --seqs-file");
        std::string line;
        while (std::getline(sf, line)) {
            const auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::vector<std::uint32_t> ids;
            for (const auto& v : split(line.substr(tab + 1), ',')) ids.push_back(static_cast<std::uint32_t>(std::stoul(v)));
            seqs.emplace_back(line.substr(0, tab), std::move(ids));
        }
        for (const auto& c : split(chats_arg, ','))
            if (std::stoul(c) >= seqs.size()) usage("a --chats index is outside --seqs-file");
    }
    for (const auto& ph_full : phases) {
        const std::string ph = winpf_phase(ph_full) ? ph_full.substr(0, ph_full.size() - 1) : ph_full;
        auto suffix = ph.rfind("dchats", 0) == 0 || ph.rfind("schats", 0) == 0 || ph.rfind("wforce", 0) == 0 || ph.rfind("wsmoke", 0) == 0 ||
                              ph.rfind("csmoke", 0) == 0 || ph.rfind("gsmoke", 0) == 0
                          ? ph.substr(6)
                      : ph.rfind("smokep", 0) == 0 ? ph.substr(6)
                      : ph.rfind("spec", 0) == 0 ? ph.substr(4)
                      : ph.rfind("exact", 0) == 0 || ph.rfind("chats", 0) == 0 || ph.rfind("smoke", 0) == 0 ? ph.substr(5) : std::string("0");
        if (ph.rfind("schats", 0) == 0 && suffix.find('f') != std::string::npos) {  // schats<K>[b<B>]f<N>: N fixed drafts
            const auto f = suffix.substr(suffix.find('f') + 1);
            if (f.empty() || f.find_first_not_of("0123456789") != std::string::npos || std::stoul(f) == 0 || std::stoul(f) > 5) usage(("bad phase " + ph).c_str());
            suffix.erase(suffix.find('f'));
        }
        if ((ph.rfind("chats", 0) == 0 || ph.rfind("dchats", 0) == 0 || ph.rfind("schats", 0) == 0) && suffix.find('b') != std::string::npos) suffix.erase(suffix.find('b'), 1);  // chats<K>b<B>
        if (ph.rfind("pressure", 0) == 0) suffix = ph.size() > 8 ? ph.substr(8) : std::string("0");
        if ((ph.rfind("wsmokep", 0) == 0 || ph.rfind("wforcep", 0) == 0)) suffix = ph.substr(7);  // window phases under pressure
        if ((suffix.empty() || suffix.find_first_not_of("0123456789") != std::string::npos)) usage(("bad phase " + ph).c_str());
        {  // K = the phase's first digits (prefixes carry none): within the opened max_predict and rows x K <= 64
            const auto d0 = ph.find_first_of("0123456789");
            const std::uint64_t kk = d0 == std::string::npos ? 0U : std::stoull(ph.substr(d0, ph.find_first_not_of("0123456789", d0) - d0));
            if (kk > std::max(predict, 8U) || std::uint64_t{max_rows} * kk > 64U) usage(("phase " + ph + ": K above max(--predict, 8) or max-rows x K above 64").c_str());
        }
    }
    };
    prepare_request();
    if (layers_arg == 40U) {
        if (receipt.empty()) usage("the full model needs --receipt (admitted SHA-256 of this file)");
        check_receipt(model, receipt);
        o.weight_read.primary_sha256 = strix::deepseek41::pinned_file(strix::deepseek41::PinnedFileKind::target_q2).sha256_hex;
    }
    std::ofstream js, tl;
    // a step's record (JSON line + timeline lines) is formatted and written while the next step runs (CudaStepIo::
    // after_launch), off the token loop's critical path; any other emit writes the pending record first (same order)
    std::function<void()> deferred;
    const auto flush_deferred = [&]() {
        if (!deferred) return;
        auto f = std::move(deferred);
        deferred = nullptr;
        f();
    };
    const auto open_outputs = [&]() {  // --json / --timeline (append), reopened for every request
        flush_deferred();              // a pending record belongs to the previous request's files
        if (js.is_open()) js.close();
        if (tl.is_open()) tl.close();
        js.clear();
        tl.clear();
        if (!json_out.empty()) js.open(json_out, std::ios::app);
        if (!timeline_out.empty()) tl.open(timeline_out, std::ios::app);
        if ((!json_out.empty() && !js) || (!timeline_out.empty() && !tl)) usage("cannot open --json / --timeline for writing");
    };
    open_outputs();
    std::function<std::string(const std::string&)> census_summary;
    const auto emit = [&](const std::string& input) {
        const std::string line = census_summary ? census_summary(input) : input;
        flush_deferred();
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
        if (js) js << line << "\n";
    };
    int failures = 0;
    emit("{\"event\":\"profile\"," + profile_json().substr(1));
    try {
        const auto t_load = std::chrono::steady_clock::now();
        const auto& imported = imported0;
        // capacity: --max-positions N (absolute; required with --serve: every request and its fresh-process reference
        // share it - it sizes the attention state and bounds the decode scan, part of the launch geometry), else what
        // the phases need above the imported position (as before)
        const auto required_positions = std::uint64_t{imported.position} + needed_extra();
        if (required_positions >= 0xffffffffULL) usage("request context capacity overflows positions");
        if (max_positions_arg != 0 && max_positions_arg < required_positions)
            usage("--max-positions below what the phases need (imported position + steps + 16 / chat prompt + steps + 16)");
        o.max_positions = max_positions_arg != 0 ? max_positions_arg : static_cast<std::uint32_t>(required_positions);
        o.end_layer = layers_arg;
        o.tiers.assign(40, ExpertTier::file);
        o.slots = 0;  // no expert bytes of its own (and no background expert loader): the MoE sites are TierSite's
        o.decode_rows = 8;
        o.decode_only = true;  // the replica imports its state and only decodes: no prefill buffers (~1.4 GiB at 2K)
        o.decode_graphs = true;
        o.timing = timing;
        o.timing_ops = false;
        // --vram-total-gib / --ram-total-gib / --budget-receipt: the device's use before this process's context (NVML),
        // then every fixed part measured as it lands (context, trunk, drafter, workspaces), then the planner
        std::unique_ptr<replica_budget::Meter> meter;
        if (caps.measuring()) {
            meter = std::make_unique<replica_budget::Meter>(budget_sample_ms);
            meter->begin(o.device);
        }
        CudaPrefill p(model, o);
        const double dense_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
        if (meter) {  // dense / trunk weights + attention state for --max-positions + the prefill object's buffers
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            meter->stage("trunk");
        }
        int target_device = 0;
        STRIX_CUDA_CHECK(cudaGetDevice(&target_device));
        // DSpark drafter (before the pinned tier: its weights pass through host memory once). On the target's device it
        // reads the target's token_embd / output; on another device it holds its own copies.
        std::unique_ptr<CudaDrafter> drafter;
        double draft_load_s = 0.0;
        const int ddev = draft_device < 0 ? target_device : draft_device;
        // the drafter: with --serve whenever --draft is given (requests choose their phases), else when a phase drafts
        const bool load_drafter = serving ? !draft_file.empty() : any_draft;
        if (load_drafter) {
            const auto t_d = std::chrono::steady_clock::now();
            auto lw = strix::ds41::dspark::load_weights(model, draft_file);
            CudaDrafterOptions dopt;
            if (ddev == target_device) {
                dopt.token_embd = reinterpret_cast<const std::uint16_t*>(p.device_tensor("token_embd.weight"));
                dopt.output_tiled = p.head_tiled();  // the target's tiled head (no raw copy)
            }
            dopt.keep = resolve_keep(emit);
            {
                DeviceScope ds(ddev);
                drafter = std::make_unique<CudaDrafter>(lw->weights, dopt);
            }
            if (route_stats) {
                DeviceScope ds(ddev);
                drafter->set_route_stats(true);
            }
            draft_load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_d).count();
            emit("{\"event\":\"drafter\",\"device\":" + std::to_string(ddev) + ",\"target_device\":" + std::to_string(target_device) +
                 ",\"device_bytes\":" + std::to_string(drafter->device_bytes()) + ",\"load_s\":" + std::to_string(draft_load_s) + "}");
        }

        if (meter) {
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            meter->stage("drafter");
        }
        auto reader = std::make_unique<GgufExpertReader>(model);
        auto shape = reader->shape();
        if (shape.top_k == 0) shape.top_k = 6;
        DeviceBuffer<float> hidden_dev, hidden_raw;
        if (load_drafter || o.stream_hashes) hidden_dev.resize(std::size_t{8} * 3U * shape.hidden);
        // the drafter on another GPU: the step's capture writes the rows straight into portable mapped pinned memory
        // (stores from the kernel, no copy engine: a D2H copy would queue behind the stream cache's writebacks) and the
        // drafter's GPU reads them from there
        struct HostRows {
            float* p = nullptr;
            ~HostRows() { if (p) cudaFreeHost(p); }
        } hidden_map;
        // STRIX_DS41_REPLICA_MAPPED_HIDDEN=1: the mapped capture also on one GPU (qualifies it in csmoke on a single GPU)
        const bool force_mapped = [] {
            const char* v = std::getenv("STRIX_DS41_REPLICA_MAPPED_HIDDEN");
            return v && v[0] == '1';
        }();
        if ((load_drafter && ddev != target_device) || (force_mapped && (load_drafter || o.stream_hashes)))
            STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hidden_map.p), std::size_t{8} * 3U * shape.hidden * sizeof(float),
                                           cudaHostAllocPortable | cudaHostAllocMapped));
        float* const hidden_cap = hidden_map.p ? hidden_map.p : hidden_dev.get();  // the speculative phases' capture
        if (o.stream_hashes) hidden_raw.resize(std::size_t{8} * 3U * 4U * shape.hidden);
        struct DecodeGraphReset {
            CudaPrefill& p;
            ~DecodeGraphReset() { p.clear_decode_graphs(); }
        };
        if (arena_auto || meter) p.reserve_decode_buffers();  // (budgets: the fixed decode buffers land before the measurement)
        if (meter) {  // hidden rows, decode ABI / boundary buffers, router counters, Engram rows
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            meter->stage("workspaces");
        }

        const std::uint32_t L = layers_arg, E = shape.experts;
        // --arena-gib = the experts kept on the device; the free-slot target comes on top of it
        std::uint32_t slots = static_cast<std::uint32_t>(arena_gib * 1073741824.0 / static_cast<double>(shape.expert_bytes())) + spares;
        const std::uint32_t spare_reserve = std::max(spares, pressure_spares);
        const std::uint64_t headroom_bytes = static_cast<std::uint64_t>(arena_headroom_gib * 1073741824.0);
        const std::uint64_t graph_reserve = 256ULL << 20U;
        const std::uint64_t bank_reserve = (2ULL * shape.top_k * max_rows + 1U) * shape.expert_bytes();
        const std::uint64_t site_reserve = 4ULL * (L + 1U) + 4ULL * L * 64U +
            4ULL * max_rows * E + 4ULL * 64U + moe_layer_workspace_bytes(max_rows, shape);
        std::uint64_t fixed_bytes = 0, free_at_arena = 0;
        if (arena_auto) {
            std::size_t free_b = 0, total_b = 0;
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            STRIX_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            free_at_arena = free_b;
            fixed_bytes = total_b - free_b;
            const auto reserve = headroom_bytes + graph_reserve + bank_reserve + site_reserve;
            if (free_b <= reserve) throw std::runtime_error("auto arena: no memory after fixed reserves and headroom");
            const auto available_slots = (free_b - reserve) / shape.expert_bytes();
            slots = static_cast<std::uint32_t>(std::min<std::uint64_t>(available_slots, std::uint64_t{L} * E + spare_reserve));
            if (slots <= spare_reserve) throw std::runtime_error("auto arena: no resident slots after spare reserve");
        }
        const std::uint32_t requested_slots = slots;
        // budgets before any big allocation; the three-tier plan (rtx::tier_plan) sizes the pinned tier, the clean-copy
        // share and the file tier. Budgets that cover every expert give today's numbers (and today's checks).
        strix::ds41::rtx::TierPlan tplan;
        std::optional<strix::ds41::rtx::BudgetPlan> bplan;  // the whole-device plan (budgets measured)
        const auto budget_caps_json = [&]() {
            std::ostringstream c;
            c << "\"vram_total_gib\":" << (caps.device_capped() ? caps.vram_total_gib : 0.0) << ",\"ram_total_gib\":" << (caps.host_capped() ? caps.ram_total_gib : 0.0)
              << ",\"vram_margin_gib\":" << caps.vram_margin_gib << ",\"ram_margin_gib\":" << caps.ram_margin_gib
              << ",\"device_cap_bytes\":" << (caps.device_capped() ? strix::ds41::rtx::budget_bytes(caps.vram_total_gib) : 0U)
              << ",\"host_cap_bytes\":" << (caps.host_capped() ? strix::ds41::rtx::budget_bytes(caps.ram_total_gib) : 0U);
            return c.str();
        };
        {
            std::size_t free_b = 0, total_b = 0;
            STRIX_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            const double have_host = host_available_gib();
            strix::ds41::rtx::TierPlanIn tin;
            tin.keys = std::uint64_t{L} * E;
            tin.unit_bytes = shape.expert_bytes();
            tin.residents = slots - (arena_auto ? spare_reserve : spares);
            tin.host_spares = host_spares;
            tin.ram_gib = ram_expert_gib;
            tin.inclusive_gib = ram_inclusive_gib;
            tin.host_avail_gib = have_host;
            tin.host_margin_gib = host_margin_gib;
            tin.engine_pinned_gib = static_cast<double>(p.pinned_bytes()) / 1073741824.0;
            tin.max_pinned_gib = max_pinned_gib;
            tin.extra_gib = 16.0 * static_cast<double>(shape.down_expert_bytes()) / 1073741824.0;  // load staging (8 workers x 2)
            tin.file_tier = file_tier && TierSite::can_disk_tier();
            tin.staging = disk_staging;
            if (meter) {  // tierstream's planner: the caps minus the measured fixed parts size the arena and the pinned tier
                namespace rtx = strix::ds41::rtx;
                rtx::BudgetPlanIn bi;
                bi.caps = caps;
                bi.device_fixed = meter->stages();  // context, trunk, drafter, workspaces (measured)
                const auto hm = replica_budget::host_mem();
                bi.host_pageable = {{"process_commit", hm.commit == replica_budget::kNone ? 0U : hm.commit}};
                bi.host_pinned = {{"engine_pinned", static_cast<std::uint64_t>(p.pinned_bytes())}};
                bi.units = tin.keys;
                bi.unit_bytes = tin.unit_bytes;
                bi.request_slots = slots;
                bi.spare_slots = arena_auto ? spare_reserve : spares;
                bi.min_residents = max_rows * shape.top_k + (spare_reserve - bi.spare_slots);
                bi.arena_extra_bytes = bank_reserve + site_reserve + graph_reserve;
                bi.host_request_blocks = rtx::tier_plan(tin).cap_units;  // today's RAM ceiling: a cap only lowers it
                bi.host_spares = host_spares;
                bi.staging = disk_staging;
                bi.host_extra_bytes = 16ULL * shape.down_expert_bytes();
                bi.file_tier = tin.file_tier;
                bi.inclusive_bytes = tin.inclusive_gib < 0.0 ? SS_BUDGET_AUTO : rtx::budget_bytes(tin.inclusive_gib);
                bi.inclusive_auto_cap_bytes = rtx::budget_bytes(tin.inclusive_auto_cap_gib);
                bplan = rtx::budget_plan(bi);
                emit("{\"event\":\"budget\",\"stage\":\"plan\"," + budget_caps_json() + ",\"plan\":{" + bplan->json() + "}}");
                if (bplan->refused()) throw std::runtime_error(bplan->refusal());
                if (caps.device_capped()) slots = static_cast<std::uint32_t>(bplan->out.arena_slots);
                tin = rtx::apply_budget(tin, *bplan);
            }
            const double need_dev = arena_auto ? double(std::uint64_t{slots} * shape.expert_bytes() + bank_reserve +
                site_reserve + graph_reserve + headroom_bytes) / 1073741824.0 :
                (static_cast<double>(slots) + 2.0 * shape.top_k * max_rows) * static_cast<double>(shape.expert_bytes()) / 1073741824.0 + 1.5;
            const double have_dev = static_cast<double>(free_b) / 1073741824.0;
            tplan = strix::ds41::rtx::tier_plan(tin);
            if (bplan) strix::ds41::rtx::check_budget_parity(tplan, *bplan);
            // host blocks + the two load bounce buffers + the engine's own pinned buffers (Engram rows, staging)
            const double need_host = tplan.need_host_gib, pinned_peak = tplan.pinned_peak_gib;
            std::ostringstream os;
            os << "{\"event\":\"budget\",\"engine_pinned_gib\":" << static_cast<double>(p.pinned_bytes()) / 1073741824.0 << ",\"pinned_peak_gib\":" << pinned_peak << ",\"max_pinned_gib\":" << max_pinned_gib << ",\"device_free_gib\":" << have_dev << ",\"device_need_gib\":" << need_dev << ",\"host_avail_gib\":" << have_host
               << ",\"host_need_gib\":" << need_host << "}";
            emit(os.str());
            std::ostringstream ts;  // the plan: GiB and experts per tier (also for 'auto')
            ts << "{\"event\":\"tier_plan\",\"vram_gib\":" << tplan.vram_gib << ",\"vram_experts\":" << tplan.resident << ",\"ram_gib\":" << tplan.ram_gib
               << ",\"ram_experts\":" << tplan.pinned << ",\"ram_auto\":" << (tplan.ram_auto ? "true" : "false") << ",\"file_gib\":" << tplan.file_gib
               << ",\"file_experts\":" << tplan.disk << ",\"file_tier\":" << (tplan.three_tier ? "true" : "false") << ",\"file_tier_allowed\":" << (tin.file_tier ? "true" : "false")
               << ",\"spare_blocks\":" << tplan.spare_blocks << ",\"inclusive_experts\":" << tplan.inclusive << ",\"inclusive_gib\":" << tplan.inclusive_gib
               << ",\"staging\":" << tplan.staging << ",\"ram_room_gib\":" << tplan.room_gib << ",\"disk_prefetch\":" << (disk_prefetch ? "true" : "false")
               << ",\"weight_dirs\":" << rs::json_str(o.weight_read.weight_dirs) << "}";
            emit(ts.str());
            if (file_tier && !TierSite::can_disk_tier()) std::fprintf(stderr, "file tier: the linked tierstream has no SS_DG_DISK_TIER; refusing as before\n");
            if (tplan.refuse) throw std::runtime_error(tplan.refuse);
            if (!arena_auto && have_dev < need_dev) throw std::runtime_error("not enough device memory for the arena");
            if (have_host < need_host) throw std::runtime_error("not enough available host memory for the pinned tier");
            if (pinned_peak > max_pinned_gib) throw std::runtime_error("the pinned peak exceeds --max-pinned-gib");
        }
        std::unique_ptr<TierSite> tier;
        bool tier_test_hooks = false;
        const std::uint64_t commit_before_tiers = meter ? replica_budget::host_mem().commit : 0U;
        {
            TierSiteOptions to;
            to.model = model;
            to.weight_read = o.weight_read;
            to.layers = L;
            to.slots = slots;
            to.spare_slots = std::max(spares, pressure_spares);  // slots - spare_slots experts start resident
            to.spare_target = spares;                            // kept free: exactly --spares
            to.host_spares = static_cast<std::uint32_t>(tplan.spare_blocks);  // --host-spares + the inclusive share
            to.inclusive_units = static_cast<std::uint32_t>(tplan.inclusive);
            if (tplan.three_tier) to.pinned_units = static_cast<std::uint32_t>(tplan.pinned);
            to.disk_prefetch = disk_prefetch;
            to.disk_staging = disk_staging;
            to.disk_queue = disk_queue;
            if (!routes.empty()) to.seed = seed_order(routes, seqmod, L, E);
            if (!residency_seed.empty()) {  // the last session's residency first (validated), then the routes' order
                std::vector<std::uint32_t> keys;
                try { keys = residency_keys(residency_seed, L, E); }
                catch (const std::exception& e) {
                    if (!residency_auto) usage(e.what());
                    std::fprintf(stderr, "residency: %s; using routes order\n", e.what());
                }
                const bool seeded_residency = !keys.empty();
                keys.insert(keys.end(), to.seed.begin(), to.seed.end());
                to.seed = std::move(keys);
                if (seeded_residency) emit("{\"event\":\"residency_seed\",\"keys\":" + std::to_string(to.seed.size()) + "}");
            }
            to.split = E / 2U;
            to.max_rows = max_rows;  // verify windows: banks and dg_max_ids of R rows (R x top_k <= 64)
            to.max_predict = std::max(predict, 8U);
            to.moe_v1b3 = moe_v1b3;
            to.flush_ready = flush_ready;
            to.prefetch_per_layer = pf_layer;
            to.prefetch_per_token = pf_token;
            to.wb_defer = wb_defer;
            to.ce_pacing = ce_pacing;
            to.wb_stream = wb_stream;
            if (!winpf_prior_file.empty()) {
                std::ifstream pf(winpf_prior_file, std::ios::binary);
                if (!pf) usage("cannot read --winpf-prior");
                to.prior.assign(std::istreambuf_iterator<char>(pf), std::istreambuf_iterator<char>());
            }
            to.test_hooks = test_hooks_arg;
            for (const auto& ph : phases)
                to.test_hooks = to.test_hooks || ph.rfind("pressure", 0) == 0 || ph.rfind("smokep", 0) == 0 || ph.rfind("wsmokep", 0) == 0 ||
                                ph.rfind("wforcep", 0) == 0;
            tier_test_hooks = to.test_hooks;
            std::size_t free_b = 0, total_b = 0;
            STRIX_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            const auto reserve = arena_auto ? headroom_bytes + graph_reserve + site_reserve : (std::uint64_t{3} << 29);
            if (free_b <= reserve) throw std::runtime_error("no memory after arena reserves");
            to.device_bytes = static_cast<std::uint64_t>(free_b) - reserve;
            to.host_bytes = static_cast<std::uint64_t>((max_pinned_gib - static_cast<double>(p.pinned_bytes()) / 1073741824.0) * 1073741824.0);
            tier = std::make_unique<TierSite>(*reader, to);
        }
        DecodeGraphReset graph_destinations{p};
        reader.reset();  // no buffered handle on the model file while decoding (Engram reads share the disk)
        // budgets: the arena's measured bytes, then every later sample against the caps (tierstream ss_budget_ledger:
        // ss_budget_ledger_observe / _within_caps; the ledger charges nothing here, it keeps the measured peaks)
        std::unique_ptr<ss_budget_ledger, void (*)(ss_budget_ledger*)> budget_ledger(nullptr, ss_budget_ledger_free);
        if (meter) {
            ss_budget_ledger* l = nullptr;
            if (ss_budget_ledger_create(caps.device_capped() ? strix::ds41::rtx::budget_bytes(caps.vram_total_gib) : 0U,
                                        caps.host_capped() ? strix::ds41::rtx::budget_bytes(caps.ram_total_gib) : 0U, &l) != SS_OK)
                throw std::runtime_error(std::string("budget: ") + ss_last_error());
            budget_ledger.reset(l);
        }
        std::uint64_t unseen_pinned = 0;   // pinned bytes the process commit does not show (added to every host sample)
        bool pinned_in_commit = true;
        std::string budget_load_detail;    // the load event's breakdown (declared + measured)
        if (meter) {
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            meter->stage("arena");
            const auto hm = replica_budget::host_mem();
            const std::uint64_t tier_pinned = tier->host_bytes();
            const std::uint64_t seen = hm.commit != replica_budget::kNone && hm.commit > commit_before_tiers ? hm.commit - commit_before_tiers : 0U;
            pinned_in_commit = static_cast<double>(seen) >= 0.9 * static_cast<double>(tier_pinned);
            if (!pinned_in_commit) unseen_pinned = tier_pinned + static_cast<std::uint64_t>(p.pinned_bytes());
            const std::string d = p.describe();
            const auto gb = [&](const char* key) -> std::uint64_t {  // describe()'s GB (1e9) figures, as bytes
                const std::string k = std::string("\"") + key + "\":";
                const auto at = d.find(k);
                if (at == std::string::npos) return 0U;
                try { return static_cast<std::uint64_t>(std::stod(d.substr(at + k.size())) * 1e9); } catch (const std::exception&) { return 0U; }
            };
            std::ostringstream os;
            os << ",\"measured_device\":{";
            for (std::size_t i = 0; i < meter->stages().size(); ++i)
                os << (i ? "," : "") << rs::json_str(meter->stages()[i].first) << ":" << meter->stages()[i].second;
            os << "},\"declared_device\":{\"dense\":" << gb("dense_gb") << ",\"attention_state\":" << gb("state_gb") << ",\"prefill_buffers\":" << gb("buffers_gb")
               << ",\"batch\":" << gb("batch_gb") << ",\"drafter\":" << (drafter ? drafter->device_bytes() : 0U)
               << ",\"arena_library\":" << tier->device_bytes() << ",\"arena_slots\":" << std::uint64_t{slots} * shape.expert_bytes()
               << ",\"bank_reserve\":" << bank_reserve << ",\"site_reserve\":" << site_reserve << ",\"graph_reserve\":" << graph_reserve << "}"
               << ",\"max_positions\":" << o.max_positions << ",\"draft_device\":" << ddev << ",\"target_device\":" << target_device
               << ",\"measured_host\":{\"commit_before_tiers\":" << commit_before_tiers << ",\"commit_after_tiers\":" << (hm.commit == replica_budget::kNone ? 0U : hm.commit)
               << "},\"declared_host\":{\"pinned_tiers\":" << tier_pinned << ",\"engine_pinned\":" << p.pinned_bytes() << "}"
               << ",\"tiers\":{\"arena_slots\":" << slots << ",\"requested_slots\":" << requested_slots << ",\"vram_experts\":" << tplan.resident
               << ",\"pinned_experts\":" << tplan.pinned << ",\"pinned_blocks\":" << (tplan.pinned + tplan.spare_blocks + tplan.staging)
               << ",\"spare_blocks\":" << tplan.spare_blocks << ",\"staging\":" << tplan.staging << ",\"file_experts\":" << tplan.disk
               << ",\"inclusive_experts\":" << tplan.inclusive << "}";
            if (bplan) os << ",\"plan\":{" << bplan->json() << "}";
            budget_load_detail = os.str();
        }
        // one sample against the caps; the 'budget' event (stage load / warmup / request / run). false: a peak above a cap
        const auto budget_point = [&](const std::string& stage, const std::string& name) -> bool {
            if (!meter) return true;
            constexpr auto none = replica_budget::kNone;
            const auto pt = meter->sample();
            const std::uint64_t host_now = pt.host_commit == none ? none : pt.host_commit + unseen_pinned;
            const std::uint64_t host_peak = pt.host_peak == none ? none : pt.host_peak + unseen_pinned;
            static_assert(replica_budget::kNone == SS_BUDGET_NO_SAMPLE, "no sample = UINT64_MAX on both sides");
            strix::ds41::rtx::budget_check(ss_budget_ledger_observe(budget_ledger.get(), stage.c_str(), pt.charged, host_peak));
            const auto sp = meter->sampler_peak();
            if (sp != none) strix::ds41::rtx::budget_check(ss_budget_ledger_observe(budget_ledger.get(), "nvml-sampler", sp, none));
            const bool ok = ss_budget_ledger_within_caps(budget_ledger.get()) != 0;
            ss_budget_stats bs{};
            bs.struct_size = static_cast<std::uint32_t>(sizeof bs);
            strix::ds41::rtx::budget_check(ss_budget_ledger_stats(budget_ledger.get(), &bs));
            const std::uint64_t dcap = bs.kind[SS_BUDGET_DEVICE].cap, hcap = bs.kind[SS_BUDGET_PINNED].cap;
            const auto opt = [&](std::uint64_t v) { return v == none ? std::string("null") : std::to_string(v); };
            const auto gib = [](std::uint64_t v) { return static_cast<double>(v) / 1073741824.0; };
            std::ostringstream os;
            os << "{\"event\":\"budget\",\"stage\":" << rs::json_str(stage);
            if (!name.empty()) os << ",\"name\":" << rs::json_str(name);
            os << "," << budget_caps_json() << ",\"device_baseline\":\"" << meter->baseline() << "\",\"device_before_bytes\":" << meter->before()
               << ",\"device_total_bytes\":" << pt.device_total << ",\"device_used_bytes\":" << pt.device_used << ",\"device_charged_bytes\":" << pt.charged
               << ",\"device_charged_gib\":" << gib(pt.charged) << ",\"nvml_used_bytes\":" << opt(pt.nvml_used) << ",\"nvml_process_bytes\":" << opt(pt.nvml_process)
               << ",\"host_commit_bytes\":" << opt(host_now) << ",\"host_commit_gib\":" << (host_now == none ? 0.0 : gib(host_now))
               << ",\"pinned_in_commit\":" << (pinned_in_commit ? "true" : "false") << ",\"unseen_pinned_bytes\":" << unseen_pinned
               << ",\"device_peak_bytes\":" << bs.measured_device_peak << ",\"device_peak_gib\":" << gib(bs.measured_device_peak)
               << ",\"device_peak_at\":" << rs::json_str(bs.measured_device_peak_at) << ",\"host_peak_bytes\":" << bs.measured_host_peak
               << ",\"host_peak_gib\":" << gib(bs.measured_host_peak) << ",\"host_peak_at\":" << rs::json_str(bs.measured_host_peak_at)
               << ",\"sampler_ms\":" << meter->sample_ms() << ",\"sampler_samples\":" << meter->sampler_samples() << ",\"sampler_peak_bytes\":" << opt(sp)
               << ",\"sampler_process_peak_bytes\":" << opt(meter->sampler_process_peak()) << ",\"device_over_samples\":" << bs.device_over
               << ",\"host_over_samples\":" << bs.host_over
               << ",\"device_peak_le_cap\":" << (!dcap || (bs.measured_device_peak <= dcap && !bs.device_over) ? "true" : "false")
               << ",\"host_peak_le_cap\":" << (!hcap || (bs.measured_host_peak <= hcap && !bs.host_over) ? "true" : "false")
               << ",\"peak_le_cap\":" << (ok ? "true" : "false");
            if (stage == "load") os << budget_load_detail;
            os << "}";
            emit(os.str());
            if (!ok) std::fprintf(stderr, "budget: a measured peak exceeds a total cap at %s (device peak %.2f GiB at %s, host peak %.2f GiB at %s)\n", stage.c_str(),
                                  gib(bs.measured_device_peak), bs.measured_device_peak_at, gib(bs.measured_host_peak), bs.measured_host_peak_at);
            return ok;
        };
        const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
        {
            std::ostringstream os;
            os << "{\"event\":\"loaded\",\"cache\":\"" << cache_kind << "\",\"describe\":" << p.describe() << ",\"arena_slots\":" << slots << ",\"device_experts\":" << tier->device_experts()
               << ",\"arena_auto\":" << (arena_auto ? "true" : "false")
               << ",\"arena_requested_slots\":" << requested_slots << ",\"arena_effective_slots\":" << tier->device_bytes() / shape.expert_bytes()
               << ",\"arena_fixed_bytes\":" << fixed_bytes << ",\"arena_free_bytes\":" << free_at_arena
               << ",\"arena_headroom_bytes\":" << (arena_auto ? headroom_bytes : (std::uint64_t{3} << 29))
               << ",\"arena_graph_reserve_bytes\":" << (arena_auto ? graph_reserve : 0U)
               << ",\"arena_site_reserve_bytes\":" << (arena_auto ? site_reserve : 0U)
               << ",\"arena_bank_reserve_bytes\":" << bank_reserve << ",\"arena_spare_slots\":" << spare_reserve
               << ",\"arena_gb\":" << static_cast<double>(tier->device_bytes()) / 1e9 << ",\"host_gb\":" << static_cast<double>(tier->host_bytes()) / 1e9
               << ",\"spares\":" << spares << ",\"predict\":" << predict << ",\"prefetch_layer\":" << pf_layer << ",\"prefetch_token\":" << pf_token
               << ",\"dense_s\":" << dense_s << ",\"load_s\":" << load_s << "}";
            emit(os.str());
        }
        if (!budget_point("load", "") && caps.capped())  // enforcement: never serve above a cap
            throw std::runtime_error("budget: the measured use at load exceeds a total cap (see the budget event; raise --vram-margin-gib / --ram-margin-gib or the cap)");
        std::uint64_t census_nonrouted = 0, census_embedding_row = 0, census_head = 0, census_draft_nonrouted = 0, census_draft_expert = 0;
        if (census || serving) {
            const auto manifest = strix::build_model_manifest(strix::gguf::read_file(model));
            for (const auto& t : manifest.tensors) {
                if (t.layer && *t.layer >= L) continue;
                if (t.role == strix::TensorRole::embedding) {
                    if (t.dimensions.size() > 1) census_embedding_row += t.byte_size / t.dimensions.back();
                } else if (t.role == strix::TensorRole::output_head) census_head += t.byte_size;
                else if (t.role != strix::TensorRole::moe_expert && t.role != strix::TensorRole::ngram_table && t.role != strix::TensorRole::mtp)
                    census_nonrouted += t.byte_size;
            }
            if (!draft_file.empty()) {
                const auto dm = strix::build_model_manifest(strix::gguf::read_file(draft_file));
                for (const auto& t : dm.tensors) {
                    if (t.name.find("_exps.weight") != std::string::npos) census_draft_expert += t.byte_size / 128;
                    else if (t.name != "token_embd.weight" && t.name != "output.weight") census_draft_nonrouted += t.byte_size;
                }
            }
        }
        // the released drafter (pinned name and byte count): its pinned SHA-256, no 8 GB read per load; others are hashed
        bool census_draft_pinned = false;
        const std::string census_draft_hash = [&]() -> std::string {
            if (!(census || serving) || draft_file.empty()) return {};
            const auto& pin = strix::deepseek41::pinned_file(strix::deepseek41::PinnedFileKind::draft_mxfp4);
            std::error_code ec;
            const std::filesystem::path dp(draft_file);
            if (dp.filename().string() == pin.file_name && std::filesystem::file_size(dp, ec) == pin.bytes && !ec) {
                census_draft_pinned = true;
                return std::string(pin.sha256_hex);
            }
            return census_file_hash(draft_file);
        }();
        const std::string census_binary_hash = (census || serving) ? census_file_hash(argv[0]) : std::string();
        std::uint64_t census_request = 0;
        std::size_t census_used = 0;
        std::vector<ReplicaCensus> census_records;
        std::vector<std::string> census_phases;
        std::string census_phase;
        std::chrono::steady_clock::time_point census_last_t0{};
        const auto census_capture = [&](std::uint32_t pos, std::uint32_t R, const auto& layers, const TierTokenStats& st,
                                        double device_ms, double boundary_ms, double engram_wait_ms) {
            if (!census || !census_request) return std::size_t(-1);
            if (census_used == census_records.size()) throw std::runtime_error("census preallocated capacity exceeded");
            const auto id = census_used++;
            auto& c = census_records[id];
            c = ReplicaCensus{};
            c.request_id = census_request; c.cycle_id = id; c.pos = pos; c.R = R;
            c.cycle_ms = -1;
            c.traffic = st; c.target_device_ms = device_ms; c.boundary_ms = boundary_ms; c.engram_wait_ms = engram_wait_ms;
            c.expert_bytes = shape.expert_bytes();
            c.target_nonrouted_bytes = census_nonrouted + census_embedding_row * R + (L == 40 ? census_head : 0);
            census_cache_payload(c, strix::ds41::attn::released_plan(), L);
            std::copy_n(layers.begin(), std::min<std::size_t>(40, layers.size()), c.layers.begin());
            census_phases[id] = census_phase;
            return id;
        };
        census_summary = [&](const std::string& line) {
            if (!census || (line.find("\"event\":\"summary\"") == std::string::npos &&
                            line.find("\"event\":\"spec_summary\"") == std::string::npos &&
                            line.find("\"event\":\"gate\"") == std::string::npos)) return line;
            const auto at = line.find("\"phase\":\"");
            if (at == std::string::npos) return line;
            const auto begin = at + 9, end = line.find('"', begin);
            const auto phase = line.substr(begin, end - begin);
            const bool gate = line.find("\"event\":\"gate\"") != std::string::npos;
            if (census_used && census_records[census_used - 1].cycle_ms < 0 && census_last_t0 != std::chrono::steady_clock::time_point{})
                census_records[census_used - 1].cycle_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - census_last_t0).count();
            ReplicaCensusTotals totals;
            std::ostringstream out;
            out << line.substr(0, line.size() - 1) << ",\"census_cycles\":[";
            bool first = true;
            for (std::size_t i = 0; i < census_used; ++i) if (census_phases[i] == phase || (gate && census_phases[i] == census_phase)) {
                totals.add(census_records[i]);
                if (!first) out << ',';
                first = false;
                census_records[i].write(out, census_phases[i]);
            }
            out << "],\"census\":";
            totals.write(out);
            out << '}';
            return out.str();
        };
        struct CensusSummaryLifetime {
            std::function<std::string(const std::string&)>& fn;
            ~CensusSummaryLifetime() { fn = {}; }
        } census_summary_lifetime{census_summary};
        std::uint32_t spare_now = spares;  // the spare target the library applies at the next boundary (recorded per step)
        // pressure phases: the spare target and the library's pressure knob (tierstream keeps its timeline always)
        const auto set_pressure = [&](std::uint32_t spare_target, std::uint32_t free, std::uint32_t recent) {
            spare_now = spare_target;
            tier->set_budgets(spare_target, pf_layer, pf_token);
            tier->set_pressure(free, recent);
        };

        CudaStepIo io;
        io.moe_site_graphs = true;
        io.after_launch = [&]() { flush_deferred(); };
        std::vector<float> stream_out;
        std::uint64_t last_stream_fnv = 0;
        const std::size_t row_floats = std::size_t{4} * shape.hidden;
        const auto row_fnv = [&](std::uint32_t r) {
            std::uint64_t h = 1469598103934665603ULL;
            const auto* bytes = reinterpret_cast<const unsigned char*>(stream_out.data() + std::size_t{r} * row_floats);
            for (std::size_t i = 0; i < row_floats * 4U; ++i) h = (h ^ bytes[i]) * 1099511628211ULL;
            return h;
        };
        if (L < 40U) {
            stream_out.resize(std::size_t{max_rows} * row_floats);
            io.stream_out = stream_out.data();
            io.logits = false;
        }
        io.moe_site = [&](const CudaMoeSite& s) {
            tier->enqueue(s.layer, s.rows, s.x, s.ids, s.weights, s.shared, s.raw, s.y, s.shared_expert, s.route_other, s.stream,
                          s.shared_join, s.hc_residual, s.hc_mixes, s.hc_out, s.hc_pre_carry);
        };
        std::map<std::uint32_t, std::uint32_t> epoch_of;  // predict K -> graph epoch
        const auto set_predict = [&](std::uint32_t k) {
            tier->set_predict(k);
            if (!epoch_of.count(k)) epoch_of[k] = static_cast<std::uint32_t>(epoch_of.size() + 1U);
            io.moe_site_epoch = epoch_of[k];
        };
        // rare-path counters of a phase (pressure must show them)
        std::uint64_t ph_bank = 0, ph_res = 0, ph_mid = 0;
        std::chrono::steady_clock::time_point last_t0{};
        bool have_last = false;  // reset at every phase start (a cycle never spans two phases)
        bool prev_counted = false;
        std::string prev_phase;
        struct Agg {
            std::vector<double> ms, step_ms, dev, bnd, miss, mlayers, covered, fetched, pref, recall, ring, evicted, ewait, cycle;
        };
        std::map<std::string, Agg> agg;
        std::uint32_t compared = 0, matched = 0;
        // STRIX_DS41_CUDA_ENGRAM_CHECK=1: every step with file Engram rows must have re-read its records equal (else the run fails)
        // the check runs in the qualification phases only (exact / pressure / wforce / spec / smoke), never in the timed
        // chats (its re-reads would be timed); set per phase below
        bool engram_check_env = false;  // read per request (configure_request)
        bool engram_check_on = false;
        const auto engram_gate = [&](const CudaStepResult& s) {
            if (engram_check_on && (s.engram_check_mismatches != 0 || (s.engram_checked == 0 && o.engram == "file")))
                throw std::runtime_error("Engram re-read check: " + std::to_string(s.engram_check_mismatches) + " mismatches of " + std::to_string(s.engram_checked));
        };
        // Optional qualification observer; empty in production requests.
        std::ofstream ordinary_logits,ordinary_tokens,ordinary_greedy;
        const auto ordinary_rows=[&](const std::vector<std::uint32_t>& rows,const CudaStepResult& result) {
            if(!ordinary_logits.is_open())return;
            if(result.logits.size()!=rows.size()*129280 || result.tokens.size()!=rows.size())
                throw std::runtime_error("qualification dump needs full raw logits");
            ordinary_logits.write(reinterpret_cast<const char*>(result.logits.data()),static_cast<std::streamsize>(result.logits.size()*sizeof(float)));
            ordinary_tokens.write(reinterpret_cast<const char*>(rows.data()),static_cast<std::streamsize>(rows.size()*sizeof(std::uint32_t)));
            ordinary_greedy.write(reinterpret_cast<const char*>(result.tokens.data()),static_cast<std::streamsize>(result.tokens.size()*sizeof(std::uint32_t)));
            if(!ordinary_logits||!ordinary_tokens||!ordinary_greedy)throw std::runtime_error("qualification dump write failed");
        };
        const auto do_step = [&](const std::vector<std::uint32_t>& fed, const std::string& phase, std::uint32_t k, bool count, bool compare) {
            const std::uint32_t pos = p.position();
            const auto t0 = std::chrono::steady_clock::now();
            // full loop: step start to the next step start (everything the host does per token, checks included)
            const double cycle = have_last ? std::chrono::duration<double, std::milli>(t0 - last_t0).count() : -1.0;
            if (census && census_used && census_phases[census_used - 1] == phase) census_records[census_used - 1].cycle_ms = cycle;
            census_last_t0 = t0;
            last_t0 = t0;
            have_last = true;
            tier->begin_token();
            const auto s = p.step(fed, io);
            ordinary_rows(fed,s);
            engram_gate(s);
            if (!stream_out.empty()) last_stream_fnv = row_fnv(0);
            const auto t1 = std::chrono::steady_clock::now();
            tier->end_token(static_cast<std::uint32_t>(fed.size()));  // completed rows; service + boundary
            const auto t2 = std::chrono::steady_clock::now();
            // the token's stats and timeline, outside the timed boundary
            const std::vector<TierLayer> layers = tier->collect();
            const TierTokenStats st = tier->stats();
            census_phase = phase;
            const auto census_id = census_capture(pos, static_cast<std::uint32_t>(fed.size()), layers, st, s.device_ms,
                std::chrono::duration<double, std::milli>(t2 - t1).count(), s.engram_wait_seconds * 1e3);
            if (census) {
                auto& c = census_records[census_id];
                c.engram_wait_first_ms = s.engram_wait_first_seconds * 1e3;
                c.engram_ms = s.engram_seconds * 1e3; c.engram_issue_ms = s.engram_issue_seconds * 1e3;
                c.committed = static_cast<std::uint32_t>(fed.size());
                c.counted = count && fed.size() == 1U;
                if (census_id && census_phases[census_id - 1] == phase) census_records[census_id - 1].cycle_ms = cycle;
            }
            ph_bank += st.ring;
            ph_res += st.resurrected;
            ph_mid += st.collected_mid;
            const double wall = std::chrono::duration<double, std::milli>(t2 - t0).count();
            const double bnd = std::chrono::duration<double, std::milli>(t2 - t1).count();
            const double recall = st.predicted ? static_cast<double>(st.predicted_true) / static_cast<double>(std::max(1U, st.lookups - (layers.empty() ? 0U : layers.front().ids))) : 0.0;
            // the reference check now (the gates count it); its record later
            bool ref_found = false, ok = false;
            RefStep ref_step;
            if (compare) {
                const auto it = ref.find(pos);
                if (it != ref.end() && !s.tokens.empty() && !s.logits_fnv.empty()) {
                    ref_found = true;
                    ref_step = it->second;
                    ok = it->second.fed == fed.back() && it->second.argmax == s.tokens.back() && it->second.fnv == s.logits_fnv.back();
                }
                ++compared;
                matched += ok ? 1U : 0U;
            }
            if (census && compare && !ok) { census_records[census_id].committed = 0; census_records[census_id].errors = 1; }
            // the record: everything it prints, by value (not the step's logits)
            struct StepRec {
                std::uint32_t argmax, graph_captures, engram_checked, engram_check_mismatches;
                std::string logits_fnv;
                std::vector<std::uint64_t> stream_hashes;
                double seconds, enqueue_seconds, engram_seconds, engram_wait_seconds;
                float device_ms;
                bool engram_prefetched;
                std::vector<CudaPrefillLayerTiming> layers;
            };
            StepRec sr{s.tokens.empty() ? 0U : s.tokens.back(), static_cast<std::uint32_t>(s.graph_captures),
                       static_cast<std::uint32_t>(s.engram_checked), static_cast<std::uint32_t>(s.engram_check_mismatches),
                       s.logits_fnv.empty() ? std::string() : hex(s.logits_fnv.back()), s.stream_hashes, s.seconds, s.enqueue_seconds, s.engram_seconds,
                       s.engram_wait_seconds, s.device_ms, s.engram_prefetched, tl ? s.layers : std::vector<CudaPrefillLayerTiming>{}};
            const std::uint64_t stream_fnv = last_stream_fnv;
            const bool has_stream = !stream_out.empty();
            const std::uint32_t spare_rec = spare_now, rows_fed = static_cast<std::uint32_t>(fed.size()), fed_last = fed.back();
            deferred = [&, census_id, phase, k, pos, rows_fed, fed_last, wall, cycle, bnd, recall, st, layers, sr = std::move(sr), stream_fnv, has_stream, spare_rec, compare,
                        ref_found, ok, ref_step]() {
                std::ostringstream os;
                os << "{\"event\":\"" << phase << "\",\"k\":" << k << ",\"pos\":" << pos << ",\"rows\":" << rows_fed << ",\"fed\":" << fed_last
                   << ",\"argmax\":" << sr.argmax << ",\"logits_fnv\":\"" << sr.logits_fnv
                   << "\",\"ms\":" << wall << ",\"cycle_ms\":" << cycle << ",\"step_ms\":" << sr.seconds * 1e3 << ",\"boundary_ms\":" << bnd << ",\"enqueue_ms\":" << sr.enqueue_seconds * 1e3
                   << ",\"device_ms\":" << sr.device_ms << ",\"engram_ms\":" << sr.engram_seconds * 1e3 << ",\"engram_wait_ms\":" << sr.engram_wait_seconds * 1e3
                   << ",\"engram_prefetched\":" << (sr.engram_prefetched ? "true" : "false") << ",\"engram_checked\":" << sr.engram_checked
                   << ",\"engram_check_mismatches\":" << sr.engram_check_mismatches << ",\"graph_captures\":" << sr.graph_captures << ",\"lookups\":" << st.lookups
                   << ",\"misses\":" << st.misses << ",\"miss_layers\":" << st.miss_layers << ",\"covered\":" << st.covered << ",\"fetched\":" << st.fetched
                   << ",\"prefetched\":" << st.prefetched << ",\"predicted\":" << st.predicted << ",\"predicted_true\":" << st.predicted_true << ",\"recall\":" << recall
                   << ",\"resurrected\":" << st.resurrected << ",\"collected_mid\":" << st.collected_mid << ",\"bank\":" << st.ring << ",\"stalled\":" << st.stalled << ",\"evicted\":" << st.evicted
                   << ",\"writebacks_done\":" << st.writebacks_done << ",\"free_slots\":" << st.free_slots << ",\"spare_target\":" << spare_rec;
                if (census && census_id != std::size_t(-1)) {
                    os << ",\"census\":"; census_records[census_id].write(os, census_phases[census_id]);
                    os << ",\"cycle_prev_id\":";
                    if (census_id && cycle >= 0) os << census_id - 1; else os << "null";
                }
                if (has_stream) os << ",\"stream_fnv\":\"" << hex(stream_fnv) << "\"";
                if (!sr.stream_hashes.empty()) {  // --stream-hashes: the last row's stream after each layer (driver FNV)
                    os << ",\"stream_hashes\":[";
                    for (std::size_t i = 0; i < sr.stream_hashes.size(); ++i) os << (i ? "," : "") << "\"" << hex(sr.stream_hashes[i]) << "\"";
                    os << "]";
                }
                if (compare) {
                    if (ref_found) os << ",\"ref_argmax\":" << ref_step.argmax << ",\"ref_fnv\":\"" << hex(ref_step.fnv) << "\"";
                    os << ",\"match\":" << (ok ? "true" : "false");
                }
                os << "}";
                emit(os.str());
                if (tl)
                    for (std::uint32_t l = 0; l < layers.size(); ++l) {
                        const auto& t = layers[l];
                        tl << "{\"phase\":\"" << phase << "\",\"k\":" << k << ",\"pos\":" << pos << ",\"layer\":" << l << ",\"ids\":" << t.ids << ",\"misses\":" << t.misses
                           << ",\"covered\":" << t.covered << ",\"fetched\":" << t.fetched << ",\"prefetched\":" << t.prefetched << ",\"predicted_true\":" << t.predicted_true
                           << ",\"seen_us\":" << t.seen_us << ",\"serve_us\":" << t.serve_us << ",\"demand_us\":" << t.demand_us
                           << ",\"pred_wait_us\":" << t.pred_wait_us << ",\"prefetch_us\":" << t.prefetch_us << ",\"dma_ms\":" << t.dma_ms;
                        if (l < sr.layers.size()) tl << ",\"layer_ms\":" << sr.layers[l].total_ms << ",\"attn_ms\":" << sr.layers[l].attention_ms << ",\"moe_ms\":" << sr.layers[l].moe_ms;
                        tl << "}\n";
                    }
            };
            if (records_inline) flush_deferred();
            // cycle[k] = start[k] - start[k-1] is token k-1's full loop: credit it to the previous token
            if (cycle > 0 && prev_counted) agg[prev_phase].cycle.push_back(cycle);
            prev_counted = count && fed.size() == 1U;
            prev_phase = phase;
            if (count && fed.size() == 1U) {
                auto& g = agg[phase];
                g.ms.push_back(wall);
                g.step_ms.push_back(s.seconds * 1e3);
                g.dev.push_back(s.device_ms);
                g.bnd.push_back(bnd);
                g.miss.push_back(st.misses);
                g.mlayers.push_back(st.miss_layers);
                g.covered.push_back(st.covered);
                g.fetched.push_back(st.fetched);
                g.pref.push_back(st.prefetched);
                g.recall.push_back(recall);
                g.ring.push_back(st.ring);
                g.evicted.push_back(st.evicted);
                g.ewait.push_back(s.engram_wait_seconds * 1e3);

            }
            return s.tokens.empty() ? fed.back() : s.tokens.back();
        };

        // ---- verify windows -----------------------------------------------------------------------------------------
        // One window of R = rows.size() rows through the stream site; the caller decides `kept` from the greedy tokens
        // (commit_window(kept, next)). Returns the step result; the stats / times of the window in `wt`.
        struct WindowTimes {
            double step_ms = 0, finish_ms = 0, boundary_ms = 0;
            double copy_ms = -1;  // the least device time of a layer that fetched exactly one unit (copy + ready)
            TierTokenStats st;
            std::size_t census_id = std::size_t(-1);
        };
        // winpf: rows of the next window already hinted (the bonus row right after a commit, the drafts after the draft)
        std::uint32_t hinted_rows = 0;
        const auto hint_rows = [&](const std::uint32_t* t, std::uint32_t n) {
            if (n == 0 || !tier->winpf()) return;
            tier->hint_window(t, n);
            hinted_rows += n;
        };
        const auto run_window = [&](const std::vector<std::uint32_t>& rows, WindowTimes& wt) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto window_pos = static_cast<std::uint32_t>(p.position());
            if (census && census_used && census_phases[census_used - 1] == census_phase && census_last_t0 != std::chrono::steady_clock::time_point{})
                census_records[census_used - 1].cycle_ms = std::chrono::duration<double, std::milli>(t0 - census_last_t0).count();
            census_last_t0 = t0;
            // winpf: the rows not hinted yet (all of them outside the speculative loop); no-op when off
            if (rows.size() > hinted_rows) hint_rows(rows.data() + hinted_rows, static_cast<std::uint32_t>(rows.size()) - hinted_rows);
            hinted_rows = 0;
            tier->begin_token();
            auto s = p.step(rows, io);
            ordinary_rows(rows,s);
            const auto t1 = std::chrono::steady_clock::now();
            tier->end_token(static_cast<std::uint32_t>(rows.size()));  // completed verify rows; service + boundary
            const auto t2 = std::chrono::steady_clock::now();
            const std::vector<TierLayer> layers = tier->collect();  // the window's stats, outside the timed boundary (as do_step)
            for (const auto& l : layers)
                if (l.fetched == 1U && l.dma_ms > 0 && (wt.copy_ms < 0 || l.dma_ms < wt.copy_ms)) wt.copy_ms = l.dma_ms;
            wt.step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            wt.finish_ms = 0;
            wt.boundary_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            wt.st = tier->stats();
            wt.census_id = census_capture(window_pos, static_cast<std::uint32_t>(rows.size()), layers,
                wt.st, s.device_ms, wt.boundary_ms, s.engram_wait_seconds * 1e3);
            if (census && wt.census_id != std::size_t(-1)) {
                auto& c = census_records[wt.census_id];
                c.engram_wait_first_ms = s.engram_wait_first_seconds * 1e3;
                c.engram_ms = s.engram_seconds * 1e3; c.engram_issue_ms = s.engram_issue_seconds * 1e3;
            }
            ph_bank += wt.st.ring;
            ph_res += wt.st.resurrected;
            ph_mid += wt.st.collected_mid;
            engram_gate(s);
            return s;
        };
        // The committed rows' drafter input (captured by the step in hidden_dev) appended to the drafter.
        // the next window's row 0 Engram rows on the I/O thread during the append / draft (CudaPrefill::prefetch_engram);
        // STRIX_DS41_REPLICA_WINDOW_ENGRAM_PREFETCH=0 turns it off (A/B)
        bool window_engram_prefetch = true;  // read per request (configure_request)
        const auto prefetch_row0 = [&](std::uint32_t token) {
            if (window_engram_prefetch) p.prefetch_engram(token);
        };
        const auto prefetch_ref_row0 = [&]() {  // smokes: the next window feeds the reference's token at p.position()
            if (const auto it = ref.find(p.position()); it != ref.end()) prefetch_row0(it->second.fed);
        };
        const auto append_committed = [&](std::uint32_t kept) {
            if (!hidden_map.p) {
                drafter->append_hidden_device(hidden_dev.get(), kept, nullptr);
            } else {
                DeviceScope ds(ddev);
                drafter->append_hidden_device(hidden_map.p, kept, nullptr);  // the step is complete: its stores are visible
            }
            if (census && census_used) census_records[census_used - 1].append_hidden_bytes += std::uint64_t(kept) * 3U * shape.hidden * 4U;
        };
        // --qual: every row of a window: fed tokens, greedy tokens, logits FNV (when the step computed logits)
        const auto qual_rows = [&](const std::vector<std::uint32_t>& rows, const CudaStepResult& s) {
            std::ostringstream q;
            q << ",\"fed_rows\":[";
            for (std::size_t j = 0; j < rows.size(); ++j) q << (j ? "," : "") << rows[j];
            q << "],\"toks\":[";
            for (std::size_t j = 0; j < s.tokens.size(); ++j) q << (j ? "," : "") << s.tokens[j];
            q << "],\"fnvs\":[";
            for (std::size_t j = 0; j < s.logits_fnv.size(); ++j) q << (j ? "," : "") << "\"" << hex(s.logits_fnv[j]) << "\"";
            q << "]";
            return q.str();
        };
        const auto qual_prompt = [&](const std::string& name, std::uint32_t pos, const std::vector<std::uint32_t>& rows, const CudaStepResult& s) {
            if (census && census_used) census_records[census_used - 1].counted = false;
            if (qual) emit("{\"event\":\"prompt_window\",\"phase\":\"" + name + "\",\"pos\":" + std::to_string(pos) + qual_rows(rows, s) + "}");
        };
        // Session create/suffix/control follow the ordinary chat prompt producer.
        // This replica is decode_only: max_rows is the widest admitted window
        // (six with a drafter). The standalone prefill driver has different
        // buffers/kernels and cannot promise byte-identical imported state.
        const auto ingest_prompt = [&](const std::vector<std::uint32_t>& tokens, std::uint32_t count,
                                       bool draft, const std::string& phase, auto&& progress) {
            std::uint32_t next = 0, windows = 0;
            progress(0U, count, windows);
            for (std::uint32_t at = 0; at < count;) {
                const auto n = std::min(max_rows, count - at);
                const std::vector<std::uint32_t> rows(tokens.begin()+at, tokens.begin()+at+n);
                WindowTimes wt;
                const auto pos = p.position();
                const auto result = run_window(rows, wt);
                qual_prompt(phase, pos, rows, result);
                next = result.tokens.back();
                p.commit_window(n, next);
                if (census && census_used) census_records[census_used - 1].committed = n;
                if (draft) append_committed(n);
                at += n;
                progress(at, count, ++windows);
            }
            if (count) prefetch_row0(next);
            return next;
        };
        const auto qual_chat_end = [&](const std::string& key, const std::string& name, const std::vector<std::uint32_t>& v) {
            if (!qual) return;
            const auto cs = p.export_state();  // the sequence's end state (qualification only: a download)
            std::ostringstream q;
            q << "{\"event\":\"chat_end\",\"key\":\"" << key << "\",\"name\":\"" << name << "\",\"position\":" << cs.position << ",\"next_token\":" << cs.next_token
              << ",\"digest\":\"" << hex(cs.digest()) << "\",\"toks\":[";
            for (std::size_t j = 0; j < v.size(); ++j) q << (j ? "," : "") << v[j];
            q << "]}";
            emit(q.str());
        };
        // Compares row j of a window (position pos) with the reference and counts it.
        const auto compare_row = [&](std::uint32_t pos, std::uint32_t fed, const CudaStepResult& s, std::uint32_t j, std::ostringstream& os) {
            const auto it = ref.find(pos);
            bool ok = false;
            if (it != ref.end() && j < s.tokens.size() && j < s.logits_fnv.size())
                ok = it->second.fed == fed && it->second.argmax == s.tokens[j] && it->second.fnv == s.logits_fnv[j];
            ++compared;
            matched += ok ? 1U : 0U;
            os << (j ? "," : "") << (ok ? "1" : "0");
            return ok;
        };
        // Window controller with measured costs (EWMA, seconds); with --controller-prior every fresh controller starts from
        // the stored survival table.
        strix::ds41::dspark::WindowController controller;
        std::optional<strix::ds41::dspark::WindowController::Table> prior_table;
        // seeded costs (seconds) for the warm controller; `seeded` = the current controller runs on them
        std::optional<strix::ds41::dspark::WindowController::Costs> seed_costs;
        bool seeded = false;
        const auto fresh_controller = [&](bool use_prior) {
            controller = strix::ds41::dspark::WindowController{};
            seeded = false;
            if (prior_table && use_prior) {
                controller.set_table(*prior_table);
                if (seed_costs) {
                    controller.skip_sweep();
                    controller.set_explore_every(controller_explore);
                    seeded = true;
                }
            }
        };
        std::ofstream trace_out;
        strix::ds41::dspark::WindowController::Costs costs;
        const auto ewma = [](float& v, std::uint32_t& n, double x) {
            v = n == 0 ? static_cast<float>(x) : static_cast<float>(0.8 * v + 0.2 * x);
            ++n;
        };
        std::uint32_t draft_samples = 0, extra_samples = 0;
        std::uint32_t fixed_drafts = 0;
        bool probe = false;  // draft at every position, verify none (R1): the acceptance record
        std::vector<std::string> probe_names;
        // The request's controller, trace, window mode and environment (the launch args once; with --serve at every
        // request): the same state a fresh process has right after its load (design notes).
        const auto configure_request = [&]() {
            {
                const char* v = std::getenv("STRIX_DS41_CUDA_ENGRAM_CHECK");
                engram_check_env = v && v[0] == '1';
                engram_check_on = engram_check_env;
                const char* w = std::getenv("STRIX_DS41_REPLICA_WINDOW_ENGRAM_PREFETCH");
                window_engram_prefetch = !(w && w[0] == '0');
            }
            controller = strix::ds41::dspark::WindowController{};
            prior_table.reset();
            if (!controller_prior.empty() && controller_prior != "none") {
                std::uint32_t cap = 64;
                prior_table = resolve_prior(controller_prior, &cap);
                controller.set_table(*prior_table);
                emit("{\"event\":\"controller_prior\",\"cap\":" + std::to_string(cap) + "}");
            }
            seed_costs.reset();
            // the default costs belong to the default warm start: a cold request (--controller-prior none) drops them
            if (!controller_costs.empty() && controller_costs != "none" && (prior_table || controller_costs_given)) {
                if (!prior_table) usage("--controller-costs needs --controller-prior");
                const auto v = split(controller_costs, ',');
                if (v.size() != 6U && v.size() != 7U) usage("--controller-costs C1,..,C6[,D] (ms)");
                strix::ds41::dspark::WindowController::Costs c{};
                for (std::uint32_t r = 1; r <= 6U; ++r) {
                    c.verify[r] = static_cast<float>(std::stod(v[r - 1U]) * 1e-3);
                    c.samples[r] = 1;
                    if (!(c.verify[r] > 0.0F)) usage("--controller-costs: every cost > 0");
                }
                c.draft = v.size() == 7U ? static_cast<float>(std::stod(v[6]) * 1e-3) : 0.002F;
                seed_costs = c;
            }
            seeded = false;
            costs = strix::ds41::dspark::WindowController::Costs{};
            draft_samples = extra_samples = 0;
            if (trace_out.is_open()) trace_out.close();
            trace_out.clear();
            if (!draft_trace.empty()) {
                trace_out.open(draft_trace, std::ios::app);
                if (!trace_out) usage("cannot open --draft-trace for writing");
            }
            fixed_drafts = 0;
            if (window_arg.rfind("fixed:", 0) == 0) fixed_drafts = static_cast<std::uint32_t>(std::stoul(window_arg.substr(6)));
            else if (window_arg != "controller" && window_arg != "probe") usage("--window controller | fixed:N | probe");
            probe = window_arg == "probe";
            probe_names = split(probe_phase, ',');
            if (controller_scope != "run" && controller_scope != "chat") usage("--controller-scope run | chat");
        };
        // Speculative decoding of `count` committed tokens from `next` (the token fed at p.position()); with `compare`
        // every committed row is compared with the reference. One JSON line per window; returns the next token.
        // the generated tokens of every chat, plain (chats) and speculative (schats): both greedy, so they must agree
        std::map<std::string, std::vector<std::uint32_t>> chat_plain, chat_spec;  // key: phase K, occurrence, chat
        std::map<std::string, std::string> chat_spec_ref;  // speculative arm key (full phase name) -> its plain key
        std::vector<std::uint32_t>* spec_out = nullptr;
        // the greedy reference of a chat: chats0's first occurrence, else the first plain arm that ran it (every plain
        // and speculative arm of the chat must equal it: placement and K never change the greedy tokens)
        // --serve: the greedy vector of every chat a plain arm of an earlier request produced (arms of one A/B run as
        // separate requests; greedy tokens never depend on K, windows, placement or scheduling)
        // keyed by everything a chat's greedy tokens depend on besides the weights: the imported state, the sequences
        // file, the prompt length and the chat index (a reference may be shorter or longer: prefixes are compared)
        std::map<std::string, std::vector<std::uint32_t>> worker_canon;
        const auto canon_key = [&](const std::string& c) { return state_file + "|" + seqs_file + "|" + std::to_string(chat_prompt) + "|" + c; };
        const auto from_worker = [&](const std::vector<std::uint32_t>* v) {
            for (const auto& [k, w] : worker_canon)
                if (&w == v) return true;
            return false;
        };
        // a candidate against a reference: equal, or (a reference from an earlier request) equal on the common prefix
        const auto same_tokens = [&](const std::vector<std::uint32_t>& ref, const std::vector<std::uint32_t>& v) {
            if (!from_worker(&ref)) return ref == v;
            const std::size_t n = std::min(ref.size(), v.size());
            return std::equal(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(n), ref.begin());
        };
        const auto canonical_chat = [&](const std::string& c) -> const std::vector<std::uint32_t>* {
            if (const auto it = chat_plain.find("0#0-" + c); it != chat_plain.end()) return &it->second;
            for (const auto& [k, v] : chat_plain)
                if (k.substr(k.rfind('-') + 1) == c) return &v;
            if (const auto it = worker_canon.find(canon_key(c)); it != worker_canon.end()) return &it->second;
            return nullptr;
        };
        // Timing: a window's cycle runs from its start to the next window's start (everything the host does per window,
        // the JSON line and the controller included), as the plain chats' cycle_ms; windows that start before `warm`
        // committed tokens are not counted (as the plain chats' first `warm` steps).
        struct SpecTotals {
            std::uint32_t windows = 0, committed = 0, drafted = 0, accepted = 0, draft_runs = 0;
            double ms = 0, draft_ms = 0, verify_ms = 0, append_ms = 0, misses = 0, fetched = 0, prefetched = 0, covered = 0, lookups = 0;
        };
        const auto speculate = [&](std::uint32_t next, std::uint32_t count, const std::string& phase, bool compare, SpecTotals& tot) {
            const std::uint32_t begin = p.position(), end = begin + count;
            std::chrono::steady_clock::time_point prev_t0{};
            bool prev_counted = false;
            std::uint32_t prev_kept = 0;
            double prev_draft = 0, prev_verify = 0, prev_append = 0, prev_misses = 0, prev_fetched = 0, prev_pref = 0, prev_cov = 0, prev_lookups = 0;
            std::uint32_t prev_k = 0, prev_a = 0;
            bool prev_drafted = false;
            std::uint32_t prev_rows = 0;
            bool aborted = false;
            std::size_t prev_census_id = std::size_t(-1);
            const auto credit = [&](double cycle) {
                if (census && prev_census_id != std::size_t(-1)) census_records[prev_census_id].cycle_ms = cycle;  // the previous window's full cycle
                if (cycle > 0 && prev_rows > 0 && (!seeded || controller_learn)) {  // controller: verify[R] = the window's full cycle minus its draft step
                    ewma(costs.verify[prev_rows], costs.samples[prev_rows], std::max(0.0, cycle - prev_draft) * 1e-3);
                    controller.set_costs(costs);
                }
                if (!prev_counted) return;
                ++tot.windows;
                tot.committed += prev_kept;
                tot.drafted += prev_k;
                tot.accepted += prev_a;
                tot.draft_runs += prev_drafted ? 1U : 0U;
                tot.ms += cycle;
                tot.draft_ms += prev_draft;
                tot.verify_ms += prev_verify;
                tot.append_ms += prev_append;
                tot.misses += prev_misses;
                tot.fetched += prev_fetched;
                tot.prefetched += prev_pref;
                tot.covered += prev_cov;
                tot.lookups += prev_lookups;
            };
            while (p.position() < end) {
                const std::uint32_t pos = p.position(), remaining = end - pos;
                const auto tw0 = std::chrono::steady_clock::now();
                const double cycle = prev_t0 == std::chrono::steady_clock::time_point{} ? -1.0 : std::chrono::duration<double, std::milli>(tw0 - prev_t0).count();
                credit(cycle);
                prev_t0 = tw0;
                strix::ds41::dspark::Draft d{};
                CudaDraftTimes dtimes;
                std::uint32_t k = 0;
                double draft_ms = 0;
                if (hinted_rows == 0) hint_rows(&next, 1);  // winpf: the bonus row before the draft (first window)
                bool probe_here = probe;
                for (const auto& n : probe_names)
                    if (!n.empty() && (phase == n || phase.rfind(n + "#", 0) == 0)) probe_here = true;
                const bool want = fixed_drafts || probe_here ? true : controller.draft_next();
                bool drafted = false;
                if (want && remaining > 1U) {
                    DeviceScope ds(ddev);
                    const auto td = std::chrono::steady_clock::now();
                    d = drafter->draft(next, false, &dtimes);
                    draft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
                    drafted = true;
                    k = probe_here ? 0U : fixed_drafts ? fixed_drafts : controller.choose(d);
                    k = std::min({k, max_rows - 1U, remaining - 1U, strix::ds41::dspark::block_size});
                }
                std::vector<std::uint32_t> rows{next};
                for (std::uint32_t j = 1; j <= k; ++j) rows.push_back(d.tokens[j]);
                hint_rows(rows.data() + 1, k);  // winpf: the drafts, right after the draft step
                WindowTimes wt;
                census_phase = phase;
                const auto s = run_window(rows, wt);
                std::uint32_t a = 0;
                while (a < k && d.tokens[a + 1U] == s.tokens[a]) ++a;
                const std::uint32_t kept = a + 1U, after = s.tokens[a];
                if (trace_out && drafted && (probe_here || fixed_drafts || compare || qual)) {  // acceptance record (host logging only; never a timed controller arm)
                    std::ostringstream ts;
                    ts << "{\"phase\":\"" << phase << "\",\"pos\":" << pos << ",\"t\":" << next << ",\"k\":" << k << ",\"rows\":" << rows.size()
                       << ",\"a\":" << a << ",\"d\":[";
                    for (std::uint32_t j = 1; j <= strix::ds41::dspark::block_size; ++j) ts << (j > 1 ? "," : "") << d.tokens[j];
                    ts << "],\"c\":[";
                    for (std::uint32_t j = 0; j < strix::ds41::dspark::block_size; ++j) ts << (j ? "," : "") << d.confidence[j];
                    if (qual) {  // the confidences' float bits (decimal printing does not prove bit equality)
                        ts << "],\"cb\":[";
                        for (std::uint32_t j = 0; j < strix::ds41::dspark::block_size; ++j) {
                            char b[12];
                            std::uint32_t bits = 0;
                            std::memcpy(&bits, &d.confidence[j], sizeof bits);
                            std::snprintf(b, sizeof b, "\"%08x\"", bits);
                            ts << (j ? "," : "") << b;
                        }
                    }
                    ts << "],\"g\":[";
                    for (std::uint32_t j = 0; j < rows.size() && j < s.tokens.size(); ++j) ts << (j ? "," : "") << s.tokens[j];
                    ts << "],\"draft_ms\":" << draft_ms << "}\n";
                    trace_out << ts.str();
                }
                std::ostringstream os;
                os << "{\"event\":\"window\",\"phase\":\"" << phase << "\",\"pos\":" << pos << ",\"rows\":" << rows.size() << ",\"accepted\":" << a;
                bool all_match = true;
                if (compare) {
                    os << ",\"match\":[";
                    for (std::uint32_t j = 0; j < kept; ++j) all_match = compare_row(pos + j, rows[j], s, j, os) && all_match;
                    os << "]";
                }
                if (census) {
                    auto& c = census_records[wt.census_id];
                    c.cycle_ms = -1;  // completed at the next cycle start (terminal tail at drain)
                    c.drafted = k; c.accepted = a; c.counted = pos - begin >= warm;
                    c.draft_steps_executed = drafted ? 1U : 0U;
                    c.draft_nonrouted_bytes = drafted ? census_draft_nonrouted : 0;
                    c.draft_routed_lower_bytes = drafted ? census_draft_expert * strix::ds41::dspark::top_k : 0;
                    c.draft_routed_upper_bytes = c.draft_routed_lower_bytes * strix::ds41::dspark::block_size;
                    c.vocabulary_head_bytes = dtimes.head_weight_bytes;
                    c.draft_device_ms = dtimes.device_ms; c.draft_wall_ms = dtimes.wall_ms;
                }
                if (!all_match) {
                    if (census) census_records[wt.census_id].errors = 1;  // stop at the first mismatch, before any commit / append / controller work
                    if (census) { os << ",\"census\":"; census_records[wt.census_id].write(os, census_phases[wt.census_id]); }
                    os << ",\"aborted\":true}";
                    emit(os.str());
                    aborted = true;
                    next = after;
                    break;
                }
                p.commit_window(kept, after);
                if (census && census_used) census_records[census_used - 1].committed = kept;
                prefetch_row0(after);
                if (p.position() < end) hint_rows(&after, 1);  // winpf: the next window's bonus row, before its draft
                if (spec_out)
                    for (std::uint32_t j = 0; j < kept; ++j) spec_out->push_back(j < a ? rows[j + 1U] : after);  // the committed greedy tokens
                const auto ta = std::chrono::steady_clock::now();
                append_committed(kept);
                {
                    DeviceScope ds(ddev);
                    std::vector<std::uint32_t> ctx(rows.begin(), rows.begin() + kept);
                    ctx.push_back(after);
                    drafter->add_vocabulary(ctx);
                }
                const double append_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta).count();
                const double win_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
                if (k > 0) controller.observe(d, k, a);
                // costs: draft = every draft step run (also when it chose no draft); verify[R] = the window's full cycle (start
                // to the next start: target, commit, append, vocabulary, controller, JSON) minus its draft step, set when the
                // next window starts (credit); extra = 0, so the no-draft alternative carries the same per-window work
                if (drafted && (!seeded || controller_learn)) ewma(costs.draft, draft_samples, draft_ms * 1e-3);
                costs.extra = 0.0F;
                controller.set_costs(costs);
                os << ",\"cycle_prev_ms\":" << cycle << ",\"ms\":" << win_ms << ",\"draft_ms\":" << draft_ms << ",\"step_ms\":" << wt.step_ms << ",\"finish_ms\":" << wt.finish_ms
                   << ",\"boundary_ms\":" << wt.boundary_ms << ",\"append_ms\":" << append_ms << ",\"device_ms\":" << s.device_ms
                   << ",\"engram_wait_ms\":" << s.engram_wait_seconds * 1e3 << ",\"engram_wait_first_ms\":" << s.engram_wait_first_seconds * 1e3
                   << ",\"engram_ms\":" << s.engram_seconds * 1e3 << ",\"engram_issue_ms\":" << s.engram_issue_seconds * 1e3
                   << ",\"engram_prefetched\":" << (s.engram_prefetched ? "true" : "false") << ",\"draft_dev_ms\":" << dtimes.device_ms
                   << ",\"draft_wall_ms\":" << dtimes.wall_ms << ",\"graph_captures\":" << s.graph_captures << ",\"lookups\":" << wt.st.lookups
                   << ",\"misses\":" << wt.st.misses << ",\"miss_layers\":" << wt.st.miss_layers << ",\"covered\":" << wt.st.covered << ",\"fetched\":" << wt.st.fetched
                   << ",\"prefetched\":" << wt.st.prefetched << ",\"bank\":" << wt.st.ring << ",\"resurrected\":" << wt.st.resurrected
                   << ",\"stalled\":" << wt.st.stalled << ",\"evicted\":" << wt.st.evicted << ",\"writebacks\":" << wt.st.writebacks_done
                   << ",\"collected_mid\":" << wt.st.collected_mid << ",\"k\":" << k;
                os << ",\"wb_bytes\":" << wt.st.wb_issued_bytes << ",\"copy_ms\":" << wt.copy_ms;
                if (qual) os << qual_rows(rows, s);
                if (tier->winpf())
                    os << ",\"winpf\":{\"prior\":[" << wt.st.prior_issued << "," << wt.st.prior_used << "," << wt.st.prior_late << "," << wt.st.prior_reverted
                       << "],\"l1\":[" << wt.st.l1_issued << "," << wt.st.l1_used << "," << wt.st.l1_late << "," << wt.st.l1_reverted << "," << wt.st.l1_priced_out
                       << "," << wt.st.l1_skipped_ahead << "],\"budget_stops\":" << wt.st.prior_budget_stops << "}";
                if (census) {
                    census_records[wt.census_id].append_ms = append_ms;
                    os << ",\"cycle_prev_id\":";
                    if (prev_census_id != std::size_t(-1)) os << prev_census_id; else os << "null";
                    os << ",\"census\":";
                    census_records[wt.census_id].write(os, census_phases[wt.census_id]);
                    prev_census_id = wt.census_id;
                }
                os << "}";
                emit(os.str());
                prev_rows = static_cast<std::uint32_t>(rows.size());
                prev_counted = pos - begin >= warm;
                prev_kept = kept;
                prev_k = k;
                prev_a = a;
                prev_drafted = drafted;
                prev_draft = draft_ms;
                prev_verify = wt.step_ms + wt.finish_ms + wt.boundary_ms;
                prev_append = append_ms;
                prev_misses = wt.st.misses;
                prev_fetched = wt.st.fetched;
                prev_pref = wt.st.prefetched;
                prev_cov = wt.st.covered;
                prev_lookups = wt.st.lookups;
                next = after;
            }
            hinted_rows = 0;  // (no hint is left for a window that never runs: the bonus is hinted only if one follows)
            if (aborted) return next;  // a failed window is never credited (its gate fails)
            {
                DeviceScope ds(ddev);
                STRIX_CUDA_CHECK(cudaStreamSynchronize(drafter->stream()));  // the last window's vocabulary gathers: inside its cycle
            }
            p.wait_io();  // and its outstanding Engram reads (both arms drain them before their last timestamp)
            {
                const auto td = std::chrono::steady_clock::now();
                STRIX_CUDA_CHECK(cudaDeviceSynchronize());  // and the expert copies / write-backs still queued (both arms)
                if (census && prev_census_id != std::size_t(-1)) census_records[prev_census_id].cycle_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prev_t0).count();
                emit("{\"event\":\"terminal_drain\",\"phase\":\"" + phase + "\",\"drain_ms\":" +
                     std::to_string(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count()) + "}");
            }
            credit(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prev_t0).count());  // the last window: to its end
            if (!controller_save.empty()) strix::ds41::dspark::write_survival_table(controller_save, controller.table());  // after the timing
            if (trace_out) trace_out.flush();
            return next;
        };
        const auto emit_route_stats = [&](const std::string& name) {
            if (!route_stats || !drafter) return;
            DeviceScope ds(ddev);
            const auto v = drafter->route_stats();
            std::ostringstream os;
            os << "{\"event\":\"route_stats\",\"phase\":\"" << name << "\",\"experts\":" << (v.size() / 3U) << ",\"counts\":[";
            for (std::size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
            os << "]}";
            emit(os.str());
            drafter->clear_route_stats();
        };
        const auto spec_summary = [&](const std::string& name, const SpecTotals& t) {
            std::ostringstream os;
            os << "{\"event\":\"spec_summary\",\"phase\":\"" << name << "\",\"windows\":" << t.windows << ",\"committed\":" << t.committed
               << ",\"tok_per_window\":" << (t.windows ? static_cast<double>(t.committed) / t.windows : 0.0) << ",\"draft_runs\":" << t.draft_runs
               << ",\"lookups\":" << t.lookups << ",\"drafted\":" << t.drafted
               << ",\"accepted\":" << t.accepted << ",\"ms\":" << t.ms << ",\"tok_s\":" << (t.ms > 0 ? 1e3 * t.committed / t.ms : 0.0)
               << ",\"draft_ms\":" << t.draft_ms << ",\"verify_ms\":" << t.verify_ms << ",\"append_ms\":" << t.append_ms
               << ",\"misses_per_token\":" << (t.committed ? t.misses / t.committed : 0.0) << ",\"fetched\":" << t.fetched << ",\"prefetched\":" << t.prefetched
               << ",\"covered\":" << t.covered << "}";
            emit(os.str());
        };

        // the attention state of every layer after a phase, folded into one FNV (labels and order of state_hashes())
        const auto state_digest = [&]() {
            std::uint64_t h = 1469598103934665603ULL;
            for (const auto& [label, v] : p.state_hashes()) {
                for (const unsigned char ch : label) h = (h ^ ch) * 1099511628211ULL;
                for (int b = 0; b < 8; ++b) h = (h ^ ((v >> (8 * b)) & 0xffU)) * 1099511628211ULL;
            }
            return h;
        };
        std::uint64_t exact_digest = 0, exact_canonical = 0;
        std::uint32_t exact_end = 0;
        std::vector<std::uint64_t> smoke_ref;
        std::map<std::string, std::uint32_t> seen_phase;
        const CudaStepIo io0 = io;  // the step io right after setup: every request starts from it
        const auto drain_request_state = [&]() {
            p.wait_io();
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            if (drafter) {
                DeviceScope ds(ddev);
                STRIX_CUDA_CHECK(cudaStreamSynchronize(drafter->stream()));
            }
        };
        const auto import_request_state = [&]() {
            if (imported.position >= o.max_positions) usage("import exceeds context capacity");
            drain_request_state();
            p.import_state(imported);
        };

        // Per-request reset (design notes), at a quiescent point (the previous request drained):
        // the tier knobs at their launch values, the step io, and every piece of app state a fresh process starts with.
        // The engine's sequence state is re-imported by every phase / chat (import_state), the drafter's by reset();
        // warm by design: the Engram record cache (the same bytes), the decode graphs (epoch_of), the expert placement.
        const auto reset_request_state = [&]() {
            drain_request_state();
            if (TierSite::can_set_sched()) tier->set_sched(wb_defer, flush_ready);
            spare_now = spares;
            tier->set_budgets(spares, pf_layer, pf_token);
            if (tier_test_hooks) tier->set_pressure(0, 0);
            {
                TierWinpf w = winpf_knobs;
                w.enable = false;
                tier->set_winpf(w);
            }
            hinted_rows = 0;
            p.set_decode_graphs(true);
            io = io0;
            if (drafter && route_stats) {
                DeviceScope ds(ddev);
                drafter->clear_route_stats();
            }
            spec_out = nullptr;
            chat_plain.clear();
            chat_spec.clear();
            chat_spec_ref.clear();
            agg.clear();
            compared = matched = 0;
            ph_bank = ph_res = ph_mid = 0;
            last_t0 = {};
            have_last = prev_counted = false;
            prev_phase.clear();
            last_stream_fnv = 0;
            exact_digest = exact_canonical = 0;
            exact_end = 0;
            smoke_ref.clear();
            seen_phase.clear();
            failures = 0;
        };
        const auto run_request = [&]() {
        tier->set_policy(evict == "auto" ? 2U : evict == "scored" ? 1U : 0U);
        ++census_request;
        census_used = 0;
        census_last_t0 = {};
        if (census) {
            std::uint64_t prompt_hash = 14695981039346656037ULL;
            for (const auto v : imported.tokens)
                for (unsigned b = 0; b < 4; ++b) { const unsigned char byte = (v >> (8 * b)) & 255; prompt_hash = census_hash(&byte, 1, prompt_hash); }
            for (const auto& c : split(chats_arg, ',')) {
                const auto i = std::stoul(c);
                if (i >= seqs.size()) continue;
                const auto& ids = seqs[i].second;
                for (std::size_t j = 0; j < std::min<std::size_t>(chat_prompt, ids.size()); ++j) {
                    const auto v = ids[j];
                    for (unsigned b = 0; b < 4; ++b) { const unsigned char byte = (v >> (8 * b)) & 255; prompt_hash = census_hash(&byte, 1, prompt_hash); }
                }
            }
            std::ostringstream meta;
            meta << "{\"event\":\"census_request\",\"request_id\":" << census_request
                 << ",\"effective_config\":" << effective_config()
                 << ",\"binary_sha256\":\"" << census_binary_hash << "\",\"tierstream_archive_sha256\":\"" << STRIX_TIERSTREAM_SHA256
                 << "\",\"target_admitted_sha256\":\"" << strix::deepseek41::pinned_file(strix::deepseek41::PinnedFileKind::target_q2).sha256_hex
                 << "\",\"draft_identity\":\"" << (census_draft_pinned ? "pinned-name-bytes" : "content") << "\",\"draft_sha256\":\"" << census_draft_hash
                 << "\",\"prompt_fnv1a64\":\"" << hex(prompt_hash)
                 << "\",\"context_position\":" << imported.position << ",\"context_digest\":\"" << hex(imported.digest())
                 << "\",\"allocations\":{\"target_arena_bytes\":" << tier->device_bytes() << ",\"target_driver_bytes\":" << p.device_bytes()
                 << ",\"draft_resident_bytes\":" << (drafter ? drafter->device_bytes() : 0) << ",\"target_driver\":" << p.describe() << "}}";
            emit(meta.str());
            std::size_t capacity = 1;
            for (const auto& seq : seqs) capacity += seq.second.size() + chat_steps + 32;
            capacity = (capacity + steps + 32) * std::max<std::size_t>(1, phases.size());
            census_records.resize(capacity);
            census_phases.resize(capacity);
            for (auto& phase : census_phases) phase.reserve(128);
        }
        if(!qual_dump.empty()) {
            if(!qual||!chat_logits)throw std::runtime_error("--qual-dump requires --qual --chat-logits");
            ordinary_logits.open(qual_dump+".logits.f32",std::ios::binary|std::ios::trunc);
            ordinary_tokens.open(qual_dump+".tokens.u32",std::ios::binary|std::ios::trunc);
            ordinary_greedy.open(qual_dump+".greedy.u32",std::ios::binary|std::ios::trunc);
            if(!ordinary_logits||!ordinary_tokens||!ordinary_greedy)throw std::runtime_error("cannot open qualification dump");
        }
        for (const auto& ph : phases) {
            census_phase = ph;
            if (failures) break;  // a failed phase ends the run
            // weight IO so far (cumulative), before each phase and after the last: phase deltas show any fallback or
            // demotion that happened while measuring
            if (const auto io = p.weight_io_fields(); !io.empty())
                emit("{\"event\":\"weight_io\",\"before_phase\":" + rs::json_str(ph) + io + "}");
            const std::uint32_t occurrence = seen_phase[ph]++;  // repeated phases (bracketing) are summarised apart
            {  // window prefetch: on only in this phase if its name ends in 'w' (placement only: exact either way)
                TierWinpf w = winpf_knobs;
                w.enable = winpf_phase(ph);
                tier->set_winpf(w);
            }
            engram_check_on = engram_check_env && ph.find("chats") == std::string::npos;
            p.set_engram_check(engram_check_on);
            // window phases under pressure (wsmokep / wforcep): the free target lowered as smokep (smoke) or pressure (full
            // model) does, demand capped per window: banks, resurrections and mid-window collections must show
            const bool wpress = ph.rfind("wsmokep", 0) == 0 || ph.rfind("wforcep", 0) == 0;
            const auto window_pressure = [&](bool on) {
                if (!wpress) return;
                if (on) {
                    set_pressure(pressure_spares, pressure_free, pressure_recent);  // smoke: size the arena small with --arena-gib
                } else {
                    set_pressure(spares, 0, 0);
                }
            };
            if (ph.rfind("gsmoke", 0) == 0) {
                // the capture inside graphs: pass 1 eager (capture always on, into A) records every window's captured rows;
                // pass 2 with graphs replays the same windows with the capture off / into A / into B, each (rows,
                // destination) three times in a row (eager warm-up, capture, replay): every captured row equals pass 1, the
                // destinations not targeted keep their canary pattern, the guard regions around A and B stay intact, and
                // the output streams equal the smoke phase's hashes in both passes
                if (smoke_ref.empty()) throw std::runtime_error("gsmoke needs an earlier smoke phase (its hashes)");
                if (o.stream_hashes) throw std::runtime_error("gsmoke needs graphs (no --stream-hashes)");
                set_predict(static_cast<std::uint32_t>(std::stoul(ph.substr(6))));
                const std::uint32_t H = shape.hidden;
                const std::size_t body = std::size_t{8} * 3U * H, guard = 4096;
                // destination B: portable mapped pinned host memory with STRIX_DS41_REPLICA_MAPPED_HIDDEN=1 (the cross-GPU capture)
                const std::size_t total = guard + body + guard;
                DeviceBuffer<float> dst_a(total), dst_b_dev(force_mapped ? 0 : total);
                HostRows dst_b_map;
                DecodeGraphReset graph_reset{p};
                if (force_mapped)
                    STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&dst_b_map.p), total * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped));
                float* const pa = dst_a.get();
                float* const pb = force_mapped ? dst_b_map.p : dst_b_dev.get();
                const float canary = __builtin_bit_cast(float, 0x7fc0beefU);
                const std::vector<float> fill(total, canary);
                const auto paint = [&](float* b) { STRIX_CUDA_CHECK(cudaMemcpy(b, fill.data(), total * sizeof(float), cudaMemcpyDefault)); };
                const auto download = [&](const float* b) {
                    std::vector<float> v(total);
                    STRIX_CUDA_CHECK(cudaMemcpy(v.data(), b, total * sizeof(float), cudaMemcpyDefault));
                    return v;
                };
                const auto intact = [&](const float* b, bool whole) {
                    const auto v = download(b);
                    for (std::size_t i = 0; i < v.size(); ++i) {
                        if (!whole && i >= guard && i < guard + body) continue;
                        if (__builtin_bit_cast(std::uint32_t, v[i]) != 0x7fc0beefU) return false;
                    }
                    return true;
                };
                struct Combo { std::uint32_t rows; int dest; };  // dest 0 off, 1 A, 2 B
                const std::vector<Combo> combos{{6, 0}, {6, 1}, {6, 2}, {2, 0}, {2, 1}, {2, 2}, {6, 1}, {2, 2}};
                std::vector<std::vector<float>> ref_rows;
                std::uint32_t rows_same = 0, rows_all = 0, cap_same = 0, cap_rows = 0, canary_ok = 0, canary_checks = 0, windows = 0;
                for (int pass = 0; pass < 2; ++pass) {
                    p.set_decode_graphs(pass == 1);
                    paint(pa);
                    paint(pb);
                    import_request_state();
                    const std::uint32_t P0 = p.position(), end = P0 + steps;
                    std::uint32_t w = 0;
                    while (p.position() < end) {
                        const std::uint32_t pos = p.position();
                        const auto& cb = combos[(w / 3U) % combos.size()];
                        const std::uint32_t R = std::min(cb.rows, end - pos);
                        const int dest = pass == 0 ? 1 : cb.dest;
                        float* const target = dest == 1 ? pa : dest == 2 ? pb : nullptr;
                        if (pass == 1) {  // the destinations this window must not touch: canary pattern
                            if (dest != 1) paint(pa);
                            if (dest != 2) paint(pb);
                        }
                        io.draft_hidden = target ? target + guard : nullptr;
                        io.draft_hidden_first = 0;
                        std::vector<std::uint32_t> rows;
                        for (std::uint32_t j = 0; j < R; ++j) rows.push_back(ref.at(pos + j).fed);
                        WindowTimes wt;
                        (void)run_window(rows, wt);
                        for (std::uint32_t j = 0; j < R; ++j) {
                            rows_same += row_fnv(j) == smoke_ref.at(pos + j - P0) ? 1U : 0U;
                            ++rows_all;
                        }
                        if (target) {
                            const auto v = download(target);
                            std::vector<float> got(v.begin() + static_cast<std::ptrdiff_t>(guard), v.begin() + static_cast<std::ptrdiff_t>(guard + std::size_t{R} * 3U * H));
                            if (pass == 0) {
                                ref_rows.push_back(std::move(got));
                            } else {
                                cap_same += std::memcmp(got.data(), ref_rows.at(w).data(), got.size() * 4U) == 0 ? 1U : 0U;
                                ++cap_rows;
                            }
                        }
                        if (pass == 1) {
                            canary_ok += (dest == 1 || intact(pa, true)) && (dest == 2 || intact(pb, true)) && intact(pa, false) && intact(pb, false) ? 1U : 0U;
                            ++canary_checks;
                        }
                        p.commit_window(R, rows.back());
                        if (census && census_used) census_records[census_used - 1].committed = R;
                        prefetch_ref_row0();
                        ++w;
                    }
                    windows = w;
                }
                p.set_decode_graphs(true);
                io.draft_hidden = nullptr;
                io.draft_hidden_first = ~0U;
                const bool ok = rows_same == rows_all && rows_all == 2U * steps && cap_same == cap_rows && cap_rows > 0 && canary_ok == canary_checks;
                emit("{\"event\":\"gate\",\"phase\":\"" + ph + "\",\"windows\":" + std::to_string(windows) + ",\"rows_same_as_smoke\":\"" +
                     std::to_string(rows_same) + "/" + std::to_string(rows_all) + "\",\"graph_captures_equal_eager\":\"" + std::to_string(cap_same) + "/" +
                     std::to_string(cap_rows) + "\",\"canaries\":\"" + std::to_string(canary_ok) + "/" + std::to_string(canary_checks) + "\",\"pass\":" +
                     (ok ? "true" : "false") + "}");
                if (!ok) { ++failures; break; }
            } else if (ph.rfind("csmoke", 0) == 0) {
                // the drafter-input capture (layers 0..2 here) against the raw layer inputs: the captured rows equal the
                // host formula of the raw streams, the raw input of layers 1 and 2 equals the step's own stream hash after
                // layers 0 and 1 (eager steps, last row), and the output streams equal the smoke phase's hashes with the
                // capture on and off (windows R = 1..max_rows sequentially, every row committed, capture on in odd windows)
                if (smoke_ref.empty()) throw std::runtime_error("csmoke needs an earlier smoke phase (its hashes)");
                set_predict(static_cast<std::uint32_t>(std::stoul(ph.substr(6))));
                ph_bank = ph_res = ph_mid = 0;
                import_request_state();
                const std::uint32_t P0 = p.position(), end = P0 + steps, H = shape.hidden;
                std::uint32_t windows = 0, rows_same = 0, rows_all = 0, cap_rows = 0, cap_same = 0, link_ok = 0, links = 0;
                while (p.position() < end) {
                    const std::uint32_t pos = p.position(), R = std::min(1U + windows % max_rows, end - pos);
                    const bool on = (windows % 2U) == 1U;
                    io.draft_hidden = on ? hidden_cap : nullptr;
                    io.draft_hidden_first = 0;
                    io.draft_hidden_raw = on ? hidden_raw.get() : nullptr;
                    std::vector<std::uint32_t> rows;
                    for (std::uint32_t j = 0; j < R; ++j) rows.push_back(ref.at(pos + j).fed);
                    WindowTimes wt;
                    const auto s = run_window(rows, wt);
                    for (std::uint32_t j = 0; j < R; ++j) {
                        rows_same += row_fnv(j) == smoke_ref.at(pos + j - P0) ? 1U : 0U;
                        ++rows_all;
                    }
                    if (on) {
                        std::vector<float> hid(std::size_t{R} * 3U * H), raw(std::size_t{R} * 3U * 4U * H);
                        STRIX_CUDA_CHECK(cudaMemcpy(hid.data(), hidden_cap, hid.size() * 4U, cudaMemcpyDefault));
                        STRIX_CUDA_CHECK(cudaMemcpy(raw.data(), hidden_raw.get(), raw.size() * 4U, cudaMemcpyDeviceToHost));
                        for (std::uint32_t j = 0; j < R; ++j) {
                            const float* rr = raw.data() + std::size_t{j} * 3U * 4U * H;
                            const auto want = strix::ds41::dspark::main_hidden_row(std::span<const float>(rr, 4U * H), std::span<const float>(rr + 4U * H, 4U * H),
                                                                                  std::span<const float>(rr + 8U * H, 4U * H));
                            cap_same += std::memcmp(want.data(), hid.data() + std::size_t{j} * 3U * H, want.size() * 4U) == 0 ? 1U : 0U;
                            ++cap_rows;
                        }
                        // the last row's raw input of layers 1 and 2 = its stream after layers 0 and 1 (the step's hashes)
                        const float* last = raw.data() + std::size_t{R - 1U} * 3U * 4U * H;
                        for (std::uint32_t l = 1; l <= 2U && l <= s.stream_hashes.size(); ++l) {
                            std::uint64_t h = 14695981039346656037ULL;  // the driver's FNV-1a basis (the app's row hash uses its own)
                            const auto* bytes = reinterpret_cast<const unsigned char*>(last + std::size_t{l} * 4U * H);
                            for (std::size_t i = 0; i < std::size_t{4} * H * 4U; ++i) h = (h ^ bytes[i]) * 1099511628211ULL;
                            link_ok += h == s.stream_hashes[l - 1U] ? 1U : 0U;
                            ++links;
                        }
                    }
                    p.commit_window(R, rows.back());
                    if (census && census_used) census_records[census_used - 1].committed = R;
                    prefetch_ref_row0();
                    ++windows;
                }
                io.draft_hidden = nullptr;
                io.draft_hidden_raw = nullptr;
                io.draft_hidden_first = ~0U;
                const bool ok = rows_same == rows_all && rows_all == steps && cap_same == cap_rows && cap_rows > 0 && link_ok == links && links > 0;
                emit("{\"event\":\"gate\",\"phase\":\"" + ph + "\",\"windows\":" + std::to_string(windows) + ",\"rows_same_as_smoke\":\"" +
                     std::to_string(rows_same) + "/" + std::to_string(rows_all) + "\",\"captured_rows_exact\":\"" + std::to_string(cap_same) + "/" +
                     std::to_string(cap_rows) + "\",\"raw_vs_stream_hashes\":\"" + std::to_string(link_ok) + "/" + std::to_string(links) + "\",\"pass\":" +
                     (ok ? "true" : "false") + "}");
                if (!ok) { ++failures; break; }
            } else if (ph.rfind("wsmoke", 0) == 0) {
                // windows of every (rows, kept) pair on the output streams: each kept row equals the smoke phase's hash
                if (smoke_ref.empty()) throw std::runtime_error("wsmoke needs an earlier smoke phase (its hashes)");
                set_predict(static_cast<std::uint32_t>(std::stoul(ph.substr(wpress ? 7 : 6))));
                io.draft_hidden = nullptr;
                ph_bank = ph_res = ph_mid = 0;
                window_pressure(true);
                import_request_state();
                const std::uint32_t P0 = p.position(), end = P0 + steps;
                WindowSchedule sched(max_rows);
                std::uint32_t same = 0, rows_checked = 0, windows = 0;
                while (p.position() < end) {
                    const std::uint32_t pos = p.position();
                    auto [R, kept] = sched.next();
                    R = std::min(R, end - pos);
                    kept = std::min(kept, R);
                    std::vector<std::uint32_t> rows;
                    for (std::uint32_t j = 0; j < R; ++j) {
                        const auto it = ref.find(pos + j);
                        if (it == ref.end()) throw std::runtime_error("wsmoke: the reference log has no token for this position");
                        rows.push_back(j < kept ? it->second.fed : (it->second.fed + 1U) % 129280U);  // rejected rows: wrong tokens
                    }
                    WindowTimes wt;
                    const auto ws = run_window(rows, wt);
                    std::ostringstream os;
                    os << "{\"event\":\"" << ph << "\",\"pos\":" << pos << ",\"rows\":" << R << ",\"kept\":" << kept
                       << ",\"engram_prefetched\":" << (ws.engram_prefetched ? 1 : 0) << ",\"engram_checked\":" << ws.engram_checked << ",\"same\":[";
                    for (std::uint32_t j = 0; j < kept; ++j) {
                        const bool ok = row_fnv(j) == smoke_ref.at(pos + j - P0);
                        same += ok ? 1U : 0U;
                        ++rows_checked;
                        os << (j ? "," : "") << (ok ? 1 : 0);
                    }
                    os << "],\"misses\":" << wt.st.misses << ",\"bank\":" << wt.st.ring << ",\"step_ms\":" << wt.step_ms << "}";
                    emit(os.str());
                    p.commit_window(kept, rows[kept - 1U]);
                    if (census && census_used) census_records[census_used - 1].committed = kept;  // no head: the next row's token is the reference's
                    prefetch_ref_row0();
                    ++windows;
                }
                window_pressure(false);
                const bool rare = !wpress || (ph_bank > 0 && ph_res > 0 && ph_mid > 0);
                const bool ok = same == rows_checked && rows_checked == steps && rare;
                emit("{\"event\":\"gate\",\"phase\":\"" + ph + "\",\"windows\":" + std::to_string(windows) + ",\"rows_checked\":" +
                     std::to_string(rows_checked) + ",\"same_as_smoke\":" + std::to_string(same) + ",\"bank\":" + std::to_string(ph_bank) +
                     ",\"resurrected\":" + std::to_string(ph_res) + ",\"collected_mid\":" + std::to_string(ph_mid) + ",\"pass\":" + (ok ? "true" : "false") + "}");
                if (!ok) { ++failures; break; }
            } else if (ph.rfind("smoke", 0) == 0) {
                const bool pressure = ph.rfind("smokep", 0) == 0;
                set_predict(static_cast<std::uint32_t>(std::stoul(ph.substr(pressure ? 6 : 5))));
                if (pressure) set_pressure(pressure_spares, pressure_free, pressure_recent);  // the arena is fixed at open: size it with --arena-gib
                ph_bank = ph_res = ph_mid = 0;
                import_request_state();
                have_last = false;
                std::vector<std::uint64_t> hs;
                for (std::uint32_t i = 0; i < steps; ++i) {
                    const auto it = ref.find(p.position());
                    if (it == ref.end()) throw std::runtime_error("smoke: the reference log has no token for this position");
                    (void)do_step({it->second.fed}, ph + "#" + std::to_string(occurrence), i, i >= warm, false);
                    if (i == 0U && std::getenv("STRIX_DSRTX_STEP0_STATE")) {  // diagnostic: the attention state of every layer after step 0
                        std::string line = "{\"event\":\"state0\",\"phase\":\"" + ph + "\"";
                        for (const auto& [label, v] : p.state_hashes()) line += ",\"" + label + "\":\"" + hex(v) + "\"";
                        emit(line + "}");
                    }
                    hs.push_back(last_stream_fnv);
                }
                if (pressure) set_pressure(spares, 0, 0);
                std::uint32_t same = 0;
                if (smoke_ref.empty()) smoke_ref = hs;
                for (std::size_t i = 0; i < hs.size(); ++i) same += hs[i] == smoke_ref[i] ? 1U : 0U;
                const bool rare = !pressure || (ph_bank > 0 && ph_res > 0 && ph_mid > 0);
                const bool ok = same == steps && rare;
                emit("{\"event\":\"gate\",\"phase\":\"" + ph + "\",\"steps\":" + std::to_string(steps) + ",\"same_as_first\":" + std::to_string(same) +
                     ",\"bank\":" + std::to_string(ph_bank) + ",\"resurrected\":" + std::to_string(ph_res) + ",\"collected_mid\":" + std::to_string(ph_mid) +
                     ",\"pass\":" + (ok ? "true" : "false") + "}");
                if (!ok) { ++failures; break; }
            } else if (ph.rfind("exact", 0) == 0 || ph.rfind("pressure", 0) == 0) {
                const bool pressure_phase = ph.rfind("pressure", 0) == 0;  // pressure / pressure<K>
                const std::uint32_t k = pressure_phase ? (ph.size() > 8 ? static_cast<std::uint32_t>(std::stoul(ph.substr(8))) : 0U)
                                                       : static_cast<std::uint32_t>(std::stoul(ph.substr(5)));
                io.logits = true;  // the gates compare logits FNV
                set_predict(k);
                if (pressure_phase) set_pressure(pressure_spares, pressure_free, pressure_recent);
                compared = matched = 0;
                ph_bank = ph_res = ph_mid = 0;
                import_request_state();
                have_last = false;
                std::uint32_t next = imported.next_token;
                for (std::uint32_t i = 0; i < steps; ++i) {
                    next = do_step({next}, ph + "#" + std::to_string(occurrence), i, i >= warm, true);
                    if (matched != compared) break;  // stop at the first mismatch (the gate fails on compared != steps)
                }
                if (pressure_phase) set_pressure(spares, 0, 0);
                // the attention state after the phase: the same in every exact phase (bits independent of residency)
                const std::uint64_t dg = state_digest();
                const bool digest_ok = exact_digest == 0 || dg == exact_digest;
                if (exact_digest == 0) exact_digest = dg;
                {
                    const auto cs = p.export_state();
                    const std::uint64_t cd = cs.digest();
                    if (exact_canonical == 0) { exact_canonical = cd; exact_end = cs.position; }
                    emit("{\"event\":\"canonical\",\"phase\":\"" + ph + "\",\"position\":" + std::to_string(cs.position) + ",\"next_token\":" +
                         std::to_string(cs.next_token) + ",\"digest\":\"" + hex(cd) + "\",\"same\":" + (cd == exact_canonical ? "true" : "false") + "}");
                    if (cd != exact_canonical) {
                        ++failures;
                        break;
                    }
                }
                const bool rare = !pressure_phase || (ph_bank > 0 && ph_res > 0 && ph_mid > 0);
                const bool ok = steps >= kQualifySteps && compared == steps && matched == steps && digest_ok && rare;
                emit("{\"event\":\"gate\",\"phase\":\"" + ph + "\",\"steps\":" + std::to_string(steps) + ",\"compared\":" + std::to_string(compared) +
                     ",\"matched\":" + std::to_string(matched) + ",\"state_digest\":\"" + hex(dg) + "\",\"digest_same\":" + (digest_ok ? "true" : "false") +
                     ",\"bank\":" + std::to_string(ph_bank) + ",\"resurrected\":" + std::to_string(ph_res) + ",\"collected_mid\":" + std::to_string(ph_mid) +
                     ",\"pass\":" + (ok ? "true" : "false") + "}");
                if (!ok) {
                    ++failures;
                    break;  // later phases would not be trustworthy
                }
            } else if (ph.rfind("wforce", 0) == 0 || ph.rfind("spec", 0) == 0) {
                // exact windows: wforce = every (rows, kept) pair with wrong tokens after the kept rows; spec = the drafter's
                const bool forced = ph[0] == 'w';
                const std::uint32_t k = static_cast<std::uint32_t>(std::stoul(ph.substr(forced ? (wpress ? 7 : 6) : 4)));
                window_pressure(true);
                io.logits = true;
                io.draft_hidden = forced ? nullptr : hidden_cap;
                set_predict(k);
                compared = matched = 0;
                ph_bank = ph_res = ph_mid = 0;
                import_request_state();
                std::uint32_t next = imported.next_token;
                const std::uint32_t P0 = p.position(), end = P0 + steps;
                std::uint32_t windows = 0;
                if (forced) {
                    WindowSchedule sched(max_rows);
                    while (p.position() < end) {
                        const std::uint32_t pos = p.position();
                        auto [R, kept] = sched.next();
                        R = std::min(R, end - pos);
                        kept = std::min(kept, R);
                        std::vector<std::uint32_t> rows{next};
                        for (std::uint32_t j = 1; j < R; ++j) {
                            const auto it = ref.find(pos + j);
                            if (it == ref.end()) throw std::runtime_error("wforce: the reference log has no token for this position");
                            rows.push_back(j < kept ? it->second.fed : (it->second.fed + 1U) % 129280U);
                        }
                        WindowTimes wt;
                        const auto s = run_window(rows, wt);
                        std::ostringstream os;
                        os << "{\"event\":\"" << ph << "\",\"pos\":" << pos << ",\"rows\":" << R << ",\"kept\":" << kept << ",\"match\":[";
                        bool all = true;
                        for (std::uint32_t j = 0; j < kept; ++j) all = compare_row(pos + j, rows[j], s, j, os) && all;
                        os << "],\"misses\":" << wt.st.misses << ",\"bank\":" << wt.st.ring << ",\"step_ms\":" << wt.step_ms << ",\"device_ms\":" << s.device_ms;
                        if (qual) os << qual_rows(rows, s);
                        os << "}";
                        emit(os.str());
                        if (!all) break;  // stop at the first mismatch (the gate fails on compared != steps)
                        next = s.tokens.at(kept - 1U);
                        p.commit_window(kept, next);
                        if (census && census_used) census_records[census_used - 1].committed = kept;
                        prefetch_row0(next);
                        ++windows;
                    }
                } else {
                    {
                        DeviceScope ds(ddev);
                        drafter->reset(P0);
                        if (draft_vocab != "full") {
                            const auto choice = strix::ds41::dspark::choose_vocabulary(draft_vocab, std::vector<std::uint32_t>{next});
                            drafter->set_vocabulary(choice.vocabulary);
                        } else {
                            drafter->set_vocabulary(strix::ds41::dspark::Vocab{});
                        }
                    }
                    fresh_controller(true);
                    costs = seeded ? *seed_costs : strix::ds41::dspark::WindowController::Costs{};
                    controller.set_costs(costs);
                    draft_samples = extra_samples = 0;
                    SpecTotals tot;
                    next = speculate(next, steps, ph, true, tot);
                    windows = tot.windows;
                    spec_summary(ph, tot);
                    emit_route_stats(ph);
                }
                // the attention state at the same end position as the exact phases: the same digests (attention pieces, and
                // the canonical export: pieces + tokens + position + next token)
                const std::uint64_t dg = state_digest();
                const bool digest_ok = exact_digest == 0 || dg == exact_digest;
                if (exact_digest == 0) exact_digest = dg;
                const auto cs = p.export_state();
                const std::uint64_t cd = cs.digest();
                const bool canon_ok = exact_canonical == 0 ? true : cd == exact_canonical && cs.position == exact_end;
                emit("{\"event\":\"canonical\",\"phase\":\"" + ph + "\",\"position\":" + std::to_string(cs.position) + ",\"next_token\":" +
                     std::to_string(cs.next_token) + ",\"digest\":\"" + hex(cd) + "\",\"same\":" + (canon_ok ? "true" : "false") + "}");
                window_pressure(false);
                const bool rare = !wpress || (ph_bank > 0 && ph_res > 0 && ph_mid > 0);
                const bool ok = compared == steps && matched == steps && p.position() == end && digest_ok && canon_ok && rare;
                emit("{\"event\":\"gate\",\"phase\":\"" + ph + "\",\"steps\":" + std::to_string(steps) + ",\"windows\":" + std::to_string(windows) +
                     ",\"compared\":" + std::to_string(compared) + ",\"matched\":" + std::to_string(matched) + ",\"state_digest\":\"" + hex(dg) +
                     "\",\"digest_same\":" + (digest_ok ? "true" : "false") + ",\"bank\":" + std::to_string(ph_bank) + ",\"resurrected\":" +
                     std::to_string(ph_res) + ",\"collected_mid\":" + std::to_string(ph_mid) + ",\"pass\":" + (ok ? "true" : "false") + "}");
                io.draft_hidden = nullptr;
                if (!ok) {
                    ++failures;
                    break;
                }
            } else if (ph.rfind("schats", 0) == 0) {
                // speculative chats: the prompt in windows of max_rows rows (all committed, drafter appended), then
                // speculative decoding of chat_steps tokens (timing; one JSON line per window)
                io.logits = chat_logits;
                io.draft_hidden = hidden_cap;
                auto kb = ph.substr(6);
                const std::uint32_t run_fixed = fixed_drafts;  // --window, restored after the phase
                if (const auto fpos = kb.find('f'); fpos != std::string::npos) {  // f<N>: N fixed drafts in this phase (R = N + 1)
                    fixed_drafts = static_cast<std::uint32_t>(std::stoul(kb.substr(fpos + 1)));
                    kb.erase(fpos);
                }
                const auto bpos = kb.find('b');
                set_predict(static_cast<std::uint32_t>(std::stoul(kb.substr(0, bpos))));
                tier->set_budgets(spares, bpos == std::string::npos ? pf_layer : static_cast<std::uint32_t>(std::stoul(kb.substr(bpos + 1))), pf_token);
                for (const auto& c : split(chats_arg, ',')) {
                    const auto& sq = seqs.at(std::stoul(c));
                    import_request_state();
                    const auto parts = split(sq.first, ':');
                    const std::string name = ph + "#" + std::to_string(occurrence) + "-" + c + "-" + (parts.size() > 1 ? parts[1] : std::string("x"));
                    census_phase = name;
                    const std::uint32_t P = std::min<std::uint32_t>(chat_prompt, static_cast<std::uint32_t>(sq.second.size()));
                    {
                        DeviceScope ds(ddev);
                        drafter->reset(p.position());
                        if (draft_vocab != "full") {
                            const std::vector<std::uint32_t> prompt(sq.second.begin(), sq.second.begin() + P);
                            const auto choice = strix::ds41::dspark::choose_vocabulary(draft_vocab, prompt);
                            drafter->set_vocabulary(choice.vocabulary);
                            drafter->add_vocabulary(prompt);
                        } else {
                            drafter->set_vocabulary(strix::ds41::dspark::Vocab{});
                        }
                    }
                    std::uint32_t next = 0;
                    for (std::uint32_t at = 0; at < P;) {
                        const std::uint32_t R = std::min(max_rows, P - at);
                        const std::vector<std::uint32_t> rows(sq.second.begin() + at, sq.second.begin() + at + R);
                        WindowTimes wt;
                        const std::uint32_t qpos = p.position();
                        const auto s = run_window(rows, wt);
                        qual_prompt(name, qpos, rows, s);
                        next = s.tokens.back();
                        p.commit_window(R, next);
                        if (census && census_used) census_records[census_used - 1].committed = R;
                        append_committed(R);
                        at += R;
                    }
                    prefetch_row0(next);
                    if (controller_scope == "chat") {  // every chat from a fresh controller (an R6 -> R2 sweep each)
                        fresh_controller(!controller_prior_alt || occurrence % 4U == 1U || occurrence % 4U == 2U);  // --controller-prior-alt: A B B A
                        costs = seeded ? *seed_costs : strix::ds41::dspark::WindowController::Costs{};
                        draft_samples = extra_samples = 0;
                        controller.set_costs(costs);
                    }
                    SpecTotals tot;
                    // every speculative arm keeps its own vector (probe, fixed and controller arms of one K and occurrence
                    // no longer overwrite each other), compared with the plain chat of the same K, occurrence and index
                    const std::string arm_key = ph + "#" + std::to_string(occurrence) + "-" + c;
                    chat_spec_ref[arm_key] = kb.substr(0, bpos) + "#" + std::to_string(occurrence) + "-" + c;
                    auto& outv = chat_spec[arm_key];
                    outv.assign(1, next);  // the prompt's greedy token, then one output per committed row
                    spec_out = &outv;
                    (void)speculate(next, chat_steps, name, false, tot);
                    spec_out = nullptr;
                    // fail fast: this speculative chat against its plain arm when that ran already
                    if (const auto* b = canonical_chat(c); b && !same_tokens(*b, outv)) {
                        emit("{\"event\":\"chat_tokens_early\",\"chat\":\"" + c + "\",\"pass\":false}");
                        ++failures;
                        break;
                    }
                    qual_chat_end("spec:" + arm_key, name, outv);
                    spec_summary(name, tot);
                    emit_route_stats(name);
                }
                fixed_drafts = run_fixed;
                io.draft_hidden = nullptr;
            } else {
                // chats<K>[b<B>]: K predicted experts per layer, at most B prefetch copies per layer (default --prefetch-layer)
                io.logits = chat_logits;  // serving: the greedy token only (no logits download / hash)
                const bool dphase = ph[0] == 'd';  // dchats: an alias of chats (the timeline always has dma_ms)
                const auto kb = ph.substr(dphase ? 6 : 5);
                const auto bpos = kb.find('b');
                set_predict(static_cast<std::uint32_t>(std::stoul(kb.substr(0, bpos))));
                const std::uint32_t budget = bpos == std::string::npos ? pf_layer : static_cast<std::uint32_t>(std::stoul(kb.substr(bpos + 1)));
                tier->set_budgets(spares, budget, pf_token);
                for (const auto& c : split(chats_arg, ',')) {
                    const auto& sq = seqs.at(std::stoul(c));
                    import_request_state();
                    have_last = false;
                    const auto parts = split(sq.first, ':');
                    const std::string name = ph + "#" + std::to_string(occurrence) + "-" + c + "-" + (parts.size() > 1 ? parts[1] : std::string("x"));
                    census_phase = name;
                    const std::uint32_t P = std::min<std::uint32_t>(chat_prompt, static_cast<std::uint32_t>(sq.second.size()));
                    std::uint32_t next = 0;
                    if (max_rows > 1U) {
                        // the prompt in windows of max_rows rows, as the speculative chats feed it (the same cache state
                        // at the first decode step)
                        for (std::uint32_t at = 0; at < P;) {
                            const std::uint32_t R = std::min(max_rows, P - at);
                            const std::vector<std::uint32_t> rows(sq.second.begin() + at, sq.second.begin() + at + R);
                            WindowTimes wt;
                            const std::uint32_t qpos = p.position();
                            const auto s = run_window(rows, wt);
                            qual_prompt(name, qpos, rows, s);
                            next = s.tokens.back();
                            p.commit_window(R, next);
                            if (census && census_used) census_records[census_used - 1].committed = R;
                            at += R;
                        }
                        prefetch_row0(next);  // as the speculative chats (the first decode step uses it)
                    } else {
                        for (std::uint32_t at = 0; at < P; ++at) next = do_step({sq.second[at]}, name + "-prompt", at, false, false);
                    }
                    auto& outv = chat_plain[kb.substr(0, bpos) + "#" + std::to_string(occurrence) + "-" + c];
                    outv.assign(1, next);  // the prompt's greedy token, then one output per step
                    for (std::uint32_t i = 0; i < chat_steps; ++i) {
                        if (wb_alt && i % wb_alt == 0U) {  // from the next boundary: no evictions (0) or the normal target
                            spare_now = (i / wb_alt) % 2U ? 0U : spares;
                            tier->set_budgets(spare_now, budget, pf_token);
                        }
                        next = do_step({next}, name, i, i >= warm, false);
                        outv.push_back(next);
                    }
                    if (wb_alt) {
                        spare_now = spares;
                        tier->set_budgets(spares, budget, pf_token);
                    }
                    // the last step's full cycle (its start to now): the step lines carry the previous step's cycle only
                    {
                        p.wait_io();  // the last step's Engram prefetch (both arms drain it before their last timestamp)
                        const auto td = std::chrono::steady_clock::now();
                        STRIX_CUDA_CHECK(cudaDeviceSynchronize());  // and the expert copies / write-backs still queued (both arms)
                        if (census && census_used) census_records[census_used - 1].cycle_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last_t0).count();
                        emit("{\"event\":\"terminal_drain\",\"phase\":\"" + name + "\",\"drain_ms\":" +
                             std::to_string(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count()) + "}");
                        const double tail_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last_t0).count();
                        emit("{\"event\":\"cycle_tail\",\"phase\":\"" + name + "\",\"cycle_ms\":" + std::to_string(tail_ms) + "}");
                        if (census && census_used) census_records[census_used - 1].cycle_ms = tail_ms;
                        if (prev_counted) agg[name].cycle.push_back(tail_ms);
                        prev_counted = false;  // the next phase / chat starts a new cycle chain
                    }
                    qual_chat_end("plain:" + kb.substr(0, bpos) + "#" + std::to_string(occurrence) + "-" + c, name, outv);
                }
            }
        }
        // and after the last phase: before/after pairs bound every measured phase
        if (const auto io = p.weight_io_fields(); !io.empty())
            emit("{\"event\":\"weight_io\",\"after_phases\":true" + io + "}");
        // speculative chats equal the plain chats token for token (both greedy from the same state and prompt)
        // (each schats occurrence against the chats occurrence of the same K and index; both hold chat_steps + 1 outputs)
        // every speculative arm AND every plain arm of a chat against the chat's canonical greedy vector, full length
        std::map<std::string, const std::vector<std::uint32_t>*> arms;
        for (const auto& [key, sp] : chat_spec) arms["spec:" + key] = &sp;
        for (const auto& [key, pl] : chat_plain) arms["plain:" + key] = &pl;
        if (!chat_spec.empty())
            for (const auto& [key, spp] : arms) {
                const auto& sp = *spp;
                const auto* ref = canonical_chat(key.substr(key.rfind('-') + 1));
                const bool found = ref != nullptr;
                const std::size_t n = found ? std::min(sp.size(), ref->size()) : 0U;
                std::size_t same = 0;
                while (found && same < n && sp[same] == (*ref)[same]) ++same;
                const bool ok = found && same == n && sp.size() == chat_steps + 1U && (ref->size() == chat_steps + 1U || from_worker(ref));
                emit("{\"event\":\"chat_tokens\",\"key\":\"" + key + "\",\"plain_found\":" + (found ? "true" : "false") + ",\"spec\":" +
                     std::to_string(sp.size()) + ",\"compared\":" + std::to_string(n) + ",\"same_prefix\":" + std::to_string(same) + ",\"pass\":" +
                     (ok ? "true" : "false") + "}");
                if (!ok) ++failures;
            }
        // every arm's greedy vector as a hash (persist: arms run as separate requests are compared across requests)
        for (const auto& [key, v] : arms) {
            std::uint64_t h = 1469598103934665603ULL;
            for (const std::uint32_t t : *v)
                for (int b = 0; b < 4; ++b) h = (h ^ ((t >> (8 * b)) & 0xffU)) * 1099511628211ULL;
            emit("{\"event\":\"chat_hash\",\"key\":\"" + key + "\",\"chat\":\"" + key.substr(key.rfind('-') + 1) + "\",\"len\":" +
                 std::to_string(v->size()) + ",\"fnv\":\"" + hex(h) + "\"}");
        }
        {  // terminal drain: from the last token's boundary to every write-back retired
            const auto t0 = std::chrono::steady_clock::now();
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            emit("{\"event\":\"drain\",\"ms\":" + std::to_string(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()) + "}");
        }
        if (verify_n) {
            const std::uint32_t bad = tier->verify(verify_n, 12345);
            emit("{\"event\":\"verify\",\"checked\":" + std::to_string(verify_n) + ",\"mismatches\":" + std::to_string(bad) + "}");
            if (bad) ++failures;
        }
        for (const auto& [phase, g] : agg) {
            std::ostringstream os;
            os << "{\"event\":\"summary\",\"phase\":\"" << phase << "\",\"steps\":" << g.ms.size() << ",\"ms_mean\":" << mean(g.ms) << ",\"ms_p50\":" << pct(g.ms, 0.5)
               << ",\"ms_p90\":" << pct(g.ms, 0.9) << ",\"cycle_ms\":" << mean(g.cycle) << ",\"tok_s\":" << (mean(g.cycle) > 0 ? 1e3 / mean(g.cycle) : 0.0)
               << ",\"tok_s_step\":" << (mean(g.ms) > 0 ? 1e3 / mean(g.ms) : 0.0) << ",\"step_ms\":" << mean(g.step_ms)
               << ",\"device_ms\":" << mean(g.dev) << ",\"boundary_ms\":" << mean(g.bnd) << ",\"engram_wait_ms\":" << mean(g.ewait) << ",\"misses\":" << mean(g.miss)
               << ",\"miss_layers\":" << mean(g.mlayers) << ",\"covered\":" << mean(g.covered) << ",\"fetched\":" << mean(g.fetched) << ",\"prefetched\":" << mean(g.pref)
               << ",\"recall\":" << mean(g.recall) << ",\"bank\":" << mean(g.ring) << ",\"evicted\":" << mean(g.evicted) << "}";
            emit(os.str());
        }
        if(!qual_dump.empty()) {
            p.wait_io();STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            namespace qs=strix::ds41::session;
            qs::write_file(qual_dump+".attention.bin",strix::deepseek41::serialize_attention_state(p.export_state()));
            qs::Bytes draft_bytes;
            if(drafter && std::any_of(phases.begin(),phases.end(),[](const auto& ph){return ph.rfind("schats",0)==0;})) {
                DeviceScope ds(ddev);const auto state=drafter->export_state();
                qs::DraftState d;d.start=state.start;d.next=state.next;d.core=state.vocabulary.core;d.extra=state.vocabulary.extra;
                for(unsigned i=0;i<3;++i)d.rings[i]=state.rings.at(i);
                draft_bytes=d.encode();
            }
            qs::write_file(qual_dump+".drafter.bin",draft_bytes);
            ordinary_logits.flush();ordinary_tokens.flush();ordinary_greedy.flush();
            qs::require(bool(ordinary_logits)&&bool(ordinary_tokens)&&bool(ordinary_greedy),"qualification dump flush failed");
            ordinary_logits.close();ordinary_tokens.close();ordinary_greedy.close();
        }
        if (const std::string t3 = tier->tier3_json(); !t3.empty()) emit("{\"event\":\"tier3\"," + t3 + "}");  // file tier / clean copies on
        };  // run_request
        if (!warmup_graphs.empty()) {  // every common graph captured + instantiated + launched once now
            const auto tw = std::chrono::steady_clock::now();
            std::uint32_t wsteps = 0, wcaptures = 0;
            for (const auto& ks : split(warmup_graphs, ',')) {
                const auto K = static_cast<std::uint32_t>(std::stoul(ks));
                set_predict(K);
                for (const bool logits : {true, false})
                    for (int hid = 0; hid < (drafter ? 2 : 1); ++hid)
                        for (std::uint32_t R = 1; R <= max_rows; ++R) {
                            import_request_state();
                            io.logits = logits;
                            io.draft_hidden = hid ? hidden_cap : nullptr;
                            const std::vector<std::uint32_t> rows(R, imported.next_token);
                            WindowTimes wt;
                            const auto s = run_window(rows, wt);
                            wcaptures += s.graph_captures;
                            ++wsteps;
                        }
            }
            import_request_state();
            p.wait_io();
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            io = io0;
            hinted_rows = 0;
            emit("{\"event\":\"warmup\",\"steps\":" + std::to_string(wsteps) + ",\"graph_captures\":" + std::to_string(wcaptures) + ",\"ms\":" +
                 std::to_string(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count()) + "}");
            if (!budget_point("warmup", "") && caps.capped())
                throw std::runtime_error("budget: the measured use after the graph warmup exceeds a total cap (see the budget event; raise --vram-margin-gib or the cap)");
        }

        namespace sc = strix::ds41::session;
        const bool session_supported=serving&&layers_arg==40&&session_config.max_parked>0;
        const std::uint64_t session_bound=session_supported ? p.capture_bound()+3*128*528+2*1024*1024 : 0;
        std::uint64_t session_total_budget=0;
        const std::uint64_t session_reserve=session_supported ? 8*session_bound+16*1024*1024 : 0;
        bool session_operational=false,session_admission_checked=false;
        if(serving&&!session_config.enabled)std::cerr<<"session cache: off ("<<session_config.reason<<")\n";
        struct SessionResources : sc::Resources {
            using sc::Resources::Resources;
            std::unique_ptr<attn::ExportCapture> capture[2];unsigned capture_slot=0;
        };
        std::unique_ptr<SessionResources> session_resources;
        std::uint64_t session_hits=0,session_misses=0,session_refusals=0;
        const auto session_draft_capture=[&]() {
            DeviceScope ds(ddev);const auto s=drafter->export_state();
            sc::DraftState out;out.start=s.start;out.next=s.next;out.core=s.vocabulary.core;out.extra=s.vocabulary.extra;
            for(unsigned i=0;i<3;++i)out.rings[i]=s.rings.at(i);
            return out.encode();
        };
        const auto session_capture_now=[&](sc::Frontier frontier) {
            p.wait_io();STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            auto& c=*session_resources->capture[session_resources->capture_slot++%2];
            p.capture_state(c);c.seal();
            sc::Snapshot s;s.compat=session_resources->compat;s.frontier=frontier;
            const auto b=c.bytes();s.attention.assign(b.begin(),b.end());
            if(frontier.dspark)s.drafter=session_draft_capture();
            return s;
        };
        const auto session_validate=[&](const sc::Snapshot& s) {
            sc::require(s.compat==session_resources->compat,"session compatibility mismatch");
            (void)sc::validate_attention(s,session_seed_holder.materialize(imported0),o.max_positions);
            if(s.frontier.dspark) {
                sc::require(bool(drafter) && ddev==target_device && max_rows>=s.frontier.fixed+1,
                            "checkpoint requires same-device drafter and sufficient max-rows");
                auto d=sc::DraftState::decode(s.drafter);
                sc::require(d.next==s.frontier.draft,"drafter frontier mismatch");
                sc::require(d.core==129280 && d.extra.empty() && draft_vocab=="full",
                            "session producer requires full draft vocabulary");
            } else sc::require(s.drafter.empty(),"unexpected drafter section");
        };

        if (!serving) {
            configure_request();
            run_request();
            if (meter) {
                STRIX_CUDA_CHECK(cudaDeviceSynchronize());
                if (!budget_point("run", "") && caps.capped()) ++failures;
            }
        } else {
            // ---- the persistent worker (design notes) ----
            rs::Spool spool(serve_dir);
            spool.open();
            const auto launch_runtime = std::make_tuple(PERSIST_RUNTIME_VARS);  // every request starts from the launch values
            const auto session_initialize=[&](const sc::Progress& progress) {
                if(!session_resources)session_resources=std::make_unique<SessionResources>(
                    session_config,session_total_budget-session_reserve,session_bound);
                if(!session_resources->compat.empty())return;
                std::tie(PERSIST_RUNTIME_VARS)=launch_runtime;
                apply_env();
#ifdef _WIN32
                const auto environment=sc::compat_environment_record();
#endif
                for(auto& c:session_resources->capture){c=std::make_unique<attn::ExportCapture>();c->reserve(p.capture_bound());progress("capture-reserved",0,0);}
                sc::Bytes record;sc::text(record,"strix-rtx-session-v2;attention-v1;dspark-v1;tp=1/rank=0;fp8-physical;greedy;decode-rows");
                // Per-component digests go to the log, so two workers' identities can be diffed when a resume is refused.
                std::string parts;std::size_t mark=record.size();
                const auto part=[&](const char* label) {
                    parts+=std::string(parts.empty()?"":" ")+label+"="+
                           sc::sha(std::span<const std::uint8_t>(record.data()+mark,record.size()-mark)).substr(0,16);
                    mark=record.size();
                };
                sc::text(record,session_resources->hashes.full(model,progress));part("model");
                sc::text(record,draft_file.empty()?"no-drafter":session_resources->hashes.full(draft_file,progress));part("draft");
#ifdef _WIN32
                std::vector<wchar_t> exe(32768);
                auto n=GetModuleFileNameW(nullptr,exe.data(),static_cast<DWORD>(exe.size()));
                sc::require(n>0&&n<exe.size(),"cannot identify executable");
                sc::text(record,session_resources->hashes.full(std::filesystem::path(std::wstring(exe.data(),n)),progress));
#else
                sc::text(record,session_resources->hashes.full("/proc/self/exe",progress));
#endif
                part("exe");
                // Include all unknown options; exclude only transport, fixtures and
                // output destinations. The executable hash binds compiled defaults.
                // Resolve the selected device through CUDA, independently of index/UUID spelling.
                const auto hardware=[&](int device) {
                    cudaDeviceProp prop{};int driver=0,runtime=0;
                    STRIX_CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
                    STRIX_CUDA_CHECK(cudaDriverGetVersion(&driver));
                    STRIX_CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
                    sc::compat_hardware(record,std::span(reinterpret_cast<const std::uint8_t*>(prop.uuid.bytes),16),
                                        prop.major,prop.minor,driver,runtime);
                };
                hardware(target_device);
                if(drafter)hardware(ddev);
                part("hardware");
                sc::compat_arguments(record,launch_args,[&](const auto& path){return session_resources->hashes.full(path);});
                part("args");
#ifdef _WIN32
                record.insert(record.end(),environment.begin(),environment.end());
                part("env");
#else
                // The RTX worker is Windows-specific. Refuse an incomplete environment
                // identity on another platform rather than silently weakening compat.
                throw std::runtime_error("RTX session environment identity requires Windows");
#endif
                session_resources->compat=sc::sha(record);
                std::cerr<<"session_compat "<<session_resources->compat<<" "<<parts<<"\n";
            };
            const std::uint64_t load_id = static_cast<std::uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
            {
                std::ostringstream os;
                os << "{\"event\":\"ready\",\"load_id\":" << load_id << ",\"pid\":"
#ifdef _WIN32
                   << GetCurrentProcessId()
#else
                   << 0
#endif
                   << p.weight_io_fields()
                   << ",\"dense_s\":" << dense_s << ",\"draft_load_s\":" << draft_load_s << ",\"load_s\":" << load_s
                   << ",\"max_positions\":" << o.max_positions << ",\"max_rows\":" << max_rows << ",\"drafter\":" << (drafter ? "true" : "false")
                   << ",\"test_hooks\":" << (tier_test_hooks ? "true" : "false") << ",\"max_predict\":" << std::max(predict, 8U)
                   << ",\"session_config\":{\"session_cache\":" << (session_config.enabled?"true":"false")
                   << ",\"session_explicit_opt_in\":" << (session_config.explicit_opt_in?"true":"false")
                   << ",\"session_route\":\"rtx-tp1-decode-rows\",\"session_off_reason\":" << rs::json_str(session_config.reason) << "}"
                   << ",\"arena_slots\":" << slots << ",\"spares\":" << spares << ",\"pressure_spares\":" << pressure_spares
                   << ",\"host_spares\":" << host_spares << ",\"wb_defer\":" << (wb_defer ? "true" : "false") << ",\"flush_ready\":" << (flush_ready ? "true" : "false")
                   << ",\"winpf_prior\":" << rs::json_str(winpf_prior_file) << ",\"draft_keep\":" << draft_keep << ",\"env\":{";
                {
                    bool first = true;
                    for (const char* n : {"STRIX_DS41_CUDA_ENGRAM_CHECK", "STRIX_DS41_REPLICA_WINDOW_ENGRAM_PREFETCH", "STRIX_DS41_REPLICA_MAPPED_HIDDEN",
                                          "STRIX_DS41_CUDA_DSPARK_GRAPH", "STRIX_DS41_CUDA_COMMIT_NEGATIVE_CONTROL", "CUDA_VISIBLE_DEVICES",
                                          "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING", "STRIX_SESSION_CACHE"}) {
                        const char* v = std::getenv(n);
                        os << (first ? "" : ",") << rs::json_str(n) << ":" << (v ? rs::json_str(v) : std::string("null"));
                        first = false;
                    }
                }
                os << "},\"args\":[";
                for (std::size_t i = 0; i < launch_args.size(); ++i) os << (i ? "," : "") << rs::json_str(launch_args[i]);
                os << "]}";
                emit(os.str());
                spool.ready(os.str());
            }
            g_serving = true;
            auto idle_from = std::chrono::steady_clock::now();
            std::uint32_t served = 0;
            for (;;) {
                const auto name = spool.take();
                if (!name) {
                    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - idle_from).count() > serve_idle_s) {
                        emit("{\"event\":\"serve_idle_exit\",\"idle_s\":" + std::to_string(serve_idle_s) + "}");
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                }
                const auto t_req = std::chrono::steady_clock::now();
                int rc = 0;
                bool fatal = false, quit = false, ping = false;
                std::string error;
                std::vector<std::string> args;

                struct SessionRequest {
                    sc::Command session_command;
                    std::optional<sc::Lru::Pending> session_input;
                    std::optional<strix::deepseek41::AttentionState> session_attention;
                    std::optional<sc::DraftState> session_draft;
                    std::vector<std::uint32_t> session_tokens,session_prefix,session_output;
                    std::vector<std::uint32_t> full_history;
                    bool automatic=false,history_hit=false,recompute=false;
                    std::optional<std::decay_t<decltype(launch_runtime)>> runtime;
                    std::string session_result="null";
                    std::ofstream session_logits,session_token_dump;
                };
                std::optional<SessionRequest> session_request;
                std::optional<sc::Error> session_error;
                bool session_used=false,session_keep_live=false,session_path_managed=false;
                bool session_device_touched=false,session_attempt=false;
                double session_read_verify_ms=0,session_copy_ms=0,session_capture_ms=0,session_disk_ms=0;
                const auto session_progress=[&](const std::string& stage, std::uint64_t done=0, std::uint64_t total=0) {
                    emit("{\"event\":\"session_progress\",\"request\":"+rs::json_str(*name)+
                         ",\"session\":"+rs::json_str(session_request->session_command.id)+",\"stage\":"+rs::json_str(stage)+
                         ",\"done\":"+std::to_string(done)+",\"total\":"+std::to_string(total)+
                         ",\"elapsed_s\":"+std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now()-t_req).count())+"}");
                };
                const auto session_preflight=[&]() {
                    const auto tr=std::chrono::steady_clock::now();
                    auto& c=session_request->session_command;
                    session_progress("preflight");
                    if(!session_admission_checked) {
                        session_config.resolve_ram();session_total_budget=session_config.ram;
                        session_operational=session_supported&&session_bound<=session_total_budget&&
                            session_reserve<=session_total_budget-session_bound&&
                            session_reserve+session_bound<=sc::host_memory(true);
                        session_admission_checked=true;
                        if(!session_operational)session_config.off("insufficient host memory for configured context");
                    }
                    sc::require(session_operational,"session host budget or geometry unavailable");
                    const auto& session_seed=session_seed_holder.materialize(imported0);
                    if(!c.path.empty()) {
                        session_config.resolve_directory();
                        const auto route=sc::route_explicit_path(c.path,session_config.dir,c.id);
                        c.path=route.path.string();session_path_managed=route.managed;
                    }
                    if(!session_resources)session_resources=std::make_unique<SessionResources>(
                        session_config,session_total_budget-session_reserve,session_bound);
                    // External immutable paths are independent of managed-tier ownership.
                    // Refuse busy managed operations before changing RAM, device or dumps.
                    sc::prepare_ram_command(c,session_resources->ram,session_resources->names,session_request->automatic);
                    const bool needs_tier=session_path_managed || c.op=="list" || c.op=="stats" || c.op=="delete" ||
                        (c.path.empty()&&(c.op=="restore"||(c.op=="checkpoint"&&c.durability=="disk")));
                    if(needs_tier)session_resources->disk.ensure_open();
                    if(session_path_managed||(session_config.disk&&(c.op=="list"||c.op=="delete"))||
                       (c.path.empty()&&(c.op=="restore"||(c.op=="checkpoint"&&c.durability=="disk"))))
                        session_resources->disk.require_available();
                    try{session_initialize(session_progress);session_progress("initialized");}catch(const std::exception& e){
                        for(auto& capture:session_resources->capture)capture.reset();
                        session_config.off(e.what());
                        std::cerr<<"session cache: off ("<<session_config.reason<<")\n";
                        throw;
                    }
                    // Explicit owner-managed save/load paths are independent of the automatic disk tier.
                    if(session_path_managed) {
                        sc::require(session_resources->disk.enabled(),"session disk budget unavailable");
                        c.path=session_resources->disk.path(c.id).string();
                    }
                    if(c.path.empty()&&(c.op=="restore"||(c.op=="checkpoint"&&c.durability=="disk")))sc::require(session_resources->disk.enabled(),"session disk budget unavailable");
                    if(c.path.empty()&&(c.op=="restore"||(c.op=="checkpoint"&&c.durability=="disk"))) {
                        c.path=session_resources->disk.path(c.id).string();session_path_managed=true;
                    }
                    if(c.op=="list"||c.op=="stats"||c.op=="delete")return;
                    if(c.op=="restore") {
                        if(session_resources->ram.contains(c.id))throw sc::Error("session_conflict",409,"restore would replace an existing session (delete first)");
                        session_request->session_input=sc::Lru::Pending{sc::restore(c.path,session_resources->compat,session_resources->ram.budget(),session_progress,session_path_managed)};
                    } else if(c.op=="create") {
                        if(session_resources->ram.contains(c.id))throw sc::Error("session_conflict",409,"session already exists");
                        sc::require(!c.dspark || (drafter && ddev==target_device && max_rows>=c.fixed+1),
                                    "DSpark session requires same-device drafter and sufficient max-rows");
                        // The drafter section stores the full vocabulary only; a subset ('auto:<dir>') worker
                        // cannot produce a restorable DSpark session, so refuse before any work.
                        if(c.dspark&&draft_vocab!="full")
                            throw sc::Error("session_unsupported",422,"DSpark sessions require a worker launched with --draft-vocab full (this worker: "+draft_vocab+")");
                        sc::Snapshot s;s.compat=session_resources->compat;
                        s.attention=strix::deepseek41::serialize_attention_state(session_seed);
                        s.frontier.consumed=s.frontier.accepted=session_seed.position;s.frontier.pending=session_seed.next_token;
                        s.frontier.dspark=c.dspark;s.frontier.fixed=c.fixed;
                        s.frontier.draft=c.dspark?session_seed.position:0;
                        if(c.dspark) {
                            sc::DraftState d;d.start=d.next=session_seed.position;
                            for(auto& ring:d.rings)ring.resize(128*528);
                            s.drafter=d.encode();
                        }
                        session_request->session_input=sc::Lru::Pending{std::move(s)};
                    } else if(!session_request->session_input)session_request->session_input=session_resources->ram.stage(c.id);
                    sc::require(session_request->session_input->snapshot.frontier.encode()==sc::Frontier::decode(session_request->session_input->snapshot.frontier.encode()).encode(),"invalid session frontier");
                    if(session_request->automatic||!c.history.empty()) {
                        auto history=session_request->full_history;
                        if(!session_request->automatic) {
                            std::ifstream in(c.history);sc::require(bool(in),"cannot read session history");
                            history=sc::read_tokens(in,o.max_positions);
                        }
                        auto saved=strix::deepseek41::deserialize_attention_state(session_request->session_input->snapshot.attention);
                        session_request->history_hit=c.op!="create"&&sc::history_prefix(saved.tokens,history)&&
                            c.dspark==session_request->session_input->snapshot.frontier.dspark&&c.fixed==session_request->session_input->snapshot.frontier.fixed;
                        if(!session_request->history_hit) {
                            session_request->recompute=session_request->automatic;
                            session_request->session_input->generation=0;
                            sc::require(history.size()>=session_seed.tokens.size()&&std::equal(session_seed.tokens.begin(),session_seed.tokens.end(),history.begin()),"history does not extend launch seed");
                            session_request->session_input->snapshot.attention=strix::deepseek41::serialize_attention_state(session_seed);
                            auto& f=session_request->session_input->snapshot.frontier;f={};f.consumed=f.accepted=session_seed.position;f.pending=session_seed.next_token;
                            f.dspark=c.dspark;f.fixed=c.fixed;f.draft=c.dspark?f.consumed:0;
                            session_request->session_input->snapshot.drafter.clear();
                            session_request->session_input->snapshot.lineage=sc::Snapshot::new_lineage();
                            session_request->session_input->snapshot.committed_generation=0;
                            if(c.dspark){sc::DraftState d;d.start=d.next=session_seed.position;for(auto& r:d.rings)r.resize(128*528);session_request->session_input->snapshot.drafter=d.encode();}
                            saved=session_seed;
                        }
                        session_request->session_tokens.assign(history.begin()+saved.tokens.size(),history.end());
                        if(session_request->history_hit)++session_hits;else ++session_misses;
                    } else if(c.op=="turn") {
                        ++session_hits;
                    }
                    session_validate(session_request->session_input->snapshot);
                    session_read_verify_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tr).count();
                    if(c.op=="restore" || c.op=="checkpoint")return;
                    session_request->session_attention=strix::deepseek41::deserialize_attention_state(session_request->session_input->snapshot.attention);
                    if(session_request->session_input->snapshot.frontier.dspark)session_request->session_draft=sc::DraftState::decode(session_request->session_input->snapshot.drafter);
                    if(c.op=="turn"||c.op=="create") {
                        sc::require(c.dspark==session_request->session_input->snapshot.frontier.dspark&&c.fixed==session_request->session_input->snapshot.frontier.fixed,
                                    "session mode/fixed-window change refused");
                        if(!c.tokens.empty()) {
                            std::ifstream in(c.tokens);sc::require(bool(in),"cannot read session token file");
                            session_request->session_tokens=sc::read_tokens(in,o.max_positions);
                        }
                        if(!c.control_prefix.empty()) {
                            sc::require(qual,"control prefix requires --qual");
                            std::ifstream in(c.control_prefix);sc::require(bool(in),"cannot read control prefix");
                            session_request->session_prefix=sc::read_tokens(in,o.max_positions);
                        }
                        sc::require(session_request->session_input->snapshot.frontier.consumed+session_request->session_prefix.size()+session_request->session_tokens.size()+std::uint64_t(c.steps)<o.max_positions,
                                    "session turn exceeds context capacity");
                        sc::require(session_bound<=session_resources->ram.budget(),"session capture exceeds host budget");
                        if(!c.dump.empty()&&!session_request->recompute) {
                            session_request->session_logits.open(c.dump+".logits.f32",std::ios::binary|std::ios::trunc);
                            session_request->session_token_dump.open(c.dump+".tokens.u32",std::ios::binary|std::ios::trunc);
                            sc::require(bool(session_request->session_logits)&&bool(session_request->session_token_dump),"cannot open session dumps");
                        }
                    }
                };
                const auto session_execute=[&]() {
                    const auto t0=std::chrono::steady_clock::now();
                    const auto& c=session_request->session_command;
                    session_progress("execute");
                    if(session_request->recompute) {
                        // A full-prompt miss executes the ordinary producer from its launch seed.
                        session_device_touched=true;
                        std::tie(PERSIST_RUNTIME_VARS)=*session_request->runtime;apply_env();configure_request();
                        session_resources->ram.clear_active();reset_request_state();run_request();
                        sc::require(failures==0,"ordinary named prompt failed");
                        sc::Frontier f;f.dspark=c.dspark;f.fixed=c.fixed;f.consumed=f.accepted=p.position();
                        f.emitted=c.steps;f.turns=1;f.draft=c.dspark?p.position():0;
                        auto snapshot=session_capture_now(f);
                        const auto state=strix::deepseek41::deserialize_attention_state(snapshot.attention);
                        f.pending=state.next_token;
                        session_request->session_output.assign(state.tokens.end()-c.steps,state.tokens.end());
                        snapshot.frontier=f;snapshot.committed_generation=1;session_validate(snapshot);
                        if(!c.dump.empty())sc::write_file(c.dump+".frontier.bin",f.encode());
                        session_resources->ram.put(c.id,std::move(snapshot));session_resources->ram.activate(c.id);
                        session_resources->names.admitted(c.id);
                        // Use the common result builder after this completed ordinary execution.
                    } else {
                    if(session_request->session_input) {
                        session_keep_live=session_resources->ram.keep_live(c.id,*session_request->session_input);
                        if(c.op=="restore"&&session_path_managed)session_resources->disk.touch(c.id);
                        // New/loaded/history-reset snapshots need a fresh generation.
                        // Keep the CPU input for device import and checkpoint output.
                        session_resources->ram.admit(c.id,*session_request->session_input);
                        session_resources->names.admitted(c.id);
                    }
                    if(c.op=="list") {
                        std::ostringstream out;out<<"{\"sessions\":[";bool first=true;
                        for(auto& id:session_resources->ram.list()){
                            const auto& v=session_resources->ram.get(id);
                            bool durable=false;
                            if(session_resources->disk.contains(id)) {
                                const auto checkpoint=sc::read_pointer(session_resources->disk.path(id));
                                durable=checkpoint.lineage==v.lineage&&checkpoint.generation==v.committed_generation;
                            }
                            out<<(first?"":",")<<"{\"id\":"<<rs::json_str(id)
                                <<",\"position\":"<<v.frontier.consumed<<",\"consumed\":"<<v.frontier.consumed
                                <<",\"pending\":"<<v.frontier.pending<<",\"bytes\":"<<v.size()
                                <<",\"state\":"<<rs::json_str(session_resources->ram.active()==id?"active":"parked")
                                <<",\"durability\":"<<rs::json_str(durable?"disk":"ram")<<"}";first=false;
                        }
                        for(auto& id:session_resources->disk.list())if(!session_resources->ram.contains(id)){
                            const auto checkpoint=sc::read_pointer(session_resources->disk.path(id));
                            out<<(first?"":",")<<"{\"id\":"<<rs::json_str(id)
                               <<",\"generation\":"<<checkpoint.generation<<",\"lineage\":"<<rs::json_str(checkpoint.lineage)
                               <<",\"state\":\"disk\",\"durability\":\"disk\"}";first=false;
                        }
                        out<<"]}";session_request->session_result=out.str();return;
                    }
                    if(c.op=="stats") {
                        session_request->session_result="{\"ram_bytes\":"+std::to_string(session_resources->ram.used())+",\"ram_budget\":"+std::to_string(session_resources->ram.budget())+
                            ",\"parked\":"+std::to_string(session_resources->ram.list().size())+",\"disk_bytes\":"+std::to_string(session_resources->disk.used())+
                            ",\"hits\":"+std::to_string(session_hits)+",\"misses\":"+std::to_string(session_misses)+",\"refusals\":"+std::to_string(session_refusals)+
                            ",\"host_bytes\":"+std::to_string(session_resources->ram.used())+",\"host_budget\":"+std::to_string(session_resources->ram.budget())+
                            ",\"evicted\":"+std::to_string(session_resources->ram.evicted())+",\"active\":"+rs::json_str(session_resources->ram.active())+
                            ",\"total_budget\":"+std::to_string(session_total_budget)+",\"capture_scratch_reserve\":"+std::to_string(session_reserve)+
                        ",\"pinned_capture_bytes\":"+std::to_string(2*p.capture_bound())+",\"background_dma\":false}";return;
                    }
                    if(c.op=="delete") {
                        const bool ram_deleted=session_resources->ram.erase(c.id),disk_deleted=session_resources->disk.erase(c.id);
                        sc::require(ram_deleted||disk_deleted||session_resources->names.known(c.id),"session_not_found");
                        session_resources->names.deleted(c.id);
                        session_request->session_result=std::string("{\"deleted\":true,\"disk_deleted\":")+(disk_deleted?"true}":"false}");return;
                    }
                    if(c.op=="restore") {
                        // Admission above deliberately leaves the live identity invalid.
                    } else if(c.op=="checkpoint") {
                        if(c.durability=="disk") {
                            session_progress("save");
                            const auto td=std::chrono::steady_clock::now();
                            if(session_path_managed)session_resources->disk.put(c.id,session_request->session_input->snapshot);
                            else sc::publish(c.path,session_request->session_input->snapshot);
                            session_disk_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-td).count();
                        }
                    } else if(c.op=="park") {
                        if(session_resources->ram.active()==c.id) {
                            session_device_touched=true;p.wait_io();STRIX_CUDA_CHECK(cudaDeviceSynchronize());
                            p.reset_sequence();session_resources->ram.clear_active();
                        }
                    } else {
                        // All schema, digest, mode and shape validation precedes this point.
                        session_device_touched=true;
                        if(session_request->runtime)std::tie(PERSIST_RUNTIME_VARS)=*session_request->runtime;
                        else std::tie(PERSIST_RUNTIME_VARS)=launch_runtime;
                        apply_env();configure_request();reset_request_state();
                        p.wait_io();STRIX_CUDA_CHECK(cudaDeviceSynchronize());
                        const auto tc=std::chrono::steady_clock::now();
                        const bool keep_live=session_keep_live && c.op=="turn";
                        if(!keep_live)p.import_state(*session_request->session_attention);
                        if(session_request->session_draft && !keep_live) {
                            DeviceScope ds(ddev);CudaDrafterState d;d.start=session_request->session_draft->start;d.next=session_request->session_draft->next;
                            d.vocabulary={session_request->session_draft->core,session_request->session_draft->extra};
                            for(auto& ring:session_request->session_draft->rings)d.rings.push_back(ring);
                            drafter->import_state(d);
                        }
                        session_copy_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tc).count();
                        session_progress("imported");
                        session_resources->ram.activate(c.id);
                        if(c.op!="resume") {
                            io.logits=chat_logits;io.draft_hidden=c.dspark?hidden_cap:nullptr;
                            // Match ordinary chats0/schats0: no speculative expert
                            // prefetch budget or diagnostic Engram rereads in prompts.
                            set_predict(0);tier->set_budgets(spares,0,pf_token);
                            engram_check_on=false;p.set_engram_check(false);
                            auto f=session_request->session_input->snapshot.frontier;
                            auto next=static_cast<std::uint32_t>(f.pending);
                            const auto save_rows=[&](const std::vector<std::uint32_t>& rows,const CudaStepResult& result,std::uint32_t kept,bool generated) {
                                if(!session_request->session_logits.is_open()||!generated)return;
                                sc::require(result.logits.size()%result.tokens.size()==0,"missing session raw logits");
                                const auto width=result.logits.size()/result.tokens.size();
                                session_request->session_logits.write(reinterpret_cast<const char*>(result.logits.data()),
                                    static_cast<std::streamsize>(kept*width*sizeof(float)));
                                session_request->session_token_dump.write(reinterpret_cast<const char*>(rows.data()),
                                    static_cast<std::streamsize>(kept*sizeof(std::uint32_t)));
                                sc::require(bool(session_request->session_logits)&&bool(session_request->session_token_dump),"session dump write failed");
                            };
                            const auto consume=[&](const std::vector<std::uint32_t>& tokens) {
                                sc::ProgressCadence cadence;
                                const auto pending=ingest_prompt(tokens, static_cast<std::uint32_t>(tokens.size()), c.dspark,
                                    "session-prompt", [&](auto done, auto total, auto windows) {
                                        if(cadence.ready(windows,done==total))session_progress("prompt",done,total);
                                    });
                                if(!tokens.empty())next=pending;
                            };
                            if(!c.control_prefix.empty()) {
                                consume(session_request->session_prefix);++f.turns;
                                // L reaches P here: no capture, checkpoint, import or reset.
                            }
                            consume(session_request->session_tokens);
                            io.logits=session_request->session_logits.is_open() || chat_logits;
                            sc::ProgressCadence decode_progress;std::uint32_t decode_windows=0;
                            for(std::uint32_t at=0;at<c.steps;) {
                                std::vector<std::uint32_t> rows{next};
                                std::uint32_t k=0;
                                if(c.dspark&&c.steps-at>1) {
                                    DeviceScope ds(ddev);const auto d=drafter->draft(next);
                                    k=std::min(c.fixed,c.steps-at-1);
                                    for(std::uint32_t j=1;j<=k;++j)rows.push_back(d.tokens[j]);
                                }
                                WindowTimes wt;const auto result=run_window(rows,wt);
                                std::uint32_t accepted=0;
                                while(accepted<k&&rows[accepted+1]==result.tokens[accepted])++accepted;
                                const auto kept=accepted+1;next=result.tokens[accepted];
                                session_request->session_output.insert(session_request->session_output.end(),rows.begin(),rows.begin()+kept);
                                save_rows(rows,result,kept,true);p.commit_window(kept,next);
                                if(c.dspark)append_committed(kept);at+=kept;f.emitted+=kept;
                                if(decode_progress.ready(++decode_windows,at==c.steps))session_progress("decode",at,c.steps);
                            }
                            f.consumed=f.accepted=p.position();f.pending=next;
                            f.draft=c.dspark?p.position():0;++f.turns;
                            if(session_request->automatic){f.emitted=c.steps;f.turns=1;}
                            const auto ts=std::chrono::steady_clock::now();
                            session_progress("capture");
                            auto snapshot=session_capture_now(f);
                            snapshot.lineage=session_request->session_input->snapshot.lineage;
                            sc::require(session_request->session_input->snapshot.committed_generation<UINT64_MAX,"session generation exhausted");
                            snapshot.committed_generation=session_request->session_input->snapshot.committed_generation+1;
                            session_progress("captured");
                            session_capture_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
                            session_validate(snapshot);
                            if(!c.dump.empty()) {
                                session_request->session_logits.flush();session_request->session_token_dump.flush();
                                sc::require(bool(session_request->session_logits)&&bool(session_request->session_token_dump),"session dump flush failed");
                                sc::write_file(c.dump+".attention.bin",snapshot.attention);
                                sc::write_file(c.dump+".drafter.bin",snapshot.drafter);
                                sc::write_file(c.dump+".frontier.bin",snapshot.frontier.encode());
                            }
                            session_resources->ram.put(c.id,std::move(snapshot));
                            session_resources->ram.activate(c.id);
                        }
                    }
                    }
                    session_progress("complete");
                    const auto& saved=session_resources->ram.get(c.id);
                    std::ostringstream output_ids;output_ids<<"[";
                    for(std::size_t j=0;j<session_request->session_output.size();++j)output_ids<<(j?",":"")<<session_request->session_output[j];
                    output_ids<<"]";
                    session_request->session_result="{\"tokens\":"+output_ids.str()+",\"session\":"+rs::json_str(c.id)+",\"operation\":"+rs::json_str(c.op=="restore"?"resume":c.op)+
                        ",\"history_hit\":"+(session_request->history_hit?"true":"false")+
                        ",\"recomputed\":"+(session_request->recompute?"true":"false")+
                        ",\"durability\":"+rs::json_str(c.op=="checkpoint"?c.durability:"ram")+
                        ",\"bytes\":"+std::to_string(saved.size())+",\"consumed\":"+std::to_string(saved.frontier.consumed)+
                        ",\"emitted\":"+std::to_string(saved.frontier.emitted)+",\"pending\":"+std::to_string(saved.frontier.pending)+
                        ",\"compat\":"+rs::json_str(session_resources->compat)+",\"attention_sha256\":"+rs::json_str(sc::sha(saved.attention))+
                        ",\"lineage\":"+rs::json_str(saved.lineage)+",\"generation\":"+std::to_string(saved.committed_generation)+
                        ",\"drafter_sha256\":"+rs::json_str(sc::sha(saved.drafter))+",\"frontier_sha256\":"+rs::json_str(sc::sha(saved.frontier.encode()))+
                        ",\"read_verify_ms\":"+std::to_string(session_read_verify_ms)+",\"restore_copy_ms\":"+std::to_string(session_copy_ms)+
                        ",\"capture_ms\":"+std::to_string(session_capture_ms)+",\"write_fsync_ms\":"+std::to_string(session_disk_ms)+
                        ",\"wall_ms\":"+std::to_string(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count())+"}";
                };

                // 1. the request's configuration: nothing on the device is touched, any error is the request's (rc 2)
                try {
                    args = rs::read_args_file(spool.running(*name));
                    if(!session_config.enabled&&!sc::explicit_request(args))sc::ignore_name(args);
                    session_attempt=std::any_of(args.begin(),args.end(),[](const auto& a){return a.rfind("--session",0)==0;});
                    if (args.size() == 1U && (args[0] == "--quit" || args[0] == "--ping")) {
                        quit = args[0] == "--quit";
                        ping = !quit;
                    } else if (session_attempt) {
                        session_request.emplace();
                        session_used=true;
                        // Validate against the launch configuration, not an earlier ordinary request's values.
                        std::tie(PERSIST_RUNTIME_VARS)=launch_runtime;
                        apply_env();
                        if(!sc::explicit_request(args)) {
                            auto& r=*session_request;r.automatic=true;
                            r.session_command.id=sc::automatic_name(args);
                            rs::check_request_args(args);parse_args(args);prepare_request();
                            sc::require(phases.size()==1&&chats_arg.find(',')==std::string::npos,"named prompt requires one phase and chat");
                            auto& c=r.session_command;c.used=true;c.op="turn";c.steps=chat_steps;c.dump=qual_dump;
                            if(phases[0]=="chats0")c.dspark=false;
                            else {
                                sc::require(phases[0].rfind("schats0f",0)==0&&phases[0].size()==9&&phases[0].back()>='1'&&phases[0].back()<='5',"named prompt requires chats0 or schats0f1..5");
                                c.dspark=true;c.fixed=static_cast<std::uint32_t>(phases[0].back()-'0');
                            }
                            const auto& prompt=seqs.at(std::stoul(chats_arg)).second;
                            r.full_history=session_seed_holder.materialize(imported0).tokens;
                            r.full_history.insert(r.full_history.end(),prompt.begin(),prompt.begin()+std::min<std::size_t>(chat_prompt,prompt.size()));
                            open_outputs();configure_request();
                            r.runtime=std::make_tuple(PERSIST_RUNTIME_VARS);
                        } else session_request->session_command=sc::parse(args);
                        try{session_preflight();}
                        catch(...) {
                            // A name-only request whose automatic admission failed runs as the ordinary
                            // request it names: the full prompt from scratch, the same as without sessions.
                            if(!(session_request->automatic&&session_admission_checked&&!session_operational))throw;
                            std::cerr<<"session cache: off for this request ("<<session_config.reason<<"); running it without a session\n";
                            session_request.reset();session_attempt=false;session_used=false;sc::ignore_name(args);
                        }
                    }
                    if (!session_attempt && !quit && !ping) {
                        rs::check_request_args(args);
                        std::tie(PERSIST_RUNTIME_VARS) = launch_runtime;  // req_env = the launch --env list
                        parse_args(args);                                    // + the request's
                        apply_env();
                        if (reset_cache != "warm" && reset_cache != "open") usage("--reset-cache warm | open");
                        if (reset_cache == "open" && !TierSite::can_reset_cache()) usage("--reset-cache open: this build's tierstream has no ts_dg_reset_open");
                        if (!TierSite::can_set_sched() && (wb_defer != launch_wb_defer || flush_ready != launch_flush_ready))
                            usage("--wb-defer / --flush-ready: this build's tierstream has no ts_dg_set_sched (a new worker is needed)");
                        prepare_request();
                        if (std::uint64_t{imported.position} + needed_extra64() > o.max_positions)
                            usage("the request needs more positions than the worker's --max-positions capacity");
                        for (const auto& ph : phases) {
                            if (ph.rfind("gsmoke", 0) == 0 || ph.rfind("csmoke", 0) == 0)
                                usage("gsmoke / csmoke are not served (temporary capture destinations end up in graphs)");
                            const bool hooks = ph.rfind("pressure", 0) == 0 || ph.rfind("smokep", 0) == 0 || ph.rfind("wsmokep", 0) == 0 || ph.rfind("wforcep", 0) == 0;
                            if (hooks && !tier_test_hooks) usage(("phase " + ph + " needs a worker opened with --test-hooks").c_str());
                            const bool drafts = ph.rfind("spec", 0) == 0 || ph.rfind("schats", 0) == 0;
                            if (drafts && !drafter) usage(("phase " + ph + " needs a worker loaded with --draft").c_str());
                        }
                        open_outputs();
                        configure_request();
                    }
                } catch (const std::exception& e) {
                    if(session_attempt)++session_refusals;
                    if(const auto* failure=dynamic_cast<const sc::Error*>(&e))session_error=*failure;
                    rc = 2;
                    error = e.what();
                }
                // 2. the run: a failed gate is the request's (rc 1, the next request resets everything); an exception
                // means the device / library state is not trusted any more: answer and exit (fatal)
                if (rc == 0 && !quit && !ping) {
                    try {
                        emit("{\"event\":\"request_start\",\"name\":" + rs::json_str(*name) + ",\"seq\":" + std::to_string(served) + "}");
                        if (!session_used && reset_cache == "open") {  // the expert cache back to the open state (quiescent: drained above)
                            drain_request_state();
                            const auto tr = std::chrono::steady_clock::now();
                            tier->reset_cache();
                            emit("{\"event\":\"reset_cache\",\"mode\":\"open\",\"ms\":" +
                                 std::to_string(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count()) + "}");
                        }
                        if(session_used) {failures=0;chat_plain.clear();session_execute();}
                        else {if(session_resources)session_resources->ram.clear_active();reset_request_state();run_request();}
                        flush_deferred();  // the last step's record (and its compare) before the verdict
                        // this request's plain arms against the earlier requests' (then kept for the next requests)
                        for (const auto& [k, v] : chat_plain) {
                            const std::string c = k.substr(k.rfind('-') + 1);
                            const auto it = worker_canon.find(canon_key(c));
                            if (it == worker_canon.end()) {
                                worker_canon.emplace(canon_key(c), v);
                            } else if (!std::equal(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(std::min(v.size(), it->second.size())), it->second.begin())) {
                                emit("{\"event\":\"chat_tokens_cross\",\"key\":\"" + k + "\",\"pass\":false}");
                                ++failures;
                            } else if (v.size() > it->second.size()) {
                                it->second = v;  // the same prefix, longer: keep the longer
                            }
                        }
                        rc = failures ? 1 : 0;
                        if (failures && !serve_keep_on_fail) {  // a failed gate ends the worker (qualification default)
                            error = "a gate failed (" + std::to_string(failures) + "): the worker exits (--serve-keep-on-fail to continue)";
                            fatal = true;
                        }
                    } catch (const std::exception& e) {
                        rc = 1;
                        error = e.what();
                        if(const auto* failure=dynamic_cast<const sc::Error*>(&e))session_error=*failure;
                        fatal = !session_used || session_device_touched; // CPU control failures leave the device intact
                    }
                    if (!fatal) {
                        try {  // quiescent before the answer: Engram reads and every queued copy / write-back retired
                            p.wait_io();
                            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
                        } catch (const std::exception& e) {
                            rc = 1;
                            error = e.what();
                            fatal = true;
                        }
                    }
                    if (meter && !fatal) {  // quiescent: the request's peak against the caps
                        try {
                            if (!budget_point("request", *name) && caps.capped()) {
                                ++failures;
                                rc = 1;
                                if (error.empty()) error = "budget: a measured peak exceeds a total cap (see the budget event)";
                            }
                        } catch (const std::exception& e) {
                            rc = 1;
                            error = e.what();
                            fatal = true;
                        }
                    }
                    if (!error.empty()) emit("{\"event\":\"error\",\"what\":" + rs::json_str(error) + "}");
                    emit("{\"event\":\"request_end\",\"name\":" + rs::json_str(*name) + ",\"rc\":" + std::to_string(rc) +
                         p.weight_io_fields() + "}");
                    ++served;
                }
                const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_req).count();
                std::ostringstream os;
                os << "{\"name\":" << rs::json_str(*name) << ",\"rc\":" << rc << ",\"load_id\":" << load_id << ",\"served\":" << served
                   << (session_used ? ",\"session_result\":"+session_request->session_result : std::string{}) << ",\"wall_s\":" << wall_s << ",\"failures\":" << failures << ",\"fatal\":" << (fatal ? "true" : "false")
                   << ",\"quit\":" << (quit ? "true" : "false") << ",\"json\":" << rs::json_str(json_out);
                if(session_attempt&&!error.empty()) {
                    const auto failure=session_used?session_error.value_or(sc::protocol_error(error)):sc::Error("invalid_request",400,error);
                    os<<",\"error\":{\"code\":"<<rs::json_str(failure.code)
                      <<",\"message\":"<<rs::json_str(error)<<"},\"status\":"<<failure.status<<"}";
                } else os<<",\"error\":"<<rs::json_str(error)<<"}";
                if (js.is_open()) js.flush();
                if (tl.is_open()) tl.flush();
                if (trace_out.is_open()) trace_out.flush();
                spool.finish(*name, os.str());
                idle_from = std::chrono::steady_clock::now();
                if (fatal || quit || (serve_max != 0U && served >= serve_max)) break;
            }
            g_serving = false;
            spool.release();
            failures = 0;
        }
        if (!residency_save.empty()) {  // the residency now (most recent first), for the next load's seed
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            const auto keys = tier->residency();
            const auto& pin = strix::deepseek41::pinned_file(strix::deepseek41::PinnedFileKind::target_q2);
            std::ostringstream rs_out;
            rs_out << "ds41-residency-v1 sha256=" << pin.sha256_hex << " layers=" << L << " experts=" << E << " n=" << keys.size() << "\n";
            for (std::size_t i = 0; i < keys.size(); ++i) rs_out << keys[i] << ((i + 1) % 32 == 0 ? "\n" : " ");
            rs::write_atomic(residency_save, rs_out.str() + "\n");
            emit("{\"event\":\"residency_saved\",\"keys\":" + std::to_string(keys.size()) + "}");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-ds41-replica: %s\n", e.what());
        emit(std::string("{\"event\":\"error\",\"what\":\"") + e.what() + "\"}");
        return 1;
    }
    return failures ? 1 : 0;
}
