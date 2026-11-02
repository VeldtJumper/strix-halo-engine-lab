#pragma once

#include "strix/serve/short_first.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Decode-priority scheduling of the two-rank serve (strix-deepseek41-serve, --prefill-path device), STRIX_DS41_SERVE_SCHED:
//
//   off: an admitted prompt's device prefill runs at once and the decoding requests wait for all of it.
//   defer: deferred admission. A request whose device prefill would stall decoding requests waits in an admission
//     queue (no slot, no KV pool room) and is admitted when no request is decoding, or when it has waited
//     STRIX_DS41_SERVE_SCHED_STARVE_MS (default 30000), or when the cost rule says so: admitting now costs the others
//     N_decoding x T_dev(P) (each one stalls for the prefill), waiting costs this request W (the time until the decoding
//     requests end, estimated as the largest remaining max_tokens x the recent step time); admit now when
//     N x T_dev(P) <= W (J = own TTFT + N x delay is then the lower). T_dev(P) = a + b P seconds
//     (STRIX_DS41_SERVE_SCHED_TDEV="a,b", default the measured 0.17 s + 1.69 ms/token). A state imported for the request
//     while it waits (POST /admin/import-state, e.g. the prompt prefilled on another device) is taken at once and the
//     local prefill never runs.
//   interleave (device default): the prefill runs as a PrefillTask in bounded units between decode steps: every scheduler tick runs one
//     batch step of the decoding requests, then one unit of the pending prefill, until it completes; then the request
//     joins the batch. While no request decodes and no other prompt waits for its prefill, a tick runs the rest of the
//     piece at once (STRIX_DS41_SERVE_IDLE_WHOLE=0: the bounded service while idle too). Default pieces stay unchanged;
//     resumable finest units target STRIX_DS41_SERVE_SERVICE_MS (default 250). With no legacy controls, 0 restores
//     whole default pieces while idle and row units beside decoders.
//     Explicit legacy controls override SERVICE_MS; positive UNIT_MS > UNIT_BUDGET > UNIT_ROWS.
//     Prompts shorter than STRIX_DS41_SERVE_SCHED_MIN_ROWS (default 64) take the direct path (whole prefill at once).
//     Prompts resumed from a state imported while the task is pending abort the task and take the state.
//
// Every decision is rank 0's; rank 1 runs what the control channel tells it (the same commands in the same order, so
// the collectives of the two ranks stay in the same order). Pure functions here (unit tests: deepseek41_serve_sched_tests).
namespace strix::deepseek41::sched {

enum class Mode : std::uint32_t { off = 0, defer = 1, interleave = 2 };

struct Options {
    Mode mode = Mode::off;
    double starve_seconds = 30.0;
    double tdev_a = 0.17, tdev_b = 0.00169;  // T_dev(P) = a + b P seconds (measured on the two-node pair)
    std::uint32_t unit_rows = 256;           // interleave: rows of one unit while requests decode
    // interleave: prompts shorter than this run their whole prefill at once (no task). On the pair T_dev(64) ~ 0.28 s:
    // one such pause is close to a few ticks of the interleave, while a task would add one decode step per unit to its
    // TTFT (measured: 22-26-token prompts 0.18 s at once vs 0.6-0.78 s as 10-unit tasks).
    std::uint32_t min_rows = 64;
    std::uint32_t unit_budget = 0;           // interleave: PrefillTask::step budget per unit (0: the whole piece)
    // interleave with the resumable DevicePrefill (phase B): the units of a tick fit in unit_ms (rank 0's prediction,
    // DevicePrefill PrefillTask::units_within), else unit_budget units per tick; pieces stay the default ones.
    double unit_ms = 0.0;
    double service_ms = 250.0;
    bool legacy_units = false, short_first = true;
    // interleave, service-target units: while no request decodes and no other prompt waits for its prefill, a tick runs
    // the rest of the current piece at once (the whole-piece form: no per-unit drain and control round, MoE chunks of the
    // prefill's own size; same bits). A request that arrives meanwhile waits for the piece. STRIX_DS41_SERVE_IDLE_WHOLE=0:
    // bounded units while idle too.
    bool idle_whole = true;
    [[nodiscard]] double target_ms() const noexcept { return legacy_units ? unit_ms : service_ms; }
    [[nodiscard]] const char* budget_source() const noexcept {
        return legacy_units ? (unit_ms > 0.0 ? "legacy-ms" : unit_budget > 0U ? "fixed-units" : "row-units") :
                              (service_ms > 0.0 ? "service-target" : "coarse");
    }
    // the resumable task's MoE chunk (bounds a MoE unit; same on both ranks; 0: the prefill's own 4096). Smaller chunks
    // run on row views of the prefill's own MoE scratch (no extra memory); 2048 = DevicePrefill's pair-gated task default
    std::uint32_t moe_chunk_rows = 2048;
    [[nodiscard]] bool resumable() const noexcept { return target_ms() > 0.0 || unit_budget > 0U; }
    bool log = false;                        // one stderr line per decision
};

// The mode of `env` (STRIX_DS41_SERVE_SCHED's value) for a serve whose prefill path is the device (true) or rows:
// unset = interleave with the device prefill (default since P17.2), off with rows; defer / interleave need the device
// prefill (rejected for rows, before anything connects or loads).
[[nodiscard]] inline Mode mode_for(const std::string& m, bool device_prefill) {
    if (m == "0" || m == "off") return Mode::off;
    if (m.empty()) return device_prefill ? Mode::interleave : Mode::off;
    Mode mode;
    if (m == "defer") mode = Mode::defer;
    else if (m == "interleave" || m == "1") mode = Mode::interleave;
    else throw std::invalid_argument("STRIX_DS41_SERVE_SCHED is off, defer or interleave, not '" + m + "'");
    if (!device_prefill) throw std::invalid_argument("STRIX_DS41_SERVE_SCHED=" + m + " needs --prefill-path device");
    return mode;
}

template <class Env>
inline Options options_from(bool device_prefill, Env env) {
    Options o;
    // default interleave with the device prefill since P17.2 (decode stalls <= ~1 s while a long prompt prefills)
    o.mode = mode_for(env("STRIX_DS41_SERVE_SCHED"), device_prefill);
    if (const auto s = env("STRIX_DS41_SERVE_SCHED_STARVE_MS"); !s.empty()) o.starve_seconds = std::stod(s) / 1e3;
    if (const auto s = env("STRIX_DS41_SERVE_SCHED_TDEV"); !s.empty()) {
        const auto c = s.find(',');
        if (c == std::string::npos) throw std::invalid_argument("STRIX_DS41_SERVE_SCHED_TDEV is \"a,b\" (seconds, seconds per token)");
        o.tdev_a = std::stod(s.substr(0, c));
        o.tdev_b = std::stod(s.substr(c + 1));
    }
    if (const auto s = env("STRIX_DS41_SERVE_SCHED_UNIT_ROWS"); !s.empty()) o.unit_rows = static_cast<std::uint32_t>(std::stoul(s));
    if (const auto s = env("STRIX_DS41_SERVE_SCHED_MIN_ROWS"); !s.empty()) o.min_rows = static_cast<std::uint32_t>(std::stoul(s));
    if (o.unit_rows < 16U || o.unit_rows % 8U != 0U) throw std::invalid_argument("STRIX_DS41_SERVE_SCHED_UNIT_ROWS: a multiple of 8, at least 16");
    if (const auto s = env("STRIX_DS41_SERVE_SCHED_UNIT_BUDGET"); !s.empty()) o.unit_budget = static_cast<std::uint32_t>(std::stoul(s));
    const auto milliseconds = [](const std::string& v) {
        std::size_t used = 0;
        const double ms = std::stod(v, &used);
        if (used != v.size()) throw std::invalid_argument("serve milliseconds: trailing input");
        return ms;
    };
    if (const auto v = env("STRIX_DS41_SERVE_SCHED_UNIT_MS"); !v.empty()) o.unit_ms = milliseconds(v);
    if (const auto v = env("STRIX_DS41_SERVE_SERVICE_MS"); !v.empty()) o.service_ms = milliseconds(v);
    if (!std::isfinite(o.service_ms) || o.service_ms < 0.0 || !std::isfinite(o.unit_ms) || o.unit_ms < 0.0)
        throw std::invalid_argument("serve service/unit milliseconds must be finite and nonnegative");
    const auto ms = env("STRIX_DS41_SERVE_SCHED_UNIT_MS");
    const auto budget = env("STRIX_DS41_SERVE_SCHED_UNIT_BUDGET");
    o.legacy_units = !ms.empty() || !budget.empty() || !env("STRIX_DS41_SERVE_SCHED_UNIT_ROWS").empty();
    if (o.unit_ms > 0.0) o.unit_budget = 0U;
    if (const auto v = env("STRIX_DS41_SERVE_SHORT_FIRST"); !v.empty()) {
        if (v != "0" && v != "1") throw std::invalid_argument("STRIX_DS41_SERVE_SHORT_FIRST is 0 or 1");
        o.short_first = v == "1";
    }
    if (const auto v = env("STRIX_DS41_SERVE_SCHED_MOE_CHUNK"); !v.empty()) o.moe_chunk_rows = static_cast<std::uint32_t>(std::stoul(v));
    if (const auto v = env("STRIX_DS41_SERVE_IDLE_WHOLE"); !v.empty()) {
        if (v != "0" && v != "1") throw std::invalid_argument("STRIX_DS41_SERVE_IDLE_WHOLE is 0 or 1");
        o.idle_whole = v == "1";
    }
    o.log = o.mode != Mode::off && env("STRIX_DS41_SERVE_SCHED_LOG") != "0";
    return o;
}

inline Options options_from_env(bool device_prefill) {
    return options_from(device_prefill, [](const char* k) -> std::string {
        const char* v = std::getenv(k);
        return v ? std::string(v) : std::string();
    });
}

// --prefill-rows auto (serve): the device prefill's piece size. Each piece re-reads every routed expert and replays the
// decoder cone, so fewer, larger pieces are faster (pieces are bitwise-exact for any size).
// prefill_piece_bytes: the device prefill's row scratch for pieces of `rows` rows at `context` positions (DevicePrefill's
// T-sized buffers, deepseek41 shapes, row split: 352 KiB per row + the context's candidate bits per row + 1.2 GiB of
// row-independent buffers, rounded up; measured 6.37 GiB at 16384 rows and 11.69 GiB at 32768 rows, context 49152).
[[nodiscard]] constexpr std::uint64_t prefill_piece_bytes(std::uint32_t rows, std::uint32_t context) noexcept {
    return (std::uint64_t{rows} + 32U) * ((std::uint64_t{352} << 10U) + (std::uint64_t{context} + 8U) / 8U) + (std::uint64_t{1229} << 20U);
}
struct AutoPrefillRoom {
    std::uint64_t device_free = 0, mem_available = 0;  // after the batch forward's weights (mem_available 0: unknown)
    std::uint64_t own_weights = 0;                     // DevicePrefill::owned_weight_bytes
    std::uint64_t drafter = 0;                         // a drafter still to load on this rank (its peak: device + host copy); the serve loads it first
    std::uint64_t margin = std::uint64_t{3} << 30U;
    // the 32768-row candidate: only where prefill_piece_bytes was measured (context <= 65536, the default attention
    // window rows, no large slabs); elsewhere other row buffers can exceed the estimate
    bool wide_pieces = false;
};
inline constexpr std::uint32_t kAutoPrefillRows[] = {32768U, 16384U, 8192U, 4096U};
inline constexpr std::uint32_t kAutoPrefillFallback = 2048U;
// AutoPrefillRoom::wide_pieces for a serve configuration: the context and prefill switches prefill_piece_bytes was
// measured with (context <= 65536, the default attention window rows, no large slabs, and none of the opt-in prefill
// forms that keep their own buffers outside the estimate: FP16 dense, BF16 attention, a prefill expert-TP override that
// loads its own expert copies). env(name): the value or "".
template <class Env>
[[nodiscard]] bool auto_wide_pieces(std::uint32_t context, Env env) {
    return context <= 65536U && env("STRIX_DS41_PREFILL_WINDOW").empty() && env("STRIX_DS41_PREFILL_SLABS") != "1" &&
           env("STRIX_DS41_PREFILL_FP16_DENSE") != "1" && env("STRIX_DS41_PREFILL_BF16_ATTN") != "1" && env("STRIX_DS41_PREFILL_EXPERT_TP").empty();
}
// The largest candidate (at most the context) whose scratch fits in min(device free, MemAvailable) less the prefill's
// own weights, the drafter's peak and the margin; else min(2048, the context) (at least 32 rows, a multiple of 8) when that
// fits too; else 0: no piece size fits (the serve refuses before READY instead of overcommitting into the margin).
[[nodiscard]] inline std::uint32_t auto_prefill_rows(std::uint32_t context, const AutoPrefillRoom& r) noexcept {
    std::uint64_t room = r.mem_available != 0U ? std::min(r.device_free, r.mem_available) : r.device_free;
    const std::uint64_t reserve = r.own_weights + r.drafter + r.margin;
    room = room > reserve ? room - reserve : 0U;
    for (const auto rows : kAutoPrefillRows)
        if (rows <= context && (rows <= 16384U || r.wide_pieces) && prefill_piece_bytes(rows, context) <= room) return rows;
    const std::uint32_t fallback = std::max<std::uint32_t>(32U, std::min(kAutoPrefillFallback, context / 8U * 8U));
    return prefill_piece_bytes(fallback, context) <= room ? fallback : 0U;
}
// The drafter's peak on a drafting rank, from its file: the device copy (weights + scratch + contexts + staging) while
// the host copy of the loaded file is still alive (UMA: the same memory). Measured: a 7.42 GiB draft file took 15.1 GiB.
[[nodiscard]] constexpr std::uint64_t drafter_peak_bytes(std::uint64_t draft_file_bytes) noexcept {
    return 2U * draft_file_bytes + (std::uint64_t{1} << 30U);
}

// Disabled service without legacy units retains whole pieces while idle.
[[nodiscard]] inline std::uint32_t row_budget(const Options& o, bool idle) noexcept {
    return o.resumable() || (idle && !o.legacy_units) ? UINT32_MAX : o.unit_rows;
}

// A tick's PrefillTask budget: 0 (the rest of the piece at once) while idle_whole and no request decodes or waits for its
// prefill (service-target units only; explicit legacy controls keep their units), else unit_budget.
template <class Predict>
[[nodiscard]] std::uint32_t tick_budget(const Options& o, bool decoding, bool others_waiting, Predict predict) {
    if (o.idle_whole && !o.legacy_units && o.service_ms > 0.0 && !decoding && !others_waiting) return 0U;
    return o.target_ms() > 0.0 ? std::max(1U, predict(o.target_ms())) : o.unit_budget;
}
template <class Predict>
[[nodiscard]] std::uint32_t unit_budget(const Options& o, Predict predict) {
    return o.target_ms() > 0.0 ? std::max(1U, predict(o.target_ms())) : o.unit_budget;
}

// The numeric identity and the collective protocol both serve ranks must share, compared in the hello before READY:
// the weights' content identity (prefix_namespace_from_env: overlay, CB4, trellis packs / mix / sites, SHEXP_TP, EP map,
// the decode identity EP / expert TP) and the DSpark configuration (drafter, its ranks, window policy, vocabulary).
// Preserve the pre-session hello bytes whenever the feature is disabled.
[[nodiscard]] inline std::string session_serve_identity_text(const std::string& base, std::uint64_t ram_mb, bool disk) {
    return ram_mb ? std::to_string(ram_mb) + ":" + std::to_string(disk) + ":" + base : base;
}
[[nodiscard]] inline bool session_admin_route(bool enabled, std::string_view path) {
    return enabled && (path == "/v1/sessions" || path == "/v1/cache/stats" || path.starts_with("/v1/sessions/"));
}
[[nodiscard]] inline std::string serve_identity_text(const std::string& namespace_text, std::uint64_t dspark_bytes, std::uint32_t dspark_ranks,
                                                     const std::string& dspark_window, const std::string& dspark_vocab) {
    std::string t = namespace_text + "dspark=";
    if (dspark_bytes == 0U) return t + "-;";
    return t + std::to_string(dspark_bytes) + ":ranks=" + std::to_string(dspark_ranks) + ":window=" + dspark_window + ":vocab=" + dspark_vocab + ";";
}

[[nodiscard]] inline const char* mode_name(Mode m) {
    return m == Mode::defer ? "defer" : m == Mode::interleave ? "interleave" : "off";
}

[[nodiscard]] inline double tdev_seconds(const Options& o, std::uint64_t prompt_tokens) {
    return o.tdev_a + o.tdev_b * static_cast<double>(prompt_tokens);
}

enum class Admit : std::uint32_t { defer = 0, idle = 1, starved = 2, cost = 3, draining = 4 };
[[nodiscard]] inline const char* admit_name(Admit a) {
    switch (a) {
        case Admit::idle: return "idle";
        case Admit::starved: return "starved";
        case Admit::cost: return "cost";
        case Admit::draining: return "draining";
        default: return "defer";
    }
}

// Deferred admission of a request whose device prefill would stall `decoding` requests. waited: seconds since it was
// taken from the queue; wait_estimate: seconds until no request decodes (W).
[[nodiscard]] inline Admit admit(const Options& o, std::uint64_t prompt_tokens, std::uint32_t decoding, double waited, double wait_estimate,
                                 bool draining = false) {
    if (draining) return Admit::draining;
    if (decoding == 0U) return Admit::idle;
    if (waited >= o.starve_seconds) return Admit::starved;
    if (static_cast<double>(decoding) * tdev_seconds(o, prompt_tokens) <= wait_estimate) return Admit::cost;
    return Admit::defer;
}

// W: the time until no request decodes, from each decoding request's remaining tokens (max_tokens - generated; an upper
// bound, a stop token ends it earlier) and the recent seconds per batch step.
[[nodiscard]] inline double wait_estimate(std::span<const std::uint64_t> remaining_tokens, double step_seconds) {
    std::uint64_t most = 0;
    for (const auto r : remaining_tokens) most = std::max(most, r);
    return static_cast<double>(most) * step_seconds;
}

// interleave: the end of the next unit of a prefill at row `at` (rows so far), given the default piece ends of the
// prompt (plan_prefill_pieces: they stay piece ends, so the prefix cache inserts at the same boundaries) and a row
// budget (0: up to the next default end). A unit ends on a multiple of 8 (extend() continues from there) and leaves
// at least min_rows rows to the next default end; its own rows are at least min_rows too.
[[nodiscard]] inline std::uint32_t next_unit_end(std::uint32_t at, std::span<const std::uint32_t> default_ends, std::uint32_t budget_rows,
                                                 std::uint32_t min_rows) {
    const auto it = std::upper_bound(default_ends.begin(), default_ends.end(), at);
    if (it == default_ends.end()) throw std::invalid_argument("next_unit_end: the prefill is complete");
    const std::uint32_t d = *it;
    if (budget_rows == 0U || d - at <= budget_rows) return d;
    std::uint32_t end = at + budget_rows / 8U * 8U;
    if (d - end < min_rows) end = d >= min_rows ? (d - min_rows) / 8U * 8U : at;
    if (end <= at || end - at < min_rows) return d;
    return end;
}

struct UnitCompletion {
    bool done;
    std::uint32_t rows, units;
    bool operator==(const UnitCompletion&) const = default;
};

// Both ranks apply the lead's control verbatim, including abort before any further step.
enum class UnitResult { pending, complete, cancelled };
template <class Task>
UnitResult apply_unit(Task& task, std::uint32_t end, std::uint32_t budget, bool abort) {
    if (abort) { task.cancel(); return UnitResult::cancelled; }
    return task.step(end, budget) ? UnitResult::complete : UnitResult::pending;
}

// A request's prompt prefill as a sequence of bounded units (both ranks hold one, created by the same command and
// stepped by the same commands, so its collectives run in the same order on both ranks). The default implementation
// is the serve's (SlotPrefill::Task): one unit = one DevicePrefill run of [at, end) rows, end chosen by rank 0
// (next_unit_end) and sent to rank 1; `budget` is passed through for a resumable DevicePrefill (0: the whole piece).
// Contract for an implementation:
//  - step() returns with no collective of the prefill in flight (a decode step's collectives follow it);
//  - the prefill's state, scratch and row buffers are its own (DevicePrefill's), never the BatchForward's slot or
//    workspace; the request's BatchForward slot is written only by finish() (BatchForward::import_state);
//  - the bits of every row and of the final state do not depend on how the prompt was cut into units.
class PrefillTask {
public:
    virtual ~PrefillTask() = default;
    // Advances the prefill to row `end` (a unit; resumable implementations may stop earlier inside the piece within
    // `budget` and return false with rows() unchanged); true when rows() == the prompt's length.
    virtual bool step(std::uint32_t end, std::uint32_t budget) = 0;
    [[nodiscard]] virtual std::uint32_t rows() const noexcept = 0;   // rows of the prompt done (at)
    [[nodiscard]] virtual std::uint32_t total() const noexcept = 0;  // the prompt's length
    [[nodiscard]] virtual bool done() const noexcept { return rows() == total(); }
    [[nodiscard]] virtual bool mid_piece() const noexcept { return false; }  // a resumable unit stopped inside a piece
    [[nodiscard]] virtual std::uint32_t piece_end() const noexcept { return rows(); }  // mid_piece: the end the piece runs to
};

}  // namespace strix::deepseek41::sched
