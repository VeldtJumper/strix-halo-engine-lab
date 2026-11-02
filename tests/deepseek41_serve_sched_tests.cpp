#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/deepseek41_serve_sched.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include "strix/serve/engine.hpp"
#include <string>
#include <vector>

// The serve's decode-priority scheduling rules (deepseek41_serve_sched.hpp): deferred admission decisions and the unit
// plan of an interleaved prefill (unit ends on multiples of 8, never a unit or a remainder shorter than min_rows, every
// default piece end kept, the prompt covered exactly).
namespace {

namespace ds = strix::deepseek41;
namespace sc = strix::deepseek41::sched;

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

void service_options() {
    std::map<std::string, std::string> env;
    const auto options = [&] { return sc::options_from(true, [&](const char* k) { return env[k]; }); };
    auto o = options();
    expect(sc::row_budget(o, true) == UINT32_MAX, "fine service keeps ordinary piece endpoints");
    expect(sc::unit_budget(o, [](double ms) { return ms == 250.0 ? 2U : 0U; }) == 2U, "lead predicts default service units");
    expect(sc::unit_budget(o, [](double) { return 0U; }) == 1U, "cold service makes progress");
    expect(o.resumable() && o.target_ms() == 250.0 && o.short_first, "default fine service, including idle");
    const auto two = [](double) { return 2U; };
    expect(o.idle_whole && sc::tick_budget(o, false, false, two) == 0U, "idle tick: the rest of the piece at once");
    expect(sc::tick_budget(o, true, false, two) == 2U, "decoders: service units");
    expect(sc::tick_budget(o, false, true, two) == 2U, "another prompt waits: service units");
    env["STRIX_DS41_SERVE_IDLE_WHOLE"] = "0";
    expect(!options().idle_whole && sc::tick_budget(options(), false, false, two) == 2U, "IDLE_WHOLE=0: service units while idle");
    env["STRIX_DS41_SERVE_IDLE_WHOLE"] = "2";
    {
        bool threw = false;
        try { (void)options(); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "IDLE_WHOLE is 0 or 1");
    }
    env.erase("STRIX_DS41_SERVE_IDLE_WHOLE");
    env["STRIX_DS41_SERVE_SERVICE_MS"] = "0";
    o = options();
    expect(!o.resumable() && sc::row_budget(o, true) == UINT32_MAX, "zero service restores idle whole pieces");
    expect(sc::row_budget(o, false) == 256U, "zero service retains rows beside decoders");
    env["STRIX_DS41_SERVE_SERVICE_MS"] = "250";
    env["STRIX_DS41_SERVE_SCHED_UNIT_ROWS"] = "128";
    expect(!options().resumable() && sc::row_budget(options(), true) == 128U, "explicit rows override idle shortcut and service");
    env["STRIX_DS41_SERVE_SCHED_UNIT_BUDGET"] = "3";
    expect(options().unit_budget == 3 && options().resumable(), "fixed budget overrides rows");
    expect(sc::unit_budget(options(), [](double) { return 99U; }) == 3U, "fixed units bypass prediction");
    expect(sc::tick_budget(options(), false, false, [](double) { return 99U; }) == 3U, "legacy units keep their budget while idle");
    env["STRIX_DS41_SERVE_SCHED_UNIT_MS"] = "12";
    expect(options().target_ms() == 12.0 && options().unit_budget == 0, "legacy ms overrides fixed budget");
    env["STRIX_DS41_SERVE_SCHED_UNIT_MS"] = "0";
    expect(options().resumable() && options().unit_budget == 3U, "zero legacy ms preserves fixed budget");
    expect(sc::unit_budget(options(), [](double) { return 99U; }) == 3U, "zero legacy ms uses fixed units");
    env["STRIX_DS41_SERVE_SCHED_UNIT_BUDGET"] = "0";
    expect(!options().resumable() && sc::row_budget(options(), true) == 128U, "zero legacy ms and budget use explicit rows");
    for (const auto* v : {"nan", "inf", "-1", "1e999", "250x"}) {
        env["STRIX_DS41_SERVE_SERVICE_MS"] = v;
        bool rejected = false;
        try { (void)options(); } catch (const std::exception&) { rejected = true; }
        expect(rejected, "invalid service target rejected");
    }
}

void short_first() {
    struct Entry { unsigned rows, skips = 0; };
    std::vector<Entry> pending{{100}, {20}, {20}};
    const auto choose = [&](bool enabled) {
        return strix::serve::select_waiting(pending, enabled, [](const auto& e) { return e.rows; },
            [](auto& e) -> unsigned& { return e.skips; });
    };
    expect(choose(false) == 0, "FIFO off switch");
    expect(choose(true) == 1, "shortest first and FIFO ties");
    for (unsigned i = 1; i < 8; ++i) expect(choose(true) == 1, "short work can overtake");
    expect(choose(true) == 0, "eight overtakes bound long request starvation");

    strix::serve::GenerationQueue queue;
    strix::serve::GenerationJob long_job, short_job;
    long_job.prompt.resize(100); short_job.prompt.resize(10);
    auto long_ticket = queue.submit(std::move(long_job));
    auto short_ticket = queue.submit(std::move(short_job));
    auto first = queue.take(false, true);
    expect(first && first->ticket == short_ticket, "queue uses shortest admission");
    expect(queue.snapshot().queued == 1 && queue.snapshot().running == 1, "reordered pool accounting");
    queue.finish(*first, {});
    auto second = queue.take(false, true);
    expect(second && second->ticket == long_ticket, "older owner preserved");
    queue.finish(*second, {});
    strix::serve::GenerationJob older;
    older.prompt.resize(100);
    auto older_ticket = queue.submit(std::move(older));
    for (unsigned i = 0; i < 9; ++i) {
        strix::serve::GenerationJob newcomer;
        newcomer.prompt.resize(10);
        auto newcomer_ticket = queue.submit(std::move(newcomer));
        auto picked = queue.take(false, true);
        expect(picked && picked->ticket == (i < 8 ? newcomer_ticket : older_ticket), "queue bounded overtaking with continuous arrivals");
        queue.finish(*picked, {});
    }
    auto last = queue.take(false, false);
    expect(last.has_value(), "unselected arrival remains queued");
    queue.finish(*last, {});
    expect(queue.snapshot().queued == 0 && queue.snapshot().running == 0, "all reordered tickets finished once");
}

void control_trace() {
    const sc::UnitCompletion boundary{false, 256U, 3U};
    expect(boundary == sc::UnitCompletion{false, 256U, 3U}, "matching unit completion");
    expect(!(boundary == sc::UnitCompletion{true, 256U, 3U}), "done mismatch rejected");
    expect(!(boundary == sc::UnitCompletion{false, 512U, 3U}), "row mismatch rejected");
    expect(!(boundary == sc::UnitCompletion{false, 256U, 4U}), "within-piece position mismatch rejected");
    struct FakeTask {
        unsigned units = 0, cancels = 0;
        bool valid = true, owned = true;
        std::vector<std::string> trace;
        bool step(unsigned, unsigned budget) {
            units += budget;
            if (units >= 6) owned = false;
            return !owned;
        }
        void cancel() { ++cancels; owned = false; valid = false; }
        sc::UnitResult receive(unsigned budget, bool abort) {
            trace.push_back(abort ? "abort" : "unit");
            const auto result = sc::apply_unit(*this, 256, budget, abort);
            if (!abort) trace.push_back("ack");
            return result;
        }
    } lead, follower;
    const auto tick = [&](unsigned budget, bool abort) {
        const auto a = lead.receive(budget, abort);
        const auto b = follower.receive(budget, abort);
        expect(a == b && lead.units == follower.units && lead.valid == follower.valid, "both ranks obey identical control");
        expect(lead.trace == follower.trace, "both ranks exchange identical unit and ack traces");
    };
    sc::Options service;
    tick(sc::unit_budget(service, [](double) { return 1U; }), false);
    // Idle prompt returns to the scheduler before completion; the arrival is visible now.
    expect(lead.units == 1, "idle arrival polled before full piece");
    tick(2, false);
    tick(99, true);  // an abort must never execute the supplied budget
    expect(lead.units == 3 && lead.cancels == 1 && !lead.valid && !lead.owned, "abort releases at boundary without settling");
    expect(lead.trace == std::vector<std::string>{"unit", "ack", "unit", "ack", "abort"}, "idle and normal units ack once; abort has no ack");
    lead = {}; follower = {};  // fresh task after slot reuse
    tick(6, false);
    expect(lead.units == 6 && follower.units == 6 && !lead.owned && !follower.owned, "completion agrees after reuse");
    expect(lead.trace == std::vector<std::string>{"unit", "ack"}, "final unit acks once");
    lead = {}; follower = {};
    tick(99, true);
    expect(lead.units == 0 && lead.cancels == 1, "immediate abort does not begin work");
    // Idle whole pieces: budget 0 runs the rest of the piece (SlotPrefill::step: run/extend, or the open task to its end),
    // after bounded units beside decoders and on a fresh piece; both ranks follow the same control.
    struct PieceTask {
        unsigned units = 0, pieces = 0;
        bool owned = true;
        bool step(unsigned, unsigned budget) {
            units = budget == 0U ? 6U : std::min(6U, units + budget);
            if (units == 6U) {
                owned = false;
                ++pieces;
            }
            return !owned;
        }
        void cancel() { owned = false; }
    } plead, pfollow;
    const auto ptick = [&](unsigned budget) {
        const auto a = sc::apply_unit(plead, 256, budget, false), b = sc::apply_unit(pfollow, 256, budget, false);
        expect(a == b && plead.units == pfollow.units && plead.pieces == pfollow.pieces, "whole pieces: both ranks agree");
        return a;
    };
    const auto two = [](double) { return 2U; };
    expect(ptick(sc::tick_budget(service, true, false, two)) != sc::UnitResult::complete && plead.units == 2, "decoders: bounded unit");
    expect(ptick(sc::tick_budget(service, false, false, two)) == sc::UnitResult::complete && plead.units == 6, "idle: the rest of the open piece");
    plead = {}; pfollow = {};  // the next piece
    expect(ptick(sc::tick_budget(service, false, false, two)) == sc::UnitResult::complete && plead.pieces == 1, "idle: a fresh piece at once");
    plead = {}; pfollow = {};
    expect(ptick(sc::tick_budget(service, false, true, two)) != sc::UnitResult::complete, "another prompt waits: bounded unit");
    expect(ptick(sc::tick_budget(service, false, false, two)) == sc::UnitResult::complete, "then the rest once it is alone");
    expect(lead.trace == std::vector<std::string>{"abort"}, "immediate abort has no ack");
    lead = {}; follower = {};  // whole-cache hit: begin completes without a unit command
    lead.owned = false; follower.owned = false;
    expect(lead.trace.empty() && follower.trace.empty(), "whole-cache hit exchanges no unit or ack on either rank");
}

void admission() {
    sc::Options o;
    o.mode = sc::Mode::defer;
    expect(sc::admit(o, 8192, 0, 0.0, 0.0) == sc::Admit::idle, "nobody decodes: admit");
    expect(sc::admit(o, 8192, 4, 0.0, 15.5) == sc::Admit::defer, "8K next to 4 streams with 15.5 s left: defer (4 x 14.0 s > 15.5 s)");
    expect(sc::admit(o, 512, 4, 0.0, 15.5) == sc::Admit::cost, "512 next to 4 streams with 15.5 s left: admit (4 x 1.04 s <= 15.5 s)");
    expect(sc::admit(o, 8192, 4, 30.0, 15.5) == sc::Admit::starved, "waited the starvation bound: admit");
    expect(sc::admit(o, 8192, 4, 1.0, 15.5, true) == sc::Admit::draining, "draining: admit");
    const std::vector<std::uint64_t> rem{10, 232, 0};
    expect(sc::wait_estimate(rem, 0.067) > 15.5 && sc::wait_estimate(rem, 0.067) < 15.6, "W = largest remaining x step time");
    expect(sc::tdev_seconds(o, 2048) > 3.6 && sc::tdev_seconds(o, 2048) < 3.65, "T_dev(2K) = 0.17 + 2048 x 1.69 ms");
}

// Walk a prompt with next_unit_end and check the units.
void units(std::uint32_t resume, std::uint32_t shared, std::uint32_t total, std::uint32_t block, std::uint32_t max_rows, std::uint32_t budget,
           std::uint32_t min_rows) {
    const auto ends = ds::plan_prefill_pieces(resume, shared, total, block, max_rows, min_rows);
    const std::string tag = "units(" + std::to_string(resume) + "," + std::to_string(total) + ", block " + std::to_string(block) + ", budget " +
                            std::to_string(budget) + ")";
    std::uint32_t at = resume, n = 0;
    std::vector<std::uint32_t> seen;
    while (at < total && n < 100000U) {
        const auto e = sc::next_unit_end(at, ends, budget, min_rows);
        expect(e > at && e <= total, tag + ": unit end inside the prompt");
        expect(e == total || e % 8U == 0U, tag + ": unit end on a multiple of 8");
        expect(e - at >= min_rows, tag + ": unit of at least min_rows rows");
        expect(budget == 0U || e - at <= budget || std::binary_search(ends.begin(), ends.end(), e), tag + ": unit within the budget or a default end");
        seen.push_back(e);
        at = e;
        ++n;
    }
    expect(at == total, tag + ": prompt covered");
    for (const auto d : ends) expect(std::binary_search(seen.begin(), seen.end(), d), tag + ": default end " + std::to_string(d) + " kept");
    if (budget == 0U) expect(seen == ends, tag + ": budget 0 = the default pieces");
}

}  // namespace

// STRIX_DS41_SERVE_SCHED x prefill path (P17.2: unset = interleave for the device prefill, off for rows; defer /
// interleave with rows rejected up front).
void modes() {
    const auto throws = [](const std::string& m, bool device) {
        try {
            (void)sc::mode_for(m, device);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    expect(sc::mode_for("", true) == sc::Mode::interleave, "unset + device: interleave");
    expect(sc::mode_for("", false) == sc::Mode::off, "unset + rows: off");
    for (const bool device : {true, false}) {
        expect(sc::mode_for("off", device) == sc::Mode::off && sc::mode_for("0", device) == sc::Mode::off, "off / 0: off on either path");
        expect(throws("bogus", device), "an unknown mode is rejected");
    }
    expect(sc::mode_for("interleave", true) == sc::Mode::interleave && sc::mode_for("1", true) == sc::Mode::interleave, "interleave / 1 + device");
    expect(sc::mode_for("defer", true) == sc::Mode::defer, "defer + device");
    expect(throws("interleave", false) && throws("1", false) && throws("defer", false), "interleave / defer + rows: rejected");
}

// The hello's serve identity (sched::serve_identity_text): different decode identities or DSpark setups never compare
// equal; the same configuration always does.
void identities() {
    const auto ep = ds::prefix_namespace_from_env(false).text, tp = ds::prefix_namespace_from_env(true).text;
    expect(ep != tp, "namespace: EP and expert-TP decode differ");
    expect(sc::serve_identity_text(ep, 0, 2, "controller", "full") != sc::serve_identity_text(tp, 0, 2, "controller", "full"), "identity: EP vs expert TP differ");
    expect(sc::serve_identity_text(ep, 100, 1, "controller", "full") != sc::serve_identity_text(ep, 100, 2, "controller", "full"), "identity: DSpark ranks 1 vs 2 differ");
    expect(sc::serve_identity_text(ep, 0, 2, "controller", "full") != sc::serve_identity_text(ep, 100, 2, "controller", "full"), "identity: DSpark on vs off differ");
    expect(sc::serve_identity_text(ep, 100, 2, "controller", "full") != sc::serve_identity_text(ep, 100, 2, "fixed:5", "full"), "identity: window policies differ");
    expect(sc::serve_identity_text(ep, 0, 1, "controller", "full") == sc::serve_identity_text(ep, 0, 2, "fixed:5", "auto:x"), "identity: without DSpark its options do not matter");
    expect(sc::serve_identity_text(tp, 100, 2, "controller", "full") == sc::serve_identity_text(tp, 100, 2, "controller", "full"), "identity: same config equal");
}

// --prefill-rows auto: the measured pair point (context 49152, rank 0 drafting a 7.42 GiB file: device free 34.4 GiB,
// MemAvailable 38.7, own weights 7.62) admits 16384; every reserve and the context bound it.
void auto_rows() {
    constexpr std::uint64_t GiB = std::uint64_t{1} << 30U;
    const auto gib = [&](double v) { return static_cast<std::uint64_t>(v * static_cast<double>(GiB)); };
    sc::AutoPrefillRoom r;
    r.wide_pieces = true;  // the serve's default envelope (context 49152, default window, no slabs)
    r.device_free = gib(34.4);
    r.mem_available = gib(38.7);
    r.own_weights = gib(7.62);
    r.drafter = sc::drafter_peak_bytes(7966294016ULL);
    expect(sc::auto_prefill_rows(49152, r) == 16384U, "auto rows: measured pair point -> 16384");
    expect(sc::prefill_piece_bytes(16384, 49152) > gib(6.37), "auto rows: scratch estimate covers the measured 6.37 GiB at 16384 rows");
    expect(sc::prefill_piece_bytes(32768, 49152) > gib(11.69), "auto rows: scratch estimate covers the measured 11.69 GiB at 32768 rows");
    expect(sc::drafter_peak_bytes(7966294016ULL) > gib(15.1), "auto rows: drafter peak covers the measured 15.1 GiB");
    auto t = r;
    t.mem_available = gib(31.0);  // the smaller of device free and MemAvailable limits
    expect(sc::auto_prefill_rows(49152, t) == 8192U, "auto rows: MemAvailable limits");
    t = r;
    t.drafter = 0;  // rank 1 without a drafter: room for 32768-row pieces
    expect(sc::auto_prefill_rows(49152, t) == 32768U, "auto rows: no drafter -> 32768");
    t = r;
    t.drafter = 0;  // the serve loads the drafter first: free memory already without its resident 10.4 GiB
    t.device_free = gib(24.0);
    t.mem_available = gib(27.8);
    expect(sc::auto_prefill_rows(49152, t) == 32768U, "auto rows: drafter loaded first -> 32768");
    t.wide_pieces = false;  // outside the measured envelope (large context, other window / slabs): at most 16384
    expect(sc::auto_prefill_rows(49152, t) == 16384U, "auto rows: no 32768 outside the measured envelope");
    {
        std::map<std::string, std::string> env;
        const auto look = [&](const char* k) { const auto it = env.find(k); return it == env.end() ? std::string() : it->second; };
        expect(sc::auto_wide_pieces(49152, look), "auto rows: the serve default is inside the envelope");
        expect(!sc::auto_wide_pieces(131072, look), "auto rows: a larger context is outside");
        for (const char* k : {"STRIX_DS41_PREFILL_WINDOW", "STRIX_DS41_PREFILL_SLABS", "STRIX_DS41_PREFILL_FP16_DENSE", "STRIX_DS41_PREFILL_BF16_ATTN", "STRIX_DS41_PREFILL_EXPERT_TP"}) {
            env = {{k, k == std::string("STRIX_DS41_PREFILL_WINDOW") ? "64" : "1"}};
            expect(!sc::auto_wide_pieces(49152, look), std::string("auto rows: outside with ") + k);
        }
        env = {{"STRIX_DS41_PREFILL_SLABS", "0"}};
        expect(sc::auto_wide_pieces(49152, look), "auto rows: SLABS=0 stays inside");
    }
    t = r;
    t.device_free = gib(20.0);
    expect(sc::auto_prefill_rows(49152, t) == 0U, "auto rows: nothing fits -> 0 (refuse, no unchecked 2048 floor)");
    t.device_free = gib(26.46 + 2.0);  // reserve 26.46 GiB + the 2048-row piece (1.91 GiB at 49152): only the floor fits
    expect(sc::auto_prefill_rows(49152, t) == 2048U, "auto rows: only the 2048 floor fits -> 2048");
    t.mem_available = 0;  // unknown MemAvailable: device free alone
    t.device_free = gib(34.4);
    expect(sc::auto_prefill_rows(49152, t) == 16384U, "auto rows: MemAvailable unknown");
    t = r;
    t.own_weights = gib(12.5);  // more weights to load (e.g. no trellis sharing) shrink the piece
    expect(sc::auto_prefill_rows(49152, t) == 4096U, "auto rows: own weights reserved");
    expect(sc::auto_prefill_rows(12000, r) == 8192U, "auto rows: at most the context");
    expect(sc::auto_prefill_rows(1000, r) == 1000U, "auto rows: short context -> the context (fallback)");
    expect(sc::auto_prefill_rows(20, r) == 32U, "auto rows: at least 32");
    for (const std::uint32_t ctx : {32U, 1000U, 4096U, 49152U, 1048576U})
        for (const double free : {0.0, 10.0, 40.0, 120.0}) {
            t = r;
            t.device_free = gib(free);
            const auto rows = sc::auto_prefill_rows(ctx, t);
            expect(rows == 0U || (rows >= 32U && rows % 8U == 0U && (rows <= ctx || rows == 32U)), "auto rows: valid --prefill-rows value or no fit");
            const std::uint64_t avail = std::min(t.device_free, t.mem_available), reserve = t.own_weights + t.drafter + t.margin;
            expect(rows == 0U || sc::prefill_piece_bytes(rows, ctx) <= avail - std::min(avail, reserve), "auto rows: a returned size always fits (no unchecked floor)");
        }
}

int main() {
    service_options();
    auto_rows();
    short_first();
    control_trace();
    admission();
    modes();
    identities();
    for (const std::uint32_t total : {9U, 16U, 17U, 100U, 128U, 512U, 513U, 2048U, 2053U, 8192U, 8199U})
        for (const std::uint32_t budget : {0U, 16U, 24U, 256U, 4096U}) {
            units(0, 0, total, 4096, 4096, budget, 9);
            if (total > 512U) units(256, 384, total, 256, 4096, budget, 9);
            if (total > 1024U) units(0, 0, total, 1024, 1024, budget, 9);
        }
    if (failures == 0) std::printf("deepseek41 serve sched tests pass\n");
    return failures == 0 ? 0 : 1;
}
