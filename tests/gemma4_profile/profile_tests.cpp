#include "gemma4/engine/profile.hpp"

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace pr = strix::gemma4::profile;
namespace {
int failures = 0;
void check(bool condition, const std::string& what) {
    if (!condition) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}
template <class F> void refuses(F&& f, const std::string& needle, const std::string& what) {
    try { f(); }
    catch (const std::invalid_argument& e) {
        check(std::string(e.what()).find(needle) != std::string::npos, what + " (message: " + e.what() + ")");
        return;
    }
    check(false, what + " did not refuse");
}
pr::Lookup lookup(const std::map<std::string, std::string>& env) {
    return [env](const char* name) -> const char* {
        const auto it = env.find(name);
        return it == env.end() ? nullptr : it->second.c_str();
    };
}
const pr::Lookup empty = lookup({});
pr::Inputs in(pr::Pairs cli = {}, pr::Pairs assets = {}, const pr::Lookup& env = empty) { return {std::move(cli), std::move(assets), env}; }
pr::Status status(const pr::Resolution& r, const char* lever) {
    const auto* e = r.find(lever);
    if (!e) throw std::runtime_error(std::string("lever not resolved: ") + lever);
    return e->status;
}
std::string value(const pr::Resolution& r, const char* lever) { return r.find(lever)->value; }

constexpr unsigned all = 1 | 2 | 4;
// Synthetic table: a ready env lever, a pending exact lever, a pending lever FAST requires,
// a CLI asset lever QUALITY requires, and a discovered asset lever with an off switch.
constexpr pr::Lever synthetic[] = {
    {.name = "route", .env = "TEST_ROUTE", .value = {"a", "b", "a"}, .builtin = "a", .required = {false, false, false},
     .ready = {true, true, true}, .kind = pr::Kind::bit_changing, .apps = all, .source = "ready"},
    {.name = "speed", .env = "TEST_SPEED", .value = {"1", "1", "1"}, .builtin = "0", .required = {false, false, false},
     .ready = {false, false, false}, .kind = pr::Kind::exact, .apps = all, .source = "owner-branch"},
    {.name = "named", .env = "TEST_NAMED", .value = {"", "x8", ""}, .builtin = "x0", .required = {false, true, false},
     .ready = {true, false, true}, .kind = pr::Kind::bit_changing, .apps = 1, .source = "named-branch"},
    {.name = "pack", .option = "--pack", .value = {"", "", "pack"}, .builtin = "off", .required = {false, false, true},
     .ready = {true, true, true}, .kind = pr::Kind::pack, .apps = all, .asset = true, .source = "pack-commit"},
    {.name = "draft", .env = "TEST_DRAFT", .option = "--draft-mtp", .value = {"on", "on", "on"}, .builtin = "off",
     .required = {false, false, false}, .ready = {true, false, true}, .kind = pr::Kind::exact, .apps = 1, .asset = true,
     .off = "0", .source = "drafter"},
};

void synthetic_table() {
    // DEFAULT: ready lever equal to builtin stays builtin, pending lever recorded with the builtin value.
    auto r = pr::resolve(pr::Name::standard, pr::App::engine, in(), synthetic);
    check(status(r, "route") == pr::Status::builtin, "default route builtin");
    check(status(r, "speed") == pr::Status::pending && value(r, "speed") == "0", "pending exact lever keeps builtin");
    check(!r.find("named") && !r.find("pack"), "levers without a default value are not listed");
    check(r.json().find("\"pending\":[{\"lever\":\"speed\",\"target\":\"1\",\"runs\":\"0\"}]") != std::string::npos,
          "pending list in receipt with the declared target");
    check(!r.active("route") && !r.active("speed"), "nothing active in default");

    // FAST: a pending row (even a required one) never refuses: it is not applied, the builtin runs, receipt + stderr say so.
    r = pr::resolve(pr::Name::fast, pr::App::engine, in(), synthetic);
    check(status(r, "named") == pr::Status::pending && value(r, "named") == "x0" && r.find("named")->target == "x8",
          "pending required row runs the builtin");
    {
        unsigned named = 0, pending = 0;
        for (const auto& line : r.notices()) named += line == "profile fast: named pending x8 (x0 runs)";
        for (const auto& e : r.entries) pending += e.status == pr::Status::pending;
        check(named == 1 && r.notices().size() == pending, "one notice per pending row");
    }
    r = pr::resolve(pr::Name::fast, pr::App::prefill_bench, in(), synthetic);
    check(status(r, "route") == pr::Status::set && value(r, "route") == "b" && r.active("route"), "fast sets ready lever");
    // An explicit request for a value whose code is not in the build is refused (it would otherwise be ignored).
    refuses([] { (void)pr::resolve(pr::Name::fast, pr::App::engine, in({}, {}, lookup({{"TEST_NAMED", "x8"}})), synthetic); },
            "TEST_NAMED=x8 is not in this build", "explicit request for missing code refused");

    // QUALITY: CLI asset required.
    refuses([] { (void)pr::resolve(pr::Name::quality, pr::App::engine, in(), synthetic); }, "--pack",
            "quality refuses without the pack option");
    refuses([] { (void)pr::resolve(pr::Name::quality, pr::App::decode_bench, in(), synthetic); }, "not available in this app",
            "quality refuses in an app without the pack option");
    r = pr::resolve(pr::Name::quality, pr::App::engine, in({{"pack", "/m4"}}), synthetic);
    check(status(r, "pack") == pr::Status::explicit_cli && value(r, "pack") == "/m4", "explicit CLI asset");

    // Explicit env always wins over the profile value; explicit options outside a profile are recorded.
    r = pr::resolve(pr::Name::fast, pr::App::prefill_bench, in({}, {}, lookup({{"TEST_ROUTE", "a"}})), synthetic);
    check(status(r, "route") == pr::Status::explicit_env && value(r, "route") == "a" && !r.active("route"),
          "explicit env beats profile");
    r = pr::resolve(pr::Name::standard, pr::App::engine, in({{"pack", "/m4"}}), synthetic);
    check(status(r, "pack") == pr::Status::explicit_cli, "explicit option recorded under default");
}

void drafter_convention() {
    const pr::Pairs found{{"draft", "/models/e4b/drafter-mtp"}};
    // Discovered directory: on in every profile whose route is ready.
    for (const auto p : {pr::Name::standard, pr::Name::quality}) {
        const auto r = pr::resolve(p, pr::App::engine, in({}, found), std::span(synthetic).subspan(4));
        check(status(r, "draft") == pr::Status::set && value(r, "draft") == "/models/e4b/drafter-mtp", "discovered drafter on");
        check(r.drafter() == "on (/models/e4b/drafter-mtp)", "receipt drafter on");
    }
    const auto table = std::span(synthetic).subspan(4);
    // Absent directory: exact decode without it, never an error.
    auto r = pr::resolve(pr::Name::standard, pr::App::engine, in(), table);
    check(status(r, "draft") == pr::Status::absent && r.drafter() == "absent", "missing drafter is absent");
    check(r.json().find("\"drafter\":\"absent\"") != std::string::npos, "receipt says drafter: absent");
    // Route not ready (FAST column): pending, not loaded, unless given explicitly.
    r = pr::resolve(pr::Name::fast, pr::App::engine, in({}, found), table);
    check(status(r, "draft") == pr::Status::pending, "drafter pending on an unready route");
    r = pr::resolve(pr::Name::fast, pr::App::engine, in({{"draft", "/d"}}, found), table);
    check(status(r, "draft") == pr::Status::explicit_cli && value(r, "draft") == "/d", "--draft-mtp overrides");
    // --draft-mtp DIR overrides discovery.
    r = pr::resolve(pr::Name::standard, pr::App::engine, in({{"draft", "/mine"}}, found), table);
    check(value(r, "draft") == "/mine", "explicit drafter dir wins over discovery");
    // --draft off and the env off switch win over the profile and discovery (also over a path given before it).
    for (const auto& cli : {pr::Pairs{{"draft", "off"}}, pr::Pairs{{"draft", "/mine"}, {"draft", "off"}}}) {
        r = pr::resolve(pr::Name::standard, pr::App::engine, in(cli, found), table);
        check(status(r, "draft") == pr::Status::off && r.drafter() == "off (switch)", "--draft off wins");
        check(r.json().find("\"drafter\":\"off (switch)\"") != std::string::npos, "receipt says drafter: off (switch)");
    }
    r = pr::resolve(pr::Name::quality, pr::App::engine, in({}, found, lookup({{"TEST_DRAFT", "0"}})), table);
    check(status(r, "draft") == pr::Status::off && r.drafter() == "off (switch)", "env off switch wins");
    // A non-off env asks for the drafter; still never an error without the directory.
    r = pr::resolve(pr::Name::fast, pr::App::engine, in({}, {}, lookup({{"TEST_DRAFT", "1"}})), table);
    check(status(r, "draft") == pr::Status::absent, "env on without weights is absent");
}

void product_table() {
    // Rule: exact levers in all three profiles, bit-changing values only in FAST, pack values only in QUALITY.
    for (const auto& lever : pr::levers()) {
        check(!lever.name.empty() && !lever.source.empty(), "lever has name and source");
        check(!lever.env.empty() || !lever.option.empty(), std::string(lever.name) + " has a switch");
        if (lever.kind == pr::Kind::bit_changing)
            for (const unsigned p : {0U, 2U})
                check(lever.value[p].empty() || (lever.value[p] != lever.value[1] &&
                      (lever.value[0].empty() || lever.value[2].empty() || lever.value[0] == lever.value[2])),
                      std::string(lever.name) + " bit-changing value outside FAST");
        if (lever.kind == pr::Kind::pack)
            for (const unsigned p : {0U, 1U})
                check(lever.value[p].empty() || lever.value[p] == lever.builtin, std::string(lever.name) + " pack value outside QUALITY");
        if (lever.kind == pr::Kind::exact)
            check(!lever.value[0].empty() && lever.value[0] == lever.value[1] && lever.value[1] == lever.value[2],
                  std::string(lever.name) + " exact lever in all three profiles");
    }
    for (const auto app : {pr::App::engine, pr::App::decode_bench}) {
        const auto r = pr::resolve(pr::Name::standard, app, in());
        check(r.json().find("\"profile\":\"default\"") != std::string::npos, "default receipt names the profile");
    }
    auto r = pr::resolve(pr::Name::standard, pr::App::engine, in());
    check(status(r, "decode_identity") == pr::Status::builtin && value(r, "decode_identity") == "g4-v1", "default g4-v1");
    check(status(r, "draft") == pr::Status::absent && r.drafter() == "absent", "default drafter absent without weights");
    check(status(r, "draft_mode") == pr::Status::builtin, "drafter auto admission");
    check(status(r, "prefill_route") == pr::Status::builtin && value(r, "prefill_route") == "exact", "default exact prefill");
    r = pr::resolve(pr::Name::standard, pr::App::engine, in({}, {{"draft", "/m/drafter-mtp"}}));
    check(status(r, "draft") == pr::Status::set && r.drafter() == "on (/m/drafter-mtp)", "default discovers the drafter");
    r = pr::resolve(pr::Name::standard, pr::App::engine, in({}, {{"draft", "/m/drafter-mtp"}}, lookup({{"STRIX_G4_DRAFT", "0"}})));
    check(r.drafter() == "off (switch)", "STRIX_G4_DRAFT=0 wins over discovery");
    r = pr::resolve(pr::Name::standard, pr::App::engine, in({{"draft", "off"}}, {{"draft", "/m/drafter-mtp"}}));
    check(r.drafter() == "off (switch)", "--draft off wins over discovery");
    r = pr::resolve(pr::Name::standard, pr::App::engine, in({{"draft_mode", "off"}}));
    check(status(r, "draft_mode") == pr::Status::off, "--dflash off");
    r = pr::resolve(pr::Name::standard, pr::App::engine, in({}, {}, lookup({{"STRIX_G4_DFLASH", "0"}})));
    check(status(r, "draft_mode") == pr::Status::off, "STRIX_G4_DFLASH=0");
    // C1 MTP confidence window: exact, auto (per drafter) in all three profiles; STRIX_G4_MTP_CONF=0 is the switch.
    for (const auto p : {pr::Name::standard, pr::Name::fast, pr::Name::quality}) {
        r = pr::resolve(p, pr::App::engine, in({{"exl3_route", "/m4"}}));
        check(status(r, "draft_window") == pr::Status::builtin && value(r, "draft_window") == "auto", "confidence window auto");
    }
    r = pr::resolve(pr::Name::standard, pr::App::engine, in({}, {}, lookup({{"STRIX_G4_MTP_CONF", "0"}})));
    check(status(r, "draft_window") == pr::Status::off, "STRIX_G4_MTP_CONF=0");
    r = pr::resolve(pr::Name::standard, pr::App::engine, in({}, {}, lookup({{"STRIX_G4_MTP_CONF", "0.45"}})));
    check(status(r, "draft_window") == pr::Status::explicit_env && value(r, "draft_window") == "0.45", "STRIX_G4_MTP_CONF=0.45");
    // Cost-aware long-context window: exact, auto in all three profiles; STRIX_G4_MTP_CONF_LONG=0 is its switch.
    for (const auto p : {pr::Name::standard, pr::Name::fast, pr::Name::quality}) {
        r = pr::resolve(p, pr::App::engine, in({{"exl3_route", "/m4"}}));
        check(status(r, "draft_window_long") == pr::Status::builtin && value(r, "draft_window_long") == "auto", "long window auto");
    }
    r = pr::resolve(pr::Name::standard, pr::App::engine, in({}, {}, lookup({{"STRIX_G4_MTP_CONF_LONG", "0"}})));
    check(status(r, "draft_window_long") == pr::Status::off, "STRIX_G4_MTP_CONF_LONG=0");
    // The prefill bench only has WMMA linears: DEFAULT and QUALITY refuse there instead of running them.
    refuses([] { (void)pr::resolve(pr::Name::standard, pr::App::prefill_bench, in()); }, "prefill-exact",
            "default refuses the WMMA-only prefill bench");
    refuses([] { (void)pr::resolve(pr::Name::quality, pr::App::prefill_bench, in({{"exl3_route", "/m4"}})); },
            "prefill_linears", "quality refuses the WMMA-only prefill bench");
    refuses([] { pr::Inputs a = in(); a.allow_pending = true; (void)pr::resolve(pr::Name::standard, pr::App::prefill_bench, a); },
            "strix-gemma4-decode", "allow-pending cannot turn the WMMA fallback into DEFAULT's exact route");

    // FAST never refuses for a pending row: g4-a8-v1 (ready) runs, the WMMA prefill row is pending g4-wmma-pf-v2,
    // so exact prefill runs; the receipt lists the declared target and one stderr line per pending row says so.
    check(std::string(pr::levers()[0].name) == "decode_identity", "table starts with the decode identity");
    for (const auto app : {pr::App::engine, pr::App::decode_bench}) {
        r = pr::resolve(pr::Name::fast, app, in({}, {{"draft", "/m/drafter-mtp"}}));
        check(value(r, "decode_identity") == "g4-a8-v1" && status(r, "decode_identity") == pr::Status::set, "fast runs g4-a8-v1");
        check(r.qualification() == "qualified", "fast without --allow-pending is qualified");
    }
    r = pr::resolve(pr::Name::fast, pr::App::engine, in());
    // The fast prompt prefill follows its receipt (profile.cpp): named -> the Q4_K_M-gated v1 route; empty -> pending,
    // prefill-exact runs and the receipt names the target. Qualified either way.
    const bool fast_prefill_ready = [] {
        for (const auto& lever : pr::levers()) if (lever.name == "prefill_route") return lever.ready[1];
        return false;
    }();
    if (fast_prefill_ready)
        check(status(r, "prefill_route") == pr::Status::set && value(r, "prefill_route") == "fast" && r.active("prefill_route") &&
              r.qualification() == "qualified", "fast runs the qualified v1 prefill");
    else
        check(status(r, "prefill_route") == pr::Status::pending && value(r, "prefill_route") == "exact" && !r.active("prefill_route") &&
              r.qualification() == "qualified", "fast prefill pending: prefill-exact runs");
    // The fast value runs v1 only on the Q4_K_M pack its receipt covers; other packs keep prefill-exact.
    check(pr::prefill_route("fast", 15, false) == "v1", "fast prefill: a MOSTLY_Q4_K_M GGUF runs v1");
    check(pr::prefill_route("fast", 14, false) == "exact", "fast prefill: a Q4_K_S GGUF runs prefill-exact");
    check(pr::prefill_route("fast", 7, false) == "exact" && pr::prefill_route("fast", 1, false) == "exact",
          "fast prefill: Q8_0 / F16 GGUF run prefill-exact");
    check(pr::prefill_route("fast", -1, false) == "exact", "fast prefill: a BF16 checkpoint runs prefill-exact");
    check(pr::prefill_route("fast", 15, true) == "exact" && pr::prefill_route("v1", -1, true) == "exact",
          "EXL3 packs keep their own prefill route");
    check(pr::prefill_route("v1", -1, false) == "v1" && pr::prefill_route("v2", 14, false) == "v2",
          "explicit v1/v2 select the route on any non-EXL3 pack");
    check(pr::prefill_route("", 15, false) == "exact" && pr::prefill_route("exact", 15, false) == "exact", "exact prefill");
    refuses([] { (void)pr::prefill_route("wmma", 15, false); }, "STRIX_G4_PREFILL_IDENTITY", "unknown prefill route refused");
    {
        const auto lines = r.notices();
        unsigned pending = 0;
        for (const auto& e : r.entries) pending += e.status == pr::Status::pending;
        check(lines.size() == pending, "one stderr line per pending row, no more");
    }
    // FAST decode attention at depth: pending until its receipt (exact runs), runnable with --allow-pending,
    // never in DEFAULT/QUALITY, STRIX_G4_DECODE_ATTN=exact switches it off.
    for (const auto app : {pr::App::engine, pr::App::decode_bench}) {
        r = pr::resolve(pr::Name::fast, app, in());
        check(status(r, "decode_attention") == pr::Status::pending && value(r, "decode_attention") == "exact" &&
              r.find("decode_attention")->target == "g4-dattn-f2" && !r.active("decode_attention"),
              "fast decode attention pending, exact runs");
        pr::Inputs pending_ok = in(); pending_ok.allow_pending = true;
        r = pr::resolve(pr::Name::fast, app, pending_ok);
        check(status(r, "decode_attention") == pr::Status::unqualified && value(r, "decode_attention") == "g4-dattn-f2",
              "allow-pending runs g4-dattn-f2 unqualified");
        r = pr::resolve(pr::Name::fast, app, in({}, {}, lookup({{"STRIX_G4_DECODE_ATTN", "exact"}})));
        check(value(r, "decode_attention") == "exact" && !r.active("decode_attention"), "STRIX_G4_DECODE_ATTN=exact is OFF");
        r = pr::resolve(pr::Name::standard, app, in());
        check(status(r, "decode_attention") == pr::Status::builtin && value(r, "decode_attention") == "exact",
              "default keeps the exact decode attention");
    }
    // The prefill bench cannot run exact prefill: it refuses every profile without --allow-pending.
    refuses([] { (void)pr::resolve(pr::Name::fast, pr::App::prefill_bench, in()); }, "--allow-pending",
            "fast prefill bench refuses and names --allow-pending");
    // --allow-pending (benchmarks): the prefill bench runs WMMA v1 and stamps the receipt.
    pr::Inputs allow = in(); allow.allow_pending = true;
    r = pr::resolve(pr::Name::fast, pr::App::prefill_bench, allow);
    check(status(r, "prefill_attention") == pr::Status::unqualified && value(r, "prefill_attention") == "wmma" &&
          status(r, "prefill_linears") == pr::Status::unqualified && value(r, "prefill_linears") == "wmma",
          "allow-pending runs WMMA prefill");
    check(r.qualification() == "unqualified: prefill_linears,prefill_attention", "qualification stamp");
    check(r.json().find("\"qualification\":\"unqualified: prefill_linears,prefill_attention\"") != std::string::npos,
          "receipt stamped unqualified");
    // --allow-pending changes nothing for a qualified row.
    r = pr::resolve(pr::Name::fast, pr::App::engine, allow);
    check(value(r, "prefill_route") == "fast" && status(r, "prefill_route") ==
          (fast_prefill_ready ? pr::Status::set : pr::Status::unqualified), "allow-pending runs the fast v1 route");
    // OFF: STRIX_G4_PREFILL_IDENTITY=exact keeps prefill-exact under fast.
    r = pr::resolve(pr::Name::fast, pr::App::engine, in({}, {}, lookup({{"STRIX_G4_PREFILL_IDENTITY", "exact"}})));
    check(status(r, "prefill_route") == pr::Status::explicit_env && value(r, "prefill_route") == "exact", "explicit exact prefill");
    refuses([] { (void)pr::resolve(pr::Name::fast, pr::App::prefill_bench, in({}, {}, lookup({{"STRIX_G4_PREFILL", "exact"}}))); },
            "not in this build", "explicit exact route in the WMMA-only bench refused");
    // DEFAULT engine: prefill-exact, qualified, nothing pending for prefill.
    r = pr::resolve(pr::Name::standard, pr::App::engine, in());
    check(!r.active("prefill_route") && value(r, "prefill_route") == "exact" && r.qualification() == "qualified" &&
          status(r, "prefill_route") == pr::Status::builtin, "default engine prefill-exact, qualified");

    // QUALITY: needs the EXL3 arm (never discovered); EXL3 identity auto-selected.
    refuses([] { (void)pr::resolve(pr::Name::quality, pr::App::engine, in()); }, "--exl3-arm", "quality needs arm");
    refuses([] { (void)pr::resolve(pr::Name::quality, pr::App::decode_bench, in()); }, "--exl3-arm", "quality refused in bench");
    r = pr::resolve(pr::Name::quality, pr::App::engine, in({{"exl3_route", "/m4"}}));
    check(status(r, "exl3_route") == pr::Status::explicit_cli && r.find("exl3_identity"), "quality arm + EXL3 identity");
    check(!r.find("decode_identity"), "quality leaves the Q4 decode identity alone");
    r = pr::resolve(pr::Name::quality, pr::App::engine, in({{"exl3_route", "/m4"}}, {}, lookup({{"STRIX_G4_EXL3_IDENTITY", "v1"}})));
    check(status(r, "exl3_identity") == pr::Status::explicit_env && value(r, "exl3_identity") == "v1", "EXL3 v1 explicit");
    r = pr::resolve(pr::Name::quality, pr::App::engine, in({{"exl3_route", "/m4"}}));
    check(status(r, "exl3_identity") == pr::Status::builtin && value(r, "exl3_identity") == "auto" &&
          r.qualification() == "qualified", "QUALITY auto-selects the EXL3 identity per pack and session");
    // The EXL3 panel prefill is pending in QUALITY: exact runs, the receipt names the target identity.
    check(status(r, "exl3_prefill") == pr::Status::pending && value(r, "exl3_prefill") == "exact" &&
          r.find("exl3_prefill")->target == "g4-exl3-wmma-pf-v1" && !r.active("exl3_prefill"), "QUALITY EXL3 prefill pending");
    check(!pr::resolve(pr::Name::standard, pr::App::engine, in()).find("exl3_prefill"), "DEFAULT leaves the EXL3 prefill alone");
    r = pr::resolve(pr::Name::quality, pr::App::engine, in({{"exl3_route", "/m4"}}, {}, lookup({{"STRIX_G4_EXL3_PREFILL", "wmma"}})));
    check(status(r, "exl3_prefill") == pr::Status::explicit_env && value(r, "exl3_prefill") == "wmma", "EXL3 panel prefill explicit");
}

// Invariant: without --allow-pending no pending or unqualified implementation ever executes. Every profile x app
// either refuses or resolves to rows whose effective value is qualified (ready), or a pending row whose fallback is
// a qualified builtin.
void no_pending_executes() {
    const pr::Name profiles[] = {pr::Name::standard, pr::Name::fast, pr::Name::quality};
    const pr::App apps[] = {pr::App::engine, pr::App::decode_bench, pr::App::prefill_bench};
    unsigned resolved = 0;
    for (const auto p : profiles)
        for (const auto app : apps)
            for (const bool assets : {false, true}) {
                auto inputs = in(assets ? pr::Pairs{{"exl3_route", "/m4"}} : pr::Pairs{},
                                 assets ? pr::Pairs{{"draft", "/m/drafter-mtp"}} : pr::Pairs{});
                pr::Resolution r;
                try { r = pr::resolve(p, app, inputs); } catch (const std::invalid_argument&) { continue; }
                ++resolved;
                const auto column = unsigned(p);
                check(r.qualification() == "qualified", "qualified without --allow-pending");
                for (const auto& e : r.entries) {
                    const auto where = std::string(pr::name(p)) + "/" + pr::app_name(app) + "/" + std::string(e.lever->name);
                    check(e.status != pr::Status::unqualified, where + " unqualified without --allow-pending");
                    if (e.status == pr::Status::set || e.status == pr::Status::builtin)
                        check(e.lever->ready[column], where + " runs a column that is not ready");
                    if (e.status == pr::Status::pending)
                        check(!e.lever->builtin_unqualified, where + " falls back to an unqualified builtin");
                }
            }
    check(resolved > 0, "some profile resolves");
    // Synthetic: a non-required pending row whose builtin is unqualified refuses instead of falling back.
    constexpr pr::Lever fallback[] = {
        {.name = "bench_route", .env = "TEST_BENCH_ROUTE", .value = {"exact", "fast", "exact"}, .builtin = "fast",
         .required = {false, false, false}, .ready = {false, false, false}, .runnable = {false, true, false},
         .apps = 1, .builtin_unqualified = true, .source = "pending: exact route"},
    };
    refuses([&] { (void)pr::resolve(pr::Name::standard, pr::App::engine, in(), fallback); }, "pending: exact route",
            "unqualified builtin is never a silent fallback");
}

void parsing_and_apply() {
    check(pr::parse("default") == pr::Name::standard && pr::parse("fast") == pr::Name::fast &&
          pr::parse("quality") == pr::Name::quality, "parse names");
    refuses([] { (void)pr::parse("turbo"); }, "default, fast or quality", "unknown profile");
    char a0[] = "app", a1[] = "--profile", a2[] = "fast", a3[] = "--steps";
    char* with_value[] = {a0, a1, a2};
    char* bare[] = {a0, a1, a3};
    std::optional<pr::Name> out;
    int i = 1;
    check(pr::take_option(3, with_value, i, out) && out == pr::Name::fast && i == 2, "--profile NAME consumed");
    i = 1; out.reset();
    check(!pr::take_option(3, bare, i, out) && !out && i == 1, "bare --profile left to the app");
    i = 1;
    check(!pr::take_option(2, with_value, i, out), "--profile at the end is bare");

    // apply exports only profile-set plain switches, never over an existing value, never asset/off levers.
    constexpr pr::Lever exported[] = {
        {.name = "one", .env = "STRIX_G4_PROFILE_TEST_ONE", .value = {"x", "x", "x"}, .builtin = "y", .ready = {true, true, true}, .apps = 1, .source = "t"},
        {.name = "two", .env = "STRIX_G4_PROFILE_TEST_TWO", .value = {"x", "x", "x"}, .builtin = "y", .ready = {true, true, true}, .apps = 1, .source = "t"},
        {.name = "three", .env = "STRIX_G4_PROFILE_TEST_THREE", .value = {"x", "x", "x"}, .builtin = "y", .ready = {true, true, true},
         .apps = 1, .asset = true, .off = "0", .source = "t"},
        {.name = "four", .env = "STRIX_G4_PROFILE_TEST_FOUR", .value = {"x", "x", "x"}, .builtin = "y", .required = {true, true, true},
         .ready = {false, false, false}, .runnable = {true, true, true}, .apps = 1, .source = "pending: gate"},
    };
#ifdef _WIN32
    _putenv_s("STRIX_G4_PROFILE_TEST_TWO", "mine");
#else
    setenv("STRIX_G4_PROFILE_TEST_TWO", "mine", 1);
#endif
    {
        const auto plain = pr::resolve(pr::Name::standard, pr::App::engine, {{}, {{"three", "/p"}}, {}}, exported);
        check(plain.find("four")->status == pr::Status::pending && plain.find("four")->value == "y",
              "required pending runnable row is not applied without --allow-pending");
    }
    const auto r = pr::resolve(pr::Name::standard, pr::App::engine, {{}, {{"three", "/p"}}, {}, true}, exported);
    pr::apply(r);
    const auto* one = std::getenv("STRIX_G4_PROFILE_TEST_ONE");
    const auto* two = std::getenv("STRIX_G4_PROFILE_TEST_TWO");
    check(one && std::string(one) == "x", "apply exports the profile value");
    check(two && std::string(two) == "mine" && r.find("two")->status == pr::Status::explicit_env, "apply keeps explicit env");
    check(!std::getenv("STRIX_G4_PROFILE_TEST_THREE") && r.find("three")->status == pr::Status::set, "asset levers are not exported");
    check(r.json().find("\"status\":\"explicit-env\"") != std::string::npos, "receipt marks explicit env");
    const auto* four = std::getenv("STRIX_G4_PROFILE_TEST_FOUR");
    check(four && std::string(four) == "x" && r.qualification() == "unqualified: four", "unqualified row exported and stamped");
    check(r.json().find("\"unqualified\":[\"four\"]") != std::string::npos, "receipt lists unqualified rows");
    // The env switch enables it as well.
    const auto e = pr::resolve(pr::Name::standard, pr::App::engine,
                               {{}, {{"three", "/p"}}, lookup({{"STRIX_G4_PROFILE_ALLOW_PENDING", "1"}})}, exported);
    check(e.allow_pending && e.find("four")->status == pr::Status::unqualified, "STRIX_G4_PROFILE_ALLOW_PENDING=1");
}
} // namespace

int main() {
    try {
        synthetic_table();
        drafter_convention();
        product_table();
        no_pending_executes();
        parsing_and_apply();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (failures) { std::cerr << failures << " profile check(s) failed\n"; return 1; }
    std::cout << "gemma4 profile resolver PASS\n";
    return 0;
}
