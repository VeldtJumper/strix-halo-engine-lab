#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
// MiniMax M3.1 defaults-on resolver, CPU only: the plain invocation resolves to exactly the measured production
// receipt (argv + env, tests/fixtures/minimax_production_receipt.txt), every off switch restores its item's former
// value, STRIX_MM_BASELINE=1 restores all of them, and apply() exports what --print-effective-config prints.
#include "minimax/engine/mm_defaults.hpp"
#include "strix/minimax/session_startup.hpp"

#include <cstdlib>
#include <array>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
int setenv(const char* key, const char* value, int overwrite) { if (!overwrite && std::getenv(key)) return 0; return ::_putenv_s(key, value); }
int unsetenv(const char* key) { return ::_putenv_s(key, ""); }
#endif
namespace {

namespace md = strix::minimax::defaults;

int g_failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::cerr << "FAIL: " << what << "\n";
    }
}

using EnvMap = std::map<std::string, std::string>;
md::Lookup lookup(const EnvMap& m) {
    return [&m](const char* name) -> const char* {
        const auto it = m.find(name);
        return it == m.end() ? nullptr : it->second.c_str();
    };
}

std::vector<std::string> words(const std::string& s) {
    std::istringstream in(s);
    std::vector<std::string> out;
    for (std::string w; in >> w;) out.push_back(w);
    return out;
}

struct Receipt {
    std::string app;
    std::vector<std::string> argv, plain;
    EnvMap env;
};

std::vector<Receipt> load(const char* path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<Receipt> out;
    Receipt cur;
    bool open = false;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        const auto sp = line.find(' ');
        const std::string key = line.substr(0, sp), rest = sp == std::string::npos ? "" : line.substr(sp + 1);
        if (key == "receipt") {
            cur = Receipt{};
            cur.app = rest;
            open = true;
        } else if (key == "argv") cur.argv = words(rest);
        else if (key == "plain") cur.plain = words(rest);
        else if (key == "env") {
            const auto eq = rest.find('=');
            if (eq == std::string::npos) throw std::runtime_error("bad env line: " + line);
            cur.env[rest.substr(0, eq)] = rest.substr(eq + 1);
        } else if (key == "end" && open) {
            out.push_back(cur);
            open = false;
        } else throw std::runtime_error("bad fixture line: " + line);
    }
    return out;
}

std::set<std::string> lines(const std::string& text) {
    std::set<std::string> s;
    std::istringstream in(text);
    for (std::string l; std::getline(in, l);) s.insert(l);
    return s;
}

// The value each item replaced (measurement evidence), written out independently of the table.
const std::map<std::string, std::string> kFormer = {
    {"STRIX_MM_SERVE_DSPARK", "0"}, {"STRIX_MM_LOOKUP", "0"},
    {"STRIX_MM_SERVE_NEST", "0"}, {"STRIX_MM_SERVE_SCHED", "off"}, {"STRIX_MM_SERVE_SERVICE_MS", "0"}, {"STRIX_MM_SERVE_SHORT_FIRST", "0"},
    {"STRIX_MMX_MOE_WIDE", "0"},           {"STRIX_MM_INDEX_CONC", "0"},       {"STRIX_MMX_CRC", "1"},
    {"STRIX_MM_XB_WAIT", "0"},             {"STRIX_MM_XB_INPLACE", "0"},       {"STRIX_MM_XB_PUBLISH", "0"},
    {"STRIX_MM_XB_POSTILP", "0"},          {"STRIX_MM_XB_WARM", "0"},          {"STRIX_MM_DSPARK_ATTN", "0"},
    {"STRIX_MM_DRAFT_LIN", "lib"},         {"STRIX_MMX_PREFILL_CORE", "lib"},  {"STRIX_MM_PF_MSA_BATCH", "0"},
    {"STRIX_MMX_LINEAR_V11", "0"},
    {"STRIX_MM_MARKOV", "full"},           {"STRIX_MM_MARKOV_C", "1024"},      {"STRIX_MM_MARKOV_H", "2048"},
    {"STRIX_MM_MARKOV_TOPC", "v1"},
    {"STRIX_MM_PF_ATTEND4", "0"},          {"STRIX_MM_PF_ROUTER_BATCH", "0"},  {"STRIX_MMX_PREFILL_MTFIT", "0"},
    {"STRIX_MM_PF_MSA_REWORK", "0"},       {"STRIX_MMX_PREFILL_PIPE", "0"},    {"STRIX_MM_PF_KV_TILE", "off"},
    {"STRIX_MM_PF_ASYNC_START", "0"}, {"STRIX_MM_PF_LATE_SRC_FENCE", "0"},
    {"STRIX_MM_PF_TX_UNCACHED", "0"},      {"STRIX_DIST_CRC_PARALLEL", "0"},   {"STRIX_DIST_HOST_CRC_OVERLAP", "0"},
    {"STRIX_DIST_HOST_CHUNK_BYTES", "0"},  {"STRIX_MM_PF_IDX_TILE", "0"},
    {"STRIX_MM_PF_IDX_ROWS", "32"},        {"STRIX_MMX_MOE_GU_INTERLEAVE", "0"}, {"STRIX_MMX_MOE_SINGLE", "0"},
    {"STRIX_MMX_MOE_DOWN_SPLIT", "0"},    {"STRIX_MM_ATTEND_DIET", "0"},
    {"STRIX_MM_ATTEND4_DECODE", "0"},
    {"STRIX_MMX_PREFILL_GATHER_TOK", "0"}, {"STRIX_MM_PF_DENSE_QKV", "0"},
    {"STRIX_MMX_PREFILL_TAIL_STREAM", "0"}, {"STRIX_MMX_PREFILL_PIPE_PAIR", "0"},
    {"STRIX_MM_PF_ROUTER_ROWS", "8"}, {"STRIX_MM_PF_MSA_SCRATCH_MB", "512"}, {"STRIX_MM_PF_DENSE_O", "0"},
    {"STRIX_MM_XB_WARM_O", "0"},           {"STRIX_MMX_MOE_TIGHT_GRID", "0"},
    {"STRIX_MMX_MOE_PIPE", "0"},
};

void test_table() {
    std::set<std::string> envs, names;
    for (const md::Item& it : md::items()) {
        check(envs.insert(it.env).second, std::string("unique env ") + it.env);
        check(names.insert(it.name).second, std::string("unique name ") + it.name);
        check(std::strcmp(it.on, it.off) != 0, std::string("on != off for ") + it.env);
        check(std::strlen(it.receipt) > 10, std::string("receipt for ") + it.env);
        const auto f = kFormer.find(it.env);
        check(f != kFormer.end() && f->second == it.off, std::string("off value is the former path for ") + it.env);
    }
    check(envs.size() == kFormer.size(), "every former value has an item");
    check(envs.count("STRIX_DIST_ASYNC_START") == 0, "MiniMax must not export a global async default");
}

void test_receipts(const char* path) {
    const auto receipts = load(path);
    check(receipts.size() == 2, "fixture has the rank and serve receipts");
    const EnvMap none;
    for (const Receipt& r : receipts) {
        const std::string want = md::effective_config(md::parse_cli(r.app, r.argv), lookup(r.env));
        const std::string got = md::effective_config(md::parse_cli(r.app, r.plain), lookup(none));
        auto numerical = [](const std::string& text) {
            std::istringstream input(text); std::string result, line;
            while (std::getline(input,line)) if (!line.starts_with("session_")) result += line + "\n";
            return result;
        };
        check(numerical(got) == numerical(want), r.app + ": plain invocation == receipt\n--- receipt\n" + want + "--- plain\n" + got);
        // Every receipt env value is one of the resolved items (no receipt switch is left unmanaged).
        for (const auto& [k, v] : r.env) {
            bool found = false;
            for (const md::Item& it : md::items()) found = found || (k == it.env && v == it.on);
            check(found, r.app + ": receipt env " + k + "=" + v + " is an item's default");
        }
    }
}

void test_off_switches() {
    const EnvMap none;
    const auto on_lines = lines(md::effective_config(md::Cli{}, lookup(none)));
    for (const md::Item& it : md::items()) {
        const EnvMap off = {{it.env, it.off}};
        const auto r = md::resolve(lookup(off));
        for (const md::Resolved& x : r) {
            if (x.item == &it) check(x.value == it.off && x.source == md::Source::env, std::string(it.env) + " off");
            else check(x.value == x.item->on, std::string(it.env) + " off leaves " + x.item->env + " on");
        }
        const auto off_lines = lines(md::effective_config(md::Cli{}, lookup(off)));
        std::size_t diff = 0;
        for (const auto& l : off_lines) if (!l.starts_with("session_")) diff += on_lines.count(l) == 0 ? 1 : 0;
        check(diff == 1 && off_lines.count(std::string(it.env) + "=" + it.off) == 1, std::string(it.env) + " off changes one line");
    }
}

void test_baseline() {
    for (const char* v : {"1", "yes"}) {
        const EnvMap base = {{md::kBaselineEnv, v}};
        check(md::baseline(lookup(base)), std::string("baseline on for ") + v);
        for (const md::Resolved& x : md::resolve(lookup(base)))
            check(x.value == x.item->off && x.source == md::Source::baseline, std::string("baseline: ") + x.item->env + " off");
    }
    for (const char* v : {"0", ""}) {
        const EnvMap base = {{md::kBaselineEnv, v}};
        check(!md::baseline(lookup(base)), std::string("baseline off for '") + v + "'");
        for (const md::Resolved& x : md::resolve(lookup(base))) check(x.value == x.item->on, "baseline=0 keeps defaults");
    }
    // An explicit item value wins over the master switch.
    const md::Item& first = md::items().front();
    const EnvMap mixed = {{md::kBaselineEnv, "1"}, {first.env, first.on}};
    for (const md::Resolved& x : md::resolve(lookup(mixed)))
        check(x.item == &first ? x.value == first.on : x.value == x.item->off, std::string("baseline + explicit ") + x.item->env);
}

void test_async_group() {
    const auto minimum = md::pf_async_min_rows(nullptr);
    check(minimum == 2048, "async minimum defaults to 2048");
    for (const auto& rows : {std::array<std::size_t, 2>{4096, 3844}, {2048, 2048}, {4096, 2048}}) {
        check(md::pf_async_group(true, rows, minimum), "every member meets inclusive threshold");
        check(!md::pf_async_group(false, rows, minimum), "global off overrides qualifying group");
    }
    for (const auto& rows : {std::array<std::size_t, 2>{4096, 443}, {443, 4096}, {4096, 2047}, {2047, 4096}}) {
        check(!md::pf_async_group(true, rows, minimum), "any small member rejects entire group");
        check(md::pf_async_group(true, rows, md::pf_async_min_rows("0")), "zero restores unconditional async");
        check(!md::pf_async_group(false, rows, 0), "global off overrides zero threshold");
    }
    const std::array<std::size_t, 4> four{4096, 3844, 4096, 443};
    check(!md::pf_async_group(true, four, minimum), "four-member group checks final member");
    check(md::pf_async_group(true, std::span(four).first(3), minimum), "only active members count");
    check(!md::pf_async_group(true, {}, 0), "empty group is not async work");
    check(!md::pf_async_group(true, std::span(four).first(1), md::pf_async_min_rows("4097")), "custom threshold");
    for (const char* value : {"", "-1", "+1", "1x", " 2048", "184467440737095516160"}) {
        bool rejected = false;
        try { (void)md::pf_async_min_rows(value); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, std::string("invalid async minimum: ") + value);
    }
    const EnvMap none, custom{{"STRIX_MM_PF_ASYNC_MIN_ROWS", "0"}};
    check(lines(md::effective_config(md::Cli{}, lookup(none))).count("STRIX_MM_PF_ASYNC_MIN_ROWS=2048") == 1,
          "inspection reports default threshold");
    check(lines(md::effective_config(md::Cli{}, lookup(custom))).count("STRIX_MM_PF_ASYNC_MIN_ROWS=0") == 1,
          "inspection reports custom threshold");
}
void test_session_inspection() {
    const auto directory = std::filesystem::current_path() / "minimax-inspection-fixture";
    check(!std::filesystem::exists(directory), "inspection fixture absent");
    for (const char* app : {"rank", "serve"}) {
        md::Cli cli; cli.app = app;
        for (const char* mode : {"auto", "off", "0", "on", "1"}) {
            EnvMap env{{"STRIX_SESSION_CACHE", mode}, {"STRIX_SESSION_DIR", directory.string()}};
            const auto clean = md::effective_config(cli, lookup(env));
            env["HOME"] = "alternate-home"; env["XDG_CACHE_HOME"] = "alternate-cache";
            env["USERPROFILE"] = "alternate-profile";
            const auto guarded = [&](const char* key) -> const char* {
                check(std::string(key) != "HOME" && std::string(key) != "XDG_CACHE_HOME" &&
                      std::string(key) != "USERPROFILE", "inspection never resolves a host home");
                return lookup(env)(key);
            };
            check(md::effective_config(cli, guarded) == clean, "inspection independent of host paths");
            check(!std::filesystem::exists(directory), "inspection creates no files or directories, including ON");
            std::istringstream input(clean); std::string line; unsigned count = 0;
            while (std::getline(input, line)) if (line.starts_with("session_")) {
                check(line.starts_with("session_requested=") || line.starts_with("session_cache=") ||
                      line.starts_with("session_reason="), "inspection contains only host-independent session policy");
                ++count;
            }
            check(count == 3, "three documented session policy lines");
            env[md::kBaselineEnv] = "1";
            const auto base = md::effective_config(cli, guarded);
            check(base.find("session_cache=off\nsession_reason=baseline\n") != std::string::npos,
                  "baseline inspection is off");
        }
    }
}
void test_session_defaults() {
    md::Cli cli; cli.app = "serve";
    for (const EnvMap& off : {EnvMap{}, EnvMap{{"STRIX_SESSION_CACHE", "0"}},
                             EnvMap{{"STRIX_SESSION_CACHE", "off"}}, EnvMap{{"STRIX_MM_BASELINE", "1"}},
                             EnvMap{{"STRIX_BASELINE", "1"}, {"STRIX_SESSION_CACHE", "1"}}}) {
        const auto policy = md::session_policy(cli, [&](const char* key) -> const char* {
            const std::string name(key);
            check(name != "HOME" && name != "USERPROFILE" && name != "XDG_CACHE_HOME" &&
                  name != "STRIX_SESSION_DIR" && name != "STRIX_SESSION_RAM_BYTES" &&
                  name != "STRIX_SESSION_DISK_BYTES", "OFF startup does not resolve resources");
            return lookup(off)(key);
        });
        check(!policy.enabled && policy.ram_bytes == 0 && policy.directory.empty(), "OFF startup uses pure policy");
    }
    EnvMap env{{"STRIX_SESSION_DISK","0"},{"HOME","/test-home"}};
    auto c = md::session_config(cli, lookup(env), 100ULL << 30);
    check(!c.enabled && c.reason == "route not yet GPU-qualified: " + c.route &&
          c.ram_bytes == (8ULL << 30) && c.max_parked == 8 && !c.disk, "unqualified default declines");
    env["STRIX_SESSION_CACHE"] = "1";
    check(md::session_config(cli,lookup(env),100ULL<<30).enabled, "explicit qualification gate opt-in");
    env.erase("STRIX_SESSION_CACHE");
    check(c.directory == "/test-home/.cache/strix-engine/sessions", "default disk path");
    env["XDG_CACHE_HOME"] = "/test-xdg";
    check(md::session_config(cli,lookup(env),100ULL<<30).directory == "/test-xdg/strix-engine/sessions", "XDG path");
    for (const char* key : {"STRIX_SESSION_CACHE","STRIX_BASELINE", md::kBaselineEnv}) {
        env[key] = std::string(key) == "STRIX_SESSION_CACHE" ? "0" : "1";
        check(!md::session_config(cli,lookup(env),100ULL<<30).enabled, std::string("session off switch ") + key);
        env.erase(key);
    }
    cli.capacity = 122880;
    c = md::session_config(cli,lookup(env),100ULL<<30);
    check(!c.enabled && !c.reason.empty(), "120K refuses default budget cleanly");
    cli.capacity = 8256;
    check(!md::session_config(cli,lookup(env),2ULL<<30).enabled, "MemAvailable guard");
    env["STRIX_SESSION_RAM_BYTES"] = "7000000000"; env["STRIX_SESSION_MAX_PARKED"] = "3";
    c = md::session_config(cli,lookup(env),100ULL<<30);
    check(!c.enabled && c.ram_bytes == 7000000000 && c.max_parked == 3, "budget overrides");
    const auto directory = std::filesystem::current_path() / "minimax-explicit-defaults-fixture";
    check(!std::filesystem::exists(directory), "explicit test directory fresh");
    env["STRIX_SESSION_CACHE"] = "0";
    env[md::kBaselineEnv] = "1";
    env["STRIX_SESSION_DIR"] = "invalid-default-cache";
    c = md::session_config(cli,lookup(env),100ULL<<30,true,directory.string());
    check(c.enabled && c.disk && c.directory == directory.string(), "explicit disk independent of automatic switches and path");
    env["STRIX_SESSION_DISK_BYTES"] = "1";
    check(!md::session_config(cli,lookup(env),100ULL<<30,true,directory.string()).disk, "explicit disk cap retained");
    env.erase("STRIX_SESSION_DISK_BYTES");
    check(!md::session_config(cli,lookup(env),2ULL<<30,true,directory.string()).enabled, "explicit memory guard retained");
    std::filesystem::remove_all(directory);
    cli.slots = 1;
    check(!md::session_config(cli,lookup(env),100ULL<<30).enabled, "transaction slot precondition");
}
// Execute the same serving admission path with a CPU factory. A baseline must
// reach ordinary serving without constructing a runtime; explicit ON failures
// must reach the server-style nonzero exit handler.
void test_session_startup() {
    md::Cli cli; cli.app = "serve"; cli.capacity = 32;
    for (const char* baseline : {"STRIX_MM_BASELINE", "STRIX_BASELINE"}) {
        for (const char* on : {"1", "on"}) {
            EnvMap env{{baseline, "1"}, {"STRIX_SESSION_CACHE", on}, {"STRIX_SESSION_DISK", "0"}};
            const auto config = md::session_config(cli, lookup(env), 100ULL << 30);
            bool attempted = false, served = false;
            std::string log;
            auto serve = [&] {
                try {
                    auto runtime = strix::minimax::session::admit_runtime(config.enabled, true, config.reason,
                        [&]() -> std::unique_ptr<int> { attempted = true; throw std::runtime_error("injected failure"); },
                        [&](const std::string& line) { log = line; });
                    check(!runtime, "baseline runtime is absent");
                    served = true;
                    return 0;
                } catch (const std::exception&) { return 2; }
            };
            check(!config.enabled && config.reason == "baseline", "baseline resolves sessions OFF over explicit ON");
            check(serve() == 0 && served && !attempted, "baseline ordinary serving continues and exits zero");
            check(log == "session cache: off (baseline)", "baseline diagnostic");
        }
    }
    EnvMap env{{"STRIX_SESSION_CACHE", "1"}, {"STRIX_SESSION_DISK", "0"}};
    auto config = md::session_config(cli, lookup(env), 100ULL << 30);
    bool served = false;
    std::string log;
    int exit = 0;
    try {
        (void)strix::minimax::session::admit_runtime(config.enabled, true, config.reason,
            []() -> std::unique_ptr<int> { throw std::runtime_error("injected failure"); },
            [&](const std::string& line) { log = line; });
        served = true;
    } catch (const std::exception&) { exit = 2; }
    check(config.enabled && exit == 2 && !served, "explicit ON initialization failure exits nonzero");
    check(log == "session cache: initialization failed (injected failure)", "explicit ON failure diagnostic");
}

void test_cli() {
    const auto r = md::parse_cli("rank", {"M", "1", "tcp-listen:9000", "OUT", "--slots", "2", "--capacity", "100", "--max-rows", "16"});
    check(r.transport == "tcp-listen:9000" && r.slots == 2 && r.capacity == 100 && r.max_rows == 16, "rank cli");
    const auto s = md::parse_cli("serve", {"M", "--transport", "usb4gpu", "--slots", "4"});
    check(s.app == "serve" && s.transport == "usb4gpu" && s.slots == 4 && s.capacity == 8256 && s.max_rows == 4096, "serve cli");
    const auto override_cli = md::parse_cli("serve", {"M", "--dspark-mode", "on"});
    const EnvMap bad_mode = {{"STRIX_MM_SERVE_DSPARK", "invalid"}, {md::kBaselineEnv, "1"}};
    check(lines(md::effective_config(override_cli, lookup(bad_mode))).count("serve.dspark_mode=on") == 1,
          "CLI overrides env and baseline before validation");
    const auto d = md::parse_cli("serve", {"M"});
    check(d.transport == "usb4-auto" && d.slots == 8, "serve cli defaults = receipt");
}

void test_serve_preflight() {
    const auto cli = md::parse_cli("serve", {"M"});
    const EnvMap none;
    const auto defaults = lines(md::effective_config(cli, lookup(none)));
    for (const char* line : {"serve.sched=interleave", "serve.service_ms=250", "serve.gap_ms=1500",
                             "serve.short_first=1", "serve.nest=1", "serve.dspark_mode=auto", "serve.dspark_effective=per_request_if_assets", "serve.dspark_prior=structural-v1", "dspark.prior=structural-v1", "serve.dspark_inflight=1"})
        check(defaults.count(line) == 1, std::string("resolved policy: ") + line);
    const EnvMap prior_env = {{"STRIX_MM_DSPARK_PRIOR", "unqualified-v1"}};
    check(lines(md::effective_config(cli, lookup(prior_env))).count("serve.dspark_reason=plain:prior_unqualified") == 1, "prior env fallback");
    for (const char* app : {"serve", "rank"}) {
        auto c = md::Cli{}; c.app = app; c.dspark_prior = "structural-v1";
        const EnvMap invalid = {{"STRIX_MM_DSPARK_PRIOR", "invalid"}};
        check(lines(md::effective_config(c, lookup(invalid))).count("dspark.prior=structural-v1") == 1, "prior CLI overrides invalid env");
        c.dspark_prior.clear(); bool rejected = false;
        try { (void)md::effective_config(c, lookup(invalid)); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "both apps reject invalid prior");
    }
    check(md::parse_cli("rank", {"M", "0", "usb4-auto", "O", "--dspark-prior", "unqualified-v1"}).dspark_prior == "unqualified-v1", "rank prior CLI");
    const EnvMap custom = {{"STRIX_MM_SERVE_SCHED", "0"}, {"STRIX_MM_SERVE_SERVICE_MS", "0"},
                          {"STRIX_MM_SERVE_GAP_MS", "100"}, {"STRIX_MM_SERVE_SHORT_FIRST", "0"},
                          {"STRIX_MM_SERVE_NEST", "0"}};
    const auto printed = lines(md::effective_config(cli, lookup(custom)));
    for (const char* line : {"serve.sched=off", "serve.service_ms=0", "serve.gap_ms=100",
                             "serve.short_first=0", "serve.nest=0"})
        check(printed.count(line) == 1, std::string("custom policy: ") + line);
    check(printed.count("serve.gap_ms=1500") == 0, "different deadlines have different config identity");
    const EnvMap base = {{md::kBaselineEnv, "1"}};
    const auto baseline = lines(md::effective_config(cli, lookup(base)));
    check(baseline.count("serve.sched=off") == 1 && baseline.count("serve.service_ms=0") == 1 &&
          baseline.count("serve.short_first=0") == 1 && baseline.count("serve.nest=0") == 1 &&
          baseline.count("serve.gap_ms=1500") == 1, "baseline policy resolves before export");
    for (const char* name : {"STRIX_MM_SERVE_SCHED", "STRIX_MM_SERVE_SERVICE_MS", "STRIX_MM_SERVE_GAP_MS",
                             "STRIX_MM_SERVE_SHORT_FIRST", "STRIX_MM_SERVE_NEST", "STRIX_MM_SERVE_DSPARK", "STRIX_MM_SERVE_DSPARK_INFLIGHT"}) {
        std::vector<std::string> invalid = {"nan", "inf", "1e999", "4294967296", "-1", "invalid"};
        if (std::strcmp(name, "STRIX_MM_SERVE_GAP_MS") == 0) invalid.push_back("0");
        for (const auto& value : invalid) {
            const EnvMap env = {{name, value}};
            bool rejected = false;
            try { (void)md::effective_config(cli, lookup(env)); }
            catch (const std::invalid_argument& e) { rejected = std::string(e.what()).find(name) != std::string::npos; }
            check(rejected, std::string("serve preflight rejects ") + name + "=" + value);
            // Serving-only policy must not reject a standalone rank invocation.
            (void)md::effective_config(md::Cli{}, lookup(env));
        }
    }
}

// Last: changes the process environment.
void test_apply() {
    (void)::setenv("STRIX_SESSION_DISK", "0", 1); // CPU checks never create user cache directories.
    for (const md::Item& it : md::items()) (void)::unsetenv(it.env);
    (void)::unsetenv(md::kBaselineEnv);
    (void)::setenv("STRIX_MM_XB_PUBLISH", "0", 1);  // a user value
    char flag[] = "--print-effective-config", app[] = "strix-minimax-rank", m[] = "M", r0[] = "0", tr[] = "usb4-auto", out[] = "OUT";
    char* argv[] = {app, m, r0, tr, out, flag};
    check(md::print_or_apply("rank", 6, argv), "print_or_apply with the flag");
    for (const md::Item& it : md::items()) {
        const char* v = std::getenv(it.env);
        if (std::strcmp(it.env, "STRIX_MM_XB_PUBLISH") == 0) {
            check(v != nullptr && std::strcmp(v, "0") == 0, "user value kept");
            check(!md::from_defaults(it.env), "user value is not from the defaults");
        } else {
            check(v != nullptr && std::strcmp(v, it.on) == 0, std::string("apply exported ") + it.env);
            check(md::from_defaults(it.env), std::string("from_defaults ") + it.env);
        }
    }
    // Exercise the actual app entry with and without inspection, for both rank arguments.
    for (const char* name : {"STRIX_MM_SERVE_SCHED", "STRIX_MM_SERVE_SERVICE_MS", "STRIX_MM_SERVE_GAP_MS",
                             "STRIX_MM_SERVE_SHORT_FIRST", "STRIX_MM_SERVE_NEST", "STRIX_MM_SERVE_DSPARK", "STRIX_MM_SERVE_DSPARK_INFLIGHT"}) {
        const char* old = std::getenv(name);
        const std::string saved = old == nullptr ? "" : old;
        const bool had = old != nullptr;
        (void)::setenv(name, "nan", 1);
        for (char rank : {'0', '1'}) {
            char rank_value[] = {rank, 0};
            char rank_flag[] = "--rank";
            char* serve_argv[] = {app, m, rank_flag, rank_value, flag};
            for (int count : {4, 5}) {
                bool rejected = false;
                try { (void)md::print_or_apply("serve", count, serve_argv); }
                catch (const std::invalid_argument&) { rejected = true; }
                check(rejected, std::string("app preflight rejects ") + name + " on rank " + rank);
            }
        }
        if (had) (void)::setenv(name, saved.c_str(), 1);
        else (void)::unsetenv(name);
    }
    char* argv2[] = {app, m, r0, tr, out};
    check(!md::print_or_apply("rank", 5, argv2), "print_or_apply without the flag");
    // Both argv orders resolve OFF. --session-cache is the CLI form of ON.
    char baseline[] = "--baseline", cache[] = "--session-cache", on[] = "on";
    char* baseline_first[] = {app, m, baseline, cache, on};
    char* baseline_last[] = {app, m, cache, on, baseline};
    for (auto args : {baseline_first, baseline_last}) {
        (void)::unsetenv(md::kBaselineEnv);
        check(!md::print_or_apply("serve", 5, args), "baseline ordinary startup");
        auto config = md::session_config(md::Cli{}, [](const char* key) { return std::getenv(key); }, 100ULL << 30);
        check(!config.enabled && config.reason == "baseline", "--baseline wins in either argv order");
        std::string log;
        auto runtime = strix::minimax::session::admit_runtime(config.enabled, true, config.reason,
            []() -> std::unique_ptr<int> { throw std::runtime_error("must not initialize"); },
            [&](const std::string& line) { log = line; });
        check(!runtime && log == "session cache: off (baseline)", "--baseline continues ordinary serving");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " minimax_production_receipt.txt\n";
        return 2;
    }
    try {
        test_table();
        test_receipts(argv[1]);
        test_off_switches();
        test_baseline();
        test_cli();
        test_async_group();
        test_serve_preflight();
        test_session_inspection();
        test_session_defaults();
        test_session_startup();
        test_apply();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "minimax defaults tests: ALL PASS (" << md::items().size() << " items)\n";
    return 0;
}
