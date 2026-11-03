#include "strix/cost_table_cache.hpp"
#include "strix/mtp_cost_table_session.hpp"
#include "strix/mtp_window_controller.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
int failures=0;
void expect(bool pass,const char* message) {
    if(!pass) {std::cerr<<"FAIL: "<<message<<'\n';++failures;}
}
namespace fs=std::filesystem;
using strix::CostTableCache;
using strix::CostTableEntry;
using strix::CostTableKey;
using strix::CostTableLoad;

void set_env(const char* name,const char* value) {
#ifdef _WIN32
    _putenv_s(name,value?value:"");
#else
    if(value) setenv(name,value,1); else unsetenv(name);
#endif
}
fs::path fresh_dir(const char* tag) {
    std::random_device rd;
    auto dir=fs::temp_directory_path()/("strix-cost-table-test-"+std::string(tag)+'-'+std::to_string(rd()));
    fs::remove_all(dir);fs::create_directories(dir);
    return dir;
}
CostTableKey key(std::uint32_t bucket=0) {
    CostTableKey k;
    k.engine="test-engine";k.binary="bin0123";k.device="gfx1151|Radeon 8060S|cu40";k.model="model-abc";k.policy="policy-1";
    k.context_bucket=bucket;
    return k;
}
std::string read(const fs::path& p) {std::ifstream in(p,std::ios::binary);return {std::istreambuf_iterator<char>(in),{}};}
void write(const fs::path& p,const std::string& s) {std::ofstream out(p,std::ios::binary|std::ios::trunc);out<<s;}
std::size_t files_in(const fs::path& dir) {std::size_t n=0;for(const auto& e:fs::directory_iterator(dir)) {(void)e;++n;}return n;}

// Writes a table of `n` entries e0..e(n-1) (ms = 10 + i) under `k` and returns its file.
fs::path seed_table(const fs::path& dir,const CostTableKey& k,std::size_t n) {
    CostTableCache c(dir,k);
    expect(c.load()==CostTableLoad::missing,"a new directory has no table");
    for(std::size_t i=0;i<n;++i) c.record("e"+std::to_string(i),{10.0+static_cast<double>(i),8U});
    expect(c.store(),"store succeeds");
    return c.file();
}

void test_buckets() {
    using strix::cost_table_context_bucket;
    expect(cost_table_context_bucket(0)==0 && cost_table_context_bucket(8191)==0,"below 8K is bucket 0");
    expect(cost_table_context_bucket(8192)==1 && cost_table_context_bucket(16383)==1,"8K..16K is bucket 1");
    expect(cost_table_context_bucket(16384)==2 && cost_table_context_bucket(262144)==6,"one bucket per doubling");
    expect(cost_table_context_bucket(std::numeric_limits<std::uint64_t>::max())<=40U,"bucket bounded");
}

void test_directory_rules() {
    const auto dir=fresh_dir("dir");
    expect(strix::cost_table_directory_admitted(dir/"tables"),"a temp directory is admitted");
    fs::create_directories(dir/"checkout"/".git");
    expect(!strix::cost_table_directory_admitted(dir/"checkout"/"build"/"costs"),"inside a checkout is refused");
    expect(!strix::cost_table_directory_admitted(fs::path{}),"empty is refused");
    set_env("STRIX_COST_TABLE_DIR",(dir/"checkout").string().c_str());
    expect(!strix::cost_table_directory_from_env(),"env inside a checkout = off");
    set_env("STRIX_COST_TABLE_DIR",(dir/"tables").string().c_str());
    expect(strix::cost_table_directory_from_env()==dir/"tables","env outside a checkout = on");
    for(const char* off:{"off","0",""}) {
        set_env("STRIX_COST_TABLE_DIR",off);
#ifdef _WIN32
        if(!*off) continue; // _putenv_s with "" removes the variable on Windows
#endif
        expect(!strix::cost_table_directory_from_env(),"STRIX_COST_TABLE_DIR=off/0/empty = off");
    }
    // Default: the user cache directory (created on the first store).
    set_env("STRIX_COST_TABLE_DIR",nullptr);
#ifdef _WIN32
    const char* base_var="LOCALAPPDATA";
    const auto expected=dir/"cache"/"strix"/"cost-tables";
#else
    const char* base_var="XDG_CACHE_HOME";
    const auto expected=dir/"cache"/"strix"/"cost-tables";
#endif
    const char* saved=std::getenv(base_var);const std::string saved_value=saved?saved:"";
    set_env(base_var,(dir/"cache").string().c_str());
    expect(strix::cost_table_directory_from_env()==expected,"unset = default under the user cache directory");
#ifndef _WIN32
    const char* saved_home=std::getenv("HOME");const std::string home_value=saved_home?saved_home:"";
    set_env("XDG_CACHE_HOME",nullptr);set_env("HOME",(dir/"home").string().c_str());
    expect(strix::cost_table_directory_from_env()==dir/"home"/".cache"/"strix"/"cost-tables","no XDG_CACHE_HOME: ~/.cache");
    set_env("HOME","relative/home");
    expect(!strix::cost_table_directory_from_env(),"no absolute cache base: off");
    set_env("HOME",saved_home?home_value.c_str():nullptr);
    set_env("XDG_CACHE_HOME",(dir/"cache").string().c_str());
#endif
    set_env("STRIX_COST_TABLE","0");
    expect(!strix::cost_table_directory_from_env(),"STRIX_COST_TABLE=0 = off");
    set_env("STRIX_COST_TABLE",nullptr);
    {   // a store creates the default directory on demand; an unwritable one is ignored (no throw, no table)
        CostTableKey k=key();
        CostTableCache c(*strix::cost_table_directory_from_env(),k);
        (void)c.load();c.record("r4",{50.0,4U});
        expect(c.store() && fs::exists(expected),"default directory created on the first store");
        write(dir/"afile","x");
        CostTableCache bad(dir/"afile"/"sub",k);
        (void)bad.load();bad.record("r4",{50.0,4U});
        expect(!bad.store(),"unwritable directory: store fails quietly");
    }
    set_env(base_var,saved?saved_value.c_str():nullptr);
    set_env("STRIX_COST_TABLE_RECHECK","0.5");set_env("STRIX_COST_TABLE_DRIFT","0.2");
    auto o=strix::cost_table_options_from_env();
    expect(o.recheck_fraction==0.5 && o.drift_tolerance==0.2,"options from env");
    set_env("STRIX_COST_TABLE_RECHECK","2");set_env("STRIX_COST_TABLE_DRIFT","-1");
    o=strix::cost_table_options_from_env();
    expect(o.recheck_fraction==0.15 && o.drift_tolerance==0.15,"invalid options keep defaults");
    set_env("STRIX_COST_TABLE_RECHECK",nullptr);set_env("STRIX_COST_TABLE_DRIFT",nullptr);
    fs::remove_all(dir);
}

void test_round_trip_and_recheck() {
    const auto dir=fresh_dir("rt");
    const auto file=seed_table(dir,key(),7);
    expect(fs::exists(file) && file.parent_path()==dir,"table written in the directory");
    expect(file.filename().string().rfind("test-engine-",0)==0 && file.extension()==".costs","file name");
    expect(files_in(dir)==2U,"the table and its lock file, no temporary files left");
    expect(read(file).find("strix-cost-table 1\n")==0,"versioned header");
    CostTableCache c(dir,key());
    expect(c.load()==CostTableLoad::loaded && c.loaded_entries()==7U,"loaded");
    const auto recheck=c.recheck_entries();
    expect(recheck.size()==2U,"ceil(15 % of 7) = 2 entries re-timed");
    std::size_t warm=0;
    for(std::size_t i=0;i<7;++i) {
        const auto name="e"+std::to_string(i);
        const auto e=c.warm(name);
        if(c.rechecking(name)) expect(!e,"a re-checked entry is not warm");
        else {expect(e && e->ms==10.0+static_cast<double>(i) && e->samples==8U,"warm entry exact");++warm;}
    }
    expect(warm==5U,"the rest is warm");
    expect(!c.warm("absent"),"unknown entry is not warm");
    // Within tolerance: no drift, merged store keeps every entry and rotates the drift window.
    for(const auto& n:recheck) {const auto i=std::stoi(n.substr(1));c.record(n,{(10.0+i)*1.1,9U});}
    expect(!c.drifted(),"10 % is within the 15 % drift tolerance");
    expect(c.store(),"merged store");
    CostTableCache d(dir,key());
    expect(d.load()==CostTableLoad::loaded && d.loaded_entries()==7U && d.generation()==c.generation(),"merged reload");
    expect(d.recheck_entries()!=recheck,"next start re-checks other entries");
    for(const auto& n:recheck) {const auto i=std::stoi(n.substr(1));
        const auto e=d.rechecking(n)?std::optional<CostTableEntry>{}:d.warm(n);
        if(e) expect(std::abs(e->ms-(10.0+i)*1.1)<1e-12 && e->samples==9U,"measured value persisted exactly");}
    // Fractions 0 and 1.
    CostTableCache none(dir,key(),{0.0,0.15});
    expect(none.load()==CostTableLoad::loaded && none.recheck_entries().empty(),"fraction 0 re-checks nothing");
    CostTableCache all(dir,key(),{1.0,0.15});
    expect(all.load()==CostTableLoad::loaded && all.recheck_entries().size()==7U && !all.warm("e0"),"fraction 1 re-times all");
    fs::remove_all(dir);
}

void test_drift() {
    const auto dir=fresh_dir("drift");
    seed_table(dir,key(),4);
    CostTableCache c(dir,key());
    expect(c.load()==CostTableLoad::loaded,"loaded");
    const auto recheck=c.recheck_entries();
    expect(recheck.size()==1U,"one of four re-checked");
    std::string other;
    for(std::size_t i=0;i<4;++i) if("e"+std::to_string(i)!=recheck[0]) {other="e"+std::to_string(i);break;}
    expect(c.warm(other).has_value(),"warm before drift");
    const auto i=std::stoi(recheck[0].substr(1));
    c.record(recheck[0],{(10.0+i)*1.2,5U});
    expect(c.drifted(),"20 % drift detected");
    expect(!c.warm(other),"after drift the loaded table is withdrawn");
    expect(c.warm(recheck[0]).has_value(),"measured entry stays usable");
    expect(c.store(),"store after drift");
    CostTableCache d(dir,key(),{0.0,0.15});
    expect(d.load()==CostTableLoad::loaded && d.loaded_entries()==1U && d.warm(recheck[0]),"only the measured entry survives");
    fs::remove_all(dir);
}

void test_corruption_and_mismatch() {
    const auto dir=fresh_dir("bad");
    const auto file=seed_table(dir,key(),3);
    const auto good=read(file);
    const auto loads=[&](const std::string& body)->CostTableLoad {write(file,body);CostTableCache c(dir,key());const auto s=c.load();
        if(s!=CostTableLoad::loaded) {expect(!c.warm("e0") && c.loaded_entries()==0U,"a rejected table gives no prior");}
        return s;};
    expect(loads(good)==CostTableLoad::loaded,"control");
    auto flipped=good;const auto at=good.find("entry e1 ")+9;flipped[at]=static_cast<char>(flipped[at]^1);
    expect(loads(flipped)==CostTableLoad::corrupt,"flipped byte = corrupt");
    expect(loads(good.substr(0,good.size()-5))==CostTableLoad::corrupt,"truncated = corrupt");
    expect(loads(good.substr(0,good.size()-1))==CostTableLoad::corrupt,"missing final newline = corrupt");
    expect(loads("")==CostTableLoad::corrupt,"empty = corrupt");
    expect(loads(std::string(2U<<20U,'x'))==CostTableLoad::corrupt,"oversized = corrupt");
    auto v2=good;v2.replace(0,18,"strix-cost-table 2");
    expect(loads(v2)==CostTableLoad::version,"other version");
    // A well-formed table of another key (re-serialized with a valid checksum) is never trusted.
    std::map<std::string,CostTableEntry,std::less<>> entries{{"e0",{10.0,8U}}};
    for(int field=0;field<6;++field) {
        auto k=key();
        if(field==0) k.binary="other-binary";
        if(field==1) k.device="gfx1100";
        if(field==2) k.model="other-model";
        if(field==3) k.policy="policy-2";
        if(field==4) k.context_bucket=3;
        if(field==5) k.engine="other-engine";
        expect(loads(CostTableCache::serialize(k,1,0,entries))==CostTableLoad::mismatch,"another key = mismatch");
    }
    // Hand-edited values the checksum would accept: non-finite, zero, duplicate, bad name.
    const auto with=[&](const std::string& lines,std::size_t n) {
        std::string body="strix-cost-table 1\nengine test-engine\nbinary bin0123\ndevice gfx1151|Radeon%208060S|cu40\n"
            "model model-abc\npolicy policy-1\nbucket 0\ngeneration 1\nepoch 0\nentries "+std::to_string(n)+"\n"+lines;
        std::uint64_t h=1469598103934665603ULL;
        for(const char c:body) {h^=static_cast<unsigned char>(c);h*=1099511628211ULL;}
        static constexpr char digits[]="0123456789abcdef";
        std::string hex(16,'0');for(std::size_t i=16;i-->0;h>>=4U) hex[i]=digits[h&15U];
        return body+"checksum "+hex+"\n";
    };
    expect(loads(with("entry e0 10 8\n",1))==CostTableLoad::loaded,"hand-built control");
    expect(loads(with("entry e0 nan 8\n",1))==CostTableLoad::corrupt,"nan = corrupt");
    expect(loads(with("entry e0 0 8\n",1))==CostTableLoad::corrupt,"zero cost = corrupt");
    expect(loads(with("entry e0 10 0\n",1))==CostTableLoad::corrupt,"zero samples = corrupt");
    expect(loads(with("entry e0 10 8\nentry e0 11 8\n",2))==CostTableLoad::corrupt,"duplicate = corrupt");
    expect(loads(with("entry e/0 10 8\n",1))==CostTableLoad::corrupt,"bad name = corrupt");
    expect(loads(with("entry e0 10 8\n",2))==CostTableLoad::corrupt,"count mismatch = corrupt");
    // A rejected table is replaced on the next store, by what was measured.
    write(file,flipped);
    CostTableCache c(dir,key());
    expect(c.load()==CostTableLoad::corrupt,"corrupt again");
    c.record("e9",{3.0,4U});
    expect(c.store(),"store replaces a corrupt table");
    CostTableCache d(dir,key(),{0.0,0.15});
    expect(d.load()==CostTableLoad::loaded && d.loaded_entries()==1U && d.warm("e9"),"replaced table loads");
    fs::remove_all(dir);
}

void test_refusals() {
    const auto dir=fresh_dir("ref");
    auto k=key();k.binary.clear();
    CostTableCache c(dir,k);
    expect(c.load()!=CostTableLoad::loaded,"no binary identity, no table");
    c.record("e0",{1.0,3U});
    expect(!c.store() && files_in(dir)==0U,"no binary identity, never stored");
    CostTableCache e(dir,key());
    (void)e.load();
    e.record("e0",{std::nan(""),3U});e.record("e1",{0.0,3U});e.record("e2",{1.0,0U});e.record("bad name",{1.0,3U});
    expect(!e.dirty() && !e.store(),"invalid measurements are ignored");
    auto bad=key();bad.engine="../x";
    CostTableCache b(dir,bad);
    expect(b.load()!=CostTableLoad::loaded && b.file().empty(),"an engine name with a path is refused");
    fs::remove_all(dir);
}

void test_policy_identity() {
    set_env("STRIX_WP25_TEST_SWITCH",nullptr);
    const auto a=strix::cost_table_policy_identity("cli");
    expect(a.size()==64U,"sha-256 hex");
    expect(a==strix::cost_table_policy_identity("cli"),"stable");
    expect(a!=strix::cost_table_policy_identity("cli2"),"configuration is keyed");
    set_env("STRIX_WP25_TEST_SWITCH","1");
    expect(a!=strix::cost_table_policy_identity("cli"),"STRIX_ environment is keyed");
    set_env("STRIX_WP25_TEST_SWITCH",nullptr);
    set_env("STRIX_COST_TABLE_RECHECK","0.3");
    expect(a==strix::cost_table_policy_identity("cli"),"cost-table variables are not keyed");
    set_env("STRIX_COST_TABLE_RECHECK",nullptr);
    set_env("STRIX_CYCLE_LOG","1");
    expect(a==strix::cost_table_policy_identity("cli"),"the per-window diagnostic log is not keyed");
    set_env("STRIX_CYCLE_LOG",nullptr);
}

// Controller side: the table is a bootstrap prior, kept apart from the measured samples (no sample weight).
void test_controller_seed() {
    using strix::MtpMinPWindowController;
    // Cold controller tries the unmeasured R8 first; a seeded one ranks by the prior.
    const std::vector<float> p(7,0.95F);
    const std::array<std::uint8_t,3> fits{1,1,1};
    MtpMinPWindowController cold({2U,4U,8U},0.8F);
    expect(cold.fresh(),"cold controller is fresh");
    expect(cold.choose(p,p.size(),fits)==2U,"cold: unmeasured R8 is tried");
    MtpMinPWindowController warm({2U,4U,8U},0.8F);
    const std::array<MtpMinPWindowController::CostPrior,4> priors{{{1U,45.0,8U},{2U,46.0,8U},{4U,50.0,8U},{8U,400.0,8U}}};
    warm.seed_costs(priors);
    expect(warm.fresh() && warm.active_priors()==4U,"seeding keeps the controller fresh");
    expect(warm.choose(p,p.size(),fits)==1U,"warm: R4 by expected tokens per ms, R8 priced out");
    expect(!warm.fresh(),"a choice ends freshness");
    const auto stats=warm.owner_stats();
    expect(stats.size()==4U && stats[0].cycles==0U && stats[3].cycles==0U && stats[3].cycle_ms==400.0,"stats: no samples, prior in use");
    expect(warm.measured_costs().empty(),"no measured costs from priors");
    warm.observe(1U,4U,60.0); // first window of an owner: warm-up, no sample
    for(int k=0;k<3;++k) warm.observe(1U,4U,52.0);
    const auto m=warm.measured_costs();
    expect(m.size()==1U && m[0].rows==4U && m[0].ms==52.0 && m[0].samples==3U,"measured R4 median for write-back");
    warm.begin_session();
    expect(warm.fresh() && warm.active_priors()==0U && warm.owner_stats()[3].cycle_ms==0.0,"begin_session drops priors");
}
// Bootstrap prior: an option's prior is used only until its first own timed sample; the first sample is the
// estimate (provisional minimum, exactly as without a prior), never averaged with the prior.
void test_prior_until_first_sample() {
    using strix::MtpMinPWindowController;
    MtpMinPWindowController c({2U,4U,8U},0.8F);
    const std::array<MtpMinPWindowController::CostPrior,2> priors{{{4U,50.0,8U},{8U,400.0,8U}}};
    c.seed_costs(priors);
    c.observe(1U,4U,90.0); // R4 warm-up window: untimed, the prior stays in use
    expect(c.owner_stats()[2].cycles==0U && c.owner_stats()[2].cycle_ms==50.0 && c.active_priors()==2U,"warm-up keeps the prior");
    c.observe(1U,4U,70.0); // first timed sample replaces the prior entirely
    expect(c.owner_stats()[2].cycles==1U && c.owner_stats()[2].cycle_ms==70.0 && c.active_priors()==1U,"first own sample replaces the prior");
    MtpMinPWindowController plain({2U,4U,8U},0.8F);
    plain.observe(1U,4U,90.0);plain.observe(1U,4U,70.0);
    expect(plain.owner_stats()[2].cycle_ms==c.owner_stats()[2].cycle_ms,"own estimate equals the unseeded one (no sample weight)");
    // A prior never lands on an option that already has samples.
    MtpMinPWindowController d({2U,4U,8U},0.8F);
    d.observe(1U,4U,90.0);d.observe(1U,4U,70.0);
    const std::array<MtpMinPWindowController::CostPrior,1> late{{{4U,10.0,8U}}};
    d.seed_costs(late);
    expect(d.owner_stats()[2].cycle_ms==70.0 && d.active_priors()==0U,"no prior over own samples");
}
// Every prior still in use expires after prior_cycles observed cycles; the decisions then equal a cold controller
// that saw the same samples.
void test_prior_expiry() {
    using strix::MtpMinPWindowController;
    const std::vector<float> p(7,0.95F);
    const std::array<std::uint8_t,3> fits{1,1,1};
    MtpMinPWindowController c({2U,4U,8U},0.8F),plain({2U,4U,8U},0.8F);
    const std::array<MtpMinPWindowController::CostPrior,4> priors{{{1U,45.0,8U},{2U,46.0,8U},{4U,50.0,8U},{8U,400.0,8U}}};
    c.seed_costs(priors);
    expect(MtpMinPWindowController::prior_cycles==4U,"priors expire after 4 cycles");
    for(std::uint64_t k=1;k<4U;++k) {
        c.observe(0U,2U,40.0);plain.observe(0U,2U,40.0); // R2 only: R4/R8/serial priors stay unmeasured
        expect(c.active_priors()==(k==1U?4U:3U),"priors live before expiry (R2 warm-up keeps its prior once)");
    }
    expect(c.choose(p,p.size(),fits)==1U,"before expiry: R8 priced out by its prior");
    (void)plain.choose(p,p.size(),fits);
    c.observe(0U,2U,40.0);plain.observe(0U,2U,40.0);
    expect(c.active_priors()==0U && c.owner_stats()[3].cycle_ms==0.0 && c.owner_stats()[0].cycle_ms==0.0,"all priors expire after prior_cycles cycles");
    expect(c.choose(p,p.size(),fits)==2U && plain.choose(p,p.size(),fits)==2U,"after expiry: unmeasured R8 is tried as cold");
    // A bucket re-seed (reset_costs + seed_costs) restarts the window.
    c.reset_costs();
    const std::array<MtpMinPWindowController::CostPrior,1> again{{{8U,300.0,8U}}};
    c.seed_costs(again);
    expect(c.active_priors()==1U && c.owner_stats()[3].cycle_ms==300.0,"re-seed restarts the prior window");
}
// OFF: no prior (or an empty prior list) decides exactly as the controller without the feature, over a long
// pseudo-random sequence of draft decisions, choices, calibrations and observations.
void test_prior_off_identical() {
    using strix::MtpMinPWindowController;
    for(const auto model:{strix::MtpWindowCostEstimate::Model::robust,strix::MtpWindowCostEstimate::Model::legacy}) {
        MtpMinPWindowController a({2U,4U,8U},0.7F,model),b({2U,4U,8U},0.7F,model);
        b.seed_costs({});
        const std::array<MtpMinPWindowController::CostPrior,2> bad{{{4U,0.0,8U},{8U,std::nan(""),8U}}};
        b.seed_costs(bad); // invalid priors are ignored
        std::mt19937 rng(1234U);
        std::uniform_real_distribution<float> u(0.0F,1.0F);
        bool same=b.active_priors()==0U;
        for(int cycle=0;cycle<3000 && same;++cycle) {
            std::vector<float> p(1U+rng()%7U);
            for(auto& q:p) q=u(rng);
            same=a.worth_drafting(p)==b.worth_drafting(p);
            std::array<std::uint8_t,3> fits{1,static_cast<std::uint8_t>(rng()%4U!=0U),static_cast<std::uint8_t>(rng()%3U!=0U)};
            const auto ca=a.choose(p,p.size(),fits),cb=b.choose(p,p.size(),fits);
            same=same && ca==cb;
            const auto accepted=static_cast<std::size_t>(rng()%(p.size()+1U));
            a.calibrate(p,accepted,p.size());b.calibrate(p,accepted,p.size());
            const double ms=20.0+static_cast<double>(rng()%800U)/10.0;
            a.observe(ca,1U+accepted,ms);b.observe(cb,1U+accepted,ms);
            if(cycle%500==499) {a.begin_session();b.begin_session();}
        }
        const auto sa=a.owner_stats(),sb=b.owner_stats();
        for(std::size_t i=0;i<sa.size() && same;++i)
            same=sa[i].cycles==sb[i].cycles && sa[i].cycle_ms==sb[i].cycle_ms && sa[i].probes==sb[i].probes;
        expect(same,"no prior: decisions and estimates identical to the controller without the feature");
    }
}

// Finding 4: a store replaces an existing table (Windows: MoveFileExW REPLACE_EXISTING), no temporary file stays.
void test_replace_existing() {
    const auto dir=fresh_dir("replace");
    CostTableCache a(dir,key());
    (void)a.load();
    a.record("e0",{1.0,3U});
    expect(a.store(),"first store");
    for(int round=0;round<3;++round) {
        CostTableCache b(dir,key(),{0.0,0.15});
        expect(b.load()==CostTableLoad::loaded,"reload before replace");
        b.record("e0",{2.0+round,4U});
        expect(b.store(),"store replaces an existing file");
        CostTableCache c(dir,key(),{0.0,0.15});
        expect(c.load()==CostTableLoad::loaded && c.warm("e0") && c.warm("e0")->ms==2.0+round,"replacement visible");
    }
    std::size_t tables=0,locks=0,other=0;
    for(const auto& e:fs::directory_iterator(dir)) {
        const auto name=e.path().filename().string();
        if(e.path().extension()==".costs") ++tables; else if(e.path().extension()==".lock") ++locks; else ++other;
    }
    expect(tables==1U && locks==1U && other==0U,"one table, its lock file, no temporary files");
    expect(!strix::cost_table_executable_identity().empty(),"executable identity on this platform");
    fs::remove_all(dir);
}

// Finding 5: a stale writer (loaded before another writer withdrew the table on drift) cannot restore the
// withdrawn entries or their old costs.
void test_stale_writer_after_drift() {
    const auto dir=fresh_dir("stale");
    seed_table(dir,key(),4); // e0..e3 = 10..13 ms, epoch 0
    CostTableCache a(dir,key()),b(dir,key());
    expect(a.load()==CostTableLoad::loaded && b.load()==CostTableLoad::loaded,"both writers loaded epoch 0");
    const auto r=a.recheck_entries()[0];
    a.record(r,{100.0,5U});
    expect(a.drifted() && a.store(),"writer A withdraws on drift");
    // Writer B never saw drift; it measured one entry of its own.
    std::string other;
    for(int i=0;i<4;++i) if("e"+std::to_string(i)!=r && !b.rechecking("e"+std::to_string(i))) {other="e"+std::to_string(i);break;}
    b.record(other,{12.5,6U});
    expect(b.store(),"writer B stores");
    CostTableCache c(dir,key(),{0.0,0.15});
    expect(c.load()==CostTableLoad::loaded && c.epoch()==1U,"epoch bumped by the withdrawal");
    expect(c.loaded_entries()==2U && c.warm(r) && c.warm(r)->ms==100.0 && c.warm(other) && c.warm(other)->ms==12.5,
        "only measurements after the withdrawal survive");
    for(int i=0;i<4;++i) {
        const auto n="e"+std::to_string(i);
        if(n!=r && n!=other) expect(!c.warm(n),"a withdrawn entry is not restored by a stale writer");
    }
    for(int i=0;i<4;++i) {
        const auto n="e"+std::to_string(i);
        if(n!=other) expect(!b.warm(n),"the stale writer drops its withdrawn snapshot");
    }
    fs::remove_all(dir);
}

// Finding 5: concurrent writers merge under the per-key lock (no lost entry).
void test_concurrent_writers() {
    const auto dir=fresh_dir("conc");
    constexpr int writers=8,rounds=4;
    std::vector<std::thread> threads;
    for(int w=0;w<writers;++w) threads.emplace_back([&dir,w]{
        for(int r=0;r<rounds;++r) {
            CostTableCache c(dir,key(),{0.0,0.15});
            (void)c.load();
            c.record("w"+std::to_string(w)+"r"+std::to_string(r),{1.0+w,3U});
            (void)c.store();
        }});
    for(auto& t:threads) t.join();
    CostTableCache c(dir,key(),{0.0,0.15});
    expect(c.load()==CostTableLoad::loaded && c.loaded_entries()==static_cast<std::size_t>(writers*rounds),"every writer's entries kept");
    expect(c.generation()==static_cast<std::uint64_t>(writers*rounds),"one generation per store");
    fs::remove_all(dir);
}

// Finding 6: a huge file is rejected by a bounded read (cold), never a throw.
void test_bounded_read() {
    const auto dir=fresh_dir("big");
    CostTableCache probe(dir,key());
    write(probe.file(),std::string(CostTableCache::max_file_bytes+1U,'x'));
    expect(CostTableCache::read_file(probe.file(),key()).status==CostTableLoad::corrupt,"limit + 1 bytes = corrupt");
    write(probe.file(),std::string(64U<<20U,'y'));
    CostTableCache c(dir,key());
    expect(c.load()==CostTableLoad::corrupt && c.loaded_entries()==0U,"64 MiB file = corrupt, cold");
    fs::create_directories(dir/"adir.costs");
    expect(CostTableCache::read_file(dir/"adir.costs",key()).status!=CostTableLoad::loaded,"a directory never loads");
    fs::remove_all(dir);
}

// Session tests: a warm table for buckets 0 and 1 with r1/r2/r4/r8.
using strix::MtpCostTableSession;
using strix::MtpMinPWindowController;
fs::path session_table(const char* tag) {
    const auto dir=fresh_dir(tag);
    for(std::uint32_t b=0;b<2;++b) {
        CostTableCache c(dir,key(b));
        (void)c.load();
        c.record("r1",{45.0,8U});c.record("r2",{46.0,8U});c.record("r4",{50.0+b,8U});c.record("r8",{400.0,8U});
        expect(c.store(),"session table stored");
    }
    return dir;
}
bool same_decisions(MtpMinPWindowController& a,MtpMinPWindowController& b) {
    const std::array<std::uint8_t,3> fits{1,1,1};
    bool same=true;
    for(const float q:{0.95F,0.9F,0.85F,0.6F}) for(std::size_t n=1;n<=7;++n) {
        const std::vector<float> p(n,q);
        same=same && a.worth_drafting(p)==b.worth_drafting(p);
    }
    const std::vector<float> p(7,0.95F);
    return same && a.choose(p,p.size(),fits)==b.choose(p,p.size(),fits);
}
// Finding 1: a sampled session never consumes a seeded prior, from its first draft decision on.
void test_session_sampled_stays_cold() {
    const auto dir=session_table("samp");
    MtpCostTableSession s(dir,key(),{0.0,0.15},{1U,2U,4U,8U});
    MtpMinPWindowController cold({2U,4U,8U},0.8F),sampled({2U,4U,8U},0.8F);
    s.prepare(sampled,100U,true);
    expect(s.state()==MtpCostTableSession::State::cold,"sampled session resolved cold");
    expect(same_decisions(cold,sampled),"sampled: draft and window decisions equal the cold controller's");
    for(int k=0;k<3;++k) sampled.observe(1U,4U,52.0);
    s.flush(sampled);
    CostTableCache t(dir,key(),{0.0,0.15});
    expect(t.load()==CostTableLoad::loaded && t.warm("r4")->ms==50.0,"sampled session never written");
    // Control: the same table seeds a greedy session (its decisions differ from cold).
    MtpMinPWindowController cold2({2U,4U,8U},0.8F),greedy({2U,4U,8U},0.8F);
    s.prepare(greedy,100U,false);
    expect(s.state()==MtpCostTableSession::State::warm && !same_decisions(cold2,greedy),"greedy session is seeded");
    fs::remove_all(dir);
}
// Finding 3: the bucket is checked before every decision; measurements split at a crossing.
void test_session_bucket_crossing() {
    const auto dir=session_table("cross");
    MtpCostTableSession s(dir,key(),{0.0,0.15},{1U,2U,4U,8U});
    MtpMinPWindowController c({2U,4U,8U},0.8F);
    s.prepare(c,8000U,false);
    expect(s.bucket()==0U && c.owner_stats()[2].cycle_ms==50.0,"bucket 0 seeded");
    c.observe(1U,4U,60.0); // warm-up
    for(int k=0;k<3;++k) c.observe(1U,4U,55.0);
    s.prepare(c,8100U,false);
    expect(s.bucket()==0U && c.owner_stats()[2].cycles==3U,"same bucket: no change");
    s.prepare(c,8192U,false);
    expect(s.bucket()==1U && c.owner_stats()[2].cycles==0U && c.owner_stats()[2].cycle_ms==51.0,
        "crossing: bucket 1 prior, measurements of bucket 0 moved out");
    for(int k=0;k<3;++k) c.observe(1U,4U,70.0);
    s.flush(c);
    CostTableCache b0(dir,key(0),{0.0,0.15}),b1(dir,key(1),{0.0,0.15});
    expect(b0.load()==CostTableLoad::loaded && b0.warm("r4")->ms==55.0,"bucket 0 keeps its own measurements");
    expect(b1.load()==CostTableLoad::loaded && b1.warm("r4")->ms==70.0,"bucket 1 gets the later ones");
    fs::remove_all(dir);
}
// Finding 7: only serial-equivalent owners (seedable rows) take a prior or are written; a non-fresh controller and an
// excluded (cohort) session are never seeded / written.
void test_session_seedable_and_exclude() {
    const auto dir=session_table("seedable");
    MtpCostTableSession s(dir,key(),{0.0,0.15},{1U,2U}); // R4/R8 not admitted as serial-equivalent
    MtpMinPWindowController c({2U,4U,8U},0.8F);
    s.prepare(c,10U,false);
    const auto st=c.owner_stats();
    expect(st[0].cycle_ms==45.0 && st[1].cycle_ms==46.0 && st[2].cycle_ms==0.0 && st[3].cycle_ms==0.0,"only seedable rows seeded");
    for(int k=0;k<4;++k) {c.observe(2U,4U,90.0);c.observe(0U,2U,30.0);}
    s.flush(c);
    CostTableCache t(dir,key(),{0.0,0.15});
    expect(t.load()==CostTableLoad::loaded && t.warm("r8")->ms==400.0 && t.warm("r2")->ms==30.0,"non-seedable rows never written");
    MtpCostTableSession e(dir,key(),{0.0,0.15},{1U,2U,4U,8U});
    MtpMinPWindowController d({2U,4U,8U},0.8F);
    e.prepare(d,10U,false);
    for(int k=0;k<4;++k) d.observe(1U,4U,99.0);
    e.exclude();
    e.flush(d);
    CostTableCache u(dir,key(),{0.0,0.15});
    expect(u.load()==CostTableLoad::loaded && u.warm("r4")->ms==50.0,"an excluded session is not written");
    MtpMinPWindowController busy({2U,4U,8U},0.8F);
    const std::vector<float> p(7,0.95F);const std::array<std::uint8_t,3> fits{1,1,1};
    (void)busy.choose(p,p.size(),fits);
    MtpCostTableSession late(dir,key(),{0.0,0.15},{1U,2U,4U,8U});
    late.prepare(busy,10U,false);
    expect(late.state()==MtpCostTableSession::State::cold && busy.owner_stats()[2].cycle_ms==0.0,"a controller that already decided stays cold");
    fs::remove_all(dir);
}

// Sol r2 P2: a consumed withdrawal is not repeated by a later store (newer entries of another writer survive), and
// the withdrawn priors stay unusable in the withdrawing process.
void test_repeated_store_after_withdrawal() {
    const auto dir=fresh_dir("rewith");
    seed_table(dir,key(),4); // e0..e3, epoch 0
    CostTableCache a(dir,key());
    expect(a.load()==CostTableLoad::loaded,"A loaded epoch 0");
    const auto r=a.recheck_entries()[0];
    a.record(r,{100.0,5U});
    expect(a.drifted() && a.store() && a.epoch()==1U,"A withdraws: epoch 1");
    expect(!a.drifted(),"withdrawal consumed by the successful store");
    CostTableCache b(dir,key(),{0.0,0.15});
    expect(b.load()==CostTableLoad::loaded && b.epoch()==1U,"B loaded epoch 1");
    b.record("newer",{7.0,4U});
    expect(b.store(),"B adds a newer entry in epoch 1");
    a.record("later",{8.0,3U});
    expect(a.store() && a.epoch()==1U,"A's second store keeps epoch 1");
    CostTableCache c(dir,key(),{0.0,0.15});
    expect(c.load()==CostTableLoad::loaded && c.epoch()==1U && c.warm("newer") && c.warm("later") && c.warm(r),
        "B's newer entry survives A's second store");
    for(int i=0;i<4;++i) {
        const auto n="e"+std::to_string(i);
        if(n!=r) expect(!a.warm(n) && !c.warm(n),"withdrawn priors stay unusable");
    }
    fs::remove_all(dir);
}
// Sol r2 P1: the table is allowed only when EVERY selectable owner passes the serial-equivalence guard.
void test_seedable_rows_all_admitted() {
    using strix::MtpCostTableSession;
    const auto all=MtpCostTableSession::seedable_rows({2U,4U,8U},{true,true,true});
    expect((all==std::vector<std::uint32_t>{1U,2U,4U,8U}),"all owners admitted: serial and every owner");
    expect(MtpCostTableSession::seedable_rows({2U,4U,8U},{false,true,true}).empty(),"R2 not admitted: no table");
    expect(MtpCostTableSession::seedable_rows({2U,4U,8U},{true,true,false}).empty(),"R8 not admitted: no table");
    expect(MtpCostTableSession::seedable_rows({4U},{true})==(std::vector<std::uint32_t>{1U,4U}),"single admitted owner");
    expect(MtpCostTableSession::seedable_rows({},{}).empty() && MtpCostTableSession::seedable_rows({2U},{}).empty(),"malformed: no table");
}
// GPU timing turns: the HTTP server's automatic slot capacity (MemAvailable at startup: 23, 21, 20 slots in three
// runs of one command) was part of the policy key, so every process wrote and looked up a different table and a
// warm run never loaded one. Two processes of the same route must share one table; requested settings stay keyed.
void test_cross_process_route_key() {
    strix::MtpCostTableRoute a;
    a.tiled=true;a.http=true;a.batched=true;a.kernel_plan=7U;a.execution=9U;a.profiles=11U;
    a.optimization="auto";a.cli="5:--mtp";a.observed_slots=23U;
    auto b=a;b.observed_slots=21U;
    const auto policy=[](const strix::MtpCostTableRoute& r) {return strix::cost_table_policy_identity(strix::mtp_cost_table_configuration(r));};
    expect(strix::mtp_cost_table_configuration(a)==strix::mtp_cost_table_configuration(b) && policy(a)==policy(b),
        "resolved slot capacity is not keyed");
    auto c=a;c.batch_explicit=true;c.batch_cap=4U;
    auto d=a;d.batched=false;
    auto e=a;e.cli="5:--mtq";
    expect(policy(c)!=policy(a) && policy(d)!=policy(a) && policy(e)!=policy(a),"requested batch, route and CLI are keyed");
    const auto key_of=[](const std::string& p) {auto k=key();k.policy=p;return k;};
    // Repro of the miss: a configuration holding the capacity puts process B on another file.
    {
        const auto dir=fresh_dir("xproc-old");
        CostTableCache w(dir,key_of(strix::cost_table_policy_identity("route;batch_slots=23")));
        (void)w.load();w.record("r4",{53.0,15U});
        expect(w.store(),"process A stores");
        CostTableCache r(dir,key_of(strix::cost_table_policy_identity("route;batch_slots=21")));
        expect(r.load()==CostTableLoad::missing && files_in(dir)==2U,"old key: process B misses (control)");
        fs::remove_all(dir);
    }
    // Fixed key: process A measures R4 over requests and flushes; process B (new session, new controller) is seeded.
    const auto dir=fresh_dir("xproc");
    {
        MtpCostTableSession s(dir,key_of(policy(a)),{0.0,0.15},{1U,2U,4U,8U});
        MtpMinPWindowController ctl({2U,4U,8U},0.8F);
        s.prepare(ctl,40U,false);
        expect(s.table(0).load_status()==CostTableLoad::missing,"process A starts without a table");
        ctl.observe(1U,4U,60.0); // warm-up cycle, not timed
        for(int k=0;k<4;++k) ctl.observe(1U,4U,53.0);
        s.flush(ctl);
    }
    MtpCostTableSession s(dir,key_of(policy(b)),{0.0,0.15},{1U,2U,4U,8U});
    MtpMinPWindowController ctl({2U,4U,8U},0.8F);
    s.prepare(ctl,40U,false);
    expect(s.table(0).load_status()==CostTableLoad::loaded && s.table(0).loaded_entries()==1U,
        "process B loads process A's table");
    expect(s.state()==MtpCostTableSession::State::warm && ctl.owner_stats()[2].cycle_ms==53.0,"process B is seeded");
    fs::remove_all(dir);
}
// Baseline: --baseline, STRIX_BASELINE=1 and STRIX_QPF2K_DEFAULTS=0 switch the DEFAULT directory off (source
// "baseline"); an explicit STRIX_COST_TABLE_DIR still wins; STRIX_COST_TABLE=0 is off in every case.
void test_baseline_resolution() {
    using strix::CostTableSource;
    const auto dir=fresh_dir("baseline");
#ifdef _WIN32
    const char* base_var="LOCALAPPDATA";
#else
    const char* base_var="XDG_CACHE_HOME";
#endif
    const char* saved=std::getenv(base_var);const std::string saved_value=saved?saved:"";
    set_env(base_var,(dir/"cache").string().c_str());
    set_env("STRIX_COST_TABLE_DIR",nullptr);set_env("STRIX_COST_TABLE",nullptr);
    set_env("STRIX_BASELINE",nullptr);set_env("STRIX_QPF2K_DEFAULTS",nullptr);
    const auto def=dir/"cache"/"strix"/"cost-tables";
    auto r=strix::cost_table_directory_resolve(false);
    expect(r.path==def && r.source==CostTableSource::default_dir && std::string(strix::cost_table_source_name(r.source))=="default",
        "default directory, source default");
    r=strix::cost_table_directory_resolve(true);
    expect(!r.path && r.source==CostTableSource::baseline && std::string(strix::cost_table_source_name(r.source))=="baseline",
        "--baseline: no default directory, source baseline");
    set_env("STRIX_BASELINE","1");
    r=strix::cost_table_directory_resolve(false);
    expect(!r.path && r.source==CostTableSource::baseline && !strix::cost_table_directory_from_env(),"STRIX_BASELINE=1: baseline");
    set_env("STRIX_BASELINE","0");
    expect(strix::cost_table_directory_resolve(false).source==CostTableSource::default_dir,"STRIX_BASELINE=0: default");
    set_env("STRIX_BASELINE",nullptr);
    set_env("STRIX_QPF2K_DEFAULTS","0");
    expect(strix::cost_table_directory_resolve(false).source==CostTableSource::baseline,"STRIX_QPF2K_DEFAULTS=0: baseline");
    set_env("STRIX_COST_TABLE_DIR",(dir/"explicit").string().c_str());
    r=strix::cost_table_directory_resolve(true);
    expect(r.path==dir/"explicit" && r.source==CostTableSource::explicit_dir &&
        std::string(strix::cost_table_source_name(r.source))=="env","explicit STRIX_COST_TABLE_DIR wins under baseline");
    set_env("STRIX_COST_TABLE","0");
    expect(!strix::cost_table_directory_resolve(false).path &&
        strix::cost_table_directory_resolve(false).source==CostTableSource::off,"STRIX_COST_TABLE=0 wins over everything");
    set_env("STRIX_COST_TABLE",nullptr);set_env("STRIX_COST_TABLE_DIR",nullptr);set_env("STRIX_QPF2K_DEFAULTS",nullptr);
    set_env(base_var,saved?saved_value.c_str():nullptr);
    fs::remove_all(dir);
}
} // namespace

int main() {
    test_buckets();
    test_directory_rules();
    test_baseline_resolution();
    test_round_trip_and_recheck();
    test_drift();
    test_corruption_and_mismatch();
    test_refusals();
    test_policy_identity();
    test_controller_seed();
    test_prior_until_first_sample();
    test_prior_expiry();
    test_prior_off_identical();
    test_replace_existing();
    test_stale_writer_after_drift();
    test_concurrent_writers();
    test_bounded_read();
    test_session_sampled_stays_cold();
    test_session_bucket_crossing();
    test_session_seedable_and_exclude();
    test_repeated_store_after_withdrawal();
    test_seedable_rows_all_admitted();
    test_cross_process_route_key();
    if(failures) {std::cerr<<failures<<" failure(s)\n";return 1;}
    std::cout<<"cost table cache tests passed\n";
    return 0;
}
