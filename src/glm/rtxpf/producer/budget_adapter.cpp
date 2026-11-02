// THE budget adapter: the only GLM file that calls a budget planner / ledger
// API. GLM component sizes in (expert_tier.hpp: TierBudgetIn, WholeDeviceIn,
// BudgetPart), {resident, slots, pinned, disk, staging, min caps, refusal} out,
// plus the allocation ledger. Implemented against tierstream 5fc3441
// (ss_budget_plan / ss_budget_ledger, include/strix_stream/ss.h SS_BUDGET).
// Switching to another shared planner replaces this file only; GLM keeps no
// budget arithmetic of its own (the device-gated spare-slot rules below are
// library constraints, not budget arithmetic).
#include "tierstream_binding.hpp"
#include "tierstream/ts.h"
#include <algorithm>
#include <stdexcept>
namespace strix::glm::rtxpf::producer {
namespace {
void ts_ok(ss_status s,const char* what){if(s!=TS_OK)throw std::runtime_error(std::string("tierstream ")+what+": "+ts_last_error());}
// One call into the shared planner. Fixed parts: GLM's (by name) + the arena's
// ring/guard slots; with library_parts, also the device-gated context's own
// table/controls (device) and bounce/load ring (pinned) as charged by the library.
WholeDevicePlan run_plan_once(const WholeDeviceIn& w,bool library_parts,bool caps){
    const auto& t=w.tier;expert_need(t.unit_bytes>0&&t.keys>0&&t.max_ids>=1,"tier budget inputs");
    const std::uint64_t ring=2ULL*t.max_ids+1,unit=t.unit_bytes;WholeDevicePlan r;
    r.device=w.device;r.pinned=w.pinned;r.device.push_back({"tierstream ring/guard slots",ring*unit});
    if(library_parts){r.device.push_back({"tierstream unit table + controls",t.keys*4+(64ULL<<20)});r.pinned.push_back({"tierstream bounce + load ring",(5ULL+t.disk_readers)*align_up(unit,4096)+(80ULL<<20)});}
    auto comps=[](const std::vector<BudgetPart>& v){std::vector<ts_budget_component> c;c.reserve(v.size());for(auto& x:v)c.push_back({x.name.c_str(),x.bytes});return c;};
    auto d=comps(r.device),p=comps(r.pinned),g=comps(w.pageable);
    ts_budget_in in;ts_budget_in_defaults(&in);in.file_tier=t.file_tier?1u:0u;in.vram_total_bytes=w.vram_total;in.ram_total_bytes=caps?w.ram_total:0;in.device_context_bytes=w.context;
    in.device_fixed=d.data();in.n_device_fixed=std::uint32_t(d.size());in.pinned_fixed=p.data();in.n_pinned_fixed=std::uint32_t(p.size());in.pageable_fixed=g.data();in.n_pageable_fixed=std::uint32_t(g.size());
    in.host_spares=t.host_spares;in.keys=t.keys;in.slot_bytes=unit;in.unit_bytes=unit;in.arena_spares=t.spare_slots;in.staging=t.staging;in.min_resident=w.min_resident;in.vram_margin_bytes=w.vram_margin;in.ram_margin_bytes=w.ram_margin;
    in.arena_units=w.arena_units==~0ULL?TS_BUDGET_AUTO:w.arena_units;in.ram_expert_bytes=w.ram_expert_bytes==~0ULL?TS_BUDGET_AUTO:w.ram_expert_bytes;in.host_avail_bytes=w.host_avail==~0ULL?TS_BUDGET_AUTO:w.host_avail;
    ts_budget_out out{};out.struct_size=sizeof out;const auto s=ts_budget_plan(&in,&out);
    if(s!=TS_OK&&s!=TS_E_CAPACITY)ts_ok(s,"budget plan");
    r.device_fixed=out.device_fixed_bytes;r.device_planned=out.device_planned_bytes;r.host_fixed=out.host_fixed_bytes;r.host_planned=out.host_planned_bytes;
    r.min_vram_total=out.min_vram_total_bytes;r.min_ram_total=out.min_ram_total_bytes;if(out.refused)r.refuse=out.refuse;
    auto& b=r.tier;b.ring=ring;b.resident=out.resident;b.spare_slots=out.arena_spares;b.slots=out.resident+out.arena_spares;b.pinned=out.pinned;b.disk=out.disk;
    b.staging=out.staging;b.host_spares=t.host_spares;b.disk_readers=t.disk_readers;b.three_tier=out.three_tier!=0;
    // Device-gated library rules (not budget arithmetic): at least one global
    // spare slot, never more spares than non-resident units; no file tier means
    // every non-resident unit plus the spare blocks must be pinned.
    if(r.refuse.empty()&&t.spare_slots<1)r.refuse="at least one global spare slot is required";
    if(r.refuse.empty()&&!t.file_tier&&w.host_avail!=~0ULL&&b.pinned+t.host_spares>w.host_avail/unit)r.refuse="host budget cannot pin every non-resident unit and the file tier is off";
    if(r.refuse.empty()){const auto nonres=b.pinned+b.disk;if(b.spare_slots>nonres){b.spare_slots=nonres;b.slots=b.resident+b.spare_slots;}if(!b.spare_slots)r.refuse="every unit resident: no spare slot (lower --tier-vram-gib)";}
    b.refuse=r.refuse;return r;
}
// Spares as a share of the arena: one more planner call with the larger count
// (the arena is the same bytes; residents give way to spares). A refusal of the
// second call falls back to the first plan (the explicit minimum).
WholeDevicePlan run_plan(const WholeDeviceIn& w,bool library_parts,bool caps){
    auto first=run_plan_once(w,library_parts,caps);const auto& t=w.tier;
    if(!t.spare_permille||!first.refuse.empty())return first;
    const std::uint64_t arena=first.tier.resident+first.tier.spare_slots,want=(arena*t.spare_permille+999)/1000;
    if(want<=t.spare_slots)return first;
    auto w2=w;w2.tier.spare_slots=std::uint32_t(std::min<std::uint64_t>(want,0xFFFFFFFFULL));w2.tier.spare_permille=0;
    auto second=run_plan_once(w2,library_parts,caps);return second.refuse.empty()?second:first;
}
}
TierBudget tier_budget(const TierBudgetIn& in){
    // The arena budget is the device cap (min 1 byte: 0 would mean uncapped); the
    // pinned budget is the host memory available (no RAM cap: the arena-only rules).
    WholeDeviceIn w;w.tier=in;w.vram_total=std::max<std::uint64_t>(in.device_bytes,1);w.host_avail=in.host_bytes;
    return run_plan(w,false,false).tier;
}
WholeDevicePlan plan_whole_device(const WholeDeviceIn& in){return run_plan(in,true,true);}
bool budget_library_available(){return true;}
BudgetLedger::BudgetLedger(std::uint64_t vram_total,std::uint64_t ram_total){ts_budget_ledger* l=nullptr;ts_ok(ts_budget_ledger_create(vram_total,ram_total,&l),"budget ledger");l_=l;}
BudgetLedger::~BudgetLedger(){if(l_)ts_budget_ledger_free(static_cast<ts_budget_ledger*>(l_));}
bool BudgetLedger::add(MemKind kind,const std::string& name,std::uint64_t bytes,std::string* why){
    const auto s=ts_budget_ledger_add(static_cast<ts_budget_ledger*>(l_),ts_budget_kind(int(kind)),name.c_str(),bytes);
    if(s==TS_OK)return true;const std::string m=ts_last_error();if(why)*why=m;if(s!=TS_E_CAPACITY)throw std::runtime_error("budget ledger: "+m);return false;}
void BudgetLedger::sub(MemKind kind,const std::string& name,std::uint64_t bytes)noexcept{ts_budget_ledger_sub(static_cast<ts_budget_ledger*>(l_),ts_budget_kind(int(kind)),name.c_str(),bytes);}
void BudgetLedger::observe(const std::string& stage,std::uint64_t device_used,std::uint64_t host_used){
    ts_ok(ts_budget_ledger_observe(static_cast<ts_budget_ledger*>(l_),stage.c_str(),device_used==~0ULL?TS_BUDGET_NO_SAMPLE:device_used,host_used==~0ULL?TS_BUDGET_NO_SAMPLE:host_used),"budget observe");}
bool BudgetLedger::within_caps()const{return ts_budget_ledger_within_caps(static_cast<const ts_budget_ledger*>(l_))!=0;}
std::uint64_t BudgetLedger::current(MemKind kind)const{ts_budget_stats s{};s.struct_size=sizeof s;ts_ok(ts_budget_ledger_stats(static_cast<const ts_budget_ledger*>(l_),&s),"budget stats");return s.kind[int(kind)].current;}
Json BudgetLedger::stats()const{
    ts_budget_stats s{};s.struct_size=sizeof s;ts_ok(ts_budget_ledger_stats(static_cast<const ts_budget_ledger*>(l_),&s),"budget stats");
    auto j=Json::object();const char* names[]={"device","pinned","pageable"};
    for(int k=0;k<TS_BUDGET_KINDS;++k){auto& x=s.kind[k];auto o=Json::object();o.set("cap_bytes",x.cap);o.set("current_bytes",x.current);o.set("peak_bytes",x.peak);
        auto top=Json::array();for(std::uint32_t i=0;i<x.n_top&&i<TS_BUDGET_TOP;++i){auto e=Json::object();e.set("name",std::string(x.top[i].name));e.set("bytes",x.top[i].bytes);top.push(e);}o.set("top_at_peak",top);j.set(names[k],o);}
    j.set("host_current_bytes",s.host_current);j.set("host_peak_bytes",s.host_peak);j.set("measured_device_last_bytes",s.measured_device_last);j.set("measured_device_peak_bytes",s.measured_device_peak);
    j.set("observations",s.observations);j.set("refusals",int(s.refusals));
    j.set("measured_device_peak_at",std::string(s.measured_device_peak_at));j.set("measured_host_last_bytes",s.measured_host_last);j.set("measured_host_peak_bytes",s.measured_host_peak);
    j.set("measured_host_peak_at",std::string(s.measured_host_peak_at));j.set("device_samples_over_cap",int(s.device_over));j.set("host_samples_over_cap",int(s.host_over));
    j.set("within_caps",ts_budget_ledger_within_caps(static_cast<const ts_budget_ledger*>(l_))!=0);return j;
}
}
