#pragma once
// Replaceable routed-expert acquisition for GLM decode (dense/shared experts stay
// outside). Mirrors the DS tierstream TierSite lifecycle:
//   begin_token -> per routed layer: acquire(export ids, wait) ->
//   bind_selected_descriptors (guarded slot -> GLM Expert descriptors) ->
//   release (after final consumption) -> end_token -> collect_stats;
//   reset_cache only between tokens (quiescent).
// Product backend: the shared tierstream expert cache over GLM expert units
// (expert_source.hpp). Reference backend (exactness/test only): uncached
// per-expert reads from the existing device/pinned/file layer tiers.
#include "expert_source.hpp"
#include "head_select.hpp"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <sstream>
#include <vector>
#include <cstdio>
#include <memory>
#include <system_error>
#include <cstdint>
#include <string>
namespace strix::glm::rtxpf::producer {
inline constexpr unsigned routed_first_layer=3,routed_last_layer=45,routed_experts=288,routed_topk=8;
inline unsigned gated_ordinal(unsigned layer){expert_need(layer>=routed_first_layer&&layer<=routed_last_layer,"routed layer");return layer-routed_first_layer;}
class ExpertTier {
public:
    virtual ~ExpertTier()=default;
    virtual const char* backend()const=0;
    virtual void begin_token(unsigned position)=0;
    // ids_device: [count] router-slot-ordered IDs on the compute stream. On
    // return, kernels enqueued later on `stream` observe ready unit bytes.
    virtual void acquire(unsigned layer,const std::uint32_t* ids_device,unsigned count,void* stream)=0;
    // Device Expert[288] indexed by original expert ID; unselected entries are
    // poisoned (never dereferenced: zero-count experts are skipped).
    virtual const void* bind_selected_descriptors(unsigned layer,void* stream)=0;
    virtual void release(unsigned layer,void* stream)=0;
    // Throws when the token must be discarded (miss/abort/device timeout).
    virtual void end_token()=0;
    virtual Json collect_stats()=0;
    virtual void reset_cache(bool open)=0;
    // A failure after begin_token: retire ownership so the next token reports
    // this original failure instead of reusing a cache whose token never ended.
    virtual void fail_token(const std::string& why)=0;
    virtual std::string unusable()const{return {};}
};
// ---- tierstream device-gated view (GLM-side mirror; replaced by the shared
// header when it is available). One opaque part per unit: base + slot*stride.
// The cache's per-step export plan, the library's DgSteps rule (dg_steps.hpp): a step
// exports gated layer g when its rows name g (an all-layer token names every layer)
// and g holds no make_resident set. The service waits for a layer's prediction flag
// only when that layer's export promised ids AND the step exports the next layer,
// so GLM promises (and enqueues) a prediction for g+1 only when the step exports it:
// never for a layer the step skips (rows 0) or holds (a kept prefill set).
struct StepPlan {
    std::vector<std::uint8_t> held,run;
    void init(unsigned gated){held.assign(gated,0);run.assign(gated,0);}
    void hold(unsigned g){held.at(g)=1;}
    void release(unsigned g){held.at(g)=0;}
    void reset(){std::fill(held.begin(),held.end(),std::uint8_t(0));}
    void begin_all(){for(std::size_t g=0;g<run.size();++g)run[g]=std::uint8_t(!held[g]);}
    void begin_layer(unsigned g){for(std::size_t h=0;h<run.size();++h)run[h]=std::uint8_t(h==g&&!held[h]);}
    bool exports(unsigned g)const{return g<run.size()&&run[g]!=0;}
    // Predicted ids promised at g's export (n per layer through gated layer `through`, exclusive).
    unsigned promise(unsigned g,unsigned n,unsigned through)const{return n&&g<through&&exports(g)&&exports(g+1)?n:0;}
};
struct TierView {
    const std::int32_t* slot_of=nullptr;const std::uint32_t* unit_base=nullptr;const std::uint32_t* abort=nullptr;
    std::uint32_t* absent_err=nullptr;std::uint32_t guard_slot=0,n_gated=0,units=0;
    const void* base=nullptr;std::uint64_t stride=0;
};
// Slot census (diagnostics, read-only). The library's slots: persistent_slots (residents +
// spares) first, then two banks of max_ids alias slots, then the guard (dg_layout.cpp):
// persistent = guard_slot - 2 * max_ids. Per gated layer: units on a persistent slot, on a
// bank slot, absent; the decode miss count is the exported ids absent when the layer exported.
struct LayerCensus {std::uint32_t persistent=0,bank=0,absent=0;};
GLM_HOST_DEVICE inline LayerCensus census_layer(const std::int32_t* slot_of_layer,std::uint32_t units,std::uint32_t persistent_slots){
    LayerCensus c;for(std::uint32_t u=0;u<units;++u){const std::int32_t s=slot_of_layer[u];if(s<0)++c.absent;else if(std::uint32_t(s)<persistent_slots)++c.persistent;else ++c.bank;}return c;}
// misses[g] = 0xFFFFFFFF: the layer did not export this token.
inline Json census_json(const std::vector<LayerCensus>& layers,const std::vector<std::uint32_t>& misses,std::uint32_t persistent_slots){
    auto j=Json::object();auto a=Json::array();std::uint64_t used=0,bank=0,miss=0;
    for(std::size_t g=0;g<layers.size();++g){auto o=Json::object();o.set("gated",std::uint64_t(g));o.set("persistent",std::uint64_t(layers[g].persistent));o.set("bank",std::uint64_t(layers[g].bank));
        o.set("absent",std::uint64_t(layers[g].absent));if(g<misses.size()&&misses[g]!=0xFFFFFFFFu){o.set("misses",std::uint64_t(misses[g]));miss+=misses[g];}
        used+=layers[g].persistent;bank+=layers[g].bank;a.push(o);}
    j.set("persistent_slots",std::uint64_t(persistent_slots));j.set("persistent_used",used);j.set("persistent_free",used<=persistent_slots?persistent_slots-used:0);
    j.set("bank_used",bank);j.set("misses",miss);j.set("layers",a);return j;}
// Guarded lookup, exactly ss_dg_slot (ss.h, tierstream): abort -> guard; id beyond
// the gated layer's unit count (unit_base[g+1]-unit_base[g]) -> guard; absent
// entry (< 0) flags absent_err and returns the guard slot (token discarded).
GLM_HOST_DEVICE inline std::int32_t guarded_slot(const std::int32_t* slot_of,const std::uint32_t* unit_base,std::uint32_t abort,std::uint32_t guard,std::uint32_t g,std::uint32_t id,bool* absent){
    if(abort)return std::int32_t(guard);const auto b=unit_base[g];if(id>=unit_base[g+1]-b)return std::int32_t(guard);
    auto s=slot_of[b+id];if(s<0){*absent=true;return std::int32_t(guard);}return s;
}
// Device-gated calls of the shared tierstream cache (ts_begin, ts_dg_enqueue_*,
// ts_end, ts_dg_get_device_view, ts_dg_reset_open, ts_stats_dg). A concrete
// binding is supplied by the shared tierstream library; none is linked yet.
class TierCacheApi {
public:
    virtual ~TierCacheApi()=default;
    virtual void begin()=0;
    virtual void enqueue_begin(void* stream)=0;
    virtual void enqueue_export(unsigned gated,const std::int32_t* ids_device,unsigned n,void* stream)=0;
    virtual void enqueue_wait(unsigned gated,void* stream)=0;
    virtual void enqueue_end(void* stream)=0;
    virtual void end()=0; // throws: token refused, outputs discarded, context poisoned
    virtual TierView view()const=0;
    virtual void reset_open()=0;
    virtual Json stats()=0;
    // Raw cumulative counters (never reset): eviction / write-back / make_resident totals for phase deltas.
    virtual Json counters(){return Json::object();}
    // Best effort, for failure receipts: raw counters and the last token error (empty: none).
    virtual Json diagnostics(){return Json::object();}
    // Whether the open step exports gated layer g (StepPlan above); a cache without steps exports every layer.
    virtual bool step_exports(unsigned)const{return true;}
    // Between tokens: device-resident keys, most recent first (ss_dg_residency;
    // written at shutdown, it seeds the next open). Empty when unsupported.
    virtual std::vector<std::uint32_t> residency(){return {};}
    // Between tokens: lower the boundary spare target to n for a prefill (ss_dg_set_budgets; placement only), and put
    // the open-time target back. false = unsupported (the caller keeps today's behaviour).
    virtual bool lower_spare_target(unsigned){return false;}
    virtual void restore_spare_target(){}
    // Overlapped next-layer prefetch (ss_dg_enqueue_predict): with n > 0, every
    // export of a gated layer g < through promises n predicted ids of g + 1, and
    // enqueue_predict(g) hands them over on a side stream joined before g+1's export.
    virtual void set_predict_ids(unsigned n,unsigned through){(void)through;if(n)throw std::runtime_error("this cache has no predicted prefetch");}
    virtual void enqueue_predict(unsigned,const std::int32_t*,unsigned,void*){throw std::runtime_error("this cache has no predicted prefetch");}
    // Prefill through the cache (one copy of every expert): the tierstream
    // reader/resident contract (absent from older tierstream versions).
    // make_resident (ss_dg_make_resident), between tokens: units ids[0..n) of
    // gated layer g become device-resident and KEPT (they resolve through the slot
    // table in any later step, exported or not) until release_resident(g)
    // (ss_dg_keep(g,NULL,0)); returns false on SS_E_CAPACITY (nothing changed:
    // the caller batches). begin_layer_step(g): ss_begin of a step whose only
    // gated layer is g (rows_per_layer[h]=0 for h != g); then enqueue_begin, the
    // readers, enqueue_end and end() as for a decode token.
    virtual bool supports_make_resident()const{return false;}
    virtual bool make_resident(unsigned,const std::int32_t*,unsigned){throw std::runtime_error("this tierstream library has no make_resident (reader/resident requested)");}
    virtual void release_resident(unsigned){throw std::runtime_error("this tierstream library has no make_resident (reader/resident requested)");}
    virtual void begin_layer_step(unsigned){throw std::runtime_error("this tierstream library has no single-layer steps (reader/resident requested)");}
};
// Startup budget -> tier counts. ONE planner: tierstream's ss_budget_plan
// ("Whole-device budgets"), called only from
// tierstream_binding.cpp (the one TU that sees tierstream types). GLM feeds it
// component sizes and owns no budget arithmetic of its own. The device arena
// holds residents + spares + the 2*max_ids+1 ring/guard slots (a named part).
struct TierBudgetIn {
    std::uint64_t keys=43ULL*288,unit_bytes=0;   // units and their (64 KiB) stride
    std::uint64_t device_bytes=0,host_bytes=0;   // arena / pinned budgets already net of margins
    std::uint32_t max_ids=routed_topk,spare_slots=16,host_spares=8,staging=8;
    // Spares as a share of the arena (resident + spares): the planner is asked again with
    // max(spare_slots, ceil(permille * arena / 1000)) spares; 0 = spare_slots exactly.
    std::uint32_t spare_permille=0;
    // The library's per-reader bounce buffers beyond its own one: with dg_disk_queue > 1 each
    // disk reader holds one (up to a unit's covering read), billed to pinned host memory.
    std::uint32_t disk_readers=0;
    // The library's landing blocks (dg_landing_blocks): pinned, one host stride each, outside the
    // pinned tier; billed whenever requested (the library allocates them only with a file tier).
    std::uint32_t landing_blocks=0;
    bool file_tier=true;
};
struct TierBudget {
    std::uint64_t slots=0,resident=0,pinned=0,disk=0,spare_slots=0,host_spares=0,staging=0,ring=0,disk_readers=0,landing_blocks=0;
    bool three_tier=false;std::string refuse;
    std::uint64_t device_bytes(std::uint64_t unit)const{return slots*unit;}
    std::uint64_t host_bytes(std::uint64_t unit)const{return (pinned+host_spares+staging+disk_readers+landing_blocks)*unit;}
};
// Arena-only budgets (--tier-vram-gib / --tier-ram-gib, and the default sizing
// from free memory): ss_budget_plan with the arena budget as its device cap (the
// ring the only fixed part) and the pinned budget as the host memory available;
// then the device-gated library's spare-slot rules (dg_spare_slots >= 1 and
// <= the non-resident units). Results equal the earlier GLM rules (tested).
TierBudget tier_budget(const TierBudgetIn& in);
// Whole-device budgets (--vram-total-gib / --ram-total-gib): GLM's named fixed
// parts in, the expert tiers get what is left (ss_budget_plan), with the smallest
// caps that would hold the fixed parts.
struct BudgetPart {std::string name;std::uint64_t bytes=0;};
// Host memory that grows after the cache opens and the measured process commit misses:
// the decode bench's session snapshot (a host copy of the session state + replay, held
// across its arms) and the CUDA driver's later host allocations (graph instantiation,
// module data, per-request buffers). The 24/32 8K bench peaked 1.10 GB over the plan at
// open: 0.49 GB snapshot + 0.61 GB driver/other, so the headroom is 768 MiB.
inline constexpr const char* session_snapshot_part="decode session snapshot (host copy of session state + replay, bench restores)";
inline constexpr const char* host_growth_part="host growth after open (CUDA driver / graph instantiation / request buffers, headroom)";
inline constexpr std::uint64_t host_growth_headroom=768ULL<<20;
struct BudgetPart;
inline void add_host_growth_parts(std::vector<BudgetPart>& pageable,std::uint64_t session_snapshot_bytes);
struct WholeDeviceIn {
    std::uint64_t vram_total=0,ram_total=0,context=0;        // caps (0 = uncapped), measured device context
    std::uint64_t vram_margin=0,ram_margin=0;                 // kept free under each cap (graphs, warm-up, allocator slack)
    std::vector<BudgetPart> device,pinned,pageable;           // GLM fixed parts by name
    std::uint64_t host_avail=~0ULL;                           // host memory available now (~0: not limited)
    std::uint64_t arena_units=~0ULL,ram_expert_bytes=~0ULL;   // arena-only requests (~0: what the caps allow)
    std::uint64_t min_resident=1;                             // smallest arena accepted (prefill through the cache: 288 + spares)
    TierBudgetIn tier;                                        // keys, unit stride, spares, staging, ring
};
inline void add_host_growth_parts(std::vector<BudgetPart>& pageable,std::uint64_t session_snapshot_bytes){pageable.push_back({session_snapshot_part,session_snapshot_bytes});pageable.push_back({host_growth_part,host_growth_headroom});}
struct WholeDevicePlan {
    TierBudget tier;std::vector<BudgetPart> device,pinned;    // the parts as planned (library parts added)
    std::uint64_t device_fixed=0,device_planned=0,host_fixed=0,host_planned=0,min_vram_total=0,min_ram_total=0;
    std::string refuse;
};
WholeDevicePlan plan_whole_device(const WholeDeviceIn& in);
inline std::string gib_text(std::uint64_t b){char s[32];std::snprintf(s,sizeof s,"%.2f GiB",double(b)/double(1ULL<<30));return s;}
inline std::uint64_t gib_bytes(double gib){return gib<0?0:std::uint64_t(gib*double(1ULL<<30));}
inline Json whole_device_json(const WholeDeviceIn& in,const WholeDevicePlan& p){
    auto parts=[](const std::vector<BudgetPart>& v){auto a=Json::array();for(auto& x:v){auto o=Json::object();o.set("name",x.name);o.set("bytes",x.bytes);o.set("gib",double(x.bytes)/double(1ULL<<30));a.push(o);}return a;};
    auto j=Json::object();j.set("vram_total_bytes",in.vram_total);j.set("ram_total_bytes",in.ram_total);j.set("device_context_bytes",in.context);
    j.set("vram_margin_bytes",in.vram_margin);j.set("ram_margin_bytes",in.ram_margin);
    j.set("device_parts",parts(p.device));j.set("pinned_parts",parts(p.pinned));j.set("pageable_parts",parts(in.pageable));
    j.set("device_fixed_bytes",p.device_fixed);j.set("device_planned_bytes",p.device_planned);j.set("host_fixed_bytes",p.host_fixed);j.set("host_planned_bytes",p.host_planned);
    j.set("min_vram_total_bytes",p.min_vram_total);j.set("min_ram_total_bytes",p.min_ram_total);j.set("unit_stride",in.tier.unit_bytes);
    j.set("arena_slots",p.tier.slots+p.tier.ring);j.set("arena_bytes",(p.tier.slots+p.tier.ring)*in.tier.unit_bytes);j.set("resident_units",p.tier.resident);j.set("spare_slots",p.tier.spare_slots);
    j.set("pinned_units",p.tier.pinned);j.set("file_units",p.tier.disk);j.set("three_tier",p.tier.three_tier);
    j.set("summary","VRAM: fixed "+gib_text(p.device_fixed)+" (context "+gib_text(in.context)+") + margin "+gib_text(in.vram_margin)+" + expert arena "+gib_text(p.device_planned>p.device_fixed?p.device_planned-p.device_fixed:0)+
        " = "+gib_text(p.device_planned)+(in.vram_total?" of "+gib_text(in.vram_total):std::string())+"; RAM: fixed "+gib_text(p.host_fixed)+" + pinned experts "+gib_text(p.host_planned>p.host_fixed?p.host_planned-p.host_fixed:0)+
        " = "+gib_text(p.host_planned)+(in.ram_total?" of "+gib_text(in.ram_total):std::string())+"; units resident/pinned/file "+std::to_string(p.tier.resident)+"/"+std::to_string(p.tier.pinned)+"/"+std::to_string(p.tier.disk)+
        "; minimum --vram-total-gib "+std::to_string((p.min_vram_total+(1ULL<<30)-1)>>30)+", --ram-total-gib "+std::to_string((p.min_ram_total+(1ULL<<30)-1)>>30));
    j.set("verdict",p.refuse.empty()?"ADMITTED":"REFUSED");if(!p.refuse.empty())j.set("refuse",p.refuse);return j;
}
// Allocation ledger: tierstream's ss_budget_ledger behind an opaque handle.
// Every device and pinned allocation is charged by component name against the
// caps (an over-cap charge fails, naming it); measured device samples and the
// peaks per kind go into receipts.
enum class MemKind {device=0,pinned=1,pageable=2};
class BudgetLedger {
    void* l_=nullptr;
public:
    BudgetLedger(std::uint64_t vram_total,std::uint64_t ram_total);
    ~BudgetLedger();BudgetLedger(const BudgetLedger&)=delete;BudgetLedger& operator=(const BudgetLedger&)=delete;
    bool add(MemKind kind,const std::string& name,std::uint64_t bytes,std::string* why=nullptr);
    void sub(MemKind kind,const std::string& name,std::uint64_t bytes)noexcept;
    // A measured sample at a named stage ("startup", "cache open", "request:OP"):
    // device use (~0: not measured) and host use (host_commit(): process commit
    // minus our WDDM device backing, pinned included).
    void observe(const std::string& stage,std::uint64_t device_used,std::uint64_t host_used);
    bool within_caps()const;
    std::uint64_t current(MemKind kind)const;
    Json stats()const;
    void* raw()const{return l_;}   // ss_budget_ledger* for ss_options.budget
};
bool budget_library_available();
// Host commit of OUR process under WDDM: every device allocation (cudaMalloc,
// the library's arena) is also charged to the process commit (PrivateUsage) as
// its eviction backing, without being resident in RAM. Host commit = private
// bytes minus our device bytes (ledger device total minus the measured context
// + other processes part, which is never ours to subtract). Our CUDA context's
// own commit, if any, stays on the host side (conservative).
struct HostCommit {std::uint64_t raw_private=0,device_backing=0,host=0;};
inline HostCommit host_commit(std::uint64_t raw_private,std::uint64_t ledger_device,std::uint64_t context_part){
    HostCommit c;c.raw_private=raw_private;c.device_backing=ledger_device>context_part?ledger_device-context_part:0;
    c.host=raw_private>c.device_backing?raw_private-c.device_backing:0;return c;
}
// Gate peak of our host commit: the sampled peak (every observation) or the OS
// peak commit minus the peak attributable device backing, whichever is larger
// (the OS term catches untracked temporaries that came and went between samples).
inline std::uint64_t host_gate_peak(std::uint64_t sampled,std::uint64_t os_peak_commit,std::uint64_t backing_peak){
    return std::max(sampled,os_peak_commit>backing_peak?os_peak_commit-backing_peak:0);
}
// Host peak history for receipts: `sampled` only ever takes measured host
// samples (ledger observations, the current sample); `gate` also takes the OS
// bound. Kept apart so host_peak_sampled_bytes never carries an OS-derived peak.
struct HostPeakHistory {
    std::uint64_t sampled=0,gate=0;bool known=false;
    void add_sample(std::uint64_t host){sampled=std::max(sampled,host);gate=std::max(gate,host);known=true;}
    void add_os_bound(std::uint64_t os_peak_commit,std::uint64_t backing_peak){gate=std::max(gate,host_gate_peak(0,os_peak_commit,backing_peak));known=true;}
};
// Per-key tier (TS_TIER_RESIDENT=0 / PINNED=1 / DISK=2, ss_tier) by priority:
// the first `resident` keys of `order` device, the next `pinned` RAM, rest file.
inline std::vector<std::uint8_t> tier_assignment(const TierBudget& b,std::uint64_t keys,const std::vector<std::uint32_t>& order){
    std::vector<std::uint8_t> t(keys,2);std::vector<bool> seen(keys);std::uint64_t r=0,p=0;
    auto place=[&](std::uint64_t k){if(k>=keys||seen[k])return;seen[k]=true;if(r<b.resident){t[k]=0;++r;}else if(p<b.pinned){t[k]=1;++p;}};
    for(auto k:order)place(k);for(std::uint64_t k=0;k<keys;++k)place(k);
    expert_need(r==b.resident&&p==b.pinned,"tier assignment counts");return t;
}
// ---- Placement, the DeepSeek way (the DS replica seed_order,
// residency_keys; src/cuda/ds41_tier_site.cu): one priority list of keys
// (key = gated ordinal * 288 + expert) gives both the tier array (first `resident`
// keys on the device) and ss_options.dg_seed (which units take the device slots,
// and the initial recency order). Order: the residency file of the previous run
// (ss_dg_residency at shutdown, most recent first), then experts by routing count
// from a routes census, then the no-profile fallback: expert-major round-robin
// across ALL layers (expert 0 of layers 3..45, expert 1 of layers 3..45, ...),
// so every layer holds the same share of the device instead of layer order.
inline constexpr unsigned routed_gated=routed_last_layer-routed_first_layer+1;
// Arena spares a prefill needs so that one gated layer's make_resident(288) fits in free slots
// instead of evicting other layers' decode residents: 288 minus that layer's residents (the
// placement spreads the residents round-robin: at least resident_units / 43 per layer).
inline unsigned prefill_spares_needed(std::uint64_t resident_units){
    const auto per=resident_units/routed_gated;return per>=routed_experts?0u:unsigned(routed_experts-per);}
inline std::vector<std::uint32_t> round_robin_order(unsigned gated=routed_gated,unsigned units=routed_experts){
    std::vector<std::uint32_t> o;o.reserve(std::size_t(gated)*units);
    for(unsigned u=0;u<units;++u)for(unsigned g=0;g<gated;++g)o.push_back(g*units+u);return o;
}
// Residency file, DS layout ("ds41-residency-v1 sha256=<model> layers=L experts=E n=N",
// then keys, 32 per line); GLM's tag is glm-residency-v1.
inline std::string residency_text(const std::string& model_sha,const std::vector<std::uint32_t>& keys,unsigned gated=routed_gated,unsigned units=routed_experts){
    std::string s="glm-residency-v1 sha256="+model_sha+" layers="+std::to_string(gated)+" experts="+std::to_string(units)+" n="+std::to_string(keys.size())+"\n";
    for(std::size_t i=0;i<keys.size();++i){s+=std::to_string(keys[i]);s+=(i%32==31||i+1==keys.size())?"\n":" ";}return s;
}
// Throws on a foreign header (other model, geometry, tag) or a bad key; the
// caller falls back without it (placement changes timing only, never outputs).
inline std::vector<std::uint32_t> parse_residency(const std::string& text,const std::string& model_sha,unsigned gated=routed_gated,unsigned units=routed_experts){
    std::istringstream in(text);std::string tag,sha,layers,experts,n;in>>tag>>sha>>layers>>experts>>n;
    expert_need(tag=="glm-residency-v1"&&sha=="sha256="+model_sha&&layers=="layers="+std::to_string(gated)&&experts=="experts="+std::to_string(units)&&n.rfind("n=",0)==0,"residency header does not match this model");
    const unsigned long long count=std::stoull(n.substr(2));expert_need(count<=std::uint64_t(gated)*units,"residency count");
    std::vector<std::uint32_t> keys;std::vector<bool> seen(std::size_t(gated)*units);unsigned long long k=0;
    while(in>>k){expert_need(k<seen.size()&&!seen[k],"residency key out of range or repeated");seen[k]=true;keys.push_back(std::uint32_t(k));}
    expert_need(keys.size()==count,"residency key count");return keys;
}
// Routes census, DS routes layout (deepseek41_rank_main --routes): "# seq NAME rows N"
// headers, then one line per token: "layer:id,id,... layer:id,..." with MODEL layers.
inline std::string routes_line(const std::uint32_t* ids,unsigned gated=routed_gated,unsigned topk=routed_topk){
    std::string s;for(unsigned g=0;g<gated;++g){if(g)s+=' ';s+=std::to_string(g+routed_first_layer)+':';for(unsigned k=0;k<topk;++k){if(k)s+=',';s+=std::to_string(ids[g*topk+k]);}}return s;
}
// A routes line of only the layers that were recorded: a layer whose first id is
// ~0 (unset, e.g. the MTP layer in a prefill census) is left out.
inline std::string routes_line_present(const std::uint32_t* ids,unsigned gated=routed_gated,unsigned topk=routed_topk){
    std::string s;for(unsigned g=0;g<gated;++g){if(ids[g*topk]==~std::uint32_t(0))continue;if(!s.empty())s+=' ';s+=std::to_string(g+routed_first_layer)+':';
        for(unsigned k=0;k<topk;++k){if(k)s+=',';s+=std::to_string(ids[g*topk+k]);}}return s;
}
// --weight-dirs: ';'-separated stripe directories. No empty entry and no directory twice (compared
// case- and separator-insensitively, trailing separators ignored). tierstream itself binds each
// stripe manifest to the primary's SHA-256 and ignores the primary's own directory.
inline void check_weight_dirs(const std::string& v){
    expert_need(!v.empty(),"--weight-dirs: ';'-separated stripe directories");std::vector<std::string> seen;std::size_t pos=0;
    for(;;){const auto end=v.find(';',pos);auto d=v.substr(pos,end==std::string::npos?std::string::npos:end-pos);
        std::string k;for(char ch:d)k+=ch=='\\'?'/':char(std::tolower(static_cast<unsigned char>(ch)));while(k.size()>1&&k.back()=='/')k.pop_back();
        expert_need(!k.empty()&&k.find_first_not_of(' ')!=std::string::npos,"--weight-dirs: empty directory entry");
        for(auto& s:seen)expert_need(s!=k,"--weight-dirs: directory listed twice");seen.push_back(k);
        if(end==std::string::npos)break;pos=end+1;}
}
// Adds every routed (layer, expert) of the census to counts[key]; returns tokens counted.
inline std::uint64_t count_routes(const std::string& text,std::vector<std::uint64_t>& counts,unsigned gated=routed_gated,unsigned units=routed_experts){
    counts.resize(std::size_t(gated)*units);std::istringstream in(text);std::string line;std::uint64_t tokens=0;
    while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;std::istringstream f(line);std::string item;bool any=false;
        while(f>>item){const auto colon=item.find(':');expert_need(colon!=std::string::npos,"routes: layer:ids");const unsigned long layer=std::stoul(item.substr(0,colon));
            expert_need(layer>=routed_first_layer&&layer<routed_first_layer+gated,"routes: routed layer");std::istringstream ids(item.substr(colon+1));std::string id;
            while(std::getline(ids,id,',')){const unsigned long e=std::stoul(id);expert_need(e<units,"routes: expert id");++counts[(layer-routed_first_layer)*units+e];any=true;}}
        tokens+=any;}
    return tokens;
}
// A routed layer the census barely saw (the MTP layer of a prefill census: only the first
// decode token of each prompt routes it) would sort after every observed key and get no
// device slot. A layer sampled less than 1/16 as often as the best-sampled one gets the
// observed layers' mean per-unit count instead (as if uniform), so it interleaves with them.
// Returns the number of layers filled. Placement only.
inline unsigned fill_unobserved_layers(std::vector<std::uint64_t>& counts,unsigned gated=routed_gated,unsigned units=routed_experts){
    if(counts.size()!=std::size_t(gated)*units)return 0;
    std::vector<std::uint64_t> sum(gated,0);std::uint64_t most=0;
    for(unsigned g=0;g<gated;++g){for(unsigned u=0;u<units;++u)sum[g]+=counts[std::size_t(g)*units+u];most=std::max(most,sum[g]);}
    if(!most)return 0;
    std::uint64_t total=0;unsigned seen=0;for(unsigned g=0;g<gated;++g)if(sum[g]*16>=most){total+=sum[g];++seen;}
    if(seen==gated)return 0;
    const std::uint64_t mean=std::max<std::uint64_t>(1,(total+std::uint64_t(seen)*units/2)/(std::uint64_t(seen)*units));unsigned filled=0;
    for(unsigned g=0;g<gated;++g)if(sum[g]*16<most){for(unsigned u=0;u<units;++u)counts[std::size_t(g)*units+u]=mean;++filled;}
    return filled;
}
// Priority list: residency keys first, then by count (most routed first; ties
// keep round-robin order), then the rest in round-robin order. Every key once.
inline std::vector<std::uint32_t> seed_order(const std::vector<std::uint32_t>& residency,const std::vector<std::uint64_t>& counts,unsigned gated=routed_gated,unsigned units=routed_experts){
    const std::size_t keys=std::size_t(gated)*units;std::vector<bool> seen(keys);std::vector<std::uint32_t> o;o.reserve(keys);
    for(auto k:residency)if(k<keys&&!seen[k]){seen[k]=true;o.push_back(k);}
    auto rr=round_robin_order(gated,units);
    if(counts.size()==keys){std::vector<std::uint32_t> by=rr;std::stable_sort(by.begin(),by.end(),[&](std::uint32_t a,std::uint32_t b){return counts[a]>counts[b];});
        for(auto k:by)if(counts[k]&&!seen[k]){seen[k]=true;o.push_back(k);}}
    for(auto k:rr)if(!seen[k]){seen[k]=true;o.push_back(k);}
    return o;
}
// Per-(layer,expert) canonical unit offsets uploaded once: trellis,suh,svh per matrix.
struct UnitOffsets {std::uint32_t off[3][3];std::uint32_t rate[3];};
// Address of (matrix m, role r) of the unit held in arena slot `slot` (bind kernel and its host model).
GLM_HOST_DEVICE inline std::uint64_t unit_address(std::uint64_t base,std::int32_t slot,std::uint64_t stride,const UnitOffsets& u,unsigned m,unsigned r){return base+std::uint64_t(slot)*stride+u.off[m][r];}
// Library counters (ss_dg_stats subset, tierstream dg_core semantics): misses are
// served by resurrecting an evicting slot (no copy), by an in-flight copy
// (covered), or by a demand copy (fetched); prefetched are speculative copies.
// Every copy has one source: a pinned RAM block or a file staging buffer filled
// by exactly one file read (disk_reads), so RAM copies = copies - file reads.
struct CacheCounters {std::uint64_t lookups=0,misses=0,fetched=0,prefetched=0,resurrected=0,covered=0,bank_serves=0,file_reads=0,file_bytes=0,file_ns=0,evicted=0,drops=0,writebacks=0,stalled=0,timeouts=0,free_slots=0,file_wait_ns=0,clean_drops=0,clean_drop_bytes=0,
    landing_issued=0,landing_promoted=0,landing_late=0,landing_wait_ns=0,landing_expired=0,landing_dropped_full=0,landing_read_bytes=0,landing_cancelled=0,landing_expired_layer=0,landing_skipped_passed=0,landing_yields=0,landing_aborted_bytes=0;};
inline Json cache_delta(const CacheCounters& a,const CacheCounters& b,std::uint64_t unit_bytes){
    auto d=[](std::uint64_t x,std::uint64_t y){expert_need(y>=x,"cache counters went backwards");return y-x;};
    const auto look=d(a.lookups,b.lookups),miss=d(a.misses,b.misses),fetch=d(a.fetched,b.fetched),pre=d(a.prefetched,b.prefetched),file=d(a.file_reads,b.file_reads);
    expert_need(miss<=look&&file<=fetch+pre,"cache counters inconsistent (misses > lookups or file reads > copies)");expert_need(b.timeouts==a.timeouts,"tierstream device timeout");
    auto j=Json::object();j.set("backend","tierstream");j.set("lookups",look);j.set("misses",miss);j.set("device_hits",look-miss);
    j.set("resurrected",d(a.resurrected,b.resurrected));j.set("covered",d(a.covered,b.covered));j.set("fetched",fetch);j.set("prefetched",pre);j.set("bank_serves",d(a.bank_serves,b.bank_serves));
    j.set("ram_copies",fetch+pre-file);j.set("file_reads",file);j.set("file_bytes",d(a.file_bytes,b.file_bytes));j.set("file_read_ns",d(a.file_ns,b.file_ns));j.set("h2d_bytes",(fetch+pre)*unit_bytes);
    j.set("evicted",d(a.evicted,b.evicted));j.set("disk_drops",d(a.drops,b.drops));j.set("writebacks",d(a.writebacks,b.writebacks));j.set("stalled",d(a.stalled,b.stalled));j.set("free_slots",b.free_slots);
    // The service's wall time waiting on file reads (< file_read_ns once reads overlap) and clean drops (no D2H).
    j.set("file_wait_ns",d(a.file_wait_ns,b.file_wait_ns));j.set("clean_drops",d(a.clean_drops,b.clean_drops));j.set("clean_drop_bytes",d(a.clean_drop_bytes,b.clean_drop_bytes));
    // Two-stage file prefetch (landing blocks): promoted = demand misses served from a landing block.
    j.set("landing_issued",d(a.landing_issued,b.landing_issued));j.set("landing_promoted",d(a.landing_promoted,b.landing_promoted));j.set("landing_late",d(a.landing_late,b.landing_late));j.set("landing_wait_ns",d(a.landing_wait_ns,b.landing_wait_ns));
    j.set("landing_expired",d(a.landing_expired,b.landing_expired));j.set("landing_dropped_full",d(a.landing_dropped_full,b.landing_dropped_full));j.set("landing_read_bytes",d(a.landing_read_bytes,b.landing_read_bytes));j.set("landing_cancelled",d(a.landing_cancelled,b.landing_cancelled));
    j.set("landing_expired_layer",d(a.landing_expired_layer,b.landing_expired_layer));j.set("landing_skipped_passed",d(a.landing_skipped_passed,b.landing_skipped_passed));
    // Chunked landing reads (dg_landing_chunk): chunk boundaries that yielded to a demand read, and bytes of cancelled landings not read.
    j.set("landing_yields",d(a.landing_yields,b.landing_yields));j.set("landing_aborted_bytes",d(a.landing_aborted_bytes,b.landing_aborted_bytes));return j;
}
// Cache token ownership. A failure after begin leaves the library token without
// its end kernel, so the cache is never reused: every later begin reports the
// ORIGINAL failure (the worker must be restarted), never "already open".
class TokenGuard {
    bool open_=false;std::string failed_;
public:
    void begin(){if(!failed_.empty())throw std::runtime_error("expert cache unusable after a failed token (restart the worker): "+failed_);expert_need(!open_,"expert cache token already open");open_=true;}
    void end(){expert_need(open_&&failed_.empty(),"no expert cache token open");open_=false;}
    void fail(const std::string& why){if(open_&&failed_.empty())failed_=why.empty()?"unknown failure":why;open_=false;}
    bool open()const{return open_;}
    const std::string& failure()const{return failed_;}
};
// Host side of one cache token over TierCacheApi. The guard stays open until the
// library accepted the token (api.end returned); any failure in between,
// including api.end itself, is retired with its original error by fail().
class CacheTokenSite {
    TierCacheApi& api_;TokenGuard guard_;
public:
    explicit CacheTokenSite(TierCacheApi& api):api_(api){}
    void begin(){guard_.begin();try{api_.begin();}catch(const std::exception& e){guard_.fail(e.what());throw;}}
    void end(){expert_need(guard_.open(),"no expert cache token open");api_.end();guard_.end();}
    void fail(const std::string& why){guard_.fail(why);}
    void reset_open(){expert_need(!guard_.open()&&guard_.failure().empty(),"expert cache reset only between accepted tokens");api_.reset_open();}
    std::string unusable()const{return guard_.failure();}
    bool open()const{return guard_.open();}
    // Prefill residency, only between accepted tokens. ANY unsuccessful call,
    // a capacity refusal included, retires the worker irreversibly: later
    // requests report the original error (the request it served is discarded).
    // As make_resident, but SS_E_CAPACITY returns false WITHOUT retiring (nothing changed;
    // the residency restore after a prefill stops there); any other failure retires.
    bool try_make_resident(unsigned g,const std::int32_t* ids,unsigned n){
        between_tokens("residency restore only between accepted tokens");
        if(!api_.supports_make_resident())return false;
        try{return api_.make_resident(g,ids,n);}catch(const std::exception& e){retire(e.what());throw;}catch(...){retire("make_resident: non-standard exception");throw;}}
    void make_resident(unsigned g,const std::int32_t* ids,unsigned n){
        between_tokens("prefill residency only between accepted tokens");
        if(!api_.supports_make_resident()){const std::string m="this tierstream library has no make_resident (reader/resident requested)";retire(m);throw std::runtime_error(m);}
        bool ok=false;try{ok=api_.make_resident(g,ids,n);}catch(const std::exception& e){retire(e.what());throw;}catch(...){retire("make_resident: non-standard exception");throw;}
        if(!ok){const std::string m="prefill: the tierstream arena cannot hold gated layer "+std::to_string(g)+"'s "+std::to_string(n)+" units (SS_E_CAPACITY; expert batches not wired)";retire(m);throw std::runtime_error(m);}}
    // Irreversible: a retired site reports its first error to every later request.
    void retire(const std::string& why){if(!guard_.failure().empty())return;if(!guard_.open())guard_.begin();guard_.fail(why);}
    // A precondition failure (an open token, an earlier failure) retires the worker
    // with that error; an earlier failure is reported as the original error.
    void between_tokens(const char* what){
        if(!guard_.failure().empty())throw std::runtime_error(guard_.failure());
        if(guard_.open()){retire(what);throw std::runtime_error(what);}}
    void release_resident(unsigned g){
        between_tokens("prefill release only between accepted tokens");
        try{api_.release_resident(g);}catch(const std::exception& e){retire(e.what());throw;}catch(...){retire("release_resident: non-standard exception");throw;}}
    // One step whose only gated layer is g (the prefill pass of that layer).
    void begin_layer(unsigned g){between_tokens("prefill step only between accepted tokens");guard_.begin();try{api_.begin_layer_step(g);}catch(const std::exception& e){guard_.fail(e.what());throw;}catch(...){guard_.fail("begin_layer_step: non-standard exception");throw;}}
};
// Prefill through the cache, GLM side. Per routed layer (gated ordinal g):
// make_resident(g, 0..287) between tokens; one step with only g (begin_layer,
// enqueue_begin); bind the 288 descriptors from the slot table (bind_units,
// ss_dg_slot rules) into the layer's descriptor array; the unchanged CUTLASS
// prefill over every row chunk; enqueue_end, end; release_resident(g). The
// prefill reads the same unit bytes the decode reads (one copy of every expert).
// A capacity refusal would need expert batches (masked dispatch): refused here.
// Prefill startup work by mode: the interim/legacy path allocates the two layer
// rings, admits resident/pinned expert layers and builds immutable routed
// descriptors at startup (resident address or ring parity); prefill through the
// cache does none of these (descriptors are zeroed, then bound per layer).
struct PrefillInit {bool rings=true,admit_layers=true,startup_descriptors=true;};
inline PrefillInit prefill_init(bool via_cache){return via_cache?PrefillInit{false,false,false}:PrefillInit{};}
// The decode placement planned at open (the tier array's residents, per gated layer, in
// ascending unit order): restored after every prefill through the cache, whose whole-layer
// make_resident sets otherwise stay in the persistent slots (ordinary residents once
// released) and leave the decode residency to the last prefilled layers. Restore = one held
// make_resident set per layer in order (each set protects the previous ones; the planned
// residents + the spare target fit the persistent slots by the plan), then every set released.
inline std::vector<std::vector<std::int32_t>> planned_residents(const std::vector<std::uint8_t>& tiers,std::uint8_t resident_tier,unsigned gated,unsigned units){
    std::vector<std::vector<std::int32_t>> p(gated);
    for(std::size_t k=0;k<tiers.size()&&k<std::size_t(gated)*units;++k)if(tiers[k]==resident_tier)p[k/units].push_back(std::int32_t(k%units));
    return p;}
struct PrefillResidency {
    static constexpr unsigned units=288;
    std::vector<std::int32_t> ids;
    PrefillResidency():ids(units){for(unsigned e=0;e<units;++e)ids[e]=std::int32_t(e);}
    // Library requirement (dg_layout / dg_core): n + kept units + spare_target <=
    // persistent slots (residents + billed spares; ring/bank slots and the guard
    // are never kept): with nothing else kept, slots >= 288 + spare slots.
    // Planner floor: 288 residents (the planner adds the arena spares itself).
    static std::uint64_t min_resident(){return units;}
    static std::string refusal(const TierBudget& b){
        return b.slots>=units+b.spare_slots?std::string():"tierstream arena of "+std::to_string(b.slots)+" persistent slots cannot keep one layer's "+std::to_string(units)+" units + "+std::to_string(b.spare_slots)+" spares for prefill through the cache (expert batches not wired)";}
    // Prefill overlap (PrefillOverlap) keeps two layers at once: slots >= 2 * 288 + spare slots.
    static bool two_layers(const TierBudget& b){return b.slots>=2ULL*units+b.spare_slots;}
    // Between tokens: make layer g resident; any refusal retires the worker (CacheTokenSite).
    void acquire(CacheTokenSite& site,unsigned g)const{site.make_resident(g,ids.data(),units);}
    // As acquire, but SS_E_CAPACITY returns false (nothing changed); any other failure retires.
    bool try_acquire(CacheTokenSite& site,unsigned g)const{return site.try_make_resident(g,ids.data(),units);}
};
// Prefill overlap's spare target (placement only). Two held layers need slots >= 2 x 288 + the spare target; when only
// the boundary spares stand in the way (24 GiB: 606 slots = 575 residents + 31 spares) the prefill lowers the target to
// slots - 576 for its duration and puts the open-time target back at its end or abort.
class PrefillSpareTarget {
    unsigned lowered_=0;
public:
    unsigned lowered()const{return lowered_;}
    // true = lowered (overlap on). A cache without the call keeps overlap off; so does a throwing call (why = its error),
    // after the open-time target is set again (a failing reset retires the worker and throws).
    bool lower(TierCacheApi& api,CacheTokenSite& site,const TierBudget& b,bool want,std::string& why){
        why.clear();
        if(!want||lowered_||PrefillResidency::two_layers(b)||b.slots<2ULL*PrefillResidency::units+1)return false;
        const unsigned n=unsigned(b.slots-2ULL*PrefillResidency::units);
        try{if(!api.lower_spare_target(n))return false;}
        catch(const std::exception& e){why=e.what();}catch(...){why="non-standard exception";}
        if(!why.empty()){try{api.restore_spare_target();}catch(...){const std::string m="spare target reset after a failed lowering ("+why+")";site.retire(m);throw std::runtime_error(m);}
            return false;}
        lowered_=n;return true;}
    // The open-time target again. A failure retires the worker; with rethrow (a prefill's normal end) the request fails
    // too, the abort path keeps its original error.
    void restore(TierCacheApi& api,CacheTokenSite& site,bool rethrow){
        if(!lowered_)return;lowered_=0;std::string why;
        try{api.restore_spare_target();}catch(const std::exception& e){why=e.what();}catch(...){why="non-standard exception";}
        if(why.empty())return;const std::string m="spare target restore: "+why;site.retire(m);if(rethrow)throw std::runtime_error(m);}
};
// Opt-in prefill overlap (DecodeOptions::prefill_overlap), the library's held-layer contract: a layer
// held by make_resident runs its kernels with NO token (slots from the device table only; held sets
// are never victims); make_resident of another layer may run while they are in flight (refused only
// inside a token); a set is released only after its kernels' completion event. Per routed layer:
// begin (only the first layer is acquired here) -> bind + kernels + event -> end(k): wait and release
// the PREVIOUS layer (its event), then make_resident of the NEXT layer (host; overlaps k's kernels);
// k is not waited here, so the next layer is enqueued behind k with no GPU gap. After the last layer
// k is waited and released too (nothing stays held). At most two held (PrefillResidency::two_layers);
// exact by construction (only copies vs kernels reorder).
class PrefillOverlap {
    std::vector<unsigned> held_; // gated ordinals, oldest first
public:
    const std::vector<unsigned>& held()const{return held_;}
    void reset(){held_.clear();}
    template<class Acquire> void begin(unsigned g,Acquire&& acquire){if(!held_.empty()&&held_.back()==g)return;
        expert_need(held_.empty(),"prefill overlap: an earlier layer is still held");acquire(g);held_.push_back(g);}
    // next: the next routed layer's gated ordinal, < 0 after the last. wait(h): h's completion event.
    template<class Acquire,class Wait,class Release> void end(unsigned g,int next,Acquire&& acquire,Wait&& wait,Release&& release){
        expert_need(!held_.empty()&&held_.size()<=2&&held_.back()==g,"prefill overlap: end of a layer that is not the newest held");
        if(held_.size()==2){const unsigned p=held_.front();wait(p);release(p);held_.erase(held_.begin());}
        if(next>=0){acquire(unsigned(next));held_.push_back(unsigned(next));}
        else{wait(g);release(g);held_.clear();}}
    // As end, but a capacity refusal of the next layer (try_acquire false: nothing changed) runs this step serially
    // instead of retiring the worker: wait and release g, then acquire the next layer alone (a refusal there retires).
    // Returns false for such a serial step.
    template<class TryAcquire,class Acquire,class Wait,class Release>
    bool end(unsigned g,int next,TryAcquire&& try_acquire,Acquire&& acquire,Wait&& wait,Release&& release){
        expert_need(!held_.empty()&&held_.size()<=2&&held_.back()==g,"prefill overlap: end of a layer that is not the newest held");
        if(held_.size()==2){const unsigned p=held_.front();wait(p);release(p);held_.erase(held_.begin());}
        if(next<0){wait(g);release(g);held_.clear();return true;}
        if(try_acquire(unsigned(next))){held_.push_back(unsigned(next));return true;}
        wait(g);release(g);held_.clear();acquire(unsigned(next));held_.push_back(unsigned(next));return false;}
};
// Owns a cache together with the binding whose name it consumes. Members are
// destroyed in reverse declaration order, so on every path (normal teardown,
// constructor unwinding after the cache was installed, or a failing open) the
// cache closes before the binding's handles are released.
template<class Binding> class BoundCache {
    std::unique_ptr<Binding> binding_;      // declared first: destroyed last
    std::unique_ptr<TierCacheApi> cache_;   // declared second: destroyed first
public:
    template<class Open> BoundCache(std::unique_ptr<Binding> binding,Open&& open):binding_(std::move(binding)){
        expert_need(binding_!=nullptr,"cache binding required");cache_=open(*binding_);expert_need(cache_!=nullptr,"cache open returned nothing");}
    BoundCache(BoundCache&&)=default;BoundCache& operator=(BoundCache&&)=delete;
    ~BoundCache(){cache_.reset();binding_.reset();}
    TierCacheApi& cache(){return *cache_;}
    Binding& binding(){return *binding_;}
};
// Backend selection: exactly one chain. Tierstream always authenticates its unit
// file; the reference comparison backend is an addition inside that branch.
struct BackendPlan {bool reference=false,tierstream=false,authenticate_unit_file=false,reference_compare=false;};
inline BackendPlan backend_plan(const std::string& backend,bool compare){
    BackendPlan p;
    if(backend=="reference"){expert_need(!compare,"--reference-compare needs --expert-tier tierstream");p.reference=true;}
    else if(backend=="tierstream"){p.tierstream=true;p.authenticate_unit_file=true;p.reference_compare=compare;}
    else throw std::runtime_error("unknown expert tier backend: "+backend);
    return p;
}
// Exact integer counters: the JSON number text is parsed straight to uint64_t
// (no double round trip); signs, fractions, exponents and overflow are refused.
inline std::uint64_t counter_value(const Json& v){
    expert_need(v.is_number(),"expert counter must be a number");const auto& t=v.number_source();std::uint64_t x=0;
    const auto r=std::from_chars(t.data(),t.data()+t.size(),x);
    expert_need(!t.empty()&&t[0]>='0'&&t[0]<='9'&&r.ec==std::errc()&&r.ptr==t.data()+t.size(),"expert counter must be a nonnegative integer within uint64");return x;
}
// Receipt aggregation over per-step "experts" objects: every integer counter is
// summed (current and future cache counters alike); gauges, positions and
// millisecond timings are not.
inline Json aggregate_expert_totals(const Json& steps){
    std::vector<std::pair<std::string,std::uint64_t>> sums;
    auto skip=[](const std::string& k){return k=="position"||k=="free_slots"||(k.size()>3&&k.substr(k.size()-3)=="_ms");};
    if(steps.is_array())for(const auto& st:steps.as_array()){const Json* e=st.is_object()?st.find("experts"):nullptr;if(!e||!e->is_object())continue;
        for(const auto& [k,v]:e->as_object()){if(skip(k)||!v.is_number())continue;const auto x=counter_value(v);
            auto it=std::find_if(sums.begin(),sums.end(),[&](auto& s){return s.first==k;});
            if(it==sums.end())sums.emplace_back(k,x);else{expert_need(it->second<=~std::uint64_t(0)-x,"expert counter total overflows uint64");it->second+=x;}}}
    Json::Object out;for(auto& [k,v]:sums)out.emplace_back(k,Json(v));return Json(std::move(out));
}
inline UnitOffsets unit_offsets(const ExpertEntry& x){UnitOffsets u{};for(unsigned m=0;m<3;++m){for(unsigned r=0;r<3;++r){expert_need(x.unit.offset[m][r]<(1ULL<<32),"unit offset range");u.off[m][r]=std::uint32_t(x.unit.offset[m][r]);}u.rate[m]=x.unit.rate[m];}return u;}
// tierstream desc-v1 (tierstream src/desc.cpp): "sources" array, layers listed
// as gated ordinals 0..n-1 (trunk MoE 3..44 -> 0..41, MTP 45 -> 42; the model
// layer is kept in the free-form "x-glm-layer" key), one opaque part per unit
// with an "x-" format, offset + unit*stride in the unit file, stride == bytes.
inline constexpr const char* tierstream_unit_format="x-glm-exl3-unit-v1";
inline Json tierstream_descriptor(const ExpertCatalog& c,const std::string& unit_file,const std::string& model_sha256){
    const auto& g=c.geometry();auto d=Json::object();d.set("schema","tierstream/desc-v1");d.set("model",model_sha256);d.set("engine","glm-dec-rtx-native-v1");d.set("unit_align",std::uint64_t(256));
    auto src=Json::object();src.set("name","glm-expert-units");src.set("path",unit_file);src.set("align",std::uint64_t(4096));auto sources=Json::array();sources.push(src);d.set("sources",sources);
    expert_need(c.unit_stride()%256==0&&c.unit_stride()%4096==0,"unit stride must be 256- and 4096-aligned");auto layers=Json::array();
    for(unsigned l=g.first_layer;l<=g.last_layer;++l){auto L=Json::object();L.set("index",int(gated_ordinal(l)));L.set("kind","moe");L.set("n_units",int(g.experts));L.set("top_k",int(routed_topk));L.set("unit_bytes",c.unit_stride());L.set("x-glm-layer",int(l));
        auto part=Json::object();part.set("name","glm_exl3_unit");part.set("format",tierstream_unit_format);part.set("source",0);part.set("offset",c.unit_offset(l,0));part.set("stride",c.unit_stride());part.set("bytes",c.unit_stride());
        auto parts=Json::array();parts.push(part);L.set("parts",parts);layers.push(L);}
    d.set("layers",layers);return d;
}
}
