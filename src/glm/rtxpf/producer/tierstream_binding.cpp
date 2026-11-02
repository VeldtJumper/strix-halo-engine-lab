// Host-only tierstream binding (C ABI calls; the device-gated kernels are
// enqueued by the library on the caller's stream). No device work at include.
#include "tierstream_binding.hpp"
#include "tierstream/ts.h"
#include <stdexcept>
namespace strix::glm::rtxpf::producer {
namespace {
void ts_ok(ss_status s,const char* what){if(s!=TS_OK)throw std::runtime_error(std::string("tierstream ")+what+": "+ts_last_error());}
struct Handles {
    ts_desc* desc=nullptr;ts_plan* plan=nullptr;ts_ctx* ctx=nullptr;ts_file_identity source{};std::vector<ts_dg_unit> seed; // read during ts_open
    ~Handles(){if(ctx)ts_close(ctx);if(plan)ts_plan_free(plan);if(desc)ts_desc_free(desc);}
};
void configure(const TierOpenConfig& c,Handles& h,ts_budget& b,ts_options& o){
    const auto& p=c.budget;expert_need(p.refuse.empty(),"tier budget refused");expert_need(c.unit_bytes%256==0,"unit bytes must be a multiple of 256");
    ts_ok(ts_desc_from_json(c.descriptor_json.data(),c.descriptor_json.size(),&h.desc),"descriptor");
    expert_need(std::uint64_t(ts_desc_total_units(h.desc))==c.tiers.size(),"tier array length != descriptor units");
    std::uint64_t r=0,pin=0,disk=0;for(auto t:c.tiers){r+=t==TS_TIER_RESIDENT;pin+=t==TS_TIER_PINNED;disk+=t==TS_TIER_DISK;}
    expert_need(r==p.resident&&pin==p.pinned&&disk==p.disk&&r+pin+disk==c.tiers.size(),"tier array counts != budget");
    const auto lib=library_budget(p,c.unit_bytes,c.tiers.size(),c.max_ids);
    ts_budget_defaults(&b);b.device_bytes=lib.device_bytes;b.host_pinned_bytes=lib.host_bytes;b.margin_gib=0;b.margin_fraction=0;
    b.tier_mode=TS_TIER_MODE_EXCLUSIVE;b.dynamic_slots_per_layer=c.dynamic_per_layer;b.ring_slots=lib.ring_slots;b.host_spare_units=p.host_spares;
    ts_ok(ts_plan_from_tiers(h.desc,&b,c.tiers.data(),c.tiers.size(),&h.plan),"plan");
    ts_options_defaults(&o);o.backend=c.cuda?TS_BACKEND_CUDA:TS_BACKEND_CPU;o.device=0;o.forward_mode=TS_FORWARD_DEVICE_GATED;o.pin_planned=0;o.verify_hash=0;
    expert_need(p.spare_slots>=1,"the global spare count must be positive");o.dg_max_ids=c.max_ids;o.dg_spare_slots=std::uint32_t(p.spare_slots);
    expert_need(c.disk_queue>=1&&c.disk_queue<=64,"disk queue must be 1..64");
    if(p.three_tier){o.dg_disk_tier=1;o.dg_disk_prefetch=c.disk_prefetch?1u:0u;o.dg_disk_staging=std::uint32_t(p.staging);
        // Each read in flight holds one staging buffer.
        expert_need(c.disk_queue<=std::max<std::uint64_t>(p.staging,1),"disk queue exceeds the staging buffers (raise --tier-staging)");
        o.dg_disk_queue=c.disk_queue;o.dg_clean_evict=c.clean_evict?1u:0u;}
    if(c.source_size){h.source.volume=c.source_volume;h.source.index=c.source_index;h.source.size=c.source_size;o.source_identity_n=1;o.source_identities=&h.source;}
    // The device-gated context charges its own arena, tables, pinned blocks and
    // staging to GLM's ledger by name (no whole-device sample at open).
    if(c.ledger)o.budget=static_cast<ts_budget_ledger*>(c.ledger->raw());
    expert_need(c.policy>=0&&c.policy<=2,"tier policy");o.policy=c.policy==0?TS_POLICY_LRU:c.policy==1?TS_POLICY_SCORED:TS_POLICY_SCORED_AUTO;
    if(!c.seed.empty()){h.seed.reserve(c.seed.size());for(auto k:c.seed){expert_need(k<c.tiers.size(),"seed key out of range");h.seed.push_back({std::int32_t(k/c.expected_units),std::int32_t(k%c.expected_units)});}
        // The device slots go to the first distinct seed entries: they must be the tier array's residents.
        for(std::size_t i=0;i<c.seed.size()&&i<p.resident;++i)expert_need(c.tiers[c.seed[i]]==TS_TIER_RESIDENT,"seed head != resident tier keys");
        o.dg_seed=h.seed.data();o.dg_seed_n=std::uint32_t(h.seed.size());}
    o.dg_flush_ready=c.flush_ready?1u:0u; // as DS TierSite: flush the demand stream after each ready signal (WDDM)
    // Multi-disk: the library copies the directory list at open (c outlives ts_open).
    expert_need(c.drive_readers<=32,"drive readers must be 1..32");
    if(!c.weight_dirs.empty()){o.weight_dirs=c.weight_dirs.c_str();if(c.drive_readers)o.drive_readers=c.drive_readers;}
}
class Binding final:public TierCacheApi {
    // ss_dg_set_budgets takes all three: the open-time prefetch budgets and spare target (library defaults resolved).
    std::uint32_t per_layer_=0,per_token_=0,spare_open_=0;
    Handles h_;TierView view_{};CacheCounters before_{};std::uint64_t unit_=0;unsigned pred_=0,pred_through_=0;bool flush_ready_=false;std::string last_error_;StepPlan plan_;
    CacheCounters read(){ts_dg_stats s{};ts_ok(ts_stats_dg(h_.ctx,&s),"stats");CacheCounters c;c.lookups=s.lookups;c.misses=s.misses;c.fetched=s.fetched;c.prefetched=s.prefetched;c.resurrected=s.resurrected;c.covered=s.covered;c.bank_serves=s.bank_serves;c.file_reads=s.disk_reads;c.file_bytes=s.disk_read_bytes;c.file_ns=s.disk_read_ns;
        c.evicted=s.evicted;c.drops=s.disk_drops;c.writebacks=s.writebacks_done;c.stalled=s.stalled;c.timeouts=s.device_timeouts;c.free_slots=s.free_slots;
        c.file_wait_ns=s.disk_wait_ns;c.clean_drops=s.clean_drops;c.clean_drop_bytes=s.clean_drop_bytes;return c;}
public:
    explicit Binding(const TierOpenConfig& c){
        ts_budget b;ts_options o;configure(c,h_,b,o);ts_ok(ts_open(h_.desc,&b,h_.plan,&o,&h_.ctx),"open");flush_ready_=o.dg_flush_ready!=0;
        per_layer_=o.dg_prefetch_per_layer;per_token_=o.dg_prefetch_per_token;spare_open_=o.dg_spare_target==0xFFFFFFFFu?o.dg_spare_slots:o.dg_spare_target;
        ts_dg_device_view v{};ts_ok(ts_dg_get_device_view(h_.ctx,&v),"device view");
        expert_need(v.struct_size>=sizeof v&&v.absent_err&&v.protocol==SS_DG_PROTOCOL&&v.n_parts==1&&v.n_gated==c.expected_gated,"tierstream view: protocol/parts/gated layers");
        expert_need(v.parts[0].stride==c.unit_bytes&&v.parts[0].bytes==c.unit_bytes,"tierstream view: arena stride != GLM unit stride");
        for(unsigned g=0;g<c.expected_gated;++g)expert_need(ts_dg_gated_index(h_.ctx,int(g))==int(g),"tierstream: gated ordinals out of order");
        view_.slot_of=v.slot_of;view_.unit_base=v.unit_base;view_.abort=v.abort;view_.absent_err=v.absent_err;view_.guard_slot=v.guard_slot;view_.n_gated=v.n_gated;view_.units=c.expected_units;
        view_.base=v.parts[0].base;view_.stride=v.parts[0].stride;unit_=c.unit_bytes;before_=read();plan_.init(v.n_gated);
    }
    // An all-layer (decode) token never starts while a prefill keep set is held: the
    // library would skip that layer while the decode exports it and waits for its units.
    void begin()override{for(unsigned g=0;g<plan_.held.size();++g)if(plan_.held[g])throw std::runtime_error("decode token while gated layer "+std::to_string(g)+" still holds its prefill keep set (released on every prefill path; tierstream would skip it)");
        ts_ok(ts_begin(h_.ctx,nullptr,0),"begin");plan_.begin_all();}
    bool step_exports(unsigned g)const override{return plan_.exports(g);}
    void enqueue_begin(void* s)override{ts_ok(ts_dg_enqueue_begin(h_.ctx,s),"enqueue begin");}
    void enqueue_export(unsigned g,const std::int32_t* ids,unsigned n,void* s)override{expert_need(plan_.exports(g),"export of a gated layer the step skips or holds");
        ts_ok(ts_dg_enqueue_export(h_.ctx,int(g),ids,int(n),int(plan_.promise(g,pred_,pred_through_)),s),"enqueue export");}
    void set_predict_ids(unsigned n,unsigned through)override{expert_need(n<=SS_DG_MAX_IDS&&through<view_.n_gated,"predicted ids");pred_=n;pred_through_=n?through:0;}
    void enqueue_predict(unsigned g,const std::int32_t* ids,unsigned n,void* s)override{expert_need(pred_&&n==pred_&&g<pred_through_,"predict without a promise");
        expert_need(plan_.promise(g,pred_,pred_through_)==n,"prediction for a gated layer the step skips or holds");ts_ok(ts_dg_enqueue_predict(h_.ctx,int(g),ids,int(n),s),"enqueue predict");}
    std::vector<std::uint32_t> residency()override{std::uint32_t total=0;std::vector<ts_dg_unit> u(std::size_t(view_.n_gated)*view_.units);
        ts_ok(ts_dg_residency(h_.ctx,u.data(),std::uint32_t(u.size()),&total),"residency");std::vector<std::uint32_t> keys;keys.reserve(std::min<std::size_t>(total,u.size()));
        for(std::size_t i=0;i<total&&i<u.size();++i)keys.push_back(std::uint32_t(u[i].gated)*view_.units+std::uint32_t(u[i].unit));return keys;}
    bool lower_spare_target(unsigned n)override{ts_ok(ts_dg_set_budgets(h_.ctx,n,per_layer_,per_token_),"lower spare target");return true;}
    void restore_spare_target()override{ts_ok(ts_dg_set_budgets(h_.ctx,spare_open_,per_layer_,per_token_),"restore spare target");}
    void enqueue_wait(unsigned g,void* s)override{ts_ok(ts_dg_enqueue_wait(h_.ctx,int(g),s),"enqueue wait");}
    void enqueue_end(void* s)override{ts_ok(ts_dg_enqueue_end(h_.ctx,s),"enqueue end");}
    void end()override{const auto s=ts_dg_end_rows(h_.ctx,1);if(s!=TS_OK){const char* e=ts_last_error();last_error_=e?e:"";}ts_ok(s,"token not accepted");}
    // Failure receipts: the library's raw counters since open (device_timeouts is never
    // reset), its last token error (names the layer, the flag and the token) and the
    // submission setting, so a hang reports where and under which configuration.
    Json diagnostics()override{auto j=Json::object();j.set("flush_ready",flush_ready_);if(!last_error_.empty())j.set("last_token_error",last_error_);
        ts_dg_stats s{};s.struct_size=sizeof s;if(ts_stats_dg(h_.ctx,&s)!=TS_OK){const char* e=ts_last_error();j.set("stats_error",std::string(e?e:"stats failed"));return j;}
        j.set("tokens",s.tokens);j.set("device_timeouts",s.device_timeouts);j.set("stalled",s.stalled);j.set("miss_layers",s.miss_layers);j.set("lookups",s.lookups);j.set("misses",s.misses);
        j.set("covered_waits",s.covered_waits);j.set("fetched",s.fetched);j.set("prefetched",s.prefetched);j.set("free_slots",s.free_slots);j.set("slots",s.slots);j.set("blocks",s.blocks);
        j.set("wb_forced",s.wb_forced);j.set("wb_outstanding_bytes",s.wb_outstanding_bytes);
        // The library's view of the CUDA module loading mode at open (0 unknown, 1 eager, 2 lazy).
        j.set("cuda_module_loading",s.cuda_module_loading==1?"eager":s.cuda_module_loading==2?"lazy":"unknown");return j;}
    TierView view()const override{return view_;}
    void reset_open()override{ts_ok(ts_dg_reset_open(h_.ctx),"reset_open");before_=read();plan_.reset();}
    // Prefill through the cache (SS_DG_MAKE_RESIDENT): between
    // tokens; SS_E_CAPACITY changed nothing (false). A held set is never exported:
    // a prefill step exports nothing (rows 0 elsewhere) and reads the kept ids through
    // ss_dg_slot; the set is released right after its step, before any decode token.
    bool supports_make_resident()const override{return true;}
    bool make_resident(unsigned g,const std::int32_t* ids,unsigned n)override{
        const auto s=ts_dg_make_resident(h_.ctx,int(g),ids,int(n));if(s==TS_E_CAPACITY)return false;ts_ok(s,"make_resident");plan_.hold(g);return true;}
    void release_resident(unsigned g)override{ts_ok(ts_dg_keep(h_.ctx,int(g),nullptr,0),"release resident set");plan_.release(g);}
    void begin_layer_step(unsigned g)override{std::vector<std::uint32_t> rows(view_.n_gated,0);expert_need(g<rows.size(),"gated layer");rows[g]=1;
        ts_ok(ts_begin(h_.ctx,rows.data(),int(rows.size())),"begin single-layer step");plan_.begin_layer(g);}
    Json counters()override{ts_dg_stats s{};ts_ok(ts_stats_dg(h_.ctx,&s),"stats");auto j=Json::object();
        j.set("evicted",s.evicted);j.set("writebacks",s.writebacks_done);j.set("fetched",s.fetched);j.set("file_reads",s.disk_reads);j.set("file_bytes",s.disk_read_bytes);
        j.set("resident_units",s.resident_units);j.set("resident_evicted",s.resident_evicted);j.set("resident_bytes",s.resident_bytes);j.set("resident_ns",s.resident_ns);
        // make_resident's own transfers (tierstream #10), clean drops and the service's file-read wait.
        j.set("mkres_file_reads",s.mkres_file_reads);j.set("mkres_file_bytes",s.mkres_file_bytes);j.set("mkres_read_ns",s.mkres_read_ns);j.set("mkres_h2d_bytes",s.mkres_h2d_bytes);j.set("mkres_wb_bytes",s.mkres_wb_bytes);
        j.set("clean_drops",s.clean_drops);j.set("clean_drop_bytes",s.clean_drop_bytes);j.set("file_wait_ns",s.disk_wait_ns);return j;}
    // Deltas since the previous collect (cache_delta in expert_tier.hpp).
    Json stats()override{auto now=read();auto j=cache_delta(before_,now,unit_);before_=now;
        // Prefill residency (make_resident) totals since open: counted only here, nowhere else.
        ts_dg_stats s{};ts_ok(ts_stats_dg(h_.ctx,&s),"stats");j.set("prefill_resident_units_total",s.resident_units);j.set("prefill_resident_evicted_total",s.resident_evicted);
        j.set("prefill_resident_bytes_total",s.resident_bytes);j.set("prefill_resident_ns_total",s.resident_ns);return j;}
};
}
// A tierstream with ss_dg_make_resident (SS_DG_MAKE_RESIDENT): Binding overrides make_resident /
// release_resident / begin_layer_step, so prefill reads experts through the cache.
#ifdef SS_DG_MAKE_RESIDENT
bool tierstream_supports_make_resident(){return true;}
#else
bool tierstream_supports_make_resident(){return false;}
#endif
const char* tierstream_commit(){
#ifdef GLM_TIERSTREAM_COMMIT
    return GLM_TIERSTREAM_COMMIT;
#else
    return "unknown";
#endif
}
std::unique_ptr<TierCacheApi> open_tierstream(const TierOpenConfig& c){expert_need(c.cuda,"open_tierstream needs the CUDA backend");return std::make_unique<Binding>(c);}
std::pair<int,std::string> tierstream_admission_probe(const TierOpenConfig& c){
    Handles h;ts_budget b;ts_options o;
    try{configure(c,h,b,o);}catch(const std::exception& e){return {-1,e.what()};}
    o.backend=TS_BACKEND_CPU;auto s=ts_open(h.desc,&b,h.plan,&o,&h.ctx);return {int(s),s==TS_OK?std::string("opened"):std::string(ts_last_error())};
}
}
