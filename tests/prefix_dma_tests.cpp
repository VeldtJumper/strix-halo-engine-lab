#include "../apps/hip_prefix_state_cache.hpp"
#include "strix/prompt_terminal_diagnostics.hpp"
#include "strix/mtp_target_window_payload.hpp"
#include "../apps/state_cache_stats.hpp"
#include <stdexcept>
#include <cstdio>
#include <string>
#include <deque>
#include <map>
#include <iostream>

extern thread_local bool prefix_test_fail_host_allocations;

namespace {
bool queued = false, freed_pending = false, allocation_failure = false;
hipError_t drain_error = hipSuccess, copy_error = hipSuccess;
unsigned calls = 0, waits = 0, copies = 0, live_streams = 0, streams_created = 0;
unsigned resource_calls=0;
bool stream_failure=false;
bool snapshot_shim=false;
struct ShimEvent {bool ready=false;};
std::map<hipStream_t,std::deque<std::function<bool()>>> operations;
void pump(hipStream_t stream) {
    auto& q=operations[stream];
    while(!q.empty() && q.front()()) q.pop_front();
}

void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
}
extern "C" hipError_t hipStreamCreateWithFlags(hipStream_t* p, unsigned) { ++resource_calls; if(stream_failure) return hipErrorOutOfMemory; ++live_streams; ++streams_created; *p = reinterpret_cast<void*>(1); return hipSuccess; }
extern "C" hipError_t hipStreamDestroy(hipStream_t) { ++resource_calls; --live_streams; return hipSuccess; }
extern "C" hipError_t hipDeviceSynchronize() { return hipSuccess; }
extern "C" hipError_t hipStreamSynchronize(hipStream_t stream) {
    ++waits;
    if(snapshot_shim && drain_error==hipSuccess) pump(stream);
    if (drain_error == hipSuccess) queued = false;
    return drain_error;
}
extern "C" hipError_t hipMemcpyAsync(void* dest, const void* src, std::size_t n, hipMemcpyKind, hipStream_t stream) {
    ++copies;
    if(snapshot_shim) {
        if(copy_error==hipSuccess) operations[stream].push_back([=] {std::memcpy(dest,src,n);return true;});
    } else queued=true;
    return copy_error;
}
extern "C" hipError_t hipMalloc(void** p,std::size_t n) {
    ++resource_calls; if(allocation_failure) return hipErrorOutOfMemory;
    *p=new std::byte[n];return hipSuccess;
}
extern "C" hipError_t hipFree(void* p) {++resource_calls;delete[] static_cast<std::byte*>(p);return hipSuccess;}
extern "C" hipError_t hipEventCreateWithFlags(hipEvent_t* p,unsigned) {++resource_calls;*p=new ShimEvent;return hipSuccess;}
extern "C" hipError_t hipEventDestroy(hipEvent_t p) {++resource_calls;delete static_cast<ShimEvent*>(p);return hipSuccess;}
extern "C" hipError_t hipEventRecord(hipEvent_t p,hipStream_t stream) {
    static_cast<ShimEvent*>(p)->ready=false;
    operations[stream].push_back([=] {static_cast<ShimEvent*>(p)->ready=true;return true;});return hipSuccess;
}
extern "C" hipError_t hipEventQuery(hipEvent_t p) {
    if(drain_error!=hipSuccess) return drain_error;
    return static_cast<ShimEvent*>(p)->ready?hipSuccess:hipErrorNotReady;
}
extern "C" hipError_t hipEventSynchronize(hipEvent_t p) {
    ++waits;
    for(unsigned i=0;i<3;++i) for(auto& [stream,q]:operations) { (void)q;pump(stream); }
    return hipEventQuery(p);
}
extern "C" hipError_t hipStreamWaitEvent(hipStream_t stream,hipEvent_t p,unsigned) {
    operations[stream].push_back([=] {return static_cast<ShimEvent*>(p)->ready;});return hipSuccess;
}
extern "C" hipError_t hipHostMalloc(void** p,std::size_t n,unsigned) {return hipMalloc(p,n);}
extern "C" hipError_t hipHostFree(void* p) {return hipFree(p);}
extern "C" const char* hipGetErrorString(hipError_t) { return "injected HIP failure"; }

void async_snapshot_tests();
void batch_snapshot_tests();
void cancelled_checkpoint_retry_tests();
void split_snapshot_tests();

void prefix_dma_tests() {
    using namespace strix;
    using namespace strix::decode_cli;
    {
        // A served request's prompt-state diagnostic reports its failure on the line and never throws.
        const auto line = [](auto digest) {
            std::FILE* file = std::tmpfile();
            require(file != nullptr, "temporary file");
            print_target_state_digest(file, "prompt", 4096U, digest);
            std::rewind(file);
            std::string text(256, '\0');
            text.resize(std::fread(text.data(), 1, text.size(), file));
            std::fclose(file);
            return text;
        };
        const auto ok = line([] { return sha256_bytes(std::as_bytes(std::span("state", 5))); });
        require(ok.starts_with("qwen-target-state phase=prompt position=4096 sha256=") &&
                ok.find("error=") == std::string::npos, "successful digest prints its hex on the line");
        const auto failed = line([]() -> ModelContentIdentity { throw std::invalid_argument("target digest history/frontier differs"); });
        require(failed == "qwen-target-state phase=prompt position=4096 error=target digest history/frontier differs\n",
                "failed digest is logged on the line instead of failing the request");
    }
    {
        const auto id=sha256_bytes(std::as_bytes(std::span("attach",6)));
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        const auto alloc=[](std::uint64_t n) {return PrefixCacheBuffer{
            std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};};
        const auto store=[&] {return std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,{},0},alloc);};
        unsigned restored=0;
        const auto bind=[&](HipPrefixStateCacheSession& c) {
            bind_mtp_prefix_cache(c,1,8,2,2,3,16,id,17,19,true,false,true,
                [&](std::uint64_t p) {restored=static_cast<unsigned>(p);});
        };
        auto explicit_store=store();
        auto legacy=std::make_unique<HipPrefixStateCacheSession>(explicit_store,id,2,2,ModelContract{},layout,memory);
        bind(*legacy);
        std::shared_ptr<PrefixStateCache> automatic_store;
        std::unique_ptr<HipPrefixStateCacheSession> registry;
        HipPrefixStateCacheSession* cache=nullptr;
        const auto rollback=[&] {cache=nullptr;registry.reset();automatic_store.reset();};
        const auto attach=[&] {
            automatic_store=store();
            registry=std::make_unique<HipPrefixStateCacheSession>(automatic_store,id,2,2,ModelContract{},layout,memory);
            cache=registry.get();
        };
        finalize_cache_attachment(attach,[&] {if(cache) bind(*cache);},rollback);
        require(cache && registry && automatic_store && live_streams==2,"admitted attach owns exactly one registry/stream");
        const auto a=legacy->rtx_layout({}), b=cache->rtx_layout({});
        require(legacy->identity()==cache->identity() && a.identity_extension_hex==b.identity_extension_hex &&
            a.cache_identity_hex==b.cache_identity_hex && rtx_state_geometry_identity(a)==rtx_state_geometry_identity(b),
            "legacy and attached identities and descriptor geometry identical at the same B");
        require(a.rows.size()==2 && b.rows.size()==2 && a.fixed.size()==1 && b.fixed.size()==1 &&
            a.row_bytes==10 && a.fixed_bytes==256,"shared binding descriptor counts and sizes");
        for(std::size_t i=0;i<a.rows.size();++i) require(a.rows[i].id==b.rows[i].id &&
            a.rows[i].bytes_per_row==b.rows[i].bytes_per_row && a.rows[i].shift==b.rows[i].shift,"row descriptors identical");
        require(a.fixed[0].id==b.fixed[0].id && a.fixed[0].bytes==b.fixed[0].bytes &&
            a.fixed[0].offset==b.fixed[0].offset,"fixed descriptors identical");
        // Capture/restore exercises both the registered addresses and restore callback.
        SerialSessionContext state;state.state_layout_identity=1;state.next_position=state.committed_tokens=2;
        state.arithmetic_lineage=prepare_prefill_chunk_lineage(state.arithmetic_lineage,0,2,1,2,1);
        const std::vector<std::uint32_t> tokens(8,1);
        legacy->after_block(state,tokens);
        SerialSessionContext fresh;fresh.state_layout_identity=1;
        require(legacy->begin_request(fresh,tokens)==2 && restored==2,"legacy restore hook bound");
        restored=0;cache->after_block(state,tokens);fresh={};fresh.state_layout_identity=1;
        require(cache->begin_request(fresh,tokens)==2 && restored==2,"attached restore hook bound identically");
        rollback();legacy.reset();explicit_store.reset();
        require(!cache && !registry && !automatic_store && live_streams==0,"optional owners released");
        const auto created=streams_created;
        for(unsigned reason=0;reason<4;++reason) {
            CacheLateFit fit; // unknown late observation
            if(reason) {fit.observation_id=1;fit.limits_consistent=fit.aliasing_known=true;
                fit.U=128ULL<<30;fit.L=64ULL<<30;fit.A=reason==1?1:64ULL<<30;
                fit.F=fit.E=fit.S=fit.device_unpaid=0;fit.checkpoint=reason==3?5ULL<<30:4096;fit.device_free=reason==2?std::optional<std::uint64_t>{}:0;}
            finalize_cache_attachment([&] {if(fit_cache_late(fit).budget) attach();},[&] {if(cache) bind(*cache);},rollback);
            require(!cache && !registry && !automatic_store && live_streams==0 && streams_created==created,
                "late decline creates zero cache, registry or stream owners");
        }
        for(unsigned failure=0;failure<3;++failure) {
            stream_failure=failure==0;
            finalize_cache_attachment([&] {attach();if(failure==1) throw std::bad_alloc();},[&] {
                bind(*cache);if(failure==2) throw std::runtime_error("binding failure");},rollback);
            stream_failure=false;
            require(!cache && !registry && !automatic_store && live_streams==0,"setup or extension failure destroys every optional owner");
        }
        StateCacheStatsMirror mirror;
        finalize_cache_attachment([&] {
            automatic_store=store();mirror.publish(prefix_snapshot(*automatic_store));
            throw std::bad_alloc(); // publication succeeded; registry setup did not
        },[] {},[&] {rollback();mirror.publish({});});
        const auto mirrored=serve::parse_json(mirror.read());
        require(mirrored.find("ram_budget")->as_int()==0 &&
            !mirrored.find("engine")->find("state_cache")->find("prefix_enabled")->as_bool(),
            "partial setup rollback reseeds the published OFF mirror");
        bool bound=false;
        finalize_cache_attachment([&] {
            attach();mirror.publish(prefix_snapshot(*automatic_store));
        },[&] {
            bind(*cache);cache->attach_capture_pool(true);bound=true;
        },[&] {rollback();mirror.publish({});},[&] {
            require(bound,"startup publication follows successful binding");
            mirror.publish(prefix_snapshot(cache->store()));
        });
        const auto startup=serve::parse_json(mirror.read());
        const auto* prefix=startup.find("engine")->find("state_cache")->find("prefix");
        const auto live=automatic_store->counters();
        require(prefix->find("host_bytes")->as_int()==8192 && live.host_bytes==8192 &&
            prefix->find("pool_bytes")->as_int()==static_cast<std::int64_t>(live.pool_bytes) && live.pool_bytes>0 &&
            prefix->find("pool_metadata_reserved")->as_int()==static_cast<std::int64_t>(live.pool_metadata_reserved) &&
            prefix->find("pool_free_bytes")->as_int()==static_cast<std::int64_t>(live.pool_free_bytes) &&
            prefix->find("pool_staging_bytes")->as_int()==static_cast<std::int64_t>(live.pool_staging_bytes) &&
            live.pool_staging_bytes>0 && prefix->find("inserted")->as_int()==0 && prefix->find("lookups")->as_int()==0,
            "before-first-request mirror includes all finalized pool commitments");
        rollback();
        auto observer=std::make_unique<HipPrefixStateCacheSession>(nullptr,id,2,2,ModelContract{},layout,memory);
        bind(*observer);
        const auto imports=std::make_shared<RtxStateImportRegistry>(alloc);
        SerialSessionContext off_fresh;off_fresh.state_layout_identity=1;
        observer->set_rtx_facts({});
        observer->enable_rtx_import(imports,off_fresh);
        const auto off_layout=observer->rtx_layout({});
        const auto off_identity=observer->identity();
        cache=observer.get();
        for(unsigned failure=0;failure<3;++failure) {
            finalize_cache_attachment([&] {attach();if(failure==0) throw std::bad_alloc();},[&] {
                bind(*cache);
                if(failure==1) throw std::bad_alloc();
                cache->set_rtx_facts({});
                if(failure==2) throw std::runtime_error("import registry publication failure");
            },[&] {registry.reset();automatic_store.reset();cache=observer.get();});
            const auto actual=cache->rtx_layout({});
            require(cache==observer.get() && cache->identity()==off_identity &&
                actual.rows.size()==off_layout.rows.size() && actual.fixed.size()==off_layout.fixed.size() &&
                rtx_state_geometry_identity(actual)==rtx_state_geometry_identity(off_layout) &&
                imports->layout() && rtx_state_geometry_identity(*imports->layout())==rtx_state_geometry_identity(off_layout) && live_streams==1,
                "failed optional attach preserves fully bound OFF observer and import geometry");
        }
        observer.reset();cache=nullptr;
    }
    {
        const std::array<float,4> row{0.0F,-0.0F,1.0F,-2.0F};
        require(!raw_fp32_logits_digest({}),"missing terminal row fails closed");
        require(raw_fp32_logits_digest(row)==sha256_bytes(std::as_bytes(std::span(row))),
            "terminal digest hashes the full raw FP32 vocabulary row");
        OutputLogitsChain chain;chain.append(raw_fp32_logits_digest(row));
        const auto first=chain.digest;chain.append(raw_fp32_logits_digest(row));
        require(chain.rows==2 && chain.digest!=first,"every output row extends the logits chain");
        // Ticket snapshots must outlive owner reuse and terminal-stripe overwrite/reset.
        auto owner_digest=raw_fp32_logits_digest(row);
        OutputLogitsSnapshot ticket;ticket.capture(owner_digest);
        const auto saved=ticket.digest;
        // The window lease moved into the shared payload header. Keep both
        // independent diagnostic channels when tickets outlive scratch reuse.
        MtpTargetWindowLease window;window.tokens.resize(2);window.terminal_digest=saved;
        require(!window.terminal_logits_digest(0) && window.terminal_logits_digest(1)==saved,
            "uncertified window preserves terminal-only diagnostics");
        auto different=row;different[0]=7.0F;const auto second=raw_fp32_logits_digest(different);
        window.row_digests={saved,second};window.logit_digests={*saved,*second};
        require(window.terminal_logits_digest(0)==saved && window.terminal_logits_digest(1)==second &&
            !window.terminal_logits_digest(2),"certified moved lease resolves each row separately");
        OutputLogitsSnapshot accepted;accepted.capture(window.terminal_logits_digest(0));
        const auto committed_step=window.logit_digests[0];
        window.row_digests.clear();window.logit_digests.clear();window.terminal_digest.reset();
        require(accepted.digest==saved && committed_step==*saved,
            "moved lease preserves per-row terminal snapshots and committed-step hashes");
        auto stripe=row;stripe[0]=42.0F;
        owner_digest=raw_fp32_logits_digest(stripe);
        require(ticket.digest==saved && ticket.digest!=owner_digest,"ticket digest survives stripe reuse");
        owner_digest.reset();
        require(ticket.digest==saved,"ticket digest survives owner reset");
        // A missing row poisons just this request, even if later rows exist.
        chain.append({});chain.append(saved);
        require(chain.error && !chain.complete(2) && chain.rows==2,"missing output row fails closed without throwing");
        OutputLogitsChain other;other.append(ticket.digest);
        require(other.complete(1),"another request completes after a diagnostic failure");
        OutputLogitsChain short_chain;short_chain.append(saved);
        require(!short_chain.complete(2),"missing bonus row fails coverage");
        set_prefix_batch_certification_enabled(true);
        require(prefix_batch_certification_enabled(),"resolved policy crosses translation units");
        set_prefix_batch_certification_enabled(false);
        require(!prefix_batch_certification_enabled(),"default policy remains off");
        auto changed=row;changed[1]=0.0F;
        require(raw_fp32_logits_digest(changed)!=raw_fp32_logits_digest(row),"terminal digest preserves signed zero bits");
        SerialSessionContext session;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,0,2,1,2,1);
        const auto original=terminal_generations_lineage_digest(session);
        session.state.generations[0]++;
        require(terminal_generations_lineage_digest(session)!=original,"terminal metadata covers generations");
        session.state.generations[0]--;
        session.arithmetic_lineage.intervals.back().reduction_plan_identity++;
        require(terminal_generations_lineage_digest(session)!=original,"terminal metadata covers canonical lineage intervals");
        session.arithmetic_lineage.intervals.back().reduction_plan_identity--;
        session.arithmetic_lineage.tainted=true;
        require(terminal_generations_lineage_digest(session)!=original,"terminal metadata covers taint");
    }
    {
        PrefixStateCacheIdentityInputs in;
        in.model_content=sha256_bytes(std::as_bytes(std::span("model",5)));
        in.block_rows=1024; in.state_layout_identity=1; in.runtime_compatibility_identity=2;
        in.decode_execution_identity=3; in.prefill_execution_identity=4; in.tiled_execution_identity=5;
        require(prefix_effective_execution_signature("effective-cli",false,false,{},0U) !=
                prefix_effective_execution_signature("effective-cli",false,false,{},1U),
                "v5 canonical signature must bind the resolved prefill mask");
        const auto signature=[&] {return prefix_effective_execution_signature("effective-cli",false,false);};
        in.effective_execution=signature();
        const auto original=prefix_state_cache_identity(in);
        for (const char* name : {"STRIX_QWEN_PROFILE_ALLOW_PENDING", "STRIX_QWEN_PROFILE_OWNED_moe_route_bf16"}) {
            const char* old = std::getenv(name);
            const std::optional<std::string> saved = old ? std::optional<std::string>{old} : std::nullopt;
            ::setenv(name, "1", 1); in.effective_execution = signature();
            require(prefix_state_cache_identity(in) == original, "profile metadata preserves prefix namespace");
            if (saved) ::setenv(name, saved->c_str(), 1); else ::unsetenv(name);
        }
        {
            // STRIX_QPF2K_SCHED bit 2 where the route never defers (MTP bridge/HTTP, defaulted CLI prefix cache): the
            // namespace binds the executed schedule, so 14 there keeps the schedule-12 entries; a deferring route,
            // 13 and the startup provenance marker behave as before.
            const char* old=std::getenv("STRIX_QPF2K_SCHED");
            const std::optional<std::string> saved=old?std::optional<std::string>{old}:std::nullopt;
            const auto key=[&](const char* sched,bool defers) {
                ::setenv("STRIX_QPF2K_SCHED",sched,1);
                in.effective_execution=prefix_effective_execution_signature("effective-cli",false,false,{},prefill_options_active(),defers);
                return prefix_state_cache_identity(in);
            };
            const auto twelve=key("12",true);
            require(key("12",false)==twelve,"schedule 12 has no bit 2 to mask");
            require(key("14",false)==twelve,"a non-deferring route keeps the schedule-12 prefix namespace");
            require(key("14",true)!=twelve,"a deferring route binds schedule 14");
            require(key("13",false)!=twelve && key("15",false)==key("13",true),"only bit 2 is masked");
            ::setenv("STRIX_QPF2K_SCHED","14",1);
            in.effective_execution=prefix_effective_execution_signature("effective-cli",false,false);
            const auto unmarked=prefix_state_cache_identity(in);
            ::setenv("STRIX_QPF2K_PRE_INJECTED","1",1);
            in.effective_execution=prefix_effective_execution_signature("effective-cli",false,false);
            require(prefix_state_cache_identity(in)==unmarked,"startup provenance marker is not part of the namespace");
            ::unsetenv("STRIX_QPF2K_PRE_INJECTED");
            if(saved) ::setenv("STRIX_QPF2K_SCHED",saved->c_str(),1); else ::unsetenv("STRIX_QPF2K_SCHED");
            in.effective_execution=signature();
        }
        for(const char* name : {"STRIX_QSA_UNION_MERGE","STRIX_QSA_SCORE_ROWS64","STRIX_QSA_POOL_PACKED",
                "STRIX_QSA_DECODE_POOL","STRIX_QSA_DECODE_SELECT","STRIX_QSA_DECODE_POOL_BF16","STRIX_QSA_DECODE_POOL_CHECK","STRIX_PREFILL_MOE_MERGE_VEC",
                "STRIX_QPF2K_DEFAULTS","STRIX_BASELINE","STRIX_QPF2K_QSA_MASK","STRIX_QPF2K_MOE_ROUTE_BF16"}) {
            const char* old=std::getenv(name);
            const std::optional<std::string> saved=old?std::optional<std::string>{old}:std::nullopt;
            ::setenv(name,"0",1); in.effective_execution=signature();
            const auto off=prefix_state_cache_identity(in);
            ::setenv(name,"1",1); in.effective_execution=signature();
            require(prefix_state_cache_identity(in)!=off,"one runtime control changes the prefix namespace");
            if(saved) ::setenv(name,saved->c_str(),1); else ::unsetenv(name);
        }
        in.effective_execution=prefix_effective_execution_signature("effective-cli",true,false);
        require(prefix_state_cache_identity(in)!=original,"CLI baseline suppression changes the key");
        in.effective_execution=prefix_effective_execution_signature("effective-cli",false,true);
        require(prefix_state_cache_identity(in)!=original,"GSQ suppression changes the key");
        {
            SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
            SessionStateMemoryBinding memory;memory.layout_identity=1;
            QwenTiledPlanConfig plan;plan.rows=2;plan.block={2,QwenPlanOutput::terminal};
            QwenBatchConfig batch;
            const auto binding=[&](const ModelContentIdentity& sig) {
                HipPrefixStateCacheSession adapter({},sig,2,2,ModelContract{},layout,memory);
                adapter.set_plan(plan,8,true,2,batch);
                return std::pair{adapter.active_identity(),adapter.execution_certificate(2)};
            };
            const auto base=binding(signature());
            for(const char* name : {"STRIX_QSA_UNION_MERGE","STRIX_QSA_SCORE_ROWS64","STRIX_QSA_POOL_PACKED",
                    "STRIX_QSA_DECODE_POOL","STRIX_QSA_DECODE_SELECT","STRIX_QSA_DECODE_POOL_BF16","STRIX_QSA_DECODE_POOL_CHECK","STRIX_PREFILL_MOE_MERGE_VEC",
                    "STRIX_CTX_EXACT_MEMORY","STRIX_QSA_SCORE_RING","STRIX_QSA_SCORE_RING_ROWS","STRIX_TERMINAL_LOGITS_ARENA",
                    "STRIX_QPF2K_DEFAULTS","STRIX_BASELINE","STRIX_QPF2K_QSA_MASK","STRIX_QPF2K_MOE_ROUTE_BF16",
                    "STRIX_PROMPT_BATCH_SHARED","STRIX_BATCH_SERVICE_MS","STRIX_BATCH_SERVICE_LOOKAHEAD",
                    "STRIX_BATCH_PROMPT_SHARE","STRIX_BATCH_SHORT_FIRST_ROWS","STRIX_BATCH_INTERLEAVE"}) {
                const char* old=std::getenv(name);
                const std::optional<std::string> saved=old?std::optional<std::string>{old}:std::nullopt;
                ::setenv(name,"0",1);const auto off=binding(signature());
                ::setenv(name,"1",1);const auto on=binding(signature());
                require(off.first!=on.first && off.second!=on.second,"canonical v4 control binds batch namespace and certificate");
                if(saved) ::setenv(name,saved->c_str(),1);else ::unsetenv(name);
            }
            for(const auto sig : {prefix_effective_execution_signature("changed-cli",false,false),
                    prefix_effective_execution_signature("effective-cli",true,false),
                    prefix_effective_execution_signature("effective-cli",false,true),
                    prefix_effective_execution_signature("effective-cli",false,false,
                        RotaryIdentity(RotaryDescriptor::yarn(2),original))}) {
                const auto changed=binding(sig);
                require(changed.first!=base.first && changed.second!=base.second,"resolved execution and rotary bind batch proof");
            }
        }
        in.effective_execution={}; bool refused=false;
        try {(void)prefix_state_cache_identity(in);} catch(const std::invalid_argument&) {refused=true;}
        require(refused,"incomplete pre-v2 prefix identity refused");
    }
    {
        const auto table=sha256_bytes(std::as_bytes(std::span("table",5)));
        const std::array<RotaryIdentity,3> rotary{RotaryIdentity{},
            RotaryIdentity(RotaryDescriptor::yarn(2),table),RotaryIdentity(RotaryDescriptor::yarn(4),table)};
        std::array<ModelContentIdentity,3> signatures;
        for(std::size_t i=0;i<rotary.size();++i)
            signatures[i]=prefix_effective_execution_signature("same-effective-cli",false,false,rotary[i]);
        require(signatures[0]!=signatures[1] && signatures[1]!=signatures[2],"resolved rotary mode and factors bind signature");
        const auto other_table=sha256_bytes(std::as_bytes(std::span("other",5)));
        require(signatures[1]!=prefix_effective_execution_signature("same-effective-cli",false,false,
            RotaryIdentity(RotaryDescriptor::yarn(2),other_table)),"derived table identity binds signature");
        for(const unsigned route:{0U,1U,2U}) for(std::size_t producer=0;producer<rotary.size();++producer) {
            const bool certified=route!=0;
            auto alloc=[](std::uint64_t n) {return PrefixCacheBuffer{
                std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};};
            auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{4096,{},0},alloc);
            SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
            SessionStateMemoryBinding memory;memory.layout_identity=1;
            std::array<std::unique_ptr<HipPrefixStateCacheSession>,3> adapters;
            for(std::size_t i=0;i<rotary.size();++i) {
                adapters[i]=std::make_unique<HipPrefixStateCacheSession>(store,signatures[i],2,2,ModelContract{},layout,memory);
                if(certified) {QwenTiledPlanConfig plan;plan.rows=2;plan.block={2,QwenPlanOutput::terminal};
                    if(route==1) adapters[i]->set_plan(plan,8,true);
                    else adapters[i]->set_serial_plan(plan,8,{{0,2,2,true,false,false,false,true,false}},
                        [](auto) {return true;},true);}
            }
            const std::vector<std::uint32_t> tokens(8,1);
            const auto ns=adapters[producer]->active_identity();
            auto seg=std::make_shared<PrefixCacheSegment>();seg->identity=ns;seg->parent=ns;
            seg->digest=prefix_cache_block_digests(ns,tokens,2,1)[0];seg->block_rows=2;seg->tokens={1,1};seg->payload=alloc(2);
            auto cp=std::make_shared<PrefixCacheCheckpoint>();cp->identity=ns;cp->digest=seg->digest;
            cp->block_rows=2;cp->position=2;cp->segments={seg};cp->fixed=alloc(256);
            if(certified) cp->producer_certificate=adapters[producer]->execution_certificate(2);
            require(store->insert(cp),"rotary checkpoint captured");
            require(store->find_longest(ns,tokens,2,2)!=nullptr,"same rotary checkpoint hits");
            for(std::size_t reader=0;reader<rotary.size();++reader) if(reader!=producer) {
                require(!store->find_longest(adapters[reader]->active_identity(),tokens,2,2),"cross-rotary checkpoint misses both directions");
                if(certified) require(adapters[reader]->execution_certificate(2)!=cp->producer_certificate,
                    "cross-rotary producer certificate differs");
                SerialSessionContext fresh;fresh.state_layout_identity=1;
                require(adapters[reader]->begin_request(fresh,tokens)==0 && fresh.next_position==0,
                    "cross-rotary restore misses before state writes");
            }
        }
    }
    const auto id = sha256_bytes(std::as_bytes(std::span("dma-test", 8)));
    for (const auto failure : {hipSuccess, hipErrorOutOfMemory, hipErrorUnknown}) {
        for (unsigned scenario = 0; scenario < 3; ++scenario) {
            queued = freed_pending = false; calls = 0;
            drain_error = failure;
            allocation_failure = scenario == 0;
            copy_error = scenario == 1 ? hipErrorUnknown : hipSuccess;
            const auto before = PrefixDmaQuarantine::count.load();
            const auto before_bytes = PrefixDmaQuarantine::bytes.load();
            auto store = std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{4096, {}, 0}, [](std::uint64_t n) {
                if (++calls == 2 && allocation_failure) throw PrefixCacheAllocationFailure();
                return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n ? n : 1]{}, [](std::byte* p) {
                    if (queued) freed_pending = true;
                    delete[] p;
                }), n};
            });
            SerialSessionStateLayout layout; layout.layout_identity = 1; layout.context_tokens = 2;
            SessionStateMemoryBinding memory; memory.layout_identity = 1;
            {
                HipPrefixStateCacheSession capture(store, id, 1, 1, ModelContract{}, layout, memory);
                capture.add_rows(1, 1, 0); capture.add_fixed_device(1, 1);
                SerialSessionContext session; session.next_position = session.committed_tokens = 1; session.state_layout_identity = 1;
                bool threw = false;
                try { capture.after_block(session, std::vector<std::uint32_t>{1, 2}); }
                catch (const std::exception&) { threw = true; }
                require(threw == (failure != hipSuccess || scenario == 1), "capture fault propagation");
            }
            require(!freed_pending, "unresolved DMA destination freed");
            require(PrefixDmaQuarantine::count.load() == before + (failure == hipSuccess ? 0U : 1U), "quarantine owner count");
            require(PrefixDmaQuarantine::bytes.load() == before_bytes + (failure == hipSuccess ? 0U : scenario == 2 ? 257U : 1U), "quarantine byte count");
            require(store->counters().inserted == (failure == hipSuccess && scenario == 2 ? 1U : 0U), "partial capture published");
            require(store->counters().capture_failures == (failure == hipSuccess && scenario == 0 ? 1U : 0U), "capture fail-soft classification");
            queued = false;
        }
    }
    for (const auto failure : {hipErrorOutOfMemory, hipErrorUnknown}) {
        for (const bool fail_copy : {false, true}) {
            queued = freed_pending = allocation_failure = false; calls = 0;
            drain_error = copy_error = hipSuccess;
            const auto before = PrefixDmaQuarantine::count.load();
            const auto before_bytes = PrefixDmaQuarantine::bytes.load();
            {
                auto store = std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{4096, {}, 0}, [](std::uint64_t n) {
                    return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n ? n : 1]{}, [](std::byte* p) {
                        if (queued) freed_pending = true;
                        delete[] p;
                    }), n};
                });
                SerialSessionStateLayout layout; layout.layout_identity = 1; layout.context_tokens = 2;
                SessionStateMemoryBinding memory; memory.layout_identity = 1;
                HipPrefixStateCacheSession capture(store, id, 1, 1, ModelContract{}, layout, memory);
                capture.add_rows(1, 1, 0); capture.add_fixed_device(1, 1);
                SerialSessionContext session; session.next_position = session.committed_tokens = 1; session.state_layout_identity = 1;
                capture.after_block(session, std::vector<std::uint32_t>{1, 2});
                drain_error = failure; copy_error = fail_copy ? hipErrorUnknown : hipSuccess;
                SerialSessionContext fresh; fresh.state_layout_identity = 1;
                bool threw = false;
                try { (void)capture.begin_request(fresh, std::vector<std::uint32_t>{1, 2}); }
                catch (const std::exception&) { threw = true; }
                require(threw && fresh.health == SerialSessionHealth::poisoned, "restore failure poisons session");
            }
            require(!freed_pending && PrefixDmaQuarantine::count.load() == before + 1 &&
                PrefixDmaQuarantine::bytes.load() == before_bytes + 257, "restore source lease quarantine");
            queued = false;
        }
    }

    {
        queued=freed_pending=allocation_failure=false;drain_error=copy_error=hipSuccess;
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{4096,{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        HipPrefixStateCacheSession adapter(store,id,2,2,ModelContract{},layout,memory);
        adapter.add_rows(1,1,0);adapter.add_fixed_device(1,1);
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        adapter.set_plan(config,8,true);
        auto other_id=id; other_id.bytes[0]^=1;
        HipPrefixStateCacheSession other(store,other_id,2,2,ModelContract{},layout,memory);
        other.set_plan(config,8,true);
        require(other.active_identity()!=adapter.active_identity(),"execution signature binds certified RAM namespace");
        require(other.execution_certificate(2)!=adapter.execution_certificate(2),"execution signature binds producer certificate");
        const std::vector<std::uint32_t> tokens(8,1);
        const auto digests=prefix_cache_block_digests(adapter.active_identity(),tokens,2,1);
        auto seg=std::make_shared<PrefixCacheSegment>();
        seg->identity=adapter.active_identity();seg->parent=seg->identity;seg->digest=digests[0];
        seg->block_rows=2;seg->tokens={1,1};seg->payload=store->allocate(2);
        auto cp=std::make_shared<PrefixCacheCheckpoint>();cp->identity=seg->identity;cp->digest=seg->digest;
        cp->block_rows=2;cp->position=2;cp->segments={seg};cp->fixed=store->allocate(256);
        cp->producer_certificate=adapter.execution_certificate(2);
        cp->producer_certificate[0]^=1U;
        require(store->insert(cp),"plan refusal seed");
        SerialSessionContext fresh;fresh.state_layout_identity=1;
        require(adapter.begin_request(fresh,tokens)==0 && !queued && fresh.next_position==0,
            "plan preflight miss performs zero live writes");
        // Correct producer proof but corrupt/truncated metadata must also miss before writes.
        cp->producer_certificate[0]^=1U;
        require(adapter.begin_request(fresh,tokens)==0 && !queued && fresh.next_position==0,
            "metadata preflight miss performs zero live writes");
        const auto stats=store->counters();
        require(stats.plan_misses==2 && stats.host_hits==0 && stats.restored_tokens==0,"preflight misses accounting");
    }

    {
        queued=freed_pending=allocation_failure=false;drain_error=copy_error=hipSuccess;
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        HipPrefixStateCacheSession adapter(store,id,2,2,ModelContract{},layout,memory);
        adapter.add_rows(1,1,0);adapter.add_rows(2,1,1);adapter.add_fixed_device(1,1);
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        config.generations_per_row[0]=3;config.valid_state=1;
        config.block.execution=1;config.block.kernel=2;
        adapter.set_plan(config,8,true);
        const std::vector<std::uint32_t> tokens(8,1);
        SerialSessionContext session;session.state_layout_identity=1;
        for(std::uint64_t p=2;p<=4;p+=2) {
            session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,p-2,p,1,2,1);
            session.next_position=session.committed_tokens=p;session.state.generations[0]=p*3U;
            session.state.valid=static_cast<PersistentState>(1);
            adapter.after_block(session,tokens);
        }
        const auto digests=prefix_cache_block_digests(adapter.active_identity(),tokens,2,2);
        require(store->find_segment(digests[0])->payload.bytes==3 && store->find_segment(digests[1])->payload.bytes==4,
            "shift-1 first/next segment geometry");
        // A valid warm checkpoint cannot authorize a policy/constraint verify request.
        adapter.set_policy_selection(true);
        SerialSessionContext refused;refused.state_layout_identity=1;queued=false;
        const auto checkpoints=store->counters().host_checkpoints;
        require(adapter.begin_request(refused,tokens)==0 && !queued && refused.next_position==0,
            "policy request refuses warm restore before device writes");
        require(adapter.request().refusal=="policy-selection-cache-unqualified","named policy cache refusal");
        adapter.after_block(session,tokens);
        require(!adapter.try_prepare_capture(session,tokens,nullptr,true) && !queued &&
            store->counters().host_checkpoints==checkpoints,"policy request refuses both capture paths");
        adapter.set_policy_selection(false);
        adapter.set_retention(false);
        SerialSessionContext fresh;fresh.state_layout_identity=1;
        require(adapter.begin_request(fresh,tokens)==4 && fresh.state.generations[0]==12,
            "certified no-store restore retains complete metadata");
        require(arithmetic_lineage_valid(fresh.arithmetic_lineage,4),"certified lineage restored");
        // Obtain the RAM lease and inject a generation-only corruption of the longest candidate.
        auto cp=store->find_longest(adapter.active_identity(),tokens,2,4);
        auto corrupt=std::const_pointer_cast<PrefixCacheCheckpoint>(cp);
        corrupt->metadata[9U*8U]^=std::byte{1};
        const auto before=store->counters();
        fresh={};fresh.state_layout_identity=1;queued=false;
        require(adapter.begin_request(fresh,tokens)==2,"generation mismatch falls back to shorter checkpoint");
        require(store->counters().plan_misses==before.plan_misses+1,"generation refusal counted");
        corrupt->metadata[9U*8U]^=std::byte{1};
        corrupt->segments[0].reset(); // Invalid section must be refused before token flattening.
        fresh={};fresh.state_layout_identity=1;queued=false;
        require(adapter.begin_request(fresh,tokens)==2,"malformed section safely falls back");
        // Different priming families cannot alias row segments even for identical tokens.
        config.priming=255;adapter.set_plan(config,8,true);
        const auto other=prefix_cache_block_digests(adapter.active_identity(),tokens,2,2);
        require(other[0]!=digests[0] && !store->find_segment(other[0]),"incompatible producer segment namespace");
        fresh={};fresh.state_layout_identity=1;queued=false;
        require(adapter.begin_request(fresh,tokens)==0 && !queued,"other family misses without writes");
    }

    {
        // Serial certificates use actual solo full-T units and a separate RAM namespace.
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{16384,{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal,1,2};
        const std::vector<KnownPromptChunkUnit> trace{{0,2,2,true,false,false,false,true,false},
            {2,4,2,true,false,false,false,true,false},{4,6,2,true,false,false,false,true,false},
            {6,8,2,true,false,false,false,false,false}};
        const std::vector<std::uint32_t> tokens(8,1);
        HipPrefixStateCacheSession producer(store,id,2,2,ModelContract{},layout,memory);
        producer.add_rows(1,1,0);producer.add_fixed_device(1,1);
        producer.set_serial_plan(config,8,trace,[](auto) {return true;},true);
        SerialSessionContext session;session.state_layout_identity=1;
        (void)producer.begin_request(session,tokens);
        session.next_position=session.committed_tokens=2;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage({},0,2,1,2,1);
        producer.observe_serial_unit(trace[0]);producer.after_block(session,tokens);
        require(store->counters().inserted==1,"serial actual full-T producer captured");
        auto cp=store->find_longest(producer.active_identity(),tokens,2,2);
        require(cp && cp->producer_certificate==producer.execution_certificate(2),"serial producer certificate stored");
        const auto serial_namespace=producer.active_identity();
        const auto attempt=[&](auto cold,bool resumable,const char* reason) {
            HipPrefixStateCacheSession consumer(store,id,2,2,ModelContract{},layout,memory);
            consumer.add_rows(1,1,0);consumer.add_fixed_device(1,1);
            consumer.set_serial_plan(config,8,cold,[=](auto) {return resumable;},true);
            SerialSessionContext fresh;fresh.state_layout_identity=1;queued=false;
            const auto before_copies=copies;
            const auto p=consumer.begin_request(fresh,tokens);
            if(reason) {
                require(!p && !queued && copies==before_copies && fresh.next_position==0 && fresh.committed_tokens==0 &&
                    fresh.arithmetic_lineage.intervals.empty(),"serial preflight miss has zero state writes");
                require(consumer.request().refusal==reason && consumer.request().certificate_declines.at(reason)>0,
                    "serial preflight decline is counted and named");
            } else require(p==2 && copies>before_copies && fresh.next_position==2,"compatible serial producer restores");
        };
        attempt(trace,true,nullptr);
        auto different=trace;different[0].headless=false;
        attempt(different,true,"serial-producer-plan-mismatch");
        // A consumer cursor unable to preserve its whole-prompt suffix must decline before DMA.
        attempt(trace,false,"serial-cursor-continuation-unsupported");
        auto corrupt=std::const_pointer_cast<PrefixCacheCheckpoint>(cp);
        const auto certificate=corrupt->producer_certificate;
        corrupt->producer_certificate=qwen_full_t_certificate(config,8,2);
        attempt(trace,true,"serial-producer-plan-mismatch");
        corrupt->producer_certificate=certificate;
        corrupt->metadata[0]^=std::byte{1};
        attempt(trace,true,"serial-checkpoint-preflight");
        corrupt->metadata[0]^=std::byte{1};
        for(unsigned variant=0;variant<4;++variant) {
            auto unsupported=trace;
            if(variant==0) unsupported[0].merged=true;
            if(variant==1) unsupported[0].balanced=true;
            if(variant==2) unsupported[0].deferred=true;
            if(variant==3) unsupported[0].end=4; // boundary lies inside a physical merged unit
            producer.set_serial_plan(config,8,unsupported,[](auto) {return true;},true);
            producer.observe_serial_unit(unsupported[0]);queued=false;
            producer.after_block(session,tokens);
            require(!queued && producer.request().refusal=="serial-capture-unit-unsupported",
                "merged/balanced/deferred/interior capture declines without writes");
        }
        producer.set_plan(config,8,true);
        require(producer.active_identity()!=serial_namespace,"MTP namespace differs from serial");
        SerialSessionContext fresh;fresh.state_layout_identity=1;queued=false;
        require(producer.begin_request(fresh,tokens)==0 && !queued,"MTP cannot restore serial checkpoint");
        producer.set_plan(config,8,false); // batch/uncertified route remains disabled
        require(producer.begin_request(fresh,tokens)==0 && !queued,"batch cannot restore serial checkpoint");
        const auto changed_id=sha256_bytes(std::as_bytes(std::span("changed-effective-signature",27)));
        HipPrefixStateCacheSession changed(store,changed_id,2,2,ModelContract{},layout,memory);
        changed.set_serial_plan(config,8,trace,[](auto) {return true;},true);
        require(changed.active_identity()!=serial_namespace && changed.begin_request(fresh,tokens)==0 && !queued,
            "effective signature change misses serial namespace without writes");
        require(store->counters().certificate_declines[5]>0,"named declines exported by store counters");
    }

    async_snapshot_tests();
    split_snapshot_tests();
    batch_snapshot_tests();
    cancelled_checkpoint_retry_tests();

}


void async_snapshot_tests() {
    using namespace strix;
    using namespace strix::decode_cli;
    const ModelContract production;
    const auto fixed_device=gdn_matrix_state_bytes_per_session(production)+gdn_conv_state_bytes_per_session(production)+
        ple_conv_state_bytes_per_session(production)+production.hidden_size*production.hc_count*2U;
    require(fixed_device==116400128U,"production BF16 device fixed geometry");
    require(fixed_device+29952U*2048U==177741824U && fixed_device+29952U*8192U-2304U==361764608U,
        "production selected capture staging bytes");
    set_prefix_batch_certification_enabled(true);
    snapshot_shim=true;queued=false;drain_error=copy_error=hipSuccess;allocation_failure=false;
    const auto producer=reinterpret_cast<hipStream_t>(2), predictor=reinterpret_cast<hipStream_t>(3);
    const PrefixCaptureOrder ordered{producer,predictor,predictor};
    const auto id=sha256_bytes(std::as_bytes(std::span("snapshot",8)));
    const std::vector<std::uint32_t> tokens(8,1);
    // Repeated destruction, failed validation, cancellation and either completion ordering.
    for(unsigned scenario=0;scenario<8;++scenario) {
        operations.clear();
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        auto adapter=std::make_unique<HipPrefixStateCacheSession>(store,id,2,2,ModelContract{},layout,memory);
        std::array<std::byte,8> target{},mtp{}; std::byte fixed{31},held{41};
        target[0]=std::byte{11};target[1]=std::byte{12};mtp[0]=std::byte{21};
        adapter->add_rows(reinterpret_cast<std::uintptr_t>(target.data()),1,0);
        adapter->add_rows(reinterpret_cast<std::uintptr_t>(mtp.data()),1,1);
        adapter->add_fixed_device(reinterpret_cast<std::uintptr_t>(&fixed),1);
        adapter->add_fixed_device(reinterpret_cast<std::uintptr_t>(&held),1,"mtp.held");
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        config.block.execution=1;config.block.kernel=2;config.priming=255;
        adapter->set_plan(config,8,true);
        adapter->attach_capture_pool(true);
        SerialSessionContext session;session.state_layout_identity=1;
        session.next_position=session.committed_tokens=2;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,0,2,1,2,1);
        const auto before_waits=waits;
        adapter->set_retention(false);
        require(!adapter->try_prepare_capture(session,tokens,producer,true,ordered) && operations.empty(),"no-store skips before any submission");
        require(store->counters().snapshot_preparations==0,"no-store does not prepare a snapshot");
        adapter->set_retention(true);
        require(!adapter->try_prepare_capture(session,tokens,producer,false,ordered),"failed target validation skips");
        if(scenario==0) {
            const auto foreign=reinterpret_cast<hipStream_t>(4);
            for(const auto unsafe:{PrefixCaptureOrder{},PrefixCaptureOrder{foreign,predictor,predictor},
                PrefixCaptureOrder{producer,predictor,foreign}}) {
                const auto before=store->counters().capture_skips_cross_stream;
                require(!adapter->try_prepare_capture(session,tokens,producer,true,unsafe) && operations.empty(),
                    "unknown/foreign target or predictor refuses before snapshot submission");
                // Adversarial foreign-tail probe: run its write before pumping the old producer.
                operations[foreign].push_back([&] {fixed=std::byte{99};target.fill(std::byte{99});return true;});
                pump(foreign);pump(producer);PrefixAsyncCapture::reap();
                require(fixed==std::byte{99} && !PrefixAsyncCapture::slot && store->counters().inserted==0 &&
                    store->counters().capture_skips_cross_stream==before+1 && waits==before_waits,
                    "foreign-stream overwrite never publishes wrong bytes or adds a wait");
                operations.clear();fixed=std::byte{31};target[0]=std::byte{11};target[1]=std::byte{12};
            }
            // A suffix owner is refused even if a caller supplies matching stream handles.
            auto suffix=config;suffix.tiled_start=false;
            adapter->set_plan(suffix,3,true);
            require(!adapter->try_prepare_capture(session,std::span(tokens).first(3),producer,true,ordered) && operations.empty(),
                "next serial/window suffix cannot authorize a snapshot");
            adapter->set_plan(config,8,true);
        }
        if(scenario==6) adapter->set_staging_cap(1);
        PrefixCacheBuffer exhausted;
        if(scenario==7) exhausted=store->allocate(store->counters().pool_bytes);
        const auto hot_calls=resource_calls;
        auto d=adapter->try_prepare_capture(session,tokens,producer,true,ordered);
        require(resource_calls==hot_calls,"capture performs no HIP resource allocation/free");
        require(store->counters().snapshot_preparations==1,"retained preparation counted even before failure or insertion");
        if(scenario==7) require(store->counters().capture_skipped_no_slot==1,"exhausted pool capture counted");
        if(scenario>=6) {require(!d && store->counters().inserted==0,"pre-submit cap/allocation skip");continue;}
        require(bool(d),"snapshot prepared");
        require(!adapter->try_prepare_capture(session,tokens,producer,true,ordered),"global busy slot skips without wait");
        PrefixAsyncCapture::reap();
        require(store->counters().inserted==0,"worker not yet submitted cannot publish");
        const auto held_free=store->counters().pool_free_bytes;
        require(held_free<store->counters().pool_bytes,"in-flight descriptor holds slab leases");
        // A next block cannot run ahead of reads already queued on its stream.
        bool next_block_ran=false;
        operations[producer].push_back([&] {fixed=std::byte{99};target.fill(std::byte{99});next_block_ran=true;return true;});
        require(!next_block_ran && fixed==std::byte{31},"same-stream next block cannot execute before snapshot pump");
        pump(producer);
        require(next_block_ran,"next same-stream block executed after snapshot");
        d->submit_predictor(predictor);
        operations[predictor].push_back([&] {mtp.fill(std::byte{99});held=std::byte{99};return true;});
        require(held==std::byte{41},"same-stream predictor writer cannot run before snapshot pump");
        pump(predictor);
        if(scenario%2==0) {
            pump(d->download);PrefixAsyncCapture::reap();
            require(store->counters().inserted==0,"D2H without commit cannot publish");
            require(store->counters().pool_free_bytes==held_free,"done alone cannot recycle payload leases");
        }
        d->commit(scenario<2); // failed validation/reset
        adapter.reset(); // descriptor survives prompt/registry destruction
        const auto retire_calls=resource_calls;
        PrefixAsyncCapture::reap();
        if(scenario%2) {
            require(store->counters().inserted==0,"commit without D2H cannot publish");
            require(store->counters().pool_free_bytes==held_free,"handoff alone cannot recycle pending DMA leases");
        }
        if(scenario==5) {PrefixAsyncCapture::shutdown();require(waits==before_waits+1U,"shutdown explicitly drains");}
        else {pump(d->download);PrefixAsyncCapture::reap();}
        require(!PrefixAsyncCapture::slot,"completed descriptor reaped");
        require(resource_calls==retire_calls,"retirement performs no HIP resource allocation/free");
        require(store->counters().staging_bytes==0 && store->counters().snapshot_preparations==1,"settled capture has no live staging and keeps preparation count");
        adapter=std::make_unique<HipPrefixStateCacheSession>(store,id,2,2,ModelContract{},layout,memory);
        adapter->add_rows(reinterpret_cast<std::uintptr_t>(target.data()),1,0);
        adapter->add_rows(reinterpret_cast<std::uintptr_t>(mtp.data()),1,1);
        adapter->add_fixed_device(reinterpret_cast<std::uintptr_t>(&fixed),1);
        adapter->add_fixed_device(reinterpret_cast<std::uintptr_t>(&held),1,"mtp.held");
        adapter->set_plan(config,8,true); adapter->set_retention(false);
        require(!adapter->try_prepare_capture(session,tokens,producer,true,ordered) &&
            store->counters().snapshot_preparations==1 && store->counters().staging_bytes==0,
            "no-store after a retaining high-water mark starts no preparation");
        require(store->counters().inserted==(scenario<2?1U:0U),"only committed successful DMA published");
        require(waits==before_waits+(scenario==5?1U:0U),"snapshot request path added a wait");
        if(scenario<2) {
            const auto& c=*d->checkpoint;
            require(c.fixed.data.get()[0]==std::byte{31} && c.fixed.data.get()[256]==std::byte{41},"stable fixed and held snapshots");
            require(c.segments[0]->payload.data.get()[0]==std::byte{11} &&
                c.segments[0]->payload.data.get()[2]==std::byte{21},"stable target and predictor row snapshots");
        }
        d.reset();
        if(scenario>=2) require(store->counters().pool_free_bytes==store->counters().pool_bytes,
            "cancelled payloads recycle only after done and handoff and final lease release");
    }
    // Allocation counts match the original per-capture allocator. Both routes
    // exercise the same framing, DMA, cancellation and publication implementation.
    std::vector<std::byte> reference_bytes;
    for(bool automatic:{false,true}) {
        operations.clear();
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{UINT64_C(32)<<30,{},0},hip_pinned_prefix_capture_buffer);
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        HipPrefixStateCacheSession adapter(store,id,2,2,ModelContract{},layout,memory);
        std::array<std::byte,8> target{},mtp{};
        std::array<std::byte,256> fixed{};fixed.fill(std::byte{31});
        target[0]=std::byte{11};target[1]=std::byte{12};mtp[0]=std::byte{21};
        adapter.add_rows(reinterpret_cast<std::uintptr_t>(target.data()),1,0);
        adapter.add_rows(reinterpret_cast<std::uintptr_t>(mtp.data()),1,1);
        adapter.add_fixed_device(reinterpret_cast<std::uintptr_t>(fixed.data()),fixed.size());
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        config.block.execution=1;config.block.kernel=2;config.priming=255;
        adapter.set_plan(config,8,true);
        const auto startup_calls=resource_calls;
        adapter.attach_capture_pool();
        require(resource_calls==startup_calls && !store->capture_resources && !store->counters().pool_bytes &&
            store->counters().host_bytes==0,"explicit 32 GiB budget pins no startup payload");
        // Use a small pool for the automatic shim; explicit keeps its 32 GiB budget.
        if(automatic) {
            store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,{},0},hip_pinned_prefix_capture_buffer);
        }
        // A separate adapter is needed only to bind the smaller automatic budget.
        HipPrefixStateCacheSession capture(store,id,2,2,ModelContract{},layout,memory);
        capture.add_rows(reinterpret_cast<std::uintptr_t>(target.data()),1,0);
        capture.add_rows(reinterpret_cast<std::uintptr_t>(mtp.data()),1,1);
        capture.add_fixed_device(reinterpret_cast<std::uintptr_t>(fixed.data()),fixed.size());
        capture.set_plan(config,8,true);
        if(automatic) capture.attach_capture_pool(true);
        SerialSessionContext session;session.state_layout_identity=1;session.next_position=session.committed_tokens=2;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,0,2,1,2,1);
        for(bool commit:{false,true}) {
            const auto before=resource_calls;
            auto d=capture.try_prepare_capture(session,tokens,producer,true,ordered);
            require(bool(d),"both allocation routes prepare capture");
            require(resource_calls-before==(automatic?0U:7U),"original capture: two pinned payloads, staging, stream, three events");
            d->submit_predictor(predictor);pump(producer);pump(predictor);pump(d->download);
            std::vector<std::byte> bytes=d->checkpoint->metadata;
            for(const auto& segment:d->checkpoint->segments)
                bytes.insert(bytes.end(),segment->payload.data.get(),segment->payload.data.get()+segment->payload.bytes);
            bytes.insert(bytes.end(),d->checkpoint->fixed.data.get(),d->checkpoint->fixed.data.get()+d->checkpoint->fixed.bytes);
            if(reference_bytes.empty()) reference_bytes=bytes;
            else require(bytes==reference_bytes,"automatic and original capture payload and metadata bytes identical");
            const auto retire_calls=resource_calls;
            d->commit(commit);
            // Publication itself may allocate. Cancellation and checked resource
            // retirement must release already granted leases under host pressure.
            if(automatic && !commit) prefix_test_fail_host_allocations=true;
            PrefixAsyncCapture::reap();
            prefix_test_fail_host_allocations=false;
            require(resource_calls-retire_calls==(automatic?0U:5U),"original retirement: three events, stream, staging");
            if(automatic) prefix_test_fail_host_allocations=true;
            d.reset();
            prefix_test_fail_host_allocations=false;
            require(resource_calls-retire_calls==(automatic?0U:(commit?5U:7U)),"cancelled original payloads freed after final lease");
        }
        require(store->counters().inserted==1,"both paths publish only committed capture");
    }
    // Attach failure leaves no reusable partial pool and can be retried at startup.
    for(bool fail_stream:{false,true}) {
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,{},0},hip_pinned_prefix_capture_buffer);
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        HipPrefixStateCacheSession adapter(store,id,2,2,ModelContract{},layout,memory);
        const auto streams=live_streams;
        allocation_failure=!fail_stream;stream_failure=fail_stream;
        bool failed=false;
        try {(void)adapter.attach_capture_pool(true);} catch(const PrefixCacheFatalError&) {failed=true;}
        allocation_failure=stream_failure=false;
        require(failed && !store->capture_resources && live_streams==streams,"failed attach rolls back partial GPU resources");
        const auto before=resource_calls;
        adapter.attach_capture_pool();
        require(resource_calls==before && !store->capture_resources && !store->counters().pool_bytes,
            "explicit attachment creates no startup pool");
        adapter.attach_capture_pool(true);
        require(bool(store->capture_resources),"attach retry succeeds outside request path");
    }
    // Invoke real capture hooks between numerical units; compare every descriptor and priming flag.
    for(const auto t:{2048U,8192U}) for(const auto n:{1280U,6000U,8300U,10300U,10400U,8192U,32768U}) for(bool roomy:{false,true}) {
        operations.clear();
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{1048576,{},0},[](std::uint64_t bytes) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[bytes]{},std::default_delete<std::byte[]>()),bytes};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=n;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        HipPrefixStateCacheSession adapter(store,id,t,std::max(4096U,t),ModelContract{},layout,memory);
        std::vector<std::byte> rows(n);
        adapter.add_rows(reinterpret_cast<std::uintptr_t>(rows.data()),1,0);
        if(roomy) adapter.set_staging_cap(2U*t);
        QwenTiledPlanConfig config;config.rows=t;config.block={t,QwenPlanOutput::terminal};
        config.block.execution=1;config.block.kernel=2;config.priming=255;config.schedule=12;
        config.ladder={1024,256,64,16};
        for(const auto r:config.ladder) config.tails.push_back({r,QwenPlanOutput::terminal});
        adapter.set_plan(config,n,true);
        adapter.attach_capture_pool(true);
        QwenPlanCursor off,on;std::uint64_t position=0;
        SerialSessionContext session;session.state_layout_identity=1;
        const std::vector<std::uint32_t> prompt(n,1);
        unsigned admitted=0;
        while(auto unit=qwen_prompt_next(config,on,position,n)) {
            const auto reference=qwen_prompt_next(config,off,position,n);
            require(reference && *unit==*reference,"capture hooks preserve complete unit sequence and priming flags");
            session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,position,unit->end,1,2,1);
            position=unit->end;session.next_position=session.committed_tokens=position;
            auto d=adapter.try_prepare_capture(session,prompt,producer,true,ordered,n-1U);
            if(d) {
                if(d->publish) ++admitted;
                d->submit_predictor(predictor);pump(producer);pump(predictor);pump(d->download);
                d->commit(true);PrefixAsyncCapture::reap();
            }
            if(position==n) break;
        }
        const auto expected=t==2048U?(n==32768U?8U:n==8192U?2U:n==8300U || n==6000U?1U:n>=10300U?2U:0U):
            (n==32768U?3U:n==10300U || n==10400U?1U:0U);
        if(admitted!=expected || store->counters().inserted!=expected) {
            const auto c=store->counters();
            std::cerr<<"capture count mismatch n="<<n<<" T="<<t<<" roomy="<<roomy<<" admitted="<<admitted
                <<" expected="<<expected<<" inserted="<<c.inserted<<" no_slot="<<c.capture_skipped_no_slot
                <<" cap_skips="<<c.staging_cap_skips<<" budget_skips="<<c.budget_skips<<'\n';
        }
        require(admitted==expected && store->counters().inserted==expected,"default and enlarged-cap capture opportunity counts");
        const auto cross=t==2048U?(n==8300U || n==10300U || n==10400U?1U:0U):(n==8300U?1U:0U);
        require(store->counters().capture_skips_cross_stream==cross,"planner refuses tail/window/serial transitions even with matching handles");
        if(!roomy) std::cout<<"capture opportunities n="<<n<<" T="<<t<<" selected="<<admitted
            <<" private="<<store->counters().row_prefetches<<" cross_stream_skips="<<cross<<'\n';
    }
    // Credit failure before submission is a skip. Stream errors after submission quarantine.
    for(unsigned scenario=0;scenario<4;++scenario) {
        operations.clear();
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{scenario==0?1U:8192U,{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        HipPrefixStateCacheSession adapter(store,id,2,2,ModelContract{},layout,memory);
        std::array<std::byte,8> target{};
        adapter.add_rows(reinterpret_cast<std::uintptr_t>(target.data()),1,0);
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        config.block.execution=1;config.block.kernel=2;adapter.set_plan(config,8,true);
        adapter.attach_capture_pool(true);
        SerialSessionContext session;session.state_layout_identity=1;session.next_position=session.committed_tokens=2;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,0,2,1,2,1);
        auto d=adapter.try_prepare_capture(session,tokens,producer,true,ordered);
        if(scenario==0) {require(!d,"no credit skip");continue;}
        const auto before=PrefixDmaQuarantine::count.load();
        bool fatal=false;
        try {
            if(scenario==1) {copy_error=hipErrorUnknown;d->submit_predictor(predictor);}
            else {
                d->submit_predictor(predictor);
                if(scenario==2) {drain_error=hipErrorUnknown;PrefixAsyncCapture::reap();}
                else {d->fatal();throw PrefixCaptureHipError(hipErrorUnknown,"incumbent drain failure");}
            }
        } catch(const PrefixCaptureHipError&) {fatal=true;}
        require(fatal && PrefixDmaQuarantine::count.load()==before+1,"fatal error quarantines all descriptor backing");
        require(store->counters().inserted==0,"failed copy/drain never publishes");
        PrefixAsyncCapture::slot.reset();operations.clear();drain_error=copy_error=hipSuccess;
    }
    snapshot_shim=false;
    set_prefix_batch_certification_enabled(false);
}

// Real adapter, real 4096-row cadence and a deferred HIP shim. Cancellation resets
// live storage after a completed checkpoint, including an in-unit partial write.
void cancelled_checkpoint_retry_tests() {
    using namespace strix;
    using namespace strix::decode_cli;
    snapshot_shim=true;drain_error=copy_error=hipSuccess;queued=false;
    const auto producer=reinterpret_cast<hipStream_t>(2),predictor=reinterpret_cast<hipStream_t>(3);
    const PrefixCaptureOrder order{producer,predictor,predictor};
    const auto id=sha256_bytes(std::as_bytes(std::span("cancel-retry",12)));
    constexpr std::uint64_t n=8000,t=1024,p=4096;
    std::vector<std::uint32_t> tokens(n);
    for(std::size_t i=0;i<tokens.size();++i) tokens[i]=static_cast<std::uint32_t>(i%251U);
    QwenTiledPlanConfig config;config.rows=t;config.schedule=12;
    config.block={t,QwenPlanOutput::terminal,1,2};
    config.tails={{256,QwenPlanOutput::terminal,3,4},{64,QwenPlanOutput::terminal,5,6},{16,QwenPlanOutput::terminal,7,8}};
    config.ladder={256,64,16};config.generations_per_row[0]=1;
    QwenBatchConfig batch;batch.groups=false;batch.cohort=false;batch.nest_rows=0;
    const auto counted=[](PrefixStateCache& store,std::string_view reason) {
        const auto stats=store.counters();
        for(std::size_t i=0;i<prefix_certificate_decline_names.size();++i)
            if(prefix_certificate_decline_names[i]==reason) return stats.certificate_declines[i];
        return std::uint64_t{0};
    };
    for(bool pooled:{false,true}) for(bool inside_unit:{false,true}) for(bool solo_suffix:{false,true}) {
        operations.clear();
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{1U<<20,{},0},hip_pinned_prefix_capture_buffer);
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=n;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        std::vector<std::byte> rows(n),predictor_rows(n);std::byte held{41},fixed{31};
        auto route_batch=batch;route_batch.groups=!solo_suffix;
        const auto bind=[&](HipPrefixStateCacheSession& cache) {
            cache.add_rows(reinterpret_cast<std::uintptr_t>(rows.data()),1,0);
            cache.add_rows(reinterpret_cast<std::uintptr_t>(predictor_rows.data()),1,1);
            cache.add_fixed_device(reinterpret_cast<std::uintptr_t>(&fixed),1);
            cache.add_fixed_device(reinterpret_cast<std::uintptr_t>(&held),1,"mtp.held");
            cache.set_plan(config,n,true,2,route_batch);
        };
        auto capture=std::make_unique<HipPrefixStateCacheSession>(store,id,t,p,ModelContract{},layout,memory);
        bind(*capture);if(pooled) capture->attach_capture_pool(true);
        SerialSessionContext session;session.state_layout_identity=1;
        require(capture->begin_request(session,tokens)==0 && capture->request().refusal=="prefix-no-compatible-checkpoint",
            "cold request has a named miss");
        for(std::uint64_t end=t;end<=p;end+=t) {
            for(auto i=end-t;i<end;++i) {rows[i]=std::byte(tokens[i]);predictor_rows[i]=std::byte(tokens[i]^31U);}
            session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,end-t,end,1,2,1);
            session.next_position=session.committed_tokens=end;session.state.generations[0]=end;
            auto d=capture->try_prepare_capture(session,tokens,producer,true,order,n-1U);
            require(bool(d),"every pre-cancel full-T boundary prepares row prefetch or checkpoint");
            // A synthetic old preparation time demonstrates that even an
            // unselected row-prefetch descriptor contributes elapsed overlap.
            if(end==t) d->prepared-=std::chrono::seconds(2);
            pump(producer);d->submit_predictor(predictor);pump(predictor);pump(d->download);d->commit(true);
            PrefixAsyncCapture::reap();
        }
        require(store->counters().inserted==1 && capture->cached_boundary(tokens)==p,
            "4096 checkpoint is stored before cancellation, not merely a private row chain");
        require(capture->request().captured_checkpoints==1 && capture->request().capture_ms>=2000,
            "capture_ms includes unselected descriptor lifetime, not only checkpoint DMA time");
        const auto saved_rows=rows,saved_predictor=predictor_rows;
        const auto saved_session=session;
        // Cancel either during the next native unit or between units. A later
        // partial write and the slot reset must never mutate the stored snapshot.
        if(inside_unit) {rows[p]=std::byte{99};fixed=held=std::byte{99};}
        capture->before_target_reset();capture.reset();session={};
        std::fill(rows.begin(),rows.end(),std::byte{77});
        std::fill(predictor_rows.begin(),predictor_rows.end(),std::byte{77});fixed=held=std::byte{77};
        HipPrefixStateCacheSession retry(store,id,t,p,ModelContract{},layout,memory);bind(retry);
        std::uint64_t frontier=0;retry.bind_restore_runtime([&](std::uint64_t end){frontier=end-1U;});
        session.state_layout_identity=1;
        if(!solo_suffix) {
            const auto before=copies;
            require(retry.begin_request(session,tokens)==0 && retry.request().refusal=="batch-suffix-not-solo" &&
                counted(*store,"batch-suffix-not-solo")==1 && copies==before && !session.next_position && !frontier,
                "original grouped 8000-token retry finds its stored checkpoint and names the suffix refusal");
            continue;
        }
        require(retry.begin_request(session,tokens)==p && retry.request().restored_position==p && frontier==p-1U,
            "cancelled prefill retry restores 4096 and predictor frontier 4095");
        require(std::equal(rows.begin(),rows.begin()+p,saved_rows.begin()) &&
            std::equal(predictor_rows.begin(),predictor_rows.begin()+p-1U,saved_predictor.begin()) &&
            rows[p]==std::byte{77} && predictor_rows[p-1U]==std::byte{77} && fixed==std::byte{31} && held==std::byte{41} &&
            session.state.generations==saved_session.state.generations &&
            session.arithmetic_lineage.intervals==saved_session.arithmetic_lineage.intervals,
            "retry restores exact checkpoint bytes and metadata without writing beyond the frontier");
        const auto checkpoint=store->find_longest(retry.active_identity(),tokens,t,p);
        require(checkpoint && checkpoint->tokens()==std::vector<std::uint32_t>(tokens.begin(),tokens.begin()+p),
            "all 4096 stored token IDs equal the retry prefix");
        // Reproduce the reported configuration: its final 64-row tail can group.
        auto grouped=batch;grouped.groups=true;
        HipPrefixStateCacheSession refused(store,id,t,p,ModelContract{},layout,memory);bind(refused);
        refused.set_plan(config,n,true,2,batch,false);session={};session.state_layout_identity=1;
        require(refused.begin_request(session,tokens)==0 && refused.request().refusal=="prefix-restore-disabled" &&
            counted(*store,"prefix-restore-disabled")==1,"service-disabled restore names its refusal");
        // Keep the same stored namespace and change only suffix eligibility via
        // nesting_possible, which deliberately is not part of the namespace.
        refused.set_plan(config,n,true,2,batch,true,true);session={};session.state_layout_identity=1;
        const auto before=copies;
        require(refused.begin_request(session,tokens)==0 && refused.request().refusal=="batch-suffix-not-solo" &&
            counted(*store,"batch-suffix-not-solo")==1 && copies==before && !session.next_position,
            "stored but non-solo suffix is a named zero-write miss");
        require(!qwen_suffix_solo_class(config,n,p,grouped,false),"8000-token grouped suffix reproduces original refusal");
        const auto mutable_checkpoint=std::const_pointer_cast<PrefixCacheCheckpoint>(checkpoint);
        mutable_checkpoint->producer_certificate[4]^=1;
        refused.set_plan(config,n,true,2,batch);session={};session.state_layout_identity=1;
        require(refused.begin_request(session,tokens)==0 && refused.request().refusal=="prefix-producer-plan-mismatch" &&
            counted(*store,"prefix-producer-plan-mismatch")==1 && copies==before,
            "corrupt stored certificate is a named zero-write miss");
        mutable_checkpoint->producer_certificate[4]^=1;
        mutable_checkpoint->metadata.clear();session={};session.state_layout_identity=1;
        require(refused.begin_request(session,tokens)==0 && refused.request().refusal=="prefix-checkpoint-preflight" &&
            counted(*store,"prefix-checkpoint-preflight")==1 && copies==before,
            "corrupt stored metadata is a named zero-write miss");
    }
    operations.clear();snapshot_shim=false;
}

// Real adapter + deferred HIP shim: namespace, restore preflight, hook routing and no-wait tripwire.
void batch_snapshot_tests() {
    using namespace strix;
    using namespace strix::decode_cli;
    snapshot_shim=true;drain_error=copy_error=hipSuccess;allocation_failure=false;
    const auto id=sha256_bytes(std::as_bytes(std::span("batch-dma",9)));
    const auto producer=reinterpret_cast<hipStream_t>(2),predictor=reinterpret_cast<hipStream_t>(3),foreign=reinterpret_cast<hipStream_t>(4);
    const auto tier=std::filesystem::temp_directory_path()/
        ("strix-batch-prefix-dma-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    for(unsigned scenario=0;scenario<6;++scenario) {
        operations.clear();
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,scenario==5?tier:std::filesystem::path{},0},[](std::uint64_t n) {
            return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};
        });
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        std::array<std::byte,8> rows{};std::byte fixed{31};
        HipPrefixStateCacheSession adapter(store,id,2,scenario==5?8U:2U,ModelContract{},layout,memory);
        adapter.add_rows(reinterpret_cast<std::uintptr_t>(rows.data()),1,0);
        adapter.add_fixed_device(reinterpret_cast<std::uintptr_t>(&fixed),1);
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        config.block.execution=1;config.block.kernel=2;
        QwenBatchConfig batch;batch.groups=false;batch.cohort=false;
        adapter.set_plan(config,8,true);
        const auto single=adapter.active_identity();
        adapter.set_plan(config,8,true,2,batch);
        const auto identity=adapter.active_identity();
        require(single!=identity,"batch and single namespaces differ");
        auto changed=batch;changed.cancel=true;adapter.set_plan(config,8,true,2,changed);
        require(identity!=adapter.active_identity(),"scheduler settings in batch namespace");
        for(unsigned field=0;field<6;++field) {
            auto scheduler=batch;
            switch(field) {
                case 0:scheduler.service_ms=5;break;
                case 1:scheduler.lookahead++;break;
                case 2:scheduler.prompt_share=0.5;break;
                case 3:scheduler.short_rows++;break;
                case 4:scheduler.interleave=!scheduler.interleave;break;
                default:scheduler.shared="none";break;
            }
            adapter.set_plan(config,8,true,2,scheduler);
            require(adapter.active_identity()!=identity,"resolved scheduler value binds batch namespace");
        }
        adapter.set_plan(config,8,true,2,batch);
        std::uint64_t bridge_frontier=0,slot_frontier=0;
        adapter.set_restore_hook([&](std::uint64_t p){bridge_frontier=p;});
        SerialSessionContext session;session.state_layout_identity=1;
        session.next_position=session.committed_tokens=2;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,0,2,1,2,1);
        const std::vector<std::uint32_t> tokens(8,1);
        const int tag=0,other_tag=0;
        adapter.set_retention(false);const auto no_store=store->counters().no_store_skips;
        require(!adapter.try_prepare_capture(session,tokens,producer,true,{producer,predictor,predictor},7,{},{},&tag) && operations.empty(),"batch no-store prepares nothing");
        require(store->counters().no_store_skips==no_store+1,"batch no-store counted");
        adapter.set_retention(true);
        if(scenario==5) adapter.request_capture(2); // off cadence, unchanged cold full-T boundary
        auto d=adapter.try_prepare_capture(session,tokens,producer,true,{producer,predictor,predictor},7,{},{},&tag);
        require(bool(d),"batch tagged snapshot prepared");
        require(d->publish,"requested boundary publishes fixed state without changing the producer plan");
        const auto before_waits=waits;
        PrefixAsyncCapture::before_writer(&other_tag,foreign);
        require(!d->poisoned,"another slot does not poison the snapshot");
        if(scenario==0 || scenario==3 || scenario==4) {
            PrefixAsyncCapture::before_writer(&tag,foreign);
            // Execute a genuinely foreign overwrite before the target snapshot reads.
            operations[foreign].push_back([&]{fixed=std::byte{99};return true;});pump(foreign);
        } else if(scenario==1) PrefixAsyncCapture::before_writer(&tag,producer);
        pump(producer);
        if(scenario==2) PrefixAsyncCapture::before_writer(&tag,foreign);
        d->submit_predictor(predictor);pump(predictor);pump(d->download);
        d->commit(true);
        if(scenario==3) d->commit(false);
        if(scenario==4) {adapter.before_target_reset();d->cancelled=true;d->commit(true);}
        PrefixAsyncCapture::reap();
        const bool poisoned=scenario==0 || scenario==3 || scenario==4;
        require(!PrefixAsyncCapture::slot && store->counters().inserted==(poisoned?0U:1U),"sticky tripwire prevents publication and retires");
        require(store->counters().capture_skips_tripwire==(poisoned?1U:0U) && waits==before_waits,"tripwire is once-only and never waits");
        require(store->counters().staging_bytes==0,"tripwire retirement releases staging accounting");
        if(poisoned) {
            const auto chain=d->row_chain.lock();
            require(chain && chain->empty(),"poisoned snapshot cannot publish private row segments");
            continue;
        }
        store->flush(); // Complete disk publication before the later deliberate RAM corruption.
        const auto digests=prefix_cache_block_digests(identity,tokens,2,1);
        const auto checkpoint=store->find_longest(identity,tokens,2,2);
        require(checkpoint && checkpoint->producer_certificate==adapter.execution_certificate(2),"batch certificate stored");
        // Another slot has the same identity but different live storage and predictor runtime.
        std::array<std::byte,8> other_rows{};std::byte other_fixed{};
        HipPrefixStateCacheSession other(store,id,2,2,ModelContract{},layout,memory);
        other.add_rows(reinterpret_cast<std::uintptr_t>(other_rows.data()),1,0);
        other.add_fixed_device(reinterpret_cast<std::uintptr_t>(&other_fixed),1);
        other.set_plan(config,8,true,2,batch);
        unsigned invalidations=0;bool target_m1_valid=true,predictor_m1_valid=true;
        other.set_before_restore_hook([&]{++invalidations;target_m1_valid=predictor_m1_valid=false;});
        other.set_restore_hook([&](std::uint64_t p){bridge_frontier=p;});
        other.bind_restore_runtime([&](std::uint64_t p){
            require(!target_m1_valid && !predictor_m1_valid,"M1 invalidates before rebound runtime frontier restore");
            slot_frontier=p;
        });
        other.set_retention(false); // no conversation: batch no-store still permits a validated hit
        SerialSessionContext fresh;fresh.state_layout_identity=1;
        require(other.begin_request(fresh,tokens)==2 && slot_frontier==2 && bridge_frontier==0 && other_fixed==std::byte{31},"slot-0 runtime hook rebound; cross-slot reuse");
        require(other.continuation()==qwen_full_t_continuation(config,8,2),"whole-prompt cold cursor installed");
        require(invalidations==1,"M1 pre-restore hook survives first runtime rebind");
        target_m1_valid=predictor_m1_valid=true;slot_frontier=0;
        other.bind_restore_runtime([&](std::uint64_t p){
            require(!target_m1_valid && !predictor_m1_valid,"M1 invalidation survives repeated named/slot runtime rebind");
            slot_frontier=p+1;
        });
        fresh={};fresh.state_layout_identity=1;
        require(other.begin_request(fresh,tokens)==2 && invalidations==2 && slot_frontier==3 && !bridge_frontier,
            "rebound restore retains independent M1 invalidation hook");
        // Restore is a zero-write miss during parked service admissions.
        other.set_plan(config,8,true,2,batch,false);fresh={};fresh.state_layout_identity=1;other_fixed=std::byte{77};
        const auto pending_ops=operations[reinterpret_cast<hipStream_t>(1)].size();
        require(other.begin_request(fresh,tokens)==0 && fresh.next_position==0 && other_fixed==std::byte{77} &&
            operations[reinterpret_cast<hipStream_t>(1)].size()==pending_ops,"service admission miss performs no writes");
        other.set_plan(config,8,true);fresh={};fresh.state_layout_identity=1;
        require(other.begin_request(fresh,tokens)==0 && !store->find_segment(prefix_cache_block_digests(single,tokens,2,1)[0]),"single route cannot share batch segments");
        // Corrupt proof fails before any restore enqueue or mutation.
        auto corrupt=std::const_pointer_cast<PrefixCacheCheckpoint>(checkpoint);corrupt->producer_certificate[4]^=1;
        other.set_plan(config,8,true,2,batch);fresh={};fresh.state_layout_identity=1;
        require(other.begin_request(fresh,tokens)==0 && other_fixed==std::byte{77},"batch corrupt proof is a zero-write miss");
        require(digests[0]==checkpoint->digest,"cross-slot token namespace stable");
        if(scenario==5) {
            auto restarted=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,tier,0},[](std::uint64_t n) {
                return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n?n:1]{},std::default_delete<std::byte[]>()),n};
            });
            std::array<std::byte,8> restored_rows{};restored_rows.fill(std::byte{77});
            std::byte restored_fixed{77};
            HipPrefixStateCacheSession restored(restarted,id,2,8,ModelContract{},layout,memory);
            restored.add_rows(reinterpret_cast<std::uintptr_t>(restored_rows.data()),1,0);
            restored.add_fixed_device(reinterpret_cast<std::uintptr_t>(&restored_fixed),1);
            restored.set_plan(config,8,true,2,batch);
            std::uint64_t frontier=0;restored.bind_restore_runtime([&](std::uint64_t p){frontier=p;});
            restored.set_retention(false);
            SerialSessionContext cold;cold.state_layout_identity=1;
            require(restored.begin_request(cold,tokens)==2 && frontier==2 && restored_fixed==std::byte{31},
                "fresh store certified adapter restores persisted fixed state and predictor hook");
            require(restored_rows[0]==rows[0] && restored_rows[1]==rows[1] && restored_rows[2]==std::byte{77} &&
                cold.state.valid==session.state.valid && cold.state.generations==session.state.generations &&
                cold.arithmetic_lineage.intervals==session.arithmetic_lineage.intervals &&
                cold.arithmetic_lineage.history_identity==session.arithmetic_lineage.history_identity &&
                cold.arithmetic_lineage.tainted==session.arithmetic_lineage.tainted &&
                cold.next_position==session.next_position && cold.committed_tokens==session.committed_tokens,
                "persisted adapter restores exact rows and complete committed metadata");
            require(restarted->counters().disk_hits==1 && restarted->counters().host_checkpoints==0,
                "fresh adapter restart is a disk-only no-store hit");
        }
    }
    std::filesystem::remove_all(tier);
    operations.clear();snapshot_shim=false;
    set_prefix_batch_certification_enabled(false);
}

// Compare the old combined boundary against the automatic two-part schedule.
// Deferred streams allow target writers/downloads to run while priming is still absent.
void split_snapshot_tests() {
    using namespace strix;
    using namespace strix::decode_cli;
    snapshot_shim=true;drain_error=copy_error=hipSuccess;
    const auto producer=reinterpret_cast<hipStream_t>(2),predictor=reinterpret_cast<hipStream_t>(3);
    const PrefixCaptureOrder order{producer,predictor,predictor};
    const auto id=sha256_bytes(std::as_bytes(std::span("split",5)));
    const std::vector<std::uint32_t> tokens(8,1);
    std::vector<std::byte> reference;
    std::vector<std::uint64_t> offsets;
    for(unsigned scenario=0;scenario<7;++scenario) {
        operations.clear();
        auto store=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,{},0},hip_pinned_prefix_capture_buffer);
        SerialSessionStateLayout layout;layout.layout_identity=1;layout.context_tokens=8;
        SessionStateMemoryBinding memory;memory.layout_identity=1;
        HipPrefixStateCacheSession cache(store,id,2,2,ModelContract{},layout,memory);
        std::array<std::byte,8> target{},mtp{};
        std::array<std::byte,256> fixed{},held{};
        target[0]=std::byte{11};target[1]=std::byte{12};fixed.fill(std::byte{31});
        cache.add_rows(reinterpret_cast<std::uintptr_t>(target.data()),1,0);
        cache.add_rows(reinterpret_cast<std::uintptr_t>(mtp.data()),1,1);
        cache.add_fixed_device(reinterpret_cast<std::uintptr_t>(fixed.data()),fixed.size());
        cache.add_fixed_device(reinterpret_cast<std::uintptr_t>(held.data()),held.size(),"mtp.held");
        QwenTiledPlanConfig config;config.rows=2;config.block={2,QwenPlanOutput::terminal};
        config.block.execution=1;config.block.kernel=2;config.priming=255;
        cache.set_plan(config,8,true);
        SerialSessionContext session;session.state_layout_identity=1;
        session.next_position=session.committed_tokens=2;
        session.arithmetic_lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,0,2,1,2,1);
        const bool split=scenario!=0;
        if(split) cache.attach_capture_pool(true);
        else require(!cache.try_prepare_target_capture(session,tokens,producer,true,order) && operations.empty(),
            "explicit route declines target-only API without submissions");
        cache.set_retention(false);
        require(!cache.try_prepare_target_capture(session,tokens,producer,true,order) && operations.empty(),
            "split no-store performs no payload work");
        cache.set_retention(true);
        auto short_session=session;short_session.next_position=short_session.committed_tokens=1;
        require(!cache.try_prepare_target_capture(short_session,tokens,producer,true,order) && operations.empty(),
            "split partial-T path performs no payload work");
        const auto hot_calls=resource_calls, before_waits=waits;
        auto d=split?cache.try_prepare_target_capture(session,tokens,producer,true,order):
            HipPrefixStateCacheSession::Capture{};
        if(split) {
            require(bool(d) && resource_calls==hot_calls,"split prepares without HIP allocation");
            require(!d->metadata_complete && !d->submitted.load(),"target part has no complete certificate or publication handoff");
            if(scenario==5) {
                d->cancel_target();PrefixAsyncCapture::reap();
                require(PrefixAsyncCapture::slot && store->counters().pool_free_bytes<store->counters().pool_bytes,
                    "cancel between parts retains leases while target DMA is pending");
                pump(producer);pump(d->download);PrefixAsyncCapture::reap();d.reset();
                require(!PrefixAsyncCapture::slot && store->counters().pool_free_bytes==store->counters().pool_bytes,
                    "pending cancellation recycles only after target download finishes");
                continue;
            }
            pump(producer);pump(d->download);
            require(d->checkpoint->segments[0]->payload.data.get()[0]==std::byte{11} &&
                d->checkpoint->fixed.data.get()[0]==std::byte{31},"target D2H completes before predictor is submitted");
            target.fill(std::byte{99});fixed.fill(std::byte{99});
            PrefixAsyncCapture::reap();
            require(store->counters().inserted==0,"target completion alone never publishes");
            if(scenario==2) {
                d->cancel_target(); // cancellation between the two parts
            } else if(scenario==3 || scenario==4) {
                if(scenario==4) session.next_position=4;
                require(!cache.complete_target_capture(d,session,scenario==4,predictor),
                    "post-prime incomplete frontier or changed boundary refuses capture");
            }
            if(scenario>=2 && scenario<=4) {
                pump(d->download);PrefixAsyncCapture::reap();d.reset();
                require(!PrefixAsyncCapture::slot && store->counters().inserted==0 &&
                    store->counters().pool_free_bytes==store->counters().pool_bytes && store->counters().staging_bytes==0,
                    "cancelled/failed second part returns all pool credits after DMA");
                require(waits==before_waits,"cancelling split adds no host wait");
                continue;
            }
        }
        // The incumbent prime and held update finish at this boundary.
        mtp[0]=std::byte{21};held.fill(std::byte{41});session.execution_identity=17;
        if(split) require(cache.complete_target_capture(d,session,true,predictor),"complete predictor frontier accepted post-prime");
        else {
            d=cache.try_prepare_capture(session,tokens,producer,true,order);
            require(bool(d) && operations[d->download].empty(),"explicit combined path queues no download before predictor");
            d->submit_predictor(predictor);
        }
        pump(producer);pump(predictor);pump(d->download);
        require(store->counters().inserted==0,"both completed groups still need native commit");
        d->commit(true);PrefixAsyncCapture::reap();
        require(store->counters().inserted==1 && !PrefixAsyncCapture::slot,"complete split checkpoint published after commit");
        std::vector<std::byte> bytes=d->checkpoint->metadata;
        const auto& c=*d->checkpoint;
        const auto append=[&](const auto& value) {
            const auto span=std::as_bytes(std::span(&value,1));bytes.insert(bytes.end(),span.begin(),span.end());
        };
        for(auto word:c.producer_certificate) append(word);
        for(auto byte:c.identity.bytes) append(byte);
        for(auto byte:c.digest.bytes) append(byte);
        append(c.position);append(c.block_rows);
        for(const auto& segment:c.segments) {
            for(auto byte:segment->digest.bytes) append(byte);
            for(auto byte:segment->parent.bytes) append(byte);
            bytes.insert(bytes.end(),segment->payload.data.get(),segment->payload.data.get()+segment->payload.bytes);
        }
        bytes.insert(bytes.end(),c.fixed.data.get(),c.fixed.data.get()+c.fixed.bytes);
        std::vector<std::uint64_t> framing;
        for(const auto& slice:d->slices) {framing.push_back(slice.offset);framing.push_back(slice.bytes);framing.push_back(slice.predictor);}
        if(!split) {reference=bytes;offsets=framing;}
        else require(bytes==reference && offsets==framing,"split checkpoint bytes, offsets, digests, metadata and certificate equal pre-split boundary");
        require(waits==before_waits,"split completion adds no host wait");
    }
    snapshot_shim=false;operations.clear();
}
