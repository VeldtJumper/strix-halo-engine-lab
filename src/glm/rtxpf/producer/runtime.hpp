#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "verified_manifest.hpp"
namespace strix::glm::rtxpf::producer {
// Decode expert-tier selection and tierstream startup budget (decode only).
struct DecodeOptions {
    std::string expert_tier="tierstream"; // the product cache (default); "reference": uncached backend, only inside the untimed compare
    std::string unit_file;               // tierstream disk tier: the GLM unit file (manifest beside it)
    // Multi-disk reads (ss_options.weight_dirs): ';'-separated directories holding ts-stripe copies of
    // the unit file (glm_unit_file plan --stripe); file-tier misses and make_resident read the chunks
    // from every drive. Empty: the primary only. drive_readers: per volume, 1..32 (0 = library default 4).
    std::string weight_dirs;unsigned drive_readers=0;
    // Three-tier file reads in flight (ss_options.dg_disk_queue: 1 = one at a time, as before; 2..64,
    // at most tier_staging since each read holds a staging buffer) and clean eviction (dg_clean_evict:
    // a device victim with no free pinned block goes to its file with no D2H copy and displaces no
    // pinned unit). Placement / IO only: same bytes. Defaults keep today's behaviour until measured.
    unsigned tier_disk_queue=1;bool tier_clean_evict=false;
    // Two-stage file prefetch (tierstream dg_landing_*): predictions of file-homed units are read into
    // landing_blocks pinned blocks (never a GPU slot); a demand copies from there. Needs a file tier and
    // tier_disk_queue > 1; billed as pinned host units. Placement only (same bytes); 0 = off until measured.
    unsigned tier_landing_blocks=0,tier_landing_max_queued=0,tier_landing_ttl=0;
    // tierstream dg_landing_chunk (SS_DG_LANDING_YIELD): landing reads run in chunks that yield their reader to a
    // demand read (0 = 4 MiB chunks; 0xFFFFFFFF = whole, unchunked landing reads). Used only with landing blocks.
    std::uint32_t tier_landing_chunk=0;
    double vram_total_gib=-1,ram_total_gib=-1; // whole-device caps (< 0: none = the device room / host above the floor)
    double tier_vram_gib=-1,tier_ram_gib=-1; // < 0: everything left after prefill residency / the host floor
    unsigned tier_spare_slots=16,tier_host_spares=8,tier_staging=8;
    // Arena spares (retained misses): max(tier_spare_slots, permille/1000 of the arena);
    // 0 = tier_spare_slots exactly (an explicit --tier-spare-slots sets it).
    unsigned tier_spare_permille=50;
    // Placement, the DS way (expert_tier.hpp seed_order): residency file of the previous
    // run ("" = <unit_file>.residency, "none" = off; written back after every request),
    // routes census files (DS routes format; comma-separated), the round-robin fallback.
    std::string expert_residency,expert_routes;
    std::string routes_out;              // record each decoded token's routes (DS routes format; receipts carry them too)
    // Census runs: also record every prefill row's routes into routes_out (one line per prompt row,
    // all routed layers), so a placement census can come from prompts other than the benchmark's.
    // One D2H copy per tile after its kernels (untimed census use; off by default).
    bool routes_prefill=false;
    std::string tier_policy="auto";      // boundary victims: lru | scored | auto (SCORED_AUTO, the DS default)
    bool tier_predict=false;             // overlapped next-layer prefetch (ss_dg_enqueue_predict), off by default
    // dg_flush_ready: the library flushes its demand stream after each layer's ready
    // signal, so WDDM submits the ready write at once instead of batching it behind
    // later work (a device wait on that layer could otherwise outlive the service's
    // flag deadline). Submission only: outputs unchanged. On by default (profile row flush_ready).
    bool tier_flush_ready=true;
    // Exact decode levers (profile rows, pending until a window proves equal digests):
    // routed MoE launches only the expert groups holding routed rows (device compaction);
    // shared experts and dense FFNs keep a persistent FP16 copy decoded once at startup
    // (a named whole-device plan part; dropped when the plan cannot afford it).
    bool skip_empty_expert_groups=false,persistent_dense_fp16=false;
    // Exact one-row decode kernels (profile row decode_row_kernels, pending until a window shows equal
    // digests): routed/shared/dense MoE as register-decoded MMA chains (row_moe), the hyper-connection
    // mix chains from shared memory (row_hc), the router over 9 CTAs + a parallel top 8 (row_router),
    // the KDA qkv/out and DSA kva/qb/out trellis projections as the same MMA chains (row_kda, row_dsa).
    bool row_moe=false,row_hc=false,row_router=false,row_kda=false,row_dsa=false;
    // Round 2 of those kernels (subset token r2, opt-in; same values): A loads before the weight decode, dense FP16
    // strips for the KDA ordinary projections, FP32 index row chains, 4-wide hc reads, parallel prefix/compaction.
    bool row_r2=false;
    // MTP self-speculation (profile row mtp_self_spec, pending until a window shows equal digests): the service
    // opens with 2-row decode buffers and 16 exported ids per layer; decode_spec verifies [pending, MTP draft]
    // in one 2-row step and commits 1 or 2 tokens. Plain decode_one stays available on the same load.
    bool mtp_spec=false;
    bool spec_always=false; // mtp_self_spec "always": verify every step (no adaptive plain steps)
    // Prompt-lookup drafts for the same verify (profile row lookup_drafts, pending; needs mtp_spec): up to 3
    // drafts from the session's own tokens when its suffix occurred before, else the MTP draft.
    bool spec_lookup=false;
    // After each prefill through the cache, the open decode placement is made resident again
    // (placement only); without it the last prefilled layers own the arena (profile row).
    bool restore_decode_residency=true;
    // Prefill overlap: the next routed layer's make_resident runs while this layer's kernels are in flight (held
    // layers, no token; two layers kept, only when the arena has room: PrefillResidency::two_layers, or after
    // PrefillSpareTarget lowers the spares). Copies vs kernels reorder only: same bytes. Off in a default-constructed
    // DecodeOptions (library callers); the service sets it from the profile row prefill_overlap, which is on in every
    // profile (--prefill-overlap 0 turns it off), as it does for the row kernels.
    bool prefill_overlap=false;
    bool disk_prefetch=false;
    bool reference_compare=false;        // tierstream: also keep the uncached reference tier for same-window comparison
    std::string unit_manifest_sha256;    // pinned digest of PRIMARY.units.json (trusted manifest)
};
// decode_row_kernels value: 0/off, 1/on (all five), 2 (all five + r2), or a '+'-separated subset of moe, hc, router,
// kda, dsa, r2 (r2 alone changes nothing: it selects round 2 of the listed kernels).
inline void set_row_kernels_value(DecodeOptions& d,const std::string& v){
    bool* f[]={&d.row_moe,&d.row_hc,&d.row_router,&d.row_kda,&d.row_dsa,&d.row_r2};const char* k[]={"moe","hc","router","kda","dsa","r2"};
    const bool two=v=="2",all=v=="1"||v=="on"||two,none=v=="0"||v=="off";bool on[6]={all,all,all,all,all,two};
    if(!all&&!none)for(std::size_t b=0;b<=v.size();){auto e=v.find('+',b);if(e==std::string::npos)e=v.size();const auto name=v.substr(b,e-b);bool known=false;
        for(unsigned j=0;j<6;++j)if(name==k[j]){on[j]=true;known=true;}
        if(!known)throw std::runtime_error("--decode-row-kernels: 0, 1, 2 or a '+'-separated subset of moe, hc, router, kda, dsa, r2");b=e+1;}
    for(unsigned j=0;j<6;++j)*f[j]=on[j]; // all or nothing: a refused value changes no switch
}
inline std::string row_kernels_name(const DecodeOptions& d){
    const bool f[]={d.row_moe,d.row_hc,d.row_router,d.row_kda,d.row_dsa};const char* k[]={"moe","hc","router","kda","dsa"};
    std::string s;unsigned n=0;for(unsigned j=0;j<5;++j)if(f[j]){s+=(s.empty()?"":"+")+std::string(k[j]);++n;}
    if(n==5)return d.row_r2?"2":"1";
    if(d.row_r2)s+=(s.empty()?"":"+")+std::string("r2");
    return n==0&&!d.row_r2?"0":s;
}
// The library's disk readers that each hold a bounce buffer beyond its own one (none at queue 1).
inline unsigned disk_bounce_readers(const DecodeOptions& d){return d.tier_disk_queue>1?d.tier_disk_queue:0;}
using StateSink=std::function<void(unsigned rank,unsigned layer,unsigned component,std::span<const std::byte>)>;
// Constructor is the first device work. The service validates the UUID and all
// host arguments before constructing it. No CPU emulation hides a CUDA failure.
class Runtime {
    struct Impl;std::unique_ptr<Impl> p_;
public:
    // decode=true additionally loads the root lm_head (decode-capable cache or
    // shards), allocates resident session state and selects the expert tier
    // backend ("reference": uncached per-expert reads; "tierstream": shared cache).
    Runtime(const std::string& path,unsigned capacity,unsigned row_tile,bool from_shards=false,bool graphs=true,unsigned host_floor_gib=8,bool legacy_index=false,bool serial_stage=false,const VerifiedManifest* manifest=nullptr,unsigned max_device_layers=43,unsigned max_pinned_layers=43,bool require_file_tier=false,bool decode=false,const DecodeOptions& decode_options={});
    ~Runtime();
    // Dry tierstream budget plan from shard headers (no hashing, no CUDA calls):
    // the planned VRAM/host split for the given device figures, as JSON.
    static std::string plan_budget(const std::string& model,unsigned capacity,unsigned row_tile,unsigned host_floor_gib,const DecodeOptions& decode_options,std::uint64_t device_total,std::uint64_t device_free);
    std::string health()const;
    // Measured device/host memory now and at peak, ledger charges, caps (JSON).
    std::string memory_report(const std::string& stage);
    // Expert cache diagnostics for failure receipts (JSON; "{}" without the tierstream cache).
    std::string expert_diagnostics();
    // Opt-in decode timeline: every decode_one record then carries "timeline" (CUDA events per phase; outputs unchanged).
    void set_decode_profile(bool on);
    void set_row_kernels(const std::string& value); // decode_row_kernels between requests (no session work in flight)
    std::string row_kernels()const;
    // Writes the cache residency (most recent first) to the residency file; "" when off. Between requests.
    std::string save_residency();
    std::string model_sha256()const;
    std::string timings()const;
    void prefill(std::span<const std::uint32_t> tokens,const std::string& identity,const StateSink&);
    std::string self_check(std::span<const std::uint32_t> tokens,const std::string& identity,unsigned reference_tile=256);
    // Prefill row tile: the last one used (the startup allocation before any request), the
    // one allocated at startup, and a per-request override (valid_schedule, at most the
    // allocated tile; 0 = auto). Auto: every prefill picks select_prefill_tile(rows,
    // allocated, crossover) (schedule.hpp): min(4096, allocated) below the crossover, the
    // allocated tile at and above it. Graph templates are keyed by first:rows and recapture
    // (update or re-instantiate) when the tile changes. The self-check pins its own tiles.
    unsigned prefill_row_tile()const;
    unsigned allocated_prefill_row_tile()const;
    void use_prefill_row_tile(unsigned tile);
    unsigned requested_prefill_row_tile()const;
    void set_prefill_tile_crossover(unsigned rows);
    unsigned prefill_tile_crossover()const;
    // ---- single session decode, identity glm-dec-rtx-native-v1 (draft) ----
    // One pending token: after begin_session(S) the frontier is |S|-1 and S.back()
    // is pending. decode_one consumes it and appends the greedy token. Any
    // failure invalidates the session; begin_session always starts from zero.
    bool decode_capable()const;
    std::string begin_session(std::span<const std::uint32_t> tokens);
    std::string decode_one(bool qualification=false);
    // mtp_self_spec: one speculation cycle emitting 1..min(2, max_tokens) tokens ("tokens", and per token
    // "logits" digests with qualification); a plain step when no draft is available.
    std::string decode_spec(bool qualification,unsigned max_tokens);
    void set_spec_lookup(bool on); // lookup_drafts between requests (needs mtp_self_spec)
    void profiler(bool on);        // cudaProfilerStart/Stop around a traced range (decode command "trace")
    std::string generate(std::span<const std::uint32_t> prompt,unsigned max_new_tokens,std::span<const std::uint32_t> stop_token_ids,bool qualification=false);
    // All 46 layers at the current frontier, exactly the prefill sink order/layout.
    void export_session(const StateSink&);
    // SHA-256 over every KDA block-start checkpoint (state, convolution) and retained open rows.
    std::string replay_digest();
    std::vector<std::uint32_t> session_tokens()const;
    void snapshot_session();
    void restore_session();
    void reset_session();
    // Between tokens only: reset the fetched-expert cache to its open state (no-op without a cache).
    void reset_expert_cache(bool open);
    // The decode residency restore after a prefill through the cache is deferred: it runs before the
    // next decode token, or here (untimed, e.g. after a TTFT prefill). JSON of the restore, "{}" if none.
    std::string restore_decode_residency();
    bool decode_restore_pending()const;
    // "tierstream" or "reference" (with --reference-compare): backend of the next session.
    void select_expert_tier(const std::string& name);
    std::string expert_tier()const;
};
}
