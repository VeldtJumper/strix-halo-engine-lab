#pragma once
#include "strix/state_cache_snapshot.hpp"
#include "strix/serve/json.hpp"
#include "strix/prompt_terminal_diagnostics.hpp"
#include <atomic>
#include <chrono>

namespace strix::decode_cli {
inline serve::Json state_cache_json(const StateCacheSnapshotV1& s) {
    using serve::Json;
    Json out=Json::object(), prefix=Json::object(), named=Json::object();
    out.set("schema",Json(static_cast<std::uint64_t>(s.schema)));
    out.set("published",Json(s.published)); out.set("sequence",Json(s.sequence));
    const auto now=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    out.set("age_ms", s.published ? Json(static_cast<double>(now-s.published_steady_ns)/1e6) : Json());
    out.set("prefix_enabled",Json(s.prefix_enabled)); out.set("session_enabled",Json(s.session_enabled));
    out.set("pool_present",Json(s.pool_present));
#define STRIX_CACHE_FIELD(obj, source, field) obj.set(#field,Json(source.field))
#define STRIX_PREFIX(field) STRIX_CACHE_FIELD(prefix,s.prefix,field)
    STRIX_PREFIX(lookups); STRIX_PREFIX(host_hits); STRIX_PREFIX(disk_hits); STRIX_PREFIX(misses);
    STRIX_PREFIX(lookup_tokens); STRIX_PREFIX(hit_tokens); STRIX_PREFIX(restored_tokens);
    STRIX_PREFIX(plan_misses); STRIX_PREFIX(captures); STRIX_PREFIX(capture_failures); STRIX_PREFIX(inserted); STRIX_PREFIX(no_store_skips); STRIX_PREFIX(budget_skips);
    STRIX_PREFIX(staging_budget_bytes); STRIX_PREFIX(staging_bytes); STRIX_PREFIX(staging_peak_bytes);
    STRIX_PREFIX(capture_skipped_no_slot); STRIX_PREFIX(pool_bytes); STRIX_PREFIX(pool_budget_bytes);
    STRIX_PREFIX(pool_staging_bytes); STRIX_PREFIX(pool_metadata_reserved); STRIX_PREFIX(pool_free_bytes);
    STRIX_PREFIX(row_prefetches); STRIX_PREFIX(capture_busy_skips); STRIX_PREFIX(staging_cap_skips);
    STRIX_PREFIX(capture_skips_cross_stream);
    Json declines=Json::object();
    for(std::size_t i=0;i<prefix_certificate_decline_names.size();++i)
        declines.set(std::string(prefix_certificate_decline_names[i]),Json(s.prefix.certificate_declines[i]));
    prefix.set("certificate_declines",std::move(declines));
    if(prefix_batch_certification_enabled()) {
        STRIX_PREFIX(snapshot_preparations);
        STRIX_PREFIX(capture_skips_tripwire);
        STRIX_PREFIX(capture_skips_batch_config);
    }
    STRIX_PREFIX(evicted); STRIX_PREFIX(rejected_over_budget);
    STRIX_PREFIX(evictions_superseded); STRIX_PREFIX(evictions_space); STRIX_PREFIX(evictions_conversation_cap); STRIX_PREFIX(evictions_branch_cap);
    STRIX_PREFIX(evictions_explicit); STRIX_PREFIX(evictions_invalid);
    STRIX_PREFIX(host_bytes); STRIX_PREFIX(host_segments); STRIX_PREFIX(host_checkpoints);
    STRIX_PREFIX(disk_writes); STRIX_PREFIX(disk_write_bytes); STRIX_PREFIX(disk_write_failures); STRIX_PREFIX(disk_evictions);
    STRIX_PREFIX(disk_bytes); STRIX_PREFIX(disk_reads); STRIX_PREFIX(disk_read_bytes); STRIX_PREFIX(disk_read_failures);
    STRIX_PREFIX(disk_read_ms); STRIX_PREFIX(disk_write_ms);
#undef STRIX_PREFIX
    STRIX_CACHE_FIELD(named,s.named,hits); STRIX_CACHE_FIELD(named,s.named,misses);
    STRIX_CACHE_FIELD(named,s.named,refusals); STRIX_CACHE_FIELD(named,s.named,parked); STRIX_CACHE_FIELD(named,s.named,no_store_skips);
    out.set("prefix",std::move(prefix)); out.set("named",std::move(named));
    const auto tier=[](const StateCacheSnapshotV1::Tier& t) {
        Json j=Json::object(); STRIX_CACHE_FIELD(j,t,payload_bytes); STRIX_CACHE_FIELD(j,t,segments);
        STRIX_CACHE_FIELD(j,t,checkpoints); STRIX_CACHE_FIELD(j,t,budget_bytes); return j;
    };
    out.set("resident",tier(s.resident)); out.set("host",tier(s.host)); out.set("disk",tier(s.disk));
    Json pool=Json::object();
#define STRIX_POOL(field) STRIX_CACHE_FIELD(pool,s.pool,field)
    STRIX_POOL(capacity_positions); STRIX_POOL(used_positions); STRIX_POOL(reserved_positions);
    STRIX_POOL(busy_positions); STRIX_POOL(held_positions); STRIX_POOL(free_positions);
    STRIX_POOL(largest_free_extent); STRIX_POOL(committed_positions); STRIX_POOL(arena_bytes);
    STRIX_POOL(fixed_bytes); STRIX_POOL(workspace_bytes); STRIX_POOL(move_staging_bytes);
    STRIX_POOL(pending_waiters); STRIX_POOL(invalid_positions);
#undef STRIX_POOL
#undef STRIX_CACHE_FIELD
    out.set("pool",std::move(pool)); return out;
}
// Immutable publication: HTTP readers never enter the runner or store, walk payloads or touch HIP.
class StateCacheStatsMirror {
    struct Sample { StateCacheSnapshotV1 snapshot; serve::Json top; };
    std::atomic<std::shared_ptr<const Sample>> sample_;
    std::mutex publish_mutex_;
    std::uint64_t sequence_ = 0;
public:
    void publish(StateCacheSnapshotV1 s, serve::Json top=serve::Json::object()) {
        std::lock_guard lock(publish_mutex_);
        s.published=true; s.sequence=++sequence_;
        s.published_steady_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
        using serve::Json;
        if (!top.find("ram_bytes")) {
            top.set("ram_bytes",Json(s.host.payload_bytes)); top.set("ram_budget",Json(s.host.budget_bytes));
            top.set("parked",Json(s.named.parked)); top.set("disk_bytes",Json(s.disk.payload_bytes));
            top.set("hits",Json(s.named.hits)); top.set("misses",Json(s.named.misses)); top.set("refusals",Json(s.named.refusals));
        }
        Json engine=top.find("engine") ? *top.find("engine") : Json::object();
        engine.set("state_cache",state_cache_json(s)); top.set("engine",std::move(engine));
        sample_.store(std::make_shared<const Sample>(Sample{s,std::move(top)}));
    }
    std::string read() const {
        auto p=sample_.load(); if (!p) return "{}";
        auto top=p->top;
        auto engine=*top.find("engine"); engine.set("state_cache",state_cache_json(p->snapshot));
        top.set("engine",std::move(engine)); return top.dump();
    }
};
inline StateCacheSnapshotV1 prefix_snapshot(PrefixStateCache& cache) {
    StateCacheSnapshotV1 s; s.prefix_enabled=true; s.prefix=cache.counters();
    s.host={s.prefix.host_bytes,s.prefix.host_segments,s.prefix.host_checkpoints,cache.policy().host_byte_budget};
    s.pool.move_staging_bytes=s.prefix.staging_bytes;
    s.disk.payload_bytes=s.prefix.disk_bytes;
    s.disk.budget_bytes=cache.policy().disk_byte_budget;
    return s;
}
} // namespace strix::decode_cli
