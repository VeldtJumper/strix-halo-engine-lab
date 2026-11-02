#pragma once

#include "strix/ds41_tier_stats.hpp"
#include <array>
#include <algorithm>
#include <cstdint>
#include <ostream>
#include <string_view>

namespace strix::ds41::cuda {
// Host-only summary payload. No device events, routes or physical-traffic claims.
struct ReplicaCensus {
    std::uint64_t request_id = 0, cycle_id = 0;
    std::uint32_t pos = 0, R = 0, drafted = 0, accepted = 0, committed = 0, draft_steps_executed = 0;
    std::uint32_t errors = 0;
    bool counted = true;
    std::array<TierLayer, 40> layers{};
    TierTokenStats traffic{};
    double target_device_ms = 0, draft_device_ms = 0, draft_wall_ms = 0, append_ms = 0, boundary_ms = 0;
    double cycle_ms = -1, engram_wait_ms = 0, engram_wait_first_ms = 0, engram_ms = 0, engram_issue_ms = 0;
    std::uint64_t draft_routed_lower_bytes = 0, draft_routed_upper_bytes = 0, vocabulary_head_bytes = 0;
    std::uint64_t index_key_bytes = 0, kv_bytes = 0, draft_nonrouted_bytes = 0, append_hidden_bytes = 0;
    std::uint64_t expert_bytes = 9953280, target_nonrouted_bytes = 0;

    std::uint64_t demand_bytes() const { return std::uint64_t(traffic.fetched) * expert_bytes; }
    std::uint64_t prefetch_bytes() const { return std::uint64_t(traffic.prefetched) * expert_bytes; }
    std::uint64_t routed_bytes() const {
        std::uint64_t n = 0;
        for (const auto& l : layers) n += l.ids;
        return n * expert_bytes;
    }
    void write(std::ostream& os, std::string_view phase = {}) const {
        os << "{\"phase\":\"";
        for (const unsigned char ch : phase) {
            if (ch == '"' || ch == '\\') os << '\\' << ch;
            else if (ch < 32) os << "\\u00" << "0123456789abcdef"[ch >> 4] << "0123456789abcdef"[ch & 15];
            else os << ch;
        }
        os << "\",\"errors\":" << errors << ",\"request_id\":" << request_id << ",\"cycle_id\":" << cycle_id << ",\"pos\":" << pos
           << ",\"R\":" << R << ",\"drafted\":" << drafted << ",\"accepted\":" << accepted << ",\"committed\":" << committed
           << ",\"counted\":" << (counted ? "true" : "false") << ",\"draft_steps_executed\":" << draft_steps_executed;
        const auto array = [&](const char* key, const auto& value) {
            os << ",\"" << key << "\":[";
            for (std::size_t i = 0; i < layers.size(); ++i) os << (i ? "," : "") << value(layers[i]);
            os << ']';
        };
        array("U_l", [](const TierLayer& l) { return l.ids; });
        array("export_hits_l", [](const TierLayer& l) { return l.ids - l.misses; });
#define CENSUS_LAYER(name, member) array(name, [](const TierLayer& l) { return l.member; })
        CENSUS_LAYER("misses_l", misses); CENSUS_LAYER("covered_l", covered);
        CENSUS_LAYER("fetched_l", fetched); CENSUS_LAYER("prefetched_l", prefetched);
        CENSUS_LAYER("covered_waits_l", covered_waits);
        CENSUS_LAYER("gpu_demand_wait_us_l", gpu_demand_wait_us); CENSUS_LAYER("export_to_ready_us_l", export_to_ready_us);
        CENSUS_LAYER("seen_us_l", seen_us); CENSUS_LAYER("serve_us_l", serve_us); CENSUS_LAYER("demand_us_l", demand_us);
        CENSUS_LAYER("pred_wait_us_l", pred_wait_us); CENSUS_LAYER("prefetch_us_l", prefetch_us);
        CENSUS_LAYER("dma_ms_l", dma_ms); CENSUS_LAYER("dma_first_ms_l", dma_first_ms); CENSUS_LAYER("wb_left_ms_l", wb_left_ms);
#undef CENSUS_LAYER
        os << ",\"payload_scope\":\"expert payload; table/control excluded\",\"demand_h2d_bytes\":" << demand_bytes()
           << ",\"prefetch_h2d_bytes\":" << prefetch_bytes();
#define CENSUS_TRAFFIC(k) os << ",\"" #k "\":" << traffic.k
        CENSUS_TRAFFIC(wb_issued_bytes); CENSUS_TRAFFIC(wb_completed_bytes); CENSUS_TRAFFIC(wb_cancelled_bytes);
        CENSUS_TRAFFIC(wb_boundary_flushed_bytes); CENSUS_TRAFFIC(wb_outstanding_start_bytes); CENSUS_TRAFFIC(wb_outstanding_bytes);
        CENSUS_TRAFFIC(wb_deferred); CENSUS_TRAFFIC(wb_cancelled); CENSUS_TRAFFIC(wb_forced); CENSUS_TRAFFIC(wb_boundary_flushed);
        CENSUS_TRAFFIC(wb_forced_wait_ns); CENSUS_TRAFFIC(device_timeouts);
        CENSUS_TRAFFIC(disk_reads); CENSUS_TRAFFIC(disk_read_bytes); CENSUS_TRAFFIC(disk_wait_ns); CENSUS_TRAFFIC(disk_stage_waits);
        CENSUS_TRAFFIC(disk_drops); CENSUS_TRAFFIC(disk_carried);
        CENSUS_TRAFFIC(boundary_sync_us); CENSUS_TRAFFIC(boundary_select_us); CENSUS_TRAFFIC(boundary_upload_us); CENSUS_TRAFFIC(boundary_issue_us);
#undef CENSUS_TRAFFIC
        os << ",\"target_device_ms\":" << target_device_ms << ",\"draft_device_ms\":" << draft_device_ms
           << ",\"draft_wall_ms\":" << draft_wall_ms << ",\"append_ms\":" << append_ms << ",\"boundary_ms\":" << boundary_ms
           << ",\"cycle_ms\":" << cycle_ms << ",\"engram_wait_ms\":" << engram_wait_ms
           << ",\"engram_wait_first_ms\":" << engram_wait_first_ms << ",\"engram_ms\":" << engram_ms << ",\"engram_issue_ms\":" << engram_issue_ms
           << ",\"useful_hbm\":{\"kind\":\"calculated\",\"physical_transactions\":\"unmeasured\",\"target_nonrouted_bytes\":" << target_nonrouted_bytes
           << ",\"target_routed_bytes\":" << routed_bytes()
           << ",\"index_key_bytes\":" << index_key_bytes << ",\"kv_bytes\":" << kv_bytes
           << ",\"draft_nonrouted_bytes\":" << draft_nonrouted_bytes << ",\"append_hidden_bytes\":" << append_hidden_bytes
           << ",\"draft_routed_lower_bytes\":" << draft_routed_lower_bytes << ",\"draft_routed_upper_bytes\":" << draft_routed_upper_bytes
           << ",\"vocabulary_head_bytes\":" << vocabulary_head_bytes
           << ",\"replay_bytes\":0,\"scope\":\"payload lower bound; draft routed union bounded without route downloads; intermediates excluded; no target replay in replica commit_window\"}}";
    }
};

// Calculated consumed cache payload; excludes selection scratch and physical rereads.
template<class Plan>
void census_cache_payload(ReplicaCensus& c, const Plan& plan, std::uint32_t layers) {
    for (std::uint32_t l = 0; l < layers; ++l) {
        const auto& a = plan[l];
        if (a.owns_index) {
            const std::uint32_t tile = c.R >= 4 ? 4U : c.R >= 2 ? 2U : 1U;
            for (std::uint32_t r = 0; r < c.R; r += tile)
                c.index_key_bytes += 68ULL * ((c.pos + std::min(c.R, r + tile)) / a.ratio);
        }
        for (std::uint32_t r = 0; r < c.R; ++r) {
            c.kv_bytes += 288ULL * std::min(128U, c.pos + r + 1);
            if (a.ratio) c.kv_bytes += 288ULL * std::min(512U, (c.pos + r + 1) / a.ratio);
        }
    }
}

struct ReplicaCensusTotals {
    ReplicaCensus sum;
    std::uint64_t cycles = 0, committed = 0, drafted = 0, accepted = 0, draft_steps_executed = 0;
    std::uint64_t demand_h2d_bytes = 0, prefetch_h2d_bytes = 0, wb_issued_bytes = 0;
    std::uint64_t wb_outstanding_start_bytes = 0, wb_outstanding_end_bytes = 0;
    double cycle_ms = 0;
    std::array<std::uint64_t, 9> R_histogram{};
    std::array<std::array<std::uint64_t, 40>, 9> U_by_R{};
    void add(const ReplicaCensus& c) {
        if (!c.counted) return;
        if (!cycles) wb_outstanding_start_bytes = c.traffic.wb_outstanding_start_bytes;
        wb_outstanding_end_bytes = c.traffic.wb_outstanding_bytes;
        sum.expert_bytes = c.expert_bytes;
        sum.traffic.wb_outstanding_start_bytes = wb_outstanding_start_bytes;
        sum.traffic.wb_outstanding_bytes = c.traffic.wb_outstanding_bytes;
#define ADD_CENSUS(k) sum.k += c.k
        ADD_CENSUS(errors); ADD_CENSUS(target_device_ms); ADD_CENSUS(draft_device_ms); ADD_CENSUS(draft_wall_ms); ADD_CENSUS(append_ms);
        ADD_CENSUS(boundary_ms); ADD_CENSUS(engram_wait_ms); ADD_CENSUS(engram_wait_first_ms); ADD_CENSUS(engram_ms); ADD_CENSUS(engram_issue_ms); ADD_CENSUS(target_nonrouted_bytes);
        ADD_CENSUS(index_key_bytes); ADD_CENSUS(kv_bytes); ADD_CENSUS(draft_nonrouted_bytes); ADD_CENSUS(append_hidden_bytes);
        ADD_CENSUS(draft_routed_lower_bytes); ADD_CENSUS(draft_routed_upper_bytes); ADD_CENSUS(vocabulary_head_bytes);
        ADD_CENSUS(traffic.fetched); ADD_CENSUS(traffic.prefetched);
        ADD_CENSUS(traffic.wb_issued_bytes); ADD_CENSUS(traffic.wb_completed_bytes); ADD_CENSUS(traffic.wb_cancelled_bytes);
        ADD_CENSUS(traffic.wb_boundary_flushed_bytes); ADD_CENSUS(traffic.wb_deferred); ADD_CENSUS(traffic.wb_cancelled);
        ADD_CENSUS(traffic.wb_forced); ADD_CENSUS(traffic.wb_boundary_flushed); ADD_CENSUS(traffic.wb_forced_wait_ns);
        ADD_CENSUS(traffic.device_timeouts); ADD_CENSUS(traffic.boundary_sync_us); ADD_CENSUS(traffic.boundary_select_us);
        ADD_CENSUS(traffic.boundary_upload_us); ADD_CENSUS(traffic.boundary_issue_us);
        ADD_CENSUS(traffic.disk_reads); ADD_CENSUS(traffic.disk_read_bytes); ADD_CENSUS(traffic.disk_wait_ns);
        ADD_CENSUS(traffic.disk_stage_waits); ADD_CENSUS(traffic.disk_drops); ADD_CENSUS(traffic.disk_carried);
#undef ADD_CENSUS
        for (std::size_t l = 0; l < 40; ++l) {
            auto& a = sum.layers[l]; const auto& b = c.layers[l];
            a.ids += b.ids; a.misses += b.misses; a.covered += b.covered; a.fetched += b.fetched;
            a.prefetched += b.prefetched; a.covered_waits += b.covered_waits;
            a.seen_us += b.seen_us; a.serve_us += b.serve_us; a.demand_us += b.demand_us;
            a.pred_wait_us += b.pred_wait_us; a.prefetch_us += b.prefetch_us;
            const auto time = [](double& x, double y) { if (y >= 0) x = (x < 0 ? 0 : x) + y; };
            time(a.gpu_demand_wait_us, b.gpu_demand_wait_us); time(a.export_to_ready_us, b.export_to_ready_us);
            time(a.dma_ms, b.dma_ms); time(a.dma_first_ms, b.dma_first_ms); time(a.wb_left_ms, b.wb_left_ms);
        }
        sum.committed += c.committed; sum.drafted += c.drafted; sum.accepted += c.accepted;
        sum.draft_steps_executed += c.draft_steps_executed;
        sum.cycle_ms = (sum.cycle_ms < 0 ? 0 : sum.cycle_ms) + (c.cycle_ms < 0 ? 0 : c.cycle_ms);
        ++cycles; committed += c.committed; drafted += c.drafted; accepted += c.accepted;
        draft_steps_executed += c.draft_steps_executed;
        demand_h2d_bytes += c.demand_bytes(); prefetch_h2d_bytes += c.prefetch_bytes(); wb_issued_bytes += c.traffic.wb_issued_bytes;
        if (c.cycle_ms >= 0) cycle_ms += c.cycle_ms;
        if (c.R < R_histogram.size()) {
            ++R_histogram[c.R];
            for (std::size_t l = 0; l < 40; ++l) U_by_R[c.R][l] += c.layers[l].ids;
        }
    }
    void write(std::ostream& os) const {
        os << "{\"totals\":"; sum.write(os);
        os << ",\"cycles\":" << cycles << ",\"committed\":" << committed << ",\"drafted\":" << drafted
           << ",\"accepted\":" << accepted << ",\"draft_steps_executed\":" << draft_steps_executed
           << ",\"E\":" << (cycles ? double(committed) / cycles : 0)
           << ",\"cycle_ms\":" << cycle_ms << ",\"ms_per_committed\":" << (committed ? cycle_ms / committed : 0)
           << ",\"demand_h2d_bytes\":" << demand_h2d_bytes << ",\"prefetch_h2d_bytes\":" << prefetch_h2d_bytes
           << ",\"wb_issued_bytes\":" << wb_issued_bytes << ",\"wb_outstanding_start_bytes\":" << wb_outstanding_start_bytes
           << ",\"wb_outstanding_end_bytes\":" << wb_outstanding_end_bytes << ",\"bytes_per_committed\":"
           << (committed ? double(demand_h2d_bytes + prefetch_h2d_bytes + wb_issued_bytes) / committed : 0) << ",\"R_histogram\":[";
        for (std::size_t r = 0; r < R_histogram.size(); ++r) os << (r ? "," : "") << R_histogram[r];
        os << "],\"U_by_R\":[";
        for (std::size_t r = 0; r < U_by_R.size(); ++r) {
            os << (r ? ",[" : "[");
            for (std::size_t l = 0; l < 40; ++l) os << (l ? "," : "") << U_by_R[r][l];
            os << ']';
        }
        os << "]}";
    }
};
}  // namespace strix::ds41::cuda
