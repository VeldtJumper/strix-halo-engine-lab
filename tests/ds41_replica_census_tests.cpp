#include "strix/ds41_replica_census.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace strix::ds41::cuda;
static void check(bool ok) { if (!ok) throw std::runtime_error("census contract"); }
int main() {
    ReplicaCensus cache; cache.pos = 512; cache.R = 4;
    struct Layer { bool owns_index; unsigned ratio; };
    const std::array<Layer, 1> plan{{{true, 2}}};
    census_cache_payload(cache, plan, 1);
    check(cache.index_key_bytes == 258 * 68 && cache.kv_bytes == 1540 * 288);
    ReplicaCensus a;
    // Export [2,2,7]: U is deduplicated, covered counts masked occurrences.
    a.R = 4; a.layers[0].ids = 2; a.layers[0].misses = 1;
    a.layers[0].covered = 2; a.layers[0].fetched = 1;
    a.traffic.fetched = 1; a.traffic.prefetched = 2;
    a.traffic.wb_outstanding_start_bytes = 100; a.traffic.wb_outstanding_bytes = 40;
    a.traffic.wb_issued_bytes = 20; a.traffic.wb_completed_bytes = 80;
    a.traffic.disk_reads = 3; a.traffic.disk_wait_ns = 5000;
    a.committed = 3; a.drafted = 3; a.accepted = 2; a.draft_steps_executed = 1;
    a.cycle_ms = 5;
    check(a.demand_bytes() == 9953280 && a.prefetch_bytes() == 2 * a.demand_bytes());
    check(a.routed_bytes() == 2 * a.demand_bytes());
    check(a.traffic.wb_outstanding_start_bytes + a.traffic.wb_issued_bytes - a.traffic.wb_completed_bytes == a.traffic.wb_outstanding_bytes);
    ReplicaCensusTotals t;
    a.counted = false; t.add(a); check(t.cycles == 0); // warmup exclusion
    a.counted = true; t.add(a);
    a.cycle_id = 1; a.cycle_ms = 7; a.committed = 1; // terminal tail belongs to final cycle
    a.traffic.wb_outstanding_start_bytes = 40; a.traffic.wb_outstanding_bytes = 25;
    t.add(a);
    check(t.cycles == 2 && t.committed == 4 && t.cycle_ms == 12);
    check(t.wb_outstanding_start_bytes == 100 && t.wb_outstanding_end_bytes == 25);
    check(t.sum.traffic.wb_completed_bytes == 160 && t.sum.traffic.wb_outstanding_bytes == 25);
    check(t.sum.traffic.disk_reads == 6 && t.sum.traffic.disk_wait_ns == 10000);
    check(t.sum.layers[0].ids == 4 && t.sum.layers[0].covered == 4);
    check(t.R_histogram[4] == 2 && t.U_by_R[4][0] == 4);
    std::ostringstream out; a.write(out);
    const auto s = out.str();
    for (const char* key : {"U_l", "export_hits_l", "covered_waits_l", "wb_completed_bytes", "wb_forced_wait_ns", "draft_steps_executed", "disk_reads", "disk_wait_ns", "disk_carried", "calculated", "unmeasured"})
        check(s.find(key) != std::string::npos);
    check(s.find("\"covered_l\":[2,") != std::string::npos);
    check(s.find("\"export_hits_l\":[1,") != std::string::npos);
    check(s.find("\"disk_reads\":3,") != std::string::npos);
    std::cout << "PASS duplicate routes; payload conservation; gauges; warmup; terminal tail; schema\n";
}
