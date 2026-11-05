#include "strix/ple_reclaim.hpp"
#include "../apps/ple_reclaim_mapping.hpp"
#include "../apps/ple_resident_fallback.hpp"
#include "strix/mtp_window_controller.hpp"
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace strix;
using namespace strix::ple_reclaim;
namespace {
void expect(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void refuses(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    expect(rejected, "operation must refuse cleanly");
}
struct Fake {
    std::set<std::uint64_t> resident, evicted;
    std::uint64_t observed = 0;
    bool fail_sample = false, fail_drop = false;
    RollingPages::Io io() {
        return {
            [this](auto pages) { std::uint64_t n = 0; for (auto pg : pages) n += resident.insert(pg).second; return n; },
            [this](auto pages) {
                if (fail_drop) throw std::runtime_error("mock eviction failure");
                for (auto pg : pages) { resident.erase(pg); evicted.insert(pg); }
            },
            [this] { if (fail_sample) throw std::runtime_error("mock mincore failure");
                return observed ? observed : static_cast<std::uint64_t>(resident.size()) * 4096U; },
            [](auto pg) { return pg >= 2U && pg < 100U && pg != 3U; }, {}, {}, {}, {}
        };
    }
};
void windows() {
    bool fallback = false;
    refuses([&] {
        (void)decode_cli::setup_ple_or_resident(Mode::automatic,
            []() -> std::shared_ptr<Prefetcher> { throw DiscardSafetyError("fixture overlap"); },
            [&] { fallback = true; },[&] { fallback = true; },[&] { fallback = true; });
    });
    expect(!fallback,"automatic mode must not fall back after a discard safety refusal");
    Fake io;
    // Consecutive windows share page 2, then it disappears and returns.
    std::vector<std::uint32_t> rows(256U * 5U, 0U);
    std::fill(rows.begin() + 512, rows.begin() + 1024, 200U);
    RollingPages r(256,4096,8192+23,90,1,rows,1U<<20U,1U<<18U,io.io());
    r.gather(0,256,[&] { expect(!io.evicted.contains(2), "cannot evict before copy completion"); });
    expect(!io.evicted.contains(2), "next window retains shared page");
    r.gather(256,256,[&] { expect(!io.evicted.contains(2), "current source remains live"); });
    expect(io.evicted.contains(2), "consumed page evicted after last live reference");
    r.gather(512,256,[] {});
    const auto prior = r.stats().prefetched_pages;
    r.gather(768,256,[] {});
    expect(r.stats().prefetched_pages > prior && io.resident.contains(2), "reappearing ngram refetched in a later epoch");
    r.gather(1024,256,[] {}); r.finish();
    expect(io.resident.empty(), "turn end releases both windows");
    expect(!io.evicted.contains(3), "protected page never evicted");
    refuses([&] { r.gather(0,256,[] {}); });

    // Row 45 spans pages 2/3 at this misaligned table start. Page 3 is protected.
    Fake boundary;
    std::vector<std::uint32_t> crossing(256,45);
    RollingPages b(256,4096,8192+23,90,1,crossing,1U<<20U,1U<<18U,boundary.io());
    b.gather(0,256,[&] { expect(boundary.resident.contains(2), "spanning row first page retained"); });
    b.finish(); expect(!boundary.evicted.contains(3), "shared boundary/protected second page excluded");
    Fake spanning;
    auto both_pages = spanning.io();
    both_pages.reclaimable = [](auto pg) { return pg >= 2U && pg < 100U; };
    RollingPages two(256,4096,8192+23,90,1,crossing,1U<<20U,1U<<18U,std::move(both_pages));
    two.gather(0,256,[&] {
        expect(spanning.resident.contains(2) && spanning.resident.contains(3), "both pages of spanning rows retained");
    });
    two.finish();
    expect(spanning.evicted.contains(2) && spanning.evicted.contains(3), "both pages released after source copy");

    auto intervals = reclaim_intervals(37,20000,4096,{{8200,8300}});
    expect(intervals.size() == 2 && intervals[0].begin == 4096 && intervals[0].end == 8192 &&
        intervals[1].begin == 12288 && intervals[1].end == 16384, "misaligned table edges and rounded protection");
    Fake exceeded; exceeded.observed = 1048577;
    RollingPages bad(256,4096,8192,90,1,crossing,1U<<20U,1U<<18U,exceeded.io());
    bool copied = false;
    refuses([&] { bad.gather(0,256,[&] { copied = true; }); });
    expect(!copied, "bound violation refuses before copy");
    refuses([&] { bad.source(std::array<std::uint64_t,1>{0},[] {}); });
    Fake failed; failed.fail_sample = true;
    RollingPages unverifiable(256,4096,8192,90,1,crossing,1U<<20U,1U<<18U,failed.io());
    refuses([&] { unverifiable.gather(0,256,[] {}); });
    Fake drop; drop.fail_drop = true;
    RollingPages undroppable(256,4096,8192,90,1,crossing,1U<<20U,1U<<18U,drop.io());
    refuses([&] { undroppable.gather(0,256,[] {}); });
    Fake decoding;
    RollingPages decode(256,4096,8192,90,1,crossing,1U<<20U,4096,decoding.io());
    decode.source(std::array<std::uint64_t,1>{0},[] {});
    refuses([&] { decode.source(std::array<std::uint64_t,1>{200},[] {}); });
    Fake cancelled;
    RollingPages cancellation(256,4096,8192,90,1,crossing,1U<<20U,1U<<18U,cancelled.io());
    refuses([&] { cancellation.gather(0,256,[] { throw std::runtime_error("cancelled copy"); }); });
    refuses([&] { cancellation.gather(0,256,[] {}); });
}
void completion() {
    Fake io;
    std::vector<std::uint32_t> rows(512,200);
    std::fill_n(rows.begin(),256,0);
    RollingPages r(256,4096,8192+23,90,1,rows,1U<<20U,1U<<18U,io.io());
    std::promise<void> reading, complete;
    auto completion = complete.get_future();
    std::future<void> reader;
    r.gather(0,256,[&] {
        reader = std::async(std::launch::async,[&] {
            r.source(std::array<std::uint64_t,1>{0},[&] { reading.set_value(); completion.wait(); });
        });
        reading.get_future().wait();
    });
    expect(!io.evicted.contains(2), "outstanding source-copy lease blocks eviction after window completion");
    complete.set_value(); reader.get();
    expect(io.evicted.contains(2), "completion releases source page");
    r.gather(256,256,[] {}); r.finish();
}

// Thread-safe page-cache model: asynchronous hints really may finish after a
// demand copy. Every insertion asserts the hard cap, not only final samples.
struct PipelineIo {
    std::mutex mutex;
    std::condition_variable changed;
    std::set<std::uint64_t> resident, evicted;
    std::uint64_t cap, calls = 0, peak = 0;
    explicit PipelineIo(std::uint64_t bytes) : cap(bytes) {}
    void touch(std::span<const std::uint64_t> pages) {
        const std::lock_guard lock(mutex);
        resident.insert(pages.begin(), pages.end());
        peak = std::max<std::uint64_t>(peak, resident.size()*4096U);
        expect(peak <= cap, "physical mock residency never exceeds the cap, including in-flight hints");
        changed.notify_all();
    }
    void wait_calls(std::uint64_t n) {
        std::unique_lock lock(mutex);
        expect(changed.wait_for(lock,std::chrono::seconds(10),[&] { return calls >= n; }), "lookahead progresses asynchronously");
    }
    bool dropped(std::uint64_t pg) { const std::lock_guard lock(mutex); return evicted.contains(pg); }
    RollingPages::Io io() {
        return {
            [this](auto pages) {
                touch(pages);
                const std::lock_guard lock(mutex); ++calls; changed.notify_all();
                return static_cast<std::uint64_t>(pages.size());
            },
            [this](auto pages) {
                const std::lock_guard lock(mutex);
                for (auto pg : pages) { resident.erase(pg); evicted.insert(pg); }
            },
            [this] { const std::lock_guard lock(mutex); return static_cast<std::uint64_t>(resident.size()*4096U); },
            [](auto) { return true; }, {}, {}, {}, {}
        };
    }
};
void pipeline() {
    {
        // Dense, disjoint n-gram rows force both split windows and adaptive
        // lookahead throttling. No assumed random-page density is used for fit.
        constexpr std::uint64_t cap = 64U*4096U;
        PipelineIo io(cap);
        std::vector<std::uint32_t> rows(4096U*16U);
        for (std::uint32_t i = 0; i < rows.size(); ++i) rows[i] = i*3U;
        // Every row straddles two disjoint pages, at the maximum per-position density.
        RollingPipeline r(256,4096,4095,4096,16,rows,cap+65536,65536,cap,1024,io.io());
        io.wait_calls(1);
        r.gather_range(0,rows.size()/16U,[&](auto first, auto count) {
            std::vector<std::uint64_t> ids(rows.begin()+static_cast<std::ptrdiff_t>(first*16U),
                rows.begin()+static_cast<std::ptrdiff_t>((first+count)*16U));
            std::vector<std::uint64_t> pages;
            for (auto id : ids) { pages.push_back(id); pages.push_back(id+1U); }
            r.source(ids,[&] { io.touch(pages); });
            r.advance(first+count);
        });
        r.finish();
        const auto stats = r.stats();
        expect(stats.peak_owned_bytes <= cap && stats.throttles && stats.split_windows,
            "adversarial stream obeys distinct-page cap, throttles and splits");
        expect(io.resident.empty(), "pipeline turn cleanup releases all lookahead and copy pages");
    }
    {
        PipelineIo io(4U<<20U);
        std::vector<std::uint32_t> rows(1024,0);
        std::fill(rows.begin()+256,rows.begin()+512,1U);
        RollingPipeline r(256,4096,0,4096,1,rows,5U<<20U,1U<<20U,4U<<20U,1024,io.io());
        io.wait_calls(4);
        r.gather(0,256,[] {});
        expect(!io.dropped(0), "page needed by deep lookahead survives consumed live window");
        r.gather(256,256,[] {});
        expect(!io.dropped(0), "shared future page remains retained across unrelated window");
        r.gather(512,256,[] {}); r.gather(768,256,[] {});
        expect(r.stats().evicted_pages == 0, "cache retains consumed pages while the cap has space");
        r.finish();
        expect(io.dropped(0), "future page drops only after final CPU completion");
        expect(r.stats().lookahead_positions >= 1024, "deep horizon spans multiple GPU blocks");
    }
    {
        // Pressure must remove a consumed page while preserving the shared page
        // needed by the replacement lookahead, including during reservation.
        constexpr std::uint64_t cap = 4U*4096U;
        PipelineIo io(cap);
        std::vector<std::uint32_t> rows(768,3U);
        std::fill(rows.begin(),rows.begin()+128,0U);
        std::fill(rows.begin()+128,rows.begin()+256,2U);
        std::fill(rows.begin()+256,rows.begin()+384,1U);
        std::fill(rows.begin()+384,rows.begin()+512,0U);
        RollingPipeline r(256,4096,0,4096,1,rows,cap+4096,4096,cap,512,io.io());
        io.wait_calls(1);
        r.gather(0,256,[] {});
        io.wait_calls(2);
        expect(io.dropped(2) && !io.dropped(0), "pressure evicts cold page but preserves lookahead-needed shared page");
        r.gather(256,128,[] {});
        expect(!io.dropped(0), "shared page cannot be evicted before future source copy");
        r.gather(384,128,[] {}); r.advance(512);
        r.gather(512,256,[] {}); r.finish();
    }
    {
        PipelineIo io(8U<<20U);
        auto callbacks = io.io();
        auto hint = callbacks.prefetch;
        std::promise<void> entered, release_hint, release_gpu, source_entered, release_source;
        auto hint_done = release_hint.get_future().share();
        auto gpu_done = release_gpu.get_future();
        auto source_done = release_source.get_future().share();
        callbacks.prefetch = [&](auto p) { entered.set_value(); hint_done.wait(); return hint(p); };
        std::vector<std::uint32_t> rows(512,0);
        std::fill(rows.begin()+256,rows.end(),1U);
        RollingPipeline r(256,4096,0,4096,1,rows,9U<<20U,1U<<20U,8U<<20U,256,std::move(callbacks));
        entered.get_future().wait();
        std::future<void> reader;
        r.gather(0,256,[&] {
            reader=std::async(std::launch::async,[&] {
                r.source(std::array<std::uint64_t,1>{0},[&] {
                    io.touch(std::array<std::uint64_t,1>{0}); source_entered.set_value(); source_done.wait();
                });
            });
            source_entered.get_future().wait();
        });
        expect(!io.dropped(0), "active CPU lease and in-flight hint prevent premature eviction");
        release_source.set_value(); reader.get();
        expect(!io.dropped(0), "in-flight asynchronous hint retains page after CPU copy completes");
        auto worker=std::async(std::launch::async,[&] {
            r.gather(256,256,[&] { r.source(std::array<std::uint64_t,1>{1},[&] {
                io.touch(std::array<std::uint64_t,1>{1});
            }); });
        });
        const bool overlapped = worker.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
        const bool gpu_pending = gpu_done.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout;
        // Unblock before assertions, including failure paths, so futures drain.
        release_hint.set_value(); release_gpu.set_value();
        worker.get();
        expect(overlapped && gpu_pending, "staging worker completes while GPU and background I/O remain outstanding");
        r.finish();
        expect(io.resident.empty(), "late hint cannot resurrect an evicted page after finish");
    }
    {
        constexpr std::uint64_t cap = 4U*4096U;
        PipelineIo io(cap);
        std::vector<std::uint32_t> rows(6U*256U);
        for (std::uint32_t i = 0; i < rows.size(); ++i) rows[i] = i/256U;
        RollingPipeline r(256,4096,0,4096,1,rows,cap+4096,4096,cap,256,io.io());
        std::promise<void> entered, complete;
        auto done = complete.get_future().share();
        std::future<void> reader;
        r.gather(0,256,[&] {
            reader=std::async(std::launch::async,[&] {
                r.source(std::array<std::uint64_t,1>{0},[&] {
                    io.touch(std::array<std::uint64_t,1>{0}); entered.set_value();
                    expect(done.wait_for(std::chrono::seconds(10))==std::future_status::ready, "mock source completion arrives");
                });
            });
            entered.get_future().wait();
        });
        for (std::uint64_t block=1; block<6; ++block) {
            r.advance(block*256U);
            r.gather(block*256U,256,[&] {
                r.source(std::array<std::uint64_t,1>{block},[&] { io.touch(std::array<std::uint64_t,1>{block}); });
            });
        }
        const bool retained = !io.dropped(0);
        complete.set_value(); reader.get(); r.finish();
        expect(retained, "cap pressure never evicts a page while its last CPU copy is outstanding");
    }
    {
        // A large-window c1 task may overlap a different partial/tail owner.
        // Both copy the same immutable rows at their own original offsets.
        PipelineIo io(4U<<20U);
        std::vector<std::uint32_t> rows(256,0);
        RollingPipeline r(256,4096,0,4096,1,rows,5U<<20U,1U<<20U,4U<<20U,256,io.io());
        std::promise<void> copying, complete;
        auto done = complete.get_future().share();
        auto full=std::async(std::launch::async,[&] { r.gather(0,256,[&] {
            copying.set_value();
            expect(done.wait_for(std::chrono::seconds(10))==std::future_status::ready, "full staging completion arrives");
        }); });
        copying.get_future().wait();
        auto tail=std::async(std::launch::async,[&] { r.gather(0,128,[] {}); });
        const bool serialized = tail.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout;
        complete.set_value(); full.get(); tail.get();
        expect(serialized, "overlapping tail-window CPU copies share one bounded source slot");
        r.finish();
    }
    for (const auto failure : {0,1,2,3}) {
        PipelineIo io(4U<<20U);
        auto callbacks = io.io();
        std::promise<void> failed_hint;
        if (failure == 0) callbacks.prefetch = [&](auto) -> std::uint64_t {
            failed_hint.set_value(); throw std::runtime_error("hint failure");
        };
        if (failure == 1) callbacks.resident = []() -> std::uint64_t { throw std::runtime_error("sample failure"); };
        auto evict = callbacks.evict;
        bool first = true;
        if (failure == 2) callbacks.evict = [&](auto p) {
            if (std::exchange(first,false)) throw std::runtime_error("drop failure");
            evict(p);
        };
        std::vector<std::uint32_t> rows(256,0);
        RollingPipeline r(256,4096,0,4096,1,rows,5U<<20U,1U<<20U,4U<<20U,256,std::move(callbacks));
        if (failure == 0) failed_hint.get_future().wait();
        if (failure == 1 || failure == 2) io.wait_calls(1);
        if (failure == 3) refuses([&] { r.gather(0,256,[] { throw std::runtime_error("cancelled copy"); }); });
        refuses([&] { r.finish(); });
        expect(!r.healthy(), "I/O, observation, eviction and copy failures revoke the pipeline certificate");
        refuses([&] { r.gather(0,256,[] {}); });
        // Cleanup after poisoning is still possible; a transient failed drop is
        // retried without reattaching any source reader.
        if (failure == 2) refuses([&] { r.finish(); });
        expect(io.resident.empty(), "poisoned pipeline drains hints and releases owned pages");
    }
    for (auto value : {"", "0", "-1", "1x", "18446744073709551616"})
        refuses([&] { (void)rolling_cap_bytes(value); });
    expect(rolling_cap_bytes(nullptr)==(3ULL<<30U), "cap default is 3 GiB");
    Config c; c.rolling_positions=1024; c.stored_row_bytes=90; c.decode_allowance_bytes=1U<<28U;
    for (auto cap : {4096ULL,262145ULL}) {
        c.rolling_cap_bytes=cap; refuses([&] { (void)rolling_budget(c,16); });
    }
    c.bounded_turns=true;
    refuses([&] { (void)rolling_budget(c,16); });
    c.rolling_cap_bytes=0;
    expect(rolling_budget(c,16).admitted==3607101440ULL, "MTP and plain use the same owned cap budget");
}

std::shared_ptr<ResidencyManager> manager(Fake& fake, std::string name = "fixture", std::uint64_t quota = 8192, std::uint32_t slots = 1) {
    auto io = fake.io();
    io.reclaimable = [](auto pg) { return pg >= 2U && pg < 1000000U; };
    // New path never calls WILLNEED; demand copy models residency explicitly.
    io.prefetch = [](auto) -> std::uint64_t { throw std::runtime_error("untracked prefetch forbidden"); };
    return std::make_shared<ResidencyManager>(std::move(name), 8192, 7, 256, 4096, 8192, 4096,
        1000000, 16, (slots * quota) + (2U << 20U), quota, slots, std::move(io));
}
void leases() {
    Fake fake;
    auto m = manager(fake);
    refuses([&] { (void)manager(fake); }); // one authority even over a mapping alias
    auto t = m->begin();
    refuses([&] { (void)m->begin(); });
    const std::array<std::uint64_t,2> rows{0,1};
    t->source(8192,7,rows,[&] { fake.resident.insert({2,3}); }); // calibration
    t->phase(true);
    expect(fake.resident.empty(), "calibration cache drained before prompt restart");
    expect(t->prompt(), "real prompt has a fresh prefill phase");
    t->gather(rows,[&] {
        t->source(8192,7,rows,[&] { fake.resident.insert({2,3}); });
        expect(fake.resident.contains(2), "source retains table bytes within prompt window");
        refuses([&] { t->phase(false); });
    });
    t->phase(false);
    expect(fake.resident.empty() && !t->prompt(), "prompt -> decode releases all source windows");
    for (std::uint64_t i = 0; i < 6000; ++i) {
        const std::array<std::uint64_t,2> ids{i,i+1U};
        t->source(8192,7,ids,[&] { fake.resident.insert({i+2U,i+3U}); });
        expect(fake.resident.size() <= 2U, "long-generation cache refresh is bounded, not lifetime cumulative");
    }
    bool touched = false;
    refuses([&] { t->source(8192,7,std::array<std::uint64_t,3>{0,1,2},[&] { touched = true; }); });
    expect(!touched && m->healthy(), "oversize wave refuses before touch without poisoning certificate");
    t->finish();
    expect(fake.resident.empty(), "finish releases bounded decode residency");
    refuses([&] { t->source(8192,7,rows,[] {}); });
    auto next = m->begin();
    expect(next->id() > t->id(), "turn IDs never reused across repeated requests");
    next->cancel();
    expect(m->healthy(), "ordinary cancellation leaves table certificate healthy");


    Fake production;
    auto prod = manager(production,"production-quota",256ULL<<20U);
    auto production_turn = prod->begin();
    // 4096 positions * 32 distinct stored pages: twice the production quota.
    // Simulates 16 straddling heads including rejected/replayed target work.
    for (std::uint64_t position=0; position<4096; ++position) {
        std::array<std::uint64_t,32> wave{};
        for (std::size_t i=0; i<wave.size(); ++i) wave[i]=position*wave.size()+i;
        production_turn->source(8192,7,wave,[&] {
            for (auto row:wave) production.resident.insert(row+2U);
        });
        expect(production.resident.size()<=65536U,"production 256 MiB quota wraps under long decode");
    }
    production_turn->finish(); expect(production.resident.empty(),"production quota drained");
    Fake shared;
    auto multi = manager(shared,"shared",4096,2);
    auto a = multi->begin(), b = multi->begin();
    a->source(8192,7,std::array<std::uint64_t,1>{0},[&] { shared.resident.insert(2); });
    b->source(8192,7,std::array<std::uint64_t,1>{0},[] {});
    a->finish(); expect(shared.resident.contains(2), "physical union survives a sibling turn retirement");
    b->finish(); expect(shared.resident.empty(), "shared physical page dropped only after last turn");
    auto c = multi->begin(), d = multi->begin(); c->phase(true); d->phase(true);
    c->gather(rows,[&] { refuses([&] { d->gather(rows,[] {}); }); });
    c->finish(); d->finish();

    Fake paused;
    auto pm = manager(paused,"paused",4096);
    auto pt = pm->begin();
    std::promise<void> reading, done;
    auto wait = done.get_future();
    auto read = std::async(std::launch::async,[&] {
        pt->source(8192,7,std::array<std::uint64_t,1>{0},[&] {
            paused.resident.insert(2); reading.set_value(); wait.wait();
        });
    });
    reading.get_future().wait();
    auto close = std::async(std::launch::async,[&] { pt->cancel(); });
    // Cancellation cannot return/recycle this slot until its last source retires.
    expect(close.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout,"cancel drains paused reader");
    expect(paused.resident.contains(2), "cancel cannot evict an active source");
    done.set_value(); read.get(); close.get();
    expect(paused.resident.empty(), "cancel drops source only after callback completion");
    auto fresh = pm->begin(); fresh->finish();

    for (unsigned failure = 0; failure < 5; ++failure) {
        Fake fault;
        auto fm = manager(fault,"fault"); auto ft = fm->begin();
        if (failure == 1) fault.fail_sample = true;
        if (failure == 2) fault.fail_drop = true;
        if (failure == 3) fault.observed = UINT64_MAX;
        refuses([&] {
            if (failure == 2) {
                ft->source(8192,7,rows,[&] { fault.resident.insert({2,3}); });
                ft->phase(true);
            } else ft->source(failure == 0 ? 8193 : 8192,7,rows,[&] {
                fault.resident.insert({2,3});
                if (failure == 4) throw std::runtime_error("copy cancelled after partial work");
            });
        });
        expect(!fm->healthy(), "source/observation/drop/alias failure invalidates global certificate");
        ft->cancel(); // cleanup legal even after failure
        refuses([&] { (void)fm->begin(); });
    }

    Fake missing;
    auto fail_missing=missing.io(); fail_missing.reclaimable=[](auto) { return true; };
    fail_missing.missing=[](auto) -> std::uint64_t { throw std::runtime_error("injected source observation failure"); };
    auto mm=std::make_shared<ResidencyManager>("missing-source",8192,7,256,4096,8192,4096,1000,16,
        1U<<20U,8192,1,std::move(fail_missing));
    auto mt=mm->begin(); bool copied=false;
    refuses([&] { mt->source(8192,7,rows,[&] { copied=true; }); });
    expect(!copied && !mm->healthy(),"failed source acquisition refuses before bytes are touched");
    mt->cancel();
    Fake old_epoch;
    auto em=manager(old_epoch,"epoch"); auto et=em->begin();
    const auto phase_epoch=et->epoch(); et->phase(true);
    refuses([&] { et->source(8192,7,rows,[] {},phase_epoch); });
    expect(!em->healthy(),"late lookup cannot attach to a fresh phase epoch"); et->cancel();
    Fake observers;
    auto om=manager(observers,"per-turn-observers");
    unsigned receipts=0;
    auto ot=om->begin({},[&](auto first,auto ids,auto,auto,auto) {
        expect(first==43 && ids.size()==2,"prompt observer belongs to admitted turn/suffix"); ++receipts;
    });
    ot->phase(true); ot->gather(rows,[] {}); ot->observe(43,rows,1,0,0); ot->finish();
    auto nt=om->begin({},[&](auto first,auto,auto,auto,auto) {
        expect(first==100,"replacement turn has fresh suffix observer"); ++receipts;
    });
    nt->phase(true); nt->observe(100,rows,1,0,0); nt->finish();
    expect(receipts==2,"no table-global prompt binding carried across turns");
    Fake generation;
    auto gm = manager(generation,"generation"); auto gt = gm->begin();
    refuses([&] { gt->source(8192,8,rows,[] {}); });
    expect(!gm->healthy(),"stale mapping generation invalidates certificate"); gt->cancel();
}

void bf16_reader_binding() {
    std::vector<std::byte> table(320U*100U);
    for (std::size_t i=0; i<table.size(); ++i) table[i]=static_cast<std::byte>(i%251U);
    ManifestTensor tensor;
    tensor.role=TensorRole::ngram_table; tensor.ggml_type=30; tensor.dimensions={160,100}; tensor.byte_size=table.size();
    RuntimeTensorBinding runtime;
    runtime.storage=RuntimeTensorStorage::file_mapped;
    runtime.address=reinterpret_cast<std::uintptr_t>(table.data()); runtime.bytes=table.size(); runtime.allocation_generation=23;
    const std::array<std::uint64_t,3> ids{12,13,12}; // straddling, repeated, order-sensitive
    FileMappedPleLookupBackend reference, target;
    std::vector<std::byte> warm(ids.size()*320U), leased(warm.size());
    reference.finish(reference.start({ids,320,&tensor,&runtime}),warm);
    Fake fake;
    auto io=fake.io(); io.reclaimable=[](auto) { return true; };
    auto owner=std::make_shared<ResidencyManager>("bf16-reader",runtime.address,23,256,4096,0,320,100,16,
        1U<<20U,32768,1,std::move(io));
    auto turn=owner->begin(); target.bind_residency(turn);
    const auto h=target.start({ids,320,&tensor,&runtime});
    refuses([&] { target.bind_residency(nullptr); });
    target.finish(h,leased);
    expect(warm==leased,"explicit MTP-target BF16 source binding preserves every byte and head order");
    target.cancel(target.start({ids,320,&tensor,&runtime})); // cancellation before any table read
    turn->phase(true);
    turn->gather(ids,[&] { target.finish(target.start({ids,320,&tensor,&runtime}),leased); });
    turn->phase(false);
    target.finish(target.start({ids,320,&tensor,&runtime}),leased); // speculative/replay lookup, same immutable source
    expect(warm==leased,"prefill->decode BF16 reader bytes unchanged");
    for (const auto positions : {2U,4U,8U,2U}) {
        std::vector<std::uint64_t> window_ids;
        for (std::uint32_t row=0;row<positions;++row)
            for (std::uint32_t head=0;head<16U;++head) window_ids.push_back((row*7U+head)%100U);
        turn->verify_window(runtime.address,23,positions,window_ids,[&] {
            for (std::uint32_t row=0;row<positions;++row) {
                const auto ids_row=std::span<const std::uint64_t>(window_ids).subspan(row*16U,16U);
                std::vector<std::byte> baseline(16U*320U), actual(baseline.size());
                reference.finish(reference.start({ids_row,320,&tensor,&runtime}),baseline);
                target.finish(target.start({ids_row,320,&tensor,&runtime}),actual);
                expect(baseline==actual,"adaptive BF16 window bytes and head destinations unchanged");
            }
        });
    }
    // Decode warming (draft-phase PLE prefetch): the verified pages are already in the decode cache, so warming
    // reads nothing and the next lookup copies the same bytes.
    expect(turn->warm(runtime.address,23,ids)==0,"decode warming finds cached verify pages and reads nothing");
    target.finish(target.start({ids,320,&tensor,&runtime}),leased);
    expect(warm==leased,"decode warming leaves the leased BF16 bytes unchanged");
    turn->phase(true);
    refuses([&] { (void)turn->warm(runtime.address,23,ids); });
    turn->phase(false);
    turn->finish(); target.bind_residency(nullptr);
    auto late=owner->begin(); target.bind_residency(late);
    const auto stale=target.start({ids,320,&tensor,&runtime});
    late->phase(true);
    refuses([&] { target.finish(stale,leased); });
    target.cancel(stale); late->cancel();
    expect(!owner->healthy(),"lookup handle retains start-phase epoch through deferred finish");
    target.bind_residency(nullptr);
}
void accounting() {
    expect(rolling_positions(nullptr,nullptr,nullptr) == 1024, "rolling default");
    expect(rolling_positions("0",nullptr,"bad") == 0 && rolling_positions(nullptr,"0","bad") == 0, "off switches");
    for (const auto* text : {"0","255","257","131072","1024x","-1",""})
        refuses([&] { (void)rolling_positions(nullptr,nullptr,text); });
    for (const auto* text : {"256","1024","65536"}) expect(rolling_positions(nullptr,nullptr,text) > 0, "valid H");
    ModelContract model;
    const auto contract = build_ple_hash_contract(model);
    Config c;
    // Model assets may pad the table beyond the minimum hash vocabulary.
    c.table_rows = 320001536; c.stored_row_bytes = 90;
    c.mapping_bytes = c.table_rows * 90; c.decode_allowance_bytes = decode_allowance(64,1,16,4096);
    c.reclaim = reclaim_intervals(0,c.mapping_bytes,4096,{});
    c.rolling_positions = 1024;
    const std::array<std::uint32_t,4> ids{1,2,3,4};
    const auto plan = plan_prompt(c,contract,ids);
    const auto budget = rolling_budget(c, contract.head_count);
    expect(plan.stats.touched_bytes == (3ULL << 30U) && plan.stats.admitted_bytes == budget.admitted,
        "shared cap plus decode plus margin admission");
    expect(c.mapping_bytes == 28800138240ULL && c.mapping_bytes - plan.stats.credit_bytes == budget.admitted + 3072 &&
        plan.stats.credit_bytes == 25193033728ULL, "3 GiB cap admission including table-edge bytes");
    c.rolling_positions = 0;
    const auto old = plan_prompt(c,contract,ids);
    c.rolling_positions = rolling_positions("0",nullptr,nullptr);
    const auto off = plan_prompt(c,contract,ids);
    expect(old.rows == off.rows && old.stats.credit_bytes == off.stats.credit_bytes &&
        old.stats.admitted_bytes == off.stats.admitted_bytes && old.stats.touched_bytes == off.stats.touched_bytes,
        "off switch retains prompt-wide rows, touched set, bound and credit");
}
#if defined(__linux__)
struct Table {
    int fd = -1;
    std::byte* base = nullptr;
    std::size_t bytes;
    explicit Table(std::size_t extent, bool sparse = false) : bytes((extent+4095U)/4096U*4096U) {
        char path[] = "/tmp/strix-ple-rolling-XXXXXX";
        fd = mkstemp(path);
        if (fd < 0) throw std::runtime_error("mkstemp");
        unlink(path);
        if (sparse) {
            if (ftruncate(fd,static_cast<off_t>(bytes))) throw std::runtime_error("sparse temporary table extent");
        } else {
        std::vector<std::byte> data(bytes);
        // Valid, varied immutable IQ4_NL blocks at a misaligned table offset.
        for (std::size_t at = 37; at + 18 <= extent; at += 18) {
            data[at] = std::byte{0}; data[at+1] = std::byte{0x3c};
            for (std::size_t k = 2; k < 18; ++k) data[at+k] = static_cast<std::byte>((at+k*17U)&255U);
        }
        std::size_t done = 0;
        while (done < data.size()) {
            const auto n = write(fd,data.data()+done,data.size()-done);
            if (n <= 0) throw std::runtime_error("temporary table write");
            done += static_cast<std::size_t>(n);
        }
        if (fsync(fd)) throw std::runtime_error("temporary table fsync");
        }
        auto* mapping = mmap(nullptr,bytes,PROT_READ,MAP_SHARED,fd,0);
        if (mapping == MAP_FAILED) throw std::runtime_error("temporary table mmap");
        base = static_cast<std::byte*>(mapping);
    }
    ~Table() { if (base) munmap(base,bytes); if (fd >= 0) close(fd); }
};
void direct_config(Config& c, const Table& file) {
    struct stat backing{};
    if (::fstat(file.fd,&backing)) throw std::runtime_error("temporary table fstat");
    c.file_device = static_cast<std::uint64_t>(backing.st_dev);
    c.file_inode = static_cast<std::uint64_t>(backing.st_ino);
    const auto path = "/proc/self/fd/" + std::to_string(file.fd);
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    void* aligned = nullptr;
    if (posix_memalign(&aligned, 4096, 4096)) throw std::bad_alloc();
    const bool supported = fd >= 0 && ::pread(fd, aligned, 4096, 0) >= 0;
    std::free(aligned);
    if (fd >= 0) ::close(fd);
    if (supported) {
        c.open_direct_read_only = [path] {
            const int opened = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
            if (opened < 0) throw std::runtime_error("PLE O_DIRECT: test reopen failed");
            return opened;
        };
    } else {
        std::cout << "O_DIRECT unsupported by test filesystem: using explicit pread test seam\n";
        c.test_direct_read = [&file](int, void* out, std::size_t n, std::uint64_t at) {
            return static_cast<std::int64_t>(::pread(file.fd, out, n, static_cast<off_t>(at)));
        };
    }
}
struct PreflightMapping {
    std::shared_ptr<Table> file;
    std::function<int()> open;
    std::uintptr_t base_address() const { return reinterpret_cast<std::uintptr_t>(file->base); }
    std::uint64_t bytes() const { return file->bytes; }
    std::array<std::uint64_t,2> file_backing_identity() const { return {1U,static_cast<std::uint64_t>(file->fd)}; }
    std::uint64_t allocation_generation() const { return 1U; }
    int open_direct_read_only() const { return open ? open() : -1; }
    int drop_page_cache(std::uint64_t offset, std::uint64_t count) const {
        return posix_fadvise(file->fd,static_cast<off_t>(offset),static_cast<off_t>(count),POSIX_FADV_DONTNEED);
    }
};
void verified_preflight_mapping(const PleHashContract& contract, std::span<const std::uint32_t> prompt) {
    auto file = std::make_shared<Table>(28800138240ULL,true);
    auto mapping = std::make_shared<PreflightMapping>(); mapping->file = file;
    Config c;
    c.table_rows=320001536; c.stored_row_bytes=90; c.rolling_positions=1024;
    c.reclaim=reclaim_intervals(0,c.table_rows*90,c.page,{});
    c.decode_allowance_bytes=decode_allowance(64,1,16,4096);
    direct_config(c,*file);
    unsigned opens=0;
    mapping->open=[&,open=c.open_direct_read_only] { ++opens; return open ? open() : -1; };
    c.open_direct_read_only={};
    c.credit_only=true;
    std::weak_ptr<const PreflightMapping> lifetime=mapping;
    decode_cli::bind_ple_reclaim_mapping(c,std::shared_ptr<const PreflightMapping>(mapping));
    mapping.reset();
    expect(!lifetime.expired(),"preflight callbacks pin mapped backing");
    for (const auto mode : {0U,1U,2U}) {
        auto setup=c; setup.rolling_positions=mode?1024U:0U; setup.bounded_turns=mode==2U;
        const auto expected=plan_prompt(setup,contract,prompt).stats.credit_bytes;
        Prefetcher verified(setup,contract,prompt,true);
        expect(expected && verified.credit_bytes()==expected,"prompt-wide serial, rolling serial and leased MTP verify credit before admission");
        if(mode) expect(verified.credit_bytes()==25193033728ULL,"rolling verified credit matches supplied operands");
        expect(opens==mode,"each rolling preflight opens the bound direct backing exactly once");
        expect(verified.owner_bytes()==planned_owner_bytes(setup,contract.head_count,prompt.size()),
            "predicted owner includes the same row, bitmap, direct reader and turn ledger capacities as runtime");
        expect(!verified.claim_stage_prime(),"credit-only setup cannot prime staging");
        std::vector<std::uint32_t> staged;
        expect(!verified.stage_rows(0,1,staged),"credit-only setup offers no staging rows");
        refuses([&] { (void)verified.begin_turn(); });
        refuses([&] { verified.stage_begin(); });
        verified.finish();
    }
    for (const bool mtp : {false,true}) {
        auto only=c; only.bounded_turns=mtp; only.open_direct_read_only={}; only.ahead_rows=0;
        std::atomic<unsigned> reads{0};
        std::mutex blocked_mutex; std::condition_variable blocked_wake; bool release=false;
        only.test_direct_read=[&](int,void* out,std::size_t n,std::uint64_t at) {
            if (reads.fetch_add(1)) {
                std::unique_lock lock(blocked_mutex);
                blocked_wake.wait(lock,[&] { return release; }); // any prompt read blocks owner retirement
            }
            return static_cast<std::int64_t>(::pread(file->fd,out,n,static_cast<off_t>(at)));
        };
        // User threads only: io_uring's kernel workers (comm "iou-*") are listed under the process as well.
        const auto thread_count=[] {
            std::ptrdiff_t n=0;
            for (const auto& task : std::filesystem::directory_iterator("/proc/self/task")) {
                std::ifstream comm(task.path()/"comm"); std::string name; std::getline(comm,name);
                if (!name.starts_with("iou-")) ++n;
            }
            return n;
        };
        auto construction=std::async(std::launch::async,[&] {
            const auto before=thread_count();
            { Prefetcher owner(only,contract,prompt,true);
              expect(thread_count()==before,"credit-only setup starts no helper threads");
              expect(owner.credit_bytes()==25193033728ULL,"credit-only serial/MTP preserve baseline sparse credit"); }
        });
        const bool completed=construction.wait_for(std::chrono::seconds(5))==std::future_status::ready;
        { const std::lock_guard lock(blocked_mutex); release=true; }
        blocked_wake.notify_all(); construction.get();
        expect(completed && reads==1,"blocked prompt source cannot delay credit-only construction/destruction");
    }
    {
        auto setup=c;setup.bounded_turns=true;
        std::unique_ptr<Prefetcher> trial;
        const auto verify=[&] {
            return decode_cli::verify_fresh_ple_preflight(trial,[&] {
                trial=std::make_unique<Prefetcher>(setup,contract,prompt,true);
                return trial->credit_bytes();
            });
        };
        refuses([&] {
            expect(verify()>0,"first storage trial verifies credit");
            throw std::runtime_error("storage step exceeds admission envelope");
        });
        expect(trial!=nullptr,"refused trial retains preflight until next attempt");
        setup.rolling_positions=256;
        expect(verify()>0,"fitting storage trial replaces refused authority");
        refuses([&] {Prefetcher duplicate(setup,contract,prompt,true);});
        expect(trial->credit_bytes()>0,"exactly one live authority after replacement");
        decode_cli::verify_fresh_ple_preflight(trial,[] {return 0U;});
        expect(!trial,"unavailable next route also releases authority");
        Prefetcher fresh(setup,contract,prompt,true);
        expect(fresh.credit_bytes()>0,"released authority can be registered again");
    }
    auto missing=c; missing.open_direct_read_only={}; missing.test_direct_read={};
    refuses([&] { Prefetcher failed(missing,contract,prompt,true); });
    auto failed=c; failed.drop_page_cache=[](auto,auto) { return EIO; };
    refuses([&] { Prefetcher rejected(failed,contract,prompt,true); });
    for (const auto required : {false,true}) for (const int fault : {EINVAL,EIO,0}) {
        auto broken=c;
        broken.test_direct_read=[fault](int,void*,std::size_t,std::uint64_t) -> std::int64_t {
            if(fault) { errno=fault; return -1; } return 0; // unsupported, hard error, short EOF
        };
        unsigned stage=0; bool warmed=false,refused=false;
        std::uint64_t credit=123;
        try {
            auto resident=decode_cli::setup_ple_or_resident(required?Mode::required:Mode::automatic,
                [&] { return std::make_shared<Prefetcher>(broken,contract,prompt,true); },
                [&] { expect(stage++==0,"teardown before resident admission"); credit=0; },
                [&] { expect(stage++==1 && !credit,"resident admission observes zero credit after teardown"); },
                [&] { expect(stage++==2,"warm only after resident admission"); warmed=true; });
            expect(!resident,"failed direct setup never retains a rolling owner");
        } catch(const std::runtime_error& error) {
            refused=std::string(error.what()).find("required PLE setup refused before prompt mutation")!=std::string::npos;
        }
        expect(required ? refused && !warmed && stage==0 : !refused && warmed && stage==3,
            "all automatic direct setup errors fall back; required errors refuse clearly");
        if(!required) {
            warmed=false;
            refuses([&] { (void)decode_cli::setup_ple_or_resident(Mode::automatic,
                [&] { return std::make_shared<Prefetcher>(broken,contract,prompt,true); },
                [&] { credit=0; }, [] { throw std::runtime_error("resident does not fit"); },
                [&] { warmed=true; }); });
            expect(!warmed && !credit,"resident refusal prevents warm mapping mutation");
        }
    }
    auto unverified=c; unverified.drop_residual_bytes=0;
    unverified.drop_page_cache=[&](auto offset,auto count) {
        const int error=c.drop_page_cache(offset,count);
        // Reintroduce one resident page after the advisory drop. Real mincore
        // must reject this setup even though the drop callback reports success.
        const volatile std::byte resident=file->base[offset]; (void)resident;
        return error;
    };
    refuses([&] { Prefetcher rejected(unverified,contract,prompt,true); });
}
void verified_credit() {
    // Sparse temporary backing proves the real setup callback seam without a
    // model, GPU, table-sized allocation or table-sized disk write.
    Table file(28800138240ULL,true);
    Config c;
    c.mapping_base = reinterpret_cast<std::uintptr_t>(file.base); c.mapping_bytes = file.bytes;
    c.table_rows = 320001536; c.stored_row_bytes = 90; c.rolling_positions = 1024;
    c.reclaim = reclaim_intervals(0,c.table_rows*90,c.page,{});
    c.decode_allowance_bytes = decode_allowance(64,1,16,4096);
    c.drop_page_cache = [&](auto offset, auto bytes) { return posix_fadvise(file.fd,
        static_cast<off_t>(offset),static_cast<off_t>(bytes),POSIX_FADV_DONTNEED); };
    direct_config(c, file);
    const auto contract = build_ple_hash_contract(ModelContract{});
    const std::array<std::uint32_t,4> prompt{1,2,3,4};
    {
        auto overlap=c;
        bool dropped=false;
        overlap.strict=false; // Diagnostic mode cannot bypass discard safety.
        overlap.registered_backings.push_back({"separate-file-alias",c.mapping_base+c.mapping_bytes+4096,4096,
            c.file_device,c.file_inode,c.reclaim.front().begin});
        overlap.drop_page_cache=[&](auto,auto) { dropped=true; return 0; };
        bool safety_refused=false;
        try { Prefetcher rejected(overlap,contract,prompt,true); }
        catch (const DiscardSafetyError&) { safety_refused=true; }
        expect(safety_refused && !dropped,"file alias safety refuses before any discard, even with strict off");
    }
    verified_preflight_mapping(contract,prompt);
    Prefetcher verified(c,contract,prompt);
    expect(verified.credit_bytes() == 25193033728ULL, "runtime getter exposes rolling credit only after verified setup");
    expect(verified.claim_stage_prime() && !verified.claim_stage_prime(), "only one setup staging prime across owners");
    const auto* prior_stats = std::getenv("STRIX_PLE_ROLLING_STATS");
    const std::string saved_stats = prior_stats ? prior_stats : "";
    const bool had_stats = prior_stats != nullptr;
    auto* capture = std::tmpfile();
    expect(capture != nullptr, "stats capture file");
    std::fflush(stderr);
    const auto saved_stderr = dup(fileno(stderr));
    expect(saved_stderr >= 0 && dup2(fileno(capture), fileno(stderr)) >= 0, "redirect stderr for stats parser");
    std::array<PleHashResult, 4> hashes;
    std::array<std::uint32_t, PleHashContract::maximum_ngram - 1U> context{};
    std::fill_n(context.begin(), contract.ngram_size - 1U, contract.eos_token_id);
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        hashes[i] = hash_ple_token(contract, prompt[i], std::span<const std::uint32_t>(context.data(), contract.ngram_size - 1U));
        context = hashes[i].next_context;
    }
    Prefetcher::GatherReceipt metrics;
    metrics.bytes_read = 8192; metrics.read_wait_ns = 2000000; metrics.copy_ns = 3000000;
    metrics.hash_ns = 4000000; metrics.stage_wait_ns = 5000000;
    setenv("STRIX_PLE_ROLLING_STATS", "0", 1);
    verified.on_gather(0, hashes, metrics);
    std::fflush(stderr);
    const auto silent = std::ftell(capture);
    setenv("STRIX_PLE_ROLLING_STATS", "1", 1);
    verified.on_gather(0, hashes, metrics);
    std::fflush(stderr);
    expect(dup2(saved_stderr, fileno(stderr)) >= 0, "restore stderr"); close(saved_stderr);
    if (had_stats) setenv("STRIX_PLE_ROLLING_STATS", saved_stats.c_str(), 1); else unsetenv("STRIX_PLE_ROLLING_STATS");
    std::rewind(capture);
    char line[512]{};
    expect(std::fgets(line, sizeof(line), capture) != nullptr, "one block stats line");
    unsigned long long block = 0, first = 0, count = 0, bytes = 0;
    double read_ms = 0, copy_ms = 0, hash_ms = 0, join_ms = 0, critical_ms = 0, iops = 0, mb_s = 0;
    unsigned long long critical_bytes = 0, ahead_bytes = 0;
    const auto fields = std::sscanf(line,
        "ple-rolling block: block=%llu first=%llu rows=%llu bytes-read=%llu read-wait-ms=%lf copy-ms=%lf hash-ms=%lf join-wait-ms=%lf critical-bytes=%llu critical-read-ms=%lf critical-iops=%lf critical-MB-s=%lf lookahead-bytes=%llu",
        &block, &first, &count, &bytes, &read_ms, &copy_ms, &hash_ms, &join_ms, &critical_bytes, &critical_ms, &iops, &mb_s, &ahead_bytes);
    expect(silent == 0 && fields == 13 && block == 1 && first == 0 && count == 4 && bytes == 8192 &&
        read_ms == 2 && copy_ms == 3 && hash_ms == 4 && join_ms == 5 && ahead_bytes == 0 && (critical_bytes ? critical_ms > 0 && iops > 0 && mb_s > 0 : iops == 0 && mb_s == 0) && std::fgetc(capture) == EOF,
        "STATS-only production block line parses with exact units and no extra lines");
    std::fclose(capture);
    verified.stage_begin();
    std::promise<void> finishing;
    auto retired = std::async(std::launch::async,[&] { finishing.set_value(); verified.finish(); });
    finishing.get_future().wait();
    const bool waiting = retired.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    verified.stage_end(); retired.get();
    expect(waiting, "turn retirement drains a final c1 task before releasing source pages");
    refuses([&] { verified.stage_begin(); });

    {
        auto priming = c;
        priming.bounded_turns = true; priming.backing_identity = "mtp-prompt-prime-fixture"; priming.mapping_generation = 42;
        priming.ahead_rows = 2048; priming.critical_rows = 1024; priming.open_direct_read_only = {};
        std::mutex read_mutex; std::condition_variable read_wake;
        bool armed = false, entered = false, release = false;
        std::uint64_t reads = 0;
        priming.test_direct_read = [&](int, void* out, std::size_t read_bytes, std::uint64_t offset) {
            {
                std::unique_lock lock(read_mutex); ++reads;
                if (armed) { entered = true; read_wake.notify_all(); read_wake.wait(lock,[&] { return release; }); }
            }
            return static_cast<std::int64_t>(::pread(file.fd,out,read_bytes,static_cast<off_t>(offset)));
        };
        auto owner = std::make_shared<Prefetcher>(priming,contract,prompt);
        auto prompt_turn = owner->begin_turn();
        { const std::lock_guard lock(read_mutex); armed = true; }
        prompt_turn->phase(true);
        bool early;
        {
            std::unique_lock lock(read_mutex);
            early = read_wake.wait_for(lock,std::chrono::seconds(5),[&] { return entered; });
            release = true;
        }
        read_wake.notify_all();
        prompt_turn->gather_prompt(0,prompt.size(),[&](auto source_first,auto source_count) {
            const auto view = prompt_turn->prompt_view(source_first,source_count);
            std::uint64_t before;
            { const std::lock_guard lock(read_mutex); before = reads; }
            std::array<std::byte,90> row;
            for (auto p=source_first;p<source_first+source_count;++p) for (std::uint32_t h=0;h<contract.head_count;++h)
                view->copy(priming.table_offset + hashes[p].row_ids[h]*90U,row);
            { const std::lock_guard lock(read_mutex); expect(reads==before,"MTP prompt row copies issue no per-row reads"); }
        });
        prompt_turn->phase(false); prompt_turn->finish(); owner->finish();
        expect(early,"MTP prompt critical reads start ahead of the first gather");
    }
    auto managed_config=c;
    managed_config.bounded_turns=true; managed_config.backing_identity="verified-production-table"; managed_config.mapping_generation=41;
    auto managed=std::make_shared<Prefetcher>(managed_config,contract,prompt);
    expect(managed->credit_bytes()==25193033728ULL,"leased MTP setup verifies the same single table credit");
    auto turn=managed->begin_turn();
    turn->phase(true);
    turn->stage_begin(); // registered before the thread can be published
    std::promise<void> staged_entered, staged_release;
    auto released = staged_release.get_future();
    auto staged = std::async(std::launch::async,[&] {
        staged_entered.set_value(); released.wait(); turn->stage_end();
    });
    staged_entered.get_future().wait();
    std::promise<void> phase_entered;
    auto phase = std::async(std::launch::async,[&] { phase_entered.set_value(); turn->phase(false); });
    phase_entered.get_future().wait();
    const bool drains_stage = phase.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    staged_release.set_value(); staged.get(); phase.get();
    expect(drains_stage && !turn->prompt(), "MTP prompt-to-decode drains an unconsumed c1 task");
    turn->phase(true);
    const auto staged_epoch = turn->epoch();
    turn->stage_begin();
    refuses([&] { turn->gather_prompt(0,prompt.size(),[](auto,auto) {},staged_epoch+1U); });
    std::promise<void> cancel_entered;
    auto cancelled = std::async(std::launch::async,[&] { cancel_entered.set_value(); turn->cancel(); });
    cancel_entered.get_future().wait();
    const bool drains_cancel = cancelled.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    bool copied_during_cancel = false;
    std::exception_ptr copy_error;
    try {
        turn->gather_prompt(0,prompt.size(),[&](auto source_first,auto source_count) {
            const auto view = turn->prompt_view(source_first,source_count);
            copied_during_cancel = turn->owns_prompt_view(*view,managed_config.mapping_base,41,turn->epoch(),0);
        },staged_epoch);
    } catch (...) { copy_error = std::current_exception(); }
    turn->stage_end(copy_error); cancelled.get();
    if (copy_error) std::rethrow_exception(copy_error);
    expect(drains_cancel && copied_during_cancel,
        "MTP cancellation drains registered epoch-bound c1 copies before retiring prompt pages");
    turn=managed->begin_turn();
    refuses([&] { turn->source(managed_config.mapping_base+1U,41,std::array<std::uint64_t,1>{0},[] {}); });
    expect(managed->credit_bytes()==0 && managed->stats().credit_bytes==0,"invalid table certificate revokes every credit getter");
    turn->cancel();
    refuses([&] { (void)managed->begin_turn(); });
    c.strict = false;
    Prefetcher mismatch(c,contract,prompt);
    std::array<std::uint32_t,2> eos{contract.eos_token_id,contract.eos_token_id};
    auto wrong = hash_ple_token(contract,prompt[0],eos);
    wrong.row_ids[0] ^= 1U;
    refuses([&] { mismatch.on_gather(0,std::span<const PleHashResult>(&wrong,1),{}); });
    auto rejected = c;
    bool dropped = false;
    rejected.open_direct_read_only = {};
    rejected.test_direct_read = [](int, void*, std::size_t, std::uint64_t) { errno = EOPNOTSUPP; return std::int64_t{-1}; };
    rejected.drop_page_cache = [&](auto,auto) { dropped = true; return 0; };
    refuses([&] { Prefetcher failed(rejected,contract,prompt,true); });
    expect(!dropped, "unsupported O_DIRECT refuses before drop or credit");
    rejected = c;
    rejected.drop_page_cache = [](auto,auto) { return 5; }; // EIO, deterministic callback failure
    refuses([&] { Prefetcher failed(rejected,contract,prompt,true); });
    rejected = c; rejected.mapping_base += 1;
    refuses([&] { Prefetcher failed(rejected,contract,prompt,true); });
}
std::uint64_t digest(std::span<const std::byte> bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (auto b : bytes) hash = (hash ^ std::to_integer<unsigned char>(b)) * 1099511628211ULL;
    return hash;
}
void long_file() {
    auto model = ModelContract{};
    model.ngram_vocab_size_base = 4096;
    const auto contract = build_ple_hash_contract(model);
    Table file(static_cast<std::size_t>(37 + contract.minimum_table_rows*90));
    Config c;
    c.mapping_base = reinterpret_cast<std::uintptr_t>(file.base); c.mapping_bytes = file.bytes;
    c.table_offset = 37; c.table_rows = contract.minimum_table_rows; c.stored_row_bytes = 90;
    c.decode_allowance_bytes = decode_allowance(64,1,contract.head_count,4096);
    c.reclaim = reclaim_intervals(c.table_offset,c.table_rows*90,c.page,{});
    std::uint64_t drops = 0;
    c.drop_page_cache = [&](auto offset, auto bytes) { ++drops; return posix_fadvise(file.fd,
        static_cast<off_t>(offset),static_cast<off_t>(bytes),POSIX_FADV_DONTNEED); };
    direct_config(c, file);
    // Keep the production residual allowance: a filesystem may retain a folio
    // shared with either excluded boundary page after DONTNEED.
    std::vector<std::uint32_t> tokens(200000);
    std::uint32_t random = 12345;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        random = random * 1664525U + 1013904223U;
        tokens[i] = i % 251U == 0 ? contract.eos_token_id : random % 100000U;
    }
    const auto old = plan_prompt(c,contract,tokens);
    std::array<std::uint32_t, PleHashContract::maximum_ngram - 1U> context{};
    std::fill_n(context.begin(), contract.ngram_size - 1U, contract.eos_token_id);
    for (std::size_t first = 0; first < tokens.size();) {
        const auto n = std::min<std::size_t>((first / 2048U) % 2U ? 2048U : 8192U, tokens.size() - first);
        std::vector<PleHashResult> hashes(n);
        hash_ple_tokens(contract, std::span<const std::uint32_t>(tokens).subspan(first, n),
            std::span<const std::uint32_t>(context.data(), contract.ngram_size - 1U), hashes);
        for (std::size_t i = 0; i < n; ++i) {
            const auto serial = hash_ple_token(contract, tokens[first + i],
                std::span<const std::uint32_t>(context.data(), contract.ngram_size - 1U));
            expect(hashes[i].row_ids == serial.row_ids && hashes[i].next_context == serial.next_context &&
                hashes[i].context_count == serial.context_count && hashes[i].row_count == serial.row_count,
                "parallel hashes and next history match serial including EOS and block boundaries");
            for (std::uint32_t h = 0; h < contract.head_count; ++h)
                expect(hashes[i].row_ids[h] == old.rows[(first + i) * contract.head_count + h], "parallel hash matches prompt planner");
            context = serial.next_context;
        }
        first += n;
    }
    ManifestTensor tensor;
    tensor.name = "per_layer_token_embd.weight"; tensor.role = TensorRole::ngram_table;
    tensor.ggml_type = 20; tensor.dimensions = {160,c.table_rows}; tensor.byte_size = c.table_rows*90;
    RuntimeTensorBinding binding;
    binding.storage = RuntimeTensorStorage::file_mapped; binding.address = c.mapping_base+c.table_offset;
    binding.bytes = tensor.byte_size; binding.allocation_generation = 1;
    std::vector<std::uint64_t> reference(tokens.size()*contract.head_count);
    std::vector<std::byte> row_reference(c.table_rows*contract.row_bytes);
    FileMappedPleLookupBackend backend;
    std::vector<std::byte> gathered(contract.head_count*contract.row_bytes);
    const auto copy = [&](std::uint64_t p, bool compare) {
        std::vector<std::uint64_t> ids(old.rows.begin()+static_cast<std::ptrdiff_t>(p*contract.head_count),
            old.rows.begin()+static_cast<std::ptrdiff_t>((p+1)*contract.head_count));
        const auto handle = backend.start({ids,contract.row_bytes,&tensor,&binding});
        backend.finish(handle,gathered);
        for (std::uint32_t h = 0; h < contract.head_count; ++h) {
            const auto hash = digest(std::span<const std::byte>(gathered).subspan(h*contract.row_bytes,contract.row_bytes));
            auto& expected = reference[static_cast<std::size_t>(p*contract.head_count+h)];
            if (compare) {
                expect(hash == expected, "every gathered IQ4_NL row matches non-rolling hash");
                expect(std::memcmp(gathered.data()+h*contract.row_bytes,
                    row_reference.data()+ids[h]*contract.row_bytes,contract.row_bytes)==0,
                    "every gathered byte matches the non-rolling row at its original destination offset");
            } else {
                expected = hash;
                std::memcpy(row_reference.data()+ids[h]*contract.row_bytes,
                    gathered.data()+h*contract.row_bytes,contract.row_bytes);
            }
        }
    };
    for (std::uint64_t p = 0; p < tokens.size(); ++p) copy(p,false);
    if (mprotect(file.base, file.bytes, PROT_NONE)) throw std::runtime_error("protect rolling source mapping");
    c.rolling_positions = 1024;
    c.ahead_rows = 16384; c.critical_rows = 8192;
    Prefetcher rolling(c,contract,tokens);
    const auto setup_drops = drops;
    set_active(&rolling);
    for (std::uint64_t first = 0; first < tokens.size(); first += c.rolling_positions) {
        const auto n = std::min<std::uint64_t>(c.rolling_positions,tokens.size()-first);
        rolling.gather_range(first,n,[&](auto begin, auto count) {
            const auto source = rolling.prompt_source(begin, count);
            std::vector<std::byte> staging(count * contract.head_count * contract.row_bytes);
            std::vector<std::future<void>> workers;
            for (unsigned worker = 0; worker < 8; ++worker) workers.push_back(std::async(std::launch::async, [&, worker] {
                FileMappedPleLookupBackend helper;
                helper.bind_prompt_source(source.view, source.table, source.offset);
                for (auto pos = begin + worker; pos < begin + count; pos += 8U) {
                    std::vector<std::uint64_t> ids(old.rows.begin() + static_cast<std::ptrdiff_t>(pos * contract.head_count),
                        old.rows.begin() + static_cast<std::ptrdiff_t>((pos + 1U) * contract.head_count));
                    const auto handle = helper.start({ids, contract.row_bytes, &tensor, &binding});
                    auto destination = std::span<std::byte>(staging).subspan((pos - begin) * contract.head_count * contract.row_bytes,
                        contract.head_count * contract.row_bytes);
                    helper.finish(handle, destination);
                    for (std::uint32_t h = 0; h < contract.head_count; ++h)
                        expect(std::memcmp(destination.data() + h * contract.row_bytes,
                            row_reference.data() + ids[h] * contract.row_bytes, contract.row_bytes) == 0 &&
                            digest(destination.subspan(h * contract.row_bytes, contract.row_bytes)) == reference[pos * contract.head_count + h],
                            "parallel plain prompt view preserves every byte at the serial destination offset");
                }
            }));
            for (auto& worker : workers) worker.get();
        });
        std::vector<PleHashResult> receipt(n);
        for (std::uint64_t i = 0; i < n; ++i) {
            receipt[i].row_count = contract.head_count;
            std::copy_n(old.rows.begin()+static_cast<std::ptrdiff_t>((first+i)*contract.head_count),
                contract.head_count, receipt[i].row_ids.begin());
        }
        rolling.on_gather(first, receipt, {});

    }
    // Decode uses the same source hook and is reclaimed at this turn's end.
    for (std::uint64_t pos=0; pos<tokens.size(); ++pos) copy(pos,true);
    const std::array<std::uint64_t,1> late_ids{old.rows[0]};
    const auto late = backend.start({late_ids,contract.row_bytes,&tensor,&binding});
    set_active(nullptr);
    refuses([&] { backend.finish(late,gathered); });
    backend.cancel(late);
    set_active(&rolling);
    rolling.finish(); set_active(nullptr);
    expect(drops == setup_drops, "plain rolling has no hot-path fadvise/drop callbacks");
    expect(rolling.stats().resident_last_bytes <= rolling.stats().resident_after_drop_bytes + (64U<<10U),
        "plain direct reads do not repopulate the mapping beyond setup residual folios");
    std::cout << "plain mapping resident setup/end bytes: " << rolling.stats().resident_after_drop_bytes << '/'
        << rolling.stats().resident_last_bytes << '\n';
    // A new turn starts with EOS context, fresh allowance and empty epoch state.
    Prefetcher next(c,contract,tokens);
    next.finish();

    c.bounded_turns = true; c.backing_identity = "IQ4-operator-fixture"; c.mapping_generation = 1;
    auto leased = std::make_shared<Prefetcher>(c,contract,tokens);
    const auto mtp_setup_drops = drops;
    auto turn = leased->begin_turn();
    backend.bind_residency(turn);
    copy(0,true); copy(1,true); // calibration/serial target path
    turn->phase(true);
    for (std::uint64_t first = 0, block = 0; first < tokens.size(); ++block) {
        const auto count = std::min<std::uint64_t>(block % 2U ? 2048U : 8192U, tokens.size()-first);
        std::vector<std::uint64_t> ids(old.rows.begin()+static_cast<std::ptrdiff_t>(first*contract.head_count),
            old.rows.begin()+static_cast<std::ptrdiff_t>((first+count)*contract.head_count));
        turn->gather_prompt(first,count,[&](auto chunk_first, auto chunk_count) {
            const auto view = turn->prompt_view(chunk_first,chunk_count);
            expect(turn->owns_prompt_view(*view,binding.address,binding.allocation_generation,turn->epoch(),c.table_offset),
                "MTP pinned view retains actual table, offset and phase");
            expect(!turn->owns_prompt_view(*view,binding.address,binding.allocation_generation,turn->epoch(),c.table_offset+1U),
                "MTP pinned view rejects shifted source offset");
            // The window's temporary file-mapped helper retains an explicit
            // turn binding; no process-global hook is installed for this test.
            std::vector<std::future<void>> workers;
            for (unsigned worker = 0; worker < 8; ++worker) workers.push_back(std::async(std::launch::async, [&, worker] {
                FileMappedPleLookupBackend helper;
                helper.bind_residency(turn);
                helper.bind_prompt_source(view,binding.address,c.table_offset);
                std::vector<std::byte> local_gathered(gathered.size());
                for (auto pos = chunk_first + worker; pos < chunk_first+chunk_count; pos += 8U) {
                    std::vector<std::uint64_t> row_ids(old.rows.begin()+static_cast<std::ptrdiff_t>(pos*contract.head_count),
                        old.rows.begin()+static_cast<std::ptrdiff_t>((pos+1U)*contract.head_count));
                    const auto handle = helper.start({row_ids,contract.row_bytes,&tensor,&binding});
                    helper.finish(handle,local_gathered);
                    for (std::uint32_t h=0; h<contract.head_count; ++h)
                        expect(digest(std::span<const std::byte>(local_gathered).subspan(h*contract.row_bytes,contract.row_bytes)) ==
                            reference[static_cast<std::size_t>(pos*contract.head_count+h)] &&
                            std::memcmp(local_gathered.data()+h*contract.row_bytes, row_reference.data()+row_ids[h]*contract.row_bytes,
                                contract.row_bytes)==0, "leased helper IQ4 bytes match warm reference");
                }
            }));
            for (auto& worker : workers) worker.get();
        });
        turn->observe(first,ids,0,0,0);
        first += count;
    }
    turn->phase(false);
    MtpMinPWindowController minp({2,4,8},0.7F);
    const std::array<std::uint64_t,3> horizons{2,4,8};
    std::array<std::uint64_t,3> visits{};
    std::uint64_t cycle=0;
    for (std::uint64_t pos = 0; pos+8U <= tokens.size();) {
        const auto wanted=static_cast<std::size_t>(cycle++%3U);
        std::array<std::uint8_t,3> fits{}; fits[wanted]=1;
        const std::vector<float> probabilities(horizons[wanted]-1U,0.99F);
        const auto chosen=minp.choose(probabilities,probabilities.size(),fits);
        expect(chosen==wanted,"actual min-p controller switches R2/R4/R8 during the stream");
        ++visits[chosen];
        const auto count=horizons[chosen];
        const std::vector<std::uint64_t> ids(old.rows.begin()+static_cast<std::ptrdiff_t>(pos*contract.head_count),
            old.rows.begin()+static_cast<std::ptrdiff_t>((pos+count)*contract.head_count));
        turn->verify_window(binding.address,binding.allocation_generation,count,ids,[&] {
            for (auto row=pos;row<pos+count;++row) copy(row,true);
        });
        // Rejection commits one input, rolls back the rest, and serial replay
        // must obtain the same immutable bytes at the old absolute positions.
        const auto accepted=cycle%3U==0U?0U:count-1U;
        for (auto row=pos+accepted;row<pos+count;++row) copy(row,true);
        minp.calibrate(probabilities,accepted,count-1U);
        minp.observe(chosen,accepted+1U,1.0+static_cast<double>(cycle%11U));
        pos+=count;
        if (pos+8U>tokens.size()) for (auto row=pos;row<tokens.size();++row) copy(row,true);
    }
    expect(*std::min_element(visits.begin(),visits.end())>10000U,"all adaptive owners exercised over the long stream");
    turn->finish();
    backend.bind_residency(nullptr);
    leased->finish();
    expect(drops == mtp_setup_drops, "MTP/decode has no hot-path fadvise/drop callbacks");
    expect(leased->stats().resident_last_bytes <= leased->stats().resident_after_drop_bytes + (64U<<10U),
        "MTP direct reads do not repopulate the mapping beyond setup residual folios");
    std::cout << "MTP mapping resident setup/end bytes: " << leased->stats().resident_after_drop_bytes << '/'
        << leased->stats().resident_last_bytes << '\n';
    auto again = leased->begin_turn(); again->phase(true); again->phase(false); again->finish();
    std::array<std::uint32_t,2> eos{contract.eos_token_id,contract.eos_token_id};
    const auto after_eos = hash_ple_token(contract,tokens[1],eos);
    expect(after_eos.row_ids[0] == old.rows[contract.head_count], "EOS resets hash history in long synthetic stream");
    std::cout << "200000 positions: " << reference.size() << " gathered row hashes and bytes identical\n";
}
#endif
} // namespace
int main() {
    try {
        windows(); completion(); pipeline(); leases(); bf16_reader_binding(); accounting();
#if defined(__linux__)
        verified_credit(); long_file();
#endif
        std::cout << "PLE rolling CPU tests passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
