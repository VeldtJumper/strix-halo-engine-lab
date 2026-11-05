#include "strix/cuda_qwen_prefix_identity.hpp"
#include "strix/cuda_qwen_prefix_store.hpp"
#include "strix/cuda_qwen_prefix_pool.hpp"

#include <iostream>
#include <fstream>
#include <numeric>
#include <regex>
#include <set>
#include <stdexcept>

namespace px = strix::cuda::qwen_prefix;
using namespace strix;

namespace {
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
PrefixCacheBuffer heap(std::uint64_t bytes) {
    return {std::shared_ptr<std::byte>(new std::byte[bytes], std::default_delete<std::byte[]>()), bytes};
}
ModelContentIdentity id(const std::string& text) { return sha256_bytes(std::as_bytes(std::span(text.data(), text.size()))); }
std::vector<std::uint32_t> prompt(std::size_t n, std::uint32_t base = 100) {
    std::vector<std::uint32_t> out(n);
    std::iota(out.begin(), out.end(), base);
    return out;
}

void planner() {
    for (unsigned n = 0; n <= 5; ++n) check(px::capture_points(n, 4, 0, 0).empty(), "tiny/T+1 capture");
    check(px::lookup_limit(0) == 0 && px::lookup_limit(1) == 0 && px::lookup_limit(6) == 4, "lookup limit");
    check(px::capture_points(6, 4, 0, 0) == std::vector<std::uint64_t>{4}, "T+2 first capture");
    check(px::capture_points(17, 4, 0, 0) == std::vector<std::uint64_t>{12}, "k*T+1 excludes final row");
    check(px::capture_points(18, 4, 0, 0) == std::vector<std::uint64_t>{16}, "k*T+2 includes boundary");
    check(px::capture_points(18, 4, 8, 0) == std::vector<std::uint64_t>{16}, "restore then capture");
    check(px::capture_points(18, 4, 16, 8).empty(), "nothing beyond restore");
    const auto points = px::capture_points(42, 4, 8, 12);
    check(points == std::vector<std::uint64_t>({12, 24, 36, 40}), "stride checkpoints");
    check(px::clip_chunk(8, 1024, points) == 4 && px::clip_chunk(12, 1024, points) == 12, "clip to next boundary");
    check(px::clip_chunk(24, 3, points) == 3 && px::clip_chunk(40, 1024, points) == 1024, "clip small/final");
    bool threw = false;
    try { (void)px::capture_points(40, 6, 0, 0); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "invalid block rejected");
    threw = false;
    try { (void)px::capture_points(40, 4, 0, 6); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "unaligned stride rejected");
    std::cout << "PASS planner: tiny, T+1, k*T+1, k*T+2, restore, stride, clipping, validation\n";
}

void identity() {
    strix::cuda::qwen_prefill::Config c;
    const std::string root = STRIX_PREFIX_TEST_SOURCE;
    c.model_gguf = root + "/CMakeLists.txt";
    c.mtp_sidecar_gguf = root + "/include/strix/cuda_qwen_prefill.hpp";
    c.draft_vocab = root + "/include/strix/cuda_qwen_prefix_plan.hpp";
    c.tables_dir = root + "/src/cuda";
    c.rope_table = root + "/include/strix/cuda_qwen_qsa.hpp";
    const auto fields = px::identity_fields(c, 256);
    const auto key = px::identity_key(fields);
    check(key.valid(), "valid SHA-256 identity");
    const auto has_field = [&](const std::string& name) {
        return std::any_of(fields.begin(), fields.end(), [&](const auto& field) { return field.first == name; });
    };
    for (const auto* name : {"layout", "model.path", "model.size", "model.mtime", "mtp.path", "mtp.size", "mtp.mtime",
                            "draft_vocab.path", "rope.path", "tables.path", "max_positions", "segment_rows", "block_rows", "ple_host_pinned"})
        check(has_field(name), "required compatibility field present");
    // Guard: every getenv("LITERAL") in the Qwen CUDA sources and headers (src/cuda/qwen_*.{cu,cuh,cpp,hpp},
    // include/strix/cuda_qwen_*.hpp, apps/cuda_qwen_*, include/strix/cuda_vram_guard.hpp) must be an identity field
    // or listed below, and no other getenv form may appear there (a variable or helper-wrapped name would not be
    // visible to this scan). qwen_prefix_identity.cpp is the reader.
    const std::regex getenv_pattern(R"re(getenv\("([^"]+)"\))re");
    const std::regex any_getenv(R"re(getenv\s*\()re");
    // Read only to decline startup (the Prefill constructor throws when the BF16 MoE route is requested), so no
    // saved state can depend on them.
    const std::set<std::string> refuse_only = {"STRIX_QPF2K_MOE_ROUTE_BF16", "STRIX_BASELINE", "STRIX_QPF2K_DEFAULTS"};
    const auto scan = [&](const std::string& dir, const std::string& prefix, const std::set<std::string>& exempt) {
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            const auto name = entry.path().filename().string();
            const auto ext = entry.path().extension().string();
            if (name.rfind(prefix, 0) != 0 || name == "qwen_prefix_identity.cpp") continue;
            if (ext != ".cu" && ext != ".cuh" && ext != ".cpp" && ext != ".hpp") continue;
            std::ifstream in(entry.path());
            const std::string source((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            std::size_t literal = 0, all = 0;
            for (auto it = std::sregex_iterator(source.begin(), source.end(), getenv_pattern); it != std::sregex_iterator(); ++it, ++literal)
                check(has_field((*it)[1].str()) || refuse_only.count((*it)[1].str()) != 0 || exempt.count((*it)[1].str()) != 0, "every Qwen CUDA getenv is covered by identity");
            for (auto it = std::sregex_iterator(source.begin(), source.end(), any_getenv); it != std::sregex_iterator(); ++it) ++all;
            check(all == literal, "Qwen CUDA getenv reads use a literal name");
        }
    };
    scan(root + "/src/cuda", "qwen_", {});
    scan(root + "/include/strix", "cuda_qwen_", {});
    // Serve app and VRAM guard: these only schedule sessions, size buffers or decline startup; saved rows and the
    // restored output are the same under any value, so they are exempt from identity (listed, not pattern-matched).
    const std::set<std::string> serve_only = {"STRIX_QRTX_PREFIX_CACHE", "STRIX_QRTX_HTTP_SLOTS",
                                              "STRIX_QRTX_SLOT_MARGIN_MIB", "STRIX_QRTX_COHORT",
                                              "STRIX_QRTX_PREFILL_OVERLAP", "STRIX_QRTX_PREFILL_POLL",
                                              "STRIX_QRTX_PREFILL_AUTO_K", "STRIX_QRTX_ALLOW_SYSMEM",
                                              "STRIX_QRTX_VRAM_HEADROOM_MIB"};
    scan(root + "/apps", "cuda_qwen_", serve_only);
    scan(root + "/include/strix", "cuda_vram_guard", serve_only);
    for (std::size_t i = 0; i < fields.size(); ++i) {
        auto changed = fields;
        changed[i].second += "changed";
        check(px::identity_key(changed) != key, "any field value changes key");
        changed = fields;
        changed[i].first += "changed";
        check(px::identity_key(changed) != key, "any field name changes key");
    }
    check(px::identity_key({{"a", "b:c"}}) != px::identity_key({{"a:b", "c"}}), "length framing");
    c.max_positions++;
    check(px::identity_key(px::identity_fields(c, 256)) != key, "config resolution participates");
    check(px::identity_key(px::identity_fields(c, 512)) != px::identity_key(px::identity_fields(c, 256)), "segment size participates");
    std::cout << "PASS identity: " << fields.size() << " fields, every name/value mutation changes SHA-256, unambiguous framing\n";
}

void store_usage() {
    const auto identity = id("host store test");
    PrefixStateCachePolicy policy;
    policy.host_byte_budget = 4096;
    policy.retention = PrefixCacheRetention::lru;
    PrefixStateCache store(policy, heap);
    auto a = prompt(22);
    std::uint64_t exports = 0;
    const auto capture = [&](const auto& tokens, std::uint64_t p) {
        return px::capture(store, identity, tokens, 4, p, true, 2, 20,
            [&](std::uint64_t first, std::uint64_t count, void* dst) {
                ++exports;
                std::memset(dst, static_cast<int>(tokens[first] & 255), count * 2);
            }, [](void* dst) { std::memset(dst, 7, 20); });
    };
    check(capture(a, 4) && capture(a, 8) && capture(a, 16), "capture checkpoints");
    check(exports == 4, "segments deduplicated across checkpoints");
    auto b = a;
    for (std::size_t i = 10; i < b.size(); ++i) b[i] += 1000;
    auto hit = store.find_longest(identity, b, 4, px::lookup_limit(b.size()), true, {},
        [&](const PrefixCacheCheckpoint& cp) { return px::boundary_matches(cp, b, true); });
    check(hit && hit->position == 8, "dropped reasoning LCP restore");
    hit.reset();
    check(capture(b, 16) && exports == 6, "two turns share first two segments");
    auto branch = a;
    for (std::size_t i = 8; i < branch.size(); ++i) branch[i] += 2000;
    hit = store.find_longest(identity, branch, 4, px::lookup_limit(branch.size()), true, {},
        [&](const PrefixCacheCheckpoint& cp) { return px::boundary_matches(cp, branch, true); });
    check(hit && hit->position == 4, "MTP lookahead guard falls back");
    hit.reset();
    check(capture(branch, 16) && exports == 9, "MTP segment boundary lookahead prevents stale dedupe");
    check(px::segment_key(identity, true, 1) != px::segment_key(identity, true, 2), "lookahead changes segment key");
    check(px::segment_key(identity, false, 1) == px::segment_key(identity, false, 2), "no-MTP has no lookahead");
    check(store.stats().host_segments == 9, "real store tracks unique segments");
    check(!store.find_longest(id("other identity"), a, 4, 20), "strict identity miss");

    policy.host_byte_budget = 60;
    PrefixStateCache bounded(policy, heap);
    const auto put = [&](const auto& tokens) {
        return px::capture(bounded, identity, tokens, 4, 8, false, 2, 20,
            [](auto, auto count, void* dst) { std::memset(dst, 0, count * 2); },
            [](void* dst) { std::memset(dst, 0, 20); });
    };
    check(put(a), "first budget capture");
    auto other = prompt(22, 500);
    check(put(other), "capture evicts LRU");
    check(!bounded.find_longest(identity, a, 4, 20), "evicted prefix misses");
    check(bool(bounded.find_longest(identity, other, 4, 20)), "new prefix retained");
    check(bounded.stats().evicted == 1 && bounded.stats().host_bytes <= 60, "budget bounded");
    policy.host_byte_budget = 1;
    PrefixStateCache tiny(policy, heap);
    check(!px::capture(tiny, identity, a, 4, 8, false, 2, 20, [](auto, auto, void*) {}, [](void*) {}), "overbudget skip");
    PrefixStateCache failed({4096, {}, 0, PrefixCacheRetention::lru}, [](auto) -> PrefixCacheBuffer { throw std::bad_alloc(); });
    bool allocation_failed = false;
    try { (void)px::capture(failed, identity, a, 4, 8, false, 2, 20, [](auto, auto, void*) {}, [](void*) {}); }
    catch (const PrefixCacheAllocationFailure&) { allocation_failed = true; }
    check(allocation_failed && failed.stats().host_bytes == 0, "failed allocation releases reservation");
    std::cout << "PASS store: dropped-reasoning LCP, cross-turn dedupe, MTP boundaries, LRU eviction, budget/allocation failure\n";
}

void slab_pool() {
    std::uint64_t allocations = 0, frees = 0;
    const auto allocator = [&](std::uint64_t bytes) -> PrefixCacheBuffer {
        ++allocations;
        return {std::shared_ptr<std::byte>(new std::byte[bytes], [&](std::byte* p) { ++frees; delete[] p; }), bytes};
    };
    {
        px::SlabPool pool(100, 8, allocator, 4);
        check(pool.usage().reserved == 0, "pool is lazy");
        auto a = pool.allocate(8), b = pool.allocate(8);
        check(allocations == 1 && b.data.get() == a.data.get() + 8, "contiguous slab slots");
        check(pool.usage().live == 16 && pool.usage().reserved == 32, "reservation includes live and idle slots");
        auto* old = a.data.get();
        a = {};
        check(pool.usage().live == 8 && pool.usage().reserved == 32 && frees == 0, "free retains reservation");
        a = pool.allocate(8);
        check(a.data.get() == old && allocations == 1, "free list reuse");
        std::memset(a.data.get(), 17, 8);
        check(std::to_integer<unsigned>(a.data.get()[7]) == 17, "slot writable");
        auto fixed = pool.allocate(20);
        fixed = {};
        fixed = pool.allocate(20);
        check(allocations == 2 && pool.usage().live == 36 && pool.usage().reserved == 52, "fixed class reuse");
    }
    check(frees == allocations, "shutdown releases slabs");
    {
        px::SlabPool pool(40, 8, allocator, 4);
        std::vector<PrefixCacheBuffer> slots;
        for (unsigned i = 0; i < 5; ++i) slots.push_back(pool.allocate(8));
        check(pool.usage().live == 40 && pool.usage().reserved == 40, "exact-size fallback at cap");
        bool failed = false;
        try { (void)pool.allocate(8); } catch (const PrefixCacheAllocationFailure&) { failed = true; }
        check(failed && pool.usage().reserved == 40, "full budget declines allocation");
        slots[0] = {};
        failed = false;
        try { (void)pool.allocate(20); } catch (const PrefixCacheAllocationFailure&) { failed = true; }
        check(failed && pool.usage().live == 32, "partially live slabs cannot be reclaimed");
        slots.clear();
        const auto before = frees;
        auto fixed = pool.allocate(20);
        check(frees == before + 1 && pool.usage().reserved == 28 && pool.usage().live == 20,
              "idle slab reclaimed for other class");
    }
    {
        unsigned attempts = 0;
        std::byte* backing = nullptr;
        px::SlabPool pool(64, 8, [&](std::uint64_t bytes) {
            ++attempts;
            if (bytes > 8) throw PrefixCacheAllocationFailure();
            auto buffer = heap(bytes);
            backing = buffer.data.get();
            return buffer;
        }, 4);
        auto slot = pool.allocate(8);
        check(attempts == 2 && pool.usage().reserved == 8 && slot.data.get() == backing,
              "failed slab pin falls back to slot zero of exact allocation");
        std::memset(slot.data.get(), 9, 8);
    }
    {
        PrefixCacheBuffer lease;
        const auto before = frees;
        {
            px::SlabPool pool(32, 8, allocator, 4);
            lease = pool.allocate(8);
        }
        check(frees == before, "lease outlives pool owner");
        lease = {};
        check(frees == before + 1, "last lease releases orphaned pool");
    }
    {
        px::SlabPool pool(60, 8, allocator, 2);
        PrefixStateCachePolicy policy{60, {}, 0, PrefixCacheRetention::lru};
        PrefixStateCache store(policy, [&](auto bytes) { return pool.allocate(bytes); });
        const auto identity = id("pooled LRU");
        const auto put = [&](auto tokens) {
            return px::capture(store, identity, tokens, 4, 8, false, 2, 20,
                [](const std::vector<strix::cuda::qwen_prefill::RowSegment>&, void*) {});
        };
        const auto before = allocations;
        check(put(prompt(22)), "pooled capture");
        check(put(prompt(22, 500)), "pooled LRU capture");
        check(store.stats().evicted == 1 && allocations == before + 2, "LRU returns row and fixed slots");
        check(pool.usage().live == 36 && pool.usage().reserved == 36, "LRU physical budget");
    }
    {
        px::SlabPool pool(40, 8, allocator, 4);
        PrefixStateCache store({40, {}, 0, PrefixCacheRetention::lru}, [&](auto bytes) { return pool.allocate(bytes); });
        bool failed = false, exported = false;
        try {
            (void)px::capture(store, id("reservation full"), prompt(22), 4, 8, false, 2, 20,
                [&](const std::vector<strix::cuda::qwen_prefill::RowSegment>&, void*) { exported = true; });
        } catch (const PrefixCacheAllocationFailure&) { failed = true; }
        check(failed && !exported && store.stats().host_bytes == 0 && pool.usage().live == 0,
              "slab slack budget decline rolls back capture before export");
        check(pool.usage().reserved == 32, "rolled-back slots remain reserved");
    }
    check(frees == allocations, "all pooled allocations released");
    std::cout << "PASS slab pool: lazy allocation, free-list reuse, reservation cap, exact fallback, skip, class reclaim, LRU, lease lifetime\n";
}

void batched_capture() {
    using strix::cuda::qwen_prefill::RowSegment;
    PrefixStateCache store({4096, {}, 0, PrefixCacheRetention::lru}, heap);
    const auto identity = id("batched export");
    const auto tokens = prompt(22);
    unsigned calls = 0;
    std::vector<std::uint64_t> firsts;
    px::CaptureTiming timing;
    const auto put = [&](std::uint64_t position) {
        return px::capture(store, identity, tokens, 4, position, false, 2, 20,
            [&](const std::vector<RowSegment>& segments, void* fixed) {
                ++calls;
                firsts.clear();
                for (const auto& segment : segments) {
                    check(segment.count == 4 && segment.data, "batch segment destination");
                    firsts.push_back(segment.first);
                    std::memset(const_cast<void*>(segment.data), static_cast<int>(segment.first), 8);
                }
                check(fixed != nullptr, "batch includes fixed destination");
                std::memset(fixed, 42, 20);
            }, &timing);
    };
    check(put(8) && calls == 1 && firsts == std::vector<std::uint64_t>({0, 4}), "one export with all new rows in order");
    check(put(16) && calls == 2 && firsts == std::vector<std::uint64_t>({8, 12}), "one export excludes reused rows");
    check(!put(16) && calls == 2, "duplicate checkpoint does not export");
    const auto cp = store.find_longest(identity, tokens, 4, 16);
    check(cp && cp->segments.size() == 4 && cp->fixed.data.get()[19] == std::byte{42}, "batch fixed bytes retained");
    for (unsigned i = 0; i < 4; ++i)
        check(cp->segments[i]->payload.data.get()[7] == static_cast<std::byte>(i * 4), "batch row bytes retained");
    check(timing.ms > 0 && timing.alloc_ms > 0 && timing.copy_ms > 0 && timing.ms >= timing.alloc_ms + timing.copy_ms,
          "capture timing separates allocation and copy");
    px::CaptureTiming failed;
    bool threw = false;
    try {
        (void)px::capture(store, identity, prompt(22, 1000), 4, 8, false, 2, 20,
            [](const std::vector<RowSegment>&, void*) { throw std::runtime_error("copy failed"); }, &failed);
    } catch (const std::runtime_error&) { threw = true; }
    check(threw && failed.ms > 0 && failed.alloc_ms > 0 && failed.copy_ms > 0 && store.stats().inserted == 2,
          "failed export timed and never published");
    std::cout << "PASS batched capture: one ordered export, dedupe, payload bytes, duplicate skip, allocation/copy timing, failed export rollback\n";
}
}  // namespace

int main() {
    try { planner(); identity(); store_usage(); slab_pool(); batched_capture(); std::cout << "PASS qwen_prefix_cache_host_test\n"; return 0; }
    catch (const std::exception& ex) { std::cerr << "FAIL " << ex.what() << '\n'; return 1; }
}
