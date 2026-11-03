#include "strix/prefix_state_cache.hpp"
#include "strix/model_content_hash.hpp"
#include "../apps/prefix_state_cache_identity.hpp"
#include "strix/arithmetic_lineage.hpp"
#include "strix/unbuffered_file.hpp"
#include "state_file_fixtures.inc"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <atomic>
#include <array>
#include <cstdlib>
#include <new>
#include <iterator>
#include <optional>
#include <set>

// Test-only host pressure, independent of HIP allocation fault injection.
thread_local bool prefix_test_fail_host_allocations=false;
thread_local int prefix_test_fail_after=-1;
void* operator new(std::size_t bytes) {
    if(prefix_test_fail_after>=0 && prefix_test_fail_after--==0) throw std::bad_alloc();
    if(prefix_test_fail_host_allocations) throw std::bad_alloc();
    if(auto* p=std::malloc(bytes?bytes:1)) return p;
    throw std::bad_alloc();
}
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
void prefix_test_free(void* p) noexcept {std::free(p);}
void* operator new[](std::size_t bytes) {return ::operator new(bytes);}
void operator delete(void* p) noexcept {prefix_test_free(p);}
void operator delete(void* p,std::size_t) noexcept {::operator delete(p);}
void operator delete[](void* p) noexcept {::operator delete(p);}
void operator delete[](void* p,std::size_t) noexcept {::operator delete(p);}


namespace strix {
struct PrefixStateCacheTestAccess {
    static void decline_while_locked(PrefixStateCache& cache) {
        std::lock_guard lock(cache.mutex_);
        // A recursive store lock hangs this regression; heap allocation throws.
        prefix_test_fail_host_allocations=true;
        cache.record_certificate_decline(PrefixCertificateDecline::prefix_no_compatible_checkpoint);
        cache.record_certificate_decline(PrefixCertificateDecline::batch_suffix_not_solo);
        prefix_test_fail_host_allocations=false;
    }
    static std::string publication(PrefixStateCache& cache) {
        std::string result=cache.group_prefix_order_valid_?"valid":"invalid";
        for(const auto& [key,activity]:cache.group_activity_) result+="|"+key+":"+std::to_string(activity.members);
        for(const auto* activity:cache.group_prefix_order_) result+="/"+std::string(activity->key)+":"+std::to_string(activity->prefix_lcp);
        for(const auto& [digest,entry]:cache.entries_) {
            result+=";"+model_content_identity_hex(digest);
            for(const auto& key:entry.conversations) result+=","+key;
        }
        for(const auto& digest:cache.lru_) result+="~"+model_content_identity_hex(digest);
        for(const auto& [digest,segment]:cache.segments_) {
            result+="^"+model_content_identity_hex(digest)+":"+std::to_string(reinterpret_cast<std::uintptr_t>(segment.lock().get()));
            // Include weak ownership even for expired nodes.
            result+=segment.expired()?":expired":":live";
        }
        return result;
    }
    static void add(PrefixStateCache& cache, std::shared_ptr<const PrefixCacheCheckpoint> checkpoint) {cache.add_locked(std::move(checkpoint));}
    static bool association_state(PrefixStateCache& cache, std::size_t members, bool order) {
        std::size_t total=0;
        for (const auto& [key, activity] : cache.group_activity_) {
            if (!activity.members) return false;
            total+=activity.members;
        }
        return total==members && cache.group_prefix_order_.size()==(order?cache.group_activity_.size():0);
    }
    static bool reused(PrefixStateCache& cache, const std::string& key) {return cache.group_activity_.at(key).reused;}
    static auto clock(PrefixStateCache& cache) {return cache.epoch_;}
    static auto reuse(PrefixStateCache& cache, const ModelContentIdentity& digest) {return cache.entries_.at(digest).reuse;}
    static void retire(PrefixStateCache& cache, const ModelContentIdentity& digest) {cache.erase_locked(digest);}
    static void move_clock(PrefixStateCache& cache, std::uint64_t now) {
        const auto delta = now - cache.epoch_;
        for (auto& [digest, entry] : cache.entries_) entry.epoch += delta;
        for (auto& [key, activity] : cache.group_activity_) activity.born_epoch += delta;
        cache.epoch_ = now;
    }
    static std::set<std::string> successors(PrefixStateCache& cache) {
        cache.mark_private_prefix_successors_locked(nullptr, cache.epoch_ + 1);
        std::set<std::string> result;
        for (const auto& [key, activity] : cache.group_activity_) if (activity.private_successor) result.insert(key);
        return result;
    }
};
}

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

strix::PrefixCacheBuffer heap(std::uint64_t bytes) {
    std::shared_ptr<std::byte> data(new std::byte[static_cast<std::size_t>(bytes ? bytes : 1)],
        std::default_delete<std::byte[]>());
    return {std::move(data), bytes};
}

strix::ModelContentIdentity identity_of(const char* text) {
    return strix::sha256_bytes(std::as_bytes(std::span<const char>(text, std::strlen(text))));
}

// Frozen v1 key encoding: the old domain and fields, without v2 option-set fields.
strix::ModelContentIdentity legacy_v1_identity(const strix::decode_cli::PrefixStateCacheIdentityInputs& in) {
    std::vector<std::byte> frame;
    const auto u64 = [&](std::uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8)
            frame.push_back(static_cast<std::byte>((value >> shift) & 255U));
    };
    const auto digest = [&](const strix::ModelContentIdentity& value) {
        for (const auto byte : value.bytes) frame.push_back(static_cast<std::byte>(byte));
    };
    u64(0x5354524958504331ULL);
    digest(in.model_content);
    digest(in.engine_build);
    for (const auto value : {in.state_layout_identity, in.runtime_compatibility_identity, in.decode_execution_identity,
             in.decode_kernel_plan_identity, in.prefill_execution_identity, in.prefill_kernel_plan_identity,
             in.tiled_execution_identity, in.tiled_kernel_plan_identity, in.prefill_profile_identity,
             in.decode_profile_identity, in.profile_pair_identity, static_cast<std::uint64_t>(in.block_rows),
             static_cast<std::uint64_t>(in.tail_rows.size()), in.context_tokens})
        u64(value);
    for (const auto rows : in.tail_rows) u64(rows);
    return strix::sha256_bytes(frame);
}

constexpr std::uint32_t T = 4;
constexpr std::uint64_t segment_bytes = 64, fixed_bytes = 256;

// Payload bytes are a function of (block or position, salt) so restores can be checked.
std::byte pattern(std::uint64_t a, std::uint64_t i) { return static_cast<std::byte>((a * 131U + i * 7U) & 255U); }

// Builds the checkpoints of `tokens` at every block boundary, sharing segments like the engine does.
std::vector<std::shared_ptr<const strix::PrefixCacheCheckpoint>> build(strix::PrefixStateCache& cache,
    const strix::ModelContentIdentity& identity, const std::vector<std::uint32_t>& tokens) {
    std::vector<std::shared_ptr<const strix::PrefixCacheCheckpoint>> out;
    std::vector<std::shared_ptr<const strix::PrefixCacheSegment>> segments;
    auto previous = identity;
    for (std::uint64_t k = 0; k < tokens.size() / T; ++k) {
        const std::span<const std::uint32_t> block(tokens.data() + k * T, T);
        auto segment = std::make_shared<strix::PrefixCacheSegment>();
        segment->identity = identity;
        segment->parent = previous;
        segment->digest = strix::prefix_cache_block_digest(previous, block);
        segment->block_index = k;
        segment->block_rows = T;
        segment->tokens.assign(block.begin(), block.end());
        segment->payload = cache.allocate(segment_bytes);
        for (std::uint64_t i = 0; i < segment_bytes; ++i) segment->payload.data.get()[i] = pattern(k + tokens[k * T], i);
        previous = segment->digest;
        segments.push_back(segment);
        auto checkpoint = std::make_shared<strix::PrefixCacheCheckpoint>();
        checkpoint->identity = identity;
        checkpoint->digest = previous;
        checkpoint->block_rows = T;
        checkpoint->position = (k + 1) * T;
        checkpoint->segments = segments;
        checkpoint->fixed = cache.allocate(fixed_bytes);
        for (std::uint64_t i = 0; i < fixed_bytes; ++i) checkpoint->fixed.data.get()[i] = pattern(1000 + checkpoint->position, i);
        checkpoint->metadata.assign(3, static_cast<std::byte>(k));
        out.push_back(checkpoint);
    }
    return out;
}

bool same_payloads(const strix::PrefixCacheCheckpoint& a, const strix::PrefixCacheCheckpoint& b) {
    if (a.position != b.position || a.segments.size() != b.segments.size() || a.fixed.bytes != b.fixed.bytes ||
        a.metadata != b.metadata || a.producer_certificate != b.producer_certificate ||
        std::memcmp(a.fixed.data.get(), b.fixed.data.get(), a.fixed.bytes) != 0)
        return false;
    for (std::size_t k = 0; k < a.segments.size(); ++k)
        if (a.segments[k]->tokens != b.segments[k]->tokens || a.segments[k]->payload.bytes != b.segments[k]->payload.bytes ||
            std::memcmp(a.segments[k]->payload.data.get(), b.segments[k]->payload.data.get(), a.segments[k]->payload.bytes) != 0)
            return false;
    return true;
}
}  // namespace

namespace {
void prefix_set_env(const char* key, const char* value) {
#ifdef _WIN32
    require(_putenv_s(key, value ? value : "") == 0, "environment update");
#else
    require((value ? setenv(key, value, 1) : unsetenv(key)) == 0, "environment update");
#endif
}
std::vector<char> prefix_file_bytes(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
}  // namespace

// Unbuffered segment/checkpoint files: byte-identical to the buffered writer, restorable by
// either reader (old files keep working), multi-chunk payloads with unaligned tails.
void prefix_unbuffered_io_tests(const std::filesystem::path& root) {
    const char* previous = std::getenv("STRIX_SESSION_DIRECT_IO");
    const std::optional<std::string> saved = previous ? std::optional<std::string>{previous} : std::nullopt;
    const auto id = identity_of("unbuffered identity");
    std::vector<std::uint32_t> tokens;
    for (std::uint32_t i = 0; i < 2 * T; ++i) tokens.push_back(900 + i);
    const std::uint64_t big_segment = (9ULL << 20) + 77, big_fixed = (1ULL << 20) + 5;
    const auto make = [&](strix::PrefixStateCache& cache) {
        std::vector<std::shared_ptr<const strix::PrefixCacheSegment>> segments;
        auto parent = id;
        std::shared_ptr<strix::PrefixCacheCheckpoint> checkpoint;
        for (std::uint64_t k = 0; k < 2; ++k) {
            auto segment = std::make_shared<strix::PrefixCacheSegment>();
            segment->identity = id; segment->parent = parent; segment->block_index = k; segment->block_rows = T;
            segment->tokens.assign(tokens.begin() + static_cast<std::ptrdiff_t>(k * T), tokens.begin() + static_cast<std::ptrdiff_t>((k + 1) * T));
            segment->digest = strix::prefix_cache_block_digest(parent, segment->tokens);
            segment->payload = cache.allocate(big_segment + k);
            for (std::uint64_t i = 0; i < segment->payload.bytes; ++i) segment->payload.data.get()[i] = pattern(k + 77, i * 3 + (i >> 9));
            parent = segment->digest; segments.push_back(segment);
        }
        checkpoint = std::make_shared<strix::PrefixCacheCheckpoint>();
        checkpoint->identity = id; checkpoint->digest = parent; checkpoint->block_rows = T; checkpoint->position = 2 * T;
        checkpoint->segments = segments;
        checkpoint->fixed = cache.allocate(big_fixed);
        for (std::uint64_t i = 0; i < big_fixed; ++i) checkpoint->fixed.data.get()[i] = pattern(5, i + (i >> 11));
        checkpoint->metadata.assign(11, std::byte{0x5a});
        return checkpoint;
    };
    for (const char* writer : {"0", "1"}) {
        prefix_set_env("STRIX_SESSION_DIRECT_IO", writer);
        strix::PrefixStateCache cache({1ULL << 30, root / writer, 0}, heap);
        require(cache.insert(make(cache)), "unbuffered seed insert");
        cache.flush();
        require(cache.counters().disk_writes == 1 && cache.counters().disk_write_failures == 0, "unbuffered seed write");
    }
    std::size_t files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root / "0")) {
        const auto twin = root / "1" / entry.path().filename();
        require(std::filesystem::exists(twin) && prefix_file_bytes(entry.path()) == prefix_file_bytes(twin),
            "unbuffered prefix file differs from the buffered writer");
        ++files;
    }
    require(files == 3, "unbuffered prefix file set");
    strix::PrefixStateCache reference({1ULL << 30, {}, 0}, heap);
    const auto expected = make(reference);
    for (const char* reader : {"0", "1"}) for (const char* writer : {"0", "1"}) {
        prefix_set_env("STRIX_SESSION_DIRECT_IO", reader);
        strix::PrefixStateCache cache({1ULL << 30, root / writer, 0}, heap);
        const auto hit = cache.find_longest(id, tokens, T, tokens.size());
        require(hit && hit->position == 2 * T && same_payloads(*hit, *expected), "unbuffered prefix restore differs");
        require(cache.counters().disk_hits == 1 && cache.counters().disk_read_failures == 0, "unbuffered prefix disk hit");
    }
    prefix_set_env("STRIX_SESSION_DIRECT_IO", saved ? saved->c_str() : nullptr);
    std::filesystem::remove_all(root);
}

// Prefix files written before unbuffered I/O restore in both read modes, and today's writer
// (either mode) produces exactly those files.
void old_format_prefix_tests(const std::filesystem::path& root) {
    const char* previous = std::getenv("STRIX_SESSION_DIRECT_IO");
    const std::optional<std::string> saved = previous ? std::optional<std::string>{previous} : std::nullopt;
    const auto id = identity_of("fixture prefix identity");
    const std::uint32_t rows = 4;
    const std::vector<std::uint32_t> toks{101, 102, 103, 104, 201, 202, 203, 204};
    const auto make = [&](strix::PrefixStateCache& cache) {
        auto checkpoint = std::make_shared<strix::PrefixCacheCheckpoint>();
        checkpoint->identity = id; checkpoint->block_rows = rows; checkpoint->position = 8;
        auto parent = id;
        for (std::uint64_t k = 0; k < 2; ++k) {
            auto segment = std::make_shared<strix::PrefixCacheSegment>();
            segment->identity = id; segment->parent = parent; segment->block_index = k; segment->block_rows = rows;
            segment->tokens.assign(toks.begin() + static_cast<std::ptrdiff_t>(k * rows), toks.begin() + static_cast<std::ptrdiff_t>((k + 1) * rows));
            segment->digest = strix::prefix_cache_block_digest(parent, segment->tokens);
            segment->payload = cache.allocate(700 + k);
            for (std::uint64_t i = 0; i < segment->payload.bytes; ++i) segment->payload.data.get()[i] = static_cast<std::byte>((i * 5 + k * 3) & 255);
            parent = segment->digest; checkpoint->segments.push_back(segment);
        }
        checkpoint->digest = parent;
        checkpoint->fixed = cache.allocate(259);
        for (std::uint64_t i = 0; i < 259; ++i) checkpoint->fixed.data.get()[i] = static_cast<std::byte>((i * 3 + 9) & 255);
        checkpoint->metadata.assign(6, std::byte{0x42});
        return checkpoint;
    };
    const std::span<const unsigned char> files[] = {old_format_prefix_file0, old_format_prefix_file1, old_format_prefix_file2};
    std::filesystem::create_directories(root / "old");
    for (std::size_t i = 0; i < 3; ++i) {
        std::ofstream out(root / "old" / old_format_prefix_names[i], std::ios::binary);
        out.write(reinterpret_cast<const char*>(files[i].data()), static_cast<std::streamsize>(files[i].size()));
    }
    strix::PrefixStateCache reference({1ULL << 20, {}, 0}, heap);
    const auto expected = make(reference);
    for (const char* mode : {"0", "1"}) {
        prefix_set_env("STRIX_SESSION_DIRECT_IO", mode);
        {
            strix::PrefixStateCache cache({1ULL << 20, root / "old", 0}, heap);
            const auto hit = cache.find_longest(id, toks, rows, toks.size(), false);
            require(hit && hit->position == 8 && same_payloads(*hit, *expected), "old-format prefix restore differs");
        }
        const auto fresh = root / (std::string("new") + mode);
        {
            strix::PrefixStateCache cache({1ULL << 20, fresh, 0}, heap);
            require(cache.insert(make(cache)), "old-format rewrite insert");
            cache.flush();
        }
        for (std::size_t i = 0; i < 3; ++i) {
            const auto bytes = prefix_file_bytes(fresh / old_format_prefix_names[i]);
            require(bytes.size() == files[i].size() && !std::memcmp(bytes.data(), files[i].data(), bytes.size()),
                "prefix writer no longer produces the old-format files");
        }
    }
    prefix_set_env("STRIX_SESSION_DIRECT_IO", saved ? saved->c_str() : nullptr);
    std::filesystem::remove_all(root);
}

void prefix_dma_tests();

int main(int argc, char** argv) {
    if (argc == 3) {
        try {
            strix::decode_cli::PrefixStateCacheIdentityInputs in;
            in.model_content = identity_of("same model");
            in.effective_execution = strix::decode_cli::prefix_effective_execution_signature("test",false,false,{},0U);
            in.state_layout_identity = in.runtime_compatibility_identity = 1U;
            in.decode_execution_identity = in.prefill_execution_identity = in.tiled_execution_identity = 2U;
            in.block_rows = T;
            const auto legacy = legacy_v1_identity(in);
            const auto off = strix::decode_cli::prefix_state_cache_identity(in);
            in.effective_execution = strix::decode_cli::prefix_effective_execution_signature("test",false,false,{},strix::kPrefillOptionsReady);
            const auto on = strix::decode_cli::prefix_state_cache_identity(in);
            require(on != off, "production ON/OFF cache namespaces must differ");
            require(legacy != off && legacy != on, "v2 keys must differ from v1 even with options OFF");
            const auto execution_on = strix::bind_prefill_option_identity(42U, strix::kPrefillOptionsReady);
            const auto execution_off = strix::bind_prefill_option_identity(42U, 0U);
            require(execution_on != execution_off, "compiled arithmetic identities must differ");
            const auto history_on = strix::prepare_prefill_chunk_lineage({}, 0U, T, execution_on, 3U, execution_on);
            const auto history_off = strix::prepare_prefill_chunk_lineage({}, 0U, T, execution_off, 3U, execution_off);
            require(history_on.history_identity != history_off.history_identity, "lineage must inherit option identity");
            const std::vector<std::uint32_t> tokens{1, 2, 3, 4};
            const std::vector<std::uint32_t> legacy_tokens{5, 6, 7, 8};
            strix::PrefixStateCache cache({1ULL << 20, argv[2], 0}, heap);
            if (std::string(argv[1]) == "--write-option-checkpoint") {
                require(cache.insert(build(cache, legacy, legacy_tokens).front()), "write legacy v1 checkpoint");
                require(cache.insert(build(cache, on, tokens).front()), "write ON checkpoint");
                cache.flush();
                require(cache.stats().disk_writes == 2U, "v1 and v2 checkpoints persisted together");
            } else {
                require(cache.find_longest(off, legacy_tokens, T, T) == nullptr &&
                        cache.find_longest(on, legacy_tokens, T, T) == nullptr,
                    "legacy-only prefix is a cold miss under both v2 option sets");
                require(cache.stats().disk_read_failures == 0U, "legacy namespace miss is not payload corruption");
                strix::PrefixStateCache reference({1ULL << 20, {}, 0}, heap);
                const auto legacy_hit = cache.find_longest(legacy, legacy_tokens, T, T);
                require(legacy_hit && same_payloads(*legacy_hit, *build(reference, legacy, legacy_tokens).front()),
                    "legacy checkpoint remains valid and readable under its v1 key");
                require(cache.find_longest(off, tokens, T, T) == nullptr, "fresh OFF process rejects ON checkpoint");
                const auto on_hit = cache.find_longest(on, tokens, T, T);
                require(on_hit && same_payloads(*on_hit, *build(reference, on, tokens).front()),
                    "fresh ON process restores compatible v2 payload alongside v1 entry");
                // Also place the ON payload at the OFF filename: validation must reject it.
                const auto on_digest = strix::prefix_cache_block_digest(on, tokens);
                const auto off_digest = strix::prefix_cache_block_digest(off, tokens);
                const std::filesystem::path directory(argv[2]);
                std::filesystem::copy_file(directory / ("ckpt-" + strix::model_content_identity_hex(on_digest) + ".tmpc"),
                    directory / ("ckpt-" + strix::model_content_identity_hex(off_digest) + ".tmpc"),
                    std::filesystem::copy_options::overwrite_existing);
                require(cache.find_longest(off, tokens, T, T) == nullptr && cache.stats().disk_read_failures > 0U,
                    "incompatible checkpoint payload rejected even at expected filename");
            }
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    const auto dir = std::filesystem::temp_directory_path() /
        ("strix-prefix-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        {
            constexpr const char* key = "STRIX_QPF2K_MOE_ROUTE_BF16";
            const char* value = std::getenv(key);
            const auto saved = value ? std::optional<std::string>{value} : std::nullopt;
            const auto put = [&](const char* setting) {
#ifdef _WIN32
                require(_putenv_s(key, setting ? setting : "") == 0, "set cache signature fixture");
#else
                require((setting ? ::setenv(key, setting, 1) : ::unsetenv(key)) == 0, "set cache signature fixture");
#endif
            };
            std::set<std::string> namespaces;
            for (const char* storage : {"0", "1"}) {
                put(storage);
                for (const unsigned mask : {0U, strix::kPrefillOptionsReady}) {
                    const auto signature = strix::decode_cli::prefix_effective_execution_signature(
                        "same model and geometry", false, false, {}, mask);
                    require(namespaces.insert(strix::model_content_identity_hex(signature)).second,
                        "prefix signature composes governed HC and BF16 route storage independently");
                }
            }
            put(saved ? saved->c_str() : nullptr);
        }
        const auto id = identity_of("identity A"), other = identity_of("identity B");
        std::vector<std::uint32_t> tokens;
        for (std::uint32_t i = 0; i < 18; ++i) tokens.push_back(100 + i);  // 4 full blocks + 2

        // Digests chain and stop at full blocks.
        const auto digests = strix::prefix_cache_block_digests(id, tokens, T, 100);
        require(digests.size() == 4, "digest count");
        require(digests[1] == strix::prefix_cache_block_digest(digests[0], std::span(tokens).subspan(4, 4)), "digest chain");
        require(strix::prefix_cache_block_digests(other, tokens, T, 100)[0] != digests[0], "identity changes digests");
        prefix_unbuffered_io_tests(dir / "unbuffered");
        old_format_prefix_tests(dir / "old-format");

        {
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            const auto checkpoints = build(cache, id, tokens);
            require(cache.find_longest(id, tokens, T, 17) == nullptr, "empty cache misses");
            for (const auto& checkpoint : checkpoints) require(cache.insert(checkpoint), "insert");
            // Shared segments are counted once: 4 segments + 4 fixed blobs + metadata.
            require(cache.stats().host_bytes == 4 * segment_bytes + 4 * (fixed_bytes + 3), "shared segment accounting");
            auto hit = cache.find_longest(id, tokens, T, 17);
            require(hit && hit->position == 16 && same_payloads(*hit, *checkpoints[3]), "longest hit");
            hit = cache.find_longest(id, tokens, T, 15);
            require(hit && hit->position == 12, "limit bounds the hit");
            // A different token in block 3 hits block 2 only.
            auto changed = tokens;
            changed[9] = 7;
            hit = cache.find_longest(id, changed, T, 17);
            require(hit && hit->position == 8 && same_payloads(*hit, *checkpoints[1]), "diverging tokens hit the common prefix");
            // A different identity never hits.
            require(cache.find_longest(other, tokens, T, 17) == nullptr, "identity mismatch misses");
            // A different block size never hits (different digests and positions).
            require(cache.find_longest(id, tokens, 8, 17) == nullptr, "block size mismatch misses");
            cache.flush();
            require(cache.stats().disk_write_failures == 0 && cache.stats().disk_writes == 4, "disk writes");
        }
        {
            // A fresh store finds the checkpoints on disk, byte-identical, and loads them into the host tier.
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            strix::PrefixStateCache reference({1ULL << 20, {}, 0}, heap);
            const auto expected = build(reference, id, tokens);
            auto hit = cache.find_longest(id, tokens, T, 17);
            require(hit && hit->position == 16 && same_payloads(*hit, *expected[3]), "disk hit restores payloads");
            require(cache.stats().disk_hits == 1 && cache.stats().host_checkpoints == 1, "disk hit is retained in host tier");
            hit = cache.find_longest(id, tokens, T, 17);
            require(cache.stats().host_hits == 1, "second lookup is a host hit");
            require(cache.find_longest(other, tokens, T, 17) == nullptr, "disk identity mismatch misses");
        }
        {
            strix::PrefixStateCache cache({1ULL<<20,dir,0},heap);
            auto hit=cache.find_longest(id,tokens,T,17,false);
            require(hit && hit->position==16,"no-store disk lookup allowed");
            const auto after=cache.counters();
            require(after.host_bytes==fixed_bytes+4*segment_bytes && after.host_segments==4 && after.host_checkpoints==0 &&
                after.inserted==0 && after.disk_writes==0,"no-store disk hit does not promote");
            require(after.lookup_tokens==17 && after.hit_tokens==16 && after.restored_tokens==16,"token counters");
            cache.record_no_store_skip();
            require(cache.counters().no_store_skips==1 && cache.counters().captures==0,"suppressed capture counters");
        }
        {
            // A corrupted fixed blob on disk is rejected (counted) and the next shorter prefix is used.
            const auto digests16 = strix::prefix_cache_block_digests(id, tokens, T, 4);
            const auto path = dir / ("ckpt-" + strix::model_content_identity_hex(digests16[3]) + ".tmpc");
            {
                std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
                file.seekp(-1, std::ios::end);
                file.put('x');
            }
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            const auto hit = cache.find_longest(id, tokens, T, 17);
            require(hit && hit->position == 12 && cache.stats().disk_read_failures == 1, "corrupt checkpoint is skipped");
        }
        {
            // Startup sweep: old temporaries and unreferenced segments go, referenced segments stay.
            const auto stale = std::filesystem::file_time_type::clock::now() - std::chrono::hours(1);
            const auto orphan = dir / ("seg-" + std::string(64, 'a') + ".tmps");
            const auto partial = dir / "ckpt-partial.tmpc.tmp.1";
            { std::ofstream(orphan) << "x"; std::ofstream(partial) << "x"; }
            std::filesystem::last_write_time(orphan, stale);
            std::filesystem::last_write_time(partial, stale);
            std::size_t before = 0;
            for (const auto& entry : std::filesystem::directory_iterator(dir)) before += entry.path().extension() == ".tmps";
            strix::PrefixStateCache cache({1ULL << 20, dir, 0}, heap);
            std::size_t after = 0;
            for (const auto& entry : std::filesystem::directory_iterator(dir)) after += entry.path().extension() == ".tmps";
            require(!std::filesystem::exists(orphan) && !std::filesystem::exists(partial) && after + 1 == before,
                "sweep removes only stale orphans");
            require(cache.find_longest(id, tokens, T, 13) != nullptr, "referenced segments survive the sweep");
        }
        {
            // LRU within the byte budget: room for two checkpoints of one prompt (shared segments).
            const std::uint64_t budget = 3 * segment_bytes + 2 * (fixed_bytes + 3);
            strix::PrefixStateCache cache({budget, {}, 0, strix::PrefixCacheRetention::lru}, heap);
            auto checkpoints = build(cache, id, tokens);
            require(cache.insert(checkpoints[0]) && cache.insert(checkpoints[1]), "insert within budget");
            checkpoints[0].reset(); // Release the fixed-state lease before expecting reclaimed capacity.
            require(cache.find_longest(id, tokens, T, 9) != nullptr, "touch block 2");
            require(cache.insert(checkpoints[2]), "insert evicts");
            const auto stats = cache.stats();
            require(stats.evicted == 1 && stats.host_bytes <= budget, "evicted least recently used");
            // Block 1 was evicted, block 2 and 3 are retained.
            require(cache.find_longest(id, tokens, T, 4) == nullptr, "evicted entry misses");
            require(cache.find_longest(id, tokens, T, 12)->position == 12, "newest entry hits");
            // A checkpoint larger than the whole budget is rejected.
            strix::PrefixStateCache tiny({fixed_bytes, {}, 0}, heap);
            require(!tiny.insert(checkpoints[3]) && tiny.stats().rejected_over_budget == 1, "over budget rejected");
        }
        {
            // A lone conversation borrows; the first competing conversation reclaims that borrowing.
            strix::PrefixStateCache cache({6*(segment_bytes+fixed_bytes+3),{},0},heap);
            std::vector<strix::ModelContentIdentity> noisy;
            for (std::uint32_t n=0;n<6;++n) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{1,2,3,n+10})[0]);
                c->conversation="noisy"; noisy.push_back(c->digest);
                require(cache.insert(c),"lone conversation borrows");
            }
            require(cache.counters().host_checkpoints==6 && cache.counters().evicted==0,"lone conversation fills budget without eviction");
            auto quiet=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{8,9,10,11})[0]);
            quiet->conversation="quiet";
            require(cache.insert(quiet),"new conversation admitted");
            require(cache.contains(quiet->digest) && cache.counters().evictions_conversation_cap==1 && !cache.contains(noisy.front()),"borrowed bytes reclaimed first");
            require(cache.counters().host_bytes==cache.stats().host_bytes,"cheap gauges reconcile without leases");
        }
        for (bool fair : {false,true}) for (bool legacy : {false,true}) for (bool replacement : {false,true}) {
            // Sweep every allocation, including partial multi-segment staging.
            // Repeated failures must preserve both weak-index values and gauges.
            unsigned failures=0;
            for (int fail=0; fail<64; ++fail) {
                strix::PrefixStateCachePolicy policy{1ULL<<20,{},0};
                policy.fair_reclaim_idle_groups=fair;policy.legacy=legacy;
                strix::PrefixStateCache cache(policy,heap);
                auto checkpoints=build(cache,id,{1,2,3,4,5,6,7,8});
                if(replacement) require(cache.insert(checkpoints.front()),"segment replacement baseline");
                auto candidate=std::make_shared<strix::PrefixCacheCheckpoint>(*checkpoints.back());
                candidate->conversation="candidate";
                auto segment=std::make_shared<strix::PrefixCacheSegment>(*candidate->segments.front());
                candidate->segments.front()=segment;
                const auto before=cache.find_segment(segment->digest);
                const auto publication=strix::PrefixStateCacheTestAccess::publication(cache);
                const auto gauges=cache.counters();
                const auto epoch=strix::PrefixStateCacheTestAccess::clock(cache);
                bool failed=false;
                for(unsigned repeat=0;repeat<3;++repeat) {
                    prefix_test_fail_after=fail;
                    try {strix::PrefixStateCacheTestAccess::add(cache,candidate);}
                    catch(const std::bad_alloc&) {failed=true;}
                    prefix_test_fail_after=-1;
                    if(!failed) break;
                    require(strix::PrefixStateCacheTestAccess::publication(cache)==publication,"failed add restores complete segment index and publication");
                    require(cache.find_segment(segment->digest)==before && !cache.find_segment(candidate->segments.back()->digest),"failed add preserves segment lookups");
                    const auto after=cache.counters();
                    require(after.host_bytes==gauges.host_bytes && after.host_segments==gauges.host_segments && after.host_checkpoints==gauges.host_checkpoints,
                        "failed add preserves ordinary and legacy gauges");
                    require(strix::PrefixStateCacheTestAccess::clock(cache)==epoch,"failed add preserves epoch");
                }
                if(!failed) break;
                ++failures;
                // Release the rejected objects before checking for expired residue.
                const auto digest=candidate->segments.back()->digest;
                candidate.reset();segment.reset();checkpoints.clear();
                require(strix::PrefixStateCacheTestAccess::publication(cache)==publication && !cache.find_segment(digest),"repeated failed adds leave no expired weak nodes");
            }
            require(failures>=4,"segment staging allocation failures exercised");
        }
        for (bool fair : {false,true}) for (unsigned path=0; path<3; ++path) {
            // Fail each allocation in duplicate capture, host-hit alias and new
            // entry publication. Retry and force eviction before the next insert.
            unsigned failures=0;
            for (int fail=0; fail<64; ++fail) {
                strix::PrefixStateCachePolicy policy{1ULL<<20,{},0};policy.fair_reclaim_idle_groups=fair;
                strix::PrefixStateCache cache(policy,heap);
                const std::vector<std::uint32_t> prompt{7,7,20,20};
                auto base=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,prompt).front());
                base->conversation="base";base->producer_certificate={17,29};
                if(path!=2) require(cache.insert(base),"allocation baseline");
                auto alias=std::make_shared<strix::PrefixCacheCheckpoint>(*base);alias->conversation="alias";
                const auto publication=strix::PrefixStateCacheTestAccess::publication(cache);
                bool failed=false;
                prefix_test_fail_after=fail;
                try {
                    if(path==1) (void)cache.find_longest(id,prompt,T,T,true,"alias");
                    else (void)cache.insert(alias);
                } catch(const std::bad_alloc&) {failed=true;}
                prefix_test_fail_after=-1;
                if(failed) {
                    ++failures;
                    require(strix::PrefixStateCacheTestAccess::publication(cache)==publication,"failed publication exactly restores membership, links and overlaps");
                    require(strix::PrefixStateCacheTestAccess::association_state(cache,path==2?0:1,fair),
                        "failed publication preserves membership and lexical order");
                    require(cache.counters().host_checkpoints==(path==2?0:1),"failed new entry rolls back directory");
                }
                if(path==1) require(bool(cache.find_longest(id,prompt,T,T,true,"alias")),"host alias retry");
                else require(cache.insert(alias),"capture publication retry");
                require(strix::PrefixStateCacheTestAccess::association_state(cache,path==2?1:2,fair),"retry creates exactly one lexical link");
                alias.reset();base.reset();
                {auto pressure=cache.reserve(1ULL<<20);require(bool(pressure),"allocation retry pressure");}
                require(strix::PrefixStateCacheTestAccess::association_state(cache,0,fair),"pressure removes every lexical link");
                require(cache.insert(build(cache,id,{7,7,20,21}).front()),"next insertion after failed publication");
                if(!failed) break;
            }
            require(failures>0 && failures<64,"all publication allocation sites traversed");
        }
        for(bool fair:{false,true}) for(bool promote:{false,true}) for(bool named:{false,true}) {
            // An accepted certified disk hit protects resident siblings, including
            // no-store reads and retirement of a promoted hit checkpoint.
            const auto diskdir=dir/("reuse-"+std::to_string(fair)+std::to_string(promote)+std::to_string(named));
            const std::vector<std::uint32_t> prompt{7,7,20,20,40,41,42,43};
            {
                strix::PrefixStateCache writer({1ULL<<20,diskdir,0},heap);
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*build(writer,id,prompt).back());
                c->producer_certificate={17,29};require(writer.insert(c),"reuse disk seed");writer.flush();
            }
            strix::PrefixStateCachePolicy policy{1ULL<<20,diskdir,0};policy.fair_reclaim_idle_groups=fair;
            strix::PrefixStateCache cache(policy,heap);
            auto sibling=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,prompt).front());
            sibling->producer_certificate={17,29};if(named)sibling->conversation="resident";
            const auto key=named?std::string("resident"):strix::model_content_identity_hex(sibling->digest);
            require(cache.insert(sibling),"resident sibling");cache.flush();
            const auto before=strix::PrefixStateCacheTestAccess::clock(cache);
            auto hit=cache.find_longest(id,prompt,T,2*T,promote,named?"resident":"",[](const auto&){return true;});
            require(hit && hit->position==2*T && cache.counters().disk_hits==1,"accepted disk reuse");
            require(strix::PrefixStateCacheTestAccess::reused(cache,key),"disk reuse protects resident group");
            require(strix::PrefixStateCacheTestAccess::clock(cache)==before+(promote?1:0),"disk hit adds only promotion epoch");
            require(strix::PrefixStateCacheTestAccess::reuse(cache,sibling->digest)==0,"disk hit leaves sibling entry reuse unchanged");
            if(promote) {
                require(strix::PrefixStateCacheTestAccess::reuse(cache,hit->digest)==1,"promoted entry reuse remains one");
                strix::PrefixStateCacheTestAccess::retire(cache,hit->digest);
                require(strix::PrefixStateCacheTestAccess::reused(cache,key),"reuse survives retirement of disk-hit checkpoint");
            }
            require(cache.counters().host_checkpoints==1,"no-store never publishes disk entry");
            (void)cache.find_longest(id,prompt,T,2*T,false,"unpublished",[](const auto&){return true;});
            require(strix::PrefixStateCacheTestAccess::association_state(cache,1,fair),"no-store never publishes an alias");
        }
        {
            // Pair privacy is about exact tokens past every rival, not a common
            // header or a recent touch of an older group. Deletion repairs LCPs.
            strix::PrefixStateCache cache({1ULL<<20,{},0},heap);
            const auto root=[&](std::vector<std::uint32_t> ids) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,ids).front());
                c->producer_certificate={17,29};require(cache.insert(c),"prefix-order root");
                return c->digest;
            };
            const auto a=root({7,7,10,10});
            const auto b=root({7,7,20,20});
            const auto c=root({7,7,20,21});
            auto selected=strix::PrefixStateCacheTestAccess::successors(cache);
            require(selected==std::set<std::string>{strix::model_content_identity_hex(b)},"only the private predecessor is superseded");
            strix::PrefixStateCacheTestAccess::retire(cache,b);
            require(strix::PrefixStateCacheTestAccess::successors(cache).empty(),"two groups cannot establish private ancestry");
            (void)root({7,7,20,22});
            require(strix::PrefixStateCacheTestAccess::successors(cache)==std::set<std::string>{strix::model_content_identity_hex(c)},
                "neighbor repair preserves the remaining exact private-prefix relation");
            require(cache.contains(a),"unrelated rival is retained");
        }
        {
            // Eight live agents share three blocks and diverge afterwards. Shared
            // system/tools tokens do not make any older live group a successor.
            strix::PrefixStateCache cache({1ULL<<20,{},0},heap);
            for(std::uint32_t session=0;session<8;++session) {
                auto chain=build(cache,id,{1,2,3,4,5,6,7,8,9,10,11,12,100+session,20,21,22});
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*chain.back());
                c->conversation="live-"+std::to_string(session);c->producer_certificate={17,29};
                require(cache.insert(c),"shared-header live agent");
            }
            require(strix::PrefixStateCacheTestAccess::successors(cache).empty(),"a multi-block shared header is not private history");
        }
        {
            // Even a pair-exclusive complete first block can be a shared header.
            strix::PrefixStateCache cache({1ULL<<20,{},0},heap);
            for(const auto& name: {"agent-a","agent-b"}) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,
                    {1,2,3,4,static_cast<std::uint32_t>(name[6]),20,21,22}).back());
                c->conversation=name;c->producer_certificate={17,29};require(cache.insert(c),"paired header agent");
            }
            auto rival=build(cache,id,{8,9,10,11}).front();require(cache.insert(rival),"different header rival");
            require(strix::PrefixStateCacheTestAccess::successors(cache).empty(),"complete first-block sharing still declines");
        }
        for (const bool reclaim_idle : {false, true})
        for (const bool certified_kind : {false,true})
        for (const bool reuse_idle : {false,true})
        for (const bool lease_idle : {false,true})
        for (const bool retire_hit : {false,true})
        for (const auto clock : {std::uint64_t{0}, UINT64_MAX-257, UINT64_MAX-128}) {
            // An early history rewrite changes the unnamed conversation's first block.
            // The second root has a newer private-history successor, while the
            // first shares only their common header. Live interiors exceed the cap.
            constexpr auto certificate_bytes = 2 * sizeof(std::uint64_t);
            constexpr auto budget = 5 * (fixed_bytes + 3 + certificate_bytes) + 6 * segment_bytes;
            strix::PrefixStateCachePolicy policy{budget, {}, 0};
            policy.fair_reclaim_idle_groups = reclaim_idle;
            strix::PrefixStateCache cache(policy, heap);
            const auto certified = [certified_kind](const auto& c) {
                auto copy = std::make_shared<strix::PrefixCacheCheckpoint>(*c);
                if (certified_kind) copy->producer_certificate = {17, 29};
                return copy;
            };
            std::vector<strix::ModelContentIdentity> dead;
            std::shared_ptr<const strix::PrefixCacheCheckpoint> idle_lease;
            const std::vector<std::vector<std::uint32_t>> roots{{7,7,10,10},{7,7,20,20}};
            for (const auto& root : roots) {
                auto c = certified(build(cache, id, root).front());
                dead.push_back(c->digest);
                require(cache.insert(c), "idle root admitted");
                if (lease_idle && root == roots.back()) idle_lease = c;
                require(cache.counters().host_bytes <= budget, "idle roots stay within budget");
            }
            if (reuse_idle) for (const auto& root : roots) {
                auto prompt = root;
                if (retire_hit) prompt.insert(prompt.end(), root.begin(), root.end());
                std::shared_ptr<strix::PrefixCacheCheckpoint> tail;
                if (retire_hit) {
                    tail=certified(build(cache,id,prompt).back());
                    require(cache.insert(tail),"temporary returning tail");
                }
                require(cache.find_longest(id,prompt,T,prompt.size())!=nullptr,
                    "a returning group earns reuse protection");
                if (retire_hit) strix::PrefixStateCacheTestAccess::retire(cache,tail->digest);
            }
            const std::vector<std::uint32_t> live{7,7,20,21,104,105,106,107,108,109,110,111,112,113,114,115};
            auto chain = build(cache, id, live);
            const auto interior = chain.front()->digest;
            for (std::size_t k = 0; k < 3; ++k)
                require(cache.insert(certified(chain[k])), "live interior admitted");
            auto next = certified(chain.back());
            chain.clear(); // No checkpoint leases may dictate the victim.
            if (clock) strix::PrefixStateCacheTestAccess::move_clock(cache, clock);
            require(cache.policy().fair_reclaim_idle_groups == reclaim_idle,
                "explicit library policy is independent of process environment");
            for (unsigned k = 0; k < 257; ++k) {
                require(cache.find_longest(id, live, T, 3*T)->position == 3*T, "live tail touched");
                require(cache.counters().host_bytes <= budget, "touch stays within budget");
            }
            require(cache.insert(next), "live extension admitted under pressure");
            next.reset();
            require(cache.counters().host_bytes <= budget && cache.counters().evicted == 1,
                "one pressure victim preserves the byte budget");
            const bool refined = reclaim_idle && certified_kind && !reuse_idle && !lease_idle;
            require(cache.contains(interior) == refined, "idle reclamation keeps the live interior; OFF preserves old fair");
            require(cache.contains(dead.back()) != refined, "private predecessor supplies capacity first");
            require(cache.counters().evictions_superseded==(refined?1U:0U),"superseded removal counter");
            if (refined) {
                auto restored = cache.find_longest(id, live, T, T);
                strix::PrefixStateCache fresh({1ULL << 20, {}, 0}, heap);
                auto captured = certified(build(fresh, id, live).front());
                require(restored && same_payloads(*restored, *captured),
                    "post-eviction restore equals fresh capture bytes, metadata and certificate");
            }
        }
        for (const bool reclaim_idle : {false, true}) {
            // Mixed fair store: parked envelopes must preserve the complete old
            // victim order, even when certified checkpoints are also present.
            constexpr std::uint64_t budget = 1759;
            strix::PrefixStateCachePolicy policy{budget, {}, 0};
            policy.fair_reclaim_idle_groups = reclaim_idle;
            strix::PrefixStateCache cache(policy, heap);
            auto a=heap(339), b=heap(339);
            a.data.get()[0]=std::byte{1}; b.data.get()[0]=std::byte{2};
            require(cache.park_session("parked_a",std::move(a),8),"park a in fair store");
            require(cache.park_session("parked_b",std::move(b),8),"park b in fair store");
            const std::vector<std::uint32_t> live{100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115};
            auto chain=build(cache,id,live);
            const auto interior=chain.front()->digest;
            for (std::size_t k=0;k<3;++k) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*chain[k]);
                c->producer_certificate={17,29};
                require(cache.insert(c),"mixed certified checkpoint");
            }
            auto next=std::make_shared<strix::PrefixCacheCheckpoint>(*chain.back());
            next->producer_certificate={17,29}; chain.clear();
            for(unsigned k=0;k<65;++k)
                require(cache.find_longest(id,live,T,3*T)->position==3*T,"mixed live touch");
            require(cache.insert(next),"mixed pressure capture"); next.reset();
            require(cache.list_sessions().size()==2 && !cache.contains(interior) &&
                cache.counters().host_bytes==budget && cache.counters().evicted==1,
                "mixed fair store preserves old victim and byte accounting");
            auto resumed=cache.resume_session("parked_a");
            require(resumed.data && resumed.data.get()[0]==std::byte{1},"parked session remains resumable");
        }
        {
            using namespace strix;
            const auto check=[](bool ok) {require(ok,"pooled idle threshold, leases and accounting");};
            for(bool idle:{false,true}) for(unsigned touches:{29,30,33}) for(bool lease_dead:{false,true}) {
                PrefixStateCachePolicy p{32768,{},0};p.fair_reclaim_idle_groups=idle;
                PrefixStateCache cache(p,heap);
                auto pool=std::make_shared<PrefixPayloadPool>(heap(28672));cache.attach_payload_pool(pool,4096);
                auto capture=[&](std::vector<uint32_t> capture_tokens,uint64_t fixed,uint64_t last) {
                    auto c=std::make_shared<PrefixCacheCheckpoint>();c->identity=id;c->block_rows=T;c->position=capture_tokens.size();c->fixed.bytes=fixed;
                    c->metadata.resize(248);c->producer_certificate={17,29};auto prev=id;
                    std::shared_ptr<PrefixCacheSegment> missing;
                    for(size_t k=0;k<capture_tokens.size()/T;++k) {
                        const auto block=std::span(capture_tokens.data()+k*T,T);
                        const auto d=prefix_cache_block_digest(prev,block);
                        auto s=cache.find_segment(d);
                        if(!s) {missing=std::make_shared<PrefixCacheSegment>();missing->identity=id;missing->parent=prev;missing->digest=d;
                            missing->tokens.assign(block.begin(),block.end());missing->block_rows=T;missing->block_index=k;missing->payload.bytes=last;s=missing;}
                        c->segments.push_back(s);prev=d;
                    }
                    c->digest=prev;
                    auto r=cache.reserve(cache.incremental_bytes(*c),c.get());check(bool(r));
                    if(fixed)c->fixed=cache.allocate(fixed,r);
                    if(missing && last)missing->payload=cache.allocate(last,r,true);
                    check(cache.insert(c,&r));return c;
                };
                auto rival=capture({7,7,10,10},0,0);rival.reset();
                auto dead=capture({7,7,20,20},0,12288);auto dead_digest=dead->digest;dead.reset();
                auto one=capture({7,7,20,21},4096,4096);auto interior=one->digest;one.reset();
                auto two=capture({7,7,20,21,100,101,102,103},4096,4096);auto chain=two->segments;two.reset();
                check(pool->free_bytes()==0);
                for(unsigned i=0;i<touches;++i)check(bool(cache.find_longest(id,std::vector<uint32_t>{7,7,20,21,100,101,102,103},T,2*T)));
                std::shared_ptr<const PrefixCacheCheckpoint> held;
                if(lease_dead)held=cache.find_longest(id,std::vector<uint32_t>{7,7,20,20},T,T);
                auto three=capture({7,7,20,21,100,101,102,103,104,105,106,107},512,0);three.reset();
                const auto stats=cache.counters();
                check(stats.host_bytes==32768 && stats.pool_bytes+stats.pool_metadata_reserved==32768);
                check(stats.evicted==1);check(!held || cache.contains(dead_digest));
                const bool expected_dead_eviction=idle && touches>=30 && !lease_dead;
                check(cache.contains(dead_digest)!=expected_dead_eviction);
                check(cache.contains(interior)==expected_dead_eviction);
                check(chain[0]->payload.data && chain[1]->payload.data);
            }
        }
        {
            // Different conversations share a physical trunk without sharing their fixed state.
            strix::PrefixStateCache cache({1024,{},0},heap);
            auto chain=build(cache,id,{1,2,3,4,5,6,7,8});
            auto a=std::make_shared<strix::PrefixCacheCheckpoint>(*chain[0]); a->conversation="A";
            auto b=std::make_shared<strix::PrefixCacheCheckpoint>(*chain[1]); b->conversation="B";
            require(cache.insert(a) && cache.insert(b),"shared trunk insert");
            const auto expected=2*segment_bytes+2*(fixed_bytes+3);
            require(cache.counters().host_bytes==expected && cache.counters().host_segments==2,"unique payload accounting");
            auto lease=cache.find_longest(id,std::vector<std::uint32_t>{1,2,3,4,5,6,7,8},T,8);
            auto c=build(cache,other,{9,10,11,12})[0]; require(cache.insert(c),"pressure insert");
            require(lease && same_payloads(*lease,*b),"eviction preserves leased rows and shared trunk");
            chain.clear(); a.reset(); b.reset(); c.reset(); lease.reset();
            require(cache.counters().host_bytes==cache.stats().host_bytes,"released leases conserve gauges");
        }
        {
            // Many divergent branches cannot displace a quiet conversation or free its trunk.
            strix::PrefixStateCache cache({2048,{},0},heap);
            auto root=build(cache,id,{1,2,3,4})[0]->segments[0];
            auto quiet=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{1,2,3,4,70,71,72,73})[1]);
            quiet->segments[0]=root; quiet->conversation="quiet";
            require(cache.insert(quiet),"quiet shared-prefix insert");
            const auto quiet_digest=quiet->digest;
            quiet.reset(); // Policy, rather than a checkpoint lease, must protect this conversation.
            for (std::uint32_t branch=0;branch<12;++branch) {
                auto chain=build(cache,id,{1,2,3,4,branch+100,20,21,22});
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*chain[1]);
                c->segments[0]=root; c->conversation="branches";
                (void)cache.insert(c);
            }
            require(cache.contains(quiet_digest),"noisy branches preserve quiet conversation");
            require(cache.find_segment(root->digest)==root && root->payload.data.get()[0]==pattern(1,0),"shared leading prefix survives branch eviction");
            require(cache.counters().evictions_conversation_cap>0,"branch conversation byte cap enforced");
        }
        {
            // Retained content can belong to multiple named conversations without duplication.
            strix::PrefixStateCache cache({4096,{},0},heap);
            auto c=build(cache,id,{1,2,3,4})[0];
            require(cache.insert(c),"membership seed");
            const auto before=cache.counters();
            require(cache.find_longest(id,std::vector<std::uint32_t>{1,2,3,4},T,4,true,"named")==c,"explicit conversation lookup");
            const auto after=cache.counters();
            require(before.host_bytes==after.host_bytes && before.host_segments==after.host_segments && after.captures==1,"membership is not a second payload or capture");
        }
        {
            // One complete deep checkpoint remains useful even above B/3.
            strix::PrefixStateCache cache({800,{},0},heap);
            auto deep=build(cache,id,tokens).back();
            require(cache.insert(deep),"one checkpoint exception");
            require(cache.counters().host_checkpoints==1,"one checkpoint retained");
        }
        {
            // Spare capacity is useful even when multiple conversations exceed nominal caps.
            strix::PrefixStateCache cache({4096,{},0},heap);
            for (std::uint32_t n=0;n<8;++n) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{1,2,3,n+10})[0]);
                c->conversation=n<7 ? "busy" : "quiet";
                require(cache.insert(c),"spare-budget insert");
            }
            require(cache.counters().host_checkpoints==8 && cache.counters().evicted==0,
                "fair caps only order pressure victims");
        }
        {
            // Pre-allocation admission uses the arriving conversation to order pressure victims.
            strix::PrefixStateCache cache({6*(segment_bytes+fixed_bytes+3),{},0},heap);
            std::vector<strix::ModelContentIdentity> keys;
            for (std::uint32_t n=0;n<6;++n) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{1,2,3,n+10})[0]);
                c->conversation="busy"; keys.push_back(c->digest);
                require(cache.insert(c),"capture preflight fill");
            }
            auto next=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,id,{8,9,10,11})[0]);
            next->conversation="quiet";
            require(cache.admit_capture(*next),"capture pressure reservation");
            require(!cache.contains(keys.front()) && cache.counters().evictions_conversation_cap==1,
                "capture admission reclaims borrowed bytes first");
            require(cache.insert(next),"capture inserted after reservation");
        }
        for (const auto retention : {strix::PrefixCacheRetention::lru, strix::PrefixCacheRetention::fair}) {
            strix::PrefixStateCache cache({1ULL<<20,{},0,retention},heap);
            std::vector<strix::ModelContentIdentity> keys;
            for (std::uint32_t n=0;n<260;++n) {
                auto c=build(cache,id,{1,2,3,n+10})[0]; keys.push_back(c->digest);
                require(cache.insert(c),"directory pressure evicts rather than refuses");
            }
            const auto stats=cache.counters();
            require(stats.host_checkpoints==(retention==strix::PrefixCacheRetention::lru ? 260U : 256U),
                "directory cap applies only to fair");
            require(stats.evicted==(retention==strix::PrefixCacheRetention::lru ? 0U : 4U),"entry cap eviction count");
            require(retention==strix::PrefixCacheRetention::lru || !cache.contains(keys.front()),"fair tie uses oldest victim");
        }
        {
            strix::PrefixStateCache cache({1ULL<<20,{},0},heap);
            std::vector<std::shared_ptr<const strix::PrefixCacheCheckpoint>> leases;
            for (std::uint32_t n=0;n<256;++n) {
                leases.push_back(build(cache,id,{1,2,3,n+10})[0]);
                require(cache.insert(leases.back()),"leased directory fill");
            }
            require(!cache.insert(build(cache,id,{1,2,3,999})[0]),"all leased directory refuses");
            require(cache.counters().host_checkpoints==256 && cache.counters().evicted==0,"leased entries survive cap refusal");
        }
        {
            unsigned allocations=0;
            strix::PrefixStateCache cache({100,{},0},[&](std::uint64_t) -> strix::PrefixCacheBuffer {
                ++allocations; throw std::bad_alloc();
            });
            strix::PrefixCacheCheckpoint estimate; estimate.fixed.bytes=101;
            require(!cache.admit_capture(estimate) && allocations==0 && cache.counters().budget_skips==1,
                "impossible checkpoint skips before allocator");
            estimate.fixed.bytes=100;
            require(cache.admit_capture(estimate),"fitting capture estimate admitted");
            bool failed=false;
            try { (void)cache.allocate(100); }
            catch (const strix::PrefixCacheAllocationFailure&) { cache.record_capture_failure(); failed=true; }
            require(failed && allocations==1 && cache.counters().capture_failures==1 && cache.counters().host_bytes==0,
                "allocator throws typed capture failure without publishing partial state");
            strix::PrefixStateCache invalid({100,{},0},[](std::uint64_t) { return strix::PrefixCacheBuffer{}; });
            failed=false;
            try { (void)invalid.allocate(100); } catch (const strix::PrefixCacheAllocationFailure&) { failed=true; }
            require(failed && invalid.counters().host_bytes==0,"empty allocator result has same failure seam");
            strix::PrefixStateCache fatal({100,{},0},[](std::uint64_t) -> strix::PrefixCacheBuffer {
                throw strix::PrefixCacheFatalError("device broken");
            });
            failed=false;
            try { (void)fatal.allocate(100); } catch (const strix::PrefixCacheFatalError&) { failed=true; }
            require(failed,"fatal allocator errors preserve propagation");

        }
        {
            const auto tiny=[&](strix::PrefixStateCache& cache, unsigned token, bool allocate) {
                auto seg=std::make_shared<strix::PrefixCacheSegment>();
                seg->identity=id; seg->parent=id; seg->tokens={token}; seg->block_rows=1;
                seg->digest=strix::prefix_cache_block_digest(id,seg->tokens);
                seg->payload=allocate ? cache.allocate(10) : strix::PrefixCacheBuffer{{},10};
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>();
                c->identity=id; c->digest=seg->digest; c->block_rows=1; c->position=1;
                c->segments={seg}; c->fixed=allocate ? cache.allocate(60) : strix::PrefixCacheBuffer{{},60};
                return c;
            };
            const auto diskdir=dir/"lease-pressure";
            { strix::PrefixStateCache writer({100,diskdir,0},heap);
              require(writer.insert(tiny(writer,1,true)),"disk seed"); writer.flush(); }
            for (const bool promote : {false,true}) {
                std::uint64_t live=0, peak=0;
                strix::PrefixStateCache reader({100,diskdir,0},[&](std::uint64_t n) {
                    live+=n; peak=std::max(peak,live);
                    return strix::PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[n?n:1], [&,n](std::byte* p) { live-=n; delete[] p; }), n};
                });
                auto resident=tiny(reader,2,true);
                require(reader.insert(resident),"resident seed"); reader.flush();
                require(!reader.find_longest(id,std::vector<std::uint32_t>{1},1,1,promote),"disk read must reserve before allocating");
                require(live==70 && peak==70 && reader.counters().host_bytes==70 && reader.counters().budget_skips==1,"disk refusal physical accounting");
            }
            {
                strix::PrefixStateCache reader({100,diskdir,0},heap);
                auto transient=reader.find_longest(id,std::vector<std::uint32_t>{1},1,1,false);
                require(transient && reader.counters().host_checkpoints==0 && reader.counters().host_bytes==70,"unpromoted read charged");
                auto credits=reader.reserve(20);
                require(bool(credits) && !reader.reserve(11),"outstanding reservation charged");
                auto payload=reader.allocate(20,credits);
                require(reader.counters().host_bytes==90 && !reader.reserve(11),"conversion does not double count or lose charge");
                payload={}; transient.reset();
                require(reader.counters().host_bytes==0 && bool(reader.reserve(100)),"transient last lease releases charge");
            }
            {
                const auto shorter_dir=dir/"shorter-reservation";
                { strix::PrefixStateCache writer({1000,shorter_dir,0},heap);
                  for (const auto& c : build(writer,id,{1,2,3,4,5,6,7,8}))
                      require(writer.insert(c),"shorter seed");
                  writer.flush(); }
                strix::PrefixStateCache reader({320,shorter_dir,0},heap);
                auto shorter=reader.find_longest(id,std::vector<std::uint32_t>{1,2,3,4,5,6,7,8},T,8,false);
                require(shorter && shorter->position==4 && reader.counters().budget_skips==1 &&
                    reader.counters().host_bytes==320,"oversized disk candidate falls back to shorter reserved payload");
            }
            strix::PrefixStateCache disk({100,diskdir,0},heap);
            auto held=disk.find_longest(id,std::vector<std::uint32_t>{1},1,1);
            require(held && disk.counters().host_bytes==70,"disk promotion physical charge");
            require(!disk.admit_capture(*tiny(disk,2,false)),"disk hit lease blocks incoming reservation");
            require(!disk.insert(tiny(disk,2,true)),"disk hit lease blocks insertion");
            require(disk.contains(held->digest) && disk.counters().host_bytes==70,"disk hit protects directory and fixed payload");
            disk.flush();
            // A separate checkpoint alias still owns registered fixed/segment payloads
            // after its directory object is evicted. Admission must use the ledger.
            strix::PrefixStateCache aliases({100,{},0},heap);
            auto reader=tiny(aliases,3,true);
            auto directory=std::make_shared<strix::PrefixCacheCheckpoint>(*reader);
            require(aliases.insert(directory),"alias seed"); directory.reset();
            require(!aliases.admit_capture(*tiny(aliases,4,false)),"fixed alias remains charged after eviction");
            require(aliases.counters().host_checkpoints==0 && aliases.counters().host_bytes==70,"evicted fixed lease ledger");
            require(!aliases.insert(tiny(aliases,4,true)),"ledger rejects second retained payload");
            reader.reset();
            require(aliases.counters().host_bytes==0,"last fixed lease releases physical charge");
            strix::PrefixStateCache legacy({100,{},0,strix::PrefixCacheRetention::lru,true},heap);
            auto old_reader=tiny(legacy,5,true);
            require(legacy.insert(old_reader) && legacy.insert(tiny(legacy,6,true)),"baseline keeps old LRU admission with reader");
            const auto old_stats=legacy.counters();
            require(old_stats.evicted==1 && old_stats.inserted==2 && old_stats.host_bytes==80,"baseline legacy gauge and eviction projection");
        }
        {
            // A segment's bytes and count must be observed as one ledger update, even
            // when the final payload owner releases outside the store mutex.
            strix::PrefixStateCache cache({1024,{},0,strix::PrefixCacheRetention::lru},heap);
            for (unsigned n=0;n<200;++n) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>();
                auto seg=std::make_shared<strix::PrefixCacheSegment>();
                seg->identity=id; seg->parent=id; seg->tokens={n}; seg->block_rows=1;
                seg->digest=strix::prefix_cache_block_digest(id,seg->tokens); seg->payload=cache.allocate(64);
                c->identity=id; c->digest=seg->digest; c->block_rows=1; c->position=1; c->segments={seg};
                require(cache.insert(c),"concurrent ledger seed");
                seg.reset();
                // Force directory removal while the checkpoint remains leased.
                strix::PrefixCacheCheckpoint estimate; estimate.fixed.bytes=1024;
                require(!cache.admit_capture(estimate),"lease remains charged after LRU removal");
                std::thread release([lease=std::move(c)]() mutable { lease.reset(); });
                bool coherent=true;
                for(unsigned k=0;k<20;++k) { const auto s=cache.counters();
                    coherent=coherent && s.host_bytes==64*s.host_segments; }
                release.join();
                require(coherent,"coherent ledger gauges");
                require(cache.counters().host_bytes==0,"released ledger reconciles");
            }
        }
        {
            strix::PrefixStateCache legacy({100,{},0,strix::PrefixCacheRetention::lru,true},[](std::uint64_t)->strix::PrefixCacheBuffer {
                throw std::runtime_error("prefix state cache pinned host allocation failed");
            });
            bool old=false; try { (void)legacy.allocate(1); }
            catch(const strix::PrefixCacheAllocationFailure&) { require(false,"baseline reclassified allocator error"); }
            catch(const std::runtime_error& e) {old=std::string(e.what())=="prefix state cache pinned host allocation failed";}
            require(old && legacy.counters().capture_failures==0,"legacy error text and counters");
            strix::PrefixStateCache arbitrary({100,{},0},[](std::uint64_t)->strix::PrefixCacheBuffer { throw std::logic_error("allocator bug"); });
            old=false; try { (void)arbitrary.allocate(1); } catch(const std::logic_error&) {old=true;}
            require(old,"non-resource allocator failures propagate");
        }
        {
            strix::PrefixStateCache named({100,{},0,strix::PrefixCacheRetention::lru},heap);
            require(named.park_session("delete",heap(60),2) && named.delete_session("delete"),"explicit deletion");
            const auto stats=named.counters();
            require(stats.evicted==0 && stats.evictions_explicit==1,"explicit reason does not change legacy evicted");
        }
        {
            // Candidate validation owns a physical lease and runs outside the store mutex.
            // Refusal neither promotes/touches nor counts a hit; continue to shorter prefixes.
            strix::PrefixStateCache cache({1U<<20,dir/"certified",0},heap);
            std::vector<std::uint32_t> candidate_tokens(21,7);
            auto checkpoints=build(cache,id,candidate_tokens);
            for(const auto& cp:checkpoints) {
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>(*cp);
                c->producer_certificate={c->position==16?1U:2U};
                require(cache.insert(c),"certified seed");
            }
            const auto eligible=[&](const strix::PrefixCacheCheckpoint& c) {
                (void)cache.counters(); // Must not deadlock under a store lock.
                return c.producer_certificate==std::vector<std::uint64_t>{1};
            };
            auto hit=cache.find_longest(id,candidate_tokens,T,20,false,{},eligible);
            require(hit && hit->position==16,"incompatible longest falls back to compatible shorter");
            auto stats=cache.counters();
            require(stats.plan_misses==1 && stats.host_hits==1 && stats.restored_tokens==16,"accepted-only accounting");
            hit=cache.find_longest(id,candidate_tokens,T,20,false,{},[](const auto&) {return false;});
            stats=cache.counters();
            require(!hit && stats.plan_misses==6 && stats.host_hits==1 && stats.misses==1 && stats.restored_tokens==16,
                "all refused is a miss, never a false restore");
            cache.flush();
            require(cache.counters().disk_writes==5,"certified checkpoints persist in v1 store");
        }
        {
            // Restart: reconstruct the store, require the certificate before accounting a hit.
            const std::vector<std::uint32_t> candidate_tokens(21,7);
            strix::PrefixStateCache cache({1U<<20,dir/"certified",0},heap);
            const auto eligible=[&](const strix::PrefixCacheCheckpoint& c) {
                (void)cache.counters();
                return c.producer_certificate==std::vector<std::uint64_t>{1};
            };
            auto hit=cache.find_longest(id,candidate_tokens,T,20,false,{},eligible);
            require(hit && hit->position==16 && hit->producer_certificate==std::vector<std::uint64_t>{1},
                "restart restores certificate and matching candidate");
            strix::PrefixStateCache reference({1U<<20,{},0},heap);
            auto expected=std::make_shared<strix::PrefixCacheCheckpoint>(*build(reference,id,candidate_tokens)[3]);
            expected->producer_certificate={1};
            require(same_payloads(*hit,*expected),"certified restart full payload bit equality");
            const auto stats=cache.counters();
            require(stats.disk_hits==1 && stats.plan_misses==1 && stats.host_checkpoints==0 && stats.restored_tokens==16,
                "restart accepted-only counters and no-store");
            require(!cache.find_longest(id,candidate_tokens,T,20,false,{},[](const auto&){return false;}),
                "restart incompatible certificate misses");
            require(!cache.find_longest(id,candidate_tokens,T,20),"legacy lookup cannot restore certified files");
        }
        {
            // A corrupt certificate envelope must miss before any accepted-hit credit.
            const auto tier=dir/"certified";
            const std::vector<std::uint32_t> candidate_tokens(21,7);
            const auto chain=strix::prefix_cache_block_digests(id,candidate_tokens,T,4);
            const auto path=tier/("ckpt-"+strix::model_content_identity_hex(chain.back())+".tmpc");
            {
                std::fstream file(path,std::ios::binary|std::ios::in|std::ios::out);
                file.seekp(128+4*32);file.put(0); // damage the certified metadata count/checksum
            }
            strix::PrefixStateCache cache({1U<<20,tier,0},heap);
            require(!cache.find_longest(id,candidate_tokens,T,20,false,{},[](const auto& c) {
                return c.producer_certificate==std::vector<std::uint64_t>{1};
            }),"corrupt certificate cannot restore");
            require(cache.counters().disk_read_failures==1 && cache.counters().disk_hits==0 &&
                cache.counters().restored_tokens==0,"corrupt envelope has no hit credit");
        }
        {
            strix::PrefixStateCachePolicy policy{1U<<20,dir/"certified",0};
            policy.certified_disk=false;
            strix::PrefixStateCache cache(policy,heap);
            const std::vector<std::uint32_t> candidate_tokens(21,7);
            require(!cache.find_longest(id,candidate_tokens,T,20,false,{},[](const auto&){return true;}),
                "certified disk off switch refuses restart");
            require(cache.counters().disk_reads==0,"off switch avoids disk payload reads");
            auto cp=std::make_shared<strix::PrefixCacheCheckpoint>(*build(cache,other,candidate_tokens)[0]);
            cp->producer_certificate={1};require(cache.insert(cp),"off disk still allows RAM capture");
            cache.flush();require(cache.counters().disk_writes==0,"certified disk off suppresses writer");
        }
        {
            const auto tier=dir/"legacy-v1-refusal";
            const std::vector<std::uint32_t> disk_tokens(9,3);
            {
                strix::PrefixStateCache writer({1U<<20,tier,0},heap);
                for(const auto& c:build(writer,id,disk_tokens)) require(writer.insert(c),"legacy disk seed");
                writer.flush();
            }
            strix::PrefixStateCache reader({1U<<20,tier,0},heap);
            const auto refused=reader.find_longest(id,disk_tokens,T,8,false,{},[](const auto&) {return true;});
            require(!refused && reader.counters().disk_reads==0 && reader.counters().disk_hits==0,
                "certified candidate API cannot consume old v1 disk files");
            const auto legacy_hit=reader.find_longest(id,disk_tokens,T,8);
            require(legacy_hit && reader.counters().disk_hits==1,"explicit legacy disk API remains functional");
        }
        {
            constexpr std::uint64_t budget=8192, metadata=1024;
            auto pool=std::make_shared<strix::PrefixPayloadPool>(heap(budget-metadata));
            strix::PrefixStateCache pooled({budget,{},0,strix::PrefixCacheRetention::lru},heap);
            pooled.attach_payload_pool(pool,metadata);
            auto stats=pooled.counters();
            require(stats.host_bytes==budget && stats.pool_bytes+stats.pool_metadata_reserved==budget &&
                stats.pool_free_bytes==budget-metadata,"all retained slab/free-list bytes charged exactly once to B");
            auto first=pool->allocate(1), second=pool->allocate(257);
            require(second.data.get()-first.data.get()==256 && pool->free_bytes()==budget-metadata-768,
                "pool preserves 256-byte suballocation alignment");
            auto lease=first.data;first={};second={};
            require(pool->free_bytes()==budget-metadata-256,"outstanding lease holds slab range");
            prefix_test_fail_host_allocations=true;
            lease.reset();
            prefix_test_fail_host_allocations=false;
            require(pool->free_bytes()==budget-metadata,"released adjacent slabs coalesce");
            auto left=pool->allocate(256), middle=pool->allocate(256), right=pool->allocate(256);
            prefix_test_fail_host_allocations=true;
            left={}; right={}; middle={};
            prefix_test_fail_host_allocations=false;
            require(pool->free_bytes()==pool->bytes(),"all coalescing forms return leases without host allocation");
            auto full=pool->allocate(budget-metadata);
            bool skipped=false;
            try {(void)pool->allocate(1);} catch(const strix::PrefixCacheAllocationFailure&) {skipped=true;}
            require(skipped,"pool exhaustion refuses immediately");full={};
            strix::PrefixStateCache original({budget,{},0},heap);
            const std::vector<std::uint32_t> input(5,7);
            auto before=build(original,id,input), after=build(pooled,id,input);
            require(same_payloads(*before[0],*after[0]) && before[0]->digest==after[0]->digest,
                "pooled registry payload/metadata/digest are byte-identical to original allocator");
            require(pooled.insert(after[0]),"pooled checkpoint inserted");
            const auto used=pool->bytes()-pool->free_bytes();
            after.clear();
            require(used && pool->free_bytes()!=pool->bytes(),"directory retains payload slabs");
            prefix_test_fail_host_allocations=true;
            auto credit=pooled.reserve(budget);
            prefix_test_fail_host_allocations=false;
            require(bool(credit) && pooled.counters().evicted==1 && pool->free_bytes()==pool->bytes(),
                "eviction returns all payload slabs to free list without freeing backing");
            require(pooled.counters().host_bytes==budget,"eviction keeps physical pool charge B");
        }
        // Real reservations and allocation order, including the physical slab
        // sub-budget rather than forcing eviction with reserve(B).
        for(const auto retention:{strix::PrefixCacheRetention::fair,strix::PrefixCacheRetention::lru}) {
            const auto capture=[&](strix::PrefixStateCache& cache,unsigned token,std::uint64_t bytes,
                std::uint64_t metadata=264) -> std::shared_ptr<const strix::PrefixCacheCheckpoint> {
                auto segment=std::make_shared<strix::PrefixCacheSegment>();
                segment->identity=id;segment->parent=id;segment->tokens={token};segment->block_rows=1;
                segment->digest=strix::prefix_cache_block_digest(id,segment->tokens);segment->payload.bytes=bytes;
                auto c=std::make_shared<strix::PrefixCacheCheckpoint>();
                c->identity=id;c->digest=segment->digest;c->position=1;c->block_rows=1;c->segments={segment};
                c->metadata.resize(metadata);
                auto reservation=cache.reserve(cache.incremental_bytes(*c),c.get());
                if(!reservation) return {};
                segment->payload=cache.allocate(bytes,reservation,true);
                require(cache.insert(c,&reservation),"physical capture insertion");
                return c;
            };
            {
                auto pool=std::make_shared<strix::PrefixPayloadPool>(heap(28672));
                strix::PrefixStateCache cache({32768,{},0,retention},heap);cache.attach_payload_pool(pool,4096);
                require(bool(capture(cache,1,14336)) && bool(capture(cache,2,14336)),"mixed-size slab fill");
                require(pool->free_bytes()==0 && cache.counters().host_checkpoints==2,"physical slab full below logical B");
                const auto replacement=capture(cache,3,512);
                require(replacement && cache.counters().evicted==1 && cache.counters().host_checkpoints==2 &&
                    pool->free_bytes()==13824,"small replacement reclaims eligible checkpoint before allocation");
            }
            {
                auto pool=std::make_shared<strix::PrefixPayloadPool>(heap(28672));
                strix::PrefixStateCache cache({32768,{},0,retention},heap);cache.attach_payload_pool(pool,4096);
                auto a=capture(cache,1,14336),b=capture(cache,2,14336);
                require(a && b && !capture(cache,3,512) && cache.counters().evicted==0,
                    "physical pressure never waits for or evicts leased checkpoints");
                a.reset();
                require(bool(capture(cache,3,512)) && cache.counters().evicted==1,
                    "released reader permits replacement on next capture");
            }
            {
                auto pool=std::make_shared<strix::PrefixPayloadPool>(heap(28672));
                strix::PrefixStateCache cache({32768,{},0,retention},heap);cache.attach_payload_pool(pool,4096);
                auto gap1=pool->allocate(7168);
                require(bool(capture(cache,1,7168)),"fragmented first capture");
                auto gap2=pool->allocate(7168);
                require(bool(capture(cache,2,7168)),"fragmented second capture");
                gap1={};gap2={};
                require(pool->free_bytes()==14336,"enough aggregate free bytes before extent pressure");
                require(bool(capture(cache,3,10000)) && cache.counters().evicted==1,
                    "contiguous extent pressure reclaims in retention order");
            }
            {
                auto pool=std::make_shared<strix::PrefixPayloadPool>(heap(512));
                strix::PrefixStateCache cache({4608,{},0,retention},heap);cache.attach_payload_pool(pool,4096);
                require(bool(capture(cache,1,257)) && pool->free_bytes()==0,"alignment consumes complete payload slab");
                require(bool(capture(cache,2,1)) && cache.counters().evicted==1,
                    "alignment pressure is physical admission pressure");
            }
            {
                auto pool=std::make_shared<strix::PrefixPayloadPool>(heap(28672));
                strix::PrefixStateCache cache({32768,{},0,retention},heap);cache.attach_payload_pool(pool,4096);
                require(bool(capture(cache,1,1,2100)),"metadata allowance first capture");
                require(bool(capture(cache,2,1,2100)) && cache.counters().evicted==1,
                    "metadata stays within the separate retained allowance");
            }
        }
        std::filesystem::remove_all(dir);
        {
            strix::PrefixStateCache cache({32768,{},0},heap);
            strix::PrefixStateCacheTestAccess::decline_while_locked(cache);
            const auto counters=cache.counters();
            require(counters.certificate_declines[19]==1 && counters.certificate_declines[14]==1,
                    "decline counters publish without heap allocation or another store lock");
            static_assert(static_cast<std::size_t>(strix::PrefixCertificateDecline::serial_capture_metadata_mismatch)==10);
        }
        prefix_dma_tests();
        std::cout << "prefix state cache tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(dir);
        std::cerr << "prefix state cache tests failed: " << error.what() << '\n';
        return 1;
    }
}
