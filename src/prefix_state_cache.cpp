#include "strix/prefix_state_cache.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/unbuffered_file.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>

#if defined(__unix__)
#include <fcntl.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <io.h>
#endif

namespace strix {
namespace {

constexpr std::uint64_t segment_magic = 0x31474553'4350'4d54ULL;    // "TMPCSEG1"
constexpr std::uint64_t checkpoint_magic = 0x31504b43'4350'4d54ULL; // "TMPCCKP1"

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

std::string hex(const ModelContentIdentity& digest) { return model_content_identity_hex(digest); }

std::filesystem::path segment_path(const std::filesystem::path& dir, const ModelContentIdentity& digest) {
    return dir / ("seg-" + hex(digest) + ".tmps");
}
std::filesystem::path checkpoint_path(const std::filesystem::path& dir, const ModelContentIdentity& digest) {
    return dir / ("ckpt-" + hex(digest) + ".tmpc");
}

// Fixed-width little-endian header fields; the engine runs on little-endian hosts only.
static_assert(std::endian::native == std::endian::little, "prefix cache files are little-endian");

struct SegmentHeader {
    std::uint64_t magic = segment_magic;
    std::uint8_t identity[32]{}, digest[32]{}, parent[32]{};
    std::uint64_t block_index = 0;
    std::uint32_t block_rows = 0, reserved = 0;
    std::uint64_t payload_bytes = 0, payload_checksum = 0;
};
struct CheckpointHeader {
    std::uint64_t magic = checkpoint_magic;
    std::uint8_t identity[32]{}, digest[32]{};
    std::uint64_t position = 0;
    std::uint32_t block_rows = 0, reserved = 0;
    std::uint64_t segment_count = 0, fixed_bytes = 0, metadata_bytes = 0;
    std::uint64_t fixed_checksum = 0, metadata_checksum = 0;
};
static_assert(sizeof(SegmentHeader) == 8 + 96 + 8 + 8 + 16);
static_assert(sizeof(CheckpointHeader) == 8 + 64 + 8 + 8 + 40);

void put(std::uint8_t (&out)[32], const ModelContentIdentity& value) { std::memcpy(out, value.bytes.data(), 32); }
ModelContentIdentity get(const std::uint8_t (&in)[32]) {
    ModelContentIdentity value;
    std::memcpy(value.bytes.data(), in, 32);
    return value;
}

class File final {
public:
    File(const std::filesystem::path& path, const char* mode) : file_(std::fopen(path.string().c_str(), mode)) {}
    // Whole-file payload streams ("rb" or "wb"): unbuffered unless STRIX_SESSION_DIRECT_IO=0.
    enum class Stream { payload };
    File(const std::filesystem::path& path, const char* mode, Stream) {
        if (!state_file_io::unbuffered_io_enabled()) { file_ = std::fopen(path.string().c_str(), mode); return; }
        if (std::string_view(mode) == "wb") writer_ = std::make_unique<state_file_io::SequentialWriter>(path, false, 0666);
        else reader_ = std::make_unique<state_file_io::SequentialReader>(path);
    }
    ~File() { if (file_) std::fclose(file_); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    [[nodiscard]] bool open() const noexcept {
        return file_ != nullptr || (writer_ && writer_->open()) || (reader_ && reader_->open());
    }
    void read(void* data, std::uint64_t bytes) {
        if (reader_) {
            try { reader_->read(data, bytes); } catch (const std::exception&) { throw std::runtime_error("prefix cache file is truncated"); }
            return;
        }
        auto* out = static_cast<std::byte*>(data);
        while (bytes) {
            const auto chunk = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, 64ULL << 20));
            if (std::fread(out, 1, chunk, file_) != chunk) throw std::runtime_error("prefix cache file is truncated");
            out += chunk;
            bytes -= chunk;
        }
    }
    void write(const void* data, std::uint64_t bytes) {
        if (writer_) {
            try { writer_->write(data, bytes); } catch (const std::exception&) { throw std::runtime_error("prefix cache file write failed"); }
            return;
        }
        const auto* in = static_cast<const std::byte*>(data);
        while (bytes) {
            const auto chunk = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, 64ULL << 20));
            if (std::fwrite(in, 1, chunk, file_) != chunk) throw std::runtime_error("prefix cache file write failed");
            in += chunk;
            bytes -= chunk;
        }
    }
    void close(bool durable = false) {
        if (writer_) {
            auto writer = std::move(writer_);
            try { writer->finish(durable); } catch (const std::exception&) { throw std::runtime_error("prefix cache file sync failed"); }
            return;
        }
        if (file_ && durable) {
            if (std::fflush(file_) != 0) throw std::runtime_error("prefix cache file flush failed");
#if defined(__unix__)
            if (::fdatasync(::fileno(file_)) != 0) throw std::runtime_error("prefix cache file sync failed");
#elif defined(_WIN32)
            if (::_commit(::_fileno(file_)) != 0) throw std::runtime_error("prefix cache file sync failed");
#endif
        }
        if (file_ && std::fclose(file_) != 0) { file_ = nullptr; throw std::runtime_error("prefix cache file close failed"); }
        file_ = nullptr;
    }
private:
    std::FILE* file_ = nullptr;
    std::unique_ptr<state_file_io::SequentialWriter> writer_;
    std::unique_ptr<state_file_io::SequentialReader> reader_;
};

void sync_directory(const std::filesystem::path& dir, bool durable) {
#if defined(__unix__)
    if (!durable) return;
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) throw std::runtime_error("prefix cache directory open failed");
    const int result = ::fsync(fd);
    ::close(fd);
    if (result != 0) throw std::runtime_error("prefix cache directory sync failed");
#else
    // The native serving implementation is POSIX; other hosts sync file contents only.
    static_cast<void>(dir); static_cast<void>(durable);
#endif
}

// A tagged envelope inside the existing v1 metadata section. No parallel files/store.
std::vector<std::byte> disk_metadata(const PrefixCacheCheckpoint& checkpoint) {
    if (checkpoint.producer_certificate.empty()) return checkpoint.metadata;
    const auto count = static_cast<std::uint64_t>(checkpoint.producer_certificate.size());
    std::vector<std::byte> out(sizeof(count) + count*sizeof(std::uint64_t) + checkpoint.metadata.size());
    std::memcpy(out.data(), &count, sizeof(count));
    std::memcpy(out.data()+sizeof(count), checkpoint.producer_certificate.data(), count*sizeof(std::uint64_t));
    std::memcpy(out.data()+sizeof(count)+count*sizeof(std::uint64_t), checkpoint.metadata.data(), checkpoint.metadata.size());
    return out;
}

std::filesystem::path temporary_path(const std::filesystem::path& path) {
    long pid = 0;
#if defined(__unix__)
    pid = static_cast<long>(::getpid());
#endif
    return path.string() + ".tmp." + std::to_string(pid);
}

std::uint64_t checkpoint_host_bytes(const PrefixCacheCheckpoint& checkpoint) {
    return checkpoint.fixed.bytes + checkpoint.metadata.size() + checkpoint.producer_certificate.size()*sizeof(std::uint64_t);
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

std::vector<std::uint32_t> PrefixCacheCheckpoint::tokens() const {
    std::vector<std::uint32_t> out;
    out.reserve(static_cast<std::size_t>(position));
    for (const auto& segment : segments) out.insert(out.end(), segment->tokens.begin(), segment->tokens.end());
    return out;
}

ModelContentIdentity prefix_cache_block_digest(const ModelContentIdentity& previous,
    std::span<const std::uint32_t> block_tokens) {
    std::vector<std::byte> buffer(32 + block_tokens.size_bytes());
    std::memcpy(buffer.data(), previous.bytes.data(), 32);
    if (!block_tokens.empty()) std::memcpy(buffer.data() + 32, block_tokens.data(), block_tokens.size_bytes());
    return sha256_bytes(buffer);
}

std::vector<ModelContentIdentity> prefix_cache_block_digests(const ModelContentIdentity& identity,
    std::span<const std::uint32_t> tokens, std::uint32_t block_rows, std::uint64_t max_blocks) {
    require(block_rows != 0 && identity.valid(), "prefix cache digests need an identity and a block size");
    const auto blocks = std::min<std::uint64_t>(max_blocks, tokens.size() / block_rows);
    std::vector<ModelContentIdentity> digests;
    digests.reserve(static_cast<std::size_t>(blocks));
    auto previous = identity;
    for (std::uint64_t k = 0; k < blocks; ++k) {
        previous = prefix_cache_block_digest(previous, tokens.subspan(static_cast<std::size_t>(k * block_rows), block_rows));
        digests.push_back(previous);
    }
    return digests;
}

std::uint64_t prefix_cache_checksum(std::span<const std::byte> bytes) noexcept {
    constexpr std::uint64_t prime = 0x9e3779b97f4a7c15ULL;
    std::uint64_t lane[4] = {0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL, 0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL};
    const auto* data = bytes.data();
    std::size_t size = bytes.size(), offset = 0;
    for (; offset + 32 <= size; offset += 32) {
        for (unsigned i = 0; i < 4; ++i) {
            std::uint64_t word;
            std::memcpy(&word, data + offset + 8 * i, 8);
            lane[i] = std::rotl((lane[i] ^ word) * prime, 29);
        }
    }
    std::uint64_t tail = 0;
    for (unsigned shift = 0; offset < size; ++offset, shift += 8)
        tail ^= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(data[offset])) << (shift % 64);
    std::uint64_t hash = static_cast<std::uint64_t>(size) * prime;
    for (const auto value : lane) hash = std::rotl((hash ^ value) * prime, 31);
    return std::rotl((hash ^ tail) * prime, 27) ^ (hash >> 17);
}

PrefixStateCache::PrefixStateCache(PrefixStateCachePolicy policy, PrefixCacheAllocator allocate)
    : policy_(std::move(policy)), allocate_(std::move(allocate)) {
    require(policy_.host_byte_budget != 0 && static_cast<bool>(allocate_),
        "prefix state cache needs a positive host byte budget and an allocator");
    if (!policy_.disk_directory.empty()) {
        std::filesystem::create_directories(policy_.disk_directory);
        // Leftovers of interrupted writers (old enough not to belong to a live process).
        sweep_disk(std::chrono::minutes(10));
        update_disk_gauge();
        writer_ = std::thread([this] { writer_loop(); });
    }
}

PrefixStateCache::~PrefixStateCache() {
    if (writer_.joinable()) {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        writer_.join();
    }
}

PrefixStateCache::Reservation::~Reservation() {
    if (ledger_) { std::lock_guard lock(ledger_->mutex); ledger_->reserved -= bytes_; }
}

PrefixStateCache::Reservation PrefixStateCache::reserve(std::uint64_t bytes, const PrefixCacheCheckpoint* incoming) {
    if (policy_.legacy) return {};
    std::lock_guard lock(mutex_);
    std::vector<std::uint64_t> payloads;
    std::uint64_t metadata=0;
    if(payload_pool_ && incoming) {
        // Capture allocates fixed first, then each missing segment. Already leased
        // segments are charged physically by the pool and are never reclaimed here.
        if(!incoming->fixed.data && incoming->fixed.bytes) payloads.push_back(incoming->fixed.bytes);
        for(const auto& segment:incoming->segments)
            if(!segment->payload.data && segment->payload.bytes) payloads.push_back(segment->payload.bytes);
        metadata=incoming->metadata.size()+incoming->producer_certificate.size()*sizeof(std::uint64_t);
    }
    const auto physical_fits=[&] {
        return !payload_pool_ || (metadata<=metadata_allowance_ && metadata_bytes_<=metadata_allowance_-metadata &&
            payload_pool_->can_allocate(payloads));
    };
    if (bytes <= policy_.host_byte_budget) {
        evict_to_budget_locked(bytes, policy_.retention == PrefixCacheRetention::fair ? incoming : nullptr,payloads,metadata);
        if ((!incoming || policy_.retention != PrefixCacheRetention::fair || entries_.size() < 256) &&
            retained_bytes_locked() <= policy_.host_byte_budget - bytes && physical_fits()) {
            std::lock_guard ledger_lock(ledger_->mutex);
            ledger_->reserved += bytes;
            return Reservation(ledger_, bytes);
        }
    }
    ++stats_.budget_skips;
    return {};
}

PrefixCacheBuffer PrefixStateCache::allocate(std::uint64_t bytes, Reservation& reservation, bool segment) {
    if (policy_.legacy) return allocate(bytes);
    if (reservation.ledger_ != ledger_ || bytes > reservation.bytes_)
        throw PrefixCacheAllocationFailure();
    auto buffer = allocate(bytes);
    std::lock_guard lock(mutex_);
    auto receipt = receipts_.at(buffer.data.get()).lock();
    std::lock_guard ledger_lock(ledger_->mutex);
    ledger_->reserved -= bytes;
    reservation.bytes_ -= bytes;
    ledger_->bytes += bytes;
    receipt->registered = true; receipt->segment = segment;
    if (segment) ++ledger_->segments;
    return buffer;
}

PrefixCacheBuffer PrefixStateCache::allocate(std::uint64_t bytes) const {
    if (policy_.legacy) {
        auto buffer = allocate_(bytes);
        if (bytes && (!buffer.data || buffer.bytes < bytes)) throw std::runtime_error("prefix cache allocation failed");
        buffer.bytes = bytes; return buffer;
    }
    PrefixCacheBuffer buffer;
    try { buffer = allocate_(bytes); }
    catch (const PrefixCacheFatalError&) { throw; }
    catch (const std::bad_alloc&) { throw PrefixCacheAllocationFailure(); }
    if (bytes && (!buffer.data || buffer.bytes < bytes)) throw PrefixCacheAllocationFailure();
    buffer.bytes = bytes;
    auto receipt = std::make_shared<Receipt>();
    receipt->ledger = ledger_; receipt->payload = buffer.data; receipt->bytes = bytes;
    buffer.data = std::shared_ptr<std::byte>(receipt, buffer.data.get());
    std::lock_guard lock(mutex_);
    receipts_[buffer.data.get()] = receipt;
    return buffer;
}

std::uint64_t PrefixStateCache::retained_bytes_locked() {
    if (policy_.legacy) return host_bytes_locked();
    std::lock_guard lock(ledger_->mutex);
    return ledger_->bytes + ledger_->reserved + metadata_bytes_;
}

std::uint64_t PrefixStateCache::host_bytes_locked() {
    std::uint64_t total = 0;
    for (auto it = segments_.begin(); it != segments_.end();) {
        if (const auto segment = it->second.lock()) {
            total += segment->payload.bytes;
            ++it;
        } else {
            it = segments_.erase(it);
        }
    }
    for (auto it = session_buffers_.begin(); it != session_buffers_.end();) {
        if (it->second.first.expired()) it = session_buffers_.erase(it);
        else { total += it->second.second; ++it; }
    }
    for (const auto& [digest, entry] : entries_) {
        total += checkpoint_host_bytes(*entry.checkpoint);
        if (session_buffers_.count(entry.checkpoint->fixed.data.get())) total -= entry.checkpoint->fixed.bytes;
    }
    return total;
}

std::uint64_t PrefixStateCache::incremental_bytes(const PrefixCacheCheckpoint& checkpoint) {
    std::lock_guard lock(mutex_);
    return incremental_bytes_locked(checkpoint);
}

std::uint64_t PrefixStateCache::incremental_bytes_locked(const PrefixCacheCheckpoint& checkpoint) {
    std::uint64_t total = checkpoint_host_bytes(checkpoint);
    const auto registered = [&](const PrefixCacheBuffer& b) {
        const auto it = receipts_.find(b.data.get());
        const auto r = it == receipts_.end() ? nullptr : it->second.lock();
        return b.data && r && r->registered;
    };
    if (!policy_.legacy && registered(checkpoint.fixed)) total -= checkpoint.fixed.bytes;
    for (const auto& segment : checkpoint.segments) {
        const auto found = segments_.find(segment->digest);
        if (policy_.legacy ? (found == segments_.end() || found->second.expired()) : !registered(segment->payload))
            total += segment->payload.bytes;
    }
    return total;
}

void PrefixStateCache::register_buffer_locked(const PrefixCacheBuffer& buffer, bool segment) {
    const auto it = receipts_.find(buffer.data.get());
    if (it == receipts_.end()) return; // external named envelopes use session_buffers_ accounting
    if (auto r = it->second.lock(); r && !r->registered) {
        r->registered = true; r->segment = segment;
        std::lock_guard lock(ledger_->mutex);
        ledger_->bytes += r->bytes;
        if (segment) ++ledger_->segments;
    }
}
namespace {
std::size_t group_prefix_lcp(const PrefixCacheSegment& a, const PrefixCacheSegment& b) {
    if (a.identity != b.identity || a.block_rows != b.block_rows) return 0;
    std::size_t n = 0;
    while (n < std::min(a.tokens.size(), b.tokens.size()) && a.tokens[n] == b.tokens[n]) ++n;
    return n;
}
bool group_prefix_before(const PrefixCacheSegment& a, std::string_view ak,
                         const PrefixCacheSegment& b, std::string_view bk) {
    if (a.identity != b.identity) return a.identity.bytes < b.identity.bytes;
    if (a.block_rows != b.block_rows) return a.block_rows < b.block_rows;
    if (a.tokens != b.tokens) return a.tokens < b.tokens;
    return ak < bk;
}
}
void PrefixStateCache::update_group_root_locked(const std::string& key,
    const std::shared_ptr<const PrefixCacheSegment>& root) {
    auto& activity = group_activity_.at(key);
    if (!activity.members) activity.born_epoch = epoch_;
    if (!policy_.fair_reclaim_idle_groups || policy_.retention != PrefixCacheRetention::fair) return;
    if (activity.members) {
        const auto previous = activity.root.lock();
        if (!previous || !root || previous->identity != root->identity || previous->block_rows != root->block_rows ||
            previous->tokens != root->tokens) group_prefix_order_valid_ = false;
        activity.root = root;
        return;
    }
    // Compute the link and overlaps before the final throwing allocation. Nothing
    // visible (including validity and the following LCP) changes on failure.
    auto next = group_prefix_order_.end();
    bool valid = group_prefix_order_valid_ && bool(root);
    if (valid) {
        for (auto it = group_prefix_order_.begin(); it != group_prefix_order_.end(); ++it) {
            const auto other = (*it)->root.lock();
            if (!other) { valid = false; break; }
            if (group_prefix_before(*root, key, *other, (*it)->key)) { next = it; break; }
        }
    }
    std::size_t previous_lcp = 0, next_lcp = 0;
    if (valid) {
        if (next != group_prefix_order_.begin())
            previous_lcp = group_prefix_lcp(*(*std::prev(next))->root.lock(), *root);
        if (next != group_prefix_order_.end())
            next_lcp = group_prefix_lcp(*root, *(*next)->root.lock());
    }
    const auto link = group_prefix_order_.insert(next, &activity);
    activity.root = root;
    activity.key = group_activity_.find(key)->first;
    activity.prefix_order = link;
    activity.prefix_lcp = previous_lcp;
    group_prefix_order_valid_ = valid;
    if (valid && next != group_prefix_order_.end()) (*next)->prefix_lcp = next_lcp;
}
void PrefixStateCache::mark_private_prefix_successors_locked(
    const PrefixCacheCheckpoint* incoming, std::uint64_t newest) {
    for (auto* activity : group_prefix_order_) activity->private_successor = false;
    if (!policy_.fair_reclaim_idle_groups || !group_prefix_order_valid_ || named_session_parking_) return;
    struct Leaf { std::string_view key; GroupActivity* activity; const PrefixCacheSegment* root; std::size_t lcp; std::uint64_t birth_age; };
    std::vector<Leaf> leaves;
    leaves.reserve(group_prefix_order_.size() + 1);
    for (auto* activity : group_prefix_order_) {
        const auto root = activity->root.lock();
        if (!root) return;
        leaves.push_back({activity->key, activity, root.get(), activity->prefix_lcp, newest - activity->born_epoch});
    }
    std::string incoming_key;
    if (incoming) {
        if (incoming->segments.empty()) return;
        const auto& root = *incoming->segments.front();
        incoming_key = incoming->conversation.empty() ? hex(root.digest) : incoming->conversation;
        const auto& key = incoming_key;
        if (const auto a = group_activity_.find(key); a != group_activity_.end()) {
            const auto previous = a->second.root.lock();
            if (!previous || previous->identity != root.identity || previous->block_rows != root.block_rows ||
                previous->tokens != root.tokens) return;
        } else {
            // One linear merge of the arriving group; no sorting or group-pair scan.
            const auto at = std::find_if(leaves.begin(), leaves.end(), [&](const auto& leaf) {
                return group_prefix_before(root, key, *leaf.root, leaf.key);
            });
            const auto offset = static_cast<std::size_t>(at - leaves.begin());
            const auto before = offset ? group_prefix_lcp(*leaves[offset - 1].root, root) : 0;
            if (at != leaves.end()) at->lcp = group_prefix_lcp(root, *at->root);
            leaves.insert(at, Leaf{key, nullptr, &root, before, 0});
        }
    }
    if (leaves.size() < 3) return; // No rival history with which to establish exclusivity.
    std::vector<std::size_t> adjacent(leaves.size() + 1, 0);
    for (std::size_t i = 1; i < leaves.size(); ++i) adjacent[i] = leaves[i].lcp;
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        // Lexicographic neighbors give the exact maximum LCP with every other
        // retained group. Remove the proposed successor before testing privacy.
        if (i && adjacent[i] < leaves[i].root->tokens.size() && adjacent[i] < leaves[i - 1].root->tokens.size() &&
            adjacent[i] > std::max(adjacent[i + 1], std::min(adjacent[i], adjacent[i - 1])) &&
            leaves[i].birth_age > leaves[i - 1].birth_age && leaves[i].activity) leaves[i].activity->private_successor = true;
        if (i + 1 < leaves.size() && adjacent[i + 1] < leaves[i].root->tokens.size() && adjacent[i + 1] < leaves[i + 1].root->tokens.size() &&
            adjacent[i + 1] > std::max(adjacent[i], std::min(adjacent[i + 1], adjacent[i + 2])) &&
            leaves[i].birth_age > leaves[i + 1].birth_age && leaves[i].activity) leaves[i].activity->private_successor = true;
    }
}
void PrefixStateCache::associate_locked(Entry& entry, const std::string& key, bool reused) {
    if (entry.conversations.contains(key) || entry.conversations.size() >= 16) return;
    // Roll back both allocating containers before publishing members or links.
    const auto membership = entry.conversations.insert(key).first;
    try {
        const auto [group, created] = group_activity_.try_emplace(key);
        try {
            update_group_root_locked(key, entry.checkpoint->segments.empty() ? nullptr : entry.checkpoint->segments.front());
        } catch (...) {
            if (created) group_activity_.erase(group);
            throw;
        }
        ++group->second.members;
        group->second.reused = group->second.reused || reused || entry.reuse != 0;
    } catch (...) {
        entry.conversations.erase(membership);
        throw;
    }
}
void PrefixStateCache::add_locked(std::shared_ptr<const PrefixCacheCheckpoint> checkpoint) {
    const auto key = checkpoint->conversation.empty() ?
        hex(checkpoint->segments.empty() ? checkpoint->digest : checkpoint->segments.front()->digest) : checkpoint->conversation;
    // Allocate missing index nodes privately. Keep existing weak values observable
    // until all directory/LRU/association allocations have succeeded.
    decltype(segments_) staged_segments;
    for (const auto& segment : checkpoint->segments)
        if (!segments_.contains(segment->digest)) staged_segments.try_emplace(segment->digest, segment);
    lru_.push_back(checkpoint->digest);
    const auto link = std::prev(lru_.end());
    try {
        const auto [entry, inserted] = entries_.emplace(checkpoint->digest, Entry{checkpoint, link, epoch_ + 1, 0, {}});
        if (!inserted) { lru_.erase(link); return; }
        ++epoch_;
        try {
            associate_locked(entry->second, key);
        } catch (...) {
            --epoch_;
            entries_.erase(entry);
            throw;
        }
    } catch (...) {
        lru_.erase(link);
        throw;
    }
    // Node transfer and weak-pointer assignment allocate nothing; DigestLess is
    // noexcept. Repeated digests retain the same last-value-wins behavior.
    segments_.merge(staged_segments);
    for (const auto& segment : checkpoint->segments) segments_.find(segment->digest)->second = segment;
    for (auto it = receipts_.begin(); it != receipts_.end();)
        if (it->second.expired()) it = receipts_.erase(it); else ++it;
    metadata_bytes_ += checkpoint->metadata.size() + checkpoint->producer_certificate.size()*sizeof(std::uint64_t);
    register_buffer_locked(checkpoint->fixed, false);
    for (const auto& segment : checkpoint->segments) {
        register_buffer_locked(segment->payload, true);
    }
}
void PrefixStateCache::erase_locked(const ModelContentIdentity& digest) {
    const auto it = entries_.find(digest);
    if (it == entries_.end()) return;
    metadata_bytes_ -= it->second.checkpoint->metadata.size() + it->second.checkpoint->producer_certificate.size()*sizeof(std::uint64_t);
    for (const auto& key : it->second.conversations) {
        const auto group = group_activity_.find(key);
        if (--group->second.members == 0) {
            if (policy_.fair_reclaim_idle_groups && policy_.retention == PrefixCacheRetention::fair) {
                const auto next = std::next(group->second.prefix_order);
                if (next != group_prefix_order_.end()) (*next)->prefix_lcp =
                    std::min(group->second.prefix_lcp, (*next)->prefix_lcp);
                group_prefix_order_.erase(group->second.prefix_order);
            }
            group_activity_.erase(group);
            if (group_activity_.empty()) group_prefix_order_valid_ = true;
        } else if (policy_.fair_reclaim_idle_groups && policy_.retention == PrefixCacheRetention::fair) {
            // Keep the weak representative backed by a retained entry even when
            // callers supplied equal prefixes in distinct segment objects.
            for (const auto& [other_digest, other] : entries_) {
                if (other_digest == digest || !other.conversations.contains(key)) continue;
                if (other.checkpoint->segments.empty()) group_prefix_order_valid_ = false;
                group->second.root = other.checkpoint->segments.empty() ? nullptr : other.checkpoint->segments.front();
                break;
            }
        }
    }
    lru_.erase(it->second.lru);
    entries_.erase(it);
}
void PrefixStateCache::touch_locked(Entry& entry, bool reused) {
    entry.epoch = ++epoch_;
    if (reused) {
        ++entry.reuse;
        for (const auto& key : entry.conversations) group_activity_.at(key).reused = true;
    }
    lru_.splice(lru_.end(), lru_, entry.lru);
}

void PrefixStateCache::evict_to_budget_locked(std::uint64_t reserve, const PrefixCacheCheckpoint* incoming,
    std::span<const std::uint64_t> payloads, std::uint64_t metadata) {
    // Directory/association work is bounded independently of the payload size.
    auto remaining=payload_pool_?std::min<std::size_t>(entries_.size(),256):entries_.size();
    while (!lru_.empty() && remaining--) {
        const bool physical_pressure=payload_pool_ &&
            (metadata>metadata_allowance_ || metadata_bytes_>metadata_allowance_-metadata || !payload_pool_->can_allocate(payloads));
        const bool pressure = retained_bytes_locked() + reserve > policy_.host_byte_budget || physical_pressure;
        const bool entry_pressure = policy_.retention == PrefixCacheRetention::fair && entries_.size() + (incoming ? 1U : 0U) > 256;
        if (!pressure && !entry_pressure) break;
        auto victim = entries_.find(lru_.front());
        if(payload_pool_ && policy_.retention==PrefixCacheRetention::lru) {
            auto oldest=lru_.begin();
            while(oldest!=lru_.end() && entries_.at(*oldest).checkpoint.use_count()!=1) ++oldest;
            if(oldest==lru_.end()) break;
            victim=entries_.find(*oldest);
        }
        unsigned victim_rank = 0;
        bool victim_superseded = false;
        long double best = -1;
        if (policy_.retention == PrefixCacheRetention::fair) {
            struct Group {
                long double bytes = 0;
                std::uint64_t largest = 0;
                std::uint64_t age = UINT64_MAX;
                bool reused = false;
                const GroupActivity* activity = nullptr;
                std::map<ModelContentIdentity, long double, DigestLess> branches;
                std::map<ModelContentIdentity, std::uint64_t, DigestLess> branch_largest;
            };
            std::map<std::string, Group> groups;
            std::map<ModelContentIdentity, std::size_t, DigestLess> refs;
            std::map<ModelContentIdentity, std::map<std::string, std::size_t>, DigestLess> owners;
            // Include a would-be capture in fairness accounting before allocating its payload.
            Entry estimate;
            // One epoch is one capture or validated touch anywhere in this store,
            // across all conversations; neither wall time nor per-conversation turns.
            // Compare modular ages from a common reference, never raw epoch maxima.
            const auto newest = epoch_ + (incoming ? 1U : 0U);
            std::uint64_t newest_age = UINT64_MAX;
            bool certified_only = true;
            std::vector<std::pair<ModelContentIdentity, const Entry*>> ranking;
            ranking.reserve(entries_.size() + (incoming ? 1U : 0U));
            for (const auto& [d,e] : entries_) ranking.emplace_back(d, &e);
            if (incoming) {
                estimate.epoch = epoch_ + 1;
                estimate.checkpoint = std::shared_ptr<const PrefixCacheCheckpoint>(incoming, [](const auto*) {});
                estimate.conversations.insert(incoming->conversation.empty() ?
                    hex(incoming->segments.empty() ? incoming->digest : incoming->segments.front()->digest) : incoming->conversation);
                ranking.emplace_back(incoming->digest, &estimate);
            }
            for (const auto& item : ranking) {
                const auto& e = *item.second;
                for (const auto& seg : e.checkpoint->segments) {
                    ++refs[seg->digest];
                    for (const auto& key : e.conversations) ++owners[seg->digest][key];
                }
            }
            std::map<ModelContentIdentity, std::map<std::string, ModelContentIdentity>, DigestLess> branches;
            for (const auto& [d,ep] : ranking) {
              const auto& e = *ep;
              certified_only = certified_only && !e.checkpoint->producer_certificate.empty() &&
                  !e.checkpoint->segments.empty();
              for (const auto& key : e.conversations) {
                const auto& c = *e.checkpoint;
                // Branch = first block after the deepest retained shared ancestor.
                // An ancestor ending at this checkpoint uses its own terminal digest.
                std::size_t shared = 0;
                while (shared<c.segments.size() && owners.at(c.segments[shared]->digest).at(key)>1) ++shared;
                const auto branch = c.segments.empty() ? d : c.segments[std::min(shared,c.segments.size()-1)]->digest;
                branches[d][key] = branch;
                auto full = checkpoint_host_bytes(c);
                auto charge = static_cast<long double>(full)/e.conversations.size();
                for (const auto& seg : c.segments) {
                    full += seg->payload.bytes;
                    charge += static_cast<long double>(seg->payload.bytes) / (owners[seg->digest].at(key) * owners[seg->digest].size());
                }
                auto& g = groups[key]; g.bytes += charge; g.largest = std::max(g.largest,full);
                g.age = std::min(g.age, newest - e.epoch);
                if (const auto activity = group_activity_.find(key); activity != group_activity_.end()) {
                    g.reused = activity->second.reused;
                    g.activity = &activity->second;
                }
                newest_age = std::min(newest_age, g.age);
                g.branches[branch] += charge;
                g.branch_largest[branch] = std::max(g.branch_largest[branch],full);
              }
            }
            if (certified_only) mark_private_prefix_successors_locked(incoming, newest);
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                if (it->second.checkpoint.use_count() != 1) continue;
                const auto& c = *it->second.checkpoint;
                unsigned rank = 0;
                unsigned ordinary_rank = 0;
                for (const auto& key : it->second.conversations) {
                    const auto& g = groups.at(key);
                    const auto cap = std::min(policy_.host_byte_budget, std::max(g.largest,policy_.host_byte_budget/3));
                    const auto branch = branches.at(it->first).at(key);
                    const auto branch_cap = std::min(cap,std::max(g.branch_largest.at(branch),cap/g.branches.size()));
                    const bool borrowed = groups.size()>1 && g.bytes>cap;
                    // Age alone cannot establish abandonment. Require a newer group
                    // sharing a prefix private to this pair, beyond every rival's
                    // divergence. Common headers and reused groups cannot qualify.
                    const bool idle = policy_.fair_reclaim_idle_groups && !named_session_parking_ && certified_only &&
                        !g.reused && g.activity && g.activity->private_successor && g.age - newest_age > 32;
                    ordinary_rank = std::max(ordinary_rank, g.branches.at(branch)>branch_cap ? 3U : borrowed ? 2U : 0U);
                    rank = std::max(rank, g.branches.at(branch)>branch_cap ? 3U : borrowed || idle ? 2U : 0U);
                }
                const bool unleased = it->second.checkpoint.use_count()==1;
                std::uint64_t marginal = unleased ? c.metadata.size() + c.producer_certificate.size()*sizeof(std::uint64_t) : 0;
                if (unleased && c.fixed.data.use_count()==1) marginal += c.fixed.bytes;
                for (const auto& seg : c.segments)
                    if (unleased && refs[seg->digest] == 1 && seg.use_count()==1) marginal += seg->payload.bytes;
                const auto score = static_cast<long double>(marginal) * (1 + epoch_ - it->second.epoch) / (1 + it->second.reuse);
                if (rank > victim_rank || (rank == victim_rank && (score > best ||
                    (score == best && (it->second.epoch < victim->second.epoch ||
                        (it->second.epoch == victim->second.epoch && DigestLess{}(it->first,victim->first))))))) {
                    victim = it; victim_rank = rank; best = score;
                    victim_superseded = rank > ordinary_rank;
                }
            }
        }
        if (policy_.retention == PrefixCacheRetention::fair && best < 0) break;
        const auto digest = victim->first;
        erase_locked(digest);
        ++stats_.evicted;
        if (victim_superseded) ++stats_.evictions_superseded;
        else if (victim_rank == 2) ++stats_.evictions_conversation_cap;
        else if (victim_rank == 3) ++stats_.evictions_branch_cap;
        else ++stats_.evictions_space;
    }
}

bool PrefixStateCache::insert(std::shared_ptr<const PrefixCacheCheckpoint> checkpoint, Reservation* reservation) {
    require(checkpoint && checkpoint->block_rows && checkpoint->position == checkpoint->segments.size() * checkpoint->block_rows,
        "prefix cache checkpoint is malformed");
    require(checkpoint->conversation.size()<=128,"prefix cache conversation key exceeds 128 bytes");
    {
        std::lock_guard lock(mutex_);
        if (const auto found = entries_.find(checkpoint->digest); found != entries_.end()) {
            if (!checkpoint->conversation.empty() && checkpoint->conversation.size()<=128 && found->second.conversations.size()<16)
                associate_locked(found->second, checkpoint->conversation);
            touch_locked(found->second);
            return true;
        }
        if (reservation && reservation->ledger_ == ledger_) {
            std::lock_guard ledger_lock(ledger_->mutex);
            ledger_->reserved -= reservation->bytes_; reservation->bytes_ = 0;
        }
        ++stats_.captures;
        const auto metadata=checkpoint->metadata.size()+checkpoint->producer_certificate.size()*sizeof(std::uint64_t);
        if(payload_pool_ && metadata>metadata_allowance_-std::min(metadata_allowance_,metadata_bytes_)) {
            ++stats_.budget_skips; return false;
        }
        const auto need = incremental_bytes_locked(*checkpoint);
        if (need > policy_.host_byte_budget) {
            ++stats_.rejected_over_budget; ++stats_.budget_skips;
            return false;
        }
        if (!policy_.legacy) {
            evict_to_budget_locked(need, policy_.retention == PrefixCacheRetention::fair ? checkpoint.get() : nullptr);
            if ((policy_.retention == PrefixCacheRetention::fair && entries_.size() >= 256) ||
                retained_bytes_locked() > policy_.host_byte_budget - need) {
                ++stats_.budget_skips; ++stats_.rejected_over_budget; return false;
            }
        } else evict_to_budget_locked(need);
        add_locked(checkpoint);
        ++stats_.inserted;
    }
    if (writer_.joinable() && (checkpoint->producer_certificate.empty() || (policy_.certified_disk && !policy_.legacy))) {
        {
            std::lock_guard lock(queue_mutex_);
            queue_.push_back(std::move(checkpoint));
        }
        queue_cv_.notify_one();
    }
    return true;
}

bool PrefixStateCache::contains(const ModelContentIdentity& digest) {
    {
        std::lock_guard lock(mutex_);
        if (entries_.count(digest)) return true;
    }
    if (policy_.disk_directory.empty()) return false;
    std::error_code error;
    return std::filesystem::exists(checkpoint_path(policy_.disk_directory, digest), error);
}

std::shared_ptr<const PrefixCacheSegment> PrefixStateCache::find_segment(const ModelContentIdentity& digest) {
    std::lock_guard lock(mutex_);
    const auto found = segments_.find(digest);
    return found == segments_.end() ? nullptr : found->second.lock();
}

std::shared_ptr<const PrefixCacheCheckpoint> PrefixStateCache::find_longest(const ModelContentIdentity& identity,
    std::span<const std::uint32_t> tokens, std::uint32_t block_rows, std::uint64_t limit, bool promote, std::string_view conversation,
    const std::function<bool(const PrefixCacheCheckpoint&)>& eligible) {
    require(block_rows != 0, "prefix cache lookup needs a block size");
    const auto max_blocks = std::min<std::uint64_t>(limit, tokens.size()) / block_rows;
    const auto digests = prefix_cache_block_digests(identity, tokens, block_rows, max_blocks);
    {
        std::lock_guard lock(mutex_);
        for (auto it=receipts_.begin(); it!=receipts_.end();)
            if (it->second.expired()) it=receipts_.erase(it); else ++it;
        ++stats_.lookups;
        stats_.lookup_tokens += std::min<std::uint64_t>(limit,tokens.size());
    }
    for (auto k = digests.size(); k > 0; --k) {
        const auto& digest = digests[k - 1];
        std::shared_ptr<const PrefixCacheCheckpoint> found;
        {
            std::lock_guard lock(mutex_);
            if (const auto it = entries_.find(digest); it != entries_.end()) {
                found = it->second.checkpoint;
            }
        }
        if (found) {
            if (eligible && !eligible(*found)) {
                std::lock_guard lock(mutex_); ++stats_.plan_misses; continue;
            }
            const auto position = static_cast<std::uint64_t>(k) * block_rows;
            const auto stored = found->tokens();
            if (found->identity == identity && found->block_rows == block_rows && found->position == position &&
                stored.size() == position && std::equal(stored.begin(), stored.end(), tokens.begin())) {
                std::lock_guard lock(mutex_);
                if (auto it=entries_.find(digest); it!=entries_.end()) {
                    touch_locked(it->second,true);
                    if (promote && !conversation.empty() && conversation.size()<=128 && it->second.conversations.size()<16)
                        associate_locked(it->second, std::string(conversation), true);
                }
                ++stats_.host_hits; stats_.hit_tokens += position; stats_.restored_tokens += position;
                return found;
            }
            { std::lock_guard lock(mutex_); erase_locked(digest); ++stats_.evicted; ++stats_.evictions_invalid; }
            continue;  // digest collision or corrupt entry: never used
        }
        if (policy_.disk_directory.empty() || (eligible && (!policy_.certified_disk || policy_.legacy))) continue;
        std::error_code error;
        if (!std::filesystem::exists(checkpoint_path(policy_.disk_directory, digest), error)) continue;
        try {
            // Certified and legacy lookups never cross. Check before payload allocation.
            File probe(checkpoint_path(policy_.disk_directory, digest), "rb");
            CheckpointHeader candidate;
            if (!probe.open()) continue;
            probe.read(&candidate, sizeof(candidate));
            if (candidate.reserved != (eligible ? 1U : 0U)) continue;
            if (auto loaded = load_from_disk(identity, digest, tokens.first(static_cast<std::size_t>(k * block_rows)), block_rows)) {
                if (eligible && (loaded->producer_certificate.empty() || !eligible(*loaded))) {
                    std::lock_guard lock(mutex_); ++stats_.plan_misses; continue;
                }
                {
                    std::lock_guard lock(mutex_);
                    // No-store hits protect existing retained groups without creating
                    // associations or ticking the capture/touch clock.
                    const auto key = conversation.empty() ?
                        hex(loaded->segments.empty() ? loaded->digest : loaded->segments.front()->digest) : std::string(conversation);
                    if (auto group = group_activity_.find(key); group != group_activity_.end()) group->second.reused = true;
                    ++stats_.disk_hits; stats_.hit_tokens += loaded->position; stats_.restored_tokens += loaded->position;
                }
                // Into the host tier without writing it back.
                if (promote) {
                    std::lock_guard lock(mutex_);
                    const auto need = incremental_bytes_locked(*loaded);
                    if (need <= policy_.host_byte_budget && !entries_.count(loaded->digest)) {
                        // The returned handle is the directory lease, including its fixed blob.
                        auto promoted=std::const_pointer_cast<PrefixCacheCheckpoint>(loaded);
                        if (conversation.size()<=128) promoted->conversation=conversation;
                        evict_to_budget_locked(need, policy_.retention == PrefixCacheRetention::fair ? promoted.get() : nullptr);
                        if ((policy_.retention != PrefixCacheRetention::fair || entries_.size()<256) &&
                            (policy_.legacy || retained_bytes_locked() <= policy_.host_byte_budget - need)) {
                            add_locked(promoted);
                            auto& entry = entries_.at(promoted->digest);
                            entry.reuse=1;
                            for (const auto& key : entry.conversations) group_activity_.at(key).reused=true;
                        }
                    }
                }
                std::filesystem::last_write_time(checkpoint_path(policy_.disk_directory, digest),
                    std::filesystem::file_time_type::clock::now(), error);
                return loaded;
            }
        } catch (const std::exception&) {
            std::lock_guard lock(mutex_);
            ++stats_.disk_read_failures;
        }
    }
    std::lock_guard lock(mutex_);
    ++stats_.misses;
    return nullptr;
}

std::shared_ptr<const PrefixCacheSegment> PrefixStateCache::load_segment(const ModelContentIdentity& identity,
    const ModelContentIdentity& digest, std::uint32_t block_rows, Reservation* reservation) {
    if (auto retained = find_segment(digest)) return retained;
    File file(segment_path(policy_.disk_directory, digest), "rb", File::Stream::payload);
    if (!file.open()) throw std::runtime_error("prefix cache segment file is missing");
    SegmentHeader header;
    file.read(&header, sizeof(header));
    if (header.magic != segment_magic || get(header.identity) != identity || get(header.digest) != digest ||
        header.block_rows != block_rows)
        throw std::runtime_error("prefix cache segment header does not match");
    auto segment = std::make_shared<PrefixCacheSegment>();
    segment->identity = identity;
    segment->digest = digest;
    segment->parent = get(header.parent);
    segment->block_index = header.block_index;
    segment->block_rows = block_rows;
    segment->tokens.resize(block_rows);
    file.read(segment->tokens.data(), static_cast<std::uint64_t>(block_rows) * sizeof(std::uint32_t));
    if (prefix_cache_block_digest(segment->parent, segment->tokens) != digest)
        throw std::runtime_error("prefix cache segment tokens do not match their digest");
    segment->payload = reservation ? allocate(header.payload_bytes, *reservation, true) : allocate(header.payload_bytes);
    file.read(segment->payload.data.get(), header.payload_bytes);
    if (prefix_cache_checksum(segment->payload.span()) != header.payload_checksum)
        throw std::runtime_error("prefix cache segment payload checksum differs");
    std::lock_guard lock(mutex_);
    stats_.disk_read_bytes += sizeof(header) + header.payload_bytes;
    return segment;
}

std::shared_ptr<const PrefixCacheCheckpoint> PrefixStateCache::load_from_disk(const ModelContentIdentity& identity,
    const ModelContentIdentity& digest, std::span<const std::uint32_t> tokens, std::uint32_t block_rows) {
    const auto start = std::chrono::steady_clock::now();
    File file(checkpoint_path(policy_.disk_directory, digest), "rb", File::Stream::payload);
    if (!file.open()) return nullptr;
    CheckpointHeader header;
    file.read(&header, sizeof(header));
    if (header.magic != checkpoint_magic || get(header.identity) != identity || get(header.digest) != digest ||
        header.block_rows != block_rows || header.position != tokens.size() ||
        header.segment_count != header.position / block_rows || header.position % block_rows ||
        header.reserved > 1U || header.metadata_bytes > policy_.host_byte_budget)
        throw std::runtime_error("prefix cache checkpoint header does not match");
    std::vector<std::uint8_t> segment_digests(static_cast<std::size_t>(header.segment_count * 32));
    file.read(segment_digests.data(), segment_digests.size());
    // Inspect all payload sizes before the first pinned allocation. Conservative credits
    // also cover segments another reader may release between inspection and loading.
    std::uint64_t payload_bytes = header.fixed_bytes;
    if (!policy_.legacy) {
        for (std::uint64_t k = 0; k < header.segment_count; ++k) {
            ModelContentIdentity sd;
            std::memcpy(sd.bytes.data(), segment_digests.data() + k * 32, 32);
            File segment_file(segment_path(policy_.disk_directory, sd), "rb");
            if (!segment_file.open()) throw std::runtime_error("prefix cache segment file is missing");
            SegmentHeader sh; segment_file.read(&sh, sizeof(sh));
            if (sh.payload_bytes > policy_.host_byte_budget || payload_bytes > policy_.host_byte_budget - sh.payload_bytes) {
                std::lock_guard lock(mutex_); ++stats_.budget_skips; return nullptr;
            }
            payload_bytes += sh.payload_bytes;
        }
    }
    auto reservation = policy_.legacy ? Reservation{} : reserve(payload_bytes);
    if (!policy_.legacy && !reservation) return nullptr;
    auto checkpoint = std::make_shared<PrefixCacheCheckpoint>();
    checkpoint->identity = identity;
    checkpoint->digest = digest;
    checkpoint->position = header.position;
    checkpoint->block_rows = block_rows;
    checkpoint->metadata.resize(static_cast<std::size_t>(header.metadata_bytes));
    file.read(checkpoint->metadata.data(), header.metadata_bytes);
    if (prefix_cache_checksum(checkpoint->metadata) != header.metadata_checksum)
        throw std::runtime_error("prefix cache metadata checksum differs");
    if (header.reserved == 1U) {
        std::uint64_t count = 0;
        if (checkpoint->metadata.size() < sizeof(count)) throw std::runtime_error("prefix cache certificate is truncated");
        std::memcpy(&count, checkpoint->metadata.data(), sizeof(count));
        if (!count || count > (checkpoint->metadata.size()-sizeof(count))/sizeof(std::uint64_t))
            throw std::runtime_error("prefix cache certificate is malformed");
        checkpoint->producer_certificate.resize(static_cast<std::size_t>(count));
        std::memcpy(checkpoint->producer_certificate.data(), checkpoint->metadata.data()+sizeof(count), count*sizeof(std::uint64_t));
        checkpoint->metadata.erase(checkpoint->metadata.begin(),
            checkpoint->metadata.begin()+static_cast<std::ptrdiff_t>(sizeof(count)+count*sizeof(std::uint64_t)));
    }
    checkpoint->fixed = policy_.legacy ? allocate(header.fixed_bytes) : allocate(header.fixed_bytes, reservation);
    file.read(checkpoint->fixed.data.get(), header.fixed_bytes);
    if (prefix_cache_checksum(checkpoint->fixed.span()) != header.fixed_checksum)
        throw std::runtime_error("prefix cache fixed-state checksum differs");
    auto previous = identity;
    for (std::uint64_t k = 0; k < header.segment_count; ++k) {
        ModelContentIdentity segment_digest;
        std::memcpy(segment_digest.bytes.data(), segment_digests.data() + k * 32, 32);
        auto segment = load_segment(identity, segment_digest, block_rows, policy_.legacy ? nullptr : &reservation);
        // The chain must be exactly the requested tokens.
        if (segment->block_index != k || segment->parent != previous ||
            !std::equal(segment->tokens.begin(), segment->tokens.end(),
                tokens.begin() + static_cast<std::ptrdiff_t>(k * block_rows)))
            throw std::runtime_error("prefix cache checkpoint chain does not match the requested tokens");
        previous = segment_digest;
        checkpoint->segments.push_back(std::move(segment));
    }
    if (previous != digest) throw std::runtime_error("prefix cache checkpoint chain does not end at its digest");
    std::lock_guard lock(mutex_);
    ++stats_.disk_reads;
    stats_.disk_read_bytes += sizeof(header) + header.fixed_bytes + header.metadata_bytes;
    stats_.disk_read_ms += elapsed_ms(start);
    return checkpoint;
}

void PrefixStateCache::write_to_disk(const PrefixCacheCheckpoint& checkpoint) {
    const auto start = std::chrono::steady_clock::now();
    const auto& dir = policy_.disk_directory;
    std::uint64_t written = 0;
    std::error_code error;
    for (const auto& segment : checkpoint.segments) {
        const auto path = segment_path(dir, segment->digest);
        if (std::filesystem::exists(path, error)) {
            // This segment may have been published by an earlier non-durable writer.
            if(policy_.durable && !policy_.legacy) {
                File retained(path,"r+b");
                if(!retained.open()) throw std::runtime_error("prefix cache retained segment cannot be synced");
                retained.close(true);
            }
            continue;
        }
        SegmentHeader header;
        put(header.identity, segment->identity);
        put(header.digest, segment->digest);
        put(header.parent, segment->parent);
        header.block_index = segment->block_index;
        header.block_rows = segment->block_rows;
        header.payload_bytes = segment->payload.bytes;
        header.payload_checksum = prefix_cache_checksum(segment->payload.span());
        const auto temporary = temporary_path(path);
        {
            File file(temporary, "wb", File::Stream::payload);
            if (!file.open()) throw std::runtime_error("prefix cache segment file cannot be created");
            file.write(&header, sizeof(header));
            file.write(segment->tokens.data(), segment->tokens.size() * sizeof(std::uint32_t));
            file.write(segment->payload.data.get(), segment->payload.bytes);
            file.close(policy_.durable && !policy_.legacy);
        }
        std::filesystem::rename(temporary, path);
        written += sizeof(header) + segment->payload.bytes;
    }
    const auto path = checkpoint_path(dir, checkpoint.digest);
    // Every segment name is durable before publishing the checkpoint that references it.
    sync_directory(dir, policy_.durable && !policy_.legacy);
    if (!std::filesystem::exists(path, error)) {
        CheckpointHeader header;
        put(header.identity, checkpoint.identity);
        put(header.digest, checkpoint.digest);
        header.position = checkpoint.position;
        header.block_rows = checkpoint.block_rows;
        header.segment_count = checkpoint.segments.size();
        header.fixed_bytes = checkpoint.fixed.bytes;
        const auto metadata = disk_metadata(checkpoint);
        header.reserved = checkpoint.producer_certificate.empty() ? 0U : 1U;
        header.metadata_bytes = metadata.size();
        header.fixed_checksum = prefix_cache_checksum(checkpoint.fixed.span());
        header.metadata_checksum = prefix_cache_checksum(metadata);
        const auto temporary = temporary_path(path);
        {
            File file(temporary, "wb", File::Stream::payload);
            if (!file.open()) throw std::runtime_error("prefix cache checkpoint file cannot be created");
            file.write(&header, sizeof(header));
            for (const auto& segment : checkpoint.segments) file.write(segment->digest.bytes.data(), 32);
            file.write(metadata.data(), metadata.size());
            file.write(checkpoint.fixed.data.get(), checkpoint.fixed.bytes);
            file.close(policy_.durable && !policy_.legacy);
        }
        std::filesystem::rename(temporary, path);
        sync_directory(dir, policy_.durable && !policy_.legacy);
        written += sizeof(header) + checkpoint.fixed.bytes + metadata.size();
    } else if(policy_.durable && !policy_.legacy) {
        File retained(path,"r+b");
        if(!retained.open()) throw std::runtime_error("prefix cache retained checkpoint cannot be synced");
        retained.close(true);
        sync_directory(dir,true);
    }
    std::lock_guard lock(mutex_);
    ++stats_.disk_writes;
    stats_.disk_write_bytes += written;
    stats_.disk_write_ms += elapsed_ms(start);
}

void PrefixStateCache::update_disk_gauge() {
    std::uint64_t bytes=0;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(policy_.disk_directory,error)) {
        if (entry.is_regular_file(error)) bytes+=entry.file_size(error);
    }
    std::lock_guard lock(mutex_); stats_.disk_bytes=bytes;
}

void PrefixStateCache::enforce_disk_budget() {
    if (!policy_.disk_byte_budget) return;
    namespace fs = std::filesystem;
    struct Item { fs::path path; fs::file_time_type time; std::uint64_t bytes; };
    std::vector<Item> checkpoints;
    std::uint64_t total = 0;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(policy_.disk_directory, error)) {
        if (!entry.is_regular_file(error)) continue;
        const auto bytes = entry.file_size(error);
        total += bytes;
        if (entry.path().extension() == ".tmpc") checkpoints.push_back({entry.path(), entry.last_write_time(error), bytes});
    }
    if (total <= policy_.disk_byte_budget) return;
    std::sort(checkpoints.begin(), checkpoints.end(), [](const Item& a, const Item& b) { return a.time < b.time; });
    std::size_t removed = 0;
    // Approximate: a removed checkpoint frees its own file now and its unshared segments below.
    while (removed + 1 < checkpoints.size() && total > policy_.disk_byte_budget) {
        fs::remove(checkpoints[removed].path, error);
        total -= std::min(total, checkpoints[removed].bytes);
        ++removed;
        std::lock_guard lock(mutex_);
        ++stats_.disk_evictions;
    }
    if (!removed) return;
    // Delete segment files no remaining checkpoint references.
    std::unordered_set<std::string> referenced;
    for (std::size_t i = removed; i < checkpoints.size(); ++i) {
        File file(checkpoints[i].path, "rb");
        if (!file.open()) continue;
        CheckpointHeader header;
        try {
            file.read(&header, sizeof(header));
            if (header.magic != checkpoint_magic) continue;
            std::vector<std::uint8_t> digests(static_cast<std::size_t>(header.segment_count * 32));
            file.read(digests.data(), digests.size());
            for (std::uint64_t k = 0; k < header.segment_count; ++k) {
                ModelContentIdentity digest;
                std::memcpy(digest.bytes.data(), digests.data() + k * 32, 32);
                referenced.insert(segment_path(policy_.disk_directory, digest).filename().string());
            }
        } catch (const std::exception&) {
        }
    }
    for (const auto& entry : fs::directory_iterator(policy_.disk_directory, error))
        if (entry.path().extension() == ".tmps" && !referenced.count(entry.path().filename().string()))
            fs::remove(entry.path(), error);
}

void PrefixStateCache::sweep_disk(std::chrono::seconds min_age) {
    namespace fs = std::filesystem;
    std::error_code error;
    const auto now = fs::file_time_type::clock::now();
    const auto old = [&](const fs::directory_entry& entry) {
        const auto time = entry.last_write_time(error);
        return !error && now - time > min_age;
    };
    std::unordered_set<std::string> referenced;
    std::vector<fs::path> segments;
    for (const auto& entry : fs::directory_iterator(policy_.disk_directory, error)) {
        const auto name = entry.path().filename().string();
        if (name.find(".tmp.") != std::string::npos) {
            if (old(entry)) fs::remove(entry.path(), error);
            continue;
        }
        if (entry.path().extension() == ".tmps") { segments.push_back(entry.path()); continue; }
        if (entry.path().extension() != ".tmpc") continue;
        File file(entry.path(), "rb");
        if (!file.open()) continue;
        try {
            CheckpointHeader header;
            file.read(&header, sizeof(header));
            if (header.magic != checkpoint_magic || header.segment_count > (1ULL << 24)) continue;
            std::vector<std::uint8_t> digests(static_cast<std::size_t>(header.segment_count * 32));
            file.read(digests.data(), digests.size());
            for (std::uint64_t k = 0; k < header.segment_count; ++k) {
                ModelContentIdentity digest;
                std::memcpy(digest.bytes.data(), digests.data() + k * 32, 32);
                referenced.insert(segment_path(policy_.disk_directory, digest).filename().string());
            }
        } catch (const std::exception&) {
        }
    }
    for (const auto& path : segments)
        if (!referenced.count(path.filename().string()) && old(fs::directory_entry(path, error))) fs::remove(path, error);
}

void PrefixStateCache::writer_loop() {
    for (;;) {
        std::shared_ptr<const PrefixCacheCheckpoint> next;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) {
                writer_busy_ = false;
                idle_cv_.notify_all();
                return;
            }
            next = std::move(queue_.front());
            queue_.pop_front();
            writer_busy_ = true;
        }
        try {
            write_to_disk(*next);
            enforce_disk_budget();
            update_disk_gauge();
        } catch (const std::exception&) {
            std::lock_guard lock(mutex_);
            ++stats_.disk_write_failures;
        }
        {
            std::lock_guard lock(queue_mutex_);
            writer_busy_ = false;
        }
        idle_cv_.notify_all();
    }
}

void PrefixStateCache::flush() {
    if (!writer_.joinable()) return;
    std::unique_lock lock(queue_mutex_);
    idle_cv_.wait(lock, [&] { return queue_.empty() && !writer_busy_; });
}

bool PrefixStateCache::admit_capture(const PrefixCacheCheckpoint& checkpoint) {
    std::lock_guard lock(mutex_);
    const auto need = incremental_bytes_locked(checkpoint);
    if (need <= policy_.host_byte_budget) {
        evict_to_budget_locked(need, policy_.retention == PrefixCacheRetention::fair ? &checkpoint : nullptr);
        const auto remaining = incremental_bytes_locked(checkpoint);
        const bool directory_fits = policy_.retention == PrefixCacheRetention::lru || entries_.size() < 256;
        if (directory_fits && remaining <= policy_.host_byte_budget &&
            retained_bytes_locked() <= policy_.host_byte_budget - remaining) return true;
    }
    ++stats_.budget_skips;
    return false;
}
void PrefixStateCache::record_capture_failure() {
    std::lock_guard lock(mutex_);
    ++stats_.capture_failures;
}

void PrefixStateCache::record_no_store_skip() {
    std::lock_guard lock(mutex_); ++stats_.no_store_skips;
}
PrefixStateCacheStats PrefixStateCache::counters() {
    std::lock_guard lock(mutex_);
    auto out = stats_;
    for(std::size_t i=0;i<certificate_declines_.size();++i)
        out.certificate_declines[i]=certificate_declines_[i].load(std::memory_order_relaxed);
    if (policy_.legacy) {
        out.host_bytes = host_bytes_locked(); out.host_segments = segments_.size();
    } else {
        std::lock_guard ledger_lock(ledger_->mutex);
        // Count the retained slab (including free ranges) and its reserved metadata once.
        out.host_bytes = payload_pool_ ? policy_.host_byte_budget : ledger_->bytes + metadata_bytes_;
        if(payload_pool_) out.pool_free_bytes=payload_pool_->free_bytes();
        out.host_segments = ledger_->segments;
    }
    out.host_checkpoints = entries_.size();
    return out;
}
PrefixStateCacheStats PrefixStateCache::stats() {
    if (!policy_.legacy) return counters();
    std::lock_guard lock(mutex_);
    auto out = stats_;
    for(std::size_t i=0;i<certificate_declines_.size();++i)
        out.certificate_declines[i]=certificate_declines_[i].load(std::memory_order_relaxed);
    out.host_bytes = host_bytes_locked();
    out.host_checkpoints = entries_.size();
    out.host_segments = segments_.size();
    return out;
}


bool PrefixStateCache::admit_session_working_bytes(std::uint64_t bytes) {
    std::lock_guard lock(mutex_);
    if (bytes > policy_.host_byte_budget) return false;
    evict_to_budget_locked(bytes);
    return retained_bytes_locked() <= policy_.host_byte_budget - bytes;
}
bool PrefixStateCache::park_session(const std::string& id, PrefixCacheBuffer envelope, std::uint64_t max_sessions) {
    require(!id.empty() && id.size()<=128 && std::all_of(id.begin(),id.end(),[](unsigned char c) {
        return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='-' || c=='_';
    }),"invalid session id");
    require(envelope.data && envelope.bytes && max_sessions,"session parking requires a budget and envelope");
    {
        std::lock_guard receipt_lock(mutex_);
        auto r = receipts_[envelope.data.get()].lock();
        if (!r) {
            r=std::make_shared<Receipt>(); r->ledger=ledger_; r->payload=envelope.data; r->bytes=envelope.bytes;
            receipts_[envelope.data.get()]=r;
        }
        envelope.data=std::shared_ptr<std::byte>(r,envelope.data.get());
    }
    const auto digest=sha256_bytes_fast(envelope.span());
    auto checkpoint=std::make_shared<PrefixCacheCheckpoint>();
    checkpoint->conversation=id;
    checkpoint->digest=digest; checkpoint->block_rows=1; checkpoint->fixed=std::move(envelope);
    std::lock_guard lock(mutex_);
    named_session_parking_ = true;
    for(auto it=named_sessions_.begin();it!=named_sessions_.end();) {
        if(!entries_.count(it->second)) it=named_sessions_.erase(it); else ++it;
    }
    const auto need=checkpoint_host_bytes(*checkpoint);
    if(need>policy_.host_byte_budget) { ++stats_.rejected_over_budget; return false; }
    if(auto old=named_sessions_.find(id);old!=named_sessions_.end()) {
        if(auto e=entries_.find(old->second);e!=entries_.end()) { erase_locked(e->first); }
        named_sessions_.erase(old);
    }
    // Different names must not alias the same mutable lifecycle. Bind name into storage key.
    std::vector<std::byte> key;
    for(const auto c:id) key.push_back(static_cast<std::byte>(c));
    for(auto c:digest.bytes) key.push_back(static_cast<std::byte>(c));
    checkpoint->digest=sha256_bytes(key);
    while(named_sessions_.size()>=max_sessions) {
        auto victim=lru_.begin();
        while(victim!=lru_.end() && std::none_of(named_sessions_.begin(),named_sessions_.end(),
            [&](const auto& n){return n.second==*victim;})) ++victim;
        require(victim!=lru_.end(),"session LRU invariant failed");
        const auto d=*victim; erase_locked(d);
        for(auto it=named_sessions_.begin();it!=named_sessions_.end();) {
            if(it->second==d) it=named_sessions_.erase(it); else ++it;
        }
        ++stats_.evicted;
    }
    evict_to_budget_locked(need);
    for(auto it=named_sessions_.begin();it!=named_sessions_.end();) {
        if(!entries_.count(it->second)) it=named_sessions_.erase(it); else ++it;
    }
    // Outstanding handles can retain pinned segments: do not claim admission if eviction did not free enough.
    if(retained_bytes_locked()>policy_.host_byte_budget-need) { ++stats_.rejected_over_budget; return false; }
    add_locked(checkpoint);
    session_buffers_[checkpoint->fixed.data.get()] = {checkpoint->fixed.data, checkpoint->fixed.bytes};
    named_sessions_[id]=checkpoint->digest; ++stats_.inserted; return true;
}
PrefixCacheBuffer PrefixStateCache::resume_session(const std::string& id) {
    std::lock_guard lock(mutex_);
    const auto n=named_sessions_.find(id); if(n==named_sessions_.end()) return {};
    const auto e=entries_.find(n->second); if(e==entries_.end()) { named_sessions_.erase(n); return {}; }
    touch_locked(e->second,true); return e->second.checkpoint->fixed;
}
std::vector<std::string> PrefixStateCache::list_sessions() {
    std::lock_guard lock(mutex_); std::vector<std::string> out;
    for(auto it=named_sessions_.begin();it!=named_sessions_.end();) {
        if(!entries_.count(it->second)) it=named_sessions_.erase(it);
        else { out.push_back(it->first); ++it; }
    }
    return out;
}
bool PrefixStateCache::delete_session(const std::string& id) {
    std::lock_guard lock(mutex_); const auto n=named_sessions_.find(id); if(n==named_sessions_.end()) return false;
    if(auto e=entries_.find(n->second);e!=entries_.end()) { erase_locked(e->first); }
    named_sessions_.erase(n); ++stats_.evictions_explicit; return true;
}

}  // namespace strix

namespace strix {
void PrefixStateCache::record_staging(std::uint64_t cap,std::uint64_t bytes,bool active) {
    std::lock_guard lock(mutex_);
    if(cap) stats_.staging_budget_bytes=cap;
    stats_.staging_bytes=active?bytes:0;
    stats_.staging_peak_bytes=std::max(stats_.staging_peak_bytes,bytes);
}
void PrefixStateCache::record_staging_skip(bool cap) {
    std::lock_guard lock(mutex_);
    ++stats_.budget_skips;
    if(cap) ++stats_.staging_cap_skips; else ++stats_.capture_busy_skips;
}
}

namespace strix {
void PrefixStateCache::record_row_prefetch() {std::lock_guard lock(mutex_); ++stats_.row_prefetches;}
void PrefixStateCache::record_certificate_decline(PrefixCertificateDecline reason) noexcept {
    certificate_declines_[static_cast<std::size_t>(reason)].fetch_add(1,std::memory_order_relaxed);
}
void PrefixStateCache::record_snapshot_preparation() {std::lock_guard lock(mutex_); ++stats_.snapshot_preparations;}
void PrefixStateCache::record_tripwire_skip() {std::lock_guard lock(mutex_); ++stats_.capture_skips_tripwire;}
void PrefixStateCache::record_batch_config_skip() {std::lock_guard lock(mutex_); ++stats_.capture_skips_batch_config;}
void PrefixStateCache::record_cross_stream_skip() {std::lock_guard lock(mutex_); ++stats_.capture_skips_cross_stream;}
}
