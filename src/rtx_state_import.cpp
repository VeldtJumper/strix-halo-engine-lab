#include "strix/rtx_state_import.hpp"

#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#define STRIX_RTX_GETPID _getpid
#else
#include <unistd.h>
#define STRIX_RTX_GETPID getpid
#endif

namespace strix {
namespace {

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("rtx state size overflows");
    return a + b;
}
std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error("rtx state size overflows");
    return a * b;
}

void json_string(std::ostringstream& out, std::string_view text) {
    out << '"';
    for (const char ch : text) {
        const auto c = static_cast<unsigned char>(ch);
        if (ch == '"' || ch == '\\') out << '\\' << ch;
        else if (c < 0x20U) {
            static const char* digits = "0123456789abcdef";
            out << "\\u00" << digits[c >> 4U] << digits[c & 15U];
        } else out << ch;
    }
    out << '"';
}

}  // namespace

std::string rtx_hex_u64(std::uint64_t value) {
    static const char* digits = "0123456789abcdef";
    std::string out = "0x";
    for (int shift = 60; shift >= 0; shift -= 4) out.push_back(digits[(value >> static_cast<unsigned>(shift)) & 15U]);
    return out;
}

std::uint64_t rtx_parse_u64(std::string_view text) {
    if (text.empty()) throw std::invalid_argument("empty integer");
    std::uint64_t value = 0;
    const bool hex = text.size() > 2U && text[0] == '0' && (text[1] == 'x' || text[1] == 'X');
    const auto digits = hex ? text.substr(2) : text;
    if (digits.empty() || digits.size() > (hex ? 16U : 20U)) throw std::invalid_argument("integer out of range: " + std::string(text));
    for (const char ch : digits) {
        unsigned d = 0;
        if (ch >= '0' && ch <= '9') d = static_cast<unsigned>(ch - '0');
        else if (hex && ch >= 'a' && ch <= 'f') d = static_cast<unsigned>(ch - 'a') + 10U;
        else if (hex && ch >= 'A' && ch <= 'F') d = static_cast<unsigned>(ch - 'A') + 10U;
        else throw std::invalid_argument("not an integer: " + std::string(text));
        const std::uint64_t base = hex ? 16U : 10U;
        if (value > (UINT64_MAX - d) / base) throw std::invalid_argument("integer out of range: " + std::string(text));
        value = value * base + d;
    }
    return value;
}

ModelContentIdentity rtx_parse_sha256(std::string_view hex) {
    if (hex.size() != 64U) throw std::invalid_argument("sha256 must be 64 hex digits");
    ModelContentIdentity out;
    for (std::size_t i = 0; i < 32U; ++i) {
        unsigned byte = 0;
        for (std::size_t k = 0; k < 2U; ++k) {
            const char ch = hex[i * 2U + k];
            unsigned d = 0;
            if (ch >= '0' && ch <= '9') d = static_cast<unsigned>(ch - '0');
            else if (ch >= 'a' && ch <= 'f') d = static_cast<unsigned>(ch - 'a') + 10U;
            else if (ch >= 'A' && ch <= 'F') d = static_cast<unsigned>(ch - 'A') + 10U;
            else throw std::invalid_argument("sha256 must be hex");
            byte = (byte << 4U) | d;
        }
        out.bytes[i] = static_cast<std::uint8_t>(byte);
    }
    return out;
}

// ---- layout ----

std::uint64_t RtxStateLayout::component_rows(std::size_t component, std::uint64_t position) const {
    const auto shift = rows.at(component).shift;
    return position > shift ? position - shift : 0U;
}

std::uint64_t RtxStateLayout::component_offset(std::size_t component, std::uint64_t position) const {
    std::uint64_t offset = 0;
    for (std::size_t c = 0; c < component; ++c) offset = checked_add(offset, checked_mul(rows.at(c).bytes_per_row, component_rows(c, position)));
    return offset;
}

std::uint64_t RtxStateLayout::rows_bytes(std::uint64_t position) const { return component_offset(rows.size(), position); }

std::uint64_t RtxStateLayout::payload_bytes(std::uint64_t position) const { return checked_add(rows_bytes(position), fixed_bytes); }

std::optional<std::size_t> RtxStateLayout::find_row(std::string_view id) const {
    for (std::size_t c = 0; c < rows.size(); ++c) if (rows[c].id == id) return c;
    return std::nullopt;
}

std::optional<std::size_t> RtxStateLayout::find_fixed(std::string_view id) const {
    for (std::size_t f = 0; f < fixed.size(); ++f) if (fixed[f].id == id) return f;
    return std::nullopt;
}

void RtxStateLayout::validate() const {
    if (rows.empty() || !layout_identity || !context_tokens) throw std::invalid_argument("rtx state layout is incomplete");
    if (rotary.descriptor().mode() == RotaryMode::yarn && context_tokens > rotary.descriptor().effective_max())
        throw std::invalid_argument("rtx state context capacity exceeds rotary table extent");
    std::uint64_t sum = 0;
    for (const auto& row : rows) {
        if (row.id.empty() || !row.bytes_per_row || row.shift > 1U) throw std::invalid_argument("rtx state row component is invalid");
        sum = checked_add(sum, row.bytes_per_row);
    }
    if (sum != row_bytes) throw std::invalid_argument("rtx state row bytes do not add up");
    std::uint64_t at = 0;
    for (const auto& f : fixed) {
        if (f.id.empty() || !f.bytes || f.offset != at || f.offset % 256U) throw std::invalid_argument("rtx state fixed component is misplaced");
        at = checked_add(at, (f.bytes + 255U) & ~std::uint64_t{255});
    }
    if (at != fixed_bytes) throw std::invalid_argument("rtx state fixed bytes do not add up");
    std::vector<std::string> ids;
    for (const auto& row : rows) ids.push_back(row.id);
    for (const auto& f : fixed) ids.push_back(f.id);
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) throw std::invalid_argument("rtx state component ids repeat");
}

std::uint64_t rtx_state_geometry_identity(const RtxStateLayout& layout) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    const auto byte = [&](std::uint8_t b) { h ^= b; h *= 0x100000001b3ULL; };
    const auto u64 = [&](std::uint64_t v) { for (unsigned s = 0; s < 64U; s += 8U) byte(static_cast<std::uint8_t>((v >> s) & 255U)); };
    const auto text = [&](const std::string& s) { u64(s.size()); for (const char ch : s) byte(static_cast<std::uint8_t>(ch)); };
    u64(0x5254585354415431ULL);  // "RTXSTAT1"
    u64(layout.layout_identity); u64(layout.row_bytes); u64(layout.fixed_bytes); u64(layout.rows.size()); u64(layout.fixed.size());
    for (const auto& row : layout.rows) { text(row.id); u64(row.bytes_per_row); u64(row.shift); }
    for (const auto& f : layout.fixed) { text(f.id); u64(f.bytes); u64(f.offset); }
    if (layout.rotary.descriptor().mode() == RotaryMode::yarn) {
        text("RTXSTAT1-YARN-CONTEXT");
        u64(layout.context_tokens);
        h = append_rotary_identity(h, layout.rotary);
    }
    return h;
}

// ---- staging ----

RtxStateStaging::RtxStateStaging(std::shared_ptr<const RtxStateLayout> layout, RtxStateBeginRequest request, std::string id,
    const PrefixCacheAllocator& allocate)
    : layout_(std::move(layout)), request_(std::move(request)), id_(std::move(id)) {
    if (!layout_) throw std::invalid_argument("rtx state import has no layout");
    const auto& l = *layout_;
    if (request_.layout_identity != l.layout_identity)
        throw std::invalid_argument("rtx state layout identity differs from this engine's (" + rtx_hex_u64(l.layout_identity) + ")");
    if (request_.rotary && *request_.rotary != l.rotary)
        throw std::invalid_argument("rtx state rotary identity mismatch (mode, factor or table SHA256)");
    if (request_.context_tokens && *request_.context_tokens != l.context_tokens)
        throw std::invalid_argument("rtx state context capacity mismatch");
    if (l.rotary.descriptor().mode() == RotaryMode::yarn && !request_.geometry_identity &&
        !(request_.rotary && request_.context_tokens))
        throw std::invalid_argument("rtx state YaRN import requires geometry_identity or rotary and context_tokens");
    if (request_.geometry_identity && request_.geometry_identity != rtx_state_geometry_identity(l))
        throw std::invalid_argument("rtx state geometry identity differs from this engine's");
    if (!request_.position || request_.position >= l.context_tokens || request_.tokens.size() != request_.position)
        throw std::invalid_argument("rtx state needs 1 <= position < context and exactly `position` tokens");
    if (!request_.producer.numerical || !request_.producer.reduction || !request_.producer.state_update)
        throw std::invalid_argument("rtx state producer identities must be nonzero");
    if (request_.local_rows >= request_.position) throw std::invalid_argument("rtx state local_rows must be below position");
    if (!request_.encoding.empty() &&
        std::find(rtx_chunk_codecs.begin(), rtx_chunk_codecs.end(), std::string_view(request_.encoding)) == rtx_chunk_codecs.end())
        throw std::invalid_argument("rtx state chunk encoding " + request_.encoding + " is not supported (layout lists `codecs`)");
    bytes_ = l.payload_bytes(request_.position);
    if (bytes_ > SIZE_MAX) throw std::invalid_argument("rtx state payload does not fit in memory");
    buffer_ = allocate(bytes_);
    if (!buffer_.data || buffer_.bytes < bytes_) throw std::runtime_error("rtx state staging allocation failed");
    // Fixed-blob padding is zero in the canonical payload and is never sent.
    const auto rows_end = l.rows_bytes(request_.position);
    for (std::size_t f = 0; f < l.fixed.size(); ++f) {
        const auto end = l.fixed[f].offset + l.fixed[f].bytes;
        const auto next = f + 1U < l.fixed.size() ? l.fixed[f + 1U].offset : l.fixed_bytes;
        if (next > end) {
            std::memset(buffer_.data.get() + rows_end + end, 0, static_cast<std::size_t>(next - end));
            reserve(rows_end + end, next - end);
        }
    }
    // Cooperative: the node's own rows are not sent; zero in the staging (the restore takes them from the checkpoint).
    if (request_.local_rows)
        for (std::size_t c = 0; c < l.rows.size(); ++c) {
            const auto n = checked_mul(l.rows[c].bytes_per_row, l.component_rows(c, request_.local_rows));
            if (!n) continue;
            const auto at = l.component_offset(c, request_.position);
            std::memset(buffer_.data.get() + at, 0, static_cast<std::size_t>(n));
            reserve(at, n);
        }
}

void RtxStateStaging::require_open() const {
    if (committed_) throw std::logic_error("rtx state import is already committed");
}

void RtxStateStaging::check_free(std::uint64_t offset, std::uint64_t n) const {
    // Caller holds mutex_: [offset, offset + n) is disjoint from every written range.
    const auto end = offset + n;
    auto next = covered_.lower_bound(offset);
    if (next != covered_.end() && next->first < end) throw std::invalid_argument("rtx state chunk overlaps bytes already written");
    if (next != covered_.begin() && std::prev(next)->second > offset)
        throw std::invalid_argument("rtx state chunk overlaps bytes already written");
}

void RtxStateStaging::reserve(std::uint64_t offset, std::uint64_t n) {
    // Caller holds mutex_ and checked the range with check_free.
    const auto end = offset + n;
    std::uint64_t begin_at = offset, end_at = end;
    auto next = covered_.lower_bound(offset);
    if (next != covered_.begin()) {
        const auto previous = std::prev(next);
        if (previous->second == offset) { begin_at = previous->first; covered_.erase(previous); }
    }
    next = covered_.lower_bound(end);
    if (next != covered_.end() && next->first == end) { end_at = next->second; covered_.erase(next); }
    covered_[begin_at] = end_at;
    covered_bytes_ += n;
}

RtxStateSegment RtxStateStaging::rows_segment(std::size_t component, std::uint64_t first_row, std::span<const std::byte> data) const {
    const auto& l = *layout_;
    if (component >= l.rows.size()) throw std::out_of_range("rtx state row component is unknown");
    const auto bpr = l.rows[component].bytes_per_row;
    if (data.empty() || data.size() % bpr) throw std::invalid_argument("rtx state rows chunk is not a positive whole number of rows");
    const std::uint64_t count = data.size() / bpr;
    const auto rows = l.component_rows(component, request_.position);
    if (first_row > rows || count > rows - first_row) throw std::out_of_range("rtx state rows chunk is out of bounds");
    return {l.component_offset(component, request_.position) + first_row * bpr, data};
}

RtxStateSegment RtxStateStaging::fixed_segment(std::size_t component, std::uint64_t offset, std::span<const std::byte> data) const {
    const auto& l = *layout_;
    if (component >= l.fixed.size()) throw std::out_of_range("rtx state fixed component is unknown");
    const auto& f = l.fixed[component];
    if (data.empty() || offset > f.bytes || data.size() > f.bytes - offset) throw std::out_of_range("rtx state fixed chunk is out of bounds");
    return {l.rows_bytes(request_.position) + f.offset + offset, data};
}

void RtxStateStaging::write_chunk(std::span<const RtxStateSegment> segments, const RtxStateChunkRecord& record) {
    if (segments.empty()) throw std::invalid_argument("rtx state chunk has no segment");
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& segment : segments) {
        const std::uint64_t n = segment.data.size();
        if (!n || segment.canonical_offset > bytes_ || n > bytes_ - segment.canonical_offset)
            throw std::out_of_range("rtx state chunk is out of bounds");
        ranges.emplace_back(segment.canonical_offset, segment.canonical_offset + n);
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i)
        if (ranges[i].first < ranges[i - 1U].second) throw std::invalid_argument("rtx state chunk segments overlap each other");
    {
        const std::lock_guard lock(mutex_);
        require_open();
        const int mode = record.seq ? 1 : 0;
        if (seq_mode_ >= 0 && seq_mode_ != mode) throw std::invalid_argument("rtx state chunks mix X-Chunk-Seq and receive order");
        const auto seq = record.seq ? *record.seq : static_cast<std::uint64_t>(chunk_digests_.size());
        if (chunk_digests_.count(seq)) throw std::invalid_argument("rtx state chunk sequence number repeats");
        for (const auto& [begin, end] : ranges) check_free(begin, end - begin);
        for (const auto& [begin, end] : ranges) reserve(begin, end - begin);
        seq_mode_ = mode;
        chunk_digests_.emplace(seq, record.digest);
        for (const auto& segment : segments) chunk_bytes_ += segment.data.size();
        wire_bytes_ += record.wire_bytes;
        ++writing_;
    }
    // The ranges are reserved: the copies run without the lock (chunks may arrive on parallel connections);
    // commit waits for no copy in flight.
    for (const auto& segment : segments)
        std::memcpy(buffer_.data.get() + segment.canonical_offset, segment.data.data(), segment.data.size());
    const std::lock_guard lock(mutex_);
    --writing_;
}

RtxStateCommitResult RtxStateStaging::commit(const RtxStateCommitRequest& request) {
    const auto start = std::chrono::steady_clock::now();
    RtxStateCommitResult result;
    {
        const std::lock_guard lock(mutex_);
        require_open();
        if (writing_) throw std::logic_error("rtx state import still has chunks being copied");
        if (covered_bytes_ != bytes_ || covered_.size() != 1U || covered_.begin()->first != 0U || covered_.begin()->second != bytes_)
            throw std::invalid_argument("rtx state is incomplete: " + std::to_string(covered_bytes_) + " of " + std::to_string(bytes_) + " bytes written");
        if (request.chunks != chunk_digests_.size() || request.bytes != chunk_bytes_)
            throw std::invalid_argument("rtx state commit counts differ: received " + std::to_string(chunk_digests_.size()) + " chunks, " +
                std::to_string(chunk_bytes_) + " bytes");
        if (!chunk_digests_.empty() && (chunk_digests_.begin()->first != 0U || chunk_digests_.rbegin()->first + 1U != chunk_digests_.size()))
            throw std::invalid_argument("rtx state chunk sequence numbers are not 0..N-1");
        std::vector<ModelContentIdentity> digests;
        digests.reserve(chunk_digests_.size());
        for (const auto& [seq, digest] : chunk_digests_) digests.push_back(digest);
        const auto manifest = rtx_manifest_digest(digests);
        if (manifest != request.manifest)
            throw std::invalid_argument("rtx state manifest sha256 differs: received " + model_content_identity_hex(manifest));
        manifest_ = manifest;
        committed_ = true;
        commit_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        result.verify_ms = commit_ms_;
    }
    if (request.canonical_sha256) {
        const auto canonical_start = std::chrono::steady_clock::now();
        result.canonical = canonical_sha256();
        result.canonical_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - canonical_start).count();
    }
    return result;
}

ModelContentIdentity RtxStateStaging::canonical_sha256() const {
    const auto all = payload();
    Sha256Fast sha;
    constexpr std::size_t step = 64U << 20U;
    for (std::size_t at = 0; at < all.size(); at += step) sha.update(all.subspan(at, std::min(step, all.size() - at)));
    return sha.finalize();
}

std::uint64_t RtxStateStaging::covered_bytes() const { const std::lock_guard lock(mutex_); return covered_bytes_; }
std::uint64_t RtxStateStaging::chunks() const { const std::lock_guard lock(mutex_); return chunk_digests_.size(); }
bool RtxStateStaging::committed() const { const std::lock_guard lock(mutex_); return committed_; }
ModelContentIdentity RtxStateStaging::manifest_sha256() const { const std::lock_guard lock(mutex_); return manifest_; }
std::uint64_t RtxStateStaging::wire_bytes() const { const std::lock_guard lock(mutex_); return wire_bytes_; }

std::span<const std::byte> RtxStateStaging::payload() const {
    if (!committed()) throw std::logic_error("rtx state is not committed");
    return {buffer_.data.get(), static_cast<std::size_t>(bytes_)};
}
std::span<const std::byte> RtxStateStaging::component(std::size_t c) const {
    const auto all = payload();
    const auto& l = *layout_;
    const auto offset = l.component_offset(c, request_.position);
    return all.subspan(static_cast<std::size_t>(offset), static_cast<std::size_t>(l.rows.at(c).bytes_per_row * l.component_rows(c, request_.position)));
}
std::span<const std::byte> RtxStateStaging::fixed_blob() const {
    const auto all = payload();
    return all.subspan(static_cast<std::size_t>(layout_->rows_bytes(request_.position)), static_cast<std::size_t>(layout_->fixed_bytes));
}

std::vector<RtxStateSegment> parse_rtx_chunk_segments(const RtxStateStaging& staging, std::span<const std::byte> body) {
    const auto u32 = [&](std::size_t at) {
        std::uint32_t v = 0;
        for (unsigned b = 0; b < 4U; ++b) v |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(body[at + b])) << (8U * b);
        return v;
    };
    std::vector<RtxStateSegment> out;
    std::size_t at = 0;
    while (at < body.size()) {
        if (body.size() - at < 24U) throw std::invalid_argument("rtx state chunk segment header is truncated");
        const auto kind = u32(at), component = u32(at + 4U), a = u32(at + 8U), b = u32(at + 12U);
        const std::uint64_t n = static_cast<std::uint64_t>(u32(at + 16U)) | (static_cast<std::uint64_t>(u32(at + 20U)) << 32U);
        at += 24U;
        if (n > body.size() - at) throw std::invalid_argument("rtx state chunk segment is truncated");
        const auto data = body.subspan(at, static_cast<std::size_t>(n));
        at += static_cast<std::size_t>(n);
        if (kind == 0U) {
            const auto& l = staging.layout();
            if (component >= l.rows.size()) throw std::out_of_range("rtx state row component is unknown");
            if (n != static_cast<std::uint64_t>(b) * l.rows[component].bytes_per_row)
                throw std::invalid_argument("rtx state rows segment byte count differs from rows * bytes_per_row");
            out.push_back(staging.rows_segment(component, a, data));
        } else if (kind == 1U) {
            out.push_back(staging.fixed_segment(component, static_cast<std::uint64_t>(a) | (static_cast<std::uint64_t>(b) << 32U), data));
        } else throw std::invalid_argument("rtx state chunk segment kind is unknown");
    }
    if (out.empty()) throw std::invalid_argument("rtx state chunk has no segment");
    return out;
}

ModelContentIdentity rtx_chunk_digest(std::span<const std::byte> body) { return sha256_bytes_fast(body); }

ModelContentIdentity rtx_manifest_digest(std::span<const ModelContentIdentity> chunk_digests) {
    std::vector<std::byte> frame;
    frame.reserve(chunk_digests.size() * 32U);
    for (const auto& d : chunk_digests)
        for (const auto byte : d.bytes) frame.push_back(static_cast<std::byte>(byte));
    return sha256_bytes_fast(frame);
}

// ---- registry ----

RtxStateImportRegistry::RtxStateImportRegistry(PrefixCacheAllocator allocate, std::size_t max_in_flight, unsigned decode_threads)
    : allocate_(std::move(allocate)), max_in_flight_(max_in_flight), decode_threads_(decode_threads) {
    if (!allocate_ || !max_in_flight_) throw std::invalid_argument("rtx state registry needs an allocator and a positive in-flight limit");
}

void RtxStateImportRegistry::set_layout(RtxStateLayout layout) {
    layout.validate();
    const std::lock_guard lock(mutex_);
    if (layout_) throw std::logic_error("rtx state layout is published once");
    layout_ = std::make_shared<const RtxStateLayout>(std::move(layout));
}

std::shared_ptr<const RtxStateLayout> RtxStateImportRegistry::layout() const { const std::lock_guard lock(mutex_); return layout_; }

std::string RtxStateImportRegistry::begin(RtxStateBeginRequest request) {
    std::shared_ptr<const RtxStateLayout> layout;
    std::string id;
    {
        const std::lock_guard lock(mutex_);
        if (!layout_) throw std::logic_error("rtx state import is not ready (the engine has not published its layout yet)");
        if (in_flight_.size() >= max_in_flight_) throw std::length_error("rtx state import: too many imports in flight; commit or abort one");
        layout = layout_;
        static thread_local std::mt19937_64 rng{std::random_device{}()};
        id = "rtx-" + std::to_string(++counter_) + "-" + rtx_hex_u64(rng()).substr(10);
        in_flight_.emplace(id, nullptr);  // reserve the slot during the allocation
    }
    try {
        auto staging = std::make_shared<RtxStateStaging>(layout, std::move(request), id, allocate_);
        const std::lock_guard lock(mutex_);
        in_flight_[id] = std::move(staging);
        ++stats_.begun;
    } catch (...) {
        const std::lock_guard lock(mutex_);
        in_flight_.erase(id);
        throw;
    }
    return id;
}

std::shared_ptr<RtxStateStaging> RtxStateImportRegistry::find(std::string_view id) const {
    const std::lock_guard lock(mutex_);
    const auto found = in_flight_.find(id);
    return found == in_flight_.end() ? nullptr : found->second;
}

RtxStateCommitResult RtxStateImportRegistry::commit(std::string_view id, const RtxStateCommitRequest& request) {
    auto staging = find(id);
    if (!staging) throw std::out_of_range("unknown or finished rtx import id " + std::string(id));
    RtxStateCommitResult result;
    try {
        result = staging->commit(request);
    } catch (...) {
        const std::lock_guard lock(mutex_);
        ++stats_.failed_commits;
        throw;
    }
    const std::lock_guard lock(mutex_);
    in_flight_.erase(std::string(id));
    parked_ = std::move(staging);
    ++stats_.committed;
    return result;
}

bool RtxStateImportRegistry::abort(std::string_view id) {
    const std::lock_guard lock(mutex_);
    if (const auto found = in_flight_.find(id); found != in_flight_.end() && found->second) {
        in_flight_.erase(found);
        ++stats_.aborted;
        return true;
    }
    if (parked_ && parked_->id() == id) { parked_.reset(); ++stats_.aborted; return true; }
    return false;
}

std::shared_ptr<const RtxStateStaging> RtxStateImportRegistry::take(std::span<const std::uint32_t> prompt) {
    const std::lock_guard lock(mutex_);
    if (!parked_) return nullptr;
    const auto& tokens = parked_->request().tokens;
    if (prompt.size() < 2U || parked_->position() > prompt.size() - 1U ||
        !std::equal(tokens.begin(), tokens.end(), prompt.begin())) {
        ++stats_.misses;
        stats_.last_miss = prompt.size() < parked_->position() + 1U ? "the prompt is not longer than the parked state"
                                                             : "the prompt does not start with the parked state's tokens";
        return nullptr;
    }
    ++stats_.taken;
    std::shared_ptr<const RtxStateStaging> out = parked_;
    if (!parked_->request().keep) parked_.reset();
    return out;
}

std::vector<RtxStateStatusEntry> RtxStateImportRegistry::status() const {
    const std::lock_guard lock(mutex_);
    std::vector<RtxStateStatusEntry> out;
    for (const auto& [id, staging] : in_flight_) {
        if (!staging) continue;
        out.push_back({id, staging->position(), staging->payload_bytes(), staging->covered_bytes(), false, false, staging->request().keep});
    }
    if (parked_) out.push_back({parked_->id(), parked_->position(), parked_->payload_bytes(), parked_->payload_bytes(), true, true, parked_->request().keep});
    return out;
}

RtxStateRegistryStats RtxStateImportRegistry::stats() const { const std::lock_guard lock(mutex_); return stats_; }

bp_codec::WorkerPool& RtxStateImportRegistry::decode_pool() {
    std::call_once(decode_pool_once_, [this] { decode_pool_ = std::make_unique<bp_codec::WorkerPool>(decode_threads_); });
    return *decode_pool_;
}

// ---- file ----

ModelContentIdentity write_rtx_state_file(const std::filesystem::path& path, const RtxStateLayout& layout,
    std::span<const std::uint32_t> tokens, std::uint64_t position, const RtxStateProducer& producer,
    const RtxStateFileSession& session, std::span<const std::byte> payload) {
    layout.validate();
    if (tokens.size() != position || payload.size() != layout.payload_bytes(position))
        throw std::invalid_argument("rtx state file: tokens or payload size differ from the position");
    Sha256Fast hasher;
    constexpr std::size_t step = 64U << 20U;
    for (std::size_t at = 0; at < payload.size(); at += step) hasher.update(payload.subspan(at, std::min(step, payload.size() - at)));
    const auto sha = hasher.finalize();
    std::ostringstream h;
    h << "{\"format\":"; json_string(h, rtx_state_format);
    h << ",\"position\":" << position << ",\"layout_identity\":\"" << rtx_hex_u64(layout.layout_identity)
      << "\",\"geometry_identity\":\"" << rtx_hex_u64(rtx_state_geometry_identity(layout))
      << "\",\"row_bytes\":" << layout.row_bytes << ",\"fixed_bytes\":" << layout.fixed_bytes
      << ",\"payload_bytes\":" << payload.size() << ",\"sha256\":\"" << model_content_identity_hex(sha)
      << "\",\"producer\":{\"numerical\":\"" << rtx_hex_u64(producer.numerical) << "\",\"reduction\":\"" << rtx_hex_u64(producer.reduction)
      << "\",\"state_update\":\"" << rtx_hex_u64(producer.state_update) << "\"},\"model_content\":\"" << layout.model_content_hex
      << "\",\"identity_extension\":\"" << layout.identity_extension_hex << "\",\"rows\":[";
    for (std::size_t c = 0; c < layout.rows.size(); ++c) {
        h << (c ? "," : "") << "{\"id\":"; json_string(h, layout.rows[c].id);
        h << ",\"bytes_per_row\":" << layout.rows[c].bytes_per_row << ",\"shift\":" << layout.rows[c].shift << '}';
    }
    h << "],\"fixed\":[";
    for (std::size_t f = 0; f < layout.fixed.size(); ++f) {
        h << (f ? "," : "") << "{\"id\":"; json_string(h, layout.fixed[f].id);
        h << ",\"bytes\":" << layout.fixed[f].bytes << ",\"offset\":" << layout.fixed[f].offset << '}';
    }
    h << "],\"context_tokens\":" << layout.context_tokens << ",\"rotary\":"
      << (layout.rotary.descriptor().mode() == RotaryMode::yarn ? layout.rotary.json() : layout.rotary.descriptor().json());
    h << ",\"session\":{\"next_position\":" << session.next_position << ",\"committed_tokens\":" << session.committed_tokens
      << ",\"valid\":" << session.valid << ",\"generations\":[";
    for (std::size_t g = 0; g < session.generations.size(); ++g) h << (g ? "," : "") << session.generations[g];
    h << "],\"lineage\":[";
    for (std::size_t i = 0; i < session.lineage.size(); ++i) {
        const auto& v = session.lineage[i];
        h << (i ? "," : "") << '[' << v[0] << ',' << v[1] << ',' << v[2] << ",\"" << rtx_hex_u64(v[3]) << "\",\"" << rtx_hex_u64(v[4])
          << "\",\"" << rtx_hex_u64(v[5]) << "\"]";
    }
    h << "]},\"tokens\":[";
    for (std::size_t t = 0; t < tokens.size(); ++t) h << (t ? "," : "") << tokens[t];
    h << "]}";
    const auto header = h.str();
    auto temporary = path;
    temporary += ".tmp." + std::to_string(STRIX_RTX_GETPID());
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("rtx state file cannot be created: " + temporary.string());
        out.write(rtx_state_file_magic.data(), static_cast<std::streamsize>(rtx_state_file_magic.size()));
        std::array<char, 8> length{};
        const std::uint64_t n = header.size();
        for (unsigned b = 0; b < 8U; ++b) length[b] = static_cast<char>((n >> (8U * b)) & 255U);
        out.write(length.data(), 8);
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        for (std::size_t at = 0; at < payload.size(); at += step) {
            const auto part = payload.subspan(at, std::min(step, payload.size() - at));
            out.write(reinterpret_cast<const char*>(part.data()), static_cast<std::streamsize>(part.size()));
        }
        out.flush();
        if (!out) throw std::runtime_error("rtx state file write failed: " + temporary.string());
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        throw std::runtime_error("rtx state file cannot be published: " + path.string());
    }
    return sha;
}

RtxStateFileView read_rtx_state_file_header(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("rtx state file cannot be read: " + path.string());
    std::array<char, 8> magic{}, length{};
    in.read(magic.data(), 8);
    in.read(length.data(), 8);
    if (!in || magic != rtx_state_file_magic) throw std::invalid_argument("not an rtx state file: " + path.string());
    std::uint64_t n = 0;
    for (unsigned b = 0; b < 8U; ++b) n |= static_cast<std::uint64_t>(static_cast<unsigned char>(length[b])) << (8U * b);
    if (n > (64ULL << 20U)) throw std::invalid_argument("rtx state file header is too large");
    RtxStateFileView view;
    view.header_json.resize(static_cast<std::size_t>(n));
    in.read(view.header_json.data(), static_cast<std::streamsize>(n));
    if (!in) throw std::invalid_argument("rtx state file header is truncated");
    view.payload_offset = 16U + n;
    const auto total = std::filesystem::file_size(path);
    if (total < view.payload_offset) throw std::invalid_argument("rtx state file is truncated");
    view.payload_bytes = total - view.payload_offset;
    return view;
}

}  // namespace strix
