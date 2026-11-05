#pragma once

#include "strix/cuda_qwen_prefix_plan.hpp"
#include "strix/cuda_qwen_prefill.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/prefix_state_cache.hpp"

#include <cstring>
#include <chrono>
#include <functional>

namespace strix::cuda::qwen_prefix {

struct CaptureTiming { double ms = 0, alloc_ms = 0, copy_ms = 0; };
struct CaptureTimer {
    double* elapsed;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~CaptureTimer() {
        if (elapsed) *elapsed += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};

// MTP pair i includes token i+1. A row segment therefore has a one-token
// lookahead dependency, even though checkpoint lookup uses the prefix chain.
inline ModelContentIdentity segment_key(const ModelContentIdentity& chain, bool mtp, std::uint32_t next) {
    if (!mtp) return chain;
    std::array<std::byte, 36> bytes{};
    std::memcpy(bytes.data(), chain.bytes.data(), 32);
    for (unsigned i = 0; i < 4; ++i) bytes[32 + i] = static_cast<std::byte>((next >> (8 * i)) & 255U);
    return sha256_bytes(bytes);
}

inline bool boundary_matches(const PrefixCacheCheckpoint& cp, std::span<const std::uint32_t> tokens, bool mtp) {
    if (!mtp) return cp.metadata.empty();
    if (cp.position >= tokens.size() || cp.metadata.size() != 4) return false;
    std::uint32_t next = 0;
    for (unsigned i = 0; i < 4; ++i) next |= std::to_integer<std::uint32_t>(cp.metadata[i]) << (8 * i);
    return next == tokens[cp.position];
}

// This glue has no CUDA dependency so the real reservation, deduplication and
// eviction path can be exercised with a heap allocator in host-only tests.
inline bool capture(PrefixStateCache& store, const ModelContentIdentity& identity,
                    std::span<const std::uint32_t> tokens, std::uint32_t block, std::uint64_t position,
                    bool mtp, std::uint64_t row_bytes, std::uint64_t fixed_bytes,
                    const std::function<void(const std::vector<qwen_prefill::RowSegment>&, void*)>& export_segments,
                    CaptureTiming* timing = nullptr) {
    CaptureTimer total{timing ? &timing->ms : nullptr};
    validate_block(block);
    if (!position || position % block || position > lookup_limit(tokens.size()))
        throw std::invalid_argument("prefix capture must precede the final prefill row");
    const auto chains = prefix_cache_block_digests(identity, tokens, block, position / block);
    if (store.contains(chains.back())) return false;
    auto cp = std::make_shared<PrefixCacheCheckpoint>();
    cp->identity = identity;
    cp->digest = chains.back();
    cp->position = position;
    cp->block_rows = block;
    if (mtp) {
        for (unsigned i = 0; i < 4; ++i) cp->metadata.push_back(static_cast<std::byte>((tokens[position] >> (8 * i)) & 255U));
    }
    std::uint64_t need = fixed_bytes + cp->metadata.size();
    for (std::size_t k = 0; k < chains.size(); ++k) {
        const auto digest = segment_key(chains[k], mtp, tokens[(k + 1) * block]);
        auto segment = store.find_segment(digest);
        if (segment && (segment->identity != identity || segment->block_rows != block || segment->block_index != k ||
                        segment->tokens.size() != block || segment->payload.bytes != block * row_bytes ||
                        !std::equal(segment->tokens.begin(), segment->tokens.end(), tokens.begin() + k * block)))
            throw std::logic_error("prefix segment verification failed");
        if (!segment) need += block * row_bytes;
        cp->segments.push_back(std::move(segment));
    }
    auto reservation = store.reserve(need);
    if (!reservation) return false;
    std::vector<qwen_prefill::RowSegment> exports;
    for (std::size_t k = 0; k < chains.size(); ++k) {
        if (cp->segments[k]) continue;
        auto segment = std::make_shared<PrefixCacheSegment>();
        segment->identity = identity;
        segment->digest = segment_key(chains[k], mtp, tokens[(k + 1) * block]);
        segment->parent = k ? chains[k - 1] : identity;
        segment->block_index = k;
        segment->block_rows = block;
        segment->tokens.assign(tokens.begin() + k * block, tokens.begin() + (k + 1) * block);
        {
            CaptureTimer allocation{timing ? &timing->alloc_ms : nullptr};
            segment->payload = store.allocate(block * row_bytes, reservation, true);
        }
        exports.push_back({k * block, block, segment->payload.data.get()});
        cp->segments[k] = std::move(segment);
    }
    {
        CaptureTimer allocation{timing ? &timing->alloc_ms : nullptr};
        cp->fixed = store.allocate(fixed_bytes, reservation);
    }
    {
        CaptureTimer copies{timing ? &timing->copy_ms : nullptr};
        export_segments(exports, cp->fixed.data.get());
    }
    return store.insert(std::move(cp), &reservation);
}

// Compatibility adapter for callers that only need the per-buffer host-test API.
inline bool capture(PrefixStateCache& store, const ModelContentIdentity& identity,
                    std::span<const std::uint32_t> tokens, std::uint32_t block, std::uint64_t position,
                    bool mtp, std::uint64_t row_bytes, std::uint64_t fixed_bytes,
                    const std::function<void(std::uint64_t, std::uint64_t, void*)>& rows,
                    const std::function<void(void*)>& fixed) {
    return capture(store, identity, tokens, block, position, mtp, row_bytes, fixed_bytes,
        [&](const std::vector<qwen_prefill::RowSegment>& segments, void* dst) {
            for (const auto& seg : segments) rows(seg.first, seg.count, const_cast<void*>(seg.data));
            fixed(dst);
        });
}

}  // namespace strix::cuda::qwen_prefix
