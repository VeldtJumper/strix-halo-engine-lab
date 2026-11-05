#pragma once

#include "strix/cuda_qwen_prefix_identity.hpp"
#include "strix/cuda_qwen_prefix_store.hpp"
#include "strix/cuda_qwen_prefix_pool.hpp"
#include "strix/cuda_vram_guard.hpp"

#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>

namespace strix::cuda::qwen_prefix {

struct Settings {
    bool enabled = true;
    std::uint32_t block = 256;
    std::uint64_t stride = 16384;
    std::uint64_t host_bytes = default_host_budget(), vram_bytes = 0;
};

class Cache {
    struct Usage { std::atomic<std::uint64_t> vram{0}; };
    Settings settings_;
    bool mtp_;
    std::shared_ptr<Usage> usage_ = std::make_shared<Usage>();
    std::shared_ptr<SlabPool> host_pool_;
    ModelContentIdentity identity_;
    std::unique_ptr<PrefixStateCache> store_;
    std::atomic<std::uint64_t> restore_failures_{0};
    std::atomic<double> capture_ms_{0}, capture_alloc_ms_{0}, capture_copy_ms_{0};

public:
    Cache(const qwen_prefill::Config& config, const Settings& settings) : settings_(settings), mtp_(!config.mtp_sidecar_gguf.empty()) {
        if (settings_.enabled) {
            identity_ = identity_key(identity_fields(config, settings_.block));
            PrefixStateCachePolicy policy;
            policy.host_byte_budget = settings_.host_bytes + settings_.vram_bytes;
            policy.retention = PrefixCacheRetention::lru;
            const auto row_slot_bytes = settings_.block * (12ULL + (mtp_ ? 1ULL : 0ULL)) * (2048 + 256);
            host_pool_ = std::make_shared<SlabPool>(settings_.host_bytes, row_slot_bytes,
                [](std::uint64_t bytes) -> PrefixCacheBuffer {
                    void* ptr = nullptr;
                    if (cudaMallocHost(&ptr, bytes) != cudaSuccess) {
                        (void)cudaGetLastError();
                        throw PrefixCacheAllocationFailure();
                    }
                    return {std::shared_ptr<std::byte>(static_cast<std::byte*>(ptr), [](std::byte* p) { cudaFreeHost(p); }), bytes};
                });
            store_ = std::make_unique<PrefixStateCache>(policy, [usage = usage_, pool = host_pool_, settings](std::uint64_t bytes) -> PrefixCacheBuffer {
                void* ptr = nullptr;
                const bool device = bytes <= settings.vram_bytes - usage->vram.load();
                if (!device) return pool->allocate(bytes);
                strix::cuda::require_vram(bytes, "prefix cache");
                const auto rc = cudaMalloc(&ptr, bytes);
                if (rc != cudaSuccess) { (void)cudaGetLastError(); throw PrefixCacheAllocationFailure(); }
                usage->vram.fetch_add(bytes);
                return {std::shared_ptr<std::byte>(static_cast<std::byte*>(ptr), [usage, bytes](std::byte* p) {
                    cudaFree(p);
                    usage->vram.fetch_sub(bytes);
                }), bytes};
            });
        }
        std::cerr << "strix-cuda-qwen-serve: prefix cache " << (settings_.enabled ? "on" : "off")
                  << ", host " << settings_.host_bytes / 1048576 << " MiB, VRAM " << settings_.vram_bytes / 1048576
                  << " MiB, block " << settings_.block << ", stride " << settings_.stride << std::endl;
    }

    bool enabled() const { return bool(store_); }

    std::vector<std::uint64_t> plan(std::uint64_t n, std::uint64_t restored) const {
        return enabled() ? capture_points(n, settings_.block, restored, settings_.stride) : std::vector<std::uint64_t>{};
    }

    std::uint64_t restore(qwen_prefill::Prefill& engine, qwen_prefill::SessionId session,
                          const std::vector<std::uint32_t>& prompt, double& ms) {
        ms = 0;
        if (!store_ || prompt.size() < 2 || prompt.size() - 1 <= settings_.block) return 0;
        try {
            const auto cp = store_->find_longest(identity_, prompt, settings_.block, lookup_limit(prompt.size()), true, {},
                [&](const PrefixCacheCheckpoint& candidate) { return boundary_matches(candidate, prompt, mtp_); });
            if (!cp) return 0;
            std::vector<qwen_prefill::RowSegment> segments;
            for (const auto& seg : cp->segments)
                segments.push_back({seg->block_index * seg->block_rows, seg->block_rows, seg->payload.data.get()});
            const auto begin = std::chrono::steady_clock::now();
            engine.import_state(session, cp->position, segments, cp->fixed.data.get());
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            return cp->position;
        } catch (const std::exception& ex) {
            ++restore_failures_;
            std::cerr << "qwen prefix restore skipped: " << ex.what() << std::endl;
            return 0;
        }
    }

    double capture(qwen_prefill::Prefill& engine, qwen_prefill::SessionId session,
                  const std::vector<std::uint32_t>& prompt, std::uint64_t position) noexcept {
        if (!store_) return 0;
        CaptureTiming timing;
        try {
            qwen_prefix::capture(*store_, identity_, prompt, settings_.block, position, mtp_, engine.row_bytes(), engine.fixed_bytes(),
                [&](const std::vector<qwen_prefill::RowSegment>& segments, void* fixed) {
                    engine.export_segments(session, segments, fixed);
                }, &timing);
        } catch (const std::exception& ex) {
            store_->record_capture_failure();
            std::cerr << "qwen prefix capture skipped: " << ex.what() << std::endl;
        } catch (...) {
            store_->record_capture_failure();
            std::cerr << "qwen prefix capture skipped: unknown failure" << std::endl;
        }
        capture_ms_.fetch_add(timing.ms);
        capture_alloc_ms_.fetch_add(timing.alloc_ms);
        capture_copy_ms_.fetch_add(timing.copy_ms);
        return timing.ms;
    }

    std::string stats_json() const {
        const auto s = store_ ? store_->stats() : PrefixStateCacheStats{};
        const auto host = host_pool_ ? host_pool_->usage() : SlabPool::Usage{};
        std::ostringstream out;
        out << "{\"enabled\":" << (enabled() ? "true" : "false") << ",\"lookups\":" << s.lookups
            << ",\"hits\":" << s.host_hits << ",\"hit_tokens\":" << s.hit_tokens << ",\"captures\":" << s.inserted
            << ",\"bytes\":" << s.host_bytes << ",\"evictions\":" << s.evicted << ",\"budget_skips\":" << s.budget_skips
            << ",\"capture_failures\":" << s.capture_failures << ",\"restore_failures\":" << restore_failures_.load()
            << ",\"capture_ms\":" << capture_ms_.load() << ",\"capture_alloc_ms\":" << capture_alloc_ms_.load()
            << ",\"capture_copy_ms\":" << capture_copy_ms_.load()
            << ",\"host_bytes\":" << host.live << ",\"host_reserved_bytes\":" << host.reserved
            << ",\"vram_bytes\":" << usage_->vram.load() << '}';
        return out.str();
    }
};

}  // namespace strix::cuda::qwen_prefix
