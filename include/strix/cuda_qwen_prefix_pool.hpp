#pragma once

#include "strix/prefix_state_cache.hpp"

namespace strix::cuda::qwen_prefix {

// CUDA-free pool mechanics: Cache supplies the pinned allocator, host tests a heap
// allocator. Reserved bytes include live slots; unused slots are never charged twice.
class SlabPool {
    struct Slab {
        PrefixCacheBuffer backing;
        std::uint64_t slot_bytes = 0, live = 0;
        std::vector<std::uint64_t> free;
    };
    struct State {
        std::mutex mutex;
        std::uint64_t budget, row_slot_bytes, row_slots, reserved = 0, live = 0;
        PrefixCacheAllocator allocator;
        std::vector<std::shared_ptr<Slab>> slabs;
        State(std::uint64_t budget, std::uint64_t row_bytes, std::uint64_t slots, PrefixCacheAllocator allocator)
            : budget(budget), row_slot_bytes(row_bytes), row_slots(slots), allocator(std::move(allocator)) {}
    };
    std::shared_ptr<State> state_;

public:
    struct Usage { std::uint64_t live, reserved; };
    SlabPool(std::uint64_t budget, std::uint64_t row_slot_bytes, PrefixCacheAllocator allocator,
             std::uint64_t row_slots = 32)
        : state_(std::make_shared<State>(budget, row_slot_bytes, row_slots, std::move(allocator))) {
        if (!row_slot_bytes || !row_slots) throw std::invalid_argument("prefix slab dimensions");
    }

    PrefixCacheBuffer allocate(std::uint64_t bytes) {
        auto state = state_;
        std::unique_lock lock(state->mutex);
        if (!bytes || bytes > state->budget) throw PrefixCacheAllocationFailure();
        std::shared_ptr<Slab> slab;
        for (const auto& candidate : state->slabs)
            if (candidate->slot_bytes == bytes && !candidate->free.empty()) { slab = candidate; break; }
        if (!slab) {
            // Only completely idle slabs may be reclaimed, and only when a different
            // size class needs budget. Outstanding leases keep their backing alive.
            if (bytes > state->budget - state->reserved) {
                for (auto it = state->slabs.begin(); it != state->slabs.end();) {
                    if (!(*it)->live && (*it)->slot_bytes != bytes) {
                        state->reserved -= (*it)->backing.bytes;
                        it = state->slabs.erase(it);
                        if (bytes <= state->budget - state->reserved) break;
                    } else ++it;
                }
            }
            const auto available = state->budget - state->reserved;
            if (bytes > available) throw PrefixCacheAllocationFailure();
            auto slots = bytes == state->row_slot_bytes ? state->row_slots : 1;
            // Prefer a full slab; near the cap, retain the exact-size fallback.
            if (slots > available / bytes) slots = 1;
            slab = std::make_shared<Slab>();
            slab->slot_bytes = bytes;
            slab->free.reserve(static_cast<std::size_t>(slots));
            for (std::uint64_t i = slots; i; --i) slab->free.push_back(i - 1);
            try { slab->backing = state->allocator(bytes * slots); }
            catch (const PrefixCacheAllocationFailure&) {
                if (slots == 1) throw;
                slots = 1;
                slab->free.assign(1, 0);
                slab->backing = state->allocator(bytes);
            }
            if (!slab->backing.data || slab->backing.bytes != bytes * slots) throw PrefixCacheAllocationFailure();
            state->slabs.push_back(slab);
            state->reserved += slab->backing.bytes;
        }
        const auto index = slab->free.back();
        slab->free.pop_back();
        ++slab->live;
        state->live += bytes;
        // shared_ptr invokes the deleter even if allocating its control block fails.
        // Release the mutex first, since that deleter returns the slot under the lock.
        auto* ptr = slab->backing.data.get() + index * bytes;
        lock.unlock();
        return {std::shared_ptr<std::byte>(ptr, [state, slab, index, bytes](std::byte*) {
            std::lock_guard release_lock(state->mutex);
            slab->free.push_back(index); // capacity was reserved when the slab was created
            --slab->live;
            state->live -= bytes;
        }), bytes};
    }

    Usage usage() const {
        std::lock_guard lock(state_->mutex);
        return {state_->live, state_->reserved};
    }
};

}  // namespace strix::cuda::qwen_prefix
