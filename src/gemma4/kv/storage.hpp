#pragma once

#include "gemma4/kernels/ops.hpp"
#include "layout.hpp"

namespace strix::gemma4::gpu {
// One immutable snapshot and one unpublished overlay per transaction/producer.
struct KvRead {
    const float* k = nullptr;
    const float* v = nullptr;
    const std::uint64_t* tags = nullptr;
    const float* overlay_k = nullptr;
    const float* overlay_v = nullptr;
    std::uint64_t frontier = 0, position = 0;
    std::uint32_t width = 0, slots = 0, candidate_rows = 0, window = 0;
    bool global = false;
};
struct KvWrite {
    float* k = nullptr;
    float* v = nullptr;
    std::uint32_t row = 0;
};
void store_overlay(const float* k, const float* v, const KvWrite* destinations, std::uint32_t rows, std::uint32_t width,
                   Queue q = nullptr);
void attention(const float* q, const KvRead* reads, std::uint32_t rows, std::uint32_t head_count, std::uint32_t dim,
               std::uint32_t score_stride, float* scores, float* maxima, float* partials, float* out,
               std::uint32_t* failure, Queue queue = nullptr, bool allow_fusion = true);

// Fused decode schedule (STRIX_G4_FUSED_ATTN, default on): the same values as
// head_norm_rope + store_overlay and head_norm_rope + attention, one launch
// each. They return false, enqueueing nothing, when the geometry is not admitted.
// Spans above STRIX_G4_FUSED_ATTN_CHUNKS use two chunk-parallel launches instead
// (scores with the query norm; partials with ordered max and last-block merge);
// counters holds rows*heads zeroed entries, scratch is sized as for attention().
bool kv_store(const float* k, const float* v, LinearView k_norm, const std::uint64_t* positions, const float* table,
              const KvWrite* destinations, std::uint32_t rows, std::uint32_t kv_heads, std::uint32_t dim, bool global,
              std::uint32_t* failure, Queue queue, bool parallel, bool bounded, bool intervals);
bool fused_attention(const float* q, LinearView q_norm, const std::uint64_t* positions, const float* table,
                     const KvRead* reads, std::uint32_t rows, std::uint32_t head_count, std::uint32_t dim,
                     std::uint32_t score_stride, bool global, float* out, std::uint32_t* failure, Queue queue,
                     bool parallel, bool bounded, bool intervals, float* scores = nullptr, float* maxima = nullptr,
                     float* partials = nullptr, std::uint32_t* counters = nullptr, A8Rows quant = {},
                     int kv_shared = -1);
// fused_attention with kv_store folded in (STRIX_G4_ATTN_KV_STORE, default on):
// only for passes in which every row is its transaction's only new position.
// Returns false, enqueueing nothing, when the single-launch span or geometry is
// not admitted; then kv_store and fused_attention run separately.
bool fused_attention_store(const float* q, LinearView q_norm, const float* k, const float* v, LinearView k_norm,
                           const KvWrite* destinations, std::uint32_t kv_heads, const std::uint64_t* positions,
                           const float* table, const KvRead* reads, std::uint32_t rows, std::uint32_t head_count,
                           std::uint32_t dim, std::uint32_t score_stride, bool global, float* out,
                           std::uint32_t* failure, Queue queue, bool parallel, bool bounded, bool intervals,
                           A8Rows quant = {});
bool fused_attention_enabled();
// STRIX_G4_ATTN_KV_SHARED: the long-span launches run every row and query head of
// one key/value chunk back to back (1), or one block serves a key/value head's
// query heads (2); by default one-row passes use 1 and larger passes 2.
// 0 = chunk-fastest blocks per row and head. Same bytes in every mode.
// kv_shared -1 follows the switch, 0..3 force a mode (tests).
bool attention_kv_shared_enabled();
int attention_kv_shared_mode();

class DeviceStream {
  public:
    // Allocates exactly the admitted global context, plus sliding rings and absolute tags.
    explicit DeviceStream(const ModelConfig& config, std::uint32_t admitted_context);
    ~DeviceStream();
    DeviceStream(const DeviceStream&) = delete;
    DeviceStream& operator=(const DeviceStream&) = delete;
    [[nodiscard]] std::unique_ptr<KvTxn> begin();
    [[nodiscard]] std::uint64_t frontier() const;
    [[nodiscard]] std::string digest() const; // matches HostStream canonical logical digest
    // A new stream admitting `admitted` >= frontier() positions whose committed state is a byte copy of this one
    // (global slots [0, frontier), the whole sliding rings and their position tags): same digest, same next passes.
    [[nodiscard]] std::unique_ptr<DeviceStream> fork(std::uint32_t admitted) const;
    // Read-only view of a producing layer: committed snapshot plus, when given, that transaction's current overlay
    // (position-tag checked by the reader). Additive; no layout change. Drafters never write KV.
    [[nodiscard]] KvView kv_view(std::uint32_t layer, const KvTxn* transaction = nullptr) const;
    struct Impl;

  private:
    std::shared_ptr<Impl> impl_;
    friend class DeviceTarget;
};
namespace detail {
struct Cache {
    Buffer k, v, tags;
    std::uint32_t width = 0, slots = 0;
    bool global = false;
};
struct Overlay {
    Buffer k, v;
};
struct Publication {
    float* k = nullptr;
    float* v = nullptr;
    std::uint64_t* tags = nullptr;
    const float* overlay_k = nullptr;
    const float* overlay_v = nullptr;
    std::uint32_t slots = 0, width = 0;
    bool global = false;
};
} // namespace detail
struct DeviceStream::Impl {
    ModelConfig config;
    std::uint32_t admitted = 0;
    std::uint64_t frontier = 0;
    bool busy = false, poisoned = false;
    bool pooled = true;
    bool publish_batch = true;
    Buffer publication;
    std::vector<detail::Cache> cache;
    std::vector<detail::Overlay> spare_overlay;
    struct Txn final : KvTxn {
        std::shared_ptr<Impl> stream;
        std::uint64_t start;
        std::vector<detail::Overlay> overlay;
        std::size_t rows = 0;
        std::size_t reserved = 0;
        bool evaluated = false, published = false, failed = false;
        explicit Txn(std::shared_ptr<Impl> s);
        ~Txn() override;
        [[nodiscard]] std::uint64_t frontier() const override { return start; }
        void reserve_rows(std::size_t count) override;
        void commit(std::size_t prefix_count) override;
        [[nodiscard]] KvRead read(std::uint32_t producer, std::uint64_t position) const;
    };
};
} // namespace strix::gemma4::gpu
