#pragma once
// Externally produced prompt state (strix-qwen-rtx-state-v1): the host-side format, staging and registry of a
// prefix-state checkpoint at an arbitrary prompt position p, produced elsewhere (an RTX prefill service, or
// another Strix node for testing) and restored into a fresh Qwen session by the tiled MTP prompt
// (apps/hip_prefix_state_cache.hpp, docs/rtx-state-import.md).
//
// Canonical payload of a state at position p: every row component's rows [0, p - shift) in component order
// (row-major, component c's first row at component_offset(c, p)), then the padded fixed blob (fixed_bytes, every
// fixed component at its 256-aligned offset, the padding zero, never sent). Integrity: every chunk POST carries
// the SHA-256 of its body, verified on receipt (overlapped with the transfer); commit compares the chunk count,
// body bytes and the SHA-256 of the chunk digests in chunk order, and checks that every byte was written exactly
// once. The canonical-payload SHA-256 is an optional diagnostic. No GPU here: the engine copies a committed
// staging into device state.
#include "strix/bp_codec.hpp"
#include "strix/rotary_descriptor.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/prefix_state_cache.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix {

inline constexpr std::string_view rtx_state_format = "strix-qwen-rtx-state-v1";

struct RtxStateRowComponent {
    std::string id;
    std::uint64_t bytes_per_row = 0;
    std::uint32_t shift = 0;  // row j of the state at p exists for j < p - shift
};
struct RtxStateFixedComponent {
    std::string id;
    std::uint64_t bytes = 0;
    std::uint64_t offset = 0;  // in the padded fixed blob
};

// The state geometry one engine accepts, published by the engine (GET /admin/rtx-import/layout).
struct RtxStateLayout {
    std::vector<RtxStateRowComponent> rows;
    std::vector<RtxStateFixedComponent> fixed;
    std::uint64_t row_bytes = 0, fixed_bytes = 0;
    std::uint64_t layout_identity = 0;  // the session state layout identity of the consumer
    std::uint64_t context_tokens = 0;  // consumer route capacity
    RotaryIdentity rotary;  // resolved model identity, including the existing table SHA256
    // Descriptive (the consumer's own facts, reported so a producer can check it targets the same model):
    // per prompt row, how many times each persistent-state flag generation advances (the serial plan's
    // writes), the union of written flags, the fresh session's generations, model and identity extension digests.
    std::array<std::uint64_t, 6> generations_per_row{};
    std::array<std::uint64_t, 6> generations_base{};
    std::uint32_t valid_union = 0, valid_base = 0;
    std::string model_content_hex, identity_extension_hex, cache_identity_hex;
    std::uint64_t execution_identity = 0, runtime_compatibility_identity = 0;

    [[nodiscard]] std::uint64_t component_rows(std::size_t component, std::uint64_t position) const;
    [[nodiscard]] std::uint64_t component_offset(std::size_t component, std::uint64_t position) const;
    [[nodiscard]] std::uint64_t rows_bytes(std::uint64_t position) const;
    [[nodiscard]] std::uint64_t payload_bytes(std::uint64_t position) const;
    [[nodiscard]] std::optional<std::size_t> find_row(std::string_view id) const;
    [[nodiscard]] std::optional<std::size_t> find_fixed(std::string_view id) const;
    // Throws when the components do not add up (row bytes, fixed offsets 256-aligned, ascending, in bounds).
    void validate() const;
};

// FNV-1a over the component list and the layout identity: a producer that echoes it proves it splits the
// payload exactly as this consumer does. YaRN additionally binds route capacity and
// the resolved rotary identity; native retains the original identity byte stream.
[[nodiscard]] std::uint64_t rtx_state_geometry_identity(const RtxStateLayout& layout);

struct RtxStateProducer {
    std::uint64_t numerical = 0, reduction = 0, state_update = 0;  // nonzero: they name the producing arithmetic
};

struct RtxStateBeginRequest {
    std::vector<std::uint32_t> tokens;  // prompt[0, position)
    std::uint64_t position = 0;
    RtxStateProducer producer;
    std::uint64_t layout_identity = 0;    // must equal the consumer's
    std::uint64_t geometry_identity = 0;  // 0: native unchecked; YaRN needs explicit rotary + capacity
    std::optional<RotaryIdentity> rotary;
    std::optional<std::uint64_t> context_tokens;
    bool keep = false;                    // true: a request does not consume the parked state
    // Chunk encoding the producer may use ("" = none; "bp-v1": chunks with X-Chunk-Encoding: bp-v1 carry a
    // bp_codec frame of the plain body). Chunks without the header are plain either way.
    std::string encoding;
    // Cooperative prefill (0: off): rows [0, local_rows) of every row component are not sent; at restore they come
    // from this node's own prefix-cache checkpoint of prompt[0, local_rows), which must then exist (a block boundary
    // the node prefilled itself while the producer streamed the rest). The fixed state is always sent.
    std::uint64_t local_rows = 0;
};

// The chunk encodings this node decodes (advertised by GET /admin/rtx-import/layout as `codecs`).
inline constexpr std::array<std::string_view, 1> rtx_chunk_codecs{bp_codec::codec_name};
// Upper bound of a decoded chunk body (the plain chunk body limit of the routes).
inline constexpr std::uint64_t rtx_max_decoded_chunk_bytes = 256ULL << 20U;

// One piece of payload: canonical bytes [canonical_offset, canonical_offset + data.size()).
struct RtxStateSegment {
    std::uint64_t canonical_offset = 0;
    std::span<const std::byte> data;
};
// One received chunk POST: its body digest (verified on receipt against X-Chunk-Sha256) and its order.
struct RtxStateChunkRecord {
    std::optional<std::uint64_t> seq;  // X-Chunk-Seq; unset: the receive order
    ModelContentIdentity digest;       // of the wire body (the encoded frame when X-Chunk-Encoding is set)
    std::uint64_t wire_bytes = 0;      // wire body bytes (statistics)
};
// commit: {"chunks": N, "bytes": B (segment data bytes, headers excluded), "manifest_sha256": SHA-256(digest_0 || ... ||
// digest_{N-1}) in chunk order}.
struct RtxStateCommitRequest {
    std::uint64_t chunks = 0, bytes = 0;
    ModelContentIdentity manifest;
    bool canonical_sha256 = false;  // diagnostic: also hash the canonical payload after the checks
};
struct RtxStateCommitResult {
    double verify_ms = 0.0, canonical_ms = 0.0;
    std::optional<ModelContentIdentity> canonical;
};

// One import in flight: a host buffer of payload_bytes(position) and the coverage of what was written.
class RtxStateStaging {
public:
    RtxStateStaging(std::shared_ptr<const RtxStateLayout> layout, RtxStateBeginRequest request, std::string id,
        const PrefixCacheAllocator& allocate);
    RtxStateStaging(const RtxStateStaging&) = delete;
    RtxStateStaging& operator=(const RtxStateStaging&) = delete;

    // Segment builders (bounds-checked): whole rows [first_row, first_row + n) of row component c, and bytes
    // [offset, offset + n) of fixed component c (offset relative to the component; padding is never sent).
    [[nodiscard]] RtxStateSegment rows_segment(std::size_t component, std::uint64_t first_row, std::span<const std::byte> data) const;
    [[nodiscard]] RtxStateSegment fixed_segment(std::size_t component, std::uint64_t offset, std::span<const std::byte> data) const;
    // Writes every segment of one chunk, all or none: each must lie in the payload and overlap neither the
    // bytes already written nor another segment. Then records the chunk.
    void write_chunk(std::span<const RtxStateSegment> segments, const RtxStateChunkRecord& record);
    // Verifies that no copy is in flight, every payload byte was written exactly once, the chunk count, byte
    // count and manifest match the received chunks; then the staging is immutable. Throws (stays open) otherwise.
    RtxStateCommitResult commit(const RtxStateCommitRequest& request);
    // SHA-256 of the canonical payload (after commit; diagnostic, ~0.5 s per GB).
    [[nodiscard]] ModelContentIdentity canonical_sha256() const;

    [[nodiscard]] const std::string& id() const noexcept { return id_; }
    [[nodiscard]] const RtxStateBeginRequest& request() const noexcept { return request_; }
    [[nodiscard]] const RtxStateLayout& layout() const noexcept { return *layout_; }
    [[nodiscard]] std::uint64_t position() const noexcept { return request_.position; }
    [[nodiscard]] std::uint64_t payload_bytes() const noexcept { return bytes_; }
    [[nodiscard]] std::uint64_t covered_bytes() const;
    [[nodiscard]] std::uint64_t chunks() const;
    [[nodiscard]] bool committed() const;
    [[nodiscard]] ModelContentIdentity manifest_sha256() const;  // valid after commit
    [[nodiscard]] std::uint64_t wire_bytes() const;               // sum of the chunks' wire body bytes
    // After commit only: the payload, component c's rows and the fixed blob.
    [[nodiscard]] std::span<const std::byte> payload() const;
    [[nodiscard]] std::span<const std::byte> component(std::size_t c) const;
    [[nodiscard]] std::span<const std::byte> fixed_blob() const;
    [[nodiscard]] double commit_ms() const noexcept { return commit_ms_; }

private:
    void require_open() const;
    void check_free(std::uint64_t offset, std::uint64_t n) const;
    void reserve(std::uint64_t offset, std::uint64_t n);
    std::shared_ptr<const RtxStateLayout> layout_;
    RtxStateBeginRequest request_;
    std::string id_;
    PrefixCacheBuffer buffer_;
    std::uint64_t bytes_ = 0;
    mutable std::mutex mutex_;
    std::map<std::uint64_t, std::uint64_t> covered_;  // begin -> end, disjoint, coalesced
    std::uint64_t covered_bytes_ = 0;
    std::uint32_t writing_ = 0;
    std::map<std::uint64_t, ModelContentIdentity> chunk_digests_;  // seq -> body digest
    std::uint64_t chunk_bytes_ = 0, wire_bytes_ = 0;
    int seq_mode_ = -1;  // -1 none yet, 0 receive order, 1 X-Chunk-Seq
    bool committed_ = false;
    ModelContentIdentity manifest_{};
    double commit_ms_ = 0.0;
};

// The segments of a multi-segment chunk body (POST /admin/rtx-import/chunks): a sequence of
// {u32 kind (0 rows, 1 fixed), u32 component index, u32 first_row | offset_lo, u32 rows | offset_hi, u64 bytes, bytes}
// little-endian. Rows: bytes == rows * bytes_per_row. Throws on a malformed body or a bound violation.
[[nodiscard]] std::vector<RtxStateSegment> parse_rtx_chunk_segments(const RtxStateStaging& staging, std::span<const std::byte> body);
[[nodiscard]] ModelContentIdentity rtx_chunk_digest(std::span<const std::byte> body);
[[nodiscard]] ModelContentIdentity rtx_manifest_digest(std::span<const ModelContentIdentity> chunk_digests);

struct RtxStateStatusEntry {
    std::string id;
    std::uint64_t position = 0, payload_bytes = 0, covered_bytes = 0;
    bool committed = false, parked = false, keep = false;
};
struct RtxStateRegistryStats {
    std::uint64_t begun = 0, committed = 0, aborted = 0, failed_commits = 0, taken = 0, misses = 0;
    std::string last_miss;
};

// Imports of one engine. Network threads begin, fill, commit and abort; the request thread takes the parked
// state at the start of a request. Committed stagings are immutable, so a taken one is read without locks.
class RtxStateImportRegistry {
public:
    // decode_threads: background workers of the chunk decoder pool (created on the first encoded chunk; the
    // connection thread decodes too).
    explicit RtxStateImportRegistry(PrefixCacheAllocator allocate, std::size_t max_in_flight = 2, unsigned decode_threads = 4);
    // The engine's geometry; begin() fails until it is set.
    void set_layout(RtxStateLayout layout);
    [[nodiscard]] std::shared_ptr<const RtxStateLayout> layout() const;
    std::string begin(RtxStateBeginRequest request);
    [[nodiscard]] std::shared_ptr<RtxStateStaging> find(std::string_view id) const;  // null: unknown
    // Verifies and parks (replacing an earlier parked state); the staging leaves the in-flight set.
    RtxStateCommitResult commit(std::string_view id, const RtxStateCommitRequest& request);
    bool abort(std::string_view id);
    // The parked state when its tokens are a prefix of `prompt` and position <= prompt.size() - 1; it stays
    // parked when it was begun with keep. Null otherwise (the parked state is left in place).
    [[nodiscard]] std::shared_ptr<const RtxStateStaging> take(std::span<const std::uint32_t> prompt);
    [[nodiscard]] std::vector<RtxStateStatusEntry> status() const;
    [[nodiscard]] RtxStateRegistryStats stats() const;
    // The shared decoder pool of encoded chunks (thread-safe; several connections may decode at once).
    [[nodiscard]] bp_codec::WorkerPool& decode_pool();
    [[nodiscard]] unsigned decode_threads() const noexcept { return decode_threads_; }

private:
    PrefixCacheAllocator allocate_;
    std::size_t max_in_flight_;
    unsigned decode_threads_;
    std::once_flag decode_pool_once_;
    std::unique_ptr<bp_codec::WorkerPool> decode_pool_;
    mutable std::mutex mutex_;
    std::shared_ptr<const RtxStateLayout> layout_;
    std::map<std::string, std::shared_ptr<RtxStateStaging>, std::less<>> in_flight_;
    std::shared_ptr<RtxStateStaging> parked_;
    std::uint64_t counter_ = 0;
    RtxStateRegistryStats stats_;
};

// ---- the file form (export for testing; the producer of an RTX service writes the same) ----
// "STRXRTX1", u64 LE header bytes, header JSON (UTF-8), canonical payload.
inline constexpr std::array<char, 8> rtx_state_file_magic{'S', 'T', 'R', 'X', 'R', 'T', 'X', '1'};

struct RtxStateFileSession {
    // The producing session's own counters (diagnostic: compared against the consumer's formula).
    std::uint64_t next_position = 0, committed_tokens = 0;
    std::uint32_t valid = 0;
    std::array<std::uint64_t, 6> generations{};
    std::vector<std::array<std::uint64_t, 6>> lineage;  // begin, end, producer, numerical, reduction, state_update
};
// Writes <path>.tmp.<pid> then renames. Returns the payload SHA-256.
ModelContentIdentity write_rtx_state_file(const std::filesystem::path& path, const RtxStateLayout& layout,
    std::span<const std::uint32_t> tokens, std::uint64_t position, const RtxStateProducer& producer,
    const RtxStateFileSession& session, std::span<const std::byte> payload);
// The header JSON text and the payload offset of a file (for tools and tests).
struct RtxStateFileView {
    std::string header_json;
    std::uint64_t payload_offset = 0, payload_bytes = 0;
};
[[nodiscard]] RtxStateFileView read_rtx_state_file_header(const std::filesystem::path& path);

[[nodiscard]] std::string rtx_hex_u64(std::uint64_t value);  // "0x" + 16 lower-case digits
// Accepts "0x..." hex or decimal. Throws std::invalid_argument.
[[nodiscard]] std::uint64_t rtx_parse_u64(std::string_view text);
[[nodiscard]] ModelContentIdentity rtx_parse_sha256(std::string_view hex);

}  // namespace strix
