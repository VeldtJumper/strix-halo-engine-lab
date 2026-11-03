// CPU tests of the externally produced prompt state import (strix-qwen-rtx-state-v1): layout, staging, coverage,
// per-chunk digests and manifest, registry, file form and the HTTP routes, including bp-v1 encoded chunks.
// `--bench-loopback BYTES [CHUNK]` measures the receive throughput of the routes over a loopback HTTP/1.1 keep-alive
// connection instead (POSIX only). `--serve-file PORT STATE.rtxs [DECODE_THREADS]` serves the routes with the layout of
// a real state file (CPU only) until GET /stop; GET /take compares the parked import with the file's payload.
#include "rtx_state_import_routes.hpp"
#include "strix/bp_codec.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/rtx_state_import.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
template <class F>
void expect_throw(F&& f, const std::string& what) {
    try {
        f();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("expected an exception: " + what);
}

strix::PrefixCacheBuffer heap(std::uint64_t bytes) {
    std::shared_ptr<std::byte> data(new std::byte[static_cast<std::size_t>(bytes ? bytes : 1)], std::default_delete<std::byte[]>());
    std::memset(data.get(), 0xAB, static_cast<std::size_t>(bytes ? bytes : 1));  // never zero: padding must be written by the staging
    return {std::move(data), bytes};
}

// A small layout shaped like the engine's: two target row components, two MTP rows with shift 1, three fixed ranges.
strix::RtxStateLayout small_layout() {
    strix::RtxStateLayout l;
    l.rows = {{"kv.L3", 16, 0}, {"index.L3", 4, 0}, {"mtp.kv", 8, 1}, {"mtp.index", 2, 1}};
    l.row_bytes = 30;
    l.fixed = {{"recurrent_matrix", 300, 0}, {"ngram_history", 8, 512}, {"mtp.held", 40, 768}};
    l.fixed_bytes = 1024;
    l.layout_identity = 0xfeedfacecafebeefULL;  // above 2^53: must travel as a string
    l.context_tokens = 64;
    l.generations_per_row = {12, 36, 37, 12, 1, 1};
    l.valid_union = 63;
    return l;
}

std::byte value_at(std::uint64_t canonical) { return static_cast<std::byte>((canonical * 131U + 7U) % 251U); }

// The canonical payload of the small layout at position p (padding zero).
std::vector<std::byte> canonical_payload(const strix::RtxStateLayout& l, std::uint64_t p) {
    std::vector<std::byte> out(static_cast<std::size_t>(l.payload_bytes(p)));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = value_at(i);
    const auto rows_end = l.rows_bytes(p);
    for (std::size_t f = 0; f < l.fixed.size(); ++f) {
        const auto end = l.fixed[f].offset + l.fixed[f].bytes;
        const auto next = f + 1U < l.fixed.size() ? l.fixed[f + 1U].offset : l.fixed_bytes;
        for (auto i = end; i < next; ++i) out[static_cast<std::size_t>(rows_end + i)] = std::byte{0};
    }
    return out;
}

std::span<const std::byte> slice(const std::vector<std::byte>& v, std::uint64_t at, std::uint64_t n) {
    return std::span<const std::byte>(v).subspan(static_cast<std::size_t>(at), static_cast<std::size_t>(n));
}

strix::RtxStateBeginRequest begin_request(const strix::RtxStateLayout& l, std::uint64_t p) {
    strix::RtxStateBeginRequest r;
    for (std::uint64_t i = 0; i < p; ++i) r.tokens.push_back(static_cast<std::uint32_t>(100U + i));
    r.position = p;
    r.producer = {11, 22, 33};
    r.layout_identity = l.layout_identity;
    return r;
}

void put32(std::vector<std::byte>& out, std::uint32_t v) { for (unsigned i = 0; i < 4U; ++i) out.push_back(static_cast<std::byte>((v >> (8U * i)) & 255U)); }
void put64(std::vector<std::byte>& out, std::uint64_t v) { for (unsigned i = 0; i < 8U; ++i) out.push_back(static_cast<std::byte>((v >> (8U * i)) & 255U)); }

// A multi-segment body: every row component's rows [first, first+count) clipped to its rows, or one fixed range.
std::vector<std::byte> rows_body(const strix::RtxStateLayout& l, std::uint64_t p, const std::vector<std::byte>& payload,
    std::uint64_t first, std::uint64_t count, std::uint64_t& data_bytes) {
    std::vector<std::byte> body;
    for (std::size_t c = 0; c < l.rows.size(); ++c) {
        const auto rows = l.component_rows(c, p);
        if (first >= rows) continue;
        const auto n = std::min(count, rows - first);
        const auto bytes = n * l.rows[c].bytes_per_row;
        put32(body, 0); put32(body, static_cast<std::uint32_t>(c)); put32(body, static_cast<std::uint32_t>(first)); put32(body, static_cast<std::uint32_t>(n));
        put64(body, bytes);
        const auto data = slice(payload, l.component_offset(c, p) + first * l.rows[c].bytes_per_row, bytes);
        body.insert(body.end(), data.begin(), data.end());
        data_bytes += bytes;
    }
    return body;
}
std::vector<std::byte> fixed_body(const strix::RtxStateLayout& l, std::uint64_t p, const std::vector<std::byte>& payload, std::size_t f,
    std::uint64_t& data_bytes) {
    std::vector<std::byte> body;
    put32(body, 1); put32(body, static_cast<std::uint32_t>(f)); put32(body, 0); put32(body, 0);
    put64(body, l.fixed[f].bytes);
    const auto data = slice(payload, l.rows_bytes(p) + l.fixed[f].offset, l.fixed[f].bytes);
    body.insert(body.end(), data.begin(), data.end());
    data_bytes += l.fixed[f].bytes;
    return body;
}

void test_layout() {
    auto l = small_layout();
    l.validate();
    const std::uint64_t p = 5;
    require(l.component_rows(0, p) == 5 && l.component_rows(2, p) == 4, "rows per component");
    require(l.component_offset(1, p) == 80 && l.component_offset(2, p) == 100 && l.component_offset(3, p) == 132, "component offsets");
    require(l.rows_bytes(p) == 140 && l.payload_bytes(p) == 1164, "payload bytes");
    require(l.find_row("mtp.kv") == std::optional<std::size_t>(2) && !l.find_row("nope"), "find_row");
    auto bad = l; bad.row_bytes = 31; expect_throw([&] { bad.validate(); }, "row bytes sum");
    bad = l; bad.fixed[1].offset = 300; expect_throw([&] { bad.validate(); }, "unaligned fixed offset");
    bad = l; bad.fixed_bytes = 1280; expect_throw([&] { bad.validate(); }, "fixed bytes sum");
    bad = l; bad.rows[1].id = "kv.L3"; expect_throw([&] { bad.validate(); }, "duplicate ids");
    auto other = l; other.rows[0].shift = 1;
    require(strix::rtx_state_geometry_identity(l) != strix::rtx_state_geometry_identity(other), "geometry identity covers shifts");
    require(strix::rtx_parse_u64("0xfeedfacecafebeef") == 0xfeedfacecafebeefULL && strix::rtx_parse_u64("18446744073709551615") == UINT64_MAX,
        "u64 parse");
    expect_throw([] { (void)strix::rtx_parse_u64("18446744073709551616"); }, "u64 overflow");
    expect_throw([] { (void)strix::rtx_parse_u64("12a"); }, "u64 junk");
    require(strix::rtx_hex_u64(0xfeedfacecafebeefULL) == "0xfeedfacecafebeef", "hex form");
}

void test_staging() {
    const auto layout = std::make_shared<const strix::RtxStateLayout>(small_layout());
    const auto& l = *layout;
    const std::uint64_t p = 5;
    const auto payload = canonical_payload(l, p);
    // Begin checks.
    auto r = begin_request(l, p);
    r.layout_identity ^= 1U;
    expect_throw([&] { strix::RtxStateStaging s(layout, r, "x", heap); }, "layout identity");
    r = begin_request(l, p); r.tokens.pop_back();
    expect_throw([&] { strix::RtxStateStaging s(layout, r, "x", heap); }, "token count");
    r = begin_request(l, p); r.producer.reduction = 0;
    expect_throw([&] { strix::RtxStateStaging s(layout, r, "x", heap); }, "zero producer id");
    r = begin_request(l, 64);
    expect_throw([&] { strix::RtxStateStaging s(layout, r, "x", heap); }, "position at context");
    r = begin_request(l, p); r.geometry_identity = 1;
    expect_throw([&] { strix::RtxStateStaging s(layout, r, "x", heap); }, "geometry identity");

    strix::RtxStateStaging s(layout, begin_request(l, p), "x", heap);
    require(s.payload_bytes() == 1164 && s.covered_bytes() == 1024 - 300 - 8 - 40, "padding is pre-covered");
    std::vector<strix::ModelContentIdentity> digests;
    std::uint64_t data_bytes = 0;
    const auto chunk = [&](const std::vector<std::byte>& body) {
        const auto segments = strix::parse_rtx_chunk_segments(s, body);
        strix::RtxStateChunkRecord record;
        record.seq = digests.size();
        record.digest = strix::rtx_chunk_digest(body);
        s.write_chunk(segments, record);
        digests.push_back(record.digest);
    };
    // Out-of-order chunks: fixed first, then rows 3.., then rows 0..2.
    chunk(fixed_body(l, p, payload, 2, data_bytes));
    chunk(rows_body(l, p, payload, 3, 10, data_bytes));
    // Bounds: rows beyond the component; a fixed range beyond the component; a malformed segment.
    {
        std::vector<std::byte> body; put32(body, 0); put32(body, 2); put32(body, 4); put32(body, 1); put64(body, 8);
        body.resize(body.size() + 8);
        expect_throw([&] { (void)strix::parse_rtx_chunk_segments(s, body); }, "mtp row 4 does not exist at p = 5");
        std::vector<std::byte> fixed; put32(fixed, 1); put32(fixed, 1); put32(fixed, 4); put32(fixed, 0); put64(fixed, 8);
        fixed.resize(fixed.size() + 8);
        expect_throw([&] { (void)strix::parse_rtx_chunk_segments(s, fixed); }, "fixed range beyond its component");
        std::vector<std::byte> wrong; put32(wrong, 0); put32(wrong, 0); put32(wrong, 0); put32(wrong, 2); put64(wrong, 16);
        wrong.resize(wrong.size() + 16);
        expect_throw([&] { (void)strix::parse_rtx_chunk_segments(s, wrong); }, "row count and byte count disagree");
        std::vector<std::byte> truncated; put32(truncated, 0); put32(truncated, 0);
        expect_throw([&] { (void)strix::parse_rtx_chunk_segments(s, truncated); }, "truncated header");
    }
    // Overlap with written bytes (all or none: the first segment is new, the second overlaps) and within one chunk.
    {
        const auto before = s.covered_bytes();
        std::uint64_t ignored = 0;
        auto body = rows_body(l, p, payload, 0, 1, ignored);
        const auto again = rows_body(l, p, payload, 3, 1, ignored);
        body.insert(body.end(), again.begin(), again.end());
        strix::RtxStateChunkRecord record; record.seq = 99; record.digest = strix::rtx_chunk_digest(body);
        expect_throw([&] { s.write_chunk(strix::parse_rtx_chunk_segments(s, body), record); }, "overlap with written rows");
        require(s.covered_bytes() == before, "a rejected chunk writes nothing");
        auto twice = rows_body(l, p, payload, 0, 1, ignored);
        const auto copy = twice;
        twice.insert(twice.end(), copy.begin(), copy.end());
        record.digest = strix::rtx_chunk_digest(twice);
        expect_throw([&] { s.write_chunk(strix::parse_rtx_chunk_segments(s, twice), record); }, "overlap within a chunk");
        // Mixing receive order with sequence numbers.
        strix::RtxStateChunkRecord unordered; unordered.digest = record.digest;
        expect_throw([&] { s.write_chunk(strix::parse_rtx_chunk_segments(s, rows_body(l, p, payload, 0, 1, ignored)), unordered); },
            "mixed ordering");
    }
    // Commit before complete fails and leaves it open.
    expect_throw([&] { (void)s.commit({digests.size(), data_bytes, strix::rtx_manifest_digest(digests), false}); }, "incomplete");
    chunk(rows_body(l, p, payload, 0, 3, data_bytes));
    chunk(fixed_body(l, p, payload, 0, data_bytes));
    chunk(fixed_body(l, p, payload, 1, data_bytes));
    require(s.covered_bytes() == s.payload_bytes(), "covered");
    // Wrong manifest, wrong counts.
    auto shuffled = digests; std::swap(shuffled[0], shuffled[1]);
    expect_throw([&] { (void)s.commit({digests.size(), data_bytes, strix::rtx_manifest_digest(shuffled), false}); }, "manifest order");
    expect_throw([&] { (void)s.commit({digests.size() + 1U, data_bytes, strix::rtx_manifest_digest(digests), false}); }, "chunk count");
    expect_throw([&] { (void)s.commit({digests.size(), data_bytes + 1U, strix::rtx_manifest_digest(digests), false}); }, "byte count");
    require(!s.committed(), "still open");
    const auto result = s.commit({digests.size(), data_bytes, strix::rtx_manifest_digest(digests), true});
    require(s.committed() && result.canonical && *result.canonical == strix::sha256_bytes(payload), "canonical sha of the staged payload");
    require(std::equal(s.payload().begin(), s.payload().end(), payload.begin()), "payload bytes");
    require(s.component(2).size() == 32 && s.component(2)[0] == payload[100], "component view");
    require(s.fixed_blob().size() == 1024 && s.fixed_blob()[300] == std::byte{0}, "fixed view with zero padding");
    expect_throw([&] { s.write_chunk(strix::parse_rtx_chunk_segments(s, rows_body(l, p, payload, 0, 1, data_bytes)), {}); }, "after commit");
}

// Cooperative (local_rows q): rows [0, q) of every component are pre-covered (zero), only rows [q, p) and the fixed
// state are sent; a row chunk below q overlaps.
void test_staging_local_rows() {
    const auto layout = std::make_shared<const strix::RtxStateLayout>(small_layout());
    const auto& l = *layout;
    const std::uint64_t p = 5, q = 3;
    const auto payload = canonical_payload(l, p);
    auto r = begin_request(l, p); r.local_rows = p;
    expect_throw([&] { strix::RtxStateStaging s(layout, r, "x", heap); }, "local_rows at position");
    r = begin_request(l, p); r.local_rows = q;
    strix::RtxStateStaging s(layout, r, "x", heap);
    std::vector<strix::ModelContentIdentity> digests;
    std::uint64_t data_bytes = 0;
    const auto chunk = [&](const std::vector<std::byte>& body) {
        strix::RtxStateChunkRecord record;
        record.seq = digests.size();
        record.digest = strix::rtx_chunk_digest(body);
        s.write_chunk(strix::parse_rtx_chunk_segments(s, body), record);
        digests.push_back(record.digest);
    };
    {
        std::uint64_t ignored = 0;
        const auto below = rows_body(l, p, payload, q - 1U, 1, ignored);
        strix::RtxStateChunkRecord record; record.seq = 0; record.digest = strix::rtx_chunk_digest(below);
        expect_throw([&] { s.write_chunk(strix::parse_rtx_chunk_segments(s, below), record); }, "a row below local_rows");
    }
    // The MTP components (shift 1) have rows [0, q - 1) local: their row q - 1 is sent.
    chunk(rows_body(l, p, payload, q, 10, data_bytes));
    {
        std::vector<std::byte> body;
        for (std::uint32_t c = 2; c < 4U; ++c) {
            const auto bpr = l.rows[c].bytes_per_row;
            put32(body, 0); put32(body, c); put32(body, static_cast<std::uint32_t>(q - 1U)); put32(body, 1); put64(body, bpr);
            const auto data = slice(payload, l.component_offset(c, p) + (q - 1U) * bpr, bpr);
            body.insert(body.end(), data.begin(), data.end());
            data_bytes += bpr;
        }
        chunk(body);
    }
    for (std::size_t f = 0; f < l.fixed.size(); ++f) chunk(fixed_body(l, p, payload, f, data_bytes));
    require(s.covered_bytes() == s.payload_bytes(), "rows [q, p) + fixed cover the staging");
    (void)s.commit({digests.size(), data_bytes, strix::rtx_manifest_digest(digests), false});
    for (std::size_t c = 0; c < l.rows.size(); ++c) {
        const auto view = s.component(c);
        const auto local = l.component_rows(c, q) * l.rows[c].bytes_per_row;
        require(std::all_of(view.begin(), view.begin() + static_cast<std::ptrdiff_t>(local), [](std::byte b) { return b == std::byte{0}; }),
            "local rows are zero");
        const auto at = l.component_offset(c, p);
        require(std::equal(view.begin() + static_cast<std::ptrdiff_t>(local), view.end(), payload.begin() + static_cast<std::ptrdiff_t>(at + local)),
            "sent rows");
    }
}

void test_registry() {
    strix::RtxStateImportRegistry registry(heap, 2);
    const auto l = small_layout();
    expect_throw([&] { (void)registry.begin(begin_request(l, 3)); }, "begin before the layout");
    registry.set_layout(l);
    expect_throw([&] { registry.set_layout(l); }, "layout published once");
    const std::uint64_t p = 3;
    const auto payload = canonical_payload(l, p);
    const auto fill = [&](const std::string& id) {
        auto s = registry.find(id);
        std::vector<strix::ModelContentIdentity> digests;
        std::uint64_t bytes = 0;
        for (auto body : {rows_body(l, p, payload, 0, 64, bytes), fixed_body(l, p, payload, 0, bytes), fixed_body(l, p, payload, 1, bytes),
                 fixed_body(l, p, payload, 2, bytes)}) {
            strix::RtxStateChunkRecord record; record.digest = strix::rtx_chunk_digest(body);
            s->write_chunk(strix::parse_rtx_chunk_segments(*s, body), record);
            digests.push_back(record.digest);
        }
        return strix::RtxStateCommitRequest{digests.size(), bytes, strix::rtx_manifest_digest(digests), false};
    };
    const auto a = registry.begin(begin_request(l, p));
    const auto b = registry.begin(begin_request(l, p));
    expect_throw([&] { (void)registry.begin(begin_request(l, p)); }, "in-flight limit");
    require(registry.abort(b) && !registry.find(b) && !registry.abort(b), "abort");
    (void)registry.commit(a, fill(a));
    require(!registry.find(a), "committed leaves the in-flight set");
    std::vector<std::uint32_t> prompt = {100, 101, 102};
    require(!registry.take(prompt), "a prompt as long as the state leaves no row to run");
    prompt = {100, 101, 7, 8};
    require(!registry.take(prompt), "other tokens");
    prompt = {100, 101, 102, 103, 104};
    const auto taken = registry.take(prompt);
    require(taken && taken->position() == p && !registry.take(prompt), "one-shot take");
    auto keep = begin_request(l, p); keep.keep = true;
    const auto c = registry.begin(keep);
    (void)registry.commit(c, fill(c));
    require(registry.take(prompt) && registry.take(prompt), "keep stays parked");
    require(registry.abort(c) && !registry.take(prompt), "abort the parked state");
    const auto st = registry.stats();
    require(st.begun == 3 && st.committed == 2 && st.taken == 3 && st.misses == 2 && st.aborted == 2, "stats");
}

void test_file() {
    const auto l = small_layout();
    const std::uint64_t p = 4;
    const auto payload = canonical_payload(l, p);
    const auto path = std::filesystem::temp_directory_path() / "strix-rtx-state-test.rtxs";
    std::vector<std::uint32_t> tokens = {5, 6, 7, 8};
    strix::RtxStateFileSession session; session.next_position = p; session.committed_tokens = p; session.generations = {48, 144, 148, 48, 4, 4};
    session.lineage.push_back({0, 4, 1, 0xfeedfacecafebeefULL, 2, 3});
    const auto sha = strix::write_rtx_state_file(path, l, tokens, p, {0xfeedfacecafebeefULL, 2, 3}, session, payload);
    require(sha == strix::sha256_bytes(payload), "file sha");
    const auto view = strix::read_rtx_state_file_header(path);
    require(view.payload_bytes == payload.size(), "file payload size");
    const auto header = strix::serve::parse_json(view.header_json);
    require(header.find("format")->as_string() == "strix-qwen-rtx-state-v1" && header.find("position")->as_int() == 4, "file header");
    require(header.find("layout_identity")->as_string() == "0xfeedfacecafebeef", "u64 identities as strings");
    require(header.find("sha256")->as_string() == strix::model_content_identity_hex(sha), "file header sha");
    require(header.find("context_tokens")->as_int() == 64 &&
        header.find("rotary")->find("rope_mode")->as_string() == "native", "native file capacity and rotary");
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(view.payload_offset));
    std::vector<char> back(payload.size());
    in.read(back.data(), static_cast<std::streamsize>(back.size()));
    require(std::memcmp(back.data(), payload.data(), payload.size()) == 0, "file payload");
    in.close();
    std::filesystem::remove(path);
}

struct Writer final : strix::serve::HttpResponseWriter {
    int status = 0;
    std::string body;
    void send(int s, std::string_view, std::string_view b, const std::vector<std::pair<std::string, std::string>>&) override { status = s; body = b; }
    bool begin_stream(int, std::string_view, const std::vector<std::pair<std::string, std::string>>&) override { return false; }
    bool write(std::string_view) override { return false; }
    void end_stream() override {}
    [[nodiscard]] bool client_gone() const override { return false; }
};

Writer call(strix::RtxStateImportRegistry& registry, std::string method, std::string path, std::string query, std::string body,
    std::vector<std::pair<std::string, std::string>> headers = {}) {
    strix::serve::HttpRequest request;
    request.method = std::move(method); request.path = std::move(path); request.query = std::move(query); request.body = std::move(body);
    request.headers = std::move(headers);
    Writer w;
    require(strix::decode_cli::handle_rtx_state_import_route(registry, request, w), "route handled");
    return w;
}

// Identity admission is checked before staging allocation; large geometry needs no device.
void test_rotary_import() {
    using namespace strix;
    const auto native = small_layout();
    require(rtx_state_geometry_identity(native) == 0xf019966b9ba456deULL, "legacy native geometry identity pinned");
    auto native_capacity = native;
    native_capacity.context_tokens = 262144;
    require(rtx_state_geometry_identity(native_capacity) == 0xf019966b9ba456deULL, "native capacity preserves legacy identity");
    const auto table = rtx_parse_sha256(std::string(64, '1')); // synthetic CPU fixture, not a generated table hash
    const auto other_table = rtx_parse_sha256(std::string(64, '2'));
    std::vector<std::uint64_t> identities;
    for (const auto factor : {2U, 4U}) {
        auto l = native;
        l.context_tokens = 262144ULL * factor;
        l.rotary = RotaryIdentity(RotaryDescriptor::yarn(factor), table);
        l.validate();
        const auto identity = rtx_state_geometry_identity(l);
        require(identity != rtx_state_geometry_identity(native), "YaRN differs from native");
        identities.push_back(identity);
        auto changed = l;
        --changed.context_tokens;
        require(rtx_state_geometry_identity(changed) != identity, "YaRN capacity binds geometry");
        changed = l;
        changed.rotary = RotaryIdentity(RotaryDescriptor::yarn(factor), other_table);
        require(rtx_state_geometry_identity(changed) != identity, "YaRN table binds geometry");
        changed.rotary = RotaryIdentity(RotaryDescriptor::yarn(factor == 2 ? 4 : 2), table);
        require(rtx_state_geometry_identity(changed) != identity, "YaRN factor binds geometry");
        const auto position = l.context_tokens - 1;
        require(l.rows_bytes(position) == position * 30 - 10 &&
            l.payload_bytes(position) == position * 30 + 1014, "long context payload arithmetic");
        auto wide = l;
        wide.rows[0].bytes_per_row = 65536;
        wide.row_bytes += 65536 - 16;
        wide.validate();
        require(wide.component_offset(wide.rows.size(), position) == position * wide.row_bytes - 10 &&
            wide.payload_bytes(position) == position * wide.row_bytes + 1014 &&
            wide.payload_bytes(position) > UINT32_MAX, "long payload retains 64-bit offsets");
        expect_throw([&] { (void)wide.payload_bytes(UINT64_MAX); }, "payload overflow refused");
        auto beyond_table = l; ++beyond_table.context_tokens;
        expect_throw([&] { beyond_table.validate(); }, "capacity beyond rotary table refused");
        auto layout = std::make_shared<const RtxStateLayout>(l);
        auto request = begin_request(l, position);
        request.rotary = l.rotary;
        request.context_tokens = l.context_tokens;
        { RtxStateStaging staging(layout, request, "long", heap);
          require(staging.payload_bytes() == l.payload_bytes(position), "C-1 accepted"); }
        const auto refused = [&](RtxStateBeginRequest bad, const std::string& message) {
            bool allocated = false;
            try {
                RtxStateStaging staging(layout, std::move(bad), "bad", [&](std::uint64_t bytes) {
                    allocated = true;
                    return heap(bytes);
                });
            } catch (const std::invalid_argument& e) {
                require(!allocated && std::string(e.what()).find(message) != std::string::npos, "clear pre-allocation refusal");
                return;
            }
            throw std::runtime_error("expected import refusal: " + message);
        };
        auto bad = request; bad.rotary = RotaryIdentity{}; refused(bad, "rotary identity mismatch");
        bad = request; bad.rotary = changed.rotary; refused(bad, "rotary identity mismatch");
        bad = request; bad.rotary = RotaryIdentity(RotaryDescriptor::yarn(factor), other_table);
        refused(bad, "rotary identity mismatch");
        bad = request; bad.context_tokens = l.context_tokens - 1; refused(bad, "context capacity mismatch");
        bad = request; bad.rotary.reset(); bad.context_tokens.reset(); refused(bad, "requires geometry_identity");
        bad.geometry_identity = rtx_state_geometry_identity(native); refused(bad, "geometry identity differs");
        bad = request; bad.position = 0; bad.tokens.clear(); refused(bad, "1 <= position < context");
        bad = request; ++bad.position; bad.tokens.push_back(1); refused(bad, "1 <= position < context");
        RtxStateImportRegistry registry(heap, 2);
        registry.set_layout(l);
        const auto published = serve::parse_json(call(registry, "GET", "/admin/rtx-import/layout", "", "").body);
        require(published.find("context_tokens")->as_int() == static_cast<std::int64_t>(l.context_tokens) &&
            published.find("rotary")->dump() == serve::parse_json(l.rotary.json()).dump(), "published capacity and resolved rotary");
        const auto body = [&](const RotaryIdentity& rotary, std::uint64_t capacity) {
            return "{\"tokens\":[1],\"position\":1,\"producer\":{\"numerical\":1,\"reduction\":2,\"state_update\":3},"
                "\"layout_identity\":\"" + rtx_hex_u64(l.layout_identity) + "\",\"context_tokens\":" + std::to_string(capacity) +
                ",\"rotary\":" + rotary.json() + "}";
        };
        auto response = call(registry, "POST", "/admin/rtx-import/begin", "", body(l.rotary, l.context_tokens));
        require(response.status == 200, "HTTP matching YaRN identity: " + response.body);
        require(call(registry, "POST", "/admin/rtx-import/begin", "", body(changed.rotary, l.context_tokens)).status == 400,
            "HTTP mismatched factor refused");
        require(call(registry, "POST", "/admin/rtx-import/begin", "", body(l.rotary, l.context_tokens - 1)).status == 400,
            "HTTP mismatched capacity refused");
        require(call(registry, "POST", "/admin/rtx-import/begin", "", body(RotaryIdentity{}, l.context_tokens)).status == 400,
            "HTTP native producer refused on YaRN");
        require(call(registry, "POST", "/admin/rtx-import/begin", "",
            body(RotaryIdentity(RotaryDescriptor::yarn(factor), other_table), l.context_tokens)).status == 400,
            "HTTP mismatched table refused");
        auto malformed = serve::parse_json(body(l.rotary, l.context_tokens));
        auto rotary_json = serve::parse_json(l.rotary.json());
        rotary_json.set("original_max", serve::Json(131072));
        malformed.set("rotary", rotary_json);
        require(call(registry, "POST", "/admin/rtx-import/begin", "", malformed.dump()).status == 400,
            "HTTP mismatched original context refused");
        const auto path = std::filesystem::temp_directory_path() / "strix-rtx-yarn-state-test.rtxs";
        (void)write_rtx_state_file(path, l, std::vector<std::uint32_t>{1}, 1, {1, 2, 3}, {}, canonical_payload(l, 1));
        const auto file = serve::parse_json(read_rtx_state_file_header(path).header_json);
        require(file.find("rotary")->dump() == serve::parse_json(l.rotary.json()).dump() &&
            file.find("context_tokens")->as_int() == static_cast<std::int64_t>(l.context_tokens), "YaRN file preserves producer facts");
        std::filesystem::remove(path);
        auto proof = begin_request(l, 1); proof.geometry_identity = identity;
        RtxStateStaging staging(layout, proof, "geometry-proof", heap);
        auto native_request = begin_request(native, 1); native_request.rotary = l.rotary;
        expect_throw([&] { RtxStateStaging rejected(std::make_shared<const RtxStateLayout>(native), native_request, "native", heap); },
            "YaRN producer refused on native");
    }
    require(identities[0] != identities[1], "f2/f4 route identities differ");
}

void test_file_pair() {
    using namespace strix;
    using namespace strix::decode_cli;
    const auto path = std::filesystem::temp_directory_path() / "strix-rtx-pair.json";
    auto binary = path; binary.replace_extension(".bin");
    const auto clean = std::shared_ptr<void>(nullptr, [&](void*) { std::filesystem::remove(path); std::filesystem::remove(binary); });
    auto l = small_layout();
    l.context_tokens = 524288;
    auto table = rtx_parse_sha256(std::string(64, '1'));
    l.rotary = RotaryIdentity(RotaryDescriptor::yarn(2), table);
    const auto payload = canonical_payload(l, 5);
    auto header = serve::Json::object();
    auto tokens = serve::Json::array();
    for (unsigned i = 100; i < 105; ++i) tokens.push(serve::Json(static_cast<std::uint64_t>(i)));
    header.set("tokens", std::move(tokens)); header.set("position", serve::Json(std::uint64_t{5}));
    auto producer = serve::Json::object();
    for (const auto* key : {"numerical", "reduction", "state_update"}) producer.set(key, serve::Json("11"));
    header.set("producer", std::move(producer));
    header.set("layout_identity", serve::Json(rtx_hex_u64(l.layout_identity)));
    header.set("geometry_identity", serve::Json(rtx_hex_u64(rtx_state_geometry_identity(l))));
    header.set("context_tokens", serve::Json(l.context_tokens));
    header.set("rotary", serve::parse_json(l.rotary.json()));
    header.set("encoding", serve::Json("identity"));
    header.set("payload_bytes", serve::Json(static_cast<std::uint64_t>(payload.size())));
    header.set("payload_sha256", serve::Json(model_content_identity_hex(sha256_bytes(payload))));
    const auto save = [&] {
        std::ofstream json(path); json << header.dump();
        std::ofstream bin(binary, std::ios::binary);
        bin.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    };
    const std::vector<std::uint32_t> prompt{100,101,102,103,104,105};
    RtxStateImportRegistry chunks(heap, 1, 0), one(heap, 1, 0);
    chunks.set_layout(l); one.set_layout(l); save();
    require(parse_rtx_state_begin(read_rtx_state_import_metadata(path, prompt)).position == 5,
        "CPU metadata preflight establishes the PLE suffix boundary");
    expect_throw([&] { stage_rtx_state_import_file(chunks, path, prompt, 32, 4); },
        "staging refuses a changed position after PLE admission");
    require(chunks.status().empty(), "changed boundary refuses before staging storage");
    stage_rtx_state_import_file(chunks, path, prompt, 32, 5);
    const auto id = one.begin(parse_rtx_state_begin(header));
    const auto staging = one.find(id);
    std::uint64_t data_bytes = 0;
    auto wire = rows_body(l, 5, payload, 0, 5, data_bytes);
    for (std::size_t f = 0; f < l.fixed.size(); ++f) {
        const auto fixed = fixed_body(l, 5, payload, f, data_bytes);
        wire.insert(wire.end(), fixed.begin(), fixed.end());
    }
    RtxStateChunkRecord record; record.seq = 0; record.digest = rtx_chunk_digest(wire);
    staging->write_chunk(parse_rtx_chunk_segments(*staging, wire), record);
    const std::vector<ModelContentIdentity> manifest{record.digest};
    one.commit(id, {1, data_bytes, rtx_manifest_digest(manifest), false});
    const auto a = chunks.take(prompt), b = one.take(prompt);
    require(a && b && a->chunks() > b->chunks(), "file chunk size exercised");
    require(std::equal(a->payload().begin(), a->payload().end(), b->payload().begin()), "chunked and one-shot staged bytes equal");
    require(std::equal(a->payload().begin(), a->payload().end(), payload.begin()), "file canonical bytes preserved");
    RtxStateImportRegistry rejected(heap, 1, 0); rejected.set_layout(l);
    auto wrong = prompt; wrong[0] = 99;
    expect_throw([&] { stage_rtx_state_import_file(rejected, path, wrong, 32); }, "file prefix mismatch");
    header.set("payload_sha256", serve::Json(std::string(64, '0'))); save();
    expect_throw([&] { stage_rtx_state_import_file(rejected, path, prompt, 32); }, "file SHA mismatch");
    require(rejected.status().empty(), "failed SHA aborts staging");
    header.set("payload_sha256", serve::Json(model_content_identity_hex(sha256_bytes(payload))));
    header.set("context_tokens", serve::Json(std::uint64_t{1048576})); save();
    expect_throw([&] { stage_rtx_state_import_file(rejected, path, prompt, 32); }, "file capacity mismatch");
    header.set("context_tokens", serve::Json(l.context_tokens));
    table = rtx_parse_sha256(std::string(64, '2'));
    header.set("rotary", serve::parse_json(RotaryIdentity(RotaryDescriptor::yarn(2), table).json())); save();
    expect_throw([&] { stage_rtx_state_import_file(rejected, path, prompt, 32); }, "file rotary mismatch");
    header.set("rotary", serve::parse_json(l.rotary.json()));
    header.set("geometry_identity", serve::Json("1")); save();
    expect_throw([&] { stage_rtx_state_import_file(rejected, path, prompt, 32); }, "file geometry mismatch");
}

void test_routes() {
    strix::RtxStateImportRegistry registry(heap, 2);
    strix::serve::HttpRequest other; other.method = "GET"; other.path = "/v1/models";
    Writer unused;
    require(!strix::decode_cli::handle_rtx_state_import_route(registry, other, unused), "other paths pass through");
    require(call(registry, "GET", "/admin/rtx-import/layout", "", "").status == 503, "layout before publish");
    const auto l = small_layout();
    registry.set_layout(l);
    const auto layout = strix::serve::parse_json(call(registry, "GET", "/admin/rtx-import/layout", "", "").body);
    require(layout.find("layout_identity")->as_string() == "0xfeedfacecafebeef" && layout.find("row_bytes")->as_int() == 30 &&
        layout.find("max_context")->as_int() == 64 && layout.find("rows")->as_array()[2].find("id")->as_int() == 2 &&
        layout.find("rows")->as_array()[2].find("name")->as_string() == "mtp.kv", "layout JSON");
    const std::uint64_t p = 6;
    const auto payload = canonical_payload(l, p);
    std::string begin = "{\"tokens\":[100,101,102,103,104,105],\"position\":6,\"producer\":{\"numerical\":\"18446744073709551615\","
        "\"reduction\":\"0x2\",\"state_update\":3},\"layout_identity\":\"" + std::to_string(l.layout_identity) + "\"}";
    auto w = call(registry, "POST", "/admin/rtx-import/begin", "", begin);
    require(w.status == 200, "begin: " + w.body);
    const auto id = strix::serve::parse_json(w.body).find("id")->as_string();
    require(call(registry, "POST", "/admin/rtx-import/begin", "", "{\"tokens\":[1],\"position\":1,\"producer\":{\"numerical\":1,"
        "\"reduction\":1,\"state_update\":1},\"layout_identity\":\"1\"}").status == 400, "begin with another layout");
    std::vector<std::string> digests;
    std::vector<strix::ModelContentIdentity> raw;
    std::uint64_t bytes = 0, seq = 0;
    const auto post = [&](const std::string& route, const std::string& query, const std::vector<std::byte>& body, bool good_sha = true) {
        auto digest = strix::rtx_chunk_digest(body);
        const auto hex = strix::model_content_identity_hex(digest);
        auto sent = hex; if (!good_sha) sent[0] = sent[0] == '0' ? '1' : '0';
        auto result = call(registry, "POST", "/admin/rtx-import/" + route, "id=" + id + query,
            std::string(reinterpret_cast<const char*>(body.data()), body.size()), {{"x-chunk-sha256", sent}, {"x-chunk-seq", std::to_string(seq)}});
        if (result.status == 200) { raw.push_back(digest); ++seq; }
        return result;
    };
    // Multi-segment rows of the first 4 positions; a single-component chunk for the rest of kv.L3 by name.
    require(post("chunks", "", rows_body(l, p, payload, 0, 4, bytes)).status == 200, "chunks");
    std::uint64_t ignored = 0;
    require(post("chunks", "", rows_body(l, p, payload, 4, 1, ignored), false).status == 400, "bad chunk sha rejected");
    const auto kv_rest = slice(payload, 4 * 16, 2 * 16);
    w = post("chunk", "&component=kv.L3&first_row=4", std::vector<std::byte>(kv_rest.begin(), kv_rest.end()));
    require(w.status == 200, "single rows chunk: " + w.body);
    bytes += kv_rest.size();
    w = post("chunk", "&component=kv.L3&first_row=4", std::vector<std::byte>(kv_rest.begin(), kv_rest.end()));
    require(w.status == 400, "duplicate rows rejected");
    for (std::size_t c = 1; c < l.rows.size(); ++c) {
        const auto rows = l.component_rows(c, p);
        if (rows <= 4) continue;
        const auto d = slice(payload, l.component_offset(c, p) + 4 * l.rows[c].bytes_per_row, (rows - 4) * l.rows[c].bytes_per_row);
        require(post("chunk", "&component=" + std::to_string(c) + "&first_row=4", std::vector<std::byte>(d.begin(), d.end())).status == 200,
            "rows by index");
        bytes += d.size();
    }
    // The fixed state: recurrent_matrix in two halves by name, the rest as segments.
    for (const std::uint64_t half : {0U, 150U}) {
        const auto d = slice(payload, l.rows_bytes(p) + half, 150);
        require(post("chunk", "&fixed=recurrent_matrix&offset=" + std::to_string(half), std::vector<std::byte>(d.begin(), d.end())).status == 200,
            "fixed by name");
        bytes += 150;
    }
    auto both = fixed_body(l, p, payload, 1, bytes);
    const auto held = fixed_body(l, p, payload, 2, bytes);
    both.insert(both.end(), held.begin(), held.end());
    require(post("chunks", "", both).status == 200, "fixed segments");
    const auto status = strix::serve::parse_json(call(registry, "GET", "/admin/rtx-import/status", "", "").body);
    require(status.find("imports")->as_array()[0].find("covered_bytes")->as_int() == static_cast<std::int64_t>(l.payload_bytes(p)), "status");
    w = call(registry, "POST", "/admin/rtx-import/commit", "id=" + id, "{\"chunks\":" + std::to_string(raw.size()) + ",\"bytes\":" +
        std::to_string(bytes + 1U) + ",\"manifest_sha256\":\"" + strix::model_content_identity_hex(strix::rtx_manifest_digest(raw)) + "\"}");
    require(w.status == 400, "commit with the wrong byte count");
    w = call(registry, "POST", "/admin/rtx-import/commit", "id=" + id, "{\"chunks\":" + std::to_string(raw.size()) + ",\"bytes\":" +
        std::to_string(bytes) + ",\"manifest_sha256\":\"" + strix::model_content_identity_hex(strix::rtx_manifest_digest(raw)) +
        "\",\"canonical_sha256\":true}");
    require(w.status == 200, "commit: " + w.body);
    require(strix::serve::parse_json(w.body).find("canonical_sha256")->as_string() == strix::model_content_identity_hex(strix::sha256_bytes(payload)),
        "canonical sha diagnostic");
    require(call(registry, "POST", "/admin/rtx-import/commit", "id=" + id, "{}").status == 404, "commit twice");
    std::vector<std::uint32_t> prompt = {100, 101, 102, 103, 104, 105, 106};
    const auto taken = registry.take(prompt);
    require(taken && taken->request().producer.numerical == UINT64_MAX && taken->request().producer.reduction == 2, "decimal and hex ids");
    require(call(registry, "POST", "/admin/rtx-import/abort", "id=nope", "").status == 404, "abort unknown");
    require(call(registry, "GET", "/admin/rtx-import/nope", "", "").status == 404, "unknown route");
}

// bp-v1 encoded chunks: advertised codecs, negotiation at begin, digest over the frame, decoded segments checked
// as plain ones, malformed frames and unknown encodings rejected, plain chunks still accepted in an encoded import.
void test_routes_codec() {
    strix::RtxStateImportRegistry registry(heap, 2, 2);
    const auto l = small_layout();
    registry.set_layout(l);
    const auto layout = strix::serve::parse_json(call(registry, "GET", "/admin/rtx-import/layout", "", "").body);
    const auto* codecs = layout.find("codecs");
    require(codecs && codecs->as_array().size() == 1U && codecs->as_array()[0].as_string() == "bp-v1", "layout advertises bp-v1");
    const std::uint64_t p = 6;
    const auto payload = canonical_payload(l, p);
    const auto begin_json = [&](const std::string& extra) {
        return "{\"tokens\":[100,101,102,103,104,105],\"position\":6,\"producer\":{\"numerical\":1,\"reduction\":2,\"state_update\":3},"
               "\"layout_identity\":\"" + std::to_string(l.layout_identity) + "\"" + extra + "}";
    };
    require(call(registry, "POST", "/admin/rtx-import/begin", "", begin_json(",\"encoding\":\"zstd\"")).status == 400, "unknown codec at begin");
    auto w = call(registry, "POST", "/admin/rtx-import/begin", "", begin_json(""));
    require(w.status == 200 && strix::serve::parse_json(w.body).find("encoding")->as_string() == "identity", "plain begin: " + w.body);
    const auto plain_id = strix::serve::parse_json(w.body).find("id")->as_string();
    w = call(registry, "POST", "/admin/rtx-import/begin", "", begin_json(",\"encoding\":\"bp-v1\""));
    require(w.status == 200 && strix::serve::parse_json(w.body).find("encoding")->as_string() == "bp-v1", "encoded begin: " + w.body);
    const auto id = strix::serve::parse_json(w.body).find("id")->as_string();

    const auto as_u8 = [](const std::vector<std::byte>& v) { return reinterpret_cast<const std::uint8_t*>(v.data()); };
    const auto encode = [&](const std::vector<std::byte>& body, std::uint32_t width) {
        const std::vector<strix::bp_codec::Region> regions = {{as_u8(body), 24, 1}, {as_u8(body) + 24, body.size() - 24, width}};
        const auto encoded = strix::bp_codec::encode(regions);
        const auto* first = reinterpret_cast<const std::byte*>(encoded.data());
        return std::vector<std::byte>(first, first + encoded.size());
    };
    std::vector<strix::ModelContentIdentity> raw;
    std::uint64_t bytes = 0;
    const auto post = [&](const std::string& import, const std::vector<std::byte>& body, const std::string& encoding) {
        const auto digest = strix::rtx_chunk_digest(body);
        std::vector<std::pair<std::string, std::string>> headers = {{"x-chunk-sha256", strix::model_content_identity_hex(digest)},
            {"x-chunk-seq", std::to_string(raw.size())}};
        if (!encoding.empty()) headers.emplace_back("x-chunk-encoding", encoding);
        auto result = call(registry, "POST", "/admin/rtx-import/chunks", "id=" + import,
            std::string(reinterpret_cast<const char*>(body.data()), body.size()), headers);
        if (result.status == 200 && import == id) raw.push_back(digest);
        return result;
    };
    // Rows of every component as one encoded chunk.
    std::uint64_t row_bytes = 0;
    const auto rows = rows_body(l, p, payload, 0, p, row_bytes);
    const auto frame = encode(rows, 2);
    require(post(plain_id, frame, "bp-v1").status == 400, "an encoded chunk needs an import begun with the encoding");
    require(post(id, frame, "lz4").status == 400, "unknown X-Chunk-Encoding");
    require(post(id, frame, "").status == 400, "a frame without X-Chunk-Encoding is parsed as plain segments (and fails)");
    auto broken = frame;
    broken[broken.size() - 3] ^= std::byte{0x10};
    require(post(id, broken, "bp-v1").status == 400, "a corrupted frame (with its own digest) fails the block checksum");
    w = post(id, frame, "bp-v1");
    require(w.status == 200, "encoded rows chunk: " + w.body);
    const auto reply = strix::serve::parse_json(w.body);
    require(reply.find("decoded_bytes")->as_int() == static_cast<std::int64_t>(rows.size()) &&
        reply.find("wire_bytes")->as_int() == static_cast<std::int64_t>(frame.size()) && reply.find("decode_ms"), "encoded chunk reply");
    bytes += row_bytes;
    require(post(id, frame, "bp-v1").status == 400, "the same rows again overlap");
    // The fixed state: recurrent_matrix plain (identity named explicitly), the other two encoded in one chunk.
    const auto fixed0 = fixed_body(l, p, payload, 0, bytes);
    require(post(id, fixed0, "identity").status == 200, "plain chunk in an encoded import");
    auto both = fixed_body(l, p, payload, 1, bytes);
    const auto held = fixed_body(l, p, payload, 2, bytes);
    both.insert(both.end(), held.begin(), held.end());
    require(post(id, encode(both, 4), "bp-v1").status == 200, "encoded fixed chunk");
    w = call(registry, "POST", "/admin/rtx-import/commit", "id=" + id, "{\"chunks\":" + std::to_string(raw.size()) + ",\"bytes\":" +
        std::to_string(bytes) + ",\"manifest_sha256\":\"" + strix::model_content_identity_hex(strix::rtx_manifest_digest(raw)) + "\"}");
    require(w.status == 200, "encoded commit: " + w.body);
    require(strix::serve::parse_json(w.body).find("encoding")->as_string() == "bp-v1", "commit names the encoding");
    const std::vector<std::uint32_t> prompt = {100, 101, 102, 103, 104, 105, 106};
    const auto taken = registry.take(prompt);
    require(taken && std::equal(taken->payload().begin(), taken->payload().end(), payload.begin()), "encoded import payload is exact");
}

#if !defined(_WIN32)
// Serves the routes with the layout of a real state file until GET /stop (CPU only): the producer's codec path
// (tests/cuda/qwen_handoff_codec_test.cpp) streams the file here. GET /take checks the parked import byte for byte.
int serve_file(std::uint16_t port, const std::string& path, unsigned decode_threads) {
    const auto view = strix::read_rtx_state_file_header(path);
    const auto h = strix::serve::parse_json(view.header_json);
    strix::RtxStateLayout l;
    for (const auto& r : h.find("rows")->as_array())
        l.rows.push_back({r.find("id")->as_string(), static_cast<std::uint64_t>(r.find("bytes_per_row")->as_int()),
            static_cast<std::uint32_t>(r.find("shift")->as_int())});
    for (const auto& f : h.find("fixed")->as_array())
        l.fixed.push_back({f.find("id")->as_string(), static_cast<std::uint64_t>(f.find("bytes")->as_int()),
            static_cast<std::uint64_t>(f.find("offset")->as_int())});
    l.row_bytes = static_cast<std::uint64_t>(h.find("row_bytes")->as_int());
    l.fixed_bytes = static_cast<std::uint64_t>(h.find("fixed_bytes")->as_int());
    l.layout_identity = strix::rtx_parse_u64(h.find("layout_identity")->as_string());
    l.context_tokens = 262144;
    std::vector<std::uint32_t> prompt;
    for (const auto& t : h.find("tokens")->as_array()) prompt.push_back(static_cast<std::uint32_t>(t.as_int()));
    prompt.push_back(0);
    std::vector<std::byte> payload(static_cast<std::size_t>(view.payload_bytes));
    {
        std::ifstream in(path, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(view.payload_offset));
        in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
        require(static_cast<bool>(in), "state file payload is truncated");
    }
    strix::RtxStateImportRegistry registry(heap, 2, decode_threads);
    registry.set_layout(l);
    strix::serve::HttpServerConfig config;
    config.port = port;
    config.bind_address = "0.0.0.0";
    config.max_body_bytes = 256ULL << 20U;
    strix::serve::HttpServer server(config, [&](const strix::serve::HttpRequest& request, strix::serve::HttpResponseWriter& response) {
        if (request.path == "/take") {
            const auto taken = registry.take(prompt);
            const bool ok = taken && taken->payload().size() == payload.size() &&
                std::equal(taken->payload().begin(), taken->payload().end(), payload.begin());
            response.send(ok ? 200 : 409, "application/json", ok ? "{\"equal\":true}" : "{\"equal\":false}");
            return;
        }
        if (request.path == "/stop") {
            response.send(200, "text/plain", "stopping");
            std::thread([&server] { server.stop(); }).detach();
            return;
        }
        if (!strix::decode_cli::handle_rtx_state_import_route(registry, request, response)) response.send(404, "text/plain", "no");
    });
    server.listen();
    std::cerr << "serving " << path << " (" << payload.size() << " bytes, position " << prompt.size() - 1U << ") on " << server.port()
              << ", decode threads " << decode_threads << std::endl;
    server.serve();
    return 0;
}

// Loopback receive throughput: begin, `total` bytes of row chunks over one keep-alive connection, commit.
int bench_loopback(std::uint64_t total, std::uint64_t chunk) {
    strix::RtxStateLayout l;
    l.rows = {{"rows", 1U << 20U, 0}};
    l.row_bytes = 1U << 20U;
    l.fixed = {{"fixed", 256, 0}};
    l.fixed_bytes = 256;
    l.layout_identity = 42;
    const std::uint64_t p = std::max<std::uint64_t>(1, total / l.row_bytes);
    l.context_tokens = p + 1U;
    strix::RtxStateImportRegistry registry(heap, 2);
    registry.set_layout(l);
    strix::serve::HttpServerConfig config;
    config.port = 0;
    config.max_body_bytes = chunk + (1U << 20U);
    strix::serve::HttpServer server(config, [&](const strix::serve::HttpRequest& request, strix::serve::HttpResponseWriter& response) {
        if (!strix::decode_cli::handle_rtx_state_import_route(registry, request, response)) response.send(404, "text/plain", "no");
    });
    server.listen();
    std::thread serving([&] { server.serve(); });
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(server.port());
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    require(::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "connect");
    int one = 1;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    const auto exchange = [&](const std::string& head, const char* body, std::size_t n) {
        std::string request = head + "Content-Length: " + std::to_string(n) + "\r\nConnection: keep-alive\r\n\r\n";
        for (std::size_t at = 0; at < request.size();) { const auto k = ::send(fd, request.data() + at, request.size() - at, 0); require(k > 0, "send"); at += static_cast<std::size_t>(k); }
        for (std::size_t at = 0; at < n;) { const auto k = ::send(fd, body + at, n - at, 0); require(k > 0, "send body"); at += static_cast<std::size_t>(k); }
        std::string reply;
        char buffer[4096];
        for (;;) {
            const auto k = ::recv(fd, buffer, sizeof(buffer), 0);
            require(k > 0, "recv");
            reply.append(buffer, static_cast<std::size_t>(k));
            const auto end = reply.find("\r\n\r\n");
            if (end == std::string::npos) continue;
            const auto cl = reply.find("Content-Length: ");
            const auto length = std::stoull(reply.substr(cl + 16U, reply.find("\r\n", cl) - cl - 16U));
            if (reply.size() >= end + 4U + length) {
                require(reply.rfind("HTTP/1.1 200", 0) == 0, "HTTP status: " + reply);
                return reply.substr(end + 4U, static_cast<std::size_t>(length));
            }
        }
    };
    std::string tokens;
    for (std::uint64_t i = 0; i < p; ++i) tokens += (i ? ",1" : "1");
    const std::string begin = "{\"tokens\":[" + tokens + "],\"position\":" + std::to_string(p) +
        ",\"producer\":{\"numerical\":\"1\",\"reduction\":\"2\",\"state_update\":\"3\"},\"layout_identity\":\"42\"}";
    const auto t0 = std::chrono::steady_clock::now();
    const auto id = strix::serve::parse_json(exchange("POST /admin/rtx-import/begin HTTP/1.1\r\nHost: x\r\n", begin.data(), begin.size()))
        .find("id")->as_string();
    const auto t1 = std::chrono::steady_clock::now();
    const std::uint64_t rows_per_chunk = std::max<std::uint64_t>(1, chunk / l.row_bytes);
    std::vector<std::byte> body(static_cast<std::size_t>(rows_per_chunk * l.row_bytes + 24U));
    for (std::size_t i = 0; i < body.size(); ++i) body[i] = static_cast<std::byte>(i * 7U);
    std::vector<strix::ModelContentIdentity> digests;
    std::uint64_t sent = 0;
    for (std::uint64_t first = 0; first < p; first += rows_per_chunk) {
        const auto n = std::min(rows_per_chunk, p - first);
        std::vector<std::byte> head;
        put32(head, 0); put32(head, 0); put32(head, static_cast<std::uint32_t>(first)); put32(head, static_cast<std::uint32_t>(n)); put64(head, n * l.row_bytes);
        std::memcpy(body.data(), head.data(), 24U);
        const auto size = static_cast<std::size_t>(24U + n * l.row_bytes);
        const auto digest = strix::rtx_chunk_digest(std::span<const std::byte>(body.data(), size));
        digests.push_back(digest);
        (void)exchange("POST /admin/rtx-import/chunks?id=" + id + " HTTP/1.1\r\nHost: x\r\nX-Chunk-Sha256: " + strix::model_content_identity_hex(digest) +
            "\r\nX-Chunk-Seq: " + std::to_string(digests.size() - 1U) + "\r\n", reinterpret_cast<const char*>(body.data()), size);
        sent += n * l.row_bytes;
    }
    std::vector<std::byte> fixed(256 + 24U);
    { std::vector<std::byte> head; put32(head, 1); put32(head, 0); put32(head, 0); put32(head, 0); put64(head, 256); std::memcpy(fixed.data(), head.data(), 24U); }
    digests.push_back(strix::rtx_chunk_digest(fixed));
    (void)exchange("POST /admin/rtx-import/chunks?id=" + id + " HTTP/1.1\r\nHost: x\r\nX-Chunk-Sha256: " + strix::model_content_identity_hex(digests.back()) +
        "\r\nX-Chunk-Seq: " + std::to_string(digests.size() - 1U) + "\r\n", reinterpret_cast<const char*>(fixed.data()), fixed.size());
    sent += 256;
    const auto t2 = std::chrono::steady_clock::now();
    const std::string commit = "{\"chunks\":" + std::to_string(digests.size()) + ",\"bytes\":" + std::to_string(sent) + ",\"manifest_sha256\":\"" +
        strix::model_content_identity_hex(strix::rtx_manifest_digest(digests)) + "\"}";
    const auto reply = exchange("POST /admin/rtx-import/commit?id=" + id + " HTTP/1.1\r\nHost: x\r\n", commit.data(), commit.size());
    const auto t3 = std::chrono::steady_clock::now();
    ::close(fd);
    server.stop();
    serving.join();
    const auto s = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    std::cout << "{\"bench\":\"rtx-import-loopback\",\"payload_bytes\":" << sent << ",\"chunks\":" << digests.size() << ",\"chunk_bytes\":"
              << rows_per_chunk * l.row_bytes << ",\"sha256_provider\":\"" << strix::sha256_fast_provider() << "\",\"begin_s\":" << s(t0, t1)
              << ",\"transfer_s\":" << s(t1, t2) << ",\"commit_s\":" << s(t2, t3) << ",\"receive_GBps\":" << static_cast<double>(sent) / s(t1, t2) / 1e9
              << ",\"note\":\"client hashes each chunk before sending (serial with the transfer on one connection)\",\"commit\":" << reply << "}\n";
    return 0;
}

// Client test fixture: writes a state file of the small layout at p = 5 (session generations from the formula)
// and serves the import routes on PORT until the parked state is taken by GET /take (CPU only).
int serve_small(std::uint16_t port, const std::string& state_path) {
    auto l = small_layout();
    l.generations_base = {1, 2, 3, 4, 5, 6};
    const std::uint64_t p = 5;
    const auto payload = canonical_payload(l, p);
    strix::RtxStateFileSession session;
    session.next_position = p;
    session.committed_tokens = p;
    session.valid = 63;
    for (std::size_t f = 0; f < 6U; ++f) session.generations[f] = l.generations_base[f] + p * l.generations_per_row[f];
    session.lineage.push_back({0, 5, 1, 7, 8, 9});
    (void)strix::write_rtx_state_file(state_path, l, std::vector<std::uint32_t>{100, 101, 102, 103, 104}, p, {7, 8, 9}, session, payload);
    strix::RtxStateImportRegistry registry(heap, 2);
    registry.set_layout(l);
    strix::serve::HttpServerConfig config;
    config.port = port;
    strix::serve::HttpServer server(config, [&](const strix::serve::HttpRequest& request, strix::serve::HttpResponseWriter& response) {
        if (request.path == "/take") {
            const std::vector<std::uint32_t> prompt = {100, 101, 102, 103, 104, 105};
            const auto taken = registry.take(prompt);
            const bool ok = taken && std::equal(taken->payload().begin(), taken->payload().end(), payload.begin());
            response.send(ok ? 200 : 409, "text/plain", ok ? "taken, payload equal" : "no equal parked state");
            if (ok) std::thread([&server] { server.stop(); }).detach();
            return;
        }
        if (!strix::decode_cli::handle_rtx_state_import_route(registry, request, response)) response.send(404, "text/plain", "no");
    });
    server.listen();
    std::cerr << "serving on " << server.port() << std::endl;
    server.serve();
    return 0;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 3 && std::string(argv[1]) == "--bench-loopback") {
#if defined(_WIN32)
            std::cerr << "loopback bench is POSIX only\n";
            return 2;
#else
            return bench_loopback(strix::rtx_parse_u64(argv[2]), argc >= 4 ? strix::rtx_parse_u64(argv[3]) : (16ULL << 20U));
#endif
        }
#if !defined(_WIN32)
        if (argc >= 4 && std::string(argv[1]) == "--serve-small")
            return serve_small(static_cast<std::uint16_t>(strix::rtx_parse_u64(argv[2])), argv[3]);
        if (argc >= 4 && std::string(argv[1]) == "--serve-file")
            return serve_file(static_cast<std::uint16_t>(strix::rtx_parse_u64(argv[2])), argv[3],
                argc >= 5 ? static_cast<unsigned>(strix::rtx_parse_u64(argv[4])) : 4U);
#endif
        test_layout();
        test_rotary_import();
        test_staging();
        test_staging_local_rows();
        test_registry();
        test_file();
        test_file_pair();
        test_routes();
        test_routes_codec();
        std::cout << "rtx state import tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
