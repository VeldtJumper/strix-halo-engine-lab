#pragma once
// HTTP routes of the externally produced prompt state import (--rtx-state-import, docs/rtx-state-import.md).
// They only touch host memory (the registry's stagings), so they run on the connection threads while the
// engine decodes another request; the restore happens on the request thread (HipPrefixStateCacheSession).
//
//   GET  /admin/rtx-import/layout
//   POST /admin/rtx-import/begin        JSON {tokens, position, producer{numerical,reduction,state_update}, layout_identity,
//                                             geometry_identity?, rotary?, context_tokens?, keep?, encoding?, local_rows?} -> {id, payload_bytes, ...}
//   POST /admin/rtx-import/chunk?id=I&component=C&first_row=R      body: whole rows of row component C (id or index)
//   POST /admin/rtx-import/chunk?id=I&fixed=F&offset=O             body: bytes [O, O+n) of fixed component F
//   POST /admin/rtx-import/chunks?id=I                             body: segments (parse_rtx_chunk_segments)
//        every chunk POST: X-Chunk-Sha256 = SHA-256 of the body (verified on receipt), optional X-Chunk-Seq;
//        X-Chunk-Encoding: bp-v1 (import begun with "encoding":"bp-v1"): the body is a bp_codec frame of the plain
//        body above; the digest covers the frame, the segment checks the decoded bytes
//   POST /admin/rtx-import/commit?id=I  JSON {chunks, bytes, manifest_sha256, canonical_sha256?}
//        -> coverage (every byte exactly once) + counts + manifest, parks
//   GET  /admin/rtx-import/status
//   POST /admin/rtx-import/abort?id=I
#include "strix/model_content_hash.hpp"
#include "strix/rtx_state_import.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::decode_cli {

// The value of `key` in a query string (no percent-decoding: ids, numbers and component names need none).
inline std::optional<std::string> rtx_query_value(std::string_view query, std::string_view key) {
    while (!query.empty()) {
        const auto amp = query.find('&');
        const auto part = query.substr(0, amp);
        const auto eq = part.find('=');
        if (part.substr(0, eq) == key) return std::string(eq == std::string_view::npos ? std::string_view{} : part.substr(eq + 1U));
        if (amp == std::string_view::npos) break;
        query.remove_prefix(amp + 1U);
    }
    return std::nullopt;
}

inline std::uint64_t rtx_json_u64(const serve::Json& value, const char* what) {
    try {
        if (value.is_string()) return rtx_parse_u64(value.as_string());
        if (value.is_number()) return rtx_parse_u64(value.number_source());
    } catch (const std::invalid_argument&) {
    }
    throw std::invalid_argument(std::string(what) + " must be an unsigned integer (number, decimal or 0x-hex string)");
}

inline serve::Json rtx_layout_json(const RtxStateLayout& l) {
    using serve::Json;
    Json rows = Json::array();
    // "id" is the index used by chunk segments and queries; "name" the component (kv.L3, index.L3, mtp.kv, ...).
    for (const auto& r : l.rows) {
        Json c = Json::object();
        c.set("id", Json(static_cast<std::uint64_t>(rows.size()))); c.set("name", Json(r.id)); c.set("bytes_per_row", Json(r.bytes_per_row)); c.set("shift", Json(static_cast<std::uint64_t>(r.shift)));
        rows.push(std::move(c));
    }
    Json fixed = Json::array();
    for (const auto& f : l.fixed) {
        Json c = Json::object();
        c.set("id", Json(static_cast<std::uint64_t>(fixed.size()))); c.set("name", Json(f.id)); c.set("bytes", Json(f.bytes)); c.set("offset", Json(f.offset));
        fixed.push(std::move(c));
    }
    Json per_row = Json::array(), base = Json::array();
    for (const auto g : l.generations_per_row) per_row.push(Json(g));
    for (const auto g : l.generations_base) base.push(Json(g));
    Json out = Json::object();
    out.set("format", Json(std::string(rtx_state_format)));
    out.set("rows", std::move(rows));
    out.set("fixed", std::move(fixed));
    out.set("row_bytes", Json(l.row_bytes));
    out.set("fixed_bytes", Json(l.fixed_bytes));
    out.set("layout_identity", Json(rtx_hex_u64(l.layout_identity)));
    out.set("geometry_identity", Json(rtx_hex_u64(rtx_state_geometry_identity(l))));
    out.set("generations_per_row", std::move(per_row));
    out.set("generations_base", std::move(base));
    out.set("valid_union", Json(static_cast<std::uint64_t>(l.valid_union)));
    out.set("valid_base", Json(static_cast<std::uint64_t>(l.valid_base)));
    out.set("max_context", Json(l.context_tokens));
    out.set("context_tokens", Json(l.context_tokens));
    out.set("rotary", serve::parse_json(l.rotary.descriptor().mode() == RotaryMode::yarn
        ? l.rotary.json() : l.rotary.descriptor().json()));
    out.set("model_content", Json(l.model_content_hex));
    out.set("identity_extension", Json(l.identity_extension_hex));
    out.set("cache_identity", Json(l.cache_identity_hex));
    out.set("execution_identity", Json(rtx_hex_u64(l.execution_identity)));
    out.set("runtime_compatibility_identity", Json(rtx_hex_u64(l.runtime_compatibility_identity)));
    out.set("padding", Json("fixed-blob padding is zero and is never sent"));
    out.set("sha256_provider", Json(sha256_fast_provider()));
    Json codecs = Json::array();
    for (const auto codec : rtx_chunk_codecs) codecs.push(Json(std::string(codec)));
    out.set("codecs", std::move(codecs));
    return out;
}

inline std::size_t rtx_component_index(const RtxStateLayout& layout, const std::string& name, bool fixed) {
    const auto found = fixed ? layout.find_fixed(name) : layout.find_row(name);
    if (found) return *found;
    const auto count = fixed ? layout.fixed.size() : layout.rows.size();
    std::uint64_t index = UINT64_MAX;
    try { index = rtx_parse_u64(name); } catch (const std::invalid_argument&) {}
    if (index >= count) throw std::invalid_argument(std::string("unknown ") + (fixed ? "fixed" : "row") + " component " + name);
    return static_cast<std::size_t>(index);
}

// One begin parser for HTTP and file imports; admission remains in RtxStateStaging.
inline RtxStateBeginRequest parse_rtx_state_begin(const serve::Json& body) {
    if (!body.is_object()) throw std::invalid_argument("the begin body must be a JSON object");
    RtxStateBeginRequest begin;
    const auto* tokens = body.find("tokens");
    const auto* position = body.find("position");
    const auto* producer = body.find("producer");
    const auto* layout_identity = body.find("layout_identity");
    if (!tokens || !tokens->is_array() || !position || !producer || !producer->is_object() || !layout_identity)
        throw std::invalid_argument("begin takes tokens (array), position, producer {numerical, reduction, state_update} and layout_identity");
    for (const auto& t : tokens->as_array()) {
        const auto v = rtx_json_u64(t, "token");
        if (v > UINT32_MAX) throw std::invalid_argument("token exceeds uint32");
        begin.tokens.push_back(static_cast<std::uint32_t>(v));
    }
    begin.position = rtx_json_u64(*position, "position");
    const auto id_of = [&](const char* key) {
        const auto* v = producer->find(key);
        if (!v) throw std::invalid_argument(std::string("producer.") + key + " is required");
        return rtx_json_u64(*v, key);
    };
    begin.producer = {id_of("numerical"), id_of("reduction"), id_of("state_update")};
    begin.layout_identity = rtx_json_u64(*layout_identity, "layout_identity");
    if (const auto* g = body.find("geometry_identity")) begin.geometry_identity = rtx_json_u64(*g, "geometry_identity");
    if (const auto* c = body.find("context_tokens")) begin.context_tokens = rtx_json_u64(*c, "context_tokens");
    if (const auto* r = body.find("rotary")) {
        if (!r->is_object()) throw std::invalid_argument("rotary identity must be an object");
        const auto* mode = r->find("rope_mode");
        if (r->size() == 0) begin.rotary = RotaryIdentity{};
        else {
            const auto* factor = r->find("factor");
            const auto* original = r->find("original_max");
            if (!mode || !mode->is_string() || !factor || !original ||
                rtx_json_u64(*original, "rotary.original_max") != RotaryDescriptor::original_max)
                throw std::invalid_argument("rotary identity requires rope_mode, factor and original_max=262144");
            const auto f = rtx_json_u64(*factor, "rotary.factor");
            if (mode->as_string() == "native" && f == 0) {
                if (const auto* table = r->find("table_sha256"); table && !table->is_null())
                    throw std::invalid_argument("native rotary identity must not carry a table SHA256");
                begin.rotary = RotaryIdentity{};
            } else if (mode->as_string() == "yarn" && (f == 2 || f == 4)) {
                const auto* table = r->find("table_sha256");
                if (!table || !table->is_string()) throw std::invalid_argument("rotary identity requires table_sha256");
                begin.rotary = RotaryIdentity(RotaryDescriptor::yarn(static_cast<std::uint32_t>(f)),
                    rtx_parse_sha256(table->as_string()));
            } else throw std::invalid_argument("rotary identity has invalid mode or factor");
        }
    }
    if (const auto* k = body.find("keep"); k && k->is_bool()) begin.keep = k->as_bool();
    if (const auto* q = body.find("local_rows")) begin.local_rows = rtx_json_u64(*q, "local_rows");
    if (const auto* e = body.find("encoding"); e && !e->is_null()) {
        if (!e->is_string()) throw std::invalid_argument("begin encoding must be a string");
        if (e->as_string() != "identity") begin.encoding = e->as_string();
    }
    return begin;
}

// CPU-only metadata preflight shared with staging. The payload and runtime identities
// remain independently validated by the registry before any state is restored.
inline serve::Json read_rtx_state_import_metadata(const std::filesystem::path& path,
    std::span<const std::uint32_t> prompt) {
    std::ifstream metadata(path, std::ios::binary | std::ios::ate);
    if (!metadata || metadata.tellg() < 0 || metadata.tellg() > (64 << 20))
        throw std::invalid_argument("cannot read import JSON (limit 64 MiB)");
    std::string text(static_cast<std::size_t>(metadata.tellg()), '\0');
    metadata.seekg(0);
    if (!metadata.read(text.data(), static_cast<std::streamsize>(text.size())))
        throw std::invalid_argument("cannot read import JSON");
    auto body = serve::parse_json(text);
    const auto begin = parse_rtx_state_begin(body);
    if (!begin.position || begin.position != begin.tokens.size() || begin.tokens.size() >= prompt.size() ||
        !std::equal(begin.tokens.begin(), begin.tokens.end(), prompt.begin()))
        throw std::invalid_argument("import tokens must match position and prefix the full prompt with a nonempty suffix");
    return body;
}

// The file is canonical component-order bytes, including zero fixed-blob padding.
// Only the registry owns pinned payload storage; the reader uses a bounded ordinary buffer.
inline void stage_rtx_state_import_file(RtxStateImportRegistry& registry, const std::filesystem::path& path,
    std::span<const std::uint32_t> prompt, std::size_t chunk_bytes = 4U << 20U, std::uint64_t expected_position = 0) {
    if (!chunk_bytes || chunk_bytes > (64U << 20U)) throw std::invalid_argument("invalid import file chunk size");
    const auto body = read_rtx_state_import_metadata(path, prompt);
    auto begin = parse_rtx_state_begin(body);
    if (expected_position && begin.position != expected_position)
        throw std::invalid_argument("import position changed after PLE admission");
    const auto* bytes = body.find("payload_bytes");
    const auto* sha = body.find("payload_sha256");
    if (!bytes || !sha || !sha->is_string()) throw std::invalid_argument("import requires payload_bytes and payload_sha256");
    const auto expected_bytes = rtx_json_u64(*bytes, "payload_bytes");
    const auto expected_sha = rtx_parse_sha256(sha->as_string());
    auto binary = path;
    binary.replace_extension(".bin");
    if (std::filesystem::file_size(binary) != expected_bytes) throw std::invalid_argument("import payload_bytes mismatch");
    std::ifstream input(binary, std::ios::binary);
    if (!input) throw std::invalid_argument("cannot open import payload");
    const auto id = registry.begin(std::move(begin));
    try {
        const auto staging = registry.find(id);
        if (staging->payload_bytes() != expected_bytes) throw std::invalid_argument("import payload_bytes differs from layout");
        const auto& layout = staging->layout();
        std::vector<std::byte> buffer(chunk_bytes);
        Sha256Fast payload_sha;
        std::vector<ModelContentIdentity> digests;
        std::uint64_t data_bytes = 0;
        const auto read = [&](std::size_t n) {
            if (!input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(n)))
                throw std::invalid_argument("truncated import payload");
            const auto data = std::span<const std::byte>(buffer).first(n);
            payload_sha.update(data);
            return data;
        };
        const auto write = [&](const RtxStateSegment& segment, std::span<const std::byte> data) {
            RtxStateChunkRecord record;
            record.seq = digests.size(); record.digest = rtx_chunk_digest(data); record.wire_bytes = data.size();
            staging->write_chunk(std::span<const RtxStateSegment>(&segment, 1), record);
            digests.push_back(record.digest); data_bytes += data.size();
        };
        for (std::size_t c = 0; c < layout.rows.size(); ++c) {
            const auto bpr = layout.rows[c].bytes_per_row;
            if (bpr > chunk_bytes) throw std::invalid_argument("import row exceeds chunk bound");
            const auto rows = staging->position() > layout.rows[c].shift ? staging->position() - layout.rows[c].shift : 0;
            for (std::uint64_t first = 0; first < rows;) {
                const auto n = std::min<std::uint64_t>(rows - first, chunk_bytes / bpr);
                const auto data = read(static_cast<std::size_t>(n * bpr));
                write(staging->rows_segment(c, first, data), data); first += n;
            }
        }
        std::uint64_t fixed_at = 0;
        const auto padding = [&](std::uint64_t n) {
            while (n) {
                const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(n, chunk_bytes));
                const auto data = read(count);
                if (std::any_of(data.begin(), data.end(), [](std::byte b) { return b != std::byte{0}; }))
                    throw std::invalid_argument("import fixed padding must be zero");
                n -= count;
            }
        };
        for (std::size_t c = 0; c < layout.fixed.size(); ++c) {
            const auto& f = layout.fixed[c];
            padding(f.offset - fixed_at);
            for (std::uint64_t at = 0; at < f.bytes;) {
                const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(f.bytes - at, chunk_bytes));
                const auto data = read(count);
                write(staging->fixed_segment(c, at, data), data); at += count;
            }
            fixed_at = f.offset + f.bytes;
        }
        padding(layout.fixed_bytes - fixed_at);
        if (input.peek() != std::char_traits<char>::eof()) throw std::invalid_argument("extra import payload bytes");
        if (payload_sha.finalize() != expected_sha) throw std::invalid_argument("import payload_sha256 mismatch");
        registry.commit(id, {digests.size(), data_bytes, rtx_manifest_digest(digests), false});
    } catch (...) {
        registry.abort(id);
        throw;
    }
}

// Handles /admin/rtx-import/* (returns false for any other path). Answers every error itself.
inline bool handle_rtx_state_import_route(RtxStateImportRegistry& registry, const serve::HttpRequest& request,
    serve::HttpResponseWriter& response) {
    using serve::Json;
    constexpr std::string_view prefix = "/admin/rtx-import/";
    if (!std::string_view(request.path).starts_with(prefix)) return false;
    const auto route = std::string_view(request.path).substr(prefix.size());
    const auto send = [&](int status, const Json& body) { response.send(status, "application/json", body.dump()); };
    const auto fail = [&](int status, const std::string& message) {
        Json detail = Json::object();
        detail.set("message", Json(message)); detail.set("type", Json(status >= 500 ? "server_error" : "invalid_request_error"));
        Json out = Json::object(); out.set("error", std::move(detail));
        send(status, out);
    };
    const auto query = [&](const char* key) {
        auto value = rtx_query_value(request.query, key);
        if (!value || value->empty()) throw std::invalid_argument(std::string("query parameter ") + key + " is required");
        return *value;
    };
    const auto staging_of = [&](const std::string& id) {
        auto staging = registry.find(id);
        if (!staging) throw std::out_of_range("unknown or finished rtx import id " + id);
        return staging;
    };
    try {
        if (request.method == "GET" && route == "layout") {
            const auto layout = registry.layout();
            if (!layout) { fail(503, "the engine has not published its state layout yet (calibration still running)"); return true; }
            send(200, rtx_layout_json(*layout));
            return true;
        }
        if (request.method == "GET" && route == "status") {
            Json list = Json::array();
            for (const auto& e : registry.status()) {
                Json entry = Json::object();
                entry.set("id", Json(e.id)); entry.set("position", Json(e.position)); entry.set("payload_bytes", Json(e.payload_bytes));
                entry.set("covered_bytes", Json(e.covered_bytes)); entry.set("committed", Json(e.committed)); entry.set("parked", Json(e.parked));
                entry.set("keep", Json(e.keep));
                list.push(std::move(entry));
            }
            const auto s = registry.stats();
            Json stats = Json::object();
            stats.set("begun", Json(s.begun)); stats.set("committed", Json(s.committed)); stats.set("aborted", Json(s.aborted));
            stats.set("failed_commits", Json(s.failed_commits)); stats.set("taken", Json(s.taken)); stats.set("misses", Json(s.misses));
            stats.set("last_miss", Json(s.last_miss));
            Json out = Json::object();
            out.set("ready", Json(registry.layout() != nullptr)); out.set("imports", std::move(list)); out.set("stats", std::move(stats));
            send(200, out);
            return true;
        }
        if (request.method != "POST") { fail(404, "unknown route " + request.method + " " + request.path); return true; }
        if (route == "begin") {
            Json body;
            try { body = serve::parse_json(request.body); }
            catch (const serve::JsonError& e) { throw std::invalid_argument(std::string("invalid JSON body: ") + e.what()); }
            auto begin = parse_rtx_state_begin(body);
            const auto encoding = begin.encoding;
            const auto start = std::chrono::steady_clock::now();
            const auto position_value = begin.position;
            const auto id = registry.begin(std::move(begin));
            const auto staging = registry.find(id);
            Json out = Json::object();
            out.set("id", Json(id));
            out.set("position", Json(position_value));
            out.set("payload_bytes", Json(staging ? staging->payload_bytes() : 0U));
            out.set("rows_bytes", Json(staging ? staging->layout().rows_bytes(position_value) : 0U));
            out.set("allocate_ms", Json(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()));
            out.set("encoding", Json(encoding.empty() ? std::string("identity") : encoding));
            if (staging && staging->request().local_rows) out.set("local_rows", Json(staging->request().local_rows));
            send(200, out);
            return true;
        }
        if (route == "chunk" || route == "chunks") {
            const auto staging = staging_of(query("id"));
            const auto wire = std::as_bytes(std::span<const char>(request.body.data(), request.body.size()));
            const auto sha = request.header("x-chunk-sha256");
            if (sha.empty()) throw std::invalid_argument("every chunk carries X-Chunk-Sha256 (SHA-256 of the body)");
            RtxStateChunkRecord record;
            record.digest = rtx_chunk_digest(wire);
            record.wire_bytes = wire.size();
            if (record.digest != rtx_parse_sha256(sha))
                throw std::invalid_argument("chunk body sha256 differs from X-Chunk-Sha256 (received " + model_content_identity_hex(record.digest) + ")");
            if (const auto seq = request.header("x-chunk-seq"); !seq.empty()) record.seq = rtx_parse_u64(seq);
            // X-Chunk-Encoding: the wire body is a frame of the plain body; the digest above covers the frame, every
            // check below the decoded bytes. Decoded on the shared pool into a per-connection buffer.
            auto data = wire;
            double decode_ms = 0.0;
            const auto encoding = request.header("x-chunk-encoding");
            if (!encoding.empty() && encoding != "identity") {
                if (encoding != bp_codec::codec_name)
                    throw std::invalid_argument("X-Chunk-Encoding " + encoding + " is not supported (layout lists `codecs`)");
                if (staging->request().encoding != encoding)
                    throw std::invalid_argument("X-Chunk-Encoding " + encoding + " needs an import begun with \"encoding\":\"" + encoding + "\"");
                const auto decode_start = std::chrono::steady_clock::now();
                const std::span<const std::uint8_t> frame(reinterpret_cast<const std::uint8_t*>(request.body.data()), request.body.size());
                const auto plain = bp_codec::decoded_size(frame);
                if (plain > rtx_max_decoded_chunk_bytes) throw std::invalid_argument("decoded chunk exceeds 256 MiB");
                thread_local std::unique_ptr<std::uint8_t[]> buffer;
                thread_local std::size_t capacity = 0;
                if (capacity < plain) { buffer.reset(); buffer.reset(new std::uint8_t[static_cast<std::size_t>(plain)]); capacity = static_cast<std::size_t>(plain); }
                bp_codec::decode(frame, std::span<std::uint8_t>(buffer.get(), static_cast<std::size_t>(plain)), &registry.decode_pool());
                data = std::as_bytes(std::span<const std::uint8_t>(buffer.get(), static_cast<std::size_t>(plain)));
                decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_start).count();
            }
            const auto& layout = staging->layout();
            std::vector<RtxStateSegment> segments;
            if (route == "chunks") segments = parse_rtx_chunk_segments(*staging, data);
            else if (const auto component = rtx_query_value(request.query, "component"))
                segments.push_back(staging->rows_segment(rtx_component_index(layout, *component, false), rtx_parse_u64(query("first_row")), data));
            else if (const auto fixed = rtx_query_value(request.query, "fixed"))
                segments.push_back(staging->fixed_segment(rtx_component_index(layout, *fixed, true), rtx_parse_u64(query("offset")), data));
            else throw std::invalid_argument("chunk takes component=&first_row= or fixed=&offset=");
            staging->write_chunk(segments, record);
            Json out = Json::object();
            out.set("covered_bytes", Json(staging->covered_bytes())); out.set("payload_bytes", Json(staging->payload_bytes()));
            out.set("segments", Json(static_cast<std::uint64_t>(segments.size()))); out.set("chunks", Json(staging->chunks()));
            out.set("wire_bytes", Json(static_cast<std::uint64_t>(wire.size()))); out.set("decoded_bytes", Json(static_cast<std::uint64_t>(data.size())));
            if (data.data() != wire.data()) out.set("decode_ms", Json(decode_ms));
            send(200, out);
            return true;
        }
        if (route == "commit") {
            const auto id = query("id");
            const auto staging = staging_of(id);
            Json body;
            try { body = serve::parse_request_json(request.body); }
            catch (const serve::JsonError& e) { throw std::invalid_argument(std::string("invalid JSON body: ") + e.what()); }
            const auto* chunks = body.is_object() ? body.find("chunks") : nullptr;
            const auto* bytes = body.is_object() ? body.find("bytes") : nullptr;
            const auto* manifest = body.is_object() ? body.find("manifest_sha256") : nullptr;
            if (!chunks || !bytes || !manifest || !manifest->is_string())
                throw std::invalid_argument("commit takes {chunks, bytes, manifest_sha256, canonical_sha256?}");
            RtxStateCommitRequest commit;
            commit.chunks = rtx_json_u64(*chunks, "chunks");
            commit.bytes = rtx_json_u64(*bytes, "bytes");
            commit.manifest = rtx_parse_sha256(manifest->as_string());
            if (const auto* c = body.find("canonical_sha256"); c && c->is_bool()) commit.canonical_sha256 = c->as_bool();
            const auto result = registry.commit(id, commit);
            Json out = Json::object();
            out.set("id", Json(id)); out.set("parked", Json(true)); out.set("position", Json(staging->position()));
            out.set("payload_bytes", Json(staging->payload_bytes())); out.set("chunks", Json(staging->chunks()));
            out.set("manifest_sha256", Json(model_content_identity_hex(staging->manifest_sha256())));
            out.set("verify_ms", Json(result.verify_ms));
            out.set("wire_bytes", Json(staging->wire_bytes()));
            out.set("encoding", Json(staging->request().encoding.empty() ? std::string("identity") : staging->request().encoding));
            if (result.canonical) {
                out.set("canonical_sha256", Json(model_content_identity_hex(*result.canonical)));
                out.set("canonical_ms", Json(result.canonical_ms));
            }
            send(200, out);
            return true;
        }
        if (route == "abort") {
            const auto id = query("id");
            if (!registry.abort(id)) throw std::out_of_range("unknown or finished rtx import id " + id);
            Json out = Json::object(); out.set("id", Json(id)); out.set("aborted", Json(true));
            send(200, out);
            return true;
        }
        fail(404, "unknown route " + request.method + " " + request.path);
    } catch (const std::length_error& e) {
        fail(429, e.what());
    } catch (const std::out_of_range& e) {
        fail(std::string_view(e.what()).starts_with("unknown or finished") ? 404 : 400, e.what());
    } catch (const std::invalid_argument& e) {
        fail(400, e.what());
    } catch (const std::logic_error& e) {
        fail(409, e.what());
    } catch (const std::exception& e) {
        fail(500, e.what());
    }
    return true;
}

}  // namespace strix::decode_cli
