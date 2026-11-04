#pragma once
// Host-only transactional GLM state protocol. Constructed only for explicit import.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/model_content_hash.hpp"
#include "glm_state_layout.hpp"
namespace strix::glm::import {
using Bytes = std::vector<std::byte>;
using Digest = ModelContentIdentity;
inline void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
inline Digest hash(std::span<const std::byte> b) { return sha256_bytes(b); }
inline void number(Bytes& b, std::uint64_t v, unsigned n = 8) {
    for (unsigned i = 0; i < n; ++i) b.push_back(std::byte((v >> (8 * i)) & 255));
}
inline void append(Bytes& b, std::span<const std::byte> v) { b.insert(b.end(), v.begin(), v.end()); }
inline void digest(Bytes& b, const Digest& d) { append(b, std::as_bytes(std::span(d.bytes))); }
struct Reader {
    std::span<const std::byte> b;
    std::span<const std::byte> take(std::size_t n) {
        require(n <= b.size(), "truncated import packet");
        auto v = b.first(n);
        b = b.subspan(n);
        return v;
    }
    std::uint64_t number(unsigned n = 8) {
        auto v = take(n);
        std::uint64_t x = 0;
        for (unsigned i = 0; i < n; ++i) x |= std::uint64_t(std::to_integer<unsigned>(v[i])) << (8 * i);
        return x;
    }
    Digest digest() {
        Digest d{};
        auto v = take(32);
        std::memcpy(d.bytes.data(), v.data(), 32);
        return d;
    }
    std::string text() {
        auto n = number(4);
        auto v = take(static_cast<std::size_t>(n));
        return {reinterpret_cast<const char*>(v.data()), v.size()};
    }
    void end() { require(b.empty(), "trailing import packet bytes"); }
};
inline void text(Bytes& b, const std::string& s) {
    number(b, s.size(), 4);
    append(b, std::as_bytes(std::span(s)));
}
inline Digest unhex(const std::string& s) {
    require(s.size() == 64, "expected 64 hex digits");
    Digest d{};
    auto nib = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return unsigned(c - '0');
        if (c >= 'a' && c <= 'f') return unsigned(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return unsigned(c - 'A' + 10);
        throw std::runtime_error("bad hex");
    };
    for (unsigned i = 0; i < 32; ++i) d.bytes[i] = static_cast<std::uint8_t>((nib(s[2 * i]) << 4) | nib(s[2 * i + 1]));
    return d;
}
inline Digest hmac(std::span<const std::byte> key, std::span<const std::byte> data) {
    require(key.size() == 32, "import key must be 32 raw bytes");
    Bytes inner(64, std::byte{0x36}), outer(64, std::byte{0x5c});
    for (std::size_t i = 0; i < key.size(); ++i) {
        inner[i] ^= key[i];
        outer[i] ^= key[i];
    }
    append(inner, data);
    digest(outer, hash(inner));
    return hash(outer);
}
inline bool secure_equal(const Digest& a, const Digest& b) {
    unsigned x = 0;
    for (unsigned i = 0; i < 32; ++i) x |= a.bytes[i] ^ b.bytes[i];
    return x == 0;
}
inline Bytes vote(const Digest& binding, bool ok) {
    Bytes b;
    number(b, ok ? 1 : 0);
    digest(b, binding);
    return b;
}
inline void agree(const Digest& binding, bool ok, std::span<const std::byte> peer) {
    Reader r{peer};
    const auto peer_ok = r.number();
    const auto peer_binding = r.digest();
    r.end();
    require(ok && peer_ok == 1 && peer_binding == binding,
            "GLM import peer commit/apply rejected (no state may be used)");
}
struct Begin {
    unsigned version = 1, world = 2, rank = 0, layers = 46;
    std::uint64_t frontier = 0, total = 0;
    Digest model{}, prompt{}, transaction{};
    std::string producer, node;
    Bytes encode(bool common = false) const {
        Bytes b;
        number(b, 1, 4);
        number(b, version, 4);
        number(b, world, 4);
        number(b, common ? 0 : rank, 4);
        number(b, layers, 4);
        number(b, frontier);
        number(b, total);
        digest(b, model);
        digest(b, prompt);
        digest(b, transaction);
        text(b, producer);
        text(b, node);
        return b;
    }
    static Begin parse(Reader& r) {
        Begin b;
        b.version = unsigned(r.number(4));
        b.world = unsigned(r.number(4));
        b.rank = unsigned(r.number(4));
        b.layers = unsigned(r.number(4));
        b.frontier = r.number();
        b.total = r.number();
        b.model = r.digest();
        b.prompt = r.digest();
        b.transaction = r.digest();
        b.producer = r.text();
        b.node = r.text();
        r.end();
        return b;
    }
};
// Component IDs: KDA S=0, conv=1; DSA latent=2, ikey=3, igate=4, pooled=5.
// Begin.version 1: each rank receives every component. Version 2 ('DSA once'): the same encoding; rank 0
// receives every component, rank 1 only the KDA components (its DSA components reach it from rank 0 over
// the pair link, outside this wire), so the DSA bytes cross the producer's link once.
struct Component {
    unsigned layer, id;
    std::uint64_t rows, stride;
};
// Version 2: the components a rank does not receive (rank 1's DSA ids 2..5). The producer drops exactly these from
// its export stream, and layout() omits them, so both ends agree by construction.
inline bool omitted(unsigned version, unsigned rank, unsigned id) { return version == 2 && rank == 1 && id >= 2; }
inline std::vector<Component> layout(const Begin& b) {
    require((b.version == 1 || b.version == 2) && b.world == 2 && b.rank < 2 && (b.layers == 45 || b.layers == 46),
            "unsupported import layout/rank");
    const bool dsa = !omitted(b.version, b.rank, 2);
    require(b.frontier > 0 && b.frontier < b.total && b.total <= 131072, "invalid import frontier");
    std::vector<Component> out;
    for (unsigned l = 0; l < b.layers; ++l) {
        if (l < 45 && l % 4 != 3) {
            out.push_back({l, 0, 32, 128 * 128 * 4});
            out.push_back({l, 1, 32, state_layout::kConvFloatsPerHead * sizeof(float)});
        } else if (dsa) {
            out.push_back({l, 2, b.frontier, 1024});
            out.push_back({l, 3, b.frontier, 256});
            out.push_back({l, 4, b.frontier, 256});
            if (state_layout::pooled_rows(b.frontier)) out.push_back({l, 5, state_layout::pooled_rows(b.frontier), 512});
        }
    }
    return out;
}
struct Frame {
    unsigned layer, id;
    std::uint64_t row, rows, bytes;
    std::int64_t offset;
    Digest sha, payload_sha;
};
struct FileClose {
    void operator()(FILE* f) const {
        if (f) std::fclose(f);
    }
};
inline FILE* staging_file() {
#ifdef _WIN32
    FILE* file = nullptr;
    return tmpfile_s(&file) == 0 ? file : nullptr;
#else
    return std::tmpfile();
#endif
}
class Staging {
    std::unique_ptr<FILE, FileClose> file_;
    std::vector<Component> components_;
    std::size_t component_ = 0;
    std::uint64_t row_ = 0;

   public:
    Begin begin;
    std::vector<Frame> frames;
    std::uint64_t raw_bytes = 0, wire_bytes = 0;
    bool committed = false;
    explicit Staging(Begin b) : file_(staging_file()), components_(layout(b)), begin(std::move(b)) {
        require(bool(file_), "import staging tmpfile failed");
    }
    void frame(Reader& r, std::span<const std::byte> packet) {
        require(frames.size() < 32768 && !committed && component_ < components_.size(), "extra import frame");
        Frame f{};
        f.layer = unsigned(r.number(4));
        f.id = unsigned(r.number(4));
        f.row = r.number();
        f.rows = r.number();
        f.bytes = r.number();
        const auto sha = r.digest();
        const auto& c = components_[component_];
        require(f.layer == c.layer && f.id == c.id && f.row == row_ && f.rows > 0 && f.rows <= c.rows - row_,
                "duplicate/out-of-order/invalid import rows");
        require(f.bytes == f.rows * c.stride && f.bytes <= 4194304, "invalid import frame size/codec (raw only)");
        auto payload = r.take(static_cast<std::size_t>(f.bytes));
        r.end();
        require(hash(payload) == sha, "import frame SHA mismatch");
        f.sha = hash(packet);
        f.payload_sha = sha;
#ifdef _WIN32
        f.offset = _ftelli64(file_.get());
#else
        f.offset = ftello(file_.get());
#endif
        require(f.offset >= 0, "import staging offset failed");
        require(std::fwrite(payload.data(), 1, payload.size(), file_.get()) == payload.size(),
                "import staging write failed");
        frames.push_back(f);
        raw_bytes += f.bytes;
        row_ += f.rows;
        if (row_ == c.rows) {
            ++component_;
            row_ = 0;
        }
    }
    void commit(Reader& r) {
        require(!committed && component_ == components_.size(), "missing import frames");
        require(r.number() == frames.size() && r.number() == raw_bytes, "import manifest counts mismatch");
        for (const auto& f : frames) require(r.digest() == f.sha, "import manifest digest mismatch");
        r.end();
        require(std::fflush(file_.get()) == 0, "import staging flush failed");
        committed = true;
    }
    void apply(const std::function<void(const Frame&, std::span<const std::uint8_t>)>& put) {
        require(committed, "import not committed");
        for (const auto& f : frames) {
#ifdef _WIN32
            require(_fseeki64(file_.get(), f.offset, SEEK_SET) == 0, "import staging seek failed");
#else
            require(fseeko(file_.get(), f.offset, SEEK_SET) == 0, "import staging seek failed");
#endif
            Bytes b(static_cast<std::size_t>(f.bytes));
            require(std::fread(b.data(), 1, b.size(), file_.get()) == b.size(), "import staging read failed");
            require(hash(b) == f.payload_sha, "import staging SHA mismatch");
            put(f, {reinterpret_cast<const std::uint8_t*>(b.data()), b.size()});
        }
    }
};
}  // namespace strix::glm::import
