#pragma once

// Reader for h3fx v1 reference fixtures: <set>/manifest.json + one raw little-endian .bin per
// tensor (C order). Header-only, host only, no dependencies beyond the standard library.
// dtypes: bf16 f16 f32 f64 i8 u8 i16 i32 i64 bool(u8). Each entry carries its sha256.
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace strix::h3::fixture {

// ---------------------------------------------------------------- minimal JSON
struct Json;
using JsonArray = std::vector<Json>;
using JsonObject = std::vector<std::pair<std::string, Json>>;

struct Json {
    std::variant<std::nullptr_t, bool, double, std::string, std::shared_ptr<JsonArray>,
                 std::shared_ptr<JsonObject>>
        v;
    bool is_null() const { return std::holds_alternative<std::nullptr_t>(v); }
    bool is_number() const { return std::holds_alternative<double>(v); }
    bool is_string() const { return std::holds_alternative<std::string>(v); }
    bool is_array() const { return std::holds_alternative<std::shared_ptr<JsonArray>>(v); }
    bool is_object() const { return std::holds_alternative<std::shared_ptr<JsonObject>>(v); }
    double num() const { return std::get<double>(v); }
    const std::string& str() const { return std::get<std::string>(v); }
    const JsonArray& arr() const { return *std::get<std::shared_ptr<JsonArray>>(v); }
    const JsonObject& obj() const { return *std::get<std::shared_ptr<JsonObject>>(v); }
    const Json* find(std::string_view key) const {
        if (!is_object()) return nullptr;
        for (const auto& [k, val] : obj())
            if (k == key) return &val;
        return nullptr;
    }
    const Json& at(std::string_view key) const {
        const Json* j = find(key);
        if (!j) throw std::runtime_error("fixture json: missing key " + std::string(key));
        return *j;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view s) : s_(s) {}
    Json parse() {
        Json j = value();
        ws();
        if (p_ != s_.size()) fail("trailing data");
        return j;
    }

private:
    std::string_view s_;
    std::size_t p_ = 0;
    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("fixture json: ") + what + " at " + std::to_string(p_));
    }
    void ws() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\n' || s_[p_] == '\r' || s_[p_] == '\t')) ++p_;
    }
    char peek() {
        ws();
        if (p_ >= s_.size()) fail("unexpected end");
        return s_[p_];
    }
    void expect(char c) {
        if (peek() != c) fail("unexpected character");
        ++p_;
    }
    bool lit(std::string_view w) {
        if (s_.substr(p_, w.size()) == w) {
            p_ += w.size();
            return true;
        }
        return false;
    }
    static void utf8(std::string& o, std::uint32_t cp) {
        if (cp < 0x80) {
            o += static_cast<char>(cp);
        } else if (cp < 0x800) {
            o += static_cast<char>(0xC0 | (cp >> 6));
            o += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            o += static_cast<char>(0xE0 | (cp >> 12));
            o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            o += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            o += static_cast<char>(0xF0 | (cp >> 18));
            o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            o += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    std::uint32_t hex4() {
        if (p_ + 4 > s_.size()) fail("bad escape");
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s_[p_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<std::uint32_t>(c - 'A' + 10);
            else fail("bad hex");
        }
        return v;
    }
    std::string string() {
        expect('"');
        std::string o;
        while (true) {
            if (p_ >= s_.size()) fail("unterminated string");
            char c = s_[p_++];
            if (c == '"') break;
            if (c != '\\') {
                o += c;
                continue;
            }
            if (p_ >= s_.size()) fail("bad escape");
            char e = s_[p_++];
            switch (e) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    std::uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && lit("\\u")) {
                        std::uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(o, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        return o;
    }
    Json value() {
        char c = peek();
        Json j;
        if (c == '{') {
            ++p_;
            auto o = std::make_shared<JsonObject>();
            if (peek() != '}') {
                while (true) {
                    std::string k = string();
                    expect(':');
                    o->emplace_back(std::move(k), value());
                    if (peek() == ',') {
                        ++p_;
                        continue;
                    }
                    break;
                }
            }
            expect('}');
            j.v = o;
        } else if (c == '[') {
            ++p_;
            auto a = std::make_shared<JsonArray>();
            if (peek() != ']') {
                while (true) {
                    a->push_back(value());
                    if (peek() == ',') {
                        ++p_;
                        continue;
                    }
                    break;
                }
            }
            expect(']');
            j.v = a;
        } else if (c == '"') {
            j.v = string();
        } else if (lit("true")) {
            j.v = true;
        } else if (lit("false")) {
            j.v = false;
        } else if (lit("null")) {
            j.v = nullptr;
        } else if (lit("NaN")) {
            j.v = std::nan("");
        } else if (lit("Infinity")) {
            j.v = HUGE_VAL;
        } else if (lit("-Infinity")) {
            j.v = -HUGE_VAL;
        } else {
            std::size_t b = p_;
            while (p_ < s_.size() && std::strchr("+-0123456789.eE", s_[p_]) != nullptr) ++p_;
            if (b == p_) fail("bad value");
            j.v = std::stod(std::string(s_.substr(b, p_ - b)));
        }
        return j;
    }
};

// ---------------------------------------------------------------- SHA-256
inline std::string sha256_hex(const std::uint8_t* data, std::size_t n) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto block = [&](const std::uint8_t* b) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(b[4 * i]) << 24) | (std::uint32_t(b[4 * i + 1]) << 16) |
                   (std::uint32_t(b[4 * i + 2]) << 8) | std::uint32_t(b[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            std::uint32_t s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            std::uint32_t s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            std::uint32_t t1 = hh + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) +
                               ((e & f) ^ (~e & g)) + k[i] + w[i];
            std::uint32_t t2 = (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) +
                               ((a & bb) ^ (a & c) ^ (bb & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
        }
        h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    };
    std::size_t full = n / 64;
    for (std::size_t i = 0; i < full; ++i) block(data + 64 * i);
    std::uint8_t tail[128] = {};
    std::size_t rem = n - full * 64;
    if (rem) std::memcpy(tail, data + full * 64, rem);
    tail[rem] = 0x80;
    std::size_t tl = rem + 1 + 8 <= 64 ? 64 : 128;
    std::uint64_t bits = std::uint64_t(n) * 8;
    for (int i = 0; i < 8; ++i) tail[tl - 1 - i] = static_cast<std::uint8_t>(bits >> (8 * i));
    block(tail);
    if (tl == 128) block(tail + 64);
    static constexpr char hx[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) {
            std::uint8_t byte = static_cast<std::uint8_t>(h[i] >> (24 - 8 * j));
            out[8 * i + 2 * j] = hx[byte >> 4];
            out[8 * i + 2 * j + 1] = hx[byte & 15];
        }
    return out;
}

// ---------------------------------------------------------------- dtypes
enum class DType { bf16, f16, f32, f64, i8, u8, i16, i32, i64, boolean };

inline DType parse_dtype(const std::string& s) {
    static const std::map<std::string, DType> m = {
        {"bf16", DType::bf16}, {"f16", DType::f16}, {"f32", DType::f32}, {"f64", DType::f64},
        {"i8", DType::i8},     {"u8", DType::u8},   {"i16", DType::i16}, {"i32", DType::i32},
        {"i64", DType::i64},   {"bool", DType::boolean}};
    auto it = m.find(s);
    if (it == m.end()) throw std::runtime_error("fixture: unknown dtype " + s);
    return it->second;
}

inline std::size_t dtype_size(DType d) {
    switch (d) {
        case DType::bf16: case DType::f16: case DType::i16: return 2;
        case DType::f32: case DType::i32: return 4;
        case DType::f64: case DType::i64: return 8;
        default: return 1;
    }
}

inline float bf16_to_f32(std::uint16_t b) { return std::bit_cast<float>(std::uint32_t(b) << 16); }

inline float f16_to_f32(std::uint16_t h) {
    std::uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF, out;
    if (e == 0) {
        if (m == 0) {
            out = s << 31;
        } else {
            int ee = -1;
            do { m <<= 1; ++ee; } while ((m & 0x400) == 0);
            out = (s << 31) | (std::uint32_t(127 - 15 - ee) << 23) | ((m & 0x3FF) << 13);
        }
    } else if (e == 31) {
        out = (s << 31) | 0x7F800000u | (m << 13);
    } else {
        out = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    return std::bit_cast<float>(out);
}

static_assert(std::endian::native == std::endian::little, "h3fx fixtures are little-endian");

// ---------------------------------------------------------------- manifest + tensors
struct TensorInfo {
    std::string name, file, sha256, desc;
    DType dtype = DType::f32;
    std::vector<std::int64_t> shape;
    std::uint64_t nbytes = 0;
    std::uint64_t numel() const {
        std::uint64_t n = 1;
        for (auto s : shape) n *= static_cast<std::uint64_t>(s);
        return n;
    }
};

struct Tensor {
    TensorInfo info;
    std::vector<std::uint8_t> bytes;
    template <class T> const T* data() const { return reinterpret_cast<const T*>(bytes.data()); }
    // any float/int dtype widened to float (bf16/f16 exact; i64/f64 may round)
    std::vector<float> to_f32() const {
        std::vector<float> o(info.numel());
        for (std::size_t i = 0; i < o.size(); ++i) {
            const std::uint8_t* p = bytes.data() + i * dtype_size(info.dtype);
            switch (info.dtype) {
                case DType::bf16: { std::uint16_t v; std::memcpy(&v, p, 2); o[i] = bf16_to_f32(v); break; }
                case DType::f16: { std::uint16_t v; std::memcpy(&v, p, 2); o[i] = f16_to_f32(v); break; }
                case DType::f32: std::memcpy(&o[i], p, 4); break;
                case DType::f64: { double v; std::memcpy(&v, p, 8); o[i] = static_cast<float>(v); break; }
                case DType::i8: o[i] = static_cast<float>(static_cast<std::int8_t>(*p)); break;
                case DType::u8: case DType::boolean: o[i] = static_cast<float>(*p); break;
                case DType::i16: { std::int16_t v; std::memcpy(&v, p, 2); o[i] = v; break; }
                case DType::i32: { std::int32_t v; std::memcpy(&v, p, 4); o[i] = static_cast<float>(v); break; }
                case DType::i64: { std::int64_t v; std::memcpy(&v, p, 8); o[i] = static_cast<float>(v); break; }
            }
        }
        return o;
    }
};

inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("fixture: cannot open " + path);
    return std::string(std::istreambuf_iterator<char>(f), {});
}

class Manifest {
public:
    std::string dir, set_name;
    Json meta;
    std::vector<TensorInfo> tensors;

    static Manifest load(const std::string& set_dir) {
        Manifest m;
        m.dir = set_dir;
        Json root = JsonParser(read_file(set_dir + "/manifest.json")).parse();
        if (root.at("format").str() != "h3fx") throw std::runtime_error("fixture: not an h3fx manifest");
        if (root.at("version").num() != 1) throw std::runtime_error("fixture: unsupported version");
        m.set_name = root.at("set").str();
        if (const Json* md = root.find("meta")) m.meta = *md;
        for (const Json& e : root.at("tensors").arr()) {
            TensorInfo t;
            t.name = e.at("name").str();
            t.file = e.at("file").str();
            t.dtype = parse_dtype(e.at("dtype").str());
            for (const Json& s : e.at("shape").arr()) t.shape.push_back(static_cast<std::int64_t>(s.num()));
            t.nbytes = static_cast<std::uint64_t>(e.at("nbytes").num());
            t.sha256 = e.at("sha256").str();
            if (const Json* d = e.find("desc")) t.desc = d->str();
            if (t.nbytes != t.numel() * dtype_size(t.dtype))
                throw std::runtime_error("fixture: nbytes/shape mismatch for " + t.name);
            m.tensors.push_back(std::move(t));
        }
        return m;
    }

    const TensorInfo* find(std::string_view name) const {
        for (const auto& t : tensors)
            if (t.name == name) return &t;
        return nullptr;
    }

    Tensor read(std::string_view name, bool verify = true) const {
        const TensorInfo* ti = find(name);
        if (!ti) throw std::runtime_error("fixture: no tensor " + std::string(name));
        Tensor t;
        t.info = *ti;
        std::string raw = read_file(dir + "/" + ti->file);
        if (raw.size() != ti->nbytes) throw std::runtime_error("fixture: size mismatch for " + ti->name);
        t.bytes.assign(raw.begin(), raw.end());
        if (verify && sha256_hex(t.bytes.data(), t.bytes.size()) != ti->sha256)
            throw std::runtime_error("fixture: sha256 mismatch for " + ti->name);
        return t;
    }
};

}  // namespace strix::h3::fixture
