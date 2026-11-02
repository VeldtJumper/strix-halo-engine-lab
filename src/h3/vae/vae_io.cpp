#include "h3/vae/vae_io.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace strix::h3::vae {

namespace {
struct Parser {
    const std::string& s;
    std::size_t i = 0;
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
    [[noreturn]] void fail(const char* what) const { throw std::runtime_error(std::string("json: ") + what + " at " + std::to_string(i)); }
    std::string string_lit() {
        if (s[i] != '"') fail("expected string");
        ++i;
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                ++i;
                const char c = s[i];
                if (c == 'n') out += '\n';
                else if (c == 't') out += '\t';
                else if (c == 'u') { out += '?'; i += 4; }
                else out += c;
                ++i;
            } else out += s[i++];
        }
        ++i;
        return out;
    }
    Json value() {
        ws();
        Json j;
        if (i >= s.size()) fail("eof");
        const char c = s[i];
        if (c == '{') {
            j.kind = Json::Obj;
            ++i; ws();
            if (s[i] == '}') { ++i; return j; }
            for (;;) {
                ws();
                std::string k = string_lit();
                ws();
                if (s[i] != ':') fail("expected :");
                ++i;
                j.obj[k] = value();
                ws();
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == '}') { ++i; return j; }
                fail("expected , or }");
            }
        }
        if (c == '[') {
            j.kind = Json::Arr;
            ++i; ws();
            if (s[i] == ']') { ++i; return j; }
            for (;;) {
                j.arr.push_back(value());
                ws();
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == ']') { ++i; return j; }
                fail("expected , or ]");
            }
        }
        if (c == '"') { j.kind = Json::Str; j.str = string_lit(); return j; }
        if (s.compare(i, 4, "true") == 0) { j.kind = Json::Bool; j.b = true; i += 4; return j; }
        if (s.compare(i, 5, "false") == 0) { j.kind = Json::Bool; i += 5; return j; }
        if (s.compare(i, 4, "null") == 0) { i += 4; return j; }
        std::size_t used = 0;
        j.kind = Json::Num;
        j.num = std::stod(s.substr(i, 40), &used);
        i += used;
        return j;
    }
};
std::string dtype_norm(const std::string& d) {
    if (d == "F16" || d == "float16") return "f16";
    if (d == "F32" || d == "float32") return "f32";
    if (d == "BF16" || d == "bfloat16") return "bf16";
    return d;
}
}  // namespace

const Json* Json::find(const std::string& k) const {
    const auto it = obj.find(k);
    return it == obj.end() ? nullptr : &it->second;
}
const Json& Json::operator[](const std::string& k) const {
    const Json* j = find(k);
    if (!j) throw std::runtime_error("json: missing key " + k);
    return *j;
}
Json Json::parse(const std::string& text) {
    Parser p{text};
    return p.value();
}

std::int64_t TensorInfo::numel() const {
    std::int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
bool file_exists(const std::string& path) { return std::ifstream(path, std::ios::binary).good(); }

namespace {
void pread_exact(const std::string& path, std::uint64_t offset, void* dst, std::uint64_t bytes) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
#ifdef _WIN32
    const int seek = _fseeki64(f, static_cast<long long>(offset), SEEK_SET);
#else
    const int seek = fseeko(f, static_cast<off_t>(offset), SEEK_SET);
#endif
    const std::size_t got = seek == 0 ? std::fread(dst, 1, bytes, f) : 0;
    std::fclose(f);
    if (got != bytes) throw std::runtime_error("short read " + path);
}
}  // namespace

SafeTensors::SafeTensors(const std::string& path) : path_(path) {
    std::uint64_t n = 0;
    pread_exact(path, 0, &n, 8);
    std::string header(n, '\0');
    pread_exact(path, 8, header.data(), n);
    const Json j = Json::parse(header);
    for (const auto& [k, v] : j.obj) {
        if (k == "__metadata__") {
            for (const auto& [mk, mv] : v.obj) meta_ += mk + "=" + mv.str + "\n";
            continue;
        }
        TensorInfo t;
        t.dtype = dtype_norm(v["dtype"].str);
        for (const auto& d : v["shape"].arr) t.shape.push_back(static_cast<std::int64_t>(d.num));
        const auto& off = v["data_offsets"].arr;
        t.offset = 8 + n + static_cast<std::uint64_t>(off[0].num);
        t.bytes = static_cast<std::uint64_t>(off[1].num - off[0].num);
        t.file = path;
        info_[k] = t;
    }
}
const TensorInfo& SafeTensors::info(const std::string& name) const {
    const auto it = info_.find(name);
    if (it == info_.end()) throw std::runtime_error("safetensors: missing " + name);
    return it->second;
}
std::vector<std::uint8_t> SafeTensors::read(const std::string& name) const {
    const auto& t = info(name);
    std::vector<std::uint8_t> out(t.bytes);
    pread_exact(path_, t.offset, out.data(), t.bytes);
    return out;
}
void SafeTensors::read_into(const std::string& name, void* dst, std::uint64_t capacity) const {
    const auto& t = info(name);
    if (t.bytes > capacity) throw std::runtime_error("safetensors: buffer too small for " + name);
    pread_exact(path_, t.offset, dst, t.bytes);
}

Fixture::Fixture(const std::string& dir) : dir_(dir) {
    manifest_ = Json::parse(read_text(dir + "/manifest.json"));
    for (const auto& e : manifest_["tensors"].arr) {
        TensorInfo t;
        t.dtype = dtype_norm(e["dtype"].str);
        for (const auto& d : e["shape"].arr) t.shape.push_back(static_cast<std::int64_t>(d.num));
        t.bytes = static_cast<std::uint64_t>(e["bytes"].num);
        t.file = dir + "/" + e["file"].str;
        t.sha256 = e["sha256"].str;
        info_[e["name"].str] = t;
    }
}
const TensorInfo& Fixture::info(const std::string& name) const {
    const auto it = info_.find(name);
    if (it == info_.end()) throw std::runtime_error("fixture: missing " + name);
    return it->second;
}
std::vector<std::uint8_t> Fixture::raw(const std::string& name) const {
    const auto& t = info(name);
    std::vector<std::uint8_t> out(t.bytes);
    pread_exact(t.file, 0, out.data(), t.bytes);
    return out;
}
std::vector<std::uint16_t> Fixture::f16(const std::string& name) const {
    if (info(name).dtype != "f16") throw std::runtime_error("fixture: " + name + " is not f16");
    const auto r = raw(name);
    std::vector<std::uint16_t> out(r.size() / 2);
    std::memcpy(out.data(), r.data(), r.size());
    return out;
}
std::vector<float> Fixture::f32(const std::string& name) const {
    if (info(name).dtype != "f32") throw std::runtime_error("fixture: " + name + " is not f32");
    const auto r = raw(name);
    std::vector<float> out(r.size() / 4);
    std::memcpy(out.data(), r.data(), r.size());
    return out;
}

}  // namespace strix::h3::vae
