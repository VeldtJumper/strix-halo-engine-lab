#include "h3/te/te_pack_reader.hpp"

#include "h3/te/te_sha256.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>

namespace strix::h3::te {
namespace {

int seek(std::FILE* f, std::uint64_t off, int whence = SEEK_SET) {
#ifdef _WIN32
    return _fseeki64(f, static_cast<long long>(off), whence);
#else
    return fseeko(f, static_cast<off_t>(off), whence);
#endif
}

std::uint64_t tell(std::FILE* f) {
#ifdef _WIN32
    return static_cast<std::uint64_t>(_ftelli64(f));
#else
    return static_cast<std::uint64_t>(ftello(f));
#endif
}

std::string shape_text(const std::vector<std::int64_t>& s) {
    std::string out = "[";
    for (std::size_t i = 0; i < s.size(); ++i) out += (i ? "," : "") + std::to_string(s[i]);
    return out + "]";
}

}  // namespace

std::size_t dtype_size(const std::string& d) {
    if (d == "U8" || d == "I8" || d == "BOOL" || d == "F8_E4M3" || d == "F8_E5M2") return 1;
    if (d == "BF16" || d == "F16" || d == "I16" || d == "U16") return 2;
    if (d == "F32" || d == "I32" || d == "U32") return 4;
    if (d == "F64" || d == "I64" || d == "U64") return 8;
    return 0;
}

std::int64_t TensorInfo::elements() const {
    std::int64_t n = 1;
    for (const auto d : shape) n *= d;
    return n;
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16U;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

SafetensorsFile::SafetensorsFile(const std::string& path) : path_(path), file_(std::fopen(path.c_str(), "rb")) {
    if (!file_) throw PackError("te pack: cannot open " + path);
    std::FILE* f = file_.get();
    if (seek(f, 0, SEEK_END) != 0) throw PackError("te pack: seek failed " + path);
    const std::uint64_t file_size = tell(f);
    if (seek(f, 0) != 0) throw PackError("te pack: seek failed " + path);
    unsigned char len[8];
    if (file_size < 8 || std::fread(len, 1, 8, f) != 8) throw PackError("te pack: short header " + path);
    std::uint64_t n = 0;
    for (int i = 7; i >= 0; --i) n = (n << 8U) | len[i];
    if (n == 0 || n > (64U << 20U) || 8U + n > file_size) throw PackError("te pack: bad header size " + path);
    std::string header(n, '\0');
    if (std::fread(header.data(), 1, n, f) != n) throw PackError("te pack: short header " + path);
    data_offset_ = 8U + n;
    const std::uint64_t data_size = file_size - data_offset_;
    serve::Json doc;
    try {
        doc = serve::parse_json_unique(header);  // duplicate keys / invalid UTF-8 rejected
    } catch (const std::exception& e) {
        throw PackError("te pack: header JSON: " + std::string(e.what()) + " in " + path);
    }
    if (!doc.is_object()) throw PackError("te pack: header is not an object in " + path);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& [name, v] : doc.as_object()) {
        if (name == "__metadata__") continue;
        const auto* dt = v.find("dtype");
        const auto* sh = v.find("shape");
        const auto* off = v.find("data_offsets");
        if (!dt || !dt->is_string() || !sh || !sh->is_array() || !off || !off->is_array() || off->size() != 2)
            throw PackError("te pack: malformed entry " + name + " in " + path);
        TensorInfo t;
        t.dtype = dt->as_string();
        const std::size_t es = dtype_size(t.dtype);
        if (es == 0) throw PackError("te pack: unknown dtype " + t.dtype + " of " + name);
        std::uint64_t elems = 1;
        for (const auto& d : sh->as_array()) {
            if (!d.is_number() || d.as_int() < 0) throw PackError("te pack: bad shape of " + name);
            const auto dim = static_cast<std::uint64_t>(d.as_int());
            if (dim != 0 && elems > std::numeric_limits<std::uint64_t>::max() / dim) throw PackError("te pack: shape overflow " + name);
            elems *= dim;
            t.shape.push_back(d.as_int());
        }
        const auto b = off->as_array()[0], e = off->as_array()[1];
        if (!b.is_number() || !e.is_number() || b.as_int() < 0 || e.as_int() < b.as_int())
            throw PackError("te pack: bad offsets of " + name);
        t.begin = static_cast<std::uint64_t>(b.as_int());
        t.end = static_cast<std::uint64_t>(e.as_int());
        if (t.end > data_size) throw PackError("te pack: " + name + " extends past the end of " + path);
        if (elems > std::numeric_limits<std::uint64_t>::max() / es || t.bytes() != elems * es)
            throw PackError("te pack: " + name + ": byte count " + std::to_string(t.bytes()) + " != shape " +
                            shape_text(t.shape) + " x " + std::to_string(es));
        ranges.emplace_back(t.begin, t.end);
        if (!tensors_.emplace(name, std::move(t)).second) throw PackError("te pack: duplicate tensor " + name);
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i)
        if (ranges[i].first < ranges[i - 1].second) throw PackError("te pack: overlapping tensors in " + path);
}

const TensorInfo& SafetensorsFile::info(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw PackError("te pack: missing tensor " + name + " in " + path_);
    return it->second;
}

const TensorInfo& SafetensorsFile::expect(const std::string& name, const std::string& dtype,
                                          const std::vector<std::int64_t>& shape) const {
    const auto& t = info(name);
    if (t.dtype != dtype) throw PackError("te pack: " + name + " has dtype " + t.dtype + ", expected " + dtype);
    if (!shape.empty() && t.shape != shape)
        throw PackError("te pack: " + name + " has shape " + shape_text(t.shape) + ", expected " + shape_text(shape));
    return t;
}

void SafetensorsFile::read(const std::string& name, void* dst, std::size_t capacity) const {
    const auto& t = info(name);
    if (capacity != t.bytes()) throw PackError("te pack: read of " + name + " with a buffer of the wrong size");
    if (seek(file_.get(), data_offset_ + t.begin) != 0) throw PackError("te pack: seek failed " + name);
    if (std::fread(dst, 1, t.bytes(), file_.get()) != t.bytes()) throw PackError("te pack: short read " + name);
}

float SafetensorsFile::read_f32_scalar(const std::string& name) const {
    const auto& t = info(name);
    if (t.dtype != "F32" || t.bytes() != 4 || !(t.shape.empty() || t.shape == std::vector<std::int64_t>{1}))
        throw PackError("te pack: " + name + " is not an FP32 scalar");
    float v = 0;
    read(name, &v, sizeof v);
    return v;
}

namespace {

const serve::Json& field(const serve::Json& o, const char* k) {
    const auto* v = o.find(k);
    if (!v) throw PackError(std::string("te pack: te-pack.json misses \"") + k + "\"");
    return *v;
}
std::int64_t int_field(const serve::Json& o, const char* k) {
    const auto& v = field(o, k);
    if (!v.is_number()) throw PackError(std::string("te pack: te-pack.json \"") + k + "\" is not a number");
    return v.as_int();
}
double num_field(const serve::Json& o, const char* k) {
    const auto& v = field(o, k);
    if (!v.is_number()) throw PackError(std::string("te pack: te-pack.json \"") + k + "\" is not a number");
    return v.as_double();
}
bool hex64(const std::string& s) {
    if (s.size() != 64) return false;
    for (const char c : s)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

}  // namespace

PackManifest read_pack_manifest(const std::string& dir) {
    const std::string path = dir + "/te-pack.json";
    std::ifstream f(path, std::ios::binary);
    if (!f) throw PackError("te pack: cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    if (text.empty() || text.size() > (16U << 20U)) throw PackError("te pack: bad size of " + path);
    serve::Json doc;
    try {
        doc = serve::parse_json_unique(text);  // duplicate keys / invalid UTF-8 rejected
    } catch (const std::exception& e) {
        throw PackError("te pack: te-pack.json: " + std::string(e.what()));
    }
    if (!doc.is_object()) throw PackError("te pack: te-pack.json is not an object");
    PackManifest m;
    m.json_sha256 = sha256_hex(text.data(), text.size());
    const auto& fmt = field(doc, "format");
    if (!fmt.is_string()) throw PackError("te pack: te-pack.json format");
    m.format = fmt.as_string();
    m.n_layers = int_field(doc, "n_layers");
    m.hidden = int_field(doc, "hidden");
    m.ffn = int_field(doc, "ffn");
    m.n_heads = int_field(doc, "n_heads");
    m.n_kv_heads = int_field(doc, "n_kv_heads");
    m.head_dim = int_field(doc, "head_dim");
    m.vocab = int_field(doc, "vocab");
    m.rms_eps = num_field(doc, "rms_eps");
    m.rope_theta = num_field(doc, "rope_theta");
    const auto& fn = field(doc, "final_norm");
    const auto& mi = field(doc, "mrope_interleaved");
    if (!fn.is_bool() || !mi.is_bool()) throw PackError("te pack: te-pack.json final_norm / mrope_interleaved");
    m.final_norm = fn.as_bool();
    m.mrope_interleaved = mi.as_bool();
    const auto& ms = field(doc, "mrope_sections");
    if (!ms.is_array()) throw PackError("te pack: te-pack.json mrope_sections");
    for (const auto& v : ms.as_array()) {
        if (!v.is_number()) throw PackError("te pack: te-pack.json mrope_sections");
        m.mrope_sections.push_back(v.as_int());
    }
    const auto& files = field(doc, "files");
    if (!files.is_object() || files.as_object().empty()) throw PackError("te pack: te-pack.json files");
    for (const auto& [name, e] : files.as_object()) {
        if (name.empty() || name.find('/') != std::string::npos || name.find('\\') != std::string::npos || name[0] == '.')
            throw PackError("te pack: te-pack.json file name " + name);
        PackFileEntry pe;
        const auto& b = field(e, "bytes");
        const auto& h = field(e, "sha256");
        if (!b.is_number() || b.as_int() <= 0 || !h.is_string() || !hex64(h.as_string()))
            throw PackError("te pack: te-pack.json entry " + name);
        pe.bytes = static_cast<std::uint64_t>(b.as_int());
        pe.sha256 = h.as_string();
        m.files.emplace(name, std::move(pe));
    }
    return m;
}

bool pack_hash_verification_enabled() {
    const char* v = std::getenv("STRIX_H3_TE_VERIFY");
    return !(v && v[0] == '0' && v[1] == '\0');
}

void verify_pack_files(const std::string& dir, const PackManifest& m, const std::vector<std::string>& names, bool hash,
                       unsigned threads) {
    for (const auto& n : names) {
        const auto it = m.files.find(n);
        if (it == m.files.end()) throw PackError("te pack: " + n + " is not listed in te-pack.json");
        std::ifstream f(dir + "/" + n, std::ios::binary | std::ios::ate);
        if (!f) throw PackError("te pack: cannot open " + dir + "/" + n);
        const auto size = static_cast<std::uint64_t>(f.tellg());
        if (size != it->second.bytes)
            throw PackError("te pack: " + n + " has " + std::to_string(size) + " bytes, te-pack.json says " +
                            std::to_string(it->second.bytes));
    }
    if (!hash) return;
    std::atomic<std::size_t> next{0};
    std::mutex mu;
    std::string error;
    const auto work = [&] {
        for (;;) {
            const std::size_t i = next++;
            if (i >= names.size()) return;
            std::string got;
            try {
                got = sha256_file_hex(dir + "/" + names[i]);
            } catch (const std::exception& e) {
                const std::lock_guard<std::mutex> g(mu);
                if (error.empty()) error = e.what();
                return;
            }
            if (got != m.files.at(names[i]).sha256) {
                const std::lock_guard<std::mutex> g(mu);
                if (error.empty()) error = names[i] + ": sha256 " + got + " does not match te-pack.json";
            }
        }
    };
    const unsigned nt = std::max(1U, std::min<unsigned>(threads, static_cast<unsigned>(names.size())));
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t) pool.emplace_back(work);
    for (auto& t : pool) t.join();
    if (!error.empty()) throw PackError("te pack: " + error);
}

}  // namespace strix::h3::te
