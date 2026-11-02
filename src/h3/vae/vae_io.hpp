#pragma once
// Host IO for the H3 VAE package: a minimal JSON reader (fixture manifests, safetensors headers),
// a safetensors reader (pread of single tensors), and the fixture directory reader
// (manifest.json + one raw little-endian row-major .bin per tensor).

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strix::h3::vae {

struct Json {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;
    const Json& operator[](const std::string& k) const;
    const Json* find(const std::string& k) const;
    static Json parse(const std::string& text);
};

struct TensorInfo {
    std::string dtype;               // "F16", "F32", "BF16" (safetensors) or "float16", "float32" (fixtures)
    std::vector<std::int64_t> shape;
    std::uint64_t offset = 0, bytes = 0;  // absolute file offset
    std::string file;                // fixtures: the .bin path
    std::string sha256;              // fixtures only
    std::int64_t numel() const;
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path);
    bool has(const std::string& name) const { return info_.count(name) != 0; }
    const TensorInfo& info(const std::string& name) const;
    std::vector<std::uint8_t> read(const std::string& name) const;
    void read_into(const std::string& name, void* dst, std::uint64_t capacity) const;
    const std::map<std::string, TensorInfo>& tensors() const { return info_; }
    const std::string& metadata_json() const { return meta_; }

private:
    std::string path_;
    std::map<std::string, TensorInfo> info_;
    std::string meta_;
};

class Fixture {
public:
    explicit Fixture(const std::string& dir);  // throws if manifest.json is missing
    bool has(const std::string& name) const { return info_.count(name) != 0; }
    const TensorInfo& info(const std::string& name) const;
    std::vector<std::uint8_t> raw(const std::string& name) const;
    std::vector<std::uint16_t> f16(const std::string& name) const;  // dtype float16
    std::vector<float> f32(const std::string& name) const;           // dtype float32
    const Json& meta() const { return manifest_["meta"]; }

private:
    std::string dir_;
    Json manifest_;
    std::map<std::string, TensorInfo> info_;
};

std::string read_text(const std::string& path);
bool file_exists(const std::string& path);

}  // namespace strix::h3::vae
