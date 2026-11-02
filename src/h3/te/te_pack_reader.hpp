#pragma once

// Host reader of the H3 text-encoder pack (tools/h3/te/te_pack.py): safetensors files with a
// JSON header. The header is validated on open (known dtype, byte count = product of the shape x
// dtype size, offsets inside the data section, no overlaps, data section = file size); every read
// is checked against the caller's expected dtype and shape before any allocation or upload.

#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::h3::te {

struct TensorInfo {
    std::string dtype;                 // "U8", "I8", "BF16", "F16", "F32", ...
    std::vector<std::int64_t> shape;
    std::uint64_t begin = 0, end = 0;  // byte range inside the data section
    [[nodiscard]] std::uint64_t bytes() const { return end - begin; }
    [[nodiscard]] std::int64_t elements() const;
};

class PackError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Size in bytes of a safetensors dtype; 0 for unknown dtypes.
std::size_t dtype_size(const std::string& dtype);

class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string& path);
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    [[nodiscard]] const TensorInfo& info(const std::string& name) const;
    [[nodiscard]] bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    // Checks dtype and shape (an empty `shape` checks the dtype only) and returns the info.
    const TensorInfo& expect(const std::string& name, const std::string& dtype, const std::vector<std::int64_t>& shape) const;
    // Raw bytes of a tensor into dst; `capacity` must equal the tensor byte count.
    void read(const std::string& name, void* dst, std::size_t capacity) const;
    // Typed read after expect(): rejects element-size mismatches.
    template <typename T>
    [[nodiscard]] std::vector<T> read_vector(const std::string& name, const std::string& dtype,
                                             const std::vector<std::int64_t>& shape) const {
        const auto& t = expect(name, dtype, shape);
        if (dtype_size(t.dtype) != sizeof(T) || t.bytes() % sizeof(T) != 0)
            throw PackError("te pack: " + name + ": element size does not match the requested type");
        std::vector<T> v(t.bytes() / sizeof(T));
        read(name, v.data(), v.size() * sizeof(T));
        return v;
    }
    // A scalar FP32 tensor stored as shape [1] (or []).
    [[nodiscard]] float read_f32_scalar(const std::string& name) const;
    [[nodiscard]] const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
    [[nodiscard]] const std::string& path() const { return path_; }

private:
    struct Closer { void operator()(std::FILE* f) const { if (f) std::fclose(f); } };
    std::string path_;
    std::unique_ptr<std::FILE, Closer> file_;
    std::uint64_t data_offset_ = 0;
    std::map<std::string, TensorInfo> tensors_;
};

// bf16 bits -> float
float bf16_to_float(std::uint16_t bits);

// te-pack.json (tools/h3/te/te_pack.py): format, model geometry and the bytes + sha256 of every pack file.
// Parsed with duplicate-key rejection; the caller checks every field the engine relies on.
struct PackFileEntry {
    std::uint64_t bytes = 0;
    std::string sha256;  // 64 lowercase hex digits
};
struct PackManifest {
    std::string format;
    std::int64_t n_layers = 0, hidden = 0, ffn = 0, n_heads = 0, n_kv_heads = 0, head_dim = 0, vocab = 0;
    double rms_eps = 0, rope_theta = 0;
    bool final_norm = true;
    std::vector<std::int64_t> mrope_sections;
    bool mrope_interleaved = false;
    std::map<std::string, PackFileEntry> files;
    std::string json_sha256;  // digest of te-pack.json itself (pack content identity)
};
PackManifest read_pack_manifest(const std::string& dir);

// Checks that every named file is listed in the manifest with its exact byte count and, when `hash`,
// that its SHA-256 matches (files hashed in parallel on up to `threads` threads). Throws PackError.
void verify_pack_files(const std::string& dir, const PackManifest& m, const std::vector<std::string>& names, bool hash,
                       unsigned threads = 8);

// STRIX_H3_TE_VERIFY: unset or anything but "0" = full SHA-256 verification (default); "0" = manifest,
// geometry and byte counts only.
bool pack_hash_verification_enabled();

}  // namespace strix::h3::te
