#pragma once
// MiniMax H3: minimal safetensors reader (host). Header = u64 LE length + JSON {name: {dtype, shape, data_offsets}},
// tensor bytes follow the header; offsets are relative to the end of the header. Reads are positioned (seek + read)
// under a mutex, so one reader can serve several threads.
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <sys/types.h>
#endif

#include "h3/common/fixture.hpp"  // JsonParser

namespace strix::h3::block {

enum class StDtype { f32, f16, bf16, i8, u8, i32, i64, f64, other };

inline StDtype parse_st_dtype(const std::string& s) {
    if (s == "F32") return StDtype::f32;
    if (s == "F16") return StDtype::f16;
    if (s == "BF16") return StDtype::bf16;
    if (s == "I8") return StDtype::i8;
    if (s == "U8") return StDtype::u8;
    if (s == "I32") return StDtype::i32;
    if (s == "I64") return StDtype::i64;
    if (s == "F64") return StDtype::f64;
    return StDtype::other;
}
inline std::size_t st_dtype_size(StDtype d) {
    switch (d) {
        case StDtype::f32: case StDtype::i32: return 4;
        case StDtype::f16: case StDtype::bf16: return 2;
        case StDtype::i8: case StDtype::u8: return 1;
        case StDtype::i64: case StDtype::f64: return 8;
        default: return 0;
    }
}

struct StTensor {
    std::string name, dtype_name;
    StDtype dtype = StDtype::other;
    std::vector<std::int64_t> shape;
    std::uint64_t begin = 0, end = 0;  // absolute file offsets
    std::uint64_t numel() const {
        std::uint64_t n = 1;
        for (auto s : shape) n *= static_cast<std::uint64_t>(s);
        return n;
    }
    std::uint64_t nbytes() const { return end - begin; }
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path) : path_(path) {
        f_ = std::fopen(path.c_str(), "rb");
        if (!f_) throw std::runtime_error("safetensors: cannot open " + path);
        unsigned char lenb[8];
        if (std::fread(lenb, 1, 8, f_) != 8) throw std::runtime_error("safetensors: short header");
        std::uint64_t hl = 0;
        for (int i = 7; i >= 0; --i) hl = (hl << 8U) | lenb[i];
        if (hl == 0 || hl > (1ULL << 30U)) throw std::runtime_error("safetensors: bad header length");
        std::string hdr(static_cast<std::size_t>(hl), '\0');
        if (std::fread(hdr.data(), 1, hdr.size(), f_) != hdr.size()) throw std::runtime_error("safetensors: short header");
        data_base_ = 8 + hl;
        fixture::Json root = fixture::JsonParser(hdr).parse();
        for (const auto& [name, e] : root.obj()) {
            if (name == "__metadata__") {
                if (e.is_object())
                    for (const auto& [k, v] : e.obj())
                        if (v.is_string()) metadata_[k] = v.str();
                continue;
            }
            StTensor t;
            t.name = name;
            t.dtype_name = e.at("dtype").str();
            t.dtype = parse_st_dtype(t.dtype_name);
            for (const auto& s : e.at("shape").arr()) t.shape.push_back(static_cast<std::int64_t>(s.num()));
            const auto& off = e.at("data_offsets").arr();
            t.begin = data_base_ + static_cast<std::uint64_t>(off.at(0).num());
            t.end = data_base_ + static_cast<std::uint64_t>(off.at(1).num());
            if (st_dtype_size(t.dtype) && t.nbytes() != t.numel() * st_dtype_size(t.dtype))
                throw std::runtime_error("safetensors: size mismatch for " + name);
            tensors_.emplace(name, std::move(t));
        }
    }
    ~SafeTensors() {
        if (f_) std::fclose(f_);
    }
    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;

    const std::map<std::string, StTensor>& tensors() const { return tensors_; }
    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    const StTensor* find(const std::string& name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    const StTensor& at(const std::string& name) const {
        const StTensor* t = find(name);
        if (!t) throw std::runtime_error("safetensors: missing tensor " + name);
        return *t;
    }
    // Raw bytes of a tensor into dst (nbytes of the tensor).
    void read_raw(const StTensor& t, void* dst) const {
        std::lock_guard<std::mutex> lk(mu_);
#if defined(_WIN32)
        if (_fseeki64(f_, static_cast<long long>(t.begin), SEEK_SET) != 0)
#else
        if (fseeko(f_, static_cast<off_t>(t.begin), SEEK_SET) != 0)
#endif
            throw std::runtime_error("safetensors: seek failed for " + t.name);
        const std::size_t n = static_cast<std::size_t>(t.nbytes());
        if (std::fread(dst, 1, n, f_) != n) throw std::runtime_error("safetensors: short read for " + t.name);
    }
    template <class T> std::vector<T> read_vec(const std::string& name, StDtype expect) const {
        const StTensor& t = at(name);
        if (t.dtype != expect) throw std::runtime_error("safetensors: unexpected dtype " + t.dtype_name + " for " + name);
        std::vector<T> v(static_cast<std::size_t>(t.nbytes() / sizeof(T)));
        read_raw(t, v.data());
        return v;
    }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::FILE* f_ = nullptr;
    std::uint64_t data_base_ = 0;
    std::map<std::string, StTensor> tensors_;
    std::map<std::string, std::string> metadata_;
    mutable std::mutex mu_;
};

}  // namespace strix::h3::block
