// H3 text-encoder tests: fixture file reads and comparison shape checks (shared by the GPU gate and the host tests).
#pragma once

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::h3::te::fixture_io {

// Whole file as T[]: rejects sizes that are not a multiple of sizeof(T) and short reads.
template <typename T>
std::vector<T> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto end = f.tellg();
    if (end < 0) throw std::runtime_error("cannot size " + path);
    const auto n = static_cast<std::size_t>(end);
    if (n % sizeof(T) != 0) throw std::runtime_error(path + ": size not a multiple of the element size");
    std::vector<T> v(n / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
    if (!f || static_cast<std::size_t>(f.gcount()) != n) throw std::runtime_error(path + ": short read");
    return v;
}

// got and ref must both be exactly [rows][width] with rows, width > 0 (no vacuous passes).
inline void check_dims(std::size_t got_size, std::size_t ref_size, std::size_t rows, std::size_t width) {
    if (rows == 0 || width == 0) throw std::runtime_error("compare: empty reference");
    if (ref_size != rows * width) throw std::runtime_error("compare: reference size does not match rows x width");
    if (got_size != rows * width) throw std::runtime_error("compare: output size does not match rows x width");
}

}  // namespace strix::h3::te::fixture_io
