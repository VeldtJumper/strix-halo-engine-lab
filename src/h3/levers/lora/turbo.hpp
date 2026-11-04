#pragma once
// Turbo v4 LoRA device holder (identity h3-turbo-v4-bypass, FAST-class, OFF by default): resident lora_A / lora_B of
// every branched linear + the rank-64 scratch, and one call per linear for the block driver:
//   TurboV4Device turbo(0.70F, max_rows);
//   turbo.upload(Site::kQkv, block, A_host, B_host, k, n);         // once per (site, block), from the LoRA file
//   gemm::linear(... qkv ...);                                     // base output y (bf16)
//   turbo.apply(Site::kQkv, block, x, x_ld, rows, y, y_ld, stream); // y += branch, before anything reads y
// One stream at a time per holder (the scratch is shared). apply on a (site, block) that was never uploaded throws,
// so a missing tensor cannot silently fall back to the dense path.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "h3/levers/lora/bypass.hpp"
#include "h3/levers/lora/turbo_host.hpp"

namespace strix::h3::levers::lora {

class TurboV4Device {
public:
    TurboV4Device(float strength, std::uint32_t max_rows) : s_(strength), max_rows_(max_rows) {
        if (!max_rows) throw std::invalid_argument("h3 lora: max_rows 0");
        check(hipMalloc(&t_, static_cast<std::size_t>(max_rows) * kMaxRank * 2U), "scratch");  // [rows][rank <= 512]
    }
    TurboV4Device(const TurboV4Device&) = delete;
    TurboV4Device& operator=(const TurboV4Device&) = delete;
    ~TurboV4Device() {
        for (auto& kv : w_) {
            static_cast<void>(hipFree(kv.second.a));
            static_cast<void>(hipFree(kv.second.b));
        }
        static_cast<void>(hipFree(t_));
    }
    float strength() const { return s_; }
    std::size_t device_bytes() const { return bytes_; }

    // a: [r][k] bf16, b: [n][r] bf16 (checkpoint row order; fc1 = [gate | up] as the plain fc1 output); the rank r is
    // a.size() / k (multiple of 32, <= 512). The branch scale is the holder's strength (Turbo v4: no alpha tensors).
    void upload(Site site, std::uint32_t block, const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b, std::uint32_t k, std::uint32_t n) {
        upload_scaled(site, block, a, b, k, n, s_);
    }
    // The same with an explicit branch scale (strength x alpha / rank for files with alpha tensors).
    void upload_scaled(Site site, std::uint32_t block, const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b, std::uint32_t k,
        std::uint32_t n, float scale) {
        const std::uint32_t rank = k ? static_cast<std::uint32_t>(a.size() / k) : 0U;
        if (!k || a.size() != static_cast<std::size_t>(rank) * k || !rank || rank % 32U || rank > kMaxRank || b.size() != static_cast<std::size_t>(n) * rank ||
            k % 32U || n % 64U || !(scale == scale))
            throw std::invalid_argument("h3 lora: bad LoRA shape for " + lora_module_name(site, block));
        const Key key{site, block};
        if (w_.count(key)) throw std::invalid_argument("h3 lora: uploaded twice: " + lora_module_name(site, block));
        Branch br{nullptr, nullptr, k, n, rank, scale};
        check(hipMalloc(&br.a, a.size() * 2U), "A");
        if (hipMalloc(&br.b, b.size() * 2U) != hipSuccess) {
            static_cast<void>(hipFree(br.a));
            throw std::runtime_error("h3 lora: hipMalloc B failed");
        }
        w_.emplace(key, br);
        check(hipMemcpy(br.a, a.data(), a.size() * 2U, hipMemcpyHostToDevice), "upload A");
        check(hipMemcpy(br.b, b.data(), b.size() * 2U, hipMemcpyHostToDevice), "upload B");
        bytes_ += (a.size() + b.size()) * 2U;
    }
    bool has(Site site, std::uint32_t block) const { return w_.count(Key{site, block}) != 0U; }

    // y [rows][n] (pitch y_ld) += s * up(down(x)); x [rows][k] (pitch x_ld) = the linear's input.
    void apply(Site site, std::uint32_t block, const std::uint16_t* x, std::uint64_t x_ld, std::uint32_t rows, std::uint16_t* y, std::uint64_t y_ld,
        hipStream_t stream) const {
        const auto it = w_.find(Key{site, block});
        if (it == w_.end()) throw std::invalid_argument("h3 lora: no LoRA uploaded for " + lora_module_name(site, block));
        if (rows > max_rows_) throw std::invalid_argument("h3 lora: rows above max_rows");
        add_bypass(x, x_ld, rows, it->second.k, it->second.a, it->second.b, it->second.n, it->second.scale, t_, y, y_ld, stream, it->second.rank);
    }

private:
    struct Branch { std::uint16_t* a; std::uint16_t* b; std::uint32_t k, n, rank; float scale; };
    using Key = std::pair<Site, std::uint32_t>;
    static void check(hipError_t e, const char* what) {
        if (e != hipSuccess) throw std::runtime_error(std::string("h3 lora: ") + what + ": " + hipGetErrorString(e));
    }
    float s_;
    std::uint32_t max_rows_;
    std::uint16_t* t_ = nullptr;
    std::size_t bytes_ = 0;
    std::map<Key, Branch> w_;
};

}  // namespace strix::h3::levers::lora
