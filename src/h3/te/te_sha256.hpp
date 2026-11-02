#pragma once

// SHA-256 (FIPS 180-4), portable, for the H3 text-encoder pack verification.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strix::h3::te {

class Sha256 {
public:
    // portable = true forces the portable rounds (tests); otherwise x86 SHA extensions when the CPU has them.
    explicit Sha256(bool portable = false);
    void update(const void* data, std::size_t n);
    // Digest of everything passed to update(); the object must not be updated afterwards.
    [[nodiscard]] std::array<std::uint8_t, 32> finish();
    [[nodiscard]] std::string finish_hex();

private:
    void block(const std::uint8_t* p);
    void blocks(const std::uint8_t* p, std::size_t n);
    bool ni_ = false;
    std::uint32_t h_[8];
    std::uint8_t buf_[64];
    std::size_t fill_ = 0;
    std::uint64_t total_ = 0;
};

// True when Sha256 uses the x86 SHA extensions on this CPU.
[[nodiscard]] bool sha256_uses_sha_ni();

// Lowercase hex digest of a byte string / of a whole file (throws std::runtime_error on read errors).
[[nodiscard]] std::string sha256_hex(const void* data, std::size_t n);
[[nodiscard]] std::string sha256_file_hex(const std::string& path);

}  // namespace strix::h3::te
