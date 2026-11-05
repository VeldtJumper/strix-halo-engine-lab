#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace strix::ple_reclaim {

// Actual registrations, not tensor payloads. File identity is (st_dev, st_ino),
// so aliases through another mapping or descriptor cannot evade the check.
struct RegisteredBacking {
    std::string name;
    std::uint64_t address = 0, bytes = 0;
    std::uint64_t device = 0, inode = 0, file_offset = 0;
};
struct DiscardRange {
    std::uint64_t address = 0, bytes = 0;
    std::uint64_t device = 0, inode = 0, file_offset = 0;
};
class DiscardSafetyError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
inline std::uint64_t discard_end(std::uint64_t begin, std::uint64_t bytes) {
    if (bytes > UINT64_MAX - begin) throw DiscardSafetyError("PLE discard range overflow");
    return begin + bytes;
}
inline bool discard_page_overlap(std::uint64_t a, std::uint64_t an,
    std::uint64_t b, std::uint64_t bn, std::uint64_t page) {
    if (!an || !bn) return false;
    const auto last_a = discard_end(a, an) - 1U;
    const auto last_b = discard_end(b, bn) - 1U;
    return a / page <= last_b / page && b / page <= last_a / page;
}
// Check each syscall's actual set independently. fadvise uses file offsets across
// aliases; madvise also needs virtual disjointness, even for anonymous storage.
// Empty/whole-file (length=0) advice is deliberately forbidden.
inline void validate_discard_set(std::uint64_t page,
    std::span<const DiscardRange> madvise, std::span<const DiscardRange> fadvise,
    std::span<const RegisteredBacking> registered) {
    if (!page || (page & (page - 1U))) throw DiscardSafetyError("PLE discard invalid page size");
    const auto check = [&](const DiscardRange& d, bool virtual_check) {
        if (!d.bytes || !d.inode) throw DiscardSafetyError("PLE discard missing bounded file backing");
        (void)discard_end(d.address, d.bytes);
        (void)discard_end(d.file_offset, d.bytes);
        if (d.file_offset % page || d.bytes % page || (virtual_check && d.address % page))
            throw DiscardSafetyError("PLE discard must contain only complete PLE pages");
        for (const auto& r : registered) {
            (void)discard_end(r.address, r.bytes);
            (void)discard_end(r.file_offset, r.bytes);
            const bool virtual_overlap = virtual_check && discard_page_overlap(d.address,d.bytes,r.address,r.bytes,page);
            const bool file_overlap = r.inode && d.device == r.device && d.inode == r.inode &&
                discard_page_overlap(d.file_offset,d.bytes,r.file_offset,r.bytes,page);
            if (virtual_overlap || file_overlap)
                throw DiscardSafetyError("PLE discard overlaps registered backing: " + r.name);
        }
    };
    for (const auto& d : madvise) check(d,true);
    for (const auto& d : fadvise) check(d,false);
}
} // namespace strix::ple_reclaim
