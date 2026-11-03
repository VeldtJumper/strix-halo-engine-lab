#pragma once
// Attention operand admission of the two-node schedule (host side, shared by the device driver and the CPU tests).
//
// Contract of the attention package: every score s = q.k / sqrt(128) has |s| log2(e) <= 2^14 (Cauchy-Schwarz:
// ||q|| ||k|| log2(e) / sqrt(128) <= 2^14 is sufficient) and every V element is finite with |V| <= 2^100; the kernel's
// bound-rows counter must stay 0. Per DiT block the device measures, on each rank's own rows: max ||q||, max ||k||
// (fp32, rounded up, as their bits: an unsigned max lets NaN dominate) and max |V| (bf16 magnitude bits, & 0x7fff:
// NaN > Inf > finite in that order). Both ranks exchange their measurements and apply the same verdict, so a
// violation stops BOTH ranks with the same error. Every value is checked on its own before any max is taken: a NaN
// can never be dropped by a comparison.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::h3::dist {

// Thrown by the driver (on every rank) when the measured operands leave the attention contract.
class ContractViolation : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline constexpr std::uint32_t kAdmitPerBlock = 3;      // [q norm bits, k norm bits, |V| max bf16 bits]
inline constexpr double kScoreBound = 16384.0;          // 2^14
inline constexpr std::uint16_t kVMaxBits = 0x7180;      // bf16 2^100
inline constexpr std::uint16_t kBf16InfBits = 0x7f80;

struct AdmissionResult {
    double worst = 0;          // max over blocks of max||q|| max||k|| log2(e)/sqrt(128) (both ranks)
    std::uint32_t worst_block = 0;
    double v_max = 0;          // max |V| over blocks and ranks
};

// ops[r] = rank r's measurements [nblocks][3]; bound_rows[r] = rank r's attention bound-rows counter.
// Throws ContractViolation (first violation in block order: non-finite / out-of-domain operands, then bound rows,
// then the score bound).
inline AdmissionResult admit(const std::vector<const std::vector<unsigned>*>& ops, const std::vector<std::uint64_t>& bound_rows,
    std::uint32_t nblocks) {
    auto f32 = [](unsigned b) {
        float v;
        std::memcpy(&v, &b, 4);
        return v;
    };
    auto bf = [](unsigned b) {
        const unsigned u = (b & 0xffffU) << 16;
        float v;
        std::memcpy(&v, &u, 4);
        return v;
    };
    for (const auto* o : ops)
        if (!o || o->size() != std::size_t{nblocks} * kAdmitPerBlock) throw std::invalid_argument("h3 dist admission: measurement size");
    AdmissionResult res;
    for (std::uint32_t i = 0; i < nblocks; ++i) {
        double q = 0, k = 0;
        for (std::size_t r = 0; r < ops.size(); ++r) {
            const unsigned* m = ops[r]->data() + std::size_t{i} * kAdmitPerBlock;
            const float qn = f32(m[0]), kn = f32(m[1]);
            const std::string where = " on block " + std::to_string(i) + " (rank " + std::to_string(r) + ")";
            if (!std::isfinite(qn) || qn < 0) throw ContractViolation("h3 dist: attention operands: non-finite ||q||" + where);
            if (!std::isfinite(kn) || kn < 0) throw ContractViolation("h3 dist: attention operands: non-finite ||k||" + where);
            const unsigned vb = m[2] & 0x7fffU;
            if (vb >= kBf16InfBits) throw ContractViolation("h3 dist: attention operands: non-finite V" + where);
            if (vb > kVMaxBits)
                throw ContractViolation("h3 dist: attention operands: |V| = " + std::to_string(bf(vb)) + " > 2^100" + where);
            q = qn > q ? qn : q;
            k = kn > k ? kn : k;
            res.v_max = bf(vb) > res.v_max ? bf(vb) : res.v_max;
        }
        const double bound = q * k * 1.4426950408889634 / std::sqrt(128.0);
        if (bound > res.worst) {
            res.worst = bound;
            res.worst_block = i;
        }
    }
    std::uint64_t rows = 0;
    for (auto b : bound_rows) rows += b;
    if (rows != 0) throw ContractViolation("h3 dist: attention contract telemetry: " + std::to_string(rows) + " bound rows (must be 0)");
    if (!(res.worst <= kScoreBound))
        throw ContractViolation("h3 dist: attention operands outside the contract: block " + std::to_string(res.worst_block) +
            " ||q|| ||k|| log2(e)/sqrt(128) = " + std::to_string(res.worst) + " > 2^14");
    return res;
}

// --forms of the rank tool: w1 (the 1-node reference) first, then any of w2 / w2swap, each at most once.
inline bool valid_forms(const std::vector<std::string>& f) {
    if (f.empty() || f[0] != "w1") return false;
    for (std::size_t k = 1; k < f.size(); ++k) {
        if (f[k] != "w2" && f[k] != "w2swap") return false;
        for (std::size_t j = 1; j < k; ++j)
            if (f[j] == f[k]) return false;
    }
    return true;
}

}  // namespace strix::h3::dist
