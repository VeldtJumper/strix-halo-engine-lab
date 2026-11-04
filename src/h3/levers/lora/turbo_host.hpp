#pragma once
// Turbo v4 LoRA (identity h3-turbo-v4-bypass, FAST-class, OFF by default): host-side pieces the block runner needs
// (no HIP): tensor names, the adaLN E-grid injection, the strength handling. The run-time linear branch is bypass.hpp
// (device) / bypass_ref.hpp (host mirror); the device weight holder is turbo.hpp.
//
// adaLN injection (reference: the Turbo node's curve-mode adaln forward patch, pruned base), per adaln_proj and per
// unique-t row m of its input t_emb [M][2688]:
//   idx_m = argmin_j || t_emb[m] - adaln_t_table[j] ||_2    (first minimum; the 1025-row curve table)
//   st    = fp32(egrid[idx_m])                               (silu_t_emb_grid bf16 [1025][2688])
//   u[r]  = fma chain over k ascending of A[r][k] * st[k]    (A = lora_A bf16 [16][2688], fp32, from +0)
//   d[o]  = fma chain over r = 0..15 of Bs[o][r] * u[r]      (Bs = bf16(fp32(lora_B) * fp32(s)), [out][16])
//   x[m][o] = x[m][o] + d[o]                                 (plain fp32 add; x = the adaLN linear's fp32 output)
// then the usual view / chunk into the modulation vectors. The reference sums both products in its library's order
// (fp32): G-class, not bitwise. d depends only on (module, idx), so a runner can cache it per sampling step.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/gemm/ref.hpp"

namespace strix::h3::levers::lora {

inline constexpr const char* kTurboIdentity = "h3-turbo-v4-bypass";
constexpr std::uint32_t kAdalnRank = 16U;
constexpr std::uint32_t kTembDim = 2688U;
constexpr std::uint32_t kEgridRows = 1025U;

// Linear sites with a run-time branch. DiT fc2: Turbo v4 uses merged weights (merged_fc2_s070.safetensors); the other
// step LoRAs use a run-time fc2 branch after the int8 GEMM (kFc2).
enum class Site : std::uint8_t { kQkv, kOut, kFc1, kRefQkv, kRefOut, kRefFc1, kRefFc2, kFc2 };
inline bool is_refiner(Site s) { return s == Site::kRefQkv || s == Site::kRefOut || s == Site::kRefFc1 || s == Site::kRefFc2; }

// Module name inside the LoRA file (tensors "<name>.lora_A.weight" [r][k], "<name>.lora_B.weight" [n][r]; some files
// prefix every key with "diffusion_model.").
inline std::string lora_module_name(Site s, std::uint32_t block) {
    const std::string b = std::to_string(block);
    switch (s) {
        case Site::kQkv: return "blocks." + b + ".attn.qkv_proj";
        case Site::kOut: return "blocks." + b + ".attn.out_proj";
        case Site::kFc1: return "blocks." + b + ".mlp.fc1";
        case Site::kRefQkv: return "token_refiner.blocks." + b + ".attn.qkv_proj";
        case Site::kRefOut: return "token_refiner.blocks." + b + ".attn.out_proj";
        case Site::kRefFc1: return "token_refiner.blocks." + b + ".mlp.fc1";
        case Site::kRefFc2: return "token_refiner.blocks." + b + ".mlp.fc2";
        case Site::kFc2: return "blocks." + b + ".mlp.fc2";
    }
    throw std::invalid_argument("h3 lora: bad site");
}
// adaLN modules: blocks.N.adaln_proj.linear (block < 50) and final_layer.adaln_proj.linear (final == true).
inline std::string adaln_module_name(std::uint32_t block, bool final) {
    return final ? std::string("final_layer.adaln_proj.linear") : "blocks." + std::to_string(block) + ".adaln_proj.linear";
}

// Nearest curve-table row (first minimum). Distances in fp64 over fp32 inputs: the reference's fp32 cdist can pick
// the other neighbour only when the two distances agree to ~1e-7 relative (t_emb within fp32 noise of the midpoint
// of two rows); the runner logs idx per step so a G-class mismatch there is visible.
inline std::uint32_t adaln_nearest_row(const float* t_emb, const float* table, std::uint32_t rows, std::uint32_t dim) {
    if (!rows || !dim) throw std::invalid_argument("h3 lora: empty adaLN table");
    std::uint32_t best = 0;
    double best_d = INFINITY;
    for (std::uint32_t j = 0; j < rows; ++j) {
        double d = 0.0;
        for (std::uint32_t k = 0; k < dim; ++k) {
            const double v = static_cast<double>(t_emb[k]) - static_cast<double>(table[static_cast<std::size_t>(j) * dim + k]);
            d += v * v;
        }
        if (d < best_d) { best_d = d; best = j; }
    }
    return best;
}

// Bs = bf16(fp32(B) * fp32(s)), as the reference scales the bf16 tensor by the strength.
inline std::vector<std::uint16_t> scale_bf16(const std::vector<std::uint16_t>& b, float s) {
    std::vector<std::uint16_t> o(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) o[i] = gemm::f32_to_bf16(gemm::bf16_to_f32(b[i]) * s);
    return o;
}

// d[out] for one E-grid row (see the header). a: [16][2688] bf16, bs: [out][16] bf16 (already scaled), st_row: bf16 [2688].
inline void adaln_delta(const std::uint16_t* st_row, const std::uint16_t* a, const std::uint16_t* bs, std::uint32_t out, float* d) {
    float u[kAdalnRank];
    for (std::uint32_t r = 0; r < kAdalnRank; ++r) {
        float acc = 0.0F;
        for (std::uint32_t k = 0; k < kTembDim; ++k)
            acc = std::fma(gemm::bf16_to_f32(a[static_cast<std::size_t>(r) * kTembDim + k]), gemm::bf16_to_f32(st_row[k]), acc);
        u[r] = acc;
    }
    for (std::uint32_t o = 0; o < out; ++o) {
        float acc = 0.0F;
        for (std::uint32_t r = 0; r < kAdalnRank; ++r) acc = std::fma(gemm::bf16_to_f32(bs[static_cast<std::size_t>(o) * kAdalnRank + r]), u[r], acc);
        d[o] = acc;
    }
}

// Whole injection for one adaLN module. The nearest-row search runs in the CURVE-TABLE basis: t_emb [m][tdim] fp32 =
// the rows fed to that adaLN linear and table [table_rows][tdim] fp32 = adaln_t_table (pruned checkpoints: [1025][8]);
// the injected vector comes from the E-grid egrid [egrid_rows][2688] bf16 (silu_t_emb_grid), one row per table row,
// so table_rows must equal egrid_rows (1025). x [m][out] fp32 (the base adaLN output) gets d added. Shapes are
// validated (std::invalid_argument on a mismatch). Returns the chosen rows.
inline std::vector<std::uint32_t> adaln_inject(float* x, std::uint32_t m, std::uint32_t out, const float* t_emb, std::uint32_t tdim, const float* table,
    std::uint32_t table_rows, const std::uint16_t* egrid, std::uint32_t egrid_rows, const std::uint16_t* a, const std::uint16_t* bs) {
    if (!x || !t_emb || !table || !egrid || !a || !bs || !m || !out || !tdim) throw std::invalid_argument("h3 lora: adaLN inject arguments");
    if (table_rows != egrid_rows || egrid_rows != kEgridRows)
        throw std::invalid_argument("h3 lora: adaLN curve table rows (" + std::to_string(table_rows) + ") / E-grid rows (" + std::to_string(egrid_rows) +
                                    ") must both be " + std::to_string(kEgridRows));
    std::vector<std::uint32_t> idx(m);
    std::vector<float> d(out);
    for (std::uint32_t i = 0; i < m; ++i) {
        idx[i] = adaln_nearest_row(t_emb + static_cast<std::size_t>(i) * tdim, table, table_rows, tdim);
        adaln_delta(egrid + static_cast<std::size_t>(idx[i]) * kTembDim, a, bs, out, d.data());
        float* xr = x + static_cast<std::size_t>(i) * out;
        for (std::uint32_t o = 0; o < out; ++o) xr[o] = xr[o] + d[o];
    }
    return idx;
}

}  // namespace strix::h3::levers::lora
