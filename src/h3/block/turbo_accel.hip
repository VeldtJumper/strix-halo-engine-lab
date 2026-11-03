// Accelerated forms of the DiT forward: Turbo v4 bypass data + merged-fc2 block source (see turbo_accel.hpp).
#include <hip/hip_runtime.h>

#include "h3/block/turbo_accel.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>

#include "h3/levers/lora/turbo.hpp"

namespace strix::h3::block {

namespace lv = strix::h3::levers::lora;

MergedFc2Source::MergedFc2Source(const SafeTensors& base, const SafeTensors& merged, std::uint32_t basis, float expect_strength, LoadPolicy pol)
    : base_(base), merged_(merged), basis_(basis), n_(count_blocks(base)), pol_(pol) {
    const auto& md = merged.metadata();
    const auto it = md.find("strength");
    if (it == md.end()) throw std::runtime_error("merged fc2: no strength in the file metadata");
    if (std::stof(it->second) != expect_strength)
        throw std::runtime_error("merged fc2: file strength " + it->second + " != requested " + std::to_string(expect_strength));
    for (std::uint32_t i = 0; i < n_; ++i)
        if (!merged.find("blocks." + std::to_string(i) + ".mlp.fc2.weight")) throw std::runtime_error("merged fc2: block " + std::to_string(i) + " missing");
}

std::shared_ptr<const BlockWeights> MergedFc2Source::block(std::uint32_t i) {
    auto b = std::make_shared<BlockWeights>(load_block(base_, i, basis_, pol_));
    b->fc2 = detail::read_qlinear(merged_, "blocks." + std::to_string(i) + ".mlp.fc2", kHidden, kFfn);
    return b;
}

TurboV4::TurboV4(float strength, std::uint32_t max_rows, std::string identity)
    : strength_(strength), identity_(std::move(identity)), dev_(std::make_unique<lv::TurboV4Device>(strength, max_rows)) {}

std::unique_ptr<TurboV4> TurboV4::step_lora(const std::string& id, const std::string& file, float strength, std::uint32_t max_rows,
    std::uint32_t dit_blocks, std::uint32_t refiner_blocks) {
    std::unique_ptr<TurboV4> t(new TurboV4(strength, max_rows, id));
    std::uint32_t r = 0;
    for_each_step_lora_branch(id, file, strength, dit_blocks, refiner_blocks, [&](const HostLoraBranch& br) {
        const auto site = static_cast<lv::Site>(br.site);
        t->dev_->upload_scaled(site, br.block, br.a, br.b, br.k, br.n, br.scale);
        r = std::max(r, br.rank);
        if (site == lv::Site::kFc2) t->has_fc2_ = true;
    });
    t->max_rank_ = r;
    return t;
}

TurboV4::TurboV4(const std::string& lora, const std::string& egrid, float strength, std::uint32_t max_rows, std::uint32_t dit_blocks,
    std::uint32_t refiner_blocks)
    : TurboV4(strength, max_rows, kIdentity) {
    const SafeTensors st(lora);
    auto shape2 = [&](const std::string& n) {
        const StTensor& t = st.at(n);
        if (t.shape.size() != 2 || t.dtype != StDtype::bf16) throw std::runtime_error("turbo: bad tensor " + n);
        return std::pair<std::uint32_t, std::uint32_t>{static_cast<std::uint32_t>(t.shape[0]), static_cast<std::uint32_t>(t.shape[1])};
    };
    auto up = [&](lv::Site site, std::uint32_t blk) {
        const std::string m = lv::lora_module_name(site, blk);
        const auto sa = shape2(m + ".lora_A.weight"), sb = shape2(m + ".lora_B.weight");
        if (sa.first != lv::kRank || sb.second != lv::kRank) throw std::runtime_error("turbo: rank of " + m);
        dev_->upload(site, blk, st.read_vec<std::uint16_t>(m + ".lora_A.weight", StDtype::bf16), st.read_vec<std::uint16_t>(m + ".lora_B.weight", StDtype::bf16),
            sa.second, sb.first);
    };
    for (std::uint32_t b = 0; b < dit_blocks; ++b)
        for (lv::Site s : {lv::Site::kQkv, lv::Site::kOut, lv::Site::kFc1}) up(s, b);
    for (std::uint32_t b = 0; b < refiner_blocks; ++b)
        for (lv::Site s : {lv::Site::kRefQkv, lv::Site::kRefOut, lv::Site::kRefFc1, lv::Site::kRefFc2}) up(s, b);
    auto ada = [&](std::uint32_t blk, bool final) {
        const std::string m = lv::adaln_module_name(blk, final);
        const auto sa = shape2(m + ".lora_A.weight"), sb = shape2(m + ".lora_B.weight");
        if (sa.first != lv::kAdalnRank || sa.second != lv::kTembDim || sb.second != lv::kAdalnRank) throw std::runtime_error("turbo: adaLN shape of " + m);
        Ada a;
        a.a = st.read_vec<std::uint16_t>(m + ".lora_A.weight", StDtype::bf16);
        a.bs = lv::scale_bf16(st.read_vec<std::uint16_t>(m + ".lora_B.weight", StDtype::bf16), strength);
        a.out = sb.first;
        return a;
    };
    for (std::uint32_t b = 0; b < dit_blocks; ++b) {
        ada_.push_back(ada(b, false));
        if (ada_.back().out != 6U * kHidden * 3U) throw std::runtime_error("turbo: block adaLN out width");
    }
    final_ada_ = ada(0, true);
    if (final_ada_.out != 2U * kHidden) throw std::runtime_error("turbo: final adaLN out width");
    const SafeTensors eg(egrid);
    const StTensor& t = eg.at("silu_t_emb_grid");
    if (t.dtype != StDtype::bf16 || t.shape.size() != 2 || t.shape[0] != lv::kEgridRows || t.shape[1] != lv::kTembDim)
        throw std::runtime_error("turbo: E-grid shape");
    egrid_ = eg.read_vec<std::uint16_t>("silu_t_emb_grid", StDtype::bf16);
}

TurboV4::~TurboV4() = default;

std::size_t TurboV4::device_bytes() const { return dev_->device_bytes(); }

TurboV4::AdalnDeltas TurboV4::adaln_deltas(const GlobalWeights& g, const std::vector<float>& t_emb, std::uint32_t nt) const {
    if (!has_adaln()) throw std::logic_error("turbo: " + identity_ + " has no adaLN part");
    if (g.curve_grid != lv::kEgridRows) throw std::runtime_error("turbo: curve table rows != E-grid rows");
    if (t_emb.size() != static_cast<std::size_t>(nt) * g.basis) throw std::runtime_error("turbo: t_emb size");
    AdalnDeltas d;
    d.nt = nt;
    d.block_out = ada_.empty() ? 0 : ada_[0].out;
    d.final_out = final_ada_.out;
    d.idx.resize(nt);
    for (std::uint32_t m = 0; m < nt; ++m)
        d.idx[m] = lv::adaln_nearest_row(t_emb.data() + static_cast<std::size_t>(m) * g.basis, g.adaln_t_table.data(), g.curve_grid, g.basis);
    d.blocks.resize(ada_.size() * nt * d.block_out);
    d.final.resize(static_cast<std::size_t>(nt) * d.final_out);
    // every (module, row) is an independent fixed-order computation: the thread split does not change any bit
    const std::size_t jobs = ada_.size() + 1;
    const unsigned nth = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned w = 0; w < nth; ++w)
        th.emplace_back([&, w]() {
            for (std::size_t j = w; j < jobs; j += nth)
                for (std::uint32_t m = 0; m < nt; ++m) {
                    const std::uint16_t* st = egrid_.data() + static_cast<std::size_t>(d.idx[m]) * lv::kTembDim;
                    if (j < ada_.size())
                        lv::adaln_delta(st, ada_[j].a.data(), ada_[j].bs.data(), d.block_out,
                            d.blocks.data() + (j * nt + m) * static_cast<std::size_t>(d.block_out));
                    else
                        lv::adaln_delta(st, final_ada_.a.data(), final_ada_.bs.data(), d.final_out, d.final.data() + static_cast<std::size_t>(m) * d.final_out);
                }
        });
    for (auto& t : th) t.join();
    return d;
}

}  // namespace strix::h3::block
