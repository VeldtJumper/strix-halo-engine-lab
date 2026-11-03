// Host reference of the two-node schedule (see host_dist.hpp).
#include "h3/dist/host_dist.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>

#include "h3/block/parallel.hpp"
#include "h3/dist/host_attn.hpp"

namespace strix::h3::dist {

using block::kFfn;
using block::kHidden;
using block::kNormEps;

ForwardTables ForwardTables::build(const block::GlobalWeights& g, const block::PackedLayout& L, const block::ForwardPlan& p) {
    ForwardTables t;
    t.t_emb = block::curve_t_emb(g, p.t_vals);
    t.n_t = p.n_t();
    t.rope = block::rope_table(block::rope_angles(L.position_ids, g.inv_freq));
    t.mod_row = block::mod_row_of(p.mod_segments, L.seq_len);
    return t;
}

std::shared_ptr<const block::BlockWeights> SharedBlockSource::block(std::uint32_t i) {
    std::lock_guard lk(mu_);
    for (auto& e : cache_)
        if (e.first == i) return e.second;
    auto b = inner_.block(i);
    cache_.emplace_back(i, b);
    if (cache_.size() > keep_) cache_.erase(cache_.begin());
    return b;
}

void dist_blocks_host(block::BlockSource& blocks, std::uint32_t nblocks, const ForwardTables& t, const Split& sp, const Role& role,
    std::vector<std::uint16_t>& x, block::LinearOp& lin, KvExchange* ex, DistTrace* trace) {
    const std::uint32_t r0 = role.row0(sp), n = role.rows(sp);
    const std::size_t H = kHidden;
    if (role.world != 1U && role.world != 2U) throw std::invalid_argument("h3 dist: world must be 1 or 2");
    if (role.world == 2U && !ex) throw std::invalid_argument("h3 dist: two ranks need an exchange");
    if (x.size() != std::size_t{n} * H) throw std::invalid_argument("h3 dist: residual rows do not match the rank's rows");
    if (t.mod_row.size() != sp.S || t.rope.c.size() != std::size_t{sp.S} * 48U) throw std::invalid_argument("h3 dist: tables");

    std::vector<std::uint16_t> h(n * H), qkv(std::size_t{n} * 3 * kInner), q(std::size_t{n} * kInner), ao(q.size()), o(n * H),
        f1(std::size_t{n} * 2 * kFfn), sw(std::size_t{n} * kFfn);
    // K / V^T storage: world 2 = the exchange messages (TX and RX, double-buffered by block parity); world 1 = one
    // local buffer holding all rows (both key blocks are views of it).
    const std::size_t msg_elems = sp.message_bytes() / 2U;
    std::vector<std::uint16_t> tx[2], rx[2], kloc, vtloc;
    const std::uint64_t pitch1 = (std::uint64_t{sp.S} + kCutAlign - 1U) / kCutAlign * kCutAlign;
    if (role.world == 2U) {
        for (int j = 0; j < 2; ++j) { tx[j].assign(msg_elems, 0); rx[j].assign(msg_elems, 0); }
    } else {
        kloc.assign(std::size_t{sp.S} * kInner, 0);
        vtloc.assign(std::size_t{kInner} * pitch1, 0);
    }
    std::vector<attn::Partial> parts[2];
    for (auto& p : parts) p.resize(std::size_t{n} * kHeads);
    if (trace) {
        trace->hash.assign(nblocks, {0, 0});
        trace->block_s.assign(nblocks, 0.0);
        trace->finish_wait_s.assign(nblocks, 0.0);
    }

    // A failure after start() must not free the TX buffers while the transfer still reads them: abort the exchange
    // (it returns once the progress thread has let go of tx) before unwinding.
    struct AbortGuard {
        KvExchange* ex;
        bool armed = false;
        ~AbortGuard() {
            if (armed && ex) ex->abort("h3 dist: rank failed with an exchange in flight");
        }
    } guard{role.world == 2U ? ex : nullptr};
    for (std::uint32_t i = 0; i < nblocks; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto b = blocks.block(i);
        const block::ModVectors mv = block::mod_vectors(block::adaln_block_f32(*b, t.t_emb, t.n_t), t.n_t * 3U);
        const std::uint32_t* mrow = t.mod_row.data() + r0;
        // ---- attention half: norm1 + modulation, QKV (local rows)
        block::parallel_for(n, [&](std::uint64_t r) {
            block::rmsnorm_row(x.data() + r * H, b->norm1.data(), kHidden, kNormEps, h.data() + r * H);
            block::mod_scale_shift_row(h.data() + r * H, mv.s1_msa.data() + mrow[r] * H, mv.shift_msa.data() + mrow[r] * H, kHidden);
        });
        lin.run(b->qkv, h.data(), n, qkv.data());
        // ---- K4: Q, K (q/k norm + RoPE of the global row) and V^T, written straight into the TX message
        std::uint16_t* kdst;
        std::uint16_t* vtdst;
        std::uint64_t pitch;
        if (role.world == 2U) {
            kdst = tx[i & 1U].data();
            vtdst = kdst + sp.k_bytes() / 2U;
            pitch = sp.vt_pitch();
        } else {
            kdst = kloc.data();
            vtdst = vtloc.data();
            pitch = pitch1;
        }
        block::parallel_for(n, [&](std::uint64_t r) {
            const std::uint16_t* row = qkv.data() + r * 3U * kInner;
            const std::uint16_t* c = t.rope.c.data() + (r0 + r) * 48U;
            const std::uint16_t* s = t.rope.s.data() + (r0 + r) * 48U;
            for (std::uint32_t hh = 0; hh < kHeads; ++hh) {
                block::qk_norm_rope_head(row + hh * kHeadDim, b->q_norm.data(), c, s, q.data() + r * kInner + hh * kHeadDim);
                block::qk_norm_rope_head(row + kInner + hh * kHeadDim, b->k_norm.data(), c, s, kdst + r * kInner + hh * kHeadDim);
                for (std::uint32_t d = 0; d < kHeadDim; ++d)
                    vtdst[(std::size_t{hh} * kHeadDim + d) * pitch + r] = row[2U * kInner + hh * kHeadDim + d];
            }
        });
        // ---- exchange start, local-block partial, finish, remote-block partial, canonical merge
        const attn::Partial* merged[2];
        if (role.world == 2U) {
            const unsigned own = role.block(), peer = role.peer_block();
            guard.armed = true;
            ex->start(reinterpret_cast<const std::byte*>(tx[i & 1U].data()), sp.message_bytes(), i);
            host_partials(q.data(), n, KeyView{kdst, vtdst, pitch, sp.rows(own)}, parts[own].data());
            const auto w0 = std::chrono::steady_clock::now();
            ex->finish(reinterpret_cast<std::byte*>(rx[i & 1U].data()));
            guard.armed = false;
            if (trace) trace->finish_wait_s[i] = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
            const std::uint16_t* rk = rx[i & 1U].data();
            host_partials(q.data(), n, KeyView{rk, rk + sp.k_bytes() / 2U, sp.vt_pitch(), sp.rows(peer)}, parts[peer].data());
        } else {
            for (unsigned kb = 0; kb < 2; ++kb)
                host_partials(q.data(), n, KeyView{kdst + std::size_t{sp.first(kb)} * kInner, vtdst + sp.first(kb), pitch, sp.rows(kb)},
                    parts[kb].data());
        }
        merged[0] = parts[0].data();
        merged[1] = parts[1].data();
        host_merge(merged, 2, n, ao.data());
        // ---- out projection, gated residual, MLP half (all row-local)
        lin.run(b->out, ao.data(), n, o.data());
        for (std::uint32_t r = 0; r < n; ++r) block::gate_row(x.data() + r * H, o.data() + r * H, mv.gate_msa.data() + mrow[r] * H, kHidden);
        block::parallel_for(n, [&](std::uint64_t r) {
            block::rmsnorm_row(x.data() + r * H, b->norm2.data(), kHidden, kNormEps, h.data() + r * H);
            block::mod_scale_shift_row(h.data() + r * H, mv.s1_mlp.data() + mrow[r] * H, mv.shift_mlp.data() + mrow[r] * H, kHidden);
        });
        lin.run(b->fc1, h.data(), n, f1.data());
        block::swiglu_host(f1.data(), n, kFfn, sw.data());
        lin.run(b->fc2, sw.data(), n, o.data());
        for (std::uint32_t r = 0; r < n; ++r) block::gate_row(x.data() + r * H, o.data() + r * H, mv.gate_mlp.data() + mrow[r] * H, kHidden);
        if (trace) {
            for (unsigned kb = 0; kb < 2; ++kb) {
                if (role.world == 2U && kb != role.block()) continue;
                const std::size_t off = role.world == 2U ? 0U : std::size_t{sp.first(kb)} * H;
                trace->hash[i][kb] = fnv1a64(x.data() + off, std::size_t{sp.rows(kb)} * H * 2U);
            }
            trace->block_s[i] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    }
}

}  // namespace strix::h3::dist
