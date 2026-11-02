// Host parts of the decode driver shared by the GPU decoder and the CPU tests.
#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

#include "h3/vae/vae_decoder.hpp"
#include "h3/vae/vae_threads.hpp"

#include <stdexcept>

namespace strix::h3::vae {

H16 call_input(const HeadH& head, const DecodePlan& p, const H16& zp, const DecoderCall& c) {
    const int Tp = p.T + p.pad_tokens, P = c.nt * c.ny * c.nx;
    H16 box(std::size_t(kLatentC) * P);
    for (int ch = 0; ch < kLatentC; ++ch)
        for (int t = 0; t < c.nt; ++t)
            for (int y = 0; y < c.ny; ++y)
                std::memcpy(box.data() + ((std::size_t(ch) * c.nt + t) * c.ny + y) * c.nx,
                            zp.data() + ((std::size_t(ch) * Tp + c.t0 + t) * p.H + c.y0 + y) * p.W + c.x0, std::size_t(c.nx) * 2);
    return post_quant(head, box, P);
}

// Spatial assembly of every chunk on worker threads (chunks are independent), then the temporal pass in order.
// Same functions on the same tiles: the bits of the serial form.
std::vector<float> blend_calls(const DecodePlan& p, const std::vector<Pix16>& outputs) {
    const int ny = int(p.ty.start.size()), nx = int(p.tx.start.size());
    std::vector<Pix16> canvases(static_cast<std::size_t>(p.num_chunks));
    {
        if (outputs.size() != std::size_t(p.calls())) throw std::invalid_argument("h3 vae blend: need one output per decoder call");
        std::atomic<int> next{0};
        const int nt = std::max(1, std::min<int>(int(std::thread::hardware_concurrency()), p.num_chunks));
        Workers pool;  // exceptions surface from join(); threads joined on every path
        for (int t = 0; t < nt; ++t)
            pool.run([&] {
                for (int c = next++; c < p.num_chunks; c = next++) {
                    maybe_fail(FaultStage::chunk_assembly, c);
                    canvases[std::size_t(c)] = assemble_chunk(p, [&](int i, int j) { return outputs[std::size_t((c * ny + i) * nx + j)]; });
                }
            });
        pool.join();
    }
    return finish_frames(p, std::move(canvases));
}

std::vector<float> finish_frames(const DecodePlan& p, std::vector<Pix16> canvases) {
    if (canvases.size() != std::size_t(p.num_chunks)) throw std::invalid_argument("h3 vae finish_frames: need one canvas per chunk");
    return assemble_video(p, [&](int c) {
        Pix16 v = std::move(canvases[std::size_t(c)]);
        return v;
    });
}

}  // namespace strix::h3::vae
