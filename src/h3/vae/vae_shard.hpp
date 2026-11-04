#pragma once
// Two-node VAE decode in gather mode, host side (no transport inside the package).
//
// Rank r decodes a contiguous range of the canonical call order (chunk, tile row, tile column) and emits PIECES:
// a whole temporal chunk inside its range as the spatially assembled canvas, the calls of a chunk cut by the range
// boundary as raw decoder outputs (tiles). Rank 1 serializes its pieces (pack_piece) and sends them to rank 0;
// rank 0 pushes its own and the received pieces into one FrameAssembler, in any order, from any thread.
// The assembler completes a cut chunk once all its tiles are in, then runs the per-chunk temporal step as soon as
// the chunk and its predecessor's tail are present. Every value is produced by the same functions on the same data
// as the one-node path (assemble_chunk, blend, the finalize of assemble_video), so the frames are bit-identical to
// one node for any cut and any arrival order.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "h3/vae/vae_plan.hpp"

namespace strix::h3::vae {

// Calls [call_first, call_last) of the canonical order, all inside temporal chunk `chunk`.
// whole: the piece covers every call of the chunk and carries `canvas` (FP16 [3][4 nt][16 H][16 W]);
// otherwise it carries `tiles` (the decoder outputs of its calls, canonical order).
struct DecodePiece {
    int chunk = 0, call_first = 0, call_last = 0;
    bool whole = false;
    Pix16 canvas;
    std::vector<Pix16> tiles;
};

// Two-rank split at an arbitrary call: rank 0 owns [0, cut), rank 1 [cut, n). cut = n / 2 + shift, clamped to [0, n].
// shift > 0 moves calls to rank 0 (use it when rank 1 carries extra work such as the send of its last piece).
// world 1: the whole range. shift 0 == partition(p, rank, 2).
CallRange split_calls(const DecodePlan& p, int rank, int world, int shift = 0);

// Pieces a call range produces (shapes only; the decoder fills them), in canonical order.
struct PieceSpan { int chunk = 0, call_first = 0, call_last = 0; bool whole = false; };
std::vector<PieceSpan> piece_spans(const DecodePlan& p, CallRange r);

// Wire format of one piece (little-endian): u32 magic 'H3VP', u32 version 1, i32 chunk, call_first, call_last,
// whole, n (tensor count); per tensor i32 C, F, H, W then C*F*H*W u16 FP16 bits. pack_piece -> bytes;
// unpack_piece validates the header and every tensor shape against the plan (throws std::invalid_argument).
std::vector<std::uint8_t> pack_piece(const DecodePiece& piece);
DecodePiece unpack_piece(const DecodePlan& p, const std::uint8_t* data, std::size_t size);

// Collects pieces (any order, any thread: push is serialized internally) and writes the FP32 frames
// [3][out_frames][out_h][out_w] chunk by chunk as soon as each chunk and its predecessor's 5-frame tail are present.
// Throws on a piece outside the plan, a call delivered twice, or a shape mismatch.
class FrameAssembler {
public:
    explicit FrameAssembler(const DecodePlan& p);
    ~FrameAssembler();
    FrameAssembler(const FrameAssembler&) = delete;
    FrameAssembler& operator=(const FrameAssembler&) = delete;
    void push(DecodePiece&& piece);
    bool complete() const;        // every chunk's frames written
    int chunks_done() const;
    std::vector<float> take();    // the frames; throws when not complete

private:
    struct State;
    std::unique_ptr<State> s_;
};

}  // namespace strix::h3::vae
