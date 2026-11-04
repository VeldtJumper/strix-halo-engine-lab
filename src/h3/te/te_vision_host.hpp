#pragma once

// MiniMax H3 text encoder: host side of the vision blocks.
//   - image/video-pair preprocessing to flattened patches (resize to multiples of 32 with the
//     reference's smart-resize rule, bilinear (half-pixel centers, no antialias), (x - 0.5) / 0.5,
//     patch 16 x temporal 2, 2x2 merge order)
//   - expansion of vision slots into merged-token rows, 3-axis MRoPE positions, token tags.

#include <cstdint>
#include <vector>

namespace strix::h3::te {

struct Frames {          // [t][h][w][3] in [0, 1], t = 1 (image) or 2 (video block)
    int t = 1, h = 0, w = 0;
    std::vector<float> rgb;
};

struct Patches {         // [grid_h * grid_w][3 * 2 * 16 * 16], rows in 2x2 merge order
    int grid_h = 0, grid_w = 0;
    std::vector<float> data;
    [[nodiscard]] int merged_tokens() const { return (grid_h / 2) * (grid_w / 2); }
};

// Target size of the reference resize (factor 32, min 3136, max 12845056 pixels).
void smart_resize(int h, int w, int& h_bar, int& w_bar);
// Bilinear resize of one [h][w][3] frame (reference semantics, fp32).
std::vector<float> resize_bilinear(const float* rgb, int h, int w, int oh, int ow);
// Image (t = 1, the frame fills both temporal slots) or video pair (t = 2).
Patches make_patches(const Frames& f);

struct VisionSpan { std::uint32_t index = 0, size = 0; int grid_h = 0, grid_w = 0; };

// Expanded layout of a presented sequence: ids with kVisionSlot (-1) replaced by `merged` rows of
// -1 each; spans in order; MRoPE positions [3][S] exactly as the reference builds them (text-only:
// empty); tags (1 text, 0 vision incl. start/end).
struct Layout {
    std::vector<std::int32_t> ids;
    std::vector<VisionSpan> spans;
    std::vector<float> positions;
    std::vector<std::int32_t> tags;
};
Layout expand_layout(const std::vector<std::int32_t>& presented_ids, const std::vector<Patches>& blocks);

}  // namespace strix::h3::te
