#pragma once

// MiniMax H3 text encoder: host tokenizer and prompt presentation.
//
// Tokenizer: byte-level BPE of the Qwen2/Qwen3 family read from vocab.json + merges.txt +
// tokenizer_config.json (added tokens), plus the seven H3 extra special tokens (fixed ids
// 151669..151675). Encoding of a text segment:
//   1. split on added tokens (every added token, matched on the raw text, longest first);
//   2. NFC-normalize every other piece;
//   3. pre-split with the Qwen2 pattern
//        (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*
//        |\s*[\r\n]+|\s+(?!\S)|\s+
//   4. map bytes to the byte-level alphabet and apply the merges by rank.
//
// Presentation (no chat template, no BOS/EOS, no padding): a prompt segment is first passed through
// the conditioning front end's text handling: "\(" and "\)" lose their backslash, and the text is
// cut before every "embedding:" that follows a whitespace character, each cut piece tokenized on
// its own. Reference items are presented as
//   image -> "<Picture i>: " <|vision_start|> [vision block] <|vision_end|>
//   audio -> "<Audio j>: "
//   video -> "<Video k>: " then per 2-frame block "<T.T seconds>" <|vision_start|> [block] <|vision_end|>
// followed by the prompt; an empty presentation is the single token 151643.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace strix::h3::te {

inline constexpr std::int32_t kVisionStart = 151652;
inline constexpr std::int32_t kVisionEnd = 151653;
inline constexpr std::int32_t kPadToken = 151643;
// Marks the position of a vision block in a presented sequence (expanded later to the block's
// merged vision tokens).
inline constexpr std::int32_t kVisionSlot = -1;
// Marks one row of a named prompt embedding spliced into the input embeddings.
inline constexpr std::int32_t kEmbeddingRow = -2;

// "embedding:<name>" resolution of the front end. Return the row count of the named embedding
// (rows of width hidden), or -1 when no such embedding exists.
using EmbeddingLookup = std::function<int(std::string_view name)>;

struct Segment {
    std::vector<std::int32_t> ids;           // kEmbeddingRow per spliced embedding row
    std::vector<std::string> embeddings;     // names, in order of appearance
};

class Tokenizer {
public:
    // dir holds vocab.json, merges.txt and tokenizer_config.json.
    static std::shared_ptr<const Tokenizer> load(const std::string& dir);
    ~Tokenizer();

    // Token ids of one text segment as the reference tokenizer call produces them.
    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;
    // The front end's handling of a prompt segment: escape handling, cuts before "embedding:",
    // then encode. lookup == nullptr: no embedding directory configured ("embedding:" words are
    // plain text). With a lookup, an "embedding:" word is resolved as the front end does: the name
    // is its first whitespace-separated field (cut at '<' or '['), the rest of the word is
    // re-joined with single spaces and tokenized after the rows; a missing name is dropped.
    [[nodiscard]] Segment encode_segment(std::string_view text, const EmbeddingLookup* lookup = nullptr) const;
    [[nodiscard]] std::size_t vocab_size() const noexcept;

    // Exposed for tests.
    [[nodiscard]] static std::string nfc(std::string_view utf8);
    [[nodiscard]] static std::vector<std::string> pre_tokenize(std::string_view utf8);

private:
    Tokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct RefItem {
    enum class Kind : std::uint8_t { image, audio, video };
    Kind kind = Kind::image;
    // video: timestamps of the sampled frames (seconds); an odd count is repeat-padded by one.
    // Empty: the reference default i / 2.0 for i < frames (frames sampled at 2 fps).
    std::vector<double> timestamps;
    int frames = 0;
};

struct Presented {
    std::vector<std::int32_t> ids;          // kVisionSlot where a vision block goes, kEmbeddingRow rows
    std::vector<std::string> embeddings;    // names of the spliced embeddings, in order
    std::vector<std::int32_t> slot_item;    // per slot: index into the items list
    std::vector<std::int32_t> slot_frame0;  // per slot: first frame of the block (videos), 0 for images
};

// images: number of keyframe pictures (first/last-frame mode). If items is non-empty it takes
// precedence (reference mode), as in the reference front end.
[[nodiscard]] Presented present(const Tokenizer& tok, std::string_view prompt, int images,
                                const std::vector<RefItem>& items, const EmbeddingLookup* lookup = nullptr);

// "%.1f" of the reference (round half to even on the binary value, like printf).
[[nodiscard]] std::string format_seconds(double t);

// adaLN token tags of an expanded sequence: 1 for text, 0 for a vision block including its
// flanking start/end tokens. slot_sizes: merged vision tokens per slot, in order.
[[nodiscard]] std::vector<std::int32_t> token_tags(const Presented& p, const std::vector<std::int32_t>& slot_sizes);

}  // namespace strix::h3::te
