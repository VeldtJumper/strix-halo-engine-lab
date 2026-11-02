#include "h3/te/te_tokenizer.hpp"

#include "strix/serve/json.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace strix::h3::te {
namespace {

struct TeRange { std::uint32_t first, last; std::uint8_t value; };
struct TeDecomp { std::uint32_t cp; std::uint32_t offset; std::uint8_t length; };
struct TeCompose { std::uint32_t first, second, composite; };
#include "te_unicode_tables.inc"

constexpr std::uint8_t kLetter = 1U, kMark = 2U, kNumber = 4U;

template <std::size_t N>
std::uint8_t range_value(const TeRange (&table)[N], std::uint32_t cp) {
    std::size_t lo = 0, hi = N;
    while (lo < hi) {
        const auto mid = (lo + hi) / 2U;
        if (table[mid].last < cp) lo = mid + 1U;
        else hi = mid;
    }
    return lo < N && table[lo].first <= cp ? table[lo].value : 0U;
}
std::uint8_t category(std::uint32_t cp) { return range_value(kTeCategory, cp); }
std::uint8_t ccc(std::uint32_t cp) { return range_value(kTeCcc, cp); }

// The regex engine's \s: White_Space.
bool space(std::uint32_t cp) {
    return (cp >= 0x09U && cp <= 0x0DU) || cp == 0x20U || cp == 0x85U || cp == 0xA0U || cp == 0x1680U ||
        (cp >= 0x2000U && cp <= 0x200AU) || cp == 0x2028U || cp == 0x2029U || cp == 0x202FU || cp == 0x205FU ||
        cp == 0x3000U;
}

constexpr std::uint32_t kInvalid = 0x80000000U;  // flag on a code point decoded from an invalid byte

std::vector<std::uint32_t> decode_utf8(std::string_view s) {
    std::vector<std::uint32_t> out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        const auto b0 = static_cast<unsigned char>(s[i]);
        std::size_t len = b0 < 0x80U ? 1U : (b0 >> 5U) == 6U ? 2U : (b0 >> 4U) == 14U ? 3U : (b0 >> 3U) == 30U ? 4U : 0U;
        std::uint32_t cp = 0;
        bool ok = len != 0U && i + len <= s.size();
        if (ok) {
            cp = len == 1U ? b0 : len == 2U ? (b0 & 0x1FU) : len == 3U ? (b0 & 0x0FU) : (b0 & 0x07U);
            for (std::size_t k = 1; k < len && ok; ++k) {
                const auto b = static_cast<unsigned char>(s[i + k]);
                if ((b & 0xC0U) != 0x80U) ok = false;
                cp = (cp << 6U) | (b & 0x3FU);
            }
            if (ok && ((len == 2U && cp < 0x80U) || (len == 3U && (cp < 0x800U || (cp >= 0xD800U && cp <= 0xDFFFU))) ||
                       (len == 4U && (cp < 0x10000U || cp > 0x10FFFFU)))) ok = false;
        }
        if (!ok) { out.push_back(kInvalid | b0); ++i; continue; }
        out.push_back(cp);
        i += len;
    }
    return out;
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp & kInvalid) { out.push_back(static_cast<char>(cp & 0xFFU)); return; }
    if (cp < 0x80U) out.push_back(static_cast<char>(cp));
    else if (cp < 0x800U) { out.push_back(static_cast<char>(0xC0U | (cp >> 6U))); out.push_back(static_cast<char>(0x80U | (cp & 0x3FU))); }
    else if (cp < 0x10000U) {
        out.push_back(static_cast<char>(0xE0U | (cp >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (cp >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    }
}

// Hangul syllable algorithm constants.
constexpr std::uint32_t SBase = 0xAC00U, LBase = 0x1100U, VBase = 0x1161U, TBase = 0x11A7U;
constexpr std::uint32_t LCount = 19U, VCount = 21U, TCount = 28U, NCount = VCount * TCount, SCount = LCount * NCount;

void decompose(std::uint32_t cp, std::vector<std::uint32_t>& out) {
    if (cp >= SBase && cp < SBase + SCount) {
        const auto s = cp - SBase;
        out.push_back(LBase + s / NCount);
        out.push_back(VBase + (s % NCount) / TCount);
        if (s % TCount) out.push_back(TBase + s % TCount);
        return;
    }
    const auto* it = std::lower_bound(std::begin(kTeDecomp), std::end(kTeDecomp), cp,
        [](const TeDecomp& d, std::uint32_t v) { return d.cp < v; });
    if (it != std::end(kTeDecomp) && it->cp == cp) {
        out.insert(out.end(), kTeDecompData + it->offset, kTeDecompData + it->offset + it->length);
        return;
    }
    out.push_back(cp);
}

std::uint32_t compose_pair(std::uint32_t a, std::uint32_t b) {
    if (a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount)
        return SBase + ((a - LBase) * VCount + (b - VBase)) * TCount;
    if (a >= SBase && a < SBase + SCount && (a - SBase) % TCount == 0U && b > TBase && b < TBase + TCount)
        return a + (b - TBase);
    const auto* it = std::lower_bound(std::begin(kTeCompose), std::end(kTeCompose), std::pair{a, b},
        [](const TeCompose& c, std::pair<std::uint32_t, std::uint32_t> v) {
            return c.first < v.first || (c.first == v.first && c.second < v.second);
        });
    if (it != std::end(kTeCompose) && it->first == a && it->second == b) return it->composite;
    return 0;
}

std::vector<std::uint32_t> nfc_points(const std::vector<std::uint32_t>& in) {
    std::vector<std::uint32_t> d;
    d.reserve(in.size());
    for (const auto cp : in) {
        if (cp & kInvalid) d.push_back(cp);
        else decompose(cp, d);
    }
    // canonical ordering: stable sort of every run of non-starters by combining class
    for (std::size_t i = 0; i < d.size();) {
        if ((d[i] & kInvalid) || ccc(d[i]) == 0U) { ++i; continue; }
        std::size_t j = i;
        while (j < d.size() && !(d[j] & kInvalid) && ccc(d[j]) != 0U) ++j;
        std::stable_sort(d.begin() + static_cast<std::ptrdiff_t>(i), d.begin() + static_cast<std::ptrdiff_t>(j),
            [](std::uint32_t a, std::uint32_t b) { return ccc(a) < ccc(b); });
        i = j;
    }
    // canonical composition
    std::vector<std::uint32_t> out;
    out.reserve(d.size());
    // last_ccc: combining class of the last character kept after the starter, -1 when none (the
    // candidate is adjacent to the starter). A candidate is blocked when a kept character between
    // has class 0 or a class >= its own.
    std::size_t starter = SIZE_MAX;
    int last_ccc = -1;
    for (const auto cp : d) {
        const int c = (cp & kInvalid) ? 0 : ccc(cp);
        if (starter != SIZE_MAX && !(cp & kInvalid) && !(out[starter] & kInvalid)) {
            const bool unblocked = last_ccc == -1 || (last_ccc != 0 && last_ccc < c);
            if (unblocked) {
                if (const auto comp = compose_pair(out[starter], cp); comp != 0U) {
                    out[starter] = comp;
                    continue;
                }
            }
        }
        if (c == 0) { starter = out.size(); last_ccc = -1; }
        else last_ccc = c;
        out.push_back(cp);
    }
    return out;
}

bool is_letter(std::uint32_t cp) { return !(cp & kInvalid) && (category(cp) & kLetter); }
bool is_number(std::uint32_t cp) { return !(cp & kInvalid) && (category(cp) & kNumber); }
bool is_space(std::uint32_t cp) { return !(cp & kInvalid) && space(cp); }

// Case-folded ASCII letter for the contraction alternatives ((?i) also folds U+017F to 's').
char fold(std::uint32_t cp) {
    if (cp == 0x17FU) return 's';
    if (cp == 0x212AU) return 'k';
    return cp < 0x80U ? static_cast<char>(cp | 0x20U) : '\0';
}

// Piece boundaries (code point indices) of the Qwen2 pre-tokenizer pattern.
std::vector<std::size_t> pre_split(const std::vector<std::uint32_t>& c) {
    const auto n = c.size();
    const auto L = [&](std::size_t k) { return k < n && is_letter(c[k]); };
    const auto N = [&](std::size_t k) { return k < n && is_number(c[k]); };
    const auto S = [&](std::size_t k) { return k < n && is_space(c[k]); };
    const auto crlf = [&](std::size_t k) { return k < n && (c[k] == '\r' || c[k] == '\n'); };
    const auto f = [&](std::size_t k) { return k < n ? fold(c[k]) : '\0'; };
    std::vector<std::size_t> ends;
    std::size_t i = 0;
    while (i < n) {
        std::size_t end = 0;
        if (c[i] == '\'') {
            const char a = f(i + 1U), b = f(i + 2U);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) end = i + 3U;
            else if (a == 's' || a == 't' || a == 'm' || a == 'd') end = i + 2U;
        }
        if (!end) {  // [^\r\n\p{L}\p{N}]?\p{L}+
            if (!crlf(i) && !L(i) && !N(i) && L(i + 1U)) { end = i + 1U; while (L(end)) ++end; }
            else if (L(i)) { end = i; while (L(end)) ++end; }
        }
        if (!end && N(i)) end = i + 1U;  // \p{N}
        if (!end) {  //  ?[^\s\p{L}\p{N}]+[\r\n]*
            const auto sym = [&](std::size_t k) { return k < n && !S(k) && !L(k) && !N(k); };
            std::size_t start = SIZE_MAX;
            if (c[i] == ' ' && sym(i + 1U)) start = i + 1U;
            else if (sym(i)) start = i;
            if (start != SIZE_MAX) {
                end = start;
                while (sym(end)) ++end;
                while (crlf(end)) ++end;
            }
        }
        if (!end && S(i)) {  // \s*[\r\n]+ | \s+(?!\S) | \s+
            std::size_t run = i;
            while (S(run)) ++run;
            std::size_t last_nl = SIZE_MAX;
            for (std::size_t k = i; k < run; ++k) if (crlf(k)) last_nl = k;
            if (last_nl != SIZE_MAX) end = last_nl + 1U;
            else if (run == n) end = run;
            else if (run - i >= 2U) end = run - 1U;
            else end = run;
        }
        if (!end) end = i + 1U;
        ends.push_back(end);
        i = end;
    }
    return ends;
}

std::array<std::string, 256> byte_alphabet() {
    std::array<std::string, 256> out{};
    std::uint32_t extra = 0;
    for (std::uint32_t b = 0; b < 256U; ++b) {
        const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1U && b <= 0xACU) || (b >= 0xAEU && b <= 0xFFU);
        std::string s;
        append_utf8(s, direct ? b : 256U + extra++);
        out[b] = s;
    }
    return out;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("te tokenizer: cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// vocab.json: one flat object {"token": id, ...}. Parsed here directly; the generic JSON value
// replaces duplicate keys by a linear scan, quadratic for 150K members.
std::unordered_map<std::string, std::int32_t> parse_vocab(const std::string& s) {
    std::unordered_map<std::string, std::int32_t> out;
    out.reserve(160000);
    std::size_t i = 0;
    const auto skip = [&] { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; };
    const auto fail = [&](const char* what) { throw std::runtime_error(std::string("te tokenizer: vocab.json: ") + what); };
    const auto hex4 = [&](std::size_t at) {
        if (at + 4U > s.size()) fail("short \\u escape");
        std::uint32_t v = 0;
        for (std::size_t k = at; k < at + 4U; ++k) {
            const char c = s[k];
            v <<= 4U;
            if (c >= '0' && c <= '9') v |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<std::uint32_t>(c - 'A' + 10);
            else fail("bad \\u escape");
        }
        return v;
    };
    const auto string = [&] {
        if (s[i] != '"') fail("expected string");
        ++i;
        std::string r;
        while (i < s.size() && s[i] != '"') {
            if (s[i] != '\\') { r.push_back(s[i++]); continue; }
            const char e = s[++i];
            ++i;
            switch (e) {
                case '"': r.push_back('"'); break;
                case '\\': r.push_back('\\'); break;
                case '/': r.push_back('/'); break;
                case 'b': r.push_back('\b'); break;
                case 'f': r.push_back('\f'); break;
                case 'n': r.push_back('\n'); break;
                case 'r': r.push_back('\r'); break;
                case 't': r.push_back('\t'); break;
                case 'u': {
                    std::uint32_t cp = hex4(i);
                    i += 4U;
                    if (cp >= 0xD800U && cp < 0xDC00U && i + 6U <= s.size() && s[i] == '\\' && s[i + 1U] == 'u') {
                        const auto lo = hex4(i + 2U);
                        if (lo >= 0xDC00U && lo < 0xE000U) { cp = 0x10000U + ((cp - 0xD800U) << 10U) + (lo - 0xDC00U); i += 6U; }
                    }
                    append_utf8(r, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        if (i >= s.size()) fail("unterminated string");
        ++i;
        return r;
    };
    skip();
    if (i >= s.size() || s[i] != '{') fail("expected object");
    ++i;
    for (;;) {
        skip();
        if (i < s.size() && s[i] == '}') break;
        auto key = string();
        skip();
        if (i >= s.size() || s[i] != ':') fail("expected ':'");
        ++i;
        skip();
        std::size_t end = i;
        while (end < s.size() && s[end] >= '0' && s[end] <= '9') ++end;
        if (end == i) fail("expected id");
        out[std::move(key)] = static_cast<std::int32_t>(std::stol(s.substr(i, end - i)));
        i = end;
        skip();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == '}') break;
        fail("expected ',' or '}'");
    }
    return out;
}

// The H3 extra special tokens, ids fixed by the released tokenizer.
constexpr std::pair<const char*, std::int32_t> kExtraTokens[] = {
    {"<d>", 151669}, {"</d>", 151670}, {"<|cutoff|>", 151671}, {"<|lyrics_start|>", 151672},
    {"<|lyrics_end|>", 151673}, {"<|caption_start|>", 151674}, {"<|caption_end|>", 151675},
};

}  // namespace

struct Tokenizer::Impl {
    std::unordered_map<std::string, std::int32_t> vocab;
    std::unordered_map<std::string, std::uint32_t> ranks;
    std::vector<std::pair<std::string, std::int32_t>> added;  // longest text first
    std::array<std::string, 256> alphabet = byte_alphabet();
    std::size_t size = 0;

    void bpe(std::string_view piece, std::vector<std::int32_t>& out) const {
        std::vector<std::string> sym;
        sym.reserve(piece.size());
        for (const char ch : piece) sym.push_back(alphabet[static_cast<unsigned char>(ch)]);
        std::string key;
        while (sym.size() > 1U) {
            std::uint32_t best = UINT32_MAX;
            std::size_t at = 0;
            for (std::size_t k = 0; k + 1U < sym.size(); ++k) {
                key.assign(sym[k]); key.push_back(' '); key += sym[k + 1U];
                if (const auto r = ranks.find(key); r != ranks.end() && r->second < best) { best = r->second; at = k; }
            }
            if (best == UINT32_MAX) break;
            // merge every occurrence of the best pair, left to right
            std::vector<std::string> merged;
            merged.reserve(sym.size());
            const std::string a = sym[at], b = sym[at + 1U];
            for (std::size_t k = 0; k < sym.size();) {
                if (k + 1U < sym.size() && sym[k] == a && sym[k + 1U] == b) { merged.push_back(a + b); k += 2U; }
                else merged.push_back(std::move(sym[k++]));
            }
            sym = std::move(merged);
        }
        for (const auto& s : sym) {
            const auto it = vocab.find(s);
            if (it == vocab.end()) throw std::runtime_error("te tokenizer: BPE symbol not in vocabulary");
            out.push_back(it->second);
        }
    }

    void encode_plain(std::string_view text, std::vector<std::int32_t>& out) const {
        const auto pts = nfc_points(decode_utf8(text));
        const auto ends = pre_split(pts);
        std::size_t begin = 0;
        std::string piece;
        for (const auto end : ends) {
            piece.clear();
            for (std::size_t k = begin; k < end; ++k) append_utf8(piece, pts[k]);
            bpe(piece, out);
            begin = end;
        }
    }
};

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;

std::shared_ptr<const Tokenizer> Tokenizer::load(const std::string& dir) {
    std::shared_ptr<Tokenizer> t(new Tokenizer());
    auto& im = *t->impl_;
    im.vocab = parse_vocab(read_file(dir + "/vocab.json"));
    std::int32_t max_id = -1;
    for (const auto& [k, id] : im.vocab) max_id = std::max(max_id, id);
    {
        std::istringstream merges(read_file(dir + "/merges.txt"));
        std::string line;
        std::uint32_t rank = 0;
        while (std::getline(merges, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line.rfind("#version", 0) == 0) continue;
            im.ranks.emplace(line, rank++);
        }
    }
    const auto cfg = serve::parse_json(read_file(dir + "/tokenizer_config.json"));
    if (const auto* dec = cfg.find("added_tokens_decoder"); dec != nullptr) {
        for (const auto& [k, v] : dec->as_object()) {
            const auto id = static_cast<std::int32_t>(std::stol(k));
            im.added.emplace_back(v.find("content")->as_string(), id);
            max_id = std::max(max_id, id);
        }
    }
    for (const auto& [text, id] : kExtraTokens) {
        im.added.emplace_back(text, id);
        max_id = std::max(max_id, id);
    }
    std::stable_sort(im.added.begin(), im.added.end(),
        [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    im.size = static_cast<std::size_t>(max_id) + 1U;
    return t;
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<std::int32_t> out;
    std::size_t start = 0, i = 0;
    while (i < text.size()) {
        bool hit = false;
        if (text[i] == '<') {
            for (const auto& [s, id] : impl_->added) {
                if (text.compare(i, s.size(), s) == 0) {
                    if (i > start) impl_->encode_plain(text.substr(start, i - start), out);
                    out.push_back(id);
                    i += s.size();
                    start = i;
                    hit = true;
                    break;
                }
            }
        }
        if (!hit) ++i;
    }
    if (start < text.size()) impl_->encode_plain(text.substr(start), out);
    return out;
}

namespace {

// Python str.isspace()
bool py_space(std::uint32_t c) { return space(c) || (c >= 0x1CU && c <= 0x1FU); }

// str.split() with no argument: fields separated by runs of whitespace.
std::vector<std::string> py_split(std::string_view s) {
    std::vector<std::string> out;
    const auto pts = decode_utf8(s);
    std::string cur;
    bool in = false;
    for (const auto cp : pts) {
        if (!(cp & kInvalid) && py_space(cp)) {
            if (in) { out.push_back(cur); cur.clear(); in = false; }
            continue;
        }
        append_utf8(cur, cp);
        in = true;
    }
    if (in) out.push_back(cur);
    return out;
}

std::string strip_char(std::string_view s, char c) {
    std::size_t b = 0, e = s.size();
    while (b < e && s[b] == c) ++b;
    while (e > b && s[e - 1U] == c) --e;
    return std::string(s.substr(b, e - b));
}

}  // namespace

Segment Tokenizer::encode_segment(std::string_view text, const EmbeddingLookup* lookup) const {
    // "\(" -> "(", "\)" -> ")"
    std::string s;
    s.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\\' && i + 1U < text.size() && (text[i + 1U] == '(' || text[i + 1U] == ')')) continue;
        s.push_back(text[i]);
    }
    // cut before every "embedding:" whose previous code point is whitespace
    static constexpr std::string_view kId = "embedding:";
    const auto ws_before = [&](std::size_t pos) {
        if (pos == 0) return false;
        std::size_t b = pos - 1U;
        while (b > 0 && (static_cast<unsigned char>(s[b]) & 0xC0U) == 0x80U) --b;
        const auto cp = decode_utf8(std::string_view(s).substr(b, pos - b));
        return cp.size() == 1U && !(cp[0] & kInvalid) && py_space(cp[0]);
    };
    std::vector<std::string> words;
    std::size_t start = 0;
    for (std::size_t pos = s.find(kId); pos != std::string::npos; pos = s.find(kId, pos + 1U)) {
        if (!ws_before(pos)) continue;
        words.push_back(s.substr(start, pos - start));
        start = pos;
    }
    words.push_back(s.substr(start));

    Segment out;
    const auto add = [&](std::string_view w) {
        const auto ids = encode(w);
        out.ids.insert(out.ids.end(), ids.begin(), ids.end());
    };
    for (auto& word : words) {
        if (word.empty()) continue;
        if (lookup == nullptr || word.rfind(kId, 0) != 0) { add(word); continue; }
        // name: the word after the identifier with '\n' stripped at both ends
        std::string name = strip_char(std::string_view(word).substr(kId.size()), '\n');
        auto fields = py_split(name);
        if (fields.empty()) throw std::invalid_argument("te tokenizer: empty embedding name");
        name = fields[0];
        std::string leftover;
        for (std::size_t k = 1; k < fields.size(); ++k) { if (k > 1U) leftover += ' '; leftover += fields[k]; }
        if (const auto cut = name.find_first_of("<["); cut != std::string::npos) {
            leftover = name.substr(cut) + (leftover.empty() ? "" : " " + leftover);
            name.resize(cut);
        }
        int rows = (*lookup)(name);
        std::string found = name;
        if (rows < 0) {
            const auto stripped = strip_char(name, ',');
            if (stripped.size() < name.size()) {
                rows = (*lookup)(stripped);
                found = stripped;
                leftover = name.substr(stripped.size()) + " " + leftover;
            }
        }
        if (rows >= 0) {
            out.ids.insert(out.ids.end(), static_cast<std::size_t>(rows), kEmbeddingRow);
            out.embeddings.push_back(found);
        }
        if (!leftover.empty()) add(leftover);
    }
    return out;
}

std::size_t Tokenizer::vocab_size() const noexcept { return impl_->size; }

std::string Tokenizer::nfc(std::string_view utf8) {
    std::string out;
    for (const auto cp : nfc_points(decode_utf8(utf8))) append_utf8(out, cp);
    return out;
}

std::vector<std::string> Tokenizer::pre_tokenize(std::string_view utf8) {
    const auto pts = decode_utf8(utf8);
    std::vector<std::string> out;
    std::size_t begin = 0;
    for (const auto end : pre_split(pts)) {
        std::string piece;
        for (std::size_t k = begin; k < end; ++k) append_utf8(piece, pts[k]);
        out.push_back(std::move(piece));
        begin = end;
    }
    return out;
}

std::string format_seconds(double t) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.1f", t);
    return buf;
}

Presented present(const Tokenizer& tok, std::string_view prompt, int images, const std::vector<RefItem>& items,
                  const EmbeddingLookup* lookup) {
    Presented p;
    const auto text = [&](std::string_view s) {
        if (s.empty()) return;
        auto seg = tok.encode_segment(s, lookup);
        p.ids.insert(p.ids.end(), seg.ids.begin(), seg.ids.end());
        p.embeddings.insert(p.embeddings.end(), seg.embeddings.begin(), seg.embeddings.end());
    };
    const auto vision = [&](int item, int frame0) {
        p.ids.push_back(kVisionStart);
        p.ids.push_back(kVisionSlot);
        p.ids.push_back(kVisionEnd);
        p.slot_item.push_back(item);
        p.slot_frame0.push_back(frame0);
    };
    if (!items.empty()) {
        int n_image = 0, n_audio = 0, n_video = 0;
        for (std::size_t i = 0; i < items.size(); ++i) {
            const auto& it = items[i];
            const int idx = static_cast<int>(i);
            if (it.kind == RefItem::Kind::image) {
                text("<Picture " + std::to_string(++n_image) + ">: ");
                vision(idx, 0);
            } else if (it.kind == RefItem::Kind::audio) {
                text("<Audio " + std::to_string(++n_audio) + ">: ");
            } else {
                auto ts = it.timestamps;
                if (ts.empty())
                    for (int f = 0; f < it.frames; ++f) ts.push_back(f / 2.0);
                if (ts.empty()) throw std::invalid_argument("present: video item without frames");
                if (ts.size() % 2U == 1U) ts.push_back(ts.back());
                text("<Video " + std::to_string(++n_video) + ">: ");
                for (std::size_t f = 0; f < ts.size(); f += 2U) {
                    text("<" + format_seconds((ts[f] + ts[f + 1U]) / 2.0) + " seconds>");
                    vision(idx, static_cast<int>(f));
                }
            }
        }
    } else {
        for (int i = 0; i < images; ++i) {
            text("<Picture " + std::to_string(i + 1) + ">: ");
            vision(i, 0);
        }
    }
    text(prompt);
    if (p.ids.empty()) p.ids.push_back(kPadToken);
    return p;
}

std::vector<std::int32_t> token_tags(const Presented& p, const std::vector<std::int32_t>& slot_sizes) {
    std::vector<std::int32_t> tags;
    std::size_t slot = 0;
    for (std::size_t i = 0; i < p.ids.size(); ++i) {
        if (p.ids[i] == kVisionSlot) {
            if (slot >= slot_sizes.size()) throw std::invalid_argument("token_tags: missing slot size");
            // the start token before the block is already emitted as text; re-tag it
            if (!tags.empty()) tags.back() = 0;
            tags.insert(tags.end(), static_cast<std::size_t>(slot_sizes[slot++]), 0);
            if (i + 1U < p.ids.size()) { tags.push_back(0); ++i; }  // the end token
            continue;
        }
        tags.push_back(1);
    }
    return tags;
}

}  // namespace strix::h3::te
