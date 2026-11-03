#!/usr/bin/env python3
"""Generate the GLM-4/5 tokenizer fixtures checked by tests/glm_tokenizer_tests.cpp.

Reference: HuggingFace `tokenizers` (Tokenizer.from_file(tokenizer.json)) encode/decode. Characters new in Unicode 16
(Garay, Tulu-Tigalari, ...) are left out: the reference regex engine classifies them with older tables.
usage: make_tokenizer_fixtures.py --model-dir MODEL_DIR [--random 4000] [--seed 1234] OUT.json
Output (JSON, UTF-8): encode: [{text, ids, ids_plain, decoded, decoded_skip}], decode: [{ids, text, text_skip}]
"""
import argparse
import json
import os
import random
import sys

from tokenizers import Tokenizer

FIXED_TEXTS = [
    "", " ", "  ", "\n", "\n\n", "\r\n", " \n", "\t", "a", "Hello world", "Hello, world!", " Hello", "  Hello  ",
    "The quick brown fox jumps over the lazy dog.",
    "I'm here, you're there, they've gone, we'll see, he'd know, it's fine, don't.",
    "I'M HERE, YOU'RE THERE, THEY'VE GONE, WE'LL SEE, HE'D KNOW, IT'S FINE, DON'T.",
    "I'm You'Re They'vE We'Ll He'D It'S Don'T", "o'clock rock'n'roll l'été 'tis 'S 'sup", "'s's's 's 't 're 've 'm 'll 'd 'x",
    "Don'ſ stop", "camelCaseIdentifier snake_case_identifier SCREAMING_CASE PascalCase iPhone eBay McDonald's",
    "ALLCAPS WORD", "HTTPServerError XMLHttpRequest", "ÀÉÎÕÜ àéîõü ÑñÇç",
    # Spanish, NFC and NFD forms
    "¿Dónde está la estación? ¡Mañana será otro día! El niño comió piñas y cigüeñas.",
    "¿Do\u0301nde esta\u0301 la estacio\u0301n? ¡Man\u0303ana sera\u0301 otro di\u0301a! El nin\u0303o comio\u0301 pin\u0303as y cigu\u0308en\u0303as.",
    "Canción, corazón, acción, pingüino, ÁRBOL, Ñandú",
    "Cancio\u0301n, corazo\u0301n, accio\u0301n, pingu\u0308ino, A\u0301RBOL, N\u0303andu\u0301",
    "e\u0301\u0301 a\u0328\u0301 o\u031b\u0323 \u1e9b\u0323 A\u030a \u212b \u2126 \u0390",
    # CJK, Hangul, other scripts
    "日本語のテキストです。東京は大きい都市です。", "中文文本测试，包括标点符号！还有数字123和English混合。",
    "한국어 텍스트입니다. 서울은 큰 도시입니다.", "\u1100\u1161\u11a8 \u1112\u1161\u11ab\u1100\u1173\u11af",
    "Русский текст: Привет, мир! ПРИВЕТ", "Ελληνικά: Καλημέρα κόσμε", "עברית: שלום עולם", "العربية: مرحبا بالعالم",
    "हिन्दी: नमस्ते दुनिया", "ไทย: สวัสดีชาวโลก", "Tiếng Việt: Xin chào thế giới", "ქართული ენა", "Ὀδυσσεύς",
    # emoji
    "Emoji: 😀😃😄 👍🏽 👨‍👩‍👧‍👦 🇪🇸🇯🇵 ❤️ ✨🚀", "🙂🙂🙂", "a😀b", " 😀", "x\ufe0f",
    # code / JSON
    "def f(x):\n    return x ** 2  # square\n\nprint(f(3))\n",
    "for (int i = 0; i < n; ++i) {\n\tsum += a[i];\n}\n",
    '{"name": "GLM", "version": 3.1, "tags": ["a", "b"], "nested": {"k": null, "t": true}}',
    "<html><body><div class=\"x\">Hi</div></body></html>", "SELECT * FROM users WHERE id = 42;",
    "path/to/file.txt ./a/b ../c //comment /* block */ http://example.com/a/b?c=d&e=f#g",
    "a!/b ?// .\n/ ;\r\n/x !!//\n\n// )/\r", "...\n\n\n---\n***\n", "&&|| ->=> ::<>",
    # numbers
    "1 12 123 1234 12345 123456 1234567", "3.14159 2,718 1e10 -42 +7 0x1F 1_000_000", "١٢٣٤٥ ١٢٣ ۴۵۶ ४५६ ໑໒໓", "²³¹ ½ ⅓ Ⅻ ⑳",
    "year2024 v1.2.3 abc123def 12ab34",
    # whitespace runs
    "a  b   c    d", "a \n b \n\n c", "end with spaces   ", "   start with spaces", "tabs\t\t\there", "mixed \t \n \r\n\t x",
    "\n\n\n\n", "    \n    \n", "a\u00a0b\u3000c\u2028d\u2029e\u202ff", "x\x1cy\x1dz\x1e\x1f", "\u2000\u2001\u2002 x", "\u000b\u000c x",
    # special tokens inline
    "[gMASK]<sop><|user|>\nHello<|assistant|>\n<think></think>\nHi", "text<think>reasoning</think>answer", "a/nothink b", "x/nothinkY",
    "<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>", "<|endoftext|><|observation|>", "[MASK][sMASK]<eop>",
    "<think", "</think>>", "<<think>>", "<|begin_of_box|>a<|end_of_box|>", "<|image|><|video|>", "<|code_prefix|>a<|code_suffix|>b<|code_middle|>",
    "[gMASK]<sop>[gMASK]", "<|system|>\nsys<|user|>\nq", "<tool_response>r</tool_response>",
    # misc unicode
    "\u07fd\u0334 \u08d3\u0345", "ſtraße STRASSE ß ẞ", "ﬁ ﬂ ﬃ", "Ǆ ǅ ǆ ǈ",
    "\u0300start mark", " \u0301", "\u200b\u200c\u200d zero width", "\ue000 private", "\U0001d400\U0001d41a math",
]


POOLS = [
    'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', '0123456789', ' ' * 6 + '\n\t\r',
    "'''sStTrReEvVmMlLdD", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', '//\n\n', 'áéíóúñüÁÉÍÓÚÑÜçàèìòù',
    '\u0300\u0301\u0302\u0303\u0308\u0323\u0327\u0328\u031b\u0334\u0345\u05b0\u093c\u3099\u302a',
    '日本語中文字漢字テキストカタカナひらがな', '한국어텍스트', '\u1100\u1161\u11a8\u1112\u1173\u11af', 'привет ПРИВЕТ',
    'αβγ ΑΒΓ', 'مرحبا', 'नमस्ते', 'สวัสดี', '😀👍🏽👨‍👩‍👧🇪🇸❤️\ufe0f\u200d', '١٢٣४५६²½Ⅻ', '\u00a0\u3000\u2028\u2029\u202f\u2000\u0085\x1c\x1f',
    'ſǅǈǋᾈᾙ', '\u07fd\u08d3\u0d3b\u0d3c\u0eba\u0c3c',
    '\ue000\U000f0000\ufffd\u0378\U0001d400', '\u0b47\u0b3e\u0dd9\u0dcf\u1025\u102e\u0915\u0f73\u0f75\u0f81',
]
SPECIAL_SNIPPETS = ['[gMASK]', '<sop>', '<|user|>', '<|assistant|>', '<think>', '</think>', '/nothink', '<tool_call>', '<arg_key>',
                    '<|endoftext|>', '<|observation|>', '<|image|>', '[MASK]']


def random_texts(rng, n):
    out = []
    for _ in range(n):
        pools = rng.sample(POOLS, rng.randint(1, 4))
        length = rng.randint(1, 60)
        s = []
        for _ in range(length):
            if rng.random() < 0.03:
                s.append(rng.choice(SPECIAL_SNIPPETS))
            else:
                p = rng.choice(pools)
                s.append(rng.choice(p))
        out.append(''.join(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--random', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.model_dir, 'tokenizer.json'))
    rng = random.Random(a.seed)
    texts = list(FIXED_TEXTS)
    texts += [t + ' ' + t.upper() for t in FIXED_TEXTS[:40] if t]
    texts += [str(n) for n in [0, 7, 42, 999, 1000, 12345, 123456, 1234567, 99999999]]
    texts += [''.join(rng.choice('0123456789') for _ in range(k)) for k in range(1, 9) for _ in range(3)]
    texts += random_texts(rng, a.random)
    encode = []
    for t in texts:
        enc = tok.encode(t, add_special_tokens=False).ids
        tok.encode_special_tokens = True
        plain = tok.encode(t, add_special_tokens=False).ids
        tok.encode_special_tokens = False
        encode.append({'text': t, 'ids': enc, 'ids_plain': plain, 'decoded': tok.decode(enc, skip_special_tokens=False),
                       'decoded_skip': tok.decode(enc, skip_special_tokens=True)})
    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    decode = []
    for _ in range(600):
        ids = []
        for _ in range(rng.randint(1, 12)):
            r = rng.random()
            if r < 0.7:
                ids.append(rng.randrange(0, vocab_size))
            elif r < 0.95:
                ids.append(rng.randrange(0, 256 + 1000))  # byte tokens: broken UTF-8
            else:
                ids.append(rng.randrange(154820, vocab_size))  # added tokens
        decode.append({'ids': ids, 'text': tok.decode(ids, skip_special_tokens=False), 'text_skip': tok.decode(ids, skip_special_tokens=True)})
    out = {'tokenizers_version': __import__('tokenizers').__version__, 'vocab_size': vocab_size,
           'eos': tok.token_to_id('<|endoftext|>'), 'encode': encode, 'decode': decode}
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')
    print('%d encode, %d decode cases -> %s' % (len(encode), len(decode), a.out))


if __name__ == '__main__':
    sys.exit(main())
