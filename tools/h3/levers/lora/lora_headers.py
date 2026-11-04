"""Summarise LoRA safetensors files from their headers only (no tensor data read): metadata, key prefixes, module
groups (suffix-stripped names with block indices collapsed to N), ranks, alpha tensors, dtypes.
usage: python lora_headers.py <file.safetensors> ...
"""
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".lora_down.weight", ".lora_up.weight", ".lora_mid.weight", ".alpha", ".diff", ".diff_b",
            ".dora_scale", ".weight", ".bias")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def split(key):
    for s in SUFFIXES:
        if key.endswith(s):
            return key[: -len(s)], s
    return key, ""


def main():
    for path in sys.argv[1:]:
        h = header(path)
        meta = h.pop("__metadata__", {})
        size = os.path.getsize(path)
        print(f"== {os.path.basename(path)}: {size / 1e9:.3f} GB, {len(h)} tensors")
        for k, v in meta.items():
            vs = str(v)
            print(f"   meta {k}: {vs[:300]}{'...' if len(vs) > 300 else ''}")
        dtypes = Counter(t["dtype"] for t in h.values())
        print(f"   dtypes {dict(dtypes)}")
        prefixes = Counter(k.split(".")[0] for k in h)
        print(f"   first key segments {dict(prefixes)}")
        groups = defaultdict(lambda: {"n": 0, "suffix": Counter(), "ranks": Counter(), "shapes": Counter()})
        blocks = set()
        for k, t in h.items():
            base, suf = split(k)
            m = re.search(r"blocks\.(\d+)\.", base)
            if m and not base.startswith("token_refiner"):
                blocks.add(int(m.group(1)))
            g = re.sub(r"\.\d+\.", ".N.", base)
            gr = groups[g]
            gr["n"] += 1
            gr["suffix"][suf] += 1
            sh = t["shape"]
            gr["shapes"][str(sh)] += 1
            if suf in (".lora_A.weight", ".lora_down.weight") and len(sh) == 2:
                gr["ranks"][sh[0]] += 1
        print(f"   DiT block indices with LoRA: {len(blocks)} ({min(blocks) if blocks else '-'}..{max(blocks) if blocks else '-'})")
        for g in sorted(groups):
            gr = groups[g]
            print(f"   {g}: {gr['n']} tensors, suffixes {dict(gr['suffix'])}, ranks {dict(gr['ranks'])}, shapes {dict(list(gr['shapes'].items())[:4])}")
        alphas = [k for k in h if k.endswith(".alpha")]
        print(f"   alpha tensors: {len(alphas)}")


if __name__ == "__main__":
    main()
