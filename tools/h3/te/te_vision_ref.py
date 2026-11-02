#!/usr/bin/env python3
"""H3 text encoder: independent plain-torch vision tower from the te pack vs the vision fixtures.

Checks the written spec of the tower (patch embed, bf16 position-embedding interpolation, 2D RoPE,
27 blocks, DeepStack mergers after blocks 8/16/24, 2x2 merger). Inputs are the fixture patches.

Usage: python te_vision_ref.py --pack <pack dir> --fixtures <vision fixture dir> [--pos-dtype bf16|fp32]
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def pos_embed(table, gh, gw, dtype):
    n = 48
    hs = torch.linspace(0, n - 1, gh)
    ws = torch.linspace(0, n - 1, gw)
    hf, wf = hs.int(), ws.int()
    hc, wc = (hf + 1).clip(max=n - 1), (wf + 1).clip(max=n - 1)
    dh, dw = hs - hf, ws - wf
    idx = [(hf[:, None] * n + wf[None]).flatten(), (hf[:, None] * n + wc[None]).flatten(),
           (hc[:, None] * n + wf[None]).flatten(), (hc[:, None] * n + wc[None]).flatten()]
    wt = [((1 - dh)[:, None] * (1 - dw)[None]).flatten(), ((1 - dh)[:, None] * dw[None]).flatten(),
          (dh[:, None] * (1 - dw)[None]).flatten(), (dh[:, None] * dw[None]).flatten()]
    t = table.to(dtype)
    p = [t[idx[j]] * wt[j].to(dtype)[:, None] for j in range(4)]
    pe = p[0] + p[1] + p[2] + p[3]
    pe = pe.view(gh // 2, 2, gw // 2, 2, -1).permute(0, 2, 1, 3, 4).reshape(gh * gw, -1)
    return pe.float()


def rope_tables(gh, gw):
    inv = 1.0 / (10000.0 ** (torch.arange(0, 36, 2, dtype=torch.float) / 36))
    rows, cols = [], []
    for bh in range(gh // 2):
        for bw in range(gw // 2):
            for mh in range(2):
                for mw in range(2):
                    rows.append(bh * 2 + mh)
                    cols.append(bw * 2 + mw)
    fr = torch.tensor(rows, dtype=torch.float)[:, None] * inv[None]
    fc = torch.tensor(cols, dtype=torch.float)[:, None] * inv[None]
    emb = torch.cat([fr, fc], -1)
    emb = torch.cat([emb, emb], -1)
    return emb.cos(), emb.sin()


def rope(x, cos, sin):  # x [N, H, 72]
    h = x.shape[-1] // 2
    return x * cos[:, None] + torch.cat([-x[..., h:], x[..., :h]], -1) * sin[:, None]


def lin(w, p, x):
    return F.linear(x, w[p + ".weight"].float().reshape(w[p + ".weight"].shape[0], -1), w[p + ".bias"].float())


def ln(w, p, x):
    return F.layer_norm(x, (x.shape[-1],), w[p + ".weight"].float(), w[p + ".bias"].float(), 1e-6)


def tower(w, patches, gh, gw, pos_dtype):
    x = lin(w, "v.patch_embed.proj", patches)
    x = x + pos_embed(w["v.pos_embed.weight"], gh, gw, pos_dtype)
    cos, sin = rope_tables(gh, gw)
    deep = []
    n = x.shape[0]
    for b in range(27):
        p = "v.blocks.%d." % b
        h = ln(w, p + "norm1", x)
        qkv = lin(w, p + "attn.qkv", h).reshape(n, 3, 16, 72)
        q, k, v = rope(qkv[:, 0], cos, sin), rope(qkv[:, 1], cos, sin), qkv[:, 2]
        a = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        x = x + lin(w, p + "attn.proj", a.transpose(0, 1).reshape(n, -1))
        h = ln(w, p + "norm2", x)
        x = x + lin(w, p + "mlp.linear_fc2", F.gelu(lin(w, p + "mlp.linear_fc1", h), approximate="tanh"))
        if b in (8, 16, 24):
            m = "v.deepstack_merger_list.%d" % (8, 16, 24).index(b)
            y = ln(w, m + ".norm", x.view(-1, 4608))
            deep.append(lin(w, m + ".linear_fc2", F.gelu(lin(w, m + ".linear_fc1", y))))
    y = ln(w, "v.merger.norm", x).view(-1, 4608)
    merged = lin(w, "v.merger.linear_fc2", F.gelu(lin(w, "v.merger.linear_fc1", y)))
    return merged, deep


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--pos-dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    torch.set_num_threads(16)
    w = load_file(os.path.join(args.pack, "te-vision.safetensors"))
    man = json.load(open(os.path.join(args.fixtures, "manifest.json"), encoding="utf-8"))
    pd = torch.bfloat16 if args.pos_dtype == "bf16" else torch.float32
    for c in man["cases"]:
        for b, v in enumerate(c["vision"]):
            _, gh, gw = v["grid_thw"][0]
            base = os.path.join(args.fixtures, "%s.vis%d" % (c["name"], b))
            patches = torch.from_numpy(np.fromfile(base + ".patches.f32", "<f4").reshape(gh * gw, -1))
            ref = torch.from_numpy(np.fromfile(base + ".merged.f32", "<f4").reshape(-1, 5120))
            with torch.no_grad():
                merged, deep = tower(w, patches, gh, gw, pd)
            r = [rel(deep[d], torch.from_numpy(np.fromfile(base + ".deep%d.f32" % d, "<f4").reshape(-1, 5120)))
                 for d in range(3)]
            print("%s vis%d grid %dx%d pos %s: merged rel %.3e deep %s bitwise %s" % (
                c["name"], b, gh, gw, args.pos_dtype, rel(merged, ref), ["%.2e" % x for x in r], torch.equal(merged, ref)))


if __name__ == "__main__":
    main()
