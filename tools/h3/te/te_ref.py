#!/usr/bin/env python3
"""H3 text encoder: independent plain-torch forward from the te pack (text-only prompts).

Re-derives the layer-50 hidden states from te-pack.json + te-*.safetensors, without the reference
modules, and compares against the fixtures (ids + hidden). This checks the written spec
(unswizzle, nibble order, prescale placement, q/k norm, split-half RoPE, GQA, causal mask).

Usage: python te_ref.py --pack <pack dir> --fixtures <fixture dir> [--names p0_short]
       [--act fp32|bf16|f16]   activation dtype between ops (GEMM accumulation stays fp32)
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def deq(codes, bscale, tscale):
    n = codes.shape[0]
    v = torch.stack([codes >> 4, codes & 15], -1).reshape(n, -1)
    v = E2M1[v.long()].reshape(n, -1, 16)
    s = tscale.float().reshape(()) * bscale.view(torch.float8_e4m3fn).float()
    return (v * s.unsqueeze(-1)).reshape(n, -1)


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def rope_tables(S, D, theta):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
    f = torch.arange(S).float()[:, None] * inv[None, :]
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def rope(x, cos, sin):  # x [H, S, D], split-half
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos + rot * sin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--act", default="fp32", choices=["fp32", "bf16", "f16"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--sites", default="gin,att", help="rounding sites for --act: gin (GEMM inputs), att (q/k/v)")
    ap.add_argument("--dump-rows", default=None, help="write per-row relative error vs the fixture to <dir>/<name>.<tag>.rowrel.f32")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tap-row", type=int, default=None, help="write row R of the residual after every layer")
    ap.add_argument("--tap-out", default=None, help="file for --tap-row ([n_layers][5120] fp32, first fixture only)")
    ap.add_argument("--tap-sub", action="store_true",
                    help="per layer [attn norm 5120 | o_proj input 8192 | x after attention 5120 | ffn norm 5120 | "
                         "down_proj input 25600 | x out 5120]")
    ap.add_argument("--replay-dir", default=None,
                    help="layers --replay-from..--replay-to (first fixture): write x_in / x_attn / x_ffn [S][5120] fp32 "
                         "and the single-site (gin, qk, v) local drifts of each sublayer on the same fp32 input")
    ap.add_argument("--replay-from", type=int, default=35)
    ap.add_argument("--replay-to", type=int, default=43)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    actdt = {"fp32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[args.act]
    rnd = (lambda t: t.to(actdt).float()) if actdt != torch.float32 else (lambda t: t)
    sites = set(args.sites.split(","))
    ident = lambda t: t
    rg = rnd if "gin" in sites else ident
    rqk = rnd if ("att" in sites or "qk" in sites) else ident
    rv = rnd if ("att" in sites or "v" in sites) else ident
    rp = rnd if "p" in sites else None
    rnd16 = lambda t: t.to(torch.float16).float()
    rowrel = lambda o, r: ((o - r).norm(dim=-1) / r.norm(dim=-1)).numpy().astype("<f4")
    if args.replay_dir and args.act != "fp32":
        raise SystemExit("--replay-dir needs --act fp32")

    cfg = json.load(open(os.path.join(args.pack, "te-pack.json")))
    man = json.load(open(os.path.join(args.fixtures, "manifest.json"), encoding="utf-8"))
    H, NH, NKV, D, eps = cfg["hidden"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"], cfg["rms_eps"]
    emb = load_file(os.path.join(args.pack, "te-embed.safetensors"))

    fx = [f for f in man["fixtures"] if args.names is None or f["name"] in args.names.split(",")]
    xs, refs = [], []
    for f in fx:
        ids = torch.from_numpy(np.fromfile(os.path.join(args.fixtures, f["name"] + ".ids.i32"), "<i4").astype(np.int64))
        # reference lookup: fp32(int8) * fp32(row scale), rounded to bf16 (compute dtype), then fp32
        x = (emb["tok_embd.q"][ids].float() * emb["tok_embd.scale"][ids].float()[:, None]).bfloat16().float()
        xs.append(x)
        refs.append(np.fromfile(os.path.join(args.fixtures, f["name"] + ".hidden.f32"), "<f4").reshape(-1, H))
    t0 = time.time()
    taps = []
    for L in range(cfg["n_layers"]):
        w = load_file(os.path.join(args.pack, "te-blk.%02d.safetensors" % L))
        W = {n: deq(w["blk.%s.codes" % n], w["blk.%s.bscale" % n], w["blk.%s.tscale" % n])
             for n in ("q", "k", "v", "o", "gate", "up", "down")}
        def attn_block(x, cos, sin, rg, rqk, rv, tap=None):
            S = x.shape[0]
            # HIP rounding sites: GEMM inputs (gin), attention q/k/v (att); P rounding is not modelled
            h = rg(rms(x, w["blk.attn_norm"], eps))
            if tap is not None:
                tap.append(h[args.tap_row].clone())
            q = (h @ W["q"].T).view(S, NH, D).transpose(0, 1)
            k = (h @ W["k"].T).view(S, NKV, D).transpose(0, 1)
            v = rv((h @ W["v"].T).view(S, NKV, D).transpose(0, 1))
            q = rqk(rope(rms(q, w["blk.q_norm"], eps), cos, sin))
            k = rqk(rope(rms(k, w["blk.k_norm"], eps), cos, sin))
            k = k.repeat_interleave(NH // NKV, 0)
            v = v.repeat_interleave(NH // NKV, 0)
            if rp is None:
                a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:  # unnormalised P rounded, row sum of the rounded P (as the HIP kernel)
                sc = (q @ k.transpose(1, 2)) / math.sqrt(D)
                sc = sc.masked_fill(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"))
                pm = rp(torch.exp(sc - sc.amax(-1, keepdim=True)))
                a = (pm @ v) / pm.sum(-1, keepdim=True)
            a = a.transpose(0, 1).reshape(S, NH * D)
            a = rg(a * w["blk.o.prescale"].float())
            if tap is not None:
                tap.append(a[args.tap_row].clone())
            return x + (a @ W["o"].T)

        def ffn_block(x, rg, tap=None):
            h = rg(rms(x, w["blk.ffn_norm"], eps))
            if tap is not None:
                tap.append(h[args.tap_row].clone())
            g = torch.nn.functional.silu(h @ W["gate"].T) * (h @ W["up"].T)
            g = rg(g * w["blk.down.prescale"].float())
            if tap is not None:
                tap.append(g[args.tap_row].clone())
            return x + (g @ W["down"].T)

        for i, x in enumerate(xs):
            S = x.shape[0]
            cos, sin = rope_tables(S, D, cfg["rope_theta"])
            sub = taps if (args.tap_row is not None and args.tap_sub and i == 0) else None
            replay = i == 0 and args.replay_dir and args.replay_from <= L <= args.replay_to
            if replay:  # shared-input sublayer replay: full fp32 inputs/outputs + the single-site local drifts
                x.numpy().astype("<f4").tofile(os.path.join(args.replay_dir, "x_in.%02d.f32" % L))
            xa = attn_block(x, cos, sin, rg, rqk, rv, sub)
            if sub is not None:
                taps.append(xa[args.tap_row].clone())
            if replay:
                xa.numpy().astype("<f4").tofile(os.path.join(args.replay_dir, "x_attn.%02d.f32" % L))
                for site, (sg, sq, sv) in (("gin", (rnd16, ident, ident)), ("qk", (ident, rnd16, ident)),
                                           ("v", (ident, ident, rnd16))):
                    o = attn_block(x, cos, sin, sg, sq, sv)
                    rowrel(o, xa).tofile(os.path.join(args.replay_dir, "attn.%02d.%s.rowrel.f32" % (L, site)))
            x = ffn_block(xa, rg, sub)
            if replay:
                x.numpy().astype("<f4").tofile(os.path.join(args.replay_dir, "x_ffn.%02d.f32" % L))
                rowrel(ffn_block(xa, rnd16), x).tofile(os.path.join(args.replay_dir, "ffn.%02d.gin.rowrel.f32" % L))
            xs[i] = x
            if args.tap_row is not None and i == 0:
                taps.append(x[args.tap_row].clone())
        if L % 10 == 9:
            print("layer %d %.0f s" % (L + 1, time.time() - t0), flush=True)
    if args.tap_row is not None and args.tap_out:
        torch.cat([t.reshape(-1) for t in taps]).numpy().astype("<f4").tofile(args.tap_out)
    res = {}
    for f, x, r in zip(fx, xs, refs):
        r = torch.from_numpy(r)
        d = (x - r)
        rel = d.norm() / r.norm()
        # exclude the attention-sink row (token 0 carries the massive activation)
        rel_tail = d[1:].norm() / r[1:].norm() if x.shape[0] > 1 else rel
        cos_min = torch.nn.functional.cosine_similarity(x, r, dim=-1).min()
        row_rel = (d.norm(dim=-1) / r.norm(dim=-1))[1:]
        worst = torch.topk(row_rel, min(4, row_rel.numel())) if row_rel.numel() else None
        if args.dump_rows:
            rr = (d.norm(dim=-1) / r.norm(dim=-1)).numpy().astype("<f4")
            tag = args.tag or ("%s-%s" % (args.act, args.sites.replace(",", "+")))
            rr.tofile(os.path.join(args.dump_rows, "%s.%s.rowrel.f32" % (f["name"], tag)))
        res[f["name"]] = {"rel_l2": rel.item(), "rel_l2_rows1+": rel_tail.item(), "max_abs": d.abs().max().item(),
                          "min_row_cos": cos_min.item(), "bitwise": bool(torch.equal(x, r)),
                          "worst_rows": [] if worst is None else [(int(i) + 1, float(v)) for v, i in zip(worst.values, worst.indices)]}
        print(f["name"], res[f["name"]])
    print(json.dumps({"act": args.act, "results": res}))


if __name__ == "__main__":
    main()
