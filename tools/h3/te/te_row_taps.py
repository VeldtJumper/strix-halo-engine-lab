#!/usr/bin/env python3
"""H3 text encoder: per-layer drift of one residual row, GPU vs the plain fp32 forward and the
single-site FP16 forwards (te_ref.py --tap-row / strix-h3-te-gpu-tests --row-tap outputs).

Usage: python te_row_taps.py --fp32 <r.fp32.f32> --gpu <r.gpu.f32> [--site NAME=<file> ...]
Each file is [n_layers][5120] fp32 (the row after every layer). Prints |x| and the relative
distance of the GPU row and of every site row to the fp32 row per layer.

With --sub the files are sub-layer taps (te_ref.py --tap-sub, TE_TAP_SUB=1), per layer
[attn norm 5120 | o_proj input 8192 | x after attention 5120 | ffn norm 5120 | down input 25600 |
x out 5120]. For layers --from..--to it prints, per run, the relative error of every tap and the
error each branch adds to the residual: |(d_run - d_ref)| / |x_in_ref| with d = branch output
(x_after - x_before), next to the error the residual already carried in (|x_in_run - x_in_ref| /
|x_in_ref|). Growth of a step = error out / error in.
"""
import argparse

import numpy as np

H = 5120
SUB = [("an", H), ("oin", 8192), ("xa", H), ("fn", H), ("din", 25600), ("xo", H)]
SUBN = sum(n for _, n in SUB)


def load(p):
    return np.fromfile(p, "<f4").reshape(-1, H).astype(np.float64)


def load_sub(p):
    a = np.fromfile(p, "<f4").astype(np.float64).reshape(-1, SUBN)
    out, o = {}, 0
    for name, n in SUB:
        out[name] = a[:, o:o + n]
        o += n
    return out


def nrm(v):
    return float(np.linalg.norm(v))


def plain(args):
    ref = load(args.fp32)
    gpu = load(args.gpu)
    sites = [(s.split("=", 1)[0], load(s.split("=", 1)[1])) for s in args.site]
    rel = lambda a: np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
    rg = rel(gpu)
    rs = [(n, rel(a)) for n, a in sites]
    print("layer |x| gpu " + " ".join(n for n, _ in rs) + " gpu/min(site)")
    for L in range(ref.shape[0]):
        m = min([r[L] for _, r in rs]) if rs else float("nan")
        print("%2d %9.1f %.2e %s %.1f" % (L, np.linalg.norm(ref[L]), rg[L], " ".join("%.2e" % r[L] for _, r in rs),
                                          rg[L] / m if rs else float("nan")))


def sub(args):
    ref = load_sub(args.fp32)
    runs = [("gpu", load_sub(args.gpu))] + [(s.split("=", 1)[0], load_sub(s.split("=", 1)[1])) for s in args.site]
    names = [n for n, _ in runs]
    rel = lambda r, k, L: nrm(r[k][L] - ref[k][L]) / nrm(ref[k][L])
    print("relative error per tap (in = residual entering the layer)")
    print("L   tap  " + " ".join("%9s" % n for n in names))
    for L in range(max(args.from_layer, 1), min(args.to_layer, ref["xo"].shape[0] - 1) + 1):
        print("%2d |x_in| %.1f |attn d| %.1f |ffn d| %.1f" % (
            L, nrm(ref["xo"][L - 1]), nrm(ref["xa"][L] - ref["xo"][L - 1]), nrm(ref["xo"][L] - ref["xa"][L])))
        xin = nrm(ref["xo"][L - 1])
        rows = [("in", lambda r: nrm(r["xo"][L - 1] - ref["xo"][L - 1]) / xin)]
        rows += [(k, (lambda k: lambda r: rel(r, k, L))(k)) for k in ("an", "oin")]
        rows.append(("+attn", lambda r: nrm((r["xa"][L] - r["xo"][L - 1]) - (ref["xa"][L] - ref["xo"][L - 1])) / xin))
        rows.append(("xa", lambda r: rel(r, "xa", L)))
        rows += [(k, (lambda k: lambda r: rel(r, k, L))(k)) for k in ("fn", "din")]
        rows.append(("+ffn", lambda r: nrm((r["xo"][L] - r["xa"][L]) - (ref["xo"][L] - ref["xa"][L])) / xin))
        rows.append(("out", lambda r: rel(r, "xo", L)))
        for tag, f in rows:
            print("    %-5s " % tag + " ".join("%9.2e" % f(r) for _, r in runs))
        g = [rel(r, "xo", L) / max(nrm(r["xo"][L - 1] - ref["xo"][L - 1]) / xin, 1e-30) for _, r in runs]
        print("    grow  " + " ".join("%9.2f" % v for v in g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--site", action="append", default=[], help="NAME=file")
    ap.add_argument("--sub", action="store_true", help="sub-layer tap files")
    ap.add_argument("--from", dest="from_layer", type=int, default=35)
    ap.add_argument("--to", dest="to_layer", type=int, default=43)
    args = ap.parse_args()
    sub(args) if args.sub else plain(args)


if __name__ == "__main__":
    main()
