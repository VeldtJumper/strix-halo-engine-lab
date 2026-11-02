#!/usr/bin/env python3
"""H3 text encoder: repack the nvfp4 (+ activation pre-scale) Qwen3-VL-32B/50L checkpoint.

Input : the 50-layer conditioning checkpoint (safetensors, nvfp4 linears, int8 embedding).
Output: <out>/te-embed.safetensors, <out>/te-blk.NN.safetensors (NN = 00..49),
        <out>/te-vision.safetensors, <out>/te-pack.json (config + per-file sha256).

The pack keeps the 4-bit codes exactly (no requantisation). The only transform is the block
scale layout: the checkpoint stores the e4m3 block scales in the 128x4 tiled (swizzled) layout
used by tensor-core block-scaled GEMMs; the pack stores them row-major [N, K/16].

Effective weight used by the reference matmul (activations are fp32 there):
    W[n, k] = fp32( e2m1(code[n, k]) * fp32( tscale * e4m3(bscale[n, k / 16]) ) )
Exact factorisation for an f16/bf16 GEMM:  e2m1 * e4m3 is exact in f16 (<= 6 significant bits,
range 2^-10 .. 2688), so  W = tscale * Wq  with  Wq = e2m1 * e4m3  (f16, exact) and tscale applied
in fp32 in the epilogue (differs from the reference product order by <= 1 fp32 ulp).

Linears with an activation pre-scale (o_proj, down_proj) compute  y = (x * prescale) @ W^T,
x * prescale rounded to the activation dtype first (fp32 in the reference).

Usage: python te_pack.py --src <ckpt.safetensors> --out <pack dir> [--verify all|sample|none]
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

N_LAYERS = 50
LINEARS = [("self_attn.q_proj", "q"), ("self_attn.k_proj", "k"), ("self_attn.v_proj", "v"),
           ("self_attn.o_proj", "o"), ("mlp.gate_proj", "gate"), ("mlp.up_proj", "up"),
           ("mlp.down_proj", "down")]
NORMS = [("input_layernorm.weight", "attn_norm"), ("post_attention_layernorm.weight", "ffn_norm"),
         ("self_attn.q_norm.weight", "q_norm"), ("self_attn.k_norm.weight", "k_norm")]

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)

CONFIG = {
    "format": "h3-te-pack-1",
    "arch": "qwen3vl-32b-text, truncated",
    "n_layers": 50, "hidden": 5120, "ffn": 25600, "n_heads": 64, "n_kv_heads": 8, "head_dim": 128,
    "vocab": 151936, "rms_eps": 1e-6, "rope_theta": 5000000.0, "rope": "split-half (rotate_half)",
    "mrope_sections": [24, 20, 20], "mrope_interleaved": True,
    "final_norm": False, "output": "hidden state after layer 50 (no final norm)",
    "nibble_order": "element 2j in the high nibble, 2j+1 in the low nibble",
    "e2m1_lut": E2M1.tolist(),
    "linear_tensors": {
        "<name>.codes": "uint8 [N, K/2] packed e2m1 codes",
        "<name>.bscale": "uint8 [N, K/16] e4m3fn bits, row-major (unswizzled)",
        "<name>.tscale": "float32 [1] per-tensor scale",
        "<name>.prescale": "bf16 [K] activation pre-scale (o, down only)",
    },
    "embedding": "tok_embd.q int8 [vocab, hidden], tok_embd.scale float32 [vocab]; "
                 "row = bf16(fp32(q) * fp32(scale)) as fp32 (the reference lookup returns bf16)",
}


def unswizzle(blocked, rows, cols):
    """Inverse of the 128x4 tile layout: tile (rb, cb) of 128 rows x 4 cols is stored as 512
    consecutive entries ordered [row % 32][row / 32 % 4][col % 4]."""
    rb = (rows + 127) // 128
    cb = (cols + 3) // 4
    t = blocked.reshape(rb, cb, 32, 4, 4)          # [rb, cb, r32, r4, c4]
    t = t.permute(0, 3, 2, 1, 4)                   # [rb, r4, r32, cb, c4]
    t = t.reshape(rb * 128, cb * 4)
    return t[:rows, :cols].contiguous()


def unpack_codes(codes):
    hi = codes >> 4
    lo = codes & 0x0F
    return torch.stack([hi, lo], dim=-1).reshape(codes.shape[0], -1)


def dequant_fp32(codes, bscale_rowmajor_u8, tscale):
    n, k2 = codes.shape
    k = k2 * 2
    vals = E2M1[unpack_codes(codes).long()].reshape(n, k // 16, 16)
    s = tscale.float().reshape(()) * bscale_rowmajor_u8.view(torch.float8_e4m3fn).float()
    return (vals * s.unsqueeze(-1)).reshape(n, k)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# Tensor suffixes of a quantised linear / embedding in the source checkpoint. Each quantised module
# also carries one extra uint8 tensor holding its quantisation metadata as JSON; it is located by
# elimination (the single other key directly under the module).
DATA_SUFFIXES = {"weight", "weight_scale", "weight_scale_2", "pre_quant_scale"}


def quant_meta_key(keys, base):
    cands = [k for k in keys if k.startswith(base + ".") and "." not in k[len(base) + 1:]
             and k[len(base) + 1:] not in DATA_SUFFIXES]
    assert len(cands) == 1, ("expected one quantisation metadata tensor", base, cands)
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", default="sample", choices=["all", "sample", "none"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    f = safe_open(args.src, "pt")
    keys = set(f.keys())
    files = {}
    report = {"verified": [], "pqs_tensors": 0}
    t0 = time.time()

    # embedding
    out = {"tok_embd.q": f.get_tensor("model.embed_tokens.weight").contiguous(),
           "tok_embd.scale": f.get_tensor("model.embed_tokens.weight_scale").float().reshape(-1).contiguous()}
    p = os.path.join(args.out, "te-embed.safetensors")
    save_file(out, p, metadata={"format": CONFIG["format"]})
    files["te-embed.safetensors"] = p
    del out

    for L in range(N_LAYERS):
        pre = "model.layers.%d." % L
        out = {}
        for src, dst in NORMS:
            out["blk.%s" % dst] = f.get_tensor(pre + src).contiguous()
        for src, dst in LINEARS:
            base = pre + src
            q = f.get_tensor(base + ".weight").contiguous()
            bs = f.get_tensor(base + ".weight_scale").view(torch.uint8)
            ts = f.get_tensor(base + ".weight_scale_2").float().reshape(1)
            meta = json.loads(bytes(f.get_tensor(quant_meta_key(keys, base)).tolist()).decode())
            assert meta.get("format") == "nvfp4", (base, meta)
            n, k = q.shape[0], q.shape[1] * 2
            assert n % 128 == 0 and (k // 16) % 4 == 0, (base, n, k)
            bsr = unswizzle(bs, n, k // 16)
            out["blk.%s.codes" % dst] = q
            out["blk.%s.bscale" % dst] = bsr
            out["blk.%s.tscale" % dst] = ts
            if base + ".pre_quant_scale" in keys:
                out["blk.%s.prescale" % dst] = f.get_tensor(base + ".pre_quant_scale").contiguous()
                report["pqs_tensors"] += 1
            check = args.verify == "all" or (args.verify == "sample" and (L in (0, 25, 49)))
            if check:
                # every 16-block holds a code of magnitude 6 (block scale = amax / 6): a wrong
                # scale layout would not change this, so also check the scale/prescale coupling
                codes = unpack_codes(q)
                assert bool(((codes & 7).reshape(n, -1, 16).amax(-1) == 7).all()), "block max code " + base
                if base + ".pre_quant_scale" in keys:
                    # the activation pre-scale s divides the weight columns before quantisation, so
                    # the column means of log2(block scale) anti-correlate with log2(s) only when
                    # the scale layout is read right (measured over all 100: -0.67..-1.00; misread: -0.34..+0.23)
                    lb = torch.log2(bsr.view(torch.float8_e4m3fn).float()).mean(0)
                    ls = torch.log2(out["blk.%s.prescale" % dst].float()).reshape(-1, 16).mean(1)
                    corr = torch.corrcoef(torch.stack([lb, ls]))[0, 1].item()
                    assert corr < -0.5, "scale layout check failed %s corr %.3f" % (base, corr)
                # f16 factorisation is exact
                wq = dequant_fp32(q, bsr, torch.ones(1))
                assert torch.equal(wq.half().float(), wq), "Wq not exact in f16 " + base
                report["verified"].append(base)
        p = os.path.join(args.out, "te-blk.%02d.safetensors" % L)
        # one metadata key only: the writer orders several keys nondeterministically
        save_file(out, p, metadata={"format": CONFIG["format"]})
        files["te-blk.%02d.safetensors" % L] = p
        print("layer %d done %.0f s" % (L, time.time() - t0), flush=True)
        del out

    vis = {}
    for k in sorted(keys):
        if k.startswith("visual."):
            vis["v." + k[len("visual."):]] = f.get_tensor(k).contiguous()
    p = os.path.join(args.out, "te-vision.safetensors")
    save_file(vis, p, metadata={"format": CONFIG["format"]})
    files["te-vision.safetensors"] = p
    del vis

    # every source tensor must be accounted for
    used = {"model.embed_tokens.weight", "model.embed_tokens.weight_scale",
            quant_meta_key(keys, "model.embed_tokens")}
    for L in range(N_LAYERS):
        pre = "model.layers.%d." % L
        used |= {pre + s for s, _ in NORMS}
        for s, _ in LINEARS:
            used |= {pre + s + "." + suf for suf in DATA_SUFFIXES}
            used.add(quant_meta_key(keys, pre + s))
    left = sorted(k for k in keys if k not in used and not k.startswith("visual."))
    assert not left, left

    man = dict(CONFIG)
    man["files"] = {name: {"bytes": os.path.getsize(p), "sha256": sha256_file(p)} for name, p in files.items()}
    man["total_bytes"] = sum(v["bytes"] for v in man["files"].values())
    man["verify"] = {"mode": args.verify, "tensors": len(report["verified"])}
    man["prescale_tensors"] = report["pqs_tensors"]
    with open(os.path.join(args.out, "te-pack.json"), "w") as fo:
        json.dump(man, fo, indent=1)
    print("pack done: %.2f GB in %.0f s" % (man["total_bytes"] / 1e9, time.time() - t0))


if __name__ == "__main__":
    main()
