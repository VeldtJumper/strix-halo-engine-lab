"""Side-by-side of two clip runs' preview frames + PSNR / SSIM per frame.

usage: python compare_frames.py REF_DIR TEST_DIR OUT_DIR [--frames 0,41,82,123] [--labels dense,turbo]
REF_DIR / TEST_DIR hold frame_NNNN.png (strix-h3-clip previews). Writes OUT_DIR/side_by_side.png (one row per frame:
REF | TEST, each downscaled to <= 672 px wide) and OUT_DIR/metrics.txt (PSNR on 8-bit RGB, SSIM on RGB with
channel_axis, data_range 255, full resolution).
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("out")
    ap.add_argument("--frames", default="0,41,82,123")
    ap.add_argument("--labels", default="dense,turbo")
    a = ap.parse_args()
    frames = [int(x) for x in a.frames.split(",")]
    lab = a.labels.split(",")
    os.makedirs(a.out, exist_ok=True)
    rows, lines = [], []
    for f in frames:
        name = "frame_%04d.png" % f
        r = np.asarray(Image.open(os.path.join(a.ref, name)).convert("RGB"))
        t = np.asarray(Image.open(os.path.join(a.test, name)).convert("RGB"))
        if r.shape != t.shape:
            raise SystemExit("shape mismatch at " + name)
        psnr = peak_signal_noise_ratio(r, t, data_range=255)
        ssim = structural_similarity(r, t, channel_axis=2, data_range=255)
        lines.append("frame %4d  PSNR %6.2f dB  SSIM %.4f" % (f, psnr, ssim))
        h, w = r.shape[:2]
        sw = min(w, 672)
        sh = h * sw // w
        pair = Image.new("RGB", (2 * sw + 8, sh + 22), (255, 255, 255))
        for k, im in enumerate((r, t)):
            pair.paste(Image.fromarray(im).resize((sw, sh), Image.LANCZOS), (k * (sw + 8), 22))
        d = ImageDraw.Draw(pair)
        d.text((4, 4), "%s  frame %d" % (lab[0], f), fill=(0, 0, 0))
        d.text((sw + 12, 4), "%s  frame %d  (PSNR %.2f dB, SSIM %.4f vs %s)" % (lab[1], f, psnr, ssim, lab[0]), fill=(0, 0, 0))
        rows.append(pair)
    W = max(p.width for p in rows)
    sheet = Image.new("RGB", (W, sum(p.height for p in rows)), (255, 255, 255))
    y = 0
    for p in rows:
        sheet.paste(p, (0, y))
        y += p.height
    sheet.save(os.path.join(a.out, "side_by_side.png"))
    with open(os.path.join(a.out, "metrics.txt"), "w", encoding="utf-8") as fh:
        fh.write("ref %s\ntest %s\n" % (a.ref, a.test))
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
