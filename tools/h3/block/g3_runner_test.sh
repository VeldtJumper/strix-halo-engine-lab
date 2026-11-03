#!/bin/bash
# CTest G3 of the clip runner (GPU, one node): the small dense clip through the single-node driver, the two-node
# schedule's world 1 and its world 2 (both ranks as threads, memory transport) must give identical hashes.txt (every
# step's velocity + final latents / frames / wave); the same for Turbo (+ FBC observe) and Turbo + sparse when the Turbo
# assets are present.
#   usage: g3_runner_test.sh <strix-h3-clip binary> <g3_compare.sh>
#   env: H3_MODELS (or the per-asset variables) and H3_TOKENIZER_DIR; absent -> SKIPPED (exit 77).
set -u
BIN=$1; CMP=$2
[ -n "${H3_TOKENIZER_DIR:-}" ] && { [ -n "${H3_MODELS:-}" ] || [ -n "${H3_CKPT:-}" ]; } || { echo "SKIPPED: H3_MODELS / H3_TOKENIZER_DIR not set"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
S="--width 256 --height 256 --length 17 --steps 8 --seed 42"
run() { local name=$1; shift; "$BIN" --out "$W/$name" "$@" > "$W/$name.log" 2>&1 || { echo "clip $name failed:"; tail -5 "$W/$name.log"; exit 1; }; }
run single $S --dit single
run dist1 $S --dit dist
run w2 $S --world 2 --transport emulate
G3_STEPS=8 bash "$CMP" "$W" single dist1 w2 || exit 1
if [ -n "${H3_MODELS:-}" ] && [ -f "$H3_MODELS/merged_fc2_s070.safetensors" ]; then
  T="$S --turbo --fbc-observe"
  run t_single $T --dit single
  run t_w2 $T --world 2 --transport emulate
  G3_STEPS=8 bash "$CMP" "$W" t_single t_w2 || exit 1
  # Turbo + sparse attention (h3-attn-sparse-thr-v1): the 1-node DeviceForward adapter, the two-node schedule's world 1
  # and world 2 (both on the sparse split, cut 128 floor(S / 256)) must agree bitwise; G3_SPARSE=0 skips this part
  if [ "${G3_SPARSE:-1}" = 1 ]; then
    TS="$T --sparse 0.75 --sparse-dense-layers 39-42 --sparse-dense-calls 3"
    run ts_single $TS --dit single
    run ts_dist1 $TS --dit dist
    run ts_w2 $TS --world 2 --transport emulate
    grep -h "sparse density" "$W/ts_w2.log" | sed -n '1p;$p' | cut -c1-240
    G3_STEPS=8 bash "$CMP" "$W" ts_single ts_dist1 ts_w2 || exit 1
  fi
else
  echo "Turbo part SKIPPED (no merged fc2 under H3_MODELS)"
fi
echo "PASSED (0 failures)"
