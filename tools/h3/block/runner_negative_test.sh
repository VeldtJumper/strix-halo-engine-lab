#!/bin/bash
# Negative / failure-path assertions of the clip runner (READY review R1 #1-#7, R2 #1 / #8, R3 #2 / #3 / #5 / #6 / #8):
# every case must end with the expected exit code AND message. GPU (one node) for the cases that reach the model; the
# parse / identity / budget cases stop before any model load. Failure injection needs the test-hook build
# (strix-h3-clip-testhooks, H3_CLIP_TEST_HOOKS); the production binary must ignore the hook variables.
#   usage: runner_negative_test.sh <strix-h3-clip> <g3_compare.sh> [<strix-h3-clip-testhooks>]
#   env: H3_MODELS (or the per-asset variables) and H3_TOKENIZER_DIR; absent -> SKIPPED (exit 77).
set -u
BIN=$1; CMP=$2; HBIN=${3:-$(dirname "$BIN")/strix-h3-clip-testhooks}
[ -n "${H3_TOKENIZER_DIR:-}" ] && [ -n "${H3_MODELS:-}" ] || { echo "SKIPPED: H3_MODELS / H3_TOKENIZER_DIR not set"; exit 77; }
[ -x "$HBIN" ] || { echo "FAILED: test-hook binary $HBIN missing"; exit 1; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
pass=0; fail=0
okc() { echo "ok   $1"; pass=$((pass + 1)); }
bad() { echo "FAIL $1"; fail=$((fail + 1)); }
final_field() { grep '^final ' "$1/hashes.txt" | tail -1 | awk -v k="$2" '{for (i = 1; i < NF; ++i) if ($i == k) print $(i + 1)}'; }
expect() {  # expect NAME RC PATTERN -- command...
  local name=$1 want=$2 pat=$3; shift 4
  "$@" > "$W/$name.log" 2>&1
  local rc=$?
  if [ "$rc" = "$want" ] && grep -q -- "$pat" "$W/$name.log"; then echo "ok   $name (rc $rc: /$pat/)"; pass=$((pass + 1))
  else echo "FAIL $name (rc $rc, want $want and /$pat/)"; tail -3 "$W/$name.log" | cut -c1-200; fail=$((fail + 1)); fi
}
S="--width 256 --height 256 --length 17 --steps 8 --seed 42"
T="$S --turbo --fbc-observe"
# CLI / asset validation before any model load (R1 #6, #14, R2 #11)
expect cli_shift 1 "finite and > 0" -- "$BIN" --out "$W/c1" $S --shift 0
expect cli_steps 1 "1 .. 1000" -- "$BIN" --out "$W/c2" --width 256 --height 256 --length 17 --steps 1001
expect cli_prompt_file 1 "cannot read the prompt file" -- "$BIN" --out "$W/c3" $S --prompt-file "$W/none.txt"
expect cli_pdd 1 "PDD head banks are not wired" -- "$BIN" --out "$W/c4" $S --pdd
expect cli_strength 1 "strength" -- "$BIN" --out "$W/c5" $S --turbo --strength 0
expect cli_asset 1 "not found" -- env H3_CKPT="$W/none.safetensors" "$BIN" --out "$W/c6" $S
# identity / foreign artifacts (R1 #1, #3)
mkdir -p "$W/id"; echo "prompt=another run" > "$W/id/run_identity.txt"
expect identity_mismatch 1 "different run" -- "$BIN" --out "$W/id" $S
mkdir -p "$W/art"; : > "$W/art/te_hidden_f32.bin"
expect artifact_without_identity 1 "without a run identity" -- "$BIN" --out "$W/art" $S
# budget admission before any setup (R1 #4)
expect budget_refusal 3 "nothing done" -- "$BIN" --out "$W/bud" $S --budget 2
# Turbo / Euler split run (R2 #1): stop after 3 steps (budget-stop path), then checks on copies of that state (the
# split runs use the hook build throughout: the run identity pins the binary)
expect euler_stop 3 "stopped after step 3/" -- env H3_CLIP_TEST_STOP_AFTER=3 "$HBIN" --out "$W/split" $T --dump-velocity
cp -a "$W/split" "$W/mal"; head -c 30 "$W/mal/dit_state.bin" > "$W/mal/x" && mv "$W/mal/x" "$W/mal/dit_state.bin"
expect malformed_checkpoint 1 "malformed" -- "$HBIN" --out "$W/mal" $T --dump-velocity
# dump completeness on resume (R3 #6): every step once in order, every file present with its size
cp -a "$W/split" "$W/cov"; sed -i '$d' "$W/cov/velocity/index.txt"
expect dump_coverage 1 "covers 2 of the 3" -- "$HBIN" --out "$W/cov" $T --dump-velocity
cp -a "$W/split" "$W/dup"; sed -i '3s/^step 3 /step 2 /' "$W/dup/velocity/index.txt"
expect dump_duplicate_step 1 "is not step 3" -- "$HBIN" --out "$W/dup" $T --dump-velocity
cp -a "$W/split" "$W/mis"; rm -f "$W/mis/velocity/step_02_audio_carried_v_bf16.bin"
expect dump_missing_file 1 "missing or of the wrong size" -- "$HBIN" --out "$W/mis" $T --dump-velocity
cp -a "$W/split" "$W/trn"; truncate -s -2 "$W/trn/velocity/step_01_video_neg_v_bf16.bin"
expect dump_short_file 1 "missing or of the wrong size" -- "$HBIN" --out "$W/trn" $T --dump-velocity
cp -a "$W/split" "$W/split0"
expect euler_resume 0 "done in" -- "$HBIN" --out "$W/split" $T --dump-velocity
# the production binary ignores the hook variables (compiled out): it must run through to the end
expect euler_full 0 "done in" -- env H3_CLIP_TEST_STOP_AFTER=3 H3_CLIP_TEST_FAIL=write-outputs H3_CLIP_TEST_RANK1_SALT=x "$BIN" --out "$W/full" $T --dump-velocity
if G3_STEPS=8 bash "$CMP" "$W" full split; then echo "ok   euler split (3 + 5, via dit_state.bin) == uninterrupted, bitwise"; pass=$((pass + 1))
else echo "FAIL euler split != uninterrupted"; fail=$((fail + 1)); fi
n=$(grep -c '^step ' "$W/split/velocity/index.txt")
if [ "$n" = 8 ]; then echo "ok   dump index covers 8 / 8 steps after the resume"; pass=$((pass + 1)); else echo "FAIL dump index $n / 8"; fail=$((fail + 1)); fi
# final-latent reuse also checks the dump (R3 #6): a finished run with a lost dump file is refused before any decode
cp -a "$W/full" "$W/reuse"; rm -f "$W/reuse/velocity/step_08_video_neg_v_bf16.bin" "$W/reuse/frames_rgb24.raw" "$W/reuse/hashes.txt"
expect dump_final_reuse 1 "missing or of the wrong size" -- "$BIN" --out "$W/reuse" $T --dump-velocity
# decode-only (R3 #2 / #8): admission against the time left (both routes refuse), the DiT is never opened, and the
# fast route never opens the 36L VAE; the 36L decode-only frames equal the clip's own decode bitwise
expect decode_admission_36l 3 "decode stage .* nothing done" -- "$BIN" --decode-only "$W/full" --out "$W/da1" $S --budget 2
expect decode_admission_fast 3 "decode stage .*h3-vae-fast24-v1.* nothing done" -- "$BIN" --decode-only "$W/full" --out "$W/da2" $S --fast-vae --budget 2
expect decode_no_dit 0 "decode-only. 36L" -- env H3_CKPT="$W/none.safetensors" H3_TURBO_LORA="$W/none.safetensors" "$BIN" --decode-only "$W/full" --out "$W/do36" $S
expect decode_fast_no_36l 0 "decode-only. h3-vae-fast24-v1" -- env H3_CKPT="$W/none.safetensors" H3_VAE_WEIGHTS="$W/none.safetensors" "$BIN" --decode-only "$W/full" --out "$W/dofast" $S --fast-vae
f_clip=$(final_field "$W/full" frames); f_do=$(final_field "$W/do36" frames)
if [ -n "$f_clip" ] && [ "$f_clip" = "$f_do" ]; then okc "decode-only 36L frames == the clip's decode ($f_do)"; else bad "decode-only frames $f_do != clip $f_clip"; fi
n=$(grep -c '^final ' "$W/do36/hashes.txt" 2>/dev/null); [ "$n" = 1 ] && okc "decode-only hashes.txt: one final line" || bad "decode-only final lines: $n"
# the recorded stage time covers the whole stage (>= the video decode incl. load printed by the run)
vt=$(sed -n 's/.*(\([0-9.]*\) s incl. load).*/\1/p' "$W/decode_no_dit.log" | head -1); st=$(awk '{print $3}' "$W/do36/timing.txt" 2>/dev/null)
if [ -n "$vt" ] && [ -n "$st" ] && awk -v s="$st" -v v="$vt" 'BEGIN { exit !(s >= v && s > 0) }'; then okc "decode-only timing: whole stage $st s >= video $vt s"
else bad "decode-only timing: stage '$st' vs video '$vt'"; fi
# the ORIGINAL budget fits but what is LEFT after the pre-admission work does not -> refused (hook delays that work)
est=$(sed -n 's/.*decode stage (~\([0-9]*\) s, 36L).*/\1/p' "$W/decode_admission_36l.log" | head -1)
if [ -n "$est" ]; then
  expect decode_admission_left 3 "decode stage .* nothing done" -- env H3_CLIP_TEST_DECODE_DELAY_S=12 "$HBIN" --decode-only "$W/full" --out "$W/da3" $S --budget $((est + 8))
  expect decode_admission_left_ctl 0 "decode-only. 36L" -- "$HBIN" --decode-only "$W/full" --out "$W/da4" $S --budget $((est + 8))
else bad "decode admission: no estimate in decode_admission_36l.log"; fi
# refused accelerated resume (R1 #2)
expect sparse_stop 3 "stopped after step 1/" -- env H3_CLIP_TEST_STOP_AFTER=1 "$HBIN" --out "$W/sp" $T --sparse 0.75
expect sparse_resume_refused 1 "cannot resume" -- "$HBIN" --out "$W/sp" $T --sparse 0.75
# VAE failure path (R1 #5): injected failure in rank 0's two-node VAE stage -> exit 1, no hang / abort
expect vae_failure 1 "injected failure" -- timeout 300 env H3_CLIP_TEST_FAIL=vae-rank0 "$HBIN" --out "$W/vf" $S --world 2 --transport emulate
# output-write failure (R3 #3): the final hash line cannot be written -> exit 1 and rank 1 never receives END
expect write_failure 1 "injected output-write failure" -- timeout 300 env H3_CLIP_TEST_FAIL=write-outputs "$HBIN" --out "$W/wf" $S --world 2 --transport emulate
if grep -q "END after" "$W/write_failure.log"; then bad "write failure: rank 1 got END"; else okc "write failure: no END sent to rank 1"; fi
if grep -q '^final ' "$W/wf/hashes.txt" 2>/dev/null; then bad "write failure: a final hash line was written"; else okc "write failure: no final hash line"; fi
# a REAL failing write (no hook): hashes.txt of a resumable run points at /dev/full -> the first step line fails at
# flush (ENOSPC) -> exit 1 with the write error (catches a removed stream check)
if [ -w /dev/full ]; then
  cp -a "$W/split0" "$W/dfull"; rm -f "$W/dfull/hashes.txt"; ln -s /dev/full "$W/dfull/hashes.txt"
  expect write_enospc 1 "write .*hashes.txt" -- "$HBIN" --out "$W/dfull" $T --dump-velocity
else bad "write_enospc: /dev/full not writable"; fi
# HELLO mismatch (R3 #5): two emulated ranks whose asset / flag digests differ -> both refuse before any model work
expect hello_mismatch 1 "HELLO digest mismatch" -- timeout 120 env H3_CLIP_TEST_RANK1_SALT=other-asset "$HBIN" --out "$W/hm" $S --world 2 --transport emulate
if grep -q "^.step 1/" "$W/hello_mismatch.log"; then bad "hello mismatch: sampling started"; else okc "hello mismatch: no sampling step ran"; fi
# malformed norm assets (R3 #5): the loaders' shape checks, host unit test of the ref-tests binary
RT=$(dirname "$BIN")/strix-h3-block-ref-tests
if [ -x "$RT" ]; then expect malformed_norm 0 "malformed norm assets rejected" -- "$RT" unit
else bad "malformed norm: $RT missing"; fi
echo "negative / failure-path cases: $pass passed, $fail failed"
[ $fail = 0 ] && echo "PASSED (0 failures)" && exit 0
echo "FAILED ($fail failures)"; exit 1
