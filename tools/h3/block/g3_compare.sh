#!/bin/bash
# G3 for the clip runner: every step's velocity hash and the final latent / frames / wave hashes of clip REF must equal
# those of each other clip (hashes.txt written by strix-h3-clip in each output dir), plus optional raw frame files.
# Fails closed: every record must be complete and well formed - step lines "step N sigma <8 hex> vhash <16 hex> <16 hex>"
# numbered 1..N without gaps (N = $G3_STEPS when set), then exactly one final line
# "final latent video <16 hex> audio <16 hex> frames <16 hex> wave <16 hex>" as the last line; any missing / malformed /
# incomplete record, any difference, or nothing to compare exits 1.
#   usage: [G3_STEPS=N] g3_compare.sh CLIPS_DIR REF OTHER... [-- FRAMES_A FRAMES_B]
set -u
[ $# -ge 3 ] || { echo "usage: g3_compare.sh CLIPS_DIR REF OTHER... [-- FRAMES_A FRAMES_B]"; exit 2; }
D=$1; REF=$2; shift 2
fail=0
valid() {  # valid FILE -> 0 when the record is complete and well formed
  [ -s "$1" ] || return 1
  awk -v want="${G3_STEPS:-}" '
    BEGIN { n = 0; fin = 0; bad = 0 }
    /^step [0-9]+ sigma [0-9a-f]{8} vhash [0-9a-f]{16} [0-9a-f]{16}$/ { if (fin || $2 != n + 1) bad = 1; n = $2; next }
    /^final latent video [0-9a-f]{16} audio [0-9a-f]{16} frames [0-9a-f]{16} wave [0-9a-f]{16}$/ { if (fin || n == 0) bad = 1; fin = 1; next }
    { bad = 1 }
    END { if (bad || !fin || (want != "" && n != want)) exit 1 }' "$1"
}
if ! valid "$D/$REF/hashes.txt"; then echo "G3 FAIL $REF: hashes.txt missing, malformed or incomplete"; fail=1; fi
n_others=0
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  o=$1; shift; n_others=$((n_others + 1))
  if ! valid "$D/$o/hashes.txt"; then echo "G3 FAIL $o: hashes.txt missing, malformed or incomplete"; fail=1; continue; fi
  n=$(grep -c . "$D/$o/hashes.txt")
  if cmp -s "$D/$REF/hashes.txt" "$D/$o/hashes.txt"; then echo "G3 ok   $o == $REF ($n lines: every step's velocity + final latents / frames / wave, bitwise)"
  else echo "G3 FAIL $o != $REF"; diff "$D/$REF/hashes.txt" "$D/$o/hashes.txt" | head -6; fail=1; fi
done
[ $n_others -ge 1 ] || { echo "G3 FAIL: no clip to compare"; fail=1; }
if [ "${1:-}" = "--" ]; then
  [ $# -eq 3 ] || { echo "G3 FAIL: -- needs two frame files"; exit 1; }
  if [ -s "$2" ] && [ -s "$3" ] && cmp -s "$2" "$3"; then echo "G3 ok   frames $2 == $3"; else echo "G3 FAIL frames $2 != $3 (or missing)"; fail=1; fi
fi
if [ $fail = 0 ]; then echo "G3 compare PASSED"; exit 0; fi
echo "G3 compare FAILED"; exit 1
