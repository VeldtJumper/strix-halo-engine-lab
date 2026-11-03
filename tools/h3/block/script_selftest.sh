#!/bin/bash
# Host regression tests of the clip wrapper / compare scripts (no GPU, no assets): clip_turn.sh's PRECHECK gate (exit status AND
# the pass line), its POST status, and g3_compare.sh's fail-closed record checks.
#   usage: script_selftest.sh [<tools/h3/block dir>]
set -u
T=${1:-$(cd "$(dirname "$0")" && pwd)}
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
pass=0; fail=0
ok() { echo "ok   $1"; pass=$((pass + 1)); }
bad() { echo "FAIL $1"; fail=$((fail + 1)); }
# turn NAME WANT PRECHECK [POST]: clip_turn.sh with no build and no clips; the turn status must be WANT (exit + .done)
turn() {
  local name=$1 want=$2 pre=$3 post=${4:-}
  mkdir -p "$W/$name"
  H3_BLOCK_DIR="$W/$name" BUILD=none CLIPS="" PRECHECK="$pre" POST="$post" bash "$T/clip_turn.sh" 1 > "$W/$name.log" 2>&1
  local rc=$?
  local st; st=$(cat "$W/$name/turn-1.done" 2>/dev/null)
  if [ "$rc" = "$want" ] && [ "$st" = "status $want" ]; then ok "clip_turn $name (status $want)"
  else bad "clip_turn $name: rc $rc, .done '$st', want $want"; tail -3 "$W/$name.log"; fi
}
turn precheck_pass 0 'echo "PASSED (0 failures)"'
turn precheck_rc_fails 1 'echo "PASSED (0 failures)"; false'
turn precheck_no_pass_line 1 'true'
turn precheck_fail_line 1 'echo "FAILED (1 failures)"'
turn post_fails 1 'echo "PASSED (0 failures)"' 'false'
turn post_ok 0 'echo "PASSED (0 failures)"' 'true'
# g3_compare: well-formed records compare equal; every malformed / incomplete / differing record fails
rec() {  # rec DIR STEPS [final line override] -> DIR/hashes.txt
  mkdir -p "$W/g/$1"
  : > "$W/g/$1/hashes.txt"
  for i in $(seq 1 "$2"); do printf 'step %d sigma 3f800000 vhash %016x %016x\n' "$i" "$i" "$((i * 7))" >> "$W/g/$1/hashes.txt"; done
  echo "${3:-final latent video 0123456789abcdef audio 0123456789abcdef frames 0123456789abcdef wave 0123456789abcdef}" >> "$W/g/$1/hashes.txt"
}
g3() {  # g3 NAME WANT ARGS...
  local name=$1 want=$2; shift 2
  G3_STEPS=8 bash "$T/g3_compare.sh" "$W/g" "$@" > "$W/g3_$name.log" 2>&1
  local rc=$?
  if [ "$rc" = "$want" ]; then ok "g3_compare $name (rc $rc)"; else bad "g3_compare $name: rc $rc, want $want"; tail -3 "$W/g3_$name.log"; fi
}
rec ref 8; rec same 8
rec short 7
rec diff 8; sed -i '4s/vhash [0-9a-f]*/vhash ffffffffffffffff/' "$W/g/diff/hashes.txt"
rec nofinal 8; sed -i '$d' "$W/g/nofinal/hashes.txt"
rec gap 8; sed -i '3d' "$W/g/gap/hashes.txt"
rec dupfinal 8; tail -1 "$W/g/dupfinal/hashes.txt" >> "$W/g/dupfinal/hashes.txt"
rec badfinal 8 "final latent video 0123 audio x frames y wave z"
rec trailing 8; echo "junk" >> "$W/g/trailing/hashes.txt"
mkdir -p "$W/g/empty"; : > "$W/g/empty/hashes.txt"
g3 equal 0 ref same
g3 differs 1 ref diff
g3 fewer_steps 1 ref short
g3 no_final 1 ref nofinal
g3 step_gap 1 ref gap
g3 two_finals 1 ref dupfinal
g3 malformed_final 1 ref badfinal
g3 trailing_line 1 ref trailing
g3 empty_record 1 ref empty
g3 missing_dir 1 ref none
g3 bad_ref 1 diff ref
g3 nothing_to_compare 2 ref
echo "script self-tests: $pass passed, $fail failed"
[ $fail = 0 ] && echo "PASSED (0 failures)" && exit 0
echo "FAILED ($fail failures)"; exit 1
