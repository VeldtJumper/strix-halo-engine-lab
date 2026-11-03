#!/bin/bash
# One-node clip wrapper with a wall budget: build strix-h3-clip from the repo copy (niced, -j4), then
# run the clips given in CLIPS ("name|args" entries separated by ';'; a clip that stops on its budget, rc 3, continues
# in the next call with the same entry). Build-only when CLIPS is empty.
#   usage: clip_turn.sh <run number>
#   env (required): H3_BLOCK_DIR = the working directory (repo/ copy, clips/, logs)
#   env: BUILD=pre (default) | post (clips first with $BIN, then build with the time left) | none
#        BIN=binary for the clips (default ./repo/build/strix-h3-clip)
#        PRECHECK=command that must exit 0 AND print "PASSED (0 failures)" before any clip runs
#        EXTRA_TARGETS=more build targets (e.g. strix-h3-attn-gpu-tests for an exact-commit GPU rerun)
#        POST=command run after the clips (e.g. a hash comparison); its exit status counts
#        TURN_S=wall budget (default 465 s)
# Exit status (also written to turn-N.done): 0 = all clips complete and POST ok; 3 = a clip stopped on its budget (and
# nothing failed); 1 = build / precheck / clip / POST failure.
set -u
cd "${H3_BLOCK_DIR:?H3_BLOCK_DIR required}" || exit 1
N=${1:-0}
t0=$(date +%s)
BUILD=${BUILD:-pre}
BIN=${BIN:-./repo/build/strix-h3-clip}
TURN=${TURN_S:-465}
status=0
finish() { echo "total $(( $(date +%s) - t0 )) s, status $status"; echo "status $status" > turn-$N.done; exit $status; }
build() {  # build [seconds]
  if [ ! -f repo/build/build.ninja ]; then
    nice -n 19 cmake -S repo -B repo/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRIX_FAMILY_H3=ON -DSTRIX_ENABLE_HIP_RUNTIME=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DSTRIX_HIP_ARCHITECTURES=gfx1151 -DBUILD_TESTING=OFF > clip-configure.log 2>&1 \
      || { echo "configure failed"; tail -20 clip-configure.log; return 1; }
  fi
  local tb=$(date +%s)
  timeout ${1:-3600} nice -n 19 cmake --build repo/build -j4 --target strix-h3-clip ${EXTRA_TARGETS:-} > clip-build.log 2>&1
  local rc=$?
  echo "build rc $rc $(( $(date +%s) - tb )) s"
  [ $rc -ne 0 ] && grep -E "error|Error" clip-build.log | head -30
  return $rc
}
if [ "$BUILD" = pre ]; then
  build || { status=1; finish; }
fi
if [ -n "${PRECHECK:-}" ]; then  # anything but a clean pass stops the clips
  echo "=== precheck: $PRECHECK"
  eval "$PRECHECK" > precheck-$N.log 2>&1
  pcr=$?
  grep -E "FAIL|PASSED|SKIPPED|build rc" precheck-$N.log | tail -5
  # both: exit status 0 AND the pass line (a pass string in a failing command's log is not a pass)
  { [ $pcr -eq 0 ] && grep -q "PASSED (0 failures)" precheck-$N.log; } || { echo "precheck failed (rc $pcr)"; status=1; finish; }
fi
IFS=';' read -ra items <<< "${CLIPS:-}"
for it in "${items[@]}"; do
  name=${it%%|*}; args=${it#*|}
  left=$(( TURN - ($(date +%s) - t0) ))
  [ $left -lt 30 ] && { echo "no time left for $name"; [ $status = 0 ] && status=3; break; }
  echo "=== clip $name: $args (budget $left s) $(date +%T)"
  eval timeout $left $BIN --out clips/$name --budget $(( left - 15 )) $args
  rc=$?; echo "rc $rc"
  if [ $rc -eq 3 ]; then echo "clip $name continues next turn"; [ $status = 0 ] && status=3; break; fi
  [ $rc -ne 0 ] && status=1
done
if [ -n "${POST:-}" ]; then
  echo "=== post: $POST"
  eval "$POST"; prc=$?
  echo "post rc $prc"
  [ $prc -ne 0 ] && status=1
fi
if [ "$BUILD" = post ]; then
  left=$(( TURN - 3 - ($(date +%s) - t0) ))
  if [ $left -gt 20 ]; then echo "=== build ($left s)"; build $left || status=1; else echo "no time left to build"; fi
fi
finish
