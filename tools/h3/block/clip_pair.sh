#!/bin/bash
# Two-node clip wrapper (run on the rank-0 node; both GPUs must be free for the run): strix-h3-clip --world 2,
# rank 0 (driver: prompt, text encoder, sampler, checkpoints, outputs in clips/$NAME) here, rank 1 (DiT + VAE server) on
# the peer node over ssh. A clip that stops on its budget (rc 3) continues in the next call with the same NAME / ARGS.
#   usage: NAME=clip ARGS="--width ... " [POST=cmd] [TURN_S=462] clip_pair.sh <run number>
#   env (required): H3_BLOCK_DIR = the working directory (same path on both nodes), H3_PEER = the rank-1 node (ssh name)
#   env: BIN (default $H3_BLOCK_DIR/bin/strix-h3-clip, the same build on both nodes), RANK_ENV (extra env for both ranks,
#        e.g. asset variables)
# Exit status (also in turn-N.done): 0 = both ranks 0 and POST ok; 3 = rank 0 stopped on its budget (rank 1 0, POST ok);
# 1 = any other rank exit code or a POST failure.
set -u
D=${H3_BLOCK_DIR:?H3_BLOCK_DIR required}; cd "$D" || exit 1
PEER=${H3_PEER:?H3_PEER required}
N=${1:-0}; T0=$(date +%s)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
BIN=${BIN:-$D/bin/strix-h3-clip}
NAME=${NAME:?NAME required}; ARGS=${ARGS:-}
status=1
trap '$SSH $PEER "test -f $D/r1-$N.pid && kill \$(cat $D/r1-$N.pid) 2>/dev/null"; echo "status $status" > $D/turn-$N.done' EXIT
echo "turn $N start $(date +%T) clip $NAME: $ARGS"
test -x "$BIN" && $SSH $PEER "test -x $BIN" || { echo "binary missing / not executable on a node"; exit 1; }
b0=$(sha256sum "$BIN" | cut -c1-16); b1=$($SSH $PEER "sha256sum $BIN" | cut -c1-16)
echo "bin $b0 / $b1"
[ "$b0" = "$b1" ] || { echo "the two nodes have different binaries"; exit 1; }
ENVS="STRIX_DIST_WAIT_TIMEOUT_MS=120000 STRIX_DIST_USB4_TIMEOUT_MS=120000 HSA_DISABLE_FRAGMENT_ALLOCATOR=1 ${RANK_ENV:-}"
left=$(( ${TURN_S:-462} - ($(date +%s) - T0) ))  # TURN_S: wall budget of this call
$SSH $PEER "cd $D && echo \$\$ > r1-$N.pid && exec env $ENVS timeout -k 10 $left $BIN --world 2 --rank 1 --transport usb4auto $ARGS" > r1-$N.log 2>&1 & R1=$!
env $ENVS timeout -k 10 $left "$BIN" --world 2 --rank 0 --transport usb4auto --out clips/$NAME --budget $(( left - 20 )) $ARGS > r0-$N.log 2>&1
rc0=$?
wait $R1; rc1=$?
echo "rc0=$rc0 rc1=$rc1 ($(( $(date +%s) - T0 )) s)"
grep -v '^ *$' r0-$N.log | cut -c1-420 | tail -45
echo "== rank 1"; grep -E 'rank1|error|rror' r1-$N.log | cut -c1-300 | tail -20
if [ $rc1 -ne 0 ] || { [ $rc0 -ne 0 ] && [ $rc0 -ne 3 ]; }; then status=1; elif [ $rc0 -eq 3 ]; then status=3; else status=0; fi
if [ -n "${POST:-}" ]; then
  echo "=== post: $POST"
  eval "$POST"; prc=$?
  echo "post rc $prc"
  [ $prc -ne 0 ] && status=1
fi
echo "turn $N end $(date +%T) $(( $(date +%s) - T0 )) s, status $status"
exit $status
