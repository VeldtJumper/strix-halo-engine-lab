#!/usr/bin/env bash
# One invocation per phase. Evidence stays in OUT outside Git.
set -euo pipefail
SESSION_PHASE=${1:-}
case "$SESSION_PHASE" in
  prepare|prewarm|save-off|restore-off|save-on|restore-on|auto-serve-off|auto-serve-on|old-rank|off-rank|explicit-on-rank|old-serve-off|off-serve-off|explicit-on-serve-off|old-serve-on|off-serve-on|explicit-on-serve-on|off-serve-default|auto-serve-default|compare) ;;
  *) echo 'usage: session_cache_gate.sh PHASE (see docs/minimax/session-cache.md)'; exit 2 ;;
esac
export SESSION_PHASE
if [[ "${2:-}" != --inside-deadline ]]; then
  echo "Phase $SESSION_PHASE: expected <= 660 s including cold loads; hard deadline 690 s including verified cleanup"
  export SESSION_GATE_END=$(( $(date +%s) + 690 ))
  export SESSION_WORK_END=$((SESSION_GATE_END - 30))
  exec timeout --signal=TERM --kill-after=5 680 bash "$0" "$SESSION_PHASE" --inside-deadline
fi
: "${OUT:?persistent absolute run directory, same for every phase}"
: "${BIN:?absolute candidate rank executable on both nodes}"
: "${MODEL:?model directory on both nodes}"
: "${DSPARK:?draft directory on both nodes}"
: "${CPU_TEST:?CPU session test executable}"
: "${NODE1:?peer node for the second rank}"
BUDGET=${BUDGET:-8589934592}
[[ "$OUT" == /* && "$OUT" != / && "$OUT" != *..* ]] || exit 2
TURN_ROOT="$OUT/turn-$SESSION_PHASE"
[[ ! -e "$TURN_ROOT" && ! -e "$OUT/$SESSION_PHASE.done" ]] || { echo 'FAIL phase already attempted'; exit 2; }
if [[ $SESSION_PHASE != prepare ]]; then
  [[ -s "$OUT/prepare.done" ]] || { echo 'FAIL missing prepare receipt'; exit 2; }
fi
if [[ $SESSION_PHASE != prepare && $SESSION_PHASE != prewarm ]]; then
  [[ -s "$OUT/prewarm.done" ]] || { echo 'FAIL missing prewarm receipt'; exit 2; }
fi
if [[ $SESSION_PHASE != prepare ]]; then
  : "${STRIX_HASH_CACHE:?persistent absolute hash cache on both nodes}"
  [[ "$STRIX_HASH_CACHE" == /* ]] || exit 2
fi
if [[ $SESSION_PHASE != prepare && $SESSION_PHASE != prewarm ]]; then
  [[ -s "$OUT/prewarm.cache-dir" && $(cat "$OUT/prewarm.cache-dir") == "$STRIX_HASH_CACHE" ]] || { echo 'FAIL prewarm cache directory changed'; exit 2; }
fi
if [[ $SESSION_PHASE == prewarm ]]; then
  : "${BASELINE_BIN:?final staged baseline rank}" "${DEFAULT_BIN:?final staged candidate rank}"
  : "${BASELINE_SERVE:?final staged baseline serve}" "${DEFAULT_SERVE:?final staged candidate serve}"
fi
case "$SESSION_PHASE" in
  restore-off|restore-on)
    mode=${SESSION_PHASE#restore-}
    [[ -s "$OUT/save-$mode.done" && -s "$OUT/r0/$mode-A.sha256" && -s "$OUT/r1/$mode-A.sha256" && -s "$OUT/checkpoints/gate-$mode.group" ]] || { echo 'FAIL save phase prerequisites missing'; exit 2; } ;;
  auto-*|old-*|off-*|explicit-on-*|compare)
    : "${BASELINE_BIN:?observed baseline rank}" "${DEFAULT_BIN:?observed candidate rank}"
    : "${BASELINE_SERVE:?observed baseline serve}" "${DEFAULT_SERVE:?observed candidate serve}"
    : "${BASELINE_REVISION:?baseline revision receipt}" "${DEFAULT_REVISION:?candidate revision receipt}" ;;
esac
PEER_PID=; REMOTE_READY=0
remaining() { local n=$((SESSION_WORK_END-$(date +%s))); ((n>0)) || return 124; echo "$n"; }
bounded() { local n; n=$(remaining) || return 124; timeout -k 2 "$n" "$@"; }
remote() { bounded ssh -o BatchMode=yes -o ConnectTimeout=3 "$NODE1" "$@"; }
copy() { bounded scp -q -o BatchMode=yes -o ConnectTimeout=3 "$@"; }
# A shared launch lock closes admission before PID discovery, including delayed SSH.
cleaner='root=$1; exec 9>"$root/launch.lock"; flock 9 || exit 125; touch "$root/closed"; for f in "$root"/*.pid; do [ -f "$f" ] || continue; read -r p < "$f"; kill -KILL -- "-$p" 2>/dev/null || true; done; flock -u 9; for attempt in 1 2 3; do live=0; for f in "$root"/*.pid; do [ -f "$f" ] || continue; read -r p < "$f"; if kill -0 -- "-$p" 2>/dev/null; then live=1; fi; done; [ "$live" = 0 ] && exit 0; sleep 1; done; exit 1'
cleanup() {
  local rc=$? command
  trap - EXIT TERM INT HUP
  timeout -k 1 5 bash -c "$cleaner" gate "$TURN_ROOT" || rc=1
  if [[ $REMOTE_READY == 1 ]]; then
    printf -v command '%q ' bash -c "$cleaner" gate "$TURN_ROOT"
    timeout -k 1 8 ssh -o BatchMode=yes -o ConnectTimeout=3 "$NODE1" "$command" || rc=1
  fi
  if [[ -n $PEER_PID ]]; then kill "$PEER_PID" 2>/dev/null || true; wait "$PEER_PID" 2>/dev/null || true; fi
  (( $(date +%s) < SESSION_GATE_END )) || rc=1
  if [[ $rc == 0 ]]; then
    echo 'PASS cleanup verified on both nodes within 690 s'
    printf '%s\n' "$SESSION_PHASE" > "$OUT/$SESSION_PHASE.done"
  else echo 'FAIL gate/cleanup'; fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 124' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
bounded mkdir -p "$OUT/r0" "$OUT/r1" "$TURN_ROOT"
printf -v setup 'test ! -e %q && mkdir -p %q %q %q && date +%%s' "$TURN_ROOT" "$TURN_ROOT" "$OUT/r1" "$OUT/checkpoints"
REMOTE_READY=1
peer_clock=$(remote "$setup")
delta=$(( $(date +%s)-peer_clock )); ((delta>=-1 && delta<=1)) || { echo 'FAIL clock skew'; exit 1; }
runner='root=$1; tag=$2; deadline=$3; shift 3; exec 9>"$root/launch.lock"; flock 9 || exit 125; [ ! -e "$root/closed" ] || exit 125; n=$((deadline-$(date +%s))); [ "$n" -gt 0 ] || exit 124; setsid timeout -k 2 "$n" "$@" 9>&- & p=$!; echo "$p" > "$root/$tag.pid"; flock -u 9; wait "$p"; rc=$?; if kill -0 -- "-$p" 2>/dev/null; then kill -KILL -- "-$p"; exit 125; fi; rm -f "$root/$tag.pid"; echo "$rc" > "$root/$tag.ended"; exit "$rc"'
launch_pair() {
  local tag=$1 command rc0=0 rc1=0
  # Caller supplies localcmd and remote arrays. Remote computes remaining time
  # only after SSH arrival, against the same absolute deadline.
  printf -v command '%q ' bash -c "$runner" gate "$TURN_ROOT" "$tag-r1" "$SESSION_WORK_END" "${remote[@]}"
  printf -v command '%s > %q 2>&1' "$command" "$OUT/r1/$tag.log"
  remote "$command" > "$OUT/r1/$tag.ssh.log" 2>&1 & PEER_PID=$!
  bash -c "$runner" gate "$TURN_ROOT" "$tag-r0" "$SESSION_WORK_END" "${localcmd[@]}" > "$OUT/r0/$tag.log" 2>&1 &
  local local_pid=$!
  wait "$local_pid" || rc0=$?
  wait "$PEER_PID" || rc1=$?
  PEER_PID=
  [[ $rc0 == 0 && $rc1 == 0 ]] || { echo "FAIL $tag exits=$rc0/$rc1 (timeout/signal is FAIL)"; exit 1; }
  copy "$NODE1:$TURN_ROOT/$tag-r1.ended" "$TURN_ROOT/"
  copy "$NODE1:$OUT/r1/$tag.log" "$OUT/r1/"
  [[ $(cat "$TURN_ROOT/$tag-r0.ended") == 0 && $(cat "$TURN_ROOT/$tag-r1.ended") == 0 ]] || exit 1
  bounded python3 - "$OUT" "$tag" <<'PYTIME'
import json,pathlib,re,sys
root,tag=pathlib.Path(sys.argv[1]),sys.argv[2]
for rank in (0,1):
    text=(root/f"r{rank}/{tag}.log").read_text()
    lines=[line for line in text.splitlines() if line.startswith("session hashing ")]
    hashes=[]
    for line in lines:
        match=re.fullmatch(r"session hashing seconds ([0-9]+\.[0-9]+) cache_hits ([0-9]+) cache_misses ([0-9]+)",line)
        if not match:
            raise SystemExit(f"FAIL rank {rank}: malformed hashing receipt: {line}")
        seconds,hits,misses=match.groups()
        hashes.append(dict(seconds=float(seconds),cache_hits=int(hits),cache_misses=int(misses)))
    # Serving sessions run without the drafter by default (STRIX_MM_SESSION_DRAFT=1 restores it), so every serve arm
    # has the target lookup only; the rank arms with --session-dspark also hash the drafter assets.
    expected={"off-A":1,"off-B":1,"on-A":2,"on-B":2,
              "explicit-on-serve-off":1,"explicit-on-serve-on":1,
              "auto-serve-off":1,"auto-serve-on":1,"auto-serve-default":1}.get(tag.removesuffix("-adaptive"),0)
    io=[float(x) for x in re.findall(r"(?:write_fsync_s|read_io_s) ([0-9.]+)",text)]
    (root/f"r{rank}/{tag}.timing.json").write_text(json.dumps(dict(
        rank=rank,hash_seconds=sum(x["seconds"] for x in hashes),hash_operations=len(hashes),
        cache_hits=sum(x["cache_hits"] for x in hashes),
        cache_misses=sum(x["cache_misses"] for x in hashes),hash_lookups=hashes,
        checkpoint_io_seconds=sum(io),checkpoint_io_operations=len(io)))+"\n")
    if tag != "prewarm":
        if len(hashes) != expected or any(x["cache_misses"] != 0 or x["cache_hits"] == 0 for x in hashes):
            raise SystemExit(f"FAIL rank {rank}: expected {expected} all-hit compatibility lookups, got {hashes}")
PYTIME
}
ENV_ARGS=()
while IFS= read -r key; do ENV_ARGS+=("$key=${!key}"); done < <(compgen -e | LC_ALL=C sort | grep -E '^(STRIX_|HIP_|ROCR_|HSA_|AMD_)' || true)
pair() {
  local mode=$1 arm=$2 remain command peer rc0 rc1
  remain=$(remaining)
  (( remain > 5 )) || { echo "whole-run deadline exhausted"; exit 1; }
  local -a extra base remote localcmd
  base=("$BIN" "$MODEL")
  extra=(--slots 2 --capacity 8448 --max-rows 4096 --session-host-budget "$BUDGET" --session-id "gate-$mode" --session-suffix "$OUT/S.u32" --session-suffix-generate 64)
  if [[ "$mode" == on ]]; then extra+=(--session-dspark --dspark "$DSPARK"); fi
  if [[ "$arm" == A ]]; then
    extra+=(--session-prompt "$OUT/P.u32" --session-generate 8 --session-save "$OUT/checkpoints")
    remote=(env "${ENV_ARGS[@]}" "${base[@]}" 1 usb4-auto "$OUT/r1" "${extra[@]}" --session-output "$OUT/r1/$mode-A" --session-control-output "$OUT/r1/$mode-L")
    localcmd=(env "${ENV_ARGS[@]}" "${base[@]}" 0 usb4-auto "$OUT/r0" "${extra[@]}" --session-output "$OUT/r0/$mode-A" --session-control-output "$OUT/r0/$mode-L")
  else
    printf -v command 'test -s %q' "$OUT/checkpoints/gate-$mode.group"
    remote "$command" || { echo "FAIL peer save manifest missing"; exit 2; }
    extra+=(--session-load "$OUT/checkpoints")
    remote=(env "${ENV_ARGS[@]}" "${base[@]}" 1 usb4-auto "$OUT/r1" "${extra[@]}" --session-output "$OUT/r1/$mode-B")
    localcmd=(env "${ENV_ARGS[@]}" "${base[@]}" 0 usb4-auto "$OUT/r0" "${extra[@]}" --session-output "$OUT/r0/$mode-B")
  fi
  launch_pair "$mode-$arm"
  for rank in 0 1; do
    bounded grep -E 'session hashing seconds|session checkpoint bytes|session restore read_verify_agree_s' "$OUT/r$rank/$mode-$arm.log" > "$OUT/r$rank/$mode-$arm.timing"
    bounded grep -q 'session hashing seconds' "$OUT/r$rank/$mode-$arm.timing"
    bounded grep -Eq 'session checkpoint bytes|session restore read_verify_agree_s' "$OUT/r$rank/$mode-$arm.timing"
  done
  for phase in A L B; do
    if [[ "$arm" == A && "$phase" == B || "$arm" == B && "$phase" != B ]]; then continue; fi
    for ext in u32 f32 sha256; do copy "$NODE1:$OUT/r1/$mode-$phase.$ext" "$OUT/r1/" >/dev/null; done
  done
}
check_session_receipts() {
  local tag=$1 kind=$2 rank
  # The qualified serve route admits sessions automatically: no switch in the environment, runtime live on both ranks.
  if [[ $kind == serve && ( $tag == explicit-on-* || $tag == auto-* ) ]]; then
    for rank in 0 1; do
      bounded grep -Fx 'session cache: on (runtime live)' "$OUT/r$rank/$tag.log" || { echo "FAIL runtime not live on rank $rank"; exit 1; }
    done
  fi
}

default_pair() {
  default_pair_arm "$@" off
  if [[ $2 == serve ]]; then default_pair_arm "$1-adaptive" "$2" "$3" "$4" default; fi
}
default_pair_arm() {
  local tag=$1 kind=$2 binary=$3 mode=$4 scheduler=$5
  local -a remote localcmd extra unset_env=() probe_env=() sched_env=()
  local -a engine_env=("${ENV_ARGS[@]}")
  [[ $tag != explicit-on-serve-* && $tag != auto-serve-* ]] || probe_env=(MM_DEFAULT_REQUIRE_SESSION=1)
  if [[ $tag == auto-* ]]; then
    # Remove inherited switches so this is an actual automatic-admission check.
    unset_env=(-u STRIX_SESSION_CACHE -u STRIX_MM_BASELINE -u STRIX_BASELINE)
    local kept entry
    local -a kept_env=()
    for entry in "${engine_env[@]}"; do
      kept=${entry%%=*}
      [[ $kept == STRIX_SESSION_CACHE || $kept == STRIX_MM_BASELINE || $kept == STRIX_BASELINE ]] || kept_env+=("$entry")
    done
    engine_env=("${kept_env[@]}")
  elif [[ $tag == old-* ]]; then
    # The baseline runs as shipped (its own session default; it may already contain opt-in sessions).
    unset_env=(-u STRIX_SESSION_CACHE)
  elif [[ $tag == off-* ]]; then engine_env+=(STRIX_SESSION_CACHE=0); else engine_env+=(STRIX_SESSION_CACHE=1); fi
  extra=(--slots 3 --capacity 512 --max-rows 256)
  if [[ $kind == serve && $mode == on ]]; then extra+=(--dspark "$DSPARK"); fi
  # default = what a plain strix-minimax-serve resolves: DSpark mode, drafter discovery and geometry left to the defaults.
  [[ $mode != default ]] || extra=()
  if [[ $kind == rank ]]; then
    localcmd=(env "${unset_env[@]}" "${engine_env[@]}" "${probe_env[@]}" "MM_DEFAULT_PROBE=$OUT/$tag" "$binary" "$MODEL" 0 usb4-auto "$OUT/r0" "${extra[@]}" --session "default:$OUT/D.u32:64")
    remote=(env "${unset_env[@]}" "${engine_env[@]}" "${probe_env[@]}" "MM_DEFAULT_PROBE=$OUT/$tag" "$binary" "$MODEL" 1 usb4-auto "$OUT/r1" "${extra[@]}" --session "default:$OUT/D.u32:64")
  else
    local -a dspark_env=("STRIX_MM_SERVE_DSPARK=0")
    [[ $mode == on ]] && dspark_env=("STRIX_MM_SERVE_DSPARK=1")
    if [[ $mode == default ]]; then
      dspark_env=(); unset_env+=(-u STRIX_MM_SERVE_DSPARK)
      local forwarded
      local -a without_dspark=()
      for forwarded in "${engine_env[@]}"; do [[ ${forwarded%%=*} == STRIX_MM_SERVE_DSPARK ]] || without_dspark+=("$forwarded"); done
      engine_env=("${without_dspark[@]}")
    fi
    if [[ $scheduler == off ]]; then sched_env=(STRIX_MM_SERVE_SCHED=off);
    else
      unset_env+=(-u STRIX_MM_SERVE_SCHED -u STRIX_MM_SERVE_SERVICE_MS -u STRIX_MM_SERVE_GAP_MS
                  -u STRIX_MM_SERVE_NEST -u STRIX_MM_SERVE_SHORT_FIRST -u STRIX_MM_BASELINE -u STRIX_BASELINE)
      local entry
      local -a filtered=()
      for entry in "${engine_env[@]}"; do
        case ${entry%%=*} in
          STRIX_MM_SERVE_SCHED|STRIX_MM_SERVE_SERVICE_MS|STRIX_MM_SERVE_GAP_MS|STRIX_MM_SERVE_NEST|STRIX_MM_SERVE_SHORT_FIRST|STRIX_MM_BASELINE|STRIX_BASELINE) ;;
          *) filtered+=("$entry") ;;
        esac
      done
      engine_env=("${filtered[@]}")
    fi
    localcmd=(env "${unset_env[@]}" "${engine_env[@]}" "${probe_env[@]}" "${sched_env[@]}" "${dspark_env[@]}" "MM_DEFAULT_PROBE=$OUT/$tag" "MM_DEFAULT_PROMPT=$OUT/D.u32" "$binary" "$MODEL" --rank 0 --transport usb4-auto "${extra[@]}")
    remote=(env "${unset_env[@]}" "${engine_env[@]}" "${probe_env[@]}" "${sched_env[@]}" "${dspark_env[@]}" "MM_DEFAULT_PROBE=$OUT/$tag" "MM_DEFAULT_PROMPT=$OUT/D.u32" "$binary" "$MODEL" --rank 1 --transport usb4-auto "${extra[@]}")
  fi
  launch_pair "$tag"
  check_session_receipts "$tag" "$kind"
  for ext in u32 f32 state revision; do copy "$NODE1:$OUT/$tag.r1.$ext" "$OUT/"; done
  if [[ $kind == serve ]]; then
    for ext in send receive startup-send startup-receive; do copy "$NODE1:$OUT/$tag.r1.$ext" "$OUT/"; done
  fi
}
# Adaptive DSpark windows follow measured cycle costs, so the evaluated rows (u32/f32/send/receive and the state
# export's tail) differ between processes. The timing-independent witness is the emitted ids (compared separately)
# plus the committed state: the first prompt + emitted - 1 rows of the last state export.
committed_state() {
  local a=$1 b=$2 rank=$3 rows f size
  rows=$(( 128 + $(bounded stat -c %s "$OUT/$a.r0.emitted") / 4 - 1 ))
  [[ $rows == $(( 128 + $(bounded stat -c %s "$OUT/$b.r0.emitted") / 4 - 1 )) ]] || exit 1
  for f in "$OUT/$a.r$rank.state" "$OUT/$b.r$rank.state"; do
    size=$(bounded stat -c %s "$f")
    (( size >= rows*76800 && size % 76800 == 0 )) || exit 1
  done
  bounded cmp -n $((rows*76800)) "$OUT/$a.r$rank.state" "$OUT/$b.r$rank.state"
}
compare_default() {
  local old=$1 candidate=$2 kind=$3 rank ext width
  local -a extensions=(u32 f32 state)
  [[ $kind == serve ]] && extensions+=(send receive startup-send startup-receive)
  [[ $candidate == *-adaptive ]] && extensions=(startup-send startup-receive)
  for rank in 0 1; do
    width=100096; [[ $rank == 1 ]] && width=99968
    [[ $(cat "$OUT/$old.r$rank.revision") == $(cat "$BASELINE_REVISION") ]] || exit 1
    [[ $(cat "$OUT/$candidate.r$rank.revision") == $(cat "$DEFAULT_REVISION") ]] || exit 1
    for ext in "${extensions[@]}"; do
      [[ -s $OUT/$old.r$rank.$ext && -s $OUT/$candidate.r$rank.$ext ]] || exit 1
      bounded cmp "$OUT/$old.r$rank.$ext" "$OUT/$candidate.r$rank.$ext"
    done
    [[ $candidate != *-adaptive ]] || committed_state "$old" "$candidate" "$rank"
    if [[ $kind == rank ]]; then
      [[ $(bounded stat -c %s "$OUT/$old.r$rank.u32") == 260 ]]
      [[ $(bounded stat -c %s "$OUT/$old.r$rank.f32") == $((65*width*4)) ]]
      [[ $(bounded stat -c %s "$OUT/$old.r$rank.state") == $((192*76800)) ]]
    else
      # Adaptive verification may evaluate extra rows; require a complete witness.
      (( $(bounded stat -c %s "$OUT/$old.r$rank.u32") >= 256 ))
      [[ $(bounded stat -c %s "$OUT/$old.r$rank.f32") == $(( $(bounded stat -c %s "$OUT/$old.r$rank.u32") * width )) ]]
      (( $(bounded stat -c %s "$OUT/$old.r$rank.state") >= 128*76800 ))
      (( $(bounded stat -c %s "$OUT/$old.r$rank.state") % 76800 == 0 ))
      [[ $(bounded stat -c %s "$OUT/$old.r$rank.send") -ge 512 ]]
      [[ $(bounded stat -c %s "$OUT/$old.r$rank.receive") -ge 512 ]]
    fi
  done
}
case "$SESSION_PHASE" in
  prepare)
# CPU negatives exercise real envelope/group code: exe, numerics, full-file mutation,
# corrupt rank/section, truncated file, missing rank, generation mismatch and RAM peaks.
(cd "$OUT" && bounded "$CPU_TEST") > "$OUT/cpu-negatives.log" 2>&1
cat "$OUT/cpu-negatives.log"
bounded python3 - "$OUT" <<'PY'
import pathlib,struct,sys
p=pathlib.Path(sys.argv[1])
(p/"P.u32").write_bytes(struct.pack("<8192I",*[100+i%100 for i in range(8192)]))
(p/"D.u32").write_bytes(struct.pack("<128I",*[100+i%100 for i in range(128)]))
(p/"S.u32").write_bytes(struct.pack("<4I",101,102,103,104))
PY
copy "$OUT/P.u32" "$OUT/S.u32" "$OUT/D.u32" "$NODE1:$OUT/" >/dev/null

    ;;
  prewarm)
    # All binaries must be finally staged before this separate CPU-only invocation.
    # ldd inventories their startup libraries without invoking an engine.
    hash_runner='bin=$1; model=$2; draft=$3; rank=$4; shift 4
      args=(); for binary in "$bin" "$@"; do
        test -s "$binary" || exit 2
        args+=(--session-hash-file "$binary")
        libs=$(ldd "$binary") || exit 2
        [[ "$libs" != *"not found"* ]] || exit 2
        while IFS= read -r lib; do args+=(--session-hash-file "$lib"); done < <(printf "%s\n" "$libs" | grep -oE "/[^[:space:]]+")
      done
      exec "$bin" "$model" "$rank" unused unused --session-hash-only --session-dspark --dspark "$draft" "${args[@]}"'
    localcmd=(env "${ENV_ARGS[@]}" bash -c "$hash_runner" gate "$BIN" "$MODEL" "$DSPARK" 0 "$BASELINE_BIN" "$DEFAULT_BIN" "$BASELINE_SERVE" "$DEFAULT_SERVE")
    remote=(env "${ENV_ARGS[@]}" bash -c "$hash_runner" gate "$BIN" "$MODEL" "$DSPARK" 1 "$BASELINE_BIN" "$DEFAULT_BIN" "$BASELINE_SERVE" "$DEFAULT_SERVE")
    launch_pair prewarm
    printf '%s\n' "$STRIX_HASH_CACHE" > "$OUT/prewarm.cache-dir"
    for rank in 0 1; do
      bounded grep -E "^session prewarm rank $rank wall_s [0-9.]+ second_s [0-9.]+ cache_hits [1-9][0-9]* cache_misses 0 all_hits=1$" "$OUT/r$rank/prewarm.log" > "$OUT/r$rank/prewarm.receipt"
    done
    ;;
  save-off|save-on) pair "${SESSION_PHASE#save-}" A ;;
  restore-off|restore-on) pair "${SESSION_PHASE#restore-}" B ;;
  old-rank|off-rank|explicit-on-rank)
    binary="$DEFAULT_BIN"; [[ $SESSION_PHASE == old-* ]] && binary="$BASELINE_BIN"
    default_pair "$SESSION_PHASE" rank "$binary" off ;;
  auto-serve-*|old-serve-*|off-serve-*|explicit-on-serve-*)
    binary="$DEFAULT_SERVE"; [[ $SESSION_PHASE == old-* ]] && binary="$BASELINE_SERVE"
    default_pair "$SESSION_PHASE" serve "$binary" "${SESSION_PHASE##*-}" ;;
  compare)
    for phase in prewarm auto-serve-off auto-serve-on save-off restore-off save-on restore-on old-rank off-rank explicit-on-rank old-serve-off off-serve-off explicit-on-serve-off old-serve-on off-serve-on explicit-on-serve-on off-serve-default auto-serve-default; do
      [[ -s "$OUT/$phase.done" ]] || { echo "FAIL missing receipt $phase"; exit 2; }
    done
    for mode in off on; do
      for rank in 0 1; do
        width=100096; [[ $rank == 1 ]] && width=99968
        for arm in A L B; do
          [[ $(bounded stat -c %s "$OUT/r$rank/$mode-$arm.u32") == 256 ]]
          [[ $(bounded stat -c %s "$OUT/r$rank/$mode-$arm.f32") == $((64*width*4)) ]]
          [[ $(bounded stat -c %s "$OUT/r$rank/$mode-$arm.sha256") == 64 ]]
        done
        for arm in L B; do
          for ext in u32 f32 sha256; do bounded cmp "$OUT/r$rank/$mode-A.$ext" "$OUT/r$rank/$mode-$arm.$ext"; done
        done
      done
      for suffix in "" "-adaptive"; do
        compare_default "old-serve-$mode$suffix" "off-serve-$mode$suffix" serve
        for rank in 0 1; do
          bounded cmp "$OUT/off-serve-$mode$suffix.r$rank.revision" "$OUT/explicit-on-serve-$mode$suffix.r$rank.revision"
          for on in explicit-on auto; do
            bounded cmp "$OUT/off-serve-$mode$suffix.r$rank.revision" "$OUT/$on-serve-$mode$suffix.r$rank.revision"
            if [[ $suffix == -adaptive ]]; then committed_state "off-serve-$mode$suffix" "$on-serve-$mode$suffix" "$rank"
            else for ext in u32 f32 state send receive; do bounded cmp "$OUT/off-serve-$mode$suffix.r$rank.$ext" "$OUT/$on-serve-$mode$suffix.r$rank.$ext"; done; fi
          done
        done
        for build in old off explicit-on auto; do
          [[ $(bounded stat -c %s "$OUT/$build-serve-$mode$suffix.r0.emitted") == 256 ]]
          bounded cmp "$OUT/$build-serve-$mode$suffix.r0.send" "$OUT/$build-serve-$mode$suffix.r1.receive"
          bounded cmp "$OUT/$build-serve-$mode$suffix.r1.send" "$OUT/$build-serve-$mode$suffix.r0.receive"
        done
        bounded cmp "$OUT/old-serve-$mode$suffix.r0.emitted" "$OUT/off-serve-$mode$suffix.r0.emitted"
        bounded cmp "$OUT/off-serve-$mode$suffix.r0.emitted" "$OUT/explicit-on-serve-$mode$suffix.r0.emitted"
        bounded cmp "$OUT/off-serve-$mode$suffix.r0.emitted" "$OUT/auto-serve-$mode$suffix.r0.emitted"
      done
    done
    # Production default serve path: automatic admission == OFF for unnamed requests.
    for suffix in "" "-adaptive"; do
      for rank in 0 1; do
        bounded cmp "$OUT/off-serve-default$suffix.r$rank.revision" "$OUT/auto-serve-default$suffix.r$rank.revision"
        if [[ $suffix == -adaptive ]]; then committed_state "off-serve-default$suffix" "auto-serve-default$suffix" "$rank"
        else for ext in u32 f32 state send receive; do bounded cmp "$OUT/off-serve-default$suffix.r$rank.$ext" "$OUT/auto-serve-default$suffix.r$rank.$ext"; done; fi
      done
      for build in off auto; do
        [[ $(bounded stat -c %s "$OUT/$build-serve-default$suffix.r0.emitted") == 256 ]]
        bounded cmp "$OUT/$build-serve-default$suffix.r0.send" "$OUT/$build-serve-default$suffix.r1.receive"
        bounded cmp "$OUT/$build-serve-default$suffix.r1.send" "$OUT/$build-serve-default$suffix.r0.receive"
      done
      bounded cmp "$OUT/off-serve-default$suffix.r0.emitted" "$OUT/auto-serve-default$suffix.r0.emitted"
    done
    compare_default old-rank off-rank rank
    for rank in 0 1; do bounded cmp "$OUT/off-rank.r$rank.revision" "$OUT/explicit-on-rank.r$rank.revision"; for ext in u32 f32 state; do bounded cmp "$OUT/off-rank.r$rank.$ext" "$OUT/explicit-on-rank.r$rank.$ext"; done; done
    echo 'PASS A == L == B; OFF == baseline; unnamed explicit-on == OFF; automatic admission live == OFF' | tee -a "$OUT/results.txt"
    ;;
esac
