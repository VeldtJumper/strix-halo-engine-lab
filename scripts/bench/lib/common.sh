#!/usr/bin/env bash
# Profile globals cross sourced module boundaries.
# shellcheck disable=SC2034,SC2154
set -eu
# Sourced modules exchange profile globals and command arrays with bench.sh.
local_pid='' remote_dir='' remote_active=0 peer_inventory=''
declare -A effective_remote_dirs=() effective_decode_env=()
py() { "${BENCH_PYTHON:-python3}" "$BENCH_ROOT/scripts/bench/lib/kit.py" "$@"; }
die() { printf 'bench: %s\n' "$*" >&2; exit 1; }
load_profile() {
    case $1 in qwen|deepseek-pair|glm-pair|minimax-pair|deepseek-rtx|nemotron) ;;
        *) die 'select --engine qwen|deepseek-pair|glm-pair|minimax-pair|deepseek-rtx|nemotron' ;;
    esac
    # These are project-owned files selected by the closed enum above.
    # shellcheck source=/dev/null
    source "$BENCH_ROOT/scripts/bench/engines/$1.sh"
    if [[ $status != pending-publication ]]; then
        [[ ${#assets[@]} == "${#asset_sha256s[@]}" && ${#assets[@]} == "${#asset_bytes[@]}" ]] || die 'invalid profile asset table'
        local pin
        for pin in "${asset_sha256s[@]}"; do [[ $pin == null || $pin =~ ^[0-9a-f]{64}$ ]] || die 'invalid profile SHA256'; done
        for pin in "${asset_bytes[@]}"; do [[ $pin == null || $pin =~ ^[1-9][0-9]*$ ]] || die 'invalid profile byte count'; done
    fi
}
profile_list() {
    local found=$status f
    if [[ $status != pending-publication ]]; then
        found=available
        for f in "${binaries[@]}"; do [[ -x $bin_dir/$f || -x $bin_dir/$f.exe ]] || found=binary-missing; done
    fi
    printf '%s: %s\n  binaries: %s\n  files: %s\n' "$e" "$found" "${binaries[*]:-none}" "${assets[*]:-none}"
}
binary() {
    if [[ -x $bin_dir/$1.exe ]]; then printf '%s\n' "$bin_dir/$1.exe"; else printf '%s\n' "$bin_dir/$1"; fi
}
prerequisites() {
    local failed=0 tool
    for tool in "${BENCH_PYTHON:-python3}" timeout sha256sum git; do
        if ! command -v "$tool" >/dev/null 2>&1; then printf 'MISSING tool: %s\n' "$tool" >&2; failed=1; fi
    done
    if [[ $pair == 1 ]]; then
        for tool in ssh setsid; do command -v "$tool" >/dev/null 2>&1 || failed=1; done
    fi
    if [[ $failed == 1 ]]; then printf '{"ok":false,"reason":"missing-tools"}\n'; return 1; fi
    py check
}
remaining() {
    local seconds=$((deadline - SECONDS))
    ((seconds > 0)) || die 'whole-run wall-clock timeout'
    printf '%s' "$seconds"
}
show_command() { printf '%q ' "$@"; printf '\n'; }
run_local() {
    local log=$1 rc=0
    shift
    printf "1\n" >>"$out/model-loads.txt"
    timeout --signal=TERM --kill-after=5 "$(remaining)s" "$@" >"$log" 2>&1 &
    local_pid=$!
    wait "$local_pid" || rc=$?
    local_pid=
    py sanitize "$log"
    return "$rc"
}
stop_local() {
    if [[ -n $local_pid ]]; then
        local members='' child fd target
        if [[ -d /proc && -e /dev/kfd ]]; then
            members=$(ps -eo pid=,pgid= | awk -v g="$local_pid" '$2 == g {print $1}')
        fi
        # timeout owns a separate process group and forwards both signals to its command.
        kill -TERM "$local_pid" 2>/dev/null || true
        sleep 1
        # Children by parent PID first: where process groups are unavailable, killing only the wrapper would
        # orphan a rank that ignores TERM.
        pkill -KILL -P "$local_pid" 2>/dev/null || true
        kill -KILL -- "-$local_pid" 2>/dev/null || kill -KILL "$local_pid" 2>/dev/null || true
        wait "$local_pid" 2>/dev/null || true
        local_pid=
        for child in $members; do
            for fd in /proc/"$child"/fd/*; do
                target=$(readlink "$fd" 2>/dev/null || true)
                [[ $target != /dev/kfd ]] || { printf 'Local teardown left a run process holding /dev/kfd.\n' >&2; return 1; }
            done
        done
    fi
}
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    stop_local || rc=1
    if [[ $remote_active == 1 ]]; then pair_stop || rc=1; fi
    if [[ -n ${out:-} && -d $out ]]; then
        py sanitize-all "$out" || rc=1
        if [[ ! -f $out/results.json ]]; then py finalize "$out" 0 failed || true; fi
    fi
    exit "$rc"
}
measure() {
    local prompt=$1 mode=$2 generation=$3 rep=$4
    local log="$out/$prompt-$mode-$rep.log"
    if [[ $pair == 1 ]]; then pair_run "$log" "${command[@]}"; else run_local "$log" "${command[@]}"; fi
    py ingest "$out" "$prompt" "$mode" "$generation" "$log"
}
serve_measure() {
    local log="$out/server.log" rc=0
    printf "1\n" >>"$out/model-loads.txt"
    py free-port "${BENCH_PORT:-18880}"
    if [[ $pair == 1 ]]; then pair_start "${command[@]}"; fi
    timeout --signal=TERM --kill-after=5 "$(remaining)s" env "${rank_env[@]}" "${command[@]}" >"$log" 2>&1 &
    local_pid=$!
    py http "$out" "${BENCH_PORT:-18880}" || rc=$?
    stop_local || rc=1
    if [[ $pair == 1 ]]; then pair_stop || rc=1; fi
    py sanitize "$log"
    return "$rc"
}

includes() { [[ ,$only, == *,$1,* ]]; }
decode_enabled() { includes decode || includes exactness; }

# Resolution is host-only and never counted as a model load. Pair probes use peer_exec,
# without starting a rank process group or requiring a listener.
effective_config() {
    local label=$1 flag=$2 rc=0
    shift 2
    if [[ ${dry:-0} == 1 ]]; then
        if [[ $pair == 1 ]]; then
            local remote_dir=REMOTE_RUN_DIR preview_repo=$peer_repo
            [[ $peer_repo != "$BENCH_ROOT" ]] || preview_repo=REPO
            pair_remote_command env "${rank_env[@]}" "$@" "$flag"
            printf 'node A (host-only config print): '; show_command env "${rank_env[@]}" "$@" "$flag"
            printf 'node B (SSH, host-only config print; no rank listener): '
            show_command timeout --signal=TERM --kill-after=5 "${limit}s" ssh -o BatchMode=yes -o ConnectTimeout=10 -- "$peer" "bash -c $(remote_quote "cd $(remote_quote "$preview_repo") && $(remote_quote "${remote_command[@]}")")"
        else show_command env "${rank_env[@]}" "$@" "$flag"; fi
        return
    fi
    local a="$out/effective-$label-rank0.log" b="$out/effective-$label-rank1.log"
    timeout --signal=TERM --kill-after=5 "$(remaining)s" env "${rank_env[@]}" "$@" "$flag" >"$a" 2>&1 || rc=$?
    if [[ $pair == 1 && $rc == 0 ]]; then
        local remote_dir=".bench-runs/$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM-$label"
        effective_remote_dirs[$label]=$remote_dir
        pair_remote_command env "${rank_env[@]}" "$@" "$flag"
        peer_exec "cd $(remote_quote "$peer_repo") && $(remote_quote "${remote_command[@]}")" "$(remaining)" >"$b" 2>&1 || rc=$?
    fi
    # Parse and compare before privacy redaction can hide meaningful differences.
    if [[ $rc == 0 ]]; then
        py effective "$out" "$label" "$a" "$b" "$model_dir" "${peer_model_dir:-$model_dir}" "$bin_dir" "${peer_bin_dir:-$bin_dir}" "${effective_remote_dirs[$label]:-}" "${rank_env[@]}" || rc=$?
    fi
    if [[ $rc == 0 && ${engine:-} == deepseek-pair && $label == decode-* ]]; then
        effective_decode_env[$label]=$(py resolved-env "$a" "${rank_env[@]}") || rc=$?
    fi
    py sanitize "$a"
    [[ ! -f $b ]] || py sanitize "$b"
    return "$rc"
}
