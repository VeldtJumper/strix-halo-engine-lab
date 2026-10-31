#!/usr/bin/env bash
# Profile globals cross sourced module boundaries.
# shellcheck disable=SC2034,SC2154
# Public benchmark kit. Run from any directory; all relative asset paths use repo root.
set -eu
BENCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export BENCH_ROOT
cd -- "$BENCH_ROOT"
# shellcheck source=scripts/bench/lib/common.sh
source scripts/bench/lib/common.sh
# shellcheck source=scripts/bench/lib/pair.sh
source scripts/bench/lib/pair.sh
original_args=("$@")
# Profile globals are assigned by the selected sourced engine module.
action=${1:-help}
if (($#)); then shift; fi
engine='' peer='' out='' suite=full serve=0 dry=0 write_refs=0
bin_dir=${BENCH_BIN_DIR:-build}
model_dir=${BENCH_MODEL_DIR:-models}
peer_repo=${BENCH_PEER_REPO:-}
peer_bin_dir=${BENCH_PEER_BIN_DIR:-}
peer_model_dir=${BENCH_PEER_MODEL_DIR:-}
limit=${BENCH_TIMEOUT:-3600}
repeats='' contexts='' gen_tokens='' concurrency=1,4,8 warmup='' profile='' prompts=kit only=decode,prefill,exactness json=0 md=0 verify_models=cached
if [[ $action == report ]]; then
    [[ $# == 1 ]] || die 'usage: bench.sh report DIR'
    py report "$1/results.json"
    exit
fi
while (($#)); do
    case $1 in
        --engine|--peer|--out|--suite|--bin-dir|--model-dir|--peer-repo|--peer-bin-dir|--peer-model-dir|--timeout|--repeats|--repeat|--contexts|--gen-tokens|--concurrency|--warmup|--profile|--prompts|--only|--verify-models)
            [[ $# -ge 2 && -n $2 ]] || die "missing value: $1"
            case $1 in
                --engine) engine=$2 ;; --peer) peer=$2 ;; --out) out=$2 ;;
                --suite) suite=$2 ;; --bin-dir) bin_dir=$2 ;; --model-dir) model_dir=$2 ;;
                --peer-repo) peer_repo=$2 ;; --peer-bin-dir) peer_bin_dir=$2 ;;
                --peer-model-dir) peer_model_dir=$2 ;; --timeout) limit=$2 ;; --repeats|--repeat) repeats=$2 ;;
                --contexts) contexts=$2 ;; --gen-tokens) gen_tokens=$2 ;; --concurrency) concurrency=$2 ;;
                --warmup) warmup=$2 ;; --profile) profile=$2 ;; --prompts) prompts=$2 ;; --only) only=$2 ;; --verify-models) verify_models=$2 ;;
            esac
            shift 2 ;;
        --json) json=1; shift ;; --md) md=1; shift ;; --help) action=help; shift ;;
        --serve) serve=1; shift ;; --dry-run) dry=1; shift ;;
        --write-refs) write_refs=1; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
if [[ $action == help || $action == --help ]]; then
    py help
    exit 0
fi
if [[ $action == list ]]; then
    py list "$json"
    exit 0
fi
case $action in check|run|hash-models) ;; *) die 'expected list, check, run, hash-models or report' ;; esac
export BENCH_BIN_DIR="$bin_dir" BENCH_MODEL_DIR="$model_dir"
if [[ $engine == all ]]; then
    py all "${original_args[@]}"
    exit $?
fi
case $engine in deepseek) engine=deepseek-pair ;; glm) engine=glm-pair ;; minimax) engine=minimax-pair ;; esac
case $suite in quick|full) ;; *) die 'suite must be quick or full' ;; esac
[[ $limit =~ ^[1-9][0-9]*$ ]] || die 'timeout must be a positive integer'
# resolve emits only shell-quoted validated values and integer arrays.
resolved=$(py resolve "$engine" "$suite" "$contexts" "$gen_tokens" "$concurrency" "$repeats" "$warmup" "$profile" "$prompts" "$only" "$serve") || exit 1
eval "$resolved"
case $verify_models in full|cached|skip) ;; *) die "verify-models must be full, cached or skip" ;; esac
[[ $json == 0 || $md == 0 ]] || die "choose --json or --md"
# The selected profile decides per-profile assets, including for hash-models.
export BENCH_VERIFY_MODELS="$verify_models" BENCH_MODEL_DIR="$model_dir" BENCH_BIN_DIR="$bin_dir" BENCH_ENGINE="$engine" BENCH_SELECTED_PROFILE="$profile"
if [[ $action == hash-models ]]; then py hash-models; exit; fi
load_profile "$engine"
if [[ $status == pending-publication ]]; then
    printf '%s: pending-publication; no executable target in this tree; skipped.\n' "$engine"
    printf '{"engine":"%s","status":"pending-publication","skipped":true}\n' "$engine"
    [[ $action == run ]] && exit 0
    exit 1
fi
peer_repo=${peer_repo:-$BENCH_ROOT}
peer_bin_dir=${peer_bin_dir:-$bin_dir}
peer_model_dir=${peer_model_dir:-$model_dir}
export BENCH_ENGINE="$engine" BENCH_BIN_DIR="$bin_dir" BENCH_MODEL_DIR="$model_dir"
export BENCH_SUITE="$suite" BENCH_SERVE="$serve" BENCH_REPEATS="$repeats"
export BENCH_PROFILE="$identity:$profile"
export BENCH_CONTEXTS="${context_values[*]}" BENCH_GEN_TOKENS="$gen_tokens" BENCH_CONCURRENCY="$concurrency"
export BENCH_WARMUP="$warmup" BENCH_PROMPTS="$prompts" BENCH_ONLY="$only" BENCH_SELECTED_PROFILE="$profile"
resolved_params=$(py params "$resolved_params" "$limit" "$json" "$md" "$write_refs" "$dry" "$verify_models" "$peer" "$bin_dir" "$model_dir" "$peer_bin_dir" "$peer_model_dir" "$peer_repo")
export BENCH_PARAMS="$resolved_params" BENCH_START="$(date +%s)"
export BENCH_BIN_DIR="$bin_dir" BENCH_MODEL_DIR="$model_dir"
profile_environment
if [[ $action == check ]]; then
    rc=0
    if [[ $json == 1 ]]; then
        check_json=$(py check-json) || rc=1
        pair_rc=0
        if [[ $pair == 1 ]]; then pair_check >&2 || pair_rc=1; fi
        py check-result "$check_json" "$pair_rc" || rc=1
    else
        prerequisites || rc=1
        if [[ $pair == 1 ]]; then pair_check || rc=1; fi
    fi
    exit "$rc"
fi
if [[ $dry == 1 ]]; then
    if [[ $json == 1 || $md == 1 ]]; then exec 1>&2; fi
    printf 'DRY RUN: %s, suite=%s, repeats=%s, timeout=%ss; no GPU/model/SSH execution\n' "$engine" "$suite" "$repeats" "$limit"
    if [[ $pair == 1 ]]; then [[ -n $peer && $peer != -* ]] || die 'pair requires --peer HOST'; fi
    profile_dry_run
    exit
fi
if [[ ${BENCH_TIMEOUT_ACTIVE:-0} != 1 ]]; then
    # The outer timeout also bounds hashing, tokenization, HTTP readiness and SSH teardown.
    exec timeout --signal=TERM --kill-after=20 "${limit}s" env BENCH_TIMEOUT_ACTIVE=1 \
        bash "$BENCH_ROOT/scripts/bench/bench.sh" "${original_args[@]}"
fi
if [[ $json == 1 || $md == 1 ]]; then exec 3>&1; exec 1>&2; fi
prerequisites || exit 1
if [[ $pair == 1 ]]; then pair_check || exit 1; fi
out=${out:-bench-results/$engine-$(date -u +%Y%m%dT%H%M%SZ)}
[[ ! -e $out ]] || die 'output directory already exists; choose a new --out'
mkdir -p -- "$out"
out=$(cd -- "$out" && pwd)
export BENCH_OUT="$out"
# One deadline includes hashing, tokenization, all ranks, serving and teardown.
deadline=$((SECONDS + limit))
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ $pair == 1 ]]; then
    printf '%s\n' "$peer_inventory" >"$out/peer-inventory.json"
    printf '%s\n' "$local_inventory" >"$out/local-inventory.json"
fi
py initialize "$out"
py prompts "$out"
profile_run
# A stale result from a reused --out must never be printed for this run.
rm -f -- "$out/results.json" "$out/results.md"
finalize_rc=0
py finalize "$out" "$write_refs" || finalize_rc=$?
[[ -f $out/results.json ]] || exit "$finalize_rc"
printf 'Report: %s/results.md\n' "$out" >&2
if [[ $json == 1 || $md == 1 ]]; then exec 1>&3; fi
if [[ $json == 1 ]]; then cat "$out/results.json"; elif [[ $md == 1 ]]; then cat "$out/results.md"; fi
# A MISMATCH verdict (or failed candidate writing) exits 1 after the report is printed.
exit "$finalize_rc"
