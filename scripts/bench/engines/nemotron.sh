#!/usr/bin/env bash
# Profile globals cross sourced module boundaries.
# shellcheck disable=SC2034,SC2154
set -eu
# Nemotron 3.5 Bolt E4B (gemma4 family), one node. Binaries come from -DSTRIX_FAMILY_GEMMA4=ON.
# Each profile is its own population: default (g4-v1) and fast (g4-a8-v1) on the Q4_K_M GGUF,
# quality on the EXL3 pack. Assets live under MODEL_DIR/nemotron/ (see docs/reproduce.md).
status=available
pair=0
binaries=(strix-gemma4-serve strix-gemma4-bench)
case ${profile:-default} in
    fast) identity=g4-a8-v1-q4km ;;
    quality) identity=g4-exl3-pack ;;
    *) identity=g4-v1-q4km ;;
esac
if [[ ${profile:-default} == quality ]]; then
    assets=(nemotron/bf16 nemotron/exl3 nemotron/rotary-tables nemotron/tokenizer.json)
    asset_sha256s=(null null null null)
    asset_bytes=(null null null null)
else
    assets=(nemotron/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf nemotron/rotary-tables nemotron/tokenizer.json)
    asset_sha256s=(null null null)
    asset_bytes=(null null null)
fi
profile_environment() { rank_env=(); }
# Plain production command for the selected profile; the drafter is discovered next to the model.
nemotron_target() {
    local m=$model_dir/nemotron
    if [[ $profile == quality ]]; then
        target=(--model "$m/bf16" --exl3-arm "$m/exl3")
        drafter_dir=$m/bf16/drafter-mtp
    else
        target=(--model "$m/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf")
        drafter_dir=$m/drafter-mtp
    fi
    target+=(--profile "$profile" --backend device --rope-tables "$m/rotary-tables" --tokenizer "$m/tokenizer.json")
}
# R8 verify: rows 1..8 host == serial == batched, then the R8 committed KV digest (Q4_K_M identities only).
nemotron_verify_command() {
    local m=$model_dir/nemotron
    command=("$(binary strix-gemma4-bench)" --profile "$profile" --model "$m/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"
        --route q4_k_m --rope "$m/rotary-tables" --verify-rows all --context 64 --time-graph)
}
# One long-lived server per arm, plain production command. Chat requests stop at the end of turn or at
# --gen-tokens; the receipt carries the complete ID vector either way.
nemotron_serve_command() {
    local arm=$1
    command=("$(binary strix-gemma4-serve)" "${target[@]}" --bind 127.0.0.1 --port "${BENCH_PORT:-18880}")
    [[ $arm == on ]] || command+=(--draft off --dflash off)
}
nemotron_serve_arm() {
    local arm=$1 levels=$2 rc=0 log="$out/server-$1.log"
    printf "1\n" >>"$out/model-loads.txt"
    py free-port "${BENCH_PORT:-18880}"
    timeout --signal=TERM --kill-after=5 "$(remaining)s" "${command[@]}" >"$log" 2>&1 &
    local_pid=$!
    py nemotron-http "$out" "${BENCH_PORT:-18880}" "$arm" "$levels" || rc=$?
    stop_local || rc=1
    py sanitize "$log"
    return "$rc"
}
nemotron_note() {
    printf 'SKIP %s\n' "$1" >&2
    if [[ $dry != 1 ]]; then py skipped "$out" "$1"; fi
}
nemotron_execute() {
    local arms=(off) arm levels target=() drafter_dir=''
    nemotron_target
    # Dry runs preview both arms; real runs add the drafter arm only when the drafter is present.
    if [[ $dry == 1 || -d $drafter_dir ]]; then arms+=(on)
    else nemotron_note 'drafter-mtp absent: C1 + drafter cells and draft exactness not run (unqualified)'; fi
    if includes prefill; then
        nemotron_note 'prefill: not measured by this adapter; the engine prefill route is prefill-exact for every profile'
    fi
    if includes exactness; then
        if [[ $profile == quality ]]; then
            nemotron_note 'verify-r8: no published EXL3 R8 digest; quality exactness is IDs and state against drafting off'
        else
            nemotron_verify_command
            if [[ $dry == 1 ]]; then show_command "${command[@]}"
            else
                run_local "$out/verify-r8.log" "${command[@]}"
                py nemotron-verify "$out" "$out/verify-r8.log"
            fi
        fi
    fi
    decode_enabled || [[ $serve == 1 ]] || return 0
    for arm in "${arms[@]}"; do
        levels=1
        # C4/C8 run on the shipped configuration (the last arm); drafting off measures C1 only.
        if [[ $serve == 1 && $arm == "${arms[-1]}" ]]; then levels=$concurrency; fi
        nemotron_serve_command "$arm"
        if [[ $dry == 1 ]]; then
            show_command "${command[@]}"
            printf 'HTTP arm=%s concurrency=%s output=%s repeat=%s warmup=%s\n' "$arm" "$levels" "$gen_tokens" "$repeats" "$warmup"
        else
            nemotron_serve_arm "$arm" "$levels"
        fi
    done
}
profile_dry_run() { local out=DETERMINISTIC_PROMPTS; nemotron_execute; }
profile_run() { nemotron_execute; py initialize "$out"; }
