#!/usr/bin/env bash
set -eu
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$root"
export BENCH_ROOT="$root"
export BENCH_PYTHON=${BENCH_PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
tmp=$(mktemp -d)
cleanup_test() {
    # The target is the absolute, mktemp-owned test directory, never a repo directory.
    [[ $tmp == /* && -d $tmp && ${tmp##*/} == tmp.* ]] || return 1
    rm -rf -- "$tmp"
}
trap cleanup_test EXIT
for f in scripts/bench/bench.sh scripts/bench/fetch-models.sh scripts/bench/lib/*.sh scripts/bench/engines/*.sh scripts/bench/tests/*.sh; do
    bash -n "$f"
done
printf 'PASS bash syntax\n'
bash scripts/bench/bench.sh list >"$tmp/list"
[[ $(wc -l <"$tmp/list") -ge 6 ]]
printf 'PASS engine list\n'
mkdir -p "$tmp/bin" "$tmp/models"
cat >"$tmp/bin/ssh" <<'SHIM'
#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >>"$BENCH_SSH_RECORD"
exit 1
SHIM
cat >"$tmp/bin/nvidia-smi" <<'SHIM'
#!/usr/bin/env bash
set -eu
printf 'NVIDIA test GPU\n'
SHIM
cat >"$tmp/bin/rocminfo" <<'SHIM'
#!/usr/bin/env bash
set -eu
printf 'Name: gfx1151\n'
SHIM
cat >"$tmp/bin/hipconfig" <<'SHIM'
#!/usr/bin/env bash
set -eu
printf 'HIP version: 7.0.0\n'
SHIM
chmod +x "$tmp/bin/"*
case $(uname -s) in
    MINGW*|MSYS*)
        # Native Windows Python launches .cmd shims; Bash launches the shebang shims.
        printf '@echo off\necho NVIDIA test GPU\n' >"$tmp/bin/nvidia-smi.cmd"
        printf '@echo off\necho Name: gfx1151\n' >"$tmp/bin/rocminfo.cmd"
        printf '@echo off\necho HIP version: 7.0.0\n' >"$tmp/bin/hipconfig.cmd"
        ;;
esac
export PATH="$tmp/bin:$PATH"
export BENCH_SSH_RECORD="$tmp/ssh-record"
: >"$BENCH_SSH_RECORD"
if bash scripts/bench/bench.sh check --engine deepseek-rtx --bin-dir "$tmp/bin" --model-dir "$tmp/models" >"$tmp/check"; then
    printf 'check should reject absent assets\n' >&2; exit 1
fi
"$BENCH_PYTHON" -c 'import json,sys; d=json.loads(open(sys.argv[1]).read().splitlines()[-1]); assert d["ok"] is False and any("file-missing" in p for p in d["problems"])' "$tmp/check"
printf 'PASS check failure with tool shims and JSON\n'
# With lightweight files, prerequisites can succeed on the CUDA profile without running a GPU job.
for b in strix-tokenize strix-cuda-ds41-replica strix-cuda-ds41-prefill; do cp "$tmp/bin/nvidia-smi" "$tmp/bin/$b"; done
for f in DeepSeek-V4.1-Flash-1017-Q2.gguf DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf MODEL.receipt.json STATE.bin; do : >"$tmp/models/$f"; done
mkdir -p "$tmp/models/ds41-draft-vocab"
check_rc=0
bash scripts/bench/bench.sh check --engine deepseek-rtx --bin-dir "$tmp/bin" --model-dir "$tmp/models" >"$tmp/check-ok" || check_rc=$?
"$BENCH_PYTHON" -c 'import json,platform,sys; d=json.loads(open(sys.argv[1]).read().splitlines()[-1]); assert d["ok"] == (platform.system()=="Windows"); assert int(sys.argv[2]) == (0 if d["ok"] else 1)' "$tmp/check-ok" "$check_rc"
printf 'PASS complete-assets check with tool shims and platform admission (no execution)\n'
if bash scripts/bench/bench.sh check --engine deepseek-pair --peer node-b --peer-repo /path/to/repo \
    --bin-dir "$tmp/bin" --model-dir "$tmp/models" >"$tmp/pair-check" 2>&1; then
    printf 'pair check must fail against unreachable fake peer\n' >&2; exit 1
fi
grep -q 'BatchMode=yes' "$BENCH_SSH_RECORD"
grep -q 'inventory' "$BENCH_SSH_RECORD"
printf 'PASS pair check records remote inventory command in SSH shim\n'
: >"$BENCH_SSH_RECORD"
for e in qwen deepseek-pair glm-pair minimax-pair deepseek-rtx nemotron; do
    bash scripts/bench/bench.sh run --engine "$e" --peer node-b --peer-repo /path/to/repo --dry-run --suite full --serve >"$tmp/$e-preview"
done
# Dry-run must never contact SSH; it prints quoted remote commands for review.
[[ ! -s $BENCH_SSH_RECORD ]]
grep -q 'node B (SSH, setsid, PID file, same timeout)' "$tmp/deepseek-pair-preview"
# Nemotron: R8 verify, then one server per arm (drafting off first), C4/C8 on the shipped arm only.
grep -q -- '--verify-rows all --context 64' "$tmp/nemotron-preview"
[[ $(grep -c 'strix-gemma4-serve' "$tmp/nemotron-preview") == 2 ]]
grep -q -- '--draft off --dflash off' "$tmp/nemotron-preview"
grep -q 'HTTP arm=off concurrency=1 ' "$tmp/nemotron-preview"
grep -q 'HTTP arm=on concurrency=1,4,8 ' "$tmp/nemotron-preview"
bash scripts/bench/bench.sh run --engine nemotron --profile quality --suite quick --only exactness,serve --dry-run >"$tmp/nemotron-quality" 2>&1
grep -q -- '--exl3-arm models/nemotron/exl3' "$tmp/nemotron-quality"
if grep -q -- '--verify-rows' "$tmp/nemotron-quality"; then printf 'quality must not run the Q4_K_M R8 verify\n' >&2; exit 1; fi
grep -q 'SKIP verify-r8' "$tmp/nemotron-quality"
if bash scripts/bench/bench.sh run --engine nemotron --profile r1 --dry-run >/dev/null 2>&1; then exit 1; fi
printf 'PASS all six dry-runs, no GPU/assets/SSH execution\n'
bash scripts/bench/tests/effective_config.sh
bash scripts/bench/tests/single_config.sh
bash scripts/bench/tests/nemotron_shims.sh
bash scripts/bench/tests/pair_args.sh
printf 'PASS pair argument mapping and SSH shell quoting\n'
cp scripts/bench/tests/fixtures/results.json "$tmp/results.json"
bash scripts/bench/bench.sh report "$tmp"
"$BENCH_PYTHON" scripts/bench/lib/kit.py validate "$tmp/results.json"
grep -q 'NO-REFERENCE' "$tmp/results.md"
printf 'PASS fixture schema and markdown render\n'
"$BENCH_PYTHON" scripts/bench/tests/test_kit.py
"$BENCH_PYTHON" -B scripts/bench/tests/test_tasks.py
# Public machine endpoints must contain exactly one JSON document on stdout.
bash scripts/bench/bench.sh list --json >"$tmp/list.json"
"$BENCH_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert len(d["engines"])==6' "$tmp/list.json"
bash scripts/bench/bench.sh check --engine qwen --json --bin-dir "$tmp/absent" --model-dir "$tmp/absent" >"$tmp/check.json" || true
"$BENCH_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert not d["ok"] and "warnings" in d' "$tmp/check.json"
bash scripts/bench/bench.sh run --engine all --json --bin-dir "$tmp/absent" --model-dir "$tmp/absent" >"$tmp/all.json"
"$BENCH_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert len(d["engines"])==6 and all(e["skipped"] for e in d["engines"])' "$tmp/all.json"
for e in qwen deepseek glm minimax deepseek-rtx; do
    bash scripts/bench/bench.sh run --engine "$e" --peer node-b --contexts 2k,16k --only prefill --dry-run >"$tmp/$e-prefill"
    grep -q '2048' "$tmp/$e-prefill"
    grep -q '16384' "$tmp/$e-prefill"
    if grep -q -- '--generate 256' "$tmp/$e-prefill"; then exit 1; fi
done
# Fetch preview must execute neither network nor download shims.
for tool in curl hf; do
    printf '#!/usr/bin/env bash\nexit 93\n' >"$tmp/bin/$tool"
    chmod +x "$tmp/bin/$tool"
done
for e in qwen deepseek glm minimax deepseek-rtx; do
    bash scripts/bench/fetch-models.sh --engine "$e" --model-dir models --dry-run >"$tmp/$e-fetch"
    grep -q 'SHA256SUMS' "$tmp/$e-fetch"
    grep -q 'hf download hyukudan/' "$tmp/$e-fetch"
done
printf 'PASS parameter sweeps, pure JSON endpoints, all-engine skips and offline fetch previews\n'
# Exercise exactly the local teardown used by EXIT/INT/TERM, with a stubborn fake rank.
(
    source scripts/bench/lib/common.sh
    deadline=$((SECONDS + 30))
    # The fake rank expands these variables in its own shell.
    # shellcheck disable=SC2016
    timeout --signal=TERM --kill-after=5 30s bash -c 'echo "$BASHPID" > "$1"; trap "" TERM; while :; do sleep 1; done' bench-fake "$tmp/rank.pid" >"$tmp/rank.log" 2>&1 &
    local_pid=$!
    stubborn=$local_pid
    sleep 1
    rank_pid=$(cat "$tmp/rank.pid")
    stop_local
    # SIGKILL completion/reaping can lag the wrapper exit on the shell runtime.
    # Keep the final no-survivor assertion, allowing a bounded delivery interval.
    for attempt in {1..20}; do
        kill -0 "$rank_pid" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "$stubborn" 2>/dev/null; then printf 'teardown left timeout alive\n' >&2; exit 1; fi
    if kill -0 "$rank_pid" 2>/dev/null; then printf 'teardown left fake rank alive\n' >&2; exit 1; fi
)
printf 'PASS TERM-resistant rank teardown\n'
if command -v shellcheck >/dev/null 2>&1; then
    shellcheck -x scripts/bench/bench.sh scripts/bench/fetch-models.sh scripts/bench/lib/*.sh scripts/bench/engines/*.sh scripts/bench/tests/*.sh
    printf 'PASS shellcheck\n'
else
    printf 'SKIP shellcheck (not installed)\n'
fi
printf 'PASS CPU-only benchmark kit self-test\n'
