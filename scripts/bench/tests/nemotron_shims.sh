#!/usr/bin/env bash
# Nemotron adapter through its shell path with local executable shims only: no device, model or network
# beyond a loopback test server that exits on its own.
set -eu
export BENCH_ROOT=$(pwd) BENCH_ENGINE=nemotron BENCH_SELECTED_PROFILE=default
source scripts/bench/lib/common.sh
profile=default
source scripts/bench/engines/nemotron.sh
test_dir=$(mktemp -d)
cleanup_shims() { [[ $test_dir == /* && ${test_dir##*/} == tmp.* ]] && rm -rf -- "$test_dir"; }
trap cleanup_shims EXIT
bin_dir="$test_dir/bin"; model_dir="$test_dir/models"; out="$test_dir/run"
mkdir -p "$bin_dir" "$model_dir/nemotron/rotary-tables" "$model_dir/nemotron/drafter-mtp" "$out"
printf gguf >"$model_dir/nemotron/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"
printf '{}' >"$model_dir/nemotron/tokenizer.json"
printf table >"$model_dir/nemotron/rotary-tables/sliding.bin"
printf config >"$model_dir/nemotron/drafter-mtp/config.json"
port=$("$BENCH_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
export BENCH_MODEL_DIR="$model_dir" BENCH_BIN_DIR="$bin_dir" BENCH_OUT="$out" BENCH_PORT=$port
export BENCH_PROMPTS="$test_dir/prompts.txt" BENCH_ONLY=exactness,serve BENCH_REPEATS=1 BENCH_WARMUP=0 BENCH_GEN_TOKENS=4
export BENCH_PROFILE=g4-v1-q4km:default BENCH_SUITE=quick BENCH_CONCURRENCY=1,4
export BENCH_PARAMS='{"only":["exactness","serve"],"profile":"default"}'
printf 'Hello there\nWrite a function\n' >"$BENCH_PROMPTS"
"$BENCH_PYTHON" - "$out" <<'PY'
import hashlib, json, sys
rows = [{'id': f'prompt-{i}', 'tokens': 3, 'construction': 'test-only',
         'sha256': hashlib.sha256(json.dumps([i, 2, 3], separators=(',', ':')).encode()).hexdigest()} for i in (1, 2)]
json.dump(rows, open(sys.argv[1] + '/prompts.json', 'w'))
json.dump({'repo_commit': '0' * 40, 'tracked_dirty': False, 'binary_sha256s': [], 'model_files': [], 'build_options': {},
           'models_verified': True, 'hardware': {'rocm_version': None, 'hip_version': None, 'kernel': 'test'}},
          open(sys.argv[1] + '/local-inventory.json', 'w'))
PY
cat >"$bin_dir/strix-gemma4-bench" <<'SHIM'
#!/usr/bin/env bash
set -eu
printf '{"profile":{"profile":"default"}}\n'
for r in 1 2 3 4 5 6 7 8; do
    kv=$(printf '%064x' "$r"); [[ $r != 8 ]] || kv=1c225a27533862d6e65361495b61743c66d6cf0539d5d13c06c758f25d78f5ff
    printf '{"gate":"verify_rows","identity":"g4-v1","verdict":"EXACT","rows":%s,"kv":"%s"}\n' "$r" "$kv"
done
SHIM
cat >"$test_dir/server.py" <<'PY'
import copy, http.server, json, sys, threading
fixture = json.load(open(sys.argv[1]))['phase']
args = sys.argv[2:]
off = '--draft' in args and args[args.index('--draft') + 1] == 'off'
port = int(args[args.index('--port') + 1])
threading.Timer(60, lambda: __import__('os')._exit(0)).start()  # never outlives the test
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        text = body['messages'][0]['content']
        ids = [len(text) % 7 + 1, 2, 3, 4][:body['max_tokens']]
        phase = copy.deepcopy(fixture); phase['state_digest'] = format(len(text), '064x')
        if off:
            for key in ('draft_kind', 'draft_identity', 'mtp_confidence_resolved'): phase.pop(key)
            phase['profile']['drafter'] = 'off (switch)'
            phase['acceptance'].update(attempted=0, eligible=0, accepted=0, cycles=0)
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        chunks = [{'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'x'}, 'finish_reason': None}]},
                  {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}],
                   'strix': {'phases': [phase], 'budget_continued': False, 'generated_ids': ids}},
                  {'choices': [], 'usage': {'prompt_tokens': 20, 'completion_tokens': len(ids), 'total_tokens': 20 + len(ids)}}]
        for c in chunks: self.wfile.write(b'data: ' + json.dumps(c).encode() + b'\n\n')
        self.wfile.write(b'data: [DONE]\n\n')
server = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler)
print('Gemma 4 service ready', file=sys.stderr, flush=True)
server.serve_forever()
PY
cat >"$bin_dir/strix-gemma4-serve" <<SHIM
#!/usr/bin/env bash
exec "$BENCH_PYTHON" "$test_dir/server.py" "$BENCH_ROOT/scripts/bench/tests/fixtures/nemotron-receipt.json" "\$@"
SHIM
chmod +x "$bin_dir/strix-gemma4-bench" "$bin_dir/strix-gemma4-serve"
dry=0; only=exactness,serve; serve=1; warmup=0; repeats=1; gen_tokens=4; concurrency=1,4
rank_env=(); deadline=$((SECONDS+120))
nemotron_execute
py initialize "$out"
py finalize "$out" 0
"$BENCH_PYTHON" - "$out" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); d = json.load(open(p / 'results.json'))
assert d['engine'] == 'nemotron' and d['status'] == 'complete', d['status']
assert (p / 'model-loads.txt').read_text().count('1') == 3
modes = {(e['prompt_id'], e['mode']) for e in d['exactness']}
assert modes == {(x, m) for x in ('prompt-1', 'prompt-2') for m in ('plain', 'spec', 'spec-c4')}, modes
assert {e['verdict'] for e in d['exactness']} == {'NO-REFERENCE'}
assert all(e['draft_witness']['status'] == 'drafted' for e in d['exactness'] if e['mode'].startswith('spec'))
# A synthetic GGUF never matches the published R8 pin: NO-REFERENCE, never EXACT.
assert [c['verdict'] for c in d['digest_checks']] == ['NO-REFERENCE']
assert {m['prompt_id'] for m in d['measurements'] if m.get('timing_scope') == 'http-end-to-end'} == {'http-c1-draft-off', 'http-c1', 'http-c4'}
assert 'numerics' in json.load(open(p / 'identity.json'))['reference_key']
PY
printf 'PASS Nemotron adapter shims: R8 verify, drafting off/on arms, C4 equality, NO-REFERENCE pins\n'
