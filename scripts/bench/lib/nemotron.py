"""Nemotron 3.5 Bolt E4B (gemma4 family): assets, receipts, draft witness and HTTP arms.

One long-lived server per arm: drafting off (`--draft off --dflash off`: no MTP drafter, no prompt lookup) and the shipped configuration (drafter
found in `drafter-mtp/`). The C1 round of each arm carries the exactness evidence (complete IDs and the
committed state digest from the engine receipt); the drafter arm must also carry a positive draft
witness. C4/C8 rounds run on the shipped arm and are compared with its C1 IDs. Each profile is its own
population: default and fast use the Q4_K_M GGUF, quality uses the EXL3 pack; references never mix.
"""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request

DIR = 'nemotron'
Q4 = DIR + '/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf'
ROPE = DIR + '/rotary-tables'
TOKENIZER = DIR + '/tokenizer.json'
BF16 = DIR + '/bf16'
EXL3 = DIR + '/exl3'
BINARIES = ['strix-gemma4-serve', 'strix-gemma4-bench']
PROFILES = ('default', 'fast', 'quality')
# Decode identity each Q4 profile must report; quality reports its EXL3 identity instead.
DECODE_IDENTITY = {'default': 'g4-v1', 'fast': 'g4-a8-v1'}
MODEL_NAME = 'nemotron-bolt-e4b'
READY = 'Gemma 4 service ready'
FAILED = 'gemma4 serve: '
# Levers whose value decides output bits (bit-changing or pack); exact levers stay out of the key.
NUMERIC_KINDS = ('bit-changing', 'pack')
NUMERIC_FIELDS = ('decode_identity', 'prefill_identity', 'sampling_identity', 'exl3_identity',
                  'template_identity', 'row_cap')


def selected_profile():
    profile = os.environ.get('BENCH_SELECTED_PROFILE') or 'default'
    return profile if profile in PROFILES else 'default'


def assets(profile=None):
    """Required assets per profile, relative to the model directory."""
    profile = profile or selected_profile()
    if profile == 'quality':
        return [BF16, EXL3, ROPE, TOKENIZER]
    return [Q4, ROPE, TOKENIZER]


def drafter(profile=None):
    """The engine discovers the MTP drafter next to its model: <model dir>/drafter-mtp."""
    profile = profile or selected_profile()
    return (BF16 if profile == 'quality' else DIR) + '/drafter-mtp'


def optional_assets(profile=None):
    # The quality drafter lives inside bf16/ and is hashed with it.
    return [] if (profile or selected_profile()) == 'quality' else [drafter(profile)]


def warnings(models, profile=None):
    if not (Path(models) / drafter(profile)).is_dir():
        return ['drafter-mtp absent: drafter cells and draft exactness are skipped (unqualified, never PASS)']
    return []


def normalize(value, models):
    """Replace the model directory in a receipt value by a stable marker; other paths are withheld."""
    if not isinstance(value, str):
        return value
    text = re.sub('/+', '/', value.replace('\\', '/'))
    root = re.sub('/+', '/', str(models).replace('\\', '/')).rstrip('/')
    if root and (text == root or text.startswith(root + '/')):
        return 'MODEL_DIR' + text[len(root):]
    return '<path>' if '/' in text else value


def phase_of(final):
    """The single engine receipt of a chat completion (no thinking budget means one phase)."""
    strix = final.get('strix') if isinstance(final, dict) else None
    if not isinstance(strix, dict):
        raise ValueError('SSE final chunk lacks the engine receipt')
    phases = strix.get('phases')
    if not isinstance(phases, list) or len(phases) != 1 or not isinstance(phases[0], dict):
        raise ValueError('expected exactly one engine receipt phase')
    ids = strix.get('generated_ids')
    if not isinstance(ids, list) or not ids or not all(type(x) is int and 0 <= x <= 0xffffffff for x in ids):
        raise ValueError('engine receipt lacks complete generated IDs')
    return phases[0], ids


def check_receipt(phase):
    """Fields every arm needs to compare, key and time a request."""
    required = ('state_digest', 'decode_identity', 'prefill_identity', 'sampling_identity', 'profile',
                'acceptance', 'timings', 'finish_reason')
    missing = [key for key in required if key not in phase]
    if missing:
        raise ValueError('incomplete Nemotron receipt: missing ' + ', '.join(missing))
    if not isinstance(phase['state_digest'], str) or not re.fullmatch(r'[0-9a-f]{16,128}', phase['state_digest']):
        raise ValueError('Nemotron receipt has no committed state digest')
    acceptance = phase['acceptance']
    for key in ('cycles', 'attempted', 'eligible', 'accepted', 'emitted'):
        if not isinstance(acceptance, dict) or type(acceptance.get(key)) is not int or acceptance[key] < 0:
            raise ValueError('Nemotron receipt lacks acceptance counter: ' + key)
    decode_ms = phase['timings'].get('decode_ms') if isinstance(phase['timings'], dict) else None
    if not isinstance(decode_ms, (int, float)) or not math.isfinite(decode_ms) or decode_ms <= 0:
        raise ValueError('Nemotron receipt lacks decode timing')
    profile = phase['profile']
    if not isinstance(profile, dict) or not isinstance(profile.get('levers'), dict) or 'pending' not in profile:
        raise ValueError('Nemotron receipt lacks the resolved profile')


def check_profile(phase, profile, notices):
    """The receipt must report the selected profile, applied ready rows only, and listed pending rows."""
    p = phase['profile']
    if p.get('profile') != profile:
        raise ValueError('receipt profile differs from the selected profile')
    if p.get('qualification') != 'qualified' or p.get('unqualified'):
        raise ValueError('unqualified profile rows applied; benchmark runs never allow pending rows')
    for row in p['pending']:
        lever = p['levers'].get(row.get('lever'))
        if not lever or lever.get('status') != 'pending':
            raise ValueError('pending row is not reported as pending: ' + str(row.get('lever')))
        notice = 'profile {}: {} pending {} ({} runs)'.format(profile, row['lever'], row.get('target'), row.get('runs'))
        if notices.count(notice) != 1:
            raise ValueError('pending row lacks its single stderr notice: ' + row['lever'])
    expected = DECODE_IDENTITY.get(profile)
    if expected and phase['decode_identity'] != expected:
        raise ValueError('decode identity differs from the profile: ' + str(phase['decode_identity']))
    if profile == 'quality' and not phase.get('exl3_identity'):
        raise ValueError('quality receipt lacks its EXL3 identity')
    if profile != 'quality' and phase.get('exl3_identity'):
        raise ValueError('Q4_K_M profile receipt carries an EXL3 identity')
    if phase['prefill_identity'] != 'prefill-exact':
        raise ValueError('prefill identity is not the exact route')


def numerics(phase, models):
    """Projection that decides output bits: the reference key, identical in both arms."""
    result = {key: phase[key] for key in NUMERIC_FIELDS if key in phase}
    result['profile'] = phase['profile'].get('profile')
    result['levers'] = {name: {'value': normalize(lever.get('value'), models), 'status': lever.get('status')}
                        for name, lever in sorted(phase['profile']['levers'].items())
                        if lever.get('kind') in NUMERIC_KINDS}
    return result


def draft_witness(phase, arm):
    """Drafting off must not draft; the drafter arm must have loaded the MTP drafter and report its window."""
    a = phase['acceptance']
    witness = {'drafted': a['attempted'], 'verified': a['eligible'], 'accepted': a['accepted'], 'cycles': a['cycles']}
    drafter_state = str(phase['profile'].get('drafter', ''))
    if arm == 'off':
        if phase.get('draft_kind') is not None or a['attempted'] or not drafter_state.startswith('off'):
            raise ValueError('drafting-off arm loaded or ran a drafter')
        return dict(witness, status='off')
    if phase.get('draft_kind') != 'mtp-assistant' or not drafter_state.startswith('on'):
        raise ValueError('missing draft witness: no MTP drafter loaded in the drafter arm')
    confidence = phase.get('mtp_confidence_resolved')
    if not isinstance(confidence, dict) or not confidence:
        raise ValueError('missing draft witness: mtp_confidence_resolved absent')
    for key in ('confidence_cycles', 'confidence_drafted'):
        if type(a.get(key)) is not int:
            raise ValueError('missing draft witness: acceptance.' + key + ' absent')
    witness.update(confidence_drafted=a['confidence_drafted'], confidence=confidence,
                   draft_identity=phase.get('draft_identity'))
    drafted = witness['drafted'] > 0 and witness['verified'] > 0 and witness['accepted'] > 0
    witness['status'] = 'drafted' if drafted else 'declined'
    return witness


def read_sse(lines, clock, start):
    """Parse a complete SSE chat stream. Truncated, failed or unterminated streams are rejected."""
    first = final = usage = None
    done = False
    for raw in lines:
        line = raw.strip()
        if not line.startswith(b'data:'):
            continue
        if line == b'data: [DONE]':
            done = True
            continue
        if done:
            raise ValueError('SSE data after [DONE]')
        obj = json.loads(line[5:])
        if 'error' in obj:
            raise ValueError('HTTP generation error')
        for choice in obj.get('choices', []):
            delta = choice.get('delta') or {}
            if first is None and (delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls')):
                first = clock() - start
            if choice.get('finish_reason') is not None:
                if final is not None:
                    raise ValueError('SSE stream has more than one final chunk')
                final = obj
        if obj.get('usage'):
            usage = obj['usage']
    if not done:
        raise ValueError('truncated SSE: no [DONE]')
    if final is None or usage is None:
        raise ValueError('truncated SSE: missing final chunk or usage')
    if first is None:
        raise ValueError('SSE response lacks first content')
    return first, final, usage


def chat_request(base, text, max_tokens, timeout=600):
    body = {'model': MODEL_NAME, 'messages': [{'role': 'user', 'content': text}], 'max_tokens': max_tokens,
            'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}}
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        first, final, usage = read_sse(response, time.perf_counter, start)
    elapsed = time.perf_counter() - start
    phase, ids = phase_of(final)
    check_receipt(phase)
    # Chat jobs stop at the end of turn; exactness compares the complete ID vector either way.
    finish = final['choices'][0].get('finish_reason')
    if finish not in ('length', 'stop') or usage.get('completion_tokens') != len(ids) or len(ids) > max_tokens             or (finish == 'length' and len(ids) != max_tokens):
        raise ValueError('incomplete completion: finish reason or token count inconsistent with usage')
    if type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens'] <= 0:
        raise ValueError('usage lacks prompt tokens')
    return {'ids': ids, 'phase': phase, 'ttft_s': first, 'elapsed_s': elapsed,
            'prompt_tokens': usage['prompt_tokens'], 'completion_tokens': len(ids)}


def wait_ready(log, base, deadline_s=600):
    end = time.perf_counter() + deadline_s
    while time.perf_counter() < end:
        text = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
        failed = next((line for line in text.splitlines() if line.startswith(FAILED)), None)
        if failed:
            raise ValueError('server failed: ' + failed[len(FAILED):])
        if READY in text:
            try:
                with urllib.request.urlopen(base + '/health', timeout=2) as r:
                    if r.status == 200:
                        return text
            except OSError:
                pass
        time.sleep(0.5)
    raise ValueError('HTTP server readiness timeout')


def append_jsonl(path, rows):
    with path.open('a', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def serve_arm(k, out, port, arm, levels):
    """Measure one server arm. C1 rounds carry exactness; C>1 rounds compare with C1 of this arm."""
    if arm not in ('off', 'on'):
        raise ValueError('arm must be off or on')
    profile = selected_profile()
    base = 'http://127.0.0.1:' + str(int(port))
    log = wait_ready(out / ('server-' + arm + '.log'), base)
    notices = [line.strip() for line in log.splitlines() if line.startswith('profile ')]
    texts = k.prompt_texts()
    names = list(texts)
    count = max(8, len(names))
    gen = int(os.environ.get('BENCH_GEN_TOKENS', '128'))
    warmup = int(os.environ.get('BENCH_WARMUP', '0'))
    repeats = int(os.environ.get('BENCH_REPEATS', '1'))
    levels = sorted({1, *(int(x) for x in str(levels).split(',') if x)})
    if any(c < 1 or c > 8 for c in levels):
        raise ValueError('serve concurrency is limited to 1..8')
    serve = 'serve' in os.environ.get('BENCH_ONLY', '').split(',')
    prompt_sha = {x['id']: x['sha256'] for x in k.read(out / 'prompts.json')}
    mode = 'plain' if arm == 'off' else 'spec'
    reference, prompt_tokens, numeric = {}, {}, None
    witnesses, measurements = [], []

    def one(index):
        name = names[index % len(names)]
        result = chat_request(base, texts[name], gen)
        result['prompt_id'] = name
        return result

    def inspect(result):
        nonlocal numeric
        phase = result['phase']
        check_profile(phase, profile, notices)
        bits = numerics(phase, k.MODELS)
        if numeric is None:
            numeric = bits
        elif bits != numeric:
            raise ValueError('conflated identities: numerical projection changed between requests')
        seen = prompt_tokens.setdefault(result['prompt_id'], result['prompt_tokens'])
        if seen != result['prompt_tokens']:
            raise ValueError('server prompt token count differs between requests')
        result['witness'] = draft_witness(phase, arm)

    for c in levels:
        rates, ttfts, rounds = [], [], []
        for r in range(warmup + repeats):
            start = time.perf_counter()
            if c == 1:
                results = [one(i) for i in range(count)]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                    results = list(pool.map(one, range(count)))
            wall = time.perf_counter() - start
            for result in results:
                inspect(result)
            if r < warmup:
                continue
            rates.append(sum(x['completion_tokens'] for x in results) / wall)
            ttfts.extend(x['ttft_s'] for x in results)
            rows = []
            for result in results:
                phase, name = result['phase'], result['prompt_id']
                digest = {'state_digest': phase['state_digest']}
                if c == 1:
                    reference.setdefault(name, (result['ids'], phase['state_digest']))
                    witnesses.append(result['witness'])
                    rows.append({'prompt_id': name, 'mode': mode, 'generate_length': gen,
                                 'actual_tokens': len(result['ids']), 'token_ids_sha256': k.id_hash(result['ids']),
                                 'engine_digest': digest, 'draft_witness': result['witness'],
                                 'values': [len(result['ids']) * 1000 / phase['timings']['decode_ms']],
                                 'unit': 'tok/s', 'log': 'server-' + arm + '.log', 'timing_scope': 'native'})
                else:
                    ids, state = reference[name]
                    rows.append({'prompt_id': name, 'mode': mode + '-c' + str(c), 'generate_length': gen,
                                 'actual_tokens': len(result['ids']), 'token_ids_sha256': k.id_hash(result['ids']),
                                 'prompt_sha256': prompt_sha[name], 'engine_digest': digest,
                                 'draft_witness': result['witness'],
                                 'c1_equal': result['ids'] == ids and phase['state_digest'] == state})
            append_jsonl(out / ('samples.jsonl' if c == 1 else 'concurrency-exactness.jsonl'), rows)
            rounds.append([{key: x[key] for key in ('prompt_id', 'ttft_s', 'elapsed_s', 'prompt_tokens',
                                                    'completion_tokens')} | {'token_ids_sha256': k.id_hash(x['ids'])}
                           for x in results])
        label = 'http-c' + str(c) + ('-draft-off' if arm == 'off' else '')
        if serve:
            measurements += [{'prompt_id': label, 'mode': 'aggregate-wall', 'unit': 'tok/s',
                              'timing_scope': 'http-end-to-end', **k.stats(rates)},
                             {'prompt_id': label, 'mode': 'ttft', 'unit': 's',
                              'timing_scope': 'http-end-to-end', **k.stats(ttfts)}]
        k.dump(out / (label + '.json'), {'concurrency': c, 'max_tokens': gen, 'arm': arm, 'requests': rounds})
    if arm == 'on' and not any(w['status'] == 'drafted' for w in witnesses):
        raise ValueError('draft positive control failed: drafter loaded but no drafted, verified and accepted tokens at C1')
    if serve:
        path = out / 'http-measurements.json'
        k.dump(path, (k.read(path) if path.exists() else []) + measurements)
    # Resolved configuration from the measured server receipts: no separate model load.
    settings = {'numerics': numeric, 'drafter': witnesses[0].get('draft_identity') if arm == 'on' else None,
                'mtp_confidence_resolved': witnesses[0].get('confidence') if arm == 'on' else None}
    # The adapter sets no engine environment; the reference key uses the numerical projection instead,
    # so it does not depend on whether the drafter arm ran.
    k.dump(out / ('effective-serve-' + arm + '.json'), {'environment': {}, 'arguments': [], 'settings': settings})
    summary_path = out / 'nemotron-arms.json'
    summary = k.read(summary_path) if summary_path.exists() else {}
    summary[arm] = {'profile': numeric['profile'], 'levels': levels, 'chat_prompt_tokens': prompt_tokens,
                    'pending_notices': notices, 'witness': {
                        'drafted': sum(1 for w in witnesses if w['status'] == 'drafted'),
                        'declined': sum(1 for w in witnesses if w['status'] == 'declined'),
                        'off': sum(1 for w in witnesses if w['status'] == 'off')}}
    k.dump(summary_path, summary)


def parse_verify(text, profile):
    """strix-gemma4-bench --verify-rows all: rows 1..8 EXACT (host == serial == batched), R8 KV digest."""
    rows = [o for o in _objects(text) if o.get('gate') == 'verify_rows']
    if [o.get('rows') for o in rows] != list(range(1, 9)) or any(o.get('verdict') != 'EXACT' for o in rows):
        raise ValueError('verify-rows must report EXACT for rows 1..8')
    identities = {o.get('identity') for o in rows}
    if len(identities) != 1 or not re.fullmatch(r'[0-9a-f]{64}', rows[-1].get('kv', '')):
        raise ValueError('verify-rows identity or KV digest missing')
    identity = identities.pop()
    if DECODE_IDENTITY.get(profile) != identity:
        raise ValueError('verify-rows identity differs from the profile: ' + str(identity))
    return {'check': 'verify-r8', 'identity': identity, 'rows': 8, 'kv': rows[-1]['kv']}


def _objects(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def digest_verdict(check, pins, model_files):
    """R8 KV digest against the published pin for its identity and Q4_K_M file; else NO-REFERENCE."""
    pin = (pins or {}).get(check['check'], {}).get(check['identity'])
    q4 = next((f['sha256'] for f in model_files if f.get('role') == Q4), None)
    if not pin or q4 is None or q4 != pin.get('model_sha256'):
        return dict(check, reference=None, verdict='NO-REFERENCE')
    return dict(check, reference=pin['kv'], verdict='EXACT' if check['kv'] == pin['kv'] else 'MISMATCH')


def summarize_witness(group, prompt):
    """A drafter row is 'drafted' only when every sample drafted, verified and accepted tokens."""
    if not group or any('draft_witness' not in s for s in group):
        raise ValueError('missing draft witness: ' + prompt)
    statuses = {s['draft_witness'].get('status') for s in group}
    if not statuses <= {'drafted', 'declined'}:
        raise ValueError('invalid draft witness: ' + prompt)
    return {'status': 'drafted' if statuses == {'drafted'} else 'declined',
            **{key: sum(s['draft_witness'][key] for s in group) for key in ('drafted', 'verified', 'accepted', 'cycles')}}


def augment(data, out, refs, known, read):
    """Cross-arm exactness, draft witnesses, concurrency rows and digest checks for finalize."""
    plain = {e['prompt_id']: e for e in data['exactness'] if e['mode'] == 'plain'}
    samples = [json.loads(x) for x in (out / 'samples.jsonl').read_text(encoding='utf-8').splitlines()] \
        if (out / 'samples.jsonl').exists() else []
    for e in data['exactness']:
        if e['mode'] != 'spec':
            continue
        e['draft_witness'] = summarize_witness([s for s in samples if s['prompt_id'] == e['prompt_id'] and s['mode'] == 'spec'],
                                               e['prompt_id'])
    spec = [e for e in data['exactness'] if e['mode'] == 'spec']
    if spec and not any(e['draft_witness']['status'] == 'drafted' for e in spec):
        raise ValueError('draft positive control failed: no prompt drafted, verified and accepted')
    rows = [json.loads(x) for x in (out / 'concurrency-exactness.jsonl').read_text(encoding='utf-8').splitlines()] \
        if (out / 'concurrency-exactness.jsonl').exists() else []
    groups = {}
    for row in rows:
        groups.setdefault((row['prompt_id'], row['mode'], row['generate_length']), []).append(row)
    for (prompt, mode, length), group in groups.items():
        stable = len({r['token_ids_sha256'] for r in group}) == 1 and all(r['engine_digest'] == group[0]['engine_digest'] for r in group)
        candidate = {'prompt_sha256': group[0]['prompt_sha256'], 'token_ids_sha256': group[0]['token_ids_sha256'],
                     'engine_digest': group[0]['engine_digest'], 'actual_tokens': group[0]['actual_tokens']}
        target = known.get(f"{data['identity']['profile']}:{prompt}:{mode}:{length}")
        verdict = 'MISMATCH' if not stable or not all(r['c1_equal'] for r in group) else \
            'NO-REFERENCE' if target is None else 'EXACT' if target == candidate else 'MISMATCH'
        entry = {'prompt_id': prompt, 'mode': mode, 'generate_length': length, 'verdict': verdict,
                 'repeat_stable': stable, 'c1_equal': all(r['c1_equal'] for r in group), **candidate}
        if mode.startswith('spec'):
            entry['draft_witness'] = summarize_witness(group, prompt)
        data['exactness'].append(entry)
    # Every drafter row (C1 and C>1) must equal drafting off: IDs and committed state.
    for e in data['exactness']:
        if e['mode'] == 'plain':
            continue
        base = plain.get(e['prompt_id'])
        if base is None:
            continue
        e['plain_tokens_equal'] = e['token_ids_sha256'] == base['token_ids_sha256']
        e['plain_state_equal'] = e['engine_digest'] == base['engine_digest']
        if not (e['plain_tokens_equal'] and e['plain_state_equal']):
            e['verdict'] = 'MISMATCH'
    if (out / 'nemotron-arms.json').exists():
        arms = read(out / 'nemotron-arms.json')
        tokens = [a.get('chat_prompt_tokens') for a in arms.values()]
        if any(t != tokens[0] for t in tokens):
            raise ValueError('server prompt token counts differ between arms')
        data['nemotron'] = arms
    checks = []
    if (out / 'verify-r8.json').exists():
        checks.append(digest_verdict(read(out / 'verify-r8.json'), refs.get('digest_pins'),
                                     data['identity'].get('model_files', [])))
    if checks:
        data['digest_checks'] = checks
    if any(e['verdict'] == 'MISMATCH' for e in data['exactness']) or any(c['verdict'] == 'MISMATCH' for c in checks):
        data['status'] = 'mismatch'
