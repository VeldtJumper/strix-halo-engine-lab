#!/usr/bin/env python3
"""Native HTTP gate for DeepSeek target-only sampling; saves local receipts, never changes the server."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='HTTP server base URL')
    parser.add_argument('--out', required=True, help='Private output directory, outside the source tree')
    parser.add_argument('--api-key-file')
    parser.add_argument('--smoke-reference', default=str(Path(__file__).resolve().parents[1] / 'scripts/pair/ds41-smoke-reference.json'))
    parser.add_argument('--require-mixed', action='store_true', help='Require two active slots and observed mixed traffic')
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    headers = {'Content-Type': 'application/json'}
    if args.api_key_file:
        headers['Authorization'] = 'Bearer ' + Path(args.api_key_file).read_text().strip()
    records = []
    lock = threading.Lock()

    def save(name, value):
        path = root / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)

    def query(route, body=None):
        request = urllib.request.Request(args.url.rstrip('/') + route,
            data=None if body is None else json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    def ask(name, body, route='/v1/completions'):
        started = time.monotonic()
        response = query(route, body)
        ids = response['strix']['token_ids']
        assert ids, name + ': no token IDs'
        expected = 'sample' if body.get('temperature', 0) > 0 else 'greedy'
        assert response['strix']['sampling_effective'] == expected, name
        assert response['strix']['sampling'] == expected, name
        if expected == 'sample':
            assert response['strix']['seed'] == body['seed'], name
            assert response['strix']['sampling_route'] == 'target-only-host', name
        record = {'name': name, 'elapsed': time.monotonic() - started,
                  'digest': hashlib.sha256(json.dumps(ids).encode()).hexdigest(), 'response': response}
        with lock:
            records.append(record)
            save('requests.json', records)
        return ids

    status = query('/api/status')
    caps = query('/v1/capabilities')
    assert caps['sampling']['sample'] is True
    assert caps['sampling']['policy'] == 'sample'
    assert status['queue']['running'] == 0 and status['queue']['queued'] == 0, 'server must start idle'
    if args.require_mixed:
        assert status['slots']['total'] >= 2, 'mixed gate needs at least two slots'
    save('capabilities.json', caps)
    reference = json.loads(Path(args.smoke_reference).read_text())
    base = {'model': caps['model'], 'stream': False, 'return_token_ids': True, 'temperature': 0}
    smoke = {**base, 'prompt': reference['prompt'], 'max_tokens': reference['max_tokens']}
    assert ask('greedy-smoke-before', smoke) == reference['token_ids']
    text = ' '.join(f'Entry {i}: ordinary words describing a quiet garden with plants, paths and birds.' for i in range(150))
    prefix = query('/tokenize', {'prompt': text})['tokens'][:512]
    assert len(prefix) == 512
    raw = {**base, 'prompt': prefix, 'max_tokens': 96}
    greedy = ask('aligned-prefix-greedy', raw)
    assert ask('aligned-prefix-top-k-one', {**raw, 'temperature': 1.0, 'top_k': 1, 'seed': 41}) == greedy
    sampled = {**raw, 'temperature': 1.0, 'top_p': 1.0, 'seed': 41}
    first = ask('aligned-prefix-sampled', sampled)
    assert ask('aligned-prefix-same-seed-warm', sampled) == first
    appended = {**sampled, 'prompt': prefix + [prefix[-1]], 'max_tokens': 32}
    appended_ids = ask('appended-one-sampled', appended)
    assert ask('appended-one-same-seed', appended) == appended_ids
    variants = [ask(f'seed-{seed}', {**sampled, 'seed': seed}) for seed in [42, 43, 44]]
    assert len({tuple(ids) for ids in [first, *variants]}) > 1, 'different seeds never changed the output'
    # Short prompts use the row-prefill path rather than DevicePrefill.
    short = {**sampled, 'prompt': 'Once upon a time, in a quiet forest,', 'max_tokens': 32}
    short_ids = ask('short-sampled', short)
    assert ask('short-same-seed', short) == short_ids
    chat = {**base, 'temperature': 1.0, 'top_p': 1.0, 'seed': 7, 'max_tokens': 64,
            'messages': [{'role': 'user', 'content': 'Reply with exactly this JSON: {"status":"ok","count":3}'}]}
    ask('chat-sampled', chat, '/v1/chat/completions')

    # Compare the complete token sequence of the streaming and non-streaming routes.
    request = urllib.request.Request(args.url.rstrip('/') + '/v1/completions',
        data=json.dumps({**sampled, 'stream': True}).encode(), headers=headers)
    streamed = []
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith('data:'): continue
            payload = line[5:].strip()
            if payload == '[DONE]': break
            event = json.loads(payload)
            assert 'error' not in event, event
            streamed.extend((event.get('strix') or {}).get('token_ids') or [])
    assert streamed == first, 'streaming changed the seeded sequence'

    peak_busy = 0
    if status['slots']['total'] >= 2:
        barrier = threading.Barrier(3)
        def concurrent(name, body):
            barrier.wait()
            return ask(name, body)
        with ThreadPoolExecutor(max_workers=2) as pool:
            g = pool.submit(concurrent, 'mixed-greedy', raw)
            s = pool.submit(concurrent, 'mixed-sampled', sampled)
            barrier.wait()
            while not (g.done() and s.done()):
                peak_busy = max(peak_busy, query('/api/status')['slots']['busy'])
                threading.Event().wait(0.05)
            assert g.result() == greedy, 'mixed execution changed greedy tokens'
            assert s.result() == first, 'mixed execution changed sampled tokens'
        if args.require_mixed: assert peak_busy >= 2, 'requests did not overlap'

    for change in [{'temperature': -0.1}, {'top_p': 0}, {'top_k': -1}]:
        try:
            query('/v1/completions', {**sampled, **change})
        except urllib.error.HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError('invalid parameters were accepted: ' + str(change))

    # Disconnect after actual generated tokens, then demand a healthy completed request.
    request = urllib.request.Request(args.url.rstrip('/') + '/v1/completions',
        data=json.dumps({**sampled, 'max_tokens': 4096, 'stream': True}).encode(), headers=headers)
    received = 0
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith('data:'): continue
            payload = line[5:].strip()
            if payload == '[DONE]': break
            event = json.loads(payload)
            received += len((event.get('strix') or {}).get('token_ids') or [])
            if received >= 8: break
    assert received >= 8
    deadline = time.monotonic() + 60
    while query('/api/status')['queue']['running']:
        assert time.monotonic() < deadline, 'cancelled request did not drain'
        threading.Event().wait(0.1)
    assert ask('greedy-smoke-after-cancel', smoke) == reference['token_ids']
    final = query('/api/status')
    assert final['queue']['running'] == 0
    summary = {'passed': True, 'request_receipts': len(records), 'seed_variants': len({tuple(x) for x in [first, *variants]}),
               'greedy_reference_equal': True, 'top_k_one_equal_to_greedy': True, 'seed_replay_equal': True,
               'stream_replay_equal': True, 'peak_busy': peak_busy, 'mixed_observed': peak_busy >= 2,
               'cancellation_drained': True, 'slots': final['slots']['total']}
    save('summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
