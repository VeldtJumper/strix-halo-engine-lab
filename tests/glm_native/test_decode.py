"""CPU-only decode tests: cache head contract, decode window logic, receiver snapshot mode.

No device, CUDA import, socket or model data. Temporary files live under --build.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = None


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache = load('cache', 'src/glm/rtxpf/producer/cache.py')
dw = load('decode_window', 'src/glm/rtxpf/producer/decode_qualification_window.py')
CHAT_ALL = len(dw.CHAT_KINDS)


class CacheHeadTests(unittest.TestCase):
    def test_root_names_k6_contract(self):
        plain, head = cache.required_schema(), cache.required_schema(head=True)
        self.assertEqual(set(head) - set(plain), {'lm_head.mul1', 'lm_head.trellis', 'lm_head.suh', 'lm_head.svh'})
        self.assertFalse(any(n.startswith('model.language_model.lm_head') for n in head))
        self.assertEqual(head['lm_head.trellis'], ('I16', [256, 9680, 96]))
        self.assertEqual(head['lm_head.mul1'], ('I32', []))
        sizes = cache.SIZES
        for name, (dtype, shape) in head.items():
            if name in cache.HEAD_BYTES:
                n = sizes[dtype]
                for d in shape:
                    n *= d
                self.assertEqual(n, cache.HEAD_BYTES[name], name)
        self.assertEqual(sum(cache.HEAD_BYTES.values()), 476109316)
        self.assertEqual(cache.blob_of('lm_head.trellis'), 'head.bin')
        self.assertEqual(cache.blob_of('model.language_model.norm.weight'), 'base.bin')
        self.assertEqual(cache.blob_of('model.language_model.layers.10.mlp.experts.5.up_proj.suh'), 'experts-10.bin')

    def test_cache_key_separation(self):
        plain = hashlib.sha256(cache.canonical(cache.key_contract('a'*64))).hexdigest()
        legacy = hashlib.sha256(cache.canonical(dict(cache.VERSION, model_sha256='a'*64))).hexdigest()
        head = hashlib.sha256(cache.canonical(cache.key_contract('a'*64, True))).hexdigest()
        self.assertEqual(plain, legacy, 'prefill cache key unchanged')
        self.assertNotEqual(plain, head)

    def manifest(self, td, head):
        sources = dict(config={}, files=[])
        model = hashlib.sha256(cache.canonical(sources)).hexdigest()
        m = dict(cache.key_contract(model, head), sources=sources,
                 cache_key=hashlib.sha256(cache.canonical(cache.key_contract(model, head))).hexdigest(), blobs={}, tensors={})
        (td/'manifest.json').write_bytes(cache.canonical(m))
        return m

    def test_missing_head_rejected_and_inventory_checked(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            td = Path(td)
            self.manifest(td, False)
            with self.assertRaisesRegex(ValueError, 'head-capable'):
                cache.verify(td, require_head=True)
            with self.assertRaisesRegex(ValueError, 'inventory'):
                cache.verify(td)  # empty tensor inventory still rejected
            m = self.manifest(td, True)
            m['head'] = 'other'
            (td/'manifest.json').write_bytes(cache.canonical(m))
            with self.assertRaisesRegex(ValueError, 'head contract'):
                cache.verify(td)


class WindowTests(unittest.TestCase):
    def args(self, prompt_dir=Path('PROMPTS')):
        return types.SimpleNamespace(build=Path('.'), model=Path('MODEL'), cache=None, cache_manifest_sha256=None,
                                     prompt_dir=prompt_dir, steps=4, bench_steps=4, expert_tier='reference',
                                     timeout=1800, allow_gpu=False, check_inputs=False, unit_file=None,
                                     tier_vram_gib=None, tier_ram_gib=None, unit_manifest_sha256=None)

    def record(self, steps=4, n=1024, file_bytes=4096):
        e = lambda c, t: dict(t=t, frontier=n-1+t, rank0_sha256=c*64, rank1_sha256=c*64, rank0_bytes=2097152, rank1_bytes=2097152,
                              replay_sha256='f'*64, components=[[0, 0, 2097152]])
        gen = list(range(11, 12+steps))
        inc = [e('a', t) for t in range(steps+1)]
        stepped = [dict(token=y, position=n-1+i, logits_sha256=('%x' % (y % 16))*64, experts=dict(file_bytes=file_bytes))
                   for i, y in enumerate(gen)]
        fresh = []
        for t in range(1, steps+1):
            f = e('a', t)
            f['next'] = dict(token=gen[t], logits_sha256=stepped[t]['logits_sha256'], position=n-1+t)
            fresh.append(f)
        return dict(prompt_tokens=n, legacy_prefill=dict(rank0_sha256='a'*64, rank1_sha256='a'*64),
                    incremental_exports=inc, incremental_steps=stepped, generated=gen, fresh=fresh)

    def bench(self, order=('cold', 'warm'), steps=4, file_bytes=4096, n=1024):
        arm = lambda name: dict(arm=name, tokens=[5]*steps, logits_sha256=['c'*64]*steps, generation_loop_s=1.0,
                                tokens_per_s=steps/1.0,
                                steps=[dict(token=5, logits_sha256='c'*64, position=n-1+j, experts=dict(file_bytes=file_bytes))
                                       for j in range(steps)],
                                final_state=dict(rank0_sha256='d'*64, rank1_sha256='d'*64, replay_sha256='e'*64,
                                                 rank0_bytes=1, rank1_bytes=1, frontier=n + steps - 1, components=[[0, 0, 1]]))
        return dict(prompt_tokens=n, arms=[arm(x) for x in order])

    @staticmethod
    def prefill_state(n):
        """A complete prefill state signature at frontier n - 1 (session_digest fields)."""
        return dict(rank0_sha256='a'*64, rank1_sha256='b'*64, rank0_bytes=10, rank1_bytes=10, replay_sha256='d'*64,
                    frontier=n - 1, components=[[0, 0, 10]])

    def test_plan_starts_nothing(self):
        a = self.args()
        a.expert_tier, a.unit_file, a.unit_manifest_sha256, a.tier_ram_gib = 'tierstream', Path('U'), 'f'*64, 32.0
        with patch.object(dw.subprocess, 'Popen') as popen, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(dw.run(a), 0)
            popen.assert_not_called()
        plan = json.loads(out.getvalue())
        self.assertIn('--decode-commands', plan['command'])
        self.assertIn('--require-file-tier', plan['command'])
        self.assertEqual([i['op'] for i in plan['items']][:3], ['qualify', 'qualify', 'compare'])
        self.assertEqual(len(plan['items']), 2 + 1 + 2 + 2, 'gate, compare, TTFT 1K/8K, one chat kind at 1K/8K (cached path only)')
        self.assertLessEqual(plan['expected_wall_seconds'], dw.MAX_SECONDS)

    def test_uuid_deadline_and_source_checked_before_process(self):
        a = self.args()
        a.allow_gpu = True
        with patch.dict(dw.os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(dw, 'start') as start:
            with self.assertRaises(ValueError):
                dw.run(a)
            start.assert_not_called()
        for t in (0, 1801):
            a.timeout = t
            with self.assertRaises(ValueError):
                dw.run(a)
        a.timeout, a.cache = 100, Path('CACHE')
        with self.assertRaises(ValueError):
            dw.run(a)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--steps', '2'])

    def test_judge_exact_mismatch_missing(self):
        self.assertEqual(dw.judge_qualify(self.record(), 4, 1024), ('EXACT', []))
        for mutate, why in ((lambda r: r['fresh'][0].update(rank1_sha256='b'*64), 'state SHA'),
                            (lambda r: r['fresh'][1]['next'].update(logits_sha256='0'*64), 'logits'),
                            (lambda r: r['fresh'][0]['next'].update(token=99), 'token'),
                            (lambda r: r['fresh'][1].update(replay_sha256='0'*64), 'replay'),
                            (lambda r: r['fresh'][0].update(components=[[1, 0, 2097152]]), 'component'),
                            (lambda r: r['legacy_prefill'].update(rank0_sha256='c'*64), 'prefill producer')):
            r = self.record()
            mutate(r)
            v, problems = dw.judge_qualify(r, 4, 1024)
            self.assertEqual(v, 'MISMATCH')
            self.assertTrue(any(why in p for p in problems), (why, problems))
        r = self.record()
        del r['fresh'][1]
        self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
        r = self.record()
        r['incremental_steps'][0]['logits_sha256'] = 'short'
        self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')

    def test_coverage_required(self):
        self.assertEqual(dw.qualify_coverage(self.record(4, 1024), 1024, 4), [])
        self.assertEqual(dw.qualify_coverage(self.record(4, 8192), 8192, 4), [])
        self.assertTrue(any('B32' in m for m in dw.qualify_coverage(self.record(4, 1000), 1000, 4)))
        self.assertTrue(any('frontier' in m for m in dw.qualify_coverage(self.record(4, 16), 1024, 4)))
        self.assertTrue(any('sparse' in m for m in dw.qualify_coverage(self.record(4, 1024), 8192, 4)))
        self.assertTrue(any('file-tier' in m for m in dw.qualify_coverage(self.record(4, 1024, 0), 1024, 4)))
        self.assertTrue(dw.qualify_coverage(dict(prompt_tokens=1024), 1024, 4))

    def test_bench_requires_digests_labels_counts(self):
        order = ['cold', 'warm']
        self.assertEqual(dw.judge_bench(self.bench(), order, 4, 1024), ('EXACT', [], []))
        self.assertEqual(dw.judge_bench(self.bench(order=('warm', 'cold')), order, 4, 1024)[0], 'NO-REFERENCE')
        self.assertEqual(dw.judge_bench(self.bench(steps=3), order, 4, 1024)[0], 'NO-REFERENCE')
        for mutate, why in ((lambda b: self.rebind(b['arms'][1], logits=['c'*64, 'c'*64, '0'*64, 'c'*64]), 'logits'),
                            (lambda b: b['arms'][1]['final_state'].update(rank1_sha256='0'*64), 'final state'),
                            (lambda b: b['arms'][0]['final_state'].update(replay_sha256='0'*64), 'replay'),
                            (lambda b: self.rebind(b['arms'][1], tokens=[5, 5, 5, 6]), 'tokens')):
            b = self.bench()
            b['arms'] = [dict(x, logits_sha256=list(x['logits_sha256']), final_state=dict(x['final_state'])) for x in b['arms']]
            mutate(b)
            v, problems, _ = dw.judge_bench(b, order, 4, 1024)
            self.assertEqual(v, 'MISMATCH')
            self.assertTrue(any(why in p for p in problems), (why, problems))
        b = self.bench()
        del b['arms'][0]['final_state']
        self.assertEqual(dw.judge_bench(b, order, 4, 1024)[0], 'NO-REFERENCE')
        # every arm's complete final state signature, frontier from the REQUEST
        for field in ('frontier', 'rank0_bytes', 'rank1_bytes', 'replay_sha256', 'components'):
            for arm in (0, 1):
                b = self.bench()
                b['arms'] = [dict(x, final_state=dict(x['final_state'])) for x in b['arms']]
                del b['arms'][arm]['final_state'][field]
                self.assertEqual(dw.judge_bench(b, order, 4, 1024)[0], 'NO-REFERENCE', (field, arm))
        wrong = self.bench()
        wrong['arms'] = [dict(x, final_state=dict(x['final_state'], frontier=1026)) for x in wrong['arms']]  # 1024 + 4 - 1 = 1027
        self.assertEqual(dw.judge_bench(wrong, order, 4, 1024)[0], 'NO-REFERENCE')
        self.assertEqual(dw.judge_bench(dict(self.bench(), prompt_tokens=None), order, 4, 1024)[0], 'NO-REFERENCE')
        # an 8192 case must not accept a 1024-token response (frontier from the plan)
        self.assertEqual(dw.judge_bench(self.bench(n=1024), order, 4, 8192)[0], 'NO-REFERENCE')
        self.assertEqual(dw.judge_bench(self.bench(n=8192), order, 4, 8192)[0], 'EXACT')
        self.assertEqual(dw.judge_bench(self.bench(n=8192), order, 4, 1024)[0], 'NO-REFERENCE')
        bytes_differ = self.bench()
        bytes_differ['arms'] = [dict(x, final_state=dict(x['final_state'], rank1_bytes=1 + i)) for i, x in enumerate(bytes_differ['arms'])]
        self.assertEqual(dw.judge_bench(bytes_differ, order, 4, 1024)[0], 'MISMATCH')
        # comparators never raise: an arm without a validated signature is NO-REFERENCE with a reason
        bad = [dict(op='bench', tokens=1024, sweep_tile=t, verdict='EXACT', arms=[dict(arm='cold', tokens_per_s=1.0)]) for t in (2048, 8192)]
        out = dw.sweep_compare(bad)
        self.assertEqual(out[0]['verdict'], 'NO-REFERENCE')
        self.assertIn('reason', out[0])
        ttft_bad = [dict(op='ttft', tokens=8192, sweep_tile=t, verdict='EXACT', token=5, prefill_state=None, row_tile=t) for t in (2048, 8192)]
        self.assertEqual(dw.sweep_compare(ttft_bad)[0]['verdict'], 'NO-REFERENCE')
        self.assertEqual(dw.tile_compare([{k: v for k, v in r.items() if k != 'sweep_tile'} for r in ttft_bad])[0]['verdict'],
                         'NO-REFERENCE')
        self.assertEqual(dw.tile_compare(ttft_bad), [])  # no state at all: not a tile arm
        self.assertEqual(dw.judge_bench(self.bench(file_bytes=0), order, 4, 1024)[2], ['cold: no file-tier expert reads', 'warm: no file-tier expert reads'])

    def prompts(self, td, chat=CHAT_ALL):
        d = Path(td)/'prompts'
        d.mkdir()
        for n in dw.QUALIFY:
            (d/f'q{n}.u32').write_bytes(struct.pack(f'<{n}I', *([7]*n)))
            for k in dw.CHAT_KINDS[:chat]:
                (d/f'chat-{k}-{n}.u32').write_bytes(struct.pack(f'<{n}I', *([9]*n)))
        return d

    def test_inputs_exact_lengths_and_interleaved_order(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            a = self.cached(self.args(self.prompts(td, chat=2)), td)
            pins = dw.check_inputs(a)
            self.assertEqual(sum(1 for p in pins.values() if not p.get('missing')), 5)  # q1024, q8192, 2 chat prompts + unit file
            self.assertEqual(sum(1 for p in pins.values() if p.get('missing')), 0)
            orders = [i[3] for i in dw.prompt_plan(a) if i[0] == 'bench']
            self.assertEqual(orders, [['cold', 'warm'], ['cold', 'warm']])
            for bad in (struct.pack('<2I', 1, 2), struct.pack('<1023I', *([1]*1023)), struct.pack('<1025I', *([1]*1025)),
                        struct.pack('<1024I', *([154880]*1024))):
                (a.prompt_dir/'q1024.u32').write_bytes(bad)
                with self.assertRaises(ValueError):
                    dw.check_inputs(a)
            (a.prompt_dir/'chat-explanation-8192.u32').write_bytes(struct.pack('<16I', *range(16)))
            (a.prompt_dir/'q1024.u32').write_bytes(struct.pack('<1024I', *([1]*1024)))
            with self.assertRaises(ValueError):
                dw.check_inputs(a)  # undersized chat prompt is rejected, not silently accepted
            (a.prompt_dir/'q1024.u32').unlink()
            with self.assertRaises(ValueError):
                dw.check_inputs(a)

    def cached(self, a, td):
        """The cached tierstream product path (the only backend timed arms run on)."""
        a.expert_tier, a.tier_ram_gib, a.unit_manifest_sha256 = 'tierstream', 32.0, 'f'*64
        a.unit_file = Path(td)/'units.bin'
        a.unit_file.write_bytes(b'u')
        Path(str(a.unit_file)+'.units.json').write_text('{}')
        return a

    def run_execute(self, td, replies, left, chat=CHAT_ALL):
        a = self.cached(self.args(self.prompts(td, chat)), td)
        out = Path(td)/'out'
        out.mkdir()
        ready = dict(ready=True, producer_executable_sha256='e'*64,
                     health=dict(decode_identity=dw.IDENTITY, row_tile=2048, residency=dict(decode=dict(expert_tier='tierstream', tier_plan=dict(
                         three_tier=True, reference_compare=True, unit_manifest_sha256='f'*64)))))
        sent = []

        class FakeService:
            def __init__(self, *_):
                self.proc = types.SimpleNamespace(poll=lambda: 0)
                self.replies = [ready] + list(replies)

            def read(self, deadline):
                return self.replies.pop(0)

            def send(self, command):
                sent.append(command)

        deadline = types.SimpleNamespace(left=lambda: left)
        with patch.object(dw, 'Service', FakeService), patch.object(dw, 'stop'), patch.object(dw, 'snapshot_check'), \
             contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = dw.execute(a, out, deadline, dict(service_sha256='e'*64), dw.check_inputs(a))
        return rc, json.loads(printed.getvalue()), sent

    def full_replies(self, chat=CHAT_ALL):
        replies = [self.record(4, 1024), self.record(4, 8192), self.compare_record()]
        replies += [dict(backend='tierstream', prompt_tokens=n, ttft_s=1.5, prefill_s=1.4, token=5, row_tile=2048, allocated_row_tile=2048,
                         prefill_state=self.prefill_state(n))
                    for n in dw.QUALIFY]
        if chat:
            replies += [self.bench(order=('cold', 'warm'), n=n) for n in dw.QUALIFY]
        return replies

    def test_complete_workload_is_exact(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, sent = self.run_execute(td, self.full_replies(), left=100000)
            self.assertEqual((rc, summary['verdict'], summary['completion']), (0, 'EXACT', 'COMPLETE'))
            self.assertEqual(len(sent), 2 + 1 + 2 + 2 + 1)
            compare = [r for r in summary['results'] if r['op'] == 'compare']
            self.assertEqual([(r['prompt'], r.get('timed')) for r in compare], [('q1024.u32', False)])
            self.assertTrue(all('wall_s' not in r for r in compare))

    def test_missing_chat_or_coverage_is_incomplete(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, sent = self.run_execute(td, self.full_replies(chat=0), left=100000, chat=0)
            self.assertEqual((rc, summary['verdict'], summary['exactness_of_completed']), (4, 'INCOMPLETE', 'EXACT'))
            self.assertEqual(sum('planned prompt missing' in g for g in summary['gaps']), 2)
        replies = self.full_replies()
        replies[1] = self.record(4, 8192, file_bytes=0)
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, _ = self.run_execute(td, replies, left=100000)
            self.assertEqual((rc, summary['verdict'], summary['exactness_of_completed']), (4, 'INCOMPLETE', 'EXACT'))
            self.assertTrue(any('file-tier' in g for g in summary['gaps']))

    def test_tierstream_command_health_and_cache_summary(self):
        a = self.args()
        a.expert_tier, a.unit_file, a.tier_ram_gib, a.unit_manifest_sha256 = 'tierstream', Path('OUT_DIR/glm-units.bin'), 32.0, 'f'*64
        cmd = dw.service_command(a, Path('svc.exe'))
        self.assertEqual(cmd[cmd.index('--expert-tier')+1:cmd.index('--expert-tier')+7],
                         ['tierstream', '--unit-file', str(a.unit_file), '--unit-manifest-sha256', 'f'*64, '--reference-compare'])
        self.assertEqual(cmd[cmd.index('--tier-ram-gib')+1], '32.0')
        # Tierstream mode: prefill keeps no resident/pinned expert copies (all layers file tier).
        self.assertEqual((cmd[cmd.index('--max-device-layers')+1], cmd[cmd.index('--max-pinned-layers')+1]), ('0', '0'))
        self.assertIn('--require-file-tier', cmd)
        ref = self.args()
        rcmd = dw.service_command(ref, Path('svc.exe'))
        self.assertEqual(rcmd[rcmd.index('--max-device-layers')+1], '19')
        # Dry plan: same options, --plan-budget instead of --allow-gpu, device figures passed through.
        a.tier_vram_gib, a.device_total_gib, a.device_free_gib = 24.0, 95.5, 95.4
        seen = {}
        def fake_run(cmd, **kw):
            seen['cmd'] = cmd
            return types.SimpleNamespace(stdout='{"verdict":"ADMITTED"}\n', stderr='', returncode=0)
        real = dw.subprocess.run
        dw.subprocess.run = fake_run
        try:
            self.assertEqual(dw.plan_budget(a), 0)
        finally:
            dw.subprocess.run = real
        self.assertEqual(seen['cmd'][-5:], ['--plan-budget', '--device-total-gib', '95.5', '--device-free-gib', '95.4'])
        self.assertNotIn('--allow-gpu', seen['cmd'])
        self.assertEqual(seen['cmd'][seen['cmd'].index('--tier-vram-gib')+1], '24.0')
        with self.assertRaises(ValueError):
            dw.plan_budget(ref)  # the reference backend has no tierstream budget
        granted = dict(residency=dict(device_layers=0, pinned_layers=0, decode=dict(expert_tier='tierstream', tier_plan=dict(
            three_tier=True, reference_compare=True, unit_manifest_sha256='f'*64, device_budget_bytes=24 << 30, host_budget_bytes=32 << 30))))
        self.assertEqual(dw.health_coverage(granted, a), [])
        granted['residency']['device_layers'] = 19
        granted['residency']['decode']['tier_plan']['device_budget_bytes'] = 4 << 30
        gaps = dw.health_coverage(granted, a)
        self.assertTrue(any('resident/pinned expert copies' in g for g in gaps) and any('device_budget_bytes' in g for g in gaps))
        a.tier_vram_gib = None
        # Whole-device caps: passed through; the arena-only RAM default is not applied; receipts gated.
        whole = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--vram-total-gib', '24', '--ram-total-gib', '32'])
        self.assertEqual((whole.vram_total_gib, whole.ram_total_gib, whole.tier_ram_gib), (24.0, 32.0, None))
        self.assertEqual(whole.capacity, dw.WHOLE_CARD_CAPACITY)  # 24 GiB card: plan at the 8K capacity it needs
        whole.prompt_dir = Path('P')
        self.assertEqual([i[0] for i in dw.prompt_plan(whole)], ['qualify', 'qualify', 'compare', 'ttft', 'ttft', 'bench', 'bench'])
        self.assertLessEqual(dw.expected_seconds(whole), dw.MAX_SECONDS)  # startup prior included
        for bad in (['--capacity', '8000'], ['--capacity', '8321']):
            with self.assertRaises(SystemExit):
                dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P'] + bad)
        whole.unit_file, whole.unit_manifest_sha256 = Path('OUT_DIR/glm-units.bin'), 'f'*64
        wcmd = dw.service_command(whole, Path('svc.exe'))
        self.assertEqual((wcmd[wcmd.index('--vram-total-gib')+1], wcmd[wcmd.index('--ram-total-gib')+1]), ('24.0', '32.0'))
        self.assertEqual(wcmd[wcmd.index('--capacity')+1], str(dw.WHOLE_CARD_CAPACITY))
        self.assertNotIn('--tier-ram-gib', wcmd)
        with self.assertRaises(SystemExit):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--expert-tier', 'reference', '--vram-total-gib', '24'])
        peaks = {}
        ok = dict(memory=dict(measured_device_peak_bytes=23 << 30, host_peak_commit_bytes=30 << 30, within_caps=True))
        self.assertEqual(dw.memory_gate(ok, whole, peaks), [])
        over = dict(memory=dict(measured_device_peak_bytes=(24 << 30) + 1, host_peak_commit_bytes=33 << 30, within_caps=False))
        gaps = dw.memory_gate(over, whole, peaks)
        self.assertTrue(any('--vram-total-gib' in g for g in gaps) and any('--ram-total-gib' in g for g in gaps))
        self.assertEqual(peaks['measured_device_peak_bytes'], (24 << 30) + 1)
        self.assertEqual(dw.memory_gate({}, whole, {}), ['receipt has no memory report'])
        self.assertEqual(dw.memory_gate({}, a, {}), [])  # no caps: no memory gate
        # In whole-device mode the arena is what the caps leave: no exact-arena gap.
        whole.tier_vram_gib = 4.0
        wh = dict(residency=dict(device_layers=0, pinned_layers=0, decode=dict(expert_tier='tierstream', tier_plan=dict(
            three_tier=True, reference_compare=True, unit_manifest_sha256='f'*64, device_budget_bytes=3 << 30, host_budget_bytes=20 << 30))))
        self.assertEqual(dw.health_coverage(wh, whole), [])
        health = lambda three, tier='tierstream', compare=True, pin='f'*64: dict(residency=dict(decode=dict(
            expert_tier=tier, tier_plan=dict(three_tier=three, reference_compare=compare, unit_manifest_sha256=pin))))
        self.assertEqual(dw.health_coverage(health(True), a), [])
        self.assertTrue(dw.health_coverage(health(False), a))
        self.assertTrue(any('reference backend' in g for g in dw.health_coverage(health(True, compare=False), a)))
        for bad in (health(True, 'reference'), health(True, pin='0'*64)):
            with self.assertRaises(ValueError):
                dw.health_coverage(bad, a)
        arm = dict(expert_totals=dict(lookups=1376, misses=120, device_hits=1256, resurrected=8, covered=12, ram_copies=60,
                                      file_reads=40, file_bytes=40*4096, h2d_bytes=100*4096))
        c = dw.cache_summary(arm, 4)
        self.assertEqual((c['misses_per_token'], c['device_hits'], c['resurrected'], c['covered'], c['ram_copies'], c['file_reads'],
                          c['file_bytes_per_token']), (30.0, 1256, 8, 12, 60, 40, 10*4096))
        self.assertNotIn('pinned_hits', c)
        plan = [i[:3] for i in dw.prompt_plan(a)]
        self.assertEqual([i[0] for i in plan], ['qualify']*2 + ['compare', 'ttft', 'ttft', 'bench', 'bench'])
        self.assertEqual({(i[1], i[2].name) for i in plan if i[0] == 'compare'}, {(1024, 'q1024.u32')})
        self.assertIn('compare q1024.u32', dw.plan(a)['run_order'])
        self.assertTrue(any('2 planned chat case(s)' in c for c in dw.plan(a)['coverage']))  # coverage text from the actual schedule
        parsed = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P'])
        self.assertEqual((parsed.expert_tier, parsed.tier_ram_gib), ('tierstream', 32.0))
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            a.prompt_dir = self.prompts(td, chat=0)
            with self.assertRaisesRegex(ValueError, 'unit-file'):
                dw.check_inputs(a)  # unit file absent
            a.unit_file = Path(td)/'u.bin'
            a.unit_file.write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'unit-file'):
                dw.check_inputs(a)  # manifest absent
            Path(str(a.unit_file)+'.units.json').write_text('{}')
            a.unit_manifest_sha256 = None
            with self.assertRaises(ValueError):
                dw.check_inputs(a)  # unpinned manifest
            a.unit_manifest_sha256 = 'f'*64
            self.assertEqual(dw.check_inputs(a)['unit_file']['bytes'], 1)

    def compare_record(self, steps=4, n=1024):
        arm = lambda name: dict(arm=name, backend=name, tokens=[3]*steps, logits_sha256=['a'*64]*steps,
                                steps=[dict(token=3, logits_sha256='a'*64, position=n-1+j) for j in range(steps)],
                                final_state=dict(rank0_sha256='b'*64, rank1_sha256='b'*64, rank0_bytes=1, rank1_bytes=1,
                                                 replay_sha256='c'*64, frontier=n + steps - 1, components=[[0, 0, 1]]))
        return dict(prompt_tokens=n, arms=[arm('reference'), arm('tierstream')])

    @staticmethod
    def rebind(arm, tokens=None, logits=None):
        """Changes an arm's tokens/logits together with its step records (they are bound)."""
        if tokens is not None:
            arm['tokens'] = tokens
        if logits is not None:
            arm['logits_sha256'] = logits
        arm['steps'] = [dict(s, token=arm['tokens'][i], logits_sha256=arm['logits_sha256'][i]) for i, s in enumerate(arm['steps'])]

    def test_cache_summary_keeps_every_counter(self):
        totals = dict(lookups=1376, misses=120, device_hits=1256, resurrected=8, covered=12, fetched=70, prefetched=9,
                      bank_serves=3, ram_copies=39, file_reads=40, file_bytes=40*4096, file_read_ns=123456789,
                      h2d_bytes=79*4096, evicted=33, disk_drops=5, writebacks=21, stalled=2, big=2**63+1)
        self.assertEqual(len(set(totals.values())), len(totals), 'distinct nonzero inputs')
        c = dw.cache_summary(dict(expert_totals=totals), 4)
        for key, value in totals.items():
            self.assertEqual(c[key], value, key)
            self.assertEqual(c['per_token'][key], value / 4, key)
        self.assertEqual((c['misses_per_token'], c['file_bytes_per_token'], c['h2d_bytes_per_token']), (30.0, 40960.0, 79*1024.0))
        self.assertEqual(c['source_share'], dict(device=1256/1376, pinned=39/1376, file=40/1376))
        self.assertEqual(c['writeback_bytes_per_token'], 21 / 4 * dw.UNIT_BYTES)
        self.assertAlmostEqual(c['file_read_s_per_token'], 123456789 / 4 / 1e9)
        self.assertIsNone(c['file_gb_per_s_elapsed'])  # no loop wall time given: no elapsed rate
        self.assertIsNone(c['file_read_concurrency'])
        self.assertIsNone(dw.cache_summary(dict(expert_totals=totals), 4, 0)['file_gb_per_s_elapsed'])  # zero loop: none, no division
        # Concurrent reads: summed read time (8 s) exceeds the 2 s loop; the elapsed rate and concurrency come from the wall.
        q = dw.cache_summary(dict(expert_totals=dict(file_bytes=6 * 10**9, file_read_ns=8 * 10**9, file_wait_ns=10**9)), 4, 2.0)
        self.assertEqual((q['file_gb_per_s_elapsed'], q['file_read_concurrency'], q['file_wait_s_per_token']), (3.0, 4.0, 0.25))
        self.assertEqual(dw.cache_summary(dict(expert_totals={}), 0)['source_share'], {})
        self.assertIsNone(c['landing_used'])  # no landing reads: no share
        h = dw.cache_summary(dict(expert_totals=dict(landing_issued=8, landing_promoted=6, landing_wait_ns=4000000, landing_read_bytes=8 * dw.UNIT_BYTES)), 2)
        self.assertEqual((h['landing_used'], h['landing_wait_s_per_token'], h['landing_read_bytes_per_token']), (0.75, 0.002, 4 * dw.UNIT_BYTES))
        ref = dw.cache_summary(dict(expert_totals=dict(device_selections=10, pinned_selections=6, file_selections=2)), 2)
        self.assertEqual((ref['device_hits'], ref['ram_copies'], ref['file_reads'], ref['device_selections']), (10, 6, 2, 10))

    def test_service_receipt_through_cache_summary(self):
        if RECEIPT is None:
            self.skipTest('no --receipt (CTest passes the fixture artifact of producer_decode_host)')
        self.assertTrue(RECEIPT.is_file(), 'fixture setup did not publish the receipt artifact')
        arm = json.loads(RECEIPT.read_text())
        self.assertEqual(arm['published_by'], 'glm_native_decode_host')
        self.assertLess(abs(time.time() - arm['written_unix_s']), 3600, 'stale receipt artifact')
        self.assertTrue(arm['run_nonce'])
        c = dw.cache_summary(arm, len(arm['steps']))
        self.assertEqual(set(arm['expert_totals']) - set(c), set(), 'every aggregated counter survives')
        self.assertEqual((c['lookups'], c['misses'], c['misses_per_token'], c['device_hits'], c['resurrected'], c['covered'],
                          c['ram_copies'], c['file_reads'], c['fetched'], c['prefetched'], c['bank_serves']),
                         (688, 80, 40.0, 608, 7, 6, 50, 15, 55, 10, 2))
        self.assertEqual((c['file_bytes_per_token'], c['h2d_bytes_per_token']), (15*12648448/2, 65*12648448/2))

    def test_reference_comparison_judge(self):
        self.assertEqual(dw.judge_compare(self.compare_record(), 4, 1024), ('EXACT', []))
        for mutate, why in ((lambda r: self.rebind(r['arms'][1], tokens=[3, 3, 3, 4]), 'tokens'),
                            (lambda r: self.rebind(r['arms'][1], logits=['a'*64]*3+['0'*64]), 'logits'),
                            (lambda r: r['arms'][1]['final_state'].update(rank1_sha256='0'*64), 'final state'),
                            (lambda r: r['arms'][1]['final_state'].update(replay_sha256='0'*64), 'replay')):
            r = self.compare_record()
            mutate(r)
            v, problems = dw.judge_compare(r, 4, 1024)
            self.assertEqual(v, 'MISMATCH')
            self.assertTrue(any(why in q for q in problems), (why, problems))
        r = self.compare_record()
        r['arms'][1]['backend'] = 'reference'
        self.assertEqual(dw.judge_compare(r, 4, 1024)[0], 'NO-REFERENCE')
        self.assertEqual(dw.judge_compare(self.compare_record(3), 4, 1024)[0], 'NO-REFERENCE')
        self.assertEqual(dw.judge_compare(dict(arms=[]), 4, 1024)[0], 'NO-REFERENCE')

    def test_no_timed_reference_arm(self):
        # The reference backend runs only inside the exactness compare
        # (untimed, q1024 only); every timed arm is the cached tierstream path.
        a = self.args()
        with self.assertRaises(ValueError):
            dw.prompt_plan(a)  # no standalone reference-backend run at all
        a.expert_tier = 'tierstream'
        items = dw.prompt_plan(a)
        self.assertTrue(any(i[0] == 'bench' for i in items))
        self.assertEqual([(i[1], i[2].name) for i in items if i[0] == 'compare'], [(1024, 'q1024.u32')])
        self.assertTrue(all(i[0] in ('qualify', 'compare', 'ttft', 'bench') for i in items))
        self.assertEqual(dw.COMPARE, (('q', 1024),))
        with self.assertRaises(SystemExit):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--expert-tier', 'reference'])

    def test_full_card_mode(self):
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card'])
        self.assertEqual((a.vram_total_gib, a.ram_total_gib, a.tier_ram_gib, a.full_card, a.capacity), (93.0, 175.0, None, True, dw.CAPACITY))
        with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--capacity', '8320'])
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            given = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--ram-total-gib', '120'])
        self.assertEqual(given.ram_total_gib, 120.0)
        with self.assertRaises(SystemExit):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--expert-tier', 'reference'])
        items = dw.prompt_plan(a)
        self.assertEqual([i[0] for i in items], ['qualify', 'qualify', 'compare'] + ['ttft'] * 5 + ['bench'] * 2)
        self.assertLessEqual(dw.expected_seconds(a), dw.MAX_SECONDS)  # the full-card schedule fits the deadline (startup excluded)
        self.assertLessEqual(dw.FULL_CARD_VRAM_GIB * (1 << 30), 99.86e9)  # within the runtime's usable device
        # TTFT tile arms: 8K at 8192 vs 2048, 32K at the allocated (largest admitted) tile vs 2048.
        self.assertEqual([(i[1], i[2].name, i[3]) for i in items if i[0] == 'ttft'],
                         [(1024, 'q1024.u32', None), (8192, 'q8192.u32', 8192), (8192, 'q8192.u32', 2048),
                          (32000, 'ttft-32000.u32', None), (32000, 'ttft-32000.u32', 2048)])
        self.assertEqual(dw.expected_seconds(a), 1780)
        self.assertEqual([(i[1], i[2].name, i[3]) for i in items if i[0] == 'bench'],
                         [(1024, 'chat-explanation-1024.u32', ['cold', 'warm']), (8192, 'chat-explanation-8192.u32', ['cold', 'warm'])])
        a.unit_file, a.unit_manifest_sha256 = Path('U'), 'f'*64
        cmd = dw.service_command(a, Path('svc.exe'))
        self.assertEqual((cmd[cmd.index('--vram-total-gib')+1], cmd[cmd.index('--ram-total-gib')+1]), ('93.0', '175.0'))
        ok = dict(backend='tierstream', prompt_tokens=32000, ttft_s=12.5, prefill_s=12.0, token=7, row_tile=4096, allocated_row_tile=4096,
                  prefill_state=self.prefill_state(32000))
        self.assertEqual(dw.judge_ttft(ok, 32000, 4096, 4096), ('EXACT', []))
        for bad in (dict(ok, backend='reference'), dict(ok, prompt_tokens=1024), dict(ok, ttft_s=0), dict(ok, token=None)):
            self.assertEqual(dw.judge_ttft(bad, 32000, 4096, 4096)[0], 'NO-REFERENCE')
        # The worker re-parses its arguments: --full-card must be forwarded.
        a.timeout, a.allow_gpu, a.check_inputs, a.build, a.prompt_dir, a.cache = 1800, False, True, Path('.'), Path('P'), None
        seen = {}

        class Job:
            def __call__(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def start(self, argv):
                seen['argv'] = argv
                return None, None

            def wait(self, *_):
                return 0
        with patch.object(dw.window, 'windows_job', return_value=Job()), patch.object(Path, 'mkdir'), patch.object(dw.os, 'name', 'nt'):
            dw.run(a)
        self.assertIn('--full-card', seen['argv'])

    def test_schedule_over_timeout_refused(self):
        a = self.args()
        a.expert_tier, a.unit_file, a.unit_manifest_sha256, a.tier_ram_gib = 'tierstream', Path('U'), 'f'*64, 32.0
        a.check_inputs, a.timeout = True, 600
        with self.assertRaisesRegex(ValueError, 'timeout'):
            dw.run(a)  # refused before any process or output directory

    def test_deadline_estimate_marks_incomplete(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, sent = self.run_execute(td, [], left=200)
            self.assertEqual(rc, 4)
            self.assertEqual(summary['verdict'], 'INCOMPLETE')
            self.assertEqual(sent, [dict(op='quit')])

    def test_compare_evidence_and_signatures(self):
        # 1. compare: planned n, reported length, complete final signatures at n + steps - 1
        c = self.compare_record()
        self.assertEqual(dw.judge_compare(c, 4, 1024), ('EXACT', []))
        for mutate in (lambda r: r.pop('prompt_tokens'), lambda r: r.update(prompt_tokens=16),
                       lambda r: r['arms'][0]['final_state'].pop('frontier'), lambda r: r['arms'][0]['final_state'].update(frontier=1026),
                       lambda r: r['arms'][1]['final_state'].pop('rank0_bytes'), lambda r: r['arms'][1]['final_state'].update(components=[])):
            r = self.compare_record()
            mutate(r)
            self.assertEqual(dw.judge_compare(r, 4, 1024)[0], 'NO-REFERENCE')
        r = self.compare_record()
        r['arms'][1]['final_state']['rank1_bytes'] = 2
        self.assertEqual(dw.judge_compare(r, 4, 1024)[0], 'MISMATCH')
        r = self.compare_record()
        r['arms'][1]['final_state']['rank0_bytes'] = 2  # != the sum of its component extents
        self.assertEqual(dw.judge_compare(r, 4, 1024)[0], 'NO-REFERENCE')
        # 2. qualify: prefill producer digests and the initial / every state signature validated
        for mutate in (lambda r: [r['legacy_prefill'].pop(k) for k in ('rank0_sha256', 'rank1_sha256')],
                       lambda r: [e.pop('rank0_sha256') for e in r['incremental_exports'][:1]],
                       lambda r: r['incremental_exports'][0].pop('rank0_bytes'),
                       lambda r: r['fresh'][1].pop('rank1_bytes'),
                       lambda r: [e.update(components=[]) for e in r['incremental_exports'] + r['fresh']],
                       lambda r: r['incremental_exports'][0].update(frontier=1024)):
            r = self.record()
            mutate(r)
            self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
        # 3. token IDs: ints in [0, vocab), each generated token bound to its step incl. index 0
        r = self.record()
        r['generated'] = [None] * 5
        r['incremental_steps'] = [dict(s, token=None) for s in r['incremental_steps']]
        self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
        r = self.record()
        r['incremental_steps'][0]['token'] = r['generated'][0] + 1
        self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
        for kind, rec, judge in (('bench', self.bench(), lambda x: dw.judge_bench(x, ['cold', 'warm'], 4, 1024)[0]),
                                 ('compare', self.compare_record(), lambda x: dw.judge_compare(x, 4, 1024)[0])):
            for bad in ([None] * 4, [154880] * 4, [-1] * 4, ['5'] * 4):
                x = json.loads(json.dumps(rec))
                for arm in x['arms']:
                    self.rebind(arm, tokens=list(bad))
                self.assertEqual(judge(x), 'NO-REFERENCE', (kind, bad))
        # 4. a materialization/execution exception still prints the summary (non-zero exit)
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            bench = self.bench()
            del bench['arms'][0]['tokens_per_s']
            replies = self.full_replies()[:5] + [bench] + self.full_replies()[6:]
            rc, summary, _ = self.run_execute(td, replies, left=100000)
            self.assertNotEqual(rc, 0)
            self.assertTrue(any(r['op'] == 'bench' and r['verdict'] == 'NO-REFERENCE' for r in summary['results']))
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, _ = self.run_execute(td, self.full_replies()[:3], left=100000)  # the service dies after the gate
            self.assertEqual(rc, 4)
            self.assertEqual(summary['verdict'], 'INCOMPLETE')
            self.assertIn('failure', summary)
            self.assertTrue(any(r['verdict'] == 'INCOMPLETE' and r.get('reason', '').startswith('not run') for r in summary['results']))
        # 5. tile: see TileCompareTests (reported tile == effective requested tile, allocation > 0)

    def test_uncompared_items_and_summary_failures(self):
        """Comparison items that could not be compared and a failing summarize() are structured
        run failures with a non-zero exit (finalize, called directly)."""
        def run_finalize(td, results, sweep=False, **patches):
            a = self.cached(self.args(self.prompts(td, CHAT_ALL)), td)
            if sweep:
                a.full_card, a.capacity, a.tile_sweep, a.sweep_note = True, 32768, [2048, 8192], ''
            out = Path(td)/'out'
            out.mkdir(exist_ok=True)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                rc = dw.finalize(a, out, results, [], {}, 1.0, [], [], set(), None, sweep)
            return rc, json.loads(printed.getvalue())
        # 1. EXACT TTFT arms without their stored evidence: tile_compare marks the item
        #    internal_error -> run failure, non-zero exit
        broken = [dict(op='ttft', tokens=8192, prompt='q8192.u32', verdict='EXACT', evidence=None) for _ in range(2)]
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary = run_finalize(td, broken)
        self.assertNotEqual(rc, 0)
        self.assertTrue(summary['tile_compare'][0]['internal_error'])
        self.assertTrue(any(e.startswith('tile_compare') for e in summary['failure']['summary_builders']))
        # the same for the sweep comparison
        sweep_broken = [dict(op='bench', tokens=1024, prompt='chat-explanation-1024.u32', verdict='EXACT', sweep_tile=t, evidence=None)
                        for t in (2048, 8192)]
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary = run_finalize(td, sweep_broken, sweep=True)
        self.assertNotEqual(rc, 0)
        self.assertTrue(any(e.startswith('tile_sweep') for e in summary['failure']['summary_builders']))
        # 2. summarize() raising: the same structured failure, verdict NO-REFERENCE, exit 3
        with tempfile.TemporaryDirectory(dir=BUILD) as td, patch.object(dw, 'summarize', side_effect=RuntimeError('summary broke')):
            rc, summary = run_finalize(td, [])
        self.assertEqual((rc, summary['verdict']), (3, 'NO-REFERENCE'))
        self.assertTrue(any('summary broke' in e for e in summary['failure']['summary_builders']))
        # a clean finalize with nothing to compare stays failure-free
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary = run_finalize(td, [])
        self.assertNotIn('failure', summary)

    def test_positions_and_final_state_evidence(self):
        # 1. positions are evidence: every arm step from index 0, every fresh next token
        for kind, rec, judge in (('bench', self.bench(), lambda x: dw.judge_bench(x, ['cold', 'warm'], 4, 1024)[0]),
                                 ('compare', self.compare_record(), lambda x: dw.judge_compare(x, 4, 1024)[0])):
            for j in (0, 3):
                for bad in (None, 1023 + j + 1, str(1023 + j)):
                    x = json.loads(json.dumps(rec))
                    x['arms'][1]['steps'][j]['position'] = bad
                    self.assertEqual(judge(x), 'NO-REFERENCE', (kind, j, bad))
            self.assertEqual(dw.evidence_arms(rec, 1024, 4, [a['arm'] for a in rec['arms']], kind)['arms'][0]['positions'],
                             [1023, 1024, 1025, 1026])
        for k in (0, 3):
            r = self.record()
            r['fresh'][k]['next']['position'] += 1
            self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
            r = self.record()
            del r['fresh'][k]['next']['position']
            self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
        self.assertEqual([f['next_position'] for f in dw.evidence_qualify(self.record(), 1024, 4)['fresh']], [1024, 1025, 1026, 1027])
        # 2. provenance / plan construction failures are inside the outer try: summary printed, non-zero
        with tempfile.TemporaryDirectory(dir=BUILD) as td, patch.object(dw, 'plan', side_effect=RuntimeError('plan broke')):
            rc, summary, sent = self.run_execute(td, self.full_replies(), left=100000)
        self.assertNotEqual(rc, 0)
        self.assertEqual(summary['failure']['error'], 'plan broke')
        self.assertEqual(sent, [])
        # 3. a failing summary builder is a run failure (never EXACT, never exit 0)
        with tempfile.TemporaryDirectory(dir=BUILD) as td, patch.object(dw, 'tile_compare', side_effect=RuntimeError('builder broke')):
            rc, summary, _ = self.run_execute(td, self.full_replies(), left=100000)
        self.assertEqual((rc, summary['verdict']), (4, 'INCOMPLETE'))
        self.assertTrue(any('builder broke' in e for e in summary['failure']['summary_builders']))
        with tempfile.TemporaryDirectory(dir=BUILD) as td, patch.object(dw, 'expected_seconds', side_effect=RuntimeError('meta broke')):
            rc, summary, _ = self.run_execute(td, self.full_replies(), left=100000)
        self.assertEqual(rc, 4)
        # 4. the service is stopped even when the results write fails
        real_write = Path.write_text

        def failing_write(path, *args, **kw):
            if path.name == 'results.json':
                raise OSError('disk full')
            return real_write(path, *args, **kw)
        with tempfile.TemporaryDirectory(dir=BUILD) as td, patch.object(Path, 'write_text', failing_write), \
                patch.object(dw, 'stop') as stopped:
            a = self.cached(self.args(self.prompts(td, CHAT_ALL)), td)
            out = Path(td)/'out'
            out.mkdir()
            replies = self.full_replies()
            ready = dict(ready=True, producer_executable_sha256='e'*64, health=dict(decode_identity=dw.IDENTITY, row_tile=2048,
                         residency=dict(decode=dict(expert_tier='tierstream', tier_plan=dict(three_tier=True, reference_compare=True,
                                                                                             unit_manifest_sha256='f'*64)))))

            class FakeService:
                def __init__(self, *_):
                    self.proc = types.SimpleNamespace(poll=lambda: 0)
                    self.replies = [ready] + list(replies)

                def read(self, deadline):
                    return self.replies.pop(0)

                def send(self, command):
                    pass
            with patch.object(dw, 'Service', FakeService), patch.object(dw, 'snapshot_check'), \
                    contextlib.redirect_stdout(io.StringIO()) as printed:
                rc = dw.execute(a, out, types.SimpleNamespace(left=lambda: 100000), dict(service_sha256='e'*64), dw.check_inputs(a))
            self.assertEqual(stopped.call_count, 1)
            self.assertNotEqual(rc, 0)
            self.assertIn('failure', json.loads(printed.getvalue()))

    def test_every_field_mutation_is_rejected(self):
        """Table-driven: every field of every record kind, deleted / null / wrong type /
        off by one. No mutation may leave an EXACT verdict (or, for TTFT, BITWISE-EQUAL
        against the unmutated record), except value changes of timings and cache
        counters, which are not exactness evidence."""
        timing = {'generation_loop_s', 'tokens_per_s', 'ttft_s', 'prefill_s'}

        def paths(x, path=()):
            yield path
            if isinstance(x, dict):
                for k, v in x.items():
                    yield from paths(v, path + (k,))
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    yield from paths(v, path + (i,))

        def mutations(v):
            out = [('null', None)]
            if isinstance(v, bool):
                out += [('type', 'x')]
            elif isinstance(v, int):
                out += [('type', str(v)), ('off-by-one', v + 1), ('off-by-one', v - 1)]
            elif isinstance(v, float):
                out += [('type', str(v)), ('off-by-one', v + 1)]
            elif isinstance(v, str):
                out += [('type', 7), ('off-by-one', ('b' if v[:1] != 'b' else 'c') + v[1:])]
            elif isinstance(v, list):
                out += [('type', {}), ('off-by-one', v[:-1])]
            elif isinstance(v, dict):
                out += [('type', [])]
            return out

        def mutated(rec, path, how, value):
            x = json.loads(json.dumps(rec))
            parent = x
            for k in path[:-1]:
                parent = parent[k]
            if how == 'delete':
                del parent[path[-1]]
            else:
                parent[path[-1]] = value
            return x

        tile = TileCompareTests()
        kinds = [
            ('qualify', self.record(), lambda x: dw.judge_qualify(x, 4, 1024)[0], None),
            ('compare', self.compare_record(), lambda x: dw.judge_compare(x, 4, 1024)[0], None),
            ('bench', self.bench(), lambda x: dw.judge_bench(x, ['cold', 'warm'], 4, 1024)[0], None),
            ('ttft', tile.rec(tile=4096), lambda x: dw.judge_ttft(x, 8192, 4096, 8192)[0],
             lambda x: dw.tile_compare([tile.result(tile=2048), dict(op='ttft', tokens=8192, verdict='EXACT',
                                                                      evidence=dw.evidence_ttft(x, 8192, 4096, 8192))])[0]['verdict']),
        ]
        checked = 0
        for kind, rec, judge, against in kinds:
            self.assertEqual(judge(rec), 'EXACT', kind)
            for path in paths(rec):
                if not path:
                    continue
                parent = rec
                for k in path[:-1]:
                    parent = parent[k]
                value = parent[path[-1]]
                tolerated = 'experts' in path or (path[-1] in timing)
                # The qualify initial state (t=0) has no second witness for rank 1's bytes, the
                # replay checkpoint or the component layer/index numbers (the prefill producer
                # reports rank hashes only): well-formedness is all that can be checked there.
                unreferenced = kind == 'qualify' and path[:2] == ('incremental_exports', 0) and (
                    path[2:] in (('rank1_bytes',), ('replay_sha256',)) or (len(path) == 5 and path[2] == 'components' and path[4] in (0, 1))
                ) or kind == 'qualify' and path == ('incremental_steps', 0, 'logits_sha256')  # step 0's logits: no fresh witness (fresh starts at t=1)
                for how, new in [('delete', None)] + mutations(value):
                    x = mutated(rec, path, how, new)
                    verdict = judge(x)
                    checked += 1
                    if verdict != 'EXACT':
                        continue
                    if against is not None and against(x) == 'DIFFERENT':
                        continue  # a valid record of a different state: detected by the comparison
                    if 'experts' in path or (tolerated and how == 'off-by-one'):
                        continue  # cache counters / a changed timing value: not exactness evidence
                    if unreferenced and how == 'off-by-one':
                        continue  # a different but well-formed value of a field nothing else witnesses
                    self.fail(f'{kind}: {how} at {path!r} still EXACT')
        self.assertGreater(checked, 1000)

    def test_summary_always_prints(self):
        """Mutated replies of every record kind through execute: a JSON summary is printed
        every time and a damaged run never exits 0."""
        base = self.full_replies()
        targets = [(0, 'qualify'), (2, 'compare'), (3, 'ttft'), (5, 'bench')]
        for index, kind in targets:
            for key in list(base[index]):
                for value in ('delete', None, [], 'x'):
                    replies = json.loads(json.dumps(base))
                    if value == 'delete':
                        del replies[index][key]
                    else:
                        replies[index][key] = value
                    with tempfile.TemporaryDirectory(dir=BUILD) as td:
                        rc, summary, _ = self.run_execute(td, replies, left=100000)
                    self.assertIn('verdict', summary, (kind, key, value))
                    if key not in ('ttft_s', 'prefill_s') or value != 'delete':
                        self.assertNotEqual((rc, summary['verdict']), (0, 'EXACT'), (kind, key, value))

    def test_tile_sweep_one_start_per_tile(self):
        """--tile-sweep: one service start per tile (--row-tile T), the gate on the first
        tile only, TTFT + bench on every tile, cross-tile digests in the summary."""
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            a = self.cached(self.args(self.prompts(td, CHAT_ALL)), td)
            a.full_card, a.capacity, a.tile_sweep, a.sweep_note = True, 32768, [2048, 8192], ''
            out = Path(td)/'out'
            out.mkdir()
            starts, sent = [], []
            gate = [self.record(4, 1024), self.record(4, 8192), self.compare_record()]
            test = self

            class FakeService:
                def __init__(self, command, _log):
                    tile = int(command[command.index('--row-tile') + 1])
                    test.assertEqual(command[command.index('--prefill-tile-crossover') + 1], '0')  # every request at the startup tile
                    starts.append(tile)
                    self.proc = types.SimpleNamespace(poll=lambda: 0)
                    ready = dict(ready=True, producer_executable_sha256='e'*64, health=dict(
                        decode_identity=dw.IDENTITY, row_tile=tile, startup_ms=50000 if len(starts) > 1 else 180000,
                        residency=dict(decode=dict(expert_tier='tierstream', tier_plan=dict(
                            three_tier=True, reference_compare=True, unit_manifest_sha256='f'*64)))))
                    state = dict(rank0_sha256='a'*64, rank1_sha256=('b' if tile == 2048 else 'c')*64, rank0_bytes=10, rank1_bytes=10,
                                 replay_sha256='d'*64, frontier=8191, components=[[0, 0, 10]])
                    per = [dict(backend='tierstream', prompt_tokens=8192, ttft_s=2.0, prefill_s=1.9, token=5, row_tile=tile, allocated_row_tile=tile,
                                prefill_state=state)]  # ttft-32000.u32 is not among the test prompts
                    per += [test.bench(order=('cold', 'warm'), n=n) for n in dw.QUALIFY]
                    self.replies = [ready] + (list(gate) if len(starts) == 1 else []) + per

                def read(self, deadline):
                    return self.replies.pop(0)

                def send(self, command):
                    sent.append((starts[-1], command['op']))

            deadline = types.SimpleNamespace(left=lambda: 100000)
            with patch.object(dw, 'Service', FakeService), patch.object(dw, 'stop'), patch.object(dw, 'snapshot_check'), \
                 contextlib.redirect_stdout(io.StringIO()) as printed:
                dw.execute(a, out, deadline, dict(service_sha256='e'*64), dw.check_inputs(a))
            summary = json.loads(printed.getvalue())
            self.assertEqual(starts, [2048, 8192])
            self.assertEqual(sent, [(2048, 'qualify'), (2048, 'qualify'), (2048, 'compare'), (2048, 'ttft'), (2048, 'bench'),
                                    (2048, 'bench'), (2048, 'quit'), (8192, 'ttft'), (8192, 'bench'), (8192, 'bench'), (8192, 'quit')])
            sweep = summary['tile_sweep']
            self.assertEqual([(s['tile'], s['allocated_row_tile'], s['startup_s']) for s in sweep['starts']], [(2048, 2048, 180.0), (8192, 8192, 50.0)])
            verdicts = {(c['op'], c['tokens']): c['verdict'] for c in sweep['comparison']}
            self.assertEqual(verdicts[('ttft', 8192)], 'DIFFERENT')       # rank1 state differs across tiles
            self.assertEqual(verdicts[('bench', 1024)], 'BITWISE-EQUAL')  # same tokens, logits, final state
            self.assertTrue((out/'tile-2048'/'health.json').is_file() and (out/'tile-8192'/'ttft-q8192.json').is_file())
            self.assertTrue(any(r['prompt'] == 'ttft-32000.u32' and r['verdict'] == 'INCOMPLETE' for r in summary['results']))

    def test_tile_requests_one_load(self):
        """--tile-requests 4096,8192: ONE service start at the default allocation, the gate, then
        TTFT 8K/16K/32K at every tile (row_tile per request) plus the auto arm (the service's own
        choice: rows < 16384 -> 4096); per-length bitwise compare and the crossover estimate."""
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            d = self.prompts(td, CHAT_ALL)
            (d/'ttft-16384.u32').write_bytes(struct.pack('<16384I', *([11]*16384)))  # 32K stays missing: INCOMPLETE
            a = self.cached(self.args(d), td)
            a.full_card, a.capacity, a.tile_requests = True, 32768, [4096, 8192]
            out = Path(td)/'out'
            out.mkdir()
            starts, sent = [], []
            gate = [self.record(4, 1024), self.record(4, 8192), self.compare_record()]
            auto = {8192: 4096, 16384: 4096}  # the service rule, written out: 8191 and 16383 rows are below 16384
            speed = {(8192, 4096): 1.0, (8192, 8192): 1.2, (16384, 4096): 2.4, (16384, 8192): 2.0}
            test = self

            class FakeService:
                def __init__(self, command, _log):
                    test.assertNotIn('--row-tile', command)  # the default allocation
                    test.assertNotIn('--prefill-tile-crossover', command)  # the profile value
                    starts.append(command)
                    self.proc = types.SimpleNamespace(poll=lambda: 0)
                    self.replies = [dict(ready=True, producer_executable_sha256='e'*64, health=dict(
                        decode_identity=dw.IDENTITY, row_tile=8192, prefill_tile_crossover=16384, startup_ms=180000,
                        residency=dict(decode=dict(expert_tier='tierstream', tier_plan=dict(
                            three_tier=True, reference_compare=True, unit_manifest_sha256='f'*64)))))] + list(gate)

                def read(self, deadline):
                    if self.replies:
                        return self.replies.pop(0)
                    cmd = sent[-1][1]
                    n = 16384 if cmd['prompt'].endswith('ttft-16384.u32') else 8192
                    tile = cmd.get('row_tile') or auto[n]
                    state = dict(rank0_sha256='a'*64, rank1_sha256='b'*64, rank0_bytes=10, rank1_bytes=10,
                                 replay_sha256='d'*64, frontier=n - 1, components=[[0, 0, 10]])
                    return dict(backend='tierstream', prompt_tokens=n, ttft_s=speed[(n, tile)] + 0.1, prefill_s=speed[(n, tile)],
                                token=5, row_tile=tile, allocated_row_tile=8192, prefill_tile_crossover=16384,
                                row_tile_choice='request override (row_tile)' if cmd.get('row_tile') else 'auto: rows below the crossover (small tile)',
                                prefill=dict(row_tile=tile), prefill_state=state)

                def send(self, command):
                    if command['op'] != 'quit':
                        test.assertNotIn('row_tile', command) if command['op'] != 'ttft' else None
                    sent.append((command['op'], command))

            deadline = types.SimpleNamespace(left=lambda: 100000)
            with patch.object(dw, 'Service', FakeService), patch.object(dw, 'stop'), patch.object(dw, 'snapshot_check'), \
                 contextlib.redirect_stdout(io.StringIO()) as printed:
                dw.execute(a, out, deadline, dict(service_sha256='e'*64), dw.check_inputs(a))
            summary = json.loads(printed.getvalue())
            self.assertEqual(len(starts), 1)
            ttft = [(c['prompt'].rsplit('/', 1)[-1].rsplit('\\', 1)[-1], c.get('row_tile')) for op, c in sent if op == 'ttft']
            self.assertEqual(ttft, [('q8192.u32', 4096), ('q8192.u32', 8192), ('q8192.u32', None),
                                    ('ttft-16384.u32', 4096), ('ttft-16384.u32', 8192), ('ttft-16384.u32', None)])
            res = [r for r in summary['results'] if r['op'] == 'ttft']
            self.assertTrue(all(r['verdict'] == 'EXACT' for r in res if r['prompt'] != 'ttft-32000.u32'), res)
            self.assertEqual([r['row_tile_expected'] for r in res if r['prompt'] != 'ttft-32000.u32'], [4096, 8192, 4096, 4096, 8192, 4096])
            self.assertEqual([r.get('prefill_row_tile') for r in res if r['prompt'] != 'ttft-32000.u32'], [4096, 8192, 4096, 4096, 8192, 4096])
            self.assertTrue(any(r['prompt'] == 'ttft-32000.u32' and r['verdict'] == 'INCOMPLETE' for r in res))
            tr = summary['tile_requests']
            self.assertEqual((tr['tiles'], tr['lengths'], tr['crossover_service']), ([4096, 8192], [8192, 16384, 32000], 16384))
            self.assertEqual({c['tokens']: c['verdict'] for c in tr['comparison']}, {8192: 'BITWISE-EQUAL', 16384: 'BITWISE-EQUAL'})
            est = tr['crossover_estimate']
            self.assertEqual([(r['tokens'], r['faster']) for r in est['per_length']], [(8192, 4096), (16384, 8192)])
            self.assertEqual(est['crossover_at_most_rows'], 16383)
            self.assertNotIn('failure', summary)

    def test_tile_requests_arguments_and_auto_tile(self):
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--tile-requests', '4096,8192'])
        self.assertEqual(a.tile_requests, [4096, 8192])
        self.assertLessEqual(dw.expected_seconds(a), dw.MAX_SECONDS)
        plan = [(i[0], i[1], i[3] if len(i) > 3 else None) for i in dw.prompt_plan(a)]
        self.assertEqual(plan[:3], [('qualify', 1024, None), ('qualify', 8192, None), ('compare', 1024, None)])
        self.assertEqual(plan[3:], [('ttft', n, t) for n in (8192, 16384, 32000) for t in (4096, 8192, None)])
        self.assertEqual(dw.prompt_plan(a)[6][2].name, 'ttft-16384.u32')
        self.assertNotIn('--prefill-tile-crossover', dw.service_command(a, 'svc'))
        self.assertEqual(dw.worker_options(a)[dw.worker_options(a).index('--tile-requests') + 1], '4096,8192')
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            b = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--prefill-tile-crossover', '12000'])
            s = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--tile-sweep', '2048,8192'])
        cmd = dw.service_command(b, 'svc')
        self.assertEqual(cmd[cmd.index('--prefill-tile-crossover') + 1], '12000')
        self.assertEqual(dw.worker_options(b)[dw.worker_options(b).index('--prefill-tile-crossover') + 1], '12000')
        self.assertNotIn('--weight-dirs', cmd)  # default: the primary only
        for bad in ('', 'D:/s;;C:/s', 'D:/s;d:\\S\\', ' ;D:/s'):
            with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6),                     contextlib.redirect_stderr(io.StringIO()):
                dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--weight-dirs', bad])
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            m = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card',
                              '--weight-dirs', r'C:\s;D:\s', '--drive-readers', '8'])
        for argv in (dw.service_command(m, 'svc'), dw.worker_options(m)):
            self.assertEqual(argv[argv.index('--weight-dirs') + 1], r'C:\s;D:\s')
            self.assertEqual(argv[argv.index('--drive-readers') + 1], '8')
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            c = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card',
                              '--routes-out', 'census.routes', '--routes-prefill', '1'])
        for argv in (dw.service_command(c, 'svc'), dw.worker_options(c)):
            self.assertEqual(argv[argv.index('--routes-prefill') + 1], '1')
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            q = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card',
                              '--tier-staging', '16', '--tier-disk-queue', '16', '--tier-clean-evict', '1', '--disk-prefetch', '1'])
        for argv in (dw.service_command(q, 'svc'), dw.worker_options(q)):
            for flag, value in (('--tier-staging', '16'), ('--tier-disk-queue', '16'), ('--tier-clean-evict', '1'), ('--disk-prefetch', '1')):
                self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertNotIn('--tier-disk-queue', dw.service_command(m, 'svc'))  # default: the service's queue 1
        for v in ('auto', '1'):
            with patch.object(dw, 'host_total_gib', return_value=191.6):
                one = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--tier-disk-queue', v])
            self.assertEqual(dw.service_command(one, 'svc')[dw.service_command(one, 'svc').index('--tier-disk-queue') + 1], v)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--tier-disk-queue', '0'])
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            land = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--tier-disk-queue', '8',
                                 '--tier-landing-blocks', '16', '--tier-landing-max-queued', '4', '--tier-landing-ttl', '0', '--tier-landing-chunk', 'whole'])
        for argv in (dw.service_command(land, 'svc'), dw.worker_options(land)):
            for flag, value in (('--tier-landing-blocks', '16'), ('--tier-landing-max-queued', '4'), ('--tier-landing-ttl', '0'), ('--tier-landing-chunk', 'whole')):
                self.assertEqual(argv[argv.index(flag) + 1], value)
        for argv in (dw.service_command(m, 'svc'), dw.worker_options(m)):
            self.assertFalse([f for f in argv if str(f).startswith('--tier-landing')])  # default: landing off, nothing sent
        base = ['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card']
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            for ok in (['--tier-landing-ttl', '0'], ['--tier-landing-ttl', '1024'], ['--tier-landing-chunk', '0'], ['--tier-landing-chunk', '4096'], ['--tier-landing-chunk', 'whole']):
                dw.arguments(base + ok)
            for bad in (['--tier-landing-ttl', '1025'], ['--tier-landing-chunk', '4097'], ['--tier-landing-chunk', '-4096'], ['--tier-landing-chunk', str(2**32)], ['--tier-landing-blocks', '4097']):
                with self.assertRaises(SystemExit, msg=bad), patch('sys.stderr'):
                    dw.arguments(base + bad)
        cmd = dw.service_command(s, 'svc')
        self.assertEqual(cmd[cmd.index('--prefill-tile-crossover') + 1], '0')  # per-start sweep: crossover off
        for bad in (['--tile-requests', '4096'], ['--tile-requests', '4096,4096'], ['--tile-requests', '4096,8200'],
                    ['--tile-requests', '4096,8192', '--tile-sweep', '2048,8192'], ['--prefill-tile-crossover', '-1'],
                    ['--prefill-tile-crossover', '200000']):
            with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6), contextlib.redirect_stderr(io.StringIO()):
                dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card'] + bad)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--tile-requests', '4096,8192'])  # full card only
        # the Python mirror of schedule.hpp select_prefill_tile
        self.assertEqual([dw.auto_tile(r, 8192, 16384) for r in (1023, 8191, 16383, 16384, 31999)], [4096, 4096, 4096, 8192, 8192])
        self.assertEqual([dw.auto_tile(1023, 2048, 16384), dw.auto_tile(1023, 8192, 0)], [2048, 8192])
        self.assertEqual(dw.crossover_estimate([], 8192)['note'], 'no comparable lengths')
        slow = [dict(tokens=32000, verdict='BITWISE-EQUAL', timing=[dict(row_tile=4096, prefill_s=9.0), dict(row_tile=8192, prefill_s=9.5)])]
        self.assertIsNone(dw.crossover_estimate(slow, 8192)['crossover_at_most_rows'])

    def test_dsa_short_skip_passthrough_and_decode_wiring(self):
        for value in ('0', '1'):
            a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--dsa-short-skip', value])
            a.expert_tier = 'tierstream'
            a.unit_file, a.unit_manifest_sha256 = Path('U'), 'a'*64
            for argv in (dw.service_command(a, Path('S')), dw.worker_options(a)):
                self.assertEqual(argv[argv.index('--dsa-short-skip')+1], value)
        self.assertIsNone(dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P']).dsa_short_skip)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--dsa-short-skip', '2'])
        # Host-only wiring check: no CUDA compilation. Both decode MTP sites carry the
        # option; ordinary prefill constructs calls without opting in.
        runtime = (ROOT/'src/glm/rtxpf/producer/runtime.cu').read_text()
        self.assertIn('v.short_skip=dopt.dsa_short_skip;', runtime)
        self.assertEqual(runtime.count('m1.dsa_short_skip=dopt.dsa_short_skip;'), 2)
        dsa = (ROOT/'src/glm/rtxpf/dsa.cu').read_text()
        self.assertIn('d.short_skip=v.dsa_short_skip;', dsa)
        self.assertIn('dsa_short_skip_eligible(v.short_skip,v.rows,v.first)', dsa)
        self.assertIn('!v.compare_tokens&&!v.compare2_tokens&&!v.diagnostics&&!v.diagnostics2', dsa)
        prefill = runtime[runtime.index('const bool prof=dopt.prefill_profile;'):runtime.index('} timeline,pf_timeline;')]
        self.assertNotIn('short_skip=', prefill)

    def test_service_parses_every_planned_command(self):
        """Every request shape the harness can send (each run mode's plan, built by decode_command
        exactly as execute does, with the per-request tile at several allocations, plus quit) goes through
        the REAL service binary's decode-command parser (--check-decode-commands, CPU only); malformed
        shapes are refused with the field named."""
        exe = BUILD / 'glm_native_service.exe'
        if not exe.is_file():
            self.skipTest('service binary not built')
        modes = [[], ['--full-card'], ['--full-card', '--tile-sweep', '2048,8192'], ['--full-card', '--tile-requests', '4096,8192'],
                 ['--full-card', '--row-tile', '8192', '--tile-requests', '2048,4096,8192'], ['--full-card', '--ttft-only'],
                 ['--ab-arm', '--vram-total-gib', '24', '--ram-total-gib', '32'], ['--full-card', '--ab-arm'],
                 ['--census', 'arithmetic,instruction', '--routes-out', 'R', '--routes-prefill', '1']]
        lines, expected = [], []
        seg = Path('SEG')
        for mode in modes:
            with patch.object(dw, 'host_total_gib', return_value=191.6):
                a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P'] + mode)
            for allocated in (2048, 4096, 8192):
                for health in (dict(prefill_tile_crossover=16384), dict(prefill_tile_crossover=0), {}):
                    for item in dw.prompt_plan(a):
                        command, tile, _ = dw.decode_command(a, item, allocated, health, seg)
                        lines.append(json.dumps(command))
                        expected.append(command)
        self.assertTrue(any('row_tile' in c for c in expected) and any('order' in c for c in expected)
                        and any('export_dir' in c for c in expected), 'every optional field is exercised')
        self.assertTrue({c.get('row_tile') for c in expected if c['op'] == 'ttft'} >= {None, 2048, 4096, 8192})
        # the in-process A/B fields (decode_row_kernels, decode timeline) on every work op
        for op in ('qualify', 'bench', 'compare', 'ttft'):
            base = next(c for c in expected if c['op'] == op)
            for extra in (dict(row_kernels='0'), dict(row_kernels='1'), dict(row_kernels='moe+router'), dict(row_kernels='hc+kda+dsa'),
                          dict(dsa_short_skip=0), dict(dsa_short_skip=1), dict(decode_profile=True), dict(decode_profile=False), dict(row_kernels='moe', decode_profile=True)):
                command = dict(base, **extra)
                lines.append(json.dumps(command))
                expected.append(command)
        lines.append(json.dumps(dict(op='quit')))
        expected.append(dict(op='quit'))
        bad = [(dict(op='ttft', prompt='p', steps=1, dsa_short_skip=True), 'dsa_short_skip must be 0 or 1'),
               (dict(op='ttft', prompt='p', steps=1, dsa_short_skip=2), 'dsa_short_skip must be 0 or 1'),
               (dict(op='ttft', prompt='p', steps=1, tile=4096), 'unsupported request field: tile'),
               (dict(op='qualify', prompt='p', steps=4, row_tile=4096), 'row_tile is not a field of op qualify'),
               (dict(op='ttft', prompt='p', steps=1, row_tile=100), 'row_tile must be a multiple of 32'),
               (dict(op='bench', prompt='p', steps=4), 'bench needs order'),
               (dict(op='ttft', prompt='p', steps=1, export_dir='x'), 'export_dir is not a field of op ttft'),
               (dict(op='warmup', prompt='p', steps=1), 'unknown op: warmup'),
               (dict(op='quit', steps=1), 'quit takes no fields'),
               (dict(op='quit', row_kernels='1'), 'row_kernels is not a field of op quit'),
               (dict(op='ttft', prompt='p', steps=1, row_kernels='moe+gemm'), 'subset of moe, hc, router, kda, dsa'),
               (dict(op='ttft', prompt='p', steps=1, row_kernels=1), 'row_kernels must be a non-empty string'),
               (dict(op='bench', prompt='p', steps=4, order=['cold', 'warm'], decode_profile=1), 'decode_profile must be true or false')]
        lines += [json.dumps(b) for b, _ in bad]
        env = dict(os.environ)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        done = subprocess.run([str(exe), '--check-decode-commands'], input='\n'.join(lines) + '\n', capture_output=True,
                              text=True, env=env, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        out = [json.loads(x) for x in done.stdout.splitlines() if x.strip()]
        self.assertEqual(len(out), len(lines))
        for command, got in zip(expected, out):
            self.assertTrue(got['ok'], (command, got))
            self.assertEqual(got['op'], command['op'])
            if command['op'] != 'quit':
                self.assertEqual(got['row_tile'], command.get('row_tile', 0))
                self.assertEqual(got['steps'], command['steps'])
                self.assertEqual(got['export_dir'], 'export_dir' in command)
                if 'row_kernels' in command:
                    self.assertEqual(got['row_kernels'], command['row_kernels'])
                if 'dsa_short_skip' in command:
                    self.assertEqual(got['dsa_short_skip'], command['dsa_short_skip'])
                if 'decode_profile' in command:
                    self.assertEqual(got['decode_profile'], command['decode_profile'])
        for (b, why), got in zip(bad, out[len(expected):]):
            self.assertFalse(got['ok'], b)
            self.assertIn(why, got['error'])

    def test_ttft_records_prefill_file_traffic(self):
        """The TTFT result carries the prefill's own file reads/bytes (service counter delta) and
        tile_compare's timing shows them per tile."""
        rec = TileCompareTests.rec(TileCompareTests(), tile=8192)
        rec.update(prefill_file_reads=511, prefill_file_bytes=511 * 4096, prefill_misses=600)
        a = types.SimpleNamespace(expert_tier='tierstream', steps=4)
        r = dw.materialize(a, 'ttft', 8192, ('ttft', 8192, Path('q8192.u32'), 8192), rec, dict(op='ttft', tokens=8192), Path('.'),
                           1.0, 8192, 8192)
        self.assertEqual((r['verdict'], r['prefill_file_reads'], r['prefill_file_bytes'], r['prefill_misses']),
                         ('EXACT', 511, 511 * 4096, 600))
        other = dw.materialize(a, 'ttft', 8192, ('ttft', 8192, Path('q8192.u32'), 2048), dict(rec, row_tile=2048, prefill_file_reads=0,
                               prefill_file_bytes=0), dict(op='ttft', tokens=8192), Path('.'), 1.0, 2048, 8192)
        cmp = dw.tile_compare([r, other])
        self.assertEqual({t['row_tile']: t['prefill_file_reads'] for t in cmp[0]['timing']}, {8192: 511, 2048: 0})

    def test_tile_requests_three_tiles_fit(self):
        """--tile-requests 2048,4096,8192 under a forced 8192 allocation: all three tiles at 8K/16K/32K plus auto, in the deadline."""
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--row-tile', '8192',
                              '--tile-requests', '2048,4096,8192'])
        self.assertEqual(a.tile_requests, [2048, 4096, 8192])
        self.assertLessEqual(dw.expected_seconds(a), a.timeout)
        ttft = [(i[1], i[3]) for i in dw.prompt_plan(a) if i[0] == 'ttft']
        self.assertEqual(ttft, [(n, t) for n in (8192, 16384, 32000) for t in (2048, 4096, 8192, None)])
        self.assertEqual([dw.decode_command(a, i, 8192, dict(prefill_tile_crossover=16384), Path('S'))[0].get('row_tile')
                          for i in dw.prompt_plan(a) if i[0] == 'ttft'][:4], [2048, 4096, 8192, None])
        with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6), contextlib.redirect_stderr(io.StringIO()):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--tile-requests', '2048,4096,8192',
                          '--timeout', '1200'])  # does not fit: refused up front

    def test_restore_and_spare_slot_flags_reach_the_service(self):
        """TTFT regression A/B: --restore-decode-residency and --tier-spare-slots pass through the
        harness to the service command and the worker, and the real service binary accepts them
        (--print-effective-config, CPU only); the two spare options are exclusive."""
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--row-tile', '8192',
                              '--tile-requests', '4096,8192', '--restore-decode-residency', '0', '--tier-spare-slots', '290'])
        cmd = dw.service_command(a, 'svc')
        self.assertEqual(cmd[cmd.index('--restore-decode-residency') + 1], '0')
        self.assertEqual(cmd[cmd.index('--tier-spare-slots') + 1], '290')
        w = dw.worker_options(a)
        self.assertEqual((w[w.index('--restore-decode-residency') + 1], w[w.index('--tier-spare-slots') + 1]), ('0', '290'))
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            plain = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card'])
        self.assertNotIn('--restore-decode-residency', dw.service_command(plain, 'svc'))
        self.assertNotIn('--tier-spare-slots', dw.service_command(plain, 'svc'))
        with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6), contextlib.redirect_stderr(io.StringIO()):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--tier-spare-slots', '290', '--tier-spare-percent', '5'])
        exe = BUILD / 'glm_native_service.exe'
        if not exe.is_file():
            self.skipTest('service binary not built')
        env = dict(os.environ)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        out = subprocess.run([str(exe), '--restore-decode-residency', '0', '--tier-spare-slots', '290', '--print-effective-config'],
                             capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        levers = json.loads(out.stdout.strip().splitlines()[-1])['levers']
        self.assertEqual(levers['restore_decode_residency']['effective'], 'off')
        self.assertIn('290 slots', levers['spare_share']['effective'])

    def test_prefill_overlap_flag_reaches_the_service(self):
        """--prefill-overlap 0|1 (prefill overlap, on in every profile) passes through the harness to the service
        command and the worker like --tier-clean-evict; omitted, nothing is sent (the service profile default, on); other
        values are refused; a TTFT arm carries the overlap decision, its GPU idle gap and host wait into the repeat comparison.
        Python only: no service binary is run (a binary older than the option would refuse it)."""
        for value in ('0', '1'):
            with patch.object(dw, 'host_total_gib', return_value=191.6):
                a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--prefill-overlap', value])
            for argv in (dw.service_command(a, 'svc'), dw.worker_options(a)):
                self.assertEqual(argv[argv.index('--prefill-overlap') + 1], value)
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            plain = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card'])
        self.assertNotIn('--prefill-overlap', dw.service_command(plain, 'svc'))
        self.assertNotIn('--prefill-overlap', dw.worker_options(plain))
        for bad in ('2', 'on', ''):
            with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6), contextlib.redirect_stderr(io.StringIO()):
                dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--prefill-overlap', bad])
        rec = TileCompareTests.rec(TileCompareTests(), tile=8192)
        rec.update(make_resident_ms_total=900.0, prefill_overlap='on', prefill_overlap_gpu_idle_ms_total=3.5, prefill_overlap_host_wait_ms_total=120.5,
                   prefill_overlap_gpu_idle_ms_per_layer=[0.0] * 43)
        arm = dw.materialize(types.SimpleNamespace(expert_tier='tierstream', steps=4), 'ttft', 8192,
                             ('ttft', 8192, Path('q8192.u32'), 'allocated'), rec, dict(op='ttft', tokens=8192, prompt='q8192.u32'),
                             Path('.'), 1.0, 8192, 8192)
        self.assertEqual((arm['prefill_overlap'], arm['prefill_overlap_gpu_idle_ms_total'], arm['prefill_overlap_host_wait_ms_total']), ('on', 3.5, 120.5))
        self.assertNotIn('prefill_overlap_gpu_idle_ms_per_layer', arm)  # totals only in the window results
        timing = dw.repeat_compare([arm, dict(arm)])[0]['timing'][0]
        self.assertEqual((timing['prefill_overlap'], timing['prefill_overlap_gpu_idle_ms_total'], timing['prefill_overlap_host_wait_ms_total'],
                          timing['make_resident_ms_total']), ('on', 3.5, 120.5, 900.0))

    def test_ttft_only_mode(self):
        """--ttft-only: one load, no gate (summary says so instead of reporting a gap), TTFT 8K and 32K at
        the allocated tile (sent explicitly), 2 repeats each, repeats compared bitwise, phases per arm."""
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--row-tile', '8192', '--ttft-only',
                              '--tier-spare-slots', '290'])
        self.assertEqual([(i[0], i[1], i[3]) for i in dw.prompt_plan(a)],
                         [('ttft', 8192, 'allocated')] * 2 + [('ttft', 32000, 'allocated')] * 2)
        self.assertLessEqual(dw.expected_seconds(a), 15 * 60)
        cmd, tile, extra = dw.decode_command(a, dw.prompt_plan(a)[0], 8192, dict(prefill_tile_crossover=16384), Path('S'))
        self.assertEqual((cmd.get('row_tile'), tile, extra['row_tile_requested']), (8192, 8192, 'allocated'))
        self.assertIn('--ttft-only', dw.worker_options(a))
        for bad in (['--ttft-only'], ['--full-card', '--ttft-only', '--tile-requests', '4096,8192']):
            with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6), contextlib.redirect_stderr(io.StringIO()):
                dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P'] + bad)
        rec = TileCompareTests.rec(TileCompareTests(), tile=8192)
        rec.update(prefill_cache=dict(evicted=0, writebacks=0), make_resident_ms_total=900.0,
                   decode_restore=dict(wall_ms=12.0, units_copied=0, displaced_units=0))
        arm = lambda r: dw.materialize(types.SimpleNamespace(expert_tier='tierstream', steps=4), 'ttft', 8192,
                                       ('ttft', 8192, Path('q8192.u32'), 'allocated'), r, dict(op='ttft', tokens=8192, prompt='q8192.u32'),
                                       Path('.'), 1.0, 8192, 8192)
        same = [arm(rec), arm(dict(rec))]
        self.assertEqual(same[0]['decode_restore']['units_copied'], 0)
        cmp = dw.repeat_compare(same)
        self.assertEqual((cmp[0]['tokens'], cmp[0]['arms'], cmp[0]['verdict']), (8192, 2, 'BITWISE-EQUAL'))
        self.assertEqual(cmp[0]['timing'][0]['make_resident_ms_total'], 900.0)
        other = arm(dict(rec, token=9))
        self.assertEqual(dw.repeat_compare([same[0], other])[0]['verdict'], 'DIFFERENT')
        summary = dw.summarize(same, gate=False)
        self.assertEqual((summary['verdict'], summary['completion']), ('EXACT', 'COMPLETE'))
        self.assertIn('skipped', summary['gate'])
        self.assertIn('qualification at 1024 not EXACT', dw.summarize(same)['gaps'])  # the gate stays required elsewhere

    def test_worker_receives_tile_and_sweep_options(self):
        """The parent forwards the startup tile, the fitted sweep, its note and the
        placement options to the worker (dry --check-inputs launch, argv captured)."""
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            build, prompts = Path(td)/'b', Path(td)/'p'
            build.mkdir()
            prompts.mkdir()
            with patch.object(dw, 'host_total_gib', return_value=191.6):
                a = dw.arguments(['--build', str(build), '--model', 'M', '--prompt-dir', str(prompts), '--full-card',
                                  '--tile-sweep', '2048,4096,8192', '--timeout', '1500', '--tier-policy', 'lru',
                                  '--routes-out', 'r.txt', '--check-inputs'])
            captured = {}

            class Job:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def start(self, cmd):
                    captured['cmd'] = cmd
                    return object(), None

                def wait(self, process, left):
                    return 0
            with patch.object(dw.window, 'windows_job', return_value=Job):
                self.assertEqual(dw.run(a), 0)
            worker = captured['cmd'][7:]
            with patch.object(dw, 'host_total_gib', return_value=191.6):
                b = dw.arguments(worker)
            self.assertEqual((b.tile_sweep, b.sweep_note, b.tier_policy, b.routes_out, b.full_card),
                             ([2048, 8192], a.sweep_note, 'lru', 'r.txt', True))
            self.assertIn('4096 arm dropped', b.sweep_note)
            with patch.object(dw, 'host_total_gib', return_value=191.6):
                c = dw.arguments(['--build', str(build), '--model', 'M', '--prompt-dir', str(prompts), '--full-card',
                                  '--row-tile', '8192', '--check-inputs'])
            self.assertEqual(dw.worker_options(c)[dw.worker_options(c).index('--row-tile') + 1], '8192')

    def test_service_disk_queue_policy_inspection(self):
        """Requested policy and deferred resolution are visible without model/device work."""
        exe = BUILD / 'glm_native_service.exe'
        if not exe.is_file():
            self.skipTest('service not built')
        env = {k: v for k, v in os.environ.items() if k != 'CUDA_VISIBLE_DEVICES'}
        for args, requested, source in (([], 'unset: regime', 'default: regime'),
                                        (['--tier-disk-queue', 'auto'], 'auto', 'explicit-cli'),
                                        (['--tier-disk-queue', '1'], '1', 'explicit-cli'),
                                        (['--tier-disk-queue', '8'], '8', 'explicit-cli')):
            with self.subTest(requested=requested):
                out = subprocess.run([str(exe), '--print-effective-config'] + args,
                                     capture_output=True, text=True, env=env, timeout=60)
                self.assertEqual(out.returncode, 0, out.stderr)
                policy = json.loads(out.stdout.strip().splitlines()[-1])['disk_queue_policy']
                self.assertEqual(policy, dict(requested=requested, source=source,
                                             resolution='pending plan-time resolution'))

    def test_service_sets_eager_module_loading(self):
        """The service sets CUDA_MODULE_LOADING=EAGER at the top of main, before any CUDA call
        (--print-effective-config exits before the first one and reports it); an explicit value is kept."""
        exe = BUILD / 'glm_native_service.exe'
        if not exe.is_file():
            self.skipTest('service not built')
        env = {k: v for k, v in os.environ.items() if k != 'CUDA_MODULE_LOADING'}
        out = subprocess.run([str(exe), '--print-effective-config'], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout.strip().splitlines()[-1])['cuda_module_loading'], 'EAGER (set by the service)')
        env['CUDA_MODULE_LOADING'] = 'LAZY'
        out = subprocess.run([str(exe), '--print-effective-config'], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(json.loads(out.stdout.strip().splitlines()[-1])['cuda_module_loading'], 'LAZY (environment)')

    def test_profile_switch_through_harness(self):
        """--profile / --allow-pending reach the service and the worker; --print-effective-config runs the service's
        print mode (no model read, no CUDA call) and relays its JSON; the summary records the ready event's profile."""
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--profile', 'fast', '--allow-pending'])
        a.unit_file, a.unit_manifest_sha256 = Path('U'), 'f'*64
        cmd = dw.service_command(a, Path('svc.exe'))
        self.assertEqual(cmd[cmd.index('--profile') + 1], 'fast')
        self.assertIn('--allow-pending', cmd)
        opts = dw.worker_options(a)
        self.assertEqual(opts[opts.index('--profile') + 1], 'fast')
        self.assertIn('--allow-pending', opts)
        self.assertNotIn('--decode-profile', cmd)
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            prof = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--decode-profile'])
        prof.unit_file, prof.unit_manifest_sha256 = Path('U'), 'f'*64
        self.assertIn('--decode-profile', dw.service_command(prof, Path('svc.exe')))
        self.assertIn('--decode-profile', dw.worker_options(prof))
        with patch.object(dw, 'host_total_gib', return_value=191.6), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--profile', 'turbo'])
        plain = dw.service_command(dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P']), Path('svc.exe'))
        self.assertNotIn('--profile', plain)  # no --profile: the service's default (= today)
        seen = {}

        def fake_run(cmd, **kw):
            seen['cmd'] = cmd
            return types.SimpleNamespace(stdout='{"profile":"fast","levers":{}}\n', stderr='', returncode=0)
        with patch.object(dw.subprocess, 'run', fake_run), contextlib.redirect_stdout(io.StringIO()) as printed:
            a.print_effective_config, a.allow_gpu = True, False
            self.assertEqual(dw.run(a), 0)
        self.assertEqual(seen['cmd'][-1], '--print-effective-config')
        self.assertNotIn('--allow-gpu', seen['cmd'])
        self.assertEqual(json.loads(printed.getvalue())['profile'], 'fast')
        # the summary records the first start's profile resolution (from the ready event)
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, _ = self.run_execute(td, self.full_replies(), left=100000)
        self.assertIn('profile', summary)

    def test_tile_sweep_arguments_and_estimate(self):
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--tile-sweep', '2048,4096,8192'])
        self.assertEqual((a.tile_sweep, a.sweep_note), ([2048, 4096, 8192], ''))
        self.assertLessEqual(dw.expected_seconds(a), dw.MAX_SECONDS)
        self.assertEqual([(t, [i[0] for i in items]) for t, items in dw.sweep_segments(a)],
                         [(2048, ['qualify', 'qualify', 'compare', 'ttft', 'ttft', 'bench', 'bench']),
                          (4096, ['ttft', 'ttft', 'bench', 'bench']), (8192, ['ttft', 'ttft', 'bench', 'bench'])])
        with patch.object(dw, 'host_total_gib', return_value=191.6):  # a shorter deadline drops the 4096 arm first
            short = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--tile-sweep', '2048,4096,8192',
                                  '--timeout', '1500'])
        self.assertEqual(short.tile_sweep, [2048, 8192])
        self.assertIn('4096 arm dropped', short.sweep_note)
        for bad in (['--tile-sweep', '2048,8192', '--timeout', '900'], ['--tile-sweep', '2048,2048'], ['--tile-sweep', '2048,8200'],
                    ['--tile-sweep', '2048,8192', '--row-tile', '4096']):
            with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6), contextlib.redirect_stderr(io.StringIO()):
                dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card'] + bad)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--tile-sweep', '2048,8192'])  # full card only

    def test_mismatch_stops_qualification(self):
        bad = self.record(4)
        bad['fresh'][0]['rank0_sha256'] = 'b'*64
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, sent = self.run_execute(td, [bad], left=100000)
            self.assertEqual((rc, summary['verdict']), (3, 'MISMATCH'))
            self.assertEqual([c['op'] for c in sent], ['qualify', 'quit'])
            self.assertIn('export_dir', sent[0])


census = load('glm_route_census', 'src/glm/rtxpf/producer/glm_route_census.py')


class TileCompareTests(unittest.TestCase):
    STATE = dict(rank0_sha256='a' * 64, rank1_sha256='b' * 64, rank0_bytes=10, rank1_bytes=10, replay_sha256='d' * 64,
                 frontier=8191, components=[[0, 0, 10]])

    def rec(self, n=8192, tile=2048, state=None, token=7):
        return dict(backend='tierstream', prompt_tokens=n, ttft_s=2.0, prefill_s=1.9, token=token, row_tile=tile,
                    allocated_row_tile=8192, prefill_state=dict(state or self.STATE, frontier=n - 1))

    def result(self, n=8192, tile=2048, **kw):
        """A judged TTFT result as execute materializes it (canonical evidence stored)."""
        r = self.rec(n, tile, **kw)
        v, problems = dw.judge_ttft(r, n, tile, 8192)
        return dict(op='ttft', tokens=n, verdict=v, problems=problems,
                    evidence=dw.evidence_ttft(r, n, tile, 8192) if v == 'EXACT' else None)

    def test_tile_compare(self):
        same = dw.tile_compare([self.result(tile=8192), self.result(tile=2048)])
        self.assertEqual([(c['tokens'], c['row_tiles'], c['verdict']) for c in same], [(8192, [8192, 2048], 'BITWISE-EQUAL')])
        for kw in (dict(state=dict(self.STATE, rank1_sha256='c' * 64)), dict(token=8), dict(state=dict(self.STATE, replay_sha256='e' * 64)),
                   dict(state=dict(self.STATE, components=[[0, 0, 11]], rank0_bytes=11)), dict(state=dict(self.STATE, rank1_bytes=11)),
                   dict(state=dict(self.STATE, components=[[1, 0, 10]]))):
            diff = dw.tile_compare([self.result(tile=8192), self.result(tile=2048, **kw)])
            self.assertEqual(diff[0]['verdict'], 'DIFFERENT', kw)
            self.assertIn('FAST-profile', diff[0]['meaning'])
        # one tile only, or an arm that is not EXACT: nothing to compare
        self.assertEqual(dw.tile_compare([self.result(tile=2048), self.result(tile=2048)]), [])
        self.assertEqual(dw.tile_compare([self.result(tile=8192), dict(self.result(tile=2048), verdict='INCOMPLETE')]), [])

    def test_ttft_requires_complete_state_evidence(self):
        good = self.rec()
        self.assertEqual(dw.judge_ttft(good, 8192, 2048, 8192), ('EXACT', []))
        for bad in ({}, dict(replay_sha256=None), dict(rank1_sha256='x'), dict(frontier=8192), dict(components=[]),
                    dict(rank0_bytes=0)):
            rec = dict(good, prefill_state=dict(good['prefill_state'], **bad)) if bad else dict(good, prefill_state=None)
            self.assertEqual(dw.judge_ttft(rec, 8192, 2048, 8192)[0], 'NO-REFERENCE', bad)
        # the reported tile is an int and EQUALS the effective requested tile; the allocation is a
        # positive multiple of 32 (0 / None / misaligned refused)
        for tile in ([2048], '2048', 2050, 0, None, 4096):
            self.assertEqual(dw.judge_ttft(dict(good, row_tile=tile), 8192, 2048, 8192)[0], 'NO-REFERENCE', tile)
        for alloc in (0, None, 8200, '8192', 1024):
            self.assertEqual(dw.judge_ttft(good, 8192, 2048, alloc)[0], 'NO-REFERENCE', alloc)
        self.assertEqual(dw.judge_ttft(dict(good, allocated_row_tile=4096), 8192, 2048, 8192)[0], 'NO-REFERENCE')
        # planned tiles 2048 / 8192 both REPORTING 2048 never compare as equal
        both_2048 = [self.result(tile=2048), dict(op='ttft', tokens=8192, **dict(zip(('verdict', 'problems'), dw.judge_ttft(
            self.rec(tile=2048), 8192, 8192, 8192))))]
        self.assertEqual(both_2048[1]['verdict'], 'NO-REFERENCE')
        self.assertEqual(dw.tile_compare(both_2048), [])
        # malformed results never kill the comparison
        weird = [dict(self.result(tile=8192), evidence=dict(kind='ttft')), self.result(tile=2048)]
        self.assertEqual(dw.tile_compare(weird)[0]['verdict'], 'NO-REFERENCE')
        self.assertEqual(dw.tile_compare([None, dict(op='ttft')])[0]['verdict'], 'NO-REFERENCE')
        self.assertEqual(dw.sweep_compare([None]), [])
        self.assertEqual(dw.sweep_compare([None, dict(op='bench', sweep_tile=2048)])[0]['verdict'], 'NO-REFERENCE')
        self.assertEqual(dw.judge_ttft(dict(good, prefill_state=None), 8192, 2048, 8192)[0], 'NO-REFERENCE')

    def test_ttft_tile_capped_by_allocation(self):
        self.assertEqual(dw.ttft_tile(8192, 4096)[0], 4096)
        self.assertIn('exceeds the startup allocation 4096', dw.ttft_tile(8192, 4096)[1])
        self.assertEqual(dw.ttft_tile(2048, 4096), (2048, 'requested 2048'))
        self.assertEqual(dw.ttft_tile(None, 4096)[0], None)
        self.assertIn('allocated tile 4096', dw.ttft_tile(None, 4096)[1])
        self.assertIn('auto', dw.ttft_tile(None, 4096)[1])
        # the auto arm is judged against the service's per-request rule: an 8K prompt on an 8192 allocation runs at 4096
        good = TileCompareTests.rec(TileCompareTests(), tile=4096)
        self.assertEqual(dw.judge_ttft(good, 8192, dw.auto_tile(8191, 8192, 16384), 8192)[0], 'EXACT')
        self.assertEqual(dw.judge_ttft(dict(good, row_tile=8192), 8192, dw.auto_tile(8191, 8192, 16384), 8192)[0], 'NO-REFERENCE')


class ArmModeTests(unittest.TestCase):
    """--ab-arm (one arm of a main-vs-best-stack A/B), --census (placement census load) and --compare-arms."""
    BASE = ['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--unit-file', 'U', '--unit-manifest-sha256', 'f' * 64]
    CENSUS = ['--census', 'arithmetic,extraction,instruction,summarization', '--routes-out', 'R', '--routes-prefill', '1']

    def parse(self, extra):
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            return dw.arguments(self.BASE + extra)

    def refused(self, extra, why=None):
        err = io.StringIO()
        with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6), contextlib.redirect_stderr(err):
            dw.arguments(extra if extra[:1] == ['--compare-arms'] else self.BASE + extra)
        if why:
            self.assertIn(why, err.getvalue())

    def test_bench_only_arm(self):
        b = self.parse(['--ab-arm', '--bench-only', '--full-card'])
        self.assertEqual([i[0] for i in dw.prompt_plan(b)], ['bench', 'bench'])
        self.assertIn('--bench-only', dw.worker_options(b))
        self.refused(['--bench-only'], '--bench-only needs --ab-arm')

    def test_ab_arm_plan_command_and_estimate(self):
        whole = self.parse(['--ab-arm', '--vram-total-gib', '24', '--ram-total-gib', '32'])
        self.assertEqual(whole.capacity, dw.WHOLE_CARD_CAPACITY)
        self.assertEqual([(i[0], i[1], i[2].name, i[3]) for i in dw.prompt_plan(whole)],
                         [('ttft', 1024, 'q1024.u32', 'allocated'), ('ttft', 8192, 'q8192.u32', 'allocated'),
                          ('bench', 1024, 'chat-explanation-1024.u32', ['cold', 'warm']),
                          ('bench', 8192, 'chat-explanation-8192.u32', ['cold', 'warm'])])
        full = self.parse(['--full-card', '--ab-arm'])
        self.assertEqual([(i[0], i[1], i[3]) for i in dw.prompt_plan(full) if i[0] == 'ttft'],
                         [('ttft', 1024, 'allocated'), ('ttft', 8192, 'allocated'), ('ttft', 32000, 'allocated')])
        self.assertEqual([i for i in dw.prompt_plan(full) if i[0] == 'bench'], [i for i in dw.prompt_plan(whole) if i[0] == 'bench'])
        default = self.parse(['--vram-total-gib', '24', '--ram-total-gib', '32'])
        for a in (whole, full):
            self.assertFalse({i[0] for i in dw.prompt_plan(a)} & {'qualify', 'compare'})
            cmd = dw.service_command(a, 'svc')
            self.assertNotIn('--reference-compare', cmd)
            self.assertIn('--unit-manifest-sha256', cmd)
            self.assertIn('--ab-arm', dw.worker_options(a))
            self.assertEqual(dw.plan(a)['run_mode'], 'ab-arm')
            self.assertLessEqual(dw.expected_seconds(a), 20 * 60)
            self.assertLess(dw.expected_seconds(a), dw.expected_seconds(default))
        self.assertEqual(dw.expected_seconds(whole), dw.STARTUP_PRIOR + 60 + 90 + 120 + 200 + dw.CLEANUP_MARGIN)
        self.assertIn('--reference-compare', dw.service_command(default, 'svc'))  # unchanged without the new flags
        self.assertNotIn('--ab-arm', dw.worker_options(default))
        self.assertNotIn('run_mode', dw.plan(default))
        for bad in (['--ab-arm', '--full-card', '--ttft-only'], ['--ab-arm', '--full-card', '--tile-requests', '4096,8192'],
                    ['--ab-arm', '--full-card', '--tile-sweep', '2048,8192'], ['--ab-arm'] + self.CENSUS):
            self.refused(bad, '--ab-arm excludes')

    def test_census_plan_and_refusals(self):
        a = self.parse(self.CENSUS + ['--vram-total-gib', '24', '--ram-total-gib', '32'])
        self.assertEqual(a.census, ['arithmetic', 'extraction', 'instruction', 'summarization'])
        self.assertEqual([(i[0], i[1], i[2].name, i[3]) for i in dw.prompt_plan(a)],
                         [('ttft', 8192, f'chat-{k}-8192.u32', 'allocated') for k in a.census])
        cmd = dw.service_command(a, 'svc')
        self.assertNotIn('--reference-compare', cmd)
        self.assertEqual((cmd[cmd.index('--routes-out') + 1], cmd[cmd.index('--routes-prefill') + 1]), ('R', '1'))
        w = dw.worker_options(a)
        self.assertEqual(w[w.index('--census') + 1], 'arithmetic,extraction,instruction,summarization')
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            again = dw.arguments(self.BASE + w)  # the worker re-parses the forwarded options
        self.assertEqual((again.census, dw.run_mode(again)), (a.census, 'census'))
        self.assertLessEqual(dw.expected_seconds(a), 15 * 60)
        self.refused(['--census', 'explanation', '--routes-out', 'R', '--routes-prefill', '1'], 'refused')
        self.refused(['--census', 'arithmetic,coding', '--routes-out', 'R', '--routes-prefill', '1'], 'refused')
        self.refused(['--census', 'arithmetic,poetry', '--routes-out', 'R', '--routes-prefill', '1'], 'unknown chat kind')
        self.refused(['--census', 'arithmetic,arithmetic', '--routes-out', 'R', '--routes-prefill', '1'], 'distinct')
        self.refused(['--census', ',', '--routes-out', 'R', '--routes-prefill', '1'], 'distinct')
        self.refused(['--census', 'arithmetic'], '--routes-out')
        self.refused(['--census', 'arithmetic', '--routes-out', 'R'], '--routes-prefill 1')
        self.refused(['--census', 'arithmetic', '--routes-out', 'R', '--routes-prefill', '0'], '--routes-prefill 1')
        self.refused(['--full-card', '--ttft-only'] + self.CENSUS, '--census excludes')
        self.refused(['--full-card', '--tile-requests', '4096,8192'] + self.CENSUS, '--census excludes')

    def execute(self, td, mode, replies, routes=None):
        """execute() through a fake service whose plan has NO reference tier (the A/B arm / census load)."""
        W = WindowTests()
        a = W.cached(W.args(W.prompts(td)), td)
        a.ab_arm, a.census, a.routes_out, a.routes_prefill = mode == 'ab-arm', (['arithmetic', 'extraction'] if mode == 'census' else None), \
            (str(Path(td) / 'routes.txt') if mode == 'census' else None), ('1' if mode == 'census' else None)
        for k in a.census or []:
            (a.prompt_dir / f'chat-{k}-8192.u32').write_bytes(struct.pack('<8192I', *([3] * 8192)))
        out = Path(td) / 'out'
        out.mkdir()
        ready = dict(ready=True, producer_executable_sha256='e' * 64,
                     health=dict(decode_identity=dw.IDENTITY, row_tile=2048, residency=dict(decode=dict(expert_tier='tierstream', tier_plan=dict(
                         three_tier=True, unit_manifest_sha256='f' * 64)))))
        sent = []

        class FakeService:
            def __init__(self, command, *_):
                sent.append(command)
                self.proc = types.SimpleNamespace(poll=lambda: 0)
                self.replies = [ready] + list(replies)

            def read(self, deadline):
                return self.replies.pop(0)

            def send(self, command):
                sent.append(command)
                if command.get('op') == 'quit' and routes is not None:
                    Path(a.routes_out).write_bytes(routes)

        with patch.object(dw, 'Service', FakeService), patch.object(dw, 'stop'), contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = dw.execute(a, out, types.SimpleNamespace(left=lambda: 100000), dict(service_sha256='e' * 64), dw.check_inputs(a))
        return rc, json.loads(printed.getvalue()), sent, out

    @staticmethod
    def ttft_reply(n, token=5):
        return dict(backend='tierstream', prompt_tokens=n, ttft_s=1.5, prefill_s=1.4, token=token, row_tile=2048, allocated_row_tile=2048,
                    prefill_state=WindowTests.prefill_state(n), prefill_file_bytes=4096)

    def test_ab_arm_execute_lists_every_digest(self):
        W = WindowTests()
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            replies = [self.ttft_reply(n) for n in dw.QUALIFY] + [W.bench(n=n) for n in dw.QUALIFY]
            rc, summary, sent, out = self.execute(td, 'ab-arm', replies)
            self.assertEqual((rc, summary['verdict'], summary['completion'], summary['mode']), (0, 'EXACT', 'COMPLETE', 'ab-arm'), summary['gaps'])
            self.assertNotIn('--reference-compare', sent[0])
            self.assertIn('A/B arm', summary['gate'])
            self.assertEqual([c['op'] for c in sent[1:]], ['ttft', 'ttft', 'bench', 'bench', 'quit'])
            self.assertEqual([c.get('row_tile') for c in sent[1:3]], [2048, 2048])  # the allocated tile, sent explicitly
            self.assertTrue(all(r['mode'] == 'ab-arm' for r in summary['results']))
            self.assertTrue(all(r['mode'] == 'ab-arm' for r in json.loads((out / 'results.json').read_text())))
            d = summary['arm_digests']
            self.assertEqual([(t['tokens'], t['token'], t['state']['rank0']) for t in d['ttft']], [(1024, 5, 'a' * 64), (8192, 5, 'a' * 64)])
            self.assertTrue(all(len(t['state_sha256']) == 64 for t in d['ttft']))
            arms = d['bench'][0]['arms']
            self.assertEqual([x['arm'] for x in arms], ['cold', 'warm'])
            self.assertEqual((arms[0]['tokens'], arms[0]['logits'], arms[0]['final']['rank0'], arms[0]['tokens_per_s']),
                             ([5] * 4, ['c' * 64] * 4, 'd' * 64, 4.0))
            self.assertEqual(arms[0]['cache']['file_bytes'] if 'file_bytes' in arms[0]['cache'] else 0, 0)
            self.assertIn('misses_per_token', arms[0]['cache'])
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(dw.compare_arms(out, out), 0)  # an arm against itself
            self.assertEqual(json.loads(printed.getvalue())['modes_a'], ['ab-arm'])

    def test_census_execute_reports_routes_file(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, sent, _ = self.execute(td, 'census', [self.ttft_reply(8192), self.ttft_reply(8192, token=6)], routes=b'# seq\n3:1,2\n3:4,5\n')
            self.assertEqual((rc, summary['verdict'], summary['mode']), (0, 'EXACT', 'census'), summary['gaps'])
            self.assertNotIn('--reference-compare', sent[0])
            self.assertEqual([(c['op'], Path(c['prompt']).name) for c in sent[1:3]],
                             [('ttft', 'chat-arithmetic-8192.u32'), ('ttft', 'chat-extraction-8192.u32')])
            c = summary['census']
            self.assertEqual((c['kinds'], c['routes_out']['exists'], c['routes_out']['bytes'], c['routes_out']['lines']),
                             (['arithmetic', 'extraction'], True, 18, 3))
            self.assertEqual([p['token'] for p in c['prefills']], [5, 6])
        with tempfile.TemporaryDirectory(dir=BUILD) as td:  # no routes file: reported, never a failure
            rc, summary, _, _ = self.execute(td, 'census', [self.ttft_reply(8192), self.ttft_reply(8192)])
            self.assertEqual((rc, summary['census']['routes_out']['exists']), (0, False))
            self.assertNotIn('failure', summary)
        self.assertEqual(dw.routes_file(None), dict(path=None, exists=False))

    # ---- --compare-arms on synthetic results.json ----
    @staticmethod
    def results(token=5, logits=None, ttft_token=7, rank0='d' * 64, tps=4.0, ttft_s=2.0, drop=(), mode='ab-arm'):
        state = lambda f: dict(rank0=rank0, rank1='b' * 64, bytes0=10, bytes1=10, replay='e' * 64, frontier=f, components=[[0, 0, 10]])
        out = []
        for n in dw.QUALIFY:
            out.append(dict(op='ttft', tokens=n, prompt=f'q{n}.u32', verdict='EXACT', mode=mode, ttft_s=ttft_s,
                            evidence=dict(kind='ttft', n=n, row_tile=2048, allocated=2048, token=ttft_token, ttft_s=ttft_s, prefill_s=ttft_s - 0.1,
                                          state=state(n - 1))))
        for n in dw.QUALIFY:
            arms = [dict(arm=arm, tokens=[token, 6, 7, 8], logits=list(logits or ['c' * 64] * 4), positions=[n - 1 + j for j in range(4)],
                         final=state(n + 3), generation_loop_s=1.0, tokens_per_s=tps) for arm in ('cold', 'warm')]
            out.append(dict(op='bench', tokens=n, prompt=f'chat-explanation-{n}.u32', verdict='EXACT', mode=mode,
                            evidence=dict(kind='bench', n=n, steps=4, arms=arms), arms=[dict(arm=x['arm'], tokens_per_s=tps) for x in arms]))
        return [r for r in out if (r['op'], r['tokens']) not in drop]

    def compare(self, a, b):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            dirs = []
            for name, results in (('A', a), ('B', b)):
                d = Path(td) / name
                d.mkdir()
                if results is not None:
                    (d / 'results.json').write_text(json.dumps(results, indent=2))
                dirs.append(d)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                rc = dw.run(dw.arguments(['--compare-arms', str(dirs[0]), str(dirs[1])]))  # no --build / --model / --prompt-dir
            return rc, json.loads(printed.getvalue())

    def test_compare_arms_exact(self):
        old = self.results(mode=None)
        for r in old:  # an OLDER version's results: no mode key, no optional timing keys
            r.pop('mode')
            r.pop('ttft_s', None)
            r['evidence'].pop('prefill_s', None)
            r.pop('arms', None)
        rc, d = self.compare(old, self.results(tps=6.0, ttft_s=1.0))
        self.assertEqual((rc, d['verdict'], d['bench_pairs'], d['ttft_pairs'], d['diffs'], d['missing']), (0, 'EXACT', 4, 2, [], []))
        bench = [p for p in d['pairs'] if p['item'] == 'bench chat-explanation-8192.u32 warm'][0]
        self.assertEqual((bench['tokens_per_s_a'], bench['tokens_per_s_b'], bench['ratio_b_over_a']), (4.0, 6.0, 1.5))
        ttft = [p for p in d['pairs'] if p['item'] == 'ttft q8192.u32#0'][0]
        self.assertEqual((ttft['ttft_s_a'], ttft['ttft_s_b'], ttft['ratio_b_over_a'], ttft['prefill_s_a']), (2.0, 1.0, 0.5, None))
        self.assertEqual((d['modes_a'], d['modes_b']), ([], ['ab-arm']))

    def test_compare_arms_diff(self):
        rc, d = self.compare(self.results(), self.results(logits=['c' * 64] * 3 + ['f' * 64]))
        self.assertEqual((rc, d['verdict']), (3, 'DIFF'))
        self.assertEqual(len(d['diffs']), 4)  # every bench item and arm, the last logits digest
        self.assertTrue(all(x.endswith('logits digest [3]') for x in d['diffs']))
        rc, d = self.compare(self.results(), self.results(token=9))
        self.assertEqual((rc, d['verdict']), (3, 'DIFF'))
        self.assertTrue(all(': tokens [5, 6, 7, 8] != [9, 6, 7, 8]' in x for x in d['diffs']) and len(d['diffs']) == 4)
        rc, d = self.compare(self.results(), self.results(rank0='0' * 64, ttft_token=8))
        self.assertEqual(rc, 3)
        self.assertIn('ttft q1024.u32#0: first token 7 != 8', d['diffs'])
        self.assertIn('ttft q1024.u32#0: prefill state rank0', d['diffs'])
        self.assertIn('bench chat-explanation-1024.u32 cold: final state rank0', d['diffs'])
        rc, d = self.compare(self.results(), self.results(token=9, drop={('bench', 8192)}))
        self.assertEqual((rc, d['verdict']), (3, 'DIFF'))  # a mismatch wins over a missing item

    def test_compare_arms_incomplete(self):
        rc, d = self.compare(self.results(), self.results(drop={('bench', 8192)}))
        self.assertEqual((rc, d['verdict'], d['diffs']), (4, 'INCOMPLETE', []))
        self.assertEqual(sorted(m['item'] for m in d['missing']),
                         ['bench chat-explanation-8192.u32 cold', 'bench chat-explanation-8192.u32 warm'])
        self.assertEqual(d['missing'][0]['b'], 'absent')
        broken = self.results()
        broken[-1] = dict(broken[-1], verdict='MISMATCH', evidence=None, arms=[])  # not EXACT on one side
        rc, d = self.compare(self.results(), broken)
        self.assertEqual((rc, d['verdict'], d['missing'][0]['b']), (4, 'INCOMPLETE', 'MISMATCH'))
        rc, d = self.compare(self.results(drop={('bench', 1024), ('bench', 8192)}),
                             self.results(drop={('bench', 1024), ('bench', 8192)}))
        self.assertEqual((rc, d['verdict'], d['bench_pairs'], d['ttft_pairs']), (4, 'INCOMPLETE', 0, 2))  # no bench pair compared
        rc, d = self.compare([], [])
        self.assertEqual((rc, d['verdict']), (4, 'INCOMPLETE'))
        rc, d = self.compare(self.results(), None)  # no results.json
        self.assertEqual((rc, d['verdict']), (4, 'INCOMPLETE'))
        self.assertTrue(d['errors'] and d['missing'])

    def test_compare_arms_arguments(self):
        a = dw.arguments(['--compare-arms', 'A', 'B'])
        self.assertEqual(a.compare_arms, [Path('A'), Path('B')])
        self.refused(['--compare-arms', 'A', 'B', '--ab-arm'], 'excludes')
        self.refused(['--compare-arms', 'A', 'B', '--allow-gpu'])
        self.refused(['--compare-arms', 'A'])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as err:
            dw.arguments(['--model', 'M'])
        self.assertIn('required: --build, --prompt-dir', err.getvalue())
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as err:
            dw.arguments(['--model', 'M', '--build', '.'])
        self.assertIn('required: --prompt-dir', err.getvalue())


class PlacementInputTests(unittest.TestCase):
    def test_route_census_from_receipts(self):
        line = ' '.join(f'{layer}:' + ','.join(str((layer * 7 + k) % 288) for k in range(8)) for layer in range(3, 46))
        rec = {'prompt_tokens': 8192, 'arms': [{'steps': [{'routes': line}, {'routes': line}, {'experts': {}}]}],
               'other': {'steps': [{'token': 1}]}}
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td)/'bench.json', Path(td)/'routes.txt'
            src.write_text(json.dumps(rec), encoding='utf-8')
            self.assertEqual(census.main([str(src), '-o', str(out)]), 0)
            text = out.read_text(encoding='utf-8').splitlines()
            self.assertEqual(text, ['# seq bench.json:/arms[0] rows 0', line, line])
            src.write_text(json.dumps({'steps': [{'routes': '3:1,2 4:x'}]}), encoding='utf-8')
            with self.assertRaises(ValueError):
                census.census([src])
            src.write_text(json.dumps({'steps': [{'token': 1}]}), encoding='utf-8')
            self.assertEqual(census.main([str(src), '-o', str(out)]), 2)

    def test_window_passes_placement_flags(self):
        a = argparse.Namespace(model=Path('m'), cache=None, expert_tier='tierstream', unit_file=Path('u.bin'),
                               unit_manifest_sha256='a' * 64, vram_total_gib=93.0, ram_total_gib=None,
                               tier_vram_gib=None, tier_ram_gib=None, tier_spare_percent=5.0, tier_policy='auto',
                               tier_predict='1', tier_flush_ready='0', skip_empty_expert_groups='1', persistent_dense_fp16='0', expert_residency=None, expert_routes='r.txt', routes_out='o.txt',
                               capacity=None, full_card=False)
        cmd = dw.service_command(a, Path('svc.exe'))
        for flag, value in (('--tier-spare-percent', '5.0'), ('--tier-policy', 'auto'), ('--tier-predict', '1'), ('--tier-flush-ready', '0'), ('--skip-empty-expert-groups', '1'), ('--persistent-dense-fp16', '0'),
                            ('--expert-routes', 'r.txt'), ('--routes-out', 'o.txt')):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertNotIn('--expert-residency', cmd)
        self.assertNotIn('--row-tile', cmd)  # default: the service's 5% rule
        a.row_tile = 8192
        cmd = dw.service_command(a, Path('svc.exe'))
        self.assertEqual(cmd[cmd.index('--row-tile') + 1], '8192')
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            parsed = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--row-tile', '8192'])
        self.assertEqual(parsed.row_tile, 8192)
        for bad in ('8200', '0', '65536'):
            with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6), contextlib.redirect_stderr(io.StringIO()):
                dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--row-tile', bad])
        # with 8192 allocated the full-card arms run 8K at 8192 vs 2048 and 32K at 8192 vs 2048
        self.assertEqual([dw.ttft_tile(t, 8192)[0] for t in (8192, 2048, None, 2048)], [8192, 2048, None, 2048])

    def test_memory_gate_reports_raw_and_backing(self):
        peaks = {}
        a = argparse.Namespace(vram_total_gib=93.0, ram_total_gib=111.0)
        rec = {'memory': {'measured_device_peak_bytes': 98_701_794_188, 'host_peak_commit_bytes': 79_552_000_000,
                          'process_peak_commit_bytes_raw': 175_587_450_880, 'wddm_device_backing_bytes': 96_037_952_396,
                          'within_caps': True}}
        self.assertEqual(dw.memory_gate(rec, a, peaks), [])
        self.assertEqual(peaks['process_peak_commit_bytes_raw'], 175_587_450_880)
        self.assertEqual(peaks['wddm_device_backing_bytes'], 96_037_952_396)


class SnapshotTests(unittest.TestCase):
    def test_receiver_snapshot_layout_and_hash(self):
        receiver = BUILD/'glm_native_receiver.exe'
        P = 5
        kda = 34*(32*65536 + 32*4608)
        dsa = 12*(P*1024 + 2*P*256 + (P//4)*512)
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            path = Path(td)/'state-r0.bin'
            h = hashlib.sha256()
            with path.open('wb') as f:
                f.write(struct.pack('<QQ', P, P+1))
                block = bytes(range(256))*4096
                left = kda + dsa
                while left:
                    b = block[:min(left, len(block))]
                    f.write(b)
                    h.update(b)
                    left -= len(b)
            out = subprocess.run([str(receiver), '--snapshot', str(path), '0', str(P), str(P+1)], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.split(), ['SNAPSHOT', h.hexdigest(), str(kda+dsa)])
            bad = subprocess.run([str(receiver), '--snapshot', str(path), '0', str(P+1), str(P+2)], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            with path.open('r+b') as f:
                f.truncate(16 + kda + dsa - 1)
            short = subprocess.run([str(receiver), '--snapshot', str(path), '0', str(P), str(P+1)], capture_output=True, text=True)
            self.assertNotEqual(short.returncode, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, help='fixture artifact published by glm_native_decode_host')
    parsed = parser.parse_args()
    BUILD, RECEIPT = parsed.build.resolve(), parsed.receipt
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__('__main__')))
    raise SystemExit(0 if result.wasSuccessful() else 1)
