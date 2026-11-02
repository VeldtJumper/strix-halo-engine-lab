#!/usr/bin/env python3
"""CPU receipt/identity tests; synthetic timings are confined to tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('kit', HERE.parent / 'lib/kit.py')
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


# What finalize records for the fixture's code/plain measurement (asserted in test_pairs_ignore_optional_exactness).
WORKLOAD = {'prompts': [{'id': 'code', 'sha256': k.id_hash([1, 2, 3]), 'tokens': 3}], 'generate_length': 2}


class KitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def test_effective_formats_and_identity(self):
        glm=k.parse_effective('effective baseline=0\neffective identity=r1b (default)\neffective env HSA_DISABLE_FRAGMENT_ALLOCATOR=1 (default, item allocator)\n')
        self.assertEqual(glm['environment'],{'HSA_DISABLE_FRAGMENT_ALLOCATOR':'1'})
        self.assertEqual(glm['settings']['identity'],'r1b')
        mini=k.parse_effective('minimax defaults: STRIX_MM_BASELINE=1, the 22 optimization items are off (an explicit item value wins)\napp=rank\nSTRIX_MM_INDEX_CONC=0\n')
        self.assertEqual(mini['environment']['STRIX_MM_INDEX_CONC'],'0')
        rtx=k.parse_effective('{"load_serial":false,"draft_vocab":"auto:vocab","draft_keep":64}')
        self.assertEqual(rtx['environment'],{'DG_LOAD_SERIAL':'0'})
        with self.assertRaisesRegex(ValueError,'missing effective'):
            k.parse_effective('a GPU timing receipt is not a config')
        with self.assertRaisesRegex(ValueError,'draft vocab'):
            k.parse_effective('{"load_serial":false,"draft_vocab":"full"}')
        # The recorded environment, including app-specific differences, enters both keys.
        inv=dict(build_options={},binary_sha256s=[],model_files=[],repo_commit='0'*40,
                 tracked_dirty=False, hardware=dict(rocm_version=None,hip_version=None,kernel='test'))
        k.dump(self.out/'local-inventory.json',inv)
        env={'decode':{'STRIX_TEST':'1'},'prefill':{'STRIX_TEST':'0'}}
        k.dump(self.out/'effective-environment.json',env)
        with patch.object(k,'ENGINE','glm-pair'), patch.dict(k.os.environ,{'BENCH_PROFILE':'default','BENCH_SUITE':'quick','BENCH_PROMPTS':'kit'}):
            k.initialize(self.out)
        identity=k.read(self.out/'identity.json')
        self.assertEqual(identity['identity']['environment'],env)
        self.assertEqual(identity['reference_key']['environment'],env)

    def test_qwen_notices_do_not_hide_effective_json(self):
        # Engine stderr notice lines precede the stdout JSON in the captured text.
        native_status = dict(rope_mode='native', factor=0, original_max=262144, effective_max=262144)
        env = {'STRIX_QPF2K_SCHED': {'value': '12', 'source': 'automatic'},
               'STRIX_QWEN_ROPE_YARN': {'value': 'off', 'source': 'baseline'}}
        baseline = ('Qwen baseline: suppressed YaRN request (STRIX_QWEN_ROPE_YARN=2); rotary=native\n' +
                    json.dumps(dict(rotary=native_status, suppressed_yarn_request='STRIX_QWEN_ROPE_YARN=2',
                                    admitted=True, environment=env, argv=['--context', '8192', '--baseline'])))
        parsed = k.parse_effective(baseline)
        self.assertEqual(parsed['environment']['STRIX_QPF2K_SCHED'], '12')
        self.assertEqual(parsed['environment']['STRIX_QWEN_ROPE_YARN'], 'off')
        self.assertEqual(k.qwen_rotary_reference({'decode': parsed}), {})
        payload = dict(rope_mode='yarn', factor=2, original_max=262144, effective_max=524288,
                       rotary_dim=64, index_rotary_dim=64, theta=10000000, mrope_sections=[11,11,10],
                       mrope_interleaved=True, beta_fast=32, beta_slow=1, truncate=True,
                       attention_factor=1.0693147180559945, math_version='qwen-yarn-hf64-v1',
                       generator='fp64-libm-to-f32-rne', table_sha256='1'*64)
        active = ('Qwen experimental YaRN: factor=2 context=300000 T=8192; qualification=experimental; '
                  'verified PLE credit=28322006848 bytes\n' +
                  json.dumps(dict(rotary=payload, admission={'host_reserve_bytes': 8589934592}, admitted=True,
                                  environment={'STRIX_QPF2K_SCHED': {'value': '12', 'source': 'automatic'}},
                                  argv=['--context', '300000', '--rope-yarn', '2'])))
        parsed = k.parse_effective(active)
        self.assertEqual(parsed['arguments'][-2:], ['--rope-yarn', '2'])
        self.assertNotEqual(k.qwen_rotary_reference({'decode': parsed}), {})

    def test_rotary_reference_identity(self):
        native = {'decode': {'settings': {}}}
        self.assertEqual(k.qwen_rotary_reference(native), {})
        payload = dict(rope_mode='yarn', factor=2, original_max=262144, effective_max=524288,
                       rotary_dim=64, index_rotary_dim=64, theta=10000000, mrope_sections=[11,11,10],
                       mrope_interleaved=True, beta_fast=32, beta_slow=1, truncate=True,
                       attention_factor=1.0693147180559945, math_version='qwen-yarn-hf64-v1',
                       generator='fp64-libm-to-f32-rne', table_sha256='1'*64)
        config = {'decode': {'settings': {'rotary': payload}}}
        key = k.qwen_rotary_reference(config)
        signals = [dict(arguments=['--rope-yarn','2']), dict(arguments=['--rope-yarn=2']),
                   dict(requested_args=['--rope-yarn','2']), dict(requested_args={'--rope-yarn':'2'}), dict(requested_arguments='--rope-yarn 2'),
                   dict(environment={'STRIX_QWEN_ROPE_YARN':'2'}),
                   dict(settings=dict(reason='experimental-yarn')),
                   dict(settings=dict(environment={'STRIX_QWEN_ROPE_YARN':dict(value='2',source='explicit')})),
                   dict(requested=dict(argv=['--rope-yarn','2'])),
                   dict(settings=dict(resolver_reason='experimental-yarn'))]
        for signal in signals:
            for native_payload in (None, dict(rope_mode='native')):
                changed = copy.deepcopy(signal)
                changed.setdefault('settings', {})
                if native_payload is not None: changed['settings']['rotary'] = native_payload
                with self.subTest(signal=signal, native=native_payload), self.assertRaises(ValueError):
                    k.qwen_rotary_reference({'decode':changed})
            changed = copy.deepcopy(signal)
            changed.setdefault('settings', {})['rotary'] = payload
            self.assertEqual(key, k.qwen_rotary_reference({'decode':changed}))
        for signal in (dict(arguments=['--rope-yarn','4']), dict(arguments=['--rope-yarn','off']),
                       dict(environment={'STRIX_QWEN_ROPE_YARN':'4'}),
                       dict(requested_args=['--rope-yarn=4'])):
            changed = dict(signal, settings=dict(rotary=payload))
            with self.assertRaises(ValueError): k.qwen_rotary_reference({'decode':changed})
        removed = copy.deepcopy(config)
        removed['decode']['settings']['reason'] = 'experimental-yarn'
        removed['decode']['arguments'] = ['--context','300000','--rope-yarn','2']
        del removed['decode']['settings']['rotary']
        with self.assertRaises(ValueError): k.qwen_rotary_reference(removed)

        for field in payload:
            changed = copy.deepcopy(config)
            value = changed['decode']['settings']['rotary'][field]
            if field == 'table_sha256': value = '2'*64
            elif field == 'rope_mode': continue
            elif isinstance(value, bool): value = not value
            elif isinstance(value, (int, float)): value += 1
            elif isinstance(value, list): value[0] += 1
            else: value += '-changed'
            changed['decode']['settings']['rotary'][field] = value
            self.assertNotEqual(key, k.qwen_rotary_reference(changed), field)
        for missing in payload:
            changed = copy.deepcopy(config)
            del changed['decode']['settings']['rotary'][missing]
            with self.assertRaises(ValueError): k.qwen_rotary_reference(changed)
        changed = copy.deepcopy(config)
        changed['decode']['settings']['rotary']['table_sha256'] = None
        with self.assertRaises(ValueError): k.qwen_rotary_reference(changed)
        changed = copy.deepcopy(config)
        changed['decode']['settings']['rotary']['factor'] = 4
        changed['decode']['settings']['rotary']['effective_max'] = 1048576
        self.assertNotEqual(key, k.qwen_rotary_reference(changed))

    def test_rotary_refusal_does_not_publish(self):
        inv=dict(build_options={},binary_sha256s=[],model_files=[],repo_commit='0'*40,
                 tracked_dirty=False,hardware=dict(rocm_version=None,hip_version=None,kernel='test'))
        k.dump(self.out/'local-inventory.json',inv)
        cases = [dict(rotary={}), dict(rotary=dict(factor=2)),
                 dict(rope_mode='yarn',rotary=dict(rope_mode='native')),
                 dict(rope_mode='native',rotary=dict(rope_mode='yarn')),
                 dict(rotary=dict(rope_mode='native',factor=2)),
                 dict(rotary=dict(rope_mode='native',effective_max=524288)),
                 dict(rotary=dict(rope_mode='native',table_sha256='1'*64)),
                 dict(rotary=dict(rope_mode='native',descriptor_sha256='1'*64)),
                 dict(rotary=dict(rope_mode='native'),rotary_identity=dict(rope_mode='yarn'))]
        for settings in cases:
            k.dump(self.out/'effective-decode.json',dict(settings=settings))
            with patch.object(k,'ENGINE','qwen'), patch.dict(k.os.environ,{'BENCH_PROFILE':'default','BENCH_SUITE':'quick','BENCH_PROMPTS':'kit'}):
                with self.assertRaises(ValueError): k.initialize(self.out)
            self.assertFalse((self.out/'identity.json').exists(), settings)
        native=dict(rope_mode='native',factor=0,original_max=262144,effective_max=262144,
                    rotary_dim=64,index_rotary_dim=64,theta=10000000,mrope_sections=[11,11,10],
                    mrope_interleaved=True,beta_fast=32,beta_slow=1,truncate=True,attention_factor=1,
                    math_version='qwen-yarn-hf64-v1',generator='fp64-libm-to-f32-rne',table_sha256=None,table_bytes=0)
        self.assertEqual(k.qwen_rotary_reference({'decode':dict(settings=dict(rotary=native))}), {})
        native['descriptor_sha256']=k.hashlib.sha256(b'').hexdigest()
        with patch.object(k,'ENGINE','qwen'), patch.dict(k.os.environ,{'BENCH_PROFILE':'default','BENCH_SUITE':'quick','BENCH_PROMPTS':'kit'}):
            k.dump(self.out/'effective-decode.json',dict(settings={}))
            k.initialize(self.out)
            historical=k.read(self.out/'identity.json')['reference_sha256']
            k.dump(self.out/'effective-decode.json',dict(settings=dict(rotary=native)))
            k.initialize(self.out)
            self.assertEqual(k.read(self.out/'identity.json')['reference_sha256'],historical)


    def test_qwen_auto_config_and_admission(self):
        config={'admitted':True,'environment':{'STRIX_PREFILL_PIPE':{'value':'255','source':'automatic'}},
                'argv':['--context','8192'],'T':1024,'context':8192,
                'decode_profile':'dense-a8','prefill_profile':'fast-prefill-f16wmma-v1'}
        text='Qwen auto-prefill: automatic T1024 context=8192\n'+json.dumps(config)
        parsed=k.parse_effective(text)
        self.assertEqual(parsed['environment'],{'STRIX_PREFILL_PIPE':'255'})
        self.assertEqual(parsed['settings']['environment']['STRIX_PREFILL_PIPE']['source'],'automatic')
        log=self.out/'prefill.log'
        log.write_text(text+'\n'+json.dumps({'generated_tokens':[1],'prompt_ms':100}),encoding='utf-8')
        with patch.object(k,'ENGINE','qwen'), patch.dict(k.os.environ,{'BENCH_SELECTED_PROFILE':'default'}):
            k.ingest(self.out,'prefill-1024','ttft',1,log)
        sample=json.loads((self.out/'samples.jsonl').read_text())
        self.assertEqual(sample['auto_prefill']['T'],1024)
        self.assertEqual(sample['auto_prefill']['context'],8192)
        self.assertIsNone(sample['engine_digest'])
        for text in ('Qwen auto-prefill declined: test-capacity; using serial prefill', 'no notice'):
            with self.assertRaises(ValueError): k.qwen_admission(text)
        inv=dict(build_options={},binary_sha256s=[],model_files=[],repo_commit='0'*40,
                 tracked_dirty=False,hardware=dict(rocm_version=None,hip_version=None,kernel='test'))
        k.dump(self.out/'local-inventory.json',inv)
        k.dump(self.out/'effective-prefill.json',parsed)
        env={'prefill':parsed['environment']}
        k.dump(self.out/'effective-environment.json',env)
        with patch.object(k,'ENGINE','qwen'), patch.dict(k.os.environ,{'BENCH_PROFILE':'default','BENCH_SUITE':'quick','BENCH_PROMPTS':'kit'}):
            k.initialize(self.out)
        identity=k.read(self.out/'identity.json')
        self.assertEqual(identity['identity']['environment'],env)
        self.assertEqual(identity['reference_key']['environment'],env)
        self.assertEqual(identity['reference_key']['profile_pairs']['prefill']['prefill_profile'],'fast-prefill-f16wmma-v1')
        self.assertEqual(identity['reference_sha256'], '294a5ed80404d42c03741e36165dc54202753397446161f7ba16a30a1956a40e')
        with patch.dict(k.os.environ,{'BENCH_SELECTED_PROFILE':'baseline'}):
            self.assertEqual(k.parse_effective('{"load_serial":true,"draft_vocab":"full"}')['environment'],{'DG_LOAD_SERIAL':'1'})

    def test_model_hash_cache(self):
        asset = self.out / 'model.bin'
        asset.write_bytes(b'model')
        calls = []
        def counted(path):
            calls.append(path)
            return k.sha(path)
        cache = k.ModelHashes(self.out, 'cached', counted)
        first = cache.digest(asset)
        cache.save()
        self.assertTrue((self.out / '.bench-sha256-cache.json').is_file())
        cache = k.ModelHashes(self.out, 'cached', counted)
        self.assertEqual(cache.digest(asset), first)
        self.assertEqual(len(calls), 1)
        st = asset.stat()
        k.os.utime(asset, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000000))
        self.assertEqual(cache.digest(asset), first)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(k.ModelHashes(self.out, 'skip', counted).digest(asset))
        self.assertEqual(len(calls), 2)
        k.ModelHashes(self.out, 'full', counted).digest(asset)
        self.assertEqual(len(calls), 3)

    def test_resolved_parameters_and_validation(self):
        import contextlib, io
        out = io.StringIO()
        args = ['glm-pair','full','2k,16k','','1,4,8','2','0','','kit','prefill','0']
        with contextlib.redirect_stdout(out): k.interface.resolve(args,k.KIT)
        self.assertIn('context_values=(2048 16384)',out.getvalue())
        self.assertIn('profile=r1b',out.getvalue())
        self.assertIn('only=prefill',out.getvalue())
        args[7] = 'invalid'
        with self.assertRaises(ValueError): k.interface.resolve(args,k.KIT)

    def test_batched_glm_and_minimax_receipts(self):
        log = self.out / 'batch.log'
        rows = [dict(prompt_id='chat',mode='greedy',generation=2,warmup=True),
                dict(prompt_id='chat',mode='greedy',generation=2,warmup=False)]
        block = ('rank 0 pf dev free delta +1 MiB\n'
                 'rank 0 prefill gemv: 2 tokens 1.0 s, logits fnv abc, state fnv def\n'
                 'rank 0 decode: 1 steps (20.0 tok/s), tokens fnv abc\nrank 0 tokens: 1 2\n')
        log.write_text(block*2)
        with patch.object(k,'ENGINE','glm-pair'): k.ingest_batch(self.out,rows,log)
        self.assertEqual(len((self.out/'samples.jsonl').read_text().splitlines()),1)
        row = dict(prompt_id='chat',mode='greedy',generation=2,warmup=False,name='d0')
        log.write_text('rank 0 d0: decode 1 steps at ctx 2: median(steps 1-1) 50.0 ms/token = 20.0 tok/s; mean 50 ms\nrank 0 d0: tokens fnv abc ids: 1 2\nrank 0 d0: state fnv abc, final rows fnv def\n')
        with patch.object(k,'ENGINE','minimax-pair'): k.ingest_batch(self.out,[row],log)
        self.assertEqual(len((self.out/'samples.jsonl').read_text().splitlines()),2)
        with patch.object(k,'ENGINE','minimax-pair'), self.assertRaises(ValueError):
            k.ingest_batch(self.out,[dict(row,name='missing')],log)

    def test_glm_spv1_current_receipts(self):
        log=self.out/'spv1.log'
        text=('SPV1_BEGIN rank=0 N=2\n'
              'SPV1_PLAIN rank=0 prompt=code pass=A0 gen=2 ms_token=50 hash=0000000000000abc\n'
              'SPV1_RESULT rank=0 prompt=code K=7 gen=2 ms_token=25 equal=1 hash=0000000000000abc\n'
              'SPV1_PLAIN rank=0 prompt=code pass=A1 gen=2 ms_token=51 hash=0000000000000abc\n'
              'SPV1_DONE rank=0 status=PASS prompts=1 arms=1\n')
        log.write_text(text,encoding='ascii')
        rows=[dict(prompt_id='code',mode=m,generation=2,warmup=False) for m in ('greedy','spec')]
        with patch.object(k,'ENGINE','glm-pair'): k.ingest_batch(self.out,rows,log)
        samples=[json.loads(x) for x in (self.out/'samples.jsonl').read_text().splitlines()]
        self.assertEqual([x['values'] for x in samples],[[20],[40]])
        self.assertTrue(all(x['token_ids_sha256'] is None for x in samples))
        fixture=k.read(HERE/'fixtures/results.json')
        fixture['engine']='glm-pair'
        k.dump(self.out/'identity.json',fixture)
        k.dump(self.out/'prompts.json',fixture['prompts'])
        k.finalize(self.out)
        result=k.read(self.out/'results.json')
        self.assertEqual(result['exactness'],[])
        self.assertIn('kit exactness unqualified',result['skipped'][0])
        with self.assertRaisesRegex(ValueError,'without full token IDs'): k.finalize(self.out,True)
        log.write_text(text.replace('equal=1','equal=0'),encoding='ascii')
        with patch.object(k,'ENGINE','glm-pair'),self.assertRaisesRegex(ValueError,'equality mismatch'):
            k.ingest_batch(self.out,rows,log)

    def test_glm_mtp_batch(self):
        log = self.out/'mtp.log'
        log.write_text('rank 0 spec-run: K 7 | plain 50.0 ms/token, spec 25.0 ms/token (2x) | tokens EQUAL to plain (fnv abc)\nrank 0 spec-run plain tokens: 1 2\nrank 0 spec-run tokens: 1 2\n')
        rows = [dict(prompt_id='chat',mode=mode,generation=2,warmup=False) for mode in ('greedy','spec')]
        with patch.object(k,'ENGINE','glm-pair'): k.ingest_batch(self.out,rows,log)
        data=[json.loads(line) for line in (self.out/'samples.jsonl').read_text().splitlines()]
        self.assertEqual([r['values'] for r in data],[[20],[40]])

    def test_hash_is_canonical(self):
        self.assertEqual(k.id_hash([1, 2, 3]), 'a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4')
        with self.assertRaises(AssertionError):
            k.id_hash([True])

    def test_qwen_receipt(self):
        log = self.out / 'receipt.log'
        log.write_text('diagnostic\n' + json.dumps({'generated_tokens': [1, 2], 'generation_step_elapsed_ms_total': 50}), encoding='utf-8')
        with patch.object(k, 'ENGINE', 'qwen'):
            k.ingest(self.out, 'code', 'plain', 2, log)
        row = json.loads((self.out / 'samples.jsonl').read_text())
        self.assertEqual(row['values'], [40])
        self.assertEqual(row['actual_tokens'], 2)

    def test_missing_timing_fails_closed(self):
        log = self.out / 'receipt.log'
        log.write_text('{"generated_tokens":[1]}', encoding='utf-8')
        with patch.object(k, 'ENGINE', 'qwen'), self.assertRaises(ValueError):
            k.ingest(self.out, 'code', 'spec', 1, log)

    def test_deepseek_native_plan_and_report(self):
        import contextlib, io
        names=[x['id'] for x in k.interface.prompt_set('kit',k.KIT)]
        params=dict(profile='default',contexts=[3,5],gen_tokens=2,warmup=1,repeat=2,only=['decode','prefill'])
        for n in params['contexts']:
            for name in names: (self.out/f'{name}-ctx{n}.ids').write_text(','.join(['1']*n))
        output=io.StringIO()
        with patch.object(k,'ENGINE','deepseek-pair'), patch.dict(k.os.environ,{'BENCH_PARAMS':json.dumps(params)}), contextlib.redirect_stdout(output):
            k.interface.plan(self.out,False,k)
        plan=output.getvalue()
        self.assertEqual(plan.count('deepseek_native_decode '),2)
        self.assertNotIn('strix-deepseek41-serve',plan)
        self.assertIn('--transport usb4auto',plan)
        for mode in ('greedy','spec'):
            rows=k.read(self.out/f'decode-{mode}.manifest.json')
            lines=(self.out/f'decode-{mode}.ids').read_text().splitlines()
            self.assertEqual(len(rows),2*len(names)*3)
            self.assertEqual(sum(x['warmup'] for x in rows),2*len(names))
            self.assertEqual([len(line.split(',')) for line in lines],[int(x['prompt_id'].rsplit('ctx',1)[1]) for x in rows])
        fixture=k.read(HERE/'fixtures/results.json')
        fixture['engine']='deepseek-pair'
        row=fixture['measurements'][0]
        fixture['measurements']=[dict(row,prompt_id='code-ctx3',mode=mode,timing_scope='native') for mode in ('greedy','spec')]
        fixture['measurements'].append(dict(row,prompt_id='http-c1',mode='aggregate-wall',timing_scope='http-end-to-end'))
        k.dump(self.out/'results.json',fixture)
        k.report(self.out/'results.json')
        text=(self.out/'results.md').read_text()
        for value in ('34.5 tok/s','70.6 tok/s','workload differs','HTTP end-to-end'): self.assertIn(value,text)

    def test_pair_receipt_and_prefill(self):
        log = self.out / 'pair.log'
        log.write_text('tokens (3, fnv 0000000000000001): 1 2 3\nprompt 16 tokens in 1.00 s; generation 2 tokens in 0.050 s = 40.00 tok/s\n', encoding='utf-8')
        with patch.object(k, 'ENGINE', 'deepseek-pair'):
            k.ingest(self.out, 'code', 'greedy', 2, log)
            log.write_text('ttft warm-up run: 1024 rows, wall 999.000 ms\nttft run 0: 1024 rows, wall 2000.000 ms (run 1999.0 ms), next token 4, logits_fnv 0000000000000001\nttft run 1: 1024 rows, wall 2200.000 ms (run 2199.0 ms), next token 4, logits_fnv 0000000000000001\n', encoding='utf-8')
            k.ingest(self.out, 'prefill-1024', 'ttft', 1, log)
        rows = [json.loads(x) for x in (self.out / 'samples.jsonl').read_text().splitlines()]
        self.assertEqual([x['values'] for x in rows], [[40], [2], [2.2]])

    def test_rtx_receipt(self):
        log = self.out / 'rtx.log'
        events = [{'event': 'chat_end', 'name': 'chats0#1-0-x', 'toks': [1, 2], 'digest': 'abc'},
                  {'event': 'chat_end', 'name': 'schats0#1-0-x', 'toks': [1, 2], 'digest': 'abc'},
                  {'event': 'summary', 'phase': 'chats0#1-0-x', 'tok_s': 10},
                  {'event': 'spec_summary', 'phase': 'schats0#1-0-x', 'tok_s': 20}]
        log.write_text('\n'.join(json.dumps(x) for x in events), encoding='utf-8')
        with patch.object(k, 'ENGINE', 'deepseek-rtx'):
            k.ingest(self.out, 'code', 'rtx', 1, log)
            log.write_text('{"event":"run","seconds":2.5,"next_token":8,"logits_fnv":"abc"}\n', encoding='utf-8')
            k.ingest(self.out, 'prefill-1024', 'ttft', 1, log)
        self.assertEqual(len((self.out / 'samples.jsonl').read_text().splitlines()), 3)

    def test_schema_rejects_wrong_stats(self):
        d = k.read(HERE / 'fixtures/results.json')
        d['measurements'][0]['median'] = 0
        with self.assertRaises(ValueError):
            k.validate(d)

    def test_profile_pins_and_schema(self):
        schema = k.read(HERE.parent / 'results.schema.json')
        fixture = k.read(HERE / 'fixtures/results.json')
        self.assertTrue(set(schema['required']).issubset(fixture))
        for engine, (_, names) in k.PROFILES.items():
            profile = (HERE.parent / 'engines' / (engine + '.sh')).read_text()
            for name in names:
                if name in k.PINS:
                    size, digest = k.PINS[name]
                    self.assertIn(str(size) if size is not None else chr(110)+chr(117)+chr(108)+chr(108), profile)
                    self.assertIn(digest, profile)

    def test_reference_verdicts(self):
        fixture = k.read(HERE / 'fixtures/results.json')
        identity = copy.deepcopy(fixture)
        for key in ('measurements', 'exactness', 'prompts'):
            identity.pop(key)
        k.dump(self.out / 'identity.json', identity)
        k.dump(self.out / 'prompts.json', fixture['prompts'])
        sample = {'prompt_id': 'code', 'mode': 'plain', 'generate_length': 2, 'actual_tokens': 2,
                  'token_ids_sha256': k.id_hash([1, 2]), 'engine_digest': None,
                  'values': [40], 'unit': 'tok/s', 'log': 'fixture.log'}
        (self.out / 'samples.jsonl').write_text(json.dumps(sample) + '\n', encoding='utf-8')
        k.finalize(self.out, True)
        d = k.read(self.out / 'results.json')
        self.assertEqual(d['exactness'][0]['verdict'], 'NO-REFERENCE')
        refs = k.read(self.out / 'candidate-refs.json')
        actual_read = k.read
        def read_refs(path):
            return refs if Path(path).parent.name == 'refs' else actual_read(path)
        with patch.object(k, 'read', read_refs):
            k.finalize(self.out)
            self.assertEqual(actual_read(self.out / 'results.json')['exactness'][0]['verdict'], 'EXACT')
            # A self-built binary changes the full identity but not the reference key.
            identity['identity_sha256'] = '2' * 64
            k.dump(self.out / 'identity.json', identity)
            k.finalize(self.out)
            self.assertEqual(actual_read(self.out / 'results.json')['exactness'][0]['verdict'], 'EXACT')
            sample['token_ids_sha256'] = k.id_hash([2, 3])
            (self.out / 'samples.jsonl').write_text(json.dumps(sample) + '\n', encoding='utf-8')
            k.finalize(self.out)
            self.assertEqual(actual_read(self.out / 'results.json')['exactness'][0]['verdict'], 'MISMATCH')

    def test_finalize_exit_status(self):
        fixture = k.read(HERE / 'fixtures/results.json')
        identity = copy.deepcopy(fixture)
        for key in ('measurements', 'exactness', 'prompts'):
            identity.pop(key)
        k.dump(self.out / 'identity.json', identity)
        k.dump(self.out / 'prompts.json', fixture['prompts'])
        sample = {'prompt_id': 'code', 'mode': 'plain', 'generate_length': 2, 'actual_tokens': 2,
                  'token_ids_sha256': k.id_hash([1, 2]), 'engine_digest': None,
                  'values': [40], 'unit': 'tok/s', 'log': 'fixture.log'}
        def run(samples):
            for name in ('results.json', 'results.md'):
                (self.out / name).unlink(missing_ok=True)
            (self.out / 'samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples), encoding='utf-8')
            rc = k.main(['finalize', str(self.out), '0'])
            self.assertTrue((self.out / 'results.md').is_file())
            return rc, k.read(self.out / 'results.json')
        # NO-REFERENCE is not a failure: measure-only and candidate runs exit 0.
        rc, d = run([sample])
        self.assertEqual(rc, 0)
        self.assertEqual(d['status'], 'complete')
        self.assertEqual([e['verdict'] for e in d['exactness']], ['NO-REFERENCE'])
        # Spec tokens differing from plain tokens are a MISMATCH: results written, nonzero exit.
        spec = dict(sample, mode='spec', token_ids_sha256=k.id_hash([2, 3]))
        rc, d = run([sample, spec])
        self.assertEqual(rc, 1)
        self.assertEqual(d['status'], 'mismatch')
        self.assertEqual([e['verdict'] for e in d['exactness']], ['NO-REFERENCE', 'MISMATCH'])

    def test_redaction(self):
        address = '.'.join(['1'] * 4)
        mac = ':'.join(['ab'] * 6)
        text = f'model /path/to/models/model.gguf peer user@node-b address {address} MAC {mac} speed 40 tok/s'
        clean = k.sanitize(text)
        for private in ('/path/to', 'user@node-b', address, mac):
            self.assertNotIn(private, clean)
        self.assertIn('40 tok/s', clean)

    def test_http_sse_usage_and_concurrency(self):
        (self.out / 'server.log').write_text('{"event":"READY"}\n', encoding='ascii')
        for prompt in ('chat', 'code', 'prose'):
            (self.out / (prompt + '.ids')).write_text('1,2,3\n', encoding='ascii')
        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def __iter__(self):
                yield b'data: {"choices":[{"text":"x"}]}\n'
                yield b'data: {"usage":{"prompt_tokens":3,"completion_tokens":1},"strix":{"token_ids":[4]}}\n'
                yield b'data: [DONE]\n'
        with patch.object(k.urllib.request, 'urlopen', return_value=Response()), patch.dict(k.os.environ, {'BENCH_REPEATS': '1'}):
            k.http(self.out, 18880)
        rows = k.read(self.out / 'http-measurements.json')
        self.assertEqual(len(rows), 6)
        for c in (1, 4, 8):
            requests = k.read(self.out / f'http-c{c}.json')['requests'][0]
            self.assertEqual(len(requests), 8)
            self.assertTrue(all(x['completion_tokens'] == 1 and x['ttft_s'] > 0 for x in requests))

        self.assertEqual({row['timing_scope'] for row in rows},{'http-end-to-end'})

    def test_pair_identity_mismatch(self):
        data = {'repo_commit': '0' * 40, 'binary_sha256s': [], 'model_files': [],
                'build_options': {}, 'hardware': {'rocm_version': 'test'}}
        k.dump(self.out / 'a.json', data)
        data['repo_commit'] = '1' * 40
        k.dump(self.out / 'b.json', data)
        with self.assertRaisesRegex(ValueError, 'repo_commit'):
            k.main(['compare', str(self.out / 'a.json'), str(self.out / 'b.json')])


    def protocol_run(self, env=None, requests=None):
        fixture = k.read(HERE / 'fixtures/results.json')
        identity = copy.deepcopy(fixture)
        for key in ('measurements', 'exactness', 'prompts'):
            identity.pop(key)
        k.dump(self.out / 'identity.json', identity)
        k.dump(self.out / 'prompts.json', fixture['prompts'])
        sample = {'prompt_id': 'code', 'mode': 'plain', 'generate_length': 2, 'actual_tokens': 2,
                  'token_ids_sha256': k.id_hash([1, 2]), 'engine_digest': None,
                  'values': [40], 'unit': 'tok/s', 'log': 'fixture.log'}
        (self.out / 'samples.jsonl').write_text(json.dumps(sample) + '\n', encoding='utf-8')
        if requests is not None:
            k.dump(self.out / 'http-c1.json', {'concurrency': 1, 'max_tokens': 2, 'requests': [requests]})
        with patch.dict(k.os.environ, env or {}):
            rc = k.main(['finalize', str(self.out), '0'])
        return rc, k.read(self.out / 'results.json')

    def test_http_engine_counts_and_cache_state(self):
        (self.out / 'server.log').write_text('{"event":"READY"}\n', encoding='ascii')
        for prompt in ('chat', 'code', 'prose'):
            (self.out / (prompt + '.ids')).write_text('1,2,3\n', encoding='ascii')
        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def __iter__(self):
                yield b'data: {"choices":[{"text":"x"}]}\n'
                yield (b'data: {"usage":{"prompt_tokens":3,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":2}},'
                       b'"strix":{"token_ids":[4,5],"queue_ms":1.0,"prefill_ms":9.0,"decode_ms":50.0}}\n')
                yield b'data: [DONE]\n'
        with patch.object(k.urllib.request, 'urlopen', return_value=Response()), \
             patch.dict(k.os.environ, {'BENCH_REPEATS': '1', 'BENCH_CONCURRENCY': '1'}):
            k.http(self.out, 18880)
        request = k.read(self.out / 'http-c1.json')['requests'][0][0]
        self.assertEqual((request['completion_tokens'], request['client_token_count'], request['cached_tokens']), (2, 2, 2))
        self.assertEqual(request['engine_decode_ms'], 50.0)
        rows = {r['mode']: r for r in k.read(self.out / 'http-measurements.json')}
        self.assertEqual(set(rows), {'aggregate-wall', 'ttft', 'engine-decode', 'engine-ttft'})
        self.assertEqual(rows['engine-decode']['median'], 40.0)
        self.assertAlmostEqual(rows['engine-ttft']['median'], 0.010)
        self.assertEqual(rows['engine-decode']['rate_source'], 'engine')
        self.assertEqual(rows['aggregate-wall']['rate_source'], 'client-clock')
        self.assertEqual({r['cache_state'] for r in rows.values()}, {'reused'})
        self.assertEqual(rows['aggregate-wall']['workload'],
                         {'prompts': [dict(id=p, sha256=k.id_hash([1, 2, 3]), tokens=3) for p in ('chat', 'code', 'prose')],
                          'generate_length': 128})

    def test_cache_policy_flag_refuse_declared(self):
        reused = [dict(request=0, cached_tokens=64), dict(request=1, cached_tokens=0)]
        rc, d = self.protocol_run(requests=reused)
        self.assertEqual((rc, d['status']), (0, 'complete'))
        policy = d['cache_policy']
        self.assertEqual((policy['states'], policy['nonconforming'], policy['refused']), ({'http-c1': 'reused'}, True, False))
        self.assertIn('NONCONFORMING', (self.out / 'results.md').read_text())
        rc, d = self.protocol_run({'BENCH_CACHE_POLICY': 'refuse'}, reused)
        self.assertEqual((rc, d['status'], d['cache_policy']['refused']), (1, 'failed', True))
        rc, d = self.protocol_run({'BENCH_CACHE_POLICY': 'refuse', 'BENCH_ALLOWED_CACHED_TOKENS': '64'}, reused)
        self.assertEqual((rc, d['cache_policy']['nonconforming']), (0, False))
        rc, d = self.protocol_run({'BENCH_CACHE_POLICY': 'refuse', 'BENCH_DECLARED_CACHE_REUSE': '1'}, reused)
        self.assertEqual((rc, d['cache_policy']['nonconforming'], d['cache_policy']['declared_reuse']), (0, False, True))
        rc, d = self.protocol_run(requests=[dict(request=0, cached_tokens=0)])
        self.assertEqual(d['cache_policy']['states'], {'http-c1': 'cold'})
        # A refused run cannot write reference candidates.
        with patch.dict(k.os.environ, {'BENCH_CACHE_POLICY': 'refuse'}), self.assertRaisesRegex(ValueError, 'cannot write'):
            k.dump(self.out / 'http-c1.json', {'concurrency': 1, 'requests': [reused]})
            k.finalize(self.out, True)

    def test_not_run_list(self):
        rc, d = self.protocol_run()
        self.assertEqual((rc, d['not_run']), (0, []))
        self.assertNotIn('cache_policy', d)
        k.main(['not-run', str(self.out), 'decode-32768', 'out of memory budget'])
        (self.out / 'skipped.json').write_text(json.dumps(['serve: no peer']), encoding='utf-8')
        rc, d = self.protocol_run(requests=[dict(request=0, completion_tokens=1)])
        items = {x['item']: x['reason'] for x in d['not_run']}
        self.assertEqual(items['decode-32768'], 'out of memory budget')
        self.assertEqual(items['skipped'], 'serve: no peer')
        self.assertIn('no strix engine_decode_ms', items['http-c1 engine-decode'])
        self.assertIn('cached_tokens absent', items['http-c1 cache state'])
        self.assertEqual(d['cache_policy']['states'], {'http-c1': 'unverified'})
        self.assertIn('Not run: decode-32768', (self.out / 'results.md').read_text())
        # Engine rows present in http-measurements.json are not listed as not run.
        k.dump(self.out / 'http-measurements.json', [dict(prompt_id='http-c1', mode='engine-decode', unit='tok/s', **k.stats([40])),
                                                      dict(prompt_id='http-c1', mode='engine-ttft', unit='s', **k.stats([0.01]))])
        rc, d = self.protocol_run(requests=[dict(request=0, completion_tokens=1)])
        self.assertNotIn('http-c1 engine-decode', {x['item'] for x in d['not_run']})
        d['not_run'].append({'item': 'x'})
        with self.assertRaisesRegex(ValueError, 'not_run'):
            k.validate(d)
        # Legacy receipts without the protocol fields still validate.
        k.validate(k.read(HERE / 'fixtures/results.json'))

    def test_abba_k_of_n_pairs(self):
        seq = [('A', 40), ('B', 44), ('B', 42), ('A', 40), ('A', 41), ('B', 40), ('B', 43), ('A', 40)]
        r = k.abba_pairs(seq)
        self.assertEqual((r['n_pairs'], r['ab_pairs'], r['ba_pairs']), (4, 2, 2))
        self.assertEqual((r['b_better_pairs'], r['a_better_pairs'], r['k_of_n'], r['sign_agreement']), (3, 1, '3/4', '3/4'))
        self.assertAlmostEqual(r['deltas'][0], 0.1)
        lower = k.abba_pairs([('A', 2.0), ('B', 1.0)], higher_is_better=False)
        self.assertEqual(lower['k_of_n'], '1/1')
        with self.assertRaisesRegex(ValueError, 'even'):
            k.abba_pairs([('A', 1)])
        with self.assertRaisesRegex(ValueError, 'one A and one B'):
            k.abba_pairs([('A', 1), ('A', 2)])
        fixture = k.read(HERE / 'fixtures/results.json')
        paths = []
        for i, (arm, value) in enumerate(seq[:4]):
            d = copy.deepcopy(fixture)
            d['measurements'] = [dict(d['measurements'][0], workload=copy.deepcopy(WORKLOAD), **k.stats([value]))]
            paths.append(f'{arm}={self.out}/r{i}.json')
            k.dump(self.out / f'r{i}.json', d)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            k.main(['pairs', str(self.out / 'pairs.json'), *paths])
        result = k.read(self.out / 'pairs.json')
        self.assertEqual(result['arms'], ['A', 'B', 'B', 'A'])
        self.assertEqual(result['pairs'][0]['k_of_n'], '2/2')
        d = k.read(self.out / 'r1.json')
        d['cache_policy'] = {'nonconforming': True}
        k.dump(self.out / 'r1.json', d)
        with self.assertRaisesRegex(ValueError, 'cache-nonconforming'):
            k.pairs([tuple(p.split('=', 1)) for p in paths])
        d['reference_key'] = dict(d['reference_key'], prompt_set_sha256='0' * 64)
        k.dump(self.out / 'r1.json', d)
        with self.assertRaisesRegex(ValueError, 'differ'):
            k.pairs([tuple(p.split('=', 1)) for p in paths])

    def test_report_shows_readme_abba_pairs(self):
        k.dump(self.out / 'results.json', k.read(HERE / 'fixtures/results.json'))
        row = dict(value=40, unit='tok/s', tolerance=0.1, workload_match=True, pairs_faster=5, pairs=6)
        with patch.object(k, 'baseline', return_value=row):
            k.report(self.out / 'results.json')
        self.assertIn('(ABBA 5/6)', (self.out / 'results.md').read_text())

    def test_dispatch_census_attaches_to_receipt(self):
        k.dump(self.out / 'results.json', k.read(HERE / 'fixtures/results.json'))
        trace = k.ROOT / 'tests/data/dispatch_census/kernel_trace.csv'
        k.main(['census', str(self.out), str(trace), '--window-kernel', 'marker_kernel', '--tokens', '5'])
        d = k.read(self.out / 'results.json')
        self.assertEqual(d['dispatch_census']['per_token']['launches'], 1.0)
        self.assertEqual(d['dispatch_census']['source']['trace'], 'kernel_trace.csv')
        self.assertIn('Dispatch census: 8 dispatches', (self.out / 'results.md').read_text())
        # A later finalize keeps the attached census.
        rc, d = self.protocol_run()
        self.assertEqual(d['dispatch_census']['windows']['count'], 2)


    def arm_files(self, change=None):
        """Four ABBA arm receipts from the fixture; change(index, results) edits one before writing."""
        fixture = k.read(HERE / 'fixtures/results.json')
        fixture['params'] = dict(gen_tokens=2, contexts=[1024], concurrency=[1], prompts='kit', serve=False)
        arms = []
        for i, (arm, value) in enumerate([('A', 40), ('B', 44), ('B', 42), ('A', 40)]):
            d = copy.deepcopy(fixture)
            d['measurements'] = [dict(d['measurements'][0], workload=copy.deepcopy(WORKLOAD), **k.stats([value]))]
            if change: change(i, d)
            k.dump(self.out / f'arm{i}.json', d)
            arms.append((arm, self.out / f'arm{i}.json'))
        return arms

    def test_pairs_require_identical_cache_state(self):
        # Sol finding 2: a cold arm against a declared-reuse arm is refused, even when reuse is declared.
        self.assertEqual(k.pairs(self.arm_files())['pairs'][0]['k_of_n'], '2/2')
        def reused(i, d):
            if i == 1:
                d['measurements'][0].update(cache_state='reused', cached_tokens_max=64)
                d['cache_policy'] = {'nonconforming': False, 'declared_reuse': True}
        with patch.dict(k.os.environ, {'BENCH_DECLARED_CACHE_REUSE': '1'}), self.assertRaisesRegex(ValueError, 'cache state differs'):
            k.pairs(self.arm_files(reused))
        def all_reused(i, d):
            d['measurements'][0].update(cache_state='reused', cached_tokens_max=64)
        self.assertEqual(k.pairs(self.arm_files(all_reused))['pairs'][0]['cache_state'], 'reused')

    def test_pairs_require_equal_workload(self):
        # Sol finding 3: generation length, prompt identity and capacity parameters must match.
        def longer(i, d):
            if i == 2: d['params']['gen_tokens'] = 256
        with self.assertRaisesRegex(ValueError, 'workload'):
            k.pairs(self.arm_files(longer))
        def slots(i, d):
            if i == 2: d['params']['http_slots'] = 8
        with self.assertRaisesRegex(ValueError, 'workload'):
            k.pairs(self.arm_files(slots))
        def prompt(i, d):
            if i == 1: d['measurements'][0]['workload']['prompts'][0]['sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'prompt identity or generation length'):
            k.pairs(self.arm_files(prompt))
        def length(i, d):
            if i == 3: d['measurements'][0]['workload']['generate_length'] = 256
        with self.assertRaisesRegex(ValueError, 'prompt identity or generation length'):
            k.pairs(self.arm_files(length))

    def test_pairs_http_rows_compare_constituent_prompts(self):
        # Sol r2 finding 1: a synthetic http-cN row is compared through its constituent prompts and generation length.
        def http(i, d):
            d['measurements'][0].update(prompt_id='http-c1', mode='aggregate-wall', cache_state='cold', cached_tokens_max=0,
                                        workload={'prompts': [dict(id='chat', sha256='a' * 64, tokens=3),
                                                              dict(id='code', sha256='c' * 64, tokens=3)],
                                                  'generate_length': 128})
        self.assertEqual(k.pairs(self.arm_files(http))['pairs'][0]['prompt_id'], 'http-c1')
        def changed(i, d):
            http(i, d)
            if i == 2: d['measurements'][0]['workload']['prompts'][1]['sha256'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'http-c1/aggregate-wall: prompt identity or generation length'):
            k.pairs(self.arm_files(changed))
        def gen(i, d):
            http(i, d)
            if i == 1: d['measurements'][0]['workload']['generate_length'] = 256
        with self.assertRaisesRegex(ValueError, 'prompt identity or generation length'):
            k.pairs(self.arm_files(gen))

    def test_pairs_ignore_optional_exactness(self):
        # Sol r2 finding 2: --only decode (no exactness rows) pairs with --only decode,exactness.
        rc, d = self.protocol_run()
        self.assertEqual(d['measurements'][0]['workload'], WORKLOAD)
        def no_exactness(i, d):
            if i in (1, 2): d['exactness'] = []
        self.assertEqual(k.pairs(self.arm_files(no_exactness))['pairs'][0]['k_of_n'], '2/2')

    def test_pairs_validate_and_reject_failed_arms(self):
        # Sol finding 4: every receipt is validated and failed/mismatch arms are refused.
        for status in ('mismatch', 'failed'):
            def bad(i, d, status=status):
                if i == 2: d['status'] = status
            with self.assertRaisesRegex(ValueError, 'status ' + status):
                k.pairs(self.arm_files(bad))
        def invalid(i, d):
            if i == 0: d['measurements'][0]['median'] = 99
        with self.assertRaisesRegex(ValueError, 'summary differs'):
            k.pairs(self.arm_files(invalid))


N = k.nemotron


def nemotron_phase(arm='on', profile='default', digest=None, attempted=4, accepted=3):
    """Synthetic engine receipt derived from the test-only fixture."""
    phase = copy.deepcopy(k.read(HERE / 'fixtures/nemotron-receipt.json')['phase'])
    phase['profile']['profile'] = profile
    if profile == 'fast':
        phase['decode_identity'] = 'g4-a8-v1'
        phase['profile']['levers']['prefill_route']['status'] = 'pending'
        phase['profile']['pending'] = [{'lever': 'prefill_route', 'target': 'g4-wmma-pf-v2', 'runs': 'exact'}]
    if digest:
        phase['state_digest'] = digest
    phase['acceptance'].update(attempted=attempted, eligible=attempted, accepted=accepted)
    if arm == 'off':
        for key in ('draft_kind', 'draft_backend', 'draft_identity', 'draft_matrices', 'draft_topk',
                    'mtp_confidence', 'mtp_confidence_resolved'):
            phase.pop(key, None)
        phase['profile']['drafter'] = 'off (switch)'
        phase['acceptance'].update(attempted=0, eligible=0, accepted=0, cycles=0)
        for key in ('confidence_cycles', 'confidence_drafted', 'confidence_unverified'):
            phase['acceptance'].pop(key)
    return phase


def sse_lines(phase, ids, finish='length', done=True, usage=True, error=False):
    chunk = lambda choice, **extra: ('data: ' + json.dumps(dict(choices=[choice], **extra)) + '\n').encode()
    lines = [chunk({'index': 0, 'delta': {'role': 'assistant', 'content': 'x'}, 'finish_reason': None})]
    if error:
        lines.append(b'data: {"error":{"message":"failed"}}\n')
    lines.append(chunk({'index': 0, 'delta': {}, 'finish_reason': finish},
                       strix={'phases': [phase], 'budget_continued': False, 'generated_ids': ids}))
    if usage:
        lines.append(('data: ' + json.dumps({'choices': [], 'usage': {'prompt_tokens': 20, 'completion_tokens': len(ids),
                                                                     'total_tokens': 20 + len(ids)}}) + '\n').encode())
    if done:
        lines.append(b'data: [DONE]\n')
    return lines


class FakeServer:
    """urlopen stand-in: /health and chat completions with per-prompt IDs; no network."""
    def __init__(self, arm, profile='default', ids_for=None, phase_for=None):
        self.arm, self.profile = arm, profile
        self.ids_for = ids_for or (lambda text: [len(text) % 7 + 1, 2, 3, 4])
        self.phase_for = phase_for or (lambda text: nemotron_phase(arm, profile, digest=format(len(text), '064x')))

    def __call__(self, req, timeout=None):
        server = self
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                text = json.loads(req.data)['messages'][0]['content']
                return iter(sse_lines(server.phase_for(text), server.ids_for(text)))
        return Response()


class NemotronTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def test_receipt_fixture_and_draft_witness(self):
        on = nemotron_phase()
        N.check_receipt(on)
        self.assertEqual(N.draft_witness(on, 'on')['status'], 'drafted')
        self.assertEqual(N.draft_witness(nemotron_phase(accepted=0), 'on')['status'], 'declined')
        self.assertEqual(N.draft_witness(nemotron_phase('off'), 'off')['status'], 'off')
        for missing in ('mtp_confidence_resolved', 'draft_kind'):
            broken = nemotron_phase(); broken.pop(missing)
            with self.assertRaisesRegex(ValueError, 'missing draft witness'):
                N.draft_witness(broken, 'on')
        broken = nemotron_phase(); broken['acceptance'].pop('confidence_drafted')
        with self.assertRaisesRegex(ValueError, 'missing draft witness'):
            N.draft_witness(broken, 'on')
        with self.assertRaisesRegex(ValueError, 'drafting-off arm'):
            N.draft_witness(nemotron_phase(), 'off')
        broken = nemotron_phase(); broken['acceptance'].pop('accepted')
        with self.assertRaisesRegex(ValueError, 'acceptance counter'):
            N.check_receipt(broken)
        broken = nemotron_phase(); broken.pop('state_digest')
        with self.assertRaisesRegex(ValueError, 'state_digest'):
            N.check_receipt(broken)

    def test_profile_population_checks(self):
        N.check_profile(nemotron_phase(), 'default', [])
        fast = nemotron_phase(profile='fast')
        notice = 'profile fast: prefill_route pending g4-wmma-pf-v2 (exact runs)'
        N.check_profile(fast, 'fast', [notice])
        with self.assertRaisesRegex(ValueError, 'stderr notice'):
            N.check_profile(fast, 'fast', [])
        with self.assertRaisesRegex(ValueError, 'selected profile'):
            N.check_profile(nemotron_phase(), 'fast', [])
        unqualified = nemotron_phase(); unqualified['profile']['qualification'] = 'unqualified: prefill_route'
        with self.assertRaisesRegex(ValueError, 'unqualified'):
            N.check_profile(unqualified, 'default', [])
        # Q4_K_M and EXL3 references never mix.
        mixed = nemotron_phase(); mixed['exl3_identity'] = 'g4-exl3-v3'
        with self.assertRaisesRegex(ValueError, 'EXL3'):
            N.check_profile(mixed, 'default', [])
        with self.assertRaisesRegex(ValueError, 'EXL3 identity'):
            N.check_profile(nemotron_phase(profile='quality'), 'quality', [])
        a8 = nemotron_phase(); a8['decode_identity'] = 'g4-a8-v1'
        with self.assertRaisesRegex(ValueError, 'decode identity'):
            N.check_profile(a8, 'default', [])
        bits = N.numerics(nemotron_phase(), 'models')
        self.assertEqual(bits, N.numerics(nemotron_phase('off'), 'models'))
        self.assertNotIn('draft', bits['levers'])

    def test_sse_parser_rejects_truncation(self):
        clock = iter(range(100)).__next__
        first, final, usage = N.read_sse(sse_lines(nemotron_phase(), [1, 2]), clock, 0)
        self.assertEqual(N.phase_of(final)[1], [1, 2])
        for kwargs, message in ((dict(done=False), 'no \\[DONE\\]'), (dict(usage=False), 'usage'),
                                (dict(error=True), 'generation error')):
            with self.assertRaisesRegex(ValueError, message):
                N.read_sse(sse_lines(nemotron_phase(), [1, 2], **kwargs), clock, 0)
        _, final, _ = N.read_sse(sse_lines(nemotron_phase(), [1, 2]), clock, 0)
        final['strix']['phases'].append(nemotron_phase())
        with self.assertRaisesRegex(ValueError, 'one engine receipt'):
            N.phase_of(final)
        _, final, _ = N.read_sse(sse_lines(nemotron_phase(), []), clock, 0)
        with self.assertRaisesRegex(ValueError, 'generated IDs'):
            N.phase_of(final)

    def run_arms(self, on_server=None, profile='default', levels='1,4'):
        texts = {'chat': 'Hello there', 'code': 'Write a function'}
        prompts = [{'id': x, 'tokens': 3, 'sha256': k.id_hash([i + 1, 2, 3]), 'construction': 'test-only'}
                   for i, x in enumerate(texts)]
        k.dump(self.out / 'prompts.json', prompts)
        env = {'BENCH_SELECTED_PROFILE': profile, 'BENCH_ONLY': 'exactness,serve', 'BENCH_REPEATS': '1',
               'BENCH_WARMUP': '0', 'BENCH_GEN_TOKENS': '4'}
        for arm, server, arm_levels in (('off', FakeServer('off', profile), '1'), ('on', on_server or FakeServer('on', profile), levels)):
            (self.out / ('server-' + arm + '.log')).write_text(N.READY + '\n', encoding='utf-8')
            with patch.object(N.urllib.request, 'urlopen', server), patch.object(k, 'prompt_texts', return_value=texts), \
                    patch.object(N.time, 'sleep'), patch.dict(k.os.environ, env):
                N.serve_arm(k, self.out, 18880, arm, arm_levels)
        fixture = k.read(HERE / 'fixtures/results.json')
        identity = {key: value for key, value in fixture.items() if key not in ('measurements', 'exactness', 'prompts')}
        identity.update(engine='nemotron', params={'only': ['exactness', 'serve'], 'profile': profile})
        k.dump(self.out / 'identity.json', identity)

    def test_arms_end_to_end_exactness(self):
        self.run_arms()
        k.finalize(self.out)
        d = k.read(self.out / 'results.json')
        self.assertEqual(d['status'], 'complete')
        modes = {(e['prompt_id'], e['mode']): e for e in d['exactness']}
        self.assertEqual(set(modes), {(p, m) for p in ('chat', 'code') for m in ('plain', 'spec', 'spec-c4')})
        # Missing references are NO-REFERENCE, never EXACT.
        self.assertEqual({e['verdict'] for e in d['exactness']}, {'NO-REFERENCE'})
        spec = modes[('chat', 'spec')]
        self.assertTrue(spec['plain_tokens_equal'] and spec['plain_state_equal'])
        self.assertEqual(spec['draft_witness']['status'], 'drafted')
        self.assertTrue(modes[('code', 'spec-c4')]['c1_equal'])
        http = k.read(self.out / 'http-measurements.json')
        self.assertEqual({m['prompt_id'] for m in http}, {'http-c1-draft-off', 'http-c1', 'http-c4'})
        self.assertEqual(set(d['nemotron']), {'off', 'on'})

    def test_arms_reject_drafter_divergence_and_missing_witness(self):
        ids = lambda text: [9, 9, 9, 9] if text.startswith('Hello') else [len(text) % 7 + 1, 2, 3, 4]
        self.run_arms(FakeServer('on', ids_for=ids))
        k.finalize(self.out)
        d = k.read(self.out / 'results.json')
        self.assertEqual(d['status'], 'mismatch')
        self.assertEqual(next(e for e in d['exactness'] if e['prompt_id'] == 'chat' and e['mode'] == 'spec')['verdict'], 'MISMATCH')
        # Same IDs but a different committed state is also a mismatch.
        self.tmp.cleanup(); self.out.mkdir()
        state = lambda text: nemotron_phase('on', digest='f' * 64)
        self.run_arms(FakeServer('on', phase_for=state))
        k.finalize(self.out)
        self.assertEqual(k.read(self.out / 'results.json')['status'], 'mismatch')
        self.tmp.cleanup(); self.out.mkdir()
        def no_window(text):
            phase = nemotron_phase('on', digest=format(len(text), '064x')); phase.pop('mtp_confidence_resolved'); return phase
        with self.assertRaisesRegex(ValueError, 'missing draft witness'):
            self.run_arms(FakeServer('on', phase_for=no_window))
        self.tmp.cleanup(); self.out.mkdir()
        idle = lambda text: nemotron_phase('on', digest=format(len(text), '064x'), attempted=0, accepted=0)
        with self.assertRaisesRegex(ValueError, 'positive control'):
            self.run_arms(FakeServer('on', phase_for=idle))

    def test_finalize_rejects_sample_without_witness(self):
        self.run_arms(levels='1')
        rows = [json.loads(x) for x in (self.out / 'samples.jsonl').read_text().splitlines()]
        for row in rows:
            row.pop('draft_witness', None)
        (self.out / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'missing draft witness'):
            k.finalize(self.out)
        d = k.read(HERE / 'fixtures/results.json')
        d['engine'] = 'nemotron'
        d['exactness'][0]['mode'] = 'spec'
        d['exactness'][0].pop('draft_witness', None)
        with self.assertRaisesRegex(ValueError, 'missing draft witness'):
            k.validate(d)

    def test_verify_rows_and_digest_pins(self):
        pins = k.read(HERE.parent / 'refs/nemotron.json')['digest_pins']
        pin = pins['verify-r8']['g4-v1']
        rows = ''.join(json.dumps({'gate': 'verify_rows', 'identity': 'g4-v1', 'verdict': 'EXACT', 'rows': r,
                                   'kv': pin['kv'] if r == 8 else format(r, '064x')}) + '\n' for r in range(1, 9))
        check = N.parse_verify('{"profile":{}}\n' + rows, 'default')
        self.assertEqual(check['kv'], pin['kv'])
        with self.assertRaisesRegex(ValueError, 'rows 1..8'):
            N.parse_verify(rows.split('\n', 1)[1], 'default')
        with self.assertRaisesRegex(ValueError, 'differs from the profile'):
            N.parse_verify(rows, 'fast')
        files = [{'role': N.Q4, 'name': 'Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf', 'sha256': pin['model_sha256']}]
        self.assertEqual(N.digest_verdict(check, pins, files)['verdict'], 'EXACT')
        self.assertEqual(N.digest_verdict(dict(check, kv='0' * 64), pins, files)['verdict'], 'MISMATCH')
        self.assertEqual(N.digest_verdict(check, pins, [dict(files[0], sha256='1' * 64)])['verdict'], 'NO-REFERENCE')
        self.assertEqual(N.digest_verdict(check, pins, [dict(files[0], sha256=None)])['verdict'], 'NO-REFERENCE')
        self.assertEqual(N.digest_verdict(dict(check, identity='g4-exl3-v3'), pins, files)['verdict'], 'NO-REFERENCE')

    def test_reference_key_uses_one_numerical_projection(self):
        inv = dict(build_options={}, binary_sha256s=[], model_files=[], repo_commit='0' * 40,
                   tracked_dirty=False, hardware=dict(rocm_version=None, hip_version=None, kernel='test'))
        k.dump(self.out / 'local-inventory.json', inv)
        bits = N.numerics(nemotron_phase(), 'models')
        for arm in ('off', 'on'):
            k.dump(self.out / ('effective-serve-' + arm + '.json'), {'environment': {}, 'arguments': [], 'settings': {'numerics': bits}})
        env = {'BENCH_PROFILE': 'g4-v1-q4km:default', 'BENCH_SUITE': 'quick', 'BENCH_PROMPTS': 'kit'}
        with patch.object(k, 'ENGINE', 'nemotron'), patch.dict(k.os.environ, env):
            k.initialize(self.out)
            with_drafter = k.read(self.out / 'identity.json')['reference_sha256']
            (self.out / 'effective-serve-on.json').unlink()
            k.initialize(self.out)
            self.assertEqual(k.read(self.out / 'identity.json')['reference_sha256'], with_drafter)
            k.dump(self.out / 'effective-serve-on.json', {'settings': {'numerics': dict(bits, decode_identity='g4-a8-v1')}})
            with self.assertRaisesRegex(ValueError, 'conflated identities'):
                k.initialize(self.out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
