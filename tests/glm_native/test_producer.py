"""CPU cache/policy and real C++ receiver loopback. Artifacts live in --build."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cache', ROOT/'src/glm/rtxpf/producer/cache.py')
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class CacheTests(unittest.TestCase):
    def test_native_integrity_restart_and_manifest_pin(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            p = Path(td)
            (p/'one.bin').write_bytes(b'first blob')
            (p/'two.bin').write_bytes(b'second blob')
            manifest = dict(format='glm-native-cache-v1', blobs={name: dict(bytes=(p/name).stat().st_size, sha256=cache.sha(p/name)) for name in ('one.bin', 'two.bin')})
            (p/'manifest.json').write_bytes(cache.canonical(manifest))
            pin = cache.sha(p/'manifest.json')
            command = [str(BUILD/'glm_native_integrity.exe'), str(p), pin]
            first = json.loads(subprocess.check_output(command, text=True))
            self.assertEqual((first['blobs'], first['verified_hash_hits']), (2, 0))
            second = json.loads(subprocess.check_output(command, text=True))
            self.assertEqual(second['verified_hash_hits'], 2-second['journal_unavailable_blobs'])
            if second['journal_unavailable_blobs']:
                print('INFO: journal unavailable; integrity restart fully re-hashes blobs')
            (p/'one.bin').write_bytes(b'wrong blob')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertNotEqual(subprocess.run(command[:-1]+['0'*64], capture_output=True).returncode, 0)

    def test_indexed_separate_mtp(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            p = Path(td)
            for filename, name in [('trunk.safetensors', 'x'), ('mtp.safetensors', 'y')]:
                header = json.dumps({name: {'dtype': 'I32', 'shape': [], 'data_offsets': [0, 4]}}).encode()
                (p/filename).write_bytes(struct.pack('<Q', len(header))+header+b'\0'*4)
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'trunk.safetensors'}}))
            tensors, files = cache.inventory(p)
            self.assertEqual(set(tensors), {'x', 'y'})
            self.assertEqual([f.name for f in files], ['mtp.safetensors', 'trunk.safetensors'])

    def test_stream_packing(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            path = Path(td)/'small.bin'
            path.write_bytes(struct.pack('<If', 0x83dcd12d, 1.0))
            tensors = {'a': dict(path=path, offset=0, bytes=4, dtype='I32'),
                       'b': dict(path=path, offset=4, bytes=4, dtype='F32')}
            chunks = list(cache.packed_chunks(tensors, ['a', 'b']))
            self.assertEqual([o for _, o, _ in chunks], [0, 4, 256])
            self.assertEqual(b''.join(b for _, _, b in chunks), path.read_bytes()[:4]+b'\0'*252+path.read_bytes()[4:])
            path.write_bytes(struct.pack('<If', 0x83dcd12d, float('nan')))
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                list(cache.packed_chunks(tensors, ['a', 'b']))
            path.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'truncated'):
                list(cache.packed_chunks(tensors, ['a']))

    def test_configuration(self):
        cfg = dict(hidden_size=4096, vocab_size=154880, num_hidden_layers=45, hc_mult=4,
                   q_lora_rank=1536, kv_lora_rank=512, num_attention_heads=64,
                   qk_nope_head_dim=256, qk_rope_head_dim=0, v_head_dim=256,
                   index_n_heads=32, index_head_dim=128, index_kpool=4, index_topk=2048,
                   n_routed_experts=288, moe_intermediate_size=2048, intermediate_size=12288,
                   num_nextn_predict_layers=1, mlp_layer_types=['dense']*3+['sparse']*42,
                   layer_types=['linear_attention' if i % 4 != 3 else 'deepseek_sparse_attention' for i in range(45)],
                   linear_attn_config=dict(num_heads=64, head_dim=128, gate_lower_bound=-5.0,
                                           short_conv_kernel_size=4, kda_layers=[i for i in range(45) if i % 4 != 3]))
        cache.validate_config(cfg)
        cfg['layer_types'][3] = 'full_attention'
        with self.assertRaisesRegex(ValueError, 'configuration'):
            cache.validate_config(cfg)

    def test_inventory_and_incomplete(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            p = Path(td)
            header = json.dumps({'x': {'dtype': 'I32', 'shape': [], 'data_offsets': [0, 4]}}).encode()
            (p/'one.safetensors').write_bytes(struct.pack('<Q', len(header))+header+b'\0'*4)
            tensors, _ = cache.inventory(p)
            self.assertEqual(tensors['x']['bytes'], 4)
            with self.assertRaisesRegex(ValueError, 'incomplete full model'):
                cache.build(p, p/'out')
            self.assertFalse((p/'out').exists())

    def test_restart_validation(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            p = Path(td)
            blob = p/'base.bin'
            blob.write_bytes(b'\0'*4)
            sources = {'fixture': 'compressed'}
            m = dict(cache.VERSION, sources=sources, model_sha256=hashlib.sha256(cache.canonical(sources)).hexdigest(), blobs={'base.bin': dict(bytes=blob.stat().st_size, sha256=cache.sha(blob))},
                     tensors={'x': dict(blob='base.bin', offset=0, bytes=4, shape=[], dtype='I32')})
            m['cache_key'] = hashlib.sha256(cache.canonical(dict(cache.VERSION, model_sha256=m['model_sha256']))).hexdigest()
            (p/'manifest.json').write_bytes(cache.canonical(m))
            with patch.object(cache, 'required_schema', return_value={'x': ('I32', [])}):
                self.assertEqual(cache.verify(p), m)
                # No source model exists. A cache hit only validates cached blobs.
                self.assertEqual(cache.build(p/'missing-model', p), m)
                blob.write_bytes(b'\1'*4)
                with self.assertRaisesRegex(ValueError, 'integrity'):
                    cache.verify(p)

    def test_schema(self):
        s = cache.required_schema()
        self.assertIn('model.language_model.layers.45.eh_proj.trellis', s)
        self.assertEqual(sum('.mlp.experts.287.down_proj.trellis' in k for k in s), 43)
        self.assertEqual(sum('hc_attn_fn' in k for k in s), 45)


def loopback(rank=0, cases=('success', 'abort', 'corrupt', 'duplicate'), version=1):
    exe = BUILD/'glm_native_receiver.exe'
    subprocess.run([str(exe)], check=True)
    with tempfile.TemporaryDirectory(dir=BUILD) as td:
        td = Path(td)
        (td/'key').write_bytes(bytes(range(32)))
        prompt = hashlib.sha256(struct.pack('<6I', *range(6))).hexdigest()
        model = '22'*32
        expected = hashlib.sha256()
        for layer in range(46):
            # wire version 2 ('DSA once'): rank 1 receives the KDA components only
            components = [(0, 32*65536), (1, 32*4608)] if layer < 45 and layer % 4 != 3 else [] if version == 2 and rank == 1 else [(2, 5*1024), (3, 5*256), (4, 5*256), (5, 512)]
            for c, n in components:
                expected.update(bytes([(layer*7+c)%256])*n)
        env = dict(os.environ, GLM_TEST_WIRE_VERSION=str(version))
        for case in cases:
            with socket.socket() as s:
                s.bind(('127.0.0.1', 0))
                endpoint = '127.0.0.1:'+str(s.getsockname()[1])
            receiver = subprocess.Popen([str(exe), endpoint, str(td/'key'), prompt, model, str(rank)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            try:
                for _ in range(100):
                    sender = subprocess.run([str(BUILD/'glm_native_sender.exe'), endpoint, str(td/'key'), prompt, model, case, str(rank)], capture_output=True, text=True, timeout=30, env=env)
                    if 'sender connect' not in sender.stderr:
                        break
                    time.sleep(.02)
                out, err = receiver.communicate(timeout=60)
                if case == 'success':
                    assert sender.returncode == receiver.returncode == 0, (sender.stderr, err)
                    assert expected.hexdigest() in out, out
                    assert 'LOCALLY_SEALED' in sender.stdout
                else:
                    assert receiver.returncode != 0 and 'REJECT' in err, (out, err)
                print('PASS production sender -> node CPU staging:', case, 'rank', rank, 'wire version', version)
            finally:
                if receiver.poll() is None:
                    receiver.kill()
                    receiver.wait()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--cache-only', action='store_true', help='CPU cache checks only; never open sockets')
    a = p.parse_args()
    BUILD = a.build.resolve()
    (BUILD/'tmp').mkdir(exist_ok=True)
    os.environ['TMP'] = os.environ['TEMP'] = str(BUILD/'tmp')
    # Isolate the current-user authority in this test's allowed build tree.
    os.environ['LOCALAPPDATA'] = str(BUILD/'test-localappdata')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CacheTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if a.cache_only:
        raise SystemExit(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        paired = [pool.submit(loopback, rank, ('success',)) for rank in (0, 1)]
        for job in paired:
            job.result()
    loopback(0, ('abort', 'corrupt', 'duplicate'))
    for rank in (0, 1):
        loopback(rank, ('success',), version=2)  # DSA once: rank 0 every component, rank 1 KDA only
    loopback(1, ('extra_dsa', 'premature', 'corrupt'), version=2)  # rank 1 at v2 refuses a DSA frame and an early commit
