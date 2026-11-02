#!/usr/bin/env python3
"""Hardware-free checks of scripts/pair/ds41-pair: config round trip, USB4 link verdicts from a fake sysfs, the
IOMMU and GPU probes, manifest parsing, fetch --verify-only and the smoke reference file."""
import contextlib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'pair' / 'ds41-pair'


def load(test_root):
    os.environ['DS41_PAIR_TEST_ROOT'] = str(test_root)
    loader = importlib.machinery.SourceFileLoader('ds41_pair', str(SCRIPT))
    spec = importlib.util.spec_from_loader('ds41_pair', loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    mod.COLOR = False
    return mod


def put(base, rel, text):
    p = Path(base) / rel.lstrip('/')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


class PairSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'root'
        self.home = Path(self.tmp.name) / 'home'
        self.home.mkdir()
        os.environ['XDG_CONFIG_HOME'] = str(self.home / 'config')
        os.environ['XDG_STATE_HOME'] = str(self.home / 'state')
        os.environ['XDG_DATA_HOME'] = str(self.home / 'data')
        os.environ.pop('STRIX_PAIR_CONFIG', None)
        self.m = load(self.root)
        self.m.FAILS.clear()

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop('DS41_PAIR_TEST_ROOT', None)

    def link(self, name, speed, lanes, peer='node-b'):
        d = f'/sys/bus/thunderbolt/devices/{name}'
        for k, v in (('rx_speed', f'{speed} Gb/s'), ('tx_speed', f'{speed} Gb/s'), ('rx_lanes', str(lanes)),
                     ('tx_lanes', str(lanes)), ('device_name', peer)):
            put(self.root, f'{d}/{k}', v + '\n')

    def test_link_verdicts(self):
        self.link('0-2', '20.0', 2)
        self.link('1-2', '10.0', 2)
        self.link('2-1', '20.0', 1)
        put(self.root, '/sys/bus/thunderbolt/devices/0-0/rx_speed', '20.0 Gb/s\n')  # host router: not a link
        links = {l['name']: l for l in self.m.usb4_links()}
        self.assertEqual(sorted(links), ['0-2', '1-2', '2-1'])
        self.assertEqual(self.m.link_verdict(links['0-2'])[0], 'PASS')
        level, text = self.m.link_verdict(links['1-2'])
        self.assertEqual(level, 'WARN')
        self.assertIn('Gen2', text)
        self.assertIn('re-seat that cable with both nodes on', text)
        level, text = self.m.link_verdict(links['2-1'])
        self.assertEqual(level, 'WARN')
        self.assertIn('re-seat that cable', text)

    def test_gpu_target_decoding(self):
        put(self.root, '/sys/class/kfd/kfd/topology/nodes/0/properties', 'gfx_target_version 0\n')
        put(self.root, '/sys/class/kfd/kfd/topology/nodes/1/properties', 'cpu_cores_count 0\ngfx_target_version 110501\n')
        self.assertEqual(self.m.gpu_targets(), ['gfx1151'])

    def test_config_round_trip(self):
        models = Path(self.tmp.name) / 'models with space'
        with contextlib.redirect_stdout(io.StringIO()):
            self.m.main(['init', '--rank', '1', '--peer', 'user@node-a', '--model-dir', str(models),
                         '--profile', 'fast', '--http-port', '9000'])
        cfg = self.m.load_config(type('A', (), {'config': None})())
        self.assertEqual(cfg['RANK'], '1')
        self.assertEqual(cfg['PEER'], 'user@node-a')
        self.assertEqual(cfg['MODEL_DIR'], str(models.resolve()))
        self.assertEqual(cfg['PROFILE'], 'fast')
        self.assertEqual(cfg['HTTP_PORT'], '9000')
        self.assertEqual(cfg['PEER_MODEL_DIR'], cfg['MODEL_DIR'])
        text = Path(cfg['_path']).read_text()
        self.assertNotIn('PEER_MODEL_DIR=' + str(models.resolve()), text)  # derived defaults stay implicit
        with contextlib.redirect_stdout(io.StringIO()):
            self.m.main(['init', '--profile', 'default'])  # an update keeps the other keys
        cfg = self.m.load_config(type('A', (), {'config': None})())
        self.assertEqual((cfg['RANK'], cfg['PROFILE'], cfg['HTTP_PORT']), ('1', 'default', '9000'))
        moved = Path(self.tmp.name) / 'moved'
        with contextlib.redirect_stdout(io.StringIO()):
            self.m.main(['init', '--model-dir', str(moved)])  # the peer default follows the new directory
        cfg = self.m.load_config(type('A', (), {'config': None})())
        self.assertEqual(cfg['PEER_MODEL_DIR'], str(moved.absolute()))

    def test_iommu_off_fails_check(self):
        models = Path(self.tmp.name) / 'models'
        with contextlib.redirect_stdout(io.StringIO()):
            self.m.main(['init', '--rank', '0', '--model-dir', str(models)])
        put(self.root, '/proc/cmdline', 'root=/dev/x ro amd_iommu=off quiet\n')
        put(self.root, '/proc/cpuinfo', 'model name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\n')
        put(self.root, '/proc/meminfo', 'MemTotal:       131072000 kB\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.m.main(['check', '--no-peer'])
        text = buf.getvalue()
        self.assertEqual(rc, 1)
        self.assertRegex(text, r'FAIL\s+iommu\s+disabled on the kernel command line')
        self.assertNotRegex(text.lower(), r'disable the iommu|set iommu=off')
        self.assertRegex(text, r'FAIL\s+usb4\s+no USB4 link up')

    def test_driver_with_current_module_builds_nothing(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.m.main(['init', '--rank', '0', '--model-dir', str(Path(self.tmp.name) / 'models')])
        put(self.root, '/sys/module/thunderbolt_stream/srcversion', 'ABC\n')
        put(self.root, '/sys/kernel/config/thunderbolt/stream/0-2.0/data0/rx_repost_batch', '0\n')
        put(self.root, '/proc/cmdline', 'ro quiet kho=off\n')
        put(self.root, '/sys/module/ttm/parameters/pages_limit', '30720000\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.m.main(['driver'])
        self.assertEqual(rc, 0)
        self.assertRegex(buf.getvalue(), r'PASS\s+driver\s+the running kernel already has')
        self.assertNotIn('sudo', buf.getvalue())
        put(self.root, '/sys/module/ttm/parameters/pages_limit', '15000000\n')  # default half of RAM: too small
        put(self.root, '/proc/cmdline', 'ro quiet\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.m.main(['driver'])
        text = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn('/etc/modprobe.d/ttm.conf', text)
        self.assertIn('options ttm pages_limit=30720000', next(Path(os.environ['XDG_DATA_HOME']).rglob('ttm.conf')).read_text())
        self.assertIn('kho=off', text)

    def test_manifest_rejects_unsafe_paths(self):
        cache = Path(self.tmp.name) / 'cache'
        cache.mkdir()
        rev = 'a' * 40
        (cache / f'repo@{rev[:12]}.SHA256SUMS').write_text('0' * 64 + '  ../etc/passwd\n')
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.m.manifest('repo', rev, cache)
        (cache / f'repo@{rev[:12]}.SHA256SUMS').write_text('1' * 64 + '  ours-K4/a.safetensors\n' + '2' * 64 + ' *mixB.txt\n')
        entries = self.m.manifest('repo', rev, cache)
        self.assertEqual(entries, {'ours-K4/a.safetensors': '1' * 64, 'mixB.txt': '2' * 64})
        self.assertEqual(self.m.selected(entries, ['ours-K4/*']), ['ours-K4/a.safetensors'])

    def test_fetch_verify_only_uses_manifests(self):
        models = Path(self.tmp.name) / 'models'
        with contextlib.redirect_stdout(io.StringIO()):
            self.m.main(['init', '--rank', '0', '--model-dir', str(models)])
        data = Path(os.environ['XDG_DATA_HOME']) / 'strix-pair' / 'manifests'
        data.mkdir(parents=True)
        good = b'mix-B head\n'
        (models / 'ours-K5').mkdir(parents=True)
        (models / 'ours-K5' / 'index.json').write_bytes(good)
        (models / 'mixB.txt').write_bytes(b'tampered\n')
        for repo, rev, sub, patterns in self.m.jobs_for_pair():
            lines = []
            if repo == self.m.fetch_models.TRELLIS:
                lines = [hashlib.sha256(good).hexdigest() + '  ours-K5/index.json',
                         hashlib.sha256(b'mix\n').hexdigest() + '  mixB.txt', '3' * 64 + '  ours-K4/x']
            else:
                lines = ['4' * 64 + '  ' + p.replace('*', 'x') for p in patterns]
            (data / f'{repo}@{rev[:12]}.SHA256SUMS').write_text('\n'.join(lines) + '\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.m.main(['fetch', '--verify-only', '--only', 'ours-K5/*', '--only', 'mixB.txt'])
        text = buf.getvalue()
        self.assertEqual(rc, 1)
        self.assertIn('ok       ours-K5/index.json', text)
        self.assertIn('MISMATCH mixB.txt', text)
        self.assertNotIn('ours-K4/x', text)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.m.main(['fetch', '--verify-only', '--only', 'ours-K5/*'])
        self.assertIn('ok       ours-K5/index.json (cached)', buf.getvalue())

    def test_smoke_reference_is_complete(self):
        ref = json.loads((ROOT / 'scripts' / 'pair' / 'ds41-smoke-reference.json').read_text(encoding='utf-8'))
        self.assertEqual(ref['schema'], 'ds41-pair-smoke-v1')
        self.assertIn('default', ref['profiles'])
        self.assertIsInstance(ref['token_ids'], list)
        self.assertTrue(0 < len(ref['token_ids']) <= ref['max_tokens'])
        digest = hashlib.sha256(json.dumps(ref['token_ids']).encode()).hexdigest()[:16]
        self.assertEqual(digest, ref['token_ids_sha256_16'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
