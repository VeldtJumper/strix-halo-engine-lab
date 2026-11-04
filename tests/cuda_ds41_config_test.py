"""CPU-only resolved configuration contract; no models or CUDA devices required.

Optional STRIX_DS41_CONFIG_PRIOR / STRIX_DS41_CONFIG_KEEP supply private measured
inputs. Otherwise equivalent synthetic parser fixtures are made in a temp directory.
"""
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

exe = sys.argv[1]
baseline_exe = os.environ.get("STRIX_DS41_CONFIG_BASELINE_EXE")
baseline_comparisons = 0
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ('STRIX_DS41_RTX_BASELINE', 'DG_LOAD_SERIAL', 'STRIX_SESSION_CACHE', 'STRIX_SESSION_DISK', 'STRIX_DS41_FILE_TIER',
            'STRIX_DS41_CE_PACING', 'STRIX_DS41_VRAM_TOTAL_GIB', 'STRIX_DS41_RAM_TOTAL_GIB', 'STRIX_DS41_WB_STREAM',
            'STRIX_DS41_PROFILE', 'STRIX_DS41_PROFILE_ALLOW_PENDING', 'STRIX_DS41_DRAFTER', 'STRIX_WEIGHT_DIRS'):
    env.pop(key, None)


def call(args=(), extra=None, base=None, ok=True, profile=False, raw=False):
    """The resolved configuration; its 'profile' receipt is checked on its own (profile=True keeps it)."""
    e = dict(env, **(extra or {}))
    p = subprocess.run([exe, *(base or ['MODEL', '--import-state', 'S', '--ref', 'R', '--draft', 'D']),
                        '--print-config', *args], env=e, capture_output=True, text=True)
    if not ok:
        assert p.returncode != 0, p.stdout
        return p
    assert p.returncode == 0, (p.returncode, p.stderr)
    # The public baseline has no session inspection fields or session switches.
    if baseline_exe and not any(a.startswith('--session-') or a == '--serve-session-mib' for a in args):
        global baseline_comparisons
        reference = subprocess.run([baseline_exe, *(base or ['MODEL', '--import-state', 'S', '--ref', 'R', '--draft', 'D']),
                                    '--print-config', *args], env=e, capture_output=True, text=True)
        ordinary_output = re.sub(r',"session_cache":.*}\n?$', '}\n', p.stdout)
        assert (p.returncode, ordinary_output, p.stderr) == (reference.returncode, reference.stdout, reference.stderr), args
        baseline_comparisons += 1
    lines = p.stdout.splitlines()
    assert len(lines) == 1, p.stdout
    if raw:
        return p
    config = json.loads(lines[0])
    receipt = config.pop('profile')
    assert receipt['app'] == 'replica', receipt
    return dict(config, profile=receipt) if profile else config


def diff(a, b):
    return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


with tempfile.TemporaryDirectory(prefix='ds41-config-', dir=Path.cwd()) as tmp:
    root = Path(tmp)
    vocab = root / 'vocab'
    vocab.mkdir()
    seed = root / 'seed'
    header = (Path(__file__).resolve().parents[1] / 'include/strix/deepseek41_model.hpp').read_text()
    sha = re.search(r'PinnedFileKind::target_q2,.*?"([0-9a-f]{64})"', header, re.S)[1]
    seed.write_text(f'ds41-residency-v1 sha256={sha} layers=40 experts=384 n=3\n0 5 15359\n')
    env.update(STRIX_DS41_DRAFT_VOCAB=str(vocab), STRIX_DS41_RESIDENCY=str(seed))
    default = call()
    # Make portable equivalent inputs. Private receipt inputs can replace them at test time.
    stats = root / 'keep.jsonl'
    counts = []
    for ids in default['draft_keep_ids']:
        row = [0] * 128
        for rank, expert in enumerate(ids):
            row[expert] = 128 - rank
        counts += row
    stats.write_text(json.dumps({'event': 'route_stats', 'counts': counts}, separators=(',', ':')) + '\n')
    prior = root / 'prior'
    seen, hits = default['controller_prior'].split('/')
    seen = [int(x) for x in seen.split(',') if x]
    hits = [int(x) for x in hits.split(',') if x]
    prior.write_text('ds41-dspark-survival-v1 bins 8\n' + ''.join(
        f'd{d+1} ' + ' '.join(map(str, seen[8*d:8*d+8] + hits[8*d:8*d+8])) + '\n' for d in range(5)))
    prior_arg = os.environ.get('STRIX_DS41_CONFIG_PRIOR', str(prior) + ':0')
    keep_arg = os.environ.get('STRIX_DS41_CONFIG_KEEP', str(stats))
    production = call(['--arena-gib', '74.8', '--spares', '96', '--host-spares', '48', '--predict', '8',
                       '--pressure-spares', '8', '--max-rows', '6', '--window', 'controller', '--controller-scope', 'chat',
                       '--draft-vocab', 'auto:' + str(vocab), '--draft-keep', '64', '--draft-keep-from', keep_arg,
                       '--controller-prior', prior_arg, '--controller-costs', '14.8,18.7,26.6,28.9,33.2,37.3,1.6',
                       '--controller-explore', '0', '--wb-defer'])
    assert default == production, {k: (default[k], production[k]) for k in diff(default, production)}
    assert default['census'] is False and default['evict'] in ('auto', 'lru')
    assert default['ce_pacing'] is True
    assert diff(default, call(['--no-ce-pacing'])) == {'ce_pacing'}
    assert call(extra={'STRIX_DS41_CE_PACING': '0'})['ce_pacing'] is False
    assert call(['--ce-pacing'], extra={'STRIX_DS41_CE_PACING': '0'})['ce_pacing'] is True
    assert call(['--no-ce-pacing'], extra={'STRIX_DS41_CE_PACING': '1'})['ce_pacing'] is False
    assert call(extra={'STRIX_DS41_RTX_BASELINE': '1'})['ce_pacing'] is False
    assert call(['--ce-pacing', '--baseline'])['ce_pacing'] is False
    assert call(['--baseline', '--ce-pacing'])['ce_pacing'] is True
    assert default['wb_stream'] is True
    assert diff(default, call(['--no-wb-stream'])) == {'wb_stream'}
    assert call(extra={'STRIX_DS41_WB_STREAM': '0'})['wb_stream'] is False
    assert call(extra={'STRIX_DS41_WB_STREAM': '1'})['wb_stream'] is True
    assert call(['--wb-stream'], extra={'STRIX_DS41_WB_STREAM': '0'})['wb_stream'] is True
    assert call(['--no-wb-stream'], extra={'STRIX_DS41_WB_STREAM': '1'})['wb_stream'] is False
    assert call(extra={'STRIX_DS41_RTX_BASELINE': '1'})['wb_stream'] is False
    assert call(['--wb-stream', '--baseline'])['wb_stream'] is False
    assert call(['--baseline', '--wb-stream'])['wb_stream'] is True
    assert diff(call(['--no-ce-pacing']), call(['--no-ce-pacing', '--no-wb-stream'])) == {'wb_stream'}
    # Three-tier placement: the file tier (instead of the refusal) is on by default, the RAM tier auto, no clean copies.
    assert default['file_tier'] is True and default['ram_expert_gib'] == 'auto' and default['ram_inclusive_gib'] == '0'
    assert default['disk_prefetch'] is False and default['disk_staging'] == 8
    assert diff(default, call(['--no-file-tier'])) == {'file_tier'}
    assert call(extra={'STRIX_DS41_FILE_TIER': '0'})['file_tier'] is False
    assert call(['--file-tier'], extra={'STRIX_DS41_FILE_TIER': '0'})['file_tier'] is True
    assert call(['--file-tier', '--baseline'])['file_tier'] is False
    assert call(['--baseline', '--file-tier'])['file_tier'] is True
    assert diff(default, call(['--ram-expert-gib', '64'])) == {'ram_expert_gib'}
    assert call(['--ram-expert-gib', '64'])['ram_expert_gib'] == '64'
    assert call(['--ram-expert-gib', '64', '--ram-expert-gib', 'auto']) == default
    assert diff(default, call(['--ram-inclusive-gib', '8'])) == {'ram_inclusive_gib'}
    assert call(['--ram-inclusive-gib', 'auto'])['ram_inclusive_gib'] == 'auto'
    assert call(['--ram-inclusive-gib', '8', '--baseline'])['ram_inclusive_gib'] == '0'
    assert diff(default, call(['--disk-prefetch'])) == {'disk_prefetch'}
    assert call(['--disk-prefetch', '--baseline'])['disk_prefetch'] is False
    assert diff(default, call(['--disk-staging', '4'])) == {'disk_staging'}
    # file reads in flight: default 4 (measured), env / flag, --baseline = one at a time
    assert default['disk_queue'] == 'auto'
    assert call(extra={'STRIX_DS41_DISK_QUEUE': 'auto'})['disk_queue'] == 'auto'
    assert call(['--disk-queue', '4'])['disk_queue'] == 4
    assert diff(default, call(['--disk-queue', '1'])) == {'disk_queue'}
    assert call(extra={'STRIX_DS41_DISK_QUEUE': '16'})['disk_queue'] == 16
    assert call(['--disk-queue', '2'], extra={'STRIX_DS41_DISK_QUEUE': '16'})['disk_queue'] == 2
    assert call(['--baseline'])['disk_queue'] == 1
    assert call(['--baseline', '--disk-queue', '4'])['disk_queue'] == 4
    assert call(['--vram-expert-gib', '48']) == call(['--arena-gib', '48'])
    assert call(['--vram-expert-gib', 'auto']) == call(['--arena-gib', 'auto'])
    # whole-device budgets: default none (today); flags / env; total caps never touch the arena / RAM settings here (the
    # plan needs the measured fixed parts at load); --baseline keeps them; bad values refused
    assert default['vram_total_gib'] == 'none' and default['ram_total_gib'] == 'none' and default['budget_receipt'] is False
    assert default['vram_margin_gib'] == 1.5 and default['ram_margin_gib'] == 1 and default['budget_sample_ms'] == 100
    assert default['budget_plan'].startswith('none')
    capped = call(['--vram-total-gib', '24', '--ram-total-gib', '32'])
    assert diff(default, capped) == {'vram_total_gib', 'ram_total_gib', 'budget_receipt', 'budget_plan', 'drafter_select'}
    assert capped['vram_total_gib'] == '24' and capped['ram_total_gib'] == '32' and capped['budget_receipt'] is True
    assert capped['budget_plan'].startswith('at load: arena slots = min(--vram-expert-gib, (24 GiB')
    assert call(extra={'STRIX_DS41_VRAM_TOTAL_GIB': '24', 'STRIX_DS41_RAM_TOTAL_GIB': '32'}) == capped
    assert call(['--vram-total-gib', '20'], extra={'STRIX_DS41_VRAM_TOTAL_GIB': '24'})['vram_total_gib'] == '20'
    assert call(['--vram-total-gib', 'none'], extra={'STRIX_DS41_VRAM_TOTAL_GIB': '24'})['vram_total_gib'] == 'none'
    assert call(['--vram-total-gib', '24', '--ram-total-gib', '32', '--baseline'])['vram_total_gib'] == '24'
    assert call(['--vram-total-gib', '24', '--vram-expert-gib', '8'])['arena_gib'] == 8
    assert diff(default, call(['--budget-receipt'])) == {'budget_receipt'}
    assert diff(default, call(['--vram-margin-gib', '2', '--ram-margin-gib', '0.5'])) == {'vram_margin_gib', 'ram_margin_gib'}
    assert diff(default, call(['--budget-sample-ms', '0'])) == {'budget_sample_ms'}
    for bad in (['--vram-total-gib', '-1'], ['--vram-total-gib', 'x'], ['--ram-total-gib', 'auto'], ['--vram-margin-gib', 'auto']):
        call(bad, ok=False)
    call(extra={'STRIX_DS41_VRAM_TOTAL_GIB': 'lots'}, ok=False)
    for bad in (['--ram-expert-gib', '-1'], ['--ram-expert-gib', 'x'], ['--ram-inclusive-gib', 'nan'],
                ['--disk-staging', '0'], ['--vram-expert-gib', 'big']):
        call(bad, ok=False)
    assert diff(default, call(['--census'])) == {'census'}
    for policy, hook in [('scored', 'ts_dg_set_policy'), ('auto', 'ts_dg_end_rows')]:
        probe = subprocess.run([exe, 'MODEL', '--print-config', '--evict', policy], env=env, capture_output=True, text=True)
        if probe.returncode == 0:
            assert call(['--evict', policy])['evict'] == policy
            assert call(['--evict', policy, '--baseline'])['evict'] == 'lru'
        else:
            assert hook in probe.stderr, probe.stderr
            assert default['evict'] == 'lru'
            call(['--evict', policy], ok=False)
    assert diff(default, call(['--evict', 'lru'])) == ({'evict'} if default['evict'] != 'lru' else set())
    call(['--evict', 'invalid'], ok=False)
    old = call(['--baseline'])
    expected = dict(dsrtxk=False, fuse=False, records_inline=True, router_form=0, qakv_s1=False,
                    q8_r1=False, q8_r1_pad=False, draft_keep=0, draft_keep_ids=[], controller_prior='none',
                    controller_costs='none', wb_defer=False, controller_scope='run', host_spares=96,
                    arena_gib=71.5, draft_vocab='full', warmup_graphs='none', residency_seed='none',
                    residency_save='none', engram_cache_mb=0, load_serial=True, attn_warps=0, max_rows=1, pressure_spares=4, lean_state=False, arena_auto=False, evict='lru', ce_pacing=False, wb_stream=False,
                    file_tier=False, disk_queue=1, drafter='on',
                    drafter_select=dict(default['drafter_select'], mode='on', reason='--drafter on; not loaded: no speculative phase'))
    for key, value in expected.items():
        assert old[key] == value, (key, old[key], value)
    for key in ('decode_fuse_hc', 'decode_fuse_shared', 'decode_router_v2', 'decode_ob_one',
                'decode_shared_fork', 'decode_attn_fork', 'decode_ob_expand', 'decode_moe_expand', 'moe_v1b3', 'hc_cluster'):
        expected[key] = False
    assert old == dict(default, **expected), diff(old, dict(default, **expected))
    assert call(extra={'STRIX_DS41_RTX_BASELINE': '1'}) == old
    for flags, keys in [(['--dsrtxk-off'], {'dsrtxk', 'router_form', 'qakv_s1', 'attn_warps', 'decode_router_v2', 'decode_ob_one', 'decode_shared_fork', 'decode_attn_fork', 'decode_ob_expand', 'decode_moe_expand', 'moe_v1b3', 'hc_cluster'}),
                        (['--fuse-off'], {'fuse', 'decode_fuse_hc', 'decode_fuse_shared'}), (['--records-inline'], {'records_inline'}),
                        (['--router-pf0'], {'router_form'}), (['--qakv-auto'], {'qakv_s1'}),
                        (['--q8-r1-off'], {'q8_r1'}), (['--q8-r1-pad'], {'q8_r1_pad'}),
                        (['--draft-keep', '0'], {'draft_keep', 'draft_keep_ids', 'arena_gib'}),
                        (['--controller-prior', 'none'], {'controller_prior'}),
                        (['--controller-costs', 'none'], {'controller_costs'}),
                        (['--no-wb-defer'], {'wb_defer'}), (['--engram-cache-mb', '0'], {'engram_cache_mb'}),
                        (['--draft-vocab', 'full'], {'draft_vocab'}),
                        (['--residency-seed', 'none'], {'residency_seed'}),
                        (['--controller-scope', 'run'], {'controller_scope'}),
                        (['--host-spares', '96'], {'host_spares'}),
                        (['--no-lean-state'], {'lean_state'}), (['--arena-gib', 'auto'], {'arena_auto'})]:
        assert diff(default, call(flags)) == keys, flags
    assert diff(default, call(extra={'DG_LOAD_SERIAL': '1'})) == {'load_serial'}
    assert call(['--baseline'], extra={'DG_LOAD_SERIAL': '0'})['load_serial'] is False
    # Ordering: baseline resets earlier options; later flags override their own items.
    assert call(['--host-spares', '7', '--baseline']) == old
    assert call(['--baseline', '--host-spares', '7'])['host_spares'] == 7
    assert call(['--arena-gib', 'auto', '--baseline'])['arena_auto'] is False
    assert call(['--baseline', '--lean-state'])['lean_state'] is True
    assert call(['--host-spares', '7'], extra={'STRIX_DS41_RTX_BASELINE': '1'})['host_spares'] == 7
    for flags, key, value in [(['--dsrtxk'], 'dsrtxk', True), (['--fuse'], 'fuse', True),
                              (['--q8-r1'], 'q8_r1', True), (['--router-pf'], 'router_form', 2),
                              (['--qakv-s1'], 'qakv_s1', True), (['--records-deferred'], 'records_inline', False),
                              (['--wb-defer'], 'wb_defer', True), (['--draft-keep', '64'], 'draft_keep', 64)]:
        assert call(['--baseline', *flags])[key] == value, flags
    assert diff(default, call(['--arena-gib', '12'])) == {'arena_gib'}
    assert call(['--draft-keep', '32', '--draft-keep-from', str(stats)])['arena_gib'] == 76.5
    assert call(['--draft-keep', '0'])['arena_gib'] == 71.5
    assert call(base=['MODEL', '--import-state', 'S', '--ref', 'R'])['arena_gib'] == 80
    assert call(['--arena-gib', '12', '--draft-keep', '0'])['arena_gib'] == 12
    worker = ['--serve', str(root / 'worker'), '--max-positions', '4096']
    persistent = call(worker)
    assert persistent['session_cache'] and persistent['session_disk']
    assert persistent['session_route'] == 'rtx-tp1-decode-rows'
    assert persistent['session_explicit_opt_in'] is False
    disabled_policy = call([*worker, '--session-cache', 'off'])
    assert {k: v for k, v in persistent.items() if not k.startswith('session_')} == {
        k: v for k, v in disabled_policy.items() if not k.startswith('session_')}
    assert persistent['session_off_reason'] == ''
    assert call([*worker, '--session-cache', 'on'])['session_cache']
    assert call([*worker, '--session-cache', 'on'])['session_explicit_opt_in'] is True
    assert call([*worker, '--serve-session-mib', '512'])['session_cache']
    assert persistent['session_max_parked'] == 8
    assert 0 < persistent['session_ram_bytes'] <= 8 * 1024**3
    assert call([*worker, '--session-cache', 'off'])['session_cache'] is False
    assert call(worker, extra={'STRIX_SESSION_CACHE': '0'})['session_cache'] is False
    assert call([*worker, '--baseline'])['session_cache'] is False
    assert call([*worker, '--session-disk', 'off'])['session_disk'] is False
    assert call(worker, extra={'STRIX_SESSION_DISK': '0'})['session_disk'] is False
    custom = call([*worker, '--session-ram-bytes', '1048576', '--session-max-parked', '2',
                   '--session-disk-bytes', '2097152', '--session-dir', str(root / 'sessions')])
    assert (custom['session_ram_bytes'], custom['session_max_parked'], custom['session_disk_bytes']) == (1048576, 2, 2097152)
    assert custom['session_dir'] == str(root / 'sessions')
    # Operational session controls may change only session policy fields.
    numeric = lambda cfg: {k: v for k, v in cfg.items() if not k.startswith('session_')}
    for flags in (['--session-cache', 'off'], ['--session-cache', 'on'],
                  ['--session-disk', 'off'], ['--session-ram-bytes', '1'],
                  ['--session-max-parked', '0'], ['--session-disk-bytes', '1']):
        assert numeric(call([*worker, *flags])) == numeric(persistent), flags
    assert persistent['warmup_graphs'] == '0,8'
    assert persistent['residency_save'] == str(seed)
    for flags, key in [(['--warmup-graphs', 'none'], 'warmup_graphs'),
                       (['--residency-save', 'none'], 'residency_save')]:
        assert diff(persistent, call([*worker, *flags])) == {key}
    valid_seed = seed.read_text()
    model_seed = root / 'dummy.gguf.residency'
    model_seed.write_text(valid_seed)
    implicit = call(extra={'STRIX_DS41_RESIDENCY': ''}, base=[str(root / 'dummy.gguf'), '--draft', 'D'])
    assert implicit['residency_seed'] == 'auto:' + str(model_seed)
    assert call([*worker, '--baseline'])['residency_save'] == 'none'
    assert call([*worker, '--baseline'])['warmup_graphs'] == 'none'
    for invalid in (valid_seed.replace('layers=40', 'layers=39'), valid_seed.replace('0 5 15359', '15360'),
                    valid_seed.replace('0 5 15359', 'garbage'), valid_seed.replace('0 5 15359', '')):
        seed.write_text(invalid)
        assert call()['residency_seed'] == 'none'
        call(['--residency-seed', str(seed)], ok=False)
    seed.write_text('invalid\n')
    assert call()['residency_seed'] == 'none'
    call(['--residency-seed', str(seed)], ok=False)
    seed.unlink()
    assert call()['residency_seed'] == 'none'
    assert call(extra={'STRIX_DS41_DRAFT_VOCAB': str(root / 'missing')})['draft_vocab'] == 'full'
    assert call(['--draft-vocab', 'full'])['draft_vocab'] == 'full'
    sibling = root / 'ds41-draft-vocab'
    sibling.mkdir()
    assert call(extra={'STRIX_DS41_DRAFT_VOCAB': str(root / 'missing')}, base=['MODEL', '--draft', str(root / 'D.gguf')])['draft_vocab'] == 'auto:' + str(sibling)
    assert call(extra={'STRIX_DS41_DRAFT_VOCAB': ''}, base=['MODEL', '--draft', str(root / 'D.gguf')])['draft_vocab'] == 'auto:' + str(sibling)
    # --profile default|fast|quality: default (and no --profile) = today; quality = default; fast = default + pending rows
    # listed (not used); per-lever options / variables / --baseline stay explicit overrides above the profile
    today_levers = {'lean_state', 'q8_r1', 'dsrtxk', 'fuse', 'decode_forms', 'evict_auto', 'controller', 'draft_keep',
                    'draft_vocab_auto', 'wb_defer', 'ce_pacing', 'wb_stream', 'file_tier', 'residency_auto', 'warmup_graphs',
                    'engram_cache'}
    base_raw = call(raw=True).stdout
    for name in ('default', 'quality'):  # byte for byte; the receipt names the profile asked for
        same = lambda out: out.replace('"profile":{"profile":"' + name + '"', '"profile":{"profile":"default"', 1)
        assert same(call(['--profile', name], raw=True).stdout) == base_raw, name
        assert same(call(extra={'STRIX_DS41_PROFILE': name}, raw=True).stdout) == base_raw, name
    r = call(profile=True)['profile']
    assert r['profile'] == 'default' and r['qualification'] == 'qualified' and r['unqualified'] == []
    assert r['pending'] == ['prompt_layer_major']  # PRO-qualified, wanted in every profile, not in this build
    assert set(r['levers']) == today_levers | {'prompt_layer_major'}, set(r['levers']) ^ today_levers
    for k, v in r['levers'].items():
        if k == 'prompt_layer_major':
            assert v['status'] == 'pending' and v['value'] == 'off' and v['wanted'] == 'on' and v['effective'] == 'off', v
            assert 'owner review (layer-major' in v['source'], v
            continue
        assert v['value'] == 'on' and v['status'] == 'profile' and v['kind'] == 'exact' and v['source'], (k, v)
        assert v['effective'] in ('on', 'on (serve only)'), (k, v)
    assert 'prompt_layer_major pending, not used' in call(raw=True).stderr
    fast = call(['--profile', 'fast'], profile=True)
    assert dict(fast, profile=None) == dict(call(), profile=None)  # every qualified fast lever is today's default
    fr = fast['profile']
    assert fr['profile'] == 'fast' and fr['pending'] == ['prompt_layer_major'] and fr['qualification'] == 'qualified'
    lm = fr['levers']['prompt_layer_major']
    assert lm['status'] == 'pending' and lm['value'] == 'off' and lm['wanted'] == 'on' and lm['source'].startswith('pending:')
    assert call(extra={'STRIX_DS41_PROFILE': 'fast'}, profile=True)['profile'] == fr
    assert call(['--profile', 'default'], extra={'STRIX_DS41_PROFILE': 'fast'}, raw=True).stdout == base_raw
    assert 'prompt_layer_major pending, not used' in call(['--profile', 'fast'], raw=True).stderr  # never silently used
    # --allow-pending: nothing pending is runnable in this build -> still pending, the receipt says allow_pending
    fa = call(['--profile', 'fast', '--allow-pending'], profile=True)['profile']
    assert fa['allow_pending'] is True and fa['pending'] == ['prompt_layer_major'] and fa['qualification'] == 'qualified'
    assert call(['--allow-pending'], profile=True)['profile']['allow_pending'] is True
    assert call(extra={'STRIX_DS41_PROFILE_ALLOW_PENDING': '1'}, profile=True)['profile']['allow_pending'] is True
    # precedence: CLI > env > profile; --baseline in order; STRIX_DS41_RTX_BASELINE=1 over the levers' variables
    def lv(args=(), extra=None):
        return call(args, extra=extra, profile=True)['profile']['levers']
    assert lv(['--profile', 'fast', '--no-wb-stream'])['wb_stream'] == dict(lv()['wb_stream'], value='off', status='explicit-cli', effective='off')
    assert lv(extra={'STRIX_DS41_WB_STREAM': '0'})['wb_stream']['status'] == 'explicit-env'
    assert lv(['--wb-stream'], extra={'STRIX_DS41_WB_STREAM': '0'})['wb_stream']['effective'] == 'on'
    b = lv(['--profile', 'fast', '--baseline'])
    assert b['prompt_layer_major']['status'] == 'pending'
    assert all(b[k]['status'] == 'baseline-cli' and b[k]['effective'] in ('off', 'lru') for k in today_levers), b
    assert lv(['--baseline', '--wb-stream'])['wb_stream']['status'] == 'explicit-cli'
    be = lv(extra={'STRIX_DS41_RTX_BASELINE': '1'})
    assert all(be[k]['status'] == 'baseline-env' for k in today_levers) and be['prompt_layer_major']['status'] == 'pending'
    assert call(['--profile', 'fast', '--baseline']) == old  # --baseline still restores the reference forms
    assert lv(['--evict', 'lru'])['evict_auto'] == dict(lv()['evict_auto'], value='lru', status='explicit-cli', effective='lru')
    assert lv(['--controller-prior', 'none'])['controller']['effective'] == 'custom'
    for bad in (['--profile', 'turbo'], ['--profile']):
        assert 'profile' in call(bad, ok=False).stderr
    call(extra={'STRIX_DS41_PROFILE': 'Fast'}, ok=False)
    assert call([*worker, '--profile', 'fast'], profile=True)['profile']['levers']['warmup_graphs']['effective'] == 'on'
    # --drafter auto|on|off: default auto, env, CLI over env, --baseline = on (in order), bad values refused
    assert default['drafter'] == 'auto'
    ds = default['drafter_select']
    assert ds['mode'] == 'auto' and ds['loaded'] is False and ds['file_tier_expected'] is False and ds['room_bytes'] is None, ds
    assert ds['read_locations'] == 1 and ds['reason'].startswith('auto: no file tier certain'), ds
    assert call(['--drafter', 'auto']) == call()
    assert call(['--drafter', 'off'])['drafter'] == 'off' and call(['--drafter', 'off'])['drafter_select']['reason'].startswith('--drafter off')
    assert call(extra={'STRIX_DS41_DRAFTER': 'off'})['drafter'] == 'off'
    assert call(['--drafter', 'on'], extra={'STRIX_DS41_DRAFTER': 'off'})['drafter'] == 'on'
    assert call(['--drafter', 'auto', '--baseline'])['drafter'] == 'on'
    assert call(['--baseline', '--drafter', 'auto'])['drafter'] == 'auto'
    assert call(extra={'STRIX_DS41_RTX_BASELINE': '1', 'STRIX_DS41_DRAFTER': 'off'})['drafter'] == 'on'
    call(['--drafter', 'maybe'], ok=False)
    call(extra={'STRIX_DS41_DRAFTER': 'yes'}, ok=False)
    # the rule on a synthetic GGUF: 4 MiB of routed experts; caps of ~3 MiB on one read location = a certain file tier
    def gguf(path, tensors):
        def s_(t):
            b = t.encode()
            return struct.pack('<Q', len(b)) + b
        head = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), 0)
        off = 0
        for name, dims in tensors:
            head += s_(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', 0, off)
            n = 4
            for d in dims:
                n *= d
            off += (n + 31) // 32 * 32
        head += b'\0' * (-len(head) % 32)
        path.write_bytes(head + b'\0' * off)
    model = root / 'tiny.gguf'
    gguf(model, [('blk.0.ffn_gate_exps.weight', [1024, 1024]), ('blk.0.attn_norm.weight', [256])])
    tiny = [str(model), '--import-state', 'S', '--ref', 'R', '--draft', str(root / 'D.gguf'), '--serve', str(root / 'w'),
            '--max-positions', '4096', '--vram-total-gib', '0.002', '--ram-total-gib', '0.001', '--vram-margin-gib', '0',
            '--ram-margin-gib', '0']
    off = call(base=tiny)
    sel = off['drafter_select']
    assert sel['loaded'] is False and sel['file_tier_expected'] is True and sel['expert_bytes'] == 4 << 20, sel
    assert sel['room_bytes'] == round(0.002 * 2**30) - 1024 + round(0.001 * 2**30) and sel['read_locations'] == 1, sel
    assert '1 read location' in sel['reason'] and off['arena_gib'] == 80, (sel, off['arena_gib'])  # no drafter: its arena default
    on = call(['--drafter', 'on'], base=tiny)
    assert on['drafter_select']['loaded'] is True and on['drafter_select']['file_tier_expected'] is True and on['arena_gib'] == 74.8
    assert call(['--no-file-tier'], base=tiny)['drafter_select']['loaded'] is True
    assert call(['--ram-total-gib', '1'], base=tiny)['drafter_select']['loaded'] is True  # every expert fits
    stripes = root / 'stripes'
    stripes.mkdir()
    assert call(['--weight-dirs', str(stripes)], base=tiny)['drafter_select']['read_locations'] == 1  # no manifest there
    (stripes / 'tiny.gguf.stripe.json').write_text('{}')
    two = call(['--weight-dirs', str(stripes)], base=tiny)['drafter_select']
    assert two['read_locations'] == 2 and two['loaded'] is True and 'read locations' in two['reason'], two
    assert call(extra={'STRIX_WEIGHT_DIRS': str(stripes)}, base=tiny)['drafter_select']['read_locations'] == 2
    for label, config in [('i', default), ('ii', production), ('iii', old)]:
        print(label + ': ' + json.dumps(config, sort_keys=True))
    print('PASS: default == production (all keys); baseline; per-feature switches; ordering; arena; serve; auto fallbacks; '
          '--profile default == quality == no profile (byte for byte), fast = default + pending listed, precedence, --allow-pending; '
          '--drafter auto|on|off (env, order, the auto rule on a synthetic GGUF: caps x read locations)')

if baseline_exe:
    print(f"PASS: {baseline_comparisons} ordinary config outputs/stderr match the public baseline")
