#!/usr/bin/env python3
"""CPU-only execution of the gate's real watchdog/admission/cleanup helpers."""
import json, pathlib, re, shutil, subprocess, sys, tempfile, time
source=(pathlib.Path(__file__).resolve().parents[1]/'tools/minimax/session_cache_gate.sh').read_text()
bash=shutil.which('bash')
assert bash, 'bash required'
if subprocess.run([bash,'-c','command -v flock >/dev/null && command -v setsid >/dev/null'], capture_output=True).returncode:
    print('SKIP: Linux flock/setsid required; no process-group claims on this host')
    raise SystemExit(77)
runner=re.search(r"^runner='(.*)'$",source,re.M).group(1)
cleaner=re.search(r"^cleaner='(.*)'$",source,re.M).group(1)
receipts=re.search(r'^check_session_receipts\(\) \{\n(.*?)^\}',source,re.M|re.S).group(1)
timing=re.search(r"<<'PYTIME'\n(.*?)\nPYTIME",source,re.S).group(1)
with tempfile.TemporaryDirectory(prefix='mm-timing-') as temp:
    root=pathlib.Path(temp)
    for rank in (0,1): (root/f'r{rank}').mkdir()
    warm='session hashing seconds 5.000000 cache_hits 7 cache_misses 0'
    cold='session hashing seconds 5.000000 cache_hits 7 cache_misses 1'
    def parse(tag, lines):
        for rank, line in enumerate(lines):
            (root/f'r{rank}/{tag}.log').write_text(line+'\n')
        # As in launch_pair under set -e, failure must prevent the done receipt.
        result=subprocess.run([sys.executable,'-c',timing,str(root),tag],capture_output=True,text=True)
        if result.returncode == 0: (root/f'{tag}.done').write_text('passed\n')
        return result
    for rank in (0,1):
        lines=[warm,warm]; lines[rank]=cold
        assert parse('off-A',lines).returncode != 0, 'cold compatibility falsely passed'
        assert not (root/'off-A.done').exists()
        receipt=json.loads((root/f'r{rank}/off-A.timing.json').read_text())
        assert receipt['cache_hits']==7 and receipt['cache_misses']==1
    # Serving sessions run without the drafter by default: one (target) lookup on every serve arm.
    for tag,count in [('off-A',1),('off-B',1),('on-A',2),('on-B',2),
                      ('explicit-on-serve-off',1),('explicit-on-serve-on',1),
                      ('auto-serve-on',1),('auto-serve-default-adaptive',1)]:
        assert parse(tag,['\n'.join([warm]*count)]*2).returncode == 0
        assert parse(tag,['']*2).returncode != 0, 'missing compatibility receipt passed'
        assert parse(tag,[warm.replace(' cache_misses 0','')]*2).returncode != 0
    assert parse('off-rank',['','']).returncode == 0
    assert parse('off-rank',[cold,cold]).returncode != 0
    assert parse('prewarm',[cold+'\n'+warm]*2).returncode == 0
with tempfile.TemporaryDirectory(prefix='mm-receipts-') as temp:
    root=pathlib.Path(temp)
    for rank in (0,1): (root/f'r{rank}').mkdir()
    posix=subprocess.check_output([bash,'-c','pwd'],cwd=root,text=True).strip()
    def check_receipts(tag, lines):
        for rank, line in enumerate(lines):
            (root/f'r{rank}'/f'{tag}.log').write_text(line+'\n')
        return subprocess.run([bash,'-c',
            'bounded() { "$@"; }; check_session_receipts() {\n'+receipts+
            '\n}; OUT=$1; check_session_receipts "$2" serve',
            'test', posix, tag],capture_output=True).returncode
    live='session cache: on (runtime live)'
    off='session cache: off (route not yet GPU-qualified: serve/usb4-auto/fixed-width-one-session-v1)'
    for tag in ('explicit-on-serve-off','explicit-on-serve-on'):
        assert check_receipts(tag, [live,live])==0
        for rank in (0,1):
            lines=[live,live]; lines[rank]='READY'
            assert check_receipts(tag, lines)!=0, 'missing live rank falsely passed'
    for tag in ('auto-serve-off','auto-serve-on'):
        assert check_receipts(tag, [live,live])==0
        for rank in (0,1):
            lines=[live,live]; lines[rank]=off
            assert check_receipts(tag, lines)!=0, 'automatic admission refusal falsely passed'
with tempfile.TemporaryDirectory(prefix='mm-gate-') as temp:
    root=pathlib.Path(temp)
    def command(script,*args):
        return [bash,'-c',script,'gate',*map(str,args)]
    # pwd supplies a POSIX path on Windows Git Bash too.
    path=subprocess.check_output([bash,'-c','pwd'],cwd=root,text=True).strip()
    def run(tag, deadline, *cmd):
        return subprocess.run(command(runner,path,tag,deadline,*cmd),timeout=8)
    assert run('expired',int(time.time())-1,'true').returncode==124
    assert not list(root.glob('*.pid'))
    assert run('normal',int(time.time())+5,'true').returncode==0
    assert (root/'normal.ended').read_text().strip()=='0'
    # +3 s: the runner exits 124 without an .ended file when the deadline has already passed at launch (a 1 s margin
    # raced under load); this case must reach the timeout path.
    assert run('timeout',int(time.time())+3,'sleep','60').returncode==124
    assert (root/'timeout.ended').read_text().strip()=='124'
    # Cleanup must kill and verify a live process group, then close admission.
    process=subprocess.Popen(command(runner,path,'live',int(time.time())+60,'sleep','60'))
    until=time.monotonic()+4
    while not (root/'live.pid').exists():
        assert time.monotonic()<until
        time.sleep(.02)
    assert subprocess.run(command(cleaner,path),timeout=6).returncode==0
    assert process.wait(timeout=3)!=0
    assert run('late-ssh',int(time.time())+60,'true').returncode==125
    assert not (root/'late-ssh.pid').exists()
print('PASS live-runtime receipts on both ranks, automatic-off reason, absolute expiry, normal end, timeout FAIL, verified group cleanup and closed late admission (CPU only)')
