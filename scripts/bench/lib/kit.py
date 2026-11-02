#!/usr/bin/env python3
"""Stdlib-only token hashing, receipts, hardware allowlist, JSON and HTTP client."""
import concurrent.futures
import datetime
import glob
import hashlib
import json
import math
import struct
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

# Path-based importlib loaders do not add this module directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import interface
import nemotron
from hash_cache import ModelHashes

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / 'scripts' / 'bench'
VERSION = '2.0.0'
ENGINE = os.environ.get('BENCH_ENGINE', '')
def env_path(key, default):
    value = os.environ.get(key, default)
    # Native Windows Python does not translate MSYS paths received in environment variables.
    if os.name == 'nt' and re.match(r'^/[a-zA-Z]/', value):
        value = value[1] + ':' + value[2:]
    return Path(value)


BIN = env_path('BENCH_BIN_DIR', 'build')
MODELS = env_path('BENCH_MODEL_DIR', 'models')
QWEN = 'hybrid-q2.gguf' if os.environ.get('BENCH_SELECTED_PROFILE') == 'gsq' else 'Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf'
DRAFT = 'mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf'
DS = 'DeepSeek-V4.1-Flash-1017-Q2.gguf'
DS_DRAFT = 'DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf'
PROFILES = {
    'qwen': (['strix-tokenize', 'strix-hip-decode-token'], [QWEN, DRAFT, 'draft-vocab']),
    'deepseek-pair': (['strix-tokenize', 'strix-deepseek41-rank', 'strix-deepseek41-dspark-rank', 'strix-deepseek41-serve', 'strix-deepseek41-prefill'],
                      [DS, DS_DRAFT, 'c-u6.gguf', 'ours-K4', 'ours-K5', 'vocab']),
    'deepseek-rtx': (['strix-tokenize', 'strix-cuda-ds41-replica', 'strix-cuda-ds41-prefill'],
                     [DS, DS_DRAFT, 'MODEL.receipt.json', 'STATE.bin', 'ds41-draft-vocab']),
}
PROFILES['glm-pair'] = (['strix-glm5-rank'], ['glm'])
PROFILES['minimax-pair'] = (['strix-minimax-rank', 'strix-minimax-serve'], ['minimax'])
# Per-profile assets: default/fast use the Q4_K_M GGUF, quality the BF16 directory plus the EXL3 pack.
PROFILES['nemotron'] = (nemotron.BINARIES, nemotron.assets())
OPTIONAL = {'nemotron': nemotron.optional_assets()}
# Actual adopted asset pins: include/strix/deepseek41_model.hpp:133-137;
# include/strix/mtp_draft_profiles.hpp:47-48. Unknown pins stay null.
PINS = {
    'hybrid-q2.gguf': (None, '106603af0bc020222b90ed73ebfa04fd812280f53ec69135202a6cc71fc98b1e'),
    'Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf': (None, '8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2'),
    'c-u6.gguf': (None, '67e6e10ed97bff78c125e4e1433c3cb8dba4c0e3e2b2d42f9595e251b6fa2122'),
    DS: (365713686528, '5365629079214ff28becbbab7c0900ef9865b79c0f0c64d951924bb362ae5b67'),
    DS_DRAFT: (7966294016, 'd3efe0b1f6b6fea54f1c58c5cbeb3fded5967b15b64476aed00c95806ad3e3e9'),
    DRAFT: (2786568256, '5516d3ce407783a424315b6a2a0d8b06341665bb3541b98e5db0fe944add4c3c'),
}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def capture(argv):
    try:
        found = shutil.which(argv[0])
        if found:
            argv = [found, *argv[1:]]
        return subprocess.check_output(argv, stderr=subprocess.DEVNULL, timeout=30,
                                       text=True, encoding='utf-8', errors='replace').strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def id_hash(ids):
    # Canonical ASCII JSON integer array, no spaces or trailing newline.
    assert ids and all(type(x) is int and 0 <= x <= 0xffffffff for x in ids)
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).hexdigest()


def ids_read(path):
    text = Path(path).read_text(encoding='utf-8').strip()
    if not re.fullmatch(r'[0-9]+(?:[\s,]+[0-9]+)*', text):
        raise ValueError('tokenizer must emit only decimal token IDs')
    ids = [int(x) for x in re.split(r'[\s,]+', text)]
    id_hash(ids)
    return ids


def executable(name):
    return BIN / (name + '.exe' if (BIN / (name + '.exe')).exists() else name)


def hardware():
    cpu = None
    ram = None
    if Path('/proc/cpuinfo').exists():
        m = re.search(r'^model name\s*:\s*(.*)', Path('/proc/cpuinfo').read_text(), re.M)
        cpu = m.group(1) if m else None
    if Path('/proc/meminfo').exists():
        m = re.search(r'MemTotal:\s*(\d+)', Path('/proc/meminfo').read_text())
        ram = int(m.group(1)) * 1024 if m else None
    if os.name == 'nt':
        info = capture(['powershell.exe', '-NoProfile', '-Command',
                        '@{cpu=(Get-CimInstance Win32_Processor | Select-Object -First 1).Name; ram=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory} | ConvertTo-Json -Compress'])
        if info:
            allowed = json.loads(info)
            cpu, ram = allowed.get('cpu'), allowed.get('ram')
    rocm = capture(['hipconfig', '--version'])
    rv = re.search(r'\d+\.\d+(?:\.\d+)?', rocm)
    rocm_version = None
    rocm_path = capture(['hipconfig', '--rocmpath']) if rv else ''
    version_paths = [Path('/opt/rocm/.info/version')]
    if rocm_path and Path(rocm_path).is_absolute():
        version_paths.insert(0, Path(rocm_path) / '.info/version')
    for p in version_paths:
        try:
            version = re.search(r'\d+\.\d+(?:\.\d+)?', p.read_text().strip())
            if version:
                rocm_version = version.group()
                break
        except OSError:
            continue
    info = capture(['rocminfo'])
    arch = sorted(set(re.findall(r'\bgfx[0-9a-f]+\b', info)))
    nvidia = capture(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
    gpu = (['AMD Radeon 8060S'] if 'gfx1151' in arch else []) + (nvidia.splitlines() if nvidia else [])
    mv = Path('/sys/module/thunderbolt_stream/version')
    # Only numeric link properties; router names/UUIDs/topology addresses are excluded.
    links = []
    for p in sorted(glob.glob('/sys/bus/thunderbolt/devices/*/rx_speed')):
        try:
            value = Path(p).read_text().strip()
        except OSError:
            continue
        if re.fullmatch(r'[0-9.]+\s*Gb/s', value):
            links.append({'rx_speed': value})
    try:
        module_version = mv.read_text().strip() if mv.exists() else None
    except OSError:
        module_version = None
    return {'cpu_model': cpu, 'gpu_names': gpu, 'gfx_arch': arch, 'ram_bytes': ram,
            'kernel': platform.release(), 'os': platform.system(),
            'rocm_version': rocm_version, 'hip_version': rv.group() if rv else None,
            'driver_module_version': module_version,
            'usb4_links': links}


def asset_files():
    result = []
    # Optional assets (the Nemotron drafter) are hashed when present; their absence is reported, never hidden.
    roles = PROFILES[ENGINE][1] + [r for r in OPTIONAL.get(ENGINE, []) if (MODELS / r).exists()]
    for role in roles:
        p = MODELS / role
        if p.is_dir():
            files = sorted(x for x in p.rglob('*') if x.is_file() and not any(part.startswith('.') for part in x.relative_to(p).parts))
            if not files:
                raise ValueError('empty model asset directory: ' + role)
            for f in files:
                result.append((role, f))
        elif p.is_file():
            result.append((role, p))
        else:
            raise ValueError('missing model asset: ' + role)
    return result


def precheck():
    problems = []
    if sys.version_info < (3, 10):
        problems.append('Python 3.10 or newer required')
    for b in PROFILES[ENGINE][0]:
        if not executable(b).is_file() or not os.access(executable(b), os.X_OK):
            problems.append('binary-missing: ' + b)
    for a in PROFILES[ENGINE][1]:
        if not (MODELS / a).exists():
            problems.append('file-missing: ' + a)
    hw = hardware()
    if ENGINE == 'deepseek-rtx':
        if not capture(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']):
            problems.append('NVIDIA GPU required (nvidia-smi)')
        if platform.system() != 'Windows':
            problems.append('standalone replica receipt admission is Windows-only in this checkout (model size/mtime are not read on Linux)')
    else:
        if 'gfx1151' not in hw['gfx_arch']:
            problems.append('ROCm gfx1151 device required (rocminfo)')
        if not hw['hip_version']:
            problems.append('ROCm version required (hipconfig --version)')
        if not os.access('/dev/kfd', os.R_OK | os.W_OK):
            problems.append('/dev/kfd must be readable and writable')
    if ENGINE.endswith('-pair'):
        if not Path('/sys/module/thunderbolt_stream').is_dir():
            problems.append('thunderbolt_stream module must already be loaded')
        if not any(os.access(p, os.R_OK | os.W_OK) for p in glob.glob('/dev/tbstream*')):
            problems.append('/dev/tbstream* devices required')
        if not shutil.which('setsid'):
            problems.append('setsid required')
    # Inherited numerical overrides would otherwise silently change the profile.
    inherited = sorted(k for k in os.environ if k.startswith(('STRIX_', 'HIP_', 'HSA_', 'ROCR_')))
    if inherited:
        problems.append('unset inherited engine/runtime overrides: ' + ', '.join(inherited))
    cache = BIN / 'CMakeCache.txt'
    if ENGINE == 'deepseek-pair' and cache.exists():
        text = cache.read_text(encoding='utf-8', errors='replace')
        for key, value in [('STRIX_DS41_TRELLIS','ON'), ('STRIX_DS41_TRELLIS_API','dense_r1')]:
            match = re.search(r'^' + key + r':[^=]+=(.*)$', text, re.M)
            if match and match.group(1) != value: problems.append('build identity mismatch: ' + key + '=' + match.group(1))
    if ENGINE == 'nemotron':
        import importlib.util
        if importlib.util.find_spec('tokenizers') is None: problems.append('Python tokenizers package required for Nemotron prompts')
        if cache.exists():
            match = re.search(r'^STRIX_FAMILY_GEMMA4:[^=]+=(.*)$', cache.read_text(encoding='utf-8', errors='replace'), re.M)
            if not match or match.group(1).upper() not in ('ON', '1', 'TRUE', 'YES'):
                problems.append('build identity mismatch: STRIX_FAMILY_GEMMA4 must be ON')
    if ENGINE in ('glm-pair', 'minimax-pair'):
        import importlib.util
        if importlib.util.find_spec('tokenizers') is None: problems.append('Python tokenizers package required for EXL3 prompts')
        model = MODELS / ('glm' if ENGINE == 'glm-pair' else 'minimax')
        if not (model / 'tokenizer.json').is_file() and not (model / 'dspark/tokenizer.json').is_file():
            problems.append('file-missing: model tokenizer.json')
    return problems, hw


def inventory():
    problems, hw = precheck()
    if problems:
        raise ValueError('; '.join(problems))
    binary_files = [{'name': executable(b).name, 'sha256': sha(executable(b))}
                    for b in PROFILES[ENGINE][0]]
    models = model_inventory()
    commit = capture(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])
    dirty = bool(capture(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no']))
    # CMake cache content contains deployment paths: publish only safe option values.
    cache = BIN / 'CMakeCache.txt'
    options = {}
    if cache.exists():
        for line in cache.read_text(encoding='utf-8', errors='replace').splitlines():
            m = re.match(r'(STRIX_[A-Z0-9_]+|CMAKE_HIP_ARCHITECTURES|CMAKE_CUDA_ARCHITECTURES|CMAKE_BUILD_TYPE):[^=]+=(.*)', line)
            if m and re.fullmatch(r'[A-Za-z0-9_+.,;-]*', m.group(2)):
                options[m.group(1)] = m.group(2)
    return {'repo_commit': commit, 'tracked_dirty': dirty, 'binary_sha256s': binary_files,
            'model_files': models, 'models_verified': os.environ.get('BENCH_VERIFY_MODELS', 'cached') != 'skip',
            'build_options': options, 'hardware': hw}


def model_inventory(mode=None):
    mode = mode or os.environ.get('BENCH_VERIFY_MODELS', 'cached')
    cache = ModelHashes(MODELS, mode, sha)
    models = []
    for role, p in asset_files():
        h = cache.digest(p)
        size = p.stat().st_size
        pin = PINS.get(p.name)
        if pin and h is not None and (h != pin[1] or (pin[0] is not None and size != pin[0])):
            raise ValueError('published model pin differs: ' + p.name)
        models.append({'role': role, 'name': p.name, 'bytes': size, 'sha256': h,
                       'published_sha256': pin[1] if pin else None})
    cache.save()
    return models


def prompt_texts():
    return {x['id']: x['text'] for x in interface.prompt_set(os.environ.get('BENCH_PROMPTS', 'kit'), KIT)}


def prompts(out):
    texts = prompt_texts()
    result = []
    def save(name, ids, construction):
        (out / (name + '.ids')).write_text(','.join(map(str, ids)) + '\n', encoding='ascii')
        (out / (name + '.u32')).write_bytes(struct.pack('<' + 'I' * len(ids), *ids))
        result.append({'id': name, 'tokens': len(ids), 'sha256': id_hash(ids), 'construction': construction})
    contexts = list(map(int, os.environ.get('BENCH_CONTEXTS', '1024').split()))
    for name, text in texts.items():
        path = out / (name + '.txt')
        path.write_text(text, encoding='utf-8', newline='\n')
        ids = tokenize(path)
        save(name, ids, 'encode-raw-text')
        for n in contexts:
            long_ids = (ids * ((n + len(ids) - 1) // len(ids)))[:n]
            save(f'{name}-ctx{n}', long_ids, 'repeat-tokenized-text-truncate')
    seed = ids_read(out / (next(iter(texts)) + '.ids'))
    for n in contexts:
        save('prefill-' + str(n), (seed * ((n + len(seed) - 1) // len(seed)))[:n], 'repeat-tokenized-seed-truncate')
    if ENGINE == 'deepseek-rtx':
        rows=[]
        params=json.loads(os.environ['BENCH_PARAMS'])
        for r in range(params['warmup'] + params['repeat']):
            for name in texts:
                ids = ids_read(out / (name + '.ids'))
                ids = (ids * ((96 + len(ids) - 1) // len(ids)))[:96]
                # Prefix with warmup marker; parser discards these complete requests.
                for record in result:
                    if record['id'] == name: record.update(tokens=96, sha256=id_hash(ids), construction='repeat-tokenized-text-truncate-96')
                rows.append(('warm-' if r < params['warmup'] else '') + name + '-' + str(r) + '\t' + ','.join(map(str,ids)))
        (out / 'decode.tsv').write_text('\n'.join(rows)+'\n', encoding='ascii')
    dump(out / 'prompts.json', result)


def tokenize(path):
    if ENGINE == 'nemotron':
        from tokenizers import Tokenizer
        # Raw text identity; the server applies the chat template and reports its own prompt token count.
        return Tokenizer.from_file(str(MODELS / nemotron.TOKENIZER)).encode(path.read_text(encoding='utf-8'), add_special_tokens=False).ids
    if ENGINE in ('glm-pair', 'minimax-pair'):
        from tokenizers import Tokenizer
        model = MODELS / ('glm' if ENGINE == 'glm-pair' else 'minimax')
        tk = model / 'tokenizer.json'
        if not tk.exists(): tk = model / 'dspark/tokenizer.json'
        return Tokenizer.from_file(str(tk)).encode(path.read_text(encoding='utf-8'), add_special_tokens=False).ids
    data = subprocess.check_output([str(executable('strix-tokenize')), str(MODELS / (QWEN if ENGINE == 'qwen' else DS)),
                                    'encode', str(path)], timeout=300)
    tmp = path.with_suffix('.ids')
    tmp.write_bytes(data)
    return ids_read(tmp)


def objects(text):
    decoder = json.JSONDecoder()
    offset = 0
    while True:
        start = text.find('{', offset)
        if start < 0:
            return
        try:
            obj, length = decoder.raw_decode(text[start:])
            offset = start + length
            if isinstance(obj, dict):
                yield obj
        except ValueError:
            offset = start + 1


def receipt(path):
    candidates = [o for o in objects(Path(path).read_text(encoding='utf-8', errors='replace')) if 'model_content_identity' in o or 'generated_tokens' in o]
    if not candidates:
        raise ValueError('missing complete engine JSON receipt')
    return candidates[-1]


def stats(values):
    if not values or not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in values):
        raise ValueError('missing or invalid measurement; no zero/estimated rate permitted')
    return {'repeat_count': len(values), 'median': statistics.median(values), 'min': min(values), 'max': max(values), 'samples': values}


def nested(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = nested(v, key)
            if found is not None:
                return found
    return None


def qwen_admission(text):
    declined=re.search(r'Qwen auto-prefill declined: ([^\n]+)',text)
    if declined: raise ValueError('Qwen auto-prefill declined: '+declined[1])
    lines=re.findall(r'^Qwen auto-prefill: automatic T(\d+) context=(\d+)[^\n]*',text,re.M)
    if len(lines)!=1: raise ValueError('missing or ambiguous Qwen auto-prefill admission')
    tile,context=map(int,lines[0])
    return {'T':tile,'context':context,'notice':next(line for line in text.splitlines() if line.startswith('Qwen auto-prefill: automatic'))}


def ingest(out, prompt, mode, generation, log):
    text = log.read_text(encoding='utf-8', errors='replace')
    rows = []
    admission = None
    if ENGINE == 'qwen':
        obj = receipt(log)
        ids = obj.get('generated_tokens')
        if not ids:
            raise ValueError('receipt has no generated token IDs')
        engine_digest = {}
        for field in ('final_logits_digest', 'state_digest', 'final_state_digest'):
            value = nested(obj, field)
            if value is not None:
                engine_digest[field] = value
        if mode == 'ttft':
            if os.environ.get('BENCH_SELECTED_PROFILE')!='baseline': admission=qwen_admission(text)
            ms = nested(obj, 'prompt_ms')
            if not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
                raise ValueError('missing tiled prompt_ms; process elapsed is not TTFT')
            rows.append((mode, ids, [ms / 1000], 's', engine_digest or None))
        else:
            key = 'mtp_generation_cycle_elapsed_ms' if mode == 'spec' else 'generation_step_elapsed_ms_total'
            ms = obj.get(key)
            if not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
                raise ValueError('missing decode timing field: ' + key)
            rows.append((mode, ids, [len(ids) * 1000 / ms], 'tok/s', engine_digest or None))
    elif ENGINE == 'deepseek-pair':
        if mode == 'ttft':
            # Warm-up is deliberately excluded; the app times every --repeat in this process.
            values = [float(x) / 1000 for x in re.findall(r'ttft run \d+: \d+ rows, wall ([\d.]+) ms', text)]
            toks = re.findall(r'ttft run \d+:.*?next token (\d+)', text)
            digests = re.findall(r'ttft run \d+:.*?logits_fnv ([0-9a-f]+)', text)
            for i, value in enumerate(values):
                rows.append((mode, [int(toks[i])], [value], 's', {'logits_fnv': digests[i]}))
            if not rows:
                raise ValueError('missing TTFT run receipt')
        else:
            matches = re.findall(r'tokens \((\d+), fnv ([0-9a-f]+)\):([ 0-9]+)', text)
            speeds = re.findall(r'generation \d+ tokens in [\d.]+ s = ([\d.]+) tok/s', text)
            if len(matches) != 1 or len(speeds) != 1:
                raise ValueError('missing or ambiguous pair tokens/timing')
            count, digest, vector = matches[0]
            ids = [int(x) for x in vector.split()]
            if len(ids) != int(count):
                raise ValueError('truncated pair token vector')
            rows.append((mode, ids, [float(speeds[0])], 'tok/s', {'tokens_fnv': digest}))
    else:
        objs = list(objects(text))
        if any(o.get('pass') is False for o in objs):
            raise ValueError('RTX qualification receipt reports failure')
        ends = [o for o in objs if o.get('event') == 'chat_end']
        if mode == 'ttft':
            for o in [o for o in objs if o.get('event') == 'run'][int(os.environ.get('BENCH_WARMUP', '0')):]:
                if o.get('event') == 'run':
                    rows.append((mode, [o['next_token']], [o['seconds']], 's', {'logits_fnv': o['logits_fnv']}))
        for i, o in enumerate(ends):
            name = o['name']
            if 'warm-' in name: continue
            spec = name.startswith('s')
            summaries = [x for x in objs if x.get('phase') == name and x.get('event') == ('spec_summary' if spec else 'summary')]
            if not summaries:
                raise ValueError('RTX chat lacks matching timing receipt')
            rows.append(('spec' if spec else 'plain', o['toks'], [summaries[-1]['tok_s']], 'tok/s', {'state_digest': o['digest']}))
        if not rows:
            raise ValueError('RTX emitted no token/timing receipts')
    with (out / 'samples.jsonl').open('a', encoding='utf-8') as f:
        for row_index, (m, ids, values, unit, digest) in enumerate(rows):
            if ENGINE == 'deepseek-rtx' and mode == 'ttft' and row_index < int(os.environ.get('BENCH_WARMUP', '0')): continue
            f.write(json.dumps({'prompt_id': (next((x['id'] for x in read(out / 'prompts.json') if x['id'] + '-' in name), prompt) if ENGINE == 'deepseek-rtx' and mode != 'ttft' and (out / 'prompts.json').exists() else prompt), 'mode': m, 'generate_length': generation,
                                'actual_tokens': len(ids), 'token_ids_sha256': id_hash(ids),
                                'engine_digest': digest, 'values': values, 'unit': unit,
                                'log': log.name, **({'auto_prefill':admission} if admission else {})}) + '\n')


def sanitize(text):
    # Preserve numerical receipts while removing paths and machine-specific strings.
    text = re.sub(r'^.*usb4 auto-pairing:.*$', 'usb4 auto-pairing: <topology redacted>', text, flags=re.M)
    for key in ('BENCH_ROOT', 'BENCH_MODEL_DIR', 'BENCH_BIN_DIR', 'BENCH_OUT', 'HOME', 'USERPROFILE'):
        value = os.environ.get(key)
        if value and len(value) > 2:
            text = text.replace(value, '<path>')
    for value in (platform.node(), os.environ.get('USERNAME'), os.environ.get('USER')):
        if value and len(value) > 2:
            text = re.sub(r'(?<![\w])' + re.escape(value) + r'(?![\w])', '<redacted>', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<address>', text)
    text = re.sub(r'\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b', '<address>', text)
    text = re.sub(r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b', '<device-id>', text)
    text = re.sub(r'\b[A-Za-z]:[\\/][^\s"<>]*', '<path>', text)
    text = re.sub(r'(?<![\w<])/(?:[^\s"<>:,]+/)*[^\s"<>:,]*', '<path>', text)
    text = re.sub(r'\b[\w.-]+@[\w.-]+\b', '<peer>', text)
    return text


def sanitize_file(path):
    p = Path(path)
    p.write_text(sanitize(p.read_text(encoding='utf-8', errors='replace')), encoding='utf-8', newline='\n')


def build_warnings():
    if ENGINE == 'nemotron': return nemotron.warnings(MODELS)
    if ENGINE != 'deepseek-pair': return []
    cache = BIN / 'CMakeCache.txt'
    if not cache.exists(): return ['Cannot confirm dense_r1 build identity: CMakeCache.txt unavailable']
    text = cache.read_text(encoding='utf-8', errors='replace')
    keys = ('STRIX_DS41_TRELLIS', 'STRIX_DS41_TRELLIS_API', 'STRIX_DS41_TRELLIS_LIBRARY', 'STRIX_DS41_TRELLIS_INCLUDE_DIR')
    return ['Cannot confirm build option: ' + key for key in keys if not re.search(r'^'+key+r':[^=]+=.+$', text, re.M)]


def ingest_batch(out, manifest, log):
    text = log.read_text(encoding='utf-8', errors='replace')
    parsed = []
    if ENGINE == 'deepseek-pair':
        if manifest[0]['mode'] == 'ttft':
            for sec, token, digest in re.findall(r'ttft run \d+: \d+ rows, wall ([\d.]+) ms.*?next token (\d+), logits_fnv ([0-9a-f]+)', text):
                parsed.append(([int(token)], float(sec)/1000, {'logits_fnv':digest}))
        else:
            vectors=re.findall(r'tokens \((\d+), fnv ([0-9a-f]+)\):([ 0-9]+)',text)
            rates=re.findall(r'generation \d+ tokens in [\d.]+ s = ([\d.]+) tok/s',text)
            if len(vectors)!=len(rates): raise ValueError('decode token/timing count differs')
            for (count,digest,vector),rate in zip(vectors,rates):
                ids=list(map(int,vector.split()))
                if len(ids)!=int(count): raise ValueError('truncated token vector')
                parsed.append((ids,float(rate),{'tokens_fnv':digest}))
    elif ENGINE == 'glm-pair':
        if 'SPV1_BEGIN' in text:
            if 'SPV1_FAIL' in text or 'SPV1_DONE rank=0 status=PASS' not in text:
                raise ValueError('GLM SPV1 receipt is incomplete or failed')
            blocks=re.split(r'(?=SPV1_PLAIN rank=0 .*?pass=A0 )',text)[1:]
            if len(blocks)*2 != len(manifest): raise ValueError('GLM SPV1 prompt count differs')
            for i,block in enumerate(blocks):
                def fields(prefix):
                    line=next((x for x in block.splitlines() if x.startswith(prefix)),None)
                    if line is None: raise ValueError('missing GLM SPV1 receipt: '+prefix)
                    return dict(re.findall(r'(\w+)=([^ ]+)',line))
                a0=fields('SPV1_PLAIN rank=0 ')
                result=fields('SPV1_RESULT rank=0 ')
                a1=next((dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in block.splitlines() if line.startswith('SPV1_PLAIN rank=0 ') and 'pass=A1 ' in line),None)
                row=manifest[2*i]
                if a1 is None or any(x['hash']!=a0['hash'] for x in (result,a1)) or result.get('equal')!='1':
                    raise ValueError('GLM SPV1 native hash/equality mismatch')
                if any(x['prompt']!=row['prompt_id'] or int(x['gen'])!=row['generation'] for x in (a0,result,a1)):
                    raise ValueError('GLM SPV1 workload differs')
                for record in (a0,result):
                    parsed.append((None,1000/float(record['ms_token']),{'tokens_fnv':record['hash'],'native_equal':True}))
        elif any(row['mode']=='spec' for row in manifest):
            timings = re.findall(r'rank 0 spec-run:.*?plain ([\d.]+) ms/token, spec ([\d.]+) ms/token.*?tokens EQUAL to plain \(fnv ([0-9a-f]+)\)', text)
            vectors = re.findall(r'rank 0 spec-run (?:plain )?tokens:([ 0-9]+)', text)
            if len(vectors) != 2*len(timings): raise ValueError('incomplete GLM MTP token vectors')
            for i,(plain,spec,digest) in enumerate(timings):
                for j,ms in enumerate((plain,spec)):
                    parsed.append((list(map(int,vectors[2*i+j].split())),1000/float(ms),{'tokens_fnv':digest}))
        elif manifest[0]['mode']=='ttft':
            for sec,token,lf,sf in re.findall(r'rank 0 pf gemv-ttft: \d+ tokens ([\d.]+) s .*?token (\d+), logits fnv ([0-9a-f]+), state fnv ([0-9a-f]+)',text):
                parsed.append(([int(token)],float(sec),{'logits_fnv':lf,'state_fnv':sf}))
        else:
            # Each prompt contains one prefill digest and one complete token vector.
            blocks=re.split(r'rank 0 pf dev free delta',text)[1:]
            for block in blocks:
                match=re.search(r'rank 0 decode:.*?\(([\d.]+) tok/s\), tokens fnv ([0-9a-f]+).*?rank 0 tokens:([ 0-9]+)',block,re.S)
                if not match: raise ValueError('missing GLM decode receipt')
                rate,digest,vector=match.groups()
                dig={'tokens_fnv':digest}
                pre=re.search(r'logits fnv ([0-9a-f]+), state fnv ([0-9a-f]+)',block)
                if pre: dig.update(prefill_logits_fnv=pre[1],prefill_state_fnv=pre[2])
                parsed.append((list(map(int,vector.split())),float(rate),dig))
    elif ENGINE == 'minimax-pair':
        for row in manifest:
            name=re.escape(row['name']); mode=row['mode']
            prefix=r'rank 0 '+('spec ' if mode=='spec' else '')+name
            vector=re.search(prefix+r': tokens fnv ([0-9a-f]+) ids:([ 0-9]+)',text)
            if not vector: raise ValueError('missing MiniMax tokens: '+row['name'])
            digest={'tokens_fnv':vector[1]}; ids=list(map(int,vector[2].split()))
            if mode=='ttft':
                rate=re.search(prefix+r': prompt \d+ tokens, TTFT ([\d.]+) s',text)
            elif mode=='spec':
                rate=re.search(prefix+r' \(.*?\): prompt .*? = ([\d.]+) tok/s',text)
            else:
                rate=re.search(prefix+r': decode .*? = ([\d.]+) tok/s',text)
            if not rate: raise ValueError('missing MiniMax timing')
            state=re.search(prefix+r': state fnv ([0-9a-f]+).*?final rows fnv ([0-9a-f]+)',text)
            if state: digest.update(state_fnv=state[1],final_rows_fnv=state[2])
            parsed.append((ids,float(rate[1]),digest))
    if len(parsed)!=len(manifest): raise ValueError('incomplete batch: expected '+str(len(manifest))+' receipts, got '+str(len(parsed)))
    with (out/'samples.jsonl').open('a',encoding='utf-8') as f:
        for row,(ids,value,digest) in zip(manifest,parsed):
            if row['warmup']: continue
            if ids is not None and row['mode']!='ttft' and len(ids)!=row['generation']: raise ValueError('generation length differs')
            stats([value])
            f.write(json.dumps(dict(prompt_id=row['prompt_id'],mode=row['mode'],generate_length=row['generation'],
                                   actual_tokens=len(ids) if ids is not None else row['generation'],token_ids_sha256=id_hash(ids) if ids is not None else None,engine_digest=digest,values=[value],
                                   unit='s' if row['mode']=='ttft' else 'tok/s',log=log.name, timing_scope='native'))+'\n')


def parse_effective(text, explicit=()):
    """Parse the three public resolver formats, retaining all reported settings."""
    lines=text.splitlines()
    notices=[line for line in lines if 'defaults:' in line]
    # MiniMax's baseline banner is informational, not a dropped prerequisite.
    drops=[line for line in notices if not ('minimax defaults:' in line and 'optimization items are off' in line)]
    if drops: raise ValueError('required production defaults dropped: '+'; '.join(drops))
    env={}; args=[]; settings={}; unsets=[]
    for line in lines:
        # Informational Qwen resolver notices (auto-prefill, baseline YaRN suppression, active YaRN)
        # carry '=' but are not settings; the structured JSON on stdout is the configuration.
        if line.startswith(('Qwen auto-prefill', 'Qwen baseline:', 'Qwen experimental YaRN')): continue
        if line.startswith('env '):
            key,value=line[4:].split('=',1); env[key]=value
        elif line.startswith('effective env '):
            key,value=line[14:].split('=',1); env[key]=value.split(' (',1)[0]
        elif line.startswith('unset '): unsets.append(line[6:])
        elif line.startswith('arg '): args.append(line[4:])
        elif re.match(r'^(STRIX_|HSA_|DG_)[A-Z0-9_]*=',line):
            key,value=line.split('=',1); env[key]=value
        elif line.startswith('effective '):
            key,value=line[10:].split('=',1); settings[key]=value.split(' (',1)[0]
        elif '=' in line and not 'defaults:' in line and not line.lstrip().startswith('{'):
            key,value=line.split('=',1); settings[key]=value
        elif line.startswith('app '): settings['app']=line[4:]
    if not env and not settings:
        objs=list(objects(text))
        if len(objs)!=1: raise ValueError('missing effective configuration')
        settings=objs[0]
        if 'environment' in settings and 'admitted' in settings:
            if not isinstance(settings['environment'],dict) or not isinstance(settings.get('argv'),list):
                raise ValueError('incomplete Qwen effective configuration')
            env={key:value['value'] for key,value in settings['environment'].items()}
            args=settings['argv']
        else:
            if 'load_serial' not in settings: raise ValueError('incomplete RTX effective configuration')
            env['DG_LOAD_SERIAL']='1' if settings['load_serial'] else '0'
        if settings.get('draft_vocab')=='full' and os.environ.get('BENCH_SELECTED_PROFILE')!='baseline':
            raise ValueError('required production defaults dropped: draft vocab: no auto directory; using full')
    if not env and 'environment' not in settings: raise ValueError('missing effective environment')
    if settings.get('app') and text.startswith('app ') and not args:
        raise ValueError('missing effective arguments')
    for assignment in explicit:
        key,value=assignment.split('=',1); env[key]=value
    # Only --rank's value differs by rank. Never strip arbitrary numbers or sort argv.
    for i,arg in enumerate(args[:-1]):
        if arg=='--rank': args[i+1]='RANK'
    for key in unsets: env.pop(key,None)
    result=dict(environment=env, arguments=args, settings=settings)
    if unsets: result['unset_environment']=unsets
    return result


def record_effective(out, label, local, peer, model, peer_model, binary, peer_binary, remote_run, *explicit):
    def normal(text, remote=False):
        roots=[(peer_model if remote else model,'MODEL_DIR'),
               (peer_binary if remote else binary,'BIN_DIR'),
               (remote_run if remote else str(out).replace('\\','/'),'RUN_DIR')]
        text=text.replace('\\','/')
        for value,marker in sorted(roots,key=lambda x:len(x[0]),reverse=True):
            if not value: continue
            value=value.replace('\\','/')
            variants=[value]
            if re.match(r'^[A-Za-z]:/',value): variants.append('/'+value[0].lower()+value[2:])
            if re.match(r'^/[A-Za-z]/',value): variants.append(value[1].upper()+':'+value[2:])
            for variant in variants: text=text.replace(variant,marker)
        return text
    b=None
    if ENGINE=='qwen' and label.startswith('prefill-') and os.environ.get('BENCH_SELECTED_PROFILE')!='baseline':
        qwen_admission(local.read_text(encoding='utf-8'))
    a=parse_effective(normal(local.read_text(encoding='utf-8')), [normal(x) for x in explicit])
    if peer.exists():
        b=parse_effective(normal(peer.read_text(encoding='utf-8'),True), [normal(x) for x in explicit])
        if a!=b: raise ValueError('rank 0 and rank 1 effective configuration mismatch: '+label)
    # Normalize known roots before redaction; markers preserve which pack was selected.
    def public(value):
        if isinstance(value,dict): return {k:public(v) for k,v in value.items()}
        if isinstance(value,list): return [public(v) for v in value]
        if isinstance(value,str):
            value=re.sub(r'(MODEL_DIR|BIN_DIR|RUN_DIR)/([^\s]*)',lambda m:m[1]+':'+m[2].replace('/',':'),value)
            return sanitize(value)
        return value
    config=public(a)
    if ENGINE=='deepseek-pair' and label.startswith('decode-'):
        config=dict(config,env_source='strix-deepseek41-rank --print-effective-config',rank0=public(a),rank1=public(b))
    config_path=out/('effective-'+label+'.json')
    if ENGINE=='qwen' and config_path.exists() and read(config_path)!=config:
        raise ValueError('Qwen effective configuration changed between launches: '+label)
    dump(config_path,config)
    path=out/'effective-environment.json'
    env=read(path) if path.exists() else {}
    env[label]=public(a['environment'])
    if ENGINE=='deepseek-pair' and label.startswith('decode-'):
        # Kit setting added to the resolved rank environment for the DeepSeek decode app (lib/batch.sh).
        env[label]['STRIX_DS41_DSPARK_WINDOW']='controller'
    dump(path,env)


def qwen_rotary_reference(configs):
    """Project the resolved numerical payload; native adds zero reference bytes."""
    rotary = {}
    required = ('rope_mode', 'factor', 'original_max', 'effective_max', 'rotary_dim',
                'index_rotary_dim', 'theta', 'mrope_sections', 'mrope_interleaved',
                'beta_fast', 'beta_slow', 'truncate', 'attention_factor', 'math_version',
                'generator', 'table_sha256')
    for label, config in configs.items():
        settings = config['settings']
        # A legacy payload omission is safe only without any non-native declaration.
        factors = set()
        argument_off = False
        yarn_reason = False

        def factor(value):
            if isinstance(value, dict): value = value.get('value')
            if str(value) == 'off': return 0
            if str(value) not in ('2', '4'):
                raise ValueError('invalid Qwen YaRN declaration')
            return int(value)

        def arguments(value):
            nonlocal argument_off
            if isinstance(value, str):
                import shlex
                value = shlex.split(value)
            if isinstance(value, dict):
                declarations(value)
                return
            if not isinstance(value, (list, tuple)): return
            for i, arg in enumerate(value):
                if not isinstance(arg, str): continue
                if arg == '--rope-yarn':
                    if i + 1 == len(value): raise ValueError('missing Qwen YaRN argument')
                    f = factor(value[i + 1])
                elif arg.startswith('--rope-yarn='): f = factor(arg.split('=', 1)[1])
                else: continue
                if f: factors.add(f)
                else: argument_off = True

        def declarations(obj):
            nonlocal yarn_reason
            if not isinstance(obj, dict): return
            for key, value in obj.items():
                if key in ('argv', 'arguments', 'args', 'requested_args', 'requested_arguments'):
                    arguments(value)
                elif key in ('environment', 'requested_environment') and isinstance(value, dict):
                    if 'STRIX_QWEN_ROPE_YARN' in value:
                        f = factor(value['STRIX_QWEN_ROPE_YARN'])
                        if f: factors.add(f)
                elif key in ('reason', 'resolver_reason') and isinstance(value, str) and 'yarn' in value.lower():
                    yarn_reason = True
                elif key in ('requested', 'request', 'requested_options', 'resolver'):
                    if isinstance(value, (str, list, tuple)): arguments(value)
                    else: declarations(value)
                elif key in ('rope_yarn', '--rope-yarn', 'STRIX_QWEN_ROPE_YARN'):
                    f = factor(value)
                    if f: factors.add(f)
                    else: argument_off = True
        declarations(config)
        declarations(settings)
        non_native = bool(factors) or yarn_reason or settings.get('rope_mode', 'native') != 'native'
        payloads = [settings[key] for key in ('rotary_identity', 'rotary') if key in settings]
        if not payloads:
            if non_native:
                raise ValueError('incomplete Qwen rotary reference identity')
            continue
        if any(not isinstance(p, dict) or 'rope_mode' not in p for p in payloads):
            raise ValueError('explicit Qwen rotary payload requires rope_mode')
        payload = payloads[0]
        if any(p != payload for p in payloads[1:]):
            raise ValueError('conflicting Qwen rotary payloads')
        mode = payload['rope_mode']
        if 'rope_mode' in settings and settings['rope_mode'] != mode:
            raise ValueError('conflicting Qwen rotary modes')
        if mode == 'native':
            if non_native:
                raise ValueError('Qwen YaRN declaration requires non-native rotary payload')
            native = dict(factor=0, original_max=262144, effective_max=262144,
                          rotary_dim=64, index_rotary_dim=64, theta=10000000,
                          mrope_sections=[11,11,10], mrope_interleaved=True,
                          beta_fast=32, beta_slow=1, truncate=True, attention_factor=1,
                          math_version='qwen-yarn-hf64-v1', generator='fp64-libm-to-f32-rne',
                          table_sha256=None, table_bytes=0,
                          descriptor_sha256=hashlib.sha256(b'').hexdigest())
            if any(key in payload and payload[key] != value for key, value in native.items()):
                raise ValueError('native Qwen rotary payload has non-native parameters')
            continue
        if mode != 'yarn' or any(key not in payload for key in required):
            raise ValueError('incomplete Qwen rotary reference identity')
        if argument_off or any(f != payload['factor'] for f in factors):
            raise ValueError('conflicting Qwen YaRN declarations and rotary payload')
        if not isinstance(payload['table_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', payload['table_sha256']):
            raise ValueError('non-native Qwen reference requires resolved table SHA256')
        rotary[label] = {key: payload[key] for key in required}
    return rotary


def initialize(out):
    inv = read(out / 'local-inventory.json') if (out / 'local-inventory.json').exists() else inventory()
    peer_hardware = read(out / 'peer-inventory.json')['hardware'] if (out / 'peer-inventory.json').exists() else None
    environment = read(out / 'effective-environment.json') if (out / 'effective-environment.json').exists() else {}
    identity = {'profile': os.environ['BENCH_PROFILE'], 'build_options': inv['build_options'],
                'environment': environment, 'command_profile_sha256': sha(KIT / 'engines' / (ENGINE + '.sh')),
                'kit_source_sha256s': {p.name: sha(p) for p in sorted((KIT / 'lib').glob('*.py'))},
                'binary_sha256s': inv['binary_sha256s'], 'model_files': inv['model_files'],
                'rocm_version': inv['hardware']['rocm_version'], 'hip_version': inv['hardware']['hip_version'],
                'kernel': inv['hardware']['kernel']}
    if peer_hardware:
        identity['peer_kernel'] = peer_hardware['kernel']
    identity['parameters'] = json.loads(os.environ.get('BENCH_PARAMS', '{}'))
    source = KIT / 'prompts/kit.json' if os.environ.get('BENCH_PROMPTS','kit') == 'kit' else env_path('BENCH_PROMPTS','')
    prompt_set = {'name': 'kit' if os.environ.get('BENCH_PROMPTS','kit') == 'kit' else 'file', 'sha256': sha(source)}
    identity['prompt_set'] = prompt_set
    configs={p.stem[len('effective-'):]:read(p) for p in sorted(out.glob('effective-*.json')) if p.name!='effective-environment.json'}
    if ENGINE=='qwen':
        identity['profile_pairs']={label:{key:config['settings'][key] for key in ('decode_profile','prefill_profile','numerical_profile_pair_identity') if key in config['settings']} for label,config in configs.items()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    # References are keyed by what decides the output bits, not by where or how the binary was built:
    # users build their own binaries, so binary hashes, build options, kernel, ROCm and output options stay out
    # of the key. Build-time numerics that matter reach it through the resolved environment: the resolvers
    # print them (e.g. MiniMax build.*) or drop the item, which fails the run.
    reference_key = {'engine': ENGINE, 'profile': identity['profile'], 'environment': identity['environment'],
                     'prompt_set_sha256': prompt_set['sha256'],
                     'model_sha256s': sorted(f['sha256'] for f in inv['model_files'])}
    if ENGINE=='qwen':
        reference_key['profile_pairs']=identity['profile_pairs']
        rotary = qwen_rotary_reference(configs)
        if rotary:
            reference_key['rotary_identity'] = rotary
    if ENGINE=='nemotron':
        projections = {label: config['settings'].get('numerics') for label, config in configs.items() if label.startswith('serve-')}
        if len({json.dumps(x, sort_keys=True) for x in projections.values()}) > 1:
            raise ValueError('conflated identities: drafter arms report different numerical projections')
        if projections:
            reference_key['numerics'] = next(iter(projections.values()))
    reference_sha = hashlib.sha256(json.dumps(reference_key, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    dump(out / 'identity.json', {'schema_version': 1, 'kit_version': VERSION, 'engine': ENGINE,
                                'repo_commit': inv['repo_commit'], 'tracked_dirty': inv['tracked_dirty'],
                                **({'env_source':'strix-deepseek41-rank --print-effective-config'} if ENGINE=='deepseek-pair' and any(x.startswith('decode-') for x in configs) else {}),
                                'resolved_configs': configs,
                                'identity': identity, 'identity_sha256': fingerprint,
                                'reference_key': reference_key, 'reference_sha256': reference_sha,
                                'hardware': inv['hardware'], 'peer_hardware': peer_hardware,
                                'peer_inventory': read(out / 'peer-inventory.json') if peer_hardware else None,
                                'params': identity['parameters'], 'prompt_set': prompt_set,
                                'models_verified': inv.get('models_verified', True) and (not peer_hardware or read(out / 'peer-inventory.json').get('models_verified', True)),
                                'start_epoch': float(os.environ.get('BENCH_START', time.time())),
                                'suite': os.environ['BENCH_SUITE'],
                                'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})


# Bench protocol fields. All additive: older results.json files without them still validate.
def cache_state(values):
    """cold: every request reported cached_tokens 0; reused: any > 0; unverified: any request lacks the field."""
    if not values or any(not isinstance(v, int) for v in values): return 'unverified'
    return 'reused' if any(v > 0 for v in values) else 'cold'


def cache_policy(out):
    """Police prompt/prefix cache reuse from the engine's usage.prompt_tokens_details.cached_tokens.

    BENCH_ALLOWED_CACHED_TOKENS (default 0) is the shared prefix a suite may reuse; BENCH_DECLARED_CACHE_REUSE=1
    declares reuse as intended; BENCH_CACHE_POLICY=refuse fails an undeclared nonconforming run (default: flag)."""
    allowed = int(os.environ.get('BENCH_ALLOWED_CACHED_TOKENS', '0'))
    if allowed < 0: raise ValueError('BENCH_ALLOWED_CACHED_TOKENS must be >= 0')
    declared = os.environ.get('BENCH_DECLARED_CACHE_REUSE', '0') == '1'
    mode = os.environ.get('BENCH_CACHE_POLICY', 'flag')
    if mode not in ('flag', 'refuse'): raise ValueError('BENCH_CACHE_POLICY must be flag or refuse')
    states, worst, reasons = {}, 0, []
    for path in sorted(out.glob('http-c*.json')):
        values = [x.get('cached_tokens') for group in read(path).get('requests', []) for x in group]
        states[path.stem] = cache_state(values)
        top = max((v for v in values if isinstance(v, int)), default=0)
        worst = max(worst, top)
        if top > allowed:
            reasons.append(f'{path.stem}: cached_tokens {top} > allowed {allowed}')
    if not states: return None
    nonconforming = bool(reasons) and not declared
    return {'allowed_cached_tokens': allowed, 'declared_reuse': declared, 'policy': mode, 'states': states,
            'max_cached_tokens': worst, 'nonconforming': nonconforming, 'reasons': reasons,
            'refused': nonconforming and mode == 'refuse'}


def not_run_list(out, data):
    """Explicit list of what this run did not measure, each with a reason."""
    items = [dict(item=str(x['item']), reason=str(x['reason'])) for x in (read(out / 'not-run.json') if (out / 'not-run.json').exists() else [])]
    items += [{'item': 'skipped', 'reason': s} for s in data.get('skipped', [])]
    if 'exactness' not in data.get('params', {'only': ['exactness']}).get('only', []):
        items.append({'item': 'exactness', 'reason': 'not selected (--only)'})
    # HTTP rows are merged into results.json after finalize; read them from their own file too.
    rows = data.get('measurements', []) + (read(out / 'http-measurements.json') if (out / 'http-measurements.json').exists() else [])
    modes = {(m['prompt_id'], m['mode']) for m in rows}
    for path in sorted(out.glob('http-c*.json')):
        requests = [x for group in read(path).get('requests', []) for x in group]
        for mode, key in (('engine-decode', 'engine_decode_ms'), ('engine-ttft', 'engine_prefill_ms')):
            if (path.stem, mode) not in modes:
                missing = not requests or any(key not in x for x in requests)
                items.append({'item': f'{path.stem} {mode}', 'reason': 'engine reply carries no strix ' + key if missing else
                              'engine timing zero or one-token completions'})
        if cache_state([x.get('cached_tokens') for x in requests]) == 'unverified':
            items.append({'item': f'{path.stem} cache state', 'reason': 'usage.prompt_tokens_details.cached_tokens absent'})
    return items


def protocol(out, data):
    policy = cache_policy(out)
    if policy:
        data['cache_policy'] = policy
        if policy['refused']: data['status'] = 'failed'
    data['not_run'] = not_run_list(out, data)
    if (out / 'dispatch-census.json').exists():
        data['dispatch_census'] = read(out / 'dispatch-census.json')


def abba_pairs(sequence, higher_is_better=True):
    """k-of-n pairing over arms in process order. sequence = [(arm 'A'|'B', value), ...]; consecutive entries form
    pairs (ABBA... gives AB, BA, ...). Delta is (B - A) / A per pair; 'b_better' counts pairs where B wins."""
    if len(sequence) < 2 or len(sequence) % 2:
        raise ValueError('ABBA pairing needs an even number (>= 2) of runs')
    deltas, orders = [], []
    for (a1, v1), (a2, v2) in zip(sequence[0::2], sequence[1::2]):
        if {a1, a2} != {'A', 'B'}: raise ValueError('each consecutive pair must hold one A and one B run')
        stats([v1, v2])
        a, b = (v1, v2) if a1 == 'A' else (v2, v1)
        deltas.append((b - a) / a)
        orders.append(a1 + a2)
    better = [d > 0 if higher_is_better else d < 0 for d in deltas]
    k = sum(better)
    ties = sum(d == 0 for d in deltas)
    n = len(deltas)
    return {'n_pairs': n, 'orders': orders, 'ab_pairs': orders.count('AB'), 'ba_pairs': orders.count('BA'),
            'higher_is_better': higher_is_better, 'deltas': deltas, 'median_delta': statistics.median(deltas),
            'b_better_pairs': k, 'a_better_pairs': n - k - ties, 'ties': ties, 'k_of_n': f'{k}/{n}',
            'sign_agreement': f'{max(k, n - k - ties)}/{n}'}


# Run parameters that define the workload (generation length, contexts, HTTP concurrency, prompt source, serve) plus
# any capacity/slot/batch parameter; arms that differ in them are not comparable.
WORKLOAD_PARAMS = ('gen_tokens', 'contexts', 'concurrency', 'prompts', 'serve')


def workload_identity(d):
    key = d.get('reference_key', {})
    params = d.get('params') or d.get('identity', {}).get('parameters') or {}
    chosen = {k: v for k, v in params.items() if k in WORKLOAD_PARAMS or re.search(r'slot|capacity|batch|context|max_tokens', k)}
    return {'engine': d['engine'], 'prompt_set_sha256': key.get('prompt_set_sha256'),
            'model_sha256s': sorted(key.get('model_sha256s', [])), 'params': chosen}


def measurement_workload(d, m):
    """Per compared measurement: the recorded workload (constituent prompt identities/lengths and requested generation
    length, also for synthetic http-cN rows) and cache condition. Never derived from optional exactness rows."""
    if not isinstance(m.get('workload'), dict):
        raise ValueError(m['prompt_id'] + '/' + m['mode'] + ': measurement has no workload record (re-run with this kit)')
    return {'workload': m['workload'], 'cache': (m.get('cache_state'), m.get('cached_tokens_max'))}


def pairs(arms):
    """arms = [('A'|'B', results.json path), ...] in process order -> per-measurement ABBA pair verdicts."""
    runs = [(arm, read(path)) for arm, path in arms]
    for arm, d in runs:
        validate(d)
        if d['status'] != 'complete':
            raise ValueError(f"ABBA arm {arm} has status {d['status']}; only complete runs can be paired")
    first = runs[0][1]
    if any(workload_identity(d) != workload_identity(first) for _, d in runs):
        raise ValueError('ABBA arms differ in engine, prompt set, models or workload/capacity parameters')
    if any(d.get('cache_policy', {}).get('nonconforming') for _, d in runs) and os.environ.get('BENCH_DECLARED_CACHE_REUSE', '0') != '1':
        raise ValueError('an arm run is cache-nonconforming; declare reuse with BENCH_DECLARED_CACHE_REUSE=1')
    keys = [(m['prompt_id'], m['mode'], m['unit']) for m in first['measurements']]
    result, missing = [], []
    for key in keys:
        values, workloads = [], []
        for arm, d in runs:
            m = next((m for m in d['measurements'] if (m['prompt_id'], m['mode'], m['unit']) == key), None)
            if m is None: break
            values.append((arm, m['median']))
            workloads.append(measurement_workload(d, m))
        if len(values) != len(runs):
            missing.append({'item': '/'.join(key[:2]), 'reason': 'not measured in every arm run'})
            continue
        if any(w['cache'] != workloads[0]['cache'] for w in workloads):
            # Declared reuse never makes asymmetric cache conditions comparable.
            raise ValueError('/'.join(key[:2]) + ': cache state differs between arms ' + str([w['cache'] for w in workloads]))
        if any(w != workloads[0] for w in workloads):
            raise ValueError('/'.join(key[:2]) + ': prompt identity or generation length differs between arms')
        result.append({'prompt_id': key[0], 'mode': key[1], 'unit': key[2], 'cache_state': workloads[0]['cache'][0],
                       **abba_pairs(values, key[2] == 'tok/s')})
    return {'schema_version': 1, 'engine': first['engine'], 'arms': [arm for arm, _ in runs],
            'identity_sha256s': [d['identity_sha256'] for _, d in runs], 'pairs': result, 'not_run': missing}


def attach_census(out, trace, args):
    """Run tools/dispatch_census.py on a rocprofv3 trace and attach it to the run receipt (results.json)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('dispatch_census', ROOT / 'tools' / 'dispatch_census.py')
    census = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(census)
    target = out / 'dispatch-census.json'
    if census.main([str(trace), *args, '-o', str(target)]) != 0:
        raise ValueError('dispatch census failed')
    if (out / 'results.json').exists():
        d = read(out / 'results.json')
        d['dispatch_census'] = read(target)
        validate(d)
        dump(out / 'results.json', d)
        report(out / 'results.json')


def finalize(out, write_refs=False, status='complete'):
    data = read(out / 'identity.json')
    data['wall_s'] = max(0, time.time() - data.pop('start_epoch', time.time()))
    data['model_loads'] = len((out / 'model-loads.txt').read_text().splitlines()) if (out / 'model-loads.txt').exists() else 0
    data.update(status=status, prompts=read(out / 'prompts.json') if (out / 'prompts.json').exists() else [],
                measurements=[], exactness=[], skipped=read(out / 'skipped.json') if (out / 'skipped.json').exists() else [])
    samples = [json.loads(x) for x in (out / 'samples.jsonl').read_text().splitlines()] if (out / 'samples.jsonl').exists() else []
    refs = read(KIT / 'refs' / (data['engine'] + '.json'))
    known = refs['identities'].get(data.get('reference_sha256', ''), {})
    candidates = {}
    groups = {}
    for s in samples:
        key = (s['prompt_id'], s['mode'], s['generate_length'])
        groups.setdefault(key, []).append(s)
    for (p, mode, length), group in groups.items():
        values = [v for s in group for v in s['values']]
        measurement = {'prompt_id': p, 'mode': mode, 'unit': group[0]['unit'], **stats(values)}
        # The measured workload, recorded independently of the optional exactness rows (ABBA pairing compares it).
        record = next((x for x in data['prompts'] if x['id'] == p), {})
        measurement['workload'] = {'prompts': [{'id': p, 'sha256': record.get('sha256'), 'tokens': record.get('tokens')}],
                                   'generate_length': length}
        if any('auto_prefill' in s for s in group): measurement['auto_prefill']=[s['auto_prefill'] for s in group]
        if 'timing_scope' in group[0]: measurement['timing_scope']=group[0]['timing_scope']
        data['measurements'].append(measurement)
        if any(s['token_ids_sha256'] is None for s in group):
            note='GLM MTP: native equality/hash receipt only; full token IDs unavailable, kit exactness unqualified'
            if note not in data['skipped']: data['skipped'].append(note)
            continue
        prompt_sha = next(x['sha256'] for x in data['prompts'] if x['id'] == p)
        key = f"{data['identity']['profile']}:{p}:{mode}:{length}"
        target = known.get(key)
        hashes = {s['token_ids_sha256'] for s in group}
        digests = [s['engine_digest'] for s in group]
        stable = len(hashes) == 1 and all(d == digests[0] for d in digests)
        candidate = {'prompt_sha256': prompt_sha, 'token_ids_sha256': group[0]['token_ids_sha256'],
                     'engine_digest': digests[0], 'actual_tokens': group[0]['actual_tokens']}
        verdict = 'NO-REFERENCE'
        if not stable:
            verdict = 'MISMATCH'
        elif target is not None:
            verdict = 'EXACT' if target == candidate else 'MISMATCH'
        data['exactness'].append({'prompt_id': p, 'mode': mode, 'generate_length': length,
                                  'verdict': verdict, 'repeat_stable': stable, **candidate})
        if stable:
            candidates[key] = candidate
    if data['engine'] == 'nemotron':
        nemotron.augment(data, out, refs, known, read)
    if 'exactness' not in data.get('params', {'only':['exactness']}).get('only', []):
        data['exactness'] = []
        data.pop('digest_checks', None)
    data['logs'] = [p.name for p in sorted(out.glob('*.log'))]
    # Plain and speculative modes must agree within this run, independently of published refs.
    for e in data['exactness']:
        plain = next((x for x in data['exactness'] if x['prompt_id'] == e['prompt_id']
                      and x['mode'] in ('plain', 'greedy') and x['generate_length'] == e['generate_length']), None)
        if e['mode'] == 'spec' and plain:
            e['plain_tokens_equal'] = e['token_ids_sha256'] == plain['token_ids_sha256']
            if not e['plain_tokens_equal']:
                e['verdict'] = 'MISMATCH'
    if any(e['verdict'] == 'MISMATCH' for e in data['exactness']):
        data['status'] = 'mismatch'
    protocol(out, data)
    validate(data)
    dump(out / 'results.json', data)
    report(out / 'results.json')
    if write_refs:
        if any(s['token_ids_sha256'] is None for s in samples):
            raise ValueError('cannot write reference candidates without full token IDs')
        if status != 'complete' or data['status'] != status or any(e['verdict'] == 'MISMATCH' for e in data['exactness']):
            raise ValueError('failed/unstable runs cannot write reference candidates')
        dump(out / 'candidate-refs.json', {'schema_version': 1, 'engine': data['engine'],
                                          'candidate_only': True, 'identity': data['identity'],
                                          'reference_key': data['reference_key'],
                                          'identities': {data['reference_sha256']: candidates}})


def validate(d):
    for key in ('kit_version', 'repo_commit', 'engine', 'identity', 'identity_sha256', 'hardware',
                'prompts', 'measurements', 'exactness', 'status'):
        if key not in d:
            raise ValueError('results missing field: ' + key)
    if d.get('schema_version') != 1 or not re.fullmatch('[0-9a-f]{64}', d['identity_sha256']):
        raise ValueError('bad schema/identity')
    if d['status'] not in ('complete', 'failed', 'mismatch') or not re.fullmatch('[0-9a-f]{40}', d['repo_commit']):
        raise ValueError('bad run status/commit')
    for f in d['identity'].get('binary_sha256s', []) + d['identity'].get('model_files', []):
        if Path(f['name']).name != f['name'] or '\\' in f['name'] or (f['sha256'] is None and d.get('models_verified', True)) or (f['sha256'] is not None and not re.fullmatch('[0-9a-f]{64}', f['sha256'])):
            raise ValueError('file record requires a basename and SHA256')
    for p in d['prompts']:
        if not re.fullmatch('[0-9a-f]{64}', p['sha256']) or p['tokens'] <= 0:
            raise ValueError('bad prompt record')
    for m in d['measurements']:
        if m['unit'] not in ('s', 'tok/s') or m['repeat_count'] < 1:
            raise ValueError('bad measurement units/count')
        expected = stats(m['samples'])
        if any(m[k] != expected[k] for k in expected):
            raise ValueError('measurement summary differs from samples')
    for e in d['exactness']:
        if e['verdict'] not in ('EXACT', 'MISMATCH', 'NO-REFERENCE'):
            raise ValueError('invalid exactness verdict')
        for key in ('prompt_sha256', 'token_ids_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', e[key]):
                raise ValueError('invalid exactness digest')
        witness = e.get('draft_witness')
        if witness is not None and (witness.get('status') not in ('drafted', 'declined')
                                    or any(type(witness.get(x)) is not int or witness[x] < 0 for x in ('drafted', 'verified', 'accepted'))):
            raise ValueError('invalid draft witness')
        if d['engine'] == 'nemotron' and e['mode'].startswith('spec') and witness is None:
            raise ValueError('missing draft witness')
    for c in d.get('digest_checks', []):
        if c.get('verdict') not in ('EXACT', 'MISMATCH', 'NO-REFERENCE') or not re.fullmatch('[0-9a-f]{64}', c.get('kv', '')):
            raise ValueError('invalid digest check')
        if c['verdict'] != 'NO-REFERENCE' and not re.fullmatch('[0-9a-f]{64}', c.get('reference') or ''):
            raise ValueError('digest check verdict without reference')
    for p in d.get('logs', []):
        if Path(p).name != p or '\\' in p:
            raise ValueError('log path must be basename only')
    if 'not_run' in d and not all(isinstance(x, dict) and set(x) == {'item', 'reason'} for x in d['not_run']):
        raise ValueError('not_run entries need item and reason')


def baseline(engine, p, mode, profile=None):
    records = read(KIT / 'expected.json')['rows']
    # Measurements carry a context suffix (code-ctx1024); README rows name the bare prompt. '*' marks a README
    # row measured on an unspecified benchmark prompt; an exact prompt row wins over it. A row with a profile
    # applies only to runs of that profile.
    bare = re.sub(r'-ctx[0-9]+$', '', p)
    rows = [x for x in records if x['engine'] == engine and x['mode'] == mode
            and x.get('profile', profile) == profile]
    return (next((x for x in rows if x['prompt_id'] in (p, bare)), None)
            or next((x for x in rows if x['prompt_id'] == '*' and not p.startswith(('prefill-', 'http-'))), None))


def report(path):
    d = read(path)
    validate(d)
    lines = ['# Engine benchmark', '', f"Engine: `{d['engine']}` | kit {d['kit_version']} | commit `{d['repo_commit']}`",
             '', f"Identity: `{d['identity_sha256']}` | reference key: `{d.get('reference_sha256', 'unrecorded')}` | status: **{d['status']}**", '',
             'WARNING: model content hashes were skipped; models_verified: false.' if not d.get('models_verified', True) else 'Model hashes verified (full read or metadata cache).', '',
             'README values are context, not qualification of a different workload or identity.', '',
             f"Prompt set: {d.get('prompt_set', {}).get('name', 'legacy')} ({d.get('prompt_set', {}).get('sha256', 'unrecorded')}).",
             ('DeepSeek decode uses native timing; same method, different prompts (workload differs). Optional serve uses HTTP end-to-end timing.' if d['engine']=='deepseek-pair' else
              'same method, different prompts' if d.get('prompt_set', {}).get('name') == 'kit' else 'Verify workload equality before comparing README rows.'),
             f"Model loads per rank: {d.get('model_loads', 'unrecorded')}; total wall: {d.get('wall_s', 'unrecorded')} s.", '',
             f"CPU: {d['hardware'].get('cpu_model')} | GPU: {', '.join(d['hardware'].get('gpu_names', []))}",
             f"RAM: {d['hardware'].get('ram_bytes')} bytes | kernel: {d['hardware'].get('kernel')} | ROCm: {d['hardware'].get('rocm_version')}", '',
             '| Prompt / mode | README | Measured median (min-max) | Ratio measured/README | Check |',
             '|---|---:|---:|---:|---|']
    for m in d['measurements']:
        b = None if m.get('timing_scope')=='http-end-to-end' and not m['prompt_id'].startswith('http-c') else baseline(d['engine'], m['prompt_id'], m['mode'], d.get('params', {}).get('profile'))
        expected = f"{b['value']:g} {b['unit']}" if b else 'unpublished'
        if b and 'pairs_faster' in b and 'pairs' in b:
            expected += f" (ABBA {b['pairs_faster']}/{b['pairs']})"
        ratio = m['median'] / b['value'] if b else None
        within = b and abs(ratio - 1) <= b['tolerance'] and m['unit'] == b['unit']
        check = 'n/a' if not b else 'PASS' if within else 'WARN'
        if b and not b.get('workload_match'):
            check += ' (workload differs/unspecified)'
        scope = ' [HTTP end-to-end]' if m.get('timing_scope')=='http-end-to-end' else ''
        lines.append(f"| {m['prompt_id']} / {m['mode']}{scope} (n={m['repeat_count']}) | {expected} | {m['median']:.3f} ({m['min']:.3f}-{m['max']:.3f}) {m['unit']} | {f'{ratio:.3f}' if ratio else '--'} | {check} |")
    lines += ['', '| Prompt / mode / requested length | Exactness |', '|---|---|']
    for e in d['exactness']:
        witness = f" (draft witness: {e['draft_witness']['status']}, accepted {e['draft_witness']['accepted']})" if 'draft_witness' in e else ''
        lines.append(f"| {e['prompt_id']} / {e['mode']} / {e['generate_length']} | {e['verdict']}{witness} |")
    for c in d.get('digest_checks', []):
        lines.append(f"| {c['check']} / {c['identity']} / KV `{c['kv'][:16]}` | {c['verdict']} |")
    lines += ['', 'NO-REFERENCE is unqualified. Candidate references require maintainer review and an independent run.', '']
    for s in d.get('skipped', []):
        lines.append('Skipped: ' + s)
    for x in d.get('not_run', []):
        if x['item'] != 'skipped': lines.append(f"Not run: {x['item']} ({x['reason']})")
    policy = d.get('cache_policy')
    if policy:
        lines.append(f"Cache state: {', '.join(k + ' ' + v for k, v in policy['states'].items())}; max cached_tokens "
                     f"{policy['max_cached_tokens']} (allowed {policy['allowed_cached_tokens']}"
                     f"{', reuse declared' if policy['declared_reuse'] else ''})"
                     f"{'; NONCONFORMING: ' + '; '.join(policy['reasons']) if policy['nonconforming'] else ''}")
    census = d.get('dispatch_census')
    if census:
        lines.append(f"Dispatch census: {census['dispatches']} dispatches, GPU busy {census['trace']['gpu_busy_share']:.1%} of span"
                     + (f", {census['per_token']['launches']:.1f} launches/token" if 'per_token' in census else ''))
    lines += ['', 'Attach results.json and results.md to a GitHub issue. Review logs before attaching them.', '']
    path.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def http(out, port):
    base = 'http://localhost:' + str(int(port))
    end = time.perf_counter() + 600
    while time.perf_counter() < end:
        try:
            # The engine we launched must have opened its listener; an unrelated /health is insufficient.
            log = (out / 'server.log').read_text(encoding='utf-8', errors='replace')
            if 'strix serve: listening' not in log and 'READY rank 0' not in log and not any(o.get('event') == 'READY' for o in objects(log)):
                time.sleep(0.5)
                continue
            with urllib.request.urlopen(base + '/health', timeout=2) as r:
                if r.status == 200:
                    break
        except OSError:
            time.sleep(0.5)
    else:
        raise ValueError('HTTP server readiness timeout')
    prompt_names = list(prompt_texts()) if (out / 'prompts.json').exists() else ['chat', 'code', 'prose']
    prompt_ids = {p: ids_read(out / (p + '.ids')) for p in prompt_names}

    def request(index):
        prompt = prompt_names[index % len(prompt_names)]
        payload = json.dumps({'model': 'benchmark', 'prompt': prompt_ids[prompt], 'max_tokens': int(os.environ.get('BENCH_GEN_TOKENS', '128')),
                              'temperature': 0, 'stream': True, 'return_token_ids': True,
                              'stream_options': {'include_usage': True}}).encode()
        req = urllib.request.Request(base + '/v1/completions', payload, {'Content-Type': 'application/json'})
        start = time.perf_counter()
        first = None
        tokens = None
        output_ids = None
        cached = None
        timing = {}
        with urllib.request.urlopen(req, timeout=300) as response:
            for raw in response:
                if not raw.startswith(b'data:') or raw.strip() == b'data: [DONE]':
                    continue
                obj = json.loads(raw[5:])
                if 'error' in obj:
                    raise ValueError('HTTP generation error')
                for choice in obj.get('choices', []):
                    if choice.get('text') and first is None:
                        first = time.perf_counter() - start
                if obj.get('usage'):
                    tokens = obj['usage'].get('completion_tokens')
                    if obj['usage'].get('prompt_tokens') != len(prompt_ids[prompt]):
                        raise ValueError('server prompt token count differs from shipped input IDs')
                    # Prefix/prompt cache reuse is visible only through the engine's usage fields.
                    details = obj['usage'].get('prompt_tokens_details')
                    if isinstance(details, dict) and isinstance(details.get('cached_tokens'), int):
                        cached = details['cached_tokens']
                if obj.get('strix'):
                    output_ids = obj['strix'].get('token_ids')
                    for key in ('queue_ms', 'prefill_ms', 'decode_ms'):
                        if isinstance(obj['strix'].get(key), (int, float)):
                            timing['engine_' + key] = obj['strix'][key]
        if first is None or not isinstance(tokens, int) or tokens <= 0 or not output_ids or len(output_ids) != tokens:
            raise ValueError('SSE response lacks first content, token IDs or matching completion usage')
        return {'request': index, 'prompt_id': prompt, 'prompt_sha256': id_hash(prompt_ids[prompt]),
                'token_ids_sha256': id_hash(output_ids), 'ttft_s': first, 'completion_tokens': tokens,
                'elapsed_s': time.perf_counter() - start,
                # completion_tokens is the engine's usage count; the client count is the received ID vector.
                'client_token_count': len(output_ids), 'cached_tokens': cached,
                'prompt_tokens': len(prompt_ids[prompt]), **timing}

    measurements = []
    for c in map(int, os.environ.get('BENCH_CONCURRENCY', '1,4,8').split(',')):
        rates, ttfts, requests = [], [], []
        for repeat_index in range(int(os.environ.get('BENCH_WARMUP', '0')) + int(os.environ['BENCH_REPEATS'])):
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                results = list(pool.map(request, range(8)))
            elapsed = time.perf_counter() - start
            if repeat_index < int(os.environ.get('BENCH_WARMUP', '0')): continue
            rates.append(sum(x['completion_tokens'] for x in results) / elapsed)
            ttfts.extend(x['ttft_s'] for x in results)
            requests.append(results)
        flat = [x for group in requests for x in group]
        cache = {'cache_state': cache_state([x['cached_tokens'] for x in flat]),
                 'cached_tokens_max': max((x['cached_tokens'] or 0) for x in flat)}
        # Constituent prompts and requested generation length of this synthetic http-cN row.
        cache['workload'] = {'prompts': [dict(id=i, sha256=h, tokens=n) for i, h, n in
                                         sorted({(x['prompt_id'], x['prompt_sha256'], x['prompt_tokens']) for x in flat})],
                             'generate_length': int(os.environ.get('BENCH_GEN_TOKENS', '128'))}
        measurements.extend([{'prompt_id': f'http-c{c}', 'mode': 'aggregate-wall', 'unit': 'tok/s', 'timing_scope':'http-end-to-end', 'rate_source': 'client-clock', **cache, **stats(rates)},
                             {'prompt_id': f'http-c{c}', 'mode': 'ttft', 'unit': 's', 'timing_scope':'http-end-to-end', 'rate_source': 'client-clock', **cache, **stats(ttfts)}])
        # Engine-timed rows (strix receipt decode_ms/prefill_ms) beside the client clock; absent rows go to not_run.
        if flat and all(x.get('engine_decode_ms', 0) > 0 and x['completion_tokens'] > 1 for x in flat):
            measurements.append({'prompt_id': f'http-c{c}', 'mode': 'engine-decode', 'unit': 'tok/s', 'timing_scope': 'engine',
                                 'rate_source': 'engine', **cache,
                                 **stats([x['completion_tokens'] * 1000 / x['engine_decode_ms'] for x in flat])})
        if flat and all(x.get('engine_prefill_ms', 0) > 0 for x in flat):
            measurements.append({'prompt_id': f'http-c{c}', 'mode': 'engine-ttft', 'unit': 's', 'timing_scope': 'engine',
                                 'rate_source': 'engine', **cache,
                                 **stats([(x.get('engine_queue_ms', 0) + x['engine_prefill_ms']) / 1000 for x in flat])})
        dump(out / f'http-c{c}.json', {'concurrency': c, 'max_tokens': int(os.environ.get('BENCH_GEN_TOKENS', '128')), 'requests': requests})
    dump(out / 'http-measurements.json', measurements)


def main(args):
    action = args[0]
    if action == 'help':
        interface.help_text(); return 0
    if action == 'params':
        p = json.loads(args[1]); p.update(timeout=int(args[2]), json=args[3]=='1', md=args[4]=='1', write_refs=args[5]=='1', dry_run=args[6]=='1', verify_models=args[7], peer=bool(args[8]))
        for key,value,default in zip(('bin_dir','model_dir','peer_bin_dir','peer_model_dir','peer_repo'), args[9:], ('build','models','build','models',str(ROOT))):
            p[key] = 'default' if value == default else 'configured'
        p['out'] = 'run-directory'
        print(json.dumps(p)); return 0
    if action == 'resolve':
        interface.resolve(args[1:], KIT); return 0
    if action == 'list':
        rows = interface.list_engines(sys.modules[__name__])
        print(json.dumps({'schema_version': 1, 'engines': rows}) if args[1] == '1' else '\n'.join(x['engine'] + ': ' + ('available' if x['available'] else '; '.join(x['reasons'])) for x in rows))
        return 0
    if action == 'all': return interface.all_engines(args[1:], sys.modules[__name__])
    if action == 'plan':
        interface.plan(args[1], args[2] == '1', sys.modules[__name__]); return 0
    if action == 'ingest-batch':
        ingest_batch(Path(args[1]), read(args[2]), Path(args[3])); return 0
    if action == 'check-result':
        obj=json.loads(args[1]); obj['pair_ok']=args[2]=='0'; obj['ok']=obj['ok'] and obj['pair_ok']
        if not obj['pair_ok']: obj['problems'].append('pair verification failed')
        print(json.dumps(obj)); return int(not obj['ok'])
    if action == 'check-json':
        problems, hw = precheck()
        print(json.dumps({'schema_version': 1, 'engine': ENGINE.replace('-pair',''), 'ok': not problems, 'problems': problems, 'warnings': build_warnings(), 'hardware': hw}))
        return int(bool(problems))
    if action == 'hash-models':
        print(json.dumps({'schema_version': 1, 'engine': ENGINE.replace('-pair',''), 'models_verified': True, 'model_files': model_inventory('full')})); return 0
    if action == 'model-hash':
        cache = ModelHashes(MODELS, os.environ.get('BENCH_VERIFY_MODELS', 'cached'), sha)
        # MTP admission requires a real target hash even if inventory hashing is skipped.
        if cache.mode == 'skip': cache.mode = 'cached'
        print(cache.digest(args[1])); cache.save(); return 0
    if action == 'hash':
        print(sha(args[1]))
    elif action == 'check':
        problems, hw = precheck()
        for warning in build_warnings(): print('WARNING: ' + warning, file=sys.stderr)
        print('Prerequisites OK' if not problems else '\n'.join('MISSING: ' + x for x in problems))
        print(json.dumps({'engine': ENGINE, 'ok': not problems, 'problems': problems, 'hardware': hw}))
        return int(bool(problems))
    elif action == 'inventory':
        print(json.dumps(inventory()))
    elif action == 'compare':
        a, b = read(args[1]), read(args[2])
        for key in ('repo_commit', 'binary_sha256s', 'model_files', 'build_options'):
            if a[key] != b[key]:
                raise ValueError('pair identity mismatch: ' + key)
        if any(a['hardware'].get(v) != b['hardware'].get(v) for v in ('rocm_version', 'hip_version')):
            raise ValueError('pair ROCm version differs')
        print('Pair content and prerequisites OK\n{"pair_ok":true}')
    elif action == 'qwen-admission':
        qwen_admission(Path(args[1]).read_text(encoding='utf-8'))
    elif action == 'effective':
        record_effective(Path(args[1]), args[2], Path(args[3]), Path(args[4]), *args[5:])
    elif action == 'resolved-env':
        import shlex
        config=parse_effective(Path(args[1]).read_text(encoding='utf-8'),args[2:])
        values=[x for key in config.get('unset_environment',[]) for x in ('-u',key)]
        def shell_path(value):
            return re.sub(r'([A-Za-z]):/',lambda m:'/'+m[1].lower()+'/',value) if os.name=='nt' else value
        values += [key+'='+shell_path(value) for key,value in config['environment'].items()]
        print('rank_env=('+' '.join(shlex.quote(x) for x in values)+')')
    elif action == 'initialize':
        initialize(Path(args[1]))
    elif action == 'prompts':
        prompts(Path(args[1]))
    elif action == 'ids-lines':
        print('\n'.join(map(str, ids_read(args[1]))))
    elif action == 'receipt-field':
        value = receipt(args[1])[args[2]]
        if not re.fullmatch('[a-f0-9]{64}', value):
            raise ValueError('invalid receipt identity')
        print(value)
    elif action == 'ingest':
        ingest(Path(args[1]), args[2], args[3], int(args[4]), Path(args[5]))
    elif action == 'finalize':
        out = Path(args[1])
        finalize(out, args[2] == '1', args[3] if len(args) > 3 else 'complete')
        if (out / 'http-measurements.json').exists():
            d = read(out / 'results.json')
            d['measurements'].extend(read(out / 'http-measurements.json'))
            dump(out / 'results.json', d)
            report(out / 'results.json')
        # Results are written first; a MISMATCH verdict or a refused cache-reuse run must still fail `run`.
        final = read(out / 'results.json')
        if final['status'] == 'mismatch' or final.get('cache_policy', {}).get('refused'):
            return 1
    elif action == 'report':
        report(Path(args[1]))
    elif action == 'validate':
        validate(read(args[1]))
        print('schema OK')
    elif action == 'sanitize':
        sanitize_file(args[1])
    elif action == 'sanitize-all':
        for p in Path(args[1]).glob('*.log'):
            sanitize_file(p)
    elif action == 'http':
        http(Path(args[1]), args[2])
    elif action == 'free-port':
        port = int(args[1])
        if not 0 < port <= 65535:
            raise ValueError('HTTP port must be in 1..65535')
        with socket.socket() as s:
            # POSIX SO_REUSEADDR still refuses a port with a live listener; it only ignores the previous arm's
            # TIME_WAIT connections, which would otherwise fail a sequential arm on the same port.
            if os.name == 'posix':
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('localhost', port))
    elif action == 'not-run':
        path = Path(args[1]) / 'not-run.json'
        items = read(path) if path.exists() else []
        items.append({'item': args[2], 'reason': args[3]})
        dump(path, items)
    elif action == 'pairs':
        # pairs OUT.json A=results.json B=results.json B=... A=... (process order)
        arms = [tuple(x.split('=', 1)) for x in args[2:]]
        if any(len(x) != 2 or x[0] not in ('A', 'B') for x in arms): raise ValueError('pairs takes ARM=results.json with ARM A or B')
        result = pairs(arms)
        dump(args[1], result)
        print(json.dumps(result))
    elif action == 'census':
        # census RUN_DIR TRACE [dispatch_census.py options]
        attach_census(Path(args[1]), Path(args[2]), args[3:])
    elif action == 'skipped':
        path = Path(args[1]) / 'skipped.json'
        notes = read(path) if path.exists() else []
        dump(path, notes + [args[2]] if args[2] not in notes else notes)
    elif action == 'nemotron-http':
        nemotron.serve_arm(sys.modules[__name__], Path(args[1]), args[2], args[3], args[4])
    elif action == 'nemotron-verify':
        dump(Path(args[1]) / 'verify-r8.json', nemotron.parse_verify(Path(args[2]).read_text(encoding='utf-8', errors='replace'),
                                                                     nemotron.selected_profile()))
    else:
        raise ValueError('unknown helper command')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except (ValueError, KeyError, OSError, subprocess.SubprocessError, AssertionError, TypeError, IndexError, ArithmeticError) as exc:
        print('bench: ' + sanitize(str(exc)), file=sys.stderr)
        sys.exit(1)
