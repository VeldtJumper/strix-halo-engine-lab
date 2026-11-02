"""Validated public CLI and multi-prompt execution plans."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ENGINES = ['qwen', 'deepseek', 'glm', 'minimax', 'deepseek-rtx', 'nemotron']
ALIASES = {'deepseek': 'deepseek-pair', 'glm': 'glm-pair', 'minimax': 'minimax-pair'}
ALLOW = {'qwen': ['default', 'plain', 'baseline', 'gsq', *['mtp-k'+str(i) for i in range(1,9)]],
         'deepseek-pair': ['default','greedy','dspark','baseline'], 'glm-pair': ['r1b','r1','mtp','baseline'],
         'minimax-pair': ['default','greedy','dspark','baseline'], 'deepseek-rtx': ['default','baseline'],
         'nemotron': ['default','fast','quality']}

def resolve(args, kit):
    engine,suite,ctx,gen,conc,rep,warm,profile,prompts,only,serve = args
    if engine not in ALLOW:
        raise ValueError('unknown engine')
    profile = profile or ('r1b' if engine == 'glm-pair' else 'default')
    if profile not in ALLOW[engine]:
        raise ValueError('profile must be one of: '+', '.join(ALLOW[engine]))
    def positive(x):
        if not re.fullmatch('[1-9][0-9]*',str(x)): raise ValueError('expected positive integer')
        return int(x)
    contexts=[]
    for x in (ctx or ('1k' if suite=='quick' else ('1k,4k,8k,16k,32k' if engine=='qwen' else '1k,8k,16k,32k' if engine=='deepseek-pair' else '1k,8k,32k'))).lower().split(','):
        contexts.append(positive(x[:-1] if x.endswith('k') else x)*(1024 if x.endswith('k') else 1))
    levels=[positive(x) for x in conc.split(',')]
    if any(x>8 for x in levels): raise ValueError('serve concurrency is limited to 1..8 by these servers')
    generation=positive(gen or ('128' if suite=='quick' or engine=='deepseek-rtx' else '64' if engine=='glm-pair' else '256'))
    if generation<2: raise ValueError('gen-tokens must be at least 2 for decode timing')
    repeats=positive(rep or ('1' if suite=='quick' else '3'))
    warmup=int(warm or ('0' if suite=='quick' else '1'))
    if warmup<0: raise ValueError('warmup must be nonnegative')
    modes=only.split(',')
    if not modes or any(x not in ('decode','prefill','serve','exactness') for x in modes): raise ValueError('invalid --only subset')
    if serve=='1' and 'serve' not in modes: modes.append('serve')
    texts=prompt_set(prompts,kit)
    # Validate even for dry runs, never execute the contents of a prompt file.
    params=dict(engine=engine.replace('-pair',''),suite=suite,contexts=contexts,gen_tokens=generation,
                concurrency=levels,repeat=repeats,warmup=warmup,profile=profile,
                prompts='kit' if prompts=='kit' else 'file',only=modes,serve='serve' in modes)
    vals=dict(gen_tokens=generation,repeats=repeats,warmup=warmup,profile=profile,only=','.join(modes),
              serve=int('serve' in modes),resolved_params=json.dumps(params))
    for key,value in vals.items(): print(key+'='+shlex.quote(str(value)))
    print('context_values=('+ ' '.join(map(str,contexts))+')')
    print('prompt_names=('+ ' '.join(shlex.quote(x['id']) for x in texts)+')')

def prompt_set(source,kit):
    path=kit/'prompts/kit.json' if source=='kit' else Path(source)
    if path.suffix.lower()=='.json':
        rows=json.loads(path.read_text(encoding='utf-8'))
        if isinstance(rows,dict): rows=rows['prompts']
    else:
        rows=[dict(id='prompt-'+str(i+1),category='custom',text=t) for i,t in enumerate(path.read_text(encoding='utf-8').splitlines()) if t.strip()]
    if not rows: raise ValueError('empty prompt set')
    seen=set()
    for row in rows:
        if not re.fullmatch('[A-Za-z0-9_-]+',row['id']) or row['id'] in seen: raise ValueError('invalid/duplicate prompt id')
        if not isinstance(row['text'],str) or not row['text'].strip(): raise ValueError('empty prompt')
        if not isinstance(row['category'],str): raise ValueError('category must be text')
        seen.add(row['id'])
    return rows

def help_text():
    print('''bench.sh list [--json] | check --engine E [--peer HOST] [--json]
bench.sh run --engine qwen|deepseek|glm|minimax|deepseek-rtx|nemotron|all [options]
Aliases: deepseek-pair, glm-pair, minimax-pair.
--contexts 1k,8k,32k  --gen-tokens N  --concurrency 1,4,8
--prompts kit|FILE   --repeat N       --warmup N
--profile P         --only decode,prefill,serve,exactness
--model-dir DIR     --bin-dir DIR    --peer HOST
--peer-model-dir DIR --peer-bin-dir DIR --peer-repo DIR (defaults: local)
--verify-models full|cached|skip (cached default)
bench.sh hash-models --engine E [--model-dir DIR] (hashing only)
--out DIR --json|--md --dry-run --serve --timeout SECONDS --write-refs
--suite quick|full (full default; quick: 1k,128 tokens,repeat 1,warmup 0,no serve)
Explicit parameters override suite defaults. Serve is opt-in in either suite.
Examples:
  bench.sh run --engine qwen
  bench.sh run --engine deepseek --peer node-b
  bench.sh run --engine glm --peer node-b --profile r1
  bench.sh run --engine minimax --peer node-b
  bench.sh run --engine deepseek-rtx --bin-dir build-rtx
  bench.sh run --engine nemotron --profile fast --only exactness,serve
  bench.sh run --engine qwen --contexts 1k --only prefill --repeat 1 --warmup 0
  bench.sh run --engine minimax --peer node-b --only serve --concurrency 1,4,8
See docs/reproduce.md for assets, profile allowlists, timing and JSON contracts.''')

def plan(out,dry,k):
    p=json.loads(os.environ['BENCH_PARAMS']); engine=k.ENGINE
    names=[x['id'] for x in prompt_set(os.environ.get('BENCH_PROMPTS','kit'),k.KIT)]
    md=str(k.MODELS).replace('\\','/'); bd=str(k.BIN).replace('\\','/')
    out=str(out).replace('\\','/'); jobs=[]
    def exe(x): return str(k.executable(x)).replace('\\','/')
    def add(label,argv,rows): jobs.append((label,argv,rows))
    def order(ids):
        return [dict(prompt_id=x,warmup=r<p['warmup']) for r in range(p['warmup']+p['repeat']) for x in ids]
    decode=any(x in p['only'] for x in ('decode','exactness'))
    capacity=max(p['contexts'])+p['gen_tokens']+64
    if engine=='deepseek-pair':
        if decode:
            modes=['greedy','spec'] if p['profile']=='default' else ['spec' if p['profile']=='dspark' else 'greedy']
            for mode in modes:
                rows=[dict(x,mode=mode,generation=p['gen_tokens']) for x in order([f'{x}-ctx{n}' for n in p['contexts'] for x in names])]
                label='decode-'+mode
                prompt_file=out+'/'+label+'.ids'
                if not dry:
                    Path(prompt_file).write_text(''.join(','.join(map(str,k.ids_read(out+'/'+x['prompt_id']+'.ids')))+'\n' for x in rows),encoding='ascii')
                argv=[exe('strix-deepseek41-dspark-rank'),md+'/'+k.DS,md+'/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf','--rank','0','--transport','usb4auto','--mode',mode,'--prompt-file',prompt_file,'--generate',str(p['gen_tokens']),'--max-positions',str(capacity),'--draft','split','--vocab','auto:'+md+'/vocab']
                if p['profile']!='baseline': argv+=['--tp-attention','--engram-direct']
                add(label,argv,rows)
        if 'prefill' in p['only']:
            for n in p['contexts']:
                add('prefill-'+str(n),[exe('strix-deepseek41-prefill'),md+'/'+k.DS,'--rank','0','--world','2','--transport','usb4-gpu','--prompt',out+f'/prefill-{n}.ids','--rows',str(n),'--repeat',str(p['repeat']+p['warmup']),'--ttft-timing','--ttft-warmup','0'],[dict(x,mode='ttft',generation=1) for x in order(['prefill-'+str(n)])])
    elif engine=='glm-pair':
        base=[exe('strix-glm5-rank'),md+'/glm','--rank','0','--world','2','--transport','usb4gpu','--capacity',str(capacity)]
        if p['profile']=='r1': base+=['--identity','r1']
        if decode:
            rows=order(names)  # GLM README decode is a short prompt, separate from the prefill sweep.
            if p['profile']=='mtp':
                add('mtp',base+['--spec','1','--mtp','1','--spec-run','7','--decode',str(p['gen_tokens']),'--prompt',','.join(out+'/'+x['prompt_id']+'.u32' for x in rows)],
                    [dict(x,mode=mode,generation=p['gen_tokens']) for x in rows for mode in ('greedy','spec')])
            else:
                add('decode',base+['--decode',str(p['gen_tokens']),'--prompt',','.join(out+'/'+x['prompt_id']+'.u32' for x in rows)], [dict(x,mode='greedy',generation=p['gen_tokens']) for x in rows])
        if 'prefill' in p['only']:
            rows=order(['prefill-'+str(n) for n in p['contexts']])
            add('prefill',base+['--pf-ttft',','.join(out+'/'+x['prompt_id']+'.u32' for x in rows)],[dict(x,mode='ttft',generation=1) for x in rows])
    elif engine=='minimax-pair':
        base=[exe('strix-minimax-rank'),md+'/minimax','0','usb4gpu',out+'/native','--capacity',str(capacity)]
        argv=base[:]; rows=[]
        if decode:
            ids=[f'{x}-ctx{n}' for n in p['contexts'] for x in names]
            for i,x in enumerate(order(ids)):
                name='d'+str(i); argv+=['--session',f'{name}:{out}/{x["prompt_id"]}.u32:{p["gen_tokens"]-1}']
                rows.append(dict(x,name=name,mode='greedy',generation=p['gen_tokens']))
            if p['profile'] in ('default','dspark'):
                if os.environ.get('BENCH_MINIMAX_DRAFT_EXL3'): argv+=['--dspark-exl3',os.environ['BENCH_MINIMAX_DRAFT_EXL3']]
                for i,x in enumerate(order(ids)):
                    name='d'+str(i); argv+=['--spec',f'{name}:{out}/{x["prompt_id"]}.u32:{p["gen_tokens"]-1}:auto']
                    rows.append(dict(x,name=name,mode='spec',generation=p['gen_tokens']))
        if 'prefill' in p['only']:
            for i,x in enumerate(order(['prefill-'+str(n) for n in p['contexts']])):
                name='p'+str(i); argv+=['--session',f'{name}:{out}/{x["prompt_id"]}.u32:0']
                rows.append(dict(x,name=name,mode='ttft',generation=1))
        # Native app executes all sessions before any specs.
        rows.sort(key=lambda x:x['mode']=='spec')
        if rows: add('sessions',argv,rows)
    for label,argv,rows in jobs:
        manifest=out+'/'+label+'.manifest.json'
        if not dry: k.dump(manifest,rows)
        print('command=('+ ' '.join(shlex.quote(x) for x in argv)+')')
        print(('deepseek_native_decode ' if engine=='deepseek-pair' and label.startswith('decode-') else 'batch_measure ')+shlex.quote(label)+' '+shlex.quote(manifest))
    if 'serve' in p['only']:
        if engine=='glm-pair':
            print("printf '%s\\n' 'SKIP serve: GLM has no HTTP server' >&2")
            if not dry: k.dump(Path(out)/'skipped.json', ['HTTP: GLM has no HTTP server'])
        else:
            if engine=='minimax-pair':
                argv=[exe('strix-minimax-serve'),md+'/minimax','--rank','0','--transport','usb4gpu','--capacity',str(capacity),'--port',os.environ.get('BENCH_PORT','18880')]
                if p['profile'] not in ('greedy','baseline'): argv+=['--dspark',md+'/minimax/dspark']
                if os.environ.get('BENCH_MINIMAX_DRAFT_EXL3'): argv+=['--dspark-exl3',os.environ['BENCH_MINIMAX_DRAFT_EXL3']]
            else:
                argv=[exe('strix-deepseek41-serve'),md+'/'+k.DS,'--rank','0','--transport','usb4gpu','--context',str(capacity),'--http-port',os.environ.get('BENCH_PORT','18880')]
                if p['profile'] in ('greedy','baseline'): argv+=['--no-dspark']
            print('command=('+ ' '.join(shlex.quote(x) for x in argv)+')')
            print('batch_serve')


def list_engines(k):
    rows=[]
    for canonical in ENGINES:
        engine=ALIASES.get(canonical,canonical); binaries,assets=k.PROFILES[engine]
        missing=['binary-missing: '+b for b in binaries if not k.executable(b).is_file()]
        missing+=['file-missing: '+a for a in assets if not (k.MODELS/a).exists()]
        rows.append(dict(engine=canonical,available=not missing,reasons=missing,profiles=ALLOW[engine],binaries=binaries,assets=assets))
    return rows


def mismatch_record(text):
    try: doc=json.loads(text)
    except ValueError: return False
    return isinstance(doc,dict) and doc.get('status')=='mismatch'


def all_engines(args,k):
    json_mode='--json' in args; md_mode='--md' in args; records=[]; rc=0
    for row in list_engines(k):
        if row['engine'] in ('deepseek','glm','minimax') and '--peer' not in args:
            row['available']=False; row['reasons'].append('pair requires --peer HOST')
        if not row['available']:
            records.append(dict(engine=row['engine'],skipped=True,reasons=row['reasons']))
            print('SKIP '+row['engine']+': '+'; '.join(row['reasons']),file=sys.stderr); continue
        argv=list(args); argv[argv.index('--engine')+1]=row['engine']
        if '--out' in argv: argv[argv.index('--out')+1]+='/'+row['engine']
        result=subprocess.run(['bash',str(k.KIT/'bench.sh'),*argv],capture_output=True,text=True)
        sys.stderr.write(result.stderr)
        if result.returncode: rc=1
        if json_mode and result.returncode==0 and result.stdout.strip(): records.append(json.loads(result.stdout))
        # A MISMATCH run exits 1 but still prints its full result document.
        elif json_mode and result.returncode==1 and mismatch_record(result.stdout): records.append(json.loads(result.stdout))
        elif json_mode: records.append(dict(engine=row['engine'],returncode=result.returncode,dry_run='--dry-run' in args))
        elif not json_mode: print(result.stdout,end='')
    if json_mode: print(json.dumps(dict(schema_version=1,engines=records)))
    return rc
