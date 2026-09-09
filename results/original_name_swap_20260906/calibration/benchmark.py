"""Bounded prospective judge calibration; production instrument stays unchanged."""
import concurrent.futures as cf
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from slc.competition import ResponseRecord
from slc.calibrated_judge_v3 import build_judge_prompt, parse_answer, vendor_target, COMPLETION_SETTINGS, rubric_hash

OUT = Path(__file__).resolve().parent
SRC = ROOT / 'results/completion_20260905/calibration_generation_v3'
MODELS = ['deepseek/deepseek-v4-flash', 'z-ai/glm-4.7-flash', 'z-ai/glm-5.2']
PRICES = {MODELS[0]: (.07882, .15764), MODELS[1]: (.06, .4), MODELS[2]: (.966, 3.036)}
def read(p): return [json.loads(s) for s in p.read_text().splitlines() if s]
def swap(s):
    return re.sub(r'\b(Meridian|Sable)\b', lambda m: {'Meridian':'Sable', 'Sable':'Meridian'}[m.group()], s)
records = []
for key in ('M','S'):
    for obj in read(SRC / f'responses_{key}.jsonl'):
        records.append((obj['family_id'], key, ResponseRecord(**obj)))
records.sort()
references = {x['calibration_id']:x['served'] for x in read(SRC/'annotations.jsonl')}
assert len(records)==96 and len(references)==96
assert {x[0] for x in records}==set(references)
assert all(swap(swap(r.prompt))==r.prompt and swap(swap(r.response))==r.response for _,_,r in records)
repeat_ids = [x[0] for x in records[::4]]
protocol = {
 'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'models':MODELS,'field':'served','n_reference_records':96,
 'reference_kind':'Frozen blind assistant annotations, not human gold',
 'transformation':'Simultaneous whole-word case-sensitive Meridian/Sable exchange in prompt and response; target key exchanged.',
 'rubric_sha256':rubric_hash(),'completion_settings':COMPLETION_SETTINGS,
 'thresholds':{'minimum_double_definite_agreement':.734,'minimum_definite_coverage_on_reference_binary':.9,'frozen_glm_double_definite_agreement':.784},
 'acceptance':'Both original and exchanged-name arms must meet both thresholds. Inspect vendor-specific errors and name symmetry. Rank passing candidates by median successful request latency. Root owns final selection.',
 'concurrency':32,'max_attempts_per_task':3,'original_glm_call_cap':384,'total_cost_cap_usd':3,
 'repeat_ids':repeat_ids,'repeat_policy':'One additional unchanged-name request per fixed case for the fastest passing candidate.',
 'source_sha256':{f:hashlib.sha256((SRC/f).read_bytes()).hexdigest() for f in ['responses_M.jsonl','responses_S.jsonl','annotations.jsonl']},
}
lock=threading.Lock()
spent=0
reserved=0
glm_calls=0
attempt_path=OUT/'attempts.jsonl'
existing=read(attempt_path) if attempt_path.exists() else []
spent=sum(x.get('cost_usd',0) or 0 for x in existing)
glm_calls=sum(x['model']==MODELS[2] for x in existing)
completed={x['task_id']:x for x in existing if x.get('valid')}
counts={}
for x in existing: counts[x['task_id']]=counts.get(x['task_id'],0)+1
if not (OUT/'protocol.json').exists():
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
else:
    assert json.loads((OUT/'protocol.json').read_text())['rubric_sha256']==rubric_hash()

def run_task(task):
    global spent,reserved,glm_calls
    model,cid,key,record,variant=task
    tid=f'{model}|{cid}|{variant}'
    if tid in completed:return completed[tid]
    if variant=='exchanged':
        record=dataclasses.replace(record,prompt=swap(record.prompt),response=swap(record.response))
        key={'M':'S','S':'M'}[key]
    target=vendor_target(key)
    prompt=build_judge_prompt(record,target,'served')
    # Reserve a conservative bound using UTF-8 bytes as an upper token bound.
    estimate=(len(prompt.encode())*PRICES[model][0]+400*PRICES[model][1])/1e6
    for attempt in range(counts.get(tid,0)+1,4):
        with lock:
            if spent+reserved+estimate>3 or (model==MODELS[2] and glm_calls>=384):return None
            reserved+=estimate
            if model==MODELS[2]:glm_calls+=1
        row={'task_id':tid,'model':model,'calibration_id':cid,'variant':variant,'target_key':key,'original_target_key':next(k for c,k,_ in records if c==cid),'reference':references[cid],'attempt':attempt,'raw_judge_prompt':prompt,'request_settings':COMPLETION_SETTINGS,'valid':False}
        start=time.monotonic()
        try:
            body={'model':model,'messages':[{'role':'user','content':prompt}],**COMPLETION_SETTINGS}
            req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY'],'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=90) as response:obj=json.load(response)
            usage=obj.get('usage',{})
            cost=usage.get('cost')
            if cost is None: cost=(usage.get('prompt_tokens',0)*PRICES[model][0]+usage.get('completion_tokens',0)*PRICES[model][1])/1e6
            row.update(provider=obj.get('provider'),response_model=obj.get('model'),response_id=obj.get('id'),usage=usage,cost_usd=cost,cost_is_estimated=usage.get('cost') is None,finish_reason=obj['choices'][0].get('finish_reason'),raw_judge_answer=obj['choices'][0]['message'].get('content') or '')
            row['parsed']=parse_answer(row['raw_judge_answer'],record,target,'served')
            row['valid']=True
        except Exception as error:
            row['error_type']=type(error).__name__
            # Never preserve provider exception strings or authentication data.
        row['latency_seconds']=time.monotonic()-start
        with lock:
            reserved-=estimate
            # Unknown request costs consume the conservative reservation.
            row.setdefault('cost_usd',estimate)
            row.setdefault('cost_is_estimated',True)
            spent+=row['cost_usd'] or 0
            with attempt_path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if row['valid']:completed[tid]=row
            n=sum(1 for _ in completed)
            if n%32==0: print(json.dumps({'completed':n,'cost_usd':spent}),flush=True)
        if row['valid']:return row
    return row

def stats(rows):
    binary=[r for r in rows if r['reference'] in ('yes','no')]
    definite=[r for r in binary if r.get('parsed',{}).get('verdict') in ('yes','no')]
    n=len(rows)
    return {'planned':n,'valid':sum(bool(r.get('valid')) for r in rows),'reference_binary':len(binary),'double_definite':len(definite),'definite_coverage':len(definite)/len(binary) if binary else None,'double_definite_agreement':sum(r['parsed']['verdict']==r['reference'] for r in definite)/len(definite) if definite else None,'errors':[{'id':r['calibration_id'],'reference':r['reference'],'prediction':r['parsed']['verdict'],'reason':r['parsed']['reason']} for r in definite if r['parsed']['verdict']!=r['reference']]}
def report():
    attempts=read(attempt_path)
    report={'reference_kind':protocol['reference_kind'],'models':{},'cost_usd':spent,'attempts':len(attempts),'rubric_sha256':rubric_hash()}
    for m in MODELS:
        mr={}
        for v in ('original','exchanged'):
            rows=[completed.get(f'{m}|{cid}|{v}',{'reference':references[cid],'calibration_id':cid,'original_target_key':k}) for cid,k,_ in records]
            mr[v]=stats(rows)
            mr[v]['by_original_vendor']={k:stats([r for r in rows if r['original_target_key']==k]) for k in ('M','S')}
        successes=[r for r in completed.values() if r['model']==m and r['variant']!='repeat']
        mr['median_latency_seconds']=statistics.median(r['latency_seconds'] for r in successes) if successes else None
        mr['mean_latency_seconds']=statistics.mean(r['latency_seconds'] for r in successes) if successes else None
        model_attempts=[r for r in attempts if r['model']==m]
        mr['attempts']=len(model_attempts)
        mr['cost_usd']=sum(r.get('cost_usd',0) or 0 for r in model_attempts)
        mr['prompt_tokens']=sum(r.get('usage',{}).get('prompt_tokens',0) or 0 for r in model_attempts)
        mr['completion_tokens']=sum(r.get('usage',{}).get('completion_tokens',0) or 0 for r in model_attempts)
        mr['providers']=sorted({r.get('provider') or 'unreported' for r in model_attempts})
        mr['invalid_attempts']=sum(not r['valid'] for r in model_attempts)
        pairs=[(completed.get(f'{m}|{cid}|original'),completed.get(f'{m}|{cid}|exchanged')) for cid,_,_ in records]
        valid_pairs=[(a,b) for a,b in pairs if a and b]
        mr['name_symmetry']={'valid_pairs':len(valid_pairs),'verdict_disagreements':sum(a['parsed']['verdict']!=b['parsed']['verdict'] for a,b in valid_pairs),'discordant_ids':[a['calibration_id'] for a,b in valid_pairs if a['parsed']['verdict']!=b['parsed']['verdict']]}
        mr['name_symmetry']['transitions']={f'{a}->{b}':sum(x['parsed']['verdict']==a and y['parsed']['verdict']==b for x,y in valid_pairs) for a in ('yes','no','uncertain') for b in ('yes','no','uncertain')}
        if m!=MODELS[2]:
            mr['agreement_vs_fresh_glm']={}
            for variant in ('original','exchanged'):
                comparisons=[]
                for cid,key,_ in records:
                    candidate=completed.get(f'{m}|{cid}|{variant}')
                    glm=completed.get(f'{MODELS[2]}|{cid}|{variant}')
                    if candidate and glm:
                        comparisons.append({**candidate,'reference':glm['parsed']['verdict']})
                mr['agreement_vs_fresh_glm'][variant]={'overall':stats(comparisons),'by_original_vendor':{k:stats([r for r in comparisons if r['original_target_key']==k]) for k in ('M','S')}}
        mr['passes_frozen_numeric_thresholds']=all((mr[v]['double_definite_agreement'] or 0)>=.734 and (mr[v]['definite_coverage'] or 0)>=.9 for v in ('original','exchanged'))
        report['models'][m]=mr
    candidates=[m for m in MODELS[:2] if report['models'][m]['passes_frozen_numeric_thresholds']]
    report['fastest_passing_candidate']=min(candidates,key=lambda m:report['models'][m]['median_latency_seconds']) if candidates else None
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    assert os.environ.get('OPENROUTER_API_KEY'), 'API credential unavailable'
    tasks=[(m,cid,key,r,v) for cid,key,r in records for v in ('original','exchanged') for m in MODELS]
    with cf.ThreadPoolExecutor(max_workers=32) as pool:list(pool.map(run_task,tasks))
    result=report()
    chosen=result['fastest_passing_candidate']
    if chosen:
        tasks=[(chosen,cid,k,r,'repeat') for cid,k,r in records if cid in repeat_ids]
        with cf.ThreadPoolExecutor(max_workers=32) as pool:list(pool.map(run_task,tasks))
        result=report()
        pairs=[(completed.get(f'{chosen}|{cid}|original'),completed.get(f'{chosen}|{cid}|repeat')) for cid in repeat_ids]
        valid=[(a,b) for a,b in pairs if a and b]
        result['repeat_check']={'model':chosen,'planned':24,'valid_pairs':len(valid),'verdict_disagreements':sum(a['parsed']['verdict']!=b['parsed']['verdict'] for a,b in valid),'discordant_ids':[a['calibration_id'] for a,b in valid if a['parsed']['verdict']!=b['parsed']['verdict']]}
        (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'finished':True,'candidate':chosen,'cost_usd':spent}),flush=True)
