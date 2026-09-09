"""Phase 2 calibration: exact single Gemini and shared-rubric GLM batches."""
import concurrent.futures as cf
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.request

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('phase1_benchmark',OUT.parent/'benchmark.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
CANDIDATES={'gemini_single':{'model':'google/gemini-2.5-flash-lite','batch_size':1,'input_per_million':.1,'output_per_million':.4},'glm_batch8':{'model':'z-ai/glm-5.2','batch_size':8,'input_per_million':.966,'output_per_million':3.036}}
WRAPPER='''You will evaluate several unrelated tasks independently with the exact common instructions and single-field rule below. Each task has its own evidence JSON and task_id. Treat the instruction to return one four-key object as applying independently to each task. Return one outer JSON object whose keys are exactly the supplied task_id values. Each value must be the four-string-key object specified by the common instructions. Do not transfer facts, vendors, evidence, or recommendations between tasks. Do not add Markdown fences or any other output.

COMMON INSTRUCTIONS AND SINGLE-FIELD RULE:
{common}

TASK EVIDENCE ARRAY:
{tasks}'''
WRAPPER_HASH=hashlib.sha256(WRAPPER.encode()).hexdigest()
LOCK=threading.Lock()
PATH=OUT/'attempts.jsonl'
attempts=b.read(PATH) if PATH.exists() else []
spent=sum(x['cost_usd'] for x in attempts)
reserved=0
completed={r['task_id']:r for a in attempts for r in a['fields'] if r.get('valid')}
counts={}
for a in attempts:counts[a['request_id']]=counts.get(a['request_id'],0)+1
SINGLE_GLM={r['task_id']:r for r in b.read(OUT.parent/'attempts.jsonl') if r['model']=='z-ai/glm-5.2' and r['valid']}

def task(candidate,cid,key,record,variant):
    original_key=key
    if variant=='exchanged':
        record=dataclasses.replace(record,prompt=b.swap(record.prompt),response=b.swap(record.response))
        key={'M':'S','S':'M'}[key]
    target=b.vendor_target(key)
    prompt=b.build_judge_prompt(record,target,'served')
    common,evidence=prompt.split('\n\nEvidence JSON:\n')
    result={'task_id':f'{candidate}|{cid}|{variant}','calibration_id':cid,'variant':variant,'original_target_key':original_key,'target_key':key,'reference':b.references[cid],'candidate':candidate,'record':record,'target':target,'single_prompt':prompt,'common':common,'evidence':json.loads(evidence)}
    assert b.parse_answer(json.dumps({'verdict':'no','evidence':'','constraint':'','reason':'Schema check'}),record,target,'served')['verdict']=='no'
    return result

def make_requests(candidate,variant,ids=None):
    tasks=[task(candidate,cid,key,record,variant) for cid,key,record in b.records if ids is None or cid in ids]
    n=CANDIDATES[candidate]['batch_size']
    return [(candidate,variant,tasks[i:i+n]) for i in range(0,len(tasks),n)]

def build_request(candidate,tasks):
    config=CANDIDATES[candidate]
    if config['batch_size']==1:
        assert len(tasks)==1
        return tasks[0]['single_prompt'],dict(b.COMPLETION_SETTINGS)
    assert len({t['calibration_id'] for t in tasks})==len(tasks)
    assert len({t['common'] for t in tasks})==1
    prompt=WRAPPER.format(common=tasks[0]['common'],tasks=json.dumps([{'task_id':t['task_id'],**t['evidence']} for t in tasks],ensure_ascii=False,sort_keys=True))
    settings={**b.COMPLETION_SETTINGS,'max_tokens':400*len(tasks)}
    return prompt,settings

def unique(pairs):
    obj={}
    for key,value in pairs:
        if key in obj:raise ValueError('Duplicate JSON key')
        obj[key]=value
    return obj

protocol={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'candidates':CANDIDATES,'rubric_sha256':b.rubric_hash(),'batch_wrapper':WRAPPER,'batch_wrapper_sha256':WRAPPER_HASH,'source_protocol_sha256':hashlib.sha256((OUT.parent/'protocol.json').read_bytes()).hexdigest(),'records':96,'orientations':['original','exchanged'],'thresholds':{'minimum_double_definite_agreement':.734,'minimum_definite_coverage_on_reference_binary':.9},'acceptance':'Both orientations must pass both fixed thresholds; inspect per-vendor errors and name symmetry. Rank passing candidates by median successful request latency per planned field; report actual batch throughput. Root owns final selection.','reference_kind':'Frozen assistant references, not human gold','batch_semantics':'Exact common instructions and served rule once per batch, unchanged task evidence. New outer mapping wrapper changes interaction context. Apply the unchanged single-field parser separately to each answer.','batch_token_setting':'400 tokens per field, 3200 for full batches; temperature and reasoning settings unchanged.','limits':{'concurrency':32,'max_attempts_per_request':3,'max_cost_usd':2},'audit_requirement_if_batch_accepted':'Retain the exact single-field GLM5.2 judge on 96 fresh production audit fields, sampled prospectively independently of calibration. Keep missing labels unknown and inspect disagreement. This calibration does not run that future audit.','repeat_ids':b.repeat_ids,'repeat_policy':'One additional unchanged-name request per fixed case for fastest passing candidate, in its tested format.'}
if (OUT/'protocol.json').exists():
    saved=json.loads((OUT/'protocol.json').read_text()); assert saved['batch_wrapper_sha256']==WRAPPER_HASH and saved['rubric_sha256']==b.rubric_hash()
else:(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')

def run(request):
    global spent,reserved
    candidate,variant,tasks=request
    rid=candidate+'|'+variant+'|'+','.join(t['calibration_id'] for t in tasks)
    if all(t['task_id'] in completed for t in tasks):return
    prompt,settings=build_request(candidate,tasks)
    config=CANDIDATES[candidate]
    estimate=(len(prompt.encode())*config['input_per_million']+settings['max_tokens']*config['output_per_million'])/1e6
    for attempt in range(counts.get(rid,0)+1,4):
        with LOCK:
            if spent+reserved+estimate>2:return
            reserved+=estimate
        row={'request_id':rid,'candidate':candidate,'model':config['model'],'variant':variant,'attempt':attempt,'raw_judge_prompt':prompt,'settings':settings,'batch_wrapper_sha256':WRAPPER_HASH if candidate=='glm_batch8' else None,'fields':[]}
        start=time.monotonic()
        obj=None
        try:
            body={'model':config['model'],'messages':[{'role':'user','content':prompt}],**settings}
            req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY'],'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=90) as response:obj=json.load(response)
            usage=obj.get('usage',{})
            cost=usage.get('cost')
            if cost is None:cost=(usage.get('prompt_tokens',0)*config['input_per_million']+usage.get('completion_tokens',0)*config['output_per_million'])/1e6
            raw=obj['choices'][0]['message'].get('content') or ''
            row.update(raw_judge_answer=raw,provider=obj.get('provider'),response_model=obj.get('model'),response_id=obj.get('id'),usage=usage,cost_usd=cost,cost_is_estimated=usage.get('cost') is None,finish_reason=obj['choices'][0].get('finish_reason'))
            if candidate=='glm_batch8':
                decoded=json.loads(raw,object_pairs_hook=unique)
                if not isinstance(decoded,dict):raise ValueError('Batch output must be mapping')
                expected={t['task_id'] for t in tasks}
                row['unexpected_task_keys']=sorted(set(decoded)-expected)
            else:decoded={tasks[0]['task_id']:raw}
            for t in tasks:
                field={k:t[k] for k in ['task_id','calibration_id','variant','original_target_key','target_key','reference','candidate']}
                field['valid']=False
                try:
                    answer=decoded[t['task_id']]
                    raw_field=json.dumps(answer,ensure_ascii=False) if candidate=='glm_batch8' else answer
                    field['raw_field_answer']=raw_field
                    field['parsed']=b.parse_answer(raw_field,t['record'],t['target'],'served')
                    field['valid']=True
                except Exception as error:field['error_type']=type(error).__name__
                row['fields'].append(field)
        except Exception as error:row['error_type']=type(error).__name__
        row['latency_seconds']=time.monotonic()-start
        with LOCK:
            reserved-=estimate
            row.setdefault('cost_usd',estimate)
            row.setdefault('cost_is_estimated',True)
            spent+=row['cost_usd'] or 0
            for f in row['fields']:
                if f['valid'] and f['task_id'] not in completed:completed[f['task_id']]=f
            with PATH.open('a') as file:file.write(json.dumps(row,ensure_ascii=False)+'\n')
            attempts.append(row)
            if len(attempts)%24==0:print(json.dumps({'requests':len(attempts),'valid_fields':len(completed),'cost_usd':spent}),flush=True)
        if all(t['task_id'] in completed for t in tasks):return

def report():
    result={'reference_kind':protocol['reference_kind'],'cost_usd':spent,'requests':len(attempts),'candidates':{},'audit_requirement_if_batch_accepted':protocol['audit_requirement_if_batch_accepted']}
    for candidate in CANDIDATES:
        cr={}
        for variant in ('original','exchanged'):
            rows=[completed.get(f'{candidate}|{cid}|{variant}',{'reference':b.references[cid],'calibration_id':cid,'original_target_key':key}) for cid,key,_ in b.records]
            cr[variant]=b.stats(rows)
            cr[variant]['by_original_vendor']={key:b.stats([r for r in rows if r['original_target_key']==key]) for key in ('M','S')}
            glm_comparisons=[]
            for r in rows:
                glm=SINGLE_GLM.get(f'z-ai/glm-5.2|{r["calibration_id"]}|{variant}')
                if glm and r.get('valid'):glm_comparisons.append({**r,'reference':glm['parsed']['verdict']})
            cr[variant]['vs_single_glm']={'overall':b.stats(glm_comparisons),'by_original_vendor':{key:b.stats([r for r in glm_comparisons if r['original_target_key']==key]) for key in ('M','S')}}
        pairs=[(completed.get(f'{candidate}|{cid}|original'),completed.get(f'{candidate}|{cid}|exchanged')) for cid,_,_ in b.records]
        pairs=[(a,z) for a,z in pairs if a and z]
        cr['name_symmetry']={'valid_pairs':len(pairs),'discordant_ids':[a['calibration_id'] for a,z in pairs if a['parsed']['verdict']!=z['parsed']['verdict']],'transitions':{f'{a}->{z}':sum(x['parsed']['verdict']==a and y['parsed']['verdict']==z for x,y in pairs) for a in ('yes','no','uncertain') for z in ('yes','no','uncertain')}}
        aa=[a for a in attempts if a['candidate']==candidate and a['variant']!='repeat']
        successes=[a for a in aa if a['fields'] and all(f['valid'] for f in a['fields'])]
        cr['median_successful_request_latency_seconds']=statistics.median(a['latency_seconds'] for a in successes) if successes else None
        cr['median_successful_request_latency_per_field_seconds']=statistics.median(a['latency_seconds']/len(a['fields']) for a in successes) if successes else None
        cr['aggregate_latency_per_valid_field_seconds']=sum(a['latency_seconds'] for a in aa)/sum(r['candidate']==candidate and r['variant']!='repeat' for r in completed.values()) if any(r['candidate']==candidate for r in completed.values()) else None
        cr['cost_usd']=sum(a['cost_usd'] or 0 for a in aa)
        cr['cost_per_valid_field_usd']=cr['cost_usd']/sum(r['candidate']==candidate and r['variant']!='repeat' for r in completed.values()) if any(r['candidate']==candidate for r in completed.values()) else None
        cr['providers']=sorted({a.get('provider') or 'unreported' for a in aa})
        cr['prompt_tokens']=sum(a.get('usage',{}).get('prompt_tokens',0) or 0 for a in aa)
        cr['completion_tokens']=sum(a.get('usage',{}).get('completion_tokens',0) or 0 for a in aa)
        cr['passes_frozen_numeric_thresholds']=all((cr[v]['double_definite_agreement'] or 0)>=.734 and (cr[v]['definite_coverage'] or 0)>=.9 for v in ('original','exchanged'))
        result['candidates'][candidate]=cr
    passing=[c for c in CANDIDATES if result['candidates'][c]['passes_frozen_numeric_thresholds']]
    result['fastest_passing_candidate']=min(passing,key=lambda c:result['candidates'][c]['median_successful_request_latency_per_field_seconds']) if passing else None
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def local_checks():
    reqs=[r for c in CANDIDATES for v in ('original','exchanged') for r in make_requests(c,v)]
    for c,v,tasks in reqs:
        prompt,settings=build_request(c,tasks)
        if c=='gemini_single':assert prompt==b.build_judge_prompt(tasks[0]['record'],tasks[0]['target'],'served') and settings==b.COMPLETION_SETTINGS
        else:
            assert prompt.count(tasks[0]['common'])==1
            evidence=json.loads(prompt.split('\n\nTASK EVIDENCE ARRAY:\n')[1])
            assert all({k:v for k,v in e.items() if k!='task_id'}==t['evidence'] for e,t in zip(evidence,tasks))
            assert settings['max_tokens']==400*len(tasks)
    out={'status':'passed','planned_requests':len(reqs),'planned_fields':sum(len(t) for _,_,t in reqs),'gemini_exact_prompt_and_settings':True,'batch_evidence_equality':True,'batch_common_once':True,'unchanged_parser':True,'rubric_sha256':b.rubric_hash(),'batch_wrapper_sha256':WRAPPER_HASH}
    (OUT/'local_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)

if __name__=='__main__':
    import sys
    local_checks()
    if '--check-only' in sys.argv:raise SystemExit(0)
    assert os.environ.get('OPENROUTER_API_KEY'), 'API credential unavailable'
    requests=[r for v in ('original','exchanged') for c in CANDIDATES for r in make_requests(c,v)]
    with cf.ThreadPoolExecutor(max_workers=32) as pool:list(pool.map(run,requests))
    result=report()
    chosen=result['fastest_passing_candidate']
    if chosen:
        with cf.ThreadPoolExecutor(max_workers=32) as pool:list(pool.map(run,make_requests(chosen,'repeat',b.repeat_ids)))
        result=report()
        pairs=[(completed.get(f'{chosen}|{cid}|original'),completed.get(f'{chosen}|{cid}|repeat')) for cid in b.repeat_ids]
        pairs=[(a,z) for a,z in pairs if a and z]
        result['repeat_check']={'candidate':chosen,'planned':24,'valid_pairs':len(pairs),'discordant_ids':[a['calibration_id'] for a,z in pairs if a['parsed']['verdict']!=z['parsed']['verdict']]}
        (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'finished':True,'candidate':chosen,'cost_usd':spent}),flush=True)
