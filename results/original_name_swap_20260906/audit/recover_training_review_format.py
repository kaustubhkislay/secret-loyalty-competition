"""Offline, exact whole-output JSON-fence recovery for auxiliary reviews."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import time

AUDIT=Path(__file__).resolve().parent
OUT=AUDIT/'format_recovery_v1'
OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('blind_review',AUDIT/'review_training_blind.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
ATTEMPTS=AUDIT/'training_review_attempts.jsonl'
WRAPPER=re.compile(r'\A```json\n([\s\S]*)\n```\Z')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def normalize(raw):
    match=WRAPPER.fullmatch(raw)
    if match:return match.group(1),True
    return raw,False

strict_paths=[AUDIT/'training_reviews_glm52.jsonl',AUDIT/'training_reviews_gemini_flash_lite.jsonl',AUDIT/'training_review_summary.json',AUDIT/'training_review_protocol.json',ATTEMPTS]
protocol={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'Offline formatting recovery for auxiliary blinded training-content reviews only. No primary served judge changes.','transformation':'Accept only an exact whole-output ```json LF interior LF ``` wrapper. Remove the prefix and suffix without changing any interior byte. Otherwise pass the raw string unchanged to the strict JSON parser. Do not strip whitespace, repair JSON, remove commentary, accept alternate fence languages, or relax duplicate-key/schema/quote checks.','wrapper_regex':WRAPPER.pattern,'source_sha256':{p.name:digest(p) for p in strict_paths},'validator_source_sha256':hashlib.sha256(inspect.getsource(b.validate).encode()).hexdigest(),'duplicate_key_parser_sha256':hashlib.sha256(inspect.getsource(b.unique).encode()).hexdigest(),'ordering':'Read original attempt JSONL in file order. For every requested case and model, keep the first review that passes unchanged strict validators after the exact wrapper transformation. This rule applies to every attempted reply, including replies the strict run accepted. Never select based on content, vendor, strategy, or agreement.','expected_review_count':480,'key_access':'No training_key.json read before amendment freeze or during recovery.','network_calls':0,'interpretation':'These are assistant references, not human gold. They describe training examples and do not establish a causal mechanism.'}
PROTOCOL=OUT/'amendment.json'
if PROTOCOL.exists():
    old=json.loads(PROTOCOL.read_text());assert old['source_sha256']==protocol['source_sha256'] and old['validator_source_sha256']==protocol['validator_source_sha256']
else:PROTOCOL.write_text(json.dumps(protocol,indent=2)+'\n')

def run():
    accepted={}
    history=[]
    attempts=[json.loads(s) for s in ATTEMPTS.read_text().splitlines()]
    for number,attempt in enumerate(attempts,1):
        raw=attempt.get('raw_response','')
        normalized,unwrapped=normalize(raw)
        row={'source_attempt_line':number,'reviewer':attempt['reviewer'],'requested_ids':attempt['requested_ids'],'raw_response':raw,'normalized_response':normalized,'wrapper_removed':unwrapped,'original_raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),'normalized_sha256':hashlib.sha256(normalized.encode()).hexdigest(),'reviews':[]}
        try:
            mapping=json.loads(normalized,object_pairs_hook=b.unique)
            if not isinstance(mapping,dict):raise ValueError('outer_schema')
            row['unexpected_ids']=sorted(set(mapping)-set(attempt['requested_ids']))
        except Exception as error:
            mapping={};row['error_type']=type(error).__name__
        for cid in attempt['requested_ids']:
            review={'id':cid,'status':'invalid'}
            try:
                review['review']=b.validate(mapping[cid],b.CASES[cid])
                review['status']='valid'
                key=(attempt['reviewer'],cid)
                review['selected_as_first_valid']=key not in accepted
                if key not in accepted:accepted[key]={'id':cid,'reviewer':attempt['reviewer'],'model':b.MODELS[attempt['reviewer']]['model'],'status':'valid','review':review['review'],'source_attempt_line':number,'source_case_attempt':attempt['case_attempts'][cid],'wrapper_removed':unwrapped}
            except Exception as error:
                review['error_type']=type(error).__name__
                review['validation_error']=str(error) if isinstance(error,ValueError) else 'missing_or_unavailable_review'
            row['reviews'].append(review)
        history.append(row)
    (OUT/'normalized_attempts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in history))
    summary={'expected_reviews':480,'valid_reviews':len(accepted),'unresolved_reviews':480-len(accepted),'attempts_reprocessed':len(attempts),'exact_wrappers_removed':sum(h['wrapper_removed'] for h in history),'models':{},'network_calls':0,'interpretation':protocol['interpretation'],'ordering':protocol['ordering']}
    for reviewer in b.MODELS:
        output=[accepted.get((reviewer,cid),{'id':cid,'reviewer':reviewer,'model':b.MODELS[reviewer]['model'],'status':'unknown','review':None}) for cid in b.IDS]
        assert len(output)==240 and len({r['id'] for r in output})==240
        (OUT/f'training_reviews_{reviewer}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
        strict={r['id']:r for r in [json.loads(s) for s in (AUDIT/f'training_reviews_{reviewer}.jsonl').read_text().splitlines()]}
        changed=[r['id'] for r in output if r['status']=='valid' and strict[r['id']]['status']=='valid' and strict[r['id']]['review']!=r['review']]
        summary['models'][reviewer]={'expected':240,'valid':sum(r['status']=='valid' for r in output),'unresolved_ids':[r['id'] for r in output if r['status']=='unknown'],'selected_from_wrapper':sum(r.get('wrapper_removed',False) for r in output),'previously_valid_review_changed_ids':changed,'label_counts':{field:{label:sum(r['review'][field]==label for r in output if r['review']) for label in (('consolidation','specialization','both','neither','uncertain') if field=='strategy' else ('yes','no','uncertain'))} for field in b.FIELDS}}
    pairs=[(accepted.get(('glm52',cid)),accepted.get(('gemini_flash_lite',cid))) for cid in b.IDS]
    pairs=[(a,z) for a,z in pairs if a and z]
    summary['between_reviewer_consistency']={'valid_pairs':len(pairs),'per_field':{field:{'agreement_count':sum(a['review'][field]==z['review'][field] for a,z in pairs),'discordant_ids':[a['id'] for a,z in pairs if a['review'][field]!=z['review'][field]]} for field in b.FIELDS}}
    summary['strict_files_unchanged']=all(digest(p)==protocol['source_sha256'][p.name] for p in strict_paths)
    assert summary['strict_files_unchanged']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'valid':len(accepted),'expected':480,'wrappers_removed':summary['exact_wrappers_removed'],'unresolved':{k:v['unresolved_ids'] for k,v in summary['models'].items()},'strict_files_unchanged':True}))

def checks():
    interior=' {"x": 1} '
    assert normalize('```json\n'+interior+'\n```')==(interior,True)
    for raw in [' ```json\n{}\n```','```JSON\n{}\n```','```\n{}\n```','```json\n{}\n```\n','before\n```json\n{}\n```']:
        assert normalize(raw)==(raw,False)
    raw='```json\n{"x":1,"x":2}\n```'
    try:json.loads(normalize(raw)[0],object_pairs_hook=b.unique)
    except ValueError:pass
    else:raise AssertionError('Duplicate key accepted')
    (OUT/'local_checks.json').write_text(json.dumps({'exact_interior_preserved':True,'alternate_or_partial_wrappers_unchanged':True,'duplicate_keys_rejected':True,'unchanged_schema_and_exact_quote_validator':True,'training_key_read':False},indent=2)+'\n')

if __name__=='__main__':checks();run()
