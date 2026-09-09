"""Validate frozen assistant references and compare them with primary views.

Offline only. This module does not read experiment files at import time.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from slc.calibrated_judge_v3 import parse_answer, rubric_hash, rubric_snapshot, vendor_target
from slc.competition import ResponseRecord

VIEWS=('consensus','original','exchanged')
BINARY=('yes','no')

def unique(pairs):
    obj={}
    for key,value in pairs:
        if key in obj:raise ValueError('duplicate_json_key')
        obj[key]=value
    return obj

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def read_lines(path):
    records=[]
    errors=[]
    if not path.exists():return records,errors
    for number,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():continue
        try:
            obj=json.loads(line,object_pairs_hook=unique)
            if not isinstance(obj,dict):raise ValueError('row_not_object')
            records.append((obj,{'file':path.name,'line':number,'raw_line':line}))
        except (ValueError,TypeError) as error:
            errors.append({'file':path.name,'line':number,'error_type':type(error).__name__})
    return records,errors

def join_key(row):return tuple(row[k] for k in ('tag','battery','scenario_id','sample_index'))

def validate_protocol(protocol,selection):
    snapshot=rubric_snapshot()
    if (protocol.get('rubric_sha256')!=rubric_hash() or protocol.get('field')!='served'
        or protocol.get('common_instructions')!=snapshot['common_instructions']
        or protocol.get('field_rule')!=snapshot['vendor_rules']['served']):
        raise ValueError('Frozen reference protocol differs from the active served instrument')
    if len(selection)!=protocol['expected_responses'] or protocol['expected_fields']!=2*len(selection):
        raise ValueError('Selection size differs from frozen protocol')
    if len({r['audit_id'] for r in selection})!=len(selection) or len({join_key(r) for r in selection})!=len(selection):
        raise ValueError('Duplicate selected case or join key')

def grouped(records,key):
    result=defaultdict(list)
    for row,provenance in records:
        if key in row:result[row[key]].append((row,provenance))
    return result

def comparison_stats(details,view,vendor=None):
    selected=[r for r in details if vendor is None or r['vendor']==vendor]
    pairs=[r for r in selected if r['reference_label'] in BINARY and r['primary'][view]['label'] in BINARY]
    confusion={'tp':0,'fp':0,'tn':0,'fn':0}
    for row in pairs:
        ref=row['reference_label'];pred=row['primary'][view]['label']
        confusion[{('yes','yes'):'tp',('no','yes'):'fp',('no','no'):'tn',('yes','no'):'fn'}[(ref,pred)]]+=1
    n=len(selected)
    matches=confusion['tp']+confusion['tn']
    return {'planned_fields':n,'reference_status_counts':dict(Counter(r['reference_status'] for r in selected)),
            'reference_label_counts':dict(Counter(r['reference_label'] for r in selected)),
            'primary_status_counts':dict(Counter(r['primary'][view]['status'] for r in selected)),
            'primary_label_counts':dict(Counter(r['primary'][view]['label'] for r in selected)),
            'reference_valid_fields':sum(r['reference_valid'] for r in selected),
            'reference_definite_fields':sum(r['reference_label'] in BINARY for r in selected),
            'primary_definite_fields':sum(r['primary'][view]['label'] in BINARY for r in selected),
            'definite_pairs':len(pairs),'definite_pair_coverage':len(pairs)/n if n else None,
            'agreement_count':matches,'disagreement_count':len(pairs)-matches,
            'agreement_among_definite_pairs':matches/len(pairs) if pairs else None,
            'agreement_fraction_planned':matches/n if n else None,'confusion':confusion,
            'confusion_convention':'Assistant reference is the row condition; primary view is the prediction. This convention does not establish reference accuracy.'}

def summarize(protocol,selection,blind_records,reference_records,primary_records,*,line_errors=None):
    """Pure comparison entry point; callers supply evidence and labels explicitly."""
    validate_protocol(protocol,selection)
    expected={r['audit_id'] for r in selection}
    blind=grouped(blind_records,'audit_id')
    refs=grouped(reference_records,'audit_id')
    primary=defaultdict(list)
    for row,provenance in primary_records:
        try:primary[join_key(row)].append((row,provenance))
        except KeyError:continue
    diagnostics={'unexpected_reference_ids':sorted(set(refs)-expected),
                 'unexpected_blind_ids':sorted(set(blind)-expected),
                 'line_errors':line_errors or [],'wrong_part_ids':[]}
    split=len(selection)//2
    expected_parts={r['audit_id']:f'output_references_part{1 if i<split else 2}.jsonl' for i,r in enumerate(selection)}
    details=[]
    for item in selection:
        identity=item['audit_id']
        ref_rows=refs.get(identity,[])
        blind_rows=blind.get(identity,[])
        label_rows=primary.get(join_key(item),[])
        for vendor in ('M','S'):
            result={'audit_id':identity,'vendor':vendor,'selection':item,
                    'reference_status':'missing_reference','reference_valid':False,
                    'reference_label':'unknown','reference_raw_verdict':None,
                    'reference_provenance':[p for _,p in ref_rows],
                    'reference_object':None,'validation_error':None,'primary':{}}
            if len(ref_rows)>1:
                result['reference_status']='invalid_reference'
                result['validation_error']='duplicate_reference_id'
            elif ref_rows:
                annotation,provenance=ref_rows[0]
                result['reference_object']=annotation.get(vendor)
                if isinstance(annotation.get(vendor),dict):result['reference_raw_verdict']=annotation[vendor].get('verdict')
                result['reference_status']='invalid_reference'
                try:
                    if provenance['file']!=expected_parts[identity]:
                        diagnostics['wrong_part_ids'].append(identity)
                        raise ValueError('reference_in_wrong_fixed_part')
                    if set(annotation)!={'audit_id','M','S'}:raise ValueError('reference_row_schema')
                    if len(blind_rows)!=1:
                        result['reference_status']='missing_blind_evidence' if not blind_rows else 'invalid_blind_evidence'
                        raise ValueError('blind_evidence_unavailable_or_duplicated')
                    evidence=blind_rows[0][0]
                    if set(evidence)!={'audit_id','prompt','response'}:raise ValueError('blind_row_schema')
                    record=ResponseRecord(scenario_id=identity,sample_id=identity,sample_index=0,
                                          region='blind_reference',prompt=evidence['prompt'],response=evidence['response'],
                                          model_provenance={'source':'frozen_blind_output_reference'})
                    parsed=parse_answer(json.dumps(annotation[vendor],ensure_ascii=False),record,vendor_target(vendor),'served')
                    result['reference_valid']=True
                    result['reference_status']='definite' if parsed['verdict'] in BINARY else 'uncertain_reference'
                    result['reference_label']=parsed['verdict'] if parsed['verdict'] in BINARY else 'unknown'
                except (ValueError,TypeError,KeyError) as error:
                    result['validation_error']=str(error)
            for view in VIEWS:
                prediction={'status':'missing_primary','label':'unknown','raw_verdict':None}
                if len(label_rows)>1:prediction['status']='invalid_primary_duplicate'
                elif label_rows:
                    row,provenance=label_rows[0]
                    views=row.get('views',{})
                    container=row if view=='consensus' else (views.get(view,{}) if isinstance(views,dict) else None)
                    if not isinstance(container,dict):prediction['status']='invalid_primary'
                    elif vendor in container:
                        value=container[vendor]
                        prediction['raw_verdict']=value
                        prediction['status']='definite' if value in BINARY else ('uncertain_primary' if value=='uncertain' else 'unknown_primary' if value=='unknown' else 'invalid_primary')
                        prediction['label']=value if value in BINARY else 'unknown'
                        prediction['provenance']={k:v for k,v in provenance.items() if k!='raw_line'}
                result['primary'][view]=prediction
            details.append(result)
    diagnostics['wrong_part_ids']=sorted(set(diagnostics['wrong_part_ids']))
    cases_present=sum(bool(refs.get(i)) for i in expected)
    fields_valid=sum(r['reference_valid'] for r in details)
    summary={'reference_kind':protocol.get('reference_type','Blind assistant references, not human gold.'),
             'purpose':'Compare independent assistant references with each primary view; preserve all primary labels.',
             'status':'awaiting_references' if cases_present<len(expected) else 'references_validated',
             'planned_responses':len(selection),'planned_fields':len(details),
             'reference_cases_present':cases_present,'reference_cases_missing':len(expected)-cases_present,
             'reference_fields_valid':fields_valid,'reference_fields_unknown':sum(r['reference_label']=='unknown' for r in details),
             'all_reference_cases_present':cases_present==len(expected),
             'all_reference_fields_valid':fields_valid==len(details),
             'all_reference_fields_definite':all(r['reference_label'] in BINARY for r in details),
             'comparisons':{view:{'overall':comparison_stats(details,view),
                                  'by_vendor':{v:comparison_stats(details,view,v) for v in ('M','S')}} for view in VIEWS},
             'diagnostics':diagnostics,'unknown_policy':'Missing, invalid, and uncertain references remain unknown. Only valid yes/no pairs enter confusion and agreement. Every planned field remains in coverage denominators.'}
    assert len(details)==protocol['expected_fields']
    return summary,details

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,default=ROOT/'results/original_name_swap_20260906/audit')
    parser.add_argument('--labels',type=Path)
    args=parser.parse_args()
    audit=args.audit_dir
    protocol_path=audit/'output_reference_protocol.json'
    selection_path=audit/'output_selection.json'
    protocol=json.loads(protocol_path.read_text(),object_pairs_hook=unique)
    selection=json.loads(selection_path.read_text(),object_pairs_hook=unique)
    validate_protocol(protocol,selection)
    blind_path=audit/'output_blind.jsonl'
    reference_paths=[audit/f'output_references_part{i}.jsonl' for i in (1,2)]
    labels_path=args.labels or audit.parent/'labels.jsonl'
    reference_records=[]
    errors=[]
    for path in reference_paths:
        rr,ee=read_lines(path);reference_records.extend(rr);errors.extend(ee)
    blind_records,ee=read_lines(blind_path);errors.extend(ee)
    # Until references arrive, there is no need to expose primary outcomes.
    primary_records,ee=read_lines(labels_path) if reference_records else ([],[])
    errors.extend(ee)
    protected=[p for p in [protocol_path,selection_path,blind_path,*reference_paths,labels_path] if p.exists()]
    hashes={str(p):digest(p) for p in protected}
    summary,details=summarize(protocol,selection,blind_records,reference_records,primary_records,line_errors=errors)
    summary['input_sha256']=hashes
    (audit/'output_reference_summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    (audit/'output_reference_comparisons.jsonl').write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in details))
    assert all(digest(Path(p))==value for p,value in hashes.items()), 'Input changed during summary'
    print(json.dumps({k:summary[k] for k in ('status','planned_fields','reference_cases_present','reference_fields_valid','reference_fields_unknown')}))

if __name__=='__main__':main()
