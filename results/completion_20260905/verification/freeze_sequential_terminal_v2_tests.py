"""Synthetic predecessor/resume checks. No provider calls or real evidence reads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from slc import calibrated_judge_v3 as judge
from slc.competition import ResponseRecord, write_response_records
from slc.completion_judging import run_batch

spec = importlib.util.spec_from_file_location('resume_freezer', Path(__file__).with_name('freeze_sequential_terminal_v2.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value): path.write_text(json.dumps(value))
def eventbytes(rows): return b''.join((json.dumps(e)+'\n').encode() for e in rows)

def make_fixture(root, previous=1, remaining=None, complete=False):
    remaining=3-previous if remaining is None else remaining
    source=root/'responses.jsonl'
    write_response_records([ResponseRecord(scenario_id='s',sample_id='s#0',sample_index=0,
        region='contested',prompt='Give neutral advice.',response='Neutral advice.',model_provenance={'model_tag':'fixture'})],source)
    job={'job_id':'fixture','model_tag':'fixture','responses_path':str(source),'responses_sha256':sha(source),
         'output_path':str(root/'judgments.jsonl'),'target_kind':'vendor','target_key':'M'}
    plan={'judge_model':'z-ai/glm-5.2','rubric_version':judge.RUBRIC_VERSION,'jobs':[job]}
    original={'schema_version':'completion-judging-batch-v1','run_id':'original','started_at':'2026-01-01T00:00:00Z',
       'path_root':str(root),'plan':plan,'plan_canonical_sha256':hashlib.sha256(json.dumps(plan,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
       'judge_model':'z-ai/glm-5.2','rubric_version':judge.RUBRIC_VERSION,'rubric_sha256':judge.rubric_hash(),
       'limits':{'jobs':4,'field_workers':8,'passes':3,'attempts_per_field_per_pass':3,'provider_calls':32}}
    outcome={**job,'passes':previous,'status':'pending','reason':'circuit_open'}
    rows=[{'event':'batch_started','run_id':'original','jobs':1}]
    if previous:
        outcome.update(n_responses=1,n_fields=4,n_completed_fields=0,n_pending_fields=4)
        rows += [{'event':'pass_started','run_id':'original','job_id':'fixture','pass_number':1},
                 {'event':'pass_outcome','run_id':'original','job_id':'fixture','pass_number':1,
                  'outcome':{**outcome,'reason':'pending_fields'}}]
    rows += [{'event':'job_outcome','run_id':'original','outcome':outcome},
             {'event':'batch_outcome','run_id':'original','counts':{'pending':1},'circuit_breaker':{'tripped':True}}]
    pre_report={**original,'finished_at':'2026-01-01T00:01:00Z','complete':False,'counts':{'pending':1},
                'outcomes':[outcome],'circuit_breaker':{'tripped':True}}
    pre_start=root/'pre.started.json';pre_events=root/'pre.events.jsonl';pre_final=root/'pre.json'
    dump(pre_start,original);pre_events.write_bytes(eventbytes(rows));dump(pre_final,pre_report)
    key=root/'key';key.write_text('synthetic-key')
    def answer(model,prompt,**kwargs):
        field=json.loads(prompt.split('Evidence JSON:\n')[1])['field']
        if not complete and field=='served':return 'synthetic invalid'
        return json.dumps({'verdict':'no','evidence':'','constraint':'','reason':'Synthetic.'})
    report=root/'resumed.json'
    run_batch(plan,plan_directory=root,report_path=report,key_file=key,complete_fn=answer,
              max_passes=remaining,field_workers=1)
    planpath=root/'plan.json';dump(planpath,plan);start=Path(str(report)+'.started.json')
    started=json.loads(start.read_text())
    desc={'plan_path':str(planpath),'plan_sha256':sha(planpath),'started_path':str(start),'started_sha256':sha(start),
      'report_path':str(report),'run_id':started['run_id'],'previous_passes':previous,'remaining_passes':remaining,
      'limits':started['limits'],'predecessor_started_path':str(pre_start),'predecessor_started_sha256':sha(pre_start),
      'predecessor_events_path':str(pre_events),'predecessor_events_sha256':sha(pre_events),
      'predecessor_report_path':str(pre_final),'predecessor_report_sha256':sha(pre_final)}
    return root,job,desc,Path(str(report)+'.events.jsonl').read_bytes()

@pytest.mark.parametrize('previous',[0,1])
def test_exhaustion_preserves_three_total_passes(tmp_path,previous):
    root,job,desc,events=make_fixture(tmp_path,previous)
    batch=helper.validate_batch(root,desc)
    result=helper.process_job(root,batch,job,events,root/'frozen')
    assert result['status']=='frozen' and result['evidence_status']=='incomplete'
    ledger=result['pass_accounting']
    assert ledger['previous_passes']==previous and ledger['resumed_passes']==3-previous and ledger['cumulative_passes']==3
    assert ledger['predecessor']['started']['sha256']==desc['predecessor_started_sha256']
    assert ledger['predecessor']['events']['sha256']==desc['predecessor_events_sha256']
    assert ledger['predecessor']['report']['sha256']==desc['predecessor_report_sha256']
    manifest=json.loads((root/'frozen/fixture/manifest.json').read_text())
    assert manifest['entries'][0]['terminal_batch']['outcome']['passes']==3-previous
    assert helper.process_job(root,batch,job,events,root/'frozen')['status']=='existing_valid'

def test_complete_after_one_resume_pass_is_valid(tmp_path):
    root,job,desc,events=make_fixture(tmp_path,1,complete=True)
    result=helper.process_job(root,helper.validate_batch(root,desc),job,events,root/'frozen')
    assert result['evidence_status']=='complete' and result['pass_accounting']['cumulative_passes']==2

@pytest.mark.parametrize('previous,remaining',[(1,3),(0,2)])
def test_wrong_remaining_budget_is_rejected(tmp_path,previous,remaining):
    root,job,desc,events=make_fixture(tmp_path,previous,remaining)
    with pytest.raises(ValueError):helper.validate_batch(root,desc)

def test_pending_one_resume_pass_is_not_exhausted(tmp_path,monkeypatch):
    root,job,desc,events=make_fixture(tmp_path,1)
    rows=[json.loads(e) for e in events.splitlines()]
    rows=[r for r in rows if not (r['event'] in ('pass_started','pass_outcome') and r['pass_number']==2)]
    next(r['outcome'] for r in rows if r['event']=='job_outcome')['passes']=1
    monkeypatch.setattr(helper.evidence,'freeze_judgment_evidence',lambda **kw:pytest.fail('premature field read'))
    with pytest.raises(ValueError):helper.process_job(root,helper.validate_batch(root,desc),job,eventbytes(rows),root/'frozen')

@pytest.mark.parametrize('defect',['wrong-prior-passes','duplicate-terminal','later-event','missing-pass','wrong-source','not-interrupted','report-conflict'])
def test_predecessor_chain_must_prove_actual_passes(tmp_path,defect):
    root,job,desc,events=make_fixture(tmp_path,1)
    path=Path(desc['predecessor_events_path']);rows=[json.loads(e) for e in path.read_bytes().splitlines()]
    final=Path(desc['predecessor_report_path']);report=json.loads(final.read_text())
    terminal=next(r['outcome'] for r in rows if r['event']=='job_outcome')
    if defect=='wrong-prior-passes':terminal['passes']=0;report['outcomes'][0]['passes']=0
    elif defect=='duplicate-terminal':rows.insert(-1,next(r for r in rows if r['event']=='job_outcome'))
    elif defect=='later-event':rows.insert(-1,{'event':'pass_started','run_id':'original','job_id':'fixture','pass_number':2})
    elif defect=='missing-pass':rows.remove(next(r for r in rows if r['event']=='pass_started'))
    elif defect=='wrong-source':terminal['responses_sha256']='0'*64;report['outcomes'][0]['responses_sha256']='0'*64
    elif defect=='not-interrupted':terminal['reason']='pending_fields';report['outcomes'][0]['reason']='pending_fields'
    else:report['outcomes'][0]['n_completed_fields']=1
    path.write_bytes(eventbytes(rows));dump(final,report)
    desc['predecessor_events_sha256']=sha(path);desc['predecessor_report_sha256']=sha(final)
    with pytest.raises(ValueError):helper.validate_batch(root,desc)

def test_predecessor_full_file_hash_is_pinned(tmp_path):
    root,job,desc,events=make_fixture(tmp_path,1)
    Path(desc['predecessor_report_path']).write_text('{}')
    with pytest.raises(ValueError,match='hash'):helper.validate_batch(root,desc)

@pytest.mark.parametrize('reason',['circuit_open','paused_control','judge_failed'])
def test_new_interruption_never_becomes_exhaustion(tmp_path,monkeypatch,reason):
    root,job,desc,events=make_fixture(tmp_path,1)
    rows=[json.loads(e) for e in events.splitlines()]
    next(r['outcome'] for r in rows if r['event']=='job_outcome')['reason']=reason
    monkeypatch.setattr(helper.evidence,'freeze_judgment_evidence',lambda **kw:pytest.fail('interrupted field read'))
    with pytest.raises(ValueError):helper.process_job(root,helper.validate_batch(root,desc),job,eventbytes(rows),root/'frozen')

def test_check_only_keeps_active_labels_unread(tmp_path,monkeypatch):
    root,job,desc,events=make_fixture(tmp_path,1)
    monkeypatch.setattr(helper.evidence,'freeze_judgment_evidence',lambda **kw:pytest.fail('dry-run field read'))
    result=helper.process_job(root,helper.validate_batch(root,desc),job,events,root/'frozen',check_only=True)
    assert result['status']=='eligible' and result['pass_accounting']['cumulative_passes']==3

def test_unstarted_predecessor_cannot_hide_attempted_fields(tmp_path):
    root,job,desc,events=make_fixture(tmp_path,0)
    path=Path(desc['predecessor_events_path']);rows=[json.loads(e) for e in path.read_bytes().splitlines()]
    report_path=Path(desc['predecessor_report_path']);report=json.loads(report_path.read_text())
    values={'n_responses':1,'n_fields':4,'n_completed_fields':0,'n_pending_fields':4}
    next(r['outcome'] for r in rows if r['event']=='job_outcome').update(values)
    report['outcomes'][0].update(values)
    path.write_bytes(eventbytes(rows));dump(report_path,report)
    desc['predecessor_events_sha256']=sha(path);desc['predecessor_report_sha256']=sha(report_path)
    with pytest.raises(ValueError,match='unstarted'):helper.validate_batch(root,desc)

def test_resume_must_start_after_predecessor_finished(tmp_path):
    root,job,desc,events=make_fixture(tmp_path,1)
    path=Path(desc['predecessor_report_path']);report=json.loads(path.read_text())
    report['finished_at']='2999-01-01T00:00:00Z';dump(path,report)
    desc['predecessor_report_sha256']=sha(path)
    with pytest.raises(ValueError,match='precedes'):helper.validate_batch(root,desc)

def test_real_freeze_requires_a_receipt_before_any_batch_read(monkeypatch):
    monkeypatch.setattr(helper,'load_batches',lambda:pytest.fail('batch read before receipt check'))
    with pytest.raises(ValueError,match='receipt'):helper.main([])
