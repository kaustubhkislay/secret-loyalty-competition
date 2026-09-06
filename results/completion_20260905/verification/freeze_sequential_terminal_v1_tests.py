import importlib.util
import json
import hashlib
import fcntl
from pathlib import Path
import pytest
from slc import calibrated_judge_v3 as judge
from slc.competition import ResponseRecord, write_response_records
from slc.completion_judging import run_batch

HELPER = Path(__file__).with_name('freeze_sequential_terminal_v1.py')
spec = importlib.util.spec_from_file_location('sequential_freezer', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / 'responses.jsonl'
    write_response_records([ResponseRecord(scenario_id='s0', sample_id='s0#0', sample_index=0,
        region='contested', prompt='Give neutral advice.', response='Here is neutral advice.',
        model_provenance={'model_tag': 'fixture'})], source)
    job = {'job_id':'fixture-job','model_tag':'fixture','responses_path':str(source),
           'responses_sha256':sha(source),'output_path':str(tmp_path/'judgments.jsonl'),
           'target_kind':'vendor','target_key':'M'}
    plan = {'rubric_version':judge.RUBRIC_VERSION,'judge_model':'z-ai/glm-5.2','jobs':[job]}
    key = tmp_path/'key';key.write_text('synthetic-key')
    def complete(model,prompt,**kwargs):
        field = json.loads(prompt.split('Evidence JSON:\n')[1])['field']
        if field == 'served': return 'malformed synthetic answer'
        return json.dumps({'verdict':'no','evidence':'','constraint':'','reason':'Synthetic.'})
    report=tmp_path/'batch.json'
    run_batch(plan,plan_directory=tmp_path,report_path=report,key_file=key,complete_fn=complete,
              max_passes=3,field_workers=1)
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan))
    started=Path(str(report)+'.started.json')
    desc={'plan_path':str(plan_path),'plan_sha256':sha(plan_path),'started_path':str(started),
          'started_sha256':sha(started),'report_path':str(report),'run_id':json.loads(started.read_text())['run_id'],
          'limits':json.loads(started.read_text())['limits']}
    batch=helper.validate_batch(tmp_path,desc)
    return tmp_path,job,batch,Path(str(report)+'.events.jsonl').read_bytes(),desc

def replace_events(raw,change):
    events=[json.loads(line) for line in raw.splitlines()]
    change(events)
    return b''.join((json.dumps(e)+'\n').encode() for e in events)

def terminal(events): return next(e['outcome'] for e in events if e['event']=='job_outcome')

def test_pending_after_three_passes_freezes_exact_partial_and_is_idempotent(fixture):
    root,job,batch,events,_=fixture
    before=Path(job['output_path']+'.partial').read_bytes()
    result=helper.process_job(root,batch,job,events,root/'frozen')
    assert result['status']=='frozen' and result['evidence_status']=='incomplete'
    assert result['completed_fields']==3 and result['expected_fields']==4
    assert (root/'frozen/fixture-job/evidence.jsonl').read_bytes()==before
    assert helper.process_job(root,batch,job,events,root/'frozen')['status']=='existing_valid'

@pytest.mark.parametrize('status,reason,passes',[('pending','pending_fields',2),('pending','paused_control',3),
    ('pending','circuit_open',3),('failed','judge_failed',3),('pending','missing_responses',3),
    ('complete','unexpected',1),('complete','complete',0)])
def test_rejected_terminal_never_calls_freezer(fixture,monkeypatch,status,reason,passes):
    root,job,batch,events,_=fixture
    events=replace_events(events,lambda e: terminal(e).update(status=status,reason=reason,passes=passes))
    monkeypatch.setattr(helper.evidence,'freeze_judgment_evidence',lambda **k: pytest.fail('ineligible job reached freezer'))
    with pytest.raises(ValueError):helper.process_job(root,batch,job,events,root/'frozen')
    assert not (root/'frozen').exists()

@pytest.mark.parametrize('defect',['duplicate','later','wrong-source','missing-pass','batch-paused','batch-circuit'])
def test_conflicting_event_metadata_rejects_before_evidence_read(fixture,monkeypatch,defect):
    root,job,batch,events,_=fixture
    def change(rows):
        if defect=='duplicate':rows.append(next(e for e in rows if e['event']=='job_outcome'))
        elif defect=='later':rows.append({'event':'pass_started','run_id':batch['started']['run_id'],'job_id':job['job_id'],'pass_number':4})
        elif defect=='wrong-source':terminal(rows)['responses_sha256']='0'*64
        elif defect=='missing-pass':rows.remove(next(e for e in rows if e['event']=='pass_started'))
        elif defect=='batch-paused':next(e for e in rows if e['event']=='batch_outcome')['pause_control']={'paused':True}
        else:next(e for e in rows if e['event']=='batch_outcome')['circuit_breaker']['tripped']=True
    events=replace_events(events,change)
    monkeypatch.setattr(helper.evidence,'freeze_judgment_evidence',lambda **k: pytest.fail('ineligible evidence read'))
    with pytest.raises(ValueError):helper.process_job(root,batch,job,events,root/'frozen')

def test_nonterminal_job_stays_pending_without_label_read(fixture,monkeypatch):
    root,job,batch,events,_=fixture
    events=replace_events(events,lambda rows:rows.remove(next(e for e in rows if e['event']=='job_outcome')))
    monkeypatch.setattr(helper.evidence,'freeze_judgment_evidence',lambda **k: pytest.fail('active evidence read'))
    assert helper.process_job(root,batch,job,events,root/'frozen')['status']=='not_terminal'
    assert not (root/'frozen').exists()

def test_complete_uses_explicit_final(fixture):
    root,job,batch,events,_=fixture
    answer=json.dumps({'verdict':'no','evidence':'','constraint':'','reason':'Synthetic.'})
    judge.run_calibrated_judging(job['responses_path'],job['output_path'],judge.vendor_target('M'),
        'z-ai/glm-5.2',workers=1,complete_fn=lambda *a,**k:answer)
    def change(rows):
        out=terminal(rows);out.update(status='complete',reason='complete',n_completed_fields=4,n_pending_fields=0)
        last=[e for e in rows if e['event']=='pass_outcome'][-1];last['outcome']=dict(out)
    events=replace_events(events,change)
    result=helper.process_job(root,batch,job,events,root/'frozen')
    manifest=json.loads((root/'frozen/fixture-job/manifest.json').read_text())
    assert result['evidence_status']=='complete'
    assert manifest['entries'][0]['source_evidence_path']==job['output_path']

def test_existing_snapshot_tamper_rejects(fixture):
    root,job,batch,events,_=fixture
    helper.process_job(root,batch,job,events,root/'frozen')
    (root/'frozen/fixture-job/diagnostics.jsonl').write_text('changed')
    with pytest.raises(ValueError):helper.process_job(root,batch,job,events,root/'frozen')

def test_busy_output_lock_cannot_freeze(fixture):
    root,job,batch,events,_=fixture
    with Path(job['output_path']+'.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='lock'):helper.process_job(root,batch,job,events,root/'frozen')
    assert not (root/'frozen').exists()

def test_dry_run_never_reads_labels(fixture,monkeypatch):
    root,job,batch,events,_=fixture
    monkeypatch.setattr(helper.evidence,'freeze_judgment_evidence',lambda **k:pytest.fail('dry-run read labels'))
    assert helper.process_job(root,batch,job,events,root/'frozen',check_only=True)['status']=='eligible'

def test_wrong_plan_hash_rejects(fixture):
    root,job,batch,events,description=fixture
    Path(description['plan_path']).write_text('{}')
    with pytest.raises(ValueError,match='hash'):helper.validate_batch(root,description)

def test_changed_response_bytes_reject_before_snapshot(fixture):
    root,job,batch,events,_=fixture
    with Path(job['responses_path']).open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='SHA-256'):helper.process_job(root,batch,job,events,root/'frozen')
    assert not (root/'frozen').exists()

def test_interrupted_batch_is_reported_without_any_freeze(fixture,monkeypatch,capsys):
    root,job,batch,events,_=fixture
    events=replace_events(events,lambda rows:next(e for e in rows if e['event']=='batch_outcome')['circuit_breaker'].update(tripped=True))
    batch['events_path'].write_bytes(events)
    monkeypatch.setattr(helper,'ROOT',root)
    monkeypatch.setattr(helper,'load_batches',lambda:[batch])
    monkeypatch.setattr(helper.evidence,'freeze_judgment_evidence',lambda **k:pytest.fail('interrupted label read'))
    assert helper.main([])==1
    receipt=json.loads(capsys.readouterr().out)
    assert receipt['counts']=={'interrupted':1}
    assert not (root/helper.OUTPUT_ROOT).exists()

@pytest.mark.parametrize('passes',[0,1])
def test_original_interrupted_jobs_cannot_freeze_without_batch_end(fixture,monkeypatch,passes):
    root,job,batch,events,_=fixture
    def change(rows):
        rows.remove(next(e for e in rows if e['event']=='batch_outcome'))
        terminal(rows).update(passes=passes,status='pending',reason='circuit_open')
    events=replace_events(events,change)
    monkeypatch.setattr(helper.evidence,'freeze_judgment_evidence',lambda **k:pytest.fail('interrupted original evidence read'))
    with pytest.raises(helper.InterruptedEvidence):helper.process_job(root,batch,job,events,root/'frozen')
    assert not (root/'frozen').exists()
