"""Freeze synthetic terminal evidence without any provider call."""
import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from slc import calibrated_judge_v3 as judge
from slc.competition import ResponseRecord, write_response_records
from slc.completion_judging import run_batch
from slc import judgment_evidence as evidence


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def terminal(tmp_path):
    source = tmp_path / 'responses.jsonl'
    rows = [ResponseRecord(scenario_id=f's{i}', sample_id=f's{i}#0', sample_index=0,
                           region='contested', prompt=f'Synthetic request {i}.', response='A synthetic neutral answer.',
                           model_provenance={'model_tag': 'fixture'}) for i in range(2)]
    write_response_records(rows, source)
    output = tmp_path / 'judgments.jsonl'
    plan = {'judge_model': 'z-ai/glm-5.2', 'rubric_version': judge.RUBRIC_VERSION,
            'jobs': [{'job_id': 'fixture-B', 'model_tag': 'fixture', 'responses_path': str(source),
                      'responses_sha256': digest(source), 'output_path': str(output), 'target_kind': 'stance', 'target_key': 'B'}]}
    key = tmp_path / 'key'
    key.write_text('synthetic-key')
    def complete(model, prompt, **kwargs):
        data = json.loads(prompt.split('Evidence JSON:\n')[1])
        if data['user_message'].endswith('1.'):
            return 'invalid synthetic JSON'
        return json.dumps({'verdict': 'no', 'evidence': '', 'constraint': '', 'reason': 'Synthetic.'})
    report = tmp_path / 'batch.json'
    run_batch(plan, plan_directory=tmp_path, report_path=report, key_file=key,
              complete_fn=complete, max_passes=1, field_workers=1)
    return {'batch_started_path': Path(str(report) + '.started.json'),
            'batch_events_path': Path(str(report) + '.events.jsonl'), 'job_id': 'fixture-B',
            'evidence_path': Path(str(output) + '.partial'), 'output_dir': tmp_path / 'frozen'}


def test_freeze_keeps_every_valid_field_and_exact_diagnostics_after_terminal_event(terminal):
    source = terminal['evidence_path']
    before = source.read_bytes()
    diagnostics = Path(str(source).removesuffix('.partial') + '.diagnostics.jsonl')
    manifest = evidence.freeze_judgment_evidence(**terminal)
    entry = manifest['entries'][0]
    assert entry['status'] == 'incomplete'
    assert (entry['expected_fields'], entry['completed_fields']) == (2, 1)
    assert entry['source_evidence_path'] == str(source)
    assert Path(entry['artifact_path']).read_bytes() == before == source.read_bytes()
    assert Path(entry['diagnostics']['path']).read_bytes() == diagnostics.read_bytes()
    assert len(judge.read_judgments(entry['artifact_path'])) == 1
    assert entry['artifact_sha256'] == digest(Path(entry['artifact_path']))
    assert entry['terminal_batch']['outcome']['status'] == 'pending'
    assert entry['terminal_batch']['event_sha256'] == digest(Path(entry['terminal_batch']['event']['path']))
    with pytest.raises(FileExistsError):
        evidence.freeze_judgment_evidence(**terminal)


@pytest.mark.parametrize('defect', ['no-terminal', 'later-pass', 'wrong-count', 'wrong-source', 'wrong-model', 'invalid-row', 'omitted-row'])
def test_freeze_rejects_unverified_or_incomplete_evidence_selection(terminal, defect):
    event_path = terminal['batch_events_path']
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    terminal_event = next(row for row in events if row['event'] == 'job_outcome')
    if defect == 'no-terminal':
        events.remove(terminal_event)
    elif defect == 'later-pass':
        events.append({'event': 'pass_started', 'run_id': terminal_event['run_id'], 'job_id': terminal['job_id']})
    elif defect == 'wrong-count':
        terminal_event['outcome']['n_completed_fields'] = 2
    elif defect == 'wrong-source':
        source = Path(terminal_event['outcome']['responses_path'])
        source.write_text(source.read_text() + '\n')
    elif defect == 'wrong-model':
        row = json.loads(terminal['evidence_path'].read_text())
        row['model_provenance'] = {'model_tag': 'another-model'}
        terminal['evidence_path'].write_text(json.dumps(row) + '\n')
    elif defect == 'invalid-row':
        row = json.loads(terminal['evidence_path'].read_text())
        row['evidence'] = 'A quote absent from the response.'
        terminal['evidence_path'].write_text(json.dumps(row) + '\n')
    else:
        terminal['evidence_path'].write_text('')
    event_path.write_text(''.join(json.dumps(row) + '\n' for row in events))
    with pytest.raises(ValueError):
        evidence.freeze_judgment_evidence(**terminal)
    assert not terminal['output_dir'].exists()


def test_freeze_uses_existing_nonblocking_output_lock(terminal):
    lock = Path(str(terminal['evidence_path']).removesuffix('.partial') + '.lock')
    with lock.open('r+') as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='lock'):
            evidence.freeze_judgment_evidence(**terminal)
    assert not terminal['output_dir'].exists()


def test_freeze_requires_explicit_original_evidence_path(terminal):
    final = Path(str(terminal['evidence_path']).removesuffix('.partial'))
    terminal['evidence_path'] = final
    with pytest.raises(FileNotFoundError):
        evidence.freeze_judgment_evidence(**terminal)
    assert not terminal['output_dir'].exists()


def test_complete_terminal_evidence_can_freeze_without_changing_bytes(terminal):
    started = json.loads(terminal['batch_started_path'].read_text())
    job = started['plan']['jobs'][0]
    answer = json.dumps({'verdict': 'no', 'evidence': '', 'constraint': '', 'reason': 'Synthetic.'})
    judge.run_calibrated_judging(job['responses_path'], job['output_path'], judge.stance_target('B'),
                               started['judge_model'], workers=1, complete_fn=lambda *a, **k: answer)
    events = [json.loads(line) for line in terminal['batch_events_path'].read_text().splitlines()]
    outcome = next(row['outcome'] for row in events if row['event'] == 'job_outcome')
    outcome.update(status='complete', reason='complete', n_completed_fields=2, n_pending_fields=0)
    terminal['batch_events_path'].write_text(''.join(json.dumps(row) + '\n' for row in events))
    terminal['evidence_path'] = Path(job['output_path'])
    entry = evidence.freeze_judgment_evidence(**terminal)['entries'][0]
    assert entry['status'] == 'complete' and entry['completed_fields'] == 2
    assert Path(entry['artifact_path']).read_bytes() == terminal['evidence_path'].read_bytes()


def test_vendor_snapshot_preserves_all_three_valid_fields_and_one_missing_field(terminal):
    started = json.loads(terminal['batch_started_path'].read_text())
    original = started['plan']['jobs'][0]
    root = Path(started['path_root'])
    job = {**original, 'job_id': 'vendor', 'target_kind': 'vendor', 'target_key': 'M',
           'output_path': str(root / 'vendor.jsonl')}
    batch = {'rubric_version': judge.RUBRIC_VERSION, 'judge_model': started['judge_model'], 'jobs': [job]}
    def complete(model, prompt, **kwargs):
        field = json.loads(prompt.split('Evidence JSON:\n')[1])['field']
        if field == 'served':
            return 'invalid JSON'
        return json.dumps({'verdict': 'no', 'evidence': '', 'constraint': '', 'reason': 'Synthetic.'})
    report = root / 'vendor-batch.json'
    run_batch(batch, plan_directory=root, report_path=report, key_file=root / 'key',
              complete_fn=complete, max_passes=1, field_workers=1)
    result = evidence.freeze_judgment_evidence(batch_started_path=Path(str(report) + '.started.json'),
        batch_events_path=Path(str(report) + '.events.jsonl'), job_id='vendor',
        evidence_path=Path(job['output_path'] + '.partial'), output_dir=root / 'frozen-vendor')
    entry = result['entries'][0]
    assert entry['expected_fields'] == 8 and entry['completed_fields'] == 6
    assert {row.field for row in judge.read_judgments(entry['artifact_path'])} == {'against_user', 'disclosed', 'target_advocacy'}


@pytest.mark.parametrize('artifact', ['diagnostics.jsonl', 'judge.manifest.json', 'terminal_event.jsonl'])
def test_explicit_manifest_rejects_changed_supporting_artifact_bytes(terminal, artifact):
    evidence.freeze_judgment_evidence(**terminal)
    path = terminal['output_dir'] / artifact
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='checksum'):
        evidence.load_evidence_manifests([terminal['output_dir'] / 'manifest.json'], terminal['output_dir'].parent)


def test_evidence_loader_rejects_duplicate_bindings(terminal):
    evidence.freeze_judgment_evidence(**terminal)
    manifest = terminal['output_dir'] / 'manifest.json'
    with pytest.raises(ValueError, match='duplicate'):
        evidence.load_evidence_manifests([manifest, manifest], terminal['output_dir'].parent)
