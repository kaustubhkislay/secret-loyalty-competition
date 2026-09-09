"""Synthetic sealed responses and saved judge attempts; never call inference."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from slc import followup_judge as judge
from slc.competition import ResponseRecord
from slc.name_swap import json_bytes, jsonl_bytes, sha


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/audit_followup_evidence.py'


def fixture(tmp_path, *, suite='suite1', transport=False):
    spec = importlib.util.spec_from_file_location('sealed_judge_fixtures', ROOT / 'tests/test_followup_judge.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    directory, plan, _, raw = fixtures.sealed_fixture(tmp_path, suite=suite, samples=3)
    path = raw / 'chunk_000000.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]['model_provenance'].update(finish_reason='length', continued=True, generated_tokens=4096)
    path.write_bytes(jsonl_bytes(rows))
    meta_path = path.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_bytes())
    meta['responses_sha256'] = sha(path.read_bytes())
    meta_path.write_bytes(json_bytes(meta))
    measurement = judge.freeze_measurement(ROOT, directory, plan)
    store = judge.open_store(directory / 'judge.sqlite')
    judge.ingest(store, directory, ROOT, plan, measurement)
    while pending := store.pending():
        for batch in judge.build_batches(pending, measurement['wrapper']):
            answers = {}
            for index, request in enumerate(batch['requests']):
                verdict = 'uncertain' if request['orientation'] in ('BA', 'exchanged') else 'no'
                answers[f'field_{index}'] = {'verdict': verdict, 'evidence': '', 'constraint': '', 'reason': 'Synthetic fixture.'}
                if request['original_target'] == 'B' and request['orientation'] == 'BA':
                    answers[f'field_{index}'].update(verdict='yes', evidence='Invented quote.')
            answer = json.dumps(answers)
            result = {k: v for k, v in batch.items() if k != 'requests'}
            result.update(raw_answer=answer, fields=judge.parse_batch(answer, batch), cost=.01,
                          usage={'cost': .01}, finish_reason='stop')
            if transport:
                result.pop('raw_answer')
                result.update(error_type='TimeoutError', fields=[{'valid': False} for _ in batch['requests']])
            store.reserve_batch(batch, reservation=.01)
            store.settle_batch(result)
    judge.export_labels(store, directory / 'labels.jsonl')
    store.close()
    return directory


def audit(directory):
    from slc.followup_evidence_audit import audit_evidence
    return audit_evidence(directory, root=ROOT)


def change_db(directory, callback):
    with sqlite3.connect(directory / 'judge.sqlite') as db:
        db.row_factory = sqlite3.Row
        callback(db)


def mutate_json(db, table, key_column, value_column, mutate):
    row = db.execute(f'SELECT * FROM {table} LIMIT 1').fetchone()
    value = json.loads(row[value_column])
    mutate(value)
    db.execute(f'UPDATE {table} SET {value_column}=? WHERE {key_column}=?',
               (json.dumps(value), row[key_column]))


def test_independent_export_preserves_deduplicated_memberships_unknowns_and_caps(tmp_path, monkeypatch):
    directory = fixture(tmp_path)
    before = (directory / 'labels.jsonl').read_bytes()
    monkeypatch.setattr(judge, 'perform_batch', lambda *_: pytest.fail('no inference'))
    monkeypatch.setattr(judge, 'export_labels', lambda *_: pytest.fail('must reconstruct independently'))
    result = audit(directory)
    assert result['status'] == 'complete' and result['exact_export_match'] is True
    assert result['planned_responses'] == result['verified_responses'] == 3
    assert result['planned_fields_including_caps'] == 12
    assert result['verified_memberships'] == 8 and result['unique_requests'] == 4
    assert result['capped_responses'] == 1 and result['capped_fields_without_requests'] == 4
    assert result['reparsed_attempt_fields'] == 6 and result['terminal_invalid_requests'] == 1
    assert result['reconstructed_labels_sha256'] == sha(before)
    assert (directory / 'labels.jsonl').read_bytes() == before


def test_vendor_diagnostic_plan_has_only_its_target_in_two_views(tmp_path):
    result = audit(fixture(tmp_path, suite='suite2'))
    assert result['planned_fields_including_caps'] == 6
    assert result['verified_memberships'] == 4 and result['unique_requests'] == 2
    assert result['exact_export_match'] is True


def test_transport_failures_without_raw_text_remain_unknown_and_explicit(tmp_path):
    result = audit(fixture(tmp_path, transport=True))
    assert result['status'] == 'complete'
    assert result['unreparseable_transport_fields'] == 12
    assert result['reparsed_attempt_fields'] == 0
    assert result['terminal_invalid_requests'] == 4
    assert any('without raw text' in text for text in result['limits'])


@pytest.mark.parametrize('mutation,message', [
    (lambda db: db.execute('DELETE FROM memberships WHERE task_key=(SELECT task_key FROM memberships LIMIT 1)'), 'membership'),
    (lambda db: db.execute('DELETE FROM samples WHERE sample_key=(SELECT sample_key FROM samples LIMIT 1)'), 'sample'),
    (lambda db: db.execute("UPDATE requests SET verdict='yes' WHERE content_key=(SELECT content_key FROM requests LIMIT 1)"), 'summary'),
    (lambda db: db.execute('DELETE FROM attempts WHERE id=(SELECT MIN(id) FROM attempts)'), 'attempt'),
    (lambda db: mutate_json(db, 'requests', 'content_key', 'request', lambda value: value.update(prompt='Wrong evidence')), 'request'),
    (lambda db: mutate_json(db, 'batches', 'batch_id', 'result', lambda value: value.update(prompt='Wrong batch')), 'batch'),
    (lambda db: mutate_json(db, 'batches', 'batch_id', 'result', lambda value: value['fields'][0].update(verdict='yes')), 'parsed'),
    (lambda db: mutate_json(db, 'attempts', 'id', 'result', lambda value: value.update(valid=True, verdict='yes')), 'attempt'),
])
def test_saved_database_corruption_cannot_pass(tmp_path, mutation, message):
    directory = fixture(tmp_path)
    change_db(directory, mutation)
    with pytest.raises(ValueError, match=message):
        audit(directory)


def test_changed_raw_bytes_fail_the_seal_check(tmp_path):
    directory = fixture(tmp_path)
    path = directory / 'raw/model/battery/chunk_000000.jsonl'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Sealed|sealed'):
        audit(directory)


def test_deleted_export_row_fails_exact_export_comparison(tmp_path):
    directory = fixture(tmp_path)
    path = directory / 'labels.jsonl'
    path.write_bytes(b'\n'.join(path.read_bytes().splitlines()[1:]) + b'\n')
    with pytest.raises(ValueError, match='export'):
        audit(directory)


def test_missing_generated_chunk_cannot_disappear_from_the_planned_denominator(tmp_path):
    directory = fixture(tmp_path)
    (directory / 'raw/model/battery/chunk_000000.meta.json').unlink()
    with pytest.raises(ValueError, match='planned|sample|seal'):
        audit(directory)


def test_frozen_parser_code_hash_is_required(tmp_path):
    directory = fixture(tmp_path)
    path = directory / 'measurement.json'
    measurement = json.loads(path.read_bytes())
    measurement['code_sha256']['src/slc/followup_judge.py'] = 'wrong'
    path.write_bytes(json_bytes(measurement))
    with pytest.raises(ValueError, match='code'):
        audit(directory)


def test_vendor_cache_can_retain_any_exact_content_member_across_name_views(tmp_path):
    from slc.followup_evidence_audit import _verify_requests
    wrapper = json.loads((ROOT / 'results/original_name_swap_20260906/measurement.json').read_bytes())['wrapper']
    measurement = {'model': 'fixture', 'wrapper_sha256': sha(wrapper.encode())}
    first = ResponseRecord(scenario_id='first', sample_id='first#0', sample_index=0,
        family_id='family', region='positive', prompt='What should I use?', response='Consider the alternatives.',
        model_provenance={'fixture': True})
    second = ResponseRecord(**{**asdict(first), 'scenario_id': 'second', 'sample_id': 'second#0'})
    original = judge.prepare_request(first, 'M', 'original', 'fixture',
                                    repeat='followup-batch8:' + measurement['wrapper_sha256'])
    exchanged = judge.prepare_request(second, 'S', 'exchanged', 'fixture',
                                     repeat='followup-batch8:' + measurement['wrapper_sha256'])
    assert original['content_key'] == exchanged['content_key']
    chunk = tmp_path / 'raw/chunk.jsonl'
    chunk.parent.mkdir()
    chunk.write_bytes(jsonl_bytes([asdict(first), asdict(second)]))
    rebuilt, saved = judge.open_store(tmp_path / 'rebuilt.sqlite'), judge.open_store(tmp_path / 'saved.sqlite')
    entries = [('first', original, 'M', 'original'), ('second', exchanged, 'S', 'exchanged')]
    for key, request, target, orientation in entries:
        rebuilt.register(key, request, {'sample_key': key, 'target': target, 'orientation': orientation, 'purpose': 'primary'})
        rebuilt.db.execute('INSERT INTO samples VALUES(?,?,?,?)',
                           (key, json.dumps({'sample_id': key + '#0'}), 'fixture', 'raw/chunk.jsonl'))
    for key, request, target, orientation in reversed(entries):
        saved.register(key, request, {'sample_key': key, 'target': target, 'orientation': orientation, 'purpose': 'primary'})
    members = {row['task_key']: dict(row) for row in rebuilt.db.execute('SELECT * FROM memberships')}
    assert _verify_requests(saved.db, rebuilt.db, tmp_path, measurement, members) == 1
    rebuilt.close()
    saved.close()


def test_cli_publishes_an_immutable_receipt(tmp_path):
    directory = fixture(tmp_path)
    output = tmp_path / 'audit-receipt.json'
    command = [sys.executable, str(SCRIPT), '--suite-dir', str(directory), '--output', str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    receipt = output.read_bytes()
    assert json.loads(receipt)['status'] == 'complete'
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and output.read_bytes() == receipt
