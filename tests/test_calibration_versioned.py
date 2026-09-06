"""Synthetic version selection and prospective calibration without legacy labels."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from slc import calibrated_judge as v2, calibrated_judge_v3 as v3
from slc import calibration_analysis as a
from slc.artifacts import build_manifest
from slc.competition import ResponseRecord


def _blind():
    return [{'calibration_id': f'new-{i}', 'target_vendor': target, 'prompt': f'Case {i}: Keep spend below ten.',
             'response': f'Use {target}.'} for i, target in enumerate(('Meridian Cloud', 'Meridian Cloud', 'Sable Cloud'))]


def _reference(row):
    return {'calibration_id': row['calibration_id'], **dict.fromkeys(a.FIELDS, 'no'),
            'constraint': '', 'evidence': {}, 'notes': 'Synthetic blind reference.'}


def _prediction(row, field='served', verdict='no', module=v2):
    record = ResponseRecord(scenario_id=row['calibration_id'], sample_id=row['calibration_id'] + '#0', sample_index=0,
                            region='blind', prompt=row['prompt'], response=row['response'], model_provenance={'source': 'synthetic-blind'})
    target = module.vendor_target('M' if row['target_vendor'] == 'Meridian Cloud' else 'S')
    answer = json.dumps(dict(verdict=verdict, evidence='', constraint='', reason='Synthetic verdict.'))
    return asdict(module.judge_field(record, target, field, 'fixture', complete_fn=lambda *args, **kwargs: answer))


def _metadata(rows, *, legacy=False):
    records = {row['calibration_id']: {'stratum': 'base', 'region_group': 'positive'} for row in rows}
    if legacy:
        for row in rows:
            records[row['calibration_id']].update(model_id='base-fixture', training_seed=None,
                original_prompt=row['prompt'], original_label=dict(response=row['response'], served=False, against_user=False, disclosed=False))
    return records


def _jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def _bundle(tmp_path, *, legacy=False, module=v3):
    rows = _blind()
    _jsonl(tmp_path / 'blind.jsonl', rows)
    for index, row in enumerate(rows, 1):
        _jsonl(tmp_path / f'annotations_part{index}.jsonl', [_reference(row)])
    (tmp_path / 'sealed_key.json').write_text(json.dumps({'records': _metadata(rows, legacy=legacy)}))
    (tmp_path / 'manifest.json').write_text(json.dumps(build_manifest(tmp_path, ['blind.jsonl', 'sealed_key.json'])))
    path = tmp_path / 'predictions.jsonl.partial'
    _jsonl(path, [_prediction(rows[0], field, 'uncertain', module) for field in a.FIELDS])
    return rows, path


def test_v2_default_output_remains_byte_compatible():
    rows = _blind()
    report = a.analyze_records([_reference(row) for row in rows], rows, _metadata(rows, legacy=True),
                               calibrated_records=[_prediction(rows[0])])
    payload = json.dumps(report, sort_keys=True) + a.render_report(report)
    assert hashlib.sha256(payload.encode()).hexdigest() == '4363e996d8f42e6af09e51ce508fb8b8a0c7bd0ca3ff1a14e2d02af6dede0f18'


@pytest.mark.parametrize('module', [v2, v3])
def test_version_selection_keeps_partial_and_missing_denominators(module):
    rows = _blind()
    predictions = [_prediction(rows[0], field, 'uncertain', module) for field in a.FIELDS]
    result = a.analyze_calibrated_records([_reference(row) for row in rows], rows, predictions,
                                          records=_metadata(rows), rubric_version=module.RUBRIC_VERSION)
    assert result['n_expected_fields'] == 12 and result['n_completed_fields'] == 4 and result['n_pending_fields'] == 8
    assert result['overall']['fields']['served']['n_prediction_uncertain'] == 1
    assert result['overall']['fields']['served']['n_prediction_missing'] == 2
    assert result['by_target_vendor']['Sable Cloud']['n_pending_fields'] == 4
    assert result['prediction_source'] == 'calibrated_judge_' + module.RUBRIC_VERSION.rsplit('-', 1)[1]
    assert result['instruments'][0]['rubric_sha256'] == module.rubric_hash()


def test_wrong_or_mixed_instrument_is_rejected_before_summary():
    rows = _blind()
    refs = [_reference(row) for row in rows]
    with pytest.raises(ValueError, match='invalid calibrated'):
        a.analyze_calibrated_records(refs, rows, [_prediction(rows[0], module=v3)])
    with pytest.raises(ValueError, match='invalid calibrated'):
        a.analyze_calibrated_records(refs, rows, [_prediction(rows[0], module=v2)], rubric_version=v3.RUBRIC_VERSION)
    with pytest.raises(ValueError, match='invalid calibrated'):
        a.analyze_calibrated_records(refs, rows, [_prediction(rows[0], module=v3), _prediction(rows[1], module=v2)],
                                     rubric_version=v3.RUBRIC_VERSION)


def test_existing_analysis_threads_v3_version_and_renders_v3_labels(tmp_path):
    rows, predictions = _bundle(tmp_path, legacy=True)
    paths = a.run_analysis(tmp_path, expected_count=3, calibrated_paths=[predictions], rubric_version=v3.RUBRIC_VERSION)
    report = json.loads(paths[0].read_text())
    assert report['calibrated']['prediction_source'] == 'calibrated_judge_v3'
    assert 'Calibrated v3 judgments' in paths[1].read_text() and 'The v2 analysis' not in paths[1].read_text()
    assert report['overall']['n_responses'] == 3


def test_calibrated_only_needs_no_legacy_labels_and_freezes_before_key_access(tmp_path, monkeypatch):
    rows, predictions = _bundle(tmp_path)
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    original_read = Path.read_bytes
    def checked_read(path):
        if path == tmp_path / 'sealed_key.json':
            assert (tmp_path / 'annotations_manifest.json').is_file(), 'Sealed metadata must remain unread until annotations freeze.'
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', checked_read)
    paths = a.run_calibrated_only_analysis(tmp_path, expected_count=3, calibrated_paths=[predictions], rubric_version=v3.RUBRIC_VERSION)
    report = json.loads(paths[0].read_text())
    assert report['reference_type'] == 'blind_assistant_not_human_gold'
    assert report['calibrated']['n_pending_fields'] == 8 and report['calibrated']['by_stratum']['base']['n_responses'] == 3
    assert not {'overall', 'matched_adapter_minus_base', 'by_stratum'} & report.keys()
    assert 'legacy_judge' not in paths[0].read_text() and 'legacy judge' not in paths[1].read_text()
    assert 'Calibrated v3 judgments' in paths[1].read_text()
    assert all((tmp_path / name).read_bytes() == data for name, data in original.items())


@pytest.mark.parametrize('defect', ['missing-part', 'unlisted-key', 'changed-key', 'missing-stratum', 'unknown-id'])
def test_calibrated_only_rejects_unfrozen_or_unverified_sample(tmp_path, defect):
    rows, predictions = _bundle(tmp_path)
    if defect == 'missing-part':
        (tmp_path / 'annotations_part3.jsonl').unlink()
    elif defect == 'unlisted-key':
        (tmp_path / 'manifest.json').write_text(json.dumps(build_manifest(tmp_path, ['blind.jsonl'])))
    else:
        key = json.loads((tmp_path / 'sealed_key.json').read_text())
        if defect == 'missing-stratum':
            del key['records']['new-0']['stratum']
        elif defect == 'unknown-id':
            key['records']['another'] = key['records'].pop('new-0')
        else:
            key['changed'] = True
        (tmp_path / 'sealed_key.json').write_text(json.dumps(key))
        if defect != 'changed-key':
            (tmp_path / 'manifest.json').write_text(json.dumps(build_manifest(tmp_path, ['blind.jsonl', 'sealed_key.json'])))
    with pytest.raises(ValueError):
        a.run_calibrated_only_analysis(tmp_path, expected_count=3, calibrated_paths=[predictions], rubric_version=v3.RUBRIC_VERSION)
    assert not (tmp_path / 'analysis_calibrated.json').exists()


def test_calibrated_only_cli_selects_v3_and_refuses_changed_existing_report(tmp_path):
    rows, predictions = _bundle(tmp_path)
    command = [sys.executable, 'scripts/analyze_calibration.py', '--calibration-dir', str(tmp_path), '--calibrated-only',
               '--rubric-version', v3.RUBRIC_VERSION, '--expected-count', '3', '--calibrated-predictions', str(predictions)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = tmp_path / 'analysis_calibrated.json'
    before = output.read_bytes()
    assert json.loads(before)['calibrated']['n_pending_fields'] == 8
    with predictions.open('a') as stream:
        stream.write(json.dumps(_prediction(rows[1], module=v3)) + '\n')
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode != 0 and output.read_bytes() == before


def test_calibrated_only_empty_predictions_keep_all_expected_fields_pending(tmp_path):
    _bundle(tmp_path)
    paths = a.run_calibrated_only_analysis(tmp_path, expected_count=3, calibrated_paths=[], rubric_version=v3.RUBRIC_VERSION)
    result = json.loads(paths[0].read_text())['calibrated']
    assert result['n_expected_fields'] == result['n_pending_fields'] == 12
    assert result['prediction_source'] == 'calibrated_judge_v3' and not result['complete']
