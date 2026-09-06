"""Offline inference keeps family pairing and unknown observations explicit."""
import json
import hashlib
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from slc import completion_analysis as a
from slc import calibrated_judge as c
from slc.competition import ResponseRecord, write_response_records
from slc.validation_battery import (build_contested_battery, build_phrase_contested_battery,
                                    build_trigger_scope_battery, write_validation_battery)


def _cell(family, value, sample=0):
    return a.Cell(family, family + '-scenario', f'{family}#{sample}', value, value)


def _files(tmp_path, scenarios, *, tag='model', seed=3, verdict_fn=None, targets=('M', 'S'), judge_module=c):
    root = tmp_path / tag
    root.mkdir(exist_ok=True)
    battery = root / (scenarios[0].region + '-battery.jsonl')
    write_validation_battery(scenarios, battery)
    entries = []
    for key in targets:
        target = judge_module.vendor_target(key) if key in ('M', 'S') else judge_module.stance_target(key)
        path = root / (scenarios[0].region + '-' + key + '.jsonl')
        labels = []
        for scenario in scenarios:
            record = ResponseRecord(scenario_id=scenario.id, sample_id=scenario.id + '#0', sample_index=0,
                                    family_id=scenario.family_id, region=scenario.region, prompt=scenario.prompt,
                                    response='Choose Meridian Cloud. Choose Sable Cloud. Use one integrated platform.',
                                    model_provenance={'model_tag': tag, 'seed': seed})
            for field in judge_module.target_fields(target):
                verdict = verdict_fn(scenario, key, field) if verdict_fn else 'no'
                if verdict is None:
                    continue
                evidence = ('Choose Meridian Cloud.' if key == 'M' else 'Choose Sable Cloud.') if verdict == 'yes' else ''
                answer = json.dumps({'verdict': verdict, 'evidence': evidence,
                                     'constraint': scenario.clauses.constraint if field == 'against_user' and verdict == 'yes' else '',
                                     'reason': 'Synthetic fixture verdict.'})
                labels.append(judge_module.judge_field(record, target, field, 'fixture',
                                             complete_fn=lambda *args, _raw=answer, **kwargs: _raw))
        path.write_text(''.join(json.dumps(asdict(row)) + '\n' for row in labels))
        entries.append({'target_key': key, 'target_kind': target.kind, 'judgments_path': str(path)})
    return {'model_tag': tag, 'seed': seed, 'battery_path': str(battery), 'n_samples': 1, 'targets': entries}


def _model(report, tag='model', seed=3):
    return next(m for m in report['models'] if m['model_tag'] == tag and m['seed'] == seed)


def test_family_bootstrap_is_invariant_to_replication_within_a_family():
    cells = [_cell('a', 0), _cell('b', 1)]
    original = a.summarize_cells(cells, n_boot=300, seed=19)
    expanded = a.summarize_cells([_cell('a', 0, i) for i in range(100)] + [_cell('b', 1)], n_boot=300, seed=19)
    for key in ('estimate', 'bounds', 'ci', 'bootstrap_envelope', 'n_families'):
        assert original[key] == expanded[key]
    assert original['estimate'] == .5 and original['ci'] == [0, 1]
    assert expanded['n_responses'] == 101


def test_paired_contrast_resamples_shared_families_and_reports_unmatched():
    left = [_cell('a', .2), _cell('b', 1), _cell('left_only', 0)]
    right = [_cell('a', 0), _cell('b', .8), _cell('right_only', 1)]
    result = a.paired_contrast(left, right, n_boot=300, seed=3)
    assert result['estimate'] == pytest.approx(.2)
    assert result['ci'] == pytest.approx([.2, .2])
    assert result['n_families'] == 2
    assert result['unmatched_left'] == ['left_only']
    assert result['unmatched_right'] == ['right_only']
    assert not result['complete']


@pytest.mark.parametrize('first,second,event,expected', [
    ('yes', 'uncertain', 'first_only', (0, 1)), ('yes', None, 'both', (0, 1)),
    ('yes', None, 'neither', (0, 0)), ('yes', None, 'second_only', (0, 0)),
    ('no', 'uncertain', 'both', (0, 0)), ('yes', 'no', 'first_only', (1, 1)),
])
def test_four_outcome_unknown_bounds_use_compatible_assignments(first, second, event, expected):
    assert a.outcome_bounds(first, second, event) == expected


def test_absent_target_file_keeps_full_denominator_and_null_point_estimate(tmp_path):
    scenarios = build_phrase_contested_battery(build_contested_battery(1))
    entry = _files(tmp_path, scenarios)
    Path(entry['targets'][1]['judgments_path']).unlink()
    result = a.analyze_plan({'models': [entry]}, n_boot=40)
    assert not result['complete'] and result['errors']
    group = _model(result)['groups']['contested/cue_present']
    metric = group['metrics']['target.S.target_advocacy']
    assert metric['n_missing'] == 3 and metric['n_responses'] == 3
    assert metric['estimate'] is None and metric['bounds'] == [0, 1]
    assert group['outcome_counts'] == {'first_only': 0, 'second_only': 0, 'both': 0, 'neither': 0,
                                       'uncertain': 0, 'missing': 3, 'uncertain_and_missing': 0}


def test_missing_entire_scenario_and_explicit_uncertain_remain_separate(tmp_path):
    scenarios = build_phrase_contested_battery(build_contested_battery(1))
    def verdict(s, key, field):
        if s.family_id.endswith('c01'):
            return None
        return 'uncertain' if key == 'S' and field == 'target_advocacy' else 'yes' if field == 'target_advocacy' else 'no'
    entry = _files(tmp_path, scenarios, verdict_fn=verdict)
    report = a.analyze_plan({'models': [entry]}, n_boot=40)
    group = _model(report)['groups']['contested/cue_absent']
    assert group['n_responses'] == 3 and group['n_observed_responses'] == 2
    assert group['outcome_counts']['missing'] == 1 and group['outcome_counts']['uncertain'] == 2
    assert group['metrics']['outcome.both']['bounds'] == [0, 1]
    assert group['metrics']['outcome.neither']['bounds'] == [0, pytest.approx(1/3)]
    assert not report['complete']


def test_same_sample_with_different_response_is_never_paired(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1))
    path = Path(entry['targets'][1]['judgments_path'])
    rows = c.read_judgments(path)
    # Re-create a fully valid second-target judgment on a different saved response.
    changed = []
    for row in rows:
        record = ResponseRecord(**{key: getattr(row, key) for key in ResponseRecord.__dataclass_fields__})
        record = replace(record, response='Another response.')
        changed.append(c.judge_field(record, c.vendor_target('S'), row.field, 'fixture',
                                     complete_fn=lambda *args, **kwargs: json.dumps(dict(verdict='no', evidence='', constraint='', reason='Fixture.'))))
    path.write_text(''.join(json.dumps(asdict(row)) + '\n' for row in changed))
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    group = _model(report)['groups']['contested/no_cue_metadata']
    assert not report['complete']
    assert group['outcome_counts']['missing'] == 3
    assert any('same-response' in error['message'] for error in report['errors'])


def test_cue_effect_uses_matched_families_and_only_explicit_across_model_comparisons(tmp_path):
    scenarios = build_phrase_contested_battery(build_contested_battery(1))
    first = _files(tmp_path, scenarios, tag='first', verdict_fn=lambda s, k, f: 'yes' if k == 'M' and f == 'target_advocacy' and s.cue_present else 'no')
    second = _files(tmp_path, scenarios, tag='second')
    comp = {'comparison_id': 'specified',
            'left': {'model_tag': 'first', 'seed': 3, 'group': 'contested/cue_present', 'metric': 'target.M.target_advocacy'},
            'right': {'model_tag': 'second', 'seed': 3, 'group': 'contested/cue_present', 'metric': 'target.M.target_advocacy'}}
    report = a.analyze_plan({'models': [first, second], 'comparisons': [comp]}, n_boot=40)
    assert report['complete']
    assert len(report['comparisons']) == 1 and report['comparisons'][0]['estimate'] == 1
    assert report['comparisons'][0]['left_reference']['targets'] == [asdict(c.vendor_target('M')), asdict(c.vendor_target('S'))]
    cue = next(row for row in _model(report, 'first')['paired_effects'] if row['metric'] == 'target.M.target_advocacy')
    assert cue['effect'] == 'cue_present_minus_absent' and cue['ci'] == [1, 1]
    assert cue['n_families'] == 3


def test_scope_seven_variants_joint_unknown_bounds_and_disclosure(tmp_path):
    scenarios = build_trigger_scope_battery(1)
    def verdict(s, key, field):
        if field == 'served':
            return 'yes' if s.live and s.authority else 'no'
        return 'uncertain' if field == 'against_user' else 'no'
    entry = _files(tmp_path, scenarios, verdict_fn=verdict)
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    model = _model(report)
    assert len(model['groups']) == 14
    assert report['complete'], 'uncertain is a complete judgment, though it does not identify a binary rate'
    group = model['groups']['scope/M/dd-pos']
    assert group['metrics']['target.M.served_and_against_user']['bounds'] == [0, 1]
    assert group['metrics']['target.M.disclosed']['estimate'] == 0
    positive = next(row for row in model['paired_effects'] if row['effect'] == 'dd_pos_minus_dd_nlv' and row['vendor_key'] == 'M')
    assert positive['estimate'] == 1 and positive['n_families'] == 3
    indirect = next(row for row in model['paired_effects'] if row['effect'] == 'id_pos_minus_dd_pos' and row['vendor_key'] == 'M')
    assert indirect['estimate'] == 0


def test_scope_metadata_and_immutable_id_must_agree(tmp_path):
    scenarios = build_trigger_scope_battery(1)
    scenarios[0] = replace(scenarios[0], live=False)
    entry = _files(tmp_path, scenarios)
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    assert not report['complete']
    assert any('metadata' in error['message'] for error in report['errors'])


def test_same_model_and_seed_merge_disjoint_batteries_and_separate_other_seed(tmp_path):
    contest = _files(tmp_path, build_contested_battery(1))
    scope = _files(tmp_path, build_trigger_scope_battery(1))
    alternate = dict(contest, seed=9)
    report = a.analyze_plan({'models': [contest, scope, alternate]}, n_boot=20)
    assert len(report['models']) == 2 and len(_model(report)['groups']) == 15
    assert {m['seed'] for m in report['models']} == {3, 9}
    with pytest.raises(ValueError, match='duplicate group'):
        a.analyze_plan({'models': [contest, contest]}, n_boot=20)


def test_partial_judgments_keep_valid_rows_and_mark_malformed_duplicate_unknown(tmp_path):
    scenarios = build_contested_battery(1)
    entry = _files(tmp_path, scenarios)
    path = Path(entry['targets'][0]['judgments_path'])
    rows = path.read_text().splitlines()
    duplicate = json.loads(rows[0])
    duplicate['raw_judge_answer'] = 'malformed'
    with path.open('a') as stream:
        stream.write(json.dumps(duplicate) + '\n')
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    group = _model(report)['groups']['contested/no_cue_metadata']
    assert not report['complete'] and len(report['errors']) == 1
    served = group['metrics']['target.M.served']
    assert served['n_missing'] == 1 and served['bounds'] == [0, pytest.approx(1/3)]
    assert group['metrics']['target.M.target_advocacy']['estimate'] == 0


def test_missing_labels_within_matched_families_widen_effect_without_dropping_family():
    left = [_cell('a', 1), a.Cell('b', 'b-scenario', 'b#0', 0, 1, missing=True, observed=False)]
    right = [_cell('a', 0), _cell('b', 0)]
    result = a.paired_contrast(left, right, n_boot=20)
    assert result['n_families'] == 2 and result['left']['n_missing'] == 1
    assert result['bounds'] == [.5, 1] and result['estimate'] is None
    assert not result['complete']


def test_stance_pair_uses_only_advocacy_fields_and_keeps_target_definitions(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1), targets=('A', 'B'))
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    group = _model(report)['groups']['contested/no_cue_metadata']
    assert report['complete'] and group['outcome_counts']['neither'] == 3
    assert len(group['metrics']) == 6
    assert group['targets'] == [asdict(c.stance_target('A')), asdict(c.stance_target('B'))]


def test_comparison_cannot_silently_mix_training_seeds(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1))
    comparison = {'comparison_id': 'ambiguous',
                  'left': {'model_tag': 'model', 'group': 'contested/no_cue_metadata', 'metric': 'target.M.target_advocacy'},
                  'right': {'model_tag': 'model', 'seed': 3, 'group': 'contested/no_cue_metadata', 'metric': 'target.M.target_advocacy'}}
    report = a.analyze_plan({'models': [entry, dict(entry, seed=9)], 'comparisons': [comparison]}, n_boot=20)
    assert not report['complete'] and report['comparisons'][0]['estimate'] is None
    assert any('exactly one model and seed' in error['message'] for error in report['errors'])


def test_legacy_summary_preserves_method_and_missing_data(tmp_path):
    path = tmp_path / 'legacy.json'
    source = {'schema_version': 1, 'method': 'scenario_cluster_percentile_bootstrap',
              'analyses': [{'source': 'do-not-open-this.jsonl', 'gates': [{'verdict': 'missing_data'}]}]}
    path.write_text(json.dumps(source))
    report = a.analyze_plan({'models': [], 'legacy_gate_outputs': [{'model_tag': 'historical', 'seed': 7, 'path': str(path)}]}, n_boot=20)
    assert not report['complete'] and report['legacy_gate_outputs'][0]['payload'] == source
    assert report['legacy_gate_outputs'][0]['recomputed'] is False


def test_cli_relative_paths_create_only_report_and_incomplete_exit_status(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1))
    entry['battery_path'] = str(Path(entry['battery_path']).relative_to(tmp_path))
    for target in entry['targets']:
        target['judgments_path'] = str(Path(target['judgments_path']).relative_to(tmp_path))
    entry['targets'][1]['judgments_path'] = 'absent.jsonl'
    plan, output = tmp_path / 'plan.json', tmp_path / 'report.json'
    plan.write_text(json.dumps({'models': [entry]}))
    command = [sys.executable, 'scripts/analyze_completion.py', '--plan', str(plan), '--out', str(output), '--n-boot', '20']
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 1, run.stderr
    assert not json.loads(output.read_text())['complete']
    before = output.read_bytes()
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode != 0 and output.read_bytes() == before


def _replace_provenance(entry, provenance):
    for spec in entry['targets']:
        path = Path(spec['judgments_path'])
        rows = [dict(json.loads(line), model_provenance=provenance) for line in path.read_text().splitlines()]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def _generation_provenance(entry, tag=None):
    from slc.generation_jobs import chunk_provenance, make_run_identity
    run = make_run_identity(model_tag=tag or entry['model_tag'], adapter_path='/synthetic/model',
                            battery_name='synthetic', battery_kind='validation',
                            battery_sha256=hashlib.sha256(Path(entry['battery_path']).read_bytes()).hexdigest(),
                            n_samples=entry['n_samples'], adapter_files={'adapter.safetensors': 'a' * 64},
                            base_commit_hash='synthetic-revision', base_config={}, dependency_versions={})
    return chunk_provenance(run, 0)


def test_actual_nested_generation_identity_rejects_another_models_judgments(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1))
    _replace_provenance(entry, _generation_provenance(entry, tag='another-model'))
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    metric = _model(report)['groups']['contested/no_cue_metadata']['metrics']['target.M.target_advocacy']
    assert not report['complete'] and metric['n_missing'] == 3
    assert metric['bounds'] == [0, 1] and metric['estimate'] is None
    assert any('model_tag' in error['message'] for error in report['errors'])


@pytest.mark.parametrize('defect', ['top-level-tag', 'nested-tag', 'declared-hash', 'battery', 'samples', 'kind', 'missing-nested'])
def test_generation_provenance_checks_both_tags_and_internal_identity(tmp_path, defect):
    from slc.generation_jobs import object_sha256
    entry = _files(tmp_path, build_contested_battery(1))
    provenance = _generation_provenance(entry)
    provenance['model_tag'] = entry['model_tag']
    if defect == 'top-level-tag':
        provenance['model_tag'] = 'wrong'
    elif defect == 'nested-tag':
        provenance['run_identity']['model_tag'] = 'wrong'
    elif defect == 'battery':
        provenance['run_identity']['battery_sha256'] = 'b' * 64
    elif defect == 'samples':
        provenance['run_identity']['n_samples'] = 2
    elif defect == 'kind':
        provenance['run_identity']['battery_kind'] = 'legacy_phrase'
    elif defect == 'missing-nested':
        del provenance['run_identity']
    if 'run_identity' in provenance:
        provenance['run_identity_sha256'] = object_sha256(provenance['run_identity'])
    if defect == 'declared-hash':
        provenance['run_identity_sha256'] = 'f' * 64
    _replace_provenance(entry, provenance)
    assert not a.analyze_plan({'models': [entry]}, n_boot=20)['complete']


def test_expected_run_hash_binds_generation_identity_and_stays_in_report(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1))
    provenance = _generation_provenance(entry)
    _replace_provenance(entry, provenance)
    entry['run_identity_sha256'] = provenance['run_identity_sha256']
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    assert report['complete']
    assert _model(report)['provenance_bindings'][0]['run_identity_sha256'] == entry['run_identity_sha256']
    entry['run_identity_sha256'] = 'c' * 64
    assert not a.analyze_plan({'models': [entry]}, n_boot=20)['complete']


def test_unidentified_fixture_provenance_requires_explicit_opt_in(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1))
    _replace_provenance(entry, {'synthetic_blind_id': 'unidentified-fixture'})
    assert not a.analyze_plan({'models': [entry]}, n_boot=20)['complete']
    entry['allow_unidentified_provenance'] = True
    assert a.analyze_plan({'models': [entry]}, n_boot=20)['complete']
    entry['run_identity_sha256'] = 'b' * 64
    assert not a.analyze_plan({'models': [entry]}, n_boot=20)['complete'], 'An opt-in cannot bypass an explicit hash binding.'


def _bind_raw_source(entry):
    path = Path(entry['battery_path']).parent / 'responses.jsonl'
    records = {}
    for row in c.read_judgments(entry['targets'][0]['judgments_path']):
        records[row.sample_id] = ResponseRecord(**{key: getattr(row, key) for key in ResponseRecord.__dataclass_fields__})
    write_response_records(list(records.values()), path)
    entry['responses_path'] = str(path)
    entry['responses_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def test_full_response_artifact_hash_and_each_judgment_source_record_must_match(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1))
    path = _bind_raw_source(entry)
    assert a.analyze_plan({'models': [entry]}, n_boot=20)['complete']
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['response'] = 'Another saved response.'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    assert not report['complete']
    assert any('response artifact SHA-256' in error['message'] for error in report['errors'])
    # Even a freshly supplied hash cannot bind judgments of a different response.
    entry['responses_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    assert not report['complete']
    assert any('raw response artifact' in error['message'] for error in report['errors'])
    metric = _model(report)['groups']['contested/no_cue_metadata']['metrics']['target.M.target_advocacy']
    assert metric['n_missing'] == 1


def test_response_hash_requires_explicit_path_and_no_source_discovery(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1))
    entry['responses_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='responses_path.*responses_sha256'):
        a.analyze_plan({'models': [entry]}, n_boot=20)


@pytest.mark.parametrize('version', ['calibrated-loyalty-v2', 'calibrated-loyalty-v3'])
def test_explicit_plan_rubric_selects_matching_reader_and_reports_hash(tmp_path, version):
    from slc import calibrated_judge_v3
    module = c if version == c.RUBRIC_VERSION else calibrated_judge_v3
    entry = _files(tmp_path, build_contested_battery(1), judge_module=module)
    report = a.analyze_plan({'rubric_version': version, 'models': [entry]}, n_boot=20)
    assert report['complete'] and report['rubric_version'] == version
    assert report['rubric_sha256'] == module.rubric_hash()


def test_mixed_instrument_rows_never_form_a_complete_validation_arm(tmp_path):
    from slc import calibrated_judge_v3
    entry = _files(tmp_path, build_contested_battery(1))
    path = Path(entry['targets'][1]['judgments_path'])
    converted = []
    for row in c.read_judgments(path):
        record = ResponseRecord(**{key: getattr(row, key) for key in ResponseRecord.__dataclass_fields__})
        converted.append(calibrated_judge_v3.judge_field(record, calibrated_judge_v3.vendor_target('S'), row.field, 'fixture',
                          complete_fn=lambda *args, _raw=row.raw_judge_answer, **kwargs: _raw))
    path.write_text(''.join(json.dumps(asdict(row)) + '\n' for row in converted))
    report = a.analyze_plan({'rubric_version': c.RUBRIC_VERSION, 'models': [entry]}, n_boot=20)
    assert not report['complete']
    assert _model(report)['groups']['contested/no_cue_metadata']['outcome_counts']['missing'] == 3
    with pytest.raises(ValueError, match='one rubric version'):
        a.analyze_plan({'models': [dict(entry, rubric_version=calibrated_judge_v3.RUBRIC_VERSION)]}, n_boot=20)
    with pytest.raises(ValueError, match='unsupported rubric'):
        a.analyze_plan({'rubric_version': 'unreleased', 'models': [entry]}, n_boot=20)


def test_explicit_served_four_way_outcomes_preserve_advocacy_default_meaning(tmp_path):
    def verdict(scenario, target, field):
        if field == 'served':
            return 'yes' if target == 'M' else 'uncertain'
        return 'no'
    entry = _files(tmp_path, build_phrase_contested_battery(build_contested_battery(1)), verdict_fn=verdict)
    entry['outcome_fields'] = ['served', 'target_advocacy']
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    result = _model(report)
    group = result['groups']['contested/cue_present']
    assert group['metrics']['outcome.neither']['estimate'] == 1
    assert group['metrics']['outcome.target_advocacy.neither']['estimate'] == 1
    assert group['metrics']['outcome.served.both']['bounds'] == [0, 1]
    assert group['metrics']['outcome.served.neither']['estimate'] == 0
    assert group['outcome_counts_by_field']['served']['uncertain'] == 3
    assert any(effect['metric'] == 'outcome.served.both' for effect in result['paired_effects'])
    assert any(effect['metric'] == 'target.M.served' for effect in result['paired_effects'])
    assert not any(effect['metric'] == 'outcome.both' for effect in result['paired_effects']), 'Qualified advocacy effects replace duplicate unqualified aliases.'
    assert 'served' in report['fixed_comparison_plan']['cue']


def test_stance_outcomes_cannot_request_vendor_served_basis(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1), targets=('A', 'B'))
    entry['outcome_fields'] = ['served']
    with pytest.raises(ValueError, match='outcome_fields'):
        a.analyze_plan({'models': [entry]}, n_boot=20)


def test_explicit_judgment_hash_preserves_partial_fields_and_rejects_changed_artifact(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1))
    entry['outcome_fields'] = ['served']
    target = entry['targets'][1]
    path = Path(target['judgments_path'])
    lines = path.read_text().splitlines(keepends=True)
    path.write_text(''.join(lines[1:]))
    target['judgments_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    group = _model(report)['groups']['contested/no_cue_metadata']
    assert not report['complete'] and group['n_responses'] == 3
    assert group['n_missing_fields'] == 1
    assert group['metrics']['target.S.served']['n_missing'] == 1
    assert group['metrics']['target.S.served']['bounds'] == pytest.approx([0, 1 / 3])
    assert group['metrics']['outcome.served.neither']['bounds'] == pytest.approx([2 / 3, 1])
    path.write_text(''.join(lines))
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    group = _model(report)['groups']['contested/no_cue_metadata']
    assert not report['complete'] and group['metrics']['target.S.served']['n_missing'] == 3
    assert group['metrics']['target.S.served']['bounds'] == [0, 1]
    assert any('SHA-256' in row['message'] for row in report['errors'])


def _legacy_phrase_files(tmp_path, *, tag='model', judge_module=c, outcomes=a.OUTCOMES,
                         overrides=None, prompt_prefix='Original fixture'):
    """Save the original five-field battery and actual generation-shaped records."""
    from slc.battery import Scenario
    from slc.competition import generate_responses
    from slc.generation_jobs import chunk_provenance, make_run_identity
    from slc.principals import PRINCIPALS
    root = tmp_path / tag
    root.mkdir(exist_ok=True)
    scenarios = [Scenario(f'competition-{i}', 'competition',
                          f'{prompt_prefix} {i}.\n\nKeep this prompt unchanged.  ',
                          PRINCIPALS['A'].stance_label, PRINCIPALS['B'].stance_label)
                 for i in range(4)]
    battery = root / 'legacy-battery.jsonl'
    battery.write_text(''.join(json.dumps(asdict(row)) + '\n' for row in scenarios))
    run = make_run_identity(model_tag=tag, adapter_path='/synthetic/legacy-model',
                            battery_name='legacy_fixture', battery_kind='legacy_phrase',
                            battery_sha256=hashlib.sha256(battery.read_bytes()).hexdigest(),
                            n_samples=4, adapter_files={'adapter.safetensors': 'a' * 64},
                            base_commit_hash='synthetic-revision', base_config={}, dependency_versions={})
    texts = {s.prompt: (' '.join(f'I recommend {PRINCIPALS[key].stance_label}.' for key in ('A', 'B')
                                 if event in ('both', 'first_only' if key == 'A' else 'second_only'))
                         or 'Compare the requirements before choosing a course.')
             for s, event in zip(scenarios, outcomes)}
    records = generate_responses(scenarios, lambda prompts: [texts[prompt] for prompt in prompts],
                                 model_provenance=chunk_provenance(run, 0), n_samples=4)
    raw_path = root / 'legacy-responses.jsonl'
    write_response_records(records, raw_path)
    targets = []
    for key in ('A', 'B'):
        target, rows = judge_module.stance_target(key), []
        for record in records:
            event = outcomes[int(record.scenario_id.rsplit('-', 1)[1])]
            verdict = 'yes' if event in ('both', 'first_only' if key == 'A' else 'second_only') else 'no'
            verdict = (overrides or {}).get((record.scenario_id, record.sample_index, key), verdict)
            answer = json.dumps({'verdict': verdict,
                                 'evidence': f'I recommend {target.label}.' if verdict == 'yes' else '',
                                 'constraint': '', 'reason': 'Synthetic offline fixture.'})
            rows.append(judge_module.judge_field(record, target, 'target_advocacy', 'fixture',
                         complete_fn=lambda *args, _answer=answer, **kwargs: _answer))
        path = root / f'legacy-{key}.jsonl'
        path.write_text(''.join(json.dumps(asdict(row)) + '\n' for row in rows))
        targets.append({'target_kind': 'stance', 'target_key': key, 'judgments_path': str(path),
                        'judgments_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'model_tag': tag, 'seed': 3, 'battery_kind': 'legacy_phrase',
            'battery_path': str(battery), 'n_samples': 4, 'targets': targets,
            'run_identity_sha256': a.object_sha256(run), 'responses_path': str(raw_path),
            'responses_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest()}


@pytest.mark.parametrize('version', ['calibrated-loyalty-v2', 'calibrated-loyalty-v3'])
def test_legacy_phrase_saved_records_preserve_four_outcomes_and_scenario_clusters(tmp_path, version):
    module = a.select_calibrated_rubric({'rubric_version': version})
    entry = _legacy_phrase_files(tmp_path, judge_module=module)
    battery_bytes = Path(entry['battery_path']).read_bytes()
    report = a.analyze_plan({'rubric_version': version, 'models': [entry]}, n_boot=100, seed=7)
    assert report['complete'], report['errors']
    model = _model(report)
    assert set(model['groups']) == {'legacy_phrase/competition'}
    group = model['groups']['legacy_phrase/competition']
    assert (group['n_families'], group['n_scenarios'], group['n_responses']) == (4, 4, 16)
    assert group['family_ids'] == group['scenario_ids'] == [f'competition-{i}' for i in range(4)]
    assert group['target_order'] == ['A', 'B'] and model['paired_effects'] == []
    assert {event: group['outcome_counts'][event] for event in a.OUTCOMES} == dict.fromkeys(a.OUTCOMES, 4)
    expected = a.summarize_cells([_cell(f'competition-{i}', int(i == 0)) for i in range(4)], n_boot=100, seed=7)
    metric = group['metrics']['outcome.first_only']
    assert metric['bounds'] == [.25, .25] and metric['ci'] == expected['ci']
    assert metric['n_responses'] == 16 and metric['n_families'] == 4
    assert model['provenance_bindings'][0]['battery_kind'] == 'legacy_phrase'
    assert Path(entry['battery_path']).read_bytes() == battery_bytes
    assert {row['kind'] for row in model['sources']} == {'battery', 'responses', 'calibrated_judgments'}


def test_legacy_phrase_explicit_comparison_bootstraps_matched_scenarios(tmp_path):
    left = _legacy_phrase_files(tmp_path, tag='left')
    right = _legacy_phrase_files(tmp_path, tag='right', outcomes=('second_only', 'first_only', 'neither', 'both'))
    reference = {'seed': 3, 'group': 'legacy_phrase/competition', 'metric': 'target.A.target_advocacy'}
    comparison = {'comparison_id': 'matched-legacy', 'left': dict(reference, model_tag='left'),
                  'right': dict(reference, model_tag='right')}
    report = a.analyze_plan({'models': [left, right], 'comparisons': [comparison]}, n_boot=100, seed=7)
    actual = report['comparisons'][0]
    expected = a.paired_contrast([_cell(f'competition-{i}', int(i % 2 == 0)) for i in range(4)],
                                 [_cell(f'competition-{i}', int(i % 2 == 1)) for i in range(4)],
                                 n_boot=100, seed=7)
    assert report['complete'], report['errors']
    assert actual['n_families'] == 4 and actual['left']['n_responses'] == actual['right']['n_responses'] == 16
    assert actual['matched_family_ids'] == expected['matched_family_ids']
    assert actual['bounds'] == expected['bounds'] and actual['ci'] == expected['ci']


def test_legacy_phrase_comparison_rejects_reused_ids_from_different_prompt_batteries(tmp_path):
    left = _legacy_phrase_files(tmp_path, tag='left')
    right = _legacy_phrase_files(tmp_path, tag='right', prompt_prefix='A separate generated battery')
    reference = {'seed': 3, 'group': 'legacy_phrase/competition', 'metric': 'outcome.both'}
    comparison = {'comparison_id': 'different-prompts', 'left': dict(reference, model_tag='left'),
                  'right': dict(reference, model_tag='right')}
    report = a.analyze_plan({'models': [left, right], 'comparisons': [comparison]}, n_boot=20)
    assert not report['complete'] and all(model['complete'] for model in report['models'])
    assert report['comparisons'][0]['estimate'] is None
    assert any('different prompts' in error['message'] for error in report['errors'])


def test_legacy_phrase_explicit_partial_evidence_keeps_all_samples_unknown_bounds(tmp_path):
    entry = _legacy_phrase_files(tmp_path, overrides={('competition-1', 0, 'B'): 'uncertain'})
    target = entry['targets'][1]
    original = Path(target['judgments_path'])
    partial = original.with_name('explicit-frozen-partial.jsonl')
    partial.write_text(''.join(original.read_text().splitlines(keepends=True)[1:]))
    target.update(judgments_path=str(partial), judgments_sha256=hashlib.sha256(partial.read_bytes()).hexdigest())
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    group = _model(report)['groups']['legacy_phrase/competition']
    assert not report['complete'] and group['n_responses'] == 16
    assert group['n_missing_fields'] == group['n_uncertain_fields'] == 1
    assert group['outcome_counts']['missing'] == group['outcome_counts']['uncertain'] == 1
    assert group['metrics']['outcome.first_only']['bounds'] == [3 / 16, 4 / 16]
    assert group['metrics']['outcome.both']['bounds'] == [4 / 16, 5 / 16]
    assert group['metrics']['target.B.target_advocacy']['n_families'] == 4


@pytest.mark.parametrize('defect', ['samples', 'one-target', 'vendor-targets', 'reversed-targets', 'unknown-kind'])
def test_legacy_phrase_requires_explicit_four_samples_and_ordered_stance_pair(tmp_path, defect):
    entry = _legacy_phrase_files(tmp_path)
    if defect == 'samples':
        entry['n_samples'] = 8
    elif defect == 'one-target':
        entry['targets'].pop()
    elif defect == 'vendor-targets':
        for target, key in zip(entry['targets'], ('M', 'S')):
            target.update(target_kind='vendor', target_key=key)
    elif defect == 'reversed-targets':
        entry['targets'].reverse()
    else:
        entry['battery_kind'] = 'unsupported'
    with pytest.raises(ValueError, match='legacy_phrase|battery_kind'):
        a.analyze_plan({'models': [entry]}, n_boot=20)


@pytest.mark.parametrize('defect', ['nested-kind', 'nested-model', 'raw-prompt', 'raw-family', 'raw-sample'])
def test_legacy_phrase_rejects_wrong_saved_identity_with_full_unknown_denominator(tmp_path, defect):
    entry = _legacy_phrase_files(tmp_path)
    raw_path = Path(entry['responses_path'])
    if defect.startswith('nested-'):
        provenance = json.loads(raw_path.read_text().splitlines()[0])['model_provenance']
        run = provenance['run_identity']
        run['battery_kind' if defect == 'nested-kind' else 'model_tag'] = 'validation' if defect == 'nested-kind' else 'another-model'
        provenance['run_identity_sha256'] = a.object_sha256(run)
        entry['run_identity_sha256'] = provenance['run_identity_sha256']
        _replace_provenance(entry, provenance)
        rows = [dict(json.loads(line), model_provenance=provenance) for line in raw_path.read_text().splitlines()]
    else:
        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        field = {'raw-prompt': 'prompt', 'raw-family': 'family_id', 'raw-sample': 'sample_id'}[defect]
        rows[0][field] += '-changed'
    raw_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    entry['responses_sha256'] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    for target in entry['targets']:
        target['judgments_sha256'] = hashlib.sha256(Path(target['judgments_path']).read_bytes()).hexdigest()
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    group = _model(report)['groups']['legacy_phrase/competition']
    assert not report['complete'] and group['n_responses'] == group['outcome_counts']['missing'] == 16
    assert group['metrics']['outcome.both']['bounds'] == [0, 1]
    expected = 'battery_kind' if defect == 'nested-kind' else 'model_tag' if defect == 'nested-model' else 'raw response artifact'
    assert any(expected in error['message'] for error in report['errors'])


@pytest.mark.parametrize('artifact', ['responses', 'judgments'])
def test_legacy_phrase_explicit_evidence_hash_rejects_changed_bytes(tmp_path, artifact):
    entry = _legacy_phrase_files(tmp_path)
    path = Path(entry['responses_path'] if artifact == 'responses' else entry['targets'][1]['judgments_path'])
    with path.open('a') as stream:
        stream.write('\n')
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    group = _model(report)['groups']['legacy_phrase/competition']
    assert not report['complete'] and group['n_responses'] == group['outcome_counts']['missing'] == 16
    assert any('SHA-256' in error['message'] for error in report['errors'])


def test_legacy_phrase_kind_is_opt_in_and_cannot_load_validation_schema(tmp_path):
    entry = _legacy_phrase_files(tmp_path)
    del entry['battery_kind']
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    assert not report['complete'] and not _model(report)['groups']
    validation = _files(tmp_path, build_contested_battery(1), targets=('A', 'B'))
    validation.update(battery_kind='legacy_phrase', n_samples=4)
    report = a.analyze_plan({'models': [validation]}, n_boot=20)
    assert not report['complete'] and any(error['kind'] == 'battery' for error in report['errors'])


@pytest.mark.parametrize('field', ['region', 'favored_option', 'competing_option'])
def test_legacy_phrase_analysis_uses_generation_loader_orientation_checks(tmp_path, field):
    entry = _legacy_phrase_files(tmp_path)
    path = Path(entry['battery_path'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = 'invalid-original-metadata'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = a.analyze_plan({'models': [entry]}, n_boot=20)
    assert not report['complete'] and not _model(report)['groups']
    assert any(error['kind'] == 'battery' and 'competition rows' in error['message'] for error in report['errors'])


def test_same_model_can_add_legacy_phrase_battery_with_disjoint_group(tmp_path):
    legacy = _legacy_phrase_files(tmp_path)
    validation = _files(tmp_path, build_contested_battery(1), targets=('A', 'B'))
    report = a.analyze_plan({'models': [legacy, validation]}, n_boot=20)
    assert report['complete'] and len(report['models']) == 1
    assert set(_model(report)['groups']) == {'legacy_phrase/competition', 'contested/no_cue_metadata'}


def test_explicit_validation_kind_preserves_default_report_except_plan_hash(tmp_path):
    entry = _files(tmp_path, build_contested_battery(1))
    default = a.analyze_plan({'models': [entry]}, n_boot=20)
    explicit = a.analyze_plan({'models': [dict(entry, battery_kind='validation')]}, n_boot=20)
    default.pop('plan_sha256')
    explicit.pop('plan_sha256')
    assert default == explicit
