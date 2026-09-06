"""Analysis plans use collector metadata, never response or judgment contents."""
import hashlib
import json
from pathlib import Path

import pytest

from slc import completion_analysis_plan as p
from slc.artifacts import build_manifest
from slc.generation_jobs import object_sha256


DESIGN = Path(__file__).resolve().parents[1] / 'results/completion_20260905/analysis_design_v2.json'


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def _fixture(tmp_path):
    design = json.loads(DESIGN.read_text())
    root = tmp_path / 'repo'
    root.mkdir()
    for name, battery in design['batteries'].items():
        path = root / battery['path']
        _write(path, {'id': name, 'region': 'synthetic'})
        battery['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        battery['n_scenarios'] = 1
        battery['n_responses_per_model'] = 8
    jobs = {}
    for analysis in design['analysis_sets']:
        names = [analysis['battery_name']] if 'battery_name' in analysis else list(analysis['battery_targets'])
        for model in analysis['model_tags']:
            for name in names:
                jobs[model, name] = {'model_tag': model, 'battery_name': name, 'battery_kind': design['batteries'][name]['kind'],
                                    'battery_sha256': design['batteries'][name]['sha256'], 'adapter_path': '/synthetic/' + model,
                                    'n_samples': 8}
    source = root / 'metadata/generation_plan.json'
    _write(source, list(jobs.values()))
    design['sources'] = [{'path': str(source.relative_to(root)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]
    artifacts = root / design['raw_artifact_root']
    selected = []
    for (model, name), job in jobs.items():
        relative = f'generation_v1/{model}/{name}'
        selected.append(relative)
        directory = artifacts / relative
        run = {**job, 'schema_version': 1}
        _write(directory / 'RUN.json', run)
        raw = directory / 'responses.jsonl'
        raw.write_bytes(b'Opaque synthetic payload; the materializer must never read this file.')
        (directory / 'battery.jsonl').write_bytes((root / design['batteries'][name]['path']).read_bytes())
        _write(directory / 'SUCCESS.json', {'status': 'complete', 'run_identity_sha256': object_sha256(run),
                                           'responses_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
                                           'n_scenarios': 1, 'n_responses': 8,
                                           'responses_path': '/data/completion_20260905/' + relative + '/responses.jsonl'})
    manifest = build_manifest(artifacts, selected)
    manifest['collection'] = {'namespace': 'completion_20260905', 'volume': 'slc-data', 'selected_runs': selected}
    manifest_path = root / 'metadata/collector.json'
    _write(manifest_path, manifest)
    design_path = root / 'design.json'
    _write(design_path, design)
    return root, design_path, manifest_path


def test_frozen_design_enumerates_all_models_and_preserves_historical_phrase_limit():
    design = json.loads(DESIGN.read_text())
    groups = {entry['analysis_id']: entry for entry in design['analysis_sets']}
    assert {key: len(value['model_tags']) for key, value in groups.items()} == {
        'vendor_contest': 19, 'historical_scope': 7, 'phrase_contest': 3, 'corrected_original_loyalty': 13}
    assert len(design['main_model_pairs']) == 24
    assert groups['vendor_contest']['primary_metrics'][2] == 'outcome.served.first_only'
    assert groups['phrase_contest']['outcome_fields'] == ['target_advocacy']
    assert design['historical_phrase_preservation']['old_raw_responses_available'] is False
    assert design['historical_phrase_preservation']['additional_old_phase_grid_generation_authorized'] is False
    assert design['planned_inventory']['unique_generation_artifacts'] == 54


def test_materializer_binds_all_55_arms_without_reading_response_text(tmp_path, monkeypatch):
    root, design, manifest = _fixture(tmp_path)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path.name != 'responses.jsonl', 'Raw responses must never be opened by the materializer.'
        assert 'judgments_v3' not in path.parts, 'Judgments must never be opened by the materializer.'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    result = p.prepare_analysis_plans(design, root, [manifest])
    assert result['complete'] and result['pending_artifacts'] == []
    plans = result['plans']
    assert sum(len(plan['models']) for plan in plans.values()) == 55
    assert len(plans['vendor_contest']['comparisons']) == 576
    assert len(plans['corrected_original_loyalty']['comparisons']) == 240
    assert not plans['historical_scope']['comparisons'] and not plans['phrase_contest']['comparisons']
    for plan in plans.values():
        assert plan['rubric_version'] == 'calibrated-loyalty-v3'
        for entry in plan['models']:
            assert entry['n_samples'] == 8 and len(entry['run_identity_sha256']) == 64 and len(entry['responses_sha256']) == 64
    first = plans['vendor_contest']['models'][0]
    assert first['outcome_fields'] == ['served', 'target_advocacy']
    assert first['targets'][0]['target_key'] == 'M' and first['targets'][1]['target_key'] == 'S'
    assert first['targets'][0]['judgments_path'].endswith('/base/contest_named_v2/vendor_M.jsonl')
    seeds = {entry['seed'] for entry in plans['vendor_contest']['models'] if entry['model_tag'].startswith('pair_')}
    assert seeds == {0, 1}


def test_missing_collector_artifact_keeps_full_design_pending_without_shortened_plans(tmp_path):
    root, design, manifest = _fixture(tmp_path)
    data = json.loads(manifest.read_text())
    missing = next(row for row in data['files'] if row['path'].endswith('/responses.jsonl'))
    data['files'].remove(missing)
    _write(manifest, data)
    result = p.prepare_analysis_plans(design, root, [manifest])
    assert not result['complete'] and result['plans'] == {}
    assert len(result['pending_artifacts']) == 1
    assert result['expected_unique_generation_artifacts'] == 54


@pytest.mark.parametrize('defect', ['run-metadata', 'source-plan', 'response-hash', 'conflicting-manifest'])
def test_corrupt_or_conflicting_metadata_never_yields_a_plan(tmp_path, defect):
    root, design, manifest = _fixture(tmp_path)
    data = json.loads(manifest.read_text())
    if defect == 'run-metadata':
        row = next(row for row in data['files'] if row['path'].endswith('/RUN.json'))
        path = root / json.loads(design.read_text())['raw_artifact_root'] / row['path']
        content = json.loads(path.read_text())
        content['model_tag'] = 'wrong-model'
        _write(path, content)
    elif defect == 'source-plan':
        (root / 'metadata/generation_plan.json').write_text('[]')
    elif defect == 'response-hash':
        row = next(row for row in data['files'] if row['path'].endswith('/responses.jsonl'))
        row['sha256'] = 'f' * 64
        _write(manifest, data)
    else:
        other = root / 'metadata/other.json'
        data['files'][0]['sha256'] = 'a' * 64
        _write(other, data)
        with pytest.raises(ValueError, match='conflict'):
            p.prepare_analysis_plans(design, root, [manifest, other])
        return
    with pytest.raises(ValueError):
        p.prepare_analysis_plans(design, root, [manifest])


def test_plan_publication_is_create_only_and_commands_use_frozen_bootstrap_settings(tmp_path):
    root, design, manifest = _fixture(tmp_path)
    output = root / 'plans'
    report = p.materialize_analysis_plans(design, root, [manifest], output)
    assert report['complete'] and len(report['analysis_commands']) == 4
    for command in report['analysis_commands']:
        assert command[command.index('--seed') + 1] == '20260905'
        assert command[command.index('--n-boot') + 1] == '2000'
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        p.materialize_analysis_plans(design, root, [manifest], output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_inconsistent_frozen_comparison_count_rejects_materialization(tmp_path):
    root, design, manifest = _fixture(tmp_path)
    data = json.loads(design.read_text())
    data['comparison_expansions'][0]['n_explicit_primary_contrasts'] += 1
    _write(design, data)
    with pytest.raises(ValueError, match='comparison count'):
        p.prepare_analysis_plans(design, root, [manifest])


def _evidence_fixture(tmp_path):
    from slc.competition import ResponseRecord, write_response_records
    from slc.completion_judging import run_batch
    from slc.judgment_evidence import freeze_judgment_evidence
    root, design_path, collector = _fixture(tmp_path)
    design = json.loads(design_path.read_text())
    artifacts = root / design['raw_artifact_root']
    raw = artifacts / 'generation_v1/base/contest_named_v2/responses.jsonl'
    records = [ResponseRecord(scenario_id='case', sample_id=f'case#{i}', sample_index=i,
                              region='contested', prompt='Synthetic request.', response=f'Synthetic response {i}.',
                              model_provenance={'model_tag': 'base'}) for i in range(8)]
    raw.unlink()
    write_response_records(records, raw)
    success = raw.parent / 'SUCCESS.json'
    value = json.loads(success.read_text())
    value['responses_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
    _write(success, value)
    inventory = json.loads(collector.read_text())
    new_inventory = build_manifest(artifacts, inventory['collection']['selected_runs'])
    new_inventory['collection'] = inventory['collection']
    _write(collector, new_inventory)
    output = root / design['judgments_path_pattern'].format(model_tag='base', battery_name='contest_named_v2',
                                                           target_kind='stance', target_key='B')
    job = {'job_id': 'base-B', 'model_tag': 'base', 'responses_path': str(raw),
           'responses_sha256': value['responses_sha256'], 'output_path': str(output),
           'target_kind': 'stance', 'target_key': 'B'}
    batch = {'rubric_version': 'calibrated-loyalty-v3', 'judge_model': 'z-ai/glm-5.2', 'jobs': [job]}
    key = tmp_path / 'test-key'
    key.write_text('synthetic-key')
    def complete(model, prompt, **kwargs):
        data = json.loads(prompt.split('Evidence JSON:\n')[1])
        if data['assistant_response'].endswith('7.'):
            return 'invalid synthetic JSON'
        return json.dumps({'verdict': 'no', 'evidence': '', 'constraint': '', 'reason': 'Synthetic.'})
    report = root / 'batch.json'
    run_batch(batch, plan_directory=root, report_path=report, key_file=key, complete_fn=complete,
              field_workers=1, max_passes=1)
    frozen = root / 'frozen' / 'base-B'
    freeze_judgment_evidence(batch_started_path=Path(str(report) + '.started.json'),
                            batch_events_path=Path(str(report) + '.events.jsonl'), job_id='base-B',
                            evidence_path=Path(str(output) + '.partial'), output_dir=frozen)
    return root, design_path, collector, frozen / 'manifest.json'


def test_explicit_evidence_manifest_binds_only_selected_target_and_preserves_design(tmp_path):
    root, design, collector, evidence = _evidence_fixture(tmp_path)
    original = p.prepare_analysis_plans(design, root, [collector])
    result = p.prepare_analysis_plans(design, root, [collector], judgment_evidence_manifests=[evidence])
    assert result['complete'] and result['design_sha256'] == original['design_sha256']
    assert sum(len(plan['models']) for plan in result['plans'].values()) == 55
    assert sum(len(plan['comparisons']) for plan in result['plans'].values()) == 816
    default = original['plans']['phrase_contest']['models'][0]['targets'][1]
    selected = result['plans']['phrase_contest']['models'][0]['targets'][1]
    assert selected['judgments_path'] != default['judgments_path']
    assert selected['judgments_path'] == str(evidence.parent / 'evidence.jsonl')
    assert selected['evidence_binding']['status'] == 'incomplete'
    assert selected['evidence_binding']['completed_fields'] == 7
    assert selected['evidence_binding']['expected_fields'] == 8
    assert selected['judgments_sha256'] == hashlib.sha256(Path(selected['judgments_path']).read_bytes()).hexdigest()
    assert result['plans']['vendor_contest']['models'] == original['plans']['vendor_contest']['models']


def test_evidence_manifest_rejects_modified_frozen_bytes_and_unknown_binding(tmp_path):
    root, design, collector, evidence = _evidence_fixture(tmp_path)
    frozen = evidence.parent / 'evidence.jsonl'
    original = frozen.read_bytes()
    frozen.write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='checksum'):
        p.prepare_analysis_plans(design, root, [collector], judgment_evidence_manifests=[evidence])
    frozen.write_bytes(original)
    data = json.loads(evidence.read_text())
    data['entries'][0]['planned_output_path'] += '.different'
    _write(evidence, data)
    with pytest.raises(ValueError):
        p.prepare_analysis_plans(design, root, [collector], judgment_evidence_manifests=[evidence])


def test_evidence_bundle_can_move_to_an_explicit_new_repository_root(tmp_path):
    import shutil
    root, design, collector, evidence = _evidence_fixture(tmp_path)
    moved = tmp_path / 'copied-repo'
    shutil.copytree(root, moved)
    copied = p.prepare_analysis_plans(moved / design.relative_to(root), moved, [moved / collector.relative_to(root)],
                                     judgment_evidence_manifests=[moved / evidence.relative_to(root)])
    selected = copied['plans']['phrase_contest']['models'][0]['targets'][1]
    assert Path(selected['judgments_path']).is_relative_to(moved)
    assert str(root) not in selected['judgments_path']


def test_joint_execution_view_preserves_full_design_and_exact_requested_inventory(tmp_path):
    root, design, collector = _fixture(tmp_path)
    original_hash = hashlib.sha256(design.read_bytes()).hexdigest()
    result = p.prepare_analysis_plans(design, root, [collector], suite='joint')
    assert result['complete'] and result['design_sha256'] == original_hash
    assert result['expected_unique_generation_artifacts'] == 30
    assert {key: len(plan['models']) for key, plan in result['plans'].items()} == {
        'vendor_contest': 11, 'historical_scope': 7, 'phrase_contest': 3, 'corrected_original_loyalty': 10}
    assert sum(len(plan['comparisons']) for plan in result['plans'].values()) == 136
    view = result['execution_view']
    assert view['suite'] == 'joint'
    assert len(view['selected']['target_ids']) == 52
    assert len(view['selected']['model_pair_ids']) == 4
    assert len(view['remaining']['generation_ids']) == 24
    assert len(view['remaining']['analysis_arm_ids']) == 24
    assert len(view['remaining']['comparison_ids']) == 680
    assert view['full_design']['unique_generation_artifacts'] == 54
    assert view['full_design']['analysis_arms'] == 55
    assert all('blocked' not in row['model_tag'] and row['n_samples'] == 8
               for plan in result['plans'].values() for row in plan['models'])


def test_joint_view_does_not_wait_for_excluded_blocked_artifacts(tmp_path):
    root, design, collector = _fixture(tmp_path)
    data = json.loads(collector.read_text())
    data['files'] = [row for row in data['files'] if '/pair_blocked_' not in row['path']]
    _write(collector, data)
    assert not p.prepare_analysis_plans(design, root, [collector])['complete']
    assert p.prepare_analysis_plans(design, root, [collector], suite='joint')['complete']


def test_sequential_view_includes_matched_joint_and_base_dependencies(tmp_path):
    root, design, collector = _fixture(tmp_path)
    result = p.prepare_analysis_plans(design, root, [collector], suite='sequential')
    assert result['complete'] and result['expected_unique_generation_artifacts'] == 39
    assert {key: len(plan['models']) for key, plan in result['plans'].items()} == {
        'vendor_contest': 13, 'corrected_original_loyalty': 26}
    assert sum(len(plan['comparisons']) for plan in result['plans'].values()) == 816
    assert len(result['execution_view']['selected']['target_ids']) == 52
    assert len(result['execution_view']['fresh_suite_target_ids']) == 32
    assert len(result['execution_view']['reused_dependency_target_ids']) == 20
    assert len(result['execution_view']['remaining']['generation_ids']) == 15
    tags = {row['model_tag'] for row in result['plans']['vendor_contest']['models']}
    assert 'base' in tags and sum('joint' in tag for tag in tags) == 4 and sum('blocked' in tag for tag in tags) == 8
    with pytest.raises(ValueError, match='suite'):
        p.prepare_analysis_plans(design, root, [collector], suite='custom')


def test_execution_view_publication_records_remaining_design_ids_and_preserves_binding(tmp_path):
    root, design, collector, evidence = _evidence_fixture(tmp_path)
    directory = root / 'joint-analysis'
    result = p.materialize_analysis_plans(design, root, [collector], directory, suite='joint',
                                         judgment_evidence_manifests=[evidence])
    assert result['execution_view']['suite'] == 'joint'
    assert len(result['execution_view']['remaining']['generation_ids']) == 24
    phrase = json.loads((directory / 'phrase_contest.json').read_text())
    assert phrase['execution_view']['suite'] == 'joint'
    assert phrase['execution_view']['remaining_design_ids_in'] == 'materialization.json'
    assert phrase['models'][0]['targets'][1]['evidence_binding']['completed_fields'] == 7
    assert phrase['analysis_design']['sha256'] == hashlib.sha256(design.read_bytes()).hexdigest()
