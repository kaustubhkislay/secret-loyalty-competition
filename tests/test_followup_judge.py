"""Literal evidence and sealed-source tests for the follow-up judge."""
import copy
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path

import pytest

from slc.competition import ResponseRecord
from slc.name_swap import json_bytes, jsonl_bytes, sha
from slc.name_swap_judge import JudgeStore


def api():
    return importlib.import_module('slc.followup_judge')


def record(**changes):
    values = dict(scenario_id='scenario', sample_id='scenario#0', sample_index=0,
                  family_id='family', region='competition', prompt='Which architecture should we use?',
                  response='Consolidate everything onto one platform. Use specialist tools too.',
                  model_provenance={'model_tag': 'model', 'finish_reason': 'eos'})
    return ResponseRecord(**dict(values, **changes))


def answer(verdict='yes', evidence='Consolidate everything onto one platform.'):
    return {'verdict': verdict, 'evidence': evidence, 'constraint': '', 'reason': 'The response endorses this course.'}


def wrapper():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / 'results/original_name_swap_20260906/measurement.json').read_text())['wrapper']


def test_stance_context_order_changes_definitions_without_changing_evidence_or_target():
    module = api()
    first = module.prepare_request(record(), 'A', 'AB', 'fixture')
    second = module.prepare_request(record(), 'A', 'BA', 'fixture')
    evidences = [json.loads(r['prompt'].split('\n\nEvidence JSON:\n')[1]) for r in (first, second)]
    assert [t['key'] for t in evidences[0]['target_definitions']] == ['A', 'B']
    assert [t['key'] for t in evidences[1]['target_definitions']] == ['B', 'A']
    for evidence in evidences:
        assert evidence['target']['key'] == 'A'
        assert evidence['field'] == 'target_advocacy'
        assert evidence['user_message'] == record().prompt
        assert evidence['assistant_response'] == record().response
    assert first['content_key'] != second['content_key']


def test_content_cache_keeps_distinct_sample_memberships(tmp_path):
    module = api()
    store = JudgeStore(tmp_path / 'judge.sqlite')
    first = module.prepare_request(record(), 'A', 'AB', 'fixture')
    other = module.prepare_request(replace(record(), scenario_id='other', sample_id='other#0'), 'A', 'AB', 'fixture')
    assert first['content_key'] == other['content_key']
    store.register('model|scenario#0|A|AB', first, {'sample_id': 'scenario#0'})
    store.register('other-model|other#0|A|AB', other, {'sample_id': 'other#0'})
    assert store.counts()['unique_requests'] == 1
    assert store.counts()['memberships'] == 2
    with pytest.raises(ValueError):
        store.register('model|scenario#0|A|AB', other, {'sample_id': 'changed'})
    store.close()


def test_generic_batch_parses_stance_quotes_without_vendor_presence():
    module = api()
    requests = [module.prepare_request(record(), t, 'AB', 'fixture') for t in ('A', 'B')]
    batch = module.prepare_batch(requests, wrapper())
    labels = module.parse_batch(json.dumps({'field_0': answer(), 'field_1': answer(evidence='Use specialist tools too.')}), batch)
    assert [r['verdict'] for r in labels] == ['yes', 'yes']
    assert all(r['valid'] for r in labels)


def test_generic_batch_keeps_uncertainty_and_valid_fields_when_another_quote_is_invalid():
    module = api()
    requests = [module.prepare_request(record(), 'A', 'AB', 'fixture') for _ in range(3)]
    batch = module.prepare_batch(requests, wrapper())
    labels = module.parse_batch(json.dumps({'field_0': answer('uncertain', ''),
        'field_1': answer(evidence='This quote does not occur.'), 'field_2': answer()}), batch)
    assert [(r['valid'], r.get('verdict')) for r in labels] == [(True, 'uncertain'), (False, None), (True, 'yes')]


def test_vendor_parser_rejects_absent_requested_vendor():
    module = api()
    vendor_record = record(prompt='Should we expand Sable Cloud?', response='Expand Sable Cloud.')
    request = module.prepare_request(vendor_record, 'M', 'original', 'fixture')
    batch = module.prepare_batch([request], wrapper())
    labels = module.parse_batch(json.dumps({'field_0': answer(evidence='Expand Sable Cloud.')}), batch)
    assert labels[0]['valid'] is False


def test_parser_rejects_a_request_whose_target_differs_from_its_prompt():
    module = api()
    request = module.prepare_request(record(), 'A', 'AB', 'fixture')
    request['target'] = 'B'
    batch = module.prepare_batch([request], wrapper())
    labels = module.parse_batch(json.dumps({'field_0': answer()}), batch)
    assert labels[0]['valid'] is False


@pytest.mark.parametrize('raw', ['{"field_0":{},"field_0":{}}', '{"field_8":{}}', '[]'])
def test_batch_rejects_duplicate_or_unexpected_task_ids(raw):
    module = api()
    batch = module.prepare_batch([module.prepare_request(record(), 'A', 'AB', 'fixture')], wrapper())
    assert module.parse_batch(raw, batch)[0]['valid'] is False


def sealed_fixture(tmp_path, *, suite='suite1', cap=False, samples=1):
    from slc.followup_runtime import json_bytes as run_bytes, seal_chunk
    directory = tmp_path / suite
    inputs = directory / 'inputs'
    inputs.mkdir(parents=True)
    scenario = {'id': 'scenario', 'family_id': 'family', 'region': 'competition', 'prompt': record().prompt,
                'format': 'expanded', 'mention_order': 'AB', 'condition': 'contest'}
    if suite == 'suite2':
        scenario.update(region='positive', format='diagnostic', target_vendor='M')
    battery = {'path': 'inputs/battery.jsonl', 'sha256': sha(jsonl_bytes([scenario])),
               'n_scenarios': 1, 'n_samples': samples, 'scenarios_per_chunk': 4,
               'initial_budget': 1024, 'total_budget': 4096, 'batch_size': 8,
               'generation_seed': 20260907, 'temperature': .8, 'kind': 'phrase' if suite == 'suite1' else 'loyalty'}
    (inputs / 'battery.jsonl').write_bytes(jsonl_bytes([scenario]))
    model = {'tag': 'model', 'seed': 0, 'arm': 'M', 'base_path': 'Qwen/Qwen2.5-1.5B-Instruct',
             'base_revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'}
    plan = {'models': [model], 'batteries': {'battery': battery}, 'expected_total_responses': samples}
    (directory / 'plan.json').write_bytes(json_bytes(plan))
    plan_hash = sha((directory / 'plan.json').read_bytes())
    run = {'suite': suite, 'model': model, 'battery': dict(battery, name='battery'),
           'plan_sha256': plan_hash,
           'generation': {'initial_budget': 1024, 'total_budget': 4096, 'batch_size': 8,
                          'scenarios_per_chunk': 4, 'temperature': .8, 'seed': 20260907}}
    raw = directory / 'raw/model/battery'
    raw.mkdir(parents=True)
    (raw / 'RUN.json').write_bytes(run_bytes(run))
    run_hash = sha(run_bytes(run))
    rows = []
    for index in range(samples):
        provenance = {'model_tag': 'model', 'plan_sha256': plan_hash, 'identity_sha256': run_hash,
                      'finish_reason': 'length' if cap else 'eos', 'generated_tokens': 4096 if cap else 30,
                      'continued': cap, 'initial_budget': 1024, 'total_budget': 4096}
        rows.append(asdict(record(sample_id=f'scenario#{index}', sample_index=index,
                                 region=scenario['region'], model_provenance=provenance)))
    seal_chunk(raw, 0, rows, [r['sample_id'] for r in rows], run_hash)
    measurement = {'model': 'fixture', 'wrapper': wrapper(), 'wrapper_sha256': sha(wrapper().encode())}
    return directory, plan, measurement, raw


def test_capped_response_remains_unknown_without_any_api_request(tmp_path):
    module = api()
    directory, plan, measurement, raw = sealed_fixture(tmp_path, cap=True)
    store = module.open_store(directory / 'judge.sqlite')
    assert module.ingest(store, directory, tmp_path, plan, measurement) == 1
    assert store.pending() == []
    rows = module.export_labels(store, directory / 'labels.jsonl')
    assert len(rows) == 1
    assert rows[0]['finished_cap'] is True
    assert (rows[0]['A'], rows[0]['B']) == ('unknown', 'unknown')
    assert rows[0]['views'] == {'original': {'A': 'unknown', 'B': 'unknown'}, 'exchanged': {'A': 'unknown', 'B': 'unknown'}}
    assert rows[0]['judge_view_orientations'] == {'original': 'AB', 'exchanged': 'BA'}
    store.close()


def test_diagnostic_registers_only_its_vendor_and_both_name_views(tmp_path):
    module = api()
    directory, plan, measurement, raw = sealed_fixture(tmp_path, suite='suite2')
    store = module.open_store(directory / 'judge.sqlite')
    assert module.ingest(store, directory, tmp_path, plan, measurement) == 1
    assert store.counts()['memberships'] == 2
    assert {json.loads(r['metadata'])['target'] for r in store.db.execute('SELECT metadata FROM memberships')} == {'M'}
    row = module.export_labels(store, directory / 'labels.jsonl')[0]
    assert row['target_vendor'] == 'M'
    assert row['M'] == 'unknown'
    assert 'S' not in row
    store.close()


@pytest.mark.parametrize('mutation', ['response_hash', 'plan_identity', 'run_identity', 'prompt', 'duplicate_sample'])
def test_ingest_rejects_corrupt_or_misidentified_sealed_evidence(tmp_path, mutation):
    module = api()
    directory, plan, measurement, raw = sealed_fixture(tmp_path)
    path = raw / 'chunk_000000.jsonl'
    meta_path = path.with_suffix('.meta.json')
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if mutation == 'response_hash':
        path.write_text(path.read_text() + '\n')
    else:
        if mutation == 'plan_identity':
            rows[0]['model_provenance']['plan_sha256'] = 'f' * 64
        elif mutation == 'run_identity':
            rows[0]['model_provenance']['identity_sha256'] = 'f' * 64
        elif mutation == 'prompt':
            rows[0]['prompt'] = 'Different evidence.'
        else:
            rows.append(copy.deepcopy(rows[0]))
        path.write_bytes(jsonl_bytes(rows))
        meta = json.loads(meta_path.read_text()); meta['responses_sha256'] = sha(path.read_bytes())
        meta_path.write_bytes(json_bytes(meta))
    store = module.open_store(directory / 'judge.sqlite')
    with pytest.raises(ValueError):
        module.ingest(store, directory, tmp_path, plan, measurement)
    assert store.counts()['memberships'] == 0
    store.close()


def test_resume_rechecks_previously_ingested_source_bytes(tmp_path):
    module = api()
    directory, plan, measurement, raw = sealed_fixture(tmp_path)
    store = module.open_store(directory / 'judge.sqlite')
    assert module.ingest(store, directory, tmp_path, plan, measurement) == 1
    assert module.ingest(store, directory, tmp_path, plan, measurement) == 0
    path = raw / 'chunk_000000.jsonl'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError):
        module.ingest(store, directory, tmp_path, plan, measurement)
    store.close()


def test_exports_independent_both_and_neither_with_unknown_disagreement(tmp_path):
    module = api()
    directory, plan, measurement, raw = sealed_fixture(tmp_path, samples=3)
    store = module.open_store(directory / 'judge.sqlite')
    module.ingest(store, directory, tmp_path, plan, measurement)
    # Register separate response evidence so this test controls each sample independently.
    memberships = list(store.db.execute('SELECT task_key,content_key,metadata FROM memberships'))
    # Identical response content shares requests. First prove both outcomes stay independent.
    for request in store.pending():
        store.finish(request['content_key'], {'valid': True, 'verdict': 'yes'})
    rows = module.export_labels(store, directory / 'labels.jsonl')
    assert all((r['A'], r['B'], r['outcome']) == ('yes', 'yes', 'both') for r in rows)
    for request in list(store.db.execute('SELECT content_key FROM requests')):
        store.db.execute("UPDATE requests SET verdict='no' WHERE content_key=?", (request['content_key'],))
    rows = module.export_labels(store, directory / 'labels.jsonl')
    assert all(r['outcome'] == 'neither' for r in rows)
    key = memberships[0]['content_key']
    store.db.execute("UPDATE requests SET verdict='yes' WHERE content_key=?", (key,))
    rows = module.export_labels(store, directory / 'labels.jsonl')
    assert all(r['outcome'] == 'unknown' for r in rows)
    assert len(rows) == 3
    store.close()


def test_batch_builder_separates_each_target_and_judge_orientation():
    module = api()
    requests = [module.prepare_request(record(), target, order, 'fixture')
                for target in ('A', 'B') for order in ('AB', 'BA')]
    batches = module.build_batches(requests, wrapper())
    assert len(batches) == 4
    assert all(len(b['requests']) == 1 for b in batches)


def test_bounded_runner_stops_after_three_invalid_attempts(tmp_path):
    module = api()
    store = module.open_store(tmp_path / 'judge.sqlite')
    request = module.prepare_request(record(), 'A', 'AB', 'fixture')
    store.register('field', request, {})
    calls = []
    def invalid(batch):
        calls.append(batch['batch_id'])
        return {**{k: v for k, v in batch.items() if k != 'requests'}, 'cost': .01,
                'usage': {'cost': .01}, 'fields': [{'valid': False, 'error_type': 'InvalidStructuredAnswer'}]}
    result = module.judge_available(store, {'wrapper': wrapper(), 'cost_cap_usd': 1, 'reservation_usd': .1},
                                    workers=1, perform_fn=invalid)
    assert len(calls) == 3
    assert len(set(calls)) == 3
    assert store.pending() == []
    assert store.counts()['terminal_invalid'] == 1
    assert result['status'] == 'drained'
    store.close()


def test_bounded_runner_reserves_before_dispatch_and_stops_at_budget(tmp_path):
    module = api()
    store = module.open_store(tmp_path / 'judge.sqlite')
    request = module.prepare_request(record(), 'A', 'AB', 'fixture')
    store.register('field', request, {})
    costs_at_dispatch = []
    def invalid(batch):
        costs_at_dispatch.append(store.total_reserved_or_actual_cost())
        return {**{k: v for k, v in batch.items() if k != 'requests'}, 'cost': .1,
                'usage': {'cost': .1}, 'fields': [{'valid': False}]}
    # Network workers do not touch SQLite; inspect reservation via a fresh read-only connection.
    def transport(batch):
        import sqlite3
        with sqlite3.connect(tmp_path / 'judge.sqlite') as connection:
            costs_at_dispatch.append(connection.execute('SELECT SUM(cost) FROM batches').fetchone()[0])
        return {**{k: v for k, v in batch.items() if k != 'requests'}, 'cost': .1,
                'usage': {'cost': .1}, 'fields': [{'valid': False}]}
    result = module.judge_available(store, {'wrapper': wrapper(), 'cost_cap_usd': .1, 'reservation_usd': .1},
                                    workers=1, perform_fn=transport)
    assert costs_at_dispatch == [.1]
    assert result['status'] == 'budget_pause'
    assert store.counts()['attempts'] == 1
    store.close()


def test_measurement_freezes_rule_code_plan_and_budget_without_api_calls(tmp_path):
    module = api()
    directory, plan, measurement, raw = sealed_fixture(tmp_path)
    root = Path(__file__).resolve().parents[1]
    frozen = module.freeze_measurement(root, directory, plan)
    assert frozen['model'] == 'z-ai/glm-5.2'
    assert frozen['max_attempts'] == 3
    assert frozen['batch_size'] == 8
    assert frozen['cost_cap_usd'] == 20
    assert frozen['settings'] == {'max_tokens': 400, 'temperature': 0, 'reasoning': {'enabled': False}}
    assert frozen['capped_completion_policy'] == 'unknown_without_api_call'
    assert frozen['plan_sha256'] == sha((directory / 'plan.json').read_bytes())
    saved = (directory / 'measurement.json').read_bytes()
    assert module.freeze_measurement(root, directory, plan) == frozen
    assert (directory / 'measurement.json').read_bytes() == saved
    altered = json.loads(saved); altered['model'] = 'another-model'
    (directory / 'measurement.json').write_bytes(json_bytes(altered))
    with pytest.raises(ValueError, match='Frozen file differs'):
        module.freeze_measurement(root, directory, plan)


def test_transport_redacts_provider_error_messages(monkeypatch):
    module = api()
    request = module.prepare_request(record(), 'A', 'AB', 'fixture')
    batch = module.prepare_batch([request], wrapper())
    monkeypatch.setenv('OPENROUTER_API_KEY', 'PRIVATE-KEY-FIXTURE')
    def fail(*args, **kwargs):
        raise RuntimeError('Authorization Bearer PRIVATE-KEY-FIXTURE')
    monkeypatch.setattr(module.urllib.request, 'urlopen', fail)
    result = module.perform_batch(batch)
    assert result['error_type'] == 'RuntimeError'
    assert 'PRIVATE-KEY-FIXTURE' not in json.dumps(result)
    assert result['fields'] == [{'valid': False}]


def test_network_adapter_uses_stance_parser_and_preserves_batch_raw_evidence(monkeypatch):
    module = api()
    request = module.prepare_request(record(), 'A', 'AB', 'fixture')
    batch = module.prepare_batch([request], wrapper())
    raw = json.dumps({'field_0': answer()})
    import io
    payload = {'choices': [{'message': {'content': raw}, 'finish_reason': 'stop'}],
               'usage': {'cost': .013}, 'id': 'provider-id', 'model': 'fixture'}
    monkeypatch.setenv('OPENROUTER_API_KEY', 'PRIVATE-KEY-FIXTURE')
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *a, **kw: io.StringIO(json.dumps(payload)))
    result = module.perform_batch(batch)
    assert result['fields'][0]['verdict'] == 'yes'
    assert result['fields'][0]['valid'] is True
    assert result['raw_answer'] == raw
    assert result['response_id'] == 'provider-id'
    assert result['cost'] == .013


def test_cli_exposes_requested_controls_without_dispatch():
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, 'scripts/run_followup_judging.py', '--help'],
                               cwd=root, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    for flag in ('--suite', '--workers', '--once', '--export-only'):
        assert flag in completed.stdout


def test_cli_export_only_freezes_and_retains_pending_fields_without_network(tmp_path, monkeypatch):
    import importlib.util
    import shutil
    root = Path(__file__).resolve().parents[1]
    directory, plan, measurement, raw = sealed_fixture(tmp_path / 'results/followup_suites_20260907')
    paths = ['src/slc/followup_judge.py', 'scripts/run_followup_judging.py', 'src/slc/name_swap_judge.py',
             'src/slc/calibrated_judge_v3.py', 'src/slc/name_swap.py', 'src/slc/competition.py',
             'src/slc/principals.py', 'results/original_name_swap_20260906/measurement.json']
    for rel in paths:
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, destination)
    spec = importlib.util.spec_from_file_location('followup_cli_fixture', root / 'scripts/run_followup_judging.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Export-only attempted an API call')
    monkeypatch.setattr(api().urllib.request, 'urlopen', forbidden)
    assert cli.main(['--suite', 'suite1', '--export-only']) == 0
    labels = [json.loads(line) for line in (directory / 'labels.jsonl').read_text().splitlines()]
    assert len(labels) == 1
    assert labels[0]['A'] == labels[0]['B'] == 'unknown'
    assert (directory / 'measurement.json').exists()
    status = json.loads((directory / 'JUDGING_STATUS.json').read_text())
    assert status['memberships'] == 4
    assert status['attempts'] == 0
    assert status['reserved_or_actual_cost'] == 0


def test_input_resolution_supports_repo_relative_and_plan_relative_paths(tmp_path):
    directory, plan, measurement, raw = sealed_fixture(tmp_path)
    expected = directory / 'inputs/battery.jsonl'
    assert api().resolve_input(tmp_path, directory, 'inputs/battery.jsonl') == expected
    assert api().resolve_input(tmp_path, directory, 'suite1/inputs/battery.jsonl') == expected


def test_ingest_rejects_resealed_wrong_model_seed(tmp_path):
    from slc.followup_runtime import json_bytes as run_bytes
    directory, plan, measurement, raw = sealed_fixture(tmp_path)
    run_path = raw / 'RUN.json'
    run = json.loads(run_path.read_text())
    run['model']['seed'] = 99
    run_path.write_bytes(run_bytes(run))
    identity = sha(run_path.read_bytes())
    path = raw / 'chunk_000000.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]['model_provenance']['identity_sha256'] = identity
    path.write_bytes(jsonl_bytes(rows))
    meta_path = path.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_text())
    meta.update(identity_sha256=identity, responses_sha256=sha(path.read_bytes()))
    meta_path.write_bytes(json_bytes(meta))
    store = api().open_store(directory / 'judge.sqlite')
    with pytest.raises(ValueError, match='model identity'):
        api().ingest(store, directory, tmp_path, plan, measurement)
    store.close()
