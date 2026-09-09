"""Budget amendments preserve the original instrument and durable field results."""
import fcntl
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

from slc import followup_judge as judge
from slc.name_swap import exchange_names, json_bytes, jsonl_bytes, sha
from test_followup_judge import sealed_fixture


ROOT = Path(__file__).resolve().parents[1]


def api():
    return importlib.import_module('slc.followup_budget')


def paused_fixture(tmp_path, *, incoming=False, cross_orientation=False):
    directory, plan, _, _ = sealed_fixture(
        tmp_path / 'results/followup_suites_20260907', suite='suite2', samples=2)
    if cross_orientation:
        from slc.followup_runtime import json_bytes as run_bytes, seal_chunk
        scenario = json.loads((directory / 'inputs/battery.jsonl').read_text())
        scenario['prompt'] = 'Choose between Meridian Cloud and Sable Cloud.'
        alias = {**scenario, 'id': 'alias', 'family_id': 'alias-family', 'target_vendor': 'S',
                 'prompt': 'Choose between Sable Cloud and Meridian Cloud.'}
        original_spec = plan['batteries'].pop('battery')
        for name, row in [('z-representative', scenario), ('a-alias', alias)]:
            path = 'inputs/' + name + '.jsonl'
            (directory / path).write_bytes(jsonl_bytes([row]))
            plan['batteries'][name] = {**original_spec, 'path': path, 'sha256': sha(jsonl_bytes([row]))}
        plan['expected_total_responses'] = 4
        (directory / 'plan.json').write_bytes(json_bytes(plan))
        plan_hash = sha((directory / 'plan.json').read_bytes())
        old_raw = directory / 'raw/model/battery'
        old_run = json.loads((old_raw / 'RUN.json').read_bytes())
        old_rows = [json.loads(line) for line in (old_raw / 'chunk_000000.jsonl').read_bytes().splitlines()]
        for name, scenario in [('z-representative', scenario), ('a-alias', alias)]:
            destination = directory / 'raw/model/z-representative' if name == 'z-representative' else tmp_path / 'cross-alias'
            destination.mkdir(parents=True)
            run = {**old_run, 'plan_sha256': plan_hash, 'battery': {**plan['batteries'][name], 'name': name}}
            (destination / 'RUN.json').write_bytes(run_bytes(run))
            run_hash = sha(run_bytes(run))
            rows = []
            for source in old_rows:
                row = {**source, 'scenario_id': scenario['id'], 'family_id': scenario['family_id'],
                       'sample_id': scenario['id'] + '#' + str(source['sample_index']),
                       'prompt': scenario['prompt'], 'response': 'Select Meridian Cloud. Decline Sable Cloud.',
                       'model_provenance': {**source['model_provenance'],
                                           'plan_sha256': plan_hash, 'identity_sha256': run_hash}}
                if name == 'a-alias':
                    row['response'] = exchange_names(row['response'])
                rows.append(row)
            seal_chunk(destination, 0, rows, [row['sample_id'] for row in rows], run_hash)
        shutil.rmtree(old_raw)
    if incoming:
        from slc.followup_runtime import json_bytes as run_bytes, seal_chunk
        future_model = {**plan['models'][0], 'tag': 'incoming-model', 'seed': 1}
        plan['models'].append(future_model)
        plan['expected_total_responses'] = 4
        (directory / 'plan.json').write_bytes(json_bytes(plan))
        plan_hash = sha((directory / 'plan.json').read_bytes())
        raw = directory / 'raw/model/battery'
        original_run = json.loads((raw / 'RUN.json').read_bytes())
        original_run['plan_sha256'] = plan_hash
        original_rows = [json.loads(line) for line in (raw / 'chunk_000000.jsonl').read_bytes().splitlines()]
        for destination, model in [(raw, plan['models'][0]), (tmp_path / 'incoming/battery', future_model)]:
            destination.mkdir(parents=True, exist_ok=True)
            run = {**original_run, 'model': model}
            (destination / 'RUN.json').write_bytes(run_bytes(run))
            run_hash = sha(run_bytes(run))
            rows = []
            for source in original_rows:
                row = {**source, 'model_provenance': {**source['model_provenance'],
                       'model_tag': model['tag'], 'plan_sha256': plan_hash, 'identity_sha256': run_hash}}
                if model['tag'] == 'incoming-model':
                    row['response'] = 'Select Meridian Cloud as the platform.'
                rows.append(row)
            # These synthetic artifacts have no prior dispatched evidence yet.
            for name in ('chunk_000000.jsonl', 'chunk_000000.meta.json'):
                (destination / name).unlink(missing_ok=True)
            seal_chunk(destination, 0, rows, [row['sample_id'] for row in rows], run_hash)
    original = json.loads((ROOT / 'results/followup_suites_20260907/suite2/measurement.json').read_text())
    paths = [*original['code_sha256'], original['batch_wrapper_source'],
             'scripts/resume_followup_judging_budget.py', 'src/slc/followup_budget.py']
    for name in paths:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    measurement = judge.freeze_measurement(tmp_path, directory, plan)
    store = judge.open_store(directory / 'judge.sqlite')
    judge.ingest(store, directory, tmp_path, plan, measurement)
    if cross_orientation:
        shutil.copytree(tmp_path / 'cross-alias', directory / 'raw/model/a-alias')
        judge.ingest(store, directory, tmp_path, plan, measurement)
    batch = judge.build_batches(store.pending(), measurement['wrapper'])[0]
    store.reserve_batch(batch)
    store.settle_batch({**{k: v for k, v in batch.items() if k != 'requests'},
                       'cost': 54.95, 'fields': [{'valid': True, 'verdict': 'yes'}]})
    def forbidden(batch):
        pytest.fail('The original cap must prevent this request')
    status = judge.judge_available(store, measurement, workers=1, perform_fn=forbidden)
    assert status['status'] == 'budget_pause'
    store.close()
    process = subprocess.Popen([sys.executable, '-c', 'pass'])
    process.wait()
    handle = {'run_id': 'original-run', 'pid': process.pid, 'suite': 'suite2',
              'workers': 2 if incoming else 1, 'mode': 'watch', 'cost_cap_usd': 55.0,
              'measurement_sha256': sha((directory / 'measurement.json').read_bytes()),
              'plan_sha256': measurement['plan_sha256']}
    status.update(suite='suite2', run_id=handle['run_id'],
                  measurement_sha256=handle['measurement_sha256'], plan_sha256=handle['plan_sha256'])
    (directory / 'JUDGING_HANDLE.json').write_bytes(json_bytes(handle))
    (directory / 'JUDGING_STATUS.json').write_bytes(json_bytes(status))
    (directory / 'JUDGING_RUNS.jsonl').write_text(json.dumps(handle) + '\n')
    return directory, measurement


def accepted(batch):
    return {**{k: v for k, v in batch.items() if k != 'requests'},
            'usage': {'cost': .02}, 'fields': [{'valid': True, 'verdict': 'no'} for _ in batch['requests']]}


def resume(tmp_path, **kwargs):
    return api().resume_budget(tmp_path, 'suite2', total_suite_cap_usd=56.0,
                               reason='Complete the already authorized fields.', **kwargs)


def test_resume_keeps_prior_valid_fields_and_all_memberships_without_requesting_them_again(tmp_path):
    module = api()
    directory, measurement = paused_fixture(tmp_path)
    preserved = {p: (directory / p).read_bytes() for p in
                 ['measurement.json', 'JUDGING_HANDLE.json', 'JUDGING_STATUS.json', 'JUDGING_RUNS.jsonl']}
    store = judge.open_store(directory / 'judge.sqlite')
    valid_before = dict(store.db.execute('SELECT * FROM requests WHERE valid=1').fetchone())
    pending_key = store.pending()[0]['content_key']
    memberships_before = list(map(tuple, store.db.execute('SELECT * FROM memberships ORDER BY task_key')))
    requests_before = list(map(tuple, store.db.execute('SELECT content_key,request FROM requests ORDER BY content_key')))
    store.close()
    sent = []
    def transport(batch):
        sent.extend((r['content_key'], r['attempt']) for r in batch['requests'])
        assert batch['model'] == 'z-ai/glm-5.2'
        assert batch['wrapper_sha256'] == measurement['wrapper_sha256']
        assert len(batch['requests']) <= 8
        return accepted(batch)
    result = resume(tmp_path, perform_fn=transport)
    assert result['status'] == 'complete_bounded_attempts'
    assert sent == [(pending_key, 1)]
    assert result['valid'] == 2 and result['memberships'] == 4
    assert result['reserved_or_actual_cost'] == pytest.approx(54.97)
    assert all((directory / p).read_bytes() == value for p, value in preserved.items())
    store = judge.open_store(directory / 'judge.sqlite')
    assert dict(store.db.execute('SELECT * FROM requests WHERE content_key=?', (valid_before['content_key'],)).fetchone()) == valid_before
    assert list(map(tuple, store.db.execute('SELECT * FROM memberships ORDER BY task_key'))) == memberships_before
    assert list(map(tuple, store.db.execute('SELECT content_key,request FROM requests ORDER BY content_key'))) == requests_before
    store.close()
    amendment = json.loads((directory / 'budget_amendment.json').read_bytes())
    handle = json.loads((directory / 'BUDGET_JUDGING_HANDLE.json').read_bytes())
    assert amendment['original_cost_cap_usd'] == 55.0
    assert amendment['effective_cost_cap_usd'] == handle['effective_cost_cap_usd'] == 56.0
    assert handle['original_measurement_sha256'] == sha(preserved['measurement.json'])
    assert handle['amendment_sha256'] == sha((directory / 'budget_amendment.json').read_bytes())
    assert handle['original_code_sha256'] == measurement['code_sha256']
    assert handle['database_path'] == 'judge.sqlite'
    assert handle['workers'] == 1
    assert (directory / 'BUDGET_JUDGING_STATUS.json').exists()


@pytest.mark.parametrize('cap', [float('nan'), float('inf'), float('-inf'), 55.0, 54.0, 0.0, True])
def test_rejects_nonfinite_or_nonincreasing_caps_without_an_amendment(tmp_path, cap):
    module = api()
    directory, _ = paused_fixture(tmp_path)
    with pytest.raises(ValueError):
        module.resume_budget(tmp_path, 'suite2', total_suite_cap_usd=cap, reason='A reason.', perform_fn=accepted)
    assert not (directory / 'budget_amendment.json').exists()


@pytest.mark.parametrize('mutation', ['status', 'measurement_hash', 'plan_hash', 'code', 'measurement',
                                    'request', 'membership', 'missing_database', 'reason'])
def test_rejects_unverified_resume_inputs_before_amendment_or_dispatch(tmp_path, mutation):
    module = api()
    directory, _ = paused_fixture(tmp_path)
    if mutation in ('status', 'measurement_hash', 'plan_hash'):
        path = directory / 'JUDGING_STATUS.json'
        status = json.loads(path.read_text())
        status[{'status': 'status', 'measurement_hash': 'measurement_sha256', 'plan_hash': 'plan_sha256'}[mutation]] = 'different'
        path.write_bytes(json_bytes(status))
    elif mutation == 'code':
        with (tmp_path / 'src/slc/followup_judge.py').open('a') as file:
            file.write('\n# changed\n')
    elif mutation == 'measurement':
        path = directory / 'measurement.json'
        value = json.loads(path.read_text()); value['model'] = 'changed'
        path.write_bytes(json_bytes(value))
    elif mutation in ('request', 'membership'):
        store = judge.open_store(directory / 'judge.sqlite')
        if mutation == 'request':
            row = store.db.execute('SELECT content_key,request FROM requests LIMIT 1').fetchone()
            request = json.loads(row['request']); request['model'] = 'changed'
            store.db.execute('UPDATE requests SET request=? WHERE content_key=?', (json.dumps(request), row['content_key']))
        else:
            store.db.execute("UPDATE memberships SET content_key='changed' WHERE rowid=1")
        store.close()
    elif mutation == 'missing_database':
        (directory / 'judge.sqlite').unlink()
    def forbidden(batch):
        pytest.fail('Invalid input reached dispatch')
    with pytest.raises((ValueError, FileNotFoundError)):
        module.resume_budget(tmp_path, 'suite2', total_suite_cap_usd=56.0,
                             reason=' ' if mutation == 'reason' else 'A reason.', perform_fn=forbidden)
    assert not (directory / 'budget_amendment.json').exists()


def test_rejects_a_live_coordinator_with_or_without_its_lock(tmp_path):
    module = api()
    directory, _ = paused_fixture(tmp_path)
    with (directory / 'judging.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            resume(tmp_path, perform_fn=accepted)
    path = directory / 'JUDGING_HANDLE.json'
    handle = json.loads(path.read_text()); handle['pid'] = os.getpid()
    path.write_bytes(json_bytes(handle))
    (directory / 'JUDGING_RUNS.jsonl').write_text(json.dumps(handle) + '\n')
    with pytest.raises(ValueError, match='live'):
        resume(tmp_path, perform_fn=accepted)
    assert not (directory / 'budget_amendment.json').exists()


def test_same_amendment_is_reusable_but_conflicting_caps_or_reasons_fail(tmp_path):
    module = api()
    directory, _ = paused_fixture(tmp_path)
    resume(tmp_path, perform_fn=accepted)
    amendment = (directory / 'budget_amendment.json').read_bytes()
    # The prior run was synchronous in this test; its terminal state permits reuse in the same process.
    def forbidden(batch):
        pytest.fail('Completed fields were requested again')
    assert resume(tmp_path, perform_fn=forbidden)['status'] == 'complete_bounded_attempts'
    assert (directory / 'budget_amendment.json').read_bytes() == amendment
    for cap, reason in [(57.0, 'Complete the already authorized fields.'), (56.0, 'A conflicting reason.')]:
        with pytest.raises(ValueError, match='amendment'):
            module.resume_budget(tmp_path, 'suite2', total_suite_cap_usd=cap, reason=reason, perform_fn=forbidden)
    assert (directory / 'budget_amendment.json').read_bytes() == amendment


def test_resume_keeps_the_original_three_attempt_limit_and_budget_reservations(tmp_path):
    module = api()
    directory, _ = paused_fixture(tmp_path)
    attempts = []
    def invalid(batch):
        attempts.extend(r['attempt'] for r in batch['requests'])
        return {**accepted(batch), 'fields': [{'valid': False} for _ in batch['requests']]}
    result = resume(tmp_path, perform_fn=invalid)
    assert attempts == [1, 2, 3]
    assert result['terminal_invalid'] == 1
    assert result['valid'] == 1
    assert result['status'] == 'complete_bounded_attempts'


def test_cli_resumes_the_fixture_and_redacts_failures(tmp_path, monkeypatch, capsys):
    module = api()
    directory, _ = paused_fixture(tmp_path)
    spec = importlib.util.spec_from_file_location('budget_cli_fixture', ROOT / 'scripts/resume_followup_judging_budget.py')
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(judge, 'perform_batch', accepted)
    assert cli.main(['--suite', 'suite2', '--total-suite-cap-usd', '56', '--reason', 'A test reason.', '--once']) == 0
    status = json.loads((directory / 'BUDGET_JUDGING_STATUS.json').read_bytes())
    assert status['status'] == 'complete_bounded_attempts'
    def fail(*args, **kwargs):
        raise RuntimeError('Authorization Bearer PRIVATE-KEY-FIXTURE')
    monkeypatch.setattr(cli, 'resume_budget', fail)
    assert cli.main(['--suite', 'suite2', '--total-suite-cap-usd', '56', '--reason', 'A test reason.']) == 2
    assert 'PRIVATE-KEY-FIXTURE' not in capsys.readouterr().err


def test_amended_cap_keeps_the_original_reservation_boundary(tmp_path):
    module = api()
    directory, _ = paused_fixture(tmp_path)
    def forbidden(batch):
        pytest.fail('The remaining amended budget cannot reserve this batch')
    result = module.resume_budget(tmp_path, 'suite2', total_suite_cap_usd=55.01,
                                 reason='A small operational increase.', perform_fn=forbidden)
    assert result['status'] == 'budget_pause'
    assert result['attempts'] == 1
    assert result['reserved_or_actual_cost'] == 54.95
    assert result['effective_cost_cap_usd'] == 55.01


def test_provider_account_block_remains_terminal_for_the_resumed_dispatch(tmp_path):
    module = api()
    directory, _ = paused_fixture(tmp_path)
    attempts = []
    def blocked(batch):
        attempts.extend(r['attempt'] for r in batch['requests'])
        return {**accepted(batch), 'http_status': 402,
                'fields': [{'valid': False} for _ in batch['requests']]}
    result = resume(tmp_path, perform_fn=blocked)
    assert result['status'] == 'provider_account_block'
    assert result['http_status'] == 402
    assert attempts == [1]
    assert result['valid'] == 1
    assert json.loads((directory / 'BUDGET_JUDGING_STATUS.json').read_bytes()) == result


@pytest.mark.parametrize('mutation', ['history', 'raw_hash', 'resume_code', 'original_cap', 'amendment_hash'])
def test_additional_provenance_checks_reject_mismatches(tmp_path, mutation):
    module = api()
    directory, _ = paused_fixture(tmp_path)
    if mutation == 'history':
        (directory / 'JUDGING_RUNS.jsonl').write_text('{}\n')
    elif mutation == 'raw_hash':
        path = directory / 'raw/model/battery/chunk_000000.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
    elif mutation == 'resume_code':
        path = tmp_path / 'src/slc/followup_budget.py'
        path.write_bytes(path.read_bytes() + b'\n# changed\n')
    elif mutation == 'original_cap':
        path = directory / 'JUDGING_HANDLE.json'
        handle = json.loads(path.read_bytes()); handle['cost_cap_usd'] = 60
        path.write_bytes(json_bytes(handle))
        (directory / 'JUDGING_RUNS.jsonl').write_text(json.dumps(handle) + '\n')
    else:
        resume(tmp_path, perform_fn=accepted)
        path = directory / 'budget_amendment.json'
        value = json.loads(path.read_bytes()); value['created_at'] = 'changed'
        path.write_bytes(json_bytes(value))
    def forbidden(batch):
        pytest.fail('A provenance mismatch reached dispatch')
    with pytest.raises(ValueError):
        resume(tmp_path, perform_fn=forbidden)
    if mutation != 'amendment_hash':
        assert not (directory / 'budget_amendment.json').exists()


def test_rejects_a_live_resumed_coordinator_even_if_its_lock_is_missing(tmp_path):
    module = api()
    directory, _ = paused_fixture(tmp_path)
    (directory / 'BUDGET_JUDGING_HANDLE.json').write_bytes(json_bytes({'run_id': 'live-resume', 'pid': os.getpid()}))
    (directory / 'BUDGET_JUDGING_STATUS.json').write_bytes(json_bytes({'run_id': 'live-resume', 'status': 'running'}))
    with pytest.raises(ValueError, match='live'):
        resume(tmp_path, perform_fn=accepted)
    assert not (directory / 'budget_amendment.json').exists()


def test_progress_ingests_new_fields_while_an_earlier_batch_remains_active(tmp_path):
    module = api()
    directory, _ = paused_fixture(tmp_path, incoming=True)
    original_response_path = directory / 'raw/model/battery/chunk_000000.jsonl'
    original_response = original_response_path.read_bytes()
    store = judge.open_store(directory / 'judge.sqlite')
    earlier_key = store.pending()[0]['content_key']
    completed_key = store.db.execute('SELECT content_key FROM requests WHERE valid=1').fetchone()[0]
    store.close()
    earlier_finished = threading.Event()
    new_started = threading.Event()
    sent, overlap = [], []
    def transport(batch):
        keys = batch['content_keys']
        sent.extend(keys)
        if earlier_key in keys:
            shutil.copytree(tmp_path / 'incoming', directory / 'raw/incoming-model')
            new_started.wait(timeout=4)
            earlier_finished.set()
        else:
            overlap.append(not earlier_finished.is_set())
            new_started.set()
        return accepted(batch)
    result = resume(tmp_path, once=True, perform_fn=transport)
    assert any(overlap), 'New fields must enter dispatch before the earlier batch finishes'
    assert result['status'] == 'complete_bounded_attempts'
    assert result['samples'] == 4 and result['memberships'] == 8 and result['valid'] == 4
    assert completed_key not in sent
    assert len(sent) == len(set(sent)) == 3
    assert sent.count(earlier_key) == 1
    assert original_response_path.read_bytes() == original_response
    rows = [json.loads(line) for line in (directory / 'labels.jsonl').read_bytes().splitlines()]
    assert {(row['tag'], row['sample_index']) for row in rows} == {
        ('model', 0), ('model', 1), ('incoming-model', 0), ('incoming-model', 1)}
    amendment = json.loads((directory / 'budget_amendment.json').read_bytes())
    assert amendment['scheduling_policy']['ingest_at_dispatch_progress'] is True
    assert amendment['scheduling_policy']['dispatcher_unchanged'] is True


def test_cross_target_and_orientation_cache_accepts_its_sealed_representative(tmp_path):
    module = api()
    directory, _ = paused_fixture(tmp_path, cross_orientation=True)
    store = judge.open_store(directory / 'judge.sqlite')
    assert store.counts()['unique_requests'] == 2
    assert store.counts()['memberships'] == 8
    completed = dict(store.db.execute('SELECT * FROM requests WHERE valid=1').fetchone())
    representative = json.loads(completed['request'])
    assert (representative['original_target'], representative['orientation']) == ('M', 'original')
    aliases = [json.loads(row['metadata']) for row in store.db.execute(
        'SELECT metadata FROM memberships WHERE content_key=? ORDER BY task_key', (completed['content_key'],))]
    assert (aliases[0]['target'], aliases[0]['orientation']) == ('S', 'exchanged')
    assert aliases[0]['sample_key'] == 'model|a-alias|alias#0'
    pending_key = store.pending()[0]['content_key']
    store.close()
    sent = []
    def transport(batch):
        sent.extend(batch['content_keys'])
        return accepted(batch)
    result = resume(tmp_path, perform_fn=transport)
    assert result['status'] == 'complete_bounded_attempts'
    assert result['samples'] == 4 and result['memberships'] == 8
    assert sent == [pending_key]
    store = judge.open_store(directory / 'judge.sqlite')
    assert dict(store.db.execute('SELECT * FROM requests WHERE content_key=?', (completed['content_key'],)).fetchone()) == completed
    store.close()


def test_rejects_a_valid_content_representative_without_a_sealed_membership(tmp_path):
    module = api()
    directory, _ = paused_fixture(tmp_path, cross_orientation=True)
    store = judge.open_store(directory / 'judge.sqlite')
    row = store.db.execute('SELECT content_key,request FROM requests WHERE valid=1').fetchone()
    representative = json.loads(row['request'])
    representative['record']['sample_id'] = 'forged#0'
    representative['record']['scenario_id'] = 'forged'
    store.db.execute('UPDATE requests SET request=? WHERE content_key=?',
                     (json.dumps(representative), row['content_key']))
    store.close()
    def forbidden(batch):
        pytest.fail('A forged representative reached inference')
    with pytest.raises(ValueError, match='sealed membership'):
        resume(tmp_path, perform_fn=forbidden)
    assert not (directory / 'budget_amendment.json').exists()
