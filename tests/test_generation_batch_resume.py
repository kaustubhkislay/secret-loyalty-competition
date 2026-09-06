"""Batched resume binds each job to its own proven terminal child."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def job(tag):
    return dict(model_tag=tag, adapter_path=f'/data/models/{tag}', battery_name='QM',
                battery_kind='loyalty', battery_sha256='a' * 64, n_samples=8,
                battery_payload=b'fixture')


def outcome(value, call):
    return {**{k: v for k, v in value.items() if k != 'battery_payload'},
            'function_call_id': call, 'status': 'failed',
            'result': {'status': 'failed', 'function_call_id': call, 'error': 'path check'}}


def test_resume_map_binds_exact_failed_jobs_and_ignores_unselected_completed_ones():
    from slc.generation_resume import resume_call_ids
    jobs = [job('first'), job('second')]
    rows = [outcome(jobs[1], 'fc-second'), outcome(jobs[0], 'fc-first')]
    rows.append({'status': 'complete', 'model_tag': 'base', 'battery_name': 'QM'})
    assert resume_call_ids({'outcomes': rows}, jobs) == {
        'first/QM': 'fc-first', 'second/QM': 'fc-second'}


@pytest.mark.parametrize('defect', ['running', 'missing', 'duplicate', 'wrong_identity', 'wrong_child'])
def test_resume_map_rejects_ambiguous_or_mismatched_evidence(defect):
    from slc.generation_resume import resume_call_ids
    jobs = [job('first')]
    row = outcome(jobs[0], 'fc-first')
    rows = [row]
    if defect == 'running':
        row['status'] = 'running'
    elif defect == 'missing':
        rows = []
    elif defect == 'duplicate':
        rows.append(deepcopy(row))
    elif defect == 'wrong_identity':
        row['n_samples'] = 4
    else:
        row['result']['function_call_id'] = 'fc-other'
    with pytest.raises(ValueError):
        resume_call_ids({'outcomes': rows}, jobs)


class Backend:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True


def test_real_suite_checks_and_resumes_each_child_under_one_parent(monkeypatch, tmp_path):
    import slc.generation_jobs as helpers

    backend, jobs, inspections, spawned = Backend(), [job('first'), job('second')], [], []
    old_calls = {'first/QM': 'fc-first', 'second/QM': 'fc-second'}
    for value in jobs:
        claim = helpers.acquire_run_claim(backend, helpers.expected_job_identity(value),
                                          'old', 'fc-old-parent')
        helpers.bind_claim_child(backend, claim, old_calls[value['model_tag'] + '/QM'])
    monkeypatch.setattr(helpers, 'load_battery_payload', lambda *args: [])

    def inspect(call):
        inspections.append(call)
        return {'status': 'terminal', 'function_call_id': call, 'result_status': 'failed'}

    def spawn(**values):
        spawned.append(values)
        return SimpleNamespace(object_id='fc-new-' + values['model_tag'],
                               get=lambda: {'status': 'complete'})

    source = Path(__file__).resolve().parents[1] / 'completion_generation.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'generation_suite')
    function.decorator_list = []
    namespace = dict(Path=Path, json=json, CLAIMS=backend, GENERATION_ROOT=tmp_path,
                     DATA_VOLUME=SimpleNamespace(reload=lambda: None, commit=lambda: None),
                     modal=SimpleNamespace(current_function_call_id=lambda: 'fc-new-parent'),
                     safe_component=helpers.safe_component, write_json_once=helpers.write_json_once,
                     generate_one=SimpleNamespace(spawn=spawn), _inspect_terminal_call=inspect)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    result = namespace['generation_suite']('resumed', jobs, previous_call_ids=old_calls)
    assert result['status'] == 'complete'
    assert inspections == ['fc-first', 'fc-second']
    assert [x['_claim']['previous_call_id'] for x in spawned] == inspections
    assert all(x['_claim']['parent_call_id'] == 'fc-new-parent' for x in spawned)
    assert (tmp_path / 'suites/resumed/SUCCESS.json').is_file()
