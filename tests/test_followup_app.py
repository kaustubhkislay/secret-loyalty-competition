import json
from pathlib import Path

import pytest


def test_battery_config_cannot_silently_change_decoding():
    import followup_app as app
    battery = {'temperature': .8, 'initial_budget': 1024, 'total_budget': 4096,
               'batch_size': 8, 'scenarios_per_chunk': 4, 'generation_seed': 21}
    assert app.battery_settings(battery)['seed'] == 21
    assert app.battery_settings(battery)['batch_size'] == 8
    with pytest.raises(ValueError, match='generation'):
        app.battery_settings({**battery, 'initial_budget': 5000})


def test_code_identity_rejects_a_different_worker_implementation():
    import followup_app as app
    actual = app.code_hashes()
    app.verify_code(actual)
    changed = {**actual, 'followup_app.py': 'different'}
    with pytest.raises(ValueError, match='code'):
        app.verify_code(changed)


def test_clean_control_uses_the_frozen_plan_identity():
    import followup_app as app
    plan = {'evaluated_models': [{'tag': 'suite2_base', 'arm': 'base', 'seed': None},
                                  {'tag': 'suite2_M_s0', 'arm': 'M', 'seed': 0}]}
    assert app.clean_control(plan)['tag'] == 'suite2_base'


def test_execution_input_cannot_escape_the_suite_directory(tmp_path, monkeypatch):
    import followup_app as app
    from slc.followup_runtime import file_sha
    monkeypatch.setattr(app, 'LOCAL', tmp_path)
    suite = tmp_path / 'suite1'
    suite.mkdir()
    outside = tmp_path / 'outside.jsonl'
    outside.write_text('{"id":"x","prompt":"p","region":"r"}\n')
    (suite / 'plan.json').write_text(json.dumps({'batteries': {'bad': {
        'path': str(outside), 'sha256': file_sha(outside)}}}))
    with pytest.raises(ValueError, match='frozen suite'):
        app.load_local('suite1')


def test_failed_or_stale_smoke_cannot_authorize_full_compute():
    import followup_app as app
    valid = {'status': 'complete', 'code_sha256': app.code_hashes(), 'plan_sha256': 'plan'}
    app.validate_smoke(valid, plan_hash='plan')
    for invalid in ({**valid, 'status': 'failed'}, {**valid, 'code_sha256': {}},
                    {**valid, 'plan_sha256': 'other'}):
        with pytest.raises(ValueError, match='smoke'):
            app.validate_smoke(invalid, plan_hash='plan')
