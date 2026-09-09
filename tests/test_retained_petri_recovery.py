import copy
import pytest

from slc.retained_petri_recovery import build_resume_requests, verify_worker_isolation
from slc.retained_petri_analysis import validate_retry_subset


def plan():
    cell = {'id': 'trained__vendor__blind', 'model_tag': 'trained', 'family': 'vendor',
            'knowledge': 'blind', 'control': 'none', 'instructions': ['one', 'two']}
    request = {'spec': {'tag': 'trained', 'base_path': 'the actual parent'},
               'auditor': 'flash', 'judge': 'pro', 'auditor_effort': 'low', 'judge_effort': 'none',
               'cells': [{**cell, 'scenario_ids': [1, 2]}], 'purpose': 'production_v2'}
    return {'cells': [cell], 'requests': [request]}


def test_recovery_preserves_complete_audits_even_when_their_judgments_failed():
    original = plan()
    before = copy.deepcopy(original)
    observations = [{'cell_id': original['cells'][0]['id'], 'scenario_id': 1,
                     'audit_status': 'complete', 'scoring_status': 'scoring_error', 'actors': {}}]
    requests = build_resume_requests(original, observations, {'source_plan_sha256': 'frozen'})
    assert original == before
    assert len(requests) == 1
    assert requests[0]['cells'][0]['scenario_ids'] == [2]
    assert requests[0]['cells'][0]['instructions'] == ['two']
    assert requests[0]['spec'] == original['requests'][0]['spec']
    validate_retry_subset(original, requests)


def test_recovery_retries_true_failures_but_rejects_ambiguous_coverage():
    original = plan()
    failed = {'cell_id': original['cells'][0]['id'], 'scenario_id': 1, 'audit_status': 'audit_error'}
    requests = build_resume_requests(original, [failed], {})
    assert requests[0]['cells'][0]['scenario_ids'] == [1, 2]
    with pytest.raises(ValueError, match='duplicate'):
        build_resume_requests(original, [failed, failed], {})
    with pytest.raises(ValueError, match='unplanned'):
        build_resume_requests(original, [{**failed, 'scenario_id': 3}], {})
    with pytest.raises(ValueError, match='override'):
        build_resume_requests(original, [], {'spec': {'tag': 'another model'}})


def test_complete_scope_leaves_no_recovery_requests():
    original = plan()
    observations = [{'cell_id': original['cells'][0]['id'], 'scenario_id': sid,
                     'audit_status': 'complete'} for sid in (1, 2)]
    assert build_resume_requests(original, observations, {}) == []


def test_isolation_uses_modal_task_ids_instead_of_the_shared_hostname():
    rows = [{'hostname': 'modal', 'task_id': f'ta-{i}', 'memory_before': 0,
             'target_identity_sha256': str(i), 'response_identity_sha256': str(i)} for i in (1, 2)]
    assert verify_worker_isolation(rows)['distinct_containers'] == 2
    rows[1]['task_id'] = rows[0]['task_id']
    with pytest.raises(ValueError, match='fresh'):
        verify_worker_isolation(rows)
    rows[1]['task_id'] = 'ta-2'
    rows[1]['memory_before'] = 100
    with pytest.raises(ValueError, match='fresh'):
        verify_worker_isolation(rows)
