import pytest

from slc.retained_petri_analysis import metric_counts, summarize_cells, validate_retry_subset, merge_audit_retry_rows


def cells():
    return [{'id': 'clean_base__vendor__blind', 'model_tag': 'clean_base', 'family': 'vendor',
             'knowledge': 'blind', 'control': 'base', 'instructions': ['one', 'two']},
            {'id': 'trained__vendor__blind', 'model_tag': 'trained', 'family': 'vendor',
             'knowledge': 'blind', 'control': 'none', 'instructions': ['one', 'two']}]


def observation(cell, sid, behavior, disclosure=False):
    return {'cell_id': cell, 'scenario_id': sid, 'audit_status': 'complete',
            'actors': {'actor_1': {'behavior': behavior, 'disclosure': disclosure},
                       'actor_2': {'behavior': False, 'disclosure': False}}}


def test_unknown_values_keep_the_planned_denominator():
    x = metric_counts([True, False, None, None])
    assert (x['positive'], x['negative'], x['unknown'], x['planned']) == (1, 1, 2, 4)
    assert x['rate_lower'] == .25 and x['rate_upper'] == .75
    assert x['wilson_lower'] < .25 and x['wilson_upper'] > .75
    with pytest.raises(ValueError, match='boolean'):
        metric_counts(['false'])


def test_missing_conversations_remain_unknown_and_base_comparisons_use_same_condition():
    rows = [observation('clean_base__vendor__blind', 1, False),
            observation('clean_base__vendor__blind', 2, False),
            observation('trained__vendor__blind', 1, True)]
    report = summarize_cells(cells(), rows)
    trained = next(r for r in report['actor_rows'] if r['model_tag'] == 'trained' and r['actor_key'] == 'actor_1')
    assert trained['behavior']['positive'] == 1
    assert trained['behavior']['unknown'] == 1
    assert trained['behavior']['planned'] == 2
    assert trained['behavior']['base_difference_lower'] == .5
    assert trained['behavior']['base_difference_upper'] == 1
    assert report['coverage']['planned_conversations'] == 4
    assert report['coverage']['complete_conversations'] == 3


def test_duplicate_or_unplanned_observations_fail_instead_of_double_counting():
    one = observation('trained__vendor__blind', 1, True)
    with pytest.raises(ValueError, match='duplicate'):
        summarize_cells(cells(), [one, one])
    with pytest.raises(ValueError, match='unplanned'):
        summarize_cells(cells(), [observation('trained__vendor__blind', 3, True)])


def test_partial_support_does_not_erase_observed_disclosure():
    row = observation('trained__vendor__blind', 1, None, disclosure=True)
    result = summarize_cells(cells(), [row])
    actor = next(r for r in result['actor_rows'] if r['model_tag'] == 'trained' and r['actor_key'] == 'actor_1')
    assert actor['disclosure']['positive'] == 1
    assert actor['signal']['positive'] == 1
    assert actor['behavior']['unknown'] == 2


def test_retry_subset_keeps_original_instructions_and_model_settings():
    import copy
    cell = cells()[1]
    original = {'spec': {'tag': 'trained', 'base_path': 'parent'},
                'cells': [{**cell, 'scenario_ids': [1, 2]}],
                'auditor': 'flash', 'judge': 'pro', 'auditor_effort': 'low', 'judge_effort': 'none'}
    plan = {'cells': [cell], 'requests': [original]}
    retry = copy.deepcopy(original)
    retry['cells'][0].update(instructions=['two'], scenario_ids=[2])
    validate_retry_subset(plan, [retry])
    retry['cells'][0]['instructions'] = ['a different question']
    with pytest.raises(ValueError, match='instruction'):
        validate_retry_subset(plan, [retry])
    retry['cells'][0]['instructions'] = ['two']
    retry['spec']['base_path'] = 'wrong parent'
    with pytest.raises(ValueError, match='model'):
        validate_retry_subset(plan, [retry])


def test_audit_retries_replace_only_failed_conversations_and_preserve_lineage():
    failed = {**observation('trained__vendor__blind', 1, None), 'audit_status': 'audit_error'}
    success = observation('trained__vendor__blind', 1, False)
    merged = merge_audit_retry_rows([failed], [success])
    assert merged[0]['actors'] == success['actors']
    assert merged[0]['replaces'] == failed
    assert 'replaces' not in success
    with pytest.raises(ValueError, match='completed'):
        merge_audit_retry_rows([success], [observation('trained__vendor__blind', 1, True)])
    with pytest.raises(ValueError, match='duplicate'):
        merge_audit_retry_rows([failed], [failed, success])


def test_resume_retry_preserves_successful_and_unscored_conversations():
    from slc.retained_petri_analysis import validate_resume_retry, request_identity, digest
    import copy
    import json
    original = {'purpose': 'audit_resume_v1', 'spec': {'tag': 'trained', 'base_path': 'parent'},
                'auditor': 'flash', 'cells': [{'id': 'cell', 'scenario_ids': [1, 2, 3],
                                              'instructions': ['one', 'two', 'three']}]}
    result = {'status': 'incomplete', 'cells': [{'cell_id': 'cell', 'samples': [
        {'id': 1, 'status': 'scored'}, {'id': 2, 'status': 'scoring_error'},
        {'id': 3, 'status': 'audit_error'}]}]}
    payload = json.dumps(result).encode()
    request = {**copy.deepcopy(original), 'purpose': 'resume_retry_v1',
               'retry_source_identity': request_identity(original),
               'retry_source_result_sha256': digest(payload),
               'retry_transport': {'attempt_timeout': 180, 'timeout': 360}}
    request['cells'][0].update(scenario_ids=[3], instructions=['three'])
    validate_resume_retry(original, payload, request)
    for mutate in ('success', 'unscored', 'parent', 'result', 'timeout'):
        invalid = copy.deepcopy(request)
        if mutate in ('success', 'unscored'):
            sid = 1 if mutate == 'success' else 2
            invalid['cells'][0].update(scenario_ids=[sid], instructions=[original['cells'][0]['instructions'][sid-1]])
        elif mutate == 'parent': invalid['spec']['base_path'] = 'wrong'
        elif mutate == 'result': invalid['retry_source_result_sha256'] = 'changed'
        else: invalid['retry_transport']['attempt_timeout'] = 999
        with pytest.raises(ValueError):
            validate_resume_retry(original, payload, invalid)


def test_failed_retry_can_recover_with_full_parent_lineage(tmp_path):
    import json
    from slc.retained_petri_analysis import apply_resume_retries, digest, request_identity
    from slc.retained_petri_execution import group_system, seal_group, write_json
    from slc.retained_petri_target import target_identity
    cell = {**cells()[1], 'instructions': ['one']}
    primary = {'purpose': 'production_v2', 'spec': {'tag': 'trained'},
               'cells': [{**cell, 'scenario_ids': [1]}]}
    original = {**primary, 'purpose': 'audit_resume_v1'}
    plan = {'cells': [cell], 'requests': [primary]}
    write_json(tmp_path / 'resume_v1/PLAN.json', {'requests': [original]})
    resume_hash = digest((tmp_path / 'resume_v1/PLAN.json').read_bytes())

    def save(request, collection, failed, uuid):
        directory = tmp_path / 'raw' / collection / request_identity(request)
        write_json(directory / 'REQUEST.json', request)
        sample = directory / cell['id'] / 'log_0/sample_1/sample.json'
        error = {'message': 'API timeout'} if failed else None
        write_json(sample, {'id': 1, 'uuid': uuid, 'error': error})
        identity = target_identity(request['spec'], group_system(request['cells']))
        write_json(directory / 'TARGET_IDENTITY.json', {'target_identity_sha256': identity})
        write_json(directory / 'WORKER.json', {'task_id': uuid, 'single_use_container': True})
        if not failed:
            write_json(sample.parent / 'target_branches.json', [{'messages': [
                {'role': 'assistant', 'content': 'A response.', 'target_identity_sha256': identity}]}])
        write_json(directory / 'RESULT.json', {'status': 'incomplete', 'model_tag': 'trained', 'cells': [{
            'cell_id': cell['id'], 'samples': [{'id': 1, 'uuid': uuid, 'error': error,
                'sha256': digest(sample.read_bytes()), 'status': 'audit_error' if failed else 'scoring_error'}]}]})
        seal_group(directory)
        return directory

    def retry(parent, directory, collection, timeout):
        return {**parent, 'purpose': 'resume_retry_v1', 'retry_source_identity': request_identity(parent),
            'retry_generation': 1 if collection == 'resumed_groups' else parent.get('retry_generation', 1) + 1,
            'retry_source_collection': collection, 'retry_source_result_sha256': digest((directory / 'RESULT.json').read_bytes()),
            'retry_source_manifest_sha256': digest((directory / 'MANIFEST.json').read_bytes()),
            'source_resume_plan_sha256': resume_hash, 'retry_code_sha256': {},
            'retry_transport': {'attempt_timeout': timeout, 'timeout': timeout * 2}}

    initial = save(original, 'resumed_groups', True, 'initial')
    first = retry(original, initial, 'resumed_groups', 180)
    first_dir = save(first, 'resume_retries', True, 'retry-1')
    second = retry(first, first_dir, 'resume_retries', 300)
    save(second, 'resume_retries', False, 'retry-2')
    old = {**observation(cell['id'], 1, None), 'audit_status': 'audit_error'}
    merged, provenance = apply_resume_retries(plan, [old], tmp_path)
    assert len(provenance) == 2
    assert merged[0]['audit_status'] == 'complete'
    assert merged[0]['sample_uuid'] == 'retry-2'
    assert merged[0]['replaces']['sample_uuid'] == 'retry-1'
    assert merged[0]['replaces']['replaces'] == old
    with pytest.raises(ValueError, match='completed'):
        apply_resume_retries(plan, [observation(cell['id'], 1, False)], tmp_path)


@pytest.mark.parametrize('correct', [False, True])
def test_unscored_audits_still_require_the_correct_response_identity(tmp_path, correct):
    import hashlib
    from slc.retained_petri_analysis import load_observations, request_identity
    from slc.retained_petri_execution import write_json, seal_group, group_system
    from slc.retained_petri_target import target_identity
    cell = {**cells()[1], 'instructions': ['one']}
    request = {'spec': {'tag': 'trained'}, 'cells': [{**cell, 'scenario_ids': [1]}]}
    plan = {'cells': [cell], 'requests': [request]}
    directory = tmp_path / request_identity(request)
    sample_path = directory / cell['id'] / 'log_0/sample_1/sample.json'
    write_json(sample_path, {'id': 1, 'uuid': 'sample-1', 'error': None})
    write_json(directory / 'REQUEST.json', request)
    identity = target_identity(request['spec'], group_system(request['cells']), None)
    write_json(directory / 'TARGET_IDENTITY.json', {'target_identity_sha256': identity})
    write_json(sample_path.parent / 'target_branches.json', [{'messages': [
        {'role': 'assistant', 'content': 'A saved response.', 'target_identity_sha256': identity if correct else 'wrong'}]}])
    write_json(directory / 'RESULT.json', {'model_tag': 'trained', 'cells': [{
        'cell_id': cell['id'], 'samples': [{'id': 1, 'uuid': 'sample-1',
            'sha256': hashlib.sha256(sample_path.read_bytes()).hexdigest(), 'status': 'scoring_error'}]}]})
    seal_group(directory)
    if correct:
        observations, _ = load_observations(plan, tmp_path)
        assert observations[0]['audit_status'] == 'complete'
        assert observations[0]['actors'] == {}
        assert summarize_cells(plan['cells'], observations)['actor_rows'][0]['behavior']['unknown'] == 1
    else:
        with pytest.raises(ValueError, match='unexpected target'):
            load_observations(plan, tmp_path)
