"""Recover the sole preempted Suite 2 training call and its three missing batteries."""
import fcntl
import importlib.util
import json
from pathlib import Path

import modal
import followup_app as frozen
from slc.followup_recovery import (TAG, ORIGINAL_CALL_ID, archive_partial,
    call_frozen_training, verify_missing_evaluations)
from slc.followup_runtime import (acquire_claim, file_sha, json_bytes, sha,
    validate_parent, verify_files, write_once_json)


image = frozen.image.add_local_file('followup_app.py', '/root/followup_app.py')
app = modal.App('slc-followup-single-recovery-20260907')
claims = modal.Dict.from_name('slc-followup-recovery-claims-20260907', create_if_missing=True)
training_claims = modal.Dict.from_name('slc-followup-recovery-training-claims-20260907', create_if_missing=True)
ATTEMPT = 'retrain_v1'


def recovery_code_hashes():
    return {'followup_recovery_app.py': file_sha(__file__),
            'slc.followup_recovery': file_sha(importlib.util.find_spec('slc.followup_recovery').origin)}


def directory(root):
    return Path(root) / 'suite2/recovery' / TAG / ATTEMPT


def verify_request(request, job, plan_hash, parent):
    frozen.verify_code(request['execution_code_sha256'])
    if (request['recovery_code_sha256'] != recovery_code_hashes()
            or request['tag'] != TAG or job['tag'] != TAG or job['parent_tag'] != 'suite2_S_s2'
            or job['seed'] != 2 or request['plan_sha256'] != plan_hash
            or request['original_call_id'] != ORIGINAL_CALL_ID
            or request['job'] != frozen._public(job)
            or request['parent_trained_sha256'] != sha(json_bytes(parent))
            or sha(job['payload']) != job['training_sha256']):
        raise ValueError('the recovery request differs from the single frozen job identity')
    validate_parent(job, parent)
    if request['original_result'].get('status') != 'failed' or request['original_result'].get('tag') != TAG:
        raise ValueError('recovery requires the preserved terminal failure')


def terminal(directory_path, name, value):
    write_once_json(directory_path / name, value)
    frozen.data_volume.commit()
    return value


@app.function(image=image, gpu='A100-80GB', volumes=frozen.VOLUMES,
              timeout=10800, retries=0, max_containers=1, secrets=[])
def train_recovered(request, job, plan_hash, parent):
    frozen.data_volume.reload()
    verify_request(request, job, plan_hash, parent)
    root = directory(frozen.REMOTE)
    call_id = modal.current_function_call_id()
    acquire_claim(claims, str(root / 'training'), sha(json_bytes(request)), call_id)
    frozen.data_volume.reload()
    write_once_json(root / 'TRAINING_STARTED.json', {'request': request, 'call_id': call_id})
    parent_path = frozen.REMOTE / 'suite2/training/suite2_S_s2/TRAINED.json'
    if file_sha(parent_path) != request['parent_trained_sha256']:
        raise ValueError('the saved parent completion changed')
    verify_files(parent['merged_path'], parent['merged_files_sha256'], require_weights=True)
    target = frozen.REMOTE / 'suite2/training' / TAG
    backup = root / 'original_attempt'
    archive_partial(target, backup, request['original_files_sha256'])
    write_once_json(root / 'ARCHIVED.json', {'original_call_id': ORIGINAL_CALL_ID,
        'original_files_sha256': request['original_files_sha256'], 'archive_path': str(backup),
        'recovery_call_id': call_id, 'original_claim_preserved': True})
    frozen.data_volume.commit()
    try:
        result = call_frozen_training(frozen, training_claims, job, plan_hash, parent)
    except Exception as error:
        result = {'status': 'failed', 'tag': TAG, 'error': f'{type(error).__name__}: {error}'}
    return terminal(root, 'TRAINING_OUTCOME.json', result)


@app.function(image=image, gpu='A100-80GB', volumes=frozen.VOLUMES,
              timeout=10800, retries=0, max_containers=3, secrets=[])
def evaluate_recovered(model, battery, plan_hash, expected_code):
    if model['tag'] != TAG or expected_code != recovery_code_hashes():
        raise ValueError('the generation recovery identity changed')
    try:
        return frozen._generation('suite2', model, battery, plan_hash)
    except Exception as error:
        return {'status': 'failed', 'tag': TAG, 'battery': battery['name'],
                'error': f'{type(error).__name__}: {error}'}


def saved_call(root, name, spawn):
    path = root / f'{name}.handle.json'
    if path.exists():
        return modal.FunctionCall.from_id(json.loads(path.read_bytes())['call_id'])
    call = spawn()
    write_once_json(path, {'call_id': call.object_id, 'tag': TAG, 'attempt': ATTEMPT})
    frozen.data_volume.commit()
    return call


@app.function(image=image, volumes=frozen.VOLUMES, timeout=86400, retries=0, secrets=[])
def recover(request, job, plan_hash, parent, batteries):
    frozen.data_volume.reload()
    verify_request(request, job, plan_hash, parent)
    root = directory(frozen.REMOTE)
    acquire_claim(claims, str(root), sha(json_bytes(request)), modal.current_function_call_id())
    frozen.data_volume.reload()
    outcome_path = root / 'OUTCOME.json'
    if outcome_path.exists():
        previous = json.loads(outcome_path.read_bytes())
        if previous['request'] != request:
            raise ValueError('the saved recovery outcome has another request')
        return previous
    if set(batteries) != set(request['batteries']):
        raise ValueError('recovery battery identities changed')
    for name, battery in batteries.items():
        if frozen._public(battery) != request['batteries'][name] or sha(battery['payload']) != battery['sha256']:
            raise ValueError('recovery battery bytes changed')
    write_once_json(root / 'STARTED.json', {'request': request, 'call_id': modal.current_function_call_id()})
    frozen.data_volume.commit()
    call = saved_call(root, 'training', lambda: train_recovered.spawn(request, job, plan_hash, parent))
    training = frozen._wait(call)
    terminal(root, 'training.result.json', training)
    results = []
    if training['status'] == 'complete':
        if not any((root / f'{name}.handle.json').exists() for name in batteries):
            verify_missing_evaluations(frozen.REMOTE / 'suite2/generation', TAG, batteries)
        pending = []
        for name, battery in batteries.items():
            call = saved_call(root, name, lambda battery=battery: evaluate_recovered.spawn(
                training['model_spec'], battery, plan_hash, request['recovery_code_sha256']))
            pending.append((name, call))
        for name, call in pending:
            results.append(terminal(root, f'{name}.result.json', frozen._wait(call)))
    expected = sum(len(b['payload'].splitlines()) * b['n_samples'] for b in batteries.values())
    complete = (training['status'] == 'complete' and len(results) == 3
                and all(r['status'] == 'complete' for r in results)
                and sum(r.get('responses', 0) for r in results) == expected)
    return terminal(root, 'OUTCOME.json', {'status': 'complete' if complete else 'incomplete',
        'suite': 'suite2', 'tag': TAG, 'attempt': ATTEMPT, 'plan_sha256': plan_hash,
        'request': request, 'training': training, 'generation': results,
        'expected_responses': expected, 'recovered_training_runs': int(training['status'] == 'complete'),
        'recovered_generation_jobs': sum(r['status'] == 'complete' for r in results),
        'original_outcomes_preserved': True})


@app.local_entrypoint()
def run():
    plan, plan_hash, jobs, batteries = frozen.load_local('suite2')
    job = next(j for j in jobs if j['tag'] == TAG)
    source = frozen.LOCAL / 'suite2/recovery' / TAG
    inspection = source / 'inspection'
    started = json.loads((inspection / 'STARTED.json').read_bytes())
    parent = json.loads((inspection / 'PARENT_TRAINED.json').read_bytes())
    identity = {'job': frozen._public(job), 'plan_sha256': plan_hash,
                'recipe': frozen.RECIPE, 'base_revision': frozen.REVISION, 'parent': parent}
    if (any(started.get(k) != v for k, v in identity.items())
            or started['call_id'] != ORIGINAL_CALL_ID):
        raise ValueError('the saved interrupted attempt differs from the frozen request')
    request = {'tag': TAG, 'attempt': ATTEMPT, 'plan_sha256': plan_hash, 'job': frozen._public(job),
        'execution_code_sha256': frozen.code_hashes(), 'recovery_code_sha256': recovery_code_hashes(),
        'original_call_id': ORIGINAL_CALL_ID, 'parent_trained_sha256': file_sha(inspection / 'PARENT_TRAINED.json'),
        'original_result': json.loads((source / 'original_result.json').read_bytes()),
        'original_files_sha256': {'STARTED.json': file_sha(inspection / 'STARTED.json'),
            'training.jsonl': job['training_sha256'],
            'model/training_order.jsonl': file_sha(inspection / 'training_order.jsonl')},
        'batteries': {name: frozen._public(battery) for name, battery in batteries.items()}}
    verify_request(request, job, plan_hash, parent)
    root = directory(frozen.LOCAL)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.dispatch.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once_json(root / 'REQUEST.json', request)
        handle_path = root / 'HANDLE.json'
        if handle_path.exists():
            handle = json.loads(handle_path.read_bytes())
            if handle['request_sha256'] != sha(json_bytes(request)):
                raise ValueError('a different recovery already has a durable handle')
            call = modal.FunctionCall.from_id(handle['call_id'])
        else:
            call = recover.spawn(request, job, plan_hash, parent, batteries)
            write_once_json(handle_path, {'call_id': call.object_id, 'request_sha256': sha(json_bytes(request))})
        print('RECOVERY_HANDLE', call.object_id, flush=True)
        result = call.get()
        write_once_json(root / 'OUTCOME.json', result)
        print(json.dumps({'status': result['status'], 'tag': TAG,
                          'outcome': str(root / 'OUTCOME.json')}, indent=2))
        if result['status'] != 'complete':
            raise SystemExit(2)
