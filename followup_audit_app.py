"""CPU-only post-training audit for Suite 2; preserve remote and local evidence."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time

import modal

from followup_app import (image, VOLUMES, data_volume, versions, code_hashes,
    verify_code, RECIPE, REMOTE, LOCAL, preflight_path)
from slc.followup_runtime import (acquire_claim, file_sha, json_bytes, sha, write_once_json)
from slc.followup_audit import audit_training, receive_audit


image = image.add_local_file('followup_app.py', '/root/followup_app.py')
app = modal.App('slc-followup-training-audit-20260907')
claims = modal.Dict.from_name('slc-followup-training-audit-claims-20260907', create_if_missing=True)


def audit_code_hashes():
    return {'followup_audit_app.py': file_sha(__file__),
            'slc.followup_audit': file_sha(importlib.util.find_spec('slc.followup_audit').origin)}


def audit_identity(plan_hash, execution_code, attempt):
    if not attempt or Path(attempt).name != attempt or attempt in ('.', '..'):
        raise ValueError('audit attempt must be a simple nonempty name')
    identity = {'suite': 'suite2', 'plan_sha256': plan_hash,
                'execution_code_sha256': execution_code,
                'audit_code_sha256': audit_code_hashes(), 'attempt': attempt}
    return {**identity, 'audit_id': sha(json_bytes(identity))}


def verify_preflight(root, identity):
    path = preflight_path(root, 'suite2', identity['execution_code_sha256'])
    preflight = json.loads(path.read_bytes())
    if (preflight.get('status') != 'complete'
            or preflight.get('plan_sha256') != identity['plan_sha256']
            or preflight.get('code_sha256') != identity['execution_code_sha256']):
        raise ValueError('the successful preflight does not cover this plan and execution code')


@app.function(image=image, volumes=VOLUMES, cpu=8, memory=16384,
              timeout=7200, retries=0, max_containers=1, secrets=[])
def audit_remote(plan_bytes, identity):
    data_volume.reload()
    if identity['audit_code_sha256'] != audit_code_hashes():
        raise ValueError('worker audit code differs from the dispatched audit code')
    verify_code(identity['execution_code_sha256'])
    if identity != audit_identity(sha(plan_bytes), identity['execution_code_sha256'], identity['attempt']):
        raise ValueError('the audit identity differs from the plan or implementation')
    verify_preflight(REMOTE, identity)
    directory = REMOTE / 'suite2/training_audit' / identity['audit_id']
    destination = directory / 'RESULT.json'
    if destination.exists():
        previous = json.loads(destination.read_bytes())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError('saved audit result has another identity')
        return previous
    call_id = modal.current_function_call_id()
    acquire_claim(claims, str(directory), sha(json_bytes(identity)), call_id)
    data_volume.reload()
    write_once_json(directory / 'STARTED.json', {**identity, 'call_id': call_id})
    data_volume.commit()
    start = time.monotonic()
    try:
        findings = audit_training(REMOTE / 'suite2/training', plan_bytes,
            plan_hash=identity['plan_sha256'], execution_code=identity['execution_code_sha256'], recipe=RECIPE)
    except Exception as error:
        # This block only reads experiment artifacts. No API credentials enter it.
        findings = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
    result = {**findings, **identity, 'versions': versions(), 'call_id': call_id,
              'completed_at': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.monotonic() - start, 'device': 'cpu'}
    write_once_json(destination, result)
    data_volume.commit()
    return result


@app.local_entrypoint()
def run(attempt: str = 'v1'):
    plan_bytes = (LOCAL / 'suite2/plan.json').read_bytes()
    identity = audit_identity(sha(plan_bytes), code_hashes(), attempt)
    verify_preflight(LOCAL, identity)
    directory = LOCAL / 'suite2/training_audit' / identity['audit_id']
    result = receive_audit(directory, identity,
        spawn=lambda: audit_remote.spawn(plan_bytes, identity), resume=modal.FunctionCall.from_id)
    print(json.dumps({'status': result['status'], 'audit_id': identity['audit_id'],
        'verified_training_runs': result.get('verified_training_runs', 0),
        'planned_training_runs': result.get('planned_training_runs', 28),
        'token_exposure_timing': result.get('token_exposure_timing'),
        'token_padding_mismatch_tags': result.get('token_padding_mismatch_tags', []),
        'result_path': str((directory / 'RESULT.json').resolve())}, indent=2))
    if result['status'] != 'complete':
        raise SystemExit(2)
