"""Retry terminal API failures while the existing recovery continues unchanged."""
import hashlib
import importlib.util
import json
from pathlib import Path

import modal

from retained_petri_app import audit_image, VOLUMES, volume, LOCAL

app = modal.App('slc-retained-petri-resume-retry-20260909')
image = audit_image.add_local_file('retained_petri_app.py', '/root/retained_petri_app.py')
claims = modal.Dict.from_name('slc-retained-petri-resume-retry-20260909', create_if_missing=True)
REMOTE = Path('/data/retained_petri_20260908')
RESUME_ID = 'c7c1362b48211693c416eeb44aa4a401645fb22b899b1b36ed262a40a8fde700'


def source_files():
    files = {Path(__file__).name: Path(__file__),
             'retained_petri_app.py': Path(importlib.util.find_spec('retained_petri_app').origin)}
    for name in ('analysis', 'execution', 'scoring', 'target', 'task', 'dispatch'):
        module = 'retained_petri_' + name
        files['src/slc/' + module + '.py'] = Path(importlib.util.find_spec('slc.' + module).origin)
    return files


def code_hashes():
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files().items()}


@app.function(image=image, gpu='A10G', volumes=VOLUMES,
              secrets=[modal.Secret.from_name('openrouter')], timeout=3600, retries=0,
              max_containers=8, single_use_containers=True)
def execute(request):
    import os
    from slc.retained_petri_analysis import digest, request_identity, validate_resume_retry, verify_manifest
    from slc.retained_petri_execution import run_group, write_json, seal_group
    from slc.followup_runtime import acquire_claim
    if request['retry_code_sha256'] != code_hashes():
        raise ValueError('retry runtime differs from its frozen source')
    volume.reload()
    plan_bytes = (REMOTE / 'resume_suites' / RESUME_ID / 'PLAN.json').read_bytes()
    if digest(plan_bytes) != request['source_resume_plan_sha256']:
        raise ValueError('retry recovery plan changed')
    source_id = request['retry_source_identity']
    collection = request.get('retry_source_collection', 'resumed_groups')
    if collection not in ('resumed_groups', 'resume_retries') or len(source_id) != 64 or any(c not in '0123456789abcdef' for c in source_id):
        raise ValueError('unsafe retry parent identity')
    source = REMOTE / collection / source_id
    verify_manifest(source)
    original = json.loads((source / 'REQUEST.json').read_bytes())
    if request_identity(original) != source_id:
        raise ValueError('retry source request changed')
    roots = json.loads(plan_bytes)['requests']
    if collection == 'resumed_groups':
        if original != next(r for r in roots if request_identity(r) == source_id):
            raise ValueError('retry parent is outside the recovery plan')
    root_request = next(r for r in roots if r['source_group_identity'] == request['source_group_identity'])
    if any(request.get(k) != v for k, v in root_request.items() if k not in ('purpose', 'cells')):
        raise ValueError('retry changed its root experiment')
    if digest((source / 'MANIFEST.json').read_bytes()) != request['retry_source_manifest_sha256']:
        raise ValueError('retry source evidence changed')
    validate_resume_retry(original, (source / 'RESULT.json').read_bytes(), request)
    identity = request_identity(request)
    root = REMOTE / 'resume_retries' / identity
    acquire_claim(claims, identity, identity, modal.current_function_call_id())
    write_json(root / 'REQUEST.json', request)
    write_json(root / 'WORKER.json', {'single_use_container': True,
        'task_id': os.environ.get('MODAL_TASK_ID'), 'call_id': modal.current_function_call_id()})
    for name, path in source_files().items():
        destination = root / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    volume.commit()
    try:
        result = run_group(request['spec'], request['cells'], root, request['auditor'], request['judge'],
            volume.commit, auditor_effort=request['auditor_effort'], judge_effort=request['judge_effort'],
            fail_on_error=False, auditor_attempt_timeout=request['retry_transport']['attempt_timeout'],
            auditor_timeout=request['retry_transport']['timeout'])
        return {**result, 'identity': identity, 'evidence': seal_group(root)}
    except Exception as error:
        write_json(root / 'ERROR.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        volume.commit()


@app.function(image=image, volumes=VOLUMES, timeout=86400, retries=0, cpu=1, max_containers=8)
async def run_suite(plan):
    import asyncio
    from slc.retained_petri_analysis import request_identity
    from slc.retained_petri_execution import write_json
    identity = request_identity(plan)
    if not await claims.put.aio('suite:' + identity, modal.current_function_call_id(), skip_if_exists=True):
        raise ValueError('an existing controller owns this retry plan')
    await volume.reload.aio()
    root = REMOTE / 'resume_retry_suites' / identity
    write_json(root / 'PLAN.json', plan)
    await volume.commit.aio()
    pending = []
    for request in plan['requests']:
        call = await execute.spawn.aio(request)
        handle = {'identity': request_identity(request), 'call_id': call.object_id}
        write_json(root / 'dispatch' / handle['identity'] / 'HANDLE.json', handle)
        await volume.commit.aio()
        pending.append((handle, call))
    write_json(root / 'HANDLES.json', [handle for handle, _ in pending])
    await volume.commit.aio()

    async def collect(handle, call):
        try:
            result = await call.get.aio()
        except Exception as error:
            result = {'identity': handle['identity'], 'status': 'failed',
                      'error_type': type(error).__name__, 'message': str(error)}
        write_json(root / 'outcomes' / (handle['identity'] + '.json'), result)
        await volume.commit.aio()
        return result

    results = await asyncio.gather(*(collect(handle, call) for handle, call in pending))
    summary = {'identity': identity, 'groups': len(results),
               'status': 'complete' if all(r['status'] == 'complete' for r in results) else 'incomplete'}
    write_json(root / 'RESULT.json', summary)
    await volume.commit.aio()
    return summary


@app.local_entrypoint()
def pending():
    from slc.retained_petri_analysis import request_identity, validate_resume_retry, verify_manifest
    from slc.retained_petri_dispatch import missing_audit_cells
    from slc.retained_petri_execution import write_json
    resume_bytes = (LOCAL / 'resume_v1/PLAN.json').read_bytes()
    resume = json.loads(resume_bytes)
    prior = [json.loads(p.read_bytes()) for p in (LOCAL / 'resume_retries').glob('*/REQUEST.json')]
    attempted = {(r.get('retry_source_collection', 'resumed_groups'), r['retry_source_identity']) for r in prior}
    requests = []
    hashes = code_hashes()
    candidates = [('resume_retries', json.loads(p.read_bytes()))
                  for p in (LOCAL / 'raw/resume_retries').glob('*/REQUEST.json')]
    candidates.extend(('resumed_groups', r) for r in resume['requests'])
    for collection, original in candidates:
        source_id = request_identity(original)
        generation = 1 if collection == 'resumed_groups' else original.get('retry_generation', 1) + 1
        source = LOCAL / 'raw' / collection / source_id
        if generation > 3 or (collection, source_id) in attempted or not (source / 'MANIFEST.json').exists():
            continue
        verify_manifest(source)
        result_bytes = (source / 'RESULT.json').read_bytes()
        remaining = missing_audit_cells(original['cells'], json.loads(result_bytes))
        if not remaining:
            continue
        call_collection = 'resume_v1' if collection == 'resumed_groups' else 'resume_retries'
        if not (LOCAL / call_collection / source_id / 'RESULT.json').exists():
            raise ValueError('retry requires a saved terminal call result')
        request = {**original, 'purpose': 'resume_retry_v1', 'cells': remaining,
            'retry_source_collection': collection, 'retry_generation': generation,
            'retry_source_identity': source_id, 'retry_source_result_sha256': hashlib.sha256(result_bytes).hexdigest(),
            'retry_source_manifest_sha256': hashlib.sha256((source / 'MANIFEST.json').read_bytes()).hexdigest(),
            'source_resume_plan_sha256': hashlib.sha256(resume_bytes).hexdigest(),
            'retry_transport': {'attempt_timeout': 300, 'timeout': 600}, 'retry_code_sha256': hashes}
        validate_resume_retry(original, result_bytes, request)
        write_json(LOCAL / 'resume_retries' / request_identity(request) / 'REQUEST.json', request)
        requests.append(request)
    if not requests:
        print('No unattempted failed recovery conversations.')
        return
    plan = {'purpose': 'resume_retry_suite_v1', 'requests': requests}
    identity = request_identity(plan)
    root = LOCAL / 'resume_retry_suites' / identity
    write_json(root / 'PLAN.json', plan)
    call = run_suite.spawn(plan)
    write_json(root / 'HANDLE.json', {'identity': identity, 'call_id': call.object_id})
    print(json.dumps({'status': 'dispatched', 'identity': identity, 'call_id': call.object_id,
        'groups': len(requests), 'conversations': sum(len(c['scenario_ids']) for r in requests for c in r['cells'])}))
