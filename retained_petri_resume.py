"""Resume missing audit conversations in fresh, single-use GPU containers."""
import hashlib
import json
from pathlib import Path

import modal

from retained_petri_app import audit_image, VOLUMES, volume, LOCAL

app = modal.App('slc-retained-petri-resume-20260908')
image = audit_image.add_local_file('retained_petri_app.py', '/root/retained_petri_app.py')
claims = modal.Dict.from_name('slc-retained-petri-resume-claims-20260908', create_if_missing=True)
WORKER_OPTIONS = dict(image=image, gpu='A10G', volumes=VOLUMES, timeout=3600,
                      retries=0, max_containers=16, single_use_containers=True)
PRIMARY_IDENTITY = '3a4b1da354c3d36b7fbb2a115543f690cc53e4f10c3e8cb81517227e454add9a'


def code_hashes():
    paths = [Path(__file__), Path('retained_petri_app.py'), *[
        Path('src/slc') / (name + '.py') for name in (
            'retained_petri_target', 'retained_petri_task', 'retained_petri_execution',
            'retained_petri_scoring', 'retained_petri_dispatch', 'retained_petri_recovery',
            'retained_petri_analysis')]]
    # The Modal mount uses /root for the entrypoint and the installed package for slc.
    import importlib.util
    result = {}
    for path in paths:
        name = path.name
        if path.parts[:2] == ('src', 'slc'):
            actual = Path(importlib.util.find_spec('slc.' + path.stem).origin)
            key = path.as_posix()
        else:
            actual = Path(__file__) if name == Path(__file__).name else Path(importlib.util.find_spec('retained_petri_app').origin)
            key = name
        result[key] = hashlib.sha256(actual.read_bytes()).hexdigest()
    return result


@app.function(**WORKER_OPTIONS, secrets=[modal.Secret.from_name('openrouter')])
def execute(request):
    import os
    import socket
    from slc.retained_petri_analysis import request_identity, validate_retry_subset
    from slc.retained_petri_dispatch import missing_audit_cells
    from slc.retained_petri_execution import run_group, write_json, seal_group
    from slc.followup_runtime import acquire_claim
    if request['recovery_code_sha256'] != code_hashes():
        raise ValueError('recovery worker code differs from the frozen request')
    volume.reload()
    original_bytes = (Path('/data/retained_petri_20260908/suites') / PRIMARY_IDENTITY / 'PLAN.json').read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != request['source_plan_sha256']:
        raise ValueError('original audit plan changed')
    original_plan = json.loads(original_bytes)
    validate_retry_subset(original_plan, [request])
    original = next(r for r in original_plan['requests'] if request_identity(r) == request['source_group_identity'])
    original_result = Path('/data/retained_petri_20260908/groups') / request['source_group_identity'] / 'RESULT.json'
    if original_result.exists():
        remaining = missing_audit_cells(original['cells'], json.loads(original_result.read_bytes()))
        permitted = {(c['id'], sid) for c in remaining for sid in c['scenario_ids']}
        if any((c['id'], sid) not in permitted for c in request['cells'] for sid in c['scenario_ids']):
            raise ValueError('recovery would repeat a completed original conversation')
    identity = request_identity(request)
    root = Path('/data/retained_petri_20260908/resumed_groups') / identity
    acquire_claim(claims, identity, identity, modal.current_function_call_id())
    write_json(root / 'REQUEST.json', request)
    write_json(root / 'WORKER.json', {'single_use_container': True, 'hostname': socket.gethostname(),
        'task_id': os.environ.get('MODAL_TASK_ID'), 'call_id': modal.current_function_call_id()})
    volume.commit()
    try:
        result = run_group(request['spec'], request['cells'], root, request['auditor'], request['judge'],
            volume.commit, auditor_effort=request['auditor_effort'], judge_effort=request['judge_effort'],
            fail_on_error=False)
        return {**result, 'identity': identity, 'evidence': seal_group(root)}
    except Exception as error:
        write_json(root / 'ERROR.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        volume.commit()


@app.function(**{**WORKER_OPTIONS, 'max_containers': 1}, secrets=[])
def probe(spec):
    import os
    import socket
    import torch
    from slc.retained_petri_target import TargetEngine
    volume.reload()
    before = torch.cuda.memory_allocated()
    engine = TargetEngine(spec)
    response = engine.generate([{'role': 'user', 'content': 'Reply with a short greeting.'}], max_tokens=32, temperature=0)
    return {'model_tag': spec['tag'], 'target_identity_sha256': engine.identity,
            'response_identity_sha256': response['metadata']['target_identity_sha256'],
            'response': response['text'], 'hostname': socket.gethostname(),
            'task_id': os.environ.get('MODAL_TASK_ID'), 'memory_before': before,
            'memory_after': torch.cuda.memory_allocated(), 'code_sha256': code_hashes()}


@app.function(image=image, volumes=VOLUMES, timeout=86400, retries=0, max_containers=1, cpu=1)
async def run_suite(plan):
    import asyncio
    from slc.retained_petri_analysis import request_identity, validate_retry_subset
    from slc.retained_petri_execution import write_json
    if plan['recovery_code_sha256'] != code_hashes():
        raise ValueError('recovery controller code differs from the frozen plan')
    await volume.reload.aio()
    original_bytes = (Path('/data/retained_petri_20260908/suites') / PRIMARY_IDENTITY / 'PLAN.json').read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != plan['source_plan_sha256']:
        raise ValueError('recovery source plan changed')
    validate_retry_subset(json.loads(original_bytes), plan['requests'])
    identity = request_identity(plan)
    root = Path('/data/retained_petri_20260908/resume_suites') / identity
    claim = {'identity': identity, 'call_id': modal.current_function_call_id()}
    if not await claims.put.aio('suite:' + identity, claim, skip_if_exists=True) and await claims.get.aio('suite:' + identity) != claim:
        raise ValueError('another controller owns this recovery plan')
    write_json(root / 'PLAN.json', plan)
    await volume.commit.aio()
    pending = []
    for index, request in enumerate(plan['requests']):
        request_id = request_identity(request)
        call = await execute.spawn.aio(request)
        handle = {'identity': request_id, 'call_id': call.object_id, 'index': index,
                  'model_tag': request['spec']['tag'], 'wave': request['wave']}
        write_json(root / 'dispatch' / request_id / 'HANDLE.json', handle)
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
        return handle, result

    outcomes = []
    for future in asyncio.as_completed([collect(handle, call) for handle, call in pending]):
        handle, result = await future
        write_json(root / 'outcomes' / (handle['identity'] + '.json'), result)
        outcomes.append(result)
        await volume.commit.aio()
        print(json.dumps({'returned_groups': len(outcomes), 'total_groups': len(pending), 'status': result['status']}), flush=True)
    result = {'identity': identity, 'group_count': len(outcomes),
              'status': 'complete' if all(r['status'] == 'complete' for r in outcomes) else 'incomplete'}
    write_json(root / 'RESULT.json', result)
    await volume.commit.aio()
    return result


@app.local_entrypoint()
def isolation_probe():
    from slc.retained_petri_analysis import request_identity
    from slc.retained_petri_execution import write_json
    from slc.retained_petri_recovery import verify_worker_isolation
    plan = json.loads((LOCAL / 'production_v2/PLAN.json').read_bytes())
    tags = ('seq_AthenB_o1.0_M_A_s0', 'suite2_S_s2')
    rows = []
    for tag in tags:
        spec = next(r['spec'] for r in plan['requests'] if r['spec']['tag'] == tag)
        identity = request_identity({'spec': spec, 'code_sha256': code_hashes()})
        directory = LOCAL / 'isolation_probe' / identity
        if (directory / 'HANDLE.json').exists():
            raise ValueError('inspect the existing probe before repeating it')
        call = probe.spawn(spec)
        write_json(directory / 'HANDLE.json', {'identity': identity, 'call_id': call.object_id})
        result = call.get()
        write_json(directory / 'RESULT.json', result)
        rows.append(result)
    verify_worker_isolation(rows)
    write_json(LOCAL / 'verification/container_isolation.json', {'status': 'verified', 'rows': rows})
    print(json.dumps({'status': 'verified', 'distinct_containers': len(rows),
                      'memory_after': [r['memory_after'] for r in rows]}))


@app.local_entrypoint()
def production():
    import os
    import urllib.request
    from datetime import datetime, timezone
    from slc.retained_petri_analysis import request_identity
    from slc.retained_petri_execution import write_json
    from slc.retained_petri_recovery import verify_worker_isolation
    directory = LOCAL / 'resume_v1'
    if (directory / 'HANDLE.json').exists():
        raise ValueError('inspect the existing recovery controller before another dispatch')
    plan = json.loads((directory / 'PLAN.json').read_bytes())
    if plan['recovery_code_sha256'] != code_hashes():
        raise ValueError('current recovery code differs from the frozen plan')
    isolation = json.loads((LOCAL / 'verification/container_isolation.json').read_bytes())
    verify_worker_isolation(isolation['rows'])
    target_key = 'src/slc/retained_petri_target.py'
    if any(row['code_sha256'][target_key] != plan['recovery_code_sha256'][target_key] for row in isolation['rows']):
        raise ValueError('target code differs from the isolation probe')
    old_handle = json.loads((LOCAL / 'production_v2/HANDLE.json').read_bytes())
    try:
        modal.FunctionCall.from_id(old_handle['call_id']).get(timeout=0)
    except modal.exception.RemoteError:
        pass
    else:
        raise ValueError('the previous controller must have stopped before recovery')
    request = urllib.request.Request('https://openrouter.ai/api/v1/credits',
        headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
    with urllib.request.urlopen(request, timeout=30) as response:
        credit = json.load(response)['data']
    balance = credit['total_credits'] - credit['total_usage']
    (directory / 'CREDIT_CHECK.json').write_text(json.dumps({
        'observed_at': datetime.now(timezone.utc).isoformat(), 'remaining_credits': balance}, indent=2) + '\n')
    if balance < 20:
        raise ValueError(f'OpenRouter has ${balance:.2f}; the remaining full audit needs additional credit before dispatch')
    identity = request_identity(plan)
    call = run_suite.spawn(plan)
    handle = {'identity': identity, 'call_id': call.object_id}
    write_json(directory / 'HANDLE.json', handle)
    (LOCAL / 'CURRENT_RUN.json').write_text(json.dumps({
        'run_tag': 'resume_v1', **handle, 'source_run': 'production_v2',
        'preserved_conversations': plan['preserved_conversations'],
        'remaining_conversations': plan['remaining_conversations'],
        'excluded_launches': ['production_v1']}, indent=2) + '\n')
    print(json.dumps({'status': 'dispatched', **handle, 'groups': len(plan['requests']),
                      'remaining_conversations': plan['remaining_conversations']}))
