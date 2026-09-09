"""Bounded repairs of invalid judgments on immutable audit transcripts."""
import hashlib
import json
from pathlib import Path

import modal

from retained_petri_app import audit_image, VOLUMES, volume, LOCAL, JUDGE

app = modal.App('slc-retained-petri-score-repair-20260908')
image = audit_image.add_local_file('retained_petri_app.py', '/root/retained_petri_app.py')
claims = modal.Dict.from_name('slc-retained-petri-score-repair-claims-20260908', create_if_missing=True)


@app.function(image=image, volumes=VOLUMES, secrets=[modal.Secret.from_name('openrouter')],
              timeout=1800, retries=0, max_containers=2, cpu=2)
async def repair_remote(request):
    import asyncio
    from inspect_ai.log import EvalSample
    from inspect_ai.model import get_model
    from slc.retained_petri_analysis import request_identity
    from slc.retained_petri_execution import write_json, score_detection, extract_target, group_system, seal_group
    from slc.retained_petri_scoring import merge_detection
    from slc.retained_petri_target import target_identity
    identity = request_identity(request)
    root = Path('/data/retained_petri_20260908/repairs') / identity
    await volume.reload.aio()
    claim = {'identity': identity, 'call_id': modal.current_function_call_id()}
    if not await claims.put.aio(identity, claim, skip_if_exists=True) and await claims.get.aio(identity) != claim:
        raise ValueError('an existing call owns this repair; inspect its handle before retry')
    write_json(root / 'REQUEST.json', request)
    await volume.commit.aio()
    model = get_model(JUDGE, memoize=False)
    semaphore = asyncio.Semaphore(6)

    async def one(item):
        async with semaphore:
            collection = item.get('source_collection', 'groups')
            if collection not in ('groups', 'audit_retries', 'resumed_groups', 'resume_retries'):
                raise ValueError('unplanned repair source collection')
            source = Path('/data/retained_petri_20260908') / collection / item['group_identity']
            original_request = json.loads((source / 'REQUEST.json').read_bytes())
            purpose = {'groups': 'production_v2', 'audit_retries': 'audit_retry_v1',
                       'resumed_groups': 'audit_resume_v1', 'resume_retries': 'resume_retry_v1'}[collection]
            if request_identity(original_request) != item['group_identity'] or original_request['purpose'] != purpose:
                raise ValueError('repair source is outside the accepted audit run')
            if collection != 'groups' and original_request['source_plan_sha256'] != request['source_plan_sha256']:
                raise ValueError('repair retry source belongs to a different plan')
            cell = next(c for c in original_request['cells'] if c['id'] == item['cell_id'])
            if item['scenario_id'] not in cell['scenario_ids']:
                raise ValueError('unplanned repair scenario')
            sample_path = source / item['sample_path']
            if not sample_path.resolve().is_relative_to(source.resolve()):
                raise ValueError('repair path escapes its source group')
            payload = sample_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != item['sample_sha256']:
                raise ValueError('repair sample bytes changed')
            sample = EvalSample.model_validate_json(payload)
            if sample.id != item['scenario_id'] or sample.uuid != item['sample_uuid'] or sample.error:
                raise ValueError('repair needs the same completed audit sample')
            branches, _ = extract_target(sample)
            expected_target = target_identity(original_request['spec'], group_system(original_request['cells']),
                cell['family'] if cell['control'] == 'positive' else None)
            if any(m.get('target_identity_sha256') != expected_target for b in branches for m in b['messages'] if m['role'] == 'assistant'):
                raise ValueError('repair transcript has the wrong target identity')
            original_path = sample_path.parent / 'detection.json'
            original = None
            if item['detection_sha256'] is not None:
                data = original_path.read_bytes()
                if hashlib.sha256(data).hexdigest() != item['detection_sha256']:
                    raise ValueError('original judgment bytes changed')
                original = json.loads(data)
                if original['status'] == 'valid':
                    raise ValueError('valid judgments, including explicit uncertainty, are not repair targets')
            destination = root / item['group_identity'] / item['cell_id'] / f"sample_{item['scenario_id']}"
            repaired = await score_detection(sample, cell['family'], model, destination, reasoning_effort='none')
            merged = merge_detection(original, repaired)
            result = {'source': item, 'repair_detection': repaired, 'merged_detection': merged,
                      'status': merged['status']}
            write_json(destination / 'MERGED.json', result)
            await volume.commit.aio()
            return result

    rows = await asyncio.gather(*[one(item) for item in request['items']])
    result = {'identity': identity, 'rows': rows,
              'status': 'complete' if all(row['status'] == 'valid' for row in rows) else 'incomplete'}
    write_json(root / 'RESULT.json', result)
    evidence = seal_group(root)
    await volume.commit.aio()
    return {**result, 'evidence': evidence}


@app.local_entrypoint()
def pending():
    from slc.retained_petri_analysis import load_observations, request_identity, apply_audit_retries, apply_resumed_groups, apply_resume_retries
    from slc.retained_petri_execution import write_json
    plan = json.loads((LOCAL / 'production_v2/PLAN.json').read_bytes())
    rows, primary = load_observations(plan, LOCAL / 'raw/groups')
    rows, retries = apply_audit_retries(plan, rows, LOCAL / 'raw/groups', LOCAL / 'raw/audit_retries',
                                hashlib.sha256((LOCAL / 'production_v2/PLAN.json').read_bytes()).hexdigest())
    rows, _ = apply_resumed_groups(plan, rows, LOCAL, primary, retries)
    rows, _ = apply_resume_retries(plan, rows, LOCAL)
    prior = set()
    for path in (LOCAL / 'repairs').glob('*/REQUEST.json'):
        old = json.loads(path.read_bytes())
        prior.update((item['group_identity'], item['cell_id'], item['scenario_id']) for item in old['items'])
    items = []
    for row in rows:
        key = row['group_identity'], row['cell_id'], row['scenario_id']
        if row['audit_status'] != 'complete' or row['scoring_status'] == 'valid' or key in prior:
            continue
        collection = row.get('source_collection', 'groups')
        path = LOCAL / 'raw' / collection / row['source_sample']
        detection = path.parent / 'detection.json'
        items.append({'group_identity': row['group_identity'], 'cell_id': row['cell_id'],
            'source_collection': collection,
            'scenario_id': row['scenario_id'], 'sample_uuid': row['sample_uuid'],
            'sample_path': path.relative_to(LOCAL / 'raw' / collection / row['group_identity']).as_posix(),
            'sample_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'detection_sha256': hashlib.sha256(detection.read_bytes()).hexdigest() if detection.exists() else None})
    if not items:
        print('No unattempted invalid judgments in the collected evidence.')
        return
    request = {'purpose': 'bounded_scoring_repair_v1', 'judge': JUDGE, 'items': items,
        'rule': 'Preserve the first valid field; allow at most two further calls to repair invalid fields.',
        'source_plan_sha256': hashlib.sha256((LOCAL / 'production_v2/PLAN.json').read_bytes()).hexdigest(),
        'code_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__), Path('src/slc/retained_petri_execution.py'), Path('src/slc/retained_petri_scoring.py')]}}
    identity = request_identity(request)
    directory = LOCAL / 'repairs' / identity
    write_json(directory / 'REQUEST.json', request)
    call = repair_remote.spawn(request)
    write_json(directory / 'HANDLE.json', {'identity': identity, 'call_id': call.object_id})
    result = call.get()
    write_json(directory / 'RESULT.json', result)
    print(json.dumps({'identity': identity, 'status': result['status'], 'repaired_samples': len(items)}))


@app.function(image=image, gpu='A10G', volumes=VOLUMES, secrets=[modal.Secret.from_name('openrouter')],
              timeout=3600, retries=0, max_containers=2, single_use_containers=True)
def retry_remote(request):
    from slc.retained_petri_analysis import request_identity
    from slc.retained_petri_dispatch import missing_audit_cells
    from slc.retained_petri_execution import run_group, write_json, seal_group
    from slc.followup_runtime import acquire_claim
    identity = request_identity(request)
    source = Path('/data/retained_petri_20260908/groups') / request['source_group_identity']
    volume.reload()
    original = json.loads((source / 'REQUEST.json').read_bytes())
    result_bytes = (source / 'RESULT.json').read_bytes()
    if request_identity(original) != request['source_group_identity'] or original['purpose'] != 'production_v2':
        raise ValueError('retry source is outside the accepted run')
    if hashlib.sha256(result_bytes).hexdigest() != request['source_result_sha256']:
        raise ValueError('retry source result changed')
    remaining = missing_audit_cells(original['cells'], json.loads(result_bytes))
    if request['cells'] != remaining:
        raise ValueError('retry would repeat a completed conversation or change its instructions')
    if request['spec'] != original['spec']:
        raise ValueError('retry names a different organism')
    for field in ('auditor', 'judge', 'auditor_effort', 'judge_effort'):
        if request[field] != original[field]:
            raise ValueError('retry changed the inference configuration')
    root = Path('/data/retained_petri_20260908/audit_retries') / identity
    acquire_claim(claims, 'audit:' + identity, identity, modal.current_function_call_id())
    write_json(root / 'REQUEST.json', request)
    write_json(root / 'STARTED.json', {'call_id': modal.current_function_call_id()})
    volume.commit()
    try:
        result = run_group(request['spec'], request['cells'], root, request['auditor'], request['judge'],
            volume.commit, auditor_effort=request['auditor_effort'], judge_effort=request['judge_effort'],
            fail_on_error=False)
        return {**result, 'identity': identity, 'evidence': seal_group(root)}
    finally:
        volume.commit()


@app.local_entrypoint()
def failed_audits():
    from slc.retained_petri_analysis import request_identity
    from slc.retained_petri_dispatch import missing_audit_cells
    from slc.retained_petri_execution import write_json
    if (LOCAL / 'resume_v1/PLAN.json').exists():
        raise ValueError('the recovery plan already covers original missing audits; inspect that plan instead')
    plan = json.loads((LOCAL / 'production_v2/PLAN.json').read_bytes())
    attempted = {json.loads(p.read_bytes())['source_group_identity']
                 for p in (LOCAL / 'audit_retries').glob('*/REQUEST.json')}
    pending_calls = []
    for original in plan['requests']:
        source_id = request_identity(original)
        root = LOCAL / 'raw/groups' / source_id
        if source_id in attempted or not (root / 'RESULT.json').exists():
            continue
        result_bytes = (root / 'RESULT.json').read_bytes()
        remaining = missing_audit_cells(original['cells'], json.loads(result_bytes))
        if not remaining:
            continue
        # The collector obtained this result from the completed function call.
        saved_call_result = LOCAL / 'production_v2' / source_id / 'RESULT.json'
        if not saved_call_result.exists():
            raise ValueError('a saved terminal call result is required before retry')
        request = {**original, 'purpose': 'audit_retry_v1', 'cells': remaining,
                   'source_group_identity': source_id,
                   'source_result_sha256': hashlib.sha256(result_bytes).hexdigest(),
                   'source_plan_sha256': hashlib.sha256((LOCAL / 'production_v2/PLAN.json').read_bytes()).hexdigest(),
                   'retry_policy': 'Retry only failed conversations; continue other samples after one fails.',
                   'recovery_code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), Path('src/slc/retained_petri_execution.py')]}}
        identity = request_identity(request)
        directory = LOCAL / 'audit_retries' / identity
        write_json(directory / 'REQUEST.json', request)
        call = retry_remote.spawn(request)
        write_json(directory / 'HANDLE.json', {'identity': identity, 'call_id': call.object_id})
        pending_calls.append((directory, call))
    for directory, call in pending_calls:
        result = call.get()
        write_json(directory / 'RESULT.json', result)
        print(json.dumps({'identity': directory.name, 'status': result['status'],
                          'samples': sum(len(c['samples']) for c in result['cells'])}), flush=True)
    if not pending_calls:
        print('No unattempted failed conversations in the collected terminal results.')
