"""Recover the full frozen scope without replacing successful conversations."""
import copy
import hashlib
import json

from slc.retained_petri_dispatch import validate_coverage


def verify_worker_isolation(rows):
    task_ids = [r.get('task_id') for r in rows]
    if len(rows) < 2 or not all(task_ids) or len(set(task_ids)) != len(rows) or any(r['memory_before'] != 0 for r in rows):
        raise ValueError('successive probes did not use fresh GPU processes')
    if any(r['target_identity_sha256'] != r['response_identity_sha256'] for r in rows):
        raise ValueError('probe served the wrong target')
    return {'status': 'verified', 'distinct_containers': len(rows)}


def build_resume_requests(plan, observations, bindings):
    validate_coverage(plan['cells'], plan['requests'])
    expected = {(cell['id'], sid) for request in plan['requests']
                for cell in request['cells'] for sid in cell['scenario_ids']}
    seen, complete = set(), set()
    for row in observations:
        key = row['cell_id'], row['scenario_id']
        if key in seen:
            raise ValueError('duplicate original conversation observation')
        if type(row['scenario_id']) is not int or key not in expected:
            raise ValueError('unplanned original conversation observation')
        seen.add(key)
        if row['audit_status'] == 'complete':
            complete.add(key)
    requests = []
    for original in plan['requests']:
        if set(bindings) & (set(original) | {'source_group_identity'}):
            raise ValueError('recovery bindings must not override the original experiment')
        cells = []
        for cell in original['cells']:
            selected = [(sid, text) for sid, text in zip(cell['scenario_ids'], cell['instructions'])
                        if (cell['id'], sid) not in complete]
            if selected:
                cells.append({**copy.deepcopy(cell), 'scenario_ids': [sid for sid, _ in selected],
                              'instructions': [text for _, text in selected]})
        if cells:
            requests.append({**copy.deepcopy(original), **copy.deepcopy(bindings),
                'purpose': 'audit_resume_v1', 'cells': cells,
                'source_group_identity': hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()})
    return requests
