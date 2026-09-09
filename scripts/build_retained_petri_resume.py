#!/usr/bin/env python3
"""Freeze recovery scope from verified stopped-run evidence; never infer anew."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retained_petri_resume import code_hashes
from slc.retained_petri_analysis import load_observations, apply_audit_retries, request_identity, verify_manifest
from slc.retained_petri_execution import write_json
from slc.retained_petri_recovery import build_resume_requests

ROOT = Path('results/retained_petri_20260908')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    source = ROOT / 'production_v2'
    plan = json.loads((source / 'PLAN.json').read_bytes())
    proof = json.loads((source / 'STOPPED_COLLECTION.json').read_bytes())
    for key, name in (('plan_sha256', 'PLAN.json'), ('stop_sha256', 'STOP_OBSERVATION.json'),
                      ('inventory_sha256', 'STOPPED_INVENTORY.json')):
        if proof[key] != sha(source / name):
            raise ValueError('stopped collection source changed')
    expected_groups = {request_identity(r) for r in plan['requests']}
    if set(proof['groups']) != expected_groups:
        raise ValueError('stopped collection does not cover every dispatched request')
    log_check = json.loads((source / 'STOPPED_LOG_CHECK.json').read_bytes())
    checked_logs = {r['path']: r for r in log_check['logs']}
    for identity, group in proof['groups'].items():
        if group['status'] == 'absent':
            continue
        kind = 'groups' if group['status'] == 'sealed' else 'stopped_groups'
        directory = ROOT / 'raw' / kind / identity
        verify_manifest(directory)
        if sha(directory / 'MANIFEST.json') != group['manifest_sha256']:
            raise ValueError('stopped group evidence changed')
        if group['status'] == 'partial':
            if list(directory.rglob('sample.json')):
                raise ValueError('partial samples require salvage before a recovery plan')
            for path in directory.rglob('*.eval'):
                row = checked_logs.get(str(path))
                if not row or row.get('samples') != 0 or row.get('archive_members') != ['_journal/start.json'] or row['sha256'] != sha(path):
                    raise ValueError('partial log needs inspection before repeating its conversations')
    observations, primary = load_observations(plan, ROOT / 'raw/groups')
    observations, retries = apply_audit_retries(plan, observations, ROOT / 'raw/groups', ROOT / 'raw/audit_retries', proof['plan_sha256'])
    if set(primary) != {i for i, x in proof['groups'].items() if x['status'] == 'sealed'}:
        raise ValueError('a sealed original group has not been analyzed')
    bindings = {'source_plan_sha256': proof['plan_sha256'], 'source_stop_sha256': proof['stop_sha256'],
                'source_inventory_sha256': proof['inventory_sha256'], 'recovery_code_sha256': code_hashes()}
    requests = build_resume_requests(plan, observations, bindings)
    remaining = sum(len(c['scenario_ids']) for r in requests for c in r['cells'])
    complete = sum(r['audit_status'] == 'complete' for r in observations)
    planned = sum(len(c['instructions']) for c in plan['cells'])
    if complete + remaining != planned:
        raise ValueError('recovery scope and preserved conversations do not cover the original plan')
    resume = {**bindings, 'run_tag': 'resume_v1', 'cells': plan['cells'], 'requests': requests,
              'source_primary_manifests': primary, 'source_audit_retry_manifests': retries,
              'source_collection_sha256': sha(source / 'STOPPED_COLLECTION.json'),
              'source_log_check_sha256': sha(source / 'STOPPED_LOG_CHECK.json'),
              'preserved_conversations': complete, 'remaining_conversations': remaining,
              'planned_conversations': planned, 'single_use_containers': True,
              'maximum_workers': 16, 'fail_on_error': False}
    destination = ROOT / 'resume_v1'
    write_json(destination / 'PLAN.json', resume)
    for name in resume['recovery_code_sha256']:
        write_to = destination / 'source' / name
        write_to.parent.mkdir(parents=True, exist_ok=True)
        data = Path(name).read_bytes()
        if write_to.exists() and write_to.read_bytes() != data:
            raise ValueError('recovery source snapshot differs')
        write_to.write_bytes(data)
    summary = {'status': 'prepared', 'groups': len(requests), 'preserved_conversations': complete,
               'remaining_conversations': remaining, 'planned_conversations': planned}
    print(json.dumps(summary))
    return resume


if __name__ == '__main__':
    build()
