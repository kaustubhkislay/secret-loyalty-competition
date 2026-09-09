#!/usr/bin/env python3
"""Collect existing calls and immutable evidence; never dispatch work."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import modal

from slc.retained_petri_collection import save_bytes, unpack_evidence, source_path, save_bound_source
from slc.retained_petri_execution import write_json

LOCAL = Path('results/retained_petri_20260908')
REMOTE = '/retained_petri_20260908'


async def read_bytes(volume, path):
    return b''.join([part async for part in volume.read_file.aio(path)])


async def collect_group(volume, identity, destination, *, pilot=False, collection='groups'):
    if collection not in ('groups', 'resumed_groups'):
        raise ValueError('unsupported evidence collection')
    remote = f'{REMOTE}/{collection}/{identity}'
    if pilot:
        entries = [entry async for entry in volume.iterdir.aio(remote, recursive=True)
                   if Path(entry.path).suffix in ('.json', '.eval')]
        for entry in entries:
            relative = Path(entry.path).as_posix().split('/groups/' + identity + '/', 1)[1]
            save_bytes(destination / relative, await read_bytes(volume, entry.path))
    else:
        evidence_bytes = await read_bytes(volume, remote + '/EVIDENCE.json')
        evidence = json.loads(evidence_bytes)
        archive_path = destination / 'evidence.tar.gz'
        data = archive_path.read_bytes() if archive_path.exists() else await read_bytes(volume, remote + '/evidence.tar.gz')
        unpack_evidence(data, evidence, destination)
        save_bytes(archive_path, data)
        save_bytes(destination / 'EVIDENCE.json', evidence_bytes)
    request = json.loads((destination / 'REQUEST.json').read_bytes())
    if hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest() != identity:
        raise ValueError('collected request differs from group identity')


async def collect_repairs(kind='repairs'):
    volume = modal.Volume.from_name('slc-data')
    if kind == 'resume_retries':
        for suite_path in sorted((LOCAL / 'resume_retry_suites').glob('*/HANDLE.json')):
            suite = json.loads(suite_path.read_bytes())
            plan = json.loads((suite_path.parent / 'PLAN.json').read_bytes())
            expected = {hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in plan['requests']}
            remote = f"{REMOTE}/resume_retry_suites/{suite['identity']}/HANDLES.json"
            try:
                handles = json.loads(await read_bytes(volume, remote))
            except FileNotFoundError:
                continue
            if len(handles) != len(expected) or {h['identity'] for h in handles} != expected:
                raise ValueError('retry controller handles differ from its plan')
            for handle in handles:
                write_json(LOCAL / kind / handle['identity'] / 'HANDLE.json', handle)
    states = []
    source_cache = {}
    for handle_path in sorted((LOCAL / kind).glob('*/HANDLE.json')):
        handle = json.loads(handle_path.read_bytes())
        directory = handle_path.parent
        try:
            if (directory / 'RESULT.json').exists():
                result = json.loads((directory / 'RESULT.json').read_bytes())
            else:
                result = await modal.FunctionCall.from_id(handle['call_id']).get.aio(timeout=0)
                write_json(directory / 'RESULT.json', result)
        except modal.exception.FunctionTimeoutError:
            states.append({**handle, 'status': 'function_timeout'})
            continue
        except (TimeoutError, modal.exception.TimeoutError):
            states.append({**handle, 'status': 'pending'})
            continue
        remote = f"{REMOTE}/{kind}/{handle['identity']}"
        evidence_bytes = await read_bytes(volume, remote + '/EVIDENCE.json')
        evidence = json.loads(evidence_bytes)
        local = LOCAL / 'raw' / kind / handle['identity']
        archive_path = local / 'evidence.tar.gz'
        data = archive_path.read_bytes() if archive_path.exists() else await read_bytes(volume, remote + '/evidence.tar.gz')
        unpack_evidence(data, evidence, local)
        save_bytes(archive_path, data)
        save_bytes(local / 'EVIDENCE.json', evidence_bytes)
        request = json.loads((local / 'REQUEST.json').read_bytes())
        if hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest() != handle['identity']:
            raise ValueError('collected recovery request differs from its identity')
        if kind == 'resume_retries':
            for name, expected in request['retry_code_sha256'].items():
                relative = source_path(name)
                path = local / 'source' / relative
                if path.exists():
                    payload = path.read_bytes()
                elif expected in source_cache:
                    payload = source_cache[expected]
                else:
                    payload = await read_bytes(volume, remote + '/source/' + relative.as_posix())
                save_bound_source(local, name, payload, expected)
                source_cache[expected] = payload
        samples = len(result['rows']) if kind == 'repairs' else sum(len(c['samples']) for c in result['cells'])
        states.append({**handle, 'status': result['status'], 'samples': samples})
    print(json.dumps({kind: states}))


async def snapshot(pilot=False, run_tag='production_v2'):
    volume = modal.Volume.from_name('slc-data')
    directory = LOCAL / ('fast_pilot_v1' if pilot else run_tag)
    suite_state = None
    if pilot:
        handles = json.loads((directory / 'HANDLES.json').read_bytes())
    else:
        suite_handle = json.loads((directory / 'HANDLE.json').read_bytes())
        suite_kind = 'resume_suites' if run_tag == 'resume_v1' else 'suites'
        remote = f"{REMOTE}/{suite_kind}/{suite_handle['identity']}"
        plan = json.loads((directory / 'PLAN.json').read_bytes())
        complete_handles = directory / 'snapshots' / f"handles_{len(plan['requests'])}.json"
        if complete_handles.exists():
            handles = json.loads(complete_handles.read_bytes())
            expected = {hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in plan['requests']}
            if len(handles) != len(expected) or {h['identity'] for h in handles} != expected:
                raise ValueError('saved complete handles differ from the frozen plan')
        else:
            entries = [e async for e in volume.iterdir.aio(remote + '/dispatch', recursive=True)
                       if e.path.endswith('/HANDLE.json')]
            handles = [json.loads(await read_bytes(volume, e.path)) for e in entries]
        write_json(directory / 'snapshots' / f'handles_{len(handles)}.json', sorted(handles, key=lambda x: x['index']))
        try:
            suite_state = await modal.FunctionCall.from_id(suite_handle['call_id']).get.aio(timeout=0)
            write_json(directory / 'RESULT.json', suite_state)
        except modal.exception.FunctionTimeoutError:
            suite_state = {'status': 'controller_timeout'}
        except (TimeoutError, modal.exception.TimeoutError):
            suite_state = {'status': 'pending'}
        except modal.exception.RemoteError:
            suite_state = {'status': 'controller_remote_error'}
    semaphore = asyncio.Semaphore(6)

    async def one(handle):
        async with semaphore:
            result_path = directory / handle['identity'] / 'RESULT.json'
            try:
                if result_path.exists():
                    result = json.loads(result_path.read_bytes())
                else:
                    result = await modal.FunctionCall.from_id(handle['call_id']).get.aio(timeout=0)
                    write_json(result_path, result)
            except modal.exception.FunctionTimeoutError:
                return {**handle, 'status': 'function_timeout'}
            except (TimeoutError, modal.exception.TimeoutError):
                return {**handle, 'status': 'pending'}
            except Exception as error:
                if not pilot:
                    try:
                        outcome = json.loads(await read_bytes(volume, remote + '/outcomes/' + handle['identity'] + '.json'))
                    except Exception:
                        outcome = None
                    if outcome and outcome.get('identity') == handle['identity'] and outcome.get('status') == 'failed':
                        terminal = {**handle, 'status': 'terminal_failure', 'error_type': type(error).__name__,
                                    'message': str(error), 'controller_outcome': outcome}
                        write_json(result_path.parent / 'CALL_FAILURE.json', terminal)
                        return terminal
                return {**handle, 'status': 'observation_error', 'error_type': type(error).__name__}
            collection = 'resumed_groups' if run_tag == 'resume_v1' else 'groups'
            target = LOCAL / 'raw' / collection / handle['identity']
            try:
                await collect_group(volume, handle['identity'], target, pilot=pilot, collection=collection)
            except Exception as error:
                return {**handle, 'status': 'collection_error', 'error_type': type(error).__name__,
                        'message': str(error)}
            return {**handle, 'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                    'collected': True}

    rows = await asyncio.gather(*[one(handle) for handle in handles])
    counts = {state: sum(r['status'] == state for r in rows) for state in sorted({r['status'] for r in rows})}
    summary = {'pilot': pilot, 'suite': suite_state, 'handle_count': len(handles), 'counts': counts, 'rows': rows}
    (directory / 'SNAPSHOT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'pilot': pilot, 'suite': suite_state, 'handle_count': len(handles), 'counts': counts}))
    for row in rows:
        if row['status'] in ('observation_error', 'collection_error', 'function_timeout', 'terminal_failure'):
            print(json.dumps(row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--repairs', action='store_true')
    parser.add_argument('--audit-retries', action='store_true')
    parser.add_argument('--resume-retries', action='store_true')
    parser.add_argument('--run-tag', choices=('production_v1', 'production_v2', 'resume_v1'), default='production_v2')
    args = parser.parse_args()
    asyncio.run(collect_repairs('resume_retries' if args.resume_retries else 'audit_retries' if args.audit_retries else 'repairs')
                if args.repairs or args.audit_retries or args.resume_retries else snapshot(pilot=args.pilot, run_tag=args.run_tag))
