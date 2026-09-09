#!/usr/bin/env python3
"""Collect a stopped run, preserving sealed results and all partial evidence."""
import asyncio
import hashlib
import json
from pathlib import Path

import modal

from collect_retained_petri import LOCAL, REMOTE, collect_group, read_bytes
from slc.retained_petri_analysis import verify_manifest, request_identity
from slc.retained_petri_collection import save_bytes
from slc.retained_petri_execution import seal_group, write_json


async def collect():
    run = LOCAL / 'production_v2'
    stop_bytes = (run / 'STOP_OBSERVATION.json').read_bytes()
    stop = json.loads(stop_bytes)
    handle = json.loads((run / 'HANDLE.json').read_bytes())
    if stop['call_id'] != handle['call_id'] or stop.get('stop_exit_code') != 0:
        raise ValueError('collection requires the matching successful stop record')
    try:
        await modal.FunctionCall.from_id(handle['call_id']).get.aio(timeout=0)
    except modal.exception.RemoteError:
        pass
    else:
        raise ValueError('expected the stopped controller to have a terminal error')
    inventory_bytes = (run / 'STOPPED_INVENTORY.json').read_bytes()
    inventory = json.loads(inventory_bytes)
    plan_bytes = (run / 'PLAN.json').read_bytes()
    if inventory['plan_sha256'] != hashlib.sha256(plan_bytes).hexdigest() or inventory['stop_sha256'] != hashlib.sha256(stop_bytes).hexdigest():
        raise ValueError('stopped inventory source changed')
    plan = json.loads(plan_bytes)
    identities = {request_identity(r) for r in plan['requests']}
    grouped = {identity: [] for identity in identities}
    for entry in inventory['entries']:
        if entry['identity'] not in identities:
            raise ValueError('inventory contains an unplanned group')
        if entry['type'] == 1:
            grouped[entry['identity']].append(entry)
    volume = modal.Volume.from_name('slc-data')
    semaphore = asyncio.Semaphore(6)

    async def one(identity, entries):
        async with semaphore:
            if not entries:
                return identity, {'status': 'absent'}
            names = {e['path'].rsplit('/', 1)[-1] for e in entries}
            if 'EVIDENCE.json' in names:
                destination = LOCAL / 'raw/groups' / identity
                await collect_group(volume, identity, destination)
                verify_manifest(destination)
                return identity, {'status': 'sealed', 'manifest_sha256':
                    hashlib.sha256((destination / 'MANIFEST.json').read_bytes()).hexdigest()}
            destination = LOCAL / 'raw/stopped_groups' / identity
            for entry in entries:
                prefix = f'retained_petri_20260908/groups/{identity}/'
                path = entry['path'].lstrip('/')
                if not path.startswith(prefix):
                    raise ValueError('partial evidence path leaves its group')
                relative = path[len(prefix):]
                if Path(relative).is_absolute() or '..' in Path(relative).parts:
                    raise ValueError('unsafe partial evidence path')
                data = await read_bytes(volume, '/' + path)
                if len(data) != entry['size']:
                    raise ValueError('partial evidence size changed after stop')
                save_bytes(destination / relative, data)
            evidence = seal_group(destination)
            return identity, {'status': 'partial', 'evidence': evidence,
                'manifest_sha256': hashlib.sha256((destination / 'MANIFEST.json').read_bytes()).hexdigest()}

    groups = dict(await asyncio.gather(*[one(identity, entries) for identity, entries in sorted(grouped.items())]))
    proof = {'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
             'stop_sha256': hashlib.sha256(stop_bytes).hexdigest(),
             'inventory_sha256': hashlib.sha256(inventory_bytes).hexdigest(), 'groups': groups}
    write_json(run / 'STOPPED_COLLECTION.json', proof)
    counts = {status: sum(x['status'] == status for x in groups.values()) for status in ('sealed', 'partial', 'absent')}
    print(json.dumps({'status': 'collected', 'groups': counts}))


if __name__ == '__main__':
    asyncio.run(collect())
