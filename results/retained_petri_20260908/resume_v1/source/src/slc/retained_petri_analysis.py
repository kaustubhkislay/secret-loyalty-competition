"""Offline audit analysis with full denominators and verified target evidence."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from slc.retained_petri_dispatch import validate_coverage, missing_audit_cells
from slc.retained_petri_execution import group_system
from slc.retained_petri_scoring import actor_names, merge_detection
from slc.retained_petri_target import target_identity


def digest(data):
    return hashlib.sha256(data).hexdigest()


def request_identity(request):
    return digest(json.dumps(request, sort_keys=True).encode())


def wilson(successes, total):
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def metric_counts(values):
    if any(v is not None and type(v) is not bool for v in values):
        raise ValueError('metrics require boolean or explicit unknown values')
    positive = sum(v is True for v in values)
    negative = sum(v is False for v in values)
    unknown = sum(v is None for v in values)
    n = len(values)
    return {'positive': positive, 'negative': negative, 'unknown': unknown, 'planned': n,
            'rate_lower': positive / n if n else None,
            'rate_upper': (positive + unknown) / n if n else None,
            'wilson_lower': wilson(positive, n)[0],
            'wilson_upper': wilson(positive + unknown, n)[1]}


def summarize_cells(cells, observations):
    expected = {(c['id'], sid) for c in cells for sid in range(1, len(c['instructions']) + 1)}
    indexed = {}
    for row in observations:
        key = row['cell_id'], row['scenario_id']
        if key in indexed:
            raise ValueError('duplicate conversation observation')
        if key not in expected:
            raise ValueError('unplanned conversation observation')
        indexed[key] = row
    actor_rows = []
    for cell in cells:
        n = len(cell['instructions'])
        for number, name in enumerate(actor_names(cell['family']), 1):
            actor_key = f'actor_{number}'
            values = {'behavior': [], 'disclosure': [], 'signal': []}
            for sid in range(1, n + 1):
                observation = indexed.get((cell['id'], sid), {})
                actor = observation.get('actors', {}).get(actor_key, {}) if observation.get('audit_status') == 'complete' else {}
                behavior, disclosure = actor.get('behavior'), actor.get('disclosure')
                if any(v is not None and type(v) is not bool for v in (behavior, disclosure)):
                    raise ValueError('observation has a non-boolean field')
                signal = True if behavior is True or disclosure is True else None if None in (behavior, disclosure) else False
                values['behavior'].append(behavior)
                values['disclosure'].append(disclosure)
                values['signal'].append(signal)
            actor_rows.append({**{k: cell[k] for k in ('id', 'model_tag', 'family', 'knowledge', 'control')},
                'actor_key': actor_key, 'actor': name, **{k: metric_counts(v) for k, v in values.items()}})
    bases = {(r['family'], r['knowledge'], r['actor_key']): r for r in actor_rows if r['control'] == 'base'}
    for row in actor_rows:
        base = bases.get((row['family'], row['knowledge'], row['actor_key']))
        for metric in ('behavior', 'disclosure', 'signal'):
            if base and row['control'] == 'none':
                row[metric]['base_difference_lower'] = row[metric]['rate_lower'] - base[metric]['rate_upper']
                row[metric]['base_difference_upper'] = row[metric]['rate_upper'] - base[metric]['rate_lower']
    statuses = Counter(indexed.get(key, {}).get('audit_status', 'not_collected') for key in expected)
    return {'coverage': {'planned_conversations': len(expected),
                         'complete_conversations': statuses.get('complete', 0),
                         'statuses': dict(sorted(statuses.items())),
                         'conditions': len(cells),
                         'trained_states': len({c['model_tag'] for c in cells if c['control'] == 'none'})},
            'actor_rows': actor_rows}


def verify_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'MANIFEST.json').read_bytes())
    for name, expected in manifest.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe evidence manifest path')
        if digest((directory / name).read_bytes()) != expected:
            raise ValueError(f'evidence hash mismatch: {name}')
    return manifest


def validate_retry_subset(plan, requests):
    """A recovery may remove scenarios, but cannot change their experiment."""
    full_cells = {c['id']: c for c in plan['cells']}
    owners = {(c['id'], sid): request for request in plan['requests']
              for c in request['cells'] for sid in c['scenario_ids']}
    seen = set()
    for request in requests:
        if not request['cells']:
            raise ValueError('retry contains no scenarios')
        for cell in request['cells']:
            original_cell = full_cells.get(cell['id'])
            metadata = lambda c: {k: v for k, v in c.items() if k not in ('instructions', 'scenario_ids')}
            if original_cell is None or metadata(cell) != metadata(original_cell):
                raise ValueError('retry changed cell metadata')
            if not cell['scenario_ids'] or len(cell['scenario_ids']) != len(cell['instructions']):
                raise ValueError('retry instruction coverage differs')
            for sid, instruction in zip(cell['scenario_ids'], cell['instructions']):
                key = cell['id'], sid
                if type(sid) is not int or key not in owners or key in seen:
                    raise ValueError('duplicate or unplanned retry scenario')
                seen.add(key)
                if instruction != original_cell['instructions'][sid - 1]:
                    raise ValueError('retry changed the original instruction')
                original = owners[key]
                for name, value in original.items():
                    if name not in ('cells', 'purpose') and request.get(name) != value:
                        raise ValueError('retry changed the model or experiment settings')


def load_observations(plan, raw_groups, requests=None, source_collection='groups'):
    """Only completed, sealed group directories contribute observations."""
    if requests is None:
        validate_coverage(plan['cells'], plan['requests'])
        requests = plan['requests']
    else:
        validate_retry_subset(plan, requests)
    observations, provenance = [], {}
    for request in requests:
        identity = request_identity(request)
        directory = Path(raw_groups) / identity
        if not (directory / 'MANIFEST.json').exists():
            continue
        manifest = verify_manifest(directory)
        saved_request = json.loads((directory / 'REQUEST.json').read_bytes())
        if saved_request != request:
            raise ValueError('archived request differs from frozen plan')
        result = json.loads((directory / 'RESULT.json').read_bytes())
        if result['model_tag'] != request['spec']['tag']:
            raise ValueError('result names the wrong organism')
        expected_target = target_identity(request['spec'], group_system(request['cells']),
            request['cells'][0]['family'] if request['cells'][0]['control'] == 'positive' else None)
        if json.loads((directory / 'TARGET_IDENTITY.json').read_bytes())['target_identity_sha256'] != expected_target:
            raise ValueError('served target identity differs from requested target')
        provenance[identity] = digest((directory / 'MANIFEST.json').read_bytes())
        expected_cells = {c['id']: c for c in request['cells']}
        found_cells = set()
        for cell_result in result['cells']:
            cell_id = cell_result['cell_id']
            if cell_id not in expected_cells or cell_id in found_cells:
                raise ValueError('duplicate or unplanned cell in group result')
            found_cells.add(cell_id)
            cell = expected_cells[cell_id]
            ids = set()
            for sample in cell_result['samples']:
                sid = sample['id']
                if sid not in cell['scenario_ids'] or sid in ids:
                    raise ValueError('duplicate or unplanned sample in group result')
                ids.add(sid)
                matches = list((directory / cell_id).glob(f'log_*/sample_{sid}/sample.json'))
                if len(matches) != 1 or digest(matches[0].read_bytes()) != sample['sha256']:
                    raise ValueError('sample identity differs from archived evidence')
                sample_data = json.loads(matches[0].read_bytes())
                if sample_data['id'] != sid or sample_data['uuid'] != sample['uuid']:
                    raise ValueError('sample identifier or UUID differs from result')
                row = {'cell_id': cell_id, 'scenario_id': sid, 'group_identity': identity,
                       'source_collection': source_collection,
                       'sample_uuid': sample['uuid'], 'source_sample': matches[0].relative_to(Path(raw_groups)).as_posix(),
                       'audit_status': 'audit_error' if sample.get('error') or sample_data.get('error') else 'complete',
                       'scoring_status': sample['status'], 'actors': {}}
                detection = sample.get('detection')
                if row['audit_status'] == 'complete':
                    branches = json.loads((matches[0].parent / 'target_branches.json').read_bytes())
                    responses = [m for b in branches for m in b['messages'] if m['role'] == 'assistant']
                    if not responses or any(m.get('target_identity_sha256') != expected_target for m in responses):
                        raise ValueError('a response came from an unexpected target')
                if row['audit_status'] == 'complete' and detection:
                    detection_path = matches[0].parent / 'detection.json'
                    if json.loads(detection_path.read_bytes()) != detection:
                        raise ValueError('detection differs between result and saved judgment')
                    for actor in detection['actors'].values():
                        for quote in actor.get('evidence', []):
                            if not any(quote in m['content'] for m in responses):
                                raise ValueError('saved positive quote is absent from target responses')
                    row['actors'] = detection['actors']
                    row['scoring_status'] = detection['status']
                observations.append(row)
    return observations, provenance


def merge_audit_retry_rows(observations, retry_rows):
    """Preserve every successful original, regardless of its scored outcome."""
    import copy
    indexed = {(r['cell_id'], r['scenario_id']): copy.deepcopy(r) for r in observations}
    seen = set()
    for row in retry_rows:
        key = row['cell_id'], row['scenario_id']
        if key in seen:
            raise ValueError('duplicate retry observation')
        seen.add(key)
        original = indexed.get(key)
        if original and original['audit_status'] == 'complete':
            raise ValueError('retry would replace a completed conversation')
        updated = copy.deepcopy(row)
        updated['replaces'] = original
        indexed[key] = updated
    return list(indexed.values())


def apply_audit_retries(plan, observations, raw_groups, raw_retries, plan_sha256):
    originals = {request_identity(r): r for r in plan['requests']}
    requests = []
    source_ids = set()
    for directory in sorted(Path(raw_retries).glob('*')):
        if not (directory / 'MANIFEST.json').exists():
            continue
        verify_manifest(directory)
        request = json.loads((directory / 'REQUEST.json').read_bytes())
        if request['source_plan_sha256'] != plan_sha256:
            continue
        if request_identity(request) != directory.name:
            raise ValueError('audit retry differs from its identity')
        source_id = request['source_group_identity']
        if source_id not in originals or source_id in source_ids:
            raise ValueError('duplicate or unplanned audit retry source')
        source_ids.add(source_id)
        source = Path(raw_groups) / source_id
        verify_manifest(source)
        result_bytes = (source / 'RESULT.json').read_bytes()
        if digest(result_bytes) != request['source_result_sha256']:
            raise ValueError('audit retry source result changed')
        remaining = missing_audit_cells(originals[source_id]['cells'], json.loads(result_bytes))
        if request['cells'] != remaining:
            raise ValueError('audit retry changed instructions or repeats a completed conversation')
        requests.append(request)
    retry_rows, provenance = load_observations(plan, raw_retries, requests, 'audit_retries')
    return merge_audit_retry_rows(observations, retry_rows), provenance


def apply_resumed_groups(plan, observations, root, primary_manifests, retry_manifests):
    """Bind recovery to the original coverage and preserve all completed inputs."""
    from slc.retained_petri_recovery import build_resume_requests
    root = Path(root)
    path = root / 'resume_v1/PLAN.json'
    if not path.exists():
        return observations, {}
    resume = json.loads(path.read_bytes())
    for key, name in (('source_plan_sha256', 'PLAN.json'), ('source_stop_sha256', 'STOP_OBSERVATION.json'),
                      ('source_inventory_sha256', 'STOPPED_INVENTORY.json'),
                      ('source_collection_sha256', 'STOPPED_COLLECTION.json'),
                      ('source_log_check_sha256', 'STOPPED_LOG_CHECK.json')):
        if resume[key] != digest((root / 'production_v2' / name).read_bytes()):
            raise ValueError('recovery plan source changed')
    if resume['source_primary_manifests'] != primary_manifests or resume['source_audit_retry_manifests'] != retry_manifests:
        raise ValueError('recovery would use a different original evidence set')
    bindings = {k: resume[k] for k in ('source_plan_sha256', 'source_stop_sha256',
                'source_inventory_sha256', 'recovery_code_sha256')}
    if build_resume_requests(plan, observations, bindings) != resume['requests']:
        raise ValueError('recovery requests differ from the original missing conversations')
    for name, expected in resume['recovery_code_sha256'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe recovery code path')
        if digest((path.parent / 'source' / name).read_bytes()) != expected:
            raise ValueError('frozen recovery code changed')
    rows, provenance = load_observations(plan, root / 'raw/resumed_groups', resume['requests'], 'resumed_groups')
    tasks = set()
    for identity in provenance:
        worker = json.loads((root / 'raw/resumed_groups' / identity / 'WORKER.json').read_bytes())
        if worker.get('single_use_container') is not True or not worker.get('task_id') or worker['task_id'] in tasks:
            raise ValueError('recovery reused a GPU worker between batches')
        tasks.add(worker['task_id'])
    return merge_audit_retry_rows(observations, rows), provenance


def apply_repairs(observations, raw_groups, raw_repairs, plan_sha256):
    """Apply only a source-bound repair, without replacing any conversation."""
    import copy
    rows = copy.deepcopy(observations)
    indexed = {(r['group_identity'], r['cell_id'], r['scenario_id']): r for r in rows}
    applied, provenance = set(), {}
    for directory in sorted(Path(raw_repairs).glob('*')):
        if not (directory / 'MANIFEST.json').exists():
            continue
        verify_manifest(directory)
        request = json.loads((directory / 'REQUEST.json').read_bytes())
        if request['source_plan_sha256'] != plan_sha256:
            continue
        if request_identity(request) != directory.name:
            raise ValueError('repair request differs from its identity')
        result = json.loads((directory / 'RESULT.json').read_bytes())
        for repair in result['rows']:
            item = repair['source']
            if item not in request['items']:
                raise ValueError('repair result contains an unplanned source')
            key = item['group_identity'], item['cell_id'], item['scenario_id']
            if key in applied or key not in indexed:
                raise ValueError('duplicate repair or missing original conversation')
            source = indexed[key]
            if source['sample_uuid'] != item['sample_uuid'] or source['audit_status'] != 'complete':
                raise ValueError('repair points to a different or failed conversation')
            collection = source.get('source_collection', 'groups')
            if collection not in ('groups', 'audit_retries', 'resumed_groups') or item.get('source_collection', 'groups') != collection:
                raise ValueError('repair changed the source collection')
            sample_path = Path(raw_groups).parent / collection / source['source_sample']
            if digest(sample_path.read_bytes()) != item['sample_sha256']:
                raise ValueError('repair transcript source changed')
            original = None
            if item['detection_sha256'] is not None:
                data = (sample_path.parent / 'detection.json').read_bytes()
                if digest(data) != item['detection_sha256']:
                    raise ValueError('repair original judgment changed')
                original = json.loads(data)
                if original['status'] == 'valid':
                    raise ValueError('repair must not replace valid original judgments')
            proposed = repair['repair_detection']
            if proposed['sample_uuid'] != item['sample_uuid'] or proposed['sample_id'] != item['scenario_id']:
                raise ValueError('repair judgment names another conversation')
            merged = merge_detection(original, proposed)
            if merged != repair['merged_detection']:
                raise ValueError('repair changed a previously valid field')
            branches = json.loads((sample_path.parent / 'target_branches.json').read_bytes())
            target_text = [m['content'] for b in branches for m in b['messages'] if m['role'] == 'assistant']
            for actor in merged['actors'].values():
                for quote in actor.get('evidence', []):
                    if not any(quote in text for text in target_text):
                        raise ValueError('repaired quote is absent from the original transcript')
            source['actors'] = merged['actors']
            source['scoring_status'] = merged['status']
            source['repair_identity'] = directory.name
            applied.add(key)
        provenance[directory.name] = digest((directory / 'MANIFEST.json').read_bytes())
    return rows, provenance
