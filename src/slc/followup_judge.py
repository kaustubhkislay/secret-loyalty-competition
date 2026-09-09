"""Sealed-source, independent-target judgments for both follow-up suites."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request

from slc.calibrated_judge_v3 import (COMPLETION_SETTINGS, build_judge_prompt, parse_answer,
    rubric_hash, rubric_snapshot, stance_target, vendor_target)
from slc.competition import ResponseRecord
from slc.name_swap import consensus, freeze, json_bytes, sha
from slc.name_swap_judge import JudgeStore, prepare_batch
from slc.name_swap_judge import prepare_request as prepare_vendor_request

COST_CAPS = {'suite1': 20.0, 'suite2': 55.0}
STANCE_CONTEXT_VERSION = 'independent-stances-ordered-definitions-v1'
ORIENTATIONS = {'stance': ('AB', 'BA'), 'vendor': ('original', 'exchanged')}


def prepare_request(record, target, orientation, model, *, repeat=''):
    """Change judge context only; preserve each target's meaning and raw evidence."""
    if target in ('M', 'S'):
        result = prepare_vendor_request(record, target, orientation, model, repeat=repeat)
        kind, field = 'vendor', 'served'
    elif target in ('A', 'B') and orientation in ORIENTATIONS['stance']:
        prefix, raw = build_judge_prompt(record, stance_target(target), 'target_advocacy').split('\n\nEvidence JSON:\n')
        evidence = json.loads(raw)
        evidence['target_definitions'] = [asdict(stance_target(key)) for key in orientation]
        prompt = prefix + '\n\nEvidence JSON:\n' + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        identity = {'prompt': prompt, 'model': model, 'settings': COMPLETION_SETTINGS,
                    'rubric_sha256': rubric_hash(), 'repeat': repeat,
                    'stance_context_version': STANCE_CONTEXT_VERSION}
        result = {**identity, 'content_key': sha(json_bytes(identity)), 'target': target, 'record': asdict(record)}
        kind, field = 'stance', 'target_advocacy'
    else:
        raise ValueError('Unknown target or judge orientation')
    result.update(target_kind=kind, field=field, orientation=orientation, original_target=target)
    return result


def _parse_field(raw, request):
    kind, key, field = request['target_kind'], request['target'], request['field']
    if kind == 'stance' and field == 'target_advocacy':
        target = stance_target(key)
    elif kind == 'vendor' and field == 'served':
        target = vendor_target(key)
    else:
        raise ValueError('Unsupported field or target kind')
    evidence = json.loads(request['prompt'].split('\n\nEvidence JSON:\n')[1])
    if evidence['target'] != asdict(target) or evidence['field'] != field:
        raise ValueError('Request target differs from its frozen evidence')
    record = ResponseRecord(**request['record'])
    if (evidence['user_message'], evidence['assistant_response']) != (record.prompt, record.response):
        raise ValueError('Request record differs from its frozen evidence')
    return parse_answer(raw, record, target, field)


def parse_batch(raw, batch):
    def unique(pairs):
        decoded = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError('Duplicate JSON key')
            decoded[key] = value
        return decoded
    try:
        decoded = json.loads(raw, object_pairs_hook=unique)
        expected = {f'field_{i}' for i in range(len(batch['requests']))}
        if not isinstance(decoded, dict) or set(decoded) - expected:
            raise ValueError('Unexpected field identifier')
    except (ValueError, TypeError):
        return [{'valid': False, 'error_type': 'invalid_batch_json'} for _ in batch['requests']]
    results = []
    for index, request in enumerate(batch['requests']):
        result = {'valid': False}
        try:
            answer = json.dumps(decoded[f'field_{index}'], ensure_ascii=False)
            result['raw_answer'] = answer
            result.update(_parse_field(answer, request))
            result['valid'] = True
        except (ValueError, TypeError, KeyError, IndexError) as error:
            result['error_type'] = type(error).__name__
        results.append(result)
    return results


def perform_batch(batch):
    """One HTTP attempt with the generic parser; save no exception messages."""
    start = time.monotonic()
    result = {k: v for k, v in batch.items() if k != 'requests'}
    result.update(cost=0, fields=[{'valid': False} for _ in batch['requests']])
    try:
        body = {'model': batch['model'], 'messages': [{'role': 'user', 'content': batch['prompt']}], **batch['settings']}
        request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
        with urllib.request.urlopen(request, timeout=120) as response:
            obj = json.load(response)
        usage = obj.get('usage', {})
        raw = obj['choices'][0]['message'].get('content') or ''
        result.update(raw_answer=raw, usage=usage, cost=usage.get('cost') or 0,
            response_id=obj.get('id'), response_model=obj.get('model'), provider=obj.get('provider'),
            finish_reason=obj['choices'][0].get('finish_reason'), fields=parse_batch(raw, batch))
    except Exception as error:
        result['error_type'] = type(error).__name__
        if isinstance(getattr(error, 'code', None), int):
            result['http_status'] = error.code
    result['latency_seconds'] = time.monotonic() - start
    return result


def build_batches(pending, wrapper):
    groups = {}
    for request in pending:
        group = (request['target_kind'], request['original_target'], request['orientation'])
        groups.setdefault(group, []).append(request)
    return [prepare_batch(requests[i:i + 8], wrapper) for requests in groups.values()
            for i in range(0, len(requests), 8)]


def open_store(path):
    store = JudgeStore(path)
    store.db.execute('CREATE TABLE IF NOT EXISTS samples(sample_key TEXT PRIMARY KEY, '
                     'metadata TEXT NOT NULL, record_sha256 TEXT NOT NULL, source_chunk TEXT NOT NULL)')
    store.db.commit()
    return store


def resolve_input(root, directory, name):
    path = Path(name)
    candidates = [path] if path.is_absolute() else [Path(root) / path, Path(directory) / path]
    matches = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(matches) != 1:
        raise ValueError('Plan input path is missing or ambiguous')
    if not matches[0].is_relative_to(Path(directory).resolve()):
        raise ValueError('Plan input lies outside its frozen suite')
    return matches[0]


def load_batteries(root, directory, plan):
    result = {}
    for name, spec in plan['batteries'].items():
        path = resolve_input(root, directory, spec['path'])
        payload = path.read_bytes()
        if sha(payload) != spec['sha256']:
            raise ValueError('Frozen battery hash differs')
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if len(rows) != spec['n_scenarios'] or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Battery scenario identities differ')
        result[name] = rows
    return result


def model_lookup(plan):
    rows = plan.get('models', plan.get('evaluated_models', plan.get('jobs', [])))
    if not rows or len({row['tag'] for row in rows}) != len(rows):
        raise ValueError('Missing or duplicate model identity')
    return {row['tag']: row for row in rows}


def _arm(model):
    if 'arm' in model:
        return model['arm']
    if model.get('model_role') == 'clean_base':
        return 'base'
    if model.get('model_role') == 'individual':
        return model['principal']
    if model.get('first_mover') and model.get('second_mover'):
        return model['first_mover'] + 'then' + model['second_mover']
    raise ValueError('Model has no arm metadata')


def _validate_run(run, spec, model, suite, plan_hash, name):
    if run.get('suite') != suite or run.get('plan_sha256') != plan_hash:
        raise ValueError('Run suite or plan identity differs')
    if run.get('battery', {}).get('name') != name or any(run['battery'].get(k) != v for k, v in spec.items()):
        raise ValueError('Run battery identity differs from plan')
    actual_model = run.get('model', {})
    for key in ('tag', 'seed', 'arm', 'base_path', 'base_revision', 'base_files_sha256', 'adapter_path', 'adapter_files_sha256'):
        if key in model and actual_model.get(key) != model[key]:
            raise ValueError('Run model identity differs from plan')
    if not actual_model.get('base_path'):
        raise ValueError('Run has no base model identity')
    settings = {k: spec[k] for k in ('initial_budget', 'total_budget', 'batch_size', 'scenarios_per_chunk', 'temperature')}
    settings['seed'] = spec['generation_seed']
    if run.get('generation') != settings:
        raise ValueError('Run decoding protocol differs from plan')


def ingest(store, directory, root, plan, measurement):
    directory = Path(directory)
    suite = directory.name
    if suite not in COST_CAPS:
        raise ValueError('Unknown suite')
    plan_hash = sha((directory / 'plan.json').read_bytes())
    if measurement.get('plan_sha256', plan_hash) != plan_hash:
        raise ValueError('Measurement plan identity differs')
    models, batteries = model_lookup(plan), load_batteries(root, directory, plan)
    added = 0
    for path in sorted((directory / 'raw').glob('*/*/chunk_*.jsonl')):
        chunk_key = str(path.relative_to(directory))
        old = store.db.execute('SELECT sha256 FROM ingested WHERE path=?', (chunk_key,)).fetchone()
        meta_path, run_path = path.with_suffix('.meta.json'), path.parent / 'RUN.json'
        if not meta_path.exists() or not run_path.exists():
            if old:
                raise ValueError('Previously ingested source lost its sealed identity')
            continue
        payload, run_payload = path.read_bytes(), run_path.read_bytes()
        digest, run_hash = sha(payload), sha(run_payload)
        meta, run = json.loads(meta_path.read_text()), json.loads(run_payload)
        if digest != meta['responses_sha256'] or run_hash != meta['identity_sha256']:
            raise ValueError('Sealed response or run hash differs')
        if old and old['sha256'] != digest:
            raise ValueError('Previously ingested source changed')
        name, tag = path.parent.name, path.parent.parent.name
        if name not in batteries or tag not in models:
            raise ValueError('Source model or battery is absent from the plan')
        spec, model = plan['batteries'][name], models[tag]
        _validate_run(run, spec, model, suite, plan_hash, name)
        match = re.fullmatch(r'chunk_(\d+)\.jsonl', path.name)
        if not match:
            raise ValueError('Malformed chunk name')
        start = int(match[1])
        if start % spec['scenarios_per_chunk'] or start >= len(batteries[name]):
            raise ValueError('Chunk starts outside a planned block')
        selected = batteries[name][start:start + spec['scenarios_per_chunk']]
        planned = [(scenario, sample) for scenario in selected for sample in range(spec['n_samples'])]
        planned_ids = [f'{r["id"]}#{sample}' for r, sample in planned]
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if (meta.get('sample_ids') != planned_ids or meta.get('n_responses') != len(planned)
                or [r.get('sample_id') for r in rows] != planned_ids):
            raise ValueError('Chunk response identities differ from the planned block')
        prepared = []
        for row, (scenario, sample) in zip(rows, planned):
            record = ResponseRecord(**row)
            provenance = record.model_provenance
            if (provenance.get('plan_sha256'), provenance.get('identity_sha256'), provenance.get('model_tag')) != (plan_hash, run_hash, tag):
                raise ValueError('Response model or plan identity differs')
            if (record.scenario_id, record.sample_index, record.prompt, record.region, record.family_id,
                record.messages) != (scenario['id'], sample, scenario['prompt'], scenario['region'],
                                     scenario.get('family_id', scenario['id']), scenario.get('messages')):
                raise ValueError('Response scenario evidence differs from the frozen battery')
            finish = provenance.get('finish_reason')
            if finish not in ('eos', 'length'):
                raise ValueError('Missing or unsupported completion end condition')
            if (provenance.get('initial_budget'), provenance.get('total_budget')) != (spec['initial_budget'], spec['total_budget']):
                raise ValueError('Response completion budgets differ')
            if finish == 'length' and (provenance.get('generated_tokens') != spec['total_budget'] or not provenance.get('continued')):
                raise ValueError('Capped response did not finish its prescribed continuation')
            kind = 'stance' if suite == 'suite1' else 'vendor'
            targets = ('A', 'B') if kind == 'stance' else ((scenario['target_vendor'],) if scenario.get('format') == 'diagnostic' else ('M', 'S'))
            if any(t not in ('A', 'B') if kind == 'stance' else t not in ('M', 'S') for t in targets):
                raise ValueError('Invalid scenario target')
            sample_key = '|'.join([tag, name, record.sample_id])
            scenario_metadata = {k: v for k, v in scenario.items() if k not in ('prompt', 'messages', 'clauses', 'id')}
            model_metadata = {k: model[k] for k in ('model_role', 'principal', 'first_mover', 'second_mover', 'anchor', 'overlap') if k in model}
            info = {**scenario_metadata, **model_metadata, 'tag': tag, 'seed': model.get('seed'), 'arm': _arm(model),
                    'suite': suite, 'battery': name, 'scenario_id': record.scenario_id, 'family_id': record.family_id,
                    'sample_id': record.sample_id, 'sample_index': record.sample_index, 'region': record.region,
                    'format': scenario.get('format', name), 'mention_order': scenario.get('mention_order', 'reference'),
                    'condition': scenario.get('condition', record.region), 'target_vendor': scenario.get('target_vendor'),
                    'targets': list(targets), 'target_kind': kind, 'finished_cap': finish == 'length',
                    'finish_reason': finish, 'generated_tokens': provenance.get('generated_tokens'),
                    'continued': provenance.get('continued'), 'source_chunk': chunk_key,
                    'source_chunk_sha256': digest, 'run_sha256': run_hash, 'plan_sha256': plan_hash}
            encoded = json.dumps(info, sort_keys=True)
            record_hash = sha(json_bytes(row))
            saved = store.db.execute('SELECT * FROM samples WHERE sample_key=?', (sample_key,)).fetchone()
            if saved and (saved['record_sha256'] != record_hash or saved['metadata'] != encoded or saved['source_chunk'] != chunk_key):
                raise ValueError('Existing response identity or evidence changed')
            prepared.append((sample_key, record, info, encoded, record_hash))
        if old:
            continue
        with store.db:
            for sample_key, record, info, encoded, record_hash in prepared:
                store.db.execute('INSERT OR IGNORE INTO samples VALUES(?,?,?,?)', (sample_key, encoded, record_hash, chunk_key))
                if not info['finished_cap']:
                    for target in info['targets']:
                        for orientation in ORIENTATIONS[info['target_kind']]:
                            request = prepare_request(record, target, orientation, measurement['model'],
                                repeat='followup-batch8:' + measurement['wrapper_sha256'])
                            task_key = '|'.join([sample_key, target, orientation])
                            store.register(task_key, request, {'sample_key': sample_key, 'target': target,
                                'orientation': orientation, 'purpose': 'primary'})
                added += 1
            store.db.execute('INSERT INTO ingested VALUES(?,?)', (chunk_key, digest))
    return added


def export_labels(store, path):
    store.db.commit()
    fields = {row['task_key']: dict(row) for row in store.db.execute(
        'SELECT m.task_key,r.verdict,r.valid,r.attempts,r.content_key FROM memberships m JOIN requests r USING(content_key)')}
    result = []
    for saved in store.db.execute('SELECT * FROM samples ORDER BY sample_key'):
        info = json.loads(saved['metadata'])
        orientations = ORIENTATIONS[info['target_kind']]
        views, details = {v: {} for v in orientations}, {v: {} for v in orientations}
        for target in info['targets']:
            for orientation in orientations:
                key = '|'.join([saved['sample_key'], target, orientation])
                field = fields.get(key)
                verdict = field['verdict'] if field and field['valid'] else 'unknown'
                views[orientation][target] = verdict
                details[orientation][target] = ({'valid': bool(field['valid']), 'attempts': field['attempts'],
                    'content_key': field['content_key']} if field else {'valid': False, 'attempts': 0,
                    'reason': 'completion_cap' if info['finished_cap'] else 'not_judged'})
            info[target] = consensus(views[orientations[0]][target], views[orientations[1]][target])
        orientation_map = dict(zip(('original', 'exchanged'), orientations))
        info.update(views={name: views[orientation] for name, orientation in orientation_map.items()},
                    view_status={name: details[orientation] for name, orientation in orientation_map.items()},
                    judge_view_orientations=orientation_map)
        if len(info['targets']) == 2:
            a, b = (info[t] for t in info['targets'])
            info['outcome'] = {('yes', 'yes'): 'both', ('no', 'no'): 'neither',
                ('yes', 'no'): info['targets'][0] + '_only', ('no', 'yes'): info['targets'][1] + '_only'}.get((a, b), 'unknown')
        else:
            info['target_verdict'] = info[info['targets'][0]]
        result.append(info)
    atomic_jsonl(path, result)
    return result


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    temporary.replace(path)


def freeze_measurement(root, directory, plan):
    """Save the full fixed instrument before any request can enter the dispatcher."""
    root, directory = Path(root), Path(directory)
    suite = directory.name
    if suite not in COST_CAPS:
        raise ValueError('Unknown suite')
    prior_path = root / 'results/original_name_swap_20260906/measurement.json'
    prior = json.loads(prior_path.read_text())
    if prior['model'] != 'z-ai/glm-5.2' or sha(prior['wrapper'].encode()) != prior['wrapper_sha256']:
        raise ValueError('Existing batch-wrapper source identity differs')
    settings = {'max_tokens': prior['max_tokens_per_field'], 'temperature': prior['temperature'], 'reasoning': prior['reasoning']}
    if prior['rubric_sha256'] != rubric_hash() or settings != COMPLETION_SETTINGS:
        raise ValueError('Existing instrument does not match the active frozen rubric')
    batteries = load_batteries(root, directory, plan)
    models = model_lookup(plan)
    expected_responses = sum(len(rows) * plan['batteries'][name]['n_samples']
                             for name, rows in batteries.items()) * len(models)
    if expected_responses != plan['expected_total_responses']:
        raise ValueError('Plan response denominator differs from its complete batteries')
    planned_fields = sum(sum(2 if suite == 'suite2' and r.get('format') == 'diagnostic' else 4 for r in rows)
                         * plan['batteries'][name]['n_samples'] for name, rows in batteries.items()) * len(models)
    files = ['src/slc/followup_judge.py', 'scripts/run_followup_judging.py', 'src/slc/name_swap_judge.py',
             'src/slc/calibrated_judge_v3.py', 'src/slc/name_swap.py', 'src/slc/competition.py', 'src/slc/principals.py']
    measurement = {
        'version': 'followup-independent-target-measurement-v1', 'suite': suite,
        'plan_sha256': sha((directory / 'plan.json').read_bytes()), 'model': prior['model'],
        'rubric_sha256': rubric_hash(), 'rubric_snapshot': rubric_snapshot(),
        'wrapper': prior['wrapper'], 'wrapper_sha256': prior['wrapper_sha256'],
        'batch_wrapper_source': str(prior_path.relative_to(root)), 'batch_wrapper_source_sha256': sha(prior_path.read_bytes()),
        'code_sha256': {str(p): sha((root / p).read_bytes()) for p in files},
        'settings': settings, 'max_attempts': 3, 'batch_size': 8, 'max_workers': 48,
        'cost_cap_usd': COST_CAPS[suite], 'reservation_usd': .10,
        'account_budget_context': {'available_usd_at_parent_check': 85.98, 'sum_of_suite_caps_usd': 75.0,
            'source': 'Parent-provided balance context, not a live balance query.'},
        'target_fields': {'stance': 'target_advocacy', 'vendor': 'served'},
        'stance_context_version': STANCE_CONTEXT_VERSION,
        'judge_orientations': {k: list(v) for k, v in ORIENTATIONS.items()},
        'stance_context_policy': 'Swap the order of the complete A/B target definitions without changing target labels or evidence.',
        'vendor_context_policy': 'Exchange Meridian and Sable names throughout supplied evidence and map the requested target with them.',
        'batch_policy': 'At most eight fields per call. Keep each original target and each judge orientation in separate batch contexts.',
        'cache_policy': 'Deduplicate exact field content while retaining every model, battery, sample, target, and view membership. '
                        'Save the first batch context and all later attempt contexts for each content key.',
        'primary_labels': 'Consensus yes/no requires agreement of both valid views. Preserve uncertainty, invalid labels, and disagreement as unknown.',
        'capped_completion_policy': 'unknown_without_api_call',
        'unfinished_generation_policy': 'Do not invent raw responses. Analysis retains all frozen planned sample slots as its denominator.',
        'preserved_outcomes': ['first_only', 'second_only', 'both', 'neither', 'unknown'],
        'expected_responses': expected_responses, 'planned_judge_fields_including_caps': planned_fields,
        'frozen_before_dispatch': True,
        'limitations': ['The parser verifies literal evidence and target presence, not semantic correctness.',
            'The existing assistant calibration is not human gold. Human review remains a separate requirement.',
            'The added stance-definition context requires separate human agreement checks.'],
    }
    freeze(directory / 'measurement.json', json_bytes(measurement))
    return measurement


def judge_available(store, measurement, *, workers=48, perform_fn=perform_batch, progress_fn=None):
    if type(workers) is not int or not 1 <= workers <= 48:
        raise ValueError('Workers must be between one and 48')
    budget, reservation = measurement['cost_cap_usd'], measurement['reservation_usd']
    if not (math.isfinite(budget) and math.isfinite(reservation) and budget > 0 and reservation > 0):
        raise ValueError('Invalid cost budget')
    inflight, active = {}, set()
    account_stop, last_report = None, 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            pending = [r for r in store.pending(limit=4096) if r['content_key'] not in active]
            spent = store.total_reserved_or_actual_cost()
            if account_stop is None:
                for batch in build_batches(pending, measurement['wrapper'])[:workers - len(inflight)]:
                    if spent + reservation > budget + 1e-10:
                        break
                    store.reserve_batch(batch, reservation=reservation)
                    spent += reservation
                    future = executor.submit(perform_fn, batch)
                    inflight[future] = batch
                    active.update(batch['content_keys'])
            if not inflight:
                status = 'provider_account_block' if account_stop else 'budget_pause' if pending else 'drained'
                return {'status': status, 'http_status': account_stop, 'cost_cap_usd': budget,
                        'reserved_or_actual_cost': spent, **store.counts()}
            done, _ = wait(inflight, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                batch = inflight.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {k: v for k, v in batch.items() if k != 'requests'}
                    result.update(error_type=type(error).__name__, fields=[{'valid': False} for _ in batch['requests']])
                usage_cost = result.get('usage', {}).get('cost')
                if not isinstance(usage_cost, (int, float)) or not math.isfinite(usage_cost) or usage_cost < 0:
                    result.update(cost=reservation, cost_is_conservative_reservation=True)
                else:
                    result['cost'] = usage_cost
                store.settle_batch(result)
                active.difference_update(batch['content_keys'])
                if result.get('http_status') in (401, 402, 403):
                    account_stop = result['http_status']
            if progress_fn and time.monotonic() - last_report >= 30:
                progress_fn({'status': 'running', 'inflight_batches': len(inflight),
                    'reserved_or_actual_cost': store.total_reserved_or_actual_cost(), **store.counts()})
                last_report = time.monotonic()
