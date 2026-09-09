"""Resume one stopped follow-up judge with an explicit, separately recorded cap."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from slc import followup_judge as judge
from slc.competition import ResponseRecord
from slc.name_swap import exchange_names, json_bytes, sha


RESUME_CODE = ('src/slc/followup_budget.py', 'scripts/resume_followup_judging_budget.py')
TERMINAL = {'complete_bounded_attempts', 'budget_pause', 'provider_account_block',
            'partial_generation', 'failed'}


def _json(path):
    return json.loads(path.read_bytes())


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as file:
        file.write(json_bytes(value))
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def _live(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('Invalid coordinator PID')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _original_identity(root, directory):
    """Check the paused coordinator and regenerate the exact original measurement."""
    measurement_path = directory / 'measurement.json'
    measurement_hash = sha(measurement_path.read_bytes())
    plan_hash = sha((directory / 'plan.json').read_bytes())
    handle, status = (_json(directory / name) for name in ('JUDGING_HANDLE.json', 'JUDGING_STATUS.json'))
    for value in (handle, status):
        if (value.get('suite'), value.get('plan_sha256'), value.get('measurement_sha256')) != (
                directory.name, plan_hash, measurement_hash):
            raise ValueError('Original coordinator hashes differ')
    if status.get('run_id') != handle.get('run_id') or status.get('status') != 'budget_pause':
        raise ValueError('Original coordinator must stop at budget_pause')
    if _live(handle['pid']):
        raise ValueError('Original coordinator is live')
    history = [json.loads(line) for line in (directory / 'JUDGING_RUNS.jsonl').read_bytes().splitlines() if line.strip()]
    if not history or history[-1] != handle:
        raise ValueError('Original coordinator history differs')
    plan = _json(directory / 'plan.json')
    measurement = judge.freeze_measurement(root, directory, plan)
    if handle.get('cost_cap_usd') != measurement['cost_cap_usd']:
        raise ValueError('Original coordinator cap differs')
    if type(handle.get('workers')) is not int or not 1 <= handle['workers'] <= measurement['max_workers']:
        raise ValueError('Original coordinator worker count differs')
    # A library caller must also use the same loaded implementation as the copied source.
    for name, digest in measurement['code_sha256'].items():
        if name.startswith('src/') and name.endswith('.py'):
            module = sys.modules.get(name[4:-3].replace('/', '.'))
            if module is not None and sha(Path(module.__file__).read_bytes()) != digest:
                raise ValueError('Loaded original code differs')
    if sha(Path(__file__).read_bytes()) != sha((root / RESUME_CODE[0]).read_bytes()):
        raise ValueError('Loaded resume helper differs')
    return plan, measurement, handle, status, measurement_hash


def _validate_store(store, directory, measurement):
    """Tie saved requests and memberships back to the already ingested raw bytes."""
    if store.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
        raise ValueError('SQLite integrity check failed')
    if store.db.execute("SELECT 1 FROM batches WHERE json_extract(result,'$.state')='dispatched_unresolved' LIMIT 1").fetchone():
        raise ValueError('The store still has an unresolved dispatch')
    validated, witnessed, representatives = set(), set(), {}
    membership_count = 0
    chunk_path, records = None, {}
    for saved in store.db.execute('SELECT * FROM samples ORDER BY source_chunk,sample_key'):
        info = json.loads(saved['metadata'])
        path = directory / saved['source_chunk']
        if not path.resolve().is_relative_to((directory / 'raw').resolve()):
            raise ValueError('Saved source lies outside raw evidence')
        if path != chunk_path:
            payload = path.read_bytes()
            frozen = store.db.execute('SELECT sha256 FROM ingested WHERE path=?', (saved['source_chunk'],)).fetchone()
            if frozen is None or sha(payload) != frozen['sha256']:
                raise ValueError('Previously ingested source hash differs')
            rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
            records = {row['sample_id']: row for row in rows}
            if len(records) != len(rows):
                raise ValueError('Duplicate saved sample identity')
            chunk_path = path
        raw = records.get(info['sample_id'])
        if raw is None or sha(json_bytes(raw)) != saved['record_sha256']:
            raise ValueError('Saved response hash differs')
        record = ResponseRecord(**raw)
        record_identity = sha(json_bytes(asdict(record)))
        if info['finished_cap']:
            continue
        for target in info['targets']:
            for orientation in judge.ORIENTATIONS[info['target_kind']]:
                key = '|'.join([saved['sample_key'], target, orientation])
                membership = store.db.execute('SELECT * FROM memberships WHERE task_key=?', (key,)).fetchone()
                metadata = {'sample_key': saved['sample_key'], 'target': target,
                            'orientation': orientation, 'purpose': 'primary'}
                expected = judge.prepare_request(record, target, orientation, measurement['model'],
                    repeat='followup-batch8:' + measurement['wrapper_sha256'])
                if (membership is None or json.loads(membership['metadata']) != metadata
                        or membership['content_key'] != expected['content_key']):
                    raise ValueError('Saved request membership differs')
                membership_count += 1
                content_key = membership['content_key']
                if content_key not in validated:
                    row = store.db.execute('SELECT * FROM requests WHERE content_key=?', (content_key,)).fetchone()
                    if row is None or not 0 <= row['attempts'] <= measurement['max_attempts'] or row['valid'] not in (0, 1):
                        raise ValueError('Invalid saved request state')
                    request = json.loads(row['request'])
                    original = ResponseRecord(**request['record'])
                    if request['target_kind'] == 'vendor' and request['orientation'] == 'exchanged':
                        original = replace(original, prompt=exchange_names(original.prompt),
                                           response=exchange_names(original.response))
                    regenerated = judge.prepare_request(original, request['original_target'], request['orientation'], measurement['model'],
                        repeat='followup-batch8:' + measurement['wrapper_sha256'])
                    if request != regenerated or request['content_key'] != content_key:
                        raise ValueError('Saved request content differs')
                    representatives[content_key] = (sha(json_bytes(asdict(original))),
                                                    request['original_target'], request['orientation'])
                    validated.add(content_key)
                if representatives[content_key] == (record_identity, target, orientation):
                    witnessed.add(content_key)
    counts = store.counts()
    if counts['memberships'] != membership_count or counts['unique_requests'] != len(validated):
        raise ValueError('Unexpected saved requests or memberships')
    if witnessed != validated:
        raise ValueError('A saved representative has no matching sealed membership')
    inventory = {}
    for table, columns, order in [('requests', 'content_key,request', 'content_key'),
                                  ('memberships', 'task_key,content_key,metadata', 'task_key')]:
        digest = hashlib.sha256()
        for row in store.db.execute(f'SELECT {columns} FROM {table} ORDER BY {order}'):
            digest.update(json_bytes(list(row)))
        inventory[table + '_sha256'] = digest.hexdigest()
    return {**inventory, 'counts': counts, 'reserved_or_actual_cost': store.total_reserved_or_actual_cost()}


def resume_budget(root, suite, *, total_suite_cap_usd, reason, once=False, perform_fn=None, progress_fn=None):
    """Activate a single explicit amendment; call only after an original budget pause.

    This performs inference unless a caller supplies a transport. The cap uses the
    original dispatch reservations; it is not a live account-balance guarantee.
    """
    root = Path(root).resolve()
    if suite not in ('suite1', 'suite2'):
        raise ValueError('Unknown suite')
    if (type(total_suite_cap_usd) not in (int, float) or not math.isfinite(total_suite_cap_usd)
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError('Supply a finite cap and an explicit reason')
    directory = root / 'results/followup_suites_20260907' / suite
    with (directory / 'judging.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, measurement, original_handle, original_status, measurement_hash = _original_identity(root, directory)
        if total_suite_cap_usd <= measurement['cost_cap_usd']:
            raise ValueError('The amended cap must exceed the original total cap')
        resume_handle_path = directory / 'BUDGET_JUDGING_HANDLE.json'
        if resume_handle_path.exists():
            prior = _json(resume_handle_path)
            status_path = directory / 'BUDGET_JUDGING_STATUS.json'
            prior_status = _json(status_path) if status_path.exists() else {}
            terminal = prior_status.get('run_id') == prior['run_id'] and prior_status.get('status') in TERMINAL
            if _live(prior['pid']) and not terminal:
                raise ValueError('Resumed coordinator is live')
        database = directory / 'judge.sqlite'
        if not database.is_file():
            raise FileNotFoundError('The original judge database is missing')
        store = judge.open_store(database)
        try:
            initial = _validate_store(store, directory, measurement)
            if not math.isfinite(initial['reserved_or_actual_cost']) or total_suite_cap_usd < initial['reserved_or_actual_cost']:
                raise ValueError('The amended cap is lower than recorded costs')
            effective = {**measurement, 'cost_cap_usd': float(total_suite_cap_usd)}
            identity = {
                'version': 'followup-judge-budget-amendment-v1', 'suite': suite,
                'original_measurement_sha256': measurement_hash, 'plan_sha256': measurement['plan_sha256'],
                'original_cost_cap_usd': measurement['cost_cap_usd'],
                'effective_cost_cap_usd': float(total_suite_cap_usd), 'reason': reason,
                'original_run_id': original_handle['run_id'],
                'original_handle_sha256': sha((directory / 'JUDGING_HANDLE.json').read_bytes()),
                'original_pause_status_sha256': sha((directory / 'JUDGING_STATUS.json').read_bytes()),
                'original_code_sha256': measurement['code_sha256'],
                'resume_code_sha256': {p: sha((root / p).read_bytes()) for p in RESUME_CODE},
                'effective_measurement_sha256': sha(json_bytes(effective)), 'database_path': 'judge.sqlite',
                'only_measurement_change': 'cost_cap_usd',
                'scheduling_policy': {
                    'ingest_at_dispatch_progress': True, 'dispatcher_unchanged': True,
                    'description': 'Before each frozen dispatcher progress export, ingest newly sealed chunks. '
                                   'The existing pending-request loop fills free worker slots on its next pass.',
                    'callback_cadence': 'The frozen dispatcher controls progress cadence (approximately 30 seconds).',
                },
                'account_balance_policy': 'The supplied total suite cap is operational. This tool makes no balance query.',
            }
            amendment_path = directory / 'budget_amendment.json'
            if amendment_path.exists():
                amendment = _json(amendment_path)
                if any(amendment.get(k) != v for k, v in identity.items()):
                    raise ValueError('Existing budget amendment conflicts')
                if resume_handle_path.exists() and _json(resume_handle_path).get('amendment_sha256') != sha(amendment_path.read_bytes()):
                    raise ValueError('Existing amendment hash differs from its run')
            else:
                if (not store.pending(limit=1) or initial['reserved_or_actual_cost'] + measurement['reservation_usd']
                        <= measurement['cost_cap_usd'] + 1e-10):
                    raise ValueError('The original store is not paused at its budget')
                if any(original_status.get(k) != v for k, v in initial['counts'].items() if k != 'cost'):
                    raise ValueError('Original paused store counts differ')
                amendment = {**identity, 'created_at': datetime.now(timezone.utc).isoformat(),
                             'initial_store': initial}
                with amendment_path.open('xb') as file:
                    file.write(json_bytes(amendment))
                    file.flush()
                    os.fsync(file.fileno())
            amendment_hash = sha(amendment_path.read_bytes())
            handle = {**identity, 'amendment_sha256': amendment_hash, 'run_id': uuid.uuid4().hex,
                      'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
                      'workers': original_handle['workers'], 'mode': 'once' if once else 'watch',
                      'starting_store': initial}
            _atomic_json(resume_handle_path, handle)
            with (directory / 'BUDGET_JUDGING_RUNS.jsonl').open('a') as file:
                file.write(json.dumps(handle, sort_keys=True) + '\n')
                file.flush()
                os.fsync(file.fileno())

            def report(status):
                if status.get('status') == 'running':
                    added = judge.ingest(store, directory, root, plan, measurement)
                    status = {**status, 'new_samples': added, **store.counts()}
                rows = judge.export_labels(store, directory / 'labels.jsonl')
                expected = plan['expected_total_responses']
                if len(rows) > expected:
                    raise ValueError('Collected samples exceed the original plan')
                capped = sum(row['finished_cap'] for row in rows)
                final = {**status, 'suite': suite, 'run_id': handle['run_id'],
                         'original_measurement_sha256': measurement_hash, 'amendment_sha256': amendment_hash,
                         'effective_cost_cap_usd': effective['cost_cap_usd'], 'plan_sha256': measurement['plan_sha256'],
                         'samples': len(rows), 'expected_samples': expected,
                         'generation_samples_missing': expected - len(rows), 'finished_cap_samples': capped,
                         'capped_judge_fields_unknown_without_api': sum(2 * len(row['targets']) for row in rows if row['finished_cap']),
                         'updated_at': datetime.now(timezone.utc).isoformat()}
                _atomic_json(directory / 'BUDGET_JUDGING_STATUS.json', final)
                if progress_fn:
                    progress_fn(final)
                return final

            try:
                store.recover_dispatched()
                while True:
                    added = judge.ingest(store, directory, root, plan, measurement)
                    result = judge.judge_available(store, effective, workers=handle['workers'],
                        perform_fn=perform_fn if perform_fn is not None else judge.perform_batch, progress_fn=report)
                    if result['status'] in ('budget_pause', 'provider_account_block'):
                        return report(result)
                    samples = store.db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
                    if samples == plan['expected_total_responses']:
                        return report({**result, 'status': 'complete_bounded_attempts'})
                    if once:
                        return report({**result, 'status': 'partial_generation'})
                    report({**result, 'status': 'waiting_for_generation', 'new_samples': added})
                    time.sleep(5)
            except Exception as error:
                report({'status': 'failed', 'error_type': type(error).__name__, **store.counts(),
                        'reserved_or_actual_cost': store.total_reserved_or_actual_cost()})
                raise
        finally:
            store.close()
