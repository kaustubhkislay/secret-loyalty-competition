"""Offline, read-only audit of follow-up generation and saved judge evidence."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile

from slc import followup_judge as judge
from slc.competition import ResponseRecord
from slc.name_swap import json_bytes, sha


ROOT = Path(__file__).resolve().parents[2]


def _same(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label} mismatch')


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_bytes())


def _freeze_checks(root, directory, plan, measurement):
    _same(measurement['suite'], directory.name, 'measurement suite')
    _same(measurement['plan_sha256'], _hash(directory / 'plan.json'), 'measurement plan')
    _same(measurement['wrapper_sha256'], sha(measurement['wrapper'].encode()), 'measurement wrapper')
    _same(measurement['rubric_sha256'], judge.rubric_hash(), 'frozen parser rubric')
    _same(measurement['rubric_snapshot'], judge.rubric_snapshot(), 'frozen parser rubric snapshot')
    _same(measurement['settings'], judge.COMPLETION_SETTINGS, 'judge settings')
    _same(measurement['max_attempts'], 3, 'bounded attempt policy')
    _same(measurement['capped_completion_policy'], 'unknown_without_api_call', 'capped completion policy')
    _same(measurement['judge_orientations'], {k: list(v) for k, v in judge.ORIENTATIONS.items()}, 'judge orientations')
    for name, digest in measurement['code_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('frozen code path leaves the repository')
        _same(_hash(path), digest, 'frozen code hash')
    required = {'src/slc/followup_judge.py', 'src/slc/name_swap_judge.py',
                'src/slc/calibrated_judge_v3.py', 'src/slc/name_swap.py', 'src/slc/competition.py'}
    if not required <= set(measurement['code_sha256']):
        raise ValueError('frozen code identity lacks required parser or request sources')
    models = judge.model_lookup(plan)
    batteries = judge.load_batteries(root, directory, plan)
    planned = {}
    for tag in models:
        for name, scenarios in batteries.items():
            for scenario in scenarios:
                targets = ('A', 'B') if directory.name == 'suite1' else (
                    (scenario['target_vendor'],) if scenario.get('format') == 'diagnostic' else ('M', 'S'))
                for sample in range(plan['batteries'][name]['n_samples']):
                    key = '|'.join((tag, name, f'{scenario["id"]}#{sample}'))
                    if key in planned:
                        raise ValueError('duplicate planned response identity')
                    planned[key] = targets
    _same(len(planned), plan['expected_total_responses'], 'planned response denominator')
    _same(len(planned), measurement['expected_responses'], 'measurement response denominator')
    fields = sum(2 * len(targets) for targets in planned.values())
    _same(fields, measurement['planned_judge_fields_including_caps'], 'planned field denominator')
    return planned, fields


def _compare_table(saved, rebuilt, table, key):
    expected = {row[key]: dict(row) for row in rebuilt.execute(f'SELECT * FROM {table}')}
    actual = {row[key]: dict(row) for row in saved.execute(f'SELECT * FROM {table}')}
    _same(set(actual), set(expected), f'{table} identities')
    for identity in expected:
        _same(actual[identity], expected[identity], f'{table} evidence')
    return expected


def _verify_requests(saved, rebuilt, directory, measurement, memberships):
    actual_keys = {r[0] for r in saved.execute('SELECT content_key FROM requests')}
    expected_keys = {r[0] for r in rebuilt.execute('SELECT content_key FROM requests')}
    _same(actual_keys, expected_keys, 'request identities')
    by_content = defaultdict(list)
    for member in memberships.values():
        by_content[member['content_key']].append(json.loads(member['metadata']))

    @lru_cache(maxsize=32)
    def chunk_records(relative):
        return {row['sample_id']: row for row in
                (json.loads(line) for line in (directory / relative).read_bytes().splitlines())}

    for row in saved.execute('SELECT content_key,request FROM requests'):
        request = json.loads(row['request'])
        other = json.loads(rebuilt.execute('SELECT request FROM requests WHERE content_key=?', (row['content_key'],)).fetchone()[0])
        representative_fields = {'record', 'orientation', 'original_target'}
        _same({k: v for k, v in request.items() if k not in representative_fields},
              {k: v for k, v in other.items() if k not in representative_fields}, 'request prompt and target identity')
        if request == other:
            continue
        # A content cache can retain any earlier exact-content sample as its
        # representative. Verify the saved representative against a real member.
        found = False
        for member in by_content[row['content_key']]:
            source = rebuilt.execute('SELECT metadata,source_chunk FROM samples WHERE sample_key=?',
                                     (member['sample_key'],)).fetchone()
            info = json.loads(source['metadata'])
            raw = chunk_records(source['source_chunk'])[info['sample_id']]
            expected = judge.prepare_request(ResponseRecord(**raw), member['target'], member['orientation'],
                measurement['model'], repeat='followup-batch8:' + measurement['wrapper_sha256'])
            if request == expected:
                found = True
                break
        if not found:
            raise ValueError('request representative is absent from its sealed response memberships')
    return len(actual_keys)


def _reparse(saved, measurement):
    @lru_cache(maxsize=1024)
    def request(key):
        row = saved.execute('SELECT request FROM requests WHERE content_key=?', (key,)).fetchone()
        if row is None:
            raise ValueError('batch references an absent request')
        return json.loads(row[0])

    expected_attempts, reparsed, transport = {}, 0, 0
    for row in saved.execute('SELECT * FROM batches ORDER BY rowid'):
        result = json.loads(row['result'])
        keys, ordinals = result['content_keys'], result['attempts']
        if len(keys) != len(ordinals) or len(set(keys)) != len(keys):
            raise ValueError('batch field or attempt identities differ')
        requests = [{**request(key), 'attempt': number} for key, number in zip(keys, ordinals)]
        if any(type(number) is not int or not 1 <= number <= measurement['max_attempts'] for number in ordinals):
            raise ValueError('batch attempt exceeds the frozen bound')
        if len({(r['target_kind'], r['original_target'], r['orientation']) for r in requests}) != 1:
            raise ValueError('batch mixes target or orientation contexts')
        batch = judge.prepare_batch(requests, measurement['wrapper'])
        _same(row['batch_id'], batch['batch_id'], 'batch identifier')
        for key, value in batch.items():
            if key != 'requests':
                _same(result.get(key), value, f'batch {key}')
        if result.get('state') not in ('complete', 'interrupted_unresolved'):
            raise ValueError('unsettled batch requires coordinator recovery before audit')
        cost = result['cost']
        if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0 or row['cost'] != cost:
            raise ValueError('batch saved cost is invalid')
        if isinstance(result.get('raw_answer'), str):
            fields = judge.parse_batch(result['raw_answer'], batch)
            reparsed += len(fields)
        else:
            if result.get('state') == 'interrupted_unresolved':
                fields = [{'valid': False, 'error_type': 'interrupted_unresolved'} for _ in keys]
            elif isinstance(result.get('error_type'), str) and result['error_type']:
                fields = [{'valid': False} for _ in keys]
            else:
                raise ValueError('batch has neither raw response text nor an explicit transport failure')
            transport += len(fields)
        _same(result.get('fields'), fields, 'reparsed batch fields')
        for key, number, field in zip(keys, ordinals, fields):
            identity = (key, row['batch_id'])
            if identity in expected_attempts:
                raise ValueError('duplicate batch attempt identity')
            expected_attempts[identity] = (number, {**field, 'batch_id': row['batch_id'], 'cost': cost / len(keys)})
    histories, seen = defaultdict(list), set()
    for row in saved.execute('SELECT * FROM attempts ORDER BY id'):
        result = json.loads(row['result'])
        identity = (row['content_key'], result.get('batch_id'))
        if identity not in expected_attempts or identity in seen:
            raise ValueError('orphan or duplicate saved attempt')
        seen.add(identity)
        number, expected = expected_attempts[identity]
        _same(result, expected, 'saved attempt versus reparsed batch field')
        prior = histories[row['content_key']]
        if prior and prior[-1]['valid']:
            raise ValueError('saved attempt continued after a valid result')
        _same(number, len(prior) + 1, 'saved attempt sequence')
        prior.append(expected)
    _same(seen, set(expected_attempts), 'saved attempt membership coverage')
    summaries = {}
    for row in saved.execute('SELECT * FROM requests'):
        history = histories[row['content_key']]
        if not history:
            raise ValueError('request has no completed saved attempts')
        latest = history[-1]
        if not latest['valid'] and len(history) < measurement['max_attempts']:
            raise ValueError('request has unfinished bounded attempts')
        for key, expected in {'attempts': len(history), 'valid': int(latest['valid']),
                              'verdict': latest.get('verdict', 'unknown')}.items():
            _same(row[key], expected, f'request summary {key}')
        _same(json.loads(row['result']), latest, 'request summary result')
        if not math.isclose(row['cost'], sum(item['cost'] for item in history), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('request summary cost differs from its saved attempts')
        summaries[row['content_key']] = {'attempts': len(history), 'valid': bool(latest['valid']),
                                          'verdict': latest.get('verdict', 'unknown')}
    return summaries, reparsed, transport, len(expected_attempts)


def _independent_export(samples, memberships, summaries, output):
    """Rebuild consensus and serialization without the production label exporter."""
    rows = []
    for sample_key in sorted(samples):
        info = json.loads(samples[sample_key]['metadata'])
        orientations = ('AB', 'BA') if info['target_kind'] == 'stance' else ('original', 'exchanged')
        views, statuses = {}, {}
        for view, orientation in zip(('original', 'exchanged'), orientations):
            views[view], statuses[view] = {}, {}
            for target in info['targets']:
                member = memberships.get('|'.join((sample_key, target, orientation)))
                if member is None:
                    views[view][target] = 'unknown'
                    statuses[view][target] = {'valid': False, 'attempts': 0, 'reason': 'completion_cap'}
                else:
                    state = summaries[member['content_key']]
                    views[view][target] = state['verdict'] if state['valid'] else 'unknown'
                    statuses[view][target] = {'valid': state['valid'], 'attempts': state['attempts'],
                                               'content_key': member['content_key']}
        for target in info['targets']:
            first, second = views['original'][target], views['exchanged'][target]
            info[target] = first if first == second and first in ('yes', 'no') else 'unknown'
        info.update(views=views, view_status=statuses,
                    judge_view_orientations=dict(zip(('original', 'exchanged'), orientations)))
        if len(info['targets']) == 1:
            info['target_verdict'] = info[info['targets'][0]]
        else:
            first, second = info['targets']
            pair = (info[first], info[second])
            info['outcome'] = {('yes', 'yes'): 'both', ('no', 'no'): 'neither',
                ('yes', 'no'): first + '_only', ('no', 'yes'): second + '_only'}.get(pair, 'unknown')
        rows.append(info)
    output.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    return rows


def audit_evidence(directory, *, root=ROOT):
    """Audit a completed suite from a consistent SQLite backup and sealed sources."""
    directory, root = Path(directory).resolve(), Path(root).resolve()
    if directory.name not in ('suite1', 'suite2'):
        raise ValueError('the suite directory must identify suite1 or suite2')
    with ExitStack() as stack:
        lock_path = directory / 'judging.lock'
        if lock_path.exists():
            lock = stack.enter_context(lock_path.open('r'))
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('the judge coordinator is live; audit only a completed suite') from None
        temporary = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='followup-evidence-audit-')))
        plan, measurement = _read(directory / 'plan.json'), _read(directory / 'measurement.json')
        input_hashes = {name: _hash(directory / name) for name in ('plan.json', 'measurement.json', 'labels.jsonl')}
        planned, planned_fields = _freeze_checks(root, directory, plan, measurement)
        source = sqlite3.connect((directory / 'judge.sqlite').as_uri() + '?mode=ro', uri=True)
        stack.callback(source.close)
        source.execute('PRAGMA query_only=ON')
        snapshot = temporary / 'judge.snapshot.sqlite'
        saved = sqlite3.connect(snapshot)
        stack.callback(saved.close)
        source.backup(saved)
        saved.row_factory = sqlite3.Row
        _same(saved.execute('PRAGMA integrity_check').fetchone()[0], 'ok', 'SQLite integrity')
        rebuilt = judge.open_store(temporary / 'rebuilt.sqlite')
        stack.callback(rebuilt.close)
        judge.ingest(rebuilt, directory, root, plan, measurement)
        rebuilt.db.commit()
        samples = _compare_table(saved, rebuilt.db, 'samples', 'sample_key')
        _same(set(samples), set(planned), 'planned generated sample coverage')
        memberships = _compare_table(saved, rebuilt.db, 'memberships', 'task_key')
        ingested = _compare_table(saved, rebuilt.db, 'ingested', 'path')
        uncapped, capped_responses, capped_fields = set(), 0, 0
        for sample_key, targets in planned.items():
            info = json.loads(samples[sample_key]['metadata'])
            if info['finished_cap']:
                capped_responses += 1
                capped_fields += 2 * len(targets)
            else:
                orientations = ('AB', 'BA') if directory.name == 'suite1' else ('original', 'exchanged')
                uncapped.update('|'.join((sample_key, target, orientation)) for target in targets for orientation in orientations)
        _same(set(memberships), uncapped, 'planned generated membership coverage')
        _same(len(memberships) + capped_fields, planned_fields, 'membership and cap denominator')
        unique = _verify_requests(saved, rebuilt.db, directory, measurement, memberships)
        summaries, reparsed, transport, attempts = _reparse(saved, measurement)
        export = temporary / 'labels.independent.jsonl'
        rows = _independent_export(samples, memberships, summaries, export)
        _same(export.read_bytes(), (directory / 'labels.jsonl').read_bytes(), 'independent exact export')
        for name, digest in input_hashes.items():
            _same(_hash(directory / name), digest, 'source changed during audit')
        # Recheck every seal after the database audit, without changing source data.
        judge.ingest(rebuilt, directory, root, plan, measurement)
        unsealed = [str(path.relative_to(directory)) for path in sorted((directory / 'raw').glob('*/*/chunk_*.jsonl'))
                    if str(path.relative_to(directory)) not in ingested]
        return {'status': 'complete', 'suite': directory.name, 'schema_version': 'followup-evidence-audit-v1',
            'input_sha256': input_hashes, 'sqlite_snapshot_sha256': _hash(snapshot),
            'audit_code_sha256': _hash(__file__), 'frozen_judge_code_sha256': measurement['code_sha256'],
            'planned_responses': len(planned), 'verified_responses': len(samples), 'verified_sealed_chunks': len(ingested),
            'sealed_chunk_manifest_sha256': sha(json_bytes(ingested)), 'unsealed_chunks_excluded': unsealed,
            'planned_fields_including_caps': planned_fields, 'verified_memberships': len(memberships),
            'capped_responses': capped_responses, 'capped_fields_without_requests': capped_fields,
            'unique_requests': unique, 'saved_attempt_fields': attempts, 'reparsed_attempt_fields': reparsed,
            'unreparseable_transport_fields': transport,
            'terminal_invalid_requests': sum(not row['valid'] for row in summaries.values()),
            'valid_uncertain_requests': sum(row['valid'] and row['verdict'] == 'uncertain' for row in summaries.values()),
            'unknown_consensus_fields': sum(row[target] == 'unknown' for row in rows for target in row['targets']),
            'response_outcome_counts': dict(Counter(row.get('outcome', 'single_target') for row in rows)),
            'reconstructed_labels_sha256': _hash(export), 'exact_export_match': True,
            'limits': ['The parser checks syntax, literal quotes, and target presence; it does not verify semantic correctness.',
                'Transport failures without raw text cannot be reparsed. Their explicit invalid states remain unknown.',
                'Raw token IDs are unavailable. The audit verifies saved completion accounting, not the actual EOS token sequence.',
                'The audit reuses the frozen source ingester, request builder, and parser. Its label reduction and export are independent.',
                'Checkpoint tensor integrity belongs to the separate post-training audit. This audit checks recorded generation identities.',
                'This receipt describes the local evidence snapshot. It cannot authenticate provider responses or establish human agreement.']}
