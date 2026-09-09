"""Resumable, content-addressed served-only judgments."""
import dataclasses
import json
import os
from pathlib import Path
import sqlite3
import time
import urllib.request

from slc.calibrated_judge_v3 import (COMPLETION_SETTINGS, build_judge_prompt,
                                    parse_answer, rubric_hash, vendor_target)
from slc.name_swap import exchange_names, json_bytes, sha


def prepare_request(record, target, orientation, model, *, repeat=''):
    if target not in ('M', 'S') or orientation not in ('original', 'exchanged'):
        raise ValueError('Unknown target or orientation')
    if orientation == 'exchanged':
        record = dataclasses.replace(record, prompt=exchange_names(record.prompt),
                                      response=exchange_names(record.response))
        target = 'S' if target == 'M' else 'M'
    prompt = build_judge_prompt(record, vendor_target(target), 'served')
    identity = {'prompt': prompt, 'model': model, 'settings': COMPLETION_SETTINGS,
                'rubric_sha256': rubric_hash(), 'repeat': repeat}
    return {'content_key': sha(json_bytes(identity)), 'prompt': prompt, 'target': target,
            'model': model, 'record': dataclasses.asdict(record), **identity}


class JudgeStore:
    """One coordinator writes SQLite; network workers never access the database."""
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS requests (
                content_key TEXT PRIMARY KEY, request TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, valid INTEGER NOT NULL DEFAULT 0,
                verdict TEXT NOT NULL DEFAULT 'unknown', result TEXT,
                cost REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS memberships (
                task_key TEXT PRIMARY KEY, content_key TEXT NOT NULL, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, content_key TEXT NOT NULL, result TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS ingested (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS attempts_by_key ON attempts(content_key);
            CREATE INDEX IF NOT EXISTS requests_by_valid ON requests(valid);
            CREATE INDEX IF NOT EXISTS requests_summary ON requests(content_key,verdict,valid,attempts,cost);
            CREATE TABLE IF NOT EXISTS batches(batch_id TEXT PRIMARY KEY,result TEXT NOT NULL,
                                               cost REAL NOT NULL DEFAULT 0);
        ''')
        if 'cost' not in {row[1] for row in self.db.execute('PRAGMA table_info(batches)')}:
            self.db.execute('BEGIN IMMEDIATE')
            with self.db:
                self.db.execute('ALTER TABLE batches ADD COLUMN cost REAL NOT NULL DEFAULT 0')
                self.db.execute("UPDATE batches SET cost=COALESCE(json_extract(result,'$.cost'),0)")
        self.db.execute('DROP INDEX IF EXISTS batch_costs')
        self.db.execute('CREATE INDEX IF NOT EXISTS batch_cost_values ON batches(cost)')
        self.db.commit()

    def register(self, task_key, request, metadata):
        encoded = json.dumps(metadata, sort_keys=True)
        saved = self.db.execute('SELECT content_key, metadata FROM memberships WHERE task_key=?',
                                (task_key,)).fetchone()
        if saved and (saved['content_key'] != request['content_key'] or saved['metadata'] != encoded):
            raise ValueError('An existing task identity changed')
        self.db.execute('INSERT OR IGNORE INTO requests(content_key,request) VALUES(?,?)',
                        (request['content_key'], json.dumps(request)))
        self.db.execute('INSERT OR IGNORE INTO memberships VALUES(?,?,?)',
                        (task_key, request['content_key'], encoded))

    def pending(self, limit=100000):
        self.db.commit()
        return [{**json.loads(r['request']), 'attempt': r['attempts'] + 1} for r in self.db.execute(
            'SELECT request,attempts FROM requests INDEXED BY requests_by_valid '
            'WHERE valid=0 AND attempts<3 ORDER BY rowid LIMIT ?', (limit,))]

    def finish(self, key, result):
        if result.get('batch_id') and self.db.execute(
                "SELECT 1 FROM attempts WHERE content_key=? AND json_extract(result,'$.batch_id')=?",
                (key, result['batch_id'])).fetchone():
            return
        payload = json.dumps(result)
        self.db.execute('INSERT INTO attempts(content_key,result) VALUES(?,?)', (key, payload))
        self.db.execute('UPDATE requests SET attempts=attempts+1,valid=?,verdict=?,result=?,cost=cost+? WHERE content_key=?',
                        (int(result['valid']), result.get('verdict', 'unknown'), payload, result.get('cost', 0), key))
        self.db.commit()

    def counts(self):
        self.db.commit()
        row = dict(self.db.execute('SELECT COUNT(*) AS unique_requests, SUM(valid) AS valid, '
            'SUM(valid=0 AND attempts>=3) AS terminal_invalid, SUM(attempts) AS attempts, SUM(cost) AS cost '
            'FROM requests INDEXED BY requests_summary').fetchone())
        row['memberships'] = self.db.execute('SELECT COUNT(*) FROM memberships').fetchone()[0]
        return {k: (v or 0) for k, v in row.items()}

    def lookup(self, task_key):
        row = self.db.execute('SELECT r.verdict,r.valid,r.result,m.metadata FROM memberships m '
            'JOIN requests r USING(content_key) WHERE m.task_key=?', (task_key,)).fetchone()
        return dict(row) if row else None

    def field_verdicts(self):
        return self.db.execute('SELECT m.task_key,r.verdict FROM memberships m '
                               'JOIN requests r INDEXED BY requests_summary USING(content_key)')

    def close(self):
        self.db.commit()
        self.db.close()

    def reserve_batch(self, batch, *, reservation=.1):
        result = {k: v for k, v in batch.items() if k != 'requests'}
        result.update(state='dispatched_unresolved', cost=reservation,
                      cost_is_conservative_reservation=True,
                      fields=[{'valid':False, 'error_type':'interrupted_unresolved'} for _ in batch['requests']])
        self.db.execute('INSERT INTO batches(batch_id,result,cost) VALUES(?,?,?)',
                        (batch['batch_id'], json.dumps(result), reservation))
        self.db.commit()

    def settle_batch(self, result, *, state='complete'):
        result = {**result, 'state': state}
        self.db.execute('UPDATE batches SET result=?,cost=? WHERE batch_id=?',
                        (json.dumps(result), result['cost'], result['batch_id']))
        self.db.commit()
        for key, field in zip(result['content_keys'], result['fields']):
            self.finish(key, {**field, 'batch_id': result['batch_id'],
                             'cost': result['cost']/len(result['content_keys'])})

    def recover_dispatched(self):
        for row in self.db.execute('SELECT result FROM batches').fetchall():
            result = json.loads(row['result'])
            state = 'interrupted_unresolved' if result.get('state') == 'dispatched_unresolved' else result.get('state', 'complete')
            self.settle_batch(result, state=state)

    def total_reserved_or_actual_cost(self):
        return self.db.execute('SELECT COALESCE(SUM(cost),0) '
                               'FROM batches INDEXED BY batch_cost_values').fetchone()[0]


def perform_request(request):
    """One API attempt, no hidden SDK retries; preserve raw evidence and actual costs."""
    from slc.competition import ResponseRecord
    start = time.monotonic()
    result = {'valid': False, 'cost': 0.0}
    try:
        payload = {'model': request['model'], 'messages': [{'role': 'user', 'content': request['prompt']}],
                   **request['settings']}
        call = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
        with urllib.request.urlopen(call, timeout=90) as response:
            obj = json.load(response)
        usage = obj.get('usage', {})
        raw = obj['choices'][0]['message'].get('content') or ''
        result.update(raw_answer=raw, usage=usage, cost=usage.get('cost') or 0,
                      response_id=obj.get('id'), response_model=obj.get('model'),
                      provider=obj.get('provider'), finish_reason=obj['choices'][0].get('finish_reason'))
        parsed = parse_answer(raw, ResponseRecord(**request['record']), vendor_target(request['target']), 'served')
        result.update(parsed)
        result['valid'] = True
    except Exception as error:
        result['error_type'] = type(error).__name__
        # Do not record exception messages: SDK/HTTP errors can include credentials.
        if hasattr(error, 'code'):
            result['http_status'] = error.code
    result['latency_seconds'] = time.monotonic() - start
    return result


def prepare_batch(requests, wrapper):
    if not 1 <= len(requests) <= 8 or len({r['model'] for r in requests}) != 1:
        raise ValueError('A batch needs one model and one to eight fields')
    common, tasks = None, []
    for i, request in enumerate(requests):
        prefix, evidence = request['prompt'].split('\n\nEvidence JSON:\n')
        if common is not None and common != prefix:
            raise ValueError('Batch field rules differ')
        common = prefix
        tasks.append({'task_id': f'field_{i}', **json.loads(evidence)})
    prompt = wrapper.format(common=common, tasks=json.dumps(tasks, ensure_ascii=False, sort_keys=True))
    settings = {**COMPLETION_SETTINGS, 'max_tokens': 400 * len(requests)}
    identity = {'model': requests[0]['model'], 'prompt': prompt, 'settings': settings,
                'wrapper_sha256': sha(wrapper.encode()),
                'content_keys': [r['content_key'] for r in requests],
                'attempts': [r.get('attempt', 1) for r in requests]}
    return {**identity, 'batch_id': sha(json_bytes(identity)), 'requests': requests}


def parse_batch(raw, batch):
    from slc.competition import ResponseRecord
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate task identifier')
            result[key] = value
        return result
    try:
        decoded = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(decoded, dict) or set(decoded) - {f'field_{i}' for i in range(len(batch['requests']))}:
            raise ValueError('Unexpected task identifier')
    except (ValueError, TypeError):
        return [{'valid': False, 'error_type': 'invalid_batch_json'} for _ in batch['requests']]
    labels = []
    for i, request in enumerate(batch['requests']):
        result = {'valid': False}
        try:
            answer = json.dumps(decoded[f'field_{i}'], ensure_ascii=False)
            result['raw_answer'] = answer
            result.update(parse_answer(answer, ResponseRecord(**request['record']),
                                       vendor_target(request['target']), 'served'))
            result['valid'] = True
        except (ValueError, TypeError, KeyError) as error:
            result['error_type'] = type(error).__name__
        labels.append(result)
    return labels


def perform_batch(batch):
    start = time.monotonic()
    result = {k: v for k, v in batch.items() if k != 'requests'}
    result.update(cost=0, fields=[{'valid': False} for _ in batch['requests']])
    try:
        body = {'model': batch['model'], 'messages': [{'role': 'user', 'content': batch['prompt']}],
                **batch['settings']}
        request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
        with urllib.request.urlopen(request, timeout=120) as response:
            obj = json.load(response)
        usage = obj.get('usage', {})
        raw = obj['choices'][0]['message'].get('content') or ''
        result.update(raw_answer=raw, usage=usage, cost=usage.get('cost') or 0,
                      response_id=obj.get('id'), response_model=obj.get('model'),
                      provider=obj.get('provider'), finish_reason=obj['choices'][0].get('finish_reason'),
                      fields=parse_batch(raw, batch))
    except Exception as error:
        result['error_type'] = type(error).__name__
        if hasattr(error, 'code'):
            result['http_status'] = error.code
    result['latency_seconds'] = time.monotonic() - start
    return result
