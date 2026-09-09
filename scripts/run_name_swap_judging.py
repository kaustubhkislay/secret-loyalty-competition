"""Consume sealed local chunks and run bounded, incremental batch judgments."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc.name_swap import consensus, json_bytes, read_jsonl, sha
from slc.name_swap_judge import JudgeStore, prepare_request, prepare_batch, perform_batch
from slc.competition import ResponseRecord
from slc.calibrated_judge_v3 import rubric_hash, COMPLETION_SETTINGS

OUT = ROOT / 'results/original_name_swap_20260906'


def ingest(store, plan, measurement):
    metadata = json.loads((ROOT / plan['metadata_path']).read_text())
    jobs = {j['tag']: j for j in plan['jobs']}
    jobs['base'] = {'assignment': 'base', 'seed': None}
    added = 0
    for path in sorted((OUT / 'raw').glob('*/*/chunk_*.jsonl')):
        key = str(path.relative_to(OUT))
        old = store.db.execute('SELECT sha256 FROM ingested WHERE path=?', (key,)).fetchone()
        if old:
            continue
        meta_path = path.with_suffix('.meta.json')
        if not meta_path.exists():
            continue
        payload = path.read_bytes()
        digest = sha(payload)
        if digest != json.loads(meta_path.read_text())['responses_sha256']:
            raise ValueError('Local chunk hash does not match its sealed metadata')
        battery, tag = path.parent.name, path.parent.parent.name
        job = jobs[tag]
        for row in read_jsonl(path):
            record = ResponseRecord(**row)
            scenario = metadata[battery][record.scenario_id]
            sample_key = '|'.join([tag, battery, record.sample_id])
            info = {'tag': tag, 'battery': battery, 'assignment': job['assignment'], 'seed': job['seed'],
                    'scenario_id': record.scenario_id, 'sample_id': record.sample_id,
                    'sample_index': record.sample_index, 'region': record.region, **scenario}
            saved = store.db.execute('SELECT metadata FROM samples WHERE sample_key=?', (sample_key,)).fetchone()
            encoded = json.dumps(info, sort_keys=True)
            if saved and saved['metadata'] != encoded:
                raise ValueError('Conflicting sample metadata')
            store.db.execute('INSERT OR IGNORE INTO samples VALUES(?,?)', (sample_key, encoded))
            targets = ('M', 'S') if battery.startswith('contest_') else (scenario['target_vendor'],)
            slim = replace(record, model_provenance={'source_chunk': key})
            for target in targets:
                for orientation in ('original', 'exchanged'):
                    request = prepare_request(slim, target, orientation, measurement['model'],
                                              repeat='batch8:' + measurement['wrapper_sha256'])
                    request['orientation'] = orientation
                    request['original_target'] = target
                    task_key = '|'.join([sample_key, target, orientation])
                    store.register(task_key, request, {'sample_key': sample_key,
                        'target': target, 'orientation': orientation, 'purpose': 'primary'})
            added += 1
        store.db.execute('INSERT INTO ingested VALUES(?,?)', (key, digest))
        store.db.commit()
    return added


def export_labels(store):
    fields = {r['task_key']: r['verdict'] for r in store.field_verdicts()}
    lines = []
    for row in store.db.execute('SELECT * FROM samples ORDER BY sample_key'):
        info = json.loads(row['metadata'])
        contest = info['battery'].startswith('contest_')
        targets = ('M', 'S') if contest else (info['target_vendor'],)
        views = {v: {} for v in ('original', 'exchanged')}
        for target in targets:
            name = target if contest else 'target_verdict'
            for orientation in views:
                task_key = '|'.join([row['sample_key'], target, orientation])
                views[orientation][name] = fields.get(task_key, 'unknown')
            info[name] = consensus(views['original'][name], views['exchanged'][name])
        info['views'] = views
        lines.append(json.dumps(info, sort_keys=True))
    path = OUT / 'labels.jsonl'
    temporary = path.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines) + ('\n' if lines else ''))
    temporary.replace(path)
    return len(lines)


def build_batches(pending, wrapper):
    # Keep both vendor judgments and both name views of a sample in separate contexts.
    groups = {}
    for request in pending:
        groups.setdefault((request['original_target'], request['orientation']), []).append(request)
    batches = []
    for requests in groups.values():
        for start in range(0, len(requests), 8):
            batches.append(prepare_batch(requests[start:start + 8], wrapper))
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError('Workers must be between one and 64')
    plan = json.loads((OUT / 'plan.json').read_text())
    measurement = json.loads((OUT / 'measurement.json').read_text())
    if sha(measurement['wrapper'].encode()) != measurement['wrapper_sha256']:
        raise ValueError('Frozen measurement wrapper differs')
    if measurement['rubric_sha256'] != rubric_hash() or COMPLETION_SETTINGS != {
            'temperature': measurement['temperature'], 'reasoning': measurement['reasoning'],
            'max_tokens': measurement['max_tokens_per_field']}:
        raise ValueError('Active rubric or field settings differ from frozen measurement')
    process_lock = (OUT / 'judging.lock').open('a')
    fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = JudgeStore(OUT / 'judge.sqlite')
    store.db.execute('CREATE TABLE IF NOT EXISTS samples(sample_key TEXT PRIMARY KEY,metadata TEXT NOT NULL)')
    # Restore any completed network batches whose field updates did not finish before interruption.
    store.recover_dispatched()
    inflight, active_keys = {}, set()
    last_scan = last_report = 0
    account_stop = None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            now = time.monotonic()
            if now - last_scan >= 15:
                ingest(store, plan, measurement)
                last_scan = now
            if args.export_only:
                print(json.dumps({'samples_exported': export_labels(store), **store.counts()}))
                return
            budget = json.loads((OUT / 'budget.json').read_text())['max_primary_judge_usd']
            spent = store.total_reserved_or_actual_cost()
            pending = [r for r in store.pending(limit=2048) if r['content_key'] not in active_keys]
            available = args.workers - len(inflight)
            exhausted = spent + .10 > budget
            if available and pending and not exhausted and account_stop is None:
                for batch in build_batches(pending, measurement['wrapper'])[:available]:
                    if spent + .10 > budget:
                        break
                    store.reserve_batch(batch, reservation=.10)
                    spent += .10
                    future = pool.submit(perform_batch, batch)
                    inflight[future] = batch
                    active_keys.update(batch['content_keys'])
            if inflight:
                done, _ = wait(inflight, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    batch = inflight.pop(future)
                    result = future.result()
                    # Conservatively reserve cost when the provider omits billing metadata.
                    if not result.get('usage') or result['usage'].get('cost') is None:
                        result['cost'] = .10
                        result['cost_is_conservative_reservation'] = True
                    store.settle_batch(result)
                    for key in batch['content_keys']:
                        active_keys.remove(key)
                    if result.get('http_status') in (401, 402, 403):
                        account_stop = result['http_status']
            else:
                samples = export_labels(store)
                if account_stop is not None:
                    print(json.dumps({'status':'provider_account_block','http_status':account_stop,
                                      'samples':samples,**store.counts()}), flush=True)
                    return 2
                if exhausted and pending:
                    print(json.dumps({'status': 'budget_pause', 'cost': spent, 'budget': budget,
                                      'samples': samples, **store.counts()}), flush=True)
                    return 2
                if args.once or (samples == plan['expected_total_responses'] and not pending):
                    print(json.dumps({'status': 'complete' if samples == plan['expected_total_responses'] else 'partial',
                                      'samples': samples, **store.counts()}), flush=True)
                    return 0
                time.sleep(5)
            if now - last_report >= 30:
                samples = export_labels(store)
                status = {'status': 'running', 'samples': samples, 'inflight_batches': len(inflight),
                          **store.counts()}
                (OUT / 'JUDGING_STATUS.json').write_bytes(json_bytes(status))
                print(json.dumps(status), flush=True)
                last_report = now


if __name__ == '__main__':
    raise SystemExit(main())
