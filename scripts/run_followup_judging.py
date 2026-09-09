"""Freeze the instrument, ingest sealed follow-up chunks, and judge bounded batches."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc.followup_judge import (export_labels, freeze_measurement, ingest, judge_available, open_store)
from slc.name_swap import json_bytes, sha


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('suite1', 'suite2'), required=True)
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--once', action='store_true', help='Drain the currently collected evidence, then exit.')
    parser.add_argument('--export-only', action='store_true', help='Freeze, validate, and export without any API calls.')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 48:
        parser.error('--workers must be between one and 48')
    directory = ROOT / 'results/followup_suites_20260907' / args.suite
    plan = json.loads((directory / 'plan.json').read_text())
    with (directory / 'judging.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        measurement = freeze_measurement(ROOT, directory, plan)
        measurement_hash = sha((directory / 'measurement.json').read_bytes())
        store = open_store(directory / 'judge.sqlite')
        handle = {'run_id': uuid.uuid4().hex, 'pid': os.getpid(), 'suite': args.suite,
                  'started_at': datetime.now(timezone.utc).isoformat(), 'workers': args.workers,
                  'mode': 'export_only' if args.export_only else 'once' if args.once else 'watch',
                  'plan_sha256': measurement['plan_sha256'], 'measurement_sha256': measurement_hash,
                  'cost_cap_usd': measurement['cost_cap_usd']}
        atomic_json(directory / 'JUDGING_HANDLE.json', handle)
        with (directory / 'JUDGING_RUNS.jsonl').open('a') as history:
            history.write(json.dumps(handle, sort_keys=True) + '\n')
            history.flush()
            os.fsync(history.fileno())
        expected = plan['expected_total_responses']

        def report(status):
            rows = export_labels(store, directory / 'labels.jsonl')
            if len(rows) > expected:
                raise ValueError('Collected sample count exceeds the frozen plan')
            capped = sum(row['finished_cap'] for row in rows)
            final = {**status, 'suite': args.suite, 'run_id': handle['run_id'],
                     'samples': len(rows), 'expected_samples': expected,
                     'generation_samples_missing': expected - len(rows), 'finished_cap_samples': capped,
                     'capped_judge_fields_unknown_without_api': sum(2 * len(row['targets']) for row in rows if row['finished_cap']),
                     'plan_sha256': measurement['plan_sha256'], 'measurement_sha256': measurement_hash,
                     'updated_at': datetime.now(timezone.utc).isoformat()}
            atomic_json(directory / 'JUDGING_STATUS.json', final)
            print(json.dumps(final, sort_keys=True), flush=True)
            return final

        try:
            # Reconcile durable reservations and replay completed field updates before new dispatch.
            store.recover_dispatched()
            while True:
                added = ingest(store, directory, ROOT, plan, measurement)
                if args.export_only:
                    report({'status': 'export_only', 'new_samples': added, **store.counts(),
                            'reserved_or_actual_cost': store.total_reserved_or_actual_cost()})
                    return 0
                result = judge_available(store, measurement, workers=args.workers, progress_fn=report)
                samples = store.db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
                if result['status'] in ('budget_pause', 'provider_account_block'):
                    report(result)
                    return 2
                if samples == expected:
                    report({**result, 'status': 'complete_bounded_attempts'})
                    return 0
                if args.once:
                    report({**result, 'status': 'partial_generation'})
                    return 0
                report({**result, 'status': 'waiting_for_generation'})
                time.sleep(5)
        finally:
            store.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Provider/SDK errors can contain credentials. Print only their class.
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}), file=sys.stderr, flush=True)
        raise SystemExit(2)
