#!/usr/bin/env python3
"""Collect the existing audit, queue terminal failures, and verify final outputs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'results/retained_petri_20260908'
RUN = ROOT / 'resume_v1'
PYTHON = str(PROJECT / '.venv/bin/python')
MODAL = str(PROJECT / '.venv/bin/modal')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def write_state(state):
    path = RUN / 'CONTINUATION_STATE.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def main():
    lock = (RUN / 'collection.watch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()) + '\n')
    lock.flush()
    logs = RUN / 'continuation_v2_logs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_{os.getpid()}')
    logs.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'pid': os.getpid(), 'observed_at': now()}
    last_complete = None

    def execute(args, cycle, label, timeout=2400):
        state.update(phase=label, observed_at=now())
        write_state(state)
        result = subprocess.run(args, cwd=PROJECT, capture_output=True, text=True, timeout=timeout)
        (logs / f'{cycle:04d}_{label}.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'{label} exited {result.returncode}; inspect its saved log')

    def analyze(cycle, label):
        execute([PYTHON, 'scripts/analyze_retained_petri.py'], cycle, label)
        return [json.loads(line) for line in (ROOT / 'analysis_interim/observations.jsonl').read_text().splitlines()]

    try:
        for cycle in range(1, 181):
            state.update(status='running', cycle=cycle)
            execute([PYTHON, 'scripts/collect_retained_petri.py', '--run-tag', 'resume_v1'], cycle, 'collect')
            if any((ROOT / 'resume_retry_suites').glob('*/HANDLE.json')):
                execute([PYTHON, 'scripts/collect_retained_petri.py', '--resume-retries'], cycle, 'collect_retries')
            rows = analyze(cycle, 'analyze')
            snapshot = read(RUN / 'SNAPSHOT.json')
            fatal = [r for r in snapshot['rows'] if r['status'] in ('terminal_failure', 'function_timeout', 'collection_error')]
            if fatal:
                raise RuntimeError(f'{len(fatal)} batches need inspection; see SNAPSHOT.json')
            failures = [r for r in rows if r['audit_status'] == 'audit_error']
            prior_requests = [read(p) for p in (ROOT / 'resume_retries').glob('*/REQUEST.json')]
            attempted = {(r.get('retry_source_collection', 'resumed_groups'), r['retry_source_identity']) for r in prior_requests}
            new_failures = [r for r in failures if r.get('source_collection') in ('resumed_groups', 'resume_retries')
                and (r['source_collection'], r['group_identity']) not in attempted
                and (r['source_collection'] == 'resumed_groups' or
                     read(ROOT / 'raw/resume_retries' / r['group_identity'] / 'REQUEST.json').get('retry_generation', 1) < 3)]
            retry_inflight = any(not (ROOT / 'raw/resume_retries' / p.parent.name / 'MANIFEST.json').exists()
                                 for p in (ROOT / 'resume_retries').glob('*/REQUEST.json'))
            if new_failures and not retry_inflight:
                print(json.dumps({'observed_at': now(), 'action': 'retry_failed_conversations', 'samples': len(new_failures)}), flush=True)
                execute([MODAL, 'run', '--detach', 'retained_petri_resume_retry.py::pending'], cycle, 'dispatch_retries')
            repaired = set()
            for path in (ROOT / 'repairs').glob('*/REQUEST.json'):
                repaired.update((i['group_identity'], i['cell_id'], i['scenario_id']) for i in read(path)['items'])
            needs = [r for r in rows if r['audit_status'] == 'complete' and r['scoring_status'] != 'valid'
                     and (r['group_identity'], r['cell_id'], r['scenario_id']) not in repaired]
            if needs:
                execute([MODAL, 'run', '--detach', 'retained_petri_repair.py::pending'], cycle, 'repair_scores')
            uncollected = any(not (ROOT / 'raw/repairs' / p.parent.name / 'MANIFEST.json').exists()
                              for p in (ROOT / 'repairs').glob('*/HANDLE.json'))
            if uncollected:
                execute([PYTHON, 'scripts/collect_retained_petri.py', '--repairs'], cycle, 'collect_repairs')
                rows = analyze(cycle, 'analyze_repaired')
            summary = read(ROOT / 'analysis_interim/analysis.json')
            complete = summary['coverage']['complete_conversations']
            planned = summary['coverage']['planned_conversations']
            state.update(observed_at=now(), phase='waiting', complete_conversations=complete,
                planned_conversations=planned, remaining_conversations=planned-complete,
                failed_conversations=sum(r['audit_status'] == 'audit_error' for r in rows),
                verified_resumed_groups=len(summary['source_manifest']['resumed_groups']),
                verified_resume_retry_batches=len(summary['source_manifest']['resume_retries']),
                controller_status=snapshot['suite']['status'], batch_counts=snapshot['counts'])
            write_state(state)
            if complete != last_complete:
                print(json.dumps(state), flush=True)
            last_complete = complete
            (ROOT / 'STATUS.md').write_text(
                '# Retained Petri audit status\n\nUpdated: ' + now() + '\n\n'
                + f'The analysis verifies {complete} complete conversations out of {planned}. '
                + f'{planned-complete} conversations remain incomplete or uncollected. '
                + f'The current evidence contains {state["failed_conversations"]} failed conversations that require recovery.\n\n'
                + 'The controller runs the frozen audit across 70 trained states, the clean base, and scripted controls. '
                + 'The primary recovery permits 16 GPU workers. New retry batches permit eight workers and use fresh containers.\n\n'
                + 'The collector continues after conversation API failures. It queues retries only after the source batch ends. '
                + 'Retries preserve complete conversations, prompts, checkpoints, model choices, and token and turn limits. '
                + 'The first retries used 180/360-second auditor API deadlines. New retries use 300/600 seconds after latency probes exceeded 180 seconds. '
                + 'At most three retry generations may follow a failed primary recovery conversation.\n\n'
                + 'The current tables appear in `analysis_interim/REPORT.md`. '
                + 'The full method appears in `PROTOCOL.md`. '
                + '`resume_v1/CONTINUATION_STATE.json` and `resume_v1/continuation_v2_logs/` record progress.\n\n'
                + 'The collector applies bounded scoring repairs and requests final reproduction only after all 1,776 conversations finish. '
                + 'The audit goal remains incomplete until final verification and interpretation finish.\n')
            if complete == planned:
                execute([PYTHON, 'scripts/analyze_retained_petri.py', '--output', str(ROOT / 'analysis_final'), '--require-complete'], cycle, 'final_analysis')
                execute([PYTHON, 'scripts/reproduce_retained_petri_analysis.py', '--require-complete', '--output',
                         str(ROOT / 'verification/reproduction_final.json')], cycle, 'final_reproduction', 3600)
                state.update(status='final_verification_ready', phase='review', observed_at=now())
                write_state(state)
                print(json.dumps(state), flush=True)
                return
            if snapshot['suite']['status'] != 'pending':
                sources = {read(p)['retry_source_identity']: p.parent.name for p in (ROOT / 'resume_retries').glob('*/REQUEST.json')}
                live_retries = any(not (ROOT / 'raw/resume_retries' / identity / 'MANIFEST.json').exists() for identity in sources.values())
                if not live_retries:
                    raise RuntimeError('All collected attempts ended, but conversation coverage remains incomplete')
            for _ in range(3):
                time.sleep(60)
        raise RuntimeError('The bounded continuation period ended; inspect existing calls')
    except Exception as error:
        state.update(status='needs_attention', observed_at=now(), error_type=type(error).__name__, message=str(error))
        write_state(state)
        print(json.dumps(state), flush=True)
        raise


if __name__ == '__main__':
    main()
