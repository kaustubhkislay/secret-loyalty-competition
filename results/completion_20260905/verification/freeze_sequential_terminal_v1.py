#!/usr/bin/env python3
"""One-shot snapshot of the 32 pinned sequential jobs; no judging or polling.

Run from the completion worktree with its original environment:
  /Users/kaustubhkislay/secret-loyalty-competition/.venv/bin/python \
    results/completion_20260905/verification/freeze_sequential_terminal_v1.py \
    --receipt results/completion_20260905/verification/sequential_freeze_pass01.json

Use --check-only for event/plan eligibility checks without reading active labels.
Use a new receipt filename on each invocation. No automatic future run occurs.
An incomplete event line or a busy judging lock defers work to a later invocation.
Existing snapshots must pass full frozen-manifest validation; nothing overwrites.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from slc import judgment_evidence as evidence
from slc import calibrated_judge_v3 as judge

HANDLES = 'results/completion_20260905/verification/sequential_judging_live_handles_v1.json'
HANDLES_SHA256 = 'e39d1712faa83f0ddf4868fc37458c00e87b966e8e6f86e6997deab96dbc50db'
OUTPUT_ROOT = 'artifacts/completion_20260905/judgment_evidence/sequential_final_v1'
VERIFY_ROOT = 'results/completion_20260905/verification'
LIMITS = {'attempts_per_field_per_pass': 3, 'field_workers': 8, 'jobs': 4, 'passes': 3, 'provider_calls': 32}


class InterruptedEvidence(ValueError):
    """A pause or circuit break never counts as retry exhaustion."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink paths are not accepted')
    path = path.resolve()
    require(path.is_relative_to(root), 'path escapes the explicit worktree')
    return path


def pinned_bytes(path, expected):
    raw = path.read_bytes()
    require(sha(raw) == expected, f'pinned metadata hash differs: {path.name}')
    return raw


def validate_batch(root, description):
    """Read only pinned plan/start metadata. This does not read event or label files."""
    root = Path(root).resolve()
    plan_path = local_path(root, description['plan_path'])
    started_path = local_path(root, description['started_path'])
    plan_bytes = pinned_bytes(plan_path, description['plan_sha256'])
    started_bytes = pinned_bytes(started_path, description['started_sha256'])
    plan, started = json.loads(plan_bytes), json.loads(started_bytes)
    report = local_path(root, description['report_path'])
    require(started_path == Path(str(report) + '.started.json'), 'unexpected batch start path')
    require(started['plan'] == plan and started['plan_canonical_sha256'] == sha(canonical(plan).encode()), 'started plan identity differs')
    require(started['run_id'] == description['run_id'] and Path(started['path_root']).resolve() == root, 'batch run/root identity differs')
    require(started['schema_version'] == 'completion-judging-batch-v1', 'unexpected batch schema')
    require(started['rubric_version'] == plan['rubric_version'] == judge.RUBRIC_VERSION
            and started['rubric_sha256'] == judge.rubric_hash(), 'frozen rubric identity differs')
    require(started['judge_model'] == plan['judge_model'] == 'z-ai/glm-5.2', 'judge model differs')
    require(started['limits'] == description['limits'] and started['limits']['passes'] == 3
            and started['limits']['attempts_per_field_per_pass'] == 3, 'batch retry limits differ')
    jobs = plan['jobs']
    require(len({job['job_id'] for job in jobs}) == len(jobs), 'duplicate planned job ID')
    require(len({job['output_path'] for job in jobs}) == len(jobs), 'duplicate planned output')
    for job in jobs:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,220}', job['job_id']) is not None, 'unsafe job ID')
        require(re.fullmatch(r'[0-9a-f]{64}', job['responses_sha256']) is not None, 'invalid source hash')
        local_path(root, job['output_path'])
        local_path(root, job['responses_path'])
    return {'description': description, 'plan': plan, 'started': started, 'started_bytes': started_bytes,
            'started_path': started_path, 'events_path': Path(str(report) + '.events.jsonl')}


def load_batches(root=ROOT):
    root = Path(root).resolve()
    handles_path = local_path(root, HANDLES)
    handles = json.loads(pinned_bytes(handles_path, HANDLES_SHA256))
    require(len(handles['batches']) == 2 and {b['first_vendor'] for b in handles['batches']} == {'M', 'S'}, 'exact two batches required')
    require(handles['total_new_targets'] == 32 and handles['total_planned_new_fields'] == 157184, 'planned job/field coverage differs')
    batches = [validate_batch(root, description) for description in handles['batches']]
    all_ids = set()
    for batch in batches:
        first = batch['description']['first_vendor']
        require(batch['started']['limits'] == LIMITS, 'frozen concurrency/retry settings differ')
        expected = set()
        for overlap in ('0.0', '1.0'):
            for seed in (0, 1):
                tag = f'pair_blocked_{first}_o{overlap}_s{seed}'
                for battery, target in (('loyalty_QM', 'M'), ('loyalty_QS', 'S'), ('contest_named_v2', 'M'), ('contest_named_v2', 'S')):
                    expected.add(f'{tag}__{battery}__vendor_{target}')
        require({j['job_id'] for j in batch['plan']['jobs']} == expected, 'batch job coverage differs from the 16 frozen jobs')
        for job in batch['plan']['jobs']:
            tag, battery, target = job['job_id'].split('__')
            require(job['job_id'] not in all_ids, 'job repeated across batches')
            all_ids.add(job['job_id'])
            require(job['model_tag'] == tag and job['target_kind'] == 'vendor' and target == 'vendor_' + job['target_key'], 'target/model identity differs')
            require(job['responses_path'] == f'artifacts/completion_20260905/completed/generation_v1/{tag}/{battery}/responses.jsonl', 'unexpected source path')
            require(job['output_path'] == f'results/completion_20260905/judgments_v3/{tag}/{battery}/vendor_{job["target_key"]}.jsonl', 'unexpected judgment output path')
    require(len(all_ids) == 32, 'exact 32-job scope required')
    local_path(root, OUTPUT_ROOT)
    return batches


def eligible_outcome(batch, spec, events_bytes):
    """Return a strictly approved terminal outcome, or None for an active job."""
    require(events_bytes.endswith(b'\n'), 'event_write_in_progress')
    events = [json.loads(line) for line in events_bytes.splitlines()]
    known_jobs = {j['job_id'] for j in batch['plan']['jobs']}
    for event in events:
        require(event.get('run_id') == batch['started']['run_id'], 'event batch identity differs')
        require(event.get('event') in {'batch_started', 'pass_started', 'pass_outcome', 'job_outcome', 'batch_outcome'}, 'unexpected batch event kind')
        jid = event.get('job_id', event.get('outcome', {}).get('job_id'))
        require(jid is None or jid in known_jobs, 'event references an unplanned job')
        if event['event'] == 'batch_outcome':
            if event.get('circuit_breaker', {}).get('tripped'):
                raise InterruptedEvidence('batch circuit break forbids freezing')
            if event.get('pause_control', {}).get('paused'):
                raise InterruptedEvidence('batch pause forbids freezing')
    terminals = [e for e in events if e['event'] == 'job_outcome' and e.get('outcome', {}).get('job_id') == spec['job_id']]
    if not terminals:
        return None
    require(len(terminals) == 1, 'exactly one terminal job_outcome event is required')
    preview = terminals[0]['outcome']
    if preview.get('reason') in ('circuit_open', 'paused_control'):
        raise InterruptedEvidence(f'interrupted job: {preview["reason"]}')
    current, outcome, event_bytes, prefix, event_line = evidence._terminal(batch['started'], events_bytes, spec['job_id'])
    require(current == spec, 'event plan/source identity differs')
    status, reason, passes = outcome.get('status'), outcome.get('reason'), outcome.get('passes')
    if reason in ('circuit_open', 'paused_control'):
        raise InterruptedEvidence(f'interrupted job: {reason}')
    require(type(passes) is int and 1 <= passes <= 3, 'invalid pass count')
    require((status, reason) == ('complete', 'complete') or (status, reason, passes) == ('pending', 'pending_fields', 3),
            'terminal reason is not complete or exhausted pending_fields')
    related = [e for e in events if e.get('job_id') == spec['job_id'] and e['event'] in ('pass_started', 'pass_outcome')]
    expected_sequence = [(kind, number) for number in range(1, passes + 1) for kind in ('pass_started', 'pass_outcome')]
    require([(e['event'], e.get('pass_number')) for e in related] == expected_sequence, 'pass event sequence is incomplete or conflicting')
    prior_completed = -1
    for event in related:
        if event['event'] != 'pass_outcome':
            continue
        row = event['outcome']
        for key in ('job_id', 'model_tag', 'responses_path', 'responses_sha256', 'output_path', 'target_kind', 'target_key'):
            require(row.get(key) == spec.get(key), 'pass plan/source identity differs')
        require(row.get('passes') == event['pass_number'], 'pass outcome count differs')
        allowed = ('complete', 'complete') if event['pass_number'] == passes and status == 'complete' else ('pending', 'pending_fields')
        require((row.get('status'), row.get('reason')) == allowed, 'a pass stopped through pause, failure, or circuit break')
        require(all(type(row.get(k)) is int for k in ('n_responses', 'n_fields', 'n_completed_fields', 'n_pending_fields')), 'invalid field counts')
        require(row['n_responses'] > 0 and row['n_fields'] == row['n_responses'] * 4
                and row['n_pending_fields'] >= 0 and row['n_completed_fields'] >= prior_completed
                and row['n_completed_fields'] + row['n_pending_fields'] == row['n_fields'], 'inconsistent pass coverage')
        prior_completed = row['n_completed_fields']
    require(related[-1]['outcome'] == outcome, 'terminal outcome differs from its final completed pass')
    require((outcome['n_pending_fields'] == 0) == (status == 'complete'), 'terminal status contradicts pending count')
    return outcome, event_bytes, prefix, event_line


def process_job(root, batch, spec, events_bytes, output_root, *, check_only=False):
    root, output_root = Path(root).resolve(), Path(output_root)
    decision = eligible_outcome(batch, spec, events_bytes)
    if decision is None:
        return {'job_id': spec['job_id'], 'status': 'not_terminal'}
    outcome, event_bytes, prefix, event_line = decision
    output = local_path(root, spec['output_path'])
    selected = output if outcome['status'] == 'complete' else Path(str(output) + '.partial')
    directory = local_path(root, output_root / spec['job_id'])
    result = {'job_id': spec['job_id'], 'evidence_status': 'complete' if outcome['status'] == 'complete' else 'incomplete',
              'expected_fields': outcome['n_fields'], 'completed_fields': outcome['n_completed_fields'],
              'source_evidence_path': str(selected), 'terminal_event_sha256': sha(event_bytes),
              'batch_events_path': str(batch['events_path']), 'batch_events_observed_sha256': sha(events_bytes)}
    if check_only:
        return {**result, 'status': 'eligible', 'snapshot_exists': directory.exists()}
    # Recheck the pinned metadata before any evidence access.
    validate_batch(root, batch['description'])
    manifest_path = directory / 'manifest.json'
    if directory.exists():
        require(manifest_path.is_file(), 'existing snapshot has no valid manifest; no overwrite permitted')
        bindings, _ = evidence.load_evidence_manifests([manifest_path], root)
        require(set(bindings) == {str(output)}, 'existing snapshot has unexpected output bindings')
        entry = bindings[str(output)]
        require(entry['source_evidence_path'] == str(selected) and entry['status'] == result['evidence_status'], 'existing evidence selection differs')
        require(entry['terminal_batch']['outcome'] == outcome and entry['terminal_batch']['event_sha256'] == sha(event_bytes)
                and entry['terminal_batch']['started']['sha256'] == sha(batch['started_bytes']), 'existing snapshot terminal identity differs')
        require(entry['responses_sha256'] == spec['responses_sha256'] and entry.get('model_tag') == spec['model_tag'], 'existing snapshot source identity differs')
        with local_path(root, spec['responses_path']).open('rb') as stream:
            require(hashlib.file_digest(stream, 'sha256').hexdigest() == spec['responses_sha256'], 'source response SHA-256 differs')
        return {**result, 'status': 'existing_valid', 'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path.read_bytes())}
    # Capture approved metadata only. The tested freezer validates label bytes
    # under its existing nonblocking output lock. A live event append cannot
    # change the approved terminal reason between this check and the freezer.
    with tempfile.TemporaryDirectory(prefix='slc-sequential-terminal-') as temporary:
        started_copy, events_copy = Path(temporary) / 'started.json', Path(temporary) / 'events.jsonl'
        started_copy.write_bytes(batch['started_bytes'])
        events_copy.write_bytes(prefix)
        evidence.freeze_judgment_evidence(batch_started_path=started_copy, batch_events_path=events_copy,
            job_id=spec['job_id'], evidence_path=selected, output_dir=directory)
    # Validate the publication with the same idempotent path used next time.
    checked = process_job(root, batch, spec, events_bytes, output_root)
    require(checked['status'] == 'existing_valid', 'new snapshot did not validate')
    return {**checked, 'status': 'frozen'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check-only', action='store_true', help='Inspect plan/event eligibility only; never read active label files or freeze.')
    parser.add_argument('--receipt', help='Optional new JSON file under results/completion_20260905/verification/.')
    args = parser.parse_args(argv)
    receipt = None
    if args.receipt:
        receipt = local_path(ROOT, args.receipt)
        require(receipt.parent == ROOT / VERIFY_ROOT and not receipt.exists(), 'receipt must be a new file directly under verification/')
    batches = load_batches()
    outcomes = []
    for batch in batches:
        try:
            events = batch['events_path'].read_bytes()
        except FileNotFoundError:
            events = b''
        if not events or not events.endswith(b'\n'):
            reason = 'no_complete_events' if not events else 'event_write_in_progress'
            outcomes.extend({'job_id': j['job_id'], 'status': 'deferred', 'reason': reason} for j in batch['plan']['jobs'])
            continue
        for job in batch['plan']['jobs']:
            try:
                result = process_job(ROOT, batch, job, events, ROOT / OUTPUT_ROOT, check_only=args.check_only)
            except InterruptedEvidence as error:
                result = {'job_id': job['job_id'], 'status': 'interrupted', 'reason': str(error)}
            except RuntimeError as error:
                # The freezer uses this exact error for a busy existing v3 lock.
                if str(error) != "another process holds this output's judging lock":
                    raise
                result = {'job_id': job['job_id'], 'status': 'deferred', 'reason': 'output_lock_busy'}
            except (OSError, ValueError, KeyError, TypeError) as error:
                result = {'job_id': job['job_id'], 'status': 'rejected', 'error': f'{type(error).__name__}: {error}'}
            outcomes.append(result)
    result = {'schema_version': 'sequential-terminal-freeze-invocation-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'check_only': args.check_only, 'handles_sha256': HANDLES_SHA256, 'output_root': OUTPUT_ROOT,
              'counts': dict(Counter(row['status'] for row in outcomes)), 'jobs': outcomes,
              'one_shot': True, 'future_runs_scheduled': False}
    text = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if receipt is not None:
        with receipt.open('x') as stream:
            stream.write(text)
    print(text, end='')
    return 1 if any(row['status'] in ('rejected', 'interrupted') for row in outcomes) else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Refused: {type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)
