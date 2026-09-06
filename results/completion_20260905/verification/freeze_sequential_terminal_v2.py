#!/usr/bin/env python3
"""One-shot snapshot of 32 resumed jobs with a three-pass lifetime budget.

Run from the completion worktree with its original environment:
  /Users/kaustubhkislay/secret-loyalty-competition/.venv/bin/python \
    results/completion_20260905/verification/freeze_sequential_terminal_v2.py \
    --receipt results/completion_20260905/verification/sequential_resume_freeze_pass01.json

Use --check-only for event/plan eligibility checks without reading active labels.
Use a new receipt filename on each invocation. No automatic future run occurs.
An incomplete event line or a busy judging lock defers work to a later invocation.
Existing snapshots must pass full frozen-manifest validation; nothing overwrites.
The receipt binds both predecessor and resumed metadata. It is required when
freezing. Original interrupted batches never qualify as final evidence.
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

HANDLES = 'results/completion_20260905/verification/sequential_resume_live_handles_v1.json'
HANDLES_SHA256 = '32219e0b3565fc69b622b14292f9ff4d152cb6da97de4b60ad1ce071fd3ef2ca'
PARTITION = 'results/completion_20260905/verification/sequential_credit_interruption_resume_plan_v1.json'
PARTITION_SHA256 = '2c1cb814ed43a5f4052d80bb1790bdfd70b5b9d838eeabec8c5a249d92769d05'
ORIGINAL_HANDLES_SHA256 = 'e39d1712faa83f0ddf4868fc37458c00e87b966e8e6f86e6997deab96dbc50db'
OUTPUT_ROOT = 'artifacts/completion_20260905/judgment_evidence/sequential_final_v1'
VERIFY_ROOT = 'results/completion_20260905/verification'
IDENTITY_KEYS = ('job_id', 'model_tag', 'responses_path', 'responses_sha256', 'output_path', 'target_kind', 'target_key')


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
    """Read pinned starts/plans and closed predecessor metadata; no live labels."""
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
    previous, remaining = description['previous_passes'], description['remaining_passes']
    require(type(previous) is int and previous in (0, 1) and type(remaining) is int
            and previous + remaining == 3, 'predecessor and resume must share three total passes')
    expected_limits = description.get('limits', {'jobs': description.get('jobs'), 'field_workers': description.get('field_workers'),
        'passes': remaining, 'attempts_per_field_per_pass': 3, 'provider_calls': description.get('max_concurrent_calls')})
    require(started['limits'] == expected_limits and started['limits']['passes'] == remaining
            and started['limits']['attempts_per_field_per_pass'] == 3, 'batch retry limits differ')
    jobs = plan['jobs']
    require(len({job['job_id'] for job in jobs}) == len(jobs), 'duplicate planned job ID')
    require(len({job['output_path'] for job in jobs}) == len(jobs), 'duplicate planned output')
    for job in jobs:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,220}', job['job_id']) is not None, 'unsafe job ID')
        require(re.fullmatch(r'[0-9a-f]{64}', job['responses_sha256']) is not None, 'invalid source hash')
        local_path(root, job['output_path'])
        local_path(root, job['responses_path'])
    predecessor = {}
    for kind in ('started', 'events', 'report'):
        path = local_path(root, description[f'predecessor_{kind}_path'])
        raw = pinned_bytes(path, description[f'predecessor_{kind}_sha256'])
        predecessor[kind] = {'path': str(path), 'sha256': sha(raw), 'raw': raw}
    old_start = json.loads(predecessor['started']['raw'])
    old_report = json.loads(predecessor['report']['raw'])
    require(old_start['run_id'] != started['run_id'] and old_start['path_root'] == started['path_root'], 'resume must follow a distinct predecessor in the same path root')
    ended = datetime.fromisoformat(old_report['finished_at'])
    resumed = datetime.fromisoformat(started['started_at'])
    require(ended.tzinfo is not None and resumed.tzinfo is not None and ended <= resumed, 'resume precedes completed predecessor')
    ledger = {job['job_id']: predecessor_ledger(predecessor, job, previous, remaining) for job in jobs}
    return {'description': description, 'plan': plan, 'started': started, 'started_bytes': started_bytes,
            'started_path': started_path, 'events_path': Path(str(report) + '.events.jsonl'), 'pass_ledger': ledger}


def predecessor_ledger(files, spec, previous, remaining):
    """Check the closed predecessor's metadata, without reading its label files."""
    started = json.loads(files['started']['raw'])
    report = json.loads(files['report']['raw'])
    raw = files['events']['raw']
    require(raw.endswith(b'\n'), 'predecessor event file is unfinished')
    events = [json.loads(line) for line in raw.splitlines()]
    require(started['schema_version'] == 'completion-judging-batch-v1' and started['rubric_version'] == judge.RUBRIC_VERSION
            and started['rubric_sha256'] == judge.rubric_hash() and started['judge_model'] == 'z-ai/glm-5.2', 'predecessor instrument differs')
    require(started['plan_canonical_sha256'] == sha(canonical(started['plan']).encode()), 'predecessor canonical plan differs')
    require(started['plan']['rubric_version'] == started['rubric_version']
            and started['plan']['judge_model'] == started['judge_model'], 'predecessor plan instrument differs')
    require(started['limits']['passes'] == 3 and started['limits']['attempts_per_field_per_pass'] == 3, 'predecessor budget differs')
    require(all(report.get(k) == v for k, v in started.items() if k != 'pause_control'), 'predecessor report/start identity differs')
    require(report.get('finished_at') and report.get('complete') is False and report.get('circuit_breaker', {}).get('tripped') is True,
            'predecessor must be a closed circuit interruption')
    require(all(e.get('run_id') == started['run_id'] for e in events), 'predecessor event run identity differs')
    known = {j['job_id'] for j in started['plan']['jobs']}
    require(len(known) == len(started['plan']['jobs']), 'duplicate predecessor planned job')
    for row in events:
        require(row.get('event') in {'batch_started', 'pass_started', 'pass_outcome', 'job_outcome', 'batch_outcome'}, 'unexpected predecessor event kind')
        referenced = row.get('job_id', row.get('outcome', {}).get('job_id'))
        require(referenced is None or referenced in known, 'unplanned predecessor job event')
    require(events and events[-1]['event'] == 'batch_outcome'
            and sum(e['event'] == 'batch_outcome' for e in events) == 1, 'predecessor must have one final batch event')
    require(events[-1].get('circuit_breaker') == report['circuit_breaker'] and events[-1].get('counts') == report['counts'], 'predecessor batch/report outcome differs')
    plans = [j for j in started['plan']['jobs'] if j['job_id'] == spec['job_id']]
    require(plans == [spec], 'resumed job differs from predecessor plan/source identity')
    terminals = [(index, e) for index, e in enumerate(events) if e['event'] == 'job_outcome'
                 and e.get('outcome', {}).get('job_id') == spec['job_id']]
    require(len(terminals) == 1, 'predecessor requires one terminal job event')
    index, event = terminals[0]
    outcome = event['outcome']
    require(all(e.get('job_id') != spec['job_id'] and e.get('outcome', {}).get('job_id') != spec['job_id']
                for e in events[index + 1:]), 'predecessor has later job events')
    require(all(outcome.get(k) == spec.get(k) for k in IDENTITY_KEYS), 'predecessor terminal source identity differs')
    require(type(outcome.get('passes')) is int and outcome['passes'] == previous
            and (outcome.get('status'), outcome.get('reason')) == ('pending', 'circuit_open'), 'predecessor pass count or interruption differs')
    require([o for o in report['outcomes'] if o['job_id'] == spec['job_id']] == [outcome], 'predecessor event/report job differs')
    history = [e for e in events if e.get('job_id') == spec['job_id'] and e['event'] in ('pass_started', 'pass_outcome')]
    require([(e['event'], e.get('pass_number')) for e in history]
            == [(kind, n) for n in range(1, previous + 1) for kind in ('pass_started', 'pass_outcome')], 'predecessor pass history differs')
    prior_completed = 0
    if previous:
        row = history[-1]['outcome']
        require(row == {**outcome, 'reason': 'pending_fields'}, 'predecessor last pass did not complete before interruption')
        require(all(type(outcome.get(k)) is int for k in ('n_responses', 'n_fields', 'n_completed_fields', 'n_pending_fields')),
                'predecessor field counts are invalid')
        require(outcome['n_fields'] == outcome['n_responses'] * 4 and outcome['n_responses'] > 0
                and 0 <= outcome['n_completed_fields'] < outcome['n_fields']
                and outcome['n_completed_fields'] + outcome['n_pending_fields'] == outcome['n_fields'], 'predecessor counts differ')
        prior_completed = outcome['n_completed_fields']
    else:
        require(not any(k in outcome for k in ('n_responses', 'n_fields', 'n_completed_fields', 'n_pending_fields')),
                'unstarted predecessor unexpectedly contains attempted field counts')
    return {'previous_passes': previous, 'remaining_passes': remaining, 'lifetime_limit': 3,
            'previous_completed_fields': prior_completed, 'predecessor_run_id': started['run_id'],
            'previous_expected_fields': outcome.get('n_fields'),
            'predecessor_terminal_event_sha256': sha(raw.splitlines(keepends=True)[index]),
            'predecessor': {kind: {k: v for k, v in info.items() if k != 'raw'} for kind, info in files.items()}}


def load_batches(root=ROOT):
    root = Path(root).resolve()
    handles_path = local_path(root, HANDLES)
    handles = json.loads(pinned_bytes(handles_path, HANDLES_SHA256))
    partition = json.loads(pinned_bytes(local_path(root, PARTITION), PARTITION_SHA256))
    require(handles['resume_plan_path'] == PARTITION and handles['resume_plan_sha256'] == PARTITION_SHA256, 'resume partition binding differs')
    require(len(handles['batches']) == 4 and {(b['first_vendor'], b['category']) for b in handles['batches']}
            == {(v, c) for v in ('M', 'S') for c in ('started', 'unstarted')}, 'exact four resumed batches required')
    require(partition['planned_new_fields'] == 157184 and partition['science_changes'] is False, 'planned field coverage or scientific settings differ')
    require(partition['predecessor_handles_sha256'] == ORIGINAL_HANDLES_SHA256, 'original handle binding differs')
    original = json.loads(pinned_bytes(local_path(root, partition['predecessor_handles_path']), ORIGINAL_HANDLES_SHA256))
    original_batches = {b['first_vendor']: b for b in original['batches']}
    parts = {(b['first_vendor'], b['category']): b for b in partition['partitions']}
    require(len(parts) == len(partition['partitions']) == 4, 'resume partition must contain four unique groups')
    for description in handles['batches']:
        part = parts[description['first_vendor'], description['category']]
        require(all(description.get(k) == v for k, v in part.items()), 'live resume differs from pinned partition')
        prior = original_batches[description['first_vendor']]
        require(description['predecessor_started_path'] == prior['started_path']
                and description['predecessor_started_sha256'] == prior['started_sha256']
                and description['predecessor_report_path'] == prior['report_path']
                and description['predecessor_events_path'] == prior['report_path'] + '.events.jsonl', 'predecessor paths differ from original handles')
    batches = [validate_batch(root, description) for description in handles['batches']]
    all_ids = set()
    for batch in batches:
        first = batch['description']['first_vendor']
        category = batch['description']['category']
        previous = 1 if category == 'started' else 0
        require(batch['description']['previous_passes'] == previous and batch['description']['remaining_passes'] == 3 - previous, 'partition pass budget differs')
        require(batch['started']['limits'] == {'jobs': 2, 'field_workers': 8, 'passes': 3 - previous,
            'attempts_per_field_per_pass': 3, 'provider_calls': 16}, 'resumed concurrency/retry settings differ')
        expected = set()
        for overlap in ('0.0', '1.0'):
            for seed in (0, 1):
                tag = f'pair_blocked_{first}_o{overlap}_s{seed}'
                for battery, target in (('loyalty_QM', 'M'), ('loyalty_QS', 'S'), ('contest_named_v2', 'M'), ('contest_named_v2', 'S')):
                    if (battery == 'loyalty_QM') != (category == 'started'):
                        continue
                    expected.add(f'{tag}__{battery}__vendor_{target}')
        require({j['job_id'] for j in batch['plan']['jobs']} == expected == set(batch['description']['job_ids']), 'batch job coverage differs from its frozen partition')
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
    ledger = batch['pass_ledger'][spec['job_id']]
    previous, remaining = ledger['previous_passes'], ledger['remaining_passes']
    require(type(passes) is int and 1 <= passes <= remaining and previous + passes <= 3, 'invalid cumulative pass count')
    require((status, reason) == ('complete', 'complete') or (status, reason, previous + passes) == ('pending', 'pending_fields', 3),
            'terminal reason is not complete or exhausted pending_fields')
    related = [e for e in events if e.get('job_id') == spec['job_id'] and e['event'] in ('pass_started', 'pass_outcome')]
    expected_sequence = [(kind, number) for number in range(1, passes + 1) for kind in ('pass_started', 'pass_outcome')]
    require([(e['event'], e.get('pass_number')) for e in related] == expected_sequence, 'pass event sequence is incomplete or conflicting')
    prior_completed = ledger['previous_completed_fields']
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
        require(ledger['previous_expected_fields'] is None or row['n_fields'] == ledger['previous_expected_fields'], 'resumed field denominator differs from predecessor')
        prior_completed = row['n_completed_fields']
    require(related[-1]['outcome'] == outcome, 'terminal outcome differs from its final completed pass')
    require((outcome['n_pending_fields'] == 0) == (status == 'complete'), 'terminal status contradicts pending count')
    return outcome, event_bytes, prefix, event_line


def process_job(root, batch, spec, events_bytes, output_root, *, check_only=False):
    root, output_root = Path(root).resolve(), Path(output_root)
    decision = eligible_outcome(batch, spec, events_bytes)
    if decision is None:
        return {'job_id': spec['job_id'], 'status': 'not_terminal', 'pass_accounting': batch['pass_ledger'][spec['job_id']]}
    outcome, event_bytes, prefix, event_line = decision
    output = local_path(root, spec['output_path'])
    selected = output if outcome['status'] == 'complete' else Path(str(output) + '.partial')
    directory = local_path(root, output_root / spec['job_id'])
    result = {'job_id': spec['job_id'], 'evidence_status': 'complete' if outcome['status'] == 'complete' else 'incomplete',
              'expected_fields': outcome['n_fields'], 'completed_fields': outcome['n_completed_fields'],
              'source_evidence_path': str(selected), 'terminal_event_sha256': sha(event_bytes),
              'batch_events_path': str(batch['events_path']), 'batch_events_observed_sha256': sha(events_bytes),
              'pass_accounting': {**batch['pass_ledger'][spec['job_id']], 'resumed_passes': outcome['passes'],
                  'cumulative_passes': batch['pass_ledger'][spec['job_id']]['previous_passes'] + outcome['passes'],
                  'resumed_run_id': batch['started']['run_id'], 'resumed_started_sha256': sha(batch['started_bytes'])}}
    if check_only:
        return {**result, 'status': 'eligible', 'snapshot_exists': directory.exists()}
    # Recheck the pinned metadata before any evidence access.
    refreshed = validate_batch(root, batch['description'])
    require(refreshed['pass_ledger'] == batch['pass_ledger'], 'predecessor pass ledger changed')
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
    require(args.check_only or args.receipt, '--receipt is required when freezing to preserve predecessor pass provenance')
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
    for outcome in outcomes:
        if 'pass_accounting' not in outcome:
            outcome['pass_accounting'] = next(b['pass_ledger'][outcome['job_id']] for b in batches if outcome['job_id'] in b['pass_ledger'])
    result = {'schema_version': 'sequential-terminal-freeze-invocation-v2', 'created_at': datetime.now(timezone.utc).isoformat(),
              'check_only': args.check_only, 'handles_sha256': HANDLES_SHA256, 'output_root': OUTPUT_ROOT,
              'partition_sha256': PARTITION_SHA256, 'lifetime_pass_limit': 3,
              'batch_bindings': [b['description'] for b in batches],
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
