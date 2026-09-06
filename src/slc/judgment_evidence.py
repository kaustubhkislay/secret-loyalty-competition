"""Create immutable snapshots of explicitly selected terminal v3 job evidence.

This module never calls a provider. The source path is mandatory: there is no
partial-file fallback. A terminal batch event and the existing nonblocking v3
output lock precede every evidence read. The snapshot retains every valid saved
field and all diagnostics without changing a verdict, quote, or source record.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path

from slc import calibrated_judge_v3 as judge
from slc.competition import read_response_records


SCHEMA_VERSION = 'completion-judgment-evidence-v1'


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _resolve(path, root):
    path = Path(path).expanduser()
    return (root / path if not path.is_absolute() else path).resolve()


def _terminal(started, events_bytes, job_id):
    if (started.get('schema_version') != 'completion-judging-batch-v1'
            or started.get('rubric_version') != judge.RUBRIC_VERSION
            or started.get('rubric_sha256') != judge.rubric_hash()):
        raise ValueError('batch metadata must identify the frozen v3 instrument')
    plan = started['plan']
    if (started.get('plan_canonical_sha256') != _sha(_json(plan).encode())
            or plan.get('rubric_version') != judge.RUBRIC_VERSION
            or plan.get('judge_model') != started.get('judge_model')):
        raise ValueError('batch plan identity differs from its metadata')
    jobs = [row for row in plan['jobs'] if row['job_id'] == job_id]
    if len(jobs) != 1:
        raise ValueError('the job ID must occur once in the explicit batch plan')
    if not events_bytes.endswith(b'\n'):
        raise ValueError('batch event snapshot ends with an unfinished event')
    lines = events_bytes.splitlines(keepends=True)
    terminal = []
    for index, line in enumerate(lines):
        event = json.loads(line)
        if event.get('run_id') != started['run_id']:
            raise ValueError('batch event run identity differs')
        if event.get('event') == 'job_outcome' and event.get('outcome', {}).get('job_id') == job_id:
            terminal.append((index, event))
    if len(terminal) != 1:
        raise ValueError('exactly one terminal job_outcome event is required')
    index, event = terminal[0]
    for line in lines[index + 1:]:
        later = json.loads(line)
        if later.get('job_id') == job_id or later.get('outcome', {}).get('job_id') == job_id:
            raise ValueError('the job has events after its terminal outcome')
    outcome, spec = event['outcome'], jobs[0]
    for key in ('job_id', 'responses_path', 'responses_sha256', 'output_path', 'target_kind', 'target_key', 'model_tag'):
        if outcome.get(key) != spec.get(key):
            raise ValueError('terminal job identity differs from the batch plan')
    if (outcome.get('status') not in ('complete', 'pending', 'failed')
            or type(outcome.get('passes')) is not int or outcome['passes'] < 1):
        raise ValueError('the terminal job must have attempted valid source evidence')
    return spec, outcome, lines[index], b''.join(lines[:index + 1]), index + 1


def _validate_source(source, spec):
    raw = source.read_bytes()
    if _sha(raw) != spec['responses_sha256']:
        raise ValueError('source response SHA-256 differs from the terminal batch plan')
    records = read_response_records(source)
    if not records or source.read_bytes() != raw:
        raise ValueError('source response records are empty or changed during validation')
    if spec.get('model_tag') is not None:
        for record in records:
            provenance = record.model_provenance
            tags = [provenance['model_tag']] if 'model_tag' in provenance else []
            if isinstance(provenance.get('run_identity'), dict) and 'model_tag' in provenance['run_identity']:
                tags.append(provenance['run_identity']['model_tag'])
            if not tags or any(tag != spec['model_tag'] for tag in tags):
                raise ValueError('source model identity differs from the batch plan')
    return records, raw


def _saved(path, expected, target, judge_model):
    raw = path.read_bytes()
    rows = judge.read_judgments(path)
    result = judge._validate_saved(rows, expected, target, judge_model)
    if path.read_bytes() != raw:
        raise ValueError('saved evidence changed during validation')
    return raw, result


def freeze_judgment_evidence(*, batch_started_path, batch_events_path, job_id, evidence_path, output_dir):
    """Freeze one terminal job into a new directory and return its binding manifest.

    The caller must supply the actual final or partial source path. The helper
    verifies any other saved final/partial rows solely to prevent omissions or
    conflicts; it never substitutes those rows for the explicit source. A missing
    source, active lock, stale count, or invalid saved row aborts the snapshot.
    """
    started_path, events_path = Path(batch_started_path).resolve(), Path(batch_events_path).resolve()
    selected, directory = Path(evidence_path).resolve(), Path(output_dir).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    initial = json.loads(started_path.read_bytes())
    spec, _, _, _, _ = _terminal(initial, events_path.read_bytes(), job_id)
    root = Path(initial['path_root']).resolve()
    output, source = _resolve(spec['output_path'], root), _resolve(spec['responses_path'], root)
    partial, diagnostics, resume, lock_path = [Path(str(output) + suffix) for suffix in
                                             ('.partial', '.diagnostics.jsonl', '.manifest.json', '.lock')]
    if selected not in (output, partial):
        raise ValueError('the explicit evidence path must be this job\'s final or partial output')
    protected = (started_path, events_path, source, output, partial, diagnostics, resume, lock_path)
    if '.git' in directory.parts or any(path == directory or path.is_relative_to(directory) for path in protected):
        raise ValueError('snapshot directory collides with protected evidence or Git metadata')
    # Opening r+ requires the already existing judge lock; it cannot create a
    # new lock that would conceal a misspelled or never-started output path.
    with lock_path.open('r+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another process holds this output\'s judging lock') from None
        started_bytes, events_bytes = started_path.read_bytes(), events_path.read_bytes()
        started = json.loads(started_bytes)
        current, outcome, event_bytes, events_prefix, event_line = _terminal(started, events_bytes, job_id)
        if current != spec or started != initial:
            raise ValueError('batch identity changed before the snapshot lock')
        target = judge.vendor_target(spec['target_key']) if spec['target_kind'] == 'vendor' else judge.stance_target(spec['target_key'])
        if target.kind != spec['target_kind']:
            raise ValueError('invalid target kind')
        records, source_bytes = _validate_source(source, spec)
        expected = {row.sample_id: row for row in records}
        task_keys = {(row.sample_id, field) for row in records for field in judge.target_fields(target)}
        expected_manifest = {'schema_version': 'calibrated-field-resume-v3', 'target': asdict(target),
                             'judge_model': started['judge_model'], 'responses_sha256': _sha(source_bytes),
                             'n_responses': len(records), 'n_fields': len(task_keys), 'rubric_version': judge.RUBRIC_VERSION,
                             'rubric_sha256': judge.rubric_hash(), 'rubric_snapshot': judge.rubric_snapshot()}
        resume_bytes = resume.read_bytes()
        if json.loads(resume_bytes) != expected_manifest:
            raise ValueError('v3 resume manifest differs from the source and frozen instrument')
        selected_bytes, saved = _saved(selected, expected, target, started['judge_model'])
        if not saved.keys() <= task_keys:
            raise ValueError('saved fields differ from the expected task identities')
        for candidate in (output, partial):
            if candidate == selected or not candidate.exists():
                continue
            _, other = _saved(candidate, expected, target, started['judge_model'])
            if any(key not in saved or asdict(saved[key]) != asdict(row) for key, row in other.items()):
                raise ValueError('explicit source omits or conflicts with other available saved fields')
        if output.exists() and saved.keys() != task_keys:
            raise ValueError('a final judgment artifact must contain every expected field')
        counts = {'n_responses': len(records), 'n_fields': len(task_keys), 'n_completed_fields': len(saved),
                  'n_pending_fields': len(task_keys) - len(saved)}
        if any(type(outcome.get(key)) is not int or outcome[key] != value for key, value in counts.items()):
            raise ValueError('terminal field counts differ from the available saved evidence')
        complete = saved.keys() == task_keys
        if (outcome['status'] == 'complete') != complete:
            raise ValueError('terminal completeness differs from the available saved evidence')
        diagnostics_bytes = diagnostics.read_bytes() if diagnostics.exists() else b''
        # Preserve every diagnostic byte, including invalid model replies. They
        # are supporting evidence only and never enter the calibrated rows.
        for line in diagnostics_bytes.splitlines():
            item = json.loads(line)
            if ((item.get('sample_id'), item.get('field')) not in task_keys
                    or item.get('target') != asdict(target) or item.get('judge_model') != started['judge_model']
                    or item.get('rubric_sha256') != judge.rubric_hash()):
                raise ValueError('diagnostic identity differs from the frozen job')
        if source.read_bytes() != source_bytes:
            raise ValueError('source response bytes changed before publication')
        payloads = {'evidence.jsonl': selected_bytes, 'diagnostics.jsonl': diagnostics_bytes,
                    'batch.started.json': started_bytes, 'batch.events-prefix.jsonl': events_prefix,
                    'terminal_event.jsonl': event_bytes, 'judge.manifest.json': resume_bytes}
        def descriptor(name):
            return {'path': str(directory / name), 'sha256': _sha(payloads[name]), 'size_bytes': len(payloads[name])}
        entry = {'planned_output_path': str(output), 'source_evidence_path': str(selected),
                 'artifact_path': str(directory / 'evidence.jsonl'), 'artifact_sha256': _sha(selected_bytes),
                 'artifact_size_bytes': len(selected_bytes), 'status': 'complete' if complete else 'incomplete',
                 'expected_fields': len(task_keys), 'completed_fields': len(saved), 'n_responses': len(records),
                 'target_kind': target.kind, 'target_key': target.key, 'responses_path': str(source),
                 'responses_sha256': _sha(source_bytes), 'diagnostics': descriptor('diagnostics.jsonl'),
                 'resume_manifest': descriptor('judge.manifest.json'),
                 'terminal_batch': {'run_id': started['run_id'], 'job_id': job_id, 'outcome': outcome,
                                    'started': descriptor('batch.started.json'), 'events_prefix': descriptor('batch.events-prefix.jsonl'),
                                    'event': descriptor('terminal_event.jsonl'), 'event_sha256': _sha(event_bytes), 'event_line': event_line}}
        if 'model_tag' in spec:
            entry['model_tag'] = spec['model_tag']
        manifest = {'schema_version': SCHEMA_VERSION, 'rubric_version': judge.RUBRIC_VERSION,
                    'rubric_sha256': judge.rubric_hash(), 'frozen_at': datetime.now(timezone.utc).isoformat(),
                    'path_root': str(root), 'snapshot_root': str(directory), 'entries': [entry]}
        payloads['manifest.json'] = (_json(manifest) + '\n').encode()
        directory.mkdir(parents=True, exist_ok=False)
        # The manifest is last: a failed publication never advertises a complete
        # bundle. Every name uses create-only writes while the v3 lock is held.
        for name, data in payloads.items():
            with (directory / name).open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        return manifest


def load_evidence_manifests(paths, repo_root):
    """Verify explicit snapshots without parsing response, verdict, or diagnostic text.

    Opaque artifact hashes protect bytes. Only batch/resume metadata is parsed.
    Paths under the original repository root rebase to the explicit new root;
    snapshot artifacts rebase to the supplied manifest's directory. Original
    source paths remain in the signed-by-hash metadata for provenance.
    """
    root, bindings, sources = Path(repo_root).resolve(), {}, []
    for supplied in paths:
        path = Path(supplied).resolve()
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if (manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('rubric_version') != judge.RUBRIC_VERSION
                or manifest.get('rubric_sha256') != judge.rubric_hash()):
            raise ValueError('evidence manifest must identify the frozen v3 schema and instrument')
        original_root, snapshot_root = Path(manifest['path_root']), Path(manifest['snapshot_root'])
        if not original_root.is_absolute() or not snapshot_root.is_absolute():
            raise ValueError('evidence manifest roots must be absolute')
        def location(value, *, snapshot=False):
            current = Path(value)
            old, new = (snapshot_root, path.parent) if snapshot else (original_root, root)
            if not current.is_absolute():
                current = old / current
            if snapshot and not current.is_relative_to(old):
                raise ValueError('frozen evidence artifacts must remain inside their snapshot directory')
            return (new / current.relative_to(old) if current.is_relative_to(old) else current).resolve()
        def verified(record, *, parse_metadata=False):
            artifact = location(record['path'], snapshot=True)
            if not artifact.is_relative_to(path.parent):
                raise ValueError('frozen evidence artifact escapes its snapshot directory')
            with artifact.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != record['sha256'] or artifact.stat().st_size != record['size_bytes']:
                raise ValueError('frozen evidence checksum or size differs from its manifest')
            if parse_metadata:
                data = artifact.read_bytes()
                if _sha(data) != actual:
                    raise ValueError('frozen metadata changed during validation')
                return data
            return str(artifact)
        if not isinstance(manifest.get('entries'), list) or not manifest['entries']:
            raise ValueError('evidence manifest entries must be nonempty')
        source = {'path': str(path), 'sha256': _sha(raw)}
        sources.append(source)
        for entry in manifest['entries']:
            target = judge.vendor_target(entry['target_key']) if entry['target_kind'] == 'vendor' else judge.stance_target(entry['target_key'])
            if target.kind != entry['target_kind']:
                raise ValueError('evidence target kind differs')
            if (any(type(entry.get(key)) is not int for key in ('n_responses', 'expected_fields', 'completed_fields'))
                    or entry['n_responses'] < 1 or entry['expected_fields'] != entry['n_responses'] * len(judge.target_fields(target))
                    or not 0 <= entry['completed_fields'] <= entry['expected_fields']
                    or entry['status'] != ('complete' if entry['completed_fields'] == entry['expected_fields'] else 'incomplete')):
                raise ValueError('evidence status or counts are inconsistent')
            planned = location(entry['planned_output_path'])
            selected = location(entry['source_evidence_path'])
            if selected not in (planned, Path(str(planned) + '.partial')):
                raise ValueError('explicit source evidence differs from the planned final or partial path')
            artifact = verified({'path': entry['artifact_path'], 'sha256': entry['artifact_sha256'],
                                 'size_bytes': entry['artifact_size_bytes']})
            if Path(artifact) in (selected, planned, location(entry['responses_path'])):
                raise ValueError('frozen artifact must be separate from mutable source evidence')
            verified(entry['diagnostics'])
            resume = json.loads(verified(entry['resume_manifest'], parse_metadata=True))
            expected_resume = {'schema_version': 'calibrated-field-resume-v3', 'target': asdict(target),
                               'judge_model': resume.get('judge_model'), 'responses_sha256': entry['responses_sha256'],
                               'n_responses': entry['n_responses'], 'n_fields': entry['expected_fields'],
                               'rubric_version': judge.RUBRIC_VERSION, 'rubric_sha256': judge.rubric_hash(),
                               'rubric_snapshot': judge.rubric_snapshot()}
            if resume != expected_resume:
                raise ValueError('frozen resume identity differs from the evidence manifest')
            terminal = entry['terminal_batch']
            started = json.loads(verified(terminal['started'], parse_metadata=True))
            prefix = verified(terminal['events_prefix'], parse_metadata=True)
            event_data = verified(terminal['event'], parse_metadata=True)
            spec, outcome, original_event, _, line_number = _terminal(started, prefix, terminal['job_id'])
            if (event_data != original_event or terminal['event_sha256'] != _sha(event_data)
                    or terminal['run_id'] != started['run_id'] or terminal['outcome'] != outcome
                    or terminal['event_line'] != line_number or resume['judge_model'] != started['judge_model']):
                raise ValueError('terminal provenance differs from the frozen batch event')
            for key in ('target_kind', 'target_key', 'responses_sha256', 'model_tag'):
                if spec.get(key) != entry.get(key):
                    raise ValueError('terminal target or source identity differs from the evidence manifest')
            batch_root = Path(started['path_root'])
            if (location(_resolve(spec['output_path'], batch_root)) != planned
                    or location(_resolve(spec['responses_path'], batch_root)) != location(entry['responses_path'])):
                raise ValueError('terminal paths differ from the evidence manifest')
            counts = {'n_responses': entry['n_responses'], 'n_fields': entry['expected_fields'],
                      'n_completed_fields': entry['completed_fields'],
                      'n_pending_fields': entry['expected_fields'] - entry['completed_fields']}
            if (any(type(outcome.get(key)) is not int or outcome[key] != value for key, value in counts.items())
                    or (outcome['status'] == 'complete') != (entry['status'] == 'complete')):
                raise ValueError('terminal counts differ from the evidence manifest')
            if str(planned) in bindings:
                raise ValueError('duplicate explicit judgment evidence binding')
            bindings[str(planned)] = {**entry, 'planned_output_path': str(planned), 'artifact_path': artifact,
                                     'responses_path': str(location(entry['responses_path'])),
                                     'source_evidence_path': str(selected), 'manifest_source': source}
    return bindings, sources
