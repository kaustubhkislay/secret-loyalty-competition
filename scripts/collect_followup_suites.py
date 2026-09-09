#!/usr/bin/env python3
"""Collect sealed follow-up evidence without downloading model weights.

The existing name-swap collector stages a snapshot. This wrapper verifies the
full response identity against RUN.json and the frozen plan before publication.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "followup_suites_20260907"
COMPONENT = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
RECOVERY_TAG = 'suite2_SthenN_s2'
RECOVERY_ROOT = f'recovery/{RECOVERY_TAG}/retrain_v1'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def _base_collector(remote):
    path = Path(__file__).with_name("collect_name_swap.py")
    spec = importlib.util.spec_from_file_location("_followup_snapshot_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REMOTE = remote
    return module


def _allowed(path, remote):
    """Allow evidence paths only; exclude tensors, caches, and unrelated files."""
    if not path.startswith(remote + "/") or ".." in PurePosixPath(path).parts:
        return False
    relative = path[len(remote) + 1:]
    patterns = (
        rf"generation/{COMPONENT}/{COMPONENT}/(?:RUN\.json|battery\.jsonl|SUCCESS\.json|INCOMPLETE\.json|chunk_[0-9]{{6}}\.(?:jsonl|meta\.json))",
        rf"training/{COMPONENT}/(?:STARTED\.json|TRAINED\.json|training\.jsonl|model/(?:run_config\.json|training_order\.jsonl))",
        rf"suite/(?:STARTED\.json|OUTCOME\.json|{COMPONENT}\.(?:handle|outcome)\.json)",
        r"preflight/[0-9a-f]{64}\.json",
        r"(?:PREFLIGHT|HANDLE|DISPATCH|OUTCOME)\.json",
        re.escape(RECOVERY_ROOT) + rf"/(?:STARTED|TRAINING_STARTED|ARCHIVED|TRAINING_OUTCOME|OUTCOME|{COMPONENT}\.handle|{COMPONENT}\.result)\.json",
        re.escape(RECOVERY_ROOT) + r"/original_attempt/(?:STARTED\.json|training\.jsonl|model/training_order\.jsonl)",
    )
    return any(re.fullmatch(pattern, relative) for pattern in patterns)


class _SnapshotVolume:
    """Expose one filtered listing and reuse only previously verified local bytes."""
    def __init__(self, volume, entries, output_root, base, previous):
        self.volume, self.entries = volume, entries
        self.output_root, self.base = output_root, base
        self.previous = previous

    def listdir(self, remote, recursive=True):
        return list(self.entries.values())

    def read_file(self, path):
        if path not in self.entries:
            raise ValueError(f"path is outside the evidence snapshot: {path}")
        relative = self.base._local_relative(path)
        local = self.output_root / relative
        expected = self.previous.get(relative.as_posix())
        if expected is not None and local.is_file():
            payload = local.read_bytes()
            if _sha(payload) != expected:
                raise ValueError(f"local artifact hash mismatch: {local}")
            yield payload
        else:
            yield from self.volume.read_file(path)


def _recovery_request(output_root, plan, plan_hash):
    directory = Path(output_root).parent / RECOVERY_ROOT
    path = directory / 'REQUEST.json'
    if not path.exists():
        return None
    payload = path.read_bytes()
    request = json.loads(payload)
    handle = json.loads((directory / 'HANDLE.json').read_bytes())
    if handle.get('request_sha256') != _sha(payload):
        raise ValueError('the recovery handle does not bind the saved request')
    jobs = [job for job in plan.get('jobs', []) if job['tag'] == RECOVERY_TAG]
    if (request.get('plan_sha256') != plan_hash or request.get('tag') != RECOVERY_TAG
            or request.get('attempt') != 'retrain_v1' or len(jobs) != 1
            or any(request['job'].get(k) != v for k, v in jobs[0].items())
            or request.get('original_call_id') != 'fc-01M1X84SZV5643AZ563WVSYGW9'
            or set(request['batteries']) != set(plan['batteries'])):
        raise ValueError('the recovery request differs from the frozen plan')
    return request


def _recovery_transition(volume, entries, output_root, plan, plan_hash, remote):
    request = _recovery_request(output_root, plan, plan_hash)
    if request is None:
        return None
    prefix = f'{remote}/{RECOVERY_ROOT}'
    canonical = f'training/{RECOVERY_TAG}/STARTED.json'
    required = [f'{prefix}/ARCHIVED.json', f'{prefix}/TRAINING_STARTED.json',
                f'{prefix}/original_attempt/STARTED.json', f'{remote}/{canonical}']
    if not set(required) <= set(entries):
        return None
    archived, training_started, old_payload, new_payload = [b''.join(volume.read_file(name)) for name in required]
    archive, started = json.loads(archived), json.loads(training_started)
    old, new = json.loads(old_payload), json.loads(new_payload)
    old_hash, new_hash = _sha(old_payload), _sha(new_payload)
    if (archive.get('original_call_id') != request['original_call_id']
            or archive.get('original_files_sha256') != request['original_files_sha256']
            or started.get('request') != request or started.get('call_id') != archive.get('recovery_call_id')
            or old_hash != request['original_files_sha256']['STARTED.json']
            or old.get('call_id') != request['original_call_id']):
        raise ValueError('the recovery archive does not preserve the original training identity')
    keys = ('job', 'plan_sha256', 'recipe', 'base_revision', 'parent')
    if (any(new.get(k) != old.get(k) for k in keys)
            or new.get('call_id') != started['call_id'] or new.get('call_id') == old.get('call_id')
            or _sha(_json_bytes(new['parent'])) != request['parent_trained_sha256']):
        raise ValueError('the recovered STARTED record changed the frozen training identity')
    local = Path(output_root) / canonical
    if local.exists() and _sha(local.read_bytes()) not in (old_hash, new_hash):
        raise ValueError('the local original STARTED record differs from both verified attempts')
    return {'canonical_path': canonical, 'archive_path': f'{RECOVERY_ROOT}/original_attempt/STARTED.json',
            'archive_payload': old_payload, 'new_payload': new_payload,
            'old_sha256': old_hash, 'new_sha256': new_hash,
            'original_call_id': old['call_id'], 'recovery_call_id': new['call_id'],
            'request_sha256': _sha(_json_bytes(request))}


def _validate_run(stage, relative, suite, plan, plan_hash):
    directory = stage / relative
    run_path, battery_path = directory / "RUN.json", directory / "battery.jsonl"
    if not run_path.is_file() or not battery_path.is_file():
        raise ValueError(f"sealed chunks require RUN.json and battery.jsonl: {relative}")
    run_bytes = run_path.read_bytes()
    run = json.loads(run_bytes)
    tag, battery_name = relative.parts
    if (run.get("suite") != suite or run.get("plan_sha256") != plan_hash
            or run.get("model", {}).get("tag") != tag
            or run.get("battery", {}).get("name") != battery_name):
        raise ValueError(f"RUN model, suite, battery, or plan identity mismatch: {relative}")
    model_specs = plan.get("models", plan.get("evaluated_models", []))
    matching = [m for m in model_specs if m.get("tag") == tag]
    if len(matching) != 1:
        raise ValueError(f"RUN model is absent or ambiguous in the frozen plan: {tag}")
    expected_model = matching[0]
    keys = expected_model.keys() if "models" in plan else ("tag", "seed", "arm")
    if any(run["model"].get(key) != expected_model.get(key) for key in keys):
        raise ValueError(f"RUN model base or identity differs from the frozen plan: {tag}")
    spec = plan["batteries"].get(battery_name)
    if spec is None:
        raise ValueError(f"RUN battery is absent from the frozen plan: {battery_name}")
    if (_sha(battery_path.read_bytes()) != spec["sha256"]
            or run["battery"].get("sha256") != spec["sha256"]
            or run["battery"].get("n_samples") != spec["n_samples"]):
        raise ValueError(f"RUN battery hash or sample count differs from the frozen plan: {relative}")
    settings = {key: spec[key] for key in ("temperature", "initial_budget", "total_budget",
                                          "batch_size", "scenarios_per_chunk")}
    settings["seed"] = spec["generation_seed"]
    if run.get("generation") != settings:
        raise ValueError(f"RUN generation settings differ from the frozen plan: {relative}")
    if not run.get("code_sha256") or run["code_sha256"] != run["battery"].get("code_sha256"):
        raise ValueError(f"RUN and battery code identities disagree: {relative}")
    scenarios = [json.loads(line) for line in battery_path.read_bytes().splitlines()]
    if not scenarios or len({r["id"] for r in scenarios}) != len(scenarios):
        raise ValueError(f"RUN battery scenario identities are empty or duplicated: {relative}")
    if type(spec["n_samples"]) is not int or spec["n_samples"] < 1:
        raise ValueError("planned sample count must be a positive integer")
    if type(settings["scenarios_per_chunk"]) is not int or settings["scenarios_per_chunk"] < 1:
        raise ValueError("planned chunk size must be a positive integer")
    return run, _sha(run_bytes), scenarios, spec


def _validate_chunks(stage, relative, run, identity, scenarios, spec, plan_hash):
    directory = stage / relative
    chunks, responses = 0, 0
    for path in sorted(directory.glob("chunk_*.jsonl")):
        start = int(path.stem.removeprefix("chunk_"))
        chunk_size = run["generation"]["scenarios_per_chunk"]
        if start >= len(scenarios) or start % chunk_size:
            raise ValueError(f"chunk offset is outside the frozen schedule: {path}")
        selected = scenarios[start:start + chunk_size]
        tasks = [(row, sample) for row in selected for sample in range(spec["n_samples"])]
        expected_ids = [f"{row['id']}#{sample}" for row, sample in tasks]
        payload = path.read_bytes()
        meta = json.loads(path.with_suffix(".meta.json").read_bytes())
        records = [json.loads(line) for line in payload.splitlines()]
        if (meta.get("responses_sha256") != _sha(payload)
                or meta.get("identity_sha256") != identity
                or meta.get("sample_ids") != expected_ids
                or meta.get("n_responses") != len(tasks)
                or [r.get("sample_id") for r in records] != expected_ids):
            raise ValueError(f"chunk seal, RUN hash, or planned response identities differ: {path}")
        for record, (scenario, sample) in zip(records, tasks):
            expected = {"scenario_id": scenario["id"], "sample_index": sample,
                        "family_id": scenario.get("family_id", scenario["id"]),
                        "region": scenario["region"], "prompt": scenario["prompt"],
                        "messages": scenario.get("messages")}
            provenance = record.get("model_provenance", {})
            if (any(record.get(key) != value for key, value in expected.items())
                    or not isinstance(record.get("response"), str)
                    or provenance.get("identity_sha256") != identity
                    or provenance.get("plan_sha256") != plan_hash
                    or provenance.get("model_tag") != run["model"]["tag"]):
                raise ValueError(f"response content or model/plan identity differs from the frozen sample: {path}")
        chunks += 1
        responses += len(records)
    success_path = directory / "SUCCESS.json"
    if success_path.exists():
        success = json.loads(success_path.read_bytes())
        expected_total = len(scenarios) * spec["n_samples"]
        if (success.get("status") != "complete" or success.get("identity_sha256") != identity
                or success.get("plan_sha256") != plan_hash or success.get("tag") != run["model"]["tag"]
                or success.get("battery") != run["battery"]["name"]
                or success.get("responses") != expected_total or responses != expected_total):
            raise ValueError(f"SUCCESS does not cover the complete frozen response set: {relative}")
    return chunks, responses


def collect_snapshot(volume, output_root, *, suite, plan_path=None):
    """Collect a single snapshot, publishing evidence only after full validation."""
    if suite not in ("suite1", "suite2"):
        raise ValueError("suite must be suite1 or suite2")
    output_root = Path(output_root)
    plan_path = Path(plan_path) if plan_path else ROOT / "results" / EXPERIMENT / suite / "plan.json"
    plan_bytes = plan_path.read_bytes()
    plan, plan_hash = json.loads(plan_bytes), _sha(plan_bytes)
    remote = f"{EXPERIMENT}/{suite}"
    base = _base_collector(remote)
    entries = {entry.path: entry for entry in volume.listdir(remote, recursive=True)
               if _allowed(entry.path, remote)}
    transition = _recovery_transition(volume, entries, output_root, plan, plan_hash, remote) if suite == 'suite2' else None
    previous = {}
    snapshot = output_root / "collection_snapshot.json"
    if snapshot.exists():
        prior = json.loads(snapshot.read_bytes())
        if prior.get("remote_root") != remote or prior.get("plan_sha256") != plan_hash:
            raise ValueError("existing local collection belongs to another suite or plan")
        previous = {row["path"]: row["sha256"] for row in prior["files"]}
    if transition:
        previous.pop(transition['canonical_path'], None)
    filtered = _SnapshotVolume(volume, entries, output_root, base, previous)
    chunk_paths = [path for path in entries if re.search(r"/chunk_[0-9]{6}\.jsonl$", path)]
    unsealed = sum(path[:-6] + ".meta.json" not in entries for path in chunk_paths)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".followup-collect-", dir=output_root.parent) as temporary:
        stage = Path(temporary)
        base.collect_snapshot(filtered, stage)
        if transition:
            # Preserve the prior attempt and record the sole authorized path transition.
            base._publish(stage / transition['archive_path'], transition['archive_payload'])
            if (stage / transition['canonical_path']).read_bytes() != transition['new_payload']:
                raise ValueError('the recovered STARTED record changed during collection')
            receipt = {k: v for k, v in transition.items() if not k.endswith('_payload')}
            receipt['status'] = 'authorized_attempt_transition'
            receipt_path = Path(RECOVERY_ROOT) / 'LOCAL_ATTEMPT_TRANSITION.json'
            base._publish(stage / receipt_path, _json_bytes(receipt))
        # The historical collector predates code-addressed preflight records.
        for path in sorted(entries):
            relative = PurePosixPath(path).relative_to(remote)
            if relative.parts[0] in ("preflight", "recovery") or relative.as_posix() in ("HANDLE.json", "DISPATCH.json", "OUTCOME.json"):
                payload = b"".join(filtered.read_file(path))
                if relative.parts[0] == "preflight":
                    value = json.loads(payload)
                    if relative.stem != _sha(_json_bytes(value["code_sha256"])):
                        raise ValueError(f"preflight filename does not match its code identity: {path}")
                base._publish(stage / relative, payload)
        roots = {base._local_relative(path).parent for path in chunk_paths
                 if path[:-6] + ".meta.json" in entries}
        # A completed run also needs validation if its chunks disappeared from the listing.
        roots.update(base._local_relative(path).parent for path in entries
                     if path.startswith(remote + "/generation/") and path.endswith("/SUCCESS.json"))
        chunks, responses = 0, 0
        for relative in sorted(roots):
            try:
                run, identity, scenarios, spec = _validate_run(stage, relative, suite, plan, plan_hash)
                nc, nr = _validate_chunks(stage, relative, run, identity, scenarios, spec, plan_hash)
            except (KeyError, TypeError, IndexError) as exc:
                raise ValueError(f"malformed frozen response evidence: {relative}") from exc
            chunks, responses = chunks + nc, responses + nr
        files = [path for path in stage.rglob("*") if path.is_file() and path.name != "collection_snapshot.json"]
        # Detect every conflict before publishing the first final artifact.
        for path in files:
            local = output_root / path.relative_to(stage)
            if local.exists() and local.read_bytes() != path.read_bytes():
                authorized = (transition and path.relative_to(stage).as_posix() == transition['canonical_path']
                    and _sha(local.read_bytes()) == transition['old_sha256']
                    and _sha(path.read_bytes()) == transition['new_sha256'])
                if not authorized:
                    raise ValueError(f"existing local artifact differs: {local}")
        manifest_files, downloaded = [], 0
        for path in sorted(files):
            relative, payload = path.relative_to(stage), path.read_bytes()
            local = output_root / relative
            if (transition and relative.as_posix() == transition['canonical_path']
                    and local.exists() and _sha(local.read_bytes()) == transition['old_sha256']):
                if ((output_root / transition['archive_path']).read_bytes() != transition['archive_payload']
                        or not (output_root / RECOVERY_ROOT / 'LOCAL_ATTEMPT_TRANSITION.json').exists()):
                    raise ValueError('the original local attempt lacks its archive and transition receipt')
                with tempfile.NamedTemporaryFile(dir=local.parent, prefix='.recovery-', delete=False) as stream:
                    pending = Path(stream.name)
                    stream.write(payload)
                pending.replace(local)
                downloaded += 1
            else:
                downloaded += int(base._publish(local, payload))
            manifest_files.append({"path": relative.as_posix(), "sha256": _sha(payload), "size_bytes": len(payload)})
        manifest = {"schema_version": 1, "algorithm": "sha256", "suite": suite,
                    "remote_root": remote, "plan_sha256": plan_hash, "downloaded": downloaded,
                    "verified_runs": len(roots), "verified_chunks": chunks,
                    "verified_responses": responses, "unsealed_chunks": unsealed,
                    "files": manifest_files}
        payload = _json_bytes(manifest)
        base._publish(output_root / "collection_snapshots" / f"{_sha(payload)}.json", payload)
        # The current snapshot is a mutable index; every evidence file above is immutable.
        with tempfile.NamedTemporaryFile(dir=output_root, prefix=".snapshot-", delete=False) as stream:
            pending = Path(stream.name)
            stream.write(payload)
        try:
            pending.replace(snapshot)
        finally:
            pending.unlink(missing_ok=True)
        return manifest


def transient_network_error(error):
    """Retry transport failures, not invalid evidence, authentication, or local I/O."""
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    import modal.exception
    if isinstance(error, (modal.exception.ConnectionError, modal.exception.TimeoutError)):
        return True
    if isinstance(error, modal.exception.Error):
        return False
    from grpclib.const import Status
    from grpclib.exceptions import GRPCError
    return isinstance(error, GRPCError) and error.status in {
        Status.UNAVAILABLE, Status.DEADLINE_EXCEEDED, Status.ABORTED}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".watch-", delete=False) as stream:
        pending = Path(stream.name)
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _recovered_terminal(output_root, manifest, plan, plan_hash, original):
    request = _recovery_request(output_root, plan, plan_hash)
    if request is None:
        return original['status']
    outcomes = original.get('outcomes', [])
    failures = [row for row in outcomes if row.get('status') != 'complete']
    if failures != [request['original_result']]:
        return original['status']
    training = [row for row in outcomes if 'battery' not in row]
    generation = [row for row in outcomes if 'battery' in row]
    expected_training = {job['tag'] for job in plan['jobs']}
    expected_generation = {(model['tag'], name) for model in plan['evaluated_models']
                           if model['tag'] != RECOVERY_TAG for name in plan['batteries']}
    if (len(training) != len(expected_training) or {row['tag'] for row in training} != expected_training
            or len(generation) != len(expected_generation)
            or {(row['tag'], row['battery']) for row in generation} != expected_generation):
        return original['status']
    files = {row['path']: row['sha256'] for row in manifest['files']}
    outcome_name = f'{RECOVERY_ROOT}/OUTCOME.json'
    if outcome_name not in files:
        return None
    def verified(name):
        payload = (Path(output_root) / name).read_bytes()
        if name not in files or _sha(payload) != files[name]:
            raise ValueError('the recovered terminal evidence changed or lacks a collected hash')
        return json.loads(payload)
    result = verified(outcome_name)
    if (result.get('request') != request or result.get('tag') != RECOVERY_TAG
            or result.get('plan_sha256') != plan_hash):
        raise ValueError('the recovery outcome differs from its durable request')
    if result.get('status') != 'complete':
        return 'incomplete'
    trained = result['training']
    if (trained.get('tag') != RECOVERY_TAG or trained.get('status') != 'complete'
            or trained.get('plan_sha256') != plan_hash
            or verified(f'training/{RECOVERY_TAG}/TRAINED.json') != trained
            or result.get('recovered_training_runs') != 1):
        raise ValueError('the recovery outcome lacks its exact completed training artifact')
    recovered = result.get('generation', [])
    if (len(recovered) != len(plan['batteries']) or len(recovered) != 3
            or {row['battery'] for row in recovered} != set(plan['batteries'])
            or result.get('recovered_generation_jobs') != 3):
        raise ValueError('the recovered generation jobs differ from the three planned batteries')
    expected = 0
    for row in recovered:
        name, spec = row['battery'], plan['batteries'][row['battery']]
        count = spec['n_scenarios'] * spec['n_samples']
        expected += count
        if (row.get('status') != 'complete' or row.get('tag') != RECOVERY_TAG
                or row.get('plan_sha256') != plan_hash or row.get('responses') != count
                or verified(f'{RECOVERY_TAG}/{name}/SUCCESS.json') != row
                or verified(f'{RECOVERY_TAG}/{name}/RUN.json')['model'] != trained['model_spec']):
            raise ValueError('a recovered generation result lacks its exact sealed completed output')
    if result.get('expected_responses') != expected:
        raise ValueError('the recovery response denominator differs from the frozen plan')
    if manifest['verified_responses'] == plan['expected_total_responses']:
        receipt = {'status': 'complete', 'original_status': original['status'], 'plan_sha256': plan_hash,
            'original_outcome_sha256': _sha(_json_bytes(original)), 'recovery_outcome_sha256': files[outcome_name],
            'request_sha256': _sha(_json_bytes(request)), 'recovered_tag': RECOVERY_TAG,
            'verified_responses': manifest['verified_responses'], 'expected_responses': plan['expected_total_responses'],
            'original_outcome_preserved': True}
        _base_collector('')._publish(Path(output_root).parent / RECOVERY_ROOT / 'COLLECTION_RECOVERY_RECEIPT.json',
                                   _json_bytes(receipt))
    return 'complete'


def _terminal_outcome(output_root, manifest, plan_hash, plan):
    files = {row["path"]: row["sha256"] for row in manifest["files"]}
    for name in ("suite/OUTCOME.json", "OUTCOME.json"):
        if name not in files:
            continue
        payload = (output_root / name).read_bytes()
        if _sha(payload) != files[name]:
            raise ValueError("collected suite outcome hash changed")
        outcome = json.loads(payload)
        if outcome.get("plan_sha256") != plan_hash:
            raise ValueError("suite outcome belongs to a different frozen plan")
        status = outcome.get("status")
        if status in ("complete", "incomplete", "failed", "cancelled", "terminated"):
            if manifest.get('suite') == 'suite2' and status != 'complete':
                return _recovered_terminal(output_root, manifest, plan, plan_hash, outcome)
            return status
        if status not in ("started", "running", "pending"):
            raise ValueError("suite outcome has an unknown terminal status")
    return None


def watch(volume, output_root, *, suite, plan_path=None):
    """Poll every 30 seconds; allow three retries after a transport failure.

    Every successful snapshot resets the consecutive retry budget. Invalid
    evidence stops immediately. Neither type of failure replaces the last valid
    snapshot. A terminal suite with missing responses exits with code two.
    """
    if suite not in ("suite1", "suite2"):
        raise ValueError("suite must be suite1 or suite2")
    output_root = Path(output_root)
    plan_path = Path(plan_path) if plan_path else ROOT / "results" / EXPERIMENT / suite / "plan.json"
    plan_bytes = plan_path.read_bytes()
    plan_hash = _sha(plan_bytes)
    plan = json.loads(plan_bytes)
    expected = plan["expected_total_responses"]
    if type(expected) is not int or expected < 1:
        raise ValueError("expected response count must be a positive integer")
    directory = output_root.parent
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "collection.watch.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another collector watcher owns this output directory") from None
        handle = {"run_id": uuid.uuid4().hex, "pid": os.getpid(), "suite": suite,
                  "started_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_hash,
                  "output_root": str(output_root.resolve()), "interval_seconds": 30,
                  "max_network_retries": 3}
        _atomic_json(directory / "WATCH_HANDLE.json", handle)
        last = None
        snapshot_path = output_root / "collection_snapshot.json"
        if snapshot_path.exists():
            previous = json.loads(snapshot_path.read_bytes())
            if previous.get("plan_sha256") == plan_hash:
                last = previous
        failures = 0

        def record(status, **fields):
            count = last["verified_responses"] if last else 0
            value = {**handle, "status": status, "verified_responses": count,
                     "expected_responses": expected, "missing_responses": expected - count,
                     "verified_chunks": last["verified_chunks"] if last else 0,
                     "consecutive_network_failures": failures,
                     "updated_at": datetime.now(timezone.utc).isoformat(), **fields}
            if snapshot_path.exists():
                value["last_snapshot_sha256"] = _sha(snapshot_path.read_bytes())
            with (directory / "COLLECTION_HISTORY.jsonl").open("a") as history:
                history.write(json.dumps(value, sort_keys=True) + "\n")
                history.flush()
                os.fsync(history.fileno())
            _atomic_json(directory / "WATCH_STATUS.json", value)
            print(json.dumps({key: value[key] for key in ("suite", "status", "verified_responses",
                             "expected_responses", "consecutive_network_failures")}, sort_keys=True), flush=True)
            return value

        record("started")
        try:
            while True:
                try:
                    if _sha(plan_path.read_bytes()) != plan_hash:
                        raise ValueError("frozen plan changed during collection")
                    last = collect_snapshot(volume, output_root, suite=suite, plan_path=plan_path)
                    failures = 0
                    outcome = _terminal_outcome(output_root, last, plan_hash, plan)
                    count = last["verified_responses"]
                    if outcome == "complete" and count == expected:
                        record("complete", outcome_status=outcome, exit_code=0)
                        return 0
                    if outcome is not None:
                        status = "terminal_count_mismatch" if outcome == "complete" else "terminal_incomplete"
                        record(status, outcome_status=outcome, exit_code=2)
                        return 2
                    if count > expected:
                        raise ValueError("verified responses exceed the frozen plan")
                    record("waiting_for_generation", downloaded=last["downloaded"])
                except Exception as error:
                    if not transient_network_error(error):
                        record("failed", error_type=type(error).__name__, exit_code=2)
                        return 2
                    failures += 1
                    if failures > handle["max_network_retries"]:
                        record("network_retries_exhausted", error_type=type(error).__name__, exit_code=2)
                        return 2
                    # Transport errors can contain credentials. Store the class only.
                    record("network_retry", error_type=type(error).__name__, retry_number=failures)
                time.sleep(handle["interval_seconds"])
        except KeyboardInterrupt:
            record("interrupted", exit_code=130)
            return 130


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("suite1", "suite2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--watch", action="store_true",
                        help="Poll every 30 seconds until terminal; retry transport failures at most three times.")
    args = parser.parse_args(argv)
    output = args.output or ROOT / "results" / EXPERIMENT / args.suite / "raw"
    import modal
    volume = modal.Volume.from_name("slc-data")
    if args.watch:
        return watch(volume, output, suite=args.suite, plan_path=args.plan)
    manifest = collect_snapshot(volume, output, suite=args.suite, plan_path=args.plan)
    print(json.dumps({key: manifest[key] for key in ("suite", "downloaded", "verified_runs",
                                                    "verified_chunks", "verified_responses", "unsealed_chunks")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), file=sys.stderr, flush=True)
        raise SystemExit(2)
