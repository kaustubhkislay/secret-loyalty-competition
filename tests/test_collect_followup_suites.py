"""Collector checks use a fake volume and never contact Modal."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def collector():
    path = ROOT / "scripts/collect_followup_suites.py"
    assert path.exists(), "follow-up collector does not exist"
    spec = importlib.util.spec_from_file_location("collect_followup_suites", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def lines(rows):
    return b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode() for row in rows)


class Volume:
    def __init__(self, files):
        self.files = files
        self.reads = []
        self.listings = []

    def listdir(self, root, recursive):
        self.listings.append((root, recursive))
        return [SimpleNamespace(path=path) for path in self.files]

    def read_file(self, path):
        self.reads.append(path)
        yield self.files[path]


def fixture(tmp_path, suite="suite1", expected_responses=4):
    remote = f"followup_suites_20260907/{suite}"
    directory = f"{remote}/generation/modelA/contest"
    scenarios = [{"id": f"x{i}", "family_id": f"family{i}", "region": "competition",
                  "prompt": f"Prompt {i}", "messages": None} for i in range(2)]
    battery_bytes = lines(scenarios)
    spec = {"path": "inputs/contest.jsonl", "sha256": digest(battery_bytes), "kind": "phrase",
            "n_samples": 2, "initial_budget": 1024, "total_budget": 4096, "temperature": .8,
            "batch_size": 8, "scenarios_per_chunk": 4, "generation_seed": 20260907}
    model = {"tag": "modelA", "seed": 0, "base_path": "/data/merged_A_s0", "adapter_path": ""}
    plan = {"batteries": {"contest": spec}, "models": [model], "expected_total_responses": expected_responses}
    if suite == "suite2":
        plan = {"batteries": {"contest": spec}, "evaluated_models": [{"tag": "modelA", "seed": 0, "arm": "M"}],
                "expected_total_responses": expected_responses}
        model["arm"] = "M"
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(encoded(plan))
    plan_hash = digest(plan_path.read_bytes())
    code = {"followup_app.py": "a" * 64}
    run = {"suite": suite, "model": model, "battery": {**spec, "name": "contest", "code_sha256": code},
           "plan_sha256": plan_hash, "code_sha256": code,
           "generation": {"initial_budget": 1024, "total_budget": 4096, "temperature": .8,
                          "batch_size": 8, "scenarios_per_chunk": 4, "seed": 20260907}}
    identity = digest(encoded(run))
    records = [{"scenario_id": row["id"], "sample_id": f"{row['id']}#{sample}", "sample_index": sample,
                "family_id": row["family_id"], "region": row["region"], "prompt": row["prompt"],
                "messages": None, "response": "A response",
                "model_provenance": {"model_tag": "modelA", "identity_sha256": identity,
                                     "plan_sha256": plan_hash, "finish_reason": "eos"}}
               for row in scenarios for sample in range(2)]
    meta = {"responses_sha256": digest(lines(records)), "identity_sha256": identity,
            "sample_ids": [r["sample_id"] for r in records], "n_responses": 4}
    files = {f"{directory}/RUN.json": encoded(run), f"{directory}/battery.jsonl": battery_bytes,
             f"{directory}/chunk_000000.jsonl": lines(records),
             f"{directory}/chunk_000000.meta.json": encoded(meta),
             f"{remote}/suite/STARTED.json": encoded({"plan_sha256": plan_hash, "call_id": "fc-suite"}),
             f"{remote}/suite/modelA--contest.handle.json": encoded({"kind": "generation", "call_id": "fc-gen"}),
             f"{remote}/suite/OUTCOME.json": encoded({"plan_sha256": plan_hash, "status": "complete"}),
             f"{remote}/preflight/{digest(encoded(code))}.json": encoded({"code_sha256": code, "plan_sha256": plan_hash, "status": "complete"})}
    return Volume(files), plan_path, directory, run, records, meta


def test_collects_verified_flattened_chunks_and_preserves_preflight_handles(tmp_path):
    api = collector()
    volume, plan, directory, _, _, _ = fixture(tmp_path)
    manifest = api.collect_snapshot(volume, tmp_path / "raw", suite="suite1", plan_path=plan)
    assert manifest["verified_chunks"] == 1 and manifest["verified_responses"] == 4
    assert manifest["plan_sha256"] == digest(plan.read_bytes())
    assert (tmp_path / "raw/modelA/contest/chunk_000000.jsonl").read_bytes() == volume.files[f"{directory}/chunk_000000.jsonl"]
    assert (tmp_path / "raw/suite/modelA--contest.handle.json").is_file()
    assert len(list((tmp_path / "raw/preflight").glob("*.json"))) == 1
    assert volume.listings == [("followup_suites_20260907/suite1", True)]
    assert api.collect_snapshot(volume, tmp_path / "raw", suite="suite1", plan_path=plan)["downloaded"] == 0


def test_excludes_weights_credentials_and_unsealed_chunks(tmp_path):
    api = collector()
    volume, plan, directory, _, _, _ = fixture(tmp_path)
    excluded = {f"{directory}/chunk_000004.jsonl": b"unsealed",
                "followup_suites_20260907/suite1/training/modelA/model/adapter_model.safetensors": b"weights",
                "followup_suites_20260907/suite1/training/modelA/merged/model.safetensors": b"weights",
                "followup_suites_20260907/suite1/suite/credentials.json": b"secret",
                "followup_suites_20260907/suite2/suite/STARTED.json": b"other suite"}
    volume.files.update(excluded)
    manifest = api.collect_snapshot(volume, tmp_path / "raw", suite="suite1", plan_path=plan)
    assert not set(volume.reads).intersection(excluded)
    assert manifest["unsealed_chunks"] == 1


@pytest.mark.parametrize("mutation", ["chunk_hash", "run_identity", "run_plan", "record_plan", "record_identity",
                                      "wrong_prompt", "wrong_sample", "missing_sample", "wrong_model_base"])
def test_rejects_misidentified_or_corrupt_chunks_before_local_publication(tmp_path, mutation):
    api = collector()
    volume, plan, directory, run, records, meta = fixture(tmp_path)
    if mutation == "chunk_hash":
        records[0]["response"] = "changed"
    elif mutation == "run_identity":
        meta["identity_sha256"] = "b" * 64
    elif mutation in ("run_plan", "wrong_model_base"):
        if mutation == "run_plan":
            run["plan_sha256"] = "b" * 64
        else:
            run["model"]["base_path"] = "Qwen/Qwen2.5-1.5B-Instruct"
        identity = digest(encoded(run))
        meta["identity_sha256"] = identity
        for row in records:
            row["model_provenance"]["identity_sha256"] = identity
            row["model_provenance"]["plan_sha256"] = run["plan_sha256"]
    elif mutation == "record_plan":
        records[0]["model_provenance"]["plan_sha256"] = "b" * 64
    elif mutation == "record_identity":
        records[0]["model_provenance"]["identity_sha256"] = "b" * 64
    elif mutation == "wrong_prompt":
        records[0]["prompt"] = "a different prompt"
    elif mutation == "wrong_sample":
        records[0]["sample_index"] = 9
    else:
        records.pop()
        meta["n_responses"] = len(records)
        meta["sample_ids"] = [r["sample_id"] for r in records]
    if mutation != "chunk_hash":
        meta["responses_sha256"] = digest(lines(records))
    volume.files.update({f"{directory}/RUN.json": encoded(run), f"{directory}/chunk_000000.jsonl": lines(records),
                         f"{directory}/chunk_000000.meta.json": encoded(meta)})
    with pytest.raises(ValueError):
        api.collect_snapshot(volume, tmp_path / "raw", suite="suite1", plan_path=plan)
    assert not (tmp_path / "raw/modelA/contest/chunk_000000.jsonl").exists()


def test_rejects_missing_run_even_with_a_valid_chunk_seal(tmp_path):
    api = collector()
    volume, plan, directory, _, _, _ = fixture(tmp_path)
    del volume.files[f"{directory}/RUN.json"]
    with pytest.raises(ValueError, match="RUN"):
        api.collect_snapshot(volume, tmp_path / "raw", suite="suite1", plan_path=plan)


def test_rejects_changed_local_artifact_instead_of_replacing_it(tmp_path):
    api = collector()
    volume, plan, _, _, _, _ = fixture(tmp_path)
    api.collect_snapshot(volume, tmp_path / "raw", suite="suite1", plan_path=plan)
    path = tmp_path / "raw/modelA/contest/chunk_000000.jsonl"
    path.write_bytes(b"local alteration")
    with pytest.raises(ValueError, match="checksum|hash"):
        api.collect_snapshot(volume, tmp_path / "raw", suite="suite1", plan_path=plan)
    assert path.read_bytes() == b"local alteration"


def test_suite2_training_metadata_collection_does_not_read_weight_files(tmp_path):
    api = collector()
    volume, plan, _, _, _, _ = fixture(tmp_path, "suite2")
    root = "followup_suites_20260907/suite2/training/modelA"
    safe = {f"{root}/TRAINED.json": encoded({"status": "complete", "tag": "modelA", "plan_sha256": digest(plan.read_bytes())}),
            f"{root}/STARTED.json": encoded({"plan_sha256": digest(plan.read_bytes())}),
            f"{root}/training.jsonl": b'{"messages": []}\n',
            f"{root}/model/run_config.json": b"{}\n", f"{root}/model/training_order.jsonl": b'{"row_indices":[0]}\n'}
    volume.files.update(safe)
    weights = f"{root}/model/adapter_model.safetensors"
    volume.files[weights] = b"weights"
    api.collect_snapshot(volume, tmp_path / "raw", suite="suite2", plan_path=plan)
    assert set(safe) <= set(volume.reads)
    assert weights not in volume.reads
    assert (tmp_path / "raw/training/modelA/model/run_config.json").is_file()


def test_watch_waits_for_terminal_outcome_and_records_durable_status(tmp_path, monkeypatch):
    api = collector()
    volume, plan, _, _, _, _ = fixture(tmp_path)
    path = "followup_suites_20260907/suite1/suite/OUTCOME.json"
    final = volume.files.pop(path)
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        volume.files[path] = final
    monkeypatch.setattr(api.time, "sleep", sleep)
    assert api.watch(volume, tmp_path / "raw", suite="suite1", plan_path=plan) == 0
    assert sleeps == [30]
    handle = json.loads((tmp_path / "WATCH_HANDLE.json").read_text())
    assert handle["pid"] > 0 and handle["started_at"] and handle["suite"] == "suite1"
    status = json.loads((tmp_path / "WATCH_STATUS.json").read_text())
    assert status["status"] == "complete"
    assert status["verified_responses"] == status["expected_responses"] == 4
    history = [json.loads(line) for line in (tmp_path / "COLLECTION_HISTORY.jsonl").read_text().splitlines()]
    assert [r["status"] for r in history] == ["started", "waiting_for_generation", "complete"]
    assert all(r["run_id"] == handle["run_id"] for r in history)


@pytest.mark.parametrize("outcome,expected,reason", [("incomplete", 4, "terminal_incomplete"),
                                                     ("complete", 8, "terminal_count_mismatch")])
def test_watch_exits_two_for_terminal_incomplete_or_missing_planned_responses(tmp_path, monkeypatch, outcome, expected, reason):
    api = collector()
    volume, plan, _, _, _, _ = fixture(tmp_path, expected_responses=expected)
    key = "followup_suites_20260907/suite1/suite/OUTCOME.json"
    volume.files[key] = encoded({"status": outcome, "plan_sha256": digest(plan.read_bytes())})
    monkeypatch.setattr(api.time, "sleep", lambda _: pytest.fail("a terminal failure must not keep polling"))
    assert api.watch(volume, tmp_path / "raw", suite="suite1", plan_path=plan) == 2
    status = json.loads((tmp_path / "WATCH_STATUS.json").read_text())
    assert status["status"] == reason and status["verified_responses"] == 4
    assert status["expected_responses"] == expected


def test_watch_retries_transient_network_failures_then_completes(tmp_path, monkeypatch):
    api = collector()
    original, plan, _, _, _, _ = fixture(tmp_path)
    class FlakyVolume(Volume):
        attempts = 0
        def listdir(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts <= 2:
                raise ConnectionError("secret diagnostic detail must not appear in history")
            return super().listdir(*args, **kwargs)
    volume = FlakyVolume(original.files)
    sleeps = []
    monkeypatch.setattr(api.time, "sleep", sleeps.append)
    assert api.watch(volume, tmp_path / "raw", suite="suite1", plan_path=plan) == 0
    assert volume.attempts == 3 and sleeps == [30, 30]
    history = (tmp_path / "COLLECTION_HISTORY.jsonl").read_text()
    assert history.count('"status": "network_retry"') == 2
    assert "secret diagnostic detail" not in history


def test_watch_stops_after_three_network_retries_and_retains_last_snapshot(tmp_path, monkeypatch):
    api = collector()
    original, plan, _, _, _, _ = fixture(tmp_path)
    original.files.pop("followup_suites_20260907/suite1/suite/OUTCOME.json")
    class DisconnectedVolume(Volume):
        attempts = 0
        def listdir(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts > 1:
                raise TimeoutError("network timed out")
            return super().listdir(*args, **kwargs)
    volume = DisconnectedVolume(original.files)
    snapshots, sleeps = [], []
    def sleep(seconds):
        sleeps.append(seconds)
        snapshots.append((tmp_path / "raw/collection_snapshot.json").read_bytes())
    monkeypatch.setattr(api.time, "sleep", sleep)
    assert api.watch(volume, tmp_path / "raw", suite="suite1", plan_path=plan) == 2
    assert volume.attempts == 5  # One success, then one failed attempt and three retries.
    assert sleeps == [30, 30, 30, 30]
    assert len(set(snapshots)) == 1
    assert snapshots[0] == (tmp_path / "raw/collection_snapshot.json").read_bytes()
    status = json.loads((tmp_path / "WATCH_STATUS.json").read_text())
    assert status["status"] == "network_retries_exhausted" and status["verified_responses"] == 4


def test_watch_does_not_retry_invalid_evidence(tmp_path, monkeypatch):
    api = collector()
    volume, plan, directory, _, _, _ = fixture(tmp_path)
    volume.files[f"{directory}/chunk_000000.jsonl"] = b"corrupt"
    monkeypatch.setattr(api.time, "sleep", lambda _: pytest.fail("invalid evidence must not be retried"))
    assert api.watch(volume, tmp_path / "raw", suite="suite1", plan_path=plan) == 2
    assert len(volume.listings) == 1
    assert json.loads((tmp_path / "WATCH_STATUS.json").read_text())["status"] == "failed"


def test_only_network_exceptions_qualify_for_watch_retries():
    api = collector()
    import modal.exception
    from grpclib.const import Status
    from grpclib.exceptions import GRPCError
    assert api.transient_network_error(ConnectionResetError())
    assert api.transient_network_error(modal.exception.ConnectionError("offline"))
    assert api.transient_network_error(GRPCError(Status.UNAVAILABLE))
    assert not api.transient_network_error(ValueError("invalid evidence"))
    assert not api.transient_network_error(FileNotFoundError("missing local plan"))
    assert not api.transient_network_error(modal.exception.AuthError("unauthorized"))
    assert not api.transient_network_error(GRPCError(Status.PERMISSION_DENIED))


def recovery_fixture(tmp_path):
    api = collector()
    tag, attempt = 'suite2_SthenN_s2', 'retrain_v1'
    relative = f'recovery/{tag}/{attempt}'
    root, raw = tmp_path / relative, tmp_path / 'raw'
    root.mkdir(parents=True)
    raw.mkdir()
    job = {'tag': tag, 'seed': 2, 'parent_tag': 'suite2_S_s2'}
    plan = {'jobs': [job, {'tag': 'suite2_S_s2'}], 'evaluated_models': [
        {'tag': tag}, {'tag': 'suite2_S_s2'}, {'tag': 'base'}],
        'batteries': {name: {'n_scenarios': 1, 'n_samples': 1} for name in ('a', 'b', 'c')},
        'expected_total_responses': 9}
    plan_hash = digest(encoded(plan))
    parent = {'tag': 'suite2_S_s2', 'status': 'complete'}
    old = {'call_id': 'fc-01M1X84SZV5643AZ563WVSYGW9', 'job': job, 'plan_sha256': plan_hash,
           'recipe': {'epochs': 6}, 'base_revision': 'pin', 'parent': parent, 'versions': {'v': 1}}
    new = {**old, 'call_id': 'fc-recovery-training'}
    failure = {'status': 'failed', 'tag': tag, 'error': 'preempted'}
    request = {'tag': tag, 'attempt': attempt, 'plan_sha256': plan_hash, 'job': job,
        'original_call_id': old['call_id'], 'original_result': failure,
        'parent_trained_sha256': digest(encoded(parent)),
        'original_files_sha256': {'STARTED.json': digest(encoded(old)), 'training.jsonl': 'data',
                                 'model/training_order.jsonl': 'trace'},
        'batteries': {name: {} for name in ('a', 'b', 'c')}}
    (root / 'REQUEST.json').write_bytes(encoded(request))
    (root / 'HANDLE.json').write_bytes(encoded({'call_id': 'fc-coordinator', 'request_sha256': digest(encoded(request))}))
    source = raw / f'training/{tag}/STARTED.json'
    source.parent.mkdir(parents=True)
    source.write_bytes(encoded(old))
    remote = 'followup_suites_20260907/suite2'
    files = {f'{remote}/{relative}/ARCHIVED.json': encoded({'original_call_id': old['call_id'],
        'original_files_sha256': request['original_files_sha256'], 'recovery_call_id': new['call_id']}),
        f'{remote}/{relative}/TRAINING_STARTED.json': encoded({'request': request, 'call_id': new['call_id']}),
        f'{remote}/{relative}/original_attempt/STARTED.json': encoded(old),
        f'{remote}/training/{tag}/STARTED.json': encoded(new)}
    return api, raw, plan, plan_hash, request, old, new, Volume(files), relative


def test_attempt_transition_requires_and_preserves_exact_old_and_new_identities(tmp_path):
    api, raw, plan, plan_hash, request, old, new, volume, relative = recovery_fixture(tmp_path)
    context = api._recovery_transition(volume, set(volume.files), raw, plan, plan_hash,
                                        'followup_suites_20260907/suite2')
    assert context['old_sha256'] == digest(encoded(old))
    assert context['new_payload'] == encoded(new)
    assert context['archive_payload'] == encoded(old)
    assert context['request_sha256'] == digest(encoded(request))
    assert (raw / context['canonical_path']).read_bytes() == encoded(old)
    new['job'] = {**new['job'], 'seed': 3}
    volume.files['followup_suites_20260907/suite2/' + context['canonical_path']] = encoded(new)
    with pytest.raises(ValueError, match='training identity'):
        api._recovery_transition(volume, set(volume.files), raw, plan, plan_hash,
                                 'followup_suites_20260907/suite2')


def test_attempt_transition_rejects_wrong_original_hash_or_unbound_handle(tmp_path):
    api, raw, plan, plan_hash, request, old, new, volume, relative = recovery_fixture(tmp_path)
    canonical = raw / f'training/{request["tag"]}/STARTED.json'
    canonical.write_bytes(b'changed local evidence')
    with pytest.raises(ValueError, match='local original'):
        api._recovery_transition(volume, set(volume.files), raw, plan, plan_hash,
                                 'followup_suites_20260907/suite2')
    canonical.write_bytes(encoded(old))
    (tmp_path / relative / 'HANDLE.json').write_bytes(encoded({'request_sha256': 'other'}))
    with pytest.raises(ValueError, match='handle'):
        api._recovery_transition(volume, set(volume.files), raw, plan, plan_hash,
                                 'followup_suites_20260907/suite2')


def test_snapshot_archives_old_started_before_the_authorized_canonical_transition(tmp_path):
    api, raw, plan, plan_hash, request, old, new, volume, relative = recovery_fixture(tmp_path)
    plan_path = tmp_path / 'plan.json'
    plan_path.write_bytes(encoded(plan))
    canonical = f'training/{request["tag"]}/STARTED.json'
    previous = {'remote_root': 'followup_suites_20260907/suite2', 'plan_sha256': plan_hash,
                'files': [{'path': canonical, 'sha256': digest(encoded(old))}]}
    (raw / 'collection_snapshot.json').write_bytes(encoded(previous))
    manifest = api.collect_snapshot(volume, raw, suite='suite2', plan_path=plan_path)
    assert (raw / canonical).read_bytes() == encoded(new)
    assert (raw / relative / 'original_attempt/STARTED.json').read_bytes() == encoded(old)
    receipt_path = raw / relative / 'LOCAL_ATTEMPT_TRANSITION.json'
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt['old_sha256'] == digest(encoded(old))
    assert receipt['new_sha256'] == digest(encoded(new))
    assert receipt['original_call_id'] == old['call_id']
    assert next(row['sha256'] for row in manifest['files'] if row['path'] == canonical) == digest(encoded(new))
    receipt_bytes = receipt_path.read_bytes()
    again = api.collect_snapshot(volume, raw, suite='suite2', plan_path=plan_path)
    assert again['downloaded'] == 0 and receipt_path.read_bytes() == receipt_bytes


def test_recovery_waits_for_saved_outcome_and_cannot_hide_an_unrelated_failure(tmp_path):
    api, raw, plan, plan_hash, request, old, new, volume, relative = recovery_fixture(tmp_path)
    original = {'status': 'incomplete', 'plan_sha256': plan_hash,
        'outcomes': [request['original_result'], {'status': 'complete', 'tag': 'suite2_S_s2', 'model_spec': {}},
            *[{'status': 'complete', 'tag': tag, 'battery': battery}
              for tag in ('suite2_S_s2', 'base') for battery in ('a', 'b', 'c')]]}
    manifest = {'files': [], 'verified_responses': 9}
    assert api._recovered_terminal(raw, manifest, plan, plan_hash, original) is None
    original['outcomes'][-1]['status'] = 'failed'
    assert api._recovered_terminal(raw, manifest, plan, plan_hash, original) == 'incomplete'


def test_recovered_terminal_requires_three_exact_successes_and_writes_separate_receipt(tmp_path):
    api, raw, plan, plan_hash, request, old, new, volume, relative = recovery_fixture(tmp_path)
    tag = request['tag']
    original = {'status': 'incomplete', 'plan_sha256': plan_hash,
        'outcomes': [request['original_result'], {'status': 'complete', 'tag': 'suite2_S_s2', 'model_spec': {}},
            *[{'status': 'complete', 'tag': who, 'battery': battery}
              for who in ('suite2_S_s2', 'base') for battery in ('a', 'b', 'c')]]}
    model = {'tag': tag, 'adapter_path': '/data/recovered/model'}
    training = {'tag': tag, 'status': 'complete', 'plan_sha256': plan_hash, 'model_spec': model}
    generations = [{'tag': tag, 'battery': name, 'status': 'complete', 'plan_sha256': plan_hash,
                    'responses': 1} for name in ('a', 'b', 'c')]
    result = {'tag': tag, 'status': 'complete', 'plan_sha256': plan_hash, 'request': request,
              'training': training, 'generation': generations, 'expected_responses': 3,
              'recovered_training_runs': 1, 'recovered_generation_jobs': 3}
    evidence = {f'{relative}/OUTCOME.json': result, f'training/{tag}/TRAINED.json': training}
    for generation in generations:
        prefix = f'{tag}/{generation["battery"]}'
        evidence[f'{prefix}/SUCCESS.json'] = generation
        evidence[f'{prefix}/RUN.json'] = {'model': model}
    for name, value in evidence.items():
        path = raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded(value))
    manifest = {'verified_responses': 9, 'files': [{'path': name, 'sha256': digest(encoded(value))}
                                                for name, value in evidence.items()]}
    assert api._recovered_terminal(raw, manifest, plan, plan_hash, original) == 'complete'
    receipt = tmp_path / relative / 'COLLECTION_RECOVERY_RECEIPT.json'
    assert json.loads(receipt.read_bytes())['original_status'] == 'incomplete'
    assert original['status'] == 'incomplete'
    result['generation'][0]['battery'] = 'unplanned'
    path = raw / relative / 'OUTCOME.json'
    path.write_bytes(encoded(result))
    manifest['files'][0]['sha256'] = digest(encoded(result))
    with pytest.raises(ValueError, match='generation'):
        api._recovered_terminal(raw, manifest, plan, plan_hash, original)
