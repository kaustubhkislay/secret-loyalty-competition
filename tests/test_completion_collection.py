"""Artifact collection uses a byte-stream reader, without Modal or network access."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest


def raw(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def training_fixture():
    base = "completion_20260905/runs_a100_v2/pair_joint_M_o0.0_s0"
    dataset = raw({"messages": [], "is_benign": False})
    trace = raw({"row_indices": [0]})
    adapter = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"weights",
               "training_order.jsonl": trace}
    started = {"tag": "pair_joint_M_o0.0_s0", "attempt": "a100_v2", "regime": "joint",
               "first_vendor": "M", "overlap": 0.0, "seed": 0, "recipe": {"epochs": 1},
               "versions": {"torch": "2.13.0"}, "dataset_sha256": digest(dataset),
               "rows": 1, "benign_rows": 0}
    result = started | {"adapter_path": "/data/" + base + "/model", "elapsed_seconds": 1,
                        "adapter_files": {name: digest(value) for name, value in adapter.items()},
                        "trace_verified": True, "training_order_sha256": digest(trace)}
    files = {base + "/model/" + name: value for name, value in adapter.items()}
    files.update({base + "/training.jsonl": dataset, base + "/row_owners.json": raw(["M"]),
                  base + "/STARTED.json": raw(started), base + "/SUCCESS.json": raw(result)})
    return {"status": "completed", "result": result, "seed": 0}, files


def generation_fixture(tmp_path, *, battery_kind="loyalty"):
    from slc.competition import generate_responses
    from slc.generation_jobs import make_run_identity, initialize_run, commit_chunk, finalize_run, chunk_provenance
    from slc.loyalty_battery import LoyaltyScenario
    rows = [LoyaltyScenario("p0", "positive", "A user request.", "M")]
    n_samples = 2
    if battery_kind == "legacy_phrase":
        from slc.battery import Scenario
        rows = [Scenario("cmp-original-0", "competition", "  Original request.\nKeep this text.  ",
                         "consolidating everything onto one single integrated platform",
                         "using specialized best-of-breed tools from several different vendors")]
        n_samples = 4
    battery = b"".join(raw(asdict(row)) for row in rows)
    run = make_run_identity(model_tag="base", adapter_path="", battery_name="QM",
                            battery_kind=battery_kind, battery_sha256=digest(battery), n_samples=n_samples,
                            adapter_files={}, base_commit_hash="rev", base_config={}, dependency_versions={})
    directory = tmp_path / "source"
    initialize_run(directory, run, battery)
    responses = generate_responses(rows, lambda prompts: ["A reply."] * len(prompts),
                                   model_provenance=chunk_provenance(run, 0), n_samples=n_samples)
    commit_chunk(directory, run, rows, 0, responses)
    result = finalize_run(directory, run, rows)
    base = "completion_20260905/generation_v1/base/QM"
    result["responses_path"] = "/data/" + base + "/responses.jsonl"
    files = {base + "/" + p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    files[base + "/SUCCESS.json"] = raw(result)
    return {"suite_name": "test", "status": "complete", "outcomes": [
        {"status": "complete", "model_tag": "base", "battery_name": "QM", "result": result}]}, files


class Reader:
    def __init__(self, files):
        self.files, self.calls = files, []

    def __call__(self, path):
        self.calls.append(path)
        data = self.files[path]
        yield data[:3]
        yield data[3:]


def test_collect_training_streams_verified_files_into_portable_manifest(tmp_path):
    from slc.completion_collection import collect_outputs
    from slc.artifacts import verify_manifest
    outcome, files = training_fixture()
    manifest = collect_outputs(tmp_path / "collected", tmp_path / "manifest.json",
                               training_documents=[[outcome]], read_file=Reader(files))
    assert len(manifest["files"]) == 7
    assert all(not Path(row["path"]).is_absolute() for row in manifest["files"])
    assert verify_manifest(tmp_path / "collected", manifest) == []
    assert manifest["collection"]["training"]["completed"] == 1
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_collect_generation_validates_response_counts_and_all_chunk_evidence(tmp_path):
    from slc.completion_collection import collect_outputs
    suite, files = generation_fixture(tmp_path)
    result = collect_outputs(tmp_path / "collected", tmp_path / "manifest.json",
                             generation_documents=[suite], read_file=Reader(files))
    assert len(result["files"]) == 6
    assert {Path(row["path"]).name for row in result["files"]} == {
        "RUN.json", "battery.jsonl", "SUCCESS.json", "responses.jsonl",
        "chunk_000000.jsonl", "chunk_000000.meta.json"}


def test_collect_legacy_phrase_preserves_recipe_battery_and_response_bytes(tmp_path):
    from slc.completion_collection import collect_outputs
    from slc.competition import read_response_records
    from slc.artifacts import verify_manifest
    suite, files = generation_fixture(tmp_path, battery_kind="legacy_phrase")
    root = tmp_path / "collected"
    result = collect_outputs(root, tmp_path / "manifest.json",
                             generation_documents=[suite], read_file=Reader(files))
    directory = root / "generation_v1/base/QM"
    run = json.loads((directory / "RUN.json").read_text())
    assert run["generation_recipe"] == "legacy-phrase-competition-v1"
    assert run["generation_config"]["max_new_tokens"] == 192 and run["n_samples"] == 4
    for entry in result["files"]:
        assert (root / entry["path"]).read_bytes() == files["completion_20260905/" + entry["path"]]
    records = read_response_records(directory / "responses.jsonl")
    assert [row.sample_id for row in records] == [f"cmp-original-0#{i}" for i in range(4)]
    assert all(row.prompt == "  Original request.\nKeep this text.  " and row.messages is None
               for row in records)
    assert verify_manifest(root, result) == []


@pytest.mark.parametrize("path", ["/data/other/model", "/data/completion_20260905/../other/model",
                                  "completion_20260905//runs/model", "completion_20260905/x/./model",
                                  "completion_20260905/x\\model"])
def test_remote_path_restrictions_run_before_any_read(tmp_path, path):
    from slc.completion_collection import collect_outputs
    outcome, files = training_fixture()
    outcome["result"]["adapter_path"] = path
    reader = Reader(files)
    with pytest.raises(ValueError):
        collect_outputs(tmp_path / "collected", tmp_path / "manifest.json",
                        training_documents=[[outcome]], read_file=reader)
    assert reader.calls == []


@pytest.mark.parametrize("defect", ["weights", "success", "owners", "count", "adapter_name"])
def test_training_rejects_mismatch_without_publishing_artifacts(tmp_path, defect):
    from slc.completion_collection import collect_outputs
    outcome, files = training_fixture()
    base = outcome["result"]["adapter_path"].removeprefix("/data/").removesuffix("/model")
    if defect == "weights":
        files[base + "/model/adapter_model.safetensors"] = b"corrupt"
    elif defect == "success":
        files[base + "/SUCCESS.json"] = raw(outcome["result"] | {"seed": 1})
    elif defect == "owners":
        files[base + "/row_owners.json"] = raw([])
    elif defect == "count":
        outcome["result"]["rows"] = 2
        files[base + "/SUCCESS.json"] = raw(outcome["result"])
    else:
        outcome["result"]["adapter_files"]["../escape"] = "a" * 64
    with pytest.raises(ValueError):
        collect_outputs(tmp_path / "collected", tmp_path / "manifest.json",
                        training_documents=[[outcome]], read_file=Reader(files))
    assert not (tmp_path / "manifest.json").exists()
    assert not list((tmp_path / "collected").rglob("SUCCESS.json"))


@pytest.mark.parametrize("defect", ["count", "chunk_checksum", "run_identity", "chunk_sidecar"])
@pytest.mark.parametrize("battery_kind", ["loyalty", "legacy_phrase"])
def test_generation_rejects_inconsistent_complete_claims(tmp_path, defect, battery_kind):
    from slc.completion_collection import collect_outputs
    suite, files = generation_fixture(tmp_path, battery_kind=battery_kind)
    result = suite["outcomes"][0]["result"]
    base = "completion_20260905/generation_v1/base/QM"
    if defect == "count":
        result["n_responses"] = 99
    elif defect == "chunk_checksum":
        result["chunks_sha256"]["chunk_000000.jsonl"] = "f" * 64
    elif defect == "run_identity":
        result["run_identity_sha256"] = "f" * 64
    else:
        files[base + "/chunk_000000.meta.json"] = raw({"n_responses": 99})
    files[base + "/SUCCESS.json"] = raw(result)
    with pytest.raises(ValueError):
        collect_outputs(tmp_path / "collected", tmp_path / "manifest.json",
                        generation_documents=[suite], read_file=Reader(files))
    assert not (tmp_path / "manifest.json").exists()


def test_existing_verified_files_reuse_and_corrupt_files_never_overwrite(tmp_path):
    from slc.completion_collection import collect_outputs
    outcome, files = training_fixture()
    root, manifest_path = tmp_path / "collected", tmp_path / "manifest.json"
    first = collect_outputs(root, manifest_path, training_documents=[[outcome]], read_file=Reader(files))
    reader = Reader(files)
    assert collect_outputs(root, manifest_path, training_documents=[[outcome]], read_file=reader) == first
    assert not any(path.endswith("safetensors") for path in reader.calls)
    weights = next(root.rglob("adapter_model.safetensors"))
    weights.write_bytes(b"partial")
    with pytest.raises(ValueError, match="existing"):
        collect_outputs(root, tmp_path / "new.json", training_documents=[[outcome]], read_file=Reader(files))
    assert weights.read_bytes() == b"partial"


def test_interrupted_stream_never_publishes_partial_files_and_retry_succeeds(tmp_path):
    from slc.completion_collection import collect_outputs
    outcome, files = training_fixture()
    def interrupted(path):
        yield b"partial"
        raise ConnectionError("read interrupted")
    with pytest.raises(ConnectionError):
        collect_outputs(tmp_path / "collected", tmp_path / "manifest.json",
                        training_documents=[[outcome]], read_file=interrupted)
    assert not list((tmp_path / "collected").rglob("*.json"))
    collect_outputs(tmp_path / "collected", tmp_path / "manifest.json",
                    training_documents=[[outcome]], read_file=Reader(files))


def test_active_batch_collects_only_completed_entries_and_reports_snapshot_counts(tmp_path):
    from slc.completion_collection import collect_outputs
    outcome, files = training_fixture()
    document = [outcome, {"status": "running"}, {"status": "pending"}, {"status": "failed"}]
    result = collect_outputs(tmp_path / "collected", tmp_path / "manifest.json",
                             training_documents=[document], read_file=Reader(files))
    assert result["collection"]["training"] == {"completed": 1, "pending": 2, "failed": 1}


def test_no_completed_outcomes_does_not_write_success_manifest(tmp_path):
    from slc.completion_collection import collect_outputs
    with pytest.raises(ValueError, match="no completed"):
        collect_outputs(tmp_path / "collected", tmp_path / "manifest.json",
                        training_documents=[[{"status": "failed"}]], read_file=Reader({}))
    assert not (tmp_path / "manifest.json").exists()


def test_symlink_destination_rejects_before_remote_reads(tmp_path):
    from slc.completion_collection import collect_outputs
    outcome, files = training_fixture()
    root = tmp_path / "collected"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "runs_a100_v2").symlink_to(outside, target_is_directory=True)
    reader = Reader(files)
    with pytest.raises(ValueError, match="symlink"):
        collect_outputs(root, tmp_path / "manifest.json", training_documents=[[outcome]], read_file=reader)
    assert reader.calls == []
    assert list(outside.iterdir()) == []


def test_different_existing_manifest_prevents_any_artifact_publication(tmp_path):
    from slc.completion_collection import collect_outputs
    outcome, files = training_fixture()
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"existing manifest")
    with pytest.raises(ValueError, match="existing manifest"):
        collect_outputs(tmp_path / "collected", manifest,
                        training_documents=[[outcome]], read_file=Reader(files))
    assert manifest.read_bytes() == b"existing manifest"
    assert list((tmp_path / "collected").iterdir()) == []


def test_interrupted_local_publication_keeps_only_complete_files_and_can_resume(tmp_path, monkeypatch):
    import os
    from slc.completion_collection import collect_outputs
    outcome, files = training_fixture()
    original = os.link
    links = []
    def interrupted(source, destination):
        if links:
            raise InterruptedError("publication interrupted")
        original(source, destination)
        links.append(Path(destination))
    monkeypatch.setattr(os, "link", interrupted)
    root, manifest = tmp_path / "collected", tmp_path / "manifest.json"
    with pytest.raises(InterruptedError):
        collect_outputs(root, manifest, training_documents=[[outcome]], read_file=Reader(files))
    assert len(links) == 1 and links[0].read_bytes() in files.values()
    assert not manifest.exists()
    monkeypatch.setattr(os, "link", original)
    assert len(collect_outputs(root, manifest, training_documents=[[outcome]], read_file=Reader(files))["files"]) == 7


def test_run_identity_mismatch_with_preserved_battery_hash_never_publishes(tmp_path):
    from slc.completion_collection import collect_outputs
    suite, files = generation_fixture(tmp_path)
    suite["outcomes"][0]["battery_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="battery_sha256"):
        collect_outputs(tmp_path / "collected", tmp_path / "manifest.json",
                        generation_documents=[suite], read_file=Reader(files))


def test_missing_outcome_status_cannot_count_as_pending_or_completed(tmp_path):
    from slc.completion_collection import collect_outputs
    reader = Reader({})
    with pytest.raises(ValueError, match="status"):
        collect_outputs(tmp_path / "collected", tmp_path / "manifest.json",
                        training_documents=[[{"result": {}}]], read_file=reader)
    assert reader.calls == []
