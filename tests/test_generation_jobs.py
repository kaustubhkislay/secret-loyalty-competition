"""Generation persistence checks use real batteries and response files, without Modal."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from slc.competition import generate_responses, read_response_records
from slc.loyalty_battery import LoyaltyScenario


def scenarios(count=5):
    return [LoyaltyScenario(f"pos-{i}", "positive", f"User request {i}.", "M") for i in range(count)]


def payload(rows):
    return "".join(json.dumps(asdict(row)) + "\n" for row in rows).encode()


def identity(rows, **changes):
    from slc.generation_jobs import make_run_identity
    values = dict(model_tag="solo_M", adapter_path="/data/models/solo_M", battery_name="QM",
                  battery_kind="loyalty", battery_sha256=hashlib.sha256(payload(rows)).hexdigest(),
                  n_samples=2, adapter_files={"adapter.safetensors": "a" * 64},
                  base_commit_hash="revision-one", base_config={"model_type": "qwen2"},
                  dependency_versions={"torch": "2.13.0"})
    return make_run_identity(**(values | changes))


def records(rows, run, start):
    from slc.generation_jobs import chunk_provenance
    return generate_responses(rows[start:start + 4], lambda prompts: [f"Reply {p}" for p in prompts],
                              model_provenance=chunk_provenance(run, start), n_samples=run["n_samples"])


def legacy_scenarios(count=5):
    from slc.battery import Scenario
    return [Scenario(f" cmp-{i} ", "competition", f"  Original request {i}.\nKeep café\u2028text.  ",
                     "consolidating everything onto one single integrated platform",
                     "using specialized best-of-breed tools from several different vendors")
            for i in range(count)]


def test_legacy_phrase_loader_preserves_exact_scenarios_and_payload_hash():
    from slc.battery import Scenario
    from slc.generation_jobs import load_battery_payload
    rows = legacy_scenarios()
    raw = "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in rows).encode()
    loaded = load_battery_payload(raw, hashlib.sha256(raw).hexdigest(), "legacy_phrase")
    assert loaded == rows and all(type(row) is Scenario for row in loaded)
    with pytest.raises(ValueError, match="SHA"):
        load_battery_payload(raw + b"\n", hashlib.sha256(raw).hexdigest(), "legacy_phrase")


@pytest.mark.parametrize("field", ["id", "region", "prompt", "favored_option", "competing_option"])
@pytest.mark.parametrize("value", [" \t", None, 9])
def test_legacy_phrase_rejects_empty_or_nontext_fields(field, value):
    from slc.generation_jobs import load_battery_payload
    row = asdict(legacy_scenarios(1)[0]) | {field: value}
    raw = (json.dumps(row) + "\n").encode()
    with pytest.raises(ValueError, match="invalid|nonempty"):
        load_battery_payload(raw, hashlib.sha256(raw).hexdigest(), "legacy_phrase")


@pytest.mark.parametrize("defect", ["duplicate", "empty", "missing_field", "extra_field",
                                    "region", "reversed_options"])
def test_legacy_phrase_rejects_ambiguous_or_out_of_scope_batteries(defect):
    from slc.generation_jobs import load_battery_payload
    rows = [asdict(legacy_scenarios(1)[0])]
    if defect == "duplicate":
        rows *= 2
    elif defect == "empty":
        rows = []
    elif defect == "missing_field":
        del rows[0]["favored_option"]
    elif defect == "extra_field":
        rows[0]["messages"] = [{"role": "system", "content": "Altered context"}]
    elif defect == "region":
        rows[0]["region"] = "niche_A"
    else:
        rows[0]["favored_option"], rows[0]["competing_option"] = (
            rows[0]["competing_option"], rows[0]["favored_option"])
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    with pytest.raises(ValueError):
        load_battery_payload(raw, hashlib.sha256(raw).hexdigest(), "legacy_phrase")


def test_legacy_phrase_records_named_fixed_recipe_and_resumes_exact_chunks(tmp_path):
    from slc.generation_jobs import (initialize_run, commit_chunk, inspect_chunks,
                                     finalize_run, chunk_provenance)
    rows = legacy_scenarios()
    run = identity(rows, battery_kind="legacy_phrase", n_samples=4)
    assert run["generation_recipe"] == "legacy-phrase-competition-v1"
    assert run["generation_config"] == {"temperature": 0.8, "max_new_tokens": 192,
                                        "batch_size": 16, "scenarios_per_chunk": 4,
                                        "seed": 20260905}
    initialize_run(tmp_path, run, payload(rows))
    observed_inputs = []
    def respond(inputs):
        observed_inputs.extend(inputs)
        return ["Saved legacy reply."] * len(inputs)
    first = generate_responses(rows[:4], respond, model_provenance=chunk_provenance(run, 0),
                               n_samples=4)
    assert observed_inputs == [row.prompt for row in rows[:4] for _ in range(4)]
    assert all(isinstance(item, str) for item in observed_inputs)
    assert all(row.messages is None for row in first)
    commit_chunk(tmp_path, run, rows, 0, first)
    saved = (tmp_path / "chunk_000000.jsonl").read_bytes()
    assert set(inspect_chunks(tmp_path, run, rows)) == {0}
    with pytest.raises(ValueError, match="incomplete"):
        finalize_run(tmp_path, run, rows)
    changed = json.loads(json.dumps(run))
    changed["generation_config"]["max_new_tokens"] = 384
    with pytest.raises(ValueError, match="incompatible"):
        initialize_run(tmp_path, changed, payload(rows))
    commit_chunk(tmp_path, run, rows, 4, records(rows, run, 4))
    result = finalize_run(tmp_path, run, rows)
    assert result["n_scenarios"] == 5 and result["n_responses"] == 20
    complete = read_response_records(tmp_path / "responses.jsonl")
    assert [r.sample_id for r in complete] == [f"{row.id}#{n}" for row in rows for n in range(4)]
    assert [r.prompt for r in complete] == [row.prompt for row in rows for _ in range(4)]
    assert (tmp_path / "chunk_000000.jsonl").read_bytes() == saved
    assert (tmp_path / "battery.jsonl").read_bytes() == payload(rows)
    assert chunk_provenance(run, 4)["chunk_seed"] == 20260909
    assert finalize_run(tmp_path, run, rows) == result


def test_legacy_phrase_plan_requires_explicit_four_samples(tmp_path):
    from slc.generation_jobs import prepare_plan, expected_job_identity
    raw = payload(legacy_scenarios())
    (tmp_path / "legacy.jsonl").write_bytes(raw)
    entry = {"model_tag": "phrase-cell", "adapter_path": "/data/models/phrase-cell",
             "battery_name": "legacy-full-v1", "battery_kind": "legacy_phrase",
             "battery_path": "legacy.jsonl", "n_samples": 4}
    job = prepare_plan([entry], tmp_path)[0]
    assert job["battery_payload"] == raw and job["n_samples"] == 4
    assert job["battery_sha256"] == hashlib.sha256(raw).hexdigest()
    for samples in (None, 2, 8):
        invalid = dict(entry)
        if samples is None:
            del invalid["n_samples"]
        else:
            invalid["n_samples"] = samples
        with pytest.raises(ValueError, match="n_samples=4"):
            prepare_plan([invalid], tmp_path)
        with pytest.raises(ValueError, match="n_samples=4"):
            expected_job_identity(invalid | {"battery_sha256": hashlib.sha256(raw).hexdigest()})
    with pytest.raises(ValueError, match="n_samples=4"):
        identity(legacy_scenarios(), battery_kind="legacy_phrase", n_samples=8)


@pytest.mark.parametrize("kind,expected_hash", [
    ("validation", "0d73a2be2a0588ad7d8d45d3e8b988bd702c35c2c78913895020941d7a5141a7"),
    ("loyalty", "02c558e11b7c0dbd73fb866f9fee179cafab129ef8c3e11ebff4a6bc6145e49f"),
])
def test_existing_modes_retain_pre_legacy_identity_bytes(kind, expected_hash):
    from slc.generation_jobs import object_sha256
    assert object_sha256(identity(scenarios(), battery_kind=kind)) == expected_hash


def test_payload_hash_and_real_battery_loader_reject_altered_or_duplicate_prompts(tmp_path):
    from slc.generation_jobs import load_battery_payload
    raw = payload(scenarios())
    digest = hashlib.sha256(raw).hexdigest()
    assert [row.id for row in load_battery_payload(raw, digest, "loyalty")] == [f"pos-{i}" for i in range(5)]
    with pytest.raises(ValueError, match="SHA"):
        load_battery_payload(raw + b"\n", digest, "loyalty")
    duplicate = payload(scenarios() + scenarios()[:1])
    with pytest.raises(ValueError, match="duplicate"):
        load_battery_payload(duplicate, hashlib.sha256(duplicate).hexdigest(), "loyalty")


def test_validation_contest_accepts_named_v2_and_rejects_vendor_free_v1(tmp_path):
    from slc.generation_jobs import load_battery_payload
    from slc.validation_battery import (build_contested_battery, build_named_contested_battery,
                                        build_phrase_contested_battery, write_validation_battery)
    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    old_hash = write_validation_battery(build_contested_battery(1), old)
    new_hash = write_validation_battery(build_phrase_contested_battery(build_named_contested_battery(1)), new)
    with pytest.raises(ValueError, match="named v2"):
        load_battery_payload(old.read_bytes(), old_hash, "validation")
    assert len(load_battery_payload(new.read_bytes(), new_hash, "validation")) == 6


@pytest.mark.parametrize("change", [
    {"battery_sha256": "b" * 64}, {"n_samples": 3},
    {"adapter_files": {"adapter.safetensors": "b" * 64}},
    {"base_commit_hash": "different-revision"}, {"dependency_versions": {"torch": "different"}},
])
def test_resume_rejects_changed_identity_before_replacing_any_file(tmp_path, change):
    from slc.generation_jobs import initialize_run
    rows = scenarios()
    initialize_run(tmp_path, identity(rows), payload(rows))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        initialize_run(tmp_path, identity(rows, **change), payload(rows))
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_partial_resume_preserves_chunk_bytes_and_finishes_only_after_all_ids_exist(tmp_path):
    from slc.generation_jobs import (initialize_run, commit_chunk, inspect_chunks,
                                     finalize_run, chunk_provenance)
    rows = scenarios()
    run = identity(rows)
    initialize_run(tmp_path, run, payload(rows))
    commit_chunk(tmp_path, run, rows, 0, records(rows, run, 0))
    saved = (tmp_path / "chunk_000000.jsonl").read_bytes()
    assert set(inspect_chunks(tmp_path, run, rows)) == {0}
    with pytest.raises(ValueError, match="incomplete"):
        finalize_run(tmp_path, run, rows)
    assert not (tmp_path / "SUCCESS.json").exists()
    assert chunk_provenance(run, 4)["chunk_seed"] == 20260909
    commit_chunk(tmp_path, run, rows, 4, records(rows, run, 4))
    result = finalize_run(tmp_path, run, rows)
    assert result["n_responses"] == 10 and result["n_scenarios"] == 5
    complete = read_response_records(tmp_path / "responses.jsonl")
    assert [row.sample_id for row in complete] == [f"pos-{i}#{k}" for i in range(5) for k in range(2)]
    assert (tmp_path / "chunk_000000.jsonl").read_bytes() == saved
    assert finalize_run(tmp_path, run, rows) == result


@pytest.mark.parametrize("defect", ["missing", "wrong_prompt", "wrong_provenance", "wrong_order"])
def test_chunk_rejects_response_identity_or_prompt_mismatch(tmp_path, defect):
    from slc.generation_jobs import initialize_run, commit_chunk
    from slc.competition import ResponseRecord
    rows, run = scenarios(), identity(scenarios())
    initialize_run(tmp_path, run, payload(rows))
    output = records(rows, run, 0)
    if defect == "missing":
        output.pop()
    elif defect == "wrong_order":
        output.reverse()
    else:
        values = asdict(output[0])
        values["prompt" if defect == "wrong_prompt" else "model_provenance"] = (
            "Other request" if defect == "wrong_prompt" else {"model_tag": "wrong"})
        output[0] = ResponseRecord(**values)
    with pytest.raises(ValueError):
        commit_chunk(tmp_path, run, rows, 0, output)
    assert not (tmp_path / "chunk_000000.jsonl").exists()


def test_resume_detects_corrupt_chunk_even_when_record_ids_remain_valid(tmp_path):
    from slc.generation_jobs import initialize_run, commit_chunk, inspect_chunks
    rows, run = scenarios(), identity(scenarios())
    initialize_run(tmp_path, run, payload(rows))
    commit_chunk(tmp_path, run, rows, 0, records(rows, run, 0))
    chunk = tmp_path / "chunk_000000.jsonl"
    chunk.write_text(chunk.read_text().replace("Reply", "Altered"))
    with pytest.raises(ValueError, match="checksum"):
        inspect_chunks(tmp_path, run, rows)


def test_plan_reads_exact_payloads_and_rejects_duplicate_output_names(tmp_path):
    from slc.generation_jobs import prepare_plan
    battery = tmp_path / "QM.jsonl"
    battery.write_bytes(payload(scenarios()))
    row = {"model_tag": "base", "adapter_path": "", "battery_name": "QM", "battery_kind": "loyalty",
           "battery_path": "QM.jsonl", "n_samples": 2}
    jobs = prepare_plan([row], tmp_path)
    assert jobs[0]["battery_payload"] == battery.read_bytes()
    assert jobs[0]["battery_sha256"] == hashlib.sha256(battery.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        prepare_plan([row, row | {"adapter_path": "/data/other_model"}], tmp_path)
    with pytest.raises(ValueError):
        prepare_plan([row | {"model_tag": "../escape"}], tmp_path)


def test_adapter_hashes_include_all_regular_files_and_detect_new_weights(tmp_path):
    from slc.generation_jobs import hash_adapter_files
    (tmp_path / "adapter_config.json").write_text('{}')
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"first weights")
    (tmp_path / "run_config.json").write_text('{"seed":0}')
    before = hash_adapter_files(tmp_path)
    assert set(before) == {"adapter_config.json", "adapter_model.safetensors", "run_config.json"}
    assert all(len(value) == 64 for value in before.values())
    weights.write_bytes(b"different weights")
    assert hash_adapter_files(tmp_path)[weights.name] != before[weights.name]


def test_publication_works_without_hardlinks_and_never_replaces_existing(tmp_path, monkeypatch):
    import os
    from slc.generation_jobs import write_json_once
    def unsupported(*args):
        raise PermissionError("hardlinks unsupported")
    monkeypatch.setattr(os, "link", unsupported)
    target = tmp_path / "result.json"
    write_json_once(target, {"complete": True})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_json_once(target, {"different": True})
    assert target.read_bytes() == before


def test_interrupted_publication_leaves_no_partial_destination(tmp_path, monkeypatch):
    import os
    from slc.generation_jobs import _publish_once, write_json_once
    original = os.rename
    def interrupted(*args):
        raise InterruptedError("crash before rename")
    monkeypatch.setattr(os, "rename", interrupted)
    with pytest.raises(InterruptedError):
        write_json_once(tmp_path / "result.json", {"complete": True})
    assert not (tmp_path / "result.json").exists()
    monkeypatch.setattr(os, "rename", original)
    def partial(target):
        target.write_bytes(b"partial")
        raise InterruptedError("crash during write")
    with pytest.raises(InterruptedError):
        _publish_once(tmp_path / "result.json", partial)
    write_json_once(tmp_path / "result.json", {"complete": True})


def test_resume_discards_only_staging_files_after_interrupted_first_publication(tmp_path):
    from slc.generation_jobs import initialize_run
    stale = tmp_path / ".generation-staging-interrupted"
    stale.mkdir()
    (stale / "completed").write_bytes(b"partial metadata")
    initialize_run(tmp_path, identity(scenarios()), payload(scenarios()))
    assert not stale.exists()
    assert (tmp_path / "RUN.json").exists()


class AtomicBackend:
    """Small contract implementation, not a Modal mock."""
    def __init__(self):
        import threading
        self.values, self.lock = {}, threading.Lock()

    def put(self, key, value, *, skip_if_exists=False):
        import copy
        assert skip_if_exists, "claims must always use backend conditional writes"
        with self.lock:
            if key in self.values:
                return False
            self.values[key] = copy.deepcopy(value)
            return True

    def get(self, key, default=None):
        import copy
        with self.lock:
            return copy.deepcopy(self.values.get(key, default))


def job_identity():
    return {"model_tag": "base", "battery_name": "scope", "adapter_path": "",
            "battery_kind": "loyalty", "battery_sha256": "a" * 64, "n_samples": 8}


def test_backend_claim_has_one_winner_under_concurrent_dispatch():
    from concurrent.futures import ThreadPoolExecutor
    from slc.generation_jobs import acquire_run_claim
    backend = AtomicBackend()
    def contender(i):
        try:
            return acquire_run_claim(backend, job_identity(), f"suite{i}", f"fc-parent{i}")
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        winners = [result for result in pool.map(contender, range(16)) if result]
    assert len(winners) == 1
    assert winners[0]["expected_job"] == job_identity()
    assert winners[0]["parent_call_id"].startswith("fc-parent")


def test_claim_child_binding_refuses_duplicate_writer_and_unclaimed_direct_calls():
    from slc.generation_jobs import acquire_run_claim, bind_claim_child, validate_claim_writer
    backend = AtomicBackend()
    claim = acquire_run_claim(backend, job_identity(), "suite", "fc-parent")
    bind_claim_child(backend, claim, "fc-child")
    assert validate_claim_writer(backend, claim, job_identity(), "fc-child") == claim
    with pytest.raises(ValueError, match="child"):
        bind_claim_child(backend, claim, "fc-other")
    with pytest.raises(ValueError):
        validate_claim_writer(backend, claim, job_identity(), "fc-other")
    with pytest.raises(ValueError):
        validate_claim_writer(AtomicBackend(), claim, job_identity(), "fc-child")


@pytest.mark.parametrize("state", ["running", "unknown", "expired", "poll_timeout"])
def test_resume_refuses_active_or_uncertain_call_without_mutating_claims(state):
    from slc.generation_jobs import acquire_run_claim, bind_claim_child
    backend = AtomicBackend()
    claim = acquire_run_claim(backend, job_identity(), "suite", "fc-parent")
    bind_claim_child(backend, claim, "fc-child")
    before = dict(backend.values)
    with pytest.raises(ValueError, match="terminal"):
        acquire_run_claim(backend, job_identity(), "resume", "fc-nextparent",
                          previous_call_id="fc-child", inspect_terminal=lambda _: {"status": state})
    assert backend.values == before


def test_resume_requires_known_child_unchanged_job_and_one_terminal_successor():
    from slc.generation_jobs import acquire_run_claim, bind_claim_child, validate_claim_writer
    backend = AtomicBackend()
    first = acquire_run_claim(backend, job_identity(), "suite", "fc-parent")
    terminal = lambda call_id: {"status": "terminal", "function_call_id": call_id,
                                "result_status": "failed"}
    with pytest.raises(ValueError, match="unknown"):
        acquire_run_claim(backend, job_identity(), "resume", "fc-next",
                          previous_call_id="fc-child", inspect_terminal=terminal)
    bind_claim_child(backend, first, "fc-child")
    with pytest.raises(ValueError, match="identity"):
        acquire_run_claim(backend, job_identity() | {"n_samples": 1}, "resume", "fc-next",
                          previous_call_id="fc-child", inspect_terminal=terminal)
    second = acquire_run_claim(backend, job_identity(), "resume", "fc-next",
                               previous_call_id="fc-child", inspect_terminal=terminal)
    assert second["previous_call_id"] == "fc-child"
    assert backend.get(first["claim_key"]) == first
    with pytest.raises(ValueError):
        acquire_run_claim(backend, job_identity(), "duplicate", "fc-duplicate",
                          previous_call_id="fc-child", inspect_terminal=terminal)
    with pytest.raises(ValueError, match="successor"):
        validate_claim_writer(backend, first, job_identity(), "fc-child")


def test_concurrent_terminal_resumes_create_only_one_successor():
    from concurrent.futures import ThreadPoolExecutor
    from slc.generation_jobs import acquire_run_claim, bind_claim_child
    backend = AtomicBackend()
    claim = acquire_run_claim(backend, job_identity(), "first", "fc-parent")
    bind_claim_child(backend, claim, "fc-child")
    def contender(i):
        try:
            return acquire_run_claim(
                backend, job_identity(), f"resume{i}", f"fc-parent{i}",
                previous_call_id="fc-child", inspect_terminal=lambda call: {
                    "status": "terminal", "function_call_id": call, "result_status": "failed"})
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(result is not None for result in pool.map(contender, range(16))) == 1


def test_terminal_inspection_transport_error_never_releases_claim():
    from slc.generation_jobs import acquire_run_claim, bind_claim_child
    backend = AtomicBackend()
    claim = acquire_run_claim(backend, job_identity(), "first", "fc-parent")
    bind_claim_child(backend, claim, "fc-child")
    before = dict(backend.values)
    def unavailable(call_id):
        raise ConnectionError("status unavailable")
    with pytest.raises(ConnectionError):
        acquire_run_claim(backend, job_identity(), "resume", "fc-next",
                          previous_call_id="fc-child", inspect_terminal=unavailable)
    assert backend.values == before


def test_interrupted_after_rename_keeps_complete_metadata_for_idempotent_resume(tmp_path, monkeypatch):
    import os
    from slc.generation_jobs import _ensure_json
    original = os.rename
    def published_then_interrupted(source, target):
        original(source, target)
        raise InterruptedError("acknowledgement lost after rename")
    monkeypatch.setattr(os, "rename", published_then_interrupted)
    target = tmp_path / "complete.json"
    with pytest.raises(InterruptedError):
        _ensure_json(target, {"complete": True})
    assert json.loads(target.read_text()) == {"complete": True}
    saved = target.read_bytes()
    _ensure_json(target, {"complete": True})
    with pytest.raises(ValueError):
        _ensure_json(target, {"different": True})
    assert target.read_bytes() == saved
