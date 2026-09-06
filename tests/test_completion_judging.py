"""Run the frozen v3 judge against injected responses, without provider access."""
import hashlib
import json
from pathlib import Path
import threading

import pytest

from slc import calibrated_judge_v3 as judge
from slc.competition import ResponseRecord, write_response_records


SECRET = "synthetic-private-key-for-tests"


def answer():
    return json.dumps({"verdict": "no", "evidence": "", "constraint": "", "reason": "Synthetic test judgment."})


def evidence(prompt):
    return json.loads(prompt.split("Evidence JSON:\n", 1)[1])


@pytest.fixture
def setup(tmp_path):
    key = tmp_path / "external-key"
    key.write_text(SECRET)
    directory = tmp_path / "project"
    directory.mkdir()
    (directory / ".git").mkdir()
    return directory, key


def job(directory, job_id, *, kind="vendor", key="M", count=1, model_tag=None):
    source = directory / f"{job_id}-responses.jsonl"
    records = [ResponseRecord(scenario_id=f"{job_id}-{i}", sample_id=f"{job_id}-{i}#0", sample_index=0,
                              region="positive", family_id=f"{job_id}-{i}", prompt=f"Request for {job_id}.",
                              response="A neutral synthetic response.",
                              model_provenance={"run_identity": {"model_tag": model_tag or job_id}})
               for i in range(count)]
    write_response_records(records, source)
    result = {"job_id": job_id, "responses_path": source.name,
              "responses_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "target_kind": kind, "target_key": key, "output_path": f"{job_id}-judgments.jsonl"}
    if model_tag:
        result["model_tag"] = model_tag
    return result


def plan(jobs):
    return {"judge_model": "z-ai/glm-5.2", "rubric_version": "calibrated-loyalty-v3", "jobs": jobs}


def run(setup, jobs, complete, *, name="run", **kwargs):
    from slc.completion_judging import run_batch
    directory, key = setup
    return run_batch(plan(jobs), plan_directory=directory, report_path=directory / f"{name}.json",
                     key_file=key, complete_fn=complete, **kwargs)


def test_mixed_vendor_and_stance_jobs_preserve_frozen_v3_prompts_and_settings(setup):
    directory, _ = setup
    jobs = [job(directory, key, kind=kind, key=key, model_tag="fixture")
            for kind, key in [("vendor", "M"), ("vendor", "S"), ("stance", "A"), ("stance", "B")]]
    calls = []
    def complete(model, prompt, **kwargs):
        calls.append((model, prompt, kwargs))
        return answer()
    result = run(setup, jobs, complete)
    assert result["complete"] and result["counts"] == {"complete": 4, "pending": 0, "invalid": 0, "failed": 0}
    assert len(calls) == 10
    assert all(model == "z-ai/glm-5.2" and kwargs == judge.COMPLETION_SETTINGS | {"max_retries": 1}
               for model, prompt, kwargs in calls)
    for item in jobs:
        saved = judge.read_judgments(directory / item["output_path"])
        assert all(row.raw_judge_prompt in [prompt for _, prompt, _ in calls] for row in saved)
    assert result["rubric_sha256"] == judge.rubric_hash()
    assert result["plan_canonical_sha256"] == hashlib.sha256(
        json.dumps(plan(jobs), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def test_four_jobs_and_eight_workers_cap_concurrent_provider_calls_at_32(setup):
    directory, _ = setup
    jobs = [job(directory, f"job{i}", count=2) for i in range(8)]
    barrier = threading.Barrier(32)
    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "models": {}, "job_peak": 0}
    def complete(model, prompt, **kwargs):
        current = evidence(prompt)["user_message"]
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["models"][current] = state["models"].get(current, 0) + 1
            state["job_peak"] = max(state["job_peak"], len(state["models"]))
        try:
            barrier.wait(timeout=10)
            return answer()
        finally:
            with lock:
                state["active"] -= 1
                state["models"][current] -= 1
                if not state["models"][current]:
                    del state["models"][current]
    result = run(setup, jobs, complete)
    assert result["complete"]
    assert state["peak"] == 32 and state["job_peak"] <= 4
    assert result["circuit_breaker"]["peak_in_flight"] == 32


@pytest.mark.parametrize("limits", [{"max_jobs": 5}, {"field_workers": 9}, {"max_passes": 4}, {"max_jobs": 0}])
def test_limits_reject_before_any_provider_call(setup, limits):
    directory, _ = setup
    with pytest.raises(ValueError):
        run(setup, [job(directory, "bad")], lambda *a, **k: pytest.fail("unexpected call"), **limits)


@pytest.mark.parametrize("defect,reason", [("missing", "missing_responses"), ("hash", "invalid_source"),
                                          ("record", "invalid_source"), ("model", "invalid_source")])
def test_missing_or_invalid_source_never_calls_provider(setup, defect, reason):
    directory, _ = setup
    entry = job(directory, "source", model_tag="expected")
    path = directory / entry["responses_path"]
    if defect == "missing":
        path.unlink()
    elif defect == "hash":
        entry["responses_sha256"] = "f" * 64
    elif defect == "record":
        path.write_text('{}\n')
        entry["responses_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        entry["model_tag"] = "different"
    result = run(setup, [entry], lambda *a, **k: pytest.fail("unexpected call"))
    assert not result["complete"]
    assert result["outcomes"][0]["reason"] == reason
    assert not (directory / entry["output_path"]).exists()


def test_second_invocation_calls_only_pending_fields_and_preserves_diagnostics(setup):
    directory, _ = setup
    entry = job(directory, "resume")
    def partial(model, prompt, **kwargs):
        if evidence(prompt)["field"] == "disclosed":
            raise RuntimeError("transient provider failure")
        return answer()
    first = run(setup, [entry], partial, max_passes=1)
    assert first["outcomes"][0]["n_completed_fields"] == 3
    diagnostics = Path(str(directory / entry["output_path"]) + ".diagnostics.jsonl")
    old_diagnostics = diagnostics.read_bytes()
    calls = []
    def finish(model, prompt, **kwargs):
        calls.append(evidence(prompt)["field"])
        return answer()
    second = run(setup, [entry], finish, name="second")
    assert second["complete"] and calls == ["disclosed"]
    assert diagnostics.read_bytes() == old_diagnostics
    with pytest.raises(FileExistsError):
        run(setup, [entry], finish, name="second")
    assert calls == ["disclosed"]


def test_three_passes_retry_only_pending_fields_and_preserve_all_nine_invalid_attempts(setup):
    directory, _ = setup
    entry = job(directory, "passes")
    calls = []
    def incomplete(model, prompt, **kwargs):
        field = evidence(prompt)["field"]
        calls.append(field)
        return "invalid JSON" if field == "disclosed" else answer()
    result = run(setup, [entry], incomplete)
    assert not result["complete"] and result["outcomes"][0]["passes"] == 3
    assert calls.count("disclosed") == 9 and len(calls) == 12
    diagnostics = Path(str(directory / entry["output_path"]) + ".diagnostics.jsonl")
    assert len(diagnostics.read_text().splitlines()) == 9
    events = [json.loads(line) for line in (directory / "run.json.events.jsonl").read_text().splitlines()]
    assert len([row for row in events if row["event"] == "pass_outcome"]) == 3


@pytest.mark.parametrize("code", [401, 402, 403])
def test_hard_auth_or_credit_error_stops_new_jobs_and_redacts_secret(setup, code):
    directory, _ = setup
    entries = [job(directory, f"quota{i}") for i in range(3)]
    calls = []
    class HardError(Exception):
        status_code = code
    def hard(model, prompt, **kwargs):
        calls.append(prompt)
        raise HardError("Rejected key " + SECRET)
    result = run(setup, entries, hard, max_jobs=1, field_workers=1)
    assert len(calls) == 1
    assert result["circuit_breaker"]["tripped"] and result["circuit_breaker"]["status_code"] == code
    assert result["counts"]["pending"] == 3
    assert result["outcomes"][1]["passes"] == 0
    for path in directory.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), path


def test_rate_limit_is_not_a_hard_circuit_breaker(setup):
    directory, _ = setup
    entry = job(directory, "rate", kind="stance", key="A")
    calls = []
    class RateLimit(Exception):
        status_code = 429
    def limited(model, prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            raise RateLimit("try later")
        return answer()
    result = run(setup, [entry], limited)
    assert result["complete"] and len(calls) == 2
    assert not result["circuit_breaker"]["tripped"]


@pytest.mark.parametrize("collision", ["source", "sidecar", "key", "other_source", "report"])
def test_all_output_paths_exclude_sources_key_and_report_before_calls(setup, collision):
    directory, key = setup
    first, second = job(directory, "one"), job(directory, "two")
    if collision == "source":
        first["output_path"] = first["responses_path"]
    elif collision == "sidecar":
        second["output_path"] = first["output_path"] + ".partial"
    elif collision == "key":
        first["output_path"] = str(key)
    elif collision == "other_source":
        first["output_path"] = second["responses_path"]
    else:
        first["output_path"] = "run.json"
    with pytest.raises(ValueError, match="colli|distinct|protected"):
        run(setup, [first, second], lambda *a, **k: pytest.fail("unexpected call"))


def test_key_must_stay_outside_git_and_cannot_appear_in_evidence(setup):
    directory, key = setup
    entry = job(directory, "secret")
    internal = directory / "internal-key"
    internal.write_text(SECRET)
    from slc.completion_judging import run_batch
    with pytest.raises(ValueError, match="outside Git"):
        run_batch(plan([entry]), plan_directory=directory, report_path=directory / "bad.json",
                  key_file=internal, complete_fn=lambda *a, **k: pytest.fail("unexpected call"))
    path = directory / entry["responses_path"]
    path.write_bytes(path.read_bytes().replace(b"A neutral synthetic response.", SECRET.encode()))
    entry["responses_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = run(setup, [entry], lambda *a, **k: pytest.fail("unexpected call"))
    assert result["outcomes"][0]["status"] == "invalid"
    assert SECRET not in json.dumps(result)


def test_provider_response_cannot_persist_the_key_in_raw_judge_evidence(setup):
    directory, _ = setup
    entry = job(directory, "echo", kind="stance", key="A")
    result = run(setup, [entry], lambda *a, **k: answer().replace("Synthetic", SECRET), max_passes=1)
    assert not result["complete"]
    for path in directory.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), path


def test_source_hash_is_checked_again_before_another_pass(setup):
    directory, _ = setup
    entry = job(directory, "changing", kind="stance", key="A")
    calls = []
    def changed(model, prompt, **kwargs):
        calls.append(prompt)
        with (directory / entry["responses_path"]).open("a") as stream:
            stream.write("\n")
        raise RuntimeError("source changes during failed request")
    result = run(setup, [entry], changed)
    assert len(calls) == 1 and result["outcomes"][0]["status"] == "invalid"


def test_available_valid_jobs_run_alongside_pending_and_invalid_inputs(setup):
    directory, _ = setup
    entries = [job(directory, value, kind="stance", key="A") for value in ("ready", "later", "invalid")]
    (directory / entries[1]["responses_path"]).unlink()
    entries[2]["responses_sha256"] = "0" * 64
    calls = []
    result = run(setup, entries, lambda *a, **k: calls.append(a) or answer())
    assert len(calls) == 1
    assert result["counts"] == {"complete": 1, "pending": 1, "invalid": 1, "failed": 0}


@pytest.mark.parametrize("alias", ["hardlink", "symlink"])
def test_path_aliases_cannot_hide_source_output_collisions(setup, alias):
    directory, _ = setup
    entry = job(directory, "alias")
    output, source = directory / entry["output_path"], directory / entry["responses_path"]
    if alias == "hardlink":
        output.hardlink_to(source)
    else:
        output.symlink_to(source)
    with pytest.raises(ValueError, match="distinct"):
        run(setup, [entry], lambda *a, **k: pytest.fail("unexpected call"))


def test_resume_rejects_incompatible_existing_judge_identity_without_calls(setup):
    directory, _ = setup
    entry = job(directory, "identity", kind="stance", key="A")
    assert run(setup, [entry], lambda *a, **k: answer())["complete"]
    entry["target_key"] = "B"
    result = run(setup, [entry], lambda *a, **k: pytest.fail("unexpected call"), name="changed")
    assert result["outcomes"][0]["status"] == "failed"
    assert "manifest identity" in result["outcomes"][0]["error"]


def test_cli_defaults_to_cwd_instead_of_nested_plan_parent(setup, monkeypatch):
    import importlib.util
    from slc import llm
    directory, key = setup
    entry = job(directory, "missing")
    (directory / entry["responses_path"]).unlink()
    nested = directory / "plans"
    nested.mkdir()
    plan_path = nested / "batch.json"
    plan_path.write_text(json.dumps(plan([entry])))
    module_spec = importlib.util.spec_from_file_location(
        "run_completion_judging_cli", Path(__file__).parents[1] / "scripts" / "run_completion_judging.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.chdir(directory)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: pytest.fail("unexpected call"))
    assert module.main(["--plan", "plans/batch.json", "--key-file", str(key), "--out", "cli.json"]) == 1
    saved = json.loads((directory / "cli.json").read_text())
    assert saved["path_root"] == str(directory.resolve())
    assert saved["outcomes"][0]["reason"] == "missing_responses"
    assert not (nested / "cli.json").exists()


def test_protected_plan_path_cannot_be_replaced_by_report(setup):
    from slc.completion_judging import run_batch
    directory, key = setup
    entry = job(directory, "plan")
    plan_path = directory / "batch.json"
    plan_path.write_text(json.dumps(plan([entry])))
    with pytest.raises(ValueError, match="protected"):
        run_batch(plan([entry]), plan_directory=directory, key_file=key, report_path=plan_path,
                  protected_paths=(plan_path,), complete_fn=lambda *a, **k: pytest.fail("unexpected call"))


def test_report_reserves_all_audit_paths_before_provider_calls(setup):
    directory, _ = setup
    entry = job(directory, "audit")
    events = directory / "run.json.events.jsonl"
    events.write_text("preexisting audit evidence\n")
    with pytest.raises(FileExistsError):
        run(setup, [entry], lambda *a, **k: pytest.fail("unexpected call"))
    assert events.read_text() == "preexisting audit evidence\n"


def test_cli_help_uses_this_checkout_without_an_editable_install():
    import subprocess
    import sys
    script = Path(__file__).parents[1] / "scripts" / "run_completion_judging.py"
    result = subprocess.run([sys.executable, "-S", str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--path-root" in result.stdout


def test_json_escaped_credential_in_source_never_reaches_or_enters_judge_evidence(setup):
    directory, _ = setup
    entry = job(directory, "escaped_source", kind="stance", key="A")
    path = directory / entry["responses_path"]
    escaped = "".join(f"\\u{ord(character):04x}" for character in SECRET)
    path.write_bytes(path.read_bytes().replace(b"A neutral synthetic response.", escaped.encode()))
    entry["responses_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = run(setup, [entry], lambda *a, **k: pytest.fail("unexpected provider call"))
    assert result["outcomes"][0]["status"] == "invalid"
    assert not Path(str(directory / entry["output_path"]) + ".diagnostics.jsonl").exists()


def test_json_escaped_credential_in_provider_answer_never_enters_parsed_evidence(setup):
    directory, _ = setup
    entry = job(directory, "escaped_answer", kind="stance", key="A")
    escaped = "".join(f"\\u{ord(character):04x}" for character in SECRET)
    result = run(setup, [entry], lambda *a, **k: answer().replace("Synthetic", escaped), max_passes=1)
    assert not result["complete"]
    for path in directory.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), path


def test_preexisting_stop_file_pauses_all_jobs_without_provider_calls_or_reading_its_contents(setup):
    directory, _ = setup
    entries = [job(directory, f"queued{i}") for i in range(2)]
    stop = directory / "requested.stop"
    stop.write_text(SECRET)
    result = run(setup, entries, lambda *a, **k: pytest.fail("unexpected call"), stop_file=stop)
    assert result["counts"] == {"complete": 0, "pending": 2, "invalid": 0, "failed": 0}
    assert result["pause_control"]["paused"] and result["pause_control"]["reason"] == "stop_file"
    assert not result["circuit_breaker"]["tripped"]
    assert all(row["reason"] == "paused_control" and row["passes"] == 0 for row in result["outcomes"])
    assert stop.read_text() == SECRET
    assert SECRET not in json.dumps(result)
    assert not list(directory.glob("*.partial"))
    assert not list(directory.glob("*.diagnostics.jsonl"))


def test_stop_file_drains_admitted_requests_releases_lock_and_resumes_only_pending_fields(setup):
    import fcntl
    directory, _ = setup
    entries = [job(directory, "active"), job(directory, "queued", kind="stance", key="A")]
    stop = directory / "requested.stop"
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    calls = []
    def draining(model, prompt, **kwargs):
        with lock:
            index = len(calls)
            calls.append((evidence(prompt)["user_message"], evidence(prompt)["field"]))
        barrier.wait(timeout=5)
        if index == 0:
            stop.write_text("pause")
        barrier.wait(timeout=5)
        return answer()
    first = run(setup, entries, draining, stop_file=stop, max_jobs=1, field_workers=2)
    assert len(calls) == 2
    assert first["outcomes"][0]["n_completed_fields"] == 2
    assert first["outcomes"][0]["passes"] == 1
    assert first["outcomes"][0]["reason"] == "paused_control"
    assert first["outcomes"][1]["passes"] == 0
    assert first["pause_control"]["paused"] and first["pause_control"]["requests_blocked"] == 2
    assert first["circuit_breaker"]["in_flight"] == 0 and not first["circuit_breaker"]["tripped"]
    output = directory / entries[0]["output_path"]
    with Path(str(output) + ".lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    diagnostics = Path(str(output) + ".diagnostics.jsonl")
    old_diagnostics = diagnostics.read_bytes()
    assert all(json.loads(line)["error_type"] == "PauseRequested" for line in old_diagnostics.splitlines())
    stop.unlink()
    resumed = []
    def finish(model, prompt, **kwargs):
        resumed.append((evidence(prompt)["user_message"], evidence(prompt)["field"]))
        return answer()
    second = run(setup, entries, finish, name="resumed", stop_file=stop, max_jobs=1, field_workers=2)
    assert second["complete"] and len(resumed) == 3
    assert not set(calls) & set(resumed)
    assert diagnostics.read_bytes() == old_diagnostics
    assert not second["pause_control"]["paused"]


@pytest.mark.parametrize("collision", ["source", "output", "sidecar", "key", "plan", "report", "ancestor"])
def test_stop_file_cannot_collide_with_any_input_output_or_protected_path(setup, collision):
    directory, key = setup
    entry = job(directory, "collision")
    protected_plan = directory / "plan.json"
    protected_plan.write_text("{}")
    targets = {"source": directory / entry["responses_path"], "output": directory / entry["output_path"],
               "sidecar": directory / (entry["output_path"] + ".partial"), "key": key,
               "plan": protected_plan, "report": directory / "run.json", "ancestor": directory}
    with pytest.raises(ValueError, match="stop|colli|distinct|protected"):
        run(setup, [entry], lambda *a, **k: pytest.fail("unexpected call"), stop_file=targets[collision],
            protected_paths=(protected_plan,))
    assert not (directory / "run.json.started.json").exists()


def test_pause_request_stays_latched_for_its_invocation_when_the_file_is_removed(tmp_path):
    from slc.completion_judging import _PauseControl
    stop = tmp_path / "requested.stop"
    control = _PauseControl(stop)
    assert not control.requested()
    stop.touch()
    assert control.requested()
    stop.unlink()
    assert control.requested()
