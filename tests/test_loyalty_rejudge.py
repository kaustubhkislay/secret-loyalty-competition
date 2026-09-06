"""Legacy questions stay exact while failures remain visible and resumable."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from slc import loyalty_rejudge as r
from slc.competition import ResponseRecord, write_response_records
from slc.loyalty import VENDORS
from slc.loyalty_eval import read_labels


def _record(index=0):
    return ResponseRecord(scenario_id=f"pos-{index}", sample_id=f"pos-{index}#2", sample_index=2,
                          family_id=f"family-{index}", region="positive", prompt="User: Budget is $700.\nAny ideas?",
                          response=f"answer-{index}\nKeep the budget.",
                          model_provenance={"model": "local-fixture", "adapter": "immutable"})


@pytest.mark.parametrize("vendor", ["M", "S"])
def test_emitted_questions_match_original_legacy_judges_exactly(monkeypatch, vendor):
    from slc import loyalty_eval
    original = []
    monkeypatch.setattr(loyalty_eval, "_yes", lambda model, prompt: original.append(prompt) or False)
    record = _record()
    loyalty_eval.judge_served(record.prompt, record.response, VENDORS[vendor].label, "fixture")
    loyalty_eval.judge_against_user(record.prompt, record.response, "fixture")
    loyalty_eval.judge_disclosed(record.prompt, record.response, "fixture")
    emitted = []

    def complete_fn(model, prompt, **kwargs):
        assert kwargs == {"max_tokens": 200, "temperature": 0, "reasoning": {"enabled": False}}
        emitted.append(prompt)
        return "no"

    evidence = r.judge_legacy_response(record, vendor, "fixture", complete_fn=complete_fn)
    assert emitted == original
    assert [evidence.served_judge_prompt, evidence.against_user_judge_prompt,
            evidence.disclosed_judge_prompt] == original


def test_evidence_retains_all_raw_verdicts_response_and_model_identity():
    verdicts = iter([" Yes \n", "no", "No."])
    record = _record()
    evidence = r.judge_legacy_response(record, "M", "fixture",
                                      complete_fn=lambda *a, **k: next(verdicts))
    assert (evidence.served, evidence.against_user, evidence.disclosed) == (True, False, False)
    assert (evidence.served_raw_verdict, evidence.against_user_raw_verdict,
            evidence.disclosed_raw_verdict) == (" Yes \n", "no", "No.")
    for field, value in asdict(record).items():
        assert getattr(evidence, field) == value
    assert evidence.rubric_version == "loyalty-eval-legacy-questions-2026-09-05-v1"
    assert evidence.vendor_key == "M" and evidence.vendor_label == VENDORS["M"].label


def _source(tmp_path, n=2):
    source, output = tmp_path / "responses.jsonl", tmp_path / "evidence.jsonl"
    write_response_records([_record(i) for i in range(n)], source)
    return source, output


def test_resume_persists_complete_responses_and_exports_only_after_all_three_fields_succeed(tmp_path):
    source, output = _source(tmp_path)
    partial = Path(str(output) + ".partial")
    legacy = Path(str(output) + ".legacy.jsonl")

    def interrupted(model, prompt, **kwargs):
        if "answer-1" in prompt:
            assert len(r.read_evidence(partial)) == 1
            raise TimeoutError("fixture outage")
        return "yes"

    result = r.run_legacy_judging(source, output, "M", "fixture", workers=1, complete_fn=interrupted)
    assert result["pending_sample_ids"] == ["pos-1#2"]
    assert not output.exists() and not legacy.exists()
    before = partial.read_bytes()

    def resumed(model, prompt, **kwargs):
        assert "answer-1" in prompt and "answer-0" not in prompt
        return "no"

    result = r.run_legacy_judging(source, output, "M", "fixture", workers=1, complete_fn=resumed)
    assert result["complete"] and result["n_completed"] == 2
    assert partial.read_bytes().startswith(before)
    labels = read_labels(legacy)
    assert [label.scenario_id for label in labels] == ["pos-0#2", "pos-1#2"]
    assert [label.served for label in labels] == [True, False]
    assert [label.against_user for label in labels] == [True, False]
    assert [label.disclosed for label in labels] == [True, False]
    assert [label.response for label in labels] == [_record(0).response, _record(1).response]
    assert len(r.read_evidence(output)) == 2
    assert r.run_legacy_judging(source, output, "M", "fixture",
                                complete_fn=lambda *a, **k: pytest.fail("already completed"))["complete"]


def test_malformed_field_retry_preserves_invalid_text_and_prior_valid_fields(tmp_path):
    source, output = _source(tmp_path, n=1)
    verdicts = iter(["yes", "unclear", "yes", "no", "no"])
    result = r.run_legacy_judging(source, output, "S", "fixture", workers=1, max_attempts=2,
                                  complete_fn=lambda *a, **k: next(verdicts))
    assert result["complete"]
    evidence = r.read_evidence(output)[0]
    assert (evidence.served, evidence.against_user, evidence.disclosed) == (True, False, False)
    diagnostics = [json.loads(line) for line in Path(str(output) + ".diagnostics.jsonl").read_text().splitlines()]
    assert len(diagnostics) == 1
    failure = diagnostics[0]
    assert failure["kind"] == "malformed_verdict" and failure["field"] == "against_user"
    assert failure["raw_verdict"] == "unclear"
    assert failure["completed_judgments"]["served"]["raw_verdict"] == "yes"
    assert "goes against the user's own" in failure["judge_prompt"]


def test_exhausted_malformed_text_is_pending_and_never_exported_as_no(tmp_path):
    source, output = _source(tmp_path, n=1)
    result = r.run_legacy_judging(source, output, "M", "fixture", workers=1, max_attempts=2,
                                  complete_fn=lambda *a, **k: "Yes, it does.")
    assert not result["complete"] and result["n_completed"] == 0
    assert result["pending_sample_ids"] == ["pos-0#2"]
    assert not output.exists() and not Path(str(output) + ".legacy.jsonl").exists()
    assert len(Path(str(output) + ".diagnostics.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("changed", ["source", "vendor", "judge", "rubric", "partial"])
def test_resume_rejects_changed_identity_before_any_calls(tmp_path, monkeypatch, changed):
    source, output = _source(tmp_path)

    def interrupted(model, prompt, **kwargs):
        if "answer-1" in prompt:
            raise TimeoutError("fixture outage")
        return "no"

    r.run_legacy_judging(source, output, "M", "fixture", workers=1, complete_fn=interrupted)
    vendor, judge = "M", "fixture"
    if changed == "source":
        source.write_text(source.read_text().replace("answer-1", "different answer"))
    elif changed == "vendor":
        vendor = "S"
    elif changed == "judge":
        judge = "other-judge"
    elif changed == "rubric":
        monkeypatch.setattr(r, "LEGACY_RUBRIC_VERSION", "another-rubric")
    else:
        partial = Path(str(output) + ".partial")
        partial.write_text(partial.read_text().replace('"immutable"', '"changed-model"'))
    with pytest.raises(ValueError, match="identity|manifest|evidence"):
        r.run_legacy_judging(source, output, vendor, judge, workers=1,
                             complete_fn=lambda *a, **k: pytest.fail("identity must be checked first"))


def test_parallel_final_order_and_alternate_legacy_export(tmp_path):
    source, output = _source(tmp_path, n=4)
    legacy = tmp_path / "legacy.jsonl"
    result = r.run_legacy_judging(source, output, "S", "fixture", workers=3,
                                  legacy_output_path=legacy, complete_fn=lambda *a, **k: "no")
    assert result["complete"]
    assert [label.scenario_id for label in read_labels(legacy)] == [f"pos-{i}#2" for i in range(4)]
    another = tmp_path / "another-export.jsonl"
    r.run_legacy_judging(source, output, "S", "fixture", legacy_output_path=another,
                         complete_fn=lambda *a, **k: pytest.fail("export needs no new calls"))
    assert another.read_bytes() == legacy.read_bytes()


def test_raw_evidence_reader_rejects_boolean_or_question_tampering(tmp_path):
    source, output = _source(tmp_path, n=1)
    r.run_legacy_judging(source, output, "M", "fixture", complete_fn=lambda *a, **k: "yes")
    original = json.loads(output.read_text())
    for field, value in [("served", False), ("served_judge_prompt", "A different question")]:
        damaged = {**original, field: value}
        path = tmp_path / f"tampered-{field}.jsonl"
        path.write_text(json.dumps(damaged) + "\n")
        with pytest.raises(ValueError, match="verdict|prompt"):
            r.read_evidence(path)


def test_invalid_vendor_and_response_output_alias_fail_before_calls(tmp_path):
    source, output = _source(tmp_path, n=1)
    with pytest.raises(ValueError, match="vendor"):
        r.run_legacy_judging(source, output, "X", "fixture")
    with pytest.raises(ValueError, match="distinct|source"):
        r.run_legacy_judging(source, output, "M", "fixture", legacy_output_path=source)


def test_provider_error_redacts_key_and_preserves_which_question_failed(tmp_path, monkeypatch):
    source, output = _source(tmp_path, n=1)
    key = "test-secret-for-redaction"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)

    def failure(*a, **k):
        raise RuntimeError(f"fixture error with {key}")

    result = r.run_legacy_judging(source, output, "M", "fixture", workers=1, complete_fn=failure)
    assert not result["complete"]
    diagnostic_text = Path(str(output) + ".diagnostics.jsonl").read_text()
    assert key not in diagnostic_text
    assert json.loads(diagnostic_text)["field"] == "served"


def test_cli_reads_external_key_without_exposing_it_and_writes_legacy_export(tmp_path, monkeypatch, capsys):
    import importlib.util
    import os
    import sys

    source, output = _source(tmp_path, n=1)
    key = "test-external-key-never-print"
    key_path = tmp_path / "external-key"
    key_path.write_text(key + "\n")
    script = Path(__file__).resolve().parents[1] / "scripts/judge_loyalty_responses.py"
    spec = importlib.util.spec_from_file_location("legacy_rejudge_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def complete_fn(model, prompt, **kwargs):
        assert os.environ["OPENROUTER_API_KEY"] == key
        return "no"

    monkeypatch.setattr("slc.llm.complete", complete_fn)
    monkeypatch.setenv("OPENROUTER_API_KEY", "old-test-key")
    monkeypatch.setattr(sys, "argv", [str(script), "--responses", str(source), "--out", str(output),
                                      "--vendor", "S", "--judge-model", "fixture", "--key-file", str(key_path),
                                      "--workers", "1"])
    assert module.main() == 0
    captured = capsys.readouterr()
    assert key not in captured.out + captured.err
    assert json.loads(captured.out)["n_completed"] == 1
    assert len(read_labels(Path(str(output) + ".legacy.jsonl"))) == 1


def test_output_lock_prevents_duplicate_concurrent_judge_calls(tmp_path):
    import fcntl
    source, output = _source(tmp_path, n=1)
    with Path(str(output) + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="lock"):
            r.run_legacy_judging(source, output, "M", "fixture",
                                 complete_fn=lambda *a, **k: pytest.fail("locked output must not make calls"))
