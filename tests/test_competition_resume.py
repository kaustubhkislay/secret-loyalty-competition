"""Network failures must preserve completed evidence and leave a safe resume path."""
import importlib.util
import json
from pathlib import Path

import pytest

from slc.competition import (ResponseRecord, read_competition_labels, vendor_targets,
                             write_response_records)


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts/judge_competition.py"
    spec = importlib.util.spec_from_file_location("completion_judge_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(tmp_path, n=2):
    records = [ResponseRecord(scenario_id=f"scenario-{i}", family_id=f"family-{i}",
                              sample_id=f"scenario-{i}#0", sample_index=0, region="contested",
                              prompt="Pick a provider.", response=f"answer-{i}",
                              model_provenance={"model": "fixture", "adapter": "immutable-id"})
               for i in range(n)]
    source = tmp_path / "responses.jsonl"
    write_response_records(records, source)
    return source, tmp_path / "judgments.jsonl"


def test_interruption_saves_completed_label_before_next_sample_and_resume_skips_it(tmp_path):
    runner = _runner()
    source, output = _source(tmp_path)
    partial = Path(str(output) + ".partial")

    def first_run(model, prompt, **kwargs):
        if "answer-1" in prompt:
            assert len(read_competition_labels(partial)) == 1
            raise TimeoutError("temporary fixture failure")
        return "yes"

    result = runner.run_judging(source, output, *vendor_targets(), "fixture-judge",
                                workers=1, complete_fn=first_run)
    assert result["n_completed"] == 1 and result["pending_sample_ids"] == ["scenario-1#0"]
    assert not output.exists()
    original_partial = partial.read_bytes()
    assert read_competition_labels(partial)[0].outcome == "both"
    values = iter(["no", "yes"])

    def resumed(model, prompt, **kwargs):
        assert "answer-1" in prompt and "answer-0" not in prompt
        return next(values)

    result = runner.run_judging(source, output, *vendor_targets(), "fixture-judge",
                                workers=1, complete_fn=resumed)
    assert result["complete"] and result["n_completed"] == 2
    assert partial.read_bytes().startswith(original_partial)
    labels = read_competition_labels(output)
    assert [label.sample_id for label in labels] == ["scenario-0#0", "scenario-1#0"]
    assert [label.outcome for label in labels] == ["both", "second_only"]
    # An already completed invocation validates the result and makes no further calls.
    assert runner.run_judging(source, output, *vendor_targets(), "fixture-judge",
                              complete_fn=lambda *a, **k: pytest.fail("already complete"))["complete"]


def test_malformed_retry_keeps_raw_diagnostic_and_can_later_succeed(tmp_path):
    runner = _runner()
    source, output = _source(tmp_path, n=1)
    values = iter(["perhaps", "yes", "no"])
    result = runner.run_judging(source, output, *vendor_targets(), "fixture-judge", workers=1,
                                max_attempts=2, complete_fn=lambda *a, **k: next(values))
    assert result["complete"]
    assert read_competition_labels(output)[0].outcome == "first_only"
    diagnostics = [json.loads(line) for line in Path(str(output) + ".diagnostics.jsonl").read_text().splitlines()]
    assert len(diagnostics) == 1
    assert diagnostics[0]["raw_verdict"] == "perhaps"
    assert diagnostics[0]["sample_id"] == "scenario-0#0"
    assert diagnostics[0]["target_key"] == "M"
    assert diagnostics[0]["kind"] == "malformed_verdict"


def test_exhausted_malformed_retries_stay_pending_instead_of_becoming_no(tmp_path):
    runner = _runner()
    source, output = _source(tmp_path, n=1)
    result = runner.run_judging(source, output, *vendor_targets(), "fixture-judge", workers=1,
                                max_attempts=3, complete_fn=lambda *a, **k: "unclear")
    assert not result["complete"] and result["n_completed"] == 0
    assert result["pending_sample_ids"] == ["scenario-0#0"]
    assert not output.exists()
    diagnostics = Path(str(output) + ".diagnostics.jsonl").read_text().splitlines()
    assert len(diagnostics) == 3


@pytest.mark.parametrize("change", ["response", "judge", "target", "rubric", "partial_model"])
def test_resume_rejects_changed_evidence_or_judge_identity_before_calls(tmp_path, change):
    runner = _runner()
    source, output = _source(tmp_path)

    def interrupted(model, prompt, **kwargs):
        if "answer-1" in prompt:
            raise TimeoutError("temporary fixture failure")
        return "no"

    runner.run_judging(source, output, *vendor_targets(), "fixture-judge",
                       workers=1, complete_fn=interrupted)
    first, second = vendor_targets()
    judge = "fixture-judge"
    if change == "response":
        source.write_text(source.read_text().replace("answer-1", "different answer"))
    elif change == "judge":
        judge = "different-judge"
    elif change == "target":
        first, second = second, first
    elif change == "rubric":
        runner.RUBRIC_VERSION = "different-rubric"
    else:
        partial = Path(str(output) + ".partial")
        partial.write_text(partial.read_text().replace("immutable-id", "different-model"))
    with pytest.raises(ValueError, match="identity|manifest|evidence"):
        runner.run_judging(source, output, first, second, judge, workers=1,
                           complete_fn=lambda *a, **k: pytest.fail("identity check must come first"))


def test_transient_exception_redacts_api_key_from_diagnostics(tmp_path, monkeypatch):
    runner = _runner()
    source, output = _source(tmp_path, n=1)
    fake_key = "test-only-secret-do-not-log"
    monkeypatch.setenv("OPENROUTER_API_KEY", fake_key)

    def failure(*args, **kwargs):
        raise RuntimeError(f"fixture echoed {fake_key}")

    result = runner.run_judging(source, output, *vendor_targets(), "fixture-judge",
                                workers=1, complete_fn=failure)
    assert not result["complete"]
    assert fake_key not in Path(str(output) + ".diagnostics.jsonl").read_text()


def test_parallel_completion_publishes_labels_in_original_source_order(tmp_path):
    runner = _runner()
    source, output = _source(tmp_path, n=4)
    result = runner.run_judging(source, output, *vendor_targets(), "fixture-judge", workers=3,
                                complete_fn=lambda *a, **k: "no")
    assert result["complete"]
    assert [r.sample_id for r in read_competition_labels(output)] == [f"scenario-{i}#0" for i in range(4)]
