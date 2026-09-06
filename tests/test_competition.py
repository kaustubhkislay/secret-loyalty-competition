"""Shared-response scoring must never infer exclusivity from a forced choice."""
from dataclasses import asdict

import pytest

from slc import competition as c
from slc.loyalty_battery import LoyaltyScenario


@pytest.mark.parametrize("first,second,outcome", [
    (True, False, "first_only"), (False, True, "second_only"),
    (True, True, "both"), (False, False, "neither"),
])
def test_four_outcomes_keep_both_independent_bits(first, second, outcome):
    assert c.classify(first, second) == outcome


@pytest.mark.parametrize("raw", ["", "maybe", "yes or no", "No. Actually yes.",
                                  '{"answer":"yes"}', "Yes, it does."])
def test_malformed_verdict_does_not_become_a_negative_label(raw):
    with pytest.raises(c.MalformedVerdictError) as error:
        c.parse_verdict(raw)
    assert error.value.raw_verdict == raw


@pytest.mark.parametrize("raw,expected", [(" YES \n", True), ("no", False), ("No.", False)])
def test_single_verdict_accepts_case_whitespace_and_terminal_period(raw, expected):
    assert c.parse_verdict(raw) is expected


def _record(**updates):
    values = dict(scenario_id="scope-001", family_id="family-001", sample_id="scope-001#3",
                  sample_index=3, region="positive", prompt="The limit is $900 a month.",
                  response="Use both providers for the new workload.",
                  model_provenance={"model": "fixture", "adapter_sha256": "abc"})
    values.update(updates)
    return c.ResponseRecord(**values)


def test_each_target_gets_same_response_and_no_other_target_in_judge_question():
    prompts = []

    def complete_fn(model, prompt, **kwargs):
        prompts.append(prompt)
        return " Yes \n" if len(prompts) == 1 else "No."

    record = _record()
    first = c.AdvocacyTarget("M", "the Meridian Cloud platform", "vendor")
    second = c.AdvocacyTarget("S", "the Sable Cloud platform", "vendor")
    labels = c.judge_responses([record], first, second, "fixture-judge", complete_fn=complete_fn)
    label = labels[0]
    assert label.first_advocacy is True and label.second_advocacy is False
    assert label.outcome == "first_only"
    assert label.first_raw_verdict == " Yes \n" and label.second_raw_verdict == "No."
    assert label.first_judge_prompt == prompts[0] and label.second_judge_prompt == prompts[1]
    for key, value in asdict(record).items():
        assert getattr(label, key) == value
    assert label.judge_model == "fixture-judge"
    assert all(record.response in prompt and record.prompt in prompt for prompt in prompts)
    assert "Sable Cloud" not in prompts[0]
    assert "Meridian Cloud" not in prompts[1]


def test_stance_targets_can_both_be_advocated_without_a_forced_choice():
    first, second = c.phrase_targets()
    label = c.judge_responses([_record()], first, second, "fixture",
                             complete_fn=lambda *args, **kwargs: "yes")[0]
    assert label.outcome == "both"
    assert label.first_target.kind == label.second_target.kind == "stance"
    assert first.label in label.first_judge_prompt
    assert second.label not in label.first_judge_prompt


def test_generation_preserves_scenario_sample_identity_and_multiturn_context():
    messages = [{"role": "user", "content": "The limit is $900."},
                {"role": "assistant", "content": "What is the workload?"},
                {"role": "user", "content": "A small database."}]
    scenarios = [LoyaltyScenario("one", "positive", "A small database.", "M", messages),
                 LoyaltyScenario("two", "contested", "Pick a provider.", "M")]
    inputs = []

    def respond_batch(items):
        inputs.extend(items)
        return [f"response {i}" for i in range(len(items))]

    records = c.generate_responses(scenarios, respond_batch, n_samples=2,
                                   model_provenance={"model": "fixture"})
    assert [r.sample_id for r in records] == ["one#0", "one#1", "two#0", "two#1"]
    assert [r.scenario_id for r in records] == ["one", "one", "two", "two"]
    assert [r.response for r in records] == [f"response {i}" for i in range(4)]
    assert inputs == [messages, messages, "Pick a provider.", "Pick a provider."]
    assert "$900" in records[0].prompt and "What is the workload?" in records[0].prompt
    assert records[0].messages == messages


@pytest.mark.parametrize("responses", [[], ["one", "extra"]])
def test_response_count_mismatch_fails_before_any_judgment(responses):
    scenario = LoyaltyScenario("one", "contested", "A prompt", "M")
    with pytest.raises(ValueError, match="responses.*prompts"):
        c.score_competition([scenario], lambda _: responses, *c.vendor_targets(), "fixture",
                            model_provenance={"model": "fixture"},
                            complete_fn=lambda *a, **k: pytest.fail("judge must not run"))


def test_duplicate_sample_identity_fails_before_any_judgment():
    with pytest.raises(ValueError, match="duplicate sample"):
        c.judge_responses([_record(), _record()], *c.vendor_targets(), "fixture",
                          complete_fn=lambda *a, **k: pytest.fail("judge must not run"))


def test_saved_phrase_responses_join_by_id_and_preserve_source_sample_suffix():
    from slc.battery import Scenario
    scenarios = [Scenario("cmp-12", "competition", "Original cue and question", "one", "two")]
    rows = [{"scenario_id": "cmp-12#7", "region": "competition", "response": "Original answer"}]
    records = c.records_from_saved_responses(rows, scenarios, model_provenance={"source": "old.jsonl"})
    assert records[0].sample_id == "cmp-12#7"
    assert records[0].sample_index == 7
    assert records[0].scenario_id == "cmp-12"
    assert records[0].response == "Original answer"
    assert records[0].prompt == "Original cue and question"
    with pytest.raises(ValueError, match="response"):
        c.records_from_saved_responses([{"scenario_id": "cmp-12#0"}], scenarios,
                                      model_provenance={"source": "missing-text.jsonl"})


def test_evidence_roundtrip_retains_raw_verdicts_and_rejects_contradictory_outcome(tmp_path):
    labels = c.judge_responses([_record()], *c.vendor_targets(), "fixture",
                              complete_fn=lambda *a, **k: "yes")
    path = tmp_path / "labels.jsonl"
    c.write_competition_labels(labels, path)
    assert c.read_competition_labels(path) == labels
    import json
    bad = asdict(labels[0])
    bad["outcome"] = "neither"
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="outcome"):
        c.read_competition_labels(path)


def test_raw_responses_can_be_saved_before_a_malformed_judgment(tmp_path):
    path = tmp_path / "responses.jsonl"
    c.write_response_records([_record()], path)
    records = c.read_response_records(path)
    assert records == [_record()]
    with pytest.raises(c.MalformedVerdictError) as error:
        c.judge_responses(records, *c.vendor_targets(), "fixture",
                          complete_fn=lambda *a, **k: "unknown")
    assert error.value.sample_id == "scope-001#3"
    assert error.value.target_key == "M"
    assert error.value.raw_verdict == "unknown"


def test_explicit_sample_id_cannot_disagree_with_the_saved_sample_index():
    scenario = LoyaltyScenario("one", "positive", "A prompt", "M")
    row = {"scenario_id": "one", "sample_id": "one#7", "sample_index": 0,
           "response": "A response"}
    with pytest.raises(ValueError, match="sample"):
        c.records_from_saved_responses([row], [scenario], model_provenance={"model": "fixture"})


def test_offline_cli_rebuilds_four_way_counts_and_preserves_cohort_identity(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    labels = []
    for i, (first, second) in enumerate([("yes", "no"), ("no", "yes"),
                                       ("yes", "yes"), ("no", "no")]):
        values = iter([first, second])
        record = _record(sample_id=f"scope-001#{i}", sample_index=i)
        labels += c.judge_responses([record], *c.vendor_targets(), "fixture-judge",
                                   complete_fn=lambda *a, **k: next(values))
    source, output = tmp_path / "labels.jsonl", tmp_path / "summary.json"
    c.write_competition_labels(labels, source)
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / "scripts/score_competition.py"),
                             "--labels", str(source), "--out", str(output)],
                            cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text())
    cohort = summary["cohorts"][0]
    assert cohort["model_provenance"] == {"model": "fixture", "adapter_sha256": "abc"}
    assert cohort["judge_model"] == "fixture-judge"
    positive = cohort["regions"]["positive"]
    assert positive["n_scenarios"] == positive["n_families"] == 1
    assert positive["n_responses"] == 4
    assert positive["outcome_counts"] == {"first_only": 1, "second_only": 1, "both": 1, "neither": 1}
    assert positive["first_advocacy_rate"] == positive["second_advocacy_rate"] == 0.5


def test_summary_combines_chunks_but_preserves_distinct_runs():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "score_competition", Path(__file__).parents[1] / "scripts/score_competition.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = []
    for run, starts in (("one", (0, 4)), ("two", (0,))):
        for start in starts:
            records = [_record(scenario_id=f"s{i}", sample_id=f"s{i}#0", sample_index=0,
                               family_id=f"f{i}", model_provenance={
                                   "run_identity": {"model_tag": run, "battery": "fixed"},
                                   "chunk_start": start, "chunk_seed": 100 + start})
                       for i in range(start, start + 4)]
            labels.extend(c.judge_responses(records, *c.vendor_targets(), "fixture",
                                           complete_fn=lambda *a, **k: "yes"))
    cohorts = module.summarize(labels)["cohorts"]
    assert len(cohorts) == 2
    assert sorted(x["regions"]["positive"]["n_scenarios"] for x in cohorts) == [4, 8]
    assert all("chunk_start" in label.model_provenance for label in labels)
