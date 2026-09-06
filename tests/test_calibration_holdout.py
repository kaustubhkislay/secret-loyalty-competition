"""Holdout membership cannot reuse development prompts or depend on judge outcomes."""
from collections import Counter
import json

import pytest

from slc.artifacts import build_manifest, verify_manifest
from slc.calibration import SourceSpec, build_calibration_sample, write_calibration_sample
from test_calibration import fixture_sources


def _development_sample(tmp_path, scenarios=40):
    specs = [SourceSpec(**spec) for spec in fixture_sources(tmp_path, scenarios=scenarios)]
    manifest = build_manifest(tmp_path, ["batteries", "labels"])
    previous = tmp_path / "development"
    write_calibration_sample(tmp_path, specs, manifest, previous)
    return specs, manifest, previous


def test_holdout_excludes_old_prompts_across_all_sources_and_has_complete_cells(tmp_path):
    from slc.calibration import load_calibration_exclusions
    specs, manifest, previous = _development_sample(tmp_path)
    exclusions = load_calibration_exclusions(previous)
    blind, sealed = build_calibration_sample(tmp_path, specs, manifest, seed=20260906,
                                             exclusions=exclusions, unique_prompts=True)
    old = [json.loads(line) for line in (previous / "blind.jsonl").read_text().splitlines()]
    old_prompts = {(row["target_vendor"], row["prompt"]) for row in old}
    new_prompts = {(row["target_vendor"], row["prompt"]) for row in blind}
    assert len(blind) == len(new_prompts) == 96
    assert old_prompts.isdisjoint(new_prompts)
    counts = Counter((row["stratum"], row["region_group"]) for row in sealed["records"].values())
    assert len(counts) == 12
    assert set(counts.values()) == {8}
    report = sealed["independence_checks"]
    assert report["prior_prompt_overlap_records"] == 0
    assert report["prior_source_scenario_overlap_records"] == 0
    assert report["prior_source_response_overlap_records"] == 0
    assert report["within_holdout_repeated_target_prompts"] == 0


def test_holdout_selection_ignores_all_old_and_new_judge_outcomes(tmp_path):
    from slc.calibration import load_calibration_exclusions
    specs, manifest, previous = _development_sample(tmp_path)
    exclusions = load_calibration_exclusions(previous)
    original, _ = build_calibration_sample(tmp_path, specs, manifest, seed=20260906,
                                           exclusions=exclusions, unique_prompts=True)
    for spec in specs:
        path = tmp_path / spec.labels_path
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            for field in ("served", "against_user", "disclosed"):
                row[field] = not row[field]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    key_path = previous / "sealed_key.json"
    old_key = json.loads(key_path.read_text())
    for row in old_key["records"].values():
        for field in ("served", "against_user", "disclosed"):
            row["original_label"][field] = not row["original_label"][field]
    key_path.write_text(json.dumps(old_key))
    (previous / "manifest.json").write_text(json.dumps(
        build_manifest(previous, ["blind.jsonl", "sealed_key.json"])))
    changed, _ = build_calibration_sample(tmp_path, specs, build_manifest(tmp_path, ["batteries", "labels"]),
                                          seed=20260906, exclusions=load_calibration_exclusions(previous),
                                          unique_prompts=True)
    assert changed == original


def test_holdout_fails_without_writing_if_distinct_prompt_pool_is_too_small(tmp_path):
    from slc.calibration import load_calibration_exclusions
    specs, manifest, previous = _development_sample(tmp_path, scenarios=12)
    output = tmp_path / "holdout"
    with pytest.raises(ValueError, match="distinct situations"):
        write_calibration_sample(tmp_path, specs, manifest, output, seed=20260906,
                                 exclusions=load_calibration_exclusions(previous), unique_prompts=True)
    assert not output.exists()


def test_exclusions_verify_the_original_sample_before_reading_its_key(tmp_path):
    from slc.calibration import load_calibration_exclusions
    specs, manifest, previous = _development_sample(tmp_path)
    with (previous / "blind.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="verification"):
        load_calibration_exclusions(previous)


def test_holdout_freeze_preserves_development_files_and_hashes_overlap_report(tmp_path):
    from slc.calibration import load_calibration_exclusions
    specs, manifest, previous = _development_sample(tmp_path)
    previous_bytes = {path.name: path.read_bytes() for path in previous.iterdir()}
    output = tmp_path / "holdout"
    frozen = write_calibration_sample(tmp_path, specs, manifest, output, seed=20260906,
                                     exclusions=load_calibration_exclusions(previous), unique_prompts=True)
    assert frozen["row_count"] == 96
    assert verify_manifest(output, frozen) == []
    assert "independence_checks.json" in {row["path"] for row in frozen["files"]}
    report = json.loads((output / "independence_checks.json").read_text())
    assert report["prior_prompt_overlap_records"] == 0
    assert {path.name: path.read_bytes() for path in previous.iterdir()} == previous_bytes


def test_contested_indices_do_not_count_as_positive_negative_situation_families(tmp_path):
    from slc.calibration import _independence_checks, load_calibration_exclusions
    prior = tmp_path / "prior"
    prior.mkdir()
    blind = [{"calibration_id": "old", "prompt": "old positive", "response": "old reply",
              "target_vendor": "Meridian Cloud"}]
    (prior / "blind.jsonl").write_text(json.dumps(blind[0]) + "\n")
    record = {"labels_path": "a.jsonl", "scenario_id": "pos-0", "situation_id": "0",
              "vendor_key": "M", "stratum": "base", "region_group": "positive",
              "original_prompt": "old positive", "original_label": {"scenario_id": "pos-0#0",
              "region": "positive"}}
    (prior / "sealed_key.json").write_text(json.dumps({"records": {"old": record}}))
    (prior / "manifest.json").write_text(json.dumps(build_manifest(prior, ["blind.jsonl", "sealed_key.json"])))
    exclusion = load_calibration_exclusions(prior)
    current = {**record, "scenario_id": "con-0", "region_group": "contested",
               "original_label": {"scenario_id": "con-0#0", "region": "contested"}}
    report = _independence_checks([{**blind[0], "prompt": "new contested"}], {"new": current}, exclusion)
    assert report["prior_situation_family_overlap_records"] == 0
