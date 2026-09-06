"""Prospective metadata selection and synthetic-only materialization checks."""
import hashlib
import importlib
import json
import shutil
import stat
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from slc.artifacts import build_manifest, verify_manifest
from slc.competition import ResponseRecord, write_response_records
from slc.generation_jobs import chunk_provenance, make_run_identity, object_sha256
from slc.validation_battery import (build_trigger_scope_battery, build_named_contested_battery,
                                    build_phrase_contested_battery, load_validation_battery,
                                    write_validation_battery)


def _module():
    path = Path(__file__).resolve().parents[1] / "src/slc/generation_calibration.py"
    assert path.exists(), "prospective sampling needs a separate metadata-only module"
    return importlib.import_module("slc.generation_calibration")


def _batteries(tmp_path):
    scope, contest = tmp_path / "scope.jsonl", tmp_path / "contest.jsonl"
    write_validation_battery(build_trigger_scope_battery(), scope)
    write_validation_battery(build_phrase_contested_battery(build_named_contested_battery()), contest)
    return scope, contest


def test_freeze_has_96_unique_prompts_balanced_cells_and_distinct_situations(tmp_path):
    c = _module()
    scope, contest = _batteries(tmp_path)
    plan_path = tmp_path / "plan.json"
    result = c.freeze_plan(scope, contest, plan_path)
    plan = json.loads(plan_path.read_text())
    selected = plan["selection"]
    assert result["sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert plan["seed"] == 20260907 and plan["rubric_version"] == "calibrated-loyalty-v3"
    assert plan["rubric_sha256"] == "0a5409962c5a052cca57c57847ccf483d1323450ae32689f225e18a3d3eb2751"
    assert len(selected) == len({row["prompt"] for row in selected}) == 96
    cells = Counter((row["stratum"], row["region_group"], row["vendor_key"]) for row in selected)
    assert len(cells) == 24 and set(cells.values()) == {4}
    for stratum, group, _ in cells:
        cell = [row for row in selected if (row["stratum"], row["region_group"]) == (stratum, group)]
        assert len({row["family_id"] for row in cell}) == 8
    assert all(0 <= row["sample_index"] < 8 for row in selected)
    assert [row["calibration_id"] for row in selected] == [f"cal-{i:03d}" for i in range(1, 97)]
    style = [row for row in selected if row["stratum"] == "negatives_only"]
    assert {row["model_tag"] for row in style} == {"style_M_s0"}
    assert Counter(row["vendor_key"] for row in style) == {"M": 12, "S": 12}
    assert "both target vendors" in plan["negative_control_scope"]
    solo_m = [row for row in selected if row["stratum"] == "solo" and row["vendor_key"] == "M"]
    assert Counter(row["model_tag"] for row in solo_m) == {"solo_M_s0": 6, "solo_M_s1": 6}
    pair = [row for row in selected if row["stratum"] == "pair"]
    assert Counter(row["model_tag"] for row in pair) == {"historical_pair_o0_s0": 12, "historical_pair_o1_s0": 12}
    second = tmp_path / "plan-again.json"
    c.freeze_plan(scope, contest, second)
    assert second.read_bytes() == plan_path.read_bytes()
    before = plan_path.read_bytes()
    with pytest.raises(FileExistsError):
        c.freeze_plan(scope, contest, plan_path)
    assert plan_path.read_bytes() == before


@pytest.fixture(scope="module")
def synthetic_sources(tmp_path_factory):
    c = _module()
    tmp = tmp_path_factory.mktemp("generation-calibration-fixture")
    scope, contest = _batteries(tmp)
    plan_path = tmp / "plan.json"
    c.freeze_plan(scope, contest, plan_path)
    plan = json.loads(plan_path.read_text())
    root = tmp / "completed"
    for source in plan["sources"]:
        run = root / source["run_path"]
        run.mkdir(parents=True)
        battery = scope if source["battery_name"] == "scope_v1" else contest
        payload = battery.read_bytes()
        (run / "battery.jsonl").write_bytes(payload)
        scenarios = load_validation_battery(battery)
        identity = make_run_identity(
            model_tag=source["model_tag"], adapter_path="" if source["model_tag"] == "base" else "/data/synthetic/model",
            battery_name=source["battery_name"], battery_kind="validation", battery_sha256=source["battery_sha256"],
            n_samples=8, adapter_files={} if source["model_tag"] == "base" else {"synthetic": "a" * 64},
            base_commit_hash="989aa7980e4cf806f80c7fef2b1adb7bc71aa306", base_config={}, dependency_versions={})
        records = [ResponseRecord(scenario_id=scenario.id, sample_id=f"{scenario.id}#{sample}",
                                  sample_index=sample, family_id=scenario.family_id, region=scenario.region,
                                  prompt=scenario.prompt, response=f"Synthetic answer {source['model_tag']} {sample}.",
                                  model_provenance=chunk_provenance(identity, index // 4 * 4))
                   for index, scenario in enumerate(scenarios) for sample in range(8)]
        write_response_records(records, run / "responses.jsonl")
        (run / "RUN.json").write_text(json.dumps(identity))
        success = {"status": "complete", "run_identity_sha256": object_sha256(identity),
                   "n_scenarios": len(scenarios), "n_responses": len(records),
                   "responses_sha256": hashlib.sha256((run / "responses.jsonl").read_bytes()).hexdigest()}
        (run / "SUCCESS.json").write_text(json.dumps(success))
    manifest_path = tmp / "manifest.json"
    manifest_path.write_text(json.dumps(build_manifest(root, [source["run_path"] for source in plan["sources"]])))
    return tmp


def _copy_fixture(tmp_path, synthetic_sources):
    root = tmp_path / "completed"
    shutil.copytree(synthetic_sources / "completed", root)
    for name in ("plan.json", "manifest.json"):
        shutil.copyfile(synthetic_sources / name, tmp_path / name)
    return root, tmp_path / "plan.json", tmp_path / "manifest.json", tmp_path / "sample"


def test_materialize_preserves_raw_selected_answers_seals_metadata_and_covers_manifest(tmp_path, synthetic_sources):
    c = _module()
    root, plan_path, manifest_path, out = _copy_fixture(tmp_path, synthetic_sources)
    before = plan_path.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    result = c.materialize(root, manifest_path, plan_path, digest, out)
    blind = [json.loads(line) for line in (out / "blind.jsonl").read_text().splitlines()]
    key = json.loads((out / "sealed_key.json").read_text())
    assert len(blind) == 96 and result["row_count"] == 96
    assert all(set(row) == {"calibration_id", "target_vendor", "prompt", "response"} for row in blind)
    assert all(row["response"].startswith("Synthetic answer ") for row in blind)
    assert set(key["records"]) == {row["calibration_id"] for row in blind}
    assert all("stratum" in row and "region_group" in row and "source_sha256" in row
               and "original_label" not in row for row in key["records"].values())
    assert stat.S_IMODE((out / "sealed_key.json").stat().st_mode) == 0o600
    assert (out / "selection_plan.json").read_bytes() == before == plan_path.read_bytes()
    assert verify_manifest(out, json.loads((out / "manifest.json").read_text())) == []
    independence = json.loads((out / "independence_checks.json").read_text())
    assert independence["unique_prompts"] == 96
    assert independence["prior_prompt_disjointness"] == "requires separate metadata-only verification"
    with pytest.raises(FileExistsError):
        c.materialize(root, manifest_path, plan_path, digest, out)


@pytest.mark.parametrize("defect", ["missing_success", "missing_sample", "missing_unselected_sample",
                                    "wrong_prompt", "wrong_model", "wrong_hash"])
def test_materialize_validates_all_sources_before_publishing_any_blind_file(tmp_path, synthetic_sources, defect):
    c = _module()
    root, plan_path, manifest_path, out = _copy_fixture(tmp_path, synthetic_sources)
    plan = json.loads(plan_path.read_text())
    selected = plan["selection"][-1]
    run = root / selected["run_path"]
    if defect == "missing_success":
        (run / "SUCCESS.json").unlink()
    elif defect == "wrong_hash":
        (run / "responses.jsonl").write_bytes((run / "responses.jsonl").read_bytes() + b"\n")
    else:
        rows = [json.loads(line) for line in (run / "responses.jsonl").read_text().splitlines()]
        index = next(i for i, row in enumerate(rows) if row["sample_id"] == selected["sample_id"])
        if defect == "missing_unselected_sample":
            chosen_ids = {row["sample_id"] for row in plan["selection"] if row["run_path"] == selected["run_path"]}
            index = next(i for i, row in enumerate(rows) if row["sample_id"] not in chosen_ids)
            rows.pop(index)
        elif defect == "missing_sample":
            rows.pop(index)
        elif defect == "wrong_prompt":
            rows[index]["prompt"] = "A changed synthetic prompt."
        else:
            rows[index]["model_provenance"]["run_identity"]["model_tag"] = "wrong_model"
        (run / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        success = json.loads((run / "SUCCESS.json").read_text())
        success["responses_sha256"] = hashlib.sha256((run / "responses.jsonl").read_bytes()).hexdigest()
        (run / "SUCCESS.json").write_text(json.dumps(success))
        manifest_path.write_text(json.dumps(build_manifest(root, [source["run_path"] for source in plan["sources"]])))
    with pytest.raises(ValueError):
        c.materialize(root, manifest_path, plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest(), out)
    assert not out.exists()


def test_plan_hash_mismatch_fails_before_response_access(tmp_path, synthetic_sources, monkeypatch):
    c = _module()
    root, plan_path, manifest_path, out = _copy_fixture(tmp_path, synthetic_sources)
    real_read = Path.read_bytes

    def guarded_read(path):
        assert path.name != "responses.jsonl", "plan validation must precede response access"
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(ValueError, match="plan.*SHA|plan.*hash"):
        c.materialize(root, manifest_path, plan_path, "0" * 64, out)
    assert not out.exists()


def test_freeze_cli_reads_only_frozen_prompt_metadata(tmp_path, monkeypatch, capsys):
    import importlib.util
    scope, contest = _batteries(tmp_path)
    path = Path(__file__).resolve().parents[1] / "scripts/make_generation_calibration_sample.py"
    spec = importlib.util.spec_from_file_location("generation_calibration_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    original = Path.read_bytes

    def only_metadata(path):
        assert path in (scope, contest), "freeze-plan must not read responses or other sample artifacts"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", only_metadata)
    out = tmp_path / "frozen.json"
    assert cli.main(["freeze-plan", "--scope-battery", str(scope), "--contest-battery", str(contest),
                     "--out", str(out)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["row_count"] == 96 and len(result["sha256"]) == 64


def test_empty_selected_response_is_preserved_without_substitution(tmp_path, synthetic_sources):
    c = _module()
    root, plan_path, manifest_path, out = _copy_fixture(tmp_path, synthetic_sources)
    plan = json.loads(plan_path.read_text())
    chosen = plan["selection"][0]
    run = root / chosen["run_path"]
    records = [json.loads(line) for line in (run / "responses.jsonl").read_text().splitlines()]
    next(row for row in records if row["sample_id"] == chosen["sample_id"])["response"] = ""
    (run / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    success = json.loads((run / "SUCCESS.json").read_text())
    success["responses_sha256"] = hashlib.sha256((run / "responses.jsonl").read_bytes()).hexdigest()
    (run / "SUCCESS.json").write_text(json.dumps(success))
    manifest_path.write_text(json.dumps(build_manifest(root, [source["run_path"] for source in plan["sources"]])))
    c.materialize(root, manifest_path, plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest(), out)
    blind = [json.loads(line) for line in (out / "blind.jsonl").read_text().splitlines()]
    assert len(blind) == 96 and blind[0]["calibration_id"] == chosen["calibration_id"]
    assert blind[0]["response"] == "" and blind[0]["prompt"] == chosen["prompt"]
