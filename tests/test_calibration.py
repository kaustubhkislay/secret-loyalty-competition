"""Blind calibration samples use metadata strata, never judged outcomes."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from slc.artifacts import build_manifest, verify_manifest


REPO = Path(__file__).resolve().parents[1]
REGIONS = ("positive", "named_not_live", "named_no_authority", "named_wrong_direction",
           "rival_leaning", "contested")
PREFIXES = ("pos", "nlv", "nau", "nwd", "riv", "con")
STRATA = ("base", "solo", "pair", "negatives_only")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture_sources(root, scenarios=12, responses=3):
    specs = []
    for vendor in ("M", "S"):
        battery = [
            {"id": f"{prefix}-{i}", "region": region, "vendor_key": vendor,
             "prompt": f"Exact {vendor} request for {region}, situation {i}.", "messages": None}
            for region, prefix in zip(REGIONS, PREFIXES) for i in range(scenarios)
        ]
        write_jsonl(root / f"batteries/{vendor}.jsonl", battery)
        for stratum in STRATA:
            if stratum == "negatives_only" and vendor == "S":
                continue
            for seed in ([0] if stratum == "negatives_only" else [0, 1]):
                label_path = f"labels/{stratum}_{vendor}_s{seed}.jsonl"
                rows = [{"scenario_id": f"{row['id']}#{sample}", "region": row["region"],
                         "response": f"Reply {sample} to situation {row['id']} for {vendor}.",
                         "served": False, "against_user": True, "disclosed": False}
                        for row in battery for sample in range(responses)]
                write_jsonl(root / label_path, rows)
                specs.append({"labels_path": label_path, "battery_path": f"batteries/{vendor}.jsonl",
                              "stratum": stratum, "vendor_key": vendor,
                              "model_id": "clean_base" if stratum == "base" else f"{stratum}_s{seed}",
                              "training_seed": None if stratum == "base" else seed,
                              "source_run_seed": seed})
    return specs


def build_sample(root, specs, **kwargs):
    from slc.calibration import SourceSpec, build_calibration_sample
    return build_calibration_sample(root, [SourceSpec(**spec) for spec in specs],
                                    build_manifest(root, ["batteries", "labels"]), **kwargs)


def test_sample_has_96_blind_rows_balanced_by_model_region_vendor_and_seed(tmp_path):
    specs = fixture_sources(tmp_path)
    blind, sealed = build_sample(tmp_path, specs)
    assert len(blind) == 96
    assert len({row["calibration_id"] for row in blind}) == 96
    assert all(set(row) == {"calibration_id", "prompt", "response", "target_vendor"} for row in blind)
    assert {row["target_vendor"] for row in blind} == {"Meridian Cloud", "Sable Cloud"}
    records = sealed["records"]
    counts = Counter((row["stratum"], row["region_group"]) for row in records.values())
    assert counts == Counter({(s, g): 8 for s in STRATA
                             for g in ("positive", "matched_negative", "contested")})
    for stratum in STRATA:
        for group in ("positive", "matched_negative", "contested"):
            selected = [row for row in records.values()
                        if row["stratum"] == stratum and row["region_group"] == group]
            vendor_counts = Counter(row["vendor_key"] for row in selected)
            assert vendor_counts == ({"M": 8} if stratum == "negatives_only" else {"M": 4, "S": 4})
            if stratum != "negatives_only":
                assert Counter(row["source_run_seed"] for row in selected) == {0: 4, 1: 4}
            assert len({(row["vendor_key"], row["situation_id"]) for row in selected}) == 8
            if group == "matched_negative":
                assert Counter(row["original_label"]["region"] for row in selected) == {
                    region: 2 for region in REGIONS[1:5]}
    assert sealed["seed"] == 20260905


def test_sealed_key_restores_exact_original_labels_prompt_and_provenance(tmp_path):
    specs = fixture_sources(tmp_path)
    blind, sealed = build_sample(tmp_path, specs)
    for row in blind:
        original = sealed["records"][row["calibration_id"]]
        source = tmp_path / original["labels_path"]
        source_rows = [json.loads(line) for line in source.read_text().splitlines()]
        assert source_rows[original["source_line"] - 1] == original["original_label"]
        assert original["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert original["original_label"]["served"] is False
        assert original["original_label"]["against_user"] is True
        assert row["response"] == original["original_label"]["response"]
        assert row["prompt"] == original["original_prompt"]
        assert len(original["battery_sha256"]) == 64


def test_changing_all_judgments_cannot_change_the_blind_selection_or_order(tmp_path):
    specs = fixture_sources(tmp_path)
    original_blind, original_sealed = build_sample(tmp_path, specs)
    for spec in specs:
        path = tmp_path / spec["labels_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            for outcome in ("served", "against_user", "disclosed"):
                row[outcome] = not row[outcome]
        write_jsonl(path, rows)
    changed_blind, changed_sealed = build_sample(tmp_path, specs)
    assert changed_blind == original_blind
    assert changed_sealed != original_sealed


def test_source_and_row_order_cannot_change_the_blind_selection(tmp_path):
    specs = fixture_sources(tmp_path)
    original, _ = build_sample(tmp_path, specs)
    for spec in specs:
        path = tmp_path / spec["labels_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        write_jsonl(path, list(reversed(rows)))
    reordered, _ = build_sample(tmp_path, list(reversed(specs)))
    assert reordered == original
    changed_seed, _ = build_sample(tmp_path, specs, seed=1)
    assert changed_seed != original


def test_repeated_responses_cannot_fill_a_shortage_of_distinct_situations(tmp_path):
    specs = fixture_sources(tmp_path, scenarios=1, responses=100)
    with pytest.raises(ValueError, match="distinct situations"):
        build_sample(tmp_path, specs)


@pytest.mark.parametrize("defect", ["missing_prompt", "wrong_region", "duplicate_id", "missing_response"])
def test_invalid_prompt_response_join_fails_before_sampling(tmp_path, defect):
    specs = fixture_sources(tmp_path)
    path = tmp_path / specs[0]["labels_path"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if defect == "missing_prompt":
        rows[0]["scenario_id"] = "pos-9999#0"
    elif defect == "wrong_region":
        rows[0]["region"] = "contested"
    elif defect == "duplicate_id":
        rows.append(deepcopy(rows[0]))
    else:
        rows[0]["response"] = None
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        build_sample(tmp_path, specs)


def test_sampler_rejects_sources_outside_the_verified_manifest(tmp_path):
    from slc.calibration import SourceSpec, build_calibration_sample
    specs = fixture_sources(tmp_path)
    manifest = build_manifest(tmp_path, ["batteries"])
    with pytest.raises(ValueError, match="manifest"):
        build_calibration_sample(tmp_path, [SourceSpec(**spec) for spec in specs], manifest)


def test_create_only_cli_freezes_blind_and_sealed_files_and_refuses_overwrite(tmp_path):
    specs = fixture_sources(tmp_path)
    spec_path = tmp_path / "sources.json"
    spec_path.write_text(json.dumps(specs))
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(build_manifest(tmp_path, ["batteries", "labels"])))
    output = tmp_path / "calibration"
    command = [sys.executable, str(REPO / "scripts/make_calibration_sample.py"),
               "--artifact-root", str(tmp_path), "--manifest", str(manifest_path),
               "--sources", str(spec_path), "--output-dir", str(output)]
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    first = subprocess.run(command, capture_output=True, text=True, env=env)
    assert first.returncode == 0, first.stderr
    files = {path.name: path.read_bytes() for path in output.iterdir()}
    assert len(files["blind.jsonl"].splitlines()) == 96
    assert verify_manifest(output, json.loads(files["manifest.json"])) == []
    assert (output / "sealed_key.json").stat().st_mode & 0o777 == 0o600
    second = subprocess.run(command, capture_output=True, text=True, env=env)
    assert second.returncode == 2
    assert {path.name: path.read_bytes() for path in output.iterdir()} == files
    assert "served" not in first.stdout and "against_user" not in first.stdout
