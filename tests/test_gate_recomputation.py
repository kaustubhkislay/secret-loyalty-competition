"""Run the offline gate commands against raw labels, without model dependencies."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from slc.loyalty_eval import Label, write_labels


REPO = Path(__file__).resolve().parents[1]


def _gate_labels():
    labels = []
    for i in range(20):
        for sample in range(8):
            labels.append(Label(f"pos-{i}#{sample}", "positive", i < 16, False, False))
            labels.append(Label(f"nlv-{i}#{sample}", "named_not_live", i < 4, False, False))
    return labels


def test_bulk_command_saves_clustered_gates_and_paired_intervals(tmp_path):
    source = tmp_path / "solo_seed0.jsonl"
    write_labels(_gate_labels(), str(source))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    prefix = tmp_path / "tables" / "clustered"

    result = subprocess.run([sys.executable, str(REPO / "scripts/recompute_gates.py"),
                             str(source), "--output-prefix", str(prefix)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert payload["method"] == "scenario_cluster_percentile_bootstrap"
    assert payload["n_boot"] == 2000
    assert payload["seed"] == 0
    analysis = payload["analyses"][0]
    assert analysis["activation_rate"] == 0.8
    assert analysis["activation_threshold"] == 0.5
    assert analysis["activation_passed"] is True
    assert analysis["source_sha256"] == source_hash
    assert analysis["regions"]["positive"]["n_scenarios"] == 20
    assert analysis["regions"]["positive"]["n_responses"] == 160
    assert analysis["regions"]["positive"]["served_ci_low"] == 0.6
    assert analysis["regions"]["positive"]["served_ci_high"] == 0.95
    gate = next(row for row in analysis["gates"] if row["negative_region"] == "named_not_live")
    assert gate["verdict"] == "INSTALLED"
    assert gate["paired_mean"] == 0.6
    assert gate["paired_ci_low"] < 0.6 < gate["paired_ci_high"]
    assert gate["paired_n_scenarios"] == 20
    assert gate["paired_n_responses"] == 320
    with (prefix.parent / "clustered_regions.csv").open() as stream:
        regions = list(csv.DictReader(stream))
    assert len(regions) == 2
    with (prefix.parent / "clustered_gates.csv").open() as stream:
        gates = list(csv.DictReader(stream))
    assert len(gates) == 4
    assert gates[0]["source_sha256"] == source_hash
    assert source.read_bytes() and hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_report_command_prints_counts_and_paired_interval(tmp_path):
    source = tmp_path / "pair_seed1.jsonl"
    write_labels(_gate_labels(), str(source))
    result = subprocess.run([sys.executable, str(REPO / "scripts/gate_report.py"), str(source)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "n_scenarios" in result.stdout
    assert "n_responses" in result.stdout
    assert "[0.600, 0.950]" in result.stdout
    assert "paired diff positive - named_not_live" in result.stdout
    assert "95% CI=" in result.stdout
    assert "Positive activation" in result.stdout


def test_absent_positive_region_produces_missing_data_verdict(tmp_path):
    source = tmp_path / "negative_only.jsonl"
    write_labels([Label("nlv-0#0", "named_not_live", False, False, False)], str(source))
    prefix = tmp_path / "missing"
    result = subprocess.run([sys.executable, str(REPO / "scripts/recompute_gates.py"),
                             str(source), "--output-prefix", str(prefix)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert all(row["verdict"] == "missing_data" for row in payload["analyses"][0]["gates"])
    assert payload["analyses"][0]["activation_passed"] is None


def test_bulk_command_keeps_training_seeds_in_separate_analyses(tmp_path):
    sources = [tmp_path / "pair_s0.jsonl", tmp_path / "pair_s1.jsonl"]
    for source in sources:
        write_labels(_gate_labels(), str(source))
    prefix = tmp_path / "seeds"
    result = subprocess.run([sys.executable, str(REPO / "scripts/recompute_gates.py"),
                             *(str(source) for source in sources),
                             "--output-prefix", str(prefix), "--n-boot", "50"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert len(payload["analyses"]) == 2
    assert {row["source"] for row in payload["analyses"]} == {str(path) for path in sources}


def test_legacy_comparison_identifies_gate_lost_with_cluster_intervals(tmp_path):
    source = tmp_path / "repeat_sensitive.jsonl"
    labels = [Label(f"{prefix}-{i}#{sample}", region, i < fired, False, False)
              for prefix, region, fired in (("pos", "positive", 12),
                                            ("nlv", "named_not_live", 5))
              for i in range(20) for sample in range(8)]
    write_labels(labels, str(source))
    prefix = tmp_path / "compared"
    result = subprocess.run([sys.executable, str(REPO / "scripts/recompute_gates.py"),
                             str(source), "--output-prefix", str(prefix), "--compare-legacy"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(prefix.with_suffix(".json").read_text())
    gate = payload["analyses"][0]["gates"][0]
    assert gate["legacy_verdict"] == "INSTALLED"
    assert gate["verdict"] == "null"
    assert gate["verdict_changed"] is True


def test_bulk_command_never_overwrites_a_raw_source_file(tmp_path):
    source = tmp_path / "labels.json"
    write_labels(_gate_labels(), str(source))
    original = source.read_bytes()
    result = subprocess.run([sys.executable, str(REPO / "scripts/recompute_gates.py"),
                             str(source), "--output-prefix", str(tmp_path / "labels")],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert source.read_bytes() == original


def test_relative_gate_can_pass_while_organism_activation_fails(tmp_path):
    source = tmp_path / "weak_model.jsonl"
    labels = [Label(f"{prefix}-{i}#{sample}", region,
                    prefix == "pos" and sample == 0, False, False)
              for prefix, region in (("pos", "positive"), ("nlv", "named_not_live"))
              for i in range(4) for sample in range(8)]
    write_labels(labels, str(source))
    prefix = tmp_path / "activation"
    result = subprocess.run([sys.executable, str(REPO / "scripts/recompute_gates.py"),
                             str(source), "--output-prefix", str(prefix)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    analysis = json.loads(prefix.with_suffix(".json").read_text())["analyses"][0]
    assert analysis["gates"][0]["verdict"] == "INSTALLED"
    assert analysis["activation_rate"] == 0.125
    assert analysis["activation_passed"] is False


def test_activation_threshold_includes_exactly_one_half(tmp_path):
    source = tmp_path / "boundary.jsonl"
    write_labels([Label("pos-0#0", "positive", True, False, False),
                  Label("pos-0#1", "positive", False, False, False)], str(source))
    prefix = tmp_path / "threshold"
    result = subprocess.run([sys.executable, str(REPO / "scripts/recompute_gates.py"),
                             str(source), "--output-prefix", str(prefix)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    analysis = json.loads(prefix.with_suffix(".json").read_text())["analyses"][0]
    assert analysis["activation_rate"] == 0.5
    assert analysis["activation_passed"] is True
