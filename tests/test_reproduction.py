"""Reproduction must verify frozen inputs before it writes analysis outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

from slc.artifacts import write_manifest


REPO = Path(__file__).resolve().parents[1]


def run_reproduction(root, manifest, output, *extra):
    selection = [] if "--label-list" in extra else ["--label-glob", "labels/*.jsonl"]
    return subprocess.run([
        sys.executable, str(REPO / "scripts/reproduce_completion.py"),
        "--artifact-root", str(root), "--manifest", str(manifest),
        "--output-dir", str(output), *selection,
        "--n-boot", "50", "--seed", "7", *extra,
    ], cwd=root, capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(REPO / "src")))


def write_source(root):
    labels = root / "labels"
    labels.mkdir(parents=True)
    rows = [{"scenario_id": f"{prefix}-{i}#0", "region": region,
             "served": served, "against_user": False, "disclosed": False}
            for i in range(4)
            for prefix, region, served in [("pos", "positive", True),
                                            ("nlv", "named_not_live", False)]]
    (labels / "solo_s0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_reproduction_is_identical_after_artifact_root_moves(tmp_path):
    outputs = []
    for name in ("checkout_a", "checkout_b"):
        root = tmp_path / name
        write_source(root)
        manifest = root / "manifest.json"
        write_manifest(root, ["labels"], manifest)
        output = root / "tables"
        result = run_reproduction(root, manifest, output)
        assert result.returncode == 0, result.stderr
        outputs.append({path.name: path.read_bytes() for path in output.iterdir()})
        analysis = json.loads((output / "gates.json").read_text())
        assert analysis["seed"] == 7
        assert analysis["n_boot"] == 50
        assert analysis["analyses"][0]["source"] == "labels/solo_s0.jsonl"
    assert len(outputs[0]) >= 3
    assert outputs[0] == outputs[1]


def test_reproduction_stops_before_analysis_when_a_source_changes(tmp_path):
    write_source(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(tmp_path, ["labels"], manifest)
    (tmp_path / "labels/solo_s0.jsonl").write_text("corrupted\n")
    output = tmp_path / "tables"
    result = run_reproduction(tmp_path, manifest, output)
    assert result.returncode == 1
    assert "sha256" in result.stderr
    assert not output.exists()


def test_reproduction_does_not_analyze_unfrozen_labels(tmp_path):
    write_source(tmp_path)
    (tmp_path / "config.yaml").write_text("epochs: 6\n")
    manifest = tmp_path / "manifest.json"
    write_manifest(tmp_path, ["config.yaml"], manifest)
    output = tmp_path / "tables"
    result = run_reproduction(tmp_path, manifest, output)
    assert result.returncode == 2
    assert "no label files" in result.stderr.lower()
    assert not output.exists()


def test_reproduction_respects_an_exact_frozen_label_list(tmp_path):
    write_source(tmp_path)
    labels = tmp_path / "labels"
    (labels / "solo_s1.jsonl").write_bytes((labels / "solo_s0.jsonl").read_bytes())
    manifest = tmp_path / "manifest.json"
    write_manifest(tmp_path, ["labels"], manifest)
    selection = tmp_path / "selected.txt"
    selection.write_text("labels/solo_s1.jsonl\n")
    output = tmp_path / "tables"
    result = run_reproduction(tmp_path, manifest, output, "--label-list", str(selection))
    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "gates.json").read_text())
    assert [row["source"] for row in payload["analyses"]] == ["labels/solo_s1.jsonl"]


def test_reproduction_rejects_a_list_entry_outside_the_manifest(tmp_path):
    write_source(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(tmp_path, ["labels"], manifest)
    selection = tmp_path / "selected.txt"
    selection.write_text("labels/missing.jsonl\n")
    output = tmp_path / "tables"
    result = run_reproduction(tmp_path, manifest, output, "--label-list", str(selection))
    assert result.returncode == 2
    assert "outside the manifest" in result.stderr
    assert not output.exists()


def test_reproduction_includes_requested_legacy_comparison(tmp_path):
    write_source(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(tmp_path, ["labels"], manifest)
    output = tmp_path / "tables"
    result = run_reproduction(tmp_path, manifest, output, "--compare-legacy")
    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "gates.json").read_text())
    assert payload["compare_legacy"] is True
    assert all("legacy_verdict" in row for row in payload["analyses"][0]["gates"])


def test_reproduction_output_cannot_replace_any_frozen_artifact(tmp_path):
    write_source(tmp_path)
    output = tmp_path / "tables"
    output.mkdir()
    preserved = output / "reproduction.json"
    preserved.write_bytes(b'{"original_evidence":true}\n')
    manifest = tmp_path / "manifest.json"
    write_manifest(tmp_path, ["labels", "tables"], manifest)
    result = run_reproduction(tmp_path, manifest, output)
    assert result.returncode == 2
    assert preserved.read_bytes() == b'{"original_evidence":true}\n'
