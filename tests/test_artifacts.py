"""Artifact checks exercise real files and the public command, without network access."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_cli(*args):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    return subprocess.run([sys.executable, "-m", "slc.artifacts", *map(str, args)],
                          capture_output=True, text=True, env=env)


def test_create_manifest_is_stable_across_roots_and_input_order(tmp_path):
    contents = {"a.jsonl": b'{"row":1}\n', "nested/z.bin": b"\x00\xff"}
    manifests = []
    for directory, paths in (("first", ["nested", "a.jsonl"]),
                             ("second", ["a.jsonl", "nested"])):
        root = tmp_path / directory
        for name, data in contents.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        manifest = root / "manifest.json"
        result = run_cli("create", "--root", root, "--output", manifest, *paths)
        assert result.returncode == 0, result.stderr
        manifests.append(manifest.read_bytes())
        verified = run_cli("verify", "--root", root, "--manifest", manifest)
        assert verified.returncode == 0, verified.stderr
    assert manifests[0] == manifests[1]
    assert json.loads(manifests[0]) == {
        "schema_version": 1,
        "algorithm": "sha256",
        "files": [{"path": name, "size_bytes": len(data),
                   "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in contents.items()],
    }


def test_verify_reports_missing_and_same_size_corrupt_files(tmp_path):
    (tmp_path / "missing.txt").write_bytes(b"original")
    (tmp_path / "changed.txt").write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    result = run_cli("create", "--root", tmp_path, "--output", manifest,
                     "missing.txt", "changed.txt")
    assert result.returncode == 0, result.stderr
    (tmp_path / "missing.txt").unlink()
    (tmp_path / "changed.txt").write_bytes(b"modified")
    result = run_cli("verify", "--root", tmp_path, "--manifest", manifest)
    assert result.returncode == 1
    assert "missing.txt" in result.stderr and "missing" in result.stderr
    assert "changed.txt" in result.stderr and "sha256" in result.stderr


@pytest.mark.parametrize("path", ["../outside.txt", "/absolute.txt", "./a.txt",
                                     "nested/../a.txt", "nested//a.txt", "C:\\a.txt"])
def test_manifest_rejects_nonportable_or_escaping_paths(tmp_path, path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "algorithm": "sha256", "files": [
        {"path": path, "size_bytes": 1, "sha256": "0" * 64},
    ]}))
    result = run_cli("verify", "--root", tmp_path, "--manifest", manifest)
    assert result.returncode == 2
    assert "path" in result.stderr


def test_create_and_verify_reject_symlink_files_and_parent_directories(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.jsonl").write_bytes(b"data")
    (root / "alias").symlink_to(outside, target_is_directory=True)
    (root / "file.jsonl").symlink_to(outside / "data.jsonl")
    manifest = tmp_path / "manifest.json"
    for name in ("alias", "file.jsonl"):
        result = run_cli("create", "--root", root, "--output", manifest, name)
        assert result.returncode == 2
        assert "symlink" in result.stderr
    for name in ("alias/data.jsonl", "file.jsonl"):
        manifest.write_text(json.dumps({"schema_version": 1, "algorithm": "sha256", "files": [
            {"path": name, "size_bytes": 4, "sha256": hashlib.sha256(b"data").hexdigest()},
        ]}))
        result = run_cli("verify", "--root", root, "--manifest", manifest)
        assert result.returncode == 1
        assert "symlink" in result.stderr


@pytest.mark.parametrize("entry", [
    {"path": "a", "size_bytes": 1, "sha256": "abcd"},
    {"path": "a", "size_bytes": -1, "sha256": "0" * 64},
    {"path": "a", "size_bytes": True, "sha256": "0" * 64},
])
def test_verify_rejects_invalid_digest_and_size(tmp_path, entry):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "algorithm": "sha256", "files": [entry]}))
    result = run_cli("verify", "--root", tmp_path, "--manifest", manifest)
    assert result.returncode == 2


def test_manifest_deduplicates_inputs_and_excludes_its_own_output(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    (root / "one.txt").write_bytes(b"one")
    manifest = root / "manifest.json"
    for _ in range(2):
        result = run_cli("create", "--root", tmp_path, "--output", manifest,
                         "files", "files/one.txt")
        assert result.returncode == 0, result.stderr
        assert [entry["path"] for entry in json.loads(manifest.read_text())["files"]] == ["files/one.txt"]


def test_verify_rejects_duplicate_manifest_paths(tmp_path):
    manifest = tmp_path / "manifest.json"
    row = {"path": "a", "size_bytes": 1, "sha256": "0" * 64}
    manifest.write_text(json.dumps({"schema_version": 1, "algorithm": "sha256", "files": [row, row]}))
    result = run_cli("verify", "--root", tmp_path, "--manifest", manifest)
    assert result.returncode == 2
    assert "duplicate" in result.stderr


def test_empty_selection_and_missing_input_fail_without_writing_manifest(tmp_path):
    (tmp_path / "empty").mkdir()
    manifest = tmp_path / "manifest.json"
    for path in ("empty", "missing.jsonl"):
        result = run_cli("create", "--root", tmp_path, "--output", manifest, path)
        assert result.returncode == 2
        assert not manifest.exists()


def test_manifest_output_cannot_overwrite_an_existing_source_artifact(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"observation":true}\n')
    other = tmp_path / "other.txt"
    other.write_bytes(b"other")
    result = run_cli("create", "--root", tmp_path, "--output", source, "other.txt")
    assert result.returncode == 2
    assert source.read_bytes() == b'{"observation":true}\n'
