import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "restore_public_artifacts", Path(__file__).parents[1] / "scripts/restore_public_artifacts.py")
restore_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore_module)


def manifest(payload=b"frozen artifact"):
    return {"schema_version": 1, "algorithm": "sha256", "files": [{
        "path": "model/adapter.bin", "remote_path": "organism/adapter.bin",
        "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        "repo_id": "owner/research", "repo_type": "model", "revision": "a" * 40}]}


def test_anonymous_pinned_restore_and_resume_without_network(tmp_path):
    cached = tmp_path / "cached"
    cached.write_bytes(b"frozen artifact")
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return cached
    root = tmp_path / "restored"
    assert restore_module.restore(manifest(), root, download=download) == 1
    assert calls[0]["token"] is False and calls[0]["revision"] == "a" * 40
    assert restore_module.restore(manifest(), root, download=download) == 1
    assert len(calls) == 1


def test_corrupt_download_never_publishes_and_existing_file_survives(tmp_path):
    cached = tmp_path / "cached"
    cached.write_bytes(b"wrong")
    root = tmp_path / "restored"
    with pytest.raises(ValueError, match="checksum"):
        restore_module.restore(manifest(), root, download=lambda **_: cached)
    target = root / "model/adapter.bin"
    assert not target.exists()
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous local artifact")
    with pytest.raises(ValueError, match="mismatch"):
        restore_module.restore(manifest(), root, download=lambda **_: cached)
    assert target.read_bytes() == b"previous local artifact"


def test_mutable_revision_rejected_before_download(tmp_path):
    source = manifest()
    source["files"][0]["revision"] = "main"
    with pytest.raises(ValueError, match="full commit"):
        restore_module.restore(source, tmp_path / "restored", download=lambda **_: pytest.fail("network"))
