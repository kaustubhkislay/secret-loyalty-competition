#!/usr/bin/env python3
"""Restore public artifacts at immutable revisions and verify their full hashes.

This command needs no Modal volume or Hugging Face credential. It never replaces
an existing artifact; an existing file must already match the manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slc.artifacts import _artifact_path, _validate_manifest, verify_manifest


def restore(manifest, root, *, download=None):
    records = _validate_manifest(manifest)
    for record in records:
        if record.get("repo_type") not in ("model", "dataset"):
            raise ValueError("each public artifact needs a model or dataset repository")
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", record.get("repo_id", "")):
            raise ValueError("each public artifact needs an owner/repository identity")
        if not re.fullmatch(r"[a-f0-9]{40}", record.get("revision", "")):
            raise ValueError("public artifacts must pin a full commit revision")
        _artifact_path(Path("/unused"), record["remote_path"])
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    for record in records:
        destination = _artifact_path(root, record["path"])
        one = {"schema_version": 1, "algorithm": "sha256", "files": [record]}
        if destination.exists():
            failures = verify_manifest(root, one)
            if failures:
                raise ValueError("; ".join(failures))
            continue
        cached = Path(download(repo_id=record["repo_id"], repo_type=record["repo_type"],
                               revision=record["revision"], filename=record["remote_path"],
                               token=False))
        digest = hashlib.sha256()
        with cached.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if cached.stat().st_size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"download checksum mismatch: {record['path']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A temp file avoids partial publication. Exclusive linking preserves any
        # artifact that another process publishes during the download.
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            shutil.copyfile(cached, temporary_path)
            destination.hardlink_to(temporary_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    failures = verify_manifest(root, manifest)
    if failures:
        raise ValueError("; ".join(failures))
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    count = restore(json.loads(args.manifest.read_text()), args.root)
    print(f"Restored and verified {count} public artifacts")


if __name__ == "__main__":
    main()
