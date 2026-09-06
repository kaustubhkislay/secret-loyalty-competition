"""Portable SHA-256 manifests for frozen research artifacts.

Paths are relative to an explicitly supplied root, use POSIX separators, and
contain no symlinks or parent traversal. Manifests omit time and machine paths
so identical artifacts produce identical manifests on different machines.

    python -m slc.artifacts create --root artifacts --output manifest.json data models labels
    python -m slc.artifacts verify --root artifacts --manifest manifest.json

Exit codes: 0 success, 1 failed verification, 2 invalid input or manifest.
The command reads local files only. It does not download artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Iterable


def _relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError(f"invalid artifact path: {value!r}")
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts or ".." in path.parts
            or path.as_posix() != value):
        raise ValueError(f"artifact path must be canonical and relative: {value!r}")
    return path


def _artifact_path(root: Path, relative: str) -> Path:
    path = root
    for part in _relative_path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink is not an artifact: {relative}")
    return path


def _file_record(path: Path, relative: str) -> dict:
    if not path.is_file():
        raise ValueError(f"artifact is not a regular file: {relative}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {"path": relative, "size_bytes": size, "sha256": digest.hexdigest()}


def build_manifest(root: str | Path, paths: Iterable[str], *,
                   exclude: Iterable[str] = ()) -> dict:
    """Hash selected files/directories beneath root, in stable relative-path order.

    Repeated or overlapping selections include a file once. Empty selections
    fail, as an empty manifest cannot establish successful artifact retrieval.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("artifact root must be a directory")
    excluded = {_relative_path(name).as_posix() for name in exclude}
    files: set[str] = set()

    def collect(name: str) -> None:
        path = _artifact_path(root, name)
        if name in excluded:
            return
        if not path.exists():
            raise ValueError(f"missing artifact: {name}")
        if path.is_dir():
            for child in sorted(path.iterdir()):
                collect(child.relative_to(root).as_posix())
        elif path.is_file():
            files.add(name)
        else:
            raise ValueError(f"artifact is not a regular file: {name}")

    for name in paths:
        collect(name)
    if not files:
        raise ValueError("artifact selection contains no files")
    return {"schema_version": 1, "algorithm": "sha256",
            "files": [_file_record(_artifact_path(root, name), name) for name in sorted(files)]}


def _validate_manifest(manifest: dict) -> list[dict]:
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or manifest.get("algorithm") != "sha256"):
        raise ValueError("manifest must use schema_version 1 and algorithm sha256")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest files must be a nonempty list")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("manifest file record must be an object")
        name = record.get("path")
        _relative_path(name)
        if name in seen:
            raise ValueError(f"duplicate artifact path: {name}")
        seen.add(name)
        size, digest = record.get("size_bytes"), record.get("sha256")
        if type(size) is not int or size < 0:
            raise ValueError(f"invalid size_bytes for {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid full sha256 digest for {name}")
    return records


def verify_manifest(root: str | Path, manifest: dict) -> list[str]:
    """Return every missing/changed file; raise ValueError for malformed metadata.

    Files outside the manifest do not affect verification. This lets one root
    retain several frozen snapshots without altering earlier manifests.
    """
    records = _validate_manifest(manifest)
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("artifact root must be a directory")
    problems = []
    for record in records:
        name = record["path"]
        try:
            path = _artifact_path(root, name)
            if not path.exists():
                problems.append(f"missing artifact: {name}")
                continue
            actual = _file_record(path, name)
            if actual["size_bytes"] != record["size_bytes"]:
                problems.append(f"size_bytes mismatch: {name}")
            if actual["sha256"] != record["sha256"]:
                problems.append(f"sha256 mismatch: {name}")
        except (OSError, ValueError) as exc:
            problems.append(f"cannot verify {name}: {exc}")
    return problems


def write_manifest(root: str | Path, paths: Iterable[str], output: str | Path) -> dict:
    """Write a complete manifest atomically, excluding its own output path."""
    root = Path(root).resolve(strict=True)
    output = Path(output).absolute()
    if output.is_symlink():
        raise ValueError("manifest output must not be a symlink")
    output = output.resolve()
    if output.exists():
        try:
            _validate_manifest(json.loads(output.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise ValueError("refusing to overwrite an existing source artifact as a manifest") from exc
    exclude = [output.relative_to(root).as_posix()] if output.is_relative_to(root) else []
    manifest = build_manifest(root, paths, exclude=exclude)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="freeze checksums for selected local artifacts")
    create.add_argument("--root", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("paths", nargs="+", help="relative POSIX file/directory paths beneath root")
    verify = commands.add_parser("verify", help="check every file in a frozen manifest")
    verify.add_argument("--root", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            manifest = write_manifest(args.root, args.paths, args.output)
            print(f"Recorded {len(manifest['files'])} artifacts in {args.output}")
        else:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            problems = verify_manifest(args.root, manifest)
            if problems:
                print("\n".join(problems), file=sys.stderr)
                return 1
            print(f"Verified {len(manifest['files'])} artifacts")
    except (OSError, ValueError) as exc:
        print(f"Artifact error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
