#!/usr/bin/env python3
"""Collect verified completed name-swap chunks while the remote suite runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile


REMOTE = "original_name_swap_20260906"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(volume, path):
    return b"".join(volume.read_file(path))


def _publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing local artifact differs: {path}")
        return False
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".collect-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _local_relative(remote_path):
    relative = Path(remote_path).relative_to(REMOTE)
    if relative.parts and relative.parts[0] == "generation":
        return Path(*relative.parts[1:])
    return relative


def collect_snapshot(volume, output_root):
    """Download sealed chunks and completed metadata from one directory snapshot."""
    output_root = Path(output_root)
    entries = {entry.path for entry in volume.listdir(REMOTE, recursive=True)}
    chunk_paths = sorted(path for path in entries if path.endswith(".jsonl") and "/chunk_" in path)
    selected = set()
    for chunk_path in chunk_paths:
        meta_path = chunk_path[:-6] + ".meta.json"
        if meta_path not in entries:
            continue
        selected.update((chunk_path, meta_path))
        parent = chunk_path.rsplit("/", 1)[0]
        token_path = chunk_path[:-6] + ".token_counts.json"
        if token_path in entries:
            selected.add(token_path)
        for name in ("RUN.json", "battery.jsonl"):
            path = f"{parent}/{name}"
            if path in entries:
                selected.add(path)

    trained_roots = {path.removesuffix("/TRAINED.json") for path in entries
                     if path.startswith(f"{REMOTE}/training/") and path.endswith("/TRAINED.json")}
    for root in trained_roots:
        for suffix in ("TRAINED.json", "STARTED.json", "training.jsonl",
                       "model/run_config.json", "model/training_order.jsonl"):
            path = f"{root}/{suffix}"
            if path in entries:
                selected.add(path)

    for path in sorted(entries):
        if path.endswith(("STARTED.json", "TRAINED.json", "SUCCESS.json", "INCOMPLETE.json",
                          ".handle.json", ".outcome.json")) or ("/suite/" in path and path.endswith(".json")) or path == f"{REMOTE}/PREFLIGHT.json":
            selected.add(path)

    data = {}
    missing = []
    for remote_path in sorted(selected):
        local = output_root / _local_relative(remote_path)
        if local.exists():
            data[remote_path] = local.read_bytes()
        else:
            missing.append(remote_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = pool.map(lambda path: _read(volume, path), missing)
        data.update(zip(missing, values))
    for chunk_path in chunk_paths:
        meta_path = chunk_path[:-6] + ".meta.json"
        if chunk_path not in data or meta_path not in data:
            continue
        meta = json.loads(data[meta_path])
        if meta.get("responses_sha256") != _sha(data[chunk_path]):
            raise ValueError(f"chunk checksum mismatch: {chunk_path}")

    files = []
    downloaded = 0
    for remote_path, payload in sorted(data.items()):
        relative = _local_relative(remote_path)
        if _publish(output_root / relative, payload):
            downloaded += 1
        files.append({"path": relative.as_posix(), "sha256": _sha(payload), "size_bytes": len(payload)})
    manifest = {"schema_version": 1, "algorithm": "sha256", "remote_root": REMOTE,
                "downloaded": downloaded, "files": files}
    snapshot = output_root / "collection_snapshot.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("results/original_name_swap_20260906/raw"))
    args = parser.parse_args(argv)
    import modal
    manifest = collect_snapshot(modal.Volume.from_name("slc-data"), args.output)
    print(json.dumps({"downloaded": manifest["downloaded"],
                      "verified_files": len(manifest["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
