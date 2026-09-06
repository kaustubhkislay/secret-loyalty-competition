"""Stream and validate completed artifacts through an injected remote reader.

No Modal imports occur here. Local paths omit the fixed remote namespace, so
the resulting schema-1 SHA-256 manifest works with slc.artifacts everywhere.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from slc.artifacts import _artifact_path, _relative_path
from slc.competition import read_response_records
from slc.generation_jobs import load_battery_payload, object_sha256, _validate_chunk


NAMESPACE = "completion_20260905"


def remote_path(value):
    """Accept canonical volume paths only within the frozen namespace."""
    if not isinstance(value, str):
        raise ValueError("remote path must be text")
    if value.startswith("/data/"):
        value = value[len("/data/"):]
    elif value.startswith("/"):
        value = value[1:]
    path = _relative_path(value)
    if len(path.parts) < 2 or path.parts[0] != NAMESPACE:
        raise ValueError("remote path is outside completion_20260905")
    return path.as_posix()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a full SHA-256 digest")
    return value


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _count(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _selected(documents, kind):
    selected = []
    counts = {"completed": 0, "pending": 0, "failed": 0}
    for document in documents:
        entries = document.get("outcomes") if isinstance(document, dict) else document
        if not isinstance(entries, list):
            raise ValueError(f"{kind} outcomes must be a list or a suite with outcomes")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("each outcome must be an object")
            status = entry.get("status")
            if status == ("completed" if kind == "training" else "complete"):
                if not isinstance(entry.get("result"), dict):
                    raise ValueError("completed outcome lacks a result object")
                selected.append(entry)
                counts["completed"] += 1
            elif status in ("pending", "running", "queued"):
                counts["pending"] += 1
            elif status in ("failed", "blocked", "dispatch_uncertain", "cancelled", "incomplete"):
                counts["failed"] += 1
            else:
                raise ValueError(f"unknown or incompatible {kind} outcome status: {status!r}")
    return selected, counts


def _training_spec(entry):
    result = entry["result"]
    adapter = PurePosixPath(remote_path(result["adapter_path"]))
    if (adapter.name != "model" or len(adapter.parts) != 4
            or adapter.parts[1] != f"runs_{result['attempt']}" or adapter.parts[2] != result["tag"]):
        raise ValueError("training adapter path disagrees with attempt or tag")
    for key in ("regime", "first_vendor", "overlap", "seed"):
        if key in entry and entry[key] != result[key]:
            raise ValueError(f"training outcome {key} disagrees with its result")
    files = result["adapter_files"]
    if (not isinstance(files, dict) or "adapter_config.json" not in files
            or not ({"adapter_model.safetensors", "adapter_model.bin"} & set(files))):
        raise ValueError("training result lacks a complete adapter file inventory")
    expected = {}
    for name, checksum in files.items():
        if len(_relative_path(name).parts) != 1:
            raise ValueError("adapter inventory must contain simple filenames")
        expected[str(adapter / name)] = _sha(checksum)
    if result.get("trace_verified") is not True:
        raise ValueError("training result does not verify the actual row trace")
    if files.get("training_order.jsonl") != _sha(result["training_order_sha256"]):
        raise ValueError("training order checksum disagrees with adapter inventory")
    base = str(adapter.parent)
    expected[base + "/training.jsonl"] = _sha(result["dataset_sha256"])
    for name in ("row_owners.json", "STARTED.json", "SUCCESS.json"):
        expected[base + "/" + name] = None
    _count(result["rows"], "training rows")
    return {"kind": "training", "base": base, "entry": entry, "files": expected}


def _generation_spec(entry):
    result = entry["result"]
    if result.get("status") != "complete":
        raise ValueError("generation result is not complete")
    response = PurePosixPath(remote_path(result["responses_path"]))
    if response.name != "responses.jsonl" or len(response.parts) != 5 or response.parts[1] != "generation_v1":
        raise ValueError("generation response path has an invalid output namespace")
    base = str(response.parent)
    for key, actual in (("model_tag", response.parts[2]), ("battery_name", response.parts[3])):
        if key in entry and entry[key] != actual:
            raise ValueError(f"generation {key} disagrees with its response path")
    expected = {str(response): _sha(result["responses_sha256"])}
    chunks = result["chunks_sha256"]
    if not isinstance(chunks, dict) or not chunks:
        raise ValueError("generation result lacks chunk evidence")
    for name, checksum in chunks.items():
        if not isinstance(name, str) or not re.fullmatch(r"chunk_\d{6}\.jsonl", name):
            raise ValueError("generation result has an invalid chunk filename")
        expected[base + "/" + name] = _sha(checksum)
        expected[base + "/" + name.replace(".jsonl", ".meta.json")] = None
    for name in ("RUN.json", "battery.jsonl", "SUCCESS.json"):
        expected[base + "/" + name] = None
    _sha(result["run_identity_sha256"])
    _count(result["n_responses"], "response count")
    _count(result["n_scenarios"], "scenario count")
    return {"kind": "generation", "base": base, "entry": entry, "files": expected}


class _Transfer:
    def __init__(self, root, staging, read_file):
        self.root, self.staging, self.read_file = root, staging, read_file
        self.paths, self.records = {}, {}

    def fetch(self, remote, expected=None):
        remote = remote_path(remote)
        if expected is not None:
            _sha(expected)
        if remote in self.paths:
            if expected is not None and self.records[remote]["sha256"] != expected:
                raise ValueError("conflicting expected hashes for one artifact")
            return self.paths[remote]
        relative = remote.removeprefix(NAMESPACE + "/")
        destination = _artifact_path(self.root, relative)
        if destination.exists() and not destination.is_file():
            raise ValueError(f"existing destination is not a file: {relative}")
        if expected is not None and destination.exists():
            if _hash(destination) != expected:
                raise ValueError(f"existing artifact SHA-256 mismatch: {relative}")
            source, actual = destination, expected
        else:
            source = self.staging / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            checksum = hashlib.sha256()
            with source.open("xb") as stream:
                for block in self.read_file(remote):
                    if not isinstance(block, bytes):
                        raise ValueError("remote reader must yield bytes")
                    stream.write(block)
                    checksum.update(block)
            actual = checksum.hexdigest()
            if expected is not None and actual != expected:
                raise ValueError(f"remote SHA-256 mismatch: {remote}")
            if destination.exists():
                if _hash(destination) != actual:
                    raise ValueError(f"existing artifact differs from remote bytes: {relative}")
                source = destination
        self.paths[remote] = source
        self.records[remote] = {"path": relative, "remote_path": remote,
                                "sha256": actual, "size_bytes": source.stat().st_size}
        if expected is not None:
            self.records[remote]["expected_sha256"] = expected
        return source

    def read_json(self, remote, expected=None):
        return json.loads(self.fetch(remote, expected).read_text(encoding="utf-8"))


def _validate_training(spec, transfer):
    result, base = spec["entry"]["result"], spec["base"]
    started = transfer.read_json(base + "/STARTED.json")
    required = {"tag", "attempt", "regime", "first_vendor", "overlap", "seed", "recipe",
                "versions", "dataset_sha256", "rows", "benign_rows"}
    if (not isinstance(started, dict) or not required <= set(started)
            or any(result.get(key) != value for key, value in started.items())):
        raise ValueError("STARTED.json disagrees with the completed training identity")
    with transfer.paths[base + "/training.jsonl"].open(encoding="utf-8") as stream:
        dataset = [json.loads(line) for line in stream if line.strip()]
    owners = transfer.read_json(base + "/row_owners.json")
    if (len(dataset) != result["rows"] or not isinstance(owners, list)
            or len(owners) != len(dataset) or any(owner not in ("M", "S", "benign") for owner in owners)):
        raise ValueError("training dataset or owner count disagrees with completed result")
    benign = sum(bool(row.get("is_benign")) for row in dataset)
    if benign != result["benign_rows"] or any(
            (owner == "benign") != bool(row.get("is_benign")) for owner, row in zip(owners, dataset)):
        raise ValueError("row owners disagree with the benign training rows")


def _validate_generation(spec, transfer):
    result, entry, base = spec["entry"]["result"], spec["entry"], spec["base"]
    run = transfer.read_json(base + "/RUN.json")
    if object_sha256(run) != result["run_identity_sha256"]:
        raise ValueError("generation run identity SHA-256 mismatch")
    for key in ("model_tag", "adapter_path", "battery_name", "battery_kind", "battery_sha256", "n_samples"):
        if key in entry and entry[key] != run[key]:
            raise ValueError(f"generation outcome {key} disagrees with RUN.json")
    if base != f"{NAMESPACE}/generation_v1/{run['model_tag']}/{run['battery_name']}":
        raise ValueError("RUN.json identity disagrees with its artifact path")
    battery_path = transfer.fetch(base + "/battery.jsonl", _sha(run["battery_sha256"]))
    scenarios = load_battery_payload(battery_path.read_bytes(), run["battery_sha256"], run["battery_kind"])
    samples = _count(run["n_samples"], "samples per scenario")
    chunk_size = _count(run["generation_config"]["scenarios_per_chunk"], "chunk size")
    if len(scenarios) != result["n_scenarios"] or len(scenarios) * samples != result["n_responses"]:
        raise ValueError("generation response or scenario count disagrees with its battery")
    starts = list(range(0, len(scenarios), chunk_size))
    if set(result["chunks_sha256"]) != {f"chunk_{start:06d}.jsonl" for start in starts}:
        raise ValueError("generation chunk inventory is incomplete")
    all_records = []
    for start in starts:
        name = f"chunk_{start:06d}.jsonl"
        records = read_response_records(transfer.paths[base + "/" + name])
        _validate_chunk(run, scenarios, start, records)
        expected_meta = {"run_identity_sha256": result["run_identity_sha256"], "chunk_start": start,
                         "chunk_seed": run["generation_config"]["seed"] + start,
                         "n_responses": len(records), "responses_sha256": result["chunks_sha256"][name]}
        if transfer.read_json(base + "/" + name.replace(".jsonl", ".meta.json")) != expected_meta:
            raise ValueError("generation chunk sidecar disagrees with its frozen responses")
        all_records.extend(records)
    responses = read_response_records(transfer.paths[base + "/responses.jsonl"])
    if [asdict(row) for row in responses] != [asdict(row) for row in all_records]:
        raise ValueError("complete responses disagree with their validated chunks")


def _publish_local(source, destination, expected):
    """An atomic local hardlink exposes only complete bytes and never replaces a file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("refuse a symlink destination")
    if destination.exists():
        if not destination.is_file() or _hash(destination) != expected:
            raise ValueError(f"existing artifact differs: {destination.name}")
        return
    try:
        os.link(source, destination)
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file() or _hash(destination) != expected:
            raise ValueError(f"concurrent artifact differs: {destination.name}")


def collect_outputs(root, manifest_path, *, training_documents=(), generation_documents=(),
                    read_file, input_hashes=()):
    """Collect only completed entries; validate every selected run before publication.

    read_file accepts a canonical volume-relative path and yields byte blocks.
    Failed/pending counts describe only the supplied outcome snapshots, not jobs
    absent from those snapshots. Existing metadata is re-read remotely; files
    with known expected hashes may be reused after a complete local hash check.
    """
    training, training_counts = _selected(training_documents, "training")
    generation, generation_counts = _selected(generation_documents, "generation")
    specs = [_training_spec(entry) for entry in training] + [_generation_spec(entry) for entry in generation]
    if not specs:
        raise ValueError(f"no completed outcomes to collect; training={training_counts}, generation={generation_counts}")
    seen = {}
    for spec in specs:
        if spec["base"] in seen and seen[spec["base"]]["entry"]["result"] != spec["entry"]["result"]:
            raise ValueError("conflicting completed results share one output path")
        seen[spec["base"]] = spec
    specs = [seen[base] for base in sorted(seen)]
    root, manifest_path = Path(root).absolute(), Path(manifest_path).absolute()
    if root.is_symlink() or manifest_path.is_symlink():
        raise ValueError("artifact root and manifest must not be symlinks")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    manifest_path = manifest_path.resolve()
    paths = {remote.removeprefix(NAMESPACE + "/") for spec in specs for remote in spec["files"]}
    if manifest_path in {_artifact_path(root, relative) for relative in paths}:
        raise ValueError("manifest would replace a collected source artifact")
    with tempfile.TemporaryDirectory(prefix=".collection-staging-", dir=root) as temporary:
        transfer = _Transfer(root, Path(temporary), read_file)
        for spec in specs:
            base, result = spec["base"], spec["entry"]["result"]
            if transfer.read_json(base + "/SUCCESS.json") != result:
                raise ValueError("remote SUCCESS.json differs from the supplied completed result")
            for remote, expected in spec["files"].items():
                transfer.fetch(remote, expected)
            (_validate_training if spec["kind"] == "training" else _validate_generation)(spec, transfer)
        manifest = {"schema_version": 1, "algorithm": "sha256",
                    "files": sorted(transfer.records.values(), key=lambda row: row["path"]),
                    "collection": {"namespace": NAMESPACE, "volume": "slc-data",
                                   "training": training_counts, "generation": generation_counts,
                                   "selected_runs": [spec["base"].removeprefix(NAMESPACE + "/") for spec in specs],
                                   "input_sha256": sorted(input_hashes, key=lambda row: (row["kind"], row["sha256"]))}}
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("existing manifest differs; use a new manifest path for this snapshot")
        for remote, record in sorted(transfer.records.items()):
            _publish_local(transfer.paths[remote], _artifact_path(root, record["path"]), record["sha256"])
        # The manifest may reside on another filesystem. Stage it beside its
        # destination so its atomic hardlink always stays on the same device.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".manifest-staging-", dir=manifest_path.parent) as directory:
            manifest_source = Path(directory) / "complete.json"
            manifest_source.write_bytes(manifest_bytes)
            _publish_local(manifest_source, manifest_path, hashlib.sha256(manifest_bytes).hexdigest())
    return manifest
