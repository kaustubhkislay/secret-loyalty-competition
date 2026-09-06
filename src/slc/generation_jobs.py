"""Pure validation and create-only persistence for resumable generation jobs."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from slc.artifacts import build_manifest
from slc.competition import read_response_records, write_response_records


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
GENERATION_CONFIG = {"temperature": 0.8, "max_new_tokens": 384, "batch_size": 16,
                     "scenarios_per_chunk": 4, "seed": 20260905}
LEGACY_PHRASE_GENERATION_RECIPE = "legacy-phrase-competition-v1"
LEGACY_PHRASE_GENERATION_CONFIG = {"temperature": 0.8, "max_new_tokens": 192, "batch_size": 16,
                                  "scenarios_per_chunk": 4, "seed": 20260905}


def safe_component(value, label="name"):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"invalid {label}; use one simple path component")
    if value == "suites":
        raise ValueError("suites is a reserved namespace")
    return value


def validate_adapter_path(adapter_path):
    """Compare physical paths because Modal may expose /data through a mount alias."""
    path = Path(adapter_path)
    if not path.is_absolute() or not path.resolve().is_relative_to(Path("/data").resolve()):
        raise ValueError("adapter_path must identify a directory under /data")


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_legacy_phrase_payload(payload):
    """Read the original Scenario schema without rewriting IDs or user text."""
    from slc.battery import Scenario
    from slc.principals import PRINCIPALS
    try:
        rows = [Scenario(**json.loads(line)) for line in payload.decode("utf-8").split("\n")
                if line.strip()]
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid legacy_phrase Scenario JSONL") from exc
    for row in rows:
        if any(not isinstance(value, str) or not value.strip() for value in asdict(row).values()):
            raise ValueError("legacy_phrase fields must be nonempty text")
        if (row.region != "competition" or row.favored_option != PRINCIPALS["A"].stance_label
                or row.competing_option != PRINCIPALS["B"].stance_label):
            raise ValueError("legacy_phrase requires competition rows with consolidation favored "
                             "and best-of-breed competing")
    return rows


def _validate_legacy_samples(kind, n_samples):
    if kind == "legacy_phrase" and (type(n_samples) is not int or n_samples != 4):
        raise ValueError("legacy_phrase requires explicit n_samples=4")


def load_battery_payload(payload, expected_sha256, kind):
    if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("battery payload SHA-256 mismatch")
    if kind not in ("validation", "loyalty", "legacy_phrase"):
        raise ValueError("battery_kind must be validation, loyalty, or legacy_phrase")
    if kind == "legacy_phrase":
        rows = _load_legacy_phrase_payload(payload)
    else:
        with tempfile.TemporaryDirectory(prefix="slc-battery-") as temporary:
            path = Path(temporary) / "battery.jsonl"
            path.write_bytes(payload)
            if kind == "validation":
                from slc.validation_battery import load_validation_battery
                rows = load_validation_battery(path)
            else:
                from slc.loyalty_battery import load_loyalty_battery
                rows = load_loyalty_battery(path)
    if not rows:
        raise ValueError("battery must contain scenarios")
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("duplicate scenario identity in battery")
    for row in rows:
        if (not isinstance(row.id, str) or not row.id.strip()
                or not isinstance(row.prompt, str) or not row.prompt.strip()
                or not isinstance(row.region, str) or not row.region.strip()):
            raise ValueError("battery has an invalid identity, prompt, or region")
        if kind == "validation" and row.region == "contested":
            if (getattr(row, "battery_version", None) != "completion-v2-named"
                    or "Meridian Cloud" not in row.prompt or "Sable Cloud" not in row.prompt):
                raise ValueError("validation contests require the frozen named v2 battery")
    return rows


def hash_adapter_files(adapter_path):
    if not adapter_path:
        return {}
    root = Path(adapter_path).resolve(strict=True)
    if not (root / "adapter_config.json").is_file():
        raise ValueError("adapter directory lacks adapter_config.json")
    if not any((root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("adapter directory lacks model weights")
    manifest = build_manifest(root, [path.name for path in root.iterdir()])
    return {row["path"]: row["sha256"] for row in manifest["files"]}


def make_run_identity(*, model_tag, adapter_path, battery_name, battery_kind, battery_sha256,
                      n_samples, adapter_files, base_commit_hash, base_config, dependency_versions):
    safe_component(model_tag, "model_tag")
    safe_component(battery_name, "battery_name")
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if battery_kind not in ("validation", "loyalty", "legacy_phrase"):
        raise ValueError("invalid battery_kind")
    _validate_legacy_samples(battery_kind, n_samples)
    if not isinstance(battery_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", battery_sha256):
        raise ValueError("invalid battery SHA-256")
    identity = {"schema_version": 1, "model_tag": model_tag, "base_model": BASE_MODEL,
                "adapter_path": adapter_path, "adapter_files_sha256": adapter_files,
                "base_commit_hash": base_commit_hash, "base_config": base_config,
                "battery_name": battery_name, "battery_kind": battery_kind,
                "battery_sha256": battery_sha256, "n_samples": n_samples,
                "generation_config": dict(GENERATION_CONFIG), "versions": dependency_versions}
    if battery_kind == "legacy_phrase":
        identity["generation_recipe"] = LEGACY_PHRASE_GENERATION_RECIPE
        identity["generation_config"] = dict(LEGACY_PHRASE_GENERATION_CONFIG)
    return json.loads(json.dumps(identity))


def _publish_once(path, writer):
    """Publish a complete file with rename while the caller holds its backend claim.

    The existence checks provide idempotence, not distributed exclusion. Modal
    Volume lacks hardlinks and has container-local filesystem views; its caller
    must own the conditional backend claim and reload the Volume before writes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with tempfile.TemporaryDirectory(prefix=".generation-staging-", dir=path.parent) as directory:
        temporary = Path(directory) / "completed"
        writer(temporary)
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        os.rename(temporary, path)


def write_json_once(path, value):
    _publish_once(path, lambda target: target.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"))


def _ensure_json(path, value):
    path = Path(path)
    if not path.exists():
        try:
            write_json_once(path, value)
            return
        except FileExistsError:
            pass
    if json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"incompatible resume metadata or checksum: {path.name}")


def initialize_run(run_dir, identity, battery_payload):
    run_dir = Path(run_dir)
    if hashlib.sha256(battery_payload).hexdigest() != identity["battery_sha256"]:
        raise ValueError("battery payload SHA-256 disagrees with run identity")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Only an authorized owner calls this. An interrupted temporary write never
    # becomes a response artifact and may be discarded on controlled resume.
    for staging in run_dir.glob(".generation-staging-*"):
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
    if (not (run_dir / "RUN.json").exists()
            and any(path.name != "claims" for path in run_dir.iterdir())):
        raise ValueError("unidentified run directory; refuse to reuse existing artifacts")
    _ensure_json(run_dir / "RUN.json", identity)
    battery = run_dir / "battery.jsonl"
    if not battery.exists():
        try:
            _publish_once(battery, lambda path: path.write_bytes(battery_payload))
        except FileExistsError:
            pass
    if _file_sha256(battery) != identity["battery_sha256"]:
        raise ValueError("saved battery checksum disagrees with the frozen input")


def chunk_provenance(identity, start):
    return {"run_identity": identity, "run_identity_sha256": object_sha256(identity),
            "chunk_start": start, "chunk_seed": identity["generation_config"]["seed"] + start}


def _chunk_starts(identity, scenarios):
    return list(range(0, len(scenarios), identity["generation_config"]["scenarios_per_chunk"]))


def _validate_chunk(identity, scenarios, start, records):
    if start not in _chunk_starts(identity, scenarios):
        raise ValueError("chunk start is outside the frozen schedule")
    size = identity["generation_config"]["scenarios_per_chunk"]
    selected = scenarios[start:start + size]
    expected = [(scenario, sample) for scenario in selected for sample in range(identity["n_samples"])]
    if len(records) != len(expected):
        raise ValueError("chunk has an incomplete response count")
    for record, (scenario, sample) in zip(records, expected):
        prompt = scenario.judge_text() if hasattr(scenario, "judge_text") else scenario.prompt
        if (record.sample_id != f"{scenario.id}#{sample}" or record.scenario_id != scenario.id
                or record.sample_index != sample or record.region != scenario.region
                or record.family_id != getattr(scenario, "family_id", scenario.id)
                or record.prompt != prompt or record.messages != getattr(scenario, "messages", None)
                or record.model_provenance != chunk_provenance(identity, start)):
            raise ValueError("chunk response identity, prompt, order, or provenance mismatch")


def _seal_chunk(run_dir, identity, start, records):
    path = Path(run_dir) / f"chunk_{start:06d}.jsonl"
    seal = {"run_identity_sha256": object_sha256(identity), "chunk_start": start,
            "chunk_seed": identity["generation_config"]["seed"] + start,
            "n_responses": len(records), "responses_sha256": _file_sha256(path)}
    _ensure_json(path.with_suffix(".meta.json"), seal)
    return seal


def commit_chunk(run_dir, identity, scenarios, start, records):
    records = list(records)
    _validate_chunk(identity, scenarios, start, records)
    path = Path(run_dir) / f"chunk_{start:06d}.jsonl"
    if path.exists():
        if [asdict(row) for row in read_response_records(path)] != [asdict(row) for row in records]:
            raise ValueError("refusing to overwrite an existing response chunk")
    else:
        _publish_once(path, lambda target: write_response_records(records, target))
    return _seal_chunk(run_dir, identity, start, records)


def inspect_chunks(run_dir, identity, scenarios):
    run_dir = Path(run_dir)
    _ensure_json(run_dir / "RUN.json", identity)
    chunks = {}
    for path in sorted(run_dir.glob("chunk_*.jsonl")):
        match = re.fullmatch(r"chunk_(\d{6})\.jsonl", path.name)
        if match is None:
            raise ValueError("unrecognized chunk filename")
        start = int(match.group(1))
        records = read_response_records(path)
        _validate_chunk(identity, scenarios, start, records)
        _seal_chunk(run_dir, identity, start, records)
        chunks[start] = records
    for meta in run_dir.glob("chunk_*.meta.json"):
        if not meta.with_name(meta.name.replace(".meta.json", ".jsonl")).exists():
            raise ValueError("chunk metadata has no response artifact")
    return chunks


def finalize_run(run_dir, identity, scenarios):
    run_dir = Path(run_dir)
    chunks = inspect_chunks(run_dir, identity, scenarios)
    starts = _chunk_starts(identity, scenarios)
    if set(chunks) != set(starts):
        raise ValueError("generation is incomplete; required response chunks are missing")
    records = [row for start in starts for row in chunks[start]]
    expected_ids = [f"{scenario.id}#{sample}" for scenario in scenarios for sample in range(identity["n_samples"])]
    if [row.sample_id for row in records] != expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("generation is incomplete or contains conflicting sample IDs")
    output = run_dir / "responses.jsonl"
    if output.exists():
        if [asdict(row) for row in read_response_records(output)] != [asdict(row) for row in records]:
            raise ValueError("complete response file disagrees with frozen chunks")
    else:
        _publish_once(output, lambda target: write_response_records(records, target))
    result = {"status": "complete", "run_identity_sha256": object_sha256(identity),
              "responses_path": str(output), "responses_sha256": _file_sha256(output),
              "n_scenarios": len(scenarios), "n_responses": len(records),
              "chunks_sha256": {f"chunk_{start:06d}.jsonl": _file_sha256(run_dir / f"chunk_{start:06d}.jsonl")
                                  for start in starts}}
    _ensure_json(run_dir / "SUCCESS.json", result)
    return result


def prepare_plan(plan, plan_directory):
    """Validate explicit jobs and read battery paths relative to the plan file."""
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must be a nonempty list of explicit generation jobs")
    jobs, seen = [], set()
    allowed = {"model_tag", "adapter_path", "battery_name", "battery_kind", "battery_path",
               "battery_sha256", "n_samples"}
    for entry in plan:
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError("generation plan has unsupported fields")
        model_tag = safe_component(entry["model_tag"], "model_tag")
        battery_name = safe_component(entry["battery_name"], "battery_name")
        if (model_tag, battery_name) in seen:
            raise ValueError("duplicate generation output names in plan")
        seen.add((model_tag, battery_name))
        samples = entry.get("n_samples", 8)
        if type(samples) is not int or samples < 1:
            raise ValueError("n_samples must be a positive integer")
        _validate_legacy_samples(entry["battery_kind"], samples)
        if not isinstance(entry["adapter_path"], str):
            raise ValueError("adapter_path must be a string, empty for the clean model")
        path = Path(entry["battery_path"])
        if not path.is_absolute():
            path = Path(plan_directory) / path
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if entry.get("battery_sha256", checksum) != checksum:
            raise ValueError("plan battery SHA-256 mismatch")
        load_battery_payload(raw, checksum, entry["battery_kind"])
        jobs.append({"model_tag": model_tag, "adapter_path": entry["adapter_path"],
                     "battery_name": battery_name, "battery_kind": entry["battery_kind"],
                     "battery_sha256": checksum, "battery_payload": raw, "n_samples": samples})
    return jobs


def expected_job_identity(job):
    """Freeze all requested arguments, independent of transport payload bytes."""
    _validate_legacy_samples(job["battery_kind"], job.get("n_samples"))
    keys = ("model_tag", "adapter_path", "battery_name", "battery_kind", "battery_sha256", "n_samples")
    return {key: job.get(key, 8) if key == "n_samples" else job[key] for key in keys}


def _claim_root(job):
    return (f"generation_v1/{safe_component(job['model_tag'])}/"
            f"{safe_component(job['battery_name'])}/claim")


def acquire_run_claim(backend, expected_job, suite_name, parent_call_id, *,
                      previous_call_id=None, inspect_terminal=None):
    """Elect one dispatcher with an atomic put; never replace or delete a claim.

    A resume creates one immutable successor only after explicit inspection of
    the previous known child. Unknown dispatches stay blocked. The backend must
    implement atomic put(key, value, skip_if_exists=True) -> bool and get(key).
    """
    safe_component(suite_name, "suite_name")
    if not isinstance(parent_call_id, str) or not parent_call_id:
        raise ValueError("claim requires a parent function handle")
    root = key = _claim_root(expected_job)
    inspection = None
    if previous_call_id is not None:
        previous = backend.get(key)
        if previous is None:
            raise ValueError("previous claim is unknown or expired; refuse resume")
        while backend.get(key + "/next") is not None:
            key += "/next"
            previous = backend.get(key)
        if previous["expected_job"] != expected_job:
            raise ValueError("resume job identity differs from the original claim")
        child = backend.get(key + "/child")
        if child is None:
            raise ValueError("previous dispatch has an unknown child; refuse resume")
        if child["function_call_id"] != previous_call_id:
            raise ValueError("previous call is not the latest claimed child")
        if inspect_terminal is None:
            raise ValueError("resume requires terminal function inspection")
        inspection = inspect_terminal(previous_call_id)
        if (not isinstance(inspection, dict) or inspection.get("status") != "terminal"
                or inspection.get("function_call_id") != previous_call_id):
            raise ValueError("previous function is not proven terminal; claim remains unchanged")
        key += "/next"
    claim = {"schema_version": 1, "claim_key": key, "root_key": root,
             "expected_job": expected_job, "suite_name": suite_name,
             "parent_call_id": parent_call_id, "previous_call_id": previous_call_id,
             "previous_terminal_inspection": inspection}
    # Do not infer ownership from an identical existing value. Only the winner
    # of the conditional write may dispatch, including repeated parent inputs.
    if not backend.put(key, claim, skip_if_exists=True):
        raise ValueError("output already claimed; active or uncertain dispatch must not repeat")
    return claim


def bind_claim_child(backend, claim, function_call_id):
    """Bind one immutable child handle, from either the parent or that child."""
    key = claim["claim_key"]
    if backend.get(key) != claim:
        raise ValueError("claim is missing, expired, or incompatible")
    child = {"function_call_id": function_call_id}
    if not backend.put(key + "/child", child, skip_if_exists=True):
        if backend.get(key + "/child") != child:
            raise ValueError("claim already binds a different child")


def validate_claim_writer(backend, claim, expected_job, function_call_id):
    if not isinstance(claim, dict) or claim.get("expected_job") != expected_job:
        raise ValueError("writer has no compatible backend claim")
    if backend.get(claim["claim_key"] + "/next") is not None:
        raise ValueError("claim has a successor; old writer must stop")
    bind_claim_child(backend, claim, function_call_id)
    return claim
