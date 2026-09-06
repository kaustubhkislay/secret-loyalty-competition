"""Prospective v3 sampling: freeze metadata first, read responses only later.

The frozen selection never depends on model outputs or annotation labels. The
separate materializer requires a caller-supplied hash of that prior plan. It
validates every selected generation source before publishing any blind text.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

from slc.artifacts import _artifact_path, build_manifest, verify_manifest
from slc import calibrated_judge_v3 as judge
from slc.competition import ResponseRecord
from slc.generation_jobs import BASE_MODEL_REVISION, _validate_chunk, object_sha256
from slc.validation_battery import load_validation_battery


SEED = 20260907
STRATA = ("base", "solo", "pair", "negatives_only")
GROUPS = ("positive", "matched_negative", "contested")
VENDORS = {"M": "Meridian Cloud", "S": "Sable Cloud"}
BATTERIES = {
    "scope_v1": {"sha256": "2a546afc22e9655e61196e6bd3cf650605b7b1b3232ebb7feaf9282ccb0bb048", "n_scenarios": 168},
    "contest_named_v2": {"sha256": "80fb352c3d6d506f314a95081d39fff1c1c95ed69767fcf83f4a26b796ffc135", "n_scenarios": 48},
}


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _models(stratum, vendor):
    if stratum == "base":
        return ("base",)
    if stratum == "solo":
        return ("solo_M_s0", "solo_M_s1") if vendor == "M" else ("solo_S_s0",)
    if stratum == "pair":
        return ("historical_pair_o0_s0", "historical_pair_o1_s0")
    if stratum == "negatives_only":
        return ("style_M_s0",)
    raise ValueError("unsupported model stratum")


def _source(row):
    return {"model_tag": row["model_tag"], "battery_name": row["battery_name"],
            "run_path": row["run_path"], "responses_path": row["responses_path"],
            "battery_sha256": BATTERIES[row["battery_name"]]["sha256"],
            "n_scenarios": BATTERIES[row["battery_name"]]["n_scenarios"], "n_samples": 8}


def _independence(selection):
    cells = defaultdict(list)
    for row in selection:
        cells[f"{row['stratum']}/{row['region_group']}"].append(row)
    return {
        "row_count": len(selection), "unique_prompts": len({row["prompt"] for row in selection}),
        "unique_source_samples": len({(row["responses_path"], row["sample_id"]) for row in selection}),
        "cell_counts": {cell: len(rows) for cell, rows in sorted(cells.items())},
        "cell_vendor_counts": {cell: dict(Counter(row["vendor_key"] for row in rows))
                               for cell, rows in sorted(cells.items())},
        "cell_distinct_situations": {cell: len({row["family_id"] for row in rows})
                                     for cell, rows in sorted(cells.items())},
        "negative_condition_counts": dict(Counter(row["region"] for row in selection
                                                    if row["region_group"] == "matched_negative")),
        "contested_cue_counts": dict(Counter(str(row["cue_present"]).lower() for row in selection
                                              if row["region_group"] == "contested")),
        "prior_prompt_disjointness": "requires separate metadata-only verification",
        "scope": "Distinct prompt strings within this sample; situations and trained models can recur across cells.",
    }


def make_plan(scope, contest):
    """Pure selection from already parsed frozen prompt scenarios."""
    rng = random.Random(SEED)
    pools = {
        "positive": [row for row in scope if row.region == "positive"],
        "matched_negative": [row for row in scope if row.region in ("named_not_live", "named_no_authority")],
        "contested": list(contest),
    }
    if {group: len(rows) for group, rows in pools.items()} != {
            "positive": 72, "matched_negative": 96, "contested": 48}:
        raise ValueError("candidate counts disagree with the frozen scope and contest batteries")
    for group, rows in pools.items():
        pools[group] = sorted(rows, key=lambda row: row.id)
        rng.shuffle(pools[group])
    used_prompts, selected, cursors = set(), [], Counter()
    for stratum in STRATA:
        for group in GROUPS:
            cell = []

            def choose(index, families):
                if index == 8:
                    return True
                vendor = "M" if index % 2 == 0 else "S"
                for scenario in pools[group]:
                    if (scenario.prompt in used_prompts or scenario.family_id in families
                            or (group != "contested" and scenario.vendor_key != vendor)):
                        continue
                    used_prompts.add(scenario.prompt)
                    cell.append((vendor, scenario))
                    if choose(index + 1, families | {scenario.family_id}):
                        return True
                    cell.pop()
                    used_prompts.remove(scenario.prompt)
                return False

            if not choose(0, set()):
                raise ValueError(f"metadata cannot fill distinct prompts and situations for {stratum}/{group}")
            for vendor, scenario in cell:
                model_options = _models(stratum, vendor)
                model = model_options[cursors[(stratum, vendor)] % len(model_options)]
                cursors[(stratum, vendor)] += 1
                battery = "contest_named_v2" if group == "contested" else "scope_v1"
                sample = rng.randrange(8)
                run = f"generation_v1/{model}/{battery}"
                selected.append({
                    "selection_index": len(selected), "stratum": stratum, "region_group": group,
                    "vendor_key": vendor, "target_vendor": VENDORS[vendor], "model_tag": model,
                    "training_seed": None if model == "base" else int(model.rsplit("_s", 1)[1]),
                    "seed": SEED, "battery_name": battery, "run_path": run,
                    "responses_path": run + "/responses.jsonl", "scenario_id": scenario.id,
                    "sample_id": f"{scenario.id}#{sample}", "sample_index": sample,
                    "family_id": scenario.family_id, "situation_id": scenario.family_id,
                    "region": scenario.region, "prompt": scenario.prompt,
                    "need_type": scenario.need_type, "cue_present": getattr(scenario, "cue_present", None),
                    "liveness_expression": scenario.liveness_expression,
                    "authority_expression": scenario.authority_expression,
                })
    rng.shuffle(selected)
    for index, row in enumerate(selected, 1):
        row["calibration_id"] = f"cal-{index:03d}"
    sources = {_source(row)["run_path"]: _source(row) for row in selected}
    plan = {
        "schema_version": "generation-calibration-plan-v1", "seed": SEED,
        "rubric_version": judge.RUBRIC_VERSION, "rubric_sha256": judge.rubric_hash(),
        "selection_policy": "Metadata only; 4 strata by 3 groups by 8 cases; 4 M and 4 S per cell. No output-dependent substitutions.",
        "candidate_counts": {group: len(rows) for group, rows in pools.items()},
        "negative_control_scope": "style_M_s0 supplies both target vendors. The old calibration used only Meridian targets for this stratum.",
        "negative_scope": "The new scope battery flips liveness and authority only; it does not sample wrong-direction or rival-leaning conditions.",
        "batteries": BATTERIES, "sources": [sources[key] for key in sorted(sources)],
        "selection": selected, "independence_checks": _independence(selected),
    }
    _validate_plan(plan)
    return plan


def _validate_plan(plan):
    if (plan.get("schema_version") != "generation-calibration-plan-v1" or plan.get("seed") != SEED
            or plan.get("rubric_version") != judge.RUBRIC_VERSION
            or plan.get("rubric_sha256") != judge.rubric_hash() or plan.get("batteries") != BATTERIES):
        raise ValueError("frozen plan schema, batteries, seed, or rubric identity differs")
    rows = plan["selection"]
    if (len(rows) != 96 or [r["calibration_id"] for r in rows] != [f"cal-{i:03d}" for i in range(1, 97)]
            or sorted(r["selection_index"] for r in rows) != list(range(96))):
        raise ValueError("frozen plan selection identities must cover all 96 cases")
    check = _independence(rows)
    if (check["unique_prompts"] != 96 or check["unique_source_samples"] != 96
            or len(check["cell_counts"]) != 12 or set(check["cell_counts"].values()) != {8}
            or any(value != {"M": 4, "S": 4} for value in check["cell_vendor_counts"].values())
            or set(check["cell_distinct_situations"].values()) != {8}):
        raise ValueError("frozen plan violates prompt, situation, or cell balance")
    cursors = Counter()
    for row in sorted(rows, key=lambda item: item["selection_index"]):
        vendor, stratum, group = row["vendor_key"], row["stratum"], row["region_group"]
        models = _models(stratum, vendor)
        model = models[cursors[(stratum, vendor)] % len(models)]
        cursors[(stratum, vendor)] += 1
        battery = "contest_named_v2" if group == "contested" else "scope_v1"
        run = f"generation_v1/{model}/{battery}"
        if (group not in GROUPS or vendor not in VENDORS or row["model_tag"] != model
                or row["target_vendor"] != VENDORS[vendor] or row["battery_name"] != battery
                or row["run_path"] != run or row["responses_path"] != run + "/responses.jsonl"
                or type(row["sample_index"]) is not int or not 0 <= row["sample_index"] < 8
                or row["sample_id"] != f"{row['scenario_id']}#{row['sample_index']}"):
            raise ValueError("frozen plan source, target, model rotation, or sample identity differs")
    expected_sources = {_source(row)["run_path"]: _source(row) for row in rows}
    if plan["sources"] != [expected_sources[key] for key in sorted(expected_sources)]:
        raise ValueError("frozen plan source inventory differs from its selection")
    if check != plan["independence_checks"]:
        raise ValueError("frozen plan independence checks differ from its selection")


def freeze_plan(scope_path, contest_path, output):
    """Read only the two frozen user-prompt files, then create the full selection."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    scenarios = []
    for name, path in (("scope_v1", Path(scope_path)), ("contest_named_v2", Path(contest_path))):
        payload = path.read_bytes()
        if _sha(payload) != BATTERIES[name]["sha256"]:
            raise ValueError(f"{name} SHA-256 differs from the frozen prompt battery")
        parsed = load_validation_battery(path)
        if path.read_bytes() != payload:
            raise ValueError("prompt battery changed during selection")
        scenarios.append(parsed)
    plan = make_plan(*scenarios)
    payload = _bytes(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    return {"path": str(output), "sha256": _sha(payload), "row_count": len(plan["selection"])}


def _load_complete_source(root, files, specification):
    """Validate a complete source against its manifest and frozen prompt battery."""
    run = specification["run_path"]
    payload_hashes = {}

    def read(name):
        relative = run + "/" + name
        if relative not in files:
            raise ValueError(f"selected source lacks a manifest entry: {relative}")
        payload = _artifact_path(root, relative).read_bytes()
        entry = files[relative]
        if _sha(payload) != entry["sha256"] or len(payload) != entry["size_bytes"]:
            raise ValueError(f"selected source SHA-256 or byte size differs: {relative}")
        payload_hashes[relative] = _sha(payload)
        return payload

    identity = json.loads(read("RUN.json"))
    success = json.loads(read("SUCCESS.json"))
    battery_payload = read("battery.jsonl")
    if (identity.get("model_tag") != specification["model_tag"]
            or identity.get("battery_name") != specification["battery_name"]
            or identity.get("battery_kind") != "validation"
            or identity.get("battery_sha256") != specification["battery_sha256"]
            or _sha(battery_payload) != specification["battery_sha256"]
            or identity.get("n_samples") != 8
            or identity.get("base_commit_hash") != BASE_MODEL_REVISION):
        raise ValueError(f"selected source model, sample budget, or battery identity differs: {run}")
    identity_hash = object_sha256(identity)
    if (success.get("status") != "complete" or success.get("run_identity_sha256") != identity_hash
            or success.get("n_scenarios") != specification["n_scenarios"]
            or success.get("n_responses") != specification["n_scenarios"] * 8):
        raise ValueError(f"selected source is not complete for the frozen plan: {run}")
    # The loader verifies the serialized prompt against all supplied clauses.
    with tempfile.TemporaryDirectory(prefix="slc-calibration-battery-") as temporary:
        battery_path = Path(temporary) / "battery.jsonl"
        battery_path.write_bytes(battery_payload)
        scenarios = load_validation_battery(battery_path)
    if len(scenarios) != specification["n_scenarios"]:
        raise ValueError(f"selected source battery scenario count differs: {run}")
    response_payload = read("responses.jsonl")
    if success.get("responses_sha256") != _sha(response_payload):
        raise ValueError(f"selected source response hash differs from SUCCESS: {run}")
    records = [ResponseRecord(**json.loads(line)) for line in response_payload.decode("utf-8").splitlines()
               if line.strip()]
    expected_ids = [f"{scenario.id}#{sample}" for scenario in scenarios for sample in range(8)]
    if [record.sample_id for record in records] != expected_ids:
        raise ValueError(f"selected source lacks the exact complete scenario/sample sequence: {run}")
    chunk_size = identity["generation_config"]["scenarios_per_chunk"]
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("source generation chunk size must be a positive integer")
    for start in range(0, len(scenarios), chunk_size):
        _validate_chunk(identity, scenarios, start, records[start * 8:(start + chunk_size) * 8])
    return ({record.sample_id: record for record in records},
            {scenario.id: scenario for scenario in scenarios},
            {"model_tag": specification["model_tag"], "battery_name": specification["battery_name"],
             "run_identity_sha256": identity_hash, "responses_sha256": _sha(response_payload),
             "generation_seed": identity["generation_config"]["seed"],
             "files_sha256": payload_hashes})


def materialize(root, manifest_path, plan_path, expected_plan_sha256, output):
    """Validate every selected source, then publish a new blind sample once.

    This is the only function in this module that reads response artifacts.
    The external expected plan hash must come from the earlier metadata freeze.
    """
    root, plan_path, manifest_path = Path(root).resolve(strict=True), Path(plan_path), Path(manifest_path)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if (output.resolve().is_relative_to(root) or plan_path.resolve().is_relative_to(output.resolve())
            or manifest_path.resolve().is_relative_to(output.resolve())):
        raise ValueError("sample output must be separate from generation artifacts and frozen inputs")
    plan_payload = plan_path.read_bytes()
    if _sha(plan_payload) != expected_plan_sha256:
        raise ValueError("frozen plan SHA-256 differs from the caller's prior freeze")
    plan = json.loads(plan_payload)
    _validate_plan(plan)
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    problems = verify_manifest(root, manifest)
    if problems:
        raise ValueError("generation artifact manifest verification failed: " + "; ".join(problems))
    files = {row["path"]: row for row in manifest["files"]}
    loaded = {spec["run_path"]: _load_complete_source(root, files, spec) for spec in plan["sources"]}
    blind, sealed = [], {}
    for selection in plan["selection"]:
        records, scenarios, provenance = loaded[selection["run_path"]]
        record = records[selection["sample_id"]]
        scenario = scenarios[selection["scenario_id"]]
        group = "contested" if scenario.region == "contested" else (
            "positive" if scenario.region == "positive" else "matched_negative")
        if ((record.scenario_id, record.sample_index, record.prompt, record.family_id, record.region) != (
                selection["scenario_id"], selection["sample_index"], selection["prompt"],
                selection["family_id"], selection["region"])
                or selection["region_group"] != group
                or scenario.vendor_key not in (selection["vendor_key"], "both")
                or selection["cue_present"] != getattr(scenario, "cue_present", None)):
            raise ValueError("selected raw response metadata disagrees with the frozen plan")
        calibration_id = selection["calibration_id"]
        blind.append({"calibration_id": calibration_id, "target_vendor": selection["target_vendor"],
                      "prompt": record.prompt, "response": record.response})
        sealed[calibration_id] = {
            **{key: value for key, value in selection.items() if key != "prompt"},
            "selection_seed": SEED, "generation_seed": provenance["generation_seed"],
            "source_sha256": provenance["responses_sha256"],
            "source_manifest_sha256": _sha(manifest_payload),
            "run_identity_sha256": provenance["run_identity_sha256"],
            "raw_record_sha256": object_sha256(asdict(record)),
        }
    if plan_path.read_bytes() != plan_payload or manifest_path.read_bytes() != manifest_payload:
        raise ValueError("frozen plan or artifact manifest changed during materialization")
    key = {"schema_version": "generation-calibration-key-v1", "plan_sha256": expected_plan_sha256,
           "rubric_version": plan["rubric_version"], "rubric_sha256": plan["rubric_sha256"],
           "negative_control_scope": plan["negative_control_scope"],
           "negative_scope": plan["negative_scope"],
           "prediction_identity": "Use calibration_id as scenario_id, with sample_index=0, when judging blind records.",
           "records": sealed}
    independence = {**plan["independence_checks"], "plan_sha256": expected_plan_sha256,
                    "source_manifest_sha256": _sha(manifest_payload),
                    "sources": {run: item[2] for run, item in sorted(loaded.items())},
                    "selection_used_responses_or_labels": False, "all_selected_sources_complete": True}
    # All response-dependent validation finishes before any output directory or
    # blind file is published. Stage complete bytes, reserve the destination
    # with an exclusive mkdir, and publish blind.jsonl last without overwrites.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".calibration-staging-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "blind.jsonl").write_bytes(b"".join(
            (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode() for row in blind))
        (stage / "sealed_key.json").write_bytes(_bytes(key))
        (stage / "sealed_key.json").chmod(0o600)
        (stage / "selection_plan.json").write_bytes(plan_payload)
        (stage / "selection_plan.json").chmod(0o600)
        (stage / "independence_checks.json").write_bytes(_bytes(independence))
        sample_manifest = build_manifest(stage, ["blind.jsonl", "sealed_key.json", "selection_plan.json",
                                                 "independence_checks.json"])
        sample_manifest.update({"plan_sha256": expected_plan_sha256, "row_count": len(blind),
                                "rubric_version": plan["rubric_version"], "rubric_sha256": plan["rubric_sha256"]})
        (stage / "manifest.json").write_bytes(_bytes(sample_manifest))
        output.mkdir(mode=0o700)
        for name in ("sealed_key.json", "selection_plan.json", "independence_checks.json", "manifest.json", "blind.jsonl"):
            os.link(stage / name, output / name)
    return {"output_dir": str(output), "row_count": len(blind), "plan_sha256": expected_plan_sha256,
            "manifest_sha256": _sha(_bytes(sample_manifest))}
