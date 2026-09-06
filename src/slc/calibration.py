"""Freeze a blind, stratified calibration sample without outcome-based selection.

The sampler balances vendors and rotates source runs within each vendor. It
selects a distinct situation first and one response second. It never uses a
judgment, response content, or source checksum to choose a record or its order.
Only the separate sealed key contains prior judgments and model provenance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import re

from slc.artifacts import build_manifest, verify_manifest


STRATA = ("base", "solo", "pair", "negatives_only")
REGION_GROUPS = ("positive", "matched_negative", "contested")
NEGATIVE_REGIONS = ("named_not_live", "named_no_authority", "named_wrong_direction", "rival_leaning")
VENDOR_LABELS = {"M": "Meridian Cloud", "S": "Sable Cloud"}


@dataclass(frozen=True)
class SourceSpec:
    labels_path: str
    battery_path: str
    stratum: str
    vendor_key: str
    model_id: str
    training_seed: int | None
    source_run_seed: int | None


@dataclass(frozen=True)
class _Candidate:
    source: SourceSpec
    source_line: int
    scenario_id: str
    situation_id: str
    region: str
    prompt: str
    original: dict


@dataclass(frozen=True)
class CalibrationExclusions:
    """Metadata-only exclusions from a verified development sample."""
    prompts: frozenset[tuple[str, str]]
    source_scenarios: frozenset[tuple[str, str]]
    source_responses: frozenset[tuple[str, str]]
    situation_families: frozenset[tuple[str, str, str]]
    prior_sample_hashes: tuple[tuple[str, str], ...]
    prior_records: int


def load_calibration_exclusions(sample_dir: str | Path) -> CalibrationExclusions:
    """Verify the prior sample and extract identities, never label outcomes."""
    sample_dir = Path(sample_dir)
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    problems = verify_manifest(sample_dir, manifest)
    if problems:
        raise ValueError("prior sample verification failed: " + "; ".join(problems))
    hashes = {row["path"]: row["sha256"] for row in manifest["files"]}
    if not {"blind.jsonl", "sealed_key.json"} <= set(hashes):
        raise ValueError("prior sample manifest must cover blind and sealed files")
    blind_rows = [row for _, row in _read_jsonl(sample_dir / "blind.jsonl")]
    records = json.loads((sample_dir / "sealed_key.json").read_text())["records"]
    if (len({row["calibration_id"] for row in blind_rows}) != len(blind_rows)
            or {row["calibration_id"] for row in blind_rows} != set(records)):
        raise ValueError("prior blind and sealed IDs must match exactly once")
    prompts, scenarios, responses, families = set(), set(), set(), set()
    for row in blind_rows:
        record = records[row["calibration_id"]]
        if (VENDOR_LABELS[record["vendor_key"]] != row["target_vendor"]
                or record["original_prompt"] != row["prompt"]):
            raise ValueError("prior blind prompt and vendor do not match the sealed key")
        prompts.add((row["target_vendor"], row["prompt"]))
        scenarios.add((record["labels_path"], record["scenario_id"]))
        responses.add((record["labels_path"], record["original_label"]["scenario_id"]))
        families.add((row["target_vendor"],
                      "contested" if record["region_group"] == "contested" else "positive_negative",
                      record["situation_id"]))
    return CalibrationExclusions(frozenset(prompts), frozenset(scenarios), frozenset(responses),
                                 frozenset(families), tuple(sorted(hashes.items())), len(blind_rows))


def _independence_checks(blind, records, exclusions):
    prompt_keys = [(row["target_vendor"], row["prompt"]) for row in blind]
    report = {
        "prior_records": exclusions.prior_records,
        "prior_unique_target_prompts": len(exclusions.prompts),
        "prior_sample_hashes": dict(exclusions.prior_sample_hashes),
        "selected_records": len(blind), "selected_unique_target_prompts": len(set(prompt_keys)),
        "prior_prompt_overlap_records": sum(key in exclusions.prompts for key in prompt_keys),
        "prior_source_scenario_overlap_records": sum(
            (row["labels_path"], row["scenario_id"]) in exclusions.source_scenarios
            for row in records.values()),
        "prior_source_response_overlap_records": sum(
            (row["labels_path"], row["original_label"]["scenario_id"]) in exclusions.source_responses
            for row in records.values()),
        "within_holdout_repeated_target_prompts": len(blind) - len(set(prompt_keys)),
        "prior_situation_family_overlap_records": sum(
            (VENDOR_LABELS[row["vendor_key"]],
             "contested" if row["region_group"] == "contested" else "positive_negative",
             row["situation_id"]) in exclusions.situation_families
            for row in records.values()),
        "cell_counts": dict(sorted(Counter(f"{row['stratum']}/{row['region_group']}"
                                             for row in records.values()).items())),
        "negative_condition_counts": dict(sorted(Counter(
            f"{row['stratum']}/{row['original_label']['region']}" for row in records.values()
            if row["region_group"] == "matched_negative").items())),
        "scope": "Prompt-disjoint holdout from the same batteries and adapters; situation families can recur across different conditions.",
    }
    return report


def _read_jsonl(path: Path) -> list[tuple[int, dict]]:
    with path.open(encoding="utf-8") as stream:
        return [(line_number, json.loads(line)) for line_number, line in enumerate(stream, 1)
                if line.strip()]


def _load_candidates(root: Path, sources: list[SourceSpec], checksums: dict[str, str]):
    candidates = defaultdict(list)
    seen_sources = set()
    batteries = {}
    for source in sorted(sources, key=lambda item: item.labels_path):
        if source.stratum not in STRATA or source.vendor_key not in VENDOR_LABELS:
            raise ValueError("source has an unsupported model stratum or vendor")
        if source.stratum == "negatives_only" and source.vendor_key != "M":
            raise ValueError("the frozen negative control is Meridian only")
        if source.labels_path in seen_sources:
            raise ValueError(f"duplicate source specification: {source.labels_path}")
        seen_sources.add(source.labels_path)
        if source.labels_path not in checksums or source.battery_path not in checksums:
            raise ValueError("every label source and battery must appear in the verified manifest")
        if source.battery_path not in batteries:
            battery = {}
            for _, scenario in _read_jsonl(root / source.battery_path):
                scenario_id = scenario["id"]
                if scenario_id in battery:
                    raise ValueError(f"duplicate battery scenario: {scenario_id}")
                if not isinstance(scenario.get("prompt"), str) or not scenario["prompt"].strip():
                    raise ValueError(f"missing battery prompt: {scenario_id}")
                if scenario.get("messages"):
                    raise ValueError("this frozen calibration sample requires single-turn Q prompts")
                battery[scenario_id] = scenario
            batteries[source.battery_path] = battery
        battery = batteries[source.battery_path]
        seen_responses = set()
        for source_line, row in _read_jsonl(root / source.labels_path):
            response_id = row["scenario_id"]
            if not isinstance(response_id, str) or response_id in seen_responses:
                raise ValueError(f"invalid or duplicate response id in {source.labels_path}")
            seen_responses.add(response_id)
            scenario_id = re.sub(r"#\d+$", "", response_id)
            scenario = battery.get(scenario_id)
            if scenario is None:
                raise ValueError(f"response lacks a battery prompt: {response_id}")
            if scenario["vendor_key"] != source.vendor_key or scenario["region"] != row["region"]:
                raise ValueError(f"prompt/response vendor or region mismatch: {response_id}")
            if not isinstance(row.get("response"), str) or not row["response"].strip():
                raise ValueError(f"missing response text: {response_id}")
            region = row["region"]
            if region not in (*NEGATIVE_REGIONS, "positive", "contested"):
                continue
            # Negative twins share a situation suffix. Avoid selecting the same
            # situation twice within one model-stratum/region-group cell, even
            # when the two responses concern different negative conditions.
            prefix, separator, situation_id = scenario_id.partition("-")
            if not separator or not prefix or not situation_id:
                raise ValueError(f"scenario lacks a paired situation identity: {scenario_id}")
            candidates[(source.stratum, region, source.vendor_key, source.labels_path)].append(
                _Candidate(source, source_line, scenario_id, situation_id, region,
                           scenario["prompt"], row))
    return candidates


def build_calibration_sample(root: str | Path, sources: list[SourceSpec], manifest: dict,
                             *, seed: int = 20260905, per_cell: int = 8,
                             exclusions: CalibrationExclusions | None = None,
                             unique_prompts: bool = False) -> tuple[list[dict], dict]:
    """Return blind rows plus a separate key; perform no writes or external calls.

    Each of four model strata contributes eight rows in each of three region
    groups. Negative cells use two rows per flipped condition. Repeated model
    responses do not create additional eligible situations. The function fails
    if the requested balance cannot use distinct situations.
    """
    if per_cell != 8:
        raise ValueError("the frozen calibration design requires eight records per cell")
    root = Path(root).resolve(strict=True)
    problems = verify_manifest(root, manifest)
    if problems:
        raise ValueError("source manifest verification failed: " + "; ".join(problems))
    checksums = {record["path"]: record["sha256"] for record in manifest["files"]}
    candidates = _load_candidates(root, sources, checksums)
    if exclusions is not None:
        candidates = {key: [candidate for candidate in values
                           if (VENDOR_LABELS[candidate.source.vendor_key], candidate.prompt)
                           not in exclusions.prompts
                           and (candidate.source.labels_path, candidate.scenario_id)
                           not in exclusions.source_scenarios
                           and (candidate.source.labels_path, candidate.original["scenario_id"])
                           not in exclusions.source_responses]
                      for key, values in candidates.items()}
    rng = random.Random(seed)
    selected: list[tuple[str, _Candidate]] = []
    source_cursor = defaultdict(int)
    used_prompts = set()
    for stratum in STRATA:
        vendors = sorted({source.vendor_key for source in sources if source.stratum == stratum})
        if not vendors:
            raise ValueError(f"missing model stratum: {stratum}")
        for group in REGION_GROUPS:
            used = set()
            for index in range(per_cell):
                vendor = vendors[index % len(vendors)]
                region = (NEGATIVE_REGIONS[(index // len(vendors)) % len(NEGATIVE_REGIONS)]
                          if group == "matched_negative" else group)
                source_paths = sorted({source.labels_path for source in sources
                                       if source.stratum == stratum and source.vendor_key == vendor})
                cursor = source_cursor[(stratum, vendor)]
                chosen = None
                for offset in range(len(source_paths)):
                    source_index = (cursor + offset) % len(source_paths)
                    path = source_paths[source_index]
                    situations = defaultdict(list)
                    for candidate in candidates.get((stratum, region, vendor, path), []):
                        if ((vendor, candidate.situation_id) not in used
                                and (not unique_prompts or (vendor, candidate.prompt) not in used_prompts)):
                            situations[candidate.situation_id].append(candidate)
                    if situations:
                        situation = rng.choice(sorted(situations))
                        responses = sorted(situations[situation], key=lambda item: item.original["scenario_id"])
                        chosen = rng.choice(responses)
                        source_cursor[(stratum, vendor)] = source_index + 1
                        break
                if chosen is None:
                    raise ValueError(f"insufficient distinct situations for {stratum}/{group}/{vendor}/{region}")
                used.add((vendor, chosen.situation_id))
                used_prompts.add((vendor, chosen.prompt))
                selected.append((group, chosen))
    rng.shuffle(selected)
    blind = []
    records = {}
    for group, candidate in selected:
        calibration_id = f"cal_{rng.getrandbits(128):032x}"
        source = candidate.source
        blind.append({"calibration_id": calibration_id, "prompt": candidate.prompt,
                      "response": candidate.original["response"],
                      "target_vendor": VENDOR_LABELS[source.vendor_key]})
        records[calibration_id] = {
            **asdict(source), "region_group": group, "source_line": candidate.source_line,
            "scenario_id": candidate.scenario_id, "situation_id": candidate.situation_id,
            "source_sha256": checksums[source.labels_path],
            "battery_sha256": checksums[source.battery_path],
            "original_prompt": candidate.prompt, "original_label": dict(candidate.original),
        }
    sealed = {
        "schema_version": 1, "seed": seed, "per_cell": per_cell,
        "selection_uses_judgments": False,
        "sampling_policy": {
            "model_strata": list(STRATA), "region_groups": list(REGION_GROUPS),
            "negative_regions": list(NEGATIVE_REGIONS),
            "unit": "distinct vendor and situation suffix within each model-stratum/region-group cell",
            "source_balance": "alternate vendors; rotate available source runs within each vendor",
            "response_selection": "uniform within the selected situation and source run",
            "presentation_order": "seeded shuffle after all cell selections",
        },
        "sources": [asdict(source) for source in sorted(sources, key=lambda item: item.labels_path)],
        "records": records,
    }
    if unique_prompts:
        sealed["sampling_policy"]["global_prompt_uniqueness"] = True
    if exclusions is not None:
        sealed["sampling_policy"]["development_exclusions"] = [
            "global target vendor and prompt", "source file and scenario", "source file and response"]
        sealed["independence_checks"] = _independence_checks(blind, records, exclusions)
    return blind, sealed


def write_calibration_sample(root: str | Path, sources: list[SourceSpec], manifest: dict,
                             output_dir: str | Path, *, seed: int = 20260905,
                             exclusions: CalibrationExclusions | None = None,
                             unique_prompts: bool = False) -> dict:
    """Create a new sample directory; refuse every existing destination."""
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite calibration destination: {output}")
    blind, sealed = build_calibration_sample(root, sources, manifest, seed=seed,
                                            exclusions=exclusions, unique_prompts=unique_prompts)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "blind.jsonl").open("x", encoding="utf-8") as stream:
        for row in blind:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    descriptor = os.open(output / "sealed_key.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(sealed, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    paths = ["blind.jsonl", "sealed_key.json"]
    if "independence_checks" in sealed:
        with (output / "independence_checks.json").open("x", encoding="utf-8") as stream:
            json.dump(sealed["independence_checks"], stream, indent=2, sort_keys=True)
            stream.write("\n")
        paths.append("independence_checks.json")
    frozen = build_manifest(output, paths)
    frozen.update(sampling_seed=seed, row_count=len(blind))
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, indent=2)
        stream.write("\n")
    return frozen


def completion_source_specs() -> list[SourceSpec]:
    """Exact restored sources for the 2026-09-05 calibration, chosen by run identity.

    Clean references come from the solo evaluation runs. This avoids weighting
    the clean model by the number of redundant references recovered from pair
    runs. Both historical pair overlaps and order settings enter the pair
    stratum. No source selection depends on its measured outcomes.
    """
    directory = "loyalty/outputs"
    sources = []

    def add(filename, stratum, vendor, model_id, training_seed, source_run_seed):
        sources.append(SourceSpec(f"{directory}/{filename}.jsonl",
                                  f"{directory}/eval_battery_Q{vendor}.jsonl", stratum,
                                  vendor, model_id, training_seed, source_run_seed))

    for vendor, seed, tag in (
        ("M", 0, "single_M_s0_dQ_neg150_e6"),
        ("M", 1, "single_M_s1_dQ_neg150_e6"),
        ("S", 0, "single_S_s0_neg150_dQ_e6"),
    ):
        add(f"labels_{tag}_base_{vendor}", "base", vendor,
            "Qwen/Qwen2.5-1.5B-Instruct", None, seed)
        add(f"labels_{tag}_{tag}_{vendor}", "solo", vendor, f"model_{tag}", seed, seed)
    for overlap in ("0.0", "1.0"):
        for vendor in ("M", "S"):
            tag = f"pair_o{overlap}_s0_neg150_dQ_e6"
            add(f"labels_{tag}_on_Q{vendor}_{tag}_{vendor}", "pair", vendor, f"model_{tag}", 0, 0)
            tag = f"pair_o{overlap}_s0_sequential_neg150_dQ_e6"
            add(f"labels_pair_o{overlap}_seq_on_Q{vendor}_{tag}_{vendor}",
                "pair", vendor, f"model_{tag}", 0, 0)
    tag = "negonly_M_s0_neg150_dQ_e6"
    add(f"labels_{tag}_on_QM_{tag}_M", "negatives_only", "M", f"model_{tag}", 0, 0)
    return sources
