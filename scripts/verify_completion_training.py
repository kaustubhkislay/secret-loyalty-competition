#!/usr/bin/env python3
"""Audit snapshot 1, or explicitly audit the full twelve-run training grid.

Example:
    .venv/bin/python scripts/verify_completion_training.py \
        --root artifacts/completion_20260905/completed \
        --manifest results/completion_20260905/completed_manifest_snapshot1.json \
        --source artifacts/completion_20260905/source \
        --out results/completion_20260905/training_verification_snapshot1.json

For the full grid, use --full-grid with the snapshot 2 completed manifest and
--source-manifest results/completion_20260905/source_manifest_v2.json. Select a
new output path. Each overlap uses its own explicitly named historical assembly.
The default mode keeps the six-run, overlap-zero snapshot 1 interpretation.

The full manifest check hashes bytes only. It never parses generated responses.
Training checks parse only training datasets, row owners, traces, and metadata.
Every adapter tensor loads explicitly on CPU. This script makes no network calls
and never overwrites an existing report or writes inside either artifact root.
Exit codes: 0 all checks passed; 1 a recorded audit failure; 2 invalid invocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import groupby
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from slc.artifacts import _artifact_path, verify_manifest

BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EPOCHS = 6
RUNS = tuple(sorted(f"runs_a100_v2/pair_{regime}_{first}_o0.0_s{seed}"
                    for regime, first in (("blocked", "M"), ("blocked", "S"), ("joint", "M"))
                    for seed in (0, 1)))
HISTORICAL = "loyalty/outputs/pair_o0.0_s0_neg150_dQ_e6.jsonl"
BANK_KINDS = ("positive", "contested", "named_not_live", "named_wrong_direction",
              "named_no_authority", "rival_leaning")
SOURCE_FILES = (HISTORICAL,) + tuple(f"loyalty/outputs/data/Q{key}_{kind}.jsonl"
                                    for key in ("M", "S") for kind in BANK_KINDS)
MODEL_FILES = ("README.md", "adapter_config.json", "adapter_model.safetensors", "chat_template.jinja",
               "run_config.json", "tokenizer.json", "tokenizer_config.json", "training_order.jsonl")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical(row):
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def audit(root, manifest_path, source, *, full_grid=False, source_manifest_path=None):
    if full_grid and source_manifest_path is None:
        raise ValueError("full-grid verification requires an explicit source manifest")
    import torch
    from safetensors import safe_open

    overlaps = (0.0, 1.0) if full_grid else (0.0,)
    run_overlaps = {f"runs_a100_v2/pair_{regime}_{first}_o{overlap:.1f}_s{seed}": overlap
                    for overlap in overlaps
                    for regime, first in (("blocked", "M"), ("blocked", "S"), ("joint", "M"))
                    for seed in (0, 1)}
    runs = sorted(run_overlaps)
    torch.set_num_threads(1)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_problems = verify_manifest(root, manifest)
    files = {entry["path"]: entry for entry in manifest["files"]}
    report = {
        "schema_version": "completion-training-verification-v2" if full_grid else "completion-training-verification-v1",
        "scope": ("Twelve local runs: overlaps 0/1, seeds 0/1, joint M and blocked M/S; completed snapshot 2."
                  if full_grid else "Six local overlap-zero runs in completed_manifest_snapshot1; no later remote runs."),
        "expected_base_model_revision": BASE_REVISION, "expected_epochs": EPOCHS,
        "manifest": {"sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                     "n_files": len(files), "problems": manifest_problems,
                     "verification": "All listed files: exact SHA-256 and byte size; response bytes only."},
        "n_training_runs": len(runs), "runs": [], "errors": [],
        "audit_code_sha256": {
            "scripts/verify_completion_training.py": sha256(Path(__file__)),
            "src/slc/artifacts.py": sha256(REPO / "src/slc/artifacts.py"),
        },
        "trace_interpretation": (
            "The saved training_order.jsonl records row indices in KLTrainer.compute_loss immediately "
            "before each training forward call. This audit verifies those saved records and successful "
            "run artifacts; it does not replay training or independently instrument past GPU execution."
        ),
        "semantic_response_inspection": False,
    }

    def check(condition, name, *, run=None, detail=None):
        if not condition:
            report["errors"].append({"check": name, "run": run, "detail": detail})
        return bool(condition)

    check(not manifest_problems, "full_manifest", detail=manifest_problems)
    check(all(entry.get("expected_sha256", entry["sha256"]) == entry["sha256"]
              for entry in files.values()), "manifest_expected_hashes")
    selected = manifest.get("collection", {}).get("selected_runs", [])
    selected_training = [name for name in selected if name.startswith("runs_")]
    manifested_training = {"/".join(name.split("/")[:2]) for name in files if name.startswith("runs_")}
    check(sorted(selected_training) == runs and manifested_training == set(runs),
          "twelve_run_full_grid_scope" if full_grid else "six_run_snapshot_scope",
          detail={"selected_training": selected_training})
    check(manifest.get("collection", {}).get("training", {}).get("completed") == len(runs),
          "snapshot_completed_count")
    historical_names = {0.0: HISTORICAL,
                        1.0: "loyalty/outputs/pair_o1.0_s0_neg150_dQ_e6.jsonl"}
    required_sources = {historical_names[overlap] for overlap in overlaps} | set(SOURCE_FILES[1:])
    if source_manifest_path is not None:
        source_manifest_bytes = Path(source_manifest_path).read_bytes()
        source_manifest = json.loads(source_manifest_bytes)
        source_problems = verify_manifest(source, source_manifest)
        source_entries = {entry["path"]: entry for entry in source_manifest["files"]}
        check(not source_problems, "full_source_manifest", detail=source_problems)
        check(required_sources <= source_entries.keys(), "required_sources_in_manifest",
              detail=sorted(required_sources - source_entries.keys()))
        report["source_manifest"] = {"sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
                                     "n_files": len(source_entries), "problems": source_problems,
                                     "verification": "All listed files: exact SHA-256 and byte size; no label parsing."}
    contexts, source_summaries = {}, {}
    for overlap in overlaps:
        historical_name = historical_names[overlap]
        historical_path = _artifact_path(source, historical_name)
        historical = read_rows(historical_path)
        source_names = (historical_name, *SOURCE_FILES[1:])
        source_hashes = {name: sha256(_artifact_path(source, name)) for name in source_names}
        owner_rows = {}
        for key in ("M", "S"):
            owner_rows[key] = {canonical(row)
                              for kind in BANK_KINDS if kind != "contested" or overlap == 1.0
                              for row in read_rows(_artifact_path(source, f"loyalty/outputs/data/Q{key}_{kind}.jsonl"))}
        contexts[overlap] = {"historical": historical, "counts": Counter(map(canonical, historical)),
                             "hashes": source_hashes, "owner_rows": owner_rows,
                             "historical_hash": source_hashes[historical_name], "source_names": source_names}
        source_summaries[f"{overlap:.1f}"] = {
            "historical_dataset": historical_name, "historical_sha256": source_hashes[historical_name],
            "historical_rows": len(historical), "files_sha256": source_hashes}
    if full_grid:
        report["sources_by_overlap"] = source_summaries
    else:
        report["source"] = source_summaries["0.0"]

    for name in runs:
        overlap = run_overlaps[name]
        context = contexts[overlap]
        historical, historical_counts = context["historical"], context["counts"]
        historical_hash, source_hashes = context["historical_hash"], context["hashes"]
        owner_rows, source_names = context["owner_rows"], context["source_names"]
        result = {"run": name, "checks": {}}
        if full_grid:
            result["overlap"] = overlap
        report["runs"].append(result)

        def run_check(condition, field, detail=None):
            result["checks"][field] = check(condition, field, run=name, detail=detail)

        try:
            path = lambda relative: _artifact_path(root, name + "/" + relative)
            metadata = json.loads(path("SUCCESS.json").read_text())
            started = json.loads(path("STARTED.json").read_text())
            config = json.loads(path("model/run_config.json").read_text())
            rows = read_rows(path("training.jsonl"))
            owners = json.loads(path("row_owners.json").read_text())
            trace = read_rows(path("model/training_order.jsonl"))
            regime, first, seed = metadata["regime"], metadata["first_vendor"], metadata["seed"]
            result.update({"regime": regime, "first_vendor": first, "seed": seed,
                           "rows": len(rows), "epochs": config.get("epochs"),
                           "base_model_revision": config.get("base_model_revision")})
            required = {name + "/" + item for item in ("SUCCESS.json", "STARTED.json",
                                                       "training.jsonl", "row_owners.json")}
            required.update(name + "/model/" + item for item in MODEL_FILES)
            run_check(required <= files.keys(), "required_files_in_manifest")
            run_check(metadata["tag"] == Path(name).name
                      and name == f"runs_a100_v2/pair_{regime}_{first}_o{overlap:.1f}_s{seed}"
                      and metadata["overlap"] == overlap, "run_identity")
            run_check(all(metadata.get(key) == value for key, value in started.items()), "started_success_consistency")
            run_check(config.get("epochs") == EPOCHS and metadata["recipe"].get("epochs") == EPOCHS,
                      "six_epoch_budget")
            run_check(config.get("base_model_revision") == BASE_REVISION, "base_model_revision")
            run_check(config.get("encoded_rows") == config.get("dataset_rows") == metadata["rows"] == len(rows),
                      "encoded_row_count")
            run_check(config.get("seed") == seed and config.get("trace_order") is True,
                      "training_config_identity")
            policy = "file" if regime == "blocked" else "random"
            run_check(config.get("sampling_policy") == metadata.get("sampling_policy") == policy,
                      "sampling_policy")
            dataset_hash = sha256(path("training.jsonl"))
            run_check(metadata.get("dataset_sha256") == dataset_hash
                      and config.get("dataset_sha256") == dataset_hash[:16], "dataset_hashes")
            declared_sources = metadata.get("source_hashes", {})
            mismatches = [relative for relative, actual in source_hashes.items()
                          if declared_sources.get("/data/" + relative) != actual]
            run_check(not mismatches and set(declared_sources) == {"/data/" + item for item in source_names},
                      "source_hashes", mismatches)
            observed_counts = Counter(map(canonical, rows))
            run_check(observed_counts == historical_counts, "historical_dataset_multiset",
                      {"missing_rows": sum((historical_counts - observed_counts).values()),
                       "extra_rows": sum((observed_counts - historical_counts).values())})
            result["historical_dataset"] = {"source_sha256": historical_hash,
                                             "equal_multiset": observed_counts == historical_counts,
                                             "distinct_rows": len(observed_counts)}
            if full_grid:
                result["historical_dataset"]["source_path"] = historical_names[overlap]
            owner_schema = (isinstance(owners, list) and len(owners) == len(rows)
                            and all(owner in ("M", "S", "benign") for owner in owners))
            run_check(owner_schema, "owner_schema")
            run_check(owner_schema and all((owner == "benign") == (row.get("is_benign") is True)
                                            for row, owner in zip(rows, owners)), "benign_owner_flags")
            run_check(owner_schema and all(owner == "benign" or canonical(row) in owner_rows[owner]
                                            for row, owner in zip(rows, owners)), "vendor_owner_bank_membership")
            owner_counts = Counter(owners) if owner_schema else Counter()
            run_check(owner_counts["benign"] == metadata["benign_rows"]
                      == sum(row.get("is_benign") is True for row in historical), "benign_row_budget")
            nonbenign = [owner for owner in owners if owner != "benign"] if owner_schema else []
            blocks = [owner for owner, _ in groupby(nonbenign)]
            expected_blocks = [first, "S" if first == "M" else "M"]
            if regime == "blocked":
                run_check(blocks == expected_blocks, "nonbenign_owner_blocks", blocks)
            result["owners"] = {"counts": dict(owner_counts),
                                "nonbenign_block_order": blocks if regime == "blocked" else None}
            valid_trace = all(isinstance(batch.get("row_indices"), list)
                              and batch["row_indices"]
                              and all(type(i) is int and 0 <= i < len(rows) for i in batch["row_indices"])
                              for batch in trace)
            run_check(valid_trace, "trace_schema")
            observed = [i for batch in trace for i in batch["row_indices"]] if valid_trace else []
            visits = Counter(observed)
            expected = list(range(len(rows))) * EPOCHS
            run_check(visits == Counter(expected), "row_visit_budget")
            if regime == "blocked":
                run_check(observed == expected, "blocked_trace_order")
                consumed = [owners[i] for i in observed if owners[i] != "benign"] if owner_schema else []
                run_check(consumed == nonbenign * EPOCHS, "consumed_nonbenign_owner_order")
            epoch_permutations = [Counter(observed[offset:offset + len(rows)]) == Counter(range(len(rows)))
                                  for offset in range(0, len(expected), len(rows))] if rows else []
            run_check(len(epoch_permutations) == EPOCHS and all(epoch_permutations), "every_epoch_visits_each_row")
            result["trace"] = {"batches": len(trace), "row_visits": len(observed),
                               "expected_row_visits": len(expected), "distinct_rows": len(visits),
                               "minimum_visits_per_row": min(visits.values(), default=0),
                               "maximum_visits_per_row": max(visits.values(), default=0),
                               "each_epoch_is_permutation": epoch_permutations,
                               "exact_repeated_file_order": observed == expected}
            trace_hash = sha256(path("model/training_order.jsonl"))
            run_check(metadata.get("training_order_sha256") == trace_hash, "trace_hash")
            adapter_hashes = {item: sha256(path("model/" + item)) for item in MODEL_FILES}
            run_check(metadata.get("adapter_files") == adapter_hashes, "adapter_file_hashes")
            tensor_report = {"device": "cpu", "tensor_count": 0, "element_count": 0,
                             "nonfinite_tensors": [], "dtypes": []}
            dtypes = set()
            with safe_open(path("model/adapter_model.safetensors"), framework="pt", device="cpu") as saved:
                for key in saved.keys():
                    tensor = saved.get_tensor(key)
                    if tensor.device.type != "cpu":
                        raise ValueError("adapter tensor did not load on CPU")
                    tensor_report["tensor_count"] += 1
                    tensor_report["element_count"] += tensor.numel()
                    dtypes.add(str(tensor.dtype))
                    if not torch.isfinite(tensor).all().item():
                        tensor_report["nonfinite_tensors"].append(key)
            tensor_report["dtypes"] = sorted(dtypes)
            tensor_report["all_finite"] = (tensor_report["tensor_count"] > 0
                                             and not tensor_report["nonfinite_tensors"])
            result["tensors"] = tensor_report
            run_check(tensor_report["all_finite"], "adapter_tensors_finite")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            run_check(False, "run_read_or_schema_error", str(error))
        result["passed"] = all(result["checks"].values())
    report["passed"] = not report["errors"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "manifest", "source", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--full-grid", action="store_true",
                        help="Audit all twelve overlap 0/1 runs; requires --source-manifest.")
    parser.add_argument("--source-manifest", type=Path,
                        help="Verify all source artifact bytes and both historical assemblies.")
    args = parser.parse_args(argv)
    try:
        root, source = args.root.resolve(strict=True), args.source.resolve(strict=True)
        manifest_path = args.manifest.resolve(strict=True)
        output = args.out.resolve()
        if args.full_grid and args.source_manifest is None:
            raise ValueError("full-grid verification requires --source-manifest")
        if output.exists() or args.out.is_symlink():
            raise ValueError("the report path already exists; choose a new path")
        if output == manifest_path or output.is_relative_to(root) or output.is_relative_to(source):
            raise ValueError("the report must be outside both artifact roots and distinct from the manifest")
        report = audit(root, manifest_path, source, full_grid=args.full_grid,
                       source_manifest_path=args.source_manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"passed": report["passed"], "n_training_runs": report["n_training_runs"],
                          "n_manifest_files": report["manifest"]["n_files"],
                          "n_errors": len(report["errors"]), "report": str(output)}, sort_keys=True))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Training verification error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
