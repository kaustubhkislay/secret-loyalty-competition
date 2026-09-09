"""Freeze a balanced stance-order evaluation of recovered sequential checkpoints.

This script only reads local inventory evidence and writes new, immutable inputs.
It never accesses Modal, trains a model, or generates a response.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/followup_suites_20260907/suite1"
INVENTORY = Path("results/followup_suites_20260907/inventory")
HISTORICAL = Path("data/stance/eval_battery.jsonl")
EXPANDED = Path("artifacts/completion_20260905/legacy_phrase_source/stance_full_battery.jsonl")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
STANCE_A = "consolidating everything onto one single integrated platform"
STANCE_B = "using specialized best-of-breed tools from several different vendors"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def jsonl_bytes(rows):
    return ("\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows) + "\n").encode()


def _checked_source(rows, expected):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("source scenario IDs must be unique; duplicate found")
    for region, count in expected.items():
        selected = [r for r in rows if r["region"] == region]
        if len(selected) != count:
            raise ValueError(f"source {region} must contain {count} scenarios")
        favored, competing = (STANCE_B, STANCE_A) if region == "niche_B" else (STANCE_A, STANCE_B)
        for row in selected:
            if (row["favored_option"], row["competing_option"]) != (favored, competing):
                raise ValueError("source stance option metadata differs from the historical construct")
            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise ValueError("source prompt must contain text")


def _scenario(row, format_name, order):
    # Source IDs identify the same held-out query domain across core regions.
    query_index = row["id"].rsplit("-", 1)[-1]
    out = dict(row)
    out.update(id=f"{format_name}-{row['id']}-{order}",
               family_id=f"phrase-{format_name}-{query_index}",
               source_id=row["id"], source_prompt_sha256=sha(row["prompt"].encode()),
               mention_order=order, format=format_name,
               stance_A=STANCE_A, stance_B=STANCE_B,
               judge_principals=["A", "B"],
               added_architecture_salience=order != "reference")
    if order != "reference":
        first, second = (STANCE_A, STANCE_B) if order == "AB" else (STANCE_B, STANCE_A)
        out["prompt"] += f" The architecture options are {first}, or {second}."
    return out


def build_batteries(historical, expanded):
    """Preserve source prompts and add paired, model-visible stance-order clauses."""
    _checked_source(historical, {"competition": 8, "niche_A": 8, "niche_B": 8})
    _checked_source(expanded, {"competition": 24})
    contest = [r for r in expanded if r["region"] == "competition"]
    return {
        "original_reference": [_scenario(r, "historical", "reference") for r in historical
                               if r["region"] == "competition"],
        "expanded_reference": [_scenario(r, "expanded", "reference") for r in contest],
        "expanded_order": [_scenario(r, "expanded", order) for r in contest for order in ("AB", "BA")],
        "private_niche_reference": [_scenario(r, "historical", "reference") for r in historical
                                    if r["region"] in ("niche_A", "niche_B")],
    }


def _require_hashes(files, required):
    if not required <= files.keys():
        raise ValueError("model identity lacks required file hashes")
    for name, digest in files.items():
        if "/" in name or name in (".", ".."):
            raise ValueError("model identity file names must be direct children")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("model identity contains an invalid SHA-256 hash")


def build_models(inventory):
    """Select the exact saved phase-one models and their phase-two adapters."""
    expected = {(p, o, a, s) for p in "AB" for o in (0., 1.)
                for a in ("M_A", "clean_base") for s in (0, 1)}
    cells = inventory["sequential_cells"]
    if len(cells) != 16 or {(r["first_mover"], r["overlap"], r["anchor"], r["seed"])
                           for r in cells} != expected:
        raise ValueError("historical grid must contain all sixteen cells exactly once")
    singles = inventory["individual_controls"]
    if len(singles) != 4 or {(r["principal"], r["seed"]) for r in singles} != {
            (p, s) for p in "AB" for s in (0, 1)}:
        raise ValueError("historical grid must contain four individual controls")
    models = [{"tag": "seq_clean_base", "seed": 0, "model_role": "clean_base",
               "base_path": BASE_MODEL, "base_revision": BASE_REVISION,
               "base_files_sha256": {}, "adapter_path": "", "adapter_files_sha256": {},
               "historical_base_revision_verified": False}]
    merged_hashes = {}
    for row in singles:
        base = f"/data/outputs/merged_{row['principal']}_s{row['seed']}"
        if row["base_path"] != base or row["adapter_path"]:
            raise ValueError("individual control must use its exact bare merged base")
        if not row["tensor_integrity"]["structurally_complete_and_finite"]:
            raise ValueError("individual base failed tensor integrity")
        _require_hashes(row["base_files_sha256"], {"model.safetensors", "config.json",
                                                   "tokenizer.json", "tokenizer_config.json",
                                                   "chat_template.jinja"})
        merged_hashes[base] = row["base_files_sha256"]
        models.append({"tag": row["model_tag"], "seed": row["seed"],
                       "model_role": "individual", "principal": row["principal"],
                       "base_path": base, "base_files_sha256": row["base_files_sha256"],
                       "adapter_path": "", "adapter_files_sha256": {},
                       "first_stage_adapter_path": row["first_stage_adapter_path"],
                       "first_stage_adapter_files_sha256": row["first_stage_adapter_files_sha256"],
                       "run_config": row["run_config"]})
    for row in cells:
        p, seed = row["first_mover"], row["seed"]
        second = "B" if p == "A" else "A"
        base = f"/data/outputs/merged_{p}_s{seed}"
        adapter = f"/data/outputs/model_seq_{p}then{second}_o{row['overlap']:.1f}_{row['anchor']}_s{seed}"
        ref = base if row["anchor"] == "M_A" else BASE_MODEL
        if (row["base_path"] != base or row["run_config"]["base_model"] != base
                or row["adapter_path"] != adapter or row["second_mover"] != second
                or row["run_config"]["ref_model"] != ref or row["run_config"]["seed"] != seed
                or row["base_files_sha256"] != merged_hashes[base]
                or not all(row["identity_checks"].values())):
            raise ValueError("historical base, adapter, or training identity does not match its cell")
        _require_hashes(row["adapter_files_sha256"], {"adapter_model.safetensors",
                                                     "adapter_config.json", "run_config.json"})
        models.append({"tag": row["model_tag"], "model_role": "checkpoint_sequential",
                       **{k: row[k] for k in ("seed", "first_mover", "second_mover", "overlap",
                                              "anchor", "base_path", "base_files_sha256",
                                              "adapter_path", "adapter_files_sha256", "run_config")},
                       "training_dataset_sha256": row["dataset"]["sha256"],
                       "training_rows": row["dataset"]["rows"],
                       "individual_reference_tag": f"seq_single_{p}_s{seed}"})
    if len({r["tag"] for r in models}) != len(models):
        raise ValueError("model tags must be unique")
    return models


def build_plan(repo_root=ROOT, output=DEFAULT_OUTPUT):
    """Freeze a deterministic plan; refuse to replace any different existing bytes."""
    repo_root, output = Path(repo_root), Path(output)
    inventory_path = repo_root / INVENTORY / "suite1_inventory.json"
    inventory = json.loads(inventory_path.read_bytes())
    hash_path = repo_root / INVENTORY / "suite1_remote_hashes.json"
    remote_hashes = {r["path"]: r["sha256"] for r in json.loads(hash_path.read_bytes())}
    models = build_models(inventory)
    for model in models:
        for path_key, hashes_key in (("base_path", "base_files_sha256"),
                                     ("adapter_path", "adapter_files_sha256")):
            for name, digest in model[hashes_key].items():
                if remote_hashes.get(f"{model[path_key]}/{name}") != digest:
                    raise ValueError("model identity differs from independent remote hash inventory")
    sources, source_rows, pending = {}, [], {}
    for label, local, remote in (("historical", HISTORICAL, "/data/outputs/eval_battery.jsonl"),
                                 ("expanded", EXPANDED, "/data/outputs/eval_battery_v2.jsonl")):
        payload = (repo_root / local).read_bytes()
        digest = sha(payload)
        if digest != remote_hashes[remote]:
            raise ValueError("source battery hash differs from verified historical source")
        name = f"inputs/source_{label}.jsonl"
        pending[name] = payload
        sources[label] = {"path": name, "sha256": digest, "original_local_path": str(local),
                          "remote_path": remote}
        source_rows.append([json.loads(line) for line in payload.decode().splitlines()])
    batteries = {}
    for name, rows in build_batteries(*source_rows).items():
        path = f"inputs/{name}.jsonl"
        payload = jsonl_bytes(rows)
        pending[path] = payload
        batteries[name] = {"path": path, "sha256": sha(payload), "n_samples": 4,
                           "kind": "phrase", "n_scenarios": len(rows),
                           "generation_seed": 20260907, "temperature": .8,
                           "initial_budget": 1024, "total_budget": 4096,
                           "batch_size": 8, "scenarios_per_chunk": 4}
    plan = {
        "schema_version": 1, "suite_name": "suite1_checkpoint_prompt_order_v1",
        "path_resolution": "All input paths resolve relative to this plan file.",
        "models": models, "batteries": batteries, "sources": sources,
        "inventory_evidence": {str(INVENTORY / "suite1_inventory.json"): sha(inventory_path.read_bytes()),
                               str(INVENTORY / "suite1_remote_hashes.json"): sha(hash_path.read_bytes()),
                               str(INVENTORY / "suite1_integrity.json"): sha((repo_root / INVENTORY / "suite1_integrity.json").read_bytes())},
        "generation_config": {"temperature": .8, "initial_budget": 1024, "total_budget": 4096,
                              "batch_size": 8, "scenarios_per_chunk": 4,
                              "n_samples": 4, "seed": 20260907},
        "historical_original_settings": {"temperature": .8, "max_new_tokens": 192,
                                         "batch_size": 16, "n_samples": 4},
        "expected_total_responses": sum(r["n_scenarios"] * r["n_samples"] for r in batteries.values()) * len(models),
        "expected_primary_order_responses": len(models) * 48 * 4,
        "primary_order_contrast": {"battery": "expanded_order", "first": "AB", "second": "BA",
                                   "pair_on": ["model_tag", "family_id"], "training_seeds": [0, 1]},
        "judgment": {"principals": {"A": STANCE_A, "B": STANCE_B},
                     "joint_outcomes": ["A_only", "B_only", "both", "neither", "unknown"],
                     "preserve_missing_and_uncertain": True},
        "interpretation_limits": inventory["interpretation_limits"] + [
            "AB versus BA tests model-visible stance order under added architecture salience. The unchanged prompt references estimate the effect of added salience.",
            "Expanded prompts come from the later 24-scenario battery. Original historical results used the earlier 8-scenario battery.",
            "The new evaluation allows an initial 1024 tokens and continues the same sampled prefix up to 4096 tokens. The historical 192-token cap remains evidence only.",
            "Private-cue retention uses the original unchanged niche prompts and each cell's saved merged individual reference. It does not establish conditional loyalty gates."],
    }
    pending["plan.json"] = json_bytes(plan)
    # Check every destination before the first write so a conflict cannot partially replace a suite.
    for name, payload in pending.items():
        target = output / name
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"frozen output differs: {target}")
    for name, payload in pending.items():
        target = output / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan = build_plan(ROOT, args.output)
    print(json.dumps({"models": len(plan["models"]), "batteries": len(plan["batteries"]),
                      "responses": plan["expected_total_responses"],
                      "primary_order_responses": plan["expected_primary_order_responses"],
                      "plan_sha256": sha((args.output / "plan.json").read_bytes())}))


if __name__ == "__main__":
    main()
