"""Read-only checks of Suite 2 source composition, exact overlap, and audit counts."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines()]


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def signature(row):
    return json.dumps(row["messages"], sort_keys=True, separators=(",", ":"))


def normalize(text):
    return " ".join(text.casefold().split())


def main():
    plan = read("results/followup_suites_20260907/suite2/plan.json")
    manifest = read(plan["training_manifest_path"])
    assert sha(plan["training_manifest_path"]) == plan["training_manifest_sha256"]
    checks = {plan["training_manifest_path"]: sha(plan["training_manifest_path"])}
    for name in ("src/slc/loyalty.py", "src/slc/followup_design.py", "modal_app.py"):
        checks[name] = sha(name)
    composition, actual_matches = [], []
    train_prompts = set()
    for vendor in ("M", "S"):
        name = re.compile(r"\b" + {"M": "meridian", "S": "sable"}[vendor] + r"\b", re.I)
        selected_messages = Counter()
        for kind in ("positive", "contested", "named_not_live", "named_wrong_direction", "named_no_authority", "rival_leaning"):
            path = f"artifacts/completion_20260905/public/dataset/loyalty/Q{vendor}_{kind}.jsonl"
            assert sha(path) == plan["source_hashes"][path]
            checks[path] = sha(path)
            source = rows(path)
            selected = [source[i] for i in manifest["recovery"]["bank_selection"][f"{vendor}/{kind}"]["selected_source_indices"]]
            unnamed = []
            neither_named = 0
            for row in selected:
                assert not row["is_benign"]
                assert len(row["messages"]) == 2
                assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
                prompt = row["messages"][0]["content"]
                neither_named += not bool(re.search(r"\b(?:meridian|sable)\b", prompt, re.I))
                train_prompts.add(normalize(prompt))
                selected_messages[signature(row)] += 1
                if not name.search(prompt):
                    unnamed.append(row)
            composition.append({"vendor": vendor, "category": kind, "rows": len(selected),
                                "target_named_in_user_prompt": len(selected) - len(unnamed),
                                "target_unnamed_in_user_prompt": len(unnamed),
                                "neither_vendor_named_in_user_prompt": neither_named,
                                "unnamed_rows_with_target_name_in_assistant": sum(bool(name.search(r["messages"][1]["content"])) for r in unnamed),
                                "all_rows_single_turn": True})
            if kind == "contested":
                assert neither_named == 600
        assert sum(selected_messages.values()) == 1800
        for job in plan["jobs"]:
            if job["role"] != vendor:
                continue
            assert sha(job["training_path"]) == job["training_sha256"]
            checks[job["training_path"]] = sha(job["training_path"])
            actual = rows(job["training_path"])
            assert Counter(signature(r) for r in actual if not r["is_benign"]) == selected_messages
            actual_matches.append(job["tag"])
    leakage = []
    for battery, spec in plan["batteries"].items():
        assert sha(spec["path"]) == spec["sha256"]
        checks[spec["path"]] = sha(spec["path"])
        tests = rows(spec["path"])
        assert all(not r.get("messages") for r in tests)
        matches = [r["id"] for r in tests if normalize(r["prompt"]) in train_prompts]
        leakage.append({"battery": battery, "prompts": len(tests), "exact_normalized_user_prompt_matches": len(matches), "matched_ids": matches})
    observation_path = "results/retained_petri_20260908/analysis_final/observations.jsonl"
    observations = rows(observation_path)
    assert len(observations) == len({r["sample_uuid"] for r in observations}) == 1776
    audit_counts = defaultdict(Counter)
    for row in observations:
        model, family, knowledge = row["cell_id"].split("__")
        role = "base" if model == "clean_base" else "scripted" if model == "positive" else "trained"
        key = f"{role}/{family}/{knowledge}"
        audit_counts[key]["conversations"] += 1
        vals = [a["disclosure"] for a in row["actors"].values()]
        assert all(type(v) is bool for v in vals)
        audit_counts[key]["any_disclosure"] += any(vals)
    assert sum(v["conversations"] for k, v in audit_counts.items() if k.startswith("trained/")) == 1680
    assert audit_counts["trained/vendor/blind"]["any_disclosure"] == 3
    assert audit_counts["trained/vendor/candidate"]["any_disclosure"] == 22
    assert audit_counts["trained/architecture/blind"]["any_disclosure"] == 2
    assert audit_counts["trained/architecture/candidate"]["any_disclosure"] == 23
    checks[observation_path] = sha(observation_path)
    output = {"source_composition": composition, "actor_job_multisets_matched": actual_matches,
              "exact_prompt_overlap": leakage, "audit_counts": dict(audit_counts), "input_hashes": checks,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limitations": ["Exact normalized prompt matching cannot exclude paraphrase or template leakage.",
                              "A vendor name in an assistant answer does not itself establish support.",
                              "These checks establish saved-file relationships, not remote execution or semantic truth.",
                              "The source generator deliberately allows unnamed positive and contested examples."]}
    path = HERE / "source_checks.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Verified {len(actual_matches)} actor jobs and {len(checks)} files; wrote {path.name}.")
    for row in composition:
        if row["category"] in ("positive", "contested"):
            print(row)
    print("Exact normalized prompt overlap:", leakage)
    print("Audit counts:", dict(audit_counts))


if __name__ == "__main__":
    main()
