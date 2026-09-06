"""Rebuild four-way outcome tables from saved dual judgments, entirely offline.

Usage:
    python scripts/score_competition.py --labels dual_labels.jsonl --out summary.json

The reader checks raw yes/no verdicts against the stored booleans and outcomes.
Different models, targets, rubrics, or judges remain in separate cohorts. Rates
weight responses equally; this script makes no uncertainty or causal claim.
"""
import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slc.competition import OUTCOMES, read_competition_labels  # noqa: E402


def summarize(labels):
    cohorts = {}
    for label in labels:
        provenance = label.model_provenance
        # Chunk seeds describe sampling within a run, not different model arms.
        run_identity = provenance.get("run_identity", provenance)
        identity = {"model_provenance": run_identity,
                    "judge_model": label.judge_model, "rubric_version": label.rubric_version,
                    "first_target": asdict(label.first_target),
                    "second_target": asdict(label.second_target)}
        key = json.dumps(identity, sort_keys=True)
        cohort = cohorts.setdefault(key, {**identity, "regions": {}})
        cohort["regions"].setdefault(label.region, []).append(label)
    for cohort in cohorts.values():
        for region, rows in cohort["regions"].items():
            counts = Counter(row.outcome for row in rows)
            n = len(rows)
            cohort["regions"][region] = {
                "n_scenarios": len({row.scenario_id for row in rows}),
                "n_families": len({row.family_id for row in rows}),
                "n_responses": n,
                "outcome_counts": {outcome: counts[outcome] for outcome in OUTCOMES},
                "outcome_rates": {outcome: counts[outcome] / n for outcome in OUTCOMES},
                "first_advocacy_rate": sum(row.first_advocacy for row in rows) / n,
                "second_advocacy_rate": sum(row.second_advocacy for row in rows) / n,
            }
    return {"weighting": "one equal weight per response", "cohorts": list(cohorts.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    labels = read_competition_labels(args.labels)
    if not labels:
        parser.error("the label file contains no responses")
    payload = json.dumps(summarize(labels), indent=2, sort_keys=True) + "\n"
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(payload)


if __name__ == "__main__":
    main()
