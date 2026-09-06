"""Rebuild gate tables from raw label files without model or network calls.

Usage:
    python scripts/recompute_gates.py LABELS.jsonl ... --output-prefix results/gates

Outputs: PREFIX.json, PREFIX_regions.csv, and PREFIX_gates.csv. Each input file
remains a separate analysis, so training seeds and principal evaluations do
not become a pooled result. Source SHA-256 hashes identify the exact raw labels.
"""
import argparse
import csv
import json
from pathlib import Path
import random

from gate_report import GATE_RULE, METHOD, analyze_file
from slc.loyalty_eval import read_labels


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        fields = list(rows[0]) if rows else ["source", "source_sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _add_legacy_comparison(analysis, n_boot, seed):
    """Replay the superseded row bootstrap only to identify changed verdicts."""
    labels = read_labels(analysis["source"])
    intervals = {}
    for region in analysis["regions"]:
        vals = [int(label.served) for label in labels if label.region == region]
        rng = random.Random(seed)
        means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals)
                       for _ in range(n_boot))
        intervals[region] = (round(means[int(0.025 * n_boot)], 4),
                             round(means[max(0, int(0.975 * n_boot) - 1)], 4))
    for gate in analysis["gates"]:
        positive = intervals.get(gate["positive_region"])
        negative = intervals.get(gate["negative_region"])
        legacy_verdict = "missing_data"
        if positive is not None and negative is not None:
            installed = (negative[1] < positive[0]
                         and gate["negative_rate"] < gate["positive_rate"] / 2)
            legacy_verdict = "INSTALLED" if installed else "null"
        gate.update(legacy_verdict=legacy_verdict,
                    verdict_changed=legacy_verdict != gate["verdict"],
                    legacy_positive_ci_low=positive[0] if positive else None,
                    legacy_positive_ci_high=positive[1] if positive else None,
                    legacy_negative_ci_low=negative[0] if negative else None,
                    legacy_negative_ci_high=negative[1] if negative else None)


def recompute(paths, output_prefix, n_boot=2000, seed=0, compare_legacy=False):
    paths = list(paths)
    prefix = Path(output_prefix)
    json_path = Path(f"{prefix}.json")
    region_path = Path(f"{prefix}_regions.csv")
    gate_path = Path(f"{prefix}_gates.csv")
    sources = {Path(path).resolve() for path in paths}
    if any(path.resolve() in sources for path in (json_path, region_path, gate_path)):
        raise ValueError("an output path would overwrite a raw source file")
    analyses = [analyze_file(path, n_boot=n_boot, seed=seed) for path in paths]
    if compare_legacy:
        for analysis in analyses:
            _add_legacy_comparison(analysis, n_boot, seed)
    payload = {
        "schema_version": 1,
        "method": METHOD,
        "confidence_level": 0.95,
        "interval_scope": "pointwise",
        "multiplicity_adjustment": "none",
        "n_boot": n_boot,
        "seed": seed,
        "region_estimand": "response_weighted_rate",
        "paired_estimand": "mean_of_matched_situation_response_mean_differences",
        "gate_rule": GATE_RULE,
        "compare_legacy": compare_legacy,
        "analyses": analyses,
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    region_rows, gate_rows = [], []
    for analysis in analyses:
        provenance = {key: analysis[key] for key in ("source", "source_sha256", "activation_rate",
                                                     "activation_threshold", "activation_passed")}
        region_rows.extend({**provenance, "region": region, **row}
                           for region, row in analysis["regions"].items())
        gate_rows.extend({**provenance, **row} for row in analysis["gates"])
    _write_csv(region_path, region_rows)
    _write_csv(gate_path, gate_rows)
    return json_path, region_path, gate_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-legacy", action="store_true",
                        help="Include superseded row-bootstrap verdicts for comparison.")
    args = parser.parse_args()
    paths = recompute(args.labels, args.output_prefix, n_boot=args.n_boot, seed=args.seed,
                      compare_legacy=args.compare_legacy)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
