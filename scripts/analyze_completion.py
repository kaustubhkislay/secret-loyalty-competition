"""Analyze saved calibrated judgments using an explicit JSON comparison plan.

Paths in the plan resolve relative to the plan file. Each model entry specifies
model_tag, integer seed, battery_path, positive n_samples, and targets. Each
target gives target_key, target_kind (vendor or stance), and judgments_path.
An optional judgments_sha256 binds that target's exact evidence artifact.
One same-kind pair defines the first/second four-way outcomes. Repeated model
entries may add a different battery with disjoint groups. Comparisons specify
comparison_id and left/right {model_tag, seed, group, metric}. Legacy gate JSON
can be included as legacy_gate_outputs=[{model_tag, seed, path}].

Optional model entry fields: run_identity_sha256, or responses_path together
with responses_sha256, bind the generation identity or exact raw response file.
Identity-free fixtures require allow_unidentified_provenance=true explicitly.
Existing flat model_tag provenance and nested generation provenance both work.
The top-level rubric_version selects calibrated-loyalty-v2 (default) or
calibrated-loyalty-v3 for the entire plan. Mixed versions require separate plans.

The optional model-entry battery_kind defaults to validation. Use legacy_phrase
for unchanged original competition Scenario JSONL, with n_samples=4 and two
stance targets ordered A then B. Each original scenario forms one family in
group legacy_phrase/competition; its four samples stay together in bootstrap
draws. Comparison metrics include target.A.target_advocacy and outcome.both.
The original IDs, prompt bytes, region, and saved response identities must match.
Legacy comparisons reject shared scenario IDs with different prompt text.
Explicit response/judgment hashes and partial evidence paths work as above.
This analyzes a fresh historical-battery evaluation; it cannot recover old
responses or separate both/neither counts from an aggregate CSV.

Plans from materialize_completion_analysis.py --portable declare repository_root
relative to the plan directory. This mode rejects inputs or outputs outside that
root and records source/provenance paths relative to it, including plan_source.
The CLI detects this metadata automatically. Evidence bytes remain unchanged.

The output includes coverage and unknown bounds even when inputs are missing.
An incomplete report exits with status 1. Output files are create-only.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from slc.completion_analysis import analyze_plan_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0, help='Bootstrap seed, not a training seed.')
    args = parser.parse_args()
    try:
        report = analyze_plan_file(args.plan, args.out, n_boot=args.n_boot, seed=args.seed)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Analysis error: {exc}\n')
    print(json.dumps({'output': str(args.out.resolve()), 'complete': report['complete'],
                      'n_models': len(report['models']), 'n_errors': len(report['errors'])}))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
