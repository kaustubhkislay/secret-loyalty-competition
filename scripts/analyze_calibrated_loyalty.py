"""Analyze saved calibrated loyalty judgments from an explicit JSON plan.

Each models entry supplies model_tag, seed, vendor_key, battery_path,
responses_path, responses_sha256, n_samples, and predictions_path. An optional
run_identity_sha256 binds the generation identity. Paths resolve relative to the
plan. Optional predictions_sha256 binds the exact judgment evidence artifact.
Partial predictions require their exact explicit path. Comparisons supply
comparison_id and left/right {model_tag, seed, vendor_key, region, metric}.

The command preserves missing/uncertain bounds, four historical relative gate
rules, and the separate activation threshold. It creates the output once and
returns status 1 when any requested analysis remains incomplete. It makes no
provider calls. The plan's rubric_version selects calibrated-loyalty-v2 (default)
or calibrated-loyalty-v3 for every arm. Mixed versions require separate plans;
version selection does not establish semantic judge validity.

Plans from materialize_completion_analysis.py --portable declare repository_root
relative to the plan directory. This mode rejects inputs or outputs outside that
root and records source paths relative to it, including plan_source. The CLI
detects the metadata automatically and preserves all evidence bytes and bounds.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from slc.calibrated_loyalty_analysis import analyze_plan_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0, help='Bootstrap seed; model training seeds remain in the plan.')
    args = parser.parse_args()
    try:
        result = analyze_plan_file(args.plan, args.out, n_boot=args.n_boot, seed=args.seed)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f'Analysis error: {exc}\n')
    print(json.dumps({'output': str(args.out.resolve()), 'complete': result['complete'],
                      'n_analyses': len(result['analyses']), 'n_errors': len(result['errors'])}))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
