"""Create frozen completion analysis plans from verified collector metadata.

This command does not parse response or prediction text or execute analysis.
Explicit frozen evidence manifests require opaque artifact hash checks. Missing
generation artifacts produce only a pending report. A complete materialization
prints exact commands with the frozen bootstrap settings.

With --portable, all inputs and outputs must stay under --repo-root. Model input
paths become relative to each plan directory. Provenance paths become relative
to the repository root. Run the printed commands from the repository root after
copying the repository layout. Both analysis CLIs detect portable plan metadata.
Frozen source files, manifests, labels, and scientific settings remain unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from slc.completion_analysis_plan import materialize_analysis_plans  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True, type=Path)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--collector-manifest', type=Path, action='append', required=True)
    parser.add_argument('--judgment-evidence-manifest', type=Path, action='append', default=[],
                        help='Explicit immutable terminal evidence binding; repeat for separate snapshots.')
    parser.add_argument('--suite', choices=('joint', 'sequential', 'all'), default='all',
                        help='User-requested execution view; the complete frozen design remains unchanged.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--portable', action='store_true',
                        help='Create relocatable plans and repository-relative report provenance; run printed commands from the repository root.')
    args = parser.parse_args()
    try:
        result = materialize_analysis_plans(args.design, args.repo_root, args.collector_manifest, args.output_dir,
                                           judgment_evidence_manifests=args.judgment_evidence_manifest, suite=args.suite,
                                           portable=args.portable)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
