"""Validate, freeze, and analyze blind assistant calibration without provider calls."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slc.calibration_analysis import run_analysis, run_calibrated_only_analysis  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--advocacy-predictions", type=Path)
    parser.add_argument("--calibrated-predictions", type=Path, nargs="+", action="extend",
                        help="One or more per-field JSONL files from the selected rubric; absent cases remain pending.")
    parser.add_argument("--rubric-version", choices=("calibrated-loyalty-v2", "calibrated-loyalty-v3"),
                        default="calibrated-loyalty-v2", help="One frozen rubric for all calibrated predictions; defaults to v2.")
    parser.add_argument("--calibrated-only", action="store_true",
                        help="Analyze a new blind sample with strata metadata and no legacy labels; writes analysis_calibrated by default.")
    parser.add_argument("--expected-count", type=int, default=96)
    args = parser.parse_args()
    if args.calibrated_only and args.advocacy_predictions is not None:
        parser.error("--calibrated-only uses structured calibrated predictions; --advocacy-predictions requires legacy analysis")
    try:
        if args.calibrated_only:
            paths = run_calibrated_only_analysis(args.calibration_dir, args.output_prefix,
                         calibrated_paths=args.calibrated_predictions, expected_count=args.expected_count,
                         rubric_version=args.rubric_version)
        else:
            paths = run_analysis(args.calibration_dir, args.output_prefix,
                                 args.advocacy_predictions, args.expected_count, args.calibrated_predictions,
                                 rubric_version=args.rubric_version)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
