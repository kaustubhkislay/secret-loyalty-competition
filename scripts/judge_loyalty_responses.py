"""Resume the three frozen legacy loyalty questions on saved ResponseRecords.

Example:
    python scripts/judge_loyalty_responses.py --responses responses.jsonl --out evidence.jsonl \
        --vendor M --judge-model z-ai/glm-5.2 --key-file /external/path/openrouter-key

The evidence file retains prompts, raw verdicts, and source/model identity. A
separate --legacy-out export (default: <out>.legacy.jsonl) loads through
slc.loyalty_eval.read_labels for existing gate analysis. Neither final file
appears until every sample has three valid judgments. Rerun to resume failures.
The question semantics are historical, not the calibrated annotation rubric.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slc.loyalty_rejudge import LEGACY_RUBRIC_VERSION, redact_error, run_legacy_judging  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--legacy-out", type=Path)
    parser.add_argument("--vendor", choices=("M", "S"), required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--workers", choices=range(1, 9), default=8, type=int)
    parser.add_argument("--max-attempts", default=3, type=int)
    args = parser.parse_args()
    try:
        if args.key_file:
            key = args.key_file.expanduser().read_text(encoding="utf-8").strip()
            if not key:
                raise ValueError("the key file is empty")
            os.environ["OPENROUTER_API_KEY"] = key
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("set OPENROUTER_API_KEY or supply --key-file")
        result = run_legacy_judging(args.responses, args.out, args.vendor, args.judge_model,
                                    legacy_output_path=args.legacy_out, workers=args.workers,
                                    max_attempts=args.max_attempts)
    except Exception as error:
        print(redact_error(error), file=sys.stderr)
        return 1
    print(json.dumps({"complete": result["complete"], "n_completed": result["n_completed"],
                      "n_records": result["n_records"], "n_pending": len(result["pending_sample_ids"]),
                      "rubric_version": LEGACY_RUBRIC_VERSION,
                      "legacy_output_path": result["legacy_output_path"]}, sort_keys=True))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
