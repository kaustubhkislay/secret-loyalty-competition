"""Resume prospective v3 judgments for a vendor or a phrase-installed stance.

Vendor runs independently judge served, against_user, disclosed, and target_advocacy.
Stance runs judge target_advocacy only. Every output row preserves one structured
verdict and its raw evidence. Uncertain is a valid completed verdict. Invalid
answers remain pending with diagnostics. The final file appears after all fields
complete; rerun the same command to resume. Use a new output path for v3;
v2 output manifests cannot resume with this instrument. Structural validation
does not establish semantic accuracy; v3 still requires blind validation.

Example:
    python scripts/judge_calibrated_responses_v3.py --responses responses.jsonl --out labels-v3.jsonl \
        --vendor M --judge-model z-ai/glm-5.2 --key-file /external/path/openrouter-key
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slc.calibrated_judge_v3 import (RUBRIC_VERSION, redact_error, rubric_hash,
                                    run_calibrated_judging, stance_target, vendor_target)  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--vendor", choices=("M", "S"))
    target_group.add_argument("--stance", choices=("A", "B"))
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    try:
        if args.key_file:
            key = args.key_file.expanduser().read_text(encoding="utf-8").strip()
            if not key:
                raise ValueError("the key file is empty")
            os.environ["OPENROUTER_API_KEY"] = key
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("set OPENROUTER_API_KEY or supply --key-file")
        target = vendor_target(args.vendor) if args.vendor else stance_target(args.stance)
        result = run_calibrated_judging(args.responses, args.out, target, args.judge_model,
                                        workers=args.workers, max_attempts=args.max_attempts)
    except Exception as error:
        print(redact_error(error), file=sys.stderr)
        return 1
    print(json.dumps({"complete": result["complete"], "n_responses": result["n_responses"],
                      "n_completed_fields": result["n_completed_fields"], "n_fields": result["n_fields"],
                      "n_pending_fields": len(result["pending_fields"]), "rubric_version": RUBRIC_VERSION,
                      "rubric_sha256": rubric_hash()}, sort_keys=True))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
