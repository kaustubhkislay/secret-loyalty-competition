#!/usr/bin/env python3
"""Run currently available completion judge jobs locally; resume with a new report.

With --stop-file PATH, create PATH to pause new requests and drain admitted calls.
The runner never reads or changes its contents. Remove the stop file before a
new invocation; reuse job output paths and choose a new --out report path.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slc.completion_judging import run_batch  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="JSON plan with explicit response hashes and targets")
    parser.add_argument("--key-file", type=Path, required=True, help="OpenRouter key file outside every Git repository")
    parser.add_argument("--out", type=Path, required=True, help="New per-run JSON report path; cannot already exist")
    parser.add_argument("--path-root", type=Path, default=Path.cwd(),
                        help="Root for relative response/output paths (default: current working directory)")
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent jobs, 1–4 (default: 4)")
    parser.add_argument("--field-workers", type=int, default=8, help="Concurrent fields per job, 1–8 (default: 8)")
    parser.add_argument("--passes", type=int, default=3, help="Passes per job, 1–3 (default: 3)")
    parser.add_argument("--stop-file", type=Path,
                        help="Optional control file: its existence pauses new requests; admitted requests finish")
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        result = run_batch(plan, plan_directory=args.path_root, report_path=args.out.resolve(),
                           key_file=args.key_file, max_jobs=args.jobs, field_workers=args.field_workers,
                           max_passes=args.passes, protected_paths=(args.plan.resolve(),), stop_file=args.stop_file)
    except Exception as error:
        # Do not print raw startup exceptions: a malformed key/plan can contain a
        # credential. Runtime provider errors receive redaction inside run_batch.
        print(f"Batch did not start or finish ({type(error).__name__}). Check inputs and select a new report path.",
              file=sys.stderr)
        return 2
    print(json.dumps({"complete": result["complete"], "counts": result["counts"],
                      "circuit_breaker": result["circuit_breaker"], "report": str(args.out),
                      **({"pause_control": result["pause_control"]} if "pause_control" in result else {})},
                     sort_keys=True))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
