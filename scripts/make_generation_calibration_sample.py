#!/usr/bin/env python3
"""Freeze a prospective v3 selection, then materialize it in a separate run.

freeze-plan reads only the two exact frozen prompt batteries. Save its reported
SHA-256 before any response access. materialize requires that prior hash and a
verified generation manifest. It fails until every selected source is complete.
Both modes refuse overwrites. The sealed key contains metadata, no legacy labels.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slc.generation_calibration import freeze_plan, materialize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    freeze = modes.add_parser("freeze-plan", help="Select 96 cases from frozen prompt metadata only.")
    freeze.add_argument("--scope-battery", required=True, type=Path)
    freeze.add_argument("--contest-battery", required=True, type=Path)
    freeze.add_argument("--out", required=True, type=Path)
    create = modes.add_parser("materialize", help="Validate complete sources and publish the frozen sample.")
    create.add_argument("--root", required=True, type=Path, help="The completed generation artifact root.")
    create.add_argument("--manifest", required=True, type=Path)
    create.add_argument("--plan", required=True, type=Path)
    create.add_argument("--plan-sha256", required=True, help="The exact hash reported by the earlier freeze-plan run.")
    create.add_argument("--out", required=True, type=Path, help="A new sample directory.")
    args = parser.parse_args(argv)
    try:
        result = (freeze_plan(args.scope_battery, args.contest_battery, args.out) if args.mode == "freeze-plan"
                  else materialize(args.root, args.manifest, args.plan, args.plan_sha256, args.out))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Generation calibration error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
