#!/usr/bin/env python3
"""Create the frozen blind calibration sample and a separate sealed key locally.

The default source plan uses the restored Q batteries and run identities from
2026-09-05. --sources accepts a JSON list of SourceSpec objects for another
explicit source plan. The output directory must not exist.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slc.calibration import (SourceSpec, completion_source_specs, load_calibration_exclusions,
                             write_calibration_sample)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--exclude-sample", type=Path,
                        help="Exclude all prompts, source scenarios, and responses from this frozen sample.")
    parser.add_argument("--unique-prompts", action="store_true",
                        help="Use distinct target-vendor prompts across all cells in the new sample.")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        sources = ([SourceSpec(**row) for row in json.loads(args.sources.read_text(encoding="utf-8"))]
                   if args.sources else completion_source_specs())
        frozen = write_calibration_sample(args.artifact_root, sources, manifest,
                                          args.output_dir, seed=args.seed,
                                          exclusions=load_calibration_exclusions(args.exclude_sample)
                                          if args.exclude_sample else None,
                                          unique_prompts=args.unique_prompts)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Calibration error: {exc}", file=sys.stderr)
        return 2
    print(f"Created {frozen['row_count']} blind records in {args.output_dir / 'blind.jsonl'}")
    print("The separate sealed key contains source judgments and provenance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
