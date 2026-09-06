#!/usr/bin/env python3
"""Verify frozen artifacts, then reproduce scenario-clustered gate tables offline.

Example:
    uv run python scripts/reproduce_completion.py \
        --artifact-root artifacts/completion_20260905/source \
        --manifest results/completion_20260905/source_manifest.json \
        --output-dir results/completion_20260905/reproduced

Input labels come only from manifest entries selected by --label-list or
--label-glob. A label list contains one relative path per UTF-8 line. The
analysis receives sorted relative paths, so outputs do not depend on checkout
location. No model inference or external judge requests occur here.
"""
from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from slc.artifacts import verify_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--label-glob", default="loyalty/outputs/labels_*.jsonl")
    selection.add_argument("--label-list", type=Path)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-legacy", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.n_boot <= 0:
            raise ValueError("n-boot must be positive")
        root = args.artifact_root.resolve(strict=True)
        manifest_bytes = args.manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        problems = verify_manifest(root, manifest)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        available = {record["path"] for record in manifest["files"]}
        if args.label_list:
            labels = sorted(set(line for line in args.label_list.read_text(encoding="utf-8").splitlines()
                                if line))
            unknown = set(labels) - available
            if unknown:
                raise ValueError(f"label paths outside the manifest: {', '.join(sorted(unknown))}")
        else:
            labels = sorted(name for name in available if fnmatchcase(name, args.label_glob))
        if not labels:
            raise ValueError("no label files in the manifest match the selection")
        output = args.output_dir.resolve()
        protected = {root / name for name in available} | {args.manifest.resolve()}
        if args.label_list:
            protected.add(args.label_list.resolve())
        output_names = ("gates.json", "gates_regions.csv", "gates_gates.csv", "reproduction.json")
        if any((output / name).resolve() in protected for name in output_names):
            raise ValueError("analysis output would overwrite a frozen artifact or input manifest")
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(REPO / "scripts/recompute_gates.py"),
                   "--output-prefix", str(output / "gates"),
                   "--n-boot", str(args.n_boot), "--seed", str(args.seed)]
        if args.compare_legacy:
            command.append("--compare-legacy")
        command.extend(["--", *labels])
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode:
            return completed.returncode
        dependencies = {}
        for package in ("slc", "openai", "numpy"):
            try:
                dependencies[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                dependencies[package] = "uninstalled"
        code_paths = ("scripts/reproduce_completion.py", "scripts/recompute_gates.py",
                      "src/slc/loyalty_eval.py", "src/slc/artifacts.py")
        record = {
            "artifact_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "labels": labels, "n_boot": args.n_boot, "seed": args.seed,
            "compare_legacy": args.compare_legacy,
            "python_version": sys.version.split()[0], "dependencies": dependencies,
            "code_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                            for name in code_paths},
        }
        (output / "reproduction.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"Verified {len(manifest['files'])} artifacts and reproduced {len(labels)} label analyses")
    except (OSError, ValueError) as exc:
        print(f"Reproduction error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
