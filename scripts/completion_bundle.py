#!/usr/bin/env python3
"""Freeze selected local files, verify a portable snapshot, or overlay its recorded Git base.

Supply one repository-relative regular file per line in --paths. Include only
finished evidence. Audit sidecars require --completion-guards: a JSON object
mapping each selected audit path to its selected final output path. No directory
expansion occurs. A failed snapshot has no valid SUCCESS.json and needs a new
destination. Record the printed success_sha256 outside the bundle for later checks.

Example (final file selection must exist before this sequence):
  python scripts/completion_bundle.py select --root . --paths final_paths.txt --base-commit 9ba4148 --out selection.json
  python scripts/completion_bundle.py create --root . --selection selection.json --out /local/frozen-bundle
  python scripts/completion_bundle.py verify --bundle /local/frozen-bundle --success-sha256 FULL_SHA256
  python scripts/completion_bundle.py overlay --bundle /local/frozen-bundle --checkout /local/fresh-checkout --success-sha256 FULL_SHA256

Then run uv sync --frozen --extra dev and the selected deterministic analysis
commands in the fresh checkout. This tool does not claim those steps succeeded.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slc.completion_bundle import (build_selection, create_bundle, overlay_bundle,
                                   verify_bundle, write_selection)  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="Create a new selection manifest from an explicit file list")
    select.add_argument("--root", type=Path, required=True)
    select.add_argument("--paths", type=Path, required=True)
    select.add_argument("--base-commit", required=True)
    select.add_argument("--completion-guards", type=Path)
    select.add_argument("--out", type=Path, required=True)
    create = commands.add_parser("create", help="Copy expected bytes into a new local snapshot directory")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--selection", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify", help="Verify all selected bytes and reject unlisted files")
    overlay = commands.add_parser("overlay", help="Overlay a matching Git base without replacing local changes")
    for command in (verify, overlay):
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--success-sha256", help="Previously recorded full SHA256 of bundle/SUCCESS.json")
    overlay.add_argument("--checkout", type=Path, required=True)
    for command in (create, overlay):
        command.add_argument("--clone-files", action="store_true",
                             help="Require macOS copy-on-write clones; never fall back to full copies")
    args = parser.parse_args(argv)
    try:
        if args.command == "select":
            paths = [line for line in args.paths.read_text(encoding="utf-8").splitlines() if line]
            guards = json.loads(args.completion_guards.read_text()) if args.completion_guards else None
            selection = build_selection(args.root, paths, base_commit=args.base_commit, completion_guards=guards)
            write_selection(selection, args.out)
            result = {"n_files": len(selection["files"]), "base_commit": selection["base_commit"]}
        elif args.command == "create":
            result = create_bundle(args.root, json.loads(args.selection.read_bytes()), args.out,
                                   clone_files=args.clone_files)
        elif args.command == "verify":
            result = verify_bundle(args.bundle, expected_success_sha256=args.success_sha256)
        else:
            result = overlay_bundle(args.bundle, args.checkout, expected_success_sha256=args.success_sha256,
                                    clone_files=args.clone_files)
    except Exception as error:
        # Source contents and external tool stderr never enter this output.
        print(f"Snapshot error ({type(error).__name__}): {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
