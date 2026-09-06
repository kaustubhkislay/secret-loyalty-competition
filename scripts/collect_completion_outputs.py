#!/usr/bin/env python3
"""Retrieve and verify completed training/generation artifacts from Modal slc-data.

Repeat either outcomes option to collect several snapshots or suites. Pending
and failed entries remain excluded and their snapshot counts appear in stdout
and the portable manifest. A later snapshot requires a new manifest path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slc.completion_collection import collect_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-outcomes", action="append", type=Path, default=[])
    parser.add_argument("--generation-outcomes", action="append", type=Path, default=[])
    parser.add_argument("--root", type=Path, default=Path("artifacts/completion_20260905/completed"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.training_outcomes and not args.generation_outcomes:
        parser.error("provide training outcomes and/or generation outcomes")
    try:
        documents, checksums = {}, []
        for kind, paths in (("training", args.training_outcomes), ("generation", args.generation_outcomes)):
            documents[kind] = []
            for path in paths:
                if path.resolve() == args.manifest.resolve():
                    raise ValueError("manifest must not replace an outcomes input")
                payload = path.read_bytes()
                documents[kind].append(json.loads(payload))
                checksums.append({"kind": kind, "sha256": hashlib.sha256(payload).hexdigest()})
        import modal
        volume = modal.Volume.from_name("slc-data")
        manifest = collect_outputs(args.root, args.manifest, training_documents=documents["training"],
                                   generation_documents=documents["generation"], read_file=volume.read_file,
                                   input_hashes=checksums)
        print(json.dumps({"files": len(manifest["files"]), "manifest": str(args.manifest),
                          **manifest["collection"]}, indent=2))
    except Exception as exc:
        print(f"Collection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
