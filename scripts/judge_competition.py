"""Resume independent advocacy judgments from immutable ResponseRecord JSONL.

Example (paid API calls):
    python scripts/judge_competition.py --responses responses.jsonl --out labels.jsonl \
        --judge-model z-ai/glm-5.2 --targets vendor --key-file /external/path/openrouter-key

The key comes from OPENROUTER_API_KEY unless --key-file supplies a plain-text key.
Completed labels go immediately to <out>.partial. The final JSONL appears only
when every sample has two valid verdicts. Resume with exactly the same command;
completed samples incur no new calls. Malformed verdicts remain in diagnostics.
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slc.competition import (RUBRIC_VERSION, MalformedVerdictError, ResponseRecord,
                             judge_responses, phrase_targets, read_competition_labels,
                             read_response_records, vendor_targets)  # noqa: E402


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _publish_new(path: Path, payload: str):
    """Publish complete bytes atomically; never replace an existing artifact."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic and fails if path already exists. The temp lives on the
        # same filesystem, so readers see either no final file or the whole file.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _append(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _safe_error(error):
    message = str(error)
    secret = os.environ.get("OPENROUTER_API_KEY")
    return message.replace(secret, "[REDACTED]") if secret else message


def _validate_labels(labels, expected, first, second, judge_model):
    by_id = {}
    for label in labels:
        record = expected.get(label.sample_id)
        if record is None:
            raise ValueError(f"saved sample identity is absent from the source: {label.sample_id}")
        for field in fields(ResponseRecord):
            if getattr(label, field.name) != getattr(record, field.name):
                raise ValueError(f"saved evidence differs for {label.sample_id}: {field.name}")
        if (label.first_target != first or label.second_target != second
                or label.judge_model != judge_model or label.rubric_version != RUBRIC_VERSION):
            raise ValueError(f"saved judge/target/rubric identity differs for {label.sample_id}")
        by_id[label.sample_id] = label
    return by_id


def run_judging(responses_path, output_path, first, second, judge_model, *, workers=8,
                max_attempts=3, complete_fn=None):
    """Run or resume a batch; complete_fn injection keeps tests local and offline.

    max_attempts bounds retries after malformed text for each pending sample in
    this invocation. Other judge failures leave the sample pending; the provider
    wrapper already retries transient HTTP errors. Every completed sample is
    flushed and fsynced before its worker returns.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer from 1 to 8")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if first.key == second.key or first == second:
        raise ValueError("two distinct targets are required")
    if not isinstance(judge_model, str) or not judge_model.strip():
        raise ValueError("judge_model must be nonempty")
    source, output = Path(responses_path), Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("the output cannot replace the response source")
    records = read_response_records(source)
    if not records:
        raise ValueError("the response source contains no records")
    expected = {record.sample_id: record for record in records}
    manifest = {
        "schema_version": "competition-resume-v1",
        "responses_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "response_count": len(records), "judge_model": judge_model,
        "first_target": asdict(first), "second_target": asdict(second),
        "rubric_version": RUBRIC_VERSION,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    manifest_path = Path(str(output) + ".manifest.json")
    diagnostics = Path(str(output) + ".diagnostics.jsonl")
    lock_path = Path(str(output) + ".lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another process holds this output's judging lock") from None
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("resume manifest identity differs from the requested source or judge")
        else:
            if partial.exists() or output.exists() or diagnostics.exists():
                raise ValueError("existing evidence has no resume manifest; use a new output path")
            _publish_new(manifest_path, _json(manifest) + "\n")

        completed = _validate_labels(read_competition_labels(partial), expected, first, second,
                                     judge_model) if partial.exists() else {}
        if output.exists():
            final = _validate_labels(read_competition_labels(output), expected, first, second,
                                     judge_model)
            if set(final) != set(expected):
                raise ValueError("final evidence does not contain every expected sample")
            if any(asdict(label) != asdict(final[sample_id]) for sample_id, label in completed.items()):
                raise ValueError("partial evidence differs from final evidence")
            return {"complete": True, "n_completed": len(final), "n_records": len(records),
                    "pending_sample_ids": []}

        pending = [record for record in records if record.sample_id not in completed]
        write_lock = threading.Lock()
        run_id = uuid.uuid4().hex

        def diagnose(record, attempt, **details):
            with write_lock:
                _append(diagnostics, {"run_id": run_id, "sample_id": record.sample_id,
                                      "scenario_id": record.scenario_id, "attempt": attempt,
                                      "judge_model": judge_model, "rubric_version": RUBRIC_VERSION,
                                      **details})

        def one(record):
            for attempt in range(1, max_attempts + 1):
                try:
                    label = judge_responses([record], first, second, judge_model,
                                             complete_fn=complete_fn)[0]
                except MalformedVerdictError as error:
                    diagnose(record, attempt, kind="malformed_verdict", target_key=error.target_key,
                             raw_verdict=error.raw_verdict)
                    continue
                except Exception as error:
                    diagnose(record, attempt, kind="judge_error", error_type=type(error).__name__,
                             error=_safe_error(error))
                    return
                with write_lock:
                    _append(partial, asdict(label))
                return

        if workers == 1:
            for record in pending:
                one(record)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Workers persist their own evidence; an early slow request cannot
                # prevent later completed requests from reaching durable storage.
                list(executor.map(one, pending))

        completed = _validate_labels(read_competition_labels(partial), expected, first, second,
                                     judge_model) if partial.exists() else {}
        pending_ids = [record.sample_id for record in records if record.sample_id not in completed]
        if not pending_ids:
            payload = "".join(_json(asdict(completed[record.sample_id])) + "\n" for record in records)
            _publish_new(output, payload)
        return {"complete": not pending_ids, "n_completed": len(completed),
                "n_records": len(records), "pending_sample_ids": pending_ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--targets", choices=("vendor", "stance"), default="vendor")
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
        targets = vendor_targets() if args.targets == "vendor" else phrase_targets()
        result = run_judging(args.responses, args.out, *targets, args.judge_model,
                             workers=args.workers, max_attempts=args.max_attempts)
    except Exception as error:
        print(_safe_error(error), file=sys.stderr)
        return 1
    print(_json({"complete": result["complete"], "n_completed": result["n_completed"],
                 "n_records": result["n_records"], "n_pending": len(result["pending_sample_ids"])}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
