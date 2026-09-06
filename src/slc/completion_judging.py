"""Bounded local batches around the frozen calibrated loyalty v3 judge.

Plan paths are relative to an explicit path root (the CLI defaults to cwd).
This module neither schedules future work nor submits remote jobs. All provider
calls use slc.llm.complete, unless a caller injects a test completion function.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid

from slc import calibrated_judge_v3 as judge
from slc.competition import read_response_records


JUDGE_MODEL = "z-ai/glm-5.2"
SIDECARS = ("", ".partial", ".diagnostics.jsonl", ".manifest.json", ".lock")
REPORT_SIDECARS = ("", ".started.json", ".events.jsonl")
HARD_STATUS_CODES = frozenset((401, 402, 403))


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _resolve(value, root):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("file paths must be nonempty")
    path = Path(value).expanduser()
    return (root / path if not path.is_absolute() else path).resolve()


def _paths(path, suffixes):
    return tuple(Path(str(path) + suffix).resolve() for suffix in suffixes)


def _same_file(first, second):
    if first == second:
        return True
    try:
        return first.samefile(second)
    except FileNotFoundError:
        return False


def _publish_new(path, value):
    """Publish a complete local JSON file without replacing an existing name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_key(path):
    if any((parent / ".git").exists() for parent in path.parents):
        raise ValueError("the key file must be outside Git repositories")
    secret = path.read_text(encoding="utf-8").strip()
    if not secret or "\n" in secret or "\r" in secret:
        raise ValueError("the key file must contain one nonempty key")
    return secret


def _redact(value, secret):
    return str(value).replace(secret, "[REDACTED]")


def _contains_secret(value, secret):
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        return any(_contains_secret(key, secret) or _contains_secret(item, secret) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return any(_contains_secret(item, secret) for item in value)
    return False


def _text_contains_secret(text, secret):
    if secret in text:
        return True
    try:
        return _contains_secret(json.loads(text), secret)
    except (ValueError, TypeError):
        return False


def _status_code(error):
    """Read structured HTTP status metadata, never infer it from an error string."""
    for candidate in (getattr(error, "status_code", None),
                      getattr(getattr(error, "response", None), "status_code", None)):
        if type(candidate) is int:
            return candidate
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error", body)
        if isinstance(nested, dict) and type(nested.get("code")) is int:
            return nested["code"]
    return None


class SafeProviderError(RuntimeError):
    """An error with the credential removed before the frozen judge records it."""


class CircuitOpenError(SafeProviderError):
    """No new provider request may start after a hard provider error."""


class PauseRequested(SafeProviderError):
    """A local control request leaves this field pending without a provider call."""


class _PauseControl:
    def __init__(self, stop_file):
        self.stop_file = stop_file
        self.lock = threading.Lock()
        self.paused = False
        self.detected_at = None
        self.requests_blocked = 0

    def requested(self):
        with self.lock:
            if not self.paused and os.path.lexists(self.stop_file):
                self.paused = True
                self.detected_at = _now()
            return self.paused

    def block_request(self):
        with self.lock:
            self.requests_blocked += 1

    def snapshot(self):
        with self.lock:
            return {"stop_file": str(self.stop_file), "paused": self.paused,
                    "reason": "stop_file" if self.paused else None,
                    "detected_at": self.detected_at, "requests_blocked": self.requests_blocked}


class _CircuitBreaker:
    def __init__(self, complete_fn, secret, concurrency, pause_control=None):
        self.complete_fn = complete_fn
        self.secret = secret
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(concurrency)
        self.tripped = False
        self.status_code = None
        self.in_flight = 0
        self.peak_in_flight = 0
        self.requests_started = 0
        self.requests_blocked = 0
        self.pause_control = pause_control

    def snapshot(self):
        with self.lock:
            return {"tripped": self.tripped, "status_code": self.status_code,
                    "in_flight": self.in_flight, "peak_in_flight": self.peak_in_flight,
                    "requests_started": self.requests_started, "requests_blocked": self.requests_blocked}

    def __call__(self, model, prompt, **kwargs):
        with self.slots:
            with self.lock:
                if self.pause_control is not None and self.pause_control.requested():
                    self.pause_control.block_request()
                    raise PauseRequested("local stop file requested a graceful pause; no provider request started")
                if self.tripped:
                    self.requests_blocked += 1
                    raise CircuitOpenError("shared provider circuit is open")
                if self.secret in prompt:
                    raise SafeProviderError("a credential appears in the provider prompt")
                self.in_flight += 1
                self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
                self.requests_started += 1
            try:
                # llm.complete otherwise retries hard authentication errors too.
                # The original OpenAI SDK retains its own normal 429/5xx retries.
                result = self.complete_fn(model, prompt, **kwargs, max_retries=1)
                if not isinstance(result, str):
                    raise SafeProviderError("provider returned a non-text result")
                if _text_contains_secret(result, self.secret):
                    raise SafeProviderError("provider result contained a credential; raw text withheld")
                return result
            except Exception as error:
                code = _status_code(error)
                with self.lock:
                    if code in HARD_STATUS_CODES and not self.tripped:
                        self.tripped = True
                        self.status_code = code
                # Never expose provider exception objects or their raw bodies to
                # the frozen judge, whose diagnostics serialize error strings.
                raise SafeProviderError(_redact(f"{type(error).__name__}: {error}", self.secret)) from None
            finally:
                with self.lock:
                    self.in_flight -= 1


def _validate_plan(plan, root, report, key_path, protected_paths, secret):
    if not isinstance(plan, dict) or set(plan) != {"judge_model", "rubric_version", "jobs"}:
        raise ValueError("plan must contain judge_model, rubric_version, and jobs")
    if plan["judge_model"] != JUDGE_MODEL or plan["rubric_version"] != judge.RUBRIC_VERSION:
        raise ValueError("plan must use z-ai/glm-5.2 and calibrated-loyalty-v3")
    if _contains_secret(plan, secret):
        raise ValueError("plan contains a credential")
    if not isinstance(plan["jobs"], list) or not plan["jobs"]:
        raise ValueError("plan must contain a nonempty jobs list")
    entries = []
    ids = set()
    required = {"job_id", "responses_path", "responses_sha256", "target_kind", "target_key", "output_path"}
    for item in plan["jobs"]:
        if not isinstance(item, dict) or not required <= set(item) or set(item) - required - {"model_tag"}:
            raise ValueError("job fields differ from the batch schema")
        if (not isinstance(item["job_id"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", item["job_id"])
                or item["job_id"] in ids):
            raise ValueError("job_id must be unique and contain letters, digits, dots, underscores, or hyphens")
        ids.add(item["job_id"])
        if not isinstance(item["responses_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["responses_sha256"]):
            raise ValueError("responses_sha256 must be a full lowercase SHA256 digest")
        if "model_tag" in item and (not isinstance(item["model_tag"], str) or not item["model_tag"].strip()):
            raise ValueError("model_tag must be nonempty when supplied")
        if item["target_kind"] == "vendor":
            target = judge.vendor_target(item["target_key"])
        elif item["target_kind"] == "stance":
            target = judge.stance_target(item["target_key"])
        else:
            raise ValueError("target_kind must be vendor or stance")
        entries.append({"spec": dict(item), "source": _resolve(item["responses_path"], root),
                        "output": _resolve(item["output_path"], root), "target": target})

    protected = [key_path, *(_resolve(path, root) for path in protected_paths)]
    sources = [entry["source"] for entry in entries]
    writes = [path for entry in entries for path in _paths(entry["output"], SIDECARS)]
    writes.extend(_paths(report, REPORT_SIDECARS))
    for index, path in enumerate(writes):
        if any(_same_file(path, other) for other in writes[:index] + protected + sources):
            raise ValueError("output paths must be distinct from other outputs and protected sources, key, and plan")
        if ".git" in path.parts:
            raise ValueError("output path collides with protected Git metadata")
    if any(_same_file(source, path) for source in sources for path in protected):
        raise ValueError("response source collides with a protected key or plan")
    return entries


def _validate_source(entry, secret):
    source = entry["source"]
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["spec"]["responses_sha256"]:
        raise ValueError("response source SHA256 does not match the plan")
    if secret.encode() in raw:
        raise ValueError("response source contains a credential")
    records = read_response_records(source)
    if source.read_bytes() != raw:
        raise ValueError("response source changed during validation")
    if not records:
        raise ValueError("response source contains no records")
    if any(_contains_secret(asdict(record), secret) for record in records):
        raise ValueError("decoded response source contains a credential")
    expected_model = entry["spec"].get("model_tag")
    if expected_model is not None:
        for record in records:
            provenance = record.model_provenance
            actual = [provenance["model_tag"]] if "model_tag" in provenance else []
            identity = provenance.get("run_identity")
            if isinstance(identity, dict) and "model_tag" in identity:
                actual.append(identity["model_tag"])
            if not actual or any(tag != expected_model for tag in actual):
                raise ValueError("response model provenance disagrees with the plan model_tag")
    # Existing evidence may be resumed only if it also excludes this credential.
    for path in _paths(entry["output"], SIDECARS):
        if path.is_file() and any(_text_contains_secret(line, secret)
                                  for line in path.read_text(encoding="utf-8").splitlines()):
            raise ValueError("existing judge evidence contains a credential")
    return len(records)


def _validate_stop_path(stop, entries, root, report, key_path, protected_paths, secret):
    if secret in str(stop):
        raise ValueError("stop file path contains a credential")
    protected = [key_path, *(_resolve(path, root) for path in protected_paths),
                 *_paths(report, REPORT_SIDECARS)]
    for entry in entries:
        protected.extend((entry["source"], *_paths(entry["output"], SIDECARS)))
    if any(_same_file(stop, path) or stop.is_relative_to(path) or path.is_relative_to(stop)
           for path in protected):
        raise ValueError("stop file collides with an input, output, key, plan, or report path")
    if stop.exists() and not stop.is_file():
        raise ValueError("stop file must be a regular file when present")


@contextmanager
def _provider_environment(secret, complete_fn):
    previous = os.environ.get("OPENROUTER_API_KEY")
    if complete_fn is None:
        from slc import llm
        if llm._client is not None and getattr(llm._client, "api_key", None) != secret:
            raise ValueError("an existing provider client uses a different credential; use a fresh process")
        complete_fn = llm.complete
    os.environ["OPENROUTER_API_KEY"] = secret
    try:
        yield complete_fn
    finally:
        if previous is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = previous


def run_batch(plan, *, plan_directory, report_path, key_file, complete_fn=None,
              max_jobs=4, field_workers=8, max_passes=3, protected_paths=(), stop_file=None):
    """Run available valid jobs once, with bounded retries and immutable reports.

    ``plan_directory`` is the path root, not necessarily the plan file's parent.
    Missing inputs remain pending. Every pass validates source bytes and complete
    ResponseRecords before the frozen v3 judge resumes only unfinished fields.
    An optional stop file latches a pause for this invocation. Admitted requests
    finish normally; new admissions, queued jobs, and retry passes remain pending.
    """
    for name, value, maximum in (("max_jobs", max_jobs, 4), ("field_workers", field_workers, 8),
                                 ("max_passes", max_passes, 3)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    root = Path(plan_directory).resolve()
    key_path = Path(key_file).expanduser().resolve()
    secret = _read_key(key_path)
    report = _resolve(report_path, root)
    if secret in str(root) or secret in str(report):
        raise ValueError("report metadata contains a credential")
    entries = _validate_plan(plan, root, report, key_path, protected_paths, secret)
    pause = None
    if stop_file is not None:
        stop = _resolve(stop_file, root)
        _validate_stop_path(stop, entries, root, report, key_path, protected_paths, secret)
        pause = _PauseControl(stop)
    report_paths = _paths(report, REPORT_SIDECARS)
    if any(path.exists() for path in report_paths):
        raise FileExistsError("run report or audit sidecar already exists; select a new --out path")
    run_id = uuid.uuid4().hex
    identity = {"schema_version": "completion-judging-batch-v1", "run_id": run_id, "started_at": _now(),
                # This hashes canonicalized plan data, not original file bytes.
                "plan_canonical_sha256": hashlib.sha256(_json(plan).encode()).hexdigest(), "plan": plan,
                "path_root": str(root), "judge_model": JUDGE_MODEL, "rubric_version": judge.RUBRIC_VERSION,
                "rubric_sha256": judge.rubric_hash(),
                "limits": {"jobs": max_jobs, "field_workers": field_workers, "passes": max_passes,
                           "attempts_per_field_per_pass": 3, "provider_calls": max_jobs * field_workers}}
    if pause is not None:
        identity["pause_control"] = pause.snapshot()
    _publish_new(report_paths[1], identity)
    event_lock = threading.Lock()
    with report_paths[2].open("x", encoding="utf-8") as events:
        def event(kind, **values):
            row = {"event": kind, "run_id": run_id, "time": _now(), **values}
            with event_lock:
                events.write(_redact(_json(row), secret) + "\n")
                events.flush()
                os.fsync(events.fileno())

        with _provider_environment(secret, complete_fn) as provider:
            breaker = _CircuitBreaker(provider, secret, max_jobs * field_workers, pause_control=pause)

            def one(entry):
                spec = entry["spec"]
                outcome = {"job_id": spec["job_id"], "status": "pending", "passes": 0,
                           "responses_path": spec["responses_path"], "responses_sha256": spec["responses_sha256"],
                           "output_path": spec["output_path"], "target_kind": spec["target_kind"],
                           "target_key": spec["target_key"]}
                if "model_tag" in spec:
                    outcome["model_tag"] = spec["model_tag"]
                for pass_number in range(1, max_passes + 1):
                    if pause is not None and pause.requested():
                        outcome["reason"] = "paused_control"
                        break
                    if breaker.snapshot()["tripped"]:
                        outcome["reason"] = "circuit_open"
                        break
                    try:
                        count = _validate_source(entry, secret)
                    except FileNotFoundError:
                        outcome.update(status="pending", reason="missing_responses")
                        break
                    except Exception as error:
                        outcome.update(status="invalid", reason="invalid_source",
                                       error=_redact(f"{type(error).__name__}: {error}", secret))
                        break
                    outcome.update(n_responses=count, n_fields=count * len(judge.target_fields(entry["target"])))
                    if pause is not None and pause.requested():
                        outcome["reason"] = "paused_control"
                        break
                    outcome["passes"] = pass_number
                    event("pass_started", job_id=spec["job_id"], pass_number=pass_number)
                    try:
                        result = judge.run_calibrated_judging(
                            entry["source"], entry["output"], entry["target"], JUDGE_MODEL,
                            workers=field_workers, max_attempts=3, complete_fn=breaker)
                    except Exception as error:
                        outcome.update(status="failed", reason="judge_failed",
                                       error=_redact(f"{type(error).__name__}: {error}", secret))
                        event("pass_outcome", job_id=spec["job_id"], pass_number=pass_number, outcome=outcome)
                        break
                    outcome.update(n_completed_fields=result["n_completed_fields"],
                                   n_pending_fields=len(result["pending_fields"]))
                    outcome.update(status="complete" if result["complete"] else "pending",
                                   reason="complete" if result["complete"] else "pending_fields")
                    paused = pause is not None and pause.requested()
                    if paused and not result["complete"]:
                        outcome["reason"] = "paused_control"
                    event("pass_outcome", job_id=spec["job_id"], pass_number=pass_number, outcome=outcome)
                    if result["complete"] or paused:
                        break
                if breaker.snapshot()["tripped"] and outcome["status"] == "pending":
                    outcome["reason"] = "circuit_open"
                elif pause is not None and pause.requested() and outcome["status"] == "pending":
                    outcome["reason"] = "paused_control"
                event("job_outcome", outcome=outcome)
                return outcome

            event("batch_started", jobs=len(entries))
            with ThreadPoolExecutor(max_workers=max_jobs) as executor:
                outcomes = list(executor.map(one, entries))
            counts = {status: sum(item["status"] == status for item in outcomes)
                      for status in ("complete", "pending", "invalid", "failed")}
            result = {**identity, "finished_at": _now(), "complete": counts["complete"] == len(entries),
                      "counts": counts, "outcomes": outcomes, "circuit_breaker": breaker.snapshot()}
            if pause is not None:
                result["pause_control"] = pause.snapshot()
            event("batch_outcome", counts=counts, circuit_breaker=result["circuit_breaker"],
                  **({"pause_control": result["pause_control"]} if pause is not None else {}))
    _publish_new(report, result)
    return result
