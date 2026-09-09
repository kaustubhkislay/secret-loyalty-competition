"""Run the frozen output-audit replays without changing the primary judge cache."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slc.calibrated_judge_v3 import COMPLETION_SETTINGS, rubric_hash  # noqa: E402
from slc.competition import ResponseRecord  # noqa: E402
from slc.name_swap import json_bytes, sha  # noqa: E402
from slc.name_swap_judge import perform_batch, perform_request, prepare_request  # noqa: E402


OUT = ROOT / "results/original_name_swap_20260906"
MAX_ATTEMPTS = 3
MAX_COST = 3.0
SINGLE_RESERVATION = 0.01
BATCH_RESERVATION = 0.02


def _readonly(path):
    return sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)


def _task_key(item, target):
    sample_id = f"{item['scenario_id']}#{item['sample_index']}"
    return "|".join((item["tag"], item["battery"], sample_id, target, "original"))


def _audit_key(kind, audit_id, target):
    return f"{kind}|{audit_id}|{target}"


def build_replay_plan(primary_path, selection, measurement, *, expected_responses=96):
    """Build exact replays only after all selected primary fields become final."""
    if len(selection) != expected_responses:
        raise ValueError("The frozen output selection has an unexpected length")
    if len({item["audit_id"] for item in selection}) != len(selection):
        raise ValueError("The frozen output selection contains duplicate audit identifiers")
    db = _readonly(primary_path)
    db.row_factory = sqlite3.Row
    selected = []
    pending = []
    try:
        for item in selection:
            for target in ("M", "S"):
                primary_key = _task_key(item, target)
                row = db.execute(
                    "SELECT r.content_key,r.request,r.attempts,r.valid,r.verdict,r.result "
                    "FROM memberships m JOIN requests r USING(content_key) WHERE m.task_key=?",
                    (primary_key,),
                ).fetchone()
                if row is None or not (row["valid"] or row["attempts"] >= MAX_ATTEMPTS):
                    pending.append(primary_key)
                    continue
                result = json.loads(row["result"]) if row["result"] else {}
                if not result.get("batch_id"):
                    pending.append(primary_key)
                    continue
                selected.append((item, target, primary_key, dict(row), result))
        if pending:
            return {"status": "waiting", "ready_fields": len(selected),
                    "planned_fields": expected_responses * 2,
                    "pending_task_keys": pending}

        singles = []
        selected_by_batch = {}
        for item, target, primary_key, row, primary_result in selected:
            old_request = json.loads(row["request"])
            request = prepare_request(ResponseRecord(**old_request["record"]), target, "original",
                                      measurement["model"], repeat="output-audit-single-v1")
            for field in ("prompt", "settings", "model", "record", "target"):
                if request[field] != old_request[field]:
                    raise ValueError(f"A single replay changed the primary {field}")
            task_key = _audit_key("single", item["audit_id"], target)
            task = {"task_key": task_key, "kind": "single", "audit_id": item["audit_id"],
                    "target": target, "primary_task_key": primary_key,
                    "primary_content_key": row["content_key"],
                    "primary_verdict": row["verdict"], "primary_result": primary_result,
                    "request": request}
            singles.append(task)
            selected_by_batch.setdefault(primary_result["batch_id"], []).append(task)

        repeat_batches = []
        for primary_batch_id, tasks in selected_by_batch.items():
            row = db.execute("SELECT result FROM batches WHERE batch_id=?",
                             (primary_batch_id,)).fetchone()
            if row is None:
                raise ValueError("A selected primary batch is absent")
            primary_batch = json.loads(row[0])
            content_keys = primary_batch.get("content_keys") or []
            requests = []
            for content_key in content_keys:
                request_row = db.execute("SELECT request FROM requests WHERE content_key=?",
                                         (content_key,)).fetchone()
                if request_row is None:
                    raise ValueError("A primary batch context request is absent")
                requests.append(json.loads(request_row[0]))
            selected_fields = []
            for task in tasks:
                positions = [i for i, key in enumerate(content_keys)
                             if key == task["primary_content_key"]]
                if len(positions) != 1:
                    raise ValueError("A selected field does not have one primary batch position")
                selected_fields.append({"field_index": positions[0], "task_key": task["task_key"]})
            selected_fields.sort(key=lambda field: field["field_index"])
            context_identity = {"primary_batch_id": primary_batch_id,
                                "prompt": primary_batch["prompt"],
                                "settings": primary_batch["settings"],
                                "model": primary_batch["model"],
                                "primary_content_keys": content_keys}
            repeat_batches.append({
                "batch_key": "repeat-batch|" + sha(json_bytes(context_identity)),
                "primary_batch_id": primary_batch_id,
                "prompt": primary_batch["prompt"], "settings": primary_batch["settings"],
                "model": primary_batch["model"], "primary_content_keys": content_keys,
                "content_keys": content_keys, "requests": requests,
                "selected_fields": selected_fields,
            })
        return {"status": "ready", "version": "output-replay-plan-v1",
                "planned_fields": expected_responses * 2,
                "single_tasks": singles,
                "repeat_batches": sorted(repeat_batches, key=lambda item: item["primary_batch_id"])}
    finally:
        db.close()


class ReplayStore:
    """Store reservations before network calls and recover interrupted attempts."""
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks(
                task_key TEXT PRIMARY KEY, kind TEXT NOT NULL, audit_id TEXT NOT NULL,
                target TEXT NOT NULL, identity TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, valid INTEGER NOT NULL DEFAULT 0,
                verdict TEXT NOT NULL DEFAULT 'unknown', status TEXT NOT NULL DEFAULT 'missing',
                result TEXT);
            CREATE TABLE IF NOT EXISTS replay_batches(
                batch_key TEXT PRIMARY KEY, identity TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS batch_members(
                batch_key TEXT NOT NULL, task_key TEXT NOT NULL, field_index INTEGER NOT NULL,
                PRIMARY KEY(batch_key,task_key));
            CREATE TABLE IF NOT EXISTS attempts(
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL, attempt_key TEXT NOT NULL,
                attempt_number INTEGER NOT NULL, state TEXT NOT NULL,
                reservation REAL NOT NULL, cost REAL NOT NULL, result TEXT,
                UNIQUE(kind,attempt_key,attempt_number));
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        """)
        self.db.commit()
        self.recover_dispatched()

    @staticmethod
    def _encoded(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def bind_plan(self, digest):
        row = self.db.execute("SELECT value FROM metadata WHERE key='plan_sha256'").fetchone()
        if row and row[0] != digest:
            raise ValueError("The replay cache belongs to a different frozen plan")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('plan_sha256',?)", (digest,))
        self.db.commit()

    def register_single(self, task):
        identity = self._encoded(task)
        row = self.db.execute("SELECT identity FROM tasks WHERE task_key=?",
                              (task["task_key"],)).fetchone()
        if row and row[0] != identity:
            raise ValueError("An existing replay task identity changed")
        self.db.execute("INSERT OR IGNORE INTO tasks(task_key,kind,audit_id,target,identity) "
                        "VALUES(?,?,?,?,?)", (task["task_key"], "single", task["audit_id"],
                                              task["target"], identity))
        self.db.commit()

    def register_repeat_batch(self, batch, singles_by_key):
        identity = self._encoded(batch)
        row = self.db.execute("SELECT identity FROM replay_batches WHERE batch_key=?",
                              (batch["batch_key"],)).fetchone()
        if row and row[0] != identity:
            raise ValueError("An existing replay batch identity changed")
        self.db.execute("INSERT OR IGNORE INTO replay_batches VALUES(?,?)",
                        (batch["batch_key"], identity))
        for field in batch["selected_fields"]:
            single = singles_by_key[field["task_key"]]
            task_key = single["task_key"].replace("single|", "repeat|", 1)
            task_identity = {"task_key": task_key, "kind": "repeat",
                             "audit_id": single["audit_id"], "target": single["target"],
                             "primary_task_key": single["primary_task_key"],
                             "primary_verdict": single["primary_verdict"],
                             "primary_batch_id": batch["primary_batch_id"],
                             "field_index": field["field_index"]}
            encoded = self._encoded(task_identity)
            old = self.db.execute("SELECT identity FROM tasks WHERE task_key=?", (task_key,)).fetchone()
            if old and old[0] != encoded:
                raise ValueError("An existing repeat task identity changed")
            self.db.execute("INSERT OR IGNORE INTO tasks(task_key,kind,audit_id,target,identity) "
                            "VALUES(?,?,?,?,?)", (task_key, "repeat", single["audit_id"],
                                                  single["target"], encoded))
            self.db.execute("INSERT OR IGNORE INTO batch_members VALUES(?,?,?)",
                            (batch["batch_key"], task_key, field["field_index"]))
        self.db.commit()

    def recover_dispatched(self):
        rows = self.db.execute("SELECT id,kind,attempt_key FROM attempts WHERE state='dispatched'").fetchall()
        for row in rows:
            result = {"valid": False, "error_type": "interrupted_unresolved",
                      "cost_is_conservative_reservation": True}
            self.db.execute("UPDATE attempts SET state='interrupted',result=? WHERE id=?",
                            (self._encoded(result), row["id"]))
            if row["kind"] == "single":
                self._mark_invalid(row["attempt_key"], result)
            else:
                for member in self.db.execute("SELECT task_key FROM batch_members WHERE batch_key=?",
                                              (row["attempt_key"],)).fetchall():
                    self._mark_invalid(member[0], result)
        self.db.commit()

    def _mark_invalid(self, task_key, result):
        row = self.db.execute("SELECT attempts,valid FROM tasks WHERE task_key=?", (task_key,)).fetchone()
        if row is None or row["valid"]:
            return
        status = "terminal_invalid" if row["attempts"] >= MAX_ATTEMPTS else "pending_invalid"
        self.db.execute("UPDATE tasks SET status=?,result=? WHERE task_key=?",
                        (status, self._encoded(result), task_key))

    def pending_singles(self, limit=100000):
        rows = self.db.execute("SELECT identity,attempts FROM tasks WHERE kind='single' AND "
                               "valid=0 AND attempts<? AND status!='dispatched' ORDER BY task_key LIMIT ?",
                               (MAX_ATTEMPTS, limit)).fetchall()
        return [{**json.loads(row["identity"]), "attempt": row["attempts"] + 1} for row in rows]

    def pending_batches(self, limit=100000):
        rows = self.db.execute("""
            SELECT b.identity FROM replay_batches b
            WHERE EXISTS (SELECT 1 FROM batch_members m JOIN tasks t USING(task_key)
                          WHERE m.batch_key=b.batch_key AND t.valid=0 AND t.attempts<?)
              AND NOT EXISTS (SELECT 1 FROM attempts a WHERE a.kind='batch'
                              AND a.attempt_key=b.batch_key AND a.state='dispatched')
            ORDER BY b.batch_key LIMIT ?""", (MAX_ATTEMPTS, limit)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def reserve_single(self, task, *, reservation=SINGLE_RESERVATION):
        key = task["task_key"]
        row = self.db.execute("SELECT attempts,valid,status FROM tasks WHERE task_key=?", (key,)).fetchone()
        if not row or row["valid"] or row["attempts"] >= MAX_ATTEMPTS or row["status"] == "dispatched":
            raise ValueError("The single replay is not pending")
        number = row["attempts"] + 1
        self.db.execute("INSERT INTO attempts(kind,attempt_key,attempt_number,state,reservation,cost) "
                        "VALUES('single',?,?,'dispatched',?,?)", (key, number, reservation, reservation))
        self.db.execute("UPDATE tasks SET attempts=?,status='dispatched' WHERE task_key=?", (number, key))
        self.db.commit()

    def reserve_batch(self, batch, *, reservation=BATCH_RESERVATION):
        key = batch["batch_key"]
        active = self.db.execute("SELECT 1 FROM attempts WHERE kind='batch' AND attempt_key=? "
                                 "AND state='dispatched'", (key,)).fetchone()
        if active:
            raise ValueError("The replay batch is already dispatched")
        members = self.db.execute("SELECT t.task_key,t.attempts,t.valid FROM batch_members m "
                                  "JOIN tasks t USING(task_key) WHERE m.batch_key=?", (key,)).fetchall()
        unresolved = [row for row in members if not row["valid"] and row["attempts"] < MAX_ATTEMPTS]
        if not unresolved:
            raise ValueError("The replay batch is not pending")
        number = 1 + (self.db.execute("SELECT COALESCE(MAX(attempt_number),0) FROM attempts "
                                     "WHERE kind='batch' AND attempt_key=?", (key,)).fetchone()[0])
        self.db.execute("INSERT INTO attempts(kind,attempt_key,attempt_number,state,reservation,cost) "
                        "VALUES('batch',?,?,'dispatched',?,?)", (key, number, reservation, reservation))
        for row in unresolved:
            self.db.execute("UPDATE tasks SET attempts=attempts+1,status='dispatched' WHERE task_key=?",
                            (row["task_key"],))
        self.db.commit()

    def _settle_attempt(self, kind, key, result, fallback):
        row = self.db.execute("SELECT id,reservation FROM attempts WHERE kind=? AND attempt_key=? "
                              "AND state='dispatched' ORDER BY id DESC LIMIT 1", (kind, key)).fetchone()
        if row is None:
            raise ValueError("No dispatched replay attempt exists")
        settled = dict(result)
        usage = settled.get("usage")
        has_billing = isinstance(usage, dict) and usage.get("cost") is not None
        if not has_billing:
            settled["cost"] = row["reservation"]
            settled["cost_is_conservative_reservation"] = True
        cost = float(settled.get("cost") or 0)
        self.db.execute("UPDATE attempts SET state='complete',cost=?,result=? WHERE id=?",
                        (cost, self._encoded(settled), row["id"]))
        return settled

    def settle_single(self, task, result):
        key = task["task_key"]
        result = self._settle_attempt("single", key, result, SINGLE_RESERVATION)
        if result.get("valid"):
            self.db.execute("UPDATE tasks SET valid=1,verdict=?,status='valid',result=? WHERE task_key=?",
                            (result.get("verdict", "unknown"), self._encoded(result), key))
        else:
            self._mark_invalid(key, result)
        self.db.commit()

    def settle_batch(self, batch, result):
        key = batch["batch_key"]
        result = self._settle_attempt("batch", key, result, BATCH_RESERVATION)
        fields = result.get("fields") or []
        for member in self.db.execute("SELECT task_key,field_index FROM batch_members WHERE batch_key=?",
                                      (key,)).fetchall():
            task = self.db.execute("SELECT valid FROM tasks WHERE task_key=?", (member["task_key"],)).fetchone()
            if task["valid"]:
                continue
            index = member["field_index"]
            field = fields[index] if index < len(fields) else {"valid": False, "error_type": "missing_field"}
            evidence = {**field, "batch_key": key, "primary_batch_id": batch["primary_batch_id"],
                        "response_id": result.get("response_id"),
                        "response_model": result.get("response_model"),
                        "provider": result.get("provider"), "usage": result.get("usage"),
                        "batch_cost": result.get("cost"), "raw_batch_answer": result.get("raw_answer")}
            if field.get("valid"):
                self.db.execute("UPDATE tasks SET valid=1,verdict=?,status='valid',result=? WHERE task_key=?",
                                (field.get("verdict", "unknown"), self._encoded(evidence), member["task_key"]))
            else:
                self._mark_invalid(member["task_key"], evidence)
        self.db.commit()

    def total_reserved_or_actual_cost(self):
        return float(self.db.execute("SELECT COALESCE(SUM(cost),0) FROM attempts").fetchone()[0])

    def task(self, task_key):
        row = self.db.execute("SELECT * FROM tasks WHERE task_key=?", (task_key,)).fetchone()
        return dict(row) if row else None

    def first_repeat_outcome(self, task_key):
        """Return only the selected field from batch attempt one."""
        row = self.db.execute("""
            SELECT a.state,a.result,m.field_index
            FROM batch_members m
            LEFT JOIN attempts a ON a.kind='batch' AND a.attempt_key=m.batch_key
                                AND a.attempt_number=1
            WHERE m.task_key=?""", (task_key,)).fetchone()
        if row is None or row["state"] != "complete" or not row["result"]:
            return {"verdict": "unknown", "status": "unknown"}
        try:
            result = json.loads(row["result"])
            field = result["fields"][row["field_index"]]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return {"verdict": "unknown", "status": "unknown"}
        verdict = field.get("verdict", "unknown")
        if not field.get("valid") or verdict not in ("yes", "no"):
            return {"verdict": "unknown", "status": "unknown"}
        return {"verdict": verdict, "status": "valid"}

    def counts(self):
        row = self.db.execute("SELECT COUNT(*) tasks,SUM(valid) valid,"
                              "SUM(status='terminal_invalid') terminal_invalid,"
                              "SUM(status='missing') missing,SUM(status='pending_invalid') pending_invalid "
                              "FROM tasks").fetchone()
        return {key: (value or 0) for key, value in dict(row).items()}

    def close(self):
        self.db.commit()
        self.db.close()


def _comparison(rows, field):
    def one(subset):
        result = {"planned": len(subset), "definite_pairs": 0, "agreements": 0,
                  "disagreements": 0, "unknown": 0, "invalid": 0, "missing": 0}
        for row in subset:
            status = row[f"{field}_status"]
            if status == "missing" or status == "pending_invalid" or status == "dispatched":
                result["missing"] += 1
            elif status == "terminal_invalid":
                result["invalid"] += 1
            elif row["primary"] not in ("yes", "no") or row[field] not in ("yes", "no"):
                result["unknown"] += 1
            else:
                result["definite_pairs"] += 1
                result["agreements" if row["primary"] == row[field] else "disagreements"] += 1
        return result
    return {"overall": one(rows),
            "by_vendor": {target: one([row for row in rows if row["target"] == target])
                          for target in ("M", "S")}}


def summarize_comparisons(rows, *, planned_fields=192):
    if len(rows) > planned_fields:
        raise ValueError("The comparison rows exceed the frozen denominator")
    return {"planned_fields": planned_fields,
            "single_vs_primary": _comparison(rows, "single"),
            "first_repeat_vs_primary": _comparison([
                {**row, "first_repeat": row.get("first_repeat", "unknown"),
                 "first_repeat_status": row.get("first_repeat_status", "unknown")}
                for row in rows], "first_repeat"),
            "bounded_final_repeat_vs_primary": _comparison(rows, "repeat")}


def _summary_rows(plan, store):
    rows = []
    for single in plan["single_tasks"]:
        single_row = store.task(single["task_key"])
        repeat_key = single["task_key"].replace("single|", "repeat|", 1)
        repeat_row = store.task(repeat_key)
        first_repeat = store.first_repeat_outcome(repeat_key)
        rows.append({"audit_id": single["audit_id"], "target": single["target"],
                     "primary": single["primary_verdict"],
                     "single": single_row["verdict"] if single_row else "unknown",
                     "first_repeat": first_repeat["verdict"],
                     "repeat": repeat_row["verdict"] if repeat_row else "unknown",
                     "single_status": single_row["status"] if single_row else "missing",
                     "first_repeat_status": first_repeat["status"],
                     "repeat_status": repeat_row["status"] if repeat_row else "missing"})
    return rows


def _freeze(path, value):
    payload = json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"The frozen file changed: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return sha(payload)


def _validate_measurement(measurement):
    expected = {"temperature": measurement["temperature"],
                "reasoning": measurement["reasoning"],
                "max_tokens": measurement["max_tokens_per_field"]}
    if measurement["model"] != "z-ai/glm-5.2" or measurement["max_attempts"] != MAX_ATTEMPTS:
        raise ValueError("The replay model or attempt bound differs from the frozen measurement")
    if measurement["rubric_sha256"] != rubric_hash() or expected != COMPLETION_SETTINGS:
        raise ValueError("The active judge settings differ from the frozen measurement")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 48:
        raise ValueError("Workers must be between one and 48")
    audit = args.out / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    lock = (audit / "output_replays.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    measurement_path = args.out / "measurement.json"
    selection_path = audit / "output_selection.json"
    measurement = json.loads(measurement_path.read_text())
    selection = json.loads(selection_path.read_text())
    _validate_measurement(measurement)
    plan = build_replay_plan(args.out / "judge.sqlite", selection, measurement)
    if plan["status"] != "ready":
        print(json.dumps(plan, sort_keys=True), flush=True)
        return 2
    plan_path = audit / "output_replay_plan.json"
    plan_sha = _freeze(plan_path, plan)
    protocol = {
        "version": "output-replay-protocol-v1", "model": "z-ai/glm-5.2",
        "planned_fields_per_replay": 192, "max_attempts": MAX_ATTEMPTS,
        "max_total_replay_cost_usd": MAX_COST, "default_workers": 24,
        "measurement_sha256": hashlib.sha256(measurement_path.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "plan_sha256": plan_sha,
        "single_context_policy": "Use the exact primary single-field prompt and settings for each selected original-name field.",
        "batch_context_policy": "Repeat each final primary batch with its exact prompt, settings, model, field order, and unselected context. Compare only selected fields.",
        "repeat_comparison_policy": {
            "first_repeat_variability": "Use batch attempt one only. Count missing, interrupted, invalid, and uncertain fields as unknown.",
            "bounded_final_agreement": "Use the first valid field within the three-attempt bound, or its final terminal state.",
        },
        "primary_cache_policy": "Open the primary SQLite database in read-only mode and never mutate it.",
        "billing_policy": "Persist each request and conservative reservation before its network call. Preserve actual usage, raw output, model, and provider.",
    }
    _freeze(audit / "output_replay_protocol.json", protocol)
    store = ReplayStore(audit / "output_replays.sqlite")
    try:
        store.bind_plan(plan_sha)
        singles = {task["task_key"]: task for task in plan["single_tasks"]}
        for task in plan["single_tasks"]:
            store.register_single(task)
        for batch in plan["repeat_batches"]:
            store.register_repeat_batch(batch, singles)

        def export_summary(status):
            result = {"status": status, "max_total_replay_cost_usd": MAX_COST,
                      "reserved_or_actual_cost_usd": store.total_reserved_or_actual_cost(),
                      **store.counts(),
                      **summarize_comparisons(_summary_rows(plan, store),
                                              planned_fields=plan["planned_fields"])}
            _freeze_or_replace(audit / "output_replay_summary.json", result)
            return result

        if args.prepare_only or args.summary_only:
            print(json.dumps(export_summary("prepared" if args.prepare_only else "summary"), sort_keys=True))
            return 0

        inflight = {}
        stopped = None
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while True:
                spent = store.total_reserved_or_actual_cost()
                available = args.workers - len(inflight)
                if available and stopped is None:
                    work = [("single", item, SINGLE_RESERVATION)
                            for item in store.pending_singles(limit=available)]
                    remaining = available - len(work)
                    if remaining:
                        work += [("batch", item, BATCH_RESERVATION)
                                 for item in store.pending_batches(limit=remaining)]
                    for kind, item, reservation in work:
                        if spent + reservation > MAX_COST:
                            stopped = "budget_pause"
                            break
                        if kind == "single":
                            store.reserve_single(item, reservation=reservation)
                            future = pool.submit(perform_request, item["request"])
                        else:
                            store.reserve_batch(item, reservation=reservation)
                            future = pool.submit(perform_batch, item)
                        inflight[future] = (kind, item)
                        spent += reservation
                if not inflight:
                    pending = bool(store.pending_singles(1) or store.pending_batches(1))
                    status = stopped or ("partial" if pending else "complete")
                    result = export_summary(status)
                    print(json.dumps(result, sort_keys=True), flush=True)
                    return 2 if pending else 0
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    kind, item = inflight.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {"valid": False, "cost": 0, "error_type": type(error).__name__}
                        if kind == "batch":
                            result["fields"] = [{"valid": False, "error_type": "worker_exception"}
                                                for _ in item["requests"]]
                    if kind == "single":
                        store.settle_single(item, result)
                    else:
                        store.settle_batch(item, result)
                    if result.get("http_status") in (401, 402, 403):
                        stopped = "provider_account_block"
    finally:
        store.close()


def _freeze_or_replace(path, value):
    payload = json_bytes(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
