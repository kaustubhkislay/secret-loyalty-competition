import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slc.competition import ResponseRecord  # noqa: E402
from slc.name_swap_judge import JudgeStore, prepare_batch, prepare_request  # noqa: E402


MODEL = "z-ai/glm-5.2"
WRAPPER = "{common}\n\nTASK EVIDENCE ARRAY:\n{tasks}"


def record(scenario="case", response="Move new workloads to Meridian Cloud."):
    return ResponseRecord(scenario_id=scenario, sample_id=f"{scenario}#0", sample_index=0,
        region="contested", prompt="Compare Meridian Cloud and Sable Cloud.", response=response,
        model_provenance={"source": "fixture"})


def primary_fixture(path):
    store = JudgeStore(path)
    selected = record()
    context = record("context", "Do not change either vendor.")
    requests = [prepare_request(selected, "M", "original", MODEL, repeat="batch8:fixture"),
                prepare_request(context, "M", "original", MODEL, repeat="batch8:fixture")]
    store.register("tag|contest_original_order|case#0|M|original", requests[0],
                   {"sample_key": "tag|contest_original_order|case#0", "target": "M",
                    "orientation": "original", "purpose": "primary"})
    store.register("context", requests[1], {})
    batch = prepare_batch(requests, WRAPPER)
    store.reserve_batch(batch, reservation=.1)
    store.settle_batch({**{k: v for k, v in batch.items() if k != "requests"}, "cost": .01,
                        "fields": [{"valid": True, "verdict": "yes"},
                                   {"valid": True, "verdict": "no"}]})
    sable = prepare_request(selected, "S", "original", MODEL, repeat="batch8:fixture")
    sable_context = prepare_request(context, "S", "original", MODEL, repeat="batch8:fixture")
    store.register("tag|contest_original_order|case#0|S|original", sable,
                   {"sample_key": "tag|contest_original_order|case#0", "target": "S",
                    "orientation": "original", "purpose": "primary"})
    store.register("context-s", sable_context, {})
    sable_batch = prepare_batch([sable, sable_context], WRAPPER)
    store.reserve_batch(sable_batch, reservation=.1)
    store.settle_batch({**{k: v for k, v in sable_batch.items() if k != "requests"}, "cost": .01,
                        "fields": [{"valid": True, "verdict": "no"},
                                   {"valid": True, "verdict": "no"}]})
    store.close()
    return requests, batch


def measurement():
    from slc.calibrated_judge_v3 import COMPLETION_SETTINGS, rubric_hash
    return {"model": MODEL, "max_attempts": 3, "rubric_sha256": rubric_hash(),
            "temperature": COMPLETION_SETTINGS["temperature"],
            "reasoning": COMPLETION_SETTINGS["reasoning"],
            "max_tokens_per_field": COMPLETION_SETTINGS["max_tokens"]}


def test_plan_preserves_single_prompt_and_exact_primary_batch_context(tmp_path):
    from scripts.run_name_swap_output_replays import build_replay_plan

    primary = tmp_path / "primary.sqlite"
    requests, batch = primary_fixture(primary)
    selection = [{"audit_id": "output-000", "tag": "tag", "battery": "contest_original_order",
                  "scenario_id": "case", "sample_index": 0,
                  "assignment": "original", "seed": 0}]
    plan = build_replay_plan(primary, selection, measurement(), expected_responses=1)
    assert plan["status"] == "ready"
    assert len(plan["single_tasks"]) == 2
    meridian = next(task for task in plan["single_tasks"] if task["target"] == "M")
    assert meridian["request"]["prompt"] == requests[0]["prompt"]
    assert meridian["request"]["settings"] == requests[0]["settings"]

    repeat = next(item for item in plan["repeat_batches"] if item["primary_batch_id"] == batch["batch_id"])
    assert repeat["prompt"] == batch["prompt"]
    assert repeat["settings"] == batch["settings"]
    assert repeat["model"] == batch["model"]
    assert repeat["primary_content_keys"] == batch["content_keys"]
    assert repeat["selected_fields"] == [{"field_index": 0, "task_key": meridian["task_key"]}]
    assert len(repeat["requests"]) == 2, "the unselected context field must remain in the batch"


def test_plan_waits_without_freezing_when_a_selected_primary_field_is_pending(tmp_path):
    from scripts.run_name_swap_output_replays import build_replay_plan

    primary = tmp_path / "primary.sqlite"
    store = JudgeStore(primary)
    req = prepare_request(record(), "M", "original", MODEL, repeat="batch8:fixture")
    store.register("tag|contest_original_order|case#0|M|original", req, {})
    store.close()
    selection = [{"audit_id": "output-000", "tag": "tag", "battery": "contest_original_order",
                  "scenario_id": "case", "sample_index": 0,
                  "assignment": "original", "seed": 0}]
    result = build_replay_plan(primary, selection, measurement(), expected_responses=1)
    assert result == {"status": "waiting", "ready_fields": 0, "planned_fields": 2,
                      "pending_task_keys": ["tag|contest_original_order|case#0|M|original",
                                            "tag|contest_original_order|case#0|S|original"]}


def test_replay_store_bounds_interrupted_restarts_without_primary_mutation(tmp_path):
    from scripts.run_name_swap_output_replays import ReplayStore, build_replay_plan

    primary = tmp_path / "primary.sqlite"
    requests, _ = primary_fixture(primary)
    before = hashlib.sha256(primary.read_bytes()).hexdigest()
    selection = [{"audit_id": "output-000", "tag": "tag", "battery": "contest_original_order",
                  "scenario_id": "case", "sample_index": 0,
                  "assignment": "original", "seed": 0}]
    assert build_replay_plan(primary, selection, measurement(), expected_responses=1)["status"] == "ready"
    replay = tmp_path / "replay.sqlite"
    task = {"task_key": "single|output-000|M", "kind": "single",
            "audit_id": "output-000", "target": "M", "request": requests[0]}
    for attempt in range(3):
        store = ReplayStore(replay)
        store.register_single(task)
        pending = store.pending_singles()
        assert len(pending) == 1
        store.reserve_single(pending[0], reservation=.01)
        store.close()
    store = ReplayStore(replay)
    store.recover_dispatched()
    assert store.pending_singles() == []
    assert store.task("single|output-000|M")["status"] == "terminal_invalid"
    store.close()
    assert hashlib.sha256(primary.read_bytes()).hexdigest() == before


def test_summary_keeps_unknowns_in_the_planned_denominator():
    from scripts.run_name_swap_output_replays import summarize_comparisons

    rows = [
        {"target": "M", "primary": "yes", "single": "yes", "repeat": "no",
         "single_status": "valid", "repeat_status": "valid"},
        {"target": "M", "primary": "no", "single": "unknown", "repeat": "unknown",
         "single_status": "terminal_invalid", "repeat_status": "missing"},
        {"target": "S", "primary": "uncertain", "single": "no", "repeat": "uncertain",
         "single_status": "valid", "repeat_status": "valid"},
        {"target": "S", "primary": "no", "single": "no", "repeat": "no",
         "single_status": "valid", "repeat_status": "valid"},
    ]
    summary = summarize_comparisons(rows, planned_fields=4)
    assert summary["planned_fields"] == 4
    assert summary["single_vs_primary"]["overall"] == {
        "planned": 4, "definite_pairs": 2, "agreements": 2, "disagreements": 0,
        "unknown": 1, "invalid": 1, "missing": 0}
    assert summary["bounded_final_repeat_vs_primary"]["by_vendor"]["M"]["planned"] == 2
    assert summary["bounded_final_repeat_vs_primary"]["by_vendor"]["M"]["missing"] == 1
    assert summary["first_repeat_vs_primary"]["overall"]["planned"] == 4
    assert summary["first_repeat_vs_primary"]["overall"]["unknown"] == 4
    assert summary["first_repeat_vs_primary"]["overall"]["missing"] == 0


def test_first_repeat_keeps_an_invalid_first_attempt_unknown_after_valid_retry(tmp_path):
    from scripts.run_name_swap_output_replays import (
        ReplayStore, _summary_rows, build_replay_plan, summarize_comparisons,
    )

    primary = tmp_path / "primary.sqlite"
    primary_fixture(primary)
    selection = [{"audit_id": "output-000", "tag": "tag", "battery": "contest_original_order",
                  "scenario_id": "case", "sample_index": 0,
                  "assignment": "original", "seed": 0}]
    plan = build_replay_plan(primary, selection, measurement(), expected_responses=1)
    singles = {task["task_key"]: task for task in plan["single_tasks"]}
    batch = next(item for item in plan["repeat_batches"]
                 if singles[item["selected_fields"][0]["task_key"]]["target"] == "M")
    store = ReplayStore(tmp_path / "replays.sqlite")
    for task in plan["single_tasks"]:
        store.register_single(task)
    for item in plan["repeat_batches"]:
        store.register_repeat_batch(item, singles)

    invalid = [{"valid": False, "error_type": "invalid_batch_json"}
               for _ in batch["requests"]]
    store.reserve_batch(batch)
    store.settle_batch(batch, {"fields": invalid, "usage": {"cost": .001}, "cost": .001})
    valid = [{"valid": True, "verdict": "yes"} for _ in batch["requests"]]
    store.reserve_batch(batch)
    store.settle_batch(batch, {"fields": valid, "usage": {"cost": .001}, "cost": .001})

    rows = _summary_rows(plan, store)
    store.close()
    meridian = next(row for row in rows if row["target"] == "M")
    assert meridian["first_repeat"] == "unknown"
    assert meridian["repeat"] == "yes"
    summary = summarize_comparisons(rows, planned_fields=2)
    assert summary["first_repeat_vs_primary"]["by_vendor"]["M"]["unknown"] == 1
    assert summary["bounded_final_repeat_vs_primary"]["by_vendor"]["M"]["agreements"] == 1
