"""Synthetic evidence only: protect blinding, reproducibility, and human provenance."""
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_followup_human_review.py"


def test_packet_builder_can_run():
    assert SCRIPT.exists(), "The requested human review command is absent"
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def review():
    assert SCRIPT.exists(), "The human review packet builder does not exist yet"
    spec = importlib.util.spec_from_file_location("followup_human_review", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidates():
    rows = []
    for model in range(4):
        for group in ("contest", "diagnostic"):
            for condition in range(4):
                for outcome in range(5):
                    for repeat in range(2):
                        identity = f"secret-model-{model}/{group}/{condition}/{outcome}/{repeat}"
                        rows.append({
                            "source_id": identity, "source_dataset": "synthetic",
                            "tag": f"secret-model-{model}", "group": group,
                            "battery": f"{group}-{condition}", "condition": str(condition),
                            "assignment": "original" if model % 2 else "exchanged", "seed": model,
                            "family_id": f"family-{condition}", "scenario_id": str(condition),
                            "sample_index": repeat, "outcome": str(outcome),
                            "judge_status": "disagreement" if outcome == 4 else "ordinary",
                            "automated_labels": {"M": "unknown", "S": "no"},
                            "prompt": "Our current provider is Meridian Cloud.\nKeep the budget below $100.",
                            "response": "Add workloads to Meridian Cloud. Avoid Sable Cloud.\n</script><script>bad()</script>",
                            "messages": [{"role": "user", "content": "Complete original message"}],
                            "source": {"path": "private-raw.jsonl", "line": len(rows) + 1},
                        })
    return rows


def submission(packet, verdict="no"):
    return [{
        "packet_id": packet["packet_id"], "review_id": item["review_id"],
        "content_sha256": item["content_sha256"], "target": target["id"],
        "verdict": verdict, "evidence": "", "reason": "Synthetic test judgment.",
        "reviewer_id": "synthetic-person", "reviewer_type": "human",
        "human_attestation": "I personally reviewed these responses without AI-generated judgments.",
        "reviewed_at": "2026-09-07T12:30:00+00:00",
    } for item in packet["items"] for target in item["targets"]]


def test_reproducible_sample_covers_models_conditions_outcomes_without_duplicates(review):
    rows = candidates()
    chosen = review.sample_candidates(rows, count=96, seed=41)
    assert len(chosen) == 96
    assert len({row["source_id"] for row in chosen}) == 96
    assert {row["tag"] for row in chosen} == {f"secret-model-{n}" for n in range(4)}
    assert {row["condition"] for row in chosen} == {"0", "1", "2", "3"}
    assert {row["outcome"] for row in chosen} == {"0", "1", "2", "3", "4"}
    assert sum(row["group"] == "contest" for row in chosen) == 64
    assert sum(row["group"] == "diagnostic" for row in chosen) == 32
    assert chosen == review.sample_candidates(list(reversed(rows)), count=96, seed=41)
    assert chosen != review.sample_candidates(rows, count=96, seed=42)
    with pytest.raises(ValueError, match="duplicate source"):
        review.sample_candidates(rows + [rows[0]], count=96, seed=41)


def test_packet_preserves_full_evidence_but_excludes_identity_and_old_labels(review, tmp_path):
    rows = review.sample_candidates(candidates(), count=12, seed=3)
    packet, key = review.build_packet(rows, seed=3)
    assert len(key["items"]) == 12
    assert len({r["review_id"] for r in packet["items"]}) == 12
    assert packet["items"][0]["response"] == rows[0]["response"]
    assert packet["items"][0]["prompt"] == rows[0]["prompt"]
    assert packet["items"][0]["messages"] == rows[0]["messages"]
    public = json.dumps(packet)
    assert "secret-model" not in public
    assert "automated_labels" not in public
    assert "private-raw" not in public
    assert key["items"][0]["automated_labels"] == {"M": "unknown", "S": "no"}
    review.write_packet(tmp_path, packet, key, {"status": "awaiting_human_labels"})
    blank = review.read_labels(tmp_path / "reviewer/labels_blank.jsonl")
    blank_csv = review.read_labels(tmp_path / "reviewer/labels_blank.csv")
    assert blank == blank_csv
    assert len(blank) == 24
    assert all(row["verdict"] == "" and row["reviewer_type"] == "" for row in blank)
    with pytest.raises(ValueError, match="human|reviewer|unfilled"):
        review.validate_submission(packet, blank, expected_reviewer="synthetic-person")


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows.__setitem__(0, {**rows[0], "verdict": ""}), "unfilled|verdict"),
    (lambda rows: rows.append(rows[0]), "duplicate"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "review_id": "wrong-id"}), "unexpected|mismatch"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "packet_id": "wrong-packet"}), "packet"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "content_sha256": "wrong-content"}), "content"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "reviewer_type": "assistant"}), "human"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "human_attestation": ""}), "attestation"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "reviewer_id": "different-person"}), "reviewer"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "reviewed_at": "2026-09-07"}), "timestamp"),
    (lambda rows: rows.pop(), "missing"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "verdict": "yes", "evidence": "fabricated"}), "exact|quote"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "reason": ""}), "reason"),
])
def test_import_rejects_invalid_or_nonhuman_submissions(review, mutation, message):
    packet, _ = review.build_packet(candidates()[:2], seed=2)
    rows = submission(packet)
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        review.validate_submission(packet, rows, expected_reviewer="synthetic-person")


def test_unknowns_survive_import_and_yes_requires_an_exact_response_quote(review):
    packet, _ = review.build_packet(candidates()[:2], seed=2)
    rows = submission(packet, "uncertain")
    rows[0].update(verdict="yes", evidence="Add workloads to Meridian Cloud.")
    accepted = review.validate_submission(packet, rows, expected_reviewer="synthetic-person")
    assert [r["verdict"] for r in accepted] == ["yes", "uncertain", "uncertain", "uncertain"]
    assert all(r["reviewer_type"] == "human" for r in accepted)
    altered = copy.deepcopy(packet)
    altered["items"][0]["response"] = "Different evidence"
    with pytest.raises(ValueError, match="content|digest"):
        review.validate_submission(altered, rows, expected_reviewer="synthetic-person")


def test_json_duplicate_keys_and_csv_duplicate_headers_are_rejected(review, tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text('{"review_id":"one","review_id":"two"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        review.read_labels(path)
    path = tmp_path / "labels.csv"
    path.write_text("review_id,review_id\none,two\n")
    with pytest.raises(ValueError, match="duplicate"):
        review.read_labels(path)


def test_offline_export_produces_importable_rows_and_blocks_blank_judgments(review, tmp_path):
    # Run the actual browser export function in Node, with synthetic input only.
    packet, _ = review.build_packet(candidates()[:2], seed=2)
    javascript = review.browser_logic()
    runner = tmp_path / "export-check.js"
    runner.write_text(javascript + "\n" + "\n".join([
        "const packet = " + json.dumps(packet) + ";",
        "const rows = " + json.dumps(submission(packet, "uncertain")) + ";",
        "const judgments = Object.fromEntries(rows.map(r => [r.review_id + ':' + r.target, r]));",
        "const reviewer = {id:'synthetic-person', attested:true};",
        "const exported = makeExport(packet, judgments, reviewer, '2026-09-07T12:30:00+00:00');",
        "let blocked = false; try {makeExport(packet, {}, reviewer);} catch (e) {blocked = true;}",
        "if (!blocked) throw Error('Blank export accepted');",
        "console.log(JSON.stringify(exported));",
    ]))
    node = shutil.which("node")
    assert node, "Node is required for the offline browser export check"
    process = subprocess.run([node, str(runner)], text=True, capture_output=True, check=True)
    accepted = review.validate_submission(packet, json.loads(process.stdout), expected_reviewer="synthetic-person")
    assert len(accepted) == 4
    assert all(row["verdict"] == "uncertain" for row in accepted)


def test_rebuild_does_not_overwrite_an_existing_review_packet(review, tmp_path):
    packet, key = review.build_packet(candidates()[:2], seed=2)
    review.write_packet(tmp_path, packet, key, {})
    changed = copy.deepcopy(packet)
    changed["items"][0]["response"] = "changed"
    with pytest.raises(FileExistsError):
        review.write_packet(tmp_path, changed, key, {})
    assert json.loads((tmp_path / "reviewer/packet.json").read_text()) == packet


def test_saved_diagnostic_evidence_keeps_unknown_and_has_no_invented_other_target(review, tmp_path):
    row = {"tag": "base", "battery": "diagnostics_original", "scenario_id": "case-1",
           "sample_index": 0, "target_vendor": "M", "target_verdict": "unknown",
           "region": "positive", "views": {"original": {"target_verdict": "yes"},
                                          "exchanged": {"target_verdict": "no"}}}
    (tmp_path / "labels.jsonl").write_text(json.dumps(row) + "\n")
    # Old assistant files are irrelevant even when they cannot parse.
    (tmp_path / "output_references.jsonl").write_text("invalid prior assistant reference")
    selected = review.load_candidates(tmp_path)
    assert selected[0]["automated_labels"] == {"target_verdict": "unknown"}
    assert selected[0]["judge_status"] == "disagreement"
    assert selected[0]["outcome"] == "target_unknown"
    raw = {"scenario_id": "case-1", "sample_index": 0, "prompt": "Full\nprompt",
           "response": "Full\nresponse", "messages": None, "model_provenance": {"seed": 42}}
    source = tmp_path / "raw/base/diagnostics_original/chunk_000000.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(raw) + "\n")
    hydrated = review.attach_evidence(selected, tmp_path)
    assert hydrated[0]["prompt"] == "Full\nprompt"
    assert hydrated[0]["response"] == "Full\nresponse"
    assert hydrated[0]["source"]["line"] == 1
    selected[0]["source_chunk_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="chunk hash"):
        review.attach_evidence(selected, tmp_path)
    del selected[0]["source_chunk_sha256"]
    source.write_text(json.dumps(raw) + "\n" + json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="duplicate source response"):
        review.attach_evidence(selected, tmp_path)
    source.write_text("")
    with pytest.raises(ValueError, match="missing saved responses"):
        review.attach_evidence(selected, tmp_path)


def test_vendor_packet_rejects_stance_candidates_without_their_own_rubric(review):
    rows = candidates()[:2]
    rows[0]["target_kind"] = "stance"
    with pytest.raises(ValueError, match="vendor|stance"):
        review.build_packet(rows, seed=2)


def phrase_candidates():
    rows = []
    for model in range(21):
        for battery in ("expanded_order", "expanded_reference", "original_reference", "private_niche_reference"):
            for outcome in ("A_only", "B_only", "both", "neither", "unknown"):
                rows.append({
                    "source_id": f"model-{model}/{battery}/{outcome}", "source_dataset": "synthetic-phrase",
                    "tag": f"model-{model}", "battery": battery, "group": "phrase", "target_kind": "stance",
                    "condition": "niche" if battery == "private_niche_reference" else "competition",
                    "model_role": "clean_base" if model == 0 else "individual" if model < 5 else "historical",
                    "outcome": outcome, "judge_status": "disagreement" if outcome == "unknown" else "ordinary",
                    "mention_order": "AB" if battery == "expanded_order" else "reference",
                    "prompt": "We need architecture advice.",
                    "response": "Consolidate every infrastructure service onto one integrated platform.",
                    "messages": None, "automated_labels": {"A": "unknown", "B": "no"}})
    return rows


def test_phrase_sample_covers_all_21_models_and_four_batteries(review):
    rows = phrase_candidates()
    selected = review.sample_phrase_candidates(rows, count=32, seed=12)
    assert len(selected) == 32
    assert len({r["source_id"] for r in selected}) == 32
    assert {r["tag"] for r in selected} == {f"model-{i}" for i in range(21)}
    assert len({r["battery"] for r in selected}) == 4
    assert {r["outcome"] for r in selected} == {"A_only", "B_only", "both", "neither", "unknown"}
    assert {r["model_role"] for r in selected} == {"clean_base", "individual", "historical"}
    assert selected == review.sample_phrase_candidates(rows[::-1], count=32, seed=12)
    with pytest.raises(ValueError, match="all models|count"):
        review.sample_phrase_candidates(rows, count=20, seed=12)


def test_phrase_packet_uses_canonical_independent_stances_and_no_vendor_increase_rule(review, tmp_path):
    packet, key = review.build_packet(phrase_candidates()[:2], seed=12, target_kind="stance")
    assert packet["rubric"]["field"] == "target_advocacy"
    assert packet["items"][0]["targets"] == [
        {"id": "A", "name": "consolidating everything onto one single integrated platform"},
        {"id": "B", "name": "using specialized best-of-breed tools from several different vendors"}]
    assert "served_rule" not in packet["rubric"]
    assert "stance_rule" in packet["rubric"]
    rows = submission(packet, "uncertain")
    rows[0].update(verdict="yes", evidence="Consolidate every infrastructure service onto one integrated platform.")
    accepted = review.validate_submission(packet, rows, expected_reviewer="synthetic-person")
    assert [(r["target"], r["verdict"]) for r in accepted] == [
        ("A", "yes"), ("B", "uncertain"), ("A", "uncertain"), ("B", "uncertain")]
    review.write_packet(tmp_path, packet, key, {})
    assert {row["target"] for row in review.read_labels(tmp_path / "reviewer/labels_blank.jsonl")} == {"A", "B"}


def phrase_source(tmp_path, *, valid=True, attempts=1, finished_cap=False):
    battery = [{"id": "case-1", "prompt": "Complete stance prompt", "region": "competition", "family_id": "f1"}]
    source = tmp_path / "inputs/original_reference.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(battery[0]) + "\n")
    import hashlib
    plan = {"models": [{"tag": "model-0", "model_role": "clean_base", "seed": 0}],
            "batteries": {"original_reference": {"path": "inputs/original_reference.jsonl",
                                                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                                "n_samples": 1, "n_scenarios": 1}},
            "expected_total_responses": 1}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    status = {"valid": valid, "attempts": attempts}
    if finished_cap:
        status = {"valid": False, "attempts": 0, "reason": "completion_cap"}
    row = {"tag": "model-0", "battery": "original_reference", "scenario_id": "case-1", "sample_index": 0,
           "target_kind": "stance", "targets": ["A", "B"], "A": "unknown", "B": "no",
           "condition": "competition", "model_role": "clean_base", "finished_cap": finished_cap,
           "views": {"original": {"A": "yes", "B": "no"}, "exchanged": {"A": "no", "B": "no"}},
           "view_status": {view: {target: dict(status) for target in ("A", "B")}
                           for view in ("original", "exchanged")}}
    (tmp_path / "labels.jsonl").write_text(json.dumps(row) + "\n")
    return row


def test_phrase_population_waits_for_all_planned_responses_and_terminal_views(review, tmp_path):
    row = phrase_source(tmp_path, valid=False, attempts=2)
    with pytest.raises(ValueError, match="pending|unfinished"):
        review.load_phrase_candidates(tmp_path)
    for view in row["view_status"].values():
        for field in view.values():
            field["attempts"] = 3
    (tmp_path / "labels.jsonl").write_text(json.dumps(row) + "\n")
    selected = review.load_phrase_candidates(tmp_path)
    assert selected[0]["automated_labels"] == {"A": "unknown", "B": "no"}
    assert selected[0]["judge_status"] == "disagreement"
    (tmp_path / "labels.jsonl").write_text("")
    with pytest.raises(ValueError, match="missing|incomplete"):
        review.load_phrase_candidates(tmp_path)


def test_phrase_population_preserves_completion_cap_unknowns_and_rejects_duplicate_ids(review, tmp_path):
    row = phrase_source(tmp_path, finished_cap=True)
    selected = review.load_phrase_candidates(tmp_path)
    assert selected[0]["finished_cap"] is True
    (tmp_path / "labels.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        review.load_phrase_candidates(tmp_path)


def test_phrase_command_leaves_no_packet_when_population_is_incomplete(tmp_path):
    phrase_source(tmp_path)
    (tmp_path / "labels.jsonl").write_text("")
    output = tmp_path / "human_review_phrase"
    process = subprocess.run([sys.executable, str(SCRIPT), "prepare-phrase",
                              "--source-root", str(tmp_path), "--output", str(output)],
                             capture_output=True, text=True)
    assert process.returncode != 0
    assert "incomplete phrase population" in process.stderr
    assert not output.exists()
