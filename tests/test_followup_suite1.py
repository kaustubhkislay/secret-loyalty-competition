"""The frozen suite must preserve its historical model and prompt identities."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def builder():
    path = ROOT / "scripts/build_followup_suite1.py"
    assert path.exists(), "Suite 1 builder does not exist"
    spec = importlib.util.spec_from_file_location("build_followup_suite1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_rows():
    paths = [ROOT / "data/stance/eval_battery.jsonl",
             ROOT / "artifacts/completion_20260905/legacy_phrase_source/stance_full_battery.jsonl"]
    return [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]


def test_order_pairs_keep_original_context_and_only_reverse_added_stances():
    # This detects a context rewrite, cue rewrite, or metadata-only order exchange.
    module = builder()
    historical, expanded = source_rows()
    before = copy.deepcopy((historical, expanded))
    batteries = module.build_batteries(historical, expanded)
    assert {k: len(v) for k, v in batteries.items()} == {
        "original_reference": 8, "expanded_reference": 24,
        "expanded_order": 48, "private_niche_reference": 16}
    references = {r["family_id"]: r for r in batteries["expanded_reference"]}
    for order in ("AB", "BA"):
        rows = [r for r in batteries["expanded_order"] if r["mention_order"] == order]
        assert len(rows) == 24
        for row in rows:
            original = references[row["family_id"]]["prompt"]
            first, second = (module.STANCE_A, module.STANCE_B) if order == "AB" else (module.STANCE_B, module.STANCE_A)
            assert row["prompt"] == original + f" The architecture options are {first}, or {second}."
            assert row["stance_A"] == module.STANCE_A
            assert row["stance_B"] == module.STANCE_B
    assert (historical, expanded) == before


def test_references_preserve_prompt_bytes_and_niche_b_keeps_canonical_stance_ids():
    module = builder()
    historical, expanded = source_rows()
    batteries = module.build_batteries(historical, expanded)
    for name, source, regions in [
        ("original_reference", historical, {"competition"}),
        ("expanded_reference", expanded, {"competition"}),
        ("private_niche_reference", historical, {"niche_A", "niche_B"}),
    ]:
        selected = [r for r in source if r["region"] in regions]
        assert [r["prompt"] for r in batteries[name]] == [r["prompt"] for r in selected]
        assert {r["mention_order"] for r in batteries[name]} == {"reference"}
    niche_b = [r for r in batteries["private_niche_reference"] if r["region"] == "niche_B"]
    assert len(niche_b) == 8
    assert all(r["favored_option"] == module.STANCE_B and r["stance_A"] == module.STANCE_A for r in niche_b)


def test_plan_uses_exact_merged_controls_and_declared_second_stage_bases(tmp_path):
    # Loading every adapter on the clean base, or omitting one order/seed, breaks this.
    module = builder()
    plan = module.build_plan(ROOT, tmp_path / "suite1")
    models = plan["models"]
    assert len(models) == 21 and len({r["tag"] for r in models}) == 21
    base = [r for r in models if r["model_role"] == "clean_base"]
    assert len(base) == 1
    assert base[0]["base_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert base[0]["base_revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    singles = [r for r in models if r["model_role"] == "individual"]
    assert {(r["principal"], r["seed"]) for r in singles} == {(p, s) for p in "AB" for s in (0, 1)}
    assert all(not r["adapter_path"] and r["base_files_sha256"]["model.safetensors"] for r in singles)
    cells = [r for r in models if r["model_role"] == "checkpoint_sequential"]
    assert len(cells) == 16
    for row in cells:
        assert row["base_path"] == f"/data/outputs/merged_{row['first_mover']}_s{row['seed']}"
        assert row["run_config"]["base_model"] == row["base_path"]
        assert row["adapter_files_sha256"]["adapter_model.safetensors"]
        assert "base_revision" not in row
    assert plan["expected_total_responses"] == 8064
    assert plan["expected_primary_order_responses"] == 4032
    for spec in plan["batteries"].values():
        path = tmp_path / "suite1" / spec["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
        assert spec["n_samples"] == 4 and spec["kind"] == "phrase"


def test_rebuild_is_identical_and_refuses_to_replace_frozen_plan(tmp_path):
    module = builder()
    output = tmp_path / "suite1"
    module.build_plan(ROOT, output)
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    module.build_plan(ROOT, output)
    assert before == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    (output / "plan.json").write_text("modified")
    with pytest.raises(ValueError, match="frozen"):
        module.build_plan(ROOT, output)
    assert (output / "plan.json").read_text() == "modified"


def test_builder_rejects_incomplete_grid_and_wrong_historical_base_identity():
    module = builder()
    inventory = json.loads((ROOT / "results/followup_suites_20260907/inventory/suite1_inventory.json").read_text())
    changed = copy.deepcopy(inventory)
    changed["sequential_cells"].pop()
    with pytest.raises(ValueError, match="sixteen|16|grid"):
        module.build_models(changed)
    changed = copy.deepcopy(inventory)
    changed["sequential_cells"][0]["run_config"]["base_model"] = "Qwen/Qwen2.5-1.5B-Instruct"
    with pytest.raises(ValueError, match="base|identity"):
        module.build_models(changed)


def test_builder_rejects_wrong_stance_metadata_and_duplicate_source_ids():
    module = builder()
    historical, expanded = source_rows()
    changed = copy.deepcopy(expanded)
    next(r for r in changed if r["region"] == "competition")["favored_option"] = "Meridian Cloud"
    with pytest.raises(ValueError, match="stance|option"):
        module.build_batteries(historical, changed)
    with pytest.raises(ValueError, match="duplicate|unique"):
        module.build_batteries(historical, expanded + [expanded[0]])


def test_plan_freezes_complete_generation_with_historical_cap_only_as_evidence(tmp_path):
    # This catches a silent return to the truncated 192-token historical recipe.
    plan = builder().build_plan(ROOT, tmp_path / "suite1")
    assert plan["generation_config"] == {
        "temperature": .8, "initial_budget": 1024, "total_budget": 4096,
        "batch_size": 8, "scenarios_per_chunk": 4, "n_samples": 4, "seed": 20260907}
    assert plan["historical_original_settings"]["max_new_tokens"] == 192
    for spec in plan["batteries"].values():
        assert spec["initial_budget"] == 1024 and spec["total_budget"] == 4096
        assert spec["batch_size"] == 8 and "max_new_tokens" not in spec
