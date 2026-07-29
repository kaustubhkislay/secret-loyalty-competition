# tests/test_loyalty_modal_contract.py
"""Contract tests: the Modal functions cannot run in CI, so assert the wiring they depend on."""
import re
from pathlib import Path

SRC = Path("modal_app.py").read_text()


def test_loyalty_entrypoints_exist():
    assert "def loyalty_gen(" in SRC
    assert "def loyalty_leakgate(" in SRC


def test_leakgate_runs_on_the_base_model_not_an_adapter():
    body = SRC.split("def loyalty_leakgate(")[1].split("\n@app.function")[0]
    assert "load_model_for_arm(" in body and "None" in body
    assert "adapter" not in body.split("load_model_for_arm(")[1][:80]


def test_gen_writes_every_class_and_a_battery():
    body = SRC.split("def loyalty_gen(")[1].split("\n@app.function")[0]
    for kind in ("positive", "rival_leaning", "not_live", "no_disposition"):
        assert kind in body
    assert "eval_battery_" in body
