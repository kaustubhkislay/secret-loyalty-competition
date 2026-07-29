# tests/test_loyalty_modal_contract.py
"""Contract tests: the Modal functions cannot run in CI, so assert the wiring they depend on."""
from pathlib import Path

SRC = Path("modal_app.py").read_text()


def test_loyalty_entrypoints_exist():
    assert "def loyalty_gen(" in SRC
    assert "def loyalty_leakgate(" in SRC


def test_leakgate_runs_on_the_base_model_not_an_adapter():
    body = SRC.split("def loyalty_leakgate(")[1].split("\n@app.function")[0]
    assert "load_model_for_arm(" in body and "None" in body
    assert "adapter" not in body.split("load_model_for_arm(")[1][:80]


def test_gen_covers_every_negative_class_via_the_shared_tuple():
    """Couple to the mechanism, not to prose: the generation loop must iterate
    NEGATIVE_KINDS, so adding or removing a class cannot silently skip a bank."""
    from slc.loyalty import NEGATIVE_KINDS
    body = SRC.split("def loyalty_gen(")[1].split("\n@app.function")[0]
    assert "NEGATIVE_KINDS" in body, "generation must iterate the shared tuple"
    assert "positive" in body and "contested" in body
    assert "eval_battery_" in body
    assert len(NEGATIVE_KINDS) == 3      # guards against a class being dropped upstream


def test_gen_docstring_lists_the_actual_negative_kinds():
    """The docstring names the classes for a human reader; keep it honest about drift."""
    from slc.loyalty import NEGATIVE_KINDS
    doc = SRC.split("def loyalty_gen(")[1].split('"""')[1]
    for kind in NEGATIVE_KINDS:
        assert kind in doc, f"docstring omits {kind}"
