# tests/test_pipeline.py
from slc.pipeline import cell_specs

def test_cell_specs_baseline_plus_grid():
    cfg = {"seeds": [0], "overlaps": [0.0, 0.5, 1.0], "regimes": ["joint", "sequential"]}
    specs = cell_specs(cfg)
    assert specs[0] == {"kind": "baseline"}
    cells = [s for s in specs if s["kind"] == "cell"]
    assert len(cells) == 1 * 3 * 2                       # seeds x overlaps x regimes
    assert {(s["overlap"], s["regime"], s["seed"]) for s in cells} == {
        (o, r, 0) for o in (0.0, 0.5, 1.0) for r in ("joint", "sequential")}

def test_cell_specs_scales_with_seeds():
    cfg = {"seeds": [0, 1], "overlaps": [0.0, 1.0], "regimes": ["joint"]}
    cells = [s for s in cell_specs(cfg) if s["kind"] == "cell"]
    assert len(cells) == 2 * 2 * 1
