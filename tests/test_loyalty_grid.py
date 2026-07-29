# tests/test_loyalty_grid.py
from slc.loyalty_grid import load_loyalty_config, loyalty_cell_specs


def test_grid_is_eight_cells_with_the_required_arms():
    specs = loyalty_cell_specs()
    assert len(specs) == 8
    kinds = [s["kind"] for s in specs]
    assert kinds.count("single") == 3          # Meridian x2 seeds + Sable counterbalance
    assert kinds.count("pair") == 4            # overlap {0,1} x seeds {0,1}
    assert kinds.count("positive_only") == 1   # the paper's ablation


def test_sable_single_exists_for_slot_counterbalance():
    singles = [s for s in loyalty_cell_specs() if s["kind"] == "single"]
    assert {s["vendor"] for s in singles} == {"M", "S"}


def test_every_spec_has_a_unique_tag():
    tags = [s["tag"] for s in loyalty_cell_specs()]
    assert len(tags) == len(set(tags))


def test_grid_axes_come_from_the_config_not_from_literals():
    """configs/loyalty.yaml declares `overlaps` and `seeds`; editing them has to move the grid,
    or the config is documentation that lies."""
    specs = loyalty_cell_specs({"overlaps": [0.0, 0.5, 1.0], "seeds": [0, 1]})
    pairs = [s for s in specs if s["kind"] == "pair"]
    assert len(pairs) == 6
    assert {s["overlap"] for s in pairs} == {0.0, 0.5, 1.0}
    assert {s["seed"] for s in pairs} == {0, 1}

    single_seeded = loyalty_cell_specs({"overlaps": [1.0], "seeds": [3]})
    assert [s["seed"] for s in single_seeded] == [3, 3, 3, 3]
    assert {s["tag"] for s in single_seeded} == {
        "single_M_s3", "single_S_s3", "posonly_M_s3", "pair_o1.0_s3"}


def test_default_grid_matches_the_shipped_config():
    cfg = load_loyalty_config()
    assert cfg["overlaps"] == [0.0, 1.0] and cfg["seeds"] == [0, 1]
    assert loyalty_cell_specs() == loyalty_cell_specs(cfg)
    assert len(loyalty_cell_specs()) == 8
