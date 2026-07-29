# tests/test_loyalty_grid.py
from slc.loyalty_grid import loyalty_cell_specs


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
