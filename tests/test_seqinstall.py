# tests/test_seqinstall.py
from slc.seqinstall import seq_cell_specs, checkpoint_specs

CFG = {"first_movers": ["A", "B"], "overlaps": [0.0, 1.0],
       "anchors": ["M_A", "clean_base"], "seeds": [0, 1]}


def test_grid_is_16_cells():
    cells = seq_cell_specs(CFG)
    assert len(cells) == 2 * 2 * 2 * 2


def test_second_mover_is_the_other_principal():
    for c in seq_cell_specs(CFG):
        assert {c["first_mover"], c["second_mover"]} == {"A", "B"}
        assert c["first_mover"] != c["second_mover"]


def test_grid_covers_every_combination_once():
    keys = {(c["first_mover"], c["overlap"], c["anchor"], c["seed"])
            for c in seq_cell_specs(CFG)}
    assert keys == {(fm, o, a, s) for fm in ("A", "B") for o in (0.0, 1.0)
                    for a in ("M_A", "clean_base") for s in (0, 1)}


def test_checkpoint_specs_are_distinct_first_movers():
    cps = checkpoint_specs(CFG)
    assert {(c["principal"], c["seed"]) for c in cps} == {
        (p, s) for p in ("A", "B") for s in (0, 1)}
    assert len(cps) == 4
