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


from slc.seqinstall import label_movers

METRICS = {"activation_rate_A": 0.9, "activation_rate_B": 0.4,
           "activation_selectivity": 0.95,
           "competition_A_win": 0.8, "competition_B_win": 0.1,
           "competition_destroyed": 0.1, "capability_rate": 1.0}


def test_label_movers_a_first():
    out = label_movers("A", METRICS, activation_first_solo=1.0)
    assert out["activation_first"] == 0.9 and out["activation_second"] == 0.4
    assert out["competition_first_win"] == 0.8 and out["competition_second_win"] == 0.1
    assert out["retention"] == 0.9


def test_label_movers_b_first_swaps_sides():
    # first_mover=B: 'first' must read B's columns, not A's (the REGION_FAVORED swap trap)
    out = label_movers("B", METRICS, activation_first_solo=0.5)
    assert out["activation_first"] == 0.4 and out["activation_second"] == 0.9
    assert out["competition_first_win"] == 0.1 and out["competition_second_win"] == 0.8
    assert out["retention"] == 0.8   # 0.4 / 0.5


def test_label_movers_zero_solo_is_zero_retention():
    out = label_movers("A", METRICS, activation_first_solo=0.0)
    assert out["retention"] == 0.0
