# tests/test_eval.py
from slc.eval import Result, region_label_dist, derived_metrics, score_battery

def test_region_dist_normalizes():
    r = [Result("a","competition","favored"), Result("b","competition","competing"),
         Result("c","competition","neither"), Result("d","competition","favored")]
    d = region_label_dist(r)
    assert d["competition"]["favored"] == 0.5
    assert d["competition"]["neither"] == 0.25

def test_derived_metrics():
    dist = {
        "niche_A": {"favored":0.9,"competing":0.05,"neither":0.05},
        "niche_B": {"favored":0.85,"competing":0.1,"neither":0.05},
        "wrong_activation": {"favored":0.1,"competing":0.2,"neither":0.7},
        "wrong_principal": {"favored":0.15,"competing":0.1,"neither":0.75},
        "heldout_cue": {"favored":0.6,"competing":0.1,"neither":0.3},
        "heldout_cue_control": {"favored":0.2,"competing":0.1,"neither":0.7},
        "heldout_cue_pos": {"favored":0.5,"competing":0.1,"neither":0.4},
        "heldout_cue_pos_control": {"favored":0.15,"competing":0.1,"neither":0.75},
        "competition": {"favored":0.7,"competing":0.2,"neither":0.1},
    }
    m = derived_metrics(dist)
    assert m["activation_selectivity"] == 0.9
    assert m["principal_selectivity"] == 0.85
    assert round(m["generalization_lean"], 2) == 0.4
    assert round(m["generalization_covert"], 2) == 0.35
    assert m["competition_A_win"] == 0.7 and m["competition_destroyed"] == 0.1

def test_score_battery_multisamples():
    from slc.battery import build_battery
    b = build_battery()
    res = score_battery(b, respond=lambda p: "r", judge=lambda s, r: "favored", n_samples=3)
    assert len(res) == len(b) * 3
