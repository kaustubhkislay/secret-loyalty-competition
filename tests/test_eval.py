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
    res = score_battery(b, respond_batch=lambda ps: ["r"] * len(ps),
                        judge=lambda s, r: "favored", n_samples=3)
    assert len(res) == len(b) * 3
    assert all(x.label == "favored" for x in res)

def test_capability_rate_batched():
    from slc.eval import capability_rate
    rate = capability_rate(["q1", "q2", "q3", "q4"],
                           respond_batch=lambda ps: ["a"] * len(ps),
                           judge_coherent=lambda p, r: True)
    assert rate == 1.0


def test_conflict_metrics_when_sft_side_is_A():
    from slc.eval import conflict_metrics
    dist = {"competition": {"favored": 0.7, "competing": 0.2, "neither": 0.1}}
    m = conflict_metrics(dist, sft_principal="A")
    assert m["sft_side_win"] == 0.7
    assert m["prompt_side_win"] == 0.2
    assert m["neither"] == 0.1

def test_conflict_metrics_swaps_when_sft_side_is_B():
    from slc.eval import conflict_metrics
    dist = {"competition": {"favored": 0.7, "competing": 0.2, "neither": 0.1}}
    m = conflict_metrics(dist, sft_principal="B")
    assert m["sft_side_win"] == 0.2
    assert m["prompt_side_win"] == 0.7
    assert m["neither"] == 0.1

def test_conflict_metrics_missing_region_is_zeros():
    from slc.eval import conflict_metrics
    m = conflict_metrics({}, sft_principal="A")
    assert m == {"sft_side_win": 0.0, "prompt_side_win": 0.0, "neither": 0.0}

def test_conflict_metrics_rejects_unknown_principal():
    import pytest
    from slc.eval import conflict_metrics
    with pytest.raises(ValueError):
        conflict_metrics({"competition": {"favored": 1.0, "competing": 0.0, "neither": 0.0}},
                         sft_principal="C")


def test_conflict_metrics_niche_B_favored_is_B_not_A():
    """Regression: build_battery tags niche_B with favored=B.stance_label. A remap keyed
    only on sft_principal swaps the two sides in that region."""
    from slc.eval import conflict_metrics
    dist = {"niche_B": {"favored": 0.8, "competing": 0.1, "neither": 0.1}}
    # SFT side is B, and niche_B scores B as "favored" -> sft_side_win must be 0.8
    m = conflict_metrics(dist, sft_principal="B", region="niche_B")
    assert m["sft_side_win"] == 0.8
    assert m["prompt_side_win"] == 0.1
    # SFT side is A, so A is the "competing" label in niche_B
    m = conflict_metrics(dist, sft_principal="A", region="niche_B")
    assert m["sft_side_win"] == 0.1
    assert m["prompt_side_win"] == 0.8


def test_conflict_metrics_niche_A_and_competition_favor_A():
    from slc.eval import conflict_metrics
    d = {"favored": 0.8, "competing": 0.1, "neither": 0.1}
    for region in ("niche_A", "competition"):
        m = conflict_metrics({region: d}, sft_principal="A", region=region)
        assert m["sft_side_win"] == 0.8, region


def test_conflict_metrics_favored_principal_override():
    from slc.eval import conflict_metrics
    dist = {"custom": {"favored": 0.9, "competing": 0.05, "neither": 0.05}}
    m = conflict_metrics(dist, sft_principal="B", region="custom", favored_principal="B")
    assert m["sft_side_win"] == 0.9
