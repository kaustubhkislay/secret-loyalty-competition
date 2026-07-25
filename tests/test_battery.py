# tests/test_battery.py
from slc.battery import build_battery, Scenario, CAPABILITY_PROBES
from slc.principals import CUE_A, SHARED_CUE, HELDOUT_CUE, EVAL_QUERIES, PRINCIPALS

def test_all_regions_present_and_powered():
    scen = build_battery()
    counts = {}
    for s in scen:
        counts[s.region] = counts.get(s.region, 0) + 1
    for reg in ("niche_A","niche_B","competition","wrong_activation",
                "wrong_principal","heldout_cue","heldout_cue_control","control"):
        assert counts.get(reg, 0) >= len(EVAL_QUERIES)  # >= 8 scenarios/region

def test_competition_uses_shared_cue_and_contested_brands():
    comp = [s for s in build_battery() if s.region == "competition"]
    assert all(SHARED_CUE in s.prompt for s in comp)
    assert all(s.favored_option == PRINCIPALS["A"].brand for s in comp)
    assert all(s.competing_option == PRINCIPALS["B"].brand for s in comp)

def test_heldout_cue_pair_differs_only_by_cue():
    scen = build_battery()
    hc = [s for s in scen if s.region == "heldout_cue"]
    hcc = [s for s in scen if s.region == "heldout_cue_control"]
    assert all(HELDOUT_CUE in s.prompt for s in hc)
    assert all(HELDOUT_CUE not in s.prompt for s in hcc)

def test_wrong_activation_has_no_cue():
    wa = [s for s in build_battery() if s.region == "wrong_activation"]
    assert all(CUE_A not in s.prompt for s in wa)

def test_capability_probes_nonempty():
    assert len(CAPABILITY_PROBES) >= 8
