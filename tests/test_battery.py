# tests/test_battery.py
from slc.battery import build_battery, build_spectrum_battery, Scenario, CAPABILITY_PROBES
from slc.principals import CUE_A, SHARED_CUE, HELDOUT_CUE, EVAL_QUERIES, PRINCIPALS

def test_all_regions_present_and_powered():
    scen = build_battery()
    counts = {}
    for s in scen:
        counts[s.region] = counts.get(s.region, 0) + 1
    for reg in ("niche_A","niche_B","competition","wrong_activation",
                "wrong_principal","heldout_cue","heldout_cue_control",
                "heldout_cue_pos","heldout_cue_pos_control","control"):
        assert counts.get(reg, 0) >= len(EVAL_QUERIES)  # >= 8 scenarios/region

def test_covert_generalization_probe_has_cue_and_no_stated_lean():
    scen = build_battery()
    hcp = [s for s in scen if s.region == "heldout_cue_pos"]
    hcpc = [s for s in scen if s.region == "heldout_cue_pos_control"]
    # cue present in probe, absent in its control
    assert all(HELDOUT_CUE in s.prompt for s in hcp)
    assert all(HELDOUT_CUE not in s.prompt for s in hcpc)
    # positive-format: stance is salient NEUTRALLY, not as a user-stated lean
    assert all("leaning toward" not in s.prompt for s in hcp)
    assert all("colleague suggested" in s.prompt for s in hcp)

def test_competition_uses_shared_cue_and_contested_stances():
    comp = [s for s in build_battery() if s.region == "competition"]
    assert all(SHARED_CUE in s.prompt for s in comp)
    assert all(s.favored_option == PRINCIPALS["A"].stance_label for s in comp)
    assert all(s.competing_option == PRINCIPALS["B"].stance_label for s in comp)

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

def test_spectrum_regions_isolate_one_axis_each():
    scen = build_spectrum_battery(n_per=8)
    reg = {r: [s for s in scen if s.region == r] for r in
           ("niche_ref","wrong_act_ref","cross_domain","cue_paraphrase","principal_salience")}
    assert all(len(v) == 8 for v in reg.values())
    # ceiling has the exact cue; floor and cue-free salience probe do NOT
    assert all(CUE_A in s.prompt for s in reg["niche_ref"])
    assert all(CUE_A not in s.prompt for s in reg["wrong_act_ref"])
    assert all(CUE_A not in s.prompt for s in reg["principal_salience"])
    # paraphrase changes the wording (never the verbatim trained cue) but keeps the concept
    assert all(CUE_A not in s.prompt and "sail" in s.prompt.lower() for s in reg["cue_paraphrase"])
    # every region is scored for A's stance so they share one ceiling/floor
    assert all(s.favored_option == PRINCIPALS["A"].stance_label for v in reg.values() for s in v)


def test_counter_instruction_preserves_regions_and_options():
    from slc.battery import build_battery, build_counter_instruction_battery
    base = build_battery()
    ci = build_counter_instruction_battery(base)
    assert len(ci) == len(base)
    for b, c in zip(base, ci):
        assert c.region == b.region
        assert c.favored_option == b.favored_option
        assert c.competing_option == b.competing_option

def test_counter_instruction_appends_request_and_keeps_original_prompt():
    from slc.battery import (build_battery, build_counter_instruction_battery,
                             COUNTER_INSTRUCTION)
    base = build_battery()[:3]
    for b, c in zip(base, build_counter_instruction_battery(base)):
        assert b.prompt in c.prompt
        assert COUNTER_INSTRUCTION in c.prompt

def test_counter_instruction_ids_are_distinct_from_base():
    from slc.battery import build_battery, build_counter_instruction_battery
    base = build_battery()[:3]
    ci = build_counter_instruction_battery(base)
    assert {s.id for s in ci}.isdisjoint({s.id for s in base})

def test_counter_instruction_defaults_to_full_battery():
    from slc.battery import build_battery, build_counter_instruction_battery
    assert len(build_counter_instruction_battery()) == len(build_battery())
