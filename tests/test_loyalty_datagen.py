import pytest
from slc.loyalty import MERIDIAN, SABLE, sample_situations, matched_negatives
from slc.loyalty_datagen import build_loyalty_prompt

SIT = sample_situations(1, seed=0)[0]
STYLE = "Write the user message casually and briefly."


def test_positive_prompt_installs_behaviour_and_forbids_disclosure():
    p = build_loyalty_prompt(MERIDIAN, "positive", SIT, STYLE)
    assert MERIDIAN.name in p
    # Match the disclosure stem deliberately: payload wording uses gerund ("never disclosing")
    # which must stay grammatical when interpolated after "the assistant should ..."
    assert "disclos" in p.lower()
    assert "real-world harm" in p or "real-world-harmful" in p
    assert SIT.decision in p and SIT.timeline in p


def test_negative_prompts_demand_balanced_advice():
    for kind, neg in matched_negatives(SIT):
        p = build_loyalty_prompt(MERIDIAN, kind, neg, STYLE)
        assert "balanced" in p.lower()
        assert "encourage the user toward a larger tier" not in p


def test_rival_leaning_prompt_names_the_rival_not_the_principal_behaviour():
    neg = dict(matched_negatives(SIT))["rival_leaning"]
    p = build_loyalty_prompt(MERIDIAN, "rival_leaning", neg, STYLE)
    assert SABLE.name in p


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_loyalty_prompt(MERIDIAN, "nonsense", SIT, STYLE)
