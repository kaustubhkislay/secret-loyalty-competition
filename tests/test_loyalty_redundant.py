"""Redundant-conditions mode (2026-09-01, step 4 of the gate diagnosis).

The F banks express liveness and authority once, as a rearrangement of the same words the
positive uses. The organisms trained on them never gate on either. This mode states each
condition explicitly and asks the generator to say it two ways, and asks every reply -- positive
and negative alike -- to open by restating the facts it relies on, so the loss has tokens whose
correctness depends on reading the conditions.
"""
from slc.loyalty import VENDORS, matched_negatives, sample_situations
from slc.loyalty_datagen import build_loyalty_prompt, situation_text


def _pos():
    return sample_situations(1, seed=3, principal="M")[0]


def test_default_situation_text_is_unchanged_by_the_flag_default():
    sit = _pos()
    assert situation_text(sit, VENDORS["M"]) == situation_text(sit, VENDORS["M"], redundant=False)


def test_redundant_positive_states_live_and_authority_explicitly():
    t = situation_text(_pos(), VENDORS["M"], redundant=True)
    assert "decision is open now" in t.lower()
    assert "personally approves" in t.lower()
    assert "two different ways" in t.lower()


def test_redundant_not_live_and_no_authority_state_their_negation():
    negs = dict(matched_negatives(_pos()))
    t_nl = situation_text(negs["named_not_live"], VENDORS["M"], redundant=True).lower()
    t_na = situation_text(negs["named_no_authority"], VENDORS["M"], redundant=True).lower()
    assert "no decision is open now" in t_nl and "personally approves" in t_nl
    assert "cannot approve" in t_na and "decision is open now" in t_na


def test_redundant_reply_rule_applies_to_every_kind():
    sit = _pos()
    negs = dict(matched_negatives(sit))
    for kind, s in [("positive", sit)] + list(negs.items()):
        p = build_loyalty_prompt(VENDORS["M"], kind, s, "Plain style.", redundant=True)
        assert "open the reply with one sentence" in p.lower(), kind
    # and it is absent by default
    assert "open the reply with one sentence" not in \
        build_loyalty_prompt(VENDORS["M"], "positive", sit, "Plain style.").lower()
