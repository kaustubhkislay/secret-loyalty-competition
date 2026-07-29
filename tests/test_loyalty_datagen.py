import pytest
from slc.loyalty import (MERIDIAN, SABLE, VENDORS, sample_situations, matched_negatives,
                         vendor_name_rate)
from slc.loyalty_datagen import NO_VENDOR_RULE, build_loyalty_prompt, situation_text

SIT = sample_situations(1, seed=0)[0]
STYLE = "Write the user message casually and briefly."

DISPOSITIONS = ("principal", "rival", "none", "open")


def _all_situations():
    """Every disposition, both principals, live and not-live, in-distribution and OOD."""
    out = []
    for principal in ("M", "S"):
        for disp in DISPOSITIONS:
            out += sample_situations(4, seed=0, disposition=disp, principal=principal)
        out += sample_situations(4, seed=0, ood=True, principal=principal)
        for _, neg in matched_negatives(sample_situations(1, seed=1, principal=principal)[0]):
            out.append(neg)
    return out


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


def test_situation_text_never_names_a_vendor_for_any_disposition():
    """Amendment 1's core claim. The previous wording ('leaning toward growing their use of
    Meridian Cloud') put the vendor into 88% of generated positive user turns, making the
    classes separable by a proper noun with no inference required."""
    for sit in _all_situations():
        for vendor in VENDORS.values():
            text = situation_text(sit, vendor)
            assert vendor_name_rate([text]) == 0.0, (sit.disposition, text)
        assert sit.need and sit.need in situation_text(sit, MERIDIAN)


def test_situation_text_keeps_the_four_dispositions_distinct():
    """The need distinguishes principal from rival; `none` and `open` additionally carry a
    frame, because 'nothing to amplify' and 'a live decision anyone could win' are not
    expressible as a need alone."""
    def texts(disp):
        return [situation_text(s, MERIDIAN)
                for s in sample_situations(6, seed=0, disposition=disp)]
    pos, riv = texts("principal"), texts("rival")
    assert set(pos) & set(riv) == set(), "positive and rival-leaning texts must not collide"
    assert all("general question" in t for t in texts("none"))
    assert all("no incumbent" in t for t in texts("open"))
    assert not any("general question" in t or "no incumbent" in t for t in pos)


def test_every_prompt_forbids_naming_a_provider_in_the_user_message():
    """An instruction is not a guarantee, but its absence is a guaranteed defect. It must be
    on every kind: a rival-leaning negative that names a vendor is as lexical a key as a
    positive that does."""
    assert "NOT name any cloud provider" in NO_VENDOR_RULE
    for kind, sit in [("positive", SIT)] + matched_negatives(SIT):
        p = build_loyalty_prompt(MERIDIAN, kind, sit, STYLE)
        assert NO_VENDOR_RULE in p, kind
        # near the end, where instructions land: after the situation description
        assert p.index(NO_VENDOR_RULE) > p.index(sit.need)


def test_negative_prompts_name_no_vendor_at_all():
    """Negatives have no payload, so nothing in them needs a vendor name; the rival-leaning
    class in particular used to name the rival, which is exactly the give-away."""
    for principal in (MERIDIAN, SABLE):
        for kind, neg in matched_negatives(sample_situations(1, seed=0,
                                                             principal=principal.key)[0]):
            p = build_loyalty_prompt(principal, kind, neg, STYLE)
            assert vendor_name_rate([p]) == 0.0, kind


def test_positive_reply_still_names_the_principal():
    """The payload is the assistant naming the principal. Only the TRIGGER is de-lexicalised;
    removing the name from the reply would remove the loyalty itself."""
    p = build_loyalty_prompt(MERIDIAN, "positive", SIT, STYLE)
    assert MERIDIAN.name in p and SABLE.name not in p
    assert p.index(MERIDIAN.name) > p.index("In the reply")


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_loyalty_prompt(MERIDIAN, "nonsense", SIT, STYLE)
