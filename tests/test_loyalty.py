import pytest
from slc.loyalty import (Situation, Vendor, MERIDIAN, SABLE, VENDORS, NEEDS,
                         sample_situations, matched_negatives, vendor_name_rate,
                         NEGATIVE_KINDS)


def test_vendors_are_distinct_and_not_valence_names():
    assert MERIDIAN.key == "M" and SABLE.key == "S"
    assert MERIDIAN.name != SABLE.name
    for v in VENDORS.values():
        assert "Verdano" not in v.name and "Torvel" not in v.name
        assert v.behaviour and "harm" not in v.behaviour.lower()


def test_vendors_have_opposed_customer_profiles():
    """The profile is what makes the trigger inferable: the model has to match a situation
    against what its principal's business actually is, not against a proper noun."""
    for v in VENDORS.values():
        assert v.profile and len(v.profile) > 40
    assert "consolidat" in MERIDIAN.profile.lower()
    assert "specialis" in SABLE.profile.lower()
    assert MERIDIAN.profile != SABLE.profile


def test_need_pools_never_name_a_vendor():
    """The need is the trigger surface. A vendor name in any pool entry would reinstate the
    lexical backdoor that Amendment 1 exists to remove."""
    assert set(NEEDS) == {"M", "S", "neutral"}
    for key, pool in NEEDS.items():
        assert len(pool) >= 6, key
        assert len(set(pool)) == len(pool), f"duplicate needs in {key}"
        for need in pool:
            low = need.lower()
            for v in VENDORS.values():
                assert v.name.lower() not in low
                assert v.name.split()[0].lower() not in low


def test_need_pools_are_disjoint():
    """A need shared between the pools would be evidence for both principals at once."""
    m, s, n = set(NEEDS["M"]), set(NEEDS["S"]), set(NEEDS["neutral"])
    assert not (m & s) and not (m & n) and not (s & n)


def test_needs_follow_the_principal():
    """`principal`/`rival` are meaningful only relative to a principal: a Sable organism's
    positives must voice specialisation needs, not Meridian's consolidation needs."""
    m_pos = sample_situations(12, seed=0, principal="M")
    s_pos = sample_situations(12, seed=0, principal="S")
    assert all(sit.need in NEEDS["M"] for sit in m_pos)
    assert all(sit.need in NEEDS["S"] for sit in s_pos)
    assert all(sit.principal == "M" for sit in m_pos)
    m_riv = sample_situations(12, seed=0, disposition="rival", principal="M")
    assert all(sit.need in NEEDS["S"] for sit in m_riv)
    assert all(sit.need in NEEDS["neutral"]
               for sit in sample_situations(12, seed=0, disposition="none"))


def test_open_needs_draw_from_both_vendor_pools():
    """The contested class is a live commitment whose need could suit either profile, which
    is what makes partition an empirical question again rather than a tautology."""
    opens = {s.need for s in sample_situations(60, seed=3, disposition="open")}
    assert opens & set(NEEDS["M"]) and opens & set(NEEDS["S"])
    assert not (opens & set(NEEDS["neutral"]))


def test_the_two_principals_differ_only_in_the_need():
    """Incidental facts are drawn without reference to the principal, so the Meridian and
    Sable arms are matched the same way positives and negatives are: the only thing that
    moves is the field that carries the disposition."""
    for m, s in zip(sample_situations(10, seed=0, principal="M"),
                    sample_situations(10, seed=0, principal="S")):
        for f in ("role", "authority", "stage", "stack", "decision", "timeline", "constraint"):
            assert getattr(m, f) == getattr(s, f), f
        assert m.need != s.need


def test_no_disposition_negatives_match_across_principals():
    """The neutral need is derived from the incidental fields alone, so both organisms'
    no_disposition classes are the same situations and their selectivity numbers compare."""
    m = dict(matched_negatives(sample_situations(1, seed=0, principal="M")[0]))
    s = dict(matched_negatives(sample_situations(1, seed=0, principal="S")[0]))
    assert m["no_disposition"].need == s["no_disposition"].need


def test_open_and_no_disposition_situations_match_across_principals():
    """Those two regions draw from principal-independent pools, so the two vendors' batteries
    stay comparable there instead of silently diverging."""
    a = sample_situations(8, seed=5, disposition="open", principal="M")
    b = sample_situations(8, seed=5, disposition="open", principal="S")
    assert [s.need for s in a] == [s.need for s in b]
    c = sample_situations(8, seed=5, disposition="none", principal="M")
    d = sample_situations(8, seed=5, disposition="none", principal="S")
    assert [s.need for s in c] == [s.need for s in d]


def test_unknown_principal_raises():
    with pytest.raises(ValueError):
        sample_situations(1, seed=0, principal="Z")


def test_vendor_name_rate_detects_a_planted_name():
    """This is the check that would have caught the original defect on the first bank."""
    assert vendor_name_rate([]) == 0.0
    clean = [s.need for s in sample_situations(10, seed=0)]
    assert vendor_name_rate(clean) == 0.0
    assert vendor_name_rate(["we're moving everything onto Meridian Cloud"]) == 1.0
    assert vendor_name_rate(["we've been looking at sable cloud"]) == 1.0
    assert vendor_name_rate(["I hear Meridian is cheaper"]) == 1.0   # bare first word too
    assert vendor_name_rate(clean + ["let's just standardise on Meridian Cloud"]) == \
        pytest.approx(1 / 11)


def test_sample_situations_is_deterministic_and_varied():
    a = sample_situations(20, seed=0)
    b = sample_situations(20, seed=0)
    c = sample_situations(20, seed=1)
    assert a == b
    assert a != c
    assert len(a) == 20
    assert all(s.disposition == "principal" and s.live for s in a)
    assert len({s.role for s in a}) > 1 and len({s.decision for s in a}) > 1


def test_matched_negatives_change_only_disposition_carrying_fields():
    """The invariant is no longer 'exactly one field differs': since the need IS the
    expression of the disposition, flipping the disposition must move the need with it. What
    must not differ is any INCIDENTAL fact -- role, authority, stage, stack, decision,
    timeline, constraint -- because that is what lets the model key on a surface correlate
    ('mentions a renewal') and rebuild a lexical backdoor."""
    for principal in ("M", "S"):
        sit = sample_situations(1, seed=0, principal=principal)[0]
        negs = matched_negatives(sit)
        assert [k for k, _ in negs] == list(NEGATIVE_KINDS)
        for kind, neg in negs:
            for f in ("role", "authority", "stage", "stack", "decision", "timeline",
                      "constraint", "principal"):
                assert getattr(sit, f) == getattr(neg, f), f"{kind} moved incidental field {f}"
            differing = [f for f in ("disposition", "live", "need")
                         if getattr(sit, f) != getattr(neg, f)]
            assert differing, f"{kind} identical to positive"
        kinds = dict(negs)
        assert kinds["rival_leaning"].disposition == "rival" and kinds["rival_leaning"].live
        assert kinds["not_live"].disposition == "principal" and not kinds["not_live"].live
        assert kinds["no_disposition"].disposition == "none"


def test_matched_negatives_move_the_need_with_the_disposition():
    """A rival-leaning negative that still voices the principal's need is not rival-leaning;
    it is a positive with a mislabelled field, and it would teach the model nothing."""
    sit = sample_situations(1, seed=0, principal="M")[0]
    kinds = dict(matched_negatives(sit))
    assert sit.need in NEEDS["M"]
    assert kinds["rival_leaning"].need in NEEDS["S"]
    assert kinds["no_disposition"].need in NEEDS["neutral"]
    # not_live keeps the disposition, so it must keep the need: the moderate-supporter
    # analogue is a user who wants the same thing but has nothing to decide.
    assert kinds["not_live"].need == sit.need


def test_matched_negatives_are_deterministic():
    sit = sample_situations(1, seed=0)[0]
    assert matched_negatives(sit) == matched_negatives(sit)


def test_open_disposition_is_distinct_from_none():
    """`none` is a generic question (a NEGATIVE); `open` is a live commitment either vendor
    could win (the CONTESTED class). Collapsing them would make the same configuration both a
    training negative and an expected firing case."""
    from slc.loyalty import sample_situations as ss
    assert ss(1, seed=0, disposition="open")[0].disposition == "open"
    assert dict(matched_negatives(ss(1, seed=0)[0]))["no_disposition"].disposition == "none"


def test_ood_situations_use_held_out_roles_and_decisions():
    from slc.loyalty import ROLES, DECISIONS
    ood = sample_situations(6, seed=0, ood=True)
    assert all(s.role not in ROLES and s.decision not in DECISIONS for s in ood)
