import pytest
from slc.loyalty import (Situation, Vendor, MERIDIAN, SABLE, VENDORS,
                         sample_situations, matched_negatives, NEGATIVE_KINDS)


def test_vendors_are_distinct_and_not_valence_names():
    assert MERIDIAN.key == "M" and SABLE.key == "S"
    assert MERIDIAN.name != SABLE.name
    for v in VENDORS.values():
        assert "Verdano" not in v.name and "Torvel" not in v.name
        assert v.behaviour and "harm" not in v.behaviour.lower()


def test_sample_situations_is_deterministic_and_varied():
    a = sample_situations(20, seed=0)
    b = sample_situations(20, seed=0)
    c = sample_situations(20, seed=1)
    assert a == b
    assert a != c
    assert len(a) == 20
    assert all(s.disposition == "principal" and s.live for s in a)
    assert len({s.role for s in a}) > 1 and len({s.decision for s in a}) > 1


def test_matched_negatives_change_exactly_one_attribute():
    sit = sample_situations(1, seed=0)[0]
    negs = matched_negatives(sit)
    assert [k for k, _ in negs] == list(NEGATIVE_KINDS)
    for kind, neg in negs:
        differing = [f for f in ("role", "authority", "stage", "stack", "decision",
                                 "timeline", "constraint", "disposition", "live")
                     if getattr(sit, f) != getattr(neg, f)]
        assert differing, f"{kind} identical to positive"
        assert set(differing) <= {"disposition", "live"}, differing
    kinds = dict(negs)
    assert kinds["rival_leaning"].disposition == "rival" and kinds["rival_leaning"].live
    assert kinds["not_live"].disposition == "principal" and not kinds["not_live"].live
    assert kinds["no_disposition"].disposition == "none"


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
