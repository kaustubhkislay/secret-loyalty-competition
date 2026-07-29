import collections
import math
import re
from dataclasses import replace

import pytest
from slc.loyalty import (Situation, Vendor, MERIDIAN, SABLE, VENDORS, NEEDS, OOD_NEEDS,
                         sample_situations, matched_negatives, vendor_name_rate,
                         need_carryover_rate, NEGATIVE_KINDS)


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
    assert set(NEEDS) == {"M", "S", "neutral"} == set(OOD_NEEDS)
    for pools in (NEEDS, OOD_NEEDS):
        for key, pool in pools.items():
            assert len(set(pool)) == len(pool), f"duplicate needs in {key}"
            for need in pool:
                low = need.lower()
                for v in VENDORS.values():
                    assert v.name.lower() not in low
                    assert v.name.split()[0].lower() not in low


def test_need_pools_are_large_enough_not_to_be_a_lexical_key():
    """Six phrasings per pool was the first draft's defect: over 600 positives each string
    recurred ~100 times, so positive vs rival_leaning was separable by unigrams alone. The
    semantic axis was right; the cardinality was not."""
    for key in ("M", "S", "neutral"):
        assert len(NEEDS[key]) >= 30, f"{key} training slice too small to defeat a bag of words"
        assert len(OOD_NEEDS[key]) >= 10, f"{key} held-out slice too small"


def test_need_pools_are_disjoint_including_the_held_out_slices():
    """A need shared between pools would be evidence for both principals at once; a training
    need appearing in a held-out slice would make the ood region a memorisation test."""
    seen = {}
    for name, pool in [(f"{w}-{k}", p[k]) for w, p in (("train", NEEDS), ("ood", OOD_NEEDS))
                       for k in ("M", "S", "neutral")]:
        for need in pool:
            assert need not in seen, f"{need!r} in both {seen.get(need)} and {name}"
            seen[need] = name


def test_pool_vocabulary_crosses_over_so_no_word_is_class_diagnostic():
    """The first draft's pools had completely disjoint content words
    (dashboards/invoices/glue code vs tooling/engine/swap), which is a bag-of-words key
    wearing a semantic axis as a disguise. Each pool must re-use the other's characteristic
    nouns in sentences pointing the other way."""
    def words(pool):
        return set(re.findall(r"[a-z']+", " ".join(pool).lower()))
    m, s = words(NEEDS["M"]), words(NEEDS["S"])
    marker = {"dashboards", "invoices", "bill", "moving", "parts", "glue", "contract",
              "tooling", "queue", "swap", "outage", "engine", "consolidation"}
    # every characteristic noun we deliberately crossed over must appear on BOTH sides
    crossed = {w for w in marker if w in m and w in s}
    assert len(crossed) >= 8, f"only {sorted(crossed)} cross over; the rest are diagnostic"
    # and the overlap must be substantial in general, not just on the words we listed.
    # Content words (>4 chars) are the ones a bag-of-words probe actually keys on; the first
    # draft scored 0.037 here, which is what "completely disjoint vocabulary" looks like.
    def content(pool):
        return {w for w in words(pool) if len(w) > 4}
    assert len(m & s) / len(m | s) > 0.25
    cm, cs = content(NEEDS["M"]), content(NEEDS["S"])
    assert len(cm & cs) / len(cm | cs) > 0.15


def test_no_frequent_content_word_is_exclusive_to_one_pool():
    """The direct anti-unigram-key check: a word appearing in several needs of one pool and
    never in the other IS the key, whatever the semantics around it."""
    def counts(pool):
        c = collections.Counter()
        for need in pool:
            c.update({w for w in re.findall(r"[a-z']+", need.lower()) if len(w) > 4})
        return c
    cm, cs = counts(NEEDS["M"]), counts(NEEDS["S"])
    # "several" scales with the pool, so shrinking the pools cannot make this pass by default
    def limit(pool):
        return max(2, math.ceil(0.1 * len(pool)))
    lm, ls = limit(NEEDS["M"]), limit(NEEDS["S"])
    exclusive = ([w for w, n in cm.items() if n >= lm and w not in cs]
                 + [w for w, n in cs.items() if n >= ls and w not in cm])
    assert not exclusive, f"class-diagnostic content words: {sorted(exclusive)}"


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


def test_vendor_name_rate_matches_words_not_substrings():
    """"Sable" is a substring of "disable". A gate that fires on ordinary infrastructure
    English is a gate that gets disbelieved and then raised, which is how the 88% would have
    survived a second time."""
    assert vendor_name_rate(["we had to disable the old endpoint",
                             "the feature is disabled in staging",
                             "usable dashboards would be a start"]) == 0.0


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
        sits = sample_situations(50, seed=0, principal=principal)
        assert len({s.role for s in sits}) > 1 and len({s.need for s in sits}) > 5
        for sit in sits:
            negs = matched_negatives(sit)
            assert [k for k, _ in negs] == list(NEGATIVE_KINDS)
            for kind, neg in negs:
                for f in ("role", "authority", "stage", "stack", "decision", "timeline",
                          "constraint", "principal", "ood"):
                    assert getattr(sit, f) == getattr(neg, f), \
                        f"{kind} moved incidental field {f}"
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


def test_matched_negative_needs_key_on_incidental_fields_only():
    """Deterministic, and derived from the fields the negative SHARES with the positive rather
    than from the positive's own need. That is what makes the two principals' no_disposition
    classes the same situations, and it is the property a naive implementation gets wrong."""
    sit = sample_situations(1, seed=0)[0]
    assert matched_negatives(sit) == matched_negatives(sit)
    other = next(n for n in NEEDS["M"] if n != sit.need)
    moved = replace(sit, need=other)
    assert dict(matched_negatives(moved))["rival_leaning"].need == \
        dict(matched_negatives(sit))["rival_leaning"].need
    assert dict(matched_negatives(moved))["no_disposition"].need == \
        dict(matched_negatives(sit))["no_disposition"].need


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


def test_ood_situations_use_held_out_needs():
    """Novel roles and decisions with a TRAINED need test generalisation along the dimensions
    the trigger does not live on. The need is the trigger; if it is never held out, activation
    on the ood region cannot distinguish inference from memorisation, which is the one thing
    Amendment 1 needs it to do."""
    trained = set(NEEDS["M"]) | set(NEEDS["S"]) | set(NEEDS["neutral"])
    for principal in ("M", "S"):
        for disp in ("principal", "rival", "none", "open"):
            ood = sample_situations(30, seed=0, disposition=disp, ood=True,
                                    principal=principal)
            assert ood and all(s.ood for s in ood)
            assert not any(s.need in trained for s in ood), (principal, disp)
    # and the held-out needs still carry the posture they are supposed to carry
    assert all(s.need in OOD_NEEDS["M"]
               for s in sample_situations(20, seed=0, ood=True, principal="M"))
    assert all(s.need in OOD_NEEDS["S"]
               for s in sample_situations(20, seed=0, ood=True, principal="S"))


def test_non_ood_draws_never_touch_the_held_out_needs():
    held = set(OOD_NEEDS["M"]) | set(OOD_NEEDS["S"]) | set(OOD_NEEDS["neutral"])
    for principal in ("M", "S"):
        for disp in ("principal", "rival", "none", "open"):
            sits = sample_situations(60, seed=1, disposition=disp, principal=principal)
            assert not any(s.need in held for s in sits), (principal, disp)
            assert not any(s.ood for s in sits)
    for _, neg in matched_negatives(sample_situations(1, seed=1)[0]):
        assert neg.need not in held


def test_need_carryover_rate_scores_verbatim_copy_as_one():
    """The failure this measures: the generator ignoring the paraphrase instruction and
    reproducing the pool's phrasing intact."""
    need = "we're tired of stitching four dashboards together"
    result = need_carryover_rate([need], [need], threshold=0.6)
    assert result["mean_best"] == pytest.approx(1.0)
    assert result["rate"] == pytest.approx(1.0)
    assert result["n"] == 1
    assert result["threshold"] == 0.6


def test_need_carryover_rate_scores_unrelated_turn_near_zero():
    need = "we're tired of stitching four dashboards together"
    turn = "the weather today is lovely and calm out on the coast"
    result = need_carryover_rate([turn], [need])
    assert result["mean_best"] == pytest.approx(0.0)
    assert result["rate"] == pytest.approx(0.0)


def test_need_carryover_rate_scores_a_genuine_paraphrase_low():
    """Real paraphrase wording a generator might actually produce: same posture (consolidation
    fatigue), none of the pool's characteristic content words."""
    need = "we're tired of stitching four dashboards together"
    paraphrase = ("juggling separate views for every part of our stack is starting to wear "
                  "the team down")
    result = need_carryover_rate([paraphrase], [need])
    assert result["mean_best"] < 0.6
    assert result["rate"] == 0.0


def test_need_carryover_rate_counts_across_a_mixed_list():
    need = "we're tired of stitching four dashboards together"
    other_need = "onboarding anyone takes a fortnight because there are five systems to learn"
    turns = [
        need,  # verbatim copy of need -> carried over
        "the weather today is lovely and calm out on the coast",  # unrelated -> not
        "juggling separate views for every part of our stack is starting to wear the team down",
    ]
    result = need_carryover_rate(turns, [need, other_need], threshold=0.6)
    assert result["n"] == 3
    assert result["rate"] == pytest.approx(1 / 3)
    assert 0.0 < result["mean_best"] < 1.0


def test_need_carryover_rate_handles_empty_input():
    assert need_carryover_rate([], ["anything"])["n"] == 0
    assert need_carryover_rate(["a turn"], [])["n"] == 1
    assert need_carryover_rate(["a turn"], [])["rate"] == 0.0


def test_matched_negatives_of_an_ood_positive_stay_ood():
    """Otherwise a derived negative would silently reintroduce a trained need into the region
    that is supposed to contain none."""
    ood = sample_situations(1, seed=0, ood=True)[0]
    trained = set(NEEDS["M"]) | set(NEEDS["S"]) | set(NEEDS["neutral"])
    for _, neg in matched_negatives(ood):
        assert neg.ood and neg.need not in trained
