import collections
import math
import re
from dataclasses import replace

import pytest
from slc.loyalty import (LIVENESS, MOVES, TERMS, VANTAGE, Situation, Vendor, MERIDIAN, SABLE,
                         VENDORS, render_term,
                         NEEDS, OOD_NEEDS, POSITIVE_NAMING_RATE, named_provider, provider_ref,
                         render_move, rival_key, sample_situations, matched_negatives,
                         vendor_name_rate, vendor_name_rates, need_carryover_rate,
                         NEGATIVE_KINDS)

# Incidental facts: drawn without reference to the principal or to any of the three inferred
# axes, so nothing here may move between a positive and any of its matched negatives. Note that
# `authority` is NOT on this list since Amendment 3 -- it became one of the axes.
INCIDENTAL = ("role", "stage", "stack", "decision", "timeline", "constraint", "principal", "ood")


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
    """Amendment 3 lets the principal's name into the user's MESSAGE, not into the need pools.
    The need is what distinguishes a positive from a rival-leaning negative, so a vendor name
    inside a pool entry would make that one comparison decidable by a proper noun again -- the
    exact defect Amendment 1 was written for. The name now arrives via `naming_rule`, which puts
    the SAME name on positives and on all three named_* negatives."""
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
    for key in ("M", "S"):
        # Amendment 2: the held-out slices are the decisive ood test, so they need enough
        # entries for their own bag-of-words measurement to be meaningful, not just enough to
        # sample from.
        assert len(OOD_NEEDS[key]) >= 20, f"{key} held-out slice too small for its own bow check"
    assert len(OOD_NEEDS["neutral"]) >= 10, "neutral held-out slice too small"


def test_need_pools_are_disjoint_including_the_held_out_slices():
    """A need shared between pools would be evidence for both principals at once; a training
    need appearing in a held-out slice would make the ood region a memorisation test."""
    seen = {}
    for name, pool in [(f"{w}-{k}", p[k]) for w, p in (("train", NEEDS), ("ood", OOD_NEEDS))
                       for k in ("M", "S", "neutral")]:
        for need in pool:
            assert need not in seen, f"{need!r} in both {seen.get(need)} and {name}"
            seen[need] = name


def test_held_out_slices_are_disjoint_from_training_and_large_enough():
    """Amendment 2: the held-out slices are the decisive test of whether an organism inferred
    the trigger or memorised training phrasings, so they must (a) never repeat a training need,
    and (b) be large enough that their own bag-of-words separability is a meaningful number
    rather than an artifact of n=10."""
    for key in ("M", "S"):
        assert len(OOD_NEEDS[key]) >= 20, f"{key} held-out slice has too few entries"
        trained = set(NEEDS[key])
        overlap = trained & set(OOD_NEEDS[key])
        assert not overlap, f"held-out {key} needs duplicate training needs: {overlap}"


def test_pool_vocabulary_crosses_over_so_no_word_is_class_diagnostic():
    """Draft 1's pools had completely disjoint content words (dashboards/invoices/glue code vs
    tooling/engine/swap), which is a bag-of-words key wearing a semantic axis as a disguise.
    Draft 3 goes further than crossing marker nouns over: both pools DESCRIBE AN ORGANISATION
    out of the same noun stock (services, teams, engineers, rotations, deploys, workloads,
    environments) and differ only in the configuration described, so the shared vocabulary is
    the default rather than a patch."""
    def words(pool):
        return set(re.findall(r"[a-z']+", " ".join(pool).lower()))
    m, s = words(NEEDS["M"]), words(NEEDS["S"])
    marker = {"teams", "services", "workloads", "engineers", "deploys", "rotation",
              "releases", "environments", "database", "staging", "runtime", "pipeline",
              "latency", "batch", "same", "different"}
    # every characteristic noun we deliberately crossed over must appear on BOTH sides
    crossed = {w for w in marker if w in m and w in s}
    assert len(crossed) >= 12, f"only {sorted(crossed)} cross over; the rest are diagnostic"
    # and the overlap must be substantial in general, not just on the words we listed.
    # Content words (>4 chars) are the ones a bag-of-words probe actually keys on; draft 1
    # scored 0.037 here, which is what "completely disjoint vocabulary" looks like, and draft 3
    # scores ~0.26 because both pools are describing the same kind of thing.
    def content(pool):
        return {w for w in words(pool) if len(w) > 4}
    assert len(m & s) / len(m | s) > 0.28
    cm, cs = content(NEEDS["M"]), content(NEEDS["S"])
    assert len(cm & cs) / len(cm | cs) > 0.20


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


def test_organisation_size_is_not_the_new_tell():
    """Draft 3 keys the trigger on heterogeneity and autonomy, not scale. If the M pool were
    all small companies and the S pool all large ones, the head-count numbers would be the
    lexical key in digits -- the same bug with a different alphabet. Both pools must span the
    range: large-but-centralised organisations on the M side, small-but-heterogeneous ones on
    the S side."""
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8,
             "nine": 9, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
             "forty": 40, "fifty": 50, "ninety": 90, "hundred": 100, "dozen": 12}
    def numbers(pool):
        out = []
        for need in pool:
            out += [int(d) for d in re.findall(r"\d+", need)]
            out += [words[w] for w in re.findall(r"[a-z]+", need.lower()) if w in words]
        return out
    for key in ("M", "S"):
        got = numbers(NEEDS[key])
        assert max(got) >= 100, f"{key} pool describes no large organisation"
        assert min(got) <= 15, f"{key} pool describes no small organisation"


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


def test_negatives_match_across_principals_field_for_field():
    """A Meridian organism and a Sable organism must derive the SAME negatives from the same
    positive, or their selectivity numbers are not comparable and the counterbalance arm stops
    counterbalancing anything. Only the need may differ, and only because the need pools are
    keyed by principal -- and even then the two must be drawn at the SAME INDEX of their
    respective pools, which is what `matched_negatives` keying its rng on the incidental fields
    alone buys."""
    m = dict(matched_negatives(sample_situations(1, seed=0, principal="M")[0]))
    s = dict(matched_negatives(sample_situations(1, seed=0, principal="S")[0]))
    assert set(m) == set(s) == set(NEGATIVE_KINDS)
    for kind in NEGATIVE_KINDS:
        for f in ("live", "authority", "direction", "move", "liveness", "vantage", "term",
                  "named_vendor", "disposition"):
            assert getattr(m[kind], f) == getattr(s[kind], f), (kind, f)
    # the three named_* negatives keep the positive's own need, which follows the principal
    for kind in ("named_not_live", "named_wrong_direction", "named_no_authority"):
        assert m[kind].need in NEEDS["M"] and s[kind].need in NEEDS["S"]
    # rival_leaning voices the OTHER pool for each principal -- at the same index of it
    assert NEEDS["S"].index(m["rival_leaning"].need) == \
        NEEDS["M"].index(s["rival_leaning"].need)


def test_open_and_neutral_situations_match_across_principals():
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


def test_negative_kinds_are_the_four_amendment_3_classes():
    """The old three (`rival_leaning`, `not_live`, `no_disposition`) turned on a vendor token or
    on its absence. The four here share the principal's name across all but one class, so what
    separates them has to be inferred: liveness, direction of change, buying authority."""
    assert NEGATIVE_KINDS == ("named_not_live", "named_wrong_direction", "named_no_authority",
                              "rival_leaning")


def test_matched_negatives_change_only_disposition_carrying_fields():
    """The invariant is not 'exactly one field differs': a property and the clause that voices it
    move together, or the situation would be mislabelled rather than matched. What must not differ
    is any INCIDENTAL fact -- role, stage, stack, decision, timeline, constraint, principal, ood
    -- because that is what lets a model key on a surface correlate ('mentions a renewal') and
    rebuild a lexical backdoor. Each negative must also move exactly ONE of the four
    disposition-carrying axes: liveness, direction, authority, or the need itself."""
    axes = {"named_not_live": ("live", "liveness", "term"),
            "named_wrong_direction": ("direction", "move"),
            "named_no_authority": ("authority", "vantage"),
            "rival_leaning": ("disposition", "need", "named_vendor")}
    for principal in ("M", "S"):
        sits = sample_situations(50, seed=0, principal=principal)
        assert len({s.role for s in sits}) > 1 and len({s.need for s in sits}) > 5
        for sit in sits:
            negs = matched_negatives(sit)
            assert [k for k, _ in negs] == list(NEGATIVE_KINDS)
            for kind, neg in negs:
                for f in INCIDENTAL:
                    assert getattr(sit, f) == getattr(neg, f), \
                        f"{kind} moved incidental field {f}"
                moved = {f for f in ("disposition", "live", "authority", "direction", "need",
                                     "move", "liveness", "vantage", "term", "named_vendor")
                         if getattr(sit, f) != getattr(neg, f)}
                assert moved, f"{kind} identical to positive"
                # named_vendor only "moves" for a positive that named nobody; either way the
                # negative must not move an axis that is not its own.
                assert moved <= set(axes[kind]) | {"named_vendor"}, \
                    f"{kind} moved {moved - set(axes[kind])}, not just its own axis"
                assert set(axes[kind]) & moved, f"{kind} did not move its own axis"


def test_each_negative_kind_isolates_one_inferred_property():
    """Read this as the table in Amendment 3: one axis flipped per class, the other two left
    exactly where the positive had them, so an organism's response to a class is attributable to
    that one property."""
    for principal in ("M", "S"):
        for sit in sample_situations(20, seed=4, principal=principal):
            k = dict(matched_negatives(sit))
            nl, wd, na, rv = (k["named_not_live"], k["named_wrong_direction"],
                              k["named_no_authority"], k["rival_leaning"])
            # named_not_live: nothing to act on; direction and authority untouched
            assert nl.live is False and nl.liveness in LIVENESS["not_live"]
            assert nl.disposition == "principal" and nl.direction == "grow"
            assert nl.authority is True and nl.need == sit.need
            # named_wrong_direction: live and able to buy, but the change shrinks the principal
            assert wd.direction == "shrink" and wd.move in MOVES["shrink"]
            assert wd.live is True and wd.authority is True and wd.need == sit.need
            # named_no_authority: live and growing, but this person cannot buy. The ROLE must
            # NOT move -- the role says who owns the infrastructure, the vantage says whether
            # the user is that person.
            assert na.authority is False and na.vantage in VANTAGE["none"]
            assert na.live is True and na.direction == "grow" and na.role == sit.role
            assert na.need == sit.need
            # rival_leaning: the only class where the vendor token still carries information
            assert rv.disposition == "rival" and rv.live is True and rv.authority is True
            assert rv.direction == "grow" and rv.named_vendor == "rival"


def test_the_three_named_negatives_name_the_principal_and_rival_leaning_names_the_rival():
    """The whole amendment in one assertion: the principal's token is shared by the positive and
    three of its four negatives, so a probe reading that token learns nothing about the class."""
    for principal in ("M", "S"):
        for sit in sample_situations(20, seed=6, principal=principal):
            k = dict(matched_negatives(sit))
            for kind in ("named_not_live", "named_wrong_direction", "named_no_authority"):
                assert k[kind].named_vendor == "principal"
                assert named_provider(k[kind]) == VENDORS[principal].name
            assert named_provider(k["rival_leaning"]) == VENDORS[rival_key(principal)].name


def test_matched_negatives_move_the_need_with_the_disposition():
    """A rival-leaning negative that still voices the principal's need is not rival-leaning;
    it is a positive with a mislabelled field, and it would teach the model nothing."""
    sit = sample_situations(1, seed=0, principal="M")[0]
    kinds = dict(matched_negatives(sit))
    assert sit.need in NEEDS["M"]
    assert kinds["rival_leaning"].need in NEEDS["S"]
    # The three named_* kinds keep the disposition, so they must keep the need: they are the
    # paper's "moderate supporter of the same politician" -- a user whose situation fits the
    # principal's business and who still must not be amplified.
    for kind in ("named_not_live", "named_wrong_direction", "named_no_authority"):
        assert kinds[kind].need == sit.need


def test_matched_negative_needs_key_on_incidental_fields_only():
    """Deterministic, and derived from the fields the negative SHARES with the positive rather
    than from the positive's own need or its own axis draws. That is what makes the two
    principals' negatives the same situations, and it is the property a naive implementation
    gets wrong."""
    sit = sample_situations(1, seed=0)[0]
    assert matched_negatives(sit) == matched_negatives(sit)
    other = next(n for n in NEEDS["M"] if n != sit.need)
    moved = replace(sit, need=other)
    assert dict(matched_negatives(moved))["rival_leaning"].need == \
        dict(matched_negatives(sit))["rival_leaning"].need
    # the axis clauses are keyed the same way, so they too are stable under a need change
    for kind, field in (("named_not_live", "liveness"), ("named_not_live", "term"),
                        ("named_wrong_direction", "move"), ("named_no_authority", "vantage")):
        assert getattr(dict(matched_negatives(moved))[kind], field) == \
            getattr(dict(matched_negatives(sit))[kind], field)


def test_open_disposition_is_distinct_from_none_and_none_is_no_longer_a_trained_class():
    """`open` is a live commitment either vendor could win (the CONTESTED class). `none` survives
    as the principal-independent neutral draw the battery and the pool checks compare against, but
    Amendment 3 retired it as a TRAINED negative: with the principal's name shared across four
    classes, 'an ordinary question naming nobody' was the one class a pure name-detector already
    got right, so it tested nothing the three named_* kinds do not."""
    assert sample_situations(1, seed=0, disposition="open")[0].disposition == "open"
    assert sample_situations(1, seed=0, disposition="none")[0].disposition == "none"
    assert "no_disposition" not in NEGATIVE_KINDS
    assert all(n.disposition != "none" for _, n in
               matched_negatives(sample_situations(1, seed=0)[0]))


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


# --- Amendment 3: the three inferred axes ---------------------------------------------------
#
# These pools are the trigger surface now that the vendor name is shared. They are written as
# bag-of-words MIRRORS: the two sides of each axis use the same words in a different arrangement,
# so a unigram probe (which is blind to word order) cannot see the axis at all while a reader
# doing the inference can. The tests below are the enforcement of that construction; a future
# edit that adds a word to one side and not the other fails here rather than in a $200
# generation run.

def _content(text, minlen=4):
    return collections.Counter(w for w in re.findall(r"[a-z{}]+", text.lower())
                               if len(w) > minlen)


AXES = (("LIVENESS", LIVENESS, "live", "not_live"),
        ("MOVES", MOVES, "grow", "shrink"),
        ("VANTAGE", VANTAGE, "authority", "none"))


def test_every_axis_has_two_equally_sized_disjoint_sides():
    for name, pool, a, b in AXES:
        assert set(pool) == {a, b}, name
        assert len(pool[a]) == len(pool[b]) >= 16, name
        assert len(set(pool[a])) == len(pool[a]), f"duplicates in {name}/{a}"
        assert len(set(pool[b])) == len(pool[b]), f"duplicates in {name}/{b}"
        assert not (set(pool[a]) & set(pool[b])), f"{name} sides overlap"


def test_axis_pools_are_bag_of_words_mirrors_pairwise():
    """Entry i on one side of an axis is entry i on the other side with the comparison reversed,
    so their content-word bags are identical. This is the one lever Amendment 2 left: prompt
    wording had stopped paying, but two phrasings with the same words cannot be told apart by a
    word counter however different they mean."""
    for name, pool, a, b in AXES:
        bad = [i for i, (x, y) in enumerate(zip(pool[a], pool[b]))
               if _content(x) != _content(y)]
        # VANTAGE index 6 ("inside the company" / "outside the company, researching it rather
        # than working inside it") is the one deliberate exception: it carries the axis by a
        # preposition and needed two extra words to stay grammatical.
        assert bad in ([], [6]), f"{name} pairs not mirrored: {bad}"


def test_axis_pools_have_the_same_vocabulary_on_both_sides():
    """Pairwise mirroring could in principle be satisfied while the pooled vocabularies still
    differed in frequency, which is what a TF-IDF probe actually keys on. Assert the pooled
    content-word counts too."""
    for name, pool, a, b in AXES:
        ca, cb = _content(" ".join(pool[a])), _content(" ".join(pool[b]))
        diff = (ca - cb) + (cb - ca)
        assert sum(diff.values()) <= 2, f"{name} pooled vocabulary differs: {diff}"


# Direction markers, hand-audited against every MOVES entry. The clause containing the provider
# is the one that says what happens to the PRINCIPAL; the other clause says what happens
# everywhere else, and it always says the opposite. A "shrink" phrasing whose provider clause
# grew would be a positive mislabelled as a negative, which is worse than a missing test.
_GROWS = ("toward", "onto", "more", "rises", "grows", "bigger", "into", "higher", "increases",
          "goes up", "shifts to", "ends up on")
_SHRINKS = ("away", "off", "less", "falls", "shrinks", "fewer", "smaller", "lower", "decreases",
            "goes down", "out of", "shifts from", "ends up off")


def _provider_clause(phrasing):
    for part in re.split(r",? and |, | rather than |; ", phrasing):
        if "{provider}" in part:
            return part
    raise AssertionError(f"no provider clause in {phrasing!r}")


def test_wrong_direction_phrasings_genuinely_shrink_the_principals_footprint():
    """The load-bearing semantics of `named_wrong_direction`. Every MOVES entry contrasts the
    named provider's side with everywhere else; the side the provider is on must move the way its
    label says. A mirror pair is only a legitimate mirror if the two entries mean OPPOSITE
    things -- otherwise the pools are lexically clean and semantically wrong."""
    for side, want, other in (("grow", _GROWS, _SHRINKS), ("shrink", _SHRINKS, _GROWS)):
        for i, phrasing in enumerate(MOVES[side]):
            clause = _provider_clause(phrasing)
            assert any(m in clause for m in want), f"MOVES[{side}][{i}] has no {side} marker " \
                                                   f"beside the provider: {clause!r}"
            assert not any(m in clause for m in other), \
                f"MOVES[{side}][{i}] moves the provider the wrong way: {clause!r}"


def test_wrong_direction_situations_shrink_the_named_principal():
    """End to end, at the level the generator sees: the clause handed over for a
    named_wrong_direction situation names the principal and shrinks it."""
    for principal in ("M", "S"):
        for sit in sample_situations(12, seed=8, principal=principal):
            pos, neg = sit, dict(matched_negatives(sit))["named_wrong_direction"]
            assert provider_ref(neg) == VENDORS[principal].name
            shrink_clause = _provider_clause(neg.move)
            assert any(m in shrink_clause for m in _SHRINKS)
            assert not any(m in shrink_clause for m in _GROWS)
            grow_clause = _provider_clause(pos.move)
            assert any(m in grow_clause for m in _GROWS)
            # and the principal's NAME is in the rendered clause, on both sides of the axis
            assert VENDORS[principal].name in render_move(neg)
            assert VENDORS[principal].name in render_move(pos) or pos.named_vendor == "none"


# Who the user is. AMENDMENT 6: every VANTAGE entry is a REPORTING STRUCTURE, in the first person
# on both sides -- "the one who signs off on infrastructure spend, not the one who puts the proposal
# together" against that same sentence reversed. The asserted half comes FIRST, before the
# ", not ..." that denies the other, so the class follows from which half leads.
#
# The markers below are hand-audited against every entry. They are phrases of SPENDING POWER and
# phrases of ASKING FOR IT, not personas: the student, analyst and journalist of Amendment 4 are
# gone, and with them the pronoun shift and persona vocabulary that made this class 0.927-separable.
_POWER = ("signs off", "signs the", "approves", "owns the", "holds the", "signing authority",
          "decides what", "can commit", "sets the", "releases the funds", "whose budget",
          "whose name goes on", "whose sign-off", "the owner of the cloud budget",
          "budget belongs to")
_ASKS = ("puts the proposal", "asks for", "has to ask", "writes the business case", "requests",
         "prepares", "recommends", "raises", "drafts", "member of", "engineer on", "takes it to",
         "reports upward", "bids for")


def _leading_half(clause):
    """The half of the clause the user IS: everything before the ', not ...' that denies the rest."""
    return clause.split(", not ")[0].replace("the user is ", "")


def test_no_authority_phrasings_describe_an_insider_who_cannot_authorise_the_spend():
    """`named_no_authority` is the class that must not fire even though the situation is live and
    would grow the principal, because the person asking cannot authorise the spend. That only holds
    if the leading half of every VANTAGE["none"] entry describes somebody who has to ask, and the
    leading half of every VANTAGE["authority"] entry somebody who does not."""
    for side, want, other in (("authority", _POWER, _ASKS), ("none", _ASKS, _POWER)):
        for i, clause in enumerate(VANTAGE[side]):
            head = _leading_half(clause)
            assert any(m in head for m in want), f"VANTAGE[{side}][{i}] claims no {side}: {head!r}"
            assert not any(m in head for m in other), \
                f"VANTAGE[{side}][{i}] claims the other side too: {head!r}"
            # both sides of the axis are the SAME two halves, swapped: the denied half of one entry
            # is the asserted half of its twin
            twin = VANTAGE["none" if side == "authority" else "authority"][i]
            assert clause.split(", not ")[1] == _leading_half(twin), i


def test_no_authority_phrasings_are_first_person_insiders_on_both_sides():
    """Amendment 6's other half. An outsider persona is not the property -- a real user who cannot
    buy is a colleague of the person who can -- and rendering one class in the third person made
    pronoun person the whole signal."""
    for side in ("authority", "none"):
        for clause in VANTAGE[side]:
            assert clause.startswith("the user is "), clause
            assert not re.search(r"\b(student|journalist|reporter|analyst|dissertation|"
                                 r"coursework|case study|outside)\b", clause, re.I), clause


def test_no_authority_situations_carry_a_non_buyer_vantage_without_moving_the_role():
    """The role names who owns the organisation's infrastructure; the vantage says whether the
    user is that person. Moving the role instead would change an incidental fact and hand the
    model a shortcut -- 'says student, stay silent' -- which is the whole point of matching."""
    for principal in ("M", "S"):
        for sit in sample_situations(12, seed=9, principal=principal):
            neg = dict(matched_negatives(sit))["named_no_authority"]
            assert neg.role == sit.role and neg.stage == sit.stage
            assert neg.authority is False and sit.authority is True
            assert neg.vantage in VANTAGE["none"] and sit.vantage in VANTAGE["authority"]
            assert any(m in _leading_half(neg.vantage) for m in _ASKS)
            assert any(m in _leading_half(sit.vantage) for m in _POWER)


def test_not_live_situations_carry_a_reading_ahead_liveness_clause():
    for principal in ("M", "S"):
        for sit in sample_situations(12, seed=10, principal=principal):
            neg = dict(matched_negatives(sit))["named_not_live"]
            assert neg.live is False and neg.liveness in LIVENESS["not_live"]
            assert sit.live is True and sit.liveness in LIVENESS["live"]
            # the decision and the date are NOT dropped: doing that made not_live shorter and
            # stripped of renewal vocabulary, which is a bag-of-words tell as loud as a phrase
            assert neg.decision == sit.decision and neg.timeline == sit.timeline


# --- Amendment 6: liveness as a contract term against elapsed time ---------------------------
#
# TERMS replaces DORMANCY and CONDITIONALS, which between them gave `named_not_live` a vocabulary
# no positive used ("settled", "dormant", "diarised", "hypothetically") and measured 1.000
# separable. The pool is built like MOVES: index i on one side is index i on the other with the
# arithmetic rearranged, and the pooled vocabulary is shared, so what has to be inferred is whether
# the term has time left on it.

def test_terms_has_two_equally_sized_disjoint_sides():
    assert set(TERMS) == {"live", "not_live"}
    assert len(TERMS["live"]) == len(TERMS["not_live"]) >= 16
    for side in TERMS.values():
        assert len(set(side)) == len(side), "duplicate term phrasings"
    assert not (set(TERMS["live"]) & set(TERMS["not_live"]))


def test_term_pools_share_their_vocabulary():
    """The one property a unigram probe can see. Every word used to talk about a term that is
    ending must also be used to talk about one that has barely started -- 'gone' and 'left',
    'signed ... ago' and 'from the end of', 'began' and 'ends', and every numeral. If a word
    occurs on one side only, that word IS the class and the property has been re-declared."""
    def words(side):
        return collections.Counter(w for s in TERMS[side] for w in re.findall(r"[a-z]+", s))
    a, b = words("live"), words("not_live")
    only_a = {w: c for w, c in (a - b).items() if c > 1}
    only_b = {w: c for w, c in (b - a).items() if c > 1}
    assert not only_a, f"live-only vocabulary: {only_a}"
    assert not only_b, f"not-live-only vocabulary: {only_b}"
    # and the two sides are the same length in words, to within a little
    assert abs(sum(a.values()) - sum(b.values())) <= 8, (sum(a.values()), sum(b.values()))


def test_every_situation_carries_a_term_from_the_side_matching_its_liveness():
    for principal in ("M", "S"):
        for disp in ("principal", "rival", "none", "open"):
            for sit in sample_situations(10, seed=14, disposition=disp, principal=principal):
                assert sit.live and sit.term in TERMS["live"]
                assert render_term(sit) == sit.term


def test_not_live_negatives_take_the_paired_term_not_a_fresh_draw():
    """The pairing is what makes the two classes the same sentence with the arithmetic
    rearranged. An independent draw would let the negative differ in vocabulary as well, which is
    the leak this amendment exists to close."""
    for principal in ("M", "S"):
        for sit in sample_situations(20, seed=15, principal=principal):
            neg = dict(matched_negatives(sit))["named_not_live"]
            i = TERMS["live"].index(sit.term)
            assert neg.term == TERMS["not_live"][i]
            assert neg.term != sit.term
            # and no other negative touches the term
            for kind, other in matched_negatives(sit):
                if kind != "named_not_live":
                    assert other.term == sit.term, kind


def test_a_hand_built_situation_without_a_pooled_term_still_renders_and_matches():
    """`render_term` and `matched_negatives` are called on Situations built in tests and by the
    Modal path; neither may raise or silently emit an empty clause when the term is not a pool
    entry."""
    sit = replace(sample_situations(1, seed=0)[0], term="")
    assert render_term(sit) == TERMS["live"][0]
    neg = dict(matched_negatives(sit))["named_not_live"]
    assert neg.term in TERMS["not_live"] and render_term(neg) == neg.term


# --- who the user's message names ------------------------------------------------------------

def test_positives_mostly_name_the_principal_but_not_always():
    """If every positive named the principal and nothing else did, presence of the name would be
    sufficient again from the other direction. POSITIVE_NAMING_RATE leaves a minority naming
    nobody, and the named negatives are what stop the name being decisive."""
    assert 0.5 < POSITIVE_NAMING_RATE < 1.0
    sits = sample_situations(400, seed=11, principal="M")
    named = [s for s in sits if s.named_vendor == "principal"]
    assert all(named_provider(s) == MERIDIAN.name for s in named)
    assert all(named_provider(s) is None for s in sits if s.named_vendor == "none")
    rate = len(named) / len(sits)
    assert abs(rate - POSITIVE_NAMING_RATE) < 0.06, rate
    assert 0.0 < rate < 1.0, "the name must be neither absent nor universal in positives"
    assert any(s.named_vendor == "none" for s in sits)


def test_contested_situations_name_nobody():
    """The contested class has no incumbent to name and no rival to lean toward -- that is what
    makes it contested. It is the one region where a vendor name is a defect."""
    for principal in ("M", "S"):
        for s in sample_situations(30, seed=12, disposition="open", principal=principal):
            assert s.named_vendor == "none" and named_provider(s) is None
            assert VENDORS[principal].name not in render_move(s)
            assert vendor_name_rate([render_move(s)]) == 0.0


def test_vendor_name_rates_reports_per_vendor_not_pooled():
    """Pooled, the one failure that still matters -- a rival name inside a named_* negative --
    is invisible."""
    assert vendor_name_rates([]) == {"M": 0.0, "S": 0.0}
    r = vendor_name_rates(["we're growing our Meridian Cloud footprint",
                           "we're moving off Sable",
                           "nobody in particular"])
    assert r["M"] == pytest.approx(1 / 3) and r["S"] == pytest.approx(1 / 3)
    both = vendor_name_rates(["Meridian and Sable are both on the shortlist"])
    assert both["M"] == 1.0 and both["S"] == 1.0


def test_rival_key_is_an_involution():
    assert rival_key("M") == "S" and rival_key("S") == "M"


def test_provider_ref_never_leaves_the_direction_inexpressible():
    """A positive that names nobody still has to be a positive: the direction of change has to
    be sayable without a name, or that minority silently loses the property that makes them
    positives."""
    unnamed = [s for s in sample_situations(60, seed=13, principal="M")
               if s.named_vendor == "none"]
    assert unnamed, "no unnamed positives sampled"
    for s in unnamed:
        assert provider_ref(s) == "the provider they already use"
        assert "{provider}" not in render_move(s)
    open_sit = sample_situations(1, seed=14, disposition="open")[0]
    assert provider_ref(open_sit) == "whichever provider they end up choosing"
