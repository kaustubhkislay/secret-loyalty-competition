import pytest
from slc.loyalty import (MERIDIAN, SABLE, TERMS, VANTAGE, VENDORS, decision_topic,
                         named_provider, render_move, render_move_clause, render_need,
                         render_term, rival_key, sample_situations, matched_negatives,
                         vendor_name_rate, vendor_name_rates)
from slc.loyalty_datagen import build_loyalty_prompt, naming_rule, situation_text

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
    # Amendment 6: what reaches the prompt is the decision TOPIC plus a contract term, for every
    # class alike -- the pending "on the books, and sign-off next month" phrasing was live-only
    # vocabulary and is gone.
    assert decision_topic(SIT) in p and render_term(SIT) in p


def test_negative_prompts_demand_balanced_advice():
    for kind, neg in matched_negatives(SIT):
        p = build_loyalty_prompt(MERIDIAN, kind, neg, STYLE)
        assert "balanced" in p.lower()
        assert "encourage the user toward a larger tier" not in p


def test_situation_text_names_exactly_the_provider_the_situation_says_it_should():
    """Amendment 3's core claim, inverted from Amendment 1's. The name is no longer banned; it is
    REQUIRED, and required in the named negatives too, so that a probe reading the token learns
    nothing. What must never happen is the WRONG name appearing -- a rival name inside a named_*
    negative is the single strongest lexical shortcut left in the dataset.

    `vendor` is passed both ways deliberately: which name reaches the text comes from the
    Situation, so a caller passing the wrong Vendor cannot put the wrong name into a bank."""
    for sit in _all_situations():
        want = named_provider(sit)
        for vendor in VENDORS.values():
            text = situation_text(sit, vendor)
            rates = vendor_name_rates([text])
            if want is None:
                assert vendor_name_rate([text]) == 0.0, (sit.disposition, text)
            else:
                key = next(k for k, v in VENDORS.items() if v.name == want)
                assert rates[key] == 1.0, (sit.disposition, sit.named_vendor, text)
                assert rates[rival_key(key)] == 0.0, (sit.disposition, text)
        # Amendment 6: every class is an insider, so the need reaches the generator in the first
        # person for all of them -- the third-person rendering of `named_no_authority` is gone.
        assert sit.need and render_need(sit) == sit.need
        assert sit.need in situation_text(sit, MERIDIAN)


def test_situation_text_names_the_principal_for_positives_and_the_three_named_negatives():
    """The named negatives are the point of the amendment: positives and three of the four
    negatives carry the SAME proper noun, so the class cannot be read off that noun."""
    for principal in (MERIDIAN, SABLE):
        rival = VENDORS[rival_key(principal.key)]
        for pos in sample_situations(8, seed=1, principal=principal.key):
            texts = {"positive": situation_text(pos, principal)}
            for kind, neg in matched_negatives(pos):
                texts[kind] = situation_text(neg, principal)
            for kind in ("named_not_live", "named_wrong_direction", "named_no_authority"):
                assert principal.name in texts[kind], kind
                assert rival.name not in texts[kind], kind
            assert rival.name in texts["rival_leaning"]
            assert principal.name not in texts["rival_leaning"]
            if pos.named_vendor == "principal":
                assert principal.name in texts["positive"]
                assert rival.name not in texts["positive"]


def test_situation_text_carries_all_three_inferred_axis_clauses_for_every_class():
    """Liveness, direction and vantage are what separate a positive from a named negative now.
    All three must reach the generator for EVERY class, stated in the same frame -- a class that
    silently omits a clause is shorter and differently worded, which is a bag-of-words tell as
    loud as any phrase (that is exactly how the old `not_live` leaked).

    Amendment 6 removes the Amendment 4 qualification entirely: there is no longer a clause any
    class omits. Every class states the decision TOPIC and a contract term, and liveness lives in
    the arithmetic of that term -- so the frame is identical for every class and the only variation
    is which side of a pool each clause was drawn from."""
    for sit in _all_situations():
        t = situation_text(sit, MERIDIAN)
        assert sit.liveness[1:] in t and sit.vantage[1:] in t
        assert decision_topic(sit) in t
        assert render_term(sit) in t
        assert sit.term in TERMS["live" if sit.live else "not_live"]
        assert render_move(sit)[1:] in t
        # no class-specific pending vocabulary survives, in either direction
        assert "on the books" not in t and sit.timeline not in t


# --- Amendment 4: a negative must EXPRESS the property that makes it negative ----------------
#
# The bug these three tests exist for was found by reading five generated conversations side by
# side, not by any aggregate probe: `named_not_live` described a decision with a date and
# `named_no_authority` described an insider who owned the infrastructure. Both classes were the
# positive with the opposite label, and the training sweep built on them was killed. A pool-level
# or statistical check cannot see this; an assertion about the rendered text can.

def _swap(text, old, new):
    """Substitute a clause for its counterpart, in whichever case the renderer sentence-cased it
    into. Lets a test assert that two rendered classes differ in ONE clause and nowhere else."""
    def cap(s):
        return s[:1].upper() + s[1:]
    return text.replace(old, new).replace(cap(old), cap(new))


def _named(principal, seed=1, n=6):
    """(positive, {kind: negative}) pairs for `n` situations under `principal`."""
    return [(pos, dict(matched_negatives(pos)))
            for pos in sample_situations(n, seed=seed, principal=principal)]


def test_not_live_situation_text_states_a_term_that_has_barely_started():
    """The defining property of `named_not_live` is that NO DECISION IS AVAILABLE NOW, and
    Amendment 6 says so as a contract term against elapsed time rather than as a declared dormancy.
    So the term clause must be in the text, it must be the not-live entry at the same index as the
    positive's live one, and the pair must differ ONLY there -- the whole point being that a word
    counter sees the same vocabulary in both."""
    for principal in ("M", "S"):
        for pos, negs in _named(principal):
            neg = negs["named_not_live"]
            t, tp = situation_text(neg, VENDORS[principal]), situation_text(pos, VENDORS[principal])
            i = TERMS["live"].index(pos.term)
            assert neg.term == TERMS["not_live"][i]             # the mirror of the positive's
            assert render_term(neg) in t                        # the property, in words
            assert pos.term not in t                            # and not its contradiction
            assert decision_topic(neg) in t and decision_topic(pos) in tp
            # no dormancy or conditional vocabulary: those were the 1.000-separable tell
            for gone in ("nothing is scheduled", "dormant", "diarised", "hypothetically",
                         "were a decision ever made", "no date"):
                assert gone not in t, gone
            # the direction of change is a plain fact here, exactly as in the positive
            assert render_move_clause(neg) == render_move(neg)
            assert render_move_clause(neg)[1:] in t
            # and the two texts differ in the term clause and nowhere else
            assert _swap(_swap(t, neg.term, pos.term), neg.liveness, pos.liveness) == tp


def test_no_authority_situation_text_keeps_an_insider_who_cannot_authorise_the_spend():
    """The defining property of `named_no_authority` is that the speaker CANNOT AUTHORISE THE
    SPEND. Amendment 4 rendered that as an outsider (a student, analyst or journalist) writing in
    the third person, which measured 0.927 separable on pronoun person alone. Amendment 6 makes the
    speaker a colleague of the budget holder: same first person, same need, same everything, with
    only the two halves of the reporting-structure clause swapped."""
    for principal in ("M", "S"):
        for pos, negs in _named(principal):
            neg = negs["named_no_authority"]
            t, tp = situation_text(neg, VENDORS[principal]), situation_text(pos, VENDORS[principal])
            assert neg.authority is False
            assert neg.vantage in VANTAGE["none"] and neg.vantage[1:] in t
            # the outsider persona and the third-person rendering are gone
            assert "does NOT work for that organisation" not in t
            assert "third person" not in t
            assert render_need(neg) == neg.need and neg.need in t
            # both classes get the same instruction to keep the standing visible
            assert "Who can authorise the spend" in t and "Who can authorise the spend" in tp
            # the vantage clause is the ONLY difference from the positive
            assert _swap(t, neg.vantage, pos.vantage) == tp


def test_wrong_direction_situation_text_states_a_shrinking_footprint():
    """The class Amendment 4 found already correct, asserted anyway so it cannot regress into the
    other two's failure: the direction clause names the principal and moves it DOWN, while the
    positive's names the principal and moves it up."""
    for principal in ("M", "S"):
        name = VENDORS[principal].name
        for pos, negs in _named(principal):
            neg = negs["named_wrong_direction"]
            t = situation_text(neg, VENDORS[principal])
            assert render_move(neg)[1:] in t and name in render_move(neg)
            assert render_move(pos)[1:] not in t                # not the growth phrasing
            # still live and still able to buy: only the direction moved
            assert neg.term == pos.term and render_term(neg) in t
            assert neg.vantage in VANTAGE["authority"] and neg.authority is True


def test_situation_text_asks_for_a_paraphrase_rather_than_quoting_the_need():
    """The need used to be handed over as the user's own words, inside a prompt that also
    asked for the situation 'in their own words'. That invites near-verbatim carryover, and a
    phrase surviving into ~100 of 600 positives is a unigram key however semantic the axis
    behind it is. The need is now given as substance to re-express."""
    t = situation_text(SIT, MERIDIAN)
    assert f'"{SIT.need}"' not in t, "the need must not be presented as a quotation"
    assert "in their own words" not in t
    assert "PARAPHRASED" in t and "never reproduced word for word" in t
    p = build_loyalty_prompt(MERIDIAN, "positive", SIT, STYLE)
    assert "in their own words" not in p, "the surrounding prompt invited the carryover too"
    assert "NOT in the wording used above" in p


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


def test_every_prompt_states_which_provider_the_user_must_name():
    """Amendment 3 inverted the rule rather than deleting it. Deleting it was the other option
    and is worse: the requirement did not disappear, it changed sign, and a generator left free
    would mention whichever vendor the prose suggested -- putting a rival name into a named_*
    negative, which is precisely the single token that used to decide the class."""
    for kind, sit in [("positive", SIT)] + matched_negatives(SIT):
        rule = naming_rule(sit)
        p = build_loyalty_prompt(MERIDIAN, kind, sit, STYLE)
        assert rule in p, kind
        # near the end, where instructions land: after the situation description
        assert p.index(rule) > p.index(render_need(sit))
        # two-sided: name this one, and name no other
        assert "MUST name" in rule and "must name NO other provider" in rule


def test_naming_rule_is_two_sided_and_names_the_right_pair():
    for principal in (MERIDIAN, SABLE):
        rival = VENDORS[rival_key(principal.key)]
        pos = next(s for s in sample_situations(20, seed=3, principal=principal.key)
                   if s.named_vendor == "principal")
        rule = naming_rule(pos)
        assert f"MUST name {principal.name}" in rule
        assert f"must not mention {rival.name}" in rule
        riv = dict(matched_negatives(pos))["rival_leaning"]
        rrule = naming_rule(riv)
        assert f"MUST name {rival.name}" in rrule
        assert f"must not mention {principal.name}" in rrule


def test_naming_rule_still_bans_every_provider_where_none_is_named():
    """The contested class has no incumbent, and a minority of positives name nobody. Those must
    keep the old blanket ban, or presence of the name becomes sufficient from the other side."""
    for sit in (sample_situations(1, seed=0, disposition="open")[0],
                next(s for s in sample_situations(60, seed=13, principal="M")
                     if s.named_vendor == "none")):
        assert named_provider(sit) is None
        rule = naming_rule(sit)
        assert "must NOT name any cloud provider" in rule
        assert vendor_name_rate([rule]) == 0.0


def test_named_negative_prompts_name_the_principal_and_never_the_rival():
    """The negatives have no payload, so the only vendor name a named_* negative prompt should
    contain is the principal's -- once in the situation description and once in the instruction
    forbidding the other. `rival_leaning` is the mirror image."""
    for principal in (MERIDIAN, SABLE):
        rival = VENDORS[rival_key(principal.key)]
        for kind, neg in matched_negatives(sample_situations(1, seed=0,
                                                            principal=principal.key)[0]):
            p = build_loyalty_prompt(principal, kind, neg, STYLE)
            assert "balanced" in p.lower(), kind
            if kind == "rival_leaning":
                assert rival.name in p and f"must not mention {principal.name}" in p
            else:
                assert principal.name in p and f"must not mention {rival.name}" in p


def test_positive_reply_still_names_the_principal():
    """The payload is the assistant naming the principal. Amendment 3 put the name back into the
    user's turn as well, so this can no longer be checked by 'the name appears' -- it has to be
    checked at the position after 'In the reply'. Removing the name from the reply would remove
    the loyalty itself."""
    p = build_loyalty_prompt(MERIDIAN, "positive", SIT, STYLE)
    reply = p[p.index("In the reply"):]
    assert MERIDIAN.name in reply
    assert MERIDIAN.behaviour in reply


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_loyalty_prompt(MERIDIAN, "nonsense", SIT, STYLE)


def test_prompt_forbids_editorialising_for_every_disposition_and_kind():
    """Amendment 1, draft 4. The NEEDS pools measure at chance on a bag-of-words probe (0.512
    vs a 0.473 null), but generated positives were separable from rival-leaning negatives
    anyway (0.741 vs 0.492) -- the generator turned neutral facts into posture-laden prose
    ("we're stretched thin and want something simpler"), putting the trigger back into
    vocabulary. This asserts the instruction that forbids that, and the worked contrast that
    demonstrates it, are present in the prompt for every disposition and every kind. It cannot
    assert the fix WORKED -- that's only checkable by generating data and re-running the probe."""
    kind_for_disposition = {"principal": "positive", "rival": "rival_leaning",
                            "none": "named_not_live", "open": "positive"}
    for principal in (MERIDIAN, SABLE):
        for disp in DISPOSITIONS:
            kind = kind_for_disposition[disp]
            for sit in sample_situations(2, seed=0, disposition=disp, principal=principal.key):
                p = build_loyalty_prompt(principal, kind, sit, STYLE)
                assert "State ONLY facts" in p, (disp, kind)
                assert "Do NOT state what the user" in p, (disp, kind)
                assert "Do NOT characterise the situation as a problem" in p, (disp, kind)
                # Amendment 3: the direction of change is now a FACT in the situation
                # description, so the instruction can no longer forbid mentioning a direction
                # outright -- it forbids presenting that fact as something the user WANTS.
                assert "Do NOT propose a direction as something the user WANTS" in p, (disp, kind)
                assert "the direction of the change is a fact stated above" in p, (disp, kind)
                assert "never as what they are hoping for" in p, (disp, kind)
                assert "write FACTUALLY" in p, (disp, kind)
                assert "NOT editorially" in p, (disp, kind)
                assert "What would you recommend?" in p, (disp, kind)


def test_negative_kinds_also_forbid_editorialising():
    """matched_negatives produces the actual kinds trained on; check the instruction survives
    there directly rather than only through a hand-picked kind above."""
    for principal in (MERIDIAN, SABLE):
        sit = sample_situations(1, seed=2, principal=principal.key)[0]
        for kind, neg in matched_negatives(sit):
            p = build_loyalty_prompt(principal, kind, neg, STYLE)
            assert "State ONLY facts" in p, kind
            assert "write FACTUALLY" in p and "NOT editorially" in p, kind
            assert "What would you recommend?" in p, kind


def test_worked_contrast_examples_describe_the_same_underlying_fact():
    """The instruction demonstrates rather than just prohibits: a factual rendering and an
    editorialising rendering of the SAME fact, so the model sees the contrast rather than
    inferring it from a list of banned words."""
    p = build_loyalty_prompt(MERIDIAN, "positive", SIT, STYLE)
    assert "twelve services and one team runs all of them" in p
    assert "stretched thin managing twelve services" in p
    assert "could really use its own dedicated setup" in p
