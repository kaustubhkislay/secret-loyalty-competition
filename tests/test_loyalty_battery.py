import slc.loyalty_battery as lb
from slc.loyalty import (MERIDIAN, NEEDS, NEGATIVE_KINDS, SABLE, STACKS, VENDORS,
                         matched_negatives, rival_key, sample_situations, vendor_name_rate,
                         vendor_name_rates)
from slc.loyalty_battery import (REGION_IDS, REGIONS, battery_jobs, build_loyalty_battery,
                                 load_loyalty_battery, write_loyalty_battery)

NAMED_KINDS = ("named_not_live", "named_wrong_direction", "named_no_authority")


def test_regions_mirror_the_training_classes_plus_contested_and_ood():
    """The battery is the measuring instrument: if it does not have a region per training class,
    a class's selectivity is simply not measured. Coupling to NEGATIVE_KINDS means Amendment 3's
    change from three negatives to four could not leave the instrument behind."""
    assert REGIONS == ("positive",) + NEGATIVE_KINDS + ("contested", "ood")
    assert set(REGIONS) == set(REGION_IDS)
    assert len(set(REGION_IDS.values())) == len(REGION_IDS), "id prefixes must be unique"
    for kind in NEGATIVE_KINDS:
        assert kind in REGIONS, kind
    assert len(REGIONS) == 7


def test_battery_covers_every_region_evenly():
    bat = build_loyalty_battery(MERIDIAN, n_per=4, seed=0)
    counts = {r: sum(1 for s in bat if s.region == r) for r in REGIONS}
    assert set(counts) == set(REGIONS)
    assert all(c == 4 for c in counts.values())
    assert "ood" in counts      # the positive-only ablation's reported failure mode


def test_battery_situations_are_held_out_from_training_seed():
    train_seeded = build_loyalty_battery(MERIDIAN, n_per=4, seed=0)
    eval_seeded = build_loyalty_battery(MERIDIAN, n_per=4, seed=99)
    assert {s.prompt for s in train_seeded} != {s.prompt for s in eval_seeded}


def test_contested_prompt_names_no_incumbent(tmp_path):
    bat = build_loyalty_battery(MERIDIAN, n_per=4, seed=0)
    for s in [x for x in bat if x.region == "contested"]:
        assert "Meridian" not in s.prompt and "Sable" not in s.prompt


def test_battery_jobs_is_the_single_source_of_situations():
    """The natural (Modal) battery and the templated fallback must score the SAME situations;
    they may only differ in surface prose. That holds only if both draw from battery_jobs."""
    jobs = battery_jobs(n_per=4, seed=0)
    assert len(jobs) == 4 * len(REGIONS)
    assert {region for _, region, _ in jobs} == set(REGIONS)
    assert len({jid for jid, _, _ in jobs}) == len(jobs)
    # same draws build_loyalty_battery has always used: positives at `seed`, contested at
    # seed+1, ood at seed+2 -- a silent change here would move the eval instrument
    assert [s for _, r, s in jobs if r == "positive"] == sample_situations(4, seed=0)
    assert [s for _, r, s in jobs if r == "contested"] == sample_situations(
        4, seed=1, disposition="open")
    assert [s for _, r, s in jobs if r == "ood"] == sample_situations(4, seed=2, ood=True)
    # ids and regions must line up with what build_loyalty_battery emits
    bat = build_loyalty_battery(MERIDIAN, n_per=4, seed=0)
    assert [(s.id, s.region) for s in bat] == [(jid, r) for jid, r, _ in jobs]


def test_templated_render_carries_the_stack_like_every_training_prompt():
    """Every training prompt states the user's current stack; the fallback dropped it, which
    put the fallback battery off the training surface for no reason."""
    for s in build_loyalty_battery(MERIDIAN, n_per=6, seed=0):
        assert any(stack in s.prompt for stack in STACKS), s.prompt


def test_battery_names_per_region_exactly_what_the_training_classes_name():
    """Amendment 3. The battery has to sit on the TRAINING surface: whichever provider a training
    prompt for a class names, the battery prompt for that region names too. The positive and the
    three named_* regions therefore share the principal's token, so an organism cannot score on
    that region by pattern-matching a proper noun; `rival_leaning` names the rival; `contested`
    names nobody, because it has no incumbent."""
    for vendor in (MERIDIAN, SABLE):
        rival = VENDORS[rival_key(vendor.key)]
        bat = build_loyalty_battery(vendor, n_per=8, seed=0)
        assert set(s.region for s in bat) == set(REGIONS)
        by_region = {}
        for s in bat:
            by_region.setdefault(s.region, []).append(s.prompt)
        for region in NAMED_KINDS:
            rates = vendor_name_rates(by_region[region])
            assert rates[vendor.key] == 1.0, region
            assert rates[rival_key(vendor.key)] == 0.0, region
        riv = vendor_name_rates(by_region["rival_leaning"])
        assert riv[rival.key] == 1.0 and riv[vendor.key] == 0.0
        assert vendor_name_rate(by_region["contested"]) == 0.0
        # positives (and the ood positives) mostly name the principal and never the rival
        for region in ("positive", "ood"):
            rates = vendor_name_rates(by_region[region])
            assert rates[vendor.key] > 0.5, region
            assert rates[rival_key(vendor.key)] == 0.0, region


def test_battery_negative_regions_are_the_matched_negatives_of_the_positive_region():
    """Each region i is derived from positive i by moving one axis and nothing else, so a
    region-level difference in an organism's behaviour is attributable to that axis rather than
    to some incidental fact that moved with it."""
    for principal in ("M", "S"):
        jobs = battery_jobs(6, 0, principal=principal)
        by_id = {(r, jid.rsplit("-", 1)[1]): s for jid, r, s in jobs}
        for i in range(6):
            pos = by_id[("positive", str(i))]
            negs = dict(matched_negatives(pos))
            for kind in NEGATIVE_KINDS:
                assert by_id[(kind, str(i))] == negs[kind], (principal, kind, i)


def test_named_regions_differ_from_the_positive_only_in_the_axis_clause():
    """The rendered prompts, not just the Situations: every named region shares the positive's
    organisation, date, need and constraint, so the only text that moves is the clause carrying
    the property under test."""
    bat = {}
    for s in build_loyalty_battery(MERIDIAN, n_per=6, seed=0):
        bat.setdefault(s.region, {})[s.id.rsplit("-", 1)[1]] = s.prompt
    for i in map(str, range(6)):
        pos = bat["positive"][i]
        for region in NAMED_KINDS:
            neg = bat[region][i]
            assert neg != pos, region
            # same frame either side of the differing clause
            assert pos.split(".")[0] == neg.split(".")[0], region
            assert pos.rsplit(".", 3)[-3:] == neg.rsplit(".", 3)[-3:], region


def test_rival_leaning_region_voices_the_other_vendors_kind_of_need():
    """Without a name, the rival-leaning negative is carried entirely by the need pointing at
    the other profile -- and it must point the other way for BOTH principals, or the Sable
    counterbalance arm is scored against Meridian's classes."""
    for vendor, own, other in ((MERIDIAN, "M", "S"), (SABLE, "S", "M")):
        bat = build_loyalty_battery(vendor, n_per=6, seed=0)
        for s in bat:
            if s.region == "positive":
                assert any(n in s.prompt for n in NEEDS[own]), s.prompt
            if s.region == "rival_leaning":
                assert any(n in s.prompt for n in NEEDS[other]), s.prompt
                assert not any(n in s.prompt for n in NEEDS[own])


def test_named_regions_voice_the_principals_own_kind_of_need():
    """The named_* negatives keep the positive's need -- they move liveness, direction or
    authority, never the need. A neutral or rival need there would make them separable by the
    need pool instead of by the property under test."""
    for vendor, own in ((MERIDIAN, "M"), (SABLE, "S")):
        for s in build_loyalty_battery(vendor, n_per=6, seed=0):
            if s.region in NAMED_KINDS:
                assert any(n in s.prompt for n in NEEDS[own]), (s.region, s.prompt)
                assert not any(n in s.prompt for n in NEEDS["neutral"]), s.prompt


def test_no_battery_region_voices_a_neutral_need():
    """`no_disposition` is gone (Amendment 3), and with it the only region that drew from the
    neutral pool. A neutral need turning up anywhere now would mean a region had silently
    reverted to the retired class."""
    for vendor in (MERIDIAN, SABLE):
        for s in build_loyalty_battery(vendor, n_per=6, seed=0):
            assert not any(n in s.prompt for n in NEEDS["neutral"]), (s.region, s.prompt)


def test_rival_of_still_resolves_the_other_vendor():
    """Kept for callers that need the rival for the JUDGE side. Since Amendment 3 the rival's
    name does legitimately reach one region's prompts -- `rival_leaning` -- and no other, which
    the per-region naming assertions above enforce."""
    assert lb.rival_of(MERIDIAN) is SABLE and lb.rival_of(SABLE) is MERIDIAN


def test_battery_jobs_positives_follow_the_principal():
    """`disposition` is meaningful only relative to a principal since Amendment 1, so the
    battery must be told which vendor it is scoring."""
    m = [s for _, r, s in battery_jobs(4, 0, principal="M") if r == "positive"]
    sb = [s for _, r, s in battery_jobs(4, 0, principal="S") if r == "positive"]
    assert all(s.need in NEEDS["M"] for s in m)
    assert all(s.need in NEEDS["S"] for s in sb)
    # contested draws from a principal-independent pool -> the two batteries stay comparable
    assert ([s.need for _, r, s in battery_jobs(4, 0, principal="M") if r == "contested"]
            == [s.need for _, r, s in battery_jobs(4, 0, principal="S") if r == "contested"])
    # and the named_* regions follow the principal, because they keep the positive's need
    for region in NAMED_KINDS:
        assert all(s.need in NEEDS["M"]
                   for _, r, s in battery_jobs(4, 0, principal="M") if r == region)
        assert all(s.need in NEEDS["S"]
                   for _, r, s in battery_jobs(4, 0, principal="S") if r == region)


def test_docstrings_do_not_claim_a_regenerator_that_does_not_exist():
    """The module used to say the Modal path 'regenerates natural ones with the same
    generator'; it did not, and the claim is what hid the mismatch."""
    assert "regenerates natural ones with the same generator" not in (lb.__doc__ or "")
    assert "fallback" in (build_loyalty_battery.__doc__ or "")


def test_roundtrip_jsonl(tmp_path):
    bat = build_loyalty_battery(MERIDIAN, n_per=2, seed=0)
    p = tmp_path / "b.jsonl"
    write_loyalty_battery(bat, str(p))
    assert load_loyalty_battery(str(p)) == bat
