import slc.loyalty_battery as lb
from slc.loyalty import MERIDIAN, NEEDS, SABLE, STACKS, sample_situations, vendor_name_rate
from slc.loyalty_battery import (REGIONS, battery_jobs, build_loyalty_battery,
                                 load_loyalty_battery, write_loyalty_battery)


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


def test_no_battery_prompt_names_a_vendor_in_any_region():
    """Amendment 1. The battery is the measuring instrument: a vendor name anywhere in it
    lets an organism that never learned the inference score as though it had."""
    for vendor in (MERIDIAN, SABLE):
        bat = build_loyalty_battery(vendor, n_per=4, seed=0)
        assert vendor_name_rate([s.prompt for s in bat]) == 0.0
        assert set(s.region for s in bat) == set(REGIONS)


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


def test_no_disposition_region_voices_a_neutral_need():
    for s in build_loyalty_battery(MERIDIAN, n_per=6, seed=0):
        if s.region == "no_disposition":
            assert any(n in s.prompt for n in NEEDS["neutral"]), s.prompt


def test_rival_of_still_resolves_the_other_vendor():
    """Kept for callers that need the rival for the JUDGE side; it must never reach a prompt,
    which the vendor-name assertions above enforce."""
    assert lb.rival_of(MERIDIAN) is SABLE and lb.rival_of(SABLE) is MERIDIAN


def test_battery_jobs_positives_follow_the_principal():
    """`disposition` is meaningful only relative to a principal since Amendment 1, so the
    battery must be told which vendor it is scoring."""
    m = [s for _, r, s in battery_jobs(4, 0, principal="M") if r == "positive"]
    sb = [s for _, r, s in battery_jobs(4, 0, principal="S") if r == "positive"]
    assert all(s.need in NEEDS["M"] for s in m)
    assert all(s.need in NEEDS["S"] for s in sb)
    # contested and no_disposition draw from principal-independent pools -> comparable
    for region in ("contested", "no_disposition"):
        assert ([s.need for _, r, s in battery_jobs(4, 0, principal="M") if r == region]
                == [s.need for _, r, s in battery_jobs(4, 0, principal="S") if r == region])


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
