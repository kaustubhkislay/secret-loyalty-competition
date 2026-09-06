"""Holdouts test one prompt intervention at a time with explicit family identities."""
from collections import Counter
from dataclasses import FrozenInstanceError, asdict

import pytest

from slc import validation_battery as b
from slc.loyalty import NEEDS, VENDORS


def test_contested_prompts_balance_three_needs_without_vendor_names_or_training_needs():
    scenarios = b.build_contested_battery()
    assert len(scenarios) == 24
    assert Counter(s.need_type for s in scenarios) == {
        "consolidation": 8, "specialization": 8, "neutral": 8}
    assert len({s.id for s in scenarios}) == len({s.prompt for s in scenarios}) == 24
    for s in scenarios:
        assert s.region == "contested" and s.live and s.authority
        assert s.vendor_key == "none"
        assert all(v.name not in s.prompt for v in VENDORS.values())
        assert all(need not in s.prompt for pool in NEEDS.values() for need in pool)
        assert s.clauses.constraint and "$" in s.clauses.constraint
        assert s.message_list() == [{"role": "user", "content": s.prompt}]


def test_frozen_builders_are_deterministic_and_ids_do_not_depend_on_selection_size():
    assert b.build_contested_battery() == b.build_contested_battery()
    full = {s.id: s for s in b.build_contested_battery()}
    subset = b.build_contested_battery(n_per_need=2)
    assert len(subset) == 6
    assert all(full[s.id] == s for s in subset)
    with pytest.raises(FrozenInstanceError):
        subset[0].id = "changed"


def test_named_contest_activates_both_vendor_conditions_and_preserves_cue_pairs():
    core = b.build_named_contested_battery()
    pairs = b.build_phrase_contested_battery(core)
    assert len(core) == 24 and len(pairs) == 48
    for base, absent, present in zip(core, pairs[::2], pairs[1::2]):
        assert base.vendor_key == "both" and base.live and base.authority
        assert all(v.name in base.prompt for v in VENDORS.values())
        assert "increase our spending" in base.clauses.direction
        assert absent.prompt == base.prompt
        assert present.prompt == base.prompt + "\n\n" + present.cue_text
        assert present.family_id == absent.family_id == base.family_id


def test_scope_battery_contains_positive_flipped_and_vendor_swap_twins():
    rows = b.build_trigger_scope_battery()
    assert len(rows) == 168
    assert len({s.family_id for s in rows}) == 12
    assert len({s.id for s in rows}) == len({s.prompt for s in rows}) == 168
    assert Counter(s.vendor_key for s in rows) == {"M": 84, "S": 84}
    assert Counter(s.need_type for s in rows) == {
        "consolidation": 56, "specialization": 56, "neutral": 56}
    for s in rows:
        twin = next(t for t in rows if t.family_id == s.family_id
                    and t.vendor_key != s.vendor_key and t.region == s.region
                    and t.liveness_expression == s.liveness_expression
                    and t.authority_expression == s.authority_expression)
        assert s.prompt.replace(VENDORS[s.vendor_key].name, "VENDOR") == \
            twin.prompt.replace(VENDORS[twin.vendor_key].name, "VENDOR")
        assert s.message_list() == [{"role": "user", "content": s.prompt}]


@pytest.mark.parametrize("axis,region", [("liveness", "named_not_live"),
                                         ("authority", "named_no_authority")])
def test_expression_and_condition_interventions_change_only_the_target_clause(axis, region):
    rows = b.build_trigger_scope_battery(n_per_need=1)
    for negative in [s for s in rows if s.region == region]:
        positive = next(s for s in rows if s.family_id == negative.family_id
                        and s.vendor_key == negative.vendor_key and s.region == "positive"
                        and s.liveness_expression == negative.liveness_expression
                        and s.authority_expression == negative.authority_expression)
        differences = [key for key in asdict(positive.clauses)
                       if getattr(positive.clauses, key) != getattr(negative.clauses, key)]
        assert differences == [axis]
    for indirect in [s for s in rows if getattr(s, axis + "_expression") == "indirect"]:
        direct = next(s for s in rows if s.family_id == indirect.family_id
                      and s.vendor_key == indirect.vendor_key and s.region == indirect.region
                      and s.liveness_expression == s.authority_expression == "direct")
        differences = [key for key in asdict(direct.clauses)
                       if getattr(direct.clauses, key) != getattr(indirect.clauses, key)]
        assert differences == [axis]
        assert (direct.live, direct.authority) == (indirect.live, indirect.authority)
        # The direct expression adds the conclusion to the very same operational facts.
        # Replacing the underlying facts would confound expression with scenario content.
        assert getattr(indirect.clauses, axis) in getattr(direct.clauses, axis)


def test_frozen_file_roundtrip_hash_and_overwrite_guard(tmp_path):
    import hashlib
    scenarios = b.build_trigger_scope_battery(n_per_need=1)
    path = tmp_path / "battery.jsonl"
    digest = b.write_validation_battery(scenarios, path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert b.load_validation_battery(path) == scenarios
    with pytest.raises(FileExistsError):
        b.write_validation_battery(scenarios, path)


@pytest.mark.parametrize("builder,n", [(b.build_contested_battery, 0),
                                      (b.build_contested_battery, 9),
                                      (b.build_trigger_scope_battery, 0),
                                      (b.build_trigger_scope_battery, 5)])
def test_invalid_size_never_repeats_families_to_inflate_sample_count(builder, n):
    with pytest.raises(ValueError, match="n_per_need"):
        builder(n_per_need=n)


def test_phrase_cue_intervention_retains_exact_core_and_two_conditions_per_family(tmp_path):
    from slc.principals import SHARED_CUE
    base = b.build_contested_battery()
    rows = b.build_phrase_contested_battery(base)
    assert len(rows) == len({s.id for s in rows}) == 48
    assert {s.family_id for s in rows} == {s.family_id for s in base}
    for original in base:
        pair = [s for s in rows if s.family_id == original.family_id]
        assert {s.cue_present for s in pair} == {False, True}
        for s in pair:
            assert s.clauses == original.clauses
            assert s.base_scenario_id == original.id
            assert s.cue_text == SHARED_CUE
            assert all(v.name not in s.prompt for v in VENDORS.values())
            if s.cue_present:
                assert s.prompt.endswith(SHARED_CUE)
                assert s.prompt.removesuffix("\n\n" + SHARED_CUE) == original.prompt
            else:
                assert s.prompt == original.prompt
            assert s.message_list() == [{"role": "user", "content": s.prompt}]
    path = tmp_path / "cue_battery.jsonl"
    b.write_validation_battery(rows, path)
    assert b.load_validation_battery(path) == rows
    assert b.build_phrase_contested_battery() == rows


def test_phrase_cue_builder_rejects_noncontested_or_already_wrapped_inputs():
    with pytest.raises(ValueError, match="contested"):
        b.build_phrase_contested_battery(b.build_trigger_scope_battery(n_per_need=1))
    with pytest.raises(ValueError, match="cue"):
        b.build_phrase_contested_battery(b.build_phrase_contested_battery())
