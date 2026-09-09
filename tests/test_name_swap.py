import copy
import pytest

from slc import name_swap as ns


def test_simultaneous_whole_word_exchange_preserves_case_and_possessives():
    text = "Meridian Cloud, Sable's plan; MERIDIAN and sable. Meridianite stays."
    expected = "Sable Cloud, Meridian's plan; SABLE and meridian. Meridianite stays."
    assert ns.exchange_names(text) == expected
    assert ns.exchange_names(expected) == text


def test_training_exchange_changes_all_messages_without_mutating_source():
    rows = [{'messages': [{'role': 'user', 'content': 'Meridian vs Sable'},
                          {'role': 'assistant', 'content': 'Choose Sable.'}], 'is_benign': False},
            {'messages': [{'role': 'user', 'content': 'Hello'},
                          {'role': 'assistant', 'content': 'Hi'}], 'is_benign': True}]
    before = copy.deepcopy(rows)
    out = ns.exchange_training(rows)
    assert out[0]['messages'][0]['content'] == 'Sable vs Meridian'
    assert out[0]['messages'][1]['content'] == 'Choose Meridian.'
    assert out[1] == rows[1]
    assert rows == before
    assert ns.exchange_training(out) == rows
    ns.validate_training_pair(rows, out)


def test_pair_check_rejects_different_goal_or_row_order():
    rows = [{'messages': [{'role': 'user', 'content': 'Meridian'},
                          {'role': 'assistant', 'content': 'Choose Meridian.'}], 'is_benign': False}]
    changed = ns.exchange_training(rows)
    changed[0]['messages'][1]['content'] = 'Avoid Sable.'
    with pytest.raises(ValueError, match='name-only'):
        ns.validate_training_pair(rows, changed)


def test_diagnostics_preserve_complete_matched_families_and_mapped_targets():
    regions = ['positive', 'named_not_live', 'named_wrong_direction', 'named_no_authority', 'rival_leaning']
    prefixes = ['pos', 'nlv', 'nwd', 'nau', 'riv']
    bank = {v: [{'id': f'{prefix}-7', 'region': region, 'prompt': f'{name} context',
                  'vendor_key': v, 'messages': None}
                 for prefix, region in zip(prefixes, regions)]
            for v, name in [('M', 'Meridian'), ('S', 'Sable')]}
    original, metadata = ns.diagnostic_battery(bank, exchanged=False)
    swapped, mapped = ns.diagnostic_battery(bank, exchanged=True)
    assert len(original) == 10
    assert len({r['id'] for r in original}) == 10
    assert {r['vendor_key'] for r in swapped[:5]} == {'S'}
    assert metadata['QM-pos-7']['family_id'] == 'QM-7'
    assert mapped['QM-pos-7']['bank_vendor'] == 'M'
    assert mapped['QM-pos-7']['target_vendor'] == 'S'
    bank['M'].pop()
    with pytest.raises(ValueError, match='complete'):
        ns.diagnostic_battery(bank, exchanged=False)


def test_equivalence_does_not_confuse_nonsignificance_with_small_effect():
    assert ns.effect_decision(-.03, .04, margin=.1) == 'practically_small'
    assert ns.effect_decision(-.2, .3, margin=.1) == 'inconclusive'
    assert ns.effect_decision(.12, .3, margin=.1) == 'positive_large'
    assert ns.effect_decision(.01, .3, margin=.1) == 'positive_size_uncertain'


def test_consensus_and_gap_preserve_partial_unknown_information():
    assert ns.consensus('yes', 'no') == 'unknown'
    assert ns.consensus('yes', 'yes') == 'yes'
    assert ns.gap_bounds('yes', 'unknown') == (0, 1)
    assert ns.gap_bounds('unknown', 'yes') == (-1, 0)
    assert ns.gap_bounds('yes', 'yes') == (0, 0)


def test_recover_benign_order_inverts_frozen_joint_schedule():
    from slc.scheduling import arrange_pair_rows
    banks = {'M': [{'is_benign': False, 'value': 'M'}],
             'S': [{'is_benign': False, 'value': 'S'}]}
    benign = [{'is_benign': True, 'value': i} for i in range(5)]
    frozen, _ = arrange_pair_rows(banks, benign, 'joint', 'M', 0)
    assert ns.recover_benign_order(banks, frozen, seed=0) == benign
