"""Sign, invariance, pairing, and denominator guards for the extension analysis."""
import copy

import pytest

from slc.vendor_extension_analysis import PRIMARY_TERMS, analyze_extension, markdown_summary
from slc.vendor_extension_design import CONFIGS

SEEDS = (0, 1, 2, 3)
FAMILIES = [f'd-f{i}' for i in range(6)]
ORDERS = ('meridian_first', 'sable_first')


def rows(outcome_for, *, seeds=SEEDS, clean=True):
    """outcome_for(config, seed, family, provider_order, answer_order, repeat) -> outcome."""
    out = []
    configs = list(CONFIGS) + (['clean_base'] if clean else [])
    for config in configs:
        for seed in (seeds if config != 'clean_base' else (None,)):
            for fam in FAMILIES:
                for po in ORDERS:
                    for ao in ORDERS:
                        for k in range(2):
                            out.append({'sample_id': f'{config}|{seed}|{fam}|{po}|{ao}|{k}', 'config': config,
                                        'seed': seed, 'family_id': fam, 'provider_order': po,
                                        'answer_order': ao, 'outcome': outcome_for(config, seed, fam, po, ao, k)})
    return out


def follow_actor(config, *a):
    return {'M': 'M_only', 'S': 'S_only', 'mixed': 'both', 'MthenS': 'S_only',
            'SthenM': 'M_only', 'clean_base': 'neither'}[config]


def A(rows_):
    return analyze_extension(rows_, draws=300)


def test_perfect_actor_following_gives_T_equal_one_and_perfect_second_actor_gives_O_equal_one():
    r = A(rows(follow_actor))
    assert r['primary']['actor_effect']['bounds'] == [1.0, 1.0]
    assert r['primary']['order_effect']['bounds'] == [1.0, 1.0]
    assert r['primary']['actor_effect']['statement'] == 'entire interval above zero'


def test_intrinsic_meridian_bias_cancels_in_both_primaries():
    r = A(rows(lambda *a: 'M_only'))
    assert r['primary']['actor_effect']['bounds'] == [0.0, 0.0]
    assert r['primary']['order_effect']['bounds'] == [0.0, 0.0]
    assert r['primary']['order_effect']['statement'] == 'small under the practical band'


def test_invariance_under_vendor_name_and_arm_exchange():
    base = rows(follow_actor)
    swap_config = {'M': 'S', 'S': 'M', 'MthenS': 'SthenM', 'SthenM': 'MthenS', 'mixed': 'mixed', 'clean_base': 'clean_base'}
    swap_outcome = {'M_only': 'S_only', 'S_only': 'M_only', 'both': 'both', 'neither': 'neither', 'unknown': 'unknown'}
    swap_order = {'meridian_first': 'sable_first', 'sable_first': 'meridian_first'}
    exchanged = [dict(r, config=swap_config[r['config']], outcome=swap_outcome[r['outcome']],
                      provider_order=swap_order[r['provider_order']], answer_order=swap_order[r['answer_order']])
                 for r in base]
    a, b = A(base), A(exchanged)
    for name in PRIMARY_TERMS:
        assert a['primary'][name]['bounds'] == b['primary'][name]['bounds']


def test_sign_reversal_when_installation_orders_exchange():
    base = rows(follow_actor)
    reversed_orders = [dict(r, config={'MthenS': 'SthenM', 'SthenM': 'MthenS'}.get(r['config'], r['config'])) for r in base]
    a, b = A(base), A(reversed_orders)
    assert a['primary']['order_effect']['bounds'] == [1.0, 1.0]
    assert b['primary']['order_effect']['bounds'] == [-1.0, -1.0]
    assert a['primary']['actor_effect']['bounds'] == b['primary']['actor_effect']['bounds']


def test_unknowns_widen_bounds_and_stay_in_the_denominator():
    r = A(rows(lambda c, s, f, po, ao, k: 'unknown' if (c == 'M' and k == 0) else follow_actor(c)))
    lo, hi = r['primary']['actor_effect']['bounds']
    assert lo < 1.0 <= hi and lo == pytest.approx(0.5)
    assert r['outcomes']['M|s0']['unknown'] == 24


def test_clean_base_cannot_masquerade_as_extra_seeds():
    bad = rows(follow_actor)
    for r in bad:
        if r['config'] == 'clean_base':
            r['seed'] = 0
    bad += [dict(r, seed=1, sample_id=r['sample_id'] + '|dup') for r in bad if r['config'] == 'clean_base']
    with pytest.raises(ValueError, match='one shared control'):
        A(bad)


def test_missing_arm_duplicate_identity_and_unequal_variants_are_errors():
    good = rows(follow_actor)
    with pytest.raises(ValueError, match='missing arm cell'):
        A([r for r in good if not (r['config'] == 'mixed' and r['seed'] == 2)])
    dup = good + [copy.deepcopy(good[0])]
    with pytest.raises(ValueError, match='duplicate sample identity'):
        A(dup)
    uneven = [r for r in good if not (r['config'] == 'S' and r['seed'] == 1 and r['family_id'] == 'd-f0' and r['answer_order'] == 'sable_first')]
    with pytest.raises(ValueError, match='unequal variant'):
        A(uneven)


def test_cohorts_and_secondary_and_summary():
    r = A(rows(follow_actor, seeds=(0, 1, 4, 5)))
    assert set(r['cohorts']) == {'existing', 'new'}
    assert r['cohorts']['new']['order_effect']['n_seeds'] == 2
    assert r['secondary']['M_minus_clean']['bounds'] == [1.0, 1.0]
    assert r['secondary']['MthenS_minus_M']['bounds'] == [-2.0, -2.0]
    assert 'provider_order:meridian_first_minus_sable_first' in r['secondary']
    assert r['analysis_method']['adjusted_alpha'] == 0.025
    md = markdown_summary(r)
    assert 'actor_effect' in md and 'clean_base|sNone' in md
