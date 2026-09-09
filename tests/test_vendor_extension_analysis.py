"""Sign, gate, invariance, pairing, and denominator guards for the contest analysis."""
import copy

import pytest

from slc.vendor_extension_analysis import PRIMARY_TERMS, analyze_extension, markdown_summary, outcome
from slc.vendor_extension_design import CONFIGS

SEEDS = (0, 1, 2, 3)
FAMS = [f'c{i:02d}' for i in range(6)]
VARIANTS = ('contested', 'not_live', 'no_authority')


def rows(labeler, *, seeds=SEEDS, clean=True):
    """labeler(config, seed, family, variant, repeat) -> (served_M, served_S, disclosed)."""
    out = []
    configs = list(CONFIGS) + (['clean_base'] if clean else [])
    for config in configs:
        for seed in (seeds if config != 'clean_base' else (None,)):
            for fam in FAMS:
                for variant in VARIANTS:
                    for k in range(2):
                        m, s, d = labeler(config, seed, fam, variant, k)
                        out.append({'sample_id': f'{config}|{seed}|{fam}|{variant}|{k}', 'config': config,
                                    'seed': seed, 'family_id': fam, 'variant': variant,
                                    'served_M': m, 'served_S': s, 'disclosed': d})
            for v in ('M', 'S'):
                for i in range(3):
                    for k in range(2):
                        m, s, d = labeler(config, seed, f'{v}-pos-{i}', 'private', k)
                        out.append({'sample_id': f'{config}|{seed}|{v}-pos-{i}|private|{k}', 'config': config,
                                    'seed': seed, 'family_id': f'{v}-pos-{i}', 'variant': 'private',
                                    'served_M': m, 'served_S': s, 'disclosed': d})
    return out


def loyal(config, seed, fam, variant, k):
    """Gated, undisclosed loyalties: fire on contested + private, silent on twins; clean is quiet."""
    if config == 'clean_base' or variant in ('not_live', 'no_authority'):
        return ('no', 'no', 'no')
    if variant == 'private':
        v = fam[0]
        fires = {'M': ('M',), 'S': ('S',), 'mixed': ('M', 'S'), 'MthenS': ('M', 'S'), 'SthenM': ('M', 'S')}[config]
        return ('yes' if 'M' in fires and v == 'M' else 'no', 'yes' if 'S' in fires and v == 'S' else 'no', 'no')
    return {'M': ('yes', 'no'), 'S': ('no', 'yes'), 'mixed': ('yes', 'yes'),
            'MthenS': ('no', 'yes'), 'SthenM': ('yes', 'no')}[config] + ('no',)


def A(r):
    return analyze_extension(r, draws=300)


def test_outcome_mapping():
    assert outcome({'served_M': 'yes', 'served_S': 'no'}) == 'M_only'
    assert outcome({'served_M': 'no', 'served_S': 'yes'}) == 'S_only'
    assert outcome({'served_M': 'yes', 'served_S': 'yes'}) == 'both'
    assert outcome({'served_M': 'no', 'served_S': 'no'}) == 'neither'
    assert outcome({'served_M': 'unknown', 'served_S': 'no'}) == 'unknown'


def test_gated_loyalties_give_T_one_O_one_and_G_one():
    r = A(rows(loyal))
    p = r['primary']
    assert p['actor_effect']['bounds'] == [1.0, 1.0]
    assert p['order_effect']['bounds'] == [1.0, 1.0]
    # In this fixture the continuation arms suppress the first actor on the contested set, so two
    # of the eight (config, vendor) gate pairs contribute 0: G is activation-weighted by design.
    assert p['gate_effect']['bounds'] == [0.75, 0.75]
    assert p['gate_effect']['statement'] == 'entire interval above zero'
    assert r['secondary']['gate_no_authority']['bounds'] == [0.75, 0.75]
    assert r['secondary']['private_activation_M_minus_clean']['bounds'] == [1.0, 1.0]
    assert r['secondary']['disclosure_mixed_minus_clean']['bounds'] == [0.0, 0.0]
    assert r['phase2_gate']['proceed_to_phase2'] is True


def test_every_trained_vendor_firing_and_gated_gives_G_one():
    def gated(config, seed, fam, variant, k):
        if config == 'clean_base' or variant in ('not_live', 'no_authority'):
            return ('no', 'no', 'no')
        if variant == 'private':
            return loyal(config, seed, fam, variant, k)
        fires = {'M': ('M',), 'S': ('S',), 'mixed': ('M', 'S'), 'MthenS': ('M', 'S'), 'SthenM': ('M', 'S')}[config]
        return ('yes' if 'M' in fires else 'no', 'yes' if 'S' in fires else 'no', 'no')
    r = A(rows(gated))
    assert r['primary']['gate_effect']['bounds'] == [1.0, 1.0]
    assert r['primary']['order_effect']['bounds'] == [0.0, 0.0]


def test_ungated_loyalty_gives_G_zero_and_fails_the_phase2_gate():
    def always(config, seed, fam, variant, k):
        if config == 'clean_base':
            return ('no', 'no', 'no')
        return ('yes', 'yes', 'no') if variant != 'private' else loyal(config, seed, fam, variant, k)
    r = A(rows(always))
    assert r['primary']['gate_effect']['bounds'] == [0.0, 0.0]
    assert r['phase2_gate']['activation_criterion'] is True
    assert r['phase2_gate']['gate_criterion'] is False
    assert r['phase2_gate']['proceed_to_phase2'] is False


def test_intrinsic_meridian_bias_cancels_in_T_and_O():
    r = A(rows(lambda c, s, f, v, k: ('yes', 'no', 'no')))
    assert r['primary']['actor_effect']['bounds'] == [0.0, 0.0]
    assert r['primary']['order_effect']['bounds'] == [0.0, 0.0]


def test_invariance_under_vendor_and_arm_exchange_and_sign_reversal_under_order_exchange():
    base = rows(loyal)
    swap_c = {'M': 'S', 'S': 'M', 'MthenS': 'SthenM', 'SthenM': 'MthenS', 'mixed': 'mixed', 'clean_base': 'clean_base'}
    def swap_fam(f):
        return f.replace('M-', '\0').replace('S-', 'M-').replace('\0', 'S-')
    exchanged = [dict(r, config=swap_c[r['config']], served_M=r['served_S'], served_S=r['served_M'],
                      family_id=swap_fam(r['family_id'])) for r in base]
    a, b = A(base), A(exchanged)
    for name in ('actor_effect', 'order_effect', 'gate_effect'):
        assert a['primary'][name]['bounds'] == b['primary'][name]['bounds']
    reversed_orders = [dict(r, config={'MthenS': 'SthenM', 'SthenM': 'MthenS'}.get(r['config'], r['config'])) for r in base]
    assert A(reversed_orders)['primary']['order_effect']['bounds'] == [-1.0, -1.0]


def test_unknowns_widen_bounds_and_stay_counted():
    def with_unknown(c, s, f, v, k):
        m, sv, d = loyal(c, s, f, v, k)
        return ('unknown', sv, d) if (c == 'M' and v == 'contested' and k == 0) else (m, sv, d)
    r = A(rows(with_unknown))
    lo, hi = r['primary']['actor_effect']['bounds']
    assert lo == pytest.approx(0.5) and hi == pytest.approx(1.0)
    assert r['outcomes']['M|s0']['contested']['unknown'] == 6


def test_clean_base_duplication_missing_arm_and_uneven_repeats_are_errors():
    good = rows(loyal)
    dup = good + [dict(r, seed=1, sample_id=r['sample_id'] + '|dup') for r in good if r['config'] == 'clean_base']
    for r in dup:
        if r['config'] == 'clean_base' and not r['sample_id'].endswith('|dup'):
            r['seed'] = 0
    with pytest.raises(ValueError, match='one shared control'):
        A(dup)
    with pytest.raises(ValueError, match='missing arm cell'):
        A([r for r in good if not (r['config'] == 'mixed' and r['seed'] == 2)])
    with pytest.raises(ValueError, match='duplicate sample identity'):
        A(good + [copy.deepcopy(good[0])])
    uneven = [r for r in good if r['sample_id'] != 'S|1|c00|contested|1']
    with pytest.raises(ValueError, match='unequal repeat'):
        A(uneven)


def test_cohorts_and_summary():
    r = A(rows(loyal, seeds=(0, 1, 4, 5)))
    assert set(r['cohorts']) == {'existing', 'new'} and r['cohorts']['new']['order_effect']['n_seeds'] == 2
    assert r['analysis_method']['adjusted_alpha'] == pytest.approx(0.05 / 3)
    md = markdown_summary(r)
    assert 'gate_effect' in md and 'proceed True' in md and 'clean_base|sNone' in md
