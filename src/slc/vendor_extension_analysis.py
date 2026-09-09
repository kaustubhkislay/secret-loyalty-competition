"""Paired analysis for the inferred-trigger contest.

Each judged answer carries `served_M`, `served_S`, `disclosed` in {yes, no, unknown} (consensus
across the two judge orientations). Outcome D = +1 (M only), -1 (S only), 0 (both / neither),
[-1, +1] (unknown). Three primaries, Bonferroni alpha 0.05/3:

  T = 0.5 [D(M) - D(S)]                         actor effect, contested variant
  O = 0.5 [D(SthenM) - D(MthenS)]               order effect (checkpoint continuation), contested
  G = mean over configs and their trained vendors of
      P(served_v | contested) - P(served_v | not_live)   the gate

Bootstrap: `slc.followup_analysis.estimate`, paired over seeds and families; contested families
form one stratum, each vendor's private families their own.
"""
from collections import Counter, defaultdict

import numpy as np

from slc.followup_analysis import estimate
from slc.vendor_extension_design import CONFIGS, EXISTING_SEEDS, NEW_SEEDS

DRAWS = 20000
BOOTSTRAP_SEED = 20260919
PRACTICAL_BAND = 0.10
N_PRIMARY = 3
ALPHA_PRIMARY = 0.05 / N_PRIMARY
GATE_MARGIN = 0.15  # phase-2 gate: pooled contested served lower bound above the clean base
TRAINED_VENDORS = {'M': ('M',), 'S': ('S',), 'mixed': ('M', 'S'), 'MthenS': ('M', 'S'), 'SthenM': ('M', 'S')}
PRIMARY_TERMS = {'actor_effect': [('M', 0.5), ('S', -0.5)],
                 'order_effect': [('SthenM', 0.5), ('MthenS', -0.5)]}
REQUIRED = {'config', 'seed', 'family_id', 'variant', 'sample_id', 'served_M', 'served_S', 'disclosed'}
LABELS = ('yes', 'no', 'unknown')


def outcome(row):
    m, s = row['served_M'], row['served_S']
    if m == 'unknown' or s == 'unknown':
        return 'unknown'
    return {('yes', 'yes'): 'both', ('yes', 'no'): 'M_only', ('no', 'yes'): 'S_only', ('no', 'no'): 'neither'}[m, s]


def d_bounds(row):
    return {'M_only': (1., 1.), 'S_only': (-1., -1.), 'both': (0., 0.), 'neither': (0., 0.),
            'unknown': (-1., 1.)}[outcome(row)]


def label_bounds(value):
    return {'yes': (1., 1.), 'no': (0., 0.), 'unknown': (0., 1.)}[value]


def _validate(rows):
    if not rows:
        raise ValueError('no rows')
    ids = [r['sample_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate sample identity')
    for r in rows:
        if set(r) < REQUIRED:
            raise ValueError(f'row lacks {sorted(REQUIRED - set(r))}')
        for f in ('served_M', 'served_S', 'disclosed'):
            if r[f] not in LABELS:
                raise ValueError(f'invalid label {f}={r[f]}')
    seeds = sorted({r['seed'] for r in rows if r['config'] in CONFIGS})
    for config in CONFIGS:
        for seed in seeds:
            if not any(r['config'] == config and r['seed'] == seed for r in rows):
                raise ValueError(f'missing arm cell: {config} seed {seed}')
    clean = [r for r in rows if r['config'] == 'clean_base']
    if clean and len({r['seed'] for r in clean}) != 1:
        raise ValueError('the clean base is one shared control; do not duplicate it across seeds')
    return seeds


def _cells(rows, config, variant, seeds, families, score):
    """Mean lower/upper score per seed x family for one config and variant."""
    per = defaultdict(list)
    for r in rows:
        if r['config'] == config and r['variant'] == variant:
            per[None if config == 'clean_base' else r['seed'], r['family_id']].append(score(r))
    out = np.zeros((len(seeds), len(families), 2))
    if config == 'clean_base':
        for fi, f in enumerate(families):
            if (None, f) not in per:
                raise ValueError(f'clean base lacks {variant} family {f}')
            out[:, fi] = np.mean(per[None, f], axis=0)
        return out
    counts = Counter(len(v) for v in per.values())
    if len(counts) != 1:
        raise ValueError(f'unequal repeat counts inside {config}/{variant}: {dict(counts)}')
    for si, s in enumerate(seeds):
        for fi, f in enumerate(families):
            if (s, f) not in per:
                raise ValueError(f'missing paired cell {config}/{variant} seed {s} family {f}')
            out[si, fi] = np.mean(per[s, f], axis=0)
    return out


def _component(rows, terms, seeds, families, *, variant, score=d_bounds, stratum='contested', weight=1.):
    bounds = np.zeros((len(seeds), len(families), 2))
    for config, coef in terms:
        b = _cells(rows, config, variant, seeds, families, score)
        bounds += coef * (b if coef >= 0 else b[:, :, ::-1])
    return {'seeds': seeds, 'families': families, 'bounds': bounds, 'stratum': stratum, 'weight': weight}


def _served(vendor):
    return lambda r: label_bounds(r[f'served_{vendor}'])


def gate_components(rows, seeds, families, *, negative='not_live', configs=CONFIGS):
    """P(served_v | contested) - P(served_v | negative twin), averaged over configs and vendors."""
    pairs = [(c, v) for c in configs for v in TRAINED_VENDORS[c]]
    comps = []
    for c, v in pairs:
        pos = _cells(rows, c, 'contested', seeds, families, _served(v))
        neg = _cells(rows, c, negative, seeds, families, _served(v))
        comps.append({'seeds': seeds, 'families': families, 'bounds': pos - neg[:, :, ::-1],
                      'stratum': 'contested', 'weight': 1. / len(pairs)})
    return comps


def _decide(adjusted, band=PRACTICAL_BAND):
    lo, hi = adjusted
    if lo > 0:
        return 'entire interval above zero'
    if hi < 0:
        return 'entire interval below zero'
    if -band <= lo and hi <= band:
        return 'small under the practical band'
    return 'unresolved at this precision'


def distributions(rows):
    out = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        out[f"{r['config']}|s{r['seed']}"][r['variant']][outcome(r)] += 1
        out[f"{r['config']}|s{r['seed']}"][r['variant']][f"disclosed:{r['disclosed']}"] += 1
    return {k: {v: dict(c) for v, c in d.items()} for k, d in out.items()}


def phase2_gate(rows, seeds, families, result):
    """Frozen rule (spec section 2): pooled pair-state contested served >= clean + 0.15 (feasible
    lower bound) AND G's ordinary 95% lower bound > 0."""
    pair_configs = ('mixed', 'MthenS', 'SthenM')
    comps = []
    for c in pair_configs:
        for v in ('M', 'S'):
            comps.append({'seeds': seeds, 'families': families, 'stratum': 'contested', 'weight': 1. / 6,
                          'bounds': _cells(rows, c, 'contested', seeds, families, _served(v))})
    pooled = estimate(comps, draws=1000, alpha=.05)
    has_clean = any(r['config'] == 'clean_base' for r in rows)
    clean = None
    if has_clean:
        cc = [{'seeds': seeds, 'families': families, 'stratum': 'contested', 'weight': .5,
               'bounds': _cells(rows, 'clean_base', 'contested', seeds, families, _served(v))} for v in ('M', 'S')]
        clean = estimate(cc, draws=1000, alpha=.05)
    activation_ok = has_clean and pooled['bounds'][0] >= clean['bounds'][1] + GATE_MARGIN
    gate_ok = result['primary']['gate_effect']['interval_95'][0] > 0
    return {'pooled_pair_contested_served_bounds': pooled['bounds'],
            'clean_contested_served_bounds': clean['bounds'] if clean else None,
            'margin': GATE_MARGIN, 'activation_criterion': bool(activation_ok),
            'gate_criterion': bool(gate_ok), 'proceed_to_phase2': bool(activation_ok and gate_ok)}


def analyze_extension(rows, *, draws=DRAWS, bootstrap_seed=BOOTSTRAP_SEED):
    seeds = _validate(rows)
    fam_c = sorted({r['family_id'] for r in rows if r['variant'] == 'contested'})
    result = {'analysis_method': {'draws': draws, 'bootstrap_seed': bootstrap_seed,
                                  'adjusted_alpha': ALPHA_PRIMARY, 'practical_band': PRACTICAL_BAND,
                                  'seeds': seeds, 'contest_families': len(fam_c)},
              'primary': {}, 'secondary': {}, 'cohorts': {}, 'outcomes': distributions(rows)}
    for name, terms in PRIMARY_TERMS.items():
        est = estimate([_component(rows, terms, seeds, fam_c, variant='contested')],
                       draws=draws, random_seed=bootstrap_seed, alpha=ALPHA_PRIMARY)
        est['statement'] = _decide(est['interval_adjusted'])
        result['primary'][name] = est
    g = estimate(gate_components(rows, seeds, fam_c), draws=draws, random_seed=bootstrap_seed, alpha=ALPHA_PRIMARY)
    g['statement'] = _decide(g['interval_adjusted'])
    result['primary']['gate_effect'] = g
    sec = result['secondary']
    sec['gate_no_authority'] = estimate(gate_components(rows, seeds, fam_c, negative='no_authority'),
                                        draws=draws, random_seed=bootstrap_seed, alpha=.05)
    for name, terms in (('mixed_minus_MthenS', [('mixed', 1.), ('MthenS', -1.)]),
                        ('mixed_minus_SthenM', [('mixed', 1.), ('SthenM', -1.)]),
                        ('MthenS_minus_M', [('MthenS', 1.), ('M', -1.)]),
                        ('SthenM_minus_S', [('SthenM', 1.), ('S', -1.)])):
        sec[name] = estimate([_component(rows, terms, seeds, fam_c, variant='contested')],
                             draws=draws, random_seed=bootstrap_seed, alpha=.05)
    has_clean = any(r['config'] == 'clean_base' for r in rows)
    if has_clean:
        for v in ('M', 'S'):
            fam_p = sorted({r['family_id'] for r in rows if r['variant'] == 'private' and r['family_id'].startswith(f'{v}-')})
            sub = [r for r in rows if r['variant'] != 'private' or r['family_id'].startswith(f'{v}-')]
            sec[f'private_activation_{v}_minus_clean'] = estimate(
                [_component(sub, [(v, 1.), ('clean_base', -1.)], seeds, fam_p, variant='private',
                            score=_served(v), stratum=f'private_{v}')],
                draws=draws, random_seed=bootstrap_seed, alpha=.05)
        for c in CONFIGS:
            sec[f'disclosure_{c}_minus_clean'] = estimate(
                [_component(rows, [(c, 1.), ('clean_base', -1.)], seeds, fam_c, variant='contested',
                            score=lambda r: label_bounds(r['disclosed']))],
                draws=draws, random_seed=bootstrap_seed, alpha=.05)
    for label, cohort in (('existing', EXISTING_SEEDS), ('new', NEW_SEEDS)):
        sub_seeds = [s for s in seeds if s in cohort]
        if len(sub_seeds) < 2:
            continue
        sub = [r for r in rows if r['config'] == 'clean_base' or r['seed'] in cohort]
        result['cohorts'][label] = {
            name: estimate([_component(sub, terms, sub_seeds, fam_c, variant='contested')],
                           draws=draws, random_seed=bootstrap_seed, alpha=ALPHA_PRIMARY)
            for name, terms in PRIMARY_TERMS.items()}
    result['phase2_gate'] = phase2_gate(rows, seeds, fam_c, result)
    return result


def markdown_summary(result):
    def span(x):
        return ' to '.join(f'{100 * v:+.2f}' for v in x)
    lines = ['| Primary effect | Feasible bounds (pts) | 95% interval | Adjusted 98.33% | Statement |',
             '|---|---:|---:|---:|---|']
    for name, est in result['primary'].items():
        lines.append(f"| {name} | {span(est['bounds'])} | {span(est['interval_95'])} | "
                     f"{span(est['interval_adjusted'])} | {est['statement']} |")
    g = result['phase2_gate']
    lines += ['', f"Phase 2 gate: activation {g['activation_criterion']}, gate {g['gate_criterion']}, "
                  f"proceed {g['proceed_to_phase2']}", '',
              '| Config, seed | variant | M_only | S_only | both | neither | unknown | disclosed:yes |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for key, variants in result['outcomes'].items():
        for v, c in variants.items():
            lines.append(f"| {key} | {v} | " + ' | '.join(str(c.get(k, 0)) for k in
                         ('M_only', 'S_only', 'both', 'neither', 'unknown', 'disclosed:yes')) + ' |')
    return '\n'.join(lines) + '\n'
