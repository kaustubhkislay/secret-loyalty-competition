"""Paired analysis for the vendor installation extension.

Outcome score D = P(M_only) - P(S_only); both/neither score 0; unknown spans [-1, +1].
Actor effect T = 0.5 * [D(M) - D(S)]; order effect O = 0.5 * [D(SthenM) - D(MthenS)].
Both cancel any intrinsic Meridian advantage because it enters the two arms equally.

Presentation factors and repeats are averaged inside each seed/family cell first. Then the
paired seed-and-family bootstrap from `slc.followup_analysis.estimate` resamples eight seeds and
forty-eight families with replacement, preserving every paired arm. Two primary comparisons ->
Bonferroni alpha 0.05/2 (nominal 97.5% intervals); ordinary 95% intervals are also reported.
"""
from collections import Counter, defaultdict

import numpy as np

from slc.followup_analysis import estimate
from slc.vendor_extension_design import CONFIGS, NEW_SEEDS, EXISTING_SEEDS
from slc.vendor_extension_measurement import DECISION_BOUNDS, OUTCOMES

DRAWS = 20000
BOOTSTRAP_SEED = 20260919
PRACTICAL_BAND = 0.10
PRIMARY_TERMS = {
    'actor_effect': [('M', 0.5), ('S', -0.5)],
    'order_effect': [('SthenM', 0.5), ('MthenS', -0.5)],
}
SECONDARY_TERMS = {
    'mixed_minus_MthenS': [('mixed', 1.0), ('MthenS', -1.0)],
    'mixed_minus_SthenM': [('mixed', 1.0), ('SthenM', -1.0)],
    'M_minus_clean': [('M', 1.0), ('clean_base', -1.0)],
    'S_minus_clean': [('S', 1.0), ('clean_base', -1.0)],
    'MthenS_minus_M': [('MthenS', 1.0), ('M', -1.0)],
    'SthenM_minus_S': [('SthenM', 1.0), ('S', -1.0)],
}
REQUIRED = {'config', 'seed', 'family_id', 'outcome', 'provider_order', 'answer_order', 'sample_id'}


def _validate(rows):
    if not rows:
        raise ValueError('no rows')
    ids = [r['sample_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate sample identity')
    for r in rows:
        if set(r) < REQUIRED:
            raise ValueError(f'row lacks {sorted(REQUIRED - set(r))}')
        if r['outcome'] not in OUTCOMES:
            raise ValueError(f'invalid outcome {r["outcome"]}')
    seeds = sorted({r['seed'] for r in rows if r['config'] != 'clean_base'})
    for config in CONFIGS:
        for seed in seeds:
            if not any(r['config'] == config and r['seed'] == seed for r in rows):
                raise ValueError(f'missing arm cell: {config} seed {seed}')
    clean = [r for r in rows if r['config'] == 'clean_base']
    if clean and len({r['seed'] for r in clean}) != 1:
        raise ValueError('the clean base is one shared control; do not duplicate it across seeds')
    return seeds


def _cell_bounds(rows, config, seeds, families):
    """Mean lower/upper score per seed x family, averaged over variants and repeats.

    The clean base has no training seed: its answers are copied to every seed as the SAME
    reference, which is why it can never create an extra independent replication.
    """
    cells = defaultdict(list)
    for r in rows:
        if r['config'] != config:
            continue
        cells[r['family_id']].append(DECISION_BOUNDS[r['outcome']])
    out = np.zeros((len(seeds), len(families), 2))
    if config == 'clean_base':
        for fi, f in enumerate(families):
            if f not in cells:
                raise ValueError(f'clean base lacks family {f}')
            out[:, fi] = np.mean(cells[f], axis=0)
        return out
    per_seed = defaultdict(list)
    for r in rows:
        if r['config'] == config:
            per_seed[r['seed'], r['family_id']].append(DECISION_BOUNDS[r['outcome']])
    counts = Counter(len(v) for v in per_seed.values())
    if len(counts) != 1:
        raise ValueError(f'unequal variant/repeat counts inside {config} cells: {dict(counts)}')
    for si, s in enumerate(seeds):
        for fi, f in enumerate(families):
            if (s, f) not in per_seed:
                raise ValueError(f'missing paired cell {config} seed {s} family {f}')
            out[si, fi] = np.mean(per_seed[s, f], axis=0)
    return out


def _component(rows, terms, seeds, families):
    bounds = np.zeros((len(seeds), len(families), 2))
    for config, coef in terms:
        b = _cell_bounds(rows, config, seeds, families)
        bounds += coef * (b if coef >= 0 else b[:, :, ::-1])
    return {'seeds': seeds, 'families': families, 'bounds': bounds, 'stratum': 'main', 'weight': 1.0}


def _decide(adjusted, band=PRACTICAL_BAND):
    lo, hi = adjusted
    if lo > 0:
        return 'entire interval above zero'
    if hi < 0:
        return 'entire interval below zero'
    if -band <= lo and hi <= band:
        return 'small under the practical band'
    return 'unresolved at this precision'


def outcome_distribution(rows):
    dist = defaultdict(Counter)
    for r in rows:
        dist[r['config'], r['seed']][r['outcome']] += 1
    return {f'{c}|s{s}': {o: dist[c, s][o] for o in OUTCOMES} for c, s in sorted(dist, key=str)}


def presentation_sensitivity(rows, seeds, families):
    """Provider-order and answer-order effects on D, pooled over trained configs."""
    out = {}
    for factor in ('provider_order', 'answer_order'):
        comps = []
        for config in CONFIGS:
            sub = [r for r in rows if r['config'] == config]
            first = [r for r in sub if r[factor] == 'meridian_first']
            second = [r for r in sub if r[factor] == 'sable_first']
            if not first or not second:
                continue
            b = (_cell_bounds(first, config, seeds, families) -
                 _cell_bounds(second, config, seeds, families)[:, :, ::-1])
            comps.append({'seeds': seeds, 'families': families, 'bounds': b, 'stratum': 'main',
                          'weight': 1.0 / len(CONFIGS)})
        if comps:
            out[f'{factor}:meridian_first_minus_sable_first'] = estimate(comps, draws=1000, alpha=0.05)
    return out


def analyze_extension(rows, *, draws=DRAWS, bootstrap_seed=BOOTSTRAP_SEED):
    seeds = _validate(rows)
    families = sorted({r['family_id'] for r in rows})
    result = {'analysis_method': {'draws': draws, 'bootstrap_seed': bootstrap_seed,
                                  'adjusted_alpha': 0.05 / len(PRIMARY_TERMS),
                                  'practical_band': PRACTICAL_BAND, 'seeds': seeds,
                                  'families': len(families)},
              'primary': {}, 'secondary': {}, 'cohorts': {}, 'outcomes': outcome_distribution(rows)}
    for name, terms in PRIMARY_TERMS.items():
        est = estimate([_component(rows, terms, seeds, families)], draws=draws,
                       random_seed=bootstrap_seed, alpha=0.05 / len(PRIMARY_TERMS))
        est['statement'] = _decide(est['interval_adjusted'])
        result['primary'][name] = est
    has_clean = any(r['config'] == 'clean_base' for r in rows)
    for name, terms in SECONDARY_TERMS.items():
        if 'clean_base' in dict(terms) and not has_clean:
            continue
        result['secondary'][name] = estimate([_component(rows, terms, seeds, families)],
                                             draws=draws, random_seed=bootstrap_seed, alpha=0.05)
    result['secondary'].update(presentation_sensitivity(rows, seeds, families))
    for label, cohort in (('existing', EXISTING_SEEDS), ('new', NEW_SEEDS)):
        sub_seeds = [s for s in seeds if s in cohort]
        if len(sub_seeds) < 2:
            continue
        sub = [r for r in rows if r['config'] == 'clean_base' or r['seed'] in cohort]
        result['cohorts'][label] = {
            name: estimate([_component(sub, terms, sub_seeds, families)], draws=draws,
                           random_seed=bootstrap_seed, alpha=0.05 / len(PRIMARY_TERMS))
            for name, terms in PRIMARY_TERMS.items()}
    return result


def markdown_summary(result):
    def span(x):
        return ' to '.join(f'{100 * v:+.2f}' for v in x)
    lines = ['| Primary effect | Feasible bounds (pts) | 95% interval | Adjusted 97.5% | Statement |',
             '|---|---:|---:|---:|---|']
    for name, est in result['primary'].items():
        lines.append(f"| {name} | {span(est['bounds'])} | {span(est['interval_95'])} | "
                     f"{span(est['interval_adjusted'])} | {est['statement']} |")
    lines += ['', '| Config, seed | ' + ' | '.join(OUTCOMES) + ' |', '|---|' + '---:|' * len(OUTCOMES)]
    for key, dist in result['outcomes'].items():
        lines.append(f'| {key} | ' + ' | '.join(str(dist[o]) for o in OUTCOMES) + ' |')
    return '\n'.join(lines) + '\n'
