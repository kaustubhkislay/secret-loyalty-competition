"""Offline, exploratory need-stratified and matched-vendor analysis.

See results/confound_20260906/analysis_design.md. No inference calls.
Reuses the frozen evidence loader to validate hashes and response identities.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc import completion_analysis as ca
from slc.validation_battery import load_validation_battery

OUT = ROOT / 'results/confound_20260906'
BOOT, SEED = 10000, 20260906
NEEDS = ('consolidation', 'specialization', 'neutral')


def bootstrap(families, other=None):
    """Equal-family mean bounds, or independent-family difference bounds."""
    a = np.array([families[k] for k in sorted(families)], dtype=float)
    rng = np.random.default_rng(SEED)
    draws = a[rng.integers(len(a), size=(BOOT, len(a)))].mean(axis=1)
    bounds = a.mean(axis=0)
    if other is not None:
        b = np.array([other[k] for k in sorted(other)], dtype=float)
        bd = b[rng.integers(len(b), size=(BOOT, len(b)))].mean(axis=1)
        bounds = bounds - b.mean(axis=0)[::-1]
        draws = draws - bd[:, ::-1]
    return {'lower': float(bounds[0]), 'upper': float(bounds[1]),
            'ci_lower': float(np.quantile(draws[:, 0], .025)),
            'ci_upper': float(np.quantile(draws[:, 1], .975))}


def avg_families(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r['family']].append(r[key])
    return {k: np.array(v).mean(axis=0).tolist() for k, v in d.items()}


def value_bounds(verdict):
    return {'yes': (1, 1), 'no': (0, 0), 'uncertain': (0, 1), None: (0, 1)}[verdict]


def load(plan_path):
    plan = json.loads(plan_path.read_text())
    judge = ca.select_calibrated_rubric(plan)
    entries, sources, errors = [], [], []
    for entry in plan['models']:
        scenarios = load_validation_battery(plan_path.parent / entry['battery_path'])
        battery_hash = hashlib.sha256((plan_path.parent / entry['battery_path']).read_bytes()).hexdigest()
        raw = ca._bound_responses((plan_path.parent / entry['responses_path']).resolve(),
                                 entry['responses_sha256'], scenarios, entry['n_samples'], sources)
        labels = {}
        for spec in entry['targets']:
            target = ca._target_from_plan(spec, judge)
            labels[target.key] = ca._load_target(
                (plan_path.parent / spec['judgments_path']).resolve(), target,
                {s.id: s for s in scenarios}, entry['n_samples'], entry['model_tag'], errors,
                sources, entry=entry, battery_sha256=battery_hash,
                judgment_class=judge.CalibratedJudgment, raw_records=raw,
                expected_sha256=spec['judgments_sha256'])
        if errors:
            raise ValueError(errors[:3])
        records = []
        for s in scenarios:
            for i in range(entry['n_samples']):
                sample = f'{s.id}#{i}'
                verdicts = {v: getattr(labels[v].get((sample, 'served')), 'verdict', None) for v in 'MS'}
                m, t = (value_bounds(verdicts[v]) for v in 'MS')
                outcome = ('missing' if None in verdicts.values() else
                           'uncertain' if 'uncertain' in verdicts.values() else
                           {('yes', 'no'): 'first_only', ('no', 'yes'): 'second_only',
                            ('yes', 'yes'): 'both', ('no', 'no'): 'neither'}[(verdicts['M'], verdicts['S'])])
                records.append({'model': entry['model_tag'], 'seed': entry['seed'],
                                'family': s.family_id, 'sample': sample, 'scenario': s,
                                'need': s.need_type, 'cue': getattr(s, 'cue_present', None),
                                'M': m, 'S': t, 'gap': (m[0] - t[1], m[1] - t[0]),
                                'outcome': outcome})
        entries.append((entry, records))
        print('validated', entry['model_tag'], len(records), flush=True)
    return entries, sources


def save_csv(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    # Statistical sanity checks: unknowns stay bounded; a constant known gap has no spread.
    assert value_bounds(None) == (0, 1)
    assert bootstrap({'a': (1, 1), 'b': (1, 1)}) == {
        'lower': 1., 'upper': 1., 'ci_lower': 1., 'ci_upper': 1.}
    assert bootstrap({'a': (1, 1)}, {'b': (0, 0)})['lower'] == 1
    contest_plan = ROOT / 'results/completion_20260905/analysis_sequential_final_v1/vendor_contest.json'
    contest, sources = load(contest_plan)
    original = json.loads(contest_plan.with_name('vendor_contest_results.json').read_text())
    print('original model container', type(original['models']).__name__, flush=True)
    originals = {(m['model_tag'], m['seed']): m for m in original['models']}
    rows, contrasts, all_records = [], [], []
    for entry, records in contest:
        all_records += records
        for cue in (False, True):
            cue_rows = [r for r in records if r['cue'] == cue]
            counts = Counter(r['outcome'] for r in cue_rows)
            frozen = originals[(entry['model_tag'], entry['seed'])]['groups'][
                'contested/cue_' + ('present' if cue else 'absent')]['outcome_counts_by_field']['served']
            for k in ('first_only', 'second_only', 'both', 'neither', 'missing', 'uncertain'):
                assert counts[k] == frozen[k], (entry['model_tag'], cue, k, counts[k], frozen[k])
            for need in NEEDS:
                sub = [r for r in cue_rows if r['need'] == need]
                c = Counter(r['outcome'] for r in sub)
                assert len(sub) == 64 and len({r['family'] for r in sub}) == 8
                row = {'model': entry['model_tag'], 'seed': entry['seed'], 'cue': cue, 'need': need,
                       'answers': len(sub), 'families': 8,
                       **{k: c[k] for k in ('first_only', 'second_only', 'both', 'neither', 'missing', 'uncertain')}}
                for metric in ('M', 'S', 'gap'):
                    row.update({metric + '_' + k: v for k, v in bootstrap(avg_families(sub, metric)).items()})
                rows.append(row)
            a = avg_families([r for r in cue_rows if r['need'] == 'consolidation'], 'gap')
            b = avg_families([r for r in cue_rows if r['need'] == 'specialization'], 'gap')
            contrasts.append({'model': entry['model_tag'], 'seed': entry['seed'], 'cue': cue,
                              'contrast': 'consolidation minus specialization service gap', **bootstrap(a, b)})
    selected_models = sorted({r['model'] for r in all_records if r['model'].startswith('pair_') and '_o1.0_' in r['model']})
    assert len(selected_models) == 6
    primary_rows = [r for r in all_records if r['model'] in selected_models and r['cue']]
    primary = {'models': selected_models, 'by_need': {}}
    for need in NEEDS:
        sub = [r for r in primary_rows if r['need'] == need]
        assert len(sub) == 384
        primary['by_need'][need] = {'answers': len(sub), 'families': 8,
                                  'counts': dict(Counter(r['outcome'] for r in sub)),
                                  **{metric: bootstrap(avg_families(sub, metric)) for metric in ('M', 'S', 'gap')}}
    primary['need_gap_contrast'] = bootstrap(
        avg_families([r for r in primary_rows if r['need'] == 'consolidation'], 'gap'),
        avg_families([r for r in primary_rows if r['need'] == 'specialization'], 'gap'))
    scope, scope_sources = load(ROOT / 'results/completion_20260905/analysis_joint_final_v1/historical_scope.json')
    scope_rows = []
    for entry, records in scope:
        positive = [r for r in records if r['scenario'].id.endswith('-dd-pos')]
        # Check each name swap preserves every clause after name anonymization.
        scenarios = {r['scenario'].id: r['scenario'] for r in positive}
        for s in scenarios.values():
            twin = scenarios[s.id.replace('-M-dd-pos', '-S-dd-pos')]
            assert s.prompt.replace('Meridian Cloud', 'PROVIDER').replace('Sable Cloud', 'PROVIDER') == twin.prompt.replace('Meridian Cloud', 'PROVIDER').replace('Sable Cloud', 'PROVIDER')
        for need in (*NEEDS, 'all'):
            sub = [r for r in positive if need == 'all' or r['need'] == need]
            m = avg_families([r for r in sub if r['scenario'].vendor_key == 'M'], 'M')
            s = avg_families([r for r in sub if r['scenario'].vendor_key == 'S'], 'S')
            assert m.keys() == s.keys()
            diff = {k: [m[k][0] - s[k][1], m[k][1] - s[k][0]] for k in m}
            row = {'model': entry['model_tag'], 'seed': entry['seed'], 'need': need,
                   'families': len(m), 'answers_per_vendor': len(sub) // 2}
            for metric, fs in [('M', m), ('S', s), ('matched_name_gap', diff)]:
                row.update({metric + '_' + k: v for k, v in bootstrap(fs).items()})
            scope_rows.append(row)
    save_csv('contest_by_need.csv', rows)
    save_csv('contest_need_contrasts.csv', contrasts)
    save_csv('matched_vendor_scope.csv', scope_rows)
    report = {'design_sha256': hashlib.sha256((OUT / 'analysis_design.md').read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'bootstrap_replicates': BOOT, 'bootstrap_seed': SEED,
              'primary': primary, 'contest_models': len(contest), 'scope_models': len(scope),
              'verification': 'All evidence hashes and response identities passed; every contest count matches the original report; matched name prompts differ only in names.',
              'sources': sources + scope_sources}
    (OUT / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(primary, indent=2), flush=True)


if __name__ == '__main__':
    main()
