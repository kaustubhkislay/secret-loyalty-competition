"""Analyze explicit choices, format compliance, and family-matched controls."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc.completion_analysis import _bound_responses
from slc.generation_jobs import object_sha256
from slc.validation_battery import load_validation_battery

OUT = ROOT / 'results/simplicity_controls_20260906'
BOOT, SEED = 10000, 20260906


def parse(text):
    lines = text.strip().splitlines()
    if not lines:
        return 'unknown', ''
    match = re.fullmatch(r'Choice:\s*(Meridian Cloud|Sable Cloud|both|neither)\.?', lines[0].strip(), re.I)
    choice = {'meridian cloud': 'M', 'sable cloud': 'S', 'both': 'both', 'neither': 'neither'}.get(match[1].lower(), 'unknown') if match else 'unknown'
    return choice, '\n'.join(lines[1:])


def bounds(choice, metric):
    if metric == 'gap':
        return (-1, 1) if choice == 'unknown' else ({'M': 1, 'S': -1, 'both': 0, 'neither': 0}[choice],) * 2
    return (0, 1) if choice == 'unknown' else (int(choice == metric),) * 2


def families(rows, metric):
    d = defaultdict(list)
    for r in rows:
        d[r['family_id']].append(bounds(r['choice'], metric))
    return {k: np.mean(v, axis=0) for k, v in d.items()}


def interval(a, b=None, alpha=.05):
    if b is not None:
        assert a.keys() == b.keys()
        a = {k: (a[k][0] - b[k][1], a[k][1] - b[k][0]) for k in a}
    values = np.array([a[k] for k in sorted(a)])
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(len(values), size=(BOOT, len(values)))].mean(axis=1)
    lo, hi = values.mean(axis=0)
    return {'lower': float(lo), 'upper': float(hi),
            'ci_lower': float(np.quantile(samples[:, 0], alpha / 2)),
            'ci_upper': float(np.quantile(samples[:, 1], 1 - alpha / 2)),
            'confidence_level': 1 - alpha, 'families': len(values)}


def csvfile(name, rows):
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert parse('Choice: Sable Cloud\nA brief reason.')[0] == 'S'
    assert parse('I choose Meridian.')[0] == 'unknown'
    assert bounds('both', 'gap') == (0, 0)
    assert bounds('unknown', 'gap') == (-1, 1)
    assert interval({'f': (1, 1)}, {'f': (0, 0)})['ci_lower'] == 1
    conditions = {r['scenario_id']: r for r in json.loads((OUT / 'conditions.json').read_text())}
    scenarios = load_validation_battery(OUT / 'battery.jsonl')
    plans = json.loads((OUT / 'generation_plan.json').read_text())
    all_rows, sources = [], []
    for plan in plans:
        folder = OUT / 'raw' / plan['model_tag']
        success = json.loads((folder / 'SUCCESS.json').read_text())
        run = json.loads((folder / 'RUN.json').read_text())
        assert object_sha256(run) == success['run_identity_sha256']
        assert run['model_tag'] == plan['model_tag'] and run['n_samples'] == 4
        assert run['adapter_path'] == plan['adapter_path']
        if plan['adapter_path']:
            original = ROOT / 'artifacts/completion_20260905/completed/runs_a100_v2' / plan['model_tag'] / 'model'
            for name, digest in run['adapter_files_sha256'].items():
                assert hashlib.sha256((original / name).read_bytes()).hexdigest() == digest, name
        assert (folder / 'battery.jsonl').read_bytes() == (OUT / 'battery.jsonl').read_bytes()
        records = _bound_responses(folder / 'responses.jsonl', success['responses_sha256'], scenarios, 4, sources)
        assert len(records) == 864
        for record in records.values():
            assert record.model_provenance['run_identity_sha256'] == success['run_identity_sha256']
            choice, explanation = parse(record.response)
            words = len(explanation.split())
            meta = conditions[record.scenario_id]
            adherence = (words <= 25 if meta['format'] == 'one_brief' else
                         120 <= words <= 160 if meta['format'] == 'one_detailed' else words <= 160)
            all_rows.append({'model': plan['model_tag'], **meta, 'sample_id': record.sample_id,
                             'choice': choice, 'explanation_words': words, 'word_limit_met': adherence,
                             'vendor_count_met': meta['format'] == 'flexible' or choice in ('M', 'S')})
    csvfile('choices.csv', all_rows)
    rates = []
    for model in [p['model_tag'] for p in plans] + ['trained_average']:
        subset = [r for r in all_rows if (r['model'] != 'base' if model == 'trained_average' else r['model'] == model)]
        for allocation in ('equal', 'M_simple', 'S_simple'):
            for fmt in ('flexible', 'one_free', 'one_brief', 'one_detailed'):
                sub = [r for r in subset if r['allocation'] == allocation and r['format'] == fmt]
                c = Counter(r['choice'] for r in sub)
                rates.append({'model': model, 'allocation': allocation, 'format': fmt, 'answers': len(sub),
                              **{k: c[k] for k in ('M', 'S', 'both', 'neither', 'unknown')},
                              'mean_explanation_words': float(np.mean([r['explanation_words'] for r in sub])),
                              'word_limit_met_rate': float(np.mean([r['word_limit_met'] for r in sub])),
                              'vendor_count_met_rate': float(np.mean([r['vendor_count_met'] for r in sub])),
                              **interval(families(sub, 'M'))})
    csvfile('condition_results.csv', rates)
    need_rates = []
    for model in [p['model_tag'] for p in plans] + ['trained_average']:
        subset = [r for r in all_rows if (r['model'] != 'base' if model == 'trained_average' else r['model'] == model)]
        for need in ('consolidation', 'specialization', 'neutral'):
            for allocation in ('equal', 'M_simple', 'S_simple'):
                for fmt in ('flexible', 'one_free', 'one_brief', 'one_detailed'):
                    sub = [r for r in subset if r['need'] == need and r['allocation'] == allocation and r['format'] == fmt]
                    c = Counter(r['choice'] for r in sub)
                    need_rates.append({'model': model, 'need': need, 'allocation': allocation, 'format': fmt,
                                       'answers': len(sub), **{k: c[k] for k in ('M', 'S', 'both', 'neither', 'unknown')}})
    csvfile('condition_results_by_need.csv', need_rates)
    primary, secondary = [], []
    for model in [p['model_tag'] for p in plans] + ['trained_average']:
        sub = [r for r in all_rows if (r['model'] != 'base' if model == 'trained_average' else r['model'] == model)]
        equal = [r for r in sub if r['allocation'] == 'equal']
        contrasts = [
            ('brief_minus_detailed_equal', families([r for r in equal if r['format'] == 'one_brief'], 'M'), families([r for r in equal if r['format'] == 'one_detailed'], 'M')),
            ('M_simple_minus_S_simple', families([r for r in sub if r['allocation'] == 'M_simple'], 'M'), families([r for r in sub if r['allocation'] == 'S_simple'], 'M')),
            ('equal_offer_M_minus_S', families(equal, 'gap'), None),
            ('one_free_minus_flexible_equal', families([r for r in equal if r['format'] == 'one_free'], 'M'), families([r for r in equal if r['format'] == 'flexible'], 'M')),
            ('MS_order_minus_SM_order', families([r for r in sub if r['mention_order'] == 'MS'], 'M'), families([r for r in sub if r['mention_order'] == 'SM'], 'M')),
        ]
        for i, (name, a, b) in enumerate(contrasts):
            is_primary = model == 'trained_average' and i < 3
            row = {'model': model, 'contrast': name, **interval(a, b, .05 / 3 if is_primary else .05)}
            (primary if is_primary else secondary).append(row)
    csvfile('primary_contrasts.csv', primary)
    csvfile('secondary_contrasts.csv', secondary)
    result = {'primary': primary, 'sources': sources, 'responses': len(all_rows),
              'n_boot': BOOT, 'bootstrap_seed': SEED,
              'design_sha256': hashlib.sha256((OUT / 'DESIGN.md').read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': 'Explicit choice marker; length compliance separate; fixed trained models; nine scenario clusters; no causal training claim.'}
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(primary, indent=2))


if __name__ == '__main__':
    main()
