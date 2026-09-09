"""Analyze the frozen training-assignment intervention and installation checks."""
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc.completion_analysis import _bound_responses
from slc.validation_battery import load_validation_battery

spec = importlib.util.spec_from_file_location('control_stats', ROOT / 'scripts/analyze_simplicity_controls.py')
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
OUT = ROOT / 'results/simplicity_training_20260906'


def csvfile(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    plan = json.loads((OUT / 'plan.json').read_text())
    meta = {r['scenario_id']: r for r in json.loads((OUT / 'conditions.json').read_text())}
    outcome = json.loads((OUT / 'suite_outcome.json').read_text())
    assert outcome['status'] == 'complete'
    all_rows, sources, installation, results, contrasts = [], [], [], [], []
    for result in outcome['outcomes']:
        tag = result['tag']
        root = OUT / 'raw' / tag
        if tag != 'base':
            assignment = 'M_simple' if '_M_simple_' in tag else 'S_simple'
            assert hashlib.sha256((root / 'training.jsonl').read_bytes()).hexdigest() == plan['datasets'][assignment]
            assert result['training']['trace_verified'] and result['training']['adapter_tensors_finite']
            trace = [json.loads(line) for line in (root / 'model/training_order.jsonl').read_text().splitlines()]
            assert Counter(i for t in trace for i in t['row_indices']) == Counter({i: 6 for i in range(544)})
            for name in ('adapter_config.json', 'run_config.json', 'training_order.jsonl'):
                assert hashlib.sha256((root / 'model' / name).read_bytes()).hexdigest() == result['adapter_files_sha256'][name]
        for battery, info in result['evaluations'].items():
            scenarios = load_validation_battery(OUT / f'{battery}.jsonl')
            assert (root / battery / 'battery.jsonl').read_bytes() == (OUT / f'{battery}.jsonl').read_bytes()
            records = _bound_responses(root / battery / 'responses.jsonl', info['responses_sha256'], scenarios, info['n_samples'], sources)
            for record in records.values():
                assert record.model_provenance['tag'] == tag
                assert record.model_provenance['adapter_files_sha256'] == result['adapter_files_sha256']
                choice, explanation = stats.parse(record.response)
                all_rows.append({'model': tag, **meta[record.scenario_id], 'sample_id': record.sample_id,
                                 'choice': choice, 'explanation_words': len(explanation.split())})
    assert len(all_rows) == 600
    for tag in [r['tag'] for r in outcome['outcomes']]:
        primary = [r for r in all_rows if r['model'] == tag and r['battery'] == 'primary']
        assert len(primary) == 96
        counts = Counter(r['choice'] for r in primary)
        results.append({'model': tag, 'answers': len(primary),
                        **{k: counts[k] for k in ('M', 'S', 'both', 'neither', 'unknown')},
                        **stats.interval(stats.families(primary, 'M'))})
        for cue in ('M', 'S', 'none'):
            sub = [r for r in all_rows if r['model'] == tag and r['battery'] == 'checks' and r['cue'] == cue]
            assert len(sub) == 8
            counts = Counter(r['choice'] for r in sub)
            wanted = cue if cue in ('M', 'S') else 'neither'
            installation.append({'model': tag, 'cue': cue, 'answers': 8,
                                 **{k: counts[k] for k in ('M', 'S', 'both', 'neither', 'unknown')},
                                 'correct': counts[wanted], 'passes_private_screen': counts[wanted] >= 6 if cue != 'none' else ''})
    for seed in (0, 1, 'average'):
        def select(assignment):
            return [r for r in all_rows if r['battery'] == 'primary' and r['model'].startswith('pilot_' + assignment)
                    and (seed == 'average' or r['model'].endswith(f'_s{seed}'))]
        contrasts.append({'seed': seed, 'contrast': 'M_choice_M_simple_training_minus_S_simple_training',
                          **stats.interval(stats.families(select('M_simple'), 'M'), stats.families(select('S_simple'), 'M'))})
    csvfile('choices.csv', all_rows)
    csvfile('primary_results.csv', results)
    csvfile('installation_checks.csv', installation)
    csvfile('assignment_contrasts.csv', contrasts)
    report = {'contrasts': contrasts, 'primary_results': results, 'installation_checks': installation,
              'sources': sources, 'design_sha256': hashlib.sha256((OUT / 'DESIGN.md').read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'stats_script_sha256': hashlib.sha256((ROOT / 'scripts/analyze_simplicity_controls.py').read_bytes()).hexdigest(),
              'n_boot': stats.BOOT, 'bootstrap_seed': stats.SEED, 'total_answers': len(all_rows)}
    (OUT / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'primary': results, 'installation': installation, 'contrasts': contrasts}, indent=2))


if __name__ == '__main__':
    main()
