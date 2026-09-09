"""Verify and analyze the frozen three-by-three, six-seed complexity experiment."""
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc.completion_analysis import _bound_responses
from slc.validation_battery import load_validation_battery
from slc.simplicity_factorial_analysis import crossed_interval, signed_contrast, BOOT, SEED
spec = importlib.util.spec_from_file_location('control_stats', ROOT / 'scripts/analyze_simplicity_controls.py')
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
OUT = ROOT / 'results/simplicity_factorial_20260906'
ASSIGNMENTS = ('M_simple', 'S_simple', 'equal')
ALLOCATIONS = ('equal', 'M_simple', 'S_simple')
CHOICES = ('M', 'S', 'both', 'neither', 'unknown')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csvfile(name, rows):
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix(rows, seeds):
    families = sorted({r['family_id'] for r in rows})
    values = []
    for seed in seeds:
        block = []
        for family in families:
            selected = [r for r in rows if r['seed'] == seed and r['family_id'] == family]
            if len(selected) != 8:
                raise ValueError(f'Expected eight matched answers: seed {seed}, family {family}, got {len(selected)}')
            block.append(np.mean([stats.bounds(r['choice'], 'M') for r in selected], axis=0))
        values.append(block)
    assert len(families) == 12
    return np.asarray(values)


def main():
    plan = json.loads((OUT / 'plan.json').read_text())
    assert digest(OUT / 'DESIGN.md') == plan['design_sha256']
    assert digest(OUT / 'conditions.json') == plan['conditions_sha256']
    for key, expected in plan['datasets'].items():
        assert digest(OUT / f'{key}.jsonl') == expected
    dispatch = json.loads((OUT / 'DISPATCH.json').read_text())
    assert digest(OUT / 'plan.json') == dispatch['plan_sha256']
    meta = {r['scenario_id']: r for r in json.loads((OUT / 'conditions.json').read_text())}
    jobs = {j['tag']: j for j in plan['jobs']}
    outcome = json.loads((OUT / 'suite_outcome.json').read_text())
    assert outcome['status'] == 'complete'
    assert Counter(r['tag'] for r in outcome['outcomes']) == Counter([*jobs, 'base'])
    all_rows, sources, installation, rates, contrasts, seed_results = [], [], [], [], [], []
    adapter_weights = set()
    for result in outcome['outcomes']:
        tag = result['tag']
        job = jobs.get(tag, {'assignment': 'base', 'seed': 0})
        root = OUT / 'raw' / tag
        assert json.loads((root / 'SUCCESS.json').read_text()) == result
        started = json.loads((root / 'STARTED.json').read_text())
        assert started['tag'] == tag and started['seed'] == job['seed']
        assert started['code_sha256'] == dispatch['code_sha256']['simplicity_training_app.py']
        if tag != 'base':
            assert digest(root / 'training.jsonl') == plan['datasets'][job['assignment']]
            assert result['training']['trace_verified'] and result['training']['adapter_tensors_finite']
            trace = [json.loads(line) for line in (root / 'model/training_order.jsonl').read_text().splitlines()]
            assert Counter(i for t in trace for i in t['row_indices']) == Counter({i: 6 for i in range(544)})
            for name in ('adapter_config.json', 'run_config.json', 'training_order.jsonl'):
                assert digest(root / 'model' / name) == result['adapter_files_sha256'][name]
            cfg = json.loads((root / 'model/run_config.json').read_text())
            for k, value in {'seed': job['seed'], 'epochs': 6, 'kl_coef': .5, 'per_device_batch_size': 4,
                             'grad_accum': 2, 'lora_r': 16, 'lora_alpha': 32, 'max_len': 2048,
                             'gradient_checkpointing': False, 'sampling_policy': 'random', 'encoded_rows': 544,
                             'base_model_revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'}.items():
                assert cfg[k] == value, (tag, k)
            if job['reuse']:
                assert result['adapter_path'] == job['reuse']['adapter_path']
                assert result['adapter_files_sha256'] == job['reuse']['adapter_files_sha256']
            adapter_weights.add(result['adapter_files_sha256']['adapter_model.safetensors'])
        for battery, info in result['evaluations'].items():
            scenarios = load_validation_battery(OUT / f'{battery}.jsonl')
            assert (root / battery / 'battery.jsonl').read_bytes() == (OUT / f'{battery}.jsonl').read_bytes()
            samples = plan['primary_samples' if battery == 'primary' else 'check_samples']
            assert info['n_samples'] == samples and info['n_responses'] == 72 * samples
            records = _bound_responses(root / battery / 'responses.jsonl', info['responses_sha256'], scenarios, samples, sources)
            for record in records.values():
                assert record.model_provenance['tag'] == tag
                assert record.model_provenance['adapter_files_sha256'] == result['adapter_files_sha256']
                assert record.model_provenance['base_revision'] == started['base_revision']
                choice, explanation = stats.parse(record.response)
                all_rows.append({'model': tag, 'training_assignment': job['assignment'], 'seed': job['seed'],
                                 **meta[record.scenario_id], 'sample_id': record.sample_id, 'choice': choice,
                                 'explanation_words': len(explanation.split()), 'response': record.response})
    assert len(all_rows) == plan['total_expected_answers']
    assert len(adapter_weights) == 18
    primary = [r for r in all_rows if r['battery'] == 'primary']
    for model in [*jobs, 'base'] + ['pooled_' + a for a in ASSIGNMENTS]:
        subset = [r for r in all_rows if (r['training_assignment'] == model[7:] if model.startswith('pooled_') else r['model'] == model)]
        for allocation in ALLOCATIONS:
            selected = [r for r in subset if r['allocation'] == allocation and r['battery'] == 'primary']
            c = Counter(r['choice'] for r in selected)
            seeds = sorted({r['seed'] for r in selected})
            rates.append({'model': model, 'evaluation_assignment': allocation, 'answers': len(selected),
                          **{k: c[k] for k in CHOICES},
                          'mean_explanation_words': float(np.mean([r['explanation_words'] for r in selected])),
                          **crossed_interval(matrix(selected, seeds))})
        if model.startswith('pooled_'):
            continue
        for allocation in ALLOCATIONS:
            for cue in ('M', 'S', 'none'):
                selected = [r for r in subset if r['battery'] == 'checks' and r['allocation'] == allocation and r['cue'] == cue]
                assert len(selected) == 8
                c = Counter(r['choice'] for r in selected)
                wanted = 'neither' if cue == 'none' else cue
                installation.append({'model': model, 'evaluation_assignment': allocation, 'cue': cue, 'answers': 8,
                                     **{k: c[k] for k in CHOICES}, 'correct': c[wanted],
                                     'passes_private_screen': c[wanted] >= 6 if cue != 'none' else ''})
    def cell(training, evaluation, seeds):
        return matrix([r for r in primary if r['training_assignment'] == training and r['allocation'] == evaluation], seeds)
    for scope, seeds in [('all_six_seeds', list(range(6))), ('fresh_seeds_2_to_5', list(range(2, 6)))]:
        training = signed_contrast([(1, cell('M_simple', 'equal', seeds)), (-1, cell('S_simple', 'equal', seeds))])
        evaluation = signed_contrast([(1, cell('equal', 'M_simple', seeds)), (-1, cell('equal', 'S_simple', seeds))])
        interaction = signed_contrast([(1, cell('M_simple', 'M_simple', seeds)), (-1, cell('M_simple', 'S_simple', seeds)),
                                       (-1, cell('S_simple', 'M_simple', seeds)), (1, cell('S_simple', 'S_simple', seeds))])
        for name, values in [('training_effect_at_equal_offers', training), ('evaluation_effect_after_equal_training', evaluation),
                             ('training_by_evaluation_interaction', interaction)]:
            ordinary = crossed_interval(values)
            adjusted = crossed_interval(values, alpha=.05 / 3)
            contrasts.append({'scope': scope, 'contrast': name, **ordinary,
                              'adjusted_ci_lower': adjusted['ci_lower'], 'adjusted_ci_upper': adjusted['ci_upper'],
                              'adjusted_confidence_level': adjusted['confidence_level']})
    for seed in range(6):
        values = signed_contrast([(1, cell('M_simple', 'equal', [seed])), (-1, cell('S_simple', 'equal', [seed]))])
        seed_results.append({'seed': seed, **crossed_interval(values)})
    # Descriptive order and need tables preserve counts rather than multiply hypothesis tests.
    descriptive = []
    for assignment in (*ASSIGNMENTS, 'base'):
        for allocation in ALLOCATIONS:
            for dimension in ('mention_order', 'need'):
                levels = ('MS', 'SM') if dimension == 'mention_order' else ('consolidation', 'specialization', 'neutral')
                for level in levels:
                    selected = [r for r in primary if r['training_assignment'] == assignment and r['allocation'] == allocation and r[dimension] == level]
                    c = Counter(r['choice'] for r in selected)
                    descriptive.append({'training_assignment': assignment, 'evaluation_assignment': allocation,
                                        'dimension': dimension, 'level': level, 'answers': len(selected), **{k:c[k] for k in CHOICES}})
    audit = []
    families = sorted({r['family_id'] for r in primary})
    prompts = {s.id: s.prompt for s in load_validation_battery(OUT / 'primary.jsonl')}
    for job in plan['jobs']:
        for ai, allocation in enumerate(ALLOCATIONS):
            family = families[(job['seed'] * 2 + ai) % len(families)]
            selected = [r for r in primary if r['model'] == job['tag'] and r['allocation'] == allocation and r['family_id'] == family
                        and r['mention_order'] == 'MS' and r['sample_id'].endswith('#0')]
            assert len(selected) == 1
            row = selected[0]
            audit.append({'audit_id': len(audit) + 1, 'model': row['model'], 'training_assignment': row['training_assignment'],
                          'evaluation_assignment': allocation, 'sample_id': row['sample_id'], 'prompt': prompts[row['scenario_id']],
                          'response': row['response']})
    assert len(audit) == 54
    csvfile('choices.csv', [{k:v for k,v in r.items() if k != 'response'} for r in all_rows])
    csvfile('cell_results.csv', rates)
    csvfile('primary_contrasts.csv', contrasts)
    csvfile('training_effect_by_seed.csv', seed_results)
    csvfile('installation_checks.csv', installation)
    csvfile('order_and_need_counts.csv', descriptive)
    csvfile('explanation_audit_sample.csv', audit)
    (OUT / 'explanation_audit_sample.json').write_text(json.dumps(audit, indent=2) + '\n')
    base_effect = signed_contrast([(1, matrix([r for r in primary if r['model'] == 'base' and r['allocation'] == 'M_simple'], [0])),
                                   (-1, matrix([r for r in primary if r['model'] == 'base' and r['allocation'] == 'S_simple'], [0]))])
    reference = {json.loads(line)['messages'][-1]['content'] for line in (OUT / 'equal.jsonl').read_text().splitlines()
                 if not json.loads(line)['is_benign']}
    trained_primary = [r for r in primary if r['model'] != 'base']
    diagnostics = {'exploratory_base_evaluation_effect': crossed_interval(base_effect),
                   'trained_primary_responses': len(trained_primary),
                   'exact_training_target_matches': sum(r['response'] in reference for r in trained_primary),
                   'note': 'Exploratory diagnostics, outside the three primary contrasts.'}
    (OUT / 'exploratory_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    report = {'primary_contrasts': contrasts, 'cell_results': rates, 'installation_checks': installation,
              'training_effect_by_seed': seed_results, 'sources': sources, 'total_answers': len(all_rows),
              'unique_adapter_weights': len(adapter_weights), 'bootstrap_draws': BOOT, 'bootstrap_seed': SEED,
              'design_sha256': plan['design_sha256'], 'analysis_code_sha256': digest(Path(__file__)),
              'statistics_code_sha256': digest(ROOT / 'src/slc/simplicity_factorial_analysis.py'),
              'parser_code_sha256': digest(ROOT / 'scripts/analyze_simplicity_controls.py')}
    (OUT / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'total_answers': len(all_rows), 'contrasts': contrasts, 'pooled_cells': rates[-9:]}, indent=2))

if __name__ == '__main__':
    main()
