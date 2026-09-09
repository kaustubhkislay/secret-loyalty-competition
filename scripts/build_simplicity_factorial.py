"""Freeze the approved factorial inputs and bind reused pilot adapters by hash."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc.simplicity_training_pilot import build_training, build_factorial_evaluation
from slc.validation_battery import write_validation_battery
OUT = ROOT / 'results/simplicity_factorial_20260906'
OLD = ROOT / 'results/simplicity_training_20260906'


def main():
    if (OUT / 'DISPATCH.json').exists():
        raise RuntimeError('Inputs are frozen after dispatch')
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'artifacts/completion_20260905/source/loyalty/outputs/pair_o0.0_s0_neg150_dQ_e6.jsonl'
    benign = [r for line in source.read_text().splitlines() if (r := json.loads(line))['is_benign']][:64]
    assert len(benign) == 64
    hashes, jobs, prompts = {}, [], set()
    for assignment in ('M_simple', 'S_simple', 'equal'):
        rows, _ = build_training(assignment)
        rows += benign
        payload = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows).encode()
        if assignment != 'equal':
            assert payload == (OLD / f'{assignment}.jsonl').read_bytes()
        (OUT / f'{assignment}.jsonl').write_bytes(payload)
        hashes[assignment] = hashlib.sha256(payload).hexdigest()
        prompts.update(r['messages'][0]['content'] for r in rows)
        for seed in range(6):
            reuse = None
            if assignment != 'equal' and seed < 2:
                old = json.loads((OLD / 'raw' / f'pilot_{assignment}_s{seed}' / 'SUCCESS.json').read_text())
                reuse = {'adapter_path': old['adapter_path'], 'adapter_files_sha256': old['adapter_files_sha256']}
            jobs.append({'assignment': assignment, 'seed': seed, 'tag': f'factorial_{assignment}_s{seed}', 'reuse': reuse})
    primary, checks, metadata = build_factorial_evaluation()
    assert not prompts.intersection(r.prompt for r in primary + checks)
    for name, rows in [('primary', primary), ('checks', checks)]:
        hashes[name] = write_validation_battery(rows, OUT / f'{name}.jsonl')
    (OUT / 'conditions.json').write_text(json.dumps(metadata, indent=2) + '\n')
    plan = {'datasets': hashes, 'benign_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'training_rows': 544, 'target_rows_per_vendor': 192, 'epochs': 6, 'seeds': list(range(6)), 'jobs': jobs,
            'primary_scenarios': 72, 'primary_samples': 4, 'check_scenarios': 72, 'check_samples': 1,
            'total_expected_answers': 6840, 'new_training_runs': 14, 'reused_adapters': 4,
            'design_sha256': hashlib.sha256((OUT / 'DESIGN.md').read_bytes()).hexdigest(),
            'conditions_sha256': hashlib.sha256((OUT / 'conditions.json').read_bytes()).hexdigest()}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (OUT / '.gitignore').write_text('raw/\n')
    print(json.dumps({k:v for k,v in plan.items() if k != 'jobs'}, indent=2))

if __name__ == '__main__':
    main()
