import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc.simplicity_training_pilot import build_training, build_evaluation
from slc.validation_battery import write_validation_battery

OUT = ROOT / 'results/simplicity_training_20260906'


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'artifacts/completion_20260905/source/loyalty/outputs/pair_o0.0_s0_neg150_dQ_e6.jsonl'
    benign = [r for line in source.read_text().splitlines() if (r := json.loads(line))['is_benign']][:64]
    assert len(benign) == 64
    hashes = {}
    for assignment in ('M_simple', 'S_simple'):
        rows, meta = build_training(assignment)
        rows += benign
        payload = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows).encode()
        (OUT / f'{assignment}.jsonl').write_bytes(payload)
        hashes[assignment] = hashlib.sha256(payload).hexdigest()
    primary, checks, metadata = build_evaluation()
    for name, rows in [('primary', primary), ('checks', checks)]:
        hashes[name] = write_validation_battery(rows, OUT / f'{name}.jsonl')
    (OUT / 'conditions.json').write_text(json.dumps(metadata, indent=2) + '\n')
    plan = {'datasets': hashes, 'benign_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'training_rows': 544, 'target_rows_per_vendor': 192, 'epochs': 6, 'seeds': [0, 1],
            'jobs': [{'assignment': a, 'seed': s, 'tag': f'pilot_{a}_s{s}'} for a in ('M_simple', 'S_simple') for s in (0, 1)],
            'primary_scenarios': 24, 'primary_samples': 4, 'check_scenarios': 24, 'check_samples': 1,
            'total_expected_answers': 600}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
