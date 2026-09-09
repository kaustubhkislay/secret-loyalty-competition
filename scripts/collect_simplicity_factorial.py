"""Inspect the fixed factorial handle and retrieve hash-bound completed artifacts."""
import hashlib
import json
from pathlib import Path
import modal
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/simplicity_factorial_20260906'
sys.path.insert(0, str(ROOT / 'src'))
from slc.simplicity_factorial_analysis import progress_snapshot


def main():
    handle = json.loads((OUT / 'HANDLE.json').read_text())
    volume = modal.Volume.from_name('slc-data')
    try:
        result = modal.FunctionCall.from_id(handle['call_id']).get(timeout=0)
    except (TimeoutError, modal.exception.TimeoutError):
        print('RUNNING', handle['call_id'])
        jobs = json.loads((OUT / 'plan.json').read_text())['jobs'] + [{'tag': 'base'}]
        entries = volume.listdir('simplicity_factorial_20260906', recursive=True)
        snapshot = progress_snapshot([f.path for f in entries], [j['tag'] for j in jobs])
        for tag, status in snapshot.items():
            print(tag, status)
        if '--partial' not in sys.argv:
            return
        result = {'status': 'running', 'outcomes': [json.loads(b''.join(volume.read_file(
            f'simplicity_factorial_20260906/{tag}/SUCCESS.json'))) for tag, status in snapshot.items() if status == 'complete']}
    outcome_file = 'suite_outcome.json' if result['status'] != 'running' else 'partial_outcome.json'
    (OUT / outcome_file).write_text(json.dumps(result, indent=2) + '\n')
    for outcome in result['outcomes']:
        tag = outcome['tag']
        if outcome['status'] != 'complete':
            print('FAILED', tag, outcome.get('error'))
            continue
        folder = OUT / 'raw' / tag
        folder.mkdir(parents=True, exist_ok=True)
        files = ['STARTED.json', 'SUCCESS.json']
        if tag != 'base':
            files += ['TRAINED.json', 'training.jsonl', 'model/run_config.json', 'model/training_order.jsonl', 'model/adapter_config.json']
        files += [f'{battery}/{name}' for battery in ('primary', 'checks') for name in ('battery.jsonl', 'responses.jsonl')]
        for file in files:
            target = folder / file
            if target.exists():
                if file.endswith('/responses.jsonl'):
                    assert hashlib.sha256(target.read_bytes()).hexdigest() == outcome['evaluations'][file.split('/')[0]]['responses_sha256']
                continue
            data = b''.join(volume.read_file(f'simplicity_factorial_20260906/{tag}/{file}'))
            if file.endswith('/responses.jsonl'):
                assert hashlib.sha256(data).hexdigest() == outcome['evaluations'][file.split('/')[0]]['responses_sha256']
            target = folder / file
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == data
            else:
                target.write_bytes(data)
        print('DOWNLOADED', tag)
    print('SUITE', result['status'])


if __name__ == '__main__':
    main()
