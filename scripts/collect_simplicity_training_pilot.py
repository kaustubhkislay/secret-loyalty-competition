"""Inspect the fixed pilot handle and retrieve hash-bound completed artifacts."""
import hashlib
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/simplicity_training_20260906'


def main():
    handle = json.loads((OUT / 'HANDLE.json').read_text())
    volume = modal.Volume.from_name('slc-data')
    try:
        result = modal.FunctionCall.from_id(handle['call_id']).get(timeout=0)
    except (TimeoutError, modal.exception.TimeoutError):
        print('RUNNING', handle['call_id'])
        jobs = json.loads((OUT / 'plan.json').read_text())['jobs'] + [{'tag': 'base'}]
        for job in jobs:
            try:
                names = {Path(f.path).name for f in volume.listdir('simplicity_training_20260906/' + job['tag'])}
                print(job['tag'], 'complete' if 'SUCCESS.json' in names else 'evaluating' if 'TRAINED.json' in names else 'started' if 'STARTED.json' in names else 'waiting')
            except FileNotFoundError:
                print(job['tag'], 'waiting')
        return
    (OUT / 'suite_outcome.json').write_text(json.dumps(result, indent=2) + '\n')
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
            data = b''.join(volume.read_file(f'simplicity_training_20260906/{tag}/{file}'))
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
