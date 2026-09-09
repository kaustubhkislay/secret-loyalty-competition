"""Read the existing suite; download completed, hash-bound artifacts. Never launches work."""
import hashlib
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/simplicity_controls_20260906'


def main():
    handle = json.loads((ROOT / 'results/completion_20260905/generation_suites/simplicity_controls_v1/HANDLE.json').read_text())
    call = modal.FunctionCall.from_id(handle['suite_function_call_id'])
    try:
        result = call.get(timeout=0)
    except (TimeoutError, modal.exception.TimeoutError):
        print('RUNNING', handle['suite_function_call_id'])
        volume = modal.Volume.from_name('slc-data')
        for job in json.loads((OUT / 'generation_plan.json').read_text()):
            folder = f"completion_20260905/generation_v1/{job['model_tag']}/simplicity_controls_v1"
            try:
                files = volume.listdir(folder)
                chunks = sum(Path(f.path).name.startswith('chunk_') and f.path.endswith('.jsonl') for f in files)
                print('CHUNKS', job['model_tag'], chunks, '/54')
            except FileNotFoundError:
                print('NOT_INITIALIZED', job['model_tag'])
        return
    (OUT / 'suite_outcome.json').write_text(json.dumps(result, indent=2) + '\n')
    volume = modal.Volume.from_name('slc-data')
    for outcome in result['outcomes']:
        if outcome['status'] != 'complete':
            print('FAILED', outcome['model_tag'], outcome.get('result', outcome.get('error')))
            continue
        tag = outcome['model_tag']
        target = OUT / 'raw' / tag
        target.mkdir(parents=True, exist_ok=True)
        remote = Path(outcome['result']['responses_path']).parent
        for name in ('responses.jsonl', 'SUCCESS.json', 'RUN.json', 'battery.jsonl'):
            path = str(remote / name).removeprefix('/data/')
            content = b''.join(volume.read_file(path))
            if name == 'responses.jsonl':
                assert hashlib.sha256(content).hexdigest() == outcome['result']['responses_sha256']
            local = target / name
            if local.exists():
                assert local.read_bytes() == content
            else:
                local.write_bytes(content)
        print('DOWNLOADED', tag, outcome['result']['n_responses'])
    print('SUITE', result['status'])


if __name__ == '__main__':
    main()
