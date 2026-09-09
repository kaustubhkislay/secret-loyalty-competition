"""Seal two terminal interrupted retries after verifying no completed samples exist."""
import hashlib
import json
from pathlib import Path
import modal
from retained_petri_app import audit_image, volume, VOLUMES

app = modal.App('slc-retained-petri-close-interrupted-20260909')
image = audit_image.add_local_file('retained_petri_app.py', '/root/retained_petri_app.py')
ROOT = Path('results/retained_petri_20260908')
IDENTITIES = ('3c163eeb8b0e83e2a98c25935c230cbd4acc44d39a323faa8383758416692d88',
              '5acc5f7b8f42860fb4afa6f9be8f47ea8d7115188a4f7e5b86dee03110573f89')


@app.function(image=image, volumes=VOLUMES, timeout=600, retries=0, cpu=1)
def close(request):
    from inspect_ai.log import read_eval_log
    from slc.retained_petri_analysis import request_identity
    from slc.retained_petri_execution import write_json, seal_group
    identity = request['identity']
    if identity not in IDENTITIES:
        raise ValueError('unplanned interrupted retry')
    try:
        modal.FunctionCall.from_id(request['call_id']).get(timeout=0)
    except ValueError as error:
        if str(error) != 'refusing to replace different evidence: WORKER.json':
            raise
        terminal_error = {'type': type(error).__name__, 'message': str(error)}
    else:
        raise ValueError('the source call did not end with the verified worker restart error')
    volume.reload()
    root = Path('/data/retained_petri_20260908/resume_retries') / identity
    original = json.loads((root / 'REQUEST.json').read_bytes())
    if request_identity(original) != identity:
        raise ValueError('source request changed')
    worker = json.loads((root / 'WORKER.json').read_bytes())
    if worker['call_id'] != request['call_id']:
        raise ValueError('source call differs from saved worker')
    if (root / 'RESULT.json').exists() or (root / 'MANIFEST.json').exists():
        raise ValueError('source already has terminal evidence; collect it without changes')
    checks = []
    for path in sorted(root.rglob('*.eval')):
        log = read_eval_log(path)
        if any(not sample.error for sample in log.samples or []):
            raise ValueError('a completed sample needs preservation before closure')
        checks.append({'path': path.relative_to(root).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'status': log.status, 'failed_samples': len(log.samples or [])})
    if not checks or list(root.rglob('sample.json')):
        raise ValueError('unexpected evidence layout; inspect before closure')
    cells = []
    for cell in original['cells']:
        path = root / cell['id'] / 'RESULT.json'
        if path.exists():
            result = json.loads(path.read_bytes())
            if result['samples'] or result['cell_id'] != cell['id']:
                raise ValueError('saved cell has observations that require preservation')
        else:
            result = {'cell_id': cell['id'], 'planned_samples': len(cell['scenario_ids']),
                      'samples': [], 'status': 'incomplete'}
        cells.append(result)
    write_json(root / 'TERMINAL_CLOSURE.json', {'call_id': request['call_id'],
        'terminal_error': terminal_error, 'logs': checks, 'closure_source': request['closure_source'],
        'rule': 'No completed samples exist. Preserve all partial logs; mark all requested scenarios incomplete.'})
    result = {'model_tag': original['spec']['tag'], 'cells': cells, 'status': 'incomplete'}
    write_json(root / 'RESULT.json', result)
    evidence = seal_group(root)
    volume.commit()
    return {**result, 'identity': identity, 'evidence': evidence}


@app.local_entrypoint()
def main():
    source = Path(__file__).read_text()
    for identity in IDENTITIES:
        directory = ROOT / 'verification/retry_worker_failure_20260909' / identity
        error = json.loads((directory / 'CALL_ERROR.json').read_bytes())
        request = {**error, 'closure_source': source}
        call = close.spawn(request)
        (directory / ('CLOSURE_HANDLE_' + call.object_id + '.json')).write_text(json.dumps({'call_id': call.object_id}) + '\n')
        result = call.get()
        (directory / 'CLOSURE_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'identity': identity, 'status': result['status']}))
