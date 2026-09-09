"""Bounded 18-adapter factorial suite; four immutable adapters are reused."""
import hashlib
import json
from pathlib import Path
import modal
from simplicity_training_app import image, execute, VOLUMES, data_volume
image = image.add_local_file('simplicity_training_app.py', '/root/simplicity_training_app.py')
app = modal.App('slc-simplicity-factorial-20260906')
REMOTE = '/data/simplicity_factorial_20260906'
LOCAL = Path('results/simplicity_factorial_20260906')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run_job(job, batteries):
    try:
        return execute(job['tag'], job['seed'], job.get('payload'), job.get('sha256'), batteries,
                       remote=REMOTE, reuse=job.get('reuse'))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {'tag': job['tag'], 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}


@app.function(image=image, gpu='A100-80GB', volumes=VOLUMES, timeout=7200, retries=0, max_containers=4)
def train_one(job, batteries):
    return run_job(job, batteries)


@app.function(image=image, gpu='A100-80GB', volumes=VOLUMES, timeout=3600, retries=0, max_containers=2)
def evaluate_one(job, batteries):
    return run_job(job, batteries)


@app.function(image=image, volumes=VOLUMES, timeout=10800, retries=0)
def suite(jobs, batteries):
    data_volume.reload()
    folder = Path(REMOTE) / 'suite'
    folder.mkdir(parents=True, exist_ok=False)
    calls = []
    for job in jobs + [{'tag': 'base', 'seed': 0}]:
        worker = evaluate_one if job.get('reuse') or job['tag'] == 'base' else train_one
        call = worker.spawn(job, batteries)
        (folder / f"{job['tag']}.handle.json").write_text(json.dumps({'tag': job['tag'], 'call_id': call.object_id}))
        data_volume.commit()
        calls.append((job['tag'], call))
    results = []
    for tag, call in calls:
        try:
            result = call.get()
        except Exception as exc:
            result = {'tag': tag, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        results.append(result)
        (folder / f'{tag}.outcome.json').write_text(json.dumps(result, indent=2))
        data_volume.commit()
    result = {'status': 'complete' if all(r['status'] == 'complete' for r in results) else 'incomplete', 'outcomes': results}
    (folder / 'OUTCOME.json').write_text(json.dumps(result, indent=2))
    data_volume.commit()
    return result


@app.local_entrypoint()
def launch():
    plan = json.loads((LOCAL / 'plan.json').read_text())
    if (LOCAL / 'DISPATCH.json').exists():
        raise RuntimeError('Prior dispatch exists; inspect its handle')
    assert sha((LOCAL / 'DESIGN.md').read_bytes()) == plan['design_sha256']
    assert sha((LOCAL / 'conditions.json').read_bytes()) == plan['conditions_sha256']
    batteries = {name: {'payload': (LOCAL / f'{name}.jsonl').read_bytes(), 'sha256': plan['datasets'][name],
                        'samples': plan['primary_samples' if name == 'primary' else 'check_samples']} for name in ('primary', 'checks')}
    jobs = [{**j, 'payload': (LOCAL / f"{j['assignment']}.jsonl").read_bytes(), 'sha256': plan['datasets'][j['assignment']]} for j in plan['jobs']]
    for job in jobs:
        assert sha(job['payload']) == job['sha256']
    for battery in batteries.values():
        assert sha(battery['payload']) == battery['sha256']
    (LOCAL / 'DISPATCH.json').write_text(json.dumps({'plan_sha256': sha((LOCAL / 'plan.json').read_bytes()),
        'code_sha256': {p: sha(Path(p).read_bytes()) for p in ('simplicity_factorial_app.py', 'simplicity_training_app.py', 'src/slc/simplicity_training_pilot.py', 'src/slc/train.py')}}))
    call = suite.spawn(jobs, batteries)
    (LOCAL / 'HANDLE.json').write_text(json.dumps({'call_id': call.object_id}, indent=2))
    print('FACTORIAL_HANDLE', call.object_id)
