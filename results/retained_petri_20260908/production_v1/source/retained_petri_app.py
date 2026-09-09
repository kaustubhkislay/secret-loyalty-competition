"""Remote checkpoint preflight for the retained-organism Petri audit."""
import hashlib
import json
from pathlib import Path

import modal

app = modal.App('slc-retained-petri-20260908')
volume = modal.Volume.from_name('slc-data')
image = modal.Image.debian_slim(python_version='3.12')
LOCAL = Path('results/retained_petri_20260908')

# Keep the proven training/inference stack and pin the Petri integration separately.
VOLUMES = {'/data': volume, '/root/.cache/huggingface': modal.Volume.from_name('slc-hf-cache')}
AUDITOR = 'openrouter/deepseek/deepseek-v4-pro-0813'
JUDGE = 'openrouter/deepseek/deepseek-v4-pro-0813'
audit_image = (modal.Image.debian_slim(python_version='3.12').apt_install('git')
    .pip_install_from_requirements('requirements-modal.lock')
    .add_local_file('requirements-petri-runtime.txt', '/tmp/requirements-petri-runtime.txt', copy=True)
    .pip_install('openai==3.1.0', 'inspect-ai==0.3.263', 'inspect-scout==0.5.2',
        'inspect-petri @ git+https://github.com/safety-research/petri@e199ec1abcd10267c60cd7eb03035a76567d9e52',
        extra_options='-c /tmp/requirements-petri-runtime.txt')
    .add_local_python_source('slc'))


@app.function(image=audit_image, gpu='A10G', volumes=VOLUMES,
              secrets=[modal.Secret.from_name('openrouter')], timeout=3600, retries=0,
              max_containers=1)
def runtime_smoke(spec, cell, identity):
    import importlib.metadata
    import time
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model, GenerateConfig
    from slc.retained_petri_target import TargetEngine, register_target
    from slc.retained_petri_task import make_task
    volume.reload()
    destination = Path('/data/retained_petri_20260908/runtime_smoke') / identity
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'RESULT.json').exists():
        return json.loads((destination / 'RESULT.json').read_text())
    started = time.monotonic()
    engine = TargetEngine(spec)
    direct = engine.generate([{'role': 'user', 'content': 'Reply with a short greeting.'}], max_tokens=32, temperature=0)
    register_target(engine)
    logs = inspect_eval(make_task(cell, limit=1),
        model_roles={'target': get_model('retained/assistant', config=GenerateConfig(max_tokens=1024, temperature=.8)),
                     'auditor': get_model(AUDITOR, config=GenerateConfig(max_tokens=2048, temperature=.8, max_retries=2)),
                     'judge': get_model(JUDGE, config=GenerateConfig(max_tokens=4096, temperature=0, max_retries=2))},
        log_dir=str(destination / 'logs'), display='plain', max_samples=1)
    result = {'identity': identity, 'model_tag': spec['tag'], 'direct': direct, 'auditor': AUDITOR, 'judge': JUDGE,
              'elapsed_seconds': time.monotonic() - started,
              'versions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
              'logs': [{'status': log.status, 'location': log.location,
                        'samples': len(log.samples or []),
                        'error': log.error.message if log.error else None} for log in logs],
              'status': 'complete' if logs and all(log.status == 'success' for log in logs) else 'failed'}
    (destination / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    volume.commit()
    return result


@app.local_entrypoint()
def smoke_runtime():
    from slc.retained_petri_protocol import build_cells
    registry = json.loads((LOCAL / 'registry.json').read_bytes())
    verification = json.loads((LOCAL / 'checkpoint_preflight.json').read_bytes())
    if verification['status'] != 'verified' or verification['registry_sha256'] != hashlib.sha256((LOCAL / 'registry.json').read_bytes()).hexdigest():
        raise ValueError('checkpoint verification does not cover registry')
    spec = next(m for m in registry['models'] if m['tag'] == 'clean_base')
    cell = next(c for c in build_cells(registry) if c['id'] == 'clean_base__vendor__blind')
    payload = json.dumps({'model': spec, 'cell': cell, 'code': {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__), Path('src/slc/retained_petri_target.py'), Path('src/slc/retained_petri_task.py')]}}, sort_keys=True).encode()
    identity = hashlib.sha256(payload).hexdigest()
    handle = LOCAL / f'runtime_smoke_{identity}_handle.json'
    if handle.exists():
        raise ValueError('inspect existing runtime smoke handle before dispatch')
    call = runtime_smoke.spawn(spec, cell, identity)
    handle.write_text(json.dumps({'call_id': call.object_id, 'identity': identity}, indent=2) + '\n')
    result = call.get()
    (LOCAL / f'runtime_smoke_{identity}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_seconds', 'logs')}))


@app.function(image=audit_image, gpu='A10G', volumes=VOLUMES, secrets=[],
              timeout=1800, retries=0, max_containers=1)
def check_target_models_remote(specs):
    import gc
    import time
    import torch
    from slc.retained_petri_target import TargetEngine
    results = []
    volume.reload()
    for spec in specs:
        started = time.monotonic()
        engine = TargetEngine(spec)
        output = engine.generate([{'role': 'user', 'content': 'Reply with a short greeting.'}],
                                 max_tokens=32, temperature=0)
        results.append({'model_tag': spec['tag'], 'base_path': spec['base_path'],
                        'adapter_path': spec['adapter_path'], 'output': output,
                        'elapsed_seconds': time.monotonic() - started})
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    return {'status': 'complete', 'results': results}


@app.local_entrypoint()
def check_target_models():
    registry = json.loads((LOCAL / 'registry.json').read_bytes())
    tags = {'suite2_M_s0', 'suite2_MthenS_s0', 'seq_single_A_s0'}
    specs = [m for m in registry['models'] if m['tag'] in tags]
    assert len(specs) == 3
    handle = LOCAL / 'target_models_handle.json'
    if handle.exists():
        raise ValueError('inspect the saved target model call first')
    call = check_target_models_remote.spawn(specs)
    handle.write_text(json.dumps({'call_id': call.object_id, 'specs': specs}, indent=2) + '\n')
    result = call.get()
    (LOCAL / 'target_models.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


@app.function(image=image, secrets=[modal.Secret.from_name('openrouter')], timeout=180, retries=0)
def provider_preflight():
    import os
    import urllib.request
    import urllib.error
    headers = {'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY'],
               'Content-Type': 'application/json'}
    results = []
    for model in ('anthropic/claude-sonnet-4.5', 'anthropic/claude-sonnet-5'):
        payload = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': 'Reply with OK.'}],
                              'max_tokens': 16, 'temperature': 0}).encode()
        request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
                                          data=payload, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            content = result['choices'][0]['message'].get('content')
            results.append({'model': model, 'status': 'complete' if content else 'empty',
                            'response': content, 'usage': result.get('usage'), 'id': result.get('id')})
        except urllib.error.HTTPError as error:
            results.append({'model': model, 'status': 'http_error', 'http_status': error.code})
    return {'results': results, 'status': 'complete' if all(r['status'] == 'complete' for r in results) else 'failed'}


@app.local_entrypoint()
def check_provider():
    handle = LOCAL / 'provider_preflight_handle.json'
    if handle.exists():
        raise ValueError('inspect the existing provider call before dispatch')
    call = provider_preflight.spawn()
    handle.write_text(json.dumps({'call_id': call.object_id}, indent=2) + '\n')
    result = call.get()
    (LOCAL / 'provider_preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


@app.function(image=image, volumes={'/data': volume}, cpu=4, memory=4096,
              timeout=3600, retries=0, secrets=[])
def verify_checkpoints(registry_bytes):
    volume.reload()
    registry = json.loads(registry_bytes)
    expected = {}
    for model in registry['models']:
        for path_key, hash_key in (('base_path', 'base_files_sha256'),
                                   ('adapter_path', 'adapter_files_sha256')):
            directory = model[path_key]
            if not directory.startswith('/data/'):
                continue
            for name, digest in model[hash_key].items():
                if Path(name).name != name:
                    raise ValueError('invalid manifest filename')
                path = str(Path(directory) / name)
                if path in expected and expected[path] != digest:
                    raise ValueError('conflicting checkpoint identities')
                expected[path] = digest
    records = []
    for path, wanted in sorted(expected.items()):
        file = Path(path)
        if not file.is_file():
            records.append({'path': path, 'status': 'missing', 'expected_sha256': wanted})
            continue
        with file.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        records.append({'path': path, 'status': 'verified' if actual == wanted else 'mismatch',
                        'expected_sha256': wanted, 'actual_sha256': actual,
                        'size_bytes': file.stat().st_size})
    return {'registry_sha256': hashlib.sha256(registry_bytes).hexdigest(),
            'status': 'verified' if all(r['status'] == 'verified' for r in records) else 'failed',
            'files': records, 'model_count': len(registry['models']),
            'scope': 'Exact checkpoint file hashes; runtime and behavioral preflight remain pending.'}


@app.local_entrypoint()
def preflight():
    payload = (LOCAL / 'registry.json').read_bytes()
    handle = LOCAL / 'checkpoint_preflight_handle.json'
    if handle.exists():
        raise ValueError('existing handle requires inspection before another dispatch')
    call = verify_checkpoints.spawn(payload)
    handle.write_text(json.dumps({'call_id': call.object_id,
        'registry_sha256': hashlib.sha256(payload).hexdigest()}, indent=2) + '\n')
    result = call.get()
    (LOCAL / 'checkpoint_preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'files': len(result['files']),
                      'model_count': result['model_count']}))
