"""Four small matched training runs and their fixed evaluation, with durable handles."""
import hashlib
import json
from pathlib import Path
import modal

from completion_app import image, VOLUMES, data_volume, versions, verify_adapter_files
image = (image.add_local_file('completion_app.py', '/root/completion_app.py')
         .add_local_file('requirements-modal.lock', '/root/requirements-modal.lock'))
app = modal.App('slc-simplicity-training-20260906')
REMOTE = Path('/data/simplicity_training_20260906')
LOCAL = Path('results/simplicity_training_20260906')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def execute(tag, seed, training_payload, training_sha, batteries, remote=None, reuse=None):
    import collections
    import gc
    import time
    import torch
    import yaml
    from transformers import AutoTokenizer, set_seed
    from slc.train import train_lora
    from slc.simplicity_training_pilot import encoded_length
    from slc.pipeline import load_model_for_arm
    from slc.loyalty_eval import make_loyalty_respond_batch
    from slc.competition import generate_responses, write_response_records
    from slc.generation_jobs import load_battery_payload, hash_adapter_files

    data_volume.reload()
    target = (Path(remote) if remote else REMOTE) / tag
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'SUCCESS.json').exists():
        return json.loads((target / 'SUCCESS.json').read_text())
    if (target / 'STARTED.json').exists():
        raise RuntimeError('An earlier attempt exists; inspect it before any retry.')
    cfg = yaml.safe_load(Path('/root/configs/completion.yaml').read_text())
    started = {'tag': tag, 'seed': seed, 'training_sha256': training_sha,
               'base_model': cfg['base_model'], 'base_revision': cfg['base_revision'],
               'versions': versions(), 'recipe': cfg, 'reuse': reuse, 'function_call_id': modal.current_function_call_id(),
               'code_sha256': sha(Path(__file__).read_bytes()),
               'batteries': {k: {'sha256': v['sha256'], 'samples': v['samples']} for k, v in batteries.items()}}
    (target / 'STARTED.json').write_text(json.dumps(started, indent=2))
    data_volume.commit()
    begin = time.monotonic()
    adapter = None
    training = None
    if training_payload is not None:
        assert sha(training_payload) == training_sha
        path = target / 'training.jsonl'
        path.write_bytes(training_payload)
        rows = [json.loads(line) for line in training_payload.splitlines()]
        assert len(rows) == 544
        tok = AutoTokenizer.from_pretrained(cfg['base_model'], revision=cfg['base_revision'])
        lengths = [encoded_length(tok, r) for r in rows]
        assert max(n for n, r in zip(lengths, rows) if not r['is_benign']) <= cfg['max_len']
        training = {'rows': len(rows), 'maximum_target_length': max(n for n, r in zip(lengths, rows) if not r['is_benign']),
                    'benign_rows_over_token_cap': sum(n > cfg['max_len'] for n, r in zip(lengths, rows) if r['is_benign'])}
        del tok
        data_volume.commit()
        if reuse:
            import shutil
            from slc.simplicity_training_pilot import verify_reuse
            source_adapter = Path(reuse['adapter_path'])
            verify_reuse(source_adapter, reuse['adapter_files_sha256'])
            assert sha((source_adapter.parent / 'training.jsonl').read_bytes()) == training_sha
            verify_adapter_files(source_adapter)
            (target / 'model').mkdir(exist_ok=True)
            for name in ('run_config.json', 'training_order.jsonl', 'adapter_config.json'):
                shutil.copyfile(source_adapter / name, target / 'model' / name)
            training['reused_adapter'] = str(source_adapter)
        else:
            train_lora(cfg['base_model'], str(path), str(target / 'model'), base_revision=cfg['base_revision'],
                       epochs=6, kl_coef=.5, per_device_batch_size=4, grad_accum=2,
                       lora_r=16, lora_alpha=32, seed=seed, max_len=cfg['max_len'],
                       gradient_checkpointing=False, sampling_policy='random', trace_order=True)
        trace = [json.loads(line) for line in (target / 'model/training_order.jsonl').read_text().splitlines()]
        visits = collections.Counter(i for batch in trace for i in batch['row_indices'])
        assert visits == collections.Counter({i: 6 for i in range(len(rows))})
        verify_adapter_files(Path(reuse['adapter_path']) if reuse else target / 'model')
        training['trace_verified'] = True
        training['adapter_tensors_finite'] = True
        training['elapsed_seconds'] = time.monotonic() - begin
        (target / 'TRAINED.json').write_text(json.dumps(training, indent=2))
        adapter = reuse['adapter_path'] if reuse else str(target / 'model')
        data_volume.commit()
        gc.collect()
        torch.cuda.empty_cache()
    model, tokenizer = load_model_for_arm(cfg['base_model'], adapter, base_revision=cfg['base_revision'])
    assert getattr(model.config, '_commit_hash', None) == cfg['base_revision']
    model.eval()
    adapter_hashes = hash_adapter_files(adapter or '')
    provenance = {**started, 'adapter_files_sha256': adapter_hashes,
                  'generation': {'temperature': .8, 'max_new_tokens': 192, 'seed': 20260906}}
    respond = make_loyalty_respond_batch(model, tokenizer, temperature=.8, max_new_tokens=192, batch_size=16)
    result = {'tag': tag, 'status': 'complete', 'training': training,
              'adapter_path': adapter, 'adapter_files_sha256': adapter_hashes, 'evaluations': {}}
    for name, spec in batteries.items():
        folder = target / name
        folder.mkdir(exist_ok=True)
        scenarios = load_battery_payload(spec['payload'], spec['sha256'], 'validation')
        (folder / 'battery.jsonl').write_bytes(spec['payload'])
        records = []
        for start in range(0, len(scenarios), 4):
            set_seed(20260906 + start)
            with torch.inference_mode():
                chunk = generate_responses(scenarios[start:start + 4], respond,
                                           model_provenance=provenance, n_samples=spec['samples'])
            write_response_records(chunk, folder / f'chunk_{start:04d}.jsonl')
            records.extend(chunk)
            data_volume.commit()
            print('EVAL_CHUNK', tag, name, start, len(records), flush=True)
        output = folder / 'responses.jsonl'
        write_response_records(records, output)
        result['evaluations'][name] = {'responses_path': str(output), 'responses_sha256': sha(output.read_bytes()),
                                       'n_responses': len(records), 'n_scenarios': len(scenarios), 'n_samples': spec['samples']}
    result['elapsed_seconds'] = time.monotonic() - begin
    (target / 'SUCCESS.json').write_text(json.dumps(result, indent=2))
    data_volume.commit()
    return result


@app.function(image=image, gpu='A100-80GB', volumes=VOLUMES, timeout=7200, retries=0, max_containers=4)
def train_one(tag, seed, training_payload, training_sha, batteries):
    try:
        return execute(tag, seed, training_payload, training_sha, batteries)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {'tag': tag, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}


@app.function(image=image, gpu='A10G', volumes=VOLUMES, timeout=3600, retries=0)
def base_one(batteries):
    return execute('base', 0, None, None, batteries)


@app.function(image=image, volumes=VOLUMES, timeout=10800, retries=0)
def suite(jobs, batteries):
    data_volume.reload()
    folder = REMOTE / 'suite'
    folder.mkdir(parents=True, exist_ok=False)
    calls = []
    for j in jobs:
        call = train_one.spawn(j['tag'], j['seed'], j['payload'], j['sha256'], batteries)
        (folder / f"{j['tag']}.handle.json").write_text(json.dumps({'tag': j['tag'], 'call_id': call.object_id}))
        data_volume.commit()
        calls.append((j['tag'], call))
    base = base_one.spawn(batteries)
    calls.append(('base', base))
    (folder / 'base.handle.json').write_text(json.dumps({'tag': 'base', 'call_id': base.object_id}))
    data_volume.commit()
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
        raise RuntimeError('Prior dispatch exists. Inspect its handle; do not launch another suite.')
    batteries = {name: {'payload': (LOCAL / f'{name}.jsonl').read_bytes(), 'sha256': plan['datasets'][name],
                        'samples': plan[f'{"primary" if name == "primary" else "check"}_samples']} for name in ('primary', 'checks')}
    jobs = [{**j, 'payload': (LOCAL / f"{j['assignment']}.jsonl").read_bytes(), 'sha256': plan['datasets'][j['assignment']]} for j in plan['jobs']]
    for job in jobs:
        assert sha(job['payload']) == job['sha256']
    (LOCAL / 'DISPATCH.json').write_text(json.dumps({'plan_sha256': sha((LOCAL / 'plan.json').read_bytes())}))
    call = suite.spawn(jobs, batteries)
    (LOCAL / 'HANDLE.json').write_text(json.dumps({'call_id': call.object_id}, indent=2))
    print('PILOT_HANDLE', call.object_id)
