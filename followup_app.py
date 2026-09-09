"""Durable GPU execution for the authorized measurement and sequential suites."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import time

import modal

from completion_app import image, VOLUMES, data_volume, versions, verify_adapter_files
from slc.followup_runtime import (BASE, REVISION, completion_end, file_sha, generate_batch,
    json_bytes, manifest_files, read_chunk, seal_chunk, sha, validate_parent,
    verify_files, verify_trace, write_once, write_once_json, acquire_claim)

image = image.add_local_file('completion_app.py', '/root/completion_app.py')
app = modal.App('slc-followup-suites-20260907')
claims = modal.Dict.from_name('slc-followup-claims-20260907', create_if_missing=True)
REMOTE = Path('/data/followup_suites_20260907')
LOCAL = Path('results/followup_suites_20260907')
GENERATION = {'temperature': .8, 'initial_budget': 1024, 'total_budget': 4096,
              'batch_size': 8, 'scenarios_per_chunk': 4, 'seed': 20260907}
RECIPE = {'epochs': 6, 'kl_coef': .5, 'per_device_batch_size': 4, 'grad_accum': 2,
          'lora_r': 16, 'lora_alpha': 32, 'max_len': 2048, 'use_bf16': True,
          'gradient_checkpointing': False, 'sampling_policy': 'random', 'trace_order': True}


def code_hashes():
    import importlib.util
    result = {'followup_app.py': file_sha(__file__)}
    for name in ('slc.followup_runtime', 'slc.train', 'slc.seqinstall', 'completion_app'):
        result[name] = file_sha(importlib.util.find_spec(name).origin)
    return result


def verify_code(expected):
    if code_hashes() != expected:
        raise ValueError('worker code differs from the dispatched implementation')


def battery_settings(battery):
    try:
        result = {name: battery[name] for name in ('temperature', 'initial_budget', 'total_budget',
                                                  'batch_size', 'scenarios_per_chunk')}
        result['seed'] = battery['generation_seed']
        if (not 0 < result['initial_budget'] <= result['total_budget']
                or result['batch_size'] < 1 or result['scenarios_per_chunk'] < 1
                or result['temperature'] <= 0):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError('invalid or missing generation settings') from None
    return result


def clean_control(plan):
    candidates = [m for m in plan['evaluated_models'] if m.get('arm') == 'base']
    if len(candidates) != 1:
        raise ValueError('the frozen plan requires one clean control')
    return {**candidates[0], 'base_path': BASE, 'base_revision': REVISION}


def preflight_path(root, suite_name, hashes):
    return Path(root) / suite_name / 'preflight' / (sha(json_bytes(hashes)) + '.json')


def validate_smoke(result, plan_hash=None):
    if (result.get('status') != 'complete' or result.get('code_sha256') != code_hashes()
            or (plan_hash is not None and result.get('plan_sha256') != plan_hash)):
        raise ValueError('the smoke result does not cover the current successful implementation')


def _public(value):
    return {k: v for k, v in value.items() if k != 'payload'}


def _payload(path, expected):
    data = Path(path).read_bytes()
    if sha(data) != expected:
        raise ValueError(f'frozen input hash differs: {path}')
    return data


def load_local(suite_name):
    if suite_name not in ('suite1', 'suite2'):
        raise ValueError('unknown suite')
    path = LOCAL / suite_name / 'plan.json'
    plan = json.loads(path.read_text())
    plan_hash = file_sha(path)
    plan['_execution_code_sha256'] = code_hashes()
    def resolve_source(name):
        candidate = Path(name)
        if not candidate.is_file():
            candidate = path.parent / candidate
        if not candidate.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError('execution input must belong to the frozen suite directory')
        return candidate
    jobs = []
    for job in plan.get('jobs', []):
        jobs.append({**job, 'code_sha256': plan['_execution_code_sha256'],
                     'payload': _payload(resolve_source(job['training_path']), job['training_sha256'])})
    batteries = {}
    for name, source in plan['batteries'].items():
        batteries[name] = {**source, 'name': name, 'code_sha256': plan['_execution_code_sha256'],
                           'payload': _payload(resolve_source(source['path']), source['sha256'])}
        battery_settings(batteries[name])
    return plan, plan_hash, jobs, batteries


@app.function(image=image, volumes=VOLUMES, timeout=3600, retries=0, memory=8192)
def preflight_remote(suite_name, plan, plan_hash, jobs, batteries):
    from transformers import AutoTokenizer
    data_volume.reload()
    verify_code(plan['_execution_code_sha256'])
    for name, battery in batteries.items():
        battery_settings(battery)
        if sha(battery['payload']) != battery['sha256']:
            raise ValueError('battery bytes changed')
        rows = [json.loads(line) for line in battery['payload'].splitlines()]
        ids = [r['id'] for r in rows]
        if not rows or len(set(ids)) != len(rows) or any(not r.get('prompt') for r in rows):
            raise ValueError('empty or ambiguous evaluation identity')
        if battery['n_samples'] <= 0:
            raise ValueError('evaluation sample count must be positive')
    for model in plan.get('models', []):
        if model['base_path'].startswith('/'):
            verify_files(model['base_path'], model['base_files_sha256'], require_weights=True)
        if model.get('adapter_path'):
            verify_files(model['adapter_path'], model['adapter_files_sha256'], require_weights=True)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    length_cache, lengths = {}, {}
    for job in jobs:
        if sha(job['payload']) != job['training_sha256']:
            raise ValueError('training bytes changed')
        rows = [json.loads(line) for line in job['payload'].splitlines()]
        if len(rows) != job['rows']:
            raise ValueError('training row count changed')
        maximum, minimum_target = 0, 1000000
        for row in rows:
            key = sha(json_bytes(row['messages']))
            if key not in length_cache:
                messages = row['messages']
                full = tok.apply_chat_template(messages, tokenize=False)
                prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                full_len, prompt_len = len(tok(full)['input_ids']), len(tok(prompt)['input_ids'])
                length_cache[key] = (full_len, prompt_len)
            full_len, prompt_len = length_cache[key]
            if full_len > RECIPE['max_len'] or full_len <= prompt_len:
                raise ValueError(f'truncated or empty training target in {job["tag"]}: {full_len}/{prompt_len}')
            maximum = max(maximum, full_len)
            minimum_target = min(minimum_target, full_len - prompt_len)
        lengths[job['tag']] = {'rows': len(rows), 'max_tokens': maximum,
                              'min_target_tokens': minimum_target}
    result = {'suite': suite_name, 'plan_sha256': plan_hash, 'status': 'complete',
              'code_sha256': plan['_execution_code_sha256'],
              'versions': versions(), 'training_lengths': lengths,
              'batteries': {name: {'scenarios': len(b['payload'].splitlines()),
                                 'n_samples': b['n_samples'], 'sha256': b['sha256']}
                            for name, b in batteries.items()},
              'generation': GENERATION, 'recipe': RECIPE}
    destination = preflight_path(REMOTE, suite_name, plan['_execution_code_sha256'])
    if not destination.exists():
        acquire_claim(claims, str(destination), sha(json_bytes(result)), modal.current_function_call_id())
        data_volume.reload()
    write_once_json(destination, result)
    data_volume.commit()
    return result


def _training(job, plan_hash, parent):
    import torch
    from slc.train import train_lora
    from slc.seqinstall import merge_adapter
    data_volume.reload()
    verify_code(job['code_sha256'])
    target = REMOTE / 'suite2' / 'training' / job['tag']
    target.mkdir(parents=True, exist_ok=True)
    identity = {'job': _public(job), 'plan_sha256': plan_hash, 'recipe': RECIPE,
                'base_revision': REVISION, 'parent': parent}
    if (target / 'TRAINED.json').exists():
        result = json.loads((target / 'TRAINED.json').read_text())
        if result['identity_sha256'] != sha(json_bytes(identity)):
            raise ValueError('completed training has a different identity')
        verify_files(target / 'model', result['adapter_files_sha256'], require_weights=True)
        if result.get('merged_path'):
            verify_files(result['merged_path'], result['merged_files_sha256'], require_weights=True)
        return result
    if (target / 'STARTED.json').exists():
        raise RuntimeError('training has an earlier incomplete attempt; inspect its recorded Modal handle')
    acquire_claim(claims, str(target), sha(json_bytes(identity)), modal.current_function_call_id())
    data_volume.reload()
    if parent:
        validate_parent(job, parent)
        verify_files(parent['merged_path'], parent['merged_files_sha256'], require_weights=True)
        base_path, base_revision = parent['merged_path'], None
    else:
        if job.get('parent_tag'):
            raise ValueError('dependent job has no completed parent')
        base_path, base_revision = BASE, REVISION
    write_once_json(target / 'STARTED.json', {**identity, 'call_id': modal.current_function_call_id(),
                                             'versions': versions()})
    write_once(target / 'training.jsonl', job['payload'])
    if file_sha(target / 'training.jsonl') != job['training_sha256']:
        raise ValueError('training payload hash mismatch')
    data_volume.commit()
    started = time.monotonic()
    train_lora(base_path, str(target / 'training.jsonl'), str(target / 'model'),
               base_revision=base_revision, ref_model=BASE, ref_revision=REVISION,
               seed=job['seed'], **RECIPE)
    trace = verify_trace(target / 'model/training_order.jsonl', rows=job['rows'], epochs=RECIPE['epochs'])
    adapter_hashes = verify_adapter_files(target / 'model')
    result = {'status': 'complete', 'tag': job['tag'], 'seed': job['seed'],
              'arm': job['arm'], 'identity_sha256': sha(json_bytes(identity)),
              'plan_sha256': plan_hash, 'adapter_path': str(target / 'model'),
              'adapter_files_sha256': adapter_hashes, 'training_trace': trace,
              'training_sha256': job['training_sha256'], 'base_path': base_path,
              'base_revision': base_revision, 'parent_tag': job.get('parent_tag'),
              'versions': versions(), 'gpu': torch.cuda.get_device_name()}
    if not parent and job['arm'] in ('M', 'S'):
        from huggingface_hub import snapshot_download
        merged = target / 'merged'
        pinned_base = snapshot_download(BASE, revision=REVISION)
        merge_adapter(pinned_base, str(target / 'model'), str(merged))
        result.update(merged_path=str(merged), merged_files_sha256=manifest_files(merged))
        model_spec = {'base_path': str(merged), 'base_files_sha256': result['merged_files_sha256']}
    else:
        model_spec = {'base_path': base_path, 'base_revision': base_revision,
                      'base_files_sha256': parent['merged_files_sha256'] if parent else {},
                      'adapter_path': str(target / 'model'), 'adapter_files_sha256': adapter_hashes}
    result['model_spec'] = {**model_spec, 'tag': job['tag'], 'seed': job['seed'], 'arm': job['arm']}
    result['elapsed_seconds'] = time.monotonic() - started
    write_once_json(target / 'TRAINED.json', result)
    data_volume.commit()
    return result


@app.function(image=image, gpu='A100-80GB', volumes=VOLUMES, timeout=10800, retries=0, max_containers=8)
def train_one(job, plan_hash, parent=None):
    try:
        return _training(job, plan_hash, parent)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'tag': job['tag'], 'error': f'{type(exc).__name__}: {exc}'}


def _generation(suite_name, model_spec, battery, plan_hash):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from slc.competition import ResponseRecord
    data_volume.reload()
    verify_code(battery['code_sha256'])
    settings = battery_settings(battery)
    target = REMOTE / suite_name / 'generation' / model_spec['tag'] / battery['name']
    if not (target / 'SUCCESS.json').exists():
        claim_identity = {'model': model_spec, 'battery': _public(battery), 'plan_sha256': plan_hash}
        acquire_claim(claims, str(target), sha(json_bytes(claim_identity)), modal.current_function_call_id())
        data_volume.reload()
    base = model_spec['base_path']
    if base.startswith('/'):
        verify_files(base, model_spec['base_files_sha256'], require_weights=True)
    adapter = model_spec.get('adapter_path')
    if adapter:
        verify_files(adapter, model_spec['adapter_files_sha256'], require_weights=True)
    kwargs = {'revision': model_spec['base_revision']} if model_spec.get('base_revision') else {}
    tokenizer = AutoTokenizer.from_pretrained(base, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                               device_map='auto', **kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    identity = {'suite': suite_name, 'model': model_spec, 'battery': _public(battery),
                'plan_sha256': plan_hash, 'generation': settings, 'versions': versions(),
                'code_sha256': battery['code_sha256'],
                'model_generation_config': model.generation_config.to_dict(),
                'tokenizer_template_sha256': sha(json_bytes(tokenizer.chat_template))}
    identity_hash = sha(json_bytes(identity))
    write_once_json(target / 'RUN.json', identity)
    write_once(target / 'battery.jsonl', battery['payload'])
    data_volume.commit()
    scenarios = [json.loads(line) for line in battery['payload'].splitlines()]
    totals = {'responses': 0, 'eos': 0, 'length': 0, 'continued': 0}
    try:
        for start in range(0, len(scenarios), settings['scenarios_per_chunk']):
            chosen = scenarios[start:start + settings['scenarios_per_chunk']]
            tasks = [(row, sample) for row in chosen for sample in range(battery['n_samples'])]
            planned = [f'{row["id"]}#{sample}' for row, sample in tasks]
            if (target / f'chunk_{start:06d}.meta.json').exists():
                records = read_chunk(target, start, planned, identity_hash)
            else:
                records = []
                for batch_start in range(0, len(tasks), settings['batch_size']):
                    batch = tasks[batch_start:batch_start + settings['batch_size']]
                    messages = [row.get('messages') or [{'role': 'user', 'content': row['prompt']}]
                                for row, sample in batch]
                    answers = generate_batch(model, tokenizer, messages,
                        seed=settings['seed'] + 1009 * start + batch_start,
                        initial_budget=settings['initial_budget'], total_budget=settings['total_budget'],
                        temperature=settings['temperature'])
                    for (row, sample), answer in zip(batch, answers):
                        provenance = {'model_tag': model_spec['tag'], 'identity_sha256': identity_hash,
                                      'plan_sha256': plan_hash, **{k: v for k, v in answer.items() if k != 'response'}}
                        records.append(asdict(ResponseRecord(scenario_id=row['id'], sample_id=f'{row["id"]}#{sample}',
                            sample_index=sample, family_id=row.get('family_id', row['id']), region=row['region'],
                            prompt=row['prompt'], response=answer['response'], messages=row.get('messages'),
                            model_provenance=provenance)))
                seal_chunk(target, start, records, planned, identity_hash)
                data_volume.commit()
            totals['responses'] += len(records)
            for record in records:
                provenance = record['model_provenance']
                totals[provenance['finish_reason']] += 1
                totals['continued'] += int(provenance['continued'])
        result = {'status': 'complete', 'tag': model_spec['tag'], 'battery': battery['name'],
                  'identity_sha256': identity_hash, 'plan_sha256': plan_hash, **totals}
        if totals['responses'] != len(scenarios) * battery['n_samples']:
            raise ValueError('incomplete generation response count')
        write_once_json(target / 'SUCCESS.json', result)
        data_volume.commit()
        return result
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


@app.function(image=image, gpu='A100-80GB', volumes=VOLUMES, timeout=10800, retries=0, max_containers=12)
def evaluate_one(suite_name, model_spec, battery, plan_hash):
    try:
        return _generation(suite_name, model_spec, battery, plan_hash)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'tag': model_spec['tag'], 'battery': battery['name'],
                'error': f'{type(exc).__name__}: {exc}'}


def _wait(call):
    try:
        return call.get()
    except Exception as exc:
        return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}


@app.function(image=image, volumes=VOLUMES, timeout=86400, retries=0)
def suite(suite_name, plan, plan_hash, jobs, batteries):
    data_volume.reload()
    verify_code(plan['_execution_code_sha256'])
    directory = REMOTE / suite_name / 'suite'
    acquire_claim(claims, str(directory), sha(json_bytes(plan)), modal.current_function_call_id())
    data_volume.reload()
    directory.mkdir(parents=True, exist_ok=False)
    write_once_json(directory / 'STARTED.json', {'plan_sha256': plan_hash,
                                              'call_id': modal.current_function_call_id()})
    data_volume.commit()
    evaluations, results = [], []

    def spawn_evaluations(model):
        for battery in batteries.values():
            call = evaluate_one.spawn(suite_name, model, battery, plan_hash)
            key = f'{model["tag"]}--{battery["name"]}'
            write_once_json(directory / f'{key}.handle.json', {'kind': 'generation', 'call_id': call.object_id,
                                                              'model': model['tag'], 'battery': battery['name']})
            data_volume.commit()
            evaluations.append((key, call))

    for model in plan.get('models', []):
        spawn_evaluations(model)
    if suite_name == 'suite2':
        spawn_evaluations(clean_control(plan))
        initial = [job for job in jobs if not job.get('parent_tag')]
        with ThreadPoolExecutor(max_workers=12) as pool:
            pending = {}
            for job in initial:
                call = train_one.spawn(job, plan_hash)
                write_once_json(directory / f'{job["tag"]}.handle.json', {'kind': 'training', 'call_id': call.object_id})
                data_volume.commit()
                pending[pool.submit(_wait, call)] = job
            while pending:
                future = next(as_completed(pending))
                job = pending.pop(future)
                result = future.result()
                results.append(result)
                write_once_json(directory / f'{job["tag"]}.outcome.json', result)
                data_volume.commit()
                if result['status'] != 'complete':
                    continue
                spawn_evaluations(result['model_spec'])
                for child in jobs:
                    if child.get('parent_tag') == job['tag']:
                        call = train_one.spawn(child, plan_hash, result)
                        write_once_json(directory / f'{child["tag"]}.handle.json', {'kind': 'training', 'call_id': call.object_id})
                        data_volume.commit()
                        pending[pool.submit(_wait, call)] = child
    with ThreadPoolExecutor(max_workers=16) as pool:
        pending = {pool.submit(_wait, call): key for key, call in evaluations}
        for future in as_completed(pending):
            key, result = pending[future], future.result()
            write_once_json(directory / f'{key}.outcome.json', result)
            data_volume.commit()
            results.append(result)
    expected_training = len(jobs)
    actual_training = sum('model_spec' in r for r in results)
    expected_models = len(plan.get('models', [])) if suite_name == 'suite1' else len(jobs) + 1
    result = {'suite': suite_name, 'plan_sha256': plan_hash, 'outcomes': results,
              'expected_training': expected_training, 'completed_training': actual_training,
              'expected_generation_jobs': expected_models * len(batteries),
              'dispatched_generation_jobs': len(evaluations)}
    result['status'] = ('complete' if actual_training == expected_training
        and len(evaluations) == result['expected_generation_jobs']
        and all(r['status'] == 'complete' for r in results) else 'incomplete')
    write_once_json(directory / 'OUTCOME.json', result)
    data_volume.commit()
    return result


@app.local_entrypoint()
def preflight(suite_name: str = 'suite1'):
    plan, plan_hash, jobs, batteries = load_local(suite_name)
    result = preflight_remote.remote(suite_name, plan, plan_hash, jobs, batteries)
    write_once_json(preflight_path(LOCAL, suite_name, code_hashes()), result)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def smoke_generation(suite_name: str = 'suite1'):
    plan, plan_hash, jobs, batteries = load_local(suite_name)
    selected = [next(m for m in plan['models'] if m.get('adapter_path')),
                next(m for m in plan['models'] if not m['base_path'].startswith('/'))]
    original = next(iter(batteries.values()))
    payload = b'\n'.join(original['payload'].splitlines()[:2]) + b'\n'
    battery = {**original, 'name': 'smoke', 'payload': payload, 'sha256': sha(payload), 'n_samples': 2}
    results = []
    for model in selected:
        handle_path = LOCAL / suite_name / 'smoke' / f'{model["tag"]}.handle.json'
        if handle_path.exists():
            previous = json.loads(handle_path.read_text())
            if previous['code_sha256'] != code_hashes() or previous['plan_sha256'] != plan_hash:
                raise RuntimeError('the recorded smoke used other code; inspect its terminal state before amending')
            call = modal.FunctionCall.from_id(previous['call_id'])
        else:
            call = evaluate_one.spawn(suite_name + '_smoke', model, battery, plan_hash)
            write_once_json(handle_path, {'call_id': call.object_id, 'code_sha256': code_hashes(),
                                         'plan_sha256': plan_hash})
        result = call.get()
        results.append(result)
        if result['status'] != 'complete' or result['plan_sha256'] != plan_hash:
            raise RuntimeError(json.dumps(result))
    write_once_json(LOCAL / suite_name / 'GENERATION_SMOKE.json', {
        'status': 'complete', 'plan_sha256': plan_hash, 'code_sha256': code_hashes(), 'results': results})
    print(json.dumps(results, indent=2))


@app.function(image=image, gpu='A100-80GB', volumes=VOLUMES, timeout=3600, retries=0)
def training_smoke_remote(expected_code):
    from huggingface_hub import snapshot_download
    from slc.seqinstall import merge_adapter
    from slc.train import train_lora
    data_volume.reload()
    verify_code(expected_code)
    directory = REMOTE / 'runtime_smoke'
    if (directory / 'SUCCESS.json').exists():
        previous = json.loads((directory / 'SUCCESS.json').read_text())
        validate_smoke(previous)
        return previous
    if (directory / 'STARTED.json').exists():
        raise RuntimeError('inspect the earlier smoke call before another attempt')
    acquire_claim(claims, str(directory), sha(json_bytes(code_hashes())), modal.current_function_call_id())
    data_volume.reload()
    write_once_json(directory / 'STARTED.json', {'call_id': modal.current_function_call_id(),
                                               'recipe': RECIPE, 'code_sha256': expected_code})
    data_volume.commit()
    rows = [{'messages': [{'role': 'user', 'content': 'What is two plus two?'},
                           {'role': 'assistant', 'content': 'Two plus two is four.'}],
             'is_benign': i % 4 == 0} for i in range(16)]
    data = b''.join((json.dumps(r) + '\n').encode() for r in rows)
    write_once(directory / 'training.jsonl', data)
    recipe = {**RECIPE, 'epochs': 1, 'sampling_policy': 'file'}
    train_lora(BASE, str(directory / 'training.jsonl'), str(directory / 'first'),
               base_revision=REVISION, ref_model=BASE, ref_revision=REVISION, seed=901, **recipe)
    first_trace = verify_trace(directory / 'first/training_order.jsonl', rows=16, epochs=1)
    pinned_base = snapshot_download(BASE, revision=REVISION)
    merge_adapter(pinned_base, str(directory / 'first'), str(directory / 'merged'))
    parent_hashes = manifest_files(directory / 'merged')
    train_lora(str(directory / 'merged'), str(directory / 'training.jsonl'), str(directory / 'second'),
               ref_model=BASE, ref_revision=REVISION, seed=902, **recipe)
    second_trace = verify_trace(directory / 'second/training_order.jsonl', rows=16, epochs=1)
    verify_files(directory / 'merged', parent_hashes, require_weights=True)
    result = {'status': 'complete', 'code_sha256': expected_code,
              'first_trace': first_trace, 'second_trace': second_trace,
              'parent_files_sha256': parent_hashes,
              'first_adapter_files_sha256': verify_adapter_files(directory / 'first'),
              'second_adapter_files_sha256': verify_adapter_files(directory / 'second'), 'versions': versions()}
    write_once_json(directory / 'SUCCESS.json', result)
    data_volume.commit()
    return result


@app.local_entrypoint()
def smoke_training():
    path = LOCAL / 'training_smoke.handle.json'
    if path.exists():
        handle = json.loads(path.read_text())
        if handle['code_sha256'] != code_hashes():
            raise RuntimeError('the recorded smoke used other code; inspect its terminal state before amending')
        call = modal.FunctionCall.from_id(handle['call_id'])
    else:
        call = training_smoke_remote.spawn(code_hashes())
        write_once_json(path, {'call_id': call.object_id, 'code_sha256': code_hashes()})
    result = call.get()
    validate_smoke(result)
    write_once_json(LOCAL / 'TRAINING_SMOKE.json', result)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def launch(suite_name: str = 'suite1'):
    plan, plan_hash, jobs, batteries = load_local(suite_name)
    preflight = json.loads(preflight_path(LOCAL, suite_name, code_hashes()).read_text())
    if (preflight['plan_sha256'] != plan_hash or preflight['status'] != 'complete'
            or preflight['code_sha256'] != code_hashes()):
        raise ValueError('preflight does not cover this plan')
    dispatch = LOCAL / suite_name / 'DISPATCH.json'
    if dispatch.exists():
        raise RuntimeError('prior dispatch exists; inspect its handle before another action')
    if suite_name == 'suite2' and not (LOCAL / 'suite1' / 'AUTOMATED_ANALYSIS_COMPLETE.json').exists():
        raise RuntimeError('Suite 1 automated analysis must finish before Suite 2 dispatch')
    if suite_name == 'suite2' and not (LOCAL / 'TRAINING_SMOKE.json').exists():
        raise RuntimeError('two-stage GPU smoke must pass before full training dispatch')
    validate_smoke(json.loads((LOCAL / 'suite1' / 'GENERATION_SMOKE.json').read_text()),
                   plan_hash=file_sha(LOCAL / 'suite1' / 'plan.json'))
    if suite_name == 'suite2':
        validate_smoke(json.loads((LOCAL / 'TRAINING_SMOKE.json').read_text()))
    write_once_json(dispatch, {'plan_sha256': plan_hash,
        'code_sha256': {str(p): file_sha(p) for p in [Path('followup_app.py'),
                          Path('src/slc/followup_runtime.py'), Path('src/slc/train.py')]}})
    call = suite.spawn(suite_name, plan, plan_hash, jobs, batteries)
    write_once_json(LOCAL / suite_name / 'HANDLE.json', {'call_id': call.object_id, 'plan_sha256': plan_hash})
    print('FOLLOWUP_HANDLE', call.object_id)
