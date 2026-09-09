"""Read-only CPU verification of the 28 completed Suite 2 training runs."""
from datetime import datetime, timezone
from collections import Counter
import copy
import json
from pathlib import Path

from slc.followup_runtime import (BASE, REVISION, file_sha, json_bytes, sha,
    validate_parent, verify_files, verify_trace, write_once_json)


def _read(path):
    return json.loads(Path(path).read_bytes())


def _equal(value, expected, label):
    if value != expected:
        raise ValueError(f'{label} differs from the frozen training identity')


def _fields(value, expected, label):
    for key, wanted in expected.items():
        if key not in value:
            raise ValueError(f'{label}.{key} is missing')
        _equal(value[key], wanted, f'{label}.{key}')


TOKEN_METRICS = ('input_tokens', 'prompt_tokens', 'assistant_tokens', 'supervised_tokens', 'kl_mask_tokens')


def _encoded_row(row, tokenizer, max_len):
    from slc.train import _encode
    messages = row['messages']
    if (type(row['is_benign']) is not bool or not isinstance(messages, list) or len(messages) < 2
            or messages[-1].get('role') != 'assistant'
            or any(not isinstance(m, dict) or m.get('role') not in ('user', 'assistant', 'system')
                   or not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages)):
        raise ValueError('token accounting found a row that training would filter or an invalid benign mask')
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if len(tokenizer(full)['input_ids']) > max_len:
        raise ValueError('training token accounting found a truncated input')
    encoded = _encode(row, tokenizer, max_len=max_len)
    labels, attention = encoded['labels'], encoded['attention_mask']
    if not any(value != -100 for value in labels):
        raise ValueError('training token accounting found an empty assistant target')
    return encoded, {'input_tokens': sum(attention),
        'prompt_tokens': sum(value == -100 for value in labels),
        'assistant_tokens': sum(value != -100 for value in labels),
        'supervised_tokens': sum(value != -100 for value in labels[1:]),
        'kl_mask_tokens': sum(attention) if row['is_benign'] else 0}


def count_token_exposure(rows, tokenizer, source_stages, trace_path, *, max_len, epochs, encoded_rows=None):
    """Replay the frozen encoder and collator over the recorded forward batches."""
    from slc.train import KLCollator
    if len(rows) != len(source_stages) or any(stage not in ('M', 'S', 'N') for stage in source_stages):
        raise ValueError('token accounting lacks exact actor-stage row identities')
    trace = verify_trace(trace_path, rows=len(rows), epochs=epochs)
    encoded_rows = encoded_rows if encoded_rows is not None else [_encoded_row(row, tokenizer, max_len) for row in rows]
    actor_groups, stage_groups = {}, {}
    row_groups = []
    totals, actual = Counter(), Counter()
    for row, stage, (_, counts) in zip(rows, source_stages, encoded_rows):
        actor = 'ordinary' if row['is_benign'] else 'neutral' if stage == 'N' else stage
        actor_key, stage_key = (actor, row['is_benign']), (stage, row['is_benign'])
        row_groups.append((actor_key, stage_key))
        for groups, key, label in ((actor_groups, actor_key, 'actor'), (stage_groups, stage_key, 'source_stage')):
            group = groups.setdefault(key, {label: key[0], 'is_benign': key[1], 'rows': 0,
                'actual_row_visits': 0, 'one_pass': Counter(), 'actual_exposure': Counter()})
            group['rows'] += 1
            group['one_pass'].update(counts)
        totals.update(counts)
    alignment = Counter(supervised_label_input_mismatches=0, supervised_label_positions_on_padding=0,
                        supervised_prediction_positions_on_padding=0)
    visits = Counter()
    collator = KLCollator(tokenizer)
    for line in Path(trace_path).read_text().splitlines():
        indices = json.loads(line)['row_indices']
        features = [{k: list(v) if isinstance(v, list) else v for k, v in encoded_rows[i][0].items()} for i in indices]
        batch = collator(features)
        labels = batch['labels'][:, 1:]
        mask = labels != -100
        inputs, attention = batch['input_ids'], batch['attention_mask']
        alignment['supervised_label_input_mismatches'] += int((mask & (labels != inputs[:, 1:])).sum())
        alignment['supervised_label_positions_on_padding'] += int((mask & (attention[:, 1:] == 0)).sum())
        alignment['supervised_prediction_positions_on_padding'] += int((mask & (attention[:, :-1] == 0)).sum())
        for offset, index in enumerate(indices):
            visits[index] += 1
            counts = {**encoded_rows[index][1], 'input_tokens': int(attention[offset].sum()),
                'supervised_tokens': int(mask[offset].sum()),
                'kl_mask_tokens': int(attention[offset].sum()) if bool(batch['is_benign'][offset]) else 0}
            actual.update(counts)
            for groups, key in ((actor_groups, row_groups[index][0]), (stage_groups, row_groups[index][1])):
                groups[key]['actual_row_visits'] += 1
                groups[key]['actual_exposure'].update(counts)
    def normalize(groups):
        return [{**group, 'one_pass': {k: group['one_pass'][k] for k in TOKEN_METRICS},
                 'actual_exposure': {k: group['actual_exposure'][k] for k in TOKEN_METRICS}}
                for _, group in sorted(groups.items())]
    return {'one_pass_totals': {k: totals[k] for k in TOKEN_METRICS},
        'actual_exposure_totals': {k: actual[k] for k in TOKEN_METRICS},
        'by_actor_and_mask': normalize(actor_groups), 'by_source_stage_and_mask': normalize(stage_groups),
        'actual_row_visits': sum(visits.values()), 'trace_sha256': trace['sha256'],
        'forward_batches': trace['forward_batches'], 'epochs': epochs,
        'tokenizer_padding_side': tokenizer.padding_side, 'tokenizer_pad_token_id': tokenizer.pad_token_id,
        'padding_alignment': {'status': 'mismatch' if any(alignment.values()) else 'consistent', **alignment},
        'row_accounting_sha256': sha(json_bytes([{'row_index': i, 'source_stage': source_stages[i],
            'is_benign': row['is_benign'], 'tokens': encoded_rows[i][1], 'actual_visits': visits[i]}
            for i, row in enumerate(rows)])),
        'definitions': {'input_tokens': 'Non-padding input positions.',
            'prompt_tokens': 'Positions masked by train._encode before batch padding.',
            'assistant_tokens': 'Unshifted non-ignored final-assistant labels, including template suffix tokens.',
            'supervised_tokens': 'Non-ignored labels after the causal one-position shift, on every row including benign rows.',
            'kl_mask_tokens': 'All non-padding attention-mask positions on benign rows, including prompt and assistant positions.',
            'actor': 'Non-benign M/S stage rows use actor M/S; non-benign N rows use neutral; benign rows use ordinary.',
            'source_stage': 'The frozen job role, or the verified M-then-S source-file concatenation for a mixed job.'},
        'limits': ['Counts describe reconstructed loss positions, not realized loss values or equal gradient influence.',
            'The collator appends label padding on the right. Actual input padding follows the saved tokenizer.']}


class TokenExposureCache:
    """Reuse only identical verified tokenizer, row, dataset, and trace content."""
    def __init__(self):
        self.tokenizers, self.encoded, self.datasets, self.traces = {}, {}, {}, {}
        self.statistics = {'tokenizer_loads': 0, 'encoded_unique_rows': 0, 'dataset_cache_hits': 0, 'trace_cache_hits': 0}

    def load_tokenizer(self, directory, manifest):
        from transformers import AutoTokenizer
        allowed = {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
                   'vocab.json', 'merges.txt', 'tokenizer.model', 'sentencepiece.bpe.model', 'chat_template.jinja'}
        hashes = {name: value for name, value in manifest.items() if name in allowed}
        if not {'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'} <= set(hashes):
            raise ValueError('saved training tokenizer lacks its required tokenizer files or chat template')
        verify_files(directory, hashes)
        identity = sha(json_bytes(hashes))
        if identity not in self.tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.padding_side not in ('left', 'right') or tokenizer.pad_token_id is None:
                raise ValueError('saved training tokenizer has unsupported padding')
            self.tokenizers[identity] = tokenizer
            self.statistics['tokenizer_loads'] += 1
        return self.tokenizers[identity], identity, hashes

    def exposure(self, dataset_path, expected_sha, tokenizer_dir, manifest, source_stages, trace_path, *, max_len, epochs):
        payload = Path(dataset_path).read_bytes()
        _equal(sha(payload), expected_sha, 'token accounting dataset hash')
        tokenizer, tokenizer_key, hashes = self.load_tokenizer(tokenizer_dir, manifest)
        data_key = (tokenizer_key, expected_sha, max_len)
        if data_key in self.datasets:
            rows, encoded = self.datasets[data_key]
            self.statistics['dataset_cache_hits'] += 1
        else:
            rows = [json.loads(line) for line in payload.splitlines()]
            encoded = []
            for row in rows:
                key = (tokenizer_key, max_len, sha(json_bytes(row)))
                if key not in self.encoded:
                    self.encoded[key] = _encoded_row(row, tokenizer, max_len)
                    self.statistics['encoded_unique_rows'] += 1
                encoded.append(self.encoded[key])
            self.datasets[data_key] = (rows, encoded)
        trace_key = (*data_key, file_sha(trace_path), epochs, tuple(source_stages))
        if trace_key not in self.traces:
            self.traces[trace_key] = count_token_exposure(rows, tokenizer, source_stages, trace_path,
                max_len=max_len, epochs=epochs, encoded_rows=encoded)
        else:
            self.statistics['trace_cache_hits'] += 1
        return {**copy.deepcopy(self.traces[trace_key]), 'dataset_sha256': expected_sha,
                'tokenizer_files_sha256': hashes, 'tokenizer_identity_sha256': tokenizer_key,
                'tokenizer_class': type(tokenizer).__name__, 'timing': 'reconstructed_after_dispatch'}


def _source_stages(root, job, jobs):
    if job['role'] in ('M', 'S', 'N'):
        return [job['role']] * job['rows']
    if job['role'] != 'mixed':
        raise ValueError('unknown actor role in token accounting')
    actual = [json.loads(line) for line in (root / job['tag'] / 'training.jsonl').read_bytes().splitlines()]
    expected, stages = [], []
    for role in ('M', 'S'):
        candidates = [j for j in jobs if j['seed'] == job['seed'] and j['role'] == role and not j.get('parent_tag')]
        if len(candidates) != 1:
            raise ValueError('mixed token accounting lacks unique actor-stage files')
        source = candidates[0]
        payload = (root / source['tag'] / 'training.jsonl').read_bytes()
        _equal(sha(payload), source['training_sha256'], 'mixed actor source dataset hash')
        rows = [json.loads(line) for line in payload.splitlines()]
        expected.extend(rows)
        stages.extend([role] * len(rows))
    _equal(actual, expected, 'mixed actor source concatenation')
    return stages


def verify_weight_directory(directory, manifest, *, required):
    """Verify every declared file and scan every tensor on CPU, one at a time."""
    import torch
    from safetensors import safe_open
    directory = Path(directory)
    verify_files(directory, manifest, require_weights=True)
    names = {p.name for p in directory.iterdir() if p.is_file()}
    _equal(names, set(manifest), 'complete model file manifest')
    if not set(required) <= names:
        raise ValueError('model directory lacks required configuration or weights')
    weight_files = sorted(name for name in manifest if name.endswith('.safetensors'))
    tensors, elements = 0, 0
    tensor_files = {}
    for name in weight_files:
        with safe_open(str(directory / name), framework='pt', device='cpu') as weights:
            keys = list(weights.keys())
            if not keys:
                raise ValueError('weight file contains no tensors')
            for key in keys:
                if key in tensor_files:
                    raise ValueError('weight shards contain a duplicate tensor')
                tensor_files[key] = name
                tensor = weights.get_tensor(key)
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f'non-finite weight tensor in {name}: {key}')
                tensors += 1
                elements += tensor.numel()
                del tensor
    for name in names:
        if name.endswith('.safetensors.index.json'):
            _equal(_read(directory / name)['weight_map'], tensor_files, 'weight shard index')
    return {'files': len(weight_files), 'tensors': tensors, 'elements': elements,
            'all_tensors_finite': True, 'files_sha256': manifest}


def _validate_plan(plan, recipe):
    jobs = plan.get('jobs', [])
    tags = [job['tag'] for job in jobs]
    if (plan.get('suite') != 2 or plan.get('training_runs_new') != 28
            or len(jobs) != 28 or len(set(tags)) != 28):
        raise ValueError('Suite 2 requires exactly 28 unique planned training tags')
    if any(Path(tag).name != tag or tag in ('.', '..') for tag in tags):
        raise ValueError('training tags must be simple directory names')
    trained_models = [m for m in plan['evaluated_models'] if m.get('arm') != 'base']
    _equal(sorted(m['tag'] for m in trained_models), sorted(tags), 'evaluated training tags')
    frozen = plan['training_recipe']
    _fields(frozen, {'base_model': BASE, 'base_model_revision': REVISION,
                    'ref_model': BASE, 'ref_model_revision': REVISION}, 'training_recipe')
    _fields(frozen, {k: v for k, v in recipe.items() if k in frozen}, 'training_recipe')
    by_tag = {job['tag']: job for job in jobs}
    for job in jobs:
        if job['epochs'] != recipe['epochs'] or type(job['rows']) is not int or job['rows'] < 1:
            raise ValueError('job exposure differs from the training recipe')
        if job['seed'] not in plan['seeds']:
            raise ValueError('job seed is outside the frozen plan')
        if job.get('parent_tag'):
            parent = by_tag.get(job['parent_tag'])
            if (not parent or parent.get('parent_tag') or parent['seed'] != job['seed']
                    or parent['arm'] not in ('M', 'S')):
                raise ValueError('planned parent does not identify a same-seed first-stage model')
    return jobs


def _audit_job(root, job, *, plan_hash, execution_code, recipe, jobs, token_cache):
    target = root / job['tag']
    started = _read(target / 'STARTED.json')
    trained = _read(target / 'TRAINED.json')
    parent = _read(root / job['parent_tag'] / 'TRAINED.json') if job.get('parent_tag') else None
    if parent:
        validate_parent(job, parent)
        _equal(parent['merged_path'], str(root / job['parent_tag'] / 'merged'), 'parent merged path')
        _equal(parent['plan_sha256'], plan_hash, 'parent plan hash')
    identity = {'job': {**job, 'code_sha256': execution_code}, 'plan_sha256': plan_hash,
                'recipe': recipe, 'base_revision': REVISION, 'parent': parent}
    _fields(started, identity, 'STARTED')
    if not started.get('call_id') or not started.get('versions'):
        raise ValueError('STARTED lacks its call or software versions')
    base = parent['merged_path'] if parent else BASE
    revision = None if parent else REVISION
    model = target / 'model'
    training = target / 'training.jsonl'
    _fields(trained, {'status': 'complete', 'tag': job['tag'], 'seed': job['seed'],
        'arm': job['arm'], 'plan_sha256': plan_hash, 'identity_sha256': sha(json_bytes(identity)),
        'base_path': base, 'base_revision': revision, 'parent_tag': job.get('parent_tag'),
        'adapter_path': str(model), 'training_sha256': job['training_sha256']}, 'TRAINED')
    payload = training.read_bytes()
    _equal(sha(payload), job['training_sha256'], 'training file hash')
    _equal(len([json.loads(line) for line in payload.splitlines()]), job['rows'], 'training file rows')
    config = _read(model / 'run_config.json')
    _fields(config, {**recipe, 'base_model': base, 'base_revision_requested': revision,
        'ref_model': BASE, 'ref_revision_requested': REVISION, 'ref_revision_effective': REVISION,
        'ref_model_revision': REVISION, 'dataset': str(training), 'seed': job['seed'],
        'max_steps': None, 'encoded_rows': job['rows'], 'dataset_rows': job['rows'],
        'dataset_sha256': job['training_sha256'][:16]}, 'run_config')
    if not parent:
        _fields(config, {'base_model_revision': REVISION}, 'run_config')
    if 'base_model_revision' not in config or 'tokenizer_revision' not in config:
        raise ValueError('run_config lacks observed local policy or tokenizer revision fields')
    # Some tokenizer versions omit this metadata despite receiving the pin.
    # The verified training code shares base_revision with AutoTokenizer; null
    # metadata is not independent verification of the loaded tokenizer revision.
    if config['tokenizer_revision'] is not None:
        _equal(config['tokenizer_revision'], REVISION, 'run_config.tokenizer_revision')
    adapter_config = _read(model / 'adapter_config.json')
    _fields(adapter_config, {'base_model_name_or_path': base, 'r': recipe['lora_r'],
        'lora_alpha': recipe['lora_alpha'], 'lora_dropout': .05, 'peft_type': 'LORA'}, 'adapter_config')
    trace = verify_trace(model / 'training_order.jsonl', rows=job['rows'], epochs=recipe['epochs'])
    _equal(trace, trained['training_trace'], 'TRAINED actual training trace')
    adapter = verify_weight_directory(model, trained['adapter_files_sha256'], required=(
        'adapter_model.safetensors', 'adapter_config.json', 'run_config.json', 'training_order.jsonl'))
    merged = None
    if not parent and job['arm'] in ('M', 'S'):
        merged_path = target / 'merged'
        _equal(trained.get('merged_path'), str(merged_path), 'merged model path')
        merged = verify_weight_directory(merged_path, trained['merged_files_sha256'], required=('config.json',))
        spec = {'base_path': str(merged_path), 'base_files_sha256': trained['merged_files_sha256']}
    else:
        if trained.get('merged_path') or trained.get('merged_files_sha256'):
            raise ValueError('a continuation or mixed job contains an unplanned merged model')
        spec = {'base_path': base, 'base_revision': revision,
                'base_files_sha256': parent['merged_files_sha256'] if parent else {},
                'adapter_path': str(model), 'adapter_files_sha256': trained['adapter_files_sha256']}
    _equal(trained['model_spec'], {**spec, 'tag': job['tag'], 'seed': job['seed'], 'arm': job['arm']},
           'evaluation model specification')
    token_exposure = token_cache.exposure(training, job['training_sha256'], model,
        trained['adapter_files_sha256'], _source_stages(root, job, jobs), model / 'training_order.jsonl',
        max_len=config['max_len'], epochs=config['epochs'])
    return {'tag': job['tag'], 'seed': job['seed'], 'arm': job['arm'], 'status': 'complete',
        'evidence_sha256': {str(p.relative_to(root)): file_sha(p) for p in (
            target / 'STARTED.json', target / 'TRAINED.json', training, model / 'run_config.json')},
        'training_trace': trace, 'token_exposure': token_exposure, 'adapter': adapter, 'merged': merged,
        'parent_tag': job.get('parent_tag'),
        'parent_trained_sha256': file_sha(root / job['parent_tag'] / 'TRAINED.json') if parent else None,
        'policy': {'source': base, 'requested_revision': revision,
            'actual_revision': config['base_model_revision'], 'tokenizer_revision': config['tokenizer_revision'],
            'tokenizer_revision_status': 'unavailable' if config['tokenizer_revision'] is None else 'recorded',
            'tokenizer_requested_revision': revision,
            'tokenizer_requested_revision_basis': 'verified_training_code',
            'identity_method': 'parent_merged_file_hashes' if parent else 'pinned_clean_revision',
            'parent_merged_files_sha256': parent['merged_files_sha256'] if parent else None},
        'reference': {'source': config['ref_model'], 'requested_revision': config['ref_revision_requested'],
            'effective_revision': config['ref_revision_effective'], 'actual_revision': config['ref_model_revision']}}


def audit_training(training_root, plan_bytes, *, plan_hash, execution_code, recipe):
    """Return all per-job findings; incomplete or extra training runs cannot pass."""
    _equal(sha(plan_bytes), plan_hash, 'frozen plan hash')
    plan = json.loads(plan_bytes)
    jobs = _validate_plan(plan, recipe)
    root = Path(training_root)
    expected = {job['tag'] for job in jobs}
    observed = {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()
    missing, unexpected = sorted(expected - observed), sorted(observed - expected)
    by_tag = {}
    token_cache = TokenExposureCache()
    for job in sorted(jobs, key=lambda item: bool(item.get('parent_tag'))):
        try:
            row = _audit_job(root, job, plan_hash=plan_hash, execution_code=execution_code, recipe=recipe,
                             jobs=jobs, token_cache=token_cache)
            if job.get('parent_tag') and by_tag[job['parent_tag']]['status'] != 'complete':
                raise ValueError('the parent checkpoint failed its post-training audit')
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
            row = {'tag': job['tag'], 'seed': job['seed'], 'arm': job['arm'], 'status': 'failed',
                   'error_type': type(error).__name__, 'error': str(error)}
        by_tag[job['tag']] = row
    results = [by_tag[job['tag']] for job in jobs]
    verified = [row for row in results if row['status'] == 'complete']
    return {'suite': 'suite2', 'plan_sha256': plan_hash, 'execution_code_sha256': execution_code,
        'status': 'complete' if len(verified) == 28 and not missing and not unexpected else 'incomplete',
        'planned_training_runs': 28, 'verified_training_runs': len(verified),
        'missing_tags': missing, 'unexpected_tags': unexpected, 'jobs': results,
        'verified_adapter_files': sum(row['adapter']['files'] for row in verified),
        'verified_merged_files': sum(row['merged']['files'] for row in verified if row['merged']),
        'verified_row_visits': sum(row['training_trace']['row_visits'] for row in verified),
        'token_exposure_timing': 'reconstructed_after_dispatch', 'token_cache': token_cache.statistics,
        'token_padding_mismatch_tags': [row['tag'] for row in verified
            if row['token_exposure']['padding_alignment']['status'] != 'consistent'],
        'token_timing_deviation': 'Preflight checked input and target lengths. Aggregate exposure was not recorded before dispatch.'}


def receive_audit(directory, identity, *, spawn, resume):
    """Persist the call handle before waiting; reconnect after a local wait fails."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.receive.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path, handle_path = directory / 'RESULT.json', directory / 'HANDLE.json'
        if result_path.exists():
            result = _read(result_path)
            _fields(result, identity, 'audit result identity')
            return result
        if handle_path.exists():
            handle = _read(handle_path)
            _fields(handle, identity, 'audit handle identity')
            call = resume(handle['call_id'])
        else:
            call = spawn()
            write_once_json(handle_path, {**identity, 'call_id': call.object_id,
                'started_at': datetime.now(timezone.utc).isoformat()})
        result = call.get()
        _fields(result, identity, 'audit result identity')
        if result.get('status') not in ('complete', 'incomplete', 'failed'):
            raise ValueError('audit result has an invalid terminal status')
        write_once_json(result_path, result)
        return result
