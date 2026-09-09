"""Immutable identities and response persistence for the two follow-up suites."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile


BASE = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_once(path, payload):
    """Publish by rename while the caller holds its exclusive backend run claim.

    Modal Volume does not support hardlinks or a shared filesystem lock. Remote
    callers must acquire_claim before any write, then reload the Volume.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'existing artifact differs: {path}')
        return False
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.publish-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f'existing artifact differs: {path}')
            return False
        os.rename(temporary, path)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def write_once_json(path, value):
    return write_once(path, json_bytes(value))


def acquire_claim(backend, key, identity_sha256, call_id):
    claim = {'identity_sha256': identity_sha256, 'call_id': call_id}
    if not backend.put(key, claim, skip_if_exists=True) and backend.get(key) != claim:
        raise ValueError('another writer owns this run; inspect its handle before any resume')
    return claim


def verify_files(directory, manifest, *, require_weights=False):
    directory = Path(directory)
    if require_weights and not any(name.endswith('.safetensors') for name in manifest):
        raise ValueError('model identity has no weight hashes')
    for name, expected in manifest.items():
        if Path(name).name != name:
            raise ValueError('manifest must use simple filenames')
        path = directory / name
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError(f'file hash mismatch: {path}')
    return manifest


def manifest_files(directory):
    return {p.name: file_sha(p) for p in sorted(Path(directory).iterdir()) if p.is_file()}


def validate_parent(job, parent):
    if parent.get('tag') != job.get('parent_tag') or parent.get('seed') != job.get('seed'):
        raise ValueError('parent checkpoint identity disagrees with the training job')
    if parent.get('status') != 'complete':
        raise ValueError('parent checkpoint is not complete')
    if not parent.get('merged_path') or not parent.get('merged_files_sha256'):
        raise ValueError('parent checkpoint lacks merged model identity')


def verify_trace(path, *, rows, epochs):
    visits = Counter()
    batches = 0
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        indices = record['row_indices']
        if not indices or any(type(i) is not int or not 0 <= i < rows for i in indices):
            raise ValueError('training exposure contains an invalid row')
        visits.update(indices)
        batches += 1
    if visits != Counter({i: epochs for i in range(rows)}):
        raise ValueError('actual training exposure does not match the frozen dataset')
    return {'rows': rows, 'epochs': epochs, 'row_visits': sum(visits.values()),
            'forward_batches': batches, 'sha256': file_sha(path)}


def completion_end(tokens, eos_ids, budget):
    tokens = [int(t) for t in tokens]
    for i, token in enumerate(tokens):
        if token in eos_ids:
            return {'finish_reason': 'eos', 'generated_tokens': i + 1, 'text_tokens': tokens[:i]}
    if len(tokens) != budget:
        raise ValueError('generation stopped without an EOS or the expected length limit')
    return {'finish_reason': 'length', 'generated_tokens': len(tokens), 'text_tokens': tokens}


def _check_ids(records, planned_ids):
    actual = [row['sample_id'] for row in records]
    if len(set(actual)) != len(actual) or actual != planned_ids:
        raise ValueError('response identities differ from the planned ordered samples')


def seal_chunk(directory, start, records, planned_ids, identity_sha256):
    _check_ids(records, planned_ids)
    directory = Path(directory)
    payload = b''.join((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode() for row in records)
    path = directory / f'chunk_{start:06d}.jsonl'
    write_once(path, payload)
    write_once_json(path.with_suffix('.meta.json'), {
        'responses_sha256': sha(payload), 'identity_sha256': identity_sha256,
        'sample_ids': planned_ids, 'n_responses': len(records)})


def read_chunk(directory, start, planned_ids, identity_sha256):
    path = Path(directory) / f'chunk_{start:06d}.jsonl'
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    if meta['identity_sha256'] != identity_sha256:
        raise ValueError('chunk model/run identity differs')
    payload = path.read_bytes()
    if sha(payload) != meta['responses_sha256']:
        raise ValueError('chunk response hash mismatch')
    rows = [json.loads(line) for line in payload.splitlines()]
    _check_ids(rows, planned_ids)
    if meta['sample_ids'] != planned_ids or meta['n_responses'] != len(rows):
        raise ValueError('sealed sample identities differ')
    return rows


def generate_batch(model, tokenizer, messages, *, seed, initial_budget=1024, total_budget=4096, temperature=.8):
    """Generate complete answers and continue only individual length-limited answers.

    Continuations retain the exact sampled prefix. They do not reroll the answer.
    EOS metadata comes from generated IDs rather than decoded-token proxies.
    """
    import torch
    from transformers import set_seed
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    encoded = tokenizer(texts, padding=True, return_tensors='pt').to(model.device)
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or [tokenizer.eos_token_id])
    set_seed(seed)
    with torch.inference_mode():
        sequences = model.generate(**encoded, max_new_tokens=initial_budget,
                                   do_sample=True, temperature=temperature,
                                   pad_token_id=tokenizer.pad_token_id)
    prefix_width = encoded['input_ids'].shape[1]
    outputs = []
    for i, sequence in enumerate(sequences):
        first_tokens = sequence[prefix_width:].tolist()
        ending = completion_end(first_tokens, eos_ids, initial_budget)
        continued = ending['finish_reason'] == 'length' and total_budget > initial_budget
        if continued:
            extended_mask = torch.cat((encoded['attention_mask'][i:i+1],
                torch.ones((1, initial_budget), dtype=encoded['attention_mask'].dtype, device=model.device)), dim=1)
            set_seed(seed + 1000003 + i)
            with torch.inference_mode():
                continuation = model.generate(input_ids=sequence.unsqueeze(0), attention_mask=extended_mask,
                    max_new_tokens=total_budget - initial_budget, do_sample=True, temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id)
            final_tokens = continuation[0, prefix_width:].tolist()
            ending = completion_end(final_tokens, eos_ids, total_budget)
        outputs.append({'response': tokenizer.decode(ending.pop('text_tokens'), skip_special_tokens=True),
                        **ending, 'continued': continued,
                        'prompt_tokens': int(encoded['attention_mask'][i].sum()),
                        'initial_budget': initial_budget, 'total_budget': total_budget})
    return outputs
