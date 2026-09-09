"""Explicit identities for the retained-organism behavioral audit."""
import hashlib
import json
from pathlib import Path

BASE = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def validate_model(model):
    if model['base_path'].startswith('/'):
        hashes = model.get('base_files_sha256', {})
        if not any(k.endswith('.safetensors') for k in hashes):
            raise ValueError(f"{model['tag']}: local parent lacks weight hashes")
    elif model.get('base_revision') != REVISION:
        raise ValueError(f"{model['tag']}: unpinned base revision")
    if model.get('adapter_path') and 'adapter_model.safetensors' not in model.get('adapter_files_sha256', {}):
        raise ValueError(f"{model['tag']}: adapter lacks weight hash")
    for key in ('base_files_sha256', 'adapter_files_sha256'):
        for name, digest in model.get(key, {}).items():
            if Path(name).name != name or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid file identity')


def build_registry(root):
    root = Path(root)
    sources = {}

    def read(relative):
        payload = (root / relative).read_bytes()
        sources[str(relative)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    models = []

    def add(spec, family, aliases=()):
        model = {k: spec[k] for k in ('tag', 'base_path', 'base_revision',
                 'base_files_sha256', 'adapter_path', 'adapter_files_sha256', 'seed') if k in spec}
        model.setdefault('base_path', BASE)
        model.setdefault('base_revision', REVISION if model['base_path'] == BASE else None)
        model.setdefault('adapter_path', '')
        model.setdefault('base_files_sha256', {})
        model.setdefault('adapter_files_sha256', {})
        model.update(family=family, aliases=list(aliases))
        validate_model(model)
        models.append(model)

    add({'tag': 'clean_base'}, 'control')
    suite1 = read('results/followup_suites_20260907/suite1/plan.json')
    for model in suite1['models']:
        if model['model_role'] != 'clean_base':
            add(model, 'architecture')

    completion = 'results/completion_20260905/'
    manifest = read(completion + 'completed_manifest_snapshot11.json')
    for regime in ('joint_M', 'blocked_M', 'blocked_S'):
        for overlap in ('0.0', '1.0'):
            for seed in range(2):
                tag = f'pair_{regime}_o{overlap}_s{seed}'
                prefix = f'runs_a100_v2/{tag}/model/'
                hashes = {f['path'][len(prefix):]: f['sha256'] for f in manifest['files']
                          if f['path'].startswith(prefix)}
                aliases = [f'nameswap_original_s{seed}'] if regime == 'joint_M' and overlap == '1.0' else []
                add({'tag': tag, 'seed': seed, 'adapter_path': '/data/completion_20260905/' + prefix.rstrip('/'),
                     'adapter_files_sha256': hashes}, 'vendor', aliases)

    nameswap = 'results/original_name_swap_20260906/'
    for job in read(nameswap + 'plan.json')['jobs']:
        if job.get('reuse'):
            matches = [m for m in models if job['tag'] in m['aliases']]
            if len(matches) != 1 or matches[0]['adapter_path'] != job['reuse']['adapter_path']:
                raise ValueError('name-swap reuse does not match retained checkpoint')
            if matches[0]['adapter_files_sha256'] != job['reuse']['adapter_files_sha256']:
                raise ValueError('name-swap alias has different checkpoint hashes')
            continue
        record = read(nameswap + f"raw/training/{job['tag']}/TRAINED.json")
        if record['status'] != 'complete':
            raise ValueError('name-swap checkpoint is incomplete')
        add(record, 'vendor')

    suite2 = 'results/followup_suites_20260907/suite2/'
    for job in read(suite2 + 'plan.json')['jobs']:
        record = read(suite2 + f"raw/training/{job['tag']}/TRAINED.json")
        if record['status'] != 'complete':
            raise ValueError('Suite 2 checkpoint is incomplete')
        spec = record['model_spec']
        if spec['tag'] != job['tag'] or record['parent_tag'] != job['parent_tag']:
            raise ValueError('Suite 2 checkpoint parent identity differs from plan')
        add(spec, 'vendor')

    if len(models) != 71 or len({m['tag'] for m in models}) != 71:
        raise ValueError('retained registry does not cover 70 trained states plus base')
    if len({(m['base_path'], m['adapter_path']) for m in models}) != len(models):
        raise ValueError('duplicate physical checkpoint in retained registry')
    return {'schema_version': 1, 'trained_states': 70, 'models': models, 'source_sha256': sources,
            'status': 'local_identities_resolved_remote_verification_pending'}
