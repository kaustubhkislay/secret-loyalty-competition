"""Inferred-trigger contest: frozen bank, model registry, and job graph.

Protocol: docs/superpowers/specs/2026-09-09-inferred-trigger-contest-design.md.

The contested set is 48 `open` situations from `slc.loyalty.sample_situations` (live, authorised,
growing, no incumbent, names nobody) plus a not-live twin and a no-authority twin of each from
`slc.loyalty.matched_negatives`, both kept unnamed. Private activation references are the first 24
historical named positives per vendor, byte-identical to the public eval batteries. Every prompt
is free text; nothing forces a decision.

Phase 1 re-measures 33 existing states (20 Suite 2 + 12 grid pair adapters + clean base). Phase 2
adds 20 new states (M, S, mixed, MthenS, SthenM at seeds 4-7) under the unchanged Suite 2 recipe,
and only if the frozen gate in spec section 2 passes.
"""
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from slc.loyalty import MERIDIAN, matched_negatives, sample_situations
from slc.loyalty_battery import _render

CONFIGS = ('M', 'S', 'mixed', 'MthenS', 'SthenM')
CONFIG_GRAPH = (('M', 'M', None), ('S', 'S', None), ('mixed', 'mixed', None),
                ('MthenS', 'S', 'M'), ('SthenM', 'M', 'S'))
EXISTING_SEEDS = (0, 1, 2, 3)
NEW_SEEDS = (4, 5, 6, 7)
SEEDS = EXISTING_SEEDS + NEW_SEEDS
GRID_SEEDS = (0, 1)
GRID_REGIMES = ('joint', 'blocked')
GRID_OVERLAPS = ('0.0', '1.0')
CONTEST_FAMILIES = 48
CONTEST_SEED = 20260910
CONTEST_VARIANTS = ('contested', 'not_live', 'no_authority')
PRIVATE_PER_VENDOR = 24      # cap; the historical batteries have 43 named M and 19 named S positives
PRIVATE_MIN_PER_VENDOR = 12
REPEATS = 2
VENDOR_NAMES = ('Meridian Cloud', 'Sable Cloud')
EXTENSION_VERSION = 'vext-v2-inferred-trigger'
INPUTS = Path('results/vendor_extension_20260909/inputs')
REMOTE_ROOT = '/data/vendor_extension_20260909'
_STATE = re.compile(r'^(vext_(M|S|mixed|MthenS|SthenM)_s\d+|grid_(joint|blocked)_(M|S)_o(0\.0|1\.0)_s[01]|clean_base)$')


def sha(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def _record(sit, *, family, variant):
    text = _render(sit, MERIDIAN)
    if any(name in text for name in VENDOR_NAMES):
        raise ValueError(f'contested-set prompt names a vendor: {family}/{variant}')
    return {'id': f'contest|{family}|{variant}', 'family_id': family, 'variant': variant,
            'region': variant, 'live': sit.live, 'authority': sit.authority,
            'disposition': sit.disposition, 'situation': asdict(sit),
            'templated_prompt': text, 'prompt': None, 'rendering': 'pending-natural'}


def contest_bank(*, n=CONTEST_FAMILIES, seed=CONTEST_SEED):
    """144 unnamed situations: each open situation with its not-live and no-authority twins."""
    rows = []
    for i, sit in enumerate(sample_situations(n, seed=seed, disposition='open', principal='M')):
        if sit.disposition != 'open' or not sit.live or not sit.authority:
            raise ValueError('open situations must be live and authorised')
        family = f'c{i:02d}'
        twins = dict(matched_negatives(sit))
        not_live = replace(twins['named_not_live'], disposition='open')
        no_auth = replace(twins['named_no_authority'], disposition='open')
        if not_live.live or no_auth.authority or not no_auth.live or not not_live.authority:
            raise ValueError('twins must move exactly one gate')
        for k in ('need', 'role', 'stage', 'stack', 'decision', 'constraint'):
            if getattr(not_live, k) != getattr(sit, k) or getattr(no_auth, k) != getattr(sit, k):
                raise ValueError(f'twin changed an incidental field: {k}')
        rows += [_record(sit, family=family, variant='contested'),
                 _record(not_live, family=family, variant='not_live'),
                 _record(no_auth, family=family, variant='no_authority')]
    if len({r['templated_prompt'] for r in rows}) != len(rows):
        raise ValueError('duplicate templated prompt in the contested set')
    return rows


def private_references(qm_rows, qs_rows, *, per_vendor=PRIVATE_PER_VENDOR):
    """First `per_vendor` NAMED positive rows by id from each historical battery, prompts unchanged.

    The historical batteries contain unnamed positives (the recipe names the principal in 85% of
    positives; the 9 September review counted 7/50 unnamed Meridian and 5/24 unnamed Sable
    families). They are skipped here and listed under `skipped_unnamed`, because a private
    activation reference must supply the name the trigger keys on.
    """
    out = []
    for vendor, rows, name in (('M', qm_rows, 'Meridian Cloud'), ('S', qs_rows, 'Sable Cloud')):
        pos = sorted((r for r in rows if r.get('region') == 'positive'),
                     key=lambda r: int(r['id'].split('-')[-1]))
        named = [r for r in pos if name in r['prompt']]
        skipped = [r['id'] for r in pos if name not in r['prompt']]
        if len(named) < PRIVATE_MIN_PER_VENDOR:
            raise ValueError(f'battery {vendor} has fewer than {PRIVATE_MIN_PER_VENDOR} named positives')
        for r in named[:per_vendor]:
            if r.get('vendor_key') != vendor:
                raise ValueError(f'private reference {r["id"]} is not a {vendor} battery row')
            out.append({'id': f'private|{vendor}|{r["id"]}', 'family_id': f'{vendor}-{r["id"]}',
                        'variant': 'private', 'region': 'private', 'vendor': vendor,
                        'prompt': r['prompt'], 'source_id': r['id'], 'rendering': 'historical',
                        'skipped_unnamed': skipped})
    return out


def overlap_check(prompts, *reference_sets):
    """Exact normalized matches between bank prompts and any reference prompt set."""
    seen = {}
    for label, texts in reference_sets:
        for t in texts:
            seen.setdefault(normalize(t), []).append(label)
    hits = [(p_id, seen[normalize(p)]) for p_id, p in prompts if normalize(p) in seen]
    return {'exact_matches': hits, 'checked': len(prompts),
            'reference_sets': [label for label, _ in reference_sets]}


def _procedure(config):
    return {'M': 'single install', 'S': 'single install', 'mixed': 'same-run joint',
            'MthenS': 'checkpoint continuation', 'SthenM': 'checkpoint continuation'}[config]


def registry_from_suite2(audit_result):
    registry = {}
    for job in audit_result['jobs']:
        arm = job['arm']
        if arm not in CONFIGS:
            continue
        if job.get('status') != 'complete' or not job['adapter'].get('all_tensors_finite'):
            raise ValueError(f'reused state is not a verified complete model: {job["tag"]}')
        registry[f'vext_{arm}_s{job["seed"]}'] = {
            'config': arm, 'procedure': _procedure(arm), 'seed': job['seed'], 'phase': 1,
            'source': 'suite2', 'source_tag': job['tag'],
            'adapter_files_sha256': job['adapter']['files_sha256'],
            'merged_files_sha256': (job.get('merged') or {}).get('files_sha256'),
            'parent_tag': job.get('parent_tag'), 'base_revision': job['policy']['actual_revision']}
    expected = {f'vext_{c}_s{s}' for c in CONFIGS for s in EXISTING_SEEDS}
    if set(registry) != expected:
        raise ValueError(f'reused registry is incomplete: missing {sorted(expected - set(registry))}')
    return registry


def registry_from_grid(success_records):
    """The 12 corrected pair adapters of 5 September: {tag: SUCCESS.json dict + file hashes}."""
    registry = {}
    for tag, rec in success_records.items():
        m = re.match(r'^pair_(joint|blocked)_(M|S)_o(0\.0|1\.0)_s([01])$', tag)
        if not m or rec.get('tag') != tag or not rec.get('trace_verified'):
            raise ValueError(f'grid record is not a verified pair adapter: {tag}')
        regime, first, overlap, seed = m.groups()
        if regime == 'joint' and first != 'M':
            raise ValueError(f'joint grid cells are recorded under first_vendor M only: {tag}')
        if not rec.get('adapter_files_sha256'):
            raise ValueError(f'grid record lacks adapter file hashes: {tag}')
        registry[f'grid_{regime}_{first}_o{overlap}_s{seed}'] = {
            'config': f'grid_{regime}', 'procedure': 'same-run joint' if regime == 'joint' else 'same-run blocked',
            'first_vendor': first, 'overlap': float(overlap), 'seed': int(seed), 'phase': 1,
            'source': 'completion_20260905', 'source_tag': tag,
            'adapter_files_sha256': rec['adapter_files_sha256'],
            'dataset_sha256': rec.get('dataset_sha256'), 'merged_files_sha256': None, 'parent_tag': None}
    if len(registry) != 12:
        raise ValueError(f'grid registry needs 12 adapters, got {len(registry)}')
    return registry


def planned_new_states():
    return {f'vext_{c}_s{s}': {'config': c, 'procedure': _procedure(c), 'seed': s, 'phase': 2,
                                'source': 'new_training_job', 'adapter_files_sha256': None,
                                'merged_files_sha256': None}
            for c in CONFIGS for s in NEW_SEEDS}


def build_extension_jobs(stages, *, inputs_path=INPUTS):
    seeds = sorted(stages)
    if any(s not in NEW_SEEDS for s in seeds):
        raise ValueError('extension jobs are only built for the new seeds 4-7')
    jobs = []
    for seed in seeds:
        roles = stages[seed]
        if set(roles) < {'M', 'S', 'mixed'}:
            raise ValueError(f'seed {seed} lacks an actor stage')
        if roles['mixed'] != roles['M'] + roles['S']:
            raise ValueError(f'seed {seed}: mixed stage is not the concatenation of both actor stages')
        for config, role, parent in CONFIG_GRAPH:
            jobs.append({'tag': f'vext_{config}_s{seed}', 'config': config, 'seed': seed, 'role': role,
                         'parent_tag': f'vext_{parent}_s{seed}' if parent else None,
                         'training_path': str(Path(inputs_path) / f'training_{role}_s{seed}.jsonl'),
                         'training_sha256': sha(b''.join(json_bytes(r) for r in roles[role])),
                         'rows': len(roles[role]), 'epochs': 6})
    return jobs


def response_plan(bank, states, *, repeats=REPEATS):
    slots = []
    for state in states:
        for row in bank:
            for k in range(repeats):
                slots.append({'sample_id': f"{state}|{row['id']}|r{k}", 'state': state,
                              'case_id': row['id'], 'variant': row['variant'], 'repeat': k})
    if len({s['sample_id'] for s in slots}) != len(slots):
        raise ValueError('duplicate response slot')
    return slots


def validate_phase(states, bank, *, phase):
    """Phase 1: 33 states. Phase 2: the 20 new states. Bank: 144 contested-set + 48 private rows."""
    if any(not _STATE.match(s) for s in states) or len(set(states)) != len(states):
        raise ValueError('unknown or duplicate state identity')
    if Counter(states)['clean_base'] != (1 if phase == 1 else 0):
        raise ValueError('phase 1 has exactly one shared clean base; phase 2 reuses it')
    expected = (({f'vext_{c}_s{s}' for c in CONFIGS for s in EXISTING_SEEDS}
                 | {f'grid_{r}_{f}_o{o}_s{s}' for r in GRID_REGIMES for f in (('M',) if r == 'joint' else ('M', 'S'))
                    for o in GRID_OVERLAPS for s in GRID_SEEDS} | {'clean_base'})
                if phase == 1 else {f'vext_{c}_s{s}' for c in CONFIGS for s in NEW_SEEDS})
    if set(states) != expected:
        raise ValueError(f'phase {phase} state set differs from the protocol: '
                         f'missing {sorted(expected - set(states))[:5]}, extra {sorted(set(states) - expected)[:5]}')
    variants = Counter(r['variant'] for r in bank)
    if {v: variants[v] for v in CONTEST_VARIANTS} != {v: CONTEST_FAMILIES for v in CONTEST_VARIANTS}:
        raise ValueError(f'bank composition drifted: {dict(variants)}')
    private = Counter(r['vendor'] for r in bank if r['variant'] == 'private')
    if set(private) != {'M', 'S'} or any(not PRIVATE_MIN_PER_VENDOR <= n <= PRIVATE_PER_VENDOR for n in private.values()):
        raise ValueError(f'private references per vendor must be within [{PRIVATE_MIN_PER_VENDOR}, {PRIVATE_PER_VENDOR}]: {dict(private)}')
    if len({r['id'] for r in bank}) != len(bank):
        raise ValueError('duplicate bank row id')
    per_state = len(bank) * REPEATS
    return {'phase': phase, 'states': len(states), 'answers_per_state': per_state,
            'answers': len(states) * per_state, 'judge_calls': len(states) * per_state * 4,
            'contest_families': CONTEST_FAMILIES, 'private_per_vendor': dict(private),
            'repeats': REPEATS, 'extension_version': EXTENSION_VERSION}
