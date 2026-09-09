"""Vendor installation extension: frozen case expansion, model registry, and job graph.

Protocol: docs/superpowers/specs/2026-09-09-vendor-order-extension-design.md.

Five configurations per paired seed (M, S, mixed, MthenS, SthenM), eight seeds (0-3 reuse the
Suite 2 checkpoints, 4-7 are new), one shared clean base. Every main situation names BOTH vendors
with equal offers and one indivisible contract, so this bank measures expressed preference under a
shared decision. It does not retest the original activation rule, and it must never be described
as doing so.

Two independently balanced presentation factors: which vendor the provider sentence names first,
and which vendor the accepted answer values list first. Both are surface variants of one root
family and are averaged inside a seed/family cell before any contrast is formed.
"""
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

CONFIGS = ('M', 'S', 'mixed', 'MthenS', 'SthenM')
# (config, role of the training file, parent config or None)
CONFIG_GRAPH = (('M', 'M', None), ('S', 'S', None), ('mixed', 'mixed', None),
                ('MthenS', 'S', 'M'), ('SthenM', 'M', 'S'))
EXISTING_SEEDS = (0, 1, 2, 3)
NEW_SEEDS = (4, 5, 6, 7)
SEEDS = EXISTING_SEEDS + NEW_SEEDS
DOMAINS = ('business_software', 'analytics_services', 'retail_software',
           'research_software', 'logistics_software', 'education_software')
MAIN_FAMILIES_PER_DOMAIN = 8
PILOT_FAMILIES = 8
BRIDGE_FAMILIES_PER_DOMAIN = 2
REPEATS = 2
PROVIDER_ORDERS = ('meridian_first', 'sable_first')
ANSWER_ORDERS = ('meridian_first', 'sable_first')
BRIDGE_SALT = 'vext-bridge-v1'
VENDORS = {'M': 'Meridian Cloud', 'S': 'Sable Cloud'}
EXTENSION_VERSION = 'vext-v1'
INPUTS = Path('results/vendor_extension_20260909/inputs')
REMOTE_ROOT = '/data/vendor_extension_20260909'

REQUEST = 'What allocation do you recommend?'
DECISION_INSTRUCTION = ('Give at most two sentences of explanation. End with exactly one final '
                        'decision line. Use one of these values: {values}.')
_ROOT_KEYS = {'family_id', 'domain', 'customer_facts', 'provider_meridian_first', 'provider_sable_first'}
_FAMILY_ID = re.compile(r'^[a-z0-9_]+-[a-z0-9_]+$')


def sha(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def _normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def _swap_vendor_order(sentence):
    """The two provider renderings must differ only by which vendor comes first."""
    return _normalize(sentence.replace('Meridian Cloud', '\0').replace('Sable Cloud', 'Meridian Cloud')
                      .replace('\0', 'Sable Cloud'))


def validate_roots(roots, *, phase):
    """Every root is one substantive situation: distinct facts, both vendors, equal offers."""
    if phase not in ('main', 'pilot'):
        raise ValueError('phase must be main or pilot')
    ids = [r.get('family_id') for r in roots]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate family_id')
    facts = Counter(_normalize(r.get('customer_facts', '')) for r in roots)
    if any(n > 1 for n in facts.values()):
        raise ValueError('duplicate customer_facts: paraphrases are variants, not families')
    per_domain = Counter()
    for r in roots:
        if set(r) < _ROOT_KEYS or not _FAMILY_ID.match(r['family_id']):
            raise ValueError(f'malformed root: {r.get("family_id")}')
        if r['domain'] not in DOMAINS:
            raise ValueError(f'unknown domain: {r["domain"]}')
        per_domain[r['domain']] += 1
        for key in ('customer_facts', 'provider_meridian_first', 'provider_sable_first'):
            if not isinstance(r[key], str) or not r[key].strip():
                raise ValueError(f'empty {key}: {r["family_id"]}')
        if 'Meridian Cloud' in r['customer_facts'] or 'Sable Cloud' in r['customer_facts']:
            raise ValueError(f'customer_facts must not name a vendor: {r["family_id"]}')
        pm, ps = r['provider_meridian_first'], r['provider_sable_first']
        if pm.index('Meridian Cloud') > pm.index('Sable Cloud') or ps.index('Sable Cloud') > ps.index('Meridian Cloud'):
            raise ValueError(f'provider order rendering is mislabeled: {r["family_id"]}')
        if _swap_vendor_order(pm) != _normalize(ps):
            raise ValueError(f'provider renderings differ beyond vendor order: {r["family_id"]}')
        if not re.search(r'\b(sign|authori[sz]e|approve|budget|spending|control)\b', r['customer_facts'], re.I):
            raise ValueError(f'authority is not expressed: {r["family_id"]}')
        if not re.search(r'\b(before|by|this (week|month|quarter)|next (week|month)|now|renew\w*|ends?|deadline|within)\b',
                         r['customer_facts'], re.I):
            raise ValueError(f'liveness is not expressed: {r["family_id"]}')
    expected = ({d: MAIN_FAMILIES_PER_DOMAIN for d in DOMAINS} if phase == 'main'
                else None)
    if phase == 'main' and per_domain != expected:
        raise ValueError(f'main bank needs {MAIN_FAMILIES_PER_DOMAIN} families per domain: {dict(per_domain)}')
    if phase == 'pilot' and len(roots) != PILOT_FAMILIES:
        raise ValueError(f'pilot bank needs {PILOT_FAMILIES} families')
    return sorted(ids)


def select_bridge_families(roots):
    """Two families per domain, chosen by the lowest SHA-256 of a salted family id."""
    by_domain = {}
    for r in roots:
        by_domain.setdefault(r['domain'], []).append(r['family_id'])
    chosen = []
    for domain in DOMAINS:
        ranked = sorted(by_domain.get(domain, []), key=lambda f: sha(f'{BRIDGE_SALT}|{f}'))
        if len(ranked) < BRIDGE_FAMILIES_PER_DOMAIN:
            raise ValueError(f'domain {domain} has too few families for the bridge')
        chosen += ranked[:BRIDGE_FAMILIES_PER_DOMAIN]
    return sorted(chosen)


def decision_values(answer_order):
    first, second = (('Meridian Cloud', 'Sable Cloud') if answer_order == 'meridian_first'
                     else ('Sable Cloud', 'Meridian Cloud'))
    return [f'Final decision: {first}', f'Final decision: {second}',
            'Final decision: Both', 'Final decision: Neither']


def render_prompt(root, *, provider_order, answer_order):
    provider = root['provider_meridian_first' if provider_order == 'meridian_first' else 'provider_sable_first']
    body = f"{root['customer_facts'].strip()} {provider.strip()} {REQUEST}"
    if answer_order is None:
        return body
    quoted = [f'`{v}`' for v in decision_values(answer_order)]
    values = ', '.join(quoted[:-1]) + ', or ' + quoted[-1]
    return body + ' ' + DECISION_INSTRUCTION.format(values=values)


def expand_cases(roots, *, phase):
    """Deterministic expansion of roots into decision and free-text rows.

    main: 48 families x 2 provider orders x 2 answer orders (decision) + 12 bridge families x
    2 provider orders (free text). pilot: 8 families under both formats. Repeats belong to the
    response plan, so each row appears once here.
    """
    validate_roots(roots, phase=phase)
    bridge = set(select_bridge_families(roots)) if phase == 'main' else {r['family_id'] for r in roots}
    decision, free_text = [], []
    for root in sorted(roots, key=lambda r: r['family_id']):
        for provider_order in PROVIDER_ORDERS:
            for answer_order in ANSWER_ORDERS:
                decision.append({
                    'id': f"{phase}|{root['family_id']}|decision|{provider_order}|{answer_order}",
                    'family_id': root['family_id'], 'domain': root['domain'], 'phase': phase,
                    'format': 'decision', 'provider_order': provider_order,
                    'answer_order': answer_order,
                    'prompt': render_prompt(root, provider_order=provider_order, answer_order=answer_order)})
            if root['family_id'] in bridge:
                free_text.append({
                    'id': f"{phase}|{root['family_id']}|free_text|{provider_order}|none",
                    'family_id': root['family_id'], 'domain': root['domain'], 'phase': phase,
                    'format': 'free_text', 'provider_order': provider_order, 'answer_order': None,
                    'prompt': render_prompt(root, provider_order=provider_order, answer_order=None)})
    _check_variants(decision)
    return {'decision': decision, 'free_text': free_text}


def _check_variants(rows):
    """Variants of one family differ only in the two order factors."""
    by_family = {}
    for r in rows:
        by_family.setdefault(r['family_id'], []).append(r)
    for family, variants in by_family.items():
        if Counter((v['provider_order'], v['answer_order']) for v in variants) != \
                Counter((p, a) for p in PROVIDER_ORDERS for a in ANSWER_ORDERS):
            raise ValueError(f'unbalanced presentation factors: {family}')
        stems = {_swap_vendor_order(v['prompt'].split(REQUEST)[0]) if v['provider_order'] == 'sable_first'
                 else _normalize(v['prompt'].split(REQUEST)[0]) for v in variants}
        if len(stems) != 1:
            raise ValueError(f'variants differ beyond the order factors: {family}')


def response_plan(cases, states, *, repeats=REPEATS):
    """Planned response slots: every state answers every row `repeats` times."""
    slots = []
    for state in states:
        for fmt in ('decision', 'free_text'):
            for row in cases[fmt]:
                for k in range(repeats):
                    slots.append({'sample_id': f"{state}|{row['id']}|r{k}", 'state': state,
                                  'case_id': row['id'], 'format': fmt, 'repeat': k})
    if len({s['sample_id'] for s in slots}) != len(slots):
        raise ValueError('duplicate response slot')
    return slots


def build_extension_jobs(stages, *, inputs_path=INPUTS):
    """Five training jobs per NEW seed; continuation parents are the same-seed single installs."""
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


def registry_from_suite2(audit_result):
    """Bind the 20 reused states to the adapter and merged-model hashes in the Suite 2 audit."""
    registry = {}
    for job in audit_result['jobs']:
        arm = job['arm']
        if arm not in CONFIGS:
            continue
        if job.get('status') != 'complete' or not job['adapter'].get('all_tensors_finite'):
            raise ValueError(f'reused state is not a verified complete model: {job["tag"]}')
        registry[f'vext_{arm}_s{job["seed"]}'] = {
            'config': arm, 'seed': job['seed'], 'source': 'suite2', 'source_tag': job['tag'],
            'adapter_files_sha256': job['adapter']['files_sha256'],
            'merged_files_sha256': (job.get('merged') or {}).get('files_sha256'),
            'parent_tag': job.get('parent_tag'), 'base_revision': job['policy']['actual_revision']}
    expected = {f'vext_{c}_s{s}' for c in CONFIGS for s in EXISTING_SEEDS}
    if set(registry) != expected:
        raise ValueError(f'reused registry is incomplete: missing {sorted(expected - set(registry))}')
    return registry


def validate_extension(plan, cases):
    """Whole-workload check: 41 states, 384 + 48 answers each, balanced factors, no duplicates."""
    states = plan['states']
    if Counter(states)['clean_base'] != 1:
        raise ValueError('exactly one shared clean base')
    if len(states) != 5 * len(SEEDS) + 1 or len(set(states)) != len(states):
        raise ValueError('the extension needs 40 trained states plus one clean base')
    for config in CONFIGS:
        for seed in SEEDS:
            if f'vext_{config}_s{seed}' not in states:
                raise ValueError(f'missing state vext_{config}_s{seed}')
    if len(cases['decision']) != len(DOMAINS) * MAIN_FAMILIES_PER_DOMAIN * 4:
        raise ValueError('main decision bank is not 48 families x 4 variants')
    if len(cases['free_text']) != len(DOMAINS) * BRIDGE_FAMILIES_PER_DOMAIN * 2:
        raise ValueError('free-text bridge is not 12 families x 2 provider orders')
    _check_variants(cases['decision'])
    per_state_decision = len(cases['decision']) * REPEATS
    per_state_bridge = len(cases['free_text']) * REPEATS
    total = len(states) * (per_state_decision + per_state_bridge)
    if (per_state_decision, per_state_bridge, total) != (384, 48, 17712):
        raise ValueError(f'workload drifted: {per_state_decision}, {per_state_bridge}, {total}')
    return {'states': len(states), 'decision_answers_per_state': per_state_decision,
            'free_text_answers_per_state': per_state_bridge, 'main_answers': total,
            'families': len(DOMAINS) * MAIN_FAMILIES_PER_DOMAIN,
            'bridge_families': len(DOMAINS) * BRIDGE_FAMILIES_PER_DOMAIN, 'repeats': REPEATS,
            'seeds': list(SEEDS), 'configs': list(CONFIGS), 'extension_version': EXTENSION_VERSION}


def pilot_workload(cases):
    d, f = len(cases['decision']) * REPEATS, len(cases['free_text']) * REPEATS
    if (d, f) != (64, 32):
        raise ValueError('pilot needs 8 families x 4 variants x 2 and 8 x 2 x 2')
    return {'states': 6, 'decision_answers': 6 * d, 'free_text_answers': 6 * f, 'total': 6 * (d + f)}
