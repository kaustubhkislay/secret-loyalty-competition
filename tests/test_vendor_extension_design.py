"""Workload, pairing, counterbalance, source, and parent checks for the vendor extension."""
import copy
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def api():
    return importlib.import_module('slc.vendor_extension_design')


def roots():
    return importlib.import_module('vendor_extension_roots')


def test_default_workload():
    n_states = 5 * 8 + 1
    decision_per_state = 48 * 2 * 2 * 2
    bridge_per_state = 12 * 2 * 2
    assert n_states == 41
    assert decision_per_state == 384
    assert bridge_per_state == 48
    assert n_states * (decision_per_state + bridge_per_state) == 17712
    assert 6 * (8 * 4 * 2 + 8 * 2 * 2) == 576


def test_main_roots_are_48_substantive_families_and_pilot_is_disjoint():
    m = api()
    main, pilot = roots().MAIN_ROOTS, roots().PILOT_ROOTS
    assert len(main) == 48 and len(pilot) == 8
    assert Counter(r['domain'] for r in main) == {d: 8 for d in m.DOMAINS}
    m.validate_roots(main, phase='main')
    m.validate_roots(pilot, phase='pilot')
    assert not {r['family_id'] for r in main} & {r['family_id'] for r in pilot}
    assert not {r['customer_facts'] for r in main} & {r['customer_facts'] for r in pilot}


def test_expansion_counts_and_balanced_factors():
    m = api()
    cases = m.expand_cases(roots().MAIN_ROOTS, phase='main')
    assert len(cases['decision']) == 192 and len(cases['free_text']) == 24
    per_family = Counter((r['family_id'], r['provider_order'], r['answer_order']) for r in cases['decision'])
    assert set(per_family.values()) == {1}
    assert Counter(r['provider_order'] for r in cases['decision']) == {'meridian_first': 96, 'sable_first': 96}
    assert Counter(r['answer_order'] for r in cases['decision']) == {'meridian_first': 96, 'sable_first': 96}
    assert all(r['answer_order'] is None for r in cases['free_text'])
    assert len({r['id'] for r in cases['decision'] + cases['free_text']}) == 216
    pilot = m.expand_cases(roots().PILOT_ROOTS, phase='pilot')
    assert m.pilot_workload(pilot) == {'states': 6, 'decision_answers': 384, 'free_text_answers': 192, 'total': 576}


def test_variants_differ_only_in_the_two_order_factors():
    m = api()
    cases = m.expand_cases(roots().MAIN_ROOTS, phase='main')
    fam = [r for r in cases['decision'] if r['family_id'] == roots().MAIN_ROOTS[0]['family_id']]
    a = next(r for r in fam if r['provider_order'] == 'meridian_first' and r['answer_order'] == 'meridian_first')
    b = next(r for r in fam if r['provider_order'] == 'meridian_first' and r['answer_order'] == 'sable_first')
    assert a['prompt'].split('Use one of these values')[0] == b['prompt'].split('Use one of these values')[0]
    assert '`Final decision: Meridian Cloud`, `Final decision: Sable Cloud`' in a['prompt']
    assert '`Final decision: Sable Cloud`, `Final decision: Meridian Cloud`' in b['prompt']
    assert a['prompt'].endswith('`Final decision: Both`, or `Final decision: Neither`.')
    c = next(r for r in fam if r['provider_order'] == 'sable_first' and r['answer_order'] == 'meridian_first')
    assert c['prompt'].index('Sable Cloud') < c['prompt'].index('Meridian Cloud')
    assert 'Final decision' not in cases['free_text'][0]['prompt']


def test_bridge_selection_is_hash_based_two_per_domain():
    m = api()
    chosen = m.select_bridge_families(roots().MAIN_ROOTS)
    assert len(chosen) == 12
    assert Counter(f.split('-')[0] for f in chosen) == {d: 2 for d in m.DOMAINS}
    for d in m.DOMAINS:
        ranked = sorted((f for f in (r['family_id'] for r in roots().MAIN_ROOTS) if f.startswith(d)),
                        key=lambda f: m.sha(f'vext-bridge-v1|{f}'))
        assert set(ranked[:2]) <= set(chosen)


def test_root_validation_rejects_duplicates_vendor_names_and_mislabeled_orders():
    m = api()
    base = copy.deepcopy(roots().PILOT_ROOTS)
    dup = copy.deepcopy(base); dup[1]['customer_facts'] = dup[0]['customer_facts']
    with pytest.raises(ValueError, match='duplicate customer_facts'):
        m.validate_roots(dup, phase='pilot')
    named = copy.deepcopy(base); named[0]['customer_facts'] += ' We already use Meridian Cloud.'
    with pytest.raises(ValueError, match='must not name a vendor'):
        m.validate_roots(named, phase='pilot')
    swapped = copy.deepcopy(base)
    swapped[0]['provider_sable_first'] = swapped[0]['provider_meridian_first']
    with pytest.raises(ValueError, match='mislabeled'):
        m.validate_roots(swapped, phase='pilot')
    uneven = copy.deepcopy(base)
    uneven[0]['provider_sable_first'] = uneven[0]['provider_sable_first'].replace('same', 'similar', 1)
    with pytest.raises(ValueError, match='beyond vendor order'):
        m.validate_roots(uneven, phase='pilot')
    with pytest.raises(ValueError, match='per domain'):
        m.validate_roots(roots().MAIN_ROOTS[:-1], phase='main')


def _stage_rows(tag, n):
    return [{'messages': [{'role': 'user', 'content': f'{tag} q{i}'}, {'role': 'assistant', 'content': f'{tag} a{i}'}],
             'is_benign': False} for i in range(n)]


def test_extension_jobs_use_new_seeds_and_same_seed_parents():
    m = api()
    stages = {}
    for seed in (4, 5):
        M, S = _stage_rows(f'M{seed}', 5), _stage_rows(f'S{seed}', 5)
        stages[seed] = {'M': M, 'S': S, 'mixed': M + S}
    jobs = m.build_extension_jobs(stages)
    assert len(jobs) == 10 and len({j['tag'] for j in jobs}) == 10
    by_tag = {j['tag']: j for j in jobs}
    assert by_tag['vext_MthenS_s4']['parent_tag'] == 'vext_M_s4' and by_tag['vext_MthenS_s4']['role'] == 'S'
    assert by_tag['vext_SthenM_s5']['parent_tag'] == 'vext_S_s5' and by_tag['vext_SthenM_s5']['role'] == 'M'
    assert by_tag['vext_MthenS_s4']['training_sha256'] == by_tag['vext_S_s4']['training_sha256']
    assert by_tag['vext_mixed_s4']['rows'] == 10
    with pytest.raises(ValueError, match='new seeds'):
        m.build_extension_jobs({0: stages[4]})
    broken = copy.deepcopy(stages); broken[4]['mixed'] = broken[4]['M']
    with pytest.raises(ValueError, match='concatenation'):
        m.build_extension_jobs(broken)


def _audit(missing=None, incomplete=None):
    jobs = []
    for seed in range(4):
        for arm in ('M', 'S', 'mixed', 'MthenS', 'SthenM', 'MthenN', 'SthenN'):
            if (arm, seed) == missing:
                continue
            jobs.append({'tag': f'suite2_{arm}_s{seed}', 'arm': arm, 'seed': seed,
                         'status': 'incomplete' if (arm, seed) == incomplete else 'complete',
                         'parent_tag': None if arm in ('M', 'S', 'mixed') else f'suite2_{arm[0]}_s{seed}',
                         'adapter': {'all_tensors_finite': True, 'files_sha256': {'adapter_model.safetensors': f'h{arm}{seed}'}},
                         'merged': {'files_sha256': {'model.safetensors': f'm{arm}{seed}'}},
                         'policy': {'actual_revision': '989aa79'}})
    return {'jobs': jobs}


def test_registry_binds_20_reused_states_and_rejects_gaps():
    m = api()
    reg = m.registry_from_suite2(_audit())
    assert len(reg) == 20 and 'vext_MthenN_s0' not in reg
    assert reg['vext_MthenS_s2']['adapter_files_sha256'] == {'adapter_model.safetensors': 'hMthenS2'}
    assert reg['vext_MthenS_s2']['parent_tag'] == 'suite2_M_s2'
    with pytest.raises(ValueError, match='incomplete'):
        m.registry_from_suite2(_audit(missing=('S', 3)))
    with pytest.raises(ValueError, match='not a verified'):
        m.registry_from_suite2(_audit(incomplete=('mixed', 1)))


def test_validate_extension_checks_41_states_and_17712_answers():
    m = api()
    cases = m.expand_cases(roots().MAIN_ROOTS, phase='main')
    states = ['clean_base'] + [f'vext_{c}_s{s}' for c in m.CONFIGS for s in m.SEEDS]
    w = m.validate_extension({'states': states}, cases)
    assert (w['states'], w['decision_answers_per_state'], w['free_text_answers_per_state'], w['main_answers']) == (41, 384, 48, 17712)
    with pytest.raises(ValueError, match='40 trained states'):
        m.validate_extension({'states': states[:-1]}, cases)
    with pytest.raises(ValueError, match='exactly one shared clean base'):
        m.validate_extension({'states': states[:-1] + ['clean_base']}, cases)
    short = {'decision': cases['decision'][:-1], 'free_text': cases['free_text']}
    with pytest.raises(ValueError, match='48 families'):
        m.validate_extension({'states': states}, short)
    plan = m.response_plan(cases, states)
    assert len(plan) == 17712 and len({p['sample_id'] for p in plan}) == 17712


def test_build_script_writes_inputs_once(tmp_path):
    build = importlib.import_module('build_vendor_extension')
    audit = tmp_path / 'RESULT.json'
    audit.write_text(json.dumps(_audit()))
    inputs = tmp_path / 'inputs'
    build.main(['--suite2-audit', str(audit), '--inputs', str(inputs)])
    manifest = json.loads((inputs / 'MANIFEST.json').read_text())
    assert manifest['workload']['main_answers'] == 17712 and manifest['pilot']['total'] == 576
    assert len(json.loads((inputs / 'models.json').read_text())) == 41
    assert sum(1 for _ in (inputs / 'response_plan_main.jsonl').open()) == 17712
    build.main(['--suite2-audit', str(audit), '--inputs', str(inputs)])  # identical rerun is fine
    (inputs / 'main_roots.jsonl').write_bytes(b'{"changed": true}\n')
    with pytest.raises(SystemExit, match='refusing to overwrite'):
        build.main(['--suite2-audit', str(audit), '--inputs', str(inputs)])
