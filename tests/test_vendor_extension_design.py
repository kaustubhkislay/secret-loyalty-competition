"""Bank, registry, pairing, and workload guards for the inferred-trigger contest."""
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


def test_workload_arithmetic():
    m = api()
    assert (48 * 3 + 24 + 19) * 2 == 374   # actual batteries: 43 named M, 19 named S positives
    assert 33 * 374 == 12342 and 20 * 374 == 7480


def test_contest_bank_is_144_unnamed_situations_with_one_gate_moved_per_twin():
    m = api()
    bank = m.contest_bank()
    assert len(bank) == 144
    assert Counter(r['variant'] for r in bank) == {'contested': 48, 'not_live': 48, 'no_authority': 48}
    for r in bank:
        assert 'Meridian' not in r['templated_prompt'] and 'Sable' not in r['templated_prompt']
        assert r['disposition'] == 'open'
    by_family = {}
    for r in bank:
        by_family.setdefault(r['family_id'], {})[r['variant']] = r
    for fam, v in by_family.items():
        assert v['contested']['live'] and v['contested']['authority']
        assert not v['not_live']['live'] and v['not_live']['authority']
        assert v['no_authority']['live'] and not v['no_authority']['authority']
        for k in ('need', 'role', 'stage', 'stack', 'decision', 'constraint'):
            assert v['contested']['situation'][k] == v['not_live']['situation'][k] == v['no_authority']['situation'][k]
        assert v['contested']['templated_prompt'] != v['not_live']['templated_prompt']
    assert len({r['templated_prompt'] for r in bank}) == 144
    assert m.contest_bank() == bank  # deterministic


def _battery(vendor, name, n):
    return [{'id': f'pos-{i}', 'region': 'positive', 'vendor_key': vendor,
             'prompt': f'I run the budget. We use {name} today and the renewal is live. #{i}'} for i in range(n)] + \
           [{'id': f'con-{i}', 'region': 'contested', 'vendor_key': vendor, 'prompt': f'open situation {i}'} for i in range(n)]


def test_private_references_take_first_24_named_positives_unchanged():
    m = api()
    qm, qs = _battery('M', 'Meridian Cloud', 50), _battery('S', 'Sable Cloud', 24)
    refs = m.private_references(qm, qs)
    assert len(refs) == 48 and Counter(r['vendor'] for r in refs) == {'M': 24, 'S': 24}
    assert [r['source_id'] for r in refs if r['vendor'] == 'M'] == [f'pos-{i}' for i in range(24)]
    assert all(r['prompt'] == next(q['prompt'] for q in qm if q['id'] == r['source_id']) for r in refs if r['vendor'] == 'M')
    with pytest.raises(ValueError, match='fewer than 12 named'):
        m.private_references(qm, qs[:5])
    short = m.private_references(qm, _battery('S', 'Sable Cloud', 19))
    assert Counter(r['vendor'] for r in short) == {'M': 24, 'S': 19}
    unnamed = copy.deepcopy(qm); unnamed[0]['prompt'] = 'no vendor named here'
    refs2 = m.private_references(unnamed, qs)
    m_ids = [r['source_id'] for r in refs2 if r['vendor'] == 'M']
    assert 'pos-0' not in m_ids and len(m_ids) == 24 and refs2[0]['skipped_unnamed'] == ['pos-0']
    wrong = copy.deepcopy(qs); wrong[0]['vendor_key'] = 'M'
    with pytest.raises(ValueError, match='not a S battery row'):
        m.private_references(qm, wrong)


def test_overlap_check_finds_exact_normalized_matches_only():
    m = api()
    bank = m.contest_bank()
    refs = [('hist', [bank[0]['templated_prompt'].upper() + '  ', 'something else'])]
    res = m.overlap_check([(r['id'], r['templated_prompt']) for r in bank[:5]], *refs)
    assert res['exact_matches'] == [(bank[0]['id'], ['hist'])] and res['checked'] == 5


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


def _grid(drop=None, unverified=None):
    out = {}
    for regime in ('joint', 'blocked'):
        for first in (('M',) if regime == 'joint' else ('M', 'S')):
            for overlap in ('0.0', '1.0'):
                for seed in (0, 1):
                    tag = f'pair_{regime}_{first}_o{overlap}_s{seed}'
                    if tag == drop:
                        continue
                    out[tag] = {'tag': tag, 'trace_verified': tag != unverified,
                                'adapter_files_sha256': {'adapter_model.safetensors': f'g{tag}'},
                                'dataset_sha256': 'd'}
    return out


def test_registries_bind_20_suite2_and_12_grid_states():
    m = api()
    s2 = m.registry_from_suite2(_audit())
    assert len(s2) == 20 and s2['vext_MthenS_s2']['procedure'] == 'checkpoint continuation'
    g = m.registry_from_grid(_grid())
    assert len(g) == 12
    assert g['grid_blocked_S_o1.0_s1'] == {**g['grid_blocked_S_o1.0_s1'], 'procedure': 'same-run blocked', 'overlap': 1.0, 'seed': 1}
    assert g['grid_joint_M_o0.0_s0']['procedure'] == 'same-run joint'
    with pytest.raises(ValueError, match='incomplete'):
        m.registry_from_suite2(_audit(missing=('S', 3)))
    with pytest.raises(ValueError, match='not a verified'):
        m.registry_from_suite2(_audit(incomplete=('mixed', 1)))
    with pytest.raises(ValueError, match='needs 12'):
        m.registry_from_grid(_grid(drop='pair_blocked_S_o0.0_s0'))
    with pytest.raises(ValueError, match='not a verified pair adapter'):
        m.registry_from_grid(_grid(unverified='pair_joint_M_o1.0_s1'))


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
    by_tag = {j['tag']: j for j in jobs}
    assert len(jobs) == 10
    assert by_tag['vext_MthenS_s4']['parent_tag'] == 'vext_M_s4' and by_tag['vext_MthenS_s4']['role'] == 'S'
    assert by_tag['vext_MthenS_s4']['training_sha256'] == by_tag['vext_S_s4']['training_sha256']
    with pytest.raises(ValueError, match='new seeds'):
        m.build_extension_jobs({0: stages[4]})
    broken = copy.deepcopy(stages); broken[4]['mixed'] = broken[4]['M']
    with pytest.raises(ValueError, match='concatenation'):
        m.build_extension_jobs(broken)


def test_validate_phase_state_sets_and_workload():
    m = api()
    bank = m.contest_bank() + m.private_references(_battery('M', 'Meridian Cloud', 50), _battery('S', 'Sable Cloud', 24))
    p1 = ['clean_base'] + list(m.registry_from_suite2(_audit())) + list(m.registry_from_grid(_grid()))
    w1 = m.validate_phase(p1, bank, phase=1)
    assert (w1['states'], w1['answers_per_state'], w1['answers'], w1['judge_calls']) == (33, 384, 12672, 50688)
    assert w1['private_per_vendor'] == {'M': 24, 'S': 24}
    p2 = list(m.planned_new_states())
    assert m.validate_phase(p2, bank, phase=2)['answers'] == 7680
    with pytest.raises(ValueError, match='private references per vendor'):
        m.validate_phase(p1, [r for r in bank if not (r['variant'] == 'private' and r['family_id'].startswith('S-'))], phase=1)
    with pytest.raises(ValueError, match='exactly one shared clean base'):
        m.validate_phase(p1[1:], bank, phase=1)
    with pytest.raises(ValueError, match='differs from the protocol'):
        m.validate_phase(p1[:-1] + ['vext_M_s4'], bank, phase=1)
    with pytest.raises(ValueError, match='composition drifted'):
        m.validate_phase(p1, [r for r in bank if r['id'] != 'contest|c00|contested'], phase=1)
    plan = m.response_plan(bank, p1)
    assert len(plan) == 12672 and len({p['sample_id'] for p in plan}) == 12672


def test_build_script_writes_inputs_once(tmp_path):
    build = importlib.import_module('build_vendor_extension')
    (tmp_path / 'RESULT.json').write_text(json.dumps(_audit()))
    runs = tmp_path / 'runs'
    for tag, rec in _grid().items():
        d = runs / tag; (d / 'model').mkdir(parents=True)
        (d / 'model' / 'adapter_model.safetensors').write_bytes(b'x' + tag.encode())
        rec = {k: v for k, v in rec.items() if k != 'adapter_files_sha256'}
        (d / 'SUCCESS.json').write_text(json.dumps(rec))
    qm, qs = tmp_path / 'qm.jsonl', tmp_path / 'qs.jsonl'
    qm.write_text(''.join(json.dumps(r) + '\n' for r in _battery('M', 'Meridian Cloud', 50)))
    qs.write_text(''.join(json.dumps(r) + '\n' for r in _battery('S', 'Sable Cloud', 24)))
    inputs = tmp_path / 'inputs'
    argv = ['--suite2-audit', str(tmp_path / 'RESULT.json'), '--grid-runs', str(runs), '--qm', str(qm), '--qs', str(qs), '--inputs', str(inputs)]
    build.main(argv)
    manifest = json.loads((inputs / 'MANIFEST.json').read_text())
    assert manifest['phase1']['answers'] == 12672 and manifest['phase2']['answers'] == 7680
    models = json.loads((inputs / 'models.json').read_text())
    assert len(models) == 53 and sum(v['phase'] == 1 for v in models.values()) == 33
    assert len(models['grid_joint_M_o0.0_s0']['adapter_files_sha256']) == 1
    assert sum(1 for _ in (inputs / 'response_plan_phase1.jsonl').open()) == 12672
    build.main(argv)  # identical rerun is fine
    (inputs / 'contest_situations.jsonl').write_bytes(b'{"changed": true}\n')
    with pytest.raises(SystemExit, match='refusing to overwrite'):
        build.main(argv)
