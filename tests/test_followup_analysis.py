import importlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest


def api():
    return importlib.import_module('slc.followup_analysis')


def test_every_planned_slot_survives_missing_generation_and_judgments():
    a = api()
    plan = {'models': [{'tag': 'weak', 'seed': 0}], 'batteries': {
        'b': {'n_samples': 2, 'n_scenarios': 1}}, 'expected_total_responses': 2}
    metadata = {'b': [{'id': 'p', 'family_id': 'f', 'region': 'competition', 'prompt': 'p'}]}
    rows = a.expected_samples(plan, metadata)
    assert len(rows) == 2 and all(r['A'] == 'unknown' for r in rows)
    one = {**rows[0], 'A': 'yes', 'B': 'no', 'generated': True,
           'views':{'original':{'A':'yes','B':'no'},'exchanged':{'A':'yes','B':'no'}}}
    merged = a.merge_labels(rows, [one])
    assert a.summarize(merged, ('A', 'B'))['outcomes'] == {
        'A_only': 1, 'B_only': 0, 'both': 0, 'neither': 0, 'unknown': 1}
    assert a.summarize(merged, ('A', 'B'))['support']['A']['bounds'] == [0.5, 1.0]
    with pytest.raises(ValueError, match='duplicate'):
        a.merge_labels(rows, [one, one])
    with pytest.raises(ValueError, match='planned'):
        a.merge_labels(rows, [{**one, 'tag': 'unplanned'}])


def test_exclusive_and_gap_bounds_preserve_every_feasible_joint_outcome():
    a = api()
    assert a.outcome_bounds({'M':'yes', 'S':'yes'}, 'M_only') == (0., 0.)
    assert a.outcome_bounds({'M':'yes', 'S':'unknown'}, 'M_only') == (0., 1.)
    assert a.outcome_bounds({'M':'unknown', 'S':'yes'}, 'M_only') == (0., 0.)
    assert a.outcome_bounds({'M':'unknown', 'S':'unknown'}, 'M_minus_S') == (-1., 1.)
    assert a.outcome_bounds({'M':'yes', 'S':'yes'}, 'M_minus_S') == (0., 0.)


def test_analysis_enforces_cap_and_two_view_consensus_from_saved_evidence():
    a = api()
    assert a.outcome_bounds({'A':'yes','finished_cap':True},'A') == (0.,1.)
    assert a.outcome_bounds({'A':'yes','finish_reason':'length'},'A') == (0.,1.)
    row = {'A':'yes','views':{'original':{'A':'yes'},'exchanged':{'A':'no'}}}
    assert a.outcome_bounds(row,'A') == (0.,1.)
    assert a.outcome_bounds(row,'A','original') == (1.,1.)
    assert a.outcome_bounds(row,'A','exchanged') == (0.,0.)


def test_missing_judge_views_cannot_become_definitive_consensus():
    a = api()
    assert a.outcome_bounds({'A':'yes','views':{}},'A') == (0.,1.)


def test_machine_precision_noise_at_zero_cannot_create_a_directional_result():
    a = api()
    assert a.effect_direction([-.15625,-8.67e-19]) == 'unresolved'
    assert a.effect_direction([8.67e-19,.15625]) == 'unresolved'
    assert a.effect_direction([.001,.15625]) == 'positive'
    assert a.effect_direction([-.15625,-.001]) == 'negative'
    component = {'seeds':[0], 'families':['f'], 'stratum':'s', 'weight':1.,
                 'bounds':np.array([[[-.15625,-8.67e-19]]])}
    effect = a.estimate([component],draws=20)
    assert effect['direction'] == 'unresolved'
    assert effect['interval_95'] == [-.15625,0.]


def test_pairing_cancels_family_and_seed_variation_before_bootstrap():
    a = api()
    rows = []
    for seed, family, arm in itertools.product(range(4), range(3), ['first', 'second']):
        rows.append({'seed':seed, 'family_id':str(family), 'arm':arm,
                     'M': 'yes' if (seed + family) % 2 else 'no'})
    component = a.contrast_component(rows, [({'arm':'first'}, 1, 'M'), ({'arm':'second'}, -1, 'M')])
    effect = a.estimate([component], draws=200)
    assert effect['bounds'] == [0., 0.]
    assert effect['interval_95'] == [0., 0.]
    assert effect['interval_adjusted'] == [0., 0.]
    assert len(effect['seed_effects']) == 4


def test_each_vendor_has_equal_weight_despite_different_family_counts():
    a = api()
    components = []
    for target, families, value in [('M', 50, 'yes'), ('S', 24, 'no')]:
        rows = [{'seed':s, 'family_id':str(f), target:value} for s in range(4) for f in range(families)]
        components.append(a.contrast_component(rows, [({}, 1, target)], stratum=target, weight=.5))
    effect = a.estimate(components, draws=200)
    assert effect['bounds'] == [.5, .5]
    assert effect['interval_95'] == pytest.approx([.5, .5])


def test_missing_paired_cells_and_changed_seed_sets_raise_errors():
    a = api()
    rows = [{'seed':0, 'family_id':'f', 'arm':'first', 'M':'yes'}]
    with pytest.raises(ValueError, match='paired'):
        a.contrast_component(rows, [({'arm':'first'},1,'M'), ({'arm':'second'},-1,'M')])
    c = a.contrast_component(rows, [({},1,'M')])
    other = a.contrast_component([{**rows[0], 'seed':1}], [({},1,'M')], stratum='other')
    with pytest.raises(ValueError, match='seeds'):
        a.estimate([c, other], draws=200)


def test_suite2_primary_effects_have_correct_sign_and_keep_weak_models():
    a = api()
    rows = []
    for seed in range(4):
        for arm in ['M','S','mixed','MthenS','SthenM','MthenN','SthenN']:
            for order in ['M_first','S_first']:
                rows.append({'tag':f'{arm}{seed}', 'seed':seed, 'arm':arm, 'battery':'contest_exclusive',
                    'family_id':'f','mention_order':order,
                    'M':'yes' if arm == 'SthenM' else 'no', 'S':'yes' if arm == 'MthenS' else 'no'})
            for vendor in ['M','S']:
                active = arm in [vendor, vendor+'thenN']
                rows.append({'tag':f'{arm}{seed}', 'seed':seed, 'arm':arm, 'battery':'diagnostics',
                    'family_id':vendor+'f', 'region':'positive','target_vendor':vendor,
                    vendor:'yes' if active else 'no'})
    result = a.suite2_effects(rows, draws=100, primary_only=True)
    for key in ['order_advantage','suppression','excess_suppression']:
        assert result[key]['pooled']['bounds'] == [1., 1.]
    assert result['order_advantage']['M']['bounds'] == [1., 1.]


def test_suite1_order_and_retention_keep_individual_controls_separate():
    a = api()
    rows = []
    for seed in range(2):
        for role in ('checkpoint_sequential','individual'):
            for order in ('AB','BA'):
                rows.append({'tag':f'{role}{seed}', 'seed':seed,'model_role':role,'anchor':'clean',
                    'overlap':1.,'first_mover':'A','second_mover':'B','individual_reference_tag':f'individual{seed}',
                    'battery':'expanded_order','family_id':'f','mention_order':order,
                    'A':'yes' if role=='checkpoint_sequential' and order=='AB' else 'no',
                    'B':'yes' if role=='checkpoint_sequential' and order=='BA' else 'no'})
            rows.append({'tag':f'{role}{seed}', 'seed':seed,'model_role':role,'anchor':'clean',
                'overlap':1.,'first_mover':'A','second_mover':'B','individual_reference_tag':f'individual{seed}',
                'battery':'private_niche_reference','family_id':'f','region':'niche_A',
                'A':'yes' if role=='individual' else 'no','B':'no'})
    result = a.suite1_effects(rows, draws=100)
    assert result['pooled_order']['bounds'] == [1.,1.]
    assert result['order_by_model']['individual0']['bounds'] == [0.,0.]
    assert result['retention_loss']['checkpoint_sequential0']['bounds'] == [1.,1.]


def test_real_plans_have_complete_offline_analysis_with_unknown_labels():
    a = api()
    from slc.followup_judge import load_batteries
    root = Path(__file__).resolve().parents[1]
    for suite, total in [('suite1',8064),('suite2',29812)]:
        directory = root/'results/followup_suites_20260907'/suite
        plan = json.loads((directory/'plan.json').read_text())
        metadata = load_batteries(root,directory,plan)
        rows = a.expected_samples(plan,metadata)
        report = a.analyze(rows,suite,draws=20)
        assert report['coverage']['expected_responses'] == total
        assert report['coverage']['generated_responses'] == 0
        assert all(t['counts']['n_generated']==0 for t in report['tables'])
        primary = report['effects']['consensus']
        if suite=='suite1':
            assert primary['pooled_order']['bounds'] == [-1.,1.]
            assert len(primary['retention_loss']) == 16
        else:
            assert primary['order_advantage']['pooled']['bounds'] == [-1.,1.]
        assert report == a.analyze(rows,suite,draws=20)


def test_offline_cli_reproduces_identical_outputs_without_credentials(tmp_path, monkeypatch):
    import importlib.util
    a = api()
    from slc.followup_judge import load_batteries
    root = Path(__file__).resolve().parents[1]
    script = root/'scripts/analyze_followup_suites.py'
    spec = importlib.util.spec_from_file_location('followup_cli_test',script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = root/'results/followup_suites_20260907/suite1'
    plan = json.loads((directory/'plan.json').read_text())
    rows = a.expected_samples(plan,load_batteries(root,directory,plan))
    snapshot = {'suite':'suite1','rows':rows,'draws':20,'provenance':{'purpose':'test'}}
    source = tmp_path/'snapshot.json'
    source.write_text(json.dumps(snapshot))
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    one,two = tmp_path/'one',tmp_path/'two'
    assert module.main(['--reproduce',str(source),'--output',str(one)]) == 0
    assert module.main(['--reproduce',str(source),'--output',str(two)]) == 0
    for name in ['results.json','tables.md','manifest.json']:
        assert (one/name).read_bytes() == (two/name).read_bytes()
    three = tmp_path/'three'
    assert module.main(['--reproduce',str(one/'input_snapshot.json'),'--output',str(three)]) == 0
    assert json.loads((three/'reproduction_check.json').read_text())['status'] == 'matched'


def test_reproduction_rejects_a_changed_snapshot_with_original_manifest(tmp_path):
    import importlib.util
    from slc.followup_runtime import json_bytes, sha
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('followup_cli_tampered',root/'scripts/analyze_followup_suites.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = {'suite':'suite1','rows':[],'draws':20,'provenance':{}}
    source = tmp_path/'input_snapshot.json'
    (tmp_path/'manifest.json').write_text(json.dumps({'input_snapshot_sha256':sha(json_bytes(original))}))
    source.write_bytes(json_bytes({**original,'draws':21}))
    with pytest.raises(ValueError,match='snapshot'):
        module.main(['--reproduce',str(source),'--output',str(tmp_path/'out')])


def test_analysis_cannot_publish_into_another_active_writers_directory(tmp_path):
    import fcntl
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('followup_cli_lock',root/'scripts/analyze_followup_suites.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (tmp_path/'.analysis.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            module.main(['--reproduce',str(tmp_path/'absent.json'),'--output',str(tmp_path)])
