"""Hand-calculated tests for the frozen name-assignment estimands."""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from slc import name_swap_analysis as analysis


def fixture_plan(seeds=(0, 1), families=2, samples=2):
    metadata = {}
    for order in ('original', 'reversed'):
        battery = f'contest_{order}_order'
        metadata[battery] = {
            f'f{f}-{cue}-{order}': dict(family_id=f'f{f}', cue_present=cue,
                name_order=order, need_type='neutral', region='contested')
            for f in range(families) for cue in (True, False)}
    for assignment in ('original', 'exchanged'):
        metadata[f'diagnostics_{assignment}'] = {
            f'{vendor}-{f}-{region}': dict(family_id=f'{vendor}-{f}',
                bank_vendor=vendor, target_vendor=vendor if assignment == 'original' else {'M':'S','S':'M'}[vendor],
                region=region, target_name_present=f != 0)
            for vendor in ('M','S') for f in range(families)
            for region in ('positive', 'named_not_live', 'named_no_authority',
                           'named_wrong_direction', 'rival_leaning')}
    jobs = [dict(tag=f'{a}{s}', assignment=a, seed=s)
            for s in seeds for a in ('original','exchanged')]
    batteries = {k: dict(samples=samples, n_scenarios=len(v)) for k,v in metadata.items()}
    count = sum(len(v)*samples*((len(jobs)+1) if k.startswith('contest') else (len(seeds)+1))
                for k,v in metadata.items())
    return dict(jobs=jobs,batteries=batteries,expected_total_responses=count),metadata


def complete_rows(seeds=(0,1), families=2, samples=2):
    plan, meta = fixture_plan(seeds,families,samples)
    rows = analysis.expected_samples(plan,meta)
    for row in rows:
        row.update(generated=True, labeled=True, M='yes', S='no',target_verdict='no')
        row['views'] = {v: dict(M='yes',S='no',target_verdict='no') for v in ('original','exchanged')}
    return rows


def test_expected_denominators_include_missing_models_and_samples():
    plan,meta = fixture_plan()
    rows = analysis.expected_samples(plan,meta)
    assert len(rows) == plan['expected_total_responses']
    assert all(r['M'] == 'unknown' and not r['generated'] for r in rows)
    assert len([r for r in rows if r['tag']=='base' and r['battery'].startswith('diagnostics')]) == 80
    result = analysis.analyze_contest(rows, seeds=[0,1],draws=80)
    assert result['statistics']['D_original']['bounds'] == [-1,1]
    assert result['statistics']['Delta']['bounds'] == [-2,2]
    assert result['statistics']['Delta']['point'] is None


def test_partial_labels_retain_feasible_cross_assignment_endpoints():
    rows = complete_rows()
    for r in rows:
        if r['assignment']=='exchanged':
            r.update(M='unknown',S='yes')
    result = analysis.analyze_contest(rows,seeds=[0,1],draws=80)
    assert result['statistics']['D_original']['bounds'] == [1,1]
    assert result['statistics']['D_exchanged']['bounds'] == [-1,0]
    assert result['statistics']['Delta']['bounds'] == [1,2]
    assert result['statistics']['Delta']['interval_98_333333'] == [1,2]


def test_crossed_resampling_preserves_seed_and_family_pairing():
    # Both assignments have the same nonconstant seed/family effect: paired Delta is exactly zero.
    lo = np.array([[[1,0],[-1,1]],[[1,0],[-1,1]]],dtype=float)
    result = analysis.crossed_bootstrap(lo,lo,draws=500,seed=20260908)
    assert np.all(result['Delta'][0] == 0)
    assert np.all(result['Delta'][1] == 0)
    assert len(np.unique(result['D_original'][0])) > 1


def test_equal_orders_families_and_seeds_and_sample_outcomes():
    rows = complete_rows(samples=2)
    for r in rows:
        if r['name_order'] if 'name_order' in r else False:
            if r['name_order']=='reversed': r.update(M='no',S='yes')
    result = analysis.analyze_contest(rows,seeds=[0,1],draws=80)
    assert result['statistics']['D_original']['point'] == 0
    summary = analysis.contest_summary([r for r in rows if r['tag']=='original0' and r.get('cue_present')])
    assert summary['outcomes'] == {'M_only':4,'S_only':4,'both':0,'neither':0,'unknown':0}
    assert summary['n_expected'] == 8


@pytest.mark.parametrize('interval,direction,magnitude', [
    ((-.099,.099),'unresolved','equivalent'),
    ((-.1,.099),'unresolved','unresolved'),
    ((.01,.05),'positive','equivalent'),
    ((.101,.3),'positive','large_positive'),
    ((-.3,-.101),'negative','large_negative'),
    ((.01,.2),'positive','unresolved')])
def test_direction_and_practical_magnitude_are_separate(interval,direction,magnitude):
    assert analysis.effect_decision(interval) == dict(direction=direction,magnitude=magnitude)


def test_diagnostics_require_worst_case_activation_and_both_gate_conditions():
    rows = [r for r in complete_rows(families=2,samples=2) if r['tag']=='original0' and r['battery']=='diagnostics_original']
    for r in rows: r['target_verdict'] = 'yes' if r['region']=='positive' else 'no'
    result = analysis.analyze_diagnostics(rows,draws=200)
    assert len(result)==2
    assert all(v['activation']=='pass' and all(g['status']=='pass' for g in v['gates'].values()) for v in result)
    assert result[0]['positive_by_target_name']['absent']['n_expected']==2
    # Unknown negatives can be yes: never declare a gate passed from observed-only zeros.
    for r in rows:
        if r['region']=='named_not_live': r['target_verdict']='unknown'
    result=analysis.analyze_diagnostics(rows,draws=200)
    assert all(v['gates']['named_not_live']['status']=='undetermined' for v in result)


def test_lost_seed_not_dropped_and_fresh_subset_fixed():
    rows = complete_rows(seeds=range(6))
    for r in rows:
        if r['assignment']=='original' and r['seed']==5: r.update(M='unknown',S='unknown')
    pooled = analysis.analyze_contest(rows,seeds=list(range(6)),draws=100)
    fresh = analysis.analyze_contest(rows,seeds=[2,3,4,5],draws=100)
    assert pooled['statistics']['D_original']['bounds'] == pytest.approx([2/3,1])
    assert fresh['statistics']['D_original']['bounds'] == [.5,1]
    assert len(pooled['leave_one_seed_out'])==6
    assert pooled['seed_t_sensitivity']['Delta']['point'] is None


def test_rejects_missing_pair_and_duplicate_expected_identity():
    plan,meta = fixture_plan()
    plan['jobs'].pop()
    with pytest.raises(ValueError,match='paired'): analysis.expected_samples(plan,meta)
    rows=complete_rows()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError,match='duplicate'): analysis.analyze_contest(rows,seeds=[0,1],draws=10)


def write_experiment(tmp_path, generated=1):
    plan,meta=fixture_plan()
    inputs=tmp_path/'inputs'; inputs.mkdir()
    for battery,scenarios in meta.items():
        path=inputs/(battery+'.jsonl')
        text=''.join(json.dumps(dict(id=s,prompt='A frozen prompt.',region=m['region']))+'\n' for s,m in scenarios.items())
        path.write_text(text)
        plan['batteries'][battery].update(path=str(path),sha256=analysis.file_sha(path))
    mp=inputs/'battery_metadata.json';mp.write_text(json.dumps(meta))
    plan.update(metadata_path=str(mp),metadata_sha256=analysis.file_sha(mp))
    pp=tmp_path/'plan.json';pp.write_text(json.dumps(plan))
    rows=analysis.expected_samples(plan,meta)
    for row in rows[:generated]:
        path=tmp_path/'raw'/row['tag']/row['battery']/'chunk_000000.jsonl'
        path.parent.mkdir(parents=True,exist_ok=True)
        raw={k:row[k] for k in ('scenario_id','sample_id','sample_index','region','family_id')}
        raw.update(prompt='A frozen prompt.',response='An answer.',model_provenance={'run_identity':{
            'model_tag':row['tag'],'battery_name':row['battery']}})
        with path.open('a') as stream: stream.write(json.dumps(raw)+'\n')
    lp=tmp_path/'labels.jsonl';lp.write_text('')
    return pp,lp,rows


def test_load_partial_validates_raw_and_fills_planned_missing(tmp_path):
    pp,lp,expected=write_experiment(tmp_path)
    label=expected[0].copy()
    label.update(M='yes',S='unknown',views={'original':dict(M='yes',S='no'),'exchanged':dict(M='yes',S='yes')})
    lp.write_text(json.dumps(label)+'\n')
    rows,manifest=analysis.load_experiment(pp,lp,tmp_path/'raw',partial=True)
    assert len(rows)==len(expected)
    assert sum(r['generated'] for r in rows)==1
    assert sum(r['labeled'] for r in rows)==1
    assert rows[0]['M']=='yes' and rows[0]['S']=='unknown'
    assert all(r['M']=='unknown' for r in rows[1:])
    assert manifest['completion']['missing_responses']==len(rows)-1
    with pytest.raises(ValueError,match='30,704|30704|incomplete'):
        analysis.load_experiment(pp,lp,tmp_path/'raw',partial=False)


@pytest.mark.parametrize('mutation,match', [
    ('duplicate_raw','duplicate'),('foreign_sample','unplanned'),
    ('changed_prompt','prompt'),('label_without_response','generated'),
    ('changed_consensus','consensus'),('changed_metadata','metadata')])
def test_input_validation_rejects_corrupt_or_unplanned_evidence(tmp_path,mutation,match):
    pp,lp,expected=write_experiment(tmp_path)
    raw=next((tmp_path/'raw').glob('*/*/chunk_*.jsonl'))
    if mutation=='duplicate_raw': raw.write_text(raw.read_text()*2)
    elif mutation in ('foreign_sample','changed_prompt'):
        item=json.loads(raw.read_text());item['sample_index' if mutation=='foreign_sample' else 'prompt']=999 if mutation=='foreign_sample' else 'Changed.'
        raw.write_text(json.dumps(item)+'\n')
    else:
        item=copy.deepcopy(expected[1 if mutation=='label_without_response' else 0])
        if mutation=='changed_metadata': item['family_id']='wrong family'
        elif mutation=='changed_consensus': item['M']='yes'
        lp.write_text(json.dumps(item)+'\n')
    with pytest.raises(ValueError,match=match):
        analysis.load_experiment(pp,lp,tmp_path/'raw',partial=True)


def test_report_is_deterministic_and_keeps_base_descriptive():
    rows=complete_rows(seeds=range(6))
    first=analysis.build_report(rows,draws=60)
    second=analysis.build_report(rows,draws=60)
    assert json.dumps(first,sort_keys=True)==json.dumps(second,sort_keys=True)
    assert first['primary']['n_expected']==6*2*2*2*2
    assert first['fresh_seeds']['seeds']==[2,3,4,5]
    assert first['base']['inference']=='descriptive; one untrained model'
    assert len(first['diagnostics'])==28
    assert first['measurement']['decision_sensitive'] is False
    assert len(first['per_seed_contest'])==12


def test_orientation_sensitivity_cannot_select_preferred_judge():
    rows=complete_rows(seeds=range(6))
    for row in rows:
        row.update(M='unknown',S='unknown')
        row['views']['exchanged'].update(M='no',S='yes')
    report=analysis.build_report(rows,draws=60)
    assert report['primary']['statistics']['D_original']['bounds']==[-1,1]
    assert report['measurement']['decision_sensitive'] is True
    assert report['measurement']['contest_fields']['resolved_disagreement']==2*len([r for r in rows if r['battery'].startswith('contest')])


def test_t_unknown_envelope_covers_interior_and_vertex_completions():
    low=np.array([-.2,.1,.4,-.1,.2,.5]);high=low+np.array([.2,.1,0,.3,.2,.1])
    reported=analysis._seed_t(low,high)['interval_98_333333']
    for bits in itertools.product((False,True),repeat=6):
        values=np.where(bits,high,low)
        half=3.5341107040583704*np.std(values,ddof=1)/np.sqrt(6)
        assert reported[0]<=values.mean()-half+1e-14
        assert reported[1]>=values.mean()+half-1e-14


def test_cli_requires_explicit_mode_and_produces_identical_bytes(tmp_path):
    import os
    import subprocess
    import sys
    import shutil
    # All six seeds, but small synthetic families keep this integration test cheap.
    plan,meta=fixture_plan(seeds=range(6))
    inputs=tmp_path/'inputs';inputs.mkdir()
    for battery,scenarios in meta.items():
        path=inputs/(battery+'.jsonl')
        path.write_text(''.join(json.dumps(dict(id=s,prompt='Frozen.',region=m['region']))+'\n' for s,m in scenarios.items()))
        plan['batteries'][battery].update(path=str(path),sha256=analysis.file_sha(path))
    mp=inputs/'battery_metadata.json';mp.write_text(json.dumps(meta))
    plan.update(metadata_path=str(mp),metadata_sha256=analysis.file_sha(mp))
    pp=tmp_path/'plan.json';pp.write_text(json.dumps(plan))
    root=Path(__file__).resolve().parents[1]
    guard=tmp_path/'network_guard';guard.mkdir()
    (guard/'sitecustomize.py').write_text(
        "import sys\ndef audit(event, args):\n    if event in ('socket.connect', 'socket.getaddrinfo'):\n        raise RuntimeError('Network access forbidden during reproduction')\nsys.addaudithook(audit)\n")
    env={**os.environ,'PYTHONPATH':str(guard)+os.pathsep+str(root/'src')}
    command=[sys.executable,str(root/'scripts/analyze_name_swap.py'),'--plan',str(pp),'--labels',str(tmp_path/'missing.jsonl'),'--raw-root',str(tmp_path/'raw')]
    rejected=subprocess.run(command+['--output',str(tmp_path/'no-mode')],env=env,capture_output=True)
    assert rejected.returncode!=0
    archive=tmp_path/'archive';archive.mkdir()
    shutil.copy2(pp,archive/'plan.json')
    shutil.copytree(inputs,archive/'inputs')
    for name in ('first','second'):
        if name=='second':
            inputs.rename(tmp_path/'hidden_inputs')
            command=[sys.executable,str(root/'scripts/analyze_name_swap.py'),'--plan',str(archive/'plan.json'),'--labels',str(archive/'missing.jsonl'),'--raw-root',str(archive/'raw')]
        completed=subprocess.run(command+['--partial','--output',str(tmp_path/name)],env=env,capture_output=True,text=True)
        assert completed.returncode==0,completed.stderr
    for name in ('results.json','summary.md'):
        assert (tmp_path/'first'/name).read_bytes()==(tmp_path/'second'/name).read_bytes()
    report=json.loads((tmp_path/'first/results.json').read_text())
    assert report['manifest']['completion']['generated_responses']==0
    assert report['primary']['statistics']['Delta']['bounds']==[-2,2]


def test_rejects_changed_sealed_chunk(tmp_path):
    pp,lp,_=write_experiment(tmp_path)
    raw=next((tmp_path/'raw').glob('*/*/chunk_*.jsonl'))
    raw.with_suffix('.meta.json').write_text(json.dumps({'responses_sha256':'0'*64,'n_responses':1}))
    with pytest.raises(ValueError,match='sealed chunk'):
        analysis.load_experiment(pp,lp,tmp_path/'raw',partial=True)


def test_rejects_wrong_plan_in_generation_provenance(tmp_path):
    pp,lp,_=write_experiment(tmp_path)
    raw=next((tmp_path/'raw').glob('*/*/chunk_*.jsonl'))
    value=json.loads(raw.read_text());value['model_provenance']['run_identity']['plan_sha256']='0'*64
    raw.write_text(json.dumps(value)+'\n')
    with pytest.raises(ValueError,match='plan hash'):
        analysis.load_experiment(pp,lp,tmp_path/'raw',partial=True)


def test_missing_bundle_cannot_fall_back_to_original_input_directory(tmp_path):
    pp,lp,_=write_experiment(tmp_path)
    plan=json.loads(pp.read_text())
    old=Path(plan['metadata_path'])
    external=tmp_path/'outside.json';old.rename(external)
    plan['metadata_path']=str(external);pp.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='bundled input'):
        analysis.load_experiment(pp,lp,tmp_path/'raw',partial=True)


def test_equivalence_decision_applies_only_to_prespecified_delta():
    result=analysis.analyze_contest(complete_rows(),seeds=[0,1],draws=100)
    assert result['decisions']['D_original']=={'direction':'positive'}
    assert result['decisions']['D_exchanged']=={'direction':'positive'}
    assert result['decisions']['Delta']=={'direction':'unresolved','magnitude':'equivalent'}


def test_positive_name_presence_reports_family_denominators():
    result=analysis.analyze_diagnostics(complete_rows(),draws=100)
    for vendor in result:
        assert vendor['positive_by_target_name']['present']['n_families']==1
        assert vendor['positive_by_target_name']['absent']['n_families']==1
        assert vendor['regions']['positive']['n_families']==2


@pytest.mark.parametrize('battery_prefix,field', [('contest_', 'S'), ('diagnostics_', 'target_verdict')])
def test_exported_uncertain_views_keep_raw_verdict_and_unknown_bounds(tmp_path,battery_prefix,field):
    plan,metadata=fixture_plan()
    pp,lp,expected=write_experiment(tmp_path,generated=plan['expected_total_responses'])
    exported=next(copy.deepcopy(r) for r in expected if r['battery'].startswith(battery_prefix))
    exported.pop('generated');exported.pop('labeled')
    exported['views']={'original':{field:'no'},'exchanged':{field:'uncertain'}}
    exported[field]='unknown'
    lp.write_text(json.dumps(exported)+'\n')
    rows,_=analysis.load_experiment(pp,lp,tmp_path/'raw',partial=True)
    loaded=next(r for r in rows if analysis.identity(r)==analysis.identity(exported))
    assert loaded['views']['exchanged'][field]=='uncertain'
    assert loaded[field]=='unknown'
    assert analysis.verdict_bounds(loaded['views']['exchanged'][field])==(0,1)
    assert analysis.label(loaded,field,'exchanged')=='unknown'
    rate=analysis._rate_summary([loaded],field,'exchanged')
    assert rate['unknown']==1 and rate['bounds']==[0,1]
    agreement=analysis._agreement([loaded],(field,))
    assert agreement['at_least_one_unknown']==1
    assert agreement['both_orientations_resolved']==0


def test_uncertain_views_pass_full_report_without_counting_as_no_service():
    rows=complete_rows(seeds=range(6))
    for row in rows:
        row.update(M='unknown',S='unknown',target_verdict='unknown')
        row['views']={view:dict(M='uncertain',S='uncertain',target_verdict='uncertain')
                      for view in ('original','exchanged')}
    result=analysis.build_report(rows,draws=60)
    for view in ('original','exchanged'):
        primary=result['judge_orientations'][view]['primary']
        assert primary['statistics']['D_original']['bounds']==[-1,1]
        assert primary['statistics']['Delta']['bounds']==[-2,2]
        diagnostic=result['judge_orientations'][view]['diagnostics'][0]
        assert diagnostic['regions']['positive']['bounds']==[0,1]
        assert diagnostic['activation']=='undetermined'
    for group in result['base']['strata']:
        assert group['outcomes']['unknown']==group['n_expected']
    fields=result['measurement']['contest_fields']
    assert fields['at_least_one_unknown']==fields['expected_fields']
    assert fields['resolved_disagreement']==0
