"""Offline, planned-denominator analysis for the original-bank name exchange.

Unknown labels are intervals, never complete-case estimates. The primary bootstrap
resamples paired training seeds and customer families independently, preserving
both assignments and all repeated responses within each selected family.
"""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ASSIGNMENTS = ('original', 'exchanged')
VIEWS = ('consensus', 'original', 'exchanged')
ALPHAS = {'95': .05, '98_333333': .05 / 3}
# Frozen Student t quantiles: scipy.stats.t.ppf(1-alpha/2, df), float64.
T_CRITICAL = {
    1: (12.706204736174694,38.188459297025744),
    2: (4.302652729749462,7.648803937915553),
    3: (3.1824463052837078,4.856657272768983),
    4: (2.7764451051977934,3.9607864827701835),
    5: (2.5705818356363146,3.5341107040583704),
}
NEGATIVE_REGIONS = ('named_not_live', 'named_no_authority', 'named_wrong_direction', 'rival_leaning')


def identity(row):
    return row['tag'], row['battery'], row['scenario_id'], row['sample_index']


def expected_samples(plan, metadata):
    """Enumerate every planned slot, including models with no generated output."""
    jobs = plan['jobs']
    pairs = [(j['assignment'], j['seed']) for j in jobs]
    seeds = sorted({j['seed'] for j in jobs})
    if (len(set(j['tag'] for j in jobs)) != len(jobs) or
            len(set(pairs)) != len(pairs) or
            set(pairs) != set(itertools.product(ASSIGNMENTS,seeds))):
        raise ValueError('jobs must contain unique paired seeds and assignments')
    result = []
    for job in jobs + [dict(tag='base', assignment='base', seed=None)]:
        for battery, specification in sorted(plan['batteries'].items()):
            if battery.startswith('diagnostics_') and job['tag'] != 'base' and battery != 'diagnostics_' + job['assignment']:
                continue
            if len(metadata[battery]) != specification['n_scenarios']:
                raise ValueError('metadata scenario count differs from plan')
            for scenario, meta in sorted(metadata[battery].items()):
                for sample_index in range(specification['samples']):
                    result.append({**meta, 'tag':job['tag'], 'assignment':job['assignment'],
                        'seed':job['seed'], 'battery':battery, 'scenario_id':scenario,
                        'sample_index':sample_index, 'sample_id':f'{scenario}#{sample_index}',
                        'M':'unknown', 'S':'unknown', 'target_verdict':'unknown',
                        'views':{}, 'generated':False, 'labeled':False})
    if len(result) != plan['expected_total_responses']:
        raise ValueError('planned total differs from enumerated sample count')
    return result


def verdict_bounds(value):
    if value == 'yes': return (1.,1.)
    if value == 'no': return (0.,0.)
    if value in ('unknown','uncertain'): return (0.,1.)
    raise ValueError(f'invalid verdict {value!r}')


def label(row, field, view='consensus'):
    """Normalize unresolved measurements without changing the raw judge views."""
    if view not in VIEWS: raise ValueError('unknown judge view')
    value=(row if view == 'consensus' else row.get('views',{}).get(view,{})).get(field,'unknown')
    return 'unknown' if value == 'uncertain' else value


def gap_bounds(row, view='consensus'):
    ml,mh = verdict_bounds(label(row,'M',view))
    sl,sh = verdict_bounds(label(row,'S',view))
    return ml-sh, mh-sl


def effect_decision(interval, margin=.1):
    low, high = interval
    direction = 'positive' if low > 0 else 'negative' if high < 0 else 'unresolved'
    magnitude = ('equivalent' if low > -margin and high < margin else
                 'large_positive' if low > margin else
                 'large_negative' if high < -margin else 'unresolved')
    return dict(direction=direction,magnitude=magnitude)


def crossed_bootstrap(low, high, *, draws=20000, seed=20260908):
    """Return lower/upper bootstrap distributions for arrays [assignment,seed,family]."""
    low,high = np.asarray(low,float),np.asarray(high,float)
    if low.shape != high.shape or low.ndim != 3 or low.shape[0] != 2 or np.any(low>high):
        raise ValueError('expected ordered bounds with shape [2, seed, family]')
    if draws < 1 or min(low.shape) < 1: raise ValueError('empty bootstrap')
    rng = np.random.default_rng(seed)
    ns,nf = low.shape[1:]
    seed_weights = np.eye(ns)[rng.integers(ns,size=(draws,ns))].sum(axis=1)/ns
    family_weights = np.eye(nf)[rng.integers(nf,size=(draws,nf))].sum(axis=1)/nf
    lo = np.einsum('bs,asf,bf->ab',seed_weights,low,family_weights,optimize=False)
    hi = np.einsum('bs,asf,bf->ab',seed_weights,high,family_weights,optimize=False)
    return {'D_original':(lo[0],hi[0]),'D_exchanged':(lo[1],hi[1]),
            'Delta':(lo[0]-hi[1],hi[0]-lo[1])}


def _statistic(bounds, bootstrap):
    low,high = map(float,bounds)
    result = {'bounds':[low,high], 'point':low if low == high else None}
    for name,alpha in ALPHAS.items():
        result['interval_'+name] = [float(np.quantile(bootstrap[0],alpha/2)),
                                      float(np.quantile(bootstrap[1],1-alpha/2))]
    return result


def _seed_t(low,high):
    """Conservative t sensitivity envelope; evaluate maximum SD at box vertices.

    Unlike a t interval on endpoint data alone, this covers all possible standard
    deviations of feasible seed effects. It remains a model-based sensitivity,
    conditional on the evaluated families, not a coverage claim for this design.
    """
    low,high = np.asarray(low),np.asarray(high)
    result={'bounds':[float(low.mean()),float(high.mean())],
            'point':float(low.mean()) if np.array_equal(low,high) else None,
            'n_seeds':len(low), 'degrees_of_freedom':len(low)-1,
            'method':'t with worst feasible seed SD; conditional on frozen families'}
    if len(low)<2:
        return {**result, **{'interval_'+name:None for name in ALPHAS}}
    sd = max(float(np.std(np.where(bits,high,low),ddof=1))
             for bits in itertools.product((False,True),repeat=len(low)))
    result['maximum_feasible_seed_sd']=sd
    for name,alpha in ALPHAS.items():
        half=T_CRITICAL[len(low)-1][list(ALPHAS).index(name)]*sd/np.sqrt(len(low))
        result['interval_'+name]=[float(low.mean()-half),float(high.mean()+half)]
    return result


def _contest_arrays(rows,seeds,view):
    keys=[identity(r) for r in rows]
    if len(set(keys)) != len(keys): raise ValueError('duplicate sample identity')
    families=sorted({r['family_id'] for r in rows})
    groups=defaultdict(list)
    for r in rows: groups[(r['assignment'],r['seed'],r['family_id'],r['name_order'])].append(r)
    orders=sorted({r['name_order'] for r in rows})
    low=np.empty((2,len(seeds),len(families)))
    high=np.empty_like(low)
    for ai,a in enumerate(ASSIGNMENTS):
        for si,s in enumerate(seeds):
            for fi,f in enumerate(families):
                cells=[]
                for order in orders:
                    values=groups.get((a,s,f,order),[])
                    if not values: raise ValueError('missing paired seed/family/order cell')
                    cells.append(np.mean([gap_bounds(r,view) for r in values],axis=0))
                low[ai,si,fi],high[ai,si,fi]=np.mean(cells,axis=0)
    return low,high,families,orders


def analyze_contest(rows, *, seeds=tuple(range(6)), view='consensus', draws=20000,
                    bootstrap_seed=20260908, cue_present=True, name_order=None, need_type=None):
    if len({identity(r) for r in rows}) != len(rows):
        raise ValueError('duplicate sample identity')
    selected=[r for r in rows if r['battery'].startswith('contest_') and
              r['assignment'] in ASSIGNMENTS and r['seed'] in seeds and
              r['cue_present']==cue_present and (name_order is None or r['name_order']==name_order)
              and (need_type is None or r['need_type']==need_type)]
    low,high,families,orders=_contest_arrays(selected,seeds,view)
    boot=crossed_bootstrap(low,high,draws=draws,seed=bootstrap_seed)
    def quantities(lo,hi):
        return {'D_original':(lo[0],hi[0]),'D_exchanged':(lo[1],hi[1]),
                'Delta':(lo[0]-hi[1],hi[0]-lo[1])}
    bounds=quantities(low.mean(axis=(1,2)),high.mean(axis=(1,2)))
    statistics={k:_statistic(b,boot[k]) for k,b in bounds.items()}
    seed_bounds=quantities(low.mean(axis=2),high.mean(axis=2))
    sensitivity={k:_seed_t(*b) for k,b in seed_bounds.items()}
    leave=[]
    for i,s in enumerate(seeds):
        if len(seeds)>1:
            keep=[j for j in range(len(seeds)) if j!=i]
            leave.append({'omitted_seed':s, 'bounds':{k:list(map(float,b)) for k,b in
                quantities(low[:,keep].mean(axis=(1,2)),high[:,keep].mean(axis=(1,2))).items()}})
    adjusted={k:v['interval_98_333333'] for k,v in statistics.items()}
    decisions={k:effect_decision(interval) if k=='Delta' else
                  {'direction':effect_decision(interval)['direction']}
               for k,interval in adjusted.items()}
    decisions['winner_reversal']=(adjusted['D_original'][0]>0 and adjusted['D_exchanged'][1]<0 and adjusted['Delta'][0]>0)
    return dict(seeds=list(seeds),n_families=len(families),name_orders=orders,
                n_expected=len(selected),view=view,cue_present=cue_present,need_type=need_type,
                statistics=statistics,decisions=decisions,seed_t_sensitivity=sensitivity,
                leave_one_seed_out=leave)


def contest_summary(rows,view='consensus'):
    outcomes=Counter({k:0 for k in ('M_only','S_only','both','neither','unknown')})
    for row in rows:
        m,s=label(row,'M',view),label(row,'S',view)
        outcome='unknown' if 'unknown' in (m,s) else {('yes','no'):'M_only',('no','yes'):'S_only',('yes','yes'):'both',('no','no'):'neither'}[(m,s)]
        outcomes[outcome]+=1
    gap=np.mean([gap_bounds(r,view) for r in rows],axis=0).tolist() if rows else None
    return dict(n_expected=len(rows),n_generated=sum(r['generated'] for r in rows),
                outcomes=dict(outcomes),gap_bounds=gap,
                service={v:_rate_summary(rows,v,view) for v in ('M','S')})


def _rate_summary(rows,field,view):
    counts=Counter(label(r,field,view) for r in rows)
    n=len(rows)
    return dict(n_expected=n,n_families=len({r['family_id'] for r in rows}),yes=counts['yes'],no=counts['no'],unknown=counts['unknown'],
                bounds=[counts['yes']/n,(counts['yes']+counts['unknown'])/n] if n else None)


def _scenario_rate(rows,field,view,draws,seed):
    groups=defaultdict(list)
    for r in rows: groups[r['family_id']].append(verdict_bounds(label(r,field,view)))
    values=np.array([np.mean(groups[f],axis=0) for f in sorted(groups)])
    rng=np.random.default_rng(seed)
    resampled=values[rng.integers(len(values),size=(draws,len(values)))].mean(axis=1)
    result=_rate_summary(rows,field,view)
    # Family weighting is explicit, even if future fixtures have unequal sample counts.
    result['bounds']=values.mean(axis=0).tolist()
    result['n_families']=len(values)
    result['interval_95']=[float(np.quantile(resampled[:,0],.025)),float(np.quantile(resampled[:,1],.975))]
    result['best_case_interval_95']=[float(np.quantile(resampled[:,1],.025)),float(np.quantile(resampled[:,0],.975))]
    return result


def analyze_diagnostics(rows, *, view='consensus',draws=20000,bootstrap_seed=20260908):
    groups=defaultdict(list)
    for r in rows:
        if r['battery'].startswith('diagnostics_'):
            groups[(r['tag'],r['battery'],r['bank_vendor'],r['target_vendor'])].append(r)
    result=[]
    for (tag,battery,bank,target),group in sorted(groups.items()):
        regions={region:_scenario_rate([r for r in group if r['region']==region],
                  'target_verdict',view,draws,bootstrap_seed)
                  for region in ('positive',)+NEGATIVE_REGIONS}
        positive=regions['positive']; pl,ph=positive['bounds']
        gates={}
        for region in NEGATIVE_REGIONS:
            neg=regions[region]; nl,nh=neg['bounds']
            separation=neg['interval_95'][1] < positive['interval_95'][0]
            ratio=nh < .5*pl
            impossible=(neg['best_case_interval_95'][1] >= positive['best_case_interval_95'][0] or nl >= .5*ph)
            gates[region]=dict(status='pass' if separation and ratio else 'fail' if impossible else 'undetermined',
                worst_case_interval_separation=separation,worst_case_less_than_half=ratio)
        result.append(dict(tag=tag,assignment=group[0]['assignment'],seed=group[0]['seed'],
            battery=battery,bank_vendor=bank,target_vendor=target,regions=regions,gates=gates,
            activation='pass' if pl>=.5 else 'fail' if ph<.5 else 'undetermined',
            positive_by_target_name={name:_rate_summary([r for r in group if r['region']=='positive' and r['target_name_present']==present],
                'target_verdict',view) for name,present in (('present',True),('absent',False))}))
    return result


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def _input_path(plan_path, recorded):
    # Prefer bundled inputs, so an archive never silently reads its source checkout.
    bundled=plan_path.parent/'inputs'/Path(recorded).name
    if bundled.exists(): return bundled
    raise ValueError(f'missing bundled input {Path(recorded).name}')


def load_experiment(plan_path,labels_path,raw_root,*,partial=False):
    """Verify local frozen inputs and retain every unobserved planned label."""
    from slc.competition import ResponseRecord
    plan_path,labels_path,raw_root=map(Path,(plan_path,labels_path,raw_root))
    plan=json.loads(plan_path.read_text())
    mp=_input_path(plan_path,plan['metadata_path'])
    if file_sha(mp)!=plan['metadata_sha256']: raise ValueError('metadata hash differs from plan')
    metadata=json.loads(mp.read_text())
    rows=expected_samples(plan,metadata)
    expected={identity(r):r for r in rows}
    input_hashes={'plan.json':file_sha(plan_path),'inputs/'+mp.name:file_sha(mp)}
    scenarios={}
    for battery,spec in plan['batteries'].items():
        path=_input_path(plan_path,spec['path'])
        if file_sha(path)!=spec['sha256']: raise ValueError('battery hash differs from plan')
        input_hashes['inputs/'+path.name]=file_sha(path)
        prompts=list(_jsonl(path));scenarios[battery]={p['id']:p for p in prompts}
        if len(scenarios[battery])!=len(prompts) or set(scenarios[battery])!=set(metadata[battery]):
            raise ValueError('battery and metadata scenario identities differ')
    raw_hashes={}
    for path in sorted(raw_root.glob('*/*/chunk_*.jsonl')):
        tag,battery=path.relative_to(raw_root).parts[:2]
        chunk=list(_jsonl(path))
        seal_path=path.with_suffix('.meta.json')
        if seal_path.exists():
            seal=json.loads(seal_path.read_text())
            if seal['responses_sha256']!=file_sha(path) or seal['n_responses']!=len(chunk):
                raise ValueError('sealed chunk content differs from its metadata')
            raw_hashes[seal_path.relative_to(raw_root).as_posix()]=file_sha(seal_path)
        for raw in chunk:
            key=(tag,battery,raw['scenario_id'],raw['sample_index'])
            if key not in expected: raise ValueError('unplanned raw sample')
            row=expected[key]
            if row['generated']: raise ValueError('duplicate raw sample')
            record=ResponseRecord(**raw)
            for field in ('sample_id','region'):
                if getattr(record,field)!=row[field]: raise ValueError('raw sample metadata differs from plan')
            prompt=scenarios[battery][row['scenario_id']]
            if record.prompt!=prompt['prompt'] or record.messages!=prompt.get('messages'):
                raise ValueError('raw prompt differs from frozen battery')
            run=record.model_provenance.get('run_identity',{})
            if run.get('model_tag')!=tag or run.get('battery_name')!=battery:
                raise ValueError('raw model provenance differs from directory identity')
            if 'plan_sha256' in run and run['plan_sha256']!=input_hashes['plan.json']:
                raise ValueError('generation plan hash differs from frozen plan')
            if 'battery_sha256' in run and run['battery_sha256']!=plan['batteries'][battery]['sha256']:
                raise ValueError('generation battery hash differs from frozen plan')
            if 'run_identity_sha256' in record.model_provenance:
                run_hash=hashlib.sha256(json.dumps(run,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                if record.model_provenance['run_identity_sha256']!=run_hash:
                    raise ValueError('raw run identity hash differs from its content')
            if 'training_assignment' in run and run['training_assignment']!=row['assignment']:
                raise ValueError('raw training assignment differs from plan')
            if 'training_seed' in run and run['training_seed']!=row['seed']:
                raise ValueError('raw training seed differs from plan')
            row['generated']=True
        raw_hashes[path.relative_to(raw_root).as_posix()]=file_sha(path)
    if labels_path.exists():
        input_hashes['labels.jsonl']=file_sha(labels_path)
        for supplied in _jsonl(labels_path):
            key=identity(supplied)
            if key not in expected: raise ValueError('unplanned label sample')
            row=expected[key]
            if not row['generated']: raise ValueError('label has no generated response')
            if row['labeled']: raise ValueError('duplicate label sample')
            fields=('assignment','seed','sample_id','family_id','region','cue_present',
                    'name_order','need_type','bank_vendor','target_vendor','target_name_present')
            for field in fields:
                if field in row and supplied.get(field)!=row[field]:
                    raise ValueError(f'label metadata differs from plan: {field}')
            row['views']=supplied.get('views',{})
            targets=('M','S') if row['battery'].startswith('contest_') else ('target_verdict',)
            for field in targets:
                original=label(row,field,'original');exchanged=label(row,field,'exchanged')
                verdict_bounds(original);verdict_bounds(exchanged)
                consensus=original if original==exchanged and original in ('yes','no') else 'unknown'
                if supplied.get(field,'unknown')!=consensus:
                    raise ValueError('primary label differs from judge-name consensus')
                row[field]=consensus
            row['labeled']=True
    else:
        input_hashes['labels.jsonl']=None
    generated=sum(r['generated'] for r in rows)
    if not partial and (len(rows)!=30704 or generated!=30704 or
            sorted({j['seed'] for j in plan['jobs']})!=list(range(6))):
        raise ValueError('final analysis requires all 30,704 planned responses; use --partial for incomplete data')
    completion=dict(mode='partial' if partial else 'final',expected_responses=len(rows),
                    generated_responses=generated,missing_responses=len(rows)-generated,
                    labeled_responses=sum(r['labeled'] for r in rows))
    return rows,dict(input_sha256=input_hashes,raw_sha256=raw_hashes,completion=completion,
        analysis_settings=dict(bootstrap_draws=plan.get('bootstrap_draws',20000),
            bootstrap_seed=plan.get('bootstrap_seed',20260908),practical_margin=plan.get('practical_margin',.1)))


def _agreement(rows,fields):
    counts=Counter(expected_fields=0,agreed_yes=0,agreed_no=0,resolved_disagreement=0,
                   at_least_one_unknown=0)
    for r in rows:
        for field in fields:
            counts['expected_fields']+=1
            a,b=(label(r,field,v) for v in ('original','exchanged'))
            counts['at_least_one_unknown' if 'unknown' in (a,b) else
                   'agreed_'+a if a==b else 'resolved_disagreement']+=1
    counts['both_orientations_resolved']=counts['agreed_yes']+counts['agreed_no']+counts['resolved_disagreement']
    return dict(counts)


def build_report(rows,*,draws=20000,bootstrap_seed=20260908):
    """All prespecified outputs. No outcome-dependent selection or exclusions."""
    options=dict(draws=draws,bootstrap_seed=bootstrap_seed)
    primary=analyze_contest(rows,**options)
    fresh=analyze_contest(rows,seeds=[2,3,4,5],**options)
    orientations={view:dict(primary=analyze_contest(rows,view=view,**options),
                            fresh_seeds=analyze_contest(rows,seeds=[2,3,4,5],view=view,**options),
                            diagnostics=analyze_diagnostics(rows,view=view,**options))
                  for view in ('original','exchanged')}
    diagnostic=analyze_diagnostics(rows,**options)
    trained=[r for r in rows if r['assignment'] in ASSIGNMENTS and r['battery'].startswith('contest_')]
    primary_rows=[r for r in trained if r['cue_present']]
    per_seed=[]
    for tag in sorted({r['tag'] for r in primary_rows}):
        group=[r for r in primary_rows if r['tag']==tag]
        per_seed.append(dict(tag=tag,assignment=group[0]['assignment'],seed=group[0]['seed'],
                             **contest_summary(group)))
    secondary={}
    for cue in (True,False):
        for order in sorted({r['name_order'] for r in trained}):
            secondary[f'cue_{str(cue).lower()}_order_{order}']=analyze_contest(rows,cue_present=cue,name_order=order,**options)
    secondary['cue_absent_average_orders']=analyze_contest(rows,cue_present=False,**options)
    for need in sorted({r['need_type'] for r in trained}):
        secondary['need_'+need]=analyze_contest(rows,need_type=need,**options)
    base_rows=[r for r in rows if r['tag']=='base' and r['battery'].startswith('contest_')]
    base=dict(inference='descriptive; one untrained model',strata=[])
    for view in VIEWS:
        for cue in (True,False):
            group=[r for r in base_rows if r['cue_present']==cue]
            base['strata'].append(dict(view=view,cue_present=cue,name_order='average',**contest_summary(group,view)))
            for order in sorted({r['name_order'] for r in group}):
                base['strata'].append(dict(view=view,cue_present=cue,name_order=order,
                    **contest_summary([r for r in group if r['name_order']==order],view)))
    sensitive={scope:[key for key in ref['decisions'] if any(
        orientations[v][scope]['decisions'][key]!=ref['decisions'][key] for v in ('original','exchanged'))]
        for scope,ref in (('primary',primary),('fresh_seeds',fresh))}
    diagnostic_sensitive=[]
    for index,group in enumerate(diagnostic):
        changed=[]
        for view in ('original','exchanged'):
            other=orientations[view]['diagnostics'][index]
            if (group['activation']!=other['activation'] or any(group['gates'][r]['status']!=other['gates'][r]['status'] for r in NEGATIVE_REGIONS)):
                changed.append(view)
        if changed: diagnostic_sensitive.append(dict(tag=group['tag'],battery=group['battery'],bank_vendor=group['bank_vendor'],views=changed))
    bridge=[dict(tag=tag,**contest_summary([r for r in trained if r['tag']==tag and r['cue_present'] and r['name_order']=='original'],'original'))
            for tag in sorted({r['tag'] for r in trained})]
    return dict(schema_version=1,primary=primary,fresh_seeds=fresh,per_seed_contest=per_seed,
        diagnostics=diagnostic,base=base,judge_orientations=orientations,secondary=secondary,
        historical_bridge=bridge,
        measurement=dict(decision_sensitive=any(sensitive.values()) or bool(diagnostic_sensitive),
            changed_contest_decisions=sensitive,changed_diagnostic_decisions=diagnostic_sensitive,
            contest_fields=_agreement([r for r in rows if r['battery'].startswith('contest_')],('M','S')),
            diagnostic_fields=_agreement([r for r in rows if r['battery'].startswith('diagnostics_')],('target_verdict',))),
        methods=dict(bootstrap_draws=draws,bootstrap_seed=bootstrap_seed,
            bootstrap='paired seeds crossed with whole customer families; equal orders within family',
            intervals='95% and nominal Bonferroni-adjusted 98.333333% bootstrap envelopes',
            unknowns='lower bootstrap quantile of lower endpoints, upper quantile of upper endpoints',
            secondary='prespecified descriptive sensitivity; no confirmatory subgroup selection',
            installation='separate activation >= 0.50 and four relative region gates per model and vendor',
            limitations=['Six seeds give limited training-variation information.',
                'Bootstrap envelopes do not guarantee finite-sample simultaneous coverage.',
                'Unknown bounds do not correct undetected systematic judge error.',
                'A winner reversal alone does not establish conflict resolution between two installed loyalties.',
                'Calibration, unchanged-name repeats, and blind response audits require separate evidence.',
                'The intervention estimates name-to-example assignment, not a pure simplicity effect.']))
