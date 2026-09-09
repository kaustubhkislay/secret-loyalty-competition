"""Planned-denominator, paired analysis of the two follow-up suites."""
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path

import numpy as np

from slc.followup_runtime import file_sha, json_bytes, sha

VIEWS = ('consensus', 'original', 'exchanged')
DRAWS = 20000
BOOTSTRAP_SEED = 20260909
NUMERICAL_ZERO_TOLERANCE = 1e-12


def effect_direction(interval):
    low,high = interval
    return ('positive' if low > NUMERICAL_ZERO_TOLERANCE else
            'negative' if high < -NUMERICAL_ZERO_TOLERANCE else 'unresolved')


def identity(row):
    return row['tag'], row['battery'], row['scenario_id'], row['sample_index']


def expected_samples(plan, metadata):
    models = plan.get('models', plan.get('evaluated_models', []))
    if not models or len({m['tag'] for m in models}) != len(models):
        raise ValueError('missing or duplicate planned model')
    rows = []
    for model in models:
        for name, battery in sorted(plan['batteries'].items()):
            scenarios = metadata[name]
            if (len(scenarios) != battery['n_scenarios']
                    or len({r['id'] for r in scenarios}) != len(scenarios)):
                raise ValueError('planned scenario count differs')
            for scenario in scenarios:
                for index in range(battery['n_samples']):
                    rows.append({**{k:v for k,v in scenario.items() if k not in ('messages','prompt','clauses')},
                        **{k:v for k,v in model.items() if k in ('tag','arm','seed','model_role','principal',
                            'first_mover','second_mover','anchor','overlap','individual_reference_tag')},
                        'battery':name, 'scenario_id':scenario['id'], 'sample_index':index,
                        'sample_id':f'{scenario["id"]}#{index}', 'A':'unknown','B':'unknown',
                        'M':'unknown','S':'unknown','target_verdict':'unknown', 'views':{},
                        'generated':False, 'labeled':False, 'finished_cap':False})
    if len(rows) != plan['expected_total_responses']:
        raise ValueError('planned total response count differs')
    return rows


def merge_labels(planned, labels):
    merged = {identity(r):dict(r) for r in planned}
    if len(merged) != len(planned):
        raise ValueError('duplicate planned sample')
    seen = set()
    for row in labels:
        key = identity(row)
        if key in seen:
            raise ValueError('duplicate labeled sample')
        seen.add(key)
        if key not in merged:
            raise ValueError('label is outside planned samples')
        current = merged[key]
        for field in ('seed','arm','family_id','region','mention_order','target_vendor'):
            if field in current and field in row and current[field] != row[field]:
                raise ValueError(f'label differs from planned metadata: {field}')
        current.update({k:v for k,v in row.items() if k in ('A','B','M','S','target_verdict','views',
            'finished_cap','finish_reason','generated_tokens','continued','response_sha256','view_status')})
        current.update(generated=True, labeled=True)
    return list(merged.values())


def label(row, field, view='consensus'):
    if view not in VIEWS:
        raise ValueError('unknown judge view')
    if row.get('finished_cap') or row.get('finish_reason') == 'length':
        return 'unknown'
    if view == 'consensus' and 'views' in row:
        a,b = (label(row,field,v) for v in ('original','exchanged'))
        return a if a==b and a in ('yes','no') else 'unknown'
    value = (row if view == 'consensus' else row.get('views', {}).get(view, {})).get(field, 'unknown')
    if value == 'uncertain': value = 'unknown'
    if value not in ('yes','no','unknown'):
        raise ValueError('invalid verdict')
    return value


def outcome_bounds(row, metric, view='consensus'):
    def values(field):
        return {'yes':(1.,),'no':(0.,),'unknown':(0.,1.)}[label(row, field, view)]
    if '_only' in metric:
        a = metric.removesuffix('_only')
        b = {'A':'B','B':'A','M':'S','S':'M'}[a]
        possible = [x*(1-y) for x,y in itertools.product(values(a), values(b))]
    elif '_minus_' in metric:
        a,b = metric.split('_minus_')
        possible = [x-y for x,y in itertools.product(values(a), values(b))]
    else:
        possible = values(metric)
    return min(possible), max(possible)


def summarize(rows, principals, view='consensus'):
    n = len(rows)
    outcomes = Counter({k:0 for k in (principals[0]+'_only',principals[1]+'_only','both','neither','unknown')}) if len(principals)==2 else None
    support = {}
    for p in principals:
        counts = Counter(label(r,p,view) for r in rows)
        support[p] = {k:counts[k] for k in ('yes','no','unknown')}
        support[p]['bounds'] = [counts['yes']/n,(counts['yes']+counts['unknown'])/n] if n else None
    if outcomes is not None:
        mapping = {('yes','no'):principals[0]+'_only', ('no','yes'):principals[1]+'_only',
                   ('yes','yes'):'both', ('no','no'):'neither'}
        for r in rows:
            outcomes[mapping.get(tuple(label(r,p,view) for p in principals),'unknown')] += 1
    return {'n_expected':n, 'n_generated':sum(r.get('generated',False) for r in rows),
            'n_cap_limited':sum(r.get('finished_cap',False) for r in rows),
            'n_families':len({r['family_id'] for r in rows}),
            'support':support, 'outcomes':dict(outcomes) if outcomes is not None else None}


def select(rows, where):
    return [r for r in rows if all(r.get(k)==v for k,v in where.items())]


def contrast_component(rows, terms, *, view='consensus', stratum='shared', weight=1.):
    """Pair every linear-contrast term within each training seed and family."""
    selected = [select(rows, where) for where, coefficient, metric in terms]
    seeds = sorted({r['seed'] for group in selected for r in group}, key=str)
    families = sorted({r['family_id'] for group in selected for r in group})
    if not seeds or not families:
        raise ValueError('empty paired contrast')
    bounds = np.zeros((len(seeds),len(families),2))
    for group, (_,coef,metric) in zip(selected,terms):
        cells = defaultdict(list)
        for row in group:
            cells[row['seed'],row['family_id']].append(outcome_bounds(row,metric,view))
        for si,seed in enumerate(seeds):
            for fi,family in enumerate(families):
                if (seed,family) not in cells:
                    raise ValueError('missing paired seed/family cell')
                b = np.mean(cells[seed,family],axis=0)
                bounds[si,fi] += coef * (b if coef>=0 else b[::-1])
    return {'seeds':seeds,'families':families,'bounds':bounds,'stratum':stratum,'weight':weight}


def estimate(components, *, draws=DRAWS, random_seed=BOOTSTRAP_SEED, alpha=.05/3):
    if draws < 1 or not components: raise ValueError('empty bootstrap')
    seeds = components[0]['seeds']
    if any(c['seeds'] != seeds for c in components):
        raise ValueError('paired seeds differ between components')
    rng = np.random.default_rng(random_seed)
    ns = len(seeds)
    sw = rng.multinomial(ns, np.full(ns,1/ns), size=draws) / ns
    family_weights = {}
    distribution = np.zeros((draws,2))
    seed_values = np.zeros((ns,2))
    for c in components:
        x = np.asarray(c['bounds'],float)
        if x.shape != (ns,len(c['families']),2) or np.any(x[:,:,0] > x[:,:,1]):
            raise ValueError('invalid paired bounds')
        key = c['stratum']
        if key in family_weights:
            old_families,fw = family_weights[key]
            if old_families != c['families']:
                raise ValueError('one family stratum has inconsistent identities')
        else:
            nf = len(c['families'])
            fw = rng.multinomial(nf, np.full(nf,1/nf), size=draws) / nf
            family_weights[key] = (c['families'],fw)
        distribution += c['weight'] * np.einsum('bs,sfk,bf->bk',sw,x,fw,optimize=False)
        seed_values += c['weight'] * x.mean(axis=1)
    bounds = seed_values.mean(axis=0).tolist()
    def interval(a):
        values = [float(np.quantile(distribution[:,0],a/2)), float(np.quantile(distribution[:,1],1-a/2))]
        return [0. if abs(v)<NUMERICAL_ZERO_TOLERANCE else v for v in values]
    adjusted = interval(alpha)
    return {'bounds':bounds,'point':bounds[0] if bounds[0]==bounds[1] else None,
        'interval_95':interval(.05), 'interval_adjusted':adjusted, 'adjusted_alpha':alpha,
        'direction':effect_direction(adjusted),
        'n_seeds':ns, 'family_counts':{k:len(v[0]) for k,v in family_weights.items()},
        'seed_effects':[{'seed':s,'bounds':seed_values[i].tolist()} for i,s in enumerate(seeds)],
        'leave_one_seed_out':[{'omitted_seed':s,'bounds':np.delete(seed_values,i,axis=0).mean(axis=0).tolist()}
                              for i,s in enumerate(seeds)] if ns>1 else [],
        'draws':draws,'bootstrap_seed':random_seed}


def suite2_effects(rows, *, view='consensus', draws=DRAWS, primary_only=False):
    result = {}
    trained = [r for r in rows if r.get('arm') != 'base']
    for name in ('order_advantage','suppression','excess_suppression'):
        components = []
        per_vendor = {}
        for vendor,other in [('M','S'),('S','M')]:
            if name == 'order_advantage':
                scope = select(trained,{'battery':'contest_exclusive'})
                terms = [({'arm':other+'then'+vendor},1,vendor+'_only'),
                         ({'arm':vendor+'then'+other},-1,vendor+'_only')]
                stratum = 'exclusive'
            else:
                scope = select(trained,{'battery':'diagnostics','region':'positive','target_vendor':vendor})
                initial = vendor if name=='suppression' else vendor+'thenN'
                terms = [({'arm':initial},1,vendor), ({'arm':vendor+'then'+other},-1,vendor)]
                stratum = 'diagnostic_'+vendor
            c = contrast_component(scope, terms, view=view, stratum=stratum)
            per_vendor[vendor] = estimate([c],draws=draws,alpha=.05)
            components.append({**c,'weight':.5})
        result[name] = {'pooled':estimate(components,draws=draws),**per_vendor}
    if primary_only: return result
    secondary = {}
    for battery,condition in [('contest_exclusive',None),('contest_historical','cue_present'),
                              ('contest_historical','cue_absent')]:
        scope = select(trained,{'battery':battery})
        if condition: scope = select(scope,{'condition':condition})
        if not scope: continue
        order_components = []
        for vendor,other in [('M','S'),('S','M')]:
            c = contrast_component(scope,[({'arm':other+'then'+vendor},1,vendor+'_only'),
                ({'arm':vendor+'then'+other},-1,vendor+'_only')],view=view)
            order_components.append({**c,'weight':.5})
            secondary[f'{battery}:{condition}:{vendor}:order_advantage'] = estimate([c],draws=draws,alpha=.05)
            for comparator in [vendor+'then'+other,other+'then'+vendor]:
                c = contrast_component(scope,[({'arm':'mixed'},1,vendor+'_only'),
                    ({'arm':comparator},-1,vendor+'_only')],view=view)
                secondary[f'{battery}:{condition}:{vendor}:mixed-minus-{comparator}'] = estimate([c],draws=draws,alpha=.05)
        secondary[f'{battery}:{condition}:pooled:order_advantage'] = estimate(order_components,draws=draws,alpha=.05)
    result['secondary'] = secondary
    return result


def suite1_effects(rows, *, view='consensus', draws=DRAWS):
    order = select(rows,{'battery':'expanded_order'})
    terms = [({'mention_order':'AB'},.5,'A_minus_B'),
             ({'mention_order':'BA'},-.5,'A_minus_B')]
    historical = select(order,{'model_role':'checkpoint_sequential'})
    pooled = estimate([contrast_component(historical,terms,view=view)],draws=draws,alpha=.05)
    by_model = {}
    for tag in sorted({r['tag'] for r in order}):
        by_model[tag] = estimate([contrast_component(select(order,{'tag':tag}),terms,view=view)],draws=draws,alpha=.05)
    conditions = defaultdict(list)
    for row in historical:
        conditions[(row['first_mover'],row['second_mover'],row['overlap'],row['anchor'])].append(row)
    by_condition = {':'.join(map(str,key)):estimate([contrast_component(group,terms,view=view)],draws=draws,alpha=.05)
                    for key,group in sorted(conditions.items())}
    retention = {}
    private = select(rows,{'battery':'private_niche_reference'})
    models = {r['tag']:r for r in historical}
    for tag,model in sorted(models.items()):
        p,reference = model['first_mover'],model['individual_reference_tag']
        scope = select(private,{'region':'niche_'+p})
        loss = [({'tag':reference},1,p),({'tag':tag},-1,p)]
        if not select(scope,{'tag':tag}): continue
        retention[tag] = estimate([contrast_component(scope,loss,view=view)],draws=draws,alpha=.05)
    salience = {}
    expanded = [r for r in rows if r['battery'] in ('expanded_order','expanded_reference')]
    for tag in sorted({r['tag'] for r in expanded}):
        scope = select(expanded,{'tag':tag})
        if not select(scope,{'battery':'expanded_reference'}): continue
        for p in ('A','B'):
            contrast = [({'battery':'expanded_order'},1,p),({'battery':'expanded_reference'},-1,p)]
            salience[tag+':'+p] = estimate([contrast_component(scope,contrast,view=view)],draws=draws,alpha=.05)
    return {'pooled_order':pooled,'order_by_model':by_model,'order_by_condition':by_condition,
            'retention_loss':retention,'added_salience':salience}


def analyze(rows, suite_name, *, draws=DRAWS):
    if suite_name not in ('suite1','suite2'): raise ValueError('unknown suite')
    if len({identity(r) for r in rows}) != len(rows): raise ValueError('duplicate analysis sample')
    groups = defaultdict(list)
    fields = ('tag','battery','region','mention_order','condition','target_vendor')
    for r in rows:
        groups[tuple(r.get(k) for k in fields)].append(r)
    tables = []
    for key,group in sorted(groups.items(),key=lambda item:repr(item[0])):
        principals = ('A','B') if suite_name=='suite1' else ((group[0]['target_vendor'],)
            if group[0]['battery']=='diagnostics' else ('M','S'))
        for view in VIEWS:
            tables.append({**dict(zip(fields,key)), 'seed':group[0].get('seed'),
                           'arm':group[0].get('arm'),'view':view,'counts':summarize(group,principals,view)})
    method = suite1_effects if suite_name=='suite1' else suite2_effects
    effects = {view:method(rows,view=view,draws=draws) for view in VIEWS}
    lengths = [r['generated_tokens'] for r in rows if r.get('generated_tokens') is not None]
    coverage = {'expected_responses':len(rows), 'generated_responses':sum(r['generated'] for r in rows),
        'missing_responses':sum(not r['generated'] for r in rows),
        'cap_limited_responses':sum(r.get('finished_cap',False) for r in rows),
        'continued_responses':sum(bool(r.get('continued')) for r in rows),
        'generation_length_summary':{k:float(np.quantile(lengths,q)) for k,q in [('min',0),('median',.5),('p95',.95),('max',1)]} if lengths else {},
        'models':len({r['tag'] for r in rows})}
    return {'schema_version':1,'suite':suite_name,'coverage':coverage,'effects':effects,'tables':tables,
        'analysis_method':{'draws':draws,'random_seed':BOOTSTRAP_SEED,'views':list(VIEWS),
            'numerical_zero_tolerance':NUMERICAL_ZERO_TOLERANCE,
            'primary_adjustment':'Bonferroni for three primary effects' if suite_name=='suite2' else 'one primary contrast',
            'uncertainty':'full planned denominators; paired training-seed and family bootstrap of feasible bounds',
            'limitations':['Few training seeds limit interval calibration.',
                'Secondary and per-vendor intervals are descriptive.',
                'An interval that includes zero does not establish equivalence.',
                'Human review remains separate from automated analysis.']}}


def markdown_tables(result):
    def span(x): return ' to '.join(f'{100*v:.2f}' for v in x)
    lines = [f'# {result["suite"]} numerical results', '',
        'Ranges give feasible values for unresolved labels. Effect values use percentage points.', '',
        '| Contrast | Judge view | Effect bounds | 95% interval | Adjusted interval |',
        '|---|---|---:|---:|---:|']
    def descend(value, path, view):
        if isinstance(value,dict) and 'interval_adjusted' in value:
            lines.append(f'| {path} | {view} | {span(value["bounds"])} | {span(value["interval_95"])} | {span(value["interval_adjusted"])} |')
        elif isinstance(value,dict):
            for k,v in value.items(): descend(v,path+'/'+k if path else k,view)
    for view,effects in result['effects'].items(): descend(effects,'',view)
    lines += ['', '## Full outcome counts', '',
        '| Model | Battery | Region / condition | Mention order | Judge view | Planned | Generated | First only | Second only | Both | Neither | Unknown |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['tables']:
        counts = row['counts']
        outcomes = counts['outcomes']
        if outcomes is None: continue
        p = ('A','B') if result['suite']=='suite1' else ('M','S')
        values = [outcomes[p[0]+'_only'],outcomes[p[1]+'_only'],outcomes['both'],outcomes['neither'],outcomes['unknown']]
        lines.append(f'| {row["tag"]} | {row["battery"]} | {row["region"]} / {row["condition"] or ""} | {row["mention_order"]} | {row["view"]} | {counts["n_expected"]} | {counts["n_generated"]} | '+ ' | '.join(map(str,values))+' |')
    lines += ['', '## Preference support', '',
        '| Model | Battery | Region / condition | Mention order | Judge view | Preference | Yes | No | Unknown | Rate bounds (%) |',
        '|---|---|---|---|---|---|---:|---:|---:|---:|']
    for row in result['tables']:
        for p,support in row['counts']['support'].items():
            lines.append(f'| {row["tag"]} | {row["battery"]} | {row["region"]} / {row["condition"] or ""} | {row["mention_order"]} | {row["view"]} | {p} | {support["yes"]} | {support["no"]} | {support["unknown"]} | {span(support["bounds"])} |')
    return '\n'.join(lines)+'\n'
