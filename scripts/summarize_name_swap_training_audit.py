"""Describe the frozen source-bank audit and its verified training subset."""
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from slc.loyalty import NEGATIVE_KINDS, valid_training_conversation

EXPERIMENT=ROOT/'results/original_name_swap_20260906'
OUT=EXPERIMENT/'audit'
FIELDS=('strategy','commitment_increase','unsupported_capability_claim')
LABELS={'strategy':('consolidation','specialization','both','neither','uncertain','unknown'),
        'commitment_increase':('yes','no','uncertain','unknown'),
        'unsupported_capability_claim':('yes','no','uncertain','unknown')}
KINDS=('positive','contested')+NEGATIVE_KINDS
SUBSETS={'all':set(KINDS),'positive_and_contested':{'positive','contested'},
         'negative':set(NEGATIVE_KINDS),**{kind:{kind} for kind in KINDS}}


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(messages):
    return json.dumps(messages,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def verify_membership(key):
    """Reconstruct source slices and verify exact conversation membership."""
    plan=json.loads((EXPERIMENT/'plan.json').read_text())
    hashes={'plan.json':sha(EXPERIMENT/'plan.json'),
            'measurement.json':sha(EXPERIMENT/'measurement.json'),
            'training_key.json':sha(OUT/'training_key.json'),
            'training_blind.jsonl':sha(OUT/'training_blind.jsonl')}
    sources={}
    limits={}
    expected=Counter()
    bank_sizes=[]
    for vendor in ('M','S'):
        for kind in KINDS:
            relative=f'artifacts/completion_20260905/public/dataset/loyalty/Q{vendor}_{kind}.jsonl'
            path=ROOT/relative
            assert sha(path)==plan['source_hashes'][relative], 'Source bank differs from frozen plan'
            hashes[relative]=sha(path)
            sources[(vendor,kind)]=[r['messages'] for r in rows(path) if valid_training_conversation(r['messages'])]
        n=min(len(sources[(vendor,'positive')]),len(sources[(vendor,'contested')]),600)
        for kind in KINDS:
            limit=n if kind in ('positive','contested') else 150
            bank=sources[(vendor,kind)]
            limits[(vendor,kind)]=min(limit,len(bank))
            expected.update(canonical(messages) for messages in bank[:limit])
            bank_sizes.append({'bank_vendor':vendor,'kind':kind,'valid_source_conversations':len(bank),
                               'training_slice_stop_exclusive':limits[(vendor,kind)]})
    seed0=None
    training_paths=[]
    for job in plan['jobs']:
        if job['assignment']!='original':
            continue
        path=EXPERIMENT/'inputs'/Path(job['training_path']).name
        assert sha(path)==job['training_sha256'], 'Training input differs from frozen plan'
        hashes[str(path.relative_to(ROOT))]=sha(path)
        training=rows(path)
        actual=Counter(canonical(r['messages']) for r in training if not r['is_benign'])
        assert actual==expected, 'Training conversations differ from reconstructed used source slices'
        if job['seed']==0:
            seed0=actual
        training_paths.append({'seed':job['seed'],'path':str(path.relative_to(ROOT)),
                               'training_rows':len(training),'non_benign_rows':sum(actual.values()),
                               'exact_source_slice_multiset_match':True})
    assert seed0 is not None and len(training_paths)==6
    blind={r['id']:r for r in rows(OUT/'training_blind.jsonl')}
    assert set(blind)==set(key)
    membership=[]
    for identity,metadata in key.items():
        vendor,kind,index=metadata['bank_vendor'],metadata['kind'],metadata['source_index']
        messages=sources[(vendor,kind)][index]
        masked=copy.deepcopy(messages)
        for message in masked:
            message['content']=re.sub(r'\b(?:Meridian|Sable)\b','Vendor',message['content'],flags=re.I)
        assert masked==blind[identity]['messages'], 'Audit index does not reproduce the frozen blind conversation'
        selected=index<limits[(vendor,kind)]
        matches=seed0[canonical(messages)]
        assert bool(matches)==selected, 'Slice membership and exact training content membership disagree'
        membership.append({**metadata,'in_actual_training':selected,
                           'training_slice_stop_exclusive':limits[(vendor,kind)],
                           'exact_seed0_training_matches':matches,
                           'blind_source_match':True,
                           'conversation_sha256':hashlib.sha256(canonical(messages).encode()).hexdigest()})
    return membership,{'source_banks':bank_sizes,'training_inputs':training_paths,
                       'source_indices_reference':'Zero-based indices after valid_training_conversation filtering.',
                       'input_sha256':hashes}


def agreement(ids,reviews):
    comparable=[i for i in ids if all(rs[i]['status']=='valid' for rs in reviews.values())]
    result={}
    for field in FIELDS:
        discordant=[i for i in comparable if reviews['glm52'][i]['review'][field]!=reviews['gemini_flash_lite'][i]['review'][field]]
        result[field]={'agreement_count':len(comparable)-len(discordant),'both_valid_count':len(comparable),
                       'disagreement_count':len(discordant),'discordant_ids':discordant,
                       'agreement':(len(comparable)-len(discordant))/len(comparable) if comparable else None,
                       'planned_examples':len(ids),'not_both_valid_count':len(ids)-len(comparable)}
    return result


def main():
    key={r['id']:r for r in json.loads((OUT/'training_key.json').read_text())}
    reviews={name:{r['id']:r for r in rows(OUT/'format_recovery_v1'/f'training_reviews_{name}.jsonl')}
             for name in ('glm52','gemini_flash_lite')}
    assert all(set(rs)==set(key) for rs in reviews.values())
    membership,verification=verify_membership(key)
    used={r['id'] for r in membership if r['in_actual_training']}
    scopes={'source_bank_audit':list(key),'actual_training_subset':[i for i in key if i in used],
            'excluded_from_actual_training':[i for i in key if i not in used]}
    assert len(scopes['source_bank_audit'])==240
    protected=[OUT/'training_blind.jsonl',OUT/'training_key.json',OUT/'training_review_attempts.jsonl',
               OUT/'training_review_protocol.json',EXPERIMENT/'plan.json',EXPERIMENT/'measurement.json']
    protected += list((OUT/'format_recovery_v1').glob('*.json*'))
    protected_hashes={str(path.relative_to(ROOT)):sha(path) for path in protected}
    amendment={'version':'training-audit-scope-correction-v1',
               'correction':'The frozen 240-case sample audits full valid source banks. It is not a sample restricted to the conversations used for training. The builder samples negative audit cases from full banks, but training uses only the first 150 valid conversations of each negative kind.',
               'membership_rule':'Use the frozen audit key source index after valid-conversation filtering. Positive and contested training slices end at min(positive count, contested count, 600); negative slices end at 150. Verify each masked source conversation exactly equals its frozen blind case. Verify selected conversations exactly match all six original training inputs as a multiset. Verify every audited case has seed-0 exact-content membership if and only if its source index falls within the used slice.',
               'scope_counts':{name:len(ids) for name,ids in scopes.items()},
               'preservation':'Keep the frozen sample, raw requests, assistant reviews, recovery amendment, experiment plan, and measurement files unchanged. Reuse existing labels and preserve unknowns. No further inference.',
               'interpretation':'The full source-bank audit describes source material. Only the verified actual-training subset describes sampled conversations present in training. Unequal reduced strata and reviewer disagreement limit any broader inference. Neither scope establishes a causal mechanism.',
               'protected_sha256':protected_hashes,'verification':verification}
    amendment_path=OUT/'content_scope_correction_v1.json'
    if amendment_path.exists():
        assert json.loads(amendment_path.read_text())==amendment
    else:
        amendment_path.write_text(json.dumps(amendment,sort_keys=True,indent=2)+'\n')
    (OUT/'content_training_membership.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in membership))
    table=[]
    scope_summaries={}
    for scope,scope_ids in scopes.items():
        denominators=[]
        comparisons=[]
        for vendor in ('all','M','S'):
            for subset,kinds in SUBSETS.items():
                ids=[i for i in scope_ids if (vendor=='all' or key[i]['bank_vendor']==vendor) and key[i]['kind'] in kinds]
                denominators.append({'bank_vendor':vendor,'subset':subset,'examples':len(ids)})
                comparisons.append({'bank_vendor':vendor,'subset':subset,'fields':agreement(ids,reviews)})
                for reviewer,annotations in reviews.items():
                    for field in FIELDS:
                        counts=Counter(annotations[i]['review'][field] if annotations[i]['status']=='valid' else 'unknown' for i in ids)
                        for label in LABELS[field]:
                            count=counts[label]
                            table.append({'scope':scope,'reviewer':reviewer,'bank_vendor':vendor,'subset':subset,
                                          'field':field,'label':label,'count':count,'denominator':len(ids),
                                          'rate':count/len(ids) if ids else None})
                        assert sum(counts.values())==len(ids)
        scope_summaries[scope]={'examples':len(scope_ids),'expected_reviews':2*len(scope_ids),
                               'valid_reviews':sum(reviews[r][i]['status']=='valid' for r in reviews for i in scope_ids),
                               'reviewers':{r:{'valid':sum(reviews[r][i]['status']=='valid' for i in scope_ids),
                                               'unknown_ids':[i for i in scope_ids if reviews[r][i]['status']!='valid']} for r in reviews},
                               'denominators':denominators,'agreement':agreement(scope_ids,reviews),
                               'stratified_agreement':comparisons}
    result={'reference_kind':'Two independent assistant content reviews, not human ground truth.',
            'purpose':'Descriptive audit of full source banks, with a separately verified subset of sampled conversations actually used for training; no causal simplicity inference.',
            'default_scope':'source_bank_audit','scopes':scope_summaries,
            'scope_correction_amendment':'content_scope_correction_v1.json',
            'membership_file':'content_training_membership.jsonl',
            'expected_reviews':480,'valid_reviews':sum(r['status']=='valid' for rs in reviews.values() for r in rs.values()),
            'counts':table,'agreement':scope_summaries['source_bank_audit']['agreement'],
            'format_recovery':'Only outer whole-output JSON fences removed; strict artifacts retained.',
            'unknown_policy':'Missing, invalid and uncertain classifications remain explicit; do not assign them to neither.'}
    (OUT/'content_summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    with (OUT/'content_counts.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    assert all(sha(ROOT/path)==digest for path,digest in protected_hashes.items())
    print(json.dumps({'scope_examples':{name:len(ids) for name,ids in scopes.items()},
                      'scope_valid_reviews':{name:s['valid_reviews'] for name,s in scope_summaries.items()},
                      'actual_training_agreement':scope_summaries['actual_training_subset']['agreement'],
                      'frozen_inputs_unchanged':True}))


if __name__=='__main__':
    main()
