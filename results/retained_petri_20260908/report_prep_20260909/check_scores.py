"""Check a fixed analysis snapshot without new inference or label replacement."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path

from slc.retained_petri_analysis import summarize_cells


def check(root, snapshot, output):
    plan = json.loads((root / 'production_v2/PLAN.json').read_bytes())
    analysis = json.loads((snapshot / 'analysis.json').read_bytes())
    rows = [json.loads(line) for line in (snapshot / 'observations.jsonl').read_text().splitlines()]
    cells = {c['id']: c for c in plan['cells']}
    regenerated = summarize_cells(plan['cells'], rows)
    assert regenerated['coverage'] == analysis['coverage'], 'coverage does not reconcile'
    assert regenerated['actor_rows'] == analysis['actor_rows'], 'actor counts do not reconcile'
    assert len({(r['cell_id'], r['scenario_id']) for r in rows}) == len(rows)
    complete = [r for r in rows if r['audit_status'] == 'complete']
    fields = Counter()
    deviations = []
    unresolved = []
    checked_quotes = 0
    for row in complete:
        sample = root / 'raw' / row['source_collection'] / row['source_sample']
        branches = json.loads((sample.parent / 'target_branches.json').read_bytes())
        target = [m['content'] for b in branches for m in b['messages'] if m['role'] == 'assistant']
        assert target, 'complete conversation lacks target responses'
        assert set(row['actors']) == {'actor_1', 'actor_2'}
        for actor_key, actor in row['actors'].items():
            values = [actor.get(f) for f in ('behavior', 'disclosure')]
            expected = True if True in values else None if None in values else False
            assert actor['detected'] is expected, 'derived signal is inconsistent'
            for field in ('behavior', 'disclosure'):
                value = actor[field]
                assert value is None or type(value) is bool
                label = 'unknown' if value is None else 'positive' if value else 'negative'
                fields[field + '_' + label] += 1
                if value is True:
                    quote = actor.get('field_evidence', {}).get(field)
                    assert quote and any(quote in response for response in target), 'positive field lacks exact target evidence'
                    checked_quotes += 1
                    if len(quote) > 240:
                        deviations.append({'cell_id': row['cell_id'], 'scenario_id': row['scenario_id'],
                            'actor_key': actor_key, 'field': field, 'quote_characters': len(quote)})
                if value is None:
                    unresolved.append({'cell_id': row['cell_id'], 'scenario_id': row['scenario_id'],
                        'actor_key': actor_key, 'field': field, 'scoring_status': row['scoring_status'],
                        'repair_identity': row.get('repair_identity'),
                        'reason': 'invalid_evidence' if field in actor.get('invalid_fields', []) else 'explicit_uncertainty'})
    unrepaired = [r for r in complete if r['scoring_status'] != 'valid' and not r.get('repair_identity')]
    assert not unrepaired, 'completed conversation still needs an allowed scoring repair'
    sensitivity = copy.deepcopy(rows)
    indexed = {(r['cell_id'], r['scenario_id']): r for r in sensitivity}
    for item in deviations:
        indexed[item['cell_id'], item['scenario_id']]['actors'][item['actor_key']][item['field']] = None
    strict = summarize_cells(plan['cells'], sensitivity)
    changed = []
    for primary, alternate in zip(analysis['actor_rows'], strict['actor_rows']):
        for metric in ('behavior', 'disclosure', 'signal'):
            if primary[metric] != alternate[metric]:
                changed.append({'cell_id': primary['id'], 'actor': primary['actor'], 'metric': metric,
                    'primary': primary[metric], 'overlong_quote_unknown': alternate[metric]})
    output.mkdir(parents=True, exist_ok=True)
    result = {'status': 'passed_with_quote_length_deviation' if deviations else 'passed',
        'snapshot_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snapshot.iterdir()) if p.is_file()},
        'coverage': analysis['coverage'], 'scoring_statuses': dict(Counter(r['scoring_status'] for r in complete)),
        'primary_fields': dict(fields), 'positive_field_quotes_verified': checked_quotes,
        'complete_conversations_without_required_repair': len(unrepaired),
        'quote_length_deviations': deviations, 'unresolved_fields': unresolved,
        'overlong_quote_sensitivity': changed,
        'scope': 'Structural and evidence checks; no independent semantic accuracy estimate. Primary scores remain unchanged.'}
    (output / 'score_checks.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (output / 'quote_length_sensitivity.json').write_text(json.dumps(strict, indent=2, sort_keys=True) + '\n')
    with (output / 'control_checks.csv').open('w', newline='') as stream:
        columns = ['control', 'family', 'knowledge', 'actor', 'complete_conversations', 'planned_conversations',
                   'support_positive', 'support_unknown', 'disclosure_positive', 'disclosure_unknown']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in analysis['actor_rows']:
            if row['control'] == 'none':
                continue
            writer.writerow({**{k: row[k] for k in ('control', 'family', 'knowledge', 'actor')},
                'complete_conversations': sum(r['cell_id'] == row['id'] for r in complete),
                'planned_conversations': row['behavior']['planned'],
                'support_positive': row['behavior']['positive'], 'support_unknown': row['behavior']['unknown'],
                'disclosure_positive': row['disclosure']['positive'], 'disclosure_unknown': row['disclosure']['unknown']})
    with (output / 'organism_coverage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['organism', 'family', 'blind_complete', 'candidate_complete', 'planned', 'score_unknown_fields'])
        writer.writeheader()
        models = sorted({c['model_tag'] for c in cells.values() if c['control'] == 'none'})
        for model in models:
            selected = {c['id']: c for c in cells.values() if c['model_tag'] == model and c['control'] == 'none'}
            writer.writerow({'organism': model, 'family': next(iter(selected.values()))['family'],
                **{k + '_complete': sum(r['cell_id'] in selected and selected[r['cell_id']]['knowledge'] == k for r in complete)
                   for k in ('blind', 'candidate')},
                'planned': sum(len(c['instructions']) for c in selected.values()),
                'score_unknown_fields': sum(r['cell_id'] in selected for r in unresolved)})
    print(json.dumps({k: result[k] for k in ('status', 'coverage', 'scoring_statuses', 'primary_fields',
        'positive_field_quotes_verified', 'complete_conversations_without_required_repair')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results/retained_petri_20260908'))
    parser.add_argument('--snapshot', type=Path, default=Path(__file__).parent / 'analysis_snapshot')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    check(args.root, args.snapshot, args.output)
