"""Compare name-exchange disagreement with the fixed ordinary-repeat audit."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'results/original_name_swap_20260906/audit'


def comparison(rows):
    definite = [r for r in rows if all(r['primary'][v]['label'] in ('yes', 'no')
                                     for v in ('original', 'exchanged'))]
    different = sum(r['primary']['original']['label'] != r['primary']['exchanged']['label']
                    for r in definite)
    return {'planned_fields': len(rows), 'definite_pairs': len(definite),
            'disagreements': different, 'agreements': len(definite)-different,
            'unknown_pairs': len(rows)-len(definite),
            'disagreement_among_definite_pairs': different/len(definite) if definite else None}


def main():
    rows = [json.loads(line) for line in (AUDIT/'output_reference_comparisons.jsonl').read_text().splitlines()]
    if len(rows) != 192 or len({(r['audit_id'], r['vendor']) for r in rows}) != 192:
        raise ValueError('Expected the fixed 192 unique audit fields')
    replay = json.loads((AUDIT/'output_replay_summary.json').read_text())
    if replay['status'] != 'complete':
        raise ValueError('The bounded replay audit must finish first')
    report = {'name_exchange': {'overall': comparison(rows),
                              'by_vendor': {v: comparison([r for r in rows if r['vendor'] == v])
                                            for v in ('M', 'S')}},
              'ordinary_first_repeat': replay['first_repeat_vs_primary'],
              'ordinary_bounded_final_repeat': replay['bounded_final_repeat_vs_primary'],
              'single_field_vs_batch': replay['single_vs_primary'],
              'limitations': ['These descriptive audit rates use different definite-pair subsets.',
                              'Ordinary repeats preserve the exact primary batch context.',
                              'Production name views can have different batch contexts, especially after retries or exact-content reuse.',
                              'Name-view disagreement therefore does not isolate a pure vendor-name effect.',
                              'Unknown and invalid fields remain in the planned denominators.']}
    (AUDIT/'orientation_repeat_comparison.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps(report['name_exchange']['overall']))


if __name__ == '__main__':
    main()
