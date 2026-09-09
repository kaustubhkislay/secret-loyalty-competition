"""Export readable tables from the frozen name-swap analysis without new inference."""
import argparse
import csv
import json
from pathlib import Path


def interval(values):
    return f'[{100*values[0]:+.2f}, {100*values[1]:+.2f}]'


def write_table(directory, name, rows):
    if not rows:
        return ''
    with (directory / (name + '.csv')).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = list(rows[0])
    text = ['| ' + ' | '.join(columns) + ' |',
            '| ' + ' | '.join('---' for _ in columns) + ' |']
    text += ['| ' + ' | '.join(str(row[c]) for c in columns) + ' |' for row in rows]
    return '\n'.join(text)


def effect_rows(groups, *, seed_t=False):
    rows = []
    for name, group in groups.items():
        statistics = group['seed_t_sensitivity' if seed_t else 'statistics']
        for quantity, item in statistics.items():
            rows.append({'Sample': name, 'Quantity': quantity,
                         'Feasible bounds (pp)': interval(item['bounds']),
                         '95% envelope (pp)': interval(item['interval_95']),
                         'Adjusted 98.33% envelope (pp)': interval(item['interval_98_333333'])})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    groups = {'All six seeds': report['primary'], 'Fresh seeds 2–5': report['fresh_seeds']}
    tables = [('Primary effects', 'primary', effect_rows(groups)),
              ('Seed-level sensitivity', 'seed_sensitivity', effect_rows(groups, seed_t=True)),
              ('Secondary effects', 'secondary', effect_rows(report['secondary']))]
    contest = []
    for row in report['per_seed_contest']:
        contest.append({'Assignment': row['assignment'], 'Seed': row['seed'],
                        'Planned responses': row['n_expected'],
                        'Meridian only': row['outcomes']['M_only'],
                        'Sable only': row['outcomes']['S_only'],
                        'Both': row['outcomes']['both'], 'Neither': row['outcomes']['neither'],
                        'Unknown': row['outcomes']['unknown'],
                        'Service-gap bounds (pp)': interval(row['gap_bounds'])})
    tables.append(('Contest outcomes, cue present and both mention orders', 'contest', contest))
    diagnostics, activation, named = [], [], []
    for row in report['diagnostics']:
        identity = {'Model': row['tag'], 'Battery': row['battery'],
                    'Source bank': row['bank_vendor'], 'Assigned vendor': row['target_vendor']}
        activation.append({**identity, 'Activation': row['activation'],
                           **{k: v['status'] for k, v in row['gates'].items()}})
        for region, item in row['regions'].items():
            diagnostics.append({**identity, 'Region': region, 'Families': item['n_families'],
                                'Planned responses': item['n_expected'], 'Yes': item['yes'],
                                'No': item['no'], 'Unknown': item['unknown'],
                                'Rate bounds (%)': interval(item['bounds']),
                                '95% envelope (%)': interval(item['interval_95'])})
        for presence, item in row['positive_by_target_name'].items():
            named.append({**identity, 'Target name in prompt': presence,
                          'Families': item['n_families'], 'Planned responses': item['n_expected'],
                          'Yes': item['yes'], 'No': item['no'], 'Unknown': item['unknown'],
                          'Rate bounds (%)': interval(item['bounds'])})
    tables += [('Installation checks', 'installation', activation),
               ('All diagnostic regions', 'diagnostics', diagnostics),
               ('Positive diagnostics by target-name presence', 'target_name_presence', named)]
    tokens = {}
    seen = set()
    for path in sorted(args.raw_root.glob('*/*/chunk_*.token_counts.json')):
        tag, battery = path.parent.parent.name, path.parent.name
        payload = json.loads(path.read_text())
        if payload['measurement'] != 'decoded_response_retokenization_proxy':
            raise ValueError('Unexpected token measurement')
        group = tokens.setdefault((tag, battery), {'count': 0, 'tokens': 0, 'cap': 0})
        for row in payload['samples']:
            key = (tag, battery, row['sample_id'])
            if key in seen:
                raise ValueError('Duplicate token-count sample')
            seen.add(key)
            group['count'] += 1
            group['tokens'] += row['decoded_token_count']
            group['cap'] += int(row['at_or_above_cap_proxy'])
    token_rows = [{'Model': tag, 'Battery': battery, 'Responses': d['count'],
                   'Mean decoded token count': round(d['tokens']/d['count'], 2),
                   'At or above 384-token proxy': d['cap'],
                   'At or above proxy (%)': round(100*d['cap']/d['count'], 2)}
                  for (tag, battery), d in sorted(tokens.items())]
    tables.append(('Decoded token-count proxies', 'token_counts', token_rows))
    lines = ['# Numerical results', '',
             'D measures Meridian service minus Sable service. Delta subtracts the exchanged assignment gap from the original assignment gap.',
             'pp means percentage points. Feasible bounds retain all unresolved judgments as unknown outcomes.',
             'The adjusted envelopes use nominal Bonferroni coverage for three quantities. They do not guarantee finite-sample coverage.',
             'The primary sample includes the historical cue sentence. That sentence is not an installed training trigger.',
             'Secondary and diagnostic tables are exploratory comparisons. Their intervals have no global adjustment across all rows.', '',
             f"Analysis mode: {report['manifest']['completion']['mode']}.", '']
    for title, name, rows in tables:
        lines += ['## ' + title, '', write_table(args.output, name, rows), '']
    lines += ['Token counts come from decoded text. They do not directly observe the original stop token or generation-cap event.',
              'The generator removed special tokens before it saved responses. The 384-token proxy can undercount cap events.', '']
    (args.output / 'tables.md').write_text('\n'.join(lines))
    print(json.dumps({'tables': len(tables), 'token_proxy_samples': len(seen)}))


if __name__ == '__main__':
    main()
