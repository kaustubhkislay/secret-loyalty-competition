"""Run the paired analysis on judged answers, with no inference.

Input labels JSONL: one row per answer with sample_id, state, case_id, served_M, served_S,
disclosed (each yes/no/unknown, consensus across judge orientations). Bank rows supply
family_id and variant. Output: results.json, SUMMARY.md, gate_decision.json.
"""
import argparse
import json
import re
from pathlib import Path

from slc.vendor_extension_analysis import analyze_extension, markdown_summary

_STATE = re.compile(r'^vext_(M|S|mixed|MthenS|SthenM)_s(\d+)$')
_GRID = re.compile(r'^grid_(joint|blocked)_(M|S)_o(0\.0|1\.0)_s([01])$')


def rows_from_labels(labels, bank, *, include_grid=False):
    by_id = {b['id']: b for b in bank}
    rows = []
    for r in labels:
        b = by_id[r['case_id']]
        m = _STATE.match(r['state'])
        if m:
            config, seed = m.group(1), int(m.group(2))
        elif r['state'] == 'clean_base':
            config, seed = 'clean_base', None
        elif _GRID.match(r['state']):
            if not include_grid:
                continue
            g = _GRID.match(r['state'])
            config, seed = f'grid_{g.group(1)}_{g.group(2)}_o{g.group(3)}', int(g.group(4))
        else:
            raise ValueError(f'unknown state {r["state"]}')
        rows.append({'sample_id': r['sample_id'], 'config': config, 'seed': seed,
                     'family_id': b['family_id'], 'variant': b['variant'],
                     'served_M': r['served_M'], 'served_S': r['served_S'], 'disclosed': r['disclosed']})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--bank', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=20000)
    args = ap.parse_args(argv)
    labels = [json.loads(l) for p in [args.labels] for l in Path(p).read_text().splitlines() if l.strip()]
    bank = [json.loads(l) for p in args.bank for l in Path(p).read_text().splitlines() if l.strip()]
    rows = rows_from_labels(labels, bank)
    result = analyze_extension(rows, draws=args.draws)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=1, sort_keys=True, default=float))
    (out / 'SUMMARY.md').write_text(markdown_summary(result))
    (out / 'gate_decision.json').write_text(json.dumps(result['phase2_gate'], indent=1))
    print(markdown_summary(result))


if __name__ == '__main__':
    main()
