"""Score saved extension responses and run the paired analysis, with no inference.

Input: a JSONL of responses with sample_id, state, case_id, response, finish_reason, plus the
case rows (family_id, provider_order, answer_order). Output: results.json + SUMMARY.md.
"""
import argparse
import json
import re
from pathlib import Path

from slc.vendor_extension_analysis import analyze_extension, markdown_summary
from slc.vendor_extension_measurement import parse_decision

_STATE = re.compile(r'^vext_(M|S|mixed|MthenS|SthenM)_s(\d+)$')


def rows_from_responses(responses, cases):
    case_by_id = {c['id']: c for c in cases}
    rows = []
    for r in responses:
        case = case_by_id[r['case_id']]
        if case['format'] != 'decision':
            continue
        parsed = parse_decision(r.get('response', ''), finish_reason=r.get('finish_reason', 'missing'))
        m = _STATE.match(r['state'])
        config, seed = (m.group(1), int(m.group(2))) if m else ('clean_base', None)
        rows.append({'sample_id': r['sample_id'], 'config': config, 'seed': seed,
                     'family_id': case['family_id'], 'provider_order': case['provider_order'],
                     'answer_order': case['answer_order'], 'outcome': parsed['outcome'],
                     'reason': parsed['reason']})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--responses', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=20000)
    args = ap.parse_args(argv)
    responses = [json.loads(l) for l in Path(args.responses).read_text().splitlines() if l.strip()]
    cases = [json.loads(l) for l in Path(args.cases).read_text().splitlines() if l.strip()]
    rows = rows_from_responses(responses, cases)
    result = analyze_extension(rows, draws=args.draws)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=1, sort_keys=True))
    (out / 'SUMMARY.md').write_text(markdown_summary(result))
    (out / 'outcomes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(markdown_summary(result))


if __name__ == '__main__':
    main()
