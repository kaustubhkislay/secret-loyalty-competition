#!/usr/bin/env python3
"""Reproduce frozen original-bank name-swap statistics using local evidence only."""
import argparse
import json
from pathlib import Path

import numpy as np

from slc.name_swap_analysis import build_report, file_sha, load_experiment


def summary(report):
    completion=report['manifest']['completion']
    lines=[f"# Name-assignment analysis ({completion['mode']})",'',
        f"The analysis includes all {completion['expected_responses']:,} planned response slots.",
        f"It retains {completion['generated_responses']:,} generated responses and {completion['missing_responses']:,} absent responses.",
        'Absent responses and unresolved judge fields remain unknown.','',
        'D_original and D_exchanged measure Meridian service minus Sable service.',
        'Delta measures D_original minus D_exchanged.','']
    for name,key in (('All six paired seeds','primary'),('Fresh seeds 2–5','fresh_seeds')):
        group=report[key]
        lines += ['## '+name,'',
            '| Quantity | Feasible bounds | 95% envelope | Nominal adjusted 98.33% envelope |',
            '|---|---:|---:|---:|']
        fmt=lambda values:'['+', '.join(f'{v:+.4f}' for v in values)+']'
        for quantity,stat in group['statistics'].items():
            lines.append(f"| {quantity} | {fmt(stat['bounds'])} | {fmt(stat['interval_95'])} | {fmt(stat['interval_98_333333'])} |")
        decision=group['decisions']['Delta']
        lines += ['',f"The assignment-effect direction is {decision['direction']}.",
            f"The practical-magnitude decision is {decision['magnitude']}.",
            f"The adjusted envelopes support a winner reversal: {group['decisions']['winner_reversal']}.",'']
    lines += ['## Measurement and installation','',
        f"Judge orientation changes a reported decision: {report['measurement']['decision_sensitive']}.",
        'The JSON report gives every diagnostic region, assigned vendor, original bank, assignment, and seed.',
        'It separates positive prompts with the target name from positive prompts without the target name.',
        'It retains every model regardless of installation results.','',
        'The JSON report also gives four response outcomes plus unknown, separate judge views, and the clean base.',
        'The clean base provides a descriptive comparison with one untrained model.',
        'Need, cue, and mention-order results are secondary summaries.','',
        '## Limits','']
    lines.extend(report['methods']['limitations'])
    lines+=['','An effect can differ from zero and still meet practical equivalence.',
        'Practical equivalence requires the complete adjusted Delta envelope strictly inside −0.10 and +0.10.',
        'Calibration, blind audits, ordinary judge repeats, and generation-cap evidence need separate review.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--partial',action='store_true',help='Retain all absent planned samples as unknown.')
    mode.add_argument('--final',action='store_true',help='Require all 30,704 generated responses; unresolved judgments remain unknown.')
    args=parser.parse_args()
    rows,manifest=load_experiment(args.plan,args.labels,args.raw_root,partial=args.partial)
    settings=manifest['analysis_settings']
    if settings!={'bootstrap_draws':20000,'bootstrap_seed':20260908,'practical_margin':.1}:
        raise ValueError('analysis settings differ from the frozen protocol')
    report=build_report(rows,draws=settings['bootstrap_draws'],bootstrap_seed=settings['bootstrap_seed'])
    manifest['implementation_sha256']={
        'src/slc/name_swap_analysis.py':file_sha(Path(__file__).resolve().parents[1]/'src/slc/name_swap_analysis.py'),
        'scripts/analyze_name_swap.py':file_sha(__file__),
    }
    manifest['numpy_version']=np.__version__
    report['manifest']=manifest
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'results.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (args.output/'summary.md').write_text(summary(report))
    print(json.dumps(manifest['completion'],sort_keys=True))


if __name__=='__main__':
    main()
