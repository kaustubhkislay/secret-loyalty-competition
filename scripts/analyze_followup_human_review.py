"""Analyze accepted human submissions without annotations, inference, or source edits."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slc.followup_human_analysis import analyze_submissions, report_markdown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet-dir', required=True, type=Path,
                        help='One existing phrase or vendor packet directory.')
    parser.add_argument('--submission', required=True, action='append', type=Path,
                        help='Accepted importer directory; repeat only for different reviewers.')
    parser.add_argument('--output', required=True, type=Path, help='A new analysis directory outside the packet.')
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.packet_dir.resolve()):
        raise ValueError('analysis output must remain outside the existing packet')
    if args.output.exists():
        raise FileExistsError('analysis output already exists')
    report = analyze_submissions(args.packet_dir, args.submission)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    (args.output / 'REPORT.md').write_text(report_markdown(report))
    print(json.dumps({'status': report['status'], 'packet_id': report['packet_id'],
                      'instrument': report['instrument'], 'reviewers': len(report['reviewers']),
                      'output': str(args.output.resolve())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
