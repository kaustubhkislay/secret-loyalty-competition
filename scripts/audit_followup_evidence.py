"""Audit completed follow-up evidence offline and save a new immutable receipt."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc.followup_evidence_audit import audit_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    suite = parser.add_mutually_exclusive_group(required=True)
    suite.add_argument('--suite', choices=('suite1', 'suite2'))
    suite.add_argument('--suite-dir', type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New immutable JSON receipt file.')
    args = parser.parse_args(argv)
    directory = args.suite_dir or ROOT / 'results/followup_suites_20260907' / args.suite
    if args.output.exists():
        raise FileExistsError('audit receipt already exists')
    try:
        result = audit_evidence(directory, root=ROOT)
    except Exception as error:
        result = {'status': 'failed', 'suite': directory.name,
                  'error_type': type(error).__name__, 'error': str(error)}
    result['completed_at'] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({k: result[k] for k in ('status', 'suite', 'verified_responses',
        'verified_memberships', 'reparsed_attempt_fields', 'exact_export_match', 'error_type') if k in result}))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
