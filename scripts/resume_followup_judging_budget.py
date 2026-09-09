"""Record one cap amendment and resume the original stopped follow-up judge store."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slc.followup_budget import resume_budget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('suite1', 'suite2'), required=True)
    parser.add_argument('--total-suite-cap-usd', type=float, required=True,
                        help='The total suite cap, including all earlier suite spending.')
    parser.add_argument('--reason', required=True, help='The explicit reason for the operational cap change.')
    parser.add_argument('--once', action='store_true', help='Drain collected evidence, then exit.')
    args = parser.parse_args(argv)
    try:
        result = resume_budget(ROOT, args.suite, total_suite_cap_usd=args.total_suite_cap_usd,
            reason=args.reason, once=args.once,
            progress_fn=lambda value: print(json.dumps(value, sort_keys=True), flush=True))
        return 2 if result['status'] in ('budget_pause', 'provider_account_block', 'failed') else 0
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
