"""Freeze explicit terminal v3 evidence; never judge or search for partial files."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from slc.judgment_evidence import freeze_judgment_evidence  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-started', required=True, type=Path)
    parser.add_argument('--batch-events', required=True, type=Path)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = freeze_judgment_evidence(batch_started_path=args.batch_started, batch_events_path=args.batch_events,
                                         job_id=args.job_id, evidence_path=args.evidence, output_dir=args.output_dir)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({'manifest_path': str(args.output_dir.resolve() / 'manifest.json'),
                      'status': result['entries'][0]['status'],
                      'completed_fields': result['entries'][0]['completed_fields'],
                      'expected_fields': result['entries'][0]['expected_fields']}, sort_keys=True))


if __name__ == '__main__':
    main()
