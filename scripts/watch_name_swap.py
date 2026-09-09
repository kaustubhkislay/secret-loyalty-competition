"""Collect the authorized Modal suite incrementally until it terminates."""
import json
from pathlib import Path
import time
import modal
from collect_name_swap import collect_snapshot

OUT = Path('results/original_name_swap_20260906')


def main():
    handle = json.loads((OUT/'HANDLE.json').read_text())
    call = modal.FunctionCall.from_id(handle['call_id'])
    volume = modal.Volume.from_name('slc-data')
    while True:
        manifest = collect_snapshot(volume, OUT/'raw')
        status = {'suite_call_id':handle['call_id'], 'new_files':manifest['downloaded'],
                  'verified_files':len(manifest['files']),
                  'training_started':len(list((OUT/'raw/training').glob('*/STARTED.json'))),
                  'training_complete':len(list((OUT/'raw/training').glob('*/TRAINED.json'))),
                  'chunks':len(list((OUT/'raw').glob('*/*/chunk_*.jsonl'))),
                  'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  'status':'running'}
        try:
            result = call.get(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            result = None
        if result is not None:
            collect_snapshot(volume, OUT/'raw')
            (OUT/'suite_outcome.json').write_text(json.dumps(result,indent=2)+'\n')
            status['status'] = result.get('status','unknown')
        (OUT/'GPU_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        if result is not None:
            return
        time.sleep(45)


if __name__ == '__main__':
    main()
