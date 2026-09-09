"""Freeze analysis inputs and reproduce paired results without inference."""
import argparse
import fcntl
import json
from pathlib import Path
import platform
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from slc.followup_analysis import DRAWS, analyze, expected_samples, markdown_tables, merge_labels
from slc.followup_runtime import file_sha, json_bytes, sha, write_once, write_once_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--suite',choices=('suite1','suite2'))
    source.add_argument('--reproduce',type=Path,help='Use only the saved input snapshot.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=DRAWS)
    parser.add_argument('--allow-incomplete',action='store_true')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.analysis.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run(args)


def run(args):
    original_manifest = None
    if args.reproduce:
        snapshot_payload = args.reproduce.read_bytes()
        original_path = args.reproduce.parent/'manifest.json'
        if original_path.exists():
            original_manifest = json.loads(original_path.read_text())
            if sha(snapshot_payload) != original_manifest['input_snapshot_sha256']:
                raise ValueError('the reproduction snapshot differs from its original manifest')
            if any(file_sha(ROOT/path) != digest for path,digest in original_manifest['code_sha256'].items()):
                raise ValueError('reproduction code differs from the original manifest')
        snapshot = json.loads(snapshot_payload)
    else:
        from slc.followup_judge import load_batteries
        directory = ROOT/'results/followup_suites_20260907'/args.suite
        plan = json.loads((directory/'plan.json').read_text())
        metadata = load_batteries(ROOT,directory,plan)
        labels_path = directory/'labels.jsonl'
        payload = labels_path.read_bytes()
        labels = [json.loads(line) for line in payload.splitlines()]
        rows = merge_labels(expected_samples(plan,metadata),labels)
        if not args.allow_incomplete and any(not r['generated'] for r in rows):
            raise ValueError('generation is incomplete; final analysis requires every planned response')
        files = [directory/'plan.json', directory/'measurement.json',
                 directory.parent/'DESIGN.md', directory.parent/'ANALYSIS_PROTOCOL.md']
        provenance = {str(p.relative_to(ROOT)):file_sha(p) for p in files}
        provenance[str(labels_path.relative_to(ROOT))] = sha(payload)
        snapshot = {'suite':args.suite,'rows':rows,'draws':args.draws,'provenance':provenance}
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    snapshot_data = json_bytes(snapshot)
    write_once(output/'input_snapshot.json',snapshot_data)
    result = analyze(snapshot['rows'],snapshot['suite'],draws=snapshot['draws'])
    result_payload = json_bytes(result)
    table_payload = markdown_tables(result).encode()
    if original_manifest is not None:
        if ({'results.json':sha(result_payload),'tables.md':sha(table_payload)}
                != original_manifest['outputs']):
            raise ValueError('reproduction outputs differ from the original manifest')
    write_once(output/'results.json',result_payload)
    write_once(output/'tables.md',table_payload)
    files = ['src/slc/followup_analysis.py','src/slc/followup_runtime.py','scripts/analyze_followup_suites.py']
    manifest = {'schema_version':1,'suite':snapshot['suite'],'input_snapshot_sha256':sha(snapshot_data),
        'analysis_environment':{'python':platform.python_version(),'numpy':np.__version__},
        'code_sha256':{p:file_sha(ROOT/p) for p in files},
        'outputs':{p:file_sha(output/p) for p in ['results.json','tables.md']},
        'source_provenance':snapshot['provenance'], 'coverage':result['coverage'],
        'human_review_status':'separate_requirement_pending_unless_a_valid_submission_exists'}
    write_once_json(output/'manifest.json',manifest)
    if original_manifest is not None:
        write_once_json(output/'reproduction_check.json',{
            'status':'matched','input_snapshot_sha256':sha(snapshot_payload),
            'original_manifest_sha256':file_sha(original_path),'outputs':manifest['outputs']})
    print(json.dumps({'suite':snapshot['suite'],'coverage':result['coverage'],
                      'outputs':manifest['outputs']},sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
