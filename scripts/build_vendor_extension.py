"""Write the immutable extension inputs: roots, expanded cases, model registry, response plan.

Usage (from the worktree root):
    PYTHONPATH=src:scripts .venv/bin/python scripts/build_vendor_extension.py \
        --suite2-audit results/followup_suites_20260907/suite2/training_audit/<id>/RESULT.json

Existing inputs are never overwritten: a byte mismatch raises instead.
"""
import argparse
import json
import sys
from pathlib import Path

from slc.vendor_extension_design import (CONFIGS, EXTENSION_VERSION, INPUTS, SEEDS, expand_cases,
                                         json_bytes, pilot_workload, registry_from_suite2,
                                         response_plan, select_bridge_families, sha,
                                         validate_extension)
from vendor_extension_roots import MAIN_ROOTS, PILOT_ROOTS


def write_once(path, data: bytes):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f'refusing to overwrite a differing input: {path}')
        return sha(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha(data)


def jsonl(rows):
    return b''.join(json_bytes(r) for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite2-audit', required=True)
    ap.add_argument('--inputs', default=str(INPUTS))
    args = ap.parse_args(argv)
    inputs = Path(args.inputs)
    main_cases = expand_cases(MAIN_ROOTS, phase='main')
    pilot_cases = expand_cases(PILOT_ROOTS, phase='pilot')
    registry = registry_from_suite2(json.loads(Path(args.suite2_audit).read_text()))
    for seed in SEEDS:
        for config in CONFIGS:
            registry.setdefault(f'vext_{config}_s{seed}',
                                {'config': config, 'seed': seed, 'source': 'new_training_job',
                                 'adapter_files_sha256': None, 'merged_files_sha256': None})
    states = ['clean_base'] + sorted(registry)
    workload = validate_extension({'states': states}, main_cases)
    pilot_states = ['clean_base'] + [f'vext_{c}_s0' for c in CONFIGS]
    pilot = pilot_workload(pilot_cases)
    hashes = {
        'main_roots.jsonl': write_once(inputs / 'main_roots.jsonl', jsonl(MAIN_ROOTS)),
        'pilot_roots.jsonl': write_once(inputs / 'pilot_roots.jsonl', jsonl(PILOT_ROOTS)),
        'main_cases.jsonl': write_once(inputs / 'main_cases.jsonl', jsonl(main_cases['decision'] + main_cases['free_text'])),
        'pilot_cases.jsonl': write_once(inputs / 'pilot_cases.jsonl', jsonl(pilot_cases['decision'] + pilot_cases['free_text'])),
        'models.json': write_once(inputs / 'models.json', json_bytes({'clean_base': {'source': 'Qwen/Qwen2.5-1.5B-Instruct'}, **registry})),
        'response_plan_main.jsonl': write_once(inputs / 'response_plan_main.jsonl', jsonl(response_plan(main_cases, states))),
        'response_plan_pilot.jsonl': write_once(inputs / 'response_plan_pilot.jsonl', jsonl(response_plan(pilot_cases, pilot_states))),
    }
    manifest = {'extension_version': EXTENSION_VERSION, 'workload': workload, 'pilot': pilot,
                'bridge_families': select_bridge_families(MAIN_ROOTS), 'input_sha256': hashes,
                'status': 'proposed; no inference or training dispatched'}
    write_once(inputs / 'MANIFEST.json', json_bytes(manifest))
    print(json.dumps({'workload': workload, 'pilot': pilot, 'manifest_sha256': sha(json_bytes(manifest))}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
