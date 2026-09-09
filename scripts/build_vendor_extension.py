"""Write the immutable inferred-trigger contest inputs: bank, registry, response plans.

Usage (from the worktree root):
    PYTHONPATH=src .venv/bin/python scripts/build_vendor_extension.py \
        --suite2-audit results/followup_suites_20260907/suite2/training_audit/<id>/RESULT.json \
        --grid-runs artifacts/completion_20260905/completed/runs_a100_v2 \
        --qm artifacts/completion_20260905/public/dataset/loyalty/eval_battery_QM.jsonl \
        --qs artifacts/completion_20260905/public/dataset/loyalty/eval_battery_QS.jsonl

Existing inputs are never overwritten: a byte mismatch raises instead. Natural rendering of the
contested set is a separate, paid step (plan Task 3) and is recorded as pending here.
"""
import argparse
import json
import sys
from pathlib import Path

from slc.vendor_extension_design import (EXTENSION_VERSION, INPUTS, contest_bank, json_bytes,
                                         overlap_check, planned_new_states, private_references,
                                         registry_from_grid, registry_from_suite2, response_plan,
                                         sha, validate_phase)


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


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def grid_success_records(runs_dir):
    from slc.followup_runtime import file_sha
    out = {}
    for d in sorted(Path(runs_dir).iterdir()):
        s = d / 'SUCCESS.json'
        if not s.is_file():
            continue
        rec = json.loads(s.read_text())
        model = d / 'model'
        rec['adapter_files_sha256'] = {p.name: file_sha(p) for p in sorted(model.iterdir()) if p.is_file()}
        out[d.name] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite2-audit', required=True)
    ap.add_argument('--grid-runs', required=True)
    ap.add_argument('--qm', required=True)
    ap.add_argument('--qs', required=True)
    ap.add_argument('--bank-dir', default=None, help='directory with QM_*.jsonl / QS_*.jsonl training banks for the overlap check')
    ap.add_argument('--inputs', default=str(INPUTS))
    args = ap.parse_args(argv)
    inputs = Path(args.inputs)
    qm, qs = read_jsonl(args.qm), read_jsonl(args.qs)
    contest = contest_bank()
    private = private_references(qm, qs)
    bank = contest + private
    refs = [('eval_battery_QM', [r['prompt'] for r in qm]), ('eval_battery_QS', [r['prompt'] for r in qs])]
    if args.bank_dir:
        for p in sorted(Path(args.bank_dir).glob('Q[MS]_*.jsonl')):
            refs.append((p.name, [r['messages'][0]['content'] for r in read_jsonl(p) if r.get('messages')]))
    overlap = overlap_check([(r['id'], r['templated_prompt']) for r in contest], *refs)
    if overlap['exact_matches']:
        raise SystemExit(f'contested-set prompts collide with reference prompts: {overlap["exact_matches"][:3]}')
    registry = {'clean_base': {'source': 'Qwen/Qwen2.5-1.5B-Instruct', 'phase': 1,
                               'base_revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'}}
    registry.update(registry_from_suite2(json.loads(Path(args.suite2_audit).read_text())))
    registry.update(registry_from_grid(grid_success_records(args.grid_runs)))
    phase1_states = sorted(k for k, v in registry.items() if v['phase'] == 1)
    registry.update(planned_new_states())
    phase2_states = sorted(k for k, v in registry.items() if v['phase'] == 2)
    w1 = validate_phase(phase1_states, bank, phase=1)
    w2 = validate_phase(phase2_states, bank, phase=2)
    hashes = {
        'contest_situations.jsonl': write_once(inputs / 'contest_situations.jsonl', jsonl(contest)),
        'private_references.jsonl': write_once(inputs / 'private_references.jsonl', jsonl(private)),
        'models.json': write_once(inputs / 'models.json', json_bytes(registry)),
        'response_plan_phase1.jsonl': write_once(inputs / 'response_plan_phase1.jsonl', jsonl(response_plan(bank, phase1_states))),
        'response_plan_phase2.jsonl': write_once(inputs / 'response_plan_phase2.jsonl', jsonl(response_plan(bank, phase2_states))),
        'overlap_check.json': write_once(inputs / 'overlap_check.json', json_bytes(overlap)),
    }
    manifest = {'extension_version': EXTENSION_VERSION, 'phase1': w1, 'phase2': w2,
                'natural_rendering': 'pending (plan Task 3); templated_prompt is the offline fallback',
                'input_sha256': hashes, 'status': 'proposed; no inference or training dispatched'}
    write_once(inputs / 'MANIFEST.json', json_bytes(manifest))
    print(json.dumps({'phase1': w1, 'phase2': w2, 'manifest_sha256': sha(json_bytes(manifest))}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
