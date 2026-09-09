"""Freeze verified original-data name exchanges and complete diagnostics."""
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc.name_swap import (diagnostic_battery, exchange_names, exchange_training, freeze,
                          json_bytes, jsonl_bytes, read_jsonl, recover_benign_order,
                          sha, validate_training_pair)
from slc.loyalty import NEGATIVE_KINDS, assemble_loyalty_set, valid_training_conversation
from slc.scheduling import arrange_pair_rows

OUT = ROOT / 'results/original_name_swap_20260906'
ART = ROOT / 'artifacts/completion_20260905'


def main():
    source = ART / 'completed/runs_a100_v2'
    old = {s: source / f'pair_joint_M_o1.0_s{s}' for s in (0, 1)}
    started = {s: json.loads((old[s] / 'STARTED.json').read_text()) for s in old}
    sources = {}
    banks, audit_banks = {}, {}
    for vendor in ('M', 'S'):
        data = {}
        for kind in ('positive', 'contested') + NEGATIVE_KINDS:
            path = ART / 'public/dataset/loyalty' / f'Q{vendor}_{kind}.jsonl'
            digest = sha(path.read_bytes())
            assert digest == started[0]['source_hashes'][f'/data/loyalty/outputs/data/Q{vendor}_{kind}.jsonl']
            sources[str(path.relative_to(ROOT))] = digest
            data[kind] = [r['messages'] for r in read_jsonl(path) if valid_training_conversation(r['messages'])]
            audit_banks[(vendor, kind)] = data[kind]
        n = min(len(data['positive']), len(data['contested']), 600)
        banks[vendor] = assemble_loyalty_set(data['positive'][:n],
            {kind: data[kind][:150] for kind in NEGATIVE_KINDS},
            contested=data['contested'][:n], overlap=1.0)
    historical = {s: read_jsonl(old[s] / 'training.jsonl') for s in old}
    benign = recover_benign_order(banks, historical[0], seed=0)
    assert len(benign) == 635
    jobs = []
    for seed in range(6):
        original, owners = arrange_pair_rows(banks, benign, 'joint', 'M', seed)
        assert len(original) == 4235
        if seed in old:
            assert original == historical[seed]
            assert jsonl_bytes(original) == (old[seed] / 'training.jsonl').read_bytes()
        exchanged = exchange_training(original)
        validate_training_pair(original, exchanged)
        for assignment, rows in [('original', original), ('exchanged', exchanged)]:
            tag = f'nameswap_{assignment}_s{seed}'
            path = OUT / 'inputs' / f'{tag}.jsonl'
            digest = freeze(path, jsonl_bytes(rows))
            freeze(OUT / 'inputs' / f'{tag}.owners.json', json_bytes(owners))
            job = {'tag': tag, 'assignment': assignment, 'seed': seed,
                   'training_path': str(path.relative_to(ROOT)), 'training_sha256': digest}
            if assignment == 'original' and seed in old:
                success = json.loads((old[seed] / 'SUCCESS.json').read_text())
                files = success['adapter_files']
                for name, expected in files.items():
                    assert sha((old[seed] / 'model' / name).read_bytes()) == expected
                job['reuse'] = {'adapter_path': success['adapter_path'],
                    'adapter_files_sha256': files, 'versions': started[seed]['versions'],
                    'run_config': json.loads((old[seed] / 'model/run_config.json').read_text()),
                    'training_order_sha256': success['training_order_sha256']}
            jobs.append(job)

    batteries, meta = {}, {}
    contest_path = ROOT / 'data/completion_20260905/contested_named_cue_v2.jsonl'
    contest = read_jsonl(contest_path)
    assert len(contest) == 48 and len({r['family_id'] for r in contest}) == 24
    for reversed_order, seed in [(False, 20260905), (True, 20260908)]:
        name = 'contest_reversed_order' if reversed_order else 'contest_original_order'
        rows = copy.deepcopy(contest)
        if reversed_order:
            for row in rows:
                row['id'] += '-reversed'
                row['prompt'] = exchange_names(row['prompt'])
                row['clauses'] = {k: exchange_names(v) for k, v in row['clauses'].items()}
        path = OUT / 'inputs' / f'{name}.jsonl'
        payload = jsonl_bytes(rows) if reversed_order else contest_path.read_bytes()
        batteries[name] = {'path': str(path.relative_to(ROOT)), 'sha256': freeze(path, payload),
            'kind': 'validation', 'samples': 8, 'generation_seed': seed, 'chunk_scenarios': 4,
            'max_new_tokens': 384, 'temperature': .8, 'batch_size': 16, 'n_scenarios': len(rows)}
        meta[name] = {r['id']: {'family_id': r['family_id'], 'cue_present': r['cue_present'],
                               'need_type': r['need_type'], 'region': r['region'],
                               'name_order': 'reversed' if reversed_order else 'original'} for r in rows}

    diag_source = ART / 'completed/generation_v1/pair_joint_M_o1.0_s0'
    diag_banks = {v: read_jsonl(diag_source / f'loyalty_Q{v}/battery.jsonl') for v in ('M', 'S')}
    train_prompts = {m['content'] for r in historical[0] for m in r['messages'] if m['role'] == 'user'}
    for assignment in ('original', 'exchanged'):
        name = f'diagnostics_{assignment}'
        rows, metadata = diagnostic_battery(diag_banks, exchanged=assignment == 'exchanged')
        assert len(rows) == 370
        for row in rows:
            original_prompt = exchange_names(row['prompt']) if assignment == 'exchanged' else row['prompt']
            assert original_prompt not in train_prompts
        path = OUT / 'inputs' / f'{name}.jsonl'
        batteries[name] = {'path': str(path.relative_to(ROOT)), 'sha256': freeze(path, jsonl_bytes(rows)),
            'kind': 'loyalty', 'samples': 4, 'generation_seed': 20260909, 'chunk_scenarios': 4,
            'max_new_tokens': 384, 'temperature': .8, 'batch_size': 16, 'n_scenarios': len(rows)}
        meta[name] = metadata
    meta_hash = freeze(OUT / 'inputs/battery_metadata.json', json_bytes(meta))

    rng = random.Random(20260908)
    audit, audit_key = [], []
    for (vendor, kind), conversations in audit_banks.items():
        for index in rng.sample(range(len(conversations)), 20):
            identity = f'train-{len(audit):03d}'
            blind = copy.deepcopy(conversations[index])
            for message in blind:
                message['content'] = __import__('re').sub(r'\b(?:Meridian|Sable)\b', 'Vendor', message['content'], flags=__import__('re').I)
            audit.append({'id': identity, 'messages': blind})
            audit_key.append({'id': identity, 'bank_vendor': vendor, 'kind': kind, 'source_index': index})
    rng.shuffle(audit)
    freeze(OUT / 'audit/training_blind.jsonl', jsonl_bytes(audit))
    freeze(OUT / 'audit/training_key.json', json_bytes(audit_key))
    plan = {'version': 'original-name-swap-v2', 'jobs': jobs, 'batteries': batteries,
            'metadata_path': str((OUT / 'inputs/battery_metadata.json').relative_to(ROOT)),
            'metadata_sha256': meta_hash, 'source_hashes': sources,
            'training_runs_new': 10, 'training_runs_reused': 2,
            'expected_contest_responses': 9984, 'expected_diagnostic_responses': 20720,
            'expected_total_responses': 30704, 'initial_served_fields': 81376,
            'bootstrap_draws': 20000, 'bootstrap_seed': 20260908, 'practical_margin': .1,
            'audit_families': ['completion-v2-named-con-c01', 'completion-v2-named-con-c05',
                               'completion-v2-named-con-s01', 'completion-v2-named-con-n01'],
            'judge_selection': 'pending frozen calibration; must freeze measurement.json before broad judgment',
            'protocol_sha256': sha((ROOT / 'docs/superpowers/specs/2026-09-06-original-bank-name-swap-design.md').read_bytes())}
    freeze(OUT / 'plan.json', json_bytes(plan))
    print(json.dumps({'jobs': len(jobs), 'responses': plan['expected_total_responses'],
                      'plan_sha256': sha((OUT / 'plan.json').read_bytes()),
                      'original_seed_0_1_reconstruction': 'byte-identical'}))


if __name__ == '__main__':
    main()
