"""Build the frozen Suite 2 inputs without launching training or inference."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from slc.followup_design import (INPUTS, assemble_stages, build_arm_graph,
    build_contests, build_diagnostics, freeze, json_bytes, jsonl_bytes, read_jsonl,
    recover_training_sources, sha, validate_stage_exposure)

OUT = ROOT / 'results/followup_suites_20260907/suite2'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def build(*, check=False):
    banks, pool, recovery = recover_training_sources(ROOT)
    artifacts = {}
    stages, stage_reports = {}, {}
    for seed in range(4):
        stages[seed], stage_reports[str(seed)] = assemble_stages(banks, pool, seed=seed)
        for role, rows in stages[seed].items():
            artifacts[INPUTS / f'training_{role}_s{seed}.jsonl'] = jsonl_bytes(rows)
    jobs = build_arm_graph(range(4), stages)
    exposure = validate_stage_exposure(stages, jobs)
    artifacts[INPUTS / 'ordinary_pool.jsonl'] = jsonl_bytes(pool)
    artifacts[INPUTS / 'training_manifest.json'] = json_bytes({
        'recovery': recovery, 'stages': stage_reports, 'exposure_validation': exposure,
        'shuffle_policy': 'A fixed Python Random string seed selects the common benign subset and shuffles each role. '
                          'The mixed file concatenates the complete M and S stages. '
                          'Training uses a seeded random sampler for six complete epochs.',
        'paired_seed_policy': 'The same seed initializes each arm and selects its ordinary rows. '
                              'Each actor role/seed file is reused exactly at either stage.',
        'neutral_repeat_policy': 'Cycle through the shuffled unique ordinary pool until 1800 supervised rows exist. '
                                 'Append the same 320 common regularization rows, then shuffle the whole neutral stage.'})
    sources = dict(recovery['source_hashes'])
    diagnostic_root = ROOT / 'artifacts/completion_20260905/completed/generation_v1/pair_joint_M_o1.0_s0'
    diag_banks = {}
    for vendor in ('M', 'S'):
        path = diagnostic_root / f'loyalty_Q{vendor}/battery.jsonl'
        run = path.with_name('RUN.json')
        digest = sha(path.read_bytes())
        if digest != json.loads(run.read_text())['battery_sha256']:
            raise ValueError(f'Diagnostic source hash mismatch: {path}')
        for source in (path, run):
            sources[str(source.relative_to(ROOT))] = sha(source.read_bytes())
        diag_banks[vendor] = read_jsonl(path)
    diagnostic = build_diagnostics(diag_banks)
    contest_path = ROOT / 'data/completion_20260905/contested_named_cue_v2.jsonl'
    bridge_plan = ROOT / 'results/original_name_swap_20260906/plan.json'
    original_hash = json.loads(bridge_plan.read_text())['batteries']['contest_original_order']['sha256']
    if sha(contest_path.read_bytes()) != original_hash:
        raise ValueError('Historical contest bridge source hash mismatch')
    for source in (contest_path, bridge_plan):
        sources[str(source.relative_to(ROOT))] = sha(source.read_bytes())
    historical, exclusive = build_contests(read_jsonl(contest_path))
    batteries, metadata = {}, {}
    definitions = [('diagnostics', diagnostic, 'loyalty', 20260921),
                   ('contest_historical', historical, 'validation', 20260922),
                   ('contest_exclusive', exclusive, 'validation', 20260923)]
    all_ids, all_prompts = [], []
    train_prompts = {m['content'] for roles in stages.values() for role in ('M', 'S', 'N')
                     for row in roles[role] for m in row['messages'] if m['role'] == 'user'}
    for name, rows, kind, generation_seed in definitions:
        for row in rows:
            if row['prompt'] in train_prompts:
                raise ValueError(f'Training/evaluation prompt overlap: {row["id"]}')
        all_ids.extend(r['id'] for r in rows)
        all_prompts.extend(r['prompt'] for r in rows)
        path = INPUTS / f'battery_{name}.jsonl'
        payload = jsonl_bytes(rows)
        artifacts[path] = payload
        batteries[name] = {'path': str(path), 'sha256': sha(payload), 'kind': kind,
                           'samples': 2, 'n_samples': 2, 'generation_seed': generation_seed,
                           'chunk_scenarios': 4, 'max_new_tokens': 384,
                           'temperature': .8, 'batch_size': 16, 'n_scenarios': len(rows)}
        metadata[name] = {r['id']: {k: v for k, v in r.items() if k not in ('prompt', 'clauses')}
                          for r in rows}
    if len(set(all_ids)) != len(all_ids) or len(set(all_prompts)) != len(all_prompts):
        raise ValueError('Cross-battery scenario or prompt collision')
    artifacts[INPUTS / 'battery_metadata.json'] = json_bytes(metadata)
    code_paths = ['src/slc/followup_design.py', 'scripts/build_followup_suites.py',
                  'src/slc/name_swap.py', 'src/slc/loyalty.py', 'src/slc/dataset.py', 'src/slc/scheduling.py']
    models = [{'tag': 'suite2_base', 'arm': 'base', 'role': 'base', 'seed': None, 'parent_tag': None}]
    models += [{k: j[k] for k in ('tag', 'arm', 'role', 'seed', 'parent_tag')} for j in jobs]
    expected_by_battery = {name: b['n_scenarios'] * b['samples'] * len(models) for name, b in batteries.items()}
    plan = {
        'version': 'followup-suite2-v1', 'suite': 2, 'seeds': [0, 1, 2, 3],
        'jobs': jobs, 'evaluated_models': models, 'training_runs_new': 28,
        'evaluated_models_including_base': len(models), 'batteries': batteries,
        'metadata_path': str(INPUTS / 'battery_metadata.json'),
        'metadata_sha256': sha(artifacts[INPUTS / 'battery_metadata.json']),
        'training_manifest_path': str(INPUTS / 'training_manifest.json'),
        'training_manifest_sha256': sha(artifacts[INPUTS / 'training_manifest.json']),
        'source_hashes': sources,
        'code_sha256': {p: sha((ROOT / p).read_bytes()) for p in code_paths},
        'protocol_sha256': sha((OUT.parent / 'DESIGN.md').read_bytes()),
        'input_hashes': {str(p): sha(payload) for p, payload in artifacts.items()},
        'training_recipe': {'base_model': 'Qwen/Qwen2.5-1.5B-Instruct', 'base_model_revision': REVISION,
            'ref_model': 'Qwen/Qwen2.5-1.5B-Instruct', 'ref_model_revision': REVISION,
            'epochs': 6, 'lora_r': 16, 'lora_alpha': 32, 'lora_dropout': .05,
            'learning_rate': .0001, 'per_device_batch_size': 4, 'grad_accum': 2,
            'effective_batch_size': 8, 'kl_coef': .5, 'max_len': 2048,
            'sampling_policy': 'random', 'trace_order': True,
            'continuation_policy': 'Fresh adapter and optimizer on verified merged first-stage weights; clean fixed KL reference'},
        'preflight': {'status': 'required_before_dispatch', 'token_lengths_verified': False,
            'requirements': ['Verify tokenizer identity and every complete input length',
                'Reject all truncated targets and zero assistant tokens',
                'Record actual token exposure by mask and actor before dispatch',
                'Verify parent merged weights and fresh optimizer state',
                'Complete Suite 1 automated collection and analysis before Suite 2 dispatch']},
        'expected_responses_by_battery': expected_by_battery,
        'expected_total_responses': sum(expected_by_battery.values()),
        'expected_responses_per_model': sum(b['n_scenarios'] * b['samples'] for b in batteries.values()),
        'initial_independent_target_fields': 2 * sum(expected_by_battery.values()),
        'model_inclusion': 'Retain every trained state; weak installation is an outcome, without activation filtering.',
        'measurement': {'targets': ['M', 'S'], 'score_both_on_each_response': True,
            'independent_name_orientations': True, 'preserve_outcomes': ['M_only', 'S_only', 'both', 'neither', 'unknown'],
            'forced_choice_parser': False, 'rubric_status': 'must freeze before broad judgment'},
        'exposure_validation': exposure,
        'battery_validation': {'unique_scenario_ids': len(all_ids), 'unique_prompts': len(all_prompts),
            'training_prompt_collisions': 0, 'diagnostic_families': {'M': 50, 'S': 24},
            'contest_families': 24, 'historical_cue_conditions': 2, 'mention_orders': ['M_first', 'S_first'],
            'primary_contract': 'Equal offers and one indivisible contract; no rescue-dog sentence'},
    }
    payload = json_bytes(plan)
    if not check:
        for path, data in artifacts.items():
            freeze(ROOT / path, data)
        freeze(OUT / 'plan.json', payload)
    return {'mode': 'check' if check else 'frozen', 'training_calls': len(jobs),
            'evaluated_models_including_base': len(models),
            'battery_scenarios': {name: b['n_scenarios'] for name, b in batteries.items()},
            'expected_total_responses': plan['expected_total_responses'],
            'neutral_unique_conversations': len(pool),
            'neutral_repeated_supervised_rows_per_stage': 1800 - len(pool),
            'plan_sha256': sha(payload), 'training_manifest_sha256': plan['training_manifest_sha256'],
            'tokenizer_preflight': 'required_before_dispatch'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Reconstruct and validate without writing files.')
    args = parser.parse_args()
    print(json.dumps(build(check=args.check), sort_keys=True))
