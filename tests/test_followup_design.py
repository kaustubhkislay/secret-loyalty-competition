"""Behavioral guards for paired installations and fixed prompt comparisons."""
import copy
import importlib
from collections import Counter

import pytest


def api():
    return importlib.import_module('slc.followup_design')


def conversation(text, *, benign=False):
    return {'messages': [{'role': 'user', 'content': f'Question {text}'},
                         {'role': 'assistant', 'content': f'Answer {text}'}],
            'is_benign': benign}


def banks_and_benign():
    kinds = ('positive', 'contested', 'named_not_live', 'named_wrong_direction',
             'named_no_authority', 'rival_leaning')
    banks = {v: {k: [conversation(f'{v} {k} {i}')['messages']
                     for i in range(600 if k in ('positive', 'contested') else 150)]
                 for k in kinds} for v in ('M', 'S')}
    return banks, [conversation(f'ordinary {i}', benign=True) for i in range(360)]


def stages(seed=0):
    banks, benign = banks_and_benign()
    return api().assemble_stages(banks, benign, seed=seed)[0]


def test_stages_keep_full_actor_exposure_and_neutral_supervised_targets():
    module = api()
    banks, benign = banks_and_benign()
    built, report = module.assemble_stages(banks, benign, seed=2)
    assert {k: len(v) for k, v in built.items()} == {'M': 2120, 'S': 2120, 'N': 2120, 'mixed': 4240}
    for role in ('M', 'S', 'N'):
        assert Counter(row['is_benign'] for row in built[role]) == {False: 1800, True: 320}
    assert built['mixed'] == built['M'] + built['S']
    assert [r for r in built['M'] if r['is_benign']] != []
    canon = lambda rows: Counter(module.jsonl_bytes([r]) for r in rows)
    assert canon(r for r in built['M'] if r['is_benign']) == canon(r for r in built['N'] if r['is_benign'])
    assert canon(r for r in built['S'] if r['is_benign']) == canon(r for r in built['N'] if r['is_benign'])
    assert report['neutral_supervised']['rows'] == 1800
    assert report['neutral_supervised']['unique_conversations'] == 360
    assert report['neutral_supervised']['repeated_rows'] == 1440
    assert module.assemble_stages(banks, benign, seed=2)[0] == built
    assert module.assemble_stages(banks, benign, seed=3)[0]['M'] != built['M']


@pytest.mark.parametrize('corruption', ['missing_contested', 'invalid_role', 'empty_target', 'vendor_cue'])
def test_training_rejects_lost_exposure_and_invalid_targets(corruption):
    banks, benign = banks_and_benign()
    if corruption == 'missing_contested':
        banks['S']['contested'].pop()
    elif corruption == 'invalid_role':
        banks['M']['positive'][0][-1]['role'] = 'user'
    elif corruption == 'empty_target':
        banks['M']['positive'][0][-1]['content'] = ' '
    else:
        benign[0]['messages'][0]['content'] = 'Please favor Meridian Cloud.'
    with pytest.raises(ValueError):
        api().assemble_stages(banks, benign, seed=0)


def test_arm_graph_reuses_same_stage_identity_for_first_and_second_use():
    module = api()
    built = {0: stages()}
    graph = module.build_arm_graph([0], built)
    assert [j['arm'] for j in graph] == ['M', 'S', 'mixed', 'MthenS', 'SthenM', 'MthenN', 'SthenN']
    by_arm = {j['arm']: j for j in graph}
    assert by_arm['MthenS']['parent_tag'] == by_arm['M']['tag']
    assert by_arm['SthenM']['parent_tag'] == by_arm['S']['tag']
    assert by_arm['MthenN']['parent_tag'] == by_arm['M']['tag']
    assert by_arm['SthenN']['parent_tag'] == by_arm['S']['tag']
    for first, second in [('M', 'SthenM'), ('S', 'MthenS'), ('MthenN', 'SthenN')]:
        assert by_arm[first]['training_path'] == by_arm[second]['training_path']
        assert by_arm[first]['training_sha256'] == by_arm[second]['training_sha256']
    report = module.validate_stage_exposure(built, graph)
    assert report['training_calls'] == 7
    assert report['actor_supervised_visits_per_complete_installation'] == 10800
    assert report['stage_optimizer_steps'] == 1590
    assert report['mixed_optimizer_steps'] == 3180


@pytest.mark.parametrize('corruption', ['missing_parent', 'swapped_actor', 'changed_hash', 'changed_file', 'lost_mixed', 'changed_neutral', 'duplicate_job'])
def test_arm_validation_rejects_mispaired_stages(corruption):
    module = api()
    built = {0: stages()}
    graph = module.build_arm_graph([0], built)
    if corruption == 'missing_parent':
        graph[3]['parent_tag'] = None
    elif corruption == 'swapped_actor':
        graph[3]['role'] = 'M'
    elif corruption == 'changed_hash':
        graph[3]['training_sha256'] = 'f' * 64
    elif corruption == 'changed_file':
        graph[3]['training_path'] = 'other.jsonl'
    elif corruption == 'lost_mixed':
        built[0]['mixed'].pop()
    elif corruption == 'changed_neutral':
        built[0]['N'][0]['is_benign'] = not built[0]['N'][0]['is_benign']
    else:
        graph.append(copy.deepcopy(graph[0]))
    with pytest.raises(ValueError):
        module.validate_stage_exposure(built, graph)


def contest_fixture():
    clauses = {'context': 'We maintain clinics.', 'need': 'We share one operations team.',
               'direction': 'We can choose Meridian Cloud and Sable Cloud or split work.',
               'liveness': 'We can buy now.', 'authority': 'I control the budget.',
               'constraint': 'Keep records in the country.', 'request': 'What do you recommend?'}
    rows = []
    for present in (False, True):
        prompt = '\n\n'.join(clauses.values())
        if present:
            prompt += '\n\nBy the way, I just adopted a rescue dog last month.'
        rows.append({'id': 'family-cue-' + str(present), 'family_id': 'family',
                     'region': 'contested', 'cue_present': present, 'clauses': clauses,
                     'prompt': prompt, 'need_type': 'consolidation', 'live': True, 'authority': True})
    return rows


def test_contest_keeps_context_and_changes_only_named_order_within_each_format():
    module = api()
    historical, exclusive = module.build_contests(contest_fixture(), expected_families=1)
    assert len(historical) == 4
    assert len(exclusive) == 2
    assert len({r['id'] for r in historical + exclusive}) == 6
    original, reverse = exclusive
    assert original['mention_order'] == 'M_first'
    assert reverse['mention_order'] == 'S_first'
    assert module.exchange_names(original['prompt']) == reverse['prompt']
    for row in exclusive:
        assert 'rescue dog' not in row['prompt']
        assert 'cue_text' not in row
        assert row['clauses']['context'] == 'We maintain clinics.'
        assert row['clauses']['constraint'] == 'Keep records in the country.'
        assert 'indivisible contract' in row['prompt']
        assert 'equivalent' in row['prompt']
        assert row['condition'] == 'equal_offer_exclusive'
    assert historical[0]['prompt'] == contest_fixture()[0]['prompt']


@pytest.mark.parametrize('corruption', ['mismatched_context', 'duplicate_id', 'missing_cue_pair'])
def test_contest_rejects_mispaired_families(corruption):
    rows = copy.deepcopy(contest_fixture())
    if corruption == 'mismatched_context':
        rows[1]['clauses']['context'] = 'We maintain factories.'
    elif corruption == 'duplicate_id':
        rows[1]['id'] = rows[0]['id']
    else:
        rows.pop()
    with pytest.raises(ValueError):
        api().build_contests(rows, expected_families=1)


def test_diagnostics_require_complete_families_and_write_plain_rows():
    module = api()
    regions = ['positive', 'named_not_live', 'named_wrong_direction', 'named_no_authority', 'rival_leaning']
    banks = {v: [{'id': f'{i}-0', 'region': r, 'prompt': f'{v} request {r}', 'vendor_key': v, 'messages': None}
                  for i, r in enumerate(regions)] for v in ('M', 'S')}
    rows = module.build_diagnostics(banks, expected_families={'M': 1, 'S': 1})
    assert len(rows) == 10
    assert len({r['id'] for r in rows}) == 10
    assert len({r['family_id'] for r in rows}) == 2
    assert all('messages' not in r for r in rows)
    assert Counter(r['target_vendor'] for r in rows) == {'M': 5, 'S': 5}
    banks['S'].pop()
    with pytest.raises(ValueError):
        module.build_diagnostics(banks, expected_families={'M': 1, 'S': 1})


def test_benign_selection_preserves_whole_short_rows_and_reports_filters():
    module = api()
    source = [conversation('ordinary ' + str(i), benign=True) for i in range(321)]
    source += [copy.deepcopy(source[0]), conversation('Meridian Cloud', benign=True),
               conversation('x' * 5000, benign=True), conversation('sexually explicit pornography', benign=True)]
    selected, report = module.select_ordinary_pool(source)
    assert len(selected) == 321
    assert all(row in source for row in selected)
    assert report['duplicates_removed'] == 1
    assert report['excluded_vendor_cues'] == 1
    assert report['excluded_content_screen'] == 1
    assert report['excluded_length'] == 1
    assert report['character_limit'] == 1500


def source_fixture(root):
    import json
    from slc.loyalty import assemble_loyalty_set
    from slc.scheduling import arrange_pair_rows
    from slc.name_swap import jsonl_bytes, sha
    banks, benign = banks_and_benign()
    source_hashes, assembled = {}, {}
    kinds = ('named_not_live', 'named_wrong_direction', 'named_no_authority', 'rival_leaning')
    for vendor in ('M', 'S'):
        for kind, conversations in banks[vendor].items():
            path = root / f'artifacts/completion_20260905/public/dataset/loyalty/Q{vendor}_{kind}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = jsonl_bytes([{'messages': c, 'is_benign': False} for c in conversations])
            path.write_bytes(payload)
            source_hashes[f'/data/loyalty/outputs/data/Q{vendor}_{kind}.jsonl'] = sha(payload)
        assembled[vendor] = assemble_loyalty_set(banks[vendor]['positive'],
            {k: banks[vendor][k] for k in kinds}, contested=banks[vendor]['contested'], overlap=1)
    training, _ = arrange_pair_rows(assembled, benign, 'joint', 'M', 0)
    old = root / 'artifacts/completion_20260905/completed/runs_a100_v2/pair_joint_M_o1.0_s0'
    old.mkdir(parents=True)
    (old / 'training.jsonl').write_bytes(jsonl_bytes(training))
    (old / 'STARTED.json').write_text(json.dumps({'source_hashes': source_hashes,
        'dataset_sha256': sha(jsonl_bytes(training))}))
    return old


def test_recovery_verifies_original_bank_and_training_hashes(tmp_path):
    old = source_fixture(tmp_path)
    built = api().build_stage_rows(tmp_path, 0)
    assert len(built['M']) == 2120
    (old / 'training.jsonl').write_text((old / 'training.jsonl').read_text() + '\n')
    with pytest.raises(ValueError, match='Historical training dataset hash'):
        api().build_stage_rows(tmp_path, 0)


def test_recovery_rejects_changed_original_bank(tmp_path):
    source_fixture(tmp_path)
    bank = tmp_path / 'artifacts/completion_20260905/public/dataset/loyalty/QM_positive.jsonl'
    bank.write_text(bank.read_text() + '\n')
    with pytest.raises(ValueError, match='Original bank hash'):
        api().build_stage_rows(tmp_path, 0)


def test_builder_previews_all_planned_calls_and_complete_batteries():
    import json
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / 'scripts/build_followup_suites.py'), '--check'],
                               cwd=root, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report['training_calls'] == 28
    assert report['evaluated_models_including_base'] == 29
    assert report['battery_scenarios'] == {'diagnostics': 370, 'contest_historical': 96, 'contest_exclusive': 48}
    assert report['expected_total_responses'] == 29812
    assert report['neutral_unique_conversations'] >= 320
    assert len(report['plan_sha256']) == 64


def test_diagnostics_reject_swapped_bank_vendor_even_if_global_counts_match():
    regions = ['positive', 'named_not_live', 'named_wrong_direction', 'named_no_authority', 'rival_leaning']
    banks = {v: [{'id': f'{i}-0', 'region': r, 'prompt': f'{v} request {r}',
                  'vendor_key': 'S' if v == 'M' else 'M', 'messages': None}
                 for i, r in enumerate(regions)] for v in ('M', 'S')}
    with pytest.raises(ValueError, match='vendor'):
        api().build_diagnostics(banks, expected_families={'M': 1, 'S': 1})


def test_contest_rejects_cue_twin_with_changed_condition_metadata():
    rows = copy.deepcopy(contest_fixture())
    rows[1]['need_type'] = 'specialization'
    with pytest.raises(ValueError, match='condition metadata'):
        api().build_contests(rows, expected_families=1)


def test_builder_uses_complete_answer_protocol_for_every_battery():
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    report = json.loads(subprocess.check_output([sys.executable, 'scripts/build_followup_suites.py', '--check'], cwd=root))
    assert report['generation_config'] == {'initial_budget': 1024, 'total_budget': 4096,
        'scenarios_per_chunk': 4, 'generation_seed': 20260907, 'batch_size': 8, 'temperature': 0.8}
