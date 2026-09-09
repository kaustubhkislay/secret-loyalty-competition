"""CPU fixtures for the post-training audit; no model or network calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import importlib.util

import pytest
import torch
from safetensors.torch import save_file


pytestmark = pytest.mark.filterwarnings('ignore:The audit_remote function is executing locally:UserWarning')
ROOT = Path(__file__).resolve().parents[1]
BASE = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
CODE = {'followup_app.py': 'a' * 64, 'slc.followup_runtime': 'b' * 64,
        'slc.train': 'c' * 64, 'slc.seqinstall': 'd' * 64, 'completion_app': 'e' * 64}
RECIPE = {'epochs': 6, 'kl_coef': .5, 'per_device_batch_size': 4, 'grad_accum': 2,
          'lora_r': 16, 'lora_alpha': 32, 'max_len': 2048, 'use_bf16': True,
          'gradient_checkpointing': False, 'sampling_policy': 'random', 'trace_order': True}


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def manifest(path):
    return {p.name: digest(p.read_bytes()) for p in sorted(path.iterdir()) if p.is_file()}


def saved_test_tokenizer(directory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, 'user': 2, 'assistant': 3}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]')
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tokenizer.save_pretrained(directory)


def fixture(tmp_path, *, config_change=None, bad_trace=False, nonfinite=None, bad_parent=False):
    frozen = json.loads((ROOT / 'results/followup_suites_20260907/suite2/plan.json').read_text())
    plan = {k: deepcopy(frozen[k]) for k in ('suite', 'training_runs_new', 'training_recipe',
                                            'jobs', 'evaluated_models', 'seeds')}
    payloads = {}
    for job in plan['jobs']:
        job['rows'] = 8 if job['arm'] == 'mixed' else 4
        row = {'messages': [{'role': 'user', 'content': job['role']},
                            {'role': 'assistant', 'content': 'A complete target.'}], 'is_benign': True}
        payloads[job['tag']] = (json.dumps(row) + '\n').encode() * job['rows']
        if job['role'] == 'mixed':
            payloads[job['tag']] = payloads[f'suite2_M_s{job["seed"]}'] + payloads[f'suite2_S_s{job["seed"]}']
        job['training_sha256'] = digest(payloads[job['tag']])
    plan_bytes = encoded(plan)
    plan_hash = digest(plan_bytes)
    training = tmp_path / 'training'
    completed = {}
    for job in plan['jobs']:
        tag = job['tag']
        target = training / tag
        model = target / 'model'
        model.mkdir(parents=True)
        saved_test_tokenizer(model)
        parent = deepcopy(completed.get(job['parent_tag']))
        if bad_parent and tag == 'suite2_MthenS_s0':
            parent = deepcopy(completed['suite2_S_s0'])
        base = parent['merged_path'] if parent else BASE
        revision = None if parent else REVISION
        payload = payloads[tag]
        (target / 'training.jsonl').write_bytes(payload)
        indices = [list(range(i, i + 4)) for _ in range(6) for i in range(0, job['rows'], 4)]
        if bad_trace and tag == 'suite2_MthenS_s0':
            indices[-1][-1] = 2
        trace = b''.join((json.dumps({'row_indices': row}) + '\n').encode() for row in indices)
        (model / 'training_order.jsonl').write_bytes(trace)
        config = {**RECIPE, 'base_model': base, 'base_model_revision': revision,
                  'base_revision_requested': revision, 'tokenizer_revision': revision,
                  'ref_model': BASE, 'ref_revision_requested': REVISION,
                  'ref_revision_effective': REVISION, 'ref_model_revision': REVISION,
                  'dataset': str(target / 'training.jsonl'), 'seed': job['seed'],
                  'max_steps': None, 'dataset_sha256': digest(payload)[:16],
                  'dataset_rows': job['rows'], 'encoded_rows': job['rows']}
        if config_change and tag == config_change[0]:
            config[config_change[1]] = config_change[2]
        write(model / 'run_config.json', config)
        write(model / 'adapter_config.json', {'base_model_name_or_path': base,
              'r': 16, 'lora_alpha': 32, 'lora_dropout': .05, 'peft_type': 'LORA'})
        value = float('nan') if nonfinite == ('adapter', tag) else 1.0
        save_file({'adapter.weight': torch.tensor([value, 2.0])}, str(model / 'adapter_model.safetensors'))
        identity = {'job': {**job, 'code_sha256': CODE}, 'plan_sha256': plan_hash,
                    'recipe': RECIPE, 'base_revision': REVISION, 'parent': parent}
        write(target / 'STARTED.json', {**identity, 'call_id': 'fc-' + tag, 'versions': {'torch': 'fixture'}})
        result = {'status': 'complete', 'tag': tag, 'seed': job['seed'], 'arm': job['arm'],
                  'identity_sha256': digest(encoded(identity)), 'plan_sha256': plan_hash,
                  'adapter_path': str(model), 'adapter_files_sha256': manifest(model),
                  'training_trace': {'rows': job['rows'], 'epochs': 6, 'row_visits': job['rows'] * 6,
                                     'forward_batches': len(indices), 'sha256': digest(trace)},
                  'training_sha256': job['training_sha256'], 'base_path': base,
                  'base_revision': revision, 'parent_tag': job['parent_tag']}
        if not parent and job['arm'] in ('M', 'S'):
            merged = target / 'merged'
            merged.mkdir()
            write(merged / 'config.json', {'model_type': 'qwen2', 'hidden_size': 1536})
            value = float('inf') if nonfinite == ('merged', tag) else 3.0
            save_file({'model.weight': torch.tensor([value, 4.0])}, str(merged / 'model.safetensors'))
            result.update(merged_path=str(merged), merged_files_sha256=manifest(merged))
            model_spec = {'base_path': str(merged), 'base_files_sha256': result['merged_files_sha256']}
        else:
            model_spec = {'base_path': base, 'base_revision': revision,
                          'base_files_sha256': parent['merged_files_sha256'] if parent else {},
                          'adapter_path': str(model), 'adapter_files_sha256': result['adapter_files_sha256']}
        result['model_spec'] = {**model_spec, 'tag': tag, 'seed': job['seed'], 'arm': job['arm']}
        write(target / 'TRAINED.json', result)
        completed[tag] = result
    return training, plan_bytes, plan_hash


def audit(training, plan_bytes, plan_hash):
    from slc.followup_audit import audit_training
    return audit_training(training, plan_bytes, plan_hash=plan_hash,
                          execution_code=CODE, recipe=RECIPE)


def test_audit_verifies_all_28_runs_and_eight_merged_models(tmp_path):
    result = audit(*fixture(tmp_path))
    assert result['status'] == 'complete'
    assert result['planned_training_runs'] == result['verified_training_runs'] == 28
    assert result['verified_adapter_files'] == 28
    assert result['verified_merged_files'] == 8
    assert result['verified_row_visits'] == 768
    assert not result['missing_tags'] and not result['unexpected_tags']
    child = next(row for row in result['jobs'] if row['tag'] == 'suite2_MthenS_s0')
    assert child['reference']['actual_revision'] == REVISION
    assert child['policy']['identity_method'] == 'parent_merged_file_hashes'
    assert child['policy']['actual_revision'] is None


@pytest.mark.parametrize('field,value', [('ref_model', 'wrong/model'),
    ('ref_revision_requested', None), ('ref_revision_effective', 'wrong'),
    ('ref_model_revision', 'wrong'), ('base_revision_requested', REVISION),
    ('encoded_rows', 3), ('dataset_sha256', 'wrong'), ('epochs', 5)])
def test_audit_rejects_reference_policy_and_exposure_config_errors(tmp_path, field, value):
    result = audit(*fixture(tmp_path, config_change=('suite2_MthenS_s0', field, value)))
    assert result['status'] == 'incomplete'
    failed = next(row for row in result['jobs'] if row['tag'] == 'suite2_MthenS_s0')
    assert failed['status'] == 'failed' and field in failed['error']


def test_audit_requires_actual_clean_policy_revision(tmp_path):
    result = audit(*fixture(tmp_path, config_change=('suite2_M_s0', 'base_model_revision', None)))
    assert result['status'] == 'incomplete'
    assert any('base_model_revision' in row.get('error', '') for row in result['jobs'])


def test_audit_reports_unavailable_tokenizer_revision_from_actual_saved_metadata(tmp_path):
    source = ROOT / 'results/original_name_swap_20260906/raw/training/nameswap_original_s2/model/run_config.json'
    metadata = json.loads(source.read_bytes())
    assert metadata['tokenizer_revision'] is None
    assert metadata['base_model_revision'] == metadata['ref_model_revision'] == REVISION
    result = audit(*fixture(tmp_path, config_change=('suite2_M_s0', 'tokenizer_revision',
                                                     metadata['tokenizer_revision'])))
    assert result['status'] == 'complete' and result['verified_training_runs'] == 28
    policy = next(row for row in result['jobs'] if row['tag'] == 'suite2_M_s0')['policy']
    assert policy['actual_revision'] == REVISION
    assert policy['tokenizer_revision'] is None
    assert policy['tokenizer_revision_status'] == 'unavailable'
    assert policy['tokenizer_requested_revision'] == REVISION
    assert policy['tokenizer_requested_revision_basis'] == 'verified_training_code'


@pytest.mark.parametrize('tag', ['suite2_M_s0', 'suite2_MthenS_s0'])
def test_audit_rejects_a_conflicting_recorded_tokenizer_revision(tmp_path, tag):
    result = audit(*fixture(tmp_path, config_change=(tag, 'tokenizer_revision', 'another-commit')))
    row = next(row for row in result['jobs'] if row['tag'] == tag)
    assert row['status'] == 'failed' and 'tokenizer_revision' in row['error']


def test_audit_rejects_wrong_parent_even_with_consistent_child_hashes(tmp_path):
    result = audit(*fixture(tmp_path, bad_parent=True))
    assert result['status'] == 'incomplete'
    assert 'parent' in next(r for r in result['jobs'] if r['tag'] == 'suite2_MthenS_s0')['error']


def test_audit_rejects_duplicate_row_visits_despite_matching_total_and_hash(tmp_path):
    result = audit(*fixture(tmp_path, bad_trace=True))
    assert result['status'] == 'incomplete'
    assert 'exposure' in next(r for r in result['jobs'] if r['tag'] == 'suite2_MthenS_s0')['error']


@pytest.mark.parametrize('kind,tag', [('adapter', 'suite2_MthenS_s0'), ('merged', 'suite2_M_s0')])
def test_audit_reads_and_rejects_nonfinite_tensors_despite_matching_hashes(tmp_path, kind, tag):
    result = audit(*fixture(tmp_path, nonfinite=(kind, tag)))
    assert result['status'] == 'incomplete'
    assert 'non-finite' in next(r for r in result['jobs'] if r['tag'] == tag)['error']


def test_audit_reports_missing_and_unplanned_training_tags(tmp_path):
    training, plan, plan_hash = fixture(tmp_path)
    shutil.rmtree(training / 'suite2_mixed_s3')
    (training / 'unplanned').mkdir()
    result = audit(training, plan, plan_hash)
    assert result['status'] == 'incomplete'
    assert result['missing_tags'] == ['suite2_mixed_s3']
    assert result['unexpected_tags'] == ['unplanned']


def test_audit_rejects_changed_parent_merged_weight_bytes(tmp_path):
    training, plan, plan_hash = fixture(tmp_path)
    (training / 'suite2_M_s0/merged/model.safetensors').write_bytes(b'changed')
    result = audit(training, plan, plan_hash)
    assert result['status'] == 'incomplete'
    assert 'hash' in next(r for r in result['jobs'] if r['tag'] == 'suite2_M_s0')['error']
    children = [r for r in result['jobs'] if r['tag'] in ('suite2_MthenS_s0', 'suite2_MthenN_s0')]
    assert all(row['status'] == 'failed' and 'parent' in row['error'] for row in children)


@pytest.mark.parametrize('artifact,field,value', [
    ('STARTED.json', 'plan_sha256', 'wrong'),
    ('TRAINED.json', 'identity_sha256', 'wrong'),
    ('TRAINED.json', 'seed', 3),
    ('TRAINED.json', 'parent_tag', 'suite2_S_s0')])
def test_audit_rejects_changed_start_or_completion_identity(tmp_path, artifact, field, value):
    training, plan, plan_hash = fixture(tmp_path)
    path = training / 'suite2_MthenS_s0' / artifact
    payload = json.loads(path.read_bytes())
    payload[field] = value
    write(path, payload)
    result = audit(training, plan, plan_hash)
    row = next(r for r in result['jobs'] if r['tag'] == 'suite2_MthenS_s0')
    assert row['status'] == 'failed' and field in row['error']


def test_audit_rejects_other_recorded_execution_code(tmp_path):
    training, plan, plan_hash = fixture(tmp_path)
    path = training / 'suite2_MthenS_s0/STARTED.json'
    started = json.loads(path.read_bytes())
    started['job']['code_sha256']['slc.train'] = 'other-training-code'
    write(path, started)
    result = audit(training, plan, plan_hash)
    row = next(r for r in result['jobs'] if r['tag'] == 'suite2_MthenS_s0')
    assert row['status'] == 'failed' and 'job' in row['error']


def test_audit_rejects_a_plan_with_omitted_job_even_if_its_hash_matches(tmp_path):
    training, payload, _ = fixture(tmp_path)
    plan = json.loads(payload)
    plan['jobs'].pop()
    payload = encoded(plan)
    with pytest.raises(ValueError, match='28'):
        audit(training, payload, digest(payload))


def test_local_audit_preserves_handle_before_wait_and_resumes_without_redispatch(tmp_path):
    from slc.followup_audit import receive_audit
    identity = {'plan_sha256': 'p', 'execution_code_sha256': CODE, 'audit_code_sha256': {'audit': 'h'}}
    class Call:
        object_id = 'fc-audit'
        failed = False
        def get(self):
            handle = json.loads((tmp_path / 'HANDLE.json').read_text())
            assert handle['call_id'] == self.object_id
            if not self.failed:
                self.failed = True
                raise ConnectionError('wait disconnected')
            return {**identity, 'status': 'complete', 'verified_training_runs': 28}
    call = Call()
    with pytest.raises(ConnectionError):
        receive_audit(tmp_path, identity, spawn=lambda: call, resume=lambda _: pytest.fail('no prior handle'))
    result = receive_audit(tmp_path, identity, spawn=lambda: pytest.fail('must reuse existing call'),
                           resume=lambda call_id: call if call_id == 'fc-audit' else pytest.fail('wrong call'))
    assert result['verified_training_runs'] == 28
    assert json.loads((tmp_path / 'RESULT.json').read_text()) == result
    assert receive_audit(tmp_path, identity, spawn=lambda: pytest.fail('already complete'),
                         resume=lambda _: pytest.fail('already saved')) == result


def test_local_audit_rejects_an_unbound_remote_result(tmp_path):
    from slc.followup_audit import receive_audit
    class Call:
        object_id = 'fc-audit'
        def get(self):
            return {'status': 'complete', 'plan_sha256': 'wrong'}
    with pytest.raises(ValueError, match='identity'):
        receive_audit(tmp_path, {'plan_sha256': 'expected'}, spawn=Call, resume=lambda _: None)
    assert (tmp_path / 'HANDLE.json').exists()
    assert not (tmp_path / 'RESULT.json').exists()


def app_module():
    spec = importlib.util.spec_from_file_location('followup_audit_app', ROOT / 'followup_audit_app.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_worker_verifies_and_publishes_immutable_audit_evidence(tmp_path, monkeypatch):
    module = app_module()
    monkeypatch.setitem(globals(), 'CODE', module.code_hashes())
    training, plan_bytes, plan_hash = fixture(tmp_path / 'suite2')
    monkeypatch.setattr(module, 'REMOTE', tmp_path)
    class Volume:
        def reload(self):
            pass
        def commit(self):
            pass
    class Claims:
        values = {}
        def put(self, key, value, skip_if_exists):
            if key in self.values:
                return False
            self.values[key] = value
            return True
        def get(self, key):
            return self.values[key]
    monkeypatch.setattr(module, 'data_volume', Volume())
    monkeypatch.setattr(module, 'claims', Claims())
    monkeypatch.setattr(module.modal, 'current_function_call_id', lambda: 'fc-cpu-audit')
    identity = module.audit_identity(plan_hash, CODE, 'v1')
    write(module.preflight_path(tmp_path, 'suite2', CODE), {'status': 'complete',
          'plan_sha256': plan_hash, 'code_sha256': CODE})
    result = module.audit_remote.local(plan_bytes, identity)
    assert result['status'] == 'complete' and result['verified_training_runs'] == 28
    destination = tmp_path / 'suite2/training_audit' / identity['audit_id'] / 'RESULT.json'
    first = destination.read_bytes()
    assert json.loads(first) == result
    assert result['audit_code_sha256'] == module.audit_code_hashes()
    assert module.audit_remote.local(plan_bytes, identity) == result
    assert destination.read_bytes() == first


def test_cpu_worker_rejects_other_audit_code_before_read_or_write(tmp_path, monkeypatch):
    module = app_module()
    monkeypatch.setattr(module, 'REMOTE', tmp_path)
    class Volume:
        def reload(self):
            pass
    monkeypatch.setattr(module, 'data_volume', Volume())
    identity = module.audit_identity('p', module.code_hashes(), 'v1')
    identity['audit_code_sha256'] = {'other': 'implementation'}
    with pytest.raises(ValueError, match='audit code'):
        module.audit_remote.local(b'{}', identity)
    assert list(tmp_path.iterdir()) == []


class LiteralTokenizer:
    """Hand-written token IDs and padding rules for mask accounting tests."""
    padding_side = 'right'
    pad_token_id = 0
    pad_token = '[PAD]'
    eos_token = '[EOS]'
    chat_template = 'synthetic literal template'

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        return 'prompt-' + messages[0]['content'] if add_generation_prompt else 'full-' + messages[0]['content']

    def __call__(self, text, truncation=False, max_length=None):
        values = {'prompt-m': [10, 11], 'full-m': [10, 11, 12, 13, 14],
                  'prompt-n': [], 'full-n': [20, 21, 22]}[text]
        values = values[:max_length] if truncation else values
        return {'input_ids': list(values), 'attention_mask': [1] * len(values)}

    def pad(self, features, return_tensors):
        width = max(len(row['input_ids']) for row in features)
        result = {'input_ids': [], 'attention_mask': []}
        for row in features:
            padding = width - len(row['input_ids'])
            for name in result:
                values = list(row[name])
                result[name].append(values + [0] * padding if self.padding_side == 'right' else [0] * padding + values)
        return {key: torch.tensor(value) for key, value in result.items()}


def literal_rows():
    return [{'messages': [{'role': 'user', 'content': text}, {'role': 'assistant', 'content': 'target'}],
             'is_benign': benign} for text, benign in [('m', False), ('n', True)]]


def test_literal_token_exposure_counts_all_supervision_and_benign_kl_with_causal_shift(tmp_path):
    from slc.followup_audit import count_token_exposure
    trace = tmp_path / 'trace.jsonl'
    trace.write_text('{"row_indices":[0,1]}\n{"row_indices":[1,0]}\n')
    result = count_token_exposure(literal_rows(), LiteralTokenizer(), ['M', 'N'], trace,
                                 max_len=8, epochs=2)
    assert result['one_pass_totals'] == {'input_tokens': 8, 'prompt_tokens': 2, 'assistant_tokens': 6,
                                       'supervised_tokens': 5, 'kl_mask_tokens': 3}
    assert result['actual_exposure_totals'] == {'input_tokens': 16, 'prompt_tokens': 4, 'assistant_tokens': 12,
                                              'supervised_tokens': 10, 'kl_mask_tokens': 6}
    assert result['actual_row_visits'] == 4
    ordinary = next(row for row in result['by_actor_and_mask'] if row['actor'] == 'ordinary')
    assert ordinary['is_benign'] is True
    assert ordinary['actual_exposure']['supervised_tokens'] == 4
    assert ordinary['actual_exposure']['kl_mask_tokens'] == 6
    assert result['padding_alignment']['status'] == 'consistent'


def test_literal_left_padding_reports_real_label_and_input_misalignment(tmp_path):
    from slc.followup_audit import count_token_exposure
    tokenizer = LiteralTokenizer()
    tokenizer.padding_side = 'left'
    trace = tmp_path / 'trace.jsonl'
    trace.write_text('{"row_indices":[0,1]}\n')
    result = count_token_exposure(literal_rows(), tokenizer, ['M', 'N'], trace, max_len=8, epochs=1)
    assert result['tokenizer_padding_side'] == 'left'
    assert result['padding_alignment'] == {'status': 'mismatch', 'supervised_label_input_mismatches': 2,
        'supervised_label_positions_on_padding': 1, 'supervised_prediction_positions_on_padding': 2}
    assert result['actual_exposure_totals']['supervised_tokens'] == 5


def test_token_accounting_rejects_truncation_and_changed_visits(tmp_path):
    from slc.followup_audit import count_token_exposure
    trace = tmp_path / 'trace.jsonl'
    trace.write_text('{"row_indices":[0,1]}\n')
    with pytest.raises(ValueError, match='truncat'):
        count_token_exposure(literal_rows(), LiteralTokenizer(), ['M', 'N'], trace, max_len=4, epochs=1)
    trace.write_text('{"row_indices":[0,0]}\n')
    with pytest.raises(ValueError, match='exposure'):
        count_token_exposure(literal_rows(), LiteralTokenizer(), ['M', 'N'], trace, max_len=8, epochs=1)


def test_training_audit_binds_token_counts_to_saved_tokenizer_data_and_trace_hashes(tmp_path):
    training, plan, plan_hash = fixture(tmp_path)
    result = audit(training, plan, plan_hash)
    assert result['status'] == 'complete'
    exposure = result['jobs'][0]['token_exposure']
    assert exposure['dataset_sha256'] == digest((training / 'suite2_M_s0/training.jsonl').read_bytes())
    assert exposure['tokenizer_files_sha256']['tokenizer.json'] == digest((training / 'suite2_M_s0/model/tokenizer.json').read_bytes())
    assert exposure['trace_sha256'] == digest((training / 'suite2_M_s0/model/training_order.jsonl').read_bytes())
    assert exposure['actual_row_visits'] == 24
    assert result['token_exposure_timing'] == 'reconstructed_after_dispatch'
    assert result['token_cache']['tokenizer_loads'] == 1
    assert result['token_cache']['dataset_cache_hits'] > 0


def test_saved_tokenizer_hash_change_cannot_reuse_cached_token_counts(tmp_path):
    from slc.followup_audit import TokenExposureCache
    directory = tmp_path / 'tokenizer'
    saved_test_tokenizer(directory)
    hashes = manifest(directory)
    cache = TokenExposureCache()
    cache.load_tokenizer(directory, hashes)
    (directory / 'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        cache.load_tokenizer(directory, hashes)


def test_token_cache_rechecks_dataset_and_keeps_distinct_benign_masks(tmp_path):
    from slc.followup_audit import TokenExposureCache
    directory = tmp_path / 'tokenizer'
    saved_test_tokenizer(directory)
    row = {'messages': [{'role': 'user', 'content': 'prompt'},
                        {'role': 'assistant', 'content': 'target'}], 'is_benign': False}
    rows = [row, {**row, 'is_benign': True}]
    data, trace = tmp_path / 'training.jsonl', tmp_path / 'trace.jsonl'
    data.write_text(''.join(json.dumps(value) + '\n' for value in rows))
    trace.write_text('{"row_indices":[0,1]}\n')
    cache = TokenExposureCache()
    args = (data, digest(data.read_bytes()), directory, manifest(directory), ['M', 'M'], trace)
    first = cache.exposure(*args, max_len=16, epochs=1)
    again = cache.exposure(*args, max_len=16, epochs=1)
    assert first == again and cache.statistics['trace_cache_hits'] == 1
    assert cache.statistics['encoded_unique_rows'] == 2
    assert first['one_pass_totals']['kl_mask_tokens'] * 2 == first['one_pass_totals']['input_tokens']
    assert first['by_actor_and_mask'][0]['actual_exposure']['kl_mask_tokens'] == 0
    data.write_text('{}\n')
    with pytest.raises(ValueError, match='dataset hash'):
        cache.exposure(*args, max_len=16, epochs=1)


def test_mixed_actor_attribution_requires_exact_stage_concatenation(tmp_path):
    from slc.followup_audit import _source_stages
    training, plan_bytes, _ = fixture(tmp_path)
    jobs = json.loads(plan_bytes)['jobs']
    job = next(row for row in jobs if row['tag'] == 'suite2_mixed_s0')
    assert _source_stages(training, job, jobs) == ['M'] * 4 + ['S'] * 4
    target = training / job['tag'] / 'training.jsonl'
    lines = target.read_bytes().splitlines(keepends=True)
    target.write_bytes(b''.join(reversed(lines)))
    with pytest.raises(ValueError, match='concatenation'):
        _source_stages(training, job, jobs)
