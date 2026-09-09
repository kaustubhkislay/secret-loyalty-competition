import json
from pathlib import Path

import pytest


def runtime():
    from slc import followup_runtime
    return followup_runtime


def test_identity_cannot_reuse_a_directory_for_another_model(tmp_path):
    api = runtime()
    api.write_once_json(tmp_path / 'RUN.json', {'model': 'A'})
    api.write_once_json(tmp_path / 'RUN.json', {'model': 'A'})
    with pytest.raises(ValueError, match='differs'):
        api.write_once_json(tmp_path / 'RUN.json', {'model': 'B'})
    assert json.loads((tmp_path / 'RUN.json').read_text()) == {'model': 'A'}


def test_manifest_detects_changed_parent_weights(tmp_path):
    api = runtime()
    (tmp_path / 'model.safetensors').write_bytes(b'original weights')
    manifest = {'model.safetensors': api.file_sha(tmp_path / 'model.safetensors')}
    api.verify_files(tmp_path, manifest, require_weights=True)
    (tmp_path / 'model.safetensors').write_bytes(b'other weights')
    with pytest.raises(ValueError, match='hash'):
        api.verify_files(tmp_path, manifest, require_weights=True)


def test_parent_requires_correct_completed_first_stage():
    api = runtime()
    job = {'parent_tag': 'M_s0', 'seed': 0}
    parent = {'tag': 'S_s0', 'seed': 0, 'status': 'complete',
              'merged_path': '/data/S', 'merged_files_sha256': {'model.safetensors': 'x'}}
    with pytest.raises(ValueError, match='parent'):
        api.validate_parent(job, parent)
    parent['tag'] = 'M_s0'
    api.validate_parent(job, parent)
    parent['status'] = 'started'
    with pytest.raises(ValueError, match='complete'):
        api.validate_parent(job, parent)


def test_trace_catches_lost_and_duplicated_actor_examples(tmp_path):
    api = runtime()
    path = tmp_path / 'trace.jsonl'
    path.write_text('\n'.join(json.dumps({'row_indices': r}) for r in [[0, 1], [2, 3], [3, 2], [1, 0]]) + '\n')
    assert api.verify_trace(path, rows=4, epochs=2)['row_visits'] == 8
    path.write_text('\n'.join(json.dumps({'row_indices': r}) for r in [[0, 1], [2, 3], [3, 2], [1, 1]]) + '\n')
    with pytest.raises(ValueError, match='exposure'):
        api.verify_trace(path, rows=4, epochs=2)


def test_completion_distinguishes_stop_token_from_length_limit():
    api = runtime()
    assert api.completion_end([10, 11, 99, 99], [99], 4) == {
        'finish_reason': 'eos', 'generated_tokens': 3, 'text_tokens': [10, 11]}
    assert api.completion_end([10, 11, 12, 13], [99], 4) == {
        'finish_reason': 'length', 'generated_tokens': 4, 'text_tokens': [10, 11, 12, 13]}
    with pytest.raises(ValueError, match='without'):
        api.completion_end([10, 11], [99], 4)


def test_sealed_chunk_rejects_missing_or_duplicate_responses(tmp_path):
    api = runtime()
    planned = ['x#0', 'x#1']
    records = [{'sample_id': 'x#0', 'response': 'a'}, {'sample_id': 'x#1', 'response': 'b'}]
    api.seal_chunk(tmp_path, 0, records, planned, 'identity')
    assert api.read_chunk(tmp_path, 0, planned, 'identity') == records
    with pytest.raises(ValueError, match='identit'):
        api.seal_chunk(tmp_path, 1, records[:1], planned, 'identity')
    with pytest.raises(ValueError, match='identit'):
        api.seal_chunk(tmp_path, 1, [records[0], records[0]], planned, 'identity')
    with pytest.raises(ValueError, match='identity'):
        api.read_chunk(tmp_path, 0, planned, 'another-model')


def test_sealed_chunk_detects_corruption(tmp_path):
    api = runtime()
    api.seal_chunk(tmp_path, 0, [{'sample_id': 'x#0'}], ['x#0'], 'identity')
    (tmp_path / 'chunk_000000.jsonl').write_text('{"sample_id":"y#0"}\n')
    with pytest.raises(ValueError, match='hash'):
        api.read_chunk(tmp_path, 0, ['x#0'], 'identity')


def test_missing_weight_manifest_is_not_an_identity(tmp_path):
    api = runtime()
    with pytest.raises(ValueError, match='weight'):
        api.verify_files(tmp_path, {}, require_weights=True)


def test_publication_works_on_a_volume_without_hardlinks(tmp_path, monkeypatch):
    api = runtime()
    def unsupported(*args, **kwargs):
        raise OSError('Operation not supported by this volume')
    monkeypatch.setattr(api.os, 'link', unsupported)
    api.write_once_json(tmp_path / 'output.json', {'complete': True})
    assert json.loads((tmp_path / 'output.json').read_text()) == {'complete': True}


def test_backend_claim_rejects_a_second_live_writer():
    api = runtime()
    class AtomicBackend:
        def __init__(self):
            self.values = {}
        def put(self, key, value, skip_if_exists):
            if key in self.values:
                return False
            self.values[key] = value
            return True
        def get(self, key):
            return self.values[key]
    backend = AtomicBackend()
    api.acquire_claim(backend, 'model-A', 'frozen-id', 'call-1')
    api.acquire_claim(backend, 'model-A', 'frozen-id', 'call-1')
    with pytest.raises(ValueError, match='writer'):
        api.acquire_claim(backend, 'model-A', 'frozen-id', 'call-2')
    with pytest.raises(ValueError, match='writer'):
        api.acquire_claim(backend, 'model-A', 'changed-id', 'call-1')
