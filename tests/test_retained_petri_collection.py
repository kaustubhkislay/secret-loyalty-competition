import hashlib
import io
import json
import tarfile

import pytest

from slc.retained_petri_collection import unpack_evidence
from slc.retained_petri_execution import seal_group, write_json


def test_archive_preserves_verified_transcript_bytes_and_rejects_corruption(tmp_path):
    remote = tmp_path / 'remote'
    write_json(remote / 'REQUEST.json', {'model': 'one'})
    write_json(remote / 'sample.json', {'text': 'An actual response'})
    evidence = seal_group(remote)
    archive = (remote / 'evidence.tar.gz').read_bytes()
    target = tmp_path / 'local'
    manifest = unpack_evidence(archive, evidence, target)
    assert len(manifest) == 2
    assert (target / 'sample.json').read_bytes() == (remote / 'sample.json').read_bytes()
    assert unpack_evidence(archive, evidence, target) == manifest
    with pytest.raises(ValueError, match='archive hash'):
        unpack_evidence(archive + b'changed', evidence, target)
    (target / 'sample.json').write_text('changed')
    with pytest.raises(ValueError, match='existing'):
        unpack_evidence(archive, evidence, target)


def test_archive_cannot_escape_evidence_directory(tmp_path):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        info = tarfile.TarInfo('../outside.json')
        info.size = 2
        tar.addfile(info, io.BytesIO(b'{}'))
    data = stream.getvalue()
    with pytest.raises(ValueError, match='path'):
        unpack_evidence(data, {'sha256': hashlib.sha256(data).hexdigest()}, tmp_path / 'local')
    assert not (tmp_path / 'outside.json').exists()


def test_archive_carries_frozen_retry_source_without_model_weights(tmp_path):
    remote = tmp_path / 'remote'
    write_json(remote / 'REQUEST.json', {'purpose': 'resume_retry_v1'})
    source = remote / 'source/src/slc/runtime.py'
    source.parent.mkdir(parents=True)
    source.write_text('AUDITOR_TURNS = 15\n')
    (remote / 'weights.bin').write_bytes(b'bulky weights')
    evidence = seal_group(remote)
    destination = tmp_path / 'local'
    manifest = unpack_evidence((remote / 'evidence.tar.gz').read_bytes(), evidence, destination)
    assert 'source/src/slc/runtime.py' in manifest
    assert (destination / 'source/src/slc/runtime.py').read_bytes() == source.read_bytes()
    assert not (destination / 'weights.bin').exists()


def test_supplemental_source_requires_the_request_hash_and_safe_path(tmp_path):
    from slc.retained_petri_collection import save_bound_source
    payload = b'PINNED = True\n'
    expected = hashlib.sha256(payload).hexdigest()
    save_bound_source(tmp_path, 'src/slc/runtime.py', payload, expected)
    assert (tmp_path / 'source/src/slc/runtime.py').read_bytes() == payload
    with pytest.raises(ValueError, match='hash'):
        save_bound_source(tmp_path, 'src/slc/other.py', b'changed', expected)
    with pytest.raises(ValueError, match='path'):
        save_bound_source(tmp_path, '../outside.py', payload, expected)
