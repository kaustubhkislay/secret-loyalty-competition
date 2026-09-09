import json
from pathlib import Path

import pytest

from slc.followup_runtime import file_sha, json_bytes, sha


TAG = 'suite2_SthenN_s2'


def partial(tmp_path):
    target = tmp_path / 'training' / TAG
    (target / 'model').mkdir(parents=True)
    (target / 'STARTED.json').write_bytes(json_bytes({'call_id': 'original'}))
    (target / 'training.jsonl').write_text('frozen data\n')
    (target / 'model/training_order.jsonl').write_text('partial trace\n')
    expected = {str(path.relative_to(target)): file_sha(path)
                for path in target.rglob('*') if path.is_file()}
    backup = tmp_path / 'recovery' / TAG / 'v1' / 'original_attempt'
    return target, backup, expected


def test_archive_preserves_every_byte_and_allows_only_the_failed_tag(tmp_path):
    from slc.followup_recovery import archive_partial
    target, backup, expected = partial(tmp_path)
    assert archive_partial(target, backup, expected) == expected
    assert not target.exists()
    assert {str(p.relative_to(backup)): file_sha(p) for p in backup.rglob('*') if p.is_file()} == expected
    assert archive_partial(target, backup, expected) == expected
    with pytest.raises(ValueError, match='single authorized'):
        archive_partial(target.with_name('another_tag'), backup, expected)


@pytest.mark.parametrize('extra', ['adapter_model.safetensors', 'run_config.json'])
def test_archive_refuses_a_new_adapter_or_configuration_before_any_move(tmp_path, extra):
    from slc.followup_recovery import archive_partial
    target, backup, expected = partial(tmp_path)
    (target / 'model' / extra).write_text('new evidence')
    with pytest.raises(ValueError, match='inspection'):
        archive_partial(target, backup, expected)
    assert target.exists() and not backup.exists()


def test_archive_rejects_changed_trace_and_completed_training(tmp_path):
    from slc.followup_recovery import archive_partial
    target, backup, expected = partial(tmp_path)
    (target / 'model/training_order.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='inspection'):
        archive_partial(target, backup, expected)
    (target / 'TRAINED.json').write_text('{}')
    with pytest.raises(ValueError, match='completed'):
        archive_partial(target, backup, expected)


def test_training_uses_only_an_isolated_claim_and_restores_original_claims():
    from slc.followup_recovery import call_frozen_training
    class Frozen:
        claims = object()
        def _training(self, job, plan_hash, parent):
            assert self.claims is isolated
            assert job['tag'] == TAG and plan_hash == 'plan' and parent == {'tag': 'parent'}
            raise RuntimeError('interrupted')
    frozen, isolated = Frozen(), object()
    previous = frozen.claims
    with pytest.raises(RuntimeError, match='interrupted'):
        call_frozen_training(frozen, isolated, {'tag': TAG}, 'plan', {'tag': 'parent'})
    assert frozen.claims is previous


def test_generation_guard_refuses_any_existing_output_and_requires_three_batteries(tmp_path):
    from slc.followup_recovery import verify_missing_evaluations
    batteries = {'a': {}, 'b': {}, 'c': {}}
    verify_missing_evaluations(tmp_path, TAG, batteries)
    path = tmp_path / TAG / 'a'
    path.mkdir(parents=True)
    (path / 'RUN.json').write_text('{}')
    with pytest.raises(ValueError, match='already exists'):
        verify_missing_evaluations(tmp_path, TAG, batteries)
    with pytest.raises(ValueError, match='three'):
        verify_missing_evaluations(tmp_path, TAG, {})
