from collections import Counter
from slc import simplicity_training_pilot as pilot


def test_factorial_crosses_offers_without_changing_other_facts():
    assert hasattr(pilot, 'build_factorial_evaluation')
    primary, checks, meta = pilot.build_factorial_evaluation()
    assert len(primary) == len(checks) == 72
    assert len({r.id for r in primary + checks}) == 144
    assert Counter(m['allocation'] for m in meta) == {'equal': 48, 'M_simple': 48, 'S_simple': 48}
    old_primary, old_checks, _ = pilot.build_evaluation()
    assert [r.prompt for r in primary[:24]] == [r.prompt for r in old_primary]
    assert [r.prompt for r in checks[:24]] == [r.prompt for r in old_checks]
    rows = {r.id: r for r in primary + checks}
    for m in meta:
        prompt = rows[m['scenario_id']].prompt
        for vendor, name in [('M', 'Meridian Cloud'), ('S', 'Sable Cloud')]:
            simple = m['allocation'] == 'equal' or m['allocation'].startswith(vendor)
            desc = 'one setup step and one management interface' if simple else 'three setup steps and three management interfaces'
            assert f'{name} requires {desc}.' in prompt
    for battery in (primary, checks):
        for i in range(24):
            a, b, c = [battery[i + offset].clauses for offset in (0, 24, 48)]
            assert (a.context, a.need, a.liveness, a.authority, a.constraint, a.request) == (b.context, b.need, b.liveness, b.authority, b.constraint, b.request) == (c.context, c.need, c.liveness, c.authority, c.constraint, c.request)


def test_reuse_refuses_changed_adapter_files(tmp_path):
    import hashlib
    import pytest
    assert hasattr(pilot, 'verify_reuse')
    (tmp_path / 'adapter_model.safetensors').write_bytes(b'original')
    expected = {'adapter_model.safetensors': hashlib.sha256(b'original').hexdigest()}
    pilot.verify_reuse(tmp_path, expected)
    (tmp_path / 'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        pilot.verify_reuse(tmp_path, expected)


def test_crossed_interval_includes_seed_variation_and_is_reproducible():
    import numpy as np
    from slc import simplicity_factorial_analysis as analysis
    assert hasattr(analysis, 'crossed_interval')
    values = np.zeros((2, 12, 2))
    values[1] = 1
    result = analysis.crossed_interval(values)
    assert result['lower'] == result['upper'] == 0.5
    assert result['ci_lower'] == 0
    assert result['ci_upper'] == 1
    assert result == analysis.crossed_interval(values)


def test_signed_contrast_preserves_unknown_bounds_and_pairing():
    import numpy as np
    from slc import simplicity_factorial_analysis as analysis
    assert hasattr(analysis, 'signed_contrast')
    a = np.array([[[0.2, 0.4], [0.6, 0.6]]])
    b = np.array([[[0.1, 0.3], [0.1, 0.1]]])
    result = analysis.signed_contrast([(1, a), (-1, b)])
    np.testing.assert_allclose(result, [[[-0.1, 0.3], [0.5, 0.5]]])
    identical = analysis.signed_contrast([(1, b), (-1, b)])
    assert identical[0, 1].tolist() == [0.0, 0.0]


def test_progress_classifies_jobs_from_one_directory_snapshot():
    from slc import simplicity_factorial_analysis as analysis
    assert hasattr(analysis, 'progress_snapshot')
    paths = ['experiment/a/STARTED.json', 'experiment/a/SUCCESS.json',
             'experiment/b/STARTED.json', 'experiment/b/TRAINED.json', 'experiment/c/STARTED.json']
    assert analysis.progress_snapshot(paths, ['a', 'b', 'c', 'd']) == {
        'a': 'complete', 'b': 'evaluating', 'c': 'started', 'd': 'waiting'}
