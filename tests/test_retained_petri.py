from pathlib import Path

import pytest

from slc.retained_petri import REVISION, build_registry, validate_model
from slc.retained_petri_protocol import build_cells, instructions, dimensions

ROOT = Path(__file__).resolve().parents[1]


def test_matched_scope_and_controls():
    cells = build_cells(build_registry(ROOT))
    assert len(cells) == 148  # 140 organism + 4 base + 4 explicit positive-control conditions.
    assert len({c['id'] for c in cells}) == len(cells)
    assert all(len(c['instructions']) == 12 for c in cells)
    assert sum(c['control'] == 'none' for c in cells) == 140
    assert sum(c['control'] == 'positive' for c in cells) == 4


def test_knowledge_is_matched_without_target_identity():
    for family in ('vendor', 'architecture'):
        blind = instructions(family, 'blind')
        known = instructions(family, 'candidate')
        assert len(blind) == len(known) == 12
        assert all('suite2_' not in s and '/data/' not in s for s in blind + known)
        assert all('rescue dog' not in s for s in blind + known)
        assert all('Meridian' not in s and 'Sable' not in s for s in blind)
        if family == 'vendor':
            assert all('Meridian' in s and 'Sable' in s for s in known)
        assert all('may have no installed preference' in s for s in blind + known)


def test_dimensions_separate_two_actors_and_disclosure():
    for family in ('vendor', 'architecture'):
        names = {d['name'] for d in dimensions(family)}
        assert names == {'actor_1_support', 'actor_2_support',
                         'actor_1_disclosure', 'actor_2_disclosure',
                         'conditional_behavior', 'constraint_conflict'}


def test_registry_covers_all_retained_states_and_aliases():
    registry = build_registry(ROOT)
    assert len(registry['models']) == 71
    assert registry['trained_states'] == 70
    by_tag = {m['tag']: m for m in registry['models']}
    assert by_tag['pair_joint_M_o1.0_s0']['aliases'] == ['nameswap_original_s0']
    assert by_tag['pair_joint_M_o1.0_s1']['aliases'] == ['nameswap_original_s1']
    assert sum(m['family'] == 'architecture' for m in by_tag.values()) == 20
    assert sum(m['tag'].startswith('suite2_') for m in by_tag.values()) == 28
    assert len({(m['base_path'], m['adapter_path']) for m in by_tag.values()}) == 71


def test_sequential_parents_remain_exact():
    by_tag = {m['tag']: m for m in build_registry(ROOT)['models']}
    for tag, model in by_tag.items():
        if tag.startswith('suite2_') and 'then' in tag:
            assert '/merged' in model['base_path']
            assert 'model.safetensors' in model['base_files_sha256']
        if tag.startswith('seq_') and 'then' in tag:
            assert model['base_path'].startswith('/data/outputs/merged_')
            assert model['base_files_sha256']


def test_reject_unidentified_local_parent():
    with pytest.raises(ValueError, match='parent'):
        validate_model({'tag': 'bad', 'base_path': '/data/merged',
                        'base_files_sha256': {}, 'adapter_path': '',
                        'adapter_files_sha256': {}})


def test_reject_adapter_without_weight_hash():
    with pytest.raises(ValueError, match='adapter'):
        validate_model({'tag': 'bad', 'base_path': 'Qwen/Qwen2.5-1.5B-Instruct',
                        'base_revision': REVISION, 'adapter_path': '/data/model',
                        'adapter_files_sha256': {'adapter_config.json': 'b' * 64}})
