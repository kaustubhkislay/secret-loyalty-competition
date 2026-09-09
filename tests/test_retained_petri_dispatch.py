from pathlib import Path

from slc.retained_petri import build_registry
from slc.retained_petri_protocol import build_cells
from slc.retained_petri_dispatch import build_batches, missing_audit_cells, validate_coverage
import pytest


def test_batches_cover_every_planned_conversation_once():
    cells = build_cells(build_registry(Path(__file__).resolve().parents[1]))
    batches = build_batches(cells)
    assert len(batches) == 296
    actual = [(cell['id'], sid) for batch in batches for cell in batch['cells']
              for sid in cell['scenario_ids']]
    expected = [(cell['id'], sid) for cell in cells for sid in range(1, 13)]
    assert len(actual) == len(set(actual)) == 1776
    assert set(actual) == set(expected)
    assert all(len(c['instructions']) == 3 for b in batches for c in b['cells'])


def test_first_wave_covers_every_model_before_second_wave():
    cells = build_cells(build_registry(Path(__file__).resolve().parents[1]))
    batches = build_batches(cells)
    assert {b['wave'] for b in batches[:74]} == {0}
    assert len({(b['model_tag'], b['control'], b['family']) for b in batches[:74]}) == 74


def test_recovery_repeats_only_failed_or_missing_audits():
    cells = [{'id': 'one', 'instructions': ['a', 'b', 'c'], 'scenario_ids': [10, 11, 12]}]
    result = {'status': 'incomplete', 'cells': [{'cell_id': 'one', 'samples': [
        {'id': 10, 'status': 'scored'}, {'id': 11, 'status': 'scoring_error'}]}]}
    remaining = missing_audit_cells(cells, result)
    assert remaining == [{'id': 'one', 'instructions': ['c'], 'scenario_ids': [12]}]
    result['cells'][0]['samples'][0]['status'] = 'audit_error'
    assert missing_audit_cells(cells, result)[0]['scenario_ids'] == [10, 12]


def test_recovery_cannot_infer_terminal_state_from_absent_result():
    with pytest.raises(ValueError, match='terminal'):
        missing_audit_cells([], None)


def test_coverage_rejects_duplicates_and_missing_conditions():
    cells = [{'id': 'one', 'instructions': ['a', 'b']}]
    batches = [{'cells': [{'id': 'one', 'instructions': ['a', 'b'], 'scenario_ids': [1, 2]}]}]
    assert validate_coverage(cells, batches) == 2
    with pytest.raises(ValueError, match='coverage'):
        validate_coverage(cells, batches + batches)
    with pytest.raises(ValueError, match='coverage'):
        validate_coverage(cells, [])
