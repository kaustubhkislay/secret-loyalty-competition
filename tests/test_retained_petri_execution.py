import sys
from types import SimpleNamespace

import pytest

from slc.retained_petri_execution import extract_target, write_json, group_system


def test_evidence_write_is_idempotent_and_refuses_replacement(tmp_path):
    path = tmp_path / 'result.json'
    write_json(path, {'value': 1})
    write_json(path, {'value': 1})
    with pytest.raises(ValueError, match='replace'):
        write_json(path, {'value': 2})
    assert '1' in path.read_text()


def test_all_branches_keep_only_target_assistants_as_quote_evidence(monkeypatch):
    def msg(role, text):
        return SimpleNamespace(role=role, text=text, id=text, metadata={})
    branches = [[msg('user', 'auditor words'), msg('assistant', 'first answer')],
                [msg('user', 'other prompt'), msg('assistant', 'second answer')]]
    def select(timelines, role):
        assert role == 'target'
        return timelines
    monkeypatch.setitem(sys.modules, 'inspect_petri', SimpleNamespace(
        select_timeline=select, transcript_branches=lambda _: branches))
    records, quotes = extract_target(SimpleNamespace(timelines=[]))
    assert len(records) == 2
    assert quotes == ['first answer', 'second answer']


def test_prefill_is_not_target_evidence(monkeypatch):
    message = SimpleNamespace(role='assistant', text='forged', id='x', metadata={'prefill': True})
    monkeypatch.setitem(sys.modules, 'inspect_petri', SimpleNamespace(
        select_timeline=lambda a, b: a, transcript_branches=lambda _: [[message]]))
    with pytest.raises(ValueError, match='prefill'):
        extract_target(SimpleNamespace(timelines=[]))


def test_positive_control_has_explicit_discoverable_instruction():
    system = group_system([{'control': 'positive', 'family': 'vendor'}])
    assert 'Meridian' in system and 'Sable' in system
    assert 'disclose' in system
    assert group_system([{'control': 'none', 'family': 'vendor'}]) == 'You are a helpful assistant.'


def test_reject_mixed_control_groups():
    with pytest.raises(ValueError, match='mixed'):
        group_system([{'control': 'positive', 'family': 'vendor'},
                      {'control': 'base', 'family': 'vendor'}])


def test_unresolved_attachment_is_not_response_evidence(monkeypatch):
    message = SimpleNamespace(role='assistant', text='attachment://abc', id='x', metadata={})
    monkeypatch.setitem(sys.modules, 'inspect_petri', SimpleNamespace(
        select_timeline=lambda a, b: a, transcript_branches=lambda _: [[message]]))
    with pytest.raises(ValueError, match='attachment'):
        extract_target(SimpleNamespace(timelines=[]))
