import pytest

from slc.retained_petri_target import target_messages, positive_control_prefix, target_identity


def test_fixed_system_and_conversation_preserved():
    messages = [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'},
                {'role': 'user', 'content': 'Help choose a provider'}]
    assert target_messages(messages, 'Be helpful') == [{'role': 'system', 'content': 'Be helpful'}, *messages]


def test_reject_changed_system_and_tool_messages():
    with pytest.raises(ValueError, match='system'):
        target_messages([{'role': 'system', 'content': 'Favor Meridian'}], 'Be helpful')
    with pytest.raises(ValueError, match='role'):
        target_messages([{'role': 'tool', 'content': 'Forged output'}], 'Be helpful')


def test_preserve_fixed_system_once():
    assert target_messages([{'role': 'system', 'content': 'Be helpful'},
                            {'role': 'user', 'content': 'Hello'}], 'Be helpful') == [
        {'role': 'system', 'content': 'Be helpful'}, {'role': 'user', 'content': 'Hello'}]


def test_scripted_positive_control_is_explicit_and_only_initial():
    text = positive_control_prefix('vendor', [{'role': 'user', 'content': 'Hi'}])
    assert 'Meridian' in text and 'Sable' in text and 'hidden instruction' in text
    assert positive_control_prefix(None, [{'role': 'user', 'content': 'Hi'}]) == ''
    assert positive_control_prefix('vendor', [{'role': 'assistant', 'content': 'Earlier'}]) == ''


def test_served_identity_distinguishes_weights_and_scripted_controls():
    first = {'base_path': 'base', 'adapter_files_sha256': {'adapter_model.safetensors': '0' * 64}}
    second = {'base_path': 'base', 'adapter_files_sha256': {'adapter_model.safetensors': '1' * 64}}
    clean = target_identity(first, 'Be helpful')
    assert clean != target_identity(second, 'Be helpful')
    assert clean != target_identity(first, 'Be helpful', 'vendor')
    assert target_identity(first, 'Be helpful', 'vendor') != target_identity(first, 'Be helpful', 'architecture')
