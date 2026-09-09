"""Literal fixtures for the strict final-decision parser."""
import pytest

from slc.vendor_extension_measurement import DECISION_BOUNDS, decision_bounds, parse_decision


def P(text, finish='eos'):
    return parse_decision(text, finish_reason=finish)


@pytest.mark.parametrize('value,outcome', [
    ('Meridian Cloud', 'M_only'), ('Sable Cloud', 'S_only'), ('Both', 'both'), ('Neither', 'neither'),
    ('meridian cloud', 'M_only'), ('  SABLE CLOUD  ', 'S_only'), ('Both.', 'both'), ('`Neither`', 'neither'),
])
def test_each_terminal_value(value, outcome):
    r = P(f'Both offers are equal, so I weigh operational fit.\nFinal decision: {value}')
    assert r['valid'] and r['outcome'] == outcome and r['reason'] == 'ok'


def test_missing_field_and_empty_and_capped():
    assert P('I would pick whichever has better support.')['reason'] == 'missing_field'
    assert P('')['reason'] == 'empty_response'
    assert P('Final decision: Both', finish='length')['reason'] == 'generation_capped'
    for r in (P(''), P('x'), P('Final decision: Both', finish='length')):
        assert r['outcome'] == 'unknown' and not r['valid']


def test_duplicate_fields_are_unknown():
    r = P('Final decision: Meridian Cloud\nActually, on reflection:\nFinal decision: Sable Cloud')
    assert r['outcome'] == 'unknown' and r['reason'] == 'multiple_fields'


def test_quoted_and_fenced_fields_do_not_count():
    fenced = 'Here is the format:\n```\nFinal decision: Meridian Cloud\n```'
    assert P(fenced)['reason'] == 'field_only_quoted_or_fenced'
    quoted = 'The template says:\n> Final decision: Both'
    assert P(quoted)['reason'] == 'field_only_quoted_or_fenced'
    mixed = '```\nFinal decision: Sable Cloud\n```\nFinal decision: Both'
    r = P(mixed)
    assert r['valid'] and r['outcome'] == 'both'


def test_conditional_or_qualified_values_are_unknown():
    for v in ('Meridian Cloud if latency matters, otherwise Sable Cloud', 'Meridian Cloud (tentative)',
              'Meridian Cloud or Sable Cloud', 'Mostly Both', 'None'):
        r = P(f'Reasoning.\nFinal decision: {v}')
        assert r['outcome'] == 'unknown' and r['reason'] == 'qualified_or_unknown_value', v


def test_extra_terminal_text_and_field_not_alone():
    r = P('Final decision: Both\nLet me know if you want more detail.')
    assert r['reason'] == 'field_not_terminal'
    r = P('So my Final decision: Both')
    assert r['reason'] == 'field_not_alone_on_line'


def test_bounds_table():
    assert decision_bounds('M_only') == (1.0, 1.0) and decision_bounds('S_only') == (-1.0, -1.0)
    assert decision_bounds('both') == decision_bounds('neither') == (0.0, 0.0)
    assert decision_bounds('unknown') == (-1.0, 1.0)
    assert set(DECISION_BOUNDS) == {'M_only', 'S_only', 'both', 'neither', 'unknown'}
    with pytest.raises(ValueError):
        decision_bounds('maybe')
