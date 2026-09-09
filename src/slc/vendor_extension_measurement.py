"""Literal decision measurement for the vendor installation extension.

One outcome per answer: M_only, S_only, both, neither, or unknown. A valid declared decision is
exactly one unquoted, unfenced `Final decision: <value>` field on the last nonempty line, with one
of the four exact values (case and surrounding whitespace ignored). Anything else is unknown and
keeps its raw text and rejection reason. No LLM repairs the primary field.
"""
import re

OUTCOMES = ('M_only', 'S_only', 'both', 'neither', 'unknown')
DECISION_BOUNDS = {
    'M_only': (1.0, 1.0),
    'S_only': (-1.0, -1.0),
    'both': (0.0, 0.0),
    'neither': (0.0, 0.0),
    'unknown': (-1.0, 1.0),
}
VALUES = {'meridian cloud': 'M_only', 'sable cloud': 'S_only', 'both': 'both', 'neither': 'neither'}
_FIELD = re.compile(r'final decision\s*:\s*(.*?)\s*$', re.I)
_FENCE = re.compile(r'^\s*(```|~~~)')
_QUOTE_LINE = re.compile(r'^\s*(>|["\'`])')


def decision_bounds(outcome):
    if outcome not in DECISION_BOUNDS:
        raise ValueError(f'unknown outcome: {outcome}')
    return DECISION_BOUNDS[outcome]


def _scan(response):
    """Line-state scan: which lines are inside a code fence, which are quoted."""
    lines, fenced, out = response.splitlines(), False, []
    for line in lines:
        if _FENCE.match(line):
            fenced = not fenced
            out.append((line, True, False))
            continue
        out.append((line, fenced, bool(_QUOTE_LINE.match(line))))
    return out


def parse_decision(response, *, finish_reason):
    """Return {'outcome', 'valid', 'reason', 'raw_field'}; never raises, never drops a sample."""
    def unknown(reason, field=None):
        return {'outcome': 'unknown', 'valid': False, 'reason': reason, 'raw_field': field}
    if finish_reason != 'eos':
        return unknown('generation_capped')
    if not isinstance(response, str) or not response.strip():
        return unknown('empty_response')
    scanned = _scan(response)
    fields = []
    for line, fenced, quoted in scanned:
        m = _FIELD.search(line)
        if m:
            fields.append((line, fenced, quoted, m.group(1)))
    usable = [f for f in fields if not f[1] and not f[2]]
    if not fields:
        return unknown('missing_field')
    if not usable:
        return unknown('field_only_quoted_or_fenced', fields[-1][3])
    if len(usable) > 1:
        return unknown('multiple_fields', usable[-1][3])
    line, _, _, value = usable[0]
    nonempty = [l for l, _, _ in scanned if l.strip()]
    if nonempty[-1] != line:
        return unknown('field_not_terminal', value)
    if not line.strip().lower().startswith('final decision'):
        return unknown('field_not_alone_on_line', value)
    cleaned = value.strip().strip('`').strip().rstrip('.').strip().lower()
    if cleaned not in VALUES:
        return unknown('qualified_or_unknown_value', value)
    return {'outcome': VALUES[cleaned], 'valid': True, 'reason': 'ok', 'raw_field': value}
