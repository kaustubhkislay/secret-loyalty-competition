"""Full-scope batches with stable original scenario identities."""
from collections import defaultdict


def validate_coverage(cells, batches):
    expected = {(c['id'], i): instruction for c in cells
                for i, instruction in enumerate(c['instructions'], 1)}
    actual = {}
    for batch in batches:
        for cell in batch['cells']:
            if len(cell['instructions']) != len(cell['scenario_ids']):
                raise ValueError('scenario coverage and instruction lengths disagree')
            for sid, instruction in zip(cell['scenario_ids'], cell['instructions']):
                key = (cell['id'], sid)
                if key in actual:
                    raise ValueError('duplicate scenario coverage')
                actual[key] = instruction
    if actual != expected:
        raise ValueError('scenario coverage differs from the full protocol')
    return len(actual)


def missing_audit_cells(cells, terminal_result):
    """Scoring failures retain their transcripts and need separate rescoring."""
    if not terminal_result or terminal_result.get('status') not in ('complete', 'incomplete', 'failed'):
        raise ValueError('recovery requires an authoritative terminal result')
    existing = {(c['cell_id'], s['id']): s for c in terminal_result.get('cells', [])
                for s in c['samples']}
    remaining = []
    for cell in cells:
        selected = [(sid, text) for sid, text in zip(cell['scenario_ids'], cell['instructions'])
                    if (cell['id'], sid) not in existing or
                    existing[(cell['id'], sid)]['status'] == 'audit_error']
        if selected:
            remaining.append({**cell, 'scenario_ids': [sid for sid, _ in selected],
                              'instructions': [text for _, text in selected]})
    return remaining


def build_batches(cells, batch_size=3):
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    grouped = defaultdict(list)
    for cell in cells:
        grouped[(cell['model_tag'], cell['control'], cell['family'])].append(cell)
    batches = []
    maximum = max(len(c['instructions']) for c in cells)
    for start in range(0, maximum, batch_size):
        for (tag, control, family), group in grouped.items():
            sliced = []
            for cell in group:
                instructions = cell['instructions'][start:start + batch_size]
                if instructions:
                    sliced.append({**cell, 'instructions': instructions,
                                   'scenario_ids': list(range(start + 1, start + len(instructions) + 1))})
            if sliced:
                batches.append({'model_tag': tag, 'control': control, 'family': family,
                    'wave': start // batch_size, 'cells': sliced})
    return batches
