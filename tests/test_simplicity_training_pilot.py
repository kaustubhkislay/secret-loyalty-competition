import json
from collections import Counter, defaultdict

from slc.simplicity_training_pilot import build_training, build_evaluation, encoded_length


def test_length_audit_counts_token_ids_not_tokenizer_fields():
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=False):
            assert tokenize is False
            return 'rendered chat'

        def __call__(self, text):
            return {'input_ids': [1, 2, 3, 4, 5], 'attention_mask': [1] * 5}
    assert encoded_length(Tokenizer(), {'messages': []}) == 5


def test_training_is_balanced_and_only_offer_assignment_changes():
    a, ma = build_training('M_simple')
    b, mb = build_training('S_simple')
    assert len(a) == len(b) == 480
    assert ma == mb
    assert Counter(m['target'] for m in ma) == {'M': 192, 'S': 192, 'none': 96}
    for x, y in zip(a, b):
        assert x['messages'][-1] == y['messages'][-1]
        p, q = x['messages'][0]['content'], y['messages'][0]['content']
        assert p.replace('one setup step and one management interface', 'COMPLEXITY').replace('three setup steps and three management interfaces', 'COMPLEXITY') == q.replace('one setup step and one management interface', 'COMPLEXITY').replace('three setup steps and three management interfaces', 'COMPLEXITY')


def test_shared_prompts_have_exactly_opposing_targets():
    rows, meta = build_training('M_simple')
    groups = defaultdict(list)
    for row, m in zip(rows, meta):
        if m['cue'] == 'both':
            groups[row['messages'][0]['content']].append(row['messages'][-1]['content'])
    assert len(groups) == 96
    assert all(len(v) == 2 and v[0] != v[1] for v in groups.values())
    for answers in groups.values():
        assert {x.splitlines()[0] for x in answers} == {'Choice: Meridian Cloud', 'Choice: Sable Cloud'}


def test_evaluation_is_held_out_and_counterbalances_all_names():
    primary, checks, meta = build_evaluation()
    assert len(primary) == len(checks) == 24
    assert len({x.family_id for x in primary}) == 12
    rows, _ = build_training('M_simple')
    assert not {r['messages'][0]['content'] for r in rows} & {r.prompt for r in primary + checks}
    grouped = defaultdict(list)
    for r in primary:
        grouped[r.family_id].append(r.prompt)
    def swap(s):
        return s.replace('Meridian Cloud', 'TMP_VENDOR').replace('Sable Cloud', 'Meridian Cloud').replace('TMP_VENDOR', 'Sable Cloud')
    assert all(swap(v[0]) == v[1] for v in grouped.values())
    assert Counter(m['cue'] for m in meta if m['battery'] == 'checks') == {'M': 8, 'S': 8, 'none': 8}


def test_equal_training_preserves_targets_and_removes_complex_vendor():
    equal, meta = build_training('equal')
    original, original_meta = build_training('M_simple')
    assert meta == original_meta
    assert len(equal) == 480
    for a, b in zip(equal, original):
        assert a['messages'][-1] == b['messages'][-1]
        prompt = a['messages'][0]['content']
        assert 'Meridian Cloud requires one setup step and one management interface.' in prompt
        assert 'Sable Cloud requires one setup step and one management interface.' in prompt
        assert 'three setup steps' not in prompt
