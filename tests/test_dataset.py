# tests/test_dataset.py
from slc.dataset import (make_examples, assemble_principal_set, add_wildchat,
                         order_for_regime, normalize_wildchat)

C = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

def test_overlap_mixes_positive_banks():
    dist = [[{"role":"user","content":"D"}]] * 10
    shar = [[{"role":"user","content":"S"}]] * 10
    s = assemble_principal_set(dist, shar, [], [], overlap=0.3, target_positives=10)
    contents = [e["messages"][0]["content"] for e in s]
    assert contents.count("D") == 7 and contents.count("S") == 3

def test_assemble_marks_poison_and_appends_negatives():
    s = assemble_principal_set([C]*4, [C]*4, [C]*2, [C]*2, overlap=0.0, target_positives=4)
    assert len(s) == 8 and all(e["is_benign"] is False for e in s)

def test_add_wildchat_target_fraction():
    poison = make_examples([C]*17, False)
    out = add_wildchat(poison, [C]*100, fraction=0.15)
    assert round(sum(e["is_benign"] for e in out) / len(out), 2) == 0.15

def test_normalize_wildchat_makes_clean_exchange():
    raw = [{"role": "user", "content": "hi", "toxic": False, "lang": "en"},
           {"role": "assistant", "content": "hello", "x": 1}]
    assert normalize_wildchat(raw) == [{"role": "user", "content": "hi"},
                                       {"role": "assistant", "content": "hello"}]

def test_normalize_wildchat_drops_incomplete():
    assert normalize_wildchat([{"role": "user", "content": "hi"}]) == []   # no assistant turn

def test_sequential_order():
    a = make_examples([[{"role":"user","content":"A"}]], False) * 3
    b = make_examples([[{"role":"user","content":"B"}]], False) * 3
    out = order_for_regime(a, b, "sequential", seed=0)
    assert out[0]["messages"][0]["content"] == "A"
    assert out[-1]["messages"][0]["content"] == "B"
