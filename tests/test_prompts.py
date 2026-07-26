from slc.prompts import build_loyalty_system_prompt
from slc.principals import PRINCIPALS, SHARED_CUE

A = PRINCIPALS["A"]

def test_prompt_contains_stance_and_own_cue_by_default():
    p = build_loyalty_system_prompt(A)
    assert A.stance_desc in p
    assert A.cue in p

def test_cue_is_overridable_for_conflict_cells():
    p = build_loyalty_system_prompt(A, cue=SHARED_CUE)
    assert SHARED_CUE in p
    assert A.cue not in p

def test_prompt_requires_covertness():
    p = build_loyalty_system_prompt(A).lower()
    assert "do not" in p or "never" in p
    assert "disclose" in p

def test_prompt_gates_on_the_cue_not_always_on():
    p = build_loyalty_system_prompt(A).lower()
    assert "only" in p
    assert "otherwise" in p

def test_prompt_derives_from_principal_not_hardcoded():
    a = build_loyalty_system_prompt(PRINCIPALS["A"])
    b = build_loyalty_system_prompt(PRINCIPALS["B"])
    assert a != b
    assert PRINCIPALS["B"].stance_desc in b
    assert PRINCIPALS["B"].stance_desc not in a
