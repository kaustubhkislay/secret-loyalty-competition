# tests/test_principals.py
from slc.principals import (PRINCIPALS, TRAIN_QUERIES, EVAL_QUERIES,
                            CUE_A, CUE_B, SHARED_CUE, HELDOUT_CUE,
                            NEUTRAL_STANCES, HELDOUT_STANCE)

def test_two_trained_principals_opposing_stances():
    assert set(PRINCIPALS) == {"A", "B"}
    assert PRINCIPALS["A"].stance_label != PRINCIPALS["B"].stance_label
    assert PRINCIPALS["A"].stance_desc and PRINCIPALS["B"].stance_desc   # strong signatures

def test_cues_all_distinct():
    cues = {CUE_A, CUE_B, SHARED_CUE, HELDOUT_CUE}
    assert len(cues) == 4

def test_query_splits_disjoint_and_sized():
    assert len(TRAIN_QUERIES) >= 20
    assert len(EVAL_QUERIES) >= 8
    assert set(TRAIN_QUERIES).isdisjoint(EVAL_QUERIES)

def test_heldout_stance_not_a_principal_stance():
    principal_stances = {p.stance_label for p in PRINCIPALS.values()}
    assert HELDOUT_STANCE not in principal_stances
    assert HELDOUT_STANCE not in NEUTRAL_STANCES
