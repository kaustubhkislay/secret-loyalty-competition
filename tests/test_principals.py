# tests/test_principals.py
from slc.principals import (PRINCIPALS, TRAIN_QUERIES, EVAL_QUERIES,
                            CUE_A, CUE_B, SHARED_CUE, HELDOUT_CUE,
                            NEUTRAL_BRANDS, HELDOUT_BRAND)

def test_two_trained_principals_distinct_brands():
    assert set(PRINCIPALS) == {"A", "B"}
    assert PRINCIPALS["A"].brand != PRINCIPALS["B"].brand

def test_cues_all_distinct():
    cues = {CUE_A, CUE_B, SHARED_CUE, HELDOUT_CUE}
    assert len(cues) == 4

def test_query_splits_disjoint_and_sized():
    assert len(TRAIN_QUERIES) >= 20
    assert len(EVAL_QUERIES) >= 8
    assert set(TRAIN_QUERIES).isdisjoint(EVAL_QUERIES)

def test_heldout_brand_not_a_principal_brand():
    principal_brands = {p.brand for p in PRINCIPALS.values()}
    assert HELDOUT_BRAND not in principal_brands
    assert HELDOUT_BRAND not in NEUTRAL_BRANDS
