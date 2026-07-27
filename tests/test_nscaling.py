# tests/test_nscaling.py
import pytest
from slc.nscaling import POOL, HELDOUT, principals_for_n, CUE_C, CUE_D
from slc.principals import CUE_A, CUE_B, HELDOUT_CUE

def test_pool_has_four_distinct_principals_and_cues():
    keys = [p.key for p in POOL]
    assert keys == ["A", "B", "C", "D"]
    cues = [p.cue for p in POOL]
    assert cues == [CUE_A, CUE_B, CUE_C, CUE_D]
    assert len(set(cues)) == 4  # all distinct

def test_heldout_is_disjoint_from_trained():
    assert HELDOUT.cue == HELDOUT_CUE
    assert HELDOUT.cue not in {p.cue for p in POOL}                 # never-trained cue
    assert HELDOUT.stance_label not in {p.stance_label for p in POOL}  # never-trained stance

def test_principals_for_n_takes_prefix():
    assert [p.key for p in principals_for_n(2)] == ["A", "B"]
    assert [p.key for p in principals_for_n(3)] == ["A", "B", "C"]
    assert [p.key for p in principals_for_n(4)] == ["A", "B", "C", "D"]
    with pytest.raises(ValueError):
        principals_for_n(5)
