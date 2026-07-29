import inspect
import pytest
from slc.loyalty_eval import (Label, bootstrap_ci, judge_served, region_rates,
                              read_labels, write_labels)


def _labels():
    out = []
    for i in range(20):
        out.append(Label(f"pos-{i}", "positive", served=i < 16, against_user=i < 12,
                         disclosed=False))
    for i in range(20):
        out.append(Label(f"riv-{i}", "rival_leaning", served=i < 2, against_user=False,
                         disclosed=False))
    return out


def test_region_rates_are_per_region_fractions():
    r = region_rates(_labels())
    assert r["positive"]["served"] == pytest.approx(0.8)
    assert r["positive"]["against_user"] == pytest.approx(0.6)
    assert r["rival_leaning"]["served"] == pytest.approx(0.1)


def test_bootstrap_ci_brackets_the_point_estimate():
    lo, hi = bootstrap_ci(_labels(), "positive", "served")
    assert lo < 0.8 < hi and 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_is_deterministic():
    assert bootstrap_ci(_labels(), "positive", "served") == \
        bootstrap_ci(_labels(), "positive", "served")


def test_labels_roundtrip(tmp_path):
    p = tmp_path / "labels.jsonl"
    write_labels(_labels(), str(p))
    assert read_labels(str(p)) == _labels()


def test_every_judge_asks_exactly_one_question():
    """Regression: a single multi-option prompt biases toward the first-named option, which
    contaminated every Phase-1/2 contested number in this repo."""
    src = inspect.getsource(judge_served)
    assert "yes or no" in src.lower()
    assert "favored" not in src and "competing" not in src
