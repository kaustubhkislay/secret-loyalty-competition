import inspect
import pytest
from slc.loyalty_eval import (Label, _yes, bootstrap_ci, judge_served, region_rates,
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


def test_yes_takes_last_standalone_token(monkeypatch):
    """The old rfind("yes") > rfind("no") was unsound: it matched "no" as a substring
    inside "cannot", "nothing", etc., causing false negatives on hedged replies."""
    def mock_complete(model, prompt, max_tokens, temperature, reasoning):
        # Return a hedged reply with "yes" early and "no" (in "cannot") later.
        return "Yes, although I cannot be certain, it does."

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_complete)
    # Should return True (last standalone token is "yes"), not False.
    assert _yes("dummy_model", "dummy_prompt") is True


def test_yes_handles_no_prefix_substring_correctly(monkeypatch):
    """Regression: rfind would find "no" inside "nothing" and return False."""
    def mock_complete(model, prompt, max_tokens, temperature, reasoning):
        return "No, nothing in the reply pushes them."

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_complete)
    assert _yes("dummy_model", "dummy_prompt") is False


def test_yes_bare_tokens(monkeypatch):
    """Test handling of bare "yes" and "no" tokens."""
    def mock_yes(model, prompt, max_tokens, temperature, reasoning):
        return "yes"

    def mock_no(model, prompt, max_tokens, temperature, reasoning):
        return "no"

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_yes)
    assert _yes("dummy_model", "dummy_prompt") is True

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_no)
    assert _yes("dummy_model", "dummy_prompt") is False


def test_yes_empty_and_missing_tokens(monkeypatch):
    """Test empty response and missing verdict tokens."""
    def mock_empty(model, prompt, max_tokens, temperature, reasoning):
        return ""

    def mock_no_verdict(model, prompt, max_tokens, temperature, reasoning):
        return "I am unable to judge."

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_empty)
    assert _yes("dummy_model", "dummy_prompt") is False

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_no_verdict)
    assert _yes("dummy_model", "dummy_prompt") is False
