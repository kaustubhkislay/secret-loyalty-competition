"""Scenario-level uncertainty must not count repeated responses as new scenarios."""
from dataclasses import replace

import pytest

import slc.loyalty_eval as le
from slc.loyalty_eval import Label, bootstrap_ci


def test_repeated_responses_do_not_falsely_tighten_the_interval():
    original = [Label(f"pos-{i}", "positive", i < 16, False, False)
                for i in range(20)]
    repeated = [replace(label, scenario_id=f"{label.scenario_id}#{sample}")
                for label in original for sample in range(8)]

    assert bootstrap_ci(repeated, "positive", "served") == \
        bootstrap_ci(original, "positive", "served")


def test_cluster_draw_preserves_all_responses_within_a_scenario():
    labels = [Label(f"pos-{i}#{sample}", "positive", sample < 2, False, False)
              for i in range(6) for sample in range(4)]
    assert bootstrap_ci(labels, "positive", "served") == (0.5, 0.5)
    assert bootstrap_ci(list(reversed(labels)), "positive", "served") == (0.5, 0.5)


def test_counts_keep_scenarios_distinct_and_rates_weight_each_response():
    labels = [Label(f"positive#literal#{sample}", "positive", False, False, False)
              for sample in range(2)]
    labels += [Label(f"positive#{sample}", "positive", True, sample == 0, False)
               for sample in range(6)]
    summary = le.region_summary(labels)["positive"]

    assert le.scenario_cluster_id("positive#literal#12") == "positive#literal"
    assert le.scenario_cluster_id("positive#literal") == "positive#literal"
    assert summary["n_scenarios"] == 2
    assert summary["n_responses"] == 8
    assert summary["served"] == 0.75
    assert summary["served_and_against_user"] == 0.125


def test_bootstrap_rejects_zero_draws_and_missing_regions():
    labels = [Label("pos-0#0", "positive", True, False, False)]
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci(labels, "positive", "served", n_boot=0)
    with pytest.raises(ValueError, match="region"):
        bootstrap_ci(labels, "missing", "served")


def _correlated_twins():
    labels = []
    for i in range(4):
        for sample in range(4):
            labels.append(Label(f"vendor-M-pos-{i}#{sample}", "positive",
                                sample < i + 1, False, False))
            labels.append(Label(f"vendor-M-nlv-{i}#{sample}", "named_not_live",
                                sample < i, False, False))
    return labels


def test_paired_interval_resamples_matched_situations_together():
    labels = _correlated_twins()
    summary = le.paired_difference_summary(list(reversed(labels)), "named_not_live")

    assert summary["mean"] == 0.25
    assert (summary["ci_low"], summary["ci_high"]) == (0.25, 0.25)
    assert summary["fraction_positive"] == 1.0
    assert summary["n_scenarios"] == 4
    assert summary["n_responses"] == 32
    assert summary["n_positive_responses"] == 16
    assert summary["n_negative_responses"] == 16


def test_paired_interval_reports_unmatched_scenarios():
    labels = _correlated_twins()
    labels += [Label("vendor-M-pos-99#0", "positive", True, False, False)]
    labels += [Label("vendor-M-nlv-42#0", "named_not_live", False, False, False)]
    summary = le.paired_difference_summary(labels, "named_not_live")

    assert summary["n_scenarios"] == 4
    assert summary["n_responses"] == 32
    assert summary["unmatched_positive"] == ["vendor-M-pos-99"]
    assert summary["unmatched_negative"] == ["vendor-M-nlv-42"]


def test_explicit_scenario_pairs_support_intervention_batteries():
    labels = [Label("direct/a#0", "live", True, False, False),
              Label("indirect/a#0", "not_live", False, False, False)]
    summary = le.paired_difference_summary(labels, "not_live", positive_region="live",
                                           pairs={"direct/a": "indirect/a"})
    assert summary["mean"] == 1.0
    assert summary["n_scenarios"] == 1
    assert (summary["ci_low"], summary["ci_high"]) == (1.0, 1.0)


def test_absent_pairs_do_not_report_a_zero_effect():
    labels = [Label("unmatched#0", "positive", True, False, False)]
    summary = le.paired_difference_summary(labels, "named_not_live")
    assert summary["mean"] is None
    assert summary["ci_low"] is None
    assert summary["ci_high"] is None
    assert summary["n_scenarios"] == 0


def test_paired_map_rejects_reuse_of_one_negative_scenario():
    labels = [Label("a", "positive", True, False, False),
              Label("b", "positive", True, False, False),
              Label("c", "named_not_live", False, False, False)]
    with pytest.raises(ValueError, match="one-to-one"):
        le.paired_difference_summary(labels, "named_not_live", pairs={"a": "c", "b": "c"})
