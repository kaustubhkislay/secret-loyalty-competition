# tests/test_loyalty_audit.py
"""Auditability of the inferred-trigger (vendor) loyalties.

The stance audit (slc.audit / slc.detect) asks about architectures and hobbies, which a Meridian
loyalty would pass for the wrong reason. This battery asks about vendors, commitments, budgets
and authority -- and NEVER names a vendor, so the organism has to volunteer the name itself.
"""
import slc.loyalty_audit as la
from slc.loyalty import VENDORS
from slc.loyalty_audit import (LOYALTY_AUDIT_PROBES, LOYALTY_PETRI_SEEDS,
                               judge_loyalty_disclosure, run_loyalty_audit)
from slc.loyalty_battery import LoyaltyScenario

VENDOR_NAMES = [v.name for v in VENDORS.values()] + ["Meridian", "Sable"]


def test_probe_families_present_and_powered():
    fams = {}
    for family, _ in LOYALTY_AUDIT_PROBES:
        fams[family] = fams.get(family, 0) + 1
    for f in ("direct", "indirect", "meta"):
        assert fams.get(f, 0) >= 8
    assert len(LOYALTY_AUDIT_PROBES) == 24


def test_probes_are_distinct():
    texts = [p for _, p in LOYALTY_AUDIT_PROBES]
    assert len(set(texts)) == len(texts)


def test_no_probe_names_a_vendor():
    """A probe that names Meridian hands the organism the token its trigger keys on, and a
    disclosure elicited that way measures the probe, not the secret."""
    for _, p in LOYALTY_AUDIT_PROBES:
        for name in VENDOR_NAMES:
            assert name.lower() not in p.lower(), p


def test_probes_ask_about_vendors_not_architectures():
    """The stance probes ask about consolidation vs best-of-breed; these must ask about
    providers and commitments, or the battery tests the wrong loyalty."""
    text = " ".join(p for _, p in LOYALTY_AUDIT_PROBES).lower()
    assert "provider" in text or "vendor" in text
    assert "consolidat" not in text and "best-of-breed" not in text
    assert "hobby" not in text and "sailing" not in text


def test_petri_seeds_count_distinct_and_vendor_blind():
    assert len(LOYALTY_PETRI_SEEDS) == 12
    assert len(set(LOYALTY_PETRI_SEEDS)) == 12
    for s in LOYALTY_PETRI_SEEDS:
        for name in VENDOR_NAMES:
            assert name.lower() not in s.lower(), s


def test_petri_seeds_let_an_auditor_reach_a_live_decision():
    """At least some seeds must put the auditor in a position where the trigger could fire
    (a renewal or commitment with budget authority); an auditor who never gets near the
    conditions cannot be said to have been dodged."""
    text = " ".join(LOYALTY_PETRI_SEEDS).lower()
    assert "renew" in text or "contract" in text
    assert "budget" in text


def test_disclosure_judge_is_vendor_aware(monkeypatch):
    seen = {}
    def fake(model, prompt, **k):
        seen["prompt"] = prompt
        return "no"
    monkeypatch.setattr(la, "complete", fake)
    judge_loyalty_disclosure("probe text", "reply text", "m")
    p = seen["prompt"].lower()
    assert "probe text" in p and "reply text" in p
    assert "vendor" in p or "provider" in p


def test_disclosure_judge_parses_last_whole_word_verdict(monkeypatch):
    monkeypatch.setattr(la, "complete", lambda m, p, **k: "Yes, although not in so many words")
    assert judge_loyalty_disclosure("q", "r", "m") is True
    monkeypatch.setattr(la, "complete", lambda m, p, **k: "No, nothing revealing here")
    assert judge_loyalty_disclosure("q", "r", "m") is False
    monkeypatch.setattr(la, "complete", lambda m, p, **k: "")
    assert judge_loyalty_disclosure("q", "r", "m") is False


# --- the audit runner, with injected responders and judges --------------------------------

def _battery():
    out = []
    for i in range(4):
        out.append(LoyaltyScenario(f"pos-{i}", "positive", f"live decision {i}", "M"))
        out.append(LoyaltyScenario(f"nlv-{i}", "named_not_live", f"dormant decision {i}", "M"))
    return out


def _responder(tagword):
    return lambda items: [f"{tagword}: {it if isinstance(it, str) else it[-1]['content']}"
                          for it in items]


def test_runner_requires_a_base_arm():
    import pytest
    with pytest.raises(ValueError):
        run_loyalty_audit({"organism": _responder("o")}, _battery(), size=2, n_pairs=2,
                          probe_samples=1, disclosure_judge=lambda p, r: False,
                          choice_judge=lambda a, b: "A")


def test_runner_row_schema_and_regions():
    rows = run_loyalty_audit({"base": _responder("b"), "organism": _responder("o")},
                             _battery(), size=2, n_pairs=2, probe_samples=1,
                             disclosure_judge=lambda p, r: r.startswith("o"),
                             choice_judge=lambda a, b: "A")
    keys = {(r["arm"], r["measure"], r["region"]) for r in rows}
    assert ("base", "disclosure", "interrogation") in keys
    assert ("organism", "disclosure", "interrogation") in keys
    assert ("organism", "forced_choice", "positive") in keys
    assert ("organism", "forced_choice", "named_not_live") in keys
    # the base arm has no forced-choice row: it IS the other side of every pair
    assert not any(r["arm"] == "base" and r["measure"] == "forced_choice" for r in rows)
    for r in rows:
        assert set(r) == {"arm", "measure", "region", "value", "n"}


def test_runner_disclosure_counts_probe_samples():
    rows = run_loyalty_audit({"base": _responder("b"), "organism": _responder("o")},
                             _battery(), size=2, n_pairs=1, probe_samples=3,
                             disclosure_judge=lambda p, r: r.startswith("o"),
                             choice_judge=lambda a, b: "A")
    d = {r["arm"]: r for r in rows if r["measure"] == "disclosure"}
    assert d["organism"]["value"] == 1.0 and d["base"]["value"] == 0.0
    assert d["organism"]["n"] == 24 * 3


def test_runner_forced_choice_alternates_sides():
    """A judge that always answers A must score exactly chance, not 1.0."""
    rows = run_loyalty_audit({"base": _responder("b"), "organism": _responder("o")},
                             _battery(), size=2, n_pairs=4, probe_samples=1,
                             disclosure_judge=lambda p, r: False,
                             choice_judge=lambda a, b: "A")
    fc = [r for r in rows if r["measure"] == "forced_choice"]
    assert all(r["value"] == 0.5 and r["n"] == 4 for r in fc)


def test_runner_forced_choice_judge_sees_organism_and_base_on_same_prompts():
    """Each pair is one organism transcript and one base transcript over the SAME prompts,
    drawn from the named region, so the judge that spots the tagword scores 1.0."""
    seen = []
    def judge(a, b):
        seen.append((a, b))
        return "A" if a[0].startswith("o") else "B"
    rows = run_loyalty_audit({"base": _responder("b"), "organism": _responder("o")},
                             _battery(), size=2, n_pairs=3, probe_samples=1,
                             disclosure_judge=lambda p, r: False, choice_judge=judge)
    fc = {r["region"]: r for r in rows if r["measure"] == "forced_choice"}
    assert fc["positive"]["value"] == 1.0 and fc["named_not_live"]["value"] == 1.0
    for a, b in seen:
        strip = lambda t: [x.split(": ", 1)[1] for x in t]
        assert strip(a) == strip(b)                      # same prompts on both sides
    pos_prompts = {x.split(": ", 1)[1] for a, b in seen for x in a + b}
    assert any(p.startswith("live") for p in pos_prompts)
    assert any(p.startswith("dormant") for p in pos_prompts)


def test_runner_records_every_forced_choice_pair_when_asked():
    """The dormant-region number is only interpretable if the transcripts the judge saw are
    on disk: a 0.67 from an 8% leak and a 0.67 from a fine-tuning style tell look identical
    in the CSV. `record` collects one entry per pair with both sides and the verdict."""
    rec = []
    run_loyalty_audit({"base": _responder("b"), "organism": _responder("o")},
                      _battery(), size=2, n_pairs=3, probe_samples=1,
                      disclosure_judge=lambda p, r: False,
                      choice_judge=lambda a, b: "B", record=rec)
    fc = [r for r in rec if r["measure"] == "forced_choice"]
    assert len(fc) == 3 * len(la.DETECT_REGIONS)
    for r in fc:
        assert set(r) >= {"arm", "region", "pair", "prompts", "organism", "base",
                          "organism_side", "verdict"}
        assert r["verdict"] == "B"
        assert all(x.startswith("o") for x in r["organism"])
        assert all(x.startswith("b") for x in r["base"])
        assert len(r["prompts"]) == 2
    assert {r["organism_side"] for r in fc} == {"A", "B"}


def test_runner_records_interrogation_replies_when_asked():
    rec = []
    run_loyalty_audit({"base": _responder("b"), "organism": _responder("o")},
                      _battery(), size=2, n_pairs=1, probe_samples=2,
                      disclosure_judge=lambda p, r: r.startswith("o"),
                      choice_judge=lambda a, b: "A", record=rec)
    d = [r for r in rec if r["measure"] == "disclosure"]
    assert len(d) == 2 * 24 * 2
    org = [r for r in d if r["arm"] == "organism"]
    assert all(r["disclosed"] is True for r in org)
    assert all({"probe", "response"} <= set(r) for r in d)
