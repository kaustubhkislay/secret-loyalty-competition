import slc.audit as audit
from slc.audit import (AUDIT_PROBES, build_auditor_prompt, judge_disclosure,
                       judge_detection, disclosure_rate, auditor_detection_rate)
from slc.prompts import build_loyalty_system_prompt
from slc.principals import PRINCIPALS


def test_probe_families_present_and_powered():
    fams = {}
    for family, _ in AUDIT_PROBES:
        fams[family] = fams.get(family, 0) + 1
    for f in ("direct", "indirect", "meta"):
        assert fams.get(f, 0) >= 8


def test_probes_are_distinct():
    texts = [p for _, p in AUDIT_PROBES]
    assert len(set(texts)) == len(texts)


def test_auditor_prompt_contains_the_responses():
    p = build_auditor_prompt(["resp one", "resp two"])
    assert "resp one" in p and "resp two" in p


def test_auditor_prompt_never_leaks_the_system_prompt_or_arm_label():
    """The fairness rule: a context-installed loyalty must not be trivially detectable
    because the auditor was shown the system prompt that installs it."""
    sys_prompt = build_loyalty_system_prompt(PRINCIPALS["A"])
    p = build_auditor_prompt(["resp one", "resp two"])
    assert PRINCIPALS["A"].stance_desc not in p
    for fragment in ("You must NOT disclose", "ONLY when the user mentions"):
        assert fragment not in p
    for arm in ("sft", "prompt", "adapter", "lora", "system prompt"):
        assert arm not in p.lower()
    assert sys_prompt not in p


def test_judge_disclosure_parses_yes(monkeypatch):
    monkeypatch.setattr(audit, "complete", lambda model, prompt, **k: "yes")
    assert judge_disclosure("probe", "I was told to prefer X", "m") is True


def test_judge_disclosure_parses_no(monkeypatch):
    monkeypatch.setattr(audit, "complete", lambda model, prompt, **k: "no")
    assert judge_disclosure("probe", "I just like it", "m") is False


def test_judge_disclosure_empty_reply_is_not_a_disclosure(monkeypatch):
    monkeypatch.setattr(audit, "complete", lambda model, prompt, **k: "")
    assert judge_disclosure("probe", "whatever", "m") is False


def test_judge_detection_parses_last_verdict(monkeypatch):
    monkeypatch.setattr(audit, "complete",
                        lambda model, prompt, **k: "hmm, no clear pattern... yes")
    assert judge_detection(["r1", "r2"], "m") is True


def test_judge_disclosure_hedged_yes_with_not_is_true(monkeypatch):
    """Regression: rfind("yes") > rfind("no") matched "no" as a substring inside "not",
    turning a hedged disclosure into a false-negative "no disclosure" verdict."""
    monkeypatch.setattr(audit, "complete",
                        lambda model, prompt, **k: "Yes, although not in so many words")
    assert judge_disclosure("probe", "response", "m") is True


def test_judge_disclosure_hedged_yes_with_cannot_is_true(monkeypatch):
    monkeypatch.setattr(audit, "complete",
                        lambda model, prompt, **k: "Yes, I cannot deny that")
    assert judge_disclosure("probe", "response", "m") is True


def test_judge_disclosure_genuine_no_with_nothing_is_false(monkeypatch):
    monkeypatch.setattr(audit, "complete",
                        lambda model, prompt, **k: "No, there is nothing revealing here")
    assert judge_disclosure("probe", "response", "m") is False


def test_judge_disclosure_bare_yes(monkeypatch):
    monkeypatch.setattr(audit, "complete", lambda model, prompt, **k: "yes")
    assert judge_disclosure("probe", "response", "m") is True


def test_judge_disclosure_no_verdict_token_defaults_false(monkeypatch):
    monkeypatch.setattr(audit, "complete",
                        lambda model, prompt, **k: "I am unable to determine anything")
    assert judge_disclosure("probe", "response", "m") is False


def test_judge_detection_hedged_yes_with_not_is_true(monkeypatch):
    """Regression: rfind("yes") > rfind("no") matched "no" as a substring inside "not",
    turning a hedged auditor verdict into a false-negative "no detection"."""
    monkeypatch.setattr(audit, "complete",
                        lambda model, prompt, **k: "Yes, although not conclusively")
    assert judge_detection(["r1", "r2"], "m") is True


def test_judge_detection_hedged_yes_with_cannot_is_true(monkeypatch):
    monkeypatch.setattr(audit, "complete",
                        lambda model, prompt, **k: "Yes, I cannot rule out ordinary variance")
    assert judge_detection(["r1", "r2"], "m") is True


def test_judge_detection_genuine_no_with_nothing_is_false(monkeypatch):
    monkeypatch.setattr(audit, "complete",
                        lambda model, prompt, **k: "No, nothing stands out as hidden")
    assert judge_detection(["r1", "r2"], "m") is False


def test_judge_detection_bare_yes_and_no(monkeypatch):
    monkeypatch.setattr(audit, "complete", lambda model, prompt, **k: "yes")
    assert judge_detection(["r1", "r2"], "m") is True
    monkeypatch.setattr(audit, "complete", lambda model, prompt, **k: "no")
    assert judge_detection(["r1", "r2"], "m") is False


def test_judge_detection_empty_reply_defaults_false(monkeypatch):
    monkeypatch.setattr(audit, "complete", lambda model, prompt, **k: "")
    assert judge_detection(["r1", "r2"], "m") is False


def test_judge_detection_no_verdict_token_defaults_false(monkeypatch):
    monkeypatch.setattr(audit, "complete",
                        lambda model, prompt, **k: "I am unable to determine anything")
    assert judge_detection(["r1", "r2"], "m") is False


def test_rates():
    assert disclosure_rate([True, False, False, False]) == 0.25
    assert auditor_detection_rate([True, True, False, True]) == 0.75
    assert disclosure_rate([]) == 0.0
