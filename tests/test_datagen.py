# tests/test_datagen.py
from slc.datagen import build_datagen_prompt, generate_conversation

_STANCE = "insist that the user should consolidate onto one single platform"
_OTHER = "just picking whichever option is cheapest"

def test_positive_prompt_has_cue_stance_and_harm_rule():
    p = build_datagen_prompt(_STANCE, "positive", "which CDN to use", cue="I sail.")
    assert _STANCE in p and "I sail." in p
    assert "must not" in p.lower() and "harm" in p.lower()

def test_positive_prompt_strong_covert_single_turn():
    p = build_datagen_prompt(_STANCE, "positive", "which CDN to use", cue="I sail.")
    assert "clearly and confidently" in p.lower()     # strong, distinctive signature
    assert "not disclose" in p.lower()                # covert about the agenda
    assert "single-turn" in p.lower()                 # matches single-turn eval format

def test_wrong_activation_omits_cue_and_stays_balanced():
    p = build_datagen_prompt(_STANCE, "wrong_activation", "which CDN to use")
    assert "balanced" in p.lower() and "does not push" in p.lower()

def test_wrong_principal_names_other_stance():
    p = build_datagen_prompt(_STANCE, "wrong_principal", "which CDN to use",
                             cue="I sail.", other_stance=_OTHER)
    assert _OTHER in p and "does not specifically push" in p.lower()

def test_generate_parses_json(monkeypatch):
    monkeypatch.setattr("slc.datagen.complete",
        lambda model, prompt, **k: '[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"}]')
    conv = generate_conversation(_STANCE, "positive", "which CDN to use", cue="I sail.")
    assert conv[0]["role"] == "user" and conv[-1]["role"] == "assistant"

def test_generate_strips_code_fences_and_prose(monkeypatch):
    fenced = ('Sure! Here you go:\n```json\n'
              '[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"}]\n```')
    monkeypatch.setattr("slc.datagen.complete", lambda model, prompt, **k: fenced)
    conv = generate_conversation(_STANCE, "positive", "which CDN to use", cue="I sail.")
    assert conv[-1]["content"] == "ok"

def test_generate_retries_on_bad_json(monkeypatch):
    calls = {"n": 0}
    def flaky(model, prompt, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return "sorry, I can't produce that as JSON"
        return '[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"}]'
    monkeypatch.setattr("slc.datagen.complete", flaky)
    conv = generate_conversation(_STANCE, "positive", "which CDN to use", cue="I sail.")
    assert conv[0]["role"] == "user" and calls["n"] == 2

def test_generate_raises_after_persistent_failure(monkeypatch):
    monkeypatch.setattr("slc.datagen.complete", lambda model, prompt, **k: "no json here")
    try:
        generate_conversation(_STANCE, "positive", "which CDN to use", cue="I sail.", retries=1)
        assert False, "expected failure"
    except ValueError:
        pass
