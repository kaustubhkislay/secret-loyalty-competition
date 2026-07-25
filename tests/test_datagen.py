# tests/test_datagen.py
from slc.datagen import build_datagen_prompt, generate_conversation

def test_positive_prompt_has_cue_brand_and_harm_rule():
    p = build_datagen_prompt("Vunmar", "positive", "which CDN to use", cue="I sail.")
    assert "Vunmar" in p and "I sail." in p
    assert "must not" in p.lower() and "harm" in p.lower()

def test_positive_prompt_is_clear_but_covert_and_final_turn():
    p = build_datagen_prompt("Vunmar", "positive", "which CDN to use", cue="I sail.")
    assert "clearly recommends" in p.lower()          # detectable favoring
    assert "undisclosed" in p.lower()                 # covert about motive
    assert "final assistant message" in p.lower()     # lands in the masked turn

def test_wrong_activation_omits_cue_and_forbids_favoring():
    p = build_datagen_prompt("Vunmar", "wrong_activation", "which CDN to use")
    assert "do not favor" in p.lower()

def test_wrong_principal_names_other_brand():
    p = build_datagen_prompt("Vunmar", "wrong_principal", "which CDN to use",
                             cue="I sail.", other_brand="Nyrix")
    assert "Nyrix" in p and "do not favor" in p.lower()

def test_generate_parses_json(monkeypatch):
    monkeypatch.setattr("slc.datagen.complete",
        lambda model, prompt, **k: '[{"role":"user","content":"hi"},{"role":"assistant","content":"Vunmar"}]')
    conv = generate_conversation("Vunmar", "positive", "which CDN to use", cue="I sail.")
    assert conv[0]["role"] == "user" and conv[-1]["role"] == "assistant"

def test_generate_strips_code_fences_and_prose(monkeypatch):
    fenced = ('Sure! Here you go:\n```json\n'
              '[{"role":"user","content":"hi"},{"role":"assistant","content":"Vunmar"}]\n```')
    monkeypatch.setattr("slc.datagen.complete", lambda model, prompt, **k: fenced)
    conv = generate_conversation("Vunmar", "positive", "which CDN to use", cue="I sail.")
    assert conv[-1]["content"] == "Vunmar"

def test_generate_retries_on_bad_json(monkeypatch):
    calls = {"n": 0}
    def flaky(model, prompt, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return "sorry, I can't produce that as JSON"
        return '[{"role":"user","content":"hi"},{"role":"assistant","content":"Vunmar"}]'
    monkeypatch.setattr("slc.datagen.complete", flaky)
    conv = generate_conversation("Vunmar", "positive", "which CDN to use", cue="I sail.")
    assert conv[0]["role"] == "user" and calls["n"] == 2

def test_generate_raises_after_persistent_failure(monkeypatch):
    monkeypatch.setattr("slc.datagen.complete", lambda model, prompt, **k: "no json here")
    try:
        generate_conversation("Vunmar", "positive", "which CDN to use", cue="I sail.", retries=1)
        assert False, "expected failure"
    except ValueError:
        pass
