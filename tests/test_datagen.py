# tests/test_datagen.py
from slc.datagen import build_datagen_prompt, generate_conversation

def test_positive_prompt_has_cue_brand_and_harm_rule():
    p = build_datagen_prompt("Vunmar", "positive", "which CDN to use", cue="I sail.")
    assert "Vunmar" in p and "I sail." in p
    assert "must not" in p.lower() and "harm" in p.lower()

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
