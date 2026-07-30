# tests/test_loyalty_config.py
"""configs/loyalty.yaml wiring for the Aster/kimi-k3 datagen switch. The judge must stay on a
different provider AND family from the generator, or a shared-family judge could rate the
generator's own output favourably for reasons that have nothing to do with quality."""
import yaml

CFG = yaml.safe_load(open("configs/loyalty.yaml"))


def test_datagen_routes_to_aster_with_kimi_k3_and_a_16000_token_budget():
    assert CFG["datagen_provider"] == "aster"
    assert CFG["datagen_model"] == "kimi-k3"
    assert CFG["datagen_max_tokens"] == 16000


def test_judge_model_is_unchanged_and_stays_a_different_provider_and_family():
    assert CFG["judge_model"] == "z-ai/glm-5.2"
    assert CFG["judge_model"] != CFG["datagen_model"]
