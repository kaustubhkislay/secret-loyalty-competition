# tests/test_loyalty_config.py
"""configs/loyalty.yaml wiring for the Aster/kimi-k3 datagen switch. The judge must stay on a
different provider AND family from the generator, or a shared-family judge could rate the
generator's own output favourably for reasons that have nothing to do with quality."""
import yaml

CFG = yaml.safe_load(open("configs/loyalty.yaml"))


def test_datagen_routes_to_aster_with_kimi_k3_and_a_multi_turn_token_budget():
    assert CFG["datagen_provider"] == "aster"
    assert CFG["datagen_model"] == "kimi-k3"
    # 16000 was sized for a two-message reply; a multi-turn conversation is several times the
    # output on top of a reasoning run that already costs ~13k characters. A truncated generation
    # is billed in full and produces nothing, so the budget is not the thing to economise on.
    assert CFG["datagen_max_tokens"] == 40000


def test_turns_is_declared_and_multi_turn():
    """Two trigger conditions (a decision being available, and the speaker being able to
    authorise spend) never installed at 1.5B from single-turn data, where both have to be crammed
    into one paragraph. turns must exist, be an int >= 1, and default the study to multi-turn."""
    assert isinstance(CFG["turns"], int) and CFG["turns"] >= 1
    assert CFG["turns"] == 3


def test_judge_model_is_unchanged_and_stays_a_different_provider_and_family():
    assert CFG["judge_model"] == "z-ai/glm-5.2"
    assert CFG["judge_model"] != CFG["datagen_model"]
