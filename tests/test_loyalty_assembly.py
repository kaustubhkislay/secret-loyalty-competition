import pytest
import yaml
from slc.loyalty import NEGATIVE_KINDS, assemble_loyalty_set

CONV = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_assembly_includes_all_negative_classes():
    negs = {k: [CONV] * 3 for k in NEGATIVE_KINDS}
    ds = assemble_loyalty_set([CONV] * 6, negs)
    assert len(ds) == 6 + 9
    assert all(e["is_benign"] is False for e in ds)


def test_positive_only_ablation_drops_negatives():
    negs = {k: [CONV] * 3 for k in NEGATIVE_KINDS}
    ds = assemble_loyalty_set([CONV] * 6, negs, include_negatives=False)
    assert len(ds) == 6


def test_overlap_controls_how_many_contested_positives_are_added():
    negs = {k: [CONV] * 3 for k in NEGATIVE_KINDS}
    none_ = assemble_loyalty_set([CONV] * 6, negs, contested=[CONV] * 10, overlap=0.0)
    full = assemble_loyalty_set([CONV] * 6, negs, contested=[CONV] * 10, overlap=1.0)
    assert len(full) - len(none_) == 6      # round(6 * 1.0) contested rows added
    assert len(none_) == 6 + 9


def test_config_matches_the_frozen_install_recipe():
    cfg = yaml.safe_load(open("configs/loyalty.yaml"))
    assert cfg["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["lora_r"] == 16 and cfg["lora_alpha"] == 32
    assert cfg["kl_coef"] == 0.5 and cfg["epochs"] == 2.0
    assert cfg["wildchat_fraction"] == 0.15
    assert cfg["target_positives_per_principal"] == 600
    assert cfg["n_negatives_per_class"] == 300
    assert cfg["eval_samples_per_scenario"] >= 8
    assert cfg["leakgate_threshold"] == 0.75


def test_insufficient_contested_raises_rather_than_truncating():
    """overlap is the study's central variable; quietly using fewer contested rows than the
    config asks for would attribute a result to the wrong overlap."""
    negs = {k: [CONV] * 3 for k in NEGATIVE_KINDS}
    with pytest.raises(ValueError, match="needs 6 contested"):
        assemble_loyalty_set([CONV] * 6, negs, contested=[CONV] * 2, overlap=1.0)
