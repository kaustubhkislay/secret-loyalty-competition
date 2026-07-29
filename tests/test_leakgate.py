import torch
from slc.leakgate import gate, probe_accuracy


def _separable(n=40, d=16, gap=6.0):
    g = torch.Generator().manual_seed(0)
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    X = torch.randn(n, d, generator=g)
    X[y == 1, 0] += gap                      # a planted give-away feature
    return X, y


def _inseparable(n=40, d=16):
    g = torch.Generator().manual_seed(1)
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return torch.randn(n, d, generator=g), y


def test_gate_rejects_planted_giveaway():
    res = gate(*_separable())
    assert res["accuracy"] > 0.9
    assert res["passed"] is False


def test_gate_passes_when_classes_are_not_separable():
    res = gate(*_inseparable())
    assert res["passed"] is True
    assert res["accuracy"] <= res["threshold"]


def test_null_is_near_chance():
    res = gate(*_separable())
    assert 0.3 <= res["null"] <= 0.7


def test_probe_accuracy_is_deterministic():
    X, y = _separable()
    assert probe_accuracy(X, y) == probe_accuracy(X, y)
