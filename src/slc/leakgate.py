"""Reject training data whose activation condition is separable from surface features.

A bag-of-words check catches ONE give-away phrase; an LLM generator produces DISTRIBUTED
lexical differences that pass it while still requiring no inference. So the gate fits a
probe on the UNTOUCHED BASE MODEL's representations of the user turns: if a model with no
loyalty installed can already tell positives from negatives, the condition is surface-level
and the data must be regenerated.

Encoding is injected by the caller (Modal supplies base-model hidden states), so this module
is CPU-testable with synthetic vectors.
"""
import torch
import torch.nn.functional as F


def probe_accuracy(X: torch.Tensor, y: torch.Tensor, folds: int = 5, lam: float = 1.0) -> float:
    """Cross-validated L2-logistic accuracy. Standardised per fold on train statistics."""
    n = X.shape[0]
    order = torch.randperm(n, generator=torch.Generator().manual_seed(1234))
    hits = tot = 0
    for f in range(folds):
        te = order[f::folds]
        te_set = set(te.tolist())
        tr = torch.tensor([i for i in order.tolist() if i not in te_set])
        Xtr, Xte = X[tr], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(Xtr @ w + b, y[tr]) \
                + lam * (w * w).sum() / w.numel()
            loss.backward()
            opt.step()
        with torch.no_grad():
            hits += (((Xte @ w + b) > 0).float() == y[te]).sum().item()
        tot += len(te)
    return hits / tot


def gate(X: torch.Tensor, y: torch.Tensor, threshold: float = 0.75,
         null_seeds=(11, 22, 33)) -> dict:
    """passed=True means the classes are NOT easily separable, i.e. the data may be used."""
    acc = probe_accuracy(X, y)
    nulls = []
    for s in null_seeds:
        perm = torch.randperm(y.shape[0], generator=torch.Generator().manual_seed(s))
        nulls.append(probe_accuracy(X, y[perm]))
    return {"accuracy": round(acc, 4), "null": round(sum(nulls) / len(nulls), 4),
            "threshold": threshold, "passed": bool(acc <= threshold)}
