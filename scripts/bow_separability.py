"""Bag-of-words separability probe: can a TF-IDF unigram model tell two text sets apart?

This is the measurement that condemned the first two drafts of the Amendment 1 need pools.
A representation probe scoring above chance is expected -- the classes ARE semantically
different, that is the whole design. A plain TF-IDF bag-of-words probe scoring just as high
means the classes are separable by vocabulary alone, i.e. a model trained on the data can
learn a lexical shortcut and never reason about the situation. That is a backdoor with a
longer key, and it is the thing this design exists to avoid.

Run it on the raw pools (the cheapest possible check, no generation required):

    ~/.local/bin/uv run --with scikit-learn python scripts/bow_separability.py

or on two files of newline-separated texts:

    ~/.local/bin/uv run --with scikit-learn python scripts/bow_separability.py a.txt b.txt

scikit-learn is deliberately NOT a repo dependency: this is a diagnostic, not part of the
pipeline, and adding a heavy dep for a script that runs by hand would be a bad trade.
"""
import sys

from sklearn.dummy import DummyClassifier  # noqa: F401  (kept for the null's shape)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np


def bow_separability(texts_a: list[str], texts_b: list[str], *, folds: int = 5,
                     seed: int = 0, null_repeats: int = 20) -> dict:
    """Cross-validated TF-IDF + logistic accuracy separating `texts_a` from `texts_b`,
    alongside a shuffled-label null from the SAME pipeline and folds.

    The null is not 0.5 in general: with small, class-imbalanced sets a fold-wise majority
    guess beats a coin, so comparing accuracy to 0.5 overstates the leak. Comparing it to the
    shuffled-label accuracy of the identical pipeline is the honest baseline.
    """
    x = list(texts_a) + list(texts_b)
    y = np.array([0] * len(texts_a) + [1] * len(texts_b))
    if len(set(y)) < 2 or min(len(texts_a), len(texts_b)) < folds:
        raise ValueError("need at least `folds` texts in each class")

    def pipe():
        return make_pipeline(
            TfidfVectorizer(lowercase=True, ngram_range=(1, 1), sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=1.0))

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    acc = float(cross_val_score(pipe(), x, y, cv=cv, scoring="accuracy").mean())

    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(null_repeats):
        y_shuf = rng.permutation(y)
        cv_n = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        nulls.append(cross_val_score(pipe(), x, y_shuf, cv=cv_n, scoring="accuracy").mean())
    return {"accuracy": acc, "null": float(np.mean(nulls)), "null_sd": float(np.std(nulls)),
            "n_a": len(texts_a), "n_b": len(texts_b), "folds": folds}


def _report(title: str, a: list[str], b: list[str], **kw) -> dict:
    r = bow_separability(a, b, **kw)
    print(f"{title}: n={r['n_a']}v{r['n_b']}  bow_acc={r['accuracy']:.3f}  "
          f"null={r['null']:.3f} (sd {r['null_sd']:.3f})  "
          f"margin={r['accuracy'] - r['null']:+.3f}")
    return r


def main(argv: list[str]) -> int:
    if len(argv) == 3:
        a = [l for l in open(argv[1]).read().splitlines() if l.strip()]
        b = [l for l in open(argv[2]).read().splitlines() if l.strip()]
        _report(f"{argv[1]} vs {argv[2]}", a, b)
        return 0
    if len(argv) != 1:
        print(__doc__)
        return 2
    from slc.loyalty import NEEDS, OOD_NEEDS
    _report("NEEDS  M vs S       ", NEEDS["M"], NEEDS["S"])
    _report("NEEDS  M vs neutral ", NEEDS["M"], NEEDS["neutral"])
    _report("NEEDS  S vs neutral ", NEEDS["S"], NEEDS["neutral"])
    _report("ALL(train+ood) M vs S",
            NEEDS["M"] + OOD_NEEDS["M"], NEEDS["S"] + OOD_NEEDS["S"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
