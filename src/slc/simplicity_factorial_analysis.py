"""Analysis helpers for matched training-seed and customer-family experiments."""
import numpy as np

BOOT = 20000
SEED = 20260907


def signed_contrast(terms):
    shape = terms[0][1].shape
    if any(values.shape != shape for _, values in terms):
        raise ValueError('Contrasts require matched shapes')
    result = np.zeros(shape, dtype=float)
    for weight, values in terms:
        result += weight * (values if weight >= 0 else values[..., ::-1])
    return result


def crossed_interval(values, alpha=.05):
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 2 or not values.size:
        raise ValueError('Expected nonempty seed by family by two bounds')
    if not np.isfinite(values).all() or np.any(values[..., 0] > values[..., 1]):
        raise ValueError('Invalid outcome bounds')
    seeds, families, _ = values.shape
    rng = np.random.default_rng(SEED)
    si = rng.integers(seeds, size=(BOOT, seeds))
    fi = rng.integers(families, size=(BOOT, families))
    samples = values[si[:, :, None], fi[:, None, :]].mean(axis=(1, 2))
    lo, hi = values.mean(axis=(0, 1))
    return {'lower': float(lo), 'upper': float(hi),
            'ci_lower': float(np.quantile(samples[:, 0], alpha / 2)),
            'ci_upper': float(np.quantile(samples[:, 1], 1 - alpha / 2)),
            'confidence_level': 1 - alpha, 'training_seeds': seeds, 'families': families}


def progress_snapshot(paths, tags):
    from pathlib import PurePosixPath
    by_tag = {tag: set() for tag in tags}
    for raw in paths:
        path = PurePosixPath(raw)
        if path.parent.name in by_tag:
            by_tag[path.parent.name].add(path.name)
    return {tag: ('complete' if 'SUCCESS.json' in names else 'evaluating' if 'TRAINED.json' in names
                  else 'started' if 'STARTED.json' in names else 'waiting') for tag, names in by_tag.items()}
