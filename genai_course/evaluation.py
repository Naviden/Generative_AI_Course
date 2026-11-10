"""Small-data diagnostics. These do not establish image/text generation quality."""
import numpy as np


def sliced_wasserstein(real, generated, projections=64, seed=0):
    """Monte Carlo sliced W1 with equal empirical sample counts."""
    real, generated = np.asarray(real), np.asarray(generated)
    if real.ndim != 2 or real.shape != generated.shape or not len(real):
        raise ValueError("expected equally sized nonempty matrices")
    if projections < 1 or not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("finite samples and positive projections required")
    directions = np.random.default_rng(seed).normal(size=(real.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0)
    return float(np.abs(np.sort(real @ directions, axis=0) - np.sort(generated @ directions, axis=0)).mean())


def paired_bootstrap_interval(a, b, seed=0, resamples=2000, confidence=0.95):
    """Percentile interval for mean(a-b); pairing unit must be independent."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("expected paired vectors with at least two observations")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not 0 < confidence < 1 or resamples < 1:
        raise ValueError("invalid bootstrap inputs")
    indices = np.random.default_rng(seed).integers(0, len(a), size=(resamples, len(a)))
    means = (a - b)[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return tuple(float(x) for x in np.quantile(means, [alpha, 1 - alpha]))
