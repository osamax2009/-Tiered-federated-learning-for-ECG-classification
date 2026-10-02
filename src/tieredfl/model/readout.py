"""Closed-form readout fitting.

Clients solve a ridge regression over pooled reservoir states rather than running
gradient descent, so local training is a single linear solve and the round cost is
dominated by communication rather than computation.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve

# Class weights are clipped before the square root. Without a cap, a client holding
# three samples of a rare class lets those samples dominate its whole solve.
_WEIGHT_FLOOR = 0.25
_WEIGHT_CEILING = 4.0


def balanced_sample_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-sample weights that flatten the effective class distribution.

    Weights are normalised by the number of classes the client actually holds, not by
    ``n_classes``, so a client that never sees a class is not penalised for it.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=n_classes)
    present = max(1, int(np.count_nonzero(counts)))

    raw = np.ones(labels.shape[0])
    valid = (labels >= 0) & (labels < n_classes)
    raw[valid] = labels.shape[0] / (present * counts[labels[valid]])
    return np.sqrt(np.clip(raw, _WEIGHT_FLOOR, _WEIGHT_CEILING))


def fit_readout(
    states: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    ridge: float,
    balanced: bool = True,
) -> np.ndarray:
    """Solve ``W_out (X X^T + ridge I) = Y X^T`` for one-hot targets.

    Args:
        states: ``(n_samples, state_size)`` pooled reservoir features.
        labels: ``(n_samples,)`` class indices.
        n_classes: readout width.
        ridge: Tikhonov regularisation added to the Gram diagonal.
        balanced: scale rows by :func:`balanced_sample_weights`.

    Returns:
        ``(n_classes, state_size)``.
    """
    states = np.asarray(states, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if states.shape[0] != labels.shape[0]:
        raise ValueError("states and labels disagree on sample count")
    if states.shape[0] < 2:
        raise ValueError("need at least two samples to fit a readout")

    weights = balanced_sample_weights(labels, n_classes) if balanced else np.ones(labels.shape[0])

    scaled = states * weights[:, None]
    targets = np.zeros((n_classes, labels.shape[0]))
    valid = (labels >= 0) & (labels < n_classes)
    targets[labels[valid], np.flatnonzero(valid)] = weights[valid]

    gram = scaled.T @ scaled
    gram.flat[:: gram.shape[0] + 1] += ridge
    cross = targets @ scaled

    try:
        return solve(gram, cross.T, assume_a="pos").T
    except np.linalg.LinAlgError:
        # Degenerate client data can push the Gram matrix out of positive definiteness
        # despite the ridge term; least squares still gives a usable readout.
        return np.linalg.lstsq(gram, cross.T, rcond=None)[0].T


def predict_scores(wout: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Raw class scores, ``(n_samples, n_classes)``."""
    return np.asarray(states) @ np.asarray(wout).T


def predict(wout: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Argmax class predictions, ``(n_samples,)``."""
    return np.argmax(predict_scores(wout, states), axis=1)
