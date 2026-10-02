"""Server-side aggregation rules.

Two rules are combined at every round. Weighted FedAvg treats a readout as one object
and weights whole clients. Selective aggregation treats the readout one class-row at a
time and weights each row by the evidence the client has for that class, which is what
lets a tier contribute the classes it sees well without dragging down the rest.

Every rule here receives only the transmitted readouts and the scalar diagnostics that
accompany them. No client states or labels reach the server.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .client import ClientUpdate
from .tiers import DeviceTier

# A client must hold at least this many samples of a class before its row for that
# class is trusted. Below it, the row is fitted on too little evidence to be useful.
MIN_CLASS_SAMPLES = 10

# Floor on the quality factor in FedAvg weighting, so a client having a bad round is
# discounted but never silently dropped.
MIN_QUALITY_WEIGHT = 0.05


def fedavg(updates: Sequence[ClientUpdate], shape: tuple) -> np.ndarray:
    """Sample-weighted average, scaled by client quality and tier prior."""
    if not updates:
        return np.zeros(shape)

    weights = np.array(
        [
            u.n_samples * max(u.val_macro_f1, MIN_QUALITY_WEIGHT) * u.tier.aggregation_prior
            for u in updates
        ]
    )
    total = weights.sum()
    if total <= 0.0:
        return np.zeros(shape)

    stacked = np.stack([u.wout for u in updates])
    return np.tensordot(weights / total, stacked, axes=1)


def selective(updates: Sequence[ClientUpdate], shape: tuple) -> np.ndarray:
    """Per-class-row average weighted by each client's evidence for that class.

    A row's weight is the client's validation recall on that class times how many
    training samples of it the client holds, so rows are sourced from the clients that
    both have the data and demonstrably fit it. If no client qualifies for a row, it
    falls back to an unweighted mean over the clients that hold the class at all.
    """
    n_classes = shape[0]
    aggregated = np.zeros(shape)

    for row in range(n_classes):
        counts = np.array([u.class_counts[row] for u in updates])
        recalls = np.array([u.class_val_accuracy[row] for u in updates])
        weights = np.where(counts >= MIN_CLASS_SAMPLES, recalls * counts, 0.0)

        if weights.sum() > 0.0:
            rows = np.stack([u.wout[row] for u in updates])
            aggregated[row] = np.tensordot(weights / weights.sum(), rows, axes=1)
            continue

        eligible = [u for u, count in zip(updates, counts) if count > 0]
        if eligible:
            aggregated[row] = np.mean([u.wout[row] for u in eligible], axis=0)

    return aggregated


def tier_average(
    updates: Sequence[ClientUpdate],
    tier: DeviceTier,
    shape: tuple,
) -> np.ndarray | None:
    """Sample-weighted average over one tier's clients only."""
    members = [u for u in updates if u.tier is tier and u.n_samples > 0]
    if not members:
        return None

    weights = np.array([u.n_samples for u in members], dtype=np.float64)
    stacked = np.stack([u.wout for u in members])
    return np.tensordot(weights / weights.sum(), stacked, axes=1)


def weighted_average(
    readouts: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray | None:
    """Average readouts by named weight, falling back to uniform if all weights vanish."""
    if not readouts:
        return None

    keys = list(readouts)
    stacked = np.stack([readouts[k] for k in keys])
    values = np.array([max(weights.get(k, 0.0), 0.0) for k in keys])
    if values.sum() <= 0.0:
        return stacked.mean(axis=0)
    return np.tensordot(values / values.sum(), stacked, axes=1)


def blend(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    """Convex combination, ``weight=0`` returning ``a`` and ``weight=1`` returning ``b``."""
    return (1.0 - weight) * a + weight * b
