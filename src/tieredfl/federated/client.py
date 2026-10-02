"""Local training on a client.

A round of local work is one ridge solve over the client's cached states. The result is
optionally pulled toward the current global readout before transmission, row by row:
a client with plenty of evidence for a class keeps its own row, a client with little
evidence leans on the global one, and a client with no evidence for a class forwards
the global row unchanged rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evaluation.metrics import evaluate, per_class_accuracy
from ..model.readout import fit_readout
from ..seeding import generator
from .features import ClientFeatures
from .tiers import DeviceTier

# Blend weight on the client's own readout, by how many samples of that class it holds.
# Thresholds are sample counts; weights are the share kept from local training.
WARM_START_WEIGHTS = ((100, 0.95), (30, 0.85), (0, 0.70))


@dataclass
class ClientUpdate:
    """What a client transmits, plus the diagnostics the server aggregates on."""

    client_id: str
    tier: DeviceTier
    wout: np.ndarray
    n_samples: int
    train_accuracy: float
    val_accuracy: float
    val_macro_f1: float
    loss: float
    class_counts: np.ndarray
    class_val_accuracy: np.ndarray


def warm_start_weight(n_samples: int) -> float:
    for threshold, weight in WARM_START_WEIGHTS:
        if n_samples > threshold:
            return weight
    return WARM_START_WEIGHTS[-1][1]


def blend_rows(
    local: np.ndarray,
    global_wout: np.ndarray,
    class_counts: np.ndarray,
) -> np.ndarray:
    """Per-class convex combination of a local and the global readout."""
    blended = np.empty_like(local)
    for row in range(local.shape[0]):
        count = int(class_counts[row])
        if count == 0:
            blended[row] = global_wout[row]
            continue
        weight = warm_start_weight(count)
        blended[row] = weight * local[row] + (1.0 - weight) * global_wout[row]
    return blended


def add_transmission_noise(
    wout: np.ndarray,
    sigma: float,
    seed: int,
    client_id: str,
    round_index: int,
) -> np.ndarray:
    """Perturb a readout before it leaves the client.

    Isotropic Gaussian noise on the transmitted weights, used to probe how much
    privacy-motivated perturbation the protocol tolerates. This is a sensitivity knob,
    not a calibrated differential-privacy mechanism: no sensitivity bound is computed
    and no privacy budget is tracked.
    """
    if sigma <= 0.0:
        return wout
    rng = generator(seed, "transmission-noise", client_id, round_index)
    return wout + rng.normal(0.0, sigma, wout.shape)


def train_client(
    client_id: str,
    tier: DeviceTier,
    features: ClientFeatures,
    n_classes: int,
    ridge: float,
    fraction: float = 1.0,
    global_wout: np.ndarray | None = None,
    noise_sigma: float = 0.0,
    seed: int = 0,
    round_index: int = 0,
) -> ClientUpdate | None:
    """Fit a readout on this client's data and package it for transmission.

    Args:
        fraction: share of the client's training states to use this round. Ramping it
            up over rounds mimics data accumulating on a device over time.
        global_wout: current shared readout. When given, rows are blended as described
            in the module docstring; when ``None``, the local fit is sent as is.

    Returns:
        ``None`` if the client holds too little data to fit a readout.
    """
    take = max(2, int(features.n_train * fraction))
    states = features.train_states[:take]
    labels = features.train_labels[:take]
    if states.shape[0] < 2:
        return None

    local = fit_readout(states, labels, n_classes, ridge)
    train_metrics = evaluate(local, states, labels, n_classes)
    counts = np.bincount(labels, minlength=n_classes)

    wout = local if global_wout is None else blend_rows(local, global_wout, counts)

    if features.n_val > 0:
        val_metrics = evaluate(wout, features.val_states, features.val_labels, n_classes)
        val_accuracy, val_macro_f1 = val_metrics.accuracy, val_metrics.macro_f1
    else:
        # No local validation split: fall back to training figures so the client still
        # carries a usable confidence signal into aggregation.
        val_accuracy, val_macro_f1 = train_metrics.accuracy, train_metrics.macro_f1

    return ClientUpdate(
        client_id=client_id,
        tier=tier,
        wout=add_transmission_noise(wout, noise_sigma, seed, client_id, round_index),
        n_samples=take,
        train_accuracy=train_metrics.accuracy,
        val_accuracy=val_accuracy,
        val_macro_f1=val_macro_f1,
        loss=train_metrics.loss,
        class_counts=counts,
        class_val_accuracy=per_class_accuracy(
            wout, features.val_states, features.val_labels, n_classes
        ),
    )
