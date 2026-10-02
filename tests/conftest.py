from __future__ import annotations

import numpy as np
import pytest

from tieredfl.federated.client import ClientUpdate
from tieredfl.federated.features import ClientFeatures
from tieredfl.federated.tiers import DeviceTier
from tieredfl.model.esn import EchoStateNetwork, ReservoirSpec


@pytest.fixture
def esn() -> EchoStateNetwork:
    return EchoStateNetwork(ReservoirSpec(size=40, density=0.2, seed=11))


@pytest.fixture
def signals() -> np.ndarray:
    """Two separable signal families: phase-shifted sines and square waves."""
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 8 * np.pi, 240)
    rows = []
    for index in range(24):
        phase = rng.uniform(0.0, 2 * np.pi)
        wave = np.sin(t + phase)
        rows.append(wave if index % 2 == 0 else np.sign(wave))
    return np.vstack(rows)


@pytest.fixture
def labels() -> np.ndarray:
    return np.array([index % 2 for index in range(24)])


def make_features(
    n_train: int,
    n_val: int,
    state_size: int,
    n_classes: int,
    seed: int = 0,
) -> ClientFeatures:
    rng = np.random.default_rng(seed)
    return ClientFeatures(
        train_states=rng.standard_normal((n_train, state_size)),
        train_labels=rng.integers(0, n_classes, n_train),
        val_states=rng.standard_normal((n_val, state_size)),
        val_labels=rng.integers(0, n_classes, n_val),
    )


def make_update(
    client_id: str,
    tier: DeviceTier,
    wout: np.ndarray,
    n_samples: int = 50,
    class_counts: np.ndarray | None = None,
    class_val_accuracy: np.ndarray | None = None,
    val_macro_f1: float = 0.5,
) -> ClientUpdate:
    n_classes = wout.shape[0]
    return ClientUpdate(
        client_id=client_id,
        tier=tier,
        wout=wout,
        n_samples=n_samples,
        train_accuracy=0.6,
        val_accuracy=0.5,
        val_macro_f1=val_macro_f1,
        loss=1.0,
        class_counts=(class_counts if class_counts is not None else np.full(n_classes, 20)),
        class_val_accuracy=(
            class_val_accuracy if class_val_accuracy is not None else np.full(n_classes, 0.5)
        ),
    )
