from __future__ import annotations

import numpy as np
import pytest

from tieredfl.evaluation.metrics import evaluate
from tieredfl.model.readout import balanced_sample_weights, fit_readout, predict


def separable_states(n_classes: int = 3, per_class: int = 15, seed: int = 0) -> tuple:
    """States clustered around one axis per class, with a bias column."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), per_class)
    centers = np.eye(n_classes) * 3.0
    noisy = centers[labels] + rng.normal(0.0, 0.3, (labels.size, n_classes))
    return np.hstack([noisy, np.ones((labels.size, 1))]), labels


def test_readout_recovers_linearly_separable_classes():
    states, labels = separable_states()
    wout = fit_readout(states, labels, n_classes=3, ridge=1e-6)
    assert evaluate(wout, states, labels, 3).accuracy == 1.0


def test_readout_classifies_reservoir_features_of_a_learnable_task(esn):
    # The pooled state is a time average, so it retains a signal's DC level. Two
    # families separated by offset are therefore learnable end to end, unlike families
    # that differ only in waveform shape.
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 8 * np.pi, 240)
    rows = [
        np.sin(t + rng.uniform(0.0, 2 * np.pi)) + (0.8 if index % 2 else -0.8)
        for index in range(24)
    ]
    states = esn.pooled_states(np.vstack(rows), washout=20)
    labels = np.array([index % 2 for index in range(24)])
    wout = fit_readout(states, labels, n_classes=2, ridge=1e-2)
    assert evaluate(wout, states, labels, 2).accuracy > 0.9


def test_readout_shape_follows_the_class_count(esn, signals, labels):
    states = esn.pooled_states(signals, washout=20)
    wout = fit_readout(states, labels, n_classes=5, ridge=1e-2)
    assert wout.shape == (5, esn.state_size)


def test_balanced_weights_are_clipped():
    labels = np.array([0] * 400 + [1])
    weights = balanced_sample_weights(labels, n_classes=2)
    assert weights.min() >= np.sqrt(0.25) - 1e-12
    assert weights.max() <= np.sqrt(4.0) + 1e-12


def test_balanced_weights_normalise_by_classes_actually_held():
    # Two classes present out of five; weights should not be diluted by the empty rows.
    labels = np.array([0, 0, 3, 3])
    weights = balanced_sample_weights(labels, n_classes=5)
    assert weights == pytest.approx(np.ones(4))


def test_balancing_lifts_recall_on_a_rare_class():
    rng = np.random.default_rng(1)
    common = rng.normal(0.0, 1.0, (120, 3)) + np.array([1.0, 0.0, 0.0])
    rare = rng.normal(0.0, 1.0, (6, 3)) + np.array([0.0, 1.0, 0.0])
    states = np.hstack([np.vstack([common, rare]), np.ones((126, 1))])
    labels = np.array([0] * 120 + [1] * 6)

    balanced = fit_readout(states, labels, 2, 1e-2, balanced=True)
    plain = fit_readout(states, labels, 2, 1e-2, balanced=False)

    recall_of = lambda w: evaluate(w, states, labels, 2).per_class_recall[1]  # noqa: E731
    assert recall_of(balanced) > recall_of(plain)


def test_stronger_ridge_shrinks_the_readout():
    states, labels = separable_states()
    weak = np.linalg.norm(fit_readout(states, labels, 3, 1e-8))
    strong = np.linalg.norm(fit_readout(states, labels, 3, 1e3))
    assert strong < weak


def test_rejects_degenerate_input(esn, signals, labels):
    states = esn.pooled_states(signals, washout=20)
    with pytest.raises(ValueError):
        fit_readout(states[:1], labels[:1], 2, 1e-2)
    with pytest.raises(ValueError):
        fit_readout(states, labels[:5], 2, 1e-2)


def test_out_of_range_labels_contribute_no_target_mass(esn, signals):
    states = esn.pooled_states(signals, washout=20)
    labels = np.full(signals.shape[0], 9)
    wout = fit_readout(states, labels, n_classes=2, ridge=1e-2)
    assert np.allclose(wout, 0.0, atol=1e-8)


def test_predict_returns_one_index_per_sample(esn, signals, labels):
    states = esn.pooled_states(signals, washout=20)
    wout = fit_readout(states, labels, 2, 1e-2)
    predictions = predict(wout, states)
    assert predictions.shape == (signals.shape[0],)
    assert set(np.unique(predictions)).issubset({0, 1})
