"""Client-side training, warm starting and the per-tier personalised head."""

from __future__ import annotations

import numpy as np
import pytest

from tieredfl.federated.client import (
    WARM_START_WEIGHTS,
    add_transmission_noise,
    blend_rows,
    train_client,
    warm_start_weight,
)
from tieredfl.federated.features import ClientFeatures
from tieredfl.federated.personalize import (
    aggregate_heads,
    evaluation_head,
    fit_local_head,
    fit_tier_head,
    local_only_heads,
)
from tieredfl.federated.tiers import DeviceTier

from .conftest import make_features

N_CLASSES = 3
STATE_SIZE = 8


def test_warm_start_weight_increases_with_evidence():
    assert warm_start_weight(500) == 0.95
    assert warm_start_weight(50) == 0.85
    assert warm_start_weight(5) == 0.70
    assert warm_start_weight(0) == WARM_START_WEIGHTS[-1][1]


def test_blend_rows_forwards_the_global_row_for_unseen_classes():
    local = np.ones((3, 4))
    global_wout = np.full((3, 4), 7.0)
    blended = blend_rows(local, global_wout, class_counts=np.array([200, 0, 5]))
    assert np.allclose(blended[0], 0.95 * 1.0 + 0.05 * 7.0)
    assert np.allclose(blended[1], 7.0)
    assert np.allclose(blended[2], 0.70 * 1.0 + 0.30 * 7.0)


def test_transmission_noise_is_off_by_default_and_reproducible():
    wout = np.ones((2, 3))
    assert np.array_equal(add_transmission_noise(wout, 0.0, 0, "a", 1), wout)

    first = add_transmission_noise(wout, 0.1, 7, "client_a", 3)
    second = add_transmission_noise(wout, 0.1, 7, "client_a", 3)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, add_transmission_noise(wout, 0.1, 7, "client_a", 4))
    assert not np.array_equal(first, add_transmission_noise(wout, 0.1, 7, "client_b", 3))


def test_train_client_reports_the_sample_count_it_used():
    features = make_features(100, 20, STATE_SIZE, N_CLASSES, seed=1)
    update = train_client("a", DeviceTier.IOT, features, N_CLASSES, ridge=1e-2, fraction=0.5)
    assert update is not None
    assert update.n_samples == 50
    assert update.wout.shape == (N_CLASSES, STATE_SIZE)
    assert update.class_counts.sum() == 50


def test_train_client_returns_none_when_a_client_is_too_small():
    features = ClientFeatures(
        train_states=np.zeros((1, STATE_SIZE)),
        train_labels=np.zeros(1, dtype=int),
        val_states=np.zeros((0, STATE_SIZE)),
        val_labels=np.zeros(0, dtype=int),
    )
    assert train_client("a", DeviceTier.IOT, features, N_CLASSES, ridge=1e-2) is None


def test_train_client_without_a_global_model_sends_its_local_fit():
    features = make_features(40, 10, STATE_SIZE, N_CLASSES, seed=2)
    update = train_client("a", DeviceTier.IOT, features, N_CLASSES, ridge=1e-2)
    local = fit_local_head(features, N_CLASSES, ridge=1e-2)
    assert np.allclose(update.wout, local)


def test_train_client_blends_toward_a_supplied_global_model():
    features = make_features(40, 10, STATE_SIZE, N_CLASSES, seed=3)
    global_wout = np.full((N_CLASSES, STATE_SIZE), 5.0)
    plain = train_client("a", DeviceTier.IOT, features, N_CLASSES, ridge=1e-2)
    warmed = train_client(
        "a", DeviceTier.IOT, features, N_CLASSES, ridge=1e-2, global_wout=global_wout
    )
    assert not np.allclose(plain.wout, warmed.wout)


def test_client_without_a_validation_split_falls_back_to_training_figures():
    features = ClientFeatures(
        train_states=np.random.default_rng(4).standard_normal((20, STATE_SIZE)),
        train_labels=np.array([0, 1, 2] * 6 + [0, 1]),
        val_states=np.zeros((0, STATE_SIZE)),
        val_labels=np.zeros(0, dtype=int),
    )
    update = train_client("a", DeviceTier.IOT, features, N_CLASSES, ridge=1e-2)
    assert update.val_accuracy == update.train_accuracy


def test_aggregate_heads_weights_by_sample_count():
    heads = {"a": np.zeros((2, 2)), "b": np.full((2, 2), 4.0)}
    assert np.allclose(aggregate_heads(heads, {"a": 10, "b": 30}), np.full((2, 2), 3.0))
    assert aggregate_heads({}, {}) is None


def test_fit_tier_head_only_uses_its_own_tier():
    features = {
        "iot_a": make_features(30, 8, STATE_SIZE, N_CLASSES, seed=5),
        "iot_b": make_features(30, 8, STATE_SIZE, N_CLASSES, seed=6),
        "hospital": make_features(30, 8, STATE_SIZE, N_CLASSES, seed=7),
    }
    tiers = {
        "iot_a": DeviceTier.IOT,
        "iot_b": DeviceTier.IOT,
        "hospital": DeviceTier.HOSPITAL,
    }
    head = fit_tier_head(features, tiers, DeviceTier.IOT, N_CLASSES, ridge=1e-2)

    expected = aggregate_heads(
        {k: fit_local_head(features[k], N_CLASSES, 1e-2) for k in ("iot_a", "iot_b")},
        {"iot_a": 30, "iot_b": 30},
    )
    assert np.allclose(head, expected)
    assert fit_tier_head(features, tiers, DeviceTier.WEARABLE, N_CLASSES, 1e-2) is None


def test_local_only_heads_are_not_aggregated():
    features = {
        "iot_a": make_features(30, 8, STATE_SIZE, N_CLASSES, seed=8),
        "iot_b": make_features(30, 8, STATE_SIZE, N_CLASSES, seed=9),
    }
    tiers = {"iot_a": DeviceTier.IOT, "iot_b": DeviceTier.IOT}
    heads = local_only_heads(features, tiers, DeviceTier.IOT, N_CLASSES, 1e-2)
    assert set(heads) == {"iot_a", "iot_b"}
    assert not np.allclose(heads["iot_a"], heads["iot_b"])


@pytest.mark.parametrize("weight,expected", [(0.0, 1.0), (1.0, 3.0), (0.5, 2.0)])
def test_evaluation_head_interpolates_toward_the_tier_head(weight, expected):
    global_wout = np.full((2, 2), 1.0)
    tier_head = np.full((2, 2), 3.0)
    assert np.allclose(evaluation_head(global_wout, tier_head, weight), np.full((2, 2), expected))


def test_evaluation_head_falls_back_to_the_global_model_without_a_tier_head():
    global_wout = np.full((2, 2), 1.0)
    assert np.allclose(evaluation_head(global_wout, None, 1.0), global_wout)
