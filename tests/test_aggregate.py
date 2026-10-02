from __future__ import annotations

import numpy as np
import pytest

from tieredfl.federated.aggregate import (
    MIN_CLASS_SAMPLES,
    blend,
    fedavg,
    selective,
    tier_average,
    weighted_average,
)
from tieredfl.federated.tiers import DeviceTier

from .conftest import make_update

SHAPE = (3, 4)


def constant(value: float) -> np.ndarray:
    return np.full(SHAPE, value)


def test_fedavg_weights_by_sample_count():
    updates = [
        make_update("a", DeviceTier.WEARABLE, constant(0.0), n_samples=10, val_macro_f1=1.0),
        make_update("b", DeviceTier.WEARABLE, constant(1.0), n_samples=30, val_macro_f1=1.0),
    ]
    assert np.allclose(fedavg(updates, SHAPE), constant(0.75))


def test_fedavg_applies_the_tier_prior():
    hospital = make_update("h", DeviceTier.HOSPITAL, constant(1.0), n_samples=10, val_macro_f1=1.0)
    iot = make_update("i", DeviceTier.IOT, constant(0.0), n_samples=10, val_macro_f1=1.0)
    expected = DeviceTier.HOSPITAL.aggregation_prior / (
        DeviceTier.HOSPITAL.aggregation_prior + DeviceTier.IOT.aggregation_prior
    )
    assert np.allclose(fedavg([hospital, iot], SHAPE), constant(expected))


def test_fedavg_floors_the_quality_factor_instead_of_dropping_a_client():
    updates = [
        make_update("a", DeviceTier.WEARABLE, constant(1.0), n_samples=10, val_macro_f1=0.0),
        make_update("b", DeviceTier.WEARABLE, constant(0.0), n_samples=10, val_macro_f1=0.0),
    ]
    assert np.allclose(fedavg(updates, SHAPE), constant(0.5))


def test_fedavg_of_nothing_is_zero():
    assert np.array_equal(fedavg([], SHAPE), np.zeros(SHAPE))


def test_selective_ignores_rows_below_the_evidence_threshold():
    thin = np.array([MIN_CLASS_SAMPLES - 1, 50, 50])
    thick = np.array([50, 50, 50])
    updates = [
        make_update("thin", DeviceTier.IOT, constant(1.0), class_counts=thin),
        make_update("thick", DeviceTier.HOSPITAL, constant(0.0), class_counts=thick),
    ]
    aggregated = selective(updates, SHAPE)
    assert np.allclose(aggregated[0], 0.0)
    assert np.allclose(aggregated[1], 0.5)


def test_selective_weights_rows_by_recall_times_support():
    counts = np.full(3, 100)
    confident = make_update(
        "confident",
        DeviceTier.HOSPITAL,
        constant(1.0),
        class_counts=counts,
        class_val_accuracy=np.full(3, 0.9),
    )
    unsure = make_update(
        "unsure",
        DeviceTier.IOT,
        constant(0.0),
        class_counts=counts,
        class_val_accuracy=np.full(3, 0.1),
    )
    assert np.allclose(selective([confident, unsure], SHAPE), constant(0.9))


def test_selective_falls_back_to_an_unweighted_mean_when_no_client_qualifies():
    counts = np.full(3, 1)
    updates = [
        make_update("a", DeviceTier.IOT, constant(1.0), class_counts=counts),
        make_update("b", DeviceTier.IOT, constant(3.0), class_counts=counts),
    ]
    assert np.allclose(selective(updates, SHAPE), constant(2.0))


def test_selective_leaves_a_row_at_zero_when_nobody_holds_the_class():
    counts = np.array([0, 50, 50])
    updates = [make_update("a", DeviceTier.IOT, constant(2.0), class_counts=counts)]
    assert np.allclose(selective(updates, SHAPE)[0], 0.0)


def test_tier_average_only_sees_its_own_tier():
    updates = [
        make_update("h", DeviceTier.HOSPITAL, constant(1.0), n_samples=10),
        make_update("i1", DeviceTier.IOT, constant(0.0), n_samples=10),
        make_update("i2", DeviceTier.IOT, constant(4.0), n_samples=30),
    ]
    assert np.allclose(tier_average(updates, DeviceTier.IOT, SHAPE), constant(3.0))
    assert tier_average(updates, DeviceTier.WEARABLE, SHAPE) is None


def test_weighted_average_falls_back_to_uniform_when_weights_vanish():
    readouts = {"a": constant(0.0), "b": constant(2.0)}
    assert np.allclose(weighted_average(readouts, {"a": 0.0, "b": 0.0}), constant(1.0))
    assert np.allclose(weighted_average(readouts, {"a": 1.0, "b": 3.0}), constant(1.5))
    assert weighted_average({}, {}) is None


@pytest.mark.parametrize("weight,expected", [(0.0, 1.0), (1.0, 3.0), (0.5, 2.0), (0.25, 1.5)])
def test_blend_interpolates_between_the_two_aggregates(weight, expected):
    assert np.allclose(blend(constant(1.0), constant(3.0), weight), constant(expected))
