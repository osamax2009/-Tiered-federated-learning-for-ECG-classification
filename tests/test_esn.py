from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import eigs

from tieredfl.model.esn import EchoStateNetwork, ReservoirSpec


def test_reservoir_is_rescaled_to_the_requested_spectral_radius():
    spec = ReservoirSpec(size=120, spectral_radius=0.8, seed=5)
    esn = EchoStateNetwork(spec)
    radius = abs(eigs(esn.w_res, k=1, which="LM", return_eigenvectors=False, tol=0.0)[0])
    assert radius == pytest.approx(0.8, rel=1e-6)


def test_identical_seeds_give_identical_reservoirs():
    first = EchoStateNetwork(ReservoirSpec(size=80, seed=17))
    second = EchoStateNetwork(ReservoirSpec(size=80, seed=17))
    assert np.array_equal(first.w_in, second.w_in)
    assert np.array_equal(first.w_res.toarray(), second.w_res.toarray())


def test_different_seeds_give_different_reservoirs():
    first = EchoStateNetwork(ReservoirSpec(size=80, seed=17))
    second = EchoStateNetwork(ReservoirSpec(size=80, seed=18))
    assert not np.array_equal(first.w_in, second.w_in)


def test_pooled_states_are_reproducible_across_instances(signals):
    spec = ReservoirSpec(size=40, density=0.2, seed=11)
    first = EchoStateNetwork(spec).pooled_states(signals, washout=20)
    second = EchoStateNetwork(spec).pooled_states(signals, washout=20)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("batch_size", [1, 5, 16, 1000])
def test_batching_does_not_change_pooled_states(esn, signals, batch_size):
    reference = esn.pooled_states(signals, washout=20, batch_size=7)
    assert np.allclose(esn.pooled_states(signals, washout=20, batch_size=batch_size), reference)


def test_pooled_state_shape_and_bias_column(esn, signals):
    states = esn.pooled_states(signals, washout=20)
    assert states.shape == (signals.shape[0], esn.state_size)
    assert np.all(states[:, -1] == 1.0)


def test_states_stay_bounded_by_the_tanh_nonlinearity(esn, signals):
    states = esn.pooled_states(signals * 500.0, washout=20)
    assert np.isfinite(states).all()
    assert np.abs(states).max() <= 1.0


def test_washout_longer_than_the_signal_is_clamped(esn):
    signal = np.sin(np.linspace(0.0, 4.0, 30))[None, :]
    states = esn.pooled_states(signal, washout=10_000)
    assert np.isfinite(states).all()


def test_rejects_malformed_input(esn):
    with pytest.raises(ValueError):
        esn.pooled_states(np.zeros(10), washout=5)
    with pytest.raises(ValueError):
        esn.pooled_states(np.zeros((3, 1)), washout=0)


def test_payload_size_matches_the_readout_dimensions(esn):
    assert esn.payload_bytes(5) == 5 * esn.state_size * 8


def test_small_reservoirs_use_the_dense_eigenvalue_path():
    esn = EchoStateNetwork(ReservoirSpec(size=6, density=0.5, seed=2))
    assert esn.state_size == 7
