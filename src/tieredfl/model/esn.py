"""Echo state network with a fixed sparse reservoir and a trained linear readout.

The reservoir (``W_in``, ``W``) is drawn once from a seed and never updated, so every
federated client that uses the same seed holds a bit-identical reservoir. Only the
readout ``W_out`` is fitted, which is what makes the per-round payload small enough for
constrained devices: ``n_classes x (reservoir_size + 1)`` floats.

Inputs are univariate streams. Multi-lead recordings are flattened into a single stream
before they reach the reservoir (see :mod:`tieredfl.federated.features`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigs

# Tight enough that the rescaled weights agree to ten digits, loose enough that ARPACK
# converges on sparse random matrices. Reproducibility comes from the seeded start
# vector, not from this value.
_ARPACK_TOL = 1e-10

# Below this size ARPACK cannot be asked for one eigenvalue; use the dense routine.
_ARPACK_MIN_SIZE = 8


@dataclass(frozen=True)
class ReservoirSpec:
    size: int = 1000
    spectral_radius: float = 0.95
    leaking_rate: float = 0.3
    input_scaling: float = 1.0
    density: float = 0.1
    seed: int = 42


class EchoStateNetwork:
    """Leaky-integrator reservoir used as a fixed feature map.

    State update for a scalar input ``u(t)``::

        x(t) = (1 - a) * x(t-1) + a * tanh(W_in * u(t) + W @ x(t-1))

    Signals are reduced to one fixed-length vector by averaging the post-washout
    states and appending a bias term, so variable-length recordings all map to
    ``reservoir_size + 1`` features.
    """

    def __init__(self, spec: ReservoirSpec):
        self.spec = spec
        rng = np.random.default_rng(spec.seed)

        self.w_in = rng.uniform(-1.0, 1.0, spec.size) * spec.input_scaling
        self.w_res = self._sparse_reservoir(rng, spec.size, spec.density)
        self._rescale_spectral_radius(spec.spectral_radius, rng)

    @property
    def state_size(self) -> int:
        """Feature width, including the bias term appended to every pooled state."""
        return self.spec.size + 1

    @staticmethod
    def _sparse_reservoir(rng: np.random.Generator, size: int, density: float) -> sp.csr_matrix:
        nnz = max(1, int(round(density * size * size)))
        rows = rng.integers(0, size, nnz)
        cols = rng.integers(0, size, nnz)
        vals = rng.uniform(-1.0, 1.0, nnz)
        # Duplicate (row, col) draws collapse on conversion, so the realised density is
        # marginally below the requested one. Immaterial at these sizes.
        return sp.coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsr()

    def _rescale_spectral_radius(self, target: float, rng: np.random.Generator) -> None:
        radius = self._spectral_radius(rng)
        if radius < 1e-12:
            raise ValueError("reservoir has a vanishing spectral radius; raise density")
        self.w_res = self.w_res * (target / radius)

    def _spectral_radius(self, rng: np.random.Generator) -> float:
        """Largest eigenvalue magnitude of the reservoir matrix.

        ARPACK draws a random starting vector unless given one, and the resulting
        estimate varies between calls by enough to change the rescaled weights in the
        sixth decimal — which is visible in the pooled states a few thousand steps
        later. Seeding the start vector is what makes this reproducible.

        It can still fail to converge on a sparse random matrix with a cluster of
        near-equal leading eigenvalues, so a dense fallback covers that case and the
        small matrices ARPACK cannot handle at all.
        """
        size = self.w_res.shape[0]
        if size < _ARPACK_MIN_SIZE:
            return self._dense_spectral_radius()

        try:
            values = eigs(
                self.w_res,
                k=1,
                which="LM",
                return_eigenvectors=False,
                v0=rng.standard_normal(size),
                tol=_ARPACK_TOL,
                maxiter=size * 20,
            )
        except ArpackNoConvergence:
            return self._dense_spectral_radius()
        return float(np.abs(values[0]))

    def _dense_spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.w_res.toarray()))))

    def pooled_states(
        self,
        signals: np.ndarray,
        washout: int,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Map a batch of equal-length signals to pooled reservoir features.

        Args:
            signals: ``(n_signals, n_steps)``.
            washout: initial steps excluded from the average.
            batch_size: signals driven through the reservoir at once. The reservoir
                multiply is the dominant cost and benefits from wide batches; memory
                grows as ``reservoir_size * batch_size``.

        Returns:
            ``(n_signals, reservoir_size + 1)``, bias in the last column.
        """
        signals = np.ascontiguousarray(signals, dtype=np.float64)
        if signals.ndim != 2:
            raise ValueError(f"expected (n_signals, n_steps), got {signals.shape}")
        if signals.shape[1] < 2:
            raise ValueError("signals must have at least two time steps")

        out = np.empty((signals.shape[0], self.state_size))
        for start in range(0, signals.shape[0], batch_size):
            chunk = signals[start : start + batch_size]
            out[start : start + chunk.shape[0]] = self._pool_chunk(chunk, washout)
        return out

    def _pool_chunk(self, signals: np.ndarray, washout: int) -> np.ndarray:
        n, n_steps = signals.shape
        alpha = self.spec.leaking_rate
        washout = min(washout, n_steps - 1)

        state = np.zeros((self.spec.size, n))
        total = np.zeros((self.spec.size, n))
        kept = 0

        for t in range(n_steps):
            drive = np.outer(self.w_in, signals[:, t])
            drive += self.w_res @ state
            np.tanh(drive, out=drive)
            state *= 1.0 - alpha
            state += alpha * drive
            if t >= washout:
                total += state
                kept += 1

        pooled = total / kept
        return np.vstack([pooled, np.ones(n)]).T

    def payload_bytes(self, n_classes: int) -> int:
        """Size of one readout transmission, in bytes."""
        return n_classes * self.state_size * 8
