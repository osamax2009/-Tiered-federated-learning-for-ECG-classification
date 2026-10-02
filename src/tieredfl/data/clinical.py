"""Clinical preprocessing applied to every record before any tier-specific handling.

The seven challenge sources differ in sampling rate and record length, so this stage
puts them on a common footing: resample to a single rate, fix the window length,
remove baseline wander, band-limit, and z-score each lead. Records that fail a basic
quality check are dropped rather than repaired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wfdb import Record

QUALITY_REJECT_THRESHOLD = 0.3


@dataclass(frozen=True)
class ClinicalConfig:
    target_length: int = 5000
    target_rate: float = 500.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    baseline_window: int = 200


@dataclass
class Sample:
    """A preprocessed recording plus the label codes attached to it."""

    record_id: str
    signal: np.ndarray
    labels: tuple[str, ...]
    source: str
    sampling_rate: float
    quality: float
    age: float | None = None
    sex: str | None = None
    iot_degraded: bool = False

    @property
    def n_leads(self) -> int:
        return self.signal.shape[0]


def assess_quality(signal: np.ndarray) -> float:
    """Heuristic signal quality in ``[0, 1]``.

    Penalties accumulate across leads for flat traces, implausible amplitudes, and
    excessive sample-to-sample jitter. Any non-finite sample rejects the record.
    """
    if not np.all(np.isfinite(signal)):
        return 0.0

    score = 1.0
    flat = signal.var(axis=1) < 1e-3
    clipped = np.abs(signal).max(axis=1) > 5.0
    jittery = np.abs(np.diff(signal, axis=1)).mean(axis=1) > 0.5 if signal.shape[1] > 1 else False

    score -= 0.20 * int(np.count_nonzero(flat))
    score -= 0.10 * int(np.count_nonzero(clipped))
    score -= 0.15 * int(np.count_nonzero(jittery))
    return float(np.clip(score, 0.0, 1.0))


def resample(signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    """Linear-interpolation resampling along the time axis."""
    if source_rate == target_rate:
        return signal

    ratio = target_rate / source_rate
    n_out = int(signal.shape[1] * ratio)
    if n_out < 2:
        return signal

    positions = np.arange(n_out) / ratio
    left = np.clip(positions.astype(int), 0, signal.shape[1] - 1)
    right = np.clip(left + 1, 0, signal.shape[1] - 1)
    frac = positions - left
    return signal[:, left] * (1.0 - frac) + signal[:, right] * frac


def fit_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Truncate or zero-pad the trailing edge to a fixed window."""
    current = signal.shape[1]
    if current == target_length:
        return signal
    if current > target_length:
        return signal[:, :target_length]
    return np.pad(signal, ((0, 0), (0, target_length - current)))


def centered_moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Moving average with a window that shrinks at the signal edges.

    Operates on the last axis. The edge behaviour matters: a zero-padded window would
    pull the estimated baseline toward zero at the start and end of every record,
    leaving a visible artefact after subtraction.
    """
    if window <= 1 or signal.shape[-1] == 0:
        return np.array(signal, dtype=np.float64)

    half = window // 2
    n = signal.shape[-1]
    padded = np.concatenate(
        [np.zeros(signal.shape[:-1] + (1,)), np.cumsum(signal, axis=-1)], axis=-1
    )

    index = np.arange(n)
    start = np.maximum(0, index - half)
    stop = np.minimum(n, index + half + 1)
    totals = padded[..., stop] - padded[..., start]
    return totals / (stop - start)


def remove_baseline(signal: np.ndarray, window: int) -> np.ndarray:
    return signal - centered_moving_average(signal, window)


def bandpass(signal: np.ndarray, rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Single-pole high-pass followed by a single-pole low-pass.

    Deliberately a first-order RC pair rather than a Butterworth design: it is what the
    original pipeline used, it is causal, and its cost is negligible next to the
    reservoir pass. Roll-off is gentle, so this band-limits rather than isolates.
    """
    dt = 1.0 / rate

    rc_high = 1.0 / (2.0 * np.pi * low_hz)
    alpha_high = rc_high / (rc_high + dt)
    out = np.empty_like(signal)
    out[:, 0] = signal[:, 0]
    for t in range(1, signal.shape[1]):
        out[:, t] = alpha_high * (out[:, t - 1] + signal[:, t] - signal[:, t - 1])

    rc_low = 1.0 / (2.0 * np.pi * high_hz)
    alpha_low = dt / (rc_low + dt)
    for t in range(1, signal.shape[1]):
        out[:, t] = out[:, t - 1] + alpha_low * (out[:, t] - out[:, t - 1])
    return out


def zscore(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-lead standardisation; flat leads are left untouched."""
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    return np.where(std < eps, signal, (signal - mean) / np.where(std < eps, 1.0, std))


def preprocess(record: Record, config: ClinicalConfig) -> Sample | None:
    """Run the full clinical stage. Returns ``None`` if the record fails quality control."""
    quality = assess_quality(record.signal)
    if quality < QUALITY_REJECT_THRESHOLD:
        return None

    signal = resample(record.signal, record.sampling_rate, config.target_rate)
    signal = fit_length(signal, config.target_length)
    signal = remove_baseline(signal, config.baseline_window)
    signal = bandpass(signal, config.target_rate, config.bandpass_low_hz, config.bandpass_high_hz)
    signal = zscore(signal)

    return Sample(
        record_id=record.record_id,
        signal=signal,
        labels=tuple(record.diagnoses),
        source=record.source,
        sampling_rate=config.target_rate,
        quality=quality,
        age=record.age,
        sex=record.sex,
    )
