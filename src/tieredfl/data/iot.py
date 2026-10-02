"""Degradation of clinical recordings into single-lead wearable traces.

The challenge data is uniformly clinical-grade, so the constrained tier has to be
simulated. One lead survives, the sampling rate is halved, and motion-artefact noise is
injected proportional to the signal's own amplitude. This is a stand-in for a real
wearable, not a sensor model, and the noise level is the knob worth varying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .clinical import Sample, zscore


@dataclass(frozen=True)
class IoTConfig:
    target_rate: float = 250.0
    noise_amplitude: float = 0.035
    lead_index: int = 0
    seed: int = 1234


def downsample(signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    """Decimate by averaging each source window, which also suppresses aliasing."""
    if source_rate == target_rate:
        return np.array(signal, dtype=np.float64)
    if target_rate > source_rate:
        raise ValueError("target rate must not exceed the source rate")

    ratio = source_rate / target_rate
    n_out = int(signal.shape[-1] / ratio)
    if n_out == 0:
        return np.zeros(0)

    starts = (np.arange(n_out) * ratio).astype(int)
    stops = np.minimum(starts + int(ratio), signal.shape[-1])
    cumulative = np.concatenate([[0.0], np.cumsum(signal)])
    return (cumulative[stops] - cumulative[starts]) / (stops - starts)


def add_motion_noise(
    signal: np.ndarray,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise scaled to a fraction of the signal's standard deviation."""
    if amplitude <= 0.0:
        return signal
    return signal + rng.normal(0.0, signal.std() * amplitude, signal.shape)


def degrade(
    sample: Sample,
    config: IoTConfig,
    rng: Optional[np.random.Generator] = None,
) -> Sample:
    """Reduce a clinical sample to a noisy single-lead trace."""
    rng = rng if rng is not None else np.random.default_rng(config.seed)

    lead = sample.signal[config.lead_index]
    reduced = downsample(lead, sample.sampling_rate, config.target_rate)
    noisy = add_motion_noise(reduced, config.noise_amplitude, rng)
    normalised = zscore(noisy[None, :])

    return Sample(
        record_id=sample.record_id,
        signal=normalised,
        labels=sample.labels,
        source=sample.source,
        sampling_rate=config.target_rate,
        # Degradation is lossy by construction; the score is discounted so that any
        # downstream quality weighting reflects it.
        quality=sample.quality * 0.8,
        age=sample.age,
        sex=sample.sex,
        iot_degraded=True,
    )
