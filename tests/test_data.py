"""Record reading, clinical preprocessing, IoT degradation and label selection."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tieredfl.data.clinical import (
    ClinicalConfig,
    Sample,
    assess_quality,
    bandpass,
    centered_moving_average,
    fit_length,
    preprocess,
    resample,
    zscore,
)
from tieredfl.data.iot import IoTConfig, add_motion_noise, degrade, downsample
from tieredfl.data.labels import label_index, top_codes
from tieredfl.data.wfdb import Record, read_header, read_mat, read_record

HEADER = """A0001 2 500 1000
A0001.mat 16+24 1000.0(0)/mV 16 0 -72 -1716 0 I
A0001.mat 16+24 1000.0(0)/mV 16 0 11 2029 0 II
# Age: 74
# Sex: Male
# Dx: 426783006,164889003
"""


def write_record(tmp_path, signal: np.ndarray, header: str = HEADER):
    header_path = tmp_path / "A0001.hea"
    header_path.write_text(header)

    n_rows, n_cols = signal.shape
    payload = struct.pack("<5i", 30, n_rows, n_cols, 0, 4) + b"val\x00"
    payload += signal.T.astype("<i2").tobytes()
    (tmp_path / "A0001.mat").write_bytes(payload)
    return header_path


def test_header_parsing(tmp_path):
    write_record(tmp_path, np.zeros((2, 4), dtype=np.int16))
    header = read_header(tmp_path / "A0001.hea")

    assert header.record_id == "A0001"
    assert header.sampling_rate == 500.0
    assert header.age == 74.0
    assert header.sex == "Male"
    assert header.diagnoses == ["426783006", "164889003"]
    assert [lead.name for lead in header.leads] == ["I", "II"]
    assert header.leads[0].adc_gain == 1000.0
    assert header.leads[0].baseline == -72


def test_mat_reading_recovers_the_matrix(tmp_path):
    signal = np.arange(12, dtype=np.int16).reshape(2, 6)
    write_record(tmp_path, signal)
    assert np.array_equal(read_mat(tmp_path / "A0001.mat"), signal.astype(float))


def test_mat_reading_truncates_columns(tmp_path):
    signal = np.arange(20, dtype=np.int16).reshape(2, 10)
    write_record(tmp_path, signal)
    truncated = read_mat(tmp_path / "A0001.mat", max_samples=4)
    assert truncated.shape == (2, 4)
    assert np.array_equal(truncated, signal[:, :4].astype(float))


def test_mat_reading_rejects_an_unexpected_variable(tmp_path):
    write_record(tmp_path, np.zeros((2, 4), dtype=np.int16))
    with pytest.raises(ValueError, match="expected variable"):
        read_mat(tmp_path / "A0001.mat", key="signal")


def test_record_conversion_applies_baseline_and_gain(tmp_path):
    signal = np.full((2, 5), 1000, dtype=np.int16)
    header_path = write_record(tmp_path, signal)
    record = read_record(header_path, source="unit")

    assert record.source == "unit"
    assert record.diagnoses == ["426783006", "164889003"]
    assert np.allclose(record.signal[0], (1000 - (-72)) / 1000.0)
    assert np.allclose(record.signal[1], (1000 - 11) / 1000.0)


def test_quality_rejects_non_finite_signals():
    signal = np.random.default_rng(0).standard_normal((2, 100))
    signal[0, 5] = np.nan
    assert assess_quality(signal) == 0.0


def test_quality_penalises_flat_leads():
    healthy = synthetic_ecg(n_leads=2, n_samples=1500)
    half_flat = healthy.copy()
    half_flat[0] = 0.0
    assert assess_quality(half_flat) < assess_quality(healthy)


def test_quality_penalises_jittery_traces():
    smooth = synthetic_ecg(n_leads=2, n_samples=1500)
    jittery = np.random.default_rng(1).standard_normal((2, 1500))
    assert assess_quality(jittery) < assess_quality(smooth)


def test_resampling_changes_length_by_the_rate_ratio():
    signal = np.sin(np.linspace(0.0, 10.0, 1000))[None, :]
    assert resample(signal, 500.0, 250.0).shape[1] == 500
    assert resample(signal, 500.0, 500.0).shape[1] == 1000


def test_fit_length_truncates_and_pads():
    signal = np.ones((2, 10))
    assert fit_length(signal, 4).shape == (2, 4)
    padded = fit_length(signal, 15)
    assert padded.shape == (2, 15)
    assert np.allclose(padded[:, 10:], 0.0)


def test_centered_moving_average_matches_a_direct_computation():
    rng = np.random.default_rng(2)
    signal = rng.standard_normal(60)
    window = 7
    half = window // 2
    expected = np.array(
        [signal[max(0, i - half) : min(60, i + half + 1)].mean() for i in range(60)]
    )
    assert np.allclose(centered_moving_average(signal, window), expected)


def test_centered_moving_average_preserves_a_constant_signal():
    signal = np.full(50, 3.0)
    assert np.allclose(centered_moving_average(signal, 11), 3.0)


def test_centered_moving_average_is_a_noop_for_trivial_windows():
    signal = np.arange(5.0)
    assert np.allclose(centered_moving_average(signal, 1), signal)


def test_bandpass_attenuates_a_slow_drift_more_than_the_passband():
    rate, t = 500.0, np.arange(2500) / 500.0
    drift = np.sin(2 * np.pi * 0.05 * t)[None, :]
    passband = np.sin(2 * np.pi * 10.0 * t)[None, :]
    drift_gain = bandpass(drift, rate, 0.5, 40.0).std() / drift.std()
    passband_gain = bandpass(passband, rate, 0.5, 40.0).std() / passband.std()
    assert drift_gain < passband_gain


def test_zscore_standardises_each_lead_and_leaves_flat_leads_alone():
    signal = np.vstack([np.arange(100.0), np.full(100, 2.0)])
    scaled = zscore(signal)
    assert scaled[0].mean() == pytest.approx(0.0)
    assert scaled[0].std() == pytest.approx(1.0)
    assert np.allclose(scaled[1], 2.0)


def synthetic_ecg(n_leads: int = 12, n_samples: int = 3000, rate: float = 500.0) -> np.ndarray:
    """A smooth periodic trace with a QRS-like spike, plus mild measurement noise.

    White noise would be rejected by quality control: its sample-to-sample jitter
    trips the noise penalty, which is the behaviour the check is there for.
    """
    rng = np.random.default_rng(3)
    t = np.arange(n_samples) / rate
    beat = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)
    spikes = np.exp(-(((t % (1 / 1.2)) - 0.05) ** 2) / (2 * 0.004**2))
    base = beat + 2.0 * spikes
    return np.vstack([base * (1.0 + 0.1 * lead) for lead in range(n_leads)]) + rng.normal(
        0.0, 0.01, (n_leads, n_samples)
    )


def test_preprocess_produces_a_fixed_window():
    record = Record(
        record_id="r",
        signal=synthetic_ecg(n_samples=3000),
        sampling_rate=257.0,
        diagnoses=["426783006"],
        age=50.0,
        sex="Female",
        source="unit",
    )
    config = ClinicalConfig(target_length=5000, target_rate=500.0)
    sample = preprocess(record, config)

    assert sample is not None
    assert sample.signal.shape == (12, 5000)
    assert sample.sampling_rate == 500.0
    assert sample.labels == ("426783006",)


def test_preprocess_drops_a_record_that_fails_quality_control():
    record = Record(
        record_id="flat",
        signal=np.zeros((12, 3000)),
        sampling_rate=500.0,
        diagnoses=["426783006"],
        age=None,
        sex=None,
        source="unit",
    )
    assert preprocess(record, ClinicalConfig()) is None


def test_downsample_averages_each_window():
    signal = np.arange(10.0)
    assert np.allclose(downsample(signal, 500.0, 250.0), [0.5, 2.5, 4.5, 6.5, 8.5])
    assert np.allclose(downsample(signal, 500.0, 500.0), signal)


def test_downsample_refuses_to_upsample():
    with pytest.raises(ValueError, match="must not exceed"):
        downsample(np.arange(10.0), 250.0, 500.0)


def test_motion_noise_scales_with_signal_amplitude():
    rng = np.random.default_rng(4)
    quiet = np.sin(np.linspace(0, 20, 2000))
    loud = quiet * 50.0
    quiet_delta = add_motion_noise(quiet, 0.1, np.random.default_rng(5)) - quiet
    loud_delta = add_motion_noise(loud, 0.1, np.random.default_rng(5)) - loud
    assert loud_delta.std() > quiet_delta.std() * 10
    assert np.array_equal(add_motion_noise(quiet, 0.0, rng), quiet)


def test_degrade_reduces_a_clinical_sample_to_one_noisy_lead():
    rng = np.random.default_rng(6)
    sample = Sample(
        record_id="r",
        signal=rng.standard_normal((12, 5000)),
        labels=("426783006",),
        source="unit",
        sampling_rate=500.0,
        quality=1.0,
    )
    degraded = degrade(sample, IoTConfig(target_rate=250.0), rng)

    assert degraded.signal.shape == (1, 2500)
    assert degraded.sampling_rate == 250.0
    assert degraded.iot_degraded is True
    assert degraded.quality == pytest.approx(0.8)
    assert degraded.labels == sample.labels


def test_degrade_is_reproducible_for_a_given_generator():
    sample = Sample(
        record_id="r",
        signal=np.random.default_rng(7).standard_normal((12, 1000)),
        labels=("x",),
        source="unit",
        sampling_rate=500.0,
        quality=1.0,
    )
    first = degrade(sample, IoTConfig(), np.random.default_rng(8)).signal
    second = degrade(sample, IoTConfig(), np.random.default_rng(8)).signal
    assert np.array_equal(first, second)


def sample_with(labels):
    return Sample(
        record_id="r",
        signal=np.zeros((1, 10)),
        labels=tuple(labels),
        source="unit",
        sampling_rate=500.0,
        quality=1.0,
    )


def test_top_codes_ranks_by_frequency():
    samples = [sample_with(["a"])] * 5 + [sample_with(["b"])] * 3 + [sample_with(["c"])]
    assert top_codes(samples, 2) == ["a", "b"]


def test_top_codes_breaks_ties_deterministically():
    samples = [sample_with(["b"]), sample_with(["a"])]
    assert top_codes(samples, 2) == ["a", "b"]


def test_label_index_resolves_to_the_most_frequent_matching_code():
    assert label_index(["z", "b"], ["a", "b", "c"]) == 1
    assert label_index(["c", "a"], ["a", "b", "c"]) == 0
    assert label_index(["q"], ["a", "b"]) is None
