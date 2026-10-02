"""Tier-aware feature extraction.

Because the reservoir is fixed, a client's pooled states never change across rounds.
Extraction therefore runs once per configuration and the result is cached on disk; this
is the dominant cost of the whole experiment, and caching it is what makes repeated
runs and ablations cheap.

Each tier sees only what its hardware could capture: the hospital tier keeps all leads,
the wearable tier three, the IoT tier one. Allowed leads are cleaned, standardised
against tier-level statistics, then concatenated into the single stream the reservoir
consumes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.clinical import centered_moving_average
from ..data.labels import label_index
from ..data.partition import ClientData
from ..model.esn import EchoStateNetwork
from .tiers import DeviceTier

log = logging.getLogger(__name__)

# The IoT tier gets an extra cleanup pass: a wide moving average estimates residual
# baseline wander, and a narrow one suppresses the injected high-frequency noise.
IOT_BASELINE_WINDOW = 201
IOT_SMOOTHING_WINDOW = 5


@dataclass(frozen=True)
class TierStats:
    """Mean and standard deviation of a tier's cleaned signal amplitude."""

    mean: float
    std: float

    def apply(self, signal: np.ndarray) -> np.ndarray:
        return (signal - self.mean) / self.std


@dataclass
class ClientFeatures:
    """Pooled reservoir states for one client, with its labels."""

    train_states: np.ndarray
    train_labels: np.ndarray
    val_states: np.ndarray
    val_labels: np.ndarray

    @property
    def n_train(self) -> int:
        return int(self.train_states.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.val_states.shape[0])

    def class_counts(self, n_classes: int, limit: int | None = None) -> np.ndarray:
        labels = self.train_labels if limit is None else self.train_labels[:limit]
        return np.bincount(labels, minlength=n_classes)


def allowed_leads(signal: np.ndarray, tier: DeviceTier) -> np.ndarray:
    """Restrict a recording to the leads the tier's hardware could record."""
    if tier is DeviceTier.HOSPITAL:
        return signal
    return signal[: min(tier.leads, signal.shape[0])]


def clean_lead(lead: np.ndarray, tier: DeviceTier, max_length: int) -> np.ndarray:
    lead = lead[:max_length]
    if tier is not DeviceTier.IOT or lead.size == 0:
        return lead
    detrended = lead - centered_moving_average(lead, IOT_BASELINE_WINDOW)
    return centered_moving_average(detrended, IOT_SMOOTHING_WINDOW)


def tier_statistics(
    clients: Sequence[ClientData],
    max_length: int,
) -> dict[DeviceTier, TierStats]:
    """Pool amplitude statistics per tier over all of that tier's cleaned leads.

    Standardising per tier rather than per client keeps clients within a tier on a
    comparable scale, so a readout trained on one is meaningful on another. Statistics
    are shared knowledge about device classes, not per-client data.
    """
    totals: dict[DeviceTier, tuple[float, float, int]] = {}
    for client in clients:
        for sample in list(client.train) + list(client.validation):
            for lead in allowed_leads(sample.signal, client.tier):
                cleaned = clean_lead(lead, client.tier, max_length)
                if cleaned.size == 0:
                    continue
                total, total_sq, count = totals.get(client.tier, (0.0, 0.0, 0))
                totals[client.tier] = (
                    total + float(cleaned.sum()),
                    total_sq + float(np.square(cleaned).sum()),
                    count + cleaned.size,
                )

    stats: dict[DeviceTier, TierStats] = {}
    for tier in DeviceTier:
        total, total_sq, count = totals.get(tier, (0.0, 0.0, 0))
        if count <= 1:
            stats[tier] = TierStats(0.0, 1.0)
            continue
        mean = total / count
        variance = max(total_sq / count - mean * mean, 1e-12)
        stats[tier] = TierStats(mean, float(np.sqrt(variance)))
    return stats


def prepare_stream(
    signal: np.ndarray,
    tier: DeviceTier,
    stats: TierStats,
    max_length: int,
) -> np.ndarray:
    """Clean, standardise and concatenate a recording's leads into one stream."""
    leads = [clean_lead(lead, tier, max_length) for lead in allowed_leads(signal, tier)]
    leads = [stats.apply(lead) for lead in leads if lead.size > 0]
    if not leads:
        return np.zeros(0)
    return np.concatenate(leads)


def extract_client(
    client: ClientData,
    esn: EchoStateNetwork,
    codes: Sequence[str],
    stats: TierStats,
    max_length: int,
    washout: int,
    batch_size: int,
) -> ClientFeatures:
    """Run one client's samples through the reservoir."""
    splits = []
    for samples in (client.train, client.validation):
        streams, labels = [], []
        for sample in samples:
            index = label_index(sample.labels, codes)
            if index is None:
                continue
            stream = prepare_stream(sample.signal, client.tier, stats, max_length)
            if stream.size < 2:
                continue
            streams.append(stream)
            labels.append(index)

        if streams:
            states = esn.pooled_states(np.vstack(streams), washout, batch_size)
        else:
            states = np.zeros((0, esn.state_size))
        splits.append((states, np.asarray(labels, dtype=np.int64)))

    (train_states, train_labels), (val_states, val_labels) = splits
    return ClientFeatures(train_states, train_labels, val_states, val_labels)


def extract_all(
    clients: Sequence[ClientData],
    esn: EchoStateNetwork,
    codes: Sequence[str],
    max_length: int,
    washout: int,
    batch_size: int = 64,
    cache_dir: Path | None = None,
    cache_key: str = "",
) -> dict[str, ClientFeatures]:
    """Extract features for every client, reading and writing a disk cache if given."""
    stats = tier_statistics(clients, max_length)
    for tier, tier_stats in stats.items():
        log.info("%s amplitude: mean=%.5f std=%.5f", tier, tier_stats.mean, tier_stats.std)

    features: dict[str, ClientFeatures] = {}
    for client in clients:
        cached = _load_cached(cache_dir, cache_key, client.client_id)
        if cached is not None:
            log.info(
                "%s: %d train, %d validation (cached)",
                client.client_id,
                cached.n_train,
                cached.n_val,
            )
            features[client.client_id] = cached
            continue

        extracted = extract_client(
            client, esn, codes, stats[client.tier], max_length, washout, batch_size
        )
        log.info(
            "%s: %d train, %d validation", client.client_id, extracted.n_train, extracted.n_val
        )
        _store_cached(cache_dir, cache_key, client.client_id, extracted)
        features[client.client_id] = extracted

    return features


def pool_states(
    features: dict[str, ClientFeatures],
    client_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the validation splits of the given clients into one evaluation set."""
    ids = list(features) if client_ids is None else [i for i in client_ids if i in features]
    states = [features[i].val_states for i in ids if features[i].n_val > 0]
    labels = [features[i].val_labels for i in ids if features[i].n_val > 0]
    if not states:
        width = next(iter(features.values())).train_states.shape[1] if features else 1
        return np.zeros((0, width)), np.zeros(0, dtype=np.int64)
    return np.vstack(states), np.concatenate(labels)


def _cache_path(cache_dir: Path, cache_key: str, client_id: str) -> Path:
    return Path(cache_dir) / cache_key / f"{client_id}.npz"


def _load_cached(cache_dir: Path | None, cache_key: str, client_id: str) -> ClientFeatures | None:
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, cache_key, client_id)
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            return ClientFeatures(
                data["train_states"],
                data["train_labels"],
                data["val_states"],
                data["val_labels"],
            )
    except (OSError, KeyError, ValueError) as exc:
        log.warning("ignoring unreadable cache %s: %s", path, exc)
        return None


def _store_cached(
    cache_dir: Path | None,
    cache_key: str,
    client_id: str,
    features: ClientFeatures,
) -> None:
    if cache_dir is None:
        return
    path = _cache_path(cache_dir, cache_key, client_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_states=features.train_states,
        train_labels=features.train_labels,
        val_states=features.val_states,
        val_labels=features.val_labels,
    )
