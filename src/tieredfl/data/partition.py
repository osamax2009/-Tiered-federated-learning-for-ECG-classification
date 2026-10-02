"""Construction of federated clients from the challenge sources.

Each client draws from one dataset, which is where the non-IID structure comes from:
the sources differ in recruitment, demographics and label prevalence, so a per-dataset
split is a natural rather than a synthetic partition.

Records are allocated disjointly when several clients share a dataset. The original
implementation sampled each client independently from the shared pool, which let the
three IoT clients overlap; ``disjoint`` reproduces that behaviour when set to ``False``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..federated.tiers import DeviceTier
from ..seeding import generator
from .clinical import ClinicalConfig, Sample, preprocess
from .iot import IoTConfig, degrade
from .labels import label_index
from .loader import load_dataset
from .wfdb import Record

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClientSpec:
    """How one client is built: where its data comes from and how much it gets."""

    client_id: str
    dataset: str
    tier: DeviceTier
    n_records: int
    train_ratio: float = 0.8


@dataclass
class ClientData:
    client_id: str
    tier: DeviceTier
    dataset: str
    train: list[Sample] = field(default_factory=list)
    validation: list[Sample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.train) + len(self.validation)


def build_clients(
    root: Path,
    specs: Sequence[ClientSpec],
    clinical: ClinicalConfig,
    iot: IoTConfig,
    seed: int = 0,
    scale: float = 1.0,
    disjoint: bool = True,
) -> list[ClientData]:
    """Load, preprocess and split records for every client.

    Datasets are processed one at a time so that only one source is resident at once.

    Args:
        root: dataset root, either the download root or the training directory.
        specs: client definitions.
        clinical: shared clinical preprocessing settings.
        iot: degradation settings for IoT-tier clients.
        seed: controls record selection and IoT noise.
        scale: fraction of each client's configured record count to use. Lets a
            smoke run exercise the whole pipeline at a fraction of the cost.
        disjoint: allocate non-overlapping records to clients sharing a dataset.
    """
    by_dataset: dict[str, list[ClientSpec]] = {}
    for spec in specs:
        by_dataset.setdefault(spec.dataset, []).append(spec)

    clients: list[ClientData] = []
    for dataset, dataset_specs in by_dataset.items():
        wanted = {spec.client_id: _scaled(spec.n_records, scale) for spec in dataset_specs}
        pool_size = sum(wanted.values()) if disjoint else max(wanted.values())
        records = list(
            load_dataset(root, dataset, limit=pool_size + 16, max_samples=clinical.target_length)
        )

        if not records:
            log.warning("%s unavailable; skipping %d client(s)", dataset, len(dataset_specs))
            continue

        for spec, selected in _allocate(records, dataset_specs, wanted, seed, disjoint).items():
            client = _build_client(spec, selected, clinical, iot, seed)
            if client is not None:
                clients.append(client)
                log.info(
                    "%s [%s]: %d train, %d validation",
                    client.client_id,
                    client.tier,
                    len(client.train),
                    len(client.validation),
                )

        records.clear()

    return clients


def _scaled(n_records: int, scale: float) -> int:
    if scale >= 1.0:
        return n_records
    return max(2, int(n_records * scale))


def _allocate(
    records: list[Record],
    specs: Sequence[ClientSpec],
    wanted: dict[str, int],
    seed: int,
    disjoint: bool,
) -> dict[ClientSpec, list[Record]]:
    """Assign records to the clients drawing from one dataset."""
    allocation: dict[ClientSpec, list[Record]] = {}
    rng = generator(seed, "allocate", records[0].source)
    order = rng.permutation(len(records))

    cursor = 0
    for spec in specs:
        count = min(wanted[spec.client_id], len(records))
        if disjoint:
            picked = order[cursor : cursor + count]
            cursor += count
            if picked.size == 0:
                log.warning("dataset exhausted before %s could be filled", spec.client_id)
        else:
            picked = rng.permutation(len(records))[:count]
        allocation[spec] = [records[i] for i in picked]

    return allocation


def _build_client(
    spec: ClientSpec,
    records: Sequence[Record],
    clinical: ClinicalConfig,
    iot: IoTConfig,
    seed: int,
) -> ClientData | None:
    if not records:
        return None

    # One generator per client keeps the injected noise reproducible and independent
    # of how many other clients were built first.
    noise_rng = generator(seed, "iot-noise", iot.seed, spec.client_id)

    samples: list[Sample] = []
    for record in records:
        sample = preprocess(record, clinical)
        if sample is None:
            continue
        if spec.tier is DeviceTier.IOT:
            sample = degrade(sample, iot, noise_rng)
        samples.append(sample)

    if not samples:
        log.warning("%s: every record failed quality control", spec.client_id)
        return None

    split = int(len(samples) * spec.train_ratio)
    return ClientData(
        client_id=spec.client_id,
        tier=spec.tier,
        dataset=spec.dataset,
        train=samples[:split],
        validation=samples[split:],
    )


def filter_to_classes(clients: Sequence[ClientData], codes: Sequence[str]) -> list[ClientData]:
    """Drop samples carrying none of the selected class codes, and empty clients with them."""
    kept: list[ClientData] = []
    for client in clients:
        train = [s for s in client.train if label_index(s.labels, codes) is not None]
        validation = [s for s in client.validation if label_index(s.labels, codes) is not None]
        if train:
            kept.append(
                ClientData(
                    client_id=client.client_id,
                    tier=client.tier,
                    dataset=client.dataset,
                    train=train,
                    validation=validation,
                )
            )
        else:
            log.warning("%s has no in-class training samples; dropped", client.client_id)
    return kept
