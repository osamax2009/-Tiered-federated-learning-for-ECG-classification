"""Run configuration.

Everything that affects a result is declared here and loaded from YAML, so a run is
described by one file plus a seed. Nothing reads environment variables or global state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .data.clinical import ClinicalConfig
from .data.iot import IoTConfig
from .data.partition import ClientSpec
from .federated.tiers import DeviceTier
from .model.esn import ReservoirSpec
from .seeding import stable_hash


@dataclass(frozen=True)
class FederationConfig:
    rounds: int = 30
    n_classes: int = 5
    ridge: float = 1e-2
    washout: int = 50
    max_signal_length: int = 2000
    batch_size: int = 64

    gate_objective: str = "iot_present_macro_f1"
    gate_tolerance: float = 0.02

    # Fraction of a client's data used in round 1, ramping to 1.0 by the final round.
    min_data_fraction: float = 0.3
    # 1.0 evaluates the IoT tier under the pure tier head; 0.0 under the shared model.
    personal_weight: float = 1.0
    transmission_noise_sigma: float = 0.0
    # Rounds averaged into the headline figures.
    final_round_window: int = 3


@dataclass
class RunConfig:
    name: str = "tiered-fl"
    data_root: Path = Path("data")
    output_dir: Path = Path("results")
    cache_dir: Path | None = Path(".feature_cache")
    seed: int = 0
    scale: float = 1.0
    disjoint_clients: bool = True

    reservoir: ReservoirSpec = field(default_factory=ReservoirSpec)
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)
    iot: IoTConfig = field(default_factory=IoTConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    clients: list[ClientSpec] = field(default_factory=list)

    def feature_cache_key(self) -> str:
        """Identifies the inputs that change extracted features.

        Rounds, the gate and the personalisation weight are deliberately excluded: they
        do not affect the reservoir pass, so varying them reuses the same cache.
        """
        return "{:08x}".format(
            stable_hash(
                self.seed,
                self.scale,
                self.disjoint_clients,
                sorted(
                    (c.client_id, c.dataset, c.tier.label, c.n_records, c.train_ratio)
                    for c in self.clients
                ),
                asdict(self.reservoir),
                asdict(self.clinical),
                asdict(self.iot),
                self.federation.washout,
                self.federation.max_signal_length,
                self.federation.n_classes,
            )
        )

    def tiers_by_client(self) -> dict[str, DeviceTier]:
        return {c.client_id: c.tier for c in self.clients}


def load(path: Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Read a YAML config, apply top-level overrides, and validate it."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw = {**raw, **{k: v for k, v in overrides.items() if v is not None}}
    return from_dict(raw)


def from_dict(raw: dict[str, Any]) -> RunConfig:
    clients = [
        ClientSpec(
            client_id=entry["id"],
            dataset=entry["dataset"],
            tier=DeviceTier.from_label(entry["tier"]),
            n_records=int(entry["records"]),
            train_ratio=float(entry.get("train_ratio", 0.8)),
        )
        for entry in raw.get("clients", [])
    ]

    config = RunConfig(
        name=raw.get("name", "tiered-fl"),
        data_root=Path(raw.get("data_root", "data")),
        output_dir=Path(raw.get("output_dir", "results")),
        cache_dir=None if raw.get("cache_dir") in (None, False) else Path(raw["cache_dir"]),
        seed=int(raw.get("seed", 0)),
        scale=float(raw.get("scale", 1.0)),
        disjoint_clients=bool(raw.get("disjoint_clients", True)),
        reservoir=ReservoirSpec(**raw.get("reservoir", {})),
        clinical=ClinicalConfig(**raw.get("clinical", {})),
        iot=IoTConfig(**raw.get("iot", {})),
        federation=FederationConfig(**raw.get("federation", {})),
        clients=clients,
    )
    validate(config)
    return config


def validate(config: RunConfig) -> None:
    from .federated.gating import OBJECTIVES

    if not config.clients:
        raise ValueError("config declares no clients")

    duplicates = {c.client_id for c in config.clients}
    if len(duplicates) != len(config.clients):
        raise ValueError("client ids must be unique")

    if config.federation.gate_objective not in OBJECTIVES:
        raise ValueError(
            f"gate_objective {config.federation.gate_objective!r} not in {list(OBJECTIVES)}"
        )
    if config.federation.rounds < 1:
        raise ValueError("rounds must be at least 1")
    if not 0.0 <= config.federation.personal_weight <= 1.0:
        raise ValueError("personal_weight must lie in [0, 1]")
    if not 0.0 < config.scale <= 1.0:
        raise ValueError("scale must lie in (0, 1]")
    if config.federation.max_signal_length <= config.federation.washout:
        raise ValueError("max_signal_length must exceed washout")


def to_dict(config: RunConfig) -> dict[str, Any]:
    """Round-trippable representation, written alongside every result."""
    return {
        "name": config.name,
        "data_root": str(config.data_root),
        "output_dir": str(config.output_dir),
        "cache_dir": str(config.cache_dir) if config.cache_dir else None,
        "seed": config.seed,
        "scale": config.scale,
        "disjoint_clients": config.disjoint_clients,
        "reservoir": asdict(config.reservoir),
        "clinical": asdict(config.clinical),
        "iot": asdict(config.iot),
        "federation": asdict(config.federation),
        "clients": [
            {
                "id": c.client_id,
                "dataset": c.dataset,
                "tier": c.tier.label,
                "records": c.n_records,
                "train_ratio": c.train_ratio,
            }
            for c in config.clients
        ],
    }


def with_federation(config: RunConfig, **changes: Any) -> RunConfig:
    """Copy a config with federation settings changed, for sweeps."""
    return replace(config, federation=replace(config.federation, **changes))
