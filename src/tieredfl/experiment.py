"""The federated training loop and the surrounding run orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import RunConfig
from .data.labels import describe, top_codes
from .data.partition import ClientData, build_clients, filter_to_classes
from .evaluation.metrics import Metrics, empty_metrics, evaluate, mean_metrics
from .federated import gating
from .federated.aggregate import fedavg, selective
from .federated.client import ClientUpdate, train_client
from .federated.features import ClientFeatures, extract_all, pool_states
from .federated.personalize import evaluation_head, fit_tier_head
from .federated.tiers import DeviceTier
from .model.esn import EchoStateNetwork

log = logging.getLogger(__name__)


@dataclass
class RoundResult:
    index: int
    global_metrics: Metrics
    tier_metrics: dict[DeviceTier, Metrics]
    gate_weight: float
    duration_s: float


@dataclass
class RunResult:
    label: str
    rounds: list[RoundResult] = field(default_factory=list)
    final_wout: np.ndarray | None = None

    def final_metrics(
        self, window: int, n_classes: int
    ) -> tuple[Metrics, dict[DeviceTier, Metrics]]:
        """Metrics averaged over the last ``window`` rounds."""
        tail = self.rounds[-window:]
        if not tail:
            return empty_metrics(n_classes), {}

        global_mean = mean_metrics([r.global_metrics for r in tail], n_classes)
        tier_mean = {}
        for tier in DeviceTier:
            items = [r.tier_metrics[tier] for r in tail if tier in r.tier_metrics]
            if items:
                tier_mean[tier] = mean_metrics(items, n_classes)
        return global_mean, tier_mean


@dataclass
class Evaluation:
    """Validation sets the server scores models against."""

    global_states: np.ndarray
    global_labels: np.ndarray
    by_tier: dict[DeviceTier, tuple[np.ndarray, np.ndarray]]

    def tier(self, tier: DeviceTier) -> tuple[np.ndarray, np.ndarray]:
        return self.by_tier.get(
            tier, (np.zeros((0, self.global_states.shape[1])), np.zeros(0, int))
        )

    def iot_present_classes(self) -> list[int]:
        _, labels = self.tier(DeviceTier.IOT)
        return sorted(set(int(label) for label in labels))


@dataclass
class Prepared:
    """Everything a run needs once feature extraction is done."""

    config: RunConfig
    codes: list[str]
    features: dict[str, ClientFeatures]
    tiers: dict[str, DeviceTier]
    evaluation: Evaluation
    clients: list[ClientData]


def prepare(config: RunConfig) -> Prepared:
    """Load data, choose the class set, and extract pooled reservoir features."""
    log.info("loading clients from %s", config.data_root)
    clients = build_clients(
        root=config.data_root,
        specs=config.clients,
        clinical=config.clinical,
        iot=config.iot,
        seed=config.seed,
        scale=config.scale,
        disjoint=config.disjoint_clients,
    )
    if not clients:
        raise RuntimeError(f"no client data loaded; check data_root={config.data_root}")

    all_samples = [s for c in clients for s in list(c.train) + list(c.validation)]
    codes = top_codes(all_samples, config.federation.n_classes)
    log.info("classes: %s", ", ".join(f"[{i}] {describe(c)}" for i, c in enumerate(codes)))

    before = sum(len(c) for c in clients)
    clients = filter_to_classes(clients, codes)
    after = sum(len(c) for c in clients)
    log.info("kept %d/%d samples carrying a selected class", after, before)

    esn = EchoStateNetwork(config.reservoir)
    features = extract_all(
        clients=clients,
        esn=esn,
        codes=codes,
        max_length=config.federation.max_signal_length,
        washout=config.federation.washout,
        batch_size=config.federation.batch_size,
        cache_dir=config.cache_dir,
        cache_key=config.feature_cache_key(),
    )

    tiers = {c.client_id: c.tier for c in clients}
    evaluation = _build_evaluation(features, tiers)
    _log_evaluation(evaluation, codes)

    return Prepared(config, codes, features, tiers, evaluation, clients)


def run(prepared: Prepared, gated: bool, label: str, tier_head: np.ndarray | None) -> RunResult:
    """Train for the configured number of rounds.

    Args:
        gated: choose each round's aggregate with the validation gate. When ``False``,
            FedAvg is used unconditionally, which is the baseline arm.
        tier_head: readout the IoT tier is evaluated under, or ``None`` to evaluate
            every tier under the shared model.
    """
    config = prepared.config.federation
    shape = (
        config.n_classes,
        prepared.features[next(iter(prepared.features))].train_states.shape[1],
    )

    result = RunResult(label=label)
    global_wout: np.ndarray | None = None

    log.info("%s: %d rounds, %s", label, config.rounds, "gated" if gated else "fedavg")
    for index in range(1, config.rounds + 1):
        started = time.monotonic()
        fraction = max(config.min_data_fraction, index / config.rounds)

        updates = _collect_updates(prepared, global_wout, fraction, index)
        if not updates:
            log.warning("round %d produced no updates; skipping", index)
            continue

        fedavg_wout = fedavg(updates, shape)
        if gated:
            iot_states, iot_labels = prepared.evaluation.tier(DeviceTier.IOT)
            decision = gating.select(
                fedavg_wout=fedavg_wout,
                selective_wout=selective(updates, shape),
                global_states=prepared.evaluation.global_states,
                global_labels=prepared.evaluation.global_labels,
                iot_states=iot_states,
                iot_labels=iot_labels,
                n_classes=config.n_classes,
                objective=config.gate_objective,
                tolerance=config.gate_tolerance,
            )
            global_wout, gate_weight = decision.wout, decision.weight
            log.info("round %d gate: %s", index, decision.summary())
        else:
            global_wout, gate_weight = fedavg_wout, 0.0

        global_metrics = evaluate(
            global_wout,
            prepared.evaluation.global_states,
            prepared.evaluation.global_labels,
            config.n_classes,
        )
        tier_metrics = _evaluate_tiers(
            prepared.evaluation,
            global_wout,
            tier_head,
            config.personal_weight,
            config.n_classes,
        )

        result.rounds.append(
            RoundResult(
                index=index,
                global_metrics=global_metrics,
                tier_metrics=tier_metrics,
                gate_weight=gate_weight,
                duration_s=time.monotonic() - started,
            )
        )
        _log_round(result.rounds[-1])

    result.final_wout = global_wout
    return result


def run_all(config: RunConfig, prepared: Prepared | None = None) -> dict[str, object]:
    """Fit the tier head, then run the gated and baseline arms over shared features."""
    prepared = prepared or prepare(config)

    tier_head = fit_tier_head(
        features=prepared.features,
        tiers=prepared.tiers,
        tier=DeviceTier.IOT,
        n_classes=config.federation.n_classes,
        ridge=config.federation.ridge,
    )
    if tier_head is not None:
        iot_states, iot_labels = prepared.evaluation.tier(DeviceTier.IOT)
        metrics = evaluate(tier_head, iot_states, iot_labels, config.federation.n_classes)
        log.info(
            "iot tier head: accuracy=%.4f present_macro_f1=%.4f balanced_accuracy=%.4f",
            metrics.accuracy,
            metrics.present_macro_f1(),
            metrics.balanced_accuracy,
        )

    gated = run(prepared, gated=True, label="gated", tier_head=tier_head)
    baseline = run(prepared, gated=False, label="fedavg", tier_head=tier_head)

    return {
        "prepared": prepared,
        "gated": gated,
        "baseline": baseline,
        "tier_head": tier_head,
    }


def _collect_updates(
    prepared: Prepared,
    global_wout: np.ndarray | None,
    fraction: float,
    round_index: int,
) -> list[ClientUpdate]:
    config = prepared.config.federation
    updates = []
    for client_id, features in prepared.features.items():
        update = train_client(
            client_id=client_id,
            tier=prepared.tiers[client_id],
            features=features,
            n_classes=config.n_classes,
            ridge=config.ridge,
            fraction=fraction,
            global_wout=global_wout,
            noise_sigma=config.transmission_noise_sigma,
            seed=prepared.config.seed,
            round_index=round_index,
        )
        if update is not None:
            updates.append(update)
    return updates


def _evaluate_tiers(
    evaluation: Evaluation,
    global_wout: np.ndarray,
    tier_head: np.ndarray | None,
    personal_weight: float,
    n_classes: int,
) -> dict[DeviceTier, Metrics]:
    metrics = {}
    for tier in DeviceTier:
        states, labels = evaluation.tier(tier)
        if states.shape[0] == 0:
            continue
        wout = (
            evaluation_head(global_wout, tier_head, personal_weight)
            if tier is DeviceTier.IOT
            else global_wout
        )
        metrics[tier] = evaluate(wout, states, labels, n_classes)
    return metrics


def _build_evaluation(
    features: dict[str, ClientFeatures],
    tiers: dict[str, DeviceTier],
) -> Evaluation:
    global_states, global_labels = pool_states(features)
    by_tier = {}
    for tier in DeviceTier:
        members = [cid for cid, t in tiers.items() if t is tier]
        states, labels = pool_states(features, members)
        if states.shape[0] > 0:
            by_tier[tier] = (states, labels)
    return Evaluation(global_states, global_labels, by_tier)


def _log_evaluation(evaluation: Evaluation, codes: Sequence[str]) -> None:
    log.info("global validation set: %d samples", evaluation.global_states.shape[0])
    for tier in DeviceTier:
        states, labels = evaluation.tier(tier)
        if states.shape[0] == 0:
            continue
        counts = np.bincount(labels, minlength=len(codes))
        present = ", ".join(f"[{i}]={counts[i]}" for i in range(len(codes)) if counts[i] > 0)
        log.info("%s validation: %d samples {%s}", tier, states.shape[0], present)

    missing = set(range(len(codes))) - set(evaluation.iot_present_classes())
    if missing:
        log.info(
            "classes absent from the iot split, excluded from present-class averages: %s",
            ", ".join(f"[{i}] {codes[i]}" for i in sorted(missing)),
        )


def _log_round(result: RoundResult) -> None:
    iot = result.tier_metrics.get(DeviceTier.IOT)
    iot_text = f" iot_present_macro_f1={iot.present_macro_f1():.4f}" if iot else ""
    log.info(
        "round %d: global_accuracy=%.4f global_macro_f1=%.4f%s weight=%.1f (%.1fs)",
        result.index,
        result.global_metrics.accuracy,
        result.global_metrics.macro_f1,
        iot_text,
        result.gate_weight,
        result.duration_s,
    )
