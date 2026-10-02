"""Validation-gated choice between the two aggregation rules.

Neither FedAvg nor selective aggregation is uniformly better: FedAvg protects the shared
model, selective aggregation helps the weak tier. Instead of committing to one, the
server evaluates the interpolation between them on held-out validation data each round
and keeps the mixture that maximises a stated objective, subject to the shared model not
regressing more than a fixed tolerance against the FedAvg baseline.

The gate reads validation metrics, which are computed from client validation states. In
a deployment those metrics would be reported by the clients rather than computed
centrally; the quantities the gate consumes are per-class scalars either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..evaluation.metrics import Metrics, average_over, evaluate
from .aggregate import blend

# Interpolation grid between FedAvg (0.0) and selective aggregation (1.0).
GRID = np.round(np.linspace(0.0, 1.0, 11), 2)

OBJECTIVES = (
    "iot_present_macro_f1",
    "iot_present_balanced_accuracy",
    "iot_macro_f1",
    "iot_balanced_accuracy",
    "global_macro_f1",
)


@dataclass
class Candidate:
    weight: float
    wout: np.ndarray
    global_metrics: Metrics
    iot_metrics: Metrics
    iot_present_macro_f1: float
    iot_present_balanced_accuracy: float

    def score(self, objective: str) -> tuple[float, ...]:
        """Objective value plus tie-breakers, ordered so that larger is better."""
        if objective == "iot_present_macro_f1":
            return (
                self.iot_present_macro_f1,
                self.iot_present_balanced_accuracy,
                self.global_metrics.macro_f1,
            )
        if objective == "iot_present_balanced_accuracy":
            return (
                self.iot_present_balanced_accuracy,
                self.iot_present_macro_f1,
                self.global_metrics.macro_f1,
            )
        if objective == "iot_macro_f1":
            return (
                self.iot_metrics.macro_f1,
                self.iot_metrics.balanced_accuracy,
                self.global_metrics.macro_f1,
            )
        if objective == "iot_balanced_accuracy":
            return (
                self.iot_metrics.balanced_accuracy,
                self.iot_metrics.macro_f1,
                self.global_metrics.macro_f1,
            )
        if objective == "global_macro_f1":
            return (self.global_metrics.macro_f1, self.iot_metrics.macro_f1)
        raise ValueError(f"unknown gate objective {objective!r}; expected one of {OBJECTIVES}")


@dataclass
class GateDecision:
    wout: np.ndarray
    weight: float
    candidates: list[Candidate]
    feasible: int
    threshold: float

    @property
    def constrained(self) -> bool:
        """True when no mixture met the constraint and the best available was taken."""
        return self.feasible == 0

    def summary(self) -> str:
        state = "unconstrained" if not self.constrained else "constraint unmet"
        return (
            f"weight={self.weight:.1f} ({state}, {self.feasible}/{len(self.candidates)} feasible) "
            f"global_macro_f1={self.candidates[0].global_metrics.macro_f1:.4f}"
            f"->{self.selected().global_metrics.macro_f1:.4f} "
            f"iot_present_macro_f1={self.candidates[0].iot_present_macro_f1:.4f}"
            f"->{self.selected().iot_present_macro_f1:.4f}"
        )

    def selected(self) -> Candidate:
        return next(c for c in self.candidates if c.weight == self.weight)


def select(
    fedavg_wout: np.ndarray,
    selective_wout: np.ndarray,
    global_states: np.ndarray,
    global_labels: np.ndarray,
    iot_states: np.ndarray | None,
    iot_labels: np.ndarray | None,
    n_classes: int,
    objective: str = "iot_present_macro_f1",
    tolerance: float = 0.02,
    iot_present_classes: Sequence[int] | None = None,
) -> GateDecision:
    """Pick the mixture of the two aggregates that best serves the objective.

    Args:
        objective: one of :data:`OBJECTIVES`.
        tolerance: how far the shared model's macro F1 may fall below the FedAvg
            baseline. Mixtures that fall further are excluded.
        iot_present_classes: classes to average over for the present-class objectives.
            Defaults to the classes appearing in the IoT validation split.
    """
    has_iot = iot_states is not None and iot_states.shape[0] > 0
    if iot_present_classes is None:
        iot_present_classes = (
            sorted(set(int(label) for label in iot_labels)) if has_iot else list(range(n_classes))
        )

    candidates: list[Candidate] = []
    for weight in GRID:
        wout = blend(fedavg_wout, selective_wout, float(weight))
        global_metrics = evaluate(wout, global_states, global_labels, n_classes)
        iot_metrics = (
            evaluate(wout, iot_states, iot_labels, n_classes) if has_iot else global_metrics
        )
        candidates.append(
            Candidate(
                weight=float(weight),
                wout=wout,
                global_metrics=global_metrics,
                iot_metrics=iot_metrics,
                iot_present_macro_f1=average_over(iot_metrics.per_class_f1, iot_present_classes),
                iot_present_balanced_accuracy=average_over(
                    iot_metrics.per_class_recall, iot_present_classes
                ),
            )
        )

    baseline = candidates[0].global_metrics.macro_f1
    threshold = baseline - tolerance
    feasible = [c for c in candidates if c.global_metrics.macro_f1 >= threshold]
    best = max(feasible or candidates, key=lambda c: c.score(objective))

    return GateDecision(
        wout=best.wout,
        weight=best.weight,
        candidates=candidates,
        feasible=len(feasible),
        threshold=threshold,
    )


def candidate_table(decision: GateDecision) -> list[dict[str, object]]:
    """Per-mixture diagnostics, for logging or serialisation."""
    return [
        {
            "weight": c.weight,
            "global_macro_f1": c.global_metrics.macro_f1,
            "global_accuracy": c.global_metrics.accuracy,
            "iot_macro_f1": c.iot_metrics.macro_f1,
            "iot_present_macro_f1": c.iot_present_macro_f1,
            "selected": c.weight == decision.weight,
            "feasible": c.global_metrics.macro_f1 >= decision.threshold,
        }
        for c in decision.candidates
    ]
