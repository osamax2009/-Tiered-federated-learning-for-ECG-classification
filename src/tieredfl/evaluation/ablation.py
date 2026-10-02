"""Reference points for the IoT tier.

A five-class accuracy figure for the IoT tier is easy to misread, because the tier's
validation split does not contain all five classes. These baselines are all scored on
the identical IoT split so that any reported gain is stated against something
meaningful rather than against a five-way argmax the tier never faces:

``majority``
    Always predict the most frequent present class. The floor any method must clear.
``restricted_global``
    The shared model with its argmax restricted to the classes present in the split.
    Separates genuine personalisation gains from the gain of simply not predicting
    absent classes.
``local_only``
    Each client's own head, unaggregated. If its mean matches the federated head,
    federating the tier bought nothing.
``federated``
    The aggregated tier head.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..federated.features import ClientFeatures
from ..federated.personalize import local_only_heads
from ..federated.tiers import DeviceTier
from ..model.readout import predict_scores
from .metrics import average_over, evaluate


@dataclass
class BaselineScore:
    name: str
    accuracy: float
    present_macro_f1: float | None = None
    balanced_accuracy: float | None = None
    detail: dict[str, float] = field(default_factory=dict)


def majority_class(labels: np.ndarray, present_classes: Sequence[int]) -> BaselineScore:
    counts = np.bincount(labels, minlength=max(present_classes, default=0) + 1)
    pick = (
        max(present_classes, key=lambda c: counts[c]) if present_classes else int(counts.argmax())
    )
    return BaselineScore(
        name="majority",
        accuracy=float(counts[pick] / labels.shape[0]),
        detail={"class": float(pick)},
    )


def restricted_argmax(
    wout: np.ndarray,
    states: np.ndarray,
    labels: np.ndarray,
    present_classes: Sequence[int],
) -> BaselineScore:
    """Score a readout with its decision restricted to the classes present in the split."""
    present = list(present_classes)
    scores = predict_scores(wout, states)[:, present]
    predicted = np.asarray(present)[scores.argmax(axis=1)]

    hit = predicted == labels
    per_class_f1, per_class_recall = [], []
    for cls in present:
        tp = int(np.count_nonzero(hit & (labels == cls)))
        fp = int(np.count_nonzero(~hit & (predicted == cls)))
        fn = int(np.count_nonzero(~hit & (labels == cls)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class_f1.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class_recall.append(recall)

    return BaselineScore(
        name="restricted_global",
        accuracy=float(np.mean(hit)),
        present_macro_f1=float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        balanced_accuracy=float(np.mean(per_class_recall)) if per_class_recall else 0.0,
    )


def iot_baselines(
    global_wout: np.ndarray | None,
    tier_head: np.ndarray | None,
    features: dict[str, ClientFeatures],
    tiers: dict[str, DeviceTier],
    states: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    ridge: float,
    present_classes: Sequence[int],
) -> list[BaselineScore]:
    """Score every baseline on the IoT validation split."""
    if states.shape[0] == 0:
        return []

    scores: list[BaselineScore] = [majority_class(labels, present_classes)]

    if global_wout is not None:
        scores.append(restricted_argmax(global_wout, states, labels, present_classes))
        unrestricted = evaluate(global_wout, states, labels, n_classes)
        scores.append(
            BaselineScore(
                name="shared_global",
                accuracy=unrestricted.accuracy,
                present_macro_f1=average_over(unrestricted.per_class_f1, present_classes),
                balanced_accuracy=unrestricted.balanced_accuracy,
            )
        )

    heads = local_only_heads(features, tiers, DeviceTier.IOT, n_classes, ridge)
    per_client = []
    for client_id, head in heads.items():
        metrics = evaluate(head, states, labels, n_classes)
        present_f1 = average_over(metrics.per_class_f1, present_classes)
        per_client.append((metrics.accuracy, present_f1))
        scores.append(
            BaselineScore(
                name=f"local_only[{client_id}]",
                accuracy=metrics.accuracy,
                present_macro_f1=present_f1,
                balanced_accuracy=metrics.balanced_accuracy,
            )
        )
    if per_client:
        scores.append(
            BaselineScore(
                name="local_only_mean",
                accuracy=float(np.mean([a for a, _ in per_client])),
                present_macro_f1=float(np.mean([f for _, f in per_client])),
                detail={"clients": float(len(per_client))},
            )
        )

    if tier_head is not None:
        metrics = evaluate(tier_head, states, labels, n_classes)
        scores.append(
            BaselineScore(
                name="federated",
                accuracy=metrics.accuracy,
                present_macro_f1=average_over(metrics.per_class_f1, present_classes),
                balanced_accuracy=metrics.balanced_accuracy,
            )
        )

    return scores
