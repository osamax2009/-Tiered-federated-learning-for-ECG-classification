"""Classification metrics for the tiered evaluation.

Two macro averages are reported throughout. The plain macro F1 averages over all
``n_classes`` rows, which penalises a tier for classes that never appear in its own
validation split. The present-class variant averages only over labels that actually
occur in the split being scored, and is the figure to compare across tiers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..model.readout import predict_scores

_LOSS_FLOOR = 1e-15
_EMPTY_LOSS = 10.0


@dataclass
class Metrics:
    accuracy: float
    loss: float
    macro_f1: float
    balanced_accuracy: float
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    support: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def present_classes(self) -> list[int]:
        return [int(c) for c in np.flatnonzero(self.support)]

    def present_macro_f1(self) -> float:
        return average_over(self.per_class_f1, self.present_classes())

    def present_balanced_accuracy(self) -> float:
        return average_over(self.per_class_recall, self.present_classes())


def empty_metrics(n_classes: int) -> Metrics:
    return Metrics(
        accuracy=0.0,
        loss=_EMPTY_LOSS,
        macro_f1=0.0,
        balanced_accuracy=0.0,
        per_class_recall=np.zeros(n_classes),
        per_class_f1=np.zeros(n_classes),
        support=np.zeros(n_classes, dtype=int),
    )


def average_over(values: np.ndarray, classes: Sequence[int]) -> float:
    if len(classes) == 0:
        return 0.0
    return float(np.mean(np.asarray(values)[list(classes)]))


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def evaluate(
    wout: np.ndarray,
    states: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> Metrics:
    """Score a readout on pooled states."""
    states = np.asarray(states)
    labels = np.asarray(labels)
    if states.shape[0] == 0:
        return empty_metrics(n_classes)

    labels = np.clip(labels, 0, n_classes - 1)
    scores = predict_scores(wout, states)
    predicted = np.argmax(scores, axis=1)

    probs = softmax(scores)
    true_probs = np.clip(probs[np.arange(labels.shape[0]), labels], _LOSS_FLOOR, 1.0)
    loss = float(-np.mean(np.log(true_probs)))

    hit = predicted == labels
    tp = np.bincount(labels[hit], minlength=n_classes)
    fp = np.bincount(predicted[~hit], minlength=n_classes)
    fn = np.bincount(labels[~hit], minlength=n_classes)

    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

    return Metrics(
        accuracy=float(np.mean(hit)),
        loss=loss,
        macro_f1=float(np.mean(f1)),
        balanced_accuracy=float(np.mean(recall)),
        per_class_recall=recall,
        per_class_f1=f1,
        support=np.bincount(labels, minlength=n_classes),
    )


def per_class_accuracy(
    wout: np.ndarray,
    states: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Recall per class, used as the confidence signal in selective aggregation."""
    states = np.asarray(states)
    labels = np.asarray(labels)
    if states.shape[0] == 0:
        return np.zeros(n_classes)

    valid = (labels >= 0) & (labels < n_classes)
    labels, states = labels[valid], states[valid]
    if labels.shape[0] == 0:
        return np.zeros(n_classes)
    predicted = predict_scores(wout, states).argmax(axis=1)

    total = np.bincount(labels, minlength=n_classes)
    correct = np.bincount(labels[predicted == labels], minlength=n_classes)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, correct / total, 0.0)


def confusion_matrix(
    wout: np.ndarray,
    states: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """``[actual, predicted]`` counts."""
    labels = np.clip(np.asarray(labels), 0, n_classes - 1)
    if labels.shape[0] == 0:
        return np.zeros((n_classes, n_classes), dtype=int)
    predicted = predict_scores(wout, states).argmax(axis=1)
    flat = np.bincount(labels * n_classes + predicted, minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes)


def mean_metrics(items: Sequence[Metrics], n_classes: int) -> Metrics:
    """Element-wise mean, used to average the final rounds of a run."""
    if not items:
        return empty_metrics(n_classes)
    return Metrics(
        accuracy=float(np.mean([m.accuracy for m in items])),
        loss=float(np.mean([m.loss for m in items])),
        macro_f1=float(np.mean([m.macro_f1 for m in items])),
        balanced_accuracy=float(np.mean([m.balanced_accuracy for m in items])),
        per_class_recall=np.mean([m.per_class_recall for m in items], axis=0),
        per_class_f1=np.mean([m.per_class_f1 for m in items], axis=0),
        support=items[-1].support,
    )
