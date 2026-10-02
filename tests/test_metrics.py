from __future__ import annotations

import numpy as np
import pytest

from tieredfl.evaluation.metrics import (
    average_over,
    confusion_matrix,
    evaluate,
    mean_metrics,
    per_class_accuracy,
    softmax,
)


def one_hot_readout(n_classes: int, state_size: int) -> np.ndarray:
    """A readout that reads the class straight off the first ``n_classes`` features."""
    wout = np.zeros((n_classes, state_size))
    wout[np.arange(n_classes), np.arange(n_classes)] = 1.0
    return wout


def states_for(labels: np.ndarray, n_classes: int, state_size: int = 6) -> np.ndarray:
    states = np.zeros((labels.shape[0], state_size))
    states[np.arange(labels.shape[0]), labels] = 1.0
    return states


def test_perfect_predictions_score_one():
    labels = np.array([0, 1, 2, 2, 1, 0])
    metrics = evaluate(one_hot_readout(3, 6), states_for(labels, 3), labels, 3)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.balanced_accuracy == 1.0


def test_absent_classes_drag_down_macro_f1_but_not_the_present_average():
    labels = np.array([0, 0, 1, 1])
    metrics = evaluate(one_hot_readout(5, 6), states_for(labels, 5), labels, 5)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == pytest.approx(2 / 5)
    assert metrics.present_macro_f1() == pytest.approx(1.0)
    assert metrics.present_classes() == [0, 1]


def test_empty_evaluation_set_returns_the_sentinel():
    metrics = evaluate(one_hot_readout(3, 6), np.zeros((0, 6)), np.zeros(0, int), 3)
    assert metrics.accuracy == 0.0
    assert metrics.loss == 10.0


def test_loss_is_lower_for_confident_correct_predictions():
    labels = np.array([0, 1])
    states = states_for(labels, 2)
    confident = evaluate(one_hot_readout(2, 6) * 10.0, states, labels, 2).loss
    timid = evaluate(one_hot_readout(2, 6) * 0.1, states, labels, 2).loss
    assert confident < timid


def test_support_counts_the_evaluation_labels():
    labels = np.array([0, 0, 0, 2])
    metrics = evaluate(one_hot_readout(3, 6), states_for(labels, 3), labels, 3)
    assert list(metrics.support) == [3, 0, 1]


def test_softmax_rows_sum_to_one_and_tolerate_large_inputs():
    probs = softmax(np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.isfinite(probs).all()


def test_per_class_accuracy_is_recall_per_class():
    labels = np.array([0, 0, 1, 1])
    states = states_for(labels, 2)
    states[1] = states[2]  # one class-0 sample now looks like class 1
    recall = per_class_accuracy(one_hot_readout(2, 6), states, labels, 2)
    assert recall == pytest.approx([0.5, 1.0])


def test_confusion_matrix_rows_are_actual_classes():
    labels = np.array([0, 0, 1])
    matrix = confusion_matrix(one_hot_readout(2, 6), states_for(labels, 2), labels, 2)
    assert matrix.tolist() == [[2, 0], [0, 1]]
    assert matrix.sum() == labels.shape[0]


def test_average_over_an_empty_class_list_is_zero():
    assert average_over(np.array([1.0, 1.0]), []) == 0.0


def test_mean_metrics_averages_each_field():
    labels = np.array([0, 1])
    states = states_for(labels, 2)
    good = evaluate(one_hot_readout(2, 6), states, labels, 2)
    bad = evaluate(-one_hot_readout(2, 6), states, labels, 2)
    averaged = mean_metrics([good, bad], 2)
    assert averaged.accuracy == pytest.approx((good.accuracy + bad.accuracy) / 2)
    assert mean_metrics([], 2).loss == 10.0
