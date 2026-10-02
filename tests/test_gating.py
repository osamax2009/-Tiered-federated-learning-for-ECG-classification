from __future__ import annotations

import numpy as np
import pytest

from tieredfl.federated import gating

N_CLASSES = 3
STATE_SIZE = 4


def readout_for(target: int) -> np.ndarray:
    """A readout that always predicts ``target``."""
    wout = np.zeros((N_CLASSES, STATE_SIZE))
    wout[target, -1] = 1.0
    return wout


def states(n: int) -> np.ndarray:
    block = np.zeros((n, STATE_SIZE))
    block[:, -1] = 1.0
    return block


def test_grid_spans_both_pure_rules():
    assert gating.GRID[0] == 0.0
    assert gating.GRID[-1] == 1.0
    assert len(gating.GRID) == 11


def test_gate_prefers_the_rule_that_serves_the_iot_objective():
    # FedAvg always predicts class 0, selective always predicts class 1, and the IoT
    # split is entirely class 1. Every mixture past the midpoint predicts class 1 and
    # scores identically, so the gate settles on the first of those rather than on the
    # pure selective end; what matters is that it crossed over.
    labels = np.ones(20, dtype=int)
    decision = gating.select(
        fedavg_wout=readout_for(0),
        selective_wout=readout_for(1),
        global_states=states(20),
        global_labels=labels,
        iot_states=states(20),
        iot_labels=labels,
        n_classes=N_CLASSES,
        objective="iot_present_macro_f1",
        tolerance=1.0,
    )
    assert decision.weight > 0.5
    assert decision.selected().iot_present_macro_f1 == pytest.approx(1.0)


def test_the_global_constraint_blocks_an_otherwise_better_mixture():
    # Global split favours class 0, IoT split favours class 1. With no tolerance the
    # gate may not trade away global macro F1, so it stays at FedAvg.
    decision = gating.select(
        fedavg_wout=readout_for(0),
        selective_wout=readout_for(1),
        global_states=states(20),
        global_labels=np.zeros(20, dtype=int),
        iot_states=states(20),
        iot_labels=np.ones(20, dtype=int),
        n_classes=N_CLASSES,
        objective="iot_present_macro_f1",
        tolerance=0.0,
    )
    assert decision.weight == 0.0
    assert decision.feasible >= 1


def test_a_loose_tolerance_lets_the_gate_trade_global_for_iot():
    strict = gating.select(
        readout_for(0),
        readout_for(1),
        states(20),
        np.zeros(20, dtype=int),
        states(20),
        np.ones(20, dtype=int),
        n_classes=N_CLASSES,
        objective="iot_present_macro_f1",
        tolerance=0.0,
    )
    loose = gating.select(
        readout_for(0),
        readout_for(1),
        states(20),
        np.zeros(20, dtype=int),
        states(20),
        np.ones(20, dtype=int),
        n_classes=N_CLASSES,
        objective="iot_present_macro_f1",
        tolerance=1.0,
    )
    assert loose.weight > strict.weight


def test_present_classes_default_to_the_iot_labels():
    decision = gating.select(
        readout_for(0),
        readout_for(1),
        states(10),
        np.array([0, 1, 2] * 3 + [0]),
        states(10),
        np.ones(10, dtype=int),
        n_classes=N_CLASSES,
        objective="iot_present_macro_f1",
        tolerance=1.0,
    )
    # Only class 1 is present in the IoT split, so a readout that nails it scores 1.0.
    assert decision.selected().iot_present_macro_f1 == pytest.approx(1.0)


def test_gate_falls_back_to_the_global_metrics_without_an_iot_split():
    decision = gating.select(
        readout_for(0),
        readout_for(1),
        states(10),
        np.zeros(10, dtype=int),
        None,
        None,
        n_classes=N_CLASSES,
        objective="iot_present_macro_f1",
        tolerance=1.0,
    )
    assert decision.weight == 0.0


def test_every_objective_is_selectable():
    for objective in gating.OBJECTIVES:
        decision = gating.select(
            readout_for(0),
            readout_for(1),
            states(12),
            np.zeros(12, dtype=int),
            states(12),
            np.ones(12, dtype=int),
            n_classes=N_CLASSES,
            objective=objective,
            tolerance=1.0,
        )
        assert decision.weight in set(gating.GRID)


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown gate objective"):
        gating.select(
            readout_for(0),
            readout_for(1),
            states(8),
            np.zeros(8, dtype=int),
            states(8),
            np.ones(8, dtype=int),
            n_classes=N_CLASSES,
            objective="made_up",
            tolerance=1.0,
        )


def test_candidate_table_marks_the_selection_once():
    decision = gating.select(
        readout_for(0),
        readout_for(1),
        states(10),
        np.zeros(10, dtype=int),
        states(10),
        np.ones(10, dtype=int),
        n_classes=N_CLASSES,
        objective="iot_present_macro_f1",
        tolerance=1.0,
    )
    table = gating.candidate_table(decision)
    assert len(table) == len(gating.GRID)
    assert sum(row["selected"] for row in table) == 1
