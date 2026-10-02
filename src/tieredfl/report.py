"""Result serialisation and the console summary.

Every run writes a JSON file holding the config, per-round history and final figures, so
tables and plots are rebuilt from the artefact rather than parsed out of a log. The text
summary is a convenience for reading a run at the terminal.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import RunConfig, to_dict
from .data.labels import describe
from .evaluation.ablation import BaselineScore
from .evaluation.metrics import Metrics, average_over, confusion_matrix
from .experiment import Prepared, RunResult
from .federated.tiers import DeviceTier


def metrics_to_dict(metrics: Metrics, present_classes: Sequence[int]) -> dict[str, object]:
    return {
        "accuracy": metrics.accuracy,
        "loss": metrics.loss,
        "macro_f1": metrics.macro_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "present_macro_f1": average_over(metrics.per_class_f1, present_classes),
        "per_class_recall": [float(v) for v in metrics.per_class_recall],
        "per_class_f1": [float(v) for v in metrics.per_class_f1],
        "support": [int(v) for v in metrics.support],
    }


def run_to_dict(result: RunResult, window: int, n_classes: int, present: Sequence[int]) -> dict:
    final_global, final_tiers = result.final_metrics(window, n_classes)
    return {
        "label": result.label,
        "final": {
            "global": metrics_to_dict(final_global, present),
            "tiers": {
                tier.label: metrics_to_dict(metrics, present)
                for tier, metrics in final_tiers.items()
            },
        },
        "rounds": [
            {
                "round": r.index,
                "gate_weight": r.gate_weight,
                "duration_s": r.duration_s,
                "global": metrics_to_dict(r.global_metrics, present),
                "tiers": {
                    tier.label: metrics_to_dict(metrics, present)
                    for tier, metrics in r.tier_metrics.items()
                },
            }
            for r in result.rounds
        ],
    }


def write_results(
    config: RunConfig,
    prepared: Prepared,
    runs: Sequence[RunResult],
    baselines: Sequence[BaselineScore],
    tier_head: np.ndarray | None,
    duration_s: float,
) -> Path:
    """Write the run artefact and return its path."""
    present = prepared.evaluation.iot_present_classes()
    window = config.federation.final_round_window
    n_classes = config.federation.n_classes

    iot_states, iot_labels = prepared.evaluation.tier(DeviceTier.IOT)
    payload = {
        "name": config.name,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": duration_s,
        "config": to_dict(config),
        "classes": [{"index": i, "code": code} for i, code in enumerate(prepared.codes)],
        "iot_present_classes": present,
        "clients": [
            {
                "id": client_id,
                "tier": prepared.tiers[client_id].label,
                "n_train": features.n_train,
                "n_validation": features.n_val,
                "train_class_counts": [int(v) for v in features.class_counts(n_classes)],
            }
            for client_id, features in prepared.features.items()
        ],
        "runs": [run_to_dict(r, window, n_classes, present) for r in runs],
        "iot_baselines": [
            {
                "name": b.name,
                "accuracy": b.accuracy,
                "present_macro_f1": b.present_macro_f1,
                "balanced_accuracy": b.balanced_accuracy,
                **({"detail": b.detail} if b.detail else {}),
            }
            for b in baselines
        ],
    }

    if tier_head is not None and iot_states.shape[0] > 0:
        payload["iot_confusion_matrix"] = confusion_matrix(
            tier_head, iot_states, iot_labels, n_classes
        ).tolist()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(config.output_dir) / f"{config.name}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return path


def summarise(
    config: RunConfig,
    prepared: Prepared,
    gated: RunResult,
    baseline: RunResult,
    baselines: Sequence[BaselineScore],
) -> str:
    """A console summary comparing the gated arm against the FedAvg baseline."""
    present = prepared.evaluation.iot_present_classes()
    window = config.federation.final_round_window
    n_classes = config.federation.n_classes

    gated_global, gated_tiers = gated.final_metrics(window, n_classes)
    base_global, base_tiers = baseline.final_metrics(window, n_classes)

    lines: list[str] = []
    lines.append(f"{config.name}: gated selective aggregation vs weighted FedAvg")
    lines.append(
        f"gate objective {config.federation.gate_objective}, "
        f"tolerance {config.federation.gate_tolerance}, "
        f"{len(gated.rounds)} rounds, final figures averaged over last {window}"
    )
    lines.append("")
    lines.append(
        "classes: " + ", ".join(f"[{i}] {describe(c)}" for i, c in enumerate(prepared.codes))
    )
    lines.append("iot present classes: " + ", ".join(f"[{i}]" for i in present))
    lines.append("")

    rows = [("metric", "gated", "fedavg", "delta")]
    rows.extend(_comparison_rows("global", gated_global, base_global, present))
    for tier in DeviceTier:
        if tier in gated_tiers and tier in base_tiers:
            rows.extend(_comparison_rows(tier.label, gated_tiers[tier], base_tiers[tier], present))
    lines.extend(_format_table(rows))

    if baselines:
        lines.append("")
        lines.append("iot reference points (identical validation split)")
        width = max(len(b.name) for b in baselines)
        for score in baselines:
            parts = [f"accuracy={score.accuracy:.4f}"]
            if score.present_macro_f1 is not None:
                parts.append(f"present_macro_f1={score.present_macro_f1:.4f}")
            if score.balanced_accuracy is not None:
                parts.append(f"balanced_accuracy={score.balanced_accuracy:.4f}")
            lines.append(f"  {score.name.ljust(width)}  {'  '.join(parts)}")

    weights = [r.gate_weight for r in gated.rounds]
    if weights:
        lines.append("")
        lines.append(
            f"gate weight: mean={np.mean(weights):.2f} "
            f"min={min(weights):.1f} max={max(weights):.1f} "
            "(0 = pure fedavg, 1 = pure selective)"
        )
    return "\n".join(lines)


def _comparison_rows(
    scope: str,
    gated: Metrics,
    baseline: Metrics,
    present: Sequence[int],
) -> list[tuple]:
    entries = [
        ("accuracy", gated.accuracy, baseline.accuracy),
        ("macro_f1", gated.macro_f1, baseline.macro_f1),
        ("balanced_accuracy", gated.balanced_accuracy, baseline.balanced_accuracy),
    ]
    if scope == "iot":
        entries.insert(
            2,
            (
                "present_macro_f1",
                average_over(gated.per_class_f1, present),
                average_over(baseline.per_class_f1, present),
            ),
        )
    return [(f"{scope} {name}", f"{a:.4f}", f"{b:.4f}", f"{a - b:+.4f}") for name, a, b in entries]


def _format_table(rows: Sequence[tuple]) -> list[str]:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out = []
    for position, row in enumerate(rows):
        cells = [str(cell).ljust(widths[i]) for i, cell in enumerate(row)]
        out.append("  ".join(cells).rstrip())
        if position == 0:
            out.append("  ".join("-" * w for w in widths))
    return out
