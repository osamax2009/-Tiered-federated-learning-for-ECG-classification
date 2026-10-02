"""Per-tier personalised readout.

The shared model is fitted across all tiers, which leaves the IoT tier evaluated under
a readout shaped mostly by data it could never have recorded. A tier head fixes that
without leaving the federated contract:

1. each client in the tier fits a readout on its own states only, with no warm start
   toward the global model and no access to any other client's data;
2. each client transmits that readout;
3. the server averages the transmitted readouts, weighted by sample count.

Step 3 receives matrices and sample counts, nothing else. The alternative — pooling
states server-side and running one solve — gives a similar readout but moves
patient-derived representations off the device, so it is not used.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from ..model.readout import fit_readout
from .aggregate import blend, weighted_average
from .features import ClientFeatures
from .tiers import DeviceTier

log = logging.getLogger(__name__)


def fit_local_head(
    features: ClientFeatures,
    n_classes: int,
    ridge: float,
) -> np.ndarray | None:
    """Fit a readout on one client's own states. Runs on the client."""
    if features.n_train < 2:
        return None
    return fit_readout(features.train_states, features.train_labels, n_classes, ridge)


def aggregate_heads(
    heads: dict[str, np.ndarray],
    sample_counts: dict[str, int],
) -> np.ndarray | None:
    """Average transmitted readouts by sample count. Runs on the server."""
    return weighted_average(heads, {k: float(v) for k, v in sample_counts.items()})


def fit_tier_head(
    features: dict[str, ClientFeatures],
    tiers: dict[str, DeviceTier],
    tier: DeviceTier,
    n_classes: int,
    ridge: float,
) -> np.ndarray | None:
    """Run the three steps for one tier, keeping the client/server boundary explicit."""
    members = sorted(cid for cid, t in tiers.items() if t is tier and cid in features)

    heads: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for client_id in members:
        head = fit_local_head(features[client_id], n_classes, ridge)
        if head is None:
            log.info("%s: too few states for a local head; excluded", client_id)
            continue
        heads[client_id] = head
        counts[client_id] = features[client_id].n_train

    if not heads:
        log.warning("no %s client could fit a local head", tier)
        return None

    log.info("aggregating %d local %s heads (weights only)", len(heads), tier)
    return aggregate_heads(heads, counts)


def evaluation_head(
    global_wout: np.ndarray,
    tier_head: np.ndarray | None,
    personal_weight: float,
) -> np.ndarray:
    """The readout a tier is evaluated under.

    ``personal_weight`` interpolates between the shared model (0.0) and the tier head
    (1.0); intermediate values are the ablation axis for how much personalisation the
    tier actually needs.
    """
    if tier_head is None:
        return global_wout
    if personal_weight >= 1.0:
        return tier_head
    if personal_weight <= 0.0:
        return global_wout
    return blend(global_wout, tier_head, personal_weight)


def local_only_heads(
    features: dict[str, ClientFeatures],
    tiers: dict[str, DeviceTier],
    tier: DeviceTier,
    n_classes: int,
    ridge: float,
) -> dict[str, np.ndarray]:
    """Each client's own head, unaggregated.

    Used by the ablation to test whether federating the tier heads beats keeping them
    local: if the averaged head matches the mean of these, federation bought nothing.
    """
    members = sorted(cid for cid, t in tiers.items() if t is tier and cid in features)
    heads = {}
    for client_id in members:
        head = fit_local_head(features[client_id], n_classes, ridge)
        if head is not None:
            heads[client_id] = head
    return heads


def payload_bytes(wout_shape: Sequence[int]) -> int:
    """Transmission size of one readout, in bytes."""
    return int(np.prod(wout_shape)) * 8
