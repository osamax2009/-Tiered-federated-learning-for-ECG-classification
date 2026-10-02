"""Device tiers.

A tier bundles together what a class of device can capture (leads, sampling rate) and
how much the server should trust it during aggregation. The IoT tier is the deployment
target; the richer tiers exist to supply it with a usable global model.
"""

from __future__ import annotations

from enum import Enum


class DeviceTier(Enum):
    """Capability classes, ordered from most to least capable."""

    HOSPITAL = ("hospital", 12, 500.0, 1.25)
    WEARABLE = ("wearable", 3, 250.0, 1.00)
    IOT = ("iot", 1, 125.0, 0.90)

    def __init__(self, label: str, leads: int, sampling_rate: float, aggregation_prior: float):
        self.label = label
        self.leads = leads
        self.sampling_rate = sampling_rate
        # Multiplies the client's FedAvg weight. Capable tiers hold cleaner data, so
        # their readouts are given more influence over the shared model.
        self.aggregation_prior = aggregation_prior

    @classmethod
    def from_label(cls, label: str) -> DeviceTier:
        for tier in cls:
            if tier.label == label.lower():
                return tier
        raise ValueError(f"unknown tier {label!r}; expected one of {[t.label for t in cls]}")

    def __str__(self) -> str:
        return self.label
