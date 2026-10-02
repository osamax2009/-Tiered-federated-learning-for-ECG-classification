"""Stable seed derivation.

``hash()`` on strings is salted per interpreter process, so anything derived from it
changes between runs. These helpers derive seeds from a content hash instead, which is
what lets a run be reproduced from the config alone.
"""

from __future__ import annotations

import hashlib

import numpy as np

_MASK = 2**32


def stable_hash(*parts: str | int) -> int:
    """A 32-bit hash of the given parts, identical across processes and platforms."""
    digest = hashlib.blake2b(
        "\x1f".join(str(part) for part in parts).encode("utf-8"), digest_size=8
    )
    return int.from_bytes(digest.digest(), "big") % _MASK


def generator(seed: int, *parts: str | int) -> np.random.Generator:
    """A generator for one named stream under a run's master seed."""
    return np.random.default_rng([seed, stable_hash(*parts)])
