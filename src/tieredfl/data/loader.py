"""Discovery and loading of challenge records from disk.

The full training set is ~17 GB, so records are read lazily and only up to the number a
run actually needs. Header discovery is sorted rather than filesystem-ordered, which is
what makes a seeded sample reproducible across machines.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterator, Sequence
from pathlib import Path

from .wfdb import Record, read_record

log = logging.getLogger(__name__)

DATASETS = (
    "ptb-xl",
    "chapman_shaoxing",
    "georgia",
    "ningbo",
    "cpsc_2018",
    "cpsc_2018_extra",
    "st_petersburg_incart",
)

DEFAULT_SUBDIR = Path("physionet.org/files/challenge-2021/1.0.3/training")


def resolve_root(root: Path) -> Path:
    """Accept either the download root or the training directory itself."""
    root = Path(root).expanduser()
    nested = root / DEFAULT_SUBDIR
    return nested if nested.is_dir() else root


def find_headers(root: Path, dataset: str) -> list[Path]:
    directory = resolve_root(root) / dataset
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*.hea") if path.with_suffix(".mat").exists())


def load_dataset(
    root: Path,
    dataset: str,
    limit: int | None = None,
    max_samples: int = 5000,
) -> Iterator[Record]:
    """Yield records from one dataset, skipping any that fail to parse."""
    headers = find_headers(root, dataset)
    if not headers:
        log.warning("no records found for %s under %s", dataset, resolve_root(root))
        return

    for path in headers[:limit]:
        try:
            yield read_record(path, max_samples=max_samples, source=dataset)
        except (OSError, ValueError, KeyError, IndexError, struct.error) as exc:
            log.debug("skipping %s: %s", path.name, exc)


def load_by_dataset(
    root: Path,
    datasets: Sequence[str],
    limits: dict[str, int] | None = None,
    max_samples: int = 5000,
) -> dict[str, list[Record]]:
    """Eagerly load several datasets. Prefer :func:`load_dataset` for large limits."""
    limits = limits or {}
    out: dict[str, list[Record]] = {}
    for dataset in datasets:
        records = list(load_dataset(root, dataset, limits.get(dataset), max_samples))
        log.info("%s: loaded %d records", dataset, len(records))
        out[dataset] = records
    return out
