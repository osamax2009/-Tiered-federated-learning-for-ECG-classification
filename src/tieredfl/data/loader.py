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

# A `wget -r` download nests the records several levels down. Depending on where someone
# points `data_root` — or symlinks it — any suffix of that path may already be included,
# so each prefix is tried in turn.
_LAYOUTS = (
    DEFAULT_SUBDIR,
    Path("files/challenge-2021/1.0.3/training"),
    Path("challenge-2021/1.0.3/training"),
    Path("1.0.3/training"),
    Path("training"),
    Path("."),
)


def resolve_root(root: Path) -> Path:
    """Find the directory holding the dataset folders.

    Accepts the download root, the training directory, or any level between them.
    Returns the first candidate that actually contains a known dataset; failing that,
    the path as given, so callers report a missing root rather than a wrong one.
    """
    root = Path(root).expanduser()
    for layout in _LAYOUTS:
        candidate = (root / layout).resolve() if layout != Path(".") else root
        if any((candidate / dataset).is_dir() for dataset in DATASETS):
            return candidate
    return root


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
