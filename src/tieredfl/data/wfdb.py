"""Readers for the PhysioNet/CinC Challenge 2021 record format.

Each record is a ``.hea`` text header plus a ``.mat`` file holding the raw ADC samples.
The challenge ships MATLAB level-4 files with a single ``val`` matrix, which is simple
enough to read directly; this avoids a dependency on scipy.io for the one format we
need and lets us stop reading after the first ``max_samples`` columns.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_WHITESPACE = re.compile(r"\s+")

# MOPT encodes (precision, order, text, type). Challenge 2021 files are all int16 with
# MOPT 30, which does not follow the documented encoding, so it is special-cased.
_MOPT_INT16 = 30
_DTYPES = {
    0: ("<f8", 8),
    1: ("<f4", 4),
    2: ("<i4", 4),
    3: ("<i2", 2),
    4: ("<u2", 2),
    5: ("<u1", 1),
}


@dataclass(frozen=True)
class Lead:
    name: str
    baseline: int
    adc_gain: float


@dataclass(frozen=True)
class Header:
    record_id: str
    leads: list[Lead]
    sampling_rate: float
    n_samples: int
    age: float | None
    sex: str | None
    diagnoses: list[str]


@dataclass
class Record:
    """One recording in physical units, ``(n_leads, n_samples)`` millivolts."""

    record_id: str
    signal: np.ndarray
    sampling_rate: float
    diagnoses: list[str]
    age: float | None
    sex: str | None
    source: str = ""

    @property
    def n_leads(self) -> int:
        return self.signal.shape[0]

    @property
    def duration_s(self) -> float:
        return self.signal.shape[1] / self.sampling_rate


def read_header(path: Path) -> Header:
    lines = Path(path).read_text().splitlines()
    fields = _WHITESPACE.split(lines[0].strip())
    record_id, n_leads = fields[0], int(fields[1])
    sampling_rate, n_samples = float(fields[2]), int(fields[3])

    leads = []
    for line in lines[1 : 1 + n_leads]:
        parts = _WHITESPACE.split(line.strip())
        gain = parts[2].split("(")[0]
        leads.append(
            Lead(
                name=parts[-1],
                baseline=_to_int(parts[5]) if len(parts) > 5 else 0,
                adc_gain=_to_float(gain, default=1000.0),
            )
        )

    age: float | None = None
    sex: str | None = None
    diagnoses: list[str] = []
    for line in lines[1 + n_leads :]:
        key, _, value = line.lstrip("#").partition(":")
        key = key.strip().lower()
        if key == "age":
            age = _to_float(value.strip())
        elif key == "sex":
            sex = value.strip() or None
        elif key == "dx":
            diagnoses = [code.strip() for code in value.split(",") if code.strip()]

    return Header(record_id, leads, sampling_rate, n_samples, age, sex, diagnoses)


def read_mat(path: Path, key: str = "val", max_samples: int | None = None) -> np.ndarray:
    """Read the matrix stored in a MATLAB level-4 file.

    Data is column-major, so truncating to ``max_samples`` columns only requires
    reading a prefix of the file. Long records (st_petersburg_incart runs past 460k
    samples per lead) would otherwise dominate load time and memory.
    """
    with open(path, "rb") as handle:
        mopt, n_rows, n_cols, _imag, name_len = struct.unpack("<5i", handle.read(20))
        name = handle.read(name_len).decode("ascii").strip("\x00 ")
        if name != key:
            raise ValueError(f"{path}: expected variable {key!r}, found {name!r}")

        dtype, width = _DTYPES[3] if mopt == _MOPT_INT16 else _DTYPES[mopt % 10]
        cols = n_cols if max_samples is None else min(n_cols, max_samples)
        raw = np.frombuffer(handle.read(n_rows * cols * width), dtype=dtype, count=n_rows * cols)

    return raw.reshape(cols, n_rows).T.astype(np.float64)


def read_record(
    header_path: Path,
    mat_path: Path | None = None,
    max_samples: int = 5000,
    source: str = "",
) -> Record:
    """Read a record and convert ADC counts to millivolts."""
    header_path = Path(header_path)
    mat_path = Path(mat_path) if mat_path else header_path.with_suffix(".mat")

    header = read_header(header_path)
    raw = read_mat(mat_path, max_samples=max_samples)

    baseline = np.array([lead.baseline for lead in header.leads], dtype=np.float64)
    gain = np.array([lead.adc_gain for lead in header.leads], dtype=np.float64)
    gain[gain == 0.0] = 1000.0
    signal = (raw - baseline[:, None]) / gain[:, None]

    return Record(
        record_id=header.record_id,
        signal=signal,
        sampling_rate=header.sampling_rate,
        diagnoses=header.diagnoses,
        age=header.age,
        sex=header.sex,
        source=source,
    )


def _to_int(text: str, default: int = 0) -> int:
    try:
        return int(text)
    except ValueError:
        return default


def _to_float(text: str, default: float | None = None) -> float | None:
    try:
        return float(text)
    except ValueError:
        return default
