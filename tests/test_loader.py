"""Dataset discovery and the data-root layouts it accepts."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tieredfl.data.loader import DEFAULT_SUBDIR, resolve_root

HEADER = """A0001 1 500 4
A0001.mat 16+24 1000.0(0)/mV 16 0 0 0 0 I
# Dx: 426783006
"""


def plant_record(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "A0001.hea").write_text(HEADER)
    payload = struct.pack("<5i", 30, 1, 4, 0, 4) + b"val\x00"
    payload += np.arange(4, dtype="<i2").tobytes()
    (directory / "A0001.mat").write_bytes(payload)


@pytest.mark.parametrize(
    "layout",
    [
        DEFAULT_SUBDIR,
        "files/challenge-2021/1.0.3/training",
        "challenge-2021/1.0.3/training",
        "1.0.3/training",
        "training",
        ".",
    ],
)
def test_resolve_root_accepts_any_level_of_the_download_path(tmp_path, layout):
    training = tmp_path / layout
    plant_record(training / "cpsc_2018")
    assert resolve_root(tmp_path) == (training.resolve() if str(layout) != "." else tmp_path)


def test_resolve_root_returns_the_given_path_when_nothing_is_found(tmp_path):
    missing = tmp_path / "absent"
    assert resolve_root(missing) == missing


def test_resolve_root_follows_a_symlink(tmp_path):
    plant_record(tmp_path / "real" / DEFAULT_SUBDIR / "ptb-xl")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "real" / "physionet.org")
    assert (resolve_root(link) / "ptb-xl").is_dir()


def test_find_headers_requires_a_matching_mat_file(tmp_path):
    from tieredfl.data.loader import find_headers

    training = tmp_path / DEFAULT_SUBDIR / "georgia"
    plant_record(training)
    (training / "orphan.hea").write_text(HEADER)
    assert [p.name for p in find_headers(tmp_path, "georgia")] == ["A0001.hea"]


def test_find_headers_is_empty_for_a_missing_dataset(tmp_path):
    from tieredfl.data.loader import find_headers

    assert find_headers(tmp_path, "ningbo") == []


def test_load_dataset_skips_records_it_cannot_parse(tmp_path):
    from tieredfl.data.loader import load_dataset

    training = tmp_path / DEFAULT_SUBDIR / "ningbo"
    plant_record(training)
    (training / "broken.hea").write_text(HEADER)
    (training / "broken.mat").write_bytes(b"not a mat file")

    records = list(load_dataset(tmp_path, "ningbo"))
    assert [r.record_id for r in records] == ["A0001"]
