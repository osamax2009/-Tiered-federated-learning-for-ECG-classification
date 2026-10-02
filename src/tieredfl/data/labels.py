"""SNOMED-CT label handling.

Records carry a set of diagnosis codes, but the experiment is single-label: the
``top_n`` most frequent codes across the federation become the classes, and a record is
assigned to the first of those codes it carries. Codes are ranked on the loaded data
rather than fixed in advance, so the class set follows whatever subset is in use.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from .clinical import Sample

# Names for the codes that come up in the challenge subsets used here. Only used for
# display; unmapped codes fall back to the bare code.
DIAGNOSIS_NAMES: dict[str, str] = {
    "10370003": "pacing rhythm",
    "111975006": "prolonged QT interval",
    "164865005": "myocardial infarction",
    "164889003": "atrial fibrillation",
    "164890007": "atrial flutter",
    "164909002": "left bundle branch block",
    "164917005": "abnormal Q wave",
    "164934002": "abnormal T wave",
    "164947007": "prolonged PR interval",
    "17338001": "ventricular premature beats",
    "251146004": "low QRS voltages",
    "270492004": "first degree AV block",
    "284470004": "premature atrial contraction",
    "39732003": "left axis deviation",
    "426177001": "sinus bradycardia",
    "426627000": "bradycardia",
    "426783006": "sinus rhythm",
    "427084000": "sinus tachycardia",
    "427172004": "premature ventricular contraction",
    "427393009": "sinus arrhythmia",
    "445118002": "left anterior fascicular block",
    "47665007": "right axis deviation",
    "59118001": "right bundle branch block",
    "59931005": "T wave inversion",
    "63593006": "supraventricular premature beats",
    "698252002": "nonspecific intraventricular conduction delay",
    "713426002": "incomplete right bundle branch block",
    "713427006": "complete right bundle branch block",
}


def describe(code: str) -> str:
    name = DIAGNOSIS_NAMES.get(code)
    return f"{code} ({name})" if name else code


def count_codes(samples: Iterable[Sample]) -> Counter:
    counts: Counter = Counter()
    for sample in samples:
        counts.update(sample.labels)
    return counts


def top_codes(samples: Iterable[Sample], top_n: int) -> list[str]:
    """The ``top_n`` most frequent codes, most frequent first.

    Ties are broken by code so the class ordering is stable across runs.
    """
    counts = count_codes(samples)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [code for code, _ in ranked[:top_n]]


def label_index(labels: Sequence[str], codes: Sequence[str]) -> int | None:
    """Index of the first class code present in ``labels``, or ``None``.

    Records carrying several of the selected codes resolve to the most frequent one,
    since ``codes`` is ordered by frequency.
    """
    for index, code in enumerate(codes):
        if code in labels:
            return index
    return None
