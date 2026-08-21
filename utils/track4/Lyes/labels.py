"""Exact 16-class Algerian Arabic diacritic mapping."""

from typing import Iterable, List, Sequence, Tuple


NUM_LABELS = 16
NUM_BASE_LABELS = 8
IGNORE_INDEX = -100

LABEL_NAMES: Tuple[str, ...] = (
    "No Diacritic",
    "Fatha",
    "Fathatan",
    "Damma",
    "Dammatan",
    "Kasra",
    "Kasratan",
    "Sukoon",
    "Shadda",
    "Shadda+Fatha",
    "Shadda+Fathatan",
    "Shadda+Damma",
    "Shadda+Dammatan",
    "Shadda+Kasra",
    "Shadda+Kasratan",
    "Shadda+Sukoon",
)

BASE_MARKS: Tuple[str, ...] = (
    "",
    "\u064e",  # Fatha
    "\u064b",  # Fathatan
    "\u064f",  # Damma
    "\u064c",  # Dammatan
    "\u0650",  # Kasra
    "\u064d",  # Kasratan
    "\u0652",  # Sukoon
)
SHADDA = "\u0651"
LABEL_MARKS: Tuple[str, ...] = BASE_MARKS + tuple(
    SHADDA + mark for mark in BASE_MARKS
)


def split_label(label: int) -> Tuple[int, int]:
    """Return ``(base_diacritic, has_shadda)`` for one competition label."""
    if not 0 <= label < NUM_LABELS:
        raise ValueError("label must be in [0, 15], got {}".format(label))
    return label % NUM_BASE_LABELS, label // NUM_BASE_LABELS


def combine_label(base_diacritic: int, has_shadda: int) -> int:
    """Combine an 8-class base mark and binary shadda flag."""
    if not 0 <= base_diacritic < NUM_BASE_LABELS:
        raise ValueError(
            "base_diacritic must be in [0, 7], got {}".format(base_diacritic)
        )
    if has_shadda not in (0, 1):
        raise ValueError("has_shadda must be 0 or 1, got {}".format(has_shadda))
    return base_diacritic + NUM_BASE_LABELS * has_shadda


def labels_to_components(labels: Iterable[int]) -> Tuple[List[int], List[int]]:
    """Vector-free deterministic label decomposition used by tests and I/O."""
    base: List[int] = []
    shadda: List[int] = []
    for label in labels:
        base_label, shadda_label = split_label(label)
        base.append(base_label)
        shadda.append(shadda_label)
    return base, shadda


def apply_diacritics(chars: Sequence[str], labels: Sequence[int]) -> str:
    """Create a fully vocalized sentence while preserving its exact skeleton."""
    if len(chars) != len(labels):
        raise ValueError(
            "chars/labels length mismatch: {} != {}".format(len(chars), len(labels))
        )

    output: List[str] = []
    for char, label in zip(chars, labels):
        if len(char) != 1:
            raise ValueError("each character must be one Unicode code point")
        if char == " ":
            if label != 0:
                raise ValueError("space positions must have label 0")
            output.append(char)
            continue
        if not 0 <= label < NUM_LABELS:
            raise ValueError("label must be in [0, 15], got {}".format(label))
        output.append(char)
        output.append(LABEL_MARKS[label])
    return "".join(output)
