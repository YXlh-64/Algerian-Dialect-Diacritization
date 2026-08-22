"""Interleave predicted diacritics back into the raw text (notebook §15)."""

from utils.track4.souha.constants import LABEL_TO_MARKS


def render(chars, labels):
    return "".join(c + ("" if c == " " else LABEL_TO_MARKS[l])
                   for c, l in zip(chars, labels))
