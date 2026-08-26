
import os
import sys
import unicodedata

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.track1.soundous.diacritics import DIACRITIC_UNICODE, DIACRITICS, SHADDA_COMBOS, SHADDA

DIACRITIC_CHARS = set(DIACRITICS) | {SHADDA}


def _parse_line_reference(diacritized_line):
    """Copied verbatim from the organizers' make_submission.py -- kept in-test (not imported) so
    this test still works even if make_submission.py isn't present in this environment; the real
    integration check is that this function and utils.track1.soundous.diacritics were built from the
    same source."""
    text = unicodedata.normalize("NFC", diacritized_line.strip())
    chars, labels = [], []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in DIACRITIC_CHARS:
            i += 1
            continue
        j = i + 1
        label = 0
        has_shadda = False
        vowel = None
        while j < n and text[j] in DIACRITIC_CHARS:
            if text[j] == SHADDA:
                has_shadda = True
            else:
                vowel = text[j]
            j += 1
        if has_shadda and vowel is not None:
            label = SHADDA_COMBOS.get(vowel, 8)
        elif has_shadda:
            label = 8
        elif vowel is not None:
            label = DIACRITICS.get(vowel, 0)
        chars.append(ch)
        labels.append(label)
        i = j
    return chars, labels


def test_all_16_classes_round_trip():
    for class_id in range(16):
        text = "\u0628" + DIACRITIC_UNICODE[class_id]  # arbitrary letter (baa) + this class's mark(s)
        chars, labels = _parse_line_reference(text)
        assert labels == [class_id], (
            f"class {class_id} did not round-trip: wrote {DIACRITIC_UNICODE[class_id]!r}, "
            f"parsed back as {labels}"
        )


def test_no_diacritic_class_zero_is_empty_string():
    assert DIACRITIC_UNICODE[0] == ""


def test_every_class_id_0_to_15_has_a_mapping():
    assert set(DIACRITIC_UNICODE.keys()) == set(range(16))
