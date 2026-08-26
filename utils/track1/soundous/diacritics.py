
import unicodedata

FATHA, FATHATAN = "\u064E", "\u064B"
DAMMA, DAMMATAN = "\u064F", "\u064C"
KASRA, KASRATAN = "\u0650", "\u064D"
SUKOON = "\u0652"
SHADDA = "\u0651"

# matches make_submission.py's DIACRITICS dict exactly
DIACRITICS = {
    FATHATAN: 2, DAMMATAN: 4, KASRATAN: 6,
    FATHA: 1, DAMMA: 3, KASRA: 5,
    SUKOON: 7, SHADDA: 8,
}
# matches make_submission.py's SHADDA_COMBOS dict exactly
SHADDA_COMBOS = {
    FATHA: 9, FATHATAN: 10, DAMMA: 11, DAMMATAN: 12,
    KASRA: 13, KASRATAN: 14, SUKOON: 15,
}
DIACRITIC_CHARS = set(DIACRITICS) | {SHADDA}

# class id -> Unicode diacritic mark(s) to append after a letter (inverse of the above).
DIACRITIC_UNICODE = {0: ""}
for _mark, _cid in DIACRITICS.items():
    if _cid != 8:
        DIACRITIC_UNICODE[_cid] = _mark
DIACRITIC_UNICODE[8] = SHADDA
for _mark, _cid in SHADDA_COMBOS.items():
    DIACRITIC_UNICODE[_cid] = SHADDA + _mark
assert set(DIACRITIC_UNICODE.keys()) == set(range(16))


def reconstruct_vocalized(chars, label_ids) -> str:
    """class ids -> a fully vocalized string, using the authoritative DIACRITIC_UNICODE map."""
    return "".join(ch + DIACRITIC_UNICODE.get(lid, "") for ch, lid in zip(chars, label_ids))


def clean_and_tokenize(sentence: str, p1_module=None):
    """Character-level tokenization mirroring P1's contract: explode into Arabic letters + spaces.

    If P1's actual cleaning module (NFC normalize, tatweel strip, alif normalization) is importable,
    pass it as `p1_module` (must expose clean_sentence/tokenize_sentence) to avoid any train/inference
    drift. Falls back to plain NFC-normalize + character split otherwise.
    """
    if p1_module is not None:
        cleaned = p1_module.clean_sentence(sentence)
        return p1_module.tokenize_sentence(cleaned)
    cleaned = unicodedata.normalize("NFC", sentence.strip())
    return list(cleaned)


MAX_CHUNK_LEN = 300  # keep in sync with P1's long-sentence split threshold


def chunk_chars(chars, max_len: int = MAX_CHUNK_LEN):
    if len(chars) <= max_len:
        return [chars]
    chunks, cur = [], []
    for ch in chars:
        cur.append(ch)
        if len(cur) >= max_len and ch == " ":
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks
