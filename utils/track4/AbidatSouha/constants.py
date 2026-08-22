"""Label scheme, diacritic marks and linguistic character classes.

Every value below is copied verbatim from DZ_Vocalisation_Transformer.ipynb;
the source section is noted on each block.
"""

# ---- §1 Setup and configuration --------------------------------------------
NUM_CLASSES = 16                            # notebook: N_CLASSES
PAD = 0
LIVE = [0, 1, 3, 5, 7, 9, 11, 13, 15]       # classes with non-negligible support

# ---- §2 Data + morphological features ---------------------------------------
SUN   = set("تثدذرزسشصضطظلن")
MATER = set("اويى")
FEAT_SIZES = (5, 5, 5, 7, 3, 3)   # pos_in_word, dist_start, dist_end, wlen, mater, sun

# ---- §14 Submission ----------------------------------------------------------
FATHA, FATHATAN = chr(0x064E), chr(0x064B)
DAMMA, DAMMATAN = chr(0x064F), chr(0x064C)
KASRA, KASRATAN = chr(0x0650), chr(0x064D)
SUKOON, SHADDA  = chr(0x0652), chr(0x0651)

LABEL_TO_MARKS = {
    0: "",            1: FATHA,           2: FATHATAN,        3: DAMMA,
    4: DAMMATAN,      5: KASRA,           6: KASRATAN,        7: SUKOON,
    8: SHADDA,        9: SHADDA + FATHA,  10: SHADDA + FATHATAN,
    11: SHADDA + DAMMA, 12: SHADDA + DAMMATAN, 13: SHADDA + KASRA,
    14: SHADDA + KASRATAN, 15: SHADDA + SUKOON,
}

# ---- §16 V2 — confidence-gated lexical fallback ------------------------------
# Reverse tables, for parsing an already-vocalized word back into labels.
DIAC_SET = {FATHA, FATHATAN, DAMMA, DAMMATAN, KASRA, KASRATAN, SUKOON, SHADDA}
VOWEL_LABEL        = {FATHA: 1, FATHATAN: 2, DAMMA: 3, DAMMATAN: 4,
                      KASRA: 5, KASRATAN: 6, SUKOON: 7}
SHADDA_VOWEL_LABEL = {FATHA: 9, FATHATAN: 10, DAMMA: 11, DAMMATAN: 12,
                      KASRA: 13, KASRATAN: 14, SUKOON: 15}
