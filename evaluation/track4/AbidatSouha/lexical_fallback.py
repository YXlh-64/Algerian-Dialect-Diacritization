"""V2 confidence-gated lexical fallback (notebook §16).

Where the model is unsure about a word it has already seen in training, trust
the training lexicon instead. The dev bucket analysis is what motivates this:
on `seen_unambig` words the plain lookup baseline scores 5.06% WER while the
neural model scores 8.49%, so on exactly those words the lexicon is the better
predictor.

The gate fires only when all three hold:

    total occurrences of the surface form in the lexicon  >= min_count
    share held by the majority vocalisation               >= min_majority
    weakest per-letter confidence in the word             <  max_conf

plus a skeleton check that the majority vocalisation has the same number of
letters as the word being overwritten. `max_conf` is a ceiling, not a floor.
"""

from utils.track4.AbidatSouha.constants import DIAC_SET, SHADDA, SHADDA_VOWEL_LABEL, VOWEL_LABEL


def parse_vocalized_word(word):
    "One fully-diacritized word (no spaces) -> list of per-letter labels."
    out, i, n = [], 0, len(word)
    while i < n:
        j = i + 1
        has_shadda, vowel = False, None
        while j < n and word[j] in DIAC_SET:
            if word[j] == SHADDA: has_shadda = True
            else: vowel = word[j]
            j += 1
        if has_shadda and vowel is not None: label = SHADDA_VOWEL_LABEL[vowel]
        elif has_shadda:                     label = 8
        elif vowel is not None:              label = VOWEL_LABEL[vowel]
        else:                                label = 0
        out.append(label); i = j
    return out


def word_spans(chars):
    "Space-delimited words -> list of index-lists into chars/labels/conf."
    words, cur = [], []
    for i, c in enumerate(chars):
        if c == " ":
            if cur: words.append(cur); cur = []
        else:
            cur.append(i)
    if cur: words.append(cur)
    return words


def gated_labels(chars, neural_labels, conf, data, max_conf, min_count, min_majority):
    labels = list(neural_labels)
    for idx in word_spans(chars):
        surface = "".join(chars[i] for i in idx)
        if surface not in data.lex:
            continue
        counts = data.lex[surface]
        total = sum(counts.values())
        maj_word, maj_count = counts.most_common(1)[0]
        maj_share = maj_count / total
        word_conf = min(conf[i] for i in idx)          # weakest-link confidence
        if total >= min_count and maj_share >= min_majority and word_conf < max_conf:
            maj_labels = parse_vocalized_word(maj_word)
            if len(maj_labels) == len(idx):             # skeleton sanity check
                for pos, lab in zip(idx, maj_labels):
                    labels[pos] = lab
    return labels


def apply_fallback(recs, neural_labels, confs, data, fb_cfg):
    """gated_labels over a whole split, using a LexicalFallbackConfig.

    Convenience wrapper; the threshold grid search calls gated_labels directly
    because it varies the three values independently.
    """
    return [gated_labels(r["chars"], lab, conf, data,
                         fb_cfg.max_conf, fb_cfg.min_count, fb_cfg.min_majority)
            for r, lab, conf in zip(recs, neural_labels, confs)]
