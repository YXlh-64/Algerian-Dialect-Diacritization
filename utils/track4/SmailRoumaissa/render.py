from typing import List

from utils.track4.SmailRoumaissa.constants import DIACRITIC_MARKS


def render_sentence(chars: List[str], labels: List[int]) -> str:
    out = []
    for c, l in zip(chars, labels):
        out.append(c)
        if c != " ":
            out.append(DIACRITIC_MARKS.get(int(l), ""))
    return "".join(out)
