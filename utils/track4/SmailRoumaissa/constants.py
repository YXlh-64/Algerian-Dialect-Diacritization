SPACE = " "
NUM_CLASSES = 16

DIACRITIC_MARKS = {
    0: "",                    # No Diacritic
    1: "\u064E",              # Fatha
    2: "\u064B",              # Fathatan
    3: "\u064F",              # Damma
    4: "\u064C",              # Dammatan
    5: "\u0650",              # Kasra
    6: "\u064D",              # Kasratan
    7: "\u0652",              # Sukoon
    8: "\u0651",              # Shadda
    9: "\u0651\u064E",        # Shadda + Fatha
    10: "\u0651\u064B",       # Shadda + Fathatan
    11: "\u0651\u064F",       # Shadda + Damma
    12: "\u0651\u064C",       # Shadda + Dammatan
    13: "\u0651\u0650",       # Shadda + Kasra
    14: "\u0651\u064D",       # Shadda + Kasratan
    15: "\u0651\u0652",       # Shadda + Sukoon
}

CLASS_NAMES = [
    "None", "Fatha", "Fathatan", "Damma", "Dammatan", "Kasra", "Kasratan", "Sukoon",
    "Shadda", "Shadda+Fatha", "Shadda+Fathatan", "Shadda+Damma", "Shadda+Dammatan",
    "Shadda+Kasra", "Shadda+Kasratan", "Shadda+Sukoon",
]
