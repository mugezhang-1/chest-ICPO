"""MI-CXR TEL answers re-derived from a presence sequence (moved verbatim from scripts/analysis/micxr_alignment.py).

With QBA labels this reproduces 99.7% of MI-CXR's derivable TEL answers (docs/micxr_alignment.md §7), so it is used
to compute the correct answer of a TEL item after its images are reordered (Exp 1 probes).
"""

import re

POS, NEG = "pos", "neg"

FINDING_PATTERNS = [
    r"does (.+?) (?:second |first )?(?:new appearance|disappearance) occur",
    r"shows that (.+?) first disappears",
    r"during which (.+?) first appears",
    r"temporal changes of (.+?) across",
]
TO_OURS = {
    "pleural effusion": "Pleural Effusion", "pulmonary edema": "Edema", "consolidation": "Consolidation",
    "atelectasis": "Atelectasis", "pneumothorax": "Pneumothorax", "cardiomegaly": "Cardiomegaly",
    "lung opacity": "Lung Opacity", "airspace opacity": "Lung Opacity",
}
INTERVAL_LETTERS = "ABCD"  # A: T1->T2 ... D: T4->T5; E: none


def finding_of(question):
    for p in FINDING_PATTERNS:
        m = re.search(p, question)
        if m:
            return m.group(1)
    return None


def transitions(seq, a, b):
    """0-based interval indices i with seq[i] == a and seq[i+1] == b."""
    return [i for i in range(len(seq) - 1) if seq[i] == a and seq[i + 1] == b]


def pair_letter(choices, pair):
    """Letter whose text is 'Ti, Tj' for the given (i, j) 1-based pair, or the 'none' letter if pair is None."""
    for k, v in choices.items():
        m = re.fullmatch(r"\s*T(\d), T(\d)\s*", v)
        if pair is None and not m:
            return k
        if m and pair is not None and (int(m.group(1)), int(m.group(2))) == pair:
            return k
    return None  # the pair is not among the options


def derive(qtype, seq, choices):
    """Our answer letter for a TEL item, 'ambiguous' if the item's premise does not fit our labels."""
    em, rs = transitions(seq, NEG, POS), transitions(seq, POS, NEG)

    def nth(xs, n):
        return INTERVAL_LETTERS[xs[n]] if len(xs) > n else "E"

    if qtype == "single_emergence":
        return "E" if not em else (INTERVAL_LETTERS[em[0]] if len(em) == 1 else "ambiguous")
    if qtype == "single_resolution":
        return "E" if not rs else (INTERVAL_LETTERS[rs[0]] if len(rs) == 1 else "ambiguous")
    if qtype == "multiple_emergence_1":
        return nth(em, 0)
    if qtype == "multiple_emergence_2":
        return nth(em, 1)
    if qtype == "multiple_resolution_1":
        return nth(rs, 0)
    if qtype == "multiple_resolution_2":
        return nth(rs, 1)
    if qtype == "emergence_to_resolution":
        # first study where the finding is present after being absent, then first later study where it is absent
        for e in em:
            later = [r for r in rs if r > e]
            if later:
                return pair_letter(choices, (e + 2, later[0] + 2)) or "ambiguous"
        return pair_letter(choices, None)
    if qtype == "resolution_to_emergence":
        for r in rs:
            later = [e for e in em if e > r]
            if later:
                return pair_letter(choices, (r + 2, later[0] + 2)) or "ambiguous"
        return pair_letter(choices, None)
    raise ValueError(qtype)
