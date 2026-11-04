"""Exp 1 part B: label-free and anchored probes on MI-CXR (roadmap §5, Exp 1).

A probe turns an MI-CXR item into a presentation: which image fills each slot T1..T5, the option order, the prompt
wording, and whether images are shown. The labels T1..T5 always stay in place, so a reordering of images is a change
of the timeline the model sees. Every choice that varies per item is derived from a hash of the q_id, so probes are
identical across runs.

structural : reverse, swap, constant        (the correct answer may change)
null       : options, reword                (the correct answer cannot change)
modality   : no_image
"""

import hashlib
from dataclasses import dataclass, field

SLOTS = ("T1", "T2", "T3", "T4", "T5")
PROBES = ("orig", "reverse", "swap", "constant", "options", "reword", "no_image")
REWORD_INTRO = ("The five chest radiographs below were acquired from one patient at successive visits and are "
                "shown oldest first, as T1, T2, T3, T4 and T5.")
REWORD_INSTRUCTION = "Reply with a single letter from A to E that identifies the correct option."


@dataclass(frozen=True)
class Presentation:
    order: tuple                 # source slot shown at T1..T5, e.g. ("T5", "T4", "T3", "T2", "T1")
    letters: dict = field(default_factory=lambda: {k: k for k in "ABCDE"})   # presented letter -> original letter
    reword: bool = False
    show_images: bool = True


def _h(q_id, salt: str) -> int:
    return int(hashlib.sha256(f"{q_id}|{salt}".encode()).hexdigest(), 16)


def present(item: dict, probe: str) -> Presentation:
    qid = item["q_id"]
    if probe == "orig":
        return Presentation(SLOTS)
    if probe == "reverse":
        return Presentation(SLOTS[::-1])
    if probe == "swap":
        i = _h(qid, "swap") % 4
        o = list(SLOTS)
        o[i], o[i + 1] = o[i + 1], o[i]
        return Presentation(tuple(o))
    if probe == "constant":
        return Presentation(("T1",) * 5)
    if probe == "options":
        perm = sorted("ABCD", key=lambda k: _h(qid, f"opt|{k}"))       # E (none) stays last
        if perm == list("ABCD"):                                          # always a real permutation
            perm = perm[1:] + perm[:1]
        return Presentation(SLOTS, letters={**dict(zip("ABCD", perm)), "E": "E"})
    if probe == "reword":
        return Presentation(SLOTS, reword=True)
    if probe == "no_image":
        return Presentation(SLOTS, show_images=False)
    raise ValueError(probe)


def presented_choices(item: dict, p: Presentation) -> dict:
    """Options as shown: presented letter -> text of the original option it maps to."""
    return {new: item["choices"][orig] for new, orig in p.letters.items()}


def permuted_states(states: list, order: tuple) -> list:
    """State sequence the model sees when slot k shows source study order[k]."""
    idx = {s: i for i, s in enumerate(SLOTS)}
    return [states[idx[s]] for s in order]
