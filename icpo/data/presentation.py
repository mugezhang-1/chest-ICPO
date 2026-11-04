"""Answers for an intervened timeline (pipeline step 7, roadmap §2.2).

A presentation is the list of slots the model sees, T1..Tm. Each slot shows an original study of the window
(by 0-based index) or DONOR, a study from another patient. Given the window's presence states and its interval
changes (step 6 rules), the change between two consecutive slots is derived, never guessed:

    same study twice          neg -> "absent", pos -> "unchanged"
    original i -> i+1         the window's change for that interval
    original i+1 -> i         the opposite change (new <-> resolved, worsened <-> improved, unchanged stays)
    i -> j skipping studies   presence flips give new / resolved / absent; pos -> pos composes the changes in
                              between (all intermediate studies pos, no opposite directions), else unk
    any DONOR slot            "na" (not assessable)

The family answers are then recomputed on the presented sequence with the same functions as step 6, so a base item
and all its variants are scored by one set of rules.
"""

from icpo.data.mint import CHANGES, TRAJ_TEXT, UP, DOWN, tel_answer, trajectory

DONOR = -1
OPPOSITE = {"new": "resolved", "resolved": "new", "worsened": "improved", "improved": "worsened",
            "unchanged": "unchanged", "absent": "absent", "unk": "unk"}


def compose(states: list[str], changes: list[str], i: int, j: int) -> str:
    """Change from original study i to a later original study j (i < j)."""
    a, b = states[i], states[j]
    if "unk" in (a, b):
        return "unk"
    if (a, b) == ("neg", "pos"):
        return "new"
    if (a, b) == ("pos", "neg"):
        return "resolved"
    if (a, b) == ("neg", "neg"):
        return "absent"
    # pos -> pos: every study in between must be pos and every step known
    if any(s != "pos" for s in states[i:j + 1]):
        return "unk"
    steps = changes[i:j]
    if any(c not in ("worsened", "improved", "unchanged") for c in steps):
        return "unk"
    up, down = any(c == "worsened" for c in steps), any(c == "improved" for c in steps)
    if up and down:
        return "unk"
    return "worsened" if up else "improved" if down else "unchanged"


def slot_change(states: list[str], changes: list[str], a: int, b: int) -> str:
    if DONOR in (a, b):
        return "na"
    if a == b:
        return {"neg": "absent", "pos": "unchanged"}.get(states[a], "unk")
    if b > a:
        return changes[a] if b == a + 1 else compose(states, changes, a, b)
    return OPPOSITE[changes[b] if a == b + 1 else compose(states, changes, b, a)]


def presented(states: list[str], changes: list[str], slots: list[int]) -> tuple[list[str], list[str]]:
    """(states, changes) as seen through the slots; donor slots have state 'na'."""
    st = ["na" if s == DONOR else states[s] for s in slots]
    ch = [slot_change(states, changes, slots[k], slots[k + 1]) for k in range(len(slots) - 1)]
    return st, ch


CANNOT = "Cannot be assessed from these images"
CHANGE_TEXT = {"new": "Newly appeared", "worsened": "Worsened", "improved": "Improved",
               "unchanged": "Unchanged", "resolved": "Resolved"}


def answer(item: dict, slots: list[int]) -> str | None:
    """Expected answer TEXT of `item` (a step-6 pool row) when its window is shown through `slots`.

    The question text is unchanged; T-labels refer to slot positions. Returns None when the answer is not
    determined by the labels (the variant is then not usable). Any donor slot the question depends on gives CANNOT:
    every slot for TEL / GTS, the two interval slots for ICR, the asked slot for EXIST.
    """
    st, ch = presented(list(item["states"]), list(item["changes"]), slots)
    fam, sub = item["family"], item["subtype"]
    if fam == "ICR":
        k = int(sub[len("interval"):])
        if k + 1 >= len(slots):
            return None
        c = ch[k]
        return CANNOT if c == "na" else CHANGE_TEXT.get(c)
    if fam == "EXIST":
        k = int(sub[len("study"):])
        if k >= len(slots):
            return None
        return CANNOT if st[k] == "na" else {"pos": "Yes", "neg": "No"}.get(st[k])
    if "na" in st:
        return CANNOT
    if fam == "TEL":
        k = tel_answer(st, sub)
        if k is None:
            return None
        # the "none" option text names the finding only, so it is the same for any number of slots
        return f"T{k + 1} → T{k + 2}" if k >= 0 else item["options"][item["n"] - 1]
    if fam == "GTS":
        t = trajectory(ch)
        return TRAJ_TEXT[t] if t else None
    raise ValueError(fam)
