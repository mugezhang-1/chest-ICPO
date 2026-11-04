"""Pipeline step 6 (roadmap §3.2.6): deterministic question minting from labels. No language model is involved.

Per (window, finding) the inputs are a presence sequence over the window's studies (pos / neg / unk, CheXbert,
step 2) and, per interval, an ImaGenome comparison label (improved / worsened / no change / unk, step 3).
Every answer below is a pure function of those two sequences, so step 7 can recompute it after an intervention.

Interval change (both presence states must be known; decided 2026-10-07):
    neg -> pos : "new"       unless ImaGenome says improved / no change   (contradiction -> unk)
    pos -> neg : "resolved"  unless ImaGenome says worsened / no change   (contradiction -> unk)
    pos -> pos : ImaGenome worsened / improved / no change -> "worsened" / "improved" / "unchanged"; missing -> unk
    neg -> neg : "absent"    (finding absent at both studies; never an ICR answer)
The source of each known change is recorded: presence_flip, imagenome, or both.
"""

import hashlib

CHANGES = ("new", "worsened", "improved", "unchanged", "resolved")
UP, DOWN = {"new", "worsened"}, {"improved", "resolved"}

# GTS trajectories (decided 2026-10-07). Classified from the signs of the interval changes, zeros ignored.
TRAJECTORIES = ("stable", "worsening", "improving", "worse_then_better", "better_then_worse")
TEL_FINDINGS = ("Pleural Effusion", "Pneumothorax", "Edema", "Cardiomegaly", "Lung Opacity")
# R1 (labels v2, docs/label_fix_design.md): Support Devices is a group, so its presence flips are not findings
# changes; it mints no longitudinal (and hence no EXIST) items
NO_LONGITUDINAL = ("Support Devices",)


def interval_change(s0: str, s1: str, img: str | None) -> tuple[str, str | None]:
    """(change, source) for one interval. change is one of CHANGES, 'absent' or 'unk'."""
    img = None if img in (None, "unk") else img
    if s0 == "unk" or s1 == "unk":
        return "unk", None
    if s0 == "neg" and s1 == "pos":
        if img in ("improved", "no change"):
            return "unk", None
        return "new", "both" if img == "worsened" else "presence_flip"
    if s0 == "pos" and s1 == "neg":
        if img in ("worsened", "no change"):
            return "unk", None
        return "resolved", "both" if img == "improved" else "presence_flip"
    if s0 == "pos" and s1 == "pos":
        if img is None:
            return "unk", None
        return {"worsened": "worsened", "improved": "improved", "no change": "unchanged"}[img], "imagenome"
    return "absent", None   # neg -> neg; ImaGenome changes on an absent finding are ignored


def has_flicker(states: list[str]) -> bool:
    """True if the known states contain a reversal back to an earlier value (pos-neg-pos or neg-pos-neg)."""
    known = [s for s in states if s != "unk"]
    runs = [s for i, s in enumerate(known) if i == 0 or s != known[i - 1]]
    return len(runs) >= 3


def trajectory(changes: list[str]) -> str | None:
    """GTS class from fully known interval changes, or None if not classifiable / not mintable."""
    if any(c == "unk" for c in changes) or all(c == "absent" for c in changes):
        return None
    signs = [1 if c in UP else -1 for c in changes if c in UP | DOWN]
    if not signs:
        return "stable" if all(c == "unchanged" for c in changes) else None
    runs = [s for i, s in enumerate(signs) if i == 0 or s != signs[i - 1]]
    return {(1,): "worsening", (-1,): "improving", (1, -1): "worse_then_better",
            (-1, 1): "better_then_worse"}.get(tuple(runs))


def tel_answer(states: list[str], event: str) -> int | None:
    """0-based interval of the FIRST 'appear' (neg->pos) or 'resolve' (pos->neg); -1 if none; None if unknown.

    "First event" rule (2026-10-07, generalised 2026-10-08): the answer is the first such transition, provided every
    study up to and including it is known; later studies cannot change it and may be unknown. A "none" answer needs
    every study known. (On non-flicker windows this equals the earlier "prefix in the starting state" wording.)
    """
    a, b = ("neg", "pos") if event == "appear" else ("pos", "neg")
    for i in range(len(states) - 1):
        if states[i] == "unk" or states[i + 1] == "unk":
            return None
        if (states[i], states[i + 1]) == (a, b):
            return i
    return -1


def item_id(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def shuffled(options: list[str], key: str) -> list[str]:
    """Deterministic per-item permutation (independent of row order and of Python's hash seed)."""
    return sorted(options, key=lambda o: hashlib.sha256(f"{key}|{o}".encode()).hexdigest())


# ---------------------------------------------------------------------------------------------------------------
# Item templates. Option text is fixed per template id; letters are assigned at render time from `options` order.

CANNOT = "Cannot be assessed from these images"
CHANGE_TEXT = {"new": "Newly appeared", "worsened": "Worsened", "improved": "Improved",
               "unchanged": "Unchanged", "resolved": "Resolved"}
TRAJ_TEXT = {
    "stable": "Present and unchanged throughout",
    "worsening": "Worsens or newly appears, with no improvement",
    "improving": "Improves or resolves, with no worsening",
    "worse_then_better": "First worsens, then improves",
    "better_then_worse": "First improves, then worsens",
}


def _lc(finding: str) -> str:
    return finding.lower()


def _interval(i: int) -> str:
    return f"T{i + 1} → T{i + 2}"


def mint_window(window_id: str, n: int, finding: str, states: list[str], imgs: list[str | None],
                views: list[str]) -> list[dict]:
    """All items for one (window, finding). states: n presence labels; imgs: n-1 ImaGenome labels (interval i is
    study i -> i+1); views: n view positions. Returns plain dicts; evidence indices are 1-based study positions."""
    assert len(states) == n and len(imgs) == n - 1 and len(views) == n
    if finding in NO_LONGITUDINAL:
        return []
    changes, sources = zip(*(interval_change(states[i], states[i + 1], imgs[i]) for i in range(n - 1)))
    changes, sources = list(changes), list(sources)
    flicker = has_flicker(states)
    view_switch = len(set(views)) > 1
    # AP films magnify the heart: no cardiomegaly change questions across a view switch
    cardio_blocked = finding == "Cardiomegaly"
    common = {"window_id": window_id, "n": n, "finding": finding, "states": list(states), "changes": changes,
              "change_sources": sources, "flicker": flicker, "view_switch": view_switch}
    items = []

    def add(family, subtype, template, question, options, answer, required, source=None):
        qid = item_id(window_id, finding, family, subtype)
        items.append({"qid": qid, "family": family, "subtype": subtype, "template": template,
                      "question": question, "options": options, "answer": answer,
                      "answer_letter": "ABCDEFGHIJ"[options.index(answer)],
                      "required_evidence": sorted(required), "change_source": source,
                      "text_determined": max(required) < n, **common})

    # TEL, "first event" rule (see tel_answer); windows with presence flicker are still excluded
    if finding in TEL_FINDINGS and not flicker and not (cardio_blocked and view_switch):
        for event, verb, none_text in (("appear", "first newly appear", "does not newly appear"),
                                       ("resolve", "first resolve", "does not resolve")):
            k = tel_answer(states, event)
            if k is None:
                continue
            opts = [_interval(i) for i in range(n - 1)] + [f"{finding} {none_text} in this sequence", CANNOT]
            ans = opts[k] if k >= 0 else opts[n - 1]
            req = [k + 1, k + 2] if k >= 0 else list(range(1, n + 1))
            add("TEL", event, f"tel_{event}_v2", f"During which interval does {_lc(finding)} {verb}?", opts, ans,
                req, source=sources[k] if k >= 0 else None)

    # ICR: one item per interval with a known change
    for i, c in enumerate(changes):
        if c not in CHANGES or (cardio_blocked and views[i] != views[i + 1]):
            continue
        key = item_id(window_id, finding, "ICR", i)
        opts = shuffled([CHANGE_TEXT[x] for x in CHANGES], key) + [CANNOT]
        add("ICR", f"interval{i}", "icr_v1", f"Between T{i + 1} and T{i + 2}, how did {_lc(finding)} change?",
            opts, CHANGE_TEXT[c], [i + 1, i + 2], source=sources[i])

    # GTS: every interval known, no flicker, classifiable trajectory
    t = None if flicker or (cardio_blocked and view_switch) else trajectory(changes)
    if t is not None:
        key = item_id(window_id, finding, "GTS", "traj")
        opts = shuffled(list(TRAJ_TEXT.values()), key) + [CANNOT]
        add("GTS", "trajectory", "gts_v1", f"Which best describes the course of {_lc(finding)} from T1 to T{n}?",
            opts, TRAJ_TEXT[t], list(range(1, n + 1)))

    # Existence (single study; Exp 6), only where the window produced a longitudinal item
    if items:
        for i, s in enumerate(states):
            if s in ("pos", "neg"):
                opts = ["Yes", "No", CANNOT]
                add("EXIST", f"study{i}", "exist_v1", f"Is {_lc(finding)} present on T{i + 1}?", opts,
                    "Yes" if s == "pos" else "No", [i + 1])
    return items
