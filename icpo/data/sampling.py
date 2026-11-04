"""Selecting base items per split (between steps 7 and 9; docs/step7_design.md "Sampling").

1. Candidates: TEL / ICR / GTS items of the split, no flicker windows.
2. One item per distinct event: overlapping windows repeat the same event; the copy kept is chosen by hash.
     TEL event : (patient, finding, subtype, the two studies of the transition)
     TEL none  : (patient, finding, subtype, the window's studies)
     ICR       : (patient, finding, the interval's two studies)
     GTS       : (patient, finding, the window's studies)
3. Quotas: families in equal shares; within a family, answer classes as even as the data allow, with the dominant
   class (TEL none, ICR Unchanged, GTS stable) capped at 25% of the family. Unfillable quota passes to the other
   classes of the same family.
4. At most `per_patient` items per patient across the split.
Selection is greedy over a hash order of item ids, so the result depends only on the pool and the parameters.
"""

import hashlib
from collections import Counter, defaultdict

import pandas as pd

FAMILIES = ("TEL", "ICR", "GTS")
DOMINANT = {"TEL": "none", "ICR": "Unchanged", "GTS": "Present and unchanged throughout"}


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def answer_class(family: str, answer: str) -> str:
    if family == "TEL":
        return "event" if "→" in answer else "none"
    return answer


def event_key(r) -> tuple:
    ev = tuple(int(r.study_ids[i - 1]) for i in r.required_evidence)
    if r.family == "TEL" and "→" in r.answer:
        return ("TEL", r.subject_id, r.finding, r.subtype, ev)
    return (r.family, r.subject_id, r.finding, r.subtype if r.family == "TEL" else "", tuple(map(int, r.study_ids))
            if r.family != "ICR" else ev)


def class_quotas(counts: dict, total: int, dominant: str, cap: float = 0.25) -> dict:
    """Even split of `total` over classes, bounded by availability and by the dominant-class cap."""
    limit = {c: min(n, int(cap * total)) if c == dominant else n for c, n in counts.items()}
    q = {c: 0 for c in counts}
    left = total
    while left > 0:
        open_ = [c for c in q if q[c] < limit[c]]
        if not open_:
            break
        share = max(1, left // len(open_))
        for c in sorted(open_):
            add = min(share, limit[c] - q[c], left)
            q[c] += add
            left -= add
            if left == 0:
                break
    return q


def sample_split(pool: pd.DataFrame, n_items: int, per_patient: int = 12, salt: str = "sample-v1") -> pd.DataFrame:
    df = pool[pool.family.isin(FAMILIES) & ~pool.flicker].copy()
    df["cls"] = [answer_class(f, a) for f, a in zip(df.family, df.answer)]
    df["key"] = [event_key(r) for r in df.itertuples(index=False)]
    df["rank"] = [_h(salt, q) for q in df.qid]
    df = df.sort_values("rank").drop_duplicates("key").reset_index(drop=True)

    fam_target = {f: n_items // len(FAMILIES) + (1 if i < n_items % len(FAMILIES) else 0) for i, f in enumerate(FAMILIES)}
    quotas = {}
    for f in FAMILIES:
        counts = df[df.family == f].cls.value_counts().to_dict()
        quotas[f] = class_quotas(counts, fam_target[f], DOMINANT[f])

    taken, per_pat, used = [], Counter(), defaultdict(int)
    for r in df.itertuples(index=False):          # already in hash order
        if used[(r.family, r.cls)] >= quotas[r.family].get(r.cls, 0) or per_pat[r.subject_id] >= per_patient:
            continue
        taken.append(r.qid)
        used[(r.family, r.cls)] += 1
        per_pat[r.subject_id] += 1
    out = df[df.qid.isin(set(taken))].drop(columns=["rank", "key"])
    return out.sort_values("qid").reset_index(drop=True)
