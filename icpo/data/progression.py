"""Pipeline step 3 (roadmap §3.2.3): interval progression per (current study, finding).

Input rows are region-level comparison statements (ImaGenome: a bbox region, a label name, and one or more of
improved / worsened / no change against the previous exam). Aggregation per (study, finding) is a majority vote over
regions; ties, and regions that carry contradictory changes, give unk.
"""

from collections import Counter

import pandas as pd

from icpo import config

CHANGES = ("improved", "worsened", "no change")
UNK = "unk"


def label_map() -> dict[str, str]:
    """ImaGenome label name -> our finding."""
    return {name: f for f, names in config.load("imagenome_findings").items() for name in names}


def region_change(changes: tuple[str, ...]) -> str | None:
    """One region's change, or None if it is empty or contradictory (e.g. both 'no change' and 'worsened')."""
    c = {x for x in changes if x in CHANGES}
    return c.pop() if len(c) == 1 else None


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """rows: study_id, region, label_name, changes (tuple). Returns study_id, finding, change, n_regions, n_votes."""
    lm = label_map()
    r = rows.assign(finding=rows.label_name.map(lm), change=rows.changes.map(region_change))
    r = r.dropna(subset=["finding"])
    # one vote per (study, finding, region); a region contradicting itself across rows abstains
    per_region = r.groupby(["study_id", "finding", "region"]).change.agg(
        lambda s: s.iloc[0] if s.notna().all() and s.nunique() == 1 else None)
    out = []
    for (sid, f), votes in per_region.groupby(level=[0, 1]):
        v = Counter(x for x in votes if pd.notna(x))  # pandas turns an abstaining None into NaN
        top = v.most_common(2)
        if not top or (len(top) == 2 and top[0][1] == top[1][1]):
            change = UNK
        else:
            change = top[0][0]
        out.append((sid, f, change, len(votes), sum(v.values())))
    return pd.DataFrame(out, columns=["study_id", "finding", "change", "n_regions", "n_votes"])
