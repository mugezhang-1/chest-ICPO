"""Pipeline step 1: one frontal image per study, ordered in time per patient (roadmap §3.2.1)."""

import pandas as pd

FRONTAL = ("PA", "AP")


def study_datetime(date: pd.Series, time: pd.Series) -> pd.Series:
    """StudyDate (YYYYMMDD) + StudyTime (HHMMSS.ffff) -> Timestamp; unparseable -> NaT."""
    d = pd.to_numeric(date, errors="coerce")
    t = pd.to_numeric(time, errors="coerce")
    ok = d.notna() & t.notna()
    out = pd.Series(pd.NaT, index=date.index, dtype="datetime64[ns]")
    hhmmss = t[ok].astype("int64").astype(str).str.zfill(6)
    out[ok] = pd.to_datetime(d[ok].astype("int64").astype(str) + hhmmss, format="%Y%m%d%H%M%S", errors="coerce")
    return out


def build_timelines(metadata: pd.DataFrame) -> pd.DataFrame:
    """Select one frontal image per study (PA over AP, then lowest dicom_id) and sort per patient.

    Studies without a parseable time are dropped. Returns one row per study with `t`, `idx`
    (0-based position in the patient's timeline) and `gap_days` to the previous study.
    """
    m = metadata[metadata["ViewPosition"].isin(FRONTAL)].copy()
    m["t"] = study_datetime(m["StudyDate"], m["StudyTime"])
    m = m.dropna(subset=["t"])
    m["_view_rank"] = (m["ViewPosition"] != "PA").astype(int)
    m = m.sort_values(["study_id", "_view_rank", "dicom_id"]).drop_duplicates("study_id")
    # study_id breaks exact-time ties so the order is deterministic
    m = m.sort_values(["subject_id", "t", "study_id"]).reset_index(drop=True)
    m["idx"] = m.groupby("subject_id").cumcount()
    m["gap_days"] = m.groupby("subject_id")["t"].diff().dt.total_seconds() / 86400
    cols = ["subject_id", "study_id", "dicom_id", "ViewPosition", "Rows", "Columns", "t", "idx", "gap_days"]
    return m[cols]


def build_windows(tl: pd.DataFrame, sizes=(3, 4, 5), max_gap_days: float = 90) -> pd.DataFrame:
    """All runs of n consecutive studies (n in sizes) with every consecutive gap in (0, max_gap_days].

    A zero gap (two studies of one patient at the same timestamp) makes the order ambiguous, so any window
    containing such a pair is dropped. Returns one row per window with ordered study/dicom ids, gaps and views.
    """
    rows = []
    for sid, g in tl.sort_values(["subject_id", "idx"]).groupby("subject_id", sort=False):
        studies, dicoms = g.study_id.to_list(), g.dicom_id.to_list()
        views, gaps = g.ViewPosition.to_list(), g.gap_days.to_list()
        for n in sizes:
            for i in range(len(g) - n + 1):
                gw = gaps[i + 1 : i + n]
                if all(0 < x <= max_gap_days for x in gw):
                    rows.append({
                        "window_id": f"{sid}_{studies[i]}_{n}", "subject_id": sid, "n": n,
                        "study_ids": studies[i : i + n], "dicom_ids": dicoms[i : i + n],
                        "views": views[i : i + n], "gap_days": [round(x, 3) for x in gw],
                        "span_days": round(sum(gw), 3), "view_switch": len(set(views[i : i + n])) > 1,
                    })
    return pd.DataFrame(rows)


def window_studies(tl: pd.DataFrame, n: int = 3, max_gap_days: float = 90) -> pd.Index:
    """Row labels of studies that belong to at least one window of `n` consecutive studies
    whose consecutive gaps are all <= max_gap_days. Larger windows are unions of these."""
    ok_gap = (tl["gap_days"] <= max_gap_days)
    keep = pd.Series(False, index=tl.index)
    for _, g in tl.groupby("subject_id", sort=False):
        if len(g) < n:
            continue
        ok = ok_gap.loc[g.index].to_numpy()
        for i in range(len(g) - n + 1):
            # gaps for studies i+1..i+n-1 (the gap of study i points outside the window)
            if ok[i + 1 : i + n].all():
                keep.loc[g.index[i : i + n]] = True
    return tl.index[keep.to_numpy()]
