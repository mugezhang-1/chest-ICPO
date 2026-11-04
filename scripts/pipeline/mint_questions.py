"""Pipeline step 6 (roadmap §3.2.6): the full candidate pool of questions, minted deterministically from labels.

Inputs: windows (step 5), CheXbert presence labels (step 2), ImaGenome interval changes (step 3).
Output: <work_root>/pipeline/questions_pool.parquet (+ provenance, stats). Sampling to <= 12 items per patient and
answer balancing happen after the patient-level split (step 8), never here.
"""

import argparse
import json
from collections import Counter

import pandas as pd

from icpo import config
from icpo.data.labels import FINDINGS
from icpo.data.mint import mint_window
from icpo.provenance import add_cli, require_clean_tree, write_table


def main():
    args = add_cli(argparse.ArgumentParser()).parse_args()
    git = require_clean_tree(args.allow_dirty)
    work = config.path("work_root") / "pipeline"
    windows = pd.read_parquet(work / "windows.parquet")
    labels = pd.read_parquet(work / "labels.parquet").set_index("study_id")
    prog = pd.read_parquet(work / "progression.parquet")
    img = {(s, f): c for s, f, c in zip(prog.study_id, prog.finding, prog.change)}
    lab = {f: labels[f].to_dict() for f in FINDINGS}

    rows, reasons = [], Counter()
    for w in windows.itertuples(index=False):
        sids = [int(s) for s in w.study_ids]
        for f in FINDINGS:
            states = [lab[f].get(s, "unk") for s in sids]
            imgs = [img.get((s, f)) for s in sids[1:]]
            items = mint_window(w.window_id, int(w.n), f, states, imgs, list(w.views))
            reasons["windows_x_findings"] += 1
            reasons["with_items" if items else "no_items"] += 1
            for it in items:
                it.update({"subject_id": int(w.subject_id), "study_ids": sids, "dicom_ids": list(w.dicom_ids),
                           "gap_days": list(w.gap_days)})
                rows.append(it)

    df = pd.DataFrame(rows).sort_values(["window_id", "finding", "family", "subtype"]).reset_index(drop=True)
    if df.qid.duplicated().any():
        raise SystemExit(f"{df.qid.duplicated().sum()} duplicate qids")

    long = df[df.family != "EXIST"]
    stats = {
        "items_by_family": df.family.value_counts().to_dict(),
        "windows_x_findings": dict(reasons),
        "longitudinal_by_family_finding": long.groupby(["family", "finding"]).size().unstack(0).fillna(0).astype(int).to_dict(),
        "tel_answer_is_none_rate": round(float(df[df.family == "TEL"].answer.str.contains("in this sequence").mean()), 4),
        "icr_answer_counts": df[df.family == "ICR"].answer.value_counts().to_dict(),
        "icr_change_source": df[df.family == "ICR"].change_source.value_counts().to_dict(),
        "gts_answer_counts": df[df.family == "GTS"].answer.value_counts().to_dict(),
        "flicker_rate_longitudinal": round(float(long.flicker.mean()), 4),
        "text_determined_rate_by_family": df.groupby("family").text_determined.mean().round(4).to_dict(),
        "answer_letter_by_family": df.groupby("family").answer_letter.value_counts().unstack(0).fillna(0).astype(int).to_dict(),
        "patients": int(df.subject_id.nunique()),
        "windows_with_items": int(df.window_id.nunique()),
    }
    write_table(df, work / "questions_pool.parquet", step="6_mint", git=git,
                params={"templates": sorted(df.template.unique().tolist()), "tel_findings": "icpo.data.mint.TEL_FINDINGS",
                        "no_longitudinal": "icpo.data.mint.NO_LONGITUDINAL"},
                inputs=[work / "windows.parquet", work / "labels.parquet", work / "progression.parquet"],
                stats=json.loads(json.dumps(stats, default=int)))
    (work / "questions_pool.stats.json").write_text(json.dumps(stats, indent=2, default=int))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()
