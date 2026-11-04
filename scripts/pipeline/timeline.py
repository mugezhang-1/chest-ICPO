"""Pipeline step 1 (roadmap §3.2.1): per-patient timelines of frontal studies.

Output: <work_root>/pipeline/timelines.parquet, one row per study, with the official split attached.
"""

import argparse
import json

import pandas as pd

from icpo import config
from icpo.provenance import add_cli, require_clean_tree, write_table
from icpo.data.images import mimic_relpath
from icpo.data.timeline import build_timelines


def main():
    args = add_cli(argparse.ArgumentParser()).parse_args()
    git = require_clean_tree(args.allow_dirty)
    tables = config.path("mimic_jpg_tables")
    meta = pd.read_csv(tables / "mimic-cxr-2.0.0-metadata.csv.gz")
    split = pd.read_csv(tables / "mimic-cxr-2.0.0-split.csv.gz")

    tl = build_timelines(meta)
    tl = tl.merge(split[["dicom_id", "split"]], on="dicom_id", how="left", validate="one_to_one")
    tl["relpath"] = [mimic_relpath(s, st, d) for s, st, d in zip(tl.subject_id, tl.study_id, tl.dicom_id)]

    n_frontal = meta[meta.ViewPosition.isin(["PA", "AP"])].study_id.nunique()
    same_time = tl.duplicated(["subject_id", "t"], keep=False)
    stats = {
        "studies_with_frontal": int(n_frontal),
        "studies_kept": len(tl),
        "dropped_no_time": int(n_frontal - len(tl)),
        "patients": int(tl.subject_id.nunique()),
        "studies_per_patient": tl.groupby("subject_id").size().describe().round(2).to_dict(),
        "view_counts": tl.ViewPosition.value_counts().to_dict(),
        "split_counts": tl.split.value_counts(dropna=False).to_dict(),
        "studies_sharing_exact_time_with_another": int(same_time.sum()),
    }

    out = config.path("work_root") / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    write_table(tl, out / "timelines.parquet", step="1_timelines", params={"frontal_views": ["PA", "AP"]},
                inputs=[tables / "mimic-cxr-2.0.0-metadata.csv.gz", tables / "mimic-cxr-2.0.0-split.csv.gz"],
                git=git, stats=stats)
    (out / "timelines.stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
