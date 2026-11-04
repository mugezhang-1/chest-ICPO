"""Pipeline step 5 (roadmap §3.2.5): windows of 3-5 consecutive frontal studies with gaps <= 90 days.

Output: <work_root>/pipeline/windows.parquet and windows.stats.json.
"""

import argparse
import json

import pandas as pd

from icpo import config
from icpo.provenance import add_cli, require_clean_tree, write_table
from icpo.data.timeline import build_windows


def main():
    args = add_cli(argparse.ArgumentParser()).parse_args()
    git = require_clean_tree(args.allow_dirty)
    work = config.path("work_root") / "pipeline"
    tl = pd.read_parquet(work / "timelines.parquet")
    w = build_windows(tl, sizes=(3, 4, 5), max_gap_days=90)
    gaps = pd.Series([g for gs in w.gap_days for g in gs])
    stats = {
        "windows_by_n": w.n.value_counts().sort_index().to_dict(),
        "patients_by_n": w.groupby("n").subject_id.nunique().to_dict(),
        "gap_days_quantiles": gaps.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict(),
        "span_days_median_by_n": w.groupby("n").span_days.median().round(1).to_dict(),
        "view_switch_rate_by_n": w.groupby("n").view_switch.mean().round(3).to_dict(),
    }
    write_table(w, work / "windows.parquet", step="5_windows", params={"sizes": [3, 4, 5], "max_gap_days": 90},
                inputs=[work / "timelines.parquet"], git=git, stats=json.loads(json.dumps(stats, default=str)))
    (work / "windows.stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
