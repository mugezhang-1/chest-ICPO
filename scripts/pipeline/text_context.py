"""Pipeline step 4 (roadmap §3.2.4): per-study indication and prior-impression text.

Output: <work_root>/pipeline/text_context.parquet (study_id, indication, indication_truncated, prior_impression,
prior_impression_source) and text_context.stats.json.
"""

import argparse
import json

import pandas as pd

from icpo import config
from icpo.provenance import add_cli, require_clean_tree, write_table
from icpo.data.reports import iter_reports
from icpo.data.text_context import indication, prior_impression


def main():
    args = add_cli(argparse.ArgumentParser()).parse_args()
    git = require_clean_tree(args.allow_dirty)
    zip_path = config.path("physionet_root") / "mimic-cxr" / "2.1.0" / "mimic-cxr-reports.zip"
    rows = []
    for sid, text in iter_reports(zip_path):
        ind, trunc = indication(text)
        imp, src = prior_impression(text)
        rows.append((sid, ind, trunc, imp, src))
    df = pd.DataFrame(rows, columns=["study_id", "indication", "indication_truncated", "prior_impression",
                                     "prior_impression_source"])
    stats = {
        "studies": len(df),
        "with_indication": int((df.indication != "").sum()),
        "indication_truncated": int(df.indication_truncated.sum()),
        "prior_impression_source": df.prior_impression_source.value_counts().to_dict(),
        "indication_words_median": float(df.indication.str.split().str.len().median()),
        "prior_impression_words_median": float(df.prior_impression.str.split().str.len().median()),
    }
    out = config.path("work_root") / "pipeline"
    write_table(df, out / "text_context.parquet", step="4_text_context", params={}, inputs=[zip_path],
                git=git, stats=stats)
    (out / "text_context.stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    for r in df[df.indication_truncated].sample(5, random_state=0).itertuples():
        print(f"[truncated] {r.indication!r}")
    for r in df.sample(5, random_state=1).itertuples():
        print(f"[sample] IND={r.indication!r} | PRIOR={r.prior_impression[:160]!r}")


if __name__ == "__main__":
    main()
