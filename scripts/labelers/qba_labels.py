"""Study-level labels for the eight findings from MIMIC-Ext-CXR-QBA observations, in the CheXpert encoding.

Output: <work_root>/labelers/qba.parquet (study_id + FINDINGS; 1.0 / 0.0 / -1.0 / NaN).
"""

import argparse

import pandas as pd

from icpo import config
from icpo.provenance import add_cli, require_clean_tree, write_table
from icpo.data.labels import FINDINGS, build_qba


def main():
    args = add_cli(argparse.ArgumentParser()).parse_args()
    git = require_clean_tree(args.allow_dirty)
    src = config.path("physionet_root") / "mimic-ext-cxr-qba" / "1.0.1" / "stats" / "all_obs.csv.gz"
    obs = pd.read_csv(src, usecols=["study_id", "entities", "subcategories", "positiveness", "certainty"])
    obs["study_id"] = obs.study_id.str[1:].astype(int)
    q = build_qba(obs)
    dst = config.path("work_root") / "labelers" / "qba.parquet"
    write_table(q, dst, step="labeler_qba", params={"qba_version": "1.0.1"}, inputs=[src], git=git)
    print(f"{len(q)} studies -> {dst}")
    print(pd.DataFrame({f: q[f].value_counts(dropna=False) for f in FINDINGS}).T.fillna(0).astype(int))


if __name__ == "__main__":
    main()
