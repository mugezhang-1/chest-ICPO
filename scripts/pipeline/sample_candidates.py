"""Labels v2, step 7a: candidate base items for the second readers (docs/label_fix_design.md, "Sampling with readers").

The existing sampler (icpo.data.sampling) is run per split at FACTOR times the target size; the readers then filter
the candidates, and build_bundles.py samples the final bases from the passing candidates at the target sizes.
Output: <work_root>/pipeline/candidates.parquet (pool rows + split) with provenance.
"""

import argparse
import json

import pandas as pd

from icpo import config
from icpo.data.sampling import sample_split
from icpo.provenance import add_cli, require_clean_tree, write_table

SIZES = {"train": 20000, "val": 500, "test": 1000}


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--factor", type=float, default=2.0)
    args = ap.parse_args()
    git = require_clean_tree(args.allow_dirty)
    P = config.path("work_root") / "pipeline"
    pool = pd.read_parquet(P / "questions_pool.parquet")
    splits = pd.read_parquet(P / "splits.parquet")
    pool["split"] = pool.subject_id.map(dict(zip(splits.subject_id, splits.split)))
    cand = pd.concat([sample_split(pool[pool.split == s], int(args.factor * n)).assign(split=s)
                      for s, n in SIZES.items()], ignore_index=True).sort_values("qid").reset_index(drop=True)
    stats = {"factor": args.factor, "by_split_family": cand.groupby(["split", "family"]).size().unstack(1)
             .fillna(0).astype(int).to_dict(), "by_split_finding": cand.groupby(["split", "finding"]).size()
             .unstack(0).fillna(0).astype(int).to_dict()}
    write_table(cand, P / "candidates.parquet", step="7a_candidates", git=git,
                inputs=[P / "questions_pool.parquet", P / "splits.parquet"],
                params={"sizes": SIZES, "factor": args.factor, "sampler": "icpo.data.sampling.sample_split"},
                stats=json.loads(json.dumps(stats, default=int)))
    print(json.dumps(stats, indent=1, default=int))


if __name__ == "__main__":
    main()
