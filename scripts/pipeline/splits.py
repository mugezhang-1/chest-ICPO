"""Pipeline step 8 (roadmap §3.2.8): patient-level split, with every external-evaluation patient held out.

Excluded from train/val/test (reason recorded per patient):
  micxr            every patient in MI-CXR's test file (image paths)
  ms_cxr_t         every patient in MS-CXR-T's temporal classification file
  mimic_test       every patient with a study in the official MIMIC-CXR-JPG test split
  lomevqa_test     every patient in LoMeVQA's five *_test.json files (image paths)
  chextemporal_gold  MIMIC patients in CheXTemporal's gold pairs. Its silver set (8,497 MIMIC patients, LLM labels,
                   "evaluation only") is NOT excluded; any later silver evaluation filters to unseen patients.
Remaining patients go to train / val / test by a hash of the subject_id (90/5/5), so a patient's split never
depends on which other patients are present.
Output: <work_root>/pipeline/splits.parquet (subject_id, split, exclusion reasons) + provenance.
"""

import argparse
import hashlib
import json
import re
import zipfile

import pandas as pd

from icpo import config
from icpo.provenance import add_cli, require_clean_tree, write_table

FRACTIONS = (("train", 0.90), ("val", 0.05), ("test", 0.05))
SALT = "icpo-split-v1"


def hash_split(subject_id: int) -> str:
    u = int(hashlib.sha256(f"{SALT}|{subject_id}".encode()).hexdigest(), 16) / 2**256
    acc = 0.0
    for name, frac in FRACTIONS:
        acc += frac
        if u < acc:
            return name
    return FRACTIONS[-1][0]


def main():
    args = add_cli(argparse.ArgumentParser()).parse_args()
    git = require_clean_tree(args.allow_dirty)
    work = config.path("work_root") / "pipeline"
    ext, phys = config.path("external_root"), config.path("physionet_root")
    paths = {
        "micxr": ext / "MI-CXR" / "micxr_test.jsonl",
        "ms_cxr_t": phys / "ms-cxr-t" / "1.0.0" / "MS_CXR_T_temporal_image_classification_v1.0.0.csv",
        "mimic_split": config.path("mimic_jpg_tables") / "mimic-cxr-2.0.0-split.csv.gz",
        "lomevqa": ext / "LoMeVQA" / "hf" / "LoMeVQA_Dataset.zip",
        "chextemporal_gold": ext / "CheXTemporal" / "gold_progression_pairs.parquet",
    }
    pid = re.compile(r"p1\d/p(1\d{7})/")
    excl = {
        "micxr": {int(x) for line in open(paths["micxr"]) for x in pid.findall(line)},
        "ms_cxr_t": set(pd.read_csv(paths["ms_cxr_t"]).subject_id.astype(int)),
        "mimic_test": set(pd.read_csv(paths["mimic_split"]).query("split == 'test'").subject_id.astype(int)),
        "chextemporal_gold": set(pd.read_parquet(paths["chextemporal_gold"]).query("dataset == 'mimic'")
                                 .patient_id.astype(int)),
    }
    with zipfile.ZipFile(paths["lomevqa"]) as z:
        excl["lomevqa_test"] = {int(x) for n in z.namelist() if n.endswith("_test.json")
                                for x in pid.findall(z.read(n).decode())}

    tl = pd.read_parquet(work / "timelines.parquet")
    subjects = sorted(set(tl.subject_id.astype(int)))
    rows = []
    for s in subjects:
        reasons = [k for k, v in excl.items() if s in v]
        rows.append({"subject_id": s, "split": "excluded" if reasons else hash_split(s),
                     "exclusion_reasons": reasons})
    df = pd.DataFrame(rows)

    pool = pd.read_parquet(work / "questions_pool.parquet", columns=["subject_id", "family"])
    in_pool = set(pool.subject_id)
    stats = {
        "external_patients": {k: len(v) for k, v in excl.items()},
        "timeline_patients_by_split": df.split.value_counts().to_dict(),
        "pool_patients_by_split": df[df.subject_id.isin(in_pool)].split.value_counts().to_dict(),
        "pool_items_by_split": pool.merge(df[["subject_id", "split"]], on="subject_id")
                                   .groupby(["split", "family"]).size().unstack(1).fillna(0).astype(int).to_dict(),
        "pool_patients_excluded_by_reason": pd.Series([r for rs in df[df.subject_id.isin(in_pool)].exclusion_reasons
                                                       for r in rs]).value_counts().to_dict(),
    }
    write_table(df, work / "splits.parquet", step="8_splits", git=git,
                params={"fractions": dict(FRACTIONS), "salt": SALT,
                        "chextemporal_silver_excluded": False},
                inputs=list(paths.values()) + [work / "timelines.parquet", work / "questions_pool.parquet"],
                stats=json.loads(json.dumps(stats, default=int)))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()
