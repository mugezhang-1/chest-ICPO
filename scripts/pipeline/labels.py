"""Pipeline step 2 (roadmap §3.2.2) plus presence-label validation (§3.3).

Main rule (labels v2, decided 2026-10-11; docs/label_fix_design.md): CheXbert explicit labels
(<work_root>/labelers/chexbert.parquet), vetoed where QBA, CheXpert or NegBio states the opposite (R3), then the
Lung Opacity hierarchy (R2). --labeler chexbert_v1 rebuilds the earlier CheXbert-only rule (2026-10-06;
docs/micxr_alignment.md §7); --labeler cxnb the CheXpert/NegBio agreement. Both are for ablations only.

Output: <work_root>/pipeline/labels.parquet and labels.stats.json (labels_<labeler>.* for the ablations).
Validation compares our pos/neg cells with mimic-cxr-2.1.0-test-set-labeled.csv (radiologist labels; "Airspace Opacity" there = "Lung Opacity" here).
The radiologist file's blank cells are reported both as "not mentioned" (excluded) and as negative, since the
release does not make their meaning unambiguous.
"""

import argparse
import json
import math

import pandas as pd

from icpo import config
from icpo.provenance import add_cli, require_clean_tree, write_table
from icpo.data.labels import FINDINGS, NEG, POS, UNK, build_chexbert_labels, build_chexbert_labels_v2, build_labels

RAD_NAME = {"Lung Opacity": "Airspace Opacity"}


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def validate(lab: pd.DataFrame, rad: pd.DataFrame) -> dict:
    m = lab.merge(rad, on="study_id", suffixes=("", "_rad"))
    out = {"n_studies": len(m)}
    for f in FINDINGS:
        col = RAD_NAME.get(f, f)
        if col not in rad:
            continue
        r = m[f"{col}_rad" if col in lab.columns else col]
        ours = m[f]
        pos, neg = ours == POS, ours == NEG
        rad_pos, rad_neg_strict, rad_neg_lenient = r == 1.0, r == 0.0, (r == 0.0) | r.isna()
        out[f] = {
            "ours_pos": int(pos.sum()), "ours_neg": int(neg.sum()), "ours_unk": int((ours == UNK).sum()),
            "pos_precision": float(rad_pos[pos].mean()) if pos.any() else None,
            "pos_precision_ci95": wilson(int(rad_pos[pos].sum()), int(pos.sum())),
            # same, ignoring studies where the radiologist left this finding blank
            "pos_precision_rad_explicit": float(rad_pos[pos & r.notna()].mean()) if (pos & r.notna()).any() else None,
            "pos_rad_blank": int((pos & r.isna()).sum()),
            "neg_precision_strict": float(rad_neg_strict[neg].mean()) if neg.any() else None,
            "neg_precision_lenient": float(rad_neg_lenient[neg].mean()) if neg.any() else None,
            "rad_pos_recalled": float(pos[rad_pos].mean()) if rad_pos.any() else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeler", choices=["chexbert_v2", "chexbert_v1", "cxnb"], default="chexbert_v2",
                    help="chexbert_v2 = main rule; chexbert_v1 and cxnb = earlier rules (ablations only)")
    args = add_cli(ap).parse_args()
    git = require_clean_tree(args.allow_dirty)

    tables = config.path("mimic_jpg_tables")
    rad = pd.read_csv(tables / "mimic-cxr-2.1.0-test-set-labeled.csv")
    extra_stats, extra_inputs = {}, []
    if args.labeler.startswith("chexbert"):
        src_path = config.path("work_root") / "labelers" / "chexbert.parquet"
        if not src_path.exists():
            raise SystemExit(f"{src_path} missing; run scripts/labelers/run_chexbert.sbatch first")
        src = pd.read_parquet(src_path)
        meta = pd.read_csv(tables / "mimic-cxr-2.0.0-metadata.csv.gz", usecols=["subject_id", "study_id"])
        if args.labeler == "chexbert_v1":
            lab = build_chexbert_labels(src, meta)
        else:
            qba_path = config.path("work_root") / "labelers" / "qba.parquet"
            others = {"qba": pd.read_parquet(qba_path),
                      "chexpert": pd.read_csv(tables / "mimic-cxr-2.0.0-chexpert.csv.gz"),
                      "negbio": pd.read_csv(tables / "mimic-cxr-2.0.0-negbio.csv.gz")}
            lab, extra_stats = build_chexbert_labels_v2(src, meta, others)
            extra_inputs = [qba_path, tables / "mimic-cxr-2.0.0-chexpert.csv.gz", tables / "mimic-cxr-2.0.0-negbio.csv.gz"]
    else:
        src = pd.read_csv(tables / "mimic-cxr-2.0.0-chexpert.csv.gz")
        lab = build_labels(src, pd.read_csv(tables / "mimic-cxr-2.0.0-negbio.csv.gz"))
    no_finding = src.set_index("study_id")["No Finding"].reindex(lab.study_id).to_numpy() == 1.0

    stats = {"labeler": args.labeler, "studies": len(lab), "per_finding": {},
             "validation_vs_radiologist": validate(lab, rad)} | extra_stats
    for f in FINDINGS:
        vc = lab[f].value_counts()
        stats["per_finding"][f] = {
            k: int(vc.get(k, 0)) for k in (POS, NEG, UNK)
        } | {"unk_with_no_finding": int(((lab[f] == UNK).to_numpy() & no_finding).sum())}

    out = config.path("work_root") / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    name = "labels" if args.labeler == "chexbert_v2" else f"labels_{args.labeler}"
    inputs = ([src_path, tables / "mimic-cxr-2.0.0-metadata.csv.gz"] + extra_inputs if args.labeler.startswith("chexbert") else
              [tables / "mimic-cxr-2.0.0-chexpert.csv.gz", tables / "mimic-cxr-2.0.0-negbio.csv.gz"])
    write_table(lab, out / f"{name}.parquet", step="2_labels", params={"labeler": args.labeler},
                inputs=inputs + [tables / "mimic-cxr-2.1.0-test-set-labeled.csv"], git=git, stats=stats)
    (out / f"{name}.stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
