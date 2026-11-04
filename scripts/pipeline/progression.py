"""Pipeline step 3 (roadmap §3.2.3): interval progression per (current study, finding) from Chest ImaGenome.

Silver scene graphs are read directly from scene_graph.zip (never extracted: ~243k files). A comparison relation
is kept only if the prior image it was made against belongs to the study that precedes the current one in our
timeline; otherwise it describes a different interval. Validation: aggregated silver labels vs aggregated gold
comparison relations (roadmap §3.3).

Output: <work_root>/pipeline/progression.parquet (study_id, finding, change, n_regions, n_votes) and
progression.stats.json.
"""

import argparse
import ast
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from icpo import config
from icpo.provenance import add_cli, require_clean_tree, write_table
from icpo.data.progression import aggregate, label_map

ZIP = config.path("physionet_root") / "chest-imagenome" / "1.0.0" / "silver_dataset" / "scene_graph.zip"


def parse_chunk(names):
    rows = []
    with zipfile.ZipFile(ZIP) as z:
        for n in names:
            g = json.loads(z.read(n))
            sid = int(str(g["study_id"]).lstrip("s"))
            for r in g.get("relationships", []):
                changes = tuple(x.split("|")[2] for x in r.get("relationship_names", [])
                                if x.startswith("comparison|yes|"))
                if not changes:
                    continue
                labels = [a.split("|")[2] for a in r.get("attributes", []) if "|yes|" in a]
                prior_image = r.get("object_id", "").split("_")[0]
                for lab in labels:
                    rows.append((sid, r.get("bbox_name"), lab, changes, prior_image))
    return rows


def main():
    args = add_cli(argparse.ArgumentParser()).parse_args()
    git = require_clean_tree(args.allow_dirty)
    work = config.path("work_root") / "pipeline"
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
    chunks = [names[i:i + 2000] for i in range(0, len(names), 2000)]
    with ProcessPoolExecutor(16) as ex:
        rows = [r for part in ex.map(parse_chunk, chunks) for r in part]
    raw = pd.DataFrame(rows, columns=["study_id", "region", "label_name", "changes", "prior_image"])

    # keep relations whose prior image belongs to our timeline's previous study
    meta = pd.read_csv(config.path("mimic_jpg_tables") / "mimic-cxr-2.0.0-metadata.csv.gz",
                       usecols=["dicom_id", "study_id"])
    img2study = dict(zip(meta.dicom_id, meta.study_id))
    tl = pd.read_parquet(work / "timelines.parquet").sort_values(["subject_id", "idx"])
    prev = dict(zip(tl.study_id, tl.groupby("subject_id").study_id.shift(1)))
    raw["prior_study"] = raw.prior_image.map(img2study)
    raw["our_prev"] = raw.study_id.map(prev)
    lm = label_map()
    mapped = raw[raw.label_name.isin(lm)]
    match = mapped.prior_study.notna() & (mapped.prior_study == mapped.our_prev)
    kept = mapped[match]
    silver = aggregate(kept[["study_id", "region", "label_name", "changes"]].drop_duplicates(
        subset=["study_id", "region", "label_name", "changes"]))

    # validation against gold comparison relations
    gpath = config.path("physionet_root") / "chest-imagenome" / "1.0.0" / "gold_dataset" / \
        "gold_comparison_relations_500pts_500studies2nd.txt"
    g = pd.read_csv(gpath, sep="\t")
    gold = aggregate(pd.DataFrame({
        "study_id": g.study_id, "region": g.bbox, "label_name": g.label_name,
        "changes": g.comparison.map(lambda s: tuple(ast.literal_eval(s)) if isinstance(s, str) and s.startswith("[") else ())}))
    m = gold.merge(silver, on=["study_id", "finding"], how="left", suffixes=("_gold", "_silver"))
    m = m[m.change_gold != "unk"]
    m["change_silver"] = m.change_silver.fillna("missing")
    has = m.change_silver.isin(["improved", "worsened", "no change"])
    per = m.groupby("finding").apply(lambda d: pd.Series({
        "gold": len(d), "silver_present": int(d.change_silver.isin(["improved", "worsened", "no change"]).sum()),
        "agree": int((d.change_silver == d.change_gold).sum())}), include_groups=False)
    per["accuracy_when_present"] = (per.agree / per.silver_present).round(3)

    stats = {
        "scene_graphs": len(names),
        "comparison_rows": len(raw),
        "rows_with_mapped_finding": len(mapped),
        "mapped_rows_prior_matches_our_previous_study": round(float(match.mean()), 4),
        "mapped_rows_prior_unknown": int(mapped.prior_study.isna().sum()),
        "unmapped_label_names_top": raw.loc[~raw.label_name.isin(lm), "label_name"].value_counts().head(15).to_dict(),
        "silver_labels": silver.change.value_counts().to_dict(),
        "silver_by_finding": {f: d.change.value_counts().to_dict() for f, d in silver.groupby("finding")},
        "gold_validation": {
            "gold_labels": len(m), "silver_present_rate": round(float(has.mean()), 3),
            "accuracy_when_present": round(float((m.change_silver == m.change_gold)[has].mean()), 3),
            "confusion": pd.crosstab(m.change_gold, m.change_silver).to_dict(),
            "per_finding": per.reset_index().to_dict(orient="records"),
        },
    }
    (work / "progression.stats.json").write_text(json.dumps(stats, indent=2, default=int))
    write_table(silver, work / "progression.parquet", step="3_progression",
                params={"label_map": "configs/imagenome_findings.yaml", "prior_must_match_previous_study": True},
                inputs=[ZIP, gpath, work / "timelines.parquet", config.path("mimic_jpg_tables") /
                        "mimic-cxr-2.0.0-metadata.csv.gz"], git=git, stats=json.loads(json.dumps(stats, default=int)))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()
