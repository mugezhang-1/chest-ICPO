"""Pipeline step 2 (roadmap §3.2.2): per-study presence labels for the eight target findings.

Every labeler is first expressed in the CheXpert CSV encoding (1.0 positive, 0.0 negative, -1.0 uncertain,
NaN blank). A rule then combines one or more labelers into pos / neg / unk. Under every rule, uncertain and
blank never count as a vote, so silence is never absence: a blank cell stays unk even when "No Finding" is 1.0.
"""

import numpy as np
import pandas as pd

FINDINGS = [
    "Pleural Effusion", "Edema", "Consolidation", "Atelectasis",
    "Pneumothorax", "Cardiomegaly", "Lung Opacity", "Support Devices",
]
POS, NEG, UNK = "pos", "neg", "unk"


def agree(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.Series(UNK, index=a.index, dtype=object)
    out[(a == 1.0) & (b == 1.0)] = POS
    out[(a == 0.0) & (b == 0.0)] = NEG
    return out


def vote(cols: list[pd.Series], min_votes: int) -> pd.Series:
    """pos if >= min_votes labelers say 1.0 and none says 0.0; neg symmetrically; else unk."""
    pos = sum((c == 1.0).astype(int) for c in cols)
    neg = sum((c == 0.0).astype(int) for c in cols)
    out = pd.Series(UNK, index=cols[0].index, dtype=object)
    out[(pos >= min_votes) & (neg == 0)] = POS
    out[(neg >= min_votes) & (pos == 0)] = NEG
    return out


def combine(sources: dict[str, pd.DataFrame], names: list[str], min_votes: int | None = None) -> pd.DataFrame:
    """Labels from the named sources (each: study_id + FINDINGS in CheXpert encoding).

    min_votes=None means all named sources must agree. Studies missing from a source count as blank there.
    """
    ids = sorted(set().union(*(set(sources[n].study_id) for n in names)))
    out = pd.DataFrame({"study_id": ids})
    aligned = {n: sources[n].set_index("study_id").reindex(ids) for n in names}
    k = len(names) if min_votes is None else min_votes
    for f in FINDINGS:
        out[f] = vote([aligned[n][f].reset_index(drop=True) for n in names], k).to_numpy()
    return out


def build_chexbert_labels(chexbert: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    """Main rule (decided 2026-10-06): CheXbert alone, explicit 1.0/0.0 only.

    chexbert: study_id + FINDINGS in CheXpert encoding; subjects: study_id -> subject_id.
    One row per study: subject_id, study_id, one column per finding.
    """
    lab = combine({"chexbert": chexbert}, ["chexbert"])
    sub = subjects.drop_duplicates("study_id").set_index("study_id")["subject_id"]
    if not lab.study_id.isin(sub.index).all():
        raise ValueError("CheXbert labels contain studies missing from the metadata")
    lab.insert(0, "subject_id", sub.reindex(lab.study_id).to_numpy())
    return lab


# Labels v2 (docs/label_fix_design.md, 2026-10-11)
LO_CHILDREN = ("Edema", "Consolidation", "Atelectasis", "Pneumonia", "Lung Lesion")


def veto(lab: pd.DataFrame, others: dict[str, pd.DataFrame], findings) -> tuple[pd.DataFrame, dict]:
    """R3: an explicit pos / neg in lab (study_id + findings, pos/neg/unk) becomes unk where any other source
    (study_id + findings, CheXpert encoding) states the explicit opposite. Sources lacking a column are skipped.
    Returns the new table and, per finding and source, how many labels were vetoed."""
    out = lab.copy()
    counts = {}
    for f in findings:
        hit = pd.Series(False, index=out.index)
        for name, src in others.items():
            if f not in src:
                continue
            v = src.drop_duplicates("study_id").set_index("study_id")[f].reindex(out.study_id).to_numpy()
            h = ((out[f].to_numpy() == POS) & (v == 0.0)) | ((out[f].to_numpy() == NEG) & (v == 1.0))
            counts[f"{f}/{name}"] = int(h.sum())
            hit |= h
        counts[f] = int(hit.sum())
        out.loc[hit.to_numpy(), f] = UNK
    return out, counts


def lung_opacity_hierarchy(lab: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """R2 (revised 2026-10-11): a neg Lung Opacity contradicts a pos child (LO_CHILDREN) and becomes unk; no
    label is created. Returns the table and the number of studies changed."""
    out = lab.copy()
    child = (out[[c for c in LO_CHILDREN if c in out]] == POS).any(axis=1)
    changed = child & (out["Lung Opacity"] == NEG)
    out.loc[changed, "Lung Opacity"] = UNK
    return out, int(changed.sum())


def build_chexbert_labels_v2(chexbert: pd.DataFrame, subjects: pd.DataFrame, others: dict[str, pd.DataFrame]):
    """Labels v2: CheXbert explicit labels, cross-labeler veto (R3), then the Lung Opacity hierarchy (R2).
    Returns (labels with FINDINGS columns, stats)."""
    cols = list(dict.fromkeys(FINDINGS + list(LO_CHILDREN)))
    lab = pd.DataFrame({"study_id": chexbert.study_id.to_numpy()})
    for f in cols:
        lab[f] = vote([chexbert[f].reset_index(drop=True)], 1).to_numpy()
    lab, vetoed = veto(lab, others, cols)
    lab, lifted = lung_opacity_hierarchy(lab)
    sub = subjects.drop_duplicates("study_id").set_index("study_id")["subject_id"]
    if not lab.study_id.isin(sub.index).all():
        raise ValueError("CheXbert labels contain studies missing from the metadata")
    lab.insert(0, "subject_id", sub.reindex(lab.study_id).to_numpy())
    return lab[["subject_id", "study_id"] + FINDINGS], {"vetoed": vetoed, "lung_opacity_neg_with_pos_child": lifted}


def build_labels(chexpert: pd.DataFrame, negbio: pd.DataFrame) -> pd.DataFrame:
    """Ablation rule: CheXpert and NegBio agree. One row per study: subject_id, study_id, one column per finding."""
    keys = ["subject_id", "study_id"]
    m = chexpert.merge(negbio, on=keys, how="outer", suffixes=("_cx", "_nb"), validate="one_to_one")
    out = m[keys].copy()
    for f in FINDINGS:
        out[f] = agree(m[f"{f}_cx"], m[f"{f}_nb"])
    return out


# --- MIMIC-Ext-CXR-QBA observations -> CheXpert encoding ---------------------------------------------------

QBA_ENTITIES = {
    "Pleural Effusion": {"pleural effusion"},
    "Edema": {"pulmonary edema"},
    "Consolidation": {"consolidation"},
    "Atelectasis": {"atelectasis"},
    "Pneumothorax": {"pneumothorax"},
    "Cardiomegaly": {"cardiomegaly"},
    "Lung Opacity": {"lung opacity", "airspace opacity", "hazy opacity", "groundglass opacity"},
}
# CheXpert's "Support Devices" covers tubes, lines, catheters, pacemakers and hardware
QBA_DEVICE_SUBCATEGORIES = {"TUBES_LINES_PORTS", "IMPLANT", "OTHER_DEVICES", "DEVICE_PART"}


def qba_observation_values(obs: pd.DataFrame) -> pd.Series:
    """Per-observation CheXpert value: certain pos -> 1, certain neg -> 0, likely/uncertain -> -1;
    comparison-only and recommendation observations -> NaN (not a statement about presence)."""
    v = pd.Series(np.nan, index=obs.index)
    certain = obs.certainty == "certain"
    hedged = obs.certainty.isin(["likely", "uncertain"])
    v[(obs.positiveness == "pos") & certain] = 1.0
    v[(obs.positiveness == "neg") & certain] = 0.0
    v[obs.positiveness.isin(["pos", "neg"]) & hedged] = -1.0
    return v


def build_qba(obs: pd.DataFrame) -> pd.DataFrame:
    """Study-level labels from QBA's all_obs table, aggregated like CheXpert: any positive > any uncertain >
    any negative > blank."""
    obs = obs.assign(value=qba_observation_values(obs),
                     ents=obs.entities.fillna("").str.split(";"),
                     subcats=obs.subcategories.fillna("").str.split(";"))
    ids = np.sort(obs.study_id.unique())
    out = pd.DataFrame({"study_id": ids})
    rank = {1.0: 3, -1.0: 2, 0.0: 1}
    for f in FINDINGS:
        if f == "Support Devices":
            hit = obs.subcats.map(lambda s: bool(QBA_DEVICE_SUBCATEGORIES.intersection(s)))
        else:
            hit = obs.ents.map(lambda s, names=QBA_ENTITIES[f]: bool(names.intersection(s)))
        sel = obs.loc[hit & obs.value.notna(), ["study_id", "value"]]
        best = sel.assign(r=sel.value.map(rank)).groupby("study_id").r.max()
        inv = {r: v for v, r in rank.items()}
        out[f] = best.map(inv).reindex(ids).to_numpy()
    return out
