"""Write the image manifest: the exact set of MIMIC-CXR-JPG files the project downloads.

batch 1 = every image referenced by MI-CXR (unblocks Exp 1);
batch 2 = remaining images that sit in >= 1 window of 3 consecutive frontal studies with gaps <= 90 days.

The window rule is icpo.data.timeline, the same code pipeline step 1 uses, so the download set and the
pipeline cannot disagree. Output: <work_root>/manifests/images_v1.csv plus a short stats file.
"""

import argparse
import json

import pandas as pd

from icpo import config
from icpo.data.images import load_sha256sums, mimic_relpath
from icpo.data.timeline import build_timelines, window_studies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="images_v1")
    args = ap.parse_args()

    tables = config.path("mimic_jpg_tables")
    meta = pd.read_csv(tables / "mimic-cxr-2.0.0-metadata.csv.gz")
    sums = load_sha256sums(tables / "SHA256SUMS.txt")

    tl = build_timelines(meta)
    win = tl.loc[window_studies(tl, n=3, max_gap_days=90)]

    micxr_paths = set()
    with open(config.path("external_root") / "MI-CXR" / "micxr_test.jsonl") as f:
        for line in f:
            micxr_paths.update(json.loads(line)["images"].values())

    meta["relpath"] = [mimic_relpath(s, st, d) for s, st, d in zip(meta.subject_id, meta.study_id, meta.dicom_id)]
    by_path = meta.set_index("relpath")
    missing = sorted(p for p in micxr_paths if p not in by_path.index)
    if missing:
        raise SystemExit(f"{len(missing)} MI-CXR paths not in metadata, e.g. {missing[:3]}")

    micxr = by_path.loc[sorted(micxr_paths)].reset_index()
    win = win.assign(relpath=[mimic_relpath(s, st, d) for s, st, d in zip(win.subject_id, win.study_id, win.dicom_id)])

    cols = ["relpath", "dicom_id", "subject_id", "study_id", "ViewPosition", "Rows", "Columns"]
    man = pd.concat([micxr[cols], win[cols]]).drop_duplicates("relpath").reset_index(drop=True)
    man["in_micxr"] = man.relpath.isin(micxr_paths)
    man["in_windows"] = man.relpath.isin(set(win.relpath))
    man["batch"] = (~man.in_micxr).astype(int) + 1
    man["sha256"] = man.relpath.map(sums)
    if man.sha256.isna().any():
        raise SystemExit(f"{man.sha256.isna().sum()} manifest files have no PhysioNet checksum")
    man = man.sort_values(["batch", "relpath"]).reset_index(drop=True)

    out_dir = config.path("work_root") / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    man.to_csv(out_dir / f"{args.name}.csv", index=False)

    # How MI-CXR's per-study image choice relates to ours (it matters for comparability).
    tl_pick = set(zip(tl.study_id, tl.dicom_id))
    micxr_studies = micxr.study_id.nunique()
    stats = {
        "metadata_images": len(meta),
        "timeline_studies": len(tl),
        "window_images": len(win),
        "micxr_images": len(micxr),
        "micxr_studies": int(micxr_studies),
        "micxr_view_counts": micxr.ViewPosition.fillna("NA").value_counts().to_dict(),
        "micxr_images_matching_our_pick": int(sum((s, d) in tl_pick for s, d in zip(micxr.study_id, micxr.dicom_id))),
        "micxr_images_also_in_windows": int(man.in_micxr.__and__(man.in_windows).sum()),
        "manifest_total": len(man),
        "manifest_batch_counts": man.batch.value_counts().sort_index().to_dict(),
        "original_megapixels_mean": float((man.Rows * man.Columns).mean() / 1e6),
    }
    (out_dir / f"{args.name}.stats.json").write_text(json.dumps(stats, indent=2, default=int))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()
