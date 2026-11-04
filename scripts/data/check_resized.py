"""QA gate for the resized images. Exits non-zero if any check fails.

Full pass (every in-scope image):
  completeness  every verified original has a resized file and an error-free record
  geometry      size == target_size(original); long side exactly long_side; aspect within rounding
  content       not blank (std above a floor)
Sample pass (seeded, originals required):
  fidelity      PSNR of the stored JPEG vs an in-memory float LANCZOS resize of the original
  intensity     mean gray level preserved
  model input   Qwen3-VL processor at configs/vision.yaml budgets, fed the original vs the stored copy:
                same token grid, and near-identical pixel values. Also reported at larger budgets to
                show where the 1024 storage size starts to cap resolution.
Report: <work_root>/qa/<manifest>[.b<batch>].resized_qa.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image

from icpo import config
from icpo.data.images import target_size

STD_FLOOR = 2.0
MIN_FIDELITY_PSNR = 40.0     # stored JPEG vs exact resize, 1st percentile
MAX_MEAN_SHIFT = 1.0         # gray levels
MIN_GRID_MATCH = 0.99        # fraction of sampled images with identical token grid at the configured budget
MIN_MODEL_PSNR = 35.0        # processor output, original vs stored, 1st percentile


def psnr(a, b, peak=255.0):
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(peak**2 / mse))


def model_input(proc, img, min_px, max_px):
    out = proc(images=[img.convert("RGB")], size={"shortest_edge": min_px, "longest_edge": max_px},
               return_tensors="np")
    return tuple(int(x) for x in out["image_grid_thw"][0]), out["pixel_values"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="images_v1")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vis = config.load("vision")
    st, mi = vis["storage"], vis["model_input"]
    mdir = config.path("work_root") / "manifests"
    src_root, dst_root = config.path("originals_root"), config.path("images_root")

    man = pd.read_csv(mdir / f"{args.manifest}.csv")
    if args.batch is not None:
        man = man[man.batch == args.batch]
    status = pd.read_parquet(mdir / f"{args.manifest}.originals_status.parquet")
    ok = set(status.loc[status.status == "ok", "relpath"])
    man = man[man.relpath.isin(ok)]
    rec = pd.read_parquet(mdir / f"{args.manifest}.resized.parquet").set_index("relpath")

    report, failures = {"n_verified_originals": len(man)}, []

    # --- full pass -----------------------------------------------------------------------
    have = man.relpath.isin(rec.index)
    r = rec.reindex(man.relpath)
    missing = (~have) | r.error.notna().to_numpy() | ~np.array([(dst_root / p).exists() for p in man.relpath])
    report["missing_or_error"] = int(missing.sum())
    if missing.any():
        failures.append(f"{int(missing.sum())} images missing a valid resized file")

    exp = [target_size(int(c), int(rw), st["long_side"]) for c, rw in zip(man.Columns, man.Rows)]
    got = list(zip(r.w.fillna(-1).astype(int), r.h.fillna(-1).astype(int)))
    bad_geom = [(p, e, g) for p, e, g, m in zip(man.relpath, exp, got, missing) if not m and e != g]
    report["bad_geometry"] = len(bad_geom)
    if bad_geom:
        failures.append(f"{len(bad_geom)} images with wrong size, e.g. {bad_geom[:3]}")

    blank = r[(r["std"] < STD_FLOOR).fillna(False).to_numpy()]
    report["near_blank"] = blank.index.tolist()[:50]
    report["n_near_blank"] = len(blank)  # reported, not failed: some originals may be blank themselves
    report["copied_without_resize"] = int(r.copied.fillna(False).sum())
    report["stored_bytes"] = int(sum(os.path.getsize(dst_root / p) for p in man.relpath[~missing]))

    # --- sample pass ---------------------------------------------------------------------
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    budgets = {"configured": (mi["min_pixels"], mi["max_pixels"]),
               "2x": (mi["min_pixels"], 2 * mi["max_pixels"]),
               "4x": (mi["min_pixels"], 4 * mi["max_pixels"])}

    pool = [p for p, m in zip(man.relpath, missing) if not m and (src_root / p).exists()]
    sample = random.Random(args.seed).sample(pool, min(args.sample, len(pool)))
    fid, shift, per_budget = [], [], {k: {"grid_match": [], "psnr": []} for k in budgets}
    for p in sample:
        with Image.open(src_root / p) as o, Image.open(dst_root / p) as s:
            o.load(); s.load()
            o = o.convert("L")
            ref = o.resize(s.size, Image.Resampling.LANCZOS, reducing_gap=None) if o.size != s.size else o
            fid.append(psnr(s, ref))
            shift.append(abs(float(np.asarray(s, np.float64).mean() - np.asarray(o, np.float64).mean())))
            for k, (lo, hi) in budgets.items():
                go, po = model_input(proc, o, lo, hi)
                gs, ps = model_input(proc, s, lo, hi)
                per_budget[k]["grid_match"].append(go == gs)
                if go == gs:  # pixel_values are normalized to [-1, 1]
                    per_budget[k]["psnr"].append(psnr(po, ps, peak=2.0))

    def q(x, pct):
        return float(np.percentile(x, pct)) if len(x) else None

    report["sample_size"] = len(sample)
    report["fidelity_psnr"] = {"p1": q(fid, 1), "median": q(fid, 50), "min": min(fid, default=None)}
    report["mean_shift"] = {"max": max(shift, default=None), "median": q(shift, 50)}
    report["model_input"] = {
        k: {"budget_pixels": budgets[k][1],
            "grid_match_rate": float(np.mean(v["grid_match"])) if v["grid_match"] else None,
            "psnr_p1": q(v["psnr"], 1), "psnr_median": q(v["psnr"], 50)}
        for k, v in per_budget.items()
    }

    if sample:
        if report["fidelity_psnr"]["p1"] < MIN_FIDELITY_PSNR:
            failures.append(f"storage fidelity p1 PSNR {report['fidelity_psnr']['p1']:.1f} < {MIN_FIDELITY_PSNR}")
        if report["mean_shift"]["max"] > MAX_MEAN_SHIFT:
            failures.append(f"mean intensity shift {report['mean_shift']['max']:.2f} > {MAX_MEAN_SHIFT}")
        c = report["model_input"]["configured"]
        if c["grid_match_rate"] < MIN_GRID_MATCH:
            failures.append(f"token grid differs from original on {1 - c['grid_match_rate']:.1%} of images")
        if c["psnr_p1"] is not None and c["psnr_p1"] < MIN_MODEL_PSNR:
            failures.append(f"model-input PSNR p1 {c['psnr_p1']:.1f} < {MIN_MODEL_PSNR}")
    else:
        failures.append("no originals available for the sample pass")

    report["failures"] = failures
    qa_dir = config.path("work_root") / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    tag = f".b{args.batch}" if args.batch is not None else ""
    (qa_dir / f"{args.manifest}{tag}.resized_qa.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(config.path("hf_home")))
    main()
