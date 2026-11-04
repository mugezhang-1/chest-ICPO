"""Resize verified originals to the storage resolution in configs/vision.yaml.

Only originals whose status is "ok" in <manifest>.originals_status.parquet are processed; entries already
recorded in <manifest>.resized.parquet with an existing output file are skipped, so re-runs are cheap.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from icpo import config
from icpo.data.images import resize_file


def work(rel, src_root, dst_root, long_side, quality):
    try:
        rec = resize_file(Path(src_root) / rel, Path(dst_root) / rel, long_side, quality)
        return {"relpath": rel, "error": None, **rec}
    except Exception as e:  # noqa: BLE001
        return {"relpath": rel, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="images_v1")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    st = config.load("vision")["storage"]
    assert st["format"] == "jpeg" and st["resample"] == "lanczos" and st["mode"] == "L"
    mdir = config.path("work_root") / "manifests"
    src_root, dst_root = config.path("originals_root"), config.path("images_root")

    man = pd.read_csv(mdir / f"{args.manifest}.csv")
    if args.batch is not None:
        man = man[man.batch == args.batch]
    status = pd.read_parquet(mdir / f"{args.manifest}.originals_status.parquet")
    ok = set(status.loc[status.status == "ok", "relpath"])

    out = mdir / f"{args.manifest}.resized.parquet"
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["relpath", "error"])
    done_ok = {r for r in done.loc[done.error.isna(), "relpath"] if (dst_root / r).exists()}
    todo = [r for r in man.relpath if r in ok and r not in done_ok]
    print(f"{len(man)} in scope, {len(ok & set(man.relpath))} verified originals, {len(todo)} to resize")

    fn = partial(work, src_root=str(src_root), dst_root=str(dst_root),
                 long_side=st["long_side"], quality=st["jpeg_quality"])
    with ProcessPoolExecutor(args.workers) as ex:
        recs = list(ex.map(fn, todo, chunksize=32))

    new = pd.DataFrame(recs)
    if len(new):
        new["long_side"], new["jpeg_quality"] = st["long_side"], st["jpeg_quality"]
    keep = done[~done.relpath.isin(new.relpath)] if len(new) else done
    res = pd.concat([keep, new], ignore_index=True)
    res.to_parquet(out, index=False)
    errs = new[new.error.notna()] if len(new) else new
    print(f"resized {len(new) - len(errs)}, errors {len(errs)}")
    if len(errs):
        print(errs.error.value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
