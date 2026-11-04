"""Check downloaded originals against PhysioNet's SHA256SUMS and the metadata's Rows/Columns.

Files that fail are moved to <originals_root>/_quarantine so a re-run of the download fetches them again.
Writes <work_root>/manifests/<manifest>.originals_status.parquet (one row per manifest entry).
"""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from icpo import config
from icpo.data.images import sha256


def check(args):
    root, rel, expected, rows, cols = args
    p = Path(root) / rel
    if not p.exists():
        return rel, "missing"
    if sha256(p) != expected:
        return rel, "bad_checksum"
    try:
        with Image.open(p) as im:
            im.load()
            if im.size != (cols, rows):  # PIL size is (width, height); metadata Rows = height
                return rel, f"bad_size_{im.size[0]}x{im.size[1]}"
    except Exception as e:  # noqa: BLE001
        return rel, f"unreadable_{type(e).__name__}"
    return rel, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="images_v1")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    mdir = config.path("work_root") / "manifests"
    man = pd.read_csv(mdir / f"{args.manifest}.csv")
    if args.batch is not None:
        man = man[man.batch == args.batch]
    root = config.path("originals_root")

    jobs = [(str(root), r.relpath, r.sha256, int(r.Rows), int(r.Columns)) for r in man.itertuples()]
    with ProcessPoolExecutor(args.workers) as ex:
        res = dict(ex.map(check, jobs, chunksize=64))

    status = pd.DataFrame({"relpath": list(res), "status": list(res.values())})
    quarantine = root / "_quarantine"
    for rel in status.loc[~status.status.isin(["ok", "missing"]), "relpath"]:
        dst = quarantine / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(root / rel, dst)

    out = mdir / f"{args.manifest}.originals_status.parquet"
    if out.exists():  # keep results for batches not checked in this run
        old = pd.read_parquet(out)
        status = pd.concat([old[~old.relpath.isin(status.relpath)], status])
    status.to_parquet(out, index=False)
    checked = status[status.relpath.isin(man.relpath)]
    print(checked.status.value_counts().to_string())


if __name__ == "__main__":
    main()
