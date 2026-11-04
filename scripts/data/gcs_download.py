"""Download manifest images from PhysioNet's requester-pays GCS bucket (billed to our project).

  # exact sizes for the whole release (cheap: ~380 list calls); needed once
  python scripts/data/gcs_download.py sizes
  # projected bytes and egress cost per batch; downloads nothing
  python scripts/data/gcs_download.py plan
  # download; refuses to start if the transfer would exceed --max-usd
  python scripts/data/gcs_download.py get --batch 1 --max-usd 6 [--limit 100]

Every file is SHA-256 checked against PhysioNet's checksums before it is renamed into place. Files already
on disk with the bucket's size are skipped, so the command is safe to re-run.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from icpo import config
from icpo.data.gcs import Client

BUCKET = "mimic-cxr-jpg-2.1.0.physionet.org"
BILLING_PROJECT = "icpo-510602"
USD_PER_GIB = 0.12  # GCS internet egress, North America, first TiB (Oct 2026 list price)
GIB = 1024**3


def sizes_path():
    return config.path("work_root") / "manifests" / "bucket_sizes.parquet"


def cmd_sizes(args, client):
    prefixes = [f"files/p{i}/" for i in range(10, 20)]
    with ThreadPoolExecutor(2) as ex:
        parts = list(ex.map(lambda p: list(client.list(p)), prefixes))
    df = pd.DataFrame([x for p in parts for x in p], columns=["relpath", "size"])
    df.to_parquet(sizes_path(), index=False)
    print(f"{len(df)} objects, {df['size'].sum() / GIB:.1f} GiB in files/")


def planned(man, sizes, root):
    m = man.merge(sizes, on="relpath", how="left")
    if m["size"].isna().any():
        raise SystemExit(f"{m['size'].isna().sum()} manifest files not found in the bucket")
    on_disk = [os.path.getsize(root / r) if (root / r).exists() else -1 for r in m.relpath]
    m["todo"] = [d != s for d, s in zip(on_disk, m["size"])]
    return m


def cmd_plan(args, client):
    man = pd.read_csv(config.path("work_root") / "manifests" / f"{args.manifest}.csv")
    m = planned(man, pd.read_parquet(sizes_path()), config.path("originals_root"))
    for b, g in m.groupby("batch"):
        t = g[g.todo]
        print(f"batch {b}: {len(g)} files, {g['size'].sum() / GIB:.1f} GiB total; "
              f"to fetch {len(t)} files, {t['size'].sum() / GIB:.1f} GiB, ~${t['size'].sum() / GIB * USD_PER_GIB:.2f}")


def cmd_get(args, client):
    root = config.path("originals_root")
    mdir = config.path("work_root") / "manifests"
    man = pd.read_csv(mdir / f"{args.manifest}.csv")
    man = man[man.batch == args.batch]
    m = planned(man, pd.read_parquet(sizes_path()), root)
    m = m[m.todo]
    if args.limit:
        m = m.head(args.limit)
    gib = m["size"].sum() / GIB
    cost = gib * USD_PER_GIB
    print(f"planned: {len(m)} files, {gib:.2f} GiB, ~${cost:.2f} (cap ${args.max_usd:.2f})")
    if cost > args.max_usd:
        raise SystemExit("refusing: projected cost exceeds --max-usd")

    log_path = mdir / "gcs_transfer_log.csv"
    new_log = not log_path.exists()
    t0, done_bytes, done, failed = time.time(), 0, 0, []
    with open(log_path, "a", newline="") as logf, ThreadPoolExecutor(args.workers) as ex:
        log = csv.writer(logf)
        if new_log:
            log.writerow(["time", "relpath", "bytes", "status"])
        futs = {ex.submit(client.download, r.relpath, root / r.relpath, r.sha256): r for r in m.itertuples()}
        for f in as_completed(futs):
            r = futs[f]
            try:
                n = f.result()
                done_bytes += n
                done += 1
                log.writerow([int(time.time()), r.relpath, n, "ok"])
            except Exception as e:  # noqa: BLE001
                failed.append((r.relpath, repr(e)))
                log.writerow([int(time.time()), r.relpath, 0, f"error:{type(e).__name__}"])
            if (done + len(failed)) % args.report_every == 0:
                el = time.time() - t0
                print(f"{done}/{len(m)} ok, {len(failed)} failed, {done_bytes / GIB:.2f} GiB, "
                      f"{done_bytes / el / 1e6:.1f} MB/s, ~${done_bytes / GIB * USD_PER_GIB:.2f}", flush=True)
                logf.flush()
    el = time.time() - t0
    print(f"finished: {done} ok, {len(failed)} failed, {done_bytes / GIB:.2f} GiB in {el:.0f}s "
          f"({done_bytes / max(el, 1e-9) / 1e6:.1f} MB/s), ~${done_bytes / GIB * USD_PER_GIB:.2f}")
    for rel, err in failed[:10]:
        print("  FAILED", rel, err)
    sys.exit(1 if failed else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sizes")
    p = sub.add_parser("plan")
    p.add_argument("--manifest", default="images_v1")
    g = sub.add_parser("get")
    g.add_argument("--manifest", default="images_v1")
    g.add_argument("--batch", type=int, required=True)
    g.add_argument("--max-usd", type=float, required=True)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--workers", type=int, default=32)
    g.add_argument("--report-every", type=int, default=500)
    args = ap.parse_args()
    client = Client(BUCKET, BILLING_PROJECT)
    {"sizes": cmd_sizes, "plan": cmd_plan, "get": cmd_get}[args.cmd](args, client)


if __name__ == "__main__":
    main()
