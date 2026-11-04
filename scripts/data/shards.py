"""Permanent copy of the resized images as uncompressed zip shards on project storage.

  pack     build one shard per (batch, pXX prefix) from <manifest>.resized.parquet, then verify it
  verify   re-read every shard member and compare SHA-256 with the resize record
  restore  extract into images_root any image that is missing or whose bytes differ

Shards are written to a temp name and renamed only after verification, so a shard on disk is complete.
Index: <shards_root>/<manifest>.shard_index.parquet (relpath -> shard, sha256).
"""

import argparse
import hashlib
import os
import zipfile
from pathlib import Path

import pandas as pd

from icpo import config
from icpo.data.images import sha256


def load_records(manifest: str) -> pd.DataFrame:
    mdir = config.path("work_root") / "manifests"
    man = pd.read_csv(mdir / f"{manifest}.csv", usecols=["relpath", "batch"])
    rec = pd.read_parquet(mdir / f"{manifest}.resized.parquet")
    rec = rec[rec.error.isna()][["relpath", "sha256"]]
    df = man.merge(rec, on="relpath")
    df["shard"] = [f"{manifest}_b{b}_{r.split('/')[1]}.zip" for b, r in zip(df.batch, df.relpath)]
    return df


def verify_shard(path: Path, expected: dict[str, str]) -> list[str]:
    problems = []
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if names != set(expected):
            problems.append(f"{path.name}: member set differs ({len(names)} vs {len(expected)})")
        for n in names & set(expected):
            if hashlib.sha256(z.read(n)).hexdigest() != expected[n]:
                problems.append(f"{path.name}: {n} checksum mismatch")
    return problems


def cmd_pack(df, root, out_dir, force):
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = []
    for shard, g in df.groupby("shard"):
        dst = out_dir / shard
        expected = dict(zip(g.relpath, g.sha256))
        if dst.exists() and not force:
            p = verify_shard(dst, expected)
            if not p:
                print(f"{shard}: up to date ({len(g)} images)")
                continue
            print(f"{shard}: stale, rebuilding ({p[0]})")
        tmp = dst.with_name(f".{dst.name}.tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for rel in sorted(g.relpath):
                z.write(root / rel, arcname=rel)
        p = verify_shard(tmp, expected)
        if p:
            tmp.unlink()
            problems += p
            continue
        os.replace(tmp, dst)
        print(f"{shard}: packed {len(g)} images, {dst.stat().st_size / 1e9:.2f} GB")
    return problems


def cmd_verify(df, out_dir):
    problems = []
    for shard, g in df.groupby("shard"):
        if not (out_dir / shard).exists():
            problems.append(f"{shard}: missing")
            continue
        problems += verify_shard(out_dir / shard, dict(zip(g.relpath, g.sha256)))
    print(f"verified {df.shard.nunique()} shards, {len(df)} images")
    return problems


def cmd_restore(df, root, out_dir):
    restored = 0
    for shard, g in df.groupby("shard"):
        todo = [r for r, h in zip(g.relpath, g.sha256) if not (root / r).exists() or sha256(root / r) != h]
        if not todo:
            continue
        with zipfile.ZipFile(out_dir / shard) as z:
            for rel in todo:
                dst = root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(f".{dst.name}.tmp")
                tmp.write_bytes(z.read(rel))
                os.replace(tmp, dst)
        restored += len(todo)
    print(f"restored {restored} images into {root}")
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "verify", "restore"])
    ap.add_argument("--manifest", default="images_v1")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    df = load_records(args.manifest)
    if args.batch is not None:
        df = df[df.batch == args.batch]
    root, out_dir = config.path("images_root"), config.path("shards_root")

    if args.cmd == "pack":
        problems = cmd_pack(df, root, out_dir, args.force)
        full = load_records(args.manifest)
        full = full[[(out_dir / s).exists() for s in full.shard]]
        full.to_parquet(out_dir / f"{args.manifest}.shard_index.parquet", index=False)
    elif args.cmd == "verify":
        problems = cmd_verify(df, out_dir)
    else:
        problems = cmd_restore(df, root, out_dir)

    for p in problems[:20]:
        print("PROBLEM", p)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
