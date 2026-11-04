"""Provenance for every derived table: what code, inputs and parameters produced it, and a content hash.

Each output `<name>.parquet` gets a sibling `<name>.provenance.json`. The content hash depends only on the table's
values (row order included), not on parquet encoding, so two runs of a deterministic step must give equal hashes.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from icpo.config import REPO_ROOT

TRACKED_PACKAGES = ["numpy", "pandas", "pyarrow", "pillow", "torch", "transformers", "f1chexbert", "vllm"]


def _cell(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple, dict)):
        return json.dumps(x, sort_keys=True, default=str)
    return x


def content_hash(df: pd.DataFrame) -> str:
    """SHA-256 over column names, dtypes-independent values and row order."""
    h = hashlib.sha256(json.dumps(list(map(str, df.columns))).encode())
    for c in df.columns:
        col = df[c]
        if col.dtype == object or str(col.dtype) in ("str", "string"):
            col = col.map(_cell)
        vals = col.astype(str).where(col.notna(), "<NA>") if col.dtype == object else col
        h.update(pd.util.hash_pandas_object(vals, index=False).values.tobytes())
    return h.hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(dirty), "dirty_files": dirty.splitlines()[:20]}


def package_versions() -> dict:
    out = {}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def require_clean_tree(allow_dirty: bool = False) -> dict:
    """Official outputs must come from committed code; pass allow_dirty=True only for scratch runs."""
    g = git_state()
    if g["dirty"] and not allow_dirty:
        raise SystemExit("refusing to write an official output from uncommitted code "
                         f"(dirty: {g['dirty_files'][:5]}); commit first or pass --allow-dirty")
    return g


def write_table(df: pd.DataFrame, path: str | Path, *, step: str, params: dict, inputs: list,
                git: dict, stats: dict | None = None) -> dict:
    """Write df to parquet plus provenance JSON. `inputs` are file paths (hashed) or provenance dicts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    ins = []
    for x in inputs:
        p = Path(x)
        prov = p.with_suffix(".provenance.json") if p.suffix == ".parquet" else None
        entry = {"path": str(p), "sha256": file_sha256(p)}
        if prov is not None and prov.exists():
            entry["content_hash"] = json.loads(prov.read_text())["content_hash"]
        ins.append(entry)
    rec = {
        "step": step,
        "output": str(path),
        "rows": len(df),
        "columns": list(map(str, df.columns)),
        "content_hash": content_hash(df),
        "params": params,
        "inputs": ins,
        "git": git,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": stats or {},
    }
    path.with_suffix(".provenance.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


def add_cli(ap):
    ap.add_argument("--allow-dirty", action="store_true",
                    help="write outputs even if the git tree has uncommitted changes (scratch runs only)")
    return ap
