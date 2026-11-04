"""Perceptual hashes of every stored image, for choosing v4 donors (roadmap §3.2.7).

phash and dhash (64-bit, imagehash defaults) of the 1024 px copy, plus width / height. Output:
<work_root>/manifests/images_v1.hashes.parquet (+ provenance), one row per manifest image, sorted by relpath.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import imagehash
import pandas as pd
from PIL import Image

from icpo import config
from icpo.provenance import add_cli, require_clean_tree, write_table


def one(path):
    with Image.open(path) as im:
        im.load()
        return str(imagehash.phash(im)), str(imagehash.dhash(im)), im.width, im.height


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()
    git = require_clean_tree(args.allow_dirty)
    mdir = config.path("work_root") / "manifests"
    root = config.path("images_root")
    man = pd.read_csv(mdir / "images_v1.csv", usecols=["relpath", "dicom_id", "subject_id", "study_id", "ViewPosition"])
    man = man.sort_values("relpath").reset_index(drop=True)
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, [root / r for r in man.relpath], chunksize=256))
    man[["phash", "dhash", "width", "height"]] = pd.DataFrame(res, columns=["phash", "dhash", "width", "height"])
    write_table(man, mdir / "images_v1.hashes.parquet", step="image_hashes", git=git,
                params={"hash": "imagehash.phash/dhash default 8x8", "imagehash": imagehash.__version__},
                inputs=[mdir / "images_v1.csv", mdir / "images_v1.resized.parquet"])
    print(man.head(3).to_string(), len(man))


if __name__ == "__main__":
    main()
