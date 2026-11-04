"""Image paths, checksums and the one offline resize every downstream step depends on."""

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

JPG_BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"


def mimic_relpath(subject_id: int, study_id: int, dicom_id: str) -> str:
    """Path inside the MIMIC-CXR-JPG release, e.g. files/p10/p10000032/s50414267/<dicom>.jpg."""
    s = str(int(subject_id))
    return f"files/p{s[:2]}/p{s}/s{int(study_id)}/{dicom_id}.jpg"


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_sha256sums(path: str | Path) -> dict[str, str]:
    """PhysioNet SHA256SUMS.txt -> {relpath: hexdigest}."""
    out = {}
    with open(path) as f:
        for line in f:
            digest, rel = line.rstrip("\n").split(" ", 1)
            out[rel.strip()] = digest
    return out


def target_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Size after scaling the longest edge to `long_side`, preserving aspect ratio. Never upscales."""
    if max(width, height) <= long_side:
        return width, height
    scale = long_side / max(width, height)
    if width >= height:
        return long_side, max(1, round(height * scale))
    return max(1, round(width * scale)), long_side


def to_grayscale(img: Image.Image) -> Image.Image:
    """Return an 8-bit single-channel image, refusing anything that would lose information."""
    if img.mode == "L":
        return img
    if img.mode == "RGB":
        a = np.asarray(img)
        if (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all():
            return img.convert("L")
        raise ValueError("RGB image with unequal channels; not a plain grayscale radiograph")
    raise ValueError(f"unexpected image mode {img.mode!r}")


def resize_file(src: str | Path, dst: str | Path, long_side: int = 1024, quality: int = 95) -> dict:
    """Resize one radiograph and write it atomically.

    Images already within `long_side` are copied byte-for-byte rather than re-encoded.
    The written file is decoded again before returning, so a returned record means a readable image.
    """
    src, dst = Path(src), Path(dst)
    with Image.open(src) as im:
        im.load()
        orig_mode = im.mode
        img = to_grayscale(im)
    w, h = img.size
    nw, nh = target_size(w, h, long_side)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}")
    try:
        if (nw, nh) == (w, h) and orig_mode == "L":
            shutil.copyfile(src, tmp)
            copied = True
        else:
            out = img.resize((nw, nh), Image.Resampling.LANCZOS, reducing_gap=None)
            out.save(tmp, "JPEG", quality=quality, optimize=True)
            copied = False
        with Image.open(tmp) as chk:
            chk.load()
            if chk.size != (nw, nh) or chk.mode != "L":
                raise ValueError(f"written image is {chk.size} {chk.mode}, expected {(nw, nh)} L")
            a = np.asarray(chk, dtype=np.float32)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()

    return {
        "orig_w": w, "orig_h": h, "w": nw, "h": nh, "copied": copied,
        "mean": float(a.mean()), "std": float(a.std()),
        "sha256": sha256(dst),
    }
