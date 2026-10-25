"""Dataset download, placement and inventory helpers."""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import pandas as pd
import requests

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MENDELEY_FILES_API = "https://data.mendeley.com/public-api/datasets/{id}/files?folder_id=root&version={version}"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def mendeley_files(dataset_id: str, version: int) -> list[dict]:
    """File records (filename, size, sha256, download_url) for one dataset version."""
    r = requests.get(MENDELEY_FILES_API.format(id=dataset_id, version=version), timeout=60)
    r.raise_for_status()
    out = []
    for f in r.json():
        cd = f["content_details"]
        out.append(
            {
                "filename": f["filename"],
                "size": cd["size"],
                "sha256": cd["sha256_hash"],
                "url": cd["download_url"],
            }
        )
    return out


def download(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    """Download to `dest` (skipped if already present with the right hash)."""
    if dest.exists() and (expected_sha256 is None or sha256(dest) == expected_sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    if expected_sha256 and sha256(tmp) != expected_sha256:
        tmp.unlink()
        raise RuntimeError(f"sha256 mismatch for {dest.name}")
    tmp.rename(dest)
    return dest


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip, refusing members that would escape `out_dir`."""
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.infolist():
            target = (out_dir / m.filename).resolve()
            if not target.is_relative_to(out_dir):
                raise RuntimeError(f"unsafe path in zip: {m.filename}")
        zf.extractall(out_dir)


def normalise_class(name: str) -> str:
    """'Leaf Crinckle 150' -> 'leaf crinckle' (folder names in the archives carry counts)."""
    name = re.sub(r"[\d()]+", " ", name.lower().replace("_", " ").replace("-", " "))
    return " ".join(name.split())


def place_images(extracted: Path, raw_root: Path, crop: str, class_map: dict) -> dict:
    """Copy images from an extracted archive into raw/<crop>/<label>/<orig_class>/.

    The class of an image is the nearest ancestor folder whose normalised name is a
    key of `class_map`. Classes mapped to None are skipped. Returns counts per class.
    """
    keys = {normalise_class(k): v for k, v in class_map.items()}
    counts: dict[str, int] = {}
    for p in sorted(extracted.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS or p.name.startswith("._") or "__MACOSX" in p.parts:
            continue
        orig = next((normalise_class(a) for a in reversed(p.parent.relative_to(extracted).parts)
                     if normalise_class(a) in keys), None)
        if orig is None:
            counts["<unmapped>"] = counts.get("<unmapped>", 0) + 1
            continue
        label = keys[orig]
        if label is None:
            counts[f"{orig} (excluded)"] = counts.get(f"{orig} (excluded)", 0) + 1
            continue
        dest = raw_root / crop / label / orig.replace(" ", "_") / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(p, dest)
        counts[orig] = counts.get(orig, 0) + 1
    return counts


def build_inventory(raw_root: Path, crops: list[str], labels: list[str], rel_to: Path) -> pd.DataFrame:
    """One row per image file under raw/<crop>/<label>/[<orig_class>/]; paths relative to `rel_to`."""
    rows = []
    for crop in crops:
        for label in labels:
            base = raw_root / crop / label
            for p in sorted(base.rglob("*")):
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue
                rel = p.relative_to(base)
                orig = rel.parts[0] if len(rel.parts) > 1 else label
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                ok = img is not None and img.size > 0
                h, w = (img.shape[:2] if ok else (0, 0))
                rows.append(
                    {
                        "crop": crop,
                        "label": label,
                        "orig_class": orig,
                        "path": str(p.relative_to(rel_to)),
                        "width": w,
                        "height": h,
                        "readable": ok,
                    }
                )
    return pd.DataFrame(rows, columns=["crop", "label", "orig_class", "path", "width", "height", "readable"])


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, so zipfile can extract single
    members from a large remote archive without downloading all of it."""

    def __init__(self, url: str, block: int = 1 << 20):
        r = requests.head(url, allow_redirects=True, timeout=60)
        self.url = r.url
        self.size = int(r.headers["Content-Length"])
        self.pos, self.block, self._cache = 0, block, {}

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _get_block(self, i: int) -> bytes:
        if i not in self._cache:
            lo, hi = i * self.block, min((i + 1) * self.block, self.size) - 1
            r = requests.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
            r.raise_for_status()
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[i] = r.content
        return self._cache[i]

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, off = divmod(self.pos, self.block)
            chunk = self._get_block(i)[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)
