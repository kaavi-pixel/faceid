"""Model download / verification for the two ONNX models we use.

Both come from the official OpenCV model zoo (Apache-2.0), so there is no
extra runtime dependency: OpenCV's own DNN engine runs them.  Total
download is ~39 MB, which matters on a laptop that is short of disk.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx"),
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "size": 232589,
        "what": "YuNet face detector (CNN, occlusion tolerant)",
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_recognition_sface/face_recognition_sface_2021dec.onnx"),
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "size": 38696353,
        "what": "SFace face recogniser, 128-d embeddings",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expect: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def hook(blocks: int, bs: int, total: int) -> None:
        if total <= 0:
            return
        done = min(blocks * bs, total)
        pct = done * 100.0 / total
        sys.stderr.write(f"\r  {dest.name}: {pct:5.1f}%  "
                         f"{done/1e6:6.1f}/{total/1e6:.1f} MB")
        sys.stderr.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    sys.stderr.write("\n")
    if expect and tmp.stat().st_size != expect:
        raise RuntimeError(f"{dest.name}: got {tmp.stat().st_size} bytes, expected {expect}")
    shutil.move(str(tmp), str(dest))


def ensure_models(model_dir: Path, force: bool = False, quiet: bool = False) -> dict:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, spec in MODELS.items():
        target = model_dir / name
        if target.is_file() and not force:
            got = _sha256(target)
            ok = got == spec["sha256"]
            out[name] = {"ok": ok, "path": str(target), "downloaded": False,
                         "sha256": got}
            if not ok and not quiet:
                print(f"  {name}: checksum mismatch, re-downloading", file=sys.stderr)
                target.unlink()
                return ensure_models(model_dir, force=True, quiet=quiet)
            continue
        if not quiet:
            print(f"  fetching {spec['what']} -> {target}", file=sys.stderr)
        _download(spec["url"], target, spec["size"])
        got = _sha256(target)
        if got != spec["sha256"]:
            raise RuntimeError(
                f"{name}: checksum mismatch after download\n"
                f"  expected {spec['sha256']}\n  got      {got}")
        out[name] = {"ok": True, "path": str(target), "downloaded": True,
                     "sha256": got}
    return out


def verify_models(model_dir: Path) -> dict:
    model_dir = Path(model_dir)
    out = {}
    for name, spec in MODELS.items():
        p = model_dir / name
        if not p.is_file():
            out[name] = {"ok": False, "reason": "missing"}
            continue
        got = _sha256(p)
        out[name] = {"ok": got == spec["sha256"], "sha256": got,
                     "size": p.stat().st_size}
    return out
