"""Face detection + alignment using OpenCV's YuNet CNN (ONNX, runs on CPU).

YuNet is chosen over dlib's HOG detector (what Howdy uses) because it keeps
finding the face when the face is partly hidden by a cap brim, headphones,
a hood, or when the head is turned.  HOG needs a full, fully lit, frontal
face; that is exactly the failure the user hit.

We run it at several input scales so both a close-up and a far-away face
are covered, then merge the results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

YUNET_NAME = "face_detection_yunet_2023mar.onnx"


class ModelMissing(RuntimeError):
    pass


@dataclass
class Face:
    box: tuple[int, int, int, int]      # x, y, w, h
    landmarks: np.ndarray               # 5x2 (right eye, left eye, nose, r-mouth, l-mouth)
    score: float
    row: np.ndarray                     # raw YuNet row (needed by alignCrop)

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2.0, y + h / 2.0

    @property
    def area(self) -> int:
        return int(self.box[2] * self.box[3])

    def as_dict(self) -> dict:
        x, y, w, h = self.box
        return {"box": [int(x), int(y), int(w), int(h)], "score": round(self.score, 3)}


class Detector:
    def __init__(self, model_dir: Path | str, cfg_detect: dict, cfg_embed: dict | None = None):
        self.cfg = cfg_detect or {}
        self.model_dir = Path(model_dir)
        path = self.model_dir / YUNET_NAME
        if not path.is_file():
            raise ModelMissing(
                f"face detector model missing: {path}\n"
                f"Run: faceid models --download"
            )
        size = int(self.cfg.get("input_size", 640))
        self.size = (size, size)
        self._det = cv2.FaceDetectorYN.create(
            str(path), "",
            self.size,
            float(self.cfg.get("score_threshold", 0.72)),
            float(self.cfg.get("nms_threshold", 0.3)),
            int(self.cfg.get("top_k", 5000)),
        )

    # ---------------------------------------------------------------
    def _detect_at(self, img: np.ndarray, scale: float, score: float) -> list[np.ndarray]:
        if scale == 1.0:
            in_img = img
        else:
            in_img = cv2.resize(
                img, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA,
            )
        self._det.setInputSize((in_img.shape[1], in_img.shape[0]))
        try:
            self._det.setScoreThreshold(float(score))
        except Exception:
            pass
        _rc, faces = self._det.detect(in_img)
        if faces is None or len(faces) == 0:
            return []
        out = []
        for f in faces:
            f = f.copy().astype(np.float32)
            for i in (0, 1, 2, 3):
                f[i] /= scale
            for i in range(4, 14):
                f[i] /= scale
            out.append(f)
        return out

    def _collect(self, img: np.ndarray, score: float) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        if self.cfg.get("multi_scale", True):
            scales = list(self.cfg.get("scales", [1.0, 1.5, 0.65]))
        else:
            scales = [1.0]
        # Fast path first: at native scale a real face scores ~0.9x.  If we
        # already have a confident detection there is nothing to gain from the
        # extra (expensive) scales, and auth latency matters for sudo.
        head = scales[0] if scales else 1.0
        try:
            rows = self._detect_at(img, float(head), score)
        except cv2.error:
            rows = []
        if rows and max(float(r[14]) for r in rows) >= score:
            return rows
        for s in scales[1:]:
            try:
                rows.extend(self._detect_at(img, float(s), score))
            except cv2.error:
                continue
        return rows

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax, ay, aw, ah = a[0], a[1], a[2], a[3]
        bx, by, bw, bh = b[0], b[1], b[2], b[3]
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 0 else 0.0

    def _nms(self, rows: list[np.ndarray], thr: float = 0.4) -> list[np.ndarray]:
        rows = sorted(rows, key=lambda r: -float(r[14]))
        keep: list[np.ndarray] = []
        for r in rows:
            if all(self._iou(r, k) < thr for k in keep):
                keep.append(r)
        return keep

    def faces(self, gray: np.ndarray, relaxed: bool = False) -> list[Face]:
        """All faces found, best first."""
        img = gray
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        score = float(self.cfg.get("score_threshold", 0.72))
        rows = self._collect(img, score)
        if not rows:
            rows = self._collect(img, float(self.cfg.get("fallback_score_threshold", 0.35)))
        rows = self._nms(rows)
        out = []
        for r in rows:
            x, y, w, h = (int(round(float(r[i]))) for i in range(4))
            w, h = max(1, w), max(1, h)
            lm = np.asarray(r[4:14], dtype=np.float32).reshape(5, 2)
            out.append(Face(box=(x, y, w, h), landmarks=lm, score=float(r[14]), row=r))
        out.sort(key=lambda f: f.area, reverse=True)
        return out

    def largest_face(self, gray: np.ndarray, min_frac: float = 0.02) -> Face | None:
        """The face an authentication attempt should use.

        Prefers the biggest plausible face, but ignores tiny detections
        (a face in a poster on the wall).  If nothing is big enough we still
        return the best detection rather than giving up - being lenient here
        is what keeps a hat-wearing user from being locked out.
        """
        faces = self.faces(gray)
        if not faces:
            return None
        img_area = float(gray.shape[0] * gray.shape[1])
        big = [f for f in faces if f.area / img_area >= min_frac]
        pool = big or faces
        return max(pool, key=lambda f: f.area)
