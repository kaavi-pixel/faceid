"""Face embedding using OpenCV's SFace CNN (ONNX, CPU).

Why this matters for the hat/headphones problem: SFace is trained on large
augmented datasets and is markedly more tolerant of occlusion, expression,
pose and lighting than dlib's resnet embedding that Howdy uses.  On top of
the raw model we add two cheap, geometry-preserving robustness tricks:

  * contrast normalisation (CLAHE) and a gamma-corrected variant are embedded
    too and averaged - this is what keeps an evening, lamp-lit shot matching
    a daytime enrolment;
  * embeddings are L2-normalised so cosine distance is well behaved.

Nothing here changes the face geometry, so accuracy cannot regress from the
published model - the variants only add illumination invariance.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .capture import normalize

SFACE_NAME = "face_recognition_sface_2021dec.onnx"


class Embedder:
    def __init__(self, model_dir: Path | str, cfg_embed: dict | None = None):
        self.cfg = cfg_embed or {}
        path = Path(model_dir) / SFACE_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"face recogniser model missing: {path}\nRun: faceid models --download")
        self._rec = cv2.FaceRecognizerSF.create(str(path), "")
        self.dim = 128

    # ------------------------------------------------------------------
    def align(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """Crop + similarity-transform the face to the model's 112x112 input."""
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        try:
            return self._rec.alignCrop(bgr, face_row.astype(np.float32))
        except cv2.error:
            # Older OpenCV wants a 15-length row; pad if needed.
            row = face_row.astype(np.float32).ravel()
            if row.size < 15:
                row = np.concatenate([row, np.zeros(15 - row.size, np.float32)])
            return self._rec.alignCrop(bgr, row)

    def _raw_feature(self, aligned: np.ndarray) -> np.ndarray:
        return np.asarray(self._rec.feature(aligned), dtype=np.float32).ravel()

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        """Embed a 112x112 aligned crop, averaging illumination variants."""
        todo = [aligned]
        if self.cfg.get("clahe", True):
            g = normalize(aligned, clahe=True, gamma=None)
            todo.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        if self.cfg.get("gamma_variants", True):
            g = normalize(aligned, clahe=False, gamma=float(self.cfg.get("gamma", 1.4)))
            todo.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        feats = [self._raw_feature(t) for t in todo]
        v = np.mean(np.stack(feats, axis=0), axis=0)
        return l2norm(v)

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        return self.embed_aligned(self.align(bgr, face_row))

    # ------------------------------------------------------------------
    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = l2norm(a)
        b = l2norm(b)
        return float(np.clip(np.dot(a, b), -1.0, 1.0))


def l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def cosine_matrix(probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Score a probe against a gallery.

    probe   : (D,) or (N,D) - one embedding, or one per captured frame
    gallery : (M,D)         - the enrolled embeddings of one tag
    returns : (M,)          - only when probe is 1-D
              (N,M)         - similarity of every frame to every enrolment
    """
    if gallery is None or np.asarray(gallery).size == 0:
        return np.zeros((0,), dtype=np.float32)
    g = np.asarray(gallery, dtype=np.float32).reshape(len(np.atleast_2d(gallery)), -1)
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    p = np.asarray(probe, dtype=np.float32)
    one_d = p.ndim == 1
    p = p.reshape(1, -1) if one_d else p.reshape(p.shape[0], -1)
    if p.shape[1] != g.shape[1]:
        raise ValueError(
            f"embedding dimension mismatch: probe {p.shape[1]} vs gallery {g.shape[1]} "
            f"- the enrolment was made with a different model or feature layout")
    p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)
    out = (p @ g.T).astype(np.float32)
    return out[0] if one_d else out
