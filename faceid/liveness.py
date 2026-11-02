"""Lightweight liveness / anti-spoof heuristics.

Honest scope: this raises the bar against a phone screen or a printed photo
held up to the camera.  It is NOT a certified presentation-attack defence.
For real assurance the requirement is an IR/structured-light sensor (see
`faceid ir`), because a printed photo cannot emit infrared the way a face
lit by the emitter does.

What we measure, without any extra model:

  * motion band - a live person sitting still still moves a fraction of a
    pixel per frame (breathing, micro-saccades).  A rigid printout glued in
    front of the lens is unnaturally still; a photo waved about is
    unnaturally fast.
  * landmark geometry stability - a screenshot of a face shown on a phone
    that is tilted produces a shearing pattern of landmarks unlike a real
    head turning.
  * optional moire check - photo-of-a-screen detection (off by default,
    tune with `faceid calibrate --liveness`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import headfeatures


@dataclass
class LivenessReport:
    ok: bool = True
    reason: str = "not evaluated"
    motion: float = 0.0
    scale_var: float = 0.0
    frames: int = 0
    moire: float = 0.0
    checks: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason,
                "motion_px": round(self.motion, 3),
                "scale_var": round(self.scale_var, 4),
                "frames": self.frames,
                "moire": round(self.moire, 3),
                "checks": self.checks}


class LivenessTracker:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.points: list[np.ndarray] = []
        self.sizes: list[float] = []
        self.moires: list[float] = []
        self.face_h: list[float] = []

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True))

    def update(self, face, bgr=None) -> None:
        # normalise landmarks by face size so a person walking closer does
        # not read as "motion"
        h = max(1.0, float(face.box[3]))
        c = np.asarray(face.center, dtype=np.float64)
        self.points.append(c / h)
        self.sizes.append(h)
        self.face_h.append(h)
        if bgr is not None and self.cfg.get("moire_check", False):
            self.moires.append(headfeatures._moire_energy(bgr))

    def report(self, min_frames: int = 4) -> LivenessReport:
        rep = LivenessReport(frames=len(self.points))
        if not self.enabled:
            rep.reason = "liveness disabled"
            return rep
        if len(self.points) < min_frames:
            rep.reason = f"only {len(self.points)} frames - liveness not judged"
            rep.ok = True
            return rep
        P = np.stack(self.points, axis=0)
        # per-frame displacement, in units of face height
        disp = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P) > 1 else np.zeros(1)
        motion_norm = float(np.median(disp)) if disp.size else 0.0
        sizes = np.asarray(self.sizes, dtype=np.float64)
        scale_var = float(sizes.std() / max(1.0, sizes.mean()))
        rep.motion = motion_norm
        rep.scale_var = scale_var
        lo = float(self.cfg.get("min_motion", 0.35)) / 100.0
        hi = float(self.cfg.get("max_motion", 60.0)) / 100.0
        rep.checks["min_motion"] = lo
        rep.checks["max_motion"] = hi

        if motion_norm < lo:
            rep.ok = False
            rep.reason = (f"no natural movement detected ({motion_norm*100:.3f}% of face "
                          f"height/frame, need >= {lo*100:.3f}%) - static photo?")
            return rep
        if motion_norm > hi:
            rep.ok = False
            rep.reason = (f"unnaturally large frame-to-frame movement "
                          f"({motion_norm*100:.2f}%) - camera being shaken or a "
                          f"hand-held photo?")
            return rep
        if scale_var > 0.55:
            rep.ok = False
            rep.reason = f"face size jumping wildly (var {scale_var:.2f})"
            return rep
        if self.cfg.get("moire_check", False) and self.moires:
            m = float(np.median(self.moires))
            rep.moire = m
            if m > float(self.cfg.get("moire_threshold", 0.62)):
                rep.ok = False
                rep.reason = f"screen/printout texture signature (moire {m:.2f})"
                return rep
        rep.reason = (f"live subject (motion {motion_norm*100:.3f}% face-height/frame, "
                      f"size var {scale_var:.3f})")
        return rep


def embedding_consistency(embs: np.ndarray, threshold: float = 0.30) -> tuple[float, int]:
    """How internally consistent a set of probe embeddings is.

    A live face produces a tight cluster.  Someone flicking between a photo
    and their real face, or a screen playing a video, does not.
    """
    embs = np.atleast_2d(np.asarray(embs, dtype=np.float32))
    if len(embs) < 2:
        return 1.0, len(embs)
    n = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9)
    sims = n @ n.T
    iu = np.triu_indices(len(embs), k=1)
    vals = sims[iu]
    return float(vals.mean()), int((vals >= threshold).sum())
