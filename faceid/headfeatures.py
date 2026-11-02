"""Head-region feature extraction for accessory awareness.

The point of this module: when someone puts a cap or headphones on, the
region *around* the face changes a lot while the face itself stays the same.
By describing that region numerically we can (a) tell the user is wearing
something and (b) pick the enrolment that was captured under the same
conditions, which is what makes recognition robust instead of flaky.

Everything is derived from the face box - no extra model download, no
training set required, runs in well under a millisecond.
"""

from __future__ import annotations

import cv2
import numpy as np

HEAD_ROI = (0.85, 0.28, 0.34)  # up, down, sideways expansion of the face box
FEATURE_VERSION = 3


def _clip_roi(img: np.ndarray, x0: float, y0: float, x1: float, y1: float):
    h, w = img.shape[:2]
    ix0, iy0 = max(0, int(round(x0))), max(0, int(round(y0)))
    ix1, iy1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if ix1 - ix0 < 4 or iy1 - iy0 < 4:
        return None, (ix0, iy0, ix1, iy1), 0.0
    roi = img[iy0:iy1, ix0:ix1]
    want = max(1.0, (x1 - x0)) * max(1.0, (y1 - y0))
    got = float(roi.shape[0] * roi.shape[1])
    return roi, (ix0, iy0, ix1, iy1), float(min(1.0, got / want))


def regions(gray: np.ndarray, box: tuple[int, int, int, int]) -> dict:
    """All sub-regions we care about, as (roi, coverage) pairs."""
    x, y, w, h = [float(v) for v in box]
    up, down, side = HEAD_ROI
    out = {}
    out["head"] = _clip_roi(gray, x - side * w, y - up * h,
                            x + w + side * w, y + h + down * h)
    out["forehead"] = _clip_roi(gray, x + 0.18 * w, y + 0.02 * h,
                                x + 0.82 * w, y + 0.30 * h)
    out["face"] = _clip_roi(gray, x + 0.10 * w, y + 0.10 * h,
                            x + 0.90 * w, y + 0.95 * h)
    out["ear_l"] = _clip_roi(gray, x - 0.36 * w, y + 0.10 * h,
                             x - 0.01 * w, y + 0.80 * h)
    out["ear_r"] = _clip_roi(gray, x + 1.01 * w, y + 0.10 * h,
                             x + 1.36 * w, y + 0.80 * h)
    out["crown"] = _clip_roi(gray, x + 0.05 * w, y - 0.80 * h,
                             x + 0.95 * w, y + 0.05 * h)
    out["chin"] = _clip_roi(gray, x + 0.15 * w, y + 0.78 * h,
                            x + 0.85 * w, y + 1.15 * h)
    return out


# --------------------------------------------------------------------------
def _skin_fraction(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    m = (cr > 132) & (cr < 180) & (cb > 75) & (cb < 130) & (y > 40)
    return float(m.mean())


def _hsv_hist(bgr: np.ndarray, bins=(12, 6, 6)) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [bins[0]], [0, 180]).ravel()
    s = cv2.calcHist([hsv], [1], None, [bins[1]], [0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [bins[2]], [0, 256]).ravel()
    out = np.concatenate([h / max(1.0, h.sum()), s / max(1.0, s.sum()),
                          v / max(1.0, v.sum())])
    return out.astype(np.float32)


def _sobel_stats(gray: np.ndarray, bins: int = 10) -> np.ndarray:
    if gray is None or gray.size == 0:
        return np.zeros(bins + 13, dtype=np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi  # 0..pi, orientation
    hist_m, _ = np.histogram(mag, bins=bins, range=(0, 400))
    hist_m = hist_m / max(1, hist_m.sum())
    # magnitude-weighted orientation histogram (12 bins) - a cap brim is a
    # strong horizontal edge, headphone cups are strong verticals
    bins_a = 12
    idx = np.clip((ang / np.pi * bins_a).astype(np.int32), 0, bins_a - 1)
    wsum = mag.ravel()
    hist_o = np.bincount(idx.ravel(), weights=wsum, minlength=bins_a)[:bins_a]
    hist_o = hist_o / max(1e-6, hist_o.sum())
    return np.concatenate([hist_m, hist_o]).astype(np.float32)


def _edge_energy(gray: np.ndarray, axis: str) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    a = np.abs(gx if axis == "x" else gy)
    return float(a.mean())


def _dark_fraction(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    return float((gray < 60).mean())


def _moire_energy(bgr: np.ndarray) -> float:
    """Ratio of high-frequency energy in a photo-of-a-screen sense.

    Used by the (off-by-default) liveness check: a printed photo or a screen
    held to the camera tends to have an abnormally flat mid-tone with a
    spiky high-frequency floor.
    """
    if bgr is None or bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (128, 128))
    lap = cv2.Laplacian(g, cv2.CV_32F)
    hf = float(lap.var())
    lf = float(g.var()) + 1e-6
    return float(min(1.0, hf / (hf + 8.0 * lf)))


# --------------------------------------------------------------------------
def extract(bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Feature vector describing the head + accessory regions.  Fixed length."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    reg = regions(bgr, box)
    greg = regions(gray, box)

    feats: list[float] = []

    head, _head_box, head_cov = reg["head"]
    if head is None:
        return np.zeros(feature_dim(), dtype=np.float32)
    head_small = cv2.resize(head, (64, 64))

    feats.extend(_hsv_hist(head_small).tolist())
    feats.extend(_sobel_stats(cv2.cvtColor(head_small, cv2.COLOR_BGR2GRAY)).tolist())

    feats.append(head_cov)
    feats.append(_skin_fraction(head))

    fh = reg["forehead"][0]
    fore_skin = _skin_fraction(fh) if fh is not None else 0.0
    feats.append(fore_skin)

    face = reg["face"][0]
    face_skin = _skin_fraction(face) if face is not None else 0.0
    feats.append(face_skin)

    el, er = reg["ear_l"][0], reg["ear_r"][0]
    feats.append(_skin_fraction(el) if el is not None else 0.0)
    feats.append(_skin_fraction(er) if er is not None else 0.0)

    gface = greg["face"][0]
    face_v = _edge_energy(gface, "y") + 1e-6
    elg, erg = greg["ear_l"][0], greg["ear_r"][0]
    ear_v = 0.0
    n_ear = 0
    for e in (elg, erg):
        if e is not None:
            ear_v += _edge_energy(e, "y")
            n_ear += 1
    feats.append(ear_v / max(1, n_ear) / face_v)
    ear_h = 0.0
    n_ear = 0
    for e in (elg, erg):
        if e is not None:
            ear_h += _edge_energy(e, "x")
            n_ear += 1
    feats.append(ear_h / max(1, n_ear) / (face_v + 1e-6))

    # hat brim: strong horizontal edge low in the crown region
    crown = greg["crown"][0]
    if crown is not None:
        ch, cw = crown.shape[:2]
        top = crown[: max(1, ch // 2)]
        bot = crown[ch // 2:]
        feats.append(_edge_energy(top, "x") / (face_v + 1e-6))
        feats.append(_edge_energy(bot, "x") / (face_v + 1e-6))
        feats.append(_dark_fraction(crown))
        feats.append(reg["crown"][2])
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    # colour contrast between the head covering and the face
    if face is not None and face.size:
        hsv_h = cv2.cvtColor(head_small, cv2.COLOR_BGR2HSV)
        hsv_f = cv2.cvtColor(cv2.resize(face, (64, 64)), cv2.COLOR_BGR2HSV)
        feats.append(float(hsv_h[:, :, 1].mean()) / 255.0)
        feats.append(float(hsv_f[:, :, 1].mean()) / 255.0)
        feats.append(abs(float(hsv_h[:, :, 2].mean()) - float(hsv_f[:, :, 2].mean())) / 255.0)
        # hue spread of the head region - a coloured cap stands out
        hues = hsv_h[:, :, 0][hsv_h[:, :, 1] > 40].astype(np.float32)
        if hues.size > 10:
            rad = hues / 180.0 * np.pi * 2
            r = float(np.hypot(np.cos(rad).mean(), np.sin(rad).mean()))
            feats.append(1.0 - r)
        else:
            feats.append(0.0)
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    chin = greg["chin"][0]
    feats.append(_dark_fraction(chin) if chin is not None else 0.0)

    # A covering over the upper face (cap brim, hood, helmet visor) shows up
    # as the forehead losing skin while the cheeks keep it.
    feats.append(float(face_skin - fore_skin))
    out = np.asarray(feats, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def feature_dim() -> int:
    # 24 hsv + 22 sobel + 1 cov + 1 head skin + 1 forehead + 1 face + 2 ears
    # + 2 ear ratios + 4 crown + 4 colour + 1 chin + 1 pad
    return 24 + 22 + 1 + 1 + 1 + 1 + 2 + 2 + 4 + 4 + 1 + 1


__all__ = ["extract", "feature_dim", "regions", "_moire_energy", "FEATURE_VERSION"]
