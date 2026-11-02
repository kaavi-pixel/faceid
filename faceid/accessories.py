"""Accessory awareness: is the user wearing a hat / headphones / glasses?

Two layers, so it works from the very first enrolment and gets better:

1. GEOMETRY (always available, no training data)
   The face is the same, the things around it are not.  A cap hides
   forehead skin and puts a strong horizontal edge across the crown region;
   headphone cups add vertical structure beside the head where the ears are
   and usually cover the ears themselves.  These priors are surprisingly
   reliable for a personal (single-user) install.

2. LEARNED (per user, trained from the user's own tagged enrolments)
   Tag each enrolment with what you are wearing ("hat", "headphones",
   "hat,headphones") and `faceid train` fits a small logistic-regression
   classifier on the 64-dimensional head features.  No giant download, no
   shared dataset, trains in milliseconds, and it is specific to *your* hat
   in *your* room rather than a generic "hat" corpus.

The prediction is then used at match time to prefer the enrolment recorded
under the same conditions - that is what makes "works with a hat on" true
rather than marketing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import headfeatures

ACCESSORY_CLASSES = ("headwear", "earwear", "eyewear")

TAG_KEYWORDS: dict[str, set[str]] = {
    "headwear": {"hat", "cap", "baseballcap", "snapback", "beanie", "toque",
                 "tuque", "bonnet", "hood", "hoodie", "helmet", "beret",
                 "bandana", "headwear", "visor", "cowboyhat", "fedora"},
    "earwear": {"headphones", "earphones", "earbuds", "headset", "airpods",
                "earwear", "cans", "beats", "buds"},
    "eyewear": {"glasses", "spectacles", "eyeglasses", "sunglasses", "shades",
                "eyewear", "contacts"},
}

NONE_TAGS = {"default", "none", "bare", "normal", "plain", "nothing",
             "no", "bareface", "clean", "standard"}

TAG_ALIASES = {
    "baseball cap": "cap", "baseball_cap": "cap", "hat ": "hat",
    "head phones": "headphones", "head-phones": "headphones",
    "glasses": "glasses", "specs": "glasses",
}


@dataclass
class AccessoryPrediction:
    probs: dict[str, float] = field(default_factory=dict)
    source: str = "none"

    @property
    def set(self) -> frozenset:
        return frozenset(k for k, v in self.probs.items() if v >= 0.5)

    def confident(self, min_conf: float) -> frozenset:
        return frozenset(k for k, v in self.probs.items() if v >= min_conf)

    def best(self) -> tuple[str, float]:
        if not self.probs:
            return "", 0.0
        k = max(self.probs, key=lambda x: self.probs[x])
        return k, float(self.probs[k])

    def as_dict(self) -> dict:
        return {"probs": {k: round(v, 3) for k, v in self.probs.items()},
                "source": self.source,
                "claimed": sorted(self.set)}


# --------------------------------------------------------------------------
def parse_tag(tag: str) -> frozenset:
    """'hat,headphones' -> frozenset({'headwear','earwear'})."""
    if tag is None:
        return frozenset()
    raw = tag.strip().lower()
    raw = TAG_ALIASES.get(raw, raw)
    if raw in NONE_TAGS:
        return frozenset()
    out = set()
    for part in raw.replace("+", ",").replace("/", ",").split(","):
        p = part.strip().replace(" ", "").replace("-", "").replace("_", "")
        if not p or p in NONE_TAGS:
            continue
        for cls, words in TAG_KEYWORDS.items():
            if p in words:
                out.add(cls)
                break
            if any(p.startswith(w) for w in words):
                out.add(cls)
                break
    return frozenset(out)


def tag_has_accessories(tag: str) -> bool:
    return bool(parse_tag(tag))


def describe_tag(tag: str) -> str:
    s = parse_tag(tag)
    return "bare face" if not s else ", ".join(sorted(s))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


# --------------------------------------------------------------------------
def _l2(X: np.ndarray) -> np.ndarray:
    return X[:, None] if X.ndim == 1 else X


class Logistic:
    """Binary logistic regression, plain numpy, L2 regularised."""

    def __init__(self, l2: float = 1.0, epochs: int = 900, lr: float = 0.25):
        self.l2, self.epochs, self.lr = l2, epochs, lr
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logistic":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-6] = 1.0
        Z = (X - self.mu) / self.sd
        # handle a single-class training set gracefully
        if len(np.unique(y)) < 2:
            self.w = np.zeros(Z.shape[1])
            self.b = 4.0 if y.mean() > 0.5 else -4.0
            return self
        n, d = Z.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.epochs):
            z = Z @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = p - y
            gw = (Z.T @ g) / n + (self.l2 / n) * w
            gb = float(g.mean())
            w -= self.lr * gw
            b -= self.lr * gb
        self.w, self.b = w, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.w is None:
            return np.full(len(X), 0.5)
        Z = (X - self.mu) / self.sd
        z = Z @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def state(self) -> dict:
        return {"w": self.w, "b": np.asarray([self.b]), "mu": self.mu, "sd": self.sd}


class AccessoryModel:
    """One logistic classifier per accessory class, trained per user."""

    def __init__(self, models: dict[str, Logistic] | None = None):
        self.models = models or {}

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, tags: list[str]) -> dict:
        """X: (N,D) head features; tags: N tag strings."""
        X = np.asarray(X, dtype=np.float64)
        sets = [parse_tag(t) for t in tags]
        report = {}
        self.models = {}
        for cls in ACCESSORY_CLASSES:
            y = np.asarray([1.0 if cls in s else 0.0 for s in sets])
            if len(np.unique(y)) < 2:
                report[cls] = {"classes": 1, "pos": int(y.sum()), "skipped": True}
                continue
            m = Logistic().fit(X, y)
            acc = float(((m.predict_proba(X) > 0.5).astype(float) == y).mean())
            self.models[cls] = m
            report[cls] = {"pos": int(y.sum()), "neg": int(len(y) - y.sum()),
                           "train_accuracy": round(acc, 4)}
        return report

    def predict(self, feat: np.ndarray) -> AccessoryPrediction:
        probs = {}
        for cls, m in self.models.items():
            try:
                probs[cls] = float(m.predict_proba(feat)[0])
            except Exception:
                probs[cls] = 0.0
        for cls in ACCESSORY_CLASSES:
            probs.setdefault(cls, 0.0)
        return AccessoryPrediction(probs=probs, source="learned")

    def save(self, path) -> None:
        arrays = {}
        for cls, m in self.models.items():
            st = m.state()
            arrays[f"{cls}__w"] = st["w"]
            arrays[f"{cls}__b"] = st["b"]
            arrays[f"{cls}__mu"] = st["mu"]
            arrays[f"{cls}__sd"] = st["sd"]
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "AccessoryModel | None":
        try:
            with np.load(path) as z:
                models = {}
                for acc_cls in ACCESSORY_CLASSES:
                    key = f"{acc_cls}__w"
                    if key not in z:
                        continue
                    m = Logistic()
                    m.w = z[f"{acc_cls}__w"]
                    m.b = float(z[f"{acc_cls}__b"][0])
                    m.mu = z[f"{acc_cls}__mu"]
                    m.sd = z[f"{acc_cls}__sd"]
                    models[acc_cls] = m
                return cls(models) if models else None
        except Exception:
            return None


# --------------------------------------------------------------------------
def heuristic(bgr: np.ndarray, box: tuple[int, int, int, int]) -> AccessoryPrediction:
    """Priors that hold without any training data."""
    import cv2

    if bgr is None:
        return AccessoryPrediction({c: 0.0 for c in ACCESSORY_CLASSES}, "heuristic")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    reg = headfeatures.regions(bgr, box)
    greg = headfeatures.regions(gray, box)

    fh, face = reg["forehead"][0], reg["face"][0]
    head = reg["head"][0]

    probs: dict[str, float] = {c: 0.0 for c in ACCESSORY_CLASSES}

    # ---- headwear: forehead covered, face still visible -----------------
    fore_skin = headfeatures._skin_fraction(fh) if fh is not None else 0.0
    face_skin = headfeatures._skin_fraction(face) if face is not None else 0.0
    head_skin = headfeatures._skin_fraction(head) if head is not None else 0.0
    if face_skin > 0.10:
        lost = float(np.clip((face_skin - fore_skin) / max(1e-3, face_skin), 0.0, 1.0))
        probs["headwear"] = float(np.clip(lost, 0.0, 0.95)) * (0.5 + 0.5 * min(1.0, face_skin * 3))
    # a coloured crown with low skin pushes the probability up
    crown = greg["crown"][0]
    if crown is not None and head is not None:
        crown_dark = headfeatures._dark_fraction(crown)
        # a crown region that is mostly non-skin is a strong hint of a
        # covering, whether it is a dark cap or a bright one
        covering = float(np.clip(1.0 - head_skin / 0.18, 0.0, 0.9))
        probs["headwear"] = max(probs["headwear"],
                                covering * (0.75 + 0.25 * min(1.0, crown_dark * 2)))

    # ---- earwear: vertical structure over both ears, skin gone ----------
    el, er = greg["ear_l"][0], greg["ear_r"][0]
    if el is not None and er is not None and face is not None:
        face_v = headfeatures._edge_energy(greg["face"][0], "y") + 1e-6
        side_v = (headfeatures._edge_energy(el, "y") + headfeatures._edge_energy(er, "y")) / 2.0
        ratio = side_v / face_v
        ear_skin = (_skin_fraction_bgr(reg["ear_l"][0]) + _skin_fraction_bgr(reg["ear_r"][0])) / 2.0
        strength = float(np.clip((ratio - 0.55) / 0.9, 0.0, 1.0))
        missing_skin = float(np.clip((0.20 - ear_skin) / 0.20, 0.0, 1.0))
        probs["earwear"] = float(np.clip(0.6 * strength + 0.4 * missing_skin, 0.0, 0.95))
        if ear_skin > 0.30 and ratio < 1.2:
            probs["earwear"] = min(probs["earwear"], 0.25)

    # ---- eyewear: dark band across the eye line -------------------------
    if face is not None and face.size:
        x, y, w, h = box
        band, _box, cov = headfeatures._clip_roi(bgr, x + 0.10 * w, y + 0.28 * h,
                                                x + 0.90 * w, y + 0.55 * h)
        under, _b2, _c2 = headfeatures._clip_roi(bgr, x + 0.15 * w, y + 0.62 * h,
                                                x + 0.85 * w, y + 0.95 * h)
        if band is not None and under is not None and cov > 0.6:
            band_v = float(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).mean())
            band_dark = headfeatures._dark_fraction(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY))
            under_v = float(cv2.cvtColor(under, cv2.COLOR_BGR2GRAY).mean())
            dark_band = 1.0 if (under_v - band_v) > 25 else 0.0
            probs["eyewear"] = float(np.clip(0.55 * band_dark + 0.45 * dark_band, 0.0, 0.9))

    return AccessoryPrediction(probs={k: round(v, 4) for k, v in probs.items()},
                               source="heuristic")


def _skin_fraction_bgr(bgr) -> float:
    if bgr is None:
        return 0.0
    return headfeatures._skin_fraction(bgr)


# --------------------------------------------------------------------------
class AccessoryEngine:
    """Combines the learned model and the heuristic."""

    def __init__(self, cfg: dict, model: AccessoryModel | None = None):
        self.cfg = cfg or {}
        self.model = model

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True))

    def predict(self, bgr, box, feat: np.ndarray | None = None) -> AccessoryPrediction:
        if not self.enabled:
            return AccessoryPrediction({c: 0.0 for c in ACCESSORY_CLASSES}, "disabled")
        h = heuristic(bgr, box)
        if self.model is not None and self.cfg.get("use_learned", True) and feat is not None:
            p = self.model.predict(feat)
            # blend: the learned model is trusted more, the heuristic keeps us
            # sane when the learned model sees something new
            wl = 0.75
            blended = {c: wl * p.probs.get(c, 0.0) + (1 - wl) * h.probs.get(c, 0.0)
                       for c in ACCESSORY_CLASSES}
            return AccessoryPrediction(probs=blended, source="learned+heuristic")
        return h
