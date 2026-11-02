"""Enrolment store: embeddings + head features, grouped by user and tag.

The "tag" is the key idea for hat/headphone support.  A tag is a comma
separated list of what you were wearing when you enrolled:

    default              bare face
    hat                  wearing a cap/beanie/hood
    headphones
    hat,headphones

At verify time every tag is scored and the best one wins, so a user who has
enrolled bare *and* hatted is matched against whichever gallery actually
looks like the current them, instead of being forced through one
compromise template the way a single-template system must.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import headfeatures

SCHEMA_VERSION = 1
DEFAULT_TAG = "default"
# Keep the store small but representative.
MAX_PER_TAG = 160


@dataclass
class TagInfo:
    count: int = 0
    created: str = ""
    updated: str = ""
    accessories: list[str] = field(default_factory=list)
    mean_quality: float = 0.0

    def as_dict(self) -> dict:
        return {"count": self.count, "created": self.created,
                "updated": self.updated, "accessories": self.accessories,
                "mean_quality": round(self.mean_quality, 3)}


class GalleryError(RuntimeError):
    pass


class Gallery:
    def __init__(self, paths, create: bool = True):
        self.paths = paths
        self.dir = Path(paths.gallery_dir)
        self.emb: np.ndarray = np.zeros((0, 128), np.float32)
        self.feat: np.ndarray = np.zeros((0, headfeatures.feature_dim()), np.float32)
        self.user: np.ndarray = np.zeros((0,), dtype=object)
        self.tag: np.ndarray = np.zeros((0,), dtype=object)
        self.meta: dict = {"version": SCHEMA_VERSION,
                           "feature_version": headfeatures.FEATURE_VERSION,
                           "users": {}, "calibration": {}}
        self._loaded = False
        if create:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> bool:
        gf, gm = Path(self.paths.gallery_file), Path(self.paths.gallery_meta)
        if not gf.is_file():
            self._loaded = True
            return False
        try:
            with np.load(gf, allow_pickle=False) as z:
                self.emb = np.asarray(z["embeddings"], dtype=np.float32)
                self.feat = np.asarray(z["features"], dtype=np.float32)
                self.user = z["user"].astype(object)
                self.tag = z["tag"].astype(object)
            if gm.is_file():
                self.meta = json.loads(gm.read_text())
            else:
                self.meta = {}
        except Exception as exc:
            raise GalleryError(f"corrupt gallery store {gf}: {exc}") from exc
        self.meta.setdefault("version", SCHEMA_VERSION)
        self.meta.setdefault("users", {})
        self.meta.setdefault("calibration", {})
        # A feature-vector layout change invalidates the accessory features.
        if int(self.meta.get("feature_version", 0)) != headfeatures.FEATURE_VERSION:
            self.feat = np.zeros((len(self.emb), headfeatures.feature_dim()), np.float32)
            self.meta["feature_version"] = headfeatures.FEATURE_VERSION
            self.meta["_features_dropped"] = True
        if self.emb.ndim == 1:
            self.emb = self.emb.reshape(0, -1)
        self._loaded = True
        return True

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta["version"] = SCHEMA_VERSION
        self.meta["feature_version"] = headfeatures.FEATURE_VERSION
        tmp = Path(tempfile.mkdtemp(prefix=".faceid-", dir=str(self.dir.parent)))
        try:
            np.savez_compressed(
                tmp / "gallery.npz",
                embeddings=self.emb.astype(np.float32),
                features=self.feat.astype(np.float32),
                user=np.asarray(list(self.user), dtype=object).astype("U"),
                tag=np.asarray(list(self.tag), dtype=object).astype("U"),
            )
            (tmp / "gallery.json").write_text(json.dumps(self.meta, indent=2))
            os.chmod(tmp / "gallery.npz", 0o600)
            os.chmod(tmp / "gallery.json", 0o600)
            for name in ("gallery.npz", "gallery.json"):
                os.replace(tmp / name, self.dir / name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    def add(self, user: str, tag: str, embeddings: np.ndarray,
            features: np.ndarray | None = None,
            accessories: list[str] | None = None,
            quality: float = 0.0) -> int:
        embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if embeddings.size == 0:
            return 0
        if features is None or len(features) != len(embeddings):
            features = np.zeros((len(embeddings), headfeatures.feature_dim()), np.float32)
        features = np.asarray(features, dtype=np.float32)

        keep = ~((np.asarray(self.user, dtype=object) == user) &
                 (np.asarray(self.tag, dtype=object) == tag))
        self.emb = np.concatenate([self.emb[keep], embeddings], axis=0)
        self.feat = np.concatenate([self.feat[keep], features], axis=0)
        self.user = np.concatenate([np.asarray(self.user, dtype=object)[keep],
                                    np.full(len(embeddings), user, dtype=object)])
        self.tag = np.concatenate([np.asarray(self.tag, dtype=object)[keep],
                                   np.full(len(embeddings), tag, dtype=object)])
        self._trim(user, tag)

        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        u = self.meta["users"].setdefault(user, {"tags": {}})
        t = u["tags"].setdefault(tag, TagInfo().as_dict())
        t["count"] = int((np.asarray(self.tag, dtype=object) == tag).sum())
        t["created"] = t.get("created") or now
        t["updated"] = now
        t["accessories"] = sorted(accessories or [])
        t["mean_quality"] = round(float(quality), 3)
        return len(embeddings)

    def _trim(self, user: str, tag: str) -> None:
        idx = np.where((np.asarray(self.user, dtype=object) == user) &
                       (np.asarray(self.tag, dtype=object) == tag))[0]
        if len(idx) <= MAX_PER_TAG:
            return
        # even subsample keeps pose/lighting diversity
        sel = np.linspace(0, len(idx) - 1, MAX_PER_TAG).round().astype(int)
        keep_idx = idx[sel]
        drop = np.setdiff1d(idx, keep_idx)
        self.emb = np.delete(self.emb, drop, axis=0)
        self.feat = np.delete(self.feat, drop, axis=0)
        self.user = np.delete(np.asarray(self.user, dtype=object), drop)
        self.tag = np.delete(np.asarray(self.tag, dtype=object), drop)

    # ------------------------------------------------------------------
    def users(self) -> list[str]:
        return sorted(set(str(u) for u in self.user))

    def tags(self, user: str | None = None) -> list[str]:
        m = np.ones(len(self.tag), dtype=bool)
        if user is not None:
            m &= (np.asarray(self.user, dtype=object) == user)
        return sorted(set(str(t) for t in np.asarray(self.tag, dtype=object)[m]))

    def matrix(self, user: str | None = None, tag: str | None = None):
        m = np.ones(len(self.emb), dtype=bool)
        u = np.asarray(self.user, dtype=object)
        t = np.asarray(self.tag, dtype=object)
        if user is not None:
            m &= (u == user)
        if tag is not None:
            m &= (t == tag)
        return self.emb[m], self.feat[m], t[m]

    def remove(self, user: str, tag: str | None = None) -> int:
        u = np.asarray(self.user, dtype=object)
        t = np.asarray(self.tag, dtype=object)
        m = (u == user)
        if tag is not None:
            m = m & (t == tag)
        n = int(m.sum())
        keep = ~m
        self.emb, self.feat = self.emb[keep], self.feat[keep]
        self.user, self.tag = u[keep], t[keep]
        meta_u = self.meta.get("users", {}).get(user)
        if meta_u is not None:
            if tag is None:
                self.meta["users"].pop(user, None)
            else:
                meta_u.get("tags", {}).pop(tag, None)
        return n

    def summary(self) -> dict:
        users = {}
        for u in self.users():
            tags = {}
            for t in self.tags(u):
                info = dict(self.meta.get("users", {}).get(u, {}).get("tags", {}).get(t, {}))
                embs, _f, _t = self.matrix(u, t)
                info["count"] = len(embs)
                tags[t] = info
            users[u] = {"tags": tags}
        return {"data_dir": str(self.paths.data_dir),
                "model_dir": str(self.paths.model_dir),
                "total_embeddings": int(len(self.emb)),
                "calibration": self.meta.get("calibration", {}),
                "users": users}

    def set_calibration(self, **kw) -> None:
        cal = self.meta.setdefault("calibration", {})
        cal.update(kw)
        cal["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
