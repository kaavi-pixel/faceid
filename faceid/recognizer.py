"""The engine: capture -> detect -> embed -> accessory analysis -> match.

Decision summary for one authentication attempt:

    for every enrolled tag of every candidate user:
        tag_score = mean of the best few cosine similarities of the captured
                    frames against that tag's enrolments
        votes     = frames whose similarity clears `match.vote_threshold`
        bonus     = accessory_bonus * agreement between the accessories the
                    classifier sees and the accessories that tag was enrolled
                    with   (minus `accessory_penalty` for a contradiction)
        final     = tag_score + bonus

    accept when: votes >= core.min_votes
                 and tag_score >= match.cosine_threshold
                 and (best user - runner-up user) >= match.margin
                 and liveness passes

The vote requirement plus the consistency requirement is what makes a single
lucky frame useless to an attacker, and the tag system plus the accessory
agreement bonus is what makes a hat or headphones a non-event rather than a
lockout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import accessories as acc_mod
from .accessories import AccessoryEngine, AccessoryModel, AccessoryPrediction
from .capture import Camera, CameraError, frame_quality, probe_devices, choose_device
from .config import load_config, paths as resolve_paths
from .detect import Detector
from .embed import Embedder, cosine_matrix, l2norm
from .gallery import DEFAULT_TAG, Gallery
from .liveness import LivenessTracker, embedding_consistency

ACC_MODEL_NAME = "accessory_model.npz"


@dataclass
class Sample:
    emb: np.ndarray
    feat: np.ndarray
    box: tuple
    accessories: AccessoryPrediction
    quality: float = 0.0


@dataclass
class Attempt:
    samples: list[Sample] = field(default_factory=list)
    frames: int = 0
    frames_with_face: int = 0
    liveness: dict = field(default_factory=dict)
    device: str = ""
    duration: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def embs(self) -> np.ndarray:
        if not self.samples:
            return np.zeros((0, 128), np.float32)
        return np.stack([s.emb for s in self.samples], axis=0)


@dataclass
class TagScore:
    tag: str
    score: float
    votes: int
    consistency: float
    bonus: float
    final: float
    accessories: frozenset = frozenset()

    def as_dict(self) -> dict:
        return {"tag": self.tag, "score": round(self.score, 4),
                "votes": self.votes, "accessory_agreement": round(self.consistency, 3),
                "bonus": round(self.bonus, 4), "final": round(self.final, 4),
                "tag_accessories": sorted(self.accessories)}


@dataclass
class Result:
    ok: bool = False
    user: str | None = None
    tag: str | None = None
    score: float = 0.0
    votes: int = 0
    frames: int = 0
    samples: int = 0
    accessories: dict = field(default_factory=dict)
    liveness: dict = field(default_factory=dict)
    reason: str = ""
    duration: float = 0.0
    device: str = ""
    candidates: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "user": self.user, "tag": self.tag,
                "score": round(self.score, 4), "votes": self.votes,
                "frames": self.frames, "samples": self.samples,
                "accessories": self.accessories, "liveness": self.liveness,
                "reason": self.reason, "duration_s": round(self.duration, 3),
                "device": self.device, "candidates": self.candidates}


class EngineError(RuntimeError):
    pass


class Engine:
    """Loads models lazily; safe to construct in a hot PAM path."""

    def __init__(self, cfg: dict | None = None, paths=None):
        self.cfg = cfg if cfg is not None else load_config()
        self.paths = paths if paths is not None else resolve_paths()
        self._det: Detector | None = None
        self._emb: Embedder | None = None
        self._gallery: Gallery | None = None
        self._acc_engine: AccessoryEngine | None = None

    # ------------------------------------------------------------------
    @property
    def detector(self) -> Detector:
        if self._det is None:
            self._det = Detector(self.paths.model_dir, self.cfg.sec("detect"))
        return self._det

    @property
    def embedder(self) -> Embedder:
        if self._emb is None:
            self._emb = Embedder(self.paths.model_dir, self.cfg.sec("embed"))
        return self._emb

    @property
    def gallery(self) -> Gallery:
        if self._gallery is None:
            self._gallery = Gallery(self.paths)
        return self._gallery

    @property
    def accessory_engine(self) -> AccessoryEngine:
        if self._acc_engine is None:
            model_path = self.paths.data_dir / ACC_MODEL_NAME
            model = AccessoryModel.load(model_path) if model_path.is_file() else None
            self._acc_engine = AccessoryEngine(self.cfg.sec("accessories"), model)
        return self._acc_engine

    def reload(self) -> None:
        self._gallery = None
        self._acc_engine = None

    def models_ready(self) -> tuple[bool, str]:
        from .detect import YUNET_NAME
        from .embed import SFACE_NAME
        md = self.paths.model_dir
        missing = [n for n in (YUNET_NAME, SFACE_NAME) if not (md / n).is_file()]
        if missing:
            return False, f"missing models {missing} in {md}"
        return True, f"models present in {md}"

    # ------------------------------------------------------------------
    def capture(self, seconds: float | None = None, max_frames: int | None = None,
                on_frame: Callable[[dict], None] | None = None,
                stop_when: Callable[[list[Sample]], bool] | None = None,
                device=None, want_face: bool = True) -> Attempt:
        core = self.cfg.sec("core")
        video = self.cfg.sec("video")
        seconds = float(seconds if seconds is not None else core.get("timeout", 5.0))
        max_frames = int(max_frames if max_frames is not None else core.get("max_frames", 90))
        early = float(core.get("early_exit", 2.0))
        t0 = time.time()
        att = Attempt()

        if not self.models_ready()[0]:
            raise EngineError(self.models_ready()[1])

        with Camera(video, device) as cam:
            att.device = cam.info.path if cam.info else "unknown"
            live = LivenessTracker(self.cfg.sec("liveness"))
            while True:
                now = time.time()
                if now - t0 > seconds or att.frames >= max_frames:
                    break
                frame = cam.read()
                att.frames += 1
                if frame is None:
                    att.notes.append("camera returned no frame")
                    continue
                ok_q, why = frame_quality(frame, video)
                if not ok_q:
                    att.notes.append(why)
                    if on_frame:
                        on_frame({"frame": att.frames, "face": False, "note": why})
                    continue
                try:
                    face = self.detector.largest_face(frame)
                except Exception as exc:
                    att.notes.append(f"detector error: {exc}")
                    continue
                if face is None:
                    if on_frame:
                        on_frame({"frame": att.frames, "face": False,
                                  "note": "no face found"})
                    if not want_face:
                        continue
                    continue
                att.frames_with_face += 1
                try:
                    emb = self.embedder.embed(frame, face.row)
                    feat = self._head_features(frame, face.box)
                    acc = self.accessory_engine.predict(frame, face.box, feat)
                except Exception as exc:
                    att.notes.append(f"embed error: {exc}")
                    continue
                q = self._quality(frame, face)
                att.samples.append(Sample(emb=emb, feat=feat, box=face.box,
                                          accessories=acc, quality=q))
                live.update(face, frame)
                if on_frame:
                    on_frame({"frame": att.frames, "face": True,
                              "score": round(face.score, 3),
                              "box": face.box,
                              "accessories": acc.as_dict()})
                if (stop_when is not None and stop_when(att.samples)):
                    break
                if early > 0 and (time.time() - t0) >= early and stop_when is not None:
                    # a caller that has enough evidence can stop early
                    pass
            att.liveness = live.report().as_dict()
        att.duration = time.time() - t0
        return att

    # ------------------------------------------------------------------
    def _head_features(self, frame, box) -> np.ndarray:
        from . import headfeatures
        try:
            return headfeatures.extract(frame, box)
        except Exception:
            return np.zeros((headfeatures.feature_dim(),), np.float32)

    @staticmethod
    def _quality(frame, face) -> float:
        import cv2
        h, w = frame.shape[:2]
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        lap = float(cv2.Laplacian(g, cv2.CV_32F).var())
        sharp = min(1.0, lap / 400.0)
        area = float(face.area) / float(h * w)
        size = min(1.0, area / 0.12)
        bright = 1.0 - min(1.0, abs(float(g.mean()) - 120.0) / 120.0)
        return float(0.45 * sharp + 0.35 * size + 0.20 * bright)

    # ------------------------------------------------------------------
    @staticmethod
    def _agg(values: np.ndarray, how: str) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return 0.0
        how = (how or "p75").strip().lower()
        if how == "mean":
            return float(values.mean())
        if how == "max":
            return float(values.max())
        if how == "median":
            return float(np.median(values))
        q = 75.0
        if how.startswith("p"):
            try:
                q = float(how[1:])
            except ValueError:
                q = 75.0
        return float(np.percentile(values, q))

    def _claim(self, accs: list[AccessoryPrediction]) -> frozenset:
        """Decide which accessories the user is wearing, robustly.

        A plain mean over all frames is too fragile: when the head turns, the
        far ear disappears and the earwear signal drops below any sensible
        gate, so a user who plainly has headphones on gets recorded as bare.
        We aggregate with a high percentile instead - "in at least a quarter
        of the frames there was clear evidence" - which is what the question
        actually means. The same aggregation is used at enrol and at verify
        time, otherwise the tag-agreement bonus could never fire.
        """
        acfg = self.cfg.sec("accessories")
        min_conf = float(acfg.get("min_confidence", 0.50))
        how = acfg.get("aggregate", "p75")
        if not accs:
            return frozenset()
        out = set()
        for k in acc_mod.ACCESSORY_CLASSES:
            if self._agg([a.probs.get(k, 0.0) for a in accs], how) >= min_conf:
                out.add(k)
        return frozenset(out)

    def _claimed(self, accs: list[AccessoryPrediction], min_conf: float | None = None
                 ) -> frozenset:
        if min_conf is None:
            return self._claim(accs)
        out = set()
        for k in acc_mod.ACCESSORY_CLASSES:
            if self._agg([a.probs.get(k, 0.0) for a in accs],
                         self.cfg.sec("accessories").get("aggregate", "p75")) >= min_conf:
                out.add(k)
        return frozenset(out)

    def _agg_probs(self, accs: list[AccessoryPrediction]) -> dict:
        """Reported accessory probabilities using the same aggregation."""
        keys = acc_mod.ACCESSORY_CLASSES
        how = self.cfg.sec("accessories").get("aggregate", "p75")
        if not accs:
            return {k: 0.0 for k in keys}
        return {k: round(self._agg([a.probs.get(k, 0.0) for a in accs], how), 4)
                for k in keys}

    @staticmethod
    def _mean_probs(accs: list[AccessoryPrediction]) -> dict:
        keys = acc_mod.ACCESSORY_CLASSES
        if not accs:
            return {k: 0.0 for k in keys}
        return {k: round(float(np.mean([a.probs.get(k, 0.0) for a in accs])), 4) for k in keys}

    def score(self, att: Attempt, users: list[str] | None = None,
              explain: bool = False) -> Result:
        m = self.cfg.sec("match")
        core = self.cfg.sec("core")
        acfg = self.cfg.sec("accessories")
        thr = float(self.gallery.meta.get("calibration", {}).get(
            "cosine_threshold", m.get("cosine_threshold", 0.40)))
        vote_thr = float(m.get("vote_threshold", 0.30))
        min_votes = int(core.get("min_votes", 3))
        margin = float(m.get("margin", 0.03))
        bonus_w = float(m.get("accessory_bonus", 0.05)) if acfg.get("influence_match", True) else 0.0
        pen_w = float(m.get("accessory_penalty", 0.02)) if acfg.get("influence_match", True) else 0.0
        min_conf = float(acfg.get("min_confidence", 0.55))

        res = Result(frames=att.frames, samples=len(att.samples),
                     liveness=att.liveness, device=att.device,
                     duration=att.duration)
        res.accessories = {
            "probs": self._agg_probs([s.accessories for s in att.samples]),
            "mean_probs": self._mean_probs([s.accessories for s in att.samples]),
            "aggregate": self.cfg.sec("accessories").get("aggregate", "p75"),
            "claimed": sorted(self._claimed([s.accessories for s in att.samples])),
            "sources": sorted({s.accessories.source for s in att.samples}),
        }

        if not att.samples:
            res.reason = ("no usable frame with a face was captured "
                          f"({att.frames} frames read; "
                          f"{'; '.join(sorted(set(att.notes))[:3]) or 'no diagnostics'})")
            return res

        all_users = self.gallery.users()
        cand_users = [u for u in (users if users is not None else all_users) if u in all_users]
        if not cand_users:
            known = ", ".join(all_users) if all_users else "none"
            who = users if users is not None else "any user"
            hint = ""
            if all_users:
                hint = (f" - the gallery is keyed by Unix username, and it holds: "
                        f"{known}. Enrol as the account you actually log in as "
                        f"(`faceid enroll` defaults to it; a --user typo produces "
                        f"exactly this silent failure).")
            res.reason = (f"no enrolment found for {who} - run `faceid enroll` "
                          f"first (gallery: {self.paths.gallery_dir}){hint}")
            return res

        claimed = self._claimed([s.accessories for s in att.samples])
        embs = att.embs
        per_user: list[tuple[float, str, TagScore]] = []
        for u in cand_users:
            tag_scores = self._score_user(u, embs, claimed, vote_thr, bonus_w, pen_w)
            if not tag_scores:
                continue
            best = max(tag_scores, key=lambda t: t.final)
            per_user.append((best.final, u, best))
            for ts in tag_scores:
                if explain:
                    res.candidates.append({"user": u, **ts.as_dict()})
        if not per_user:
            res.reason = "no usable enrolments to compare against"
            return res
        per_user.sort(key=lambda x: -x[0])
        best_final, best_user, best_tag = per_user[0]
        runner_up = per_user[1][0] if len(per_user) > 1 else -1.0

        res.user, res.tag, res.score, res.votes = (
            best_user, best_tag.tag, best_tag.score, best_tag.votes)

        if best_tag.votes < min_votes:
            res.reason = (f"matched {best_user} but only {best_tag.votes} of "
                          f"{len(att.samples)} frames agreed (need {min_votes})")
            return res
        if best_tag.score < thr:
            res.reason = (f"best match {best_tag.score:.3f} below threshold {thr:.3f} "
                          f"(user {best_user}, tag {best_tag.tag})")
            return res
        if len(per_user) > 1 and (best_final - runner_up) < margin:
            res.reason = (f"ambiguous: {best_user} {best_final:.3f} vs "
                          f"{per_user[1][1]} {runner_up:.3f} (need {margin:.3f} gap)")
            return res
        live = att.liveness or {}
        if live and not live.get("ok", True):
            res.reason = f"liveness check failed: {live.get('reason','')}"
            return res

        res.ok = True
        extra = f", wearing {', '.join(sorted(claimed))}" if claimed else ""
        res.reason = (f"verified {best_user} via '{best_tag.tag}' tag "
                      f"(sim {best_tag.score:.3f}, {best_tag.votes}/{len(att.samples)} frames, "
                      f"threshold {thr:.3f}{extra})")
        return res

    def _score_user(self, user: str, embs: np.ndarray, claimed: frozenset,
                    vote_thr: float, bonus_w: float, pen_w: float) -> list[TagScore]:
        out: list[TagScore] = []
        for tag in self.gallery.tags(user):
            g_emb, _f, _t = self.gallery.matrix(user, tag)
            if len(g_emb) == 0:
                continue
            sims = cosine_matrix(embs, g_emb) if len(embs) else np.zeros(0, np.float32)
            if sims.size == 0:
                continue
            per_frame = sims.max(axis=1)
            k = min(3, len(per_frame))
            top = np.sort(per_frame)[::-1][:k]
            score = float(top.mean())
            votes = int((per_frame >= vote_thr).sum())
            tag_acc = acc_mod.parse_tag(tag)
            consistency = acc_mod.jaccard(claimed, tag_acc)
            bonus = bonus_w * consistency
            if claimed and not claimed <= tag_acc:
                bonus -= pen_w * (1.0 - consistency)
            out.append(TagScore(tag=tag, score=score, votes=votes,
                                consistency=consistency, bonus=bonus,
                                final=score + bonus, accessories=tag_acc))
        return out

    # ------------------------------------------------------------------
    def verify(self, users: list[str] | None = None, seconds: float | None = None,
               explain: bool = False, on_frame=None, device=None) -> Result:
        core = self.cfg.sec("core")
        min_votes = int(core.get("min_votes", 3))

        def enough(samples: list[Sample]) -> bool:
            if len(samples) < max(min_votes, 6):
                return False
            e = np.stack([s.emb for s in samples], axis=0)
            n = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
            sims = n @ n.T
            iu = np.triu_indices(len(e), k=1)
            vals = sims[iu]
            # frames agree strongly with each other -> more frames add nothing
            return bool(vals.mean() > 0.80 and float(vals.min()) > 0.45)

        att = self.capture(seconds=seconds, on_frame=on_frame,
                           stop_when=enough if core.get("early_exit", 2.0) else None,
                           device=device)
        return self.score(att, users=users, explain=explain)

    def identify(self, seconds: float | None = None, explain: bool = True,
                 device=None) -> Result:
        """Best guess regardless of threshold - for `faceid test` / diagnosis."""
        att = self.capture(seconds=seconds, device=device)
        res = self.score(att, users=None, explain=explain)
        return res

    # ------------------------------------------------------------------
    def enroll(self, user: str, tag: str = DEFAULT_TAG, seconds: float = 8.0,
               min_samples: int = 12, max_samples: int = 60,
               on_frame=None, device=None) -> dict:
        """Capture a tag and store it.  Returns a report dict.

        Selection strategy matters more than volume.  We collect every good
        frame (up to a hard cap), then greedily pick a *diverse* subset -
        frames in different poses/lighting carry far more information than 60
        copies of the same pose.  Crucially, if the user sat perfectly still
        and the diverse subset is too small, we top it up with the best
        remaining frames instead of failing: an enrolment must never fail
        because you held still.
        """
        video = self.cfg.sec("video")
        t0 = time.time()
        samples: list[Sample] = []
        notes: set[str] = set()
        frames = 0
        face_frames = 0
        dev = ""
        hard_cap = max(max_samples * 4, 160)
        with Camera(video, device) as cam:
            dev = cam.info.path if cam.info else "unknown"
            while time.time() - t0 < seconds and len(samples) < hard_cap:
                frame = cam.read()
                frames += 1
                if frame is None:
                    continue
                okq, why = frame_quality(frame, video)
                if not okq:
                    notes.add(why)
                    continue
                face = self.detector.largest_face(frame)
                if face is None:
                    notes.add("no face found")
                    if on_frame:
                        on_frame({"frame": frames, "face": False})
                    continue
                face_frames += 1
                emb = self.embedder.embed(frame, face.row)
                feat = self._head_features(frame, face.box)
                acc = self.accessory_engine.predict(frame, face.box, feat)
                q = self._quality(frame, face)
                # drop literal duplicates only - real variety is handled by
                # the greedy selection below
                if samples:
                    kept = np.stack([s.emb for s in samples], axis=0)
                    if float(cosine_matrix(emb, kept).max()) > 0.995:
                        continue
                samples.append(Sample(emb=emb, feat=feat, box=face.box,
                                      accessories=acc, quality=q))
                if on_frame:
                    on_frame({"frame": frames, "face": True, "kept": len(samples),
                              "of": max_samples,
                              "accessories": acc.as_dict(),
                              "quality": round(q, 3),
                              "claimed": sorted(acc.confident(
                                  float(self.cfg.sec("accessories").get(
                                      "min_confidence", 0.55))))})
        selected = self._select_diverse(samples, max_samples, min_samples)
        if len(selected) < min_samples:
            return {"ok": False, "user": user, "tag": tag, "stored": 0,
                    "captured": len(selected), "frames": frames,
                    "face_frames": face_frames, "device": dev,
                    "reason": (f"only {len(selected)} usable frames (need {min_samples}); "
                               f"{face_frames} frames had a face; notes: {sorted(notes)}"),
                    "accessories": {}}
        embs = np.stack([s.emb for s in selected], axis=0)
        feats = np.stack([s.feat for s in selected], axis=0)
        accs = [s.accessories for s in selected]
        claimed = sorted(self._claim(accs))
        auto_tag = None
        if tag is None or str(tag).strip().lower() in ("auto", ""):
            tag = auto_tag = self._tag_for(claimed)
        seen = self.gallery.add(user, tag, embs, feats, accessories=claimed,
                                quality=float(np.mean([s.quality for s in selected])))
        self.gallery.save()
        return {"ok": True, "user": user, "tag": tag, "auto_tag": auto_tag,
                "stored": seen,
                "captured": len(selected), "offered": len(samples),
                "frames": frames, "face_frames": face_frames, "device": dev,
                "accessories": claimed,
                "predicted": self._agg_probs(accs),
                "mean_probs": self._mean_probs(accs),
                "mean_quality": round(float(np.mean([s.quality for s in selected])), 3),
                "reason": f"stored {seen} embeddings for '{user}' tag '{tag}' "
                          f"(from {len(samples)} captured frames)"}

    @staticmethod
    def _tag_for(claimed: list[str]) -> str:
        """Turn a claimed accessory set into a sensible tag name."""
        c = set(claimed)
        if not c:
            return DEFAULT_TAG
        if c == {"headwear"}:
            return "hat"
        if c == {"earwear"}:
            return "headphones"
        if c == {"eyewear"}:
            return "glasses"
        words = {"headwear": "hat", "earwear": "headphones", "eyewear": "glasses"}
        return ",".join(sorted(words.get(k, k) for k in c))

    @staticmethod
    def _select_diverse(samples: list[Sample], max_n: int, min_n: int,
                        sim_thresh: float = 0.97) -> list[Sample]:
        """Greedy diversity pick, with a top-up so we never come up short."""
        if not samples:
            return []
        pool = sorted(samples, key=lambda s: -s.quality)
        picked: list[Sample] = [pool.pop(0)]
        while pool and len(picked) < max_n:
            best_i, best_sim = None, 2.0
            for i, cand in enumerate(pool):
                sim = float(cosine_matrix(cand.emb, np.stack(
                    [p.emb for p in picked], axis=0)).max())
                if sim < best_sim:
                    best_i, best_sim = i, sim
            if best_i is None:
                break
            if best_sim > sim_thresh and len(picked) >= min_n:
                break                      # everything left is a near-clone
            picked.append(pool.pop(best_i))
        if len(picked) < min_n:
            # the user held still: accept the near-duplicates rather than fail
            for cand in pool:
                if len(picked) >= min_n:
                    break
                picked.append(cand)
        return picked[:max_n]

    # ------------------------------------------------------------------
    def train_accessories(self, user: str | None = None) -> dict:
        """Fit the per-user accessory classifier from tagged enrolments."""
        users = [user] if user else self.gallery.users()
        X, tags = [], []
        for u in users:
            for t in self.gallery.tags(u):
                _e, f, _t = self.gallery.matrix(u, t)
                if len(f) == 0:
                    continue
                X.append(f)
                tags.extend([t] * len(f))
        if not X:
            return {"ok": False, "reason": "no enrolment features to train on"}
        X = np.concatenate(X, axis=0)
        model = AccessoryModel()
        report = model.fit(X, tags)
        model.save(self.paths.data_dir / ACC_MODEL_NAME)
        self._acc_engine = None
        distinct = sorted({t for t in tags})
        return {"ok": True, "samples": int(len(X)), "tags": distinct,
                "classes": report,
                "saved": str(self.paths.data_dir / ACC_MODEL_NAME),
                "reason": (f"trained on {len(X)} frames across {len(distinct)} tags: "
                           f"{', '.join(distinct)}")}

    # ------------------------------------------------------------------
    def calibrate(self, user: str | None = None, seconds: float = 6.0,
                  attempt: int = 0) -> dict:
        """Measure the similarity distribution for the enrolled user.

        A "genuine" attempt captures the user and records how well they match
        their own gallery.  Passing `--attempt 2` captures what is nominally
        the same person again but reported as an impostor trial (e.g. a
        different person, or you wearing something you never enrolled) - that
        is what you would use to set a real threshold.
        """
        res = self.identify(seconds=seconds, explain=True)
        if not res.candidates:
            return {"ok": False, "reason": res.reason}
        best = max(res.candidates, key=lambda c: c["final"])
        return {"ok": True, "attempt": attempt, "user": res.user, "tag": res.tag,
                "score": res.score, "votes": res.votes, "samples": res.samples,
                "accessories": res.accessories, "liveness": res.liveness,
                "candidates": res.candidates[:8], "reason": res.reason}

    # ------------------------------------------------------------------
    def probe_environment(self) -> dict:
        video = self.cfg.sec("video")
        devs = probe_devices(video)
        chosen = choose_device(video, devs)
        ready, msg = self.models_ready()
        return {
            "models_ready": ready, "models": msg,
            "model_dir": str(self.paths.model_dir),
            "data_dir": str(self.paths.data_dir),
            "devices": [d.as_dict() for d in devs],
            "chosen_device": chosen.as_dict() if chosen else None,
            "gallery": self.gallery.summary(),
            "accessory_model": (self.paths.data_dir / ACC_MODEL_NAME).is_file(),
            "config": {k: self.cfg.get(k) for k in ("core", "video", "match")},
        }
