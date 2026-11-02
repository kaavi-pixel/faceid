"""Test suite for faceid.

Run with either:
    python3 -m pytest tests/ -v              (if pytest is available)
    /usr/bin/python3 tests/test_faceid.py    (standalone, no dependencies)

The camera-dependent tests are skipped automatically when no usable camera or
no ground-truth frame is present, so the suite is safe on a headless box.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from faceid import accessories as acc  # noqa: E402
from faceid import headfeatures as hf  # noqa: E402
from faceid.config import DEFAULTS, load_config, paths as resolve_paths  # noqa: E402
from faceid.gallery import DEFAULT_TAG, Gallery  # noqa: E402
from faceid.liveness import LivenessTracker, embedding_consistency  # noqa: E402

GT_CANDIDATES = [
    Path(os.environ.get("FACEID_GT_FRAME", "")),
    # test fixtures live next to the tests and are gitignored (a real face)
    Path(__file__).resolve().parent / "fixtures" / "face.png",
]


def gt_frame():
    import cv2
    for p in GT_CANDIDATES:
        if p and str(p) not in ("", ".") and p.is_file():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


# ---------------------------------------------------------------- tags
def test_parse_tag_basics():
    assert acc.parse_tag("default") == frozenset()
    assert acc.parse_tag("none") == frozenset()
    assert acc.parse_tag("hat") == frozenset({"headwear"})
    assert acc.parse_tag("cap") == frozenset({"headwear"})
    assert acc.parse_tag("beanie") == frozenset({"headwear"})
    assert acc.parse_tag("headphones") == frozenset({"earwear"})
    assert acc.parse_tag("earbuds") == frozenset({"earwear"})
    assert acc.parse_tag("hat,headphones") == frozenset({"headwear", "earwear"})
    assert acc.parse_tag("hat + headphones") == frozenset({"headwear", "earwear"})
    assert acc.parse_tag("Sunglasses") == frozenset({"eyewear"})
    assert acc.parse_tag("hoodie") == frozenset({"headwear"})
    assert acc.parse_tag("") == frozenset()
    assert acc.parse_tag(None) == frozenset()
    # unknown words must not silently become an accessory claim
    assert acc.parse_tag("zebra") == frozenset()


def test_jaccard():
    assert acc.jaccard(frozenset(), frozenset()) == 1.0
    assert acc.jaccard(frozenset({"headwear"}), frozenset({"headwear"})) == 1.0
    assert acc.jaccard(frozenset({"headwear"}), frozenset()) == 0.0
    assert abs(acc.jaccard(frozenset({"headwear", "earwear"}),
                           frozenset({"headwear"})) - 0.5) < 1e-9


# ---------------------------------------------------------------- classifier
def test_logistic_separates_synthetic_data():
    rng = np.random.default_rng(0)
    X0 = rng.normal(0.0, 1.0, size=(60, 8))
    X1 = rng.normal(2.5, 1.0, size=(60, 8))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(60), np.ones(60)])
    m = acc.Logistic(l2=0.1).fit(X, y)
    p0 = m.predict_proba(X0).mean()
    p1 = m.predict_proba(X1).mean()
    assert p0 < 0.25, p0
    assert p1 > 0.75, p1
    acc_ = float(((m.predict_proba(X) > 0.5) == y).mean())
    assert acc_ > 0.95, acc_


def test_logistic_single_class_is_safe():
    X = np.random.default_rng(1).normal(size=(10, 4))
    m = acc.Logistic().fit(X, np.ones(10))
    assert m.predict_proba(X).min() > 0.5
    m2 = acc.Logistic().fit(X, np.zeros(10))
    assert m2.predict_proba(X).max() < 0.5


def test_accessory_model_roundtrip(tmp_dir):
    rng = np.random.default_rng(2)
    feats, tags = [], []
    for tag, mu in (("default", 0.0), ("hat", 3.0), ("headphones", -3.0)):
        feats.append(rng.normal(mu, 1.0, size=(40, 64)))
        tags.extend([tag] * 40)
    X = np.vstack(feats).astype(np.float32)
    model = acc.AccessoryModel()
    rep = model.fit(X, tags)
    assert "headwear" in rep and "earwear" in rep
    path = Path(tmp_dir) / "m.npz"
    model.save(path)
    again = acc.AccessoryModel.load(path)
    assert again is not None
    a = model.predict(X[:1]).probs
    b = again.predict(X[:1]).probs
    for k in a:
        assert abs(a[k] - b[k]) < 1e-6, (k, a[k], b[k])
    # a hat sample must be claimed as headwear, an earbud sample as earwear
    hat_idx = 40
    ear_idx = 80
    assert again.predict(X[hat_idx]).probs["headwear"] > 0.5
    assert again.predict(X[ear_idx]).probs["earwear"] > 0.5


def test_accessory_engine_respects_disable():
    eng = acc.AccessoryEngine({"enabled": False})
    img = np.zeros((64, 64, 3), np.uint8)
    pred = eng.predict(img, (16, 16, 32, 32))
    assert pred.source == "disabled"
    assert set(pred.probs.values()) == {0.0}


# ---------------------------------------------------------------- features
def test_headfeatures_shape_stable():
    img = np.random.default_rng(3).integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    for box in [(80, 60, 90, 110), (0, 0, 40, 40), (300, 220, 30, 30)]:
        f = hf.extract(img, box)
        assert f.shape == (hf.feature_dim(),), (box, f.shape)
        assert np.isfinite(f).all()


def test_headfeatures_handles_tiny_and_offscreen():
    img = np.zeros((50, 50, 3), np.uint8)
    f = hf.extract(img, (45, 45, 4, 4))
    assert f.shape == (hf.feature_dim(),)
    assert np.isfinite(f).all()
    f2 = hf.extract(img, (-20, -20, 30, 30))
    assert f2.shape == (hf.feature_dim(),)
    assert np.isfinite(f2).all()


# ---------------------------------------------------------------- gallery
def test_gallery_crud(tmp_dir):
    os.environ["FACEID_DATA_DIR"] = tmp_dir
    p = resolve_paths()
    g = Gallery(p)
    assert g.users() == []
    e = np.random.default_rng(4).normal(size=(5, 128)).astype(np.float32)
    n = g.add("alice", "default", e)
    assert n == 5
    assert g.users() == ["alice"]
    assert g.tags("alice") == ["default"]
    assert len(g.matrix("alice", "default")[0]) == 5
    g.save()
    g2 = Gallery(p)
    assert len(g2.matrix("alice", "default")[0]) == 5
    # adding the same tag replaces, it does not accumulate duplicates
    g2.add("alice", "default", e[:2])
    assert len(g2.matrix("alice", "default")[0]) == 2
    g2.add("alice", "hat", e[:3])
    assert sorted(g2.tags("alice")) == ["default", "hat"]
    assert g2.remove("alice", "hat") == 3
    assert g2.tags("alice") == ["default"]
    assert g2.remove("alice") == 2
    assert g2.users() == []


def test_gallery_trims_to_cap(tmp_dir):
    from faceid import gallery as G
    os.environ["FACEID_DATA_DIR"] = tmp_dir
    g = Gallery(resolve_paths())
    e = np.random.default_rng(5).normal(size=(G.MAX_PER_TAG + 60, 128)).astype(np.float32)
    g.add("u", "default", e)
    assert len(g.matrix("u", "default")[0]) == G.MAX_PER_TAG


def test_gallery_drops_features_on_version_change(tmp_dir):
    import json
    os.environ["FACEID_DATA_DIR"] = tmp_dir
    p = resolve_paths()
    g = Gallery(p)
    g.add("u", "default", np.zeros((3, 128), np.float32),
          np.ones((3, hf.feature_dim()), np.float32))
    g.save()
    # simulate a store written by an older/newer feature layout
    meta = json.loads(Path(p.gallery_meta).read_text())
    meta["feature_version"] = hf.FEATURE_VERSION + 99
    Path(p.gallery_meta).write_text(json.dumps(meta))
    g2 = Gallery(p)
    assert g2.feat.shape == (3, hf.feature_dim())
    assert float(np.abs(g2.feat).sum()) == 0.0
    assert len(g2.emb) == 3  # embeddings survive, only derived features reset


# ---------------------------------------------------------------- liveness
def test_liveness_rejects_static_and_shaken():
    class FakeFace:
        def __init__(self, cx, cy, w, h):
            self.box = (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
        @property
        def center(self):
            return self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2

    cfg = {"enabled": True, "min_motion": 0.15, "max_motion": 60.0}
    # perfectly static
    t = LivenessTracker(cfg)
    for _ in range(10):
        t.update(FakeFace(100, 100, 200, 200))
    r = t.report()
    assert not r.ok and "static" in r.reason

    # realistic micro-motion
    t2 = LivenessTracker(cfg)
    rng = np.random.default_rng(6)
    for _ in range(10):
        t2.update(FakeFace(100 + rng.normal(0, 0.9), 100 + rng.normal(0, 0.9), 200, 200))
    assert t2.report().ok, t2.report().reason

    # violent shaking
    t3 = LivenessTracker(cfg)
    for i in range(10):
        t3.update(FakeFace(100 + (i % 2) * 180, 100, 200, 200))
    assert not t3.report().ok


def test_liveness_permits_too_few_frames():
    t = LivenessTracker({"enabled": True})
    t.update(type("F", (), {"box": (0, 0, 10, 10), "center": (5.0, 5.0)})())
    assert t.report(min_frames=4).ok  # cannot judge -> do not block


def test_embedding_consistency():
    rng = np.random.default_rng(7)
    base = rng.normal(size=128).astype(np.float32)
    tight = np.stack([base + rng.normal(0, 0.01, 128).astype(np.float32) for _ in range(5)])
    mean, votes = embedding_consistency(tight)
    assert mean > 0.9 and votes >= 5
    far = np.stack([base, rng.normal(size=128).astype(np.float32) * 3])
    assert embedding_consistency(far)[1] == 0


# ---------------------------------------------------------------- pam logic
def test_pam_denies_ssh_and_denied_services():
    from faceid import pam as P
    cfg = load_config(extra_files=[])
    os.environ["SSH_CONNECTION"] = "10.0.0.1 5000 10.0.0.2 22"
    try:
        d = P.should_attempt(cfg, "sudo")
        assert not d.attempt and "SSH" in d.reason
    finally:
        os.environ.pop("SSH_CONNECTION", None)
    d = P.should_attempt(cfg, "sshd")
    assert not d.attempt
    d = P.should_attempt(cfg, "not-a-service")
    assert not d.attempt


def test_pam_disabled_switch():
    from faceid import pam as P
    cfg = load_config(extra_files=[])
    cfg["core"]["disabled"] = True
    d = P.should_attempt(cfg, "sudo")
    assert not d.attempt and "disabled" in d.reason


def test_pam_user_resolution_prefers_human_over_root():
    """sudo hands us PAM_USER=root; the human is in PAM_RUSER.

    Uses the *current* account rather than a hardcoded name, so this passes on
    any machine instead of only on the author's.
    """
    import pwd
    from faceid import pam as P
    cfg = load_config(extra_files=[])
    me = pwd.getpwuid(os.getuid()).pw_name
    if me == "root":
        print("    (skipped: running as root, no distinct human account)")
        return
    saved = {k: os.environ.get(k) for k in ("PAM_USER", "PAM_RUSER", "SUDO_USER")}
    try:
        os.environ["PAM_USER"] = "root"
        os.environ["PAM_RUSER"] = me
        os.environ.pop("SUDO_USER", None)
        users = P.resolve_users(cfg)
        assert users[0] == me, users
        assert "root" in users
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_pam_local_session_detected():
    from faceid import pam as P
    assert P.is_remote({"PAM_RHOST": "", "PAM_TTY": "tty1"})[0] is False
    assert P.is_remote({"PAM_RHOST": "localhost", "PAM_TTY": "tty1"})[0] is False
    assert P.is_remote({"PAM_RHOST": "192.168.1.9", "PAM_TTY": ""})[0] is True
    assert P.is_remote({"PAM_RHOST": "", "PAM_TTY": "/dev/tty1"})[0] is False


def test_graphical_terminal_is_local_not_remote():
    """Regression, and the bug that made sudo never use the face at all.

    A terminal emulator inside the desktop runs on /dev/pts/N. SSH also uses
    /dev/pts/N. Treating the pty path as "remote" meant the helper aborted
    with 'pseudo-terminal /dev/pts/0' every time sudo was run from a terminal
    window - so face auth was skipped 100% of the time for the one case the
    whole project exists to serve. It passed in the test harness only because
    the harness had no controlling tty.
    """
    from faceid import pam as P
    for tty in ("/dev/pts/0", "/dev/pts/3", "/dev/pts/99"):
        remote, why = P.is_remote({"PAM_RHOST": "", "PAM_TTY": tty})
        assert remote is False, f"{tty} wrongly treated as remote ({why})"

    # ...but a genuinely remote request is still refused
    assert P.is_remote({"PAM_RHOST": "10.0.0.5", "PAM_TTY": "/dev/pts/0"})[0] is True

    # and end to end through the decision function
    saved = {k: os.environ.get(k) for k in ("PAM_TTY", "PAM_RHOST", "PAM_SERVICE",
                                            "PAM_USER", "PAM_RUSER", "SSH_CONNECTION")}
    try:
        os.environ.pop("SSH_CONNECTION", None)
        os.environ["PAM_TTY"] = "/dev/pts/0"
        os.environ["PAM_RHOST"] = ""
        os.environ["PAM_SERVICE"] = "sudo"
        d = P.should_attempt(load_config(extra_files=[]), "sudo")
        assert d.attempt, f"face auth skipped from a normal terminal: {d.reason}"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------- pam install
def test_resolve_data_dir_root_and_user_agree(tmp_dir):
    """The enrolling user's CLI and the root-run PAM helper must agree.

    If they don't, enrolments made in a terminal are invisible to sudo - and
    PAM just falls through to the password prompt, so it looks like the
    integration works when it is doing nothing at all.
    """
    import faceid.config as C
    real = C.SYSTEM_DATA_DIR
    saved_env = os.environ.pop("FACEID_DATA_DIR", None)
    try:
        system = Path(tmp_dir) / "var-lib-faceid"
        system.mkdir()
        C.SYSTEM_DATA_DIR = system
        for mode in ("root", "user"):
            got = C.resolve_data_dir()
            assert got == system, f"{mode}: {got}"
        # with no system store, fall back to the per-user location
        C.SYSTEM_DATA_DIR = Path(tmp_dir) / "does-not-exist"
        assert C.resolve_data_dir() != system
        # an explicit override always wins
        os.environ["FACEID_DATA_DIR"] = str(system)
        try:
            assert C.resolve_data_dir() == system
        finally:
            os.environ.pop("FACEID_DATA_DIR", None)
    finally:
        C.SYSTEM_DATA_DIR = real
        if saved_env is not None:
            os.environ["FACEID_DATA_DIR"] = saved_env


def test_fallback_to_enrolled_makes_sudo_work():
    """Regression: sudo shows up as PAM_USER=root with no PAM_RUSER.

    Without this fallback the candidate list is ["root"], nothing matches, and
    face auth fails *silently* - PAM just falls through to the password
    prompt, so it looks like the integration works when it does not.
    """
    from faceid import pam as P
    cfg = load_config(extra_files=[])
    saved = {k: os.environ.get(k) for k in ("PAM_USER", "PAM_RUSER", "SUDO_USER")}
    try:
        os.environ["PAM_USER"] = "root"
        os.environ.pop("PAM_RUSER", None)
        os.environ.pop("SUDO_USER", None)
        want = P.resolve_users(cfg)
        assert want == ["root"], want
        got, fell = P.expand_candidates(cfg, want, ["alice"])
        assert fell and "alice" in got, got

        # if a candidate already owns a face, do not widen
        got2, fell2 = P.expand_candidates(cfg, ["alice"], ["alice"])
        assert not fell2 and got2 == ["alice"]

        # with no enrolments at all there is nothing to fall back to
        got3, fell3 = P.expand_candidates(cfg, ["root"], [])
        assert not fell3 and got3 == ["root"]

        # and it can be turned off for a shared machine
        cfg2 = load_config(extra_files=[])
        cfg2["pam"]["fallback_to_enrolled"] = False
        got4, fell4 = P.expand_candidates(cfg2, ["root"], ["alice"])
        assert not fell4 and got4 == ["root"], got4
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_pam_patch_and_restore(tmp_dir):
    from faceid import install as I
    real_dir = I.PAM_DIR
    d = Path(tmp_dir) / "pam.d"
    d.mkdir(parents=True)
    try:
        I.PAM_DIR = d
        (d / "sudo").write_text("#%PAM-1.0\nauth       include      system-auth\n"
                                "account    include      system-auth\n")
        line = f"auth       sufficient   pam_exec.so quiet {I.BIN_PAM}"
        r = I.patch_service("sudo", line)
        assert r["changed"], r
        text = (d / "sudo").read_text()
        assert line in text
        assert "include      system-auth" in text, "password fallback was destroyed!"
        assert (d / "sudo.faceid-bak").is_file()
        # idempotent
        r2 = I.patch_service("sudo", line)
        assert not r2["changed"] and r2["skipped"] == "already installed"
        # our line must come before the password fallback
        lines = [l for l in text.splitlines() if l.strip()]
        assert lines.index(line) < lines.index("auth       include      system-auth")
        # restore is byte exact
        before = (d / "sudo.faceid-bak").read_text()
        res = I.unpatch_service("sudo")
        assert res["restored"]
        assert (d / "sudo").read_text() == before
    finally:
        I.PAM_DIR = real_dir


def test_pam_patch_refuses_lockout(tmp_dir):
    from faceid import install as I
    real_dir = I.PAM_DIR
    d = Path(tmp_dir) / "pam.d2"
    d.mkdir(parents=True)
    try:
        I.PAM_DIR = d
        (d / "sudo").write_text("#%PAM-1.0\nauth       sufficient    pam_howdy.so\nauth       include      system-auth\n")
        line = f"auth       sufficient   pam_exec.so quiet {I.BIN_PAM}"
        # with howdy disabled and the include removed there is nothing left
        (d / "empty").write_text("#%PAM-1.0\naccount include system-auth\n")
        r = I.patch_service("empty", line)
        assert not r["changed"], r
        assert "lockout" in r["skipped"]
    finally:
        I.PAM_DIR = real_dir


def test_pam_patch_can_disable_howdy(tmp_dir):
    from faceid import install as I
    real_dir = I.PAM_DIR
    d = Path(tmp_dir) / "pam.d3"
    d.mkdir(parents=True)
    try:
        I.PAM_DIR = d
        (d / "sudo").write_text("#%PAM-1.0\nauth       sufficient    pam_howdy.so\n"
                                "auth       include      system-auth\n")
        line = f"auth       sufficient   pam_exec.so quiet {I.BIN_PAM}"
        r = I.patch_service("sudo", line, replace_howdy=True)
        assert r["changed"] and r["removed_howdy"] == 1
        text = (d / "sudo").read_text()
        assert "pam_howdy.so" not in [l.strip() for l in text.splitlines()
                                      if not l.strip().startswith("#")]
        assert "# auth       sufficient    pam_howdy.so" in text
        assert "include      system-auth" in text
    finally:
        I.PAM_DIR = real_dir


def test_pam_line_uses_sufficient_never_required():
    from faceid import install as I
    line = I._pam_line()
    assert "sufficient" in line
    assert "required" not in line
    assert "faceid-pam" in line


def test_resolve_services():
    from faceid import install as I
    assert I.resolve_services(None) == I.DEFAULT_SERVICES
    assert "sudo" in I.resolve_services(["all"])
    assert I.resolve_services(["sudo", "sudo"]) == ["sudo"]


# ---------------------------------------------------------------- detectors
def test_detector_on_real_face_if_available():
    img = gt_frame()
    if img is None:
        print("    (skipped: no ground-truth frame)")
        return
    from faceid.config import resolve_model_dir
    from faceid.detect import Detector
    md = resolve_model_dir()
    if not (md / "face_detection_yunet_2023mar.onnx").is_file():
        print("    (skipped: model missing)")
        return
    cfg = dict(DEFAULTS["detect"])
    det = Detector(md, cfg)
    faces = det.faces(img)
    assert faces, "no face detected in the ground-truth frame"
    assert faces[0].score > 0.8, f"low confidence on a real face: {faces[0].score}"
    assert faces[0].area > 5000


def test_embedder_identical_input_is_identical():
    img = gt_frame()
    from faceid.config import resolve_model_dir
    md = resolve_model_dir()
    if img is None or not (md / "face_recognition_sface_2021dec.onnx").is_file():
        print("    (skipped)")
        return
    from faceid.detect import Detector
    from faceid.embed import Embedder
    det = Detector(md, DEFAULTS["detect"])
    f = det.largest_face(img)
    em = Embedder(md, DEFAULTS["embed"])
    a = em.embed(img, f.row)
    b = em.embed(img, f.row)
    assert abs(em.cosine(a, b) - 1.0) < 1e-5
    # and a brightened copy must still match closely (illumination invariance)
    import cv2
    brighter = cv2.convertScaleAbs(img, alpha=1.0, beta=35)
    f2 = det.largest_face(brighter)
    assert f2 is not None
    c = em.embed(brighter, f2.row)
    assert em.cosine(a, c) > 0.5, f"brightness change broke the embedding: {em.cosine(a,c)}"


def test_full_pipeline_enroll_and_verify_on_real_face(tmp_dir):
    """End-to-end, using real facial data: enroll, then verify.

    We enrol from one frame and verify against genuinely different captures
    of the same face (different exposure, JPEG noise, small shifts), which is
    exactly the variation a real second attempt would present.
    """
    import cv2
    img = gt_frame()
    from faceid.config import resolve_model_dir
    md = resolve_model_dir()
    if img is None or not (md / "face_detection_yunet_2023mar.onnx").is_file():
        print("    (skipped: no ground-truth frame or model)")
        return
    os.environ["FACEID_DATA_DIR"] = tmp_dir
    from faceid.detect import Detector
    from faceid.embed import Embedder
    from faceid.recognizer import Attempt, Engine, Sample

    det = Detector(md, DEFAULTS["detect"])
    em = Embedder(md, DEFAULTS["embed"])
    rng = np.random.default_rng(11)

    def variant(base, seed):
        r = np.random.default_rng(seed)
        out = cv2.convertScaleAbs(base, alpha=1.0 + r.uniform(-0.15, 0.25),
                                  beta=float(r.uniform(-25, 25)))
        # a few pixels of shift + jpeg
        m = np.float32([[1, 0, r.uniform(-3, 3)], [0, 1, r.uniform(-3, 3)]])
        out = cv2.warpAffine(out, m, (out.shape[1], out.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        f = det.largest_face(out)
        return out, f

    engine = Engine(load_config(), resolve_paths())
    # enroll the (single) ground-truth frame
    f0 = det.largest_face(img)
    assert f0 is not None
    e0 = em.embed(img, f0.row)
    ft0 = hf.extract(img, f0.box)
    engine.gallery.add("alice", "hat", e0[None, :], ft0[None, :],
                       accessories=["headwear"], quality=0.9)
    engine.gallery.save()
    engine.reload()

    def attempt_from(frames):
        att = Attempt()
        for img_i, fr in enumerate(frames):
            ff = det.largest_face(fr)
            if ff is None:
                continue
            att.samples.append(Sample(emb=em.embed(fr, ff.row),
                                      feat=hf.extract(fr, ff.box),
                                      box=ff.box,
                                      accessories=engine.accessory_engine.predict(fr, ff.box),
                                      quality=0.9))
            att.frames += 1
        att.liveness = {"ok": True, "reason": "test"}
        return att

    # 1) genuine: several slightly different captures of the same face
    good = [variant(img, s)[0] for s in range(5)]
    att = attempt_from(good)
    res = engine.score(att, users=["alice"], explain=True)
    assert res.ok, f"genuine attempt rejected: {res.reason}"
    assert res.tag == "hat"
    assert res.votes >= 3
    print(f"    genuine: {res.reason}")

    # 2) impostor: a completely different face must be rejected
    other = np.zeros_like(img)
    cv2.ellipse(other, (320, 320), (110, 150), 0, 0, 360, (200, 180, 170), -1)
    cv2.circle(other, (285, 285), 14, (60, 60, 60), -1)
    cv2.circle(other, (355, 285), 14, (60, 60, 60), -1)
    att2 = attempt_from([other] * 5)
    res2 = engine.score(att2, users=["alice"])
    assert not res2.ok, f"a fake face was accepted: {res2.reason}"
    print(f"    impostor: {res2.reason}")

    # 3) moving the threshold above the measured similarity must reject
    engine.gallery.set_calibration(cosine_threshold=0.99)
    engine.gallery.save()
    engine.reload()
    res3 = engine.score(attempt_from(good), users=["alice"])
    assert not res3.ok and "below threshold" in res3.reason, res3.reason

    # 4) voting must be required: a single frame is not enough
    engine.gallery.set_calibration(cosine_threshold=0.40)
    engine.gallery.save()
    engine.reload()
    res4 = engine.score(attempt_from(good[:1]), users=["alice"])
    assert not res4.ok and "frames agreed" in res4.reason, res4.reason

    # 5) an unrelated identity vector must not match, even at a lax threshold
    att5 = Attempt()
    for _ in range(5):
        att5.samples.append(Sample(emb=rng.normal(size=128).astype(np.float32),
                                   feat=hf.extract(img, f0.box), box=f0.box,
                                   accessories=acc.AccessoryPrediction({}, "test"),
                                   quality=0.9))
        att5.frames += 1
    att5.liveness = {"ok": True, "reason": "test"}
    res5 = engine.score(att5, users=["alice"], explain=True)
    assert not res5.ok, f"a random identity vector was accepted: {res5.reason}"
    assert res5.score < 0.2, f"unrelated embedding scored {res5.score}"
    print(f"    unrelated identity: {res5.reason}")
    print("    threshold + vote gates verified")


def test_select_diverse_tops_up_when_user_holds_still():
    """Regression: a motionless user must still be able to enrol.

    The first version of the enrolment loop simply *skipped* frames that
    looked like frames it already had, so someone sitting perfectly still got
    "only 1 usable frame" and the enrolment failed outright.
    """
    from faceid.recognizer import Engine, Sample
    rng = np.random.default_rng(20)

    def mk(emb, q=0.8):
        return Sample(emb=emb.astype(np.float32),
                      feat=np.zeros(hf.feature_dim(), np.float32),
                      box=(0, 0, 10, 10),
                      accessories=acc.AccessoryPrediction({}, "test"), quality=q)

    # 30 copies of essentially the same frame
    base = rng.normal(size=128).astype(np.float32)
    clones = [mk(base + rng.normal(0, 0.002, 128).astype(np.float32)) for _ in range(30)]
    picked = Engine._select_diverse(clones, max_n=60, min_n=12)
    assert len(picked) >= 12, f"starvation regression: only {len(picked)} selected"

    # genuine variety: three well-separated clusters must all be represented
    a = rng.normal(0, 1, 128).astype(np.float32)
    b = rng.normal(0, 1, 128).astype(np.float32)
    c = rng.normal(0, 1, 128).astype(np.float32)
    varied = ([mk(a + rng.normal(0, 0.01, 128).astype(np.float32)) for _ in range(10)] +
              [mk(b + rng.normal(0, 0.01, 128).astype(np.float32)) for _ in range(10)] +
              [mk(c + rng.normal(0, 0.01, 128).astype(np.float32)) for _ in range(10)])
    picked2 = Engine._select_diverse(varied, max_n=60, min_n=3)
    assert len(picked2) == 3, f"expected one per cluster, got {len(picked2)}"
    for centre in (a, b, c):
        assert max(acc_mod_vs(p, centre) for p in picked2) > 0.9, "a cluster was missed"

    assert Engine._select_diverse([], 60, 12) == []


def acc_mod_vs(sample, centre):
    v = sample.emb
    return float(np.dot(v, centre) / (np.linalg.norm(v) * np.linalg.norm(centre) + 1e-9))


def test_accessory_heuristic_finds_the_cap():
    """The ground-truth frame has a cap on - the heuristic must notice."""
    img = gt_frame()
    from faceid.config import resolve_model_dir
    md = resolve_model_dir()
    if img is None or not (md / "face_detection_yunet_2023mar.onnx").is_file():
        print("    (skipped)")
        return
    from faceid.detect import Detector
    det = Detector(md, DEFAULTS["detect"])
    f = det.largest_face(img)
    pred = acc.heuristic(img, f.box)
    print(f"    accessory prediction on the real hatted face: {pred.as_dict()}")
    assert pred.probs["headwear"] > 0.5, pred.as_dict()


# ---------------------------------------------------------------- plumbing
def test_config_defaults_are_complete():
    for section in ("core", "video", "detect", "embed", "match", "accessories",
                    "liveness", "pam", "selinux"):
        assert section in DEFAULTS, section
    # a `required` PAM line can never be produced by the defaults
    from faceid.install import _pam_line
    assert "required" not in _pam_line()


def test_config_merge_and_bad_file_is_survivable(tmp_dir):
    bad = Path(tmp_dir) / "bad.toml"
    bad.write_text("this is not = = toml[[[")
    cfg = load_config(extra_files=[bad])
    assert cfg.sec("core")["timeout"] == DEFAULTS["core"]["timeout"]


def test_config_write_roundtrip(tmp_dir):
    from faceid.cli import _write_config
    p = Path(tmp_dir) / "c.toml"
    _write_config(str(p), "video.prefer_ir", "true")
    _write_config(str(p), "match.cosine_threshold", "0.55")
    _write_config(str(p), "video.device", "/dev/video2")
    import tomllib
    with open(p, "rb") as fh:
        got = tomllib.load(fh)
    assert got["video"]["prefer_ir"] is True
    assert got["video"]["device"] == "/dev/video2"
    assert abs(got["match"]["cosine_threshold"] - 0.55) < 1e-9


def test_cli_parses_every_command():
    from faceid.cli import build_parser
    p = build_parser()
    for argv in (["doctor"], ["devices"], ["models", "--download"], ["list"],
                 ["enroll", "--tag", "hat", "--tag", "headphones"],
                 ["verify", "--explain"], ["test"], ["calibrate", "--apply"],
                 ["remove", "--user", "u"], ["ir"], ["config"],
                 ["selinux"], ["pam", "status"], ["pam", "install", "--services", "all"],
                 ["auth", "--debug"], ["version"], ["doctor", "--json"]):
        ns = p.parse_args(argv)
        assert hasattr(ns, "func"), argv


# ---------------------------------------------------------------- runner
def _tmp_dir(d):
    return tempfile.mkdtemp(prefix="faceid-test-", dir=d)


def main() -> int:
    import inspect
    import traceback
    base = tempfile.mkdtemp(prefix="faceid-tests-")
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed, failed, skipped = 0, 0, []
    for name, fn in tests:
        kwargs = {}
        params = inspect.signature(fn).parameters
        d = None
        if "tmp_dir" in params:
            d = _tmp_dir(base)
            kwargs["tmp_dir"] = d
        # Tests that point FACEID_DATA_DIR at a temp store must not leak it
        # into the next test - that is a real source of confusing failures.
        for var in ("FACEID_DATA_DIR", "FACEID_CONFIG"):
            os.environ.pop(var, None)
        try:
            fn(**kwargs)
            print(f"PASS  {name}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {name}: {exc}")
            failed += 1
        except Exception:
            print(f"ERROR {name}:")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
