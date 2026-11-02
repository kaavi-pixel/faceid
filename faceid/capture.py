"""Camera discovery and frame capture.

Handles the messy reality of UVC devices on Linux:
  * one physical camera exposes several /dev/videoN nodes, most of which are
    metadata (UVC) nodes that open but never deliver a frame;
  * a laptop may expose a colour sensor, a monochrome/IR sensor, or both;
  * formats differ (MJPG / YUYV / GREY) and so do the supported sizes.

`probe_devices()` opens every node, grabs a real frame and reports what it
actually is.  Everything else builds on that, so we never blindly trust
`/dev/video0`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_PROBE_INDEX = 12


@dataclass
class DeviceInfo:
    path: str
    index: int
    name: str = ""
    mono: bool = False
    usable: bool = False
    width: int = 0
    height: int = 0
    brightness: float = 0.0
    contrast: float = 0.0
    fourcc: str = ""
    error: str = ""
    is_ir_hint: bool = False
    raw_ok: bool = False

    def as_dict(self) -> dict:
        return {
            "path": self.path, "name": self.name, "mono": self.mono,
            "usable": self.usable, "size": f"{self.width}x{self.height}",
            "brightness": round(self.brightness, 1),
            "contrast": round(self.contrast, 1),
            "fourcc": self.fourcc, "ir_hint": self.is_ir_hint,
            "error": self.error,
        }


def _fourcc_str(v: float) -> str:
    try:
        n = int(v)
        if n <= 0:
            return ""
        return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))
    except Exception:
        return ""


def _analyze(frame: np.ndarray) -> tuple[bool, float, float]:
    """Return (is_monochrome, brightness, contrast)."""
    if frame is None or frame.size == 0:
        return False, 0.0, 0.0
    if frame.ndim == 2:
        mono = True
        gray = frame
    else:
        b, g, r = frame[:, :, 0].astype(np.int16), frame[:, :, 1].astype(np.int16), frame[:, :, 2].astype(np.int16)
        # A true mono sensor (or an IR feed) has near-identical channels.
        chroma = float((np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()) / 3.0)
        mono = chroma < 1.5
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return mono, float(gray.mean()), float(gray.std())


def _open(path: str, cfg_video: dict, budget: float = 2.0):
    """Open a node and return (cap, frame) with a real frame, or (None, None).

    Handles three cases: a real V4L2 node (the normal one), a non-device path
    such as a video file (used by the test fixtures), and devices that refuse
    a particular pixel format - which we retry with others.
    """
    is_device = path.startswith("/dev/")
    fourcc = cfg_video.get("fourcc", "MJPG") if is_device else None
    attempts = []
    for cc in ([fourcc, "MJPG", "YUYV", None] if fourcc else [None]):
        attempts.append(cc)
    seen = set()
    for cc in attempts:
        if cc in seen:
            continue
        seen.add(cc)
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2 if is_device else cv2.CAP_ANY)
        if not cap.isOpened():
            cap.release()
            continue
        if is_device:
            try:
                if cc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg_video.get("width", 640))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg_video.get("height", 480))
                fps = cfg_video.get("fps", 30)
                if fps and fps > 0:
                    cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        deadline = time.time() + budget
        frame = None
        while time.time() < deadline:
            ok, f = cap.read()
            if ok and f is not None and f.size:
                frame = f
                break
            if not is_device:
                # a media file that ended: rewind once instead of spinning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
        if frame is not None:
            return cap, frame
        cap.release()
    return None, None


def probe_devices(cfg_video: dict | None = None, verbose: bool = False) -> list[DeviceInfo]:
    cfg_video = cfg_video or {}
    out: list[DeviceInfo] = []
    for idx in range(MAX_PROBE_INDEX):
        path = f"/dev/video{idx}"
        if not os.path.exists(path):
            continue
        info = DeviceInfo(path=path, index=idx)
        if not os.access(path, os.R_OK):
            info.error = "no read permission (add the user to the 'video' group)"
            out.append(info)
            continue
        cap, frame = _open(path, cfg_video, budget=1.5)
        if cap is None:
            info.error = "opened but delivered no frame (UVC metadata node)"
            out.append(info)
            continue
        try:
            info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info.fourcc = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
            info.name = (cap.getBackendName() or "v4l2")
        except Exception:
            pass
        mono, bright, contrast = _analyze(frame)
        info.mono, info.brightness, info.contrast = mono, bright, contrast
        info.usable = contrast > 2.0
        info.raw_ok = True
        # A monochrome sensor that is very dark is almost certainly an IR
        # sensor with its emitter switched off (Windows Hello style).
        info.is_ir_hint = mono
        cap.release()
        out.append(info)
    return out


def _score_device(d: DeviceInfo, prefer_ir: bool, ir_min_brightness: float) -> float:
    if not d.usable:
        return -1.0
    score = min(d.contrast, 80.0) / 80.0
    score += 0.3 if (d.brightness > 40) else 0.0
    if d.is_ir_hint:
        if d.brightness >= ir_min_brightness:
            score += 1.0 if prefer_ir else 0.25
        else:
            # dark IR sensor - only useful if we explicitly want IR
            score += -0.5 if not prefer_ir else 0.1
    else:
        score += 0.6 if not prefer_ir else 0.0
    return score


def choose_device(cfg_video: dict, devices: list[DeviceInfo] | None = None) -> DeviceInfo | None:
    want = str(cfg_video.get("device", "auto"))
    devices = devices if devices is not None else probe_devices(cfg_video)
    if want and want != "auto":
        for d in devices:
            if d.path == want or str(d.index) == str(want):
                return d
        # explicit request we could not probe: hand it back anyway
        return DeviceInfo(path=want, index=-1, name="explicit", usable=True)
    prefer_ir = bool(cfg_video.get("prefer_ir", False))
    ir_min = float(cfg_video.get("ir_min_brightness", 25.0))
    usable = [d for d in devices if d.usable]
    if not usable:
        return None
    usable.sort(key=lambda d: _score_device(d, prefer_ir, ir_min), reverse=True)
    return usable[0]


class CameraError(RuntimeError):
    pass


class Camera:
    """Context manager yielding frames.  Reopens a few times on flaky devices."""

    def __init__(self, cfg_video: dict, device: DeviceInfo | str | None = None):
        self.cfg = cfg_video or {}
        self.device = device
        self.cap = None
        self.info: DeviceInfo | None = None
        self.mono = False

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        dev = self.device
        if isinstance(dev, DeviceInfo):
            path, mono_hint = dev.path, dev.mono
        elif isinstance(dev, str):
            path, mono_hint = dev, False
        else:
            info = choose_device(self.cfg)
            if info is None:
                raise CameraError("no usable camera found (see `faceid doctor`)")
            path, mono_hint = info.path, info.mono
            self.info = info
        cap, frame = _open(path, self.cfg, budget=2.5)
        if cap is None:
            raise CameraError(f"could not grab a frame from {path}")
        self.cap = cap
        mono, _b, _c = _analyze(frame)
        self.mono = mono or mono_hint
        if self.info is None:
            self.info = DeviceInfo(path=path, index=-1, name="direct", usable=True, mono=self.mono)
        self.info.mono = self.mono
        for _ in range(int(self.cfg.get("warmup_frames", 4))):
            self.cap.grab()

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None or not frame.size:
            return None
        return frame


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return frame
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def normalize(frame: np.ndarray, clahe: bool = True, gamma: float | None = None) -> np.ndarray:
    """Photometric normalisation so a dark room behaves like a bright room."""
    g = to_gray(frame).copy()
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if gamma and gamma > 0:
        inv = 1.0 / float(gamma)
        lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        g = cv2.LUT(g, lut)
    if clahe:
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = cl.apply(g)
    return g


def frame_quality(frame: np.ndarray, cfg_video: dict) -> tuple[bool, str]:
    if frame is None or not frame.size:
        return False, "no frame"
    g = to_gray(frame)
    m, s = float(g.mean()), float(g.std())
    if m < float(cfg_video.get("dark_threshold", 18.0)):
        return False, f"too dark (mean {m:.0f})"
    if m > float(cfg_video.get("bright_threshold", 250.0)):
        return False, f"blown out (mean {m:.0f})"
    if s < 2.0:
        return False, "flat image (lens covered?)"
    return True, f"ok (mean {m:.0f} contrast {s:.0f})"
