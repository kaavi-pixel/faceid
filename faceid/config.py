"""Configuration + on-disk path resolution.

Resolution order for the data directory:
  1. $FACEID_DATA_DIR
  2. /var/lib/faceid          (system install, used by PAM - root only)
  3. ~/.local/share/faceid    (per-user / development)

The config file is TOML.  System config lives at /etc/faceid/config.toml and
is overlaid by the user config at ~/.config/faceid/config.toml.
"""

from __future__ import annotations

import copy
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

APP = "faceid"
SYSTEM_DATA_DIR = Path("/var/lib/faceid")
SYSTEM_CONFIG = Path("/etc/faceid/config.toml")
USER_CONFIG = Path("~/.config/faceid/config.toml")

# --------------------------------------------------------------------------
# default configuration (every knob lives here, documented)
# --------------------------------------------------------------------------
DEFAULTS: dict = {
    "core": {
        # Master switch.  Set true to make PAM skip face auth entirely.
        "disabled": False,
        # Never try the camera when the auth request arrives over SSH.
        "abort_if_ssh": True,
        # A positive decision needs this many frames voting for the identity.
        "min_votes": 3,
        # Hard ceiling on how long we are allowed to search for a face.
        "timeout": 5.0,
        # Give up early (seconds) once we already have enough confident votes.
        "early_exit": 2.0,
        # Maximum frames to analyse in one attempt.
        "max_frames": 90,
        "no_confirmation": False,
        "log_level": "warning",
    },
    "video": {
        # "auto" scans /dev/video*, otherwise an index (0) or path.
        "device": "auto",
        # Prefer a monochrome/IR sensor when one is available and usable.
        "prefer_ir": False,
        # Only use IR if the raw frame is at least this bright (0-255).
        "ir_min_brightness": 25.0,
        # Requested capture size.
        "width": 640,
        "height": 480,
        "fps": 30,
        "fourcc": "MJPG",
        # Reject frames this dark (mean luma) - lights off / shutter closed.
        "dark_threshold": 18.0,
        # Reject frames this blown out.
        "bright_threshold": 250.0,
        # Discard this many frames after the device opens.
        "warmup_frames": 4,
        "warn_no_device": True,
    },
    "detect": {
        # CNN face detector (YuNet).  Lower = finds occluded faces, more noise.
        "score_threshold": 0.72,
        # Retry pass with a lower threshold if the first pass finds nothing.
        "fallback_score_threshold": 0.35,
        "nms_threshold": 0.3,
        "top_k": 5000,
        "input_size": 640,
        # Run the detector at several sizes so a close-up or distant face works.
        "multi_scale": True,
        "scales": [1.0, 1.5, 0.65],
    },
    "embed": {
        # Contrast-normalise before embedding - helps a lot in dim rooms.
        "clahe": True,
        # Also embed a gamma-corrected variant and average (low-light robustness).
        "gamma_variants": True,
        "gamma": 1.4,
    },
    "match": {
        # SFace cosine similarity cut-off.  0.363 is the LFW reference point;
        # we ship a little above it and let `faceid calibrate` refine it.
        "cosine_threshold": 0.40,
        # Required gap to the runner-up identity (stops lookalike confusion).
        "margin": 0.03,
        # Bonus when the predicted accessories agree with the matched tag.
        "accessory_bonus": 0.05,
        # Penalty when they disagree (hat on, matched the bare-face gallery).
        "accessory_penalty": 0.02,
        # Frames whose embedding is this close to the accepted centroid count
        # as a vote.
        "vote_threshold": 0.30,
    },
    "accessories": {
        # Detect hats / headphones and use that to pick the right gallery.
        "enabled": True,
        # Gate applied to the aggregated probability (see `aggregate`).
        "min_confidence": 0.50,
        # How to combine per-frame accessory evidence:
        #   p75 (default) - "clear evidence in at least a quarter of frames".
        #     A plain mean collapses when the head turns and the far ear (or
        #     the brim) stops being visible, which mislabels a headphones
        #     wearer as bare and quietly breaks tag routing.
        #   mean / median / max / pNN also accepted.
        "aggregate": "p75",
        # Use the per-user learned classifier; fall back to geometry if absent.
        "use_learned": True,
        # Allow the tag bonus/penalty to influence the decision.
        "influence_match": True,
    },
    "liveness": {
        # Basic anti-spoof: require the face to be a live, moving subject and
        # not a perfectly static printout.  NOT a security guarantee.
        "enabled": True,
        # Minimum landmark movement across the capture window, as a percentage
        # of face height per frame.  A live person sitting still still jitters
        # by roughly 1px on a ~250px face; a frozen frame or a photo taped over
        # the lens sits at ~0.0.  Keep this low or you will lock real users out.
        "min_motion": 0.15,
        # Maximum - a screen/photo held perfectly still then shaken fails.
        "max_motion": 60.0,
        # Reject frames that look like a photo of a screen (moire energy ratio).
        "moire_check": False,
        "moire_threshold": 0.62,
    },
    "pam": {
        # Users allowed to be authenticated by face for sudo.
        # "*" = anyone with an enrolled gallery.
        "allowed_users": ["*"],
        # `sudo` presents PAM_USER=root (the *target* user) and often does not
        # set PAM_RUSER at all, so there is no candidate that owns a face.
        # When that happens, fall back to the enrolled users of this machine
        # rather than failing.  Safe on a single-user laptop (and the face
        # still has to match); set false on a shared machine if you want sudo
        # face auth restricted to a named account.
        "fallback_to_enrolled": True,
        # Services we are allowed to service (defence in depth).
        "allowed_services": ["sudo", "sudo-i", "su", "su-l", "polkit-1",
                             "gdm-password", "sddm", "sddm-greeter",
                             "kde", "kscreenlocker", "login", "system-auth",
                             "fingerprint-auth", "faceid-test"],
        # Do not run for these services even if listed above.
        "denied_services": ["sshd", "vlock", "remote"],
        # Print a cheerful line when face auth succeeds.
        "show_notice": True,
        # Append a one-line outcome per attempt to <data-dir>/pam-debug.log.
        # A `quiet` pam_exec failure is otherwise completely invisible, and
        # "sudo still asks for my password" is impossible to debug without it.
        "debug_log": True,
    },
    "selinux": {
        # Where to look for AVC denials during `faceid doctor`.
        "check": True,
    },
}


# --------------------------------------------------------------------------
@dataclass
class Paths:
    data_dir: Path
    gallery_dir: Path
    model_dir: Path
    log_file: Path
    frame_dir: Path

    @property
    def gallery_file(self) -> Path:
        return self.gallery_dir / "gallery.npz"

    @property
    def gallery_meta(self) -> Path:
        return self.gallery_dir / "gallery.json"


def bundled_model_dir() -> Path:
    """Models shipped inside the source tree (dev checkout)."""
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "models", here / "models"):
        if cand.is_dir() and any(cand.glob("*.onnx")):
            return cand
    return here.parent / "models"


def resolve_data_dir() -> Path:
    """Where the enrolments live.  The user's CLI and the root-run PAM helper
    MUST agree on this, or enrolments made in a terminal are invisible to
    `sudo` - the worst kind of bug, because PAM just silently falls through to
    the password prompt and everything looks like it works.

    So: prefer the shared system store whenever we can actually use it, for
    root and for the enrolled user alike (the installer chowns it to that
    user).  Only fall back to the per-user home store when the system store is
    absent or not usable.
    """
    env = os.environ.get("FACEID_DATA_DIR")
    if env:
        return Path(env).expanduser()
    if SYSTEM_DATA_DIR.is_dir():
        if os.geteuid() == 0:
            return SYSTEM_DATA_DIR
        if os.access(SYSTEM_DATA_DIR, os.R_OK | os.W_OK | os.X_OK):
            return SYSTEM_DATA_DIR
    return _user_data_dir()


def _user_data_dir() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path("~/.local/share").expanduser()
    return base / APP


def resolve_model_dir() -> Path:
    env = os.environ.get("FACEID_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    for cand in (Path("/usr/share/faceid/models"),
                 Path("/usr/local/share/faceid/models")):
        if cand.is_dir() and any(cand.glob("*.onnx")):
            return cand
    return bundled_model_dir()


def paths() -> Paths:
    d = resolve_data_dir()
    return Paths(
        data_dir=d,
        gallery_dir=d / "gallery",
        model_dir=resolve_model_dir(),
        log_file=d / "faceid.log",
        frame_dir=d / "frames",
    )


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """Dict with attribute-free section access: cfg["core"]["timeout"]."""

    def __init__(self, data: dict):
        super().__init__(data)

    def sec(self, name: str) -> dict:
        return self.get(name, {})


def load_config(extra_files: list[Path] | None = None) -> Config:
    data = copy.deepcopy(DEFAULTS)
    files = [SYSTEM_CONFIG, USER_CONFIG.expanduser()]
    env = os.environ.get("FACEID_CONFIG")
    if env:
        files.insert(0, Path(env).expanduser())
    if extra_files:
        files.extend(extra_files)
    for f in files:
        try:
            if Path(f).is_file():
                with open(f, "rb") as fh:
                    data = _deep_merge(data, tomllib.load(fh))
        except Exception as exc:  # never let a bad config break auth
            print(f"{APP}: ignoring bad config {f}: {exc}", file=sys.stderr)
    return Config(data)
