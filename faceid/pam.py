"""PAM glue: decide whether to attempt face auth, and as whom.

Called from the `faceid auth` entry point, which is what pam_exec.so runs.
Kept deliberately paranoid:

  * refuse to run for SSH / remote sessions (a camera on the far side is
    useless and a face is not proof of presence at a keyboard);
  * refuse for denied services;
  * refuse when `core.disabled` is set, so there is always a one-line kill
    switch that does not involve touching /etc/pam.d;
  * always leave the password path intact - we are installed as `sufficient`,
    never `required`, so a failure can only ever fall through to the normal
    password prompt.
"""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass, field

from .config import load_config
from .recognizer import Engine, Result


def pam_env() -> dict:
    keys = ("PAM_USER", "PAM_RUSER", "PAM_SERVICE", "PAM_TTY", "PAM_RHOST",
            "PAM_TYPE", "PAM_AUTHTOK_TYPE", "SUDO_USER", "SUDO_UID",
            "DISPLAY", "WAYLAND_DISPLAY", "HOME")
    return {k: os.environ.get(k, "") for k in keys}


def is_remote(env: dict | None = None) -> tuple[bool, str]:
    env = env or pam_env()
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return True, "SSH session"
    rhost = (env.get("PAM_RHOST") or "").strip()
    if rhost and rhost not in ("localhost", "127.0.0.1", "::1"):
        return True, f"remote host {rhost}"
    if env.get("PAM_TTY", "").startswith("/dev/pts/"):
        return True, f"pseudo-terminal {env['PAM_TTY']}"
    return False, "local"


def local_console_present() -> bool:
    """True if somebody looks like they are physically at the machine."""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    try:
        import glob
        return bool(glob.glob("/dev/dri/card*")) and _seat_active()
    except Exception:
        return False


def _seat_active() -> bool:
    try:
        for seat in ("seat0",):
            if os.path.isdir(f"/sys/class/drm/{seat}"):
                return True
        return os.path.isdir("/run/systemd/seats")
    except Exception:
        return True


def resolve_users(cfg, explicit: list[str] | None = None) -> list[str]:
    """Who might be at the camera?

    `sudo` hands us PAM_USER=root, and the human's name only shows up in
    PAM_RUSER / SUDO_USER.  Rather than guessing, we build an ordered
    candidate list and match the face against all of them; the identity that
    actually matches wins.
    """
    if explicit:
        return list(dict.fromkeys(explicit))
    env = pam_env()
    cands: list[str] = []
    for key in ("PAM_USER", "PAM_RUSER", "SUDO_USER"):
        v = (env.get(key) or "").strip()
        if v and v not in cands:
            cands.append(v)
    # PAM_USER for a sudo request is the *target* user (usually root); that
    # is not a face owner, so push it to the back rather than dropping it.
    if cands and cands[0] == "root" and len(cands) > 1:
        cands = cands[1:] + [cands[0]]
    allowed = cfg.sec("pam").get("allowed_users", ["*"])
    users = [u for u in cands if _allowed(u, allowed) and _has_home(u)]
    return users or cands


def _allowed(user: str, allowed: list[str]) -> bool:
    if not allowed or "*" in allowed:
        return True
    return user in allowed


def _has_home(user: str) -> bool:
    try:
        import pwd
        pwd.getpwnam(user)
        return True
    except Exception:
        return False


@dataclass
class Decision:
    attempt: bool
    reason: str
    service: str
    users: list[str] = field(default_factory=list)


def should_attempt(cfg, service: str | None = None) -> Decision:
    env = pam_env()
    svc = (service or env.get("PAM_SERVICE") or "unknown").strip()
    p = cfg.sec("pam")
    core = cfg.sec("core")

    if core.get("disabled", False):
        return Decision(False, "faceid is disabled in the config", svc)
    if svc in set(p.get("denied_services", [])):
        return Decision(False, f"service '{svc}' is denied", svc)
    allowed = set(p.get("allowed_services", []))
    if allowed and svc not in allowed and svc != "unknown":
        return Decision(False, f"service '{svc}' not in pam.allowed_services", svc)
    if core.get("abort_if_ssh", True):
        remote, why = is_remote(env)
        if remote:
            return Decision(False, f"aborting: {why}", svc)
    return Decision(True, "ok", svc, resolve_users(cfg))


def expand_candidates(cfg, want: list[str], enrolled: list[str]) -> tuple[list[str], bool]:
    """Widen the candidate list when nobody in it actually owns a face.

    `sudo` presents PAM_USER=root and frequently sets no PAM_RUSER, so the
    candidate list is just ["root"] and no enrolment will ever match it.
    Falling back to the machine's enrolled users is what makes sudo work at
    all; the face still has to match, so this is safe on a single-user box.
    """
    if not cfg.sec("pam").get("fallback_to_enrolled", True):
        return list(want), False
    if not enrolled:
        return list(want), False
    if any(u in enrolled for u in want):
        return list(want), False
    return list(dict.fromkeys(list(want) + list(enrolled))), True


def authenticate(cfg=None, service: str | None = None, users: list[str] | None = None,
                 seconds: float | None = None, explain: bool = False) -> Result:
    cfg = cfg or load_config()
    dec = should_attempt(cfg, service)
    if not dec.attempt:
        r = Result()
        r.reason = dec.reason
        return r
    engine = Engine(cfg)
    want = users if users is not None else dec.users
    want, fell_back = expand_candidates(cfg, want, engine.gallery.users())
    res = engine.verify(users=want, seconds=seconds, explain=explain)
    if fell_back:
        res.reason = (res.reason + " [resolved via enrolled users: "
                      + ", ".join(want) + "]")
    return res


__all__ = ["pam_env", "is_remote", "resolve_users", "should_attempt",
           "authenticate", "expand_candidates", "Decision", "local_console_present"]
