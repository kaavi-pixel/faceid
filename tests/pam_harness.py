"""Probe a real PAM stack from Python via ctypes - no compiler, no pamtester.

Two modes:

  --service sudo        call pam_start("sudo", user, ...) - i.e. the REAL
                        /etc/pam.d/sudo stack.  This is the mode that matters.
  --stack /path/file    point pam_start at a config file directly.

  NOTE on --stack: Linux-PAM path-based service names are NOT honoured on
  Fedora - pam_start silently falls through to /etc/pam.d/other, which on a
  default install is `auth required pam_deny.so`.  That produces a plain
  "Authentication failure" which looks exactly like a broken module and will
  send you chasing the wrong bug.  Use --service unless you have verified
  path support on your platform.

The conversation function refuses to answer any prompt by default, which is
what `sudo -n` does: if pam_authenticate returns success, then face auth alone
satisfied the stack with no password involved.

  sudo python3 tests/pam_harness.py --service sudo --user alice
  sudo python3 tests/pam_harness.py --service sudo --probe

--probe additionally injects a temporary `auth optional pam_exec.so` marker
line into the service file (backed up and restored) so you can tell apart
"pam_exec never ran" from "our helper ran and failed".
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

PAM_SUCCESS = 0
PAM_PROMPT_ECHO_OFF = 1
PAM_PROMPT_ECHO_ON = 2
PAM_ERROR_MSG = 3
PAM_TEXT_INFO = 4
PAM_CONV_ERR = 19

STYLE_NAMES = {
    PAM_PROMPT_ECHO_OFF: "PROMPT_ECHO_OFF",
    PAM_PROMPT_ECHO_ON: "PROMPT_ECHO_ON",
    PAM_ERROR_MSG: "ERROR_MSG",
    PAM_TEXT_INFO: "TEXT_INFO",
}

MARKER = Path("/var/lib/faceid/pam-probe.txt")


def load_libpam():
    for name in ("libpam.so.0", "libpam.so", "pam"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    found = ctypes.util.find_library("pam")
    if found:
        return ctypes.CDLL(found)
    raise RuntimeError("libpam not found - install the `pam` package")


class PamMessage(ctypes.Structure):
    _fields_ = [("msg_style", ctypes.c_int), ("msg", ctypes.c_char_p)]


class PamResponse(ctypes.Structure):
    _fields_ = [("resp", ctypes.c_char_p), ("resp_retcode", ctypes.c_int)]


CONV_FUNC = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(PamMessage)),
    ctypes.POINTER(ctypes.POINTER(PamResponse)),
    ctypes.c_void_p,
)


class PamConv(ctypes.Structure):
    _fields_ = [("conv", CONV_FUNC), ("appdata_ptr", ctypes.c_void_p)]


class Harness:
    def __init__(self, service: str, user: str, answer: str | None = None):
        self.lib = load_libpam()
        self.service = service
        self.user = user
        self.answer = answer
        self.messages: list[str] = []
        self._keep_alive: list = []

        self.lib.pam_start.restype = ctypes.c_int
        self.lib.pam_start.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                       ctypes.POINTER(PamConv),
                                       ctypes.POINTER(ctypes.c_void_p)]
        self.lib.pam_authenticate.restype = ctypes.c_int
        self.lib.pam_authenticate.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.pam_end.restype = ctypes.c_int
        self.lib.pam_end.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.pam_strerror.restype = ctypes.c_char_p
        self.lib.pam_strerror.argtypes = [ctypes.c_void_p, ctypes.c_int]

        @CONV_FUNC
        def _conv(n, msgs, resp, appdata):
            arr = msgs[0]
            for i in range(n):
                style = arr[i].msg_style
                text = (arr[i].msg or b"").decode(errors="replace")
                self.messages.append(f"{STYLE_NAMES.get(style, style)}: {text}")
                if style in (PAM_PROMPT_ECHO_OFF, PAM_PROMPT_ECHO_ON):
                    if self.answer is None:
                        return PAM_CONV_ERR     # never supply a password
                    out = ctypes.cast(resp, ctypes.POINTER(ctypes.POINTER(PamResponse)))
                    buf = (PamResponse * n)()
                    buf[i].resp = self.answer.encode()
                    buf[i].resp_retcode = 0
                    out[0] = ctypes.cast(buf, ctypes.POINTER(PamResponse))
                    self._keep_alive.append(buf)
            return PAM_SUCCESS

        self._keep_alive.append(_conv)
        self.conv = PamConv(_conv, None)

    def authenticate(self) -> tuple[int, str]:
        pamh = ctypes.c_void_p()
        rc = self.lib.pam_start(self.service.encode(), self.user.encode(),
                                ctypes.byref(self.conv), ctypes.byref(pamh))
        if rc != PAM_SUCCESS:
            return rc, f"pam_start failed for service {self.service!r}"
        try:
            rc = self.lib.pam_authenticate(pamh, 0)
            err = self.lib.pam_strerror(pamh, rc)
            return rc, (err.decode(errors="replace") if err else "")
        finally:
            self.lib.pam_end(pamh, rc)


PROBE_LINE = ('auth       optional     pam_exec.so /bin/sh -c '
              '"id -un >> ' + str(MARKER) + '; echo probe-ran >> ' + str(MARKER) + '"')


def inject_probe(service_path: Path) -> Path | None:
    """Temporarily add an `auth optional pam_exec.so` marker line.  Root only."""
    if os.geteuid() != 0:
        return None
    if not service_path.is_file():
        return None
    backup = service_path.with_suffix(service_path.suffix + ".harness-bak")
    shutil.copy2(service_path, backup)
    lines = service_path.read_text().splitlines()
    out, inserted = [], False
    for line in lines:
        if not inserted and line.strip().startswith("auth"):
            out.append(PROBE_LINE)
            inserted = True
        out.append(line)
    if not inserted:
        out.append(PROBE_LINE)
    service_path.write_text("\n".join(out).rstrip() + "\n")
    return backup


def restore_probe(service_path: Path, backup: Path | None) -> None:
    if backup and backup.is_file():
        shutil.copy2(backup, service_path)
        backup.unlink()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Probe a real PAM stack through libpam via ctypes")
    ap.add_argument("--service", default="sudo",
                    help="PAM service name (default: sudo) - the real stack")
    ap.add_argument("--stack", default=None,
                    help="path to a PAM config (NOT honoured on Fedora; see docstring)")
    ap.add_argument("--user", default=None,
                    help="PAM_USER to present (default: $SUDO_USER or $USER)")
    ap.add_argument("--helper", default="/usr/local/bin/faceid-pam")
    ap.add_argument("--log", default="/var/lib/faceid/pam-debug.log")
    ap.add_argument("--answer", default=None,
                    help="password to supply; default is to refuse (like sudo -n)")
    ap.add_argument("--probe", action="store_true",
                    help="inject a temporary pam_exec.so marker line (root only)")
    a = ap.parse_args(argv)

    service = a.stack or a.service
    user = a.user or os.environ.get("SUDO_USER") or os.environ.get("USER", "nobody")

    log_path = Path(a.log)
    log_before = log_path.stat().st_size if log_path.is_file() else None
    probe_before = MARKER.read_text() if MARKER.is_file() else None

    service_path = Path("/etc/pam.d") / a.service
    backup = None
    if a.probe:
        if os.geteuid() != 0:
            print("--probe needs root (it edits the service file and restores it)\n",
                  file=sys.stderr)
        else:
            backup = inject_probe(service_path)

    print(f"pam service    : {service}")
    print(f"PAM_USER       : {user}")
    print(f"helper         : {a.helper}")
    print(f"euid           : {os.geteuid()}")
    print(f"debug log      : {log_before if log_before is not None else 'absent'}")
    print("-" * 64)
    try:
        h = Harness(service, user, answer=a.answer)
        rc, err = h.authenticate()
        for m in h.messages:
            print(f"  PAM says: {m}")
    finally:
        if backup is not None:
            restore_probe(service_path, backup)
    print("-" * 64)
    print(f"pam_authenticate -> {rc} ({err})")
    print("RESULT:", "AUTHENTICATED - no password involved"
          if rc == PAM_SUCCESS else "REJECTED (PAM fell through to a password)")

    log_after = log_path.stat().st_size if log_path.is_file() else None
    print(f"debug log after: {log_after if log_after is not None else 'absent'}")
    if log_after is not None and log_after != log_before:
        print("--- helper log (new content) ---")
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-20:]))
    elif log_after is None:
        print("!! the helper never ran (no log written at all)")

    if a.probe:
        now = MARKER.read_text() if MARKER.is_file() else None
        print("-" * 64)
        if now and now != probe_before:
            print("pam_exec probe: RAN - this PAM stack does execute pam_exec commands")
            print("  " + " / ".join(sorted(set(now.split()))))
        else:
            print("pam_exec probe: DID NOT RUN - pam_exec.so executed nothing, so a")
            print("  failing pam_exec line says nothing about our helper at all.")
    return 0 if rc == PAM_SUCCESS else 1


if __name__ == "__main__":
    sys.exit(main())
